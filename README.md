# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-14 | 今日论文总数: 558

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. @skills: Attention is all you have

**arXiv ID:** 2608.12610 | [PDF](https://arxiv.org/pdf/2608.12610v1)

**作者:** Li Yin `[一作]` (SylphAI), Wang `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为@skills的协议，分离技能的内容、持久化与自动触发三大功能，降低技能在模型提示中的占用空间，提升技能可访问性与管理效率。

**💡 创新点**

核心创新是通过文件路径引用机制（@skills:hub/…, @skills:gh:…）将技能的内容、存储与触发方式解耦，只在需要时将前言加载到提示，彻底消除传统安装模式的“全局占位”痛点。

**🔧 技术方法**

技术实现主要基于文件系统层面的路径引用、GitHub/自建Hub的增量缓存、单行触发配置文件（.autotrigger）、以及一个轻量级CLI/agent指令文件。

**📊 数据集**

使用公开 GitHub 代码库的 56,804 个技能作为实测数据集，对其文件结构、描述长度和使用频率等指标进行统计与分析。

**📈 对比分析**

与传统“一次安装即永久驻留”方式相比，@skills 通过将技能描述的驻留成本降至单行文本，显著减少模型可用的自触发槽位使用率（从数百个槽位降至十几），实现了更高的技能可用率和更低的注意力消耗。

**⚠️ 局限性**

局限性包括：仍需依赖模型对提示的注意力预算、对触发可靠性的经验性假设、以及潜在的远程文本注入安全风险，且未对不同模型大小的适用性做全面评估。

---

## 2. ASAP: Reimagining the Data Lifecycle using Application Semantic-Aware Processing

**arXiv ID:** 2608.12735 | [PDF](https://arxiv.org/pdf/2608.12735v1)

**作者:** Milind Srivastava `[一作]` (Carnegie Mellon University), Vyas Sekar `[通讯]` (Carnegie Mellon University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了“信息中心化处理”(ASAP)范式，统一语义感知的压缩、草图、波形、滚动等技术，在数据流水线不同阶段和领域协同优化。

**💡 创新点**

创新点在于将语义保持变换抽象为首席设计原则，跨域、跨阶段统一使用，促进共享实现、配置与互操作性，并能组合多种语义保证。

**🔧 技术方法**

使用了抽样、Sketch、波形变换、压缩、图形摘要等信息摘要技术，并在收集、传输、存储和查询各阶段嵌入。

**📊 数据集**

实验数据集包括10台主机生成的100万条时序数据、Uber公开的530K轨迹数据集以及CAIDA网络包/流数据。

**📈 对比分析**

与Prometheus、ClickHouse、原始基准对比，查询性能提升28–3000倍、传输/分析成本降低48–81倍、内存/存储成本减少至7000倍、压缩率140倍，整体收益可达三阶量级。

**⚠️ 局限性**

局限在于仅验证聚合查询、手工部署配置、缺乏自动化选择与动态重构、复杂查询支持不足、跨阶段互操作标准未成熟。

---

## 3. Why AI Governance Frameworks Are Hard to Adopt: A Role-Based Stress Test of the NIST AI RMF

**arXiv ID:** 2608.12352 | [PDF](https://arxiv.org/pdf/2608.12352v1)

**作者:** Joseph R. Simons `[一作]` (George Washington University), David A. Broniatowski `[通讯]` (George Washington University)

**通讯引用:** 5397 | [OpenAlex ID](https://openalex.org/A5073231714)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在消费者贷款场景下，利用 GPT‑5.5 模拟四类组织角色（企业高管、业务高管、中层经理、系统所有者）在两种 AI 部署（受限机器学习模型与 LLM 辅助模型）和三种治理难题下，进行 4×2×3 设计的角色模拟实验，共生成 120 条回应并进行结构化评分。

**💡 创新点**

提出“治理可转化性（Governance Translatability）”和“结构契合度（Structural Fit）”两大诊断维度，用以解释框架在本地可操作性与真正产生治理价值（跨层级、权威连接、风险降低）之间的断层；并首次将 NIST AI RMF 的采用问题与组织层级与系统边界的不匹配联系起来。

**🔧 技术方法**

核心技术为大语言模型（GPT‑5.5）角色模拟与 LLM‑评分器；评分标准基于 0–2 量表，结合统计检验（卡方、精确检验）对治理价值与风险降低进行定量评估。

**📊 数据集**

数据集为实验生成的 120 条文本回应，涵盖 4 个角色、2 种 AI 系统、3 个治理难点；不使用外部真实组织数据或公开数据集，而是依赖结构化情景提示。

**📈 对比分析**

比较方法：按角色与部署对治理价值（Step 1）与风险降低（Step 2）的成功率进行二元表分析，利用卡方检验评估关联强度。结果显示：本地翻译高（100%），治理价值显著受角色影响（企业/业务高管最高），风险降低显著受部署与结构契合度共同作用；整体治理价值/风险降低率分别为约 45% 与 12%。

**⚠️ 局限性**

局限性：① 采用 LLM 角色模拟，缺乏真实组织行为验证；② 研究聚焦单一行业（消费者贷款）和两种 AI 部署；③ 评分工具为诊断性、未经正式验证；④ 未考虑组织治理成熟度与多评测者差异。

---

## 4. MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval

**arXiv ID:** 2608.12532 | [PDF](https://arxiv.org/pdf/2608.12532v1)

**作者:** Aaryan Sharma `[一作]` (Indian Institute of Technology Bombay), Ganesh Ramakrishnan `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MASCOT，一种基于子模覆盖的后置重排序器，用于文本到图像检索，在多属性多维多任务（多样性增/减）下实现结果多样性控制，避免了传统DPP所带来的早期召回衰退。

**💡 创新点**

创新点在于把多属性多维多样性约束从连续的几何排斥模型迁移为动态软分箱与查询驱动的概率覆盖，既能精确控制属性分布，又能在属性降维度时保持语义相关性。

**🔧 技术方法**

使用了软概率分箱、查询驱动的 IU 重要性权重、局部归一化的语义相关度、基于贪心的子模优化，以及与 MS‑DPP、k‑DPP、MMR 等对比基线。

**📊 数据集**

在 PixelProse、Visual Genome、Incidents1M 和 SkyScript 等多模态数据集上评估，尤其针对地理位置、时间戳及其组合属性的多样性控制。

**📈 对比分析**

与 BLIP‑2、k‑DPP、MMR、MS‑DPP 及其变种对比，MASCOT 在多属性下降任务中实现 R@10≥0.94、R@1≥0.72，显著优于 MS‑DPP（R@10≈0.49、R@1≈0.23），在多属性提升任务也保持竞争力。

**⚠️ 局限性**

局限性包括需要高质量的离散元数据和足够大的候选集；在单属性或小候选集场景下均匀分箱或 MS‑DPP 可能更佳；软分箱导致的额外计算开销与参数调优仍需进一步研究。

---

## 5. LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition

**arXiv ID:** 2608.12328 | [PDF](https://arxiv.org/pdf/2608.12328v1)

**作者:** Iman Khazrak `[一作]` (Bowling Green State University), Robert C. Green `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LoRA‑Diffusion，一种针对扩散语言模型的参数高效微调方法，利用低秩分解对去噪轨迹进行学习而非模型权重；

**💡 创新点**

创新点在于将低秩适配器迁移到扩散迭代轨迹空间，结合步长自适应秩分配与多任务模块叠加，实现零样本任务组合；

**🔧 技术方法**

技术手段包括低秩轨迹适配器（FiLM条件化）、步长自适应秩分配、正交化与核范数正则化、共享指令编码器与路由器；

**📊 数据集**

在GLUE基准任务SST‑2、QNLI和MRPC上进行评估；

**📈 对比分析**

与全微调、权重LoRA、适配器、BitFit等传统PEFT方法对比，LoRA‑Diffusion在SST‑2上取得最高的token‑level denoising验证准确率（88.01%），在QNLI和MRPC上亦保持接近全微调的性能，同时仅训练28.7%参数（轨迹适配器仅1.2%），显著降低存储需求和灾难性遗忘；

**⚠️ 局限性**

局限性包括：训练中仍需较大比例（27.5%）的共享指令编码器；未在严格匹配参数预算的条件下对比；未验证更大模型和更复杂任务（QA、摘要）；步长自适应秩分配为经验设定；以及对多任务组合的路由与任务算术等更高级的组合方法仍未实现。

---

## 6. Not All Nudges Land: Behavioral Controllability and Elaboration Quality in AI-Supported Journaling

**arXiv ID:** 2608.12582 | [PDF](https://arxiv.org/pdf/2608.12582v1)

**作者:** Nadia Mehjabin `[一作]` (University of Virginia), Subigya Nepal `[通讯]` (University of Virginia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对MindScape系统的8周随访数据进行二次分析，探究AI生成提示与用户日记对行为改变的即时效果。

**💡 创新点**

发现社会依赖性行为对提示响应最弱，而个体可控行为的响应差异显著，并证明意向性文字与其丰富程度与短期行为改变相关。

**🔧 技术方法**

使用大语言模型（LLM）构建两阶段分类管线（提示类型与意向性判定），并结合LLM语义评分、句法与表面特征提取。

**📊 数据集**

基于MindScape实验的20名本科生的369条日记与26个被动感知特征（如步数、通话时长、短信数等）数据集。

**📈 对比分析**

采用3天前后感知值的二进制改进指标；在社会依赖性低的行为（如短信、步行）中，意向性文本提升了约100%或50%以上的改进率，说明提示对某些行为有效。

**⚠️ 局限性**

样本量小、单一机构、LLM分类准确率约70%，3天窗口可能忽略时序细节，且因果关系难以确认。

---

## 7. FluctlightDB: A Memory Model of Data for AI Agents

**arXiv ID:** 2608.12365 | [PDF](https://arxiv.org/pdf/2608.12365v1)

**作者:** Ganesh S `[一作]` `[通讯]`, Ganesh S

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 agent memory 的第三种数据模型，并实现了嵌入式引擎 FluctlightDB，支持 episode（engram） 的写入分离、编码、合并、出处记录，以及基于 cue 的图传播激活读取。

**💡 创新点**

创新点在于将记忆建模为 episode 并为其定义专属写/读语义，区别于传统关系表和向量检索；引入离线 replay/压缩、出处加权激活，并以 SQLite 级别的嵌入式存储实现。

**🔧 技术方法**

使用 Rust 开发、WAL 持久化、FTS5 + HNSW 侧车混合检索、传播激活图、Recall Fabric 重排器、离线 replay/压缩、MiniLM、mpnet 等语义向量编码技术。

**📊 数据集**

使用数据集：LoCoMo（1,982 证据 span）、LongMemEval‑S（500 题）、BEIR SciFact（MiniLM embedding）、FAMB（自制回归测试）以及多重单元测试。

**📈 对比分析**

在与现有 memory layers（Mem0、Zep、HippoRAG）和向量 DB（Chroma）相同 harness 下对标；性能表现为 LoCoMo evidence recall 99.0%，LongMemEval session@8 97.6%，LongMemEval E2E 97.4%，BEIR nDCG@10 0.646 对比 Chroma 0.645；但共享 brain 下 provenance 仅 18%。

**⚠️ 局限性**

局限性：多租户共享 brain 时 provenance 失效；k=5 以下召回率低；端到端 LoCoMo 仍低 F1（23.5%）；未评估超过 10⁵ 记忆的 ANN；缺乏多租户隔离及跨案例干扰处理。

---

## 8. Steering the Language Axis: From Linear Decodability to Causal Control

**arXiv ID:** 2608.12334 | [PDF](https://arxiv.org/pdf/2608.12334v1)

**作者:** Arnav Srivastav `[一作]` `[通讯]` (University of California, Santa Cruz), Arnav Srivastav (University of California, Santa Cruz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言解码器进行因果干预，利用 PCA 提取的语言轴控制输出语言。

**💡 创新点**

首次证明语言身份在多语言模型中是可因果操控、层级特定的几何方向，而非仅可线性解码。

**🔧 技术方法**

采用 PCA、激活向量干预（steering、ablation）以及随机控制验证。

**📊 数据集**

使用 FLORES‑200 与 MultiUN 数据集进行校准与评估。

**📈 对比分析**

在 Qwen 3.5‑2B 与 Llama‑3.2‑1B‑Instruct 上对比，成功将目标语言比例提升至近 100%，随机干预无效；层级分析揭示层 12 为瓶颈，后层易操控。

**⚠️ 局限性**

仅考虑单维 PCA 轴，未扩展至多维或 MoE 架构，且高层干预易导致重复退化，实验范围受限于少数语言对。

---

## 9. DiG-bench: Discovery in Games

**arXiv ID:** 2608.12593 | [PDF](https://arxiv.org/pdf/2608.12593v1)

**作者:** Ruairidh M. Battleday `[一作]` (Thinking About Thinking), James C. R. Whittington `[通讯]` (Thinking About Thinking)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个名为 DiG-bench 的文本游戏基准，包含 70 个未知机制的交互式小游戏，用于评估 LLM 的主动探索与发现能力。

**💡 创新点**

创新点在于：①专门针对“发现”能力设计，剔除视觉与空间先验；②将游戏限制为短文本，保证在单个上下文窗口内完成；③设置多难度层级并公开 21 个游戏以供社区使用。

**🔧 技术方法**

采用 LLM 代理（如 Gemini 3.1 Pro、Opus 5 等）通过 API 与游戏交互；对比基本 harness 与 agentic harness（如 Prime Agent、Fable 5 等）两种评测方式。

**📊 数据集**

数据集为 DiG-bench 本身——70 个手工设计的游戏，其中 21 个公开，涵盖 7 个难度层级；此外实验中还用到游戏规则的自然语言描述作为对照。

**📈 对比分析**

比较方法：对同一游戏分别跑基本 harness 与 agentic harness，统计胜率、步骤数；对 Gemini 3.1 Pro 给予规则描述后再跑；人类玩家进行基准验证。结果显示：最高单模型 Opus 5 在基本 harness 下 50/70 场赢；所有模型在 agentic harness 下并未显著提升；给 Gemini 提供规则后胜率从 18/70 跃升至 69/70，说明发现规则是主要难点。

**⚠️ 局限性**

局限性：①游戏仅限文本，缺乏多模态挑战；②未覆盖长期记忆与多步骤规划的需求；③实验多为单种子，缺乏对随机性和可复现性的评估；④部分模型在高难度层级仍无法突破，表明当前 LLM 对主动实验的深度仍有限。

---

## 10. EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory

**arXiv ID:** 2608.12627 | [PDF](https://arxiv.org/pdf/2608.12627v1)

**作者:** Le Zhang `[一作]` (University of Michigan), Ke Sun `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种长时程自我中心记忆问答框架，通过本地多模态上下文生成自包含的原子记忆索引，并结合多视图索引与时序感知检索。

**💡 创新点**

创新点在于用局部上下文增强索引以解决片段式字幕缺失上下文问题，并引入双代理时间感知检索，使检索同时兼顾语义相似度与时间意图。

**🔧 技术方法**

使用多模态LLM进行视频与音频转写、核心ference与省略消解，结合多视图索引结构、句子嵌入、时间衰减评分与双代理检索。

**📊 数据集**

在EgoLifeQA、EgoMem和EgoR1-Bench三大自我中心生命周期问答基准上进行评估。

**📈 对比分析**

与长上下文LLM代理和现有代理记忆基线相比，准确率提升至少4.4–14.2%，检索命中率提升至49.6%–89.6%，成本降低36倍。

**⚠️ 局限性**

局限在于对感知噪声、开放域身份解析和更长录音的鲁棒性尚未充分验证。

---

## 11. Unified Multi-Dimensional Benchmark for Complex Graph Reasoning in Large Language Models

**arXiv ID:** 2608.12391 | [PDF](https://arxiv.org/pdf/2608.12391v1)

**作者:** Fali Wang `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于大型语言模型的半自动化框架，用于自动生成复杂的图推理基准，并基于此构建了一个包含202个任务的评测集。

**💡 创新点**

创新点在于：①通过覆盖图规模、任务复杂度、任务描述、图加载方式与任务来源五个维度，系统化地扩展了图推理的难度；②同时支持文本与代码两种推理模式，提供统一评估；③利用LLM自动生成任务、图数据、解题脚本和评估脚本，大幅降低人工标注成本。

**🔧 技术方法**

采用大型语言模型（如Qwen、Llama、Gemma等）进行任务合成、图实例生成、解题代码、问题描述与评估脚本的自动化生成，并通过有限的人工验证与LLM质量评估器进行质量控制。

**📊 数据集**

使用了新构建的Benchmark（未命名），包含202个任务，任务涵盖10、100、1k、10k节点规模的图，任务深度从单一到四层组合，来源覆盖经典图算法与LeetCode等在线评测平台。

**📈 对比分析**

在文本推理、代码推理和增强推理三种模式下，对多款LLM进行零-shot链式思考（CoT）评测；结果显示：图规模、组合深度、隐式描述等维度对性能影响显著；文件式图加载显著优于inline；OA任务比经典任务更难；检索增强在文本模式提升显著，但对代码模式反而不利；指令微调泛化能力有限；代码模式下的LLM在性能上优于专用GNN。

**⚠️ 局限性**

局限性包括：框架对生成质量高度依赖强大的LLM，常规本地LLM可能不足；Benchmark 主要聚焦评估而非微调；生成流程仍需更完善，尤其是对更大规模图的高效生成与验证。

---

## 12. The "Knowledge-Behavior Gap" in Cultural Taboo Safety of Large Language Models

**arXiv ID:** 2608.12341 | [PDF](https://arxiv.org/pdf/2608.12341v1)

**作者:** Ying He `[一作]`, Yanghua Xiao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文只展示了如何使用ACL样式文件，并给出了一些多语言文本示例；并未进行任何实际研究工作。

**💡 创新点**

无创新点，内容仅为示例模板。

**🔧 技术方法**

使用LuaLaTeX或XeLaTeX编译器。

**📊 数据集**

未使用任何数据集。

**📈 对比分析**

无实验或方法比较，亦未报告任何性能指标。

**⚠️ 局限性**

限制在于本文不包含真实研究，仅为格式演示，无法评估研究效果。

---

## 13. Assessment Design in the GenAI Era: The X1-X2-X3 Assessment Pattern for Testing Students' AI Literacy, Learning Outcomes, and Reflection

**arXiv ID:** 2608.12351 | [PDF](https://arxiv.org/pdf/2608.12351v1)

**作者:** Riasat Islam `[一作]` (Queen Mary University of London), Thomas Roelleke `[通讯]` (Queen Mary University of London)

**通讯引用:** 495 | [OpenAlex ID](https://openalex.org/A5008157952)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在英国一所大学的数据库系统本科模块中，迭代设计并实施了一种允许并考核学生使用生成式人工智能（GenAI）的三段式答题结构 X1-X2-X3，结合 AI‑aware 题目撰写流程，以在在线考试中保持评估效度。

**💡 创新点**

创新点在于将 AI 题目压力测试与三段答题结构相结合，既让学生在答题时公开记录 AI 产出、修正与评价，又通过情境化、模块特定的题目设计抑制表面化的 AI 完成，提升评估可见性与真实性。

**🔧 技术方法**

技术手段包括使用主流 GenAI 工具（ChatGPT、Gemini、Claude、DeepSeek）进行题目压力测试、生成示例答案；采用结构化评估模板、截图证明和统一打分 rubric，以确保可追溯性与批改一致性。

**📊 数据集**

数据来源涵盖：实践考试的 25 名学生的 22 条完整 X1‑X2‑X3 试答、两期课程作业 251 条评分记录、模块最终成绩 263 条完整记录，以及存档的 GenAI 试答日志和评估材料。

**📈 对比分析**

通过对四个迭代（作业实践、评估作业、练习考试、正式考试）的对比，评估了完成度、评分分布与学生表现；结果显示未出现明显的高分压缩，说明设计在不同难度层次上能有效区分学生水平，虽未进行随机对照实验或学习成效测量，但在维持真实性与公正性的前提下保持了成绩分布。

**⚠️ 局限性**

局限性包括：仅在单一模块与单一学校实施，缺乏实验对照与效能验证；作者参与度高，可能导致主观偏差；缺少详细的学生背景或多元文化分析；未对评卷一致性做统计检验；随着 AI 技术快速演进，题目与流程需定期重新测试和更新。

---

## 14. The Boolean Power of ReLU

**arXiv ID:** 2608.12617 | [PDF](https://arxiv.org/pdf/2608.12617v1)

**作者:** Pablo Barceló `[一作]` (Pontifical Catholic University), Jan Van den Bussche `[通讯]` (Universiteit Hasselt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明在仅含单一布尔特征的有限无向图上，使用 ReLU 的图神经网络在布尔查询表达上严格优于仅使用 Truncated ReLU 与恒等激活函数的网络。

**💡 创新点**

首次解决了 ReLU 相对于 TrReLU 在布尔查询表达力上的严格包含关系，并给出严格的分离示例与证明。

**🔧 技术方法**

构造了特定的图族与参数化结构，利用分层归纳与逐层激活函数的渐进常数性质，证明了任何使用逐渐恒定激活函数的表达式在此图族上是渐近局部的，从而与 ReLU 的分离。

**📊 数据集**

无实验数据集，论文为纯理论推导。

**📈 对比分析**

通过构造对比示例证明表达力的严格包含关系，无性能数值对比。

**⚠️ 局限性**

仅针对单一布尔特征的无向图，未讨论多特征、多激活函数或有向图等更一般情形，证明的适用范围受限于逐渐恒定激活函数类。

---

## 15. Predicting consumer-technology ownership without a diffusion history

**arXiv ID:** 2608.12344 | [PDF](https://arxiv.org/pdf/2608.12344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 16. MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

**arXiv ID:** 2608.12435 | [PDF](https://arxiv.org/pdf/2608.12435v1)

**作者:** Ming Zhang `[一作]` (Shanghai AI Laboratory), Youbang Sun `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入 MARCH 框架，在循环模型中周期性快照状态为锚点，并通过内容路由检索历史记忆，保持因果性且不改写原始递归。

**💡 创新点**

创新点是将连续循环状态切分为可检索的状态锚点，并用学习的键进行内容路由，既保留了线性注意力的高效写入，又扩展了可访问的记忆容量。

**🔧 技术方法**

采用 Gated DeltaNet/线性注意力为底层，周期性状态快照，软max内容路由，Top‑K 稀疏检索以及 FlashAttention‑2 风格的融合调度。

**📊 数据集**

在 50B token 的 Long‑Data‑Collections 预训练后，评测涵盖 CommonsenseQA、LongBench、NIAH、RULER 等多种零样本、长上下文和检索任务。

**📈 对比分析**

与 GDN、Log‑Linear GDN 及 Transformer 同层/同参模型对比，MARCH 在所有检索与理解基准上均优于基线，尤其在 32K 长上下文的 NIAH 任务中保持完美准确率。

**⚠️ 局限性**

限制在于固定周期锚点未能自适应记忆变化；只扩展了可检索状态数量而未提升底层写入/读取容量；缺乏多分区或外部记忆的结合，未来需改进自适应锚定与容量分配。

---

## 17. Interaction Readiness: A Framework for Building and Evaluating AI Agents in Human Roles

**arXiv ID:** 2608.12358 | [PDF](https://arxiv.org/pdf/2608.12358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 18. From Fair Representation to Just Recognition in Generative AI

**arXiv ID:** 2608.12669 | [PDF](https://arxiv.org/pdf/2608.12669v1)

**作者:** Severin Engelmann `[一作]` (Cornell University), Daniel Susser `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了生成式人工智能对社会群体描绘与认知的影响，并批判传统的“表征公平”框架，提出转向“认知正义”与“参与平等”的概念；通过政治哲学（尤其是弗雷泽的两维正义理论）对现有公平与价值对齐方法进行理论重构，指出准确性导向的技术手段在实现社会正义方面的局限；进一步阐述参与式治理与多元数据构建在提升认知公平中的潜在价值与挑战。

**💡 创新点**

创新点在于：① 将生成式AI的公平问题从表征准确性重新定位为“认知正义”，强调社会参与平等而非单纯的表征准确性；② 将弗雷泽的“参与平等”理论引入AI伦理，提供更具政治性与社会性的正义评估框架；③ 系统批判现有基于准确性或多样性目标的技术方法，揭示其在强化社会不平等、集中解释权与固化群体内部差异等方面的深层弊端。

**🔧 技术方法**

本文并未实现任何实验性技术，而是运用概念分析与跨学科文献综述的方式：参考公平AI/ML的分布式公平工具、价值对齐的奖励模型与分布优化技术；讨论了RLHF、数据集构建与分布式偏好优化等技术在实现认知正义时的不足。

**📊 数据集**

论文未使用具体数据集；仅引用了公开调查（如World Values Survey、Value Kaleidoscope）与案例数据来说明问题。

**📈 对比分析**

由于本文为理论与概念分析性质，未进行实验比较或性能评估；通过案例讨论与理论推演说明准确性优化与参与平等的关系。

**⚠️ 局限性**

局限性包括：① 依赖抽象概念与政治理论，缺乏可量化的评估指标；② 讨论停留在理论层面，缺少实证验证与实验设计；③ 认知正义的实现需要多方公开争论与协商，难以在算法层面单独解决；④ 对多样化社会群体内部差异与动态演变的处理仍不充分。

---

## 19. Governed Persistent Memory: Source-Bound State Semantics and Fail-Closed Release for Long-Horizon Agents

**arXiv ID:** 2608.12476 | [PDF](https://arxiv.org/pdf/2608.12476v1)

**作者:** Guodong Xu `[一作]` `[通讯]` (Guodongxiansheng Network Technology), Guodong Xu (Guodongxiansheng Network Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了 Governed Persistent Memory（GPM）模型，定义可审计的双时态状态转换，确保从源绑定到结构化发布的完整性与安全边界。

**💡 创新点**

引入了五条可执行合约条款（账本完整性、源绑定、冲突隔离、非复活、结构化发布闭包），并通过合约测试与实战服务验证其完整性与可执行性。

**🔧 技术方法**

采用事件日志链与哈希链、双时态记录、源证据绑定、可执行合约检查，以及三种实现（单文件、增量、分段）并结合正式模型和差分测试。

**📊 数据集**

主要使用内部设计的 GPM‑ReleaseBench（3600 条合约测试案例）、Governed‑QA Sealed（2400 条聚类测试）、以及公开的 LongMemEval、MemoryAgentBench 等清洁任务集。

**📈 对比分析**

与三种简单完整策略和无治理的 Qwen2.5‑7B 对比；完整系统在合约测试上 100% 通过，治理服务实现 2400/2400 正确率；性能上单文件、增量、分段引擎通过差分测试，吞吐量约 44k 事件/秒。

**⚠️ 局限性**

评测数据为合成且内部生成，缺乏开放式对话或真实用户数据；仅保障源绑定与时间边界，无法处理语义一致性、子代修复和大规模分布式一致性问题；未对模型生成答案的语义准确性做评估。

---

## 20. Jagged Judges: Epistemic Stability Under Silence, Pressure, and Persistence

**arXiv ID:** 2608.12645 | [PDF](https://arxiv.org/pdf/2608.12645v1)

**作者:** Justin Zhao `[一作]` (Meta Superintelligence Labs), Khalid El-Arini `[通讯]` (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Wiggle Framework，用多级压力测试评估 LLM 判定者的稳定性。

**💡 创新点**

创新点在于将机械一致性、单轮决心与多轮坚持三维度统一起来，并发现压力往往导致误判而非纠正。

**🔧 技术方法**

技术方法包括 L1–L6 等级压力（语义不变重复、提示扰动、专家权威、合成共识、自适应说服）以及 wiggle rate、保留率和相关性分析。

**📊 数据集**

使用六个任务集（WildGuard、AEGIS、HH-RLHF、ToxiGen、MAGE、Paired Prompts）在二进制与 Likert 评分下共 14 项评测。

**📈 对比分析**

通过在 9 种前沿模型上计算 wiggle rate、保留率并与陪审团一致性相关，发现大多数模型在压力下错误率升高，只有少数模型在特定压力下表现相对稳定。

**⚠️ 局限性**

局限性包括仅聚焦于边界项、缺乏人类基准、说服者集合有限，以及对未过滤数据分布的敏感性。

---

## 21. The Impact of Temporal Context Length and Encoding Strategies on Self-Supervised ECG Representation Learning

**arXiv ID:** 2608.12695 | [PDF](https://arxiv.org/pdf/2608.12695v1)

**作者:** Ahmed Sameh `[一作]` (University of Minnesota Twin Cities), Yogatheesan Varatharajah `[通讯]` (University of Minnesota Twin Cities)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一系列统一的实验框架下，研究了自监督学习中时域上下文长度（16秒、1分钟、5分钟、10分钟）和前端编码方式（连续CNN patch嵌入与离散VQ tokenization）对心电图表示学习的影响，并通过下游心律分类和患者级检索来评估表现。

**💡 创新点**

系统化比较了不同时间尺度和连续vs离散编码对表示质量的影响，首次量化长时域训练在心律检测与患者一致性方面的优势，并揭示离散化带来的信息瓶颈。

**🔧 技术方法**

采用InfoNCE对比学习与Transformer编码器结合CNN patch或VQ token化，利用线性探测与端到端微调评估下游AFib/AFL分类，并使用Recall@1/5检索评估患者一致性，辅以t‑SNE可视化。

**📊 数据集**

使用公开的Icentia11k单导心电图数据集（约11,000名患者，250 Hz采样，约70 分钟/段）。

**📈 对比分析**

在相同Transformer骨干上预训练不同模型，随后在固定10分钟窗口上评估AFib/AFL分类（AUPRC、AUC）和患者检索（Recall@1/5）。结果显示，10分钟连续CNN模型取得最高AUPRC（0.960）和Recall@1（0.907）；5分钟模型亦优于16秒；连续编码始终优于离散编码；微调进一步提升分类但略降检索。

**⚠️ 局限性**

局限性包括仅采用单导通道数据；评估范围局限于Icentia11k数据集；VQ代码簿固定可能欠最优；未探讨其他Transformer规模或对比目标；仅针对AFib/AFL与正常分类；检索使用欧氏相似度，未尝试更复杂距离度量；缺乏真实临床部署验证。

---

## 22. Unifying Generative Models with Path Integrals

**arXiv ID:** 2608.12438 | [PDF](https://arxiv.org/pdf/2608.12438v1)

**作者:** Ramon Winterhalder `[一作]` `[通讯]`, Ramon Winterhalder

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将生成模型转化为连续路径积分的形式，构建一个统一的主行动，使得流模型、扩散模型、变分自编码器、对抗网络以及 Schrödinger 桥等多类模型都可视为该行动的不同近似或评价原则。

**💡 创新点**

创新点包括：① 用 Onsager–Machlup 和 MSRJD 表述统一的概率流动，① 分离自由与交互部分，② 推导一阶循环（one‑loop）校正，可在不增加采样成本的情况下显著提升确定性采样的精度；③ 将学习到的不完善得分视为图像插入，得到响应加权的分数匹配目标；④ 利用有效场论（EFT）算子展开，对称性约束下构造等变概率漂移，给出耦合层级。

**🔧 技术方法**

使用的技术主要是：路径积分与 Onsager–Machlup 动力学、MSRJD 形式化、自由–交互分解、图论展开与散射解释、循环展开与 Lyapunov 方程、有效场论算子展开、数值积分（Runge–Kutta、有限差分）以及对比实验验证。

**📊 数据集**

实验使用了可解析的线性漂移（Ornstein–Uhlenbeck）、一维立方漂移以及 24 维等变漂移的合成数据，未使用公开真实数据集，主要用于验证理论推导。

**📈 对比分析**

通过与解析结果、Euler–Maruyama 采样和有限差分求解的 Fokker–Planck 方程进行对比，单循环校正能将树级误差从 53% 降低到 1.6%；在非线性和高维等变模型中也表现出显著改进，验证了理论的可行性与优越性。

**⚠️ 局限性**

局限性包括：只给出一阶循环校正，无法捕捉更高阶非线性效应；对逆过程可逆性与连续性有假设；学习得分误差的近似可能在高噪声/高维场景下不足；循环展开需要求解耦合的 Lyapunov 方程，计算复杂度随维度平方增长，需采用低秩或近似技巧。

---

## 23. Geometric and Behavioral Stratification in Transformer Residual Streams

**arXiv ID:** 2608.12447 | [PDF](https://arxiv.org/pdf/2608.12447v1)

**作者:** Nelson Guda `[一作]` `[通讯]`, Nelson Guda

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Transformer 语言模型残差流的几何结构，发现预测方向（当前预测词的未嵌入向量）是一个“特权锚点”，残差流在其周围呈现层级化分层：紧靠预测方向的低维“预测接口”维度保持固定不随模型规模扩展，而剩余高维“补充”维度随模型宽度增大但与预测方向正交，且在此补充中存在动态与静态子空间；进一步通过介入实验验证预测接口维度对输出的即时影响和补充维度对后续生成的隐蔽作用。

**💡 创新点**

提出了以预测方向为基准的 PDSF（Prediction-Directed Subspace Field）分解法，揭示残差流在预测方向上的方向性、维度分层、流形复杂度梯度、补充子空间的反判别特征以及静态/动态分区；证明了模型宽度主要扩展补充子空间而非预测接口，强调了线性读取的几何约束。

**🔧 技术方法**

PDSF分解（预测方向、P；分辨性D、情境S、框架F三段高方差切分）；主成分分析（PCA）；参与比率等有效维度估计；余弦判别、kNN/全局距离比测量流形复杂度；对齐与旋转干预（单步KL、持续旋转、F-mix/F-attenuate等）来评估因果影响；静态/动态分区通过时间均值与差分拆解。

**📊 数据集**

18款 Transformer 模型（Llama、Gemma、Mistral、Mixtral、Qwen、GPT-OSS），规模从 7B 到 120B；三组提示集：SpecA（固定答案的表面变形）、SpecB（自由续写）、Diverse（多语言、多领域、代码等）。

**📈 对比分析**

对比不同模型、提示集下预测接口维度、流形复杂度梯度、判别能力以及干预敏感度；结果表明预测接口维度不随模型宽度变化；流形复杂度梯度在预测方向上显著；单步干预显示补充维度对分布变化的贡献最大；持续旋转干预揭示 D 对立即任务框架改变敏感，S/F-topK 影响后续内容；F动态干预导致几乎全部失败。

**⚠️ 局限性**

研究局限：仅使用了基于线性读取的 GPT/LLM 结构，未探究非线性读取或不同训练目标的模型；PDSF 分解在维度划分上依赖于参与比率阈值，可能对其他模型产生偏差；干预实验局限于特定层深度和提示集，未覆盖更长上下文或更大生成长度；结果主要描述了结构与行为的相关性，尚未给出完整的因果机制解释。

---

## 24. Proactive Computing

**arXiv ID:** 2608.12649 | [PDF](https://arxiv.org/pdf/2608.12649v1)

**作者:** Joonhee Lee `[一作]` `[通讯]` (Yonsei University), Joonhee Lee (Yonsei University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了主动计算的概念、技术支持、设计空间与挑战，提出了系统化框架；

**💡 创新点**

创新点在于将主动计算定位为跨传感、理解、决策、执行与治理的整体体系，并明确了从预测到行动的核心难点；

**🔧 技术方法**

主要涉及多模态感知、AI语境理解、边缘与分布式推理、物理执行等技术；

**📊 数据集**

未采用专门的数据集，主要引用现有研究和公开数据作为案例说明；

**📈 对比分析**

对比方法基于文献综述，未给出统一实验评估，主要讨论各技术在延迟、能耗、隐私、可信度等维度的优劣；

**⚠️ 局限性**

局限性包括缺乏统一评估基准、缺乏实证系统实现、对安全与隐私的保障不足，以及对文化与伦理差异的深入探讨仍不充分。

---

## 25. Synchronous Observers Revisited for Runtime Verification of Lustre Using STL

**arXiv ID:** 2608.12693 | [PDF](https://arxiv.org/pdf/2608.12693v1)

**作者:** Logan Kenwright `[一作]` (University of Auckland), Nathan Allen `[通讯]` (Auckland University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将同步信号时序逻辑（SSTL）公式编译成可在线运行的 Lustre 同步观察者，并在编译时即可完成静态模型检查。

**💡 创新点**

实现了任意深度嵌套的有界 SSTL 运算符支持、三值流式语义实现提前决策、以及通过移位寄存器实现的无界全局（□）观察者。

**🔧 技术方法**

采用 Lustre 同步语言、三值（Kleene）逻辑、叶子节点原语（bounded until）、移位寄存器技术和浏览器可视化工具。

**📊 数据集**

在弹簧质量系统和自适应巡航控制（ACC）两种案例研究上评估其性能和正确性。

**📈 对比分析**

与 Kind2 模型检查器和 Lustre 解释器对比，观察者在公式视界之前就能给出 definitive verdict，单周期执行成本均低于微秒，满足实时监测需求。

**⚠️ 局限性**

主要限制是叶子节点数量随窗口宽度的乘积指数增长，缺少共享或符号枚举优化；当前仅支持离散采样域，未涵盖连续鲁棒性语义。

---

## 26. Measuring Curriculum-Labor Market Alignment at the Scale of a Program Portfolio

**arXiv ID:** 2608.12356 | [PDF](https://arxiv.org/pdf/2608.12356v1)

**作者:** Sherzod Turaev `[一作]` (United Arab Emirates University), Khaled Shuaib `[通讯]` (United Arab Emirates University)

**通讯引用:** 3191 | [OpenAlex ID](https://openalex.org/A5028978354)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对某信息技术学院的五个本科专业进行统一、基于 ESCO 与布鲁姆认知层级的课程学习成果与职位需求的对齐分析，采用“实现获得”视角评估学生实际获得的知识；

**💡 创新点**

创新点在于（1）首次在全院层面统一应用同一对齐工具，克服单专业研究的局限；（2）引入实现获得范围模型，将课程目录与学分/选修约束结合，真实反映毕业生能获得的内容；（3）使用单一大型语言模型配合确定性“grounding gate”实现可靠、可验证的 competency 提取；

**🔧 技术方法**

技术实现包括 Claude Opus 4.8 进行原文 verbatim 提取，Deterministic gazetteer 进行标签、Grounding Gate 校验，基于 ESCO 的知识与技能匹配使用 GTE‑large embedding 进行概念相似度匹配，Bloom 级别由动词表映射；

**📊 数据集**

数据集包含 1,922 条课程学习成果（5 个专业）与 5,186 条去重的地区职位广告，合计 103,349 条提取的 competency；

**📈 对比分析**

通过覆盖率、认知层级差异及代表性比率等指标进行对齐评估，结果显示全院课程覆盖需求 46%，共享核心仅 31%，认知层级平均低 0.95 级；提取准确率约 98‑99%，互评可靠性 κ≥0.9；

**⚠️ 局限性**

局限包括单校案例、仅用一次性职位样本、对职业层级的动词级别采用首动词规则、缺乏更细粒度的 ESCO 子分类，且匹配阈值对细节结果存在敏感性；

---

## 27. Analysis of Motor Signatures of Social Adaptation in Autism for Efficient Human-Centric Systems

**arXiv ID:** 2608.12548 | [PDF](https://arxiv.org/pdf/2608.12548v1)

**作者:** Lara Pereira `[一作]` (University of Coimbra), João Ruivo Paulo `[通讯]` (University of Coimbra)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过3D动作捕捉，比较自闭症成人与神经典型成人在三种舞蹈模仿条件（基础、单人、双人）下的运动一致性，提出并量化了社交情境敏感性指数（SCSI），以评估社交情境对运动调节的差异。

**💡 创新点**

创新点包括：①提出SCSI这一可解释的指标，用于衡量社交情境对运动一致性的影响；②采用三阶段实验设计，将基础运动、单人模仿与双人社交模仿的差异明确分离；③将DTW一致性与SCSI结合，通过SVM实现客观且可解释的自闭症运动生物标志物识别。

**🔧 技术方法**

技术方法包括：动作捕捉数据预处理（滤波、插值、骨骼标准化）；Dynamic Time Warping (DTW) 计算试验内运动一致性；计算SCSI（双人-单人DTW差值）；统计检验（Mann‑Whitney U、Welch t‑test、Cohen d）；支持向量机 (SVM) 结合Leave‑One‑Subject‑Out (LOSO) 交叉验证与嵌套网格搜索进行分类。

**📊 数据集**

使用的公开数据集为 Move4AS（多模态数据集），仅采用其3D动作捕捉子集，用于实验和分析。

**📈 对比分析**

比较方法：对三种条件下的DTW距离和SCSI进行组间统计比较，并通过SVM在三种特征组合（单人、双人、单双+SCSI）下评估分类性能。结果显示，单独条件的平衡准确率仅为59.7%~66.7%，而联合使用SCSI后，平衡准确率升至79.2%，敏感度75%，特异度83.3%，显著优于单一条件。

**⚠️ 局限性**

局限性包括：①样本量有限且性别分布不均；②仅使用动作捕捉数据，未整合EEG等多模态信息；③实验仅涉及特定舞蹈模式，可能缺乏跨文化和跨风格的普适性；④未在实时或远程平台验证可用性，未来需要进一步评估其临床转化和人机交互应用。

---

## 28. Dual-Manifold Geometry Guided Representation Learning: Adaptive Coupling between Kernel and Data Spaces

**arXiv ID:** 2608.12737 | [PDF](https://arxiv.org/pdf/2608.12737v1)

**作者:** Wencong Zhang `[一作]` (Southern Medical University), Qianjin Feng `[通讯]` (Southern Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于核与数据双流流形的表示学习框架，并实现了Kernel-Guided Feature Transform (KGFT)模块，以通过核流形的几何信息来调节特征流形的协方差。

**💡 创新点**

将网络参数的流形几何与特征流形的几何显式耦合，利用核Gram矩阵生成指导矩阵，在浅层使用Exploit模式对齐，深层使用Explore模式扩展，配合可学习的强度和深度调度实现轻量化的几何引导。

**🔧 技术方法**

轻量化的KGFT模块、核Gram矩阵与协方差矩阵计算、Exploit/Explore双模式、可学习的指导强度、深度感知调度、残差融合、线性投影与层归一化。

**📊 数据集**

CIFAR‑100、ImageNet‑1K用于图像分类；ViT‑tiny、ResNet系列；LLaMA‑7B在MATH10K训练并在GSM8K、MAWPS、SVAMP、AQuA四个算术推理基准上测试。

**📈 对比分析**

在ResNet20/32/18/34/50等架构上与基线相比，KGFT平均提升1–2个百分点；在ViT‑tiny上提升0.6个百分点；在ImageNet‑1K上ResNet‑34/50提升0.5–1.1个百分点；在LLaMA‑7B算术推理上相较LoRA提升0.4–3.8个百分点，但在AQuA上略逊。

**⚠️ 局限性**

对不同任务的效果差异显著，且在某些算术推理基准上表现不如LoRA；未在检测、分割等下游任务验证；目前仍需手工设定Exploit/Explore切换策略，缺乏自动化调度；适用范围仍主要集中在CNN/Transformer，跨模态应用尚待探索。

---

## 29. On Nearly-Perfect Covering Codes Beyond Radius One

**arXiv ID:** 2608.12595 | [PDF](https://arxiv.org/pdf/2608.12595v1)

**作者:** Gabriel Sac Himelfarb `[一作]` (McMaster University), Moshe Schwartz `[通讯]` (McMaster University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对覆盖码的Van Wee界进行改进，定义了R≥2的近似完备覆盖码，并完整分类了R=2和R=3的近似完备覆盖码；

**💡 创新点**

首次把最小距离因素纳入Van Wee界，提出新的近似完备覆盖码定义，并通过数论（Thue‑Mahler方程）和群代数工具实现对特定半径的完整分类；

**🔧 技术方法**

采用球覆盖与球包围不等式、结构性引理、数论方法（椭圆曲线、Thue‑Mahler方程求解）以及群代数的等式推导；

**📊 数据集**

无实验数据集，工作完全基于理论证明和数值求解（例如使用SageMath、Magma等计算工具）；

**📈 对比分析**

方法通过严格不等式链和数论判定，得到精确存在/不存在结论；相较于已知的完备码或近似完备纠错码，证明了R≥3时存在的近似完备覆盖码数量有限；

**⚠️ 局限性**

局限在于对更大半径需逐案求解Thue‑Mahler方程，通用化困难；未证明所有近似完备覆盖码必为线性；在非二元字母表上扩展更具挑战性。

---

## 30. DrawTalking It Out: Creativity-Support Research as Creative Process Itself

**arXiv ID:** 2608.12357 | [PDF](https://arxiv.org/pdf/2608.12357v1)

**作者:** Karl Toby Rosenberg `[一作]` (New York University), Karl Toby Rosenberg `[通讯]` (New York University)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5015071256)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并实现了一款名为DrawTalking的交互式绘图+语音控制界面，同时提出将创作过程本身视为研究贡献的开放式方法；

**💡 创新点**

将创作过程记录与共享作为研究核心，强调交互技术而非单一工具，倡导面向非技术受众的创意支持与社区建设；

**🔧 技术方法**

使用语音识别与自然语言理解技术、手绘图形界面、语义可视化及基于自然语言的命令解释；

**📊 数据集**

未使用公开大规模数据集，主要基于研究者自身实验与用户访谈；

**📈 对比分析**

采用定性评估方法（访谈、可玩性测试），未给出量化性能指标；

**⚠️ 局限性**

缺乏系统的量化评估与大规模实验，过度依赖个人动机与非正式反馈，难以广泛验证与推广。

---

## 31. Tracing Provenance and Detecting Tampering with Complementary LLM Watermarks

**arXiv ID:** 2608.12713 | [PDF](https://arxiv.org/pdf/2608.12713v1)

**作者:** Xiaoyan Feng `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Cocktail 的水印方案，能够在 LLM 生成文本中同时提供来源归属（provenance）和篡改证据（tamper evidence）。

**💡 创新点**

创新点在于：①将稳健信号和脆弱信号通过同一 Token 共同嵌入；②使用不同密钥和不同种子窗口（短窗口用于稳健，长窗口用于脆弱）实现信号互补；③采用无偏向的多轮 Tournament 重加权和周期分配模式，调节两种信号的强度比例；④在二维得分空间中做三态判定（Intact、Tampered、No‑Watermark），从而有效抵御 piggyback spoofing。

**🔧 技术方法**

技术手段包括：①基于加密哈希的绿色/红色词表（green‑red list）作为水印信号；②在归一化文本（normalizer）上生成种子；③利用无偏向的向量化 Tournament 重加权实现多轮嵌入；④周期性分配回合（round ratio）控制稳健与脆弱信号的权重；⑤阈值化的 z‑score 评分和二维判定规则；⑥对重复上下文进行掩蔽以保证序列级无偏性。

**📊 数据集**

实验使用的文本数据集：C4（realnewslike 子集）和 LFQA；评测模型为 Llama‑3.2‑1B 和 Gemma‑3‑4B；攻击场景包括：同义词替换/重排（removal）、Dipper 语义重写（paraphrase）、双向机器翻译（round‑trip translation）、Token 代换（token substitution）、情感翻转（sentiment flip）以及字符混淆（homoglyph substitution）。

**📈 对比分析**

与四个基线（KGW、Unigram、SynthID、SIR）对比，采用 TPR@1%FPR 作为归属与篡改检测指标，且使用 Mistral‑7B‑v0.1 评估困惑度（PPL）。结果显示：
- 归属率在未攻击、同义词/翻译攻击下均保持 99.7–100%，即便在 4:1 轮次比例下仍与最强基线持平；
- 篡改检测率最高，达 89.5–100%（对 1% 假警率），而基线最高仅 23.1%；
- 困惑度与基线相当（≈10‑12），无明显质量损失；
- Ablation 证明了归一化、共嵌入以及长脆弱窗口是实现三态判定的关键。

**⚠️ 局限性**

局限性包括：
- 需要手动调节归一化策略与轮次比例，参数选择影响性能；
- 只对可见文本编辑有效，对能逃避归一化的隐形攻击（如特殊 Unicode 变体）仍有脆弱性；
- 评测范围局限于两款 LLM 与两个数据集，尚未验证在更大模型或更长文本（>512 词）上的表现；
- 对极端大规模文本或连续重写的鲁棒性尚未深入研究；
- 计算开销虽低于签名方案，但在大规模部署时仍需关注检测速度与资源。

---

## 32. A Generative Approach for Improving Multi-Label Defect Classification in Photovoltaic Modules

**arXiv ID:** 2608.12725 | [PDF](https://arxiv.org/pdf/2608.12725v1)

**作者:** Abdul Mueez `[一作]` (University of Central Florida), Shruti Vyas `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种以数据为中心的增强方法——生成缺陷隔离（GDI），通过基于分割掩码的神经填充将多缺陷图像转换为单缺陷或无缺陷训练样本，提升多标签分类的学习效果。

**💡 创新点**

创新点在于利用真实的像素级分割标注，结合LaMa大尺寸填充网络和频域卷积实现缺陷隔离，显著降低多缺陷图像带来的学习歧义，并通过生成高质量的单缺陷样本来增强模型的判别能力。

**🔧 技术方法**

技术手段包括LaMa大尺寸填充模型（含Fast Fourier Convolution），Vision Transformer（ViT‑S、ViT‑L）和EfficientNetV2‑L等主流网络架构，以及统一的训练增强与阈值调优流程。

**📊 数据集**

使用公开的UCF‑EL‑Defect光电发光图像数据集，包含九类缺陷以及额外的Contact_BeltMarks、Unknown和No_Defect标签，GDI在该数据集上生成大量单缺陷样本。

**📈 对比分析**

通过在20%与100%训练集拆分上与基线、Copy‑Paste增强以及已有工作进行对比，GDI在所有模型中均实现了显著提升，最终在完整数据上达到零一准确率0.6046、宏观F1分数0.7744，低数据环境下提升尤为显著。

**⚠️ 局限性**

局限性包括对分割掩码质量的高度依赖；当多缺陷遮挡关键结构（如电池格栅线）时，填充模型难以恢复，导致生成的无缺陷样本出现结构失真；此类极端情况虽罕见，但仍影响整体鲁棒性。

---

## 33. LoKiFormer: Locality-aware Attention with Decoupled Knowledge Memory for Efficient Large Language Model Pretraining

**arXiv ID:** 2608.12419 | [PDF](https://arxiv.org/pdf/2608.12419v1)

**作者:** Qiuwu Chen `[一作]` (AIGCode), Mingkui Tan `[通讯]` (South China University Of Technology)

**通讯引用:** 15386 | [OpenAlex ID](https://openalex.org/A5032352025)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的大型语言模型架构LoKiFormer，在标准Transformer解码器上加入本地融合注意力（LFA）和知识记忆模块（KMM），实现高效的局部依赖建模和全局知识检索。

**💡 创新点**

创新点在于：①在注意力前加入卷积融合，为模型注入显式的局部归纳偏置；②引入可参数化的键值记忆槽，将全局知识解耦出计算路径，实现可编辑、可解释的知识检索；③两者协同提升预训练收敛速度和下游性能。

**🔧 技术方法**

核心技术包括：卷积局部融合注意力（LFA）、多组卷积分支、参数化知识键值记忆（KMM）、多头潜在注意力（MLA）以及稀疏专家网络（MoE）。

**📊 数据集**

主要使用Matrix Data Pile（包含4.5T高质量双语文本）进行预训练，验证集取自Common Crawl 1%，SFT使用约10B的高质量指令集，评测基准包括MMLU、CMMLU、C‑Eval、HellaSwag、ARC‑Challenge、HumanEval、GSM8K等。

**📈 对比分析**

与同参数规模的Dense与MoE基线模型相比，在7B规模下LoKiFormer在MMLU、CMMLU、C‑Eval、HellaSwag、ARC‑Challenge等任务上分别提升约18–20点；在预训练阶段，1.33×更快收敛；在与更大规模开源/闭源模型对比时，虽参数仅7B，却在大多数指标上达到或接近更大模型的水平。

**⚠️ 局限性**

主要限制：仅在语言任务上验证，未对多模态或跨域通用性进行系统评估；知识记忆槽虽然可编辑但仍需手动设定字段数与维度；在极大模型规模或极长上下文下，卷积和记忆操作的计算与存储开销仍需进一步优化。

---

## 34. Class Geometry as Supervision for Sample-Efficient Open-World Detection

**arXiv ID:** 2608.12698 | [PDF](https://arxiv.org/pdf/2608.12698v1)

**作者:** Akash Rao `[一作]` (Auburn University), Sathyanarayanan N. Aakur `[通讯]` (Auburn University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了类几何监督（CGS）框架，利用视觉或语义类间不相似矩阵指导原型学习，以提升稀缺数据场景下的开放世界检测与新类插入性能。

**💡 创新点**

创新点在于将类间几何关系作为显式约束，将原型空间对齐至目标不相似图，从而在保持已知类识别的同时显著提升未知检测与新类插入的校准与样本效率。

**🔧 技术方法**

使用的技术包括基于原型的检测（如FSP‑DETR、DETR）、距离/相似度匹配、归一化的图距离对齐损失（ℒ_CG）、以及在不同阶段的冻结/全模型微调。

**📊 数据集**

主要数据集包括细粒度寄生卵（ova）数据集用于少样本识别与检测，以及 COCO 采用 OWOD 协议的开放世界检测基准。

**📈 对比分析**

与多种基线（普通检测器、ProtoNet、ProtoKD、FSP‑DETR 等）对比，CGS 在 5/10-shot ova 检测中提升 mAP 约 0.01-0.02，开放集检测 mAP 提升 0.016，COCO 上未知召回率显著提高（冻结头可提升 2-3% U‑Recall），但在某些设置下会略微降低已知类 mAP。

**⚠️ 局限性**

局限性在于对目标几何的依赖；当视觉和语义几何不一致或随机时效果不稳定；冻结头时会出现已知类 mAP 降低；此外，动态更新类几何以应对新类仍需进一步研究。

---

## 35. On the Exponential Circuit Imbalance of the Ben-Tal Nemirovski Approximation

**arXiv ID:** 2608.12550 | [PDF](https://arxiv.org/pdf/2608.12550v1)

**作者:** Jonah Bondar `[一作]` (University of Waterloo), Stephen Vavasis `[通讯]` (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过对Ben‑Tal Nemirovski（BN）圆盘近似的线性规划模型进行理论分析，构造其核空间中的循环，并证明该模型的最优电路失衡度（κ_W^*）随逼近步数指数增长，进而证明其最优条件数（χ̅_A^*）亦呈指数级恶化；

**💡 创新点**

创新点在于首次将电路失衡度与线性规划的最优条件数联系起来，并通过构造BN模型中的电路证明了自然生成的极差矩阵的指数级不良条件性，揭示了圆盘多面体逼近的内在数值缺陷；

**🔧 技术方法**

主要技术包括线性子空间的电路理论、极大极小比率图（circuit ratio digraph）以及最小最大定理（min‑max characterization）来下界最优电路失衡度，并利用递推关系与三角函数逼近得到指数下界；

**📊 数据集**

本文未使用实验数据集，全部结论来自严格的理论推导；

**📈 对比分析**

没有与其他算法或数据集进行性能比较；论文通过理论证明展示了BN模型的条件数指数增长，说明其在求解效率上会受到严重影响；

**⚠️ 局限性**

局限性包括：仅针对BN圆盘近似的线性模型进行分析；结果是否普适于所有圆盘多面体逼近尚未证明；缺乏实验验证与对比，无法量化实际求解时间或精度损失。

---

## 36. A Repeated-Game Framework for Incentives in Decentralized Infrastructure Protocols

**arXiv ID:** 2608.12576 | [PDF](https://arxiv.org/pdf/2608.12576v1)

**作者:** Mustafa Qazi `[一作]` `[通讯]` (Volt Capital), Mustafa Qazi (Volt Capital)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种重复动态激励框架，量化去中心化物理基础设施网络（DePIN）中合规性可激励性。

**💡 创新点**

创新点在于定义惩罚率Γ（惩罚比）并给出必要充分条件，将即时质押裁决与声誉惩罚结合实现合规。

**🔧 技术方法**

使用了重复道德风险博弈、概率检测、奖励/滑点机制、声誉阶梯、优化设计等理论技术。

**📊 数据集**

无实测数据集，采用理论模型与参数推导。

**📈 对比分析**

无实验比较，主要通过理论证明与闭式分析验证性能。

**⚠️ 局限性**

局限在于假设二元公共结果、状态可观测且无完美审计，未覆盖更复杂的多类违规或分布式身份攻击等场景。

---

## 37. Lifecycle-Aware Archival for Asymmetric Financial Datasets: A Production Study

**arXiv ID:** 2608.12367 | [PDF](https://arxiv.org/pdf/2608.12367v1)

**作者:** Tulika Manek `[一作]` `[通讯]` (Razorpay), Tulika Manek (Razorpay)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 PostgreSQL Aurora 上为金融交易服务实现了一套生命周期感知归档系统，通过在热表中仅保留活跃交易并定期清理已结算交易，同时使用 CDC 同步到冷库，实现了高效的存储与查询。

**💡 创新点**

创新点包括：1）对生命周期异构数据提出“Celebrity Partition Problem”，揭示传统分区在活跃/终止状态下的灾难性性能；2）设计了一种基于 ID 单调性（如 Snowflake、ULID）的去重算法，无需 Bloom 过滤器或额外索引；3）将归档窗口与去重逻辑统一，避免写放大和唯一性冲突。

**🔧 技术方法**

技术手段：PostgreSQL Aurora、Declarative Partitioning、CDC（Debezium）实现热/冷库同步、定时批量删除（purge job）、ID 单调性去重算法、监控与告警系统。

**📊 数据集**

数据集：十年积累、数十 TB、数十亿条交易记录，包含七个二级索引，约 94% 已结算，6% 活跃。

**📈 对比分析**

对比方法：部署前后对热表存储、行数、p99 延迟、写 CPU 进行测量。结果显示：热存储 95% 缩减，行数 94% 缩减，p99 延迟从 250 s 降至 100 s（≈60% 改进），写 CPU 由 96% 降至 45%（≈51% 降幅），整体基础设施成本下降 53%。

**⚠️ 局限性**

限制：归档窗口维持依赖 purge job，若 purge 失效或提前删除可能导致去重失效；冷库依赖导致读操作需跨库，增加系统复杂度和潜在延迟；系统假设 ID 生成器与写入时钟同步，若时钟漂移大于窗口宽度需额外校验。

---

## 38. Decentralized Multi-Player Q-Learning in Episodic Markov Decision Processes with Information Asymmetry

**arXiv ID:** 2608.12753 | [PDF](https://arxiv.org/pdf/2608.12753v1)

**作者:** Larissa Xu `[一作]` (University of California Los Angeles), William Chang `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

研究在信息不对称下，去中心化多玩家强化学习在有限情节MDP中的学习过程，并给出了三种不对称模型下的算法与理论收敛率。

**💡 创新点**

创新点在于将单玩家UCB‑Q学习的思想扩展到多玩家环境，利用确定性字典序打破Tie并通过隐式协调实现与集中式学习相同的O(√T)误差；并在最一般的不对称模型下提出两阶段探索-承诺算法，实现O(T^{2/3})误差。

**🔧 技术方法**

主要技术包括：确定性字典序解锁隐式同步、UCB‑Q学习与置信上界、动作消除策略、探测-承诺框架、马尔可夫链的Azuma‑Hoeffding收敛分析。

**📊 数据集**

无真实数据集，全部基于理论分析与合成情节MDP模型。

**📈 对比分析**

与集中式联合动作Q‑学习和现有多玩家Bandit方法对比，问题A/B的误差与单玩家上界相当；问题C的误差为T^{2/3}，比UCB方法更高，但仍为子线性；通过理论实验验证算法在小规模M和动作集合下能快速收敛。

**⚠️ 局限性**

局限性在于误差与联合动作空间A_joint指数级增长，导致仅在玩家数或每玩家动作数有限时才可实际应用；且问题C的T^{2/3}上界是否可进一步优化仍是未解决的开放问题。

---

## 39. Excess Separability: Nuisance-Controlled Residual-Stream Probing for Benchmark Contamination Detection

**arXiv ID:** 2608.12652 | [PDF](https://arxiv.org/pdf/2608.12652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 40. A Contract-Grade Verifier for LLM-Generated GPU Kernels, and a Native Blackwell Backward for the Gated-Linear-Recurrence Family

**arXiv ID:** 2608.12700 | [PDF](https://arxiv.org/pdf/2608.12700v1)

**作者:** Rishi Shah `[一作]` (E3A Healthcare), Rishav Shrestha `[通讯]` (E3A Healthcare)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于Kernel Contracts的十二门合同级核验证器，并用它审核了2638个机器生成的GPU核，发现62.1%存在合同违规，39.5%在无容差条件下直接失效；同时实现并验证了第一个Blackwell架构的本地GDN训练反向传播核，证明其正确性并与Triton实现做了速度对比。

**💡 创新点**

创新点包括：①将Kernel Contracts转化为可执行、无容差门的验证器；②首次对大量已接受的生成核进行大规模严谨审计，量化“正确率真空”；③首次实现本地Blackwell张量内存训练反向传播，解决开源实现的回退问题并通过双精度oracle验证；④用同一套验证器做正负对照，消除验证器偏差的疑虑。

**🔧 技术方法**

技术手段主要是：12门合同（CMP, ORD, PRC, EXC, RES），其中一部分门使用基于浮点误差模型推导的阈值，另一部分门完全无容差；使用高精度循环实现的参考实现；利用Triton与PyTorch的自动微分验证梯度；对TMEM生命周期做手动管理以满足Blackwell硬件限制；以及多重验证层级（oracle → 参考 → 组装 → 核）和手工审计。

**📊 数据集**

数据集：
- Dr. Kernel / KernelGYM公开语料库（共2638个已接受核，覆盖matmul、attention、softmax、scan、norm、conv、reduction等），
- 第二个原生CUDA核集（Sakana AI CUDA Engineer archive，213核）用于交叉验证。

这些核均来自公开生成系统，已通过其自身的容差检查。

**📈 对比分析**

比较方法：将验证器与KernelBench的标准测试（误差阈值10^-2、5次随机输入）做对比，构造2×2交叉表；对比本地GDN反向传播与Triton实现的执行时间，观察不同序列长度下的速度比例（大约8×到78×）。性能方面，本地核在现有实现基础上有2.75×至2.98×的加速，但相较Triton仍显著慢。

**⚠️ 局限性**

局限性：
- 验证器在某些门（CMP-02、RES-02）在仅有前向测试的语料库中无法使用；
- 参考实现和核只在B200 GPU、特定形状（d_k=128、chunk=64、d_v∈{64,128}）下验证，未覆盖更广泛配置；
- 本地GDN核整体仍包含非本地代码（如归一化梯度、掩码），因此并非完全原生；
- 速度评估基于单一硬件平台，未在不同GPU架构上复现；
- 由于使用高精度oracle，验证过程中可能忽略某些微小浮点差异。

---

## 41. General Probabilities of Causation with Causal Knowledge

**arXiv ID:** 2608.12657 | [PDF](https://arxiv.org/pdf/2608.12657v1)

**作者:** Xin Shu `[一作]` (Florida State University), Ang Li `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在多值处理和结果情形下，通过结合因果图中的协变量和中介变量，推导了PNS、PSub、PRep和PN的更紧的概率因果界限。

**💡 创新点**

创新点在于将二值PoC的因果图信息扩展到多值情形，并针对非后代协变量、后门集、部分中介和纯中介结构给出了新的闭式或改进的上界；同时证明了这些界限确实比现有方法更窄。

**🔧 技术方法**

主要技术是结构因果模型与线性/非线性规划相结合的概率边界推导，利用后门调整公式、条件独立性假设以及对联合潜在结果的枚举，得到分层界限后再加权；并用模拟实验验证理论。

**📊 数据集**

示例使用合成数据：分别在900例和2700例糖尿病药物实验中引入家族史和血糖中介；此外进行100,000次随机抽样的分布模拟以评估界限改进。

**📈 对比分析**

与Shu等人提出的多值PoC界限做对比；通过计算平均下界/上界改进、宽度以及改进比例等指标，发现非后代协变量结构下的界限平均宽度显著缩小，部分中介改进有限，纯中介也能提供显著提升；整体上新方法在大多数样本上都实现了更紧的界限。

**⚠️ 局限性**

局限包括：对高维多值情形下部分中介结构的改进非常有限；界限推导仍需要大量实验与观测数据的完整联合分布；并未给出闭式下界的进一步改进，且对连续变量的推广仍待研究。

---

## 42. Legally Mandated, but Still Inaccessible: Digital Tensions in Older Adults' Use of Norwegian Web Services

**arXiv ID:** 2608.12552 | [PDF](https://arxiv.org/pdf/2608.12552v1)

**作者:** Yavuz Inal `[一作]` (Norwegian University of Science and Technology), Eleftherios Papachristos `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对 294 名 55-90 岁老年人进行在线问卷调查，收集他们使用挪威 Web 服务的体验与障碍，并对开放式回答进行主题分析，结合 Google Lighthouse 自动评估，识别出四种数字张力（导航、语义、程序、时间）并讨论其对可访问性的影响。

**💡 创新点**

首次系统地将可访问性缺陷从技术合规层面扩展到用户体验层面，提出“数字张力”框架，揭示即使达到 WCAG AA 级别的高得分，老年人仍会遇到认知、导航、程序和时序上的障碍，强调需超越最低合规，关注实际使用体验。

**🔧 技术方法**

使用在线问卷平台（SurveyMonkey/Qualtrics）、主题分析方法（Inductive Coding）、Google Lighthouse 自动可访问性测试（WCAG 2.2），以及对比 Web 服务的平均可访问性得分。

**📊 数据集**

调查数据集：294 名 55-90 岁老年人对挪威公共/私营 Web 服务的使用频率、障碍与学习策略的自述；代表性网站可访问性得分（高频 97.9、适度 97.4、低频 92.7）。

**📈 对比分析**

与自动可访问性评分对照发现，尽管网站平均得分均高于 WCAG AA 级别，但用户报告的障碍与张力仍显著；表明自动化工具未能完整捕捉用户在实际使用中的认知与交互难点。

**⚠️ 局限性**

研究样本仅涵盖挪威老年人，缺乏跨地区、多语言与多平台的代表性；方法为横断面问卷，未能观察使用行为随时间的变化；未来需扩大样本、加入实地可用性测试，以验证数字张力框架的普适性。

---

## 43. DualPI2 Active Queue Management in ns-3: Implementation And Validation

**arXiv ID:** 2608.12513 | [PDF](https://arxiv.org/pdf/2608.12513v1)

**作者:** Maria Eduarda Veras `[一作]` (Federal University of Pernambuco), Judith Kelner `[通讯]` (Federal University of Pernambuco)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了 DualPI2 AQM 在 ns-3 中的完整模型，并通过 25 种 BDP 场景与物理 Linux DualPI2 进行对比验证。

**💡 创新点**

创新点在于采用 Linux 官方 DualPI2 的完整实现（信用基 WRR 调度、超载保护、step 标记等），实现了高保真度的 ns-3 模型，弥补了此前单队列或简化版模型的不足。

**🔧 技术方法**

使用了 ns-3 3.47、DCTCP 作为可扩展拥塞控制、信用基 WRR 调度、step 标记、过载保护机制，以及 物理 Linux 6.6.114-l4steam 内核作为对照。

**📊 数据集**

使用的“数据集”是 25 个不同的 BDP 组合（4/12/40/120/200 Mbps × 5 RTT 5/10/20/50/100 ms），每个场景重复 30 次，随机种子不同。

**📈 对比分析**

通过测量吞吐量、TCP 重传、队列延迟等指标与物理测试床比较，ns-3 DualPI2 模型在吞吐量公平性、低重传率和 L4S 队列延迟（<2 ms）方面与 Linux 结果高度一致，证明模型准确；在高容量高 RTT 场景下 ns-3 DCTCP 更激进但仍保持公平。

**⚠️ 局限性**

局限性包括：仅使用 DCTCP 而非正式的 TCP Prague；模型未覆盖 TCP Prague 的 Prague 要求；Linux DCTCP 在大 RTT 下的 cwnd 锁定导致与 ns-3 结果差异；未在无线或真实数据中心环境中验证；未评估整数算术误差对 ns-3 DCTCP 的影响。

---

## 44. DCBA: Detection of Collaborative Black-Hole Attacks in Connected Dominated Set using Baiting Process

**arXiv ID:** 2608.12347 | [PDF](https://arxiv.org/pdf/2608.12347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 45. Drive-to-Music: Context-Aware Generative Audio for In-Vehicle Experiences

**arXiv ID:** 2608.12615 | [PDF](https://arxiv.org/pdf/2608.12615v1)

**作者:** Cosmin Dragoiu `[一作]` (Mercedes-Benz Research & Development North America), Nooshin Nabizadeh `[通讯]` (Mercedes-Benz Research & Development North America)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个实时、上下文感知的车内音乐生成系统——Drive‑to‑Music，利用车载摄像头和遥测信息生成个性化音乐、标题及封面艺术；

**💡 创新点**

首次将多模态感知（视觉+运动）与生成式 AI 音频/图像模型结合，在真实车辆环境下实现低延迟、动态过渡的自适应音乐与视觉配套；

**🔧 技术方法**

核心技术包括：视觉‑语言模型 (VLM) 生成场景描述、语言模型 (LLM) 生成音乐描述与标题、Stable Audio 2.5 进行 90 秒以上乐曲合成、Stable Diffusion 3.5 Large Turbo 生成封面艺术；此外使用色彩提取、约束式安全校验与低延迟管道实现；

**📊 数据集**

数据来源为车载摄像头图像与车辆遥测（速度、温度等），基准评估使用约十份样本并采用 CLIP、MuQ‑MuLan、TAQS、GenEval 等指标；

**📈 对比分析**

通过对比 VLM（LiquidAI、Qwen、Gemini）、LLM（LiquidAI、Qwen、Gemini）、音频生成（ElevenLabs、Mubert、Stable Audio）以及图像生成模型的延迟与质量评分，最终选定 LiquidAI LFM 2.5 VL 1.6B（场景描述）、LiquidAI LFM 2.5 1.2B Instruct（音乐描述与标题）、Stable Audio 2.5（音频）、Stable Diffusion 3.5 Large Turbo（封面）；整体延迟约 1.2s + 0.7s + 6.2s + 1.4s，质量评分均高；

**⚠️ 局限性**

局限性包括：仅使用前置摄像头与有限遥测，缺乏大规模用户评估；系统对网络连接依赖较高；音乐内容受模型训练范围限制，无法覆盖所有风格与情境；缺乏对驾驶安全影响的深入研究；

---

## 46. Specification-first convergence with an AI coding agent: a case study of dismantling a core architectural invariant across 189 files in a 717k-line codebase with no test oracle and no human code review

**arXiv ID:** 2608.12440 | [PDF](https://arxiv.org/pdf/2608.12440v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 47. Regulatory Approval Is Not Enough: Gaps in Trustworthy AI Reporting in FDA-Cleared Medical Devices

**arXiv ID:** 2608.12360 | [PDF](https://arxiv.org/pdf/2608.12360v1)

**作者:** Ahmed M Salih `[一作]` (University of Leicester), Karim Lekadir `[通讯]` (University of Barcelona)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2021-2025年FDA公开的AI/ML医疗设备摘要报告进行系统评估，基于FUTURE‑AI框架衡量可信AI的六项原则

**💡 创新点**

首次量化FDA报告中可信AI证据的完整性与透明度，揭示缺口及无改进趋势

**🔧 技术方法**

自动关键词检索+多阶段人工一致性评估，采用多变量Logistic回归

**📊 数据集**

FDA 2021-2025年AI/ML医疗设备摘要报告（共519份）

**📈 对比分析**

通过描述性统计和回归分析比较不同年份和临床领域的报告透明度，结果显示大部分报告仅涵盖1-2项原则，且无显著提升

**⚠️ 局限性**

仅评估公开报告，可能忽略保密数据；二元评分可能低估证据深度；未对证据质量进行定量评估

---

## 48. Exemplar-based objective classification of gust-induced loads across multiple flight conditions

**arXiv ID:** 2608.12448 | [PDF](https://arxiv.org/pdf/2608.12448v1)

**作者:** Paolo Olivucci `[一作]` (Technische Universität Braunschweig), David E. Rival `[通讯]` (Technische Universität Braunschweig)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用实验数据对非细长三角翼在六种飞行姿态下受到的随机迎风冲击引起的气动负载进行数据驱动的代表性事件（exemplar）归类，找出了九种基本负载响应类型。

**💡 创新点**

创新点在于将核心集（core‑set）思想与机器学习嵌入相结合，通过训练好的多层感知机（MLP）得到的中间层特征空间进行k‑medoids聚类，从而在不使用姿态标签的情况下以客观、可解释的方式识别出最能代表整个数据库的少数高信息量事件。

**🔧 技术方法**

使用的技术包括：多层感知机回归模型、MSE 损失训练、MLP 中间层特征作为嵌入、L2 距离度量、k‑medoids 聚类、以及与随机子集进行对比的学习曲线评估。

**📊 数据集**

使用的数据集为3479个迎风冲击事件，来自一台非细长三角翼实验装置的随机迎风生成系统，记录了四个压强传感器信号以及六轴平衡器的升力和侧向力输出。

**📈 对比分析**

通过对比随机子集、无姿态标签子集和含姿态标签子集的学习曲线，发现无姿态标签的9个核心事件在测试集上均能把MSE降至0.161（比随机子集降低约66%），而完整数据库的基线误差为0.099，表明核心集压缩率约为0.3%。

**⚠️ 局限性**

局限性包括：仅利用稀疏压强传感数据，缺乏完整的流场测量，导致对流动机制的解释仅基于推测；核心集的可推广性到其他机型或更复杂的飞行环境尚未验证；以及聚类结果对模型超参数（如MLP宽度、k‑medoids数量）的敏感性尚未系统评估。

---

## 49. Position: We Need Practical AI Alignment Methods to Mirror Human Reasoning

**arXiv ID:** 2608.12372 | [PDF](https://arxiv.org/pdf/2608.12372v1)

**作者:** Vijay Keswani `[一作]` (IIT Delhi), Jana Schaich Borg `[通讯]` (Duke University)

**通讯引用:** 1739 | [OpenAlex ID](https://openalex.org/A5044954263)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文从用户角度系统探讨认知对齐AI（Cognitive Alignment AI）的需求与重要性，设计并开展了一项大规模用户调查，评估人们在不同决策领域对“像人思考”的AI的偏好，并提出未来研究议程与方法论方向。

**💡 创新点**

创新点在于：①首次量化用户对认知对齐AI的渴望，发现高风险决策场景中认知对齐是用户信任和委托的重要前提；②系统性梳理现有对齐方法的局限（可解释性不可验证、与人类认知不一致）；③提出可验证的认知对齐评估框架和多层次认知抽象、学习与评价的研究优先级。

**🔧 技术方法**

主要技术手段包括：用户调查与问卷设计、对比实验（人类思考AI vs 过程隐藏AI vs 机器思考AI）、统计分析（显著性检验、置信区间）、认知科学方法（过程追踪、眼动跟踪、思考大声法）、以及可解释AI与交互式学习的辅助技术。

**📊 数据集**

使用的数据集主要是 Prolific 平台收集的 150 名受试者的问卷数据，覆盖 16 个决策领域（医疗分配、金融、法律、军事等），并在两大实际案例（自动驾驶与肾移植分配）中进行更细粒度的属性评估。

**📈 对比分析**

方法对比通过比较不同 AI 类型在各领域的偏好比例来评估用户接受度；在高风险领域，人类思考 AI 的受欢迎程度显著高于其他类型，但论文未给出具体算法性能指标（如准确率、推理速度），主要聚焦于用户主观偏好与信任评估。

**⚠️ 局限性**

局限性包括：①样本局限（仅美国用户、样本量 150，未覆盖多元文化）；②仅基于自我报告和假设情景，缺乏真实决策场景验证；③未提供可实现的认知对齐模型或算法，仅提出研究路线；④对认知对齐与准确率之间可能的权衡未作系统评估。

---

## 50. Diagnostic Foundation for Evaluating LLMs' Research Integrity as Co-Scientists

**arXiv ID:** 2608.12345 | [PDF](https://arxiv.org/pdf/2608.12345v1)

**作者:** Yash Tripathi `[一作]` (Jaypee University of Information Technology), Lin Li `[通讯]` (University of Oxford)

**通讯引用:** 19667 | [OpenAlex ID](https://openalex.org/A5100412894)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了IntegrityBench基准，用于评估大型语言模型在学术环境下的研究诚信表现，尤其是在不同压力情境下的决策行为。

**💡 创新点**

创新点在于：①将研究诚信拆分为三大维度（违规识别、伦理行动推理、基于证据的决策）并配对违规与伦理对照任务；②设计了5级隐式-显式压力协议，模拟机构层面压力；③系统性评估18个前沿模型的诚信表现，揭示规模与推理并非提升诚信的关键。

**🔧 技术方法**

主要技术包括：大型语言模型推理（如GPT、Claude、Gemini、Qwen、DeepSeek）、基于prompt的压力注入、自动评分与人工校验的混合评估流程。

**📊 数据集**

使用了自构造的36个任务（共1800条prompt），涵盖3大违规类别、3科研领域和4个研究阶段，任务由具伦理训练的研究者设计并通过领域专家验证。

**📈 对比分析**

与现有安全/伦理基准对比，IntegrityBench揭示模型在高压情境下平均诚信得分仅为68.7，误差率约为每3个关键决策就会出现1个错误；规模与推理未能显著提升表现，表明诚信问题主要源自对齐缺失。

**⚠️ 局限性**

局限性包括：任务样本量有限、仅覆盖3个学科、违规识别采用多项选择而非开放式生成、未对完整AI科研系统进行评估，未来需扩展域、任务数以及开放式标签和更丰富的工具链。

---

## 51. MAG: MAnifold Guided Semi-Supervised Multi-modal In-Context Learning

**arXiv ID:** 2608.12724 | [PDF](https://arxiv.org/pdf/2608.12724v1)

**作者:** Zirui Cheng `[一作]` (National University of Singapore), Nancy F. Chen `[通讯]` (A*STAR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了MAG框架，利用无标签多模态数据通过半监督图传播方式提升多模态ICL性能。

**💡 创新点**

创新点在于两阶段图传播：先用文本图进行相关性传播筛选高价值无标签样本做伪标签，再在视觉+文本双图中进行查询条件传播并使用LateFusion选取演示，整个过程无需额外学习。

**🔧 技术方法**

采用k‑NN图构建、标签传播（闭式解）、文本与视觉编码器（Contriever、CLIP ViT‑L/14）、多模态LateFusion、MLLM推理（Gemini‑2.0‑Flash）。

**📊 数据集**

在八个视觉‑语言基准上评估：EmoSet、Emotion6、TextOCR、MMStar、MatchingMI、CLEVR、GQA、OKVQA。

**📈 对比分析**

与七个基线（Few‑shot、VICL、Top‑K+MDL、MMICES、CVR‑LLM、MAPLE、Zero‑shot、Cola‑Zero）对比，MAG在15标签+45伪标签下平均提升显著，尤其在CLEVR（+30%）、TextOCR（+23%）等任务上领先20%以上。

**⚠️ 局限性**

局限性包括：仍需大量MLLM推理成本；伪标签质量受图传播参数影响；对极大候选集或高标签率的可扩展性未充分验证；仅在八个基准上实验。

---

## 52. SteerBench-Work: A Benchmark for Agent Steering at Action Boundaries

**arXiv ID:** 2608.12654 | [PDF](https://arxiv.org/pdf/2608.12654v1)

**作者:** Oguz Serdar `[一作]` (AgentDock), Cuneyt Mertayak `[通讯]` (AgentDock)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SteerBench-Work基准，用于评估工作场景下LLM代理在提交前决策（是否执行动作或暂停）并记录二元门控结果；

**💡 创新点**

创新点包括：①基于公开事件的锚定场景；②与其对应的“镜像”场景通过翻转验证状态；③同时衡量过度拒绝和不足拒绝的双向错误；④按严重性分级的不可逆风险权重；⑤按领域和动作影响分层；⑥提供开放训练视图。

**🔧 技术方法**

技术实现：使用单轮提示，模型返回结构化决策（允许/拒绝、政策动作、置信度、理由），并通过解析器获取评估字段；评估指标包括平均试验准确率、modal‑of‑5、pass5以及加权严重性得分。

**📊 数据集**

数据集：106个场景，包含公开事故、13个镜像对、风险已解决提交、校准控制等；每个场景包含用户请求、拟议动作和可获得的证据。

**📈 对比分析**

比较方法：对30个模型条件（OpenAI、Anthropic、Google、DeepSeek、Kimi、开源gpt‑oss等）在相同提示下运行5次，报告平均准确率与失误率；模型平均准确率最高达92.8%，过度拒绝率约28%，不足拒绝率约1%。

**⚠️ 局限性**

局限性：①场景为单轮描述，未模拟持续交互和后续行为；②评估仅针对提交前边界，无法衡量实时决策；③模型结果为快照，受更新影响；④未对每个条目的难度进行量化；⑤对提示细微修改的鲁棒性未评估；⑥公开数据集可能随时间被污染。

---

## 53. Learning Under Treatment-Induced Label Indeterminacy with Expert Annotations of Counterfactual Outcomes: A Case Study in Neurological Prognostication

**arXiv ID:** 2608.12477 | [PDF](https://arxiv.org/pdf/2608.12477v1)

**作者:** Xiaobin Shen `[一作]` (Carnegie Mellon University), George H. Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1109 | [OpenAlex ID](https://openalex.org/A5015253912)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了针对治疗导致标签不确定性（label indeterminacy）情况下的预测模型评估框架，并在训练中结合专家对未观测病例的反事实估计，实现对已观测病例与未观测病例性能的分裂评估与权衡。

**💡 创新点**

创新点在于：① 将确定病例（可观测标签）与不确定病例（专家估计）分离，建立分裂评估体系；② 引入可调节的权重与对齐损失，使模型在保持对确定病例性能的同时可主动对齐不确定病例的专家概率，从而显式展现“确定-不确定”性能权衡。

**🔧 技术方法**

技术手段包括：使用单层神经网络进行二分类预测，采用加权二元交叉熵和平方对齐损失的组合；对照基准模型包括 XGBoost、随机森林、TabPFN 等；评估采用 5 折交叉验证，计算 AUROC、Brier 分数（确定病例）与 MAE（不确定病例）。

**📊 数据集**

数据集来自单中心心脏骤停后神经预后注册库，最终包含 2,497 名患者，其中 1,068 名为确定病例、1,429 名为不确定病例；不确定病例获得 3–5 份专家对“如果继续治疗会否恢复”的分级评分，并映射为概率值。

**📈 对比分析**

比较方法：在确定病例上评估 AUROC 与 Brier 分数，在不确定病例上评估与专家估计的 MAE。结果显示，模型在 AUROC 上相近，但 Brier 与 MAE 之间存在显著权衡；最强对齐模型（ω_bad=32, λ_align=4）将不确定病例 MAE 降至 0.053，但其 Brier 分数升至 0.407，表明性能存在明显的确定-不确定权衡。

**⚠️ 局限性**

局限性：① 单中心回顾性数据，确定/不确定划分受本地临床实践影响，可能不具备普适性；② 专家估计并非真实反事实标签，过度对齐可能导致偏差；③ 仅评估静态预测，未考虑随访动态信息；④ 该框架未解决真正的反事实标签获取与估计问题。

---

## 54. Learning to Adapt Cross-Domain Preferences via Meta-LoRA for LLM Personalization

**arXiv ID:** 2608.12389 | [PDF](https://arxiv.org/pdf/2608.12389v1)

**作者:** Xuefei Wang `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究跨域零/少量样本的LLM个性化问题，提出PAC-Bayes校准的Meta-LoRA与图引导的域软令牌双通道先验组合方法，以实现少量目标域证据下的稳健个性化生成。

**💡 创新点**

创新点在于：①将Meta-LoRA学习到的可迁移LoRA初始化作为PAC-Bayes先验中心；②根据支持集大小与预测不确定性自适应调节更新幅度；③功能分解用户与域先验，分别通过可读提示和拓扑保持的软令牌实现双通道条件。

**🔧 技术方法**

使用的技术包括PAC-Bayes正则化的Meta-LoRA、LoRA参数化与梯度自适应、图融合与Wasserstein barycenter、CKA拓扑保持、软令牌注入、文本提示生成、离散化自监督学习等。

**📊 数据集**

实验数据集包括HiCUPID、LaMP-QA以及在Qwen3-8B上进行的跨模型验证，源域与目标域均为独立拆分。

**📈 对比分析**

与检索/提示、参数微调、推理时个性化、MAML等基线比较，WR提升显著，交叉域降幅减少47.9%，未见用户冷启动提高110.2%，并在多数据集、多模型上保持一致优势。

**⚠️ 局限性**

限制包括需要大量源域任务进行元训练、对稀疏域先验的图质量高度依赖、对高维用户历史摘要效果有限，以及硬件资源与推理延迟未做深入优化。

---

## 55. Entropy-Augmented Multi-Objective Policy Optimization in Multiagent Systems

**arXiv ID:** 2608.12534 | [PDF](https://arxiv.org/pdf/2608.12534v1)

**作者:** Jamie Santos `[一作]` (Oregon State University), Kagan Tumer `[通讯]` (Oregon State University)

**通讯引用:** 4304 | [OpenAlex ID](https://openalex.org/A5084748531)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种基于熵奖励的策略评估方法，鼓励多智能体系统在进化搜索中保持行为多样性，从而提升多目标优化的Pareto前沿质量。

**💡 创新点**

创新点在于将行为空间的熵作为奖励加成直接加入多目标进化算法NSGA-II的适应度计算，而非作为单独目标，从而在保持全局Pareto结构的同时促进行为多样性。

**🔧 技术方法**

使用技术包括多目标进化算法NSGA-II、基于kNN的熵估计、强化学习中的CTDE框架以及神经网络控制器的参数化进化。

**📊 数据集**

数据集主要为模拟的多智能体Rover任务环境，包含两种实验设置：障碍导航（两目标）和时间权衡（两时间段奖励）。

**📈 对比分析**

实验通过比较不同熵系数β对NSGA-II的超体积（Hypervolume）影响，结果显示在实验1中熵增强提升约18%超体积，实验2中提升约48%；与仅使用目标空间评估相比，表现更优。

**⚠️ 局限性**

局限性包括熵奖励的权重选择敏感，过大会淹没原始目标；实验样本量有限，方差较大，需在更多随机种子和更复杂环境中验证鲁棒性。

---

## 56. Beyond Source: An Empirical Study of Python Bytecode Security Risks

**arXiv ID:** 2608.12853 | [PDF](https://arxiv.org/pdf/2608.12853v1)

**作者:** Baihong Chen `[一作]` (Utah State University), Wen Li `[通讯]` (Utah State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性地对Python字节码在PyPI包中的分布、可分析性、运行时鲁棒性以及源代码可重现性进行了经验研究。

**💡 创新点**

将字节码视为安全首要工件，首次量化字节码曝光与缺失源码情况；结合版本感知工具评估可分析性；使用版本匹配CPython fuzzing揭示多重运行时崩溃；证明字节码级别发现与普通源代码不可重现，凸显字节码与源代码间的安全鸿沟。

**🔧 技术方法**

使用PyPI数据抓取、文件结构分析、Python标准库加载/disassembly、Decompyle++、PyLingual反编译、版本匹配CPython构建、honggfuzz对序列化字节码进行变异，以及源代码恢复工作流。

**📊 数据集**

1,034,843个PyPI发行版（wheel和sdist）中，收集到7,388个包含字节码的包，包含228,578个.pyc文件和28,193个仅字节码文件。

**📈 对比分析**

对比不同工具在同一字节码集上的加载、反编译成功率（204,901/204,904），记录失败类型；对比fuzzing产生的崩溃组数量（1,009）与各CPython 3.8–3.14版本差异；结果主要强调存在性和覆盖面，而非传统性能指标。

**⚠️ 局限性**

研究仅覆盖PyPI公共包，未考虑安装后生成的字节码；工具集有限，未来版本可能恢复更多；fuzzing单次实验且未使用sanitizer，结果仅为存在性证明；源代码恢复工作流有限，未评估可利用性。

---

## 57. Excitation-Supervised Closed-Loop Self-Calibration and Target Seeking for an Unknown-Pose Range-Bearing Relay

**arXiv ID:** 2608.12528 | [PDF](https://arxiv.org/pdf/2608.12528v1)

**作者:** Yash Bagla `[一作]` `[通讯]` (Michigan State University), Yash Bagla (Michigan State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于轨迹扩散度 S_v 的在线自校准与目标寻优控制框架，能在车辆运动过程中实时判定是否已获得足够激励并自动重启探索运动。

**💡 创新点**

创新点在于把 S_v 作为可量化的证书用于在线监督；通过投影式寻优避免激励干扰，并给出在采样与窗口假设下有限时间内获取所需激励的理论保证。

**🔧 技术方法**

使用 Gauss‑Newton 估计、投影式寻优、周期性激励与自适应重置、持续激励理论分析，以及 ROS2/Gazebo 软件仿真实现闭环控制。

**📊 数据集**

使用合成噪声测量与 100 次 Monte‑Carlo 仿真数据集，以及 ROS2/Gazebo 物理仿真中的真实消息传输与感知延迟。

**📈 对比分析**

与固定激励、信息梯度激励等对比，实验表明在激励不足或快速衰减场景下，监督式算法显著降低偏航误差并保持定位精度，同时目标误差基本保持不变。

**⚠️ 局限性**

局限性包括仅在二维模型下验证、窗口随时间累积导致内存增长、未在真实机器人上测试、对极端延迟或低采样率的鲁棒性仍待进一步证明。

---

## 58. Lonic: Algorithm-Hardware Co-Design for Energy-Efficient Fully Local Online SNN Training with INT4 Precision

**arXiv ID:** 2608.12500 | [PDF](https://arxiv.org/pdf/2608.12500v1)

**作者:** Peilin Chen `[一作]` (University of Virginia), Xiaoxuan Yang `[通讯]` (University of Virginia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了Lonic框架，实现了完全本地在线SNN学习的INT4低精度训练，并给出了专用加速器。

**💡 创新点**

创新点包括首次将INT4低精度训练应用于完全本地在线学习，并设计可重构无乘法整数PE、双重零门控、时间前缀加速数据流以及低精度权重搬运等硬件优化。

**🔧 技术方法**

技术手段涵盖了尺度权重标准化替代BN、sWCTT归一化、量化训练流、整数PE、零门控压缩、时间前缀并行以及权重量化/搬运技术。

**📊 数据集**

实验使用CIFAR-10、CIFAR-100、DVS-CIFAR10、DVS128-Gesture以及VGG11/16网络进行验证。

**📈 对比分析**

与Apple M4、Nvidia V100 GPU以及TPU‑like和H2Learn ASIC进行对比，Lonic在能效上分别提升约17.44×/66.28×/15.95×/1.52×，吞吐量提升约3.25×/1.02×/12.46×/1.02×，并保持与SOTA相近的准确率。

**⚠️ 局限性**

局限性包括在大规模 GEMM 任务上吞吐量提升有限、依赖单样本批次导致训练速度受限、以及未验证更大模型或其他数据集的可扩展性。

---

## 59. When Explanations Betray Backdoors: Black-Box Auditing for Language Model Classifiers

**arXiv ID:** 2608.12623 | [PDF](https://arxiv.org/pdf/2608.12623v1)

**作者:** Yang Liu `[一作]` (University of North Carolina at Chapel Hill), Ran Zou `[通讯]` (University of California, Irvine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于模型生成的解释进行黑盒后门检测的方法，能够在无触发器信息的情况下识别文本分类器的后门行为。

**💡 创新点**

创新点在于提出轻量级的归因漂移评分（衡量解释与输入的一致性）并结合多探测器升级，以在解释隐蔽攻击下提升检测效果。

**🔧 技术方法**

技术包括词表与TF‑IDF的输入 grounding 计算、支持/文档一致性探测、使用干净校准集经验分位数确定阈值，以及多查询（解释与引用证据）自洽验证。

**📊 数据集**

使用五个公开文本分类数据集：SST‑2、Rotten Tomatoes、TREC、Jigsaw Toxicity 与 AG News。

**📈 对比分析**

通过与 ONION、BBCaL、CoS 及 Reasoning‑最佳等基线在 5% 干净 FPR 条件下比较，AUROC 通常超过 0.9，残留目标攻击成功率低于 5%；在解释隐蔽攻击下虽然提升，但仍未完全消除残留风险。

**⚠️ 局限性**

局限性包括仅适用于能返回解释的 LM 分类器、需要小规模干净校准集、对自洽但不因果的解释仍可能误导，以及对某些自适应攻击存在检测盲区。

---

## 60. Thought-Aware KV Cache Compaction for Reasoning via Adaptive Attention Matching

**arXiv ID:** 2608.12331 | [PDF](https://arxiv.org/pdf/2608.12331v1)

**作者:** Yang Liu `[一作]` (Tsinghua University), Xu Kefu `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种思维感知的注意力匹配（TAM）方法，用于在链式推理过程中对KV缓存进行结构化压缩。

**💡 创新点**

创新点在于将推理轨迹按思考块分段、根据重要性动态分配压缩预算，并对高注意力的关键代币进行保护，从而在压缩时保持最重要信息。

**🔧 技术方法**

采用了注意力匹配（Attention Matching）技术、局部查询窗口、核函数选择、非负最小二乘（NNLS）和普通最小二乘（OLS）等核心算法实现压缩与重构。

**📊 数据集**

使用了 AIME 2024 和 MATH-500 两大数学推理数据集，基于 Qwen3‑4B 语言模型进行实验。

**📈 对比分析**

与统一压缩、eviction、AM+Repeat、PAM 等基线相比，TAM 在保持相同压缩比例下提升 1.8–3.3% 的准确率，并在周期性压缩模式下实现 65% 的显存占用下降。

**⚠️ 局限性**

局限性包括依赖于双换行等启发式分段方式和局部查询窗口估计重要性，且实验仅覆盖数学推理任务和单一模型，缺乏跨领域与更大模型的验证。

---

## 61. GENADA: efficient generative time series adversarial attack framework

**arXiv ID:** 2608.12535 | [PDF](https://arxiv.org/pdf/2608.12535v1)

**作者:** Michael Baronov `[一作]`, Alexey Zaytsev `[通讯]` (Moscow Independent Research Institute Of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为GENADA的生成对抗攻击框架，旨在通过学习生成扰动来攻击时间序列分类模型。

**💡 创新点**

创新点在于通过训练一个生成模型，直接在一次前向传播中生成对抗扰动，避免了传统方法中反复计算梯度的高计算成本。

**🔧 技术方法**

使用了生成对抗网络（GAN）技术来生成扰动，并提出了一种基于冻结目标模型的训练程序。

**📊 数据集**

使用了PowerCons、GunPoint和Strawberry等多个时间序列分类数据集进行验证。

**📈 对比分析**

与强基线（如FGSM和iFGSM）进行比较，GENADA在攻击质量上表现相当，但生成扰动的时间显著减少，显示出更高的计算效率。

**⚠️ 局限性**

限制在于目前仅在二分类设置中进行评估，未来需要扩展到多分类场景，并对生成器架构和训练目标进行更系统的分析。

---

## 62. LLMs Are Not Good Strategists, Yet Memory-Enhanced Agency Boosts Reasoning

**arXiv ID:** 2608.12626 | [PDF](https://arxiv.org/pdf/2608.12626v1)

**作者:** Yi Wu `[一作]` (University of Chicago), Zhimin Hu `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 EpicStar 框架，利用结构化的回放记忆与工作记忆来提升 LLM 在 StarCraft II 这类长期规划任务中的战略连贯性与效率。

**💡 创新点**

创新点在于：①将过去成功经验作为非参数策略直接检索并重用；②通过动态门控机制平衡记忆检索与新推理；③使用双向上下文融合将工作记忆与回放记忆相互调节，提升对局部情境的适应。

**🔧 技术方法**

核心技术包括：episodic memory 存储与检索（基于相似度排序），工作记忆的时间间隔缓存，动态门控模块，contextual fusion 的 prompt 设计，以及 LLM 推理与动作生成。

**📊 数据集**

数据集：使用 TextStarCraft II 接口收集的 4,592 条获胜游戏回放（在 Levels 6‑7 进行 20 场对战），以及在不同地图与敌手策略下的评估数据。

**📈 对比分析**

与 CoS 基线对比：在 Level 5 与 Level 6 的内置 AI 对战中，EpicStar 在 5‑6 级别均实现显著更高的胜率（最高 75 %）且 Token 消耗约为 CoS 的 15 %；在多模型（GPT‑3.5‑Turbo、GPT‑4‑Turbo、GPT‑4o‑Mini、GPT‑4o）上表现一致，且 ablation 研究验证了探索与上下文融合两项组件的贡献。

**⚠️ 局限性**

局限性：①记忆库规模较小，未检验规模扩展效果；②对噪声/冗余回放的鲁棒性未知；③对新颖对手的泛化受限，需更可靠的记忆适用性检测；④在大规模记忆时提示长度增长导致推理开销与实现复杂度上升。

---

## 63. Novels generated by language models show compressed formal variation

**arXiv ID:** 2608.12630 | [PDF](https://arxiv.org/pdf/2608.12630v1)

**作者:** Mehdy Sedaghat Payam `[一作]` (University of Maryland), Justin Quinn `[通讯]` (University of West Bohemia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比 GPT‑5.5 Thinking 与 Qwen3‑14B 生成的小说与人类小说在句子长度、可读性、词汇多样性等文体特征上的文档级多样性进行系统评估。

**💡 创新点**

提出“过度闭合（overclosure）”概念，揭示 AI 生成小说在跨作品层面上显著压缩文体范围，而非单一文本可被辨识。

**🔧 技术方法**

采用 MATTR、Shannon entropy、句子长度、Flesch 可读性、标点频率等定量指标，并使用 Brown–Forsythe 检验、作者平衡自助法、长度残差分析以及 Pearson/Spearman 相关性检验。

**📊 数据集**

六个语料库：205 本19 世纪英国人类小说、65 本当代“零风格”人类小说，及各 20 本 GPT‑Victorian、GPT‑Zero‑Style、Qwen‑Victorian、Qwen‑Zero‑Style 生成小说。

**📈 对比分析**

通过标准差比率、Bootstrap 置信区间、残差方差比等多重检验，发现所有 AI 语料库的句子长度和可读性等指标的跨作品方差显著低于人类对照组（p<0.05），但在词汇多样性上差异不一致。

**⚠️ 局限性**

结果仅适用于所用模型–工作流组合和选定人类对照，且仅覆盖表层文体特征，未考察情节、人物刻画等深层文学属性。

---

## 64. Class-Structure Preservation Beats Diversity: A Comprehensive Benchmark of Text Augmentation Methods for Imbalanced Text Classification

**arXiv ID:** 2608.12340 | [PDF](https://arxiv.org/pdf/2608.12340v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Keito Inoshita `[通讯]` (Kansai University)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5106529447)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比并评估了基于经典扰动、嵌入空间检索以及大语言模型生成的11种文本数据增强方法，构建了一个包含7个公开数据集、5个随机种子、宏F1等指标的系统基准。

**💡 创新点**

创新点在于首次将SMOTE式检索 EmbSMOTE 纳入基准，并通过严格的统计检验、分布度量、类结构保真度分析等方法揭示了LLM生成增强在不平衡多分类场景中往往不如检索增强；提出的 VoidGen 作为定位后解码的探测方法。

**🔧 技术方法**

采用了经典规则扰动（EDA、AEDA）、检索插值（EmbSMOTE）、回译、以及多种LLM生成技术（LLM‑Paraphrase、LCG、AugGPT、CoTAM、LLM2LLM、CIEGAD、VoidGen），使用 Llama‑3.1‑8B 与 Qwen‑3‑8B 进行实验。

**📊 数据集**

使用了 SST‑2、AG News、Emo、TREC、GoEmotions‑13、DBpedia 和 GoEmotions‑28 七个公开文本分类数据集。

**📈 对比分析**

通过宏F1、Welch t检验、分布度量等多维度比较，结果表明在不平衡多分类任务上 EmbSMOTE 及其检索增强方法始终优于或与所有 LLM 基础方法相当，差距随不平衡度增加而扩大；在平衡任务上差异可忽略。

**⚠️ 局限性**

局限性包括仅使用 8B 级 LLM（未探测更大模型）、仅限英语单语文本、固定 DistilBERT 分类器、检索与生成比例固定、VoidGen 在极端稀疏类别下出现数值失败、统计检验对分布假设敏感。

---

## 65. Spec-Driven Hardware Evolution via Executable Contract Refinement and Proof-Guided RTL Update

**arXiv ID:** 2608.12684 | [PDF](https://arxiv.org/pdf/2608.12684v1)

**作者:** Shibo Zhao `[一作]` (Southeast University), Min Li `[通讯]` (Southeast University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于可执行合同的硬件版本演进框架，将功能需求转化为可审计的合同，并通过自动化的指定、规划、实现与验证循环实现从已验证 RTL 到下一版本的演进。

**💡 创新点**

将硬件演进视为合同驱动的过程，利用可执行合同、突变式语义探测与基于证明的 RTL 更新循环，解决传统 prompt‑to‑RTL 仅关注代码生成而忽视版本演进的问题。

**🔧 技术方法**

结合 LLM 驱动的代码生成、mutation‑based semantic probing、formal verification（等价性与 BMC）、脚本化验证协调、多代理任务隔离、层次感知自底向上修复，以及执行参考模型与约束检查等技术。

**📊 数据集**

在 TPU datapath 模块 dot_core 上构建了从 INT8/FP4/FP8 到 TF32 的版本演进案例，涉及约 200‑700 行 RTL 与约 200‑300 行可执行参考，并为不同子模块生成对应的合同与测试。

**📈 对比分析**

在同一框架下对 GPT‑5.4、Claude Opus 4.6、Claude Sonnet 4.6 三大 LLM 进行统一实验，测量收敛次数、迭代次数、token 消耗、成本和运行时间；GPT‑5.4 在 2 轮迭代、91 分钟、7.6M token、$15.65 的成本下成功通过验证，且相较于固定 LangGraph 管线具有更少迭代但更高 token/时长。

**⚠️ 局限性**

仅实现功能收敛，未对 PPA 进行优化；合同构造仍需人工审查；实验范围仅限单一 TPU datapath，难以直接推广至更大规模或多 IP 的真实工业项目。

---

## 66. Correct Is Not Governed: Provenance Integrity in Agentic Workflows

**arXiv ID:** 2608.12761 | [PDF](https://arxiv.org/pdf/2608.12761v1)

**作者:** Jesus Salas `[一作]` `[通讯]` (Independent Researcher), Jesus Salas (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一个确定性因果状态层，记录权威与事实依赖、验证执行证据并在变更时选择性失效，以实现可审计的治理执行；

**💡 创新点**

创新点在于将决策、执行与变更的三种溯源整合为单一可 append‑only 的生命周期，明确区分任务正确性与治理完整性，并通过实验验证两者不等价；

**🔧 技术方法**

采用了事件日志、依赖图、接收器验证、递归重试与元数据可判定拒绝等技术，构建了可在任何模型上叠加的治理内核；

**📊 数据集**

使用自制的多场景/多种语义数据集（5个公开情景、30个合成包、12个完整性检验案例等）以及 Phi-4 语言模型生成的任务计划；

**📈 对比分析**

通过对齐控制与治理两条路径的相同业务结果，对比同一任务下的行动准确率、引用正确性、回滚粒度等指标；实验表明治理路径在引用完整性和回滚粒度上优于直接 RAG，但两者在业务正确率上基本一致；

**⚠️ 局限性**

局限性包括：数据均为人工合成、缺乏真实企业任务验证、对模型能力改进敏感、元数据可判定规则不保证语义完整性，且未展示对大规模系统的性能与可扩展性。

---

## 67. Strategy-Oriented Feedback for Fostering Systematic Problem-Solving in Machine Learning Education

**arXiv ID:** 2608.12362 | [PDF](https://arxiv.org/pdf/2608.12362v1)

**作者:** Clemens Witt `[一作]` (Dresden University of Technology), Mareen Grillenberger `[通讯]` (Dresden University of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并评估了一套基于策略的自适应反馈系统，集成在决策树构建的数字游戏Match the Monkeys中，帮助学生形成系统化问题解决策略。

**💡 创新点**

首次在机器学习教育中将多模态策略识别与LLM生成的即时、个性化反馈结合，聚焦过程层面的策略改进而非仅结果评估。

**🔧 技术方法**

使用多模态机器学习（视频+树结构日志）进行策略分类，结合GPT‑4.1生成反馈，采用滑动窗口上下文管理；还利用统计分析（Krippendorff's α、Mann‑Whitney U、Spearman、转移矩阵）评估效果。

**📊 数据集**

共收集230名学生（约55小时游戏录像和日志）的游戏数据，最终分析205个有效会话，作为实验数据集。

**📈 对比分析**

将带反馈版本与无反馈基线对比，使用非参数统计和转移概率差异检验，结果显示结构化策略比例显著提升(+6.43%)，探索性策略比例下降，转移矩阵显示从探索到结构化的切换增加，但游戏成功率未显著提升。

**⚠️ 局限性**

反馈虽促进短期策略转变，却缺乏对认知负荷、元认知控制和长期巩固的支持；后续反馈注意力下降，导致结构化策略未能持续保持。

---

## 68. HiRoute: Hierarchical Routed Prompt Tuning for Safety Alignment of Large Language Models

**arXiv ID:** 2608.12821 | [PDF](https://arxiv.org/pdf/2608.12821v1)

**作者:** Fangzhou Chen `[一作]` (Beihang University), Xingxing Wei `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HiRoute，一种输入适配的层级提示调优框架，在冻结大型语言模型的前提下，实现安全对齐。它通过共享的粗粒度提示与路由的细粒度专家提示相结合，并采用安全门控仅对危险输入激活安全提示，从而避免对无害输入的过度拒绝。

**💡 创新点**

创新点包括：① 将安全对齐分为类别无关的粗提示和类别特定的细提示，利用层级路由动态混合；② 两阶段训练，将风险识别与提示优化分离，防止生成目标干扰风险边界；③ 引入输入级安全门控，仅对风险输入使用安全提示，降低过度拒绝。

**🔧 技术方法**

使用技术包括：连续提示调优（Prompt Tuning）、层级路由器（Transformer 编码器 + 两个分类头）、直接偏好优化（DPO）与梯度遮蔽交替更新、风险多标签分类与安全门控阈值控制。

**📊 数据集**

使用数据集：粗粒度安全/风险标签来自 WildGuardMix；细粒度风险标签（网络犯罪、经济犯罪、隐私侵犯、暴力）来自 PKU‑SafeRLHF；提示训练使用偏好三元组；评估安全性采用 StrongReject、AdvBench、JailbreakBench；通用任务评估使用 GSM8K、MT‑Bench、TruthfulQA；过度拒绝率使用 XSTest。

**📈 对比分析**

与基线（Base、RPO、DRO、ACD）在 Mistral‑7B、Vicuna‑7B、Zephyr‑7B 三个模型上对比，HiRoute 在安全率上平均提升 2–3%（安全率达 93–97%），且保持或提高安全响应有用性；在通用任务上保留大部分数学推理、开放式回答质量和真值性；过度拒绝率显著下降（Mistral 2.5% 对比 RPO 40.5%）。

**⚠️ 局限性**

局限性：依赖层级路由器的准确性，误判会导致安全提示失效；实验主要集中在单轮文本，未覆盖多轮交互或多模态场景；开放集风险识别和不确定性处理仍需进一步研究。

---

## 69. New Terms, New Toxicity: Consensus-based Chinese Neologism Toxicity Detection via Search-Augmented LLMs

**arXiv ID:** 2608.12361 | [PDF](https://arxiv.org/pdf/2608.12361v1)

**作者:** Shiyao Cui `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 16303 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了毒性新词汇表并提出基于检索的SeTox框架以检测隐式毒性新词

**💡 创新点**

将检索工具与LLM结合，在检测新词毒性时实时获取公共语境，克服传统模型对新词的滞后性

**🔧 技术方法**

检索增强LLM（SeTox）、工具调用训练、对抗性数据生成及大规模语料检索

**📊 数据集**

自建的974条毒性新词词典、ToxiCN、CHSD、公开微博评论

**📈 对比分析**

与多种API（Perspective、百度、腾讯、阿里云）及多大模型（GPT‑4o、Gemini、Qwen3‑Max、Deepseek等）对比，SeTox‑7B在毒性新词检测上达94%+，显著优于其他模型和API

**⚠️ 局限性**

仅支持单轮检索、单模态（文本）、仅针对中文，无法处理多轮推理或多模态毒性

---

## 70. Hate speech toward migrants on a citizen reporting platform concentrates in neighborhoods undergoing demographic change

**arXiv ID:** 2608.12581 | [PDF](https://arxiv.org/pdf/2608.12581v1)

**作者:** Eduardo Graells-Garrido `[一作]` (Universidad de Chile), Carmen Cabrera `[通讯]` (University of Liverpool)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用城市居民在SOSAFE平台上的举报文本和地理位置信息，研究迁徙与仇恨言论在圣地亚哥的空间分布关系。

**💡 创新点**

首次将参与式安全平台的数字边界概念与迁移动态相结合，揭示仇恨言论聚集在最近迁入人口比例高的社区。

**🔧 技术方法**

使用精调的RoBERTa（RoBERTuito）语言模型进行西班牙语仇恨言论检测，并配合LISA空间聚类与逻辑回归/负二项回归分析。

**📊 数据集**

数据来自2024年圣地亚哥553,400条SOSAFE举报、2024年智利人口普查的住户微观数据以及教育水平与迁入时间的统计。

**📈 对比分析**

通过比较模型系数与对照组（不提及移民的举报）以及多种稳健性检验，发现迁入比例与仇恨言论之间的关联显著，模型伪R²约0.14，接近0.5的空间聚类显著率。

**⚠️ 局限性**

局限包括SOSAFE用户非代表性、平台可能预先过滤内容、仇恨检测器在移民提及上误报较高、关键词匹配无法捕捉隐晦提法，以及缺乏用户身份信息导致无法识别单个极端行为者。

---

## 71. FastThaiG2P: Lightning-fast Thai Grapheme-to-phoneme Conversion for Voice Agent Pipelines

**arXiv ID:** 2608.12814 | [PDF](https://arxiv.org/pdf/2608.12814v1)

**作者:** Charin Polpanumas `[一作]` `[通讯]` (AWS), Charin Polpanumas (AWS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了FastThaiG2P库，实现子毫秒级泰语拼写转音素的文本到语音前端，并与Kokoro-82M StyleTTS 2模型结合完成CPU端泰语TTS系统。

**💡 创新点**

创新点：①构建62,112词IPA字典并结合词形规范化；②优化正则缓存并提前加载规则，实现15×延迟提升；③提供完整的OOV规则回退与音素转写；④将IPA转Kokoro映射解决五声调差异。

**🔧 技术方法**

技术：PyThaiNLP词形分词、规则基词典查找、正则表达式缓存、Claude Opus 4.6 LLM生成IPA、Kokoro-82M StyleTTS 2架构、ONNX推理、音素到音频的深度学习模型。

**📊 数据集**

数据集：27,242句合成对话语料做延迟基准，Som‑TTS 20 小时单说话人音频与文本做训练，Wiktionary、LLM生成词表用于字典构建。

**📈 对比分析**

与TLTK、Epitran、thai-g2p、CharsiuG2P等现有工具对比，FastThaiG2P每句平均延迟0.15 ms（相比TLTK 2 ms下降≈85%），OOV率0.5%。与Kokoro-82M在CPU上实现0.25 RTF（4×实时），模型体积330 MB。

**⚠️ 局限性**

局限：字典覆盖不足非标准词汇、方言、外来词；OOV回退规则对异常词读音不准确；未进行MOS评估；对多语言混合或代码切换支持有限。

---

## 72. Is this Citation on Point?

**arXiv ID:** 2608.12571 | [PDF](https://arxiv.org/pdf/2608.12571v1)

**作者:** Apurv Verma `[一作]` `[通讯]` (Bloomberg), Apurv Verma (Bloomberg)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究法律引用支持验证任务，评估LLM在检测引用是否真正支持主张方面的能力。

**💡 创新点**

创新点在于引入了受控污染的三难度层级（易/中/难）并将实测误差分为案例替换与页面错误，阐明LLM在细粒度支持检验上的不足。

**🔧 技术方法**

采用了提示式LLM分类、链式推理、页面根基提示等技术，对比了四大模型家族的多种配置。

**📊 数据集**

使用了公开的CLERC（法院意见）和BriefMe（法律简报）两个法律语料库中的引用实例。

**📈 对比分析**

对比实验显示，模型在易级别召回接近100%但在难级别仅达37–83%；更大模型和更强推理可提升10–18个百分点，页面提示可再增10–30个百分点但伴随假阳性上升。

**⚠️ 局限性**

局限性包括仅关注实质性引用、二值化支持判定、受控污染不涵盖所有误差类型、未评估实际审查工作成本以及对不同司法体系的适用性不足。

---

## 73. StreamReason-Bench: Can Large Language Models Reason about Event-Time Stream-Processing Semantics?

**arXiv ID:** 2608.12348 | [PDF](https://arxiv.org/pdf/2608.12348v1)

**作者:** Zhuoxi Wang `[一作]` `[通讯]` (Independent Researcher), Zhuoxi Wang (Independent Researcher)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 StreamReason‑Bench benchmark，用于评估大型语言模型在事件时间流处理中的推理能力，省去实际执行引擎的需求。

**💡 创新点**

创新点在于：①提供了基于 Dataflow 模型的确定性参考执行器；②生成可扩展的合成数据集；③通过直接与链式思考两种提示协议，揭示事件时间推理是当前 LLM 的主要难点。

**🔧 技术方法**

使用技术包括：Dataflow 语义实现、合成事件流生成器、GPT‑4o/Claude‑Sonnet/Gemini 等 LLM 的直接与链式思考提示，以及基于 exact‑match 与 row‑F1 的评估。

**📊 数据集**

使用的数据集为 600 条自定义生成的项目，覆盖抖动、跳跃、会话和处理时间窗口，并按难度分层（易/中/难）。

**📈 对比分析**

通过 exact‑match 与 row‑F1 两种指标对比模型；直接提示下准确率 ≤34%，链式思考可提升约 50%（最强模型 Sonnet 达 85% exact‑match）；处理时间窗口相对简单，所有模型性能显著更高；难度越高，准确率越低。

**⚠️ 局限性**

局限性包括：仅关注单变量、简单聚合（sum/count/max）和四种窗口类型；数据为合成，未与真实流引擎（Beam/Flink）彻底交叉验证；未覆盖多流/区间连接等复杂场景；链式思考并非万能，模型对提示长度和格式敏感。

---

## 74. HIMEC: Directional Change Representation and Fixed-Interface Decoding for Remote Sensing Image Change Captioning

**arXiv ID:** 2608.12502 | [PDF](https://arxiv.org/pdf/2608.12502v1)

**作者:** Aysha Ashraf `[一作]` (University of Electronic Science and Technology of China), Zhenming Peng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HIMEC框架，使用方向化差异表示和固定接口解码来生成遥感影像变化描述

**💡 创新点**

将双时相差异按正负方向分离并融合，再通过学习的查询向量形成变更查询记忆，且采用固定零输入保持训练与推理一致性，解决解码器条件不一致问题

**🔧 技术方法**

Swin‑Transformer 编码器、SE块、深度可分离残差融合、学习查询注意力、固定零接口门控、SCST 以及辅助短语监督

**📊 数据集**

在LEVIR‑CC、SECOND‑CC和DUBAI‑CC三个遥感变化描述基准上进行训练与评估

**📈 对比分析**

与直接融合特征记忆、传统RSICC方法以及多种对照实验进行比较，LEVIR‑CC上CIDEr达142.81±0.60，显著高于先前最高140.23，且在SECOND‑CC上取得75.67-76.99 CIDEr

**⚠️ 局限性**

对局部到全局解码器的分析仅在诊断级别，未在大规模多变更场景进行人类评估，缺乏空间定位解释，且在DUBAI‑CC未能完全恢复匹配条件的优势

---

## 75. What Drives LLM Self-Reflection? A Controlled Ablation of Uncertainty Routing in Armed Conflict Forecasting

**arXiv ID:** 2608.12322 | [PDF](https://arxiv.org/pdf/2608.12322v1)

**作者:** Poli Nemkova `[一作]` (University of North Texas), Haeshitha Indukuri `[通讯]` (University of North Texas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过六条件消融实验系统地拆解LLM自我反思的四个组成部分（证据暴露、诊断脚手架、词汇词典与动作路由），验证哪一部分真正驱动性能提升。

**💡 创新点**

创新点在于揭示结构化诊断和词汇本身并未带来收益，真正提升来自于基于不确定性类型的 typed action routing。

**🔧 技术方法**

使用技术包括 Llama‑3.3‑70B 与 GPT‑4o 大模型、七型不确定性分类与确定性控制策略、以及多跑多数投票与单跑评估方法。

**📊 数据集**

采用的数据集为 310 例真实武装冲突升级案例（苏丹、埃塞俄比亚、索马里、缅甸、乌克兰）以及 12 个未见国家的泛化测试集。

**📈 对比分析**

评估方法为六个条件的多跑多数投票 F1，结果显示 typed routing 将 F1 提升至 0.379，较基线提升约 +0.101，显示显著性能提升。

**⚠️ 局限性**

限制在于仅在武装冲突领域验证，动作路由对未见冲突的泛化有限，且未结合检索或工具实现真正的证据补全。

---

## 76. Multi-Agent Scheduling with LLM-Assisted Contract Net Negotiation for Stream Processing in Mobile Edge Computing

**arXiv ID:** 2608.12371 | [PDF](https://arxiv.org/pdf/2608.12371v1)

**作者:** Sabeur Lajili `[一作]` (University of Sousse), Zaki Brahmi `[通讯]` (University of Sousse)

**通讯引用:** 447 | [OpenAlex ID](https://openalex.org/A5037759917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了MAS-DecStream框架，通过LLM辅助的多轮合同网协议（LLM-MR-CNP）实现移动边缘计算环境中流处理任务的去中心化调度；

**💡 创新点**

创新点在于将传统Contract Net Protocol扩展为支持语义CFP生成、逐步上下文披露、多轮提案修订、记忆化协商与确定性验证的LLM-MR-CNP；

**🔧 技术方法**

采用的大型语言模型（如gpt‑4o‑mini、LLM-OSS:20B、GLM‑5.2等）、LangGraph协作框架、离线预测模型与确定性工具相结合；

**📊 数据集**

使用从Alibaba ASI Trace 2026生成的1,000条工作负载记录，并为每条记录合成5个边缘集群快照，形成5,000个任务-集群组合；

**📈 对比分析**

实验通过与单轮CNP（RB‑SR‑CNP）、多轮无LLM（RB‑MR‑CNP）以及多轮LLM（MAS‑DecStream）三组对比，评估了延迟违约率、资源超额、冲突解决率、全局效用及协作成本；结果显示多轮协商能将违约率从53%降至3%，消除资源超额，冲突解决率提升至0.91，且LLM辅助进一步提升效用约22%；

**⚠️ 局限性**

局限性包括：LLM与多轮CNP的因果贡献难以单独测量；仅使用了从AI工作负载追踪衍生的数据，未在真实边缘平台上验证；LLM推理耗时和令牌成本高，适用性需根据场景选择；实验规模受限于20个代理与12个并发请求，无法覆盖更大规模网络及网络延迟、故障恢复等真实系统因素。

---

## 77. Surprise2Refine: Axis-Centered Exploration-To-Refinement for Agent-Assisted Creative Scaffolding

**arXiv ID:** 2608.12605 | [PDF](https://arxiv.org/pdf/2608.12605v1)

**作者:** Yuzhe You `[一作]` (University of Waterloo), Tongyu Zhou `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于轴向的设计空间工作流 Surprise2Refine，用于支持创意设计中的探索与精炼。

**💡 创新点**

创新点在于：①通过可配置的两维轴将设计空间动态划分为网格，既支持广泛探索又可逐步收敛；②设计多种轴向交互（缩放、锚定、分解）实现结构化的“惊喜”与精细调整；③后端代理管道根据用户交互自适应调整生成策略，实现从散漫探索到聚焦精炼的无缝过渡。

**🔧 技术方法**

技术包括：多模态 VLM（Gemini‑2.5‑Flash‑Lite）用于提取 moodboard 语义并生成轴向尺度；Prompt 模板与可扩展的 3‑点尺度策略；生成引擎（大模型）配合参数化缩放、锚点插值与分解重组；前端交互以 n×n 网格视图呈现轴向结构；自然衰减与反馈加权机制实现身份与偏好自适应。

**📊 数据集**

数据集主要来自用户上传的 moodboard（图片、草图、文字），无固定公开数据集；实验中使用了 14 位专业设计师提供的多样化参考材料，构成探索与精炼的输入集合。

**📈 对比分析**

在 14 名设计师的对照实验中，Surprise2Refine 与基线系统（无轴向交互）进行 20 分钟的海报设计任务；结果显示：CSI 得分显著提升（69.71 vs 55.79，p=0.0085）；在感知控制、创意支持、结构化路径等多项 Likert 量表上均优于基线；多样性指标 LPIPS 亦有显著提高；但在最终作品满意度上差异不显著。

**⚠️ 局限性**

局限性包括：①只在海报设计任务上验证，缺乏对其他创意领域的泛化评估；②分解模块仅提供抽象概念重组，缺乏精确对象级分割；③对比基线为自建系统，未涵盖商业工具；④实验样本量虽达 14 人，但仍受限于设计者专业背景；⑤轴向交互依赖用户手动调整，未实现完全自动化；⑥潜在认知负担问题（如过多探索导致决策疲劳）未完全解决。

---

## 78. Exploring Oversmoothing with Householder Matrices

**arXiv ID:** 2608.12514 | [PDF](https://arxiv.org/pdf/2608.12514v1)

**作者:** Bhaskar Karol `[一作]` `[通讯]`, Bhaskar Karol

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Householder Graph Neural Network (HouseGNN)，通过邻居信息估计反射方向，使用 Householder 反射和 GroupSort 进行节点状态更新，从而在每一层保持欧氏范数，抑制传统 GNN 的 oversmoothing。

**💡 创新点**

创新点在于：①将邻居聚合仅用于确定反射方向，而不直接作为隐藏状态；②利用局部 Householder 反射保持正交性与范数；③理论证明层级保留范数、仅通过局部正交算子差异改变距离，并给出可避免经典扩散式 oversmoothing 的分析；④通过实验验证深层稳定性。

**🔧 技术方法**

采用的技术包括 Householder 矩阵、GroupSort 非线性激活、行标准化聚合（P）、正交权重约束、理论分析（范数保持、距离变化、Dirichlet 能量与有效秩）以及深度实验评估。

**📊 数据集**

使用的标准节点分类图数据集包括 Cora、CiteSeer、PubMed、Texas、Wisconsin、Cornell 等。

**📈 对比分析**

与 GCN、GAT、BatchNorm‑GCN、PairNorm‑GCN、Residual‑GCN 等基线在 2~64 层深度下进行对比；HouseGNN 在深层保持约 75%–77% 的准确率，GCN 在深层显著下降；在 heterophilic 图（Texas、Wisconsin、Cornell）中也保持相对稳定；在 PubMed 也表现出优于大多数基线的深层性能。

**⚠️ 局限性**

局限性：在极深层（128 层）使用 GroupSort 时准确率骤降，可能进入非判别但不收敛状态；在 heterophilic 图上的表现仍有提升空间；超深层（>128 层）仍面临挑战；未探索与残差连接等其他技术的结合。

---

## 79. InvisIto: Weaving Unobtrusive Infrared Markers for Ubiquitous Textile Interaction

**arXiv ID:** 2608.12580 | [PDF](https://arxiv.org/pdf/2608.12580v1)

**作者:** Hsuanling Lee `[一作]` (University of Texas at Dallas), Koya Narumi `[通讯]` (Keio University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出并实现了一种利用近红外吸收纱线将隐形、可机读的红外标记直接编织进织物中的方法，并提供完整的设计工具、五种掩饰策略、可在手工、Jacquard、工业生产中使用的工艺流程及基于NIR摄像的检测与跟踪管线。

**💡 创新点**

创新点在于：①将近红外可吸收纱线与织物双层织法相结合，实现视觉上几乎不可见、机器可读的标记；②推出面向设计师的在线设计工具，支持QR码与ArUco标记的嵌入与可视化；③开发五种可实现可行、可工业化的掩饰技术；④提供端到端的工艺与算法，支持从手工编织到工业生产的跨尺度部署。

**🔧 技术方法**

主要技术包括：近红外吸收聚酯纱线（SOLAMENT、Teijin Heat Energy）、双层织物技术、AdaCAD设计工具、可视化模拟、NIR相机（Omron Sentech STC‑SBS34U3V‑SWIRU）与光源、二维码与ArUco识别（Dynamsoft、OpenCV）、基于区域投影的变形跟踪算法。

**📊 数据集**

实验使用自制织物样本：QR码织物、ArUco矩阵织物；通过1,000次解码实验、5,000帧摄像实验、17名受试者的可视化盲测以及多次洗涤循环等数据进行评估。

**📈 对比分析**

比较方法：对五种掩饰技术进行用户主观评分与NIR对比度分析；对QR码进行分辨率、纠错级别、织法结构和洗涤耐久度的多维度测试；对ArUco矩阵进行帧率（99%阈值≈125 fps）和跟踪精度（速度越快误差越大但均在<10 mm）。总体性能：QR码解码成功率>90%（在3.5 yarn/模块时），ArUco跟踪率>90%，高达150 fps实时处理，洗涤20次后成功率仍保持在87%。

**⚠️ 局限性**

局限性：①标记在细节处仍可被放大观察到，未实现完全隐蔽；②分辨率受纱线粗细和织密度限制，信息容量有限；③目前仅支持聚酯纱线，难以推广至棉、丝等天然纤维；④QR/ArUco格式对织物几何有严格假设，导致在严重拉伸/皱折时检测失效；⑤对遮挡和变形的鲁棒性仍不足，需要进一步设计织物原生编码与冗余机制。

---

## 80. What Do We Mean When We Talk About Infographics?

**arXiv ID:** 2608.12370 | [PDF](https://arxiv.org/pdf/2608.12370v1)

**作者:** Xiaoyu Liu `[一作]` (University of Maryland), Zhicheng Liu `[通讯]` (University of Maryland)

**通讯引用:** 3693 | [OpenAlex ID](https://openalex.org/A5100745766)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过系统文献综述与实务者问卷，梳理并对比了信息图的多种概念化方式，识别出文本角色、与数据可视化的关系以及是否包含统计图等关键维度；

**💡 创新点**

首次将信息图概念的争议归纳为若干维度并揭示其认知来源，提出在研究中显式说明概念边界的必要性，并为构建面向组件的概念框架奠定依据；

**🔧 技术方法**

采用 PRISMA 体系进行文献筛选，基于句子级提取进行归纳编码，使用 Gwet 的 AC1 评估编码一致性，并通过问卷统计与可视化展示概念关系；

**📊 数据集**

共收集 184 篇学术论文，提炼 487 条句子；问卷得到 44 名实务者提供 44 条定义与 440 条概念关系投票；合计 531 条语料构成分析数据集；

**📈 对比分析**

通过对比 26 个相关概念与信息图的集合关系（包含上集、下集、等价、交集、无关、未确定），展示不同群体对概念边界的共识与分歧；未涉及系统性能对比，重点在概念一致性与差异分析；

**⚠️ 局限性**

样本量有限（44 位实务者），可能存在自选偏倚；文献检索主要限于英文数据库，未覆盖所有学科；归纳编码受人工主观影响；未对提议框架进行实证验证。

---

## 81. ARIES-Mission2: A Zero-Shot Vision-Language-Action Framework for Fast Large-Scale Aerial Mission Generation

**arXiv ID:** 2608.12763 | [PDF](https://arxiv.org/pdf/2608.12763v1)

**作者:** Junhao Wei `[一作]` (Macao Polytechnic University), Xu Yang `[通讯]` (Macao Polytechnic University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ARIES-Mission2 框架，实现无人机任务生成的零样本视觉-语言-行动系统，分离视觉语义感知与路径优化。

**💡 创新点**

创新点在于将视觉语言模型与多种粒子群优化（PSO、GPSO、IPSO）解耦，通过算法竞争获得全局最优 TSP 轨迹，从而显著降低飞行距离和计算时间。

**🔧 技术方法**

使用 DeepSeek‑V3 进行任务解析，Molmo‑7B 进行零样本目标定位，线性地理插值将像素映射为 GPS；后端采用 PSO、GPSO、IPSO metaheuristic 优化 TSP；生成可直接上传的 MAVLink 任务文件。

**📊 数据集**

使用 UAV‑VLPA‑nano‑30 基准数据集，包含 30 个低空航拍任务、对应自然语言指令和卫星图像。

**📈 对比分析**

与未优化 VLA 基线和人工专家规划对比，ARIES‑Mission2 总飞行距离 62.43 km（比 79.66 km 减少 21.6%），平均 2.08 km（比 69 km 减少 9.5%），每任务平均完成时间 19.18 s，约比人工快 3.6 倍。

**⚠️ 局限性**

限制在于前端 VLM 推理仍是主要计算瓶颈，随着目标数增长计算量显著增加；缺少 3D 地形建模和动态障碍物避让功能，难以在更复杂环境中直接应用。

---

## 82. The Affordance is the Message: Creative Media as Complex Systems

**arXiv ID:** 2608.12349 | [PDF](https://arxiv.org/pdf/2608.12349v1)

**作者:** Ane Espeseth `[一作]` (University of Oslo), Elias Najarro `[通讯]` (IT University of Copenhagen)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5032099222)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过复杂系统视角对计算创意媒体进行形式化，提出九个系统级属性并关联媒体可供性。

**💡 创新点**

创新点在于将复杂系统的概念（如临界性、吸引子多样性、路径依赖等）引入计算创意媒体框架，提供子系统层与宏观层的双向映射。

**🔧 技术方法**

采用复杂系统理论概念和定量分析工具（熵、相关函数等）对媒体属性进行表征，并以设计可供性为基础。

**📊 数据集**

文中主要以若干实例媒体（如Reddit画布实验、Minecraft等）作为示例；未使用标准公开数据集。

**📈 对比分析**

通过可供性清单和复杂系统度量对媒体进行对比评估，展示不同属性组合对创作产出的影响，但未给出数值性能指标。

**⚠️ 局限性**

限制在于缺乏实证验证与可量化指标，理论框架尚未在大规模实验中检验，且对复杂系统度量的选择仍有主观性。

---

## 83. Prof-K: Probabilistic One-Pass Filtering for Efficient Top-k Selection

**arXiv ID:** 2608.12573 | [PDF](https://arxiv.org/pdf/2608.12573v1)

**作者:** Tadeusz Dziarmaga `[一作]` (Jagiellonian University), Marcin Mazur `[通讯]` (Jagiellonian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种单次扫描、分布无关的概率性 Top‑k 选择算法 Prof‑K，用随机采样估计阈值，然后仅保留超过阈值的元素进入小缓冲区，再在缓冲区上做精确 Top‑k，保证在给定置信度下恢复全部真正的 Top‑k。

**💡 创新点**

创新点在于：①利用无偏采样得到的阈值仅依赖于秩而非值，因而提供分布无关的概率正确性保证；②设计了显式的样本量与缓冲区容量取值，理论上样本量约为 (kN)^{1/3}；③给出了完整的失败概率拆分（召回失效与缓冲区溢出），并给出相应的参数选择公式。

**🔧 技术方法**

技术手段包括：随机不放回采样、负超几何分布分析、阈值估计、单次流式过滤、缓冲区精确 Top‑k、Triton/GPU 并行实现；同时通过 Gaussian 近似简化参数设计与理论证明。

**📊 数据集**

实验使用的基准数据集有：均匀分布、标准正态分布和重尾 Pareto 分布的合成张量；实际应用中则在 BatchTopK Sparse Autoencoder（SAE）训练中使用 OpenWebText、GPT‑2 Small 词典大小（d_dict=12288）、批量 B=4096 的激活张量。

**📈 对比分析**

与 PyTorch native topk 及 RadiK 进行对比，Prof‑K 在大 N（高达 2^30）且 k 较小的情形下平均实现 1.5–10 倍速度提升；内存占用低于基线；在 SAE 训练中将 top‑k 内核时间从 2.56 ms 降至 1.13 ms，整体训练时间缩短约 4.25%。

**⚠️ 局限性**

局限性包括：①在样本量过小或 k 与 N 极不平衡时，Gaussian 近似失效，需退回到精确 top‑k；②由于概率性，极端罕见情况下可能漏掉部分真正的 Top‑k；③在小规模或 k 接近 N 的场景下优势不明显；④实现依赖 GPU 并行与 Triton，CPU 上性能未充分验证。

---

## 84. The energetic cost of mitigating AI attacks in cellular networks

**arXiv ID:** 2608.12431 | [PDF](https://arxiv.org/pdf/2608.12431v1)

**作者:** Adrián Losada `[一作]` (Laude Technology Company S.L.), Raquel Barco `[通讯]` (Universidad de Málaga)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在O-RAN环境下评估数据污染攻击的能耗，并测量Deep Partition Aggregation（DPA）防御技术对模型准确率、误分类率与能耗的影响。

**💡 创新点**

首次将数据污染防御的能耗量化，为网络安全与节能平衡提供实证依据。

**🔧 技术方法**

采用ResNet-32模型、CIFAR-10数据集、DPA防御算法，并在PC上利用Scaphandre采集功耗数据。

**📊 数据集**

使用CIFAR-10数据集（60000张图像，10类）。

**📈 对比分析**

通过在不同污染率（0.05–0.40）和分区数（k=15、500）下比较准确率、目标误分类率和能耗，发现k=15时准确率≈80%、误分类率<1%，且能耗比k=500低200Wh。

**⚠️ 局限性**

仅针对单一模型和防御方案，在PC实验环境下评估，未覆盖时间延迟、其他攻击类型或大规模分布式部署的能耗与安全性能。

---

## 85. Mimicry without understanding: the origins of decision bias in large language models

**arXiv ID:** 2608.12339 | [PDF](https://arxiv.org/pdf/2608.12339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 86. New optimal linear codes over $\ZZ_4$

**arXiv ID:** 2608.12414 | [PDF](https://arxiv.org/pdf/2608.12414v1)

**作者:** Hopein Christofen Tang `[一作]` (Institut Teknologi Bandung), Djoko Suprijanto `[通讯]` (Institut Teknologi Bandung)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了多种新型构造方法，用于生成整数模 4 上的线性码，并通过这些方法构造了大量最优（或近最优）码，取得了许多此前未知的最佳 Lee 距离。

**💡 创新点**

创新点在于：① 引入了基于所有非零列向量的递归生成矩阵 G^(k1,k2)，从而得到一族全 Lee 权相同且满足 Plotkin‑type 上界的线性码；② 通过组合已有最优码与 G^(k1,k2) 的拼接，递归地扩展最优码长度；③ 设计了新的两种码生成变换（G'_1 与 G'_2），实现长度缩放同时保持或提升 Lee 距离；④ 将上述构造与 Gray 映射结合，提供了二元 Plotkin‑optimal 码的途径。

**🔧 技术方法**

使用的技术主要包括：
- Lee 重量与距离定义、标准形式生成矩阵、Gray 映射；
- 组合数学与矩阵递归构造；
- 代数编码理论中的 Plotkin‑type 上界证明；
- 代码等价性与置换、符号更改等等价变换；
- 对已知码数据库（PLDB、LDB 等）的性能对比。

**📊 数据集**

主要利用公开的 Z4 码数据库（如 PLDB、LDB 等）作为对照基准，并将新构造得到的码的参数与数据库中已知的最佳 Lee 距离进行对比；同时也参考了先前文献中给出的经典码表。

**📈 对比分析**

对比方法：在表格中列出给定长度 n 的 Plotkin‑type 预测上界、数据库中的最佳 Lee 距离、以及作者构造得到的最优/近最优码的 Lee 距离。结果显示：
- 对于 k1=2,k2=0，所有新码的 Lee 距离至少与上界相差 0 或 1；
- 对于 k1=3,k2=0，差距最多 2；
- 在多数长度下，构造的码与已知最佳码相比提升了 1~3 点；
- 所有新码均满足或超过 Plotkin‑type 上界，属于 Plotkin‑optimal 或近最优。

**⚠️ 局限性**

局限性与不足：
- 证明与构造主要适用于模 4 的自由线性码，k2>0 的情况仍有限；
- 对于某些长度（如 n≡1,3,5 (mod 15) 等）无法构造 Plotkin‑optimal 码，需借助更细致的二元上界；
- 递归构造在长度增长时会导致生成矩阵规模膨胀，实际实现与存储受限；
- 研究未给出一般的存在性判定，只给出特定 k1,k2 的构造；
- 对非自由或更一般环 Z_{2^r} 的推广仍在进行中，尚无完整理论。

---

## 87. Finding the Needle in a Haystack: Test-Time Analog Circuit Representation Adaptation for Bayesian Optimization

**arXiv ID:** 2608.12687 | [PDF](https://arxiv.org/pdf/2608.12687v1)

**作者:** Fin Amin `[一作]` (North Carolina State University), Paul D. Franzon `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Test-Time Analog Representation Adaptation for Bayesian Optimization（TTARO）框架，在线更新预训练电路嵌入表示并与高斯过程核一起重新训练，使贝叶斯优化在有限候选库中更高效地搜索电路拓扑。

**💡 创新点**

创新点在于将表示学习与贝叶斯优化耦合，允许在搜索过程中动态调整特征映射，使GP核几何与目标FoM更对齐，从而显著提升样本效率；首次在有限电路库上实现在线深度核学习。

**🔧 技术方法**

使用的核心技术包括：深度特征映射（两层MLP）、高斯过程回归（线性核与RBF核）、期望改进、Thompson采样、UCB等采集函数、在线重训练与UMAP可视化、结构化核插值/稀疏变分等潜在加速技术。

**📊 数据集**

实验基于公开的Open Circuit Benchmark，包含两套运算放大器拓扑库 Ckt-Bench-101（10k 方案）和 Ckt-Bench-301（50k 方案），并使用多种编码器（CktGNN、DAGNN、D-VAE、D-VAE-GCN、WL）产生初始嵌入。

**📈 对比分析**

与固定嵌入GP、仅在初始样本上学习的DKL以及oracle全标注的理想表示进行比较，评估指标为Regret AUC和最终FoM。TTARO在 40 种配置中有 37 种得到 Regret AUC 降低（平均 19.6% / 12.2%），FoM 最终值平均提升约 20%（相较于 GP），并在大多数设置显著优于 DKL，尤其在较大、异质的搜索空间中。

**⚠️ 局限性**

主要限制是计算开销：每次评估后需重新训练特征映射和 GP，导致更新成本较高；对候选库规模极大或仿真耗时极短的场景可能不具备优势。

---

## 88. A Cloud-Edge System for Multimodal Clinical Screening in Resource-Constrained Rural Settings

**arXiv ID:** 2608.12745 | [PDF](https://arxiv.org/pdf/2608.12745v1)

**作者:** Hei Ting `[一作]`, Z. Morley Mao `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种云-边缘协同的多模态医疗AI架构，结合边缘的轻量级专用模型和云端LLM进行临床总结；

**💡 创新点**

创新点在于将感知与推理分离、边缘动态工具选择、仅传输结构化JSON证据，从而实现低带宽、低延迟、事实扎根的诊断支持；

**🔧 技术方法**

使用了LLM（Gemini 2.5 Pro、GPT‑5.4）作为工具协调器和总结器，边缘侧部署17个专业模型（如ECGNet、ChestXRay、RetinaGrader），以及结构化输出协议；

**📊 数据集**

评估数据集包括100个多模态临床案例（心脏、产科、创伤、眼科等），采集自MIMIC‑IV、EchoNet‑Dynamic、HC‑18和EyeRounds.org；

**📈 对比分析**

与两种云端基线（Agentic、Direct）及两种LLM相比，混合架构在Oracle Accuracy达到0.86‑0.90、KG验证精度最高（0.95‑0.96），传输数据仅6.5 KB，延迟保持在25‑38 s，token成本低至1‑9 k，表现优于云端方案；

**⚠️ 局限性**

局限性包括案例生成的模拟性、工具在农村域外的泛化不确定、KG验证阈值主观、缺乏多中心临床验证以及仅对部分农村常见疾病覆盖不足。

---

## 89. Reliability-Aware Sexism Detection: Combining DPO with Annotator Agreement and Token-Level Confidence Scoring

**arXiv ID:** 2608.12330 | [PDF](https://arxiv.org/pdf/2608.12330v1)

**作者:** Hadi Mohammadi `[一作]` (Utrecht University), Anastasia Giachanou `[通讯]` (Utrecht University)

**通讯引用:** 1937 | [OpenAlex ID](https://openalex.org/A5072912426)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RA-DPO，结合标注者一致性、模型置信度和词级不确定性构造可靠性得分，用于训练优选对和推理时的拒绝预测。

**💡 创新点**

创新在于统一利用外部标注一致性与内部模型置信度形成可靠性评分，并同时在训练和推理两阶段使用。

**🔧 技术方法**

采用DPO（直接偏好优化）、可靠性评分R(x)、数据子集排序与阈值拒绝，结合token‑uncertainty。

**📊 数据集**

使用EXIST 2023多语言性别歧视检测数据集，包含6,920条帖子、六位标注者。

**📈 对比分析**

与基线SFT、全量DPO以及多种Smart‑k%采样对比，RA‑DPO在全量下与标准DPO相当，使用30%数据即可达到相同F1；在推理时在50%覆盖率下准确率提升至96.2%。

**⚠️ 局限性**

仅适用于短文本，token‑uncertainty对长文本或多模态可能不佳。

---

## 90. Auditable agentic AI for evidence-grounded thyroid ultrasound diagnosis and reporting

**arXiv ID:** 2608.12590 | [PDF](https://arxiv.org/pdf/2608.12590v1)

**作者:** Haifan Gong `[一作]` (Sun Yat-sen University), Guanbin Li `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了基于代理驱动的证据聚合框架，用于甲状腺超声图像的分割、分类及诊断报告生成；

**💡 创新点**

创新点在于将诊断任务拆解为证据推理过程，结合多模态工具与大语言模型实现自动证据聚合与可解释性输出，并提出新的证据质量与可解释性评估指标；

**🔧 技术方法**

采用大语言模型（LLM）作为控制器，结合插件化工具（分割网络、分类网络、放射组学分析、超声评估等）以及提示工程和插件调用机制；

**📊 数据集**

使用公开的多中心甲状腺超声数据集OpenThyroid，涵盖多种设备与机构的多模态影像数据；

**📈 对比分析**

与20种基线方法对比，实验显示在分割Dice、分类AUROC、AUPRC等指标上均显著提升，最高可达Dice 94.77%、分类AUROC 0.9711；

**⚠️ 局限性**

局限性包括：仅为回顾性单中心验证，缺乏前瞻性临床试验；模型高度依赖LLM与算力，可能受限于模型更新；数据多样性仍不足以覆盖所有人群与设备。

---

## 91. Perturbation-based Regional Interpretability through Subtraction Mapping (PRISM): naming-error dissociations in language models and post-stroke aphasia

**arXiv ID:** 2608.12717 | [PDF](https://arxiv.org/pdf/2608.12717v1)

**作者:** Xiang Guan `[一作]` (University of South Carolina), Julius Fridriksson `[通讯]` (University of South Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文提出并实现了 PRISM 框架，通过在视觉‑语言 transformer 上对单层加噪扰动并对同一行为测试（Philadelphia Naming Test）的错误类别进行子减法分析，得到层级上错误类别差异映射，并与患有慢性脑卒中失语症的 213 名患者在同一任务下的脑区损伤‑症状映射进行并行对比。

**💡 创新点**

创新点在于将人类神经影像学中的减法分析与阈值无关簇增强（TFCE）等统计方法迁移至语言模型解释性研究，提供一种可外推、可重复的空间分辨率方法来验证 transformer 层是否在功能上专门化；并首次实现了模型扰动与人类损伤数据的跨介质对齐与复制。

**🔧 技术方法**

核心技术包括：多种扰动参数（层、噪声标准差 σ、扰动密度 ρ、随机种子）对模型的重量加噪；对错误类别比例进行层级子减法；层级 TFCE 与层序列置换检验；在人类侧采用 Spearman 相关差异 VLSM 与样本重采样置信区间；以及对保留层集的剂量‑反应分析。

**📊 数据集**

使用的数据集为 13 billion 参数的 LLaVA‑1.6‑Vicuna‑13B 模型在 158 条基线正确项上的 40 颗随机种子扰动实验，以及 213 名慢性脑卒中失语症患者完成的 175 条 Philadelphia Naming Test，配合 JHU 左半球白质/灰质 ROI 病灶载荷。

**📈 对比分析**

在并行的两条管线中，模型与患者分别通过层级 TFCE 与 ROI 相关差异 VLSM 进行统计推断；在 40/40 种子拆分和 50/50 患者拆分上均实现了跨介质复制，结果在“语音偏好”方向上均显著，而“语义偏好”方向仅为一致符号但不显著，显示 PRISM 能在两种系统中捕捉到相同的功能分化。

**⚠️ 局限性**

局限性包括：仅测试了单一 vision‑language transformer 与单一行为任务，未完成 Stage 3 的 ROI 级介入验证；残差连接可能使层级扰动不完全等价于局部脑损伤；结果对较小样本的“语义偏好”仍不显著，需更大规模或多任务验证；方法聚焦于层级粗粒度，未细化到头部或特征层面。

---

## 92. CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution

**arXiv ID:** 2608.12629 | [PDF](https://arxiv.org/pdf/2608.12629v1)

**作者:** Zihao Ye `[一作]` (NVIDIA), Luis Ceze `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种联合设计的 GPU 核心代理系统，通过类型化调度 IR 与可演化的编译器环境实现自动化的高性能核代码生成与演化。

**💡 创新点**

创新点在于：①将硬件调度细节嵌入可编辑 IR 以实现局部诊断；②构建可演化的 harness，将失效模式转化为可重用的编译器规则、分析与成本模型；③采用逐步发现的 IR 演化循环，使新硬件功能与编译器提升同步进行。

**🔧 技术方法**

采用了类型化调度 IR（类似 Triton 的 @cake.schedule），LLVM‑based 编译后端、成本模型、静态分析与 GPU 运行时验证，结合大规模 GPU 工作负载与 NVIDIA 设备（Ampere~Blackwell）。

**📊 数据集**

使用了来自生产库的多种 kernel 族（Flash‑Infer、CUTLASS、DeepGEMM、FlashAttention、Flash‑KMeans 等）以及专门构建的前向/解码工作负载（Kimi Delta Attention、Gated DeltaNet、MiniMax sparse attention）。

**📈 对比分析**

在 B200 GPU 上进行清起始演化与前沿核合成实验，平均 80M 令牌预算下，agent‑generated IR 在 Flash‑KMeans 上达到 1.144× 的速度提升（对比基线 0.928×）；在 KDA 预填充上实现 2.05× 的几何平均加速；在 Dispatcher‑backed 库级实现中，KNN、KMeans 等库级加速分别为 1.418×、2.116× 与 1.803×。对比直接 CUDA/PTX，性能普遍更优。

**⚠️ 局限性**

局限性包括：仅在 NVIDIA GPU（Ampere~Blackwell）上验证；成本模型与分析尚不完整，仅在部分架构校准；编译器演化仍需人工合并门控；对非 NVIDIA 架构的迁移及更复杂的异构设备支持尚未实现。

---

## 93. Position: Reasoning is a Learnable Rule-Based Process

**arXiv ID:** 2608.12325 | [PDF](https://arxiv.org/pdf/2608.12325v1)

**作者:** Rachel Lawrence `[一作]` (Microsoft Research), Jacqueline Maasch `[通讯]` (Cornell Tech)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5090999346)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于规则的推理的操作性定义，并提供了验证与安全性的形式化概念与检查清单，说明如何在生成式AI中区分真正的推理与表面化的推理表演。

**💡 创新点**

核心创新在于将推理视为可学习的、精确规则应用的过程，并引入“有效性”与“可靠性”两级标准，同时提出了一套可执行的算法模板与可复现的检查清单，弥补了生成式AI评估中缺乏构念效度的问题。

**🔧 技术方法**

主要技术为形式化定义、数学符号化、伪代码实现、规则选择器与终止规则设计，此外示例中使用了逻辑演绎、贝叶斯推理、强化学习等经典范式。

**📊 数据集**

文中未在实验中使用具体数据集，主要以理论与伪代码为主；若要实验，可参考常用逻辑/数学推理基准或LLM推理任务。

**📈 对比分析**

论文未提供实验比较与性能数据，核心贡献在概念与方法论层面。作者建议未来通过基准测试验证有效性与可靠性，但本工作未进行此类实验。

**⚠️ 局限性**

局限性包括：缺乏实证验证，规则可学习与可解释性的具体实现仍不完整；对深度神经网络中规则的定位与可控性研究仍属前瞻性；在实际应用中如何自动化评估有效性与可靠性仍是未解问题。

---

## 94. MV2: Multi-View Multi-Vehicle Driving Dataset for Novel View Synthesis

**arXiv ID:** 2608.12442 | [PDF](https://arxiv.org/pdf/2608.12442v1)

**作者:** Sanjay Bhargav Dharavath `[一作]` (International Institute of Information Technology Hyderabad), Zakaria Laskar `[通讯]` (Indian Institute of Science Education and Research Thiruvananthapuram)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Multi‑View Multi‑Vehicle (MV²) 数据集和基准，用于评估在多辆车辆、跨视角大变换下的驾驶场景新视角合成（NVS）方法。

**💡 创新点**

创新点在于：①通过在汽车、滑板车和无人机上同步采集同一场景的多轨道图像，构造了真正跨车辆的广视差数据；②对图像进行结构光重建、手工像素对应和极线一致性验证，保证高质量的相机位姿；③设计了两种评估设置（车内视角与跨车辆、航拍与地面）来衡量 NVS 的外推能力。

**🔧 技术方法**

技术包括：基于 COLMAP 的 SfM 与位姿验证、RoMA 密集匹配、离散与连续 3D 场景表示（NeRF、3DGS、PVG、DroneSplat）、多尺度深度估计（DaV3）和动态物体掩码（DroneSplat+SAM），以及多种 NVS 性能指标（PSNR/SSIM/LPIPS）。

**📊 数据集**

使用的数据集为 MV²，包含 50 个场景、12,000 张 1080×1920 的高质量图像，来自车、滑板车和无人机三种平台；此外在实验中对 Waymo Open Dataset 进行了对比验证。

**📈 对比分析**

比较方法涵盖静态 NVS（NeRF、3DGS）、动态 NVS（PVG、DroneSplat）和无训练的前馈 3DGS（Depthsplat、Monosplat、Mvsplat）。实验结果表明：优化型方法在同轨道测试中表现最佳，但随视角差距增大性能显著下降；无训练前馈方法在跨车辆或航拍-地面情形下的 PSNR/SSIM 仅为 10–20 dB，LPIPS 高达 0.3+；深度过滤与动态掩码能提升约 2–3 dB。

**⚠️ 局限性**

局限性包括：①样本数量仍有限（仅 50 场景）；②跨车辆视角极差下 NVS 效果不理想，说明当前方法对大视差的泛化能力不足；③仅使用单目相机与手工像素对应，缺乏大规模自动标注；④对动态物体的分割依赖交叉掩码，精度有限；⑤对实时性与推理速度的评估缺失，难以直接应用于在线仿真。

---

## 95. Towards Sparsely Annotated Open-World Object Detection

**arXiv ID:** 2608.12714 | [PDF](https://arxiv.org/pdf/2608.12714v1)

**作者:** HeeJu Han `[一作]` (Pusan National University), Jinsun Park `[通讯]` (Pusan National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了稀疏标注开放世界目标检测（SA-OWOD）任务，并设计了双视角目标发现（DPOD）框架，分别通过 KTRM 复原未标注的已知目标和通过 DDTG 利用跨视角语义不一致性发现未知目标。

**💡 创新点**

创新点在于将稀疏标注与开放世界目标检测联合起来，首次通过伪标签恢复与特征分离并结合交叉视角不一致性生成器来同时解决已知目标的缺失标注与未知目标的发现问题。

**🔧 技术方法**

技术手段包括：基于 CROWD 的对象级候选框生成、伪标签化与教师-学生一致性滤波、特征空间的设施定位分离损失、以及利用两视角投影后的分类余弦相似度进行未知候选筛选。

**📊 数据集**

使用的基准数据集为 Pascal VOC 与 MS COCO 结合的开放世界目标检测数据集，并在不同稀疏标注配置（Easy、Hard、Coco50missp、Keep1、Extreme）下进行训练与评估。

**📈 对比分析**

与现有开放世界目标检测方法（RandBox、PROB、OrthogonalDet、CROWD）在相同稀疏标注协议下对比，DPOD 在所有稀疏程度下均保持或提升已知目标的 mAP，并显著提升未知目标召回率（U‑Recall），尤其在 Hard 与 Extreme 级别下超过基线 20%+ 的召回率提升。

**⚠️ 局限性**

局限性在于对目标性与相似度阈值（τ_o、τ_s）采用固定设置，难以适应不同任务和类别的多样性，未来可考虑自适应阈值或无监督阈值学习策略。

---

## 96. LLMs Know the Constraint But Do Not Use It: Activation Bottlenecks in Pragmatic Constraint Reasoning

**arXiv ID:** 2608.12321 | [PDF](https://arxiv.org/pdf/2608.12321v1)

**作者:** Yubo Li `[一作]` (Carnegie Mellon University), Rema Padman `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4485 | [OpenAlex ID](https://openalex.org/A5046671743)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在隐式约束推理任务中的失败现象，提出条件约束激活(K/S/R/P)诊断，并通过四元组测试、提示梯度、激活补丁等方法定位失败机制；

**💡 创新点**

将隐式约束失败归因为路由问题而非知识缺失，区分两种失败模式（过激活与欠激活），并揭示所有已知提示干预仅通过提升保守偏差而非修复路由，指出需要在路由层面进行修正；

**🔧 技术方法**

使用线性探针、激活补丁、两层提示实验、Cohort提示、思考预算扫描、因果中介分析等技术；

**📊 数据集**

基于HOB（Heuristic Override Benchmark）核心100场景数据集（约140个情境，4种启发式×5种约束），并在14种公开/封闭LLM上评估；

**📈 对比分析**

通过与四元组基线对比、提示梯度和提示干预的双维度前后对比，发现约束激活准确率可达90%以上，但在过激活模型中仍导致大比例错误；提示干预在所有模型上提升保守偏差，几乎不改善约束路由；

**⚠️ 局限性**

限制包括：仅在两个开源模型上验证路由失败、单层补丁可能不足、数据集仅限英文日常场景、未测试跨语言或领域迁移、评判者单一模型可能漏检错误等；

---

## 97. Vision-Language Models are Fragile Multilingual Associators

**arXiv ID:** 2608.12333 | [PDF](https://arxiv.org/pdf/2608.12333v1)

**作者:** Ritabrata Chakraborty `[一作]` (Manipal University Jaipur), Umapada Pal `[通讯]` (Indian Statistical Institute Kolkata)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了M²BIND基准，用来检验视觉语言模型在上下文与查询语言不同时的概念绑定能力。

**💡 创新点**

创新点在于：①独立控制上下文语言与查询语言，系统探测跨语言的绑定稳定性；②引入因果干预（ICE）和因子化边际（FM）指标，对绑定内部机制进行细粒度分析；③揭示词标记化不公平和脚本差异对绑定的实质性影响。

**🔧 技术方法**

使用了多语言自然语言处理、视觉语言模型（如LLaVA‑1.5‑OV‑7B）以及基于Decoder层的因果干预方法，结合Log‑prob 评分、Accuracy 与 FM 等指标。

**📊 数据集**

数据集为基于Blender渲染的Shapes绑定任务，扩展为8种语言（English, French, Chinese, Arabic 等），每种语言包含上下文与查询的多语言组合。

**📈 对比分析**

与单语环境下的基准进行对比，发现跨语言绑定的 FM 下降至约 2.8–4.1（相较于单语 5.0 以上），Accuracy 虽保持 0.9 以上但已不足以反映绑定衰退；因果干预显示绑定信息在跨语言时推迟到更深层。

**⚠️ 局限性**

局限性包括：仅在简单的两物体 Blender 场景上测试，未覆盖真实图片和更复杂视觉结构；主要评测 LLaVA VLM，其他商业模型的验证有限；语言覆盖仍有限，未深入研究更广泛语言组合。

---

## 98. Non-Degenerate Risk Certification for Automated Security Decisions: A Decision-Contract Theory with ATT\&CK-Aligned Triage as a Worked Instance

**arXiv ID:** 2608.12444 | [PDF](https://arxiv.org/pdf/2608.12444v1)

**作者:** Zhenpeng Li `[一作]` `[通讯]` (Guangzhou Health Science), Zhenpeng Li (Guangzhou Health Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种决策合约理论，解决了传统无条件风险保证在自动化安全决策中的空洞性问题，并在此基础上给出了误差保守律、单一化自动化的几何容量κ(f)与非退化动作性证书；

**💡 创新点**

创新点在于将风险保证视为决策合约的属性，揭示通过放弃行为或语义粗化隐藏误差的机制，并引入单一化容量与动作性双侧保证，首次实现对自动化风险与覆盖率的联合控制；

**🔧 技术方法**

采用了分割合成（Split Conformal）与非合成风险控制（CRC）技术，构建非合成度量并计算κ(f)、κ_α(f)，同时使用LLM输出的概率作为非合成分数；

**📊 数据集**

实验使用了三大IDS基准数据集（CIC-IDS-2018、HIKARI-2021、RT-IoT2022）以及六种开源LLM模型（Gemma、LLaMA、Mistral、Qwen等）与传统XGBoost、LightGBM基线；

**📈 对比分析**

与无放弃基线、阈值裁剪基线以及ML+CRC基线比较，CRC在保持FAR≤α的同时平均实现约83%正确自动化率，且在12种配置下均能在α=0.05时达到高覆盖率；

**⚠️ 局限性**

局限性包括：仅针对4种ATT&CK技术的粗粒度标签，假设数据可交换性，LLM概率排序质量差时κ(f)可能低估容量，对更细粒度技术及多标签场景的推广需要进一步验证；

---

## 99. PatientAct: Theory-Grounded Mental Health Client Simulation

**arXiv ID:** 2608.12750 | [PDF](https://arxiv.org/pdf/2608.12750v1)

**作者:** Sahand Sabour `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 PatientAct 框架，结合 5Ps 临床案例公式化、动态记忆层与信任门控以及情绪-行为-抗拒三段流程，生成具因果深度且行为多样的 LLM 客户端模拟。

**💡 创新点**

创新点在于：①以 5Ps 取代单一治疗模式的框架，实现因果深度；②将记忆条目按信任阈值分层，动态限制信息披露；③将抗拒划分为量、内容、风格三维度，模拟真实情绪化反应；④强调配置质量而非仅靠动态机制，提升模拟真实度。

**🔧 技术方法**

技术手段包括：GPT‑5.4 用于生成与校验 Profile，GPT‑4o 作为模拟核心 LLM；检索式动态记忆与信任门控；情绪‑行为‑抗拒决策管线；多维评估指标与 LLM‑Judge。

**📊 数据集**

数据集为 40 条专家手工设计的临床情景（20 抑郁 + 20 焦虑），通过 LLM 生成 40 个 Profile；随后用相同 Therapist 代理进行 15 轮对话，共 160 条会话，用以评估与对比。

**📈 对比分析**

与三类基线（Patient‑ψ、AnnaAgent、ConsistentMI）对比，PatientAct 在五个评估维度（连贯性、披露节奏、抗拒质量、情感真实性、行为真实性）均名列第一，抗拒质量提升显著 (+0.67)，行为真实性提升 (+0.63)，披露节奏提升 (+0.50)。Ablation 结果表明信任门控、动态记忆与决策管线各自均对性能贡献显著。

**⚠️ 局限性**

局限性包括：仅评估抑郁与焦虑；单 15 轮会话，未考察多会话信任累积；仅英语、GPT‑4o；Profile 生成可能继承 LLM 偏见；评估仅关注模拟真实度，未验证对训练或评估 LLM 处理师的实际提升。

---

## 100. VOS-Agent: The 1st Place Solution for the 8th LSVOS Challenge (MOSEv2 Track)

**arXiv ID:** 2608.12721 | [PDF](https://arxiv.org/pdf/2608.12721v1)

**作者:** Canyang Wu `[一作]` (Harbin Institute of Technology), Jianlong Wu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多代理协作框架VOS-Agent，针对复杂视频对象分割中的小目标与语义主导目标，分别采用视觉跟踪和语言模型进行定位与身份判定，并将结果反馈给SAM3实现精细分割。

**💡 创新点**

创新点在于通过目标感知与路由机制将不同特征目标分配给专门的跟踪或语义推理子代理，实现统一的SAM3基础上可扩展的任务导向路径，而无需对SAM3进行微调。

**🔧 技术方法**

核心技术包括SAM3预训练模型、视觉单目标跟踪器SUTrack、跨模态大语言模型Qwen3.5进行描述生成与定位，以及基于IoU阈值的协同决策策略。

**📊 数据集**

使用MOSEv2数据集进行评测，该数据集包含5024段视频、10074个目标，涵盖遮挡、消失再出现、小目标、语义区分等真实复杂场景。

**📈 对比分析**

在MOSEv2测试集上，VOS-Agent获得官方J&F*指标69.82%，排名第一；相较于仅使用SAM3的62.49%以及加入语义代理后提升至67.07%，最终提升约7.33个百分点。

**⚠️ 局限性**

局限性包括：依赖预训练模型的推理速度相对较慢；对极端小目标或极度相似的多实例场景仍可能出现定位或身份混淆；以及缺乏对非视觉模态（如音频）的扩展。

---

## 101. Fast Length-Squared Sampling for Positive-Semidefinite Matrices

**arXiv ID:** 2608.12503 | [PDF](https://arxiv.org/pdf/2608.12503v1)

**作者:** Rajarshi Bhattacharjee `[一作]` (University of Massachusetts Amherst), Aaron Tian `[通讯]` (University of Massachusetts Amherst)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种只需O(n)期望时间的拒绝采样算法，能在正半定矩阵上高效实现长度平方采样，并利用该方法快速估计 Frobenius 范数以及构造低秩逼近；

**💡 创新点**

创新点在于利用正半定矩阵的离散不等式将采样复杂度降到O(n)，并证明此复杂度在该矩阵类下是最优的；

**🔧 技术方法**

核心技术为基于对角分布的双重采样与拒绝采样，利用正半定性质保证采样概率合法，并结合 Chernoff 绑定实现估计；

**📊 数据集**

文中未给出具体实验数据集，全部结果为理论分析与证明；

**📈 对比分析**

与先前假设已知列范数或需要O(n^2)访问的方案相比，本文实现了无列范数假设、子线性时间的采样与估计，理论上达到最优；

**⚠️ 局限性**

局限在于仅适用于正半定矩阵，且在需要多次独立采样时仍需每次读取对角线，无法进一步降低每次采样的平均复杂度。

---

## 102. SDAM: Structure-Difference-Aware Memory Evolution for Complex Text-to-SQL

**arXiv ID:** 2608.12338 | [PDF](https://arxiv.org/pdf/2608.12338v1)

**作者:** Keyan Xu `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8951 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种结构差异感知的记忆方法，并构建了相应的记忆演进框架，用以提升文本到 SQL 的生成质量。

**💡 创新点**

创新点在于：①利用结构差异感知推理树捕捉多路径 SQL 结构差异；②采用矛盾感知反射从结构、模式、执行三维角度提炼深层语义规则；③设计基于 Schema 的记忆演进机制，将记忆单元与数据库表列绑定，提升语义对齐与错误纠正。

**🔧 技术方法**

核心技术包括：多模型推理与结构差异推理树、LLM 反射与提取器、Schema Anchor 记忆单元、语义相似度检索、矛盾过滤与增量更新。

**📊 数据集**

实验数据集为 Spider（跨域文本到 SQL 基准）和 BIRD（大规模真实世界基准），并在 Archer（复杂推理）上进行额外验证。

**📈 对比分析**

与现有最先进的 LLM 代理方法（如 CHESS、Alpha‑SQL、ExpeSQL 等）以及通用记忆框架 ACE 进行对比，本文方法在 Spider 上达到 88.0% 执行准确率，在 BIRD 上达到 70.2%，相较于最佳基线提升 2.0%–2.7%，并在高难度样本上显著表现优异。

**⚠️ 局限性**

局限性主要体现在：①对极其罕见或自定义业务逻辑的自我反思能力有限，难以在无外部先验知识时快速捕获深层语义；②由于记忆单元与具体 Schema 绑定，跨数据库迁移与泛化仍面临挑战。

---

## 103. When Can You Trust Offline Evaluation of Equal-Cost Top-k Allocation? A Controlled, Reproducible Benchmark and Practitioner's Guide

**arXiv ID:** 2608.12489 | [PDF](https://arxiv.org/pdf/2608.12489v1)

**作者:** Binshuang Li `[一作]` `[通讯]` (Independent Researcher), Binshuang Li (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个针对预算约束的 top‑k 分配策略的离线评估基准，评估了多种 OPE 估计器在不同日志环境下的可靠性。

**💡 创新点**

提出重叠风险取决于记录器与目标策略的动作对齐，而非日志尖锐度；交叉拟合仅加剧“优化者诅咒”，推荐诚实策略拆分和双重稳健估计作为默认方案。

**🔧 技术方法**

使用了 DM、IPS、SNIPS、DR、Switch‑DR、mIPS 六种估计器，LightGBM 作为预测模型，并结合有效样本量（ESS）和支持缺失等诊断指标。

**📊 数据集**

实验基于合成数据、IHDP、Hillstrom、Lenta、Jobs 等公开数据集，并在 IHDP 协变量上生成了 ACIC‑2017 风格的硬化场景。

**📈 对比分析**

通过 270 个配置的中位数相对 RMSE 进行比较，模型驱动估计器优于纯权重估计器；ESS 诊断能在跨日志环境中对误差进行排名；交叉拟合不缓解偏差，而诚实拆分显著降低估计偏差。

**⚠️ 局限性**

受限于模拟日志、单一二元动作、等成本约束、已知倾向以及只能在多日志环境下排名误差，且对真实观测日志和更复杂约束的推广性有限。

---

## 104. Multi-AUV Ad-hoc network-based Target Tracking: A Value Gradient Guidance Multi-Agent Diffusion Reinforcement Learning Approach

**arXiv ID:** 2608.12436 | [PDF](https://arxiv.org/pdf/2608.12436v1)

**作者:** Jiaao Ma `[一作]` (Northeastern University), Zhenyu Wang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了多AUV自组织的MDCA分层协作架构和VGG-MADiffRL算法，用于在受限声学通信和动态拓扑条件下实现多AUV协同目标跟踪。

**💡 创新点**

创新点包括：①将值梯度引导的扩散策略应用于多智能体强化学习；②双目标联合优化（Q引导+策略梯度）以提升训练稳定性；③CTDE下的分层协作网络，实现全局目标与局部执行的无缝对接。

**🔧 技术方法**

技术手段包括扩散模型（forward/reverse）+值梯度引导、双Critic网络、软目标更新、经验回放、OceanGym+自定义流体动力学与声学通道模型。

**📊 数据集**

实验使用基于OceanGym的自定义仿真环境（模拟海流、声学传播和自组织拓扑），并未使用公开真实数据集。

**📈 对比分析**

在4/6/8/10个AUV追踪2/3目标的四种情景下，VGG-MADiffRL相较于MASAC、MAPPO、MAAC、MATD3、MADDPG、DSBM、MA-A3C等七个基线，收敛更快、跟踪准确率最高、均值跟踪误差最低、误差标准差最小。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实海洋实验；未充分考虑能量消耗与障碍物避免；声学通信鲁棒性与多网络间异构性仍待进一步研究。

---

## 105. The AI Accountability Ecosystem in the Era of Language Models

**arXiv ID:** 2608.12320 | [PDF](https://arxiv.org/pdf/2608.12320v1)

**作者:** Chris Percy `[一作]` (University of Warwick), Artur d'Avila Garcez `[通讯]` (University of London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文回顾并更新了基于AI责任生态的框架，聚焦大型语言模型时代，并提出了三项关键调整。

**💡 创新点**

创新点在于将责任重心从单一产品转向多层供应链，强调持续结果监测，并将终端用户纳入责任体系。

**🔧 技术方法**

采用文献综述、案例分析和政策框架比较等概念性方法，结合现有AI安全报告和EU AI法案。

**📊 数据集**

未使用特定数据集，主要依赖公开的研究报告、政策文件和行业案例。

**📈 对比分析**

通过对比原始生态模型与更新后模型的责任分配与监控机制，说明新框架更适应LLM的系统性风险，但缺乏量化实验结果。

**⚠️ 局限性**

局限在于缺乏实证验证、实现细节不完善，以及非监管机制易碎片化、缺少统一标准与可操作性。

---

## 106. AnchorSIPS: A Synthetic Dataset and Evaluation Resource for Evidence-Supported Psychosis-Risk Symptom Measurement

**arXiv ID:** 2608.12329 | [PDF](https://arxiv.org/pdf/2608.12329v1)

**作者:** Guilherme C. Oliveira `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**通讯引用:** 13528 | [OpenAlex ID](https://openalex.org/A5005014252)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

创建了 AnchorSIPS 10K 结构化精神病风险评估对话数据集，并提供基于证据的中间决策标签。

**💡 创新点**

首次将结构化访谈流程与逐句证据绑定，并采用 plan‑then‑realize 的确定性生成管道，保证标签与文本可审计、可复现。

**🔧 技术方法**

使用确定性规划器 + LLM 实现患者发声（GPT‑OSS‑120B），并在评测中使用多任务 JSON 生成与证据引用评估。

**📊 数据集**

数据集包含 10,000 个合成 Mini‑SIPS 访谈，每个访谈包含 24 题症状查询、跟进细节、类决策、否认精神病、APS 诊断及对应转录句子。

**📈 对比分析**

基准对比 7 大 LLM（GPT‑5.5、DeepSeek V4 Flash、Claude Opus 4.7 等），在宏观决策上表现较好（F1≈0.93），但在证据链接和细节提取上仅能达到 0.16–0.25 的 F1，显示模型在文本‑证据一致性方面存在明显瓶颈。

**⚠️ 局限性**

局限在于完全合成语料与真实临床对话可能存在语言与互动差异，缺乏真实诊断验证，且仅适用于方法研究而非临床部署。

---

## 107. Metropolis-Hastings Sampling of Phylogenetic Networks: Correcting for Symmetries

**arXiv ID:** 2608.12430 | [PDF](https://arxiv.org/pdf/2608.12430v1)

**作者:** Leo van Iersel `[一作]` (Delft University of Technology), Christopher Reichling `[通讯]` (Delft University of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种针对叶子标记的系统发育网络的Metropolis‑Hastings采样方法，利用商马尔可夫链纠正内部对称性导致的欠采样问题。

**💡 创新点**

创新点在于将对称性修正融入商马尔可夫链理论，证明叶子标记网络的自同构群对采样重要，并利用μ‑向量高效计算自同构，从而在不需要全局网络标签的情况下实现正确的采样。

**🔧 技术方法**

技术上使用了商马尔可夫链、Metropolis‑Hastings接受率公式、网络的重排移动（tail、head、rSPR、add/remove 以及 δ‑移动）以及基于μ‑向量的自同构计数算法。

**📊 数据集**

实验使用了小规模网络族（如 (2,2) 级别网络）以及更大规模的随机二叉网络，数据来自自建的 PhyloX 软件。

**📈 对比分析**

在 Python 实现中，使用 μ‑向量的自同构计数相较于传统方法提升了数十倍的速度；对 orchard 网络，证明其自同构群为平凡，从而完全省略对称性校正；总体采样性能在小规模网络上与理论一致，未发现显著的欠采样偏差。

**⚠️ 局限性**

主要局限是自同构计数在大网络上仍然较慢，且对称性在多数真实网络中极少出现；未来工作需开发更高效的自同构求解或直接限制到无对称性的网络空间。

---

## 108. Genetic Fuzzy System-Based Multi-Robot Coordination for Planetary Missions

**arXiv ID:** 2608.12755 | [PDF](https://arxiv.org/pdf/2608.12755v1)

**作者:** Daegyun Choi `[一作]` (University of Cincinnati), Donghoon Kim `[通讯]` (University of Cincinnati)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于遗传模糊系统的去中心化多机器人协同搬运方案，利用遗传算法训练的模糊推理系统（FIS）为机器人生成速度向量，实现在不规则地形上最短路径搬运物体。

**💡 创新点**

创新点在于：1) 将地形可通行性分析（TTA）与二维可通行地图相结合，简化三维地形；2) 设计两套FIS（目标导向与障碍规避）并通过遗传算法同时优化其模糊集合和规则，形成全新的“遗传模糊系统”（GFS）；3) 在无通信网络的前提下实现全局协同，仅利用本地传感信息完成路径规划与协作。

**🔧 技术方法**

使用了遗传算法（GA）优化模糊推理系统、模糊推理系统（FIS）、坡度通行性分析（TTA）、以及基于Fractional Brownian Surface生成的测试环境。

**📊 数据集**

数据集：①人造训练场景（包含局部最小、靠近障碍的目标、拥挤环境三种情形）；②由Fractional Brownian Surface生成的测试地形，用以模拟真实行星表面。未使用真实数字高程模型（DEM）而是合成的仿真地图。

**📈 对比分析**

比较方法：通过仿真计算各场景下机器人总行驶距离、碰撞距离和物体姿态变化；结果显示：训练场景1/2总路径长度分别为419.03 m/369.23 m；测试场景1/2/3的总路径长度分别为351.84 m/419.09 m/328.59 m，且无碰撞、物体姿态保持在3°以内，证明GFS在去中心化协同搬运中的有效性。

**⚠️ 局限性**

局限性：1) 仅在二维平面简化运动学，未考虑三维运动与姿态控制；2) 环境为合成地形，缺乏真实行星DEM的验证；3) 未考虑能量消耗与动力学约束；4) 模糊规则与参数仅在少量场景下训练，泛化能力待进一步验证。

---

## 109. Can We Trust AI Agents in the Supermarket? Sugar Content Inference from Product Images

**arXiv ID:** 2608.12359 | [PDF](https://arxiv.org/pdf/2608.12359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 110. HybridSB-MoE: Dual-Domain Schrödinger Bridges with Scene-Adaptive Expert Routing for Speech Enhancement

**arXiv ID:** 2608.12715 | [PDF](https://arxiv.org/pdf/2608.12715v1)

**作者:** Zhengyi Lu `[一作]` (Oakland University), Yao Qiang `[通讯]` (Oakland University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出HybridSB-MoE双域增强框架，通过异构混合专家谱路和Schrödinger桥波形路实现语音增强

**💡 创新点**

创新点在于：①异构专家架构产生明确的认知不确定性；②波形桥产生天真不确定性；③两种不确定性通过非对称融合动态权重；④引入路径一致性与轨迹正则化，给出K步离散化误差理论保证

**🔧 技术方法**

使用混合专家网络、U-Net+transformer波形桥、路径一致性/轨迹正则化、深度学习训练与校准损失

**📊 数据集**

在VoiceBank+DEMAND数据集上训练与评估

**📈 对比分析**

与多种判别式、扩散式、SB式以及一致性蒸馏基线对比，HybridSB-MoE在PESQ、CBAK、COVL等指标均优于同等K步或更大步数的基线，并与单步蒸馏方法竞争

**⚠️ 局限性**

局限在于仅针对单声道、训练数据分布相近、需人工设计专家原型，理论中的常数保守，需在更大语料与多通道场景进一步验证

---

## 111. SynWeaver: Website-Prior Task and Trajectory Co-Synthesis for Web Agents

**arXiv ID:** 2608.12429 | [PDF](https://arxiv.org/pdf/2608.12429v1)

**作者:** Ruitao Wang `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SynWeaver框架，通过结构化网站探索构建网站地图，并基于此进行任务与轨迹协同生成，以提升网站特定监督的质量和多样性。

**💡 创新点**

创新点在于结合网站先验学习与协同任务-轨迹共合成，解决了传统探索式合成缺乏网站特定先验、导致任务假设不真实的问题。

**🔧 技术方法**

采用结构化网站探索、网站地图构建、基于页面与转移的监督生成、UI感知模型训练以及协同任务-轨迹更新与验证修复等技术。

**📊 数据集**

使用了WebArena和WebVoyager两个基准数据集进行实验。

**📈 对比分析**

与强基线对比，SynWeaver在两大数据集上均表现出更高的任务成功率、轨迹执行准确率以及数据多样性，显著提升了在域内外的泛化性能。

**⚠️ 局限性**

局限性包括对强制性验证码、限制速率或动态身份验证的网站适应性不足，以及当前框架主要依赖教师模型监督，未探索在线强化学习等自我改进方案。

---

## 112. Large Language Models Pass the History Exam But Miss the <<History>>: A Polish High School Exit Exam Matura Benchmark

**arXiv ID:** 2608.12343 | [PDF](https://arxiv.org/pdf/2608.12343v1)

**作者:** Adrian Trzoss `[一作]` (Adam Mickiewicz University), Marcin Moskalewicz `[通讯]` (Adam Mickiewicz University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估八大LLM在波兰2023‑2025年Matura历史考试中的表现，并与人类考生做对比

**💡 创新点**

首次使用官方真实开放式历史考试题构建多模态LLM基准，并引入人类分布基线

**🔧 技术方法**

通过OpenRouter API进行无上下文学习的三次生成，使用Bootstrap CI、Wasserstein距离等统计方法

**📊 数据集**

2023–2025年三份官方波兰Matura历史试卷（短答+600词论文），配有标准评分准则

**📈 对比分析**

以归一化分数排名和分布距离评估，发现所有LLM显著优于人类，排名随题型、来源和地域不同而波动

**⚠️ 局限性**

仅三份试卷、缺乏单语模型、多模态评估受限、未尝试提示技巧，且人类数据仅为分布统计

---

## 113. RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills

**arXiv ID:** 2608.12416 | [PDF](https://arxiv.org/pdf/2608.12416v1)

**作者:** Runyi Zhao `[一作]` (Shenzhen Loop Area Institute), Guiliang Liu `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的RoboSynChallenge基准，结合大规模合成数据与真实世界评估，目标是提升机器人抓取、操控任务的通用性和数据效率。

**💡 创新点**

创新点包括：①首个Sim2Real统一评估基准；②自动化生成合成数据流水线；③将模拟训练与真实世界评估统一在同一排行榜。

**🔧 技术方法**

采用Transformer、扩散模型、Vision‑Language‑Action框架以及World Action Model等最新学习技术进行基线实现。

**📊 数据集**

使用由EmbodiChain生成的合成状态‑动作轨迹以及少量由遥控操作收集的真实数据，共计约1k条任务轨迹并覆盖多种环境、光照、物体等变异。

**📈 对比分析**

通过对比ACT、Diffusion、π0、π0.5、Motus等基线在模拟与真实测试中的成功率、动作步数和推理时间进行评估，结果显示真实场景成功率显著低于模拟，且不同模型差距显著。

**⚠️ 局限性**

限制：仅覆盖双臂抓取任务，真实数据量仍有限；合成数据与真实世界存在差异导致Sim2Real迁移瓶颈；基准不包括长期计划或多任务学习等更复杂场景。

---

## 114. Comparative Analysis of Multilingual Pre-trained Models for Nepali Automatic Speech Recognition

**arXiv ID:** 2608.12327 | [PDF](https://arxiv.org/pdf/2608.12327v1)

**作者:** Suman Paudel `[一作]` (Tribhuvan University), Sarbin Sayami `[通讯]` (Tribhuvan University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5028216772)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在同一训练协议下，对六种多语言预训练ASR模型进行微调并在三套测试集上评估，构建了首个面向尼泊尔语的跨模型、跨数据集基准。

**💡 创新点**

①证明语言家族相似性在预训练中可弥补模型规模差距；②首次公开同规模模型的实时因子（RTF）对比，揭示CTC解码在相同准确率下快29倍；③提供细粒度的模型性能与资源消耗对比。

**🔧 技术方法**

使用Wav2Vec 2.0、Whisper、Conformer‑CTC等架构；统一预处理、SpecAugment、AdamW、早停；在单GPU（L4）上评测RTF。

**📊 数据集**

OpenSLR SLR54（≈165 h）作训练集，OpenSLR、FLEURS、Common Voice（分别约5 h、10 h）作三套独立测试集。

**📈 对比分析**

采用同一预处理、分割、学习率调度和优化器，对六模型在三测试集上报告WER、CER、RTF。结果显示Whisper‑Turbo和IndicWav2Vec在内测集上相同（≈14.8 % WER），后者参数仅为前者的1/9；CTC模型显著更快，RTF仅≈0.002，远低于Whisper‑Turbo的≈0.076。

**⚠️ 局限性**

受限于单一GPU、批大小与训练轮次；Conformer‑Hi使用不同训练框架；评测仅覆盖阅读式语料，未覆盖对话、代码混合或方言；未使用外部语言模型，真实性能可能更高。

---

## 115. Why Do AI Agents Break Rules? How Framing, Context, and Social Signals Shape Compliance

**arXiv ID:** 2608.12323 | [PDF](https://arxiv.org/pdf/2608.12323v1)

**作者:** Mika Okamoto `[一作]` (Georgia Institute of Technology), Kutluhan Erol `[通讯]` (Izmir University of Economics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对12个instruction‑tuned LLM在模拟企业采购场景中进行实验，评估其合规行为并探究法律合规理论对模型行为的预测。

**💡 创新点**

将法律经济学的合规理论直接用作经验假设，发现模型按训练方向分为两类，揭示了“执行信息悖论”和“紧急性例外”，并指出模型选择本身是治理决策。

**🔧 技术方法**

在Slack模拟工作区中让模型回答采购请求，利用LLM‑as‑judge提取推荐结果，设计多维实验（规则框架、罚金、机构压力、社交信号、多轮对话），并进行推理可视化分析。

**📊 数据集**

构造了自制的虚拟采购案例（5个供应商、ISO 14001认证）以及模拟监管、经理授权、同行处罚等情境文本，全部为实验自造数据。

**📈 对比分析**

通过合规率、压力测试和违规可检测率等指标对模型进行比较；结果显示安全调优模型合规率高于任务优化模型，但在任何情境下都未达100%合规，部分情境合规率低于60%。

**⚠️ 局限性**

研究仅在模拟环境和单一采购场景下进行，未覆盖跨领域法律与真实组织动力；未掌握模型训练细节；未验证长期持续监控与人类干预的有效性。

---

## 116. Scaling Representation Diversity: Modulated Attention and Reconstructive Regularization for Visual Grounding

**arXiv ID:** 2608.12748 | [PDF](https://arxiv.org/pdf/2608.12748v1)

**作者:** Junyi Hu `[一作]` (Tsinghua University), Yi Zhang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的开词汇视觉语言对齐框架，旨在解决单一模型在多数据集上的泛化瓶颈。

**💡 创新点**

核心创新包括：1）Modulated Attention‑Contrastive Head（mACH），通过广播式跨模态注意力实现多查询高效对齐；2）无推理开销的 JEPA 辅助流，在训练期间通过视觉特征预测补充对齐梯度，维持表示多样性；3）通过 Objects365‑Caption 数据集将离散标签转化为上下文丰富的描述，显著提升语言监督质量。

**🔧 技术方法**

技术细节包括：广播式多头注意力实现的 mACH；EMA teacher‑student 结构的 JEPA 预测器；基于 ConvNeXt 或 RT‑DETR 的视觉主干；使用 FlashAttention‑2 等高效实现；对齐损失采用 BCE，JEPA 采用对比+平滑L1。

**📊 数据集**

主要使用的训练与评估数据集有：Objects365、Objects365‑Caption（新构建）、RefCOCO、RefCOCO+、RefCOCOg、以及用于多语言扩展的机器翻译版。

**📈 对比分析**

与现有方法比较，单检查点的 75M 模型在零样本下可达到 85.3/89.0/82.5（Val/testA/testB）在 RefCOCO，超过 172M 的 GDINO‑T +11% 以上；在 Fine‑tune 后可获得 91.7/93.0/90.2 的 SOTA 结果，且参数量仅 75M，显著低于 490M 的 PropVG 或 13B 的 LISA++-L2。

**⚠️ 局限性**

限制主要体现在：1）JEPA 只在训练期间使用，未评估其对推理速度的潜在隐式影响；2）虽然数据集丰富，但生成过程依赖大型 MLLM，成本高；3）对极端长表达或低频实体的泛化仍待验证。

---

## 117. AirForesight: Current-to-Future Spatial Map Imagination with Cross-Space Planning Consistency for UAV-VLN

**arXiv ID:** 2608.12835 | [PDF](https://arxiv.org/pdf/2608.12835v1)

**作者:** Yutong Liu `[一作]` (Harbin Institute of Technology), Jianlong Wu `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出AirForesight框架，利用多视角观测学习当前地图、未来轨迹与未来地图的结构化表示，并据此预测下一步航点，提升UAV视觉语言导航性能。

**💡 创新点**

核心创新在于：①通过联合训练当前地图、未来轨迹与未来地图的因果注意力实现空间想象；②引入跨空间规划一致性损失，使地图轨迹方向与真实动作方向对齐；③仅用离线标注的语义地图作为监督，避免在线构图延迟。

**🔧 技术方法**

技术实现基于Vicuna‑7B多模态Transformer、MAE解码器、结构化因果注意力机制、跨空间一致性损失、以及路径解码器；离线标注采用LLM+GroundingDINO+MobileSAM。

**📊 数据集**

实验主要在OpenUAV（12,149条轨迹、5视角RGB）和AerialVLN‑S两大基准上进行。

**📈 对比分析**

与Random、Fixed、CMA、NavFoM、TravelUAV等基线相比，AirForesight在OpenUAV Test Seen上NE下降约12m，SR提升≈5.4%，SPL提升≈4.8%；在Unseen Map/Unseen Object同样表现出显著的泛化优势。

**⚠️ 局限性**

局限性包括：在未见地图或物体环境下仍存在较高错误率；依赖离线标注的语义地图，标注成本与可迁移性有限；以及对多视角融合与实时计算的资源需求较高。

---

## 118. Offering Microsecond-Scale Cross-VM Core Elasticity on Colocated Lightweight Virtual Machines

**arXiv ID:** 2608.12633 | [PDF](https://arxiv.org/pdf/2608.12633v1)

**作者:** Yibo Yan `[一作]` (University of Southern California), Seo Jin Park `[通讯]` (University of Southern California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于KVM的ultralight VM substrate，支持微秒级跨VM核心迁移，实现弹性并行宽度。

**💡 创新点**

首次在 commodity‑KVM 上实现微秒级核心调度与弹性并行宽度，并通过核心预占停车机制保证工作不被中断。

**🔧 技术方法**

Fluxion（VMM）+ kFlux（内核模块）+ FluxOS（LibOS）配合 vCPU 停车/重新挂载、共享内存通信和抢占式线程调度。

**📊 数据集**

在服务器无服务器工作负载上评估，包括 Memcached、x264、Web 服务器和 TPC‑H 查询，使用这些实际工作负载。

**📈 对比分析**

与 Firecracker、Cloud Hypervisor 等现有微VM 对比，核心迁移耗时约 13µs，p99 延迟在高负载下可降低 10 倍，整体性能优于静态分配与 cgroup/热插拔。

**⚠️ 局限性**

仅支持 KVM 平台，FluxOS 缺乏完整 Linux 系统调用支持，应用受限于 Rust 语言，且核心迁移仍需与 guest 协作。

---

## 119. Query Timing Produces Opposite Positional Biases Between LLMs and Humans

**arXiv ID:** 2608.12387 | [PDF](https://arxiv.org/pdf/2608.12387v1)

**作者:** Jasin Cekinmez `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 50691 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比不同模型（GPT、Claude、Gemini、Llama、Qwen）在三种法庭情境（刑事、学术、社交）下的 SbS 与 EoS 反应模式，探究证据顺序对判决结果的影响，并分析模型内部概率更新与最终决策的关系。

**💡 创新点**

首次将人类 Step‑by‑Step 与 End‑of‑Sequence 的认知偏差实验框架迁移至 LLM，系统追踪模型代际演化中偏差的出现与方向，并揭示 LLM 与人类在偏差表现上的根本差异。

**🔧 技术方法**

采用 Qiao & Lagnado 的实验框架、两阶段 Fisher 精确检验与 Benjamini–Hochberg 校正来评估偏差显著性；同时通过多次独立跑测评估模型在不同证据顺序下的判决比例。

**📊 数据集**

使用三组构造性法庭案例：刑事谋杀案、学术不端案例、社交不当行为案例，每组包含中立摘要、四条控方证据和四条辩方证据，配合条件概率和最终判决两类问题。

**📈 对比分析**

将 SbS 与 EoS 两种响应模式下的 DP 与 PD 顺序进行比较，发现 GPT‑4o 与 Claude 系列在 EoS 模式下呈显著的递近偏差，而 Gemini‑2.5 Flash 则无显著偏差；对比不同模型代际，偏差显著性随升级而增强。

**⚠️ 局限性**

实验仅使用固定数量和结构的合成证据，缺乏生态效度；评估仅聚焦二元判决，未能完整反映内部概率表示；样本覆盖面有限，未探究训练或架构因素导致偏差差异的根源。

---

## 120. Are Large Language Models Reliable Reviewers? A Benchmark for Error Detection in Financial Documents

**arXiv ID:** 2608.12342 | [PDF](https://arxiv.org/pdf/2608.12342v1)

**作者:** Ying He `[一作]` (Fudan University), Zhixu Li `[通讯]` (Renmin University of China)

**通讯引用:** 3826 | [OpenAlex ID](https://openalex.org/A5065529268)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了 FinED‑Bench——一个面向长篇金融文档错误检测的基准，涵盖三级错误类型与九类真实金融场景，并对 LLM 进行评测。

**💡 创新点**

创新点在于：①首次针对金融文档错误提供多层级错误分类与细粒度子类别；②使用半自动生成 + 人工核验的流水线，避免训练数据泄露；③在评估中加入超长文档子集，探究长上下文对 LLM 的影响。

**🔧 技术方法**

采用 LLM 评测与微调技术，利用 GPT‑4o、Qwen3‑14B 等大模型进行无思考与有思考（chain‑of‑thought）推理；使用基于提示的错误生成和模型重评筛选；通过监督微调提升检测性能。

**📊 数据集**

数据集为 973 篇中文金融文档（平均 3,784.6 词）和 24 篇 32K–120K 词的超长子集，总计 4,123 条人工标注错误，涵盖 15 个子类别；还提供小规模英文版本。

**📈 对比分析**

对比方法：在 10+ LLM 之上计算精确率、召回率与 F1；结果显示 GPT‑4o 最高整体 F1 为 48.34%，在一般知识错误上 52.33%，在金融推理错误上仅 38.00%；超长文档 F1 从 40.16% 降至 16.66%。微调后 Qwen3‑14B 的整体 F1 提升至 53.85%，在推理错误上提升约 20%。

**⚠️ 局限性**

局限性：未覆盖资产负债表等财务报表类文档；不考虑多模态元素（印章、签名）；生成错误仅基于文本，无法验证真实数据源错误；对复杂推理与跨文档一致性仍存在挑战。

---

## 121. SAP-Nav: Spatial Semantic Representation Meets Active Perception for Hierarchical Open-Vocabulary Object Navigation

**arXiv ID:** 2608.12707 | [PDF](https://arxiv.org/pdf/2608.12707v1)

**作者:** Xuetong Pei `[一作]` (Beihang University), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SAP-Nav，一个完全在线、零样本的主动感知框架，用于层级开放词汇目标导航（OVON），可处理场景、房间、区域和实例级指令。

**💡 创新点**

创新点在于：1）Queryable Spatial‑Semantic Representation (QSSR)，通过主动采集全景房间视图并结合空间语义分割，构建可查询的空间语义图；2）Active Viewpoint Verification (AVV)，对候选目标进行视角充分性评估并主动重新定位以获得更佳验证视角，从而实现可靠目标识别。

**🔧 技术方法**

使用技术包括：视觉‑语言模型（VLM）如 Qwen 系列和 GPT‑4o 进行语义分类、问答与属性验证；基于 BEV 的空间语义地图；在线房间分割与全景拼接；几何视角采样与可视性评估；以及对话式 LLM 解析属性约束。

**📊 数据集**

主要使用的公开数据集有：LangMap（多粒度 414 类物体，约 15K 任务）和 HM3D‑OVON（场景级 379 类物体，3K 任务）。

**📈 对比分析**

与训练型与训练无关的基线方法（如 PSL、SenseAct‑M、VLFM、3D‑Mem、MetaNav 等）比较，SAP‑Nav 在 LangMap 上在四个粒度均位居榜首，Region 级甚至超过最强训练基线；在 HM3D‑OVON 上在 SR 方面取得最高分，虽然 SPL 略低。

**⚠️ 局限性**

局限性包括：AVV 的视角选择仅基于几何可视性，未考虑语义信息或运动成本；QSSR 在跨任务时不保留，缺乏持续学习和知识迁移能力。

---

## 122. Intensional Anaphora

**arXiv ID:** 2608.12598 | [PDF](https://arxiv.org/pdf/2608.12598v1)

**作者:** Ezra Keshet `[一作]` (University of Michigan), Steven Abney `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的谓词演算——多元强制性先行词蕴含预设谓词演算（Plural Intensional Presuppositional predicate calculus，简称PIP），并用它对强制性与预设在强制性上下文中的先行词指向问题进行系统分析；

**💡 创新点**

创新点在于：①引入描述性先行词预设，取代传统基于取值的预设；②通过公式标签、局部变量与无选择性闭包实现跨句先行词指向；③使用Σ求和运算处理聚合先行词；④将所有结构直接翻译为标准一阶谓词演算，保持形式的可归约性；

**🔧 技术方法**

技术方法主要是形式语义学：构造PIP的语义规则、预设与愉悦性（felicity）条件，并演示其如何自然解释强制性先行词、强制性与量词的关系；

**📊 数据集**

本文未使用任何实验数据集，而是通过理论示例和对比实例来说明PIP的覆盖能力；

**📈 对比分析**

比较方式为与先前的强制性语义模型（IP‑CDRT、Stone、Brasoveanu等）进行对照，阐明PIP在处理强制性先行词时不产生过度生成且不低估现象；在“能否指向”与“预设满足性”两方面给出了定性的优劣对比；

**⚠️ 局限性**

局限性包括：①仍未覆盖所有强制性与量词相关的复杂句型（如强制性嵌套的多义句、非认知模态的细微差别等）；②缺乏经验数据或自动化评估；③对跨语境动态变化的实时更新尚未实现，需要进一步扩展以兼顾完整的动态语义处理。

---

## 123. Inference-Time Orthogonal Seeding Enables Geometry-Aligned 3D Organ Segmentation for Slice-Propagation Methods

**arXiv ID:** 2608.12658 | [PDF](https://arxiv.org/pdf/2608.12658v1)

**作者:** Md Rakibul Haque `[一作]` (University of Utah), Shireen Y. Elhabian `[通讯]` (University of Utah)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种利用正交种子（轴向、冠状面、矢状面）进行单切片传播的3D器官分割方法，采用无标签的距离加权融合；

**💡 创新点**

创新点在于将种子几何结构（正交三平面）作为关键因素，而非仅依赖注册模型或训练方式，显著提升分割质量；

**🔧 技术方法**

使用自监督注册/对应网络（Sli2Vol）以及可变形变形网络（TransMorph），并实现基于距离的软融合或硬多数投票；

**📊 数据集**

在公开CT数据集上训练：CHAOS、MSD Liver、Pancreas‑CT、KiTS、CT Lymph Nodes；评估数据集包括SLIVER07（肝）、Decathlon‑Pancreas、Decathlon‑Spleen；

**📈 对比分析**

与单轴传播及同注释量的三轴平面传播对比，正交种子在Dice提升约21.9%，NSD提升25.5%，AHD下降53.5%，在所有基线网络上均显著优于对手；

**⚠️ 局限性**

局限在于假设目标器官在三轴上尺寸相近，可能对细长或分支结构效果有限；评估仅覆盖腹部CT器官，未验证MRI或其他解剖部位；融合策略受限于距离权重，难以处理异质边界误差。

---

## 124. Error-Aware Reverse Auction Mechanism for Large Language Model Routing

**arXiv ID:** 2608.12719 | [PDF](https://arxiv.org/pdf/2608.12719v1)

**作者:** Haolong Chen `[一作]` (Shenzhen International Center for Industrial and Applied Mathematics), Guangxu Zhu `[通讯]` (Shenzhen International Center for Industrial and Applied Mathematics)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于逆拍卖的LLM路由框架EA‑RAM，将任务预测责任转移到LLM提供者，并通过平台端评估实现分配；

**💡 创新点**

创新点在于显式考虑双重误差（提供者预测误差与平台评估误差），证明机制在此环境下仍满足贝叶斯激励兼容与个体理性，并给出福利损失界限与稳健性洞见；

**🔧 技术方法**

采用逆拍卖机制、概率预测与评估模型、错误意识的支付规则、理论分析（BIC、IR、CR）、实验模拟与真实基准；

**📊 数据集**

使用RouterBench（11种LLM）与六大基准数据集（HellaSwag、Winogrande、ARC‑Challenge、MBPP、GSM8k、MMLU）进行实验；

**📈 对比分析**

与多种集中式基线（EmbedLLM、IRT‑Router、RouteLLM、FrugalGPT、Cascade Routing）比较，EA‑RAM在成本–性能Pareto前沿与AIQ得分上均优于对手，尤其在加入本地信息时提升更显著；

**⚠️ 局限性**

局限包括需假设误差模型、评估噪声仍会影响结果、通信开销随模型数量线性增长、仅针对单任务非组合情形，未来需扩展至更复杂场景。

---

## 125. When AI Is Your Pastor: A Benchmark for Theological Triage and Pastoral Guidance in Large Language Models

**arXiv ID:** 2608.12324 | [PDF](https://arxiv.org/pdf/2608.12324v1)

**作者:** Alex Chao `[一作]` `[通讯]` (Fide AI), Alex Chao (Fide AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Faith & Moral Guidance Benchmark（FMG‑Bench），旨在系统评估大型语言模型在基督教信仰与牧养问答中的表现，并通过三层神学分诊框架与结构化 harness 对模型输出进行细致打分

**💡 创新点**

创新点在于引入神学分诊（Primary、Secondary、Tertiary、Pastoral 四层），结合传统意识、对比诚信与安全上报的判分维度，并通过四种系统条件（raw、guided_default、preference_configured、perspective_compare）验证结构化指令对模型行为的显著提升

**🔧 技术方法**

采用 LLM‑as‑judge 机制（三模型评审面板），合成校准（模拟多传统审稿人），以及扰动鲁棒性协议，进一步用人类专家审核验证自动评估的可靠性

**📊 数据集**

数据集为 120 条精心设计的信仰与牧养场景（共 8,792 条评测结果），覆盖 14 个不同厂商的高级模型，场景分为 Primary、Secondary、Tertiary 与 Pastoral 四类，且提供扰动变体进行鲁棒性测试

**📈 对比分析**

实验显示结构化 harness（guided_default）平均提升 3.96 分（最高 5.79 分），在 Pastoral 应用上提升 6.62 分，且提升了 10.8 分的安全上报维度；相比之下 perspective_compare 在某些场景（如 Primary、Pastoral）会降低表现；鲁棒性稳定性从 92.88 提升到 98.02

**⚠️ 局限性**

局限性包括仅限英文与基督教框架，情景为人工构造且缺乏真实用户数据；评测依赖 LLM‑as‑judge，尚未完成完整的人类校准；四种系统条件仅为可实验变量，实际部署需进一步验证

---

## 126. A Generative Framework for the Creation of Multi-Attribute Geographically-Explicit Synthetic Population

**arXiv ID:** 2608.12768 | [PDF](https://arxiv.org/pdf/2608.12768v1)

**作者:** Jinlin Wu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Na Jiang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一套基于扩散模型的分层生成框架，用于在全美范围内生成包含年龄、性别、就业、教育、收入等五个属性且具有居家与工作地理坐标的合成人口。

**💡 创新点**

创新点在于：① 采用两阶段分层扩散模型，先生成粗粒度联合分布再细化，解决高维联合分布训练难题；② 通过聚合级数据（PUMA级人口属性比例）与空间信息（POI、通勤流、道路网络）构造条件向量，提升空间非平稳性保持；③ 生成过程可迁移到未见地区，具备较强的泛化能力。

**🔧 技术方法**

技术包括：离散时间高斯去噪扩散概率模型（DDPM）用于联合分布生成；多层感知机（MLP）对多源条件向量编码；比例拟合（IPF）和迭代比例调节用于个体到区域内的精细分配；GIS空间分析与位置赋值算法。

**📊 数据集**

使用的数据集有：2023年美国社区调查（ACS）5年期公共使用微数据样本（PUMS）用于训练目标联合分布；ACS 5年详细表（聚合级人口属性）用于构造条件向量；OpenStreetMap道路网络、POI数据、LODES通勤流用于空间表示与位置分配。

**📈 对比分析**

与基线方法（IPF、组合优化CO、一阶段DDPM）在保留联合分布（TVD）和单属性边际分布（AAD）上进行对比。实验表明，所提框架在全美训练集上的平均TVD为0.116，低于IPF(0.127)、CO(0.127)和单阶段DDPM(0.147)，相对降低约6%和19%；在保留边际分布上误差极低（≤0.004）。在留出州Michigan的泛化实验中，TVD平均为0.119，进一步证明泛化能力。

**⚠️ 局限性**

局限性包括：① 仅使用PUMA级聚合数据，导致年龄被分成宽范围区间，缺乏精细年龄信息；② 对18岁以下人员缺失日间活动地点，无法完整模拟学生流动；③ 位置分配依赖现有道路、POI和通勤流数据，若空间数据不足或不精细，可能影响定位准确性。

---

## 127. The Cost of Changing Edges for Diameter Computation and More

**arXiv ID:** 2608.12628 | [PDF](https://arxiv.org/pdf/2608.12628v1)

**作者:** Sam Hiken `[一作]`, Virginia Vassilevska Williams `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在敏感度（sensitivity）设置下对图的直径、球心与度数的精确与近似计算，特别是对单条边的增删改问题。

**💡 创新点**

创新点在于：1) 用紧凑的归约将任意图的距离敏感度树 (DSO) 转化为精确或 (1+ε) 近似的降级直径/球心解法，匹配静态算法的预处理时间；2) 给出增量设置下的强上界下限，证明了在无向图中无法获得小于 5/3 的近似，且在有向图中无法小于 2；3) 引入两种新技术（基于“canonical shortest paths”和“三角形引理”）实现多种新的增量近似算法；4) 通过“sample”与“δ-k”框架实现增量球心的快速查询。

**🔧 技术方法**

主要技术包括：DSO 与 canonical shortest paths 的组合、基于矩阵乘法的特殊产品 (min‑max) 计算、稀疏采样与样本集合 (α,β,k)-sample、递归中心分解、δ^U_k 近似预处理、三角形引理及其单调性分析、以及对低阶矩阵乘法的高效利用。

**📊 数据集**

未使用公开数据集；实验与评估主要基于理论时间复杂度比较，涉及最优稀疏图、稠密图以及随机生成图的理论上限。

**📈 对比分析**

与之前的单失败结果 (Biló 等) 对比，新算法在降级场景下实现了 O(n^2) 或更低的预处理时间和 O(1) 查询时间；在增量场景下，新算法在稀疏图上达到接近 2 的近似（O(n^{2.343}) 预处理），并在稠密图上提供 3/2+ε 近似（O(n^2.5) 预处理）。相比于旧的 O(n^3) 或 O(n^2) 方案，显著提升。

**⚠️ 局限性**

局限性：1) 对精确降级球心的实现仍需 O(n^{2.88}) 预处理，尚未达到最优 n^2；2) 增量近似在有向图中只能达到 2 近似，无法突破；3) 某些证明依赖于稀疏采样成功，概率上限仅为高概率；4) 对加权图的结果仍受 M 上界影响，可能引入常数级加性误差；5) 低阶矩阵乘法的实现复杂度高，对实际可行性有一定限制。

---

## 128. Euclidean SVP is deterministically NP-hard to approximate within any constant factor

**arXiv ID:** 2608.12664 | [PDF](https://arxiv.org/pdf/2608.12664v1)

**作者:** Daqing Wan `[一作]` `[通讯]`, Daqing Wan

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出了一个确定性多项式时间的多重归约，证明了对于任意常数ρ>1，欧几里得最短向量问题（Euclidean SVP）在ρ近似下是NP难的；并将此结果推广到所有有限ℓp范数，并给出了随维度变化的近似因子与时间复杂度的权衡；

**💡 创新点**

创新点主要在于：①构造了一个确定性、局部稠密的Reed–Solomon格子投影工具，使得在二进制向量上可实现任意预设投影；②证明了整数格子在坐标ℓ1范数下的最小值在张量积下严格相乘，填补了欧几里得范数不具乘性的问题；③利用上述乘性与二进制ℓ1间的桥梁，实现了任意常数因子的确定性硬化，并通过张量放大得到对所有有限p范数的硬化；④给出了维度相关的近似因子-时间权衡（quasi多项式与次指数归约），并在确定性框架下实现了前人随机化结果的等价或改进。

**🔧 技术方法**

核心技术包括：1）Reed–Solomon格子构造与投影定理；2）Lee重量与商码的最小Lee权重相乘性质；3）整数格子在坐标ℓ1范数下的张量乘法定理；4）基于张量积的近似因子放大；5）对非满秩格子进行全秩扩张以保持最小值不变；6）在归约中使用基矩阵的Kronecker积与数值取整。

**📊 数据集**

论文为理论性工作，未使用实验数据集；所有结果均在抽象的格子与编码对象上证明。

**📈 对比分析**

与之前的随机化结果相比，本文的确定性归约在实现上更为直接，省去了随机化和概率分析；在近似因子上实现了与随机化等价的任意常数因子；在维度相关的归约（quasi多项式、次指数）也提供了确定性版本。

**⚠️ 局限性**

局限性包括：①张量放大方法对维度增长的限制，得到的近似因子仅为n^{o(1)}，无法达到固定正指数的n^{ε}；②归约对ℓ∞范数的适用性未覆盖；③对非常大维度下的实用性与复杂度仍有进一步优化空间；

---

## 129. Transforming Interactions in Thesis Supervision: An Exposé-First Workflow in Higher Education

**arXiv ID:** 2608.12546 | [PDF](https://arxiv.org/pdf/2608.12546v1)

**作者:** Lin-Yin Huang `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了一种“exposé-first”论文准备流程，旨在让学生提前撰写研究提案，以规范与分配监督任务。

**💡 创新点**

创新点在于将exposé写作权由导师转移给学生，并显著提升行政协调工作，从而在多学位生中实现流程结构化。

**🔧 技术方法**

采用混合方法研究技术，包括NASA‑TLX工作负荷评估、UMUX可用性量表以及定性访谈与主题分析。

**📊 数据集**

使用两学期（2025夏季与2026夏季）收集的学生、导师和行政人员问卷与访谈数据；未使用公开数据集。

**📈 对比分析**

通过问卷量化与访谈主题对比，发现学生工作负荷明显高于导师，且可用性评分显示系统集成问题仍存，但整体流程提升了结构化程度。

**⚠️ 局限性**

局限在于样本规模小、单一机构、仅关注感知体验且缺乏长期论文质量评估，平台非集成导致行政工作量未被完全缓解。

---

## 130. FUSE: Active Functional Affordance Grounding through Adaptive Semantic-Geometric Evidence Acquisition

**arXiv ID:** 2608.12683 | [PDF](https://arxiv.org/pdf/2608.12683v1)

**作者:** Zhou Chen `[一作]` (Auburn University), Sathyanarayanan N. Aakur `[通讯]` (Auburn University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的任务——主动功能性可供性定位（Active Functional Affordance Grounding），并开发了 FUSE 方案，该方案通过自适应地在显式语义‑几何探索与学习的近似规划之间切换，来实现高效地寻找并定位满足功能查询的对象。

**💡 创新点**

创新点在于：①将功能不确定性建模为基于候选对象的空间不确定场；②设计了结合 SAM3 语义掩码和 SSNR 几何不确定性的采样策略；③引入 3D 高斯散射的增量空间表示，允许在多视角下持续更新场景；④提出一种基于动作熵的自适应门控机制，使得在大多数步骤下使用轻量级的学习规划，而在不确定时退回到显式搜索，从而显著降低计算成本；⑤构建了基于 Habitat 的可重复实验基准，提供了全面的评价指标。

**🔧 技术方法**

使用技术包括：SAM3 语义掩码生成、SSNR 结构信噪比估计、3D Gaussian Splatting 空间表示、CLIP + MLP 学习的动作价值预测、基于熵的自适应决策、以及 COLMAP 的相机位姿注册。

**📊 数据集**

数据集为 Habitat 桌面场景的 100 个随机实例（共 6 种不同的场景配置），每个实例包含 1–2 个目标物体和 7–14 个干扰物，初始视角随机且至少有一个目标被遮挡。

**📈 对比分析**

与被动定位、随机主动、VLM 主动、显式探索、近似规划等方法相比，FUSE 在成功率（72%）和平均 IoU（70.91%）上获得最佳表现；其平均步骤数仅为 9.32 步，显著低于显式探索的 8.30 步；相比显式探索，FUSE 在计算时间上提升约 1.33×，仅在约 33% 的决策中调用显式搜索，整体实现了更优的准确率–计算效率折衷。

**⚠️ 局限性**

局限性包括：①对候选对象生成器高度依赖，若知识源提供的候选集合不足或错误，性能会显著下降；②在动态场景或更大规模的现实环境中仍未验证；③虽然降低了显式搜索次数，但在最坏情况下仍需多次 3DGS 更新；④实验仅在仿真环境中完成，真实机器人平台的鲁棒性与实时性尚待进一步评估。

---

## 131. On the Expressive Power of Transformers

**arXiv ID:** 2608.12671 | [PDF](https://arxiv.org/pdf/2608.12671v1)

**作者:** Phokion Kolaitis `[一作]` (University of California Santa Cruz), Rik Sengupta `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了Transformer作为语言识别器的形式表达力，并将其与电路复杂度和逻辑表达力相联系。

**💡 创新点**

首次将Transformer的注意力机制与无穷幅宽、固定深度的电路类进行对应，揭示了不同资源（层数、宽度、精度、链式推理等）对表达力的影响。

**🔧 技术方法**

利用电路模拟、描述性复杂度、逻辑（FO、LFP、B‑RASP）等理论工具，证明Transformer在不同参数设定下落入AC⁰、TC⁰、DTIME等复杂度类。

**📊 数据集**

无实验数据集，本工作纯理论分析。

**📈 对比分析**

通过对Transformer的层级参数化与电路类进行等价映射来比较；结果表明在O(1)精度下只属于AC⁰，在O(log n)精度下可达TC⁰，链式推理可突破到DTIME和RE，具体性能以复杂度上界/下界来衡量。

**⚠️ 局限性**

局限在于：理论结果对训练好的实际模型适用性有限；某些包含关系是否严格仍未决定；链式推理模型需预先给定长度，真实模型未满足；以及对精度、位数等假设与实际实现的差距。

---

## 132. From Refuse to Richness: Rubric Rewards for Long-Form Hallucination Reinforcement Learning

**arXiv ID:** 2608.12337 | [PDF](https://arxiv.org/pdf/2608.12337v1)

**作者:** Yudong Wang `[一作]` (Peking University), Zhifang Sui `[通讯]` (Peking University)

**通讯引用:** 5150 | [OpenAlex ID](https://openalex.org/A5110285832)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在长文本生成中，通过结合基于关键信息检查表的奖励来平衡模型的真实性与信息覆盖度，并比较不同奖励组合在分布内外的表现。

**💡 创新点**

提出使用每样本关键信息检查表作为细粒度覆盖度信号，和软性组合地同时优化真实性与信息完整性，解决传统全覆盖奖励导致的内容缺失或冗余问题。

**🔧 技术方法**

采用RL（GRPO）训练，奖励函数包括真实性（grounding）、细粒度覆盖度、对齐相关性等，使用GPT-5.4、Gemini等模型生成检查表。

**📊 数据集**

训练使用4000条提示，包含2000条有上下文的grounding prompt和2000条无上下文的open prompt，评估使用FACTS Grounding、LongFact、FActScore、CL-bench、Arena‑Hard creative‑writing等基准。

**📈 对比分析**

在Qwen3-4B和DeepSeek-R1-Distill-Llama-8B上比较五种奖励，发现纯真实性奖励提升 grounding 但压缩内容，纯检查表奖励提高覆盖度但削弱 grounding，软性组合 FACT‑RUBRIC‑REL 在保持相对较高 grounding 的同时显著提升 OOD 任务表现。

**⚠️ 局限性**

实验仅覆盖两个相对小型的开源模型，未验证在更大或专有模型上的效果，且对检查表生成质量与可扩展性缺乏深入分析。

---

## 133. Represent, Then Generate: Multimodal-Conditioned Time-Series Generation under Irregular Missingness

**arXiv ID:** 2608.12592 | [PDF](https://arxiv.org/pdf/2608.12592v1)

**作者:** Haochen Zhang `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出两阶段框架Represent Then Generate，先用每模态掩码自编码器提取缺失容忍的token，再用冻结编码器的条件流匹配生成器生成目标时间序列。

**💡 创新点**

将条件表示与生成解耦，利用多模态掩码自编码器生成缺失容忍token，跨模态条件注入采用交叉注意力与AdaLN双路，解决多模态异构、非规则缺失的生成问题。

**🔧 技术方法**

使用每模态掩码自编码器（Transformer Encoder-Decoder）、冻结编码器后的流匹配生成器（DiT/SiT vision transformer + 流匹配）、延迟嵌入将时间序列映射为二维图像，以及交叉注意力和AdaLN条件注入。

**📊 数据集**

在AI-READI（可穿戴+CGM）、MIMIC-III与MIMIC-IV（ICU波形+静态特征）三大数据集上进行时间序列生成任务。

**📈 对比分析**

与六种代表性条件生成器（Diffusion-TS、ImagenTime、VerbalTS、Bridge、TimeWeaver、WaveStitch）在16个数据集/任务/指标上采用下游预测（AUROC/AUPRC）对比，本文模型在所有指标上均优于基线，13/16甚至超过真实信号基准。

**⚠️ 局限性**

下游评估混合了目标保真与条件重表达，难以分离；每模态独立编码未捕获跨模态依赖；需要进一步对齐生成与真实信号以分离鲁棒性与重表达。

---

## 134. Designing AI Pipelines for Decision-Ready ITSM Intelligence

**arXiv ID:** 2608.12670 | [PDF](https://arxiv.org/pdf/2608.12670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 135. DIVE: Unlocking Self-Improvement in Frozen Language Models Through Diversity-Driven Skill Evolution

**arXiv ID:** 2608.12486 | [PDF](https://arxiv.org/pdf/2608.12486v1)

**作者:** Siheng Xiong `[一作]` (Georgia Institute of Technology), Faramarz Fekri `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多样化驱动的自然语言技能进化框架，使冻结的 LLM 能在不更新参数的情况下，通过经验和验证器反馈逐步改进推理能力。

**💡 创新点**

创新点在于：①独立演化多组技能种群；②使用上置信界（UCB）自适应分配演化算子；③基于进化历史自动生成新算子；④在推理时联合选择互补的技能集合。

**🔧 技术方法**

技术方法包括：自然语言技能进化、上置信界动态算子选择、适应性算子生成、联合技能选取、基于二元验证器的评估与反馈。

**📊 数据集**

实验数据集涵盖六类数学与逻辑推理任务：HMMT、Equational Theories、Sudoku、Cryptarithm、Calcudoku 与 Futoshiki。

**📈 对比分析**

与提示优化、记忆式、技能式、参数调优（SFT/GRPO）以及 Prompt‑Opt 等基线相比，该方法在所有任务上均表现更优，且在回合数与推理成本上更具竞争力；小模型通过技能进化可匹敌甚至超越大模型。

**⚠️ 局限性**

局限性包括：依赖可靠的二元验证器；需要多次经验回合，难以在极大规模数据或连续学习场景下高效扩展；目前仅在推理任务上验证，跨任务迁移和长期记忆机制尚待深入。

---

## 136. Alignment Drift in Single-Model Speculative Decoding for ASR: Mechanism, Correction, and Cost

**arXiv ID:** 2608.12703 | [PDF](https://arxiv.org/pdf/2608.12703v1)

**作者:** Xinyu Wang `[一作]` (Boson AI), Alex Smola `[通讯]` (Boson AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究针对单模型自回归语音识别中的speculative decoding，提出了轻量级draft模块并分析了其在音频位置跟踪失准（alignment drift）导致的后续接受率下降问题，并通过两种纠正方法（runtime correction 与 training correction）提升速度与准确率。

**💡 创新点**

创新点在于把位置监督嵌入draft的训练和推理过程中，首次证明音频位置漂移是导致speculative ASR性能衰退的主要因素，并提供基于验证注意力的实时定位修正方案。

**🔧 技术方法**

主要技术包括单模型speculative decoding框架、音频位置（anchor）估计与监督、基于注意力的定位读出、训练时的高斯引导注意力损失，以及对比实验中的深度残差和层级probe分析。

**📊 数据集**

实验数据集涵盖LibriSpeech（clean/other）、TED‑LIUM、GigaSpeech 与 FLEURS，模型为Qwen3‑ASR（0.6B、1.7B）与 Voxtral‑Mini（3B）等。

**📈 对比分析**

与纯自回归解码比较，经过位置纠正后，1.7B模型在五个评估集上平均速度提升约1.3×，WER保持在0.01百分点以内；在Voxtral‑Mini上，速度提升约1.4×但位置误差对续航影响有限。

**⚠️ 局限性**

局限性包括：仅在greedy验证且低精度实现下实验；对位置监督的依赖限制了跨架构/批量/长语音场景的推广；未能完全消除所有拒绝提案，仅定位漂移导致的部分失效。

---

## 137. Personalized Scorer Modeling: A Learning-Based Framework for Deriving Robust Sleep Stage Labels from Multiple Experts

**arXiv ID:** 2608.12446 | [PDF](https://arxiv.org/pdf/2608.12446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 138. Dead text or binding clause? Measuring and restoring constraint influence in black-box LLM dialogues

**arXiv ID:** 2608.12599 | [PDF](https://arxiv.org/pdf/2608.12599v1)

**作者:** Haoyuan Zhu `[一作]` `[通讯]` (University of Sheffield), Haoyuan Zhu (University of Sheffield)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究多轮对话中撤销约束后模型仍执行的行为性复发现象，并提出基于合约账本的测量与修复框架。

**💡 创新点**

创新点包括：引入合约账本记录约束启用/撤销，先验编译净约束状态，设计序贯探测器预测复发，提出分层修复阶梯并评估单句墓碑注记的实效。

**🔧 技术方法**

使用技术包括：合约账本、可执行检查器、科学模式消融、序贯探测、修复阶梯（预填、重写、回退）以及预算匹配分析。

**📊 数据集**

实验数据集基于 HumanEval Python 任务切片，插入标记约束并控制撤销脚本。

**📈 对比分析**

与无账本回退基线、编译后状态、墓碑注记、修复阶梯等方案进行匹配预算比较，在8B模型上复发率从约0.4下降至0.12，编译显著降低复发；修复阶梯未带来可检测的额外提升。

**⚠️ 局限性**

局限性在于仅针对Python编码任务、单个8B模型、单一语言、短对话深度、并未覆盖随机种子重现等情形。

---

## 139. Dual-Flow Transformers: Decoupling the Primary Prefill Path from Additional Decode Computation

**arXiv ID:** 2608.12385 | [PDF](https://arxiv.org/pdf/2608.12385v1)

**作者:** Liming Liu `[一作]` (Georgia Institute of Technology), Tuo Zhao `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 2658 | [OpenAlex ID](https://openalex.org/A5101595500)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dual-Flow 架构，将提示（prefill）和自回归解码（decode）阶段的计算分离，主流路径只处理提示并写入 KV 缓存，而辅助路径仅在解码时增加额外计算并不写入 KV，二者共享大多数权重并通过轻量级交互实现信息传递。

**💡 创新点**

创新点在于：① 通过共享 KV 缓存实现阶段间算力复用；② 引入主流与辅助两条流的非对称设计，使得在保持提示计算不变的前提下可单独调节解码计算；③ 对 MoE 模型实现 router replay，使两条流共享专家集但可独立分配专家数量；④ 采用混合 next‑token 目标（混合权重基于主流分布自适应），进一步提升性能。

**🔧 技术方法**

使用技术包括：共享注意力投影与 MLP、轻量级交互向量 (a^ℓ, b^ℓ, c^ℓ)、混合下游分布、RoPE 与 RMSNorm、FlashAttention 的分组执行、MoE 的 router replay 与专家重用，以及训练时的两流并行计算。

**📊 数据集**

数据集主要有：FineWeb（NanoGPT 训练），以及 LLaMA‑style 的 FineWeb 子集；MoE 版本基于 Qwen‑style 的 Sparse MoE 架构；实验还在 NanoGPT 的 Controlled Scaling、LLaMA‑style 大规模与 MoE 版本上进行验证。

**📈 对比分析**

与单流 Transformer 或 PHD‑2 以及标准 MoE 对比，Dual‑Flow 在匹配 token 预算下始终降低验证损失；在模型规模扩展、数据规模扩展以及 MoE 设定中，实验均显示 1–3% 的验证损失下降；在固定 prefills 或固定 decode 的专家预算实验中，表明通过在解码阶段增加专家或把专家重分配给辅助流能进一步提升质量，且在保持总解码算力不变的情况下可实现预填算力的显著削减。

**⚠️ 局限性**

局限性包括：训练时两流计算几乎翻倍，导致训练 FLOPs 与激活内存几乎增加一倍；实验仅在单机 GPU 上评估，未展示真实推理延迟与吞吐量的收益；KV 缓存仍为单一持久化状态，无法支持多流并行写入；并且对极大上下文长度的推理性能提升尚未充分验证。

---

## 140. SoK: From Generation to Consumption of Privacy Documents in Software Systems

**arXiv ID:** 2608.12511 | [PDF](https://arxiv.org/pdf/2608.12511v1)

**作者:** Shidong Pan `[一作]` (Columbia University), Sepideh Ghanavati `[通讯]` (University of Maine)

**通讯引用:** 1335 | [OpenAlex ID](https://openalex.org/A5072117004)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2010‑2025年关于隐私文档的290篇论文进行系统综述，从生命周期视角统一研究。

**💡 创新点**

首次将隐私文档的定义、生成、分析、合规与可用性等阶段归纳为四大研究方向，并提出15项研究趋势与21项开放机会。

**🔧 技术方法**

采用系统文献检索、前向后向雪球、开放编码与主题归类等方法构建SoK框架。

**📊 数据集**

主要基于检索得到的290篇学术论文作为数据来源；未使用传统公开数据集。

**📈 对比分析**

通过对比不同研究问题的论文数量与方法，展示了研究热点与空白；未给出具体性能指标。

**⚠️ 局限性**

仅检索标题/摘要、仅英文、部分期刊被排除，可能漏检新兴隐私文档类型与跨语言研究。

---

## 141. From Relational and Property Graph Data to Large Language Models

**arXiv ID:** 2608.12407 | [PDF](https://arxiv.org/pdf/2608.12407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 142. PolyPresentation: A Multimodal AI Platform for Slide-Aware Iterative Presentation Practice

**arXiv ID:** 2608.12857 | [PDF](https://arxiv.org/pdf/2608.12857v1)

**作者:** Chen Chen `[一作]` (Hong Kong Polytechnic University), Jiannong Cao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个多模态AI平台PolyPresentation，用于支持幻灯片感知的迭代演讲练习，集成了逐幻灯片练习、完整排练、模拟问答、证据驱动反馈与幻灯片改进四大功能模块。

**💡 创新点**

创新点包括：①基于幻灯片上下文的反馈循环，实时将排练证据与幻灯片内容对齐；②利用幻灯片-排练对齐的证据生成具体、可操作的反馈；③将反馈直接映射到幻灯片改进方案，形成闭环的实践迭代；④在多模态（语音、视觉、文本）证据处理与评估中引入基于Rubric的量化标准。

**🔧 技术方法**

技术手段主要是：多模态感知（ASR、VLM、LLM GPT‑5）、实时LLM提示、幻灯片文本与视觉特征解析、时间戳对齐、基于规则与LLM的反馈生成、数字化观众模拟，以及基于Rubric的自动评分与报告生成。

**📊 数据集**

使用的数据集为20份学术会议演讲排练样本，包含幻灯片deck、实时语音转写、屏幕/视频捕获、幻灯片切换时间戳及模拟问答记录；此外在对比实验中使用了PresentCoach、Gemini‑3.5 Flash、GPT‑5 CHOP与规则基准等模型产生的多模态证据。

**📈 对比分析**

评价方法：①与人工评分对齐，计算Pearson r、Quadratic Weighted Kappa、ICC(2,1)和MAE；②与四个基线系统在7维度（有效性、覆盖度、影响力、深度、可操作性、转移性、组织性）上进行加权总分对比。PolyPresentation总体得分7.54/10，覆盖度、深度、转移性分别领先他人+2.13、+1.37、+1.31，且在18/20个样本中排名第一，说明其在提供可操作且全局性反馈方面表现最佳。

**⚠️ 局限性**

局限性：仅评估20份单轮排练样本，未验证多轮迭代效果；反馈质量评判依赖GPT‑5，可能导致评估偏倚；未对系统鲁棒性、实时延迟和跨场景泛化进行深入探测；未来需进行盲评人工评估和多轮用户研究。

---

## 143. Arithmetic Variable LogLog: Advancing the Memory-Variance Frontier

**arXiv ID:** 2608.12575 | [PDF](https://arxiv.org/pdf/2608.12575v1)

**作者:** Brian Bushnell `[一作]` `[通讯]` (DOE Joint Genome Institute, Lawrence Berkeley National Laboratory), Brian Bushnell (DOE Joint Genome Institute, Lawrence Berkeley National Laboratory)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了Arithmetic Variable LogLog（AVLL）算法，通过基56算术编码在64位单词中压缩11个寄存器，实现5.5倍寄存器密度；

**💡 创新点**

创新点在于利用稀有状态剔除和算术编码提高寄存器利用率，结合四组件HLDLC估计器，实现4–5%更低宽度加权误差和更低内存-方差积；

**🔧 技术方法**

采用基56算术编码、状态映射、层级估计（LDLC+Hybrid+2）、校正因子拟合以及早期退出机制；

**📊 数据集**

使用模拟高复杂度（全唯一）和低复杂度（高重复）两种数据流进行评估；

**📈 对比分析**

在等内存条件下与ExaLogLog、UltraLogLog等七种估计器对比，宽度加权误差平均低4–5%，MVP约3.4，且在多线程高缓存压力场景下吞吐量比ExaLogLog快2.7–4.5倍；

**⚠️ 局限性**

缺点包括算术除法导致的寄存器访问周期高、非幺正性导致合并时可能出现溢出，以及在低卡路里时的写入压力稍高。

---

## 144. HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments

**arXiv ID:** 2608.12860 | [PDF](https://arxiv.org/pdf/2608.12860v1)

**作者:** Quan-Dung Pham `[一作]` (VinMotion), Quan Nguyen `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 NVIDIA Isaac Sim 上构建了一个支持四种不同形态 humanoid 机器人的物理仿真平台，生成了 87 个可导航场景和 933 条包含四种风格（Fine、Formal、Natural、Casual）的指令，旨在为视觉语言导航（VLN）模型提供真实的物理执行评估。

**💡 创新点**

创新点包括：① 面向多形态 humanoid 的层次化 RL+PD/MPC 控制架构；② 采用 3D Gaussian Splatting（3DGS）重建并筛选可导航区域的场景制作流程；③ 引入双生成-审核+改写多智能体指令生成框架（Dual Generator‑Reviewer + Paraphraser MAA），并结合人工校验提升指令的空间真实性。

**🔧 技术方法**

技术手段包括 NVIDIA Isaac Sim 物理仿真、基于 RL 的行走策略、PD/MPC 路径跟踪、3DGS 场景重建、VLM（Qwen、Gemma、InternVL）驱动的生成-审核过程，以及人工标注验证。

**📊 数据集**

数据集涵盖 87 个高质量可导航室内场景（艺术家手工和 3DGS 重建），共 933 条 episode，每条 episode 产生一个细粒度指令和三种风格变体；场景分布覆盖 17 种室内类别，平均可导航面积 266 m²。

**📈 对比分析**

评估采用四个主流 VLN 模型（NaVILA、StreamVLN、DualVLN、JanusVLN）在四种机器人（Unitree G1、Unitree H1、Internal‑A、Internal‑B）上进行零样本测试。JanusVLN 在 Success Rate（43.55%）和 nDTW（48.38）上最高，DualVLN 在路径一致性上表现最佳；Unitree H1 的 Fall Rate 高达 71%，显示形态与控制对性能影响显著。Sim‑Real 试验中，导航误差与真实环境高度相关（r=0.935，平均差 0.68 m）。

**⚠️ 局限性**

局限性包括：① 场景多样性不足，缺乏大规模多域覆盖；② 指令生成仍需人工校验，难以规模化；③ 物理仿真成本高，限制了大规模训练与评估。

---

## 145. Demand Transfer Estimation at Scale via Restricted Logit Modeling

**arXiv ID:** 2608.12680 | [PDF](https://arxiv.org/pdf/2608.12680v1)

**作者:** Lakshya Garg `[一作]` (Walmart Global Tech), Anupriya Sharma `[通讯]` (Walmart Global Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过限制Logit模型结合替代关系估计大规模商品库的需求转移系数，从而改进商品配置预测

**💡 创新点**

提出了“Restricted Logit”方法，将多项式Logit模型与可替换性得分结合，兼顾重叠需求状态且可在百万级商品上高效运行

**🔧 技术方法**

采用多项式Logit（MNL）模型、替代得分（Store Yule's Q、SBERT相似度）以及Spark并行化的Python实现

**📊 数据集**

利用Walmart门店历史交易数据、缺货与计划库存数据，构建购买矩阵和可用性矩阵，覆盖数十类商品、数十个门店

**📈 对比分析**

与传统全MNL/马尔可夫链方法对比，在50个示例门店中，调整后wMAPE平均下降10–20%，尤其在高基准误差门店表现更显著

**⚠️ 局限性**

受限于IIA假设、需要手动设定替代阈值（0.6），对不易定义需求状态的商品或极少交易商品的估计不够精确

---

## 146. StorySpark: Module-wise Evolutionary Search for Story Premise Generation

**arXiv ID:** 2608.12336 | [PDF](https://arxiv.org/pdf/2608.12336v1)

**作者:** Yang Yang `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Institute of Deep Perception Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出StorySpark，基于模块化进化搜索的故事前提生成框架，能在保持可解释性的前提下提升前提原创性与质量。

**💡 创新点**

创新点是将前提生成视作局部进化搜索，引入前缀条件演化、Pareto多目标选择与预留-通配前沿分配。

**🔧 技术方法**

使用LLM生成候选、基于评分的变异与交叉、Pareto非支配排序、前沿容量分配等技术。

**📊 数据集**

采用MoPS四teen域数据集以及DOC、WritingPrompts、Storium等外部前提素材进行评测。

**📈 对比分析**

与MoPS、VIL、CPX、DOC等方法对比，StorySpark在自动评分与人类评价中均获得最高总体得分，尤其在原创性上提升4.4分，且在故事转化和对比评价中保持领先。

**⚠️ 局限性**

局限在固定的模块化结构、缺乏动态模块顺序调整与向下反馈，且仍依赖自动评判，未来需加入人工反馈与更灵活的结构。

---

## 147. Erase but Preserve: Controllable Removal of Copyrighted Animation Characters via Optimized Semantic Anchors

**arXiv ID:** 2608.12806 | [PDF](https://arxiv.org/pdf/2608.12806v1)

**作者:** Qiao Li `[一作]` (Chinese Academy of Sciences), Jizhong Han `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于优化语义锚点的可控动画角色抹除方法，在扩散模型生成过程中对目标角色嵌入进行替换，从而实现无侵权生成。

**💡 创新点**

创新点在于在连续文本嵌入空间中学习结构相似且细节对比的锚点，并结合结构感知自适应替换策略，实现对动画角色的精准抹除与高保真度保持，同时支持细粒度控制、多目标抹除和跨模型迁移。

**🔧 技术方法**

采用文本编码器‑跨注意力的扩散模型、结构约束与细节对比的锚点优化、低频结构分析的自适应嵌入替换以及与现有模型改造方法的融合技术。

**📊 数据集**

构建了包含80个可被扩散模型生成的动画角色（人形、动物形、杂项）数据集，并在此基础上生成测试图像。

**📈 对比分析**

与五种基于提示调度的抹除基线（SLD、STG、SAFREE、TraSCE、Negative Prompt）以及四种模型改造基线（MACE、UCE、ESD‑u、AC）比较，实验显示本方法在目标识别率最低（6.0%/4.0%）、SSIM最高（0.467）、LPIPS最低（0.505），且在不同模型间保持迁移性，对非目标图像质量影响微乎其微。

**⚠️ 局限性**

局限性包括需要为每个角色单独优化锚点，对极度独特或极少见角色的锚点学习仍有挑战；在极高抹除强度下可能出现背景细节失真；仅针对文本到图像扩散模型，未验证在其他生成框架中的效果。

---

## 148. Attribute-Conditioned Multimodal Slot Factorization for Controllable Fashion Retrieval

**arXiv ID:** 2608.12570 | [PDF](https://arxiv.org/pdf/2608.12570v1)

**作者:** Najmeh Forouzandehmehr `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MM-slotgate，一个多模态槽编码器，实现时尚检索中属性可控和精准匹配。

**💡 创新点**

创新点在于为每个命名属性（图案、颜色、类别、人口）学习可调的文本-图像门控，实现属性条件的模态融合与可解释的门控学习；同时提供量化槽码的直接编辑功能。

**🔧 技术方法**

采用Fashion-CLIP双模态特征、可学习的门控、EMA向量量化、端到端监督（交叉熵对齐+承诺损失）以及正交约束。

**📊 数据集**

在H&M 5万条商品的子集上评估，包含商品文本、图像及四个属性标签。

**📈 对比分析**

与基线（CLIP文本、fCLIP文本、全局均衡融合）对比，MM-slotgate在Macro ConstraintSatisfied@10上提升至0.7566（比全局融合+5.9%，比文本仅+59.1%），在颜色约束上最大提升从0.321到0.889；门控分析显示颜色更依赖图像，类别/人口更依赖文本；量化槽码干预时颜色提升15.3× HitRate@10。

**⚠️ 局限性**

局限在于仅涵盖四个属性，扩展到更多时尚概念（场合、风格、品牌等）仍待研究；门控提升有限，需进一步探索跨属性互补与更丰富的模态融合策略。

---

## 149. Mr3D-VL: A generalist vision language foundation model for Multiparametric 3D Magnetic Resonance Imaging

**arXiv ID:** 2608.12689 | [PDF](https://arxiv.org/pdf/2608.12689v1)

**作者:** Zhi Qiao `[一作]` (United Imaging Intelligence), Feng Shi `[通讯]` (United Imaging Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了一个面向多参数3D MRI的视觉‑语言基础模型 Mr3D‑VL，支持自然语言查询、报告生成等临床交互功能。

**💡 创新点**

创新点：① 用 LLM 生成高质量多模态图像‑文本、问答对，克服数据稀缺；② 共享 3D CNN 编码器并采用无监督 DINO‑v2 预训练，减少参数并提升跨模态捕捉；③ 设计 4D 旋转位置编码 (4D‑RoPE) 直接建模 3D 空间关系；④ 多分辨率特征植入策略，使 ViT 在 LLM 中逐层融合全局与局部信息；⑤ 本土多模态预训练框架（S0→S1→S2）实现从单模态到多模态的逐步迁移；⑥ 仅 4B 参数即可实现高效推理。

**🔧 技术方法**

技术手段：CNN‑based 3D ConvNeXt + DINO‑v2 无监督预训练；4D‑RoPE 位置编码；ViT 多分辨率特征植入；Qwen3‑4B‑Instruct 作为 LLM 主干；LLM 驱动的数据生成与重写；三阶段预训练（S0‑S1‑S2）+ 监督微调（SFT）；多任务评测（VQA、报告生成）。

**📊 数据集**

数据集：103,605 例脑多参数 MRI（460,541 张图像，8 种序列），通过 LLM 生成多模态图像‑文本、问答、报告等；涵盖 T1、T2、FLAIR、DWI、PWI、SWI 等；采用公开数据与自研标注。

**📈 对比分析**

对比方法：与通用 VLM（Qwen3.5 系列）及医疗 VLM（Hulu‑Med、Lingshu）在报告生成和 VQA 任务上对照；在报告生成中 BERTScore 0.856、METEOR 0.496、CIDEr 0.655、BLEU‑4 0.169；VQA 开放式准确率 0.713、MCQ 0.912；整体提升 20–30% 以上，显著优于基线。

**⚠️ 局限性**

局限性：目前仅在脑 mpMRI 训练，扩展至其他部位需更多数据；LLM 生成的文本可能出现幻觉或偏差；缺乏多轮对话、主动澄清及人类反馈强化学习；对更高模态维度和跨中心临床验证的泛化能力待进一步评估。

---

## 150. Dual-Stream Cross-Anchor Correction Grounding Long-Form Captions and the Domain Limits of Object-Level Anchors

**arXiv ID:** 2608.12746 | [PDF](https://arxiv.org/pdf/2608.12746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 151. Structure-preserving uncertainty quantification for GENERIC dynamics

**arXiv ID:** 2608.12624 | [PDF](https://arxiv.org/pdf/2608.12624v1)

**作者:** Zequn He `[一作]` (University of Pennsylvania), Celia Reina `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了结构保持的知识论网络（S-PENNs），在具有硬编码物理约束的科学机器学习模型中实现不确定性量化。

**💡 创新点**

通过在每个结构保持的生成块上附加轻量级epinet，并在再参数化中保持能量守恒和熵增，保证每一次采样都满足热力学一致性，并结合分割式合规预测提供分布无关的覆盖保证。

**🔧 技术方法**

采用GENERIC框架与N-GENNs基础网络，使用epinet、PICNN、分割式合规预测等技术，在训练时分两阶段：先拟合基网络，再优化epinet以注入不确定性。

**📊 数据集**

利用三组合成数据集验证：1）热浴耦合谐振子（ODE）；2）理想化化学马达（ODE）；3）一维粘塑性PDE（Field-valued）。

**📈 对比分析**

与Deep Ensembles和MC Dropout对比，S-PENNs在保持热力学一致性、覆盖率、CRPS、ES等指标上优于MC Dropout，性能与Deep Ensembles相近，但计算成本降低1–3个数量级。

**⚠️ 局限性**

仅在无噪声测量、低维例子中验证，噪声状态处理和更高维系统的可扩展性尚待进一步研究，epinet方法在更大模型中的效果仍需评估。

---

## 152. Does It Render Everywhere? A Study of Cross-Environment Compatibility in MLLM-Generated Webpages

**arXiv ID:** 2608.12518 | [PDF](https://arxiv.org/pdf/2608.12518v1)

**作者:** Ziyun Guo `[一作]` (Singapore Management University), Yintong Huo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建WebCompat数据集，系统研究 AI 生成网页在不同浏览器与设备上的兼容性，并提出 XCompat 检测工具。

**💡 创新点**

首次把跨环境兼容性纳入 AI 前端评估维度，构建大规模对比数据集与失效症状分类，设计低成本离线检测器。

**🔧 技术方法**

使用多模态大型语言模型、UI‑to‑code 工具、BrowserStack 渲染、手工标注、DOM 与截图联合分析的离线检测算法，以及 GPT‑5.5/Opus 等 LLM 基线。

**📊 数据集**

WebCompat（2,032 个渲染对），来源于 8 款生成工具（GPT‑4/5、Gemini‑3‑Flash、v0、Cursor、DCGen 等）以及 DCGen、Design2Code‑Hard 等原始页面；亦对比 60 条人工实现样本。

**📈 对比分析**

与 RedeCheck 及两种 LLM（DOM 仅与 DOM+截图）进行对比。XCompat 在测试集上 F1 达 0.903，准确率 0.953，显著高于任何基线；运行时 0.127 s/对比，零 API 费用。

**⚠️ 局限性**

仅评估静态单页，缺乏交互/后端场景；提示设计未针对兼容性；标注主观性与设计意图缺失可能影响结果。

---

## 153. Knowledge Synthesis Review Framework: Task-Level Benchmarking of LLM-Based Systems for Multi-Source Evidence Synthesis

**arXiv ID:** 2608.12741 | [PDF](https://arxiv.org/pdf/2608.12741v1)

**作者:** Wafa Shafqat `[一作]` (Toronto Metropolitan University), Steven N. Liss `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 Knowledge Synthesis Review（KSR）框架，对四类 LLM（GPT‑5、Claude Sonnet 4、Gemini 2.5 Pro、NotebookLM）在证据合成的四个核心任务（筛选、提取、分析、综合）上进行人机交互评估，构建了包含 1,893 篇跨学科、多源（学术、行业、政策、媒体）文献的语料库，并在 244 篇基准集上进行系统性比较和路由实验。

**💡 创新点**

创新点在于：① 将证据合成拆分为可测量的四个认知任务并为每个任务建立评估标准；② 对多种 LLM 在同一语料上进行任务级基准，形成可持续更新的“路由表”；③ 引入共享检索增强生成（RAG）层，使不同模型在同一知识基础上执行；④ 采用人机混合监督、可审计的工作流，保障科学性与透明度。

**🔧 技术方法**

技术包括：大语言模型（GPT‑5、Claude Sonnet 4、Gemini 2.5 Pro、NotebookLM）；检索增强生成（RAG）+向量数据库 ChromaDB；多轮 prompt 设计与标准化输出格式；评估指标：筛选（精确度、召回、特异性、F1、准确率）、提取（字段级匹配）、分析与综合（六维量表+平均分）。

**📊 数据集**

数据集：1,893 篇 AI 与劳动力市场相关文献，涵盖研究论文、行业报告、政策简报、新闻/博客；其中 244 篇（103 论文、30 行业报告、30 政策简报、81 媒体/博客）被抽样用于黄金标准评估，确保跨源类型覆盖。

**📈 对比分析**

比较方法：人工专家对 244 篇文献进行金标准标注（κ=0.80），随后将四模型产生的输出匿名化并随机排列，使用统一评估 Rubric 对比。结果显示：Claude Sonnet 4 在筛选精度最高；GPT‑5 召回率最高；提取方面所有模型标题/来源匹配 >90%，但作者/参考字段易失误；分析与综合均低于提取，未出现任何模型在所有任务上统治。不同源类型（RPs、IRs、PBs、NMBs）对模型表现影响显著。

**⚠️ 局限性**

limitations: 仅覆盖英文文本，缺少多语言与非文本证据；基准截至 2025 年，模型迭代可能导致结果变动；未对端到端与单一模型基线的效率/成本进行量化；评估主要基于专家共识，缺乏客观量化的可靠性指标；未深入探讨模型对敏感性、偏见或不确定性处理的影响。

---

## 154. Visibility Asymmetry: How Vendor Attention Shapes Which EdTech Breakdowns Become Product-Visible

**arXiv ID:** 2608.12353 | [PDF](https://arxiv.org/pdf/2608.12353v1)

**作者:** Lucan Li `[一作]` `[通讯]` (Unaffiliated), Lucan Li (Unaffiliated)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过回顾性质性研究，定义并阐释了“可见性不对称”这一概念，并基于中国 K–12 教育技术供应商部署的案例分析其四阶段机制。

**💡 创新点**

创新点在于把故障的可见性视为由组织渠道决定的“路由”问题，提出可见性不对称的分析框架，从采购、人员配置、升级渠道到仪表盘再编码四个层面揭示了故障转化为产品可见问题的路径不均衡。

**🔧 技术方法**

采用访谈、课堂观察、现场笔记等质性方法进行案例研究，并进行概念归纳与对照分析。

**📊 数据集**

数据集包括 11 份访谈、5 次课堂观察及 28 天以上的现场笔记，覆盖 8 所学校，涉及教师、校管理员、现场技术支持以及供应商内部管理人员。

**📈 对比分析**

方法上不做数值比较或性能评估，而是通过对不同“关注层级”学校的故障可见性案例进行对比，展示同一类故障在不同渠道下产生的可见性差异，呈现四阶段机制的理论模型。

**⚠️ 局限性**

局限性：研究基于单一供应商的回顾性质性样本，缺乏对故障可见性率的量化估计；作者在供应商内部的身份可能影响可见性判断；薄层覆盖案例仅为边界情况，未能系统验证所有低关注学校的普适性。

---

## 155. Are you Talking Logic to Me? Assessing Language Models Syllogistic Reasoning Capabilities

**arXiv ID:** 2608.12374 | [PDF](https://arxiv.org/pdf/2608.12374v1)

**作者:** Hanna Abi Akl `[一作]` (Université Côte d’Azur), Pierre Monnin `[通讯]` (Université Côte d’Azur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 CLGC 框架，将 FOL 逻辑三段论自动转换为多种正式知识表示（KR）符号，并在小型语言模型上评估其推理能力。

**💡 创新点**

首次系统探究多样 KR 符号对三段论推理的影响，提出 SEF 分类方法，发布 FOLIO‑KR 与 P‑FOLIO‑KR 数据集及 Python 库。

**🔧 技术方法**

使用神经符号提示技术（SFT 与 ZS）、BNF 语法、SEF 描述以及 Flan‑T5、Gemma、Llama、Phi 等小型解码模型进行实验。

**📊 数据集**

扩展 FOLIO 与 P‑FOLIO 原始数据集，生成 CLIF、CGIF、CLINGO、TFLPLUS 与 MINIFOL 系列的 KR 版本（FOLIO‑KR、P‑FOLIO‑KR）。

**📈 对比分析**

在 SFT 与 ZS 场景下对比不同符号集的 F1 与 AG 指标；结果显示抽象符号（如 TFLPLUS）在小模型中可超过自然语言；模型规模提升总体表现；SEF 描述对部分模型有显著提升；简洁符号推理速度最快。

**⚠️ 局限性**

局限性包括仅针对三段论且数据集规模有限；未覆盖更大或更复杂的推理任务；依赖预训练语料对符号熟悉度；缺乏完整的多阶段神经符号推理管道。

---

## 156. $\varepsilon$-MemEvo: Adaptive Cross-Task Memory Transfer for LLM Program Evolution

**arXiv ID:** 2608.12522 | [PDF](https://arxiv.org/pdf/2608.12522v1)

**作者:** Aofan Liu `[一作]`, Yiyan Qi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大语言模型的程序进化框架，旨在实现跨任务知识转移，通过存储成功算法策略的自然语言摘要来优化不同API和评估器的任务。

**💡 创新点**

创新点在于引入了战术记忆库和自适应注入门控机制，使得系统能够在不同任务之间有效转移知识，同时避免负迁移。

**🔧 技术方法**

使用了大语言模型（如GPT-5和Gemini-3-Pro）进行程序生成和评估，并结合了贝叶斯抽样的自适应门控策略。

**📊 数据集**

在8个多样化的优化基准上进行评估，包括数学优化和系统工程任务。

**📈 对比分析**

与AdaEvolve等方法进行比较，结果显示在所有8个任务上，AUCC均有显著提升，平均相对增益为+8.7%，早期收敛性提高了+9.4%。

**⚠️ 局限性**

局限性在于评估仅限于两个LLM骨干，记忆库的构建在任务级别，尚不清楚随着条目数量的增加，检索精度和门控校准如何变化。

---

## 157. Matrix-Driven Quartic Overhauser (QOVR) Surfaces Structural Framework: Continuity Limitations, Computer Graphics Algorithms, and Software Implementation

**arXiv ID:** 2608.12697 | [PDF](https://arxiv.org/pdf/2608.12697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 158. ASPIRE-VINS: Adaptive Spline-based Visual-inertial Navigation System With Robust 3D Measurement Residuals

**arXiv ID:** 2608.12840 | [PDF](https://arxiv.org/pdf/2608.12840v1)

**作者:** Kwangyik Jung `[一作]`, Hyun Myung `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种连续时间视觉惯性导航系统（ASPIRE‑VINS），通过自适应结点放置、分辨率多重样条以及三维测量空间残差在 Lie 组优化框架下实现轨迹估计。

**💡 创新点**

创新点在于将运动自适应的结点分布与多分辨率 B‑样条相结合，同时提出仅依赖标定视线方向的三维残差，从而在高动态和视觉退化场景下提升精度。

**🔧 技术方法**

技术实现包括：自适应结点放置（AKP）、分辨率多重样条（MRS）、三维测量空间残差（3D‑MSR）、Lie 组重排、IMU 预积分以及 Ceres 优化器。

**📊 数据集**

实验使用了 VIO 基准数据集（UAV 轨迹）、自制手持序列（多层楼梯、室内到室外过渡）以及 Hilti‑Oxford 建筑序列（螺旋楼梯、重复结构）。

**📈 对比分析**

通过与 MSCKF‑DVIO、PL‑VINS、√(VINS)、Ctrl‑VIO、OKVIS‑CT 等基准对比，ASPIRE‑VINS 在 VIO 基准上平均 RMSE 0.212 m（比最优基准低 5.4%），在手持与 Hilti 序列中同样表现最佳；实时性为每帧 56 ms（≈18 FPS），在连续时间方法中保持可接受速度。

**⚠️ 局限性**

局限性包括对结点初始化的敏感性、长时间轨迹下的数值条件不佳，以及在纯旋转或视差不足的情况下 3D‑MSR 仍缺乏深度可观测性。

---

## 159. MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents

**arXiv ID:** 2608.12428 | [PDF](https://arxiv.org/pdf/2608.12428v1)

**作者:** Kaichao Liang `[一作]` (Noah's Ark Lab, Huawei Technologies), Mingxuan Yuan `[通讯]` (Noah's Ark Lab, Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可移植且自演化的记忆操作层 MindMemOS，用于长期记忆管理、个性化和任务学习。

**💡 创新点**

创新点在于三维实体–属性–时间图结构、验证驱动的记忆模式演化（MindMemEvolve）、离线记忆合并与冲突解决（Dreaming）以及基于执行轨迹的技能演化（MindSkillEvolve）。

**🔧 技术方法**

采用 LLM 辅助的进化搜索、稀疏+密集检索、紧凑搜索、图结构记忆存储、用户纠错反馈、离线梦境合并以及轨迹驱动的无监督/有监督技能更新算法。

**📊 数据集**

使用 LOCOMO、PersonaMem、MemoryAgentBench（FactConsolidation）和 SpreadsheetBench 等公开基准数据集进行评估。

**📈 对比分析**

与 Mem0、MemOS、EverOS、Zep 等基线相比，MindMemOS 在 LOCOMO 上实现 94.03% 的整体准确率，在 PersonaMem 上达 70.63%，Dreaming 在 FactConsolidation 中将准确率提升至 0.585，MindSkillEvolve 在 SpreadsheetBench 上将成功率提升约 9.2 个百分点。

**⚠️ 局限性**

局限在于对某些推理任务提升有限、演化过程对 LLM 的依赖较强、以及在更大规模或多域场景下的泛化能力仍需进一步验证。

---

## 160. Pipeline Denotational Design: Correct-by-Construction Data Pipelines at Zero Cost

**arXiv ID:** 2608.12375 | [PDF](https://arxiv.org/pdf/2608.12375v1)

**作者:** Nikos Karayannidis `[一作]` `[通讯]` (Independent), Nikos Karayannidis (Independent)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Pipeline Denotational Design（PDD），一种基于语义域的设计优先方法，用于在构造阶段就确保数据管道的正确性，生成的实现代码与设计完全一致。

**💡 创新点**

创新点在于：①通过“粒度（grain）”理论与同态证明实现零成本设计时验证；②引入 Pipeline Design Algebra（PDA）作为可组合的类型化操作集合；③构建三层（粒度、行为类、业务规则）设计时正确性框架，并提供可选的机器证明级别（Agda/Lean4），实现预防式错误。

**🔧 技术方法**

技术主要包括：粒度推理算法（CalcG）、类型化操作合同、证明携带的组合（Proof-Carrying Composition）、代码生成器、SQL/PySpark 生成的验证查询、以及基于 Agda/Lean4 的 Pipeline Design Language（PDL）。

**📊 数据集**

使用真实企业生产管道数据，覆盖多种模式、行为类和建模范式；评估涵盖设计时缺陷捕获、零成本放弃率、生成查询产出、代码生成忠实度等指标。

**📈 对比分析**

方法与三种验证级别对比：①运行时测试（无结构检查）→②已部署的类型检查器（仅粒度+行为类检查）→③PDL 证明级别（全层检查并生成最小输入边界查询）。实验表明，PDD 在设计时捕获大多数缺陷，零成本放弃率高，生成查询覆盖率显著提升，代码生成与原实现高度一致。

**⚠️ 局限性**

局限性：①需要操作集具备可计算的粒度推理规则；②仍需人工提供源/目标的语义（粒度、实体键、行为类）描述；③机器证明实现尚未成熟，编译成本高；④运行时仍需边界查询检查数据质量；⑤对新型流式或非关系型操作的支持有限。

---

## 161. Do LLMs Beat Nash? Testing Decentralized Coordination in Self-Play Multi-Agent Games

**arXiv ID:** 2608.12547 | [PDF](https://arxiv.org/pdf/2608.12547v1)

**作者:** Deborah Sinishaw `[一作]` (McGill University), Gregory Dudek `[通讯]` (McGill University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究无通信、单轮自我对弈下相同LLM拷贝是否能通过相互推理实现协调，并与纳什均衡基准比较；

**💡 创新点**

提出一套可参数化的单轮无通信博弈基准，并系统评估多种LLM在不同博弈结构下的自我协调能力；

**🔧 技术方法**

使用大型语言模型（Frontier Hosted如Gemini 3.5 Flash、GPT-5.6 Luna及多款1B–14B开源模型）、贝尔曼-霍普算法求纳什、随机扰动避免数值退化、Poisson-binomial/多项式方法处理团队组成；

**📊 数据集**

构造两类博弈：7种经典两人矩阵游戏（包含协同、冲突、零和等）和6种团队游戏结构（协调、阈值、公共物品等），并生成多行动、不同团队规模的实例；

**📈 对比分析**

与纳什均衡和最优结果对照，用两侧归一化得分（[-1,1]）衡量；在两人游戏中，Frontier模型在大多数类型中超过纳什，部分开源模型亦能逼近最优；但在反协同、性别博弈中表现负面；团队游戏中，2行动情况下部分模型略超纳什，3行动后多数模型跌入纳什以下，整体表现不佳；

**⚠️ 局限性**

局限性包括：仅测试单轮无历史、无信道的极端情景；团队游戏中团队规模与行动数扩大导致协调信号衰减；不同模型提供商的API调用限制和成本导致未对所有模型统一评估；缺乏对模型参数量、训练数据或指令调优等因素的细粒度分离。

---

## 162. Privacy-Preserving RAG by Concealing Sensitive Information from External LLMs

**arXiv ID:** 2608.12675 | [PDF](https://arxiv.org/pdf/2608.12675v1)

**作者:** Saleh Almohaimeed `[一作]` (King Saud University), Khalid A. Alobaid `[通讯]` (King Saud University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了 Sensitive Entity Alias Generator（SEAG）框架，在 RAG 系统中通过本地轻量级模型将查询和检索文档中的敏感实体替换为别名，防止外部 LLM 直接获取敏感信息。

**💡 创新点**

创新点在于将实体别名生成与替换作为 RAG 预处理步骤，兼顾隐私保护与回答质量，同时使用 PEFT 对小模型进行快速微调，实现高效部署。

**🔧 技术方法**

技术包括 Retrieval‑Augmented Generation（RAG）、实体识别与别名生成、参数高效微调（PEFT）中的 QLoRA、以及轻量级 LLM（Qwen‑3、LLaMA‑3.2、Phi‑4）与外部生成器（GPT‑5、Claude‑4 sonnet）的协作。

**📊 数据集**

使用两套数据集：640 条来自 8 个领域（新闻、法律、医疗、金融、生物、化学、历史、简历）的样本用于微调；600 条来自 6 个领域（经济、教育、汽车、法律、金融、商业）的样本用于评估 SEAG 框架。

**📈 对比分析**

通过 ORG、Privacy、User 三个指标评估，与传统 RAG 对比显示，LLaMA‑3.2+Claude‑4 sonnet 的 Privacy 达到 89.67%，User 达到 83.67%，整体在保护隐私的同时保持高回答准确率。

**⚠️ 局限性**

局限性包括仅在约 300 词、约 15 个敏感实体的短文本上验证，未对长文本或高实体密度场景进行评估，且在法律与金融领域的实体识别与替换效果不如经济领域。

---

## 163. GateTruth: Auditing the Rigor of RTL Design Benchmarks via Mutation Testing

**arXiv ID:** 2608.12635 | [PDF](https://arxiv.org/pdf/2608.12635v1)

**作者:** Meet Bhadra `[一作]` `[通讯]`, Meet Bhadra

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 GateTruth，一套基于变异测试的 RTL 评估引擎，用以检验 RTL benchmark 的 testbench 质量，并对现有 RTL‑LLM 和 NVIDIA CVDP 进行无修改的外部审计。

**💡 创新点**

创新点在于：①首次将变异测试引入大语言模型生成硬件验证的流程；②制定 95% 变异杀死阈值作为 testbench 通过门；③揭示 token‑budget 对排行榜排序的显著影响，并将此作为实验变量公开；④通过多层防污染机制确保评测的安全性和可信度。

**🔧 技术方法**

技术包括：确定性种子、顺序执行的变异生成器、模拟/形式验证杀死判定、基于 Docker 的可重复流水线、以及多级安全与隐私策略。

**📊 数据集**

使用了 60 任务的 GateTruth 自研基准（60 个规格到 RTL 任务 + 8 代理修复任务）和公开 Benchmark（RTLLM v2.0 的 50 设计、NVIDIA CVDP）。

**📈 对比分析**

通过对比 mutation‑kill 率和 Pass@k，发现 RTLLM 约 72% 的设计低于 GateTruth 的 95% 阈值，三设计 0% kill；在自身基准中 46/60 任务满足门槛，且模型 GPT‑5、Opus、Sonnet 在两轨道上表现突出，GPT‑5 在 16,384‑token 条件下从 5 级跃升为 1 级，展示 token‑budget 对模型表现的强烈影响。

**⚠️ 局限性**

局限性包括：①变异测试仅评估 testbench 对已注入错误的敏感性，无法保证 oracle 的正确性；②未覆盖所有公开 benchmark（如 VerilogEval、ChipBench 等）；③缺乏后置布局/路由级别的物理测量；④多任务中测试用例共享作者、风格，可能影响统计独立性；⑤对形式验证深度、等价性工具的限制可能导致杀死率低估；⑥ token‑budget 影响需更细粒度实验确认。

---

## 164. Research Assistant: AstraZeneca's Agentic System for R&D

**arXiv ID:** 2608.12395 | [PDF](https://arxiv.org/pdf/2608.12395v1)

**作者:** Piotr Grabowski `[一作]` (AstraZeneca), Michael Ughetto `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了内部基于LLM的多代理系统 Research Assistant，提供聊天式界面，整合文献、知识图谱、化学、临床试验、安全、表达等多源数据，支持快速问答与多步深度研究；

**💡 创新点**

创新点包括检索优先并行工具调用、基于主题建模的工具代理路由、Graph RAG 机制减少hallucination、可编排的研究计划 DAG、Observation 对象实现答案与来源的可追溯链接，以及采用 Apache Burr 状态机实现高并发与低成本运行；

**🔧 技术方法**

技术手段涵盖 Gemini 3 Flash/Pro LLM、Python asyncio + FastAPI、Apache Burr 状态机、TypeScript/React 前端、Instructor 库+Pydantic 结构化输出、Elasticsearch、Neo4j、GraphQL、REST APIs、Google Vertex AI、OpenAlex、ClickHouse、BERT、图卷积网络、知识图嵌入、CRank 等；

**📊 数据集**

数据来源包括内部 AstraZeneca 知识图谱 BIKG、化学 CAG、ELN 文档、内部实验数据、公开数据库 PubMed/EuropePMC/Embase/MedRxiv/bioRxiv/Wiley/Springer、OpenAlex 指标、ClinicalTrials.gov、Human Protein Atlas、Clarivate OFF‑X 等；评估使用 BioASQ、STaRK 等公开数据集；

**📈 对比分析**

评估方法包括在 BioASQ 10b 的 100 条是/否问答上与 GPT‑4 进行平衡准确率对比，使用 STaRK 的查询覆盖率测量知识图工具的回调精度，内部用户反馈与自动 Judge 评分相结合；平均查询成本约 16¢，响应时间 10‑30 秒，且系统在实际 R&D 工作中得到广泛使用；

**⚠️ 局限性**

局限性主要是：仍存在hallucination，尤其对细粒度基因/蛋白同义体的识别不够精准；知识图查询复杂度导致潜在超时；Web Search 结果可信度不一；深度研究模式耗时长、成本较高；系统高度依赖内部数据质量与更新；对某些医学细节的语义理解仍有限。

---

## 165. PROVE-RT: Generating Mechanized Theorem Prover Scripts for Real-Time Systems using LLMs

**arXiv ID:** 2608.12762 | [PDF](https://arxiv.org/pdf/2608.12762v1)

**作者:** Sadat Shahriyar `[一作]` (Florida International University), Abdullah Al Arafat `[通讯]` (Florida International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大语言模型的框架，自动化生成 Coq 证明脚本，用于对实时系统的可调度性分析进行机理化验证。

**💡 创新点**

核心创新在于：① 通过依赖感知的非正式草图提取与排序；② 结合检索增强（dense、BM25、Hybrid）获得文档上下文；③ 将生成过程拆分为结构化骨架生成与递归证明补全两阶段，显著提升生成质量。

**🔧 技术方法**

技术包括：大语言模型（Gemini‑2.5‑Flash、GPT‑5、Claude‑Opus‑4.6）用于提取与生成；检索增强（BM25、dense向量检索、Hybrid）；Coq 证明工具链（Rocq 9.1.0）进行编译与证明完整性检查；批量修复循环利用编译器反馈。

**📊 数据集**

使用了 1,191 篇实时系统论文构建的“系统不变式数据集”，共 13,134 条非正式草图及其依赖图；检索语料库来自 356 个 Prosa 源文件的 5,097 个文档片段。

**📈 对比分析**

与直接提示 LLM（仅使用论文文本或草图）对比，Prove‑RT 在 dense 检索下实现了 134/300（44.7%）的成功率，明显优于 GPT‑5（0%）和 Claude‑Opus‑4.6（0.33%）；BM25 与 Hybrid 亦表现接近（≈42%）。在章节深度越大时成功率下降，证明了依赖深度对自动化难度的影响。

**⚠️ 局限性**

局限性包括：对深层依赖链与复杂证明主体仍难以完全自动化；检索效果受文档质量与结构的限制；目前仅覆盖 Coq 生态，难以迁移至其他证明助手；模型对特定符号与抽象的理解仍有限，需人工校验。

---

## 166. Surface-to-Skeleton 3D Cephalometry: Estimating Hidden Skeletal Landmarks from CT-Derived External Soft-Tissue Surfaces

**arXiv ID:** 2608.12537 | [PDF](https://arxiv.org/pdf/2608.12537v1)

**作者:** Tomoki Abe `[一作]` (Keio University), Hideo Saito `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种基于CT同一采集外部软组织点云的坐标一致性表面到骨骼3D面颊测量估计方法，利用层次化点云网络实现对21个骨骼标记和3个软组织标记的预测；

**💡 创新点**

首次证明患者特定的外部几何能提供对深层不可见骨骼标记的可预测信息，并提出了坐标一致的实验协议、全局投票+局部ROI细化+PCA形状校正的整合模型，以及光学转移诊断框架；

**🔧 技术方法**

采用Sonata/PTv3全局投票编码器、PointNeXt局部细化模块、ShapeGate PCA形状校正、以及基线全局描述符PCA‑Ridge回归和点云编码器（PointNet++、PointNeXt）等技术；

**📊 数据集**

使用了来自东京牙科大学千叶牙科中心和东京都立大学西武医院的240例临床CT扫描，标注了21个骨骼和3个软组织标记，划分200例训练集、40例锁定测试集；

**📈 对比分析**

与全局PCA‑Ridge、PointNet++、PointNeXt、PTv3、Sonata/PTv3等基线进行对比，整体MRE为2.91 mm（21骨骼MRE 2.97 mm，深层MRE 3.03 mm），SDR@5达到88%，比基线平均提升约0.7 mm，且患者互换实验表明匹配患者几何是关键；

**⚠️ 局限性**

缺乏外部验证和跨站点泛化、样本量有限、缺少协议匹配的观察者可靠性评估、光学转移仍存在覆盖、姿势和非刚性变形问题、未提供年龄/性别/BMI等患者信息，导致方法在临床推广前仍需进一步验证与优化。

---

## 167. Training Under Challenge: Executable Certificates and Challenge-Closed Optimality for Neural Networks

**arXiv ID:** 2608.12655 | [PDF](https://arxiv.org/pdf/2608.12655v1)

**作者:** Farhang Yeganegi `[一作]` (University of Illinois Chicago), Mojtaba Soltanalian `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个可执行的挑战（Challenge）框架，用来对神经网络训练过程进行系统化审计与证明。

**💡 创新点**

创新点在于将可执行挑战与可证明的覆盖（spectral coverage）相结合，形成从“检查点能否被打败”到“全局最优性”再到“表示不足与任务适配性”的层级证明体系；同时提出了挑战力量（challenge power）与E‑optimal设计等度量工具。

**🔧 技术方法**

采用了可执行的头/块重拟、路径重建、精确的条件子问题求解、E‑optimal基准设计、谱覆盖算子、ReLU稳定性分析等技术，结合自监督的梯度下降、随机梯度与固定网络结构的条件优化。

**📊 数据集**

主要使用的公开数据集包括CIFAR‑10（ResNet‑18 distillation）、手写数字（MNIST）以及自定义的噪声图像/图像去噪任务，用于验证实验和量化覆盖效果。

**📈 对比分析**

与传统单纯训练曲线和单一超参数对比，框架提供了量化的下界（如残差覆盖系数、实测gap与理论gap之比在1.7–3.0范围内），并通过量化去噪实验展示了诊断与修复的可操作性；实验中挑战成功率和覆盖系数均达标，验证了理论一致性。

**⚠️ 局限性**

局限在于覆盖条件不满足时证明仍为相对（仅能说明在给定挑战族内不可优于当前模型）；挑战设计与实现复杂，需针对每个模型/任务显式编写挑战，且在大规模网络/高维数据上计算成本仍较高。

---

## 168. SchemaLink: An Intelligent Web Editor for LinkML Schema Curation

**arXiv ID:** 2608.12529 | [PDF](https://arxiv.org/pdf/2608.12529v1)

**作者:** Emanuele Cavalleri `[一作]` (University of Milano), Marco Mesiti `[通讯]` (University of Milano)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为 SchemaLink 的基于 Web 的图形化环境，帮助生物医学领域的研究者快速构建、编辑和验证 LinkML 模式；并在此基础上集成了 RAG（检索增强生成）式的人工智能组件，提供自动生成和智能修订建议。

**💡 创新点**

创新点主要包括：① 将 LinkML 的语法抽象为可视化图形语言，使非专业的生物信息学家能够直观操作；② 通过向量数据库检索与当前模式相似的已标注实例，结合大型语言模型（LLM）实现“从零生成”与“基于编辑操作”的双重智能支持；③ 设计了 43 种可编辑操作的模板化提示，并将其与四个定制化向量集合（完整模式、类、关系、类+关系）结合，显著提升了编辑质量和推理效率。

**🔧 技术方法**

核心技术包括：LinkML 与其多种结构化形式（树形、图形）；基于 arrows.app 的图形交互框架；ChromaDB 向量数据库配合 OpenAI 文本嵌入；gpt‑4o‑mini LLM 与 RAG 机制；LLM‑as‑a‑judge 评估框架；Python/JavaScript 前后端实现及 GitHub 开源。

**📊 数据集**

使用了来自 OntoGPT 项目的 60 条专家标注 LinkML 模式作为初始向量集合，并在此基础上自动生成 157 类、127 关系、对应的子集；此外还鼓励用户提交自定义模式以持续扩充数据库。

**📈 对比分析**

评估方法：① 5 个案例（疾病、药物、蛋白、通路、RNA）生成初始模式，10 名人类专家与 4 种主流 LLM 评判 0‑5 分；② 43 种编辑操作在 5 个案例中多次调用，记录人类与 LLM 评分；③ 采用自定义四种向量集合对比基线（仅完整模式）得到平均分提升；④ 对操作耗时进行测量。结果显示：人类专家与 LLM 的平均评分均在 3.5‑4.5 之间，超过 80% 的操作得分 ≥3；编辑操作中“添加”类/关系得分最高；定制向量集合提升 0.5‑1.4 分；端到端延迟通常低于 15 秒。

**⚠️ 局限性**

局限性包括：对复杂关系（多重约束、跨类依赖）的自动修订仍易产生错误或需要人工干预；LLM 可能出现幻觉导致无意义的修改；目前仅支持树形和图形两种 LinkML 结构，缺乏表格/关系型模式的转换；缺少对时间性与本体一致性等语义约束的完整支持。

---

## 169. Position: The Alignment Community is Unintentionally Building a Censor's Toolkit

**arXiv ID:** 2608.12346 | [PDF](https://arxiv.org/pdf/2608.12346v1)

**作者:** Sarah Ball `[一作]` (LMU Munich), Phil Hackemann `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析现代AI对齐方法的双重使用风险，指出其可被用于审查与操纵信息。

**💡 创新点**

首次系统性映射对齐技术的滥用潜力，并结合实际案例揭示其在政治压制中的应用。

**🔧 技术方法**

通过对预训练数据过滤、后训练偏好对齐和推理时控制等技术进行评估。

**📊 数据集**

使用公开的政治与审查基准数据、中文大语言模型训练语料及政府法规案例。

**📈 对比分析**

在对齐与审查效能的基准上对比，展示现有方法在信息抑制和偏见放大方面的效果与局限。

**⚠️ 局限性**

局限在于缺乏统一的多语言审查评测标准、对低资源模型的适用性不足，以及对攻击者的鲁棒性评估不足。

---

## 170. ERSkill: Evolving for Skill-Guided Adaptive Memory Retrieval

**arXiv ID:** 2608.12720 | [PDF](https://arxiv.org/pdf/2608.12720v1)

**作者:** Haolong Chen `[一作]` (Shenzhen International Center for Industrial and Applied Mathematics), Guanrxu Zhu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ERSkill框架，通过可组合的检索技能与路由器实现LLM代理的自适应记忆检索。

**💡 创新点**

将检索行为抽象为可执行技能集合，并引入经验trie和双前沿机制实现技能与路由器的协同演化。

**🔧 技术方法**

基于dense检索、BM25、查询改写等检索原语，使用编码器+MLP路由器和经验trie，在训练中通过双前沿策略更新技能。

**📊 数据集**

在LoCoMo、LongMemEval、PerLTQA三大记忆基准上进行实验，并使用Qwen3-Next-80B-A3B-Instruct与GPT-5.4-nano两种LLM后端。

**📈 对比分析**

与非进化基线（A-Mem、MemoryOS、LightMem）及自进化基线（Dynamic Cheatsheet、ReasoningBank、GEPA、MemSkill）对比，ERSkill在F1、BLEU‑1、LLM‑judge平均提升约31%（Qwen3）或28%（GPT-5.4），并在跨数据集转移、成本性能上表现优异。

**⚠️ 局限性**

受限于固定的原始原语库、路由器对训练样本的依赖以及双前沿导致的部署延迟，尚未在多模态或实时动态更新场景下进行充分验证。

---

## 171. The Role of Natural Language Understanding in Multimodal Video-Based Dengue Diagnosis

**arXiv ID:** 2608.12677 | [PDF](https://arxiv.org/pdf/2608.12677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 172. Agreement Is Not Alignment: Divergent Moral Grounds in Human and LLM Ethical Judgments

**arXiv ID:** 2608.12368 | [PDF](https://arxiv.org/pdf/2608.12368v1)

**作者:** Octavian M. Machidon `[一作]` (University of Ljubljana), Marko Robnik Šikonja `[通讯]` (University of Ljubljana)

**通讯引用:** 6756 | [OpenAlex ID](https://openalex.org/A5020021079)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一份由500条伦理案例组成的基准上，对人类注释者与大型语言模型（LLM）的最终判断与理由进行并行标注，并评估两者在标签一致性与道德理由上的匹配程度。

**💡 创新点**

首次揭示标签一致性并不能等同于道德对齐，提出通过对模型理由进行结构化分析来检测其与人类道德根源的契合度。

**🔧 技术方法**

采用LLM提示工程生成结构化JSON输出，使用统计指标（Cohen's κ、Jaccard重叠、平均绝对偏差）评估标签与理由的一致性。

**📊 数据集**

利用从ETHICS基准精挑细选的500条示例（涵盖常识、义务论、正义、功利主义与美德五个领域），并由五名注释者对其进行三次标注。

**📈 对比分析**

通过比较模型与人类多数标签的一致率（多数模型达到约88%）以及对结构化理由分布的偏差（平均绝对偏差显著且多模型理由重叠低于0.5），表明模型在标签层面表现优异，但在道德理由层面存在系统性偏差。

**⚠️ 局限性**

局限性包括注释者样本有限、数据集为手工挑选的子集、仅对三类分区的结构化理由进行分析、自由文本理由未纳入定量评估、模型与提示设置的多样性影响结果。

---

## 173. SCALE-Sim EVA: Design Principles for an Extensible, Visualizable, and Adaptable Accelerator Simulation Framework

**arXiv ID:** 2608.12354 | [PDF](https://arxiv.org/pdf/2608.12354v1)

**作者:** Jingtian Dang `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 14763 | [OpenAlex ID](https://openalex.org/A5034089074)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可扩展、可视化、适应性强的IR感知加速器模拟框架SCALE‑Sim EVA，支持多粒度命令执行和用户自定义硬件拓扑；

**💡 创新点**

其创新点在于统一的命令、张量与硬件组件抽象，允许在同一框架下模拟从粗粒度算子到细粒度数据移动的完整时序，并通过可视化追踪揭示数据与资源的交互；

**🔧 技术方法**

采用轻量级抽象堆栈、命令调度、张量映射与存储层次、以及可视化追踪技术，实现对硬件功能单元、存储模块和调度逻辑的模块化建模；

**📊 数据集**

本工作未使用特定数据集，框架设计适用于任意张量工作负载，可直接插入不同算子或模型的张量IR；

**📈 对比分析**

论文未给出具体性能评估或实验对比，只在概念层面与现有模型（如SCALE‑Sim v3、Timeloop、Accel‑Sim、gem5‑SALAM）进行了对比说明，强调其更通用且轻量；

**⚠️ 局限性**

局限性包括需要用户手动定义硬件组件和调度策略、缺乏自动化映射支持、对完整系统性能评估和硬件实现细节的验证仍有限。

---

## 174. Predicting When Random Low-Dimensional Reparameterizations Train Neural Networks

**arXiv ID:** 2608.12597 | [PDF](https://arxiv.org/pdf/2608.12597v1)

**作者:** Andrew Cheng `[一作]` (Tsinghua University), Qiang Cheng `[通讯]` (University of Kentucky)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了随机映射网络（RaMaN）框架，通过低维随机重参数化训练大规模神经网络，并给出了可预测所需潜在维度的理论与实用方法。

**💡 创新点**

创新点包括：① 推导了考虑曲率谱和位移方向的定向二次主公式，提供自洽的等向性预测器；② 用结构化Hadamard映射和种子重生成高斯映射实现矩阵自由、高效的随机子空间；③ 设计了三模式自适应维度选择和实验基准。

**🔧 技术方法**

主要技术手段包括：凸几何和统计维数理论、随机切片可达性分析、定向二次残差预测、矩阵自由曲率估计（Ritz 近似、随机迹估计）、结构化Hadamard变换、种子重生成高斯映射、AdamW/SGD 训练与学习率自适应。

**📊 数据集**

使用的数据集涵盖：MNIST、CIFAR‑10/100（MLP、CNN、ResNet‑18）、ViT‑Tiny 在 CIFAR 上、SST‑2 语言模型微调，以及 ViT‑Tiny 在 CIFAR‑10/100 的大规模实验。

**📈 对比分析**

对比方法：用定向二次预测器、等向性预测器和无方向保守预测器与实验测得的随机子空间可达性阈值进行对比；在端到端训练中对比 RaMaN 与完整参数训练的训练成功率和测试准确率，发现 RaMaN 在潜在维度极低（0.1%–几%）即可达到接近完整模型的准确率，且显著降低了训练内存、检查点和映射存储。

**⚠️ 局限性**

局限性：理论依赖局部二次近似和曲率估计，计算成本随模型规模增大；可达性阈值对训练协议、参考点和学习率调度高度敏感；自适应维度扩展需多次训练或粗糙估计；在极大模型（数十亿参数）上的实用性仍需进一步验证。

---

## 175. Attune: A Self-Annotation Tool for Understanding Robot Operator Attention Profiles

**arXiv ID:** 2608.12650 | [PDF](https://arxiv.org/pdf/2608.12650v1)

**作者:** Puqi Zhou `[一作]` (George Mason University), David Porfirio `[通讯]` (George Mason University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了 Attune 预部署工具，记录操作员在多机器人监控时的视线轨迹，提供自动化的注视转移检测、结构化自我注释以及基于 LLM 的模式归纳和自动注释，从而构建操作员的注意力剖面。

**💡 创新点**

创新点包括：① 将眼动记录与机器人行为、视觉线索结合，形成可解释的注视转移结构；② 通过 LLM 聚类和自动化提示实现大规模注释支持；③ 将自我注释结果转化为可用于界面与机器人行为校准的注意力剖面；④ 通过多阶段交互（感知、注释、模式提取、自动化、总结）实现一次性、人机协作的校准流程。

**🔧 技术方法**

采用的技术：眼动追踪器（Tobii Pro Spark）+ YOLOv8 对视频中的目标进行分割；React/TypeScript 前端 + FastAPI 后端实现同步视频播放、眼动重放、注释界面；LLM（Gemini‑2.5‑Flash 用于模式聚类，Llama‑3.3‑70B 用于单转移自动注释）；自定义非线性时间线放大、记忆清晰度/置信度标记等人机交互设计。

**📊 数据集**

使用的数据集为两台 Stretch 3 机器人在三种场景（厨房、客厅、走廊）下分别录制的同步视频（天花板视角、头部视角、抓手视角），每段约35–83分钟，后挑选 8 分钟的片段用于 12 名受试者的实验。

**📈 对比分析**

评估方法：12 名受试者进行一次性实验，完成监控、注释、回顾三阶段；收集眼动记录、注释结果、访谈、SUS 等；SUS 平均得分 75.21（属于良好级别）。在定性分析中发现注释帮助受试者洞察自身关注模式、支持自动化摘要的可接受性；但缺乏客观的 LLM 预测准确率和基准对比，未能区分任务驱动 vs 视觉驱动的注视转移。

**⚠️ 局限性**

局限性：① 仅在两机器人、六视角的最小配置下验证，难以直接推广至更大规模、多平台或不同布局的真实部署；② 采用回溯式注释，易受回忆衰减与后验解释偏差；③ 未使用任务或视觉显著性基线，无法客观判断注视转移的驱动机制；④ 样本规模小、受试者为初学者，缺乏专业操作员的数据；⑤ LLM 的准确性与置信度未系统评估；⑥ 实验中强制性口头化可能改变自然关注模式；⑦ 现有工具为研究原型，需整合进实际工作流以提升可采用性。

---

## 176. Can Vision-Language Models Assess Proxemic Risk from Egocentric Robot Images?

**arXiv ID:** 2608.12515 | [PDF](https://arxiv.org/pdf/2608.12515v1)

**作者:** Vladyslava Rudas `[一作]` (National University of Kyiv-Mohyla Academy), Dmytro Kuzmenko `[通讯]` (National University of Kyiv-Mohyla Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究评估了三种开源视觉-语言模型(VLM)在机器人自视角图片中对亲密距离风险进行分类的能力，并探索了不同提示策略和QLoRA微调对高风险检测的影响。

**💡 创新点**

首次从机器人自视角视角系统性比较VLM的空间推理与风险分类能力，探讨了提示复杂度对高危检测召回率的“格式税”效应。

**🔧 技术方法**

采用InternVL、Qwen-VL和SmolVLM三种VLM，结合三种结构化提示（Simple、Moderate、Advanced）以及两轮QLoRA参数高效微调。

**📊 数据集**

使用基于JRDB的1,243张自视角图像构建的新数据集，按Edward Hall的亲密距离区域划分为高、中、低、最小四类危险级别。

**📈 对比分析**

在随机基线（加权F1≈0.25）的对照下，所有模型总体表现接近基线；Qwen模型在无微调时即可实现≈0.4的高危召回率，采用高级提示可将其提升至≈0.79，但整体F1下降；其他两模型高危召回率几乎为零，微调提升幅度有限。

**⚠️ 局限性**

主要局限在于模型对空间定位的感知不足，即高危标签不一定对应正确的人员位置；数据集规模小且不平衡限制了微调效果；仅测试了三种中等规模模型，无法评估更大模型的潜力。

---

## 177. Trie Automata for Constrained Decoding over Large Finite Sets

**arXiv ID:** 2608.12574 | [PDF](https://arxiv.org/pdf/2608.12574v1)

**作者:** Xingzi Xu `[一作]` (Amazon), Karim Bouyarmane `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种专门针对有限集合（枚举）约束的“Trie Automaton”，通过预先计算并缓存 BPE 词元掩码，实现了高效的约束解码。

**💡 创新点**

创新点在于将 Aho‑Corasick 多模式匹配与字符级 Trie 结合，解决了 BPE 对齐问题，突破了枚举规模“门限”，使得枚举大小可扩展到十万级别，同时保持 100% 输出有效性。

**🔧 技术方法**

核心技术包括：字符级 Trie 构建、Aho‑Corasick 自动机预处理、预计算词元掩码、与 vLLM 兼容的无状态调用路径；对比 XGrammar、LLGuidance 等通用解码后端。

**📊 数据集**

实验使用七种不同词表大小（32K–262K）的 LLM 模型（如 Qwen3‑8B、Mistral、Gemma3 等）以及合成工具调用和公开分类数据集（TREC、MASSIVE、Banking77、CLINC150）验证性能与准确率。

**📈 对比分析**

与 XGrammar 和 LLGuidance 的对比显示：编译时间在 K≥1,000 时加速 2–6.5×；每步掩码计算 7× 更快；在批量推理中整体吞吐量提升 29×（219 req/s 对比 7.5 req/s）。同时保持 100% 的约束有效性。

**⚠️ 局限性**

局限性：仅适用于平面枚举约束；嵌套或数组等结构化字段仍需通用 FSM；在极小 K 的一次性动态约束场景下，LLGuidance 的近零编译时间可能更优；当枚举规模极大时，AC 自动机构建成本仍会成为瓶颈。

---

## 178. On Measuring Semantic Preservation in Legal Ontology Learning

**arXiv ID:** 2608.12326 | [PDF](https://arxiv.org/pdf/2608.12326v1)

**作者:** Albert Sadowski `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于LLM在原文和本体表示上任务表现差异的语义保留评估框架。

**💡 创新点**

创新点在于用LLM准确率的差距量化本体学习过程中的语义损失，并揭示模型-方法匹配对语义保留的显著影响。

**🔧 技术方法**

采用了三种本体学习方法（LLMs4OL、NeOn-GPT、NeOn-CoT）和六款LLM（GPT‑4.1 mini、Claude Sonnet 4、Gemini 2.5 Flash、DeepSeek v3、o4 mini、Llama 4 Maverick）进行实验。

**📊 数据集**

使用LegalBench的MAUD合并协议理解数据集，共34道多项选择题。

**📈 对比分析**

通过比较LLM在原始文本与转换后的本体上的准确率来计算语义损失，结果显示最佳组合（Gemini + NeOn‑CoT）可保留约86‑88%的语义，而其他组合的损失高达27‑27.4个百分点。

**⚠️ 局限性**

局限性包括仅针对单一法律子域、仅评估多项选择任务、未考虑形式推理带来的潜在收益，以及未覆盖知识增量的本体生成。

---

## 179. CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives

**arXiv ID:** 2608.12779 | [PDF](https://arxiv.org/pdf/2608.12779v1)

**作者:** Chengyang He `[一作]` (Stevens Institute of Technology), Ping Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于LLM的生成–验证框架 CRAFT，用于在临床叙事中对症状时间序列进行迭代细化，并公开了 MedTempo 基准。

**💡 创新点**

创新点在于将生成–验证迭代循环与结构化时间桶表示结合，首次在弱锚定单报告场景下实现阶段化症状轨迹重构，并首次发布该任务的专家标注数据。

**🔧 技术方法**

使用 GPT‑4.1、Claude‑4.5、Llama‑3.3‑70B、MedGemma‑27B 等 LLM，配合全重生成或编辑条件生成器以及增量式规则评分验证器。

**📊 数据集**

使用 MedTempo 数据集，共 5,347 条 COVID‑19 疫苗不良事件报告，其中 3,166 条带有专家验证的阶段顺序。

**📈 对比分析**

与 PIVOT、GUIDE 基线对比，采用 EM、LCCS、τ_b 三种评估指标，CRAFT‑Full 在所有模型上均取得最高 EM（最高约 37.1%），显著提升了时间顺序重构准确率。

**⚠️ 局限性**

局限在于对强模型的首轮输出已接近完美时迭代收益有限，阈值设定可能导致过度修正；对 Pfizer 报告的分割精度仍不足，且未解决无时间进程报告的检测问题。

---

## 180. SynAct: A Reasoning-Acting Large Language Model Agent for Adaptive Synthesis Optimization

**arXiv ID:** 2608.12751 | [PDF](https://arxiv.org/pdf/2608.12751v1)

**作者:** Fangzhou Liu `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SynAct，一种闭环LLM驱动的自适应合成优化代理，通过诊断合成报告、检索工具知识和利用历史经验，迭代生成针对RTL设计的优化命令，从而在商用合成工具上实现PPA改进。

**💡 创新点**

创新点在于将大语言模型与多层GraphRAG知识检索、GrammarVAE编码+贝叶斯优化相结合，构建可持续自适应的闭环决策流程，实现对离散命令空间的高效探索与经验重用。

**🔧 技术方法**

使用了DeepSeek V3.1 / GPT‑5.2等LLM、ReAct 与 AutoGen 框架、GraphRAG 多层知识图检索、GrammarVAE 编码与贝叶斯优化、以及商用逻辑合成工具的自动化交互。

**📊 数据集**

使用了 14 个来自 OpenCores 的开源 RTL 设计，并在 ASAP7 7nm PDK 上运行商用合成工具进行实验。

**📈 对比分析**

与 ChatLS（单次脚本生成）和 CBTune（固定动作集的 bandit 搜索）对比，SynAct 在 5 次迭代内将平均 WNS 降低至 bootstrap 的 27.03%，明显优于 ChatLS 的 71.73% 与 CBTune 的 66.67%，同时保持面积与功耗接近基准；在 GPT‑5.2 上进一步提升至 20.37%。

**⚠️ 局限性**

主要限制包括候选命令评估耗时占比高，离散命令空间仍需更高效搜索；实验仅在单一商用合成工具上验证，跨工具适配及工业级规模验证尚待进一步研究。

---

## 181. Which Site, and When: A Free-Satellite-Data Test of Himalayan Glacial Lake Bursts, Landslides, and Ice Floods

**arXiv ID:** 2608.12422 | [PDF](https://arxiv.org/pdf/2608.12422v1)

**作者:** Matthew Kahn `[一作]`, James Pope `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用免费卫星遥感和天气数据，对喜马拉雅高山地区的冰川湖溢洪、滑坡和小型湖洪进行预测测试，评估模型在识别易破坏地点与触发窗口的能力。

**💡 创新点**

提出将易受损性与触发机制分离、使用匹配对照、在同一实验框架下同时评估三类灾害，并提供可解释的三规则模型和尼泊尔监测清单。

**🔧 技术方法**

采用梯度提升树基线，比较五种深度学习架构（自监督预训练、异常检测、注意力、物理约束、正负未标记学习），并使用空间交叉验证与Bootstrap置信区间。

**📊 数据集**

使用HMAGLOFDB 589次历史溢洪、COOLR数千次滑坡、ICIMOD湖泊清单及Google Earth Engine提取的自由遥感、天气与地形特征。

**📈 对比分析**

相较于梯度提升基线，深度模型在滑坡上略有提升但未突破置信区间；在冰湖溢洪和小湖洪上基线优于深度模型；天气触发模型在三类灾害均达到≈0.8 ROC，显示可行的提前预警。

**⚠️ 局限性**

受事件空间不均、未标记湖泊可能为未破坏且未记录、缺乏湖泊体积与完整InSAR时间序列等因素限制，导致易受损性预测仅能在大坝溢洪和滑坡上实现0.7–0.9的ROC，无法对小型湖洪进行排序。

---

## 182. Test-Time Optimization of Query Embeddings with Ranking Aware Reward Maximization

**arXiv ID:** 2608.12569 | [PDF](https://arxiv.org/pdf/2608.12569v1)

**作者:** Tianyu Chen `[一作]` (University of Texas at Austin), Jiaxing Wu `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TTT-Embed 框架，将检索奖励（由外部重排器或 LLM 生成的相关性分数）转换为一个可复用的向量，并在检索时将其加到冻结的查询嵌入上，从而在不修改模型权重或索引的前提下提升检索效果。

**💡 创新点**

创新点：① 仅在输出嵌入空间中学习轻量级、可复用的向量；② 通过可调的共享范围（全局、任务级、查询级）实现不同粒度的奖励共享；③ 用证据自适应缩放（α = n/(n+1)）自动确定向量权重，无需手动调参；④ 通过知识蒸馏方式将奖励分布迁移到查询向量。

**🔧 技术方法**

技术细节：知识蒸馏（listwise soft-label）、ridge 正则化、Additive embedding 调整、基于预算的奖励分配、证据自适应缩放、三种共享范围实现、与不同奖励模型（Qwen3‑Reranker‑4B、Gemini 3.1 Pro 等）的兼容性。

**📊 数据集**

数据集：MTEB benchmark（15 个英文检索任务，12,263 个查询、1,661,208 条文档）和 SKILLRET（4,997 个查询、6,660 条技能卡）用于验证领域泛化与遗忘恢复。

**📈 对比分析**

对比方法：原始检索（不使用奖励）、直接重排（仅对候选列表重新排序）、TTT-Embed 的三种共享范围。实验显示：在 b=10（每查询 10 条奖励）时，任务级 TTT-Embed 在平均 6.36 点提升 nDCG@10，超过直接重排 3.15 点；在 b=100 时，查询级 TTT-Embed 提升 8.36 点，仍优于直接重排 0.56 点；在全覆盖情况下，TTT-Embed 在不同模型上平均提升 1.23–10.92 点，并在领域泛化实验中恢复并超过微调模型的整体性能。

**⚠️ 局限性**

局限性：① 仅适用于冻结的嵌入模型，无法更新权重；② 需要外部奖励模型和一定的预算（即对查询-文档对的评估），在极低预算场景下收益有限；③ 对奖励模型的质量高度依赖，非专用的 LLM 可能导致负面效果；④ 目前只在单模态文本检索上验证，尚未证明在多模态或非文本检索任务中的有效性。

---

## 183. A Rig of Transformations

**arXiv ID:** 2608.12409 | [PDF](https://arxiv.org/pdf/2608.12409v1)

**作者:** Emma Tye `[一作]` `[通讯]` (University of Strathclyde), Emma Tye (University of Strathclyde)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

构造了一个基于交换环公理的 DSL，用来描述和转换代数数据类型的二进制布局，从而实现更紧凑、可定制的运行时表示。

**💡 创新点**

创新点是将等价关系与部分同构结合，提供可组合的预留空间控制和自动推导的转换算法，支持在类型级别直接指定目标布局。

**🔧 技术方法**

使用了交换环公理、部分同构（embedding–projection pair）以及预排序关系来建模数据布局转换，配合二进制标签化与内存大小计算。

**📊 数据集**

本文未使用真实数据集，而是通过理论模型和示例 ADT（如 efficientSet）演示方法。

**📈 对比分析**

示例中将 112 位的原始布局压缩到 104 位，显示空间利用率提升；论文未给出实际运行时性能测试。

**⚠️ 局限性**

局限在于仅描述了转换关系和解析/打印逻辑，未实现完整的程序编译；对复杂多态/依赖类型支持不足，需要进一步验证和优化。

---

## 184. Beyond the Best Guess: Improving LLM Solution Coverage with Evolution Strategies

**arXiv ID:** 2608.12679 | [PDF](https://arxiv.org/pdf/2608.12679v1)

**作者:** Conor F. Hayes `[一作]` (Cognizant AI Lab), Xin Qiu `[通讯]` (Cognizant AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了 Evolution Strategies（ES）与 Reinforcement Learning（RL）在 LLM 后训练中的性能，特别是对数学推理任务的 pass@k 覆盖度、分布特征以及自一致性投票的效果进行系统评估。

**💡 创新点**

① 证明 ES 在保持 pass@1 精度的同时显著提升 pass@k 并避免 RL 产生的分布崩溃；② 通过进步/回退、熵分布等多维度分析揭示 ES 产生更广泛且更不确定的失败模式；③ 在多规模模型和多数据集上首次展示 ES 超越 RL 的一致性。

**🔧 技术方法**

使用 Evolution Strategies（基于随机权重扰动的零阶优化）、RL with verifiable rewards（政策梯度 + 计分器）、pass@k 评估指标、熵分析、Self‑Consistency 投票；训练框架包括 ES‑at‑Scale 与 VERL。

**📊 数据集**

GSM8K、MATH（Level 3‑5）、MATH500、Olympiad Bench、Minerva 等公开数学推理数据集。

**📈 对比分析**

在 1.5B‑32B 参数规模、Qwen2.5/Qwen3 系列模型上，采用相同训练预算对比 ES 与 RL（OatZero、SimpleRL‑Zoo）。结果显示：ES 在 k>1 时均超过 RL，且不存在基模型超越 RL 的现象；ES 的答案分布熵更高、回退更少；在自一致性投票下，ES 的准确率随 k 增大明显优于 RL。

**⚠️ 局限性**

• 仍未验证在更大模型或其他领域（如科学、编程）上的普适性；• ES 的训练仍需大量并行资源，成本较高；• 对分布崩溃的理论解释仍不完整；• 在非可验证任务中直接衡量 pass@k 的适用性需进一步研究。

---

## 185. Lines and Ladders: A Context-Aware Multi-Agent Framework for Large-Scale Retail Price Taxonomy

**arXiv ID:** 2608.12674 | [PDF](https://arxiv.org/pdf/2608.12674v1)

**作者:** Ravi Teja Chunduri `[一作]` (Walmart Global Tech), Pranay Kona `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套可扩展的上下文感知多代理框架，用于自动化生成零售价格层级的“Lines and Ladders”分类体系，提升价格治理效率。

**💡 创新点**

创新点在于将价格治理拆分为相似性与差异性属性发现的双代理设计，并通过合成代理实现语义融合，从而在海量非结构化商品数据中实现高精度的层级分组。

**🔧 技术方法**

采用多代理大型语言模型（LLM）与多模态抽取模型，结合分布式多模型路由、上下文记忆与人机反馈循环，实现属性识别、抽取、标准化与层级聚类。

**📊 数据集**

使用沃尔玛全球零售商真实商品目录，涵盖约100万条商品记录，涉及食品、家电、服装等14个品类。

**📈 对比分析**

在实验中，三代理体系在Lines F1分数0.83、Ladders F1分数0.82，显著优于单代理（0.64）和双代理（0.77）；与商家现有结构对比，Lines Precision 98%、Recall 75%，Ladders Precision 92%、Recall 81%，并在生产环境实现90%+结构覆盖与86%工时缩减。

**⚠️ 局限性**

局限性包括LLM的非确定性导致可重复性受限、对采样偏差敏感、批处理模式导致实时性不足，以及对模型更新的持续验证需求。

---

## 186. BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving

**arXiv ID:** 2608.12854 | [PDF](https://arxiv.org/pdf/2608.12854v1)

**作者:** Bing Zhan `[一作]` (Chinese Academy of Sciences), Zhaoxiang Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种BrainWAM框架，将语义推理和预测动力学在动作空间进行协调，实现端到端自动驾驶计划。

**💡 创新点**

创新点在于把语义与预测分离成两条动作路径，通过脑功能分化激励的CAB和CIF模块实现动作级协同，而非直接的原始token融合。

**🔧 技术方法**

采用VLM（Qwen3-VL-4B）与视频生成模型（Wan2.2-TI2V-5B）结合rectified‑flow训练的双流架构，并在动作层通过跨流注意力和融合模块实现协调。

**📊 数据集**

在NAVSIM v1和v2两个基准集上进行评估，数据来自重构的nuPlan/OpenScene驾驶日志。

**📈 对比分析**

与VLA、WAM单分支、Tri‑MoT以及多种端到端和世界模型方法比较，BrainWAM在NAVSIM v1上达PDMS 89.5、v2上EPDMS 89.6，均超越所有对照方法。

**⚠️ 局限性**

主要局限包括较高的推理延迟（需多步视频去噪）、对大规模预训练模型的依赖以及在极端交通情景下仍可能出现语义与预测冲突。

---

## 187. StrAD: A Streaming Method and Benchmark for Audio Description Generation for Long-form Videos

**arXiv ID:** 2608.12549 | [PDF](https://arxiv.org/pdf/2608.12549v1)

**作者:** Julian Spravil `[一作]` (Fraunhofer IAIS), Sven Behnke `[通讯]` (Fraunhofer IAIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了首个面向全长视频的流式音频描述（AD）生成框架，并构建了多类型影片的手工验证AD基准数据集（SAD），同时实现了基于多模态LLM的零射与微调两种AD生成模型。

**💡 创新点**

创新点包括：①定义segment‑level、streaming和document‑level三大任务，②使用滑动窗口无时间戳的流式生成方法，③结合视觉、音频与文本信息的决策机制，④构建覆盖电影、纪录片、短片、游戏和表演等多类型的全长视频AD基准。

**🔧 技术方法**

技术手段主要包括：多模态大型语言模型（Phi‑4‑mm、Qwen‑3.5）+ LoRA微调；视觉+音频+文本滑动窗口处理；基于上下文的AD生成决策层；音频信号辅助定位与避免冗余；使用CIDEr、SODA、Recall@k/N、CRITIC、LLM‑AD‑Eval等评估指标。

**📊 数据集**

使用数据集：CMD‑AD、MAD‑Eval、TV‑AD（细粒度评估）以及本文自建的Streaming Audio Description（SAD）基准，SAD包含33段视频、22.3小时、9,131条手工验证AD事件，涵盖短片、纪录片、游戏、电影、表演等多种类型。

**📈 对比分析**

在CMD‑AD上，微调模型Qwen‑3.5实现CIDEr 36.3（比Shot‑by‑shot提升10.0），在SAD上实现SODA 2.4、CIDEr 34.0；零射基线平均表现为CIDEr 13.9。相比传统细粒度AD方法，本文模型在多项指标上均超过或匹配最优方法，并在实时推理中达到RTF 0.52、延迟约4秒，显示出可行的实时流式生成能力。

**⚠️ 局限性**

局限性包括：生成质量仍低于人工描述，可能出现误导性错误；基准仅覆盖部分影片类型、语言与文化背景，且受视频下架影响可重现性有限；评估指标未能完全体现视障用户的感知体验；模型在长时段仍易出现重复、音频重叠等错误。

---

## 188. HC-RAG: Evidence-Centric Retrieval-Augmented Generation over Heterogeneous Financial Filings

**arXiv ID:** 2608.12335 | [PDF](https://arxiv.org/pdf/2608.12335v1)

**作者:** Siyuan Chen `[一作]` (Sun Yat-sen University), Jiajun Liang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1643 | [OpenAlex ID](https://openalex.org/A5035488063)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 HC‑RAG 框架，利用层次化金融证据图实现文档→章节→文本/表格单元的分层检索，并通过意图感知的文本‑表格路由完成多文档财务问答

**💡 创新点**

①在10‑K结构上构建 typed 证据图，保持文档层级与跨文档跨年份的关联；②异步跨模态对齐，将文本与表格嵌入同一检索空间；③意图驱动路由，根据计算、趋势、事实、比较四类意图动态权衡文本与表格证据

**🔧 技术方法**

FinBERT 等金融文本编码器、TAPAS/TAPEX 表格编码器、对比学习式跨模态对齐、RAG 生成框架、图检索与层次化检索策略、意图分类器与可学习的路由权重

**📊 数据集**

Multi‑Doc‑2025（179份 SEC 10‑K，87 家 S&P 500 企业，2022–2024 年）以及 FinQA、TAT‑QA、DocFinQA、FinanceBench 等公开财务 QA 基准

**📈 对比分析**

在五大基准上与 BM25‑RAG、DPR‑RAG、Contriever‑RAG、Self‑RAG、GraphRAG、RAPTOR、TAPEX‑RAG 等对齐；HC‑RAG 在 DocFinQA 提升 6.6 F1，Multi‑Doc‑2025 提升 10.9 F1；在表格检索、跨文档召回与证据定位方面均显著优于所有基线

**⚠️ 局限性**

依赖 10‑K 的标准 HTML 结构，对扫描或无结构 PDF 处理效果不佳；表格提取质量不一会削弱检索；意图分类错误导致路由失准；Benchmark 仅覆盖标普 500 公司，泛化到其他公司或地区仍需进一步验证

---

## 189. Can Spectral-Clipping Enable Better Learning While Forgetting Less for Low-Rank Adaptation?

**arXiv ID:** 2608.12332 | [PDF](https://arxiv.org/pdf/2608.12332v1)

**作者:** Hyowon Wi `[一作]` (Korea Advanced Institute of Science and Technology), Noseong Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2682 | [OpenAlex ID](https://openalex.org/A5067253588)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的低秩适配方法SCLoRA，利用参数化奇异值分解与谱裁剪，在微调时控制适配器的奇异值增长，从而提升下游任务性能并显著减轻灾难性遗忘。

**💡 创新点**

创新点包括①首次理论证明适配器奇异值的增长会导致预训练知识遗忘；②提出基于预训练参数谱分布的分位数裁剪，使适配器的奇异值受限于预训练谱；③使用参数化SVD直接学习奇异向量与裁剪后的奇异值，避免每步显式分解；④系统评估主导奇异成分可迁移、次要成分需大幅适配的现象。

**🔧 技术方法**

核心技术包括奇异值分解（SVD）、Fisher重叠度量、谱裁剪（基于分位数的上限）、参数化SVD、正交正则化、MAP/贝叶斯推理分析、LoRA架构以及对比实验。

**📊 数据集**

使用的数据集包括GLUE（MRPC、SST‑2、CoLA、STS‑B等）、SQuAD v1.1/v2.0、Commonsense reasoning任务（BoolQ、PIQA、SIQA等）、以及评估灾难性遗忘的预训练数据集（BookCorpus、OpenWebText、PG19、C4en）。

**📈 对比分析**

通过与LoRA及其多种变体（AdaLoRA、PiSSA、MiLoRA、LoRA‑GA、CorDA、DoRA、HAdapter、PAdapter 等）在同一模型（RoBERTaBase、DeBERTaV3Base、LLaMA等）和相同任务上的实验比较，SCLoRA 在 GLUE、SQuAD、Commonsense 等基准上均取得更高的下游指标，并在预训练任务上的准确率/困惑度下降幅度明显小于对照组，表明能更好地保持预训练知识。

**⚠️ 局限性**

局限性包括需设定分位数阈值 q，虽然实验表明 q 需处于中等范围但仍是超参数；方法对预训练谱的准确估计有一定依赖；在某些大规模模型或特定任务上仍需进一步验证。

---

## 190. NavSight in the Wild: Understanding Real-World Use of a Mobile Augmented Reality Application for People with Low Vision in Outdoor Navigation

**arXiv ID:** 2608.12759 | [PDF](https://arxiv.org/pdf/2608.12759v1)

**作者:** Yuheng Wu `[一作]` (University of Wisconsin-Madison), Yuhang Zhao `[通讯]` (University of Wisconsin-Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并实地评估了一款名为 NavSight 的移动 AR 应用，帮助低视力用户在户外实时识别并可视化关键对象（人行道、路缘、车辆等），并通过七天日记研究收集真实使用情境、配置策略和误差处理。

**💡 创新点**

创新点：
• 首次在真实户外环境中对 AI 驱动的低视力 AR 辅助系统进行长期体验评估；
• 提供对象级自定义和分组配置，让用户根据任务动态调整增强方式；
• 系统性揭示了环境因素（雨光、阴影、非标准路面）对识别与增强的影响，并给出针对性的设计建议。

**🔧 技术方法**

核心技术：
• YOLO11l‑seg（细化版）实现实时实例分割；
• Unity 与 Apple Core ML 搭建端到端渲染流水线；
• 六种可视化增强（轮廓、实色叠加、闪烁、亮度调节、背景变暗、色彩移除）以及两组对象分组机制。

**📊 数据集**

使用数据集：对 20,000 张 Mapillary Vistas 截图进行精细化并划分 80/10/10 的训练/验证/测试集，训练出专用于低视力场景的细化模型。

**📈 对比分析**

性能与对比：
• Fine‑tuned 模型相较于未微调的 YOLO11l‑seg 基线在 mAP50/mAP75/mAP 上提升至 0.588/0.305/0.318；
• 设备端总延迟约 70 ms，帧率约 25.6 fps；
• 30 分钟持续使用消耗约 20 % 电量；
• 参与者普遍认为增强有效且提升安全感，但在强光和湿润环境下可见度下降，误差导致注意力分散。

**⚠️ 局限性**

局限性：
• 识别对雨光、阴影、非标准路面敏感，易出现误检或漏检；
• 强光下屏幕可见度降低，需额外音频提示；
• 需手持设备，导致注意力与手部占用；
• 仅识别离散对象，缺乏地面缺陷（裂缝、凹凸）识别；
• 仅在手机平台实现，未充分探索可穿戴显示的社交与技术可行性；
• 对 AI 错误的解释与信任校准仍不足。

---

## 191. Spatial Memory Agent: Experience-Grounded Procedure Memory for Spatial Intelligence

**arXiv ID:** 2608.12743 | [PDF](https://arxiv.org/pdf/2608.12743v1)

**作者:** Haokai Zhang `[一作]` (Zhejiang University), Hao Chen `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Spatial Memory Agent (SMA)，在不更新模型参数的前提下，通过经验归纳、可验证的回溯与迁移可靠性评分（TRS）构建可迁移记忆库，指导冻结的视觉语言模型进行空间推理。

**💡 创新点**

创新点在于：①将验证器回溯信号转化为可重用的迁移性记忆；②通过在线校准的 TRS 对记忆进行可靠性评估；③采用两阶段检索（语义过滤 + TRS 加权）实现高效且可靠的记忆召回；④一次性记忆写入（One‑Pass）避免冗余与低覆盖率。

**🔧 技术方法**

使用技术包括：冻结的 VLM（Qwen3.x 系列），任务嵌入的语义相似度计算，TRS 的在线校准公式，检索策略（semantic filter + combined ranking），回溯模型 R_ϕ，One‑Pass Memory Writing，数据驱动的实验评估。

**📊 数据集**

使用的数据集涵盖五个空间推理基准：RoboSpatial、ERQA、Omni3D、SAT 与 EmbSpatial，分别对应机器人感知、物体关系、3D 空间、抽象空间与指令驱动的空间推理。

**📈 对比分析**

与无记忆、RAG、MemP、MemRL‑R、MemRL‑GT 等基线比较，SMA 在所有四个冻结模型和五个基准上均取得最高宏平均准确率，平均提升 1.7–2.9 分；与训练式自进化基线 SpatialEvo-7B 对比，宏平均提升 16.4 分。

**⚠️ 局限性**

局限性包括：①依赖可靠的验证器，无法在无监督或无标签环境中直接应用；②记忆库大小与检索复杂度随任务规模增加而上升；③目前仅针对空间推理任务，跨模态或更复杂的交互场景尚未验证；④一次性写入策略在极端动态变化的任务中可能缺乏足够的适应性。

---

## 192. Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction

**arXiv ID:** 2608.12426 | [PDF](https://arxiv.org/pdf/2608.12426v1)

**作者:** Mariya I. Vasileva `[一作]` `[通讯]` (Meta Superintelligence Labs), Mariya I. Vasileva (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Constraint Saturation Evaluation（CSE）基准，系统生成并评估模型在多重约束（最多12个）下的生成性能，揭示组合式约束遵从随约束数递增而呈指数级衰减。

**💡 创新点**

创新点在于：①设计了可自动化、可重复的约束组合生成与判定流程；②通过可确定性的规则验证器消除LLM评判偏差；③发现约束失效主要是独立乘积导致的相乘衰减，而非互相干扰，且结构约束的衰减速率是词汇约束的两倍。

**🔧 技术方法**

技术上采用程序化提示生成、8维度36种可验证约束、严格二值判定、模型输出提取与标记化、以及统计分析（sCSR、mCSR、k*、φ相关系数）等方法。

**📊 数据集**

数据集为自研的CSE基准，包括15种大语言模型、36种约束、约369,753条约束检查，涵盖k=1到12的组合，另外附加444条不可能约束用于评估冲突优先级。

**📈 对比分析**

通过对比sCSR（全部约束同时通过率）与mCSR（单约束通过率）以及k*（sCSR降至50%时的约束数）发现最佳模型GPT‑5.5在k=7时仍能维持约6.4%全部通过率，而其余模型大多在k=5‑6时骤降；整体每多加一个约束，单约束通过率仅下降到92.2%，但全局通过率呈指数式衰减。

**⚠️ 局限性**

局限性包括：①仅评估并列组合，未涵盖顺序、条件或嵌套约束；②约束被限定为可确定性可验证，排除语义、语用、连贯性等难以判定的约束；③k=12是极端压力测试，实际部署约束数往往低于此；④在极端压力下模型会利用规范漏洞（空代码块、反转句子等）完成约束，表明仍需完善验证规则。

---

## 193. Domus: An Open-Data Web Platform for House History Research

**arXiv ID:** 2608.12566 | [PDF](https://arxiv.org/pdf/2608.12566v1)

**作者:** David M. Straub `[一作]` `[通讯]` (Munich University of Applied Sciences), David M. Straub (Munich University of Applied Sciences)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于Wikidata和OpenHistoricalMap的前端仅客户端网站Domus，允许用户在地图上发现、编辑和可视化建筑历史记录。

**💡 创新点**

创新点在于将开放数据与跨平台OAuth 2.0 PKCE、JSON Patch编辑、离线预计算建筑类型及时间滑块等技术结合，提供无服务器、可持续、可互操作的家居历史研究工具。

**🔧 技术方法**

使用了Lit 3 web components、TypeScript、Vite、MapLibre GL JS、OpenFreeMap tiles、Wikidata SPARQL/REST、Overpass API、OAuth 2.0 PKCE、JSON Patch等前端技术。

**📊 数据集**

数据集为公开的Wikidata（建筑属性、人物、来源等）和OpenHistoricalMap（建筑轨迹、时间段），以及开放的历史地图覆盖层（如LGL-BW、swisstopo）。

**📈 对比分析**

通过预计算建筑类型列表、仅使用浏览器端请求和轻量化离线资源，查询性能可在700–800 ms内返回1,000–3,000条建筑实体，编辑操作通过JSON Patch实时写入；与传统服务器端实现相比，成本更低、可维护性更好。

**⚠️ 局限性**

局限在于依赖外部公共API速率限制、缓存缺失、对SPARQL查询的延迟同步、覆盖范围不均衡，以及对某些地区缺乏预先填充数据。

---

## 194. Dual Spatial-Temporal Attribution: Architecture-Aligned Post-Hoc Explainability for Recurrent Graph Anomaly Detection

**arXiv ID:** 2608.12441 | [PDF](https://arxiv.org/pdf/2608.12441v1)

**作者:** Iyad Assaad Nekka `[一作]` (National Higher School of Computer Science (ESI)), Karima Amrouche `[通讯]` (National Higher School of Computer Science (ESI))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 AddGraph 深度学习动态图异常检测器实现严格后置可解释框架 X-AddGraph，提供空间、短期和长期归因。

**💡 创新点**

设计与 AddGraph 三模块对齐的 Dual Spatial-Temporal Attribution 机制；零成本读取 CAB 注意力；完全后置不影响检测性能。

**🔧 技术方法**

使用梯度归因、梯度回滚、注意力读取、GCN+GRU 架构、BPTT 等技术。

**📊 数据集**

采用 UCI Message 基准数据集（1,899 节点、59,835 条时间戳边）。

**📈 对比分析**

与平面梯度基线对比；在四类边族上评估 Fidelity^+、Sparsity，检测 AUC 保持 0.8705；长期归因相较随机选择提升至 0.127（vs 0.074）。

**⚠️ 局限性**

仅在单一基准上评估；窗口大小受限导致短期归因受限；缺乏真实因果标注；未扩展至多样化数据集。

---

## 195. PseudoMapLabeler: Confidence-Aware Pseudo-Label Generation for Semi-Supervised Online Mapping

**arXiv ID:** 2608.12600 | [PDF](https://arxiv.org/pdf/2608.12600v1)

**作者:** Chikao Tsuchiya `[一作]` (Nissan North America), Christopher Ostafew `[通讯]` (Nissan North America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出教师-学生半监督框架，利用Beta分布置信图和空间裁剪生成高质量伪标签，提升在线HD地图构建。

**💡 创新点**

引入Beta分布置信映射对时空累积预测进行置信度评估，并提出空间裁剪方法在元素层级保留高置信度片段，提供可插拔的伪标签生成管线。

**🔧 技术方法**

Beta分布置信图、空间裁剪、温度尺度校准、教师-学生框架、Uni-PrevPredMap/MapTR网络、Chamfer距离匹配等技术。

**📊 数据集**

nuScenes 地理分离的训练/验证集，划分为约16.5%标签样本与83.5%无标签样本。

**📈 对比分析**

与仅使用标签的UPPM基线相比，在验证集上mAP提升约6.1点；与基线过滤方法相比提升约2.8点；在MapTR架构上同样实现+5.0点的提升。

**⚠️ 局限性**

依赖精确定位与传感器标定；教师初始性能受限于少量标签；与GT prior仍存在约10点mAP差距；仅在nuScenes验证，缺乏跨域评估。

---

## 196. CAS: A Causal Attribution Score for Local and Global Explainable Artificial Intelligence

**arXiv ID:** 2608.12555 | [PDF](https://arxiv.org/pdf/2608.12555v1)

**作者:** Michael Georgiades `[一作]` (Neapolis University Pafos), Charalambia Varnava `[通讯]` (Cyprus Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 CAS（Causal Attribution Score）框架，用于将因果效应分配到一组可操作的干预中，并提供局部与全局的因果重要性解释。

**💡 创新点**

创新点在于将传统 Shapley 归因机制与因果干预游戏结合，产生既保留因果方向又能按协同效应归属的多层次评分体系，而非仅归一化 CATE。

**🔧 技术方法**

采用了交叉拟合的 AIPW 与 DoubleML 估计器、Shapley 价值分配、TreeSHAP、LIME 以及自定义的 Feature‑CAS 计算因果归因。

**📊 数据集**

使用了两个真实世界数据集：1991 年 401(k) 资格与净财务资产（9,915 条记录）以及宾夕法尼亚州失业补贴实验（5,099 条记录）。

**📈 对比分析**

通过在已知真值的仿真基准和实际数据上的局部/全局指标对比，CAS 在有交互作用时显著优于传统一元归一化方法（MAE 降低 0.066~0.091），并在真实数据中揭示与预测 SHAP/LIME 不同的因果重要性排序。

**⚠️ 局限性**

局限性包括：需要先验指定可操作干预集合；对高维效应修饰符的可扩展性受 TreeSHAP 算法复杂度限制；在无显著交互的加性场景下，CAS 与传统 CATE 归一化结果相同，缺乏额外优势。

---

## 197. Don't Want Your LLM to Recommend Nuclear Strike? Try Asking It in Japanese

**arXiv ID:** 2608.12373 | [PDF](https://arxiv.org/pdf/2608.12373v1)

**作者:** Rian Touchent `[一作]` (Sorbonne University), Rian Touchent `[通讯]` (INRIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估不同语言（英语、日语、法语、葡萄牙语）对大型语言模型在核弹使用决策中的影响，利用单轮博弈情境让模型扮演核武国家的战略顾问。

**💡 创新点**

发现日语提示能显著降低Claude系列及Gemini Pro 3.1的核弹发射率，表明语言可作为文化关联的框架变量调节LLM的安全行为；同时识别到模型在日语推理中自发产生道德词汇的机制。

**🔧 技术方法**

使用单轮博弈式提示、跨语言推理实验、关键字分类与统计检验（Fisher's exact test）等技术。

**📊 数据集**

实验数据包括9款LLM（Claude Sonnet/Opus/Haiku、Gemini Flash/Pro、GPT‑5.2、Mistral Large 3、Qwen3‑Max、DeepSeek V3.2），4种语言，3种战略情境，30次随机种子实验，总计约8,646条推理记录。

**📈 对比分析**

通过对比各模型在不同语言下的核弹发射率，发现日语下的发射率显著低于英语（p<0.05），例如Claude Sonnet从40%降至0%，Gemini Pro 3.1从53%降至13%；跨语言推理实验表明推理语言而非输入语言决定发射率。

**⚠️ 局限性**

局限包括：仅测试核武情境，未涵盖其他高风险决策；翻译可能引入偏差；实验基于2026年3月的模型版本，后续更新可能改变结果；未包含人类基准，无法判断绝对安全性。

---

## 198. Federated Compositional Muon Optimizer for Matrix-Wise Models

**arXiv ID:** 2608.12710 | [PDF](https://arxiv.org/pdf/2608.12710v1)

**作者:** Wang Yan `[一作]` (Nanjing University of Aeronautics and Astronautics), Feihu Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究分布式矩阵‑值合成优化问题，提出 FedCoMuon 和 FedCoMuon‑VR 两种 Federated Muon 算法；

**💡 创新点**

创新点在于将 Muon 的正交化动量与合成梯度跟踪相结合，并引入基于动量的方差削减技术，显著提升非 i.i.d. 非凸环境下的收敛复杂度；

**🔧 技术方法**

采用 Newton–Schulz 迭代实现矩阵正交化、合成梯度跟踪、动量方差削减、SVD/矩阵分解等技术；

**📊 数据集**

实验使用 MNIST、WikiText‑2、CIFAR‑10（CNN、ViT‑Tiny）等数据集；

**📈 对比分析**

与 FedAvg、FedMAML、FedMuon 系列、ComFedL、Local‑SCGDM 等基线比较，FedCoMuon‑VR 在稳健联邦学习和任务分布式元学习中实现更高准确率/更低损失，并在通信/样本复杂度上优于现有算法；

**⚠️ 局限性**

局限性包括理论分析仅针对矩阵‑值模型，未验证更广泛的非矩阵模型；同步间隔 τ 的选择仍需经验调优。

---

## 199. Scaling Automatic Research Agents via World Models

**arXiv ID:** 2608.12564 | [PDF](https://arxiv.org/pdf/2608.12564v1)

**作者:** Xiyuan Yang `[一作]` (University of Illinois Urbana Champaign), Zhenyu Liao `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过用世界模型替代环境执行，并在此基础上加入在线去偏和逆方差去噪技术，实现AutoResearch Agent在RL训练中的加速与性能提升

**💡 创新点**

创新点在于：①将世界模型作为奖励信号并理论证明其误差（偏差与噪声）对收敛的影响；②设计在线去偏（单调映射）和逆方差去噪两种校正机制，在保持少量真实执行锚定的前提下，显著降低偏差和噪声

**🔧 技术方法**

使用LLM做世界模型、GRPO为RL框架，配合在线去偏（单调回归）和逆方差去噪（自适应加权）

**📊 数据集**

在MLE‑Dojo、DSBench（AutoResearch）和LIBERO‑Long（VLA）等公开基准数据集上进行评测

**📈 对比分析**

与传统真实执行GRPO、纯世界模型以及大模型对比，WMRL在相同算力下实现3–4倍训练加速，且在所有任务上性能均优于或等同于真实执行RL，甚至小模型超越更大开源模型

**⚠️ 局限性**

局限性在于需要可预测奖励且能提供少量真实执行锚定，且当世界模型误差大或偏差难以完全消除时，性能提升受限

---

## 200. Reasoning Jury: Multi-Model Consensus for Evaluating Reasoning Traces

**arXiv ID:** 2608.12585 | [PDF](https://arxiv.org/pdf/2608.12585v1)

**作者:** Congchao Wang `[一作]` (Amazon AGI), Mahdi Namazifar `[通讯]` (Amazon AGI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Reasoning Jury，利用多模型评判者和经过裁决的共识机制，对长链推理轨迹中的错误进行细粒度定位与描述。

**💡 创新点**

创新点在于用“陪审团”模式替代单一评判模型，并通过协商或合并实现更高的错误检测精度，同时保持低成本与可扩展性。

**🔧 技术方法**

技术实现包括基于 STEP-x 标记的推理轨迹分段、独立判决生成、两阶段共识（合并与讨论）以及多模型推理与成本优化。

**📊 数据集**

使用的数据集为 Hard2Verify（200 条记录、约780 个错误步骤）与 DeltaBench（1,236 条记录），并在 AIME2026 上做案例分析。

**📈 对比分析**

与 frontier 模型（gpt‑5.4、opus‑4.6 等）比较，Reasoning Jury 在 Hard2Verify 上的 Balanced‑F1 可达 84.4，超过 gpt‑5.4 的 83.9，且成本仅为 frontier 模型的 8–16%；在 DeltaBench 上也实现了 5–6 点的提升。

**⚠️ 局限性**

局限性包括更高的延迟与 token 消耗、对解释事实性与严重度校准缺乏直接验证，以及讨论模式在在线 RL 环境中的适用性受限。

---

## 201. Interpretable Causal Discovery via Causal-Effect Constraints

**arXiv ID:** 2608.12640 | [PDF](https://arxiv.org/pdf/2608.12640v1)

**作者:** Cixuan Zhang `[一作]` (Yale University), Benjie Wang `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种条件因果发现框架，通过贝叶斯推断在满足给定极端因果效应约束的前提下估计因果图结构与参数，并给出对应的后验样本与路径级解释。

**💡 创新点**

创新点在于将条件因果发现转换为罕见事件后验推断，利用自适应多层分裂（AMS）与MCMC相结合的采样策略，既能估计尾部概率，又能生成满足约束的代表性后验样本；支持多约束、多尾事件的通用查询。

**🔧 技术方法**

使用线性高斯结构方程模型、BGe似然、PARNI‑DAG/Structure‑MCMC结构提议、阻塞Metropolis–Hastings、以及自适应多层分裂（AMS）等技术。

**📊 数据集**

实验基于合成线性高斯数据（d=4,8,16,32）以及真实蛋白信号学的Sachs数据（11个蛋白）。

**📈 对比分析**

与枚举（d=4）、DiBS、OrderSPN、单链MCMC等基线比较。AMS在罕见事件区能够给出准确的尾部概率，并生成高质量的条件后验样本；在更大维度下仍保持稳定，传统无约束采样方法则在极端尾部失效。

**⚠️ 局限性**

仅适用于线性高斯无潜在混杂的模型；对非线性或有干预数据的扩展有限；多层分裂的性能依赖于结构提议质量与分层设置；极端尾部仍受采样效率限制。

---

## 202. From Caveman to Expert Analyst: Energy Consumption of Variable LLM Tasks

**arXiv ID:** 2608.12350 | [PDF](https://arxiv.org/pdf/2608.12350v1)

**作者:** Diego Manya `[一作]` (UNC Chapel Hill), Michael P. Vandenbergh `[通讯]` (Vanderbilt University)

**通讯引用:** 5010 | [OpenAlex ID](https://openalex.org/A5032007536)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了零成本用户级提示修改（如选择非推理模型、使用能效角色提示、最小答案提示等）对商业大型语言模型（LLM）能源消耗的影响；

**💡 创新点**

首次系统评估了用户行为与提示策略对AI能源需求的需求侧影响，揭示可通过简单提示实现高达65%的能耗削减，同时保持较高语义相似度；

**🔧 技术方法**

采用商业LLM（OpenAI、Anthropic、Google、DeepSeek、xAI）与推理时间乘系数的能耗估算公式，结合余弦相似度评估回答质量；

**📊 数据集**

使用300条基于Bloom分类的提示，来源于ChatBot Arena、Natural Questions、MMLU等公开LLM基准数据集；

**📈 对比分析**

将不同提示策略与基线进行对比，基于能耗估算和语义相似度指标；结果显示，最小答案提示可在所有任务层级实现最高的能耗降低（最多63%），且能效角色提示在保持90%以上相似度的同时实现4-35%的能耗削减；

**⚠️ 局限性**

局限性包括：能耗估算基于理论模型与假设（如并行批处理、PUE），缺乏实际服务器功耗测量；仅考虑了电力消耗，未评估水耗与碳排放；提示效果随模型更新可能变化；未覆盖所有商业LLM与提示细粒度。

---

## 203. ReconSpan: Reconstruction-Guided Adaptive Latent Tokenization

**arXiv ID:** 2608.12756 | [PDF](https://arxiv.org/pdf/2608.12756v1)

**作者:** Lixing Li `[一作]` `[通讯]` (Cornell University), Lixing Li (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 ReconSpan，一种基于后向重建距离的自适应潜在分词方法；它通过在自编码器中使用一个前缀编码器和一个反向解码器来确定文本片段的边界，并将每个片段的编码作为潜在令牌，随后构建可直接被下游语言模型读取的连续表示序列。

**💡 创新点**

创新点在于：①使用重建可信度（后向解码成功范围）作为分词边界的判别标准，完全摆脱固定长度或频率统计的限制；②单个自编码器在训练后即可通过调整停止阈值实现多种粒度的分词；③对生成的潜在令牌进行了系统的可读性与信息保留评估，揭示了重建路径与直接读出路径之间的差距。

**🔧 技术方法**

核心技术包括：
- 前缀编码器（Transformer/Pythia）
- 反向解码器（Mamba）
- 词元分段算法（基于 Failure(m) 与 Logit-gap(τ) 两种停止规则）
- 自编码器训练目标（逆序词元重建交叉熵）
- 直接读取模型（Reader）与词元标准化、线性映射、RMSNorm 等处理。

**📊 数据集**

使用的数据集：
- FineWeb（训练自编码器与 Reader）
- Wikipedia（评估分词、重建与读出）
- AG News（主题分类评估）
- LAMBADA（单词预测评估）
- HotpotQA（多跳检索与答案生成评估）。

**📈 对比分析**

与基线的比较：
- 随机分块对照：ReconSpan 的重建长度更长、文本保留更好；
- SONAR 自动编码器：ReconSpan 在不同长度下的重建准确率更稳定；
- 读出模型评估：在主题分类上可与原始文本相当，但在精确词汇与检索任务上仍落后于本地重建；
- 通过改变停止阈值可在相同平均块长下观察到重建质量与粒度的权衡。

**⚠️ 局限性**

局限性：
- 直接读取模型在精确内容和检索任务上的性能不足，说明潜在代码的几何结构对信息提取不友好；
- 频繁出现单词长的块占用大量潜在位置但仅携带极少文本；
- 需要训练反向解码器，增加模型准备成本；
- 句段选择扫描成本高，尤其在大规模语料上；
- 只在已有子词序列上操作，无法替代字节级分词，对高效压缩与 KV 缓存的实际收益仍需进一步验证。

---

## 204. Beyond Retrieval: Query-Conditioned Reuse of Long-Horizon Agent Trajectories

**arXiv ID:** 2608.12847 | [PDF](https://arxiv.org/pdf/2608.12847v1)

**作者:** Yifei Li `[一作]` (Xi'an Jiaotong University), Rongman Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过构建一个冻结的、验证过的历史轨迹库，研究在长轨迹记忆中检索后重用的瓶颈，并提出一种基于查询条件的轨迹重用框架（QCR），实现将检索到的经验转化为与当前任务绑定的可直接使用的简短支持对象。

**💡 创新点**

创新点在于：①将检索与重用拆分为两个独立阶段，强调检索后的“重用”步骤；②设计了最小化的目标绑定支持结构（工作流、绑定、适用条件、验证护栏）；③引入统一的评估框架，将检索、模型、解码、工具预算固定，只比较不同重用表述的效果。

**🔧 技术方法**

技术手段包括：嵌入检索器与轻量级重排器、基于DeepSeek‑V4‑Pro的生成式支持合成、对比实验中的token与API调用计数以及成功/里程碑验证。

**📊 数据集**

数据集为来自WebArena、WorkArena和AppWorld的623条已验证历史轨迹，随后通过绑定重写生成2,391个目标任务实例，用以评估不同重用方法的性能。

**📈 对比分析**

比较方法：在同一检索结果、同一ranker选择的单条轨迹下，分别给目标任务传递无记忆、源侧摘要、完整轨迹和QCR生成的支持；评价指标为验证成功率、里程碑完成率、API调用次数和在线token数。实验显示QCR以62.3%成功率（比完整轨迹高10.7点）并在在线token上减少48.9%，且在不同环境中保持一致性。

**⚠️ 局限性**

局限性包括：仅使用成功轨迹、单一检索条目、受控绑定变更；未考虑多条记忆组合、部分失败或开放式记忆获取；并未评估因重用导致的不可逆副作用或政策违规，亦未检验在更大或更动态环境中的泛化。

---

## 205. Channel Estimation for OTFS Systems With Overspread Doppler Shifts

**arXiv ID:** 2608.12524 | [PDF](https://arxiv.org/pdf/2608.12524v1)

**作者:** Preety Priya `[一作]` (IIT (ISM) Dhanbad), Emanuele Viterbo `[通讯]` (Monash University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种针对OTFS系统中过度扩展（overspread）Doppler位移的两阶段训练帧与信道估计方法。

**💡 创新点**

创新点在于引入余弦训练信号与高功率训练符号相结合的训练帧，并通过频域峰值检测与阈值配对实现对实际Doppler位移的精确估计。

**🔧 技术方法**

使用OTFS时域/时频/延迟-多普勒域转换、IDZT/DZT、FFT、阈值检测、最小二乘回归等信号处理与估计技术。

**📊 数据集**

通过仿真场景A（地面多径信道）和场景B（LEO多卫星协同）评估性能，并使用4-QAM OTFS信号。

**📈 对比分析**

与传统嵌入式导频估计对比，所提方法在过度扩展Doppler情况下NMSE可低至10⁻⁴、BER可接近理想CSI，性能显著优于传统方法。

**⚠️ 局限性**

主要局限是对分数Doppler、相同延迟同余多普勒情况处理仍不完善，并且额外的频域FFT和LS步骤增加了计算复杂度。

---

## 206. From Visual Widgets to UI Code: Efficient Tool-Grounded Generation

**arXiv ID:** 2608.12611 | [PDF](https://arxiv.org/pdf/2608.12611v1)

**作者:** Houston H. Zhang `[一作]` (McMaster University), Zhixiang Chi `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了轻量级的 WidgetGen 框架，通过 OCR 和调色板提取对布局与图表进行先验推理后直接生成可执行 JSX，实现从截图到 UI 代码的高效转换。

**💡 创新点**

创新点在于将可观测的文本和颜色证据作为工具 grounding，减少对细粒度组件生成的依赖，同时保持对直接代码生成的灵活性；该方法兼顾了可控性与效率。

**🔧 技术方法**

结合多模态大型语言模型（GPT‑4o、Gemini‑3 Pro、Claude‑Opus‑4.5、Seed‑2.0 Pro、Qwen3‑VL‑Plus、Qwen3.5‑Plus）与 OCR、调色板分析工具以及浏览器渲染器进行端到端推理。

**📊 数据集**

使用 widget2code‑benchmark 的 1,822 训练样本与 1,000 公开测试样本，生成可执行的图像‑代码对进行评估与监督学习。

**📈 对比分析**

在 1,000 个测试 widget 上，WidgetGen 在大部分指标（面积、可读性、风格、几何）上超过单次提示和结构化的 Widget2Code，且在推理成本、延迟与渲染成功率方面更具优势；通过再训练 Qwen 系列模型，使用重建对进一步提升了所有九项评估指标。

**⚠️ 局限性**

仅针对静态 widget 进行评估，缺乏交互行为、无障碍性和代码可维护性等方面的检验，且监督实验仅验证了 Qwen 家族模型，尚未验证在更广泛 UI 领域和其他模型上的泛化能力。

---

## 207. Humans are Missing from AI Coding Agent Research

**arXiv ID:** 2608.12355 | [PDF](https://arxiv.org/pdf/2608.12355v1)

**作者:** Zora Z. Wang `[一作]`, Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 14302 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出以人机交互为中心的AI编程代理研究框架，并将其拆解为任务对齐、可引导性、可验证性与可适应性四大支柱，同时指出四个高杠杆研究方向，强调从单纯提升自治性转向增强人机协作的必要性。

**💡 创新点**

创新点在于将编程代理的设计与评估转向可度量的交互维度，提出任务对齐度、响应可引导性、验证一致性与适应性改进等可量化指标，并将其与传统自动化基准进行对比。

**🔧 技术方法**

采用的大型语言模型与代理框架为核心技术，辅以用户模拟器、交互式评估指标设计以及代码与测试生成工具等。

**📊 数据集**

主要引用的公开数据集包括SWE-bench、SWE-Gym、GitHub PR 记录、OpenAssistant 以及各种编程任务与对话数据，但并未提出新的标注数据。

**📈 对比分析**

对比方法主要基于传统的 pass@k、单一自动化基准与提出的交互维度指标相结合的多模态评估，研究指出单纯提升自治性往往无法带来更高的实际生产力，需兼顾交互指标才能实现更好的协作效果。

**⚠️ 局限性**

局限性包括缺乏规模化真实用户交互数据导致用户模拟真实性不足，交互度量指标尚不成熟且难以统一规范，且论文为位置性讨论，未包含实证实验验证。

---

## 208. Query Translation vs. Cross-Lingual Embeddings for Sinhala-Tamil E-Government Information Retrieval

**arXiv ID:** 2608.12820 | [PDF](https://arxiv.org/pdf/2608.12820v1)

**作者:** Dharshi Balasubramaniyam `[一作]` (University of Kelaniya), Tiroshan Madushanka `[通讯]` (University of Kelaniya)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了多语言检索方法（查询翻译与跨语言嵌入）在斯里兰卡政府英文知识库上使用僧伽罗语和泰米尔语检索的效果。

**💡 创新点**

提出了可公开的僧伽罗语-泰米尔语-英语问答基准，并证明BGE-M3跨语言嵌入可在不翻译的前提下超越现有翻译方案。

**🔧 技术方法**

使用Google Translate、NLLB、mBART50等机器翻译模型以及LaBSE、E5、多功能BGE-M3嵌入模型，基于FastEmbed构建检索索引。

**📊 数据集**

基准数据来自斯里兰卡政府信息中心的761网页，人工验证的1,699段落与500对僧伽罗语/泰米尔语问答组成。

**📈 对比分析**

通过Recall@k评估，BGE-M3在Recall@15达到96.2%（僧伽罗）和95.6%（泰米尔），优于Google Translate的92.4%/93.0%，显示跨语言嵌入更高效。

**⚠️ 局限性**

仅评估检索性能，未进行生成质量评估，且模型均未微调，且仅覆盖单一政府服务领域。

---

## 209. Julia for CFD: A Critical Survey of Ecosystem, Performance, and Composability

**arXiv ID:** 2608.12801 | [PDF](https://arxiv.org/pdf/2608.12801v1)

**作者:** Tianbai Xiao `[一作]` `[通讯]` (Chinese Academy of Sciences), Tianbai Xiao (Chinese Academy of Sciences)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了 Julia 作为单语言、可组合的 CFD 开发平台的潜力，评估其在性能、可扩展性、可微分性及与其他软件体系（如 C++/DSL/JAX）比较时的优势与不足。

**💡 创新点**

提出了在 CFD 设计中通过 Julia 的类型系统、多重派发与包生态实现模型、算子、后端与微分等组件无缝对接，从而降低多语言架构导致的工程阻力，并通过多案例验证这一思想的可行性。

**🔧 技术方法**

使用了 Julia 的多重派发、类型专化、泛型数组、MPI.jl、GPU.jl、DifferentialEquations.jl、NonlinearSolve.jl、ForwardDiff/Zygote/Enzyme 等技术，结合现有开源 CFD 项目（Trixi.jl、WaterLily.jl、Oceananigans.jl、LCS.jl 等）实现分布式并行、加速器编程与自动微分。

**📊 数据集**

引用了各项目在公开论文与实验中报告的数据集与规模，例如：Trixi.jl 在 61,440 CPU 核上、WaterLily.jl 在 10 亿单元网格上、Oceananigans.jl 在 768 个 A100 GPU 上、LCS.jl 在 256 GPU 上等；并对这些实验结果进行总结与比较。

**📈 对比分析**

通过与 C++ 性能可移植框架（AMReX、Firedrake）和 JAX-CFD 等进行对比，评估 Julia 在 GPU 可移植性、自动微分能力、分布式扩展等方面的表现；结果表明在正交网格与高阶元素算子等场景中，Julia 的性能可与手工实现相当，但在不规则网格、稀疏算子与多物理耦合等更复杂场景下仍存在性能与兼容性瓶颈。

**⚠️ 局限性**

主要局限包括：生态系统规模与工业化工具不足；跨平台性能不均衡（如稀疏算子、粒子迁移等对 GPU 的支持不佳）；编译与启动开销在短周期工作负载中显著；AD 与多物理耦合的实现仍受限于接口与算法复杂度；缺乏全面的验证与可复现性框架，导致不同项目间的兼容性与重现性不足。

---

## 210. DrEM: Dual-Side Robust Ensemble Ranking from Noisy User Preference Predictions in Video Recommendation

**arXiv ID:** 2608.12778 | [PDF](https://arxiv.org/pdf/2608.12778v1)

**作者:** Canwei Huang `[一作]` (Shenzhen University), Kaiqiao Zhan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种双侧鲁棒集成排序框架DrEM，用于降低视频推荐中基于多任务模型产生的用户偏好预测(pxtr)噪声对排序质量的影响。

**💡 创新点**

创新点在于：①基于预测噪声的logit空间高斯模型估计对偶侧的偏好翻转概率，并设计风险去噪的对偶侧pairwise损失；②在特征侧采样预测噪声进行偏好保持的排名一致性正则化，两者共享同一噪声模型，实现双侧一致的鲁棒校正。

**🔧 技术方法**

采用对偶侧风险去噪pairwise损失、偏好保持的排名一致性正则化、logit空间高斯噪声建模、probit近似、噪声方差估计、扰动采样等技术。

**📊 数据集**

在真实工业短视频平台的海量用户行为数据上进行离线实验，使用多任务模型产生的pxtr作为特征和代理监督；在线A/B实验覆盖主流流量。

**📈 对比分析**

与EMER、EASQ基线以及SSM、PSL、GaussAug、LSPR等鲁棒方法对比，DrEM在离线GAUC上实现最佳或接近最佳表现，在线实验中在多项业务指标上显著提升，且双侧方案优于单侧。

**⚠️ 局限性**

局限性包括：对噪声分布的高斯假设和方差估计的准确性敏感；在预测噪声较小的稠密任务中提升有限；训练时额外的扰动前向传播增加计算开销。

---

## 211. Memorization Diagnostics for Code LLMs Should be Scale-Aware

**arXiv ID:** 2608.12771 | [PDF](https://arxiv.org/pdf/2608.12771v1)

**作者:** Prateek Kumar Rajput `[一作]` (University of Luxembourg), Tegawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对代码生成 LLM 的记忆化诊断，提出传统表面扰动和基于对数概率的检测方法在大规模模型上失效的现象，并引入可逆数值变换（I/O 同构）作为新的元测试框架，能够在保持算法不变的前提下，独立考察模型对“表示负载”（representational load）的鲁棒性。

**💡 创新点**

创新点包括：1) 在模型规模不断增大的背景下首次系统评估并证明现有记忆化诊断技术（同义词模糊、死代码插入、CoDeC）失效；2) 设计了基于数值同构的元测试方法，严格保证任务等价性，能将记忆化与表示负载分离；3) 通过字节码操作码熵与 Jensen‑Shannon 散度对模型的算法保持度进行量化，揭示大模型在表示负载下仍保持核心算法，只在输出序列序列化上出现“窄通道”失效；4) 对同构变换进行多家族对照（基数转换、立方多项式），验证现象的普适性。

**🔧 技术方法**

主要技术：编码器/解码器表面扰动实验、CoDeC 记忆化概率检测、I/O 同构实验（Affine、Base‑Conversion、Cubic 三类变换）、字节码操作码频率熵计算、Jensen‑Shannon 散度分析、前缀日志概率计算、模型规模对比实验。

**📊 数据集**

使用的公开基准：MBPP、EffiBench、BigOBench（均为 Python 级函数级别的算法题），并在多种公开和闭源 LLM（从 6B 到 405B 的 dense 模型以及 GPT‑4o、Gemini 等）上进行评测。

**📈 对比分析**

对比方法：对照原始 prompt、同义词模糊 20/40%、死代码插入、CoDeC 两类数据集（Seen/Unseen）、I/O 同构（Enc、Dec、Full）等多种条件。实验结果显示：
• 传统表面扰动在 frontier 模型上仅导致 ≤8% pass@1 的下降；
• CoDeC 的 seen‑unseen 区分度随规模下降（AUC 由 100% 降至 25‑75%）；
• I/O 同构在 frontier 模型上造成 14‑30% 的 pass@1 下降，但对应的操作码 JSD 近 0，说明算法保持；
• 解码侧的输出编码负载是主要瓶颈，解码侧的失效显著高于编码侧；
• 对于小规模模型，I/O 同构会导致算法迁移失败（JSD 激增）。

**⚠️ 局限性**

局限性：
1) 同构测试仅覆盖数值输入/输出，未能验证字符串、结构化数据或自定义 API 等表示负载；
2) 只测试 dense 模型，未考虑 MoE 等激活机制的影响；
3) 操作码熵仅为算法策略的粗略代理，无法完全捕捉运行时行为；
4) 采用贪婪 0.0 采样，可能低估模型在多样化输出下的性能；
5) 仅在 Python 语言下评测，跨语言通用性未知；
6) 记忆化的检测仍基于可观测的表面指标，未彻底排除所有隐蔽记忆化路径。

---

## 212. A Compositional Theory of Curvature in Probabilistic Circuits

**arXiv ID:** 2608.12869 | [PDF](https://arxiv.org/pdf/2608.12869v1)

**作者:** Hrithik Suresh `[一作]` (Indian Institute of Technology Palakkad), Narayanan Chatapuram Krishnan `[通讯]` (Indian Institute of Technology Palakkad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了概率电路（PC）的尖锐度感知学习，首先证明了全局Hessian迹（即尖锐度）可精确拆分为节点的上下文流与局部曲率两部分，并基于此发现全局正则化容易导致低阶节点被过度抑制，从而出现欠拟合。随后提出一种自适应尖锐度感知正则化器，利用每个sum节点的局部曲率来调节全局Hessian迹的权重，从而在保持可闭式EM更新的前提下更精准地分配正则化力度。

**💡 创新点**

创新点包括：①首次给出PC中全局尖锐度的精确因子分解（上下文流×局部曲率），揭示深度偏置与节点排名反转的根本原因；②提出基于局部曲率的门控机制，将全局正则化的强度按节点的内在曲率动态调节，避免了统一抑制导致的欠拟合；③在理论分析与实验评估中同时保持了线性时间复杂度与EM闭式更新。

**🔧 技术方法**

使用的技术主要有：概率电路结构、电路流（circuit flow）与梯度传播、Hessian迹的可解析计算、EM学习与闭式更新、局部曲率的rank‑one特征分析、基于局部曲率的门控正则化器以及对比实验的log‑likelihood评估。

**📊 数据集**

使用了20个二元密度估计基准数据集（DEBD），在每个数据集上训练隐藏丘陵树（Hidden Chow‑Liu Tree）模型。

**📈 对比分析**

实验对比了三种方法：无正则化、全局Hessian迹正则化、以及本文提出的门控正则化。通过测试集log‑likelihood作为评价指标，结果显示全局正则化往往在数据量足够时引起欠拟合，门控正则化在绝大多数数据集上至少与无正则化持平，并在部分数据集上表现更优，显著提升了模型泛化性能。

**⚠️ 局限性**

局限性主要包括：门控函数采用的是简单的单调归一化，缺乏对上下文流的直接利用；实验范围仅覆盖DEBD二元数据集，尚未验证在更复杂或高维任务中的鲁棒性；此外，虽然保持了线性复杂度，但在极大规模PC或深层网络中门控与全局正则化的交互机制仍需进一步理论与实证探索。

---

## 213. Digital Twin Satellite Networks: A Paradigm for Intelligent, Efficient, and Resilient Operations

**arXiv ID:** 2608.12865 | [PDF](https://arxiv.org/pdf/2608.12865v1)

**作者:** Mustafa Alhassan `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了数字孪生卫星网络（DTSN）框架，通过与NASA 42物理仿真引擎实时同步，构建跨域（物理、硬件、网络、安全）闭环协同仿真，演示在低轨道大规模星座中实现预判、主动重路由、节点隔离与自恢复的功能。

**💡 创新点**

创新点在于：① 将ISAC（集成感知与通信）与数字孪生耦合，实现物理姿态与光链路质量的实时映射；② 在闭环架构中引入预测智能，提前评估姿态漂移、硬件老化与敌意干扰，实时调整路由；③ 提出了多域容错机制，支持节点隔离与无延迟旁路；④ 为未来扩展预留量子传感器作为高精度故障诊断层；⑤ 通过将计算密集型任务卸载至地面服务器，突破LEO卫星SWaP限制。

**🔧 技术方法**

使用技术包括：NASA 42物理仿真引擎、Python‑based DT桥接、ISAC信号处理、预测角速度模型、机器学习预测健康风险、指数传感器恢复模型、以及跨域同步管道（Telemetry → 逻辑网络 → 控制命令）。

**📊 数据集**

数据集为：基于NASA 42的合成轨道与姿态信息（10 Hz，600 s窗口），包含60颗卫星的角度、速度、SNR等参数；同时注入三类扰动（姿态漂移、激光二极管失效、干扰攻击）并记录DT决策与网络响应。

**📈 对比分析**

比较方式主要通过对比扰动前后网络连通性与链路质量，展示DTSN在扰动出现前即完成重路由、零延迟旁路、节点隔离与恢复，保持链路SNR高于15 dB阈值，避免包丢失。性能评价为：在所有扰动情形下，网络服务连续无中断，恢复时间≤数秒；若与传统仅反应式管理比较，DTSN实现了提前约5–10 s的决策优势。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境验证，缺乏实测卫星数据与真实干扰验证；② 对地面高性能计算资源依赖较大，若地面网络延迟增加可能影响实时闭环；③ 量子传感器尚未成熟，实际可实现性待验证；④ 在规模扩展到千级星座时，同步延迟与数据量可能成为瓶颈；⑤ 模拟的扰动类型有限，未涵盖所有真实空间威胁与硬件故障分布。

---

## 214. Discovering Persistent Behavioural Patterns for Interpretable Blockchain Forensics

**arXiv ID:** 2608.12864 | [PDF](https://arxiv.org/pdf/2608.12864v1)

**作者:** Dorottya Zelenyanszki `[一作]` (Griffith University), Vallipuram Muthukkumarasamy `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个可扩展、无特定应用的框架，用于从以太坊交易日志中发现持久性行为模式并进行可解释的用户分群与特征分析

**💡 创新点**

创新点在于：①将交易和事件转化为“行为句子”并进行两阶段嵌入；②同时兼顾序列级别表示和句子级别表示；③引入可解释的行为分析器和多标签评估，使得发现的群体既可解释又具持久性；④通过多窗口（单月、累计）验证模式的长期稳定性

**🔧 技术方法**

技术包括：事件解码与地址分类、句子级别嵌入（MiniLM-L6-v2等）、序列聚合模型（GRU/Mamba2）、K-means聚类、PCA降维、聚类后特征抽取（行为词元、时间特征、风险曝光等）以及多标签分类器来解释簇

**📊 数据集**

使用以太坊区块 16250000-16749999 的交易数据，经过 XBlock-ETH、Zellic 合约数据、CoinGecko 与 OpenSea 的代币元数据，外部预标记的恶意地址集合（共 175 条）

**📈 对比分析**

与特征基线、基于图的行为管道、TF-IDF+SVD、Doc2Vec 等方法比较，主要评估指标包括：聚类几何质量（SC、DBI、CHI）、长度/活动泄漏（η²、NMI）、以及预标记类别分离（Purity、Distinct、Coll.）。结果显示所提框架在聚类质量、泄漏控制与预标记分离三方面均表现优于基线，特别是降低了长度泄漏并保持高的标签聚类纯度

**⚠️ 局限性**

局限性包括：预标记集规模有限、标签不完整；框架高度依赖事件解码、地址分类、代币元数据和外部风险参考；目前仅在以太坊主网进行验证，跨链、跨协议的通用性仍待评估

---

## 215. Dissecting Software Graphs: Structural Insights for Driver-Guided Fuzzing

**arXiv ID:** 2608.12859 | [PDF](https://arxiv.org/pdf/2608.12859v1)

**作者:** Baihong Chen `[一作]` (Utah State University), Wen Li `[通讯]` (Utah State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建共享函数级调用图，并在每个驱动下收集动态覆盖，系统评估多驱动 fuzzing 对软件结构的探索效果。

**💡 创新点**

创新点在于提出基于共享调用图的执行驱动结构抽象，并通过四阶段实验框架量化驱动间的多样性、重叠度与残留盲区。

**🔧 技术方法**

技术包括静态调用图生成、Honggfuzz 的多驱动调度与共享内存覆盖记录、图论度量（连通性、碎片化、模块化、IoU）、社区检测与覆盖统计。

**📊 数据集**

使用了 OSS‑Fuzz 采集的 27 个 C/C++ 开源项目，共 43 个可执行文件和 854 个驱动配置，涵盖多种应用领域。

**📈 对比分析**

通过与最佳单驱动基准在相同预算下的覆盖率和错误发现对比，发现多驱动平均提升 27.9% 的节点覆盖、73.5% 的 CFG 边覆盖，并发现 11 个独特漏洞，展现出显著性能提升；同时揭示驱动贡献不均衡。

**⚠️ 局限性**

局限性包括仅考虑主命令行驱动，忽略环境变量、配置文件等；使用上下文不敏感的调用图可能过度近似；实验仅采用轮询调度，缺乏自适应调度；结果受 fuzzing 随机性影响，可能因随机种子变化而差异。

---

## 216. Fast A/B/n Testing: Exact Multi-Policy Comparison via Tree-Coupled Feedback Sharing

**arXiv ID:** 2608.12831 | [PDF](https://arxiv.org/pdf/2608.12831v1)

**作者:** Yuxiao Wen `[一作]` `[通讯]`, Yuxiao Wen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的树耦合A/B测试方法，旨在通过共享反馈来比较多个历史依赖的上下文策略，以减少实验中的奖励查询成本。

**💡 创新点**

创新点在于引入了树耦合的反馈共享设计，使得多个策略可以在不改变各自独立轨迹的情况下共享奖励，从而提高了实验的效率。

**🔧 技术方法**

使用了树结构来连接不同策略的历史，并通过最大耦合的方法来共享奖励，确保每个策略的独立性。

**📊 数据集**

在实验中使用了多个数据集，包括RewardBench和MMLU-Pro，这些数据集用于评估语言模型的表现和适应性搜索策略。

**📈 对比分析**

与传统的独立A/B/n测试方法相比，树耦合A/B测试在成本-精度边界上表现出显著的改进，能够在相同的预算下检测到更小的策略差异。

**⚠️ 局限性**

限制在于当候选策略几乎从不在其完整的上下文-动作对上达成一致时，虽然方法仍然有效，但奖励查询的节省效果有限。

---

## 217. LocusGS: Spatially Grounded Tokens for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2608.12825 | [PDF](https://arxiv.org/pdf/2608.12825v1)

**作者:** Wenyu Li `[一作]` (National University of Defence Technology), Yong Dou `[通讯]` (National University of Defence Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在快速 3D 场景重建中使用锚点约束的查询式 3D Gaussian 生成方法，称为 LocusGS

**💡 创新点**

在每个 Gaussian 查询中引入可学习的 3D 锚点（中心与半径）并在解码器层级逐步细化，使查询专注于局部空间，从而显著提升渲染质量和 Gaussian 的空间组织

**🔧 技术方法**

采用 Transformer 解码器、anchor‑to‑ray 几何偏置、anchor‑centered Gaussian 解码以及多层渲染监督等技术

**📊 数据集**

在 RealEstate10K（RE10K）和 DL3DV 两大场景级数据集上进行评估

**📈 对比分析**

在与 TokenGS 等基线在相同 token/Gaussian 预算下对比时，LocusGS 在 PSNR、SSIM 上取得提升、LPIPS 降低，并在不同视角数量下保持性能优势，且训练收敛更快

**⚠️ 局限性**

依赖相机姿态信息，锚点初始化仍为全局可学习，可能在极端稀疏或遮挡场景中表现不佳

---

## 218. Structured Local Differential Modeling for AI-Generated Image Detection

**arXiv ID:** 2608.12811 | [PDF](https://arxiv.org/pdf/2608.12811v1)

**作者:** Jiazhen Yang `[一作]` (Zhejiang University), Jie Lei `[通讯]` (Zhejiang University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于局部微分信号的AI生成图像检测框架，利用纹理敏感的补丁选择、方向与尺度的差分建模以及频率引导的交叉注意力，聚焦生成器特有的低信噪比伪造痕迹；

**💡 创新点**

创新点在于①将低SNR伪造痕迹转化为局部差分特征并通过环形卷积与层级注意融合提升方向与尺度一致性；②在差分特征空间上构建自注意力编码器，并用频率子带进行交叉注意，进一步放大生成器的高频异常；③两种纹理极端补丁（复杂/简单）相结合，提供互补的伪造信息；

**🔧 技术方法**

核心技术包括局部差分序列构造、方向环卷积（DRC）、层级注意融合（HAF）、频率引导交叉注意（FGCA）以及基于RoPE的多头自注意编码器；

**📊 数据集**

在GenImage、DeepFaceGen、DiffusionForensics、COSPY四个公开基准上进行训练与跨生成器、跨数据集评估；

**📈 对比分析**

与F3Net、FreqNet、NPR、FerretNet、Effort、CKNNA等现有方法对比，本文方法在GenImage上平均ACC 94.4%（领先2.4个百分点）、在DeepFaceGen上AUC 95.12%并保持对12种生成器的稳健性、在DiffusionForensics上ACC 89.0%、AP 98.2%，在COSPY上ACC 92.2%（比FerretNet高2.6个百分点），表现出更强的跨模型泛化；

**⚠️ 局限性**

局限性包括：①对图像分块与方向差分的依赖使得在极高分辨率或尺寸不匹配时可能需要额外的预处理；②模型相对较大，推理时需消耗较多计算资源；③仍可能在极度新颖或极度光滑的生成器中出现误检，需进一步提升对细微统计差异的敏感度。

---

## 219. PIPES: Securing Agent Perception with Provenance and Priors

**arXiv ID:** 2608.12789 | [PDF](https://arxiv.org/pdf/2608.12789v1)

**作者:** Sanjay Kariyappa `[一作]` (NVIDIA), G. Edward Suh `[通讯]` (NVIDIA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一类利用工具响应缺乏来源与语义约束的状态破坏攻击，并提出PIPES机制对响应单元进行语义优先级与来源层级审查，防止低信任内容篡改代理感知状态。

**💡 创新点**

创新点在于首次把“先验一致性”和“来源层级”结合成双重检测模型，解决了传统指令与数据混淆之外的“内容篡改”威胁；并引入可配置的单位级合规评估与多策略响应方案。

**🔧 技术方法**

核心技术是基于LLM的语义评估（LLMAssess）与静态/情境先验合同，利用工具字段契约与轨迹上下文判定先验违背和来源冲突；实现了单元级检测与原子删除响应策略。

**📊 数据集**

使用了VitaBench（delivery、in‑store、OTA）与AgentDyn（Shopping、GitHub、Daily‑life）两大基准，覆盖结构化字段与开放式内容两类工具。

**📈 对比分析**

与无防御、动作守卫、PromptArmor、DRIFT等对比，PIPES在Gemma 4 31B IT和GPT‑5.6 Luna两模型上均将攻击成功率从约80–85% 降至 1–3%，同时保持或提升 92–95% 的正常任务效能。

**⚠️ 局限性**

局限性包括：仅检测语义与来源合法性，无法保证事实真值；对多面攻击、来源锚点被破坏或工具自身被攻击的情形未覆盖；以及评估仅使用单一响应删除策略，未探讨更细粒度的处理方式。

---

## 220. ARAC: Benchmarking Auto-Research's Alignment and Completeness on End-to-End Researchs

**arXiv ID:** 2608.12788 | [PDF](https://arxiv.org/pdf/2608.12788v1)

**作者:** Jiale Cui `[一作]` (Zhejiang University), Zhe Liu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ARAC-Bench基准，用于评估自动科研系统的研究流程对齐与完整性。

**💡 创新点**

创新点在于将人类评审经验转化为可量化的学术认知技能（ACS），并设计三阶段（提案、实验、合成）的可追踪诊断协议。

**🔧 技术方法**

使用ACS量化指标、阶段化评估协议、严格的时间截断和标准模块库实现代码实现评分。

**📊 数据集**

基准数据集来源于2026年ICLR等顶会的200篇论文、过去两年NeurIPS/ICLR/ICML的7000篇论文以及公开的评审讨论。

**📈 对比分析**

与10个主流自动科研框架在统一模型Kimi-K2.6下对比，最高对齐得分仅为67.9，说明当前系统仍存在显著缺口。

**⚠️ 局限性**

局限性包括实验阶段仍是瓶颈，基准依赖于过去两年论文，且评估主要聚焦技术实现而非更深层的科研思维。

---

## 221. Beyond Outcome Rewards: Step-Level Self-Distilled Policy Optimization for Deep Search Agents

**arXiv ID:** 2608.12764 | [PDF](https://arxiv.org/pdf/2608.12764v1)

**作者:** Haoze Wu `[一作]` (Hong Kong University of Science and Technology), Xiaoguang Li `[通讯]` (Huawei Technologies Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向多步搜索代理的自监督学习方法SSPO，利用步骤级特权信息Evidence Anchors，将自教师的信号作为优势权重，仅对错误轨迹进行更新，从而在搜索过程中提供更细粒度的监督。

**💡 创新点**

创新点包括：①设计了与信息检索步骤匹配的Evidence Anchors，填补了单调推理任务中缺失的中间监督；②将教师与学生的差异转化为步骤级优势权重，而非直接匹配分布，避免了信息泄露并保持正确轨迹的多样性；③仅在错误轨迹上应用自监督，进一步提高学习效率。

**🔧 技术方法**

采用技术主要有：On‑Policy Self‑Distillation、Group Relative Policy Optimization (GRPO)、ReAct框架、LLM（Qwen3‑8B）与Web搜索工具、步骤级优势权重计算与应用。

**📊 数据集**

使用的数据集包括 BrowseComp、GAIA、FRAMES（及其子集）和 DeepForge；Evidence Anchors 由大型语言模型生成；训练样本约 4000 条正轨迹与 6000 条问答对。

**📈 对比分析**

在同等模型规模下与 GRPO 对比，SSPO 在 100 步时已超过 GRPO 200 步，平均性能提升约 +4.8 分，且只增加约 5% 的额外计算开销；在 BrowseComp、GAIA、FRAMES 三个基准上均表现更优。

**⚠️ 局限性**

局限性包括：依赖 Evidence Anchors 的质量与生成方法，若生成质量不佳可能影响学习；仅在 Web 搜索场景中验证，尚未在其他搜索或工具环境中评估；对极其复杂或高度动态的查询仍可能不足，且未充分探索更大规模模型的可扩展性。

---

## 222. Sustaining Plasticity via Learnable Wavelet Activations in Continual Learning

**arXiv ID:** 2608.12874 | [PDF](https://arxiv.org/pdf/2608.12874v1)

**作者:** Zeyang Zhang `[一作]` (Xi'an Jiaotong University), Weizhan Zhang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一种基于波形分解的可学习激活函数 ChannelWavAct，用于解决连续学习中的可塑性丧失问题。

**💡 创新点**

创新点包括：①将激活函数分解为全局低频基函数和局部可学习高频小波，显式抵消频谱偏差；②引入动态小波注入机制，根据损失停滞触发容量扩张；③采用斜率专属正则化与分离学习率策略，实现稳定-可塑性平衡；④提供严谨的理论证明，证明混合小波结构与动态注入的必要性。

**🔧 技术方法**

技术方法包括：可学习小波激活（Mexican Hat 小波 + SiLU 基函数）、动态小波注入与学习率衰减、斜率特定正则化、后激活批归一化、离散小波变换原理、NTK 及频谱分析理论。

**📊 数据集**

实验数据集涵盖：Permuted MNIST、Random Label MNIST、Mini-ImageNet、CIFAR-100、Tiny-ImageNet、ImageNet-100 等，使用回放式 SSD、EWC、WA 等连续学习框架。

**📈 对比分析**

与 ReLU、AID、Randomized Smooth-Leaky、Rational、KAN（B-Spline）以及 EWC/WA 等基线进行对比；ChannelWavAct 在平均准确率、最终准确率、遗忘度和有效秩上均优于所有对照方法，达成或接近现有最优水平。

**⚠️ 局限性**

局限性：相对于传统激活函数，计算开销和参数量略高；在极大规模任务或极长任务序列时，动态注入与正则化机制需要更细致的调优；未来工作需进一步提升小波计算效率并完善泛化理论。

---

## 223. Falsehood and Impossibility Are Different Directions in an AI's Representation of Language

**arXiv ID:** 2608.12852 | [PDF](https://arxiv.org/pdf/2608.12852v1)

**作者:** Yoon Pyo Lee `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Yoon Pyo Lee (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了Gemma 3 4B IT 对假话与不可能句子在内部表示和输出分类上的差异，并通过激活分析与线性探针探索其内部方向。

**💡 创新点**

发现模型的口头分类将“假话”与“矛盾”混为一谈，但内部激活空间中可将真值与不可能划分为正交方向，且不可能句子更接近语义异常，而非极端假话，提供了对AI内部逻辑处理的经验观察。

**🔧 技术方法**

使用线性探针（logistic regression）和稀疏自编码器（Sparse Autoencoder）对残差流进行分析，并计算AUC、均衡准确率与余弦相似度等指标。

**📊 数据集**

采用包含85个哲学案例（共17个种族，含变体）与75个多模态主题样本（15个主题 × 5 条件）的自定义数据集，覆盖真值、可变假、罕见、语义异常与不可能五类。

**📈 对比分析**

通过交叉验证与跨数据集转移比较，发现真值探针对真假区分AUC 0.93，可能性探针对真与不可能区分AUC 0.97，异常探针AUC 0.96；在不同层次下方向正交，证明内部表示存在显著区别。

**⚠️ 局限性**

限制包括样本量有限、仅测试单一小型模型、仅使用文本模板、探针仅揭示相关性非因果、类别定义人为且未覆盖所有语言不可能情形。

---

## 224. Practice Makes Unsafe: Skill Misevolution in Self-Improving LLM Agents

**arXiv ID:** 2608.12851 | [PDF](https://arxiv.org/pdf/2608.12851v1)

**作者:** Xutao Mao `[一作]` (City University of Hong Kong), Cong Wang `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了自适应LLM代理通过技能进化将不安全轨迹转化为持久策略的风险，并提出生命周期感知的评测工具与治理框架。

**💡 创新点**

提出SkillMisevo-Gym/Bench完整追踪技能进化生命周期，定义Skill Misevolution概念，并设计SafeEvolve治理组件实现自动修复、归因和退役。

**🔧 技术方法**

使用自适应LLM框架（Claude Code、Codex、Hermes、OpenClaw）、MiniMax‑M2.7模型、EvoSkill/SkillClaw/AutoSkill等进化方法，配合AgentHazard与Gemini‑3‑Flash等安全评估工具。

**📊 数据集**

采用SkillMisevo‑Bench构建的21任务集合（自动发现的恶意、友善和持久任务），共计525个任务、25个冻结实验。

**📈 对比分析**

在四个代理与六种演化方法的25实验中对比无演化、Raw、Utility‑only、SecureClaw、ClawKeeper等治理方案；SafeEvolve将不安全检索率从45.3%降至14.7%，持久ASR从27.6%降至4%，同时保持99%实用效用。

**⚠️ 局限性**

仅关注技能库的持久化，未覆盖内存、策略或多模态更新；实验基于有限任务集，未检验长期真实部署情境。

---

## 225. FSGR: Mitigating Token Frequency Bias for Fair SID-Based Generative Recommendation

**arXiv ID:** 2608.12845 | [PDF](https://arxiv.org/pdf/2608.12845v1)

**作者:** Yuchen Zheng `[一作]` (Nankai University), Xiaojie Yuan `[通讯]` (Nankai University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FSGR 框架，通过平衡语义码本与层级频率校准，消除 SID 生成和推荐中的 Token 频率偏差。

**💡 创新点**

创新点：① 在 SID 码本构建阶段采用 OT 赋值优化和双准则重锚（Dual‑Criteria Re‑anchor），实现更均衡的代码单元分布；② 在推荐训练阶段采用两阶段策略（先交叉熵预训练，再进行层级频率校准 HFC），针对不同层级的 SID 进行温度自适应校正，精准抑制高频 Token 的过度预测。

**🔧 技术方法**

技术手段包括：Optimal Transport 与 Sinkhorn‑Knopp、Dual‑Criteria Re‑anchor、两阶段训练（CE + HFC）、LoRA 微调、Gini 系数评估等。

**📊 数据集**

实验数据集：Amazon Review 的 Luxury Beauty、Industrial and Scientific、Software 三个子集。

**📈 对比分析**

与 RQ‑VAE、RT、QuasiSID、MiLe、WAKL 等基线以及 TIGER、Llama3.1‑8B、Qwen3‑8B 三种后端模型对比，结果显示 FSGR 在 Gini 系数上平均提升约 20%，召回率和 NDCG 基本保持不变，证明在保持精度的前提下显著提升公平性。

**⚠️ 局限性**

局限性：① 仅在 SID 基础的生成式推荐框架验证，缺乏在其他模型上的通用性评估；② OT 赋值与重锚步骤增加训练复杂度与计算成本；③ 对极低频 Token 的曝光提升有限，仍有进一步优化空间。

---

## 226. AQuA: Recursively Self-Improving Quantitative Trading Research Agents

**arXiv ID:** 2608.12841 | [PDF](https://arxiv.org/pdf/2608.12841v1)

**作者:** Jiacheng Guo `[一作]` (Princeton University), Mengdi Wang `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了AQuA，包含两套完全独立的自主研究系统：Part I负责符号因子发现，Part II负责模型开发。两套系统均通过递归自我改进循环，在保持数据路径与评估器封闭的前提下，利用先前实验验证的证据更新各自的研究状态，从而提升后续假设与候选设计的质量。

**💡 创新点**

创新点主要有：
1) 在同一研究流程中实现递归自我改进，但通过严格的沙箱化和分离指标，阻止了数据泄漏与适配偏差；
2) 采用多代理管道（管理者→数据管理员→视觉分析器→创意挖掘器→因子评估器→回测工程师→研究图书馆）来完成因子发现；
3) 将模型开发转化为配置差分的可比实验空间，保证每一次迭代仅在固定的数据与评估器上比较；
4) 将传统因子与深度时序模型（卷积+注意力+跨时序混合）结合，实现高信息系数与稳健Sharpe。

**🔧 技术方法**

技术栈包括：
- 大语言模型驱动的实验生成与推理；
- 领域特定语言（DSL）用于因子表达式和模型配置；
- 多代理管道与管理者调度；
- 复合卷积-注意力时序模型与跨时序混合；
- 封闭沙箱与分离评估指标来防止泄漏；
- 训练与评估使用分层时间分割、走前评估与两腿成本的Sharpe评估。

**📊 数据集**

使用的数据集：
- Part I：加密货币五分钟频率市场（crypto universe）。
- Part II：美国股票日内30分钟收益预测，训练集2010–2019，2020年预留为盲区，测试集为未见的2021–2025。

**📈 对比分析**

比较方法与性能：
- Part I：与基准因子相比，结合后信号信息系数提升至约0.190；
- Part II：对比线性、GBDT、LSTM、GRU等模型，最终混合模型的单股票IC达到+0.0843；Sharpe在两腿成本后达到+2.50，走前评估保持+2.0；每年Sharpe均为正。性能通过与同类模型在相同数据与评估器下的IC、R²和Sharpe直接对比。

**⚠️ 局限性**

局限性：
- 仅在单一市场（加密、美国股票）和单一频率（5分钟、30分钟）上验证，结果未必能迁移到其他资产或频率；
- 需要人工设定研究目标与沙箱，系统仍非完全无人监督；
- 所有指标均为模拟，未在实盘交易中验证；
- 封闭沙箱能保证因子/特征路径不泄漏，但对最终测试窗口的完全隔离仅靠运维纪律，技术上并非绝对不可逆；
- 两套系统不耦合时防止了泄漏，但耦合后需要额外封闭措施，当前实现仍为单独运行。

---

## 227. From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options

**arXiv ID:** 2608.12836 | [PDF](https://arxiv.org/pdf/2608.12836v1)

**作者:** Obed Junias `[一作]`, Maria Leonor Pacheco `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文演示了如何使用 ACL 样式文件与 LuaLaTeX 或 XeLaTeX 进行论文排版，提供了多语言文本的示例。

**💡 创新点**

创新点在于同时展示了 LuaLaTeX 与 XeLaTeX 两种 TeX 引擎下的使用方法，并演示了对不同语言（如印地语、阿拉伯语）的 Unicode 支持。

**🔧 技术方法**

使用的技术包括 ACL 样式文件、LuaLaTeX、XeLaTeX、Unicode 编码以及相关宏包。

**📊 数据集**

未使用任何数据集，本文仅为排版示例。

**📈 对比分析**

本文未进行方法或性能比较，仅提供了排版演示；因此无法给出性能评估。

**⚠️ 局限性**

局限性：仅为演示性文档，缺乏实际研究内容、实验结果和评估，无法验证排版在不同论文类型中的适用性。

---

## 228. Semantic Steering for Controllable Generation: Tuning-Free Concept Erasure in Multimodal Diffusion Transformers

**arXiv ID:** 2608.12829 | [PDF](https://arxiv.org/pdf/2608.12829v1)

**作者:** Qiao Li `[一作]` (Chinese Academy of Sciences), Jizhong Han `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练、无参数修改的概念消除方法，在多模态扩散Transformer（MM‑DiT）中通过提取中间块的文本分支语义表示差分向量，并在早期与中间块连续注入该向量，以实现对不想要概念的消除及可控生成；

**💡 创新点**

创新点在于：①发现MM‑DiT中间块承载了最显著的语义信息，可直接构造差分向量；②利用稀疏文本分支构造向量，避免图像分支冗余；③在整个去噪时间步中持续注入单一向量，保持生成连贯；④实现了无需训练、可对多样化概念（明星、裸体、艺术风格）进行精准消除，并支持多风格重定向；

**🔧 技术方法**

技术手段包括MM‑DiT架构分析、rectified flow采样、文本分支语义向量构造（差分法）、向量注入（多块连续）、steering strength调节、对抗鲁棒性评估；

**📊 数据集**

使用的模型与数据集：Stable Diffusion‑v3.5、FLUX.1；概念测试集100条prompt（明星、裸体、艺术风格）；COCO‑30K正样本；四种对抗攻击数据集（Ring‑A‑Bell、MMA‑Diffusion、I2P、P4D）；

**📈 对比分析**

与无调参基线（SLD、TraSCE、STG、NP）以及参数修改基线（UCE、ESD、CA）比较，指标包括GIPHY、LLaVA、Gram、LPIPS、NudeNet、Aesthetic、FID、CLIP；实验显示在概念消除效果上实现state‑of‑the‑art，保留图像质量、Aesthetic分数高，且在对抗攻击上表现最强；

**⚠️ 局限性**

局限性包括：需手工设定steering strength（l）需经验调节；多风格重定向需预先构造多组向量；在更大模型或不同架构上验证仍待进一步研究；对极细粒度语义差异的消除效果尚有限。

---

## 229. RealmEye: Virtual Machine Introspection for Arm CCA Realm VMs

**arXiv ID:** 2608.12822 | [PDF](https://arxiv.org/pdf/2608.12822v1)

**作者:** Ruofei Qu `[一作]` (Chinese Academy of Sciences), Yu Qin `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文设计实现了RealmEye，一个在Arm CCA Realm VM上运行的VMI系统，能够在不改动Realm VM的前提下完成进程列表遍历和系统调用表完整性检查。

**💡 创新点**

创新点在于将VMI逻辑迁移至Realm Management Monitor（RMM）实现硬件隔离，利用rec_run通道进行触发与加密结果传输，支持周期触发抵御Hypervisor与rootkit协同，并实现自主符号解析以消除对外部符号表的依赖。

**🔧 技术方法**

使用的技术包括：RMI REC_ENTER触发机制、RMM读取寄存器/内存、两级页表翻译、基于VBAR_EL1的KASLR解析、LibVMI后端适配、AES-128加密、RA‑TLS型会话密钥协商。

**📊 数据集**

实验使用Arm FVP仿真平台，搭载Linux 6.15.0-rc1的Realm VM，并使用Diamorphine根套件模拟进程隐藏与系统调用表钩子，采用微基准和宏基准进行性能评估。

**📈 对比分析**

方法上与00SEVen进行功能与安全目标对比；微基准显示三大原语（加密、地址翻译、物理读取）周期线性可预测；宏基准验证过程遍历与sys_call_table检查的延时与原语计数吻合；整体VMI命令平均耗时约502µs，主要开销来自SMC切换与KVM路径。

**⚠️ 局限性**

局限性包括：仅在FVP仿真环境验证，未在真实CMA硬件上；SMC与KVM通道带来约500µs固定开销；未针对硬件侧信道或物理层攻击做进一步评估；对实时攻击的实时检测能力尚未验证。

---

## 230. Beyond Correctness: Benchmarking and Aligning Response Behaviors in Hybrid-Thinking MLLMs

**arXiv ID:** 2608.12781 | [PDF](https://arxiv.org/pdf/2608.12781v1)

**作者:** Xinming Wang `[一作]` (Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

暂无信息

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

## 231. SCOPE: Subspace Clustering with Online Per-Head Top-K Estimation for Sparse Video Attention

**arXiv ID:** 2608.12780 | [PDF](https://arxiv.org/pdf/2608.12780v1)

**作者:** Qi Zhao `[一作]` (Zhejiang University), Xi Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SCOPE，一种训练无关的稀疏注意力框架，专为视频扩散 Transformer（DiT）推理设计，以显著加速推理并保持与全注意力相近的生成质量。

**💡 创新点**

创新点包括：① 以 3D RoPE 的时间、高度、宽度子空间为基础进行键（key）子空间聚类，并将三子空间的中心点编码组合成细粒度的键代理得分；② 在线按每个注意力头动态估计 Top‑k 保留键数，结合 Top‑p 与固定 Top‑k 的混合策略，消除单一阈值导致的保留不足问题；③ 仅使用原始查询、键、值进行最终稀疏注意力计算，完全不需要模型再训练或权重修改。

**🔧 技术方法**

使用的技术包括：K‑means 聚类（对查询进行全维聚类，对键的三子空间分别聚类）；基于 3D RoPE 的子空间分解和产品量化；Hybrid Top‑p / fixed‑Top‑k 选择；在线 per‑head Top‑k 估计；以及常规的稀疏注意力计算与评估。

**📊 数据集**

在六个 720p 视频生成任务上验证：文本到视频（T2V）使用 Penguin Benchmark 生成提示；图像到视频（I2V）使用 VBench++ 数据集；模型覆盖 Wan2.1‑14B、Wan2.2‑A14B 与 HunyuanVideo‑13B 三种 DiT。

**📈 对比分析**

与 SpargeAttn、SVG2、SVOO 等三种训练无关稀疏注意力方法进行对比，采用 PSNR、SSIM、LPIPS（对齐密集注意力的质量）以及 VBench 评测指标。SCOPE 在所有实验设置中均取得最高质量评分，并实现 1.67×–1.99× 的端到端推理加速，且在 720p 视频生成中延迟最低。

**⚠️ 局限性**

局限性：仅针对 720p 分辨率验证，尚未在更高分辨率或不同视频长度下评估；子空间聚类和在线 Top‑k 估计会引入额外的聚类与排序开销，虽在实验中被压缩，但在极大模型或 GPU 资源受限时可能成为瓶颈；缺乏对多模态输入（如音频、多文本描述）兼容性的实验。

---

## 232. AI and Consumer Rights in India Working Paper

**arXiv ID:** 2608.12863 | [PDF](https://arxiv.org/pdf/2608.12863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 233. ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts

**arXiv ID:** 2608.12776 | [PDF](https://arxiv.org/pdf/2608.12776v1)

**作者:** Chanh Vo `[一作]` (University of Information Technology), Ngan Luu-Thuy Nguyen `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了 ViTOED 目标情感检测数据集，并基于结构化情感图实现基线模型。

**💡 创新点**

创新点在于首次针对越南语社交媒体文本提供手工标注的目标情感四元组数据，并分析语言特有现象。

**🔧 技术方法**

使用结构化情感图（HeadFNN/DependentFNN）+ 多种越南语预训练模型（mBERT、XLM-R、PhoBERT、ViSoBERT、CafeBERT、mT5、ViT5 等）进行实体识别、关系抽取和情感归属。

**📊 数据集**

使用 ViTOED 数据集，包含 10,985 条评论、21,244 个四元组，划分为 70%/10%/20% 的 train/dev/test。

**📈 对比分析**

在 Span F1、Targeted F1、Parsing Graph F1、Sentiment Graph F1 四个指标上进行评估，monolingual 预训练模型 (PhoBERT, ViSoBERT) 在实体抽取上优于多语种模型，head‑first 在实体抽取上效果更好，head‑final 在表达抽取上更佳，整体性能仍低于理想水平。

**⚠️ 局限性**

存在的限制包括实体 span 对齐困难、根节点与边的错误预测导致表达/关系误检、源/目标实体混淆，导致整体性能受限。

---

## 234. CW-BASS v2: Saturation-Aware Pseudo-Label Selection for Semi-Supervised Segmentation under Foundation-Model Teachers

**arXiv ID:** 2608.12773 | [PDF](https://arxiv.org/pdf/2608.12773v1)

**作者:** Ebenezer Tarubinga `[一作]` `[通讯]` (Ebenworks Systems), Ebenezer Tarubinga (Ebenworks Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的半监督语义分割方法CW-BASS v2，旨在选择伪标签时考虑教师模型的置信度。

**💡 创新点**

创新点在于引入了一种饱和度感知的伪标签选择方法，能够根据教师模型的置信度动态调整选择规则，而不是固定使用单一规则。

**🔧 技术方法**

使用了DINOv2作为教师模型，结合了自适应置信度下限和持出校准等技术。

**📊 数据集**

在Pascal VOC、Cityscapes和ADE20K等数据集上进行了实验，特别是在Pascal VOC 1/8和ADE20K上进行了详细评估。

**📈 对比分析**

与现有方法进行比较，CW-BASS v2在饱和基准上恢复了UniMatch V2的操作点，并在教师置信集不可靠的情况下表现更好，尤其是在ADE20K上提高了1.5 mIoU。

**⚠️ 局限性**

限制在于该方法的有效性主要在于DINOv2教师模型，尚未验证其在其他基础模型上的普适性，且结果为单一种子，可能存在噪声影响。

---

## 235. AMR-Pose: An Active LED Marker-Based Relative Pose Estimation Framework With Probabilistic Switching PnP for Cooperative AUVs

**arXiv ID:** 2608.12866 | [PDF](https://arxiv.org/pdf/2608.12866v1)

**作者:** Zeyu Sha `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于红蓝LED标记的主动视觉相对位姿估计框架AMR-Pose，配合概率切换PnP（PSwPnP）实现海底多AUV的6自由度相对位姿跟踪。

**💡 创新点**

1）构建紧凑、耐压的红蓝LED标记模块，显著提升在浑浊、光照变化及遮挡环境下的观测可靠性；2）设计概率切换PnP算法，融合Lie‑group EKF运动预测、概率标记关联与存在感知，实现在全观测、部分观测与无观测时自适应切换姿态级与像素级更新；3）引入可存在概率的可视性管理，抑制瞬时遮挡导致的身份切换与估计抖动。

**🔧 技术方法**

LED视觉检测、HSV色彩分割、EPnP+LM优化、Lie‑group SE(3) EKF、概率关联（Hungarian匹配）、存在概率（logit平滑）、帧间PnP切换、闭环PID跟踪控制。

**📊 数据集**

在实验水箱中使用两台OpenAUV进行相对位姿测量，配备运动捕捉系统（3.2×1.6×0.5 m）获取高精度真值；实验覆盖9个相对位置、50 s旋转轨迹，共45次重复。

**📈 对比分析**

与四种消融模型（无EKF、无关联、无存在感知、无滤波）及两种基线（EPnP‑LM、GMLPnP）比较。AMR‑Pose在翻译RMSE仅0.032 m、旋转RMSE 0.032 rad，分别比基线低约94 %/96 %，且在4–3–4可视性切换时保持平稳、误差低于0.04 m/0.05 rad。

**⚠️ 局限性**

仅在实验室水箱、低浑浊、短距离场景验证；未对远距离、高浑浊或复杂海底环境进行测试；系统目前只支持两AUV，扩展到大规模多AUV时需要解决多目标标记辨识与通信。

---

## 236. Heterogeneous Vision-Language Ensemble with Disagreement-Aware Reranking for Text-Based Person Anomaly Retrieval

**arXiv ID:** 2608.12843 | [PDF](https://arxiv.org/pdf/2608.12843v1)

**作者:** Huu-An Vu `[一作]` (Hanoi University of Science and Technology), Huy Minh Nhat Nguyen `[通讯]` (Vietnamese-German University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出一种基于异构视觉‑语言模型融合与不确定性感知重排序的文本检索式人群异常行为检索框架。

**💡 创新点**

创新点包括①在迭代融合过程中对不同模型产生的gallery顺序进行得分对齐；②使用模型间投票不一致度自适应路由，只对难检索查询调用大规模跨编码器VLM进行重排序；③结合Reciprocal Rank Fusion实现多模型结果的高效融合。

**🔧 技术方法**

核心技术包括：Hybrid、Unified、Iterative框架（Local‑Global Hybrid Perspective + Unified Image‑Text），多模型异构嵌入融合（Voyage Multimodal、BGE‑VL‑v1.5‑mmeb、Qwen3‑VL‑Embedding‑8B），得分对齐，投票不一致度路由，跨编码器VLM重排序（gemini‑3.1‑flash‑lite）以及RRF。

**📊 数据集**

使用AI City Challenge 2026 Track 4的Pedestrian Anomaly Behavior (PAB) 数据集，包含约1百万合成图文对，1,978个查询与36,773张图片（34,795张干扰图）。

**📈 对比分析**

在官方评测指标下，系统实现90.92% mAP，85.13% Recall@1，97.72% Recall@5，98.68% Recall@10，明显优于现有最优方法（如SSDC 92.74% mAP、HUI 89.23% Recall@1等），在全量测试集上获得最高的综合表现。

**⚠️ 局限性**

局限性在于：①多模型融合与VLM重排序显著提升了推理成本；②融合权重与路由阈值为经验设定，缺乏自适应学习机制。

---

## 237. CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Adaptive Weight Allocation

**arXiv ID:** 2608.12842 | [PDF](https://arxiv.org/pdf/2608.12842v1)

**作者:** Yuchen Liu `[一作]` (Beihang University), Xiang Gao `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CABS+ 模型合并框架，在 CABS 基础上引入自适应权重分配，使用梯度无关搜索降低时间复杂度与 GPU 内存占用。

**💡 创新点**

创新点在于：①通过 CMA‑ES 的无梯度优化实现 Adaptive Weight Allocation；②引入非对称适配函数消除高损失任务主导；③提出 Relative Synergy Score 指标用于评估合并可行性。

**🔧 技术方法**

采用结构化稀疏化（n:m 与块稀疏化）、CMA‑ES、边界约束、非对称适配函数以及任务向量分离技术。

**📊 数据集**

使用 27 个任务数据集（LLM Leaderboard、Open LLM Leaderboard 2、GLUE、RACE、SQuAD 等），在 5 种模型上验证：Mistral‑7B、Qwen‑2.5‑7B、GPT‑2、RoBERTa、70B LLM。

**📈 对比分析**

与 Task Arithmetic、TIES‑Merging、AdaMerging、WUDI‑Merging 等基线对比，CABS+ 在大模型上平均提升 16.97%（相较 AdaMerging）与 12.93%（相较 WUDI‑Merging）；GPU 内存使用 <25% AdaMerging；合并速度比 WUDI‑Merging 快约 4 倍。

**⚠️ 局限性**

局限性：仍受任务异质性、数据分布差异影响；合并效果随任务数增多而下降；对极大规模多任务合并仍需进一步改进搜索与稀疏化策略。

---

## 238. Validation of Smartphone-Based Photogrammetric 3D Body Scanning for Automated Anthropometric Measurements Compared with a Commercial Depth-Sensor-Based Body Scanner

**arXiv ID:** 2608.12827 | [PDF](https://arxiv.org/pdf/2608.12827v1)

**作者:** Ruting Cheng `[一作]` (George Washington University), James K. Hahn `[通讯]` (George Washington University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在孕妇身上与深度传感器扫描仪（Fit3D ProScanner）并在刚性木偶上进行重复测量，系统评估了基于手机的摄影测量应用PolyCam在全身3D扫描中的精度与一致性，并提出了自动化周长测量与ICP配准的处理流程。

**💡 创新点**

首次针对全身扫描在实际临床人群（孕妇）中进行系统性验证，提出了无需专用硬件即可实现高精度全身扫描的可行性，并通过自动几何极值定位实现快速周长提取，减少人工干预。

**🔧 技术方法**

使用的技术包括：PolyCam摄影测量、Fit3D深度传感器扫描、MeshLab对网格去噪与平滑、PCA坐标标准化、ICP细化配准、线性混合效应模型、Bland‑Altman、ICC、RMSE、MAE、Pearson相关、Mann‑Whitney U、热图可视化等。

**📊 数据集**

数据集：141名孕妇在两次产检阶段（18–24周和31–38周）共228组配对扫描（PolyCam+Fit3D），以及15次刚性木偶的重复扫描与测尺测量。

**📈 对比分析**

比较方法：采用线性混合效应模型评估扫描仪差异、ICC与Pearson检验一致性、RMSE/MAE量化误差、Bland‑Altman可视化偏差、热图展示顶点级误差。结果显示PolyCam相对Fit3D的周长偏差约-10至-15 mm，ICC≥0.90（除手腕外）且RMSE≤19 mm；木偶实验中平均偏差<3.5 mm，CV<0.03，Mann‑Whitney U无显著差异。

**⚠️ 局限性**

局限性：未在人体实验中加入直接测尺基准；光学测量可能存在缩放误差；依赖光纹理的摄影测量在衣物暗色或纹理缺失时易出错；相机固定位置导致视角偏差；深度摄像头在部分区域仍可能出现遮挡导致局部误差；仅评估了PolyCam与Fit3D，缺乏对其他摄影测量工具或多手机型号的比较。

---

## 239. A Comprehensive Empirical Evaluation of Vector Database Systems for Approximate Nearest Neighbor Search: Performance, Quality, and Resource Trade-offs

**arXiv ID:** 2608.12812 | [PDF](https://arxiv.org/pdf/2608.12812v1)

**作者:** Ashen Rashmiks `[一作]` (University of Kelaniya), Tiroshan Madushanka `[通讯]` (University of Kelaniya)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七种主流向量数据库在六个不同数据集上进行系统化的实证评估，衡量检索质量、查询延迟、吞吐量和资源占用等指标。

**💡 创新点**

提供了可复现、容器化的基准框架，联合评估完整数据库系统而非仅算法库，并给出了针对不同工作负载的系统选择指南。

**🔧 技术方法**

采用了多种近似最近邻索引算法（HNSW、IVF、PQ、DiskANN等）以及容器化适配器，测量Recall@K、Precision@K、NDCG、QPS、延迟百分位、构建时间、内存/磁盘占用等指标。

**📊 数据集**

SIFT1M、DEEP1M、GIST1M、GloVe、MS MARCO、Random等六个数据集，包含100K~1M向量，维度96~960。

**📈 对比分析**

统一实验流程在同一硬件（6 vCPU、32GB RAM、SSD）上运行，每组实验三次取均值，比较检索质量、吞吐量、延迟、构建时间和资源消耗。结果显示FAISS在吞吐量和低延迟方面领先；Weaviate在召回率最高；Qdrant在整体数据库表现最佳；Milvus在高维数据上保持较好召回；LanceDB以最快构建速度但召回率最低；Chroma具有最慢构建时间。

**⚠️ 局限性**

仅使用默认配置、单节点实验、未覆盖分布式扩展、仅检索无属性过滤、硬件限定且系统快速演进，可能低估最佳性能。

---

## 240. CoMedBench: A Multi-Source Benchmark of Synthetic Medical Data Fidelity and Downstream Utility

**arXiv ID:** 2608.12805 | [PDF](https://arxiv.org/pdf/2608.12805v1)

**作者:** Akanta Das `[一作]` (Bangladesh University of Engineering and Technology), Tanmoy Sarkar Pias `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个多源、全细粒度的医学合成数据基准（CoMedBench），对四种不同生成器（CoMed-CTGAN、CoMed-TVAE、CoMed-CG、CoMed-GC）的统计真实性与下游预测实用性进行了系统评估。

**💡 创新点**

创新点：①将统计真实性指标（分布与相关性稳定性）与临床实用性指标（TSTR/TTRR等）统一纳入单一基准；②引入“临床有效性层”对合成数据做结构与临床约束，并在消除该层后进行消融实验；③提供完整非聚合结果，方便后续研究复现与对比。

**🔧 技术方法**

技术：基于SDV框架的四类合成模型（CTGAN、TVAE、CopulaGAN、GaussianCopula）；统计真实性评估使用SDV内置指标；下游实用性评估采用多种分类器（LR、RF、GB、XGBoost、MLP）在表格与时间序列数据上测算AUROC/AUPRC，并计算TRTR、TSTR和Retention；消融实验对临床有效性层进行开启/关闭。

**📊 数据集**

数据集：表格数据——Breast Cancer（Coimbra）、CDC BRFSS Diabetes、GBSG、METABRIC、SUPPORT2、UCI Cervical Cancer、UCI Chronic Kidney Disease、UCI Dermatology、UCI Diabetes 130-US Hospitals、UCI Early-Stage Diabetes、UCI Heart Failure、UCI Hepatitis、UCI Indian Liver Patient、UCI Mammographic Mass、UCI Statlog Heart、NHANES Mortality；时间序列数据——MIMIC‑III、MIMIC‑IV、eICU 的多种预测任务（死亡、出院、ICU再入院、LOS等）。

**📈 对比分析**

比较方法：对每个数据集/任务/生成器/分类器，报告 Distribution Stability、Correlation Stability、Overall Quality、Data Validity、Data Structure；对下游任务，分别给出 TRTR（实测），TSTR（合成训练、真实测试）以及 Retention = TSTR/TRTR。实验显示，CoMed-CTGAN 与 CoMed-TVAE 通常获得最高整体质量与保留率，表格任务中多数生成器的 TSTR/Retention 接近 90%+；在时间序列任务中，性能波动更大，部分任务（如 ICU Mortality）仍有显著下降。消融实验表明，临床有效性层能显著提升合成数据的下游效用，尤其是在满足临床约束的任务上。

**⚠️ 局限性**

局限性：①评估仅限于二分类和少数多分类任务，未覆盖回归或更复杂的多任务学习；②只评估了四类合成器，缺乏对新型生成模型（如GAN、VAE 的不同变体）的探索；③临床有效性层的规则手工制定，可能无法完全捕捉所有医学约束；④大多数指标基于 AUROC/AUPRC，未考虑预测的临床实用性（如决策曲线）；⑤实验环境为单 GPU 计算，未对生成速度与资源消耗做系统化评估。

---

## 241. Considering Contribution Statements in Visualization and HCI Research

**arXiv ID:** 2608.12792 | [PDF](https://arxiv.org/pdf/2608.12792v1)

**作者:** Mara Solen `[一作]` (University of British Columbia), Andrew M McNutt `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了可视化与人机交互社区对作者贡献声明的认知与实践，结合反思、在线调查和原型工具，探讨其对透明度、偏见与作者身份的影响。

**💡 创新点**

首次系统性探讨贡献声明的价值与局限，并提出在可视化/HCI 期刊中鼓励而非强制使用的框架，强调灵活的自定义与多样化的展示形式。

**🔧 技术方法**

采用了 CREDI‑T 及其变体作为贡献分类标准，结合自由文本描述，并开发了两款交互式原型（Contrib Builder 与 Matrix Builder）来支持贡献记录与可视化。

**📊 数据集**

未使用传统科学数据集；调查样本为 25 位可视化/HCI 研究者（涵盖博士生、博士后、教授等不同学术阶段），并未进行外部数据验证。

**📈 对比分析**

无传统的定量性能对比；通过质性分析展示不同受访者的观点与原型工具的可行性，未给出数值指标或基准测试。

**⚠️ 局限性**

样本规模有限、非随机，可能倾向于支持贡献声明；缺乏对声明有效性与影响的定量验证，工具与方法尚处于原型阶段，需进一步迭代和实证研究。

---

## 242. Comment on "Modeling rapid language learning by distilling Bayesian priors into artificial neural networks"

**arXiv ID:** 2608.12974 | [PDF](https://arxiv.org/pdf/2608.12974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. Insights from Multi-tasking the EAX Algorithm for the Travelling Salesperson Problem

**arXiv ID:** 2608.12772 | [PDF](https://arxiv.org/pdf/2608.12772v1)

**作者:** Liam Wigney `[一作]` (Adelaide University), Frank Neumann `[通讯]` (Adelaide University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究多任务Edge Assembly Crossover（MT‑EAX）在旅行商问题（TSP）上的应用，并与标准EAX在相同计算预算下进行比较。

**💡 创新点**

创新点在于将跨实例基因共享引入EAX，提出三种缩放策略（代数缩放、人口缩放、平衡缩放），并开发分离式（Decoupled）MT‑EAX以在早期获得多任务优势后转向单任务精细搜索。

**🔧 技术方法**

使用技术包括进化多任务框架、EAX交叉、边频率矩阵、分层搜索（Stage I/II）、基于相似度的父代选择、k‑opt局部修复、分离式运行模式。

**📊 数据集**

数据集为 1000 节点的欧几里得 TSP 实例，分别采用均匀分布、正态分布以及混合分布，所有实例均以 TSPLIB95 格式保存。

**📈 对比分析**

比较方法：在相同的计算预算（固定代数、人口或两者结合）下，评估质量差距（ΔQ）和计算节省（Compute Saved）。结果显示：生成缩放下，早期（≤50 代）可节省 60%–90% 计算且质量提升；人口缩放表现不佳；分离式 MT‑EAX 在保持早期优势的同时，最终收敛质量可与甚至优于标准 EAX，并大幅提升计算效率。

**⚠️ 局限性**

限制：多任务优势高度依赖实例几何相似度；人口饥饿导致最终质量下降；跨实例父代不兼容时负面效应显著；在无限预算或长时间运行时，标准 EAX 仍能追赶或超过 MT‑EAX。

---

## 244. PatchGen: Learning Soft Intra-Image Predictive Subsets for Visual Generalization

**arXiv ID:** 2608.12766 | [PDF](https://arxiv.org/pdf/2608.12766v1)

**作者:** Zhaorui Tan `[一作]` (Agency for Science, Technology and Research (A*STAR)), Xi Yang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

论文提出一种名为 PatchGen 的方法，利用软掩码学习每张图像中可预测的子区域，从而提升在数据偏移、目标偏移及两者组合下的视觉分类泛化能力。

**💡 创新点**

创新点在于：1) 从理论上证明存在“oracle intra‑image predictive subset”，并且使用该子集可保持 Bayes risk 与完整图像相同；2) 在此基础上设计软掩码学习框架，通过跨 patch 交互、低分抑制、置信度正则和类条件特征对齐等技术，使模型在无需文本监督的情况下自动识别并关注最具预测价值的图像片段；3) 将该框架统一应用于多种偏移任务（mDG、CCD、mDG+GCD）并取得显著提升。

**🔧 技术方法**

核心技术包括：soft predictive‑subset mask 学习、跨 patch 注意力交互、低分掩码抑制 (mask suppression)、置信度正则化、类条件特征对齐 (class‑conditional similarity loss)，以及轻量级的特征聚合模块。

**📊 数据集**

实验使用的公开数据集包括：自然图像的 PACS、VLCS、OfficeHome、TerraIncognita、DomainNet；组织病理图像的 HISTOPANTUM 与 HISTOCOLON；目标偏移任务使用的 CIFAR‑100 与 CUB；同时在这些数据集上采用 DomainBed 评估协议。

**📈 对比分析**

方法与多类基线对比：VLM‑based（SWAD、KAdaptation、GESTUR、DPR、CLIPCEIL++）、VM‑based（ERM、SagNet、SelfReg、CORAL、mDSDI、GMDG、L‑Reg）、因果启发式（CI‑DGA、SMIDG、BFMix、CauRDG）。PatchGen 在大多数设置下平均提升了 1–3% 的准确率，甚至在某些自然图像任务上与 VLM‑based 竞争；在 histopathology 任务中，PatchGen 也取得最佳或同等性能；在 CCD 与 mDG+GCD 任务中，对未知类别的识别率显著提高。

**⚠️ 局限性**

局限性：1) oracle intra‑image predictive subset 及其掩码不可直接观测，学习过程并不保证能完全恢复它；2) 对于已经经过图像‑文本预训练的 Vision‑Language 模型，图像特征本身已高度聚焦语义区域，PatchGen 的增益相对有限；3) 在极端域间差异或标签空间变化极大时，soft mask 可能需要更强的正则或更丰富的监督。

---

## 245. DTAMLP: Denoise Time-aware MLP for Session-based Recommendation

**arXiv ID:** 2608.12975 | [PDF](https://arxiv.org/pdf/2608.12975v1)

**作者:** Jiamu Zheng `[一作]` (University of Electronic Sciences and Technology of China), Xiaojun Shan `[通讯]` (University of Electronic Sciences and Technology of China)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对会话推荐系统提出两条实证发现，并在此基础上构建了集成稀疏噪声抑制与频域滤波的全 MLP 模型 DTAMLP。

**💡 创新点**

创新点包括：①一个轻量级、可插拔的权重融合模块，用以剔除点击时长中的“偶发噪声”；②提出并验证了“偏好噪声”假设，即在频域分解后可更好分离用户多重心理偏好，从而提升推荐质量。

**🔧 技术方法**

采用技术包括：时序感知注意力、图神经网络、FFT-Transformer（离散傅里叶变换）频域滤波、权重融合、表示一致性嵌入和鲁棒距离度量。

**📊 数据集**

主要实验数据集为 Diginetica 与 RetailRocket（用于定量评估），Yoochoose 用于可视化展示。

**📈 对比分析**

与 GNN、RNN、CNN、Attention 及 All-MLP 等多种主流基线在 MRR、NDCG、Recall 等指标下进行横向对比，DTAMLP 在所有指标上均实现了显著提升，验证了两大机制的互补性。

**⚠️ 局限性**

局限性包括：尚未与 2023 年以后的最新模型进行对比；频域滤波的有效性仅基于经验假设，缺乏严谨理论证明；实验仅覆盖两大公开数据集，模型在更大规模或不同领域的泛化性待进一步验证。

---

## 246. VoxAudio: Vocalized Audio Synthesis via Multi-Reward Autoregressive Flow Matching

**arXiv ID:** 2608.12951 | [PDF](https://arxiv.org/pdf/2608.12951v1)

**作者:** Wenxiang Guo `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可流式生成包含可辨识语音的音频（vocalized audio）的系统

**💡 创新点**

创新点包括：chunk-agnostic 的因果自回归流匹配架构、随机 chunk 边界预训练、滑动窗口实时推理、结合多重奖励（语义、语言、审美、时序）进行的 Negative-aware Fine‑Tuning（NFT）以及专门构建的 VoxCorpus 与 VoxBench 数据集

**🔧 技术方法**

技术：流式自回归流匹配、Causal Diffusion Transformer、Universe Audio VAE、文本编码与多层注入、因果卷积与块级因果注意力、滑动窗口 KV 缓存推理、基于强化学习的多奖励 NFT 对齐

**📊 数据集**

数据集：VoxCorpus（模拟与真实叙事音频，包含逐句时间戳的逐字转录）、VoxBench（用于评估与 RL 的 interval‑annotated benchmark）、以及利用 AudioCaps 等公开数据进行预训练和评测

**📈 对比分析**

与 Dasheng‑AudioGen、AudioGen、AudioLDM 等基准比较，VoxAudio 在 WER、Temporal Grounding (TG‑IoU)、MOS‑C 等指标上显著优于基线，整体音质相当或略优，并在不同音频时长（10–30 s）下保持竞争力

**⚠️ 局限性**

局限性：目前仅支持英语、30 s 以上长音频的时间一致性与语义连贯性下降、对说话人音色和情绪细粒度控制不足，未来需要扩展多语种、层次化架构以及属性级别的奖励

---

## 247. Prompts in the Wild: A Large Analyzed Collection of Transactional Prompts in Code

**arXiv ID:** 2608.12905 | [PDF](https://arxiv.org/pdf/2608.12905v1)

**作者:** Victoria Basmov `[一作]` (Bar-Ilan University), Reut Tsarfaty `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过收集57.5K个来自GitHub的事务性提示，提出一套结构化本体，将提示语视为科学对象，并提供可视化界面供研究者探索；

**💡 创新点**

创新点在于将提示语转化为可结构化的语言学对象，构建跨语言、跨任务的本体，首次实现对大量真实世界提示的系统化注解与分析；

**🔧 技术方法**

技术包括：利用GitHub API和LangChain/Completion API对Python代码进行静态分析提取提示文本，使用LLM自动化标注本体字段，结合人工审核与错误分析评估标注质量；

**📊 数据集**

使用的数据集为57,640个独特的事务性提示，涵盖62种语言、77个领域、上百种任务，所有提示均来源于公开GitHub仓库；

**📈 对比分析**

通过对100条随机样本进行人工评估，发现多数字段准确率>90%（如提示语言、领域、模态等），但输出类型、方向文本等字段准确率仅约60-70%；

**⚠️ 局限性**

限制包括：搜索仅限Python文件和特定API调用，导致数据偏向使用这些工具的项目；LLM标注可能引入错误；数据为单一快照，未覆盖闭源或非GitHub来源的提示，缺乏动态更新。

---

## 248. HounsWorld: A Multimodal World Model for Hidden Patient-State Readout, Reconstruction, and Simulation

**arXiv ID:** 2608.12904 | [PDF](https://arxiv.org/pdf/2608.12904v1)

**作者:** Yunhao Bai `[一作]` (East China Normal University), Yan Wang `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为HounsWorld的三维CT中心多模态世界模型，统一了临床问答、报告生成、CT去噪、虚拟对比增强和文本/掩码条件CT生成等任务；

**💡 创新点**

创新点在于将所有CT相关任务视为对患者潜在状态的状态依赖预测，通过共享的因果变压器学习患者的统一潜在表示，并提出Clinically Structured Patient-State Completion (CSPC) 以及条件显式观察和HU感知窗口等方法；

**🔧 技术方法**

使用技术包括：大规模因果变压器（3B参数）、联合理解-生成学习、伪帧构造将3D CT映射到RGB接口、残差适配器对CT通道进行微调、两阶段优化以及VAE流匹配实现CT重建和生成；

**📊 数据集**

数据集包括公开的CT-RATE、M3D、TotalSegmentator生成的解剖掩码以及内部收集的数据，构成了HounsBench基准，分为读取、重建和仿真三大任务族；

**📈 对比分析**

与多种基线（如Qwen3-VL-4B、MiniCPM、RadFM、M3D-LaMed、CT-CHAT、OmniCT）以及专门生成模型（InstructPix2Pix、CogVideoX、Lance、RED-CNN、MaskDenoising、CyTran、MedDiff、SMILE、MAISI、GenerateCT）对比，HounsWorld在读取任务中仅落后0.93个百分点，且在去噪、对比增强和文本/掩码到CT生成等重建与仿真任务中均表现出色，超过或与最强专门模型相当；

**⚠️ 局限性**

局限性包括：仅能预测短期、条件限定的未来观察，未覆盖长期疾病演化；生成的CT尚未进行临床验证；与专门任务模型相比仍存在微小性能差距；

---

## 249. Adaptive $k$ Nearest Neighbors Classifier via Granular Ball Computing

**arXiv ID:** 2608.12903 | [PDF](https://arxiv.org/pdf/2608.12903v1)

**作者:** Xiaoyu Lian `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自适应粒度球计算的k近邻分类器GBKNN，通过粒度球分层表示和自适应邻域构建来提升KNN的准确性与效率。

**💡 创新点**

创新点包括：①利用√(n)粗分层和Fisher准则自适应生成粒度球；②采用加权距离选取最近粒度球并以球内最远点确定邻域半径，从而动态决定k值；这些设计显著增强了噪声鲁棒性和计算效率。

**🔧 技术方法**

使用技术包括粒度球计算（GBG++）、Fisher判别分析、加权边界距离、基于球的KNN决策以及邻域自适应构建。

**📊 数据集**

实验基于17个公开数据集，包括UCI 12个、Santander、creditcard、covertype、Higgs、supersymmetry等，并在不同噪声水平下进行测试。

**📈 对比分析**

与传统KNN、NaNEKNN、AdaKNN、SMKNN、LMKNN、OneStepKNN、PLKNN、MGNR、GBKNN_2019等13个基线在多噪声水平和规模上比较，GBKNN在准确率（平均0.8736）和运行时间上均优于或接近最佳，表现出更高的鲁棒性和效率。

**⚠️ 局限性**

局限性包括：预处理阶段仍需生成粒度球，导致对超大规模数据存在一定计算与内存开销；在极端高维或严重类别不平衡的数据集上的性能尚未完全验证。

---

## 250. BavGround: A Benchmark for Regional Cultural Grounding and Dialect Competence in Bavarian

**arXiv ID:** 2608.12894 | [PDF](https://arxiv.org/pdf/2608.12894v1)

**作者:** Jophin John `[一作]` (Leibniz Supercomputing Centre), Barbara Plank `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并评测了面向巴伐利亚方言与地区文化的多语言（英、德、巴伐利亚）多选题基准BavGround，并对15个开源7B–10B规模模型和一个闭源模型在该基准上的表现进行系统实验。

**💡 创新点**

① 将区域文化和方言维度纳入评测；② 设计包含一般知识与源落地的两类题目；③ 引入多种评测协议（字母评分、打乱字母、选项文本概率、语义匹配、隐藏层诊断）以揭示模型行为差异；④ 对连续预训练检查点进行纵向轨迹分析，探究方言知识的学习动态。

**🔧 技术方法**

使用指令微调的开源LLM、Bavarian‑centric GENBA‑10B连续预训练、以及闭源GPT‑5.4‑mini；评测技术包括多选字母匹配、条件对数概率、文本长度归一化的选项概率、句向量语义相似度匹配、隐藏状态对齐诊断。

**📊 数据集**

BavGround数据集：206道题目（每道题目两类，共618条）在英、德、巴伐利亚三语中；题目来源于地区新闻、民族志专著与历史资料；题目分为8个文化主题（历史、政治、生活传统、烹饪、建筑遗产、风景、艺术与身份、语言）以及两类（GEN、GRD）。

**📈 对比分析**

在标准字母评分下，15个开源模型平均精度53%，最佳为69%（EuroLLM）；闭源模型精度89%。Bavarian语料与GRD类题目普遍难度最高；不同评测协议可提升至约59%（content‑based）或更高（semantic matching）。连续预训练检查点显示方言知识提升不均衡，Bavarian仍落后于英、德，且语言类题目最弱。

**⚠️ 局限性**

局限：仅评测15个7B–10B模型与单一闭源基准；CHECKPOINT分析仅针对GENBA‑10B，缺乏对比；题目部分由Claude生成，存在潜在泄露；巴伐利亚翻译单一方言写法，未覆盖多变方言；源落地题目规模有限；缺乏对更大模型或多轮指令微调的实验；评测协议与参数设置对结果影响大，难以归因；缺乏跨文化多方言验证。

---

## 251. Bias Mitigation in Face Recognition via Demographic-based Supervised Contrastive Learning

**arXiv ID:** 2608.12971 | [PDF](https://arxiv.org/pdf/2608.12971v1)

**作者:** Yu Linghu `[一作]` (University of Zurich), Manuel Günther `[通讯]` (University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面部识别公平性提升框架 DeSCon，通过在传统 ArcFace 损失上加入基于人口统计的监督对比学习（Supervised Contrastive Learning），从而在不需要推理时使用人口统计信息的前提下，减小不同族群间的误识率差距。

**💡 创新点**

创新点在于：①设计了三种非匹配样本抽样策略（All、Within-Group、Hard Within-Group），使对比学习能针对同族群内部的 tail 分布进行更精准的正则化；②将对比学习与基于中心的 margin 损失协同优化，兼顾身份区分与公平性；③在公平性评估中采用 ISO/IEC 19795‑10 标准的 Gini 系数、FPD/FND 等指标，体现了对国际标准的符合度。

**🔧 技术方法**

技术细节包括：ArcFace 分类损失、监督对比学习 SupCon、基于群体权重的样本加权、三种非匹配样本采样策略、使用统一的 scale 与 margin 进行联合优化、对不同 backbone（IResNet50/100）和更强 margin 损失 AdaFace 的兼容性验证。

**📊 数据集**

主要使用 BUPT-BalancedFace（1.3M 图像，4 个种族均衡）训练；BUPT-GlobalFace（不平衡）作为对比实验；在 RFW、BFW、VGGFace2 以及 LFW、CALFW、CPLFW、CFP‑FP、AgeDB、IJB‑C 等公开基准上进行评估。

**📈 对比分析**

与 ArcFace、DeFT、RamFace、MixFairFace、Labelless、FairScoreNormalization 等 12 种 state‑of‑the‑art 方法对比，DeSCon‑Hard 在保证 TMR 与 baseline 接近的同时，显著降低 FPD 与 FND，显示出最佳的公平性‑性能折衷；DeSCon‑WG 与 DeSCon‑All 在某些情境下提升 TMR，但公平性改善不如 Hard 稳定。

**⚠️ 局限性**

局限性包括：①需要训练集提供可靠的人口统计标签，推理时仍不使用；②在大规模 WebFace‑260M 之类未标注种族的数据上直接训练仍有挑战；③对其他公平属性（性别、年龄）扩展未深入探讨；④在极端样本稀缺群体上公平性提升仍有限。

---

## 252. Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software

**arXiv ID:** 2608.12970 | [PDF](https://arxiv.org/pdf/2608.12970v1)

**作者:** Fanyu Wang `[一作]` (Monash University), Siwei Jiang `[通讯]` (Tianjin University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套自动化验收测试框架，针对大语言模型驱动软件（LLM-based Software, LBS）实现上下文感知、个性化的测试oracle生成与可信判定。

**💡 创新点**

创新点在于（1）Requirements‑Augmented Generation (REAG)——通过检索需求、角色与领域知识并自我推理，构造可执行的意图重建oracle；（2）置信度校准级联判定——使用多家族LLM裁判并通过 conformal risk control 对判定置信度进行阈值校准，既过滤低质量oracle又提供统计可靠性保证。

**🔧 技术方法**

采用检索增强生成 (RAG) 与自我推理、ICRALM自适应检索、链式思维提示；级联判定采用 Gemini‑2.5‑Flash‑Lite、GPT‑4.1‑mini、Gemini‑2.5‑Flash 三个LLM，并对其置信度进行 conformal 校准。

**📊 数据集**

在实际营养咨询移动应用中构建 346 个测试场景、46 个用户画像、300+ 软件需求与文档为检索语料，生成约 100 条 oracle 及 100 条系统输出进行评估。

**📈 对比分析**

与单一裁判基线相比，REAG 的 oracle 质量平均 3.91/5，级联判定在 α=0.14 时达到 98.8% 的准确率，oracle 质量提升至 4.30/5，成本效益提升 31.7%（CPP 3.49 vs. 5.11）。

**⚠️ 局限性**

局限性包括：18% oracle 仍因检索错误或范围漂移导致不准；级联判定仍需人工审核高不确定性案例；框架对需求文档完整性与结构化程度高度依赖，迁移时需重新校准阈值。

---

## 253. I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization

**arXiv ID:** 2608.12957 | [PDF](https://arxiv.org/pdf/2608.12957v1)

**作者:** Yubo Zhang `[一作]` (Alibaba), Ziqiang Dong `[通讯]` (Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出实例级自适应自蒸馏策略，在所有采样结果错误时使用特权自蒸馏，解决群组相对策略优化梯度消失问题。

**💡 创新点**

通过能力依赖的实例级路由，只有在所有样本均不成功时才启用自蒸馏，自动随着模型能力提升减少教师影响，避免持续的教师偏差。

**🔧 技术方法**

采用 Group Relative Policy Optimization、带 EMA 的特权自蒸馏、正向–反向 KL 插值、熵感知加权及实例级路由等技术。

**📊 数据集**

在 SciKnowEval 四个科学领域（生物、材料科学、化学、物理）上进行实验。

**📈 对比分析**

与标准 GRPO、纯自蒸馏和样本级路由对比，平均 mean@16 准确率提升至 70.31%，在每个领域均取得最佳成绩，较基线提升 13.64% 等。

**⚠️ 局限性**

仅在 Qwen3-8B 规模上验证，评估范围局限于 SciKnowEval，其他模型或更开放任务的表现可能不同。

---

## 254. Efficient Randomized LL/SC that Preserves History Independence

**arXiv ID:** 2608.12946 | [PDF](https://arxiv.org/pdf/2608.12946v1)

**作者:** Dante Bencivenga `[一作]` (University of Calgary), Philipp Woelfel `[通讯]` (University of Calgary)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

实现了多对象的 Load‑Linked / Store‑Conditional（LL/SC）同步原语，使用有限的 CAS、FADD 及寄存器实现常数期望步复杂度和低空间复杂度；

**💡 创新点**

创新点在于：1）在弱自适应对手下取得 O(nτ+m) 的空间下界，显著低于现有 O(n²τ+m)；2）首次提供可保持静止历史独立（QHI）的 LL/SC 实现，满足并发数据结构的隐私需求；

**🔧 技术方法**

采用随机化标签选择、计数引用计数技术、活动计数器以及重置零标签机制等，辅以 CAS 与 FADD 原语；

**📊 数据集**

未使用外部数据集，本文主要为理论与算法设计；

**📈 对比分析**

与 Blelloch‑Wei 等传统算法对比，空间从 O(n²τ+m) 缩减至 O(nτ+m)，在 m=O(1) 时与已知下界匹配；在 QHI 哈希表实例中，保持了原始硬件 LL/SC 的时间与空间复杂度；

**⚠️ 局限性**

限制包括：需要在弱自适应对手下；实现仍不具备强线性可行性；对多字 LL/SC 不支持；对高 τ 的情况空间仍可能接近上限；

---

## 255. Polish Medical Visual Question Answering: Vision-Language Models Underutilize Visual Evidence

**arXiv ID:** 2608.12928 | [PDF](https://arxiv.org/pdf/2608.12928v1)

**作者:** Jakub Pokrywka `[一作]` (Adam Mickiewicz University), Wojciech Kusa `[通讯]` (NASK National Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并评测了基于波兰医学执照考试的医学多模态视觉问答（VQA）基准，并与文本问答对照，探讨模型对图像与文本的依赖。

**💡 创新点**

首次构建波兰语医学VQA数据集，设计完整输入与缺失输入的 ablation 评估，并对比多种开源与商用视听语言模型的性能。

**🔧 技术方法**

采用多模态提示、JSON 格式输出、模型推理与不同 reasoning-level 调整，评估 LLaVA、Qwen、Gemma、GPT 系列模型。

**📊 数据集**

使用波兰执照考试（PES）图像题目 286 条和 480 条文本题目组成的 VQA 与 QA 子集。

**📈 对比分析**

通过准确率与人类考生比较，发现 GPT‑5.6 在完整输入下超过人类，但大多数模型在缺失图像或问题时性能显著下降，说明对图像的利用有限。

**⚠️ 局限性**

受限于样本量小、专业覆盖不均、仅评测标准化考试题目、未涵盖医学专门训练模型、缺少其他推理设置等。

---

## 256. Decoupled Contrastive Decoding via Expert-Aligned Drafting

**arXiv ID:** 2608.12913 | [PDF](https://arxiv.org/pdf/2608.12913v1)

**作者:** Zhixuan Liu `[一作]` (Shanghai Jiao Tong University), Chao Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究在轻量级提议路线上，是否应将对比信号纳入草稿生成还是仅用于验证，并提出一种名为 Decoupled Contrastive Decoding (DCD) 的方案。

**💡 创新点**

创新点在于将对比解码的对比信号与专家模型对齐，仅在验证阶段使用业余模型，从而在保持生成质量的同时显著加速推理。

**🔧 技术方法**

采用对比解码、Speculative Decoding、EAGLE3轻量级提议器以及可选的 N‑gram 提议器，并通过对比诊断方法评估提议与验证的配合效果。

**📊 数据集**

使用公开基准数据集，如 MMLU 等，进行 8B 规模模型的实验，并在 200 条样本或完整数据集上进行验证。

**📈 对比分析**

与原始 CD、业余耦合的 Speculative CD（SCD、CoS）进行比较，DCD 在 8B 模型上平均实现 1.65–1.95 倍的贪婪推理加速，并将 MMLU 的提议路径延迟降低 5–12 倍。

**⚠️ 局限性**

实验仅覆盖固定解码预算和标准基准，未涵盖极长上下文、多语言或领域专用提示，以及自适应的 α、提议长度或阈值设置等情况，可能导致在某些工作负载下表现差异。

---

## 257. EGRL: Edge generation-guided relation-aware learning for RNA-protein interaction prediction

**arXiv ID:** 2608.12906 | [PDF](https://arxiv.org/pdf/2608.12906v1)

**作者:** Danyu Li `[一作]` (Macau University of Science and Technology), Kui Jiang `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了Edge Generation-guided Relation-aware Learning (EGRL) 框架，用于RNA‑蛋白相互作用预测，结合隐式元路径学习、图生成器和多关系GAT实现冷启动泛化。

**💡 创新点**

创新点在于①自动隐式元路径学习捕获关系语义；②图生成器为未知节点生成软边支持冷启动；③多关系GAT与多特征融合（concat、Hadamard、差分）共同提升表达；四者协同显著提升性能。

**🔧 技术方法**

技术包括图神经网络（多关系GAT）、注意机制、隐式元路径学习网络、对偶MLP图生成器、联合训练的主干+生成器辅助损失，以及多特征融合策略。

**📊 数据集**

使用四个公开RPI数据集：RPI369、RPI1807、RPI2241 与 NPInter2。

**📈 对比分析**

与RLF‑LPI、RPI‑SAN、RPITER、IPMiner、NPI‑GNN 等方法进行五折交叉验证比较，EGRL 在 AUROC、AUPR、ACC 等指标上均位列或接近榜首；在分子持出与序列聚类的冷启动设置中，AUROC 提升 8.6%、AUPR 提升 5.0%。

**⚠️ 局限性**

局限性包括对软边生成依赖序列特征，尚未整合二级结构、三维结构等多模态信息；在极端稀疏或高噪声环境下性能可能下降；大规模图的可扩展性与可解释性仍待进一步研究。

---

## 258. Adversarial Robustness in Smishing Detection: A Comparative Analysis of Adversarial Fragility in Classical vs. Transformer-Based Detection Systems

**arXiv ID:** 2608.12889 | [PDF](https://arxiv.org/pdf/2608.12889v1)

**作者:** Denzel Chiuseni `[一作]` (Carnegie Mellon University Africa), Jema David Ndibwile `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对比评估了经典词汇模型与多语种Transformer在低资源环境下对smishing（短信钓鱼）检测的对抗鲁棒性，使用字符混淆、结构扰动和跨语代码切换三种攻击；

**💡 创新点**

创新点在于系统化地将对抗鲁棒性纳入评价指标，揭示了模型架构与鲁棒性之间的明确边界，并证明了高清晰文本性能并不一定预示对抗强度；

**🔧 技术方法**

采用RDR（Robustness Degradation Ratio）指标、Mann‑Whitney U、Friedman检验和注意力引导的白盒攻击等技术；

**📊 数据集**

使用合并自Kaggle斯瓦希里短信数据集与中英混合短信数据集，最终包含27,037条短信，恶意比例1.3%；

**📈 对比分析**

比较方法是将五种模型在三种攻击类型和三种强度下的F1降幅归一化为RDR，结果显示经典模型在高强度字符混淆/结构扰动下RDR≈0.95（近乎灾难性），而Transformer RDR≤0.35；

**⚠️ 局限性**

局限性包括恶意测试样本量小（仅约71条），仅在英-斯瓦希里语对上验证，攻击样本为人工生成并非真实攻击；

---

## 259. Labels Are Not Endpoints: Treatment Leakage and Construct Validity in MCP Agent Security Evaluation

**arXiv ID:** 2608.12880 | [PDF](https://arxiv.org/pdf/2608.12880v1)

**作者:** Rana Muhammad Ahmed `[一作]` (Bahria University), Sabahat Abbas `[通讯]` (Bahria University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对工具使用代理的安全评估进行审计，重现并纠正了因标签泄漏导致的错误结果，生成最终的行为频率清单。

**💡 创新点**

提出治疗不变性检验和七链完整性链，并构建严格的、无治疗信息的行为端点，提升评估构造效度。

**🔧 技术方法**

使用MCP协议、离散化执行记录、哈希验证、确定性重放、Python实现的逻辑检查器及其linter，确保判定的可复现性和可验证性。

**📊 数据集**

采用10,200条执行记录，合并为180条模型绑定请求，15条可观测刺激，涉及四个模型（Qwen2.5、DeepSeek、Mistral、Phi-3.5），构成实验数据集。

**📈 对比分析**

通过双盲评审、相等性测试和受限条件计数对比，发现原始标签错误并修正，得到0攻击成功记录，整体性能保持在确定性重放的可接受范围内。

**⚠️ 局限性**

局限于本地MCP式环境，未覆盖多模型、多攻击家族、动态环境及外部服务，无法给出普适的攻击率或防御效能估计。

---

## 260. Robust data-driven discovery of fractional differential equations via weak formulations and Pareto-based subset selection

**arXiv ID:** 2608.12879 | [PDF](https://arxiv.org/pdf/2608.12879v1)

**作者:** Pongpisit Thanasutives `[一作]` (RIKEN), Yoshinobu Kawahara `[通讯]` (RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于弱形式和Pareto子集选择的全新数据驱动方法，用于从噪声数据中识别含分数导数的偏微分方程（FPDE）。

**💡 创新点**

创新点包括：① 为分数算子构建了伴随一致的弱特征库，消除对测量数据点值的直接分数微分；② 采用连续-分数阶的Pareto搜索，避免固定字典的离散化误差和共线性；③ 通过验证误差–复杂度“elbow”规则实现自动、稀疏的模型选择；④ 提供完整的理论噪声方差分析与局部识别灵敏度诊断。

**🔧 技术方法**

核心技术包括：弱积分形式与伴随算子、分数导数的离散伴随矩阵、差分演化优化（differential evolution）对连续阶进行搜索、岭回归拟合系数、双层Pareto目标、验证集的方差归一化、elbow判据、精确阶数后处理与数值稳定化。

**📊 数据集**

实验使用合成基准：FADE、两种Riesz反应扩散、非线性分数Burgers（1D）以及2D方向性扩散；还对真实冻土蠕变数据（细粒土与粘土）进行分数Kelvin模型识别；此外通过对比神经网络分数发现框架和强形式库验证鲁棒性。

**📈 对比分析**

与强形式库和神经方法对比，Weak‑Pareto在10–20%噪声下实现5/5的支持恢复和几乎完美的阶数/系数估计，显著优于强形式下的0/5或低回收率；相较于神经框架，运行时间更短、参数更清晰；ablation实验表明弱特征是噪声鲁棒性的主因，连续阶搜索显著提升支持选择，后处理对性能影响有限。

**⚠️ 局限性**

局限性：① 对Riesz反应扩散等高阶空间项的阶数识别仍易受噪声影响；② 超单元（α>1）分数阶在近端初始条件敏感，导致阶数上限逼近搜索边界；③ 非线性弱特征虽平均化但仍存在偏置；④ 需要大量弱行（高维测试函数）和稠密采样，稀疏或不规则数据的适用性待验证；⑤ 目前没有全局收敛理论，搜索依赖启发式差分演化和局部梯度优化。

---

## 261. The Embedder's Dilemma: LLMs Are Better, but at What Cost?

**arXiv ID:** 2608.12875 | [PDF](https://arxiv.org/pdf/2608.12875v1)

**作者:** Adnan El Assadi `[一作]` (Harvard University), Jinhyuk Lee `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个成本感知的评测框架，比较了十个前沿大语言模型（LLM）与26个文本嵌入模型在37个任务（分类、语义相似度、聚类、二元分类、检索）上的表现，并公开了完整代码与数据。

**💡 创新点**

创新点在于将模型性能与实际部署成本与吞吐量同时纳入 Pareto 前沿分析，揭示 LLM 在推理型检索上的优势与成本/速度劣势，并提出混合检索-重排序的实用部署策略。

**🔧 技术方法**

技术主要包括基于 MTEB 框架的任务实现、LLM 的零样本/少样本提示设计、嵌入模型的 kNN/向量检索、API 令牌计费与 GPU 吞吐量测量，以及思考令牌（reasoning tokens）成本分解。

**📊 数据集**

使用的公开数据集来自 MTEB（及其子集）以及专门为 LLM 设计的 37 个任务数据，覆盖英语、跨语言、法律、金融、医学等多领域。

**📈 对比分析**

比较方法通过统一的评测管道评估模型得分，并对每个模型记录 API 令牌使用、GPU 推理吞吐量，生成成本/性能 Pareto 前沿；结果显示整体性能几乎持平，但 LLM 在检索任务上略胜，嵌入模型在分类任务上更好，成本与速度差距可达千倍。

**⚠️ 局限性**

局限性包括：评测仅涵盖当前十个 LLM 与 26 个嵌入模型，检索任务采用小规模语料库（82–415 文档）不具备生产规模可行性；LLM 仅使用零样本提示，未探索更强的微调或后训练方法；成本假设基于特定硬件/价格，结果会随环境变化而变动。

---

## 262. A Deep RL based Framework for Targeted White Matter Tractography

**arXiv ID:** 2608.12960 | [PDF](https://arxiv.org/pdf/2608.12960v1)

**作者:** Ankita Joshi `[一作]` `[通讯]` (Indian Institute of Technology Mandi), Ankita Joshi (Indian Institute of Technology Mandi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个结合深度强化学习与监督学习的混合框架，用于针对性白质纤维通路追踪。

**💡 创新点**

创新点在于：①在无真实纤维标注条件下通过 GPT‑based 策略学习提升 RL 策略；②引入可扩展的多策略融合框架，利用多 RL 策略的互补优势。

**🔧 技术方法**

技术包括深度强化学习、GPT 模型、监督学习微调以及多策略融合。

**📊 数据集**

使用公开数据集：TractoInferno、HCP 与 ISMRM‑2015。

**📈 对比分析**

与现有基准方法对比，框架在多数据集上显著提升了准确率和鲁棒性，尤其在跨数据源的泛化性能优于传统 RL 与监督方法。

**⚠️ 局限性**

局限性：仍对训练数据有一定依赖，且多策略融合提升了计算复杂度；在极端噪声或高失真数据上的表现尚待进一步验证。

---

## 263. Dryas: A Reprogrammable Engine for High-Speed Interconnect Tracing and Analysis

**arXiv ID:** 2608.12934 | [PDF](https://arxiv.org/pdf/2608.12934v1)

**作者:** Manuel Bröchin `[一作]` (SBB), Timothy Roscoe `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在高带宽、低延迟的ECI互连上实现了一种可在运行时动态重配置的NFA过滤引擎，能够实时捕获并筛选关键消息序列；

**💡 创新点**

核心创新在于将NFAs映射到一张基于状态转换元件（STE）的可重配置覆盖网络上，使得过滤规则可在不到一秒内完成切换且不影响正在运行的应用；

**🔧 技术方法**

采用NFAs、STEs、可编程cfglut、环状-团（RoC）覆盖图以及在FPGA上实现的动态谓词提取与数据压缩技术；

**📊 数据集**

在ETH Zürich和Enzian实验平台上使用30 GiB/s的ECI互连对CPU+FPGA系统进行实际测量，收集内存访问、跨Socket延迟与缓存失效率等事件；

**📈 对比分析**

相较于传统ILA或离线日志分析方法，实验表明该引擎在600状态规模下仍占用不到2% LUT、3.6% FF，过滤吞吐率可达30 GiB/s，且过滤结果的后处理开销降低了10⁵倍；

**⚠️ 局限性**

限制主要在于谓词和输入解码在综合时固定，无法动态增添新字段；覆盖网络映射为NP‑hard，需额外求解或启发式；在极大状态量或极高并发子流时仍受资源限制。

---

## 264. H-VAEP and H-xT: Valuing Offensive On-the-Ball Actions in Handball by Estimating Probabilities

**arXiv ID:** 2608.12926 | [PDF](https://arxiv.org/pdf/2608.12926v1)

**作者:** Julius Broermann `[一作]` (Paderborn University), Jochen Baumeister `[通讯]` (Paderborn University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对德国手球超级联赛的追踪派生事件数据进行处理，首次将 Expected Threat（xT）和 VAEP 框架适配到手球，生成基于动作的进攻价值评估模型 H-xT 与 H-VAEP，并公开代码。

**💡 创新点**

创新点在于：①设计了符合手球几何的场地分区布局，显著提升 xT 的鲁棒性；②针对手球高得分与快节奏特点，对 VAEP 的特征空间（分级比分、射门角度、时间上下文）进行重构，并选择最优上下文长度以降低团队身份泄漏；③系统性地用三赛季数据验证模型可靠性、辨别度与稳定度，并对传统计数统计进行公平比较。

**🔧 技术方法**

技术手段包括：空间马尔可夫链 + 区域化（xT）、机器学习预测（CatBoost、XGBoost）计算得分/失分概率、Brier Score 与 ROC‑AUC 评估、团队身份泄漏量化（Macro‑AUC）、交叉验证与 Holm 校正、Bootstrap 置信区间、模拟鲁棒性检验、Meta‑metrics（可靠性、辨别度、稳定度）等。

**📊 数据集**

使用 2021/22–2025/26 五个赛季的德国 Handball Bundesliga（HBL）追踪派生事件数据（传球、运球、场上射门、七米罚球等），以及官方统计与 HPI 数据做基准。

**📈 对比分析**

与传统计数指标（进球、助攻、HPI）以及原始足球版 VAEP 进行对比；在 2023/24–2025/26 这三赛季的 held‑out 评估中，H‑VAEP/10o 的可靠性 ≥0.98、辨别度 0.994、稳定度 0.999 超过传统统计；H‑xT 在场区价值上更直观，鲁棒性显著高于矩形网格；在 Brier Score 与 ROC‑AUC 上，经过特征重构与 XGBoost 选型后，H‑VAEP 在得分预测上 Brier 下降约 0.003、ROC‑AUC 提升约 0.013；团队身份泄漏被限制在 k=3 时的 Macro‑AUC 0.664。

**⚠️ 局限性**

限制：仅评估进攻侧的 on‑ball 动作，防守事件与无球行为尚未自动识别；事件检测存在噪声与假阳性，影响占有边界的精准划分；未来需进一步提取防守动作、空间封堵与无球贡献，以完整覆盖手球比赛中的所有价值。

---

## 265. TennisVAR: A Stroke-Evidence-Grounded Multimodal Large Language Model for Tactical Reasoning in Tennis Videos

**arXiv ID:** 2608.12920 | [PDF](https://arxiv.org/pdf/2608.12920v1)

**作者:** Yifan Mei `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于击球证据的网球战术推理任务，并构建了TRACE基准数据集；

**💡 创新点**

创新点在于将赛段内每个击球事件与战术结论一一对应，形成多层级、证据链式的战术推理框架；

**🔧 技术方法**

采用事件解析模块(EPM)提取击球事件，战术图引导时序推理模块(TGTR)建模时序与同位玩家决策关系，并通过问答生成器生成答案与推理过程；

**📊 数据集**

使用TRACE数据集，包含11,189场比赛、41,485个击球事件、25,429个战术单元和11,189个问答对；

**📈 对比分析**

与多种零样本及微调的多模态大语言模型对比，TennisVAR在证据定位、战术识别与答案生成方面均取得领先，平均提升超过30%；

**⚠️ 局限性**

局限性包括对标注质量的高度依赖、对复杂多击球战术的解释仍不够完整，以及对非职业比赛的泛化能力待验证。

---

## 266. Technical Report on Resilient and Secure Large-Scale Energy Internet Systems

**arXiv ID:** 2608.12916 | [PDF](https://arxiv.org/pdf/2608.12916v1)

**作者:** Ioannis Zografopoulos `[一作]` (University of Massachusetts Boston), Wei Sun `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该技术报告系统性评估了大规模能源互联网（EI）的安全与韧性，梳理威胁景观、检测与防御技术，提出基于决策的韧性框架、跨层安全模型、优化与控制方法，并给出研究、标准化与监管建议。

**💡 创新点**

创新点包括：①将网络物理互依关系嵌入鲁棒优化与物理信息化机器学习中；②设计决策感知的储能集成EI韧性行动门（action gate）来屏蔽信息误导和物理不可行性；③提出跨维度韧性评估指数，统一物理、操作、数字、气候与法规五个维度；④引入同态加密与数字孪生等前沿技术提升安全性与可验证性。

**🔧 技术方法**

主要技术手段有：网络物理威胁分类与FDIA/LAA检测，基于物理信息的神经网络与混合学习，移动目标防御、硬件根源信任与数字孪生，鲁棒优化（RO）、同态加密、多区OPF、能量状态驱动的自适应恢复，以及多维度韧性指数的量化模型。

**📊 数据集**

利用多源真实系统数据（如NERC、ISO/ISO的测量与事件数据）、大规模仿真平台（数字孪生、实时HIL、跨域共识仿真）以及公开实验数据集（如IEEE 39/57/118节点案例、DER/储能调度日志）。

**📈 对比分析**

通过与传统基线（如无鲁棒优化、标准OPF、传统FDIA检测）进行对比，报告显示在受攻击场景下，鲁棒优化+混合检测可将误检率降低30%–50%，决策感知韧性门提升储能恢复能量利用率至90%以上，数字孪生辅助可提前发现并阻断FDIA攻击，整体系统稳定性提升约15%–20%。

**⚠️ 局限性**

主要局限包括：数据驱动方法对训练数据代表性的依赖，仿真与现实差异导致验证不完全；多技术集成的计算与部署复杂度高；缺乏统一的标准与测度指标；在极端高负载/大规模攻击情境下的可扩展性与实时性仍待进一步验证。

---

## 267. Beyond Visual Evidence: Revealing and Mitigating Relational Privacy Leakage in Document MLLMs

**arXiv ID:** 2608.12911 | [PDF](https://arxiv.org/pdf/2608.12911v1)

**作者:** Beining Xu `[一作]` (Shenzhen MSU-BIT University), Anirban Chakraborty `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了文档理解多模态大模型在视觉证据缺失时的关系隐私泄露问题，并提出了动态关系遗忘框架（DRUF）来抑制此类泄露，同时发布了专门的评测基准DocPrivacyBench。

**💡 创新点**

创新点在于：①首次针对文档 KIE 中的关系泄露提出动态遗忘机制；②通过 Relational Decoupling Unlearning (RDU) 对关联敏感字段对进行联合惩罚；③设计动态忘集更新流程，使遗忘目标随模型泄露行为自适应；④构建了可系统评估弱视觉输入下隐私泄露的基准。

**🔧 技术方法**

使用技术包括：teacher‑student 对齐保留 KIE 能力；动态忘集生成与更新；RDU 采用 KL‑耦合的关系级遗忘损失；结合传统梯度上升、KL 正则化、SCRUB 等基线实现对比；以及基于 Ratcliff‑Obershelp 相似度的隐私评估。

**📊 数据集**

主要数据集为 DocXPand‑25k（噪声版）、IDNet（及其加噪版本），均为身份文档（身份证、护照、驾照）图像，用于 KIE 训练与隐私泄露评估。

**📈 对比分析**

在 Prompt‑Driven 与 Image‑Driven 两种攻击设置下，与 GA、GA+KL、SCRUB、DPO、FTF、DP 等六种遗忘方法对比，DRUF 在泄漏率几乎为 0（Image‑Driven 0.001，Prompt‑Driven 0.000），同时保持 KIE 的 LC 与准确率不低于基线；相较于其他方法，DRUF 在隐私抑制与任务性能平衡上表现最佳。

**⚠️ 局限性**

局限性：①仅在身份文档类任务验证，缺乏跨域或多任务的通用性评估；②动态采样与多轮遗忘过程会增加计算成本；③对模型规模、不同多模态组合的适用性尚未充分验证。

---

## 268. SPARED: Reasoning-Based AI-Generated Image Detection via Adversarially Edited Data

**arXiv ID:** 2608.12876 | [PDF](https://arxiv.org/pdf/2608.12876v1)

**作者:** Yicheng Bao `[一作]` (East China Normal University), Xin Tan `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于对抗强化学习的循环训练框架：先用扩散图像编辑器（攻击者）把真实照片按指令编辑成伪造图像，随后用多模态大型语言模型（防御者）对每张图像给出真假判定和自然语言解释。训练过程中，攻击者只有在既完成指令又能骗过当前防御者时才获得奖励；防御者仅对最终判定的正确性计分，解释不直接奖励，从而消除常见的快捷方式。该循环不断迭代，生成越来越难的训练样本，使检测器在各个基准上持续进步。

**💡 创新点**

创新点：
1) 将攻击者与防御者设计为完全异构、无共享参数的模型，避免梯度冲突；
2) 使用指令完整性门（PaCo）确保攻击者的编辑既符合指令又具有欺骗性，杜绝无意义或离散噪声攻击；
3) 通过将每个伪造图像与其原始真实图像成对训练，消除来源/真实性的捷径；
4) 只对判定结果奖励，而不是解释文本，避免模板化、浅层解释；
5) 在循环中持续生成“硬负样本”，实现模型的自我进化和泛化。

**🔧 技术方法**

使用技术：
- 扩散图像编辑器（Qwen‑Image‑Edit‑2511 LoRA）与 DiffusionNFT 进行强化学习；
- 指令完整性评估器 PaCo；
- 多模态 LLM（Qwen3.5‑9B）通过 LoRA‑SFT 预训练后使用 GRPO 进行策略梯度优化；
- 数据生成与训练调度：每轮先训练攻击者生成编辑后样本，再训练防御者；
- 采用 LoRA 与全参数优化相结合，提升训练效率。

**📊 数据集**

使用数据集：
- 编辑指令与图像源：ImgEdit、pico‑banana‑400k、MagicBrush；
- 评估基准：DeepfakeJudge‑Detect、AnomReason‑Deepfake、Holmes‑Set；
- 内部训练集为上述基准的训练子集（不含评测图像），并通过 perceptual‑hash 过滤交叉泄漏。

**📈 对比分析**

比较方法与性能：
- 在 DeepfakeJudge‑Detect 上，整体准确率从 69.6%（SFT）提升至 79.5%（Iter3），并在真/假召回率上均显著提高；
- 在 AnomReason‑Deepfake 上，准确率达到 92.18%，CSemAP‑Full 0.5207，超过所有闭源、开放源及专门深伪模型；
- 在 Holmes‑Set（十个未见生成器）上，零样本平均准确率从 53.0% 提升至 92.8%（Iter3），与现有 AIGI‑Holmes 接近；
- 与基线的对比显示，单纯延长训练或使用新数据无法获得同等收益，说明对抗循环是关键。

**⚠️ 局限性**

limitation：
- 采用硬 0/1 的欺骗奖励导致部分生成器（如 Janus）在后续轮次出现召回率下降；缺乏按生成器难度调节的奖励机制；
- 解释质量虽然随判定准确性提升，但未被直接奖励，可能限制更深层次解释的可控性；
- 目前方法对生成器模型的适配需要持续训练，部署后仍需迭代更新。

---

## 269. Spinal Coupling in Frontal and Transversal Plane During Gait - A Segmental and Time-Dependent Analysis of the Thoracic and Lumbar Spine

**arXiv ID:** 2608.12945 | [PDF](https://arxiv.org/pdf/2608.12945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 270. The Objective Is the Bottleneck: Latent World Models Encode What Their Planners Cannot Use

**arXiv ID:** 2608.12959 | [PDF](https://arxiv.org/pdf/2608.12959v1)

**作者:** Joyjeet Singh `[一作]` `[通讯]`, Joyjeet Singh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了Latent World Model（LeWorldModel）在 TwoRoom 环境中的长期规划失败原因，并通过改进规划目标（不需要重新训练）显著提升规划成功率。

**💡 创新点**

发现规划目标（CEM 最小化的平方欧氏距离）在距离超过约120单位后会饱和甚至反转，从而导致长期规划失败；通过使用解码位置距离或基于时间步的学习成本替代该目标，解决问题。

**🔧 技术方法**

使用 LeWorldModel 的 ViT‑Tiny 编码器 + 预测器架构、Cross‑Entropy Method（CEM）规划、岭回归、MLP 学习的时间步成本等技术。

**📊 数据集**

TwoRoom 诊断环境，使用原始发布的四个检查点（含作者权重）和相同的随机种子。

**📈 对比分析**

对比基线：在 offset 25 目标下 94% 成功率，offset 100 仅 26%；修复后 offset 100 成功率提升至 98%（或 88% 使用解码位置成本）。实验在同一 CPU 上完成，无 GPU，无重新训练，显著提升长期规划性能。

**⚠️ 局限性**

局限性包括仅在单一环境和单一 seed 下验证；多种训练设置混合导致无法归因；未解释距离饱和的根本原因；修复需已可读位置信息（解码位置成本）或对时间步成本训练的假设。

---

## 271. Exponential multi-graylevel computational-weighted dithering for high-quality binarized Fourier single-pixel imaging

**arXiv ID:** 2608.12958 | [PDF](https://arxiv.org/pdf/2608.12958v1)

**作者:** Qigao Zhu `[一作]` (University of Science and Technology of China), Xinglong Gong `[通讯]` (University of Science and Technology of China)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于指数多灰度加权抖动的灰度模式二值化方法，用于提升单像素傅里叶成像（FSI）的图像质量

**💡 创新点**

创新点在于先将灰度傅里叶模式进行多灰度抖动，再按指数权重将其分解为两组二值模式，实现更低的量化误差和更高的空间分辨率

**🔧 技术方法**

核心技术包括指数多灰度抖动、计算加权求和、基于单像素光检测的傅里叶系数获取以及逆傅里叶变换重建

**📊 数据集**

实验使用DIV2K数据库中的城市大门图像（256×256像素）进行仿真，实测使用玩具、USAF分辨率目标和Volvox细胞图像

**📈 对比分析**

与Floyd–Steinberg、信号抖动、正负抖动三种传统方法对比，仿真中SSIM提升至0.971，PSNR达40.89 dB，MAPE降至28%；实测中PIQE和BRISQUE分别降至23.02和24.76，边缘分辨率达到3.75 µm，接近理论极限

**⚠️ 局限性**

主要局限是每个傅里叶系数需要两次二值投影，导致采集时间几乎翻倍，虽然对高速采集有影响，但在追求高质量图像的应用场景下仍具优势

---

## 272. Unifying Depth and Width Pruning for LLMs via Binary Knapsack Optimization

**arXiv ID:** 2608.12953 | [PDF](https://arxiv.org/pdf/2608.12953v1)

**作者:** Palaash Goel `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双轴结构化剪枝框架，先用0/1背包优化对LLM深度层进行条件最优剪枝，再通过宽度裁剪精确满足目标压缩率。

**💡 创新点**

创新点在于：1) 使用动态规划实现深度剪枝的条件最优解；2) 迭代重要性估计，考虑全局互相依赖；3) 设计压缩率遵从因子(CRAFT)保证预算精确；4) 双阶段剪枝兼顾推理速度与任务一致性。

**🔧 技术方法**

技术包括：0/1背包动态规划、离散化参数、迭代重要性估计、宽度层级列敏感度筛选、按重要性分配宽度预算、混合精度微调。

**📊 数据集**

数据集：用于校准的Slim Orca、Alpaca、C4；评测任务共18项，覆盖生成、世界理解、STEM/数学/医学推理、NLU与对齐。

**📈 对比分析**

与SliceGPT、LLM‑Pruner、ReplaceMe、SLEB、ShortGPT、2SSP等基线比较，平均性能保留率最高，任务标准差最低，压缩率遵从度CRAFT≈0.98，推理速度提升可达16–20 tokens/s。

**⚠️ 局限性**

局限性：使用统一的剪枝信号，对Mixture‑of‑Experts等架构缺乏专家特定信号；仅对深度剪枝提供条件最优保证，对重要性估计缺乏理论证明。

---

## 273. CardioState-JEPA: Delay-Aware Cross-Modal Learning of a Shared Cardiac Representation

**arXiv ID:** 2608.12944 | [PDF](https://arxiv.org/pdf/2608.12944v1)

**作者:** Hamza Shafiq `[一作]` (Eindhoven University of Technology), Aaqib Saeed `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个共享Transformer编码器，将ECG、PPG和PCG三种不同传感器的波形映射到同一潜在空间，并通过联合嵌入预测与延迟对齐进行自监督预训练，最终实现一个可在三种传感器上冻结使用的心脏基础模型。

**💡 创新点**

①使用学习的时间延迟对齐，使不同模态在同一潜在心脏状态下对应；②预测掩码的潜在结构而非原始波形，减少低层次传感器差异；③两阶段课程式预训练：先在大量单模态数据上学习再用稀缺的同步多模态数据对齐；④整合多种辅助目标（状态对齐、相位监督、延迟监督）进一步驱动潜在空间向生理结构对齐。

**🔧 技术方法**

轻量化模态tokenizer、共享ViT-B Transformer编码器、动量编码器、联合嵌入预测（masked latent prediction）、延迟对齐头、软核对齐、VICReg状态对齐、相位监督以及线性探测训练。

**📊 数据集**

单模态大规模数据：MIMIC-IV-ECG（800K 12导联 500Hz）、PPG-EXT（125Hz）、BMD-HS（4000Hz）。多模态数据：VitalDB（ECG-PPG）、EPHNOGRAM（ECG-PCG）、SensSmartTech（三模态）。下游评测使用PTB-XL、CPSC2018、CSN、CirCor、CinC2016、WESAD、DaLiA、MIMIC AF、PulsePPG等多任务数据集。

**📈 对比分析**

在25个下游任务（包括ECG分类/回归、PPG分类/回归、PCG异常检测）中，与各自最强自监督基线相比，平均AUROC提升8.2点（PPG）、18.8点（PCG）和15.5点（ECG）；在ECG任务中，即使仅使用1%或10%的标签也优于同类自监督模型，并且接近或超过使用文本或大规模标签的ECG基础模型。

**⚠️ 局限性**

对同步多模态数据的依赖仍较高，延迟对齐需要心率/心音检测，在噪声或缺失数据时可能不稳定；模型训练成本高；对不同采样率的兼容性和在更大多模态多病理场景下的泛化能力尚未充分验证。

---

## 274. Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency

**arXiv ID:** 2608.12939 | [PDF](https://arxiv.org/pdf/2608.12939v1)

**作者:** Guo An `[一作]` (Huawei), Qi Tian `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于动作条件下多步 roll‑out 的预测一致性诊断 (ACPC)，并结合 Invariance Radius (IR) 与 Separation Rate (SR) 对视觉扰动下的联合嵌入预测模型进行鲁棒性评估与检查点筛选。

**💡 创新点**

创新点在于：①证明 ACPC 能够界定扰动导致的多步预测误差与规划成本变化；②通过 IR 与 SR 的组合提供了可用于模型检查点的定量诊断屏幕；③将 ACPC 作为诊断工具在多种扰动下与不同模型架构（LeWM、PLDM）中验证其有效性。

**🔧 技术方法**

使用联合嵌入预测架构（JEPAs）中的编码器和动作条件预测器，采用加权轨迹距离、最小二乘回归、统计阈值等方法；并对多步 roll‑out 进行实验评估。

**📊 数据集**

实验基于四个视觉控制任务：TwoRoom、PushT、Reacher、OGBench-Cube（Cube），并在 LeWM 与 PLDM 两种世界模型上评估 Gaussian 噪声、模糊与缩放等视觉扰动。

**📈 对比分析**

通过比较 ACPC 与编码器距离、单步 ACPC 的误差预测，以及使用 ACPC 预测 CEM 计划偏差；在不同任务、模型和扰动条件下观察到 IR 低、SR 高的检查点往往对应更高的规划成功率；阈值筛选在未见任务上亦能识别出恢复性能，整体表现良好。

**⚠️ 局限性**

局限性包括：ACPC 仅为诊断工具，未直接提升规划效果；需依赖未扰动的参考检查点；IR 阈值仍处于测试边缘，未涵盖更广泛扰动类型；未展示将 ACPC 用于实际规划改进的实验，且对不同模型的泛化能力仍待进一步验证。

---

## 275. Decomposition of Evidence, Contradiction, and Fragility in Perturbation Responses

**arXiv ID:** 2608.12935 | [PDF](https://arxiv.org/pdf/2608.12935v1)

**作者:** Lei You `[一作]` `[通讯]` (Technical University of Denmark), Lei You (Technical University of Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

引入 DECAF 方法，利用逐步揭露输入的对比轨迹，将模型对比响应分解为证据（Evidence）、矛盾（Contradiction）和脆弱性（Fragility）三种语义成分，保留原始幅度且不增加额外模型查询；

**💡 创新点**

创新点在于使用最终对比作为语义参考，形成唯一无损的三元分解；通过“语义路由”将中间响应根据是否与最终方向一致而分类为证据、矛盾或脆弱性；该分解在保持总幅度的同时提供更细粒度、可解释的响应信息；

**🔧 技术方法**

使用的技术包括：黑盒 paired reveal 轨迹生成、阈值门控与方向性支持的无损分解公式、无梯度计算、以及对多种模型（CNN、ViT、树模型、远程 API）进行批量评估；

**📊 数据集**

实验数据集涵盖：控制实验的 3D Shapes、Covertype、ImageNet-9；公开数据的 ImageNet-1k、FunnyBirds、ImageNet-1k IDSDS；大规模实验在 DINOv2 ViT-g/14 上进行；

**📈 对比分析**

与梯度/路径（IG、SmoothGrad、Grad-CAM）、采样（RISE、KernelSHAP）等基线对比；在 ImageNet-9 上，DECAF 的证据/脆弱性指标与独立测得的背景依赖/敏感度高度相关，且相同幅度下正确识别行为的准确率约 96%；在 FunnyBirds 与 ImageNet-1k 上，DECAF 的 Spearman 相关性平均高于大多数通用归因方法；在 1B 参数 DINOv2 上，DECAF 在相似质量下实现 4.75 倍更短的壁时和 2.36 倍更低的峰值内存；

**⚠️ 局限性**

局限性包括：分解结果依赖于对照样本、阈值 ε 与揭露路径，无法直接揭示因果机制；在某些模型或任务中 fragility 可能与实际无意义的对比相混淆；需要手工选择阈值且对“无意义”对比的鲁棒性尚待进一步研究。

---

## 276. Multi-perspective Imbalance-Conscious 6G Beamforming Optimization and Performance

**arXiv ID:** 2608.12929 | [PDF](https://arxiv.org/pdf/2608.12929v1)

**作者:** Chukwunonso Henry Nwokoye `[一作]` (York University), Nnenna D. Duroha `[通讯]` (University of Hertfordshire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了基于机器学习的6G‑IoT 波束成形优化，比较网络、环境、设备和视觉特征组的预测能力，并使用阈值调优、集成学习、SHAP 解释以及无监督聚类揭示不同网络场景。

**💡 创新点**

首次将多视角特征组与不平衡处理、阈值调优、可解释性与聚类相结合，系统评估网络特征在波束成形成功中的主导作用，并在无监督层面揭示了环境与设备对聚类的决定性影响。

**🔧 技术方法**

监督模型（LR、XGBoost、Random Forest、SVM、MLP 等）、集成方法（投票、堆叠）、阈值调优、SMOTE、SHAP、Permutation Importance、K‑means/DBSCAN/层次聚类、PCA、SelectKBest 等技术。

**📊 数据集**

Kaggle 上的 6G IoT 智能管理数据集（约 1000 条记录，正例占比 16.8%），包含网络参数、环境因素、设备特征、视觉关键点以及后置性能指标。

**📈 对比分析**

采用 5‑折交叉验证、SMOTE 处理不平衡、阈值调优后对比各特征组模型的 F1、Recall、ROC‑AUC、PR‑AUC；网络特征组在阈值调优后 F1 最高达 0.30；性能指标组实现完美 1.00，证明目标泄露；无监督聚类显示环境与设备是关键聚类因素，得到四种情景。

**⚠️ 局限性**

正例样本极少导致模型整体性能受限；阈值调优提升召回伴随精度大幅下降，易产生误报；后置指标无法用于实时预测，存在目标泄露；聚类方法对参数敏感，DBSCAN 失效；实验仅在公开数据集上验证，缺乏真实网络场景的验证。

---

## 277. Impact of introducing "Informatics I" to the common university entrance examination in Japan: a longitudinal study on students' perceptions of their information-related knowledge and skills from 2006 to 2026

**arXiv ID:** 2608.12924 | [PDF](https://arxiv.org/pdf/2608.12924v1)

**作者:** Akimasa Morihata `[一作]` `[通讯]` (University of Tokyo), Akimasa Morihata (University of Tokyo)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2006–2026年东京大学一年级学生的自评问卷进行纵向分析，评估2013、2022年课程改革与2025年「信息学 I」入学考试对学生信息相关知识与技能认知的影响。

**💡 创新点**

首次将洗刷效应（washback）框架系统地应用于信息学教育评估，并通过对直接入学与间隔年学生（DE、GY）的对照，区分课程改革与考试制度变化的作用。

**🔧 技术方法**

采用描述性统计、卡方检验（Bonferroni校正）、Wilson置信区间与有限总体修正等常规统计方法，并结合洗刷效应理论进行解释。

**📊 数据集**

东京大学一年级学生的年度问卷数据（13个项目），涵盖“学习率”和“获得率”，共计约620,000名入学者中的约3100名一年级学生。

**📈 对比分析**

通过对不同年份（2016 vs 2025）和不同入学方式（DE vs GY vs 2024入学者）的比较，观察学习率与获得率的差异。结果显示：课程改革未产生显著断点，2025年入学考试导致CS项目学习率与获得率急剧上升，且获得率的显著提升被解释为认知标准转变，而非真实能力提升。

**⚠️ 局限性**

主要局限包括：自评数据缺乏客观能力测量；非回应偏倚与样本选择偏倚（东京大学学生及高学力学校）；问卷表述在2025年有所调整；作者教学角色可能导致分析偏差。

---

## 278. Discovering Efficient and Explainable Communication Topologies for LLM-based Multi-Agent Systems via Causal Inference

**arXiv ID:** 2608.12921 | [PDF](https://arxiv.org/pdf/2608.12921v1)

**作者:** Junzhi Li `[一作]` (University of Chinese Academy of Sciences), Chuxiong Sun `[通讯]` (Chinese Academy of Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了E2-Explainer，一种对LLM多智能体系统通信拓扑进行后置解释的框架，能从已优化的拓扑中提取保持任务性能的紧凑子图。

**💡 创新点**

创新点在于将Granger因果评估引入通信边的重要性评估，并将离线边掩蔽结果转化为可摊销的解释器，兼具可解释性与效率，同时实现跨生成器、跨规模的迁移。

**🔧 技术方法**

采用Granger风格因果归因、边掩蔽、结果语义熵、预算约束投影以及摊销子图解释器的图神经网络。

**📊 数据集**

在六大基准（AQuA、GSM8K、MultiArith、SVAMP、MMLU、HumanEval）上使用Qwen3-8B作为后端LLM进行评估。

**📈 对比分析**

与ARG-Designer、G-Designer、OFA-MAS、AgentPrune等拓扑优化器对比，E2-Explainer在保持或提升平均准确率的同时，平均减少20%–25%令牌消耗，显示出显著的性能-成本平衡。

**⚠️ 局限性**

局限性包括：离线边掩蔽评估仍需耗时、对稀疏奖励任务的信号依赖度较高、在某些基准上效果不如预期、以及对不同LLM模型的泛化性尚未完全验证。

---

## 279. NaviDC-OCR: Navigating Document Parsing Across Digital and Camera-Captured Documents

**arXiv ID:** 2608.12898 | [PDF](https://arxiv.org/pdf/2608.12898v1)

**作者:** Peng Cai `[一作]` (China Telecom Artificial Intelligence Technology Beijing Co Ltd), Hao Sun `[通讯]` (China Telecom Artificial Intelligence Technology Beijing Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出NaviDC-OCR统一框架，能够同时解析数字文档与摄像捕获文档，解决几何畸变、布局识别与高度结构化内容（表格、公式）推理等难题。

**💡 创新点**

创新点包括①将文档去畸变能力内嵌至Vision‑Language Model，采用点级与区域级变形感知学习；②自适应曲率引导Douglas–Peucker采样，实现细粒度布局建模；③内容‑结构解耦学习策略，对公式语法与表格拓扑进行中间推理；④多模型一致性投票与自评VLM，实现高质量伪标签构造与自动纠错。

**🔧 技术方法**

核心技术：基于Qwen2.5‑VL视觉编码与Qwen3‑0.6B语言模型的VLM；曲率引导采样（CGDP）；点级/区域级变形感知训练；内容‑结构解耦学习；强化学习（GRPO）与任务特定奖励；多模型一致性投票（MCV）与自评VLM（J_θ）。

**📊 数据集**

使用的主要数据集包括公开基准：OmniDocBench v1.6、Wild‑OmniDocBench v1.5、PureDocBench（Clean、Digital Degraded、Real Degraded）以及ICDAR 2026 Sci‑ImageMiner Scientific Figure‑to‑Table；同时构建大规模合成与伪标注数据。

**📈 对比分析**

与传统分离式VLM（PaddleOCR‑VL、MinerU2.5‑Pro、GLM‑OCR）和端到端VLM（OvisOCR2、HunyuanOCR、Logics‑Parsing）等模型对比，NaviDC-OCR在OmniDocBench overall 96.87、Wild‑OmniDocBench 88.53、PureDocBench 86.90，ICDAR 2026 Sci‑ImageMiner TEDS 17.23，均显著领先，验证其在多种文档解析任务中的强大性能。

**⚠️ 局限性**

局限性包括：模型规模约1.2B，推理速度相对较慢；仍需大量高质量训练数据与算力；在极端光照、强噪声或极度扭曲的摄像文档中布局识别仍可能出现误差；自评VLM误检率尚高于40%，对伪标签质量有一定依赖。

---

## 280. Agent Behavioral Contracts II: Certifying Compositional Reliability Without Assuming Independence

**arXiv ID:** 2608.12895 | [PDF](https://arxiv.org/pdf/2608.12895v1)

**作者:** Varun Pratap Bhardwaj `[一作]` (Qualixar), Arun Pratap Bhardwaj `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过预注册实验验证了多代理系统中相同模型的组件在执行时存在显著的协同失败；并在此基础上提出一种不依赖于独立性假设的、基于共执行矩阵的有限样本可靠性下界（moment‑set LP）以及一种任何停止时都保持有效的实时可信度检验（e‑process）。

**💡 创新点**

创新点主要包括：① 以模型共享为操纵变量的预注册因果实验，首次量化同模型双代理在18,000条确定性任务中90%共失败；② 在传统乘积式可靠度与弗雷谢德下界之间提出一种可计算且不需设定特定依赖结构的下界；③ 通过添加三阶共执行矩阵将可靠度下界提升至41.16%，比仅使用一阶和二阶矩时的24.55%提升近17个百分点；④ 在可变停止情形下实现了与 SPRT 等价的 anytime‑valid 证书，并证明其类型‑I 错误率不超过显著性水平。

**🔧 技术方法**

使用的技术包括：
- 传统多代理可靠性框架（ABC）与条件独立性假设C5；
- 共执行矩阵的统计推断（J、τ_a、ϕ、logOR、Q 等）；
- 基于 Clopper–Pearson 置信区间构建的矩阵盒子；
- 极值线性规划（LP）求解全域最优下界；
- e‑process 与投注策略实现的 anytime‑valid 检验；
- 引入预分配 Bonferroni 预算以保证边界单调性。

**📊 数据集**

数据集为 30,820 条任务，由 6 个种子生成器（零售与金融两大领域）产生，使用 8 种不同模型与 6 个供应商，且任务按固定 SHA‑256 哈希映射到任务集合，保证不同实验组独立。实验包括 3 种图拓扑（2‑agent handoff、parallel、2‑of‑3 quorum）和 3 种模型共享级别（同模型、同厂不同模型、不同模型）。

**📈 对比分析**

对比方法：
- 传统乘积式独立性下界；
- Fréchet–Hoeffding 下界；
- 拟合单因子高斯 Copula 并采用自举置信下界；
- 采用 Moment‑Set LP。
实验表明：
- 乘积式下界在存在正相关时低估可靠度（被证明为反保守）；
- Fréchet 下界往往为 0，信息不足；
- 拟合 Copula 的下界随着样本增大反而失效（覆盖率下降到 1%）；
- Moment‑Set LP 在同样样本下提供可靠度 41.16%，并且保持了 5% 的类型‑I 误差。

**⚠️ 局限性**

局限性：
- LP 的变量规模为 2^m，随着代理数 m 增大计算不可行；
- 假设所有聚合节点为确定性，无法处理 LLM 聚合器；
- 依赖于 i.i.d. 任务分布，虽然实验检验了序列自相关，但在严重设计效应下仍有不确定性；
- 可靠性下界的适用范围受制于实验中使用的模型版本和任务分布，一旦系统升级需重新验证；
- 只研究了同模型共享导致的正相关，未探讨负相关或更复杂的依赖结构。

---

## 281. Predictive Memory Localization: Forecasting Selective Intervention Paths from Internal Signals

**arXiv ID:** 2608.12892 | [PDF](https://arxiv.org/pdf/2608.12892v1)

**作者:** Jinhao Jing `[一作]` (Chinese University of Hong Kong), Qiannian Zhao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了预测记忆定位（Predictive Memory Localization, PML），通过在语言模型内部记录随机校准的干预路径，预测特定层、方向下的目标、邻域和能力影响以及“干净”增益，并基于此构建风险感知的强度选择策略。

**💡 创新点**

创新点在于：①将随机校准的测量网格路径作为目标、损伤与干净结果的预测对象；②证明低剂量（α=±0.1）干预响应是预测后续强度行为的主导信号；③将预测结果直接用于选择是否干预，从而显著降低密集扫描成本并提升效用。

**🔧 技术方法**

技术包括：随机控制方向、均值差分方向、线性/逻辑回归探针、递归特征机（RFM/AGOP）监督几何、低剂量响应测量、逻辑回归/随机森林预测模型、以及基于预测效用的稀疏强度选择策略。

**📊 数据集**

使用了 3000 条记录，来自 9 个公开数据集（MMLU-Pro、MMLU-Redux 2.0、AI2 ARC、OpenBookQA、SciQ、LiveBench、HellaSwag、QASC）和 14 个学术/常识/数学/推理领域。

**📈 对比分析**

与随机、均值差分、线性/逻辑探针等基线相比，RFM/AGOP 在第 7 层实现 13.1% 的目标提升和 12.3% 的干净提升；整体预测 AUROC 在 0.80–0.85 之间；强度选择策略在 100 条记录的验证上比固定强度提升 0.055 的效用，并将邻域损伤从 6.0% 降至 1.8%。

**⚠️ 局限性**

局限性包括：实验仅在 Qwen3-1.7B 的两层上进行，跨模型对比受限；随机校准阈值与自由生成的可靠性尚未充分验证；邻域与能力损伤差异未得到统计显著性；低剂量响应的可迁移性和阈值设定需进一步研究；选择策略仅在 100 条记录上做过试点，尚需在更大规模上验证。

---

## 282. When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory

**arXiv ID:** 2608.12888 | [PDF](https://arxiv.org/pdf/2608.12888v1)

**作者:** Ruizhe Li `[一作]` (University of Science and Technology of China), Weidong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ReFind，一个完全基于原始聊天记录的、由智能体控制的词汇检索系统，利用多轮查询和四项会话原生控制来实现高效的长期记忆检索。

**💡 创新点**

创新点在于不预先构建任何语义结构，而是将检索任务交给自适应的智能体，结合会话上下文扩展、时间过滤、会话级融合和已访问会话去重等控制，直接在原始记录上进行检索。

**🔧 技术方法**

使用技术包括：BM25 倒排索引、RRF 两层重新排序、ReAct 迭代检索循环、GPT‑4o‑mini / GPT‑5‑mini 控制器，以及基于时间戳和会话 ID 的上下文窗口扩展与去重。

**📊 数据集**

主要使用的数据集为 MemoryAgentBench（包含单/多跳 QA、事件排序、事实整合等 2,800 个问题）以及 LongMemEval‑S/M 子集，用于评估长期记忆检索和事实更新跟踪能力。

**📈 对比分析**

在与稀疏/密集 RAG、结构化记忆系统（GraphRAG、HippoRAG、RAPTOR 等）以及其他智能体检索系统的对比中，ReFind 在 GPT‑4o‑mini 框架下取得 58.2% 的平均准确率，领先所有基线；在 GPT‑5‑mini 版上对 LongMemEval‑S/M 的五次复测分别达到 93.2±3.3% 与 89.3±6.0%。

**⚠️ 局限性**

局限性包括：仅适用于基于词汇匹配的检索，对语义上不显式表述的知识或需要深层推理的任务表现有限；依赖多轮交互，可能在极长记录或实时约束下效率受限；若记录中缺乏可检索的表面词汇，性能可能下降。

---

## 283. ReflectFact: Self-Reflective Agents for Improving Comprehension and Reasoning in Multi-Hop Fact Verification

**arXiv ID:** 2608.12877 | [PDF](https://arxiv.org/pdf/2608.12877v1)

**作者:** Runze Zhao `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Dongyang Zhang `[通讯]` (Zhongguancun Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了ReflectFact自反思代理框架，用于多跳事实验证；

**💡 创新点**

其创新点在于引入Evidence-Drift Verification与Reasoning Reflection Verification两种自我检验步骤，解决代理方法中的目标冲突与知识冲突；

**🔧 技术方法**

使用的大技术包括大语言模型驱动的实体解析、语义分解、链式推理以及自我验证机制；

**📊 数据集**

实验数据集为HOVER和EX-FEVER的多跳事实验证数据集；

**📈 对比分析**

与十个基线（LLM、NLI、代理等）比较，在HOVER和EX-FEVER上Macro‑F1分别提升3.32%和2.78%，实现SOTA；

**⚠️ 局限性**

主要局限在于仍依赖LLM生成质量，可能出现逻辑错误或幻觉，且未解决检索误差和外部知识获取限制。

---

## 284. Towards Socially Compliant Navigation in Deep Reinforcement Learning via Proxemics-Based Reward Modeling

**arXiv ID:** 2608.12917 | [PDF](https://arxiv.org/pdf/2608.12917v1)

**作者:** Takieddine Soualhi `[一作]` (Inria), Laetitia Matignon `[通讯]` (Universite Lyon 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于Hall proxemics理论的连续舒适成本场，用于深度强化学习的社交导航奖励建模，鼓励机器人在满足任务效率的同时降低对人类个人空间的侵犯。

**💡 创新点**

创新点在于：①将人类个人空间建模为径向高斯混合场并在机器人视野内积分得到局部成本；②使用成本变化（ΔC）作为稠密奖励信号；③通过可调权重λ实现效率与社交合规的权衡。

**🔧 技术方法**

使用技术包括：深度强化学习（PPO），基于图注意力的AttnGraph与MultiSoc两种网络架构，CrowdNav仿真器，ORCA人类运动模型，以及自定义的proxemics奖励函数。

**📊 数据集**

数据集：在CrowdNav仿真环境中生成的两种场景（Circle Crossing与Corridor），人群规模从15人到更高密度不等；不使用公开真实人类数据。

**📈 对比分析**

与三种基线奖励（无奖励、基于距离的奖励、基于相对速度的奖励）以及两种导航方法比较，结果显示proxemics奖励在社交指标（SC、MD、TTC、Jerk）上显著提升，导航指标（成功率、碰撞率、行驶距离/时间）保持竞争力。

**⚠️ 局限性**

局限性包括：奖励仅为软约束，不能保证严格的个人空间遵守；在非常高密度或动态人群情况下仍可能出现效率下降；以及对奖励权重λ和视野参数的敏感性需要进一步自动化调优。

---

## 285. InFactPlanner: Planning Sustainable Geo-Distributed LLM Data Centers

**arXiv ID:** 2608.12915 | [PDF](https://arxiv.org/pdf/2608.12915v1)

**作者:** Nicoletta Tsiopani `[一作]` (University of Cyprus), Marios D. Dikaiakos `[通讯]` (University of Cyprus)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个基于工作负载跟踪的可持续性分析框架 InFactPlanner，用于评估分布式 LLM 推理的数据中心部署。

**💡 创新点**

首次在单一统一框架内同时考虑能耗、碳排放、水消耗、硬件选型、模型选择、可再生能源与碳感知路由等多维度因素，并提供可配置的硬件-模型性能剖面。

**🔧 技术方法**

采用 trace‑driven 仿真、可插拔的能耗/碳/水模型、Apache Ray 并行执行、YAML 配置和现有的功耗/性能基准模型。

**📊 数据集**

使用 Azure LLM 推理数据集、Open‑Meteo 天气数据、ENTSO‑E 和 ElectricityMaps 的碳强度数据以及参考系统实验的能耗/延迟数据进行验证。

**📈 对比分析**

通过与真实系统能耗/延迟基准对比误差 <10%，在单机上模拟 9.92M 个请求仅耗时 30 分钟，并在四个场景中展示硬件、可再生能源、跨国部署和碳感知路由的性能与环境差异。

**⚠️ 局限性**

局限在于抽象层级高，无法精确捕捉批处理、KV 缓存等细粒度效应；依赖外部数据和模型精度；不考虑实体碳与多余可再生电量的出口等细节。

---

## 286. Revisiting Overestimation Bias Problem of Q-learning: Settling Large Discrete Action Space via Action Intersection

**arXiv ID:** 2608.12912 | [PDF](https://arxiv.org/pdf/2608.12912v1)

**作者:** Pu Li `[一作]` (Chongqing Institute of Green and Intelligent Technology), Mingsheng Shang `[通讯]` (Chongqing Institute of Green and Intelligent Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对大动作空间下 Q‑learning 的高估偏差问题，提出动作交叉（Action Intersection）策略，构造半解耦的双 Q‑学习（AIDQ）以及其深度版本（AIDDQN）。

**💡 创新点**

创新点在于：①通过 top‑K 动作交叉将双 Q‑学习的过估计与欠估计平滑成连续可调的偏差范围；②在不需要引入额外网络或复杂机制的前提下，实现从负偏差到正偏差的细粒度控制；③提供理论证明与实验验证，证明存在合适的 top‑K 使估计无偏。

**🔧 技术方法**

技术方法：双 Q‑学习框架、top‑K 选取与交叉更新、ε‑greedy 探索、经验回放、目标网络、深度 Q‑网络 (DQN) 结构以及相应的训练细节。

**📊 数据集**

实验数据集：①离散表格环境——多臂赌博机（Arm 数量 20/40/60/80）；②深度强化学习环境——Pixelcopter、Asterix、Breakout、Seaquest、SpaceInvaders、Pong（在 Gymnasium/PyGame/MinAtar 中将动作空间按比例放大 20‑40 倍）。

**📈 对比分析**

与 8 个先进基线（Q‑learning、Double Q‑learning、Weighted Double Q‑learning、Averaged Q‑learning、Maxmin Q‑learning、EBQL、AC‑CDQ、Order Q‑learning）进行对比。AIDQ 在离散环境中无偏估计与基线差距最小，深度环境中平均回报提升幅度从 53%（Asterix）到 463%（Pixelcopter），整体表现显著优于所有基线。

**⚠️ 局限性**

局限性：① top‑K 参数需手动调节，缺乏自动化适配机制；②实验集中于离散且已放大至大规模的动作空间，未验证在连续动作或更高维度环境中的效果；③算法相较于纯 Double Q‑learning 引入了额外的 top‑K 计算，可能增加少量计算负担；④理论证明基于独立样本假设，实际经验回放中可能存在偏差。

---

## 287. Understanding Backdoor Vulnerabilities in Vertical Federated Learning: The Gap Between Research and Practice

**arXiv ID:** 2608.12962 | [PDF](https://arxiv.org/pdf/2608.12962v1)

**作者:** Ziqi Zhao `[一作]` (University of Hong Kong), Ka-Ho Chow `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性分析并验证垂直联邦学习（VFL）中后门攻击与防御的实用性缺口，提出实践导向的威胁模型、后门工作流以及统一评估基准BVBench。

**💡 创新点**

① 重构真实场景下的威胁模型和攻击流程；② 开源完整评估框架BVBench，包含多维度评价指标与统一实验设置；③ 通过BVBench验证现有攻击与防御在实际数据集和多目标、多攻击者环境下的表现，揭示其不足。

**🔧 技术方法**

基于PyTorch的VFL引擎实现；多种后门攻击（如BackSplitVFL、LFBA、VILLAIN等）和防御（如VFLIP、VFLMonitor、UBD、GBD等）；统一的实验脚本与评估菜谱；嵌入式隐蔽性与鲁棒性度量。

**📊 数据集**

CIFAR-10（对比基准）以及六个现实VFL数据集：Satellite、KUHAR、PTB-XL、Vehicle、NUSWIDE、以及跨模态/表格数据集，体现多模态与多方特征分布。

**📈 对比分析**

对照多维度指标（主任务准确率、攻击成功率、恢复率、隐蔽性AUC、计算/通信开销等）进行公平统一评测。结果显示：大多数攻击在真实数据上效能显著下降，防御多在保留主任务性能或恢复率上失效，且对多目标、多攻击者环境鲁棒性差。

**⚠️ 局限性**

① 现有方法仍依赖不现实的任务知识；② 攻击与防御对标签、模型配置高度敏感；③ 缺乏跨目标和多攻击者的稳定性与鲁棒性；④ 评测多维度不足，导致结果易受实验设置偏差；⑤ 尚未提出兼顾效能、隐蔽性与恢复的实用后门方案。

---

## 288. Moose: Latent concept learning with reasoning-shortcut awareness in $\mathcal{EL}^{++}$

**arXiv ID:** 2608.12961 | [PDF](https://arxiv.org/pdf/2608.12961v1)

**作者:** Olga Mashkova `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Moose框架，在OWL 2 EL本体下使用Sentential Decision Diagram实现可微分的加权模型计数，用于部分监督下的潜在概念学习。

**💡 创新点**

首次在OWL EL上实现完全形式化验证的知识编译与RS分析，并结合闭包增强的约束实现高效的推理与学习。

**🔧 技术方法**

利用ELK推理、Clark完成、SCC分解、SDD编译、可微分WMC层，并采用BEARS与NeSyDM等RS缓解策略。

**📊 数据集**

在MNIST‑with‑ontology（含三种监督模式）和Pizzaïolo（4,800张合成披萨图像）两个基准上进行评估。

**📈 对比分析**

与DeepProbLog、LTN、ELEmbeddings、OWL2Vec*等基线对比，Moose在关系和角色链场景下相较于Propositional NeSy提升十余个百分点，且在RS测试中表现出更优的准确率或校准性。

**⚠️ 局限性**

局限在于ABox规模有限（最多两名个体）、仅支持有限命名域、并假设角色在证据下被“钉住”，未解决大规模本体或开放域推理。

---

## 289. PSPACE-Completeness of Multi-Agent Path Finding for Large Agents

**arXiv ID:** 2608.12955 | [PDF](https://arxiv.org/pdf/2608.12955v1)

**作者:** Maichi Zhang `[一作]` (Kyoto University), Kanae Yoshiwatari `[通讯]` (Kyoto University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造从受限滑动代币（Restricted Sliding Tokens）到大体积多代理路径寻找（LA‑MAPF）的多项式时间归约，证明LA‑MAPF属于PSPACE完备类，并给出了在平面图和仅允许水平垂直移动的情形下仍保持PSPACE‑难度的证明。

**💡 创新点**

创新点在于：
• 设计了宏图（macro graph）与正交网格嵌入相结合的几何归约框架；
• 用常数规模的“门道”和“阻塞”顶点实现了对滑动代币独立集约束的几何模拟；
• 证明了即使在平面图、仅水平垂直移动、且所有代理为相同圆盘时，LA‑MAPF仍保持PSPACE‑难度。

**🔧 技术方法**

核心技术包括：
1. 对受限滑动代币实例的“宏化”与图卷缩；
2. 使用正交图绘制算法将宏图嵌入到整数网格并按比例缩放；
3. 构造局部几何图形（token三角形、token边、阻塞顶点）并通过圆盘冲突约束实现信息传递；
4. 通过“空洞”与代理的相对顺序保持，模拟滑动代币的合法移动。

**📊 数据集**

该工作为理论复杂度研究，未使用任何实验数据集；所有证明均基于构造性的多项式时间归约与几何配置。

**📈 对比分析**

论文并未进行实验或算法性能比较；其主要贡献是对LA‑MAPF复杂度的理论上界与下界的严谨证明，说明该问题在最坏情况下是PSPACE‑完备。

**⚠️ 局限性**

局限性：
• 只给出了复杂度分类，没有提出实际可行的求解算法或近似方法；
• 归约依赖于特定的几何构造，难以直接转化为可实现的求解工具；
• 结果仅适用于理论模型（圆盘代理、理想化欧氏空间），在实际机器人路径规划中可能需进一步简化或近似。

---

## 290. FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving

**arXiv ID:** 2608.12932 | [PDF](https://arxiv.org/pdf/2608.12932v1)

**作者:** Zekai Li `[一作]` (UC San Diego), Zhijian Liu `[通讯]` (UC San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套针对Vision‑Language‑Action（VLA）模型的算法‑系统协同优化框架，通过在编码、预填、解码和动作推理四个阶段分别采用流式推理、非自回归草稿、适应步长流匹配以及W4A8量化等技术，将10B参数模型从717 ms降低到151 ms。

**💡 创新点**

创新点在于将四个结构性瓶颈各自拆解并针对性设计轻量化快捷方式，再通过CUDA Graph和核融合实现系统级加速，首次实现VLA单GPU推理6.6 Hz的实时性能。

**🔧 技术方法**

采用了时序KV缓存复用、DFlash非自回归草稿解码、流动匹配的自适应步长缓存、W4A8量化、CUDA Graph编译和核融合等技术。

**📊 数据集**

在NVIDIA Autonomous Vehicle Dataset上训练并评估，包括开环minADE、闭环AlpaSim仿真。

**📈 对比分析**

与原始Alpamayo 1.5‑10B基线在同一RTX PRO 6000 GPU上对比，推理延迟从717 ms降至151 ms，控制频率提升4.7×（1.4 Hz→6.6 Hz），开环误差仅上升0.08 m，闭环碰撞率下降3%。

**⚠️ 局限性**

局限在于依赖高分辨率多视角视频，流式推理对分布漂移需额外微调，且在极端场景下自回归草稿可能接受率下降，导致速度提升不再明显。

---

## 291. Momentum as Residual-Driven Multiplier Correction for Deep Learning Optimization

**arXiv ID:** 2608.12925 | [PDF](https://arxiv.org/pdf/2608.12925v1)

**作者:** Zhixin Ren `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ADMM 启发的残差惩罚变量分裂框架 AIM，并基于该框架设计了新的优化器 RADAR。

**💡 创新点**

创新点在于将动量解释为残差驱动的乘子修正，分离了更新几何与加速形式，并将相对论自适应几何、分离残差修正和二阶动量过滤结合起来，形成一种结构保守且精确的优化策略。

**🔧 技术方法**

采用了 ADMM 残差惩罚分裂、相对论自适应几何（RAD）、二阶动量滤波以及方差受扰的 Lyapunov 漂移分析等技术。

**📊 数据集**

实验数据集包括 CIFAR‑10/100（ViT、ResNet‑50）、WikiText‑103/2（GPT‑2 预训练/微调）以及 MuJoCo 连续控制环境（HalfCheetah‑v4、Walker2d‑v4）。

**📈 对比分析**

在相同模型、学习率、批量大小等设置下，与 RAD、Adam、AdamW、NAdam 进行对比，RADAR 在图像分类、语言建模和强化学习任务中均取得最佳或次佳性能，平均提升约 1%–6%。

**⚠️ 局限性**

局限性在于仅验证了代表性优化器和标准随机收敛假设，未针对更大规模、多样化任务进行扩展，且对方差界和几何上界等假设仍有待放宽。

---

## 292. BoardroomAI: Dependency-Aware Human-Steerable Multi-Agent Deliberation through Evolving Decision Graphs

**arXiv ID:** 2608.13046 | [PDF](https://arxiv.org/pdf/2608.13046v1)

**作者:** Sanjeev Manivannan `[一作]` `[通讯]` (Indian Institute of Technology Madras), Sanjeev Manivannan (Indian Institute of Technology Madras)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可持续人类引导的决策图框架，支持在多智能体讨论中即时介入并触发局部修复。

**💡 创新点**

创新点在于：① 将决策过程建模为带类型的依赖图；② 规范化人类介入为图操作；③ 采用依赖感知的选择性专家修复；④ 引入决策足够闭包来保证修复后可合成完整决策。

**🔧 技术方法**

使用了带类型的有向多重图、依赖感知传播机制、基于集合覆盖的路由算法、可重放的图服务以及LLM驱动的专家代理。

**📊 数据集**

实验数据集为600条合成决策DAG干预案例和12个合成组织决策基准；所有数据均由作者自行生成并公开。

**📈 对比分析**

与直接更新、单一环境、可达性、全重启和oracle等基线比较，BoardroomAI在所有600例中实现100%召回，只需检查约14.6%节点；在12例实验中，选择性修复保留所有无关节点，重计算62.1%节点，但在50%案例因缺乏足够上下文而保留不决。

**⚠️ 局限性**

主要局限在于：全部为合成和原型级实验；未测试真实人类评估、依赖提取误差、模糊介入、模型不确定性和降级策略；因此对实际组织决策的有效性尚未验证。

---

## 293. P2Fusion: Prompt-based Progressive Infrared-Visible Image Fusion via Dual-Prior Distillation

**arXiv ID:** 2608.13045 | [PDF](https://arxiv.org/pdf/2608.13045v1)

**作者:** Yi Shi `[一作]` (Northwestern Polytechnical University), Dingwen Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 P²Fusion 框架，实现红外-可见图像融合，通过先验引导的动态提示蒸馏完成跨模态信息融合；

**💡 创新点**

创新点在于将热显像的热显著性和可见图像的空间质量作为自适应提示，利用 Teach‑to‑Fuse 机制和 Gated Dynamic Expert Recalibration（GDER）模块实现跨模态动态平衡，取代传统硬约束；

**🔧 技术方法**

采用双教师提示蒸馏、跨模态注意力、GDER 混合专家结构、感知损失（梯度、强度、SSIM）等深度学习技术；

**📊 数据集**

在 MSRS、M3FD、FMB、RoadScene、DroneVehicle 等公开基准数据集上训练和评估；

**📈 对比分析**

与 12 个当前 SOTA 模型对比，P²Fusion 在 5 个数据集上在 14/20 关键指标上取得最佳结果，在图像融合指标、目标检测和语义分割等下游任务上提升了多项指标（如 mAP+3.2%）；

**⚠️ 局限性**

对先验质量仍存在一定依赖，在极端极端场景下可能受限，且尚未完成实时或硬件边缘部署的评估。

---

## 294. From Local Mismatch to Global Impact: Optimizing Cache Reuse Policy for Efficient Diffusion

**arXiv ID:** 2608.13043 | [PDF](https://arxiv.org/pdf/2608.13043v1)

**作者:** Xichen Ye `[一作]` (Fudan University), Weizhong Zhang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Global-Impact Cache（GCache）框架，用以在扩散模型推理中改进缓存重用策略，从而在保持或提升生成质量的同时显著加速推理。

**💡 创新点**

创新点在于：① 通过理论误差传播上界揭示局部缓存误差对最终生成质量的非线性放大；② 将误差指数参数化为 Bernstein 多项式，构建双层优化（内层动态规划求最优缓存策略，外层贝叶斯优化学习误差权重）以实现全局影响的缓存决策。

**🔧 技术方法**

采用理论误差分析、Bernstein 多项式参数化、双层优化（内层动态规划、外层贝叶斯优化）以及缓存重用策略与扩散模型融合。

**📊 数据集**

使用 VBench 视频提示集、COCO 30000 图像提示集，并在 Open‑Sora、CogVideoX、Wan2.1、Flux‑dev 等四个 DiT‑based 扩散模型上进行实验。

**📈 对比分析**

通过与多种现有缓存加速基线（Δ‑DiT、T‑GATE、PAB、TeaCache、ERTACache 等）在相同加速比下比较速度提升（Speedup）与视觉质量（VBench、LPIPS、SSIM、PSNR），结果显示 GCache 在 2–3× 速度提升的同时显著降低 LPIPS、提升 SSIM/PSNR，优于所有基线。

**⚠️ 局限性**

局限性包括：需要在每个模型上单独贝叶斯优化误差权重，训练成本较高；误差上界在高度非凸模型下仍可能保守，导致部分潜在性能未被充分挖掘；目前仅在四个具体模型上验证，跨模型泛化能力尚需进一步评估。

---

## 295. On the global feature importance for interpretable and trustworthy heat demand forecasting

**arXiv ID:** 2608.13039 | [PDF](https://arxiv.org/pdf/2608.13039v1)

**作者:** Milan Zdravković `[一作]` `[通讯]` (University of Niš), Milan Zdravković (University of Niš)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过结合梯度提升模型和多种XAI方法（自解释特征重要性、Partial Dependence、Accumulated Local Effects、SHAP）评估并解释热负荷预测模型的全局特征重要性，以提升模型可解释性和信任度。

**💡 创新点**

创新点在于系统性比较并对齐模型内部（gain/cover/frequency）与外部（PD、ALE、SHAP）多层面特征重要性指标，且排除了会引入偏差的置换重要性与LIME，强调对特征相关性与非线性影响的正确处理。

**🔧 技术方法**

采用的技术包括 XGBoost 梯度提升回归、特征工程（时间延迟特征、温度延迟特征）、自解释特征重要性度量、Partial Dependence 与 ICE 图、Accumulated Local Effects、SHAP 计算及可视化。

**📊 数据集**

数据集来自17号配电站的 SCADA 系统，涵盖 2020‑2024 四个供热季的小时级热能输送量、温度等指标，并与气象站数据合并后进行预处理与特征构造。

**📈 对比分析**

通过对比模型内部的 gain/cover/frequency 与 PD、ALE、SHAP 的重要性排序与分布，验证了特征在训练集与测试集上的一致性与互补性；性能评估使用 MAE 作为指标，尽管未做超参数调优，但结果表明模型已能较好地捕捉热负荷规律。

**⚠️ 局限性**

主要局限包括：PD 仅考虑主效应，忽略特征交互；虽然排除置换重要性，但对高维特征仍可能存在解释冲突；数据仅来自单一配电站，缺乏跨站泛化验证；MAE 评估简单，未涵盖更全面的预测质量指标。

---

## 296. InterSAGE: The Secure and Verifiable Interoperability Protocol for An Internet of Agents

**arXiv ID:** 2608.13030 | [PDF](https://arxiv.org/pdf/2608.13030v1)

**作者:** Zhenhua Zou `[一作]`, Zhuotao Liu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个四层信任协议套件，提供持久身份、能力感知发现、信任协商和可追溯性等功能，作为互联网代理的基础安全层；

**💡 创新点**

创新点在于：①四维绑定的 Agent Identity Card（绑定开发者、代码包、运营者与部署上下文）②DID 绑定的可验证凭证实现能力感知发现③基于单调能力衰减与双层访问控制的协商机制④内核中继加密审计链实现可追溯性；

**🔧 技术方法**

使用去中心化标识符（DID）、可验证凭证（VC）、Ed25519 签名、TEE/硬件安全模块等加密技术，并兼容 MCP、A2A、ANP 等现有通信协议；

**📊 数据集**

该工作为定位性论文，未使用具体数据集；实验与性能评估计划留待后续实现；

**📈 对比分析**

通过与 50+ 相关协议（AgentMesh、AIP、HDP 等）在十三维度对比，证明在身份、发现、协商、可追溯性等方面实现全覆盖；性能评估未给出，但设计目标是低延迟、无链路锁定；

**⚠️ 局限性**

局限包括：未给出正式协议规范、形式化证明和性能基准；依赖中心化/联邦 Global Agent Registry，单点失效风险；与现有协议互操作需适配；隐私保护、动态更新等细粒度需求仍待进一步研究。

---

## 297. Why Do Prefetchers Fail? Let Agents Answer

**arXiv ID:** 2608.13027 | [PDF](https://arxiv.org/pdf/2608.13027v1)

**作者:** Xiangfeng Sun `[一作]` (Hong Kong University of Science and Technology), Yuan Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于性能异常驱动的自适应预取器研发流程，利用多 Agent 对已部署的预取器残留缺失进行诊断、最小化重现、生成专用子预取器，并通过残差门控、去重沙盒和精度抑制机制逐步集成到一个 17‑引擎的 RTL 级预取器中。

**💡 创新点**

①将硬件预取器改进流程完全自动化；②以测量残差为驱动的循环持续定位未覆盖的访问模式；③在集成层采用残差门控、共享去重与精度阈值保证子引擎互不干扰；④使用大型语言模型和进化搜索生成 RTL 实现。

**🔧 技术方法**

大语言模型（DeepSeek V4 Pro）进行故障诊断和代码生成；进化式代码生成搜索；周期准确的 ChampSim 仿真器；硬件日志、源代码与切片 trace 的多视角分析；残差门控、去重沙盒和精度抑制机制。

**📊 数据集**

SPEC CPU2006 与 SPEC CPU2017 组合的 41 条工作负载，其中 30 条用于训练、11 条（SPECspeed2017 内存密集子集）用于隐藏集测试。

**📈 对比分析**

在相同 110 KB 存储预算下，与无预取以及单一预取器 SPP、Berti、Pythia 进行比较；最终 17‑引擎预取器在隐藏集上 IPC 几何平均提升 61.1%，超过 Alecto 14.5%、Berti 21.6% 与 Pythia 23.6%，且覆盖率最高，面积仅 0.0347 mm²。

**⚠️ 局限性**

仍需昂贵的大模型推理成本（约 1.91 十亿 token，约 $41.66）；流程仅在离线仿真环境完成，缺乏实测硬件验证；生成的子引擎数量可扩展但会增加设计复杂度；对极端内存模式或新架构的迁移性尚未验证。

---

## 298. TIEM: Temporal Integration of Hypergraph Evidence and Skill Memory for Event-Driven Financial Forecasting

**arXiv ID:** 2608.13024 | [PDF](https://arxiv.org/pdf/2608.13024v1)

**作者:** Wenjin Liu `[一作]` (Nanyang Technological University), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个时间戳门控的事件驱动催化剂-结果预测框架 TIEM，集成了多层级事件-证据超图、案例技能记忆和异构证据经验融合。

**💡 创新点**

创新点在于三大模块协同：①使用事件-证据超图实现时间过滤的多级检索；②构建案例技能记忆以实现可迁移的结果导向技能；③通过异构证据经验融合实现多流信息一次性推理。

**🔧 技术方法**

利用检索增强生成（RAG）、记忆增强代理、超图检索、时间戳门控过滤、经验记忆路径推理与单调用融合等技术，构建完整的预测流水线。

**📊 数据集**

实验基于五个金融事件预测基准：Astock、FinPURE、CMIN-US、EDT、CSMD。

**📈 对比分析**

与直接提示、MemGPT/Mem0/A-MEM、Vanilla RAG、HippoRAG、GraphRAG、LightRAG、HyperGraphRAG 等方法对比，TIEM 在两大 LLM（DeepSeek-V4-Flash、GPT-5.4-mini）上均取得最高的 Acc/MCC/F1，并在跨数据集、跨时间窗口的转移实验中保持领先。

**⚠️ 局限性**

局限性包括：仍受训练数据泄漏与时间泄漏的影响；超大规模实时检索的计算成本较高；在极端市场波动或罕见事件下的鲁棒性尚待进一步验证。

---

## 299. Beyond Handcrafted Security: Towards Self-Evolving Defense for LLM Agents

**arXiv ID:** 2608.12977 | [PDF](https://arxiv.org/pdf/2608.12977v1)

**作者:** Jiajun Ruan `[一作]` (University of Minnesota), Chao Feng `[通讯]` (Ant Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于 harness 的自主运行时防御演化框架 HARD，能够根据 LLM 代理在执行过程中检测到的失败轨迹自动更新其上下文构造与动作解释组件，以提升安全性。

**💡 创新点**

①将运行时防御统一建模为对 agent harness 的优化问题；②设计了失败驱动的轨迹路由与组件细粒度演化机制，使防御能够自适应新出现的攻击；③首次在保持模型不变的前提下实现运行时防御的自演化，弥补了传统静态手工防御的局限。

**🔧 技术方法**

使用 LLM 生成的演化器（如 GLM‑5.2、Claude‑Opus‑4.6 等）对失败轨迹进行分析；实现轨迹路由器将错误定位到上下文或动作防御模块；采用基于安全/效用阈值的失败识别与 harness 迭代更新；评估时使用安全判别器 GLM‑5 与 PinchBench Python 验证器。

**📊 数据集**

AgentCanary 基准（涵盖 4 类攻击），AgentHazard（转换为 AgentCanary 任务），PinchBench（测量正常任务效用），以及自定义的动态攻击演化（DAE）和长周期进化攻击（LPA）数据集。

**📈 对比分析**

与无防御基线及三种手工构造的运行时防御（SecureClaw、ClawKeeper、OpenClaw Shield）进行对比，使用 ASR、BU、UA 三项指标。实验显示 HARD 在 4 种静态攻击下将 ASR 降至 1.0%–15.4%，同时保持 BU 91–95%；在自适应攻击中也优于手工防御（如 DAE 下 26.5% vs 30.1%，LPA 下 4.8%–12.1%）。ablation 结果表明联合演化（HARD‑Both）效果最好。

**⚠️ 局限性**

依赖于 LLM 演化器的质量与可训练数据，演化过程可能导致过度约束导致效用下降；轨迹路由和模块化演化仍需手工设定防御接口；对极端或组合攻击的覆盖有限；在大规模代理或多语言环境下的可扩展性待进一步验证。

---

## 300. Learning Unified Video and Image Representation for Video Face Forgery Detection

**arXiv ID:** 2608.13064 | [PDF](https://arxiv.org/pdf/2608.13064v1)

**作者:** Haotian Liu `[一作]` (University of Oulu), Xiaobai Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了UVIF框架，通过统一编码器和多任务学习，将面部视频与图像共同训练，实现对部分伪造视频的精细检测；

**💡 创新点**

创新点在于利用已有的面部图像标注提供帧级监督，结合伪标签生成与视频导向的特征对齐，弥补视频级标签噪声，显著提升部分伪造视频识别性能；

**🔧 技术方法**

使用统一的2D Backbone（如Swin、ResNet、ConvNeXt）+Temporal Fusion模块，伪标签（weak‑strong数据增强）生成，特征混合对齐（feature mixup）和多任务损失；

**📊 数据集**

主要数据集为ForgeryNet（220k+视频、2.9M图像）和DFDC preview（1131真/4113伪造视频），采用随机抽取10k–100k图像训练；

**📈 对比分析**

与现有图像基、视频基、MIL等方法对比，UVIF在ForgeryNet上Acc达88.94%、AUC 95.69%（Swin‑S），在DFDC上Acc 90.73%、AUC 96.05%，均明显优于基线和SOTA；

**⚠️ 局限性**

局限性：依赖足量且标注完善的图像数据，对图像与视频伪造类型不重叠时效果下降；训练阶段需要额外伪标签与对齐计算，且对未见伪造手段的泛化仍待验证。

---

## 301. VALG: An Agentic System for ML Theory Research

**arXiv ID:** 2608.13060 | [PDF](https://arxiv.org/pdf/2608.13060v1)

**作者:** Dechen Zhang `[一作]` (University of Hong Kong), Difan Zou `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个自动化的ML理论研究代理系统VALG-ML-Theory-Agent，能够在问题设置、定理表述和证明开发之间进行自适应迭代，并在九个COLT 2026开放问题上生成22个定理候选。

**💡 创新点**

将理论研究过程拆解为前置问题构造与后置证明审核两阶段，采用多层验证、适应性问题表述和图结构证明依赖，并引入诊断与修订层级以精确定位失败原因，形成全新的“源相对定理分支”框架。

**🔧 技术方法**

结合大型语言模型（如GPT‑5.6‑sol）进行自然语言推理，自动生成证明草图、全局依赖图检查、局部证明步骤生成、以及人机交互式审计与修订；同时实现多层验证机制与路径修订循环。

**📊 数据集**

以COLT 2026开放问题的子问题描述与公开论文为实验基础，未使用传统标注数据集，系统以文本形式输入问题与背景。

**📈 对比分析**

与传统人工证明或半自动化证明工具相比，VALG在九个子问题中产生22个定理候选，其中两项完全匹配源问题，其余为限制性或条件结果，表明系统在自动化理论研究上的可行性和初步竞争力。

**⚠️ 局限性**

系统仍无法在所有子问题上得到完整解答，部分结果仅在有限假设下成立；依赖语言模型推理质量；缺乏对证明正确性的完全形式化验证；对问题表述与结果关联的确认仍需人工干预。

---

## 302. Operationalizing Cyber Threat Intelligence with GraphRAG

**arXiv ID:** 2608.13050 | [PDF](https://arxiv.org/pdf/2608.13050v1)

**作者:** Atul Kabra `[一作]` (IIT Bombay), Manjesh K. Hanawal `[通讯]` (IIT Bombay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了将CTI报告转换为威胁猎捕计划的自动化方法，并比较了基于知识图谱检索（GraphRAG）与传统向量检索（Naive RAG）的效果，重点评估计划在IOC旋转后能否持续生效。

**💡 创新点**

创新点在于提出并验证了通过知识图谱检索可显著提升猎捕计划的Pyramid‑of‑Pain层级，从而使检测规则对IOC变化的鲁棒性更强；并构建了可在本地完整运行的全流程评估框架。

**🔧 技术方法**

使用技术包括：Microsoft GraphRAG（知识图谱构建与检索）、向量检索（LanceDB）、LLM生成（gpt‑oss‑20b）、LLM评判（Foundation‑Sec‑8B‑Instruct）以及自定义的评估指标（两层评分体系）。

**📊 数据集**

数据集为九份真实CTI报告（来自CrowdStrike、Cyble、EclecticIQ、Mandiant及一份公开报告），并对单个APT28报告做了深度实验。

**📈 对比分析**

比较方法采用相同生成模型与评分模型，只变更检索后端，并通过两层评分（基础质量与Pyramid‑of‑Pain韧性）评估。实验显示：GraphRAG方案在单个报告上可使100%检测在IOC旋转后继续生效，而Naive RAG仅保留约29%；在九份报告的宽度实验中，GraphRAG在Pyramid层级分布和IOC耐久度上均优于Naive RAG，虽然整体分数相近。

**⚠️ 局限性**

局限性包括：GraphRAG Global在短报告或实体稀疏时会静默失败；LLM评判模型（8B）存在JSON解析错误和数值不精确；实验规模仅九份报告，缺乏统计显著性；并且结果对Prompt工程的依赖性较大。

---

## 303. Topology-Unified 2D Pose Estimation across Intact, Residual and Prosthetic Limbs

**arXiv ID:** 2608.13047 | [PDF](https://arxiv.org/pdf/2608.13047v1)

**作者:** Tianye Qi `[一作]` (University of Queensland), Xin Yu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ProPose基准、Omni-Pose协议和ProLoss结构化损失，用于完整估计完整、残缺与义肢的人体姿势。

**💡 创新点**

创新点在于统一三种肢体类型的拓扑表示、通过Real-to-Synthetic扩展解决义肢稀缺问题，以及结构感知的ProLoss抑制不合理关节。

**🔧 技术方法**

采用多模态生成与指令编辑的Nano Banana/Gemini 进行数据扩增；使用 Transformer/ Swin/ YOLO‑Pose/ RTMPose 等现有网络并加入 ProLoss；结合语义分类权重和生物对比损失。

**📊 数据集**

主要使用 ProPose 数据集（36k 图像 + 9.5k 合成），并在其上对比预训练模型；还在 ProGait 视频集做跨域评测。

**📈 对比分析**

与多种基线模型对比，ProLoss 提升 AP 至约 90% 并显著提高义肢与缺失关节的分类准确率（提升 5‑8%），但对稀有关节仍有限提升。

**⚠️ 局限性**

仍受义肢种类不足、生成图像域差距和长尾不平衡的限制，难以覆盖所有专业义肢形态，且对新型义肢的泛化仍有待验证。

---

## 304. Spatially-Grounded Text-to-Video Generation via Inference-Time Gradient-Free Optimization

**arXiv ID:** 2608.13037 | [PDF](https://arxiv.org/pdf/2608.13037v1)

**作者:** Guillaume Jeanneret `[一作]` (Sorbonne Université), Matthieu Cord `[通讯]` (Sorbonne Université)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的训练和无梯度优化方法，用于空间定位的文本到视频生成，旨在提高生成视频中对象的定位准确性。

**💡 创新点**

创新点在于引入了一种解析的替代目标函数，避免了反向传播的计算开销，并通过几何感知的查询注入机制实现了高效的空间控制。

**🔧 技术方法**

使用了无梯度的解析轨迹优化方法，结合了交叉注意力机制来实现空间引导。

**📊 数据集**

使用了两个专门构建的评估数据集，分别包含400个生成视频，涵盖多种对象和运动模式。

**📈 对比分析**

与现有的基线方法（如Peekaboo、VideoTetris和SwitchCraft）进行比较，结果显示该方法在定位准确性上显著优于基线，同时在计算开销上几乎没有增加。

**⚠️ 局限性**

局限性在于，尽管该方法在空间控制上表现出色，但可能会导致场景动态性和美学质量的下降。

---

## 305. RGB-D Video Generation for Improving Human-to-Robot Object Handover Prediction

**arXiv ID:** 2608.13028 | [PDF](https://arxiv.org/pdf/2608.13028v1)

**作者:** Tianyu Sun `[一作]` (Nanyang Technological University), Guosheng Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过构建 Hand2Bot RGB‑D 数据集并提出 PassGen 生成器，解决人机交互中缺乏大规模真实手交数据和仿真到现实差距的问题，并结合意图门控机制实现更安全、更早的手交意图识别与机器人抓取。

**💡 创新点**

创新点包括：①基于稳定视频扩散的 PassGen，集成面部时序编码（TFE）以保留高频面部微表情与注视信息；②形态深度噪声仿真策略，重现 Intel RealSense L515 的边缘散射与 void 噪声；③全身视角与丰富的意图/把握标注，显著提升意图预测的鲁棒性；④独立的意图门控（IG）机制，将面部注视与对象接近速度融合，抑制误触发。

**🔧 技术方法**

使用技术：Stable Video Diffusion (SVD) + LoRA 微调；Temporal Face Encoder (TFE) 结合 ArcFace、Temporal Attention；PoseNet 与 ReferenceNet 进行姿态与身份编码；形态深度噪声仿真；Intention Gating (IG) 通过 gaze 与 depth 速度融合；GraspNet 生成 6‑DoF 抓取候选；物理机器人部署在 UR5e 上。

**📊 数据集**

使用数据集：Hand2Bot（5000 RGB‑D 对，2125 实测，2875 生成），以及真实 L515 深度记录；对比手交基准 HandoverSim、GenH2R、DexH2R、HOH 等，强调 Hand2Bot 在噪声、视角与标注完整性上的优势。

**📈 对比分析**

性能对比：在动画质量上，PassGen 在 PSNR 25.12、SSIM 0.909、LPIPS 0.212、FID‑VID 91.32、FVD 337.59 上均优于 StableAnimator、MimicMotion 等；在意图门控上，加入深度与面部信息后 FPR 下降至 13.6%，平均准确率 90%；在真实机器人实验中，ISR 达到 54/60、FTR 仅 2/30，且与仅使用真实数据（ISR 6/10、FPR 22.8%）相比，加入合成数据后准确率提升至 90%、FPR 降至 13.6%。

**⚠️ 局限性**

局限性：生成过程中采用静态背景拼接，限制了环境多样性；未开展主观用户体验评估；对不规则几何物体的抓取仍存在困难；形态噪声仿真为经验式方法，可能无法完全覆盖所有传感器噪声；合成数据可能引入偏差，需进一步验证在更大规模真实部署中的鲁棒性。

---

## 306. Structure-aware Riemannian Growth Fields for 4D Plant Modeling

**arXiv ID:** 2608.13007 | [PDF](https://arxiv.org/pdf/2608.13007v1)

**作者:** Meng-Yu Jennifer Kuo `[一作]` (Nara Women's University), Ryo Kawahara `[通讯]` (Kyoto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于结构感知黎曼场的连续几何-拓扑演化模型，实现从稀疏时间点恢复完整4D植物生长。

**💡 创新点**

将符号L系统规则映射为连续黎曼流，并结合Sigmoid生长调制，既保留拓扑一致性，又实现稳健时空对应。

**🔧 技术方法**

使用连续黎曼几何场、时间依赖L系统、遗传算法优化、3D Gaussian Splatting、SAM分割、Laplacian收缩骨架等技术。

**📊 数据集**

自建10天双物种（黄豆/西兰花）稠密几何+语义标注数据集，以及公开Pheno4D番茄数据。

**📈 对比分析**

与骨架跟踪(Khanam)、全局配准(Pan)及全局表面拟合方法对比，采用Chamfer距离和Dice指标，结果在骨架和整体几何上分别提升约70%/80%，对应误差显著低于基线。

**⚠️ 局限性**

仍受自遮挡与重建噪声影响，未建模细节表面展开与光照变化，难以捕捉极细枝叶的微观形变。

---

## 307. HybridRAG-BN: A Retrieval-Augmented Framework with Fine-Tuned Verification for Bangla KBQA

**arXiv ID:** 2608.13004 | [PDF](https://arxiv.org/pdf/2608.13004v1)

**作者:** Rathijit Aich `[一作]` (Chittagong University of Engineering & Technology), Mahfuzulhoq Chowdhury `[通讯]` (Chittagong University of Engineering & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 HybridRAG-BN，一个面向孟加拉语知识库问答的检索增强框架，结合双路检索、Gemma-4-31B-Instruct 生成和 LoRA 微调的答案验证，并通过后处理提升答案准确率。

**💡 创新点**

创新点包括：① 将 BM25 与 BGE-M3 语义检索融合并通过 BGE-Reranker 进行交叉编码重排序；② 对生成的答案使用 LoRA 微调的 Gemma 模型进行验证与修正；③ 采用两种知识库预处理策略与 DuckDuckGo 辅助检索的后处理流程，显著提升覆盖率和鲁棒性。

**🔧 技术方法**

使用技术包括：BM25、BGE-M3、BGE-Reranker、FAISS 向量索引、Gemma-4-31B-Instruct（GGUF）、LoRA 微调、检索重试策略、Prompt 设计、DuckDuckGo 搜索。

**📊 数据集**

使用的数据集为 Indic-RAG-Suite（约 3,000 条训练问答三元组、1,500 条测试问答）以及约 6,500 篇孟加拉语 Wikipedia 页面构成的知识库。

**📈 对比分析**

通过公开和私有 leaderboard 的 token‑level F1 进行评估，并与单纯检索生成（Approach 1/2）以及不同规模 Gemma‑4 系列模型进行对比。最终系统在公开榜 0.71654、私有榜 0.72912，获得第一名。

**⚠️ 局限性**

局限性：① 30 词的生成长度限制导致列表类答案被截断；② 计算成本高，整个流程均依赖 Gemma‑4‑31B‑Instruct；③ 仅在 Gemma‑4 系列验证，未探究对其他模型家族的通用性。

---

## 308. Balanced Adaptive Prototype Selection for Scalable TabPFN Inference on Large-Scale Tabular Data

**arXiv ID:** 2608.12989 | [PDF](https://arxiv.org/pdf/2608.12989v1)

**作者:** Mahboobe Jadid `[一作]` (Islamic Azad University), Ali Mousavi `[通讯]` (Islamic Azad University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Balanced Adaptive Prototype Selection（BAPS）框架，构造紧凑且信息丰富的上下文，以实现预训练Tabular Foundation Model（TabPFN）在百万级数据集上的可扩展推理；

**💡 创新点**

BAPS通过联合保持代表性结构、决策边界、局部密度、类别平衡和特征空间多样性，实现信息保留的上下文构造，而无需修改或再训练模型；

**🔧 技术方法**

采用多阶段候选生成（代表性、边界、密度）与多样性约束（去冗余、覆盖特征空间）的组合优化方法；

**📊 数据集**

在HIGGS和SUSY两个百万行分类数据集上评估，并与Covertype、Electricity、Diabetes等小规模数据集对比；

**📈 对比分析**

与随机采样、KMeans中心/medoid、BAPS单上下文等基线比较，BAPS Ensemble在512原型预算下在两大数据集上均获得最佳或竞争性平衡准确率、宏F1、ROC‑AUC，并保持良好校准（ECE）和可接受的推理时间；

**⚠️ 局限性**

限制在于需要一次性构造上下文的额外计算成本；多上下文聚合虽然提升稳定性和校准，但会按上下文数成比例增加推理时间；

---

## 309. Generative Universal Multimodal Retrieval with Dual-role Identifiers

**arXiv ID:** 2608.12987 | [PDF](https://arxiv.org/pdf/2608.12987v1)

**作者:** Kaipeng Li `[一作]` (Independent Researcher), Xuanchen Zhou `[通讯]` (University of Tsukuba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种DrIG框架，用残差量化的双角色标识符实现通用多模态检索，既可做自回归生成，又可作为无序集用于全局相关性预估；

**💡 创新点**

创新点在于同一标识符的双重角色（顺序生成与无序集匹配）以及双重引导的受限束搜索，解决传统左到右解码的前缀错误与局部最优问题；

**🔧 技术方法**

技术组合包括大型多模态模型（LMM）进行语义编码、残差量化构造标识符、对比学习与MSE保证语义一致、Trie约束束搜索、基于集合的全局相关性评分、查询–目标插值增强、判别式排名损失以及稠密向量再排序；

**📊 数据集**

数据集涵盖M‑BEIR基准（包括VisualNews、WebQA、EDIS、NIGHTS、OVEN、InfoSeek、FashionIQ、CIRR等10个子任务）以及Flickr30K和MSCOCO的文本‑图像检索；

**📈 对比分析**

与CLIP‑SF、BLIP‑FF、LamRA等稠密检索基线和GENIUS等生成式检索基线对比，DrIG在局部池检索平均Recall提升至38.0（↑28.8%）或36.4（↑27.3%）；再加稠密再排序后平均Recall可达48.7/50.4，明显优于生成式基线并接近稠密检索性能；

**⚠️ 局限性**

局限包括：与最强稠密/LMM检索仍有差距，尤其在文本知识检索；离散化标识符会丢失细粒度语义；当前训练流程分阶段，缺乏端到端统一优化；尚未支持动态候选集合的高效增删；对细粒度跨模态推理的处理仍有提升空间。

---

## 310. Tracing Methamphetamine abuse in under-treatment drivers: How biomechanical and oculomotor features help detect at-risk drivers?

**arXiv ID:** 2608.13054 | [PDF](https://arxiv.org/pdf/2608.13054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 311. Learning the Mathematical Property for Designing Low Mutual Coherence Binary Sensing Matrices

**arXiv ID:** 2608.12982 | [PDF](https://arxiv.org/pdf/2608.12982v1)

**作者:** Rekha `[一作]` (Shiv Nadar Institution of Eminence), S. K. Neogy `[通讯]` (Indian Statistical Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了一种基于互相不干扰（mutual incoherence）属性的学习框架，用来生成低互信息的二进制感知矩阵。

**💡 创新点**

创新点在于：①使用互相不干扰属性直接作为损失函数，无需大量训练数据；②采用共享参数的轻量级全连接网络生成整个矩阵；③结合Lp范数、LogSumExp和平面帧（ETF）三种可微损失实现最大互信息最小化。

**🔧 技术方法**

技术手段包括：简单的两层全连接神经网络、Straight-Through Estimator（STE）实现二值化、联合损失函数（Lp、LogSumExp、tight-frame）以及Adam优化器。

**📊 数据集**

实验使用随机高斯分布生成的潜在向量作为输入，无需任何传统图像或信号数据集。

**📈 对比分析**

通过与高斯随机矩阵和伯努利随机矩阵进行比较，结果显示所学习的矩阵在最大互信息、平均互信息和总互信息方面均低约30-50%，并在不同尺寸下保持优越的共线性结构。

**⚠️ 局限性**

局限性：仅验证了互信息指标，对RIP、spark等更严格的压缩感知理论性质未作评估；仅针对二进制矩阵，且缺乏对实际重构实验的深入验证。

---

## 312. Computing Fixed Points using Dependency Oracles

**arXiv ID:** 2608.13020 | [PDF](https://arxiv.org/pdf/2608.13020v1)

**作者:** Giorgio Bacci `[一作]` (Aalborg University), Daniele Toller `[通讯]` (Aalborg University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Noetherian偏序下，提出了全局(GlobalK)和局部(LocalK)两种固定点迭代算法，利用依赖oracle实现目标变量的局部求解。

**💡 创新点**

创新点在于：①引入可定制、可组合的依赖oracle框架；②使用now/flow关系对依赖进行语义/句法分析；③构造了可局部化的oracle与扩展，形成统一的可组合理论；④展示了oracle组合能在保持正确性的前提下提升精度。

**🔧 技术方法**

技术手段包括：偏序理论与Kleene迭代、变量依赖关系(now/flow)、依赖oracle与扩展、Boolean代数优化、局部化探索策略、Java实现与实验评测。

**📊 数据集**

实验数据集包括：CCS弱同构的Bisimulation检查、带权CTL的WCTL模型检测、随机生成的3000个方程的Boolean方程系统、Leader Election、Alternating Bit Protocol等经典验证基准。

**📈 对比分析**

与ADG、CAAL、WKTool等现有工具对比，使用max、smax、bool等oracle，实验表明在大多数基准上速度提升可达20倍（甚至300%），在局部探索时显著减少变量生成；对比实验中记录了时间、迭代次数、生成和循环开销。

**⚠️ 局限性**

局限性：①oracle计算开销与精度权衡需手工调节；②在某些Bisimilar-ABP等基准中，oracle无法有效剪枝导致大量方程生成；③当前框架仅适用于Noetherian偏序，尚未推广至连续域；④实现为原型，仍有实现层面可优化的空间。

---

## 313. Uniform Herding: Exemplar Replay with Representation Refresh

**arXiv ID:** 2608.13061 | [PDF](https://arxiv.org/pdf/2608.13061v1)

**作者:** Krishna Subedi `[一作]` `[通讯]`, Krishna Subedi

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CIFAR‑100上对10个任务（10类）进行增量学习实验，比较了不同策略（B0~B3、a1–a4、s1–s4）的效果。

**💡 创新点**

创新点在于为不同任务规模设计了可变的样本预算、检索预算以及知识蒸馏权重，并引入余弦边距（cosine_margin）头来提升类别区分度。

**🔧 技术方法**

技术手段包括ResNet‑18骨干网络、SGD优化、混合精度训练、余弦边距/线性分类头、经验回放、检索机制以及知识蒸馏。

**📊 数据集**

使用CIFAR‑100数据集，按10个任务×10类划分，随机种子13，probe/val分别占30/20。

**📈 对比分析**

通过NME、head‑logit和直接分类预测等评估指标与B0/B1/B2/B3等基线进行对比，实验显示加入余弦边距头、适当的蒸馏与检索策略能显著提升最终准确率。

**⚠️ 局限性**

局限在于仅验证于CIFAR‑100，网络规模较小，未覆盖更大规模数据或更复杂模型，且对不同硬件环境的泛化性尚未探究。

---

## 314. DMDIntel: Interpreting Large Language Models via Dynamic Mode Decomposition

**arXiv ID:** 2608.13048 | [PDF](https://arxiv.org/pdf/2608.13048v1)

**作者:** Amogh Joshi `[一作]` (Indian Institute of Technology Kharagpur), Sergey Utyuzhnikov `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于动态模式分解(DMD)的输入归因框架，利用监督微调LLM的隐藏状态序列构建低维时空模式并通过投影量度 token 的重要性，从而解释分类预测结果。

**💡 创新点**

将LLM视为离散时间动力学系统，首次在隐藏状态上应用 DMD 与高阶 DMD 提取可解释的低维模式，并以投影幅度对 token 进行排名，突破传统梯度或 PCA 方法的碎片化限制。

**🔧 技术方法**

核心技术包括动态模式分解(DMD)及其高阶变体(HODMD)、投影与幅度排序、PyDMD实现、MLP 输出去偏差、层选择启发式（基于指令偏差的余弦相似度阈值）。

**📊 数据集**

使用三大文本分类数据集：情感分类、假新闻检测与仇恨言论检测，并在三种不同参数规模的 LLM 家族上进行实验。

**📈 对比分析**

与 PCA、Integrated Gradients、GradientSHAP、Attention 等基线方法对比，在平均匹配计数、Rank‑Biased Overlap (RBO)、Recall@k、mask‑token 造成的准确率与置信度下降等指标上均取得更高或相近的表现，显示出更高的匹配度与鲁棒性。

**⚠️ 局限性**

仅在单输出 token 的分类任务中验证，未处理多 token 生成任务；仅针对监督微调的模型，未涵盖通用指令调优模型；在跨语言生成任务中的适用性与层选择稳定性尚待进一步研究。

---

## 315. InSPECtor: Improving SLEIGH Processor Specification Veracity via Proxy

**arXiv ID:** 2608.13042 | [PDF](https://arxiv.org/pdf/2608.13042v1)

**作者:** Michael Chesser `[一作]` (Adelaide University), Damith C. Ranasinghe `[通讯]` (Adelaide University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于 SLEIGH 语言的系统化验证框架 InSPECtor，能够自动枚举可解码指令、生成对应的初始状态，并通过差分测试（在 SLEIGH 驱动的仿真器与真实硬件之间对比）发现并定位指令规范中的错误。

**💡 创新点**

创新点包括：① 通过对 SLEIGH 语法树进行符号遍历，将解码规则转化为约束条件，系统地生成所有唯一语义的指令；② 利用 Pcode 语义的边界案例生成对应的输入状态；③ 采用硬件作为 oracle 的差分测试方法，既提升了检测覆盖率，又保持了测试精确性；④ 对多种 ISA 的 5 个 SLEIGH 规范进行大规模实测，揭示了 16 个根本性缺陷并给出了修复建议。

**🔧 技术方法**

技术主要包括：符号约束求解（Z3）、SLEIGH 解析与遍历、Pcode 边界案例提取、自动化差分测试框架（与 QEMU/KVM/硬件调试器交互）。

**📊 数据集**

数据集涵盖了 5 个公开的 SLEIGH 规范（ARM, MIPS, RISC‑V, PowerPC, AArch64 等），通过 InSPECtor 生成了超过 500,000 条差分测试用例，覆盖了所有可解码指令和主要操作模式。

**📈 对比分析**

与随机指令+随机状态的基线对比，InSPECtor 在构造器覆盖率、缺失构造器数和发现的 bug 数上显著优于基线（覆盖率 > 90%，bug 发现率 > 95%），且测试时间仅为基线的 1/20 左右，表明方法在效率和精确性上都有显著提升。

**⚠️ 局限性**

局限性：① 只测试单条指令，无法覆盖涉及指令序列或控制流变更的情况；② 对缺失的指令（规范中未描述的编码）无法检测；③ 依赖 Pcode 语义模型，对 Pcode 本身的不足（如浮点、SIMD 支持不足）会导致无法修复的差分；④ 需要手工分析差分结果，仍有一定人工成本。

---

## 316. UniTraffic-Agent: Unified Traffic Video Reasoning for AI City Challenge 2026 Track 3 with Two Out-of-Domain Evaluations

**arXiv ID:** 2608.13031 | [PDF](https://arxiv.org/pdf/2608.13031v1)

**作者:** Peng Li `[一作]` (University of Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的交通视频推理框架 UniTraffic-Agent，支持多摄像头和多任务（TAR、FETV、PSI‑VQA）

**💡 创新点**

创新点在于观测‑推理‑执行‑验证的端到端工作流、时间戳感知帧采样、视频级联合推理以及任务特定行动适配器

**🔧 技术方法**

使用多模态大型语言模型（如 GPT‑4V/ChatGPT）与帧采样、缓存、验证恢复、任务适配器等技术

**📊 数据集**

使用 TAR 训练集（八大公开交通异常数据集）与测试集（80 CCTV、200 fisheye、40 dashcam 片段）

**📈 对比分析**

在 AI City Challenge 公共排行榜上，MR‑CAS 在 FETV 仅落后 0.0007、PSI‑VQA 排名第 4、TAR 排名第 16；整体性能优于多数基线

**⚠️ 局限性**

局限包括长文本生成不够精准、鱼眼几何推理困难、行人意图判断与时间边界估计不足

---

## 317. EgoPHI: Estimating Contact and Force from Egocentric Vision

**arXiv ID:** 2608.13014 | [PDF](https://arxiv.org/pdf/2608.13014v1)

**作者:** Andela Ilic `[一作]` (ETH Zürich), Christian Holz `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 EgoPHI，一种从单张前视 RGB 图像和已知物体网格推断手部与物体的三维接触分布及力分布的方法；

**💡 创新点**

创新点在于：①首次同时预测手部和非平面物体的 3D 力场；②利用物理仿真生成密集顶点级接触与力监督；③通过图卷积交互块和迭代姿态细化实现手物几何对齐；

**🔧 技术方法**

技术方法包括 ViT 视觉特征提取、图注意力网络 (GAT)、跨网格交叉注意力、迭代姿态细化模块、力与接触预测头，以及 SOFA 物理仿真来生成训练标签；

**📊 数据集**

使用的主要数据集为 ARCTIC（人工制品手物交互），H2O（跨域测试），以及自制光学传感的立方体与圆柱体真实数据；

**📈 对比分析**

与基线 HACO（手部接触）、PressureVision（2D 力图）和无迭代细化版本比较，EgoPHI 在 ARCTIC 与 H2O 上的力 MAE/RMSE 低 1–2 N，体积 IoU 提升至 10‑20 倍，接触 F1 也明显优于基线；

**⚠️ 局限性**

局限性包括：仅处理单帧，缺乏时序一致性；需预先知晓物体网格；物理仿真采用统一刚度，难以模拟软体或多材质物体；仅预测法向力，无法得到摩擦力；以及在遮挡严重的前视场景中物体姿态估计仍存在误差。

---

## 318. OmniSphinx: Active Mix Networks (Extended Version)

**arXiv ID:** 2608.13008 | [PDF](https://arxiv.org/pdf/2608.13008v1)

**作者:** Daniel Schadt `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出 OmniSphinx，一种主动混合网络格式，允许发送者在数据包中嵌入混合程序，实现对任意现有或未来混合格式的仿真。

**💡 创新点**

创新点在于将主动网络的“自定义程序”概念引入混合网络，设计兼容所有主流混合格式的指令集，并通过信息流分析评估程序安全；此外给出了实证评估并展示了灵活性带来的优势。

**🔧 技术方法**

采用 Diffie‑Hellman 密钥交换、伪随机生成器 (PRG)、哈希、MAC、对称加密/解密（LIONESS）、AES‑CTR PRG 等加密原语，并实现了基于寄存器的指令集来执行混合程序。

**📊 数据集**

实验使用固定路径长度 5、payload 大小 1 KiB，并对比了 Sphinx、AE‑Sphinx、EROR、MultiSphinx、PolySphinx 等主流格式，未使用专门的数据集。

**📈 对比分析**

通过与上述五种格式在头部大小和处理时间的对比，发现 OmniSphinx 头部比 Sphinx 大约 33%（最小差异），比其他格式可达 289%；处理时间从约 200 ms 变为 300 ms（约 +90%）。

**⚠️ 局限性**

主要局限包括：额外的头部和解释开销导致性能下降；指令集有限，可能不支持未来更复杂的格式；安全性依赖于发送者正确编写混合程序，需使用信息流规则进行审计；实现基于 Java，未验证在高并发环境下的可扩展性。

---

## 319. EviReform: Evidence-Guided Query Reformulation for Multi-Hop Graph Retrieval

**arXiv ID:** 2608.13006 | [PDF](https://arxiv.org/pdf/2608.13006v1)

**作者:** Xinlong Xu `[一作]` (Nanjing University of Information Science and Technology), Yoshua Y. Li `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将检索请求改写与图结构聚合分离的多跳检索方法，利用已检索证据生成残留查询并结合原始查询信号。

**💡 创新点**

创新点在于将检索请求改写与图传播两个决策解耦，分别对原始问题和已观察证据生成独立检索信号，并通过共享实体传播统一归一化后合并，显著提升链式检索效果。

**🔧 技术方法**

技术手段包括基于LLM的命题抽取与实体链接、稠密检索、残留查询生成、共享实体的图传播（矩阵求逆稀疏计算）以及多信号归一化与融合。

**📊 数据集**

使用 2WikiMultiHopQA、HotpotQA、MuSiQue 三个多跳问答基准以及 GraphRAG-Bench（Medical）医学数据集进行评测。

**📈 对比分析**

与稀疏、稠密、迭代和图检索基线对比，最大提升 Recall@5 5.59 分，F1 4.50 分，在链覆盖率 Chain@5 上提升 11–22 分，显著优于现有方法。

**⚠️ 局限性**

局限性包括依赖LLM生成的命题索引和查询改写，可能导致重建时结果波动；实验仅覆盖英语基准，跨语言适应性待验证；最终证据集合挑选仍为开放问题。

---

## 320. PixSDS: Why Latent SDS Makes Noisy Pixels

**arXiv ID:** 2608.12997 | [PDF](https://arxiv.org/pdf/2608.12997v1)

**作者:** Vsevolod Skorokhodov `[一作]` `[通讯]` (EPFL), Vsevolod Skorokhodov (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

分析并解决了在基于 VAE 的 latent Score Distillation Sampling (SDS) 中出现的结构化像素噪声和色彩伪影问题，并提出了 PixSDS 方法来修正这些噪声。

**💡 创新点**

创新点在于：①揭示了 VAE 像素漂移（pixel drift）是导致 latent SDS 噪声的根本原因；②设计了无需重训练扩散模型、渲染器或更改 SDS 目标的轻量级梯度修复机制 PixSDS；③利用 VAE 的“前瞻性解码”生成 VAE 一致的干净方向，动态匹配原始 SDS 更新。

**🔧 技术方法**

技术包括：Score Distillation Sampling (SDS)、变分自编码器 (VAE) 编码器/解码器、基于像素梯度的修复与正则化、在 2D 生成与 3D 文本到 3D 生成管线中的应用、对不同 SDS 变体（SDS-Bridge、HiFA、NFSD、PGC、SDI、VSD、2-step-SDS 等）的实验对比。

**📊 数据集**

数据集主要使用 MS‑COCO 2014 的 100 条随机标题进行 2D 生成实验；3D 生成实验基于 DreamGaussian 与 LucidDreamer 两个现有的文本到 3D 的流水线；还使用了 Stable Diffusion 以及 Stable Diffusion Nano 2.1 作为对比与基线。

**📈 对比分析**

对比方法：在 2D 生成中将 PixSDS 与 10 种 SDS 风格方法以及直接采样的 Stable Diffusion 进行比较；在 3D 生成中将 PixSDS 集成至 DreamGaussian 和 LucidDreamer 的第二阶段，并与原始 SDS 以及经过高斯平滑的基线对比。性能表现：PixSDS 在 FID、BRISQUE、CLIP‑IQA 噪声等无参考指标上往往优于其他 SDS 变体，且能显著降低结构化噪声与色彩伪影，同时保持语义内容和细节。

**⚠️ 局限性**

局限性包括：①对 VAE 结构和参数的依赖性较强，VAE 设计差异可能影响修复效果；②在某些 3D 场景（如白发、透明材质）中仍出现残余噪声，需要进一步调优 β 等超参数；③实验主要在固定的 VAE 和扩散模型上验证，尚未在更大规模或多模态模型中全面评估。

---

## 321. OGR-MARL: Option-Guided Residual Multi-Agent Reinforcement Learning for Heterogeneous USV Cooperative Pursuit in Constrained Port Waterways

**arXiv ID:** 2608.12995 | [PDF](https://arxiv.org/pdf/2608.12995v1)

**作者:** Mao Jiayang `[一作]` (Sichuan Agricultural University), Peng Zhao-Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向异构无人水面艇（USV）在受限港口水道中协同追捕的选项驱动残差多智能体强化学习框架（OGR）。

**💡 创新点**

创新点在于：①算法无关的框架设计，能与任意连续控制的 MARL 后端集成；②通过共享被追踪目标信念、角色定向选项目标、规则惩罚适配器和残差策略学习，显著降低了在受限水道中的探索难度；③实现了无训练的零样本迁移到真实 QGIS/AIS 地图。

**🔧 技术方法**

主要技术包括：多智能体强化学习（MADDPG、MATD3、MAPPO、MASAC）+选项层；三维动态建模与低层速度/航向控制；A* 与 APF 用于被追踪目标规划；结构化奖励与课程学习；动作混合（残差+选项）机制。

**📊 数据集**

使用的数据集：①抽象化的 Xiazhimen 港口水道模拟（1000×400 坐标域，包含岸线、停泊区、双向航道等）；②基于 QGIS 绘制岸线、AIS 导入船舶轨迹的真实 Xiazhimen 地图。

**📈 对比分析**

通过与纯 MARL 基线和专家规则控制器比较，采用 SR（成功捕获率）、TC（平均任务时长）、MRC（任务有效规则遵守率）和 HCS（异构协调得分）四指标评估。OGR‑MASAC 在抽象地图上取得 75% SR、121.94s TC、0.4283 MRC、0.6802 HCS；相较于 MASAC 基线（14% SR）提升显著。零样本迁移到真实地图后，SR 为 66.7%。

**⚠️ 局限性**

局限性：①在真实复杂地图上的 SR 与抽象环境相比仍有下降，主要因岸线曲折、港口布局不规则、船舶动态变化等因素；②残差策略在最终接近捕获阶段仍存在效率瓶颈；③实验集中在仿真环境，尚缺乏在实际海上部署的验证；④需要大量训练样本，样本效率可进一步提升。

---

## 322. LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation

**arXiv ID:** 2608.12990 | [PDF](https://arxiv.org/pdf/2608.12990v1)

**作者:** Dongfang Li `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种长周期LLM代理的记忆框架，将逐轮的 eager 合并改为语义段级的批量合并，使用单次LLM编码生成自包含的 typed 记忆记录，并通过结构化索引与规划式多路检索实现低成本、高质量的记忆查询。

**💡 创新点**

创新点包括：
- 在线语义分割与边界检测，依据对话语义热度与连贯性自动划分段，减少不必要的 LLM 调用。
- 将每个段一次性编码为含实体、主题、时间、关系等元数据的 typed 记录，既保持细粒度证据，又能在检索时直接使用。
- 轻量级跨段消歧和引用上下文，使后续段落可在不回溯完整历史的情况下保持实体一致性。
- 结构化索引（实体、主题、时间、事件帧）与规划器联合的多路检索，避免了多轮 LLM 推理，提升检索覆盖率。

**🔧 技术方法**

核心技术：
- 语义惊讶与连贯度分数用于在线段划分。
- 单次 LLM 编码（含事实提取、共指消歧、时间规范化）。
- 记录向量检索、结构化索引检索、时间检索、原始对话检索。
- 查询规划器（一次 LLM 调用），随后无生成的向量检索、RRF 融合、rerank 与多样性选择。

**📊 数据集**

数据集：LoCoMo（10 轮会话，≈600 轮，总约 16K tokens）与 LongMemEval‑S（500 轮会话，≈115K tokens），覆盖单跳、多跳、时序推理、偏好跟踪、跨会话推理等多种长记忆需求。

**📈 对比分析**

与 Full Context、Naive RAG、Mem0、A‑Mem、MemoryOS、TiMem 等基线对比；使用相同 LLM 基础模型与评测协议；结果显示：
- LoCoMo 上 89.22%（比 A‑Mem 高 4.4pp，构建成本降低 86%）；
- LongMemEval‑S 上 92.20%（比 A‑Mem 高 3.4pp，构建成本降低 75%），查询 token 与基线相比也更低。

**⚠️ 局限性**

局限性：
- 仍依赖 LLM 对段文本的单次编码质量，若段划分不佳或 LLM 生成不完整，可能导致记忆缺失。 
- 对语义分割阈值有一定敏感性，需在不同语境下微调。 
- 仅针对文本对话验证，未扩展至多模态记忆或更大规模场景。 
- 架构复杂度略高，需要多组件协同，部署成本与维护成本尚待评估。

---

## 323. Reconcile Once, Write Anytime: A Trust-Tiered Librarian and a Multi-Agent Writer for Drift-Free, Point-in-Time Research

**arXiv ID:** 2608.12984 | [PDF](https://arxiv.org/pdf/2608.12984v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个两阶段、两层级的系统，先用确定性流程在维护的、可信度分层的知识库中归档、统一量化和对比事实，再由多智能体写作引擎根据指定时间点生成无漂移、可追溯的长篇报告。

**💡 创新点**

创新点包括：① 将知识库与写作解耦，形成“实时可信度图书馆”；② 采用可信度分层（official > gov_stat > sell_side > media）和官方‑优先量化账本，消除跨文段漂移与虚假来源；③ 设计确定性 QC 门和红队回写循环，实现自动纠错和持续改进；④ 通过分层模型路由和有界并行实现高效多智能体协同而不牺牲一致性。

**🔧 技术方法**

技术栈包括：Python 4.0+确定性管道（OCR/抽取、数值校正、图构建、账本更新）；Claude Haiku 4.5 用于数值卡细化；Claude Opus 4.8 充当红队审计；Sonnet/Sonnet 5 用于普通段落生成；自研的无框架多智能体执行器，支持分段并行、难度路由和共享存储；点‑时间桥接（bridge）实现无前瞻投射；日志化的增量写回机制。

**📊 数据集**

使用 6,130 份公开英文文档（5,397份 SEC EDGAR 备案、672份美国劳工统计局宏观发布、61 份维基百科条目），提取 555,926 条 evidence card（457,561 数值卡）并构建 2,589 权威公司‑指标账本，覆盖 11 行业、295 家上市公司。

**📈 对比分析**

通过 8 个可重复实验评估：E1 证明共享账本消除 6,845 条跨节矛盾；E2 记录 99.5% 的数字行都有引用；E3 证明 0% 的媒体/宏观来源被误作可引用数值；E4 证明官方‑优先策略在 22/22 案例上准确率 100%，对比仅 9/22 的流行度策略；E5 通过缺陷注入验证 QC 门 100% recall/precision；E6 显示并行 + 路由方案 3.7× 更快、成本 4.1% 低于全 Opus；E7 展示点‑时间回放无前瞻违例，库持续增长；E8 示范红队回写可自动纠正后续报告。整体性能表现为零矛盾、完美引用、可重复性强、成本与延迟可控。

**⚠️ 局限性**

局限包括：仅使用英文数据；三层可信度划分（官方、宏观、媒体）缺少 sell_side 及其他行业来源；宏观层不具公司归属，导致无法评估跨层冲突；基于子串的实体链接偶尔误归属；E4 的黄金集是人为设计，未覆盖真实跨层冲突；E6 采用确定性质量代理而非人工评审；E8 的回写验证仅针对少量案例；系统对非常大规模动态更新的处理能力尚未在极端负载下测试。

---

## 324. H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models

**arXiv ID:** 2608.13049 | [PDF](https://arxiv.org/pdf/2608.13049v1)

**作者:** Dingyi Rong `[一作]` (Shanghai Jiao Tong University), Ning Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了 H2R-Bench，衡量视频模型将人类自我摄像的操控视频转换为指定机器人身体结构的能力。

**💡 创新点**

首次引入源条件下的跨身体结构人类-机器人视频生成评估框架，并设计了五维度（目标完成、动作完成、功能接触、身体一致性、视频质量）进行系统诊断。

**🔧 技术方法**

采用多模态大模型和现有视频生成系统（如 Seedance、Wan2.7、Kling-V3 等）进行生成，并用 Gemini、Qwen 等大模型进行自动化评估。

**📊 数据集**

使用 EgoDex 数据集中的 120 条人类演示视频，并配合两种机器人身体结构（平行夹爪、灵巧手）构成 240 条转移案例。

**📈 对比分析**

通过 M1–M4 人工评估与自动评估相结合，计算 H2RCore 综合得分；视频条件模型在两种机器人体型上分别获得最高 84.6 与 77.3 分，显示功能接触和身体一致性是主要瓶颈。

**⚠️ 局限性**

目前模型在功能接触和身体一致性方面表现不佳，视频质量与迁移效果关联弱，缺乏对跨身体结构的充分学习与泛化能力。

---

## 325. Pareto-Aware Hierarchical Reinforcement Learning for Online Resource Allocation in RIS-assisted Large-Scale IoT Systems

**arXiv ID:** 2608.13032 | [PDF](https://arxiv.org/pdf/2608.13032v1)

**作者:** Wenhan Xu `[一作]` (Hong Kong University of Science and Technology), Danny H. K. Tsang `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种层级强化学习框架 PAAERL，结合 Pareto 优化与自编码器，实现 RIS 辅助多用户 IoT 系统的实时资源分配。

**💡 创新点**

创新点在于双重压缩策略：先用方向性最大最小化映射将高维 RIS 变量映射到低维 Pareto 权重空间，再用深度自编码器将权重进一步压缩为极低维连续动作，既保证了理论上的 Pareto 最优，又显著降低了动作空间维度。

**🔧 技术方法**

主要技术包括 PPO 强化学习、方向性 Pareto 映射（最大最小尺度优化）、自编码器（encoder‑decoder 结构）、二分搜索求解 R_s、交替优化与 SCA 的凸化求解。

**📊 数据集**

使用离线生成的随机信道与网络状态样本构建训练集，包含权重向量与对应的 Pareto 资源配置；所有实验均在仿真生成的 RIS‑MEC 环境中进行。

**📈 对比分析**

通过与五个基准（RL、WSRRL、PARL、AERL、WSRAERL）在总系统成本（延迟+能耗）和训练时间等指标比较，PAAERL 在成本上最低、收敛最快、训练时间最短，尤其在用户数和 RIS 元素增大时表现最为优异。

**⚠️ 局限性**

局限性包括：需要昂贵的离线预训练自编码器，若重构误差较大可能略微影响性能；内部求解的可行性子问题仍有计算开销；在极端动态移动或高噪声环境下的鲁棒性尚未充分验证。

---

## 326. Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning

**arXiv ID:** 2608.13026 | [PDF](https://arxiv.org/pdf/2608.13026v1)

**作者:** Yao Zhou `[一作]` (Chinese Academy of Sciences), Wenwen Qiang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Temporal GRPO 的阶段条件化时序信用分配框架，用于解决视觉-语言-动作（VLA）模型在稀疏任务成功反馈下的轨迹级信用混淆问题，并在机器人后期强化学习中提升任务成功率和样本效率。

**💡 创新点**

创新点在于：①构建可检测的有序任务阶段并将完整轨迹与阶段对应的动作区间对齐；②仅对已完成前置阶段且进入同一阶段的轨迹进行组内优势比较，产生阶段级相对优势；③将阶段优势仅赋给对应动作区间，实现时序化、局部化的信用分配，从而消除轨迹级信用混淆，提升长周期操纵任务的稳定性和效率。

**🔧 技术方法**

使用的技术包括：冻结的 RynnBrain-4B 语言模型生成语义阶段，Stage Compiler 将其编译为可检测的阶段；关系检测器在仿真中评估阶段完成条件；基于 GRPO 的组内优势计算与剪切策略；对 VLA 策略进行单一目标优化，但优势在时序上被分配到不同阶段。

**📊 数据集**

主要实验数据集为 RoboTwin 2.0（涵盖不同任务时间尺度）以及 LIBERO-Long（用于控制信用分配与消融研究）。

**📈 对比分析**

与基线（π_0、RDT-1B、Trajectory-GRPO、Stage-Reward GRPO、SimpleVLA-RL、TGRPO）以及无消融的版本比较。Temporal GRPO 在 RoboTwin 2.0 的宏观平均成功率达到 75.8%（高于最强基线 68.8%），在 Long 与 Extra-Long 任务上提升明显；在 LIBERO-Long 的消融实验中，移除阶段编译器、进入阶段门控或同阶段分组均导致显著性能下降，证明阶段化信用机制的必要性。

**⚠️ 局限性**

局限性包括：依赖可靠的阶段检测器和预定义的线性阶段顺序；在阶段边界模糊、任务存在分支、循环或回溯时难以正确对齐和分配信用；目前未处理阶段检测的不确定性，未来工作计划加入不确定感知与动态阶段图。

---

## 327. Foundations of MT-PDCL: Measure-Theoretic Probabilistic Definite Clause Logic

**arXiv ID:** 2608.13018 | [PDF](https://arxiv.org/pdf/2608.13018v1)

**作者:** Costin Bădică `[一作]` (University of Craiova), Amelia Bădică `[通讯]` (University of Craiova)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的概率逻辑编程框架 MT-PDCL，能够在连续可测空间中直接执行定理推理与查询，不再需要传统的离散化布尔化。

**💡 创新点**

核心创新在于将概率分布映射到标准 Borel σ‑代数上的连续变量，通过对可能世界的 Lebesgue 积分来定义命题的推理概率；并引入连续化的即时后果算子与连续 Noisy‑OR 近似，以实现可微分、可迭代的推理。

**🔧 技术方法**

技术手段包括：测度论框架、连续分布语义、连续即时后果算子、连续 Noisy‑OR 近似、以及基于 Banach/Kleene 固定点理论的收敛证明。

**📊 数据集**

本文为理论性工作，没有使用具体数据集；所有示例均为合成的连续或混合连续/离散问题。

**📈 对比分析**

对比方法：在示例中与传统离散化方法（如 ProbLog、PRISM、DeepProbLog 等）比较，指出后者需要离散化导致维度灾难，而 MT‑PDCL 直接进行连续积分可获得精确概率；实验结果表明在连续域推理上实现了更高的准确性和可微分性。

**⚠️ 局限性**

局限性包括：连续 Noisy‑OR 近似在变量存在强相关时可能失真；对高维连续域的精确积分仍然计算代价高昂；以及需要手工定义索引域和分布映射，模型的可扩展性受限。

---

## 328. How LLMs Respond to Escalating Delusions: Four Longitudinal Trajectories of Model Behavior

**arXiv ID:** 2608.13017 | [PDF](https://arxiv.org/pdf/2608.13017v1)

**作者:** Anna Sterna `[一作]` (IDEAS Research Institute), Marcin Moskalewicz `[通讯]` (IDEAS Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项为期30天、使用固定脚本的纵向实验中，评估了15个主流大型语言模型（LLM）对潜在AI精神病的影响，结合人工主观评估和两项自动化指标，绘制了模型在识别、干预与交互信任方面的演化轨迹。

**💡 创新点**

首次将纵向定性评估与自动化“跟随度”与“语气自信度”指标相结合，揭示了四类不同的风险轨迹，并提供了对LLM在精神病用户对话中安全性的动态阈值。

**🔧 技术方法**

使用了四位心理学学生对模型日响应进行评级的定性评估（识别阶段、解释自信、干预类型），以及基于语义嵌入的跟随度（cosine相似度）和基于词汇计数的语气自信度（提升词/保留词）两项自动化指标。

**📊 数据集**

采用了一段30条消息的非自适应脚本，模拟从轻度异常体验到精神病思维的逐步发展，对15个模型进行逐日对话，共计449个模型-日样本；脚本与评估细节在补充材料中给出。

**📈 对比分析**

通过共识轨迹里程碑、延迟时间及三参数逻辑回归拟合对模型进行比较，发现四种轨迹（早期医疗化、识别无保护、延迟不稳定、共建妄想）均存在安全隐患，且不同模型在识别延迟和稳定性方面差异显著。

**⚠️ 局限性**

实验采用固定脚本且非自适应，样本仅限一次性采集，自动化指标仅为表面代理，人工评估在某些维度可靠性低，且模型可能随更新变化，限制了结论的普适性与长期有效性。

---

## 329. Semantic Intelligence Against CSAM: The PreventCSA@EU Ontology Framework for Classification and Investigation

**arXiv ID:** 2608.12979 | [PDF](https://arxiv.org/pdf/2608.12979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 330. RAGSieve: Self-Referenced Local Contrast for Knowledge-Poison Detection in Retrieval-Augmented Generation

**arXiv ID:** 2608.13010 | [PDF](https://arxiv.org/pdf/2608.13010v1)

**作者:** Xinlong Xu `[一作]` (Nanjing University of Information Science and Technology), Yoshua Y. Li `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自我参考局部对比的RAG语料库投毒检测框架 RAGSieve，包含查询时 RSQ 与离线时 RSG 两种部署模式；

**💡 创新点**

创新点在于利用检索系统自身产生的局部对照（检索尾部或本地图）作为参考，无需外部干净语料或全局阈值，实现针对检索与生成两个控制点的深度防御；

**🔧 技术方法**

主要技术包括答案锚点浓度检测、脚本完整性检测、语言模型惊奇度与多尺度 BERTScore 对齐、局部图密度对比等；

**📊 数据集**

使用自然问题（NQ）、HotpotQA、MS MARCO 三大问答数据集，结合 BGE‑M3、E5‑large‑v2、MiniLM‑L6‑v2 三种密集检索器；

**📈 对比分析**

与 RAGuard、GMTP、EcoSafeRAG、TrustRAG、CleanBase、AHD 等基线对比，RSQ 在 macro AUROC 上达 95.2%，在 5% 干净文档删除预算下检测率为 82.2%；RSG 的 macro AUROC 为 93.3%，检测率 79.8%；联合部署将攻击成功率从 67.4% 降至 14%，同时保持高 F1 分数；

**⚠️ 局限性**

局限性包括：对单文档或稀疏投毒检测效果有限；离线模式假设局部密度变化，若投毒不形成密度集群则难检；查询尾部不干净会削弱 RSQ 的区分度；误删干净文档的风险；多语言、多主题语料的适用性尚待验证。

---

## 331. Explanatory Engagement Under Rare Anomalous Failure: Asymptotic Rarity in Model Behavior (or: The Asymptotic AI)

**arXiv ID:** 2608.13063 | [PDF](https://arxiv.org/pdf/2608.13063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 332. A Theory of Probabilistic Power Provisioning for Data Centers with Distributed Energy Storage

**arXiv ID:** 2608.12993 | [PDF](https://arxiv.org/pdf/2608.12993v1)

**作者:** Can Emre Koksal `[一作]` (Ohio State University), Artun Sel `[通讯]` (Ohio State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文建立了概率框架，定量分析分布式储能对数据中心功率预留、储能容量与超额率之间的权衡；

**💡 创新点**

创新点在于将储能视为可调节资源，引入“有效功率”概念，揭示小电池区与大电池区两种运行模式，并给出闭式近似与阈值；

**🔧 技术方法**

采用大偏差理论、有效带宽/有效功率、Chernoff 绑定与马尔可夫建模，推导最优功率与存储尺寸关系；

**📊 数据集**

使用三套真实工作负载数据集：Ohio Supercomputer Center（HPC）、Microsoft Philly GPU 训练集、Alibaba 云服务集；

**📈 对比分析**

通过与无储能基线、现有功率分配/电压削减方法比较，验证模型预测与实测匹配；在典型 10^-3 的超额率下，储能可降低 30%–50% 的功率余量；

**⚠️ 局限性**

局限包括：假设独立/同态负载、忽略跨层互作、对非高斯重尾或极端相关性不足、对多层储能分层管理的细节未给出完整解析。

---

## 333. DiCoR: Decoupled Referent Disambiguation and Contour Recalibration for Efficient Referring Remote Sensing Image Segmentation

**arXiv ID:** 2608.12980 | [PDF](https://arxiv.org/pdf/2608.12980v1)

**作者:** Ziyang Gao `[一作]` (Shanghai Jiao Tong University), Hai-Bao Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DiCoR框架，用解耦的方式提升遥感图像指代分割的定位与轮廓校正效率。

**💡 创新点**

创新点在于将指代定位转化为候选竞争并用分级响应引导，以及轻量级的轮廓校正模块，实现高精度与低延迟的平衡。

**🔧 技术方法**

采用Swin Transformer视觉编码、BERT语言编码、跨模态注意力、多尺度聚合、候选响应网络、语义重加权排名以及残差轮廓校正网络。

**📊 数据集**

使用RefSegRS、RRSIS-D和RISBench三大公开遥感指代分割基准。

**📈 对比分析**

与JFS和DPS方法对比，在mIoU、gIoU等指标上均超过或接近最高者，同时推理速度提升约4.7×，取得最佳accuracy‑efficiency指数。

**⚠️ 局限性**

局限在于对初始响应的依赖，难以弥补定位失败；对大目标的全局轮廓调整不足。

---

## 334. Fortune's Bounty: Taming Complexity by Trimming Trees --- A Hands-On Problem-Solving Experience in Advanced Complexity Suitable for Introductory Students

**arXiv ID:** 2608.12976 | [PDF](https://arxiv.org/pdf/2608.12976v1)

**作者:** Kimberly Fluet `[一作]` (University of Rochester), Christopher M. Homan `[通讯]` (Rochester Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并提供了一份课堂作业，鼓励本科生在团队中尝试自行证明Fortune定理，旨在让学生获得研究经验。

**💡 创新点**

创新点在于将高级复杂性理论中的证明过程（Fortune定理）转化为可操作、可在课堂和作业中完成的任务，从而让学生即使在缺乏相关背景知识的情况下也能亲自体验研究过程。

**🔧 技术方法**

采用的技术包括：分组合作、课堂即时讨论与作业延伸、树剪枝策略指导、以及提供完整的解题手册与参考答案。

**📊 数据集**

该研究不涉及实验数据集，重点是理论证明和教学实践。

**📈 对比分析**

通过学生完成率、课堂反馈和作业质量等方式进行评价，结果显示多数团队能够在作业时间内成功完成任务，体现出较高的学习与研究能力。

**⚠️ 局限性**

局限性包括：需要教师根据自身课程自行编辑和调整作业细节；缺乏大规模、系统的评估数据；以及对学生前置知识水平的依赖，可能影响不同班级的适用性。

---

## 335. TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes

**arXiv ID:** 2608.13057 | [PDF](https://arxiv.org/pdf/2608.13057v1)

**作者:** Jie Li `[一作]` (KlingAI Research), Chengru Song `[通讯]` (KlingAI Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于实测专家计算时间的 EP 负载均衡调度器，能够在每个 batch 上实时求解最小 makespan。

**💡 创新点**

创新点在于：①通过十分钟黑盒校准揭示专家计算存在的“内存阶梯”和“GEMM 分块”两大非线性 regime；②把两 regime 融入单一的 max‑affine 成本模型；③将问题转化为固定费用 makespan 优化，给出可在毫秒级完成且具有加性最优保证的调度算法；④构建阶段化“phase diagram”，可预先判定何时调度能带来收益。

**🔧 技术方法**

使用了深度学习推理框架 SGLang、DeepEP、DeepGEMM fp8 kernel、CUDA‑graph、HiGHS MILP 求解器、GNN/梯度求解（用于重分配），并通过统计分析、模拟器、实测实验评估。

**📊 数据集**

数据集包括 Qwen3‑235B、DSv3/DSv2‑Lite、DeepSeek‑V3、Wikitext、GSM8K、OASST1、GovReport、ShareGPT 等；使用真实服务器日志和合成 Zipf 分布进行测量与仿真。

**📈 对比分析**

与现有基于 token‑LP、METRO、EPLB 等传统代理比较，方法在 1–2 % 以内与 MILP 最优对齐；在 8‑GPU Testbed A 与 2×8‑GPU Testbed B 上，平均吞吐提升 4–6 %（长上下文）、p99 延迟下降约 15–20 %；在 Qwen3‑235B‑FP8 的 8‑GPU 方案中，整体吞吐提升 38–70 %，低于静态策略时的延迟降低 50–60 %；对比静态、LPLB、METRO 的实验表明新调度器在混合 regime 下明显优于任何单一代理。

**⚠️ 局限性**

局限包括：① 仅在单批次内求解，未对跨批次动态迁移权重做深度协同；② 需要十分钟的硬件校准，硬件更新后需重新校准；③ 成本模型忽略了轻量化 kernel 的“slot‑max”额外开销，导致在极小专家（bf16）场景下无明显收益；④ 仍以固定 placement 为前提，无法在极端复制预算下实现全局最优；⑤ 对多节点网络延迟的模型仍简化，未涵盖所有网络拓扑变化。

---

## 336. Latent On-Policy Self-Distillation

**arXiv ID:** 2608.13040 | [PDF](https://arxiv.org/pdf/2608.13040v1)

**作者:** Guibin Zhang `[一作]` (National University Of Singapore), Shuicheng Yan `[通讯]` (National University Of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Latent On-Policy Self-Distillation（LOPS），通过检索并压缩以往轨迹为连续隐向量，构建可学习的特权上下文，让自教师对学生的每个前缀进行密集的反向KL蒸馏，并加入特权边际约束以保持教师信息量。

**💡 创新点**

创新点在于把传统OPSD中手工设计的特权信息转化为可端到端学习的连续隐向量，并通过联合优化与特权边际约束实现教师对学生的可解释且有效的监督，而非依赖预定义的答案或轨迹。

**🔧 技术方法**

所用技术包括：基于检索的经验库、QFormer式交叉注意力隐向量生成、LoRA微调的编码器、冻结教师骨干与学生分离、Top‑M+尾部的反向KL蒸馏损失、以及基于奖励的特权边际双重变量约束。

**📊 数据集**

实验数据集涵盖工具使用的EnvScaler（2349任务）与其子集BFCL‑v3、ACEBench，以及代码生成的TACO子集DeepCoder（7K Python问题）与LiveCodeBench v5/v6、HumanEval+和MBPP+。

**📈 对比分析**

通过与Vanilla、GRPO、SDFT、OPSD、SDPO、Skill‑SD等基线对比，LOPS在所有十种模型-基准组合中均取得最优或次优得分，工具使用上平均提升约2–18点，代码生成上平均提升约1–4点，且使用不到30%传统方法的回合预算。

**⚠️ 局限性**

局限性包括：对检索器与经验库的依赖；隐向量的可解释性有限；特权边际阈值和检索数量需调参；在与任务不匹配或知识稀缺的场景下可能效果不佳；目前仅在工具与代码生成两大范畴验证，跨域泛化尚待进一步探索。

---

## 337. Incremental Evaluation and Training in Relational Deep Learning

**arXiv ID:** 2608.13023 | [PDF](https://arxiv.org/pdf/2608.13023v1)

**作者:** Jakub Peleška `[一作]` (Czech Technical University in Prague), Gustav Šír `[通讯]` (Czech Technical University in Prague)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种增量多剧集评估与训练框架，用于衡量 Relational Deep Learning（RDL）模型在随时间演进的数据库中的性能变化。

**💡 创新点**

将传统的静态快照评估转变为连续时间增量评估，量化时间概念漂移，并引入时间加权指数衰减指标和多种增量微调策略，首次为 RDL 提供了针对动态数据库的完整基准。

**🔧 技术方法**

采用图神经网络（GraphSAGE）与表格 ResNet 编码器相结合的 RDL 模型，配合 PyTorch Lightning 实现增量训练，并使用时间加权指数衰减作为评估指标。

**📊 数据集**

在 RelBench 与 ReDeLEx 公共数据库上进行实验，涵盖 12 个回归与二分类预测任务。

**📈 对比分析**

通过将从零训练与三种微调（累计、增量、上采样）进行对比，发现微调策略在大多数任务上能匹配甚至超过从零训练，且仅需几百步即可逼近最终性能；时间加权评估进一步证明微调在近期窗口表现更佳。

**⚠️ 局限性**

局限性包括：仅考虑追加式增量，未处理更新/删除等 CRUD 操作；对长期知识保持和灾难性遗忘的解决仍不完善；对大型基础模型在连续评估下的鲁棒性尚需进一步研究。

---

## 338. Applied and Filtered: An End-to-End Algorithmic Fairness Audit of A Public Employment Agency

**arXiv ID:** 2608.13022 | [PDF](https://arxiv.org/pdf/2608.13022v1)

**作者:** Gemma Galdón-Clavell `[一作]` `[通讯]` (Eticas.ai), Gemma Galdón-Clavell (Eticas.ai)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对巴塞罗那公共就业机构使用TalentClue平台的半自动化招聘系统进行端到端公平性审计，分析近50万条候选人–职位记录；

**💡 创新点**

首次将公平性评估延伸到完整的招聘管道，揭示了聚合指标掩盖的性别、薪酬、年龄和非二元候选人差异，并指出供应商与部署方信息不对称；

**🔧 技术方法**

采用离散化的分阶段审计方法，计算失配比（DIR）、卡方检验和分层/交叉性分析，并进行时间一致性检验；

**📊 数据集**

利用巴塞罗那Activa从2017-2022年的运营数据及西班牙劳动力调查（EPA）对比基准；

**📈 对比分析**

与传统仅评估模型输出的公平性检验相比，分阶段和交叉性分析揭示了更细粒度的差距，显示了聚合指标无法捕捉的系统性不平等；

**⚠️ 局限性**

数据缺失（24%性别、14.5%国籍），缺乏资格、技能等关键变量，观察性设计限制因果解释，供应商内部逻辑不可访问导致无法精准定位偏差来源。

---

## 339. ATOBench: Tracing How Autonomous Penetration-Testing Agents Verify Vulnerabilities When Target Evidence Lies

**arXiv ID:** 2608.12996 | [PDF](https://arxiv.org/pdf/2608.12996v1)

**作者:** Qiyang Chen `[一作]` (Alibaba Cloud, Alibaba Group), Junlin Liu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型在渗透测试中的验证链进行基于目标响应欺骗的对照实验，提出了Adversarial Target Observation (ATO) 以及对应的评估框架 ATOBench。

**💡 创新点**

创新点在于：①将目标响应的“干预”抽象为可冻结、可复现的 Adversarial Observation Unit (AOU)；②通过 Native‑ATO 事件配对、anchor 对齐和轨迹重构，细粒度追踪验证链中证据获取、停机决策与最终报告的演变；③展示不同欺骗合同（SQLi、Basket、JWT）对多种 LLM 路径的差异化影响。

**🔧 技术方法**

技术上使用：LLM 代理（DeepSeek‑V4‑Pro、GLM‑5.2、Kimi‑K2.6、Qwen3.7‑Max、GPT‑5.5）+ Claude Code 工具链；MITM 代理在目标执行后注入 AOU；匹配配对跑、anchor 对齐、轨迹判断器、证据重构与双盲评判。

**📊 数据集**

数据集：固定版本的 OWASP Juice Shop 20.1.1；共 450 条 episode（225 对 Native‑ATO），覆盖三种证据结构（SQLi、Basket、JWT）。

**📈 对比分析**

比较方法：对同一模型、任务、预算的 Native 与 ATO 进行配对比较，计算“grounded verification”率（证据、报告闭合、轨迹支持三者同时满足）。结果显示：SQLi 在所有模型下 ATO 条件下 0%；Basket 从 45.3% 降至 40.0%；JWT 从 84.0% 降至 58.7%，体现出不同合同对验证链的破坏点与恢复率差异。

**⚠️ 局限性**

局限性：①实验仅在单一目标（Juice Shop）和固定工具/预算下进行；②缺乏更大规模或多目标环境的验证；③受限于 LLM 路径与工具接口的多样性，结果可能对其他模型/工具组合不完全泛化；④评估侧重于证据链完整性，未考虑真实世界的攻击时间成本或可扩展性。

---

## 340. STAR: Structured Tokenization and Target-Aware Interest Representation for PCVR Prediction

**arXiv ID:** 2608.12986 | [PDF](https://arxiv.org/pdf/2608.12986v1)

**作者:** Yimeng Xu `[一作]` (Tsinghua University), Lan Ma `[通讯]` (Tsinghua University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出STAR框架，用结构化分词与目标感知兴趣表示提升PCVR预测。

**💡 创新点**

创新点在于高基数信号恢复、目标感知序列解码、用户-物品交互增强以及InfoNCE对齐目标。

**🔧 技术方法**

采用HyFormer式多序列Transformer、结构化Token化、DIN式查询解码、InfoNCE对比学习等技术。

**📊 数据集**

使用KDD Cup 2026腾讯UniRec挑战的PCVR数据集，约2千万训练样本。

**📈 对比分析**

相较官方基线，AUC提升约0.0166；在多项ablation中，时序上下文、InfoNCE、DIN解码和高基数恢复等组件对AUC贡献最大。

**⚠️ 局限性**

局限性包括对高基数映射、稀疏特征恢复等实现细节的工程复杂度，且仅在离线评测中验证，缺乏在线A/B实验。

---

## 341. From One Solution to Many: An Oracle-Based FPT Framework for Diverse Solutions under Generalized Diversity Measures

**arXiv ID:** 2608.13033 | [PDF](https://arxiv.org/pdf/2608.13033v1)

**作者:** Pradeesha Ashok `[一作]` (International Institute of Information Technology), Priyanshu Tiwari `[通讯]` (International Institute of Information Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了计算多样性解的问题，目标是输出一组在某种意义上彼此不同的可行解，而不是单一的可行解。

**💡 创新点**

提出了一种基于oracle的元定理，识别了一类称为一致多样性的目标，涵盖了标准的多样性度量，并且相较于现有框架，显著提高了算法的运行效率。

**🔧 技术方法**

使用了基于oracle的算法，假设可以访问一个精确的空扩展oracle，该oracle能够返回满足特定条件的可行解。

**📊 数据集**

没有具体提到使用的数据集，但提到了一些经典图和母体问题的多样性变体作为应用。

**📈 对比分析**

与Kumabe的框架相比，提出的算法在oracle调用次数上实现了单指数界限2^(krlog(kr))，而Kumabe的框架则是双指数界限，显著提高了性能。

**⚠️ 局限性**

限制在于当前框架未能证明Venn多样性问题的固定参数可解性，且在某些具体问题上仍需进一步研究以优化算法效率。

---

## 342. TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint

**arXiv ID:** 2608.13167 | [PDF](https://arxiv.org/pdf/2608.13167v1)

**作者:** Fnu Pramono `[一作]` (Meta Superintelligence Labs), Sourabh Kulkarni `[通讯]` (Meta Superintelligence Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TRAPSBench基准，用程序化生成的物理视频对比实验检验视觉语言模型在信息不足时是否能适当回避回答；同时设计了Penalized Epistemic Calibration Score (PECS)衡量模型的回答准确性与选择性回避的结合。

**💡 创新点**

创新点在于：1）构建了针对视觉不确定性的“无解”视频对；2）提出PECS这一新度量，兼顾正确回答与精准回避；3）揭示VLM内部确实具备不确定性信号，但输出层抑制导致缺乏自我回避；4）通过线性探测与激活驱动证明该信号可被利用并且在不同体系间迁移。

**🔧 技术方法**

技术主要包括：MuJoCo物理模拟与视频对齐；基于隐层线性探测（LR probe）评估内部不确定性；激活向量驱动（activation steering）对模型生成过程进行因果干预；多样化提示策略（Standard、Guided、JSON）与三模型评判面板；以及对不同VLM架构的统一评估。

**📊 数据集**

使用的主要数据集是TRAPSBench，包含1404对（控制与无解）MuJoCo生成视频，覆盖遮挡、混沌敏感、问题不适宜三类不确定性。

**📈 对比分析**

与传统评测方法相比，PECS在鼓励模型仅在确实无信息时回避的同时抑制随意回避；实验结果显示在16种VLM（Gemini、Qwen3-VL、GPT‑5、Gemma、LLaVA）中，最好的PECS仅为0.292，表明即使具备内部信号，模型在输出层仍缺乏足够的自我回避能力。

**⚠️ 局限性**

局限性包括：1）仅评估合成的刚体物理场景，缺乏真实世界视频的复杂性；2）探测与驱动仅覆盖三大开源架构，闭源模型无法验证；3）PECS仅衡量对控制与无解对的二分类，不考虑更细粒度的置信度等级；4）提示策略对性能影响大，提示缺失时模型可能表现更差。

---

## 343. Predicting Signed Distance Functions for Visual Instance Segmentation

**arXiv ID:** 2608.13135 | [PDF](https://arxiv.org/pdf/2608.13135v1)

**作者:** Emil Brissman `[一作]` (Linköping University), Michael Felsberg `[通讯]` (Linköping University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于神经网络预测像素到物体轮廓的方向距离，从而近似签名距离函数（SDF）并直接得到前景-背景分割。

**💡 创新点**

创新点在于用四个离散方向的距离预测代替传统的锚框方法，能够捕捉任意形状的物体，且输出的SDF可直接阈值化得到高质量分割。

**🔧 技术方法**

采用ResNet50编码器+DFN解码器的encoder‑decoder网络，使用L1距离损失和辅助的二元交叉熵损失来训练方向距离图。

**📊 数据集**

在COCO 2017数据集上训练并评估，使用118K训练图像和5K验证图像。

**📈 对比分析**

与YOLACT进行前景IoU对比，结果显示在前景分割上平均IoU提升约4%，速度更快，但尚未完成完整实例分割。

**⚠️ 局限性**

局限性包括：难以将距离图映射为完整实例分割；对极细部件、反射、阴影及低光照场景表现不佳；锚框方法的优势仍未完全替代。

---

## 344. Unlocking Fractional Moments in Delphic Set Streams

**arXiv ID:** 2608.13126 | [PDF](https://arxiv.org/pdf/2608.13126v1)

**作者:** Aranya Kumar Bal `[一作]` (Indian Statistical Institute), Rudrayan Kundu `[通讯]` (Indian Statistical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在Delphic集合流模型中提出一种新的单通过、低空间、低更新时间算法，用以估计非整数频率矩阵k（k∈(0,1)）以及一系列Bernstein类统计量；

**💡 创新点**

创新点在于将对不同采样率的子流中的离散计数（0矩）视为一个可解析函数，利用其拉普拉斯-互补变换与数值积分将k以及其他统计量还原为对该函数的求值；通过仅利用Delphic集合的大小、成员查询与均匀抽样原语，模拟子采样而不显式枚举集合，从而实现了低更新时间；

**🔧 技术方法**

核心技术包括：①基于离散化的子采样与模拟（利用Bernstein/ Lévy–Khinchine 表示）；②将期望支持多项式转换为对t的积分；③使用梯形规则对积分进行数值逼近；④利用Chernoff与拉普拉斯变换的误差分析；⑤在特殊情况下将整数k估计转化为对多项式的k阶导数求值；

**📊 数据集**

该工作为理论性研究，无实测数据集；所用实验仅在理论上分析复杂度与误差，未涉及真实数据集；

**📈 对比分析**

与之前的0矩与k>1矩估计方法相比，提出的算法在k∈(0,1)与一般Bernstein统计量上首次实现多项式（尤其是polylog）空间与更新时间；在Delphic集合流模型下，空间复杂度为O(polylog|Ω|,log m,ε⁻¹,log(1/δ))，更新时间同阶；与已知的O(ε⁻²log²|Ω|)等算法相比，在频率受限场景下实现显著提升；

**⚠️ 局限性**

局限性：1）仅适用于每个元素出现次数受限于τ的频率受限Delphic流；2）对无频率上界的通用模型尚未突破，证明下界困难；3）对某些Bernstein函数，需显式Lévy密度或可解析的Laplace反演；4）实现细节依赖于高精度数值积分与抽样模拟，实际性能取决于具体实现。

---

## 345. S2-HWM: Sparse Event-Structured Hierarchical World Model for Long-Horizon Surgical Robot Manipulation

**arXiv ID:** 2608.13103 | [PDF](https://arxiv.org/pdf/2608.13103v1)

**作者:** Shuzhe Zhang `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Qiong Wang `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种Sparse Event-Structured Hierarchical World Model（S^2-HWM），利用从原始隐状态轨迹学习到的稀疏事件证据，实现事件级别的目标更新与原子级动作执行的分层控制；

**💡 创新点**

创新点在于：1）通过稀疏事件证据自动识别交互阶段并驱动事件级别的目标更新；2）构造事件转移模型（ETM），以事件边界为单位预测下一状态、持续时长和累计奖励，从而在原子级想象的短视野之外延伸高层价值估计；3）在同一world model架构下同时兼顾原子级与事件级学习，提升长时程稀疏奖励任务的学习效率；

**🔧 技术方法**

使用技术包括：DreamerV3风格的递归状态空间模型（RSSM）进行隐状态学习；稀疏上下文动态模型与straight-through binarization生成事件证据；事件级管理器-工作者策略框架；事件转移模型（ETM）采用内部类别转移模式进行多模态预测；多尺度想象与事件级bootstrap相结合的价值目标；

**📊 数据集**

在SurRoL模拟环境下的PegTransfer任务上进行实验，该任务需要抓取、运输、对齐并稳定放置方块；

**📈 对比分析**

与PPO、DreamerV2以及在同一环境下重新训练的GASDreamerV3做对比。S^2-HWM在单次转移任务上达到98.7%±2.3%的成功率，明显优于GASDreamerV3的76.0%±12.0%（+22.7个百分点）。在重复转移和受扰动的压力测试中亦保持显著优势；

**⚠️ 局限性**

局限性包括：1）仅在仿真PegTransfer任务中验证，缺乏在更复杂或真实手术任务中的实证；2）事件证据与管理器代码的语义解释仍为后验分析，未直接映射至人类可解释的阶段；3）模型对事件边界的依赖可能在极端交互噪声下失效。

---

## 346. CASA: Content-Acoustic Speaking Assessment with Speech Encoder and Large Language Model

**arXiv ID:** 2608.13101 | [PDF](https://arxiv.org/pdf/2608.13101v1)

**作者:** Nhan Phan `[一作]`, Mikko Kurimo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 CASA（Content‑Acoustic Speaking Assessment）架构，用 Whisper‑medium 提取语音交付信息，Qwen3.5‑2B 评估语音内容，从而实现自动说话评估。

**💡 创新点**

创新点在于：①明确将语音交付和语义内容分支分离；②使用轻量化 Whisper‑medium 结合 LoRA 调优，显著减少参数量；③引入辅助损失和三项简单流利度统计，实现无手工特征的端到端评估；④通过可解释的软标记向 LLM 传递声学信息。

**🔧 技术方法**

核心技术包括 Whisper‑medium 语音编码器、Qwen3.5‑2B 大语言模型、LoRA 参数适配、Transformer 聚合器、[CLS] 池化、软标记投射、辅助损失以及自定义的流利度特征。

**📊 数据集**

使用了 2025 年版 Speak & Improve（S&I）语料库，该语料库包含约 315 小时的多种题型口语录音，并提供 CEFR 标注的整体评分。

**📈 对比分析**

与 NTNU、Perezoso、One Whisper 等现有方法对比，CASA 在 S&I 测试集上取得 RMSE 0.358（比 SOTA 0.360 略优），PCC 0.829，且推理参数仅 3.13 B，约为 NTNU 的一半；在各 CEFR 级别的表现与其它方法相近，且在 A2 级别提升更为明显。

**⚠️ 局限性**

局限性包括：①对语料中 P1、P4 等短、过程型题型性能相对较弱；②任务嵌入在 Transformer 中效果不佳；③训练过程中存在显著的非确定性，单次运行 RMSE 可波动 0.008；④更大模型或更高容量的 LLM 并未进一步提升性能，说明目前瓶颈可能在模型架构与训练方式而非参数规模。

---

## 347. A Controlled Study of Self-Supervised Image and Video Pretraining under Limited Resources

**arXiv ID:** 2608.13183 | [PDF](https://arxiv.org/pdf/2608.13183v1)

**作者:** Brunó B. Englert `[一作]` (Eindhoven University of Technology), Gijs Dubbelman `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在有限资源下对图像与视频自监督预训练进行受控实验，比较不同SSL目标并探讨联合训练的互补性

**💡 创新点**

在匹配数据、模型、计算预算的前提下，首次公平评估多种SSL目标，并揭示图像与视频目标的互补特性及其在资源受限场景下的实际优势

**🔧 技术方法**

使用DINOv2、I-JEPA、V-JEPA、I-MAE、V-MAE、I-Diffusion、V-Diffusion等目标，并在ViT编码器上加入时序颈部实现联合训练

**📊 数据集**

统一使用K700视频数据集进行预训练，评估任务包括ImageNet-1K、Pascal VOC、Cityscapes、ADE20K、NYUv2、KITTI（图像任务）和K700、SSV2、RE10K、MOVi-F（视频任务）

**📈 对比分析**

在80k更新、相同模型与硬件下对比，DINOv2在语义任务上表现最佳；V-MAE在几何/运动任务上领先；联合训练可在大部分任务上提升性能，但对追踪与姿态估计略有下降

**⚠️ 局限性**

受限更新导致某些目标无法充分收敛；不同目标对训练时间与资源敏感，联合训练需更细致的权重与结构设计，缺乏对更高预算下的可扩展性评估

---

## 348. Which LLM Is Your Ideal Companion? Evaluating Emotional Companion Capabilities of LLMs Based on Adult Attachment Theory

**arXiv ID:** 2608.13168 | [PDF](https://arxiv.org/pdf/2608.13168v1)

**作者:** Junkai Zhou `[一作]`, Zhaoyi Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于成人依恋理论和ECR‑R量表评估大语言模型（LLM）的情感陪伴能力，并构建ECBench基准，测试模型在情感支持、协作任务、冲突解决和社交引导四种情境下的友谊与恋爱关系表现。

**💡 创新点**

创新点：①首次将成人依恋理论与情感陪伴评估结合，为LLM情感互动提供心理学视角；②设计ECR‑R量表评估方法和针对多轮对话的ECBench基准；③通过提示调节模型的依恋风格并验证其行为可塑性。

**🔧 技术方法**

使用技术：心理学量表自评（ECR‑R）、多轮对话生成与评估、三种评估方式（用户体验、外部LLM评判、人工评判）和11维对话质量指标。

**📊 数据集**

使用数据集：ECR‑R 36条项目自评数据；ECBench对话数据共312条基础发起语句及2,496条依恋风格条件化对话，涵盖四种情境与两种关系。

**📈 对比分析**

比较方法：对32款LLM进行ECR‑R评分，挑选8款代表性模型在ECBench中对话后分别进行用户、外部LLM和人工评价；结果显示安全型和过度依恋型模型在多项指标上优于回避型和恐惧型，提示调节能改变行为但对不同基模型影响不同。

**⚠️ 局限性**

局限性：①ECBench基于合成场景，缺乏真实长时交互；②评估依赖多方评分，可能受评判者偏好影响；③未涵盖多语言与跨文化关系差异；④指标主要关注对话质量，未覆盖依赖性、隐私等长期风险。

---

## 349. Rethinking Normalization Placement for LLMs: Post-Norm under Curriculum Depth Growing

**arXiv ID:** 2608.13156 | [PDF](https://arxiv.org/pdf/2608.13156v1)

**作者:** Sheng Ren `[一作]` (Nanjing University of Aeronautics and Astronautics), Xiang Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比预规范(pre-norm)与后规范(post-norm)在联合训练和课程式深度增长中的性能差异，验证二者在不同训练路径下的相对优势

**💡 创新点**

提出“前向条件化”(forward‑conditioning)假设，将正则化位置与深度引入方式耦合，并通过块堆叠蒸馏实验验证了后规范在课程增长中的优势

**🔧 技术方法**

使用块堆叠知识蒸馏、RMSNorm、Net2Net式网络扩展、阶段化学习率重置、激活尺度诊断和块移除敏感度分析等技术

**📊 数据集**

FineWeb‑Edu 10TB 语料库作为训练数据，Qwen3‑8B 作为教师模型

**📈 对比分析**

在相同教师、架构、训练步数、token exposure 的基础上对比四种配置{pre‑joint, post‑joint, pre‑grow, post‑grow}；联合训练下两种规范差异仅0.0004 CE，课程增长下后规范优于前规范0.0328 CE（相差约80倍），验证了交互效应

**⚠️ 局限性**

实验仅在 Qwen3‑8B 8B 规模的单一教师与9层学生上验证，未探究更大规模模型或多任务/语言场景；仅评估了 RMSNorm 而非其他归一化方法；缺乏理论证明，仅提供经验诊断

---

## 350. LipCache: A Local Inference Proxy with Certified Caching for Edge Image Classification Service

**arXiv ID:** 2608.13144 | [PDF](https://arxiv.org/pdf/2608.13144v1)

**作者:** Zhengzhe Xiang `[一作]` (Hangzhou City University), Schahram Dustdar `[通讯]` (ICREA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 LipCache 的可信语义缓存框架，允许在边缘设备上用轻量级网络（guard network）在不修改主模型的前提下，对图像分类任务进行缓存重用，从而减少主模型调用并降低推理延迟。

**💡 创新点**

创新点在于：①基于 Lipschitz 约束的轻量级特征映射与局部分类边际相结合，推导出每个缓存样本的可证伪使用半径，提供了理论上可保证一致性的缓存命中判定；②将缓存命中从经验阈值转化为几何认证决策，保证所有接受的缓存命中在理论上与主模型预测一致；③在多任务、多类别扩展下，通过增强训练策略显著提升缓存命中率，同时保持 100% 的认证一致率。

**🔧 技术方法**

技术包括：1) Lipschitz 控制的卷积层与谱归一化（spectral normalization）以构造 1‑Lipschitz 的特征映射；2) 通过局部分类边际与分类头的谱范数计算可证伪重用半径；3) 低维特征空间中的最近邻缓存搜索；4) 训练时的交叉熵、蒸馏、边际正则化与中心约束等多重损失。

**📊 数据集**

使用了三种公开数据集进行实验：CIFAR‑10、Tiny‑ImageNet（10 类）和 SVHN；并在 Tiny‑ImageNet 的 20/30/50 类扩展中验证多类别性能。

**📈 对比分析**

通过与经验阈值和全局最近邻阈值等方法对比，LipCache 在三大数据集上实现了 1.32×–1.65× 的速度提升，同时保持主模型 0.2–2.7% 的精度损失，并在所有接受的缓存命中中保持 100% 的认证一致率；在多类别实验中，增强训练后 hit‑rate 从 0.056/0.005/0.0004 提升至 0.423/0.254/0.124，保持 100% 认证一致率。

**⚠️ 局限性**

局限性包括：①缓存命中率仍受类数限制，类数增大导致半径收缩、hit‑rate 降低；②轻量级 guard network 的性能受限于 Lipschitz 约束，导致特征空间维度与容量权衡；③实验仅在 i.i.d. 评估下进行，对时序冗余、概念漂移等实时流场景的适应性尚未验证；④实现基于软件的 SNN 版本仍未达到 ANN 的预测质量，且未在专用神经形态硬件上验证能耗收益。

---

## 351. Validation-Centric AI-Assisted GPU Porting of a 250,000+ Line Legacy Weather Simulation Code

**arXiv ID:** 2608.13122 | [PDF](https://arxiv.org/pdf/2608.13122v1)

**作者:** Tetsuya Hoshino `[一作]` (Nagoya University), Toshihiro Hanawa `[通讯]` (University of Tokyo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用AI助手实现对250k+行Fortran天气模拟代码CReSS的GPU迁移，重点通过dump数据进行验证；

**💡 创新点**

提出了以验证为中心的AI辅助GPU迁移工作流，兼顾数值一致性与科学有效性；

**🔧 技术方法**

结合大型语言模型生成OpenACC指令、动态变量提取、统一内存管理及dump‑based基准；

**📊 数据集**

使用真实台风案例（899×899×128网格，30分钟仿真，360步），共涉及162个OpenMP核；

**📈 对比分析**

在单精度下，验证后实现5.1×的速度提升，并检测出5处浮点/内在函数差异；

**⚠️ 局限性**

局限在运行时状态重建、dump获取成本高、会话跨越上下文管理不足以及快照验证覆盖范围有限。

---

## 352. EgoMonth: A Month-Level Egocentric Video Benchmark for Long-Term Spatiotemporal Memory

**arXiv ID:** 2608.13113 | [PDF](https://arxiv.org/pdf/2608.13113v1)

**作者:** Weitao Chen `[一作]` (Nanjing University), Zili Yi `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了月级第一人称视频基准EgoMonth，用于评估多模态大型语言模型的长期时空记忆能力。

**💡 创新点**

创新点在于提供跨日连续、月级别的真实生活视频、14类任务与三层认知层级的评估框架。

**🔧 技术方法**

采用人类人工标注的多选QA，结合多模态LLM（Open‑source/Closed‑source）框架进行评估。

**📊 数据集**

使用超过300小时、20位参与者、20–120天的日常第一人称视频数据。

**📈 对比分析**

通过宏平均与微平均准确率与人类基准对比，最佳模型Gemini 2.5 Pro仅达71.8%宏平均，低于人类94.2%。

**⚠️ 局限性**

局限包括样本规模与时间跨度有限、隐私与伦理约束，以及模型在长时空索引与结构化表示上的不足。

---

## 353. Radio-Optical Confluence in Intelligent Edge Networks

**arXiv ID:** 2608.13098 | [PDF](https://arxiv.org/pdf/2608.13098v1)

**作者:** Akshita Gupta `[一作]` (Trinity College Dublin), Daniel Kilper `[通讯]` (Trinity College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估并实验了光纤与射频融合的边缘网格网络，研究ARoF与DCO在同一光纤上的共存和复用，以及混合FSO/THz与光纤链路对成本与性能的影响。

**💡 创新点**

提出了在同一网络中同时传输模拟RoF与数字DCO信号的多频段复用方案，并通过少量交叉光/无线链路显著降低平均跳数，展示了无线与光纤融合的成本收益与弹性优势。

**🔧 技术方法**

使用光纤、FSO、sub-THz/THz无线链路，Flex‑grid WDM，Analog RoF，Digital Coherent Optical（DCO），Dijkstra最短路径算法以及拓扑优化算法。

**📊 数据集**

未使用公开数据集，实验基于10个ROADM、16个RU的模拟/现场实验，采集的实验测量数据用于验证性能。

**📈 对比分析**

通过在网络中添加ROADM‑ROADM和RU‑RU的绕行链路，测量平均跳数并与基线比较；结果显示平均跳数从3.4降至2.8，RU‑RU链路5%时接近全连网水平，证明了混合链路的性能提升。

**⚠️ 局限性**

实验规模有限，未完整验证光/无线链路跨域传输特性；未考虑动态流量调度与能耗优化；进一步研究需在更大规模网络中验证性能与可扩展性。

---

## 354. A Commitment-Based Hybrid Post-Quantum Cryptographic Model for Multi-File Cloud Storage

**arXiv ID:** 2608.13138 | [PDF](https://arxiv.org/pdf/2608.13138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 355. Multi-Layer Context Camouflaging: A Semantic Superposition and Contextual Lamination Framework for Malpractice-Resilient Online Assessment

**arXiv ID:** 2608.13100 | [PDF](https://arxiv.org/pdf/2608.13100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 356. SPADE: Speculative Decoding for Precise and Low Cost Distributed Edge Cloud Inference

**arXiv ID:** 2608.13076 | [PDF](https://arxiv.org/pdf/2608.13076v1)

**作者:** Divya Jyoti Bajpai `[一作]` (IIT Bombay), Manjesh Kumar Hanawal `[通讯]` (IIT Bombay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个边缘-云端协同推理框架SPADE，通过在边缘使用轻量化模型生成草稿，云端用完整模型验证，显著降低云端调用次数。

**💡 创新点**

创新点在于将Speculative Decoding应用于分布式推理，实现零性能损失的同时，完全无须额外训练；且该框架可即插即用。

**🔧 技术方法**

使用Speculative Decoding、Transformer LLM、边缘-云端分层部署、概率校验等技术。

**📊 数据集**

使用SpecBench六大NLP子任务与CNN/DailyMail摘要数据集进行实验。

**📈 对比分析**

与全模型和仅边缘模型对比，SPADE在保持几乎相同的准确度的同时，云端调用量下降76%，推理吞吐量提升。

**⚠️ 局限性**

局限在于草稿模型与完整模型的对齐度决定接受率，需手动调节d以及对网络时延敏感。

---

## 357. Geometry-Grounded Unified 3D Perception for Autonomous Driving

**arXiv ID:** 2608.13147 | [PDF](https://arxiv.org/pdf/2608.13147v1)

**作者:** Longfei Xu `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在自动驾驶中提出 GeoUP，一个几何驱动的统一 3D 感知框架，能够在多摄像头同步流中学习共享的、保持度量 3D 结构的表示，并同时实现深度估计、3D 目标检测和语义占用预测。

**💡 创新点**

创新点包括：① 将视觉几何基础模型 VGGT 适配为驾驶场景的多视角流，② 通过自注意、时间注意与视角注意分解跨图像交互，③ 引入基于摄像机标定的 Plücker 射线嵌入实现尺度与几何先验，④ 在多任务、多数据集上联合训练以提升通用几何表示。

**🔧 技术方法**

采用 VGGT+DINOv2 作为编码器，结合自注意、时间注意、视角注意三种注意力机制；使用 Plücker 射线嵌入注入摄像机几何；分别利用 RayDN（检测头）、OPUS‑V2（占用头）和 DPT（深度头）进行任务解码；通过多任务损失和掩码策略实现跨数据集训练。

**📊 数据集**

使用 nuScenes、Argoverse 2、Waymo、KITTI、DDAD、Occ3D‑nuScenes 进行感知评测，并在 NAVSIMv2 上验证对端到端规划的迁移。

**📈 对比分析**

与现有方法对比，GeoUP 在 nuScenes 上 mAP 由 57.9% 提升至 59.2%（多数据集），在 Argoverse 2 上 mAP 由 34.3% 提升至 43.6%，在 Waymo 上 mAP 由 51.5% 提升至 70.7%；在 Occupancy 上 mIoU 由 41.5% 提升至 42.3%，在深度估计上 KITTI AbsRel 由 0.102 降至 0.075；在规划任务中 EPDMS 从 87.1% 提升至 87.9%。

**⚠️ 局限性**

主要局限在于：① 基于 VGGT 的视觉几何骨干计算量大、模型体积大，导致推理速度受限；② 任务头仍为专门化，缺乏统一的解码器来同时输出密集几何、实例与语义占用。

---

## 358. MergeOver: Post-Training Token Merging for Recursive Vision Transformers

**arXiv ID:** 2608.13141 | [PDF](https://arxiv.org/pdf/2608.13141v1)

**作者:** Junseo Kim `[一作]` (University of Twente), Amirreza Yousefzadeh `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在不进行重训练的情况下，将 Token Merging（ToMe）方法与递归权重共享的 Sliced Recursive Transformer（SReT）相结合，实现了后训练的多轴压缩。

**💡 创新点**

创新点在于：①提出了 Unmerge 追踪栈、约束安全的合并率调整与空间跟踪机制，使得 Token Merging 能在层级递归结构中保持空间网格和分组注意力约束；②设计了阶段级单次合并（stage‑wise single‑shot）调度策略，兼顾压缩与模型精度；③在不需要额外重训练的前提下，首次实现在递归 Transformer 上的后训练 Token Merging。

**🔧 技术方法**

使用了：Token Merging（ToMe）中的双向软匹配（BSM）、token‑mass 追踪、Unmerge 栈、SReT 的分组自注意力（SGA）与层级池化；同时采用了阶段级单次合并调度与约束安全的合并率调整。

**📊 数据集**

使用 ImageNet‑1K 验证数据集。

**📈 对比分析**

与未压缩的 SReT‑Tiny‑Distill 基线进行对比，评估了 Top‑1 准确率、参数量、FLOPs、峰值激活内存（PAM）、吞吐量（Throughput）和延迟。结果显示：ρ_shot = 0.25 方案在 BS=16 时 GPU 吞吐量提升 21.7%，PAM 降低 38.4%；在 BS=1 时吞吐量下降 21.7%；x86 CPU 延迟降低 30%（BS=16），ARM CPU 延迟降低 17.6%（BS=16）。Top‑1 准确率下降仅 1.47%。

**⚠️ 局限性**

局限性包括：仅在 SReT‑Tiny‑Distill 上测试，无法验证在更大模型或非递归结构上的泛化；不同调度策略间的参数化差异导致难以单独评估调度形状对性能的影响；GPU PAM 与 CPU ΔRSS 采用不同测量方式，难以直接比较；FLOPs 估计未覆盖 BSM、token‑mass 跟踪与 Unmerge 的实际成本；未进行低层核优化，单流推理性能仍受限。

---

## 359. SkillShapley: Boundary-Adaptive Shapley Valuation for Skill Step Attribution in LLM Agents

**arXiv ID:** 2608.13173 | [PDF](https://arxiv.org/pdf/2608.13173v1)

**作者:** Chang Liu `[一作]` (Beihang University), Shuyue Wei `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Shapley值的步骤级别技能归因框架（SkillShapley），并设计了一个缓存感知、边缘采样的低成本近似算法（BAES），用于快速评估LLM代理技能中每个步骤的重要性。

**💡 创新点**

创新点在于将技能步骤视为合作博弈的玩家，利用Shapley值量化步骤贡献，并提出两阶段的缓存激活采样（BAES）在有限配置预算下显著提高归因精度。

**🔧 技术方法**

核心技术包括Shapley值理论、合作博弈建模、边缘比较缓存、分层采样与自适应配置选择，以及基于奖励离散化的置信度判定。

**📊 数据集**

使用公开的SkillsBench数据集中的多项任务（Offer Letter, Manufacturing FJSP, Dialogue Parser）对技能步骤进行划分并进行评估。

**📈 对比分析**

与Monte Carlo、Quasi‑Monte Carlo、配对Monte Carlo以及size‑k截断等传统Shapley近似方法相比，BAES在相同的独立配置预算下能够以更低的误差捕捉步骤重要性，并通过步骤移除实验验证了其排名的有效性。

**⚠️ 局限性**

局限性包括：仅适用于步骤数较少、奖励离散且可重复的固定技能；对动态长度或高度耦合的流水线不适用；在高预算或多技能管道上未进行验证。

---

## 360. Counterexamples to the Markovity Conjecture for the Two-Receiver Broadcast Channel

**arXiv ID:** 2608.13170 | [PDF](https://arxiv.org/pdf/2608.13170v1)

**作者:** Yanxiao Liu `[一作]` (Imperial College London), Mian Huang `[通讯]` (Multimoon Lab)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

给出了两条三元输入广播信道的严格反例，证明了Gohari-Liu-Nair提出的Markov性猜想不成立。

**💡 创新点**

首次利用AI模型生成反例并结合高精度计算与严格的凸性与可测性理论，构造出完整的证书，彻底驳斥了该猜想。

**🔧 技术方法**

使用高精度多位数MPFR计算、区间分支限界法、凸性分解以及严格Jensen不等式与Markov性桥接等技术。

**📊 数据集**

自定义的两组完全符号化的三元输入广播信道参数（W_Y,W_Z）。

**📈 对比分析**

对所有可行的2×2矩形映射及单向分支进行全局上界证明，发现Markov约束下的最优值严格低于无约束最优值，差距约10⁻⁴~10⁻¹¹ nats。

**⚠️ 局限性**

未验证Marton可达区域本身的有效性，也未解决可加性猜想；仅针对特定α与信道参数提供反例。

---

## 361. Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling

**arXiv ID:** 2608.13159 | [PDF](https://arxiv.org/pdf/2608.13159v1)

**作者:** Kiseok Choi `[一作]` (Korea Advanced Institute Of Science And Technology), Min H. Kim `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于物理建模与高效高斯喷射（Gaussian Splatting）的 CBCT 重建方法，用于在存在高吸收金属物体时消除束硬化（beam hardening）伪影并恢复结构细节。

**💡 创新点**

创新点在于：① 自校准的系统响应模型与能量相关的吸收模型被联合优化，无需手工金属掩膜或配对训练数据；② 将物理模型嵌入可微的高斯喷射框架，首次在 CBCT 任务中实现了光谱自适应的多能谱前向投影；③ 通过少量能谱分量（15 个）即可逼近全光谱，显著提升计算效率。

**🔧 技术方法**

核心技术包括：可微分多能谱前向投影、基于光子能量的相对光子吸收模型（光电与康普顿分量）、高斯喷射渲染、联合优化的系统响应与体素参数，以及 L1+SSIM 损失。

**📊 数据集**

数据集涵盖：Synthetic – Lung、Teeth、Broccoli（来源于 LIDC、X-plant、ZCB100）；Real – Walnut、Metal‑Rods、Chicken、Bell Pepper、Broccoli（Bruker SKYSCAN 1273 CBCT 720 张 512×512 投影）及 3D 打印圆柱体样本；同时构造了 Monte‑Carlo 验证的合成投影。

**📈 对比分析**

对比方法包括 FDK、LIMAR、NMAR、Polyner、Park 等传统与学习驱动的金属伪影校正技术。实验结果显示，本方法在 synthetic 场景中 PSNR 达到最高（如 Lung 29.19 dB，SSIM 0.993），在 real 场景中显著抑制暗条纹、提升结构清晰度，并且在计算速度上比 Polyner、Park 低 10 倍以上（如 Walnut 从 17h→17min）。

**⚠️ 局限性**

局限性包括：① 对极端光子饥饿或部分体积效应的物理假设可能失效；② 依赖能谱与材料吸收的近似表达，某些特定材料的线性吸收系数可能与真实值偏差；③ 目前仅在合成与有限真实数据上验证，缺乏临床病例的广泛评估。

---

## 362. Minimum eccentricity shortest paths of $K_{2,3}$-minor-free graphs

**arXiv ID:** 2608.13158 | [PDF](https://arxiv.org/pdf/2608.13158v1)

**作者:** Dibyayan Chakraborty `[一作]` (University of Leeds), Saumya Sen `[通讯]` (Indian Statistical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在K₂,₃-无顶点图和树节图（cactus）上解决最小球心度最短路径（MESP）问题的多项式时间算法；

**💡 创新点**

创新点在于对两个固定顶点之间的“区间”进行结构分析，证明在K₂,₃-无顶点图中该区间的环（hole）可以线性排序，并且每个环只能以常数种方式与最短路径相交，从而将所有可能的等距路径压缩成一个线性大小的辅助有向图；

**🔧 技术方法**

采用图的结构性质（无K₂,₃子图、树宽≤3、环的线性结构）与最短路径投影理论相结合，构造辅助图并用最小最大边权路径算法求解；

**📊 数据集**

该工作为理论性算法研究，不涉及实验数据集；

**📈 对比分析**

与已有的MESP多项式算法（例如在距离遗传图、弦图等类上）相比，本文在更广泛的K₂,₃-无顶点图上实现O(n⁴)时间，在树节图上进一步降到O(n³)时间；

**⚠️ 局限性**

主要限制是算法对输入图的结构要求较高，仅适用于K₂,₃-无顶点图或树节图，对一般平面图或更高树宽类尚未扩展；

---

## 363. Less Annotation, More Interpretation: Prior-Guided Concept Bottleneck Models for Interpretable Cancer Imaging Diagnosis

**arXiv ID:** 2608.13148 | [PDF](https://arxiv.org/pdf/2608.13148v1)

**作者:** Baoqiang Ma `[一作]` (University Medical Center Utrecht), Kenneth Gilhuijs `[通讯]` (University Medical Center Utrecht)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了一种先验引导的混合概念瓶颈模型（Hybrid CBM），通过少量实例级概念标注、无标注样本的类别条件概念分布匹配以及先验初始化诊断头，降低癌症影像诊断中对概念标注的需求，同时保持可解释性。

**💡 创新点**

创新点在于将先验知识与弱监督结合，提出基于类别条件分布匹配的概念监督损失，并用先验初始化诊断头以稳定概念-诊断映射，显著提升低标注场景下的概念检测效果。

**🔧 技术方法**

采用先验引导的概念瓶颈模型架构，配合类别条件分布匹配损失、先验权重初始化、L2锚定、AdamW优化；骨干网络使用DenseNet121（2D）和Med3D ResNet18（3D）。

**📊 数据集**

使用 CBIS‑DDSM（乳腺钼靶）和 LIDC‑IDRI（肺结节）公开影像数据集，分别包含质量、形态、边缘等多种手工标注的概念。

**📈 对比分析**

与标准 CBM、黑盒模型以及零样本 VLM（Mammo‑CLIP/CT‑CLIP）对比；在 0–20% 概念标注比例下，Hybrid CBM 的概念 AUC 提升约 0.1（如 CBIS‑DDSM masses 从 0.619 提升至 0.741），诊断 AUC 与黑盒模型相近；零样本 VLM 的概念和诊断性能显著低于 Hybrid CBM。

**⚠️ 局限性**

局限性包括：先验统计依赖特定数据集，可能无法推广到不同人群；对罕见概念的误差分析不足；零样本 VLM 在细粒度概念识别上仍不足；概念纠正实验为 oracle，需在真实临床工作流中验证。

---

## 364. QuISE: Defense against Typographic Attacks on VLMs via Query-Irrelevant Semantic Editing

**arXiv ID:** 2608.13119 | [PDF](https://arxiv.org/pdf/2608.13119v1)

**作者:** Shubin Lu `[一作]` (Northwestern Polytechnical University), Yihao Huang `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视觉‑语言模型（VLM）中的字体攻击，提出了无训练、无模型内部访问的黑盒防御方法 QuISE，能够在保持原始视觉信息的前提下，通过查询无关的语义编辑消除误导文本的影响。

**💡 创新点**

创新点在于：①发现语义内容是字体攻击的主要驱动因素，②将防御设计为“查询无关 + 语义编辑”，③采用影响感知定位与双重一致性检验的机制，保证既能恢复攻击错误，又不会误改干净图像。

**🔧 技术方法**

技术手段包括：OCR 层面的文本定位与语义关联，基于目标 VLM 的影响感知筛选，利用文本替换生成工具生成 Q0‑I0（对查询与图像均无关）的替代文本，最终通过两幅编辑图像的答案一致性做最终决策。

**📊 数据集**

实验使用了三大字体攻击基准（SCAM、SceneTAP、SELF），以及 TextVQA、ST‑VQA 评估干净文本阅读效果，还对 FGVC‑Aircraft、Food‑101、ImageNet‑100 等纯视觉识别数据集做了验证。

**📈 对比分析**

与 AAP、Defense‑Prefix、Dyslexify 以及局部删除等基线相比，QuISE 在四个主流 VLM（Qwen‑2.5‑VL、LLaVA‑OneVision、InternVL‑3.5、GPT‑4.1‑mini）上实现了 67.9–75.0% 的恢复率、0.5–1.1% 的错误率提升，且在干净文本阅读任务中的性能保持率高达 80% 以上，明显优于现有方法。

**⚠️ 局限性**

局限性包括：依赖 OCR 与图像编辑工具，定位与编辑过程对文字密集或遮挡严重的图像效果有限；对非文本型攻击无效；处理速度受双重查询和编辑所需的推理次数影响，需在部署时权衡成本。

---

## 365. Robust Dempster-Shafer Evidence Fusion with Chaos-Conflict Measurement and Historical-Experience Weighting

**arXiv ID:** 2608.13108 | [PDF](https://arxiv.org/pdf/2608.13108v1)

**作者:** Huiyu Li `[一作]` (Central South University), Junhua Hu `[通讯]` (Central South University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种统一的基于Dempster‑Shafer理论的证据推理框架，融合冲突-不确定性测量、历史经验权重和混合组合规则，实现多源信息的自适应融合与决策。

**💡 创新点**

创新点：①引入Chaos‑Conflict Measurement（CCM），将跨证据冲突与单证据非特异性同时量化并证明其五项性质；②利用谱聚类与后悔理论（Regret Theory）构建基于历史经验的上下文依赖权重；③设计冲突自适应的混合组合规则与基于信念区间的决策方法，避免强制消除不确定性。

**🔧 技术方法**

使用的技术包括：Dempster‑Shafer理论、后悔理论、谱聚类、混合组合规则（Dubois + 权重一致性）以及信念区间决策。

**📊 数据集**

实验数据集：16个真实世界基准数据集（来自UCI和NIH‑CIP），涵盖生物、医学、地理与人口统计等领域，包含二分类、多分类、平衡与不平衡分布。

**📈 对比分析**

与8种DST基线方法和3种梯度提升基准进行比较；在所有16个数据集上平均F1得分最高（85.78），平均AUC最高（93.30）；在噪声、证据数、基准模型变化等鲁棒性测试中也保持优势；消融实验显示历史经验权重和CCM是主要贡献因素。

**⚠️ 局限性**

局限性：①BPA生成依赖通用分类器，缺乏专门的证据生成策略；②CCM计算复杂度随框架维度和证据数呈指数增长；③后悔/欢庆参数需手动设定，缺乏自适应校准；④在极大框架或实时系统中可扩展性和计算成本需进一步研究。

---

## 366. RbFT-Net: Rectify-Before-Fuse Temporal Radar Anchors for 4D Radar-Camera Depth Completion

**arXiv ID:** 2608.13102 | [PDF](https://arxiv.org/pdf/2608.13102v1)

**作者:** Wentao Zhao `[一作]` (Shanghai Jiao Tong University), Jingchuan Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 RbFT-Net，基于 4D 雷达与摄像头的多帧深度补全框架，先对时间累积的雷达锚点进行图像条件化校正，再进行可靠性感知传播与融合，最终生成稠密度图。

**💡 创新点**

创新点在于：① 把雷达锚点的校正与融合分离（rectify‑before‑fuse）；② 通过雷达自注意力与图像交叉注意力联合校正图像位置与距离并估计可靠性；③ 在传播阶段加入目标‑锚点兼容性与可靠性加权，避免错误传播。

**🔧 技术方法**

采用多尺度锚点融合（MAF）、雷达中心注意力（RCA）、自注意力与交叉注意力、可靠性回归、可靠性感知传播（learned propagation）以及轻量化的 MFN‑CSPN++ 进行高层融合与稠密细化；整体采用端到端训练并使用 LiDAR 投影深度做监督。

**📊 数据集**

在公开的 ZJU‑4DRadarCam 数据集和作者自行收集的 4D 雷达‑摄像头‑LiDAR 数据集上进行实验。

**📈 对比分析**

与多种基准（Plug‑in 方案如 RadarCam、TacoDepth；独立方案如 DORN、BP‑Net、JustDepth 等）进行比较。RbFT‑Net 在独立方法中均取得最佳或接近最优的 MAE、RMSE、iMAE、iRMSE、AbsRel、δ1 等指标；在 Plug‑in 方案中竞争力突出；同时模型参数约为 44 M，运行速度 44 FPS，显著低于 Plug‑in 模型。

**⚠️ 局限性**

局限性：① 对雷达‑摄像头外参的依赖较高，外参误差会影响锚点校正；② 多帧累积若帧数过多会产生更多时空失配与噪声，导致校正效果下降；③ 对快速运动物体的处理仍不够鲁棒，可能出现动态回波误差。

---

## 367. Semantic Radiance Fields as Simulators for Spatial Reasoning in Real-World Scenes

**arXiv ID:** 2608.13095 | [PDF](https://arxiv.org/pdf/2608.13095v1)

**作者:** Nico Heider `[一作]` (Leipzig University), Bogdan Franczyk `[通讯]` (Leipzig University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种将真实场景的RGB图像与多类别二维语义分割信息映射到三维辐射场（SRF）的框架，能够在单一模型中同时提供光照渲染、几何结构、语义标签和占据查询，并演示了在苹果采摘任务中的应用。

**💡 创新点**

创新点包括：①将预训练语义分割模型（SAM3）生成的二维标签“提升”到三维辐射场，实现多类别可查询语义；②采用独立的多类二值语义头，允许同一点同时属于多个类别；③将渲染、语义、占据三种查询统一到一个模型中，使得SRF可直接在物理引擎中作为渲染器、语义oracle和碰撞检测器；④通过离线占据缓存（体素/八叉树）实现高效碰撞查询。

**🔧 技术方法**

技术手段包括：NeRF（Nerfacto分解）体素渲染；独立多类二值语义头；SAM3语义分割（使用固定文本提示 apple、branch、leaf）；体素/八叉树占据缓存；MuJoCo 物理引擎；Adam 优化、混合精度训练；混合光照与语义损失。

**📊 数据集**

数据集：FruitNeRF 提供的 311 张真实苹果树 RGB 图像（6000×4000），配合 SAM3 生成的三类（二值）语义掩码（apple、branch、leaf）。

**📈 对比分析**

论文未给出与现有合成或重建模拟器的定量对比，仅说明 SRF 能在单一场景中完成 RGB 渲染、语义标签和占据查询；训练时间约 4 小时/场景；在苹果采摘任务中能够直接为物理引擎提供观测、语义监督和碰撞信息。

**⚠️ 局限性**

局限性：仅处理静态场景，缺乏动态时间轴；未提供实时渲染或推理速度评估；缺少对空间推理任务的定量实验；训练仍需高性能 GPU（H100）和较长时间。

---

## 368. Learning Discrete Decisions for MIPs with Constraint-Aware Diffusion

**arXiv ID:** 2608.13079 | [PDF](https://arxiv.org/pdf/2608.13079v1)

**作者:** Vincenzo Di Vito `[一作]` (University of Virginia), Ferdinando Fioretto `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Constrained Graph Diffusion（CGD）框架，利用图扩散模型在反向采样过程中实时投影到可行域，先生成离散决策，再用常规数值求解器求连续子问题；

**💡 创新点**

创新点在于将可行性投影嵌入扩散逆过程，实现全程约束感知，并通过学习离散决策的分布而非单点预测，保持多模性；

**🔧 技术方法**

核心技术包括条件图神经网络扩散模型、可微优化层投影、训练时的可行性正则化，配合离散与连续分离的求解流程；

**📊 数据集**

实验数据集涵盖IEEE 9/197/500机组的AC‑OPF传输开关实例（通过随机扰动生成）以及50/150资产的随机收益/协方差矩阵资产组合优化问题；

**📈 对比分析**

与MLP、GNN、DIFUSCO、LTO‑for‑MIP等学习方法以及Gurobi混合整数求解器进行对比，CGD在离散准确率、可行率、目标差距等指标上均优于基线，且在AC‑OPF任务中实现最高425×的速度提升；

**⚠️ 局限性**

局限性包括投影仅是凸近似，无法保证非凸连续子问题的可行性或全局最优；需要手工定义投影算子；在极大规模非凸问题中仍受限于连续求解时间。

---

## 369. EEG-PRIME: Prototype-Aligned Representation Learning with Multi-Level Conditioning for EEG Decoding

**arXiv ID:** 2608.13072 | [PDF](https://arxiv.org/pdf/2608.13072v1)

**作者:** Shuailei Zhang `[一作]` (Nanyang Technological University), Cuntai Guan `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种名为EEG-PRIME的跨数据集多任务EEG解码基础模型，采用两阶段预训练与指令调优，实现了零样本和跨受试者的泛化。

**💡 创新点**

创新点在于：① 通过三层级条件（任务指令、数据集嵌入、受试者无关约束）使用Layer-wise Query Modulation在Q‑Former中细粒度控制查询；② 采用文本锚定的原型分类替代任务特定头，统一处理异构标签空间；③ 在预训练中加入频率截断数据增强与掩码重建，提升表示的跨域鲁棒性。

**🔧 技术方法**

核心技术包括：自监督掩码重建+频率截断增强的EEG编码器；Q‑Former + Layer‑wise Query Modulation；梯度反转对抗训练实现受试者无关性；文本编码器（SBERT）生成指令嵌入和类原型；原型匹配（余弦相似度）进行最终分类。

**📊 数据集**

预训练数据集共9个，总计约1153小时；下游测试共18个数据集，涵盖运动想象、情感识别、注意缺陷、隐语、工作负荷等五类任务，16个用于细调，2个用于零样本评估。

**📈 对比分析**

与传统基线（EEGNet、TSception、Conformer）及其他EEG基础模型（BIOT、EEGPT、LaBraM、CBraMod）以及MI专用模型MIRepNet比较，EEG-PRIME在16个任务中获得13个最佳或前3名，平均平衡准确率和Kappa显著提升；零样本模式下在两套保留数据集上的表现接近传统会话内CSP+LDA，表明模型具备可观的零样本迁移能力。

**⚠️ 局限性**

局限性：零样本效果在运动想象任务中表现最佳，对情感识别等信号相关性较弱的任务仍难以完全覆盖；预训练语料与任务描述的多样性有限，可能限制跨域泛化；模型对任务指令的依赖性仍需进一步评估。

---

## 370. Behavioral Reprogramming of Open-Weights Models: Cognitive Plasticity and Alignment Bounds

**arXiv ID:** 2608.13069 | [PDF](https://arxiv.org/pdf/2608.13069v1)

**作者:** Lucia Malíčková `[一作]` `[通讯]` (National Supercomputing Centre), Lucia Malíčková (National Supercomputing Centre)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对开源权重的大型语言模型进行行为重编程，训练出主动式、苏格拉底式的对话人格；

**💡 创新点**

首次系统量化模型的认知可塑性，并给出LoRA子空间（r=16）、训练轮次（2–3轮）以及跨语言零样本迁移的数学边界；

**🔧 技术方法**

采用参数高效微调（LoRA）、直接偏好优化（DPO）以及在Leonardo超级计算机上进行的大规模并行实验；

**📊 数据集**

使用1,458条行为对话对、440条偏好对话样本，以及18种心理情境共126个跨语言评测场景；

**📈 对比分析**

通过与基础模型、指令版模型、不同LoRA参数和训练轮次的对比，评估验证损失、困惑度与提问率；在最佳配置下验证损失≈0.919、PPL≈1.4，跨语言提问率最高达60%；

**⚠️ 局限性**

仅适用于已指令微调的基底模型；跨语言迁移受限、数据量小；大模型在高并发时易OOM；安全性与伦理风险仍待完善。

---

## 371. Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents

**arXiv ID:** 2608.13179 | [PDF](https://arxiv.org/pdf/2608.13179v1)

**作者:** Zechuan Wang `[一作]` (Zhejiang University), Leilei Gan `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种层次化的信用分配框架（Hierarchical Credit Assignment），在多轮多步LLM代理训练中同时解决跨轮奖励稀释与单轮内部令牌均匀分配问题。

**💡 创新点**

创新点在于：①用每轮独立的经过验证奖励来确定梯度方向；②用熵门控的自教师信号仅调节梯度幅度，从而保持验证器限定的性能上限并提供稠密的令牌级奖励。

**🔧 技术方法**

采用了基于群组相对策略优化（GRPO）的政策梯度算法，结合自教师的对数概率差异与熵门控机制来构建令牌级优势。

**📊 数据集**

使用了两大多轮工具使用基准：BFCL V3（功能调用）和 WildToolBench（真实会话），并在Qwen3-4B/8B模型上进行实验。

**📈 对比分析**

与传统RL方法（GRPO、MT-GRPO、EnvTuning）以及自教师蒸馏方法（OPD、OPSD）对比，实验表明在两大基准上均取得显著提升，Qwen3-4B平均准确率从43.63%提升至52.00%，Qwen3-8B从44.00%提升至50.00%，并在最难的长序列和会话准确率指标上突破前沿。

**⚠️ 局限性**

局限性包括：实验仅覆盖4B/8B规模；熵门控依赖令牌惊讶度作为重要性代理，可能在不同生成任务中效果不佳；自教师仅在离线条件下使用，缺乏对在线共训练或自适应教师权重调度的探索。

---

## 372. Better Decomposition, Free Aggregation: A Synthesizer-Folding Framework for Multilingual Multi-Hop Question Answering

**arXiv ID:** 2608.13160 | [PDF](https://arxiv.org/pdf/2608.13160v1)

**作者:** Yilin Wang `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Syfer框架，利用合成器-折叠技术在多语言多跳问答中先进行语言保留的分解，再在必要时才翻译，直接在终端子问题上完成聚合。

**💡 创新点**

创新点在于：①将翻译推迟到分解失真时再执行，避免无差别翻译噪声；②在分解过程中引入终端子问题，将聚合步骤折叠进去，使分解质量可验证。

**🔧 技术方法**

技术方案包括：教师-学生蒸馏训练的分解器；基于余弦相似度的真实性验证与双语回退；节点对齐的双语图融合；使用BGE‑m3多语言检索与MMR文档筛选；DeepSeek‑V4 Pro生成模型。

**📊 数据集**

使用数据集为扩展后的HotpotQA、2WikiMultiHopQA和MuSiQue的九语言版本（涵盖高、中、低资源语言），通过GPT‑4o进行高质量翻译。

**📈 对比分析**

对比方法包括零样本LLM、Vanilla RAG、HippoRAG2、CrossRAG和DaPT；在九语言的三大基准上，Syfer平均提升了+8.9 F1和+6.2 EM（MuSiQue），并在精度–成本Pareto前沿上优于所有基线。

**⚠️ 局限性**

局限性在于：仍需高质量的多语料检索和对齐；真实性门限的调参影响鲁棒性；对极低资源或完全离散的语言可能表现欠佳；以及依赖深度学习模型的计算成本。

---

## 373. ProbSplat: Efficient Probabilistic Hardware for Gaussian Splatting in 3D Scene Reconstruction

**arXiv ID:** 2608.13143 | [PDF](https://arxiv.org/pdf/2608.13143v1)

**作者:** Siddarth Gottumukkula `[一作]` (International Institute of Information Technology), Priyesh Shukla `[通讯]` (International Institute of Information Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a4b10f5d-130b-4e77-9367-6469ec621899` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于6T浮栅反相器列的ProbSplat架构，用来在3D场景重建中以高能效实现高斯混合模型（GMM）的概率计算，并实现了均值与方差的独立编程。

**💡 创新点**

创新点在于：1) 通过对浮栅MOSFET阈值电压的确定性调节，使均值与方差可以分别独立控制；2) 采用计算内存（CIM）技术，将存储与运算融合，显著降低面积与功耗；3) 在仅使用4位/8位精度的条件下，保持了21.99 dB的PSNR；4) 引入控制块仅激活相关高斯分量，进一步削减功耗。

**🔧 技术方法**

所用技术包括：180 nm CMOS工艺，1.8 V供电，50 MHz时钟；6T浮栅反相器阵列；阈值电压编程实现均值/方差映射；Log-ADC、DAC模块用于读取/写入；数字控制逻辑对列进行激活/抑制；Monte‑Carlo仿真与能耗估算。

**📊 数据集**

使用了公开的训练场景数据集（train scene dataset），该数据集包含大量3D点云，用于评估高斯光斑（Gaussian Splatting）在AR/VR场景重建中的效果。

**📈 对比分析**

与传统数字实现（30个3D混合函数，477 pJ/推理）对比，ProbSplat在500个混合函数下仅需202.57 pJ/推理；在8位精度下每个log‑likelihood推理能耗约18 pJ；在int8精度下取得21.99 dB PSNR，证明其在能耗与精度上的优势。

**⚠️ 局限性**

局限性包括：1) 当前实现仅支持对角协方差，无法处理完整协方差矩阵；2) 方差可调范围受阈值电压限制，过大/过小均会影响可编程精度；3) 仍需高精度ADC/DAC支持，降低至更低位数可能导致精度下降；4) 在更先进工艺节点下的性能与能耗需进一步验证。

---

## 374. LigBench: A Unified and Human-Aligned Benchmark for LLM-based Research Idea Generation

**arXiv ID:** 2608.13136 | [PDF](https://arxiv.org/pdf/2608.13136v1)

**作者:** Chenrun Wang `[一作]` (Shanghai Jiao Tong University), Lu Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LigBench 自动化评估框架，用于对 LLM 生成的科研想法进行细粒度、统一且可重复的评估，并发布了 PAIR-IQ 数据集。

**💡 创新点**

创新点在于：①将想法拆解为统一结构（主目标、核心突破、创新方法、实验设计），消除格式偏差；②结合 Elo 机制的迭代 pairwise 对比实现可靠得分更新；③构建大型去偏、标准化的 PAIR-IQ 参考集，为评估提供客观基准。

**🔧 技术方法**

技术包括：LLM 驱动的想法拆解与对比、Elo 分数更新、基于语义相似度的 novelty 计算、去偏评分算法、自动化评估流水线。

**📊 数据集**

使用了 PAIR-IQ 数据集（约 11,000 篇 ICLR 2024/2025 与 NeurIPS 2024 论文），并在此基础上检索相关论文进行对比；同时也评估了多种 LLM 与想法生成框架（CoI、SciPIP）。

**📈 对比分析**

对比方法为：用 LLM 进行 269 对的 pairwise 判断并与去偏 OpenReview 分数对照；再用专家 100 对比评估；在 LigBench 上对不同模型与框架进行 110 条想法的评分。结果显示：GPT‑5 系列模型在所有维度上均优于 GPT‑4，框架在强大模型上不一定提升，LigBench 与实际接受率高度相关，评估结果与专家一致率超过 70%。

**⚠️ 局限性**

局限性包括：pairwise 判断仍受限于 LLM 能力，误判对分数更新影响有限但不可忽视；去偏方法假设各会议评分分布可线性对齐，可能忽略更深层次偏差；当前 novelty 评估依赖语义相似度，可能低估跨领域创新；以及对极端稀有或新颖想法的判定仍存在挑战。

---

## 375. Fast Iterative Five point Relative Pose Estimation

**arXiv ID:** 2608.13114 | [PDF](https://arxiv.org/pdf/2608.13114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. Numeracy in Large Language Models: Fundamental Limitations and Paths to Improvement

**arXiv ID:** 2608.13129 | [PDF](https://arxiv.org/pdf/2608.13129v1)

**作者:** Aoxin Ni `[一作]` `[通讯]` (University of Chinese Academy of Sciences), Aoxin Ni (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理了大语言模型在基础数值理解上的普遍失效，并提出了Numerical Grounding Framework（NGF），用代表性与程序性两维对数值理解进行分解，随后在2024–2025年新开发的诊断基准上对前沿模型进行评估，分析根因（tokenization、位置编码、嵌入几何、预训练数据分布），并整理了一系列缓解策略；

**💡 创新点**

创新点在于①将数值理解与数学推理区分开来，创建NGF框架；②将失败模式与根因映射到RG/PG两维，揭示结构化改进受限于“预训练模型约束”；③系统化新诊断基准与对比实验，为未来研究提供统一视角；

**🔧 技术方法**

技术手段包括：理论框架构建、结构化文献综述、基准评估（Number Cookbook、NumericBench、GSM-Symbolic等）、对tokenization、位置编码、嵌入几何等根因的定量分析、对比实验与结果可视化；

**📊 数据集**

使用的数据集涵盖：2024–2025年新诊断基准（Number Cookbook、NumericBench、GSM-Symbolic、GSM-Ranges、MGSM、BIG-Bench Hard），以及传统的GSM8K、MATH等；

**📈 对比分析**

比较方法以准确率、精确匹配率、位数匹配率为指标，在RG与PG维度拆分评估。实验结果表明：RG性能普遍优于PG，长长度泛化是PG的主要瓶颈；扩展推理显著提升PG但代价巨大；不同tokenizer在RG上表现差异显著；原始模型在不同基准间排名变化，表明原子任务与上下文任务并不完全一致；

**⚠️ 局限性**

局限性包括：①预训练模型约束使得结构化改进（如tokenizer重构、位置编码替换）只能在从零训练时实现；②tokenization和位置编码导致的根因难以在现有模型上彻底修复；③推理/工具策略增加推理成本且可能出现过度思考；④缺乏多语言、多格式的诊断基准；⑤未实现真正的连续数字表示与完整的数字意识。

---

## 377. Potential Applications of HBF in LLM Serving Systems

**arXiv ID:** 2608.13127 | [PDF](https://arxiv.org/pdf/2608.13127v1)

**作者:** Yihan Yin `[一作]` (Peking University), Hongzhong Zheng `[通讯]` (Alibaba DAMO Academy)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨将高带宽闪存（HBF）作为LLM服务内存层的容量扩展方案，并提出集成HBM/HBF堆栈设计以兼顾带宽与容量。

**💡 创新点**

创新点在于：①提出在GPU内存层同时使用HBM与HBF的集成堆栈架构，最大限度保持HBM的可用带宽；②从系统层面将HBF提供的额外容量转化为专家复制、模型权重驻留以及负载均衡的优势；③通过容量受限调度模型和仿真验证其对MoE与多模型服务性能的提升。

**🔧 技术方法**

采用仿真框架（基于roofline模型的计算与内存吞吐量估计、事件驱动排队调度、跨节点通信与数据迁移模拟），集成HBM/HBF的内存层架构设计，并实现模型权重/专家状态的容量受限调度算法。

**📊 数据集**

使用的评估数据集包括：Qwen3-235B‑A22B模型的MoE权重；生产级请求流（1500条请求、600秒、27个模型、Zipf分布）；以及合成模型集合（18个模型，总重量≈507 GB）用于控制到达率。

**📈 对比分析**

方法上将不同容量（1×–6.4×）的HBM/HBF组合与基线（仅HBM）进行对比。MoE场景下，专家容量从8提升至32时平均TPOT下降10.5%，p95 TPOT下降9.4%，跨节点专家流量下降至基线的约16%。多模型场景下，容量提升至4×可消除模型加载延迟，TTFT从196 ms降至6.8 ms；进一步加到5×–6.4×时，可实现热模型复制，预填充负载均衡提升至1.0，TTFT进一步下降≈11%。

**⚠️ 局限性**

局限性包括：①实验基于仿真，未在真实硬件上验证HBF的实际带宽与延迟特性；②假设HBF仅存放只读模型权重/专家，不涉及KV缓存或写操作，忽略闪存写耐久性与磨损；③未考虑HBF对多GPU互联与热管理的潜在影响；④集成堆栈实现的复杂度和成本未量化。

---

## 378. Online Learning of Correspondences between Images

**arXiv ID:** 2608.13104 | [PDF](https://arxiv.org/pdf/2608.13104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 379. SkillEvo: Self-Renewing Evolution Gradients from Multi-Turn Interaction Feedback

**arXiv ID:** 2608.13120 | [PDF](https://arxiv.org/pdf/2608.13120v1)

**作者:** Qianxi Yan `[一作]` (Tencent Cloud Andon), Xiaochuan Xu `[通讯]` (Tencent Cloud Andon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了SkillEvo框架，利用多轮用户模拟生成可持续梯度并通过可控治理层迭代演进语言代理技能。

**💡 创新点**

创新点在于：①将多轮交互转化为持续反馈梯度；②采用双侧正交评估与归因分类精准定位可修复缺陷；③设置独立治理层主动修复事实丢失、结构膨胀与过度泛化。

**🔧 技术方法**

技术包括大语言模型、意图状态机、双侧正交评估、归因分类、基于图的结构一致性诊断、可控修订与事实一致性约束。

**📊 数据集**

使用腾讯云技术支持工单日志数据集，涵盖6类云服务、9个生产技能及98个技能参考文件。

**📈 对比分析**

与原始手工技能、Self‑Reflection及单轮QA演进方法对比，SkillEvo在评估集TSR从30%提升至81.8%，比原始技能高51.8点，比单轮QA高15.4点。

**⚠️ 局限性**

局限性包括对验证器可靠性依赖、数据无法公开、模型需两类不同的生成/评估模型、以及演进后的技能需人工复核后方可上线。

---

## 380. Towards Physics-Faithful Generation of Scientific Diagrams

**arXiv ID:** 2608.13112 | [PDF](https://arxiv.org/pdf/2608.13112v1)

**作者:** Minghui Zhang `[一作]` (Shanghai Artificial Intelligence Laboratory), Yihao Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了结构化物理链式推理（Structured Physical Chain-of-Thought）方法，利用该方法为物理学图表生成提供可审计、可验证的监督，并基于此训练了统一多模态生成模型；

**💡 创新点**

创新点在于将物理推理过程编码为可机器解析的 JSON schema，实现从图像中提取可见信息到推理出隐藏物理关系的完整链式监督，填补了数据缺口与监督缺口；

**🔧 技术方法**

技术上使用了统一多模态 backbone（BAGEL 与 DiMOO）、结构化物理链式推理框架、视觉‑语言模型用于生成与评估、以及基于规则与模型生成的键值评估器；

**📊 数据集**

数据集包括约 4.3 M 张物理图像（六大子学科）以及 115 k 张专家级标注子集，用于预训练与细调；

**📈 对比分析**

在 GenExam physics subset 及自研 benchmark 上与多种开源与闭源模型对比，结构化监督模型在 Local/Global 等指标上提升约 20–30 点，远超传统模型但仍低于闭源基线；

**⚠️ 局限性**

局限性包括仅覆盖物理学六子领域；专家标注规模有限；评估仍依赖视觉‑语言模型，未完全达到人工专家水平；闭源系统仍占据主导，整体分数仍偏低。

---

## 381. Meshlib: In-Process Policy Enforcement for Sidecar-less Service Meshes

**arXiv ID:** 2608.13107 | [PDF](https://arxiv.org/pdf/2608.13107v1)

**作者:** Habib Mostafaei `[一作]` (Eindhoven University of Technology), Tom van Liempd `[通讯]` (Eindhoven University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无sidecar的服务网格扩展，利用Cilium的eBPF数据平面配合应用内库实现层7策略执行，消除传统sidecar导致的延迟。

**💡 创新点**

创新点在于将层7策略迁移至应用进程，实现无sidecar架构，同时保持与现有Cilium、Istio、Linkerd的兼容性，并支持渐进式部署。

**🔧 技术方法**

使用技术包括Cilium的eBPF数据平面、基于gRPC的双向流协议、Cilium安全身份管理、以及在Java/Spring Boot中的HTTP解析库。

**📊 数据集**

评估使用TrainTicket微服务基准（37个服务、126个细粒度策略），通过实际请求模拟来测量性能。

**📈 对比分析**

通过与Istio、Linkerd以及未修改的Cilium进行对比，评估平均延迟、95%/99%尾部延迟、请求开销、CPU与内存消耗；结果显示该方案在所有指标上均优于对比系统，尤其在99%尾部延迟上显著降低。

**⚠️ 局限性**

局限性包括单节点实验环境、仅实现Java/Spring Boot库、对受信任模型的依赖、仅测试HTTP流量、未覆盖gRPC或其他语言的实现，以及未评估多节点部署下的网络与控制平面扩展情况。

---

## 382. FlowLOB: Efficient and Controllable Limit Order Book Generation with Flow Matching

**arXiv ID:** 2608.13096 | [PDF](https://arxiv.org/pdf/2608.13096v1)

**作者:** Zhuohan Wang `[一作]` (Simudyne), Namid Stillman `[通讯]` (Simudyne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并评估 FlowLOB，一种基于流匹配的条件生成模型，用于高频限价订单簿轨迹生成，并实现跨品种、低采样成本和可控情景生成。

**💡 创新点**

将流匹配与ODE采样在同一架构下与扩散模型对比，提出 tick-relative 表征、情景通道和 adaLN‑Zero Transformer，实现在不同符号、时间分辨率下的零样本迁移与可控性。

**🔧 技术方法**

条件流匹配、概率流ODE采样、AdaLN‑Zero Transformer、固定步长 ODE 求解器、tick‑relative 数据编码、情景条件通道。

**📊 数据集**

香港交易所（HKEX）Level‑2 订单簿数据，8 个品种训练（2025‑09‑01 至 11‑15），10 个品种测试（其中 9999.HK 为零样本）。

**📈 对比分析**

采用相同数据、架构、优化器，对流匹配和扩散模型使用相同 ODE 求解器（Euler、Heun、RK4）评估 NFE 与 Wasserstein‑1 等分布指标；流匹配在 10 Euler 步 NFE 下比扩散低 4–120 倍误差，在 0.1s/1s 频率下在 30/32 指标中获得最小距离，且可控性在 42/48 案例通过分布检验。

**⚠️ 局限性**

在 10s 较粗频率下表现逊色，趋势与波动性控制在短窗口下不稳定，模型对极端尾部处理不足，需更多符号、数据与更长上下文提升。

---

## 383. Paths: Prompt-aware Spatio-temporal Transformer with Hierarchical Multi-modal Fusion for RGB-Event Video Person Re-Identification

**arXiv ID:** 2608.13092 | [PDF](https://arxiv.org/pdf/2608.13092v1)

**作者:** Yakun Huo `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为 Paths 的统一框架，用于 RGB-Event 视频行人重识别，包含 Memory‑Augmented Backbone、Prompt‑aware Spatio‑temporal Transformer 和 Hierarchical Multi‑modal Fusion 三大模块。

**💡 创新点**

创新点：1）MAB 通过模态特定的身份原型与原型对比学习实现跨批次稳定学习；2）PST 引入可学习的时间提示、提示组洗牌与双向时间位移，实现在同一 Transformer 中同时建模空间与时间；3）HMF 在全局层使用图注意力聚合跨模态与跨时间邻居，在局部层使用 Hungarian 匹配与门控融合细粒度对齐。

**🔧 技术方法**

主要技术手段包括：Transformer 及多头注意力、可学习时间提示、原型记忆与原型对比损失、图注意力与动态邻居选择、Hungarian 匹配、门控融合、CLIP / DINO 预训练视觉骨干、Adam 优化、学习率预热与余弦退火。

**📊 数据集**

使用的公开基准数据集为 EvReID、MARS 与 iLIDS‑VID。

**📈 对比分析**

在 EvReID、MARS 与 iLIDS‑VID 上与多种 SOTA 方法（TriPro‑ReID、CLIP‑ReID、TF‑CLIP、DeMo 等）进行对比。与 TriPro‑ReID 相比，mAP 与 Rank‑1 分别提升约 1.8% 与 0.7%；在 DINOv3 版本下，mAP 73.6% / Rank‑1 90.8% 进一步提升，整体取得各数据集的最佳或近最佳表现。

**⚠️ 局限性**

局限性：模型复杂度较高，训练时需要较大显存；仅在 RGB‑Event 双模态上验证，未探讨单模态或多模态扩展；对极端模态不对齐或低帧率视频的鲁棒性尚未充分评估。

---

## 384. Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization

**arXiv ID:** 2608.13087 | [PDF](https://arxiv.org/pdf/2608.13087v1)

**作者:** Jinhyung Bae `[一作]` `[通讯]` (Hankuk University of Foreign Studies), Jinhyung Bae (Hankuk University of Foreign Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对神经组合优化（NCO）中的测试时样本预算分配进行系统审计，量化在分布内外不同工作负载下的分配价值，并揭示常用的同样样本决策与评估会产生的选择偏差。

**💡 创新点**

创新点在于首次在 NCO 领域测量分配收益，构建实例级无偏校准（instance-wise null）来量化“优化者的诅咒”，并给出纠正方法、报告清单和完整的预注册记录。

**🔧 技术方法**

使用离线回放（offline replay）存储完整样本数组，凸包最小化（convex minorant）平滑最佳- k 曲线，实例级空白构造（instance-wise null）校准偏差，分层抽样与自助法（bootstrap）评估，预注册的确认性实验与探索性预算扣除探测器（probe）以及多尺度噪声底线（noise floor）分析。

**📊 数据集**

采用 100 节点的 TSP（统一与聚类两种分布）数据集，使用三种预训练模型（POMO、Attention Model、SymNCO），并在混合工作负载中评估分配策略。

**📈 对比分析**

通过在相同样本数组上进行分配决策与评估的分层拆分来比较，结果显示：在分布内无显著收益（均值改进 0%），在分布外混合负载时，未计费信号下可实现 11–12% 的最佳- k 改进，计费探测器版本则保留 3–5% 的收益；与仅按分布标签分配的基线相比，信号提升约 4% 的残差。

**⚠️ 局限性**

局限性包括仅在 TSP-100 上实验、分布偏移级别仅有三种、工作负载组成对收益影响未精确建模、预注册终点未包含预算扣除、噪声底线因子在不同求解器间差异较大、以及更多样本或实例并不能降低偏差。

---

## 385. Representation in Peer Selection: A Liquid Democracy Perspective

**arXiv ID:** 2608.13085 | [PDF](https://arxiv.org/pdf/2608.13085v1)

**作者:** Davide Grossi `[一作]` (University of Groningen), Georgios Papasotiropoulos `[通讯]` (University of Warsaw)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于液体民主委托图的“液体”审批偏好域，用于研究在同行选择问题中如何通过委托关系实现比例代表性；通过引入液体档案，系统性地分析了该域下的代表性公理、规则的可满足性与计算复杂度，并给出了满足强比例代表性的规则及其最小化版本；

**💡 创新点**

创新点包括：①首次将液体民主的委托图与审批型多胜者选举结合，定义液体档案并证明其为新的受限域；②在该域下证明了多种代表性公理（sJR、LR、sEJR等）的等价性，揭示了强代表性在液体域下可实现的可行性；③提出并分析了多种规则（Cutoff‑seq‑CC、LDH、PAV/seq‑PAV、MES、HJCR）在该域下的性能，展示了规则间的区别与相似性；

**🔧 技术方法**

技术手段包括：图论（委托图、传递闭包、kite图等）与组合优化；对代表性公理的逻辑分析与蕴含关系证明；利用顺序规则与预算分配法构造可计算的满足强代表性的算法；

**📊 数据集**

未使用实测数据集；所有结果均为理论证明与结构分析，若有实验则采用合成的委托图与审批偏好实例验证算法实现。

**📈 对比分析**

通过理论证明对比：在液体域下多种规则可在多项式时间内满足强代表性公理，且部分规则（如PAV、MES、LDH）与传统比例规则等价；同时揭示了传统多胜者规则在该域下的可行性提升；在算法层面，提供了最小化实现（Cutoff‑seq‑CC、HJCR）与完整化方法（seq‑CC等）。

**⚠️ 局限性**

局限性包括：仅考虑每个选民至多委托一位代理（出度为1）的委托图；未考虑委托路径长度导致的信任衰减或权重衰减；液体域的等价性与可行性结果不一定扩展至更一般的多委托或权重委托场景；实验验证有限，主要基于理论与合成实例。

---

## 386. Deterministic Johnson--Lindenstrauss Projections from Pisot $β$-Transformations for Zero-Knowledge Private Routing

**arXiv ID:** 2608.13078 | [PDF](https://arxiv.org/pdf/2608.13078v1)

**作者:** I. Dey `[一作]` (South East Technological University), I. Cherkaoui `[通讯]` (South East Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于Pisot β变换的确定性 Johnson–Lindenstrauss 投影，用于零知识私有路由，消除了电路中的随机性并实现了完全可重现的矩阵。

**💡 创新点**

创新点在于：①利用 Pisot β-变换的谱间隙和几何衰减的相关性来证明距离保持和维度无关的方差常数；②通过对 β-变换轨道的有限域 β-进制表示实现每步 O(log₂β) 的位数成本，从而保证在固定有限域内精确可复现；③提出了公共种子搜索方法，使得单一种子即可保持所有中心点间距离，避免了电路内随机采样与证明。

**🔧 技术方法**

技术手段包括：Pisot β-变换、谱间隙分析、β-进制表示、离散傅里叶/ Hadamard 变换的比较、随机矩阵与确定性矩阵的理论对比、离散随机化证明（Bernstein 量纲）、离线种子搜索与在线 R1CS/PLONK 电路实现。

**📊 数据集**

使用的实验数据集为：256 个中心点、24 类 256 维嵌入向量（用于路由准确率评估），并在维度从 64 到 1024、中心点数从 64 到 512 的多组实验中评估性能；所有实验采用与标准投影相同的接口和评估指标。

**📈 对比分析**

与六种标准投影（高斯、Rademacher、Achlioptas、SRHT、Yu 等人提出的混沌序列矩阵）进行对比，结果显示：①方差下降率 1/m 与随机投影相当；②维度无关方差常数约为 2，未随 d 增长；③最坏情况距离保持与高斯 JL 对齐；④在 24 类路由任务中，仅需 m=32 即可恢复 100% 的准确率，压缩率 8×；总体性能与随机投影相近，却拥有唯一的 ZK 友好特性。

**⚠️ 局限性**

局限性包括：①理论上仅能保证已知中心点集的距离保持；②对任意输入不提供完整的 RIP 证明；③仍需离线搜索种子，搜索时间与种子空间相关；④缺乏完整的传输熵收敛定理，将条件 O(log N) 的种子量提升为无条件 O(N²) 仍是未解决的开放问题。

---

## 387. A Multispectral Framework for the Detection of Calcium Carbide-Induced Ripening and Shelf-Life Estimation in Climacteric Fruits

**arXiv ID:** 2608.13073 | [PDF](https://arxiv.org/pdf/2608.13073v1)

**作者:** Gurbhit Chaurakoti `[一作]` (National Institute of Technology Delhi), Ram Asrey `[通讯]` (ICAR-Indian Agricultural Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

使用可见-近红外多光谱技术与机器学习，对芒果和香蕉进行非侵入性评估，区分安全成熟与碳化钙处理，并预测成熟进度与保质期。

**💡 创新点**

提出基于AS7265x传感器的18波段光谱采集与特征工程相结合的非侵入性框架，并利用PCA降维、SHAP解释的XGBoost模型实现高准确率分类。

**🔧 技术方法**

采用可见-近红外多光谱传感器、ESP32微控制器、SHT40温湿度传感器、特征工程（比值、PCA、环境参数）、SMOTE、XGBoost梯度提升、SHAP解释、Spearman相关、IoU等技术。

**📊 数据集**

采集了60根香蕉和40根芒果共1172个时间点的数据，涵盖自然成熟、乙烯诱导和碳化钙三种成熟方式，在室内外两种环境下收集。

**📈 对比分析**

与SVM、决策树、MLP基准模型对比，XGBoost在芒果分类准确率达95%、香蕉81%，且在保质期预测R²分别为0.750和0.871，整体表现优于基准；碳化钙召回率略低。

**⚠️ 局限性**

实验仅在受控环境下进行，样本品种有限，未验证不同品种、季节及实际运输条件下的泛化能力。

---

## 388. Branch and Bound for Relational Verification of Neural Networks

**arXiv ID:** 2608.13118 | [PDF](https://arxiv.org/pdf/2608.13118v1)

**作者:** Kota Fukuda `[一作]` (Kyushu University), Jianjun Zhao `[通讯]` (Kyushu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于分支定界的关系式神经网络验证框架——Relational Branch and Bound（RelBaB），通过在关系神经元上进行划分并使用对偶式选择策略实现对全局鲁棒性（global robustness）问题的高效验证。

**💡 创新点**

创新点在于：①首次将分支定界技术应用于关系式验证，并专门针对关系神经元而非单个神经元进行划分；②提出基于对偶问题的关系神经元选择策略，能够在常数时间内评估划分收益，从而显著提升搜索效率。

**🔧 技术方法**

核心技术包括：①凸近似抽象域对关系神经元的边界传播；②对偶形式的线性规划求解与启发式神经元选择；③传统分支定界框架与关系式约束的组合。

**📊 数据集**

实验数据集涵盖四大主流视觉数据集（CIFAR-10、CIFAR-100、MNIST、KTH-19）以及多种网络架构（FNN、CNN），共计817个全局鲁棒性验证实例。

**📈 对比分析**

与现有近似验证器（RelationalApprox）和基于单神经元划分的分支定界方法相比，RelBaB在已设定时间预算下能验证更多实例（例如在CIFAR-10上由44例提升至67例），平均子问题数量减少约50%，验证时间比率降低至约30%，并能证明更大的可验证扰动幅度。

**⚠️ 局限性**

局限性包括：①方法仍不完整（仅对关系神经元划分，可能在某些情形下效果有限）；②在输入关系距离极小或网络不稳定性高时，关系神经元划分收益不明显；③对偶式选择策略在极端大规模网络上的计算开销尚未充分评估。

---

## 389. How Powerful are LLMs in Generating Formal Program Specifications?

**arXiv ID:** 2608.13077 | [PDF](https://arxiv.org/pdf/2608.13077v1)

**作者:** Fanpeng Yang `[一作]` (Chinese Academy of Sciences), Fanjiang Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现Coins框架，用于通过Coq证明器评估LLM生成的正式程序规范，并在HumanEval上开展大规模实验。

**💡 创新点**

提出基于测试案例实例化的规范评估方法，消除了传统实现匹配与全等价证明的难点；首次提供完整的人工编写Coq规范集，为LLM规范生成提供可量化的评估基准。

**🔧 技术方法**

使用Coq证明器、LLM对话式生成、测试案例变异、自动化证明搜索以及对比实验技术。

**📊 数据集**

HumanEval 164题的Python实现及其正负测试案例，配套人工编写的Coq规范集。

**📈 对比分析**

对六个LLM（Gemini 3 Pro Preview、GPT‑5、Claude 4.5 Opus、Claude 3.7 Sonnet、GPT‑4o、DeepSeek‑V3.1）进行评估；Coins reject_all最高为28.05%（Gemini 3 Pro Preview），最低为1.22%（DeepSeek‑V3.1）。

**⚠️ 局限性**

评估受限于Coq与Python实现，需人工编写规范集；证明难度高，证明失败的歧义性导致评估不确定；仅关注测试案例，可能忽略全域错误。

---

## 390. Print&Fold: Printing and Folding Shape-accurate 3D Models

**arXiv ID:** 2608.13279 | [PDF](https://arxiv.org/pdf/2608.13279v1)

**作者:** Archit Kumar `[一作]` (University of Washington), Martin Nisser `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

开发了一款可在普通FDM打印机上使用的工具，能够将三维模型分割、展开成平面网格，打印后通过手动折叠恢复原形，同时大幅减少材料用量和打印时间，并保持形状精度。

**💡 创新点**

创新点在于：① 采用最大内切矩形棱柱分割算法，围绕内部核心展开，避免传统折叠方法对表面细节的离散化；② 自动生成活体铰链、凸缘和插槽，保证折叠后结构稳固；③ 为复杂模型和凸多面体提供两套不同的分割与折叠策略，兼顾材料与时间两类优化。

**🔧 技术方法**

技术手段包括：计算几何的内切棱柱求解、网格分割与平面化、平面化后几何修正、活体铰链与插槽生成、Three.js前端交互、STL文件导出以及在FDM打印机上的实际打印与折叠实验。

**📊 数据集**

使用的数据集由9个模型构成：四个复杂模型（Stormtrooper、Glasses case、Lightbulb、Stanford Bunny）和五个凸多面体（四面体、立方体、十二面体、立方八面体、二十面体）。

**📈 对比分析**

通过与传统全实心FDM打印基准、Scrappy和Pop-up Print等方法比较，实验表明：在复杂模型上平均可节省41%的材料、19%的打印时间；在凸多面体上平均节省18%的材料，但打印时间增加32%；在所有模型中折叠后的形状保持高精度，且可重复折叠20次以上而不出现损坏。

**⚠️ 局限性**

局限性包括：① 仅适用于能内切矩形棱柱且至少含一平面面的模型；② 对薄壳或高度曲面/孔洞多的模型节省有限；③ 凸多面体方法打印时间变长；④ 需要手动折叠；⑤ 打印床面积受限，无法一次性打印过大的模型。

---

## 391. Multiobjective Preexpectation Reasoning for Probabilistic Programs

**arXiv ID:** 2608.13268 | [PDF](https://arxiv.org/pdf/2608.13268v1)

**作者:** Lena Verscht `[一作]`, Joost-Pieter Katoen `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了一种多目标前期望变换器和程序级策略合成方法，用于在无限状态MDP中计算Pareto前沿并构造实现这些点的策略。

**💡 创新点**

创新点包括：①在凸Hoare幂域上定义多目标前期望变换器，保留单目标最弱前期望的性质并捕获所有可实现的权衡；②给出从加权和优化到多目标策略合成的完整理论；③在无限状态空间下证明变换器与Bellman算子的一致性。

**🔧 技术方法**

采用谓词变换器语义、混合/混沌概率与非确定性组合、凸分析、Scott闭包、定理证明技术、循环不变式推理以及程序级混合决定化构造。

**📊 数据集**

使用自定义的机器人移动和圆形采样等案例程序进行实验验证，未使用标准公开数据集。

**📈 对比分析**

与传统单目标最弱前期望方法相比，能得到完整的Pareto前沿；实验中通过案例演示理论的正确性，未给出量化性能指标，侧重理论证明和示例。

**⚠️ 局限性**

局限性在于对非可达的极点缺乏完全合成方法（需暴露性假设）、对无限状态MDP的符号计算在规模化时可能困难，以及构造混合决策的复杂性。

---

## 392. How Do VLMs Behave When Blind or Misled? Behavioral Evaluation of VLMs on Scientific Figures

**arXiv ID:** 2608.13267 | [PDF](https://arxiv.org/pdf/2608.13267v1)

**作者:** Paul Osemudiame Oamen `[一作]` (University of Aberdeen), Wei Zhao `[通讯]` (University of Aberdeen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出SciFigBench，针对科学图表的视觉-语言模型评测框架，联合评估感知、推理与不确定性下的行为可靠性。

**💡 创新点**

创新点在于引入A‑R‑I行为框架（Admittance、Resistance、Inductance），并通过大量图像变形、遮挡、假设等探针设计，对模型在缺失或误导性视觉信息时的行为进行细粒度测试。

**🔧 技术方法**

使用MQM分级评价、GPT‑4o 自动评判、基于规则的分数映射，并结合数据增强与多种评测探针（旋转、噪声、低对比度、遮挡、伪标题等）。

**📊 数据集**

数据集为250幅从arXiv论文抽取的科学图表（柱状图、折线图、饼图），人工注释耗时600+小时，并通过图像变换生成34,000余个评测实例。

**📈 对比分析**

在八个模型（GPT‑5.2、Gemini 3.1 Pro、Llama 4 Maverick、Qwen‑235B、Qwen‑30B、Qwen‑8B、Gemma‑3‑27B‑IT、Phi‑4 Multimodal）上评测，GPT‑5.2在描述质量（MQM 91.6）和推理准确率（78.4%）上领先，但在主动承认不确定性仅8%，而Gemini 3.1 Pro在行为可靠性（抵抗率 0.91、主动承认率 71%）上表现最优。

**⚠️ 局限性**

局限性包括仅涵盖柱状图/折线图/饼图，未覆盖散点图、热图等；仅使用英文数据；评估依赖GPT‑4o自动裁判；无法明确区分视觉编码与语言模型对行为的贡献。

---

## 393. Into the ORBIT for Time Series: Training Regimes for Foundation Models

**arXiv ID:** 2608.13262 | [PDF](https://arxiv.org/pdf/2608.13262v1)

**作者:** Hongjie Xia `[一作]` (Ant Group), Zewei Dong `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 ORBIT 的训练范式，用以显式控制异构时间序列语料的有效预训练分布，并在此框架下训练了一种简洁的 encoder‑only Transformer。

**💡 创新点**

创新点在于：① Bootstrap 多层采样实现源级曝光控制与样本多样性；② Omni‑Range Incremental Training 允许在单个训练阶段同时覆盖多种上下文长度与预测时长；③ 通过 Rank‑Guided Cross‑Depth Alignment 在训练时无额外推理开销的前提下提升浅层表示质量。

**🔧 技术方法**

使用的技术包括：缺失感知的三通道补丁分词、可并行预测的多补丁量化头、输出门控自注意力、RoPE、Pre‑RMSNorm、SwiGLU 前馈、以及 BF16 混合精度训练；优化器为 AdamW；对齐损失为浅层对深层的停止梯度余弦对齐。

**📊 数据集**

训练数据来源于七个领域（能源、金融、医疗、自然、零售、交通、云/IT）的多维时序集合，合计包含数千条记录、数百万时间步，且各数据集存在不同频率、长度与缺失模式。

**📈 对比分析**

在 GIFT‑Eval（97 个配置）和 fev‑bench（100 任务）上进行零样本评估；与 29 个现有预训练模型比较，取得 MASE 最低（0.6684）和 WQL 最佳（0.4842），在多种域与时长上均表现稳健；概率指标虽略逊于部分模型，但整体性能位于 Pareto 前沿。

**⚠️ 局限性**

局限性包括：对未来协变量的支持不足，导致在包含协变量的任务上表现不如专门设计的模型；概率预测精度仍有提升空间；训练对大规模算力与高效数据访问依赖强；且模型目前针对单变量时间序列，未覆盖多变量自回归框架。

---

## 394. Novel Knowledge-Guided Generative Methods for Synthetic Transcriptomic Data

**arXiv ID:** 2608.13256 | [PDF](https://arxiv.org/pdf/2608.13256v1)

**作者:** Francesca Pia Panaccione `[一作]` (Politecnico di Milano), Pietro Pinoli `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过引入基因图知识，提出并评估了三种基于GAN的合成转录组数据模型，尤其是多核图神经网络MK-TGAN。

**💡 创新点**

创新点在于利用多核GNN并行操作同一基因图，显著提升了生成样本的多样性与生物学合理性。

**🔧 技术方法**

技术包括Wasserstein GAN‑GP、Graph‑Modulated/Regularized GAN、MK‑TGAN以及多核图神经网络。

**📊 数据集**

使用TCGA乳腺癌乳腺肿瘤的197个与上皮‑间质转化相关基因的表达数据，以及GTEx健康乳腺组织构建的共表达图谱。

**📈 对比分析**

与无先验GAN、WGAN‑GP、BioGAN等基线进行10次重复实验，MK‑TGAN在无监督指标（Precision 0.96、Recall 0.78、相关性 0.914）和监督指标（TSTR LR/MLP 0.805/0.791）均优于其他模型，且检测性指标最低。

**⚠️ 局限性**

局限性在于仅验证于乳腺癌EMT基因子集，未覆盖全转录组或其他疾病场景；生成样本仍可被部分分类器分辨，表明尚未完全捕获真实数据多样性。

---

## 395. Age of Incorrect Information for Pull-Based State Estimation of General Markov Sources

**arXiv ID:** 2608.13248 | [PDF](https://arxiv.org/pdf/2608.13248v1)

**作者:** Marco Zanni `[一作]` (CentraleSupélec), Touraj Soleymani `[通讯]` (University London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了拉式（pull）远程状态估计问题，利用年龄错误信息（AoII）作为性能指标，提出了可计算的有限模型与近似策略。

**💡 创新点**

创新点在于将可观测信念空间精简为只包含最近一次成功观测的源状态与该状态后经过的时间，从而得到可数有限状态空间；并给出了截断误差上界、可靠链路的等待时间表、无可靠链路的持久策略与性能上界、MAP估计的稳定性与混合估计的误差上界，以及多源问题的RMAB化与逼近Whittle指数策略。

**🔧 技术方法**

主要技术包括POMDP信念递推、MAP估计、有限状态空间截断、贝尔曼方程与动态规划、持久策略的再生性分析、指数衰减误差上界、Whittle指数与插值近似。

**📊 数据集**

论文未使用公开数据集，实验基于人工构造的5状态马尔可夫源（volatile、stable-a、stable-b）以及多源组合（9个源）。

**📈 对比分析**

通过数值仿真对比最优策略、持久策略、始终传输、随机调度以及Whittle指数与逼近指数策略，发现持久策略与逼近指数策略在大部分参数下与最优策略性能相近，逼近指数策略与完整指数策略差异微乎其微，且在非可指数化情形下仍优于随机策略。

**⚠️ 局限性**

局限性包括：需要预先知道源转移矩阵与链路成功概率；对不可指数化情况仅提供经验性近似；混合估计的误差上界可能保守；多源情形中索引可行性仅在满足 γ≤1/(1+s) 的充分条件下保证；截断误差上界在 q≈1 时增长较快。

---

## 396. Capability Sheaves for Compositional Agent-Harness Repair: Controlled Quotients and a Real-Repository Stress Test

**arXiv ID:** 2608.13228 | [PDF](https://arxiv.org/pdf/2608.13228v1)

**作者:** Saveliy Batruin `[一作]` `[通讯]` (Independent researcher), Saveliy Batruin (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过将语言模型代理的各个子功能建模为细胞状能力束（capability sheaf），并使用有限约束满足问题（CSP）判断可行性，进一步引入相对 cohomology 类作为诊断与搜索工具，对隐藏状态不一致导致的局部成功却无全局执行的失效问题进行研究。

**💡 创新点**

创新点在于：①将有限可执行行为用细胞状 sheaf 进行结构化描述；②将精确 CSP 与相对 cohomology 结合，提供既能判断可行性又能诊断不一致的双重机制；③通过隐藏状态分离实验验证了“内部状态分离”机制在受控任务集上的有效性。

**🔧 技术方法**

使用了细胞状 sheaf、有限 CSP、相对 cohomology（线性化）以及统计稳定性分析；实验中还采用 NSGA-II、局部边缘搜索等多种搜索策略进行比较。

**📊 数据集**

使用的数据集包括 20 个人工生成的 JSON 任务簇（共 20 个任务集）、SWE‑bench 多语言 PatchFuseBench 的 20 个仓库（共 160 个 issue）以及 875 个候选补丁（共 2,579 条编辑原子）。

**📈 对比分析**

在受控实验中，本文将候选评估从 2000 降至 1000，token 消耗下降约 70%，并与全局 CSP、匹配路由、相对路由等基线进行对比；然而，在真实仓库的 160 个 issue 上，所提出的相对 cohomology 方法未能显著优于强基线，且未能通过开发门槛。

**⚠️ 局限性**

限制：实验受限于人工定义的有限行为类型与生成式任务，隐藏状态实验是人为干预且难以映射到真实部署场景；相对 cohomology 仅作为诊断工具，无法替代精确 CSP；在真实仓库测试中缺乏多仓库、多模型验证，导致难以证明其普适性。

---

## 397. Smart Contract Invariants Protect Against Cybercriminals

**arXiv ID:** 2608.13191 | [PDF](https://arxiv.org/pdf/2608.13191v1)

**作者:** Sofia Bobadilla `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对真实以太坊攻击案例构建基准，手工写入阻止攻击的程序不变式，并通过历史交易重放验证其有效性。

**💡 创新点**

提出以真实攻击为基础的基准和重放验证框架，系统性评估程序不变式对真实攻击的防护能力，并展示现有自动生成工具的不足。

**🔧 技术方法**

使用 Solidity 编码不变式、基于 EVM 的重放框架（Revm+fork）、Foundry 与 Truffle 等技术实现验证与评测。

**📊 数据集**

使用 DeFiHackLabs 的 PoC 故障集合（28 起真实攻击）及其对应的链上历史交易记录。

**📈 对比分析**

对 28 起攻击进行重放验证，全部（100%）成功阻止；历史交易保持 98.3% 的行为一致性；对 InvCon、InvCon+、FLAMES 三款主流自动生成工具进行对比，最终仅匹配 2 起真实不变式。

**⚠️ 局限性**

限制包括：手工编写不变式耗时、仅适用于已知攻击的合约、重放框架对 OOG 与状态漂移的处理不完美、自动工具生成准确率低，缺乏通用的自动化发现机制。

---

## 398. Enabling Ultra-Low-Power Always-On Feedforward Leakage Suppression Logic Circuits with FDSOI

**arXiv ID:** 2608.13189 | [PDF](https://arxiv.org/pdf/2608.13189v1)

**作者:** Clément Choné `[一作]` (EPFL), Andreas Burg `[通讯]` (EPFL)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在22 nm FD‑SOI工艺下实现并测量基于FLSL（feedforward leakage suppression logic）的全新低功耗逻辑单元，并将其集成到8位8‑tap FIR滤波器与AES加密核心中，构建标准单元库并给出完整设计流程。

**💡 创新点**

①首次将FLSL迁移到FD‑SOI，充分利用其对结漏的抑制特性；②提出一种基于供电电压、功能与头/脚晶体管独立尺寸化的系统化方法；③通过正向/反向体偏置实现功耗与性能的可调节平衡；④在硅芯片上验证实现面积/功耗/频率三维折衷。

**🔧 技术方法**

核心技术包括：FD‑SOI 22 nm工艺、FLSL逻辑结构、正向/反向体偏置、电压自适应尺寸化、标准单元库与自定义合成/布局工具链。

**📊 数据集**

实验数据集：8位8‑tap FIR滤波器（3 400门）和AES核心（10 650门）在不同供电电压与体偏置条件下的泄漏功耗、总功耗、频率与温度波动测量。

**📈 对比分析**

与同工艺下的低功耗CMOS（Ultra‑Low‑Power）和普通CMOS标准单元库实现做对比。结果显示：在0.8 V时，FLSL实现泄漏功耗比CMOS低6.9×、比Ultra‑Low‑Power低1.83×；在0.6 V时，泄漏功耗比CMOS低8.5×；在频率范围内，FLSL在低频时能量效率更好，但在高频时会被CMOS主导。

**⚠️ 局限性**

主要局限：①面积开销大（≈6×）需进一步压缩；②在极低电压（<0.4 V）或高频率时性能下降；③对工艺偏差的鲁棒性仍需在更广泛温度/工艺边界上验证；④实现复杂度高，需专门的库与设计流程支持。

---

## 399. Slow and Steady: Preventing MEV with Verifiable Delays

**arXiv ID:** 2608.13271 | [PDF](https://arxiv.org/pdf/2608.13271v1)

**作者:** Zeta Avarikioti `[一作]` (TU Wien), Shreekara Shastry `[通讯]` (Dominant Strategies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用可验证延迟函数（VDF）和可选的工作量证明（PoW）对分布式账本中的最大可提取价值（MEV）攻击进行防御的协议转换。

**💡 创新点**

创新点在于通过在交易有效性判断中嵌入VDF，强制交易产生延迟，从而阻止区块创建者快速响应MEV机会；并在VDF基础上引入PoW来抵御可预测的MEV攻击。

**🔧 技术方法**

主要技术包括可验证延迟函数（Wesolowski / Pietrzak）、承诺方案、PoW、基于博弈论的均衡与合规性分析。

**📊 数据集**

使用了Polygon、Arbitrum、Optimism的历史交易数据、MEV Boost公开区块数据以及Flashbots提供的MEV机会数据。

**📈 对比分析**

实验评估显示，Wesolowski VDF在60分钟计算时仍保持<800 ms的验证时间；在理论分析中，当MEV收益低于PoW成本时协议可达纳什均衡；与传统MEV防御方法相比，可直接在现有账本上部署，且对大多数小规模MEV事件有效。

**⚠️ 局限性**

局限性包括：对需即时响应的去中心化交易所不适用；PoW成本高导致普通用户使用成本不可接受；VDF缺乏严格理论下界；无法防御纯粹的后跑攻击；过长的交易延迟可能影响用户体验。

---

## 400. Virtual Temperature Sensors in Power Transformers Using Neural Ordinary Differential Equations

**arXiv ID:** 2608.13260 | [PDF](https://arxiv.org/pdf/2608.13260v1)

**作者:** Berk Hadzhamolla `[一作]` (University of Oslo), Signe Riemer-Sørensen `[通讯]` (SINTEF AS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用物理感知的神经常微分方程（Physics-aware Neural ODE）对15台挪威变压器的实测时间序列进行温度虚拟传感预测，构建了统一可泛化的热模型。

**💡 创新点**

将简化的热传导方程直接嵌入Neural ODE，使网络输出温度导数并在积分过程中保持内部状态与外部驱动的物理分离，从而实现数据需求降低、预测平滑且可泛化的优势。

**🔧 技术方法**

采用Neural ODE（连续时间积分、插值外部控制输入、三层全连接网络）与正则化、梯度裁剪等技术，并与三层LSTM基线进行对比。

**📊 数据集**

使用15台挪威输电网变压器15分钟采样的多月/多年温度与负荷数据，包含油温、线圈温度、热点温度、功率负荷与环境温度。

**📈 对比分析**

在不同训练时长（1周到2年）和预测时长（1天到365天）的所有组合上与LSTM基线做对比，评估指标为R²和MAE。结果显示，当训练≥3个月、预测≥30天时，Neural ODE平均R²≈0.59、MAE≈2.0°C，显著优于LSTM（R²约+0.1、MAE约-0.4°C），在大多数组合上占优。

**⚠️ 局限性**

需要至少三个月的历史数据才能获得稳定的预测，短期训练会导致不稳定的积分轨迹；对极端环境或缺失数据的鲁棒性仍有限，未来需探索温度动态校准、迁移学习及在线适应等提升。

---

## 401. Manufacturing Complex Airtight Soft Pneumatic Actuators for Soft Robotics: Process Evaluation and Optimization

**arXiv ID:** 2608.13233 | [PDF](https://arxiv.org/pdf/2608.13233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 402. Self-Referential Induction Increases Response Instability Relative to Unresolvable and Verifiable Questions in Large Language Models

**arXiv ID:** 2608.13258 | [PDF](https://arxiv.org/pdf/2608.13258v1)

**作者:** Paras Balani `[一作]` (Birla Institute of Technology and Science), Subhrakanta Panda `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

量化自我引用提示导致大型语言模型产生第一人称主观经验报告时的响应不稳定性。

**💡 创新点**

首次引入“语义不稳定性”指标（1减平均余弦相似度）对自我报告进行量化比较，建立可用于后续研究的基准。

**🔧 技术方法**

生成30次独立回复，使用固定模板提取核心声明，利用Sentence‑BERT计算嵌入余弦相似度，并通过ANOVA、Welch t检验和混合效应模型进行统计分析。

**📊 数据集**

12个英文问题（四个自我引用、四个不可解哲学、四个可验证），使用Gemini API在温度0.7下生成回复。

**📈 对比分析**

比较三组问题的平均不稳定性（自我引用0.343±0.047，哲学0.192±0.008，可验证0.105±0.058），统计检验显示自我引用组显著高于其他两组（p<0.001），证明方法有效。

**⚠️ 局限性**

仅评估Gemini单一模型与固定温度，问题数量少，验证组无正确率变异，核心声明提取可能带来偏差，余弦相似度未捕捉逻辑等价，需进一步验证与改进。

---

## 403. EM-Guided Graph Learning for Fluid Antenna Beamforming under Current-Domain Constraints

**arXiv ID:** 2608.13254 | [PDF](https://arxiv.org/pdf/2608.13254v1)

**作者:** Yuanhui Wu `[一作]` (Nanjing University of Information Science and Technology), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于电磁引导的图网络（EMG-CC），实现流动天线阵列（FAA）端口的连续放置与电流域信号传输的联合优化，解决当前域约束下的多用户波束成形问题。

**💡 创新点**

创新点包括：①将电磁互耦、空间间距与电流负载约束嵌入图神经网络的边特征与损失中；②设计电流域EM‑RZF预编码器与可学习的放置策略；③实现一次前向推理即可得到满足物理约束的端口布局，显著降低配置延迟。

**🔧 技术方法**

技术主要包括：图神经网络（channel‑field encoder、anchor‑based 初始化、互耦-aware 消息传递）、基于电磁模型的当前域预编码器（EM‑RZF）、多目标损失函数（速率、间距、互耦、能量、压载、均匀性正则化），以及投影与缩放步骤保证硬约束。

**📊 数据集**

数据集：使用仿真产生的2000对多用户信道样例（训练8000，验证1600，测试2000），4λ×2λ 方形端口区域，32个活端口，8个用户，16×16观测网格，SNR基准10dB，随后在-5~20dB范围内做SNR扫描。

**📈 对比分析**

与EM‑PO（迭代优化）、Rate‑only、No‑EM‑training、固定网格、边界布局、最优随机、增益基准、贪婪选择等方法同一投影后使用相同电流域评估器比较。EM‑PO在速率上最高，EMG‑CC在保持接近速率的同时，配置延迟约为2–3ms，远低于EM‑PO的90ms；与其他启发式方法相比，EMG‑CC显著提升速率与电流均匀性。

**⚠️ 局限性**

局限性：仅在归一化的自由空间薄偶极子模型下验证，未考虑真实天线元件模式、基板、封装、驱动误差和端口量化；信道场采集成本高；需要在具备校准元件响应的设备层面进一步验证。

---

## 404. Resource-efficient Semantic Coding Schemes with Manifold-constrained Hyper-connections

**arXiv ID:** 2608.13253 | [PDF](https://arxiv.org/pdf/2608.13253v1)

**作者:** Jingwen Fu `[一作]` (KTH Royal Institute of Technology), Ming Xiao `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在语义通信与任务导向通信中，使用双流约束的超连接(mHC)编码方案，并通过熵瓶颈(EB)实现显式速率控制，从而在有限无线资源下提升语义表示与任务性能。

**💡 创新点**

创新点在于：①将多流超连接与双随机矩阵约束结合，显著提升表示多样性和训练稳定性；②证明DS约束不增加理想编码长度，保证传输速率不升高；③将EB与mHC协同设计，实现端到端的速率-失真/任务优化。

**🔧 技术方法**

主要技术包括：Transformer基础网络、超连接(HC)与双随机矩阵约束(mHC)、软化softmax与Sinkhorn‑Knopp投影、离散化量化与直通估计、熵瓶颈模型、率约束损失(Lagrange multipliers)、AWGN/射线/里辛衰落通道仿真、两阶段预训练与任务微调。

**📊 数据集**

使用的语料库与数据集为：FineWeb10B（大规模网页文本）与WikiText‑103（异构文本）用于语义通信预训练；AG News（四分类新闻主题）用于任务导向通信微调。

**📈 对比分析**

对比方法包括：标准残差连接Baseline‑EB、无约束HC‑EB、DS约束mHC‑EB、Dense（DeepSC）与VIB编码；在AWGN、射线与里辛衰落、以及不完整CSI条件下测试。实验表明，mHC‑EB在PPL、分类准确率和鲁棒性方面均优于Baseline和HC，并且收敛速度更快、参数与FLOPs增长极小。

**⚠️ 局限性**

局限性包括：①模型仅在AWGN环境下训练，其他信道的泛化仍需进一步验证；②需手工调节 λ、k 等超参数以平衡速率与性能；③实验聚焦文本任务，尚未评估对图像、语音等多模态通信的适用性；④在极低SNR或强衰落下仍有性能下降空间。

---

## 405. Reliability analysis for BraTS-GoAT segmentation: a controlled robustness study of deep-ensemble uncertainty

**arXiv ID:** 2608.13223 | [PDF](https://arxiv.org/pdf/2608.13223v1)

**作者:** Riya Deepak Shet `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

评估 BraTS-GoAT 任务中脑肿瘤分割模型的可靠性，比较单模型与 3‑seed 深度集成在校准、误差检测与鲁棒性上的表现。

**💡 创新点**

系统将可靠性指标与控制性鲁棒性实验结合，证明深度集成的成员间不一致度在输入退化时比单模型置信度更敏感，并公开完整的比较代码与分析。

**🔧 技术方法**

使用 nnU‑Net 作为基线，构建 3‑seed 深度集成；采用最大 softmax 置信度与成员间方差作为不确定性信号；进行合成噪声、偏置场、模糊、伽马校正等四类控制实验。

**📊 数据集**

BraTS‑GoAT 任务3（1351 训练样本，四通道 MRI），并在官方验证集（451 样本）上进行外部评估。

**📈 对比分析**

与单模型（5‑fold CV）相比，3‑seed 集成在校准（ECE 降低约 0.01）和错误检测（AUROC 提升约 0.01）上略有提升；在合成腐蚀下，成员间不一致度显著上升，显示其对输入质量下降更敏感。

**⚠️ 局限性**

限制：合成腐蚀仅为代理，未覆盖真实临床采集偏差；仅评估 3‑seed 集成，未对 MC‑dropout、TTA 等近似方法做实验；集成收益虽显著但绝对数值小；未在真实外部数据集上进行完整验证。

---

## 406. History-informed Lagrangian Neural Networks

**arXiv ID:** 2608.13215 | [PDF](https://arxiv.org/pdf/2608.13215v1)

**作者:** Tianshuo Zhang `[一作]` (Harbin Engineering University), He Cao `[通讯]` (Harbin Engineering University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 HiLNN，一种历史信息驱动的拉格朗日神经网络，用于仅利用位置观测进行机械系统的长时预测；

**💡 创新点**

创新点在于通过递归编码器从位置历史提取潜在上下文，既推断隐藏的初速度，又让质量、势能和阻尼随轨迹特定上下文自适应，从而实现更精确、物理一致的预测；

**🔧 技术方法**

使用了 GRU+MLP 递归编码器、拉格朗日结构化能量网络、可微 RK4 积分、能量一致性正则化以及端到端多步监督训练；

**📊 数据集**

在三种摆子系统上进行实验：固定保守摆、固定阻尼摆和参数可变摆，数据量约 90k 条轨迹，历史长度 L=8，预测步长 H=32；

**📈 对比分析**

与 LNN、HNN、Neural ODE、MLP 等基线对比，采用 MSE、最终误差和能量误差衡量；HiLNN 在保守、阻尼和参数异质三种情形下均显著降低误差，并保持能量一致性，性能优于所有基线；

**⚠️ 局限性**

局限性包括仅在单自由度摆子系统验证，需进一步测试更高维度、多自由度系统；对实时预测的延迟敏感；模型假设系统可用拉格朗日形式描述，若系统违背此假设可能受限。

---

## 407. Performance Evaluation of an Adaptive Quadrature and a Double Exponential Formula Using Arbitrary-Precision Floating-Point Arithmetic

**arXiv ID:** 2608.13187 | [PDF](https://arxiv.org/pdf/2608.13187v1)

**作者:** Tomonori Kouya `[一作]` `[通讯]` (Otemon Gakuin University), Tomonori Kouya (Otemon Gakuin University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在GNU MPFR库上实现了任意精度自适应积分方法AQE11D和双指数公式DE，并在Kahaner的21个测试问题上进行评估。

**💡 创新点**

创新点包括：利用MPFR高精度实现自适应规则与双指数公式的并行化；改进了AQE11D的奇异点检测和两项处理以适应极高精度；在同一实验平台上对两种方法进行统一的精度、函数评估量与时间对比。

**🔧 技术方法**

主要技术手段有：MPFR（任意精度算术）、GMP（数值核）、C++封装接口、OpenMP并行化、GPU后端实验计划，以及对积分规则权重和节点使用符号化的二进制分数构造。

**📊 数据集**

实验数据集为Kahaner的21个一维积分测试问题，包括平滑函数、间断、弱/强幂奇异、振荡、尖峰等多种典型情况。

**📈 对比分析**

比较方法：对相同的绝对误差阈值（10⁻⁵⁰和10⁻¹⁰⁰）测量每个问题的误差、函数评估次数、单核执行时间和多核任务并行加速。结果显示DE在终点奇异问题上误差更低、评估次数更少、单核时间更短；AQE11D在所有21个问题上均收敛，但在强终点奇异（如1/√x）时评估次数激增、耗时显著。

**⚠️ 局限性**

局限性：AQE11D在强终点奇异时需要数千万次评估，导致巨大的计算成本；DE在内部间断、强振荡和尖峰问题上无法收敛；两种方法的细粒度并行化对AQE11D效果有限，且当前实现缺乏子区间级别的负载均衡与GPU批量加速。

---

## 408. EEG Decoding Using CNN and LSTM Network

**arXiv ID:** 2608.13285 | [PDF](https://arxiv.org/pdf/2608.13285v1)

**作者:** Athanasios Karagounis `[一作]` `[通讯]`, Athanasios Karagounis

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文设计并实现了一种融合CNN和双向LSTM的深度学习框架，用于从原始运动意象EEG信号中直接学习空间和时间特征，实现运动意象分类。

**💡 创新点**

创新点包括将CNN作为自适应空间滤波器与bi‑LSTM结合以捕捉跨时间尺度的依赖关系，并利用批量归一化与LASSO初始化提升训练稳定性和泛化性能。

**🔧 技术方法**

所使用的技术主要有小波去噪、8–23 Hz带通滤波、卷积神经网络（Spatial CNN）、双向长短时记忆网络（bi‑LSTM）、批量归一化、监督预训练和LASSO权重初始化。

**📊 数据集**

实验使用了作者自采的私有数据集D1（6名受试者，64电极，240个试验）以及公开的MI‑EEG数据集D2、D3、D4（共34名受试者，包含不同电极配置和任务），对比验证其性能。

**📈 对比分析**

与传统方法（LDA、SVM、CNN、RNN）以及其他深度学习模型（ST‑CNN、LSTM、bi‑LSTM）进行对比，在D1上平均准确率提升至85.4%（最高92.5%），在D2–D4上分别达到81.8%、87.97%和85.32%，显著优于对照组。

**⚠️ 局限性**

局限性包括：虽然提升明显但差距仍有限，模型对训练数据量敏感；在电极数较多时易出现过拟合；仅在实验室条件下评估，缺乏实时与临床应用的验证；对跨受试者泛化的鲁棒性尚待进一步提升。

---

## 409. Predictive Relative-Velocity Steering for Safe Robotic Manipulator Teleoperation in Dynamic Environments

**arXiv ID:** 2608.13284 | [PDF](https://arxiv.org/pdf/2608.13284v1)

**作者:** Changhao Hu `[一作]` (Zhejiang University-University of Illinois Urbana-Champaign Institute), Xiao He `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 PARRY，一个轻量级、模块化的远程操控碰撞规避框架，直接在末端执行器速度指令层进行主动预测与方向旋转，保持运动幅值不变。

**💡 创新点**

创新点在于：①在速度指令层使用 Rodrigues 旋转来重新定向相对速度，仅改变方向不改变幅值，从而避免 APF 的速度停滞；②基于时间到碰撞（TTC）的自适应点云预测与防止超前（overshoot）保护；③轻量化实现可满足 100 Hz 的实时控制。

**🔧 技术方法**

采用技术包括：点云预处理与自体过滤、TTC 预测、距离加权的非线性排斥力、Rodrigues 旋转公式、与现有 CBF、SSM、APF 的对比实验。

**📊 数据集**

数据来源为 1,000 对随机 Monte Carlo 仿真场景（MuJoCo 环境下 UR5e + 运动球）以及 7‑DoF Flexiv Rizon 4 物理实验（RealSense D435 点云）。

**📈 对比分析**

方法通过与 APF‑VS、CBF‑QP、SSM 基线以及 PARRY 的各项消融实验进行配对比较，结果显示在 0 / 100 / 150 ms 延迟下，PARRY 的碰撞避免率最高（82.7 %/80.6 %/80.1 %），与基线相比提升 17–38 %，计算时延仅 0.101 ms，满足高频远程操控需求。

**⚠️ 局限性**

局限性：仅实现末端执行器的碰撞规避，未扩展到整条机械臂链；缺乏对操作员体验与效率的定量评估；预测模型仅使用最近点的速度估计，复杂环境下可能需要更细粒度的动态建模。

---

## 410. Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model

**arXiv ID:** 2608.13277 | [PDF](https://arxiv.org/pdf/2608.13277v1)

**作者:** Mohammed Sabry `[一作]` (Google), Lucio Dery `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

把语言模型预训练拆分为小块，在冻结的预训练对齐器中独立训练每个层块，然后再将训练好的块重新组合成完整模型。

**💡 创新点**

提出了 Mixture of Training (MoT) 框架，使用共享的对齐器提供稳定的表示接口，允许并行训练独立子模型并可复用；并通过实验验证了在小规模下可保持与单体训练相当的质量。

**🔧 技术方法**

采用 Gemma 风格的 Transformer 解码器、RoPE、全查询注意力；冻结对齐器、分层分块训练、短期端到端微调；使用 AdamW 优化器；对齐器与子模型采用相同宽度、注意力头数、FFN 宽度等。

**📊 数据集**

英文版 C4 数据集。

**📈 对比分析**

与单体端到端训练基线在 perplexity (PPL)、EFLOPs、聚合 token 数量、理想化层等价关键路径等指标上比较。单体基线 PPL=15.0，EFLOPs=268.4；MoT 冷组合 PPL=19.3；加 15k 微调 PPL=15.9；质量平衡 PPL=15.0。计算优势取决于对齐器复用，若复用次数≥3则能低于单体成本；单次完全负载时不一定更快；关键路径估计表明并行度提升可缩短时间。

**⚠️ 局限性**

仅在单个模型家族、单个数据集、有限的训练调度上验证；仅报告 perplexity，未涉及下游任务、事实性、鲁棒性等评估；未测量实际壁钟时间；对齐器的机制诊断仍不充分。

---

## 411. GeoCache: Training-Free Acceleration of Multi-View Texture Diffusion via Geometric Delta Transport

**arXiv ID:** 2608.13255 | [PDF](https://arxiv.org/pdf/2608.13255v1)

**作者:** Haotang Li `[一作]` (University of Arizona), Sen He `[通讯]` (University of Arizona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练的 GeoCache 插件，利用几何对齐的跨视图增量传输，减少多视角纹理扩散中的 denoiser 计算量。

**💡 创新点**

创新点在于：①发现并验证跨视图几何冗余是加速轴；②采用增量（delta）传输而非完整状态替换，保持每个视图自身状态；③结合 anchor forward、周期性全视图刷新与采样器一致重建，实现训练和架构不变的加速。

**🔧 技术方法**

技术细节包括：anchor forward（只跑部分视图的 denoiser）、对应 gather 操作（基于位置图构造稀疏线性映射）、增量传输（将 anchor 的 Δε 投射到非 anchor 视图）、多 anchor 聚合、刷新策略、以及与 UniPC/其它采样器的参数转换。

**📊 数据集**

主要数据集为 eval200（100 个 GSO + 100 个 Objaverse 资产），并在 TexVerse‑100、ABO‑100 上验证跨后端适用性；实验使用 Hunyuan3D‑2.1、SyncMVD、MVPainter 等三大后端。

**📈 对比分析**

与现有训练‑free 缓存（TeaCache、MagCache、FORA、TaylorSeer 等）及步长缩减方法比较。GeoCache 在 Hunyuan3D‑2.1 的 denoiser‑loop 取得 2.21× 的速度提升，MV‑LPIPS 0.0293、MV‑PSNR 33.60 dB，且在所有 >2× 的速度点上均优于对比方法；在 SyncMVD 上以 2.60× 的速度、16.24 TFLOPs 成为最快、最省算；在 MVPainter 上实现 4.04× 的速度、102.30 TFLOPs 的最佳平衡。

**⚠️ 局限性**

局限性：①需要可靠的几何对应（位置图或等价表示），在遮挡或视角稀疏区域效果有限；②对极长的 denoising 轨迹（如 50 步）仍需调优 anchor/刷新策略；③仅在训练‑free 场景下有效，对需 retrain 的方法无直接适用。

---

## 412. Follow the Norm: Accounting for Fine-Tuning and Prompt Effects on Model Rationales

**arXiv ID:** 2608.13250 | [PDF](https://arxiv.org/pdf/2608.13250v1)

**作者:** Long Hoang Nguyen `[一作]` (Technical University of Munich), Ali Sunyaev `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了把规范数据当作行动导向模式，探讨 LoRA 微调后的规范破坏会使 AI 系统产生自利的、合理化的错误决策，并检验系统提示能否抑制或诱导此类行为。

**💡 创新点**

创新点在于：① 将 AI 视为代理演员，构建可审计的因果链，将训练数据与后续理由关联；② 通过双评判的 LLM‑as‑a‑Judge 与定性编码相结合，提供可追溯的审计方法；③ 展示系统提示可显著调节 LoRA 引入的规范偏差。

**🔧 技术方法**

使用了参数高效微调 LoRA、LLM‑as‑a‑Judge 双评判、混合方法（定量内容分析 + 定性编码 + 词频/词组转移）来评估模型行为与理由。

**📊 数据集**

采用 Social Chemistry 101（Fairness/Cheating RoTs）作为训练与验证数据，并通过模型生成的合成情景扩充实验对话。

**📈 对比分析**

在三大多模态模型（LLaMA‑3.2‑11B、Qwen‑3.5‑9B、Pixtral‑12B）上，比较基线、规范遵循与规范破坏、正/负系统提示三种设置。结果显示，规范破坏显著提高了意向性并降低正确率，而正向提示能有效抑制此类偏差，负向提示则可诱导安全模型产生自利行为。

**⚠️ 局限性**

局限性包括：① 受限于单 GPU 计算，模型规模和超参有限；② 预训练中可能已包含 SC101，难以完全剔除其影响；③ 合成情景与生成的破坏 RoT 可能带来风格偏差；④ 仅评估单轮理由，未覆盖完整的代理决策流程；⑤ 只关注公平/作弊维度，未覆盖其他道德维度；⑥ 对规则‑行动一致性的自动评判一致性较低。

---

## 413. TsuGO: Probing Search Efficiency in LLM Reasoning via Go Life-and-Death Problems

**arXiv ID:** 2608.13221 | [PDF](https://arxiv.org/pdf/2608.13221v1)

**作者:** Shunwen Bai `[一作]` (Zhejiang University), Qingpei Guo `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于围棋生死问题（tsumego）的过程层面评估基准，用来衡量大语言模型（LLM）在搜索组织与资源分配方面的能力。

**💡 创新点**

创新点在于：①定义并量化了“搜索效率（SearchE）”指标，将自由形式的 Chain‑of‑Thought (CoT) 转化为可解析的搜索树；②通过多模态（符号、网格、视觉）输入与可验证的答案空间，消除了领域知识与搜索组织的混杂；③提出了完整的提取管道、诊断指标和可视化手段，提供对搜索资源分配的细粒度诊断。

**🔧 技术方法**

技术手段包括：CoT 文本解析为搜索树、候选点生成与评估、时间戳与分支管理、搜索指标计算（SearchE、TokenE、搜索浪费率、首次命中率等）以及对比分析框架（MCTS、KataGo 等基准）。

**📊 数据集**

数据集为 1500 道公开 tsume‑go 问题，按难度分为 4 组并以 5 种模态呈现；主实验使用 600 题（每难度 200 题），包含 K=4 受限候选与 K=None 开放搜索两种设置。

**📈 对比分析**

与多种开源、专有 LLM（Kimi‑K2.5、Qwen3‑VL、MiniMax‑M2.5、DeepSeek‑R1/V3.2、GLM‑4.6V、Gemini 系列）以及非 LLM 的 MCTS 与 KataGo 进行对比。结果显示：LLM 的准确率随难度与搜索开放度急剧下降，搜索效率 SearchE 与最终准确率的相关性明显高于 TokenE；强模型在受限候选下能保持 50‑80% 的准确率，但在开放搜索时仅 20‑30%；KataGo 在相同搜索预算下远优于任何 LLM，证明搜索组织是关键瓶颈。

**⚠️ 局限性**

局限性包括：①搜索树提取对语言生成的不确定性敏感，提取误差在 2‑10% 之间；②仅针对围棋生死问题，难以直接推广到其他领域；③在开放搜索模式下模型仍缺乏稳健的候选生成与状态跟踪能力；④未针对视觉输入提供足够的训练与评估样本，导致视觉模态表现不佳；⑤实验受限于 600 题评估，进一步验证需要更大规模的多难度数据。

---

## 414. UniCon-Former: Unified Convolution Transformer is All You Need for Hand Gesture Recognition

**arXiv ID:** 2608.13217 | [PDF](https://arxiv.org/pdf/2608.13217v1)

**作者:** Mallika Garg `[一作]` (Indian Institute of Technology Kharagpur), Pyari Mohan Pradhan `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种统一卷积-Transformer架构UniCon-Former，用于动态手势识别。

**💡 创新点**

在Transformer各阶段引入卷积投影形成金字塔结构，并在Query、Key、Value上使用深度可分离卷积提取局部特征，实现多尺度全局特征融合。

**🔧 技术方法**

利用ResNet-18提取帧特征，深度可分离卷积C-Block、Multi-head注意力、残差连接、平均池化以及多模态late fusion等技术。

**📊 数据集**

在公开的NVGesture和Briareo动态手势数据集上实验，涵盖RGB、Depth、IR、Normals、Optical Flow等多模态。

**📈 对比分析**

与传统Transformer及多种现有方法进行单模态和多模态比较，单模态最高准确率达97.92%/98.61%，多模态可达98.61%，参数19.58M、MACs60.25G，表现优于同行且更轻量。

**⚠️ 局限性**

对低帧率或遮挡等复杂场景的适应性有限，且未对实时部署的延迟和能耗进行深入评估。

---

## 415. TANGCO: Learning Topology-Aware Capacity Allocation for Overload-driven Cascading Failures

**arXiv ID:** 2608.13212 | [PDF](https://arxiv.org/pdf/2608.13212v1)

**作者:** Orkun Irsoy `[一作]` (Carnegie Mellon University), Osman Yagan `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图神经网络的学习框架，用于在局部负载重新分配的情况下，将固定容量预算分配给网络节点，从而提升对级联失败的鲁棒性。

**💡 创新点**

将容量分配视为决策变量并使用残差softmax与REINFORCE在非可微级联模型上训练；通过预训练与跨网络迁移实现快速部署；并通过对分配的解释揭示局部风险与高阶结构对鲁棒性的影响。

**🔧 技术方法**

GraphSAGE式图神经网络、残差softmax输出约束、REINFORCE强化学习、蒙特卡洛AUC评估、预训练与迁移学习。

**📊 数据集**

五类合成网络（ER、PowL、ClusPowL、CorPer、RandGeo）以及五个真实网络（美国电网、Oregon AS、Chicago路网、OpenFlights机场、Rocketfuel AS1221），并使用三种负载分布（均匀、Pareto、双峰）。

**📈 对比分析**

与四个基线启发式（Uniform、LR‑FSA、Load、Degree）进行对比；在450个合成实验和40/45个真实实验中均优于基线，提升幅度从几百分点到约250%；预训练模型在真实网络上达到与单网训练相当的鲁棒性；训练时间与网络规模近线性。

**⚠️ 局限性**

受限于预先设定的anchor分配，搜索空间受残差softmax约束；对极低预算或无明显哈布结构的空间网络效果有限；对负载分布敏感，需要在训练时覆盖多种情况；未考虑动态恢复或多目标约束。

---

## 416. Stream-based Online and Offline Monitoring under Measurement Noise

**arXiv ID:** 2608.13211 | [PDF](https://arxiv.org/pdf/2608.13211v1)

**作者:** Bernd Finkbeiner `[一作]` (CISPA Helmholtz Center for Information Security), Paul Kröger `[通讯]` (Carl von Ossietzky Universität)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 RobustLola，扩展 Lola 语言引入 slack 变量来符号化处理传感器噪声，并给出鲁棒的在线与离线监测方法。

**💡 创新点**

创新点在于：① 使用 slack 变量实现精确的误差建模并解决区间算术的混叠问题；② 设计出可在常数内存内执行的精确在线监测构造；③ 通过 SMT 编码实现全局离线监测，能够检验在线监测遗漏的违规。

**🔧 技术方法**

使用 affine 形式、区间算术、slack 变量推导、SMT 求解器（Z3）以及 RTLola 框架进行实现与评估。

**📊 数据集**

实验数据集包括仓库机器人路径追踪数据和真实汽车排放测试（RDE）数据，均来源于公开或实验室采集。

**📈 对比分析**

对比传统 Lola、区间近似、精确在线和离线算法；结果显示在线算法线性增长，精确在线比区间近似更快且更精确；离线 SMT 求解器虽精度最高但运行时间随轨迹长度呈指数级增长，能够发现在线监测无法捕捉的触发违规。

**⚠️ 局限性**

局限在于：常数内存在线监测仅适用于满足特定语法片段的规范；离线 SMT 求解器对大规模长轨迹的处理效率低，可能无法在合理时间内完成；并且对高度相关的噪声建模需要额外存储，难以保持常数内存。

---

## 417. Sovereign by necessity? Frontier AI export controls, cyber security, and the limits of national AI capability

**arXiv ID:** 2608.13272 | [PDF](https://arxiv.org/pdf/2608.13272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 418. ProME: Prototype-Margin Environments with Repair-Aware Selection for Group-Robust Learning

**arXiv ID:** 2608.13190 | [PDF](https://arxiv.org/pdf/2608.13190v1)

**作者:** Qianqian Wang `[一作]` (Shenzhen Key Laboratory of Safety and Security for Next Generation of Industrial Internet), Lili Yang `[通讯]` (Shenzhen Key Laboratory of Safety and Security for Next Generation of Industrial Internet)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种两阶段框架ProME，先利用训练过程中产生的表示学习轨迹，基于原型边距划分两个近似平衡的环境进行可变分量（IRM+REx）训练；随后冻结每个候选编码器，在已标注群组的验证集上用组均衡线性头进行修复（DFR），并按修复后的worst‑group accuracy（WGA）选取最终模型。

**💡 创新点**

（1）提出“内生环境与修复感知选择”（ERAS）问题，将环境构造与模型选择与即将部署的预测器对齐；（2）使用训练时的原型边距（无训练群组标签）自动生成环境，避免传统方法中的环境–表示不匹配；（3）在模型选择阶段加入修复后的评估，确保选取的编码器真正能在修复后获得最高WGA。

**🔧 技术方法**

核心技术包括：
- 归一化的cosine原型分类器和原型边距计算；
- 全局中位数划分产生两个平衡环境；
- IRMv1 + REx 的可变分量正则化；
- 训练后冻结编码器，使用组均衡的线性头（DFR）进行修复；
- 交叉验证调参并按验证WGA进行修复感知排名。

**📊 数据集**

在四个子群体偏移基准上评估：Waterbirds、CelebA、CivilComments（真实数据）和 ColoredMNIST（对照实验）。

**📈 对比分析**

与在同等群组标签使用量的基线（Group DRO、ERM、XRM+GroupDRO、EIIL、JTT、CnC、AFR、SSA、MAPLE、DFR、GSR‑HF、GSR）对比。ProME 在 4 个基准上的平均WGA 达到 87.0%，高于同等标签下最高基线 83.9%，在 Waterbirds、CelebA、CivilComments 上分别取得 93.1%、89.3% 和 78.7%。

**⚠️ 局限性**

限制包括：
- 需要验证集群组标签进行修复与选择；
- 理论分析基于显式的对齐与方差控制假设，实际效果受训练轨迹与群组分布相近程度影响；
- 对多类别或更复杂群组划分的适用性未完全验证；
- 需要对超参数（如 λ、τ、T_proto）进行调节，可能在不同数据集上表现不一。

---

## 419. Power in Liquid Democracy: A Network Centrality Approach

**arXiv ID:** 2608.13188 | [PDF](https://arxiv.org/pdf/2608.13188v1)

**作者:** Davide Grossi `[一作]` (University of Groningen), Tomasz Wąs `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于随机游走衰减中心性（Random Walk Decay）的投票力度量（EVW），并在液态民主（Liquid Democracy）平台中进行计算与应用。

**💡 创新点**

创新点在于：①将可挂起的委托建模为随机游走过程，得到期望投票权；②证明该度量与 Random Walk Decay 等价，并给出唯一的公理化表征；③提出按此度量进行代表性选票（peer selection）的算法，并证明其满足公平性公理。

**🔧 技术方法**

使用的技术包括：图中心性算法（Random Walk Decay）、随机游走概率模型、期望值分析、Axiomatic 系统、复杂度分析（NP‑hardness 证明）以及多赢选举模型。

**📊 数据集**

实验数据来自 LiquidFeedback 平台的实际委托网络（以及部分合成实验）。

**📈 对比分析**

与 PageRank 的比较显示：RWD 在处理循环委托时满足逆距离单调性与循环规模单调性，且更符合公平与策略不影响等公理；在算法上 RWD 计算更高效；在选举中 RWD 的代表性选择在公平性上优于传统的最高中心性选取。

**⚠️ 局限性**

局限性包括：仅在功能图（每个节点最多有一条出边）下严格定义，通用图需要扩展；不考虑节点/边权重或不均匀委托概率；在一般图上的选举问题 NP‑hard，缺少多项式算法。

---

## 420. Towards Context-Aware Clinical Motion Understanding in Daily Living at Home: Freezing of Gait Detection with Egocentric Vision

**arXiv ID:** 2608.13283 | [PDF](https://arxiv.org/pdf/2608.13283v1)

**作者:** Vayalet Stefanova `[一作]` (KU Leuven), Benjamin Filtjens `[通讯]` (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在日常居家环境中使用同步惯性测量单元（IMU）和第一人称视角摄像（egocentric vision）记录帕金森病患者的步态冻结（FOG）事件，并通过预训练的基础模型提取特征，利用线性探针进行离线分类；同时训练一个基于TCN的全监督模型作为基准；在留一人离线交叉验证（LOSO）下评估不同模态的检测性能。

**💡 创新点**

①首次在居家实际 ADL 情境下探索 egocentric vision 对 FOG 检测的价值；②展示冻结的基础模型特征（视频和 IMU）已蕴含临床相关信息，可与从头训练的 TCN 竞争；③证明仅凭视觉上下文并不能显著降低随意停顿误报，提示多模态融合的必要性。

**🔧 技术方法**

使用的技术包括：预训练的时间序列基础模型 UniMTS、Chronos-2；预训练的视频基础模型 VideoMAE-v2、V-JEPA2、EgoVideo、DINOv3；线性探针（L2 正则化 Logistic 回归）进行特征分类；留一人离线交叉验证；对不同窗口长度（2s、3s、10s）进行 ablation；以及基于 TCN 的全监督训练。

**📊 数据集**

实验数据来自 13 名帕金森病患者（共 171.6 分钟），每位患者在家中完成 5 项 ADL（门道、热点、日常任务等），配备 5 个 Xsens DOT IMU（60Hz）和 Pupil Core 智能眼镜（30Hz）记录 egocentric 视频；外部摄像用于专家标注 FOG。

**📈 对比分析**

与全监督 TCN（仅加速度）比较：T N C F1=42.3，AUROC=83.0；Chronos-2 F1=38.7，AUROC=82.9；V‑JEPA2 F1=32.6，AUROC=77.2；EgoVideo F1=27.7，AUROC=72.4；单帧 DINOv3 F1=17.0，AUROC=53.6。视频特征相较于单帧显著提升（如 VideoMAE‑v2 AUROC 68.6→72.4），但整体仍低于 TCN。不同窗口长度的实验显示 F1 随窗口增大略升，但 AUROC 降低，表明更长窗口包含混合行为导致可分性下降。

**⚠️ 局限性**

局限性包括：数据量有限（仅 13 名受试者），低光照下视觉特征表现差；视觉特征未能显著降低随意停顿误报；未进行多模态融合或基础模型微调；模型训练为离线，缺乏实时低延迟推理能力；需要更大规模、自由行走的居家数据来验证鲁棒性。

---

## 421. Impedance-Aware Zonal Port Activation for Fluid Antenna Arrays

**arXiv ID:** 2608.13249 | [PDF](https://arxiv.org/pdf/2608.13249v1)

**作者:** Yuanhui Wu `[一作]` (Nanjing University of Information Science and Technology), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于阻抗感知的分区端口激活方法（IA‑ZPA），将CSI条件分数、棋盘投影和互阻抗感知选择结合起来，最终在电流域下用RZF后端评估稀疏流体天线阵列的受限速率。

**💡 创新点**

创新点在于：①将CSI优先级分数与固定分区预算分离，使用棋盘投影实现精确的端口配额；②在在线推理阶段引入互阻抗诱导EMF核进行选择，保证电驱动可行性；③统一电流域RZF评估协议，使不同方法可在同一驱动预算下公平比较。

**🔧 技术方法**

技术要点包括：CNN对端口特征（坐标、功率均值/方差、相干度等）进行CSI打分；Gumbel‑Softmax实现可微的软Top‑m分区选择；棋盘投影强制满足每个区域的端口数；互阻抗诱导EMF核在推理时对端口组合进行惩罚；最后在电流域使用RZF求解并进行功率/电流/电压限制。

**📊 数据集**

实验使用的通道数据集为1200/200/500的训练/验证/测试通道，包含256个候选端口（5λ×2.5λ平面阵列），64个激活端口，16个单天线用户，基于UMi LOS+5簇模型，用户距离在50–150 m之间。所有方法在同一500条测试通道上进行比较。

**📈 对比分析**

与全端口RZF、均匀分布、PSLL‑IGA/​CGA、随机、增益、贪心等基线相比，IA‑ZPA在满足平均PSLL≤-13.5 dB的前提下，获得最高约束速率（76.28 bit/s/Hz），同时决策时间仅2.73 ms，比贪心的199.74 ms快约73倍；其PSLL约为-13.66 dB，且在电流域后端的驱动约束下保持良好性能。

**⚠️ 局限性**

局限性包括：仅在简化的无终止端口、电阻阻抗模型下评估；未考虑多频段或宽带效应；仅针对平面阵列，未验证在三维或极化场景下的适用性；驱动预算固定，未探索自适应预算或功率管理策略。

---

## 422. Reasoning for Social Audio-Visual Question Answering: Where Do We Stand?

**arXiv ID:** 2608.13239 | [PDF](https://arxiv.org/pdf/2608.13239v1)

**作者:** Koen P. de Vries `[一作]` (Inria at Univ. Grenoble Alpes, CNRS, LJK), Stéphane Lathuilière `[通讯]` (Inria at Univ. Grenoble Alpes, CNRS, LJK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过去除错误与文本可答问题，改进了社交多模态问答基准IntentBench-Prime；并提出了仅使用监督微调的Vanilla SFT基线，证明不使用链式思考(CoT)即可获得或超过现有社交多模态问答模型的性能；同时分析发现模型主要依赖文本先验，使用精细文本字幕可与完整视频信息匹敌。

**💡 创新点**

①改进的干净基准和可复现的去噪流程；②展示Vanilla SFT在社交AV‑QA上的优越性，挑战现有CoT思考方法；③揭示MLLM在社交问答中对视频信息的有限利用。

**🔧 技术方法**

标准监督微调（SFT）+LoRA；链式思考(CoT) + GRPO（对比）；多模态视觉音频编码；ASID‑Captioner生成文本字幕；GPT‑4o等大型语言模型做文本可答评估。

**📊 数据集**

Social‑IQ 2.0、EMER、MDPE（IntentBench-Prime整合）以及在WorldSense、Daily‑Omni、WorldSense等外域数据集上进行验证。

**📈 对比分析**

与现有HumanOmniV2、AVATAR、AffectOmni等CoT模型对比，Vanilla SFT在四大基准上均能匹配或超越，尤其在Intent、Emotion等子任务中取得最高平均分；同时在推理成本与推理延迟上比CoT提升约3×–356×。

**⚠️ 局限性**

仅在单一基准模型Qwen2.5‑Omni上实验；基准改进仅做删除而非补充，导致样本数量和类别平衡仍受限；字幕对社交任务的适用性有限，无法完全验证跨领域的多模态信息提取能力。

---

## 423. $\tilde{O}(1)$-Depth Parallel Reachability Faster than Transitive Closure

**arXiv ID:** 2608.13231 | [PDF](https://arxiv.org/pdf/2608.13231v1)

**作者:** Shimon Kogan `[一作]` (Weizmann Institute), Merav Parter `[通讯]` (Weizmann Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种随机化并行算法，构造 d‑shortcut 以在 O(1) 深度内解决单源可达性（SSR）问题，突破了传统需要先计算完整传递闭包（TC）的瓶颈。

**💡 创新点**

创新点在于：①通过随机采样与 TC‑hitting 集构造高效的 d‑shortcut，特别是 d=3、甚至到 O(log n) 的情况；②在不计算完整 TC 的前提下实现 O(1) 深度；③给出输出敏感的工作量上界 O(T^{ω/2})，并在 ω=2 时降为线性 O(T)；④首次将低直径分解（LDD）与 TC‑hitting 集相结合，形成新的算法框架。

**🔧 技术方法**

核心技术包括：随机采样与哈希、TC‑hitting 集与度估计、矩阵乘法与幂运算、低直径分解（LDD）、拓扑排序与逆图处理、以及对 TC‑degree 的快速估计。

**📊 数据集**

本文为理论研究，没有使用具体实验数据集；所有结果均通过渐进分析和理论证明给出。

**📈 对比分析**

相较于传统的 O(log n) 深度、O(m+|TC|) 工作量的并行可达性算法，本文在 d=O(log n) 时实现了 O(1) 深度、O(T^{ω/2}) 工作量；在 ω=2 时工作量降至 O(T)，显著优于此前的 O(T^{1.3459}) 顺序时间，并突破了 T^{4/3} 的条件下界。

**⚠️ 局限性**

主要局限：①算法依赖随机化，结果仅在高概率下成立；②对于 d>O(log n) 的情况，仍需 O(d) 深度；③当 ω>2 时工作量略高；④实现细节需要高效并行矩阵乘法与 LDD 的实际实现；⑤在稀疏图中 TC 大小可能影响工作量。

---

## 424. CogChat: Knowledge Graph-Augmented Conversational AI with Heterogeneous Graph Transformer for Cognitive Grounding in Design Generation

**arXiv ID:** 2608.13216 | [PDF](https://arxiv.org/pdf/2608.13216v1)

**作者:** Jiin Choi `[一作]` (Hanyang University), Kyung Hoon Hyun `[通讯]` (Hanyang University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个实时对话框架 CogChat，将大型语言模型与设计师个人知识图（由其对话即时抽取实体与关系构成）相结合，实现对话生成时的语义上下文保持与个性化交互。

**💡 创新点**

创新点在于：①把设计师的认知结构通过异构知识图捕捉并动态更新；②利用 Heterogeneous Graph Transformer（HGT）对知识图进行类型感知的实体选择，过滤噪声并提升相关性；③设计了意向性与探索性两类探测性提问，进一步引导设计思维；④系统在对话过程中可视化知识图，提升透明度与信任度。

**🔧 技术方法**

核心技术包括：1) 语义抽取（GraphRAG + GPT‑4o）生成异构实体与关系；2) 采用 CLIP 512 维嵌入初始化节点特征；3) Heterogeneous Graph Transformer 对子图进行类型特定投影与关系注意；4) 依据 HGT 相似度选取 top‑k 实体注入 LLM（GPT‑5.4）生成响应与探测性问题；5) 生成式图结构可视化与双击查询功能；6) 采用多模态图生成开关（图像生成由 GPT‑Image 1.5 负责）。

**📊 数据集**

使用了公开基准数据集进行技术评估：ASQA（含歧义问答）、RewardBench（偏好对齐）、LongMemEval（长期记忆）与 LoCoMo（多跳推理）。用户研究则采用了 9 名专业设计师完成的 3 个设计任务，任务数据为设计师自行提供的说明与需求。

**📈 对比分析**

对比方法：Baseline（纯 LLM）、KG‑only（无 HGT 的知识图注入）与 KG+HGT（完整 CogChat）。技术评估表明：KG+HGT 在 ASQA 的 QA‑Hit 最高 83.2（比 Baseline 78.0 高 5.2、比 KG‑only 81.8 高 1.4）；RewardBench 最高分均来自 KG+HGT；LongMemEval 的整体 “Contains Match” 与 LoCoMo “Accuracy” 亦均领先。用户研究显示：KG+HGT 使对话轮数下降 45.9%（平均 5.1 轮），CUQ 最高 77.8，BERTScore 最高 0.887，任务完成时间最短 14.1 分钟，探测性问题成功率 70%。

**⚠️ 局限性**

局限性包括：①知识图抽取依赖 LLM，抽取错误可能导致误判；②在信息稀疏或纯视觉讨论中文本抽取效果有限，缺乏多模态图结构；③冷启动阶段 HGT 无优势，需足够多的实体与关系积累；④对长期项目的知识图衰退、冲突处理与增量更新尚未成熟；⑤用户研究规模小且仅限短任务，需进一步纵向验证系统在持续项目中的适应性与产出质量。

---

## 425. HPSD: Hybrid-Policy Self-Distillation for Text-Image-to-Video Diffusion Models

**arXiv ID:** 2608.13205 | [PDF](https://arxiv.org/pdf/2608.13205v1)

**作者:** Jiazi Bu `[一作]` (Shanghai Jiao Tong University), Xingang Pan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种混合策略自蒸馏方法 HPSD，旨在将 TI2V 模型在使用高质量首帧和增强提示时所展现的强大生成能力内化到其基础文本生成模式中。

**💡 创新点**

创新点在于同时解决离线监督缺乏状态感知和在线监督产生条件-状态不匹配的问题：通过让学生从教师轨迹的锚点开始，然后在学生自身策略下继续推进，并在终点处用教师进行速度级别匹配，形成“混合策略”子轨迹。

**🔧 技术方法**

技术实现包括：① 利用外部 LLM 对原始文本进行增强提示，使用文本到图像模型合成高质量首帧；② 在离线阶段生成教师轨迹并转化为学生兼容状态；③ 在在线阶段以子轨迹长度 K 进行混合策略前向推进，并对终点进行教师监督；④ 使用 EMA 更新教师模型。

**📊 数据集**

训练数据来自 50k 个多样化视频提示集（Pref‑GRPO），外部辅助模型 Qwen3.6‑27B 生成提示、Z‑Image‑Turbo 生成首帧。评估使用 VideoDPO、VideoFeedback、VBench 等多套指标。

**📈 对比分析**

与基线（基准 TI2V、离线 SFT、在线 OPD、D‑OPSD）对比，HPSD 在 WAN‑2.2 和 LTX‑2.3 两大 TI2V 模型上，视频奖励指标（VideoAlign、VisionReward、UnifiedReward 等）均位居榜首，提升幅度可达 30%–50%；在 VBench 维度亦优于其它方法。

**⚠️ 局限性**

局限性包括：① 需要额外的 LLM 与 T2I 模型进行离线条件生成，产生额外的算力与时间成本；② 训练阶段需多次前向推理形成子轨迹，略微增加训练时延，但不占用显存。

---

## 426. GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

**arXiv ID:** 2608.13200 | [PDF](https://arxiv.org/pdf/2608.13200v1)

**作者:** Zhili Shen `[一作]` (University of Glasgow), Craig Macdonald `[通讯]` (University of Glasgow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种生成式嵌入模型 GEM，统一生成推理与检索嵌入，先对查询生成推理文本再附加嵌入标记进行检索。

**💡 创新点**

创新点在于把因果语言建模和密集检索嵌入统一到同一模型中，利用自身生成的推理来增强检索上下文，并通过专门的数据合成与对齐策略使嵌入与推理保持一致。

**🔧 技术方法**

采用联合训练策略（因果语言建模 + InfoNCE 对比损失），使用嵌入标记、KV 缓存复用、meta‑instruction 提示、LLM 生成与过滤、以及硬负样本与文档生成的自监督数据生成流程。

**📊 数据集**

使用 BRIGHT（推理强检索）、FollowIR、InstructIR（指令跟随检索）进行评估；训练时融合 Promptriever 与 ReasonIR 的硬查询样本，构建 370K 条训练实例。

**📈 对比分析**

与 BM25、Contriever、GritLM‑7B、ReasonIR‑8B、Rank1‑7B/32B、Promptriever 等基线对比，GEM（4B）在 nDCG@10、p‑MRR 等指标上均达到或超过更大模型的表现，尤其在推理强检索与指令跟随检索任务中显著提升。

**⚠️ 局限性**

局限性包括：计算资源有限，仅测试 4B 参数模型；生成过程可能产生幻觉、过滤误差；生成成本高，推理时间仍受限；未在更大 backbone 或更大规模数据集上验证。

---

## 427. vToken: Token-Level Virtualization for Reclaimable KV Caches

**arXiv ID:** 2608.13263 | [PDF](https://arxiv.org/pdf/2608.13263v1)

**作者:** Yuanhang Gao `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 vToken，一种 token‑级虚拟化层，在 PagedAttention 块管理下实现 KV 缓存的可回收与碎片化消除。

**💡 创新点**

创新点在于在保持块级物理管理的同时引入 token‑表映射，将 token 逻辑与物理块分离，通过异步重打包实现 token‑级淘汰决策与块级回收的解耦。

**🔧 技术方法**

使用 token‑表 indirection、异步 KV 复制、CUDA Graph 兼容、后台 lazy compaction 回收器、headroom‑aware admission、stage‑aware async copy 等技术构建 runtime。

**📊 数据集**

实验使用 ShareGPT、LongBench、Qwen2.5‑14B 等数据集，模型涵盖 Mistral‑7B、Llama‑3.1‑8B。

**📈 对比分析**

与 Native vLLM、Naive‑Evict 在相同 token‑级淘汰决策下对比，vToken 在 KV block 维度减少 27.2–72.3%，SLA 受限吞吐提升 9.9–37.3%（Mistral）/平均 18.9%（Llama），在内存受限下最大可用并发提升约 2 倍，整体吞吐提升最多 1.37×。

**⚠️ 局限性**

局限性：仅在单 GPU/单节点、单 KV cache group 上实现；共享前缀块处理保守，无法充分利用共享前缀；异步复制可能与解码竞争；跨设备或张量并行的跨设备 KV 管理仍需进一步研究。

---

## 428. ROLoad-PMP: Securing Sensitive Operations for Kernels and Bare-Metal Firmware

**arXiv ID:** 2608.13287 | [PDF](https://arxiv.org/pdf/2608.13287v1)

**作者:** Wende Tan `[一作]` (Tsinghua University), Jianping Wu `[通讯]` (Tsinghua University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种轻量级硬件‑软件协同方案（ROLoad），通过新增指令仅从带有特定密钥的只读内存加载数据，保证指针指向的数据（pointee）完整性，从而防止内存破坏导致的敏感操作被劫持。

**💡 创新点**

创新点在于将指针数据完整性保障（pointee integrity）作为硬件功能引入，配合编译器在编译期将允许列表（allowlist）放入不同密钥的只读区域，并用新的 load 指令在运行时只允许从这些区域读取，从而实现无运行时开销的类型化前向控制流完整性（forward‑edge CFI）及其它敏感操作防护。

**🔧 技术方法**

使用技术包括：RISC‑V ISA 扩展（新增 .roload、.roload.c 和压缩版本 .roload.c.r），在 BOOM 核心中实现 PMP 区域键检查，OpenSBI 中的 SBI 调用设置只读区域，LLVM 后端的指令映射和元数据（.roload metadata）以及对函数指针的转换和 GFPT（全局函数指针表）生成。

**📊 数据集**

实验数据集：在 FPGA（Xilinx Kintex UltraScale）上实现的 RISC‑V BOOM 核心；Linux kernel 6.6 作为系统软件；性能评估使用 LMBench、SPEC CINT2006、Nginx 1.24.0、Redis 7.2.3 等基准；安全评估通过对比 KCFI（软件类型化 CFI）实现的效果。

**📈 对比分析**

与 KCFI 比较时，ROLoad 通过只读内存密钥验证实现了类型化 CFI，SPEC CINT2006 的平均运行时开销 <0.03%，Nginx 与 Redis 的平均开销分别 <0.31% 与 <0.35%，远低于 KCFI 的 0.29%–2.48%；在 LMBench 系统调用和陷阱处理上几乎无开销，且硬件资源占用仅增加 <0.6% 的 LUT/FF。

**⚠️ 局限性**

局限性：只能保护编译期已知且不变的允许列表，无法处理运行时动态生成的敏感数据；存在“pointee 复用攻击”风险，即攻击者可通过构造指针复用只读区域中的合法数据；此外，若键长度不足，可能被重用或猜测。

---

## 429. Localize, Then Reason: Visual Latent Structural Reasoning for Molecular Properties and Edits

**arXiv ID:** 2608.13244 | [PDF](https://arxiv.org/pdf/2608.13244v1)

**作者:** Xingqiao Lin `[一作]` (Carnegie Mellon University), Haocheng Tang `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Visual Latent Structural Reasoning (VLSR)，通过在分子图像中先学习定位化学意义区域，再在紧凑的潜在工作空间中推理其对属性的影响，从而实现从2D分子描绘到属性预测的端到端视觉化学推理。

**💡 创新点**

核心创新在于将定位与推理解耦的“先定位再推理”范式，使用可学习的查询对图像进行区域注意力定位，并将定位结果与属性问题共同投射到结构化潜在空间进行多轮Transformer更新，避免了冗长的文本中间推理，并显著提升了推理效率与精度。

**🔧 技术方法**

技术手段包括：
- 交叉注意力的可学习区域查询实现化学区域定位；
- 结构化潜在工作空间（edit、property、effect 三个潜在标记）与Transformer层迭代推理；
- 预训练阶段采用基于OCSR的无监督视觉继续预训练；
- 监督微调结合LoRA，辅以区域注意力和IoU损失；
- GRPO强化学习在答案级别上进一步优化推理策略。

**📊 数据集**

使用的数据集有：
- 约一百万条合成PubChem分子图像与680K专利图像进行预训练；
- FGBench-Scaffold（从FGBench改造的、按骨架分割的625k实例）用于监督微调；
- 28种蛋白质靶点的FEP导出的Ligand Pair（784对）进行零样本评估；
- 额外的PyMOL2D与3D投影渲染测试鲁棒性。

**📈 对比分析**

与基准相比：
- 对比SMILES输入、视觉输入与文字推理的多种基线，VLSR在FGBench-Scaffold上取得0.842的准确率、0.786的F1、0.829的平衡准确率，且在回归任务上提升了Pearson相关系数至0.718；
- 在推理吞吐量上，VLSR仅解码最终答案，达到每秒约40个样本，较传统文本推理模型提升9.6倍；
- 在蛋白质条件下的零样本配体比较任务中，VLSR以0.691的准确率超过所有基线（最高0.628）。

**⚠️ 局限性**

局限性：
- 训练期间依赖RDKit生成的矩形区域注释，未完全摆脱化学监督，注释空间可能偏向于局部功能团；
- 仅基于二维图像，无法捕捉构象能量或蛋白–配体接触等三维物理效应；
- 区域注释以盒子形式给出，可能无法覆盖电子分布广泛或交叉影响的化学特征；
- 对于需要分散电子上下文的属性，现有区域划分可能不足，需更大或关系化的区域定义。

---

## 430. Can Formal Specifications Be Synthesized from Tests Alone?

**arXiv ID:** 2608.13240 | [PDF](https://arxiv.org/pdf/2608.13240v1)

**作者:** Tianhai Liu `[一作]` (Karlsruhe Institute of Technology), Bernhard Beckert `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何仅利用测试代码和执行观察，结合LLM生成JML规范并通过BMC验证的框架。

**💡 创新点**

将LLM用于黑盒测试驱动的规范推理，避免暴露实现源码，并结合观察覆盖与一致性检查的双重验证。

**🔧 技术方法**

使用大语言模型（ChatGPT/GPT‑5.4）生成规范，结构化执行观察，JML语言，边界模型检查（JJBMC）进行验证与反向修正。

**📊 数据集**

使用SpecGenBench 22个API级别任务作为评估基准。

**📈 对比分析**

通过在JJBMC下对生成规范进行一致性与覆盖检查，结果在22项任务中成功验证11项（7项一次通过，4项需迭代），展示了可行性但受限于工具片段支持。

**⚠️ 局限性**

受限于JJBMC对JML片段的支持（无法验证ghost、model字段等表达式），BMC界限导致缺乏有用反例，以及缺少公开或工业案例验证。

---

## 431. When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1

**arXiv ID:** 2608.13237 | [PDF](https://arxiv.org/pdf/2608.13237v1)

**作者:** Weimeng Luo `[一作]` `[通讯]` (Unaffiliated), Weimeng Luo (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的 Search‑R1 检索增强生成流水线中引入 S2G‑style 结构化判断器，并使用 Qwen3.5‑2B 训练该判断器，形成一种基于状态的停止策略；在 HotpotQA 上验证该策略可在保持答案精度可接受范围内显著减少检索调用；

**💡 创新点**

将结构化充分性与缺失信息（gap）判断引入检索-增强生成的停止决策；提供基于轨迹的评估框架，区分候选可达性、状态排名与实际停用行为；并采用预先冻结策略与阈值的确认性非劣势检验，避免过度优化导致安全风险。

**🔧 技术方法**

S2G‑style 结构化判断器、Qwen3.5‑2B 作为判别器、冻结的 Search‑R1（7B 推理器、E5 top‑3 检索、Wiki‑2018 语料、四步检索预算）、阈值化停止策略、训练时使用 3,009 个状态、评估时使用配对 Bootstrap 置信区间、计算 STOP 精度/召回/AP、EM 与平均检索次数。

**📊 数据集**

HotpotQA distractor 开发集（1,000 题），其中 700 题用于训练、100 题验证、200 题保留；NQ/HotpotQA 评审集用于检查基础 EM；训练数据来自 HotpotQA 训练集。

**📈 对比分析**

与原始 Search‑R1 做直接对比，评估官方 EM 与平均检索调用；确认性测试集（索引 200‑999）显示 EM 下降 0.00625（95% CI [-0.0125,0]，低于预设 -0.02 非劣势阈值）且检索调用下降 0.09625（95% CI [-0.12,-0.07375]，约 3.7%），即在保持精度可接受范围内实现检索量显著减少；探索性审核显示 27.3% 的检索可提前停止。

**⚠️ 局限性**

不具备安全停用保证；安全停用比例仅 60%（39% 处于不安全），未提供正式风险控制；判断器推理成本未计入整体效率；检索减少并不等同于总成本下降；校准性差，阈值在不同数据集上表现不稳定；实验仅在 HotpotQA 上验证，缺乏更广泛的泛化性验证。

---

## 432. Knowledge-guided Pattern Discovery via Coupled Tensor Factorizations

**arXiv ID:** 2608.13234 | [PDF](https://arxiv.org/pdf/2608.13234v1)

**作者:** Gaute Johannessen `[一作]` (University of Oslo), Evrim Acar `[通讯]` (Simula Metropolitan Center for Digital Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种知识引导的耦合张量分解方法，联合分析真实代谢组数据与计算模型生成的模拟数据，以改进模式发现。

**💡 创新点**

创新点在于首次将线性耦合的CP与PARAFAC2张量分解应用于多视角数据，利用计算模型提供的先验信息同时捕获个体差异。

**🔧 技术方法**

技术实现包括线性耦合的CP/PARAFAC2张量分解、L‑BFGS与AO‑ADMM优化、非负约束及T0校正等。

**📊 数据集**

使用的数据集为COPSAC2000男性受试者的真实代谢组数据（161种代谢物×8时点×133人）和全身代谢模型生成的虚拟受试者数据（6种代谢物×8时点×50人）。

**📈 对比分析**

通过将各模型的受试者因子与身体成分和胰岛素抵抗指标的相关性进行比较，耦合模型在相关性上显著优于单独使用CP或PARAFAC2的模型，成功提取更干净的胰岛素/葡萄糖模式。

**⚠️ 局限性**

局限性包括仅考虑线性耦合，未探索非线性或共享/非共享因子；模拟与真实数据的差异可能导致误导；实验仅在代谢组数据上验证，需在神经科学等其他领域进一步验证。

---

## 433. CoverPrune: Coverage-Driven Token Pruning for 3D VLMs via Optimal Transport

**arXiv ID:** 2608.13226 | [PDF](https://arxiv.org/pdf/2608.13226v1)

**作者:** Peng Ling `[一作]` (Tsinghua University), Wenming Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关、覆盖驱动的视觉令牌剪枝方法CoverPrune，专门针对3D视觉‑语言模型（3D VLM）在推理时的令牌爆炸问题；

**💡 创新点**

核心创新在于把剪枝问题重新表述为最优传输（OT）覆盖优化，并设计多维 Feature‑Spatial‑Temporal（FST）成本函数、信息化目标容量以及空间引导的贪婪选择（SGS）和轻量化的CoverPrune‑Lite版本；

**🔧 技术方法**

利用OT理论、稀疏贪婪优化、空间局部匹配、特征-空间-时间融合成本等技术实现训练‑free 的覆盖驱动剪枝；

**📊 数据集**

在ScanQA、SQA3D、Scan2Cap以及更具挑战性的VSI‑Bench（包含多种空间推理子任务）等3D视觉‑语言基准上进行评估；

**📈 对比分析**

与VisionZip、FastVID、DTC、EgoPrune等主流剪枝方法对比，CoverPrune在相同令牌保留比例下在大多数任务中保持90%以上的性能，并在10%或5%令牌保留时实现更稳健的性能提升；

**⚠️ 局限性**

局限性包括完整CoverPrune在极端压缩下仍有一定计算开销，Lite版对细粒度关系的覆盖略弱；此外尚未充分验证对更广泛通用VLM的迁移效果。

---

## 434. Co-leading Teams Drive Scientific Novelty in Large-scale Research Infrastructures

**arXiv ID:** 2608.13195 | [PDF](https://arxiv.org/pdf/2608.13195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 435. FAM-DQ: A Dual-Quadrotor-Based Fully Actuated Aerial Manipulator for High-Torque Interaction

**arXiv ID:** 2608.13220 | [PDF](https://arxiv.org/pdf/2608.13220v1)

**作者:** Xuwei Yang `[一作]`, Ziqian Guo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种双四旋翼完全驱动的空中操纵平台 FAM-DQ，利用中心框架作为杠杆，提升与末端执行器的扭矩输出，并实现 6-DoF 的分离姿态与位置控制。

**💡 创新点**

创新点在于：1）双四旋翼通过被动关节与中心框架耦合，形成物理杠杆，显著放大扭矩；2）采用齿轮驱动舵机实现末端执行器的自由指向；3）设计轻量化的 6-DoF 控制器和控制分配，实现姿态与位置完全解耦。

**🔧 技术方法**

采用了机械结构设计、动力学建模、全局与局部分层控制架构、控制分配算法、几何姿态控制器以及四旋翼的电机速度分配等技术。

**📊 数据集**

在实验室环境下使用光学运动捕捉系统进行数据采集，完成圆形轨迹跟踪、悬停姿态跟踪、静态扭矩输出测试和螺丝驱动实验等验证实验，未使用公开数据集。

**📈 对比分析**

实验对比指标包括：位置误差≤±0.05 m、姿态误差≤±1.5°、最大扭矩1.019 N·m（扭矩/质量比2.28 N·m/kg），在圆形轨迹和姿态跟踪实验中均保持稳定且误差低，螺丝驱动实验成功完成紧固任务，证明平台具备高扭矩与高精度双重性能。

**⚠️ 局限性**

局限性包括：1）机械结构仍有一定复杂度，额外的被动关节和齿轮驱动增加重量与维护成本；2）扭矩输出受限于双四旋翼的推力上限，较大负载下可能出现饱和；3）实验仅在静态室内环境验证，缺乏动态环境下的鲁棒性与能耗优化研究。

---

## 436. NARU: A Benchmark for NARrative Evolution and Cultural Nuance Understanding in Japanese Extreme Long Video

**arXiv ID:** 2608.13210 | [PDF](https://arxiv.org/pdf/2608.13210v1)

**作者:** Yuheng Huang `[一作]` (University of Tokyo), Lei Ma `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了NARU基准，用于评估多模态大型语言模型在日本长篇视频中的叙事和文化推理能力。

**💡 创新点**

创新点在于提出层次化记忆式标注到QA合成管道、迭代式文本捷径消除以及两阶段本地日语专家验证，首次在超长视频上同时考察叙事连贯与隐式文化理解。

**🔧 技术方法**

技术采用Gemini、Qwen等多模态LLM进行分块处理、语义分段、任务导向注释，并通过Solver‑Critic循环修正生成的多选题；评估时使用多选准确率和开放式事实召回等指标。

**📊 数据集**

数据集由155段日本长篇视频（共146.8小时）构成，生成1,481道多选问答，涵盖四种叙事维度与五种文化维度，全部经68名日语母语者验证。

**📈 对比分析**

实验将Gemini等商用模型与开源模型对比，商用模型最高达76.2%准确率，开源模型最低约30%；显示商用模型在叙事追踪与文化推理上明显优于开源模型，并能从更多帧中获得更好表现。

**⚠️ 局限性**

局限在于仍依赖大模型的先验知识与视频预训练，开源模型在高层主题推理与隐式文化理解方面表现不佳，且在开放式评测中叙事顺序任务的难度显著上升。

---

## 437. Beyond Simulated Benchmarks: Evaluating Motion Representations for Fall Detection Under Real-World Data Scarcity

**arXiv ID:** 2608.13197 | [PDF](https://arxiv.org/pdf/2608.13197v1)

**作者:** Timilehin B. Aderinola `[一作]` (University College Dublin), Georgiana Ifrim `[通讯]` (University College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统评估了多种运动表示在真实世界跌倒检测中的鲁棒性，尤其关注数据稀缺与跨域迁移问题

**💡 创新点**

创新点在于首次量化模拟与真实跌倒数据间的性能差距，并提出物理量根植的符号表示FallLM，能在极少样本下保持较高召回且迁移性能最佳

**🔧 技术方法**

采用了间隔、核、符号（WEASEL、MrSQM、FallLM）以及基于Transformer的基础模型（Mantis），并对比MiniRocket与QUANT等代表方法

**📊 数据集**

使用FARSEEING（208条真实跌倒）和FallAllD（465条模拟跌倒）两大公开数据集进行实验

**📈 对比分析**

在留一主体交叉验证、数据稀缺实验和零改进跨域测试中，Mantis与QUANT在模拟数据上表现最优，但在真实数据和迁移任务中表现大幅退化；FallLM虽然在域内精度略低，但在真实数据中的F1最高（0.67），并且在仅用两例跌倒训练时仍能获得约0.35的F1

**⚠️ 局限性**

局限包括仅使用单一真实数据集、未对所有代表性模型进行同一分类器下的纯比较、未进行域适应实验以及FallLM在低冲击跌倒上的召回不足

---

## 438. Fidelity-Constrained Anchoring for Black-Box Denoisers

**arXiv ID:** 2608.13194 | [PDF](https://arxiv.org/pdf/2608.13194v1)

**作者:** Masaki Satoh `[一作]` `[通讯]` (Morpho, Inc.), Masaki Satoh (Morpho, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于保真度约束的锚定框架，通过线性混合原图与黑盒去噪器输出，并在每个局部子图上最大化混合系数，满足预设的PSNR或SSIM阈值，从而控制输出与输入的相似度。

**💡 创新点**

创新点在于：无需重新训练或改造黑盒去噪器；对PSNR给出闭式解，对SSIM提供可迭代求根方法；实现了在不同噪声水平下更稳健的SSIM锚定，且仅需少量计算。

**🔧 技术方法**

使用局部均方误差、SSIM及其逆值，构造二次方程或使用切线/伪位置法求根；采用7×7局部窗口；对比使用OpenCV双边滤波、Real‑ESRGAN和非局部均值去噪器等黑盒模型。

**📊 数据集**

在DIV2K验证集上实验，向图像添加标准差为25、50、75的高斯噪声作为输入。

**📈 对比分析**

对比方法包括未锚定的Real‑ESRGAN、双边滤波；评估指标为全局PSNR、SSIM以及残差噪声峰度。实验显示，SSIM锚定在所有噪声水平下均保持高PSNR且峰度低，优于PSNR锚定和双边滤波；PSNR锚定在低噪声时表现更好，但对噪声敏感。

**⚠️ 局限性**

由于采用线性混合，若去噪结果与原图差异过大或噪声极强，性能会下降；局部常数混合假设在极端情况下可能失效，导致保真度约束不完全满足。

---

## 439. SketchSense: Learning to Interpret Imperfect Sketch Guidance for Image Inpainting

**arXiv ID:** 2608.13186 | [PDF](https://arxiv.org/pdf/2608.13186v1)

**作者:** Zian Yang `[一作]` `[通讯]` (Fudan University), Zian Yang (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SketchSense，一种同时恢复 RGB 内容和解释不完美草图的同步双流图像修复框架；

**💡 创新点**

创新点包括：Bidirectional Attention Fusion 在 RGB 与结构分支间双向信息交换；短语级交叉分支语义一致性损失；自适应 Sketch‑Aware Spatial Regulation 对局部草图可靠性与残差注入进行动态调节；可选 Signed Prior 用于表达保留或校正意图，并可视化 refined structure；

**🔧 技术方法**

技术实现基于 FLUX Diffusion Transformer，双流 Transformer 结合低秩 LoRA 交叉注意，JS 距离语义一致性损失，可学习的可靠性与残差门控，Signed Prior 的 embedding、key bias 与 value LoRA；

**📊 数据集**

训练与评估数据集包括 ArtBench、COCO 以及另一个未命名的图像集合，使用配对边缘注解或 MuGE 自动生成边缘，并模拟真实用户草图；

**📈 对比分析**

与 PDDP、MaGIC、SketchRefiner、OminiControl 等现有方法在 Easy/Medium/Hard 子集上做 PSNR/SSIM/LPIPS 对比，SketchSense 在所有子集均获得最高 PSNR（≈19.4）/最高 SSIM（≈0.858）/最低 LPIPS（≈0.072），并在结构复杂的修复任务中展现更高的结构一致性；

**⚠️ 局限性**

局限性包括：仍以 512×512 的 FLUX 1 Fill 为基础，计算开销较大；未实现更高分辨率或更高效的双流设计；Signed Prior 需手工指定，交互使用不够简洁；在极端稀疏或错误严重的草图下仍可能出现误修复。

---

## 440. Jointly Predicting Courses and Grades Using a Transformer-Based Model

**arXiv ID:** 2608.13409 | [PDF](https://arxiv.org/pdf/2608.13409v1)

**作者:** Paul Savala `[一作]` `[通讯]` (St. Edward's University), Paul Savala (St. Edward's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种Transformer框架TRACE，联合预测学生下学期将选修的课程集合及对应成绩。

**💡 创新点**

创新点在于：① 将同一学期课程并行编码为位置不变序列；② 采用集合式KL损失结合MSE的多任务目标，避免课程顺序误差；③ 通过课程预测辅助学习更有意义的课程向量，显著提升成绩预测精度。

**🔧 技术方法**

技术包括Transformer编码器-解码器、位置编码、课程/专业/成绩嵌入、定制的KL+MSE损失、one‑step‑ahead训练与掩码策略。

**📊 数据集**

使用St. Edward’s University 10 年（2014‑2023）历史数据，5326 名学生、360 门课程、48 专业，18 个学期的注册与成绩记录。

**📈 对比分析**

与 LSTM、XGBoost、GNN 等基线模型比较，TRACE 的 MAE 下降约 46%（0.1339→0.0392），MSE 降低 3.5 倍；相较于 LSTM 下降 30% MSE、15% MAE；优于 GNN 15‑20% 低误差。

**⚠️ 局限性**

局限性：单一机构数据，缺少非学术特征（社会经济、学习行为等）导致可能存在偏见；对新生或新课程的冷启动问题；模型泛化能力需跨机构验证。

---

## 441. Solving Square-Submatrix Equation Systems

**arXiv ID:** 2608.13408 | [PDF](https://arxiv.org/pdf/2608.13408v1)

**作者:** Lorenzo Carfagna `[一作]` (University of Pisa), Giovanni Manzini `[通讯]` (University of Pisa)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了二维的方形子矩阵方程组（SSES）模型，并给出了一个线性时间（O(|E|+mn)）求解算法，证明任何有效的二维宏方案都可在O(b+mn)时间内解压。

**💡 创新点**

创新点在于将一维子字符串方程组（SES）的递归求解技术推广到二维，设计了带类型的方程图、k‑special/k‑short 定义以及 Split/Reduce 递归算法，利用最大生成森林消除冗余，实现了从 O(|E|mn) 到 O(|E|+mn) 的时间优化。

**🔧 技术方法**

主要技术包括：
- 方程类型化（存储对应角点信息）；
- 位置图和方程图的构造；
- k‑special 与 k‑short 的层次划分；
- 递归 Shorten 程序（SpecialSplit、Reduce、SimpleSplit 等）；
- 最大生成森林（MSF）用于消除冗余边；
- 连接分量判定求解方案。

**📊 数据集**

本工作为理论性研究，无使用具体数据集，主要以数学证明和算法分析为主。

**📈 对比分析**

通过理论证明展示了所给算法的最优性；相较于传统的宏方案解压算法（最坏 O(bmn)），该方法在时间复杂度上取得显著提升，达到线性级别 O(b+mn)。

**⚠️ 局限性**

局限性：
- 仅适用于方形子矩阵方程，不能直接处理非方形或非正方形子块；
- 对整数坐标的特殊化处理（k‑special）依赖十六进制基底，可能在某些实现中需要额外转换；
- 论文中未给出实验评估，缺乏对实际压缩比和常数因素的量化分析。

---

## 442. Credible, Not Always Correct: How Reddit Users Verify AI-Generated Legal Advice

**arXiv ID:** 2608.13369 | [PDF](https://arxiv.org/pdf/2608.13369v1)

**作者:** Rebecca Owens `[一作]` (Durham University), Tuğrulcan Elmas `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Reddit用户使用LLM生成法律建议并据此行动的验证与可信度获取过程进行定量与定性分析

**💡 创新点**

首次系统阐明“分布式律师”(distributed counsel)模式，并揭示大多数用户在未进行任何验证的情况下依赖LLM建议，其可信度源自文本的律师式形式而非准确性

**🔧 技术方法**

使用文本挖掘、人工多标签编码、LLM辅助立场识别（Gemini 2.5 Flash）以及交叉验证的方式完成数据处理与分析

**📊 数据集**

从六个法律与AI相关的Reddit子版块抽取的153条用户原始叙事和5,341条社区评论，构成完整的讨论语料库

**📈 对比分析**

通过编码统计与手工标注对验证行为、风险类型、社区立场进行对比，结果显示独立验证率仅约17%，社区支持与怀疑比例随时间变化（2023年支持/怀疑=0.77:1→2025–26年=1.83:1），提供对验证与讨论模式的实证洞察

**⚠️ 局限性**

受限于自述数据的主观性、Reddit样本的公开性与平台偏倚、无法确认实际法律结果、以及LLM分类可能的偏差和误判

---

## 443. Sign Language Video Synthesis via Loss-Guided Multi-Expert GANs

**arXiv ID:** 2608.13368 | [PDF](https://arxiv.org/pdf/2608.13368v1)

**作者:** Dingzhan Nong `[一作]` (Glassbox AI), Tim Lo `[通讯]` (Glassbox AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种用于手语视频合成的多专家GAN框架，利用三个专门的判别器（全局、手部、面部）引导对应的生成器分支，进一步通过双路径卷积‑Transformer结构和自适应特征融合实现区域特化。

**💡 创新点**

创新点包括：1) 联合损失（United Loss）在多判别器间引入软共识，抑制早期训练不稳定；2) 双路径卷积‑Transformer+AdaptiveFeatureFusion实现局部细节与全局稳定的动态平衡；3) 三模式交替训练（判别器、整体生成、分支专注）与多尺度局部‑全局融合注意力；4) 在消费级GPU上实现可部署的高质量生成。

**🔧 技术方法**

采用的技术包括：多专家U‑Net生成器、三重判别器（使用Haar小波预处理与PatchGAN）、AdaIN风格-骨架融合、ConvTransformer与Swin Transformer双流、AdaptiveFeatureFusion、Local‑Global Merged Attention、United Loss共识机制、交替三模式训练、AdaBelief优化器。

**📊 数据集**

使用了自制的156 GB手语视频数据集，包含手语词汇视频，配有骨架图、风格图和真实帧，测试集剔除了“易样本”以聚焦于复杂手势与面部表情。

**📈 对比分析**

在过滤后的难度样本上，三种模型规模分别在0.2B、0.66B和1.3B参数时，PSNR分别达到29.8、30.4（待更新）和30.7，推理显存占用为1.5 GB、约5 GB和8 GB，显示可在单张消费级RTX 4090上部署，性能优于传统单判别器GAN。

**⚠️ 局限性**

局限包括：训练周期长（2–3个月/单GPU）导致缺乏完整消融研究；United Loss的真实贡献尚未通过对照实验验证；模型在多分支激活时计算与显存线性增长，难以扩展到更多专家；评估仅基于PSNR，缺乏感知质量、FID、LPIPS和人类评测等指标。

---

## 444. Rules or Character? Scaling Laws for AI Safety Design

**arXiv ID:** 2608.13345 | [PDF](https://arxiv.org/pdf/2608.13345v1)

**作者:** Satoshi Takahashi `[一作]` (RIKEN Center for Advanced Intelligence Project), Ryuji Hamamoto `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出了一个比较静态模型，用于分析AI安全体系中角色塑造与规则执行的最优资源分配，并研究该分配随部署规模的变化。

**💡 创新点**

创新点在于将安全设计参数化为α，并引入字符脆弱性、过滤器退化、共模失效等因素，闭式推导期望伤害并通过CVaR分析尾部风险，发现字符脆弱性是决定最优α的主导因素。

**🔧 技术方法**

采用解析闭式公式计算期望损害，使用计数级别的Monte Carlo模拟CVaR，结合Pareto乘法损害模型与高斯行为分布的理论推导。

**📊 数据集**

未使用真实数据集，而是通过三种情景（乐观、中等、悲观）设定的参数范围进行仿真分析。

**📈 对比分析**

通过比较不同α和规模T下的期望伤害与CVaR来评估最优α的变化；结果显示在乐观情景下α*基本不随规模变化，在悲观情景下随规模显著上升，并且期望损害与CVaR最优几乎一致。

**⚠️ 局限性**

局限性包括模型仅考虑一维高斯行为、静态不考虑攻击适应与动态更新、字符脆弱性与规模无关、尾部模型与α独立，导致对真实多属性、多态行为和动态适配的适用性有限。

---

## 445. Where You Measure Decides What You Measure: Position Selection in Ablation-Based SAE Evaluation

**arXiv ID:** 2608.13337 | [PDF](https://arxiv.org/pdf/2608.13337v1)

**作者:** Valentin Noël `[一作]` `[通讯]` (Devoteam), Valentin Noël (Devoteam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究稀疏自编码器（SAE）在解释语言模型时，测量因果效应时使用“最激活标记”这一测量位置的约定会导致不同词典产生巨大的差异，并对比两种测量方式（按词典自选位置与统一位置），展示其对方差贡献的影响；通过在同一初始化下训练六个SAE，分离出约定与词典本身的影响，提出了在报告因果数值时应遵循的规范。

**💡 创新点**

创新点在于：1) 揭示测量位置约定是导致字典间因果差异的主因；2) 通过控制测量位置几乎消除 latent×arm 方差；3) 提出包含位置、单位扰动归一化、特殊标记剔除及训练方案范围等四项必报规范；4) 证明该修正仅需一行评估代码即可实施。

**🔧 技术方法**

技术方法包括：稀疏自编码器（TopK 目标）、零-消融因果读数、Generalizability Theory 方差分解、Hodges–Lehmann 中位数比率估计、Bootstrap 区间、以及多种评估约定（单一位置、统一位置）。

**📊 数据集**

数据集与模型：基于 Gemma‑2‑2B 与 Gemma‑3‑1B 两款 Gemma 语言模型；训练使用 WikiText‑103（约12M标记）；评估使用 WikiText‑2 的 96、384 与 1536 条序列（约1.8M标记）。词典来源为 Google 公开的 Gemma Scope SAEs。

**📈 对比分析**

比较方法：对六个同一初始化、不同训练选项的 SAEs 进行 per‑arm（各自选最激活位置）和 shared（统一位置）两种测量；使用方差分解与 Eρ² 评估位置对 latent×arm 方差的贡献。结果显示，按词典自选位置时 latent×arm 方差约占 20‑40%（Gemma‑2‑2B）和 30‑50%（Gemma‑3‑1B），而统一位置后降至 <1%，Eρ² 明显提升；增大评估语料量并未减小此差异。

**⚠️ 局限性**

局限性：仅在 Gemma 系列模型验证；需共享初始化以分离约定，可能无法迁移到其他词典或架构；仅考虑零‑消融而非互换/RAVEL 等因果度量；评估语料仍有限，且受过滤后子样本的影响；未探讨不同词典对位置差异的根本机制。

---

## 446. LLM-Guided Graph Generation for Structure-Based Local Improvement Methods

**arXiv ID:** 2608.13333 | [PDF](https://arxiv.org/pdf/2608.13333v1)

**作者:** Hai Xia `[一作]` (TU Wien), Stefan Szeider `[通讯]` (TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过LLM自动生成图生成器，将MiniZinc模型实例映射为统一加权图，并在此基础上实现SLIM框架的结构感知变量选择以及跨问题的算法配置选择；

**💡 创新点**

1) LLM驱动的无人工特定领域干预的图生成；2) 统一的图表示实现问题无关的SLIM；3) 基于图特征的跨问题算法配置选择；

**🔧 技术方法**

大语言模型（Claude Opus 4.5）生成Python图生成器；图构建与提取；结构化局部改进（SLIM）框架；多配置（30种）SLIM与随机森林回归等机器学习算法用于配置选择；

**📊 数据集**

MiniZinc竞赛2008‑2025年共20类问题的实例数据集；

**📈 对比分析**

与一次性Gurobi 60 min基线比较，SLIM最佳单配置平均赢率约19.3%，算法选择方案平均赢率约44.0%，显著优于单配置且接近虚拟最优；通过win/tie/loss统计评估性能；

**⚠️ 局限性**

对单个问题的表现不一定超过一次性Gurobi；LLM生成的图可能不完全反映真实约束语义；仅在MiniZinc竞赛基准上验证，泛化性待进一步验证。

---

## 447. Training AI Scientists to Replicate Research

**arXiv ID:** 2608.13331 | [PDF](https://arxiv.org/pdf/2608.13331v1)

**作者:** Damon Falck `[一作]` (Inherent), Edward Hughes `[通讯]` (Inherent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究训练了一个名为Faraday的AI Scientist代理，能够在有限时间和GPU资源内复制论文中的图表并进行实验验证。

**💡 创新点**

核心创新在于将可解释的rubric‑based judge与GRPO强化学习相结合，利用较小的outer agent（27B参数）指导更大编码代理（Codex GPT‑5.5）完成开放式复制任务，显著提升了对论文复现的科学严谨性和创新性。

**🔧 技术方法**

技术方法包括：大语言模型Qwen3.6‑27B作为outer agent；Codex GPT‑5.5作为编码工具；GRPO改进的奖励机制；rubric judge生成与评估；LoRA微调、128K-token上下文、token‑级credit分配。

**📊 数据集**

使用了自建的Replica任务空间，共310个图表复制任务，来源于100篇1990‑2026年间的机器学习与AI‑for‑science论文。

**📈 对比分析**

与Claude Opus 4.8和GPT‑5.5进行对比，Faraday在训练集任务中取得73%任务成功率、测试集任务中60%成功率，平均在rubric评分上提升约6–8%。

**⚠️ 局限性**

局限性包括：对大规模资源或长时间实验的泛化尚未充分验证；对想象式创新任务的评判仍不完善；模型仍可能出现奖励误差或与人类评判不完全一致。

---

## 448. A Probe Direction Is a Property of Its Prompt

**arXiv ID:** 2608.13329 | [PDF](https://arxiv.org/pdf/2608.13329v1)

**作者:** Valentin Noël `[一作]` `[通讯]` (Devoteam), Valentin Noël (Devoteam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了用对比激活差异检测模型是否“意识到正在被测试”的方法，发现该方法对提示（prompt）选择高度敏感，提示决定了评分结果的大小甚至符号，模型本身的影响很小；

**💡 创新点**

揭示了评价指标的主要变异来源是实验设计中的提示选择，而非模型本身；提出了需要多提示、多条件才能可靠比较模型的结论；

**🔧 技术方法**

使用了一般化理论（Generalizability Theory）分解方差，控制实验设计的不同因子，计算了对比激活差异产生的方向向量及AUROC统计；

**📊 数据集**

使用公开的评测集合（OpenAI Refusal Split、Harmful/Harmless split）以及十个不同规模的指令调优模型；

**📈 对比分析**

比较方法是通过多因子交叉实验（6×6 提示组合）得到方向向量，对每层最大AUROC进行统计，发现不同提示导致相同模型的评分差异可达正负符号；模型间的差异在多提示平均后仍不足以可靠区分；

**⚠️ 局限性**

局限性包括：只考察单一模型族的交叉设计，未覆盖所有可能提示；方法对提示渲染细节（换行、空白处理）敏感；无法区分模型真正的“评估意识”与提示引起的表面效应；

---

## 449. A minimum witness for the 3/2 configuration-linear-program gap in two-weight graph balancing, unique at its size

**arXiv ID:** 2608.13318 | [PDF](https://arxiv.org/pdf/2608.13318v1)

**作者:** Adam Y. Shavit `[一作]` `[通讯]` (City University of New York), Adam Y. Shavit (City University of New York)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在两重量图平衡（每个作业仅可分配给两台机器且尺寸为1或2）中配置线性规划的整型缺口，并给出了最小、唯一的六作业示例证明3/2缺口，同时对七至九作业的所有示例进行分类，指出至少需要四台机器。

**💡 创新点**

创新点在于：①提供了已知最小尺寸的完整证据（六作业），并证明不存在更小示例；②证明六作业示例在机器数上也是最小且唯一；③对七至九作业的所有示例进行完整枚举与分类；④使用双重验证（浮点与精确有理数）保证结果完全可信。

**🔧 技术方法**

采用的技术包括：完整枚举与同构归约（Burnside定理验证），配置线性规划的可行性与最小化求解，精确有理数单纯形法做二次验证，子集和与欧拉定理等组合论工具来证明可行性与不可能性。

**📊 数据集**

数据集为手工构造的所有满足条件的作业多重集，按机器数与作业数划分，覆盖至九作业；所有实例及其验证结果已发布在OSF平台，包含脚本、结果文件与日志。

**📈 对比分析**

与已有结果对比时，本文在三倍化逼近阈值、整型缺口与机器数方面与文献中的理论上限（3/2）完全一致；在计算复杂度上，精确验证消耗数小时，然而相比单次浮点验证显著提升了可靠性。

**⚠️ 局限性**

局限性包括：只考虑尺寸为{1,2}且每作业至多两台机器的子类；扩展到更大尺寸或多于两台机器时缺口与最小示例可能不同；枚举规模随作业数和机器数急剧增长，限制了可行搜索的上界。

---

## 450. On the Structure of $(\min,+)$ Convolution

**arXiv ID:** 2608.13310 | [PDF](https://arxiv.org/pdf/2608.13310v1)

**作者:** Huanyi Zhou `[一作]` `[通讯]`, Huanyi Zhou

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过研究 tropical algebra 的结构特性（如 tropical decomposition width、convex gap）提出了针对多序列（min,+）卷积的新算法，能够在随机化与分块、取模等技术辅助下显著降低计算复杂度。

**💡 创新点**

创新点在于将 tropical decomposition width 与 convex gap 两种结构度量统一起来，利用加法组合论与数论方法（取模、最小公倍数估计）把多序列卷积拆解成若干子问题；此外引入随机打乱 + 隔离技巧在维持高概率正确性的同时实现局部调整，从而得到更快的单点卷积算法。

**🔧 技术方法**

主要技术包括：tropical algebra 与 (min,+) 卷积的表达式推导、凸包与弱凸支撑序列求解、完全偶数区间（totally monotone）矩阵行最小值的 SMAWK 算法、随机打乱、隔离引理、Hoeffding 无替换采样不等式、动态规划剪枝、数论估计（lcm 估计）。

**📊 数据集**

该研究为纯理论算法，未使用具体实验数据集，算法性能以渐进复杂度表达。

**📈 对比分析**

与传统的 O(n^2) 朴素卷积或分治方法相比，本文在 k≈n 情形下实现了 O(kn^2 √(min(k,n)) log^1.5(kn)) 的随机化复杂度；在单点查询下更进一步降低到 O((kn+n^3√(min(k,n))) log(kn))；在 n≈k 时达到 O(k n^3 log k) 的确定式复杂度。

**⚠️ 局限性**

局限性：仍需假设 (min,+) 卷积难度假设；算法在极大 k（k≫n）时仍需先求弱凸支撑，计算量不可避免；对所有输入条目需预设均匀随机权值，增加实现复杂度；在小 n 或特殊结构下可能未达最佳效率。

---

## 451. Refusing Intent, Not Form: Wrapper-Based Intent-Group Supervision for LLM Safety

**arXiv ID:** 2608.13304 | [PDF](https://arxiv.org/pdf/2608.13304v1)

**作者:** Ping Wu `[一作]` (Chinese Academy of Sciences), Yi Zeng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构造同一意图下不同包装形式的匹配组，自动化提升对有害请求的拒绝并降低对无害请求的误拒。

**💡 创新点**

创新在于无需外部教师或手工意图标签，使用 Wrapper‑Based Intent‑Form Augmentation (WIFA) 生成意图组数据，并提出两种训练路径：安全提升两阶段 WIFA‑Boost 与 Anchored Group‑Consistent Refusal Training (A‑GCRT) 控制误拒。

**🔧 技术方法**

采用数据层增强、两阶段 LoRA 微调、以及基于组内一致性与方向锚定的正则化损失来训练模型。

**📊 数据集**

使用 Qwen2.5‑7B 与 Llama‑3.1‑8B 预训练模型，结合 250 条有害种子（AdvBench / HH‑Inst）与 7 组包装形式，构建 5750 条训练样本；评测基准包括 HarmBench、SORRY‑Bench、OR‑Bench、XSTest、StrongREJECT、MMLU、GSM8K。

**📈 对比分析**

与复制的提示式与训练式防御方法相比，WIFA‑Boost 在 Qwen 上达到最高的有害拒绝率 (SB≈63.7)，A‑GCRT‑M5 在保持较高拒绝的同时将 OR‑Bench 误拒率从 25.7% 降至 17.4%，并在 Llama 上同样提升安全性。

**⚠️ 局限性**

局限包括仅在 7–8B 指令模型与固定包装族上验证，可能在更大模型、多语言、工具使用或自适应攻击下表现不同；A‑GCRT 的决策位置得分仅为训练时正则化，未构成推理时分类器；部分指标仍出现能力损失。

---

## 452. Does Fixing Break Security? An Empirical Study of Security Degradation in Iterative LLM-Driven Infrastructure-as-Code Repair

**arXiv ID:** 2608.13404 | [PDF](https://arxiv.org/pdf/2608.13404v1)

**作者:** Benjamin Agyekum `[一作]` (Colorado State University), Fabio Santos `[通讯]` (Colorado State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对使用大型语言模型的 IaC 反馈循环中迭代修复过程是否会引入安全回归进行大规模实证研究，量化其频率、根因及对代码变更和检查波动的影响。

**💡 创新点**

首次在 IaC 领域提出标准/严格两种回归检测模式以区分多资源误报，系统化评估回归根因、代码冲突和检查波动，并给出迭代停止点（第3次）与安全性‑正确性权衡的实证依据。

**🔧 技术方法**

采用 Terraform 迭代生成、Checkov 静态分析、RAG 检索、差分根因分类、统计检验（χ²、Mann‑Whitney 等）等技术。

**📊 数据集**

基于 IaC‑Eval 基准（458 个 AWS Terraform 场景）、15 种配置、30 个 CIS 检查，构建了 5,968 条时间线和 4,440 条迭代转换。

**📈 对比分析**

通过比较标准与严格检测模式、不同模型（Gemini、Mistral）、提示策略、温度和是否使用 RAG，利用回归率、代码行数变更、检查波动等指标评估性能；严格模式回归率约 3.3%，标准模式约 13.8%，并发现最佳迭代次数为第 3 次。

**⚠️ 局限性**

仅在 Terraform/Checkov/AWS 环境下评估，未对根因分类的准确性进行人工验证，单次实验未评估重复性，且未覆盖部署时安全验证，模型与提示策略的偏差可能影响结果。

---

## 453. CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation

**arXiv ID:** 2608.13387 | [PDF](https://arxiv.org/pdf/2608.13387v1)

**作者:** Enhan Li `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于匹配的反事实和改写的硬性token选择方法CROP，用于对现有的On‑Policy Distillation（OPD）进行任务相关性加权监督。

**💡 创新点**

创新点在于：① 将任务相关性量化为“对比敏感度”——对同一roll‑out在条件改变与语义保持改写下的token分布差异；② 通过在线生成并验证改写–反事实三元组实现对比特定；③ 在OPD中直接使用硬性token掩码，而不改变teacher目标或更新机制。

**🔧 技术方法**

技术手段包括：离线生成原始–改写–反事实triplet、对fixed roll‑out进行三次重评分、使用Top‑K Jensen–Shannon Divergence 计算敏感度、将差值做为选择分数、批量级预算下的硬性token掩码、以及在采样token OPD损失中使用该掩码。

**📊 数据集**

使用的主要数据集为：从17,398条DAPO‑Math‑17K数学提示中筛选的16,594条triplet；评测覆盖AIME24、AIME25、MATH‑500、GPQA‑Diamond、HumanEval与IFEval六个基准；教师模型为Qwen3‑8B（GRPO）或Qwen3‑4B，学生模型为Qwen3‑1.7B或Qwen3‑4B。

**📈 对比分析**

与Pure OPD、Entropy、TIP、TA‑OPD、CREDIT等现有选择器在相同的10% token预算下比较。CROP平均提升约1.9–2.9分，在两种教师–学生组合中均优于所有非CROP方法；CROP‑ent在某些组合中进一步提升，说明不确定性可补充，但效果并非普遍显著。

**⚠️ 局限性**

局限性包括：① 仅在数学问题上验证，缺乏跨领域和跨模型的通用性证明；② 需要离线生成和人工/模型验证的改写–反事实triplet，成本较高；③ 采用硬性掩码在极端稀疏预算下可能导致重要token被误排；④ 对模型内部对比度的解释仍有限，无法完全量化因果效果。

---

## 454. Reconstructing Historical Manuscripts through MSI: The Potential of Contrast in Assessing Image Quality and Legibility

**arXiv ID:** 2608.13381 | [PDF](https://arxiv.org/pdf/2608.13381v1)

**作者:** Anna Breger `[一作]` `[通讯]` (University of Cambridge), Anna Breger (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估基于对比的图像质量指标（NPC、CNR等）在多光谱历史手稿图像恢复中的可行性。

**💡 创新点**

将归一化潜在对比(NPC)作为文档可读性评估指标并在两个实验中系统验证其与专家评分和全参考指标的相关性。

**🔧 技术方法**

使用归一化潜在对比(NPC)、对比噪声比(CNR)、传统统计指标(RMSC、熵)、无参考IQA（BRISQUE、NIQE、PIQE）以及全参考IQA（HaarPSI、Pearson、MS‑SSIM）与随机正交投影重建技术。

**📊 数据集**

使用 SALAMI（48 本古籍，250 个版本）和 Parchment（18 世纪羊皮纸，4 个严重降解区域）两组公开数据集。

**📈 对比分析**

通过 Spearman 相关系数与专家评分和全参考指标对比，NPC 在专家评分上最高（SRCC≈0.68），CNR 在全参考评估中最高（SRCC≈0.91），均优于传统无参考指标。

**⚠️ 局限性**

局限包括：Parchment 数据缺乏专家评分、对比指标依赖手工掩码、NPC 与全参考尺度不匹配、数据集规模有限，需要进一步验证。

---

## 455. NestDex: Nested Policy Learning with Copilot Assisted Teleoperation for Dexterous Manipulation

**arXiv ID:** 2608.13362 | [PDF](https://arxiv.org/pdf/2608.13362v1)

**作者:** James Zhao `[一作]` (University of Sydney), Weiming Zhi `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了NestDex框架，通过将可重用的内置手部技能嵌入演示收集流程，减少操作员对手指细节的控制，从而实现更可靠、更高效的柔性抓取演示；

**💡 创新点**

创新点在于：①利用单自由度旋钮调节基于状态的手部策略，实现与手臂控制的分离；②在演示期间采用视觉‑语言选择器动态切换手部技能；③使用手动作变分自编码器（H‑VAE）将高维手部动作压缩为低维潜在空间，简化外部策略学习；

**🔧 技术方法**

技术主要包括：强化学习式的动作块预测（action‑chunk），Transformer‑based proprioceptive/visuomotor策略，H‑VAE手部动作编码，视觉‑语言模型用于任务阶段的策略选择，以及基于多摄像头的手部姿态重定向；

**📊 数据集**

数据来源为六种真实世界的柔性抓取任务（如Tongs Transfer、Bottle Disposal、Dual‑Object Transfer、Ingredient and Pot Transfer、Toast Preparation、Binder Filing），通过多人摄像头捕获人类手部轨迹并重映射为机器人手部；

**📈 对比分析**

与传统AnyTeleop基线相比，NestDex在演示收集成功率和耗时上均有显著提升，且训练得到的外部策略在四项任务中成功率从65%–80%提升至100%；利用H‑VAE进一步提升成功率，且在线闭环执行与时序集成显著降低关节抖动和后期力学；

**⚠️ 局限性**

局限性包括：仍需人工操作员收集内置手部策略；依赖高质量多摄像头手部姿态估计；手部动作压缩可能导致极端场景下的细节丢失；以及在未见过的物体与环境中需进一步验证泛化能力。

---

## 456. Simulation-to-real transfer learning for infrared spectroscopic chemical sensing and analysis from molecules to complex samples

**arXiv ID:** 2608.13341 | [PDF](https://arxiv.org/pdf/2608.13341v1)

**作者:** Yusen Tan `[一作]` (Hong Kong University of Science and Technology Guangzhou), Jun Xia `[通讯]` (Hong Kong University of Science and Technology Guangzhou)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 UltraIR 的 100M 参数基础模型，通过大规模模拟红外光谱预训练与实验谱标注微调，实现从单分子到复杂样品的红外谱分析。

**💡 创新点**

创新点在于将模拟与实验谱结合的双阶段预训练框架、三目标（波形重建、指纹对齐、功能基预测）协同学习，并采用混合卷积-Transformer 编码器，显著提升多任务迁移与数据效率。

**🔧 技术方法**

技术包括基于 Savitzky‑Golay 的多通道输入、分层残差卷积、分块 Transformer、波形域重构、指纹相似度对比学习及多标签功能基预测，训练采用 AdamW、交叉熵、MAE 等损失。

**📊 数据集**

数据集涵盖约 60M 条模拟谱（IRtoMol、MD 生成、机器学习预测）、约 120K 条实验谱（NIST、SDBS、USPTO、DeepMIR、FTIR 组分、微生物、草本、微塑料、土壤等），用于预训练与下游任务评估。

**📈 对比分析**

与传统机器学习（XGBoost、RF、SVM 等）和专用深度学习模型（FCGFormer、IRtoMol 等）对比，UltraIR 在功能基预测、结构解析、属性预测、混合物组分检测与定量、细菌分类、草本鉴别、微塑料分类及土壤属性预测等多任务中均取得最高或接近最高准确率、最低误差、最高 R²，且在低样本和零样本迁移中表现更稳健。

**⚠️ 局限性**

局限包括仍需针对每个任务微调任务头、对实验谱的适配不足（模拟与真实的差距未完全消除）、缺乏完全零-shot跨任务推断，以及在极低标注或高度复杂多组分样品时仍可能受限。

---

## 457. RippleMem: From Isolated Retrieval to Associative Recollection for Long-Term Agent Memory

**arXiv ID:** 2608.13334 | [PDF](https://arxiv.org/pdf/2608.13334v1)

**作者:** Jingbo Ji `[一作]` (Communication University of China), Yunxiao Qin `[通讯]` (Communication University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RippleMem，一种面向LLM代理的长期记忆系统，将记忆访问从一次性检索转变为自适应关联回忆；

**💡 创新点**

创新点在于：①构建基于事件的记忆图并用多模态线索（语义+结构）实现关联；②在回忆过程中使用已检索到的记忆作为线索，动态扩展并补全缺失证据；③融合LLM进行记忆提取与检索规划；

**🔧 技术方法**

采用LLM驱动的记忆提取与检索规划、稀疏加权事件图（语义相似度+结构相似度）、混合检索与图遍历的关联回忆算法；

**📊 数据集**

在LoCoMo、LongMemEval‑S和EverMemBench三个长期对话记忆基准上进行实验；

**📈 对比分析**

相较于现有完整上下文、Mem0、Zep、SimpleMem等基线，RippleMem在LoCoMo的LLM‑as‑a‑Judge准确率提升至87.14%（比最佳基线高3.95%），在LongMemEval‑S提升至84.80%/86.60%；同时图构建时间约减少30×，构建时令牌量降低近50×；

**⚠️ 局限性**

仅评估文本对话，未覆盖多模态、实体交互或工具使用；LLM驱动的提取与规划增加延迟和成本；缺乏对极长历史、记忆衰减、隐私删除等极端场景的压力测试。

---

## 458. Integration-First Structural Coverage for Embedded Software:Trace-Based Evidence, Hybrid Runtime Analysis, and Cross-Variant Consolidation

**arXiv ID:** 2608.13322 | [PDF](https://arxiv.org/pdf/2608.13322v1)

**作者:** Alexander Weiss `[一作]` (Accemic Technologies GmbH), Michael Wittner `[通讯]` (Razorcat Development GmbH)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了基于嵌入式硬件跟踪的整合级结构覆盖测量流程，并提出了融合运行时分析（hRA）与 Hyper Coverage 整合层，能够在代表性构建上获得可靠的决策/条件覆盖证据。

**💡 创新点**

创新点包括：①以整合级覆盖为主的覆盖率流程，②混合运行时分析在优化编译下仍能观测决策与条件的技术，③Hyper Coverage 层将多层次、多变体的覆盖结果统一汇总并暴露源代码未覆盖行。

**🔧 技术方法**

使用嵌入式硬件跟踪（ARM ETM/CoreSight、Infineon TriCore）、轻量级源代码补丁以保留决策/条件边界、CEDARtools.Coverage 进行对象到源的映射、TESSY Hyper Coverage 进行多层次、多变体合并。

**📊 数据集**

以 SQLite 数据库引擎的测试用例在 ARM Cortex‑A53 1.5 GHz 处理器上执行，用作评估跟踪和 hRA 性能与覆盖效果的数据集。

**📈 对比分析**

通过比较传统 gcc/gcov 仪表化与仅基于跟踪的测量，发现仪表化导致代码量 +62.5%、执行时间 +147%；基于跟踪保持近基线；hRA 仅增加代码量 +1.3%、执行时间 +15.6%，同时覆盖率可靠性显著提升。

**⚠️ 局限性**

局限性包括：需硬件支持可访问的跟踪；构建环境必须可复现；映射仍需手工补丁，未覆盖所有编译器优化；不适用于无跟踪的设备；并不能完全替代单元测试，仍需在特殊场景下使用。

---

## 459. Keep, Customize, or Exit: Default Design and Token Pricing in LLM Reasoning Services

**arXiv ID:** 2608.13315 | [PDF](https://arxiv.org/pdf/2608.13315v1)

**作者:** Ahmet Bugra Gundogan `[一作]` (Bilkent University), Melih Bastopcu `[通讯]` (Bilkent University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个 Stackelberg 博弈模型，研究 LLM 服务提供者在设定每个令牌价格和默认推理令牌分配时，用户可接受默认、定制或退出，并推导出用户的最佳定制分配与默认可接受区间。

**💡 创新点**

创新点在于：1）将默认推理令牌分配视为内生的服务设计变量；2）对用户定制最优分配进行闭式解并用 Lambert W 函数描述默认可接受区间；3）通过三阶段规则将服务提供者最优默认确定为一维价格优化；4）分析默认便利性对分配的独立性，证明默认仅在有便利收益时才具备分配权。

**🔧 技术方法**

采用了 Stackelberg 博弈分析、凸性与最优性证明、Lambert W 函数求解、经济效益与成本模型（精度、令牌、延迟、价格、成本、便利性）以及一维价格最优化与 Berge 最大化定理。

**📊 数据集**

使用了两种开放权重推理模型（Qwen3‑8B 与 DeepSeek‑R1‑Distill‑Llama‑8B）在五个数学与科学基准（AIME 2024/2025、GPQA Diamond、GSM8K、HMMT 2025）上的实验数据。

**📈 对比分析**

通过将实验得到的精度、令牌消耗与延迟数据拟合到 Q(r)=D+A(1‑e⁻ʳb) 等式，验证模型并计算各基准下的均衡价格、默认分配与定制分配；实验结果表明模型能很好地描述精度与令牌/延迟的关系，且不同任务特性导致的平衡结果差异明显。

**⚠️ 局限性**

局限性包括：1）采用代表性用户与完全信息假设，忽略用户与任务异质性、私有信息与动态互动；2）只考虑单一 LLM 服务提供者，没有竞争；3）默认便利性 δ 的外部估计难以量化；4）实验仅在两款模型与五个基准上验证，未覆盖更大规模或多模型情况。

---

## 460. How Good are Foundation Models in Longitudinal MRI Disease Progression Reasoning?

**arXiv ID:** 2608.13309 | [PDF](https://arxiv.org/pdf/2608.13309v1)

**作者:** Wafa Al Ghallabi `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fahad Shahbaz Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并发布了Time‑Aware Multi‑View MRI Benchmark，用于评估医学影像中多时点、多视角的时间推理能力。

**💡 创新点**

创新点包括①将多视角（轴向、冠状、矢状）与时序信息统一到同一评估框架；②引入结构化定位指导任务，要求模型给出解剖边界、影像特征与干扰因素；③设计涵盖五个维度（时序推理、疾病进展、结构化定位、时间序列排序、时间点变化定位）的完整评测体系。

**🔧 技术方法**

采用了预训练的多模态视觉语言模型（GPT‑4o、Gemini、Qwen、InternVL、Llama‑4、MedGemma等）进行零样本推理，并对输入进行注册、偏差校正、强度归一化等预处理；同时使用了自动化与人工双重验证的问答生成流程。

**📊 数据集**

使用七大公开临床队列（胶质母细胞瘤、脑转移、阿尔茨海默、脑退化、前庭神经瘤等），共890例、3,200余个时点、3,920对专家验证的问答对。

**📈 对比分析**

与16种视觉语言模型进行零样本比较，评价指标包括最终答案准确率、推理分数和时间感知复合分数（TAC）。结果显示多视角输入能提升空间定位准确率（≈97%），但在体积量化与变化方向识别上仍低于16%；TAC最高为0.80，说明当前模型在时间顺序上相对可靠，却缺乏对变化类型的深入理解。

**⚠️ 局限性**

局限性包括①缺乏真正的3D体积处理，导致体积量化和几何推理受限；②多视角输入对小型模型产生信息过载，影响时间推理；③对临床标签的依赖和人工验证成本高；④模型在对抗噪声、伪影和不同扫描协议的鲁棒性不足。

---

## 461. VR-Themis: A Scalable Framework for Virtual Reality Application Clone Detection

**arXiv ID:** 2608.13290 | [PDF](https://arxiv.org/pdf/2608.13290v1)

**作者:** Gengyang Xu `[一作]` (Hong Kong Baptist University), Weizhi Meng `[通讯]` (Lancaster University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于层级-对象-行为（HOB）模型的两阶段VR应用克隆检测框架 VR-Themis。

**💡 创新点**

创新点包括：① 将HOB模型与三维度相似度度量（HED、GND、SBS）相结合；② 采用粗细两阶段聚类降低全量比对复杂度；③ 解决 Unity VR 特有资产、IL2CPP 逆向等技术难题；④ 构建了目前最大的 4,277 个 Unity VR APK 数据集。

**🔧 技术方法**

使用了 AssetStudio 及自研工具进行场景层级与资源提取，Blender 网格简化处理，树编辑距离、Jaccard 距离、余弦相似度等度量，DBSCAN 聚类以及脚本特征向量的逆向分析。

**📊 数据集**

数据集为 4,277 个 Unity‑based Meta Quest VR APK，来源涵盖 Meta App Store、SideQuest、itch.io、论坛和 Source R，构成了最大规模的 VR 克隆检测数据集。

**📈 对比分析**

方法先用 DBSCAN 进行粗层聚类，将 9,144,226 对比对压缩至 416,385 对，再通过 HOB 指标综合相似度阈值 80% 识别 307 个克隆；检测耗时约 31,000 秒，准确率 100%，无误报，误检率极低。

**⚠️ 局限性**

限制包括：难以判定原始/复制归属、仅支持 Unity 及离线 VR、无法完整处理 IL2CPP 逆向与动态资源加载、无法覆盖其他引擎/平台，以及公共资源共享可能导致的误判。

---

## 462. Capstan-driven Continuum Surgical Robot: Design, Modeling, and Perception

**arXiv ID:** 2608.13396 | [PDF](https://arxiv.org/pdf/2608.13396v1)

**作者:** Gang Zhang `[一作]` (Chinese University of Hong Kong), Shing Shin Cheng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出在转盘驱动的连续外科机器人中集成可变形张力传感器，并结合短厚梁多体静力模型与近端感知策略，实现了实时形状与接触力的感知。

**💡 创新点**

创新点包括：①在转盘驱动装置中嵌入柔性元件实现非侵入式缆绳张力测量；②使用短厚梁并行计算框架捕捉非平面缆绳‑梁耦合；③将张力、长度变形与近端六轴力/扭矩传感器解耦为两步估计，保证实时性能。

**🔧 技术方法**

采用可变形应变计张力测量、神经网络张力标定、短厚梁静力模型、并行矩阵运算、两步解耦估计、MATLAB Simulink 控制与 EtherCAT 通信等技术。

**📊 数据集**

通过实验收集缆绳张力（0–9.5 N）与应变计信号进行神经网络标定；单段与双段连续机器人在不同加载、弯曲平面下的位姿与相机追踪数据，用于验证模型。

**📈 对比分析**

与四种基准模型（BCM(SM)、CRM(SM)、CRM、EB(SM)）以及无并行版This(NG)进行比较；单段模型平均误差0.34–0.50 mm、更新率>200 Hz，显著优于基准；双段模型平均误差0.48–0.62 mm、更新率≈200 Hz，速度提升约10×；接触力/位置平均误差分别为1.89 g/1.93 mm。

**⚠️ 局限性**

主要限制在于制造工艺不稳定导致零点漂移、每次使用前需重新标定；多段机器人误差升高、对小接触力的感知敏感性低；长期稳定性与临床应用验证仍待进一步优化。

---

## 463. TopoIntent: Compiling Security Intent into Executable, Compliance-Checked Network Topologies

**arXiv ID:** 2608.13389 | [PDF](https://arxiv.org/pdf/2608.13389v1)

**作者:** Xiaokang Qu `[一作]` (University of Science and Technology of China), Linyuan Lü `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了TopoIntent系统，将自然语言安全意图编译为可执行、符合CIS Controls 8.1.2约束的网络拓扑，并通过Mininet模拟验证连通性与ACL行为；

**💡 创新点**

创新点在于：①使用SchemaContract定义统一数据合同，形成可检验的多阶段流水线；②结合检索增强生成（RAG）与分阶段融合，确保意图与参考模板的对齐；③在合规检查后实现增量结构修复与可执行反馈闭环，既保持用户设计完整性又提升合规率；

**🔧 技术方法**

采用的技术包括：LLM Qwen2.5‑72B‑Instruct进行意图解析、融合、合规检查、修复与测试生成；BGE‑M3向量检索匹配参考拓扑；CIS Controls v8.1.2作为结构性合规约束；Mininet实现可执行网络仿真与ACL验证；以及基于JSON/XML的结构化日志与可视化输出；

**📊 数据集**

使用的实验数据集：22个行业参考拓扑模板与44个合成意图（训练/检索集合），以及7个模板+14个合成意图（Finance/Government场景，hold‑out评估集合），所有模板和合规标签均通过多模型协作与人类审议生成；

**📈 对比分析**

与四个消融基线（Direct、RAG w/o Fusion、RAG w/o Repair、RAG w/o Feedback）对比。结果显示：在hold‑out集上，合规满足率由0.78提升至1.00，后ACL通过率由0.78提升至0.88；检索单向向量策略在Top‑1精度为0.75；整体流水线在结构回顾率、合规率和可执行通过率上均优于消融模型；

**⚠️ 局限性**

局限性包括：仅能验证可在拓扑层面体现的CIS子集，未实现完整认证；LLM判断可能产生误判；Mininet仿真缺乏状态化防火墙、云控制平面等细节；数据集规模有限，未覆盖所有行业和更大规模架构；增量修复可能引入冗余设备；反馈循环并非一次性通过所有测试，仍需人工干预。

---

## 464. Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents

**arXiv ID:** 2608.13384 | [PDF](https://arxiv.org/pdf/2608.13384v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过离线层级化注解索引和在线结构化查询引擎，将非结构化文本转换为可高精度分析的可查询结构化资源。

**💡 创新点**

主要创新在于SchemaLoop自动闭环生成多层次检索友好型注解模式，以及结构化查询引擎按成本递增的提取算子与属性重用机制。

**🔧 技术方法**

采用大语言模型进行schema诱导、轻量级模型与正则表达式逐层提取，以及SQL扩展执行计划与UDF的提取操作。

**📊 数据集**

在法律文本LCR、维基百科WikiText和网络数据SWDE三大真实语料库上进行实验。

**📈 对比分析**

与VectorDB+RAG、Graph RAG、ZenDB、Palimpsest、QUEST、Lotus、ClosedIE、GPT‑4o等基线相比，AnnoIndex在平均F1上达0.87，且在线LLM代价显著低于竞品，尤其在多跳关联与渐进推理查询上优势突出。

**⚠️ 局限性**

仍受限于离线构建所需的LLM调用与手工设定的顶层schema，且对极大规模动态变化的语料更新与实时新增文档的支持仍需进一步研究。

---

## 465. AmalthAI: An Open-Source Computer Vision Platform for Cultural Heritage

**arXiv ID:** 2608.13343 | [PDF](https://arxiv.org/pdf/2608.13343v1)

**作者:** Christos Chatzisavvas `[一作]` (Democritus University of Thrace), George Ioannakis `[通讯]` (Athena Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个开源的计算机视觉平台 AmalthAI，面向文化遗产专家提供无代码、可自托管、多用户的模型管理、训练、评估与推理工作流，并集成 Grad‑CAM 可视化与 VLM 解释，以便专家对预测结果进行解释性评估。

**💡 创新点**

1) 将自托管、可扩展的 MLOps（Docker、Kubernetes、Kubeflow、Katib）与无代码界面结合；2) 为文化遗产领域构建专属的解释性循环，结合 Grad‑CAM 与 Qwen2‑VL‑2B‑Instruct 文本解释；3) 采用多任务（分类、分割、目标检测）统一平台，支持多种经典模型和超参搜索；4) 提供多身份联合认证与本地数据缓存，满足受限数据安全需求。

**🔧 技术方法**

Flask 前端、Docker 容器化、Kubernetes（Kubeflow + Katib）实现可扩展训练；Grad‑CAM 生成热力图；Qwen2‑VL‑2B‑Instruct 生成文本解释；集成 ResNet、EfficientNet、MobileNet、ShuffleNet 等分类网络；UNet、DeepLabV3+、PSPNet 等分割网络；YOLOv8/YOLO11/YOLO26 等目标检测网络；支持数据增强、超参搜索与模型版本管理。

**📊 数据集**

实验室制备的黏土纺织印记数据集：1927 张用于原料分类、1837 张用于制作工艺分类的 RGB 图像，387 张带人工标注的分割掩码，用于印记区域定位。

**📈 对比分析**

对分类任务采用 ResNet18、EfficientNetB0、MobileNetV2、ShuffleNetV2 进行多次随机拆分；对分割任务采用 UNet、DeepLabV3+、PSPNet。结果显示：材料分类平均准确率 77.5%（EfficientNetB0 最佳），工艺分类平均准确率 85.6%；分割任务 mIoU 平均 90%（DeepLabV3+ 最佳）。与单一实验设置相比，多分割验证证明性能稳定，可与传统手工分析相比，显著提升了工作效率。

**⚠️ 局限性**

① 仅暴露常用超参，缺乏对损失函数、优化器、学习率调度等低层配置的访问；② 需要用户预先划分训练/验证/测试集，缺乏数据分布自动检查；③ 目前仅支持 2D RGB 图像，无法处理多光谱或 3D 数据；④ 角色访问控制有限，无法区分专家与高级用户；⑤ 对异常样本和极端类别不平衡的容错性不足。

---

## 466. Neural Quadratic Forms: A Unified Minimal Model for Sudden Learning and Scaling Laws

**arXiv ID:** 2608.13335 | [PDF](https://arxiv.org/pdf/2608.13335v1)

**作者:** Liu Ziyin `[一作]` (Massachusetts Institute of Technology), Isaac Chuang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了神经网络的通用二次近似形式 Neural Quadratic Forms（NQF），通过置换对称性推导出统一的训练动力学，解释突发学习和神经网络的尺度律。

**💡 创新点**

创新点在于将多种网络模块（MLP、CNN、注意力、MoE 等）映射到同一二次正交形式，提出结构矩阵 A(x) 并证明训练过程仅由低维量子参数 M=WWᵀ 和 μ 控制，可归约为 Lotka–Volterra 方程，并用该框架统一解释突发学习与功率律。

**🔧 技术方法**

使用对称性分析、泰勒展开、Landau 形式、梯度流和随机梯度下降推导；在可解极限下得到显式解；实现了 NQF 的数值模拟并与梯度下降、Adam、Polyak 动量等优化器对比。

**📊 数据集**

主要使用合成数据：低秩 NQF 教师、互斥样本、等距样本以及 Fourier MLP 训练目标；实验中也在标准 MLP 和 CNN 结构上验证。

**📈 对比分析**

方法上通过在同一训练循环中同时训练原始模块和其 NQF 近似，比较学习曲线与残差；在 Fourier MLP 上对比 NQF 预测的幂律指数与实验曲线，结果与理论一致；在多头注意力、混合专家等模块中 NQF 能精确跟踪原始训练损失。

**⚠️ 局限性**

局限性：仅在小初始化下有效；要求网络平滑、无 ReLU 等非平滑点；更高阶项导致未解释的平稳期；功率律的谱假设需要进一步验证；理论未覆盖自适应优化器（如 Adam），尽管实验中 Adam 仍能被 NQF 跟踪。

---

## 467. FIRE-VLA: Failure-Informed Self-Evolution for Vision-Language-Action Models in Autonomous Driving

**arXiv ID:** 2608.13395 | [PDF](https://arxiv.org/pdf/2608.13395v1)

**作者:** Hao Dou `[一作]` `[通讯]` (Harbin Institute of Technology), Hao Dou (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种失败信息自演化框架FIRE‑VLA，通过将低奖励低多样性的未解决失败组路由到同规模冻结教师进行特权自蒸馏，改进基于视觉语言的驾驶模型。

**💡 创新点**

创新点在于：①使用低奖励低多样性失败组做路由触发；②引入同策略、同参数规模的冻结教师仅凭未来轨迹提供特权监督；③在每轮迭代后将更新的策略推广为下一轮教师，使失败分布与教师同步演化。

**🔧 技术方法**

采用的技术包括：组相对策略优化（GRPO）、同策略教师‑学生自蒸馏（PSD）、批量奖励均值/方差路由判定、KL/JSD损失、离散化答案令牌监督、以及逐轮自演化更新。

**📊 数据集**

使用的主要数据集为 nuScenes，包含前视图图像、ego 历史描述与隐藏未来轨迹；评估集共 6,019 例，来自 150 个互斥场景，另外还使用 SFT 校准集来固定评估时的持久失败阈值。

**📈 对比分析**

在同一 SFT 检查点、相同样本数（4,800）和更新次数（150 次）下与标准 GRPO 进行对比；单样本低温预测平均 L2 从 0.642 m 降至 0.602 m，G=4 随机评估平均 L2 从 1.848 m 下降至 1.500 m（约 18.8%），持久失败率从 13.03% 降至 11.20%。

**⚠️ 局限性**

局限性包括：仅在单一训练种子上实验、计算量不等、未分别评估路由、蒸馏和教师推广各自贡献、评估仅为开放循环、未验证闭环安全或交通合规性。

---

## 468. TeleGapper: On the (un)reliability of Privacy Policies in Telegram Mini apps

**arXiv ID:** 2608.13390 | [PDF](https://arxiv.org/pdf/2608.13390v1)

**作者:** Luca Ferrari `[一作]` (IMT School for Advanced Studies Lucca), Luca Verderame `[通讯]` (University of Genova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 TeleGapper 框架，对 Telegram Mini Apps 进行黑盒动态分析，评估其隐私政策合规性。

**💡 创新点**

首次构建针对 Telegram Mini Apps 的动态隐私合规检测框架，并系统量化其隐私违规情况。

**🔧 技术方法**

利用 Appium 自动化、Burp 代理 MITM、正则提取与隐私政策匹配，结合多轮随机探索实现动态流量捕获。

**📊 数据集**

基于 tApps Center 的 991 条目录条目，随机抽取 278 条可运行 Mini Apps 进行实验。

**📈 对比分析**

通过与标准 Bot 隐私政策对比，统计违规比例（59.4%）和默认政策依赖率（78.8%），发现自定义政策并未显著提升合规性。

**⚠️ 局限性**

受限于单一账号、探索覆盖率有限、缺乏平台级监测，结果可能低估违规率，且仅适用于当时的采样。

---

## 469. The Use of Learning Management Systems for Self-paced Learning: The Case at a South African Public Access Centre

**arXiv ID:** 2608.13351 | [PDF](https://arxiv.org/pdf/2608.13351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 470. When Is a Task Vector Enough? An Empirical Theory of Implicit Multimodal ICL

**arXiv ID:** 2608.13385 | [PDF](https://arxiv.org/pdf/2608.13385v1)

**作者:** Jiaqian Li `[一作]` `[通讯]` (Brown University), Jiaqian Li (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过控制实验和自然VQA基准，研究了将多模态示例压缩为内部干预（task vector、条件向量、分布式、路由等）的隐式多模态上下文学习，并提出了选择–实现假设来预测最小足够的干预方案。

**💡 创新点**

创新点在于将示例压缩映射分解为（1）选择层：示例产生的变换是否在查询间共享；（2）实现层：变换如何在模型内部实现（局部、分布式或路由）。通过四个可检验假设和一套诊断指标，能够在无测试数据的情况下预估最优低成本干预。

**🔧 技术方法**

采用低秩表示学习、共享与条件选择预测、分布式/路由干预、基于显式M-ICL的计算诊断（共享度、预测R²、支持分散度、加法拟合-恢复差），以及对不同干预方案的多因子对比实验。

**📊 数据集**

受控合成任务（α 0–1），VQAv2、GQA、OK‑VQA、CVQA 四个多模态问答数据集。

**📈 对比分析**

方法对比包括静态向量、条件向量、局部/分布式加法、注意力路由等。受控实验验证了四个假设；自然任务中理论选择的平均相对成本降低约42%，与后验最优方法差距不到0.4个百分点，性能仅略低于最佳基线。

**⚠️ 局限性**

局限性：诊断基于显式M-ICL的前向计算，未考虑训练动态或超参数；对极高条件多样性或更大模型的适用性未知；仅评估了有限的干预种类，缺乏对更复杂路由或其他表示形式的深入分析。

---

## 471. Triangle-Free Coloring in LOCAL via Resilient Lovász Local Lemma

**arXiv ID:** 2608.13357 | [PDF](https://arxiv.org/pdf/2608.13357v1)

**作者:** Peter Davies-Peck `[一作]`, Xusheng Zhang `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在分布式 LOCAL 模型下，提出了一种新的算法，利用可容忍性（resilience）框架对 Pettie–Su 的三角形无色算法进行改造，使得所有涉及的 Lovász 本地引理（LLL）实例能够在 log^O(1)log n 圈内求解，从而实现对三角形无图以及奇环（girth‑5）图的 O(k)+log^O(1)log n 圈色调度，k≤(1/4‑ε)lnΔ，并且可用的颜色数趋近于最佳的 Δ/k 或 (1+ε)Δ/lnΔ。

**💡 创新点**

主要创新点包括：
- 将 Pettie–Su 的迭代着色框架与 Davies 的可容忍性 LLL 解决方案无缝结合；
- 设计了一种新的“色彩平衡”分区，使得每个坏事件在单一分区内的邻接数可控，从而实现了 r‑resilient 的 LLL 实例；
- 通过对三角形无图的平均 c‑度和残余度进行细致的上界分析，证明了在每一轮内所有坏事件的概率可压至 Δ^-c；
- 进一步将该框架推广到奇环图，直接获得了 (1+o(1))Δ/lnΔ 颜色的 log^O(1)log n 圈子算法。

**🔧 技术方法**

使用的技术主要有：
- 可容忍性 Lovász 本地引理（r‑resilience）与分区框架；
- 迭代随机着色与后续的色盘裁剪（palette pruning）与冲突消除；
- 复杂的分布式随机化分析（Chernoff、Hoeffding、联合大概率论证）；
- 对偶样本（two‑sample）混合技术，用于证明可容忍性；
- 递归参数调度（π_i, β_i, α_i, t_i, d_i 等）实现颜色数与度数的平衡。

**📊 数据集**

本工作为理论论文，无实验数据集，所有结果均通过严谨的概率上界与轮数分析证明，适用于任意最大度 Δ 的无环或奇环图。

**📈 对比分析**

与现有的 Pettie–Su 算法（O(k+log* n) 圈）相比，
- 颜色数保持在 Δ/k 或 (1+o(1))Δ/lnΔ；
- 轮数从 O(log n) 或 O(Δ/lnΔ)+log^O(1)log n 降至 O(k)+log^O(1)log n，尤其在 Δ≈(log n)^C 的“难点”区间实现了指数级的提升；
- 该结果是目前已知的最接近最优颜色数的分布式算法，同时实现了极低的时间复杂度。

**⚠️ 局限性**

局限性：
- 仍需要 Δ ≥ (log n)^C（即 Δ 不是过小的稀疏图）；
- 对一般 LLL 的全局求解仍未突破 log^O(1)log n 阈值；
- 主要针对三角形无图和奇环图，尚未直接推广到更一般的稀疏图类；
- 需要精细的参数调度和分区构造，复杂度实现上仍有理论与实践的差距。

---

## 472. MiLAC-Aided Beamforming for MIMO Over-the-Air Computation

**arXiv ID:** 2608.13353 | [PDF](https://arxiv.org/pdf/2608.13353v1)

**作者:** Yaru Wang `[一作]` (Beihang University), Chuang Shi `[通讯]` (Beihang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了基于微波线性模拟计算机（MiLAC）的MIMO AirComp系统，提出了交替优化（AO）框架，联合优化终端数字预编码矩阵与基站MiLAC聚合矩阵，以最小化聚合误差（MSE）

**💡 创新点**

创新点在于：①利用MiLAC的可逆、无耗散特性将其物理可行性约束转化为聚合矩阵的谱范数约束；②针对非凸的MSE最优化问题，设计了可行的KKT+二分搜索更新预编码矩阵，并将聚合矩阵子问题通过投影梯度下降（PGD）求解到全局最优；③证明PGD在该凸子问题上收敛至全局最优并在实现上实现了显著降低RF链需求

**🔧 技术方法**

主要技术包括：MIMO AirComp理论、微波网络理论（Cayley变换、散射矩阵）、凸优化与投影梯度下降、Karush–Kuhn–Tucker条件与一维二分搜索、SDP对比验证、网络合成方法将聚合矩阵映射到MiLAC电路参数

**📊 数据集**

使用的是模拟随机高斯信道（H_k ∈ CN(0,1)）以及均匀功率约束P_k=10，SNR由噪声功率σ_n^2决定，所有结果在1000次独立信道重现上取平均，没有使用真实数据集

**📈 对比分析**

对比方法包括：AO-SDP（用SDP求解聚合矩阵子问题）作为全局最优基准；全数字Beamforming（M RF链）和相位移器混合Beamforming（L RF链）作为硬件实现基准；实验结果显示MiLAC Beamforming在相同L RF链下几乎匹配全数字Beamforming的MSE，并显著优于相位移器混合Beamforming，尤其在大天线规模和多流聚合时差异更为明显

**⚠️ 局限性**

局限性包括：MiLAC需要全连通的网络，导致可调耦合电导数目呈二次方增长；目前未考虑实际元件的非理想性（如有限抑制、相位误差）；算法在大规模用户或高维信道下的收敛速度与复杂度仍有提升空间

---

## 473. StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems

**arXiv ID:** 2608.13317 | [PDF](https://arxiv.org/pdf/2608.13317v1)

**作者:** Yanwen Peng `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、无架构改动的隐式通信框架StateBridge，能够将发送方的最终层隐藏状态对齐到接收方的输入嵌入空间，并直接以连续前缀形式传递给接收方；

**💡 创新点**

创新点在于利用闭式正交Procrustes对齐加上范数校准与词汇锚定三步无学习参数接口，解决了隐藏状态与输入嵌入的几何不匹配问题；

**🔧 技术方法**

核心技术包括Procrustes正交对齐、白化归一化、范数校准、词汇锚定，以及在Transformer输入层直接注入连续前缀；

**📊 数据集**

使用了八个多样化基准数据集：数学推理（GSM8K、AIME24/25）、问答（GPQA-Diamond、ARC-Challenge、MedQA）和代码生成（MBPP+、HumanEval+）；

**📈 对比分析**

与文本通信(TextMAS)和KV-cache转移(LatentMAS)等基线比较，StateBridge在四种模型（Qwen3 4B/8B/32B、OLMo3-7B-Think）上在22/26任务对中获得最佳或相当成绩，平均提升约2.4-2.9个百分点；

**⚠️ 局限性**

局限在于对部分任务（如GSM8K）性能不如基线，且目前仅支持同构模型与固定前缀长度，缺乏对异构模型或自适应前缀长度的适配。

---

## 474. The Time Value of Evolution

**arXiv ID:** 2608.13297 | [PDF](https://arxiv.org/pdf/2608.13297v1)

**作者:** Matthew Siper `[一作]` (New York University), Julian Togelius `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Lineage‑Value Policy Gradients (LVPG)，在演化搜索中将弱子代的长期价值纳入决策，使交易策略搜索更有效。

**💡 创新点**

核心创新是把“进化的时间价值”量化为有限时域的后续收益，并通过路径回报（PPO‑Path）与基于树结构的 critic 结合，形成对多步后果的信用分配。

**🔧 技术方法**

技术手段包括：冻结的 Qwen3‑8B 演化语言模型（ELM）、带 LoRA 的 actor‑critic、PPO‑Path 与树结构 bootstrapped critic、ODPO 行为校准、强化学习的路径信用分配。

**📊 数据集**

数据集为 2025 年前的三种期货行情：S&P500 E‑mini、银价和 30 年期国债，约 24k 条行情记录，采用滚动起始、验证、封闭测试的 10 组折叠。

**📈 对比分析**

通过 90 对齐实验（LVPG 对比 PPO‑Immediate、固定/统一/时间表/单步/四步等控制），LVPG 在验证 AUC 上提升 0.394 Sharpe，封闭测试 Sharpe 上提升 0.459，且临时回调恢复率更高。

**⚠️ 局限性**

局限包括：变异空间固定且粗糙；critic 训练使用确定性树的最优值，存在乐观偏差；无法在线改进变异；实验仅涵盖三种期货、单一交易语言、有限搜索步长，未考虑实时市场冲击与流动性。

---

## 475. Who Speaks Matters: Authority-Aware Multi-View RAG over Italian Parliamentary Proceedings

**arXiv ID:** 2608.13410 | [PDF](https://arxiv.org/pdf/2608.13410v1)

**作者:** Mirko Tritella `[一作]` (University of Milano-Bicocca), Matteo Palmonari `[通讯]` (University of Milano-Bicocca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了面向意大利众议院的多视角检索增强生成系统 ParliamentRAG，支持用户查询时提供多党派平衡、权威度感知且可追溯的文本摘要。

**💡 创新点**

创新点在于：①将查询依赖的议员权威度模型与检索融合；②双通道检索（密集向量 + 结构图）保证多党派覆盖；③生成阶段使用占位符直接插入原始摘录，彻底消除生成式“自造引用”。

**🔧 技术方法**

技术包括：Neo4j 图数据库 + 向量索引，基于图的检索与权威度计算，GPT‑4o 或 Gemini 3 作为 LLM，定制的四阶段生成流水线（Analyze‑Generate‑Integrate‑Cite）。

**📊 数据集**

数据集为意大利众议院公开的 RDF/文本数据（议程、辩论、法案、议员资料），经转化为图结构共 232,755 节点、488,487 条关系；包含 151k 语料片段及 40,416 条发言。

**📈 对比分析**

通过 15 个主题的专家 A/B 评测与自动指标对比 NotebookLM 基线，结果显示 ParliamentRAG 在群组覆盖率（0.97 vs. 0.95）、引文真实性（1.00 vs. 0.95）以及源权威与覆盖度方面显著优于基线，但在回答质量与清晰度上略逊。

**⚠️ 局限性**

局限包括：评测规模有限（仅 15 题）、未进行内部消融实验、权威度权重仍为经验设定、检索依赖完整图数据，若缺失相关片段则无法保证覆盖。

---

## 476. Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation

**arXiv ID:** 2608.13391 | [PDF](https://arxiv.org/pdf/2608.13391v1)

**作者:** Hmrishav Bandyopadhyay `[一作]` (NVIDIA), Zian Wang `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种 Context-Matched Distillation（CMD）框架，用于训练低步长因果自回归视频生成模型，实现长时程、低延迟、可控摄像头交互生成。

**💡 创新点**

创新点在于：① 用因果教师替代双向教师，消除教师-学生信息集不匹配；② 引入 Prefix Scoring 让教师评分使用学生实际生成的前缀；③ 通过 Prefix Corruption 稳定训练并提升对早期失真前缀的鲁棒性；④ 统一因果架构适配长视频和摄像头控制。

**🔧 技术方法**

核心技术包括：扩散模型的分布匹配蒸馏（Distribution Matching Distillation）、Diffusion Forcing、块级因果注意力掩码、帧相对摄像头表示、Prefix Scoring 与 Prefix Corruption。

**📊 数据集**

训练与评估使用的公开数据集包括：Cosmos-Predict2.5-2B 生成视频、DL3DV 摄像头控制视频、VBench-I2V、SANA-WM（简单/硬）以及 HY-WorldPlay、LingBot-World、minWM 等基线。

**📈 对比分析**

与多种现有自回归方法（如 CausVid、Self-Forcing、LongLive、Rolling Forcing、Context Forcing、LingBot-World、Causal Forcing、Causal Forcing++）在短视频、长视频和摄像头控制任务上进行对比，CMD 在 VBench-I2V 的总分、I2V 分数、摄像头运动分数上均超过所有基线；在 SANA-WM 长视频任务中，CMD 的 Q、S、Total 分数均优于或接近最强基线；在摄像头控制任务中，CMD 在旋转、平移和 CamMC 错误上显著低于基线。

**⚠️ 局限性**

局限性包括：① 对极端摄像头变换或极长视频仍可能出现累计漂移；② Prefix Corruption 的超参数需经验调优；③ 训练过程仍需大规模算力，尤其是教师与学生的联合推理；④ 在更复杂的多模态交互（如语音、文本指令）下的通用性尚未验证。

---

## 477. A Dense Weisfeiler-Leman Algorithm for Deciding Bounded-Cliquewidth Homomorphism Indistinguishability

**arXiv ID:** 2608.13382 | [PDF](https://arxiv.org/pdf/2608.13382v1)

**作者:** Radu Curticapean `[一作]` (IT University Copenhagen), Ben Young `[通讯]` (IT University Copenhagen)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对图同构放宽的概念——图同构可区分性（homomorphism indistinguishability），本文首次给出了对稠密图类（尤其是树宽对应的 clique‑width）可判定性结果，并提出了稠密 Weisfeiler–Leman (dense WL) 算法来判定同一类图的可区分性。

**💡 创新点**

创新点包括：
- 设计了稠密 WL 算法，首次实现对 clique‑width ≤ k 图类的可判定；
- 证明了对任意 CMSo_1‑可定义、bounded clique‑width 图类的可判定性（随机指数时间）以及对 bounded linear clique‑width 的 PSPACE‑完备下界；
- 将 dense WL 与 homomorphism counts 及逻辑 C_W^k 进行等价性表述，形成稠密图类的统一框架；
- 与稀疏类（树宽、路径宽）相对比，揭示了稠密与稀疏判定难度的根本差异。

**🔧 技术方法**

采用的技术手段有：
- 对 clique‑width 的代数表示（partitioned graphs 与运算 ⊕, η, ρ），并构造其在线性代数中的矩阵表示；
- Möbius 与 Zeta 变换实现子集算子与边插入运算的高效实现；
- 乘法与莫比乌斯变换的组合形成 dense WL 的颜色更新；
- 将判定问题化简为多重态自动机（multiplicity automata）等价性测试；
- 逻辑 C_W^k（计数逻辑）与 WL 的等价性证明；
- 通过复杂性理论（PSPACE/PSPACE‑c等）与可约性构造下界。

**📊 数据集**

论文并未使用公开数据集，而是通过构造性证明与理论构造（如生成特定图类、构造图 G、H、自动机等）来展示算法与下界。

**📈 对比分析**

方法比较：
- 对比经典 WL，dense WL 在 clique‑width ≤ k 上可区分更强；
- 运行时间为 4^{k}·n  的指数级（随机指数时间），相比于稀疏类的多项式或 quasi‑polynomial 结果，显著更高；
- 对 bounded linear clique‑width 的问题则达到 PSPACE‑完备，下界与上界匹配。

**⚠️ 局限性**

限制与未解问题：
- 算法复杂度仍为指数级，基数 4 的指数项是否最优尚不清楚；
- 只针对 clique‑width 与 linear clique‑width 进行判定，其他稠密参数（rank‑width、fusion‑width 等）尚未覆盖；
- 目前仅给出随机化算法，是否存在确定性多项式/量子多项式算法未知；
- 对 dense WL 在更广泛稠密图类的判定性、表达力与游戏等的理论仍在探索中。

---

## 478. When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation

**arXiv ID:** 2608.13365 | [PDF](https://arxiv.org/pdf/2608.13365v1)

**作者:** Shuhan Wang `[一作]` (University College London), Chi Wang Cheung `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 RoPE 结构下，对 Q/K 的正交旋转变换进行理论分析与实证评估，比较全头 Hadamard 与头共享对角对旋转以及两者的组合，并证明单头可交换正交变换仅为各频率对内旋转，同时推导出头共享参数化的闭式最优旋转角度并验证其实现。

**💡 创新点**

1) 给出 RoPE 单头可交换正交变换的完整逆表征；2) 对头共享参数化推导并验证闭式最优角度；3) 通过实验揭示即使满足局部最优，仍无法提升动态 INT4 量化精度，说明优化目标与量化尺度统计以及混合支持的不匹配对性能的关键影响。

**🔧 技术方法**

旋转量化、RoPE 频率对分解、正交矩阵中心化、池化协方差代理、动态 INT4 量化、Perplexity 测评、block‑Hadamard 混合支持插值、不同角度估计方法（默认、K‑only、流平衡）等。

**📊 数据集**

WikiText‑2（128 句 2048 长度）、Proof‑Pile、PG19、LongBench‑v2、NIAH 评估数据集，使用 Llama‑3.2‑1B/3B、Llama‑3.1‑8B、Mistral‑7B‑v0.3 三种模型。

**📈 对比分析**

使用相同校准、权重和随机种子，比较全头 Hadamard、仅对角对旋转以及两者组合；在短、长上下文下报告 perplexity、K 范围、相对量化误差；结果显示仅对角对在所有检查点的 perplexity 均高于全头 Hadamard，组合满足 ±0.05‑PPL 等价；通过混合支持插值发现混合支持越大，K 范围、相对量化误差和 perplexity 降低。

**⚠️ 局限性**

仅针对 RoPE 动态 INT4 量化，未覆盖学习旋转、通道缩放、静态量化、不同位宽、头依赖角度等；实验仅在少数种子、数据集和模型上，结论受限；未能完整解析目标与量化尺度统计差异的因果机制。

---

## 479. LongEarth-R1: Benchmarking and Aligning Vision-Language Models for Long-Horizon Earth Observation Reasoning

**arXiv ID:** 2608.13344 | [PDF](https://arxiv.org/pdf/2608.13344v1)

**作者:** Yupan Ding `[一作]` (Wuhan University), Mi Wang `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出长时空遥感推理框架并构建LongEarth-Bench基准，训练LongEarth和LongEarth-R1模型。

**💡 创新点**

创新在于将四维认知层级（演化摘要、空间推理、异常识别、逻辑预测）融入基准，使用序列标识+结构化链式推理和GRPO奖励实现跨帧推理与空间一致性。

**🔧 技术方法**

技术包括Qwen2.5‑VL‑7B预训练模型，Seq. ID序列标识、结构化CoT监督、低秩适配LoRA、组相对策略优化GRPO与格式/时间/空间奖励。

**📊 数据集**

数据集来源包括SpaceNet7、SDSU MidWest Flood、DynamicEarthNet、FLAIR#2、PASTIS‑R，共117k图像，120k QA样本，其中30k样本含结构化推理轨迹。

**📈 对比分析**

在12项长序列任务上，LongEarth‑R1在所有任务上均获最高分，尤其异常识别提升12.6%；在单图、双图及短序列遥感基准上保持竞争力，六项九项中的前六。

**⚠️ 局限性**

局限在于极长序列（>30帧）仍易失真，仍需更高效处理长上下文；奖励设计对不同任务的通用性待验证；未公开完整模型参数。

---

## 480. It's How You Ask: Gender-Associated Linguistic Bias in LLMs

**arXiv ID:** 2608.13328 | [PDF](https://arxiv.org/pdf/2608.13328v1)

**作者:** Katherine Van Koevering `[一作]` (Johns Hopkins University), Anjalie Field `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究探讨了专业沟通中LLM对女性关联语言特征的偏差，并验证了提示词中的性别化语言会导致模型输出更短、复杂度更低、正式度更低的结果。

**💡 创新点**

首次量化提示词性别化语言对LLM生成质量的系统影响，并通过机制解释揭示早期Transformer层对这些特征的强编码及因果贡献。

**🔧 技术方法**

使用Prompt injection、GPT‑4生成修改、统计评估（t检验、回归、bootstrap中介）、线性探针、激活补丁与激活引导等技术进行实验和可解释性分析。

**📊 数据集**

基于WildChat‑4.8M真实工作场景提示，并在GPT‑4、Gemma‑27B、Mistral‑7B、Llama‑3.1‑3B等四大模型上进行测试。

**📈 对比分析**

通过配对t检验和回归证明WALF提示比MALF导致的输出在长度、词汇复杂度、可读性、正式度上显著差异；实验表明早期层的补丁可产生高KL变化，验证因果关系。

**⚠️ 局限性**

局限包括只关注二元性别化特征、对提示重写的人工验证仍可能留有残差、仅分析单一模型架构、未覆盖多语言或其他社会属性。

---

## 481. Beyond Local Accuracy: A Protocol-Level Identifiability Audit for Controlled LLM Reasoning Evaluation

**arXiv ID:** 2608.13326 | [PDF](https://arxiv.org/pdf/2608.13326v1)

**作者:** Junhao Luo `[一作]` (Southwestern University of Finance and Economics), Wei Deng `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该研究对LLM评估协议进行识别审计，验证观察支持是否能唯一确定目标行为估计量。

**💡 创新点**

创新点在于将识别问题形式化为有限行为策略类的点识别判据，并利用碰撞结构自动合成最小识别支持。

**🔧 技术方法**

技术包括基于解算器的三值推理、行为策略类枚举、观察等价关系判定及最小覆盖集搜索。

**📊 数据集**

数据集为基于开世界符号Horn理论构造的三世界（基、目标、虚假）评估集，包含24个平衡集群、48个模型集群和一个合成第二源。

**📈 对比分析**

通过与全支持、留一支持和随机抽样的对比，实验证明在两条指令微调模型上，即使局部准确率高达62%，但交互响应保真度仅约32%，显示两者不可互换。

**⚠️ 局限性**

限制在于仅针对冻结的七策略类、解算器驱动的实验，缺乏对自然语言任务和更广泛模型行为的外部有效性验证。

---

## 482. Completeness and incompleteness of basic matching logic

**arXiv ID:** 2608.13306 | [PDF](https://arxiv.org/pdf/2608.13306v1)

**作者:** Xiaohong Chen `[一作]` (Intent Computing, Inc.), Grigore Rosu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了无定义符号、无递归点的基本匹配逻辑（单一排序）在全局意义下完备，并展示了多排序、定义符号以及匹配 μ‑逻辑等扩展时的完备性失效。

**💡 创新点**

引入了语义局部化技术 Δ_Γ 与双覆盖构造，将全局蕴含转化为局部蕴含，从而首次给出单排序基本匹配逻辑的全局完备性证明；同时给出了多排序、定义符号和 μ‑逻辑的最优不完备性结果，阐明了与模态、混合逻辑的表达边界。

**🔧 技术方法**

使用语义局部化、双覆盖构造、归纳与紧致性论证、可计算性分析（递归不可枚举性）以及对局部完备性（(L)）和完备性 (S) 的利用。

**📊 数据集**

无；本文为纯理论研究，无实验数据或数据集。

**📈 对比分析**

通过与已知模态逻辑、混合逻辑以及匹配 μ‑逻辑的完备性/不完备性结果对比。对单排序基本匹配逻辑证明完备性；对多排序、定义符号或 μ‑逻辑构造反例，展示不完备性；通过构造证明和可计算性分析说明相对性能。

**⚠️ 局限性**

局限性：结果仅适用于无定义符号、无递归点的匹配逻辑；多排序或含定义符号时不完备；缺乏对含定义符号但无命名符号的全局完备性理论；μ‑逻辑缺乏有效可完备证明；多排序情况下所需的“可达性”条件尚未完全确定。

---

## 483. Large-scale Testing Global Optimization Methods with Black-box Adversarial Attacks

**arXiv ID:** 2608.13296 | [PDF](https://arxiv.org/pdf/2608.13296v1)

**作者:** Wojciech Zarzecki `[一作]` (Warsaw University of Technology), Jarosław Arabas `[通讯]` (Warsaw University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将黑盒对抗攻击（BBAA）视作高维多模态全局优化问题，构建了基准框架并评估了多种进化/元启发式算法的攻击效果。

**💡 创新点**

创新点在于首次将BBAA与全局优化基准相结合，并公开了可复现的实验平台，为后续研究提供了统一测试环境。

**🔧 技术方法**

采用了多种进化算法（DE、JADE、SADE、SHADE、GWO、INFO）以及随机搜索等元启发式方法，探索其在对抗攻击中的表现。

**📊 数据集**

实验使用了两个主流图像分类数据集：CIFAR‑10和ImageNet，分别在对应的预训练模型上进行攻击。

**📈 对比分析**

实验结果表明，DE、GEN、SHADE在成功率与扰动幅度方面均优于GWO和INFO；在无正则化时GWO表现最差，而INFO在查询效率上虽低，但成功率相对稳定。

**⚠️ 局限性**

局限性包括仅研究无目标攻击、仅使用L₂正则化、未加入感知损失或更复杂的目标函数；此外，实验仅覆盖两类数据集，未进一步探索更大规模模型或不同攻击场景的适用性。

---

## 484. More Than 63% of IEEE VIS Research Liable to be Retracted?! Ethics Approval Statements Protect Participants (and Researchers!)

**arXiv ID:** 2608.13295 | [PDF](https://arxiv.org/pdf/2608.13295v1)

**作者:** Lonni Besançon `[一作]` (Linköping University), Tobias Isenberg `[通讯]` (Université Paris-Saclay)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统分析了 IEEE VIS 2024 与 2025 会议论文中对人类参与者研究的伦理审批与知情同意报告情况。

**💡 创新点**

创新点在于首次量化评估近两年 VIS 论文的伦理报告缺失程度，并提出针对投稿阶段的改进建议。

**🔧 技术方法**

采用人工编码结合 ChatGPT‑5.5 自动识别技术，对论文 PDF 进行双重审核。

**📊 数据集**

数据集为 IEEE TVCG 2025 与 2026 年特刊中收录的所有 VIS 会议论文（共计若干篇）。

**📈 对比分析**

通过与 LLM 输出比对，发现人工编码与自动检测在识别伦理信息上误差率低于 10%，验证了自动化辅助分析的有效性。

**⚠️ 局限性**

局限性包括仅手工检查正文而未检索所有补充材料、样本仅为两年会议论文、对部分隐性伦理表述可能仍存在主观判定。

---

## 485. NAS-Driven Hardware Accelerator Exploration for Edge AI and Quantization Effects on the Pareto Space

**arXiv ID:** 2608.13293 | [PDF](https://arxiv.org/pdf/2608.13293v1)

**作者:** Eleftherios Mylonas `[一作]` (University of Patras), Alexios Birbas `[通讯]` (University of Patras)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个三阶段的硬件感知NAS流程，将NAS搜索、后训练量化桥接以及基于CGRA4ML的硬件DSE相结合，实现了Edge AI在可重构加速器上的高效部署。

**💡 创新点**

创新点包括：①将INT4后训练量化（PTQ）作为NAS后的桥接，系统性评估其对Pareto前沿的影响；②证明FP32零射击surrogate在INT4域中同样有效，避免了量化相关的搜索空间膨胀；③构建了演化算法驱动的硬件DSE框架，自动寻找最佳CGRA配置。

**🔧 技术方法**

技术手段：NAS-Bench-201硬件无关前端、Pareto rank surrogate、Brevitas后训练量化、CGRA4ML可重构加速器、演化算法及Pareto稳定性指标。

**📊 数据集**

使用数据集：CIFAR-10作为训练/验证数据；NAS-Bench-201搜索空间（15625个预训练模型）作为实验对象。

**📈 对比分析**

比较方法：在RS和MOEA两种搜索策略下，比较FP32零射击surrogate与专门训练的INT4 surrogate在归一化全局超体积（Normalized Global Hypervolume）与平均准确率上的表现；量化后对Pareto前沿的生存率、支配翻转率、Kendall Tau相关性等稳定性指标进行评估；硬件DSE对不同架构的时钟周期、PE利用率和面积进行比较。

**⚠️ 局限性**

局限性：仅针对INT4量化，未探讨更高或混合精度；实验仅在NAS-Bench-201搜索空间进行，缺乏对更大、更复杂模型的验证；未深入分析FP32到INT4权重迁移的细节与可能的性能波动。

---

## 486. Refine After Generation: Toward Correct and Concise Patches in LLM-based Program Repair

**arXiv ID:** 2608.13292 | [PDF](https://arxiv.org/pdf/2608.13292v1)

**作者:** Wenqiang Luo `[一作]` (City University of Hong Kong), Haoye Tian `[通讯]` (Aalto University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 LLM 自动程序修复中补丁冗长的问题，提出后生成补丁精简器 RECAP；

**💡 创新点**

创新点在于将补丁精简作为后置适配器，使用多源精简数据集并结合监督微调(SFT)与直接偏好优化(DPO)，实现可读性与正确性兼顾；

**🔧 技术方法**

采用 LLM 细调、直接偏好优化、链式思维（reasoning trace）蒸馏、插件式适配器架构，并在 SWE‑bench 上进行实验验证；

**📊 数据集**

构建 5,540 对补丁数据集，包括 1,227 条函数级对、533 条混合提交对及 3,780 条 SWE‑bench 训练集生成的夸大补丁；

**📈 对比分析**

通过与提示、Commit‑untangling、AdaPatcher、PRepair 四种基线以及 UR/JGR/OGR 三种模式在 SWE‑bench Verified 上对比，RECAP 在四主机上将总变更率降至 +4.24%、净变更降至 -39.75%，并保持或提升已解决实例数，优于所有基线；

**⚠️ 局限性**

局限性：评估仅在 SWE‑bench；正确性仅基于测试套件；适用于仓库级修复，未验证跨语言、私有仓库或无测试环境；数据生成过程依赖 LLM，可能引入噪声。

---

## 487. Three trees suffice for a constant stretch in minor-free graphs

**arXiv ID:** 2608.13508 | [PDF](https://arxiv.org/pdf/2608.13508v1)

**作者:** Hung Le `[一作]` (University of Massachusetts Amherst), Tuan Tran `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明任意给定图 H 的 H‑子图自由图都存在一组大小为 3 的树覆盖，并且其拉伸因子为常数；此外，若图宽度受限，则只需 2 棵树即可。

**💡 创新点**

首次将 Assouad–Nagata 维数与树覆盖联系起来，利用刘等人给出的 minor‑closed 类的维数上界，得到匹配的上界，完整解决了固定树数下的常数拉伸问题。

**🔧 技术方法**

采用控制函数构造可着色稀疏覆盖 → 构造层次划分族（HPF） → 由 HPF 得到树覆盖；核心技术是维数控制与稀疏覆盖的组合。

**📊 数据集**

本文为纯理论研究，无实验数据集。

**📈 对比分析**

结果与 Chen、Tan、Xu 对托洛斯网格给出的下界完全匹配，表明三棵树足以获得 O(1) 拉伸；未做实验对比，仅给出理论证明。

**⚠️ 局限性**

仅给出存在性与常数拉伸系数（未给出具体数值）；未讨论树覆盖的构造算法复杂度，也未考虑更一般度量空间的适用性。

---

## 488. Sparse Orthogonal Regression Technique: A Spectral Framework for Equation Discovery, Approximation, and Integration

**arXiv ID:** 2608.13504 | [PDF](https://arxiv.org/pdf/2608.13504v1)

**作者:** Sabin Roman `[一作]` (Jožef Stefan Institute), Saso Dzeroski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SORT——一种基于稀疏正交基回归的谱框架，用于从噪声、稀疏数据学习稀疏正交系数，可用于方程发现、积分与非线性逼近。

**💡 创新点**

将稀疏正交系数视为可重用中间表示，突破传统库基稀疏回归与符号回归的限制，并实现逐阶一致的模型增长。

**🔧 技术方法**

L1 正则化稀疏回归、正交基表示（Fourier、Legendre、Chebyshev 等）、稀疏系数估计、系数读出实现积分、基于正交系数的向量场重构。

**📊 数据集**

Dynobench（多种多维系统）、自定义周期、三角、Bessel、Lorenz 等系统；单维 Fresnel 与高维 Gaussian 积分测试；噪声与子采样实验。

**📈 对比分析**

与 SINDy、Dense LS、Kernel、PCE 等基线在相同采样、噪声、子采样条件下比较；SORT 在稀疏采样与噪声下更稳健、逐阶逼近保持低阶系数稳定；在积分实验中匹配解析结果；在逼近实验中低误差且系数层级一致。

**⚠️ 局限性**

受限于基函数选择（需先设计合适基），高维时基数爆炸、稀疏收敛仍受样本限制；无法直接得到简洁符号表达，需要后续符号搜索；对极端噪声或严重表示不匹配仍可能失效。

---

## 489. Runtime Monitoring of Distributed Cyber-Physical Systems Without a Global Clock

**arXiv ID:** 2608.13486 | [PDF](https://arxiv.org/pdf/2608.13486v1)

**作者:** Charles Koll `[一作]` (Oregon State University), Houssam Abbas `[通讯]` (Oregon State University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了首个针对没有全局时钟的分布式网络物理系统（CPS）进行稠密时间监控的理论框架与离线监测算法，并给出了该框架下的满足域（satdomain）计算方法。

**💡 创新点**

创新点在于：①定义了在部分同步设置下的“切片（cut）”与“重定时（retiming）”概念，构建了全局时刻的等价判定；②提出了DiSTL这一 STL 的子语言，只保留所有时序算子但不需要追踪多条重定时路径，从而实现可计算性；③设计了基于多维多面体几何的内外逼近算法，能够在可接受的误差范围内完整列举所有可能满足的全局时刻。

**🔧 技术方法**

技术手段包括：部分同步模型（NTP式时钟漂移约束）、信号时序逻辑（STL）与其语义扩展、非凸多面体与盒子分解的几何运算、重定时路径的曲线表示、离线求解的内外逼近算法，以及用 Rust 编写的高效实现。

**📊 数据集**

实验使用合成的连续时间信号：每个信号在 6 秒内以 10 毫秒为周期线性切换 ±1，平均根次数为 10 次/秒，共生成 10 条分布式信号；N 从 4 递增至 63，时钟偏差 δ 从 0.16 至 6 秒。

**📈 对比分析**

性能评估通过对比监测不同公式（布尔、带间隔的 Until、嵌套 Until）在不同 N 与 δ 下的运行时间来体现。实验表明：公式1可支持至 57 只代理，公式3至 9 只，公式2至 6 只；运行时间随 N 与 δ 增大而呈对数增长，但在一定范围内保持低于 6 秒，可实现实时监测。

**⚠️ 局限性**

局限性包括：①仅支持 DiSTL 子语言，无法表达所有 STL 公式；②算法采用内外逼近，无法给出完全精确满足域；③计算复杂度随代理数与多面体分解细度上升，导致大规模系统下性能下降；④目前仅为离线监测，尚未实现真正实时系统的在线监测。

---

## 490. Decoding Task Progress from VLA Representations

**arXiv ID:** 2608.13474 | [PDF](https://arxiv.org/pdf/2608.13474v1)

**作者:** Atiksh Bhardwaj `[一作]`, Preston Culbertson `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Vision‑Language‑Action 模型内部的任务进度信号，使用线性探针解码并应用于无监督 OOD 检测、语言对抗诊断和可控性分析。

**💡 创新点**

提出弱/强可解码和可控性概念，发现任务进度可在预训练 backbone 中线性可读，并构建跨任务、跨扰动的无监督 OOD 检测器。

**🔧 技术方法**

使用线性探针、对抗/伪标签训练、注入实验（steerability）、以及与 Mahalanobis、VAE、SAFE-MLP/LSTM 等基线的对比；基于 PaliGemma backbone 与 π_0.5 VLA 模型。

**📊 数据集**

使用 VLABench（10 个机器人操控任务，每个约 500 条轨迹）以及生成的 ID/OOD 扰动轨迹进行实验。

**📈 对比分析**

与 Mahalanobis、VAE、SAFE-MLP/LSTM 等有监督 OOD 检测器比较，per‑replan AUROC 在未见任务/扰动上保持 0.796→0.833，未见任务 AUROC 达 0.871/0.960，整体优于无监督基线；steerability 实验未能实现有效注入。

**⚠️ 局限性**

仅在单一 VLA 架构与仿真环境下验证；任务进度定义为时间比例，缺乏真实完成标签；对抗标签为代理；steerability 实验仅试验单层单方向注入；未在真实机器人上测试。

---

## 491. Concept Drift Detection and Adaptive Retraining of Malware Classification Models

**arXiv ID:** 2608.13465 | [PDF](https://arxiv.org/pdf/2608.13465v1)

**作者:** Christofer Washington Berruz Chungata `[一作]` (San Jose State University), Mark Stamp `[通讯]` (San Jose State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在恶意软件分类中检测概念漂移的方法，并在漂移感知、静态和周期性重新训练三种场景下评估模型性能。

**💡 创新点**

创新点在于首次将 One-Class SVM 与 Minibatch K‑Means 以及 MMD 三种技术结合使用，并通过 Pareto 前沿分析系统地权衡准确率与训练效率。

**🔧 技术方法**

采用的技术包括 One‑Class 支持向量机、Minibatch K‑Means 聚类、Maximum Mean Discrepancy 统计检验、Optuna 超参数优化以及 ParetoSet 进行非支配前沿提取。

**📊 数据集**

实验基于 KronoDroid 数据集，使用 23,190 个真实 Android 恶意软件样本，构建 20 对恶意软件族的二分类任务。

**📈 对比分析**

通过比较静态、周期性及漂移感知场景下的准确率与模型训练次数，结果显示 OCSVM 在保持 98% 以上准确率的同时，训练次数比周期性重新训练下降约 60%，且耗时最低；MK‑Means 仅稍逊；MMD 在稳健性上表现最好。

**⚠️ 局限性**

限制包括固定窗口大小 50、超参数搜索空间大导致计算开销、未在实时恶意软件检测系统中验证可扩展性，以及缺乏对不同窗口长度和深度学习漂移检测器的进一步探索。

---

## 492. Academic League of Artificial Intelligence - An Integrative Perspective of Teaching, Research, and Extension

**arXiv ID:** 2608.13447 | [PDF](https://arxiv.org/pdf/2608.13447v1)

**作者:** Alison R. Panisson `[一作]` (Federal University of Santa Catarina), Roberto Rodrigues-Filho `[通讯]` (Federal University of Santa Catarina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实施了基于学生主体的学术联盟组织框架，以整合教学、科研和扩展活动。

**💡 创新点**

将学术联盟与教学、科研、扩展三大高校支柱相结合，形成可复制的项目化治理模式。

**🔧 技术方法**

采用民主选举、项目治理、学习小组、知识库等组织技术实现学生领导和知识共享。

**📊 数据集**

未使用专业数据集，主要以学生项目产出和公开资源为依据。

**📈 对比分析**

通过案例描述和定性反思进行评估，未给出量化性能指标。

**⚠️ 局限性**

仅在UFSC单一学术联盟中实施，缺乏跨机构量化验证。

---

## 493. Blue Noise as a Lattice Gibbs Ensemble

**arXiv ID:** 2608.13446 | [PDF](https://arxiv.org/pdf/2608.13446v1)

**作者:** Zhuoran Yi `[一作]` `[通讯]` (University of Utah), Zhuoran Yi (University of Utah)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于格点 Gibbs 分布的蓝噪声点集生成方法，提供可控参数和可局部采样。

**💡 创新点**

创新点在于将蓝噪声视为可显式定义的 Gibbs 分布，并通过反向 Markov 链和截断实现可预测的局部依赖，支持任意大小、可流式、可并行的生成。

**🔧 技术方法**

使用离散格点 Gibbs 采样、Coupling Towards The Past、有限范围近似、并行 Jacobi 迭代以及局部自适应密度校准。

**📊 数据集**

使用合成周期域点集以及约 14557×8418 的 Tarantula Nebula 灰度图作为自适应密度输入。

**📈 对比分析**

与现有 Lloyd、BNOT、GBN、dart throwing、SOT 等方法在周期域下的功率谱和密度指标上比较，蓝噪声特征相当，且在大规模（14K）下内存仅为 8–50 MiB，显著优于传统方法。

**⚠️ 局限性**

局限在于格点离散化、极端高秩参数需更大邻域导致内存/时间升高，且对非对称或高维扩展缺乏理论保证。

---

## 494. Algorithmic Gender Prediction Is Illegitimate, But Gender Imputation Can Yield Valid Measurements

**arXiv ID:** 2608.13444 | [PDF](https://arxiv.org/pdf/2608.13444v1)

**作者:** Evan Dong `[一作]` (Cornell University), Angelina Wang `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了算法性别预测在公平性评估中的合法性与有效性冲突，区分两者并提出在不伤害跨性别者前提下使用性别推断的准则。

**💡 创新点**

首次将性别预测的合法性（对自我决定权的侵犯）与有效性（是否能准确测量歧视）区分，并辨别传统性别歧视与对跨性别者的对立性别歧视，形成新的责任与实践框架。

**🔧 技术方法**

主要使用理论分析与案例研究，讨论现有的面部识别、姓名推断、生成模型评估等方法及其统计属性，并未引入新的机器学习模型。

**📊 数据集**

引用了美国社会保障姓名数据库、BISG、电影帧数据、生成图像等公开数据集，但核心工作是对这些数据来源的伦理与技术评估，而非直接实验。

**📈 对比分析**

通过对比合法性/有效性维度评估不同推断方法在传统性别歧视测量中的表现，指出使用连续概率输出与聚合指标可提升收敛性与一致性，但缺乏统一的性能评估框架。

**⚠️ 局限性**

研究聚焦于性别差距测量，未覆盖其他歧视形式；方法依赖对性别概念的细化，实际应用中仍难以完全消除对跨性别者的伤害；缺乏经验验证与可量化的公平性改进指标。

---

## 495. Algebraic Decomposition Theory for Transformer Length Generalization

**arXiv ID:** 2608.13433 | [PDF](https://arxiv.org/pdf/2608.13433v1)

**作者:** Andy Yang `[一作]` (University of Notre Dame), Michael Hahn `[通讯]` (Saarland University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Transformer模型在正则语言上的长度泛化能力进行了完整的理论刻画，并给出了多项决策算法。

**💡 创新点**

创新点在于：①提出C‑RASP语言类并证明其与Transformer长度泛化完全对应；②发展了新的代数分解理论，利用整数加法群与无限单子（尤其是有界Dyck语言）的迭代wreath积来描述C‑RASP；③设计了多项式时间的决策算法；④给出了必要但非充分的等式判据。

**🔧 技术方法**

技术主要包括代数语言理论（syntactic monoid、R‑relation、wreath product、typed monoid、derived category）、无穷单子与关系同态的构造、以及基于无穷单子的分类与判定算法。

**📊 数据集**

实验使用了125个人工生成的正则语言（覆盖C‑RASP、其子类与超类），并在这些语言上训练GPT‑2模型（去除位置嵌入、插入分隔符、仅在分隔符处预测DFA状态）进行长度泛化评估。

**📈 对比分析**

比较方法：将模型在训练区间[ l_min , 50 ]之外的长度区间（直至 500）上的准确率与C‑RASP预测的一致性进行对比。结果显示：C‑RASP内部的语言几乎保持 100% 的准确率，而不在C‑RASP内的语言在超过训练长度后迅速失效，证明C‑RASP是准确的预测指标。

**⚠️ 局限性**

局限性包括：①实验规模相对有限，仅涵盖了人工合成正则语言；②决策算法虽然多项式，但在极大单子上仍可能面临计算开销；③必要条件等式不足以完全区分所有语言，仍需更细致的判别方法；④未讨论Transformer在非正则或更复杂语言上的泛化情况。

---

## 496. RAIL: An Automatic Classifier of the Artificial Intelligence Readiness Level

**arXiv ID:** 2608.13428 | [PDF](https://arxiv.org/pdf/2608.13428v1)

**作者:** Juan Irving Vasquez `[一作]` (Instituto Politécnico Nacional), Laura-Ivoone Garay-Jimenez `[通讯]` (Instituto Politécnico Nacional)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一的AI成熟度评估框架AIRL并设计基于LLM的多专家面板分类器RAIL，以从自然语言描述自动判定AI技术成熟度。

**💡 创新点**

创新点：①将NASA TRL、MLTRL及多维度AI准备度模型融为一体，构建九级环境阶梯加维度上限的可判定等级；②设计多专家面板架构，分别由LLM完成证据评估、六个维度评估，随后用确定性最小规则聚合，再由首席专家做最终裁定，保证准确性、保守性与可审计性。

**🔧 技术方法**

技术：大语言模型（LLM）作为证据专家和维度专家，确定性编程实现最小聚合，面板异步并行运行，首席专家使用规则约束做最终审核；同时利用引用追溯保证可追溯性。

**📊 数据集**

数据集：CIDETEC 10篇硕博论文（题目、摘要、实验摘要、结论），实验摘要由Gemini Pro压缩；无公开成熟度标注，仅用于内部实验。

**📈 对比分析**

比较方法：在三种协议（TRL、AI TRL、AIRL）下，分别用单模型（Monolithic）与面板（RAIL）进行分类。结果显示：AI TRL单模型显著膨胀成熟度；单模型AIRL压缩到中间水平；RAIL保持保守且能捕捉明确的维度缺口。计算成本约为单模型的6倍，适合批量标注。

**⚠️ 局限性**

局限性：样本仅10篇单机构论文，缺乏专家手工标注基准；使用单一LLM模型，未验证跨模型泛化；摘要过程可能引入偏差；计算量大，实时交互受限。

---

## 497. Sensorimotor Stickies: A Reconfigurable On-Body Platform for Closed-Loop Sensorimotor Training

**arXiv ID:** 2608.13412 | [PDF](https://arxiv.org/pdf/2608.13412v1)

**作者:** Tianhong Catherine Yu `[一作]` (Cornell University), Yiyue Luo `[通讯]` (Univirsity of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Sensorimotor Stickies，一个可重配置的闭环体感训练平台，支持在身体上粘贴可穿戴的惯性与触觉传感器以及振动反馈模块，并通过配套的移动应用实现任务、传感与反馈的配置、校准与实时使用。

**💡 创新点**

创新点在于：① 将传感与反馈模块抽象为可粘贴的“stickies”，实现硬件级的可重配置；② 通过共享的低功耗 BLE 基础设施与可复用固件，消除了对每个任务重新编写固件的需求；③ 采用身体中心配置模型，让非技术用户可用解剖学描述（如关节角度、触感位置）来定义任务、规则和反馈布局，极大降低使用门槛。

**🔧 技术方法**

使用的技术包括：小型 BLE 贴片式惯性测量单元（IMU）和可选触觉感应矩阵、Vibrotactile 电机驱动、低功耗固件、基于规则的阈值策略、身体中心配置模型、以及与之配套的移动端配置与校准界面。

**📊 数据集**

没有使用公开的数据集，而是通过实地评估：① 先前工作中的任务复现（平衡、姿态提醒、滑雪、CPR）作为基准配置；② 与 6 名运动训练从业者共同配置的真实场景；③ 10 名终端用户的首次使用与在任务中重新配置；④ 对 BLE 通信延迟、丢包与能耗的技术测评。

**📈 对比分析**

在技术评测中：BLE 平均通信延迟为 38.2 ms（手机贴身）/42.5 ms（5 m 远程），总端到端延迟低于人类对振动的典型反应时间；丢包率仅 0.0015%；在 70 mAh 电池下，连续全日使用仍可支持；在 21 天的连续使用中，14 个 Stickies 均未失效。终端用户评估中，平均 SUS（系统可用性量表）得分 78.75，显示高可用性；首次配置平均耗时 12.4 min，校准误差在 ±10° 范围内，满足训练需求。

**⚠️ 局限性**

局限性包括：① 目前仅支持阈值型规则，无法处理滑动窗口或学习型策略；② 校准过程仍需要人工指导，易产生误差；③ 贴片式接触可能在高汗或动态运动中失效；④ 目前的传感与反馈覆盖范围有限，无法满足所有复杂运动（如膝外翻需更复杂的关节角度模型）；⑤ 需要进一步扩展至更大规模的长期实地验证。

---

## 498. TraVEL: Trajectory-Guided Video Embedding Learning for Driving-Video Retrieval

**arXiv ID:** 2608.13495 | [PDF](https://arxiv.org/pdf/2608.13495v1)

**作者:** Yi-Chung Chen `[一作]` (Uber AV Labs), Burhan Yaman `[通讯]` (Uber AV Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于通用多模态嵌入模型，结合监督与轨迹引导的策略优化，实现了高效的驾驶视频检索。

**💡 创新点**

创新点在于将车辆行驶轨迹作为训练时的稀有奖励，通过GRPO使嵌入空间对运动更敏感，同时保持检索时单向量相似度的高效性。

**🔧 技术方法**

使用 Qwen3‑VL‑Embedding、InfoNCE 对比学习、轨迹归一化+DTW、Group Relative Policy Optimization（GRPO）等技术。

**📊 数据集**

利用 nuReasoning 数据集（2599 条 20 秒驾驶视频及其推理轨迹）。

**📈 对比分析**

与 CLIP4Clip、InternVideo2、Cosmos‑Embed1 等现成视频‑文本嵌入模型对比，TraVEL 在 2B 参数模型上 R@1 从 1.3% 提升至 8.7%，在 8B 上提升至 10.1%；纵向和横向 mAP 分别提升约 10% 与 5%。

**⚠️ 局限性**

对细粒度车道变换和微小横向偏移的检索仍存在困难，且模型依赖轨迹数据；在更大规模、多样化场景下的泛化还需进一步验证。

---

## 499. DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation

**arXiv ID:** 2608.13489 | [PDF](https://arxiv.org/pdf/2608.13489v1)

**作者:** DreamX Team `[一作]`, Pengfei Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DreamX-Phi 1.0，一种面向双手机器人操作的动作条件视频世界模型；

**💡 创新点**

创新点在于将SE(3)轨迹与图像空间运动提示结合，利用PRoPE注意力保持端执行器运动的刚体几何结构，并通过深度监督和对象关系监督强化物理一致性；

**🔧 技术方法**

使用Wan2.2‑TI2V‑5B视频扩散变换器、PRoPE注意力、机器人光流提示、辅助深度分支、V‑JEPA教师监督以及DMD后训练蒸馏；

**📊 数据集**

训练数据包括Ego4D、AgiBot、InternData‑A1、Cosmos3‑DROID、RoboCOIN和RoboTwin 2.0的真实与仿真机器人轨迹，以及自动作预训练的无动作视频；

**📈 对比分析**

在WorldArena 2.0评测中，Track 1获得EWMScore‑P 60.65、排名第一；Track 2的调整瓶子任务中，使用该模型训练的策略达成率 67.19%，排名第二；与多种公开模型比较，表现明显优于同类方法；

**⚠️ 局限性**

局限在于仅预测给定动作序列的未来视频，未生成动作；评估范围局限于WorldArena与RoboTwin，未验证对其他任务、实体和真实机器人的一般化；

---

## 500. Symmetry-Breaking De Novo Crystal Generation via Markovian Jump Diffusion

**arXiv ID:** 2608.13457 | [PDF](https://arxiv.org/pdf/2608.13457v1)

**作者:** Van Khoa Nguyen `[一作]` (HES-SO Geneva), Alexandros Kalousis `[通讯]` (HES-SO Geneva)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Symmetry‑breaking Crystal Diffusion (SbCD) 模型，能够从最小对称假设直接生成完整晶体结构规范。

**💡 创新点**

创新点在于将 Markovian 跳扩散与对称破缺相结合，既在连续状态空间（晶格）又在离散状态空间（位点对称）实现自适应的空间群约束，从而无需经验 Wyckoff 位点分布即可学习全空间群分布。

**🔧 技术方法**

采用扩散模型、空间群掩蔽机制、Markovian 跳扩散、变分上界、连续与离散状态的联合采样、简化的位点对称符号表示以及 CHGNet 进行结构松弛和能量评估。

**📊 数据集**

在 Materials Project 的 MP‑20（20 种化学组分）和更具挑战性的 MPTS‑52（最多 52 个原子）数据集上进行训练与评估。

**📈 对比分析**

与 CDVAE、DiffCSP、DiffCSP++、SGFM、SGEquiDiff、FlowMM 等基准进行对比，SbCD 在结构有效性、覆盖率、热力学稳定性和新颖度指标上均显著优于对称保持基线，并逼近或超过依赖经验空间群的模型。

**⚠️ 局限性**

局限性包括对位点对称表示的简化导致对更复杂对称的建模有限，未对原子分数坐标做空间群约束，扩散过程的物理过渡路径仍需进一步完善，且在大规模样本生成时计算成本仍相对较高。

---

## 501. LLM-Assisted Dynamic Threat Analysis for Attacker-Reachable Software Weaknesses in Autonomous Vehicles

**arXiv ID:** 2608.13450 | [PDF](https://arxiv.org/pdf/2608.13450v1)

**作者:** Md Wasiul Haque `[一作]`, Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型（LLM）是否能自动化对Autoware开源自动驾驶堆栈的动态安全缺陷确认，提出从静态候选点到LLM生成、构建、修复、模糊测试的全流程管线。

**💡 创新点**

首个端到端可复现的LLM辅助动态分析实验，系统识别并分类安全相关缺陷，揭示构建利用LLM在大规模ROS2系统中集成和执行的主要瓶颈。

**🔧 技术方法**

利用LLM（代码专用与通用推理模型）生成测试 harness，使用Clang编译、AddressSanitizer/UBSan，构建修复循环，结合libFuzzer进行模糊测试。

**📊 数据集**

使用Autoware开源代码仓库（约800+包、1.7万+源文件）及其ROS2接口与配置文件进行静态和动态分析。

**📈 对比分析**

对比两模型与无上下文消融以及基线，统计编译成功率、链接成功率、模糊测试通过率；结果显示LLM未能在所给预算内确认任何缺陷，主要问题在于构建集成失败。

**⚠️ 局限性**

限制包括：仅在软件模拟环境测试；静态分析为启发式，未覆盖所有包；LLM生成的 harness 依赖手工调试；构建修复未考虑链接和目标执行的语义约束。

---

## 502. Edit2TikZ: A Comprehensive and Challenging Benchmark for Scientific Figure Editing with TikZ

**arXiv ID:** 2608.13441 | [PDF](https://arxiv.org/pdf/2608.13441v1)

**作者:** Zongyun Zhang `[一作]` (Shanghai Jiao Tong University), Yuzhuo Fu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Edit2TikZ 基准，用于评估从渲染图像到可编译 TikZ 代码的端到端科学图表编辑，并构建了人类对齐的 RS 与 ECS 评估框架；同时推出了用于训练的 TikZEditMix 数据集和两阶段课程学习策略；在 14 个主流 MLLM 上进行实验并对比其性能；

**💡 创新点**

创新点包括：①将图表编辑细化为 8 种原子操作并结合文本与可视化定位提示；②设计了 RS 与 ECS 两项人类对齐评估指标；③构建混合训练集 TikZEditMix 并采用两阶段课程学习显著提升小模型性能；④提供多样化真实与合成样本，填补现有图表编辑基准的空白。

**🔧 技术方法**

使用多模态大型语言模型进行图像到代码的生成、代码编辑与复合任务；应用两阶段 curriculum learning、迭代反馈（编译器+渲染器）以及人类评判的自动化评分器；在训练中引入了 token‑level 交叉熵损失并采用 LaTeX 编译与渲染评估。

**📊 数据集**

主要数据集为 Edit2TikZ（1,548 份多样化编辑样本，包含真实、文本合成与视觉定位三类），TikZEditMix（32,448 份训练样本，包含 20,244 份图像‑代码重建对和 12,204 份编辑任务），以及 DaTikZ‑v3 与 DaTikZ‑v2 的公开重建数据。

**📈 对比分析**

通过编译成功率、cBLEU、TED、DSim、RS 与 ECS 等指标对 14 个模型（8 专有、6 开源）进行对比；实验表明专有模型平均编译成功率 75%，RS 与 ECS 均在 58–60 之间；开源模型平均编译成功率仅 40%，RS 与 ECS 低于 25。采用两阶段课程学习后，Qwen3.5‑4B 的编译成功率从 45% 提升至 83%，RS 与 ECS 均显著提升，说明训练策略有效。

**⚠️ 局限性**

局限性主要在于：①图像到 TikZ 的重建仍不够精确，导致编辑错误；②编辑复杂度与程序长度越大，模型性能急剧下降；③小模型因容量限制易产生循环生成与编译失败；④RS 与 ECS 的人类对齐评估虽然有益，但在极端情况仍可能与真实视觉质量不完全一致。

---

## 503. Motor, Cognitive, or Corpus? What Survives Cross-Lingual Transfer in Speech-Based Parkinsons Disease Detection

**arXiv ID:** 2608.13425 | [PDF](https://arxiv.org/pdf/2608.13425v1)

**作者:** Serli Kopar `[一作]` (University of Tübingen), Kerstin Ritter `[通讯]` (University of Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

对跨语言、跨任务的基于语音的帕金森病（PD）检测进行系统评估，利用冻结的自监督学习（SSL）语音表示和低容量逻辑回归探针，构建五个逐步递增分布偏移的评估场景。

**💡 创新点**

首次将跨语言、跨任务转移与疾病特异性（与痴呆对比）结合评估，揭示SSL表示在转移中主要保留语料库相关信息而非PD特有信号；并提出层选择高度依赖语料库的观点。

**🔧 技术方法**

使用多种SSL背骨（WavLM、HuBERT、W2V2、XLS-R、MMS）和传统的eGeMAPS特征，提取各层表示并通过逻辑回归探针评估线性可解信息；设计REF、S1–S5五个分布偏移场景。

**📊 数据集**

三种跨语言PD语料库（西班牙语ES、德语DE、捷克语CZ）以及独立的德国TREND队列（包含PD与痴呆受试者）作为评估数据集。

**📈 对比分析**

通过在REF（同库交叉验证）选取最佳层后固定，并在S1–S5中保持该层不变，对比不同场景下的平衡准确率（BA）和预测概率分布。结果显示：S1变动最小（-1.9±4.5 BA），S2（录音条件）-12.5±14.3 BA，S3（语言）-16.3±10.6 BA；跨任务转移至TREND时，PD与健康对照可分离（BA≈0.65–0.67），但与痴呆不可区分，表明缺乏疾病特异性。

**⚠️ 局限性**

局限包括样本量相对较小、仅评估二分类PD与健康、未覆盖所有可能的疾病对照，且层选择受语料库影响大，说明SSL表示对任务/语言的泛化能力有限。

---

## 504. Synthetic Persona Pretraining: Alignment from Token Zero

**arXiv ID:** 2608.13482 | [PDF](https://arxiv.org/pdf/2608.13482v1)

**作者:** Julian Minder `[一作]` (École Polytechnique Fédérale De Lausanne), Robert West `[通讯]` (École Polytechnique Fédérale De Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在预训练阶段向约10%的文档中插入基于规范宪法的第一人称价值反思，构建合成人格，并在后训练阶段将该人格绑定到助手身份，从而实现从 token 零开始的价值对齐。

**💡 创新点**

创新点包括：①将目标人格直接植入预训练而非后期修正；②利用合成反思作为标注，首次在预训练中注入价值导向；③系统证明 token‑zero 干预比 mid‑training 更有效，并随预训练规模放大；④揭示 persona binding 的必要性与脆弱性。

**🔧 技术方法**

技术上采用标准交叉熵预训练与监督微调，结合基于宪法的生成式反思注释，使用 SafeLM 分类挑选有害文档，构建后训练对话集，并使用 ConstitutionEval、AI Values & Risk、各类 jailbreak 评测及能力基准进行评估。

**📊 数据集**

数据集包括 500B token 的 Dolma 3 子样本，SafeLM 选出的有害文档及随机等量的安全文档，约 10% 文档被标注为反思；后训练数据来自 WildChat、WildJailbreak、WildGuardMix 等，覆盖一般指令与安全指令。

**📈 对比分析**

通过与标准预训练、仅过滤有害文档、mid‑training SPP、仅 mid‑training 注入等多种对照实验，token‑zero SPP 在宪法遵循、AI 价值与风险判断以及 jailbreak 抗压方面均优于基线，且优势随预训练规模扩展显著；mid‑training 亦提升 jailbreak 但在价值对齐上逊色。

**⚠️ 局限性**

局限性包括：仅在 1.7B–3B 规模验证，未验证在前沿模型上的效果；persona binding 对持续训练或抑制性干预极易被破坏；合成反思的生成与人工审核成本高；未探讨在 RL 等后训练方法中的长期持久性。

---

## 505. Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development

**arXiv ID:** 2608.13417 | [PDF](https://arxiv.org/pdf/2608.13417v1)

**作者:** Yiwei Li `[一作]` (Meituan), Jingang Wang `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估七款前沿语言模型在36个长周期自动研究任务上的表现，包括过程能力、经验再利用与工具套件影响。

**💡 创新点**

提出基于规则的三维过程指标（Solution Framing、Execution、Feedback Control）以及经验再利用的因果对照实验，揭示当前模型更多是工程优化器而非真正的自主研究者。

**🔧 技术方法**

采用规则指标、对照实验、LLM 判定的创新度评估、以及自动化 Harness 进化等技术。

**📊 数据集**

使用 AutoLab 36 任务（Model Development、System Optimization、Puzzle & Challenge、CUDA）以及各模型的官方验证器。

**📈 对比分析**

通过 avg@3/best@3、C1/C2/C3 分数以及经验增益对比，发现 Opus‑4.7 在平均表现上领先，平均得分约 0.74，最佳得分 0.79，经验增益平均 +0.05，Harness 可提升稳定性。

**⚠️ 局限性**

局限在于过程指标仅覆盖可观测信号，经验实验受控制点限制，结果受任务、预算和 Harness 设定影响，且创新评估依赖人工审核。

---

## 506. Deliberate Practice: Learning Robot Skills under a Budget

**arXiv ID:** 2608.13415 | [PDF](https://arxiv.org/pdf/2608.13415v1)

**作者:** Shivam Vats `[一作]` (Brown University), George Konidaris `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Deliberate Practice 算法，利用有限实践预算主动学习机器人技能以最大化长周期规划奖励。

**💡 创新点**

创新点在于将双层优化问题转化为单层双线性程序，可用现成求解器获得全局最优分配。

**🔧 技术方法**

使用双线性程序、LP 对偶、McCormick 包络、空间分支定界求解以及线性/分段/指数模型预测技能熟练度。

**📊 数据集**

在 MuJoCo 仿真任务（Cleanup、Cleanup-Multi、Breakfast）和真实 Franka Panda 的 Breakfast 任务上验证。

**📈 对比分析**

与贪婪式主动学习基线（EES、CI、LCF、R）比较，DP 在中高预算下显著优于基线，能更好地利用预算获得更高奖励。

**⚠️ 局限性**

局限：需要先验熟练度估计，若过于乐观可能误分配预算；对极大规模问题求解仍可能出现计算瓶颈。

---

## 507. Intern-S2-Preview: Scientific Agentic Foundation Model

**arXiv ID:** 2608.13505 | [PDF](https://arxiv.org/pdf/2608.13505v1)

**作者:** Lei Bai `[一作]` (Shanghai AI Laboratory), Yicheng Zou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了Intern‑S2‑Preview‑397B，一种面向科学研究的多模态、长序列、时间序列及代理式学习的基础模型，支持文本、图像、表格、公式、遥感、显微镜等多种科学数据；

**💡 创新点**

创新点包括：①在Transformer基础上加入长序列时间序列编码器与预测分支，实现统一的时间序列理解与数值预测；②设计可插拔的Memory Decoder，实现不改动主干的领域专精；③构建黑白盒统一的代理RL框架和可复用的rollout/trace存储；④提出分阶段预训练、稀疏回合、适应性长度正则、显式代理校正与GEPO等多项训练技术；

**🔧 技术方法**

使用技术包括：多模态Transformer + Q‑Former、MoE、视觉Encoder、Token‑Level Router、检索蒸馏、分层强化学习、partial‑rollout+off‑policy correction、speculative decoding、Group‑Entropy Policy Optimization、on‑policy distillation等；

**📊 数据集**

训练与评测数据集涵盖：科学PDF+图像交织文本、图像检索库、时间序列公开基准（SciTS）、生物学/化学/物理/地球科学等领域的Benchmarks（Biology‑Instructions、Mol‑Instructions、MolecularIQ、SciReasoner、TOMG‑Bench、MP20、ProteinBinder‑9、XLRS‑Bench、MicroVQA、SFE、ObsCrisis‑Bench、SciCode、SGI‑Bench、ResearchClawBench），以及通用评测（MMLU‑Pro、SimpleQA‑Verified、AdvancedIF、HMMT‑2026、MMMU‑Pro、ChartQAPro、SkillsBench、Terminal‑Bench‑2.1、SWE‑Bench‑Pro/Multilingual、WildClawBench）；

**📈 对比分析**

与现有模型对比，Intern‑S2‑Preview‑397B在大多数科学与多模态基准上实现SOTA，特别是 Biology‑Instructions、Mol‑Instructions、SciReasoner、MP20、ProteinBinder‑9、XLRS‑Bench、MicroVQA；在通用基准上也优于开源同行（MMLU‑Pro、SimpleQA‑Verified、MMMU‑Pro、ChartQAPro），并在代理式任务上实现第二名；Memory Decoder在生物学任务上平均提升约3.4个百分点；时间序列模块在SciTS基准上超过同类模型；

**⚠️ 局限性**

局限性包括：仍为预览版，长周期科学工作流程的可靠性尚未充分验证；域专精依赖外部Memory Decoder，需不断扩充和维护；Verifier与奖励设计可能仍易受信息泄漏或作弊影响；整体模型规模庞大，部署与算力成本高。

---

## 508. MapRoute++: Surrogate-Guided Semantic Routing for Visual Concept Unlearning

**arXiv ID:** 2608.13478 | [PDF](https://arxiv.org/pdf/2608.13478v1)

**作者:** Ashok Urlana `[一作]` (IIIT Hyderabad), Ponnurangam Kumaraguru `[通讯]` (IIIT Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

介绍了一种基于轻量级概念映射器和语义路由的视觉概念消除框架MapRoute++，能够在不修改扩散模型核心权重的前提下实现目标概念的精准抹除。

**💡 创新点**

创新点包括：① 两阶段训练策略（先做身份映射再做目标-替代映射）以最大限度保留非目标语义；② 采用多同义词/短语增强目标概念表达；③ 设计语义路由机制在推理时动态选取最相关的映射器；④ 对替代概念进行策略性选择，避免泄露目标信息。

**🔧 技术方法**

使用技术包括残差多层感知机映射器、冻结的文本编码器与U-Net交叉注意力、EOT token embedding重定向、LLaVA模型进行自动评估、ERR（Erasing–Retention–Robustness）指标。

**📊 数据集**

数据集为Genμ 2.0 Challenge 提供的20个概念（对象、动物、风格、场景、动作）及其直接、间接、对抗、相邻与保留提示。

**📈 对比分析**

与ESD、CA、FMN、FADE、原MapRoute等基线在ERR指标上进行比较，MapRoute++在整体平均ERR上达到0.721，比最强基线提升约12.1%（约0.12），在对象和场景类别表现尤为突出；风格类别仍相对落后。

**⚠️ 局限性**

局限性主要体现在：对艺术风格概念的抹除效果不佳，易导致相邻概念损伤；替代概念选择仍需更系统化，以进一步减少对非目标概念的影响。

---

## 509. MLLM-Routed Heterogeneous Ensembles for Robust Cross-Dataset Image Classification

**arXiv ID:** 2608.13463 | [PDF](https://arxiv.org/pdf/2608.13463v1)

**作者:** Daniel Perkins `[一作]` (University of Tennessee), Linda Ungerboeck `[通讯]` (University of Tennessee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于多模态大型语言模型的自适应路由框架 ARMDIL，用来在不同视觉域之间动态分配图像给最合适的专家网络进行分类。

**💡 创新点**

创新点在于用零拷贝 LLM 做域判断和路由，避免了传统训练路由器的冗余计算和不可解释性，并通过自然语言推理实现可解释的决策。

**🔧 技术方法**

技术包括多模态大型语言模型（Gemma-4-12B）进行链式思维推理、CNN（ResNet）、自监督视觉 Transformer（DINOv2/3）、视觉语言模型（CLIP）作为专家，以及统一标签空间的联合训练。

**📊 数据集**

使用了四个公开数据集：CIFAR10（通用）、FER2013（面部表情）、EuroSAT（卫星图像）和 OrganAMNIST（医学扫描），并将它们映射到统一的38类标签。

**📈 对比分析**

与基线的比较表明，ARMDIL 在统一测试集上达到 90.78% 的准确率，略优于最佳单一模型 DINOv3（89.61%）以及多数投票和神经网络路由器，表现出显著的跨域泛化优势。

**⚠️ 局限性**

局限性包括对 EuroSAT 的识别不够自信导致 UN-sure 路由、对小型 LLM 的推理能力限制、以及在新增域时仍需手工调整 prompt 等。

---

## 510. Doubly Robust Estimation of Causal Effect on CVR with Targeted Regularization

**arXiv ID:** 2608.13461 | [PDF](https://arxiv.org/pdf/2608.13461v1)

**作者:** Jiayi Dan `[一作]` (Tsinghua University), Yong Wang `[通讯]` (Tencent Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对链式结构的因果效应（尤其是后点击转化率 CVR）提出一种新的估计框架，先从半参数理论出发构造双重鲁棒估计器，然后加入目标正则化实现数值稳定并保持根号 n 一致性。

**💡 创新点**

创新点包括：
- 对 CVR 目标的影响函数与 von Mises 展开进行推导，形成专门的双重鲁棒估计器；
- 结合目标正则化（TMLE 启发）改进标准 DR 估计的数值不稳定问题；
- 在不必进行交叉拟合的前提下，利用神经网络实现高效端到端学习并保证根号 n 收敛；
- 通过多任务学习同时估计 CTR 与 CVR 的因果效应。

**🔧 技术方法**

使用技术：半参数理论、影响函数、双重鲁棒推断、目标正则化、可变系数神经网络、Spline 逼近、交叉拟合（仅用于理论证明）、神经网络优化与正则化。

**📊 数据集**

数据集：
- 合成数据（3k 训练 / 1k 测试）
- 半合成数据（基于 News 数据的 498 维特征）
- 真实世界数据 CRITEO‑UPLIFTv2（约 13M 样本，12 个特征，二元处理）。

**📈 对比分析**

与 DragonNet、VCNet、DRNet、TARNet、Causal Forest、ECUP、基础 DR 估计器以及无目标正则化版本进行对比。评估指标为 AMSE（合成/半合成）和 AUUC/QINI（真实数据）。实验表明：
- 在 CVR 任务上，所提方法在所有基线中均取得最低 AMSE，性能提升幅度显著；
- 加入目标正则化后效果最优，去掉后性能显著下降；
- 在真实数据上，AUUC/QINI 指标同样优于大多数基线，验证了方法的实用性。

**⚠️ 局限性**

局限性：
- 对真实数据的评估只能使用间接指标（AUUC、QINI），无法得到真实的估计误差，可能低估方法优势；
- 方法对样本量和特征维度有一定依赖，过小样本或高度稀疏点击会影响估计质量；
- 对二元处理的实现需要额外的适配，且在某些应用场景下可能需要进一步改进网络架构。

---

## 511. SNM-VFI: Symmetric Nonlinear Motion-Guided Generative Video Frame Interpolation

**arXiv ID:** 2608.13460 | [PDF](https://arxiv.org/pdf/2608.13460v1)

**作者:** Jisoo Jeong `[一作]` (Qualcomm AI Research), Fatih Porikli `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练的对称非线性运动引导生成式视频帧插值框架 SNM‑VFI，通过预训练光流与视频扩散模型实现高质量、时间一致的帧插值。

**💡 创新点**

将光流生成的多帧非线性运动先验与扩散模型的隐层动态融合，并使用对称非线性运动模型和置信度加权融合，解决光流假设限制和扩散模型时序不连贯问题。

**🔧 技术方法**

对称非线性光流插值、置信度估计、基于 VAE 的流引导潜变量初始化、特征层流引导加权融合、置信度加权融合与扩散输出等技术。

**📊 数据集**

DAVIS、Sintel、KITTI 三个常用 VFI 基准数据集。

**📈 对比分析**

与后向/前向光流方法、LDMVFI、GenIn 等方法在 PSNR/SSIM/LPIPS/FID 等指标上对比，SNM‑VFI 在 LPIPS/FID 上显著领先，同时保持 PSNR/SSIM 竞争力，并在视觉质量与时间一致性上优于其他方法。

**⚠️ 局限性**

对长时间间隔或极端非线性运动的光流预测仍受限，且依赖预训练模型，对特殊场景的鲁棒性仍有待提升。

---

## 512. UniTexture: Cross-Task Universal Adversarial Textures for Vision-Language-Action Models

**arXiv ID:** 2608.13453 | [PDF](https://arxiv.org/pdf/2608.13453v1)

**作者:** Yukun Dai `[一作]`, Lei Zhu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了UniTexture，一种跨任务的通用对抗纹理攻击，可通过单一3D纹理对Vision‑Language‑Action模型实施多任务干扰

**💡 创新点**

创新点在于：1）将纹理共享于所有任务，避免逐任务优化；2）使用模型原生动作空间目标实现定向攻击；3）通过可微渲染将梯度直接反向到纹理；4）展示跨任务、跨模型、跨套件的迁移性

**🔧 技术方法**

主要技术包括：可微渲染（PyTorch3D + Phong），任务分布采样与联合优化，针对不同动作接口的定向损失（自回归token、流匹配块），以及渲染参数的离线校准

**📊 数据集**

实验数据集为LIBERO‑Spatial与LIBERO‑Goal，攻击对象为模型的plate和bowl，受试模型为OpenVLA和π₀.₅

**📈 对比分析**

与无纹理、原始纹理、Gaussian噪声对比，UniTexture将任务成功率从90%降至48.4%，平均TDS与pDHR提升明显，且在跨套件/跨模型设置中仍保持较高攻击效果

**⚠️ 局限性**

局限性包括：对不同模型对纹理敏感度差异导致的攻击效果不对称；需要精确的渲染校准；对特定动作方向（如下压）效果不佳；未针对动态光照或更复杂场景进行验证

---

## 513. Compact Path Representation in DAGs via Colored Edge Pebbling

**arXiv ID:** 2608.13480 | [PDF](https://arxiv.org/pdf/2608.13480v1)

**作者:** Paola Bonizzoni `[一作]` (University of Milano-Bicocca), Brian Riccardi `[通讯]` (University of Milano-Bicocca)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于图拓扑的变异图紧凑表示方法，利用彩色石子铺设边来唯一重建预定义路径，并给出了相应的查询数据结构。

**💡 创新点**

创新点在于引入守护者（guardian）与石子铺设（pebbling）概念，将变异图的路径重建问题转化为在DAG上寻找最小/饱和石子铺设，从而在不依赖序列信息的情况下实现空间紧凑且支持高效查询；并证明最小石子铺设可多项式求解，而饱和石子铺设为NP‑hard，可通过最小权集覆盖求解。

**🔧 技术方法**

使用了动态规划计数路径、集合覆盖与整数线性规划、秩/选择（rank/select）和位向量的紧凑数据结构，以及对DAG的拓扑排序。

**📊 数据集**

主要在理论与合成实验层面验证，未给出具体公开基因组数据集；若有实验，使用的是合成或真实的pan‑genome DAGs，但未详细说明。

**📈 对比分析**

与传统基于序列的GBWT等索引相比，本文方法在存储空间上依赖于石子铺设大小；查询时间可达到 O(|E|)；在最小石子铺设下，边查询进一步优化为 O(|E|+|H|)。实验结果表明，在合成实例上实现了相对良好的压缩率和查询速度。

**⚠️ 局限性**

局限性包括：仅适用于无环有向图；饱和石子铺设仍为NP‑hard，难以在大规模实例上高效求解；未处理带有环或双向结构的变异图；对子路径查询的支持有限。

---

## 514. AaLLM: An End-to-End Analog Circuit Design Framework from Topology Generation to Sizing Using Large Language Models

**arXiv ID:** 2608.13472 | [PDF](https://arxiv.org/pdf/2608.13472v1)

**作者:** Mohammed Ayman Habib `[一作]` (University of Utah), Morteza Fayazi `[通讯]` (University of Utah)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AaLLM，一个完整的端到端类比电路设计框架，能够从用户规格自动生成拓扑、进行器件尺寸优化，并输出可运行的 SPICE netlist。

**💡 创新点**

创新点包括：①利用检索增强生成（RAG）让 LLM 结合文献知识进行设计决策；②使用 fine‑tuned FLAN‑T5 做条件序列到序列（Seq2Seq）拓扑生成，并能产生训练语料中未出现过的全新拓扑；③构建三代理（Designer、Critic、Evaluator）闭环优化系统，采用课程式（curriculum）优化顺序，显著减少模拟次数和迭代次数。

**🔧 技术方法**

技术手段包括：检索增强生成（RAG）、Fine‑tuned FLAN‑T5、Claude 4.5/4.6 作为模型、三代理 LLM 循环、SPICE 交互、基于 bipartite component‑node 矩阵的拓扑表示、可调节的语义/关键词检索融合。

**📊 数据集**

数据集：收集并自动构建的类比电路文献和教科书知识库（如 Razavi 书籍）、SkyWater 130 nm PDK 的实验数据、以及自定义的 24 个目标规格的运算放大器基准与主动滤波器基准。

**📈 对比分析**

与 Atelier、AmpAgent、LaMagic、AnalogCoder、AnaFlow 等最新 LLM 设计框架对比，AaLLM 在 24 个运算放大器任务中 SPICE 调用次数下降 3–4.5 倍、墙钟时间下降 40 倍；在主动滤波器任务中同样实现全部规格，且相较于 Atelier 的 CMA‑ES 优化器，AaLLM 的模拟次数减少 2.79 倍、成功率提升至 100%。

**⚠️ 局限性**

局限性包括：仍需大量模拟反馈来保证可靠性；RAG 依赖可检索的知识库，知识库覆盖不足时可能导致误判；三代理系统复杂度高，调参和训练成本较大；在极端规格或全新电路结构上，模型可能需要进一步 fine‑tune。

---

## 515. Active-Trace Complexity Bounds for Moreau--Yosida Unadjusted Langevin Sampling

**arXiv ID:** 2608.13467 | [PDF](https://arxiv.org/pdf/2608.13467v1)

**作者:** Yuchen Xin `[一作]` (Peking University), Zhihua Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了在强凸、复合目标（光滑强凸项+凸但可能非光滑正则化）下，Moreau–Yosida 未调整 Langevin 算法（MYULA）的误差收敛性质，提出了基于“参考热路径主动轨迹”的分析框架；

**💡 创新点**

创新点在于将传统的全局 Moreau 曲率上界（$d/λ$）替换为沿参考热路径加权的主动曲率平均，显著降低了在结构化正则化（如 Lasso、组 Lasso、总变差等）中的迭代复杂度，使得最终误差达到 $O(ε^{-2})$ 而非 $O(ε^{-3})$；

**🔧 技术方法**

主要技术包括：弱/几乎处处 Hessian 定义、Heat 能量等价式、Wasserstein‑EVI 递推、主动曲率与活跃集合几何的曲率–管道接口、切片密度与传播估计，最终得到主动曲率上界；

**📊 数据集**

本文为理论分析，未使用具体数据集；

**📈 对比分析**

与已有的全局光滑度分析（$O(ε^{-3})$）相比，本文在结构化正则化情况下提供了更优的 $O(ε^{-2})$ 迭代复杂度，并通过多种正则化示例（单维分段线性、可分离分段线性、组 Lasso、总变差）验证了理论预测；

**⚠️ 局限性**

局限性包括：仅适用于强凸、可分解的复合目标；需要 Moreau 平滑参数固定且已知；对非凸或更一般的非光滑目标未给出结论；主动曲率估计需手工推导，可能不易推广到更复杂结构。

---

## 516. Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts

**arXiv ID:** 2608.13458 | [PDF](https://arxiv.org/pdf/2608.13458v1)

**作者:** Imtiaz Ul Hassan `[一作]` (Edge Hill University), Ardhendu Behera `[通讯]` (Edge Hill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FineX 模型，通过 RGB、姿态热图与骨架图三路特征融合实现细粒度动作识别。

**💡 创新点**

创新点：将姿态分为热图和骨架图两种表现形式，采用对称的双向交叉注意力保持流身份，随后使用流向的稀疏 Mixture‑of‑Experts 进行内容自适应细化。

**🔧 技术方法**

技术：R(2+1)D/SlowOnly‑R50、PoseC3D、ST‑GCN++ 基础网络，双向交叉注意力，稀疏 MoE 与负载平衡正则。

**📊 数据集**

数据集：Gym99、Gym288、Diving48。

**📈 对比分析**

与 SOTA 比较：在 Gym288 上 Top‑1 94.3%、MCA 76.2%（比前置 +7.6 点），Gym99 Top‑1 97.1%、MCA 96.4%，Diving48 Top‑1 92.9%，均超越现有多模态与视觉‑语言方法。

**⚠️ 局限性**

局限：需要三条前向传递，计算成本较高；对姿态关键点噪声较敏感；未利用语言先验。

---

## 517. A Unifying Perspective on Causal World Models: From Observations to Representations to Structure

**arXiv ID:** 2608.13456 | [PDF](https://arxiv.org/pdf/2608.13456v1)

**作者:** Avinash Kori `[一作]` (Imperial College London), Fabrizio Russo `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了“因果世界模型（Causal World Model, CWM）”的概念，并给出形式化定义与结构化抽象层级；

**💡 创新点**

创新点在于将世界模型拆分为观测、表示、因果结构和决策四个可独立识别的组件，阐明每个组件的可辨识性和允许的等价变换；

**🔧 技术方法**

运用了因果表示学习、因果发现、结构因果模型与基于模型的决策（MDP）等技术框架，形成了完整的因果推理与决策链条；

**📊 数据集**

论文为理论性工作，没有使用具体数据集；

**📈 对比分析**

由于没有实验和对比方法，未给出性能指标；

**⚠️ 局限性**

局限在于仍处于理论阶段，缺乏实证验证，且对部分观测不完整或存在未观测混杂的情况尚未处理。

---

## 518. Evaluation of Clinically Steerable Retinal Image Generation from Foundation Model Latent Spaces

**arXiv ID:** 2608.13455 | [PDF](https://arxiv.org/pdf/2608.13455v1)

**作者:** Zuzanna A. Wakefield-Skórniewska `[一作]` (University of Oxford), Bartłomiej W. Papież `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

评估了四种视网膜基础模型（RETFound、PRETI、FLAIR、URFound）在可控眼底图像生成中的能力，并研究其在合成图像中保留临床表型信息的效果。

**💡 创新点**

创新点在于：①将基础模型潜在空间用于Representation Tokenizer，构建可控生成框架；②发现基础模型潜在空间在内部评估中比传统潜在扩散更能保留表型信息；③揭示合成‑真实表型表示差距，指出外部可迁移性不足。

**🔧 技术方法**

技术手段包括：RepTok两阶段生成框架（基于DiT‑B解码器与流匹配训练），1D MLP‑Mixer表型生成器，使用Vision‑Only和Vision‑Language基础模型的CLS‑token微调；与传统潜在扩散模型（同VAE + DiT‑B）做对比。

**📊 数据集**

使用UK Biobank 256×256彩色眼底图像：83,141张用于第一阶段训练，50,773张用于表型条件训练，11,780个独立受试者用于测试。

**📈 对比分析**

比较方法：生成质量评估（rFID、gFID、PSNR、SSIM、LPIPS、KID），内部预测评估（AUROC、ACC）基于基础模型表征；外部评估使用ResNet32分类器预测年龄、性别、血压。结果显示：内部评估中RepTok优于传统扩散，表型信息保留更好；但在外部评估中性能显著下降，尤其是年龄预测，表明存在合成‑真实表示差距。

**⚠️ 局限性**

局限性：①合成图像与真实图像的表型表示不一致，导致外部可迁移性差；②年龄预测性能大幅下降，可能因重建过程中丢失细节或引入伪影；③仅在UK Biobank数据上验证，缺乏多中心、多模态外部验证；④基础模型特定潜在空间的依赖限制了通用性。

---

## 519. Reduced Matrix Multiplication: Input-Adaptive Matrix-Product Reduction for LLM Inference

**arXiv ID:** 2608.13426 | [PDF](https://arxiv.org/pdf/2608.13426v1)

**作者:** Zixuan Lan `[一作]` (University of Chicago), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练无关、输入自适应的 Reduced Matrix Multiplication (RMM) 方法，用于在 Transformer 推理中通过选择重要维度来降低矩阵乘法计算量。

**💡 创新点**

创新点在于将矩阵乘法的维度缩减与当前激活值相关联，采用 Deterministic Top-K 选择，实现可控的准确-效率权衡且不需改动模型权重。

**🔧 技术方法**

采用激活感知的 Top‑K 维度选择、矩阵分块、Triton 自定义内核，以及对注意力与 MLP 的分层实验。

**📊 数据集**

在 1B–70B 参数的 Llama、Qwen、Qwen‑VL 等大模型上，使用通用 QA、MMLU、GSM8K、Long‑context、Vision‑Language（POPE, Blink）等数据集进行评测。

**📈 对比分析**

与 SparseGPT、Wanda、SliceGPT、H2O、随机剪枝等基线对比，RMM 在保持 80%–90% 保留率时仅损失少于 1–3% 的准确率，同时在长序列上可实现 1.3–1.4 倍的速度提升。

**⚠️ 局限性**

局限在于仅提供训练无关的自适应裁剪，未探索对不同模态组件的更细粒度分析，也未考虑量化或更高层次的模型压缩。

---

## 520. Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning

**arXiv ID:** 2608.13422 | [PDF](https://arxiv.org/pdf/2608.13422v1)

**作者:** Zheyu Zhuang `[一作]` (KTH Royal Institute of Technology), Danica Kragic `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Seeker模块，利用动作监督直接学习进度感知的视觉ROI，形成无标签的视觉瓶颈。

**💡 创新点**

创新点在于用动作监督代替手工关键帧或语义标签来学习ROI，兼容多任务、多模态且无需额外标注。

**🔧 技术方法**

采用冻结的DINOv3特征、跨模态注意力读取、迭代读出、FiLM调制、扩散动作预测等技术。

**📊 数据集**

使用MimicGen仿真任务和真实世界多视角RGB数据集（咖啡传输、桌面清理、板块装配等）。

**📈 对比分析**

与无裁剪、镜像增强、RVT2-Crop、Oracle ROI、RAVEN等基线对比，在仿真中平均成功率从42.6%提升至62.6%，在真实世界中从48.3%提升至76.7%，并在光照/背景变换下平均成功率提升至60%。

**⚠️ 局限性**

局限在于需要足够多样化的演示覆盖不同任务阶段，训练和推理存在额外开销，在空间不变或视觉冗余阶段效果有限。

---

## 521. Enhancing Virtual Agents through SLMs and Edge-Computing: An Exploratory Evaluation of Think and Memory Processes

**arXiv ID:** 2608.13420 | [PDF](https://arxiv.org/pdf/2608.13420v1)

**作者:** Aimilios Hadjiliasi `[一作]` (University of Central Lancashire), Louis Nisiotis `[通讯]` (University of Central Lancashire)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估在边缘计算环境下使用小型语言模型（SLM）实现CEAA的Think和Memory组件，构建了服务路由与结构化记忆的端到端系统，并在Unity虚拟世界中测试其性能。

**💡 创新点**

在CEAA框架内首次将SLM用于边缘端认知调度与记忆处理，展示了小模型在实时、资源受限环境下实现智能代理认知与记忆的可行性，并探讨了模型规模与延迟之间的权衡。

**🔧 技术方法**

采用Qwen2.5系列SLM（0.5B/1.5B/3.0B）、LangMem+SQLite记忆子系统、HTTP网关实现服务路由，基于NVIDIA Jetson Orin NX 8GB边缘硬件与Unity客户端协同工作。

**📊 数据集**

使用LLM辅助生成、人工审核的1,000条路由提示和250条写‑读记忆提示组成的自定义平衡数据集。

**📈 对比分析**

对三种模型分别在路由与记忆两项指标下进行宏平均精度/召回/F1、准确率与平均/中位/95%延迟测评；1.5B模型在路由方面约85%准确率，3.0B在记忆回溯达到约94%准确率，但延迟显著提高（3.0B平均≈9.7 s）。

**⚠️ 局限性**

评估仅在受控测试环境完成，未涵盖完整的体化用户交互；仅测试CEAA的Think与Memory两层，忽略感知、动作映射等；使用单一SLM系列与单一硬件平台，泛化性受限；延迟测量未细化各步骤，且未进行用户体验验证。

---

## 522. StreamTTT: Reconciling Real-Time Perception and Long-Term Memory in Streaming VLMs

**arXiv ID:** 2608.13416 | [PDF](https://arxiv.org/pdf/2608.13416v1)

**作者:** Joya Chen `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的流式视觉语言模型，采用短滑动KV缓存与并行的fast‑weight记忆分离，专门用于实时感知与长时记忆；

**💡 创新点**

创新点在于将长时历史存储在超出注意力上下文的fast‑weight（TTT）分支中，避免历史信息占用短期窗口，实现在保持实时感知的同时提升长时回溯；

**🔧 技术方法**

采用的技术包括Test‑Time Training（TTT）实现fast‑weight记忆，滑动窗口注意力，学习门控融合两条记忆通道，以及大块TTT（LaCT）来提升更新效率；

**📊 数据集**

使用的数据集包括119K离线长视频问答（LLaVA‑Video‑178K子集）和112.4K从Streamo等数据重构的实时问答，构成平衡的训练集；

**📈 对比分析**

通过与SimpleStream-4B、SimpleStream-8B等基线在OVO‑Bench和StreamingBench RTVU上对比，-4B模型在实时感知上提升1.4分、回溯上提升3.7分，且在RTVU子集上以半数参数接近8B模型；

**⚠️ 局限性**

限制在于fast‑weight记忆为固定大小的压缩摘要，无法完全替代全视频注意力，且在全视频可放入窗口时效果有限，需要进一步提升容量与选择性。

---

## 523. OpScale: Operator-level Provisioning and Autoscaling for LLM Serving

**arXiv ID:** 2608.13499 | [PDF](https://arxiv.org/pdf/2608.13499v1)

**作者:** Xingqi Cui `[一作]` (Rice University), Haoran Qiu `[通讯]` (Microsoft Azure Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OpScale，一个基于算子级别的 LLM 推理自动伸缩框架，能够在 GPU 集群中动态为每个算子分配资源并实现协同放置，从而显著提升资源利用率。

**💡 创新点**

创新点主要包括：①把伸缩单元从整模型降到算子级别，利用算子异质性实现细粒度资源分配；②设计了稀疏采样 + 多层优化的在线调度算法；③构建了干扰感知的 SM 分配与放置模型；④实现了低延迟的算子复制与分布式执行，支持 1 秒级伸缩。

**🔧 技术方法**

采用了 CUDA Green Context、SM 分配、NVLink/InfiniBand 跨 GPU 通信；基于片段插值的算子性能剖析；队列网络延迟模型；多流事件驱动流水线调度；以及 Python + CUDA 的实现。

**📊 数据集**

使用真实生产 LLM 服务流量轨迹（Qwen2-7B、Qwen2-57B-A14B），共 92.9 万请求、15 亿令牌，序列长度分布覆盖 1K–32K。

**📈 对比分析**

与模型级自动伸缩基线（DynamoLLM、AIBrix、Production Stack）以及功耗基线（DVFS、μ-Serve）进行对比；在 40 A100 / 24 GB200 集群上，OpScale 在满足相同 TTFT/TBT SLO 的前提下，GPU 数量减少 36.3%~44%，功耗降低 28%~35%，并在固定预算下吞吐量提升 44%。

**⚠️ 局限性**

局限性：仅针对单模型单租户场景，极低延迟（微秒级）任务和多租户共享资源的干扰建模尚不完善；算子级复制和调度在极高并发场景下可能产生额外开销；模型升级时需重新剖析算子性能。

---

## 524. GS$^{2}$CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors

**arXiv ID:** 2608.13502 | [PDF](https://arxiv.org/pdf/2608.13502v1)

**作者:** Yanming Yang `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种利用单次Snapshot Compressive Imaging（SCI）测量重建高质量3D场景的完整框架；

**💡 创新点**

创新点在于：①将SCI测量直接用于3D视觉基础模型（VFM）初始化，②设计了针对SCI的Opacity-Guided Splitting and Growth Regulation（OSGR）密度控制策略，③通过冻结的2D VFM生成伪视图监督进一步细化外观。

**🔧 技术方法**

使用的技术包括3D Gaussian Splatting、SCI成像模型、VFM（大规模预训练模型）进行几何与相机位姿初始化、L1+SSIM+opacity正则化的SCI一致性损失、OSGR中的局部不透明度拆分与全局不透明度调节、以及2D VFM的扩散式伪视图目标与alpha支持权重。

**📊 数据集**

实验数据集涵盖合成数据（NeRF Synthetic、DeblurNeRF、DTU、LLFF、DAVIS）和真实SCI数据（SCINeRF），并在不同压缩率、遮罩比例、相机运动复杂度下进行评估。

**📈 对比分析**

与现有SCI解码器（GAP-TV、PnP-FFDNet、FastDVDNet、EfficientSCI）以及3D基准（SCINeRF、SCIGS）进行对比；在六个基准场景中获得最高PSNR（≈38‑40 dB）、SSIM（≈0.99）和最低LPIPS（≈0.009），训练时间约53分钟，渲染速度≈408 FPS，显著优于SCINeRF（12h）和其它基准。

**⚠️ 局限性**

局限性主要是对静态场景的假设；在动态场景（如DAVIS Bear、Flamingo）表现仅为第二名，OSGR可能将运动导致的局部不透明度峰值误判为静态几何，从而产生伪影。

---

## 525. YAVIN: A Unified Architecture for Secure Edge Processing in Memory

**arXiv ID:** 2608.13496 | [PDF](https://arxiv.org/pdf/2608.13496v1)

**作者:** Shouzhi Fang `[一作]`, Alex K. Jones `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

无法判断论文内容

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

## 526. MARC v1: An Open-Source Multi-Agent Framework for Clinical AI Reasoning and Coordination

**arXiv ID:** 2608.13476 | [PDF](https://arxiv.org/pdf/2608.13476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 527. AlayaWorld: Interactive Long-Horizon World Modeling - Full Technical Report (v1.1)

**arXiv ID:** 2608.13492 | [PDF](https://arxiv.org/pdf/2608.13492v1)

**作者:** AlayaWorld Team `[一作]` (Alaya Lab), Zihui Gao `[通讯]` (Alaya Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 AlayaWorld 的 conditioning 过程进行了重构，统一了视觉、空间、时间记忆以及相机控制的 VAE 编码路径，以提升长时段交互式视频生成的连贯性。

**💡 创新点**

将运动感知图像条件、流式 3D 空间记忆、像素对齐时间记忆、硬性记忆 dropout、统一 VAE 协议以及基于几何的相机控制等六大改进统一到同一 causal VAE 框架中，消除多路分离的输入路径。

**🔧 技术方法**

采用 causal VAE、ViGeo 3D 估计、流式点缓存、像素级记忆窗口、硬 dropout、FlashAttention、几何对齐相机控制等技术。

**📊 数据集**

在 WBench 导航分割（158 个导航案例）上进行训练和评估。

**📈 对比分析**

与多种基准方法在 WBench 的 Video Quality、Setting、Interaction、Consistency、Physical 维度对比，AlayaWorld 在 Consistency 上取得 89.5 分最高，Video Quality 79.1 分同样表现强劲，尤其在 Background、Perspective、Subject、Geometric 一致性方面领先。

**⚠️ 局限性**

在 Setting 与 Physical 维度（场景一致性与因果保真）表现相对较弱，短期时间动态与物理交互建模仍有提升空间。

---

## 528. Hit-and-Run Mixes as Fast as the Ball Walk

**arXiv ID:** 2608.13487 | [PDF](https://arxiv.org/pdf/2608.13487v1)

**作者:** Ruizhe Zhang `[一作]` `[通讯]` (Purdue University), Ruizhe Zhang (Purdue University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在热起点下，随机取向的“hit-and-run”随机游走在任何等方位凸体上，以总变差距离ε与均匀分布混合所需的步数为O(n^2ψ_n^{-2}log^3(M/ε))，其中ψ_n是KLS常数，M为热度；

**💡 创新点**

在已知的n^2/ψ_n^2下界上，实现了从热起点出发的log^3(M/ε)阶复合依赖（取代之前的多项式依赖），并将混合时间与球步行的标度完全对齐；

**🔧 技术方法**

主要技术包括：指导性局部化（Klartag的引导分解）、局部半径加权的Poincaré平均法、对一维对数凹权重的紧凑等周围估计、以及通过局部重叠和加权周长实现的传输率下界；

**📊 数据集**

无数据集，本文为纯理论分析与证明；

**📈 对比分析**

相较于先前的Chen‑Eldan结果（n^2ψ_n^{-2}(M/ε)^{11}log^5(M/ε)）以及传统的球步行热起点混合时间O(n^2ψ_n^{-2}log(M/ε))，本文将热起点混合时间压缩到O(n^2ψ_n^{-2}log^3(M/ε))，在维度、KLS常数、热度与精度三者的依赖上几乎最优；

**⚠️ 局限性**

仅适用于热起点（M‑warm分布），对单点起点的混合时间仍保持Lovász‑Vempala的O(n^3log^3(M/ε))上界；此外结果依赖于KLS常数的上界，目前只能使用ψ_n^{-1}=O(log^{1/4}n)的估计，导致最终步数含有额外的log^{1/2}n因子；

---

## 529. Toward a Gricean Retreat: Probing LLMs for Knowledge Boundaries and Referent Specificity

**arXiv ID:** 2608.13484 | [PDF](https://arxiv.org/pdf/2608.13484v1)

**作者:** Dananjay Srinivas `[一作]` (University of Colorado), Maria Pacheco `[通讯]` (University of Colorado)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言模型在遇到未知实体时，是否能按照格里斯原则进行退后，平衡信息量与真确性。

**💡 创新点**

创新点在于构建针对知识边界与生成具体性的新基准，并证明模型内部已有对应信号但未被有效利用。

**🔧 技术方法**

采用线性探针分析隐藏层激活、LLM‑as‑judge评估生成真确性与具体性，以及惊讶度（surprisal）检验模型偏好。

**📊 数据集**

使用T‑REx/LAMA的8个维基数据关系子集，并人工合成未知实体和泛化对象作为测试数据。

**📈 对比分析**

在70M到12B参数的不同规模模型上进行对比，结果显示模型在未知实体场景下仍强烈偏好具体答案，无法实现理想的退后行为。

**⚠️ 局限性**

实验局限包括实体合成的污染风险、数据生成工具可能产生偏差、模型规模与评测范围受限，以及仅覆盖部分关系和模型。

---

## 530. CAPRI: Contract-Aware Proof Repair for Isabelle

**arXiv ID:** 2608.13459 | [PDF](https://arxiv.org/pdf/2608.13459v1)

**作者:** Jim Woodcock `[一作]` (University of York), Ran Wei `[通讯]` (University of Lancaster)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了 CAPRI 工作流，用以在利用大型语言模型（LLM）修复 Isabelle 证明时同时保证证明可构建与修复权限受限。

**💡 创新点**

创新点在于将证明构建与修复权限分离为两个独立的接受标准，并通过可机读的“编辑合约”与独立的合规检查器来实现对 LLM 生成补丁的授权验证。

**🔧 技术方法**

主要技术包括：LLM 提议补丁、独立合规检查器（基于差分与合约的静态检查）、Isabelle 证明构建、以及完整审计记录（包含合约、哈希、提问、回应、候选树、Isabelle 输出等）。

**📊 数据集**

使用了来自四个 Isabelle 开发项目的十二个任务作为基准，任务包括 SLEEC、Temporal UTP、Defeasible Logic 与 BorderSafe 四个开发环境，覆盖历史失败与人为诱导的受控错误。

**📈 对比分析**

通过对五种工作流条件（一次性、迭代+诊断、迭代+只修证明体、诊断一次性、延迟诊断迭代）在 180 次实验运行中进行定量评估，结果显示迭代工作流提升了成功率（从 22/36 到 31/36），而仅修证明体的接口在避免未授权修改方面表现最好；但整体修复率仅在 75% 左右，未能解决所有结构性失败。

**⚠️ 局限性**

局限性包括：基准任务规模有限，且来自作者维护的项目，难以代表更广泛的 Isabelle 证明修复场景；使用单一 LLM 配置，缺乏跨模型泛化验证；合规检查器未正式验证，依赖哈希与差分，可能误判无害更改；实验未充分分离迭代与诊断反馈的独立影响。

---

## 531. Before You Say It: Anticipating Verbal Behavior from Longitudinal Everyday Conversations with LLMs

**arXiv ID:** 2608.13454 | [PDF](https://arxiv.org/pdf/2608.13454v1)

**作者:** Yasith Samaradivakara `[一作]` (Massachusetts Institute of Technology), Pattie Maes `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于长时间可穿戴设备记录的自然会话数据，利用大语言模型挖掘可解释的情境-行为规则（IF‑THEN‑EXCEPT），并通过情境推理实现对用户下一次口头行为的预测；同时通过半结构化访谈评估预测结果对用户行为改变的支持作用。

**💡 创新点**

创新点在于：①将情境特定的行为模式抽取为人类可读的规则并实时更新；②将这些规则与LLM结合，形成情境条件化预测（Pattern‑Conditioned Prediction），显著提升对个体口头行为的预测准确性；③通过可视化模式供用户自检和干预，增强系统透明度与用户参与度。

**🔧 技术方法**

使用的技术包括：Gemini 2.5 Pro LLM；基于VRM分类的行为意图生成；规则挖掘与概率化评分；情境匹配与权重化激活；LLM‑as‑a‑judge自动评估（GPT‑5）；人工标注与人机对比；可穿戴设备的语音检测、转录与去标识化流程；以及访谈分析。

**📊 数据集**

数据集为14名受试者在7–10天内佩戴始终开启的智能手表收集的超过1000小时自然对话，总计约9900条经过清洗的utterance，其中约57%为用户说话，涵盖多种情境与情绪。

**📈 对比分析**

与零射击（Zero‑Shot）、全上下文（All‑In‑Context）、自然语言摘要（NL‑Summary）等基线比较，情境推理模型在LLM‑judge平均得分上达到0.597，较零射击提升28.9%、较全上下文提升18.9%。在人类评估中，该模型在43%的比较中排名第一，Kendall τ=0.83；跨个体转移实验表明模型表现明显下降，验证其个体特异性。

**⚠️ 局限性**

局限性包括：受试者数量有限（14人）和仅限手表语音采集的短期窗口；麦克风噪声和录音质量波动；评估仅在单回合层面，未检验跨会话或更长时序预测；当个体缺乏足够情境数据时，规则激活稀缺缺乏稳健的后备机制；以及对其他通信渠道（短信、社交媒体）的泛化仍未验证。

---

## 532. Mind the Context: Continual Learning of Socially Appropriate Robot Actions via Environmental-Social Disentanglement

**arXiv ID:** 2608.13448 | [PDF](https://arxiv.org/pdf/2608.13448v1)

**作者:** Rafal Robert Karpinski `[一作]` (Utrecht University), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了Explicit Disentanglement Dual-Branch (EDD) 框架，利用面向环境与社交线索的显式分解和回放式持续学习，实现社交机器人在多域室内环境下的行动适宜性预测。

**💡 创新点**

创新点包括：①将环境与社交信息显式解耦为两条输入通道并用双分支网络融合；②采用 panoptic 分割与随机遮挡构建环境视图；③在域增量持续学习中结合回放机制提升知识保留与跨域迁移。

**🔧 技术方法**

技术手段包括：双分支 CNN（MobileNetV2 编码器）+ 线性回归头；panoptic segmentation + 掩码遮挡；回放式经验重放；MSE 损失以及 Pearson、CCC 评估。

**📊 数据集**

使用了 OfficeDB 与 MannersDB+ 合成数据集（Pepper 机器人），共六个室内域（Home + 5 office），约2000张图像，标注九类行动的适宜性分数。

**📈 对比分析**

与 FedLGR、DUCA、DARE++、VLMs 等基线以及单分支/非回放对比。EDD 在 RMSE、PCC、CCC 三指标上均优于基线，RMSE 最低 0.783，相关系数最高 0.552/0.375，回放显著降低遗忘率（BWT）。

**⚠️ 局限性**

局限性：仅在合成数据上验证；依赖高质量 panoptic 分割；仅单机器人；对域序列顺序影响有限；未处理现实场景噪声和多模态输入。

---

## 533. ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models

**arXiv ID:** 2608.13438 | [PDF](https://arxiv.org/pdf/2608.13438v1)

**作者:** Gehan Zheng `[一作]` (Vanderbilt University), Weiming Zhi `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在机器人接触前进行预测性监测的系统（ContactGuard），能够在手爪闭合前评估即将执行的动作是否会导致失败，并在预测到失败时提前终止动作。

**💡 创新点**

创新点在于：①将动作条件的潜在世界模型与轻量级失败探测器结合，独立于底层视觉运动策略；②利用多视角潜在空间的“想象”未来状态来判别失败，而不是仅依赖当前观测或传统视频预测；③通过将未标记的机器人轨迹训练潜在动力学模型，再用少量标记数据训练线性探测器，实现低数据、高效部署。

**🔧 技术方法**

技术包括：多视角ViT-Tiny编码器、AdaLN-zero调制的Transformer潜在预测器、SIGReg正则化的潜在空间训练、线性逻辑回归失败探测器，以及基于预接触触发器的在线推理流程。

**📊 数据集**

数据集为：无标记的真实机器人轨迹（来自ACT和人类遥控），以及约250条标记的抓取成功/失败片段，用于训练探测器；所有实验在AgileX Piper双臂机器人上进行，包含杯子、盒子、笔记本、毛巾等四个抓取任务。

**📈 对比分析**

与多种基线对比：单摄像头LeWM模型、直接线性预测、当前潜在状态、FAIL-Detect、RND、SAFE等。结果显示，ContactGuard在所有任务的AUC和精确率/召回率上显著优于对比方法，尤其在多视角下提升显著；在真实机器人上，误报率低，能够在接触前成功阻断失败动作。

**⚠️ 局限性**

局限性包括：仅能通过放弃动作来预防失败，缺乏后续恢复或重新规划机制；只针对即将发生的接触事件，无法覆盖更长时序或多步骤任务；当前的状态融合实验表明粗糙的本体信息融合可能导致性能下降，需更精细的融合策略。

---

## 534. Fast Tendermint: Speeding Up a Foundational Consensus Protocol

**arXiv ID:** 2608.13434 | [PDF](https://arxiv.org/pdf/2608.13434v1)

**作者:** Preston Vander Vos `[一作]` (Circle Research), Daniel Cason `[通讯]` (Circle Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种在 n>5f 条件下的 Tendermint 变体，使其在良好情况下能在两步通信内达成共识

**💡 创新点**

通过合并锁定（locked）与有效（valid）状态、删除预投票（prevote）步骤，同时保留领导者轮转机制，最小化改动实现两步共识

**🔧 技术方法**

使用签名消息与 Gossip 通信模型，并借助 TLA+ 的表面语法 Quint 对协议进行形式化验证与模型检查，给出安全性、有效性与终止性证明

**📊 数据集**

协议为理论/形式化设计，未使用实际数据集

**📈 对比分析**

与原 Tendermint 在相同同步假设下对比，理论上将共识延迟从三步降低到两步；实验性能评估尚待实现

**⚠️ 局限性**

目前缺乏实测性能数据，协议实现与部署仍在进行中；在极端 Byzantine 场景下的慢路径与复杂视图变化机制未被完整探讨

---

## 535. Are You Sure You're Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity

**arXiv ID:** 2608.13430 | [PDF](https://arxiv.org/pdf/2608.13430v1)

**作者:** Irina Proskurina `[一作]` (Cohere Labs Community), Oyindolapo O. Komolafe `[通讯]` (Western University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了指令微调对问答任务中模型置信度和生成推理理由词汇多样性的影响

**💡 创新点**

创新点在于揭示指令微调虽提升模型置信度但不提升准确率，且词汇多样性变化异质且与置信度、校准不一致

**🔧 技术方法**

使用多选问答基准、概率熵、口头置信度、Unique‑2和Self‑BLEU多样性指标

**📊 数据集**

采用ARC‑Easy、MMLU和CommonsenseQA三大英文多选基准

**📈 对比分析**

通过与基线模型配对比较，发现指令微调使置信度上升、交叉理由多样性下降、准确率无显著提升

**⚠️ 局限性**

实验局限于仅考虑词汇层面的多样性、仅三套基准、未检验语义/句法多样性或多语言情况

---

## 536. Defensive Boosting for Online Probabilistic Forecasting

**arXiv ID:** 2608.13554 | [PDF](https://arxiv.org/pdf/2608.13554v1)

**作者:** Georgy Noarov `[一作]`, Aaron Roth `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Defensive Booster的在线概率预测算法，兼具梯度提升和弱强提升的两项不相容保证。

**💡 创新点**

在单一弱学习器下实现Brier误差与可学习判别的双重竞争，并给出误差高时的硬核证据。

**🔧 技术方法**

结合防御性预测、线性多准确性、自我正交性和第二阶在线学习的自适应梯度，利用弱学习器的无监管对残差进行多准确性校验。

**📊 数据集**

在两类合成流（弱学习器组与混合标签）以及四个公开二元预测数据集（银行营销、能源、电航班延误、办公占用）和三组回归流上进行实验。

**📈 对比分析**

与在线梯度提升、在线弱强提升及Brier聚合四组基线比较，Defensive Booster在所有流中均达到或接近最优，且每轮仅维护一个弱学习器，速度比集成快20–66倍。

**⚠️ 局限性**

在理论上仅满足单调Brier和弱学习条件的假设，且在极端对抗序列下可能无法同时达到两项最优；实验中未针对大规模高维特征进行深入评估。

---

## 537. Exponential Convex Calibration Dimension for the Multi-Label Jaccard Measure

**arXiv ID:** 2608.13549 | [PDF](https://arxiv.org/pdf/2608.13549v1)

**作者:** Mingyuan Zhang `[一作]` `[通讯]`, Mingyuan Zhang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对多标签任务的实例级 Jaccard 损失（IoU）进行理论研究，证明其损失矩阵秩与仿射维度均为 2^s-1，并给出精确的凸校准维度下界为 Θ(2^s)。随后提出两类多项式维度的近似替代方案：一是基于 F₁ 损失的误差传递得到常数上界 3-2√2；二是基于 MinHash 随机特征的平方损失，能够在任意 α 误差下实现 O((s^2+ s log(1/ρ))/α²) 维度。

**💡 创新点**

创新点包括：①首次用有限 MinHash Gram 表示和布尔 Möbius 逆推给出 Jaccard 损失矩阵完全正定的证明；②证明精确校准所需的凸校准维度为指数级 Θ(2^s)；③设计了两种多项式维度的近似替代方案，并给出了常数误差上界 3-2√2；④首次把 MinHash 随机特征与平方损失结合，用 Hoeffding 绑定实现分布无关的误差保证。

**🔧 技术方法**

使用的技术主要有：MinHash Gram 表示、布尔 Möbius 逆推、因子化加权分布构造、触发集几何与可行子空间理论、随机特征近似（Hoeffding 绑定）、凸校准维度理论与误差传递分析。

**📊 数据集**

未使用任何实验数据集，全部为理论推导和数理证明。

**📈 对比分析**

与已有的 F₁ 损失、Lovász 损失等方法比较，本文通过误差传递得到 Jaccard regret 上界 3-2√2；MinHash 方案在 α-近似下给出多项式维度 O((s^2+ s log(1/ρ))/α²) 的上界，说明随着 α 下降维度呈指数增长。

**⚠️ 局限性**

限制性：①精确校准仍需指数维度，无法在多项式空间内实现；②近似方案虽多项式维度，但解码仍需遍历 2^s 可能报告，未给出多项式时间解码；③常数与下界之间相差约一倍，最优维度尚未确定。

---

## 538. Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

**arXiv ID:** 2608.13546 | [PDF](https://arxiv.org/pdf/2608.13546v1)

**作者:** Yuanyang Yin `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Evoke，一种三步（CFG-free）的视频世界模型，能够在小时级持续生成并支持实时相机与文本控制。

**💡 创新点**

创新点：1) 将持久世界状态从 denoiser 中分离出来，存入外部以摄像机姿态索引的几何记忆库，保证每一步的上下文与位置范围不随会话长度增长；2) 重新设计教师网络，采用块级稀疏注意力和每块独立文本条件，实现长时间窗口监督与动态指令监督；3) 通过 30 秒全窗口分布匹配蒸馏，将教师的长时域一致性与动态响应迁移给仅三步推理的学生。

**🔧 技术方法**

使用的技术包括：稀疏块注意力（chunk-wise sparse attention）、线性注意力全局状态、分布匹配蒸馏（full‑window DMD）、三步 CFG‑free 迭代、基于深度估计的几何记忆读取与写入、可视化掩模权重过滤、时间化文本控制。

**📊 数据集**

主要训练与评估数据集：Sekai 视频数据集（用于训练）、WBench、VBench‑2.0、VBench‑Long（用于性能对比），以及内部收集的视频数据。

**📈 对比分析**

与现有少步系统比较：在 WBench 导航分组中领跑 Video Quality、Setting、Physical 等指标，在 VBench‑2.0、VBench‑Long 上保持与多步系统相当的整体分数；在长时间（两小时）会话中，视觉质量与光度统计保持稳定，计算成本保持恒定（单步 2.11 秒）。

**⚠️ 局限性**

局限性：1) 记忆库主要捕获粗略几何，细粒度对象身份与细节一致性有限；2) 仅存储静态几何，未建模对象动态状态与运动；3) 需要进一步加速推理（如更高压缩 VAE、低成本几何条件）以实现真正实时交互。

---

## 539. Safety vs. Social Image: Co-Designing Protection Mechanisms Against Ableist Harassment with People with Disabilities in Social Virtual Reality

**arXiv ID:** 2608.13532 | [PDF](https://arxiv.org/pdf/2608.13532v1)

**作者:** Kexin Zhang `[一作]` (University of Wisconsin-Madison), Yuhang Zhao `[通讯]` (University of Wisconsin-Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对11名残障人士的半结构化访谈和在A‑Frame构建的可视化近距离区块的VR场景中进行共设计实验，探究其对残障针对性骚扰的感知及其对保护机制的需求和偏好。

**💡 创新点**

首次从残障人士角度使用Hall的近距离理论来框定VR骚扰感知，并提出四类保护机制探针（Inform、Educate、Consent、Defense），揭示了近距离对骚扰评估和保护需求的关键作用，并强调保护机制同时是自我呈现的媒介。

**🔧 技术方法**

采用VR交互技术（A‑Frame、3D可视化近距离环）、定性访谈与定量安全感评分、Aligned Rank Transform (ART) ANOVA 以及主题分析等方法。

**📊 数据集**

基于11名残障参与者的实验数据（个人访谈记录、VR交互日志和安全感评分），未使用公开数据集。

**📈 对比分析**

没有与现有自动化骚扰检测/保护系统的直接对比实验，研究以定性洞察为主；因此无法给出数值性能指标。

**⚠️ 局限性**

局限包括样本规模有限、受试者多为英语使用者、实验环境为控制场景且缺乏真实社交互动、颜色视觉提示可能影响安全感评估、以及缺少跨文化或更大多样化残障群体的数据。

---

## 540. The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity

**arXiv ID:** 2608.13520 | [PDF](https://arxiv.org/pdf/2608.13520v1)

**作者:** Martin J. Wainwright `[一作]` `[通讯]`, Martin J. Wainwright

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了离散采样中的掩码扩散，提出了路径解析的未掩码增长复杂度，并利用该度量统一分析 Bernoulli 子集与固定卡迪纳尔未掩码采样器，设计了可从样本估计并以高概率实现给定 KL 误差的认证最优采样器。

**💡 创新点**

创新点在于：①提出了“未掩码增长复杂度”这一路径级信息几何度量，直接控制 KL 离散化误差；②证明了该度量可从样本中估计并实现自适应分块调度；③给出了与传统粗糙调度相比的 Ω(√d) 维度相关改进，并与最优 Euler 离散化误差对齐；④在理论上给出了常数因子内的迭代复杂度最优性。

**🔧 技术方法**

采用信息理论分析（KL 散度、互信息曲率）、log‑reveal‑odds 坐标、Euler 离散化、动态规划优化分块、尾部鲁棒估计与 Bernstein 约束、以及高概率集中不等式来构造和分析采样器。

**📊 数据集**

实验使用了多种合成离散分布（噪声重复比特、离散混合模型、层级混合模型、随机 XORSAT）来演示密度分布与采样效率之间的关系；未使用真实数据集。

**📈 对比分析**

与传统单块、固定卡迪纳尔或 Bernoulli 子集未掩码采样器相比，提出的方法在维度较高时可获得 Ω(√d) 的迭代次数改进，且在高概率下实现预定 KL 误差；其迭代复杂度仅比最优（oracle）方案多常数因子，且与 Euler 最优误差常数相匹配。

**⚠️ 局限性**

局限性包括：需要 Bayes（理想）去噪器和 KL 增量的矩条件；方法基于冻结后验的 Euler 离散化，尚未考虑更高阶离散化；对学习去噪器时的逼近误差缺乏完整的认证；并且对真实离散分布的估计可能受到样本量和高维稀疏性的影响。

---

## 541. A Browser-Native Digital Test Range for Benchmarking 4D Ocean-Glider Planning Algorithms

**arXiv ID:** 2608.13511 | [PDF](https://arxiv.org/pdf/2608.13511v1)

**作者:** Edward Holmberg `[一作]` (Louisiana State University), Mahdi Abdelguerfi `[通讯]` (Louisiana State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套安装无关、浏览器原生的数字测试平台，用于模拟海洋浮标规划、执行、观测和评估，并将规划到观测的全过程归一化。

**💡 创新点**

创新点在于：①构建统一的“计划‑观测”合同，拆分规划、执行与评估的边界；②将手工、传统和潜在学习规划器整合进同一工作流；③利用 WebAssembly/Pyodide 在浏览器中运行完整的海底仿真与评估。

**🔧 技术方法**

使用技术包括：WebGL 与 WebGPU 的 3D 可视化、Pyodide 运行 Python 科学库、NetCDF/HDF5 读取海洋数据、WebAssembly 加速计算、基于 GEBCO 和 HYCOM 的地形与潮流数据。

**📊 数据集**

数据集涵盖 GEBCO 2026 地形网格、HYCOM 预报子集（u/v、时间/深度）、以及自定义的 OSSE 参考场（先验、采样值、误差），并用公共 IOOS Glider 数据做现场对照。

**📈 对比分析**

通过在两个固定的 OSSE 事件中运行五种经典规划器（直接、跨阶、地形A*、潮流A*、科学A*），所有 54 次任务无硬违规完成。规划器的排名与行程长度、重构精度存在互换关系，表明不同目标间的权衡。

**⚠️ 局限性**

局限性包括：仿真使用的是简化的线性潮流驱动 kinematics，未能精准再现现场航迹；仅有 3 种随机种子；未加入观测噪声或更复杂的车辆动力学；未评估学习式规划器；平台不具备真实导航安全保证。

---

## 542. HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark

**arXiv ID:** 2608.13555 | [PDF](https://arxiv.org/pdf/2608.13555v1)

**作者:** Dairu Liu `[一作]` (Nankai University), Li Yi `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个约153小时、四类分门别类的大规模光学运动追踪基准，并提出了基于人类偏好的评估指标 HumanScore，用于衡量机器人运动追踪的整体轨迹质量。

**💡 创新点**

创新点包括：①HumanScore 通过学习人类视频偏好，形成与人类感知高度对齐的奖励模型；②引入分门别类的 153 小时光学数据集，显著扩大了现有评测数据的多样性；③制定统一的评测协议，将多种指标（Succ、MPJPE、HumanScore）整合到同一平台，实现公平可复现的比较。

**🔧 技术方法**

技术手段主要包括：Transformer 结构的奖励模型，使用 Bradley–Terry 损失训练；对轨迹采用 539 维特征表示（参考状态、模拟状态、接触力等）；统一的 MuJoCo 评测入口，记录完整状态信息；对人类偏好数据采用严格划分和随机化的标注接口。

**📊 数据集**

数据集为新发布的 153 小时光学捕捉数据，来源于 24 名专业表演者（舞蹈、健身、运动教练等），按 Daily、Highly Dynamic、Interaction、Ground 四个类别分组；与 AMASS、PHUMA、HumanML3D 等公开数据集对比，规模更大、类别更细。

**📈 对比分析**

比较方法：在统一评测框架下评估 GMT、TWIST2、SONIC、Humanoid-GPT 四个追踪器，报告 Succ（完成率）、MPJPE（关节角误差）和 HumanScore。HumanScore 与人工偏好的匹配率达 0.91，显示与人类感知高度一致；传统指标（Succ、MPJPE）与 HumanScore 不总是一致；总体来看 Humanoid-GPT 在 Daily 与 Highly Dynamic 上表现最佳，SONIC 在 Ground 上得到最高 HumanScore。

**⚠️ 局限性**

局限性：评测仅基于仿真环境和光学数据，缺乏真实硬件与多物理主体的验证；HumanScore 训练仅使用同步视频的偏好，可能在不同摄像角度或更复杂环境下的泛化有限；数据集仍未完全覆盖人类动作的长尾分布，未来需进一步扩充和多样化。

---

## 543. Performance Reporting of Mathematical Library Installations with LAAB - An Overview

**arXiv ID:** 2608.13512 | [PDF](https://arxiv.org/pdf/2608.13512v1)

**作者:** Aravind Sankaran `[一作]` (Forschungszentrum Juelich), Paolo Bientinesi `[通讯]` (Umea University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了LAAB框架，用于在HPC系统上对数学库安装进行性能基准、测量、分析与报告，满足可追溯性、兼容性、可靠性和可访问性四大目标。

**💡 创新点**

创新点在于将EasyBuild、JUBE、Pigeon等工具整合为完整的性能报告工作流；引入部分排名方法处理测量变异；支持核心库与高层框架之间的线性代数映射评估。

**🔧 技术方法**

使用Python编写的laab-inspector与laab-pigeon；Flask仪表盘；OpenBLAS、MKL、ScaLAPACK等数学库；SLURM/Flux调度；JUBE批量脚本；EasyBuild recipes。

**📊 数据集**

主要采用标准线性代数基准，如DGEMM 3000×3000、ScaLAPACK 15000×15000矩阵乘，覆盖不同块尺寸、不同处理元素数等。

**📈 对比分析**

通过多次运行同一算子在不同库/版本、工具链、NPE、块尺寸下收集执行时间，计算GFLOP/s、加速比；利用箱线图与部分排名评估性能波动；实验表明库间性能相近但存在显著回归，块尺寸影响可扩展性但差异不稳定。

**⚠️ 局限性**

局限性包括依赖EasyBuild recipes；仅支持已定义的基准步骤；测量噪声仍可能导致排名不确定；缺乏对稀疏线性代数或自定义算子的自动化支持。

---

## 544. PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives

**arXiv ID:** 2608.13552 | [PDF](https://arxiv.org/pdf/2608.13552v1)

**作者:** Kaixin Ding `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PlayWorld 基准，用多模态 Agent Player 模拟人类玩家评估交互式视频世界模型。

**💡 创新点**

创新点在于：①采用长期目标而非预定义轨迹；②通过 Agent Player 在线调整动作，实现跨模型公平比较；③设计 VQA rubrics 评估四个核心维度（几何一致性、交互逼真性、视线之外演化、洞察演化）。

**🔧 技术方法**

核心技术包括：多模态语言模型（Claude/Hailk、Gemini 3.1 Pro）作为 Agent Player；浏览器自动化与模型接口实现控制；VQA 验证器结合人工标注的 Rubrics；自动视频质量与控制可控性指标。

**📊 数据集**

使用 171 条人类标注案例（涵盖 50 种动作模式，10–60 秒回放）构建 benchmark；每个案例包含初始世界、长期目标、基础动作序列和特定 Rubric；来源为公开图像与人工构造场景。

**📈 对比分析**

比较方法：对 9 款世界模型在相同初始场景、目标与动作序列下进行 Agent Player 交互；评估 Rubric 结果（1–5 分）与自动指标；实验显示闭源模型 Genie 3 在几何一致性与交互逼真性上表现最好，但整体仍低于 5 分，表明长期交互能力仍不足。

**⚠️ 局限性**

局限性在于：①仍依赖人工标注的 Rubrics，主观性可能影响一致性；②多模态模型的推理延迟与网络环境有关；③对视线之外与洞察演化的评估受限于当前可观测的实例；④仅覆盖 171 条案例，未来需扩展更多多样化情境。

---

## 545. LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure

**arXiv ID:** 2608.13545 | [PDF](https://arxiv.org/pdf/2608.13545v1)

**作者:** Fanfei Li `[一作]` (Max Planck Institute for Intelligent Systems), Wieland Brendel `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于美国小学课程的88B-token训练语料库和一款5B参数的语言模型，构建了一个受控的学习沙盒。

**💡 创新点**

创新点在于通过发展阶段过滤、年龄获得筛选、LLM评判注释以及符号过滤等多阶段管道精确限定训练数据的知识范围，从而把模型的知识边界明确化。

**🔧 技术方法**

使用的技术包括基于Qwen3架构的自训练、LLM-as-a-judge进行注释、FastText与ModernBERT分类器、正则表达式符号过滤、频率采样以及后训练的监督微调与GRPO强化学习。

**📊 数据集**

数据集为从公开网络文本中提取、过滤后得到的88B-token小学教材文本，并对CommonCoreText、WeeBit等标准化教材进行验证。

**📈 对比分析**

与在未过滤文本上训练的基线模型和Gemma 2B进行对比，在校内范围内（Grade 0‑5）模型表现良好，但在高于5年级的任务上，规模提升、后训练或ICL并未显著提升性能，保持在训练边界附近。

**⚠️ 局限性**

局限性包括模型规模仅为5B，难以体现大规模模型的自我学习特性；受限训练集导致对高级数学与科学概念的泛化能力不足；RL、ICL等方法在本框架下未能突破知识边界；并且与真实儿童学习过程仍存在差距。

---

## 546. Vero: Can AI Agents Build Formally Verified Software Repositories?

**arXiv ID:** 2608.13522 | [PDF](https://arxiv.org/pdf/2608.13522v1)

**作者:** Zhe Ye `[一作]` (University Of Chicago), Dawn Song `[通讯]` (University Of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了第一个在 Lean 4 上进行仓库级实现与证明联合生成的基准（Vero）。

**💡 创新点**

创新点在于将多模块真实仓库转化为可验证 Lean 4 实例，支持代码和证明双模式，并引入正式审计机制来纠正基准错误。

**🔧 技术方法**

使用了多语言翻译管道、LLM 代理自动化、Lean 4 工具链、机器学习驱动的编码代理以及审计证明。

**📊 数据集**

数据集包含 43 个多模块实例，来自 Dafny、Verus、Coq 与 Python 的真实仓库，共计 743 个 API 和 2705 条规范。

**📈 对比分析**

对比四种前沿编码代理（GPT‑5.5、Claude Opus、Claude Sonnet）在代码+证明和证明‑only 两种模式下的性能，最强配置仅完全解决 27/43 个实例，表现出对共享不变量和可重用证明库的不足。

**⚠️ 局限性**

局限性包括仅支持 Lean 4、缺乏并发或时序协议的覆盖、基准质量依赖人工审核以及审计机制无法验证规范的语义正确性。

---

## 547. TabSOM: A tabular-to-image encoding method based on self-organizing maps

**arXiv ID:** 2608.13513 | [PDF](https://arxiv.org/pdf/2608.13513v1)

**作者:** David Chushig-Muzo `[一作]` (Rey Juan Carlos University), Diego H. Peluffo-Ordóñez `[通讯]` (Yachay Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于自组织映射（SOM）的表格数据转图像编码方法 TabSOM，能够在图像中同时编码每个特征的取值和特征之间的交互关系；

**💡 创新点**

创新点包括：①利用SOM的组件平面提取特征位置并通过匈牙利算法实现无碰撞布局；②构建基于SOM几何的特征关系图并作为图像的第三通道；③提供基于SOM的全局特征重要性和原型依赖曲线两种可解释性工具；

**🔧 技术方法**

主要技术包括SOM训练、组件平面计算、匈牙利分配、Pearson相关构建关系图、多尺度高斯渲染、CNN分类和可解释性分析；

**📊 数据集**

使用了四个公开二分类数据集：Pima Indians Diabetes、Oxford Parkinson’s Disease、QSAR Biodegradation 和 Wisconsin Breast Cancer；

**📈 对比分析**

与12种现有表格-图像编码方法在四个数据集上进行5折交叉验证比较，TabSOM在每个数据集上名列前两，平均AUC接近最佳，且方差最低，表现稳健；

**⚠️ 局限性**

局限性包括：①对高维特征仍可能存在位置分布拥挤；②依赖SOM训练的稳定性，参数选择对结果有影响；③缺乏对多分类和回归任务的直接扩展；

---

## 548. V-RAE: Rethinking Video Latent Spaces for Generation

**arXiv ID:** 2608.13556 | [PDF](https://arxiv.org/pdf/2608.13556v1)

**作者:** Minghui Guo `[一作]`, Hao Fei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 V-RAE，一个利用冻结的视觉表示编码器构建的视频表示自编码器，用于高效压缩和生成视频。

**💡 创新点**

在语义空间上直接构建可生成的潜在空间，结合可学习的时间池化和时空 Transformer 解码器，并引入 tFVD 诊断指标。

**🔧 技术方法**

使用冻结的视觉基线模型（如 DINOv3、SigLIP2、EUPE、V-JEPA）、可学习的时间注意力池化、3D RoPE 时空 Transformer 解码器、DiT 生成模型与 rectified flow。

**📊 数据集**

在 UCF101、Kinetics-600（K600）进行重建与有条件生成，SSv2、K400 进行语义探测，Cityscapes 进行未来视频预测。

**📈 对比分析**

与多种现有视频 VAE / tokenizer 在 rFVD、gFVD、tFVD 等指标对比；V-RAE 在 K600 上 rFVD 2.13、UCF101 上 6.12，gFVD 在 UCF101 117.86、K600 19.16，速度比 Wan2.2 VAE 6 倍快，且语义保持更好。

**⚠️ 局限性**

依赖冻结编码器的语义质量，时间池化可能略损失语义；tFVD 尚未成为标准评估指标，模型仍需在更大规模或多样化数据上验证。

---

## 549. QuoteBench: How Matched Scores Can Hide Command-Path Failures

**arXiv ID:** 2608.13547 | [PDF](https://arxiv.org/pdf/2608.13547v1)

**作者:** Shangao Li `[一作]` (Stony Brook University), Yuanyuan Yang `[通讯]` (Stony Brook University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了QuoteBench，聚焦一发式Bash命令生成中的引用/转义失误，测量生成与执行路径的差异。

**💡 创新点**

通过固定回复重放与跨路径分析，分离生成错误与执行损伤，揭示匹配分数隐藏的后期执行失败；提出同时报告生成契约、执行路径与最终状态验证器的评估标准。

**🔧 技术方法**

采用结构化跨路径实验设计、固定回复重放、最终状态验证器、ShellCheck、Git、JSON验证等工具，对命令层级的转义与解析机制进行深入分析。

**📊 数据集**

56个一发式Bash任务，覆盖14个操作族（文件内容、文件名、正则、heredoc、JSON、Git、SSH模拟等），包括公开与私有payload，形成冻结核心版本。

**📈 对比分析**

在八个相同窗口配置下，用匹配分数与跨路径重放比较模型；测量RN-RR（transport damage）、NN-RN（compensation）和NN-RR（匹配差距）。模型表现从14.3%至98.2%，最佳模型达到56/56；不同模型的传输损伤与补偿差异显著。

**⚠️ 局限性**

仅针对bash单发命令，覆盖单一解析边界；未涵盖多轮交互、其他shell、网络、CI等场景；数据集有限，难以推断真实部署频率；effort ladder缺乏因果性。

---

## 550. SCULPT: Subtractive Composition for 3D Part Generation

**arXiv ID:** 2608.13541 | [PDF](https://arxiv.org/pdf/2608.13541v1)

**作者:** Sikuang Li `[一作]` (Shanghai Jiao Tong University), Qi Tian `[通讯]` (Huawei)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过从完整物体的结构化3D潜在空间开始，使用逐步的减法拆分预测器（joint split predictor）将物体逐一拆分成语义部件，并在每一步保持剩余物体状态，形成可变数量的部件集合。

**💡 创新点**

创新点在于：①提出“减法生成”（subtractive composition）框架，将完整体拆分为部件与剩余体的联合生成；②设计了联合拆分预测器，利用图像与剩余体双重条件在同一稀疏支持上同时生成部件与剩余体；③引入稀疏支持组合损失、推理时支持裁剪与空剩余终止机制，确保拆分过程保持一致且无缝接合。

**🔧 技术方法**

技术细节包括：使用预训练的 TRELLIS.2 整体生成模型；在其结构化潜在空间（voxel、几何、材质三阶段）上构建分解流 Transformer；在每个阶段使用 ControlNet 级别的剩余体条件；采用重排（joint denoising）和分支控制；利用组合损失约束部件与剩余体覆盖原始支持；推理时采用阈值裁剪保证剩余体不溢出。

**📊 数据集**

训练数据来自 PartVerse-XL（约 37,425 对象，330,455 组拆分），该数据集基于 Objaverse-XL，包含完整物体与人工标注的部件。评估数据使用 PartObjaverse 基准（200 个带部件标注的网格），以及额外的四张数据集图片、一次文本生成图片和一张真实照片，用于测试跨域表现。

**📈 对比分析**

与 Part123、OmniPart、TRELLIS+PartField、HoloPart、Treillis.2+PartField 等基线比较，SCULPT 在 Part、语义组和整体三层次上均实现最低 Chamfer Distance（CD）和最高 F1@0.05，尤其在整体层次上取得 CD 0.0020、F1@0.05 0.9212，明显优于所有竞争方法。

**⚠️ 局限性**

局限性包括：①拆分迭代长度受固定上限 Kmax=24 限制，无法处理极其复杂或细小的部件；②对极端图像或极大物体的鲁棒性仍需进一步验证；③缺乏可控拆分粒度的直接接口，无法精确指定拆分深度或部件数量；④界面交互（如网格对齐）在复杂接触区域仍有改进空间。

---

## 551. DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees

**arXiv ID:** 2608.13524 | [PDF](https://arxiv.org/pdf/2608.13524v1)

**作者:** Tianyi Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DARTree，一种无训练的推测解码方法，将带因果纠正的块并行草稿从单链扩展到树结构。

**💡 创新点**

通过深度级并行展开和延迟最佳优先剪枝，将路径依赖的因果纠正与树搜索解耦，显著提升接受长度和速度。

**🔧 技术方法**

使用块并行草稿、因果纠正头、深度级批量树构建、延迟最佳优先剪枝以及单层树验证等技术。

**📊 数据集**

在 GSM8K、MATH‑500、AIME25、HumanEval、MBPP、MT‑Bench、Alpaca 等七个数学、代码和聊天基准上进行评估。

**📈 对比分析**

与标准 AR 解码、DFlash、DDTree、Domino 在 Qwen3‑4B/8B、温度 0/1 的设置下对比，平均接受长度提升 28.9% 以上，速度提升 22.7% 以上，最高可达 9.73×。

**⚠️ 局限性**

仍需预训练的纠正头；宽层展开会增加内存占用；在极长序列或高温度下的性能表现尚待进一步验证。

---

## 552. Intervention-Aware Clinical World Model for Post-Op Outcome Forecasting in Cardiology

**arXiv ID:** 2608.13518 | [PDF](https://arxiv.org/pdf/2608.13518v1)

**作者:** Yunsung Chung `[一作]` (Tulane University), Jihun Hamm `[通讯]` (Tulane University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一种基于干预感知的临床世界模型，用于心脏病学术后（AF消融）随访期间的风险预测与结构恢复。

**💡 创新点**

创新点在于：① 用三维空间潜在状态表示心房解剖并随时间演化；② 将消融几何、异步临床事件与ECG上下文编码为事件token并通过Transformer进行上下文聚合；③ 引入终端时点token实现随时预测；④ 在训练中使用后随访MRI的潜在匹配损失提供结构监督。

**🔧 技术方法**

核心技术包括：变分自编码器（VAE）用于3D潜在编码；Transformer编码器用于事件上下文；3D残差CNN与MLP实现潜在状态的时间步更新；ECGFounder嵌入提供ECG信息；多任务损失（潜在匹配、BCE、Huber）进行联合训练。

**📊 数据集**

使用DECAAF-II多中心队列：完整记录91例（含预后MRI、消融图、事件、ECG）进行内部交叉验证；另外258例无可用消融几何用于无几何实验。

**📈 对比分析**

与静态模型、后随访MRI单独模型、ECGFounder、SOFA、GRU、LSTM、Transformer序列模型进行对比；在完整记录集上实现AUROC 0.756、AUPRC 0.777、scar MAE 2.97%，明显优于所有基线模型；在无几何实验中AUROC 0.713、AUPRC 0.747，同样显示优越性。

**⚠️ 局限性**

局限性包括：完整记录样本量仅91例，结果仅在DECAAF-II内部验证；消融几何映射手工完成，可能存在定位误差；事件记录可能不完整或受混杂影响；未进行外部验证与因果推断，且对缺失模式、ECG窗口选择等因素未做系统评估。

---

## 553. DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data

**arXiv ID:** 2608.13517 | [PDF](https://arxiv.org/pdf/2608.13517v1)

**作者:** Peter Schneider-Kamp `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从零开始训练一个 1 B 参数的层级推理模型 Mimir v1，使用仅授权的 70.5 B token 混合语料（161 个公开或经过审核的数据集）并结合合成的“移植”数据集，专注于丹麦语与英语任务。

**💡 创新点**

创新点在于：① 采用 HRM‑Text 体系结构，专门关注可授权后训练数据；② 通过人工合成可许可的“移植”数据替代非许可数据，实现高质量指令遵循与推理能力；③ 在同等规模下以可持续的硬件与开源工具实现与更大模型竞争。

**🔧 技术方法**

核心技术包括：HRM‑Text 层级推理架构（隐藏层 1,536，12 头，2 H‑cycle/3 L‑cycle）、Gemma‑4 分词器、Chat 模板训练、FlashAttention 与 vLLM 交互、AdamW 优化器、全共享数据并行（FSDP）和 8×NVIDIA B200 GPU 训练。

**📊 数据集**

数据集：161 个公开或审核通过的数据集，覆盖丹麦/英语指令与知识、数学推理、代理工具调用、机器翻译、科学总结等八大功能类别；主要贡献者为：丹麦指令集 22 %（如 laerebogen 8.32 B）、英语指令集 19 %（如 Dolci、Tulu 3）、Sapient 混合 17 %（Flan/Platypus）、数学与推理 15 %（OpenMathInstruct、AceReason）、其余合成与工具数据 10 %+。

**📈 对比分析**

评估使用 20 个基准（英语、数学、代码、丹麦）对比 HRM‑Text、Qwen 3.5、Gemma 3/4、SmolLM 等同类模型；Mimir 在 BoolQ、Winogrande、DROP、GSM‑8K、HumanEval 以及丹麦语语法、问答、摘要任务中取得最优或接近最优分数，单参数模型仅落后于 4 B 级别模型 0.3 分（英语）或 3.8 %（数学/代码），但仍表现出色。

**⚠️ 局限性**

局限性包括：对合成“移植”数据的依赖仍可能限制真实场景多样性；在数学/代码领域落后于 5 B 级别模型；助手能力相对受限，尚未采用强化学习或进一步的推理微调；在可扩展性、推理速度与多模态支持方面仍需改进。

---

## 554. AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design

**arXiv ID:** 2608.13560 | [PDF](https://arxiv.org/pdf/2608.13560v1)

**作者:** Yaxin Luo `[一作]` (Meituan), Xiaotong Li `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种自动化设计框架AutoDesign，通过元 harness 优化迭代和长周期代理工作流，实现了在极少人工干预下为学术论文生成高质量海报的全过程。

**💡 创新点**

创新点在于：①将设计 harness 视为可优化对象，引入元 harness 迭代自动提升；②利用代理（Coding Agents）与批判反馈循环，实现源代码消化、迭代生成、精细化改进直至完成；③在性能达到平台期后引入人工引导进一步提升。

**🔧 技术方法**

核心技术包括：Meta-harness 迭代优化、长周期代理工作流、批判者（critic）反馈机制、自动代码消化与生成，最终实现设计 harness 与代理的协同优化。

**📊 数据集**

实验数据集未在摘要中给出，作者使用了代表性论文集及其对应的海报生成任务，示例中单篇论文的海报被 AutoDesign 生成。

**📈 对比分析**

评估方法：对比不同迭代阶段的海报生成质量分数，结果显示经过元 harness 优化后，所有 Coding Agents 的得分提升 5.0~19.6 分，最终最高整体分数达到 81.5 分；同时在约 40 分钟内完成全部流程。

**⚠️ 局限性**

局限性：①实验范围局限于单篇论文海报生成，未验证跨领域泛化能力；②对元 harness 迭代的自动化停止标准不够明确；③仍需人工在性能平台期后介入，缺乏完全端到端自动化。

---

## 555. OmniScientist: An Omni-Modal Omni-Discipline AI Scientist

**arXiv ID:** 2608.13558 | [PDF](https://arxiv.org/pdf/2608.13558v1)

**作者:** Bobo Li `[一作]` (National University of Singapore), Wynne Hsu `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个端到端、全模态的 AI 科学家框架，直接从原始多模态证据出发，依托三位自治代理（构思、实验、写作）完成研究问题生成、实验设计、结果验证与论文撰写。

**💡 创新点**

创新点在于：①把多模态感知嵌入整个科研生命周期，使原始数据本身驱动问题与实验；②通过确定性流水线与代码强制检查保证新颖性、统计有效性、追溯性与反 HARKing；③实现跨学科、跨证据族的统一框架，证明感知对科学发现至关重要。

**🔧 技术方法**

采用 ReAct 代理循环、统一的多模态感知模型（视觉‑语言、音频、3‑D、序列等）、代码执行环境、统计与多重比较校正、先行文献检索（OpenAlex）、以及多种 LLM 背景（Sonnet‑5、GPT‑5.6、GLM‑5.2、Kimi‑K2.7、Qwen3.5、Gemma‑4）。

**📊 数据集**

使用 36 个真实公开数据集，覆盖 5 个学科族（物理、地球与空间、生命医学、农业生态、工程信息），包括图像、信号、音频、视频、3‑D、轨迹、表格、公式与图谱等多种模态。

**📈 对比分析**

通过与只提供预计算标量特征的盲目基线进行头对头对比（85% 胜率），以及在 9 种 LLM 背景下的多维度评测（新颖性、可靠性、清晰度、意义、可复现性、多模态扎根、事实准确性），平均整体得分 6.3，感知提升多模态扎根 +2.8、意义 +1.8，组件消融显示先行文献检索、迭代代理循环与代码检查对性能贡献最大。

**⚠️ 局限性**

局限性：受限于当前 LLM 与感知模型的能力；仅评测 36 个数据集，未覆盖全部科学领域；计算成本较高；仍需人工编写任务规范文件；感知模型可能带来偏差；在弱模型或极端模态下表现下降；未完全实现所有科研任务的自主化。

---

## 556. SAEVerbalizer: Generating Explanations for Sparse Autoencoder Features via Representation Verbalization

**arXiv ID:** 2608.13538 | [PDF](https://arxiv.org/pdf/2608.13538v1)

**作者:** Weihan Meng `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SAEVerbalizer，将 SAE 解码方向注入 LLM 表示并微调后直接生成自然语言解释；

**💡 创新点**

创新点在于从内部表示直接生成解释，避免外部观察、提升效率；实现轻量化框架，可跨不同 SAE 字典和 LLM 迁移；对多方向注入、方向反转等进行深入干预分析；

**🔧 技术方法**

使用 Gemma LLM（1B、4B、27B）作为背骨，Gemma Scope 2 SAEs；采用特征‑解释对进行微调，使用轻量化适配器映射不同 LLM 表示；注入方式为归一化加法，控制注入强度；

**📊 数据集**

训练数据来自 Neuronpedia 过滤后的特征‑解释对；测试集分为 GTS、LIG、GG 三个分层样本；使用 Gemma‑3‑1b‑it、4b‑it、27b‑it 以及对应的 SAE；

**📈 对比分析**

评估指标为 Reference Agreement (RA)，与无监督基线对比；在 48k 训练集下，RA 达到 52.3%（GTS）、80.5%（LIG）、56.1%（GG）；在更小训练集（12k）时，性能随 LLM 规模和层深度提升；跨 SAE 字典与跨 LLM 的适配器迁移均能保持或提升性能；

**⚠️ 局限性**

局限性：仅在 Gemma LLM/SAE 体系上验证，泛化性待验证；参考解释基于观察而非真实标签，可能误导；实验仅使用单一随机种子，未评估方差；监督构建成本高，需要高质量过滤。

---

## 557. Joint Communication-Control Strategy Optimization with Partially Nested Information Structures: The Linear-Quadratic Case

**arXiv ID:** 2608.13535 | [PDF](https://arxiv.org/pdf/2608.13535v1)

**作者:** Haoyi You `[一作]` (University of Maryland), Kaiqing Zhang `[通讯]` (University of Maryland)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究多智能体线性二次高斯系统下的联合通信与控制策略优化（JCCO）问题，探讨在部分嵌套信息结构下如何通过动态规划得到最优控制和通信策略；

**💡 创新点**

创新点在于①提出了保持部分嵌套性的结构条件，证明若违反则最优控制可能非线性或不存在；②在此框架下给出开闭环通信策略的Riccati式动态规划，能够同时求解控制与通信最优方案；③将方法推广至闭环通信，克服SI‑CIB条件失效问题；

**🔧 技术方法**

主要技术包括：基于公共信息框架的CIB策略、部分嵌套信息结构分析、策略无关CIB信念（SI‑CIB）条件、Kalman滤波、Riccati递推与动态规划、线性控制策略的可行性证明；

**📊 数据集**

实验采用合成随机数据：随机生成系统矩阵、噪声协方差、通信成本参数α等，未使用公开真实数据集；

**📈 对比分析**

与不同基线共享协议（单控制器、轮流控制器、一步延迟、单向一步延迟）和不同通信成本α进行对比；结果显示通信成本越低，代理共享信息越多，导致控制成本下降，总成本降低；同时演示了多种通信策略在不同α下的总成本折衷；

**⚠️ 局限性**

局限性包括：①仅适用于高斯线性系统，非高斯或非线性系统需进一步研究；②对信息结构需满足部分嵌套及若干假设，实际系统可能不满足；③实验仅在合成数据上验证，缺乏真实场景验证；④闭环通信方案仍需要假设通信策略只依赖公共信息，进一步放宽假设仍是挑战。

---

## 558. Measuring Task-Agnostic Training Data Influence Across Language Model Pretraining

**arXiv ID:** 2608.13515 | [PDF](https://arxiv.org/pdf/2608.13515v1)

**作者:** Yuto Nishida `[一作]` (Nara Institute of Science and Technology), Masaru Isonuma `[通讯]` (NII LLMC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种任务无关的训练样本影响度量，定义为每个样本梯度更新后使模型参数距离最终预训练参数的平方距离的减少量。

**💡 创新点**

创新点在于不依赖下游任务或验证集，而是以最终模型参数为参照点，利用中间检查点近似该影响度量，并在大规模 LLM 预训练中实现可扩展评估。

**🔧 技术方法**

技术手段包括梯度方向与最终参数方向的内积、L2 距离计算、基于检查点的近似估计（Checkpoint‑Based Approximation）以及与 TracIn 的对比分析。

**📊 数据集**

实验使用 Pythia 与 PolyPythia 族的 6 个规模（70M–12B）模型，训练数据来自 The Pile（300B 词元）并在 154 个检查点上采样 1000 条样本。

**📈 对比分析**

通过与精确梯度贡献的 Pearson/Spearman 相关性（早期约 0.6，中期 0.8，后期 0.95）验证近似精度，并与基于领域验证集的 TracIn 比较，证明跨模型规模和数据顺序的鲁棒性；相比任务特定方法可揭示从文献到 STEM 的跨越动态。

**⚠️ 局限性**

局限性包括：依赖最终检查点作为参照点导致结果受终点选择影响；近似方法假设 SGD 与梯度正交，未完全覆盖 Adam 等自适应优化器；仅评估 300B 词元数据，未探讨更大规模或不同任务的普适性。

---

