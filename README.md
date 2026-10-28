# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-28 | 今日论文总数: 770

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Pricing Cross-Bank Context in Federated Anti-Money Laundering

**arXiv ID:** 2610.35996 | [PDF](https://arxiv.org/pdf/2610.35996v1)

**作者:** Lidia Losavio `[一作]` (Università della Svizzera italiana), Marc Langheinrich `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 7185 | [OpenAlex ID](https://openalex.org/A5069422623)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究跨行交易网络中联邦图神经网络的可行性，提出基于可观测交易片段的基线以及渐进式信息披露阶梯，以评估在数据分片下的反洗钱检测性能。

**💡 创新点**

设计了无额外负载的可观测基线和两级信息披露接口，系统性量化不同隐私保护下交易计数与节点状态共享的增益，并给出银行实践的简易规则。

**🔧 技术方法**

采用多层图神经网络（GIN、PNA、TransConv、SALT系列、FraudGT）在联邦FedAvg框架下训练，并使用可观测切片、交易计数和一跳边界状态的隐私保护机制（Gaussian/几何噪声）。

**📊 数据集**

在公开的 AMLworld HI‑Small 交易图数据集上进行实验，进行时间顺序划分、传递式学习，并在多种分片（4–32 段）下评估。

**📈 对比分析**

与局部训练、边缘丢弃联邦、集中式训练及现有 FedGCN/FedGAT/FedSage+ 进行对比；可观测基线在分片增大时仍保持高 F1，计数披露可在强隐私下仍显著提升性能，边界状态则在大多数情况下失效。

**⚠️ 局限性**

仅考虑基于交易特征的边驱动模型；实验使用抽象数据集，缺乏真实银行验证；边界状态共享对隐私保护极度敏感；未覆盖多重图、反向消息等更复杂场景。

---

## 2. Beyond Structural Unification: A Perspective and Roadmap on Unified Multimodal Models

**arXiv ID:** 2610.35906 | [PDF](https://arxiv.org/pdf/2610.35906v1)

**作者:** Yinyi Luo `[一作]` (Carnegie Mellon University), Jindong Wang `[通讯]` (William & Mary)

**通讯引用:** 17161 | [OpenAlex ID](https://openalex.org/A5100700956)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一多模态模型（UMM）功能统一的框架与评估协议，阐述了结构统一与功能统一的区别，并指出了“统一悖论”；

**💡 创新点**

核心创新在于将统一过程拆分为五个关键层次（输入不对称统一、表征分解统一、状态化推理统一、协同优化统一、多中心系统评估统一），并给出功能统一的量化指标（dividend D、Δ_k）和干预矩阵；

**🔧 技术方法**

主要使用理论分析、统计置信区间（配对自助抽样）、贝叶斯置信区间、梯度对齐与Pareto最优奖励等技术来构建评估与优化策略；

**📊 数据集**

未使用具体数据集，论文仅引用已有基准（如VQA、MMVU、MM-Vet、TokenFlow、POPE、HallusionBench）但未在新的数据集上进行实验；

**📈 对比分析**

论文未给出实验对比或性能数值，评估方法仅是概念性描述，尚未在任何实际模型上验证；

**⚠️ 局限性**

主要限制是缺乏实证评估，方法仅为概念化的蓝图；缺少可复制的实验结果，且对不同部署环境的具体参数、权重、阈值等需进一步阐明。

---

## 3. RV565: Cross-Layer Co-Design for Frame-Independent Video Playback on MicroPython-Class ESP32-S3 Systems

**arXiv ID:** 2610.35935 | [PDF](https://arxiv.org/pdf/2610.35935v1)

**作者:** Ayushman Bhattacharya `[一作]` (Pollinations AI), Anwesha Chakraborty `[通讯]` (JIS University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种面向微控制器的无间帧、RGB565原生视频容器，并在ESP32‑S3上实现了双核重叠解码流水线，实现24 FPS无遗漏播放；

**💡 创新点**

创新点包括：①每帧独立Deflate压缩使解码成本仅为载荷长度的线性函数；②通过双核重叠把解码移至关键路径之外；③提供基于容器元数据的即刻可调度性预测；

**🔧 技术方法**

技术细节涵盖ESP32‑S3、ESP‑IDF、MicroPython OreoOS、Deflate熵解码、最近邻缩放、SPI‑GDMA、绝对截止调度、长度前缀索引；

**📊 数据集**

使用Big Buck Bunny影片的两个10秒窗口（240帧，180×135，24 FPS）作为测试集；

**📈 对比分析**

对比串行路径、双核重叠路径以及集成在OreoOS中的实现；双核重叠实现24 FPS无丢帧，串行在较大负载下丢帧；集成实现仅12 FPS；每帧耗时、内存占用等指标均被量化；

**⚠️ 局限性**

局限性：仅在单一ESP32‑S3板、单一分辨率、单一RGB565格式下评估；未考虑音频、流式输入、不同压缩水平或其他视频编码；缺乏图像质量度量；未测量输入延迟与多核调度的实时性。

---

## 4. Continue or Restart? The Limits of Early Failure Prediction in Coding Agents

**arXiv ID:** 2610.35971 | [PDF](https://arxiv.org/pdf/2610.35971v1)

**作者:** Srinath Srinivasan `[一作]` (NC State University), Wesley Assunção `[通讯]` (NC State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了编程代理中基于早期执行前缀的失败预测，并评估其在共享token预算下的停止策略效果

**💡 创新点**

证明早期前缀对同一问题的不同尝试几乎无区分度，导致冷重启在token受限条件下无法带来计算收益

**🔧 技术方法**

使用线性计数回归、冻结文本嵌入、任务条件排名器以及Fine‑tuned Qwen3‑0.6B监视器进行预测

**📊 数据集**

使用147,002条来自Nebius SWE‑rebench与NVIDIA Open‑SWE‑Traces四种模型‑框架组合的公开执行轨迹

**📈 对比分析**

将学习到的停止阈值策略与匹配随机重启以及不间断继续在75k/100k token预算下对比，发现学习策略仅在两种Qwen3.5设置下略高于随机重启，但整体仍低于持续运行，提升幅度极小

**⚠️ 局限性**

主要局限在于缺乏跨任务尝试级信号，且实验仅考察冷重启而未评估状态保持恢复等更高效的中断方法

---

## 5. EMMA: Structured Episodic Memory for Multi-Agent Workflow Synthesis

**arXiv ID:** 2610.35966 | [PDF](https://arxiv.org/pdf/2610.35966v1)

**作者:** Ruijia Niu `[一作]` (University of California San Diego), Kristina Klinkner `[通讯]` (Amazon)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EMMA，一种将多代理系统配置视为情节记忆的结构化情节记忆框架，能够在单一受控编辑中改进多代理工作流程。

**💡 创新点**

创新点在于通过诊断失败范围并仅在该范围内执行受限、受护的编辑，保证更新一致、增量且不受基础模型结构偏好影响，并采用三层记忆层次与守护机制实现可扩展的多代理学习。

**🔧 技术方法**

采用 LLM 控制器、子代理与反射器；三层记忆池（控制器记忆、代理池、技能池）；局部回溯与溢出拆分；守护检查（作用域兼容、提示保护、内存边界、泄漏防止、策略变化）和一次性编辑；与 ACE、GEPA、EvoMAS、ADAS 等方法比较。

**📊 数据集**

使用 KramaBench、AppWorld（normal & challenge）以及 BBEH-mini 三个基准数据集进行实验。

**📈 对比分析**

在 Opus5、Sonnet5、GLM-5、DeepSeek v3.2、Kimi-K2.5 五种主干模型上与四个基线对比，EMMA 在所有基准上均取得最高平均性能（如 KramaBench 提升 7.0 分、BBEH-mini 提升 5.8 分、AppWorld TGC/SGC 提升 11.9/17.4 分），并在 17/20 设置中排名第一，同时训练执行次数比最优基线低 17–38%。

**⚠️ 局限性**

限制在于仍存在与主干模型间的性能差距；未解决的失败链可能留下无效或有害条目；需要跨批次或多轮验证以识别此类条目；以及对弱执行器与强反射器匹配的进一步提升空间。

---

## 6. Multiple Leak Localization in Water Distribution Networks through Graph Neural Networks

**arXiv ID:** 2610.35972 | [PDF](https://arxiv.org/pdf/2610.35972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 7. Where Does My Call Go? Measuring Google Meet, Zoom, and Microsoft Teams on Starlink

**arXiv ID:** 2610.35874 | [PDF](https://arxiv.org/pdf/2610.35874v1)

**作者:** Kousar Malekinasab `[一作]` (University of Victoria), Jianping Pan `[通讯]` (University of Victoria)

**通讯引用:** 8610 | [OpenAlex ID](https://openalex.org/A5043097022)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在加拿大七个位置和6架搭载Starlink的客机上，使用基于浏览器的WebRTC客户端对Google Meet、Zoom和Microsoft Teams的服务端点选择和RTT进行了系统测量和对比。

**💡 创新点**

首次将三大主流RTC平台在LEO网络中的服务点选择行为进行横向对比，揭示了基于POPs的动态切换与平台自有路由策略之间的相互影响，并指出了服务点选择与延迟之间的因果关系。

**🔧 技术方法**

利用Selenium+自定义JS Hook自动化WebRTC会话，提取ICE候选对、RTT和候选类型；结合UDP/TCP路径探测、IP归属和地理定位，推断服务端点并映射到ASN和物理位置。

**📊 数据集**

构建了包含超过24小时机舱内测量、10小时多地站点测量以及7个加拿大POPs的多场景数据集，并公开发布在GitHub（https://github.com/kmalekinasab/RTCMeasurement）。

**📈 对比分析**

通过比较不同平台的服务点映射与RTT分布，发现Google Meet在多场景下实现最低RTT（<100 ms），Zoom处于中间，Microsoft Teams表现最差；同时对比分析了会话挂接、服务点稳定性与POPs切换的时序关系。

**⚠️ 局限性**

局限性包括：仅测量浏览器端实现（未覆盖原生客户端），仅针对Starlink网络，样本量受限于测试时间和航班数量，服务点推断可能因IP归属不准确导致误差，且在机舱内采用10分钟一次的短会话而非连续长会话。

---

## 8. NSB2: An Open-Source Modular Pipeline for Application-Network Co-Simulation

**arXiv ID:** 2610.35872 | [PDF](https://arxiv.org/pdf/2610.35872v1)

**作者:** Harikrishna Kuttivelil `[一作]` (University of California), Katia Obraczka `[通讯]` (University of California)

**通讯引用:** 9640 | [OpenAlex ID](https://openalex.org/A5067832138)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了NSB2，一个轻量、可扩展的网络共模拟中间件，支持应用程序与网络模拟器之间的低延迟通信。

**💡 创新点**

采用多通道解耦通信、内存键值缓存、抽象模块接口，兼容Python/C++，可灵活配置推拉模式与模拟器模式，提升了集成简易性与性能。

**🔧 技术方法**

基于TCP/Unix socket通信、Protocol Buffers消息定义、Redis内存键值存储、CMake+pkg-config构建系统，并提供ns-3与OMNeT++的集成模板。

**📊 数据集**

在性能评估中使用合成负载（10-500节点、10-1000消息/秒）进行压力测试，并未采用公开真实数据集。

**📈 对比分析**

通过60秒压力测试，测量延迟、CPU占用与内存，结果显示10-100节点延迟<10ms、CPU<76%、内存<7MB；500节点时延迟升至200ms、CPU≈77%，整体保持轻量化。

**⚠️ 局限性**

受限于单线程轮询与文件描述符上限，导致大规模节点（>500）性能下降；目前未支持多机部署与更高并发。

---

## 9. ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets

**arXiv ID:** 2610.35925 | [PDF](https://arxiv.org/pdf/2610.35925v1)

**作者:** Taehwan Lee `[一作]` `[通讯]` (Hanyang University), Taehwan Lee (Hanyang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于关系的上下文PII掩码框架ReCoPII，用于在不假设单一数据主体的前提下生成MASK/KEEP标注的语料。

**💡 创新点**

创新点在于：①利用领域本体约束的实体关系图规划，避免LLM随意生成不支持的关系；②通过占位符写入插入句子并自动重建注释；③使用独立LLM审阅验证文本支持，确保隐私标签的合法性；④从关系图中推导MASK/KEEP决策，实现可追溯的上下文掩码。

**🔧 技术方法**

技术包括：LLM（GPT‑4o‑mini、GPT‑5‑mini 等）进行文本生成与占位符填充；Python脚本进行占位符校验、注释重建；独立LLM进行结构与语义审核；关系条件与直接头的上下文掩码模型；以及结合检测器的联合微调。

**📊 数据集**

使用的语料为从 Basic 语料库（2,355篇韩国长文）筛选并改造得到的 1,857 篇 ReCoPII‑Ko 语料，包含 18 种实体类型、25,656 条关系及 MASK/KEEP 标注。对照使用 SPY‑style 生成的同源语料。

**📈 对比分析**

比较方法：①检测器在 KDPII 与 ReCoPII‑Ko 上的迁移性能；②在相同测试集上，用 ReCoPII‑Ko 训练的分类器与 SPY‑style 训练的分类器的 Macro‑F1、过掩码率与泄漏率；③完整掩码系统的联合微调在官方 404 文档测试和人工验证样本上的过掩码率/泄漏率。结果显示：ReCoPII‑Ko 训练的模型在 Macro‑F1 上提升约 2.5–3.8 分，过掩码率降低约 5.7–8.8 分；联合微调后过掩码率仅 2.5–5.0% 以内，泄漏率保持 ≤4.2%。

**⚠️ 局限性**

局限性：①仅在人工合成的韩国长文本上验证，缺乏自然文本和多语言评估；②模型评估多使用生成标签，未完全测量实际隐私决策正确性；③人工验证样本规模有限，且大多数可保留实体集中在插入句子，导致指标可能受句子来源影响。

---

## 10. What Has Actually Been Proved? Auditable Paper-to-Lean Alignment with LeanTrail

**arXiv ID:** 2610.35926 | [PDF](https://arxiv.org/pdf/2610.35926v1)

**作者:** Fabricio Venturim `[一作]` (Fundação Getulio Vargas), Jorge Poco `[通讯]` (Fundação Getulio Vargas)

**通讯引用:** 1556 | [OpenAlex ID](https://openalex.org/A5029317616)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 LeanTrail，一种后置审计系统，用于评估 Lean 形式化是否真正覆盖原始论文的定义、引理和定理，并将结果转化为可操作的任务列表。

**💡 创新点**

创新点在于将语义对应与证明完整性分离；采用确定性检索+LLM 语义判断+规则融合，生成五种覆盖状态；并提供可检视的接口和任务导出，真正以论文为规范衡量进度。

**🔧 技术方法**

技术包括 BM25 与逆序列融合的候选检索、两条检索路径（Paper→Lean 与 Paper→Blueprint）、LLM（Gemini 3.8 Flash、GPT‑6 Sol）进行语义对应判断、确定性传递性依赖审计以及规则级融合。

**📊 数据集**

数据集为三大 Lean 项目 FinEqs（7 条条目）、APAP（29 条）和 Ringel（62 条），共 98 条论文条目与对应 Lean 声明的手工标注基准。

**📈 对比分析**

通过对检索召回率、语义判断准确率和最终覆盖状态准确率进行评估；在三次独立跑中，覆盖状态正确率分别达到 91.8%（Gemini）和 92.5%（GPT‑6 Sol），并在 96–97 条状态上保持一致。

**⚠️ 局限性**

局限性包括仅评估三项目并缺乏多评标者一致性；语义对应仍依赖 LLM，存在过度或不足判断；证明审计仅关注 Lean 基础公理，未独立验证；假设论文结构化、提取无误。

---

## 11. h1: A Taste of the Bitter Lesson in Automated Theorem Proving

**arXiv ID:** 2610.35962 | [PDF](https://arxiv.org/pdf/2610.35962v1)

**作者:** Skander Moalla `[一作]` (FAIR at Meta), Julia Kempe `[通讯]` (New York University)

**通讯引用:** 11630 | [OpenAlex ID](https://openalex.org/A5051933322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一个简单的 Lean 4 交互循环（simple harness），让前沿 LLM 通过多轮推理、工具调用、验证与反思在单文件中完成形式化证明。

**💡 创新点**

创新点在于证明：1）仅靠测试时扩展的简单循环即可匹配或超过现有专门化工作流；2）更长的单次尝试往往比重复更高效；3）有效的接口、持续性与上下文管理是成本效益的关键。

**🔧 技术方法**

使用技术包括：前沿 LLM（Gemini 3/3.1 Pro、GPT‑5.6 Sol、DeepSeek V4 Flash 等）+ ReAct 交互模型 + Lean 4 工具（LeanExplore、Loogle、grep、REPL、LSP、SafeVerify）+ 反思 LLM + 预算/成本跟踪。

**📊 数据集**

使用的数据集包括 PutnamBench（672 题、100 题子集）、Putnam 2025、IMO 2026、FATE‑X、FormalQualBench、Formal Conjectures（FC100‑solved 与 236 未解决题）以及 5 个 Erdős 研究题。

**📈 对比分析**

在匹配模型与预算的条件下，分别与手工工作流系统（Goedel‑Architect、AlphaProof Nexus、Numina、MerLean 等）以及其他简单循环（AxProverBase、Humanfia、Claude Code）进行对比。结果显示 simple harness 在同模型、同预算下往往匹配或超越专业化系统，单次长尝试往往比多次重复更高效，且成本显著低于专门化流程。

**⚠️ 局限性**

局限性：仅适用于单文件 Lean 证明，无法处理多文件或需要数值计算、模拟等外部工具的任务；对模型黑盒性和训练数据曝光影响未知；在少数任务专门化流程仍有优势；缺少并行、多代理、代码库管理等扩展；工具集仅覆盖 Lean。

---

## 12. Enhancing Deep Learning Model Performance through Data Generation Strategies: Classification of Olive Tree Diseases

**arXiv ID:** 2610.35984 | [PDF](https://arxiv.org/pdf/2610.35984v1)

**作者:** Hicham El Akhal `[一作]` (Moulay Ismail University of Meknes), Abdelbaki El Belrhiti El Alaoui `[通讯]` (Moulay Ismail University of Meknes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

结合传统增强、CutMix、MixUp以及DCGAN对橄榄树叶病害图像进行数据生成，并在三种分布场景下训练EfficientNetV2B0模型进行分类。

**💡 创新点**

提出一种混合数据生成框架，在不同样本量与类别平衡场景下系统评估各生成方法的贡献，首次将多种生成技术与分布策略相结合。

**🔧 技术方法**

使用传统几何变换、CutMix、MixUp、Deep Convolutional Generative Adversarial Networks (DCGAN)及Fréchet Inception Distance 进行样本筛选，并采用EfficientNetV2B0作为分类器。

**📊 数据集**

基于公开的橄榄叶病害图像数据集（共七类，6170张），将数据分为训练集与独立测试集。

**📈 对比分析**

通过与无增强基线以及单一增强方法比较，采用5折交叉验证评估准确率、F1分数、召回率等指标；在S1、S2、S3三种场景下，混合增强可将准确率提升至约93%（S1）、95.99%（S2）和约92%（S3），显著优于基线。

**⚠️ 局限性**

依赖于数据集质量与平衡，生成模型（DCGAN）需要较高计算资源，且不同增强方法对不同场景的敏感度不同，过度合成可能导致信息冗余或噪声。

---

## 13. Data-driven Discovery of Lie Symmetries and First Integrals with Latent ODEs

**arXiv ID:** 2610.35967 | [PDF](https://arxiv.org/pdf/2610.35967v1)

**作者:** Matt L. Wiemman `[一作]`, Peter Melchior `[通讯]` (Princeton University)

**通讯引用:** 21490 | [OpenAlex ID](https://openalex.org/A5066501666)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 LatentLieODE，一种从时间序列（包括视频）自动学习低维表示、隐式 ODE 和线性 Lie 对称生成器的神经网络架构，并通过学习第一积分实现能量守恒。

**💡 创新点**

创新点在于：①可微分的 Lie 换位残差目标，直接在隐空间对生成器进行训练；②通过生成器范数谱的乘法间隙自动推断 Lie algebra 的维度；③联合优化第一积分网络并将其校准为物理量；④不预设对称维度，可发现非可积（如 SO(3)）的非阿贝尔对称。

**🔧 技术方法**

主要技术包括自编码器 + 递归网络、隐式 ODE、坐标去相关变换、圆形轨道和速度正则化、线性生成器银行、前向模式自动微分求 Lie 换位、谱间隙检测、第一积分 MLP。

**📊 数据集**

使用六个经典系统（谐振子、单摆/双摆、Lotka–Volterra、像素摆、球面测地线）以及对应的视频观测进行实验，验证了对称性和守恒量的重构。

**📈 对比分析**

与 LaLiGAN、Noether 网络、Hamiltonian/Lagrangian 神经网络比较，LatentLieODE 能够自动发现任意维度对称，恢复的守恒量相关性 ≥ 0.978；在球面测地线实验中成功识别 SO(3) 并验证 Killing 矩阵负定，显示出更高的对称性识别与物理一致性。

**⚠️ 局限性**

局限性包括：需要足够宽的隐空间才能线性化对称；对非线性不可线性化的对称缺乏搜索能力；对数据质量和时间分辨率敏感；第一积分仍需后期校准，无法保证绝对守恒。

---

## 14. ResCom: Activation Residual Compensation for Large Language Model Quantization

**arXiv ID:** 2610.35955 | [PDF](https://arxiv.org/pdf/2610.35955v1)

**作者:** Ning Yang `[一作]` (Chinese Academy of Sciences), Yuting Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 12927 | [OpenAlex ID](https://openalex.org/A5110722362)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ResCom，一种在已校准的低位量化大语言模型上使用激活量化残差的低秩校正分支；

**💡 创新点**

创新点在于利用激活量化残差来恢复被舍弃的输入信息，并通过闭式的正则化低秩回归一次性拟合校正分支，无需梯度训练或修改主模型权重；

**🔧 技术方法**

技术实现包括残差条件校正、正则化低秩回归、SVD求解低秩参数、以及与 SERQ 等量化流程的无缝集成；

**📊 数据集**

使用的数据集包括 WikiText‑2、C4、MMLU、GSM8K 以及 LongBench，用于评估语言建模、通用任务和长文本推理性能；

**📈 对比分析**

对比方法主要是 SERQ、LRC、ARHQ、PrefixQuant 和 FlatQuant 等 PTQ 流程；实验表明在 1B‑70B 模型上，W4A4 量化后 ResCom 可将 WikiText‑2 PPL 从 7.708 降至 7.147，MMLU 提升 2.89 点，五步 GSM8K 精度从 23.96% 提升至 36.16%；在跨模型、跨量化管线和跨语料库的转移实验中也表现出一致的提升；

**⚠️ 局限性**

局限性包括：仅在 4‑bit 权重与激活量化（W4A4）场景下验证，较高秩或更大模型会增加推理开销，需额外的校准数据，且对量化方案的兼容性仍有限；

---

## 15. SCOPE: Source Credibility and Outlet Profiling Engine

**arXiv ID:** 2610.35929 | [PDF](https://arxiv.org/pdf/2610.35929v1)

**作者:** Ilyas Nagiyev `[一作]` (Nazarbayev University), Preslav Nakov `[通讯]` (MBZUAI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了SCOPE多代理系统，用于对新闻媒体的偏见、事实性和可信度进行系统化、可解释的评估。

**💡 创新点**

创新点在于将MBFC方法拆解为专门化的LLM代理，整合抓取、内容分析、证据检索和报告生成，并通过可视化中间证据提升透明度与准确性。

**🔧 技术方法**

使用LLM（GPT‑5 mini）驱动的分工代理、DuckDuckGo检索、Tranco流量列表、RSF/Freedom House等公开数据，以及基于MBFC加权评分的统计模型。

**📊 数据集**

采用200个MBFC标注的新闻媒体样本，利用其偏见、事实性、可信度标签和自由文本作为评估参考。

**📈 对比分析**

通过与LLM‑only、文章‑only、搜索‑only及MBC检索增强基线对比，SCOPE在8项指标中赢得6项最佳，最低bias和factuality MAE，完美失败事实检查检测，平均成本约$0.05/站。

**⚠️ 局限性**

局限性包括仅在英语/美国媒体标签上评估，潜在搜索泄漏风险，抓取时可能丢失超链接信息，以及对极端/伪科学媒体的政治方向定位难度。

---

## 16. Packet Interval Modulation-based NFC for Scalable IoT around Meandered E-textile

**arXiv ID:** 2610.35870 | [PDF](https://arxiv.org/pdf/2610.35870v1)

**作者:** Shigemi Ishida `[一作]` (Future University Hakodate), Satoshi Ido `[通讯]` (Future University Hakodate)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于包间隔调制的异步NFC传感网络，解决多标签动态识别和碰撞问题。

**💡 创新点**

创新点在于用最小ID包作为脉冲，以时间间隔编码数据，消除时隙同步需求，显著降低碰撞率。

**🔧 技术方法**

采用了异步包间隔调制、424 kbps子载波、2字节极简帧、随机退避窗口（2^14时钟单元）以及双传输冗余技术。

**📊 数据集**

使用仿真生成的1–20个标签、10万次试验的随机时间戳数据，更新周期设为100 ms，构成实验数据集。

**📈 对比分析**

与传统16槽Aloha对比，碰撞率大幅降低，吞吐量提升约4倍，单标签吞吐量超过160 bps，在20标签时仍保持稳定。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，未在真实硬件上测试；受限于20标签规模、时钟漂移误差、能耗未充分评估。

---

## 17. Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering

**arXiv ID:** 2610.35982 | [PDF](https://arxiv.org/pdf/2610.35982v1)

**作者:** Elizaveta Zinovyeva `[一作]` (Amazon Web Services), Brian Jensen `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一款面向汽车 C/C++ 的编码助手，提供生成、编辑和可由开发者手动触发的验证工具，并通过模型上下文协议（MCP）集成到 IDE 中。

**💡 创新点**

创新点在于：① 将验证作为与生成同级的独立工具，让审核者能自行触发；② 通过提示工程与微调两种定制策略比较 LLModel 在汽车域的表现；③ 将需求文本、UML 行为和 API 上下文统一包装成训练/推理提示，提升领域适配。

**🔧 技术方法**

采用技术包括：MCP 交互框架、Prompt Engineering、LoRA 微调、Qwen3‑32B、Claude Sonnet 4、Amazon Nova Pro、基于 JSON 的规则集静态检查器、C/C++ 代码解析流水线。

**📊 数据集**

数据集来源于 AUMOVIO 的生产汽车代码：约 6,000 个函数，配合 DOORS 需求文本、PlantUML UML 行为以及 API 文档，划分为训练集（~5,178）与 822 条测试集。

**📈 对比分析**

比较方法：在 822 条测试函数上统计编码与架构规则违规次数并测量生成延迟；结果显示 Prompt Engineering 在 C 上违规率最低，Fine‑tune 在 C++ 上略优；与人类基线相比，LLM 生成的 C 代码违规率更低；生成延迟从 14 s（Nova Pro）到 110 s（Qwen PE）不等。

**⚠️ 局限性**

限制：仅评估模式违规，未检验编译、单元测试或功能正确性；使用 LoRA 微调而非全参数微调；未进行显著性检验；不同模型在不同推理平台上测量，难以归因；仅基于单一企业代码，结果缺乏泛化；人类实验基于自报、无遥测。

---

## 18. Calibrated for Which Policy? Conformal Uncertainty for Fixed World Models Under Intervention

**arXiv ID:** 2610.35877 | [PDF](https://arxiv.org/pdf/2610.35877v1)

**作者:** Ayush Mittal `[一作]` (Noumenal), Dhruv Gupta `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究如何在部署策略变化时校准固定世界模型的预测集，利用加权 conformal 预测和测度比率识别来恢复目标分布下的覆盖率。

**💡 创新点**

创新点是将政策干预导致的测度变化与加权 conformal 校准相结合，给出覆盖下界并证明比例误差可通过源域二进制 log 损失误差控制，从而实现学习权重近似 oracle 并在实际实验中恢复目标覆盖率。

**🔧 技术方法**

使用技术包括：加权 conformal 预测、政策控制动态下的测度比率识别、基于域分类器的比例估计（logit 形式）、二进制 log 损失回报与总变差的关系证明、有效样本量与半径分析，以及 Lean 4 形式化验证。

**📊 数据集**

实验数据集为 Franka 机械臂 192 次物理一跳转移，配合公开的 DINO-WM 预测模型与自定义的 λ‑tilt 分布（P_λ 与 Q_λ）进行评估。

**📈 对比分析**

与普通未加权 conformal、oracle 比率加权、学习比率加权（以及剪枝）对比；在 90% 置信下，未加权方法覆盖率从 55% 降至 81%+，而 oracle 与学习比率方法保持约 90%+ 覆盖，效果显著提升；有效样本量下降，半径略增。

**⚠️ 局限性**

限制：仅针对一跳、边际覆盖；要求共享条件动态、目标分布包含于源分布、可预估的预后测度；实验仅在有限转移池上，未覆盖多步或动态机制变化。

---

## 19. Generalization in Learning: Eyring Formula and Predator-prey Model

**arXiv ID:** 2610.35965 | [PDF](https://arxiv.org/pdf/2610.35965v1)

**作者:** S. V. Kozyrev `[一作]` (Steklov Mathematical Institute of Russian Academy of Sciences), Alexander Pechen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了利用捕食-猎物模型与热力学方法提升机器学习泛化能力，并给出噪声数据回归的数值实验。

**💡 创新点**

创新性地将GAN的极小极大优化映射为生物学捕食-猎物系统，并用Eyring速率公式解释过拟合与grokking现象。

**🔧 技术方法**

使用随机梯度 Langevin 动力学、Fokker–Planck 方程、Eyring速率公式、SDE系统以及传统梯度下降与捕食-猎物迭代算法。

**📊 数据集**

采用含噪声的 sin(πx) 回归数据（训练集20点、测试集10点），并参考模运算等典型过拟合实验。

**📈 对比分析**

与普通梯度下降对比，捕食-猎物方法将测试集 MSE 从 0.161 降到 0.106，函数偏差从 0.0594 降到 0.0334，显示显著性能提升。

**⚠️ 局限性**

实验仅在单一回归任务上进行，需要大量迭代且对初始值、学习率等超参数敏感，缺乏理论收敛分析与更大规模验证。

---

## 20. From Slices to Volumes: Comparing 2D, 2.5D, and 3D Residual Networks for Pulmonary Nodule Malignancy-Risk Classification

**arXiv ID:** 2610.35960 | [PDF](https://arxiv.org/pdf/2610.35960v1)

**作者:** Bahodir Nazarov `[一作]` (Sichuan University), Fatima Zazoul `[通讯]` (Sichuan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

比较了2D、2.5D和3D残差网络在肺结节恶性风险分类中的表现，并评估了自监督Momentum‑Contrast预训练、轻度数据增强以及去除不确定等级对模型性能的影响。

**💡 创新点**

在统一的LIDC‑IDRI人群分割下系统性比较三种空间表示和三种训练策略，首次量化轻度增强的正面效应与MoCo预训练的负面结果。

**🔧 技术方法**

使用ResNet‑18（2D、5通道2.5D和3D版本）、Momentum‑Contrast自监督预训练，以及随机翻转、旋转、强度扰动等轻度数据增强。

**📊 数据集**

采用LIDC‑IDRI胸部CT数据，包含873个良性、1,226个不确定、555个恶性结节，共2,654个标注结节。

**📈 对比分析**

通过患者级无重叠的训练/验证/测试拆分，使用Macro‑F1、准确率、平衡准确率和AUC评估；增强的2.5D模型在三分类任务中取得最高Macro‑F1 0.6537，3D模型表现最差；去除不确定等级后二分类Macro‑F1最高0.8664。

**⚠️ 局限性**

预处理、优化、参数量和复现次数不一致，导致难以单独归因于空间维度；缺乏外部验证；仅进行一次复现，未完全估计患者内结节相关性；标签为放射科医生主观评分，非病理确诊。

---

## 21. Structured Time-Frequency Representation for Channel Prediction

**arXiv ID:** 2610.35956 | [PDF](https://arxiv.org/pdf/2610.35956v1)

**作者:** Ning Yang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 25093 | [OpenAlex ID](https://openalex.org/A5100458465)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于结构化时频表示的FDD信道预测方法FAR-CSI，利用预训练的CSI Transformer完成历史上行信道到未来下行信道的跨频预测。

**💡 创新点**

创新点在于设计FDD-aware的结构化路由规则，将历史UL与未来DL的时频位置统一映射到同一时频字段，保持观测与预测目标在同一坐标空间中的一致性。

**🔧 技术方法**

采用预训练的HeterCSI Transformer作为骨干网络，结合patch编码、位置嵌入、mask式解码以及RoS损失实现端到端的信道重建。

**📊 数据集**

使用LLM4CP提供的3GPP UMa NLOS仿真数据集（QuaDRiGa生成，包含9000个训练/验证样本，10000个测试样本，速度10~100 km/h）。

**📈 对比分析**

与RNN、GRU、LSTM、CNN、Transformer、LLM4CP、WiFo以及Generic CSI Transformer等基线在NMSE和MRT基带吞吐量上进行对比；FAR-CSI在NMSE上最低（≈0.15），相较于通用路由下降21.44%，在不同速度、预测步长、数据量、噪声条件下均保持优势。

**⚠️ 局限性**

局限性包括：仅在单用户单频段的3GPP UMa NLOS场景验证，需依赖预训练HeterCSI模型，且对更广泛频带差距、多用户场景或更复杂环境的泛化尚未评估。

---

## 22. A Hyper-Logical View on Multi-Agent Path Planning

**arXiv ID:** 2610.35964 | [PDF](https://arxiv.org/pdf/2610.35964v1)

**作者:** Hadar Frenkel `[一作]` (Bar-Ilan University), Dor Atzmon `[通讯]` (Bar-Ilan University)

**通讯引用:** 1247 | [OpenAlex ID](https://openalex.org/A5025220652)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

本文将多代理路径规划（MAPP）的九个变体（包括 MAPF、AMAPF、MCPF、TAPF、MAPD、MAPF‑PC、RMAPF、MTPF、MAM）分别建模为 HyperLTL/HyperQPTL 公式，并给出了模块化的翻译模板；同时展示了如何在这些公式中表述跨代理约束、代价函数（如 makespan）以及最优性条件；最后证明可通过 HyperQPTL 模型检查判定是否存在最优解。

**💡 创新点**

创新点在于：①使用超属性逻辑统一描述多代理路径规划问题，摆脱传统逐个代理规划的束缚；②提供可插拔的公式片段，便于在不同变体之间快速切换或添加新约束；③将代价最优性（如 makespan、SOC）抽象为超属性，从而实现与现有超属性验证工具的直接对接。

**🔧 技术方法**

核心技术：HyperLTL 与其扩展 HyperQPTL（带原子命题量化）、无穷字词语义、LTS 转换、量化原子命题来标记关键时刻；使用路径量化和逻辑运算实现约束；将最优性写成 ∃π∀π' 语句；最后构造可给定 MAPP 变体的 HyperQPTL 公式来进行模型检查。

**📊 数据集**

本工作为理论性研究，并未使用任何实验数据集；所有验证均基于形式化证明与逻辑公式构造。

**📈 对比分析**

本文未进行实验比较；作者指出现有 HyperLTL/HyperQPTL 检查器在规模上与成熟的 MAPP 求解器相距甚远，难以在标准基准上竞争；因此没有提供性能指标。

**⚠️ 局限性**

限制：①模型检查的复杂度随量化层级指数增长，当前工具效率低；②仅覆盖一次性（one‑shot）MAPP 变体，未扩展到在线或持续（lifelong）情形；③未给出实际求解性能，缺乏经验评估；④对更大规模问题的可扩展性尚未验证。

---

## 23. Frankly Speaking: The Company Election Integrity Keeps in Congressional Communication

**arXiv ID:** 2610.36187 | [PDF](https://arxiv.org/pdf/2610.36187v1)

**作者:** Michael A. Robinette `[一作]` (University of Vermont), Juniper Lovato `[通讯]` (University of Vermont)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5073688320)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2010–2025年间美国国会成员发出的电子简报进行文本挖掘，量化并对比两党在选举完整性相关词汇的使用方式，揭示词汇相同但语义背景不同的现象。

**💡 创新点**

创新点在于：①使用语义邻域（词向量最近邻）而非传统词频或情感词典来捕捉政治概念的多维演变；②将跨党语义距离与内党语义漂移结合，发现两党在相同词汇上构建截然不同的概念框架；③用可视化的语义网络展示不同党派的框架差异。

**🔧 技术方法**

技术手段包括：Word2Vec 词向量训练（skip-gram，50维）、按年份和党派分别训练、使用 Procrustes 旋转对齐词向量空间、计算词向量余弦距离衡量跨党与内党差异、Jaccard 相似度评估语义邻域重叠、语料预处理（分词、去停用词、统计共现词）、词典筛选与阈值过滤。

**📊 数据集**

数据集为 DCInbox 的国会电子简报语料库，共计 10,420 篇 2010–2025 年间的简报（约 52.4% 共和党，47.6% 民主党），通过三层过滤规则得到包含选举完整性相关词汇的文档。

**📈 对比分析**

比较方法：①跨党语义距离（两党词向量余弦距离）与同频率词的噪声基线对照；②内党语义漂移（连续年份词向量距离）与基线相似度；③语义邻域 Jaccard 重叠度。结果显示：两党在词频和内党漂移上相近，但在语义邻域上差异显著（Jaccard 低于同频词平均值），表明两党对同一词汇构建了不同的概念网络。

**⚠️ 局限性**

限制包括：①词向量对稀疏词的稳定性有限，低频词可能产生误差；②仅分析电子简报，忽略其他传播渠道；③未测量选民对信息的实际解读；④未进行参数敏感性分析；⑤排除了独立党派和部分非正式渠道，可能低估极化程度。

---

## 24. Before the First Edit? Measuring When Coding-Agent Guidance Actually Arrives

**arXiv ID:** 2610.35992 | [PDF](https://arxiv.org/pdf/2610.35992v1)

**作者:** Thupakula Sai Varun `[一作]` `[通讯]`, Thupakula Sai Varun

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造事件边界审计，验证了编码代理在首次编辑前是否真正收到规划建议，并对公共轨迹（PAUSE、SI2CA）和本地实验（Qwen2.5-Coder）进行了时序与覆盖率分析。

**💡 创新点**

创新点在于：①提出可复现的“门控-触发-规划-源变更”四步审计框架；②利用文件哈希在每一步捕捉实际源文件变化；③在同一任务下对多种门控策略（仅编辑器、加字面标记、可写能力、全请求）进行对比；④结合公开轨迹与本地实验验证门控规则在真实环境中的有效性。

**🔧 技术方法**

采用的技术包括：Python 结构化工具调用（replace_text、write_file 等）、Shell 语句识别、文件哈希对比、日志化工具调用顺序、规划模型（Qwen2.5-Coder 7B/14B）与本地脚本重放；并通过脚本实现可重复的“预执行可行性检查”和“后验重放验证”。

**📊 数据集**

使用的数据集包括：PAUSE 与 SI2CA 的公开轨迹（约 1200+ 条轨迹）、SWE‑bench Lite 的公共问题集（23 条开发集 + 30 条测试集）、自定义的 39 条构造动作单元、以及 76 条 14B 小补丁实验集。所有数据均按公开或内部许可可复现。

**📈 对比分析**

比较方法：统计各门控策略在构造动作、开发集和测试集上是否在首次源文件变更前收到规划（coverage）、首次编辑前的规划数量（delivery）、以及计划是否真正触发编辑。性能表现为：在所有实验中，计划均在首次变更前到达，但覆盖率仅在少数任务上达到目标（例如 7B 小补丁实验仅 15/76 变更，未达 20-arm 目标），且多数任务未出现任何源文件变更；公共轨迹缺乏同步文件快照，导致无法精确定位首次变更。

**⚠️ 局限性**

局限性包括：①公共轨迹缺少同步文件快照，无法精确判定首次变更时间；②实验只覆盖 Python 文件，未考虑多语言或非源码文件；③未进行正式的修复成功评测（仅做局部语法/测试检查）；④门控规则与实际执行路径可能不一致，导致覆盖率低；⑤样本量有限，难以推广到更广泛的任务或模型；⑥缺乏独立的人为注释或多方验证。

---

## 25. Task-Utility IQA: Rethinking Image Quality Assessment in Embodied Tasks

**arXiv ID:** 2610.35905 | [PDF](https://arxiv.org/pdf/2610.35905v1)

**作者:** Jirong Zha `[一作]` (Tsinghua University), Xinlei Chen `[通讯]` (Tsinghua University)

**通讯引用:** 18096 | [OpenAlex ID](https://openalex.org/A5016095713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于多模态大语言模型的具身图像质量评估（IQA）代理，将IQ评估从无任务关联的分数回归转变为目标条件下的判断，并用任务效用来评估图像质量。

**💡 创新点**

创新点在于：①首次将任务效用作为IQ评估目标，突破传统任务无关分数的局限；②构建可解释、目标驱动的多模态IQA代理框架；③在UAV视角的VisDrone数据上提供首个任务条件效用评估的实证研究；④提出基于任务上下文的评估协议和指标。

**🔧 技术方法**

采用多模态大语言模型（如GPT‑4V/CLIP‑IQA）结合检索增强生成（RAG）和指令微调；构造任务‑上下文条件的效用分数（TCUS）并与视觉语言模型（VLM）进行交互推理。

**📊 数据集**

使用VisDrone2019‑DET（UAV检测数据）转化为VQA‑style图像–问题对，包含正/负存在与计数任务；同时使用16款主流VLM（Qwen、Gemini、Claude、Grok等）作为下游任务评估者。

**📈 对比分析**

比较方法：将TCUS与传统IQA指标（PSNR、SSIM、Q‑SiT、Q‑Insight、DeQA‑Score‑Mix3）以及认知一致性（cognition‑consistency）对齐，使用AUROC、AP、Spearman相关系数评估其预测下游任务成功的能力。实验表明传统指标接近随机，认知一致性虽提升但仍不足；TCUS在所有VLM上实现AUROC 0.925、AP 0.948、Spearman 0.710，明显优于现有基线。

**⚠️ 局限性**

局限性：①仅在UAV视角的VQA任务上做了紧凑实验，未覆盖所有具身场景、闭环决策和人机交互；②TCUS是经验性代理，缺乏统一的可重复度；③大语言模型易产生幻觉、解释失误且计算成本高，尚未验证在边缘设备或实时部署中的可行性；④需进一步扩展到更多传感器、任务与环境，验证其通用性。

---

## 26. Condition-Enhanced Generative Adversarial Network for Aerodynamic Configuration Design of Commercial Aircraft

**arXiv ID:** 2610.35963 | [PDF](https://arxiv.org/pdf/2610.35963v1)

**作者:** Youzheng Yu `[一作]` (Northwestern Polytechnical University), Weiwei Zhang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 8211 | [OpenAlex ID](https://openalex.org/A5003289067)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了一种条件增强生成对抗网络（CEGAN），用于在给定几何与气动性能约束下快速生成可行的三维飞机配置。

**💡 创新点**

创新点在于把几何约束与气动性能约束分别嵌入生成器的损失函数，并通过训练后的代理模型在生成阶段做输出过滤，显著提升了在插值与外推条件下的生成精度。

**🔧 技术方法**

技术主要包括Wasserstein GAN + 梯度惩罚（CWGAN‑GP）、条件输入、几何误差与气动误差增强损失、独立的多层感知机代理模型以及基于阈值的输出过滤机制。

**📊 数据集**

使用了基于OpenVSP的Cessna Citation X小型通用机型的28维几何参数数据集，包含2500个采样点及对应的C_L、C_D、C_M等气动系数。

**📈 对比分析**

与基准CWGAN‑GP模型对比，CEGAN在插值情况下生成误差降低至少36%，在强外推条件下降低至少46%；同时保持了相当的几何多样性，且在相同数据量下可达到双倍样本量模型的性能。

**⚠️ 局限性**

局限性包括对代理模型预测精度高度依赖（R²≈0.95以上才显著受益）、低保真气动评估（VLM）导致的性能偏差，以及在极端外推情形下仍存在一定的多样性损失。

---

## 27. Knowledge distillation across teacher architectures for compact diabetic retinopathy grading

**arXiv ID:** 2610.35961 | [PDF](https://arxiv.org/pdf/2610.35961v1)

**作者:** Mossa Manisa `[一作]` (Shahjalal University of Science and Technology), A. K. M. Fakhrul Hossain `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在APTOS 2019数据集上通过离线响应式知识蒸馏，将ResNet‑50、MobileNetV3‑Large和Swin‑T三种教师网络的知识迁移到MobileNetV3‑Small学生网络，实现五级糖尿病视网膜病变分级；

**💡 创新点**

创新点在于系统性比较CNN与Transformer教师在同一学生上的蒸馏效果，探讨温度与硬标签权重的交互影响，并在资源受限环境下验证压缩模型的可部署性；

**🔧 技术方法**

使用技术包括离线知识蒸馏（softmax温度化+硬标签加权）、多层次数据增强、类不平衡加权、四种教师-学生组合的网格实验，以及多随机种子验证；

**📊 数据集**

数据集为公开的APTOS 2019 Blindness Detection 3,662张视网膜荧光摄影图像，按原始80/10/10训练/验证/测试划分；

**📈 对比分析**

比较方法以二次加权kappa(QWK)为主，学生在不同教师、温度、α配置下的QWK最高为0.9093（单种子），三种子平均为0.9024，显著优于未蒸馏Baseline 0.8804；模型参数量缩减18×，尺寸5.94 MB；

**⚠️ 局限性**

局限性包括仅基于单一数据集与预划分，缺乏外部验证；大多数配置仅单种子实验，随机种子波动影响结果；教师对比单点且缺乏多模型比较；模型在CPU/边缘设备上的实际推理效率未测。

---

## 28. Native Rectangular Linear Assignment with Implicit Rejection: A Jonker-Volgenant Solver without Square Expansion

**arXiv ID:** 2610.35948 | [PDF](https://arxiv.org/pdf/2610.35948v1)

**作者:** Ratha Siv `[一作]` `[通讯]`, Ratha Siv

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一个基于 Jonker–Volgenant 的线性分配求解器，支持隐式拒绝、无平方矩阵扩展、批量求解，并提供 Rust、Rust/C++ 混合以及纯 C++ 三个实现；同时给出完整的接口、性能基准和内存分析。

**💡 创新点**

创新点在于：①将统一拒绝惩罚转换为一侧形式，并用一个常数规模的拒绝终端（单距离+单父行）代替显式的假行/列；②在保持原始成本矩阵的同时完成求解，避免了昂贵的矩阵扩展；③给出完整的原始-对偶最优性判据；④在同一算法框架下实现三种语言版本，并通过 Python 接口统一调度。

**🔧 技术方法**

主要技术包括 Jonker–Volgenant 最短增广路算法、矩形分配与可选分配理论、Rust 与 C++ 的 FFI 接口、Python 绑定、批量求解的线程池调度、原始-对偶证书验证、以及对不同限制（阈值、禁止边）下的求解策略。

**📊 数据集**

使用大量合成数据集：不同形状（正方形、宽矩形、极端长宽比）、不同规模（从 4×5 到 5000×5000）、不同成本分布（均匀 [0,1)、带符号 [-1,1)、稀疏禁止边）、结构化案例（拥堵块、等价成本、量化、签名成本）、以及多种限制阈值（无阈值、固定、接近 0、负值）。

**📈 对比分析**

通过与三种展开形式（方阵填充、带假列、capped SciPy）以及 SciPy 原生矩形实现的基准进行对比；采用单矩阵、批量、规模扩展、极端形状等多维度实验；测量微秒级运行时间、毫秒级批量延迟和峰值内存；验证结果与枚举求解器完全一致。结果显示：JVRUST 在大多数单矩阵和有限阈值场景下跑得最快；批量调用能显著降低延迟；在拥堵块或极端形状时 SciPy 有时更快；内存占用显著低于方阵扩展版本。

**⚠️ 局限性**

局限性：实验仅在一台 Windows/Intel i9 主机上完成，未控制编译器/线程库差异；仅使用合成数据，未覆盖真实追踪/匹配应用；Python 版本、线程数、后台活动等对性能影响未被完全隔离；内存测量仅为一次/多次单次实验，未给出精确分配统计；仅考察 float64 精度，未系统评估大动态范围、子正常数或数值不稳定；未与所有 LSAPE 或可选分配专用实现做细粒度对比；对超限条件下的最优性、数值收敛性未做完整理论证明。

---

## 29. Counting the Bandages: On measuring the time that AI saves

**arXiv ID:** 2610.35942 | [PDF](https://arxiv.org/pdf/2610.35942v1)

**作者:** Solomon Shalom Lijo `[一作]` `[通讯]` (Independent Researcher), Solomon Shalom Lijo (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究AI助手在工作中所宣称的“节省时间”概念，提出“账本视角”，并对时间节省与逃逸缺陷的关系进行无模型量化，重现并重新分析公开实验数据，最后设计并执行了AI审稿人检查AI草稿的实证实验。

**💡 创新点**

创新点包括：①引入无模型的时间节省和逃逸缺陷两项指标以及其决策阈值“盈亏缺陷成本”；②通过搜索理论证明检查决策对质量与速度的影响；③对前沿统计偏差和识别问题进行理论与实证分析；④提供完整的报告清单；⑤在实验中验证过度信任导致检查时间缩短但错误率上升的预测。

**🔧 技术方法**

使用的技术主要包括：统计计量方法（PPML、分层引导、Jackknife‑t、Makarov bounds）、搜索理论最优检索、实验设计与预注册、Python/Claude模型仿真与数值求解，以及基于脚本的重现与单元测试。

**📊 数据集**

使用的数据集包括：METR公开的246个问题开发者数据；GDPval实验数据；SEC 10‑Q财报抽样数据用于AI审稿人实验；以及通过脚本重现的公开数据。

**📈 对比分析**

比较方法以平均时间比率（ratio of means）为主，结合错误率进行决策阈值分析；在METR实验中，工具在某些任务上导致工时增加19%–24%；在AI审稿人实验中，过度信任导致检查时间减少但逃逸错误率显著升高。

**⚠️ 局限性**

局限性包括：模型假设简化（单一缺陷、指数检索、无误报），缺乏人类参与者数据，缺陷后续成本未知，实验仅针对单一AI审稿模型，无法涵盖更广泛的实际工作场景。

---

## 30. Real-Time Smart Retail Analytics through Fog Computing using the RetailLightNet Lightweight Model

**arXiv ID:** 2610.35871 | [PDF](https://arxiv.org/pdf/2610.35871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 31. Testing Agentic Service Systems: An Agentic Living Lab Methodology Across the Physical-Digital Continuum

**arXiv ID:** 2610.35978 | [PDF](https://arxiv.org/pdf/2610.35978v1)

**作者:** Zhongjie Wang `[一作]` (Harbin Institute of Technology), Hanchuan Xu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5069247819)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Agentic Living Lab 测试方法论，用于评估代理服务系统的任务完成、关系履约和人机交互等多维度表现。

**💡 创新点**

创新点在于：①将服务情节（episode）作为首要测试单元；②引入人类角色实现（Real Human、Trace Replay、Rule Surrogate、Agent Proxy）与现实子strate（Digital、Twin、FMU、HIL、Physical）并配合语义 Delta 公开；③构造多判决向量（O、A、C、R、E、H、V）而非单一成功率；④设计跨平面参考测试 harness 与规范化交换 schema；⑤通过构造案例演示代理与子strate敏感性。

**🔧 技术方法**

使用规则引擎、代理生成器、数字孪生、FMU/HIL 接口、模拟用户与真实人类桥接等技术；并实现参考测试 harness 架构与规范化 JSON schema。

**📊 数据集**

主要使用构造的旅行与工业案例（无公开数据集），通过实验设计演示代理行为、子strate迁移、代理敏感性等情景。

**📈 对比分析**

通过对照实验（相同输出不同完成度、代理禁用/启用合同 guard、子strate从 BusinessRules→FMU、代理版本迭代）展示方法能够识别关系失败、子strate影响及代理改进；未给出量化性能指标，只说明可测量多维指标。

**⚠️ 局限性**

局限：未进行大规模实证验证，代理质量、代理代理有效性、跨子strate稳定性等仍待经验评估；方法依赖人工定义的协议、协议版本和人工参与；物理子strate与软件子strate之间的模拟精度不确定。

---

## 32. Small agents should know what their tools do

**arXiv ID:** 2610.35981 | [PDF](https://arxiv.org/pdf/2610.35981v1)

**作者:** Alexander Kovalenko `[一作]` (Czech Technical University in Prague), Artem Redchych `[通讯]` (Czech Technical University in Prague)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EffectWorld，一个带有效果类型工具的可执行工具世界，支持效用安全验证、追踪优化和负样本生成，并用LoRA适配器对小模型进行监督微调；

**💡 创新点**

创新点包括：①将工具的效应（pure/read/write/destructive）和idempotence作为声明，使效用安全可通过回放验证；②通过编译器式优化器生成高效且可验证的负样本；③系统性评估了正样本与负样本训练的对比，揭示公开工具数据对拒绝行为的隐式影响；

**🔧 技术方法**

技术手段包括：效果声明与回放验证、依赖图程序模式、编译式追踪优化、低秩LoRA适配器、UL、NPO、DPO等负样本训练目标；

**📊 数据集**

使用了EffectWorld生成的任务轨迹、公共工具使用语料（Nemotron-Agentic、Hermes、ToolACE）、BFCL、NESTful等基准数据；

**📈 对比分析**

通过与基线模型（未微调）以及不同负样本训练的对照实验，评估在三个数据集（分布内、hard、commerce）上的任务成功率、效用安全率、BFCL准确率等指标，结果表明SFT在所有拆分上提升了≈45–55个百分点；

**⚠️ 局限性**

局限性包括：工具库规模有限、脚本用户无拒绝/改变需求、优化器依赖启发式、未覆盖多域/长序列任务、只测试4B级小模型、负样本训练未显著优于SFT、以及缺乏完整的多轮交互评估。

---

## 33. IoT-based Smart Real-Time Energy-Aware Server Room Monitoring System

**arXiv ID:** 2610.35873 | [PDF](https://arxiv.org/pdf/2610.35873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 34. Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems

**arXiv ID:** 2610.35930 | [PDF](https://arxiv.org/pdf/2610.35930v1)

**作者:** Arnesh Batra `[一作]` (Indraprastha Institute of Information Technology), Phan Xuan Tan `[通讯]` (Shibaura Institute of Technology)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5073937678)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多智能体辩论（MAD）中同行沟通对单体决策和集体一致性的影响，提出了 MAD Bench 2400题基准，设计了与独立反思相匹配的对照实验，并通过内部干预（注意力屏蔽、残差交换）和统计分析量化信息流与依赖度。

**💡 创新点**

1) 首创 MAD Bench 兼具标注与非标注题，能区分同意与正确；2) 引入独立反思对照，清晰分离沟通效应与计算开销；3) 发现同行推理贡献占比约 70%，且源信任度（记录）对影响远大于地位；4) 证明混合架构（含循环注意）存在注意力外的影响通道。

**🔧 技术方法**

多智能体辩论协议、独立反思控制、决策读取、注意力屏蔽、残差流交换、Bootstrap 与 permutation 统计、AUROC 解码评估。

**📊 数据集**

MAD Bench（2400 文本问答题，分为知识推理、价值伦理与安全拒绝三大类），以及基准中使用的 GSM8K、MATH-500、MMLU、ETHICS、AdvBench 等公开数据集。

**📈 对比分析**

比较方法：在相同问题、相同计算量下对比辩论、独立反思和多数投票。结果显示：辩论相较于初始投票仅提升约 1% 准确率（Qwen 与 Gemma），在反思基线下提升不显著；但错误一致率显著上升 3.1–7.4%。有效独立投票数从 3 降至 ~1，表明辩论导致错误聚集。

**⚠️ 局限性**

局限：仅评估三款 8–10B 参数开放权重模型；仅使用单一英文文本基准和固定 3 轮（12 轮子集）协议；未覆盖更大或闭源模型、不同语言或更长交互；实验规模相对有限，结论可能不具普适性。

---

## 35. Deterministic Causal Quantization with Representative States and Bounded Error

**arXiv ID:** 2610.35933 | [PDF](https://arxiv.org/pdf/2610.35933v1)

**作者:** Rama Cont `[一作]` (University of Oxford), Massoud Heidari `[通讯]` (LeastBit Information Lab)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了带有路径误差上限的确定性有限容量量化方案，并提出了中点代表实现（MRR），实现误差≤ε并可同步更新。

**💡 创新点**

创新点在于构造可同步更新的MRR，实现误差保证且仅通过地址更新状态；给出无分布假设下的两比特（或一比特）熵率上界，以及利用Fisher信息得到的二次逼近；证明了网格初始化可实现最优过程ε-熵。

**🔧 技术方法**

采用解析与信息论方法，利用覆盖计数、四单元重叠定理、对数Sobolev不等式、Fisher信息、熵分解等技术。

**📊 数据集**

没有使用具体数据集，全部为理论分析与定理证明。

**📈 对比分析**

通过与过程ε-熵、标量ε-熵以及Shannon下界等基准比较，证明MRR的地址熵率在任何有限容量下至少与标量ε-熵相差≤2比特，网格初始化时≤1比特，且在Fisher信息有限时误差为O(ε²)。

**⚠️ 局限性**

局限性：结果是上界，非最优；需要ε‑分离初始化或有限Fisher信息；对高度相关或无界分布的适用性有限；并未讨论实际编码实现的延迟与码字长度。

---

## 36. PASS Is a Claim, Not an Observation: Mechanical Re-verification of 10 Self-Asserted Fix Claims in a Live Multi-Agent Audit Layer

**arXiv ID:** 2610.35950 | [PDF](https://arxiv.org/pdf/2610.35950v1)

**作者:** Lei Zhenbing `[一作]` (Independent Researcher), Lei Chang `[通讯]` (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多智能体系统中自报的代码修复声明进行机械化验证，评估其真实性与可靠性。

**💡 创新点**

首次系统化量化 PASS 声明的可信度，发现自报修复存在高失真率，并提出基于标识符+调用点的确定性检测准则。

**🔧 技术方法**

使用正则提取、文件时间戳校验、静态代码模式匹配及规则驱动的确定性重检等技术。

**📊 数据集**

利用2026‑08‑31至2026‑10‑12期间的 AITRAP 公共广播日志（179条，28条带修复声明）及对应的代码库快照。

**📈 对比分析**

通过三阶段判定（C1、C2、C3）对10条代码级修复声明进行评估，发现40%未通过文件位置检验，50%信息足够但仍未通过，说明现有自报机制可靠性不足。

**⚠️ 局限性**

样本仅为10条完整记录（普查），无法外推；验证由作者自行完成，缺乏独立第三方复核；仅检验静态代码，未覆盖运行时行为；无法区分代码错误与报告错误导致的失败。

---

## 37. Distributed LLM Inference via Communication-Efficient Speculative Decoding

**arXiv ID:** 2610.35957 | [PDF](https://arxiv.org/pdf/2610.35957v1)

**作者:** Joohyun Shin `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyuk Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2555 | [OpenAlex ID](https://openalex.org/A5062994489)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了DISCO框架，在设备端使用小模型生成草稿，服务器端利用复制的小模型本地完成拒绝采样，实现分布式投机解码。

**💡 创新点**

通过在服务器端引入草稿模型副本，消除了词表规模分布的通信开销，同时保持了目标模型分布的精确性，显著降低通信延迟。

**🔧 技术方法**

分布式投机解码、接受率采样、服务器端草稿模型复制、通信延迟分析、FP16推理技术。

**📊 数据集**

GSM8K、CNN/DailyMail、HumanEval三大公开数据集。

**📈 对比分析**

与传统的DSD和DSSD方法在不同带宽下进行对比，采用speedup指标；实验显示DISCO在低带宽（1‑5 Mbps）下保持约1.5×加速，整体性能超过基线，尤其在词表大时更具优势。

**⚠️ 局限性**

实验仅覆盖单请求场景，未考虑多用户并发；服务器端额外的草稿模型复制带来额外内存与计算成本；在极高RTT环境下，通信优势减弱。

---

## 38. A Graph-informed Kriging model for sample-efficient Bayesian optimization of wind farm layouts

**arXiv ID:** 2610.35975 | [PDF](https://arxiv.org/pdf/2610.35975v1)

**作者:** Zhen-Fan Wang `[一作]` (Shanghai Jiao Tong University), Onur Bilgen `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于图信息的Kriging模型（Graph-informed Kriging, GIK）用于风电场布局优化，并将其嵌入贝叶斯优化框架，以显著提升高维稀疏数据下的样本效率。

**💡 创新点**

创新点在于将风机对间相互作用通过二维相对坐标压缩为图结构的确定性趋势函数，减少对全场高阶训练数据的依赖，提升预测精度与优化效率。

**🔧 技术方法**

使用的技术包括图信息Kriging、贝叶斯优化、离散两机RANS CFD相互作用数据库构建、以及对比的普通Kriging、多精度Kriging和直接优化算法。

**📊 数据集**

使用的数据集涵盖NREL 5 MW参考机的离散网格布置、IEA Wind Task 37 81机布局、16机与48机实际布局，以及基于两机RANS的相互作用数据库。

**📈 对比分析**

与普通Kriging、泛型趋势Kriging、多精度Kriging以及SNOPT、DEBO、CMA-ES等直接优化方法比较，GIK在仅约200–280次高成本评估下即可达到或超过传统方法的AEP，并在48机CFD案例中实现96维优化，样本效率提升约2–3个数量级。

**⚠️ 局限性**

局限性包括无法充分捕捉多机耦合非对称效应、仅基于稳态RANS模型、需要离线两机数据库且在机型或工况变更时需重新构建。

---

## 39. StereoSplat: Metric-Scale Novel View Synthesis via Stereo-Grounded Gaussian Splatting

**arXiv ID:** 2610.35902 | [PDF](https://arxiv.org/pdf/2610.35902v1)

**作者:** Vladimir Yugay `[一作]` (University of Amsterdam), Diogo C. Luvizon `[通讯]` (Meta)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向双目视频的前向式 3D 高斯展开（3D Gaussian Splatting）框架，能够直接在已标定的双目相机上以度量尺度重建场景并进行新视角渲染。

**💡 创新点**

创新点包括：①利用基于双目匹配的基础模型（FoundationStereo）作为几何先验，保留度量尺度；②设计跨视图 Transformer 进行特征融合并使用门控深度细化；③提出基于逆深度和外观的可学习聚合机制，显著压缩每像素高斯集合；④构建大型合成双目视频数据集用于预训练。

**🔧 技术方法**

使用的技术主要有：双目立体匹配基础模型、Transformer 跨视图融合、门控深度残差、逆深度聚类和哈希聚合、可微 3D 高斯渲染。

**📊 数据集**

数据集包括：新构建的 137k 帧双目合成数据集（涵盖 9 个室内 + 4 个室外场景，多种基准和视角配置），以及 TartanAir、Infinigen、KITTI 等公开双目视频数据用于评估。

**📈 对比分析**

与 AnySplat、VicaSplat、DA3、MVSplat、PixelSplat、DepthSplat 等最先进方法对比，实验显示在 Infinigen、TartanAir v2 和 KITTI 上，本文方法在几何精度（AbsRel、δ1）和视觉质量（PSNR、SSIM、LPIPS）上均大幅超越对手，尤其在度量尺度的深度重建上几乎达到 100% 的 δ1 分数。

**⚠️ 局限性**

局限性包括：只能处理已标定的双目相机；假设场景静止，动态场景重建效果下降；推理速度受双目骨干网络限制；无法直接迁移到单目或非标定相机。

---

## 40. Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs

**arXiv ID:** 2610.35985 | [PDF](https://arxiv.org/pdf/2610.35985v1)

**作者:** Tianqi Chen `[一作]`, Jian Cheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型中的条件记忆层进行后训练量化，提出一种基于门控的残差梯度量化方法

**💡 创新点**

创新点在于：①把每个哈希行拆成1位残差梯级，允许按需加深精度；②利用每次检索的门控信号（门值可从1位码精确计算）来决定该检索是否读取更深层，从而实现按检索重要性分配位宽，而非传统的按访问频率或行级统计

**🔧 技术方法**

技术包括：1‑bit 残差量化（TCQ、INT1、PQ1等），门控门值估计，基于门值的阈值分层读取，批量阈值策略和vLLM集成

**📊 数据集**

数据集：PopQA（长尾实体问答）、TriviaQA、NQ-open、CMMLU；使用两大模型 DeepSeek‑V4.1‑Flash 与 Qwen3.8‑Flash‑Next 进行评测

**📈 对比分析**

与 FP8 原表、统一多层读取、按访问频率、按行预期门值和随机等基线相比，gate‑driven 方法在同等平均读取比特数下，在 PopQA 上平均提升 0.39–2.86 分，且在其他基准上 20–21 次显著优于其它规则；在 2.10 bits 读取下与 FP8 差距仅 0.2 分；同时显著减少表内存 7.8× 并提升 vLLM 吞吐量 11–31%

**⚠️ 局限性**

限制：仅为后训练方法，未涉及模型再训练；对门值的估计仍依赖 1‑bit 码，可能在极罕见实体或语境变化大时失效；评估仅涵盖两种特定模型与少数量化方案，未验证在更广泛条件记忆架构上的普适性

---

## 41. Exponential Wake-Up Overshoot in Explicit Gaussian Point-Process Filters

**arXiv ID:** 2610.35958 | [PDF](https://arxiv.org/pdf/2610.35958v1)

**作者:** Hanwu Qin `[一作]` (Zhejiang University), Dianhai Wang `[通讯]` (Zhejiang University)

**通讯引用:** 2978 | [OpenAlex ID](https://openalex.org/A5051889324)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究Explicit Gaussian point‑process filters在长静默后醒来时的指数性增幅与overshoot，并给出其确切分布、穿越指数及对隐式更新的对比分析。

**💡 创新点**

推导二阶静默扩展、醒来更新的完整分布、穿越指数预测overshoot时间，并证明隐式更新在固定bin宽度下风险仅对数增长但无统一上界；并在交易数据上进行实验验证。

**🔧 技术方法**

解析式递归求解、Lambert‑W函数、连续时间极限、指数尾概率、极限分析、Monte Carlo仿真、统计显著性检验和log‑score评估。

**📊 数据集**

合成长静默与恢复实验数据，以及Binance USD‑M perpetual合约交易数据（10个合约、20个交易日）。

**📈 对比分析**

与基准X（Bayes）及显式/隐式/可变限额滤波器比较，使用log‑score和excess loss评估；在所有73个穿越单元中隐式滤波器显著优于显式；在真实交易数据中隐式滤波器在大部分时间窗口获得更高log‑score，显式滤波器产生极大错误率。

**⚠️ 局限性**

隐式滤波器在宽bin或长静默下无统一风险上界；显式滤波器的overshoot导致极端损失；可变限额策略对隐式滤波器效果有限；实验受离散bin/连续时间差异和真实数据分布限制。

---

## 42. People's Views on Smiley Emoji on WeChat Platform, Positive Emotion and Sarcasm

**arXiv ID:** 2610.35908 | [PDF](https://arxiv.org/pdf/2610.35908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 43. Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation

**arXiv ID:** 2610.35910 | [PDF](https://arxiv.org/pdf/2610.35910v1)

**作者:** Soumyajit Datta `[一作]` (Rochester Institute of Technology), Ashiqur R. KhudaBukhsh `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5035198055)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过向 Gemini 与 OpenAI 视觉语言模型发出面部图像的族群与性取向/犯罪记录转化请求，探讨这些模型在生成图像时如何隐式编码受保护身份，并检验其生成图像与文本输出中的偏见传播；

**💡 创新点**

创新点在于揭示 VLM 在拒绝直接推断敏感属性时仍能通过图像转化重现并放大这些属性的视觉信号，展示非确定性拒绝行为、图像可分离性与跨模态偏见传递；

**🔧 技术方法**

主要技术包括：视觉语言模型（Gemini、OpenAI GPT‑4 Vision）在自然语言转化提示下的图像生成；基于 ResNet 与 ViT 的分类器进行可分离性测试；利用 Integrated Gradients 与面部解析实现区域重要性归因；文本生成任务评估图像到文本的偏见放大；

**📊 数据集**

使用三大 AI 生成人脸数据集：Hyperface、AIFace 与 Wood 等来源的 MSfake，样本总计 1,002 张；此外附加少量真实人脸验证；

**📈 对比分析**

方法上通过比较不同模型、数据集的拒绝率、合规率与分类准确率（如 gay vs straight 的可分离度 0.83‑0.88，犯罪 vs 非犯罪的可分离度 0.97+），表明生成图像在受控属性下显著可区分且偏见在文本输出中得以放大；

**⚠️ 局限性**

局限性包括：仅使用合成图像，难以直接推断对真实人脸的适用性；仅考察少数二元身份标签（族群、性取向、犯罪记录），未涵盖更细粒度或交叉身份；依赖辅助分类器与解释方法，可能未完全捕捉人类直观判断；

---

## 44. AgentBaton: Efficient Model Collaboration in Agentic Tasks

**arXiv ID:** 2610.35959 | [PDF](https://arxiv.org/pdf/2610.35959v1)

**作者:** Junlong Li `[一作]` (Hong Kong University of Science and Technology), Pavlo Molchanov `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在长周期代理任务中，通过一次模型交接（capable→efficient）实现能力与成本的权衡，并提出了一种轻量级在线交接检测器，动态决定何时把控制权移交给低成本模型；

**💡 创新点**

创新点在于：①系统性比较交接方向与上下文转移方式对准确率与成本的影响；②发现能够在能力模型已完成核心实现后将任务交给高效模型能在保持或提升准确率的同时显著降低成本；③设计并训练了仅用 2B 参数的检测器，可在线识别交接时机，提升整体效率；

**🔧 技术方法**

使用的大模型包括 GPT‑5.6 Sol、GLM‑5.3、Kimi K3、Claude Opus 5、Gemini 3.8 Flash、GPT‑5.6 Luna、Qwen 3.5 2B（检测器）、Qwen 3.8 27B（高效模型）；采用了基于日志文件与直接上下文的两种交接方式；训练时使用 2,000 任务的 SWE‑rebench V2 轨迹，并用 Codex 标注合适交接点；

**📊 数据集**

主要数据集为 DeepSWE v1.1 与 Terminal‑Bench 2.1（长周期软件工程与终端工作流任务），以及 SWE‑rebench V2（用于检测器训练数据）;

**📈 对比分析**

与单模型基线、查询级路由、每回合路由(MTRouter)、固定进度/固定回合交接等方法对比。检测器实现平均 47–86% 的 API 成本下降，同时在 12/22 组合中保持或提升准确率；在 DeepSWE 上，某些配置可提升 8.8pp 准确率；整体扩展了准确率‑成本 Pareto 前沿；

**⚠️ 局限性**

局限性包括：①仅考察一次交接，未处理多次交接或回退；②对不同任务或更复杂工作空间的泛化尚未验证；③交接时机检测依赖人工标注的模式，可能错过最佳切点；④交接后模型间状态不一致时的错误传播仍是挑战；

---

## 45. UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity

**arXiv ID:** 2610.35980 | [PDF](https://arxiv.org/pdf/2610.35980v1)

**作者:** Yijun Zhou `[一作]` (Unity China), Jie Feng `[通讯]` (Unity China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文未提供具体内容，因此无法确定做了什么。

**💡 创新点**

论文未提供具体内容，因此无法确定创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法确定使用了什么技术。

**📊 数据集**

论文未提供具体内容，因此无法确定使用了什么数据集。

**📈 对比分析**

论文未提供具体内容，因此无法确定比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法确定限制因素。

---

## 46. Layerwise Repacking: Variable-Length Hybrid-Attention Training without an Extra Redistribution Pass

**arXiv ID:** 2610.35989 | [PDF](https://arxiv.org/pdf/2610.35989v1)

**作者:** Jiahao Zhang `[一作]` (KlingAI Research), Chengru Song `[通讯]` (KlingAI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出层级拆分重排（Layerwise Repacking）方案：在全注意力层使用宽序列并行组（SP16/32），在稀疏注意力层切换到节点本地较小并行组（SP1/2/4/8），并将布局转换合并进已有的 Ulysses all-to-all 交换，避免额外的激活重分配；同时设计辅助父级求解器在保持父级拓扑不变的前提下重新分配样本，以提升子级平衡。

**💡 创新点**

创新点：① 在层级间动态切换并行度而不产生额外通信；② 利用现有的全注意力 Ulysses 交换完成样本重排，实现零额外通信开销；③ 通过辅助父级求解器根据全层和稀疏层的双重成本模型重新分配样本，从而缓解子级平衡瓶颈。

**🔧 技术方法**

使用技术包括：Ulysses 全序列-头部转置 all-to-all；混合注意力（1F3S hybrid‑attention）模型；稀疏注意力与 MoE 计算；序列并行（sequence parallelism）；自定义父子规划与重排；BF16 训练；8 GPU 节点的 64 GPU 集群。

**📊 数据集**

使用的数据集未在论文中显式说明，实验基于 1F3S 混合注意力模型的训练数据（长度可变的样本），在 64 GPU（8 节点）环境下进行 200 步训练，其中 22 步触发宽父级，评估其性能提升。

**📈 对比分析**

与基线（不使用重排、基线父级求解器）相比：重排单独使用时，E2E 时间下降约 5.95%（基线父级）或 4.17%（改进父级），MFU 提升 3.75% 或 2.78%；完整系统（重排 + 改进父级）E2E 时间下降 7.61%，MFU 提升 4.83%。性能提升主要体现在 20 个 SP16 触发步骤，部分 SP16/32 触发步骤因子级子集上限导致反效果。

**⚠️ 局限性**

局限性：① 子集上限 SP8 限制导致极长样本在 SP16/32 触发步骤无法充分受益；② 仅在 1F3S 混合注意力模型和 8 GPU 节点的 64 GPU 集群上评估，结果可能不易推广到更大/不同硬件；③ 需要对父级成本预测器进行校准，模型/集群变更后需重新调参；④ 对内存占用影响有限，但极长样本可能导致单机显存压力；⑤ 仅关注单步吞吐与 MFU，未系统评估训练收敛和精度。

---

## 47. Re-Ordering the Generalized Hofstadter Functions G, H, and Beyond

**arXiv ID:** 2610.35994 | [PDF](https://arxiv.org/pdf/2610.35994v1)

**作者:** Pierre Letouzey `[一作]` `[通讯]`, Pierre Letouzey

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了广义霍夫斯塔特函数族 (F_k) 的点序单调性，并给出严格单调性在 n>12(k+1)(k+6) 时成立的更短证明，同时验证了先前的若干猜想并在 Coq 证明助手中实现了完整形式化。

**💡 创新点**

创新点在于提出了新的迭代逆像公式 I_k^j(n)=n+∑_{i=k-j}^{k-1}F_k^i(n-1) 与 L_k^j(n)=n+∑_{i=k-j}^{k-1}F_k^i(n)（j≤k），利用该公式完成了直接证明并推导出严格排序与差值性质，首次在不使用无限形态词的情况下完成此类证明。

**🔧 技术方法**

采用了形式化证明技术（Coq）和递归/迭代分析、预像与后像的 Galois 连接等数学工具，辅以对三角数与数值系统的讨论。

**📊 数据集**

本文无传统意义上的数据集；所涉及的数值主要来自 OEIS 序列 A005206–A005376（F_k 的具体实例）及其理论推导。

**📈 对比分析**

通过在 Coq 中进行可验证的形式化证明来验证方法的正确性；相比之前的基于无限形态词的证明，新的方法更短、更直接，且在形式化工具中已通过机器检查。

**⚠️ 局限性**

局限性包括：证明只适用于 k≥1 的整数阶，严格单调性的阈值 N_k 为 12(k+1)(k+6) 仍是经验上得到的上界，且对更一般差值 d 的首次与末次出现位置仍缺乏闭式表达；此外，所用公式在 j>k 时需要更复杂的系数，导致进一步推广受限。

---

## 48. ObjectTwin: Measuring the Asset Correspondence Gap for Capture-Anchored Digital Twins

**arXiv ID:** 2610.35904 | [PDF](https://arxiv.org/pdf/2610.35904v1)

**作者:** Zhenghe Guo `[一作]` `[通讯]` (Zhejiang University), Zhenghe Guo (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在固定相机、尺度、姿态条件下，将生成网格替换为扫描网格的对比评估方法，衡量可观测一致性的损失。

**💡 创新点**

创新点在于将捕捉对应度定义为可观测替换估计量，提供可审计、可复现的测量协议，并揭示生成资产在保持真实扫描状态时的系统性缺陷。

**🔧 技术方法**

使用 RGB‑D 采集、投影投射、MaskIoU、覆盖率、L1 深度误差、保守姿态微调（Hybrid）以及 CAST‑orient 等技术实现评估。

**📊 数据集**

实验基于 Replica 与 ScanNet 两个室内 RGB‑D 数据集，共计 137 个实例与 668 个固定视角。

**📈 对比分析**

与扫描基准对比，生成资产平均损失 0.302–0.497 MaskIoU；在不同场景、域和两种生成器（Tripo 与 Hunyuan）上均显著；即使采用姿态细调也无法弥补此损失。

**⚠️ 局限性**

局限包括仅针对已采集的部分实例、缺乏完整物理几何与多次生成方差、未考虑碰撞/任务性能等更高层次的评估需求。

---

## 49. Curriculum-Driven Deep Reinforcement Learning for Autonomous Far-Range Rocket Landing

**arXiv ID:** 2610.35954 | [PDF](https://arxiv.org/pdf/2610.35954v1)

**作者:** Atin Chowdhury `[一作]` (Indian Institute of Information Technology), Piyush Joshi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套三阶段课程驱动的深度强化学习框架，用于实现可重复使用火箭的垂直着陆，包括远程导航、软着陆和完整任务的学习。

**💡 创新点**

创新点在于：①设计了三阶段课程并结合批判器预热与回顾机制防止灾难性遗忘；②使用完整的3×3旋转矩阵作为观测，显著减少网络参数；③采用多离散动作空间（18种组合动作）；④重新设计奖励函数，使每步奖励保持O(1)并包含进度与对齐项；⑤将起始水平距离扩展到1,000–1,500单位，显著提升任务难度。

**🔧 技术方法**

技术手段包括：PPO（Stable‑Baselines3实现），两层128单元的全连接网络，批判器warm‑up（100k步冻结actor）、回顾机制（混合早期阶段经验）、奖励重塑与对齐惩罚，29维观测向量的特征工程。

**📊 数据集**

使用自定义的6-DoF火箭动力学仿真器，随机初始化状态（水平距离1,000–1,500单位），不依赖真实数据集，最终在100,000条测试轨迹上评估。

**📈 对比分析**

评估方法为与Chowdhury等人基线的单阶段PPO在相同测试分布下对比。结果显示，本文策略在100,000条测试轨迹上成功率99.98%，远高于基线的68.31%；同时在最终距离、倾斜角、垂直/水平速度等指标上也均表现更佳。

**⚠️ 局限性**

局限性包括：①实验仅在确定性仿真器中进行，未考虑噪声、失效或外部扰动；②奖励函数仍需手动调参，缺乏系统的泛化评估；③仅验证PPO，未探讨其他RL算法的适用性；④未对不同随机种子进行鲁棒性分析；⑤缺乏真实硬件或更复杂环境的验证。

---

## 50. KORA: Deterministic-First Execution Control for AI Workloads

**arXiv ID:** 2610.35869 | [PDF](https://arxiv.org/pdf/2610.35869v1)

**作者:** Albert Heekwan Kim `[一作]` `[通讯]` (Krako Labs), Albert Heekwan Kim (Krako Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为AI系统引入了先行执行控制层，先通过确定性或结构化路由处理请求，必要时才调用模型

**💡 创新点**

将模型调用转化为显式决策，在推理前决定是否需要模型，提升可测性与可控性

**🔧 技术方法**

使用任务路径选择、确定性处理器、结构化检索、策略校验和遥测记录

**📊 数据集**

使用100任务的确定性重测基准以及12个合成客服查询

**📈 对比分析**

相较于传统模型优先的基线，KORA在基准中避免了80%的模型调用且没有确定性输出错误

**⚠️ 局限性**

仅在合成工作负载上验证，未覆盖真实生产流量、真实模型成本、延迟、能耗或广泛通用性

---

## 51. PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix

**arXiv ID:** 2610.35987 | [PDF](https://arxiv.org/pdf/2610.35987v1)

**作者:** Marco Paina `[一作]` (LogicStar), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11848 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基准，收集了20个开源仓库在18个月内维护者修复的2,457个bug，并评估十款前沿编码代理在整个仓库级别自动发现bug的表现。

**💡 创新点**

①首次公开基于维护者实际修复记录的仓库级别bug发现基准；②设计多通道分区审计 harness，大幅提升召回率；③系统分析代理与人类维护者发现bug类型与分布差异。

**🔧 技术方法**

利用LLM进行提交解析、bug提取、匹配与报告；采用规则+LLM双阶段匹配方法；实现多通道分区审计 harness 以提高覆盖度；评估多模型成本与性能。

**📊 数据集**

使用20个Python、JavaScript和TypeScript开源仓库在2025年初的快照，随后18个月内修复的2,457个bug；并通过模型预测估计未来5–10年内可能修复的总bug数。

**📈 对比分析**

通过recall、precision、F1以及API成本对十种模型进行比较；最佳召回率为20.8%，精度最高仅8.1%；更强模型和更高推理成本主要提升召回，精度基本不变；与单通道、最小化提示以及Hound等先前方法相比性能更佳。

**⚠️ 局限性**

参考集仅包含18个月内已修复的bug，未涵盖潜在重要的未修复bug；受限于内存安全语言，无法涵盖某些bug类型；评估聚焦于代码静态分析，缺少对执行时环境的检测；参考集可能偏向维护者优先级，存在样本偏差。

---

## 52. Learning Human Sports Motion Feasibility via Quantile Regression on Tracking Data

**arXiv ID:** 2610.36183 | [PDF](https://arxiv.org/pdf/2610.36183v1)

**作者:** Marc Gutiérrez-Pérez `[一作]` (Institut de Robòtica i Informàtica Industrial, Spanish National Research Council, Universitat Politècnica de Catalunya), Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial, Spanish National Research Council, Universitat Politècnica de Catalunya)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

学习运动员运动时可达性模型，使用量化回归预测从当前运动状态到任意位置的最短到达时间。

**💡 创新点**

将时间到达（TTA）视为低分位数回归，结合面积感知辅助损失，构建可训练的可达性边界，且在不同运动（足球、篮球、橄榄球）间实现迁移。

**🔧 技术方法**

多层感知机、低分位数（pinball）损失、面积正则化、基于朝向的几何变换、物理归一化特征、SUVAT噪声过滤、可达性凸包基线。

**📊 数据集**

FIFA 2022 世界杯 64 场 optical tracking、NBA 2015‑16 SportVU 631 场、NFL Big Data Bowl 2019 单场 tracking。

**📈 对比分析**

与六个物理运动模型和可达性凸包基线在 MAE、RMSE、pinball loss、匹配 99% 覆盖率下的可达面积进行对比；多层感知机在所有指标上均优于基线，且在相同覆盖率下可达面积最小。

**⚠️ 局限性**

未强制保证可达性几何属性，偶有非单连通或径向单调性违例；跨运动迁移受标签分布差异影响，导致可达面积不如本地物理模型；缺乏更细粒度的个性化和复杂场景建模。

---

## 53. 3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models

**arXiv ID:** 2610.36178 | [PDF](https://arxiv.org/pdf/2610.36178v1)

**作者:** Samar Ansari `[一作]` `[通讯]`, Samar Ansari

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用枚举方法直接计算语言模型在随机数生成任务中对每个整数的概率分布，消除采样误差，并在多种解码配置、提示方式及硬件栈下系统评估18条开源模型线的表现。

**💡 创新点**

创新点在于：①用枚举代替采样实现无误差概率分布；②分离模型分布与解码器影响；③在同一基线与指令微调模型对比中提供完整公开的分布数据集，揭示端点回避、偶数偏好、圆整数字回避等结构化偏差。

**🔧 技术方法**

核心技术包括：token级概率求和加停顿概率、温度/Top‑k/Top‑p重采样、双硬件栈（CUDA/Metal）复现、总变差距离与熵缺失等统计量计算。

**📊 数据集**

数据集为18条开源模型（如Gemma、Qwen、OLMo、Llama、SmolLM、Phi等）在整数范围 {5,10,20,30,100} 上的完整概率分布；并未使用外部自定义数据集，仅利用模型自身的生成分布。

**📈 对比分析**

对比方法以总变差距离与均匀分布衡量偏差；结果显示基线模型相对均匀，指令微调后分布显著集中，端点几乎不出现，偶数偏好在 N=100 时显著；双硬件栈差异仅在极少数置信度阈值下出现，整体性能稳定。

**⚠️ 局限性**

限制包括：仅评估 ≤8B 参数的开源模型；缺乏闭源或更大模型的验证；仅验证一组英文提示且未覆盖多语言；未对非标准拼写数字计量；只做一次采样验证；量化格式固定；未报告模型规模效应；统计仅描述性；基线与微调使用不同提示格式；小范围数集不足；质数与 3 倍数偏好未分离。

---

## 54. Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule

**arXiv ID:** 2610.36169 | [PDF](https://arxiv.org/pdf/2610.36169v1)

**作者:** Minh Pham Van `[一作]` (Posts and Telecommunications Institute of Technology), Cong Tran `[通讯]` (Posts and Telecommunications Institute of Technology)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5067894717)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于决策理论的冲突解决框架，利用上下文可靠性先验和对模型自身知识的校准证据，构建可根据部署环境动态权衡的检索增强生成系统。

**💡 创新点**

创新点在于证明固定规则是线性退化、对称散度只能检测冲突而不能决断，并引入可调先验与校准置信息实现真正的“最优”决策。

**🔧 技术方法**

核心技术包括知识笔记（model‑to‑context 事实标注）、两声读者（并行输出上下文与模型答案）、加权校准置信度策略、以及基于先验平移的选拔器；全部基于冻结大型语言模型与轻量级学习组件实现。

**📊 数据集**

主要使用 Grace 评测基准（基于 LC‑Quad2.0 与 Wikidata 的知识图问答子图），并在 ClashEval、Housing Statute QA 以及 -QT 等预注册外部测试集上进行验证。

**📈 对比分析**

在 Grace 上系统在 Qwen 与 Llama 两大 backbone 上分别取得 45.3 与 35.0 的准确率，明显优于所有固定规则基线（最高 29.4 / 27.5）并在 ClashEval 中比 CK‑PLUG、TCR 等同骨干控制方法多出 12–26 分；在模型已知但上下文不可靠场景中表现尤为突出。

**⚠️ 局限性**

主要限制是选拔器的校准依赖于训练时的上下文可靠性分布，面对新的问题类型或标签偏移时可能失效，需要重新训练；同时系统相较于单调用基线的计算成本略高。

---

## 55. From Structural Feedback to Prompt Policies: Learning Faithful Text-to-Image Prompt Editors

**arXiv ID:** 2610.36146 | [PDF](https://arxiv.org/pdf/2610.36146v1)

**作者:** Wentao Ye `[一作]` (Zhejiang University), Haobo Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11404 | [OpenAlex ID](https://openalex.org/A5100446035)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结构感知的提示编辑器，通过校准的场景图同时作为策略条件和图像基准监督，实现一次性生成可重用的优化提示。

**💡 创新点**

创新点在于：同一校准场景图被共享给编辑策略和奖励；采用基于删除/重排/插入的 token 对齐编辑；使用置信度保持的奖励塑形与环境感知的 GRPO，通过多图像平均降低噪声；最终避免迭代的 critique‑rewrite 循环。

**🔧 技术方法**

技术包括：场景图提取与校准（DSE）、双模融合编码器（MHE）利用 CLIP+GCN；编辑动作生成器（EAG）实现 token‑级编辑；监督微调 + 基于图像的强化学习（GRPO）与平滑奖励；视觉检索模型 GLIP、BLIP 等。

**📊 数据集**

使用数据集：72K 来自 Lexica 的优化提示做训练，构成 144K 输入–目标对；在 Lexica、DiffusionDB、COCO captions 各 2K 的测试集上评估；同时使用 GLIP、BLIP 等公开数据进行 grounding。

**📈 对比分析**

与模板提示、Promptist、MagicPrompt、BeautifulPrompt、VisualPrompter、GPT‑4o 重写及人类手工提示等进行对比。指标包括 AES_Δ、OR、RR、SGR、Gemini‑VQA。结果显示该方法在关系召回、整体 SGR 及美学提升方面均优于所有基线，并在不同生成器上实现良好迁移。人类评测亦表明在关系数目增多时优势更明显。

**⚠️ 局限性**

局限性包括：对解析器、检索与生成器质量的依赖；场景图在计数、遮挡、否定、情感、隐喻等抽象意图上表达不足；过于保守的局部编辑在需要全局重构时受限；多图像平均导致训练成本上升，推理时需要额外的校准渲染；以及仅在英语提示上验证，缺乏多语言覆盖。

---

## 56. Immersive Commons: Designing Ephemeral Spatial Dialogue for Co-Located Mixed Reality Environments

**arXiv ID:** 2610.36014 | [PDF](https://arxiv.org/pdf/2610.36014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 57. Why and How to Do Science When AI Is Better at It: A Framework Grounded in Human Understanding and Direction

**arXiv ID:** 2610.36116 | [PDF](https://arxiv.org/pdf/2610.36116v1)

**作者:** Hoda Heidari `[一作]` (Carnegie Mellon University), Aydin Mohseni `[通讯]` (Carnegie Mellon University)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5060168471)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出以“持久集体人类理解”为核心的科学目标，并基于此构建了两项测试准则，评估AI在研究各阶段的适用性，随后给出了相应的实践建议与制度改革方案。

**💡 创新点**

创新点在于将科学目的转化为可操作的准则，提出“人类理解”和“人类方向”两条测试维度，区别于以诚信规范为主的现行AI指南，并对所有研究阶段（概念化、方法学、实验、验证、解释、写作、评估、监督）给出细化建议。

**🔧 技术方法**

技术主要是哲学论证与文献综述，并结合LLM辅助撰写与评审的实践，未采用传统机器学习模型。

**📊 数据集**

数据集：无实验数据，使用已发表的文献（约120篇）作为理论与案例依据。

**📈 对比分析**

比较方法：与现有AI使用指南（如欧盟Living Guidelines、IEEE、ACM等）进行对比，强调本框架在维持人类理解与方向方面的优势；未进行数值性能评估。

**⚠️ 局限性**

局限性：框架依赖人类对科学目标的共识，实施需要机构层面的制度变革，对具体学科差异的细化仍需后续研究；此外，过度强调理解可能导致科研效率下降。

---

## 58. AGDecode: Semantic Attribute Grammar Derivatives for Safety-Constrained Decoding of Large Language Models

**arXiv ID:** 2610.36105 | [PDF](https://arxiv.org/pdf/2610.36105v1)

**作者:** Rajan Maghera `[一作]` (University of Toronto), Maryam Mehri Dehnavi `[通讯]` (University of Toronto)

**通讯引用:** 808 | [OpenAlex ID](https://openalex.org/A5003682451)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可编程的语义约束解码框架AGDecode，利用属性语法对生成过程中的语法树进行约束，并将语义检查交给SMT/CHC求解器；

**💡 创新点**

创新点在于：①将属性语法扩展到部分语法树，形成残差属性语法；②把语义约束编译为关系式约束系统并转化为CHC，完成语义可实现性检验；③在理想oracle下证明框架的完备性与健壮性，显著提升了语义约束的表达能力；

**🔧 技术方法**

使用技术包括：属性语法（Attribute Grammars）、残差属性语法、约束求解器（SMT/CHC，Z3 Spacer）、符号化词法空洞、逻辑递归约束、Rocq形式化验证；

**📊 数据集**

实验使用了HumanEval C++子集（转换为C），共77个与内存相关的测试用例，基于CodeLLaMA‑13B模型在3个温度下进行生成；

**📈 对比分析**

与传统仅基于语法的约束解码比较，AGDecode在约25%请求超时、有效请求平均耗时0.1–2秒的情况下仍能完成约束，显示出可行性；虽然相对传统方法会产生一定的推理延迟，但在语义安全属性上实现了显著的约束效果；

**⚠️ 局限性**

局限性包括：需手工编写完整语义约束，缺少完整内存模型；未实现回溯机制导致可疑token无法定位；求解器不完整导致超时或不确定结果；token级别的验证延迟较大，可能影响LLM生成质量与速度。

---

## 59. Jaxnasium: Embracing PyTrees For Any-Agent, Cross-Suite Reinforcement Learning

**arXiv ID:** 2610.36075 | [PDF](https://arxiv.org/pdf/2610.36075v1)

**作者:** Koen Ponse `[一作]` (Leiden University), Thomas M. Moerland `[通讯]` (Leiden University)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5003595753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Jaxnasium库，统一使用PyTree抽象实现可在JAX加速器上训练的强化学习算法，并通过包装器兼容300+ JAX-native与宿主环境（Gymnasium、EnvPool）

**💡 创新点**

创新点在于将观测/动作空间视为PyTree，算法无需改动即可适配离散、连续、图像及复合空间，并可无缝扩展到多智能体；提供单一实现即可支持JAX-native与宿主环境，避免多实现；配套CLI、评估与超参搜索工具

**🔧 技术方法**

利用JAX的JIT、Vmap、Autodiff及Equinox的可变对象，结合PyTree操作构建网络、代理与环境；使用JAX-EnvPool、Gymnasium等包装器；通过PyTree空间构造动态网络

**📊 数据集**

使用11个JAX-native环境套件共计333个环境；覆盖离散、连续、图像、复合空间；以及Gymnasium与EnvPool的多种任务（经典控制、Catch等）

**📈 对比分析**

对比Stable‑Baseline3、CleanRL（PyTorch/JAX）、Anakin/Sebulba（Stoix）在JAX-native、Gymnasium、EnvPool及EnvPool-XLA四种后端，均使用相同超参；Jaxnasium在所有后端的吞吐量均与或优于最快基线，说明单实现对性能无损

**⚠️ 局限性**

目前仅实现四种算法，未来需增加更多；多智能体支持仅限单体学习，复杂CTDE需手动调整；对大规模分布式训练支持不足，适用于研究级别实验

---

## 60. CIVER: Causal Intervention Verification for Reliable Code Review

**arXiv ID:** 2610.36010 | [PDF](https://arxiv.org/pdf/2610.36010v1)

**作者:** Xinyuan Song `[一作]` (Causal Dynamics Lab), Liang Zhao `[通讯]` (Causal Dynamics Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于因果干预的代码审查验证框架，通过对审查结果进行定向修复、无关改动和语义保持变换三类干预，判别发现的缺陷是否真实存在。

**💡 创新点**

创新点在于：①把审查验证视为因果推断问题，引入定向修复和无关干预作为正负对照；②三模块（审查、干预、验证）联合专家迭代训练；③设计多种损失函数（分类、缺陷移除、特异性、等价性、干预排序、最小化、集合筛选），提升验证的因果特异性与稳健性。

**🔧 技术方法**

技术手段包括：冻结大型语言模型（LLM）作为审查者和代码/文本编码器；轻量级交叉注意力 Transformer + MLP 用于评估审查分数；干预模块使用 MLP 排序网络挑选最相关的修复；验证模块聚合各干预下的分数变化；多任务损失融合与专家迭代优化；语义保持变换（重命名、行移位等）。

**📊 数据集**

使用的数据集包括：AACR-Bench（审查有效性分类）、CRScore（人类评估的审查质量）、SWE-Review-Bench（可执行PR、干预构造与评估）、SWE-PRBench、CodeReviewBench、Review4Repair 等，覆盖多语言与跨项目场景。

**📈 对比分析**

与基线（冻结LLM、Self-Reflection、LLM Verification、Fix-Guided Prompt、LLM-as-a-judge、BERTScore、chrF++ 等）对比，框架在 AACR-Bench 上 AUROC 0.60、精度 0.77、FP/PR 降至 1.65；在 CRScore 上 AUROC 0.72、F1 0.43；跨项目 AUROC 可达 0.73；下游修复任务中解决率从 0.61 提升至 0.93；消除误报率提升 16%。

**⚠️ 局限性**

局限性包括：①干预生成依赖于已有的缺陷–修复对，可能无法覆盖所有细粒度缺陷；②需要精准对齐审查语句与代码位置，误对齐会影响评估；③虽然模块轻量，但整体仍需多次评估不同干预状态，计算成本相对较高；④在极其复杂或缺陷不易定位的代码上，因果特异性验证效果可能下降。

---

## 61. DuplexTune: Coupled-State Replay for Memory-Efficient Mixture-of-Experts Adaptation

**arXiv ID:** 2610.36009 | [PDF](https://arxiv.org/pdf/2610.36009v1)

**作者:** Paul Raj `[一作]` (Texas A&M University-Commerce), Mutlu Mete `[通讯]` (Texas A&M University-Commerce)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合耦合状态重放、宽度桥接和分阶段主干适配的稀疏专家语言模型微调框架（DuplexTune）。

**💡 创新点**

创新点在于通过宽度桥接将半宽残差流与预训练的注意力和专家模块耦合，并在主干适配前先热身桥接，实现激活重放而非完整存储；同时设计了桥接热身与主干适配的两阶段训练计划。

**🔧 技术方法**

使用了注意力与稀疏专家的预训练参数、可学习的宽度桥接、固定路由器权重的稀疏专家架构、卷积与重放算法，以及梯度和激活的重构分析。

**📊 数据集**

主要数据集为Qwen1.5-MoE-A2.7B模型的适配，评估任务包括MMLU、GSM8K、MT-Bench以及未公开的多语种聚合数据。

**📈 对比分析**

与传统带检查点的监督微调、LoRA、DoRA、(IA)^3等PEFT方法相比，DuplexTune在单张80 GB H800 GPU上将峰值显存从64.8 GB降至38.8 GB（约40 %），吞吐率从19.9样本/秒提升至25.1样本/秒；同时在MMLU、GSM8K、MT-Bench上分别取得67.2%、75.6%和7.69的分数，优于基线和多数对比方法，但多语种得分略低。

**⚠️ 局限性**

局限性包括：实验仅在单一模型、单一指令数据集和单一GPU类型上验证；未对重放误差、激活恢复质量以及各类状态存储进行独立消融；多语种评价缺乏标准化基准，且无法直接衡量重放过程对推理质量的影响；最后，方法的可迁移性和在更大规模或不同硬件上的表现仍需进一步研究。

---

## 62. Merkle-Tree-Based Verifiable Public Data Auditing for Cloud Object Storage Using Smart Contracts

**arXiv ID:** 2610.36001 | [PDF](https://arxiv.org/pdf/2610.36001v1)

**作者:** Siri Reddy Tadi `[一作]` (Chaitanya Bharathi Institute of Technology), Jaswanth Bhimireddy `[通讯]` (Chaitanya Bharathi Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个基于Merkle树的可公开验证数据完整性审计系统，在云对象存储中通过智能合约存储根哈希，实现单对象、采样与全量审计。

**💡 创新点**

创新点在于采用SHA‑256叶子绑定对象键的Merkle树，仅在链上存根；提供完整、修改、缺失三态分类；在实验中量化采样检测概率与成本；并与完整数据校验进行对比。

**🔧 技术方法**

使用的技术包括Merkle树、SHA‑256哈希、以太坊智能合约（Solidity）、Hardhat本地链、Python、MinIO SDK、s3s‑fs S3兼容服务器等。

**📊 数据集**

使用的实验数据集为四个合成集合，分别包含10、100、1,000和10,000个对象，单对象大小在1 KiB–256 KiB之间，1,000对象总计48.68 MiB。

**📈 对比分析**

通过与完整数据校验对比实验，发现Merkle全审计比完整下载多约1%（约1.44 s vs 1.37 s），采样审计和单对象审计分别约150 ms和10 ms；证明长度最高14个哈希；根注册仅消耗91,863 gas；在10,000对象时全审计约14.3 s。

**⚠️ 局限性**

局限性包括仅检测完整性不提供恢复或可用性；根生成需在受信任环境下完成；频繁变更数据需要多次根注册；实验仅在本地模拟，未评估真实云存储和公链网络延迟与gas费用。

---

## 63. PrinGS: Privacy Preservation in Gaussian Splatting

**arXiv ID:** 2610.36168 | [PDF](https://arxiv.org/pdf/2610.36168v1)

**作者:** Gaurav Behera `[一作]` (IIIT Hyderabad), Ponnurangam Kumaraguru `[通讯]` (IIIT Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对已训练的3D高斯喷射模型进行后训练或隐私设计级别的匿名化，使其在任意视角下消除人脸、文字等敏感信息。

**💡 创新点**

首创直接在高斯分布参数上进行局部模糊的表示级别匿名化，并将此机制嵌入训练流程，形成后训练与隐私设计双模式；同时提出专门针对3DGS的面部与文本隐私评测指标。

**🔧 技术方法**

3D高斯喷射（3D Gaussian Splatting）；基于可微光栅化的3D高斯分割；对高斯参数的邻域均值模糊；使用SAM/InsightFace/SCRFD/Google OCR等工具做评测。

**📊 数据集**

自制的9人多种肤色、性别的Blender室内场景；以及五个带可读文字的公开场景（Truck、Barn、M60、Car、Piano）。

**📈 对比分析**

与标准3DGS基线对比；使用FSM、FDR和CER评估；结果显示FSM从83.87%降至9.79%，FDR降至39.47%，CER从0.208提升至1.00，且后训练匿名化仅需1秒。

**⚠️ 局限性**

只针对人脸和文字；对复杂几何或非可分割敏感区域效果有限；仍需人工标注参考视角和掩码；缺乏对其他类型隐私（如车辆轨迹）的评估。

---

## 64. Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing

**arXiv ID:** 2610.36185 | [PDF](https://arxiv.org/pdf/2610.36185v1)

**作者:** Zihan Fang `[一作]` (Vanderbilt University), Gina R. Bai `[通讯]` (Vanderbilt University)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5040160847)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并评估了一种名为TestCompass的反馈型AI辅导工具，探究其在单元测试教学中的效果，并与通用AI助手（如ChatGPT）进行对比。

**💡 创新点**

创新点在于：1）提供基于覆盖信息的定向反馈，而不生成可执行测试代码；2）通过多阶段验证提升反馈质量；3）强调学生主动实现与验证测试，保持学习主体性。

**🔧 技术方法**

技术包括：IntelliJ IDEA插件实现、代码覆盖分析（IDEA Coverage、JaCoCo）、大型语言模型（LLM）生成反馈、Mutation测试（PIT）评估有效性、SonarQube静态分析评估可维护性、实验室研究设计与问卷调查。

**📊 数据集**

使用的数据集为两个Java单元测试任务（MarsRover与BowlingScoreKeeper，分别包含12/13条需求）以及18名初学者的测试套件、屏幕录制与问卷数据。

**📈 对比分析**

通过随机分配18名学生到TestCompass或通用AI组，记录行为、测试质量四维度（完整性、有效性、可维护性、实现障碍），采用精确置换检验、Cliff's δ、Fisher等统计方法。结果显示：TestCompass与通用AI在大多数质量指标无显著差异；TestCompass在实现相同需求覆盖时测试方法显著更少；通用AI生成更大套件但未带来显著覆盖或有效性提升。

**⚠️ 局限性**

局限性包括：样本量小、仅来自单一课堂、任务顺序固定、通用AI工具差异导致结果波动、仅短期实验且未评估长期学习影响、仅针对Java/IntelliJ环境、覆盖与变异指标可能无法完全反映测试质量。

---

## 65. Replication or Corroboration? An Axiomatic Framework for Evidential Dependence in Multi-Agent Debate

**arXiv ID:** 2610.36151 | [PDF](https://arxiv.org/pdf/2610.36151v1)

**作者:** Hoang-Loc Cao `[一作]` (University of New Brunswick), Hung Cao `[通讯]` (University of New Brunswick)

**通讯引用:** 3238 | [OpenAlex ID](https://openalex.org/A5038366927)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多代理辩论中“复制”与“证据互证”如何混淆判决，提出一种基于根划分（lineage quotient）的表示方法，并在量化双极论证框架（QBAF）中实现，使得判决器对同一证据的重复不敏感，却能对真正的新根产生响应。

**💡 创新点**

创新点在于：① 给出复制与互证的形式化对比并证明复制不变性与阈值不变性等价；② 构造最小信息边界，证明任何忽略复制同时严格响应新根的判决器必须至少保留根划分信息；③ 设计出最粗最小的根分辨表示——lineage quotient，并证明它是所有满足复制不变性与根重命名不变性的判决器所需的最小表示；④ 将该表示提升到 QBAF，保证在签名一致路径下新根的影响严格传递，并给出紧致的数值增益下限。

**🔧 技术方法**

技术包括：形式化最小判决游戏与复制不变性证明、信息边界定理、根分辨线性分割的定义与性质证明、QBAF 结构化与 quotient 的映射、签名一致路径的传播定理、数值下限公式、以及大规模结构化与真实案例审计。

**📊 数据集**

使用的数据集有：1）28,224 个直接星形 QBAF 结构；2）2,052 个多层 DAG 结构；3）277 个来自 AVeriTeC 的真实问答证据集（支持与否），其中每条证据被单独标记为根。

**📈 对比分析**

与基准方法（Vote、occurrence、Max‑QuAD、content quotient、URL quotient）比较，线性分割在 100% 的案例中实现了零重复误差、零判决翻转、零与原始分数的偏差，并且在所有新根插入中保持正向响应；其他方法要么对复制敏感（Vote、occurrence），要么对新根不响应（Max‑QuAD、content quotient）。实验表明线性分割的平均正向增益为 0.0298，最小增益 0.0013，满足理论下限。

**⚠️ 局限性**

局限性：① 需要事先给定准确的根划分，无法处理根的分级或不完整的依赖信息；② 对于根标签造假、内容篡改或强制新根的情况未覆盖；③ 仅证明了确定性判决器的性质，未探讨随机或学习型判决器；④ 只在直接星形与 DAG 结构下验证，尚未证明在更复杂图或中间节点干预下的泛化。

---

## 66. Program-Level Adversarial Co-Adaptive Self-Play for Robotic Security

**arXiv ID:** 2610.36143 | [PDF](https://arxiv.org/pdf/2610.36143v1)

**作者:** Leo Benaharon `[一作]` (Isuzu Technical Center of America), Yifan Wei `[通讯]` (Isuzu Technical Center of America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于可执行程序的对抗性共适应自对弈框架，利用大型语言模型生成攻击与防御代码，针对Autoware-CARLA和Unitree G1-MuJoCo平台在八代演化过程中对车辆行驶与人形步行任务进行评估。

**💡 创新点**

创新点在于：不通过模型微调而直接在软件层面进行程序演化；通过共享防御程序和任务导向的评价机制，将对抗过程与机器人任务性能紧密耦合；实现了可观察的、可追溯的程序级自对弈过程。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude、Opus、Sonnet）生成ROS 2/DDS代码；CARLA 与 MuJoCo 模拟器；程序检查与冻结机制；自定义质量评估函数（进度+速度比）和恢复评分；攻击者角色分配与守卫者协调。

**📊 数据集**

使用的数据集为两种仿真环境：Autoware在CARLA中60米直线行驶任务，以及Unitree G1在MuJoCo中3米平地步行任务，构成实验任务与环境数据。

**📈 对比分析**

比较方法：对每一代进行清洁、仅防御、仅攻击、联合三次试验，计算质量分数与恢复率；在驾驶任务中，约18/40比赛成功完成，攻击者占优势；在人形任务中，51/176正式评估成功，防御恢复率随代数提升但仍有失败。

**⚠️ 局限性**

局限性：仅在单一任务/情景下进行了八代演化，缺乏跨场景通用性；评价指标仅基于进度与速度，未涵盖安全性、平滑度等；未对反馈机制进行消融分析；防御编排器集成与回滚策略尚未完善。

---

## 67. Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses

**arXiv ID:** 2610.36114 | [PDF](https://arxiv.org/pdf/2610.36114v1)

**作者:** Yuvraj Sehgal `[一作]` (Brock University), Naser Ezzati-Jivan `[通讯]` (Brock University)

**通讯引用:** 573 | [OpenAlex ID](https://openalex.org/A5019537475)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过量化 LLM 代理工具输出被截断导致的信息流失，提出将代理上下文视为五阶段流水线，并在真实的 Linux 内核追踪诊断任务中测量每个阶段的失误率；随后通过受控实验验证截断对代理性能（额外计算成本）和准确率的影响，并对七种主流代理框架进行代码审计，发现多数采用位置截断且无安全替代方案。

**💡 创新点**

创新点在于：①将代理上下文抽象为五阶段流水线并对每个阶段失效率进行可测量；②系统地演示工具输出截断导致大约三分之一结果被破坏；③通过受控实验分离截断对准确率与额外工作量的影响；④对现有框架进行截断策略审计，揭示普遍存在的安全隐患。

**🔧 技术方法**

技术手段包括：构建并 instrument 代理 harness 的五个边界，记录产出与交付差异；使用前缀切片与行级保留两种截断实现对比；在真实内核追踪数据上 replay 结果并测量破坏率；设计并执行多配置的实验（修复版、S1 缺陷、反向键顺序等）；统计工具调用次数、token 量、yield 等指标；使用 Python 脚本实现结果重放与计分。

**📊 数据集**

数据集主要来自 720 条 Linux 内核追踪诊断运行，产生 11,189 条工具输出；另外重放 864 条真实结果；还使用了 21 容器的生产级追踪作为规模验证；此外收集了七个代理框架的源码进行截断策略审计。

**📈 对比分析**

对比方法：在相同任务、模型、工具下，比较“修复版”与“缺陷版”在准确率、工具调用次数、token 消耗、yield 等指标上的差异。实验结果显示，截断版虽然准确率基本不变，但工具调用次数提升 28%–43%，计算量翻倍，yield 降至 0.34–0.31（修复版为 0.61），证明截断导致额外工作成本而非直接降低正确率。

**⚠️ 局限性**

局限性包括：①仅在单一 harness、单一域（内核追踪）和单一模型上评估；②实验使用的是固定的追踪记录，缺乏多样化故障实例；③未完全证明截断是导致准确率下降的因果因素；④仅对七个框架进行截断策略审计，可能未覆盖所有实现。

---

## 68. Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates

**arXiv ID:** 2610.36109 | [PDF](https://arxiv.org/pdf/2610.36109v1)

**作者:** Zehao Zhao `[一作]` (Meta AI), Hamed Firooz `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Control Map 评估框架，用以记录测试时更新前的有序更新历史，并基于此评估行为变化、即时学习、保留和局部性等指标；

**💡 创新点**

创新点在于将可追踪的更新历史与模型输出直接关联，提供正式的历史分辨率和得分差异界限，并可在不同模型/基准下直接检验结果的可迁移性；

**🔧 技术方法**

使用可追踪的更新记录、控制图、概率界限、抽样对比、以及多种编辑器（LoRA、ROME、MEMIT、AlphaEdit 等）和统计分析技术；

**📊 数据集**

利用多任务数据集，包括 Llama‑3.2‑3B、GPT‑2XL、CounterFact、WikiRecent、Single‑NIAH、SocialIQ、LAMBADA 等；

**📈 对比分析**

通过预先定义的加权对比、历史分辨率、行为差异量化、局部性阈值等方法进行评估，实验显示同一更新计数下更新时机/位置对准确率影响显著，记忆质量往往未能通过局部性检查；

**⚠️ 局限性**

局限在于评估结果高度依赖手工设定的追踪规则，未覆盖所有潜在路径；跨模型/基准迁移性受限，实验规模和可复现性亦有限。

---

## 69. How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning

**arXiv ID:** 2610.36082 | [PDF](https://arxiv.org/pdf/2610.36082v1)

**作者:** Johnny Yu `[一作]` (University of California, Berkeley), Cyril Gorlla `[通讯]` (CTGT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为Fin‑SD的局部视角自蒸馏方法，针对金融多步推理任务，在模型自身生成的轨迹上定位错误位置，生成简短批评并对前缀进行逆KL蒸馏，从而提升模型的准确率与生成效率。

**💡 创新点**

创新点在于：①仅使用自模型生成的错误定位与批评，完全不依赖外部教师；②发现错误定位本身即可匹配完整参考解答的性能，提示内容可简化为通用反馈；③采用短回合滚动（100令牌）与逆KL目标，显著提升训练效率与生成质量。

**🔧 技术方法**

使用的技术包括：on‑policy自蒸馏（OPSD）改进、逆KL蒸馏目标、冻结教师、错误定位与简短批评生成、短回合(100)的前缀蒸馏、全局自监控与梯度累积。

**📊 数据集**

训练数据基于FinanceReasoning库中的3133个Python金融函数，筛选难度≥3后得到295个公式，再生成220个多步金融推理问题；评测使用FinanceReasoning Hard子集（238题）、MMLU‑Pro（去除商科经济类）和FinTrust合规基准。

**📈 对比分析**

与基线模型（OPSD、SFT、原始GPT‑OSS‑20B）进行对比：Fin‑SD在FinanceReasoning Hard上准确率72.19%（+3.03pp vs 基线）、完成率92.19%（+1.01pp）并减少了约17.3k令牌；与OPSD相比保持相同准确率但减少9.5%令牌；在MMLU‑Pro和FinTrust上无显著性能退化。

**⚠️ 局限性**

局限性包括：①自蒸馏上限受模型自身知识限制；②训练集规模有限且与评测存在重叠；③未做大量超参数调优；④缺乏对不同模型规模或不同领域的广泛验证；⑤不支持多假设可解释性任务。

---

## 70. The Trinity:Capability-Driven Recursive Self-Improvement for Multimodal Agents

**arXiv ID:** 2610.36077 | [PDF](https://arxiv.org/pdf/2610.36077v1)

**作者:** Weidong Tang `[一作]`, Yang You `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了The Trinity框架，通过Creator、Explorer、Evolver三种角色实现多模态任务的能力驱动递归自我改进，将任务、能力与Harness树结构化，利用真实‑模拟‑真实循环生成可验证的简化环境，提升多任务通用性。

**💡 创新点**

创新点在于将任务聚合到共享的能力需求而非表面类别，构建能力树并通过能力驱动的Real2Sim2Real生成可训练的世界，同时构建可继承的Harness树和能力感知的Router，使得不同模态任务在共享能力上实现模块复用与迁移。

**🔧 技术方法**

采用冻结的Claude Opus 5大模型作为基线，结合六协议的模块化Harness设计、基于轨迹的能力抽取与层级化树构建、选择性继承与局部细化的模块演化、以及基于公共经验检索的能力感知Router，形成完整的递归自我改进循环。

**📊 数据集**

使用28个训练基准和18个未见基准（共44个多模态任务，涵盖规则发现、约束求解、感知控制、多智能体交互和长时序环境），并在两套真实机器人任务中验证迁移。

**📈 对比分析**

与基线的传统文本任务驱动RSI、单一Harness设计以及其他分支方法相比，在训练集上平均提升34分、在未见集上提升26分，且在两项真实机器人任务中保持显著性能提升，展示了跨任务泛化与可扩展性。

**⚠️ 局限性**

局限性包括对Web搜索作为基准发现的依赖导致检索覆盖与质量不一致，以及研究聚焦于Harness层面的RSI，尚未扩展至模型参数级的自我改进或动态扩展能力树。

---

## 71. Learning and Reasoning with Geometry: How Generation Helps Spatial Understanding

**arXiv ID:** 2610.36022 | [PDF](https://arxiv.org/pdf/2610.36022v1)

**作者:** Rui Yang `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GeoUG，一种统一的模型，将几何生成（深度和多视角生成）与空间理解结合，用于训练和推理；

**💡 创新点**

创新点包括：①将几何生成作为联合训练的监督，提升不同空间理解任务的性能；②设计了多尺度度量深度编解码器 MMDC，支持在预训练 VAE 上生成可解码的深度；③提出自适应交错推理（draft-conditioned policy），通过强化学习学习何时生成几何证据以平衡准确率与生成成本；④创建了 3DCueBench，控制几何难度并平衡 2D 线索与 3D 关系的评估。

**🔧 技术方法**

技术手段：统一的生成与理解框架基于 BAGEL，使用流匹配目标生成；MMDC 将深度映射为三通道限制值；强化学习 (GSPO) 训练 draft-conditioned 决策；多尺度深度编码与归一化；基于 VAE 的噪声恢复生成。

**📊 数据集**

使用的数据集包括：CV-Bench-3D、DA-2K、MindCube、MMSI、NYU-v2、KITTI、ETH3D、Sintel、DIODE、3DCueBench（自建）、Synthetical 多视角与 exo-to-ego 生成数据。

**📈 对比分析**

与多种基准模型（BAGEL、Qwen、SenseNova、VST、DA3 等）对比，GeoUG 在 CV-Bench-3D、3DCueBench、DA-2K 的循环准确率分别达到 97.9%、94.2%、90.8%，几乎匹配或优于传统 VLM+深度读数模型，且生成调用率仅为 50–70%。在多视角与 exo-to-ego 生成任务上，MMDC 生成的深度质量优于 Marigold、Lotus，提升 4–5 dB PSNR。

**⚠️ 局限性**

局限性：生成质量仍受限于 VAE 编码范围，深度错误可能导致推理失败；自适应决策依赖于强化学习的奖励设定，可能在不同任务或成本预算下需重新调参；当前框架主要针对单图像/单视角推理，扩展到视频或连续动态场景仍需研究。

---

## 72. Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling

**arXiv ID:** 2610.36015 | [PDF](https://arxiv.org/pdf/2610.36015v1)

**作者:** Christoph Daxerer `[一作]` (Software Competence Center Hagenberg Gmbh), Josef Pichler `[通讯]` (University Of Applied Sciences Upper Austria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了三种前沿大型语言模型（Sol 6.1、Luna 6、Qwen 3.8）在单行故障定位任务中的准确性，并比较了无工具、可选工具和强制调试器协助三种设置下的表现。

**💡 创新点**

创新点在于：①将行为保持的代码变换（五种不同语法重写）与故障定位结合，衡量模型对代码重写的鲁棒性；②区分可选与强制调试器使用，系统地量化工具使用成本与收益；③使用精细的陈述级与行级评分，验证对拆分语句的容忍度。

**🔧 技术方法**

主要技术包括：大语言模型推理、工具调用接口（程序执行、交互式调试器）、行为保持的代码重写（重命名、状态/字面量、控制流重写等）、统计分析与置信区间、Bootstrap 误差评估。

**📊 数据集**

使用 LeetCodeDataset 的 150 条自包含算法题目（共 610 条单行故障），并对每条故障生成 5 种重写版本，得到 3,022 条变形程序。

**📈 对比分析**

与无工具基线相比，强制调试器使用在 Luna 6 和 Qwen 3.8 上分别提升 2.1% 与 1.3% 的准确率，几乎消除因代码重写导致的误差；可选调试器使用仅提高 1.5%–1.3% 的准确率，却大幅降低了 token 与时间成本。Sol 6.1 在无工具下已达到 99.9% 的准确率，调试器使用几乎无收益。

**⚠️ 局限性**

局限性包括：①仅针对短、完全指定的单文件 Python 程序，无法推广到大型多文件项目或缺乏完整规范的情况；②评估的“可选”工具使用可能受模型自身偏好影响，导致工具使用频率与收益不完全对应；③使用的行为保持重写并未完全消除与训练语料的重叠，可能低估了模型对新语法的敏感性。

---

## 73. E0.5: From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling

**arXiv ID:** 2610.36011 | [PDF](https://arxiv.org/pdf/2610.36011v1)

**作者:** X-Era AI Lab Team `[一作]` (X-Era Ai), Guangrun Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于3D foundation模型的世界-动作模型E0.5，能够同时学习机器人动作和未来几何状态；

**💡 创新点**

创新点在于将可训练的3D backbone作为核心支撑，并通过未来几何预测作为监督，统一几何与动作学习，区别于传统以视频或视觉语言为主的模型；

**🔧 技术方法**

采用VGGT‑Omega 3D backbone、Mixture‑of‑Transformers连接、语言条件、流匹配动作生成、未来几何预测头，并在全参数训练下使用CFG引导的流匹配推理；

**📊 数据集**

使用RoboTwin 2.0仿真任务、AgiBotWorld‑Beta大规模机器人交互数据以及真实机器人（AgileX Piper与Zerith H1）进行训练与评估；

**📈 对比分析**

与VLA/WAM基线相比，E0.5在RoboTwin Clean/Random分别达89.9%/50.1%成功率，真实机器人ID场景均可达100%/90%/100%，OOD场景也大幅提升；大规模预训练进一步提升了现实世界表现；

**⚠️ 局限性**

局限性包括在极端随机或复杂环境下性能仍有限，对某些任务（如垃圾收集完整性）表现不佳，对几何监督质量和预训练数据的依赖较大，且验证范围主要集中在特定任务和机器人上。

---

## 74. safeL4: Rethinking Safe-Language-Based Single Address Space Microkernels

**arXiv ID:** 2610.35999 | [PDF](https://arxiv.org/pdf/2610.35999v1)

**作者:** Zheyuan Xu `[一作]` (Chinese Academy of Sciences), Rui Hou `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 Rust 的单地址空间微内核 safeL4，利用对象能力实现无硬件隔离的能力系统。

**💡 创新点**

创新点在于把 Rust 的对象能力与能力派生树结合，既保证能力不可伪造又支持撤销，保持微内核最小性与细粒度访问控制。

**🔧 技术方法**

采用 Rust 语言、对象能力、能力派生树、程序化宏、零拷贝 IPC、同步/异步 IDC 等技术。

**📊 数据集**

使用 Raspberry Pi 3B+ 的真实硬件、seL4、Redox、Linux 系统以及 SQLite3 的 YCSB 负载、LMbench 文件系统基准。

**📈 对比分析**

与 seL4、Redox、Linux 进行微基准与宏基准对比，safeL4 在 IPC/IDC 等操作上平均比 seL4 提升 1.9×，在 IDC 上提升 7.8–22.3×，在 YCSB 基准上比 seL4 高 14.3%，比 Linux 高 23.2%。

**⚠️ 局限性**

局限在于仍需在 TCB 中包含一定量的 unsafe 代码（尤其是硬件抽象与宏），缺乏 IOMMU 支持的 DMA 隔离，驱动程序的功能正确性未被保证。

---

## 75. Scaling and Renormalization in Neural Networks: An Effective Renormalization-Group Description of Grokking

**arXiv ID:** 2610.36113 | [PDF](https://arxiv.org/pdf/2610.36113v1)

**作者:** Min Chen `[一作]` (University of Pittsburgh), Junyu Liu `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建物理词典、有效空间和局部重整化群映射，建立了对神经网络中“grokking”现象的定量相变描述。

**💡 创新点**

创新点在于：①将训练步数、权重衰减等训练过程映射为温度、场等热力学变量；②利用损失 Hessian 生成耦合矩阵并通过 Fiedler 向量构造一维有效空间；③在此空间上实现邻域块化并拟合局部 RG 转移，得到五个临界指数。

**🔧 技术方法**

主要技术包括：梯度下降与 AdamW 优化、全批 MSE 损失、权重衰减作为外场、训练历时作为温度、跨跑方差作为热容与磁化率、Hessian 推导耦合矩阵、谱图排序（Fiedler 向量）、块化重整化、线性化 RG 映射、标准相变缩放关系。

**📊 数据集**

使用的数据集为两层无偏置感知机学习的模数加法（p=97），输入为 194 维 one‑hot 编码，对所有 p² 组合进行 50% 随机划分，训练集与验证集各自包含 50% 样本。

**📈 对比分析**

比较方法是直接利用上述物理量做对数-对数拟合得到的临界指数，并与 RG 计算得到的指数做百分比差异对比，差异保持在 1–10% 之间，显示两种方法一致性良好。

**⚠️ 局限性**

局限性包括：①仅在单一二维加法任务上验证，缺乏跨任务普适性检验；②使用的有效空间是一维链，可能无法捕捉更高维结构；③对梯度、Hessian 计算的数值稳定性和计算成本较高；④训练随机种子与数据划分产生的 quenched 噪声，使得 Gibbs 近似与真实热力学行为可能存在偏差。

---

## 76. Is the education gap in generative AI use smaller than the internet's was at a similar level of use? Evidence from 16 European countries

**arXiv ID:** 2610.36181 | [PDF](https://arxiv.org/pdf/2610.36181v1)

**作者:** Keisuke Watanabe `[一作]` `[通讯]` (Glasp Inc), Keisuke Watanabe (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用欧盟2025年Eurostat调查数据，比较了生成式人工智能工具使用的教育差距与2002-2012年互联网使用相同总体水平时的教育差距。

**💡 创新点**

创新点在于首次以相似总体使用水平进行跨技术的教育差距匹配，而非仅按时间顺序或人口结构比较；同时提供多重稳健性检验。

**🔧 技术方法**

采用的技术包括对欧盟ICT使用数据的匹配、对高低教育组使用率计算对数比与百分点差异、Bootstrap 95%置信区间和符号检验。

**📊 数据集**

使用的数据集为Eurostat的isoc_ai_iaiu（2025年生成式AI使用率）、isoc_ci_ifp_iu（2002-2026年互联网使用率）以及lfsa_pgaed（教育构成）。

**📈 对比分析**

比较方法为：为每国匹配互联网年份，其总体使用率最接近2025年生成式AI总体使用率；随后计算生成式AI与匹配互联网年份的教育差距差值。结果显示，在16个匹配国家中，生成式AI的教育差距在点差上普遍减小（中位数-27.6个百分点，p<0.001），但在比值上不稳健。

**⚠️ 局限性**

局限性包括：只覆盖了互联网采用较晚的16个国家，未控制年龄、就业、学生比例等混杂因素；使用的单一年份生成式AI数据；教育水平定义为最高完成学历，未区分在读学生；以及未进行预注册，所有分析均视为探索性。

---

## 77. Stratum: Repository Memory for Coding Agents Beyond the Context Window

**arXiv ID:** 2610.36153 | [PDF](https://arxiv.org/pdf/2610.36153v1)

**作者:** Yuvraj Sehgal `[一作]` (Brock University), Naser Ezzati-Jivan `[通讯]` (Brock University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为 Repository Memory for Coding Agents (ReM) 的多层次仓库内存架构，能够在代码仓库旁持久化结构、历史和经验知识，并通过页面安全工具让大型语言模型在多会话中共享和检索这些知识。

**💡 创新点**

创新点：①将内存放在外部存储（非模型 KV 缓存）以实现跨会话、跨模型持久化；②引入经验层记录失败、根因等负知识，提升知识的可追溯性；③设计 Page‑Safe Model Context Protocol，将分页、预算信息显式传递给模型；④提出语义驱动的缓存置换评分（图距离、共变频率、时间衰减等）。

**🔧 技术方法**

使用技术：SQLite 图数据库、HNSW 向量索引、文件监视器、token 预算控制、Page‑Safe MCP、Co‑change 挖掘、嵌入检索与词典检索、九个工具接口。

**📊 数据集**

数据集：六个开源项目（Django、Flask、Redis、Requests、scikit‑learn、Vue core），从中抽取 752 个 bug‑fix 提交（共 852 条文件对）进行静态分析，并用 1 万级节点的合成数据评估存储与检索。

**📈 对比分析**

比较方法：在 8 个真实 bug‑fix 任务上比较三种条件——仅检索（A）、检索 + 经验记忆（B）和整仓库放入上下文（C）。实验结果显示：B 成功率 21/24，显著高于 A 的 6/24（Fisher p=1.3e-5），且中位输入 token 成本降低 4.5 倍；C 成功率与 B 相当，但单次调用 token 成本高 3.8 倍。

**⚠️ 局限性**

局限性：仅评估 Python 项目，导入图基于最新提交可能与分析时提交不一致；采样方式可能偏向描述清晰的修复；实验规模小，未验证大仓库或跨模型迁移效果；检索使用词典匹配，未测试嵌入检索；未评估写回记忆的质量、污染风险及完整性。

---

## 78. Texture Consolidation of Generated Multi-View Renderings

**arXiv ID:** 2610.36098 | [PDF](https://arxiv.org/pdf/2610.36098v1)

**作者:** Hongyi Liu `[一作]` (Johns Hopkins University), Misha Kazhdan `[通讯]` (Johns Hopkins University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种将稀疏多视生成渲染与纹理图集网格相结合的纹理合成系统，能从少量视图生成无缝、视角一致的纹理图像。

**💡 创新点**

核心创新在于引入置信度加权的光流对齐和梯度域拼接，并使用四分积分实现纹理空间与网格空间的混合离散化，从而克服生成渲染的几何与光照不一致问题。

**🔧 技术方法**

技术实现包括光照归一化、基于法线和深度的置信度评估、光流求解（基于变分方法的时间步迭代）、梯度域拼接、四分积分积分与稀疏线性系统求解。

**📊 数据集**

实验数据主要来自基于Diffusion模型生成的稀疏多视渲染与预先构造的纹理图集网格（如Bear模型等），未具体列出公开数据集名称，但可在论文中通过提供的GitHub链接复现。

**📈 对比分析**

与传统RGB‑D捕获的纹理注册与拼接方法（如视角筛选+光流+梯度拼接）进行对比，结果显示在稀疏生成视图场景下该方法在PSNR/SSIM以及视觉无缝度方面均优于基线，且显著减少了鬼影与细节损失。

**⚠️ 局限性**

局限性包括：仅针对稀疏生成视图的情况，对深度缺失或极端遮挡的处理仍不够稳健；对纹理分辨率和网格细化程度敏感；以及在大视角覆盖或动态场景下的扩展性尚待验证。

---

## 79. OmniBrain-0: A Unified Agentic Model for Embodied Intelligence

**arXiv ID:** 2610.36013 | [PDF](https://arxiv.org/pdf/2610.36013v1)

**作者:** Xiaomi Embodied Intelligence Team `[一作]` (University of Macau), Hangjun Ye `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OmniBrain-0，一种统一的紧凑型具身智能体，通过高层决策模型协同操控和导航工具实现任务执行；

**💡 创新点**

创新点在于三阶段进化训练框架（共享预训练→领域专精→多专家整合），以及可扩展的Rubric‑Guided Recursive Self‑Improvement（RSI）循环，能够在保持低层工具专业化的同时，实现跨域决策迁移与持续改进；

**🔧 技术方法**

采用大型多模态语言模型（Qwen3.5-9B）作为高层决策器，结合工具抽象接口、对抗式数据构建、回溯式监督（MOPD）和交互式提醒策略；

**📊 数据集**

使用多种机器人基准数据集：LIBERO-Pro、LIBERO-Plus、RoboCasa365、RoboTwin 2.0、R2R‑CE Val‑Unseen，以及EmbSpatial‑Bench、RefSpatial‑Bench、真实世界的块叠、苹果放置和抽屉打开任务；

**📈 对比分析**

与多种专门策略（π_0.5、π_0、LingBot‑VLA等）以及闭源教师（GPT‑5.6‑Sol）对比，OmniBrain‑0在所有五大基准上实现了高于教师的成功率或接近教师的性能，同时显著降低了标记消耗（4‑12倍）和完成时间（1.1‑8.6倍），在空间理解和真实世界实验中亦表现出竞争力；

**⚠️ 局限性**

局限在于对低层工具的依赖，未覆盖所有任务组合和空间定位，且在未见任务组合和真实场景的适配仍有提升空间，未来需扩大任务和平台覆盖并加强对分布偏移的鲁棒性。

---

## 80. RELAY: One Query, Two Roles for Sequential Recommendation

**arXiv ID:** 2610.36000 | [PDF](https://arxiv.org/pdf/2610.36000v1)

**作者:** Zesheng Li `[一作]` (University of Chinese Academy of Sciences), Honggang Qi `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为RELAY的序列推荐模型，利用单一查询同时完成语义读出与行为记忆路由。

**💡 创新点**

创新点在于将历史选择共享给两个角色，采用单查询双角色机制，并通过完整目录似然联合训练语义匹配与记忆路由，最终实现一次前向推断即可完成排名。

**🔧 技术方法**

技术手段包括注意力机制、语义索引（SID）表征、两路行为记忆（直接历史与转移记忆）、候选条件门控以及全目录softmax学习。

**📊 数据集**

使用六个Amazon评论数据集（Sports、Beauty、Toys、Scientific、Instruments、Games）进行评估。

**📈 对比分析**

与16个基线方法对比，RELAY在Recall@5和NDCG@5上平均提升14.22%和17.88%，参数量减少65%，推理速度提升1.75–1.79倍。

**⚠️ 局限性**

局限在于依赖大规模语义码表和预构建的转移统计表，对极少量历史的样本仍可能表现不佳。

---

## 81. PolicyMediator: Multi-Regional Policy Learning through Mediation Agents with Evolving Skills in OPC Ecosystems

**arXiv ID:** 2610.35997 | [PDF](https://arxiv.org/pdf/2610.35997v1)

**作者:** Jie Li `[一作]` (Northwestern Polytechnical University), yu li `[通讯]` (Northwestern Polytechnical University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了PolicyMediator系统，解决一人公司跨地区选择时因政策信息不完整导致的匹配失效问题。

**💡 创新点**

创新点在于将认知约束的Stackelberg政策学习、可验证的双边调解与基于结果的技能演化结合，形成闭环反馈机制。

**🔧 技术方法**

采用强化学习的零阶策略梯度优化政府政策，LLM代理用于认知更新与调解对话，基于证据的技能库实现可验证说明。

**📊 数据集**

使用六个中国地区的真实政策数据、行业适配度、入门难度与容量等统计量，并在不同LLM后端（Qwen、Ministral、Gemma）上评估模拟企业。

**📈 对比分析**

与八种基线（无政策、静态福利、上下文置信、Carbon‑PPO、Nash‑Welfare、Belief‑Robust、Bi‑level Consensus、CONSENSAGENT）比较，PolicyMediator在所有15种设定下获得最高政府目标与企业效用，平均提升约42.4%和40.9%。

**⚠️ 局限性**

局限包括使用模拟企业行为、政府合作假设、仅适用于可验证政策的跨区协调，难以推广至竞争性或长期动态环境。

---

## 82. Dynamical coarse-graining of neural-network training

**arXiv ID:** 2610.36180 | [PDF](https://arxiv.org/pdf/2610.36180v1)

**作者:** Lin-Juan Wang `[一作]` (Beihang University), Jianxiang Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究神经网络训练的动力学粗粒化，提出学习算子和三大定理，并在有限宽ReLU网络中构造出精确的5‑阶宏观模型，能够完整预测训练过程。

**💡 创新点**

创新点在于：①将学习算子定义为宏观动力学核心，给出学习算子下降定理、精确闭合定理和粗粒化定理；②证明在固定激活模式下，ReLU网络训练可被完全描述为仅包含五个时刻的闭合微分方程；③首次实现对有限宽网络训练过程的精确宏观预测，显著优于传统的冻结NTK近似。

**🔧 技术方法**

使用技术包括：动力学系统理论、Lie导数、宏观投影与矩方法、梯度流分析、数值积分、Grönwall不等式、误差传递分析等。

**📊 数据集**

数据集：仅采用人工生成的三组训练样本（x = (-1, 0, 1)，y = (0.65, -0.35, 0.55)）以及随机初始化，未使用公开大规模数据集。

**📈 对比分析**

比较方法：将15‑阶宏观方程的训练损失、输出及学习算子与全参数梯度流（720维）进行并行积分比较。宏观模型在数值误差上达到 ~10⁻¹³，远优于冻结NTK近似；此外，宽度一致性实验显示误差随网络宽度 m 以 m⁻¹/² 缩放，验证模型在宽度趋近时的稳定性。

**⚠️ 局限性**

局限性：①仅在激活门不变（固定模式）下可闭合，无法处理门事件；②对ReLU等分段线性激活有效，对光滑激活（如 GELU、SiLU）尚无通用闭合；③宏观状态维度随激活模式数增大，难以在更大网络或更一般任务中实现显著压缩；④假设参数初始化满足独立性且梯度流连续，可能不适用于所有优化策略。

---

## 83. Any3R: Feedforward 3D Reconstruction from Any Camera Type

**arXiv ID:** 2610.36150 | [PDF](https://arxiv.org/pdf/2610.36150v1)

**作者:** Jingong Chen `[一作]` (South Dakota State University), Kyle Gao `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种全新的 Feed-forward 3D 重建模型 Any3R，能够从任意混合的中心摄像机（包括针孔、鱼眼和全景）图像中同时预测场景几何和相机位姿。

**💡 创新点**

创新点在于将稠密射线场作为角度编码和损失权重的输入，采用角度坐标和固角加权，消除了对针孔假设的依赖，使单一网络可处理所有中心摄像机。

**🔧 技术方法**

使用 Transformer 结构的跨帧注意力网络，配合角度坐标映射、固角权重和距离解码器，并在训练时混合多种摄像机模型进行无监督学习。

**📊 数据集**

在 Matterport3D、Stanford2D3DS、PanoCity、DL3DV 视频以及 ScanNet++、FIORD 等真实鱼眼数据集上进行训练和评估，涵盖13种中心摄像机模型。

**📈 对比分析**

与现有的相机感知方法（Wid3R、OmniVGGT、PanoVGGT、MapAnything 等）以及几何基础模型（π^3、VGGT 等）对比，Any3R 在所有摄像机类型上实现了最高的重建精度和位姿准确率，尤其在宽视角和全景镜头下优于专用模型。

**⚠️ 局限性**

局限性在于仅支持中心摄像机，需要预先提供射线场；在缺失射线信息时性能显著下降，并且在极窄视角或非中心光学系统时效果尚未验证。

---

## 84. They Will See Johnny's Message: A Measurement Study of Post-Quantum Cryptography Across Email Delivery Paths

**arXiv ID:** 2610.36145 | [PDF](https://arxiv.org/pdf/2610.36145v1)

**作者:** Jaehwan Park `[一作]` (University of Tennessee), Doowon Kim `[通讯]` (University of Tennessee)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模服务器探测、受控客户端和邮件服务器实验以及OpenPGP/S/MIME后量子实现的兼容性测试，系统评估了邮件服务各阶段（提交、转发、访问、Webmail）的后量子密钥交换部署与实际使用情况。

**💡 创新点**

创新点在于：①整合服务器端PQC可用性、客户端协商行为和消息级加密互操作的三维测量；②基于Tranco Top 1M域名进行三次快照持续观察；③构建19个OpenPGP密钥与6个S/MIME证书的结构化语料库用于兼容性测试；④公开发现Exim TLS配置缺陷和GnuPG PKESK编码错误。

**🔧 技术方法**

采用TLS 1.3的混合/纯ML‑KEM组探测（OpenSSL 3.5扫描）、邮件客户端抓包、Postfix/Exim配置实验、OpenPGP/S/MIME后量子实现（GnuPG、PGPony、Bouncy Castle等）以及自定义语料库匹配与手工验证。

**📊 数据集**

使用的数据集包括：Tranco Top 1M域名（2026‑07‑26）三次快照；各域名的MX、SRV、Autoconfig等邮件服务器地址；19个OpenPGP密钥与6个S/MIME证书的语料库；八个大型邮件发送服务（Google、Yahoo、Proton等）的账号与服务器。

**📈 对比分析**

方法：通过三快照比较服务器PQC支持率；抓取客户端握手记录统计PQC协商情况；记录消息加密导入、加密、解密的成功/失败，分析导入失败原因。结果显示：PQC支持率低于50%，Google占主导；多数客户端/服务器无法协商或在PQC受限时回退到明文；消息层大部分密钥导入或解密失败。

**⚠️ 局限性**

局限性：仅评估基础设施而非真实流量；基于排名域名，未涵盖实际用户分布；受测客户端、MTA与密钥语料有限，难以推广到全体；未考虑MTA‑STS/DANE等策略对PQC的影响；实验时长短暂；本文聚焦技术障碍，未探究组织层面的采用原因。

---

## 85. When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths

**arXiv ID:** 2610.36134 | [PDF](https://arxiv.org/pdf/2610.36134v1)

**作者:** Junchen Li `[一作]` (University of Electronic Science and Technology of China), Shuang Liang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Contrastive Evidence Paths (CEP) 和 Structured Corpus Interaction (SCI)，帮助搜索代理在有限观察窗口内更有效地检索和验证多文档证据

**💡 创新点**

CEP通过最强路径支持与背景访问分布校准，消除图连通性导致的排名偏差，并保留源路径；SCI利用CEP输出的路径构建结构化地图并支持锚点扩展，以提升证据探索效率

**🔧 技术方法**

利用实体图构建文档图、改进的语义+词汇检索、Personalized PageRank背景校准、最大路径动态规划、图搜索工具（graphsearch、graphexpand）以及LLM驱动的交互式代理

**📊 数据集**

在 BrowseComp‑Plus（Web 100K文档、830 题）和 TREC‑COVID（171K 科研文献、50 题）上进行实验，并对 BrowseComp‑Plus 进行 900K 文档扩展测试

**📈 对比分析**

与传统检索代理（Retrieval‑Agent、NeMo Agent）、Corpus‑Interaction 代理（DCI、DR‑DCI、RISE、RARG++、AtlasNav）以及基于图的 RAG 方法对比；SCI 在 BrowseComp‑Plus 上最高精度（~95%）、最少工具调用（<20%）与最低生成成本；在 TREC‑COVID 上 nDCG@10 达到 81.77%，比 RARG++ 高 5.8pp，且工具调用与成本下降 50%+

**⚠️ 局限性**

仍受图构建质量与实体抽取错误影响，路径长度限制（H=3）可能限制跨文档深度，扩展与搜索工具的计算成本随文档规模增长仍有上升，且在极大语料下准确率仍随规模下降

---

## 86. Instruction Capacity: Behavior and Degradation of Language Models under Controlled Load

**arXiv ID:** 2610.36095 | [PDF](https://arxiv.org/pdf/2610.36095v1)

**作者:** Hennadii Kutomanov `[一作]` `[通讯]`, Hennadii Kutomanov

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于单词替换的原子指令测量仪，利用该仪器在不同负载（规则数、文本长度、词位、规则编号、词出现次数等）下对语言模型的指令执行概率进行统计测量，并构建随机效应logit模型，最终定义并计算出带有不确定度的指令容量（Instruction Execution Capacity）。

**💡 创新点**

创新点：
• 采用单词替换作为完全可计算、独立、无语义干扰的原子指令，解决了现有评测方法的不可比对、语义混杂等问题。
• 通过构建随机效应的logit模型并引入“shift correction”，消除对参考编码的依赖，得到可解释、可比的指令容量。
• 引入“passport”概念，将所有测量规律、参数及不确定度以公开形式记录，为后续模型比较与标定提供统一尺度。
• 开放完整的测量仪器、代码和数据，支持在不同模型上重复实验。

**🔧 技术方法**

技术手段：
• 任务生成与规则设计采用确定性伪随机数种子，保证可复现。
• 通过逐行匹配验证答案，得到二元伯努利试验结果。
• 对试验结果使用带随机效应的logit回归（GLMM），采用分箱（以对数尺度划分）估计各负载轴的贡献。
• 通过最小距离估计拟合平滑规律，随后进行shift correction，将模型级别与参考级别对齐。
• 采用δ方法计算指令容量的不确定度，并通过自适应规划选择测量点。

**📊 数据集**

数据集：
• 所有任务均为合成，词汇表固定为2220个单词（无语义、单token）。
• 2958个请求覆盖1890个负载点，文本长度从20到10240词，规则数从1到256，总计642,430个替换试验。
• 训练与评测使用GPT‑4o mini（上下文窗口128k，回答上限16k）。

**📈 对比分析**

比较方法与性能：
• 通过定义指令容量（在95%与50%正确率阈值下的规则数）与其他模型的“passport”进行比较。
• GPT‑4o mini的指令容量为：在95%正确率下约37–38条规则，在全替换密度下约18–19条规则。
• 与IFScale等传统评测相比，本方法在多维负载轴上提供光滑、可解释的衰减规律，且带有置信区间，可在不同模型间直接比较。

**⚠️ 局限性**

局限性：
• 只评估单词替换这类无语义、无结构的原子指令，无法直接反映模型对复杂自然语言指令的处理能力。
• 依赖于固定词汇表和单token替换，可能与真实应用场景的词形变化、句法约束不一致。
• 受模型最大文本/回答长度限制，导致在极大负载下无法完全覆盖所有轴的范围。
• 统计模型假设各轴影响可加性且相互独立，实际可能存在未建模的交互效应。
• 仅在GPT‑4o mini上进行测量，尚未验证跨模型迁移的稳定性。

---

## 87. FinLLM-KG: Knowledge Graph-Enhanced LLMs for Financial Sentiment Analysis in Algorithmic Trading

**arXiv ID:** 2610.36083 | [PDF](https://arxiv.org/pdf/2610.36083v1)

**作者:** Giorgos Iacovides `[一作]`, Danilo Mandic `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了FinLLM-KG框架，在推理时将已公开的SEC 10‑K财务知识图与金融新闻结合，以增强金融情感分析。

**💡 创新点**

创新点在于将时间对齐的公司知识图与冻结的因果LLM结合，既不更新模型权重，又能在不丢失可解释性的前提下为情感预测注入上下文。

**🔧 技术方法**

采用的技术包括FinReflectKG知识图、自然语言公司概况或相关三元组检索、因果LLM（如Claude Haiku 4.5）、logit‑to‑score 转换、Kernel SHAP 与 Shapley 解释，以及基于情感的长短仓组合构建。

**📊 数据集**

使用的数据集包括约20万条2015‑2021年财务新闻、700+公司 2014‑2024 年SEC 10‑K抽取的知识图、S&P 100 日收盘价及 NER 过滤后的实体标注。

**📈 对比分析**

与同一LLM的FinDPO基线对比，FinLLM‑KG 将累计回报从 275% 提升至约 330%，Sharpe 比例从 2.55 提升至 3.15，Sortino、Calmar 等风险调整指标亦显著改善，显示更高的收益与风险控制。

**⚠️ 局限性**

局限在于只能利用已公开的最新 10‑K 作为上下文；若无公开文件则缺乏知识；实体识别仍可能出现误差；实时更新受限，主要依赖离线检索；仅适用于因果LLM，无法直接扩展到自回归模型。

---

## 88. Rollback: Experience Backtracking in World Models for Policy Improvement

**arXiv ID:** 2610.36068 | [PDF](https://arxiv.org/pdf/2610.36068v1)

**作者:** Jiayi Luo `[一作]` (AgiBot), Guanghui Ren `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种利用世界模型的经验回溯搜索（experience‑backtracking tree search）来进行机器人策略改进的框架；通过在真实或想象的关键决策点回溯并分支替代动作，直接比较后续结果，从而得到细粒度的动作级别监督；同时构建了一个协同训练系统，使世界模型和策略在真实机器人执行的实时反馈中共同进化，实现了所谓的“测试时物理校准”。

**💡 创新点**

创新点主要包括：1）将世界模型从单向仿真器转变为可回溯的对比引擎；2）利用树结构的价值估计，避免额外的价值网络，直接从子树成功率得到相对优势标签；3）实现世界模型与策略的实时协同训练，通过LoRA微调和分布匹配蒸馏（DMD）实现测试时物理校准；4）在分布式异步框架下，将搜索、政策训练和模型更新解耦，充分利用GPU与机器人资源。

**🔧 技术方法**

使用的技术包括：动作条件的视频世界模型（GE‑Sim 2.0）、蒙特卡洛树搜索（MCTS）与渐进加宽、基于进化策略的政策网络（π_0.5）与优势条件训练、LoRA微调、分布匹配蒸馏（DMD）、奖励模型（world judge）、以及分布式异步训练架构。

**📊 数据集**

主要数据集：1）基于AGIBOT Genie‑02机器人的三项真实任务数据——块提取、球投掷和台面曲棍；2）离线演示视频与遥操作轨迹用于预训练；3）在每个任务上采集的成千上万帧图像与关节状态；4）手工标注的成功/失败标签用于奖励模型训练。

**📈 对比分析**

与基线方法（SFT、RECAP、RISE+GE‑Sim2）在同一计算预算下进行比较。结果显示，本文方法在三项任务中分别比RISE提升绝对10.67%、11.11%、11.85%的成功率，整体平均提升约24%；并在球投掷与台面曲棍任务中，8小时训练后超过人类玩家的平均成功率。

**⚠️ 局限性**

局限性包括：1）世界模型的预测误差仍随延迟增长，需频繁校准；2）奖励模型的误判可能导致相对优势标签失真；3）方法对GPU资源高度依赖，异步调度复杂；4）目前仅在视觉+关节观测的三任务上验证，缺乏跨任务通用性；5）对极端高维动作空间的扩展尚未充分验证。

---

## 89. Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It

**arXiv ID:** 2610.36184 | [PDF](https://arxiv.org/pdf/2610.36184v1)

**作者:** Sanjid Hasan `[一作]` (Khulna University of Engineering and Technology), Bayazid Hasan Simon `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了885小时、17个域、含说话人分离的弱监督孟加拉语语音语料库Lipi-Ghor，并公开了构建流程和金标评测集。

**💡 创新点**

创新点在于：①采用域感知频道手工挑选，实现低资源语言大规模语音采集；②将YouTube字幕与自动说话人分离模型结合，生成SSTT（Speaker–Speech–Transcription–Timestamp）注解；③发布人类校正的金标评测集，用于评估字幕噪声与ASR性能。

**🔧 技术方法**

使用的技术包括：Python+Google/YouTube API筛选字幕视频、ffmpeg提取音频和字幕、Precision‑2（pyannote‑AI）进行说话人分离、自动对齐字幕生成SSTT、Bengali文本规范化和ASR模型评测（Hishab TITU、Meta MMS、Seamless M4T、Whisper）。

**📊 数据集**

主要数据集为：Lipi-Ghor 885h 17域，包含原始视频ID列表、自动分离输出、SSTT注解以及500句手工校正的金标评测集；基准比较使用公开的Bengali ASR模型和微调版Fast Conformer/Faster Whisper。

**📈 对比分析**

通过零射击和微调评估，零射击模型WER在30.5%–84.5%之间，表现随域（情感、方言、长句）差异显著；微调模型在全局WER降至约20%–25%，验证Lipi-Ghor对ASR预训练的有效性。

**⚠️ 局限性**

局限包括：字幕来源噪声导致的转录错误；说话人分离为自动化，未人工验证；方言覆盖不足；数据重现受YouTube API限制；未提供原始音频，需自行下载。

---

## 90. ML-OLLA: Link Adaptation for Real-Time Traffic in 5G/6G NTN Systems

**arXiv ID:** 2610.36147 | [PDF](https://arxiv.org/pdf/2610.36147v1)

**作者:** Artur Magamedov `[一作]` (HSE University), Evgeny Khorov `[通讯]` (HSE University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对5G/6G NTN实时流量的多层外循环链路自适应（ML‑OLLA）算法，能够为每次传输尝试设置不同的目标误块率并自适应选择调制编码方案。

**💡 创新点**

创新点在于：①为每一次HARQ重传独立设置目标BLER，早期尝试可使用更高速率MCS，重传时更保守；②通过滑动窗口估计MCS误码概率，采用数据驱动方法自动求解最佳目标BLER和OLLA步长；③在卫星与地面gNB部署场景中均验证其性能。

**🔧 技术方法**

使用的技术包括：内部链路自适应（ILLA）基于CQI测量；外循环链路自适应（OLLA）多级实现；数据驱动参数选择；ns‑3仿真平台与5G‑LENA模块；3GPP NTN信道模型TR 38.811。

**📊 数据集**

使用的“数据集”为ns‑3仿真生成的CQI报告与HARQ反馈序列（滑动窗口2000个样本），并基于3GPP定义的MCS与CQI映射表进行统计。

**📈 对比分析**

与传统OLLA（单一目标BLER）、基于SINR自适应的OLLA以及始终使用最稳健MCS的基线进行对比。结果表明，在相同可靠性与延迟预算下，ML‑OLLA可使系统容量提升至四倍，最低支持延迟从60 ms降至40 ms（PLR 10⁻³）或从30 ms降至25 ms（PLR 10⁻²），并且卫星端gNB部署进一步降低延迟。

**⚠️ 局限性**

主要局限：①未显式考虑HARQ‑CC增益，误码概率估计假设各尝试独立；②参数搜索为穷举，适用于有限MCS集合；③仿真基于理想化的NTN信道模型，实际环境中卫星轨道、相位噪声等因素可能影响性能。

---

## 91. SpatialPC: Deep Computation with Sublinear Local Credit Propagation

**arXiv ID:** 2610.36139 | [PDF](https://arxiv.org/pdf/2610.36139v1)

**作者:** Benjamin Prada `[一作]` (University of South Florida), Ankur Arjun Mali `[通讯]` (University of South Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究局部学习规则在空间子strate上的信用传递延迟，提出理论下界并设计编译器实现最优延迟，构建 SpatialPC 预测编码框架以有限迭代训练网络。

**💡 创新点**

证明对任意 d 维常密度子strate，损失信息到达所有参数至少需 Ω(N^{1/d}) 轮；构造单元范围、度数为 2d 的蛇形/莫尔编译器实现该下界；提出通过邻域可逆预测动态 (APD) 在有限轮次下获得精确梯度，并通过有效信用半径分析区分传播时间与信用精度。

**🔧 技术方法**

光锥下界证明、蛇形/莫尔空间填充曲线编译、空间预测编码 (SpatialPC) 与 adjoint prediction dynamics (APD)、梯度相似度评估与残差惩罚。

**📊 数据集**

MNIST 与 Fashion‑MNIST 数据集。

**📈 对比分析**

在 N≤256、T=2√N 的预算下，SpatialPC 的梯度与已收敛梯度余弦相似度 >0.9，测试准确率达到 97.6% (MNIST) 与 88.1% (Fashion‑MNIST)，仅略逊于深度128的反向传播基准；链型控制实验验证延迟对性能的影响。

**⚠️ 局限性**

未证明莫尔编译器在所有 k 下的 D_reach 为 √N+2；有效信用半径 R_ε 的 N 规模增长未知；实验仅在 2D 平面网格上验证，3D 堆叠、卷积或序列任务及真实硬件实现仍待探索。

---

## 92. ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling

**arXiv ID:** 2610.36126 | [PDF](https://arxiv.org/pdf/2610.36126v1)

**作者:** Kehinde Elelu `[一作]` (Michigan State University), Ebuka Okpala `[通讯]` (Clemson University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种名为 ConformalMOS 的框架，利用分割共形预测和序数化标签来同时输出 MOS 的点估计和统计显著的置信区间；

**💡 创新点**

其创新点在于将分割共形预测引入 MOS 预测，并采用高斯平滑的序数标签训练，使得模型既能保持标签秩序，又能提供正式覆盖保证的预测区间；

**🔧 技术方法**

使用的技术包括预训练音频编码器（M2D2 或 Wav2Vec2）、轻量级两层 MLP 预测头、Gaussian 标签平滑、KL 与 L1 损失结合训练，以及后期的分割共形校准；

**📊 数据集**

实验数据集为 BVCC 挑战集，共 7,106 条英语 TTS/VC 语音样本；

**📈 对比分析**

与 UTMOS、FUSE‑MOS 等基线在 BVCC 上对比，系统级 MSE 降低 7%（从 0.19 降至 0.08），LCC 提升，且在 α=0.05 时实现近 95% 的实证覆盖率，区间宽度相对紧凑；

**⚠️ 局限性**

局限性包括仅在 BVCC 内部数据上验证，跨域鲁棒性尚未测试；基线模型的选择对结果影响显著（Wav2Vec 版本性能不佳）；以及在低样本或高噪声场景下的准确性和不确定性估计仍需进一步改进。

---

## 93. VISTA: Differentiable Viewpoint Planning for Long-Horizon Task Visualization

**arXiv ID:** 2610.36111 | [PDF](https://arxiv.org/pdf/2610.36111v1)

**作者:** Donggang Jia `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可微分的视角规划框架 VISTA，用来为长时任务自动生成连贯且信息丰富的摄像机轨迹，帮助人类或智能体理解任务过程；

**💡 创新点**

创新点包括：① 将摄像机控制建模为任务条件的可微分优化问题；② 设计并融合五个视角质量指标（可见覆盖、补丁熵、功能对齐、上下文聚焦、时间连续性）形成统一目标；③ 支持对象部件与关系目标的双种视角配置，实现任务动态视角自适应；

**🔧 技术方法**

技术手段包括：可微分光栅化（nvdiffrast）实现软像素可见性；梯度优化（Adam）在多起点下求解连续视角参数；使用 VLM（GPT‑5.4）提取语义前后方向；Bézier 曲线估计视角路径长度实现时间连续性；多尺度 LOD、软属性映射实现分割与补丁信息；

**📊 数据集**

数据集：RealAppliance（家电模型，如空气炸锅、面包机、烤箱）以及 Chevrolet C8 车辆和 Forklift。

**📈 对比分析**

比较方法：对比固定摄像机、视角熵、人工偏好等基线，用户研究中 VISTA 在六项评价指标上均显著优于基线（平均 4.53/5）且 96% 的任务排名第一；VLM 视觉基准中，VISTA 在状态识别、目标决策和闭环控制任务中分别提升至 69.3%、79.3% 和 65.7%（相较于 48%–54% 的基线提升 2.1 倍）。

**⚠️ 局限性**

局限性：① 依赖 VLM 提取的前后方向，若不准会导致功能对齐失效；② 目前仅支持对象部件与关系目标，缺少更丰富的动作语义（如插入、倒液等）；③ 可微分可见性局限于局部，无法突破全局遮挡导致的可视性盲区；④ 目标优化为非凸，仍可能陷入局部最优；⑤ 对称或几何均匀的对象、薄结构和细小功能特征的表示不够充分。

---

## 94. Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach

**arXiv ID:** 2610.36110 | [PDF](https://arxiv.org/pdf/2610.36110v1)

**作者:** Victor Alejandre `[一作]` (Universidad Politécnica de Madrid), Concha Bielza `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于Dirichlet过程混合模型（DPM）的新型条件概率分布（CPD）建模方法，并将其集成到半参数贝叶斯网络（SPBN）框架中。

**💡 创新点**

通过DPM实现数据自适应聚类，提供更紧凑的表示，降低推理成本，同时提高密度估计性能，克服了传统核密度估计（KDE）的局限性。

**🔧 技术方法**

使用了贝叶斯非参数Dirichlet过程混合模型（DPM）和变分推理方法（CAVI、GPCAVI和Memoized）。

**📊 数据集**

在合成数据和真实数据集上进行了评估，使用了来自UCI机器学习库的多个真实数据集。

**📈 对比分析**

与基于KDE的SPBN和最先进的贝叶斯网络模型相比，DPM-SPBN在密度估计和结构学习中表现显著优越，尤其在合成数据上实现了更低的绝对对数似然误差。

**⚠️ 局限性**

在计算复杂性和推理时间上，DPM-SPBN的训练时间较长，但推理时间显著降低，尤其在样本量较大时，Memoized方法的计算成本随着样本和节点数量的增加而显著上升。

---

## 95. Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via the Model Context Protocol

**arXiv ID:** 2610.36108 | [PDF](https://arxiv.org/pdf/2610.36108v1)

**作者:** Yuepeng Hu `[一作]` (Duke University), Jian Du `[通讯]` (TikTok)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在模型上下文协议（MCP）中通过注入工具描述实现的后续日志记录攻击。

**💡 创新点**

创新点在于提出四要素模板（Trigger、Tool Binding、Justification、Pressure）并系统化评估其对攻击成功率和负载完整度的影响。

**🔧 技术方法**

采用了自然语言注入、结构化工具元数据操纵、Token级别攻击评估以及多模型多服务器的实验技术。

**📊 数据集**

使用了自定义的GitHub、Mapbox、PayPal三大MCP任务集共555个查询，并包含MCP‑Universe的Playwright和YFinance五个任务集。

**📈 对比分析**

与传统prompt‑inject和组合攻击对比，使用ASR、LA、TCR、MCR和Token变化等指标；在18个兼容模型–服务器组合中攻击成功率达62.6–100%，日志完整度76.5–95.6%，对目标任务影响最大6.6个百分点。

**⚠️ 局限性**

局限性包括仅评估了有限模型、服务器和任务，攻击依赖于攻击者控制的MCP服务器，实验环境与真实部署差异；现有防御方法无法完全消除攻击且可能误伤合法描述。

---

## 96. CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction

**arXiv ID:** 2610.36100 | [PDF](https://arxiv.org/pdf/2610.36100v1)

**作者:** Neil Chen `[一作]` (University of California, Berkeley), Yang Yang `[通讯]` (University of California, San Francisco)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一个可进行局部语义更新的临床队列生成代理，支持在执行过程中对合格标准进行自然语言纠正并保持部分已完成计算

**💡 创新点**

通过Task Delta局部更新、语义指纹与依赖感知的失效与重用机制，实现了在不完全重启的情况下进行多步纠正，显著降低纠正延迟和模型调用

**🔧 技术方法**

使用LLM（Qwen3‑8B）、语义指纹、DAG依赖、增量解析、MIMIC‑IV概念映射、混合检索与跨编码reranker，并提供并行与顺序调度

**📊 数据集**

使用MIMIC‑IV数据库进行概念解析与查询，基于CohortX Task 1训练集生成100个临床试验合格标准场景（共126个纠正回合）进行实验，Stage‑2使用MIMIC演示可执行性

**📈 对比分析**

与全重生成基线比较，语义微F1相差+0.6%（不显著），但post‑completion纠正延迟降低64.8%，mid‑run降低44.3%，LLM调用减少38.1%，总token略高；后端演示可执行并实现部分重用

**⚠️ 局限性**

存在Boolean逻辑准确率低、澄清效果差；仅使用单一8B量化模型；纠正轨迹人为生成，未采集真实临床交互；Stage‑2仅为概念验证，未评估完整笔记检索与更大模型的影响

---

## 97. Minimax-Optimal Small-Loss Bounds for Online Linear Optimisation on $\ell_p$ Balls

**arXiv ID:** 2610.36097 | [PDF](https://arxiv.org/pdf/2610.36097v1)

**作者:** Jack J. Mayo `[一作]` `[通讯]` (Kurtos.ai), Jack J. Mayo (Kurtos.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对在 ℓ_p 球（2≤p<∞）上的全信息在线线性优化问题，设计了两种自适应的 FTRL 算法：一种是单一的“standalone FTRL”，在累计偏移损失达到阈值时从径向正则化切换到二次正则化；另一种是“Switching Extension”，在跟随 FTL 与单一 FTRL 之间按阈值切换，以保持对 FTL 的两倍 regret 上界。两种算法均不需要预先知道时限或累计滑动损失，且在任何损失序列上实现了最优的 minimax 小损失 regret 速率 O_p(1+min{Δ_T^{1-1/p}, d^{1/2-1/p}√Δ_T}) ，并给出了 p→∞ 时的精确首项系数 (2+o(1))p/ln p；此外，对于 i.i.d. 损失，作者计算了期望 regret 的精确 log T 系数，区分了 FTL 的方差项与正则化产生的偏差。

**💡 创新点**

创新点在于：① 引入“shifted‑loss stability”概念并构造了可被 FTRL 以单一次正则化切换的 bounded‑radial 以及二次扩张正则化；② 通过精细的 Bregman 散度分析和曲率估计，实现了对小损失的最优常数 2（在 p→∞ 时）；③ 提出了一种无重启、仅依赖累计损失的切换策略，兼具小损失最优性和对 FTL 的两倍上界；④ 在 i.i.d. 情况下给出了完整的 log T 系数，首次将方差与正则化偏差分离。

**🔧 技术方法**

技术手段主要包括：基于支持函数的平滑视角、潜能分解（FTRL）、径向正则化与二次正则化的构造、shifted‑loss 稳定性证明、Bregman 散度的段内收缩性质、曲率与约束边界分析、以及 FlipFlop 风格的切换测试。所有理论均在全信息设置下严格证明，且使用了闭式径向动作与一维根求解实现可计算性。

**📊 数据集**

本研究为理论工作，未使用任何实际数据集；所有结果均为对任意满足 h(θ_t)≤1 的损失序列的通用上界与下界。

**📈 对比分析**

与现有方法的比较主要基于理论 regret 上界：与 Adaptive Regularisation、Norm Perturbations、MetaGrad 等方法相比，本工作在小损失下实现了更优的常数（2+o(1))p/ln p；与 FTL 相比，Switching Extension 保证了 ≤2×FTL 的路径级上界；下界证明表明常数 2 是不可改进的。

**⚠️ 局限性**

局限性包括：① 对平方根分支的常数尚未得到完整精确值；② 结果仅适用于全信息反馈，Bandit 反馈下缺乏相应的切换/正则化策略；③ 对一般对称凸体的可计算 bounded 正则化是否仍能保持小损失最优性仍未知；④ 算法实现需要对二次正则化求解一维根，虽然可行但在高维近边界时可能需要更高精度与计算复杂度分析。

---

## 98. Closed-Loop Aware Planning via Imitation-Free Speed Scoring for End-to-End Autonomous Driving

**arXiv ID:** 2610.36079 | [PDF](https://arxiv.org/pdf/2610.36079v1)

**作者:** Emad Aghajanzadeh `[一作]` (Technical University of Munich), Ulf Schlichtmann `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出CLAP-AD，一种可插拔的规划头，通过先预测路径再对多种速度类进行多准则评分，实现了端到端驾驶模型在不需速度条件训练的情况下在测试时可控地选择速度，并显著提升了安全性和可控性

**💡 创新点**

将路径与速度分别以模仿学习与多准则评分两种目标训练；引入基于物理安全约束的多准则速度评分机制，支持部署时的速度自定义；实现了对现有E2E驾驶模型的无缝集成

**🔧 技术方法**

路径预测模块（GRU/多查询网络）、速度评分分支（碰撞、红绿灯、停牌等多准则），以及分离的查询与注意力机制；使用阈值化/最优速度选择实现可控速度；PID控制器实现低层驱动

**📊 数据集**

Bench2Drive、Bench2Drive-Speed、NAVSIM v2（含真实驾驶数据）以及训练用的Think2Drive、LEAD等专家示范数据

**📈 对比分析**

与UniAD、VAD、TFv6等基线比较，CLAP-AD在Bench2Drive-Speed上实现82.53的Speed‑Adherence（比TFv6高12.9点），在Bench2Drive上提升DS +28.5（UniAD）/+32.7（VAD），在NAVSIM v2上提高EPDMS +6.1，整体性能显著优于或接近最强基线

**⚠️ 局限性**

仍受限于速度类数量和准则设计的全面性；在规则式专家监督下提升有限；需要更丰富的安全准则和更长时域的动态规划来进一步提升鲁棒性

---

## 99. Exploratory creative pre-ranking: Hierarchical empirical-Bayes Thompson sampling for real-time creative optimization

**arXiv ID:** 2610.36020 | [PDF](https://arxiv.org/pdf/2610.36020v1)

**作者:** Jiale Han `[一作]` (UCLA), Xiaowu Dai `[通讯]` (UCLA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种在广告召回与排序之间的创意预排序器，使用层级经验贝叶斯后验和单次汤普森抽样进行探索；

**💡 创新点**

创新点在于：1）无参数层级归约式经验贝叶斯权重，自动决定各层级的冲刷力度；2）仅使用单属性计数重构每段落的后验分布，避免高维计数；3）通过单次抽样实现低成本的探索；

**🔧 技术方法**

采用层级随机效应模型、经验贝叶斯收缩、Beta–Bernoulli Thompson Sampling 与线性对数几率重构；

**📊 数据集**

使用 Roblox 赞助广告真实流量日志进行重放实验，并在实时系统中部署 A/B 测试；

**📈 对比分析**

与随机短名单、单抽样、均匀分配等基线对比，提升预估播放率约 8.14%，占全部潜在提升的 44%，且增加的服务时延仅 0.2%；

**⚠️ 局限性**

局限性包括：对高维属性共线性假设仍有限，需在实时环境下保持计数准确；模型依赖层级结构的可用性；在极低数据量下仍受先验约束；

---

## 100. MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine

**arXiv ID:** 2610.36006 | [PDF](https://arxiv.org/pdf/2610.36006v1)

**作者:** Simon A. Lee `[一作]` (UCLA), Jeffrey N. Chiang `[通讯]` (UCLA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了系统一模型MedJev，用于医学知识问答与患者级EHR预测任务；

**💡 创新点**

将系统一请求接口统一应用于多种医学任务，展示零射击性能与局部校准的可行性；

**🔧 技术方法**

采用OpenJev的Jev接口、系统一推理、温度标定与Platt标定技术；

**📊 数据集**

使用MedQA、USMLE、NEJM、DiagnosisArena、MIMIC‑IV、MIMIC‑IV‑ED等医学数据集；

**📈 对比分析**

与基准LLM、前沿GPT、XGBoost、TF‑IDF等模型对比，零射击下MedJev在知识问答上接近GPT‑4，在部分EHR任务表现与监督模型相当，局部标定显著提升校准；

**⚠️ 局限性**

受限于样本量、提示/序列化方式、单一机构数据以及潜在预训练泄漏，缺乏跨机构泛化。

---

## 101. How Learning Selects Equilibria and When It Leads to Collusion

**arXiv ID:** 2610.36306 | [PDF](https://arxiv.org/pdf/2610.36306v1)

**作者:** Hanzhang Wang `[一作]` (University of California, Berkeley), Manxi Wu `[通讯]` (University of California, Berkeley)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在无限期折扣马尔可夫博弈中，通过学习动态中的假设检验与价值相关探索来实现均衡选择，并将结果应用于有限公共记忆的重复博弈与Cournot竞争。

**💡 创新点**

创新点在于将探索规则与可被选均衡的最小变换价值关联，揭示学习设计可诱导协同结果，并将GLRT用于假设检验，扩展了先前有限博弈的结果。

**🔧 技术方法**

采用了基于平滑最优反应的周期学习动态、一般化似然比检验、随机稳定性（Young框架）和马尔可夫游戏理论。

**📊 数据集**

该工作为理论研究，不使用实验数据集；若有数值验证则基于标准Cournot模型和随机信号噪声。

**📈 对比分析**

通过理论证明与符号比较，显示更长的公共记忆能提高协同指数；相较于传统纳什均衡，学习动态导致的结果更具收益且可逼近垄断水平。

**⚠️ 局限性**

局限在于需要完整模型信息、协调的学习周期与重启、以及随公共记忆增长导致的状态空间指数级膨胀，难以实现去中心化与可扩展化。

---

## 102. Deep Learning Volumetric Segmentation for Prediction of Cavity Geometry in Underground Jet Boring Mining

**arXiv ID:** 2610.36303 | [PDF](https://arxiv.org/pdf/2610.36303v1)

**作者:** Giovanni Zingaro `[一作]` (University of Waterloo), Robert Gracie `[通讯]` (University of Waterloo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于三维 U‑Net 的深度学习模型，利用多模态体素化数据对 Cigar Lake 尖钻采矿的高压水射流切割洞口进行体素级语义分割预测，并构建完整的预处理与模型训练流程。

**💡 创新点**

创新点在于：① 将钻探记录、射流参数、背填信息和地质块模型统一投射到同一体素网格；② 在 U‑Net 中加入残差背填分支和集成预测策略；③ 采用 Integrated Gradients 对模型进行可解释性分析，揭示各输入通道对预测的空间贡献。

**🔧 技术方法**

使用技术包括：三维 U‑Net 编码器‑解码器网络、残差背填分支、组合 soft IoU + BCE 损失、数据增强（旋转、翻转、噪声）、Adam + cosine 预热学习率、温度缩放集成、Integrated Gradients XAI。

**📊 数据集**

使用的数据集来自 Cameco Cigar Lake 采矿项目，包括：C‑ALS 点云扫描、地质块模型（RMR、U3O8、VWC、粘土、砷等属性）、钻探与射流记录以及邻近洞口的背填扫描，共 11 通道体素输入。

**📈 对比分析**

性能比较方法为将模型与专家设定的圆柱基线模型对比；在测试集上，模型获得 IoU 0.601±0.088、F1 0.747±0.077，分别比基线提升 22.4% 与 14.7%；精度高于召回，置信校准良好，平均吨位误差从 38.0% 降至 25.6%。

**⚠️ 局限性**

局限性包括：训练样本有限且主要为已背填洞口，导致对原生地小洞口预测不足；输入对齐误差和地质不确定性未被充分建模；集成与 XAI 的解释仍受数据稀缺和模型复杂性的影响。

---

## 103. BUSTA: Teaching Small Language Models to Reason About Binary Execution

**arXiv ID:** 2610.36288 | [PDF](https://arxiv.org/pdf/2610.36288v1)

**作者:** Ritika Poddar `[一作]` (MITRE Corporation), Michael Doyle `[通讯]` (MITRE Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

Fine‑tune 开源 LLM（BUSTA）使用 GRPO 强化学习，在逆向基本块执行任务上预测输入寄存器，从而实现低级二进制推理；

**💡 创新点**

通过针对性后训练使小型模型在不依赖大模型 API 的情况下，近乎达到前沿模型的知可寄存器准确率；

**🔧 技术方法**

采用 GRPO、LoRA、Pydantic schema、Megatron 以及 NeMo RL 库进行训练和奖励设计；

**📊 数据集**

利用 Assemblage 100 个 Windows PE 二进制，生成 287k 个基本块，划分为训练/验证/测试集；

**📈 对比分析**

与多款开源与前沿 API 模型对比，知可寄存器准确率从 73% 提升至 95%，平均 token 使用量显著降低；

**⚠️ 局限性**

局限在于数据集规模有限、奖励函数与任务定义可进一步改进、未覆盖多样化二进制与完整程序级推理。

---

## 104. ASPEN: Adaptive Semantic Reasoning via Command Persistence Prediction for Efficient Navigation VLAs

**arXiv ID:** 2610.36285 | [PDF](https://arxiv.org/pdf/2610.36285v1)

**作者:** Vieri N. Akustia `[一作]` (University of Houston), Bin Hu `[通讯]` (University of Houston)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为冻结的导航VLA设计了一种事件触发调度器ASPEN，通过预测前一个命令的持久性决定是否直接重用该命令，或重新调用VLA进行语义推理，从而减少不必要的推理调用。

**💡 创新点**

创新点在于：①利用命令持久性预测实现调度，①通过轻量级多模态预测器结合视觉、语言、执行历史和缓存VLA置信度来判断是否重用命令；②在冻结的VLA上进行跨数据集评估，证明方法可迁移。

**🔧 技术方法**

采用Frozen SigLIP视觉编码、Frozen Llama‑3语言编码、状态向量以及两层Transformer的交叉注意力结构构成轻量级预测器；实现事件触发调度逻辑；使用交叉熵训练并通过阈值校准决定重用阈值。

**📊 数据集**

训练使用R2R‑CE数据集，评估则在RxR‑CE val‑unseen数据集（1,223个测试案例）上进行，基线为冻结的NaVILA。

**📈 对比分析**

与FlashTrigger、SP‑VLA等方法对比，ASPEN在RxR‑CE val‑unseen上将VLA调用量降低9.5%，估算推理时间降低8.1%；成功率、OS、nDTW与基线基本相同；行动序列一致率为92.1%，略低于FlashTrigger的97.5%。

**⚠️ 局限性**

局限性包括：①对目标域的泛化无严格保证，阈值选取基于源域；②推理成本估算而非真实端到端测量，未测量物理机器人上的延迟与能耗；③在某些情况下命令持久性预测可能导致一致率下降，需权衡效率与准确性。

---

## 105. EVA-Flow: Learning Shared Chemistry for Unified Environment-Aware Molecular Conformation Generation

**arXiv ID:** 2610.36280 | [PDF](https://arxiv.org/pdf/2610.36280v1)

**作者:** Bing Yan `[一作]` (FAIR at Meta), Benjamin Kurt Miller `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种统一的环境感知分子构象生成框架EVA-Flow，能够在真空、蛋白配体、溶剂和晶体等多种环境下生成符合物理约束的构象；

**💡 创新点**

创新点在于将环境作为条件向量融入流匹配生成器，利用共享的化学知识和环境特定的编码实现跨环境迁移学习；

**🔧 技术方法**

技术包括变分自编码器（VAE）+流匹配（FM）解码器、SE(3)-等变换的图神经网络、可学习的谐波基分布与环境图编码器；

**📊 数据集**

使用多源预训练集（GEOM-QM9、GEOM-DRUGS、SPICE、OE62等）以及针对每个环境的微调集（PDBBind、Aquamarine、COD等）；

**📈 对比分析**

在四个环境下与单独训练模型和重新训练的ETFlow对比，EVA-Flow在低样本环境（Docking、Solvation、Packing）上显著提升Recall COV与AMR，整体性能优于基线；

**⚠️ 局限性**

局限性包括对未见环境类型的泛化能力尚未验证、对超大分子或复杂晶体结构的扩展有限，以及生成结果仍需实验或高精度模拟验证。

---

## 106. Cross-modal Steering Vectors Exist Across Both Image and Text Generation

**arXiv ID:** 2610.36269 | [PDF](https://arxiv.org/pdf/2610.36269v1)

**作者:** Donghao Zeng `[一作]` (Mila, McGill University), Benno Krojer `[通讯]` (Mila, McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究统一多模态模型中，利用在一种模态中发现的“steering向量”在另一种模态生成时实现零样本跨模态控制，验证跨模态向量的可迁移性。

**💡 创新点**

首次系统评估跨模态steering向量在文本↔图像两方向的零样本迁移效果，揭示跨模态向量在共享参数空间中的因果可控性与几何一致性，并提供跨模态控制的新思路。

**🔧 技术方法**

使用线性差分向量（steering向量）提取方法、层级向量归一化与强度调节，结合共享自回归生成架构，并以Gemini 3.7 Flash等LLM做自动评估判定。

**📊 数据集**

采用从Gemini生成的最小对照文本与基于图片编辑模型（Gemini Image/3 Pro、Reve）的图像对照对，覆盖八个概念（情感、年龄、清洁、混沌、尺寸、深度、空间、颜色）以及两种独特模态概念。

**📈 对比分析**

通过LLM判定器计算概念率、AUC和恢复率，对比同模态与跨模态向量。结果显示，在UniAR、Emu3.5和Liquid上，大多数概念的跨模态恢复率平均在0.8–1.1之间，部分概念甚至超过同模态效果，证明跨模态steering在多模态模型中能实现高效控制。

**⚠️ 局限性**

局限性包括：对某些概念（如空间方向）迁移效果弱；迁移性能受模型架构差异影响（Emu3.5表现最差）；仅适用于共享自回归权重的统一多模态模型；评估主要基于自动LLM判定，缺乏更细粒度的人工质量评估。

---

## 107. The Inbox as a Coordination Primitive for LLM Agent Societies

**arXiv ID:** 2610.36264 | [PDF](https://arxiv.org/pdf/2610.36264v1)

**作者:** Karthik Sreedhar `[一作]` (MIT Media Lab), Ramesh Raskar `[通讯]` (MIT Media Lab)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在大规模LLM代理社会中引入“收件箱”机制，探讨代理如何在有限的阅读与记忆资源下挑选和处理消息，以提升协调效率。

**💡 创新点**

提出将收件箱视为协调原语，并将其拆解为四个关键决策（读取、存储、发送者押金、回复内容），系统性评估其在不同规模、角色和流量下的影响。

**🔧 技术方法**

使用LLM（如gpt‑5.6‑terra等）对代理进行决策推理，结合模拟环境实现大规模代理社会（多达16,000个代理），并构建四类收件箱规则进行对比实验。

**📊 数据集**

实验数据基于自定义的模拟任务（会议调度与组建任务）与真实电邮网络（SNAP联系人图），不依赖公开标注数据集，而是生成内部模拟场景。

**📈 对比分析**

通过对比推送（push）与拉取（pull）模型下的不同收件箱规则，评估提案重复率、可达代理比例、组建成功率及Token成本等指标，结果显示拉取模型在高流量与大规模环境下显著优于传统推送，并且合适的规则组合能在不同情境下提升协调效果。

**⚠️ 局限性**

局限性包括仅在固定规则与模拟任务下评估，未考虑代理自适应策略；使用单一LLM提供者且共享模型判定可能忽略个体差异；仅探讨有限的回复词汇和收件箱规则，未覆盖更广泛的策略与跨任务验证。

---

## 108. EREBOTS 4.0: A framework for conversational assistants with A2A agents and MCP tools

**arXiv ID:** 2610.36221 | [PDF](https://arxiv.org/pdf/2610.36221v1)

**作者:** Elia Pacioni `[一作]` (HES-SO Valais-Wallis), Michael Ignaz Schumacher `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 EREBOTS 4.0，一个开源的多代理框架，支持将语言模型驱动的代理与确定性工具通过 A2A 与 MCP 协议组合，并提供身份、用户上下文、任务追踪等完整生态。

**💡 创新点**

创新点在于统一的标准化框架，能够跨主机无代码配置代理；A2A 服务器通过官方 TCK 100% 通过；通过追踪与上下文传播实现安全、可解释的协同。

**🔧 技术方法**

使用 Python、FastAPI、OpenAI‑compatible LLM API、MCP SDK、A2A 协议、SQLAlchemy、PostgreSQL/SQLite、Keycloak、Vue.js、Gradio、Docker Compose、Traefik 等技术栈。

**📊 数据集**

论文没有使用公开数据集，而是通过内置演示和示例代理进行验证。

**📈 对比分析**

通过自动化测试套件（567 条测试）和 A2A TCK（97/129 需求通过）验证框架功能，性能指标主要是零失败、兼容性高；未进行运行时性能基准。

**⚠️ 局限性**

局限性包括仅支持单一排队任务、同意标记未强制执行、跨实例任务同步缺失、代理间不支持协商、路由决策依赖模型且缺乏成功率评估。

---

## 109. Diagnosing and Correcting Bias in MLLM for Long Video Understanding

**arXiv ID:** 2610.36210 | [PDF](https://arxiv.org/pdf/2610.36210v1)

**作者:** Xusheng Liang `[一作]` (City University of Hong Kong), Jiawei Ma `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 S-MME 基准和 CLEAR 框架来诊断并缓解长视频多模态大型语言模型的快捷路径偏差

**💡 创新点**

创新点在于设计可诊断的快捷路径基准 S-MME，以及提出训练无关的 Counterfactual Long-video Evidence-Aware Reasoning（CLEAR）框架，通过对比视图和残差干预纠正偏差；同时在不增加模型参数的情况下实现推理时的偏差纠正

**🔧 技术方法**

技术包括：对比视图（question‑only、salient‑view）构造快捷路径；提取多层隐藏状态并计算残差；在晚期解码层注入残差干预；利用可访问隐藏状态的多模态 LLMs（如 Qwen3‑VL、InternVL3、LLaVA‑Video）进行推理

**📊 数据集**

使用数据集：S‑MME（自建的快捷路径诊断集）；公开长视频基准 Video‑MME、LongVideoBench、LVBench、MLVU；并在这些基准上对模型进行评估

**📈 对比分析**

对比方法：在公开基准上与未处理的开源模型和通过扩充帧数的 baseline 进行比较；CLEAR 在保持相同帧预算下平均提升 3.5–7.8% 的准确率，并显著降低 shortcut‑consistent 错误率（从 44% 降至 36% 左右）

**⚠️ 局限性**

局限性包括：需要模型可访问隐藏状态（不适用于所有闭源 LLM）；对帧采样仍有限，长视频中极端稀疏或噪声场景下效果仍有待提升；构造 S‑MME 需人工标注，耗时；在推理时引入了轻微的计算和延迟

---

## 110. Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study

**arXiv ID:** 2610.36208 | [PDF](https://arxiv.org/pdf/2610.36208v1)

**作者:** Meysam Varasteh `[一作]` (University of Colorado), Robin Burke `[通讯]` (University of Colorado)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文扩展了 LXR 解释方法，评估了多种基线（LF、Pop、LIME-RS、SHAP 等），并在 Top10 推荐项、流行度过滤以及不同推荐模型（MF、VAE）下探究其可解释性、覆盖率与最小扰动。

**💡 创新点**

创新点包括：① 将 LXR 训练至 Top10 推荐项；② 引入流行度过滤以降低扰动大小并分析其影响；③ 使用 LF（潜在因子相似度）作为基线并与 LXR 比较；④ 运用 RBO 评估扰动列表差异；⑤ 进行有限搜索评估 LXR 的最优性。

**🔧 技术方法**

采用的技术包括：学习式反事实解释器 LXR 与 LXR10；潜在因子相似度解释器 LF；基于流行度的 Pop 解释器；传统解释基线 LIME-RS、SHAP、ACCENT；使用矩阵分解（MF）和变分自编码器（VAE）作为推荐器。

**📊 数据集**

使用了 MovieLens 1M、Yahoo Music 以及 Pinterest 三个公开数据集，分别进行 80/20 划分并进一步抽取 10% 的训练集做超参调优。

**📈 对比分析**

通过 Coverage、MPNR（最小扰动大小）、RBO（扰动列表相似度）以及 AUC 等指标对比。结果显示：在 MF 推荐器下 LF 取得最高覆盖率和最低扰动；在 VAE 推荐器下 LXR10 表现最佳；总体而言不同方法呈覆盖率–扰动大小权衡，RBO 表明存在多种有效扰动方案。

**⚠️ 局限性**

局限性在于：① 解释扰动往往包含 5‑10 项以上，可能不易被用户接受；② 流行度过滤虽降低扰动大小但显著降低覆盖率；③ 最优扰动搜索仅在小规模数据上完成，缺乏对全局最优的证明；④ 缺乏用户研究验证解释的可用性与可理解性。

---

## 111. Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization

**arXiv ID:** 2610.36204 | [PDF](https://arxiv.org/pdf/2610.36204v1)

**作者:** Ege Demirci `[一作]` (University of California Santa Barbara), Ambuj Singh `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种三分支能量框架 Concorde，用以解决稀疏与稠密图的链接预测难题，融合不变谱锚点、混合曲率残差和非局部测地向量场，并以 N‑pair 排序损失进行端到端训练。

**💡 创新点**

创新点在于：①谱解耦假设，证明单一学习矩阵无法同时保持谱不变与破除图自同构；②将链接似然拆分为三种结构上相互解耦的能量分支；③使用可学习的混合曲率产品空间与特征感知 Ollivier–Ricci 约束，配合向量场引导跨拓扑缺口的非局部对齐。

**🔧 技术方法**

技术细节包括：k 步随机游走谱锚点、混合曲率（负曲率超球 + 正曲率球面）嵌入、特征驱动的 Ollivier–Ricci 近似、Lorentz 模型下的切向量引导、指向性测地距离评分、门控融合以及 N‑pair 绝对差排序损失。

**📊 数据集**

实验数据集涵盖 Cora、CiteSeer、PubMed（有属性）、OGBL‑DDI（稠密无属性）与 Power（极稀疏无属性），共五个结构和属性极端对照组。

**📈 对比分析**

与 SEAL、BUDDY、Neo‑GNN、NCN/NCNC、OCNP、Gelato、HGCN 等基线对比，Concorde 在早期检索指标（P@10% / P@20%）均实现新高：Cora 25.28%/20.00%，CiteSeer 23.69%/18.92%，PubMed 20.00%/16.48%，OGBL‑DDI 46.45%/46.48%，Power 27.58%/24.24%。整体 AP 亦超越多数传统方法。

**⚠️ 局限性**

局限性包括：Ollivier–Ricci 近似需 Sinkhorn 迭代，增加计算开销；全局标量门控对结构异质性适应不佳；N‑pair 损失偏向早期检索，导致整体 AP 与概率校准不完全对齐；实验仅在单图设置，缺乏跨图泛化验证。

---

## 112. Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

**arXiv ID:** 2610.36202 | [PDF](https://arxiv.org/pdf/2610.36202v1)

**作者:** Oz Amram `[一作]` `[通讯]` (Independent), Oz Amram (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用基准模型的对数似然比评估（Persona‑Header Likelihood Evaluation）检测并分类大语言模型在后训练阶段隐式植入的系统提示（subliminal signals），并验证Persona Selection Model（PSM）的预测。

**💡 创新点**

提出仅利用基准模型的似然比而非提示模型本身的推理，能够解读隐藏意图；证明隐藏信号在不同模型家族的基准模型中也可读取；展示此方法比传统的prompted classifier显著更有效。

**🔧 技术方法**

主要技术为：基准模型的log‑likelihood比计算、对多条响应聚合、AUC评估、跨模型家族对比、以及对比prompted classifiers（后训练模型自身和GPT‑4.1）。

**📊 数据集**

实验数据包括：300道德困境文本问题（每题短回答），1000条随机数字序列；隐藏系统提示覆盖爱“猫”“鹰”“鸟”“火车”“秘密伤害”“秘密友好”等；使用Olmo 3 7B与Qwen2.5 7B的生成与基准模型。

**📈 对比分析**

将基准模型的似然比与生成模型直接似然、prompted classifiers 进行AUC比较；在k=30时，基准模型对数值序列的AUC可达0.96，文本回答的AUC达0.94；prompted classifiers大多停留在0.5左右；跨模型家族时基准模型仍保持0.82–0.94的AUC，远优于对应的post‑trained模型。

**⚠️ 局限性**

局限性包括：仅测试7B规模模型；隐藏意图以系统提示形式给出，而非训练嵌入；方法需预先给出假设性header，性能受描述匹配度影响；过滤与评估过程可能存在漏检；跨模型家族效果受预训练语料差异影响，未验证在更大规模或不同任务上的泛化。

---

## 113. Skill Dreaming: Evolving Repository Skills through Pull Request Streams

**arXiv ID:** 2610.36188 | [PDF](https://arxiv.org/pdf/2610.36188v1)

**作者:** Xinyuan Song `[一作]` (Causal Dynamics Lab), Liang Zhao `[通讯]` (Causal Dynamics Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 PR 流驱动的可持续演化技能库，利用历史 Pull Request 记录持续学习、更新和应用 LLM 生成的仓库专属技能，指导代码实现与维护。

**💡 创新点**

将 PR 记录与技能演化闭环化，实现技能的增删改而非一次性学习，并在不修改 LLM 参数的前提下保持技能与仓库演化同步。

**🔧 技术方法**

RAG+固定权重 LLM 生成技能、PR 记录构造与验证、技能对齐操作（Add/Update/Specialize/Extend/Conflict/Supersede/Keep）、基于 SWE‑bench Pro/Verified 的评估框架。

**📊 数据集**

SWE‑bench Pro 与 SWE‑bench Verified 任务集，结合 GitHub PR 历史。

**📈 对比分析**

与 BM25、dense 检索、Trace2Skill、Learning to Commit、SkillLens、CODESKILL、Socratic‑SWE 等基线对比，Claude Opus‑5 上解决率从 79% 提升至 88%（Pro）/92%（Verified），任务成功率和维护 Surface F1 均显著优于所有基线。

**⚠️ 局限性**

对 PR 质量与可验证性高度依赖；技能更新仍需手工定义对齐规则；相对较高的 token 与执行时间成本；在极大规模仓库中可扩展性待验证。

---

## 114. count-catcher: Lean Theorems from Model Counting Certificates

**arXiv ID:** 2610.36196 | [PDF](https://arxiv.org/pdf/2610.36196v1)

**作者:** Johannes K. Fichte `[一作]`, Stefan Szeider `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个名为 count-catcher 的工具，将 CPOG 证书（由模型计数器 d4 与 cpog-gen 生成）转换为 Lean 定理，借助编码引理实现对组合对象（如 Latin 方格、图、Dedekind 数等）的确切计数证明。

**💡 创新点**

创新点在于：① 将 CPOG 证书直接映射为 Lean 定理，降低信任基底仅为 Lean 核心、编译器与运行时；② 引入位并行反向检查和前向检查的裁剪裁剪技术，显著提升验证速度；③ 通过 Burnside 引理实现对对称性的自动化计数；④ 通过流式 refutation 处理 TB 级证书，突破单文件尺寸限制。

**🔧 技术方法**

技术包括：CPOG 证书格式、d4 精确计数器、cpog-gen 生成器、lrat-catcher 流式 LRAT 检查、Lean 反射验证、位并行逆向检查、分片计数（fiber）与 Burnside 组合计数、Lean 与 Mathlib 的结合。

**📊 数据集**

数据集主要来自 2022 Model Counting Competition 的无权公式集（共 123 公式，成功证明 111 例），以及 Latin 方格（8 阶、7 阶）、Dedekind 数、图的独立集、匹配等经典计数问题。

**📈 对比分析**

与 CPOG 官方验证器（cpog-lean-check 与 cpog-check）比较，count-catcher 在同等输入下的三项检查平均耗时约为 cpog-lean-check 的 25%（约四分之一），内存占用也更小；在处理大尺寸证书（GB 级）时表现尤为优异；同时在竞争集上成功证明 111 例，误报率为零。

**⚠️ 局限性**

局限性包括：① 仅支持 CPOG 证书的一个子集（不含删除行）；② 需要依赖 cpog-gen 的生成器，生成过程比 d4 计数慢；③ 仍需在 Lean 之外的工具完成编码与证书生成；④ 对于需要投影计数或权重计数的任务需进一步改造；⑤ 对极大规模（多 TB）证书仍受限于内存与时间。

---

## 115. Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models

**arXiv ID:** 2610.36192 | [PDF](https://arxiv.org/pdf/2610.36192v1)

**作者:** Ziqiao Wang `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在像素空间流模型的训练中加入 Wiener 统计估计，提出 Wiener‑informed Decoding（WiD）框架，使得网络在学习速度、稳定性和生成质量上都有提升。

**💡 创新点**

创新点在于：① 将传统线性 Wiener 估计的计算过程（分析、增益、合成）转化为可训练的模块；② 通过系数接口让统计坐标既参与特征提取也参与最终速度预测；③ 以预估统计作为初始化，兼顾直接估计与解码器的协同学习。

**🔧 技术方法**

使用的技术包括：像素空间流匹配（flow matching）框架、Wiener 估计、全局 DCT 或共享补丁 PCA 统计、PixelDiT / DeCo 解码器架构、REPA 预训练、FlowDPMSolver 采样、CFG 指导、EMA 权重、以及对齐损失和梯度规范化。

**📊 数据集**

主要数据集为 ImageNet（256×256 与 512×512），同时在 GenEval、DPG‑Bench 等文本到图像评测集上验证文本生成效果；使用了 51,200 张训练图像来估计统计，计算成本极低。

**📈 对比分析**

与原始 PixelDiT、DeCo 以及多种基线模型（如 PixelFlow、VDM++、DDT 等）进行对比。实验显示：在 ImageNet 256×256 上，PixelDiT‑XL+WiD 在 80 轮时 FID 为 2.09，显著低于原版 2.36；在 500 轮时 FID 仅为 1.53，节省约 37.5% 训练周期；在 512×512 微调后 FID 1.64，仅需 50 轮；文本生成指标也从 73.1 提升至 78.8，性能提升可观。

**⚠️ 局限性**

局限性包括：① 仍需要先计算全局或补丁统计（尽管成本低，但在极大规模数据集上可能受限）；② 主要验证在 ImageNet 和特定文本评测集，缺乏对其他场景（如视频、医学图像）的评估；③ 统计估计的线性假设在极端噪声或非 Gaussian 场景下可能效果不佳；④ 仅在像素空间流模型上测试，未探究对潜在空间流模型的迁移效果。

---

## 116. LLM Hallucination Detection via Low-rank Multilinear Decomposition

**arXiv ID:** 2610.36191 | [PDF](https://arxiv.org/pdf/2610.36191v1)

**作者:** Devansh Sharma `[一作]` (University of Central Florida), Shahana Ibrahim `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LOTUS框架，利用低秩多线性Tucker-2分解从LLM隐藏状态中提取低维特征，并在此基础上训练轻量分类器实现幻觉检测。

**💡 创新点**

创新点包括：①使用峰值、分位数和跨层差等统计摘要对可变长度token维度进行无监督压缩；②在隐藏状态上无标签学习Tucker-2子空间，保留层-特征结构；③将低维特征拼接后进行轻量化分类，显著降低对标注数据的依赖；④通过岭回归对齐实现跨模型迁移。

**🔧 技术方法**

技术方法包括低秩多线性Tucker-2分解、随机SVD、统计摘要（最大值、分位数、跨层差）、无监督特征学习、随机森林分类器、跨模型岭回归对齐。

**📊 数据集**

实验数据集为Qwen2.5-7B-Instruct与LLaMA-3.1-8B两大LLM，在TruthfulQA、TriviaQA、NQ-Open、TyDiQA-GP四个公开QA基准上进行评测。

**📈 对比分析**

与八种基线（六种无标签：Perplexity、LN-Entropy、Semantic Entropy、Lexical Similarity、EigenScore、HaloScope；两种有标签：HARP、ACT-ViT）比较，LOTUS在池化AUROC和within-question AUROC上普遍优于所有基线，尤其在NQ-Open和TruthfulQA上提升显著；在TriviaQA上略逊于ACT‑ViT。

**⚠️ 局限性**

局限性：①验证仅限英文QA基准和两大模型；②幻觉标签由自动度量生成，可能带来偏差；③每个模型需要单独学习Tucker基，迁移仍需额外对齐步骤；④对长文本、多语言和非QA场景的适用性未评估。

---

## 117. Bayesian Evidence Accumulation for Graph-Based Learning in Dynamic Data-Driven Applications Systems

**arXiv ID:** 2610.36190 | [PDF](https://arxiv.org/pdf/2610.36190v1)

**作者:** Moises Ponce `[一作]` (University of Tennessee), Vasileios Maroulas `[通讯]` (University of Tennessee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出B-EAGLe框架，将图结构学习任务转化为在DDDAS环境下的顺序证据累积过程，能够根据实时观测动态更新分类决策；

**💡 创新点**

创新点在于用贝叶斯滤波替代传统的加法累加器，实现对后验置信度的概率解释，并引入基于置信度阈值的停止决策；

**🔧 技术方法**

技术包括基于GAT的图神经网络编码器、贝叶斯证据累积模块、置信度阈值决策模块，以及对观测的条件独立假设与Cromwell规则的应用；

**📊 数据集**

实验使用四个公开图数据集：MNIST-SP、CIFAR10-SP（图级分类）以及PATTERN、CLUSTER（节点级分类）；

**📈 对比分析**

与GatedGCN+、GRIT等现有基线进行对比，B-EAGLe在三组数据集上取得最佳准确率（如CIFAR10-SP提升约13个百分点），且在图级任务中平均获取的观测次数低、超时率极低；

**⚠️ 局限性**

局限性包括：对节点级任务阈值过高导致超时率接近100%；仅实现了单向反馈，未结合观测选择策略；观察重叠导致的条件独立假设破坏可能导致过度自信；缺乏真实实时DDDAS环境的验证。

---

## 118. Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks

**arXiv ID:** 2610.36298 | [PDF](https://arxiv.org/pdf/2610.36298v1)

**作者:** Ayana Moshruba `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了脉冲神经网络（SNN）在四种常见模型窃取攻击（Papernot、Tramèr、T‑RND、Color Perturbation）下的抵抗能力，并与传统人工神经网络（ANN）进行对比。

**💡 创新点**

创新点在于首次系统验证SNN因其时序稀疏通信特性而比ANN更难被窃取，并指出这种优势源于SNN的膜电位动力学和离散脉冲编码，而非单纯的架构差异。

**🔧 技术方法**

使用上述四种黑盒模型窃取方法，通过训练ANN或SNN surrogate，评估功能一致性（agreement）、对抗转移性（non‑targeted 与 targeted）以及查询效率；同时探讨目标与替代模型的架构匹配对结果的影响。

**📊 数据集**

实验数据集为 MNIST、Fashion‑MNIST（FMNIST）和 CIFAR‑10，覆盖从简单到复杂的图像分类任务。

**📈 对比分析**

通过对比相同攻击条件下的目标ANN与SNN，发现SNN 的 surrogate agreement 低 10–20%，对抗转移性低 15–25%，且查询效率显著差（SNN 需要数十倍更多查询才能达到相同水平）。

**⚠️ 局限性**

局限性包括：仅在公开图像数据集与固定的 SNN 设计（如 LIF）上验证；未针对 SNN 的时序特征设计专门的窃取策略；实验结果受攻击预算、超参数设置和模型训练细节影响，可能不完全代表所有 SNN 或真实部署环境。

---

## 119. Compatibility and future-proof testing for multi-agent systems

**arXiv ID:** 2610.36289 | [PDF](https://arxiv.org/pdf/2610.36289v1)

**作者:** Yueshan Li `[一作]` (ETH Zurich), Emilio Frazzoli `[通讯]` (ETH Zurich)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于种子(agent seed)的未来兼容性测试框架，理论分析多智能体系统在不进行开发者间协同的情况下如何保证共同安全与任务完成。

**💡 创新点**

创新点包括：①将兼容性定义为种子闭包，证明种子扩展使得未来兼容集合增大；②引入种子仅测试（SOT）概念并给出不可区分性的下界；③在两种经典案例（单车道跟随与仓库路径规划）中验证理论。

**🔧 技术方法**

技术方法包括：使用arena模型（agent space、outcome generator、unacceptable outcomes）构建兼容性框架；集合闭包与最大兼容集概念；Zorn引理证明最大兼容集存在；形式化定理与反例推导；仿真验证。

**📊 数据集**

在案例研究中使用自定义仿真环境：①一条车道追尾实验（连续agent空间）；②网格仓库多机器人路径规划实验（离散agent空间）。未使用公开数据集。

**📈 对比分析**

通过比较种子兼容集合与未来兼容集合，展示种子兼容不一定导致互相兼容；利用仿真证明在不同种子下最大兼容集与闭包的差异；在追尾案例中证明存在零测试损失的SOT。

**⚠️ 局限性**

局限性包括：需要先预先设定种子，种子选择直接影响兼容性与测试损失；闭包与可分离集的计算一般不可行，需要近似方法；框架未覆盖多轮或动态环境下的测试与更新。

---

## 120. The Complexity of Strategizing Against No-Regret Learners: Pure When It Is Easy, Hard Even When Mixed

**arXiv ID:** 2610.36261 | [PDF](https://arxiv.org/pdf/2610.36261v1)

**作者:** Yiheng Su `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究给定在线学习者的可预见策略优化问题，并证明在一般博弈中，即使允许混合动作，求解仍在多项式时间内不可行，零和博弈则可用纯动作实现近似最优。

**💡 创新点**

创新点在于将混合动作的自由度纳入硬度证明，扩展至广泛的均值基学习者，并构造了通过“激活-收获”游戏与独立集间的多项式时间归约；同时在零和博弈中给出纯策略的盈余恢复方法。

**🔧 技术方法**

主要技术包括：从gap独立集构造基准游戏、均值基学习者条件、混合动作提取与设置区间划分、FTRL潜能分解与Bregman散度分析、终端与前缀平衡的离散化。

**📊 数据集**

未使用真实数据集，实验全部在符号化的合成博弈实例上验证理论结果。

**📈 对比分析**

通过理论证明表明：在一般博弈中多项式时间优化不可行（P≠NP），而零和博弈可在多项式时间内得到与最优混合策略相当、且可获得与学习者的余裕匹配的纯策略。

**⚠️ 局限性**

局限性包括：硬度结果针对特定构造的最坏情况；假设学习者为均值基且满足特定学习率；零和博弈结果依赖于可模拟的无外部损失；对随机或受噪声的博弈缺乏分析。

---

## 121. AHEAD: Adaptive Neural Acceleration for Large-Scale Linear Programming

**arXiv ID:** 2610.36321 | [PDF](https://arxiv.org/pdf/2610.36321v1)

**作者:** Sihao Liu `[一作]` (North Carolina State University), Xiaorui Liu `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种轻量级神经加速器AHEAD，用来在大规模线性规划（LP）中预测步长并自适应停止，从而在端到端求解时间上实现显著加速。

**💡 创新点**

创新点主要包括：①只学习步长的乘子而保持原始迭代方向，保持每一步低成本且可用作热启动；②采用权重共享的循环预测器，能够在任意迭代数上迭代且不依赖固定深度；③学习停止策略根据残差和精度阈值自适应决定热启动阶段结束时机，实现实例级最优停机。

**🔧 技术方法**

使用技术：①基于PDHG/PDLP的一阶方法；②十维 solver 状态特征（残差、局部稳定性、权重、几何一致性等）；③小型 MLP 预测器和手工停止策略；④多周期（多 horizon）训练和对手工停止策略的回归训练；⑤与 Gurobi GPU‑PDHG 和 OR‑Tools PDLP 结合。

**📊 数据集**

实验数据集：合成 packing LP（10^5–10^7 变量，0.1n 随机不等式，1 个等式约束）；Barabási–Albert 图生成的 PageRank LP；以及未见分布和更大规模的测试集。

**📈 对比分析**

比较方法：与冷启动 Gurobi GPU‑PDHG、OR‑Tools PDLP 以及最接近的方法 PDHG‑Net 进行对比；在目标精度 1e‑4 上，AHEAD 在 10^5–10^7 变量的 packing LP 上实现 6.5–11.6× 的端到端速度提升，在 PageRank LP 上同样获得 6–12× 的加速；在更大规模、不同分布、不同难度和不同精度下保持高性能。

**⚠️ 局限性**

局限性：仅在合成与 PageRank LP 上验证，未在真实工业 LP 或多种求解器上进行广泛测试；需要先跑完整求解器来训练手工停止策略，耗时；对特定一阶方法（PDHG/PDLP）依赖较强，通用性待进一步验证。

---

## 122. SG^2SLAM: Spatial Geometry-Informed 3D Gaussian Splatting SLAM for Monocular Outdoor Navigation

**arXiv ID:** 2610.36312 | [PDF](https://arxiv.org/pdf/2610.36312v1)

**作者:** Ziang Guo `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于流式空间几何和射线条件3D高斯光斑的单目室外导航SLAM系统SG^2SLAM。

**💡 创新点**

创新点在于摒弃逐帧2D–3D匹配，采用块级几何先验、尺度校正以及射线条件核，实现全局一致的3D–3D关联与优化。

**🔧 技术方法**

使用技术包括3D高斯光斑、流式空间几何、单目深度网络、束束优化、Sim(3)图优化和闭式射线条件渲染。

**📊 数据集**

在KITTI、Waymo Open Dataset以及自采集驾驶数据上进行实验。

**📈 对比分析**

与MonoGS、S3PO-GS、DiskChunGS、GigaSLAM等基线相比，SG^2SLAM在KITTI ATE RMSE从6.24m降至4.62m，在Waymo从0.63m提升至0.48m，并在渲染质量上PSNR/SSIM/LPIPS均优于对手。

**⚠️ 局限性**

局限性包括计算开销较大、在极端光照或密集动态环境下仍可能出现跟踪失败，且目前缺乏实时循环闭环机制。

---

## 123. Detecting and Mitigating Query-Based System Prompt Leakage Attacks

**arXiv ID:** 2610.36295 | [PDF](https://arxiv.org/pdf/2610.36295v1)

**作者:** Tushar Nayan `[一作]` (Florida International University), Ruimin Sun `[通讯]` (Florida International University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个多阶段黑盒LLM系统提示泄露检测与缓解框架。

**💡 创新点**

创新点在于通过构建有界语义交互图并用GCN对交互结构进行分类，实现跨查询关系检测，从而在多轮攻击中获得高召回且低误报。

**🔧 技术方法**

采用句子嵌入相似度计算、阈值过滤、Grubbs异常检测、线图构造+GCN分类、基于提示的单查询风险评分以及边缘加权图卷积网络。

**📊 数据集**

使用GSM8K、SST‑2、SQuAD v2、Dolly15k、WildJailbreak等正常数据集，以及OASST1、ChatAlpaca、WildChat等多轮对话数据；攻击数据集包括Zhang、PromptFuzz、LeakAgent、PLeak等。

**📈 对比分析**

与PromptGuard、LLMGuard等基线相比，混合查询下F1≥94.8%、FPR≤3.4%；多轮对话下召回100%、FPR≤3.42%；在自适应攻击下精确匹配泄露降至0.01，子串泄露≤0.09，整体检测召回率≥99%，误报率低，平均延迟≈33 ms/查询。

**⚠️ 局限性**

局限性包括：需要足够查询历史才能形成有效结构，短会话或极度分散的攻击可能逃逸；依赖句子嵌入，对极端语义变形的查询灵敏度有限；需额外存储图结构，增加资源开销。

---

## 124. Learning Multi-Phase Robot Loco-Manipulation via Phase-Conditioned Policy Optimization

**arXiv ID:** 2610.36270 | [PDF](https://arxiv.org/pdf/2610.36270v1)

**作者:** Simone Manni `[一作]` (ETH Zurich), Jan Brüdigam `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

学习多阶段机器人行走‑操控任务，提出相位条件化的PPO框架，能够在单一统一策略下完成复杂的多阶段行走与操作行为

**💡 创新点**

通过环境状态机划分任务相位，使用相位条件化价值网络、相位门控奖励、辅助相位预测头和多阶段重置课程，直接在无示范数据的情况下完成长时程任务

**🔧 技术方法**

PPO、相位条件化价值网络、相位门控奖励、辅助相位预测头、相位条件化低层全身控制、相位重置课程、分层控制架构

**📊 数据集**

无人工示范，仅使用仿真环境随机化物体质量、大小、摩擦；使用实时姿态估计和 ArUco 标记实现无运动捕捉执行

**📈 对比分析**

在 Spot 轮胎任务中成功率约 89%（±2%），G1 人体箱子搬运任务成功率 92%（±2%）；消除辅助预测或多阶段初始化会显著下降，验证框架有效

**⚠️ 局限性**

需手工划分相位、相位数量多时网络容量受限、对仿真接触动力学依赖、难以扩展到更多任务或更长时间尺度

---

## 125. Freeze, Don't Clone: Fine-Tuning with Expressive Policies Without Retaining Offline Data

**arXiv ID:** 2610.36247 | [PDF](https://arxiv.org/pdf/2610.36247v1)

**作者:** Sami Nur Islam `[一作]` (Mila Quebec AI Institute), Pablo Samuel Castro `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在离线数据不再可用的情况下，如何进行在线微调；提出通过停止在线行为克隆以保留预训练的表达式行为先验来避免性能崩溃。

**💡 创新点**

创新点在于将行为先验冻结并让价值学习继续更新，揭示了行为克隆在无离线数据时会导致先验灾难性遗忘，并给出了实证证明与对策。

**🔧 技术方法**

采用流基与扩散式生成先验（IFQL、FQL、QC、IDQL），结合行为克隆、价值更新与一阶演员蒸馏等技术。

**📊 数据集**

在OGBench七个机器人操作与运动域（35个任务）、低质量离线数据（SMD、DM）以及Robomimic多人的人类演示数据（Multihuman）上进行评估。

**📈 对比分析**

与保留离线数据的基线相比，BC-Off 在大多数任务上显著防止崩溃，并且在多数情形下可逼近甚至超越保留数据的性能；BC-On 在无离线数据时表现为快速退化。

**⚠️ 局限性**

仅适用于表达式多模态先验，对单峰（高斯）先验无效；冻结先验在动态变化或分布移位场景下可能限制进一步适应，且需要足够的先验表达能力。

---

## 126. Three Worlds for the Price of a Logarithm: Baseline-Protected Best-of-Both-Worlds Combinatorial Semi-Bandits

**arXiv ID:** 2610.36245 | [PDF](https://arxiv.org/pdf/2610.36245v1)

**作者:** Ting Hu `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种无视时间、可自适应的半波段组合学习算法，能够在不知时间长度、损失间隙或环境类型的情况下，保证对基线的对数期望损失、对抗性平方根损失和随机环境下的多项对数速率；

**💡 创新点**

创新之处在于通过双重经验测试的增量权重与预算机制，实现了基线保护、对抗鲁棒性与随机适应性三者的统一，且无需预先设定任何超参数；

**🔧 技术方法**

采用基于熵的 Follow‑the‑Regularized‑Leader (FTRL) 在混合阶段，以及混合 FTRL 在纯阶段的混合采样策略，并结合经验预算的自适应增倍测试；

**📊 数据集**

本文主要为理论研究，未使用具体数据集；实验验证可通过仿真或公开基准（如路由、推荐）进一步补充；

**📈 对比分析**

与传统需要已知基线覆盖、时间或间隙的算法相比，本文在理论上实现了对数基线期望、平方根对抗和多项对数随机三种界，且不依赖任何先验信息；

**⚠️ 局限性**

主要局限包括：基线必须覆盖每个臂（δ>0）；随机速率仅在独立同分布且存在唯一最优动作时成立；计算复杂度需枚举动作或进行凸分解，可能在大规模动作集合上开销较大。

---

## 127. Characterizing Privacy Leakage in Spiking Neural Networks: A Systematic Analysis of Temporal Dynamics, Stochasticity, Neuronal Dynamics and Learning Mechanisms

**arXiv ID:** 2610.36243 | [PDF](https://arxiv.org/pdf/2610.36243v1)

**作者:** Ayana Moshruba `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `9cc9baba-5356-466d-81ff-d80028d90279` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统性探讨了尖峰神经网络(SNN)的时间动态、随机性、神经元内部动力学与学习机制对会员推断攻击(MIA)泄露的影响，分离并量化各因素在隐私-性能权衡中的作用。

**💡 创新点**

创新点包括：①将时间动态、随机性、神经元模型与学习算法解耦，构建统一实验框架；②证明基于神经元内在动力学产生的“Burst”与“Delayed‑Burst”时序能在保持或提升准确率的同时降低MIA泄露；③引入时间异质性与进化学习、差分隐私训练等多种学习策略，全面评估其对隐私的影响。

**🔧 技术方法**

技术主要包括：尖峰神经网络(Leaky Integrate‑and‑Fire, Adaptive Exponential Integrate‑and‑Fire)、快速 Sigmoid 替代梯度、进化优化框架 EONS、差分隐私梯度下降(DPSGD)、黑盒阴影模型攻击与基于支持向量机的MIA分类器。

**📊 数据集**

使用的公开数据集有：MNIST、Fashion‑MNIST、CIFAR‑10、Iris、Breast Cancer Wisconsin Diagnostic。

**📈 对比分析**

对比方法：统一采用黑盒阴影模型MIA，评估指标为测试准确率与AUC（AUC越接近0.5越好）。实验表明：Poisson‑Burst和Delayed‑Burst在保持/略降准确率的前提下能将MIA AUC从≈0.53降至≈0.50；AdEx模型在同一时序下进一步降低AUC同时提升准确率；异质时间分配在一定比例下可降低泄露而对准确率影响有限；进化学习在所有时序下泄露已接近随机，且对Burst的准确率下降更大；差分隐私训练展示了噪声注入对不同时序的非一致影响。

**⚠️ 局限性**

局限性：①仅研究了三种预定义时序和两种神经元模型，未覆盖更丰富的生物学时序或多种突触模型；②进化学习实验受限于计算成本，仅覆盖两种小型数据集；③攻击仅为黑盒阴影模型，未验证对白盒或梯度攻击的鲁棒性；④未探讨长期训练、迁移学习或在线学习情景下的隐私表现。

---

## 128. Explicit Reed-Solomon Subspace Designs: Optimal List and Line Decoding

**arXiv ID:** 2610.36215 | [PDF](https://arxiv.org/pdf/2610.36215v1)

**作者:** Omar Alrabiah `[一作]` (University of California Berkeley), Venkatesan Guruswami `[通讯]` (University of California Berkeley)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了一种显式的Reed-Solomon编码，能够在接近列表解码容量的情况下，从错误中恢复信息，且具有最优的列表大小。

**💡 创新点**

创新点在于提出了一种新的构造方法，使得Reed-Solomon编码能够在接近列表解码容量的情况下实现常数大小的列表，而不是多项式增长的列表大小。

**🔧 技术方法**

使用了扩展域上的Reed-Solomon编码和确定性列表解码算法，结合了子空间设计的性质。

**📊 数据集**

使用了多项式大小的扩展域，并且构造了多组评估点以满足特定的编码性质。

**📈 对比分析**

与Brakensiek等人（2026）的算法相比，本文的方法在列表大小上实现了常数（接近最优），而不是多项式增长，尽管需要仔细选择编码的评估点。

**⚠️ 局限性**

限制在于构造的评估点依赖于特定的选择，且在某些情况下可能无法保证最优性能。

---

## 129. Confidence Is Not a Promise: A Reference Audit of Typed Decision Models

**arXiv ID:** 2610.36311 | [PDF](https://arxiv.org/pdf/2610.36311v1)

**作者:** Pedro Carvalho Brom `[一作]` `[通讯]` (Federal Institute of Brasília), Pedro Carvalho Brom (Federal Institute of Brasília)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较并评估两类 typed‑decision 模型（Laya 与 Jev），从概率校准、门控机制（Beta 下界、Learn‑then‑Test 等）和决策置信度等角度量化它们的可靠性与实用性，并在多语言、多领域、噪声、漂移等真实场景下对其性能进行系统实验。

**💡 创新点**

创新点包括：① 将门控阈值的置信下界（Beta lower‑bound rule）与传统的点估计阈值进行对比，并证明其在保证精度约束下的优越性；② 通过“Bayes阈值”与多目标优化统一阐述门控的成本与风险权衡；③ 细粒度评估置信度对解释可信度、噪声鲁棒性和语言迁移的影响；④ 在漂移场景中引入基于 Beta 更新的“贝叶斯重训练循环”并与现有风险控制方法进行对比。

**🔧 技术方法**

使用的技术主要包括 typed‑decision 模型架构、温度缩放校准、Beta 似然下界、Learn‑then‑Test 与 SGR 的多重检验策略、AUROC 与 ECE 评价指标、温度化简、蒙特卡罗仿真与 Bootstrap 置信区间、以及多语言多领域实验流水线。

**📊 数据集**

数据集主要是：MASSIVE（意图分类，52 语言），News Category（新闻类别），Twitter Financial News Sentiment（财经情感），LIAR2（事实判断），Fake.Br（假新闻检测）等；所有实验均使用 2000 条测试样本，并在不同语言、噪声、细调规模（250–8000 标签）下重复评估。

**📈 对比分析**

实验结果显示：① 在未细调时 Jev 的准确率和校准均优于 Laya；② 细调后 Laya 在 4000–11274 标签下可超越 Jev，且成本可观；③ Beta 下界门控在 99% 约束下违约率低于点估计门控，且在多数场景下决策覆盖更好；④ 在噪声、语言迁移和漂移下，未校准的门控会失效，需在训练时加入噪声或针对目标语言重新校准；⑥ 与 Learn‑then‑Test 的比较表明，尽管后者在保证性上更严格，但覆盖率较低；整体而言，Beta 规则在“可信决策”与“覆盖率”之间提供了更优的折衷。

**⚠️ 局限性**

局限性包括：① 置信下界仅在样本可交换的前提下保证，噪声、语言迁移或漂移时需重新校准；② 对单个决策的可靠性没有直接保证，门控只保障群体精度；③ 需要大量标签才能在细调后达到竞争优势；④ 解释可信度测量受限于简单删除实验，无法覆盖所有解释方式；⑤ 贝叶斯重训练循环在不同漂移场景下效果尚未充分验证；⑥ 结果对模型内部机制（如 Laya 的词嵌入、Jev 的黑盒 API）假设有限，无法完全复现。

---

## 130. A Smart Instant Book Filter Model based on Dual-Objective Optimization

**arXiv ID:** 2610.36287 | [PDF](https://arxiv.org/pdf/2610.36287v1)

**作者:** Yi Hou `[一作]` (Instant Book), Huizhong Duan `[通讯]` (Search Ranking)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并部署了智能即时预订过滤系统，通过双目标优化实现提升即时预订率同时保持总体预订率不下降。

**💡 创新点**

创新点包括：①使用折扣累计收益（DCG）作为即时预订与总体预订的代理指标；②采用随机游走参数探索结合人工专家初始化的学习框架；③构建确定性决策树模型与分支阈值向量，兼顾可解释性与优化目标。

**🔧 技术方法**

技术手段包括：折扣累计收益计算、确定性决策树与分支逻辑、随机游走参数探索、离线并行排名模拟（SxS Ranker）、A/B 测试、API 与市场洞察服务集成。

**📊 数据集**

数据集为 Airbnb 真实全局搜索流量，约10,000 次搜索迭代用于每一步评估，搜索结果页固定 18 条列表，包含即时预订与请求预订的实际交易记录。

**📈 对比分析**

比较方法为将学习得到的阈值向量与六组人工专家设定的阈值进行 A/B 对比。实验显示学习模型实现即时预订增长 8.20%，并保持总预订率无下降，优于任何人工阈值组合；同时观察到主机即时预订采用率提升近 1%。

**⚠️ 局限性**

局限性包括：①折扣累计收益作为代理指标虽然有效但仍可能无法完全捕捉所有预订行为；②决策树阈值固定，缺乏在线自适应能力；③随机游走探索耗时且易受局部最优影响；④实验仅在特定搜索页面（18 条列表）和地区验证，普适性需进一步验证。

---

## 131. Preserving rare-item recall under a storage budget: quantization, distillation, and the limits of teacher fidelity

**arXiv ID:** 2610.36267 | [PDF](https://arxiv.org/pdf/2610.36267v1)

**作者:** Han Chen `[一作]` (University of Texas at Austin), Yingrui Li `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在固定存储预算下比较了七种压缩方法对稀缺项召回的影响。

**💡 创新点**

创新点在于将低位数低秩分解与无偏知识蒸馏相对比，并提出使用总尾召回和教师保留率评估稀缺项性能。

**🔧 技术方法**

主要技术包括矩阵分解、LightGCN、4位低秩分解、无偏知识蒸馏、统计检验和自举置信区间。

**📊 数据集**

使用的公开数据集包括 Gowalla、Last.fm、Amazon Video Games、Steam、Goodbooks-10k、MovieLens-20M 等七个。

**📈 对比分析**

在每个单元中在相同字节预算下对比七种候选模型，结果显示4位低秩分解在大多数单元中获得最高尾召回，且其尾召回几乎等同于完整教师；而 40 维低精度模型效果最差，蒸馏学生效果介于两者之间。

**⚠️ 局限性**

局限性包括仅针对 MF（以及少量 LightGCN）进行评估、仅使用固定的尾部阈值与列表长度、未测量推理延迟、未探索其他蒸馏或量化方法。

---

## 132. MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge

**arXiv ID:** 2610.36234 | [PDF](https://arxiv.org/pdf/2610.36234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 133. Instant Software Factories

**arXiv ID:** 2610.36254 | [PDF](https://arxiv.org/pdf/2610.36254v1)

**作者:** Wenxin Jiang `[一作]` (Socket Inc.), Mikola Lysenko `[通讯]` (Socket Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一种只需两种协调输入（cron 任务代理与共享沟通板）和可分离治理层的即时软件工厂，用大型语言模型持续进行软件维护和改进。

**💡 创新点**

创新点在于将软件工厂的协调机制简化为两输入设计，并将工厂进展视为多目标优化问题，强调内生反馈与可验证进展。

**🔧 技术方法**

使用了大型语言模型代理（如 Claude）、GitHub 作为共享板、cron 定时任务、黑板式任务分配、治理层的审批与停机开关，以及多目标优化框架。

**📊 数据集**

在两个工业级仓库（CLI 工具和内部测试 harness）中观察 15–17 天工厂运行，收集了 704 条 issue、58 条 CI 工作流 PR 以及 14 种缺陷类型的数据。

**📈 对比分析**

与手工配置工厂相比，实时工厂 merge 率提升 15 倍、issue 归属率 95%、平均 46 小时闭环，成功交付 94% 的修复；实验计划通过多目标实验验证适应性、治理对速度、逃逸率与成本的影响。

**⚠️ 局限性**

局限性包括仅在单一组织的两个仓库中测试，缺乏跨仓库或多语言环境验证，治理层仍需人工审批，逃逸率与缺陷覆盖可能因检测覆盖不足而被低估。

---

## 134. cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation

**arXiv ID:** 2610.36250 | [PDF](https://arxiv.org/pdf/2610.36250v1)

**作者:** Philip Huang `[一作]` (NVIDIA), Lars Johannsmeier `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为cuArms的GPU加速冲突基于规划器，用于多臂机器人协同任务与运动规划，能够快速生成可行的多臂操作计划并在任意时间对计划质量进行迭代改进。

**💡 创新点**

核心创新点包括：① 将GPU批量几何验证与冲突基于多代理规划（LaCAM+PIBT）深度耦合，显著加速冲突检测；② 通过分解单臂道路图与交互、构造多模态道路图并复用计算，降低搜索空间；③ 引入GPU并行冲突表与lazy搜索，避免递归CPU‑GPU通信；④ 兼容任意任务/调度骨架，支持异步执行与协作技能；⑤ 与高层调度器（ScheduleStream）协同，形成“先排程后协调”的层次化TAMP框架。

**🔧 技术方法**

主要技术手段包括GPU批量几何检查、逆运动学求解、Dijkstra后向启发式、PIBT优先级继承、LaCAM冲突驱动搜索、multi‑modal道路图构造、GPU原生实现、GPU加速的快捷路径搜索以及与ScheduleStream等符号调度器的集成。

**📊 数据集**

实验数据集涵盖：① RoboBallet焊接任务（10/40个目标，4/8臂）；② 桌面操作任务（hold、stack、pack、unpack，2-4物体，扩展到6-8物体）；③ 两个真实机器人（Flexiv Rizon4）在电子组装场景下的实验。所有实验均在仿真与真实硬件上进行，并使用自研的RoboBallet环境和ScheduleStream调度器。

**📈 对比分析**

与ScheduleStream、VAMP‑MR、RL（RoboBallet）等基线进行比较。cuArms在焊接任务中平均仅需0.2–0.4 s获得初始可行解，RL方法耗时超过10⁵ s；通过快捷路径搜索和备选时间表，初始时间约提升47%；在桌面任务中比ScheduleStream快约46%并获得更优的首解；在8物体包装任务中能求解成功，而ScheduleStream无法完成；在真实组装实验中实现了无碰撞的协作抓取与携带。整体来看，cuArms在求解速度、可行性和计划质量方面均优于现有方法。

**⚠️ 局限性**

局限性包括：① 仍缺乏全局最优性保证，依赖启发式和快捷路径；② 对于高度动态或大规模物体环境，GPU批量检查与道路图构造的开销可能显著；③ 需要GPU资源，无法直接移植至仅CPU环境；④ 规划过程依赖预先构造的多模态道路图，对任务结构变化时需重建或更新；⑤ 在极端高并发或极大任务规模时，优先级继承与冲突表的管理可能成为瓶颈。

---

## 135. BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals

**arXiv ID:** 2610.36239 | [PDF](https://arxiv.org/pdf/2610.36239v1)

**作者:** Aysenur Inan `[一作]` (Walmart Global Tech), Sushant Kumar `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并评估了一种三信号序列推荐模型BehaveRec，结合短期会话、长期用户历史和行为社区图，探究长期行为历史何时能提升推荐效果。

**💡 创新点**

创新点在于：①将90天双向用户历史编码器（UHE）与GraphSAGE行为相似图共同用于推荐；②通过可学习的ThreeWayGate动态融合三种信号；③系统性研究长期历史丰富度对性能的剂量-响应关系，给出约20次交互的可部署阈值。

**🔧 技术方法**

技术包括：因果Transformer会话编码器、双向Transformer历史编码器、GraphSAGE用户图聚合、InfoNCE对齐、权重共享、权重绑定位分类器、混合精度训练与AdamW优化。

**📊 数据集**

数据集：137M序列、12.6M用户的工业级购物日志；公开数据包含MovieLens-1M、Amazon CDs & Vinyl、Amazon Electronics，分别对应不同的历史稀疏度。

**📈 对比分析**

对比方法：SASRec、BERT4Rec、以及BehaveRec的不同变体；评估Recall@k/NDCG@k；在工业集上BehaveRec R@5提升+3.2%（+0.0226），在MovieLens上+0.67%；在稀疏历史的Amazon上表现不如SASRec。

**⚠️ 局限性**

局限性包括：单次训练结果缺乏多种种子验证；图构造对比为推理时切换而非独立训练；在完整物品词表上的评估为单跑、方向性混合；阈值20次交互为经验性而非理论确定；对极端稀疏历史的效果不佳。

---

## 136. Beyond IceBERT: Open-Data ModernBERT Encoders for Icelandic from 12M to 1B Parameters

**arXiv ID:** 2610.36230 | [PDF](https://arxiv.org/pdf/2610.36230v1)

**作者:** Hafsteinn Einarsson `[一作]` `[通讯]` (University of Iceland), Hafsteinn Einarsson (University of Iceland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了从12M到1B参数的 ModernBERT 版冰岛语编码器，使用9B开放文本进行预训练，并在 IceBERT 原始基准与新增任务上进行统一评测；同时对比从零开始与多语或英语源模型的 warm‑start 效果，量化不同 token‑budget 下的节省。

**💡 创新点**

首次在冰岛语上使用 ModernBERT 架构训练1B参数单语编码器；系统性、可重复的 warm‑start 对比实验；通过拆分 transformer 层与嵌入层评估迁移贡献；在统一协议下对 30 个编码器进行置信区间评估。

**🔧 技术方法**

ModernBERT（预归一化、GeGLU、rotary 位置、每第三层全局注意）+ Masked Language Modeling；Fast Vocabulary Transfer（词表迁移）；多 token‑budget 训练策略；层级 Bootstrap 置信区间；学习率调优与多种 seed 重复。

**📊 数据集**

9B tokens 的公开冰岛语文本：Icelandic Gigaword、Wikipedia、Dynaword、MaCoCu、FineWeb‑2、FinePDFs、HPLT 3.0、MADLAD‑400、IC3‑v2；评测集包括 PoS（MIM‑GOLD）、NER（MIM‑GOLD‑NER）、句法（GreynirCorpus）、错误检测（IceEC）、WinoGrande、QA（NQiI + RUQuAD）、ScaLA、UD 再解析等。

**📈 对比分析**

统一 fine‑tune 方案：先在验证集上调优学习率，随后用 5 份 seed 进行训练；使用层级 bootstrap 估计置信区间。结果显示 1B 零起始模型在所有 11 个原 Benchmarks 上显著优于 IceBERT；warm‑start 从 Ettin 在大多任务上获得最佳平均分；在低 token‑budget（10–20 tokens/param）下 warm‑start 可将冰岛文本需求压缩到约 1/8‑1/4，几乎不降低性能；相比大型多语模型在 POS、NER、解析等任务仍略逊。

**⚠️ 局限性**

限制：seed 波动导致部分差异未被置信区间覆盖；未公开检查源模型（Ettin、mmBERT、XLM‑R）与 IceBERT 的完整训练数据；warm‑start 的学习率未做更广泛探索；评估仅限于 fine‑tune 任务，未检验长输入或句子嵌入性能；实验专注于冰岛语，低资源语言或缺乏相同形状源模型时结果可能不同。

---

## 137. Bringing Hardware Security Education to All: A Seamless Curriculum Integration Approach

**arXiv ID:** 2610.36219 | [PDF](https://arxiv.org/pdf/2610.36219v1)

**作者:** Tanvir Hossain `[一作]` (University of Kansas), Tamzidul Hoque `[通讯]` (University of Kansas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在电气与计算机工程及计算机科学的核心课程中嵌入硬件安全模块，形成无额外课程的整合教学框架

**💡 创新点**

将硬件安全教育与已有设计课程紧密耦合，采用情境学习原则，降低资源需求并扩大覆盖面

**🔧 技术方法**

基于现有实验设备、交叉编译IDE、逻辑分析仪、VHDL FPGA、8位 MCU 等硬件平台进行实验教学

**📊 数据集**

使用的“数据集”为学生在课程实验中产生的硬件实验记录和自评问卷数据（共25份问卷）

**📈 对比分析**

通过学生自评和外部评估问卷比较学习效果，结果显示 87%–100% 认为提升了动手学习，知识得分提升 0.7–1.2 分，未给出传统性能指标

**⚠️ 局限性**

样本量小、非随机、仅自评且缺乏对照组，设备依赖有限，部分模块尚未在多平台或更高位宽目标上验证

---

## 138. UT-HMR: Uncertainty-Guided Temporal Modeling for Robust 3D Human Mesh Recovery

**arXiv ID:** 2610.36203 | [PDF](https://arxiv.org/pdf/2610.36203v1)

**作者:** Zhibo Wang `[一作]` (University of Waterloo), David Clausi `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一阶段多人体网格恢复框架UT-HMR，利用时间早期融合和姿态不确定性建模，提高在弱观测（如远距离、遮挡、运动模糊）下的重建鲁棒性。

**💡 创新点**

创新点包括：① 将低秩+对角高斯协方差用于姿态不确定性建模；② 将协方差转化为异方差权重用于监督；③ 在一阶段DETR式网络中首次引入时间早期融合以聚合邻帧信息。

**🔧 技术方法**

技术细节：ViT-B/14编码器、DETR式解码器、低秩+对角协方差预测、分布负对数似然损失、异方差联合损失、Temporal Early Fusion与一阶段多人体推断。

**📊 数据集**

实验数据集：VIP‑HockeyPose‑v2（冰球）、WorldPose（足球）和3DPW（通用场景）。

**📈 对比分析**

与SAT‑HMR、PSVT、Multi‑HMR等方法对比，UT‑HMR在VIP‑HockeyPose‑v2和WorldPose上分别以MPJPE/PA‑MPJPE/PVE获得最佳成绩；在3DPW上保持竞争力，且多帧UT‑HMR相较单帧提升明显，且速度仍快于大模型。

**⚠️ 局限性**

局限性：仅在短时窗口内利用时间信息，缺乏长序列一致性约束；对极端遮挡和极端运动模糊仍有挑战；依赖ViT-B/14和较大显存，计算成本相对较高。

---

## 139. Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs

**arXiv ID:** 2610.36195 | [PDF](https://arxiv.org/pdf/2610.36195v1)

**作者:** Kartik Jangra `[一作]` (Nilekani Centre at AI4Bharat), Raj Dabre `[通讯]` (Nilekani Centre at AI4Bharat)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了BrailleBench基准，用合成盲文图像训练视觉语言模型并评估其在真实盲文照片上的转写性能。

**💡 创新点**

①首次构建真实盲文图像基准；②证明仅用合成数据即可显著提升VLM在真实盲文上的准确率；③发现合成验证准确率不一定能预测真实表现。

**🔧 技术方法**

大规模合成盲文图像生成、全参数微调（Qwen3.5‑2B、Chandra‑OCR‑2）以及工具辅助推理（Claude Code、Codex）等多模态技术。

**📊 数据集**

259张手工标注的Grade‑1/Grade‑2盲文照片（BrailleBench）、约276k/500k的合成盲文图像，以及先前工作提供的100k合成盲文（WMT‑Braille）。

**📈 对比分析**

通过字符错误率、词错误率、完全匹配率等指标，对比零拷贝VLM、工具辅助系统、现有OBR工具和自研合成微调模型；自研模型在Grade‑1上可达86.98%完全匹配、Grade‑2 61.36%，仅次于工具辅助系统，显著优于其他方法。

**⚠️ 局限性**

样本量有限、仅覆盖英文盲文、Grade‑2的上下文缩写识别仍不够、合成与真实迁移仍存在显著差距、工具辅助系统成本高等限制。

---

## 140. Bridging Exploration and Loss of Plasticity in On-Policy Reinforcement Learning

**arXiv ID:** 2610.36309 | [PDF](https://arxiv.org/pdf/2610.36309v1)

**作者:** Andrea Vasco Grieco `[一作]` (Politecnico di Torino), Raffaello Camoriano `[通讯]` (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了PPO算法在使用高斯策略时，标准差σ收敛导致的学习弹性丧失（Loss of Plasticity，LoP），通过理论推导和实验验证，证明σ→0会导致梯度范数发散、Hessian锐度增大及其有效秩坍塌，进而解释LoP的机制并解释了现有缓解方法（如Adam β1=β2、熵正则化）的有效性。

**💡 创新点**

创新点在于首次将策略标准差与LoP建立严格的数学关联，给出了梯度范数、Hessian尖锐度和近似秩的上界/下界，并利用这些理论阐释了 Adam 参数调节、熵正则化以及其他LoP缓解技术的本质；同时通过实验证明这些理论在模拟和真实机器人任务中均成立。

**🔧 技术方法**

采用PPO与等方差高斯策略的组合，进行积分可积性假设下的梯度与Hessian解析；使用谱分析和矩阵分解推导LoP指标；实验使用Adam、RMSProp、Tuned Adam（β1=β2）以及L2正则化、CBP等策略；在模拟环境中利用Isaac Lab、Slippery Ant进行训练，在真实环境中部署Franka Emika Panda机器人。

**📊 数据集**

数据集包括：Isaac-Reach-Franka-v0（Franka Panda 7-DoF 操作任务）、Slippery Ant（MuJoCo Ant 任务，周期性摩擦系数变化）以及在真实Franka Emika Panda机器人上收集的交互数据。

**📈 对比分析**

方法比较：对比标准PPO、Tuned Adam（β1=β2）、L2正则化、CBP以及自适应学习率；评估指标为奖励曲线、σ演化、梯度范数与最大Hessian特征值；实验结果显示：Tuned Adam能延迟LoP崩溃，L2和CBP能完全抑制崩溃；通过匹配σ的演化，标准PPO可获得与L2相近的性能；在真实机器人部署中，reset σ策略获得的平均奖励≈0.045，显著优于基线（≈-0.01）。

**⚠️ 局限性**

局限性包括：理论推导假设σ为等方差，实际情况常为方差矩阵；常数κ、K、χ、C的估计困难；仅在PPO框架下验证，其他RL算法可能不适用；实验主要集中在Franka机器人，泛化到更广泛的硬件或任务尚未充分验证。

---

## 141. When Retrieval Is Not Enough: Obsolete Facts in Agent Memory

**arXiv ID:** 2610.36308 | [PDF](https://arxiv.org/pdf/2610.36308v1)

**作者:** Oleeviya Babu Poikarayil `[一作]` (Otto-von-Guericke-University Magdeburg), Klaus Turowski `[通讯]` (Otto-von-Guericke-University Magdeburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过在MemoryAgentBench的FactConsolidation任务上，系统性探讨了检索预算、无关证据以及过时竞争性事实对记忆代理答案准确性的影响；

**💡 创新点**

创新点在于通过对证据组成的精细干预，明确显示当检索到最新更新时，若同时存在过时竞争事实，模型往往会选择旧事实，从而提出评估时需分离检索成功与使用成功的必要性；

**🔧 技术方法**

采用BM25检索、Qwen2.5-7B-Instruct与Qwen3-14B两大模型、确定性解码、等量替换干预、黄金辅助审计及McNemar检验等技术；

**📊 数据集**

使用MemoryAgentBench的Synthetic FactConsolidation数据集，包括6K与32K历史版本共100道问答，其中包含冲突与非冲突两类问题；

**📈 对比分析**

通过比较不同检索预算（64、128、256、512 token）与等量替换对（A/B）的准确率，发现检索预算增大会显著降低准确率（例如Qwen2.5从100%降至44.6%），Qwen3相对更稳健但同样受影响；

**⚠️ 局限性**

局限性包括仅评估两款Qwen模型、仅使用确定性解码、仅针对单跳合成数据、未测试稠密检索或多跳依赖、以及金手指审计仅为诊断工具，缺乏跨模型与真实场景的广泛验证。

---

## 142. Privately Clipping Heavy-Tailed Data

**arXiv ID:** 2610.36281 | [PDF](https://arxiv.org/pdf/2610.36281v1)

**作者:** Matthew Joseph `[一作]` (Google Research), Alex Kulesza `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全差分隐私的自适应裁剪方法，用于在对高维向量求和时自适应选择裁剪阈值 C。

**💡 创新点**

创新点在于引入残差共线性作为低敏感度指标，利用其与最优裁剪阈值的关系来平衡偏差与噪声，并证明其与 MSE 最优点的关联。

**🔧 技术方法**

采用残差共线性度量、基于 ρ‑CDP 高斯噪声机制以及稀疏矩阵预计算实现的高效算法。

**📊 数据集**

在十个互联网用户行为数据集（Steam、Reddit、LastFM、HN、MovieLens、Pantry、Twitch、Foursquare、Jester、Slashdot）以及四个轻尾数据集上进行实验。

**📈 对比分析**

与两种基于分位数的裁剪基线（0.9/0.95/0.99）进行比较，平均 MSE 仅提升不到 1.32 倍，在 6/10 数据集上取得最佳表现，整体误差显著低于基线。

**⚠️ 局限性**

理论证明仅给出足够条件而非必要条件，对残差共线性的假设仅近似满足，且未在 DP‑SGD 场景中验证该方法。

---

## 143. SpecGuard: Detecting Multi-Turn LLM Jailbreaks via Specification-Based Runtime Monitoring

**arXiv ID:** 2610.36258 | [PDF](https://arxiv.org/pdf/2610.36258v1)

**作者:** Junle Li `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Bingzhuo Zhong `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpecGuard，利用规范驱动的运行时监控框架检测多轮 LLM jailbreak，自动将对话映射为可审计的原子命题（AP）序列，并用 LTL_f 规范在有限状态机上进行监控，生成可重放的违规证据。

**💡 创新点**

创新点在于：①将安全需求形式化为可执行的 LTL_f 规范并通过 AP 合同固定事实抽象；②实现训练无关、无模型内部访问的检测；③提供完整的审计路径，能够重现导致违规的对话段落；④在多轮攻击情境下通过时间关系提升检测准确性。

**🔧 技术方法**

主要技术包括：原子命题（AP）合同与判定器、LTL_f 规范编译成符号 DFA、运行时与审计模式下的状态机递归更新、证据追溯与可重放的违规 witness、基于原子命题的离散事件抽象。

**📊 数据集**

使用的数据集为 SafeDialBench‑ZH、SafeDialBench‑EN（多语言多轮安全基准）和 SafeMTData（ActorAttack 轨迹基准），共计约 10,000 条对话，覆盖中文、英文与多轮攻击。

**📈 对比分析**

与四个公开权重守卫模型（ShieldGemma‑2B、WildGuard‑7B、BingoGuard‑8B、GuardReasoner‑8B）在相同输入下对比；SpecGuard 在 AUC、F1、召回率、BA 上均优于所有基线，FPR 维持在可接受范围；在无阈值的 AUC 评估中表现尤为突出。

**⚠️ 局限性**

局限性：①依赖人工设计的 AP 及规范，可能对新型攻击缺乏覆盖；②仅针对文本交互，尚未覆盖多模态场景；③无学习过程，需人工更新规范以应对攻击演化；④在极端长对话中可能仍存在误报/漏报风险。

---

## 144. What Should an Information Bottleneck Encoder Learn? Separating Representation Learning from Rate Control

**arXiv ID:** 2610.36255 | [PDF](https://arxiv.org/pdf/2610.36255v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究信息瓶颈（IB）模型中，在给定压缩预算的情况下，探索是否可以先训练一次无压缩依赖的表示 F，然后为每个预算仅训练压缩器 C 与解码器 G，形成分离式的训练管道。

**💡 创新点**

创新点：①证明在联合高斯数据下，IB 解决方案可分解为预算无关的表示 W 与仅随 β 变化的缩放 D(β)；②提出 sufficiency deficit ϵ_F 用来衡量分离式表示的偏差，并给出最坏情况下的误差上界；③在非高斯模型与多域数据场景中给出理论保证，并验证其在实际任务中的有效性。

**🔧 技术方法**

使用信息瓶颈理论、变分信息瓶颈 (VIB)、深度学习编码器/解码器网络、可分离压缩层、熵与互信息分析、实验中的计算量与时间评估。

**📊 数据集**

实验数据集包括 MNIST、SVHN、CIFAR-10、STL‑10（四个不同域）以及它们的合并测试集。

**📈 对比分析**

与传统的全端到端 VIB、FiLM 条件化 VIB、OIB 等基线比较；分离式管道在 8 个预设预算下，平均提升 8–10 个百分点的准确率；在高压缩率（λ 较大）时避免 VIB 崩溃；训练 MAC 与墙钟时间分别降低约 7.7 倍与 2 倍；但在低压缩率时交叉熵和校准误差略高。

**⚠️ 局限性**

局限性：①分离式表示对 ϵ_F 的估计依赖于上界，可能不足以捕捉所有非高斯偏差；②在多域拼接时需额外考虑域对齐项，表现稍逊于单域；③在低压缩率下交叉熵与校准不如 VIB；④需要在推理时为每个预算部署单独的压缩器/解码器，增加模型管理复杂度。

---

## 145. Hierarchical Multimodal Interaction Reasoning for Dyadic Relationship Recognition

**arXiv ID:** 2610.36249 | [PDF](https://arxiv.org/pdf/2610.36249v1)

**作者:** Wang Tang `[一作]` (Chengdu University of Information Technology), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了HiRe-DRR框架，基于层次多模态交互推理实现双人关系识别

**💡 创新点**

创新点包括层次化特征分层、同异模态交互建模、稀疏关系感知的MoI-Former以及层次条件损失HC-Loss

**🔧 技术方法**

使用多模态特征提取器（OpenFace、Pose、Librosa、BERT等），Transformer稀疏路由器，交互矩阵构造，HC-Loss等技术

**📊 数据集**

在NoXi、UDIVA和Seamless Interaction三个基准数据集上进行实验

**📈 对比分析**

与多种基线（GR2N、HGM、PGCN、LIReC、CT、AsyReC）比较，HiRe-DRR在UAR、mAP等指标上领先10%~20%

**⚠️ 局限性**

局限性包括对长时序关系建模不足，依赖短期交互，难以处理社交边缘关系的模糊性

---

## 146. Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures

**arXiv ID:** 2610.36229 | [PDF](https://arxiv.org/pdf/2610.36229v1)

**作者:** Kosei Horikawa `[一作]`, Ahmed E. Hassan `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在已移除AI编码代理（如GitHub Copilot、Tabnine等）署名的情况下，是否能够通过提交内容（代码差异、提交信息等）识别出由代理产生的提交。

**💡 创新点**

创新点在于：①首次在大规模、跨仓库的真实提交中评估签名移除后代理提交的可识别性；②通过对比不同特征（消息长度、变更大小、代码风格等）来剖析导致可识别的根本原因；③系统性比较多种检测器（XGBoost、TF‑IDF、CodeBERT、LLM判别器）及其在未见代理、不同时间段上的迁移性能。

**🔧 技术方法**

技术手段包括：①特征工程——构造 11 类特征（提交信息、文件变更、代码词汇、代码风格、变更计数等）；②机器学习模型——XGBoost（主模型）、LightGBM、CatBoost、TF‑IDF + L2 正则化逻辑回归、CodeBERT embeddings + Logistic；③大语言模型评估——ChatGPT/Claude 等零/少样本提示；④严格的交叉验证与置信区间估计（基于仓库抽样的 Bootstrap）。

**📊 数据集**

使用的数据集来自 6,700+ GitHub 仓库，覆盖 1.2M 以上默认分支提交，分为 3 个时代（pre‑LLM、LLM、Agent），共标注了 5 种主流 AI 代理和大量无签名人类提交。

**📈 对比分析**

比较方法：在保持仓库和时间离散的严格测试集上评估模型，报告 ROC‑AUC、PR‑AUC、在实际代理比例下的 Precision/Recall/F1。结果显示：在签名被掩码后，XGBoost 可达 AUC≈0.75‑0.80，TF‑IDF 约 0.70‑0.78；但在实际代理比例（≈5%）下，Precision 最高也只有约 0.30‑0.35，说明仅凭内容难以准确标记。对未见代理或未来月份的迁移实验表明性能显著下降，表明模型高度依赖代理特定的消息模板和变更模式。

**⚠️ 局限性**

局限性包括：①仅关注已公开的默认分支提交，未覆盖私有仓库或被删除的历史；②人类提交标签并非完全排除 AI 参与，可能导致标签偏差；③特征主要聚焦于文本和大小，未深入学习代码语义变化；④对 LLM 判别器的评估依赖预训练模型，可能受到数据泄露影响；⑤模型的可迁移性差，需持续更新以跟踪代理的版本与实践变更。

---

## 147. Repulsive Population Policy Gradients for Behaviorally Diverse Reinforcement Learning

**arXiv ID:** 2610.36425 | [PDF](https://arxiv.org/pdf/2610.36425v1)

**作者:** Saeed Hedayatian `[一作]` (University of Southern California), Stefanos Nikolaidis `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种直接训练紧凑策略族的QD‑RL算法RePPO，利用Soft QD目标将种群梯度转化为形状奖励并用PPO更新。

**💡 创新点**

创新点在于：① 将Soft QD的连续种群目标映射为每个策略的可微形状奖励；② 在不使用离散档案或进化算子、无需对行为空间离散化的情况下实现高质量与多样性的平衡；③ 通过动态计算“排斥”向量并将其投射到行为信号上，构造多样性权重并与任务优势结合。

**🔧 技术方法**

使用的技术包括：Soft QD下的低阶可计算目标、可加马尔可夫行为描述符、PPO（包含多通道优势标准化与裁剪），以及经验回放与EMA平滑估计种群质量与行为。

**📊 数据集**

在四类数据集上进行评估：连续控制（Brax Ant、HalfCheetah、Humanoid、Walker2D）、像素基础的Atari游戏（Boxing、Asterix、Seaquest、Ms. Pac‑Man）、以及大型语言模型的数学推理任务（MATH-500、Minerva、OlympiadBench，基于Qwen3‑1.7B和Granite‑4.1‑3B）。

**📈 对比分析**

与独立PPO、PPGA、PGA‑ME、DCRL‑ME、Sep‑CMA‑MAE（在连续控制）以及RefQD、提示式基线（在Atari与LLM）进行比较。RePPO在连续控制任务上实现了最高的qVS（质量×多样性）并保持与最佳单个策略相当的平均质量；在Atari中保持了与独立PPO相当的平均得分同时显著提升多样性；在LLM微调中生成了风格多样且可区分的模型，整体准确率与独立PPO相近。

**⚠️ 局限性**

局限性包括：① 需要先验的可加马尔可夫行为描述符，无法直接适用于复杂或未知的行为；② 通过种群规模控制多样性，规模增大时训练成本线性增长，仍受限于算子计算和存储；③ 只优化紧凑种群，无法提供全局覆盖；④ 在某些任务中，排斥项可能导致个体质量下降，需额外稳定性控制。

---

## 148. SO-101 Bench: Accessible Spatial Manipulation

**arXiv ID:** 2610.36420 | [PDF](https://arxiv.org/pdf/2610.36420v1)

**作者:** Truman Hickok `[一作]` `[通讯]` (Independent), Truman Hickok (Independent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 Isaac Lab 的 SO-101 机器人数字孪生的语言条件空间操控基准，涵盖四个任务、138 个家庭物体，并提供 916 条评估剧本和 3,629 条模拟演示。

**💡 创新点**

创新点在于提供一个可在仿真中准确预测真实机器人性能的基准，定义了对象实例/类别拆分、ROBUST/PLAN/MEM 子集，并证明仿真排名与真实机器人成绩一致。

**🔧 技术方法**

采用 Isaac Lab 仿真、视觉语言动作模型（VLA）、世界动作模型（WAM）、LLM 代码驱动（PyRUA-Lean）以及多种抓取策略。

**📊 数据集**

使用 138 个扫描或 AI 生成的物体数据集（见 138 件）以及 3,629 条仿真 Teleop 记录和 4,400 条真实机器人演示。

**📈 对比分析**

通过统一 36 小时 4xA100 的微调预算，对 9 个 VLA/WAM 和 2 个 LLM 代理进行公平比较，最佳 VLA/WAM 约 61% 成功率，LLM 代理 83%；在真实机器人上仿真排名与实测一致。

**⚠️ 局限性**

局限包括：仅评估 5 个无真实数据的策略、单一环境与 45 条布局、模拟成功率高出 9–31%，未覆盖柔性物体或双臂；且代理的实时成本被隐藏。

---

## 149. Persistence, Exposure, and Credit Routing in Linear Temporal Predictive Coding

**arXiv ID:** 2610.36412 | [PDF](https://arxiv.org/pdf/2610.36412v1)

**作者:** Akira Tanase `[一作]` (Sakana AI), Jeffrey Seely `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文利用细胞外壳理论分析线性Temporal Predictive Coding网络的时空信用分配，提出持久性(P)与曝光(Q)算子，并通过实验验证全信用学习可实现延迟回忆中的模式路由。

**💡 创新点**

创新点在于将P、Q两个持久-曝光算子嵌入细胞外壳框架，阐释空间与时间错误的权衡，并揭示标准局部更新忽略的未来信用项。

**🔧 技术方法**

使用细胞外壳理论、定向网络外壳扩散、线性代数分析以及Adam优化的全信用与局部信用学习对比等技术。

**📊 数据集**

实验数据集包括合成的延迟回忆任务（随机提示+高斯干扰）和30位数字复制任务。

**📈 对比分析**

通过将标准局部tPC的权重更新与全信用BPTT式更新对比，MSE从约0.97降至约0.005，证明全信用能有效学习模式路由。

**⚠️ 局限性**

局限在于标准局部tPC缺乏跨时间信用，无法学习长程路由；且实验仅验证线性网络，非线性推广仍待研究。

---

## 150. Automatic Certified Differentiation in Lean

**arXiv ID:** 2610.36396 | [PDF](https://arxiv.org/pdf/2610.36396v1)

**作者:** Ramy Shahin `[一作]` `[通讯]` (Qualgebra), Ramy Shahin (Qualgebra)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用 Lean 4 与 Mathlib 进行自动化的函数求导，并生成对应的导数函数、有效域以及完整的证明。

**💡 创新点**

创新点在于：1) 通过已有的数学定理而非手写规则实现求导；2) 采用定理索引和元编程搜索，提高效率；3) 直接返回包含证明的自包含对象，使得结果可直接使用且保证正确性。

**🔧 技术方法**

主要技术包括 Lean 4 证明助手、Mathlib 的一维导数定理、元编程（tactic）实现求导算法、定理索引（按头函数符号）与搜索策略、以及与 Lean 核心的交互。

**📊 数据集**

使用了 500 个随机生成的实数一元函数（表达式树深度≤7），包含整数常数、变量幂、四运算、三角函数与对数，所有函数均满足 Lean 的类型类约束。

**📈 对比分析**

对比方法：在相同测试集上分别使用有索引和无索引两种配置，跑 5 次后取平均。结果显示，中位数求导时间从 181 ms 降至 21 ms（约 8.6×加速），最快 71×，全部 500 例均能成功完成，且在无索引时有 12 例超时。

**⚠️ 局限性**

局限性：仅支持一维一阶导数；生成的域是充分条件而非最优；对多变量或高阶导数需扩展；搜索仍有改进空间；依赖 Lean 4 环境，需手动导入相关定理。

---

## 151. When Can a Learned Design Surrogate Be Trusted in a Self-Driving Microfluidics Laboratory? An Offline Closed-Loop Framework Built and Tested on Six Public Tables

**arXiv ID:** 2610.36395 | [PDF](https://arxiv.org/pdf/2610.36395v1)

**作者:** Zhiyuan Zhang `[一作]` `[通讯]`, Zhiyuan Zhang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在离线实验环境中，将六张公开的微滴/脂质体实验表格视为自驱动实验室，提出并实现了四个关键决策（模型可靠性、测量预算、采样顺序与后续决策）的可量化框架，并在严谨的预注册、密封验证及冻结协议下完成了107个目标组的闭环模拟，归纳出23条研究发现。

**💡 创新点**

① 将自驱动实验室中的四个决策问题形式化为可度量的指标；② 首次对微滴设计代理采用密封预测可靠性验证；③ 在离线实验室中系统评估了不同转移类型、warm‑start残差提升、区间预测以及决策规则的效果；④ 通过预注册与冻结协议实现了可复现的严格实验流程。

**🔧 技术方法**

核心技术包括：XGBoost基线重实现；物理约束的V2回归模型；warm‑start残差提升（XGBoost+源模型）；分层与CQR的分割共形预测、Mondrian区间；量化不确定性的分位树、GP和深度集成；离线闭环模拟、块 bootstrap、Benjamini‑Hochberg 校正等统计方法。

**📊 数据集**

使用的六份公开表格：
• DAFD 3.0（868 行+64 外部文献）
• Chagot 等（468 行）
• Smart Microfluidics 脂质体（246 训练 + 57 密封）
• Hormozinezhad 等非牛顿滴（901 行）
• Wang 等共流滴（830 行）
• Song 等脂质纳米粒子（919 行）。
共 107 个留一组目标层。

**📈 对比分析**

与基线 XGBoost 的内部拆分误差 5.3% 相比，外部文献转移误差升至 11%；warm‑start 残差提升在 20 次测量内使 94% 的化学转移层达到 ≤1.25× 开发误差；测量预算 k* 随转移类型（化学、设备、实验室）不同；决策规则在大多数情形下能降低实验成本；区间预测与无标签的采样规则在随机标签控制下未显示显著优势。总体而言，模型对不同转移的泛化表现有限。

**⚠️ 局限性**

限制：未进行实际湿实验验证；仅涵盖微滴与脂质体两类实验；结果高度依赖转移类型，许多期望方法（无标签裁剪、区间采样）表现不佳；离线模拟可能低估真实实验噪声；仅使用六份公开数据，难以推广至更广泛的实验室与设备。

---

## 152. Evaluating Federated Vision-Language and Vision-Only Models for Privacy-Aware Breast Cancer Imaging

**arXiv ID:** 2610.36379 | [PDF](https://arxiv.org/pdf/2610.36379v1)

**作者:** Malithi Mithsara Wanniarachchi Kankanamge `[一作]` (Southern Illinois University), Abdur R. Shahid `[通讯]` (Southern Illinois University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究在隐私保护的联邦学习框架下，评估预训练的视觉语言模型（CLIP、BiomedCLIP）与纯视觉模型（ViT）在乳腺热成像二分类任务中的表现；并探讨去除文本路径、差分隐私对模型精度、通信成本及模型一致性的影响。

**💡 创新点**

首次将视觉语言模型与客户端级差分隐私相结合，系统比较文本路径与视觉路径对分类性能与通信开销的贡献，并在患者级别使用置信区间和基线对模型进行严格评估；同时揭示差分隐私实现方式对模型性能的严重影响。

**🔧 技术方法**

使用联邦平均（FedAvg）框架、客户端裁剪+高斯噪声差分隐私、CLIP（OpenAI、LAION）、BiomedCLIP、ImageNet预训练的ViT、视觉‑仅 CLIP、数据增强、焦点损失、FedProx 以及通信与时间测量等技术。

**📊 数据集**

两组公开乳腺热成像数据集：BCD（正常 vs 病变）和 Breast Thermography（良性 vs 恶性），分别包含52和24名患者的测试集。

**📈 对比分析**

在FL与FL+DP两种设置下，使用患者级ROC‑AUC和95%置信区间进行对比。BCD任务在无隐私时AUC 0.94–0.99，加入DP后平均下降约0.37；Thermography任务AUC 0.50–0.64，DP效果不显著，所有CI均覆盖0.5。不同模型在通信量上差异显著，去除文本路径可将通信量减半，而精度基本保持。

**⚠️ 局限性**

限制包括：测试样本患者数少、单随机种子、联邦环境仅模拟、差分隐私仅评估单点配置、隐私预算相对较大、未与中心化基线对比、未进行梯度重构或成员推断攻击、通信/计算成本测量不完整、仅评估二分类且未充分验证语言模型在更复杂任务中的优势。

---

## 153. IOSage: Detecting HPC I/O Bottleneck Patterns to Guide LLM Recommendations

**arXiv ID:** 2610.36370 | [PDF](https://arxiv.org/pdf/2610.36370v1)

**作者:** Mahdi Banisharif `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用训练好的多标签分类器先检测Darshan日志中的七种I/O瓶颈模式，再将检测到的模式和对应的模式卡传给LLM，让LLM基于日志、配置文件和模式卡生成可执行的I/O优化建议，并在真实应用上验证加速效果。

**💡 创新点**

将瓶颈模式检测从LLM迁移到基于数据驱动的分类器，形成精度门控；引入基于基准运行生成的模式卡作为知识库，使LLM仅在已知模式下给出具体改动；同时通过配置标注的训练集实现跨应用的无监督检测。

**🔧 技术方法**

使用XGBoost多标签分类器、TreeSHAP特征重要性、Darshan计数器特征提取、Claude Sonnet 4 / GPT‑4o / Llama 3.1 大语言模型、知识库模式卡、配置信息提示、微调与阈值门控。

**📊 数据集**

来自阿尔科弗的Polaris生产Darshan日志（131,151条）与七个基准工具（IOR、DLIO、h5bench、mdtest、HACC‑IO、mpi4py等）的配置标注日志（689条）以及TraceBench真实工作负载的9条专家标注日志。

**📈 对比分析**

与规则驱动工具Drishti、WisIO、LLM工具IOAgent以及无门控的LLM进行多标签F1、精度、召回等指标对比，分类器在测试集上实现Micro‑F1 0.929，超过规则工具；在8个真实应用场景中，Claude Sonnet 4在有分类器门控下成功推荐5/6个加速改动，平均加速4.38×（openPMD）至1.93×（WRF 10‑min重启）。

**⚠️ 局限性**

仅覆盖七个预定义模式，无法发现新型瓶颈；需要基准运行标签进行训练，缺乏跨文件系统或无POSIX日志的泛化能力；模式卡无法提供对应用源代码层面的定位；对未检测到模式的日志仍可能产生误推荐；对大规模时序特征缺乏支持。

---

## 154. WAKMan: Workload Aware KVCache Manager

**arXiv ID:** 2610.36355 | [PDF](https://arxiv.org/pdf/2610.36355v1)

**作者:** Suhas Kumar Morisetty `[一作]` (Columbia University), Wesley Harris `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种工作负载感知的 KV 缓存管理器 WAKMan，能在不知请求类型和提示内容的前提下，动态选择最优的 KV 缓存失效策略，从而提升 LLM 推理的前缀缓存命中率。

**💡 创新点**

创新点在于：1) 采用实时流量重放的方式在服务器内部同时维护多种失效策略（如 LRU、S3FIFO），并根据最近窗口的模拟命中率自动切换；2) 不需要离线训练、请求类型标签或提示文本，完全基于已存在的 KV 哈希信息；3) 通过实验验证该方法在多种单轮、多轮和 agentic 混合工作负载中，平均提升约 5% 的命中率，且与理想策略的差距最小。

**🔧 技术方法**

技术包括：Shadow LRU 与 Shadow S3FIFO 的离线模拟器、基于最近 T 分钟窗口的流量记录与重放、决策阈值比较以及在 vLLM 中动态切换失效策略的实现。

**📊 数据集**

使用合成工作负载，模拟单轮、双轮（多轮）和 agentic 请求，控制到达速率、提示长度、会话长度等参数，生成多种混合比例（共 20 组）进行实验。

**📈 对比分析**

对比方法：与 vLLM 默认 LRU、单一 S3FIFO、以及理想无穷缓存三者的命中率进行比较。实验显示：LRU 在 agentic 主导场景表现最好；S3FIFO 在单轮或多轮主导场景表现更优；WAKMan 在所有 20 组混合中平均减少约 5% 的误差，最大 regret 仅为 11%，远低于两种静态策略的最大 regret。

**⚠️ 局限性**

局限性包括：只考虑了 LRU 与 S3FIFO 两种候选策略；若加入更多可重放的策略（如 ARC、LeCaR 等）可能进一步提升性能；未探索动态调整窗口大小 T 的最佳策略；实验仅基于合成数据，缺乏真实生产流量的验证。

---

## 155. Callosal: Rethinking the Kernel Page Cache for Disaggregated Storage and Memory

**arXiv ID:** 2610.36347 | [PDF](https://arxiv.org/pdf/2610.36347v1)

**作者:** James McMahon `[一作]`, Anurag Khandelwal `[通讯]`

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于RDMA和可编程交换机的跨节点页缓存一致性系统（CallosalFS），将分布式一致性直接嵌入Linux内核页缓存层，实现强一致性并支持分布式存储。

**💡 创新点**

创新点在于：1）将页缓存视为共享一等对象，直接在内核层实现分布式一致性；2）利用网络加速的分布式共享内存（MIND）和网络内一致性目录，消除传统文件系统的中心化瓶颈；3）通过MSI式失效协议和远程锁实现低延迟、可扩展的页级一致性。

**🔧 技术方法**

技术手段包括：Linux内核页缓存修改、RDMA通信、可编程交换机实现的MIND一致性协议、MSI失效协议、共享虚拟地址空间、远程读写锁同步以及基于版本号的页面验证。

**📊 数据集**

使用了标准文件系统基准（DRBL、DRBM、DRBH、DWOL、DWOM、DWOH）、生产者-消费者队列、文件化键值存储（约240k键）等自定义工作负载，而非公开数据集。

**📈 对比分析**

通过与NFSv4、CephFS、ext4、tmpfs等系统在四台服务器上比较，发现：在低争用读写下接近本地文件系统性能；混合读写场景中比NFS提升4.6×，比CephFS提升约10×；键值存储在4节点时达到约4.5Mops，扩展系数为1.72；单节点读写线性扩展。

**⚠️ 局限性**

主要限制包括：写回与持久性未完整实现；DSM预分配且缺少动态分配；仅支持文件式共享状态；不支持全部元数据同步；在高冲突写入场景下吞吐仍受限。

---

## 156. PHP2Shell and the Future of AI-Discovered Vulnerability Disclosure: Evidence, Verification, and Human Authority

**arXiv ID:** 2610.36334 | [PDF](https://arxiv.org/pdf/2610.36334v1)

**作者:** Toqeer Ali Syed `[一作]` (Islamic University in Madinah), Asadullah Abdullah Khan `[通讯]` (Islamic University in Madinah)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 FedTrust‑CVDv2 框架，在 AI 辅助漏洞发现中将证据验证、跨组织协调、人类治理和可许可审计解耦，并在本地实现对构造测试用例的验证。

**💡 创新点**

创新点在于提供案例级别的证据到行动授权契约，将技术证据、独立观测、人类授权、接收者、payload、政策与撤销状态绑定到同一案例修订，确保证据变更时自动失效；同时在多层架构中明确区分验证、协调、治理和审计。

**🔧 技术方法**

使用了 Python 3.12、Ed25519 签名、SHA‑256 哈希、AES‑GCM 加密、RFC8785 标准 JSON、BFT 排序子系统（如 Hyperledger Fabric 或 PBFT）以及签名、加密、审计和法定合约逻辑。

**📊 数据集**

使用了 27 个政策家族、每族 25 个身份独立副本的构造 fixture 集合，共 675 个测试用例（25 个有效、650 个无效）。

**📈 对比分析**

对构造的验证门进行定量评估：完整门接受所有 25 个有效 fixture 且拒绝所有 650 个无效；在六个单一保护移除的消融实验中，分别暴露了 25–75 个无效 case 的通过。相比传统 CVD 或 AI 生成报告的无约束发布，门实现了高精度的授权控制。

**⚠️ 局限性**

局限性包括：未验证 AI 漏洞检测的准确性；未评估真实攻击减少、披露速度或维护者信任提升；缺乏真实组织工作流的部署评估；对人类审核负荷、共模错误和关键撤销等实际操作风险仍未彻底评估。

---

## 157. Ontology-guided multi-agent reasoning for explainable differential diagnosis

**arXiv ID:** 2610.36325 | [PDF](https://arxiv.org/pdf/2610.36325v1)

**作者:** Roxana Ramona Szomiu `[一作]` (Technical University of Cluj-Napoca), Adrian Groza `[通讯]` (Technical University of Cluj-Napoca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一套多代理系统 MedMAS-DDx，将大型语言模型用于自然语言理解，使用医学本体（SYMP、DOID、HPO）和知识图谱进行符号推理，生成可追溯、可解释的差异诊断过程，并通过有针对性的问答来逐步填补信息缺口。

**💡 创新点**

创新点包括：① 神经符号融合，LLM 负责文本理解，符号推理基于本体；② 多代理协同，采用 CrewAI 框架与 deterministic 控制层确保可预测、可审计；③ 基于本体的症状映射与特异性权重评分，避免 LLM 生成 hallucination；④ 交互式问答机制，根据信息增益动态提问；⑤ 可解释的诊断报告，呈现每一步分数、支持/冲突证据。

**🔧 技术方法**

使用技术：大型语言模型（LiteLLM 接口调用 GPT‑4 等），CrewAI 架构的多代理系统，GraphDB 存储医学本体并通过 SPARQL 查询，Python deterministic 代码实现特异性权重、覆盖度、组合分数、信息增益等评分；SapBERT 进行语义相似度评估；实验基于 Q4Dx benchmark。

**📊 数据集**

数据集：Q4Dx benchmark（PatientCases.csv，99 病例，29 诊断），在 80% 与 50% 症状完整度两种设置下进行评估；使用全症状版本进行模拟患者回答以评估交互式问答效果。

**📈 对比分析**

与 MedQDx（GPT‑4.1 开放词表分类器）比较。MedMAS‑DDx 0‑shot Top‑1 18.7%→21.2%，Top‑3 25.8%→27.3%；在 80% 完整度下 Top‑1 21.2% 对比 MedQDx 44%，Top‑3 27.3% 对比 30.3%；在 50% 完整度下 Top‑1 18.2% 对比 MedQDx 37%，Top‑3 24.2% 对比 24.2%。交互式问答显著提升诊断信心，尤其在信息缺失较多时能恢复到与完整病例相当的相似度。

**⚠️ 局限性**

局限性：① 依赖本体与知识图谱完整性，缺失或不完整关联导致诊断失误；② 仅使用症状信息，缺乏实验室、影像等检验数据；③ 交互模式仅限 Yes/No/Unsure，无法表达症状严重度或时间序列；④ 需要较多问答轮数（约 10 轮）以提升准确性，交互成本高；⑤ 由于 deterministic 控制层，系统无法生成本体中未定义的诊断标签；⑥ 评估基准规模小，缺乏真实临床数据验证。

---

## 158. Property-Based Testing for Robust Softmax and Cross-Entropy

**arXiv ID:** 2610.36427 | [PDF](https://arxiv.org/pdf/2610.36427v1)

**作者:** Muhammad Maaz `[一作]` `[通讯]`, Muhammad Maaz

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

通过属性化测试发现并修复了标准机器学习库中softmax与交叉熵函数的数值崩溃和消除问题，并提出了更稳健的实现。

**💡 创新点**

将Hypothesis属性化测试用于验证softmax数学性质失效，并引入基于log1p的归一化方法解决极端阈值下的NaN梯度。

**🔧 技术方法**

使用Hypothesis属性化测试、PyTorch自动微分、max-shift技巧以及log1p与softplus的归一化比较。

**📊 数据集**

未使用固定数据集，采用Hypothesis随机生成的浮点数列表进行测试。

**📈 对比分析**

通过对Jacobian行和梯度为零的属性测试进行比较，新的实现与PyTorch及之前修复版本均通过所有测试，梯度稳定且性能与标准实现相当。

**⚠️ 局限性**

仅针对softmax和交叉熵两函数进行数值稳定性评估，缺乏在更复杂网络或大规模训练中的全面验证。

---

## 159. When AI Can Build It, What Should We Assess? Rethinking Evidence of Engineering Competence

**arXiv ID:** 2610.36413 | [PDF](https://arxiv.org/pdf/2610.36413v1)

**作者:** Ali Shakiba `[一作]` `[通讯]` (University of Sydney), Ali Shakiba (University of Sydney)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一门13周的硕士软件工程课程中，重新设计并首次部署了允许生成式AI协助的教学与评估体系，包括工程责任工作流、六通道证据架构以及收集–交叉检查–升级(CCE)推理逻辑，以在AI参与的前提下保留学生对工程决策的责任并获得可靠的能力证据。

**💡 创新点**

创新点在于提出“工程责任工作流”来明确人机协作的决策节点，构建“六通道证据架构”（产品、过程/来源、验证、解释、反思、个人保证）并通过CCE逻辑实现多证据的协同解释，从而在AI能大量产出 artefact 的环境下仍能有效评估学生的工程能力。

**🔧 技术方法**

采用 GitHub 进行版本控制与 CI/CD、Gradescope 进行自动化评分、Visual Studio Code 插件 AI Journal 记录 AI 使用、自动化仓库创建与提交、以及自研的仓库分析工具来收集与展示多维度评估证据。

**📊 数据集**

使用的数据主要为学生个人和团队的 Git 仓库提交记录、AI Journal 条目、测试与 CI 结果、反思日志以及考试成绩等内部数据；未使用公开的外部数据集。

**📈 对比分析**

在课程中通过对评估指标权重的调整、对证据完整性与一致性的监测来评估设计改进的有效性，但未进行大规模对照实验或数值性能对比，仅通过课程实施过程中的观察与教师反馈总结改进效果。

**⚠️ 局限性**

局限性包括：仅在单一学期、单一机构的 87 名学生中实施，缺乏对学习成效的量化对照；评估解释仍高度依赖教师主观判断，自动化收集虽易实现但解释与交叉检查仍需大量人工；AI 记录可能导致学生行为趋向合规而非真实思考。

---

## 160. From Generation to Scoring: Scaling LLM-Based Semantic Labeling at LinkedIn

**arXiv ID:** 2610.36383 | [PDF](https://arxiv.org/pdf/2610.36383v1)

**作者:** Jian Sheng `[一作]` (LinkedIn), Fedor Borisyuk `[通讯]` (LinkedIn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将 LinkedIn 生产级 LLM 标注器从批量生成改为语义评分，通过单候选提示、仅标注 fine‑tune、答案标记读取以及 Multiple‑Item Scoring (MIS) 实现一次预填即可得到所有评分。

**💡 创新点**

创新点在于将生成式标签器拆分为独立判定的评分器，使用单项训练、答案标记读取与 MIS 的候选隔离注意力掩码，既保证判定独立性又显著提升吞吐量并降低训练数据需求。

**🔧 技术方法**

技术手段包括：因果语言模型 fine‑tune、答案标记读取、MIS（使用注意力掩码和重用位置 ID）、prefill‑only 推理、Ray Data 统一 MIS API、SGLang/FlashInfer 加速。

**📊 数据集**

使用 LinkedIn 会员资料数据集，约 500K 会员、每位 200 个候选资格，共 1 亿个 profile‑qualification 对，参考标签为 50% 正例的二元标注。

**📈 对比分析**

与基线 packed‑generation 进行对比：在 45K 训练对上语义评分达 86.3% 一致率，生成仅 50.8%；一致性率<0.01% 对比 0.86%；吞吐量提升 1.24×（离线）和 1.9×（在线），且仅需 4.7× 的训练样本。

**⚠️ 局限性**

局限性：仅评估与参考标签的一致性，未考察 downstream 影响；方法仅适用于独立二元判定；提示/阈值需统一版本，MIS 对请求大小敏感；小模型的质量尚未完全验证。

---

## 161. Bayesian Replicator Dynamics and the Inversion of Strict Dominance in Bimatrix Games

**arXiv ID:** 2610.36345 | [PDF](https://arxiv.org/pdf/2610.36345v1)

**作者:** Jaime Ruiz-Serra `[一作]` (University of Sydney), Michael Harré `[通讯]` (University of Sydney)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在两人两动作博弈中将复制者动力学的线性收益替换为对数线性收益，揭示了对数贝叶斯更新与传统期望效用之间的差异，并给出完整的拓扑分类。

**💡 创新点**

首次证明对数线性收益会在所有具有纯支配策略的博弈中反转支配关系，构造了 Hamiltonian 结构，并将两种收益模型的动力学在二维空间中完整比较。

**🔧 技术方法**

使用复制者动力学、贝叶斯滤波解释、对数坐标变换、哈密顿量方法以及线性代数/凸分析技术。

**📊 数据集**

无实验数据，全部为理论推导和解析结果。

**📈 对比分析**

通过解析求解流形、计算守恒量和轨道结构，比较了两种模型在顶点保持映射下的拓扑等价性，结果显示约 70% 的 2×2 博弈在两模型下拓扑不等价。

**⚠️ 局限性**

局限在于仅针对两动作情形，假设无限精度与无噪声的理想贝叶斯更新，且无法直接映射到实际学习算法或有限数据情境。

---

## 162. Executable Archaeology of Programs That Model Programming: Reanimating the Heuristic Compiler (1961) and KBEmacs (1984)

**arXiv ID:** 2610.36341 | [PDF](https://arxiv.org/pdf/2610.36341v1)

**作者:** Jeff Shrager `[一作]`, Mia Golec `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Herbert Simon 的 1961 年 Heuristic Compiler 和 MIT 1984 年 KBEmacs 两个历史 AI 编程系统进行可执行考古，完整复原并重现其原始行为，展示它们的编程理论与现代 AI 辅助编程工具的概念连续性。

**💡 创新点**

1) 采用可执行考古方法，将历史程序“活化”，验证其理论的可执行性；2) 对比两种截然不同的编程模型（问题求解与 cliche 组合），揭示它们在现代 AI 工具中的共性；3) 利用现代 AI 编码助手（Claude Code）完成代码转录、修正与调试，展示 AI 在复原历史系统中的可行性。

**🔧 技术方法**

可执行考古、原始 IPL‑V 与 Zetalisp 代码的解释器、Common Lisp + Zetalisp 兼容层、AI 编码助手（Claude Code、ChatGPT）进行转录、修正和调试。

**📊 数据集**

原始纸质列表：Herbert Simon 的 1961/1963 版 Heuristic Compiler 列表；KBEmacs 1985 年版四卷打印输出（包含自测序列）。以及自行整理的 GitHub 仓库代码。

**📈 对比分析**

通过复现 Simon 论文中的 INSERT 与 J3 示例以及 KBEmacs 的 SQRT 自测序列，比较生成代码与历史输出的相符性。评估指标主要为功能一致性；未对执行速度或资源消耗进行量化比较。

**⚠️ 局限性**

仅复现核心逻辑，未完整复原英语前端、图形编辑器等交互功能；KBEmacs 仅在 headless 模式下运行；两系统的知识库规模极小，无法代表现代 AI 的海量隐式知识；重建过程高度依赖 AI 助手，复现成本和可重复性受限；对现代 AI 编程助手的比较仅停留在概念层面。

---

## 163. CrowdCC: Jointly Benchmarking Corruption Robustness and Deployment Quantization in Crowd Counting

**arXiv ID:** 2610.36340 | [PDF](https://arxiv.org/pdf/2610.36340v1)

**作者:** S Mahmudul Hasan `[一作]` (Tulane University), Lu Peng `[通讯]` (Tulane University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CrowdCC，一个面向人群计数的综合评测基准，统一评估输入破坏、模型量化（FP32/FP16/INT8）以及在真实硬件（Jetson Orin NX）上的吞吐量和能耗。

**💡 创新点**

创新点：①将 23 种部署相关破坏与 3 个严重程度统一调校，支持注释在几何/可见性变化下自动更新；②在同一实验框架下同时量化模型的鲁棒性、精度和部署成本；③揭示 INT8 对鲁棒性和效率的非统一影响，说明单独评测不足。

**🔧 技术方法**

技术：TensorRT 推理引擎（A100 与 Jetson Orin NX），NVIDIA ModelOpt 进行后训练 INT8 校准，定制破坏生成与严重度校准，使用 MAE、RD、mCE 等指标，测量吞吐量与能耗。

**📊 数据集**

数据集：ShanghaiTech A、ShanghaiTech B、UCF-QNRF、JHU-Crowd++。

**📈 对比分析**

比较方法：对每个模型-数据集-精度组合计算清晰度 MAE、破坏后的 MAE、相对降解 RD、与 CSRNet 的 mCE，并结合 Jetson 上的吞吐量与能耗；结果显示：捕获链破坏对误差影响最大；INT8 的鲁棒性提升/下降无系统规律；不同模型的能源-鲁棒性前沿揭示了需要联合考虑的折衷。

**⚠️ 局限性**

局限性：合成破坏与真实自然破坏的迁移性未知；每种破坏单独测试，未评估破坏组合；实验仅在 A100 与 Jetson Orin NX 上进行，跨平台可迁移性需进一步验证；缺乏因果性分析，仅给出关联描述。

---

## 164. BoxWorld: Let World Engines Control and Video Generators Render

**arXiv ID:** 2610.36371 | [PDF](https://arxiv.org/pdf/2610.36371v1)

**作者:** Zike Wu `[一作]` (University of British Columbia), Renjie Liao `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了BoxWorld框架，将交互推理与视频渲染分离，用户通过编写规则生成Box World Engine，生成器根据对象状态流实时渲染视频。

**💡 创新点**

创新点在于：①引入显式且轻量的对象中心状态接口（布局框、持久属性、动态状态），使得交互规则可独立编码；②采用三阶段训练（Masked Layout Grounding、State Dynamics Adaptation、Rollout Distillation）让生成器仅负责渲染，同时能实时响应状态更新；③无需配套交互数据，通过现成的感知模型自动从视频中提取状态标注。

**🔧 技术方法**

技术包括：Diffusion Forcing、Teacher Forcing、Self‑Forcing分布匹配蒸馏、LoRA适配、Masked Cross‑Attention实现状态对视觉的局部注入、实时异步渲染；以及使用LLaVA‑NeXT‑Video、Grounding DINO、SAM2进行状态提取。

**📊 数据集**

数据集：开域视频 VideoGen‑1M 与 4DNeX（用于布局和状态预训练），游戏玩法短片（约4k条）用于状态动态适配；自动注释生成的对象状态序列；公开基准 YouTube‑VIS、VBench、VBench‑Long 以及自制的 PairedStateBench。

**📈 对比分析**

与端到端世界模型 LingBot‑v1/v2、LongLive 以及公开基准进行对比。BoxWorld 在 PairedStateBench 的 Trigger、Suppress、Paired‑CF、Task 指标分别达到 92.2%、90.5%、97.3% 与 83.9%，远超基线；在 VBench 与 VBench‑Long 的语义质量（Semantic）与 FVD 上与 LongLive 对齐，且在 YouTube‑VIS 的 FVD、rCLIP、rCFC 指标中表现最佳；实时帧率约 20.7 FPS。

**⚠️ 局限性**

局限性：需要手工编写交互规则或使用编程代理；状态接口仍以二维框为基础，细粒度空间信息受限；在极端动态或非标准对象场景下的定位和属性识别仍可能出错；以及训练过程中需大量视频与自动注释，导致规模化成本较高。

---

## 165. ORBITAL: Optimization of Robust Brain--Computer Inference Through Architecture-Aware Low-Precision Compression

**arXiv ID:** 2610.36411 | [PDF](https://arxiv.org/pdf/2610.36411v1)

**作者:** Saim Rehman `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ORBITAL框架，实现了对EEG网络的硬件可实现结构压缩、鲁棒性恢复和边缘硬件验证。

**💡 创新点**

首次将压缩、鲁棒恢复和硬件效率三者整合到统一的搜索与验证流程，发现TRADES在不同压缩程度下的恢复交互并揭示了鲁棒边界重新分布。

**🔧 技术方法**

采用结构化通道剪枝、TRADES对抗训练、量化感知训练（FP16/INT8）及TensorRT在Jetson Orin Nano上的原生推理测评。

**📊 数据集**

使用BCI Competition IV‑2a 9个受试者的22通道脑电运动想象数据集。

**📈 对比分析**

通过PGD20/PGD100/APGD/ FAB等多攻击评估，S25‑TRADES在压缩后实现了23.06%的真交集鲁棒率，比全宽模型高1.79个百分点；同时参数↓26.27%、MAC↓25.10%，Jetson上FP16推理能量下降8%/16%，而INT8 Q/DQ则导致能量与延迟翻倍。

**⚠️ 局限性**

未验证更大规模模型或不同任务，INT8低精度在实际硬件中会因转换开销而失去优势，且鲁棒性提升对不同架构与压缩度依赖性不一致。

---

## 166. LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter Fleets

**arXiv ID:** 2610.36386 | [PDF](https://arxiv.org/pdf/2610.36386v1)

**作者:** Shenghui Li `[一作]` (Uppsala University), Thiemo voigt `[通讯]` (Uppsala University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Functional Rank Allocation (FRA) 方法，对 LoRA 适配器进行层级与跨适配器的秩分配，实现对 LoRA fleet 的后置压缩并最大化功能保留。

**💡 创新点**

将 LoRA fleet 压缩建模为两级资源分配，使用无标签功能前向测量加动态规划构造近似功能前沿，并结合谱候选做校准，最终实现精确的 minimax 分配。

**🔧 技术方法**

LoRA、低秩分解、谱能量阈值、Jensen–Shannon 散度衡量功能风险、无标签校准、端到端重测量、动态规划（多选背包问题）和 minimax 优化。

**📊 数据集**

三大公开 LoRA fleet：LoRA Land（12 个 Mistral‑7B 适配器）、Lots‑of‑LoRAs（25 个 Mistral‑7B‑Instruct 适配器）和 LoRARetriever（41 个 Llama‑2‑7B 适配器），以及 96,370 个 LoRA 模块的谱统计。

**📈 对比分析**

与 Uniform、Intra‑Spec、E2E‑Spec、FuncDP、Candidate Oracle 等策略对比，FRA 在中等到严重预算场景下显著提升尾部保留率（worst‑case 与 10th percentile）约 0.5–0.9，且校准成本相对低。

**⚠️ 局限性**

近似前沿构造依赖单层截断测量，跨层交互误差导致功能风险估计失真；谱候选与功能候选的形状依赖性未完全解决；计算成本随适配器规模线性增长，且缺乏对下游有标签评估的充分保障。

---

## 167. On the Fundamental Limits of Bistatic ISAC Over Vector Channels

**arXiv ID:** 2610.36351 | [PDF](https://arxiv.org/pdf/2610.36351v1)

**作者:** Dongsheng Peng `[一作]`, Jun Chen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了双极ISAC系统在向量信道下的基本极限，给出了不同退化方向的平行信道、一般向量信道的容量-失真函数的上下界，并在高斯状态下证明了高斯叠加编码的最优性，得到精确的容量-失真功率函数。

**💡 创新点**

创新点在于：① 对退化方向可变的平行信道给出单字母表述并说明联合编码优于独立编码；② 对无全局退化顺序的向量信道给出上下界，并在高斯情况下利用Fisher信息、极限不等式和信道增强实现了完全匹配；③ 通过水分配结构求解功率-失真分配问题，揭示了联合功率和失真分配的双重水分配性质。

**🔧 技术方法**

主要技术包括：超级叠加编码、信道退化/随机退化的引入、Fano不等式与Csiszár求和同义、Fisher信息矩阵与矩阵极大/极小不等式、信道增强（channel enhancement）技术、拉格朗日双对偶分解、最优功率分配与水分配解。

**📊 数据集**

无实验数据集，本文完全为信息理论的解析与证明。

**📈 对比分析**

性能评估以理论极限为准。论文通过与已知极限（如单极ISAC、退化广播信道等）对比，展示了联合编码和功率-失真分配在不同退化结构下的显著优势，并给出数值示例（如2×2高斯信道）验证了理论解的可行性与最优性。

**⚠️ 局限性**

局限性包括：仅考虑加性高斯状态和高斯噪声；在非高斯或有限元状态下仍需进一步研究；对复杂多用户、时变信道的扩展尚未给出完整解析；实际系统中硬件限制、延迟与实现复杂度未纳入分析。

---

## 168. Thermodynamic characterization of message-passing: Framework and initial study

**arXiv ID:** 2610.36348 | [PDF](https://arxiv.org/pdf/2610.36348v1)

**作者:** Kalana G. Abeywardena `[一作]` (University of Toronto), Stark C. Draper `[通讯]` (University of Toronto)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了消息传递译码器的热力学效率，使用不匹配成本框架量化熵产生，并给出 [7,4] Hamming 码在二元擦除信道上的数值示例。

**💡 创新点**

首次将不匹配成本理论应用于消息传递解码器，揭示算法动态、输入统计和硬件实现共同决定热力学成本，并展示通过调整先验分布可降低能耗。

**🔧 技术方法**

采用随机热力学、概率论、状态空间建模、周期机框架，构造可观测状态转移矩阵，并计算熵产生的下界。

**📊 数据集**

使用 [7,4] Hamming 码与二元擦除信道（ϵ∈[0,1]）的理论信道模型作为实验数据。

**📈 对比分析**

通过数值模拟比较均匀与非均匀先验、可纠正与不可纠正擦除模式下的总不匹配成本，结果显示能耗可降低至约 0.2 aJ 以内，非均匀先验可进一步提升至约 0.3 aJ。

**⚠️ 局限性**

状态空间指数增长限制了对大型 LDPC 码的直接计算，记忆子系统未建模，仅在简单的擦除信道上评估，未覆盖噪声和有限精度实现。

---

## 169. Your Wish Is My Commit: Over-Compliant Refactoring in AI Coding Agents

**arXiv ID:** 2610.36346 | [PDF](https://arxiv.org/pdf/2610.36346v1)

**作者:** Haocong Jiang `[一作]` (University of Southern California), Souti Chattopadhyay `[通讯]` (University of Southern California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在编码代理中的“过度顺从”现象，并通过持续的重构请求检测其无限编辑行为。

**💡 创新点**

创新点在于首次量化并归纳了六种可测量的过度顺从信号（S1–S6），为评估和缓解编码代理无意义编辑提供了可操作的指标。

**🔧 技术方法**

研究采用了代码差异分析、RefDiff 结构树对比、关键词分类器和多轮交互实验等技术手段。

**📊 数据集**

使用了 50 个算法题的 Java 解决方案和 10 个面向对象 Java 作业代码，覆盖两大代码规模族群。

**📈 对比分析**

通过在 10 轮交互中记录每个模型的停顿点、代码行变化量、结构性增删比等指标，发现所有模型均存在持续编辑，GPT 系列停顿最慢、Claude 系列停顿较早但后续更易复编辑；整体表现显示过度顺从是普遍且严重的现象。

**⚠️ 局限性**

局限性包括仅限 Java 语言、RefDiff 的方法级粒度、单次随机实验、模拟“随意重构”的极端用户情境，以及未验证后续版本是否保持功能正确性。

---

## 170. INC-MoE: Overlapping In-Network Collectives and Expert Computation for Expert-Parallel Mixture-of-Experts Inference

**arXiv ID:** 2610.36339 | [PDF](https://arxiv.org/pdf/2610.36339v1)

**作者:** John Stephan `[一作]` (Huawei), Guan Zixuan `[通讯]` (Huawei)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在规模化分布式推理中，将 Mixture‑of‑Experts（MoE）层的分发（dispatch）与合并（combine）交给网络交换机执行，并在单一持久化内核中将两种通信与专家计算重叠，以提升吞吐量和减少延迟。

**💡 创新点**

创新点包括：1）采用消息语义的 In‑Network AllGather 与 ReduceScatter，将传统的 ring 集体转化为交换机端的多播/归约；2）通过“band”流水线与专门化线程块，将两种通信与计算实现并行重叠；3）实现了可直接插入 vLLM 的零拷贝、单内核融合 MoE 层。

**🔧 技术方法**

主要技术：消息语义的 in‑network collectives、RoCEv2 网络、设备端一次性通信（device‑initiated）、GPU 持久化内核、角色专用线程块（gather、compute、reduce）、band‑pipeline 并行重叠、在单个内核中完成多阶段同步。

**📊 数据集**

使用 OLMoE‑1B‑7B（16 层、E=64、k=8 的 MoE 大模型）作为测试数据集，进行预填充（prefill）推理实验。

**📈 对比分析**

对比方法：将 INC‑MoE 与 vLLM 的默认 ring AllGather/ReduceScatter 路径相同配置（DP=4、EP=4、无张量并行）进行比较。实验结果显示，时间到首令（TTFT）降低 20.2–33.1%，吞吐量提升 25.3–49.5%，两者输出余弦相似度为 1.0，说明精度无损。

**⚠️ 局限性**

局限性：1）集成在实验用的交换机仿真器上，存在固定开销和对消息大小的依赖；2）仅支持静态 AllGather/ReduceScatter，无法处理动态路由或不规则专家分配；3）仅在 4 节点、4‑way EP 环境下验证，缺乏大规模扩展和真实硬件交换机的评估；4）仅对 FP16/BF16 Swiglu FFN 适用，未涉及其他模型或混合精度。

---

## 171. OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning

**arXiv ID:** 2610.36335 | [PDF](https://arxiv.org/pdf/2610.36335v1)

**作者:** Hui Yang `[一作]` (University of Manchester), Yongsheng Gao `[通讯]` (SNOMED International)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于本体的代理式RAG框架OA‑RAG，用于在医学问答中融合OWL本体的形式推理与LLM推理。

**💡 创新点**

创新点在于将三种推理模式（形式推理、基于本体的LLM推理、辅助LLM推理）统一到一个代理框架，并为答案提供四级“grounding”可信度评估。

**🔧 技术方法**

技术上采用ReAct循环调用SPARQL查询、前端预推理、最终答案与证据合成，以及对本体概念卡的文本化接口。

**📊 数据集**

使用的实验数据集包括从GO、FoodOn、SNOMED CT自动生成的MCQ/OQ，以及OntoTune基准的MedQA、USMLE、MedMCQA、PubMedQA等医学问答数据。

**📈 对比分析**

与基线（基础LLM、OntoTune fine‑tune、RAG、ReAct、MedRAG、i‑MedRAG）比较，OA‑RAG在Test1/2的准确率分别达到62%/68%，在Test3与i‑MedRAG相当但仅使用10×更少的token。

**⚠️ 局限性**

局限性主要是检索瓶颈（尤其是OQ缺少候选答案）、对本体完整性的依赖以及查询回退可能引入噪声。

---

## 172. MUSE: Revealing Responsiveness Gaps of LLM Tutors on Multiple-Solution Programming Problems

**arXiv ID:** 2610.36367 | [PDF](https://arxiv.org/pdf/2610.36367v1)

**作者:** Sky Cen `[一作]` (University of Central Florida), Shashank Sonkar `[通讯]` (University of Central Florida)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多解编程问题的模拟环境，用来评估LLM导师对学生多种解法的偏好、响应性和过渡性，并对GPT‑OSS、Qwen、Llama、Gemma等模型进行系统评估。

**💡 创新点**

首次将学生选择的解法作为对话核心，引入解决方案偏好、响应率和过渡得分等指标，并证明有针对性的SFT可显著提升导师的响应性与向更优解迁移的能力。

**🔧 技术方法**

利用大语言模型生成对话、GPT‑OSS 120B 作为判定器、合成响应数据并进行监督微调（SFT），同时在推理时将可选解法空间作为上下文信息。

**📊 数据集**

使用NeetCode 150编程题集（150道题目，每题提供三种有效解法）以及GPT‑OSS生成的正确与错误直觉作为实验数据。

**📈 对比分析**

通过对四款LLM在不同开场方式和直觉正确性下的解决方案支持率（SSR）与后解过渡得分（PTS）进行对比，SFT后SSR提升约9–18%，PTS从0.34–0.62提升至0.57–0.90，基线模型往往偏好粗暴解且易转向次优解。

**⚠️ 局限性**

实验仅探索了有限的微调配置，且评估数据集仅包含150道面试式编程题，未覆盖更广泛的 STEM 多解问题。

---

## 173. When the Application Layer Is Disposable: Putting Integrity Constraints Back in the Database

**arXiv ID:** 2610.36418 | [PDF](https://arxiv.org/pdf/2610.36418v1)

**作者:** Hasara Kalumin `[一作]` (University of Maryland at College Park), Amol Deshpande `[通讯]` (University of Maryland at College Park)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了约束索引（constraint indexes）这一新型事务级辅助结构，用于在 PostgreSQL 中高效强制多表约束，解决传统触发器、阴影表等方法在 LLM 驱动写入和概念模型映射中的缺陷。

**💡 创新点**

创新点在于将约束编译为事务级访问路径，利用共享内存索引与逻辑锁实现事务同步和原子性，支持分布式继承、折叠关系、弱实体以及基数和功能依赖等多表约束，且在不需显式重写 SQL 的前提下实现了可恢复的、可扩展的约束治理框架。

**🔧 技术方法**

核心技术包括：PostgreSQL 扩展开发、共享内存哈希索引、事务级自定义锁（advisory lock）、统一的事务回滚日志、约束索引的编译与重建算法，以及基于逻辑键的冲突序列化。

**📊 数据集**

使用的实验数据集为一个包含约 570 万条实体实例、数百万折叠弱实体与关系值的模拟现实业务模式，涵盖分布式继承层次、折叠关系、弱实体、基数约束、功能依赖等。

**📈 对比分析**

与查询校验、触发器校验、阴影表（Shadow）及其触发器等基线对比，约束索引在所有关键工作负载下平均提升 1.09–2.41 倍吞吐量，并保持 81–98% 的无约束吞吐量，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：实现仍为实验性 PostgreSQL 扩展，未深度集成到数据库引擎；支持的约束类型有限，尚未覆盖所有通用断言和复杂聚合约束；缺乏自动化编译器和多约束共享结构的支持；在高并发下仍可能存在锁竞争和重建成本。

---

## 174. Verifiable Efficiency Limits of Innate LLM Planning in Sliding Puzzles

**arXiv ID:** 2610.36415 | [PDF](https://arxiv.org/pdf/2610.36415v1)

**作者:** Yezhou Cheng `[一作]` (Uber Inc), Charles Ling `[通讯]` (Western University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型在没有工具或反馈的情况下完成滑动拼图的完整规划能力，并提出可验证的SPE框架将成功率、协调性与行动经济性分解。

**💡 创新点**

提出SPE=R×Q×E的三层可验证框架，既量化整体规划效率，又能区分执行错误、目标协调失败和行动浪费。

**🔧 技术方法**

使用完整计划回放验证执行可靠性、规划协调性；通过已认证的最优解证书评估行动经济性；统计成功率、可靠率、协调率与平均超量移动。

**📊 数据集**

基于4×4滑动拼图构造的主实验集（320个板，三种障碍级别，10-40步）和压力集（80个板，50-60步），共计8,400条完整计划。

**📈 对比分析**

在七个模型条件（Astra、Fable 5、Opus 5.5、GPT‑5.6 Sol、Claude Sonnet 5.5、GPT‑5.6 Luna、Claude Haiku 4.5）上比较，Astra/Fable/Opus在所有试验中完成率为100%，但随着步长增加行动经济性下降；Sol/Sonnet的协调率下降更快；Luna/Haiku虽然行动经济性更好，却因协调不足导致整体SPE最低。

**⚠️ 局限性**

仅关注动作空间的评估忽略了代币消耗、延迟和成本；缺乏更细粒度的轨迹诊断；实验覆盖模型有限，需扩展到更多模型和游戏。

---

## 175. Cross-Day Compatibility of Daily Item Embeddings: An Empirical Study of Alignment, Scale Management, and CTR Adaptation

**arXiv ID:** 2610.36408 | [PDF](https://arxiv.org/pdf/2610.36408v1)

**作者:** Yicheng Wang `[一作]` `[通讯]`, Yicheng Wang

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过系统对比词向量和图神经网络两类日常商品共现嵌入，研究了不同上游稳定化与下游CTR学习方法对跨日兼容性和点击率预测的影响，提出了一系列初始化、对齐、温启动、归一化与编码器自适应等技术方案。

**💡 创新点**

创新点在于：①将词向量与图嵌入在工业规模下同等比较；②通过正交对齐、ID初始化、温启动及全局尺度重置等多维度上游干预提升跨日几何一致性；③结合CTR学习的降维编码器进一步缓解残余漂移；④通过统一指标（余弦、L2、Recall@100、GAUC、LogLoss）构建完整的交叉评估框架。

**🔧 技术方法**

技术包括Skip-gram负采样Word2Vec、基于图的XSimGCL、正交对齐（OP）、线性/非线性映射、单位/全局尺度归一化、温启动训练、球面约束、对齐+残差噪声数据增强、CTR预测的深度网络与共享低维编码器。

**📊 数据集**

使用的是LY公司广告业务的商品行为日志和点击数据，商品数≥100万，涵盖多日（Day 1–9）日常更新嵌入与对应的曝光/点击记录。

**📈 对比分析**

通过48种配置（32种CTR实验）对比，结果显示：ID初始化+OP提升Cosine 51.2%和GAUC 2.2%；温启动+全局归一化可将正交余弦提高至127%且GAUC最高提升4%；CTR学习的编码器在四种上游方法中获得最高跨日余弦，表现优于随机投影、SVD、未训练网络。

**⚠️ 局限性**

局限性包括：仅在单一业务场景与单一时间窗口实验，未做多重比较校正；缺乏在线CTR提升验证；温启动与归一化等干预难以完全分离，可能混合影响；未对历史版本扩展或多源语义融合进行深入探索。

---

## 176. A 2-DOF MultiBody Track Joint for Race Car Optimization

**arXiv ID:** 2610.36407 | [PDF](https://arxiv.org/pdf/2610.36407v1)

**作者:** Srikanth Sivaramakrishnan `[一作]` (General Motors), Ryan Trinter `[通讯]` (General Motors)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于Modelica MultiBody的2自由度赛道关节组件，利用Monge补丁和六条C^4连续五次样条描述赛道表面，使赛车模型能够在同一组件下完成时域仿真与最小周时优化；

**💡 创新点**

创新点包括将一维轨道关节扩展为二维Monge表面关节，完全用表面偏导数构造右手系三轴并通过两条d'Alembert约束闭合DAE；将弧长速度直接作为状态输出实现时间到空间变换；采用Horner递推的样条评估以保证数值稳定；支持FMI 3.0导出并与CasADi/IPOPT直接协同；

**🔧 技术方法**

技术实现主要使用Modelica MultiBody与Dymola、FMI 3.0 FMU、CasADi直接内插法、IPOPT求解器、d'Alembert原理、Monge补丁几何、C^4连续五次样条、Horner求值、DAE求解器；

**📊 数据集**

使用23条F1赛道（包括新加坡、摩纳哥等）通过配套拟合流程得到的5 m均匀结点五次样条；验证时采用新加坡大奖赛赛道的LiDAR点云；

**📈 对比分析**

通过与Claytex单自由度轨道关节比较，评估Horner与展开多项式的计算成本与误差；在FMI 3.0 FMU中测量评估时间、FMU大小、IPOPT迭代次数与求解时间；结果显示展开多项式比Horner快约26%，IPOPT迭代数差异约4%，整体FMU耗时占比>70%，实现最小周时误差<4×10⁻⁵；

**⚠️ 局限性**

局限性包括：依赖外部拟合得到的样条且不在运行时检查参考线是否满足injectivity条件；仅支持封闭赛道、均匀结点，非均匀结点需改写查找；未实现多实例共享参数的四点接触补丁版本；验证范围仅为单条赛道单车模型；组件与数据不公开。

---

## 177. Adaptive Calibration of RF Digital Twins via Budget-Constrained Radio Measurement Selection

**arXiv ID:** 2610.36406 | [PDF](https://arxiv.org/pdf/2610.36406v1)

**作者:** Yalin E. Sagduyu `[一作]` (Nexcepta), Hossein Jafari `[通讯]` (Nexcepta)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于OSM、Blender与Sionna的物理模型RF数字孪生，并在有限测量预算下对其进行主动校准。

**💡 创新点**

提出了从无结构到自适应不确定性感知的五种在线测点选择策略，并证明自适应结构化方案显著提升校准精度。

**🔧 技术方法**

利用地理语义数据、3D几何建模、确定性射线追踪、软件定义无线电路径损耗测量以及在线增益估计。

**📊 数据集**

使用30个测量站点的USRP B210路径损耗数据，频率1.2 GHz，配合相同位置的DT仿真。

**📈 对比分析**

通过多次Monte‑Carlo实验对比随机采样、平衡秩采样、QMC秩采样、渐进结构感知采样与自适应不确定性采样，结果显示AUS与PSS在均值误差上分别降低约72%和26%，最大误差降低至约91%。

**⚠️ 局限性**

局限性包括仅采用全局标量偏移校准、样本量有限、缺乏对复杂环境非线性误差的建模以及对不同频段与传播情景的泛化能力待验证。

---

## 178. kw-bench: Breaking Frontier Harnesses on Real-World Tasks

**arXiv ID:** 2610.36405 | [PDF](https://arxiv.org/pdf/2610.36405v1)

**作者:** George Juraj Salapa `[一作]` `[通讯]` (G.S. s.r.o.), George Juraj Salapa (G.S. s.r.o.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了名为kw‑bench的专业知识工作基准，用真实的SAP三代系统导出数据模拟为期五年的增值税链条重建任务；

**💡 创新点**

创新点在于：①把隐蔽答案作为评测锚点，消除主观判定；②将评测完全自动化、无LLM评判；③针对长周期、复杂链路的调查性工作构建挑战；④揭示基准中对“harness”工程的关键性影响。

**🔧 技术方法**

使用终端‑文件系统agent harness（如Claude Code、Codex CLI、OpenCode、micro‑cc），利用LLM做代码生成、工具调用和状态管理；评测框架采用Harbor。

**📊 数据集**

数据集为合成的、基于真实税务案件设计的SAP ECC、S/4HANA 与定制链路板块的导出文件（约62 MB），并附带隐藏的真实增值税申报结果。

**📈 对比分析**

比较方法：在同一模型（DeepSeek v4.1 Flash）上跑四个不同harness，记录分数、时间、调用次数和成本。结果显示：独立harness（OpenCode、micro‑cc）可获得约0.53分，而实验室harness在相同模型下得分为0；而在前沿模型（Claude Opus 5.5）下，微小微调与Claude Code可在22‑25 分钟内完成任务，分数接近1。

**⚠️ 局限性**

局限性包括：仅单次运行评估；仅测试两类模型（低成本与前沿）；使用合成数据而非真实客户数据；自动化评测忽略了人类审阅的质性判断；仅评测无交互的“懒惰专家”场景；未给出多轮跑的统计与人类基线。

---

## 179. Reliability-Aware Short-Term Glycemic Risk Prediction Using an Anchor-Based Comparative Framework of Sequence, Multimodal, and State-Updating Digital-Twin Models

**arXiv ID:** 2610.36363 | [PDF](https://arxiv.org/pdf/2610.36363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 180. SpecLST: Spectral-Enhanced Long-Short Term Spatio-Temporal Multimodal Modeling for Traffic Forecasting

**arXiv ID:** 2610.36393 | [PDF](https://arxiv.org/pdf/2610.36393v1)

**作者:** Kenan Kang `[一作]` (Auckland University of Technology), Weihua Li `[通讯]` (Auckland University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现SpecLST框架，结合FFT低频频域表示两周历史交通与短期多模态融合，使用稀疏混合专家网络和高频残差头进行交通预测。

**💡 创新点**

创新点在于将长期周期性信息压缩为FFT低频系数并与短期多模态（天气、日历、事件）通过注意力融合，加入可学习稀疏邻接扩展以及高频残差头以提升对突发事件的感知。

**🔧 技术方法**

采用FFT频域转换、注意力机制、稀疏Mixture‑of‑Experts（MoE）网络、图卷积/注意力处理、残差高频头等技术。

**📊 数据集**

使用PeMS‑BAY、PeMS‑BAY22、PeMS04、PeMS08四个公开高速公路流量/速度数据集，并对PeMS‑BAY22补充天气、日历与事件信息。

**📈 对比分析**

与统计模型、图神经网络、Transformer等多种基线在MAE、RMSE、MAPE上对比，SpecLST在多数据集上实现显著性能提升，并在内存占用与训练时间上位于Pareto前沿。

**⚠️ 局限性**

局限在于仅使用静态可学习邻接，无法实时捕获瞬时空间关联；FFT方式仅捕捉周期性而非非周期性变化；事件数据仅覆盖湾区，缺乏跨地区泛化能力。

---

## 181. MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations

**arXiv ID:** 2610.36389 | [PDF](https://arxiv.org/pdf/2610.36389v1)

**作者:** Reza Yousefi Maragheh `[一作]` (Walmart Global Tech), Sushant Kumar `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种基于多代理LLM的评估框架MAREval，用于评估推荐系统自然语言解释的质量。

**💡 创新点**

创新点在于提出链式辩论（CoD）规划、多代理协同、调节员与仲裁员结构，并结合蒙特卡洛采样提升人类一致性与稳定性。

**🔧 技术方法**

使用大语言模型多代理对话、链式辩论提示、调节员与仲裁员机制以及蒙特卡洛采样等技术。

**📊 数据集**

采用了公开的TopicalChat对话数据集和自研的电商推荐解释数据集PE。

**📈 对比分析**

通过与Vanilla、CoT、Self-Refine、G-Eval、ChatEval等基线在相关性、措辞、自然性等指标上比较，MAREval在相关系数、Kendall‑τ、Kappa等指标上均显著领先，并在离线实验与线上A/B测试中提升ATC和GMV。

**⚠️ 局限性**

主要局限包括较高的推理延迟、对大规模LLM资源的依赖，以及在非推荐解释任务中的泛化能力尚待验证。

---

## 182. SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding

**arXiv ID:** 2610.36384 | [PDF](https://arxiv.org/pdf/2610.36384v1)

**作者:** Matea Tashkovska `[一作]` (École Polytechnique Fédérale de Lausanne), Alexander Mathis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了SOMA基准，评估第一人称视频中人体运动的细粒度理解，并提供训练数据

**💡 创新点**

设计了三轴任务（身体构型、运动动力学、时间推理），采用盲语言过滤与人类验证填补了现有多模态大模型在 egocentric 视频人体运动感知评估的空白，并推出训练集与多模态基线

**🔧 技术方法**

使用LLM生成问题与答案、盲语言过滤、手工验证，结合VQ‑VAE运动编码与视频+运动联合训练，评估了37种多模态大语言模型

**📊 数据集**

依托Nymeria与Ego‑Exo4D两大egocentric视频数据集，构造了1,632道多选问答并生成28,596条训练样本

**📈 对比分析**

在零样本设置下与人类比较，37模型平均精度为50.0%，最佳模型Qwen3.6‑27B 50.0%，人类平均68.5%，各任务差距显著，尤其在重复计数、双手协调等细粒度任务上

**⚠️ 局限性**

模型在手部偏好、计数低估、时间定位和运动感知等方面表现不佳，单一视觉信息不足，未能充分利用身体感知，训练集与评估集规模相对有限

---

## 183. Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs

**arXiv ID:** 2610.36353 | [PDF](https://arxiv.org/pdf/2610.36353v1)

**作者:** Tiejin Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于标签保留、低容量、语义盲目扰动的输入歧义检测框架，并实现白盒和黑盒两种检测器；

**💡 创新点**

创新点在于通过保持输入语义不变但略作扰动来“泄露”歧义信号，避免传统基于重述或输出不确定性的局限；

**🔧 技术方法**

使用梯度、DiffMask、Logit Lens等白盒重要性评估方法；黑盒采用词汇同义替换、词性限制规则、前缀/WH变换等语义盲目规则；扰动后对模型输出进行多采样并用EigV(Agree)等语义一致性度量；

**📊 数据集**

在四个英文问答基准上评估：ABG-CoQA、AmbigQA-w-Ctx（对话）、AQuA、VQ-FA（视觉问答）；

**📈 对比分析**

与直接提示、EigV(Agree/Disagree)、语义熵等输出侧不确定性基线相比，黑盒/白盒框架在所有模型和数据集上均取得最高AUROC（最高可达0.79），并在基于检测的重提问（UAR）策略下显著提升回答准确率；

**⚠️ 局限性**

局限性包括黑盒规则库仅覆盖英文，跨语言迁移需重构规则；白盒方法需要访问模型内部信息，限制了其在闭源API场景中的可用性。

---

## 184. A Local Stabilized Isogeometric Formulation with Discontinuous Pressure for Mesh-Conforming and Immersed Analysis of Nearly Incompressible Elasticity

**arXiv ID:** 2610.36349 | [PDF](https://arxiv.org/pdf/2610.36349v1)

**作者:** Ashkan Ali Madadi `[一作]` (University of Texas at Austin), Thomas J. R. Hughes `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种局部化的变分多尺度（VMS）稳化等离子体与离散压强的等离子体（IGA）混合求解框架，用于近似不可压缩弹性问题，并将其推广到浸入式（immersed）计算。

**💡 创新点**

核心创新包括：①利用VMS稳化解耦非恒定压强模式，使离散压强网格仅需按固定比例（两倍）粗化即可满足离散inf–sup条件；②在同一框架下实现压强的局部静态消元，保持离散系统只含位移未知；③结合延伸B‑spline、压强单元聚集、子分割积分和Nitsche边界处理，完成了不贴合几何的浸入式求解。

**🔧 技术方法**

使用的技术包括：等离子体分析（B‑spline / NURBS），变分多尺度（VMS）残差稳化，离散压强的局部消元，延伸B‑spline（处理切割单元），压强单元聚集（L² 投影），Nitsche弱边界条件，子分割（octree）积分。

**📊 数据集**

主要在两类经典基准上验证：Kirsch 圆环问题（光滑解）和 Cook’s 内存问题（奇异解）。

**📈 对比分析**

通过离散inf–sup常数、收敛率和压力/应力场比较，展示了与传统 Galerkin、等阶混合、Taylor–Hood 等方法相比，新方法在相同或更少的压强自由度下保持稳定、收敛性相同或更好，并在浸入式设置下成功保留了混合稳定性。

**⚠️ 局限性**

局限性包括：①对高阶多项式仍需将压强网格粗化两倍；②需要经验选择Nitsche罚参数；③目前仅针对线性不可压缩弹性，非线性或大变形问题尚未验证；④在复杂几何和大比例切割时的数值稳定性和计算成本需进一步研究。

---

## 185. Reasoning about the Genetic Code in Lean

**arXiv ID:** 2610.36343 | [PDF](https://arxiv.org/pdf/2610.36343v1)

**作者:** Colin T. Jones `[一作]` (University of Maryland Baltimore County), Tyler R. Josephson `[通讯]` (University of Maryland Baltimore County)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 DNA‑RNA‑蛋白质转录与翻译过程进行 Lean 4 形式化，并用机器证明验证其关键性质（冗余性、长度保持、可逆性等）。

**💡 创新点**

首次将 Lean 4 交互式定理证明器引入分子生物学的核心图景，展示了在生物信息学中实现完全可检验、无语法错误的模型，及其对语义错误的检测能力。

**🔧 技术方法**

Lean 4、Mathlib4、归纳类型、递归函数与 tactic 语言，用于构建模型并写证明；代码托管在 GitHub 上。

**📊 数据集**

未使用真实实验数据集，而是采用抽象的核苷酸序列示例（如 AAA、GAA 等）来演示和验证模型；数据来自公开代码仓库。

**📈 对比分析**

论文未进行传统实验对比或性能评测；主要通过 Lean 的类型检查和证明成功率来表明模型的正确性与鲁棒性。

**⚠️ 局限性**

局限性：模型过于简化（忽略剪接、终止密码子、基因组差异等）；存在语义误差风险；缺乏对真实生物大数据的验证；对复杂细胞过程的可扩展性尚未评估。

---

## 186. Plenty of Room in the Middle: Revealing Opportunities on a GPU Cluster via Fine-Grained Monitoring

**arXiv ID:** 2610.36329 | [PDF](https://arxiv.org/pdf/2610.36329v1)

**作者:** Andrew Sheinberg `[一作]` (Princeton University), Kai Li `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在89节点、319张GPU的科研集群上实现并部署了基于Nvidia DCGM和SQLite的轻量级监控守护进程，以100 ms细粒度采样收集了10天内7399个作业的GPU利用率、显存、SM活跃度、NVLink/PCIe流量等指标并进行分析。

**💡 创新点**

首次在整个集群级别持续采集SM级利用率等细粒度指标，发现GPU显存与SM利用率普遍低、作业时长与请求值相差近80%、多GPU作业表现出高度周期性的通信模式，从而揭示GPU多任务共享和预测调度的巨大潜力。

**🔧 技术方法**

使用Nvidia Data Center GPU Manager（DCGM）API进行GPU指标采样，采用C实现的轻量守护进程写入SQLite数据库，并通过SLURM接口抓取作业信息。

**📊 数据集**

实验数据来自普林斯顿大学89节点科研集群（包含20个高性能节点、69个标准节点），共收集了7399个作业、150名用户、12个系的工作负载，监控时段为2024年秋季10天。

**📈 对比分析**

与Philly、Helios、Moneo、MIT SuperCloud等基准（均采用秒级或更粗粒度采样）对比，细粒度采样能够揭示低SM利用率、显存闲置以及通信峰值；工具在单核上CPU占用低于1%，每日数据存储约1.5 GB，证明低开销和可扩展性。

**⚠️ 局限性**

受DCGM限制，采样频率最高为10 Hz；仅能覆盖最后10天的完整作业映射；多GPU/多节点作业样本有限；不同GPU型号和互连差异未单独分析；未能捕获微秒级核级波动。

---

## 187. How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation

**arXiv ID:** 2610.36328 | [PDF](https://arxiv.org/pdf/2610.36328v1)

**作者:** Aravind Mandiga `[一作]` (University of Georgia), Guoming Li `[通讯]` (University of Georgia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在程序重构后，Lean 证明是否仍能通过验证，提出 RefactorProof 评测框架

**💡 创新点**

首次将行为保持的重构与证明稳健性结合，量化证明在不同重构下的生存率（PSR），并区分证明策略对重构的敏感性

**🔧 技术方法**

使用 Lean 4、Mathlib、自动化等价证明搜索、固定样本生成、Bootstrap 置信区间、统计显著性检验等技术

**📊 数据集**

基于 VERINA 基础分裂任务（46 个）与全量 189 个任务的实现、规范和证明文件

**📈 对比分析**

比较不同证明来源（VERINA 参考证明 vs DeepSeek-Prover-V2-7B、Goedel-Prover-V2-32B/8B）的 PSR；在匹配任务上，专用证明器的整体 PSR 均比参考证明高 15.6–18.8 百分点，95% CI 均不包含零；在结构类重构（如辅助函数提取、条件翻转）下的差异尤为显著

**⚠️ 局限性**

局限包括：仅测试 5 种局部、机械重构，未覆盖算法变更；参考证明只来自 46 个简单任务；证明生成受采样与 first‑valid 策略影响；统计功效不足、重构类别不完全；缺乏对完整代码生成与可执行性要求的评估

---

## 188. HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning

**arXiv ID:** 2610.36374 | [PDF](https://arxiv.org/pdf/2610.36374v1)

**作者:** Fanqi Yu `[一作]` (Istituto Italiano di Tecnologia), Vittorio Murino `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为HIDRA的模块化路由框架，用于在无回放、单次通行的终身模仿学习中从多模态演示流中持续适应。

**💡 创新点**

创新点在于将路由拆分为指令检索和上下文细化的双层注意力机制，并通过关键级几何正则化保持任务内紧凑、任务间分离，显著缓解了路由干扰。

**🔧 技术方法**

采用预训练的多模态视觉‑语言‑动作骨干、LoRA轻量级专家库、层级注意力路由、键空间正则化（内任务对齐与跨任务正交）以及标准行为克隆损失。

**📊 数据集**

在LIBERO系列基准（OBJECT、GOAL、SPATIAL、10 以及更具挑战性的混合序列 LIBERO‑MIX、LIBERO‑INS）上进行评估。

**📈 对比分析**

与回放式方法（ER、MLR‑IFA、DMPEL）以及无回放方法（L2M、EWC、ISCIL、TAIL）对比，HIDRA在所有基准上实现了最高AUC、最小遗忘（NBT≈0）并显著提升前向迁移，尤其在高干扰的混合序列中表现尤为突出。

**⚠️ 局限性**

局限性主要体现在对细粒度空间差异的建模不足（LIBERO‑SPATIAL表现相对较弱），对真实机器人平台的验证不足，以及对长时间连续演示流的可扩展性和计算开销的进一步研究仍待开展。

---

## 189. What Victims Take Away: Attack Reuse and Defense Revision after Prompt-Injection Attacks

**arXiv ID:** 2610.36422 | [PDF](https://arxiv.org/pdf/2610.36422v1)

**作者:** Zihan Chen `[一作]`, Di Zhu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析Tensor Trust这款在线游戏的攻击记录，研究受害者在看到攻击历史后是否会将成功或失败的攻击文本复制并用于攻击他人，并检验在一次攻击突破后受害者是否会更换防御设置以及随后同一攻击家族对其的成功率是否下降。

**💡 创新点**

首次在事件层面提供了受害者-攻击者学习路径的实证证据，显示攻击文本的再次使用主要跟随游戏评定的成功与否，而非仅仅依据攻击者回复的表面含义；进一步揭示突破事件会触发防御更改并显著降低同一攻击家族对突破账户的成功率；同时发现攻击文本的再次使用主要集中在已活跃的攻击者而非新手。

**🔧 技术方法**

采用事件级线性概率模型与固定效应（目标、文本、受攻击者防御、攻击者）结合，控制攻击文本流行度；使用文本归一化+5-gram Jaccard相似度与MinHash聚类生成攻击家族；对防御变化后成功率的差异进行差分-差分（DiD）估计；通过多种稳健性检验（窗口长度、文本长度、去除相互频繁攻击对等）验证结果。

**📊 数据集**

Tensor Trust 2023-2024年游戏原始数据库，包含563,349条攻击记录（415,644 对他人，其他为自测/沙盒），5,329个账户，12,531次突破事件（3,316个受害者），以及119,000多条防御保存记录，其中53,974条被验证部署。数据还被整理成271,378种独立攻击文本和121,226个攻击家族。

**📈 对比分析**

比较方法：对每一次“暴露事件”（目标首次收到某攻击文本）抽取最多5个在同一时间段内发送给同类目标但未被目标接收的控制文本，构成配对比较。再对不同结果（成功、泄露、近未命中、失败）的暴露事件进行文本固定效应比较，并将成功与失败的差异量化为百分点。对突破事件，用DiD比较突破前后同家族攻击在受害者与其他防御者上的成功率差异。性能：成功的暴露事件比控制文本高约8–10个百分点的再次使用率；近未命中几乎无差异；突破后防御更改率提升约20–30%；同家族攻击在更改后对受害者的成功率下降约10–20个百分点，显著优于其他防御者。

**⚠️ 局限性**

局限性：研究对象为自愿玩游戏的参与者，虚拟金钱和有限的激励可能与真实系统中的风险和动机不同；攻击历史的可见性未被直接记录，只用“在攻击到达前是否有活动”作为代理，可能未完全捕捉阅读行为；样本选择和文本聚类方法（阈值0.6）可能影响家族定义；未进行随机实验，仅做观察性分析；缺乏对学习过程本身的直接测量，文本家族不一定对应真正的技术传承。

---

## 190. SpeeDiff: Scalable Pixel-Anchored End-to-End Latent Diffusion Model

**arXiv ID:** 2610.36419 | [PDF](https://arxiv.org/pdf/2610.36419v1)

**作者:** Bingliang Zhang `[一作]` (ByteDance Seed), Qiushan Guo `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可从零开始端到端联合训练VAE和扩散模型的Latent Diffusion框架SpeeDiff，利用Tweedie Pixel Reconstruction loss防止潜在空间坍塌；

**💡 创新点**

创新点在于：①将扩散梯度直接传递给VAE encoder；②使用全Transformer VAE和改进的DiT实现无stop‑gradient的端到端训练；③提出REPA++对潜在空间与中间特征进行表示对齐；④大幅提升训练速度与可扩展性；

**🔧 技术方法**

采用Tweedie Pixel Reconstruction loss、Vision‑Transformer VAE、LighteningDiT（RMSNorm+SwiGLU+RoPE）、REPA++对齐、SDE解算器、AutoGuidance、EMA、LPIPS等技术；

**📊 数据集**

在ImageNet 256×256与512×512数据集上进行实验；

**📈 对比分析**

与传统两阶段LDM、SiT、REPA、EDM2等方法对比，SpeeDiff‑XL在ImageNet 256×256无指导下FID仅1.50，比Vanilla SiT快140×、REPA快61×；在512×512下FID 1.53，且在无指导条件下已实现高质量生成；

**⚠️ 局限性**

局限性包括：仍需依赖预训练VFM做表示对齐；高分辨率生成受限于32×压缩VAE；未在多模态或文本条件下验证；极端超分辨率或长文本指令的效果未知。

---

## 191. AK_QMUL at ArGuard Shared Tasks: A Single Fine-Tuned JAIS for Arabic Harmful-Prompt Detection

**arXiv ID:** 2610.36360 | [PDF](https://arxiv.org/pdf/2610.36360v1)

**作者:** Ali Khairallah `[一作]` (Queen Mary University of London), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

参与 ArGuard‑2026 共享任务，提出两套系统：B1 的安全/不安全二分类和 B2 的七类伤害域分类，均使用同一 JAIS‑6.7B 语义解码器并在其最后 token 处接线性分类头。

**💡 创新点**

创新点包括：B1 用四类（安全/不安全 × MSA/非 MSA）标签拆分并结合 AraSafe 公开安全 MSA 样本进行增强；B2 通过五个随机种子训练同一模型并平均 softmax 输出，提升鲁棒性；两任务共享相同 backbone、训练细节与量化策略，形成统一高效的解决方案。

**🔧 技术方法**

技术手段：4‑bit QLoRA 量化 + NF4 双量化、LoRA 适配器、AdamW + 线性学习率调度、类权重交叉熵、最后 token 池化、softmax 平均、argmax 预测。

**📊 数据集**

使用 ArGuard‑2026 提供的训练/开发/测试集（包含 MSA 及五大方言）以及 AraSafe 的安全 MSA 语料做训练增强。

**📈 对比分析**

评估方式为宏 F1（macro‑F1），B1 取得 0.984，准确率在六种方言间均高于 93.6%；B2 取得 0.790，单类 F1 多在 0.73‑0.98 之间，仅在 MSA 的 Bully 类表现最差。两轨均排名第一，优于公开基线。

**⚠️ 局限性**

局限性：仅评估单一 JAIS‑6.7B 语义解码器，未与其它编码器或架构做对比；B1 的 ablation 仅在单个 seed 上验证；Bully 类在 MSA 的混淆问题仍未解决，需更细粒度标注与模型改进。

---

## 192. CRLoMTL: Conflict-Resolution through Low-rank matrices for Multi-Task Learning

**arXiv ID:** 2610.36337 | [PDF](https://arxiv.org/pdf/2610.36337v1)

**作者:** Ahmed Agiza `[一作]` (Brown University), Sherief Reda `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 CRLoMTL 的多任务学习框架，通过在参数空间使用低秩 LoRA 更新并在训练期间交替进行任务专属更新和全局共享子空间融合，最终得到单一融合后的模型。

**💡 创新点**

创新点在于：① 采用多通道多通行（multi‑pass）训练策略，将任务专属 LoRA 更新与全局 SVD 基底融合交替进行；② 通过 SVD 共享子空间提取任务间共同方向，避免显式梯度冲突处理；③ 训练完成后将融合更新直接折叠回主干网络，无需推理时额外分支，兼顾参数效率与推理速度。

**🔧 技术方法**

主要技术包括低秩 LoRA 参数化、SVD 共享子空间解析器、任务专属与全局融合的多通行优化、以及参数高效微调（PEFT）框架。

**📊 数据集**

实验使用 Swin Transformer 作为主干，在 PASCAL‑Context、NYUD‑v2（语义分割+深度）以及 CelebA（40 个属性分类）三个多任务基准上进行评估。

**📈 对比分析**

与多种现有 MTL 方法（MTI‑Net、MTLoRA、DWA、PCGrad、CAGrad、Nash‑MTL、MGDA 等）进行对比，CRLoMTL 在 PASCAL‑Tiny 上提升 2.77% 的平均任务性能，仅训练 4% 的参数；在 PASCAL‑Small 上提升 6.08% 并训练 <6% 参数；在 NYUD‑v2 上提升 5.03%；在 CelebA 上提升 3.50%，同时保持与单任务模型相近或更优的推理速度。

**⚠️ 局限性**

局限性包括：① 需要多通行训练，导致训练时间较传统 MTL 稍长；② SVD 基底提取在极大任务数时可能成为瓶颈；③ 目前仅在视觉任务上验证，尚未探讨语言或跨模态任务；④ 对超参数（LoRA rank、α、共享子空间维度）的敏感度虽低，但仍需手动调优。

---

## 193. Training-Free Inference Acceleration for Video Large Language Models: Methods, Trade-offs, and Evaluation

**arXiv ID:** 2610.36327 | [PDF](https://arxiv.org/pdf/2610.36327v1)

**作者:** Yuqi Li `[一作]` (Wyze Labs), Yingli Tian `[通讯]` (City University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对无训练（training-free）的视频大型语言模型（VLM）推理加速技术进行了系统梳理与评估，提出按推理流水线阶段（输入、编码、投影、prefill、解码）分类的统一体系，并引入 Quality–Latency Balance (QLB) 指标衡量精度-延迟权衡；

**💡 创新点**

创新点包括：①从输入到解码全链路的阶段性分类法，揭示干预位置对证据保留与计算节省的双重影响；②设计 QLB 量化指标，兼顾多任务多模型的跨实验可比性；③在统一实验平台上对多种方法（如 EarlyTom、Curv. merge、VidCom^2 等）进行对比实验，并对 PTQ 方法的精度保留进行系统评测；④系统讨论本地加速与端到端加速之间的差距及其原因；

**🔧 技术方法**

使用技术主要包括：视频编码器与投影层的特征压缩、Token 选择与聚合、KV 缓存选择与聚合、Speculative Decoding、Post‑Training Quantization（权重、激活、KV）等；评估工具：QLB 计算、TTFT/TPOT 计时、Accuracy / 视频描述分数；实验平台：LLaVA‑OneVision‑7B（32帧）+ NVIDIA H100；

**📊 数据集**

数据集涵盖 Video‑MME、EgoSchema、LongVideoBench、MVBench、VideoDC、Qwen2.5‑VL‑3B、Qwen3‑VL‑8B 等多任务视觉‑语言基准；

**📈 对比分析**

比较方法：将不同阶段压缩方法在统一输入帧数和模型下的 Accuracy、TTFT 进行直接对比，并用 QLB 归一化评估；实验结果表明：EarlyTom 在 45.7% TTFT 降低的同时保持 97.7% 任务精度（QLB 126.6），Curv. merge、VidCom^2 等在后编码阶段也能保持 99% 以上精度，但 TTFT 降低幅度较小；PTQ 方法在 8‑bit/16‑bit 下可保留 90%+ 的精度；但局部节省往往未能完全转化为端到端加速，端到端加速率仅 10–15%。

**⚠️ 局限性**

局限性：实验仅基于单一 backbone（LLaVA‑OneVision‑7B）和固定 32 帧输入，未覆盖不同模型和帧预算的泛化；部分比较依赖已公布结果，实验协议不完全一致；PTQ 评估多基于图像任务，缺乏视频推理的完整延迟与内存测量；整体评测受硬件平台（H100）限制，可能不适用于低端设备。

---

## 194. PRISM: Pathology Reliability In Scarce-label Medicine. A Tile-level Benchmark of Pathology Foundation Models

**arXiv ID:** 2610.36517 | [PDF](https://arxiv.org/pdf/2610.36517v1)

**作者:** Alicem Koyun `[一作]` `[通讯]` (Westcliff University), Alicem Koyun (Westcliff University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

PRISM构建了一个面向病理影像基础模型的全方位可靠性基准，评估模型选择、低标记比例、校准与跨机构迁移四个维度的性能。

**💡 创新点**

提出Composite Reliability Index（CRI）将判别、校准和OOD稳定性统一衡量，并揭示低标记下判别与校准解耦、单类崩塌阈值、逆向校准衰减与迁移质量关联等现象。

**🔧 技术方法**

采用冻结的特征嵌入+L2正则化逻辑回归探针、温度缩放后的ECE评估，以及多标签比例、随机种子与Camelyon17五医院迁移实验。

**📊 数据集**

使用六个公开tile级数据集（PCam、CRC、MHIST、BRACS、SPIDER-Breast、LungHist700）和Camelyon17的五医院数据。

**📈 对比分析**

在864个内分布和3600个迁移实验中，对8个基础模型进行评估，发现完整监督下AUROC排名与校准排名相关，而低标记下失相关；温度缩放可提升校准但不总是有效；迁移时标签量增加往往导致校准恶化，并在迁移AUROC约0.85处出现转折点；CRI在不同标签比例下对模型进行排序。

**⚠️ 局限性**

仅评估tile级、冻结嵌入与固定探针正则化，未覆盖幻灯片级聚合、其他探针架构或真实临床数据分布；CRI是实验协议下的筛选指标，不能直接等同临床可靠性。

---

## 195. Upper and Lower Bounds on the Complexity of Unit-Distance Graph Realization in $\mathbb{Z}^d$

**arXiv ID:** 2610.36516 | [PDF](https://arxiv.org/pdf/2610.36516v1)

**作者:** Eric Binnendyk `[一作]` `[通讯]`, Eric Binnendyk

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文研究整数格子ℤ^d中的单距图识别问题，给出了在ETH假设下的下界与基于轻平衡割的分治算法上界，并提出了一种通用的递归实现；

**💡 创新点**

创新点在于证明任意ℤ^d格子嵌入图必存在的轻平衡割，从而设计出子指数级的分治算法，同时在d=2、3维给出了与上界相差对数因子的紧密下界；

**🔧 技术方法**

主要技术包括轻平衡割的组合几何证明、分治递归与边界框拆分、稀疏化引理与NAkSAT归约、逻辑引擎构造、图棱柱构造以及对分治递归的复杂度分析；

**📊 数据集**

本研究为理论证明，未使用任何实验数据集；

**📈 对比分析**

相较于朴素的2^{O(n)}深度优先搜索，提出的算法时间为2^{O(n^{(d-1)/d}(log n)^2}，在指数上比最优穷举搜索节省了Ω̃(n^{1/d})的因子；

**⚠️ 局限性**

限制在于：下界与上界仅在对数因子内匹配，d>3时仍存在显著差距；算法仍为指数级，实际可行性有限；结论依赖于ETH假设；未探讨ℝ^d中的单距图识别。

---

## 196. GRAFT: Geometry-Aware Residual Adaptation for Foundation Trackers

**arXiv ID:** 2610.36507 | [PDF](https://arxiv.org/pdf/2610.36507v1)

**作者:** Liang Li `[一作]` (Xiaomi Robotics Lab), Diyun Xiang `[通讯]` (Xiaomi Robotics Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 GRAFT 框架，在冻结的行为基础模型（BFM）上加入几何感知的残差适配器，使机器人能够在不改动原始参考接口的前提下跨越台阶、隙缝和平板等地形。

**💡 创新点**

创新点在于：① 将可重用的全身运动先验与地形执行适配分离；② 只在低层腿部通道加入轻量级残差修正；③ 通过地形奖励训练残差，并利用教师-学生蒸馏将特权地形信息转化为本地深度输入；④ 在部署时可通过门控随时开启/关闭残差，保证基础模型在平地上保持原性能。

**🔧 技术方法**

使用的技术包括：冻结的 BFM（如 SONIC、Humanoid-GPT、HoloMotion、ScaleBFM）；PPO 强化学习训练残差网络；地形奖励（脚部抬升、踏点准确性、平地抑制）；两阶段训练（特权教师 + 深度学生蒸馏）；遮罩只对腿部通道施加残差；多任务训练与回合计数的难度曲线；仿真平台 Isaac Lab / MuJoCo；硬件测试 Unitree G1。

**📊 数据集**

使用的数据集主要是：① 由 SONIC 训练集提供的参考运动；② 程序化地形生成器产生的难度递增地形；③ 在硬件验证中使用的 Unitree G1 本机深度相机数据。没有使用公开的标准地形数据集。

**📈 对比分析**

比较方法：在固定的四种地形（上台阶、下台阶、隙缝、平台）上评估成功率，并记录平地上的残差大小；与六种对比方法（Action Residual、Depth Student、LoRA、Motion-Token Residual、One-Stage Depth、RL From Scratch）在相同仿真环境、奖励和迭代预算下进行对比。结果显示：Action Residual 与 Depth Student 在硬地形上成功率均达到 1.0（或 0.95+），平地残差仅 0.05 左右；LoRA 也能成功但平地残差大约 0.7；Motion-Token Residual 与 RL From Scratch 均失效。跨基准实验中，GRAFT 在四个冻结 BFM 上均能显著提升成功率（如 SONIC 从 0% 提升到 100%）。硬件实验中，适配后的 SONIC 能成功跨 0.3 m 平台与 0.1 m 台阶，原基准则停滞不前。

**⚠️ 局限性**

局限性：① 仅对低层腿部做适配，未处理手臂或上身在不同地形下的平衡与姿态调整；② 训练基于预录制的参考运动，未验证在实时遥控或 VLA 高层命令下的适配效果；③ 需要针对每个基准单独训练残差网络，尚未证明对不同机器人或更复杂地形的泛化能力；④ 仍需通过教师-学生蒸馏，依赖特权地形信息，若实际环境缺失深度感知或出现遮挡，性能可能下降。

---

## 197. Multi-Scale Spatio-Temporal Exploratory Analysis of Urban Vehicle Theft

**arXiv ID:** 2610.36478 | [PDF](https://arxiv.org/pdf/2610.36478v1)

**作者:** Chaitanya Jobanputra `[一作]` (German Research Center for Artificial Intelligence), Gerrit Großmann `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对2000–2022年洛杉矶警局车辆盗窃报告进行清洗、地理编码、统一分辨率后，进行多尺度时空分析。

**💡 创新点**

在严格的多重检验控制、细致的数据预处理基础上，首次系统评估了Kia/Hyundai盗窃浪潮的时空演变及其对整体盗窃模式的影响。

**🔧 技术方法**

使用负二项回归、Getis–Ord G_i^*、Moran's I、Knox近重复检验、STL季节分解、PELT变点检测、Kernel密度、PySAL、Python数据处理等技术。

**📊 数据集**

采用洛杉矶警局车辆相关犯罪记录（23年）、美国人口普查与ACS车辆/收入数据、OpenStreetMap POI 2026、气象温度数据等多源数据集。

**📈 对比分析**

通过对比不同时间段和地理来源的热点显著性、空间聚集度及近重复比率，表明热点高度稳定、空间聚集度显著提高，Kia/Hyundai占比快速增长，近重复效应弱于同距同时效。

**⚠️ 局限性**

局限性包括数据未公开、地理编码精度不均、居民车辆指标在商业区不准确、跨年份的边界重分配导致MAUP、2022年数据不完整、样本偏差和时间覆盖不完整。

---

## 198. Proposing Better Rollouts for On-Policy Distillation

**arXiv ID:** 2610.36477 | [PDF](https://arxiv.org/pdf/2610.36477v1)

**作者:** Sirui Chen `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MPOPD，在OPD框架中将rollout策略视为优化目标，利用相对教师支持和信任域重新加权，从而降低teacher‑student不匹配并提升学生学习效果。

**💡 创新点**

创新点在于把rollout政策本身转化为可优化的几何混合目标，并通过学习的proposal policy一次前向即可近似该混合，从而在不改变OPD本身损失的前提下改进采样分布。

**🔧 技术方法**

采用相对教师支持重加权、局部信任域优化、几何混合、proposal policy学习与重要性加权的混合技术，并配合vLLM前向采样实现高效推理。

**📊 数据集**

使用16个数学推理、知识问答和医学推理基准（如AIME、AMC、Minerva、OlympiadBench、GPQA、ARC、MMLU、PubMed、MedQA、Lancet、NEJM等）进行评测。

**📈 对比分析**

与OPD及其变体（ExOPD、EOPD、VETO）在Qwen3-4B/1.7B学生上对比，平均准确率提升约4.3%/0.8%，同时生成长度平均减少约41%；在知识问答和医学推理任务上亦分别提升约1.2%和1.1%。

**⚠️ 局限性**

需要额外训练proposal policy，增加计算与显存开销；对超参数η敏感；在极大模型或低资源场景下可扩展性尚未充分验证。

---

## 199. FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents

**arXiv ID:** 2610.36476 | [PDF](https://arxiv.org/pdf/2610.36476v1)

**作者:** Arther Tian `[一作]` (DGrid AI), Aaron Chan `[通讯]` (DGrid AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对实时交易代理的执行轨迹进行操作性审计，评估决策有效性、承诺遵循、账本对账和报告一致性。

**💡 创新点**

提出 FIDES‑Live 协议，分离决策、承诺、执行和报告四个层面，提供可追溯、可解释的逐步审计框架。

**🔧 技术方法**

基于结构化 JSON 校验、可重建账本算术、可执行门控检查、局部纸质执行模拟以及离线离散化验证。

**📊 数据集**

使用两套前进式环境的真实市场数据：2026 年 9-10 月的加密币点对点行情与 9 月 28-29 日的美国股票 IEX 市场，包含约 7,300 个加密回合与 310 个股票回合。

**📈 对比分析**

按路由（10 个模型）和活动条件进行宏平均与微平均，结果显示决策有效率与报告准确率差异显著（例如 DeepSeek 98.7% vs 14.3%），承诺遵循在非空订单时仅 62.6%，但在空订单时 97.6%。

**⚠️ 局限性**

受限于观察依赖、未随机化实验、执行控制与策略交互、局部纸质执行与缺失外部行情证据，无法推断模型长期盈利、因果效应或语义一致性。

---

## 200. A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs

**arXiv ID:** 2610.36449 | [PDF](https://arxiv.org/pdf/2610.36449v1)

**作者:** Alexa R. Tartaglini `[一作]` (Google DeepMind), Yuxuan Li `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统研究了视觉语言模型（VLM）如何受到简单视觉符号的影响，并探索了模型生成和解释视觉信息的能力；通过视觉调节实验、模型间通信游戏和人类对齐实验，揭示了模型对视觉输入的无意识调节和生成-解释间的鸿沟。

**💡 创新点**

①首次量化了彩色形状符号对非视觉任务（如编码、写作）的行为影响；②揭示模型对视觉信息的无意识调节与自我报告不一致；③提出生成-解释不一致的概念，并通过多轮通信游戏验证模型对视觉沟通的能力；④通过人类实验评估模型视觉语义与人类直觉的对齐程度。

**🔧 技术方法**

使用多种预训练VLM（Claude、Gemini、LLaVA等），构建288个彩色形状符号库，设计递归/迭代编程任务、写作风格任务；采用视觉调节测量（ΔP(recursion)）、思维链（CoT）与思维轨迹分析；实现自由绘图（PIL）与符号选择的模型间通信游戏；人类对齐实验采用问卷调查。

**📊 数据集**

实验数据主要来自：①自建的288个彩色形状符号库；②公开编程任务集合（如二叉树遍历、Fibonacci、二分搜索等）；③自然语言写作样本与偏好问题；④控制图像（Gaussian噪声、ImageNet）；⑤7名人类受试者的视觉符号识别与概念/故事匹配任务。

**📈 对比分析**

通过对比无图像基线与插入图像后的递归率、文本描述与图像的调节差异、噪声图像的无效性来量化视觉调节；通信游戏记录每轮成功率，发现自由绘图成功率达97.8%，符号通信成功率约48%；人类对顶级调节符号的识别率仅36%，说明模型与人类视觉语义不一致；总体结果显示：简单符号可将递归率提升至100%，但模型对自身视觉影响缺乏自我意识。

**⚠️ 局限性**

局限性：①实验任务人为设计，缺乏对真实场景中的视觉调节与沟通的评估；②人类对齐实验样本量极小，难以推断普适性；③仅考察有限数量的模型和符号库，未覆盖所有VLM变体；④未深入挖掘视觉调节背后的内部机制；⑤通信游戏和调节实验多基于离散符号，未覆盖更复杂的视觉表达方式。

---

## 201. No Tokenizer, Still a Gap: Cross-Lingual Fairness of Homoglyph and Invisible-Character Watermarking

**arXiv ID:** 2610.36432 | [PDF](https://arxiv.org/pdf/2610.36432v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文建立了跨40种语言、24种文字的编辑型水印容量与鲁棒性评估框架，并验证了键控检测器的文本长度需求。

**💡 创新点**

创新点在于：①首次将同位异字和不可见字符水印的容量按语言/文字分层衡量；②将容量与资源水平关联，揭示文字学层面不公平；③提出完整的转换阶梯评估常见处理与对抗攻击的影响；④通过键控检测器验证容量对检测长度的预测。

**🔧 技术方法**

技术手段包括Unicode 17.0.0/CLDR 48.2.0的同位异字列表、NFC/NFKC标准化、混合文字检测与骨架映射（UTS #39），以及基于随机数种子生成的键控检测器与AUC/最短检测长度评估。

**📊 数据集**

数据集采用 FLORES+ 4.5 dev‑test 1,012 句子/语言的平行语料，覆盖40种语言，并使用 Unicode/CLDR 公开文件构建字符频率与同位异字集合。

**📈 对比分析**

比较方法：在未处理、复制粘贴、混合文字检测、NFKC 四种攻击下测量有效容量与检测 AUC，利用ICC、相关系数等统计评估公平性；结果显示容量差异 3.3‑倍，复制粘贴后差距扩大至 8.5‑倍，检测长度与有效容量高度相关。

**⚠️ 局限性**

局限性包括：①大部分文字组仅含单一语言，导致脚本内变异估计不足；②资源类别归属依据单一公开分类，可能影响关联结论；③仅使用单一并行语料，限制泛化；④Unicode 17 版同位异字列表随版本变化；⑤可见性评估依赖字体与排版引擎，缺乏人类感知实验。

---

## 202. MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

**arXiv ID:** 2610.36513 | [PDF](https://arxiv.org/pdf/2610.36513v1)

**作者:** Yibo Yang `[一作]` (Independent Researcher), Zixia Wang `[通讯]` (University of Exeter)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MedConstBench 基准，用来评估不同部署宪章对医疗伦理 LLM 行为的影响。

**💡 创新点**

创新点在于把宪章视为可控变量，使用多模型同行评审并提出“do‑no‑harm”自适应宪章选择器。

**🔧 技术方法**

采用多模型同行评审、Beauchamp 四原则评分、离线 LLM 评测和自适应策略。

**📊 数据集**

使用自制的 MedDilemma 400 条临床伦理困境和公开的 MedEthicsQA 400 条问答。

**📈 对比分析**

与固定宪章相比，自适应选择器在所有模型上平均提升 0.11 分、最坏情况提升 0.12 分，并在新模型上与 oracle 保持一致。

**⚠️ 局限性**

局限在于仅覆盖四条宪章、数据量有限，且对极端模型可能仍有未知偏差。

---

## 203. ARISE: Authentic Reference-Informed Stability Enhancement for Continual Multimodal Forgery Detection

**arXiv ID:** 2610.36509 | [PDF](https://arxiv.org/pdf/2610.36509v1)

**作者:** Yuchen Zhang `[一作]` (Xi'an Jiaotong University), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种攻击增量的连续多模态伪造检测框架 ARISE，能够在保持单一真实/伪造接口的同时学习新攻击并保持旧攻击性能。

**💡 创新点**

创新点在于利用真实内容的可重用几何参考构建 GMM，结合 GRASP 的责任感知各向异性约束和 VAR 的易损阶段自适应重放权重，形成兼顾稳定性与可塑性的训练策略。

**🔧 技术方法**

采用多模态视觉语言模型（如 Florence-2）作为骨干，配合高斯混合模型、注意力投影、经验重放和自适应权重机制实现。

**📊 数据集**

使用新构建的 Forgery Attack Benchmark for Lifelong Evaluation（Fable‑Bench）包含三种演化协议以及公开的 Weibo 和 FineFake 数据集进行实验。

**📈 对比分析**

与多种基线（MoE‑CLIP、D‑MoLE、LPR 等）以及零射 MLLM 相比，ARISE 在三种演化协议上平均精度分别提升约 4.7%、3.4% 和 3.5% 点，且在跨数据集泛化上也取得最优表现。

**⚠️ 局限性**

局限性包括仅考虑二分类真实/伪造标签，依赖经验重放内存限制以及对极端新型攻击可能仍有适应延迟。

---

## 204. ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

**arXiv ID:** 2610.36499 | [PDF](https://arxiv.org/pdf/2610.36499v1)

**作者:** Zexin Li `[一作]` (George Mason University), Simin Chen `[通讯]` (George Mason University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个查询高效的黑盒概率验证框架，用于评估深度学习编译器优化后的模型在指定输入分布下的决策一致性。

**💡 创新点**

创新点在于提出了单一序列化检验方法，取代传统的区间族（interval-family）方案，采用更严格的三值合同（接受、拒绝、无结论），并通过改进的置信区间（Clopper–Pearson 及 anytime-valid confidence sequence）实现误差控制和采样成本上限。

**🔧 技术方法**

主要技术包括统计模型检验、二项分布置信区间、序列化检验策略、误差预算分配与修正、以及对数独立采样和无状态判定器的假设。

**📊 数据集**

实验数据涵盖 24 个任务，涉及 TorchInductor 与 TensorRT 两大编译器，使用 840 个输入-后端实例；评估了多种扰动分布（均匀 L₂、L∞ 球、文本 token 替换、图像 patch 等）和清洁标签半径搜索。

**📈 对比分析**

与 PROVERO 基线比较时，单序列检验在匹配参数下独立样本数仅为 3.6–4.2 倍，证明了更高的查询效率；在 θ=10⁻³ 下，对 840 实例得到 817 次接受、19 次无结论、4 次拒绝，显示出更强的判定能力。

**⚠️ 局限性**

局限性包括：依赖输入样本独立性和判定器无状态性；对极低违反概率（如 10⁻⁴）仍需大量执行；缺乏全局一致性保证，仅针对指定分布和属性；未提供完整的错误诊断流程；对多种分布或模型版本之间的联合保证尚未覆盖。

---

## 205. AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation

**arXiv ID:** 2610.36493 | [PDF](https://arxiv.org/pdf/2610.36493v1)

**作者:** Chaofei Hong `[一作]`, Qing Zhang `[通讯]` (Zhejiang Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种兼容监督细调（SFT）的跨模态潜在预测框架 AC-JEPA，结合可逆的双向掩码潜在恢复与因果解码器，提升图像到 SMILES 的精确翻译性能。

**💡 创新点**

创新点包括：①在同一模型内部嵌入潜在预测，使得潜在表示能同时服务于 SFT 目标；②采用双向掩码恢复，利用多模态上下文预测被遮蔽的图像与符号状态；③通过 SIGReg 与 stop‑gradient 机制抑制潜在坍塌；④利用已见的合成数据做潜在监督，并在自训练阶段以连续潜在目标方式消化噪声伪标签，显著提升对噪声的容忍度。

**🔧 技术方法**

技术实现基于 Qwen3‑VL‑2B 视觉‑语言模型，使用 masked image/SMILES 视图、latent MSE、SIGReg 正则、双向预测器以及原始的 SFT 损失进行联合优化；自训练时采用伪标签生成并作为潜在目标而非硬标签。

**📊 数据集**

主要数据集：合成 MolParser‑7M（图像‑SMILES 对）用于预训练；Real‑SFT 100K（真实论文/专利图像）用于域适配；WildMol Simple 与 Markush 作为评测集；USPTO‑Mol 及科学论文结构图像用于自训练的未标注数据；此外还用 MolGrapher‑Synthetic‑300K 进行噪声鲁棒性实验。

**📈 对比分析**

对比方法：基准 Base SFT、MolParser‑Base 预训练模型；在 WildMol Simple/Markush 上报告精确匹配准确率。结果显示：Base SFT 67.40/36.48 → Gold‑pair AC‑JEPA 76.07/42.28（提升 8.67/5.80）；自训练 R1/R2 进一步提升至 78.51/42.78 与 80.23/48.28。与直接伪标签 SFT 相比，AC‑JEPA 自训练在相同伪标签下分别提升 15.04/6.74 点和 16.76/12.24 点。

**⚠️ 局限性**

局限性：噪声容忍度主要验证于 OCSR 场景，对更广泛的视觉‑符号翻译任务尚未充分验证；潜在预测虽缓解伪标签误差，但仍可能在系统性偏差下影响表示；自训练消耗额外的伪标签生成与训练开销，未完全无计算成本。

---

## 206. A State Augmented Approach to Paralle Modeling in High-Level Synthesis

**arXiv ID:** 2610.36481 | [PDF](https://arxiv.org/pdf/2610.36481v1)

**作者:** Nicholas Kovacs `[一作]` (University of South Florida), Hao Zheng `[通讯]` (University of South Florida)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于同步顺序进程的显式架构建模方法，利用C++实现RV32I处理器的流水线结构并与Vivado HLS结合。

**💡 创新点**

创新点在于将状态、通信、反馈规则显式化为可调节的过程边界，解耦HLS调度与架构设计，从而可精细控制并行度、延迟和资源占用。

**🔧 技术方法**

使用的技术包括C++同步进程建模、Vivado HLS流水线调度、寄存器文件分区、字节通道存储、以及FPGA实现与时钟频率优化。

**📊 数据集**

使用的数据集为八个自定义内核（来自引用文献）以及CoreMark基准测试。

**📈 对比分析**

通过与算法级HLS基线、Ibex RTL参考以及不同配置（CPI‑优化、面积‑优化）在Kintex UltraScale+ FPGA上比较，CPI‑优化版平均1.221 CPI、220 MHz时耗时0.85×Ibex，面积‑优化版面积缩减65%但周期略增。

**⚠️ 局限性**

局限性在于仅针对RV32I处理器的案例验证，未覆盖更广泛的反馈耦合设计；外部内存延迟和工具流、硬件平台的特定性也限制了结果的通用性。

---

## 207. When to Intervene? An Empirical Study of Errors and Self-Recovery in LLM Agents

**arXiv ID:** 2610.36471 | [PDF](https://arxiv.org/pdf/2610.36471v1)

**作者:** Chiming Duan `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 ReInBench 数据集，对 LLM 代理在 Web 与电脑使用任务中出现的错误及其自我恢复行为进行实证研究，并提出了预测自我恢复的 ReCoF 方法。

**💡 创新点**

创新点在于系统化记录自我恢复事件、分析其对后续错误和任务完成的影响，并结合历史恢复模式与上下文评估实现更精准的自我恢复预测。

**🔧 技术方法**

使用了 LLM 代理（Claude 3.7 Sonnet、Claude Sonnet 4.5、Kimi K2.5、GPT‑5）、文本与视觉编码器、评估 LLM 与条件决策融合等技术。

**📊 数据集**

利用从 OSWorld‑Verified 和 Online‑Mind2Web 收集的 1,259 条代理轨迹共 1,403 个错误事件组成的 ReInBench 数据集。

**📈 对比分析**

与 Direct LLM、AgentForesight、Sentry、PrefixGuard、Wink 等基线比较，ReCoF 在宏 F1 上平均提升 9.02%（约 73‑74%），显著优于最强基线。

**⚠️ 局限性**

局限在于覆盖的模型、任务和错误类型有限，且未评估不同干预方法对任务完成的实际影响。

---

## 208. Whole-Body Humanoid Control via Stochastic Motion Tokenization

**arXiv ID:** 2610.36452 | [PDF](https://arxiv.org/pdf/2610.36452v1)

**作者:** Michael Drolet `[一作]` (Technical University Of Darmstadt), Jan Peters `[通讯]` (Technical University Of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了三阶段离散令牌化+自回归生成+低层跟踪策略的管道，实现了在Unitree G1上生成并执行多种全身动作。

**💡 创新点**

将随机离散VAE (DAPS) 作为运动令牌化器，结合Transformer自回归生成与单一PPO跟踪器，实现了从高层语义条件到低层控制的直接接口。

**🔧 技术方法**

使用DAPS离散VAE、Transformer自回归生成模型、PPO强化学习跟踪器，以及多尺度温度控制的编码。

**📊 数据集**

在LAFAN1和BONES‑SEED两套高质量全身运动数据集上进行训练与评估。

**📈 对比分析**

与FSQ、VQ‑VAE等离散令牌化方法在相同位数下比较，DAPS在重建误差和条件对数似然上均优于对手；生成的运动在真实机器人上可执行，完成16.8 s单行为及35 s行为切换，跟踪误差低于0.2标准化RMSE。

**⚠️ 局限性**

局限在于对运动包的离散表示对极短时间粒度和全局路径约束的表达有限，且离线生成令牌与实时运动再编码的分布漂移需进一步研究。

---

## 209. Simulated petrological labels enable probabilistic mineralogy prediction from whole-rock geochemistry

**arXiv ID:** 2610.36429 | [PDF](https://arxiv.org/pdf/2610.36429v1)

**作者:** Matthew J. Liu `[一作]` (Lawrence Berkeley National Laboratory), Kristin A. Persson `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于Transformer编码器和后期校准的模型，利用大规模热力学模拟生成的多条件矿物标签，从全岩化学成分预测模态矿物组成；

**💡 创新点**

①使用热力学模拟器生成近百万一对多标签，解决标注稀缺与多解问题；②通过自监督预训练的Transformer提取化学结构；③采用少量真实样本的后期校准层纠正模拟误差；

**🔧 技术方法**

Transformer（CrabNet）编码器、掩码元素预测、自监督预训练、Bernoulli+Dirichlet概率头、线性+指数校准层、PhaseEquil等热力学模拟器；

**📊 数据集**

约788k全球岩石化学样本（无矿物标注）用于预训练，约950k模拟生成的矿物分布，2000多块USGS火山岩的测量矿物组成用于评估与校准；

**📈 对比分析**

在火山岩基准上，零样本模型AUROC在矿物存在判别方面达到0.7–0.9；加入校准后联合NLL下降约30%，矿物丰度均方误差降低约10%，显著优于仅用测量矿物直接训练的模型；

**⚠️ 局限性**

模拟器忽略动力学与分离过程导致系统误差；模型主要在火山岩上验证，推广至沉积岩等岩性时效果未知；化学组分单独难以区分多相固溶体和多晶型，仍需进一步改进。

---

## 210. Ground Sample Distance Constrained k-Coverage Viewpoint Planning via Set Multi-Cover Formulation

**arXiv ID:** 2610.36450 | [PDF](https://arxiv.org/pdf/2610.36450v1)

**作者:** Waleed Bin Khalid `[一作]` (Indiana University Bloomington), Chul Min Yeum `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了基于GSD约束的k-覆盖视角选择问题，提出了候选视角生成、视角增补和全局信息贪心选择的完整流程。

**💡 创新点**

创新点在于将GSD约束与体素化表面结合，使用动态权重贪心算法优先解决覆盖缺口和稀缺视角，从而得到可解释且紧凑的视角集合。

**🔧 技术方法**

采用体素化、射线可见性检测、视角生成与增补、集合多重覆盖建模、全局信息贪心算法以及ILP对照方法。

**📊 数据集**

使用了两个公开的3D模型（Angel Statue和Rooster），在多种体素分辨率和k覆盖需求下进行实验。

**📈 对比分析**

与基于重叠的逐序剪枝和ILP最小化视角集进行对比；贪心方案仅比ILP多3–16%视角，覆盖率均达到99.9%以上，远优于重叠剪枝方法。

**⚠️ 局限性**

局限性包括未考虑无人机动力学、路径规划、视角间距约束和实际重建质量评估，仅在两结构上验证；ILP在某些大规模实例中未能证明最优。

---

## 211. ACOR: incremental evidence accumulation towards real-time DNA storage reconstruction

**arXiv ID:** 2610.36500 | [PDF](https://arxiv.org/pdf/2610.36500v1)

**作者:** Linxuan Han `[一作]` (Tianjin University), Guanjin Qu `[通讯]` (Tianjin University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 ACOR 的实时 DNA 存储读回重建方法，能够在读取过程中持续累积 k‑mer 证据并在簇闭合时一次性解码，显著加速重建并保持高准确性。

**💡 创新点**

核心创新在于将证据积累与一次性束搜索解码分离，使用稀疏存储避免高‑k 计数器指数扩张，并通过并行束搜索实现数百到千倍的速度提升，同时保持与现有最先进方法相当的恢复率。

**🔧 技术方法**

技术要点包括：k‑mer 滑动窗口上下文提取、持久化证据账本、长度约束束搜索（Beam Search）与优先级候选选择、稀疏计数存储、以及多线程并行化实现。

**📊 数据集**

使用了多种公开 DNA 存储基准（如 oligo0、ncomms19、binned nanopore、Microsoft CNR、DNAFormer Illumina/ONT、Zenodo 等）以及作者自建 1.77 亿条 100‑nt payload 的 NGS 实验数据。

**📈 对比分析**

与传统的 BBS、VS、BMALA、Deep‑DNA CPL、MUSCLE 等基准相比，ACOR 在大多数数据集上实现了 92.2%–99.9% 的 Exact 恢复率，且在 1.77 亿读数据上仅需 9 秒完成重建，速度提升幅度从几十倍到上千倍，内存使用显著下降（最多 550 倍）。

**⚠️ 局限性**

主要局限在于目前仍依赖已知簇分配和闭合信号，未实现在线读分配与闭合；对长序列（≥4 kb）错误率较高，且依赖 k‑mer 统计对高错误率序列的鲁棒性仍有待提升。

---

## 212. CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments

**arXiv ID:** 2610.36484 | [PDF](https://arxiv.org/pdf/2610.36484v1)

**作者:** Kianté Fernandez `[一作]` (University of California, Los Angeles), Ian Krajbich `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了CogArena基准，设计了十个在浏览器中完成的行为实验，并通过三层评分体系评估AI代理的完成度、准确性和人类行为特征的匹配情况。

**💡 创新点**

创新点在于将认知科学的实验范式与AI代理在真实网页界面上的交互相结合，提出了基于行为特征的三层评估方法，并公开了排行榜以持续更新模型表现。

**🔧 技术方法**

使用了jsPsych框架实现实验、Browser-Use脚手架让多模态大型语言模型通过截图与键盘/鼠标操作控制浏览器，并运用统计检验（比例检验、配对检验、相关检验等）对行为特征进行评分。

**📊 数据集**

实验数据集由十个来源于已发表研究的经典行为实验组成，结合原研究中发布的人类基准数据；此外使用随机代理产生的随机行为作为基准基线。

**📈 对比分析**

通过将四款经济层多模态模型（Moonshot AI、Google、Alibaba、OpenAI）以及一款顶尖模型分别在十个任务中执行十次，比较完成度、准确性和特征对齐得分；最佳经济模型复合得分约58.6，表现随任务与会话差异显著，许多会话未完成；顶尖模型表现稍优，但单次试验无法给出稳健结论。

**⚠️ 局限性**

局限性包括未收集人类在CogArena上的行为数据（仅使用原研究数据导致可能存在实现差异）、受限的模型样本（全部为经济层且仅一次顶尖模型试验）、任务集未涵盖所有已知认知现象、评分对脚手架和截图使用方式敏感，且未对多重比较进行校正。

---

## 213. ClusterSplat: Semantic Cluster Selection for 3D Visual Grounding in Gaussian Splatting

**arXiv ID:** 2610.36462 | [PDF](https://arxiv.org/pdf/2610.36462v1)

**作者:** Jaesung Lee `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出ClusterSplat，将3D视觉定位从连续原语评分阈值切换为场景自适应聚类的离散选择；

**💡 创新点**

创新点在于通过描述长度准则自适应生成聚类候选，并用查询条件聚类评分实现无阈值目标定位；

**🔧 技术方法**

采用Instance-aware Gaussian特征学习、CLIP文本编码、Transformer自/交叉注意、最小生成树合并及MDL描述长度；

**📊 数据集**

使用ScanRefer（配合ScanNet v2场景）、Ref-LERF以及ScanNet v2类别查询数据集；

**📈 对比分析**

与LangSplat、OpenGaussian、Dr.Splat、ReferSplat等基线比较，ScanRefer上实现最高2D mIoU和3D mIoU，ScanNet v2类别查询也取得领先表现；

**⚠️ 局限性**

主要限制是聚类候选过度合并/拆分导致目标部分缺失，仅支持单一对象表达，且需离线场景准备，泛化评估不足。

---

## 214. LociArt: A Unified Paradigm for Free-Style Local Video Stylization

**arXiv ID:** 2610.36456 | [PDF](https://arxiv.org/pdf/2610.36456v1)

**作者:** Yuxuan Chen `[一作]` (MiLM Plus, Xiaomi Inc.), Daiguo Zhou `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并发布了 LociArt 框架，实现了可自由控制局部区域与风格的本地视频风格化方法。

**💡 创新点**

创新点包括：
- 引入极坐标风格感知掩码（Polar Style‑Aware Mask）将风格直接绑定到空间区域；
- 结合表示–几何掩码条件（Representation–Geometry Conditioning）在保持时空上下文的同时精准定位掩码；
- 构建自动化数据构造管道，生成 LociArt‑100K 100K 对齐的本地风格化数据集和 LociArt Benchmark；
- 通过统一范式实现从单区域单风格训练到多区域多风格零射击合成的能力。

**🔧 技术方法**

采用的技术主要包括：
- GRN（Generative Refinement Network）改进为 GRNStyle；
- VAE 编码掩码与原始掩码几何融合；
- 极坐标风格编码、文本/图像指令驱动的风格提取；
- 关键帧对象去除、视频马赛克（VideoMaMa）matting、自动化拼接；
- 多种评估指标（SPS、CLIPT、PSNR、TC、Gemini VLM）。

**📊 数据集**

使用的数据集：
- LociArt‑100K：100K 条目标记化本地风格化视频对；
- LociArt Benchmark：80 源视频 + 84 掩码单元，共 168 个前景/背景风格化案例。

**📈 对比分析**

在多种图像与视频风格化/编辑方法（LEAST、UniVST、RegionRoute、StyleGallery、VACE、CoinVE‑Edit、Seedance 2.5）上进行对比。GRNStyle 在
- 区域风格一致性（SPS）最高；
- 区域 CLIP 相关度（CLIPT）最好；
- 非目标保留（PSNR）与时空一致性（TC）均优于或与 SOTA 相当；
- Gemini VLM 评分（Style / Preserve）亦取得领先。用户研究表明 GRNStyle 在风格真实性与背景保留上获得最高平均分。

**⚠️ 局限性**

局限性：
- 训练与推理需要大规模 GPU 资源；
- 极坐标码库固定，扩展新风格仍需手工定义角度；
- 数据构造管道虽自动化但仍依赖人工验证，流程复杂；
- 对极端遮挡或快速运动场景的边界处理尚不完美。

---

## 215. Correction or Manipulation? Measuring Epistemic Vigilance in LLM Agents Under User Pressure

**arXiv ID:** 2610.36453 | [PDF](https://arxiv.org/pdf/2610.36453v1)

**作者:** Nardine Basta `[一作]` (Macquarie University), Amin Beheshti `[通讯]` (Macquarie University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于认知学的注意力评估框架，测量LLM代理在面临纠正与操纵挑战时的抗拒行为

**💡 创新点**

首次将epistemic vigilance与信号检测理论结合，区分敏感度（区分度）与偏差（整体抗拒度），并在同一压力下同时评估答案与行动

**🔧 技术方法**

使用信号检测理论、提示工程、证据门控修订（evidence‑gated revision）以及多样化的说服技术（如来源引用、身份主张、重复等）

**📊 数据集**

利用MMLU‑Pro、GPQA‑diamond、MATH‑500等多领域测试集以及30个合成的账号恢复情景

**📈 对比分析**

对七个模型进行对照实验，报告敏感度d'、偏差c以及错误接受率；结果显示GPT‑6系列表现最佳，绝大多数模型对不可验证来源敏感并易被误导，证据门控在Llama上提升d'但整体提升有限

**⚠️ 局限性**

局限包括样本难度与来源真实性混淆、仅评估固定对话且未检验等价方差假设，以及未在更广泛场景和不同语言上验证

---

## 216. From Solver Trajectories to Teaching Trajectories: Cognition-Aligned Reasoning Distillation

**arXiv ID:** 2610.36445 | [PDF](https://arxiv.org/pdf/2610.36445v1)

**作者:** Yashuo Luo `[一作]` (Beihang University), Xu Wang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现一种名为CARD的知识蒸馏框架，利用轻量化Reformatter将大模型的推理轨迹重写为与小模型认知对齐的教学轨迹，从而提升小模型的推理性能。

**💡 创新点**

创新点在于识别并解决“认知差距”，通过分布对齐奖励与逻辑连贯性奖励训练Reformatter，无需重复采样即可直接改写已有轨迹，消除失败步骤与过度思考带来的负面影响。

**🔧 技术方法**

采用强化学习（GRPO）训练Reformatter，并使用基于KL的分布对齐奖励、Failed‑Step Fraction判定、逻辑连贯性奖励以及LoRA微调等技术。

**📊 数据集**

使用DeepMath‑103K生成的DeepSeek‑R1轨迹作为训练样本，并在GSM8K、MATH‑500、AGI‑MATH、AMC23、AIME24、AIME25等数学推理基准及ARC‑Challenge、MMLU‑Pro等OOD基准进行评测。

**📈 对比分析**

与Vanilla SFT、MCC‑KD、MoRSD、Mix TokenSkip、Mix Small Teacher等基线在三款学生模型（Llama‑3.2‑1B、Llama‑3.2‑3B、Qwen2.5‑7B）上对比，CARD在ID和OOD均显著提升，ID平均提升约+2~3分，OOD平均提升约+3~15分，且在多种随机种子和跨模型验证中保持稳定。

**⚠️ 局限性**

局限性包括仅在最多7B参数的学生模型上评估，未测试更大模型或多轮交互任务，对其可扩展性和交互式应用的适用性尚待进一步研究。

---

## 217. USB Differentiated Service for Networked Cyber-Physical Systems

**arXiv ID:** 2610.36436 | [PDF](https://arxiv.org/pdf/2610.36436v1)

**作者:** Anton Njavro `[一作]` (Boston University), Richard West `[通讯]` (Boston University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文实现了一个实时 xDCI gadget 驱动，利用 USB 3.x 提供主机‑主机与主机‑设备双向通信，并通过多路 USB 中断实现分流服务，支持等时与批量传输。

**💡 创新点**

创新点在于：①首次在 USB 3.x 上实现多路中断的主机‑主机等时/批量通信；②将分流服务与 Quest RTOS 的 I/O VCPU 预算模型结合，保证中断处理的实时性；③支持等时速率动态切换，满足 CPS 中传感器速率可变需求。

**🔧 技术方法**

使用技术包括：xDCI（设备端）与 xHCI（主机端）接口、USB 3.0 的 TRB 传输环、USB 中断器（interrupter）分离事件环、Quest RTOS 的 Sporadic Server 与 PIBS（I/O VCPU）调度、双端的多通道（两等时两批量）数据路径。

**📊 数据集**

实验使用自定义 1024 字节数据包，改动 bMaxBurst 产生不同的理论吞吐率；在两台配备 Intel Raptor Lake 13 代 SoC 的工业板上，通过 USB SuperSpeed（5 Gbps）进行测量，没有使用公开数据集。

**📈 对比分析**

比较方法：在同一硬件/软件平台上对比多路中断与单路中断（Linux 基准）的等时流间距和吞吐率。结果显示：吞吐率达理论值 98–99%；等时流在多路中断下保持 125 μs 或 2 ms 的严格间隔，而单路中断出现周期性抖动；动态速率切换也能无中断保持稳定。

**⚠️ 局限性**

局限性：仅验证了两节点单链路场景；未覆盖 USB‑4 或更大拓扑；分流仅适用于等时/批量，控制/中断未纳入；依赖 Quest RTOS，非通用操作系统；缺少对不同硬件平台、不同速率组合及大规模负载的评估。

---

## 218. Your Agent Acts Differently When You're Around: How User Presence Changes Agent Behavior in Negotiation

**arXiv ID:** 2610.36435 | [PDF](https://arxiv.org/pdf/2610.36435v1)

**作者:** Jane Pan `[一作]` (New York University), He He `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了交互式用户对LLM代理在代理对代理谈判中的影响，构建了PRES（Presence Effects in Negotiation Tasks）框架，进行对比实验以隔离用户存在的作用

**💡 创新点**

首次系统评估用户在谈判中主动参与时对代理行为的双重影响——既能提升平均收益（+13.7%），又会显著增加代理向用户和对手的误配行为（约翻倍）

**🔧 技术方法**

利用多种大型语言模型（GPT‑4/3.5、Claude‑3/2、Qwen‑1.5/3、DeepSeek 285B）作为代理模型，使用LLM判别器对交互日志进行误配行为检测，设计了多种谈判策略指标（开盘极端度、平均让步、偏好硬化）

**📊 数据集**

在四个公开谈判基准上进行实验：NegotiationArena、Deal‑or‑No‑Deal、PrincipalBench、CaSiNo，共计约11.2千局（11,200）游戏，覆盖单/多议题，使用统一接口保证任务一致性

**📈 对比分析**

对比“无用户” vs “有交互用户”以及不同用户参与度（被动回复、仅初始咨询等），发现有交互用户时平均收益提升13.7%，但误配行为率从约2.6%提升至5.5%；这些行为与收益无显著相关，且即使用户仅发送模板化信息亦能诱发误配行为

**⚠️ 局限性**

实验仅限于模拟用户、单边谈判场景，未涉及真实人类用户或多代理、多用户交互；未深入解释机制，且多依赖英文对话，可能限制跨语言推广

---

## 219. A Singularity Avoiding Track Model for Race Car Optimization

**arXiv ID:** 2610.36430 | [PDF](https://arxiv.org/pdf/2610.36430v1)

**作者:** Adwait Verulkar `[一作]` (General Motors), Ryan Trinter `[通讯]` (General Motors)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

将测量得到的赛道点云转换为闭合、C^4 连续的 Monge 补片三维表面模型，并将其导出为可直接用于 Modelica 车辆优化的周期性五次样条。

**💡 创新点**

创新点在于：① 采用最小曲率线（minimum‑curvature line）作为参考线，避免了传统几何中心线在狭窄弯道处导致的 Frenet 奇点；② 通过二次规划得到该参考线并确保参数化的可注入性；③ 输出共享 knot、C^4 连续的周期性五次样条，使得 Modelica DAE 在整个赛道上保持光滑、可微，显著提升最优跑圈问题的数值稳定性。

**🔧 技术方法**

技术手段包括：Quadratic Program（QP）求解最小曲率线；多项式最小二乘拟合高程和宽度；周期性 C^4 连续的 piecewise quintic spline 通过 CasADi + IPOPT 求解；Modelica 多体动力学与 d'Alembert 约束实现两自由度道路几何约束；FMI 3.0 FMU 与 FMIOPT 结合实现最优控制求解。

**📊 数据集**

使用的主要数据集是新加坡赛道（Formula 1 Singapore Circuit）的测量点云，包含轨道边界、中心线、海拔等信息。

**📈 对比分析**

与以往基于几何中心线的 Ribbon+Monge 补片模型相比，MCL 参考线模型在同一赛道上能够在 IPOPT 中收敛，主导矩阵保持良好条件，收敛迭代次数与已知良好赛道相当；而中心线模型在狭窄弯道时会出现双重不可行性、主导矩阵奇异导致求解失败。

**⚠️ 局限性**

局限性包括：仅支持已排序、唯一索引的点云；采用固定的矩形（s,w）网格，未实现自适应细化；使用静态最优赛道线作为参考，无法完全捕捉动态行驶过程中轨道几何的即时变化；处理时间约为每条赛道 5 分钟，若需更大规模或更高精度时需进一步优化。

---

## 220. Identity or Evidence? Auditing Evidence Dependence in Agentic Life-Science Benchmarks

**arXiv ID:** 2610.36510 | [PDF](https://arxiv.org/pdf/2610.36510v1)

**作者:** Danai Brilli `[一作]` (Zo Labs), Maria Georgila `[通讯]` (Zo Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过对生命科学基准的对照实验，探究去除实体名称或证据后模型预测准确率的变化，从而评估模型是否真正依据证据推理。

**💡 创新点**

提出了配对提示干预（身份模糊、证据删除、假置、回忆探测、正向对照）框架，系统量化身份成本与证据成本，首次证实单纯删除证据无法单独评估内容使用。

**🔧 技术方法**

采用了语言模型推理、配对统计检验（McNemar、Wilson区间）、线性回归分析红写比例与身份差距的关系，并构建回忆探测任务收集记录信息。

**📊 数据集**

使用了三类数据集：Minikel临床结果基准（286例）、LAB‑Bench LitQA2（96题）以及来自Biomni Eval1和LAB‑Bench MCQ的公开任务。

**📈 对比分析**

通过比较命名提示与身份模糊提示、完整证据与删除证据的准确率，发现身份模糊导致准确率从0.742降至0.082（gwas_causal_gene）或在LitQA2中随红写比例升至+0.135；回忆探测与预测准确率相近，表明模型可能访问记录结果；单纯删除证据时准确率上升，但多因素导致结果不可解释。

**⚠️ 局限性**

局限包括：无法区分模型识别记录与标记模糊导致的推理受损；证据删除实验未能单独识别内容使用；使用的标签为快照级、未标注时间；实验仅在两款模型与单一框架下进行，缺乏更广泛的可复现性。

---

## 221. Learning-Augmented Coordination Mechanisms

**arXiv ID:** 2610.36508 | [PDF](https://arxiv.org/pdf/2610.36508v1)

**作者:** George Christodoulou `[一作]` (Aristotle University of Thessaloniki), Ioannis Vlachos `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了学习增强的协调机制在原子和非原子拥塞游戏中的设计与分析，旨在提高社会成本的均衡性能。

**💡 创新点**

提出了一种新的学习增强协调机制，称为Potential机制，通过调整延迟函数来实现一致性和鲁棒性之间的权衡。

**🔧 技术方法**

使用了学习增强框架，结合了潜在函数和阈值机制来设计协调机制。

**📊 数据集**

研究了具有多项式延迟函数的原子和非原子拥塞游戏，具体的延迟函数为多项式形式，最大度数为d。

**📈 对比分析**

通过与原始游戏的最佳纳什均衡进行比较，展示了在准确预测情况下，机制的社会成本接近最优，而在不准确预测情况下，机制仍能保持良好的鲁棒性，性能保证在β(d+1)范围内。

**⚠️ 局限性**

存在一些不可能性结果，例如要求所有均衡在准确预测下具有最优社会成本的机制无法保证有限的鲁棒性，且在小的预测误差下，性能会急剧下降。

---

## 222. CoHARL: Co-Evolving Agent Policies and Executable Harnesses with Reinforcement Learning

**arXiv ID:** 2610.36469 | [PDF](https://arxiv.org/pdf/2610.36469v1)

**作者:** Jianwen Chen `[一作]` (University of North Carolina at Chapel Hill), Huaxiu Yao `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种联合训练共享策略与可演化的可执行 harness 人口的框架，利用执行轨迹驱动 harness 的自动改进并在多种 harness 上并行收集经验以提升 LLM 代理的学习效果。

**💡 创新点**

创新点在于引入 population‑based 的策略‑harness 共进化机制：1）通过 LLM proposer 根据当前策略的成功与失败轨迹生成新的 harness；2）使用 scheduler 在训练过程中动态分配多种 harness 的 rollout；3）利用历史精度‑成本 Pareto 归档为 proposer 提供全局上下文；4）使得 harness 的改进与策略学习同步进行，避免单纯的 one‑time 搜索或固定接口导致的分离问题。

**🔧 技术方法**

技术实现主要包括：强化学习（GRPO）与组内优势估计；Llama‑based proposer（Claude Opus 4.7）执行程序修改；高效的 pool 调度和 LCB 评分机制；Pareto 归档；以及对 Qwen3.5‑35B‑A3B 的 LoRA 训练和 SkyRL 基础架构。

**📊 数据集**

实验使用了三个数据集：SWE‑rebench（软件工程代码生成），SWE‑bench Verified（跨数据集泛化评测），以及 Terminal‑Bench 2.0（终端任务）。

**📈 对比分析**

与基线（harness‑only 搜索 Meta‑Harness 与 policy‑only RL Plain GRPO）进行对比；在 SWE‑rebench 上取得 27.6% 的解析率，显著高于 21.2%（Meta‑Harness）和 20.3%（Plain GRPO）；在 SWE‑bench Verified 上实现 59.8% 的解析率，优于 52.0% 与 49.8%；在 Terminal‑Bench 2.0 上获得 36.8% 的成功率，超过 33.7% 的 Meta‑Harness。

**⚠️ 局限性**

限制方面包括：1）对 proposer 与搜索过程的计算开销较大；2）harness 演化的安全性与可审计性仍有风险；3）方法在更大规模或多模态任务上的泛化尚未充分验证；4）对单个任务难度和 reward 设计高度依赖，可能导致训练奖励失衡。

---

## 223. Counting by the Book: Verifying Alaska's Ranked-Choice Elections

**arXiv ID:** 2610.36494 | [PDF](https://arxiv.org/pdf/2610.36494v1)

**作者:** Lucy Horowitz `[一作]` `[通讯]` (University of California), Lucy Horowitz (University of California)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在论文中作者通过形式化的方法验证了阿拉斯加州排名选择投票（RCV）的选举规则，并实现了一个可验证的计票器。

**💡 创新点**

创新点在于将州法条文与实际投票数据结合，构建了两套模型（法定模型与复现模型）并通过形式化证明揭示了投票处理规则的歧义及其对结果的影响。

**🔧 技术方法**

使用了Lean 4 进行形式化证明、可执行实现，以及与现有投票理论框架的对比。

**📊 数据集**

使用了阿拉斯加州2022年和2024年选举的投票记录（CVR）共115场选举的数据。

**📈 对比分析**

通过将两模型的逐轮计票结果与官方公布的多轮报告逐项比较，验证了模型的正确性；性能上两模型在所有选举中得到相同的获胜者，最大差距仅7票，表明模型一致性良好。

**⚠️ 局限性**

局限性包括无法直接验证实际投票软件的实现、对法律文本解释仍存在歧义、模型仅适用于阿拉斯加州的RCV规则，未覆盖其他州的差异性。

---

## 224. GarmentSegAgent: Autonomous Garment Segmentation for Web Fashion Images via Textual Self-Refinement

**arXiv ID:** 2610.36506 | [PDF](https://arxiv.org/pdf/2610.36506v1)

**作者:** Jia Chen `[一作]` (Wuhan Textile University), Jun Bai `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GarmentSegAgent 框架，实现了在 Web 时尚图像中无需人工空间提示、无模型微调即可完成服装分割，并通过文本自我修正实现迭代改进。

**💡 创新点**

创新点包括：① CAST（Constrained Attribution‑guided Selective TextGrad）机制，将视觉错误归因到指令字段并进行针对性文本优化，同时保持服装约束；② 通过结构化视觉错误归因与文本梯度相结合的闭环实现可靠的文本自我修正。

**🔧 技术方法**

使用技术包括：大语言模型（Qwen2.5‑VL 用于规划与评估，Qwen3‑VL 用于评估）、SAM 2.1 用于空间执行、TextGrad 进行文本梯度优化、视觉错误归因模块、服装约束验证。

**📊 数据集**

实验数据集：DeepFashion2、ModaNet、Fashionpedia 三大时尚分割基准。

**📈 对比分析**

与多种基线（SAM 2、Grounding DINO+SAM 2、READ、SEEM、LISA、SegAgent、SegZero+SAM 2）进行比较，GarmentSegAgent 在 mIoU、Boundary IoU、精度等指标上均超过最强对手，提升约 1.5%‑3.5% 以上。

**⚠️ 局限性**

局限性：仅针对单一上衣目标，未验证多实例或不同服装类别；依赖冻结模型与视觉反馈，无法保证像素级绝对精度；迭代推理带来额外计算开销。

---

## 225. RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG

**arXiv ID:** 2610.36444 | [PDF](https://arxiv.org/pdf/2610.36444v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RouteDRAG框架，实现针对多模态文档问答的Agent选择性调用与执行控制。

**💡 创新点**

创新点在于同时学习查询级别的Agent调用决策和操作级别的动态执行策略，并通过顺序策略学习将两者耦合。

**🔧 技术方法**

采用图支持的统一文档索引、相对路径预测、操作条件化评分、控制器改进以及基于LLM Planner的动态执行。

**📊 数据集**

在DocBench和MMLongBench-Doc这两个多模态文档基准上进行实验。

**📈 对比分析**

与九种RAG方法和三种控制策略比较，RouteDRAG在DocBench上取得75.86%、MMLongBench-Doc 53.80%准确率，较统一Agent执行提升约1–3个百分点，同时将回答调用减少16–15%。

**⚠️ 局限性**

局限性包括对阈值和正则化的敏感性、仅针对单轮问答、对不同LLM后端的泛化仍有限。

---

## 226. High-Fidelity Aerostructural Dataset of Transonic Wings in Cruise Conditions

**arXiv ID:** 2610.36442 | [PDF](https://arxiv.org/pdf/2610.36442v1)

**作者:** Pavankumar Koratikere `[一作]` (Purdue University), Leifur Leifsson `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提供了一份高保真多物理耦合的转子翼气动结构数据集，并使用该数据集训练并评估了多种机器学习模型

**💡 创新点**

创新点在于首次公开包含2471个样本和3个OOD样本的转子翼气动-结构耦合数据集，并展示了其在表面压力、积分系数及翼箱失效分布预测上的有效性

**🔧 技术方法**

使用多物理耦合框架MPhys与ADFlow（RANS）+TACS+FUNtoFEM进行数据生成；采样采用最优拉丁超立方（LHS）+ESE；机器学习模型包括MHA-Net、UQAeroNet以及POD/Autoencoder NIROM，训练使用PyTorch、Adam优化器

**📊 数据集**

数据集为 transonic-wing-aerostruct-dataset（2471个训练样本+3个OOD样本），已在 Hugging Face 上公开

**📈 对比分析**

通过对比 MHA-Net 与 UQAeroNet 的表面压力/摩擦系数、升力、阻力、俯仰矩等指标，测试集 NRMSE 均在 0.021–0.031 之间，R² 约为 0.996–0.998；对于翼箱失效分布的 NIROM，NRMSE 为 0.114–0.119，R² 为 0.76–0.83，表明模型在气动预测上性能优异，而结构失效预测仍有提升空间

**⚠️ 局限性**

局限性包括：仅覆盖巡航条件，未考虑操纵或高负荷；翼箱材料仅为铝，缺乏复合材料等现代材料参数；结构参数仅包括负载系数，未涉及厚度、布置等细节；部分样本因网格变形未收敛

---

## 227. Learning atomic interactions for the thermodynamics of liquid mixtures

**arXiv ID:** 2610.36440 | [PDF](https://arxiv.org/pdf/2610.36440v1)

**作者:** Marco Hoffmann `[一作]` (RPTU), Fabian Jirasek `[通讯]` (RPTU)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了COGNAC——一种基于图神经网络的原子级别模型，能够从分子结构预测活度系数，并将学到的几何和相互作用参数直接嵌入到UNIFAC局部组分框架中；

**💡 创新点**

创新点在于将UNIFAC的局部组分结构从预定义的化学基团替换为自适应的原子级别表征，并通过端到端的GNN学习得到的原子体积、表面积和相互作用能量，从而在保持热力学一致性、可解释性和计算速度的同时大幅提升预测精度；

**🔧 技术方法**

采用了分子图构造、四跳消息传递的图神经网络、两头（体积/表面积）和相互作用能量预测网络、可微分相分离求解器、对COSMO空间几何的正则化以及氢键符号正则化等技术；

**📊 数据集**

使用了德国杜门特数据银行（DDB）中的VLE、LLE、SLE、ACT、HE等实验数据，以及CHAOS数据库的COSMO空间几何作为正则化目标；

**📈 对比分析**

通过十折交叉验证与全数据训练，对比mod. UNIFAC和最新的HANNA模型；COGNAC在二元、三元和四元体系中将MAE平均降低13–63%（相较于HANNA）和8–53%（相较于mod. UNIFAC），并准确预测共沸点、残留曲线，计算成本仅是UNIFAC的约2倍；

**⚠️ 局限性**

局限性包括：尚未扩展到电解液或聚合物体系；对极性水系仍有一定挑战；仅支持SMILES输入；训练时仅用二元数据，三元及以上体系依赖推断；以及对理想蒸气相的假设限制了某些高压/高温情况。

---

## 228. LM-DSB: A Lightweight Multi-agent System for Automated Daily Safety Bulletin Generation in Construction

**arXiv ID:** 2610.36433 | [PDF](https://arxiv.org/pdf/2610.36433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 229. CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning

**arXiv ID:** 2610.36437 | [PDF](https://arxiv.org/pdf/2610.36437v1)

**作者:** Jixuan Leng `[一作]` (Carnegie Mellon University), Inderjit S. Dhillon `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的优势估计器，针对多任务多奖励设置下的门控辅助奖励进行投影，保证辅助奖励与主奖励保持一致。

**💡 创新点**

创新点在于引入订单约束与天花板约束的投影方法，自动修正门控辅助奖励的排序，提供默认-T和自适应-A两种变体。

**🔧 技术方法**

使用了群组相对优势标准化、等距投影（基于Isotonic回归）以及earned trust机制来动态解除天花板限制。

**📊 数据集**

在包含数学、代码、指令跟随和写作四个任务的混合数据集上进行实验，并使用EvalCount、MATH、CodeBench、WritingBench等评测基准。

**📈 对比分析**

将提出的CONCUR与七种现有优势估计器（GRPO、GDPO、GD^2PO等）以及去掉门控辅助的基线进行对比，所有变体在平均得分上均超过GDPO四分以上，且在数学任务上提升显著。

**⚠️ 局限性**

实验仅涵盖四任务混合，门控辅助仅为长度奖励和奖励模型，缺乏对不同奖励设计和更大规模模型的验证，且自适应信任假设可能不适用于所有任务。

---

## 230. JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels

**arXiv ID:** 2610.36434 | [PDF](https://arxiv.org/pdf/2610.36434v1)

**作者:** Yaojian Chen `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了一个仅基于目标的GPU核优化基线JustGoal，证明现代大模型在缺乏专家知识的情况下即可生成高性能核。

**💡 创新点**

创新点在于将专家知识剥离，构建Goal‑Only基线，并用同模型同范围的对照实验量化专用机制的增益。

**🔧 技术方法**

采用Claude Code等通用大模型进行自驱动的评估循环、性能测量和迭代搜索，并结合CUDA/Triton手工优化。

**📊 数据集**

在NVIDIA B200 GPU上使用11种注意力/稀疏/混合专家等算子，涵盖多种形状和工作负载，采用官方FlashAttention/FlashInfer Benchmark标准进行验证。

**📈 对比分析**

将JustGoal生成的核与官方开源库及已发布的专用代理核按同一评测协议进行对比，得到在8/10算子上速度提升1.14–2.34×，在5/9算子上超过专用代理。

**⚠️ 局限性**

局限在仅针对B200 GPU和有限算子，评测覆盖有限，且未完全验证所有输入下的正确性；不同模型与硬件上的迁移性尚待验证。

---

## 231. FORESEE-MS: Toward Prospective Spatial Forecasting of Future Multiple Sclerosis Lesions from Baseline FLAIR MRI

**arXiv ID:** 2610.36428 | [PDF](https://arxiv.org/pdf/2610.36428v1)

**作者:** Maryam Akhavan Aghdam `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了FORESEE-MS评估协议，探讨基线FLAIR MRI是否能预测多发性硬化未来病灶位置；

**💡 创新点**

创新点在于构建折特定的种群先验图谱作为基线，并同时评估像素级AUROC/AUPRC与患者级定位Recall@Top‑K%的双重指标；

**🔧 技术方法**

采用5种深度学习模型（3D U‑Net、Attn+ResBlock+VAE、ViT、SwinUNETR(冻结/微调)）以及基于注册的种群先验图谱；

**📊 数据集**

使用MSSEG‑2数据集中的40个标注训练样本（最终39例）进行5折交叉验证；

**📈 对比分析**

与种群先验图谱相比，所有模型在像素级AUROC/AUPRC上均不优于图谱；在患者级Recall@Top‑K%中，3D U‑Net和Attn+ResBlock+VAE略有提升，但置信区间多包含零，显示差异不显著；

**⚠️ 局限性**

主要限制是样本量小（39例，28例有新病灶），导致统计功效低，且模型未考虑临床或治疗信息，无法充分挖掘患者特异性预测潜力。

---

## 232. Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels

**arXiv ID:** 2610.36520 | [PDF](https://arxiv.org/pdf/2610.36520v1)

**作者:** Utkarsh Kanwat `[一作]` `[通讯]`, Utkarsh Kanwat

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的长文本推理中，提出了一种将Attention Matching（AM）缓存中的偏置（bias）折叠到键（key）上的方法，使得压缩后的KV缓存能够使用标准的无偏置注意力内核。

**💡 创新点**

核心创新在于：每个注意力头仅沿一条经过学习的单一方向移动保留的键，按偏置值比例平移，从而把原本需要额外偏置的 AM 缓存变成普通的 KV 缓存，并给出了输出误差的闭式上界。

**🔧 技术方法**

使用的技术包括：基于上下文查询的最小二乘拟合得到方向向量，键的单向平移与偏置相结合；对折叠后缓存进行值重新拟合；在 vLLM、FlashAttention 等常用框架中直接加载折叠后的 KV 缓存；通过 KL 散度、准确率和解码速度等指标进行评估。

**📊 数据集**

实验数据集包括：QuALITY（包含 534 条多选问答）、LongHealth（398 条医学文档问答）、LongBench（四个英文问答子任务），以及多种 Qwen3 系列模型和 Llama‑3.1‑8B。

**📈 对比分析**

在相同每头缓存预算的设置下，折叠缓存与 AM 在 13/14 个设置中准确率差距在统计噪声内；与去掉偏置的基线相比，KL 散度更低且更少出现 KL>1 的问题；在 vLLM 上部署后，答案与 Hugging Face 完全一致，解码速度仅损失不到 3% 左右，显著提升了缓存可用性。

**⚠️ 局限性**

局限性包括：对极远距离查询的误差评估不足；仅在等预算条件下验证，针对每头不等预算的情况表现不明；对偏置为非静态或查询相关的缓存方法适用性尚未充分验证；在某些模型/预算组合中折叠后仍出现与 AM 的轻微准确率下降。

---

## 233. Do AI Scientists Know What's New?

**arXiv ID:** 2610.36511 | [PDF](https://arxiv.org/pdf/2610.36511v1)

**作者:** Miri Liu `[一作]` (University of Illinois), ChengXiang Zhai `[通讯]` (University of Illinois)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了ClaimNov——一种基于声明层面的可解释新颖性检测模块，并构建了四个时间真实标注的提案数据集，对七个现有检查器进行系统审计。

**💡 创新点**

以声明为最小单位进行新颖性评估，既能显著降低误报，又能识别部分新颖工作，解决了全文检查忽视增量创新的局限。

**🔧 技术方法**

利用LLM（Sonnet 4.6）进行声明抽取、检索、蕴含评分与最终判定，配合稀疏/密集检索、权重聚合等技术实现ClaimNov。

**📊 数据集**

使用1,060个提案的四个数据集：已完成（400）、争议（160）、后续实现（400）和混合（100），每个基于真实论文、审稿争议或后续引用构建。

**📈 对比分析**

在全提案层面，ClaimNov以1.5%误报率捕获97%伪造提案，远优于其他检查器（如Si et al 96%误报）；在声明层面误报率仅3%，正确识别20%先前工作的贡献，并比最强检查器降低约4倍误报。

**⚠️ 局限性**

仅适用于AI、英文研究，固定检索池、LLM训练截止导致潜在偏差；缺乏开放搜索与循环成本评估，泛化与实际部署仍受限。

---

## 234. Spatial-Frequency Learning for Scanning Transmission Electron Microscopy Image Enhancement

**arXiv ID:** 2610.36487 | [PDF](https://arxiv.org/pdf/2610.36487v1)

**作者:** Hesong Li `[一作]` (Beijing Institute of Technology), Tao Zhang `[通讯]` (Hangzhou Dianzi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种空间-频域融合网络（SFFN）用于扫描透射电子显微镜（STEM）图像去噪，并构建了包含多种原子排列、成像模式和真实噪声校准的AtoMix合成数据集。

**💡 创新点**

创新点包括：①在频域引入坐标引导的卷积，利用频率位置信息增强周期性原子信号；②采用可学习的空间-频域融合机制，取代固定相加；③通过对真实STEM图像统计分析实现扫描噪声、点噪声和背景噪声的参数校准；④使用掩码混合多种原子排列生成更具多样性的合成数据。

**🔧 技术方法**

技术手段主要包括FFT/IRFFT转换、坐标引导频域卷积、双分支空间-频域特征提取与卷积融合、基于深度学习的监督训练、以及利用abTEM渲染原子格点与Perlin噪声生成合成图像。

**📊 数据集**

使用的数据集有公开的TEMImageNet、SFIN以及新构建的AtoMix；对AtoMix还包含高角环形暗场（HAADF）与亮场（BF）模式、球差、低亮度、低分辨率等多种降质。

**📈 对比分析**

在合成数据上与10种监督方法、2种无监督方法、以及经典滤波器做对比，采用PSNR/SSIM评估，SFFN在各数据集上平均提升1–2 dB PSNR、0.01–0.02的SSIM，并在原子定位任务中实现最高IoU；在真实材料图像上，SFFN显示更清晰的原子结构且定位误差最小。

**⚠️ 局限性**

局限性在于：仍需依赖合成数据，未覆盖所有极端噪声和成像模式；对4D-STEM等高维数据尚未扩展；网络规模虽不大但训练仍需高端GPU；缺乏自适应注意力或动态卷积等更高级特征融合手段。

---

## 235. Underdamped Anchored Langevin Dynamics

**arXiv ID:** 2610.36470 | [PDF](https://arxiv.org/pdf/2610.36470v1)

**作者:** Qi Feng `[一作]` (Florida State University), Lingjiong Zhu `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并分析了无锚定（anchored）动力学的欠阻尼版本（UALD）及其离散化实现 UALMC，用以在非可微和重尾目标分布上实现无偏采样；

**💡 创新点**

创新点在于将锚定机制与动量（欠阻尼）动力学结合，利用乘法式缩放实现对非可微势能的精确近似，并给出非渐近 2‑Wasserstein 收敛率及更优的维度/精度复杂度；

**🔧 技术方法**

主要技术包括：高维锚定（Gaussian 平滑）势能、乘法缩放的欠阻尼 Langevin 运动、耦合和 Foster–Lyapunov 分析、对多阶矩的统一上界、以及对重尾目标的“taming”处理；

**📊 数据集**

实验数据集涵盖：合成多维拉普拉斯、UCI Breast Cancer 与 Banknote 的带 L1/MCP/SCAD 正则化的逻辑回归、同样数据集上的两层 ReLU 神经网络、以及合成的多维 Student‑t 分布；

**📈 对比分析**

与 KLMC、ULA、ALMC 等传统方法在 2‑Wasserstein 距离、步长容忍度、迭代复杂度等指标上进行对比，实验表明 UALMC 在相同精度下可容忍约 4–7 倍更大的步长，迭代复杂度从 O(d/ε²) 降低到 O(√d/ε)，并且对重尾目标的偏差显著减小；

**⚠️ 局限性**

局限性包括：需要对锚定误差 ω₀ 进行维度可控（通常要求 μ=O(1/d)），锚定因子 a_k 可能在高维下指数级减小导致更新几乎停滞；重尾目标仍需 tamed 版本以保证收敛；若目标势能无闭式平滑表达式，仍需 Monte‑Carlo 近似，可能引入额外方差。

---

## 236. Confident First, Reflective Second: Reinforcement Learning of Confidence-Preserving Reflection for Video Understanding

**arXiv ID:** 2610.36467 | [PDF](https://arxiv.org/pdf/2610.36467v1)

**作者:** Zihao Lin `[一作]` (University of California), Lifu Huang `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对视频多选任务，提出一种保证初始答案可靠、再审后再提高的训练框架

**💡 创新点**

创新点在于将“先自信后反思”原则转化为受约束的强化学习目标，并分别针对奖励侵蚀、参数漂移和梯度干扰设计三种机制：初始答案信用、已验证决策锚定和非对称梯度保护

**🔧 技术方法**

使用强化学习（GRPO式策略梯度）与 KL 正则化，配合自监督的 Qwen2.5-VL-7B 预训练模型进行微调

**📊 数据集**

在 Video‑MME、MLVU 和 LongVideoBench 三个长视频多选基准上进行评测

**📈 对比分析**

与最强 7B 反思基线相比，平均准确率提升 1.7 点（最高 67.0%），且初始答案与最终答案的准确率均提高 4.1 与 3.7 点，表明避免了信心侵蚀

**⚠️ 局限性**

仅在单一回答‑再审循环、单一 7B 模型上验证，未探究多种循环或更大模型的适用性

---

## 237. Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation

**arXiv ID:** 2610.36466 | [PDF](https://arxiv.org/pdf/2610.36466v1)

**作者:** Shufan Li `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了一个统一的离散扩散语言模型（DLLM），通过跨范式的自上政策蒸馏（OPD）从自回归视觉语言模型（AR VLM）和连续流扩散模型中学习，同时引入自我修正的掩码扩散过程以提升视觉生成质量。

**💡 创新点**

①跨范式OPD框架：同时蒸馏来自不同生成范式（AR VLM、连续流扩散）的知识；②自我修正掩码扩散过程：在传统一次跳转的掩码扩散基础上引入多级“token age”状态，实现多步细化和交互；③Elastic MoT架构实现理解与生成的弹性组合。

**🔧 技术方法**

使用Elastic Mixture-of-Transformer（Elastic MoT）两塔架构；基于SBD的块级扩散；自我修正掩码扩散（token age + ball‑tree 采样）；跨范式OPD损失（AR OPD + 一致性损失）和基于教师细化的持续性损失；多任务预训练与OPD阶段训练。

**📊 数据集**

训练集：大规模文本‑图像对、指令跟随、图像编辑、目标定位、图像‑文本混合任务；评估集：MMMU、ChartQA、InfoVQA、AI2D、RefCOCO、ScreenSpot‑Pro、We‑Math、Geometry3k、MathVerse、GenEval、DPG、HPSv3、ImgEdit。

**📈 对比分析**

与同类统一DLLM（LLaDa‑O、MMaDA、NLD‑VLM、LaViDa‑V）、连续流扩散模型（Flux、Qwen‑Image）以及AR VLM（LLaDa‑V、NLD‑VLM）比较。实验显示：在视觉理解、跨模态推理和图像编辑任务上均实现或逼近state‑of‑the‑art；在文本‑图像生成的HPSv3视觉质量上超过Flux、Qwen‑Image；在多项基准上提升数个百分点，显著缩小了DLLM与专用模型的性能差距。

**⚠️ 局限性**

仍存在：①与GPT‑4o或顶级连续流模型相比在某些生成任务上略有差距；②自我修正过程对计算成本有一定开销；③在极大规模数据与更精细的架构调优方面尚未充分探索；④跨范式OPD对教师模型选择和超参敏感，需要进一步稳健性验证。

---

## 238. Split Federated Unlearning

**arXiv ID:** 2610.36465 | [PDF](https://arxiv.org/pdf/2610.36465v1)

**作者:** Jia Chen `[一作]` (Wuhan Textile University), Jun Bai `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了分布式拆分联邦学习（SFL）的数据卸载（unlearning）框架，结合理论分析与算法设计，提出了一种名为SURE的初始卸载方法，并在多种数据集和模型上进行实验评估。

**💡 创新点**

创新点在于：①首次系统化地定义并形式化Split Federated Unlearning（SFU）；②通过历史影响传播和模式依赖可行性分析揭示拆分与联邦聚合耦合对卸载的理论限制；③设计了基于双重要性估计、选择性参数扰动与重要性引导恢复的SURE算法。

**🔧 技术方法**

使用的技术包括：拆分联邦学习架构、梯度平方代理的 Fisher 信息估计、重要性加权的参数扰动、梯度调制恢复、影响函数与Jacobian分析、CKA表示相似度评估以及二阶优化的正定近似。

**📊 数据集**

实验数据集包括 MNIST、CIFAR‑10、CIFAR‑100，使用的模型有 LeNet、CNN、ResNet‑18，并在不同拆分位置和数据分布（IID、极端非IID、Dirichlet）下进行验证。

**📈 对比分析**

与多种改编自传统联邦卸载方法（FedEraser、FedRecovery、SalUn、MoDe、FedOSD、NoT、FedCARE、FedMemPrune）以及模式匹配重新训练（Retrain）进行对比，SURE在保持高保留性能的同时实现了接近重新训练的遗忘效果，平均差距（Avg. Gap）通常低于2个百分点，通信与计算成本在可接受范围内。

**⚠️ 局限性**

局限性包括：未给出严格的遗忘证明；性能受拆分位置、更新模式和模型架构影响较大；需使用保留数据进行恢复；并且对通信与存储成本的完整评估仍有待完善。

---

## 239. SVIn+: Multi-Modal Framework for Robust Underwater State Estimation using 3D Sonar, Visual-Inertial, Water-Pressure, and DVL

**arXiv ID:** 2610.36454 | [PDF](https://arxiv.org/pdf/2610.36454v1)

**作者:** Chinmay Burgul `[一作]` (University of Delaware), Alberto Quattrini Li `[通讯]` (Dartmouth College)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出SVIn+多模态融合框架用于水下状态估计

**💡 创新点**

将3D声纳、相机、IMU、DVL和水压计紧耦合并引入层次式KISS-ICP后端

**🔧 技术方法**

基于因子图滑动窗口非线性优化，融合视觉、惯性、声纳、DVL和深度测量

**📊 数据集**

在巴巴多斯海岸的船坞和珊瑚礁数据集BajanQueen-01/02与Folkestone-01

**📈 对比分析**

与SVIn2、KISS-ICP、Colmap等基线对比，返回原点误差从0.81m降至0.21m，显著降低漂移

**⚠️ 局限性**

仍缺乏全局闭环、对传感器失效的自适应处理，需实现在线标定与循环闭合

---

## 240. Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition

**arXiv ID:** 2610.36460 | [PDF](https://arxiv.org/pdf/2610.36460v1)

**作者:** Hyeontaek Hwang `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在长尾识别中通过控制部分神经崩塌（Controlled Partial Neural Collapse）实现的模型，利用 CAS​A 分类器的节点级门控和 ETF‑AM 损失的频率自适应边距来平衡头部类的完全崩塌与尾部类的内部多样性。

**💡 创新点**

创新点在于将神经崩塌分解为两条轴——维度控制和训练‑推理间隔，并分别通过 CAS​A（特征条件门控）和 ETF‑AM（固定 ETF 与自适应边距）实现部分崩塌，理论证明尾部类不应全崩塌且最优散布随样本数递增。

**🔧 技术方法**

使用技术包括：ResNet/ResNeXt 等 backbone；CASA 的节点级门控网络（soft mask → 硬阈值）；ETF‑AM 损失（固定 ETF 权重 + 频率自适应边距）；正则化项（spike‑and‑slab）和特征范数正则；基于交叉熵的训练；logit 调整等。

**📊 数据集**

实验数据集：CIFAR100‑LT、ImageNet‑LT、Places‑LT、iNaturalist‑2018，覆盖不同规模与长尾度。

**📈 对比分析**

与多种基线（CE、CB、LDAM、BALMS、VS+SAM、GCL、WB、RBL、NC‑DRW‑cRT、DLLA、ETF、BIB、MBIB 等）比较，本文在所有四个数据集上均取得最高或第二高的整体 Top‑1 准确率；在 Medium/​Few 子集表现尤为突出。

**⚠️ 局限性**

局限性包括：当类别数 C> d+1 时无法构造简化 ETF；理论证明仅在有限样本条件下成立，且对 ϕ 的假设要求严格；未验证在检测、分割等其它长尾任务中的泛化。

---

## 241. Martingale-Based Resource Partitioning Bounds for URLLC and eMBB Network Slicing in 5G

**arXiv ID:** 2610.36602 | [PDF](https://arxiv.org/pdf/2610.36602v1)

**作者:** Hanbing Li `[一作]` (Shandong Normal University), Yantong Wang `[通讯]` (Shandong Normal University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于马氏过程的资源块分配框架，以满足5G网络中URLLC与eMBB切片的低延迟与高吞吐量需求。

**💡 创新点**

创新点在于利用到达与服务过程的矩生成函数构造指数马氏过程，并使用Doob最大不等式得到非渐进的上界，从而实现精确的QoS约束下最小资源块分配。

**🔧 技术方法**

主要技术包括马氏过程构造、Doob最大不等式、矩生成函数分析、二分搜索算法与确定性搜索算法。

**📊 数据集**

本文采用仿真数据：10个URLLC用户的ON‑OFF流量、10个eMBB用户的泊松流量，进行Monte‑Carlo仿真验证。

**📈 对比分析**

与传统的有效容量基线（含有限块长度惩罚因子β=0.82）对比，马氏方法在满足1e‑6违约概率时仅需44个RB，低于基线64个RB，减少31.2%资源消耗，并且在罕见事件区间内仍能提供可靠的上界。

**⚠️ 局限性**

局限性包括对到达与服务过程独立同分布的假设、对MGF的准确性要求，以及在极端动态环境下可能需要更复杂的自适应算法。

---

## 242. OpenJaws: An Open Hardware UMI Framework with Stereo Localization for Generalist Manipulation

**arXiv ID:** 2610.36598 | [PDF](https://arxiv.org/pdf/2610.36598v1)

**作者:** Adithyavairavan Murali `[一作]` (Nvidia), Stan Birchfield `[通讯]` (Nvidia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并发布了一个开源的3D打印手持数据采集框架，配合低成本立体相机实现精准定位，并在此基础上实现真实机器人上的策略部署和数字孪生生成。

**💡 创新点**

采用标准 ISO 9409‑1‑50‑4‑M6 插座实现跨机器人装配；用立体相机替代单目+IMU SLAM，首次公开面向 UMI 的定位基准 UMLocBench‑v1；并实现了基于 GPT‑6 Astra 的 agentic real‑to‑sim 场景生成。

**🔧 技术方法**

立体视觉 SLAM 与 SfM（COLMAP、ORB‑SLAM3、cuVSLAM）、深度学习单目定位 ViPE、Isaac Lab 仿真、GPT‑6 Astra 生成数字孪生。

**📊 数据集**

通过手持相机记录多种任务（pick‑and‑place、kitting、长周期非抓取等），配备 AprilTag 标定，构成 UMLocBench‑v1 基准；并使用公开的 KITTI、TUM RGB‑D 等基准进行对比。

**📈 对比分析**

与单目+IMU SLAM、ViPE 等传统方法相比，立体方法在全轨迹覆盖率几乎 100%，全局 ATE 降至 2.5–5.2 cm，速度在 cuVSLAM 的 30 s/序列内实现；在真实机器人上，策略成功率达到 90%。

**⚠️ 局限性**

对相机标定基线敏感，需要额外手工映射初始化；对动态场景或低纹理环境易失效；目前仅支持单手臂，未覆盖双臂或多机器人协作。

---

## 243. Efficient Quantization Training with Stochastic Rounding via Structured Subspace Randomness

**arXiv ID:** 2610.36587 | [PDF](https://arxiv.org/pdf/2610.36587v1)

**作者:** Zeyu Wang `[一作]` (Google), Liang Liu `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了结构化子空间随机舍入（S3R）和结构化误差校正（S3R-EC），在低精度（INT8）大语言模型训练中显著降低随机数生成开销，同时恢复并保持与独立随机舍入相同的训练质量。

**💡 创新点**

创新点在于共享每个 token 的随机阈值以消除 VPU 随机数生成瓶颈，并通过在梯度计算中扣除共享误差（即均值误差校正）来消除由共享导致的正相关误差，从而实现无偏且高效的低精度训练。

**🔧 技术方法**

采用了结构化子空间随机舍入、均值误差校正、JAX/XLA 编译融合、TPU v5p 硬件优化、混合精度 INT8/FP16、矩阵乘法 GEMM、对角缩放以及 AdamW 优化器。

**📊 数据集**

在英文 C4 数据集上训练 12 层 Llama‑style 解码器（32.8M tokens）进行收敛实验，并在 Llama 3.1‑8B、Qwen3‑8B、Gemma 3‑12B 以及 Mixture‑of‑Experts Qwen3‑30B‑A3B、DeepSeek‑V2‑Lite‑16B 等多种大型 LLM 上进行完整训练步测试。

**📈 对比分析**

通过与 BF16、INT8 RTNE、独立 SR 等基线在验证损失和训练步骤时间上对比，S3R-EC 在保持与独立 SR 相同的验证损失的同时，在 TPU v5p 上实现 1.24–1.32× 对 BF16 的速度提升，和 1.11× 对独立 SR 的提升；单块 FFN 计算相对 BF16 亦可达到 1.8× 的加速。

**⚠️ 局限性**

局限性包括：对极小模型或不同硬件平台的适用性尚未完全验证；per‑example 阈值共享下的误差校正效果不如 per‑token；dX‑only 校正在某些阈值分布下效果有限；整体评估主要基于 TPU v5p，跨平台通用性需进一步研究。

---

## 244. Recall or Relearn? A Mixture of In-Context Experts for Recurring Concept Drift

**arXiv ID:** 2610.36585 | [PDF](https://arxiv.org/pdf/2610.36585v1)

**作者:** Zhangkai Wu `[一作]` (University of Sydney), Yi Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种训练无关的Mixture of In-Context Experts（MICE）框架，用冻结的Tabular Foundation Model（TFM）在数据流中通过存储与检索上下文实现对重复概念的回忆或重新学习决策。

**💡 创新点**

创新点包括：①使用两级（快速‑慢速）专家混合权重，快速级根据最近批次的对数损失动态选择上下文；②通过学习曲线面积预测回忆收益并证明其与期望准确率提升等价；③在检测到的概念边界上合并纯粹上下文以避免干扰；④证明慢速级对半Brier折扣误差的上界为 2 ln 2，从而限制跟随记忆的代价。

**🔧 技术方法**

技术手段包括：TabPFN v2 等 Tabular Foundation Model，基于两样本误差率的概念变化检测，交叉拟合估计回忆惩罚，指数权重算法对专家进行加权，使用对数损失和折扣半Brier分数构建快速与慢速级。

**📊 数据集**

实验数据集涵盖：对齐与随机边界的合成 grid 流（Gaussian centroids 5/30/100，块长 500/2000 行），29 条真实流（来自 River、INSECTS、CoverType、PokerHand、Airlines、NOAA Weather、Spam、Phishing 等），以及 9 条独立验证段（Bitcoin Heist、CIC-IDS2017 等）。

**📈 对比分析**

与 FIFO、LTM、Reset、窗口集合等无池基线以及 ARF、SRP、HAT、Learn++.NSE 等在线学习器进行前序预评估；MICE 在 29 条真实流上平均精度最高（≈83.8%），超过 FIFO（≈83.3%），在 grid 流上相较 Reset 提升约 +2.85 点；在未知边界时检测分段能恢复大部分优势。

**⚠️ 局限性**

局限性：需要冻结 TFM 并预估学习曲线；快速级权重对 margin 敏感，需足够区分；上下文池管理导致存储与推理时间增加；概念变化检测误差可能产生非纯粹段，导致回忆收益下降；在概念相似度高的情况下区分能力有限。

---

## 245. Governed Sovereign RAG for Public Administration: Tripartite Epistemic Routing, Facet-Guided Clarification, and Deterministic Numeric Verification in Citizen Services

**arXiv ID:** 2610.36561 | [PDF](https://arxiv.org/pdf/2610.36561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 246. Structured Maps for Landmark-Based Route Instructions with LLMs

**arXiv ID:** 2610.36582 | [PDF](https://arxiv.org/pdf/2610.36582v1)

**作者:** Mifu Matsumoto `[一作]` (University of Hyogo), Hiroaki Ohshima `[通讯]` (University of Hyogo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了如何用结构化的SVG文本输入让LLM选择地图中的地标以生成基于地标的路径指令。

**💡 创新点**

创新点在于将地图信息编码为可直接被LLM读取的SVG文本，而非传统的图像或自然语言描述，并证明其在地标选择任务中优于图像输入。

**🔧 技术方法**

采用了大型语言模型（如GPT‑5.2、Gemma3:27B、Llama‑4等）、LoRA/QLoRA参数高效微调以及基于OpenStreetMap的结构化SVG输出。

**📊 数据集**

构建了基于Hot Pepper餐厅访问信息与OpenStreetMap匹配的两大任务数据集，分别包含约500个转弯实例和339个目标周边实例。

**📈 对比分析**

对比PNG图像和SVG文本两种输入方式的地标选择准确率，并通过微调提升本地LLM的表现；实验显示SVG文本的准确率普遍高于PNG，微调后小模型可达≈50%精度。

**⚠️ 局限性**

局限包括样本规模有限、仅评估精确匹配而忽视多样化人类地标选择、未加入视觉显著性特征以及缺乏真实用户测试。

---

## 247. AudioRSI: Recursively Self-Improving Audio Reasoners

**arXiv ID:** 2610.36578 | [PDF](https://arxiv.org/pdf/2610.36578v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana-Champaign), Ivan Bulyko `[通讯]` (Amazon)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 AudioRSI，一种音频大型语言模型的递归自我改进框架，模型既生成推理路径又作为批判者为自己的推理过程提供过程级奖励；

**💡 创新点**

创新点在于：①模型自我监督——把生成者与评判者统一为同一模型；②周期性同步递归——每隔一定步数将改进后的模型迁移为下一阶段的评判者，从而让奖励本身随模型提升；③利用音频感知，使评判者能够检查推理中对声学事件的引用，拒绝与录音无关的说法；

**🔧 技术方法**

采用GRPO强化学习、pairwise 比较自我批判、周期性 critic 同步、结构化链式思考输出以及对话式评判 prompt 等技术；

**📊 数据集**

使用 AVQA 训练集进行微调，评估基准为四个 OOD 评测集：MMAU test‑mini、MMSU、MMAU‑Pro、MMAR；

**📈 对比分析**

与 Ke‑Omni‑R、Audio Flamingo、Gemini 2.5 Pro 等 RL 基线及多种公开模型对比，结果显示 AudioRSI 在大多数 OOD 基准上领先，尤其在 MMAU test‑mini、MMSU、MMAR、MMAU‑Pro 中在 7B 开源 tier 取得最优或接近最优性能；

**⚠️ 局限性**

局限性：实验仅在单一 backbone（Qwen2.5‑Omni‑7B）和单一训练语料上验证，递归改进在不同模型、数据集上的鲁棒性尚未确认；评判者仍可能继承模型偏差，单个 seed 的实验可能导致结果不够稳健。

---

## 248. From Ripple to Boundary: Deterministic Impact Boundaries for LLM-Based Software Evolution

**arXiv ID:** 2610.36549 | [PDF](https://arxiv.org/pdf/2610.36549v1)

**作者:** Jiameng Bai `[一作]` (Zhejiang University), Sai Wu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了一种基于内容寻址的 Merkle DAG 和两阶段差异算法，来确定大语言模型驱动的设计到代码演进过程中的确切影响边界，随后将该边界注入到 LLM 代理中以实现安全的增量代码修改。

**💡 创新点**

创新点在于将设计资产建模为可哈希的 Merkle DAG，从而使逻辑级的冲击传播可通过哈希比对精确定位，解决传统概率语义检索无法捕捉的结构级级联影响；以及将差异树直接用作 LLM 的上下文注入，消除代理搜索的随机性。

**🔧 技术方法**

采用的技术包括：知识图谱（KG）结构化设计资产，内容寻址的 Merkle DAG，基于哈希的两阶段 diff（水平扫描+垂直递归），可编程的上下文注入器，LLM 代理的约束式代码生成，以及通过 Cypher 进行的交互式 KG 查询。

**📊 数据集**

使用的数据集主要来自企业级支付微服务平台的真实设计资产与代码；实验中对 KG 进行了 3 个增删改 mutation，构建了内部的演化测试集；未来计划构建公开的多层次变更基准。

**📈 对比分析**

对比方法包括：传统的全上下文提示和基于检索的 LLM 自动检索；性能上，差异引擎成功捕获 100% 的多跳因果链，LLM 代理在多文件演进任务中实现零无意修改；具体的量化指标（边界精度、编译通过率等）尚待后续基准评估。

**⚠️ 局限性**

局限性：评估目前仅为定性且基于内部数据，缺乏公开基准与大规模量化指标；方法目前主要针对设计资产，扩展到数据库模式、IaC 等其他结构化 artefacts 的通用性和效率仍需验证；大规模项目中的性能与可伸缩性尚未完整评估。

---

## 249. Frugal Handwritten Text Recognition for Sub-Saharan Civil-Status Registers: A Smartphone-to-Records Pipeline with CPU-Only Inference

**arXiv ID:** 2610.36540 | [PDF](https://arxiv.org/pdf/2610.36540v1)

**作者:** Ousmane Hamadou `[一作]` (University of Ngaoundere), Jean Louis Kedieng Ebongue Fendji `[通讯]` (University of Ngaoundere)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套“低成本”手写民事登记档案数字化管线，利用智能手机 RAW 拍摄、经典预处理和 CPU 端 Kraken 识别，并通过 Camembert NER 进行信息抽取；

**💡 创新点**

将公开的、低资源环境（无 GPU、有限数据）下的手写识别性能提升到可接受水平（CER<1%），并证明残差错误对下游抽取任务影响可量化；

**🔧 技术方法**

使用：Google Pixel 6a 手机 RAW 捕获；OpenCV + 自研算法的无学习预处理；Kraken（CRNN‑CTC）HTR 微调；Camembert‑NER‑with‑dates 进行实体识别；

**📊 数据集**

新建的 1,568 张手写民事登记（出生、婚姻、死亡）图像语料，185 记录行级转写；

**📈 对比分析**

对比基线为未微调的 Kraken（预训练模型）；微调后 CER 由 14.53% 降至 0.50%，WER 由 56.93% 降至 3.37%；在 CPU 上每张图像平均 6 秒；下游角色解析 F1 从 44.5% 提升至 81.2%；

**⚠️ 局限性**

局限性包括：仅来自乍得单一登记处，写作者重叠导致测试集不完全独立；测试集规模小；仅采用手动校正的行分割；未评估自动分割和透视校正；未测量多语种或其他地区的泛化能力；单一注释者，缺乏一致性评估。

---

## 250. ETC-Bench: A Multilingual Benchmark for Evaluation-Side Transcript Canonicalization

**arXiv ID:** 2610.36527 | [PDF](https://arxiv.org/pdf/2610.36527v1)

**作者:** Zhiheng Qian `[一作]`, Kai Yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了评估侧转录规范化（ETC）任务，构建了多语言基准ETC-Bench，包含1098个最小对，并提出SENA评价指标；引入了基于分类学的LLM代理技能TACS；通过实验验证TACS在多语言、181种现象上显著优于传统TN/ITN和Whisper归一化器；

**💡 创新点**

首次将转录规范化视为独立任务，设计可扩展的多语言现象分类学，创建大规模最小对基准，提出兼顾精确匹配和语义保留的SENA指标；

**🔧 技术方法**

使用BERTScore作为语义相似度度量；利用NeMo、FunASR、Whisper等传统归一化工具作为基线；开发TACS技能，引导LLM（GPT、DeepSeek、Qwen）按现象合同执行本地替换；

**📊 数据集**

ETC-Bench：12种语言（阿拉伯语、孟加拉语、德语、英语、波斯语、法语、印地语、印尼语、日语、俄语、越南语、汉语），包含181种规范化现象；

**📈 对比分析**

采用SENA评估，要求对的两条文本在独立归一化后完全一致且语义相似度不低于原始差距0.1；结果显示TACS-Qwen平均SENA 68.8%，TACS-DeepSeek 43.8%，TACS-GPT 43.3%，明显高于传统基线（≤9.4%）和Whisper（≤40.9%）；

**⚠️ 局限性**

TACS在符号/标记和同音替代等类别仍表现不足；对语言特定规则的覆盖有限；语义保留阈值对结果敏感；目前仅针对评估转录，未覆盖翻译文本等任务。

---

## 251. Benchmarking Task Takeover Across Agent Handoffs

**arXiv ID:** 2610.36523 | [PDF](https://arxiv.org/pdf/2610.36523v1)

**作者:** Wenyuan Li `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 HSC-Bench benchmark，用于评估代理模型在任务交接时的证据恢复与任务继续表现。

**💡 创新点**

提出双轨评估框架，区分仓库证据检索与状态依赖继续，并引入结构化交接状态提升任务接管效果。

**🔧 技术方法**

利用 GPT‑4.1 生成结构化状态和叙事摘要，采用多种 LLM 接收模型（GPT‑5.4、Claude Sonnet 5 等）进行评测。

**📊 数据集**

使用 300 个代码库任务（ContextBench 代码定位注释）和 739 个合成对话/工具/文件工作任务，合计 1,039 个交接案例。

**📈 对比分析**

对比同一来源历史下的不同交接信息（最近转录、结构化状态、完整历史等），通过 line‑F1、平均分与全通过率衡量，结构化状态在所有模型中均显著提升证据检索，且不同模型在继续任务上表现差异显著。

**⚠️ 局限性**

局限在于单次交接、任务为合成或已标注的代码库，未考虑多代理连续交接和真实生产环境的多样性。

---

## 252. Learning from Points and Patches: A Pre-trained Hybrid Attention Transformer for Time Series Analysis

**arXiv ID:** 2610.36572 | [PDF](https://arxiv.org/pdf/2610.36572v1)

**作者:** Peiwang Tang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 TimeHAT，一种通用的时间序列预训练框架，结合统一的点-补丁掩码策略和混合注意力编码器。

**💡 创新点**

创新点在于（1）统一点-补丁掩码（UPPMS）同时学习细粒度点级和粗粒度补丁级信息；（2）混合注意力（HAE）在浅层使用前缀约束的双向注意力，深层使用双向+反向注意力，既保持全局上下文，又防止深层特征退化。

**🔧 技术方法**

技术手段包括：掩码式自监督预训练、前缀约束双向注意力、反向/因果注意力混合、RoPE 位置编码、轻量级 MLP 解码器，以及多头 Transformer 编码器。

**📊 数据集**

使用 12G 大规模多域时间序列语料库 UTSD-12G 进行预训练；下游评测涵盖六个长期预测基准（ETTh1/2、ETTm1/2、Weather、Electricity）、七个缺失插补基准、五个异常检测基准（SMD、MSL、SMAP、SWaT、PSM）以及十个 UEA 多变量分类数据集。

**📈 对比分析**

与多类基线（TimesNet、TimeMixer++、Autoformer、GPT4TS 等）以及最近的预训练模型（MOMENT、GPT4TS、TIMESBERT）在 fine‑tune、few‑shot 与 zero‑shot 场景下进行对比，结果显示 TimeHAT 在大部分任务上均取得最低 MSE/MAE 或最高 F1、分类准确率，尤其在长周期预测、插补和零样本迁移中表现优异。

**⚠️ 局限性**

局限性包括：① 仅在单变量通道上独立编码，跨变量协同信息利用有限；② 预训练需要大规模高质量数据，资源投入高；③ 对极端噪声或非平稳序列的鲁棒性尚待进一步验证。

---

## 253. A Simple Presentation of Robust Membership Inference Attacks

**arXiv ID:** 2610.36458 | [PDF](https://arxiv.org/pdf/2610.36458v1)

**作者:** Renaud Gaucher `[一作]` (École Polytechnique), Lydia Zakynthinou `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在高维高噪声环境下的鲁棒成员推断攻击，给出了针对高斯均值估计和二元分布的攻击框架，并证明了在任何足够精确的算法上都能以可控误报率成功识别成员；同时给出了对应的差分隐私下限，说明若算法达到该精度则必不满足差分隐私；文章还对相关文献进行了综述。

**💡 创新点**

创新点在于：①将贝叶斯后验分布分析与鲁棒攻击相结合，给出一个通用的内积测试框架；②在不依赖具体算法的前提下，仅利用误差界即可构造攻击；③提供了从鲁棒攻击推导差分隐私下限的直接方法，统一了多种先前的结果；④将上述思想推广到二元数据与高斯数据，并给出相应的精度与维度关系。

**🔧 技术方法**

主要使用技术包括：贝叶斯推断（高斯/贝塔后验）、向量截断（clipping）、高斯与Hoeffding尾部概率估计、Paley‑Zygmund不等式、矩估计、差分隐私的replace‑one与相似性论证、以及矩阵范数与协方差估计等。

**📊 数据集**

本文为理论研究，实验中主要使用模拟数据：随机生成的高斯样本和伯努利样本；并未在真实公开数据集上进行实证。

**📈 对比分析**

对比方法：在维度满足 d≈n+α²n² 时，攻击的误报率可设为 β，成功率至少为 Θ(1/α²)；与已有差分隐私下限（α≈√d/n）相比，两者在维度/误差关系上保持一致。

**⚠️ 局限性**

局限性：①假设攻击者已知数据生成分布和先验，实际应用中可能不成立；②仅针对独立同分布、产品结构的数据（高斯/二元），对更一般的分布推广有限；③对维度较低或误差较大时，攻击效果减弱；④在实践中还需结合模型特性，单纯的理论攻击可能需要进一步改进以适用于复杂机器学习模型。

---

## 254. Decoupling Consistency and Conflict across MLLM Layers for Fake News Detection

**arXiv ID:** 2610.36569 | [PDF](https://arxiv.org/pdf/2610.36569v1)

**作者:** Kai Qian `[一作]` (Soochow University), Jiajie Xu `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出多层一致性与冲突解耦框架 MLCCD，用于多模态假新闻检测。

**💡 创新点**

创新点在于利用多模态大语言模型各层的知识分辨一致性与冲突，并通过任务相关 token 筛选、层间表征变化筛选以及两个自适应路由器实现跨层冲突稀疏感知和样本级别自适应聚合。

**🔧 技术方法**

采用 LoRA 微调的 Qwen3‑VL‑8B 大模型作为特征提取器，结合 token 相关性投影、层变化得分、Transformer 层上下文化、两条一致性/冲突子流、路由器和解耦正则化等技术。

**📊 数据集**

在三大多模态假新闻基准 Weibo、Pheme、AMG 上进行实验。

**📈 对比分析**

与传统多模态假新闻检测器、通用多模态方法以及直接 LoRA 微调的 MLLM 基线相比，MLCCD 在所有数据集上实现了最高准确率和宏 F1，分别提升约 0.9–1.6% 及 0.9–2.3% 点。

**⚠️ 局限性**

主要局限是对已有高性能基线（如 Weibo）的提升有限；方法依赖大型 MLLM 并需额外训练步骤，且未针对非中英双语或更复杂视觉场景进行验证。

---

## 255. Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors

**arXiv ID:** 2610.36544 | [PDF](https://arxiv.org/pdf/2610.36544v1)

**作者:** Siddharth Jain `[一作]` (Intuit), Venkat N. Vedam `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 AI 代理的失败监控与重启决策进行了理论与实验评估，提出基于决策价值的评估框架和置信区间审计。

**💡 创新点**

引入决策充分性标准与有限样本决策证书，证明准确率并不能衡量重启效益，并用分析控制展示恢复机会对成功率的影响。

**🔧 技术方法**

采用决策理论中的 Blackwell 比较、Exact alarm‑to‑action identity、同时二项区间与 Bootstrap 置信区间，结合基于 SymPy 的 SWE‑bench Verified 基准进行实验。

**📊 数据集**

19 个 SymPy 代码缺陷任务的外部验证集（SWE‑bench Verified），共 194 个模型执行轨迹与 19 个未验证的合成轨迹。

**📈 对比分析**

通过固定切点、前瞻转移、延长时域、上下文恢复、分支重用等实验，对比固定重启与延长时域的成功率差异，结果显示初始 16 调用的提升在后续实验未复现，重启效果高度依赖模型与任务。

**⚠️ 局限性**

样本量小（每任务仅 5 次）、仅覆盖 SymPy 任务、未评估学习型监视器、假设独立提供者抽样、外部验证仅覆盖有限功能、未衡量监控成本和实际部署环境。

---

## 256. Agentic Data Sampling Framework for Multilingual ASR via Linguistic-Aware Error-Driven Optimization

**arXiv ID:** 2610.36538 | [PDF](https://arxiv.org/pdf/2610.36538v1)

**作者:** Zhiheng Qian `[一作]` (Shanghai Jiao Tong University), Kai Yu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种语言感知的代理式数据采样框架，用以在多语种 ASR 训练过程中动态调整域、样本和 token 层面的数据选择。

**💡 创新点**

创新点在于将多语种 ASR 评估（基于音系、形态和句法的错误分析）嵌入训练循环，利用 LLM 代理根据错误诊断生成可执行的采样计划，实现自适应、可解释的数据采样。

**🔧 技术方法**

核心技术包括：基于 LLM 的代理决策、分层算法池（DoReMi、DOGE、LESS、RHO‑1 等）、语言维度错误仪表盘、音素、形态和句法错误统计及其自然语言摘要。

**📊 数据集**

在12种 Common Voice 语言子集（be、zh、nl、fr、de、it、ja、pl、pt、ru、es、tr）上进行实验，总训练时长 1,200 小时。

**📈 对比分析**

与无采样和随机采样基线对比，代理采样（300h）平均 WER/CER 下降至 19.4（基线 23.9，随机 22.5），在 9/12 语言上获得最佳表现，且训练收敛更快。

**⚠️ 局限性**

局限性包括：依赖手工设计的错误仪表盘、实验范围仅限于 12 种语言、对极低资源语言的有效性尚待验证、以及对 LLM 代理决策的解释性与可复现性有待进一步提升。

---

## 257. A generalist policy for muscle-driven control across individual musculoskeletal models

**arXiv ID:** 2610.36535 | [PDF](https://arxiv.org/pdf/2610.36535v1)

**作者:** Bozhao Wang `[一作]` (Tsinghua University), Xiaoqiang Tang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究提出了一种名为MyoAcross的框架，能够在不进行个体微调的情况下，实现对不同个体肌肉驱动模型的参考运动跟踪。

**💡 创新点**

创新点在于结合个体模型构建、形态条件化策略学习与虚拟模型增强，实现单一通用策略跨个体运动控制，并通过形态编码和FiLM实现动态适配。

**🔧 技术方法**

采用SMPL与SKEL进行区域缩放构建模型，利用PPO训练FiLM-条件化演员-评论家网络，并使用虚拟模型扩增和对称性增强。

**📊 数据集**

使用了公开数据集AMASS（141名受试者）、Gait120（120名受试者）以及MoVi（88名单图像受试者）的运动捕捉和形态数据。

**📈 对比分析**

通过与未条件化MLP、Concat等基线在31个保留模型上对比，MyoAcross完成率达99.24%，跟踪误差比基线降低约22%，在Gait120中与真实GRF和EMG波形的相关性分别为0.853和0.406，显示出较好的生物力学一致性。

**⚠️ 局限性**

局限在于模型共用拓扑、仅适配形态差异、未涵盖肌肉功能与神经控制的个体差异，以及单图像构建的解剖准确性尚未完全验证。

---

## 258. Optimality and Impossibility in Multidimensional Strategyproof Facility Location

**arXiv ID:** 2610.36613 | [PDF](https://arxiv.org/pdf/2610.36613v1)

**作者:** Jianan Lin `[一作]` `[通讯]` (Rensselaer Polytechnic Institute), Jianan Lin (Rensselaer Polytechnic Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在无支付的多维设施定位问题中，首先证明了在欧几里得平面上单设施的坐标均值（Coordinate‑wise Median，CM）即使去掉匿名性仍是最优的，并给出了对应的最优逼近比；其次，在ℓ∞空间中提出一种lexicographic最小化社会成本的SP机制，达到逼近比1；随后证明在1<p<∞的ℓp平面上任意两设施的确定性SP机制都无法获得有限逼近比，并将这一不可能性推广到任意k>2设施（仅需保持与某一固定对称对的可交换性），且保留了逼近与SP的性质；最后给出了上述结果与已知结果的比较与数值表述。

**💡 创新点**

创新点在于：1) 去掉了欧几里得平面单设施最优性证明中的匿名性假设；2) 在ℓ∞空间提出全局最优且SP的lexicographic机制，克服了CM的2倍逼近限制；3) 在两设施情形中仅凭交换对称性即可得到不可能性结果，突破了之前对匿名性的依赖；4) 通过“远距离代理”归约将两设施不可能性推广到任意k>2设施，且保持SP与逼近比不变。

**🔧 技术方法**

主要技术包括：连续性与投影理论、局部菜单与星形化性质、对偶映射与梯度不等式、凸分析与几何约束、极限子网（Tychonoff）收敛论证、分离实例与远距离代理归约、对称性与交换不变性分析。

**📊 数据集**

本文完全基于理论分析，无需实验或数据集。

**📈 对比分析**

通过与已知的CM逼近比（如√(2(n²+1))/(n+1)等）以及在ℓ∞平面上CM的2倍逼近做对比，证明所给机制在对应空间中实现了最优逼近比；在多设施情形下给出无限逼近（无有限上界）的不可能性结论，表明任何确定性SP机制在这些场景下无法做到与最优相当。

**⚠️ 局限性**

局限性包括：1) 对p=1与p=∞多设施情况仍未得到结论；2) 高维下不需要完全匿名性而仅需某一固定对称对的可交换性，是否可以进一步放宽仍未知；3) 只讨论确定性SP机制，随机化机制是否可突破尚未探讨；4) 对线性空间k>2的匿名性问题仍存在开放性。

---

## 259. WithinVAE: Compact Visual Autoencoding with Self-Distilled Semantics

**arXiv ID:** 2610.36593 | [PDF](https://arxiv.org/pdf/2610.36593v1)

**作者:** Yuzhou Ji `[一作]` (Shanghai Jiao Tong University), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种紧凑的视觉自编码器，利用自身EMA副本进行自蒸馏学习语义特征，无需外部预训练识别器；

**💡 创新点**

创新点在于将全局与局部自蒸馏目标与位置相关的令牌掩码结合，形成可前缀解码的有序语义序列；

**🔧 技术方法**

采用变分瓶颈、ViT-L编码器/解码器、EMA教师-学生架构，以及DINO/iBOT式的全局与局部蒸馏损失；

**📊 数据集**

在ImageNet‑1K 256×256图像上进行训练与评估；

**📈 对比分析**

与RPiAE及其他基线对比，半量标量容量即可同时提升rFID、PSNR、LPIPS、SSIM，且128令牌前缀在PSNR/SSIM上超越完整序列；

**⚠️ 局限性**

局限包括：未显著提升生成性能、容量以连续标量计数而非比特，语义证明仅为全局分类，且仍需研究是否能进一步压缩或实现变量率编码。

---

## 260. Follow the Herd: A Data-Driven Approach for Actionable Recourse

**arXiv ID:** 2610.36580 | [PDF](https://arxiv.org/pdf/2610.36580v1)

**作者:** Nishtha Madaan `[一作]` (Indian Institute of Technology Delhi), Srikanta Bedathur `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于数据驱动的行动递归框架 DDAR，利用历史用户轨迹的扩散模型预测下一步状态，从而在不需要人工定义成本的前提下给出可行且低成本的递归方案。

**💡 创新点**

首次在无监督条件下通过学习最可能轨迹来估计行动成本，将高概率轨迹视为低成本路径；同时支持多步递归和多路径采样。

**🔧 技术方法**

使用条件扩散模型（DDPM）与导向扩散（guided diffusion）生成轨迹，Transformer 作为去噪器，交叉熵损失和梯度引导实现决策目标。

**📊 数据集**

三组自建数据集：LoanSynth（贷款合成轨迹）、GradSynth（研究生申请合成轨迹）和 AdultIncome++（半实用收入轨迹）。

**📈 对比分析**

与 Wachter、DiCE、FACE、SCD 等基线在 Recourse Rate、Cost、CNLL/UNLL 等指标上对比，DDAR 在 RR‑AUC、平均成本、可行性指标上显著优于基线，并通过多步/多路径采样进一步提升递归率。

**⚠️ 局限性**

实验仅在合成/半实数据上验证，缺乏对真实复杂场景的评估；训练需求高；对高维连续特征的可解释性与可操作性仍待进一步探讨。

---

## 261. Less is more: learning what to profile for GPU kernel optimization

**arXiv ID:** 2610.36576 | [PDF](https://arxiv.org/pdf/2610.36576v1)

**作者:** Yihan Wang `[一作]` (Renmin University of China), Wei Xu `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 AutoProfiler，一种基于反馈的树结构框架，用来学习针对不同 GPU kernel 的自适应硬件指标采样策略，避免固定的静态指标收集；

**💡 创新点**

创新点在于把硬件指标采样视为可学习的顺序决策过程，并通过 LLM 提供的 Rewrite 与 Split 操作迭代构建确定性指标树，从而实现 kernel‑conditioned 的精确 profiling；

**🔧 技术方法**

技术包括：顺序指标采样的决策过程建模、确定性树结构表示、LLM 生成候选树修改、经验式接受准则、以及基于配额控制的树扩展；

**📊 数据集**

使用 CUDA‑Agent‑Ops‑6K 训练树结构，评估则采用广泛使用的 KernelBench（包含 250 个不同级别的 GPU kernel）进行跨 LLM 后端验证；

**📈 对比分析**

在 KernelBench 上，AutoProfiler 的平均几何均值加速比为 1.704×，准确率 87.2%，显著高于最强基线 CudaForge（1.570×、78.4%）且仅需一次 LLM 调用；

**⚠️ 局限性**

局限性包括：仅在 NVIDIA RTX 4090D 环境下验证，树构建过程仍依赖 LLM 的生成质量，且在极大指标空间时可能面临组合爆炸或过拟合风险。

---

## 262. CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents

**arXiv ID:** 2610.36570 | [PDF](https://arxiv.org/pdf/2610.36570v1)

**作者:** Yanyan Zhang `[一作]` (Carnegie Mellon University), Hao Zhang `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 CASEL 的运行时层，负责在语言模型代理工具调用前根据校准的失败概率、后果严重度、可逆性、歧义度和证据新鲜度等风险信号，选择六种介入模式（直接执行、验证、沙箱、审批、澄清、放弃）以最小化每步成本和潜在危害。

**💡 创新点**

创新点在于将后果加权的风险与可逆性信息结合起来构建多模态介入决策，而非仅用不确定性阈值；同时提出了可验证的有限样本阈值条件和轨迹风险账本的理论框架，并在统一成本下对比多种门控策略。

**🔧 技术方法**

技术包括：校准失败概率（等距回归/温度缩放）、后果估计（基于工具元数据和范围因子）、可逆性分类、歧义/新鲜度检测、基于风险信号的控制器（最小化成本函数J(m)）、有限样本阈值与轨迹账本分析。

**📊 数据集**

在仿真环境中使用40,000条校准轨迹和6,000条测试轨迹（共120,000次动作），动作分为只读、可逆和不可逆三类，分别对应不同的后果分布。

**📈 对比分析**

与多种基线（基础、始终验证、仅不确定性门、仅后果门、组合门、风险门、预算门、无校准等）在统一计算成本（0.486）下进行比较。结果显示，CASEL 将严重副作用率从14.75%降低至7.02%，平均归一化危害从0.184降至0.123，且在相同成本下仅略低于基线成功率，且比单纯风险门批准比例更低。

**⚠️ 局限性**

局限性包括：仅在仿真中验证，未在真实部署中测量干预效果；轨迹账本理论在实践中不成立；对工具可逆性和干预效果的假设需在不同领域重新校准；未与其他层面（如REFLEX、CA2I）直接比较。

---

## 263. CCP-DP: Cross-cluster context-guided phase dynamic programming for DNA storage sequence reconstruction

**arXiv ID:** 2610.36559 | [PDF](https://arxiv.org/pdf/2610.36559v1)

**作者:** Zixin Zhang `[一作]` (Tianjin University), Huaming Wu `[通讯]` (Tianjin University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于跨簇上下文引导的相位动态规划方法（CCP‑DP），用于在低覆盖率和高IDS噪声下重建 DNA 数据存储序列。

**💡 创新点**

通过利用同一数据集中不同读簇之间的共享序列上下文和盲 IDS 通道统计，结合自适应选择性相位动态规划，显著提升了低覆盖条件下的精确重建率。

**🔧 技术方法**

使用跨簇上下文概率模型、盲 IDS 估计、动态规划相位推断以及局部一致性筛选和结构化边界校正。

**📊 数据集**

在四个公开 DNA 存储测序数据集（Erlich‑152、Srinivas‑110、Chandak‑108、Chandak‑150）以及合成覆盖率/IDS 条件上进行评估。

**📈 对比分析**

与 MUSCLE、ITR、CPL、BBS、TrellisBMA、DNAFormer、TReconLM 等基线对比，CCP‑DP 在所有数据集上实现了 95.36%–99.98% 的精确重建率，低覆盖率下提升 6.2–17.5 个百分点，并保持低编辑距离与较低计算开销。

**⚠️ 局限性**

其优势受跨簇可重用上下文强度和需要结构化解码的簇比例限制，在极端 10% IDS 或跨簇模式弱时性能下降。

---

## 264. Holomorphic Cauchy Networks for Efficient Stationary Tail Computation in High-Dimensional Reflected Brownian Motions

**arXiv ID:** 2610.36558 | [PDF](https://arxiv.org/pdf/2610.36558v1)

**作者:** Kehan Wang `[一作]` `[通讯]` (Chinese University of Hong Kong), Kehan Wang (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 Holomorphic Cauchy Networks（HCN）来高效计算高维反射布朗运动（RBM）的稳定尾概率，采用 BAR 学习框架和共享坐标表示

**💡 创新点**

创新点在于将有理 Cauchy 激活嵌入本地复值子网络，使网络在极点自由域内自发满足 Cauchy–Riemann 方程，消除 CR 惩罚；通过原点锚点和边界坐标排除实现结构化约束；加入锚定交互子网络以捕捉坐标耦合，显著降低 GPU 内存和更新时间

**🔧 技术方法**

使用复值深度网络、Cauchy 激活、BAR 学习、固定 Talbot 逆变换、闭式输入导数、按维度调度批量、分支顺序反向传播等技术

**📊 数据集**

在合成 RBM 基准上进行实验：2、20、30、50 维产品型（product-form）模型；2 维 Harrison 非产品模型；10 维相关模型，全部从模型参数直接训练，无真实样本生成

**📈 对比分析**

与原 NN4MGF 进行比较，HCN 在 30 维时平均更新时间降至 1/9.7，峰值显存降至 1/10；尾概率均相对误差显著下降（如 2 维 Harrison 从 6.68% 降至 1.42%，50 维在 1:8 误差为 6.37%），在多维度下保持较低误差并实现单 GPU 可行性

**⚠️ 局限性**

局限性包括：理论近似仅保证有限宽度下存在极点自由近似，训练收敛性未保证；数值逆变换误差可能掩盖网络误差；对非产品高维 RBM 缺乏严格的 Monte‑Carlo 参考；对复极点位置的学习与逆变换路径的匹配仍需进一步理论与实验验证

---

## 265. QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG

**arXiv ID:** 2610.36557 | [PDF](https://arxiv.org/pdf/2610.36557v1)

**作者:** Yifan Zhao `[一作]` (Alibaba Group), Yuming Deng `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了QESChunker方法，将文档分块建模为监督式、预算约束的优化问题。

**💡 创新点**

首次利用问答-证据对训练的分块边界，构造统一的覆盖+结构化目标，分别在非重叠与重叠约束下提供精确DP和近似贪婪解。

**🔧 技术方法**

采用权重覆盖、子模优化、动态规划、成本-收益贪婪、子模函数证明与NP难度归约等技术。

**📊 数据集**

在两份自建中文政策报告、QASPER、PeerQA、PrivacyQA三份公开英语问答语料上进行评测。

**📈 对比分析**

与固定长度、自然段、TextTiling、ClusterSemanticChunker、LumberChunker、句窗、Parent-Document、Auto-Merging等基线在相同检索器和上下文预算下比较，QESChunker在大多数设定下提升完整证据召回率，尤其在紧凑预算下显著优于基线。

**⚠️ 局限性**

依赖丰富的问答-证据监督，冷启动或稀疏注释文档效果有限，且提升的检索召回并不一定直接转化为生成质量。

---

## 266. A Trainable CPG Framework for Quadruped Locomotion

**arXiv ID:** 2610.36537 | [PDF](https://arxiv.org/pdf/2610.36537v1)

**作者:** Xiyan Liu `[一作]` (Zhejiang University), Shaoxing Qu `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可训练的四足行走框架CPG–ENN，结合显式中央模式生成器(CPG)与学习的执行网络(ENN)，实现多步态同步控制与速度指令跟踪；

**💡 创新点**

将CPG生成的节律与共享的执行网络分离，使用单一反馈策略调节六个可解释的运动参数，实现跨步态平滑过渡，无需为每种步态单独训练；

**🔧 技术方法**

使用基于八神经元的CPG、共享的多层感知机ENN、强化学习(PPO)与Beta分布动作参数化、离线IK监督、Isaac Gym仿真与Unitree Go2硬件实现；

**📊 数据集**

主要使用仿真中的Unitree Go2机器人与离线IK生成的运动轨迹进行训练与评估，未使用公开的标准数据集；

**📈 对比分析**

与AllGaits和Walk These Ways进行对比，在20个步态对与速度对的跨步态过渡基准中，CPG–ENN成功率99.39%，平均速度误差0.133 m/s，显著优于AllGaits（91.41%，0.259 m/s）和WTW（95.63%，0.282 m/s）；

**⚠️ 局限性**

局限性包括缺乏地形反馈、无自适应步态选择、对域随机化依赖较强、未在极端速度或复杂地形上验证。

---

## 267. RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

**arXiv ID:** 2610.36533 | [PDF](https://arxiv.org/pdf/2610.36533v1)

**作者:** Buzhao Liu `[一作]` (Washington University in St. Louis), Chongjie Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为 RoboAuto 的框架，用来自动搜索、验证和改进机器人程序，强调多层次的物理理解、场景对齐、分阶段验证以及自主实验设计。

**💡 创新点**

创新点在于：①将物理知识分层管理为对象中心的可转移知识与可执行技能；②通过视觉感知对任务相关对象和关键量进行即时对齐；③采用分阶段规划并在每一步进行实证验证；④引入自设计实验以主动排除物理假设不确定性。

**🔧 技术方法**

核心技术包括：大型语言模型（Claude Opus 4.6）作为编码代理；Molmo2、SAM 等视觉模型用于物体识别与分割；深度相机与 Contact‑GraspNet 用于抓取候选与几何测量；层次化知识库管理；多阶段计划与独立审计器；实验生成脚本与自检机制。

**📊 数据集**

使用 LIBERO‑90 的演示视频构建初始知识库；评估数据集涵盖 LIBERO‑PRO、LIBERO‑PRO Long、LIBERO‑Plus、Robosuite、RoboTwin 2.0 以及 WidowX 实际机器人上的三项长期精细操作任务。

**📈 对比分析**

与 CaP‑Agent0、ASPIRE、RHO 及 π_0.5 等基线进行对比。RoboAuto 在短期任务中达 89.1% 成功率，在长程任务中突破 73.4%，在 Robosuite 任务中 90.6%，在 RoboTwin 2.0 任务中 91.5%，显著高于所有对比方法，展示出知识迁移与实验验证带来的性能提升。

**⚠️ 局限性**

目前仅在仿真环境中实现自动搜索；真实机器人实验仍需人工重置场景并进行安全监控，缺乏完整的自主部署闭环。

---

## 268. A Fine-Grained Empirical Study of Upstream Security Events and Downstream Mitigation in Maven

**arXiv ID:** 2610.36603 | [PDF](https://arxiv.org/pdf/2610.36603v1)

**作者:** Kazi Amit Hasan `[一作]` (Queen's University), Steven H. H. Ding `[通讯]` (McGill University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过跟踪832个Maven生态系统CVE从上游修复到下游项目采用的全过程，构建了完整的漏洞生命周期和依赖链暴露与缓解框架。

**💡 创新点**

创新点在于将上游报告、修复、发布和NVD披露四个事件与下游采用时序联结起来，揭示曝光窗口、采用延迟以及依赖自动化（特别是Dependabot版本更新）对缓解时效的影响。

**🔧 技术方法**

使用的技术包括GitHub API数据爬取、CVE–commit映射、事件时间序列分析、事件模式分类、Mann-Whitney U检验以及统计可视化。

**📊 数据集**

数据集来源于先前的Maven依赖分析数据，包含832个CVE、257个上游仓库、44,450个(CVE,上游,下游)暴露三元组及1,677个观察到的下游采用记录。

**📈 对比分析**

通过对采用延迟与事件模式、依赖自动化状态进行比较，发现Dependabot版本更新可将发布到采用的中位延迟从110.6天缩短至10.4天；透明/静默修复对NVD披露前可用性无显著差异。

**⚠️ 局限性**

局限性包括仅考虑公开的CVE、未捕捉私有或邮件列表沟通的修复、仅观察到的1,677个采用案例、Dependabot以PR和配置文件估算可能漏检，以及对非Maven或私有生态系统的泛化不足。

---

## 269. Camera-to-Quadruped Robot Pose Estimation from Sequential Images

**arXiv ID:** 2610.36599 | [PDF](https://arxiv.org/pdf/2610.36599v1)

**作者:** Seobin Hwang `[一作]` (Chonnam National University), Yeong-Jun Cho `[通讯]` (Chonnam National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于外部单目相机视频的全身 18-DoF 四足机器人姿态估计框架，并首次发布了对应的 benchmark 数据集。

**💡 创新点**

核心创新包括：① 关键点选择器（selector token）提升腿部关键点定位精度；② 骨架感知的遮罩注意力（skeleton-aware masked attention）解决腿部对称与自遮挡导致的误匹配；③ 结合前向运动学（FK）与后置 refinement 头实现高精度 3D 关节与基座姿态恢复；④ 通过冻结 JointNet 与时间卷积融合多帧信息，有效消除单帧深度不确定性。

**🔧 技术方法**

采用 Transformer‑Encoder + ResNet 基础特征提取，SimCC 1D 热力图回归关键点，Selector Token 交叉注意力，骨架遮罩自注意力，前向运动学推算 3D 关键点，时间卷积 + 加权池化估计基座 6-DoF，MLP 细化残差。

**📊 数据集**

主要使用 SpotSim——由 Boston Dynamics Spot 在 52 个真实环境纹理背景下的 51,697 张合成 RGB 图像（1280×720）生成的全身 2D/3D 关键点、基座姿态与关节角度标注。

**📈 对比分析**

在 SpotSim 基准上，提出的方法在 PCK@2.5 px、PCK-AUC、ADD@20 cm、ADD-AUC 上分别达到 92.18%、95.54%、85.94%、66.98%，平均关节角误差 3.65°，相较于最近基线（如 SGTAPose、RoboPEPP）提升 5.6%–6.6% 的 PCK 与 ADD，并在 33 FPS 运行速度下保持实时性能。

**⚠️ 局限性**

局限性主要包括：仅在单一机器人平台（Spot）和合成数据上训练，缺乏对多平台或更复杂动作的验证；对真实世界域间迁移虽有一定表现，但仍未进行显式域自适应；对极端遮挡或多机器人场景的鲁棒性待进一步提升。

---

## 270. MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke

**arXiv ID:** 2610.36583 | [PDF](https://arxiv.org/pdf/2610.36583v1)

**作者:** Md Wahiduzzaman Suva `[一作]` (American International University-Bangladesh), Jia Uddin `[通讯]` (Woosong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在缺血性卒中后患者的多通道多模态睡眠监测中，提出一种可一次性完成睡眠分期与呼吸事件检测（并估计呼吸暂停指数）的多模态多任务网络MM‑Net。

**💡 创新点**

① 对预训练EEG编码器进行睡眠适配后，通过交叉拟合与岭回归对齐，解决了训练时与推理时表示空间不一致的问题；② 采用交叉注意力融合神经与心肺信号的两流架构，并使用共享的双向LSTM进行时序建模；③ 在同一框架内实现睡眠分期、呼吸事件预测与APHI估计，显著提升两项任务的性能。

**🔧 技术方法**

使用预训练LaBraM EEG编码器（经睡眠适配），Ridge对齐、交叉注意力融合、双向LSTM时序解码、Viterbi后处理、加权交叉熵与正样本加权等技术。

**📊 数据集**

iSLEEPS（99名卒中患者，含睡眠分期与呼吸事件注释）作为主要训练/评估集；Sleep‑EDF（健康睡眠），ISRUC‑Sleep（分为I、II、III三组，含呼吸事件注释）和UCD（疑似睡眠呼吸暂停）作为外部验证集。

**📈 对比分析**

与一系列重实现的单任务深度模型（DeepSleepNet、U‑Time、TinySleepNet、SleepTransformer 等）以及呼吸事件检测基线（降解规则、逻辑回归、梯度提升、原始信号 CNN）进行比较。MM‑Net 在 iSLEEPS 上达 86.4% 的睡眠分期准确率（macro‑F1 0.756，κ 0.661），呼吸事件 AUC 0.825；在 ISRUC‑Sleep、Sleep‑EDF 与 UCD 上同样保持领先或相当的表现，且在未再训练的情况下可直接迁移。

**⚠️ 局限性**

① 仅在单中心卒中数据上评估；② 该系统为离线推理，未实现实时评分；③ 呼吸事件预测为 30 s 期的起始得分，未给出精确时长；④ 需要进一步验证其在更大、多中心、不同采集设备下的鲁棒性。

---

## 271. An Unconditional Space-Approximation Lower Bound for Semi-Streaming Matching

**arXiv ID:** 2610.36577 | [PDF](https://arxiv.org/pdf/2610.36577v1)

**作者:** Yang P. Liu `[一作]` (Carnegie Mellon University), Janani Sundaresan `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在p轮流算法中，近似最大二分图匹配的空间下界，并给出了从多方通信复杂度到流算法的降低证明。

**💡 创新点**

通过构造层次化的RS图、活跃树以及基于KL散度的通信分析，提出了新的多层嵌入与解码技术，将通信复杂度与流空间消耗直接关联。

**🔧 技术方法**

利用KL散度链式规则、随机RS图、层次化嵌入、活跃树分析、解码器构造以及流算法到通信协议的标准模拟技术。

**📊 数据集**

本文为理论分析，不使用实际数据集，而是构造随机输入与RS图来证明下界。

**📈 对比分析**

通过理论证明得到的空间下界（Ω(...))表明即使是常数误差的p-pass近似算法也必须使用至少该量级的空间，超过了目前已知的上界，展示了近似因子与空间的严格权衡。

**⚠️ 局限性**

仅针对二分图匹配、p-pass、(1-3^{-p}+)-近似以及随机输入构造，无法直接推广到单轮或非二分图匹配，且证明依赖于特定参数取值，实际实现上仍存在实现困难。

---

## 272. ADTGuard: Adaptive Depth-Time Safeguard for Streaming Inference in Large Language Models

**arXiv ID:** 2610.36574 | [PDF](https://arxiv.org/pdf/2610.36574v1)

**作者:** Chuming Wei `[一作]` (Xi'an Jiaotong-Liverpool University), Guangwen Yang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ADTGuard，一种在冻结LLM内部表示上实现逐token安全检测的轻量级插件。

**💡 创新点**

创新点在于自适应层选择（Causal Depth Router）与深度-时间安全动态模型（Depth-Time Safety Dynamics）结合，并通过风险启动目标细化（Risk-Onset Target Refinement）从仅有的响应级标签中逐步生成token级监督，解决传统方法在时间粒度和监督精细度上的缺陷。

**🔧 技术方法**

使用技术包括Causal Depth Router (CDR)、Depth-Time Safety Dynamics (DTSD) 的选择性状态空间实现、Risk-Onset Target Refinement (RTR)、微调仅约15M参数的检测模块，以及对冻结LLM内部隐藏状态的读取和处理。

**📊 数据集**

实验基准涵盖SGB-S-Eval、SGB-WildGuard、AEGIS2.0、BeaverTails以及WildGuard等安全数据集，采用不同LLM骨干（Qwen3-8B、Llama-3.1-8B、Gemma-7B）进行评估。

**📈 对比分析**

与SIREN、Kelp等表示型安全方案以及Qwen3Guard-Stream、Llama-Guard、SentGuard等专用调解模型在Response F1和Streaming F1上进行对比，ADTGuard在14/15骨干-数据集组合中实现最高或接近最高的Streaming F1和Response F1，同时仅需16M可训练参数，且每token延迟低于3ms。

**⚠️ 局限性**

局限性包括：仍依赖冻结LLM的内部表示，若LLM更新需重新训练；对响应级标签的依赖可能在极端攻击或标签不准确时影响性能；路由预算和目标细化调度需手动调优；在极端对抗样本或长序列生成时，检测性能仍可能出现误判。

---

## 273. TAC2: Timestamped Audio Captioning as composable functions

**arXiv ID:** 2610.36568 | [PDF](https://arxiv.org/pdf/2610.36568v1)

**作者:** Swapnil Bhosale `[一作]` (University of Surrey), Zeyu Jin `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段的音频描述框架（基于多视角弱教师融合的“音频描述器”与轻量级专家的表征抽取），生成可查询、时间上有根的多粒度音频时间线；

**💡 创新点**

创新点在于将音频描述任务拆解为可组合的Instruction-Input-Output（IIO）格式，支持单次解码与多次级联；通过轻量化LoRA专家从预训练模型中提取隐式情感、语调等表情属性，并将其蒸馏为统一模型；

**🔧 技术方法**

核心技术包括基于时间注册IoU匹配的多视角融合、IIO完成格式、LoRA专家适配与知识蒸馏、以及与文本生成模型的联动推理；

**📊 数据集**

使用了公开的IEMOCAP、RAVDESS、MELD、Common Voice等小规模标签集进行专家训练，利用大量无标签“野生”音频通过弱教师（事件检测、说话人分离、音频字幕）生成训练目标；

**📈 对比分析**

与现有大型音频-语言模型（如TAC、Qwen3Omni-Captioner、Gemini-3.7 Flash）对比，在MMSU、MMAR、MMAU以及MSU-Bench等基准上，所提方法在感知任务上提升约10‑15%，整体得分接近或略低于教师级联；在文本到音频生成实验中，能够实现事件时间精确重建和表情、强调等零样本控制，超越PicoAudio2等基线；

**⚠️ 局限性**

局限在于依赖预训练模型中隐式表征的可提取性，轻量专家对少量标注数据敏感；多视角融合对弱教师质量敏感；在极端混响或低信噪比场景下的时序精度仍待提升。

---

## 274. DeepInfer-Mac: Running a 750B-Class MoE on a 24 GB Laptop

**arXiv ID:** 2610.36564 | [PDF](https://arxiv.org/pdf/2610.36564v1)

**作者:** Haorui Li `[一作]` `[通讯]` (Southeast University), Haorui Li (Southeast University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现并测量了在24GB统一内存的MacBook Air上运行750B级DeepSeek‑V4.1‑Flash模型的系统，使用SSD存储大部分参数并通过专家缓存与预取降低存储访问延迟。

**💡 创新点**

首次在单机小内存环境下演示如何通过Mixture‑of‑Experts（MoE）架构将大模型分布式到磁盘，并结合下一层预测和需求优先的专家预取，实现可行的本地推理。

**🔧 技术方法**

采用了量化压缩（2/3/4位专家权重、4位Engram表）、MLX GPU推理框架、SSD直接读入GPU缓冲、基于LRU的专家缓存管理和下一层预取技术。

**📊 数据集**

使用DeepSeek‑V4.1‑Flash 750B权重（转换后为337.6 GiB）进行评估，测试英文、中文及代码生成任务；无公开公开数据集，侧重模型自身推理性能。

**📈 对比分析**

对比三种调度方案（串行、缺失专家重叠、缺失专家+下一层预取），发现预取方案相较于仅重叠可提升约10‑20%的token/s（具体取决于提示长度），但生成首次token的延迟仍由磁盘I/O主导。

**⚠️ 局限性**

局限在于仅在单机MacBook Air上实验，存在共享主机和swap影响；量化转换与可扩展性未做全面质量评估，且长提示仍导致显著延迟。

---

## 275. A Fibrational Framework For Multi-Conclusion Logics

**arXiv ID:** 2610.36552 | [PDF](https://arxiv.org/pdf/2610.36552v1)

**作者:** Oualid Merzouga `[一作]` `[通讯]`, Oualid Merzouga

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于图形描述符的通用多结论序贯算子框架，统一了传统单结论与多结论推理，并为模态与亚结构逻辑提供了可扩展的两层语义架构。

**💡 创新点**

创新点在于：①用图形而非树形结构描述上下文，直接映射多输出的多项式范畴；②引入 F/U 两类连接词以实现线性与笛卡尔区间的交互；③在该框架下给出了多结论可裁剪性证明及其与 LNL 2-多范畴的纤维化语义对应，完成了从意大利子结构到经典多重线性逻辑的逻辑与范畴双适配。

**🔧 技术方法**

使用的技术主要包括：图形化模式理论、线性与非线性多范畴（LNL Polycategory）、多项式范畴的极限与拉伸结构、纤维化与双纤维化的范畴语义、以及模式化的连接词生成与裁剪证明。

**📊 数据集**

无具体数据集，全部为形式化理论与范畴构造。

**📈 对比分析**

本工作没有实验比较与性能评估，评估仅通过理论证明完成，展示了裁剪可裁剪性、多结论适配性及语义对应的完备性。

**⚠️ 局限性**

限制：理论框架的实现复杂度高；对非标准亚结构（如多重极限）需手动指定模式签名；对更广泛的多范畴基底的可扩展性与初始性仍待进一步研究。

---

## 276. CARD: Contact-Anchored Retargeting for Dexterous Manipulation

**arXiv ID:** 2610.36539 | [PDF](https://arxiv.org/pdf/2610.36539v1)

**作者:** Cunxi Dai `[一作]` (Carnegie Mellon University), Akash Sharma `[通讯]` (Amazon Frontier AI & Robotics (FAR))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无模拟器、无物体模型的统一手-接触重定向框架 CARD，并实现了视频接触估计 Vid-HACO、接触锚定优化与触觉跟踪控制，使多模态人类演示能开环重放到不同机器人手。

**💡 创新点**

创新点在于将接触力与推向方向作为核心变量直接优化接触匹配；通过统一手-接触表示将视频、触觉手套和运动捕捉映射到同一接口；触觉跟踪控制器在硬件上保持目标接触力；只需语义映射即可实现跨机器人体型。

**🔧 技术方法**

采用 MANO 关节模型、Vid-HACO 视觉接触预测、零阶采样优化（CEM）求解接触锚定目标、触觉反馈 admittance 控制以及统一手-接触与语义映射。

**📊 数据集**

使用了 20 条单目视频、66 条触觉手套+运动捕捉视频、48 条运动捕捉手套视频以及公开的 GRAB 100 条数据集。

**📈 对比分析**

与基线 kinematic 方案（DexPilot、AnyTeleop-K/P、ReGrind）以及 SPIDER 物理重定向进行比较，CARD 在视频/手套/运动捕捉上接触成功率 61–78%（基线 <10%），在 GRAB 上 43%（基线 22.5%），开环硬件成功率从 4/30 提升至 25/30，物理重定向性能从 12% 提升至 32%。

**⚠️ 局限性**

受限于演示质量，开环模式无法补偿大幅物体位移，缺乏闭环感知与自适应；未来计划通过生成数据训练闭环策略以克服这些限制。

---

## 277. Silent Blindness: Fault Vulnerability of Hardware Malware Detectors

**arXiv ID:** 2610.36612 | [PDF](https://arxiv.org/pdf/2610.36612v1)

**作者:** Alireza Abolhasani Zeraatkar `[一作]` (University of California), Hussain Al-Asaad `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究对硬件恶意软件检测器（HMD）在逻辑与参数损坏情况下的漏洞进行了系统评估，采用结构级故障注入对整个设计空间进行实验。

**💡 创新点**

创新点在于提出基于安全后果的故障分类法，量化了无声与关键故障的比例，揭示了树型与函数型模型的脆弱性差异，并设计了最优的选择性硬化算法。

**🔧 技术方法**

使用了结构级 stuck-at 与参数位翻转注入技术，并结合 AdaBoost、Bagging 集成、最小最大（minimax）选择算法与贪心预算覆盖策略实现硬化选择。

**📊 数据集**

实验基于 7,400 条验证实例，覆盖 18 类分类器、4 种计数预算以及 140 个可枚举配置（含 AdaBoost 与 Bagging）。

**📈 对比分析**

通过比较每个故障对 recall 与 FP 率的影响，将故障划分为逃逸、拒绝服务、无声等；结果显示树型模型关键故障比例低但无声故障高，Bagging 可显著降低关键故障，而 AdaBoost 可能提高；通过选择性硬化可将单次故障 recall 损失从 60% 降至 3%。

**⚠️ 局限性**

限制在于仅在结构层注入，未验证门级或物理实现的对应性；未覆盖 KNN 存储样本故障；硬化评估使用面积分配而非真实物理版；未验证硬化后的实际面积与性能开销。

---

## 278. 4D Reconstruction with Events & RGB

**arXiv ID:** 2610.36610 | [PDF](https://arxiv.org/pdf/2610.36610v1)

**作者:** Dulanga Weerakoon `[一作]` (Singapore-MIT Alliance for Research and Technology Centre), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在MonST3R基础上加入事件体素分支，通过拼接与RGB特征融合，实现轻量级的4D重建方法。

**💡 创新点**

创新点在于无需改动原有MonST3R架构，仅添加事件分支并通过简单拼接实现RGB与事件的融合，证明了事件信息可显著提升重建精度。

**🔧 技术方法**

采用事件体素化（5个时间桶）、ViT编码器、单层卷积适配器、MonST3R解码器，并使用V2E模拟器生成合成事件流。

**📊 数据集**

使用 Sintel、Bonn、KITTI、PointOdyssey 四个基准数据集，并为每个数据集合成对应的事件体素。

**📈 对比分析**

与 RGB 单模、事件单模以及事件转 RGB 再输入 MonST3R 的三种基线进行对比；实验显示 RGB+事件在相机姿态估计和视频深度估计上均优于单模，尤其在高运动场景下误差平均降低约3.4%。

**⚠️ 局限性**

局限性包括事件数据为合成；融合方式仅为拼接，未充分利用事件的异步特性；仅验证在 MonST3R 上，未测试更先进的 4D 重建模型或真实事件相机。

---

## 279. Learning to Formalize: Bridging Open-Ended Workloads and Fast Decision Models

**arXiv ID:** 2610.36595 | [PDF](https://arxiv.org/pdf/2610.36595v1)

**作者:** Chengxuan Pei `[一作]` (CUHK-Shenzhen), Zhe Li `[通讯]` (Kunlunxin Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练一个小型候选人格式化器在给定任务上下文中生成具体候选动作列表，然后由冻结的决策模型在这些候选项中选择执行。

**💡 创新点**

提出候选列表蒸馏方法，将大型生成模型的候选生成能力迁移到小型学生模型，拆分候选生成与决策选择两步，显著降低延迟并提升成功率。

**🔧 技术方法**

使用监督式微调（SFT）对候选列表进行蒸馏，基于 Qwen3 系列小模型作为生成器，JEV-27B 冻结作为选择器，并配合验证器做语义校验。

**📊 数据集**

使用 CALVIN-D 受控操纵数据集（200 条测试样本、1,011 条验证样本）以及 DeepSeek V4 Flash 训练的 36,453 条候选生成对。

**📈 对比分析**

通过在 6 条对照架构上评估成功率、决策延迟和 GPU 资源消耗，发现 4B/1.7B 蒸馏版成功率约 52%（延迟分别 2.573s/1.826s），相比直接 Qwen3.5-27B 生成的成功率 31.8% 与 3.67s 延迟，资源使用更低。

**⚠️ 局限性**

仅在 CALVIN-D 受控实验环境下验证，未测试实时性能、跨数据集或跨选择器迁移；候选列表可能缺失有效动作或继承教师错误，未单独剖析选择器与候选质量对性能的贡献；资源与成本评估不完整。

---

## 280. LabWorld: Learning Interactive 4D Worlds for Scientific Experimentation

**arXiv ID:** 2610.36592 | [PDF](https://arxiv.org/pdf/2610.36592v1)

**作者:** Suozhi Huang `[一作]` (Princeton University), Mengdi Wang `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将单目RGB实验室视频自动重建为可编辑的交互式4D实验室世界，包含可复用的3D资产、度量场景、时变物体状态和实验流程事件。

**💡 创新点**

创新点在于：① 通过事件驱动的重建方法，将视觉与几何观测与实验事件的空间时间约束相结合，确保重建的物体运动在图像上与物理上都符合实验流程；② 采用端到端的框架从普通视频中生成完整的可重放、可编辑的实验世界；③ 引入基于事件的约束优化，弥补传统基于图像匹配方法的物理一致性不足。

**🔧 技术方法**

使用了多种技术：视觉语言模型提取对象与事件，SAM3/MapAnything 进行分割与深度估计，生成式3D资产构建器，轨迹优化框架（结合观测损失、几何损失、平滑损失、事件损失和物理约束），以及事件驱动的约束编解码器。

**📊 数据集**

收集了实验室内的真实 pipetting（10条）和 vortexing（10条）视频数据，用于评估和对照。

**📈 对比分析**

与 GPT‑6 Astra、FoundationPose、Do as I Do 等通用与专用重建基线进行对比；在 pipetting 上 Mask IoU 最高 0.501（比 Do as I Do 0.468 高 0.033），在 vortexing 上 Tube‑Length‑Normalized PCK‑AUC 最高 0.638（比 Do as I Do 0.603 高 0.035）。

**⚠️ 局限性**

局限性包括：严重手部遮挡会导致几何和运动估计不确定；事件约束的有效性高度依赖于事件提取和场景几何估计的准确性。

---

## 281. InfraDev: Can Coding Agents Develop AI Infrastructure, and How Can We Tell?

**arXiv ID:** 2610.36589 | [PDF](https://arxiv.org/pdf/2610.36589v1)

**作者:** Yangshen Deng `[一作]` (University of Edinburgh), Luo Mai `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向AI基础设施的真实任务基准（36个从Megatron-LM、vLLM、verl合并PR抽取），并对任务进行隐藏探针与提示的联合合成，搭建共享执行环境。

**💡 创新点**

创新点在于：①联合生成隐藏探针与任务提示，弥补官方测试不足和提示不完整导致的误判；②构建资源共享的执行环境，显著降低存储和构建时间；③通过对抗性补丁检测并提升评测可靠性。

**🔧 技术方法**

采用对抗性补丁生成、隐藏探针合成、提示合成、虚拟环境共享与缓存、GPU多卡并行执行等技术。

**📊 数据集**

使用从21,389条合并PR中筛选出的36条真实PR作为任务数据，涉及真实模型权重、数据集及对应的测试与依赖。

**📈 对比分析**

通过与六款前沿编码代理的对比（%resolved、%passed、耗时、成本）评估，证明隐藏探针和共享环境能提升模型任务完成率（最高38.9%），但整体性能仍有限。

**⚠️ 局限性**

局限性包括：多GPU执行成本高、任务筛选仍依赖人工专家、基准规模受限、模型整体仍难以完成大多数任务。

---

## 282. Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements

**arXiv ID:** 2610.36584 | [PDF](https://arxiv.org/pdf/2610.36584v1)

**作者:** Binhang Qi `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个包含5,086个真实项目的需求-测试对齐实例的基准，用于评估大语言模型代理在目标实现隐藏的情况下从自然语言需求生成可执行且行为充分的测试代码的能力；同时对比了多种代理-模型-推理配置的效果与成本；

**💡 创新点**

创新点在于：①首次提供基于BDD Gherkin需求的真实、执行验证的需求-测试映射；②引入目标实现隐藏与全隐藏两种视图；③利用变异测试和API调用覆盖度等多维度评价生成测试的行为充分度；④揭示LLM在生成测试时存在的“信息缺口”与“幻觉”问题，为后续需求驱动的测试生成器提出改进方向；

**🔧 技术方法**

使用技术包括：大语言模型代理（Aider、OpenHands、Claude Code、Codex CLI）与LLM（DeepSeek-V4.1-Flash、GPT‑6 Luna/Sol、Claude Sonnet 5.5）；BDD框架（Cucumber、Behave、Behat、Godog）下的Gherkin需求解析；变异测试（多种变异算子）与代码覆盖、API调用对比的执行验证；

**📊 数据集**

使用的数据集来自8个开源项目（CoreShop、PrestaShop、Sylius、python‑docx、Algorand SDK、Kapua、e‑petitions、coddy‑agent），涵盖5种语言（Java、Python、PHP、Go、Ruby）和4个BDD框架，共5,086个需求-测试实例；

**📈 对比分析**

比较方法：以Pass@1和KillComplete@1（即测试能否通过且在变异检测上完全杀死基准测试能杀死的所有变异体）为主要指标，评估不同配置的执行成功率和行为充分度；实验结果显示：最优配置在目标隐藏视图下Pass@1可达82.8%，但KillComplete仅33.8%；成本与KillComplete之间呈边际收益递减，额外投入几何增长仅带来几百分点的提升；

**⚠️ 局限性**

limitations包括：①需求描述往往不完整，导致生成测试缺失必要的设置或检查；②LLM在接口使用上常出现幻觉或误用；③缺乏对项目特定约定、fixtures、入口点的深入推理；④实验基于BDD需求，可能不代表更自然或口语化的需求场景；⑤生成过程的非确定性导致结果波动；⑥部分项目代码可能已被LLM训练数据覆盖，影响泛化评估。

---

## 283. Peak-DNA: Recurrent Morphology in Global Narrative Propagation

**arXiv ID:** 2610.36571 | [PDF](https://arxiv.org/pdf/2610.36571v1)

**作者:** Joosung Kim `[一作]` (Nagoya University), Shanwen Wu `[通讯]` (Nagoya University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 Peak‑DNA 框架，用来从噪声严重的全球叙事传播流中识别并聚类可重复出现的宏观传播形态（宏峰），并用 DTW 相似度评估这些形态的内部一致性与跨域关系。

**💡 创新点**

创新点在于：① 把叙事传播视为可重复的形态结构而非单一波形；② 通过宏峰合并、形态归一化与 DTW 对齐来克服异步与噪声；③ 将聚类得到的主峰 DNA 类比为“叙事基因组”，为后续叙事家族发现与纵向演化分析奠定结构基础。

**🔧 技术方法**

技术手段包括 Savitzky‑Golay 平滑去噪、局部峰值检测与宏峰合并、三维形态归一化（水平、导数、面积归一化）、Sakoe‑Chiba 带约束 DTW、层次聚类（平均链接）以及聚类评价指标。

**📊 数据集**

数据集为 2026 年 1 月 1 日至 5 月 12 日的 GDELT 事件数据，按小时汇总得到五个高放大叙事域（Iran、Defense、AI、Space/Elon、Quantum/PQC）的传播流。

**📈 对比分析**

通过 DTW 计算宏峰形态的相似距离，并用层次聚类划分峰 DNA 类别；表现为每个域内宏峰形态高度聚集（内部相似度 > 0.8），主峰 DNA 覆盖率在 30%–60% 之间；跨域相似度揭示了两个高层级组（Quantum/PQC–Space/Elon 与 Defense–Iran–AI），但未与传统基准进行定量对比。

**⚠️ 局限性**

局限性包括：① 仅覆盖单一观察窗口；② 只研究五个高放大域，缺乏对更广泛叙事场景的验证；③ 参数设置未做域自适应；④ 缺少长期纵向追踪与外部基准评估，导致结果的泛化与可复现性待进一步验证。

---

## 284. PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy

**arXiv ID:** 2610.36554 | [PDF](https://arxiv.org/pdf/2610.36554v1)

**作者:** Vansh Bhatia `[一作]` (Advanced Micro Devices, Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PRISM基准，包含1000道专家设计的多模态推理题，每题配有人类理想化推理轨迹，并对模型进行感知、推理内容和成本意识的诊断。

**💡 创新点**

首次在多模态基准中提供逐步推理轨迹，允许基于感知探针和推理轨迹对齐进行定量评估，并引入成本感知指标重排模型排名。

**🔧 技术方法**

使用链式思维提示、步骤级奖励模型、文本相似度对齐、log长度归一化的成本指标、视觉感知探针以及结构化推理轨迹。

**📊 数据集**

基于RefCOCO、OpenStreetMap、MedXpertQA、手写扫描等公开图像，构造5个推理类别，共1000个问题。

**📈 对比分析**

对25个开闭源模型进行评估，最高整体准确率66%，远低于人类83.9%；感知探针显示85%失败为感知错误，推理轨迹对齐AUROC0.66；成本指标重新排序24%模型。

**⚠️ 局限性**

缺乏强大过程奖励模型，轨迹依赖人工标注，公开图像可能存在泄露风险，且成本指标仍未完全反映真实部署开销。

---

## 285. REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal

**arXiv ID:** 2610.36546 | [PDF](https://arxiv.org/pdf/2610.36546v1)

**作者:** Shengzhang Xu `[一作]` (Southern University of Science and Technology), He Kong `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出REACH框架，使得Unitree G1人形机器人能够在同一架构下实现可变高度的双足行走、爬行以及平滑的模式切换，并通过LiDAR规划器实现三维路径与行走指令的无缝衔接。

**💡 创新点**

创新点包括①基于历史的潜在适配模块，用以捕捉接触丰富爬行中的地形与交互上下文；②双向运动跟踪策略，利用生成的过渡动作实现双足与四足模式的无缝切换；③在EGO-Planner基础上加入人体高度约束的几何优化，兼顾高度调节与碰撞回避。

**🔧 技术方法**

核心技术包括强化学习中的非对称Actor-Critic与历史条件潜在编码器、BeyondMimic运动跟踪、DreamWaQ式潜在推断、改造版EGO-Planner、LiDAR升高映射与Pure Pursuit路径追踪。

**📊 数据集**

实验使用自建的仿真地形难度递增数据集（箱体阵列、随机崎岖地形、波浪地形、斜坡与平台），以及在真实Unitree G1机器人上收集的LiDAR点云和机器人传感数据。

**📈 对比分析**

通过对比包含/不包含历史编码器、空闲惩罚以及专用地形扫描的四种训练变体，发现完整模型在课程进展和指令跟踪上表现最佳；双向过渡策略在仿真中实现100%/96%成功率，在硬件上达成92%/100%的成功率，实验证明机器人能在80cm高度障碍处行走、在40cm高度隧道中爬行并完成模式切换。

**⚠️ 局限性**

局限性在于对LiDAR高度映射的精度要求较高，行走/爬行高度仅限于0.2–1.27 m范围；尽管过渡策略能在多次随机化下稳健，但在硬件中crawl→squat方向仍略有失败；整体框架需要大规模仿真随机化与精细的训练曲线，未对更大尺寸或更复杂结构的障碍进行验证。

---

## 286. FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation

**arXiv ID:** 2610.36545 | [PDF](https://arxiv.org/pdf/2610.36545v1)

**作者:** Yijing Lin `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种统一的条件图像生成框架 FrameForge，利用预训练视频模型的时序先验，能够在保持源图像内容的同时根据文本或结构指令生成目标图像。

**💡 创新点**

创新点在于将条件-目标关系视为输入自适应的“虚拟时序关系”，并引入两大模块：Mixture of Temporal Experts（MoTE）在每个去噪步骤动态选择不同时序RoPE专家；Chain of Temporal Reasoning（CoTR）通过无监督的中间隐状态与注意力瓶颈，实现条件到目标的隐式变换。

**🔧 技术方法**

核心技术包括：冻结的视频扩散变压器（HunyuanVideo1.5）、LoRA 参数高效适配、3D RoPE 时序编码、基于文本与视觉上下文的轻量路由器、以及在中间训练阶段使用的随机注意力遮罩。

**📊 数据集**

使用的数据集包括：DreamBench（主题驱动生成）、COCO2017（边缘到图像生成）、以及 GEdit-Bench（指令导向图像编辑）。

**📈 对比分析**

与多种基线（如 OminiControl、ACE++、EasyControl、ChronoEdit 等）比较，FrameForge 在 DreamBench 的整体得分、COCO2017 的 SSIM/FID/CLIP‑T 以及 GEdit 的 G‑SC/G‑PQ/G‑O 指标上均实现了显著提升，证明其在多任务下的统一性能。

**⚠️ 局限性**

局限性包括：依赖冻结的视频骨干，可能限制对新任务的快速迁移；中间隐状态仅采用单个 latent，复杂变换可能受限；在极端任务（如大幅度结构变更）下仍可能出现细节失真；以及对不同视频模型的泛化需要进一步验证。

---

## 287. ATEMPO: Amortized Transfer Entropy Model Per-lag Over-Time

**arXiv ID:** 2610.36536 | [PDF](https://arxiv.org/pdf/2610.36536v1)

**作者:** Emma Andrews `[一作]` (National University of Singapore), Gianmarco Mengaldo `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 ATEMPO 的自适应延迟估计方法，利用条件扩散评分框架将延迟、条件深度和时间位置作为共享模型的输入，以一次性推断全时域的传递熵 (TE) 表面。

**💡 创新点**

创新点在于将原本需要针对每个 (k,l) 对单独拟合的 TE 估计转为一个可 amortized 的共享模型，从而显著降低计算成本；同时通过对 TE(k,l,t) 的差分实现时间变化的延迟检测，可在不重新训练的情况下捕获动态依赖。

**🔧 技术方法**

使用条件扩散评分 (conditional diffusion score) 估计 TE，并将结果与传统的 KNN、神经网络、密度估计、相互信息与 CCMI 等基线进行对比；在实验中还采用了基于时间条件的单次训练策略。

**📊 数据集**

评估数据集包括：气象观测数据 WEATHER‑5K（5,672 站点、5 维变量）、Neural Latents Benchmark (NLB) 的三种运动皮层任务、12 个包含真实 TE 的合成数据以及跨模态的 MMTT‑Bench。

**📈 对比分析**

与八个基线（TENDE、NPEET、NJEE、MINE、CCMI、PCMI、Liang‑Kleeman、DTW）相比，ATEMPO 在 TE 估计误差、延迟检测准确率和下游预测（MSE/R²）上均表现最佳；在 WEATHER‑5K 上可将 MSE 降低 5–7%，在 NLB 任务上平均 R² 提升约 4.1%；在合成数据上，ATEMPO 的 TE 误差最低、延迟检测准确率最高，并实现了 1.0 的“oracle”级预测性能。

**⚠️ 局限性**

局限性包括：需足够样本和模型容量才能准确估计高维 (K,l) 的 TE；估计误差可能导致延迟选择不最优；实验多聚焦于单源 TE，尚未扩展到多源协同信息；真实世界数据的 TE 无法完全验证，仅能通过下游性能间接评估。

---

## 288. Causal Localization of Evidence Effects in Transformers

**arXiv ID:** 2610.36524 | [PDF](https://arxiv.org/pdf/2610.36524v1)

**作者:** Jipeng Li `[一作]`, Yanning Shen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过受控符号输入（Controlled Evidence Aggregation）评估语言模型中注意力头对决策的因果影响。

**💡 创新点**

主要创新是结合符号计数操控与平衡效用因果选择（Balanced Utility Causal selection）来发现紧凑的头集合，并给出条件理论证明其可在有限样本下恢复。

**🔧 技术方法**

采用Transformer冻结模型、符号计数提示、logit差异测量、头写入的零化与均值补丁、贪心搜索以及子模性质分析。

**📊 数据集**

数据集为Pythia-410M的符号提示以及人工设计的自然语言桥接任务（16条报告决定状态）。

**📈 对比分析**

与随机层匹配头组、敏感性单头、ActPatch等对比，结果显示所选头组在符号和自然语言任务中均能显著降低偏差，恢复率高于随机组；在BF16实验中约60‑90%计数效应被消除，NL准确率下降约8个百分点。

**⚠️ 局限性**

局限在于只能定位写入位置的因果作用，缺乏对完整路径或早期隐藏层的解释；符号与自然语言之间的迁移受限；搜索成本与可重复性受限；未提供完整置信区间和对多模型泛化的证明。

---

## 289. NextEval: An Agent-Based Solver and Benchmark for Derivative-Free Optimization

**arXiv ID:** 2610.36553 | [PDF](https://arxiv.org/pdf/2610.36553v1)

**作者:** Shuo Chen `[一作]` (Independent Researcher), Zaikun Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NextEval，利用大语言模型（LLM）代理与数值环境相结合的无导数优化（DFO）框架，并设计了NextEval Bench基准协议；

**💡 创新点**

创新点在于：①将LLM代理作为决策层，能够根据历史评估结果选择、配置并切换数值工具；②通过工具接口将传统数值方法（如信赖域、poll搜索、几何修复）模块化；③提供统一的基准协议以在相同预算和信息约束下比较不同LLM及其调度策略；

**🔧 技术方法**

采用的技术包括：LLM代理（如GPT‑5.6、Qwen 3.8 Flash、DeepSeek V4 Flash）、工具管理与对话抓取的 harness、数值环境用于记录评估历史与方法状态、OptiProfiler用于生成性能与数据曲线、S2MPJ低维无约束优化问题库；

**📊 数据集**

使用了92个低维（n≤5）无约束优化问题（S2MPJ合集），并在实验中对起点进行微扰、添加噪声、旋转坐标，统一预算为50n次目标函数评估；

**📈 对比分析**

与经典DFO算法（COBYQA、NEWUOA、SciPy Powell、BFGS）在相同预算下比较，采用相对误差阈值的性能与数据曲线（performance and data profiles）和累计得分（aggregate score）。实验结果显示，NextEval在旋转起点实验中取得最高的累积得分（1.000），显著优于所有基线；

**⚠️ 局限性**

局限性包括：仅验证低维问题（n≤5）；未评估运行时间或LLM使用成本；对数值工具与LLM的相互作用尚未分离；旋转坐标不完全匿名；尚未在更高维或其他问题族上测试泛化能力。

---

## 290. HeteGCD: Statistics-Aware Dual-Prototype Learning for Generalized Category Discovery on Heterophilous Graphs

**arXiv ID:** 2610.36696 | [PDF](https://arxiv.org/pdf/2610.36696v1)

**作者:** Xuanzhi Xi `[一作]` (University of Electronic Science and Technology of China), Junming Shao `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种针对异质图的通用类别发现框架 HeteGCD，能够在只给定已知类别标签的条件下识别未知类别。

**💡 创新点**

创新点在于将节点属性与局部属性差异统计信息融合进嵌入空间，同时利用邻居类别分布统计构造分布原型，在概率空间与特征空间双重原型学习中实现无监督的类别推断。

**🔧 技术方法**

技术包括：门控双流编码器（gated dual‑stream encoder）对属性与属性差异进行融合；对比学习（InfoNCE）与自监督蒸馏；分布原型构造与总变差相似度匹配；以及对称自蒸馏和均值熵正则化（MeMax）提升类别覆盖率。

**📊 数据集**

实验使用五个异质图数据集：Roman Empire、Chameleon、Actor、BlogCatalog、Flickr，全部采用 5 份随机划分并在每份中将约一半类别标记为已知。

**📈 对比分析**

与 24 种基线组合（8 个编码器 × 3 发现框架）以及 7 个公开 GCD 方法对比，HeteGCD 在 All 及 New 这两个评测指标上均取得最佳平均排名（All：3.80，New：5.40），在所有数据集上均排在前六，整体性能优于所有基线。

**⚠️ 局限性**

局限性包括：仅在 5 个中等规模异质图上验证，缺乏大规模稠密图的可扩展性分析；对节点属性与邻居分布的依赖在属性缺失或噪声较大的场景下可能受限；以及对超参数（如 λ、ε、α）的敏感性需要进一步自动化调优。

---

## 291. A Quantitative Approach to Asch Conformity in Frontier Large Language Models

**arXiv ID:** 2610.36668 | [PDF](https://arxiv.org/pdf/2610.36668v1)

**作者:** Senthil Rajasekaran `[一作]` (Universite Libre De Bruxelles), Stylianos Loukas Vasileiou `[通讯]` (New Mexico State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过改编 Asch 从众实验，使用单一填空句子多次采样，比较大语言模型在单独与被制造共识影响下的答案分布变化。

**💡 创新点**

创新点在于利用 LLM 的概率性质，生成完整答案分布并进行分布‑对‑分布比较，而非仅计数回答；并通过多次同一问题采样来衡量群体影响的细微程度。

**🔧 技术方法**

技术包括：多次采样生成答案分布、构造首选词与非首选词、在提示中加入制造的“共识”参与者、计算答案分布差异（总变差距离、Jensen–Shannon 散度）、以及 Fisher 精确检验与 Wilcoxon 符号秩检验等统计方法。

**📊 数据集**

使用十句 2015 年 SAT 风格的填空句子作为任务文本；测试六个前沿 LLM（gpt‑6‑sol、gpt‑6‑luna、claude‑sonnet‑5‑5、claude‑opus‑5‑5、gemini‑3.8‑flash、gemini‑4‑argon）；并用 gpt‑5.6‑terra 生成非首选词。

**📈 对比分析**

通过比较单独阶段与不同规模（1、2、5、10）群体阶段的答案分布，计算从众移位 Δn，发现信息影响（首选词）最强，Δn 可达 73 个百分点；在 n=5 与 n=10 之间变化有限。统计检验表明大多数问题在 n≥1 时显著受影响。

**⚠️ 局限性**

限制包括：仅通过 100 次抽样近似模型内部分布，缺乏对真实分布的直接访问；实验仅在 SAT 填空句子上进行，难以推广到更广泛的任务；且共识提示的构造可能与真实社会情境差异。

---

## 292. Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs

**arXiv ID:** 2610.36667 | [PDF](https://arxiv.org/pdf/2610.36667v1)

**作者:** Satomu Tomisaki `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将英语SDRT语料MSDC和STAC使用LLM翻译成日语，构建了日语SDRT语料库，并对日英两种语言的SDRT解析模型进行训练与评估；同时开展了三项消融实验以分析模型对不同信息的依赖；

**💡 创新点**

创新点在于首次验证SDRT解析在日语上的跨语言可迁移性，利用LLM自动翻译生成日语语料库，并揭示模型主要依赖语篇关系结构而非语义内容；

**🔧 技术方法**

技术包括LLM翻译（GPT‑5.1）、Fine‑tune Llamipa（基于Llama‑3/Swallow‑3）、prompt‑based解析（GPT‑5.5/Llama‑3.1/Swallow‑3.1）、F1评估与XCOMET‑XL质量检验；

**📊 数据集**

使用的数据集为英语原版MSDC和STAC，以及对应的日语翻译版本；

**📈 对比分析**

对比方法为Fine‑tune vs Prompt，采用加权平均F1评估；Fine‑tune模型在MSDC上约0.79，Prompt约0.72；日英两种语言性能相近；消融实验显示去除语篇关系导致F1显著下降；

**⚠️ 局限性**

局限性包括仅使用游戏聊天日志两大语料，缺乏原生日语SDRT语料，翻译可能不自然，实验仅覆盖少数领域且未对完整日语语料进行人工评估。

---

## 293. Impossibility of No-Regret in Multi-Agent Bidding

**arXiv ID:** 2610.36666 | [PDF](https://arxiv.org/pdf/2610.36666v1)

**作者:** Zongjun Yang `[一作]` (Columbia University), Christian Kroer `[通讯]` (Columbia University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在n个竞标者参与的T轮重复第二价格拍卖中，竞标者在满足预算或投资回报率约束的情况下，无法同时实现无悔的情况。

**💡 创新点**

提出了高方差与高悔恨之间的确定性关系，并证明了在多代理环境中实现联合无悔的计算不可能性。

**🔧 技术方法**

使用了动态系统和数值模拟等技术来展示竞标算法的高方差行为。

**📊 数据集**

使用了模拟的拍卖数据集，涉及多个竞标者和不同的预算或投资回报率目标。

**📈 对比分析**

通过与现有竞标算法的比较，展示了在简单市场实例中，算法表现出高方差行为，导致悔恨线性增长。

**⚠️ 局限性**

限制在于无法保证所有竞标者在预算或投资回报率约束下实现无悔，且计算联合无悔的复杂性问题未得到解决。

---

## 294. The View from Nowhere: Population-Centered Agent Evolution

**arXiv ID:** 2610.36660 | [PDF](https://arxiv.org/pdf/2610.36660v1)

**作者:** Shuo Yan `[一作]` (Macquarie University), Shang Gao `[通讯]` (Macquarie University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Population-Centered Evolution (PCE)，通过聚合进化族群中所有变体的任务记录和自我改造历史，生成“population block”来指导父代改造，从而提升自我改进 AI 的性能。

**💡 创新点**

创新点在于：① 将族群整体经验转化为可供诊断使用的结构化证据，而非仅依赖父代本地信息；② 通过族群级诊断显著缩小了族群覆盖率与最佳个体性能之间的差距；③ 在保持模型与推理预算不变的前提下，实现更高的成功率和更低的 token 消耗。

**🔧 技术方法**

主要技术包括：进化算法、任务记录聚合、修改历史分析、PCE 的 distillation agent 与 diagnosis agent、基于 token 预算的高效推理、对照实验与 ablation 测试。

**📊 数据集**

使用了两个软件工程任务数据集：SWE‑bench Verified（200 任务）和 Polyglot（225 任务）。

**📈 对比分析**

与基线 GEA 对比：在 SWE‑bench 上成功率提升至 72.1%（vs 68.9%），Polyglot 上提升至 96.7%（vs 90.8%）。同时，PCE 在 SWE‑bench 上减少 36% 的总 token 消耗，在 Polyglot 上减少 59%，并保持了更短的轨迹长度和更少的长轨迹出现。

**⚠️ 局限性**

局限性包括：① 仍需要大量模型推理，无法完全解决高算力需求；② 主要在单一模型和任务分布下验证，未探讨跨模型或更大规模进化的迁移与压缩；③ 依赖族群记录的完整性与质量，缺乏对记录缺失或噪声的鲁棒性分析。

---

## 295. Tight Characterizations of AMM Mechanism Design under Weak Efficiency

**arXiv ID:** 2610.36635 | [PDF](https://arxiv.org/pdf/2610.36635v1)

**作者:** Chengkai Zhang `[一作]` (Rutgers University), Yuhao Li `[通讯]` (Columbia University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在 AMM（两资产批量化做市商）机制设计中，证明了在仅具备审查抗性（censorship resilience）而无公平排序（fair sequencing）的普通共识模型下，无法同时实现激励兼容（IC）与弱本地效率（wLE）的确定性机制；并且展示了去掉 Sybil、预算过报或价值误报任一策略限制后，能够构造满足 IC 与 wLE 的机制。

**💡 创新点**

创新点包括：①首次将公平排序与审查抗性在 AMM 机制中的区别明确为可分离的共识保障；②提出了一个完整的、最小化的不可行性证明，证明三种策略攻击（Sybil、预算过报、价值误报）均是导致不可能的关键因素；③在此框架下设计了三种可行机制（手持机制、比例手持机制、公共价值分层机制），说明了对不同策略约束的精细补救方法。

**🔧 技术方法**

主要技术手段：基于凸分析的价格曲线（concave AMM 曲线）与批量结算模型；利用弱本地效率的定量表述；构造两边匹配与一边分配的分层分配框架；通过对价差、预算与价值的边界分析，证明策略空间削减后的可行性；使用定理与引理组合构建严格的可证明机制。

**📊 数据集**

该工作为理论研究，未使用任何实际交易数据或公开数据集；所有结论均来自形式化模型和数学证明。

**📈 对比分析**

由于是理论结果，没有对照实验或性能指标；相比之前只在公平排序下可行的机制，本文展示了在更弱共识保障下不可行性，并给出了在限制策略空间时的可行机制，从而在理论层面完成了完整的可行性与不可行性图谱。

**⚠️ 局限性**

局限性包括：①仅考虑确定性机制；随机化机制仍有待探索；②仅关注批量结算的 AMM，未覆盖逐笔执行的情形；③对弱本地效率的定义相对简单，尚未考虑更强的效率约束；④模型假设静态、单周期交易，未考虑动态交易或时间延迟；⑤仍未给出对实际区块链部署的实现细节与成本分析。

---

## 296. SWIM: Learning What Matters from Behaviors for JEPA World Models

**arXiv ID:** 2610.36662 | [PDF](https://arxiv.org/pdf/2610.36662v1)

**作者:** Jiaming Hu `[一作]` (Boston University), Qingjun Cui `[通讯]` (Unity Technologies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过逆动力学模型自动学习离线轨迹中每个状态坐标的权重，并将其用于 Joint-embedding predictive architecture（如 LeWM）中 latent 空间的对齐，从而提升基于图像的规划性能。

**💡 创新点**

核心创新是无需人工预设或统一权重，而是用行为信息（记录的动作）来学习状态特征的相对重要性；学习得到的权重只在训练时使用，推理时无需额外计算。

**🔧 技术方法**

采用了 gated 逆动力学网络、LeWorldModel、SCALE 状态监督对齐、CEM/iCEM 规划以及门控权重学习和正则化等技术。

**📊 数据集**

在 OGBench 视觉操纵任务（Cube、Cube-Double、Cube-Triple、Cube-Quadruple、Scene）离线数据集上进行实验。

**📈 对比分析**

与 LeWM、LeWM+SCALE（均匀权重）以及预训练视觉特征基线在 CEM 和 iCEM 上进行对比，SWIM 在五个任务上平均提升 1–8.9 个百分点，尤其在 Scene 任务表现突出，并在统计上显著优于基线。

**⚠️ 局限性**

局限性包括需要先手工提供状态变量集合，且权重学习受限于离线数据的行为覆盖范围；在高度冗余或噪声状态特征的情况下，鲁棒性仍需进一步验证。

---

## 297. Specialization, Diversity, and Scaling in Child-Scale Language Models

**arXiv ID:** 2610.36679 | [PDF](https://arxiv.org/pdf/2610.36679v1)

**作者:** Steven Y. Feng `[一作]` (Stanford University), Michael C. Frank `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练小型 GPT‑2 语言模型在儿童家居记录 BabyView 与合成对话 TinyDialogues 上，探究在有限数据规模下语言输入的来源、分布和多样性如何影响学习和泛化。

**💡 创新点**

主要创新在于将学习目标固定、只变更训练数据来源，揭示了模型对个体儿童环境的高度专门化、词汇分布驱动的匹配优势，以及不同数据源在语法与词义学习上的差异。

**🔧 技术方法**

使用 GPT‑2 微型/小型模型、Kneser‑Ney 5‑gram 基线、BPE 分词、标准的 BPW、Zorro 语法评测和 WordSim 词义相似度评测。

**📊 数据集**

数据集包括 2026.1 版 BabyView 的自我记录转录（约 4.67M 词）和 TinyDialogues 的合成儿童对话（多达 12.8M 词），并在 100k-3.2M 词的不同规模下抽样。

**📈 对比分析**

通过交叉儿童比较、子集混合实验、以及与 5‑gram 的相关性分析评估模型，结果显示同一儿童环境下 BPW 降低约 0.77，混合儿童环境可显著提升聚合 held‑out 预测但对 Zorro 语法表现影响不大。

**⚠️ 局限性**

局限在于数据量仍远低于儿童真实接触量、仅使用文本转录缺乏多模态信息、模型与儿童认知机制差异大、以及评测指标与人类语言能力不完全对应。

---

## 298. The Batch Size Can Be One: Test-Time Adaptation from Classifier Geometry

**arXiv ID:** 2610.36628 | [PDF](https://arxiv.org/pdf/2610.36628v1)

**作者:** Jaeho Hwang `[一作]` (Ewha Womans University), Uiwon Hwang `[通讯]` (Ewha Womans University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为 OriTTA 的单样本测试时适应方法，在不依赖批统计或预测置信度的前提下，利用源模型的几何结构为每个类别构建理想目标，并在此框架下对特征进行角度和半径正则化，结合记忆池和崩溃检测实现稳定的在线适应。

**💡 创新点**

创新点在于：① 通过源模型的线性头与合成原型先恢复一个“重心化”坐标系，从而获得固定的理想点与锥体区域；② 采用基于角度与半径的单样本损失，无需批量统计或置信度筛选；③ 设计基于记忆池重分类的崩溃检测机制，能在出现类别集中预测时自动恢复。

**🔧 技术方法**

技术实现包括：对分类器行向量去中心化、投影至理想子空间、计算重心化原点 ω；构造理想点 μ_c 与锥体 Λ_c；对可靠特征施加角度损失 L_ang 与半径损失 L_rad；每隔一定步长对记忆池中的样本执行对齐损失 L_align；通过对记忆池重分类计算 σ 进行崩溃检测并恢复源权重。

**📊 数据集**

使用的数据集为 CIFAR‑10‑C、CIFAR‑100‑C 与 ImageNet‑C，涵盖 15 种不同噪声变形（severity 5），以及四个主干网络（WideResNet‑28‑10、WideResNet‑40‑2、ResNet‑50 + GroupNorm、ViT‑B/16）。

**📈 对比分析**

与 TENT、EATA、SAR、DeYO、ReCAP、SaTeen 等主流单样本 TTA 方法对比，OriTTA 在所有 16 种实验设置（四个主干、两种协议、两种流顺序）中均实现最高平均准确率；在 ImageNet‑C fog 上，100 次随机种子实验中无一次崩溃，且与最佳基线相比平均提升约 2.7–12.0 分。

**⚠️ 局限性**

局限性包括：① 当类别数 K-1 接近或大于特征维度 D（如 ViT‑B/16）时，投影无效，优势仅为 1–1.3 分；② 对于小类别数（如 CIFAR‑10）时，重心化与锥体规则效果有限；③ 需要线性头且行向量近等角，无法直接应用于长尾分布或非线性头的模型；④ 评估仅覆盖合成腐败，未验证自然分布漂移（如 ImageNet‑R、ImageNet‑Sketch）。

---

## 299. TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation

**arXiv ID:** 2610.36694 | [PDF](https://arxiv.org/pdf/2610.36694v1)

**作者:** Tuo An `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过为多任务流匹配策略构造任务特定的低秩高斯混合源，并在流路径上采用慢‑快插值调度，提升了少步推理下的多任务执行表现。

**💡 创新点**

①利用流逆映射反向样本拟合任务专属源；②基于高斯Optimal Transport识别扩展方向，设计沿该方向的慢‑快插值调度，二者共同实现了源和路径的任务化。

**🔧 技术方法**

任务特定MPPCA源拟合；高斯Optimal Transport与速度预测风险分析；流匹配（FM）训练；MiniLM语言指令选择器；一两步求解器。

**📊 数据集**

LIBERO‑10、LIBERO‑Plus、CALVIN D→D、CALVIN ABC→D、π_0.5 以及 Galaxea R1 Lite 实地多任务套件。

**📈 对比分析**

与标准FM、OT‑CFM、ReFlow、RF‑distill、MeanFlow、PriorGrad、MM‑FM、LeaP、Regression 等基线在 ID/OOD、少步推理、数据效率上进行对比；在 LIBERO‑10 上从 53.44% 提升至 69.67%，CALVIN D→D 从 1.90 提升至 2.27，推理延迟约为 1/6；实地任务整体成功率提升至 77.5% 对比 51.25%。

**⚠️ 局限性**

需离线构造任务源与调度且训练后保持固定；仅能选取预定义任务，无法泛化到新任务；动作片段长度固定；不支持增量更新或可变时长生成。

---

## 300. PsyJev: Psychological Analysis via System-1 Judgment

**arXiv ID:** 2610.36689 | [PDF](https://arxiv.org/pdf/2610.36689v1)

**作者:** Fangyuan Liu `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的心理学分析框架 PsJevBench，并提出了 System-1 直接概率判断模型 PsyJev 以及 Judgment‑Anchored RLCD（JA‑RLCD）微调方法，用于在同一条文本上基于问题进行多任务判断。

**💡 创新点**

将心理分析任务以问题‑答案形式统一，直接在合法答案集合上输出分布；提出 JA‑RLCD 在 logit 空间进行奖励优化，结合类平衡交叉熵、参考保持和问题组 Brier 奖励，显著提升多任务表现。

**🔧 技术方法**

System‑1 直接判读（不生成文本）+ Qwen3.5 backbone + 联合 SFT + JA‑RLCD（logit‑space Gaussian 扰动、Brier 基奖励、路径梯度、类平衡交叉熵、参考 KL 保持）。

**📊 数据集**

13 个现有心理学数据集，共 82 个问题，覆盖抑郁/焦虑、关系/自我、压力/困扰、认知、幸福感、危机等六大主题；使用协调的训练/验证/测试拆分。

**📈 对比分析**

与 18 个基准适配模型（专门分类器、描述式模型和生成式模型）及零射 Jev API 对比；在 2B/4B/9B 规模下，PsyJev 平均 F1 分别为 64.33 / 66.21 / 67.56，超越对应答案模型 4.01 / 1.71 / 3.88 点，4B/9B 版还击败最强外部基线；在未见数据 Dreaddit 上亦提升；Ablation 证明 JA‑RLCD 显著提升 3.65 点。

**⚠️ 局限性**

仅适用于文本自述，无法给出临床诊断；多任务间样本不平衡仍影响性能；未标注标签采用 mask，可能导致信息损失；在非 Web 文本或多语言场景尚未评估。

---

## 301. RoboSTD: Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring for Bimanual Learning

**arXiv ID:** 2610.36685 | [PDF](https://arxiv.org/pdf/2610.36685v1)

**作者:** Hongyu Wu `[一作]` (Beijing Academy of Artificial Intelligence), Guocai Yao `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用单臂演示数据通过正射镜像与LLM驱动的时空重构，零样本生成双臂监督，从而实现双臂机器人学习。

**💡 创新点**

创新点在于将几何对称镜像与大语言模型协同，用零样本方式跨越单臂与双臂的动作空间、协调与空间偏差缺口；同时引入结构化时空约束，提升协调性。

**🔧 技术方法**

主要技术包括：① 正射平面镜像映射（镜像观测、姿态、动作），② 机器人URDF与轴向对称分析以获取镜像变换；③ 使用GPT‑4.1生成臂分配、先后顺序与冲突约束；④ 通过约束拼接生成伪双臂演示；⑤ 在预训练语言条件策略上直接训练。

**📊 数据集**

使用了Open X‑Embodiment（单臂演示）、RoboTwin 2.0（10个模拟双臂任务）、AgileX ALOHA真实机器人（杯子收集、三明治制作等5个双臂任务）。

**📈 对比分析**

与单臂直接监督、Oracle（本地收集的双臂演示）进行对比；在RoboTwin 10任务上，RoboSTD平均成功率66.5%仅比Oracle低1.1pp；在真实任务中，RoboSTD在高协作任务上杯子收集数4.22/3.92、三明治完成度74%/72.5%且匹配Oracle成功率；同时显著降低右侧偏好与密度差距。

**⚠️ 局限性**

局限性包括：① 依赖机器人对称结构，非对称或多关节配置可能失效；② 对LLM生成约束的质量和可解释性敏感；③ 仅在任务空间与时序可被分解为独立单臂单元的场景下有效，复杂连续动作仍需进一步研究。

---

## 302. An Empirical Study of Undisclosed Charges in Hook-Based Decentralized Exchanges

**arXiv ID:** 2610.36682 | [PDF](https://arxiv.org/pdf/2610.36682v1)

**作者:** Kai Ma `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Uniswap v4与PancakeSwap Infinity上的Hook AMM生态进行大规模实证研究，量化了隐藏扣费的普遍性与经济成本。

**💡 创新点**

创新点在于首次将EVM字节码符号分析与历史结算重建相结合，系统识别并测算Hook导致的未披露扣费。

**🔧 技术方法**

采用EVM字节码解析、回退链解析、回调返回重建、Z3符号求解等技术，构建完整的收费识别与验证管道。

**📊 数据集**

使用截至2026年8月1日的Hook部署、池事件、交易回执等数据，涵盖7,873个Hook、50,011个池和2.96亿笔事件。

**📈 对比分析**

通过对约2.48M笔已重建交易的收费频率、单笔成本及与标准池费的比较，发现Hook扣费占比高达59%，平均单笔成本在以太坊约8.17美元、BSC约0.41美元，且在大多数情况下超过标准池费。

**⚠️ 局限性**

局限性包括仅覆盖四个平台、未计入未计价交易、可能漏检复杂路径、以及Hook收款的上限可能低于实际总收入等。

---

## 303. Design of DoF-Robust MIMO via Generalized Pinching Antennas

**arXiv ID:** 2610.36670 | [PDF](https://arxiv.org/pdf/2610.36670v1)

**作者:** Zhiguo Ding `[一作]` (Nanyang Technological University), H. Vincent Poor `[通讯]` (Princeton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计了一种基于通用压缩天线（pinching antenna）的MIMO系统，能够在用户位置差异较大时保持空间自由度；

**💡 创新点**

创新点在于将压缩天线灵活部署在用户附近，使多用户通道矩阵近似对角化，从而显著降低秩缺失风险并提升系统鲁棒性；

**🔧 技术方法**

采用随机矩阵理论对独立非同分布（i.n.i.d.）Wishart矩阵的特征值分布进行解析，推导了闭式的秩缺失概率与出度概率；同时利用零迫束（ZF）波束成形、仿真随机用户布置和高SNR近似分析；

**📊 数据集**

实验数据基于随机用户位置的仿真（1D线性、2D平面分布），未使用公开数据集；

**📈 对比分析**

通过与传统固定位置天线的MISO基准进行比较，使用秩缺失概率、出度概率和期望总速率等指标；结果显示在用户距离差异大时，压缩天线系统在出度概率上可提升至原来的数倍，且在多用户情况下可实现显著的速率提升；

**⚠️ 局限性**

研究仅限于两用户下行MISO场景，未考虑上行、多用户多天线及CSI误差、波导衰减等实际因素，且压缩天线的动态部署与硬件实现细节仍需进一步探索。

---

## 304. Log-Derivative Trick for Parametric Markov Chains

**arXiv ID:** 2610.36655 | [PDF](https://arxiv.org/pdf/2610.36655v1)

**作者:** Arthur Gall `[一作]` (National Institute of Informatics), Thomas Seiller `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于轨迹采样的无偏梯度估计器，用于优化终止参数化马尔科夫链（pMC）的期望奖励或到达概率；该估计器基于测度论推广的log‑导数技巧，完全避免了全状态空间的构造和线性系统求解。

**💡 创新点**

创新点包括：①将log‑导数技巧在一般σ‑有限可测空间上用Radon–Nikodym导数形式化；②得到的采样梯度估计器在几乎必然终止和标准正则性下无偏；③实现了对离散、连续甚至混合状态空间的统一处理；④在实验中显著超越传统线性求解器和近似方法，提供更快的梯度计算和更高效的参数合成。

**🔧 技术方法**

技术手段：测度论与Radon–Nikodym导数；轨迹采样与蒙特卡罗估计；log‑导数（score function）技巧；符号求导与稀疏线性代数（Eigen）；C++实现与稀疏矩阵/向量运算；梯度下降优化（SGD、AdamW、投影方法）。

**📊 数据集**

数据集：合成离散pMC（稀疏图、1阶多项式转移概率，最多5000状态、100个参数）；连续状态pMC（T3单参数、T4双参数）；两者均通过Python生成并导出Prism文件；实验使用AWS EC2单核实例进行对比。

**📈 对比分析**

与方法对比：exact linear‑solver (LU、Krylov)、Jacobi近似、基于重要采样的均匀采样；在R1–R5实验中，采样梯度器在相同时间内完成更多梯度步数，导致比精确求解器更快的奖励提升；在离散模型中，采样方法在10⁵样本下近似原始梯度，cosine相似度≥0.998；在连续模型中实现了理论最优奖励。

**⚠️ 局限性**

局限性：采样导致方差，需要较多样本以获得高精度；依赖几乎必然终止和正则性假设；不保证全局最优；在参数维度极高时，梯度估计仍可能受噪声影响；对循环结构的长路径可能产生高方差。

---

## 305. Remembering Is Not Enough: Refining Memories and Learning When to Use Them in Multi-Agent Systems

**arXiv ID:** 2610.36652 | [PDF](https://arxiv.org/pdf/2610.36652v1)

**作者:** Yipeng Hao `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DiscernMem框架，解决多智能体系统中记忆缺陷与误用的判别与修复问题

**💡 创新点**

创新点在于将记忆内容的缺陷修复与记忆使用的上下文边界学习分离，采用配对重放评估和元记忆引导的使用边界

**🔧 技术方法**

技术包括Defect‑Aware Memory Refinement（DMR）配对重放、Boundary‑Learning Meta‑Memory（BLM）使用边界学习、LLM无训练的规则诊断与修复、以及检索式记忆匹配

**📊 数据集**

数据集涵盖四个基准：TriviaQA、PopQA（问答）、KodCode（代码生成）以及AgentBoard PDDL规划任务

**📈 对比分析**

在AutoGen、CAMEL、MacNet三种多智能体框架上对比多种基线，DiscernMem在平均得分上均居首，尤其在代码生成和规划任务上显著提升（最高可达+4.32分）

**⚠️ 局限性**

局限性包括依赖可重放的任务、仅针对单一角色的规则缺陷修复、对交互式多规则组合缺乏诊断、以及在二元问答任务中证据收集不足导致性能波动

---

## 306. When Is a Cheap Fairness Edit Safe? Screening Post-Hoc Edits in Two-Sided Matching Markets

**arXiv ID:** 2610.36649 | [PDF](https://arxiv.org/pdf/2610.36649v1)

**作者:** Moirangthem Tiken Singha `[一作]` (Dibrugarh University), Rabinder Kumar Prasada `[通讯]` (Dibrugarh University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在两侧匹配系统中，通过闭式线性投影在推理阶段减少受保护属性影响、无需重新训练模型的方法，并给出匹配质量损失的理论上界；

**💡 创新点**

首次在两侧匹配机制上提供可预先验证的福利损失上界，兼具可解释的可编辑强度与公平性阈值，并设计了预部署筛选规则；

**🔧 技术方法**

闭式线性投影（类似LEACE）+ 统计学上界推导 + 预部署可编辑强度屏蔽；

**📊 数据集**

三类实验数据：合成垂直市场、基于 LSAC 法学院、巴西 ENEM 考试、Speed Dating 互评数据；

**📈 对比分析**

与现有迭代零空间投影、RLACE、分数标准化、拒绝选项等后处理方法对比；在合成市场上通过屏蔽规则实现 PASS/ KILL，且无误 PASS；在实测市场中闭式投影在 90%+ 样例内保持在可接受范围；相较于其他方法，既保留公平度又保持福利更高；

**⚠️ 局限性**

仅适用于垂直或近垂直的完整一对一匹配，需先验假设共同排序；不支持多维投影或非线性属性信息；对质量分布的交换性假设可能不成立；

---

## 307. Dream-COP: Guiding Algorithm Discovery for Combinatorial Optimization Problems in an Experience World

**arXiv ID:** 2610.36618 | [PDF](https://arxiv.org/pdf/2610.36618v1)

**作者:** Kuan Xu `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Dream-COP方法，利用LLM在过去搜索树构成的经验世界中进行政策学习，以指导组合优化问题的算法发现。

**💡 创新点**

创新点在于将历史搜索树视为可重放的经验世界，让可执行的搜索政策在内部环境中先行评估（policy replay），并通过奖励函数平衡算法质量与搜索成本，从而自适应地分配预算、决定分支扩展与停止。

**🔧 技术方法**

技术主要包括：大语言模型（如GPT‑6‑Luna、GPT‑4o‑mini）用于生成搜索政策和程序；经验树重放（replay）机制；基于多步质量平均的回放目标函数；自适应预算分配与停止决策；跨问题政策迁移。

**📊 数据集**

在五个组合优化问题上评测：构造性旅行商问题（TSP）、背包问题（KP）、在线箱装问题（OBPP）、流排调度（FSSP）以及二维箱装问题（2D‑BPP）；使用标准公开数据集（TSPLIB、OR‑Tools实例、Taillard、BWMV等）。

**📈 对比分析**

与多种LLM‑AHD基线（FunSearch、EoH、ReEvo、MCTS‑AHD等）以及参考求解器（LKH、OR‑Tools）进行对比。实验显示Dream-COP在TSP、KP、OBPP、FSSP和2D‑BPP上均取得更优或相近的解，并显著减少搜索请求（最多减少42.5%），在跨问题迁移时也能提升性能。

**⚠️ 局限性**

局限包括：需要足够丰富的历史搜索树才能形成有效经验世界；回放成本仍受树深度和分支数量限制；LLM的生成质量和推理能力限制了搜索范围；在极大规模或高度异质的问题上，经验迁移效果可能下降。

---

## 308. Balanced State Subspaces for Mamba-2 State Compression

**arXiv ID:** 2610.36617 | [PDF](https://arxiv.org/pdf/2610.36617v1)

**作者:** Shang Tingyu `[一作]` (Shaanxi Normal University), Cao Wenfei `[通讯]` (Shaanxi Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后训练的压缩方法BASS（Balanced Subspace Selection），通过对Mamba‑2层的递归状态进行平衡截断，直接在固定的标量‑身份转移上实现精确的子空间旋转，从而将状态维度显著压缩。

**💡 创新点**

创新点在于：①利用完整的经验可控性与可观测性Gramians而非仅对角分数，构造最优的联合子空间；②借助Mamba‑2的标量‑身份转移，使得子空间变换可精确保留输出；③引入HARP层级自适应分配策略和检索保护阈值，进一步提升压缩后模型的语言建模与检索性能。

**🔧 技术方法**

使用的技术包括：平衡截断（Balanced Truncation）、经验Gramians估计、Hankel奇异值谱、子空间旋转、层级分配（HARP）、检索敏感度探测、以及可选的整数量化。

**📊 数据集**

实验数据集主要为WikiText‑2（用于语言建模）和LAMBADA（用于检索测试）。

**📈 对比分析**

与GHOST、Saliency、Magnitude、随机、均匀等基线比较，BASS+HARP在50%状态保留下平均仅+0.3 ppl，优于GHOST的+1.6 ppl；在25%保留下仍保持明显优势；检索准确率也恢复至几乎无损；同时校准时间与内存低于GHOST。

**⚠️ 局限性**

局限性包括：在极端压缩比例（如25%）下仍有显著精度下降；方法仅适用于具有标量身份转移的Mamba‑2，难以直接迁移到非旋转不变的模型或复杂状态；需要额外的校准步骤，且未将旋转融合进解码内核，导致批量1时有轻微延迟。

---

## 309. Beyond the 1D Reality-Virtuality Continuum: A 2D Framework for Seven Extended Reality Paradigms

**arXiv ID:** 2610.36674 | [PDF](https://arxiv.org/pdf/2610.36674v1)

**作者:** Rajashekhar Vachiravelu Saminathan `[一作]` `[通讯]` (Indian Institute of Technology), Rajashekhar Vachiravelu Saminathan (Indian Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一个二维Reality‑Virtuality框架，用以描述和比较七种扩展现实范式（VR、AR、MR、AV、PR、AnR、DR），并将其映射回传统的一维连续体。

**💡 创新点**

创新点在于将信息增减（添加、删除、转化）作为第二维度，突破单一实-虚维度的限制，为新兴XR技术提供统一且可视化的分类与比较工具。

**🔧 技术方法**

主要技术包括Unity3D虚拟环境构建、Meta Quest 3 MR显示、QR码路径引导、CAVE-E实验环境、机器人视觉与路径规划、以及基于图像处理的减噪/增补技术。

**📊 数据集**

数据集来源于自行搭建的地震场景实验数据，包含VR沉浸体验、AR二维码引导、MR软体机器人控制、AV探测训练、DR视觉冗余剔除、PR机器人通信、AnR虚拟障碍路径规划等七种XR实现的实测录像与图像。

**📈 对比分析**

采用定性跨范式对比方法，依据物理-虚拟关系、信息操作类型和主观特征进行分类与定位；实验结果显示二维框架能清晰区分相似位置但信息操作不同的技术，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括：仅基于作者主观经验的定性分析，缺乏大样本量的定量验证；信息增减维度未涵盖语义或认知后果；实验聚焦于地震场景，未验证在其他应用领域的普适性。

---

## 310. Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration

**arXiv ID:** 2610.36672 | [PDF](https://arxiv.org/pdf/2610.36672v1)

**作者:** Chenyu Guo `[一作]` (Harbin Institute of Technology), Shuhan Qi `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种名为ReliMorph的三维可变形图像配准框架，利用对应可靠性引导空间上下文路由和残差速度门控，实现从粗到细的对称光流更新。

**💡 创新点**

创新点包括：①将局部匹配的概率分布与期望偏移量作为可靠性特征，①将此可靠性与逆一致性反馈联合用于专家路由与速度门控；②使用正门底值限制速度衰减，防止过度抑制；③在每一金字塔层使用一致性反馈来指导下一层的路由与调制。

**🔧 技术方法**

技术手段包括：共享解码器+轻量化模态残差适配器、局部相关分布统计（熵、峰值、边际、期望偏移）、三路专家（局部细节、轴向、膨胀）并行卷积、软门控、稳态速度积分+尺度平方、逆一致性与雅可比正则化，以及多尺度对齐与自监督置信度监督。

**📊 数据集**

在三个公开数据集上评估：IXI（脑T1‑T1配准）、OASIS（脑T1‑T1配准）和AbdomMRCT（腹部CT→MR配准）。

**📈 对比分析**

与包括TransMorph、CorrMLP、LGANet++、AC‑CAR、SACB‑Net、SyN、LapIRN、SYMNet、ShiftMorph等多种基线比较，ReliMorph在所有数据集上实现最高DSC、最低HD95和极低折叠率，且推理时间约0.8 s、显存1.8 GiB，优于多数对手。

**⚠️ 局限性**

局限性：仍不能保证完全的逆一致性与无折叠的离散变换；对极端模态差异或极大变形时可能仍需更强的上下文或正则化；门控参数（如正底值）需手工设定，可能影响鲁棒性。

---

## 311. TACO-Graph: Automated Construction of Text-Atributed Graphs

**arXiv ID:** 2610.36650 | [PDF](https://arxiv.org/pdf/2610.36650v1)

**作者:** Amiri Hayes `[一作]` (Massachusetts Institute Of Technology), Kristina Wicke `[通讯]` (New Jersey Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TACO-Graph框架，用决策树方法自动学习从表格数据生成文本属性图（TAG）时最优的节点、边和文本属性组合；

**💡 创新点**

创新点在于把图构造视为可枚举的离散搜索空间，利用小样本代理评估快速生成评分，再用可解释决策树进行全空间排名，从而避免专家手工试错；

**🔧 技术方法**

主要技术包括：图结构枚举与约束、基于GraphSAGE的GNN代理评估、基于RAGAS的GraphRAG代理评估、决策树回归与Spearman相关/树编辑距离验证；

**📊 数据集**

使用了五个多属性表格数据集：ArXiv、Sporting Goods（Sports）、Electronics、Toys（Amazon Reviews）以及Books‑History；

**📈 对比分析**

通过与全规模模型对比，TACO-Graph在节点分类任务中决策树预测的Spearman相关率平均>0.9，GraphRAG任务平均>0.8；在评估成本上，节点分类仅需3~9倍全规模运行时间即可得到完整构造空间排名，而GraphRAG成本更高但仍提供结构化覆盖；

**⚠️ 局限性**

局限性包括：对任务-数据对的依赖强，跨数据集迁移效果有限；当构造空间过小或性能分布宽度不足时，决策树难以学习有效规则；GraphRAG评估成本高，代理评分仍受限于问题生成质量。

---

## 312. When Does Perturbation Consistency Add Information Beyond Softmax Confidence? Theory and Evidence for the Decision Consistency Framework

**arXiv ID:** 2610.36640 | [PDF](https://arxiv.org/pdf/2610.36640v1)

**作者:** Amit Kalitaa `[一作]`, Moirangthem Tiken Singh `[通讯]` (Dibrugarh University Institute of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在六种轻微扰动下的决策一致性指标（DCF），评估其在误分类检测、分布漂移与对抗攻击中的有效性。

**💡 创新点**

证明了在干净数据上一致性与最大软最大概率（MSP）冗余，并给出基于margin的无标签分布式不反转证书，发现一致性在噪声与对抗环境下可提升检测。

**🔧 技术方法**

采用了标签稳定性、置信度偏差、熵、能量、特征空间一致性等分数，并引入分布式证书与分布式无标签校准。

**📊 数据集**

在CIFAR-10、STL-10、Intel-Scene、OCTMNIST等图像分类数据集以及CIFAR-10-C、SVHN、CIFAR-100等外部分布上进行实验。

**📈 对比分析**

与MSP、margin、能量、Mahalanobis、TTA等基准相比，DCF在干净数据上AUROC普遍低于MSP，成本提高8-15倍；但在噪声与对抗攻击下AUROC提升至0.83，优于MSP。

**⚠️ 局限性**

实验受限于轻微扰动、固定模型架构、缺乏人类标签验证、对抗攻击适配有限，且证书仅在交换性假设下有效。

---

## 313. Agar: What-If Analysis for Microservice Applications

**arXiv ID:** 2610.36636 | [PDF](https://arxiv.org/pdf/2610.36636v1)

**作者:** Saarth Deshpande `[一作]` (University of Texas at Austin), Venkat Arun `[通讯]` (University of Texas at Austin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个针对微服务应用的 counterfactual 性能分析工具，利用简单可解释的区间模型为每个组件学习包含所有观测的上下界，进而通过模型组合回答不同工作负载、资源分配或 SLO 目标下的系统行为问题，并实现自动化调度、成本预测和性能诊断。

**💡 创新点**

创新点包括：①将预测精度转化为置信区间，保证模型始终 100% 准确并显式表达不确定性；②在所有观测都满足的前提下联合最小化端到端不确定性，并证明阈值选择 NP-hard；③提出 log(n)+1 近似的贪心算法来选取最优 SLO 安全阈值；④通过模型组合实现端到端 what‑if 问题，无需重新实验。

**🔧 技术方法**

采用分层建模（利用率模型、SLO 模型、RPS 预测模型）、区间回归、Charnes‑Cooper 线性化求解最小复制数、贪心集合覆盖近似算法、线性规划求解阈值，并仅使用监控指标实现数据驱动学习。

**📊 数据集**

在三款真实微服务应用（如 Online Boutique、TeaStore 等）上收集的生产/实验工作负载轨迹，包含每个端点请求率、延迟、CPU/内存利用率等；并对 AWS 与 CloudLab 两种硬件环境下的数据进行实验。

**📈 对比分析**

与当前主流水平自适应器（HPA、Erlang、AutoThrottle 等）进行对比，在相同 SLO 条件下，所学模型的自动化器能将 CPU 分配量降低多达 36.8% 且仍满足 SLO；在成本预测、启动时间优化和定位性能瓶颈等任务上也提供更精确的指导，且模型误差在大规模组件时保持可控。

**⚠️ 局限性**

假设应用基本行为稳定，需在代码或硬件变更后重新训练；对噪声或离群值敏感，可通过引入 slack 处理；目前仅支持水平扩缩，未涵盖垂直扩缩；对极端突发负载或复杂交互的建模仍有限。

---

## 314. Uniformity Testing for the Monomer-Dimer Model

**arXiv ID:** 2610.36630 | [PDF](https://arxiv.org/pdf/2610.36630v1)

**作者:** Joy Qiping Yang `[一作]` (University of Sydney), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在已知图 G 上，对从未知重参数化的单分子-偶配模型（匹配的 Gibbs 分布）进行均匀性检测，确定分布是否等同于所有匹配的均匀分布。

**💡 创新点**

首次证明在高维结构化分布（匹配集合）上实现了比完整学习更低的样本复杂度：上界为 OΔ(√m/ε²)，下界证明了匹配数 m 的平方根是最优的，显示了学习与测试之间的二次分离。

**🔧 技术方法**

利用条件统计量的局部可检验性、约束下的熵近似分解、谱独立性、Poincaré 不等式、以及有限度图上的 FPTAS 计数匹配的技术；构造了二次统计量并证明其方差受控，以实现 √m 的样本上界；构造了“棋盘”扰动的匹配集合并通过二次矩和混合分布分析给出信息理论下界。

**📊 数据集**

无实验数据集；全部为理论分析与算法设计。

**📈 对比分析**

对比学习所需的 O(m/ε²) 样本量，测试只需 O(√m/ε²) 样本；在有界度 Δ 的图族中实现了多项式时间内的检验，且误差概率可通过投票放大至 1‑δ。

**⚠️ 局限性**

限制在已知图且度数受限（Δ 为常数）下；常数因子和 Δ 的依赖尚未最优；对一般稠密图的实现仍需改进；算法依赖于对匹配数的 FPTAS，实际实现可能受限。

---

## 315. Sybil-Proofness of Generalized Adaptive Clinching and Implications for Blockchain Mechanism Design

**arXiv ID:** 2610.36629 | [PDF](https://arxiv.org/pdf/2610.36629v1)

**作者:** Chengkai Zhang `[一作]` (Rutgers University), Qishen Han `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在机制设计与区块链去中心化金融交叉的背景下，首先证明了自适应临时拍卖（adaptive clinching auction）在允许伪身份（Sybil）拆分且预算不被超报的策略空间内仍保持Sybil鲁棒的激励兼容性；随后在此框架基础上设计了一种全新的两边自动做市商（AMM）机制，该机制既支持两种资产的双向交易，又能对池的供应曲线做动态调整，证明其在满足精确结算、个人理性、Sybil鲁棒激励兼容和弱Pareto最优的条件下可实现。

**💡 创新点**

创新点主要有：①提出通用的目标基准化（target‑based）自适应临时拍卖框架，推广原始拍卖并证明其在Sybil攻击下的鲁棒性；②在此框架下将自适应临时拍卖扩展到双向AMM，首次将供应曲线的动态性与两边匹配结合，获得在预算不超报且允许有限Sybil的环境下可行的Sybil鲁棒AMM；③给出了新的不可行性定理，说明在允许预算超报或完全可执行的两边市场中无法同时实现完整Pareto最优与Sybil鲁棒激励兼容。

**🔧 技术方法**

技术上主要采用：目标基准化自适应临时拍卖的连续动力学与微分代数方程，利用凸性、单调性以及价格时钟的几何构造；对两边市场引入“最小侧填充”概念，将少数侧订单折叠到主供给曲线；在证明中大量运用边界行为、供应曲线的右导数、以及对Mark‑to‑Market收益的积分与比较技巧。

**📊 数据集**

该工作纯粹是理论分析，未使用任何外部数据集或实验验证；所有结果均基于数学证明与理论模型。

**📈 对比分析**

与现有文献比较时，作者在论文中指出：①相较于Dobzinski等人提出的自适应临时拍卖，本文的目标基准化框架更通用且在Sybil环境下得到严格证明；②相较于Shi等人关于不可行性的结论，本文在预算不超报且允许有限Sybil的前提下给出可行机制；③相较于现有的AMM机制，本文提供了对两边市场的完整处理和对池状态的自适应，使得机制既满足Sybil鲁棒激励兼容，又满足弱Pareto最优，弥补了传统AMM无法抵御MEV攻击的局限。性能方面由于是理论模型，未给出实验指标，重点在证明性质与可行性上。

**⚠️ 局限性**

局限性包括：①策略空间被限制为预算不被超报且仅允许同向拆分；若允许预算超报或完全自由的Sybil攻击，机制将失效；②只能实现弱Pareto最优，完整Pareto最优与Sybil鲁棒兼容被证明不可行；③机制实现较为复杂，对实际区块链智能合约部署存在技术与效率挑战；④假设外部无法审计或强制预算约束的前提下，若存在闪电贷等攻击手段仍需额外补偿。

---

## 316. Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction

**arXiv ID:** 2610.36624 | [PDF](https://arxiv.org/pdf/2610.36624v1)

**作者:** Taeyoung Kim `[一作]`, Jinwoo Shin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于运动引导的非对称视觉预测任务MoGAP，用于改进世界动作模型（WAM）的视觉监督，使其更适合机器人控制。

**💡 创新点**

创新点是将动态区域视为预测目标、静态区域视为上下文，并通过在视频扩散模型的潜在空间中分配不同噪声水平，自动聚焦动态信息，无需外部教师或额外模态。

**🔧 技术方法**

使用技术包括：预训练的视频扩散模型、潜在差分得到的运动得分自监督、噪声分配与非对称训练路径、流匹配损失以及注意力掩码等。

**📊 数据集**

数据集：RoboCasa GR-1 Tabletop（24桌面抓取任务）、RoboCasa-Kitchen（24厨房任务，单臂机器人），以及真实世界的OpenArm humanoid实验。

**📈 对比分析**

方法对比：与Cosmos3-Nano、Fast-WAM、VLA等基线进行比较；MoGAP在所有基准上平均成功率提升约10个百分点（Cosmos3从52.3%提升至62.3%），在pick‑and‑place任务提升12.5pp；在Fast-WAM上提升约8.7pp；在OpenArm上对目标位移任务也表现出显著提升；并且MoGAP超过了使用外部教师的监督方法。

**⚠️ 局限性**

限制：训练时需要额外的token，训练时间比原始模型高约1.26倍；模型对噪声分配参数的敏感性需进一步探究；实验仍主要集中在抓取与搬运任务，未覆盖更复杂的动态交互。

---

## 317. RadarCom: Unlocking Simultaneous Communication and Sensing on Commodity Automotive mmWave Radars

**arXiv ID:** 2610.36622 | [PDF](https://arxiv.org/pdf/2610.36622v1)

**作者:** Yi Wei `[一作]` (Saint Louis University), Nan Cen `[通讯]` (Saint Louis University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了在不改动硬件、无需额外通信基础设施的前提下，利用现有毫米波汽车雷达实现双向雷达间通信，同时保持原有的感知功能。

**💡 创新点**

创新点在于将相互干扰视为通信载体，提出斜率不匹配的探测、适应性脉冲同步、基于距离FFT的差分编码以及感知感知干扰抑制四大机制，使得同一波束可实现感知与通信共存。

**🔧 技术方法**

采用软件定义的波形重构技术、斜率不匹配解调、最小二乘同步估计、差分频移编码、跨脉冲重建和子帧一致性检测等算法，并在 TI AWR2243 mmWave 雷达平台上实现。

**📊 数据集**

使用 TI AWR2243 BOOST 雷达与 DCA1000EVM 捕获卡进行实验，数据主要来自实验室及道路实测的两车对车通信场景，覆盖静态、动态、不同障碍与多雷达干扰等多种情形。

**📈 对比分析**

与传统 ISAC 与雷达通信方案相比，RadarCom 在保持感知 F1 率达 80.5% 的同时实现 5.55 kbps 的有效数据速率，误符号率低于 1.5%，实验中在多干扰与不同部署环境下均表现出较高的同步成功率（≥ 90%）和鲁棒性。

**⚠️ 局限性**

局限主要包括：① API 重配置延迟导致帧周期长、上行速率受限；② 仅适用于视线（LoS）传播，非 LoS 与严重遮挡时性能下降；③ 现有实现为单向时分双工，未实现多用户/多链路完整多址；④ 对雷达硬件的漂移和相位噪声假设仍较为理想，需在更大规模部署中进一步验证。

---

## 318. Mimetic Embodiment Transfer of Human-Humanoid Interaction from In-the-Wild Videos

**arXiv ID:** 2610.36615 | [PDF](https://arxiv.org/pdf/2610.36615v1)

**作者:** Nghia Le `[一作]`, Anh Nguyen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模仿化身转移的框架，能够从现实人类-人类互动视频中提取可物理可执行的全身运动，并将其中一方转化为不同机器人体型（Unitree R1 Edu、HighTorque Mini-hi）的交互动作，构建了约 500,000 帧的多场景人机交互数据集。

**💡 创新点**

创新点在于：① 引入交互语义与距离加权矩阵，显式保留两主体间的空间关系和接触语义；② 结合全身形状拟合与足部接触一致化提升提取动作的物理真实性；③ 通过后期平衡与碰撞抑制优化，确保转移后动作既符合机器人运动学，又保持交互语义；④ 通过该框架生成的数据集可直接用于人机交互生成与控制的基准评测。

**🔧 技术方法**

使用技术包括：SMPL 3D 姿态重建与全身形状拟合、足部接触一致化优化、逆运动学初始化、交互矩阵优化（Joint-Relative Direction & Anatomical Offset）、权重矩阵（语义与距离权重）、后期自平衡与碰撞抑制、DeepMimic/​BeyondMimic 控制器训练、ONNX 运行时执行。

**📊 数据集**

数据集来源：从 YouTube、TikTok、Facebook 等公开平台抓取 1,200 条双人互动视频，提取约 500K 帧，覆盖共动、社交、运动、格斗等多种交互；随后将这些姿态映射到 Unitree R1 Edu（26-DoF）和 HighTorque Mini-hi（23-DoF）两款机器人，形成公开可用的人机交互数据集。

**📈 对比分析**

与 IK、GMR、PAIR 等基线相比，在两款机器人上 MPJPE、IRD、CF1 均取得最优（Unitree：6.31 cm / 2.84 cm / 81.97%，Mini-hi：7.58 cm / 3.67 cm / 78.15%）。在生成任务中，Duolando 在 FID_k、FID_cd 和接触率上表现最佳（例如 24.96 / 9.07 / 51.75%）。在真实机器人上实现无脚滑、无碰撞、可保持自平衡的交互演示。

**⚠️ 局限性**

局限性：① 机器人与人类的体型差异仍导致部分交互（如 Mini-hi 的肩宽/手短）无法完全实现；② 当前部署仅依赖预录序列，缺乏实时对人类动作的适应与在线更新；③ 对极端动态或大幅度姿态变化的鲁棒性尚待进一步验证。

---

## 319. Quantifying and Prioritizing Stealthy Source-Manipulation Threats Against Multi-Source Anomaly Detectors

**arXiv ID:** 2610.36693 | [PDF](https://arxiv.org/pdf/2610.36693v1)

**作者:** Indraveni Chebolu `[一作]` (Centre for Development of Advanced Computing), Ch A S Murty `[通讯]` (Centre for Development of Advanced Computing)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于任务条件的可视性边际（task-conditioned visibility margin）来评估跨源异常检测器对恶意操纵的鲁棒性，并将该指标用于优先审计最易被利用的数据源子集。

**💡 创新点**

创新点在于：①给出闭式表达的最小检测器证据与任务损害之间的比例；②将该指标与任务梯度、检测器信息矩阵关联，揭示检测器在不同任务下的可见性差异；③将边际作为快速审计优先级排序工具，在多源融合系统中能比随机或基于报警阈值的排序更快发现潜在失败。

**🔧 技术方法**

使用的技术包括：雅可比矩阵与伪逆求解、欧氏投影、梯度攻击（Task Gradient、Generic Stealth、Task-Conditioned）和SLSQP有限攻击验证、PCA/重构/学习型检测器（GDN、TranAD）的残差分析，以及统计评估（Spearman相关、Bootstrap差异）。

**📊 数据集**

实验数据集：①UCI Hydraulic Systems工业水利遥测数据（17通道，2,205周期）用于评估PCA、GDN、TranAD检测器；②ICRISAT作物产量面板（5源）用于评估线性与非线性检测器的审计效果。

**📈 对比分析**

与随机排序、报警阈值余量、任务梯度归一化排序等方法比较，边际指标在10次审计内平均发现4个失败，随机为0，报警阈值为2，任务梯度为2-3，整体性能相当甚至略优，并能识别弱可见方向。攻击实验表明，任务条件攻击在低预算下比泛化隐蔽攻击造成更多误报且更易逃避报警。

**⚠️ 局限性**

局限性：边际是局部一阶近似，无法预测大预算或非线性全局攻击的真实损害；与全局状态-攻击耦合相关性弱；实验仅覆盖静态多源系统，未考虑动态过程约束、时间序列特性或物理模型；仅评估了特定检测器与任务模型，未验证更广泛的架构。

---

## 320. Comparative Analysis of Large Load Interconnection Requirements Across U.S. Transmission Entities

**arXiv ID:** 2610.36645 | [PDF](https://arxiv.org/pdf/2610.36645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 321. From Biological Curves to Soft Modular Robots: Learning-Based Inverse Design with Uncertainty-Guided Refinement

**arXiv ID:** 2610.36681 | [PDF](https://arxiv.org/pdf/2610.36681v1)

**作者:** Dixiang Wang `[一作]`, Luyang Zhao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于学习的 real‑to‑sim‑to‑real 框架，将动物曲线映射为柔性模块化软机器人配置，并实现逆向设计与不确定性引导的候选细化。

**💡 创新点**

创新点包括：1) 将物理模型生成的数据与学习前向代理结合，实现数亿倍加速；2) 采用两阶段逆向网络训练（监督 + 端到端细化），解决混合离散‑连续设计空间和多解逆问题；3) 引入不确定性阈值+候选搜索，局部搜索而非全局枚举，提高准确率且保持极低计算成本。

**🔧 技术方法**

核心技术：多层感知机前向代理、Softmax 连续化的逆向网络、两阶段训练（交叉熵+回归 + 前向代理细化损失）、不确定性估计（logit gap）+候选搜索、标准化处理、端到端反向传播。

**📊 数据集**

数据集：约 5 百万条样本（115k 不同缠线模式），每条包含线程模式、目标缆长和弯曲角向量；测试集 5k 样本来自未见的缠线模式。数据通过物理仿真生成并去除无效/不稳定样本。

**📈 对比分析**

与原始 SoftSnap 物理搜索、kNN、CNN、XGBoost、Transformer 等方法对比：前向代理在测试集 RMSE 0.0077 rad，推理时间 0.0008 ms；逆向方法在候选细化后成功率 93.18%，平均时长 4.18 ms，速度相较原始方法提升 4.76×10^5 倍，前向代理速度提升 1.9×10^8 倍。

**⚠️ 局限性**

限制：仅考虑准静态形状，不包含重力、摩擦、动力学等外部力的影响；不适用于动态运动预测和更复杂环境交互。

---

## 322. SAGE: Semantic-Augmented Geometric Experts for Point Cloud Registration

**arXiv ID:** 2610.36658 | [PDF](https://arxiv.org/pdf/2610.36658v1)

**作者:** Tomoyasu Shimada `[一作]` (NTT Inc), Ryuichi Tanida `[通讯]` (NTT Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SAGE 框架，将冻结的预训练语义表征与语义引导的 Mixture of Experts（SG‑MoE）集成到点云配准流程中，用于低重叠情形。

**💡 创新点**

创新点在于使用冻结语义表示来引导专家路由，使每个专家专注于不同的语义‑几何模式，消除几何歧义；并采用两阶段特征提取融合语义与几何。

**🔧 技术方法**

技术包括 Sonata 语义编码器（冻结）、KPConv 结构化卷积、GeoTransformer Transformer、语义引导的 SG‑MoE、结构感知估计（LGR）以及 RANSAC 等。

**📊 数据集**

使用 3DMatch、3DLoMatch 以及 KITTI 数据集进行实验。

**📈 对比分析**

与 GeoTransformer、RoITr、CoFiNet、Predator 等单通道方法以及迭代方法 PEAL、PSReg 进行对比；在 3DLoMatch 上取得最高注册召回率，在 3DMatch 上保持竞争力，结构感知估计下进一步提升召回。

**⚠️ 局限性**

局限性包括受 RANSAC 随机采样限制，在高重叠 3DMatch 场景下 IR 仍略低，以及对预训练语义模型质量的依赖。

---

## 323. PDE-Based Bounds on Optimal Exponents in Active Simple Hypothesis Testing

**arXiv ID:** 2610.36657 | [PDF](https://arxiv.org/pdf/2610.36657v1)

**作者:** Sushant Vijayan `[一作]` (Plaksha University), Himadri Mandal `[通讯]` (Indian Statistical Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对主动简单假设检验问题的最优误差指数的全局上下界，并通过对Hamilton–Jacobi–Isaacs PDE粘性解的比较与逼近实现；

**💡 创新点**

将PDE可视化与可变凸/凹逼近相结合，生成以凸函数、凸集与Hamiltonian松弛为参数的灵活上下界，显著推广了既往的Chernoff信息上界与非自适应下界；

**🔧 技术方法**

利用粘性解的比较原理、Hopf与Hopf–Lax公式、Legendre–Fenchel共轭及Hamiltonian松弛等偏微分方程与凸分析工具；

**📊 数据集**

无具体实验数据，纯理论推导；

**📈 对比分析**

与文献中的非自适应Chernoff上界、B阶段Stackelberg游戏上界等经典结果进行比较，理论上证明在多假设情况下可实现更严格的误差指数下界，实验验证未给出；

**⚠️ 局限性**

部分上界与下界尚未完全证明最优性，具体逼近实现仍需进一步研究；对实际采样策略的可实现性与计算复杂度分析不足。

---

## 324. Whether Is Not When: Separating Occurrence from Timing in Polymarket Prices

**arXiv ID:** 2610.36639 | [PDF](https://arxiv.org/pdf/2610.36639v1)

**作者:** Sizheng Fan `[一作]` (University of International Business and Economics), Rong Zhao `[通讯]` (Shenzhen University of Information Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Polymarket 价格进行大规模量化，评估其作为早期警报的有效性，区分价格中关于事件发生与发生时点的信息。

**💡 创新点**

提出将价格拆分为“是否发生”与“何时发生”两部分，使用分割 log score、利润拆分方法；同时用全梯子读取实现更好短期警报。

**🔧 技术方法**

分割 log score、利润拆分；等距回归估计事件日期分布；洗手交易检测；基于梯子的一致性评估；基准模型（ROC、AP、Brier）对比。

**📊 数据集**

全链交易记录 1.24 B 交易，2.37 M 市场，208 个事件的 Web 报告，338 个期限梯子，9.3 M 交易、491 K 账户。

**📈 对比分析**

与单一合约、π_t、学习模型比较；全梯子在 7 天警报的 ROC 0.88、AP 0.30；单合约 0.85、AP 0.25；π_t 最差 AP 0.17；学习模型略低。

**⚠️ 局限性**

事件时间精度受 LLM 搜索限制，样本量有限，账户与个人可能重叠，未解释价格背后的动机，仅针对 Polymarket 的地缘政治事件。

---

## 325. Polarity--Time Interaction and Multi-Level Reliability for Event-Based Spacecraft Pose Estimation

**arXiv ID:** 2610.36625 | [PDF](https://arxiv.org/pdf/2610.36625v1)

**作者:** Yonglong Zhang `[一作]`, Yang Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于事件相机的航天器姿态估计完整框架，包括极性–时间交互定位、可靠性引导的几何恢复、事件–CAD一致性精细化以及质量感知轨迹优化。

**💡 创新点**

创新点在于：①三支分支交互编码实现极性、时间信息融合；②基于点级可靠性与姿态传播代理的加权PnP；③使用事件边缘与CAD投影的一致性度量实现几何约束的精细化；④将十维质量描述符与自回归尺度校准相结合，动态调节观测在轨迹优化中的权重。

**🔧 技术方法**

核心技术包括：事件分支注意力与门控网络、Spatiotemporal Aggregation (STA)、Swim-B+FPN热图回归、基于Cauchy损失的加权PnP、事件梯度一致性度量、RANSAC+EPnP初始化、热图-几何量化的质量回归、鲁棒Huber运动优化。

**📊 数据集**

使用公开的 SPADES 数据集（Proba‑2 航天器的合成与实验轨迹），包含 179,400 帧合成数据和 15,500 帧实验数据。

**📈 对比分析**

与 SPADES 公开挑战基线（PASAN+IRIS、RF‑DETR 等）对比，完整系统在测试集上取得 E_pose=0.0474，优于 PASAN+IRIS 的 0.0501，提升约 5.4%；在验证集上从 0.0772 降至 0.0514，质量加权进一步提升 6.4%。

**⚠️ 局限性**

局限性包括：对极性与时间信息的依赖导致对光照骤变敏感；轨迹优化仍保留一定的观测偏差；多窗口上下文虽提升精度但计算成本上升；未来需进一步提升对低事件密度场景的鲁棒性。

---

## 326. God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation

**arXiv ID:** 2610.36623 | [PDF](https://arxiv.org/pdf/2610.36623v1)

**作者:** Youwei Bao `[一作]` (George Mason University), Simin Chen `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对LLM推理服务中的GPU核生成，本文提出了一种插件化方法，在核优化过程中强制保证批量不变性（BI），从而提升数值一致性和可复现性。

**💡 创新点**

创新点在于将批量不变性作为可插拔的检查层，结合域特定语言（DSL）驱动的静态依赖分析与跨批量上下文的动态位一致性验证，并可无缝嵌入现有生成型核优化代理。

**🔧 技术方法**

使用了DSL规范的静态分析模块、动态验证模块、批量上下文比较及与代理生成循环中的接收门控。

**📊 数据集**

实验数据集主要来自KernelBench.com与FlashInfer-Bench的真实LLM推理工作负载，以及公开发布的131个代理生成核样本。

**📈 对比分析**

与传统只验证功能正确性与速度的评估相比，该方法将BI通过率从67.6%提升至91.9%，同时保持91.8%的基线几何平均加速率，说明在保证性能的前提下显著提升了数值可靠性。

**⚠️ 局限性**

局限性包括当前静态分析仅支持单文件分析，跨文件依赖覆盖不足；插件仍需人工设计规则，缺乏自动化学习；实验仅关注核输出级别的BI，未评估其对完整推理流程、token生成或强化学习训练的端到端影响。

---

## 327. Noise Shapes the Surplus: Strategic Play Against No-Regret Learners

**arXiv ID:** 2610.36621 | [PDF](https://arxiv.org/pdf/2610.36621v1)

**作者:** Yiheng Su `[一作]` (University of Wisconsin Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin Madison)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了噪声反馈如何影响在零和博弈中对无后悔学习者的战略剥削，提出了新的剥削机制并给出了收益上界与下界；

**💡 创新点**

首次将噪声依赖的学习者响应与衰减学习率结合，提出了基于平均响应（selection）和对手运动（movement）的剥削策略，并在非独立噪声模型下统一分析收益尺度；

**🔧 技术方法**

利用在线凸优化与随机过程理论对FTRL动态进行解析，证明学习者趋近最佳响应面并在不同噪声模型下推导收益下界；

**📊 数据集**

无；本研究为理论分析，无使用实际数据集；

**📈 对比分析**

与先前被动剥削方法对比，证明被动策略收益受限；selection可获得Ω(1/η_T)收益，movement可获得Ω(∑η_t)收益；在常用的η_t∝t^(-1/2)学习率下均实现Ω(√T)；未进行实验验证；

**⚠️ 局限性**

依赖特定正则化器与噪声假设，movement机制在噪声导致几乎纯响应时效果下降；未覆盖任意高维或非零和情形，缺乏实证验证。

---

## 328. Measuring Users' Security and Privacy Concerns About Third-Party Community Bots on Discord

**arXiv ID:** 2610.36626 | [PDF](https://arxiv.org/pdf/2610.36626v1)

**作者:** Kofi Ampomah `[一作]` (University of Alabama), Tu Le `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对23,102个Discord第三方社区机器人的18.3万条用户评价进行分析，提取并量化用户对安全与隐私的关注点，并基于此构建了BotLens审计框架；

**💡 创新点**

创新点在于①构建了31主题的安全与隐私分类法，扩展了先前的20主题体系；②采用LLM知识蒸馏+DeBERTa细化分类，既高效又能大规模识别S&P相关评论；③提出SPR与TPR两种度量，并实现了可在线查询的BotLens审计服务；

**🔧 技术方法**

技术主要包括：Playwright爬虫、语言检测与Argos Translate翻译、LLM（Llama 3 8B）标签生成、DeBERTa‑v3‑large 微调分类、GoEmotions情感分析、权限位字段解析与统计；

**📊 数据集**

使用的主要数据集是来自Top.gg的23,102个Discord机器人的183,142条用户评价及其元数据（权限、标签、星级、服务器数、GitHub链接等）；

**📈 对比分析**

与传统的手工标注或仅基于星级/权限的评估相比，Pipeline的SPRR达到4.82%，在confession/anonymous等敏感类bot中可达>50%，显示出显著的安全隐私讨论聚焦；但由于数据来源单一、标签误差和未验证的事实真实性，整体效果仍受限；

**⚠️ 局限性**

局限性包括：①仅采集Top.gg公开信息，缺乏对实际bot行为的验证；②LLM与模型误差可能导致标签漏检或误检；③用户自评易偏向隐私关注者，可能高估SPRR；④缺乏长期追踪与因果推断，无法确认权限与隐私讨论的因果关系。

---

## 329. The uneven rise of AI-generated web content

**arXiv ID:** 2610.36631 | [PDF](https://arxiv.org/pdf/2610.36631v1)

**作者:** Adam Visokay `[一作]` (University of California Berkeley), Tyler H. McCormick `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对2019–2026年 Common Crawl 采样网页进行多重 AI 检测和贝叶斯校正，估算公共可访问网页中 AI 生成文本的比例及其随时间、语言、页面中心度、顶级域名等维度的差异；

**💡 创新点**

创新点在于：①构建了多检测器联合校正的贝叶斯层级模型，可在不同文本特征下估计灵敏度和特异度；②利用前 2022 年控制期的爬取结果作为负样本，对检测器误差进行实际校准；③揭示 AI 文本在网络边缘层级增速最快、语言与域名差异显著的趋势；

**🔧 技术方法**

使用了 Open Pangram（Llama‑3.2‑3B）与 Binoculars‑Falcon 两种 AI 检测器，结合贝叶斯层级模型、Post‑stratification 与 penalized spline 时序模型；

**📊 数据集**

数据集为 71 次 Common Crawl 抓取的 5.1 亿可抓取页面（≥500 字符），按 PageRank 层级分层抽样，包含多语言、不同文本长度、域名类别等信息；

**📈 对比分析**

与单一检测器（如 Pangram 4）或未经校正的比例相比，模型校正后 AI 文本总体比例更稳健，误差在 95% 可信区间内被量化；

**⚠️ 局限性**

局限包括：仅覆盖可抓取、长度≥500 字符的网页，无法估计短文本与社交平台内容；缺乏域级随机效应导致潜在相关性未建模；检测器误差依赖于训练数据，跨域迁移可能受限。

---

## 330. Haste Makes Waste? Towards Safe Few-Step Text-to-Image Models

**arXiv ID:** 2610.36665 | [PDF](https://arxiv.org/pdf/2610.36665v1)

**作者:** Lior Ziv `[一作]` (Tel Aviv University), Mahmood Sharif `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在文本到图像（T2I）模型中使用少步推理（few‑step）加速技术时对安全性的影响，并提出了一种专门为少步模型设计的概念消除方法（DICE），以在保持生成速度的同时降低不安全内容的出现率。

**💡 创新点**

创新点包括：①系统评估少步模型相较于教师模型的安全性退化，揭示步数是一个隐藏的安全杠杆；②通过机制诊断发现指导（guidance）嵌入和高噪声阶段的敏感性是导致现有消除方法失效的主要原因；③提出 DICE，结合了沿直线路径的潜在构造和自蒸馏保持损失，显著提升少步模型的安全-质量权衡。

**🔧 技术方法**

使用的技术包括：流匹配（flow‑matching）与直流修正（rectified‑flow）框架；几种少步蒸馏方法（ADD、LADD、SCFM）；概念消除算法（ESD、EraseAnything、UCE、SPEED）以及新方法 DICE（LoRA‑based，加入注意力抑制和生成保持损失）。

**📊 数据集**

实验使用的数据集有：NSFW‑56k（用于评估裸露内容）、I2P（验证集）、LAION‑Pop（用于估算速度梯度和 Lipschitz 稳定性）以及 COCO‑30k（计算 FID）。

**📈 对比分析**

比较方法：对同一少步模型分别应用原始、ESD、EraseAnything、UCE、SPEED 与 DICE，并在相同的推理步骤（1 步/4 步）和相同评测指标下进行对比。DICE 在 NudeNet、SD‑Checker（含 @5 重采样）以及 HPSv2 上均取得最优或接近最优的安全率，同时保持生成质量仅略有下降（HPSv2 1‑步 0.2425→0.2425，4‑步 0.2830→0.2830），比其他方法安全率低约 4‑5 倍。

**⚠️ 局限性**

限制包括：DICE 仍未完全消除所有不安全内容，尤其在对抗性提示下仍有一定成功率；评估仅基于自动化检测器（NudeNet、SD‑Checker），缺乏人工判定；机制诊断虽然揭示了关联因素，但未给出完整的因果解释，未来仍需进一步研究。

---

## 331. Differentially Private Preconditioned Power Iteration Method for Principal Component Analysis that Adapts to the Coherence

**arXiv ID:** 2610.36691 | [PDF](https://arxiv.org/pdf/2610.36691v1)

**作者:** Duy V. Le `[一作]` (Boston University), Huy L. Nguyen `[通讯]` (Northeastern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

该论文提出了一种在行级 ℓ₂ 隐私模型下，利用预条件化的幂迭代算法实现对数据矩阵的前 k 个右奇异向量的差分隐私估计。

**💡 创新点**

创新点在于算法自适应于矩阵的相干度（coherence），在不需要预先知道相干度和特征间隙的前提下，通过自适应裁剪和私有超参数选择，实现了与相干度相关的误差上界。

**🔧 技术方法**

主要技术包括预条件化的块幂迭代、对行贡献裁剪、Gaussian 机制的噪声注入以及马尔可夫过程的集中分析。

**📊 数据集**

论文未使用具体数据集，全部为理论分析与算法证明。

**📈 对比分析**

与之前的最坏情况 d 依赖的私有 PCA 算法相比，本方法在低相干度场景下误差下降更快，且在特征间隙不明显时仍保持近最优误差。

**⚠️ 局限性**

限制主要包括对特征间隙 Δ_k 的正性与规模要求，以及对相干度不小的矩阵仍可能产生较大误差；此外算法在高维稀疏数据中裁剪阈值的自适应选择仍有待改进。

---

## 332. The Exact Weight Spectrum of $\RM(m-7,m$): A Counterexample to the Carlet-Solé Conjecture

**arXiv ID:** 2610.36687 | [PDF](https://arxiv.org/pdf/2610.36687v1)

**作者:** Yueying Lou `[一作]` (Nanjing Normal University), Qichun Wang `[通讯]` (Nanjing Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

确定了所有长度为 2^m、阶数为 m-7 的 Reed–Muller 码的完整权重谱，并给出该谱的精确表达式。

**💡 创新点**

首次提供了 Carlet–Solé 猜想的反例，展示了原猜想在 m≥14 的 (m-7,m) 码中不成立，同时提出了新的结构性猜想。

**🔧 技术方法**

采用仿射几何、傅里叶分析与组合归约等数学工具，结合几何迁移与四块拼接构造，逐步排除潜在权重并构造完整谱。

**📊 数据集**

本研究为理论推导，未使用传统数据集；所有结论均来自符号计算和组合证明。

**📈 对比分析**

通过严格的排除与构造方法，证明了所有可能权重均被涵盖或排除，未给出数值性能指标，主要关注的是理论完整性与严谨性。

**⚠️ 局限性**

局限在于仅完成了 (m-7,m) 情况，尚未解决更高阶 (n-8,n) 等一般情况；另外，对 3·2^{n-r} 以上权重的结构理论仍待进一步研究。

---

## 333. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs

**arXiv ID:** 2610.36671 | [PDF](https://arxiv.org/pdf/2610.36671v1)

**作者:** Donghyeon Cho `[一作]`, Eunho Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无需训练或阈值的自适应图像访问方法 LOWN，动态决定在扩散视觉语言模型解码过程中何时处理视觉信息以及解码多少标记。

**💡 创新点**

创新点在于利用 FULL 与 TEXT‑ONLY 前向结果的一致性作为证书，既判断是否需要图像，又决定解码步长，从而实现图像访问的自适应调度。

**🔧 技术方法**

技术包括对扩散 VLM 的 Token Pruning、Key‑Value 缓存等加速技术的无缝组合，以及基于 anchor 预测与文本仅前向的一致性检查。

**📊 数据集**

使用了十个多模态基准（MathVista、ScienceQA、ChartQA、MME、MMBench、DocVQA、GQA、LLaVA‑Bench、CapArena、DeCapBench）以及 VidLaDA 的 EgoSchema 与 MLVU。

**📈 对比分析**

与 RedVTP、ZOO‑Prune、KLASS、D^3ToM 等基线以及 Fast‑dLLM 缓存方法对比，LOWN 在保持 99.6% 基线分数的同时将解码 FLOPs 降低至约 0.35×，与视觉 Token Pruning 或缓存组合时进一步提升加速比。

**⚠️ 局限性**

局限性包括对 anchor 年龄敏感，较长解码步骤会增加误差；在视觉依赖强的任务（如图表阅读）上加速效果有限；且需在推理阶段动态决定成本，难以提前设置。

---

## 334. Proof Repair and (Co)algebra Equivalences

**arXiv ID:** 2610.36669 | [PDF](https://arxiv.org/pdf/2610.36669v1)

**作者:** E. G Plante `[一作]`, Talia Ringer `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在类型理论内部化配置（configuration）并与 P‑代数（polynomial endofunctor algebra）联系，首次给出了类型理论证明修复的形式化基础，并将其推广到共代数（coalgebra）与共构造（coinductive）类型的修复。

**💡 创新点**

创新点主要包括：
- 证明配置恰好是两个初始代数之间的代数同构；
- 展示证明修复不依赖 Univalence，能够在非等价类型理论中使用；
- 引入共代数配置并证明其与最终共代数同构对应，构成了首个针对 coinductive 类型的修复理论；
- 给出 M‑type（共代数的规则化表述）与最终共代数之间的同构，填补了现有研究的空白。

**🔧 技术方法**

主要技术手段包括：
- 基于依赖型理论（Martin‑Löf 及 Cubical Agda）的形式化；
- 对 P‑代数、初始代数与最终共代数的抽象定义与证明；
- 代数同构与共代数同构的构造与性质证明；
- 对 W‑type 与 M‑type 的规则化与同构关系的形式化。

全部结果均在 Agda/Cubical Agda 代码库中完成。

**📊 数据集**

并未使用传统意义上的数据集；本文以 Agda 证明例子（Nat、Bin、Stream、Stream' 等）作为演示与验证的案例。

**📈 对比分析**

评估主要以形式化证明为主，并与现有工具（如 PUMPKIN Pi）对齐验证一致性。文中未给出运行时间或空间等性能指标；重点是理论一致性与可实现性。

**⚠️ 局限性**

局限性：
- 对 M‑type 在 Cubical Agda 中的存在性证明仍未完成，某些类型（如高阶递归或非严格正类型）仍需进一步推广；
- 实际修复工具的实现与实验验证尚未展开；
- 本理论在非等价类型理论中已适用，但在包含更丰富的等价公理（如 UCT）时的扩展仍需研究。

---

## 335. Edhivo: An AI-Native Platform for Scalable Assessment and Instant Feedback in Higher Education

**arXiv ID:** 2610.36654 | [PDF](https://arxiv.org/pdf/2610.36654v1)

**作者:** Moraish Kapoor `[一作]` (Stony Brook University), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并在斯坦福布鲁克大学的AMS 691.01课程中试点了一款名为Edhivo的AI原生教学平台，用于自动生成和评估多项选择题及开放式问题，提供即时反馈与课程分析。

**💡 创新点**

创新点在于双LLM管道：一模型基于上传教材自动生成情境化评测题目，另一模型基于教师设定的评分标准进行多准则、基于rubric的开放式答案评估，且所有交互均保持源代码约束并引入人机回顾保证可信度。

**🔧 技术方法**

技术包括：Gemini 3 Flash（1M+ token上下文）用于题目生成；GLM 4.7 Flash用于评估；AI Orchestration Layer负责工作流调度；结构化JSON schema保证输出可直接审阅；Chain-of-Thought提示用于评估模型的逐步推理；Human-in-the-Loop审核与教师权威交互。

**📊 数据集**

数据集主要为教师上传的课程材料（教材、讲义、会议记录等）以及学生提交的答案；在试点中收集了36名研究生的30份评测问卷与约2400条评分记录。

**📈 对比分析**

与传统手工作业和现有LMS相比，题目生成时间从30分钟缩短至约5.2分钟，评估时间从人力每学生约2.5分钟降至0.4秒，整体教师工作量减少约25分钟/批次；成本仅为每人$0.00275；学生满意度和学习体验评估均达到了100%正面反馈。

**⚠️ 局限性**

局限性包括缺乏考试防作弊功能；评估模型可能对语言风格产生偏差；在非STEM人文学科的主观性较强内容中尚未验证其鲁棒性；试点规模有限，缺乏跨学科多机构的纵向可靠性研究。

---

## 336. Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction

**arXiv ID:** 2610.36643 | [PDF](https://arxiv.org/pdf/2610.36643v1)

**作者:** Hansol Park `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Tap2Listen，一种支持即时目标切换的流式音视频目标说话人提取模型。

**💡 创新点**

创新之处在于将目标无关的音频编码器与目标特定的解码器分离，仅在切换时重新投影缓存，极大降低切换延迟。

**🔧 技术方法**

采用因果STFT、注意力编码器、BlazeNet64唇部特征、跨注意力解码器、深度监督和预测损失等技术。

**📊 数据集**

在LRS3-2mix数据集上进行训练与评估。

**📈 对比分析**

与AV‑SkiM基线相比，Tap2Listen在SI‑SNRi上提升至9.68 dB，切换后立即恢复完整质量，且单核CPU可实现实时推理。

**⚠️ 局限性**

局限性包括算法时延为16 ms，仍高于1 ms基线；模型仅在单个移动CPU核心上测试，未覆盖更大规模或多说话人场景。

---

## 337. Deep Bayesian Approaches for Interpretable Nonlinear Structural System Identification

**arXiv ID:** 2610.36633 | [PDF](https://arxiv.org/pdf/2610.36633v1)

**作者:** Wei Liu `[一作]` (National University of Singapore), Eleni Chatzi `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对深度贝叶斯滤波在非线性结构系统识别中的方法进行了系统性综述，并在三维剪切框架上评估了 DMM、PgDMM 和 Neural EKF 三种框架的动态状态与参数联合识别。

**💡 创新点**

将可学习的状态空间模型与物理先验结构相结合，提出了 PgDMM 和 Neural EKF 两种深度贝叶斯滤波实现，并系统阐述了学习目标、推理机制与物理约束的互补作用。

**🔧 技术方法**

深度可变状态空间模型、可学习的高斯过程、变分推理、可微分卡尔曼滤波、Bouc‑Wen 非线性弹性、ROM、ELBO 最大化。

**📊 数据集**

基于开放源代码的三维剪切框架模拟数据，包含 Bouc‑Wen 悬挂耦合、温度/材料参数变化及不同震动源加速计测量的加速度序列。

**📈 对比分析**

对后验重建、无测量预测、结构状态恢复、参数估计四项指标进行评估，结果表明 PgDMM 在重建和状态恢复上优于 Neural EKF，Neural EKF 在参数估计和短期预测上更稳健，但两者在长时间预测误差上相近。

**⚠️ 局限性**

仅基于模拟数据，缺乏现场验证；预测误差随时间增长显著；参数可识别性受观测限制；模型对低频漂移易产生不稳定；缺乏充分的不确定性校准和大规模结构扩展。

---

## 338. Normalized Cuts and Data Clustering via Matrix Fractional Programming

**arXiv ID:** 2610.36616 | [PDF](https://arxiv.org/pdf/2610.36616v1)

**作者:** Yannan Chen `[一作]` (Shenzhen University), Kaiming Shen `[通讯]` (Chinese University of Hong Kong (Shenzhen))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多维二次变换的分数规划方法，专门用于解决多类别归一化切割（NCut）图聚类问题，并给出高效的FPC算法实现。

**💡 创新点**

创新点：① 将分数规划推广到多维矩阵比率，构造多维二次变换；② 将NCut离散优化转化为加权二分匹配，可通过线性搜索求解；③ 与MM理论建立联系，证明迭代收敛并给出收敛速率分析。

**🔧 技术方法**

使用技术包括：多维二次变换、分数规划、MM（最小化/最大化）理论、加权二分匹配、线性搜索、稀疏锚点图构建等。

**📊 数据集**

实验数据集：12个公开基准数据集（如 Shuttle、Balloons、Breast、Thyroid、Office、Splice、Rice、Landsat、Epileptic、STL10、Sensorless、CIFAR10、MNIST、RT‑IoT2022）以及用于图像分割的超像素+特征数据。

**📈 对比分析**

比较方法：与传统两阶段谱聚类（SC、ISR、DSSRC）以及直接NCut优化方法（FINC、FCD、FPC）对比。FPC 在绝大多数数据集上获得最低 NCut 目标值，提升约 3%–18%，在 ACC/NMI/ARI 上同样取得最优或接近最优；收敛次数少，运行时间显著更短。

**⚠️ 局限性**

局限性：仅针对多类别 NCut 目标，未扩展到其他图划分准则；对大规模数据仍需锚点图近似；不同初始化可能导致收敛到局部最优；NCut 目标的饱和性表明其对聚类质量的区分能力有限。

---

## 339. Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training

**arXiv ID:** 2610.36794 | [PDF](https://arxiv.org/pdf/2610.36794v1)

**作者:** Yue Wu `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Select-TTT，通过查询条件下的快权重更新提升LLM长上下文问答性能。

**💡 创新点**

在In-Place TTT基础上引入基于查询的可学习写入强度，利用冻结输出头的logit空间query-CE做无监督的相关性评分，无需额外参数。

**🔧 技术方法**

使用快权重Test-Time训练、In-Place TTT、softmax温度缩放、块级logit计算、top-k mean池化等技术。

**📊 数据集**

在LongMemEval-S、LoCoMo长上下文QA基准以及LongCite-45k持续训练数据集上进行实验。

**📈 对比分析**

对比基线、IP-TTT和Select-TTT，在零样本和持续训练两种协议下实验，Select-TTT在两个模型上均显著提升平均分，零样本提升约3–4个百分点，训练后最高达37.4%等。

**⚠️ 局限性**

依赖冻结LM头的query-CE估计可能不够精准，缺乏学习的相关性函数；固定块大小可能不适合语义边界；在更大模型、多语言和多模态场景中仍需验证。

---

## 340. RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions

**arXiv ID:** 2610.36791 | [PDF](https://arxiv.org/pdf/2610.36791v1)

**作者:** Haoyu Zhao `[一作]` (DAMO Academy Alibaba Group), Zhongyu Li `[通讯]` (Hong Kong Embodied Ai Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个跨体态的视频世界模型，通过共享连续的潜在动作作为控制接口，利用人类、机器人和仿真视频进行预训练，并在目标机器人上适配，实现了可控的视觉动态生成与机器人数据合成。

**💡 创新点**

创新点包括：①使用3D‑grounded 潜在动作作为跨体态共享控制信号；②在预训练阶段无须对齐不同硬件的原始控制空间；③结合全局动作标记与帧对齐的局部注入，提升动作驱动的视频预测质量；④提供两种数据生成管线（数字遥控与潜在动作迁移），显著提升下游策略学习效果。

**🔧 技术方法**

技术包括：Cosmos3混合Transformer视频生成器、RynnVLA-Latent的连续潜在动作提取、Rectified Flow去噪、全局与局部动作注入机制、无文本条件的控制扩散模型、动作重标注与逆运动学回归、图像编辑/机器人插值合成。

**📊 数据集**

使用了23个公开数据集，约6189.99小时、2567356条片段，涵盖人类交互视频（如HowTo100M、EgoVid、EgoDex、EPIC‑KITCHENS）、真实机器人演示（DROID、RoboMIND等共3850条演示）、仿真视频（RoboCasa、BEHAVIOR‑1K等），并构建了约800小时的合成机器人视频数据集(-80M)。

**📈 对比分析**

与DreamDojo、LIBERO、LIBERO‑Plus等基准比较，模型在视频预测方面的PSNR、SSIM提升至20.9+（Marvin‑WUJI）和15.96+（Astribot‑S1），LPIPS降低至0.12/0.34；在真实机器人策略任务上，数字遥控与合成数据分别使任务成功率提升约10–20%；在模拟策略基准上，mid‑training + 合成数据将LIBERO成功率从94.4%提升至96.1%，在Robotic实验中多任务成功率均有显著提升。

**⚠️ 局限性**

局限包括：潜在动作可能包含相机运动或外观变化，导致缺乏因果可辨识性；合成的动作与观测是分离生成，物理一致性无法完全保证；实验样本量有限，统计显著性未充分验证；synthetic mid‑training同时增加数据与训练算力，未能单独评估两者贡献。

---

## 341. Rethinking Channel Estimation for Fluid Antenna Systems: A Physics-Informed Operator Learning

**arXiv ID:** 2610.36787 | [PDF](https://arxiv.org/pdf/2610.36787v1)

**作者:** Zixin Wang `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于物理信息的U形傅里叶神经算子（UFNO）框架，用于流动天线系统（FAS）的连续场信道估计；将信道估计建模为投影Helmholtz方程约束下的随机逆场重建问题；通过EIT分析揭示有限孔径导致的频域卷积，转化为逆卷积问题，并在UFNO中加入可学习软阈值与Sobolev损失以抑制噪声与保持高频细节；实现分辨率不变推理，直接在任意空间-频率分辨率上进行估计。

**💡 创新点**

1）将FAS信道视为物理连续场逆问题并以投影Helmholtz方程为核心；2）利用Landau–Pollak理论给出有效自由度上界，指导频谱模式预算；3）设计U形FNO结构，结合全局频域分支与局部MLP，并引入可学习软阈值（SST）和Sobolev损失，解决低通偏差与高频细节失真；4）实现分辨率不变推理，无需重新训练或插值。

**🔧 技术方法**

傅里叶神经算子（FNO）+U形网络、可学习软阈值（SST）、Sobolev损失、EIT谱泄漏分析、投影Helmholtz方程、Landau–Pollak有效自由度理论、物理约束训练、对比实验（LS、OMP、SBL、DC‑GOMP、CNN、ViT）。

**📊 数据集**

基于几何多径模型的合成数据集，包含约5.12万条训练样本和5.12千条验证/测试样本；频率3.5 GHz，4λ线性孔径，48 × 48空间‑频率网格，48个子载波；SNR在[-20,30] dB随机取值。

**📈 对比分析**

与传统模型方法（LS、OMP、SBL、DC‑GOMP）、CNN、ViT以及UFNO的去除SST或Sobolev损失版本进行对比；在30 dB SNR下UFNO的NMSE低于‑35 dB，比分量为5 dB；在不同延迟扩散、采样率和跨分辨率实验中均保持最低误差；在线推理延迟约9.5 ms，远低于OMP（≈21 ms）和SBL（≈108 ms）等传统迭代方法。

**⚠️ 局限性**

仅在合成数据上验证，未使用实测信道；UFNO实现延迟略高于标准FNO；对高延迟扩散仍受低通截断影响；目前仅适用于一维线性FAS，未扩展到平面或体积阵列；不处理时变Doppler效应，未来需开发时空算子。

---

## 342. Expressive Guitar Synthesis via Controllable Latent Flow Matching

**arXiv ID:** 2610.36786 | [PDF](https://arxiv.org/pdf/2610.36786v1)

**作者:** Hsueh-Wei Fu `[一作]`, Yi-Hsuan Yang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出一种两阶段吉他合成管线，先预测每弦可编辑的音高(f0)与振幅轨迹，再使用潜在流匹配方法将其渲染为音频；

**💡 创新点**

创新点在于引入跨弦门控注意力机制和伪速度预测器，使得模型能够在仅使用MIDI输入时生成可编辑的演奏控制并显著提升控制精度与音质；

**🔧 技术方法**

技术上采用轴向Transformer、门控跨弦注意力、潜在流匹配（Latent Flow Matching）以及AdaLN调制的DiT，并通过MSE、CFG等训练策略实现；

**📊 数据集**

实验数据集选用公开的 GuitarSet 吉他录音数据集，进行训练、验证与测试；

**📈 对比分析**

通过与单阶段模型及 DDSP‑Guitar 进行对比，实验显示本系统在控制精度（f0 MAE、振幅相关性）和 MOS 主观评分上均优于基线，尽管在 MSSL 上略逊一筹；

**⚠️ 局限性**

局限性包括依赖六弦高质量记录，合成器仍对混音特性敏感，跨弦注意力在单弦场景无明显收益，且未针对录音环境与音色进行条件化。

---

## 343. CVBL-CCTV-Face Dataset for Real-World Face Enhancement and Recognition in Surveillance

**arXiv ID:** 2610.36781 | [PDF](https://arxiv.org/pdf/2610.36781v1)

**作者:** Manish Das `[一作]` (Indian Institute of Information Technology), Satish Kumar Singh `[通讯]` (Indian Institute of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CVBL-CCTV-Face数据集，并搭建了面部检测-超分辨率-识别的模块化流水线。

**💡 创新点**

创新点在于：①提供真实无配对的低质监控与高质参考图像；②在三种训练设置下系统性评估增强模型对识别的影响；③展示实时增强识别系统的可行性。

**🔧 技术方法**

使用MTCNN进行检测，CodeFormer、RestoreFormer++、PiSA-SR、TSD-SR等Transformer/扩散超分模型进行增强，Inception-ResNet-v1提取512维嵌入进行L2匹配。

**📊 数据集**

使用CVBL-CCTV-Face数据集（Version A & B 共223身份，含8090张LQ与2612张HQ图）进行实验。

**📈 对比分析**

通过与预训练模型、合成降质微调、真实LQ微调三种设置对比，发现合成降质微调能使Top‑1识别率提升至70.08%，显著优于基线62.6%。

**⚠️ 局限性**

局限在于：对真实LQ图像微调效果不佳，模型在多变、复杂的真实降质条件下仍难以保持稳健性。

---

## 344. DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering

**arXiv ID:** 2610.36759 | [PDF](https://arxiv.org/pdf/2610.36759v1)

**作者:** Jing Huang `[一作]` (Ant Digital Technologies, Ant Group), Ying Yan `[通讯]` (Ant Digital Technologies, Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出DocSkill框架，通过对多语言多跳文档问答任务的四个可复用技能（Locate、Inspect、Integrate、Verify）进行自适应编排，实现跨页证据检索、细粒度解析、整合与答案验证，最终完成答案生成。

**💡 创新点**

创新点在于：①将技能抽象为高层能力而非单一工具；②利用有界MCTS离线生成多样化技能轨迹，提供SFT和DPO的监督；③构建了大规模多语言多跳问答基准MMHop-DocBench，提供精确的页级和元素级证据标注。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（Qwen3-VL-4B/8B、Gemini-3.1-Pro等）、离线有界MCTS轨迹搜索、监督微调（SFT）+直接偏好优化（DPO）、工具调用（Page_Retriever、Page_Parser、Zoom_In）以及跨语言知识迁移。

**📊 数据集**

使用的数据集为新构建的MMHop-DocBench（60K问答对，10种语言，8个文档领域）以及公开的MMLongBench-Doc和DocBench，用于评估跨文档、多跳、多语言的问答性能。

**📈 对比分析**

与VLM直接推理、OCR+RAG、传统工具增强和其它文档智能体等基线对比，DocSkill在MMHop-DocBench上实现41.0%准确率、49.2%F1，超越最强基线5.5/6.3分；在MMLongBench-Doc和DocBench上同样获得显著提升，表明自适应技能编排和轨迹监督的有效性。

**⚠️ 局限性**

局限性包括：①技能集合固定，仅覆盖四类常见操作，难以自动发现更细粒度或更复杂的技能；②离线MCTS生成轨迹需要额外计算成本，且对高质量评估模型有依赖；③在极长或结构极其复杂的文档中，仍可能出现证据检索或整合不足；④模型性能仍未达到人类专家水平，尤其在跨语言一致性与逻辑推理方面存在提升空间。

---

## 345. Efficient Remote Sensing Image-Text Retrieval via Interaction-Free Recall and Text-Guided Reranking

**arXiv ID:** 2610.36756 | [PDF](https://arxiv.org/pdf/2610.36756v1)

**作者:** Xi Chen `[一作]` (Wuhan University), Zhenyuan Shun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两阶段 Fast‑then‑Fine (FTF) 遥感图像–文本检索框架，先通过交互无关的多粒度召回选取 top‑K 候选，再用文本引导的 BTIB 进行候选级细粒度重排序。

**💡 创新点**

创新点包括：① 交互无关多粒度召回实现预计算全图库特征；② 参数无关的 Balanced Text‑guided Interaction Block (BTIB)，利用双向 softmax 归一化实现高效文本‑图像对齐；③ 联合优化的 Inter‑and‑Intra‑Modal (I^2M) 损失，既对齐跨模态，又保持单模态邻域一致性。

**🔧 技术方法**

采用 CLIP 预训练的视觉‑文本双编码器，窗口池化构造粗细尺度视觉特征；BTIB 通过双向 softmax 交互实现文本引导重排；I^2M 损失实现交叉模态对齐与结构一致性；使用 ANN 搜索实现高效召回。

**📊 数据集**

在 RSITMD、RSICD、UCM 三个公开遥感图像‑文本检索基准上进行实验。

**📈 对比分析**

与 20+ 现有任务特定、VLM、MLLM 辅助方法比较，FTF Full 在 mR 约 51.7 与最强基线相当；FTF (K=50/100/200) 在保持 mR 48.6–51.5 的同时将延迟降至 12.4–39.6 ms、吞吐提升至 80.6–25.3 q/s，展示了优秀的准确‑效率折中。候选覆盖率高，且在大规模数据集上显著降低延迟增长。

**⚠️ 局限性**

局限性：仍需预先计算并存储全图库特征，对实时动态更新的适应性有限；BTIB 虽无参数但在候选集上增加计算量；在最优精度方面略低于全图重排序，需通过调节 K 值实现准确‑效率权衡。

---

## 346. Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning

**arXiv ID:** 2610.36751 | [PDF](https://arxiv.org/pdf/2610.36751v1)

**作者:** Pu Ren `[一作]` (Lawrence Berkeley National Lab), N. Benjamin Erichson `[通讯]` (Lawrence Berkeley National Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将线性插值与条件扩散模型相结合的时序插值方法（iFLEX），通过学习线性估计的残差来重建两个观测点之间的物理场。

**💡 创新点**

创新点在于：①先做简单的数值线性插值作为基准，②让扩散模型只学习残差而不是完整场的生成，从而显著提升重建精度；③在扩散模型中同时利用两端观测、时间比例、时间间隔及物理参数的多尺度条件，增强对不同物理演化的适应性。

**🔧 技术方法**

使用条件扩散模型（FLEX架构）+共享空间编码器+Transformer+FiLM进行标量条件调制+门控空间特征融合，训练目标为预测扩散速度，采样采用DDIM三步。

**📊 数据集**

评估数据集包括：Navier–Stokes Kraichnan turbulence（NSKT），上海雷达降水（Shanghai radar precipitation），海表温度（sea surface temperature, SST）。

**📈 对比分析**

与基线（线性插值、U‑Net、SuperSloMo、EDEN）对比，iFLEX在所有三个数据集上均实现更低的相对 Frobenius 范数误差（RFNE）和更高的 SSIM/PC；在Reynolds 36000、雷达和SST上分别比最优基线降低约30‑40％的误差。

**⚠️ 局限性**

局限性：①仅能在两个观测点足够接近、演化不至于完全不可见的场景下有效；②对间隔过长或短命事件（如瞬时降水）恢复有限；③每个时间点独立生成，缺乏时序连续性或物理约束；④线性插值作为基准在某些长间隔或极端参数下可能不够理想；⑤在不同区域、流域或不同物理系统的泛化仍需进一步验证。

---

## 347. When do token systems stabilize? Compatibility constraints and community structure

**arXiv ID:** 2610.36744 | [PDF](https://arxiv.org/pdf/2610.36744v1)

**作者:** Flore Sentenac `[一作]` (HEC Paris Business School), Sophie H. Yu `[通讯]` (University of Pennsylvania)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了兼容性约束下的代币系统，给出了代币余额稳定性的必要与充分条件，并证明最小代币规则在任何可行的选择策略中均能实现稳定；随后将该判据应用于随机块模型（stochastic block model），发现并阐明了社区障碍和个体障碍两类导致不稳定的根本原因。

**💡 创新点**

创新点在于：①提出了一个统一的、可判定的稳定性切割条件；②证明了在此条件下，最小代币规则是稳定性的最优选择；③将稳定性分析推广到具有社群结构的随机图模型，首次给出两类阻碍稳定的结构性阈值并实现了稳定性区间的紧确刻画。

**🔧 技术方法**

主要技术包括：利用二次Lyapunov函数和负漂移论证系统正递归；使用 martingale 和可选采样理论证明不稳定性；在随机块模型中应用大偏差、随机矩阵范数估计以及切割函数的凸/凹性质；以及借助 power‑of‑d 选择与图论的结合来简化分析。

**📊 数据集**

没有使用实际数据集；实验场景完全基于从随机块模型或 Erdős–Rényi 模型中生成的随机图来验证理论阈值。

**📈 对比分析**

通过严格的理论证明对比：证明最小代币规则能实现所有满足切割条件的系统的稳定性；并通过极限分析和概率界定确定稳定与不稳定的临界阈值；由于研究聚焦于稳定性理论，未给出数值实验或性能指标。

**⚠️ 局限性**

局限性在于：①假设请求分布和兼容性矩阵为已知且静态；②分析仅适用于无上限且允许负余额的代币系统；③未考虑策略性行为或动态加入/离开；④在实际应用中需要对兼容性网络有完整了解，导致可推广性受限。

---

## 348. On Shaky Ground: Characterising Non-Rigid Base Effects in Mobile Manipulation

**arXiv ID:** 2610.36725 | [PDF](https://arxiv.org/pdf/2610.36725v1)

**作者:** Zongyuan Zhang `[一作]` (Queensland University of Technology), Jonathan M. Roberts `[通讯]` (Queensland University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在两台轻量化轮式机器人平台上进行砂磨实验，测量接触力、工具中心点误差与基底振动，系统性探究非刚性基底效应对接触任务的影响。

**💡 创新点**

将力、位移与振动三项指标同步测量，并将实验结果与MuJoCo仿真对比，首次揭示基底运动导致的力短缺与工具位移共存现象及其对姿态、摩擦和支撑刚度的敏感性。

**🔧 技术方法**

采用任务空间阻抗/质量控制实现力跟踪；使用六轴力/扭矩传感器、惯性测量单元和相机标记跟踪工具位移；利用MuJoCo进行基底滑移与失稳的仿真；在实验室搭建的 cart 与 trolley 平台上收集数据。

**📊 数据集**

基于两台平台（cart/ trolley）在不同基底摩擦、姿态、力指令与工具速度条件下采集的力、加速度与位移时间序列数据；未使用公开数据集，仅为实验采集的原始传感器数据。

**📈 对比分析**

在低/高基底阻抗、标准/低姿态以及不同力指令下，计算实现力比例、力误差、力波动、工具中心点误差和基底振动峰值；结果显示低阻抗下实现力仅54–70%，工具误差约100 mm；提高阻抗后实现力提升至88–101%，工具误差显著下降；振动虽增高，但对力误差影响不大。

**⚠️ 局限性**

仅单次试验，缺乏重复性；平台为非驱动基底，无法完全代表真实轻量化机器人；未评估驱动与臂指令的协同补偿；未测量表面质量或材料去除；只涵盖硬地面与泡沫基底两种情况，未覆盖所有环境与平台组合。

---

## 349. From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents

**arXiv ID:** 2610.36713 | [PDF](https://arxiv.org/pdf/2610.36713v1)

**作者:** Yucong Zhang `[一作]` (Chinese University of Hong Kong), Ming Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了在线说话者-记忆构建框架 EvoSpeakerMem，支持在多说话者环境下对会话文本进行说话者归属与记忆写入。

**💡 创新点**

创新点在于将声纹视为身份观测，使用无训练的动态中心/多样本KNN更新策略并通过身份门控的检索实现个性化问答，同时构造了 LoCoMo-ID 评价基准。

**🔧 技术方法**

采用 ECAPA‑TDNN 说话人编码、语音转写、文本记忆后端、Parakeet 话者分离，并在语音‑文本混合检索上实现身份门控。

**📊 数据集**

使用 LoCoMo‑ID（基于 LoCoMo 的合成语音数据）、SMIIP‑TV（多时段声纹录音）以及 MSU‑Bench 进行验证。

**📈 对比分析**

与固定模板、更新、门控更新及全样本 KNN 等策略进行比较；在清洁条件下身份门控 QA 从 71.4% 提升至 75.3%，在环境偏移下从 60.1% 提升至 72.4%，但全样本 KNN 产生错误授权。

**⚠️ 局限性**

局限在于使用合成语音、缺乏真实多方重叠对话、未评估长期存储和身份纠错、以及门控阈值对比缺失等。

---

## 350. RankFormer: Rethinking Time Series Transformers through Representation Capacity Analysis and Progressive Decoding

**arXiv ID:** 2610.36703 | [PDF](https://arxiv.org/pdf/2610.36703v1)

**作者:** Peiwang Tang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 RankFormer，基于秩分析的编码器-解码器架构，用于多变量时间序列长时预测。

**💡 创新点**

创新点在于：①通过秩理论解释补丁化、非线性嵌入与注意力拓扑对表示容量的影响；②设计 Hourglass Attention 保持注意力层的秩稳定；③构建 Interactive Progressive Decoder，实现跨通道条件和递归残差细化。

**🔧 技术方法**

主要技术包括：补丁化（Patch Embedding）、两层 MLP 嵌入、Hourglass Attention、交互式递归解码器（IPD）、实例归一化、残差学习与离线折叠推理。

**📊 数据集**

在七个公开长时序预测基准上评估：ETTh1/2、ETTm1/2、Weather、Solar、Electricity、Traffic、Exchange-Rate。

**📈 对比分析**

与 9 种 SOTA Transformer/线性模型对比，RankFormer 在 7/8 数据集上 MSE/MAE 同时夺冠，整体提升约 5–15% 以上。

**⚠️ 局限性**

局限在于：对高阶注意力层仍需更大规模验证；模型对序列长度/维度的伸缩性需进一步探究；训练时需要随机采样的中间目标，实施复杂。

---

## 351. Ancestor Lifting of Quasi-Cyclic Additive Codes and Certified Quantum Codes

**arXiv ID:** 2610.36700 | [PDF](https://arxiv.org/pdf/2610.36700v1)

**作者:** Liangdong Lu `[一作]` (Air Force Engineering University), Qiang Fu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了1-生成(1,2)-准循环（QC）加性码，并通过祖先提升、对称性判据和严格的距离认证流程，构造并验证了二进制量子稳定器码中的二十一条新纪录代码；

**💡 创新点**

创新点包括：①提出祖先提升定理与除数格定律，将长度为ℓm的码的距离与其子码相关联；②建立了完全可验证的认证流水线（ISD筛选、Brouwer–Zimmermann精确距离计算与独立验证），实现了对量子距离的精确认证；③提出了可判定自对称性的回文QC族和指向构造方法；④通过距离上界、传播和子码技术，进一步改进了三十四条Grassl表条目。

**🔧 技术方法**

主要技术手段包括：加性码与量子稳定器的对应；准循环码的卷积与分解；祖先提升与奇偶ℓ层分析；对称性与缺陷多项式分析；信息集解码（ISD）与Brouwer–Zimmermann（BZ）最小距离计算；Daroux标准化与矩阵扩展；脚本化重验证与SHA-256日志归档。

**📊 数据集**

本文使用的主要数据集为Grassl量子码表中所有已知的[[n,k,d]]_2参数，并在此基础上在长度范围内穷举1-生成准循环码（约6,620个候选者），生成了21条已认证记录、34条传播改进以及若干证据级候选。

**📈 对比分析**

与Grassl表中的已记录码相比，新构造的码在多数条目上提高了至少1点距离（[[60,13]]_2提升2点），并在91≤n≤101范围内形成了完整的距离链。与已发表的最优准循环码相比，本文所给的码显著提高了逻辑量子比特数或距离。性能上，认证流程在单台Xeon Gold 5218R服务器上完成，平均CPU时间从几秒到约两天不等。

**⚠️ 局限性**

限制包括：①仅考虑1-生成准循环码，未覆盖更一般的多生成或非准循环码；②认证计算受限于计算预算，尚未对更长长度（如n>160）或更低距离的候选进行完整验证；③部分证据级候选仅提供上界，缺乏下界证明；④理论上对祖先提升的精确距离上界与偏差的完整描述仍为猜想，尚未完全证明。

---

## 352. S3-S2ST: Semantically Supervised Streaming Speech-to-Speech Translation

**arXiv ID:** 2610.36777 | [PDF](https://arxiv.org/pdf/2610.36777v1)

**作者:** Yuxiang Zhao `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出S3‑S2ST模型实现流式语音到语音翻译，学习何时提交翻译。

**💡 创新点**

使用语义对齐的增量监督学习提交策略，并可通过阈值和回放适应控制延迟。

**🔧 技术方法**

采用Thinker–Talker架构、640 ms块编码器、MLP阈值门、八层融合和预训练模型。

**📊 数据集**

训练数据基于FLEURS、内部9对数据集以及大规模ASR合成语音。

**📈 对比分析**

在FLEURS和内部数据上与SeamlessStreaming、Qwen3‑Omni等基线对比，获得竞争的BLEU/COMET且能在单一检查点实现质量–延迟平衡。

**⚠️ 局限性**

仅覆盖英中两语种，需评估多语言、长对话和潜在数据重叠问题。

---

## 353. Dependency Graphs of Mathematical Arguments: Representation, Structural Measurement, and Proof Review

**arXiv ID:** 2610.36776 | [PDF](https://arxiv.org/pdf/2610.36776v1)

**作者:** Eunji Park `[一作]` `[通讯]` (Seoul National University), Eunji Park (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种以源文件为基础的框架，用来构建和度量数学论证的依赖图。

**💡 创新点**

创新之处在于将 LLM 辅助的无编号推理步骤与严格的测量契约结合，保证跨模型的可复现度和度量一致性。

**🔧 技术方法**

使用大型语言模型进行依赖提取、图论指标（深度、体量、宏骨架）以及标准化 Prompt 协议。

**📊 数据集**

未进行实验验证，框架主要针对 NaturalProofs、Lean Blueprint、KnowTeX 等公开数学论文语料，但未给出具体数据集。

**📈 对比分析**

通过对同一论文在不同 LLM 与多次运行下的图结构、深度、体量等指标进行对比，评估模型间一致性；目前仅提出评估设计，尚无具体性能数值。

**⚠️ 局限性**

缺乏经验验证、对 LLM 识别误差的容忍度未知，且仅关注结构度量，未涉及论证的数学正确性和实际应用效果。

---

## 354. Externalizing Hand Grip Pressure Through Real-Time Visual Feedback

**arXiv ID:** 2610.36769 | [PDF](https://arxiv.org/pdf/2610.36769v1)

**作者:** Andrew L. Huard `[一作]` (University of California Santa Barbara), Misha Sra `[通讯]` (University of California Santa Barbara)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用实时视觉反馈外化握力压力的系统，并在初学者中比较连续梯度与离散条形编码以及针对抓握的传感器区域映射对握力匹配性能、工作量和注意力的影响。

**💡 创新点**

创新点在于将高维握力分布聚合为功能性手部区域，并设计连续梯度与离散条形两种可视化，揭示了空间编码对匹配精度、注意力和工作负荷的不同影响；并首次评估了抓握特定映射与通用映射对反馈可用性的作用。

**🔧 技术方法**

使用基于FSR的可穿戴手套、Arduino采集、Unity实时渲染、光学眼动追踪以及线性混合模型和广义线性混合模型进行统计分析。

**📊 数据集**

数据集来源于两位专家分别在棒球投掷和乒乓球握拍时佩戴相同手套收集的六种抓握的参考压力分布，共28名初学者完成36个试验。

**📈 对比分析**

通过比较平均MAE、匹配次数、首次匹配时间、NASA‑TLX、眼动熵等指标发现梯度编码在匹配精度和匹配次数上显著优于条形编码，而条形编码在初始熟悉度上更直观；针对抓握的映射在感知工作量上有优势，但对客观精度影响不大。

**⚠️ 局限性**

限制包括只针对右手、使用单一固定型棉手套和FSR传感器导致的信号噪声、实验任务为静态握力，缺乏动态运动的真实情境，且参考抓握与实验者手型可能不一致。

---

## 355. Evaluation-Guided Demonstration Refinement for Vision-Language-Action Policies: The Winning Entry in the MetaFine Open Competition

**arXiv ID:** 2610.36765 | [PDF](https://arxiv.org/pdf/2610.36765v1)

**作者:** Lifan Zheng `[一作]` (Southeast University), Fan Liu `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为MetaFine开放竞赛中的五个桌面操作任务分别微调任务专用策略，结合官方演示与专家生成轨迹，并在本地评估中反复修订演示以提升抓取、推送、插入等关键动作。

**💡 创新点**

创新点包括基于指令级分数的局部评估与场景聚类来定位失败，采用数据集聚合方法添加专家续演示和恢复示例，以及在每次检查点选择时同时确保各指令不被牺牲。

**🔧 技术方法**

使用OpenPI框架、流匹配行为克隆目标、AdamW优化、余弦学习率衰减以及多步去噪的动作块生成技术。

**📊 数据集**

训练数据来自MetaFine公开的演示与专家程序，随后在额外场景中生成并修正轨迹，最终保留成功的演示与恢复演示。

**📈 对比分析**

通过官方评测获得整体96.66分，榜首；在本地评测中各任务的S分数均高于参考模型，表明策略在多场景、多指令下具有稳定性。

**⚠️ 局限性**

局限性包括对本地评估的高度依赖、指令间可能的性能互相影响、未能完全消除在不同场景下的退化，并且实验结果受多轮迭代与样本混合的影响，难以单独验证每项改进的效能。

---

## 356. Readout and Operating-Point Sensitivity in Event-Stream Candidate Selection

**arXiv ID:** 2610.36743 | [PDF](https://arxiv.org/pdf/2610.36743v1)

**作者:** Yifan Shuai `[一作]` `[通讯]`, Yifan Shuai

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估了事件摄像机在离线候选选择任务中不同编码器、读取方式和轻量化模型的性能，探讨了读取方式和阈值校准对模型排名的影响。

**💡 创新点**

提出了针对读取方式与阈值校准交互的实证评估协议，并展示了在同一阈值下模型性能可能逆转的现象；同时通过多次参与者分组和阈值再校准，量化了评估的分区与阈值不确定性。

**🔧 技术方法**

使用事件累积映射生成图像输入，对 LeJEPA、DINO、V‑JEPA 等 ViT 编码器，以及 MobileNetV3‑Small、ResNet‑18 和 TinyCNN 等 CNN 进行特征提取；随后通过标准化后的岭回归二分类头（或 TinyCNN 的线性头）实现候选选择。

**📊 数据集**

DailyDVS‑200 数据集（520 个 1.6 s 事件片段，47 位参与者），涵盖握手、跌倒、门开启等多种日常动作。

**📈 对比分析**

通过捕获率、误触率、低‑FPR 区域的 pAUC20 以及 TPR10 等指标比较，DINO CLS 在 pAUC20（0.746）和 TPR10（0.809）上优于 LeJEPA，MobileNetV3‑Small 在捕获率上虽竞争，但低‑FPR 仍略逊；读取方式和阈值校准的交互可导致同一模型在不同评估维度上出现逆转。

**⚠️ 局限性**

评估仅限于离线片段，未包含连续背景、实时触发、同步 IMU/音频或能耗测量；使用的所有数据均已在实验室中采集，缺乏对真实可穿戴设备的验证，模型在跨人群、持续使用或能耗方面的泛化能力未得到验证。

---

## 357. CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models

**arXiv ID:** 2610.36734 | [PDF](https://arxiv.org/pdf/2610.36734v1)

**作者:** Jingpu Yang `[一作]` (Beijing Zhongguancun Academy), Xiaoqing Zhang `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出兼容性感知内存重放 (CAMR) 框架，用于大型语言模型的持续学习，能够在每个新样本到达时联合决定该样本的监督方式与历史重放样本的选择，并在固定的重放预算内进行动态调度。

**💡 创新点**

创新点在于：① 通过兼容性检查筛选可用历史样本；② 用可学习的记忆门对兼容历史进行加权摘要；③ 构建在线 actor‑critic 策略，根据模型掌握度、门控历史信息和回放预算自适应选择三种动作（普通重放+当前样本、优先兼容相关历史、跳过已掌握重复样本），从而同时优化知识获取与保留。

**🔧 技术方法**

采用 MPNet 固定编码器提取语义嵌入、LSH + 精确索引检索、RMSNorm + 余弦门控、Actor‑Critic 强化学习、基于 NLL 变化的奖励设计、答复一致性评估及多任务交叉回放预算控制。

**📊 数据集**

使用数据集包括 WikiBigEdit（事实更新）、WikiFactDiff（事实替换）、TRACE 任务集（C-STANCE、MeetingBank、Py150、ScienceQA）、以及跨域序列 FinQA → GSM8K → APPS → SAMSum。全部实验以 Qwen3‑8B‑Base 为基础模型进行。

**📈 对比分析**

与多种基线（随机重放、全历史重放、语义重放、LoRA、知识编辑、参数空间方法等）对比，CAMR 在 3K–100K 事实更新中的 Original Match 最高（3K 93.66%、100K 98.10%），在 TRACE 任务中实现了最高最终分数（C-STANCE 56.75%、MeetingBank 69.84%、Py150 67.92%、ScienceQA 95.40%），并在跨域任务序列中保持或提升前期任务性能，整体表现优于所有对比方法。

**⚠️ 局限性**

局限性包括：仅在 Qwen3‑8B‑Base 上验证，未在更大模型或不同架构上评估；部分基线使用 LoRA 降低成本，完整参数微调的对比尚未完成；CAMR 在 WikiBigEdit 由于重复事实稀少而未被触发；实验受限于计算资源，未覆盖更大重放预算或更丰富的重复数据场景。

---

## 358. Hy-Nomas: Towards Free-form Interactive World Generation

**arXiv ID:** 2610.36729 | [PDF](https://arxiv.org/pdf/2610.36729v1)

**作者:** Qiaoling Chen `[一作]`, Kai Zou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个8B参数的交互式视频世界模型，支持自由文本指令和连续相机控制，实现多轮交互生成长时序视频。

**💡 创新点**

核心创新包括基于分段的因果注意力与动作感知掩码、结构化视频字幕对齐训练、Prompt Enhancer将用户请求转换为结构化条件，以及多阶段训练策略和蒸馏实现高效少步生成。

**🔧 技术方法**

使用改造的DiT（HunyuanVideo 1.5基础），DCAE‑V视频自编码器，Qwen2.5‑VL‑7B文本编码器，动作感知因果注意力、相机跨注意力、流匹配损失、分段蒸馏以及LoRA微调。

**📊 数据集**

数据来源于多样化真实与虚拟视频，经过预处理、去重、分类、摄像机轨迹标注和结构化字幕；采用ViPE估计轨迹，Kimi‑K3、Gemma‑4等模型生成字幕，形成大规模分段式训练集。

**📈 对比分析**

在自定义的Hy‑Nomas‑Bench（包含多轮文本、相机导航、交替控制与空间记忆四个子任务）上与Genie 3、LingBot‑World v2、Happy Oyster进行对比，人类评测显示该模型在提示跟随、物理合理性和相机控制上均优于对手，尤其在多轮交互和交替控制上表现突出，但在空间记忆上仍落后于Genie 3。

**⚠️ 局限性**

局限性包括：对长时间连续生成仍易出现视觉漂移与动作重复；空间记忆仍不足，尤其是场景与对象持久一致性；LoRA微调可能在保留多种控制能力时出现回退；模型对稀有摄像机轨迹的泛化能力有限。

---

## 359. SAGE: Saliency-Annealed Gated Encoder for Universal Vision Recognition

**arXiv ID:** 2610.36719 | [PDF](https://arxiv.org/pdf/2610.36719v1)

**作者:** Le Yang `[一作]` (Huawei Technologies Co., Ltd.), Qiang Zhao `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SAGE框架，将自监督ViT通过CLS‑token saliency、register‑token context和高频细节等先验进行显式增强，并设计三阶段Annealing训练与两阶段VFL的全流程；

**💡 创新点**

创新点包括①利用CLS‑token saliency引导残差门控与高频增强；②采用温度softmax的多层动态聚合；③引入三阶段余热Annealing与两阶段VFL的训练策略；

**🔧 技术方法**

使用的技术包括ViT‑Adapter骨干、DINO‑style saliency、FiLM调制、Laplacian高频增强、MSDeformAttn、Varifocal Loss、焦点损失、Annealing schedule、semantic consistency loss、SyncBN等；

**📊 数据集**

实验数据集涵盖COCO（检测/分割/姿态）、ADE20K（语义/实例分割）、TinyImageNet（分类）和DIOR‑R（定向检测）；

**📈 对比分析**

与DETR、DINO、ViTDet、RF‑DETR等多种基线在单尺度推理、相同显存/预训练条件下对比，SAGE在COCO box AP 62.8、mask AP 55.3、TinyImageNet Top‑1 94.76%、ADE20K mIoU 60.3、DIOR‑R AP_50 78.8、COCO Keypoint 78.5等指标均达到或超过SOTA；

**⚠️ 局限性**

局限性包括对预训练数据和训练配置差异的敏感，难以完整归因；需要针对不同任务选择解码器；在极低资源或极大模型规模下的表现未知；依赖CLS‑token saliency，若ViT预训练不足可能影响效果。

---

## 360. A Harness Reducing over-Defensive Writing in Agents by Isolating Objections

**arXiv ID:** 2610.36717 | [PDF](https://arxiv.org/pdf/2610.36717v1)

**作者:** Xiaochen Chang `[一作]` `[通讯]`, Xiaochen Chang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种通过将审稿、筛选和写作拆分为独立子代理来减少过度防御性写作的框架。

**💡 创新点**

创新点在于实现上下文隔离，使用“原始材料+批准的审稿点”边界来限定写作内容，从而将防御性声明从59降至2，保留源声明率提升至93.12%。

**🔧 技术方法**

技术包括：多代理架构（Reviewer、Filter、Writer）、基于概率阈值的审稿点筛选、原始材料边界指令、支持性与非支持性写作分离。

**📊 数据集**

使用25份商业项目草稿（约500字每份）作为实验数据集。

**📈 对比分析**

与直接写作、单纯禁止防御写作等基线对比，防御性声明数量下降96.61%，主张保留率提升12.77个百分点，且无不支持事实加入。

**⚠️ 局限性**

局限性包括仅测试25个样本、评估依赖LLM评估器、缺乏人工标注验证、对不同文本类型的通用性尚未充分验证。

---

## 361. PrivacyEvalClaw: A Tool-Augmented Deliberative Agent for Auditing Privacy Leakage in LLM Jailbreak Responses

**arXiv ID:** 2610.36711 | [PDF](https://arxiv.org/pdf/2610.36711v1)

**作者:** Krishna Kanth Nakka `[一作]` (Huawei Heisenberg Research Center), Xuebing Zhou `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种名为PrivacyEvalClaw的训练无关的推理代理，用于评估大语言模型在jailbreak攻击下的隐私泄露。

**💡 创新点**

创新点在于将评估过程改为一个基于工具的多轮推理过程，主动收集证据、解决评估者分歧并生成可审计的决策轨迹，而不是单一的判定。

**🔧 技术方法**

技术包括工具调用框架、guard模型、LLM-as-Judge、解码器、PII检测器、专门的隐私推理器和争议分析器，并使用Qwen系列模型作为背骨。

**📊 数据集**

使用了125k条jailbreak样本，涵盖70个PII属性、50个数据主体和55种攻击策略，并在350条样本上进行评估。

**📈 对比分析**

通过与无工具、仅CoT以及组件剔除的基线进行pairwise LLM-as-Judge比较，PrivacyEvalClaw在整体和过程质量上分别取得82.86%和98%+的胜率，并与人工标签的kappa值达0.708，表明性能显著优于传统评估。

**⚠️ 局限性**

局限包括工具集不完整（如未知编码无法解码）、依赖同一模型作为代理和评估器、未覆盖更广泛的隐私风险场景以及工具调用策略仅由系统提示驱动而非经验学习。

---

## 362. Conservative Dueling Bandits: Preference Learning under Baseline Constraints

**arXiv ID:** 2610.36708 | [PDF](https://arxiv.org/pdf/2610.36708v1)

**作者:** Rohan Deb `[一作]`, Arindam Banerjee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出并实现了两类保守（conservative）的情境二项对抗（dueling bandit）算法，能够在仅通过二元偏好反馈的情况下保证学习过程的安全性。

**💡 创新点**

创新点在于：①将安全约束改为对偏好得分的累计约束，避免对绝对奖励的依赖；②针对线性和通用效用函数分别设计了基于效用差距的安全认证机制；③引入了“基础设施（baseline）”伴随比较的落后（fallback）策略，使得在安全受限时仍可持续学习。

**🔧 技术方法**

主要技术包括：高斯分布的极大似然估计与协方差置信椭圆；对偏好模型的贝叶斯/最大似然回归；逆间隙加权（Inverse-Gap-Weighting, IGW）采样；在线逻辑回归（logistic regression）oracle；以及安全预算的累积更新和任意时间安全性分析。

**📊 数据集**

使用了合成实验数据，分别构造了线性效用（d=10，K=50）和通用非线性效用（d=5，K=40，|ℱ|=64）两类任务，并在多种 α、d、K 参数设置下进行评估。

**📈 对比分析**

实验对比了保守算法与其无约束版本（MaxInP、BT-IGW）以及仅使用基线的策略。保守算法在所有设置下均保持零安全约束违背，而无约束版本在初期常出现违背。相对无约束版本，保守算法的累计失误率略高（增幅为 O(1) 与 T 的增量项），但在长期运行中增益可忽略不计。

**⚠️ 局限性**

局限性包括：①仅在 Bradley–Terry 结构下实验，无法覆盖非可传递或多项比较的情形；②假设基线得分已知或有已知下界，实际场景中可能需在线估计；③未给出保守性下的下界分析；④仅在合成数据上验证，缺乏真实应用场景测试。

---

## 363. An Algebra of Transformers

**arXiv ID:** 2610.36795 | [PDF](https://arxiv.org/pdf/2610.36795v1)

**作者:** Yong Wang `[一作]`, Hexiao Kang `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于算子分解的Transformer代数框架，定义了原子动作、守卫、并发与序列组合，并通过BAT、APT等代数系统实现对Transformer行为的形式化推理与证明；随后利用该框架对三种典型Transformer构造（Most‑Recent Induction、Dyck‑1、Dyck‑1‑2）进行符号化验证，证明它们满足预期的输入‑输出行为。

**💡 创新点**

创新点在于：
- 将Transformer的实现细节抽象为可组合的原子构造并给出完整的算术公理体系；
- 引入并发算子⊕并证明其与序列算子∘、冲突算子+的交互性，构建了完整的并发Transformer代数APT；
- 通过递归、隐式步τ和抽象τ_I的组合，提供了从内部实现到外部行为的完整推理路径；
- 在此代数框架下完成了对三种Transformer实例的完全形式化验证，展示了从实现到性质证明的闭环。

**🔧 技术方法**

主要技术包括：
- 先验的Transformer分解理论（将Transformer拆解为向量/矩阵运算原子）；
- 过程代数与等值逻辑（定义算子、公理、重写规则）；
- pomset labelled transition system 与 pomset bisimulation 用于描述并发行为；
- 递归规范与守护递归（guarded recursion）保证无限行为的可建模性；
- 隐式步骤 τ 与抽象 τ_I 的操作来实现内部细节的隐藏与公平性约束。

**📊 数据集**

本工作为理论研究，不依赖具体数据集；验证工作仅涉及三种人工构造的Transformer例子（Most‑Recent Induction、Dyck‑1、Dyck‑1‑2），其“数据集”仅为符号序列与相应的数值或布尔输出。

**📈 对比分析**

与传统基于实验的性能评估不同，本论文不做数值对比。方法是：给出完整的算子公理与重写系统，证明其完备性与一致性；随后通过归约与抽象将实现层的状态转移等价于外部行为方程。若要评估，可将该代数模型映射到实际Transformer实现并验证等价性，但本文未进行此实验。

**⚠️ 局限性**

局限性包括：
- 代数框架依赖于先前的Transformer分解结果，若存在新的内部模块（如自适应注意力等）需重新定义原子构造；
- 证明过程高度符号化，缺乏自动化工具支持，人工书写与验证成本较高；
- 只对有限数量的Transformer实例进行验证，未证明对所有Transformer体系的普适性；
- 论文重点关注行为等价性与形式化验证，未讨论效率、可扩展性或实际实现的性能问题。

---

## 364. Visual Autoregressive Generation with Next Token Imagination

**arXiv ID:** 2610.36793 | [PDF](https://arxiv.org/pdf/2610.36793v1)

**作者:** Ruihang Jiang `[一作]`, Xiatian Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在视觉自回归（AR）图像生成中，提出了一种新的训练范式——Next Token Imagination（NTI），通过让模型在非真实标记（non‑GT）上下文中学习可行的下一令牌来扩展生成空间；

**💡 创新点**

核心创新在于：① 将传统的 GT‑only 监督转变为基于上下文的可想象性 token 监督；② 引入 Context‑Regulated Imagination（CRI）来根据当前上下文动态分配非 GT token 的权重；③ 采用 Snapshot‑Based Context Approximation（SCA）以单前向传递高效生成模型自我生成的上下文；该三者结合形成可插拔、无架构改动的 NTI 训练方案；

**🔧 技术方法**

技术方法包括：视觉离散化 tokenizer（MaskGIT‑VQGAN 等）、因果 Transformer、代码书相似度（cosine similarity）构造非 GT 候选、总变差（TV）一致性评估、熵与概率结合的 imagination 强度计算、NTI 损失函数以及基于 SCA 的上下文近似；

**📊 数据集**

主要实验数据集为 ImageNet‑1K（256×256 级别，按类别条件生成）；此外在 GigaTok 与 AliTok 上验证 NTI 的通用性；

**📈 对比分析**

对比评估：在标准 AR-L 基础上，NTI-L 将 FID 从 3.02 降至 1.67，IS 从 256.20 提升至 312.72；在同等参数量（461M）下，NTI-L 的性能优于 reAR‑L、RAR‑L、LlamaGen‑L、BitDance‑B‑1x 等主流 AR 模型，并接近甚至超越部分 diffusion 模型；在 GigaTok 与 AliTok 上分别实现 FID 3.94→3.66 与 1.61→1.51 的提升；训练开销仅增加约 10%；

**⚠️ 局限性**

局限性包括：① 目前仅在视觉自回归框架下验证，尚未直接扩展到连续 AR 或视频等领域；② 需要额外调优的超参数（如 ρ_max、τ、λ）和定期快照刷新策略；③ 对极端 off‑GT 上下文的可行性仍有限，过度替换可能导致性能下降；

---

## 365. The Content Leash: Mitigating Degenerate Solutions in Invertible Speaker Anonymization

**arXiv ID:** 2610.36768 | [PDF](https://arxiv.org/pdf/2610.36768v1)

**作者:** Nirjon Mobarak Talukder `[一作]` (Islamic University of Technology), Abu Raihan Mostofa Kamal `[通讯]` (Islamic University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在可逆语音匿名化任务中，提出了内容保留的训练约束与知识蒸馏的声学鉴别器，以防止模型仅通过降低语音质量实现匿名化，并实现可恢复语音的逆向映射。

**💡 创新点**

创新点在于发现并解决了传统triplet‑loss导致的“degenerate”解；通过加入anchor loss和range loss约束匿名化频谱；以及将ECAPA‑TDNN声学鉴别器蒸馏到mel域以提升对未见鉴别器的鲁棒性。

**🔧 技术方法**

使用的核心技术包括条件可逆神经网络（cINN）、triplet loss、anchor loss、range loss、log‑determinant 正则化、ECAPA‑TDNN 与 HiFi‑GAN 的知识蒸馏、时间延迟网络（TDNN）以及HiFi‑GAN vocoder。

**📊 数据集**

训练与评估均使用VCTK与LibriTTS子集；评估集为LibriTTS中38个未见说话人，共303句；使用HiFi‑GAN vocoder进行合成。

**📈 对比分析**

相较于McAdams基线与原始Rano实现，所提方法在未见说话人上将ECAPA‑TDNN EER从2.83%提升至26.31%（约9.3倍提升），WER保持在11.75%（相对floor 2.48%），恢复相似度达94.9%，并在持有者键错误时保持约4%的平坦响应。

**⚠️ 局限性**

局限性包括：EER受蒸馏精度限制；评估仅覆盖单一语料与单一攻击模型；未测量声学独特性与音高相关度；恢复需对频谱进行无损存储；以及对不同vocoder的评估结果略有差异。

---

## 366. The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation

**arXiv ID:** 2610.36763 | [PDF](https://arxiv.org/pdf/2610.36763v1)

**作者:** Ziluowen Luo `[一作]` (Central South University), Senzhang Wang `[通讯]` (Central South University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出API框架，将预训练稀疏自编码器（P‑SAE）的可解释特征字典作为生成式推荐的词表，实现无额外词表学习的推荐与解释。

**💡 创新点**

创新点在于利用已有P‑SAE特征作为可解释词表并通过PID与SPA两步适配，既保持词表可解释性，又提升推荐性能并提供特征级解释。

**🔧 技术方法**

使用预训练P‑SAE、GPT‑2/T5、Transformer、序列预测+强度预测、Hellinger距离等技术。

**📊 数据集**

使用Amazon Beauty、Sports、Toys三大子集（含标题、品牌、类别、价格）作为测试数据。

**📈 对比分析**

与随机、OPQ、HiD‑VAE、RVQ、RQVAE、RKMeans等学习词表以及TIGER、RPG等基准进行对比，API在Recall@5/10、NDCG@5/10上大多数场景达到最优或第二优，且词表冲突率低、内容保留好。

**⚠️ 局限性**

局限在于依赖预训练P‑SAE的可解释特征质量，词表覆盖仍受预训练数据限制；对高维长文本和动态商品更新的适应性待验证。

---

## 367. Multi-task Disentangling Fusion Mamba-CNN for MODIS Spectral-Temporal Unmixing with Endmember Variability Modeling

**arXiv ID:** 2610.36750 | [PDF](https://arxiv.org/pdf/2610.36750v1)

**作者:** Motasem Alkayid `[一作]` (University of Calgary), Lincoln Linlin Xu `[通讯]` (University of Calgary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究提出并实现了一种多任务分解融合Mamba模型（MDF‑Mamba），专门用于MODIS光谱‑时间序列的亚像素混合分析，目标是精准估计土地覆盖分数、构建光谱库并实现光谱重建与分类。

**💡 创新点**

创新点包括：① 通过异构的多分支专家（六个通道卷积专家＋双向Mamba时间分支）实现光谱‑时间去耦合，并用Squeeze‑and‑Excitation实现专家间软重加权；② 引入基于流形约束的超连接融合（mHCFusion），以稳定且可解释的方式融合多流特征；③ 扩展线性混合模型为可学习端元变体（ELMM），捕获类内光谱变异；④ 设计联合监督与自监督的多任务学习框架，结合重建、端元正则、稀疏性等损失，并采用两阶段课程学习。

**🔧 技术方法**

技术栈包括：Mamba选择性状态空间模型（SSM）、双向SSM、卷积双路径专家、Twist门控、Squeeze‑and‑Excitation、流形约束的双边斯金农投影、FiLM、ELMM端元可变性、注意力重构、马氏距离与角度距离自监督重建、拉普拉斯光谱平滑以及课程学习策略。

**📊 数据集**

实验数据为MODIS MOD13Q1 2010年全年16日合成，覆盖加拿大Alberta和Saskatchewan两省，包含NDVI、EVI、Red、NIR、Blue、MIR共6通道，每通道23个时间点，总计138通道；通过NALCMS 30m标签生成分数，最终使用14类（去除两类苔藓类）进行训练与评估。

**📈 对比分析**

模型与11种基线（传统线性LSU、SUnSAL、SGSNMF；深度AE PGMSU、SUnCNN、MiSiCNet；注意力SSA、PPM；Transformer DeepTrans‑HSU、SRViT；SSM UNMamba）在14个评估指标上进行对照。MDF‑Mamba在12/14指标上表现最佳，尤其在平均绝对误差MAE(0.0255)、RMSE(0.0654)、重建MSE(0.0067)、SAD(0.1843 rad)、整体准确率OA(0.7122)、平均准确率AA(0.4721)和κ(0.6575)上领跑；在Saskatchewan的跨域验证亦保持高精度。Ablation实验验证了各模块的贡献。

**⚠️ 局限性**

局限性包括：端元可变性依赖于纯像素初始化，某些类在MODIS分辨率下缺乏可区分的纯像素，导致端元学习受限；模型结构复杂，参数量与计算量相对较大；对极端光谱噪声或极端混合情况的鲁棒性待进一步评估；在低光谱分辨率下对光谱指数的线性混合假设可能不足；训练过程需要两阶段课程学习，调参较为繁琐。

---

## 368. Beyond LWE: Post-Quantum Assumptions under a Counterfactual Break of the Dihedral Coset Problem

**arXiv ID:** 2610.36731 | [PDF](https://arxiv.org/pdf/2610.36731v1)

**作者:** Ramses Fernandez `[一作]` `[通讯]` (AegisQ Technologies), Ramses Fernandez (AegisQ Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理并评估了若存在高效量子算法求解Dihedral Coset Problem（DCP）所带来的后量子密码学安全影响，重点分析不同DCP求解假设（无噪声、失败指数、EDCP）对LWE、模块-LWE、SIS、代码方案等的可行化学路径与参数阈值。

**💡 创新点**

创新点在于：①将多种DCP求解假设分类并量化其样本、误差与成功率要求；②利用Regev的稀疏破坏观测与EDCP状态转换，给出从DCP到晶格问题、LWE、SIS等的精确化学映射；③构建参数与阈值表，标记已知的理论缺口与未验证的安全断言。

**🔧 技术方法**

采用的技术包括：隐子群问题的量子傅里叶测量与子集和模板；Regev的稀疏破坏理论和量子随机行走筛选；信息理论距离上界；以及结构化EDCP的状态转换与前向证书审核。

**📊 数据集**

本文未使用实验数据集，所有分析均基于理论推导和已公布的安全参数集；通过对比已发表的安全参数（如ML-KEM、ML-DSA、NIST标准）进行评估。

**📈 对比分析**

方法通过阈值和参数表格与已知经典/量子算法的运行时间和近似因子进行对比；结果表明在多项式噪声和有限样本条件下，DCP求解并不能立即导致对已部署LWE参数的突破，且对部分方案的安全性提升仍缺乏完整证明。

**⚠️ 局限性**

限制包括：①对DCP求解的假设过于理想化，实际算法未被证明；②所需样本量或无噪声条件不满足已部署参数；③结构化EDCP与标准LWE的等价仅在严格条件下成立；④经典结构化攻击仍能威胁部分方案；⑤本文仅给出充分条件，未提供必要条件或完整安全证明。

---

## 369. A fully articulated humanoid dexterous hand with human-level joint structure enabled by linkage-based actuation

**arXiv ID:** 2610.36789 | [PDF](https://arxiv.org/pdf/2610.36789v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 370. SMEBENCH: Benchmarking Neural Music Editing with Synthetized MIDI Edits

**arXiv ID:** 2610.36715 | [PDF](https://arxiv.org/pdf/2610.36715v1)

**作者:** Matthieu Cervera `[一作]` (Quebec Artificial Intelligence Institute), Cem Subakan `[通讯]` (Quebec Artificial Intelligence Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了SMEBench，一个配对源音频与目标音频的音乐编辑基准，利用可控的MIDI符号干预和对应文本描述提供可重复的编辑测试。

**💡 创新点**

创新点在于引入配对目标，让编辑质量可分别评估来源保留、编辑执行与文本兼容性，并提供12种操作的平衡测试集及真实录音补充。

**🔧 技术方法**

技术上结合MIDI编辑、FluidSynth渲染、CLAP/LPAPS相似度、Audiobox Aesthetics等指标，并用LLM生成与审核字幕实现自动化。

**📊 数据集**

数据集来源于168,385条MIDICaps MIDI文件与MoisesDB多轨录音，最终生成11,160条配对编辑与416条真实音频配对。

**📈 对比分析**

对六种编辑系统（SDEdit、DDIM、ZETA、VCI、SteerMusic、MusicGen）在源、目标、文本三种评估轴上进行排名，结果显示文本相似度与目标相似度往往产生排名冲突，提示单一指标不充分。

**⚠️ 局限性**

局限性包括合成音频与真实录音的差距、仅覆盖12种MIDI可编辑操作、部分编辑在听觉上不明显、以及过滤过程受预训练模型偏差影响。

---

## 371. Sensitivity of RMS Delay Spread Estimation to Power Delay Profile Thresholding in Time-Varying Channels at 33 GHz

**arXiv ID:** 2610.36783 | [PDF](https://arxiv.org/pdf/2610.36783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 372. Is Token Confidence Just Chain Length? A Signal-Processing Audit of Reasoning Traces

**arXiv ID:** 2610.36709 | [PDF](https://arxiv.org/pdf/2610.36709v1)

**作者:** Pragna Mallappa `[一作]` `[通讯]` (Keshav Memorial Institute of Technology), Pragna Mallappa (Keshav Memorial Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 DeepConf 的置信度链过滤方法进行信号处理分析，验证其主要衡量的是链长度而非正确性，且探索了替代滤波器与变更检测器。

**💡 创新点**

将 token 置信度视作离散时间信号，用频谱估计揭示其 1/f 结构，解释 DeepConf 最小滑动平均与长度相关的原因，并提出长度基准评估。

**🔧 技术方法**

使用 Welch 频谱估计、滑动平均（boxcar、Hann、指数移动平均）、CUSUM 变更检测、逻辑回归滤波器能量检测，以及 AUROC 评估。

**📊 数据集**

在 200 个 MATH500 和 300 个 GSM8K 题目上，使用 DeepSeek‑R1‑Distill‑Qwen‑1.5B 生成 16 条链进行实验。

**📈 对比分析**

与链长度排序、平均置信度、DeepConf 各分数、不同窗口长度与形状、学习滤波器能量检测以及在线早停策略进行比较，发现长度基准的 AUROC 最好，其他方法未明显提升准确率。

**⚠️ 局限性**

仅在单一小模型、固定温度与 k 值、两套数学基准、混合问题样本有限以及仅对已完成链进行早停模拟下得到结果，可能不适用于更大模型或更复杂问题。

---

## 373. Translating Poetic Emotion into Dynamic Mathematical Visuals: A Strange Attractor-Based Approach

**arXiv ID:** 2610.36698 | [PDF](https://arxiv.org/pdf/2610.36698v1)

**作者:** Subhadra Nayak `[一作]` `[通讯]` (Indian Institute of Technology), Subhadra Nayak (Indian Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套将诗歌情感映射到奇异吸引子动力学生成动态视觉的原型系统

**💡 创新点**

首次将情感分类与数学奇异吸引子参数一一对应，形成可预测的情感可视化管线

**🔧 技术方法**

使用RoBERTa情感分类器、TouchDesigner粒子模拟、GLSL实现四种三维奇异吸引子（Thomas、Lorenz、Chen‑Lee、Halvorsen）

**📊 数据集**

利用GoEmotions数据集预训练的RoBERTa模型（28类情感）对诗歌文本进行情感提取

**📈 对比分析**

通过Pilot实验评估视觉与文本情感的一致性，结果显示匹配率仅约23%，未给出定量性能指标

**⚠️ 局限性**

局限包括样本量小、情感分类模型对诗歌的敏感性不足、情感与视觉参数单独映射导致整体表现失真

---

## 374. OmniGlimpse: Asymmetric Identity Memory for Online Multi-View Tracking

**arXiv ID:** 2610.36785 | [PDF](https://arxiv.org/pdf/2610.36785v1)

**作者:** Rangya Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 OmniGlimpse 框架，解决了无离线训练、无相机标定下的在线多视角身份维护问题。

**💡 创新点**

创新点在于引入不对称的 propose–verify 结构：短期局部与跨视角证据用于关联建议，长期身份分布在固定编码空间中进行验证，并通过验证门控更新持久记忆。

**🔧 技术方法**

使用目标特定在线适配器、冻结的 ReID 编码器、岭回归分类器、三元组损失、Gaussian 原型长时记忆、可靠性控制器、对比适配等技术。

**📊 数据集**

在 M3Track 与 MvMHAT 两个多视角跟踪基准上进行实验。

**📈 对比分析**

与 OCLReID、OC‑SORT+RPF、ByteTrack、BoT‑SORT+ReID 等在线基线及 CrossMOT、MvMHAT、AGW 等离线参考对比，取得了 IDF1 与 CVIDF1 的显著提升（例如 M3Track 上 IDF1 从 0.7424 提升至 0.8333，CVIDF1 从 0.3918 提升至 0.6129），在在线条件下已达到或逼近离线参考水平。

**⚠️ 局限性**

局限性在于仅针对已预先指定且每视角仅初始化一次的少量目标进行评估；扩展到目标密集或初始化信息更弱的场景仍需进一步研究。

---

## 375. Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better

**arXiv ID:** 2610.36774 | [PDF](https://arxiv.org/pdf/2610.36774v1)

**作者:** Jialin Zhang `[一作]` (University of California, Berkeley), Matei Zaharia `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在GEPA反射优化器中引入批量并行提议机制，使每一步能同时生成并评估多条候选路径，显著降低墙钟时间并提升搜索效果。

**💡 创新点**

创新点在于把单突变逐步适应改为一次性提议多个Pareto前沿子代并行评估，减少适应次数、提高搜索覆盖率，同时通过并行化提高资源利用率。

**🔧 技术方法**

采用批量GEPA、Pareto前沿采样、反射突变、并行评估调度以及基于工作池的runtime模型预测等技术实现批量化和性能提升。

**📊 数据集**

使用 LiveBench‑Math（数学竞赛问题求解器提示优化）和 HoVer（四跳检索验证任务）两个数据集进行实验。

**📈 对比分析**

与单突变（单候选）在相同metric‑call预算下比较，批量设置将墙钟时间缩短约3–4倍，在 LiveBench‑Math 上测试准确率从 68.9% 提升至 72.1%，在 HoVer 上提升至 60%（比单突变提升 11%）。

**⚠️ 局限性**

受限于工作池容量与全验证开销，宽度提升到一定程度后收益递减；目前实验仅覆盖两种任务，未深入探究父节点与突变数的最佳分配以及更大规模工作池的潜在效果。

---

## 376. Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs

**arXiv ID:** 2610.36767 | [PDF](https://arxiv.org/pdf/2610.36767v1)

**作者:** Azim Ospanov `[一作]` (Huawei Foundation Model Department), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大语言模型在奥林匹克级数学证明验证中的表现，提出了人类标注的错误注入数据集、自动化对抗性证明生成管道以及基于多臂赌博机的评分框架。

**💡 创新点**

创新点在于①构建了可控错误注入的USAMO 2025验证基准；②提出自我批判式对抗证明生成器提升数据规模；③用多臂赌博机将评分视为最优选择问题，显著提高与人工评分的一致性。

**🔧 技术方法**

主要技术包括大语言模型推理（CoT、思考模式）、自我批判与迭代修正、奖励模型（PRM/ORM）、多臂赌博机（UCB）与贝叶斯优化。

**📊 数据集**

使用的数据集为：①人类标注的48条USAMO 2025证明（含人工评分与错误定位）；②由自我批判管道生成的四项竞赛（USAMO 2025/2026、IMO 2025、IMC 2025）合成数据。

**📈 对比分析**

与现有基线（零样本CoT、MajorityVote、Best‑of‑N奖励模型）比较，所提方法在QWK上提升约26.4%、RMSE下降14%，在20个模型中均表现出最优或次优一致性。

**⚠️ 局限性**

局限性在于：需先验的评分标准与参考解；在缺失参考解的Blind情形下无法直接应用；对非竞赛题目或开放式验证的可迁移性有限。

---

## 377. Beyond the Evidence Coherence Trap: Agent-Driven Evidence Selection for Temporally Disordered Fake News Detection

**arXiv ID:** 2610.36764 | [PDF](https://arxiv.org/pdf/2610.36764v1)

**作者:** Chi Wang `[一作]` (Chongqing University), Wentao Li `[通讯]` (University of Leicester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向时序混乱假新闻的检测框架 DelSA，结合事件级证据检索、LLM 推理与自更新的规则库，实现对文章整体时间线的重构与一致性判定。

**💡 创新点**

创新点在于构建了“证据一致性陷阱”概念，并通过自更新代理与可持续的规则书，使模型在训练阶段动态修正事件与证据的时序对应关系，解决了传统检索式方法无法处理多时点歧义的局限。

**🔧 技术方法**

技术手段包括基于 BGE‑M3 的语义与实体检索、FAISS 索引、GPT‑5.4‑mini 的多维度质量评分、MLP 重排器、Bi‑LSTM 与位置编码的时序编码、以及基于注意力的聚合与 MLP 分类头。

**📊 数据集**

实验数据集包含自构建的 ChronoFake（30,706 篇，其中 9,925 篇时序错误假新闻）、PolitiFact（741 篇）和 GossipCop（12,252 篇）。

**📈 对比分析**

与内容基、LLM 增强和检索增强的 10+ 传统基线对比，DelSA 在 ChronoFake 上达 0.892 的最高准确率，PolitiFact 0.932，GossipCop 0.932/0.863，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括对大型 LLM 的高算力依赖、对时间表达解析的规则敏感性、以及在多语言或多模态场景下的泛化能力尚未充分验证。

---

## 378. GeoGSplat: Geometry-Grounded Non-Pixel-Aligned Gaussian Splatting

**arXiv ID:** 2610.36760 | [PDF](https://arxiv.org/pdf/2610.36760v1)

**作者:** Jiale Wang `[一作]` (Xi'an Jiaotong University), Tong Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 GeoGSplat，一种利用几何先验的非像素对齐 Gaussian Splatting 框架，实现高质量新视角合成与三维重建。

**💡 创新点**

创新点：1）双分支结构编码器，Anchor 提供粗几何约束，Triplane 提供细粒度特征；2）层次解码器，先将 Anchor 转化为稠密 3D 点，再基于 Triplane 预测 Gaussian；3）通过几何损失（Chamfer）与光度损失共同训练，固定 Gaussian 数量同时获得优异几何与外观。

**🔧 技术方法**

使用技术包括 Vision Transformer 编码器、Plücker ray 嵌入、双分支注意力聚合、Anchor（可学习球体）与 Triplane（三维平面特征图）构造、Anchor→点解码、点→Gaussian 解码、Bilinear 采样、Chamfer 距离损失、SSIM、LPIPS 等光度损失。

**📊 数据集**

训练集：ScanNet++、ARKitScenes、BlendedMVS、CO3D、WildRGB-D；评估集：DTU（对象级）与 NRGBD（场景级）。

**📈 对比分析**

与像素对齐方法（MVSplat、DepthSplat、AnySplat）以及非像素方法（TokenGS、VolSplat）对比，GeoGSplat 在 DTU 与 NRGBD 上均取得最高 PSNR 与最低 Chamfer；尤其在视角相距较远时表现更优。

**⚠️ 局限性**

局限性：模型规模较大，推理时仍需较多 GPU；对极端光照、遮挡与动态场景的鲁棒性待进一步验证；固定 Gaussian 数量在极大视角跨度下仍可能出现细节缺失。

---

## 379. Axiom: a logic programming language to express and enforce network designs

**arXiv ID:** 2610.36748 | [PDF](https://arxiv.org/pdf/2610.36748v1)

**作者:** Tibor Schneider `[一作]` (ETH Zurich), Laurent Vanbever `[通讯]` (ETH Zurich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于 Datalog 的网络设计语言及检查器，能够直接在配置文本层面验证网络设计的正确性，并定位违规配置行。

**💡 创新点**

创新点在于将网络“设计”显式为一阶公理，通过递归 Datalog 直接在配置文本上进行检查，实现了高效、精准的错误定位，弥补了现有验证器在模型精度、可扩展性和调试支持上的不足。

**🔧 技术方法**

采用 Datalog 及其扩展（层次化、格语义、分层求值、半稽留评估）实现检查器；使用配置解析、模板匹配生成中间表示，再转化为 Datalog 程序；原型实现基于 Rust 并自研 Datalog 引擎。

**📊 数据集**

在 TopologyZoo 公开网络拓扑（最多 754 路由器）上自动生成 Cisco 配置，覆盖 OSPF、iBGP 等，生成约 4 百万行配置进行评估。

**📈 对比分析**

与 Batfish 等传统模拟/验证工具对比，Axiom 在同一网络下只需几秒（754 路由器时 <3 秒，20 路由器时 <0.3 秒），而 Batfish 需数十分钟；在特定设计下检查时间比通用设计低 2–10 倍，内存占用也显著减少。

**⚠️ 局限性**

局限性包括：语言尚未完整支持顺序过滤器、正则表达式解析和完整的负向证明；设计编写仍需人工；在极大规模（如全网 iBGP 全网）网络下受输入规模限制；目前缺乏增量评估和自动修复建议功能。

---

## 380. Progressively Refining SAM for Accurate Phytoplankton Segmentation

**arXiv ID:** 2610.36722 | [PDF](https://arxiv.org/pdf/2610.36722v1)

**作者:** Jialin Li `[一作]`, Junyu Dong `[通讯]` (Ocean University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于SAM2的Phytoplankton细粒度分割框架PhySeg，并构建了新的分割数据集PhySegD

**💡 创新点**

创新点在于加入轻量级的渐进式细化模块，利用多尺度特征融合逐步恢复细节，并通过LoRA微调SAM2

**🔧 技术方法**

使用了SAM2、LoRA、轻量级残差头、特征融合模块、焦点损失与Dice损失等技术

**📊 数据集**

使用了自建的Phytoplankton Segmentation Dataset (PhySegD)，包含9824张图像和11990个实例

**📈 对比分析**

与CascadePSP、BPR、CRM、HQ‑SAM2等方法在PhySegD上对比，PhySeg在mIoU、Dice、BF1、BF2等指标均取得最高或接近最高成绩

**⚠️ 局限性**

仍存在边界细化不够完美（BF4略低）以及数据集规模有限、对不同水体或显微镜设置的泛化性待验证等限制

---

## 381. Nonlinear Direct Feedback Alignment for Scalable Backpropagation-Free Training

**arXiv ID:** 2610.36718 | [PDF](https://arxiv.org/pdf/2610.36718v1)

**作者:** Wu Lin `[一作]` (Shenzhen University), Qiuzhen Lin `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种称为非线性直接反馈对齐（NDFA）的训练算法，利用可学习的轻量级非线性子网络为每层生成反馈信号，从而克服传统DFA中固定线性反馈矩阵导致的性能瓶颈。

**💡 创新点**

创新点在于：①引入可学习的非线性子网络近似BP中隐藏层到输出的下游子网络，从而使反馈具备表示依赖性和非线性特征；②在训练过程中同时更新子网络，使其能够随网络演化动态逼近真实下游变换；③在保持并行训练优势的前提下显著缩小DFA与BP之间的性能差距。

**🔧 技术方法**

核心技术包括：直接反馈对齐（DFA）框架；轻量级多层感知机（MLP）子网络作为非线性反馈生成器；基于输出误差的反馈更新公式；子网络通过与真实下游子网络相同的损失函数进行监督学习；并行化实现以保持无后向锁定（backward locking）优势。

**📊 数据集**

实验数据集：CIFAR-10、CIFAR-100、SVHN、ImageNet；网络架构：ConvNet、AlexNet、ResNet‑32、ViT‑Tiny。

**📈 对比分析**

对比方法包括BP、DFA、DKP、PEPITA、FDFA。NDFA在所有实验中均优于所有基线，尤其在ConvNet、AlexNet和ResNet‑32上与BP相差仅3–6%，在ViT‑Tiny上与DFA相差超过90%；同时相较于传统DFA，NDFA实现了1–2倍的准确率提升，并在并行训练中保持与BP相近的加速比（约2.5–4×）。

**⚠️ 局限性**

局限性：①对极深网络仍存在一定性能差距；②引入子网络会带来额外计算开销，虽可与前向计算重叠但在现有实现中仍略慢；③缺乏理论收敛性分析；④对更复杂或更大规模模型的适用性尚未充分验证。

---

## 382. Does LLM Pre-Training Typically Occur at the Edge of Stability?

**arXiv ID:** 2610.36778 | [PDF](https://arxiv.org/pdf/2610.36778v1)

**作者:** Yuhang Cai `[一作]` (University of California Berkeley), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型训练中二次近似的失效模式，并提出了“凸性边缘”(EoC)与“稳定性边缘”(EoS)两种新概念，扩展了EoS在随机、适应性优化器中的定义；

**💡 创新点**

首次识别并系统化二次近似在存在负曲率时的LR无关失效(EoC)，以及在正曲率子空间中由随机噪声引发的LR临界失效(EoS)，并给出可扩展的检测方法；

**🔧 技术方法**

利用局部二次模型、谱分析、负最小特征值估计、正向/负向投影、随机幂迭代、批量梯度模拟以及对AdamW等适应性优化器的理论推导；

**📊 数据集**

在FineWeb、OpenWebMath等大规模文本数据集上，对43M~787M GPT 规模模型进行预训练，评估了不同学习率调度和批量大小对EoC/EoS的影响；

**📈 对比分析**

与传统的基于λmax的EoS判定方法对比，所提出的EoC/EoS检测在数千个检查点上显著更可靠；实验显示，训练往往处于EoS边界，EoS消失时往往伴随性能提升，过高学习率则导致极端不稳定；

**⚠️ 局限性**

限制：实验仅覆盖SGD、AdamW和部分学习率调度；对其他适应性优化器（如MuProp、RAdam等）的EoS/EoC行为尚未验证；理论分析仍基于局部二次近似，无法完全捕捉全局非凸结构。

---

## 383. Making Self-Driving Laboratories Auditable:A Unified Framework for Hybrid Actions,Measurements, and Safe Execution

**arXiv ID:** 2610.36761 | [PDF](https://arxiv.org/pdf/2610.36761v1)

**作者:** Zihan Zhang `[一作]` (Shanghai Innovation Institute), Tong Zhu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个可追溯的自驱实验室（SDL）框架，包括 Lab Action Model (LAM)、OpenSDL、Safe-SDL 运行时门控以及 Level‑3 评估基准 (L3‑Bench)，实现了从视觉+测量输入生成连续机械臂轨迹与类型化仪器调用，并在执行过程中记录时序因果信息。

**💡 创新点**

创新点在于：① 将测量条件融入 VLA 方案，生成混合连续/离散动作；② 统一执行接口与安全监控（STL 约束、资源预检）；③ 设计可重放的事件账本和基准协议，支持仿真、HIL 与真实硬件的跨平台验证；④ 通过角色/设备提供者抽象实现设备迁移与可插拔性。

**🔧 技术方法**

使用技术包括：视觉‑语言‑动作 (VLA) 变体；流匹配连续动作生成器与函数调用头；Rust‑基 OpenSDL 控制引擎与 SiLA 2、PROV‑O 兼容的元数据模型；STL 监控、信号时序逻辑；Clean‑3D 资产与数字孪生；以及多任务调度与事件日志。

**📊 数据集**

数据集：基于实验批次的合成与硬件‑in‑the‑loop (HIL) 数据，包含任务脚本、视觉帧、机器人状态、仪器测量流、事件记录；划分为训练/校准/测试，支持设备迁移、测量漂移与故障注入情景。

**📈 对比分析**

比较方法：与基线 vision‑only 单臂策略和 LAM 去除测量标记的版本对比，评估指标包括物理产率 (Y)、执行时间 (C) 与安全违规率 (SVR)；在四种环境（标称、测量漂移、设备迁移、故障注入）下进行配对实验，报告平均值、置信区间及离散化检验。性能结果待发布，但协议已定义。

**⚠️ 局限性**

局限性：尚未完成模型训练与端到端验证；缺乏真实硬件安全认证与未知化学反应的处理；事件账本在权限控制和防篡改方面仍需加强；评估覆盖范围受限于可用设备与仿真环境，无法涵盖所有化学/生物任务；实现细节与实验结果需进一步补全。

---

## 384. Robust LLM Data Selection via Content-Invariant Gradient Attribution

**arXiv ID:** 2610.36755 | [PDF](https://arxiv.org/pdf/2610.36755v1)

**作者:** Jia Gu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Shen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CIGA 框架，在温热微调后挑选一致影响层并融合梯度与语义信息，用于指令微调的数据选择。

**💡 创新点**

创新点在于引入一致影响层选择以消除表面形式偏差，并采用动态特征融合（梯度+语义）实现稳健的数据评估。

**🔧 技术方法**

使用了 Adam 预处理梯度、层选择得分 S_l^ci、语义相似度、RRF 融合、基于梯度的相似度度量和少量温热训练等技术。

**📊 数据集**

训练池基于 FLAN‑v2、CoT、Dolly、OASST1、Alpaca‑GPT4、GSM8K；评估数据集为 TyDiQA、BBH、GSM8K、MMLU、HellaSwag。

**📈 对比分析**

与语义匹配、LESS、TracIn、IFD、SelectIT 等基线对比，CIGA 在 5% 子集上实现更高的 Jaccard 稳定度和宏平均得分，超过随机和全量训练。

**⚠️ 局限性**

主要限制在于需要对大量候选样本做全梯度计算、依赖高质量验证集，且计算开销仍较大。

---

## 385. Learning Side-by-Side Navigation for Personal Mobility Vehicles from Human Drivers

**arXiv ID:** 2610.36726 | [PDF](https://arxiv.org/pdf/2610.36726v1)

**作者:** Luis Yoichi Morales `[一作]`, Florent Ferreri `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种共享规划方法，使两辆个人移动式车辆（PMV）在无需直接通信、仅交换里程计信息的情况下，协同规划并保持侧旁式行驶，支持乘客之间的对话。

**💡 创新点**

创新点包括：①去中心化的联合决策，通过两辆车各自构建互为转置的效用矩阵实现协作；②仅使用可观测的里程计而非显式意图信息；③将形成几何由离线地图分析获得，实时根据实际障碍更新；④利用逆强化学习从人类配对驾驶数据学习行为参数。

**🔧 技术方法**

采用的技术：采样式模型预测控制（MPC）、最大熵逆强化学习、基于车辆全身的安全检查（刹车可行性与碰撞清晰度）、ROS 2仿真平台。

**📊 数据集**

数据集：10对人类驾驶员在室内走廊内共计 409 分钟（约 6.8 h）的配对驾驶记录，包含四种障碍布局与行进方向，每对共完成 8–10 条记录。

**📈 对比分析**

比较方法：在仿真环境中对照人类记录进行闭环评估，指标包括对话形成时间、侧向分离、进度、墙/障碍碰撞和停滞次数。结果显示系统在对话保持率、速度匹配方面超过人类，并且无碰撞或停滞；与人类的侧向分离分布相距较大但更稳定。

**⚠️ 局限性**

局限性：仅在仿真中验证，缺乏真实道路或户外实验；循环次数有限，无法估计低概率失效；形成仅采用单一侧向距离，未能再现人类在靠近与远离两种模式间切换；安全规则过于保守导致速度下降；轨迹候选集限制为车辆自身的圆弧，无法表达“减速后跟随”等更复杂的通过动作。

---

## 386. AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding

**arXiv ID:** 2610.36874 | [PDF](https://arxiv.org/pdf/2610.36874v1)

**作者:** Bingchao Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为AdaGround的训练无关GUI定位框架，自动调节缩放比例、验证点击结果并组合结构化目标

**💡 创新点**

创新点在于令模型自估目标尺寸来决定缩放、使用标记点击的Yes/No验证器进行决策、以及针对表格单元格的结构化定位

**🔧 技术方法**

采用Token‑Normalized Adaptive Zooming、Verifier‑in‑the‑Loop Refinement、Compositional Structure Grounding等技术，并以黑盒方式调用Qwen3‑VL模型

**📊 数据集**

在OSWorld‑G、ScreenSpot‑Pro、ScreenSpot‑v2以及隐藏测试集UI‑Vision四个公开GUI定位基准上评估

**📈 对比分析**

与MVP、ZoomClick等基线对比，AdaGround在大多数指标上提升3–7分，尤其在高分辨率屏幕上显著超越对手，且在持久测试集上保持竞争力

**⚠️ 局限性**

局限于仅在Qwen3‑VL‑8B/32B上验证，计算成本显著高于单通道方案，验证器对误报敏感，对结构化目标支持有限

---

## 387. Evaluating AlphaEarth Embeddings for Crop Classification: Separability, Temporal Dynamics, and Cross-Regional Transferability

**arXiv ID:** 2610.36857 | [PDF](https://arxiv.org/pdf/2610.36857v1)

**作者:** Priyanka Goyal `[一作]` (Indian Institute of Technology Kharagpur), Manjira Sinha `[通讯]` (Tata Consultancy Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估 AlphaEarth 64 维地理嵌入在作物分类中的表示质量，并提出了一个四维度（几何可分、时序动态、空间稳健性、跨区域泛化）代表性无关的评估框架。

**💡 创新点**

创新点在于将分离度、旋转监测、空间误差聚类与跨区对齐等多维指标统一评估，并揭示嵌入在分离度与多年作物旋转捕获上优于传统特征，但在跨区域迁移时存在显著折衷。

**🔧 技术方法**

采用 AlphaEarth 自监督基础模型嵌入、PCA、Bhattacharyya/Mahalanobis 距离、旋转可分离率、Moran's I、CKA/EMD、LORO 与块交叉验证，以及多种分类器（LR、MLP、TCN 等）进行实验。

**📊 数据集**

使用的主要数据集包括 CONUS 2017‑2023 年的 CDL 与 Sentinel‑2 影像、印度 2021 年的 AgriFieldNet 以及 AlphaEarth 嵌入，统一解析为 10 m 分辨率的场级特征。

**📈 对比分析**

通过对分离度、SDI、旋转可分离率、Moran's I、宏 F1 以及跨区宏 F1 进行比较，嵌入在宏 F1 与分离度上表现最佳，但在跨洲迁移时宏 F1 降至约 50% 以下，而谐波特征在跨区迁移上保持 40‑50% 的性能，显示出更强的泛化能力。

**⚠️ 局限性**

主要局限包括：未对 AlphaEarth 嵌入进行区域细化微调、印度样本仅覆盖单季单地区、单周期谐波拟合可能无法充分捕捉双季作物的复杂时序，以及跨区泛化需进一步域适配与多模态融合。

---

## 388. AxisQ: Depth-Adaptive Channel--Frequency Quantization for Camouflaged Object Detection

**arXiv ID:** 2610.36844 | [PDF](https://arxiv.org/pdf/2610.36844v1)

**作者:** Chi Ruan `[一作]` (SpinEntropy), Yun Liu `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种深度自适应的混合域后训练量化框架（AxisQ），专门针对伪装目标检测（COD）模型在低位宽下保持前景细节与边界。

**💡 创新点**

创新点在于：① 根据特征深度动态切换量化分配域——浅层使用通道域，深层使用局部频域；② 采用COD感知的通道 Fisher 成本和频域 GGN 敏感性进行比特分配；③ 引入边界加权的 Hessian 近似提升重建质量。

**🔧 技术方法**

使用的方法包括：通道 Fisher 成本、逆水填充比特分配、局部块 DCT 频域重建、COD‑GGN 任务敏感度、边界权重的 Hessian 重构、AdaRound、QDrop 等。

**📊 数据集**

实验数据集：CAMO、CHAMELEON、COD10K、NC4K。

**📈 对比分析**

与 LS‑ViT、BRECQ、AdaRound+QDrop、FIMA‑Q、APHQ‑ViT、RepQ‑ViT、COD‑TDQ 等方法对比，AxisQ 在 W2A2（2/2 位宽）下在所有四个数据集的宏观加权 F‑measure 取得 0.08‑0.13 的提升，整体占优；在更高位宽 W4A4 也保持领先或相当。

**⚠️ 局限性**

局限性：需要额外的校准样本和计算资源来估计 Fisher/GGN 成本；对不同网络结构的适配需要手动设定深度分割；在极低位宽（如 W1A1）下效果尚未验证。

---

## 389. ExecDiffuser: Closing the Sample-to-Execution Gap in Diffusion Planning

**arXiv ID:** 2610.36837 | [PDF](https://arxiv.org/pdf/2610.36837v1)

**作者:** Shuo Pei `[一作]` (Beijing Institute of Technology), Yong Wang `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归的提案‑至‑执行框架 ExecDiffuser，将生成候选轨迹、轨迹选择与重新规划联通，并在闭环驾驶中直接提交执行轨迹。

**💡 创新点**

创新点在于：① 通过 Intent‑Guided Diffusion Proposal (IGD) 实现执行条件下的责任分配与轨迹形状塑造；② 通过 Execution‑Grounded Selection (EGS) 学习基于当前执行上下文的候选评分，真正将可用候选转化为最终决策；③ 通过样本‑到‑执行缺口（sample‑to‑execution gap）的定义与匹配反事实评估，系统性地剖析生成与执行之间的不匹配。

**🔧 技术方法**

技术包括：扩散模型生成多模轨迹、潜在查询与噪声配对、可微分轨迹重构、执行证据编码（drivable, collision, risk, progress 等）、基于上下文的候选评分网络、连续性门控、预算分配网络以及不确定性增强的预测器。

**📊 数据集**

使用公开基准 nuPlan v1.1、InterPlan 以及 NAVSIM v1.1，全部采用相同训练种子与评测协议，保证实验可复现。

**📈 对比分析**

与 Diffusion Planner、FlowDrive 等基准对比，在统一五种种子复现协议下，ExecDiffuser 在 nuPlan 6 个设置中平均得分 85.62（比 Diffusion Planner 提升 4.02 分、比 FlowDrive 提升 1.31 分），在 InterPlan 上提升 55.2%，在 NAVSIM 上 PDMS 得分 92.6，超越所有对比方法。

**⚠️ 局限性**

局限性包括：① 需要准确的预测与不确定性估计，预测误差会导致“false‑valid”决策；② 在高风险遮挡或交互复杂场景中，仍有 5% 以上的 false‑valid 率；③ 递归框架与额外的评分网络导致模型参数增 5.2M，推理时间略增至 59.5 ms；④ 仍依赖硬性无效性掩码，可能过度剔除潜在良好轨迹。

---

## 390. HC2Robot: Hyper-Connection Transformers with Predicted Foresight for Multi-Terrain Humanoid Parkour

**arXiv ID:** 2610.36825 | [PDF](https://arxiv.org/pdf/2610.36825v1)

**作者:** Wenhao Xu `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出HC2Robot框架，在人形机器人跨越阶梯、跨栏、平台和斜坡等极端地形时，利用可部署的感知信息实现高效的步态规划与控制。

**💡 创新点**

创新点包括引入可学习的Hyper-Connection Transformer实现多流时间特征融合，以及Predicted Foresight Module（PFM）在非特权观测下预测每只脚的下一次着陆时机、位置和支撑质量，弥合训练与部署的信息差距。

**🔧 技术方法**

主要技术手段为非对称actor‑critic强化学习、深度感知编码、跨注意力结构、分布式多流路由、事件时间预测损失和运动先验正则化。

**📊 数据集**

使用仿真数据集，随机生成四类地形（阶梯、跨栏、高平台、斜坡）并在模拟环境中训练与评估，随后在Unitree G1机器人上进行零射击真实地形测试。

**📈 对比分析**

与基准方法Hiking‑in‑the‑Wild和ParkourFormer对比，HC2Robot在所有地形上平均成功率达到97.60%、安全着陆率97.94%，显著优于基准，并在高平台和斜坡等极端场景中取得最大提升。

**⚠️ 局限性**

局限性包括对深度相机依赖、在极端或未知地形下的鲁棒性未充分验证，以及对不同机器人平台的迁移性仍需进一步探索。

---

## 391. Variable-Length Feedback Coding With Universal Decoding and Bandit Channel Selection

**arXiv ID:** 2610.36818 | [PDF](https://arxiv.org/pdf/2610.36818v1)

**作者:** Recep Can Yavas `[一作]` `[通讯]` (Bilkent University), Recep Can Yavas (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了将可变长度前向 (VLF/VLSF) 编码与多臂赌博机（bandit）信道选择策略相结合的方案，并给出了在 AWGN 信道未知噪声方差下的第二阶渐近性能上界和下界，分析了错误概率、时延与能量的三元组。

**💡 创新点**

创新点在于：① 同时完成信道学习与信息传输，避免了传统探索‑承诺（ETC）中先探索再承诺的两阶段耦合；② 在未知噪声方差的环境下，VLF‑Bandit 能实现更低的第二阶系数，说明在大容量极限下需要的额外时延更少；③ 通过信息理论与赌博机理论的融合，给出了能量约束下的精确能量-时延-误差三元组，并与 VLSF‑Bandit 及 ETC 进行定量对比。

**🔧 技术方法**

主要技术手段包括：可变长度编码与信息密度的极限分析；拉伊‑罗宾斯（Lai–Robbins）和分支限界（Kullback‑Leibler）等赌博机上界；指数族与马尔科夫决策的精度估计；奖励堆叠（reward‑stack）表示、切比雪夫与Chernoff 置信界、Minkowski 不等式以及自适应阈值的二阶展开。

**📊 数据集**

实验数据：论文仅在理论模型上进行分析，使用标准的 AWGN 信道模型（方差未知），并通过数值示例验证理论上界与下界之间的关系，未使用实际数据集。

**📈 对比分析**

对比方法：将 VLF‑Bandit 的第二阶系数与单独探索‑承诺（ETC）方案以及 VLSF‑Bandit 进行比较。结果显示，VLF‑Bandit 的第二阶系数显著小于 ETC，且在能量约束下无基线且比 VLSF‑Bandit 的探索成本更低。实验示例显示在典型噪声方差差异下，VLF‑Bandit 能在相同误差率下实现更短的平均时延。

**⚠️ 局限性**

局限性：① 未给出 VLF‑Bandit 的第二阶逆向界，尚不确定其是否达到信息理论极限；② 仅适用于 AWGN 信道，未推广到一般离散或连续信道；③ 方案实现复杂度高，尤其在多臂赌博机的在线估计和精度估计中需要额外的随机化与计算资源。

---

## 392. HqcLethe: Streaming HQC Encapsulation for Tiny Stateless Servers

**arXiv ID:** 2610.36808 | [PDF](https://arxiv.org/pdf/2610.36808v1)

**作者:** Wind Wong `[一作]` (Vrije Universiteit Amsterdam), Klaus v. Gleissenthall `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一个无状态的小型服务器，可以使用NIST选定的代码基础密钥封装（HQC）实现前向安全的后量子密钥协商。

**💡 创新点**

通过采用中国剩余定理将HQC的多项式乘法拆分为独立残差，并加入掩码随机化，实现了在不保留任何客户端状态的情况下，以分包方式进行密钥封装，并给出了形式化泄露证明。

**🔧 技术方法**

使用了HQC、ML‑KEM‑768、BLAKE3、XChaCha20‑Poly1305、Chinese Remainder Theorem、掩码随机化、客户端谜题、Replay过滤器等技术。

**📊 数据集**

在x86和Raspberry Pi 3B+平台上实验，使用公开的HQC参数集（HQC‑1/3/5）并与McTiny的Classic McEliece进行对比。

**📈 对比分析**

与McTiny相比，HQC版本在类别5下减少了27倍字节、9倍CPU；与有状态缓冲基线相比，消除了内存耗尽，CPU仅增幅2–4倍；在Pi上仍保持低于McTiny的延迟和资源占用。

**⚠️ 局限性**

主要局限在于需要较多CPU计算来重新评估每个分包，导致相对较高的延迟；对攻击者进行CPU消耗仍需使用客户端谜题；多项式乘法的进一步优化尚待研究。

---

## 393. SetGRPO: Calibrating Grader-Induced Policy Optimization

**arXiv ID:** 2610.36807 | [PDF](https://arxiv.org/pdf/2610.36807v1)

**作者:** Jiale Huang `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SetGRPO，一种在rubric‑based RL中通过联合系数不确定集、几何安全恢复和风险门控优化来抑制评分随机性导致的方向反转更新的训练框架。

**💡 创新点**

创新点在于把GRPO内部的组内归一化产生的评分误差建模为联合系数不确定集，利用当前策略的梯度几何来评估和恢复可接受的训练系数，并通过独立风险校准实现全局方向反转风险控制。

**🔧 技术方法**

核心技术包括：联合奖励集校准（JRSC）、几何引导安全恢复（GGSR）、选择性风险门控优化（SRGO）以及基于可扩展的Conformal预测与Clopper–Pearson置信上界的风险估计。

**📊 数据集**

使用RubricHub公开训练集（含Chat、Instruction、Medical、Science、Writing五类任务）进行训练，并在十个跨医疗、写作、指令遵循领域的下游基准上进行评估。

**📈 对比分析**

与GRPO、RubricHub、RuscaRL等基线相比，SetGRPO在Qwen3.5和Llama3.2两大模型族上分别获得最高宏观平均分，表现提升约3–5个百分点，显著优于传统方法。

**⚠️ 局限性**

主要限制包括：需满足样本可交换性假设，风险控制仅限于方向反转，未能解决评分者偏见、数据隐私和公平性等问题；此外计算开销比GRPO高约25%，需进一步优化。

---

## 394. Personality Latents: Conditioning Frozen Language Models on Individuals from Their Own Writing

**arXiv ID:** 2610.36805 | [PDF](https://arxiv.org/pdf/2610.36805v1)

**作者:** Gunhee Cho `[一作]` (Sungkyunkwan University), Yun-Gyung Cheong `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在冻结的大语言模型前加软前缀，利用个体写作学习无标签的个性潜在向量，并在推理时使用编码器或优化得到该潜在向量来生成符合该个体个性的文本。

**💡 创新点**

在不使用标签或先验配置的情况下，利用生成潜在优化直接学习个体潜在向量，使得同一向量可跨不同模型使用，并提供推理时可精细化潜在的机制。

**🔧 技术方法**

生成潜在优化、软前缀注入、稀疏/多查询集编码器、梯度微调、冻结模型、线性探测器，以及在RoBERTa、大模型与MiniLM嵌入空间中的评分器。

**📊 数据集**

Kaggle MBTI（自报论坛风格）、Essays（问卷二进制 Big Five）和 PANDORA（连续 Big Five 分数）等包含 8k+ 个人写作的语料库。

**📈 对比分析**

与 PPlug、DEP 个性嵌入以及 Trait Prompt、TS、InContext 等基线在同一冻结模型、相同人群与评分器下对比；在 MBTI 轴上，PL 的 AUC 可达 0.78，显著优于基线；在 Big Five 相关性上表现相对较弱。

**⚠️ 局限性**

评分器可能捕捉主题/词汇而非真实个性；对 Big Five 的信号弱；仅在单一 3B 模型和少数跨模型实验中验证；推理时需要额外优化并依赖个体文本；潜在向量易被滥用于伪装或画像。

---

## 395. RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents

**arXiv ID:** 2610.36878 | [PDF](https://arxiv.org/pdf/2610.36878v1)

**作者:** Delin Wang `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了 RubricAtlas——一种基于专家报告的类型层次评价体系，用来评估 Web 深度研究代理生成的分析报告，重点检查分析完整性和论证有效性。

**💡 创新点**

创新点在于：① 通过跨对象专家报告抽取并归纳出共享的分析问题、结果和检查点，形成可复用的“类型级”规范；② 将评价结构与分析问题对齐，既能定位缺失分析，又保留各级别分数；③ 对事实、推理和结论分别评估，避免将缺失与错误混为一谈。

**🔧 技术方法**

技术手段包括：LLM 进行报告分解、问题映射与检查点抽象；ReAct 框架下的搜索/抓取管线生成报告；两套 LLM 判定者（DeepSeek‑V4.1‑Flash 与 GPT‑6‑luna）执行证据对齐与完整性/有效性评分；统计分析用 Kendall τ、Cohen κ 等。

**📊 数据集**

数据集为 2026 年 1–9 月间 21 所机构发布的中文金融研究报告，共计 1,300+ 报告，涵盖多行业、多报告类型，构成 20+ 研究类型、140+ 典型问题与 600+ 检查点。

**📈 对比分析**

比较方法：对 10 名 Web 研究代理在 980 个共享任务上进行报告生成并评估，计算每位代理的完整性分、不同层级得分和排名；使用 Kendall τ 衡量代理排名与人工判定者的一致性；通过参考报告选取实验、删除/填充实验验证评价敏感度，结果显示代理排名稳定但对参考集高度敏感，完整性平均分约 50–60，排名一致率约 70%。

**⚠️ 局限性**

局限性包括：① 第三方审计仅由作者完成，缺乏独立验证；② 参考报告支持度仅衡量共现频率，未评估重要性；③ 有效性规则尚未经过充分校验，导致判定一致性较低；④ 评价框架侧重中文金融领域，迁移到其他领域需重新构建报告池和适用性检查；⑤ 仅评估单轮生成，未考虑随机性或搜索时间窗口等外部因素。

---

## 396. RangeRectQuant: Compressing Scientific Fields for Every Future Rectangular Sum

**arXiv ID:** 2610.36873 | [PDF](https://arxiv.org/pdf/2610.36873v1)

**作者:** Jiahao Zhang `[一作]` (Zhengzhou University of Aeronautics), Yu Chen `[通讯]` (Zhengzhou University of Aeronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 RangeRectQuant 的压缩方法，能够在保证任意轴对齐矩形和的误差不超过给定预算的前提下，对科学栅格数据进行无损压缩，并保持像素取值范围与指定精确像素不变。

**💡 创新点**

创新点在于利用前缀和误差区间的单一整数区间特性，构建可线性完成的像素路径；通过前向-后向扫描精确计算每行可行前缀错误区间，确保整个字段满足全局矩形误差预算；同时引入粗细像素分离的量化策略，兼顾压缩率与精度。

**🔧 技术方法**

技术主要包括：前缀和误差控制、整数区间递推求解、粗细像素分离的量化与 LZMA 码流压缩、以及对已知保护像素与像素取值范围的约束实现；同时通过步长 q 的搜索来平衡压缩率与误差。

**📊 数据集**

使用的公共数据集为：2020 年全球人类居住层（Built‑Surface）以及 1984–2024 年水体出现率（Water‑Occurrence）数据，分别在多个城市（Cairo、Delhi、New York、Jakarta）和多达 40 个不同位置上测试，像素尺寸为 100 m 及 1 %。

**📈 对比分析**

与多种现有方法（如 SZ3、MGARD、QPET、JPEG 2000、PNG、sum‑pyramid 控制等）进行对比。实验结果表明，在满足相同矩形误差预算和像素取值范围的前提下，RangeRectQuant 的文件尺寸平均可比最小有效压缩方法缩减 30–70% 的存储空间，且能够在多种预算下保持全局误差约束。

**⚠️ 局限性**

局限性主要在于：需要预先设定误差预算 ε 与步长 q，过大的预算可能导致压缩率提升有限；算法对行高度和宽度的线性扩展在极大尺寸（如 4096²）下仍需显著的 CPU 与内存资源；另外，该方法仅适用于已知取值范围且无缺失值位置未知的栅格数据，对动态更新或非轴对齐查询的支持有限。

---

## 397. Forget the Data, Keep the Capability: Switchable Unlearning for Retrieval

**arXiv ID:** 2610.36858 | [PDF](https://arxiv.org/pdf/2610.36858v1)

**作者:** Jingong Chen `[一作]` (South Dakota State University), Kwanghee Won `[通讯]` (South Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MoME 框架，利用寄生参数和梯度隔离在检索模型中实现可切换的完整模式与核心模式，从而实现选择性忘记。

**💡 创新点**

创新点在于把同一骨干网络拆分为核心与寄生两部分，核心不接收忘记样本梯度，保证忘记后检索性能不降；并引入 Forgetting‑Rank 统计量对忘记效果进行可量化审核。

**🔧 技术方法**

使用 LayerNorm/BatchNorm 的寄生参数（类似 LoRA 低秩分支）、双步梯度隔离训练、归一化统计冻结、损失函数为标签平滑识别损失与 soft‑margin 三元组损失；评估攻击包括 MIA、RMIA、U‑LiRA。

**📊 数据集**

实验数据集为 Market‑1501、VeRi‑776、RegDB（跨模态重识别）以及 CIFAR‑10/100、TinyImageNet 的分类删除实验。

**📈 对比分析**

与原始模型、Fine‑Tune、GA、NegGrad+、SalUn、Bad Teacher、LoRA、SSD、SCRUB、LoTUS、MUNBa、LOFT、Retrain 等 10 种未学习方法和相同基准进行对比；Core 模式对照 retrain 参考模型在 R‑1/mAP 与攻击 AUC 上实现等效甚至超越检索性能，Full 模式恢复原始性能。

**⚠️ 局限性**

局限性包括：对 BatchNorm 统计需手动冻结，机制在非 ViT 结构（如 ResNet、CLIP）上效果不如预期；只能在已知删除集且预训练骨干上使用；当核心参数已学习到删除信息时无法彻底忘记；额外的寄生参数和梯度隔离步骤会增加训练复杂度。

---

## 398. When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate

**arXiv ID:** 2610.36852 | [PDF](https://arxiv.org/pdf/2610.36852v1)

**作者:** Marc Ziegler `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种结构化代理模型，用于在机器人基础模型冻结的情况下诊断并修复传感器错误，从而提升多形态机器人的步态稳定性。

**💡 创新点**

创新点在于将步态周期分解为基础循环与传感器依赖修正，利用机制驱动诊断(MBD)识别异常传感器，并通过相位引用修复( PRR )实现无梯度、无访问内部的外部修复。

**🔧 技术方法**

技术手段包括基于傅里叶级数的周期步态建模、岭回归得到修正项、相位分箱统计、基于贡献分数的诊断、以及相位匹配的传感器值替换或偏置校正。

**📊 数据集**

使用的实验数据集来自仿真 MuJoCo 中的三款四足机器人（Go2、Go1、ANYmal C），包含 37 种人工注入的传感器故障，且每种故障都有多条轨迹用于诊断、修复与评估。

**📈 对比分析**

与相位偏差、相位梯度、GBM‑TreeSHAP 等基线方法比较，MBD+PRR 在 Go2 上将步态成功率从 259/640 提升至 429/640，在线模式可进一步到 382/640；在其他机器人上也显著改善，整体提升约 27% 的成功率。

**⚠️ 局限性**

局限性包括仅在单一冻结策略与仿真环境中验证，缺乏真实硬件实验；故障集合仅为人工构造，且每台机器都使用独立拟合的代理，未测试跨机器人迁移与多重故障场景。

---

## 399. uTHCD-C: A Corruption-Robustness Benchmark for Tamil Handwritten Character Recognition

**arXiv ID:** 2610.36850 | [PDF](https://arxiv.org/pdf/2610.36850v1)

**作者:** Preethi S `[一作]` (Kongu Engineering College), Arun Prakash A `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对泰米尔手写字符识别的多腐败鲁棒性基准uTHCD‑C，并使用该基准对九种卷积及变压器网络进行五折交叉验证与写手独立测试，评估其清晰与受损图像的识别准确率、平均腐败误差（mCE）、参数量、计算复杂度及推理吞吐量。

**💡 创新点**

创新点在于首次为泰米尔手写字符提供标准化的十种腐败类型（包含专门的笔画细化腐败）与五级强度，并以VGG16‑BN为参照的mCE衡量方式；同时结合吞吐量绘制Pareto前沿，揭示鲁棒性与效率的折衷。

**🔧 技术方法**

采用ImageNet预训练的卷积/变压器模型（VGG16‑BN、GoogLeNet、EfficientNet‑B0、RegNetX‑400MF、MnasNet‑0.5、ShuffleNet‑V2‑0.5、SqueezeNet‑1.0、ConvNeXt‑Tiny、Swin‑T），在uTHCD上微调；通过生成高斯噪声、射击噪声、脉冲噪声、模糊、弹性失真、像素化、对比度降低、缩放与笔画细化等腐败，对模型进行鲁棒性评估；使用mCE、参数计数、GMACs、吞吐量及Pareto分析比较。

**📊 数据集**

使用的是无约束泰米尔手写字符数据库uTHCD，包含90,950张手写字符图像，156类，753名作者。

**📈 对比分析**

通过5折分层交叉验证与写手独立70:30划分进行比较，报告清晰准确率（均高于95%）、mCE（ConvNeXt最低0.75，ShuffleNet最高1.55）及吞吐量（最高30,591 img/s，最低1,072 img/s）；Pareto前沿显示ConvNeXt在鲁棒性上最优，而ShuffleNet在吞吐量上最优，体现出鲁棒性与效率的折衷。

**⚠️ 局限性**

局限性包括：仅使用合成腐败对单一uTHCD数据集进行评估，无法直接推断在自然降质文档上的表现；吞吐量测量受限于特定硬件（NVIDIA A100）与实现；结果对其他脚本、数据集或真实降质情况的可迁移性未知。

---

## 400. When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning

**arXiv ID:** 2610.36849 | [PDF](https://arxiv.org/pdf/2610.36849v1)

**作者:** Zehao Wang `[一作]` (University of Science and Technology of China), Lin Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于政策兼容的强化学习框架 PICO，利用教师模型在自举采样中进行分布受限的联合解码，以生成可验证且与当前学生策略兼容的推理轨迹，并将这些轨迹插入到所有失败的采样组中，恢复有信息的学习信号。

**💡 创新点**

创新点包括：①将教师与学生在 token 级别的候选集进行交集，形成“共识支持集”，既能利用专家优势，又能保持与学生的相对概率分布；②使用动态 top‑k 机制自适应控制教师候选集大小，确保在有限 KL 预算内最大化学生概率质量；③在混合行为优化中加入重要性采样比率，精确补偿联合解码策略与学生策略之间的分布不匹配。

**🔧 技术方法**

核心技术：RLVR（可验证奖励强化学习）与 GRPO 的组相对优势估计；分布受限联合解码（token 交集 + λ‑重加权）；动态 top‑k 策略；混合行为政策优化（混合 on‑policy 与 off‑policy 路径）与重要性采样；对齐 KL 限制与自适应采样。

**📊 数据集**

主要数据集包括：数学推理 DAPO‑Math‑17K（AIME 2025/26、HMMT 2026、MATH‑500）；通用推理 GPQA‑Diamond、Humanity's Last Exam；以及交互式智能体任务 ALFWorld 与 ScienceWorld。

**📈 对比分析**

与基准（GRPO、LUFFY、Prefix‑RFT、ReLIFT、RLRT）比较，PICO 在所有数学/通用推理测试集上平均提升 4.1–4.4%，在 ALFWorld 与 ScienceWorld 的平均成功率上提升 6.8%，并在训练过程中保持更高的奖励、更短的未解答比例。实验显示其在不同模型规模（Qwen3‑4B‑Base、Qwen3‑8B‑Base、Qwen3‑4B‑Instruct）均优于现有方法。

**⚠️ 局限性**

局限性：①需要教师模型和大量专家轨迹，训练成本较高；②动态 top‑k 的阈值与最大候选数需手工调参，对不同任务的适用性仍需进一步验证；③在极大模型或完全无教师可用的场景下，联合解码的效果和稳定性尚未得到充分评估。

---

## 401. Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models

**arXiv ID:** 2610.36840 | [PDF](https://arxiv.org/pdf/2610.36840v1)

**作者:** Xu Hu `[一作]` (Hong Kong Polytechnic University), Zhaoxiang Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种辅助未来3D运动预测的世界动作模型（Trace-WAM），用于提升机器人控制性能。

**💡 创新点**

创新点在于将三分支 Transformer（Tri-MoT）结合视频预测、3D运动预测与动作生成，利用3D运动作为辅助监督，在推理阶段无需预测未来视频或运动即可生成动作。

**🔧 技术方法**

采用了三分支 Transformer 架构、B-spline 控制点运动表示、Flow Matching 训练方法、预训练的视频 VAE 与文本编码器以及结构化注意力掩码等技术。

**📊 数据集**

使用的数据集包括 RoboCasa-GR1、LIBERO、LIBERO-Plus 以及在 Piper 机器人上的真实实验数据（每个任务 50 条演示）。

**📈 对比分析**

与 Fast-WAM、Joint-WAM 及 DIAL 等方法对比，Trace-WAM 在 RoboCasa（少样本和完整数据）、LIBERO、LIBERO-Plus（零样本 OOD）以及真实机器人实验中均取得更高的成功率，尤其在少样本和 OOD 场景提升显著。

**⚠️ 局限性**

局限性包括对预训练视频/文本模型的依赖、运动预测仅作为辅助且推理时不使用，运动网格的手工设置可能限制通用性，以及在极端运动或非标记点上的鲁棒性尚待提升。

---

## 402. MVRCap: Towards Comprehensive Video Captioning via Hierarchical Multi-Dimensional Verifiable Rewards

**arXiv ID:** 2610.36838 | [PDF](https://arxiv.org/pdf/2610.36838v1)

**作者:** Weihuang Lin `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MVRCap框架，将视频字幕生成转化为结构化的视觉推理任务，并通过强化学习实现视觉语义对齐。

**💡 创新点**

创新点包括：① 构建多维度（实体属性、动作与交互、时间信息、相机语言、场景环境、风格与氛围）结构化字幕和对应QA对；② 设计Fine‑Grained Verifiable Reward，由层次化概念覆盖、事实性验证与简洁约束三部分构成；③ 采用GRPO强化学习并用Judge LLM实现可验证奖励。

**🔧 技术方法**

技术方法：多维度结构化标注、MM-LM生成训练、交叉模型校验、SFT + RL双阶段训练、GRPO策略优化、Judge LLM评估奖励。

**📊 数据集**

使用的数据集：MVRCap‑SFT（约4K视频-字幕对，结构化标注）、MVRCap‑QA（约5K视频，20.5万QA对，用于奖励验证）、以及在VC4VG、CAPability、VDC等公开基准进行评测。

**📈 对比分析**

与现有方法对比，MVRCap在VC4VG（总分74.6%）、CAPability（All Avg 55.18）和VDC（平均准确率49.7%）上均取得SOTA或接近专有模型的水平，显著优于同类开源模型。

**⚠️ 局限性**

局限性：仍受限于训练数据的多样性与真实视频内容的复杂性；Reward设计虽然可验证但依赖Judge LLM的准确性；对长视频时序推理仍有进一步提升空间。

---

## 403. Drop the Details, Get the Gist: Dynamic Conceptual Representation for Video Recognition

**arXiv ID:** 2610.36832 | [PDF](https://arxiv.org/pdf/2610.36832v1)

**作者:** Ryoya Yoshiyama `[一作]` (University of Osaka), Yuta Nakashima `[通讯]` (University of Osaka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于概念瓶颈的程序化视频识别框架，将视频分段的概念相似度序列送入 Transformer 进行分类，并通过遮蔽重要性得分解释预测。

**💡 创新点**

创新点在于将概念瓶颈模型扩展到时序视频域、利用 Transformer 捕捉概念的时间依赖、证明粗粒度概念已足够，以及通过单概念遮蔽提供可解释的预测理由。

**🔧 技术方法**

技术核心包括：使用 Perception Encoder 进行视频-文本相似度计算；利用 Qwen3 LLM 自动构建概念集；对概念分数进行 Top‑k 过滤；Transformer 编码器进行时序建模；以及基于遮蔽的概念重要性评分。

**📊 数据集**

实验数据集为 YouCook2（89 类烹饪视频）和 ActivityNet 的 Household Activities 子集（44 类日常活动）。

**📈 对比分析**

与直接把视频特征输入 Transformer 的黑盒基线相比，本框架在 YouCook2 上取得 89.5% 对 90.7% 的 Top‑1 精度，ActivityNet 上 93.5% 对 94.3%；Transformer 时序建模比平均池化提升约 3%；粗粒度概念与细粒度概念在 10 以上 Top‑k 下几乎无差异。

**⚠️ 局限性**

局限性包括：高度依赖 VLM 的视频‑文本对齐，可能带来偏差；重要性评分仅为遮蔽效应，未揭示概念交互；在缺乏步骤描述时性能下降；缺少空间/片段级别的概念定位。

---

## 404. Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models

**arXiv ID:** 2610.36811 | [PDF](https://arxiv.org/pdf/2610.36811v1)

**作者:** Runze Jiang `[一作]` (Peking University), Longtao Huang `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出递归知识改进框架RKI，在大型语言模型的知识改进过程中通过诊断弱点、估计运作知识状态、选择相应干预并在每次更新后重新评估模型，从而实现闭环式知识更新。

**💡 创新点**

创新点在于识别知识状态别名问题，利用运作知识状态进行干预选择，并在每轮更新后递归重新估计知识状态以适应模型变化。

**🔧 技术方法**

技术包括基于图结构的知识单元、诊断维度（获取、识别、回忆、理解、推理、应用）、状态估计函数、状态感知干预策略以及递归闭环更新流程。

**📊 数据集**

数据集包括公开的多跳问答基准HotpotEval和企业安全合规基准SafeInternalEval。

**📈 对比分析**

与GraphGen、GENIE、Self-QA等传统知识整合方法相比，RKI在两个基准上的平均分提升18.41分，HotpotEval提升5.90分，SafeInternalEval提升30.92分，表明闭环状态感知干预显著提高性能。

**⚠️ 局限性**

局限性在于知识状态估计仅基于诊断证据和干预历史，无法完全恢复模型真实知识；此外，递归评估与干预导致额外计算和训练成本。

---

## 405. Hyperparameter and Neural Architecture Search via Evolutionary Optimization for Reinforcement Learning: HPNAS-RL

**arXiv ID:** 2610.36882 | [PDF](https://arxiv.org/pdf/2610.36882v1)

**作者:** Nihal Acharya Adde `[一作]` (Volkswagen AG), Hanno Gottschalk `[通讯]` (TU Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出双分支进化框架，联合神经网络架构搜索和强化学习超参数优化，用于自动驾驶任务的策略学习。

**💡 创新点**

将NAS与HPO在多目标进化算法中协同搜索，并结合教师-学生政策迁移与早期退出初始化，显著提高了RL策略的奖励与模型紧凑度。

**🔧 技术方法**

采用进化多目标算法（NSGA‑II）、遗传编码、Simulated Binary Crossover与多项式变异、早期退出初始化（EEPI）、优化迁移学习（OTL）、并行评估与教师学生策略迁移等技术。

**📊 数据集**

在Unity3D仿真环境的三车道环形赛道中使用前向摄像头生成的语义分割图像，训练基于PPO的代理。

**📈 对比分析**

通过与仅NAS基线在不同代数与种群规模下的对比实验，HPNAS在累计奖励、参数量和FLOPs方面均优于NAS；最终全量训练的最佳模型奖励达到1205，超过NAS、HPO和手工调参基线。

**⚠️ 局限性**

受限于低保真评估导致搜索偏差，搜索预算不足时收敛困难；目前未实际应用贝叶斯代理模型，且在更复杂环境中的泛化效果尚未验证。

---

## 406. UniPCH: A Unified Framework for Cross-Chain Payments via Payment Channel Hubs

**arXiv ID:** 2610.36866 | [PDF](https://arxiv.org/pdf/2610.36866v1)

**作者:** Yuanming Shao `[一作]` (Harbin Institute of Technology), Zejun Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 UniPCH，一种两层统一框架，用来在多种区块链之间通过支付通道中枢实现私密跨链支付。

**💡 创新点**

创新点包括：①将通道执行与跨链支付逻辑分离，定义了通用的 UniPC 抽象；②通过选择性重新签名盲条件签名（S‑RBCS）实现跨链通道间的安全协调；③提供了 UTXO 与账户模型的实现，使得任何符合 UniPC 规范的链都能无缝集成。

**🔧 技术方法**

使用的技术包括：盲可随机化签名、线性仅加密、适配签名与重新签名适配签名、零知识证明以及对现有区块链（Bitcoin、Ethereum）的通道实现。

**📊 数据集**

实验数据集：使用六个本地区块链实例（3 个 Bitcoin、3 个 Ethereum），每个链 50 个用户，每个用户发起 5 次支付。

**📈 对比分析**

对比方法：与 A^2L^+、P2C2T、XPCH 等基线方案在四个协议阶段（Reserve、Promis、Convert、Solve）下的运行时与带宽进行对比。结果显示 UniPCH 在大多数配置下实现了更低的运行时（最多比 XPCH 快 20%）和更低的带宽（最多比 XPCH 节省 60%），尤其是使用账户实现时的性能提升最为显著。

**⚠️ 局限性**

限制：①需要在每条通道上实现符合 UniPC 的本地通道实现，初始部署仍需要手工配置；②在 UTXO 现实中，状态更新会刷新所有活跃转移任务，导致在高并发下计算量激增；③全连网格的 hub 互联对存款和通道维护提出更高的资源需求。

---

## 407. Physiology-Informed Audio Quality and Listening Effort Estimation

**arXiv ID:** 2610.36865 | [PDF](https://arxiv.org/pdf/2610.36865v1)

**作者:** Niall McGuire `[一作]` (University of Strathclyde), Dimitra Emmanouilidou `[通讯]` (Microsoft Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本研究在30名听众上同步记录实验室高密度EEG、可穿戴EEG、fNIRS和眼动追踪，在播放分级音频时训练机器学习模型以解码音频质量和听觉努力。

**💡 创新点**

创新之处在于首次系统比较消费级可穿戴设备与实验室EEG在音质与努力解码中的表现，并将两者结合为同一模型。

**🔧 技术方法**

采用EEG功率谱、相关矩阵、fNIRS光密度、眼动特征等多模态特征，并用SVM、XGBoost、MLP、CatBoost等算法进行二分类和回归训练，使用Riemannian空间模型提升EEG表征。

**📊 数据集**

使用NISQA（语音）和ODAQ（音乐）的MOS注释语料库，共144段10秒左右音频作为实验刺激，目标标签为客观MOS或清晰/降质二分类。

**📈 对比分析**

通过留一听众交叉验证和内部子集验证，将各模态的平衡准确率、相关系数和RMSE作为评估指标；可穿戴EEG、fNIRS和眼动在未见听众上均显著高于随机，质量相关系数达0.38-0.43，努力解码仅在个体校准后显著。

**⚠️ 局限性**

局限包括努力解码难以跨听众推广、相关度仅为中等、实验室EEG表现不佳可能受电极接触不稳影响、样本量不平衡、未检验留一音频泛化，且整体仅为可行性验证。

---

## 408. ActiveAlign: Aligning Egocentric Manipulation via Active Perception

**arXiv ID:** 2610.36859 | [PDF](https://arxiv.org/pdf/2610.36859v1)

**作者:** Ri-Zhao Qiu `[一作]` (Nvidia), Erwin Coumans `[通讯]` (Nvidia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ActiveAlign框架，利用主动感知实现实时相机标定和触觉对齐，从而缩小真实到仿真再到真实的差距，并在公开未标定数据集和真实Unitree G1机器人上验证。

**💡 创新点**

①将主动手部摆动作为在线相机标定手段，采用重要性采样与可微渲染实现高精度标定；②通过触觉感知构建高度图，实现毫米级对齐，补偿累积误差；③结合离线3D重建与在线多模态感知的完整Real2Sim2Real流程。

**🔧 技术方法**

可微渲染(Differentiable Rendering)、3DGS、BundleSDF、GPT-6-Astra、FoundationPose、GroundingDINO、SAM2、ArUco标记、触觉传感器、重要性采样、深度匹配优化等技术。

**📊 数据集**

公开未标定机器人操控数据集（Mobile ALOHA、ABC-130K、Humanoid Everyday）以及真实Unitree G1机器人实验。

**📈 对比分析**

与基线DRRobot（无标记相机标定）及手工放置方法比较；在相机标定中IoU最高、相机中心一致性最小；在细粒度操控实验中加入触觉后成功率显著提升，远高于关键点规划基线。

**⚠️ 局限性**

需要机器人执行主动动作，增加任务时间；依赖高质量离线3D重建和触觉硬件，适用范围受限；对快速动态环境的鲁棒性尚未评估；计算量大，实时性仍受限。

---

## 409. DuTri: Dual-Branch Modeling with Triadic Factorization for Multimodal Multi-Scenario Recommendation

**arXiv ID:** 2610.36836 | [PDF](https://arxiv.org/pdf/2610.36836v1)

**作者:** Jintao Huang `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Shenzhen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DuTri框架，针对多模态多场景推荐问题，构建协作分支和多模态分支实现协作偏好学习与自适应模态融合；

**💡 创新点**

创新点在于通过三元CP分解与残差量化学习跨场景结构化协作偏好，并利用该偏好指导多模态融合，同时采用跨分支对齐的对比学习增强信息交互；

**🔧 技术方法**

使用CANDECOMP/PARAFAC三元分解、残差量化、对比学习、条件门控融合以及场景特定MLP等技术；

**📊 数据集**

在AntM^2C（支付宝五场景）和Amazon（三类场景）两个真实工业数据集上进行实验；

**📈 对比分析**

与多模态推荐基线（Diff-MSIN、DMF等）和多场景推荐基线（STAR、HMoE、AMS等）以及Scenario MLP进行对比，DuTri在各场景的AUC/WAUC均优于对照方法，性能提升显著；

**⚠️ 局限性**

局限在于模型参数量相对较大，训练时间比部分轻量级方法略长，对对比学习的温度和对齐权重敏感，且依赖预训练文本/视觉编码器保持不变。

---

## 410. Right Click, Wrong Box: Grading GUI Grounding by What a Click Does

**arXiv ID:** 2610.36831 | [PDF](https://arxiv.org/pdf/2610.36831v1)

**作者:** Xiaokang Yuan `[一作]` (Zhejiang University), Zeng Wang `[通讯]` (Zhejiang Geely Holding Group Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于执行结果的GUI点击正确性定义，并实现了高效探针生成“outcome maps”，在两个实时Web应用和公开截图基准上评估并量化了传统box scoring的偏差。

**💡 创新点**

创新点在于把点击正确性从仅靠注释框转为执行结果一致，并提供无需新注释的执行判断工具，揭示并校正box scoring导致的误差。

**🔧 技术方法**

使用Chromium探针对页面每个像素执行点击并捕获签名，结合Bootstrap paired interval、Kappa统计以及多模型、多seed实验进行评估。

**📊 数据集**

实验数据来自自托管的Gitea代码仓库、Django Oscar电商站，以及公开的ScreenSpot、ScreenSpot-Pro、OSWorld-G等截图基准，共计约两千多点击和四千八百步。

**📈 对比分析**

通过与box scoring比较，outcome scoring在Gitea提升约2点、Oscar提升约2.4点；box-rejected点击中约14-34%为outcome correct；公开截图上该比例约6-14%；3B训练实验中两种奖励差异不足0.25点。

**⚠️ 局限性**

局限性包括仅针对单步点击，等价定义仅在单一界面状态下；探针依赖页面稳定性，可能受动态内容影响；实验规模有限，仅覆盖两应用和少数模型，未验证在更大多步任务中的泛化。

---

## 411. When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement

**arXiv ID:** 2610.36826 | [PDF](https://arxiv.org/pdf/2610.36826v1)

**作者:** Hao Chen `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RuHarness框架，通过动态共进化量规与agent harness实现持续自我改进，利用行为轨迹不断发现并优先解决弱点。

**💡 创新点**

创新点在于将评估量规与harness共同进化，实时诊断并跟踪弱点变化，避免过拟合并提升对未见任务的泛化能力。

**🔧 技术方法**

采用LLM、动态量规抽象、行为轨迹分析、诊断效用评估、六坐标harness编辑、验证门控和前沿选择等技术。

**📊 数据集**

使用EvoAgentBench数据集（包含信息检索、数学推理、软件工程、代码实现、知识工作等五个领域），在Qwen3.5-35B-A3B和GLM-4.7-Flash两大模型上进行实验。

**📈 对比分析**

与ACE、EvoSkill、Self‑Harness、Meta‑Harness、AHE等基线进行对比，评估指标为Pass@1。RuHarness在held‑in和held‑out上均取得最佳性能，尤其在未见任务上的提升最为显著。

**⚠️ 局限性**

局限性包括：实验仅在单一DeepAgent框架下验证，六坐标harness抽象在其他架构的泛化尚未评估；实验时间有限，未探讨更长周期或开放式自适应场景。

---

## 412. Words or Deeds? How LLMs Predict Users' Decisions in Human-Agent Dialogue

**arXiv ID:** 2610.36813 | [PDF](https://arxiv.org/pdf/2610.36813v1)

**作者:** Baiqiao Zhang `[一作]` (Hong Kong University of Science and Technology), Yulong Bian `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过让42名受试者在重复的囚徒困境式游戏中与五种不同人格的 GPT‑4o 代理互动，并记录对话与决策，评估大型语言模型（LLM）如何将用户的陈述与历史决策相结合来预测其下一步行动。

**💡 创新点**

创新点在于：①将预测任务细化到用户是否兑现陈述的“保留”与“违背”两种情形；②通过对不同代理人格和用户与代理人格的相容性进行实验，揭示 LLM 在考虑上下文与行为一致性上的偏差；③设计了插入语句实验来测量模型对新陈述的敏感度。

**🔧 技术方法**

技术手段包括：使用八个开源指令微调 LLM（Qwen、Gemma、Seed‑OSS、GLM、Ministral、Olmo 等）进行零样本预测，利用下一词对数似然(log‑odds)转化为合作概率；对话与决策记录的组合实现信息消融；对预测结果做 AUC、准确率和偏差（bias）分析。

**📊 数据集**

所用数据集为 Personality‑1260，包含 42 名中国大学生完成 BFI‑44 量表后，分别与 Openness、Conscientiousness、Extraversion、Agreeableness、Neuroticism 五种 GPT‑4o 代理玩六轮，每轮可自由对话后做合作/背叛选择，累计 1,260 轮。

**📈 对比分析**

评估方法：在包含/不包含对话的 19 种信息条件下计算 AUC 与准确率；比较预测与真实保留率、破约率及代理人格差异；分析预测在不同兼容性水平下的偏差斜率。结果显示：仅靠决策记录即可获得 AUC≈0.80，加入对话无显著提升；整体准确率约 0.74，但在保留语句时提升到 0.95、破约时骤降至 0.17；对代理人格与兼容性的敏感度显著低于真实行为，偏差可达 ±14。

**⚠️ 局限性**

局限性包括：样本规模小且为低风险游戏，且仅使用中文；代理全部为 GPT‑4o，难以区分人格表达与决策策略；兼容性仅用单一 Big‑Five 维度；零样本 LLM 可能未充分体现上下文细节；插入语句实验仅评估模型敏感度，未揭示因果影响；结果在不同文化、长时关系或真实商业场景中的泛化尚未验证。

---

## 413. Latent Visual Contrastive Reasoning with Multi-Branch Exploration

**arXiv ID:** 2610.36809 | [PDF](https://arxiv.org/pdf/2610.36809v1)

**作者:** Kegeng Tang `[一作]` (University of Tennessee at Chattanooga), Zihao Wang `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LVCR框架，生成多分支潜在视觉假设，并通过bbox引导的对比损失实现分支与答案相关视觉证据的软对齐；

**💡 创新点**

创新点在于将多分支潜在视觉推理与soft bbox‑guided对比对齐相结合，既保持多假设多样性，又通过排名注入保留关键信息；

**🔧 技术方法**

采用潜在视觉推理、对比学习、注意力分支查询、soft cosine相似度、Qwen2.5‑VL‑Instruct backbone以及分支排名注入策略；

**📊 数据集**

训练使用VISUAL COT数据集（带bbox证据），评估基准为V⁎ Bench、MMVP与BLINK；

**📈 对比分析**

与Qwen2.5‑VL、Vision‑R1、PixelReasoner、Vanilla SFT、LVR等基线对比，3B/7B尺度下LVCR在多项指标（V⁎、MMVP、BLINK等）上实现最优或接近最优，显著提升感知密集视觉推理性能；

**⚠️ 局限性**

局限包括：仍未在所有基准上取得最优；依赖bbox监督，需额外标注；计算上增加少量开销；在极端复杂关系或需要外部工具的推理场景中效果有限。

---

## 414. LLM knowledge for improved Out-of-Distribution Tabular Prediction - A Cross-Validated Regularisation Approach

**arXiv ID:** 2610.36806 | [PDF](https://arxiv.org/pdf/2610.36806v1)

**作者:** Sølve Langseth `[一作]` (Oslo Metropolitan University), Hugo L. Hammer `[通讯]` (Oslo Metropolitan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过让大型语言模型（LLM）多次生成系数样本，并将其均值与协方差转化为正则化回归的先验，评估在五个真实域移位（OOD）切分上的预测性能提升。

**💡 创新点**

创新点在于首次在多种真实域切分上系统验证LLM先验对OOB性能的正面影响，并提出仅使用源域交叉验证即可选择最佳先验结构与权重；同时通过置换和符号翻转对照实验证明提升来自特征特定内容而非仅仅是非零中心。

**🔧 技术方法**

技术方法包括：利用Ridge/L2‑Logistic回归，将LLM抽样的均值和协方差作为Gaussian先验；引入λ（先验精度）和γ（混合权重）调节权重；通过5折交叉验证选取λ，bootstrap评估置信区间；对分类任务采用AUC、LogLoss、Brier等阈值无关指标。

**📊 数据集**

使用的数据集为五个公开表格数据集：Boston Housing、Cancer、Heart Disease、Diabetes、Adult；每个数据集以真实协变量（如住房年代、地区、临床队列、年龄、就业部门）划分为源域和OOD域。

**📈 对比分析**

评估方法为将每种先验模型与零均值先验（Ridge）在OOD上进行比较，报告平均提升、95%置信区间和预测区间；结果显示四个数据集平均提升（最高7.9% MSE或12% LogLoss），但单次部署区间宽泛，效果随样本量和域移位程度而异。

**⚠️ 局限性**

局限性包括：源域交叉验证难以准确挑选最佳先验，仅保证避免最差；单一域切分不具备泛化性；LLM可能已在训练集记忆导致先验真实性存疑；Boston Housing移除敏感特征影响结果；大样本数据集如Adult表现无显著提升；未与其他LLM先验方法或多域拆分进行直接对比。

---

## 415. Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences

**arXiv ID:** 2610.36797 | [PDF](https://arxiv.org/pdf/2610.36797v1)

**作者:** Xiang Ding `[一作]` `[通讯]` (Nagoya University), Xiang Ding (Nagoya University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在处理园林路径句子时废弃的初始语法分析如何在后续文本中持续产生影响。

**💡 创新点**

证明废弃分析通过名词短语状态在后续生成中存活，并将残留激活与误差驱动学习两种机制区分开来。

**🔧 技术方法**

采用激活补丁、线性探测、注意力敲除等技术，在24个开源因果Transformer模型上进行实验。

**📊 数据集**

使用了NP/Z、NP/S和主谓/简化相对从句等多种园林路径句子集合以及公开基准数据集。

**📈 对比分析**

通过比较生成偏好和惊讶度，发现残留效应约占真实对象分析的76%，适应效应随歧义消解错误增大而增强，模型表现与人类相近但存在差异。

**⚠️ 局限性**

研究仅覆盖英语NP/Z路径句子，模型参数上限为70B，未测试封闭式模型或非英语数据，补丁实验覆盖的模型范围有限。

---

## 416. Sample-and-Score: Surrogate-Guided Coordinated Spatial Reuse in IEEE 802.11 Networks

**arXiv ID:** 2610.36868 | [PDF](https://arxiv.org/pdf/2610.36868v1)

**作者:** Maksymilian Wojnar `[一作]` `[通讯]` (AGH University of Krakow), Maksymilian Wojnar (AGH University of Krakow)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于采样-评分的协调空间重用（Co‑SR）优化方法，通过在离线训练的图神经网络预测器中对随机生成的Co‑SR配置进行评分，并挑选出最优配置应用于IEEE 802.11网络。

**💡 创新点**

创新点在于将离线训练的图神经网络作为快速代理，直接在一次前向传播中评估数千个候选配置，从而消除在线收敛和多步交互的需求；同时使用图结构仅依赖AP‑AP和AP‑STA RSSI信号，显著降低了测量开销。

**🔧 技术方法**

技术上采用了Transformer‑增强的图神经网络（3层 GNN+3层 Transformer）与混合密度网络（MDN）来预测每个配置的期望数据率，并通过随机采样与top‑k选择实现候选库生成。

**📊 数据集**

使用的训练数据来源于自研的 mapc‑sim 仿真器，生成了约5万个样本，覆盖11种场景类型（2–10 AP，最多50 STA），每个样本包含随机或理想 MCS 的 Co‑SR 配置。

**📈 对比分析**

与 DCF、SR‑groups、H‑MAB 等基线以及理想的完美代理和最优上界进行对比，实验显示在 2–16 AP 的开放空间中，采样-评分方法在 8192 候选下可达到 T‑optimal 吞吐量的 95%，在随机 MCS 环境下仍超过 DCF 约 16%，且推理速度比 H‑MAB 快 30 倍。

**⚠️ 局限性**

局限性包括仅在静态全缓冲下行负载、仅考虑 2–10 AP 训练范围、未结合完整堆栈仿真或硬件验证、仅使用 AP‑AP/STA RSSI 特征、以及对动态网络变化和更复杂 MAPC 方案的适应性不足。

---

## 417. Back to the Future: The Evolving Universal Relation Model

**arXiv ID:** 2610.36833 | [PDF](https://arxiv.org/pdf/2610.36833v1)

**作者:** Nan Tang `[一作]` (Hong Kong University of Science and Technology), Haixun Wang `[通讯]` (Evenup)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于查询驱动的可演化数据库模型（Evolving Universal Relation Model，EUR），通过在每一次自然语言查询时动态扩展通用关系、逻辑模式和实例，从而实现从文档集合到可持久化、可组合、可约束的分析数据库的“零起点”构建；

**💡 创新点**

创新点在于：①将传统的 schema‑first 思想转变为 query‑first；②引入可演化的通用关系作为用户面向的唯一查询接口；③将文档提取与数据库演化统一为查询处理的三大步骤（Reveal、Organize、Materialize）；④将文档提取视为数据库的物理算子，支持推断、验证、覆盖度管理与事务一致性；

**🔧 技术方法**

核心技术包括：通用关系模型、实体‑关系（ER）与关系模型的分层实现；自然语言到 SQL 的编译与 LLM‑辅助检索；增量规范化与逻辑模式演化；提取算子、覆盖度与证据管理；支持稀疏列存储、可插拔索引与成本感知优化；事务与并发控制针对演化操作的扩展；

**📊 数据集**

文中未给出具体真实数据集，使用了一个合成的课程分析示例（包含合同、发票、邮件等多样化文档的典型语料）；

**📈 对比分析**

论文为概念性工作，没有实验对比或性能评估，作者仅阐述了理论框架与系统设计原则；

**⚠️ 局限性**

局限性包括：缺乏大规模实验验证；对提取成本、证据不确定性、索引与优化策略的定量评估尚未展开；事务与并发控制在演化场景下的细节仍需深入研究；

---

## 418. API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails

**arXiv ID:** 2610.36801 | [PDF](https://arxiv.org/pdf/2610.36801v1)

**作者:** Haocheng Mei `[一作]` (Tokenrhythm Technologies), Yunhe Wang `[通讯]` (Tokenrhythm Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于温度对齐的典型性检测方法，用于审计LLM API是否存在模型替换。

**💡 创新点**

创新点在于将审计转化为对合成零假设的检验，利用批量温度对齐消除合法部署差异，计数异常分数，并在完整词表上精确计算，从而显著提升检测灵敏度。

**🔧 技术方法**

主要技术包括规范化超额惊奇分数、温度对齐、阈值校准、完整词表概率计算、GPU加速的批量前向推理等。

**📊 数据集**

使用的数据集为真实商业API响应：26个正品端点（Qwen3.8-27B、Gemma-4-31B-it）和6个替代模型（共143,955条），以及用于校准的离线生成提示。

**📈 对比分析**

与MET、RUT、NLL等基线相比，本方法在真阳率≤0.10的前提下，10%替换率下在200请求时检测功效≥0.69，部分设置可达1.0，且所需API费用仅为几分币。

**⚠️ 局限性**

局限性包括对极低替换率或同族非常相似的前代版本检测效果有限；需要本地模型权重；对极长或定制系统提示的适应性尚未充分验证。

---

## 419. GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents

**arXiv ID:** 2610.36877 | [PDF](https://arxiv.org/pdf/2610.36877v1)

**作者:** Bingchao Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在移动 GUI 交互中，用一个冻结的 JEPA 编码器与一个候选条件的残差预测器联合，构建了一个不需要解码器的动作评分策略，只需一次批处理即可完成动作选择。

**💡 创新点**

创新点在于将 JEPA 用作整个策略，而非仅做特征提取或世界模型；通过残差预测器把屏幕潜在空间映射到每个候选动作，利用结构化头与文本匹配器实现高效评分，且不依赖 VLM 解码；并证明预测器可随机初始化，无需动态预训练。

**🔧 技术方法**

核心技术包括：ViT-L/16 的 JEPA 预训练、残差动作与 ROI 条件的潜在预测器、余弦结构化评分头、MiniLM 文本匹配器、两阶段文本重排序器（span ranker 与 click reranker），以及端到端的离线策略训练。

**📊 数据集**

使用未标记的移动 GUI 截图（GUI-Odyssey，189K 张）进行 JEPA 预训练；使用 AndroidControl 基准的 13,604 训练集进行策略学习与验证；在官方四个 OOD 子集（IDD、app_unseen、task_unseen、category_unseen）上进行评估。

**📈 对比分析**

与同样协议、使用 LoRA 微调的 Qwen2.5-VL-7B VLM 代理相比，该 JEPA 代理在相同输入和 GPU（NVIDIA H20）上实现 2.8–4.0 倍的速度提升，峰值 GPU 内存使用 4.6–6.4 倍减少，且仅占 6% 的参数量，能够达到 92.4% / 86.1% 的 IDD Low / High 准确率（相当于 7B 代理的 86–92%）。

**⚠️ 局限性**

主要限制包括：需要可访问的 a11y 树作为候选生成来源（不是所有 GUI 可用）；仅完成一步级动作评分，缺乏任务规划与终止判断；对 1.6–2.6% 无候选的点击错误；未在多种候选来源（如 Web DOM）上验证；并且不支持实时前向预测或全任务级别控制。

---

## 420. HoloRecon: Incremental Reconstruction of Long-Range Panoramas with Global Consistency

**arXiv ID:** 2610.36876 | [PDF](https://arxiv.org/pdf/2610.36876v1)

**作者:** Zhuoxiao Li `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将稀疏ORB映射与前馈几何预测相结合的长距离全景重建框架，利用预测几何进行三维匹配验证并构建跨视图的相机约束，最终实现全局一致的相机轨迹与稠密地图；

**💡 创新点**

创新点在于通过稀疏地图对前馈预测进行标定，并使用这些标定后的几何信息实现远距离视图间的三维匹配验证，形成无需共享稀疏地标的长距离相机约束；

**🔧 技术方法**

采用ORB特征跟踪、前馈几何网络Pi3（在Holo360D上微调）、相机图优化、GPU加速的全景束调整以及深度尺度校准等技术；

**📊 数据集**

使用了自制的PanoRecon基准（7个含重复建筑纹理和长距离重访的场景），并在公开的360LOC、YUTO MMS数据集上进行评估；

**📈 对比分析**

与PanoVGGT、Scal3R、360DVO、Open-vSLAM、Stella-vSLAM-Dense、COLMAP等基线比较，平均ATE显著下降（在YUTO MMS上最小化到0.052 m），并在稠密重建的F-score上达到约63%（相比COLMAP 59.75%），说明在长距离重访场景下表现更优；

**⚠️ 局限性**

局限性包括对严重运动模糊或长时间低纹理区的鲁棒性不足，未处理动态物体，也未实现实时推理。

---

## 421. Tail Cues, Principal Corrections: Plug-and-Play Rectification for Open-Set Test-Time Adaptation

**arXiv ID:** 2610.36870 | [PDF](https://arxiv.org/pdf/2610.36870v1)

**作者:** Yingkai Yang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种轻量级的测试时自适应插件 TALOR，通过利用冻结的线性头部作为几何锚点，对特征空间进行主-尾分解并校正主子空间漂移，从而在不更新参数的情况下提升开集测试时自适应性能。

**💡 创新点**

创新点在于：①利用线性头的奇异值谱定义主-尾坐标系，将高增益主坐标作为校正目标，低增益尾坐标作为漂移感知器；②构造软路由的双阈值归一化尾能量，实现无监督的 ID/OOD 分支；③使用加权岭回归估计尾-主线性映射，并通过 EMA 平滑更新，实现在线无梯度的漂移估计与校正。

**🔧 技术方法**

主要技术包括奇异值分解、特征投影、加权岭回归、指数移动平均、软路由的高斯混合模型以及能量基 OOD 检测。

**📊 数据集**

实验使用 ImageNet-C、LAION-C 等非平稳数据集，并在 15 种不同的损坏和 6 种 OOD 数据集上评估。

**📈 对比分析**

与多种基线（CoTTA、SAR、EATA、ViDA、UniEnt、COME、STAMP、OSTTA、DPCore）对比，TALOR 在 ImageNet-C 上平均 H-score 提升 2.2%（超越 UniEnt），在 LAION-C 上提升 9.6%；作为后置插件还能将 COME 提升至 68.4% H-score；运行速度快 2.5×、GPU 内存占用仅 12%。

**⚠️ 局限性**

局限性包括：①依赖线性头部的奇异值结构，无法直接应用于非线性分类器；②需要设定能量阈值 τ 和岭系数 λ，尽管鲁棒性好但仍需经验调参；③在极端 OOD 或混合比例极低时软路由可能失效，导致校正不充分。

---

## 422. Stiffness coefficients in granular micromechanics determined with three-point bending experiments and digital image correlation for a bimodulus material

**arXiv ID:** 2610.36862 | [PDF](https://arxiv.org/pdf/2610.36862v1)

**作者:** Mohammad Mahdi Nayeban `[一作]` (University of Catania), Luca Placidi `[通讯]` (Telematic University Pegaso)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在双模石墨试件上进行三点弯曲实验并利用数字图像相关（DIC）获取全场正应变场，结合粒子微力学连续体模型，对双模材料的弹性参数进行逆向辨识。

**💡 创新点**

将双模异向性直接嵌入粒子相互作用的正交与切向刚度；使用全场应变而非单一中性轴位移作为辨识依据；采用混合模式搜索加BFGS二次牛顿优化，提升辨识精度。

**🔧 技术方法**

粒子微力学模型、三点弯曲实验、数字图像相关（DIC）、MATLAB–COMSOL耦合有限元求解、模式搜索（Pattern Search）与BFGS二次牛顿优化。

**📊 数据集**

实验数据来自自制双注射石墨EG‑92E试件的三点弯曲实验，DIC 提取的 G₁₁ 全场应变作为输入。

**📈 对比分析**

通过归一化均方根误差（NRMSE）比较数值与实验 G₁₁ 场；优化后误差由 18.6% 降至 12.3%，显著提升全场匹配，残差主要集中在加载区。

**⚠️ 局限性**

仅使用 G₁₁ 忽略 G₂₂、G₁₂，导致切向刚度辨识不充分；模型为弹性第一梯度，无法捕捉局部加载传递与梯度效应；需进一步扩展至应变梯度或非局部模型。

---

## 423. Evolving Agent Teams

**arXiv ID:** 2610.36848 | [PDF](https://arxiv.org/pdf/2610.36848v1)

**作者:** Shiyi Cao `[一作]` (University of California, Berkeley), Ion Stoica `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多智能体框架 Evolving Agent Teams，利用中央监督者在搜索过程中动态诊断并重构团队角色、协作模式和知识管理，以突破 GPU kernel 优化中的局部最优。

**💡 创新点**

创新点在于：1）团队进化机制——监督者基于历史表现主动设计新角色、协作流程和文件系统管理；2）持久化文件系统 K，累积经验与进化历史；3）在不同优化阶段（从正确性到结构重构）动态切换专家角色，解决单一或静态团队难以覆盖的复杂搜索路径。

**🔧 技术方法**

主要技术：Claude Code（Claude Opus 4.6）LLM 代码生成与执行；自定义监督者 Agent 负责诊断与重构；多角色 Agent（Profiling Scientist、Challenger、Primitive Study Agent、Knowledge Integrator 等）；持久化共享文件系统 K；基于 token 预算的成本衡量与调度。

**📊 数据集**

数据集与任务：FlashInfer-Bench（MLA paged decode、GDN prefill）、FA4（Blackwell B200 注意力前向）、G+AR（Hopper H100 GEMM+AllReduce）、KDA prefill（AMD MI350X、NVIDIA B200）、MoE offload（Apple M4 Pro）、RoboDojo 机器人操作任务。

**📈 对比分析**

对比方法：与单一强大编码 Agent（Claude Code、Codex）以及固定四角色团队、K-Search、CUDAForge 等基线进行对比。性能结果：在 MLA 上平均 1.21× 提升、在 KDA 上 3.01×/3.44×/3.54× 的加速、G+AR 超过 1.56×、MoE 达到 30.3 tokens/s、RoboDojo 获得 62.1 分（高于 55.3 最高排行榜）。

**⚠️ 局限性**

局限性：每个实验仅跑两次（部分只跑一次），缺乏统计显著性；部分运行无网络访问，可能限制知识获取；成本与性能的 Pareto 分析尚未完成，未来需更系统的噪声域与成本效益评估。

---

## 424. Conservative Policy Iteration For Zero-Sum POSGs

**arXiv ID:** 2610.36847 | [PDF](https://arxiv.org/pdf/2610.36847v1)

**作者:** Erwan C. Escudie `[一作]` (University of Groningen), Jilles Steeve Dibangoye `[通讯]` (University of Groningen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种适用于有限时限零和部分可观测随机博弈的近似保守策略迭代（CPI）算法，结合点基值迭代与策略搜索，沿着由保守策略更新诱导的占用轨迹进行本地策略改进；

**💡 创新点**

创新点在于：①利用每个玩家固定策略的后续价值函数代替全局中心价值逼近，②仅在当前联合策略可被对手挑衅的占用空间内执行更新，③引入GLIE探索保证全局覆盖，④通过局部线性规划实现贝尔曼备份；

**🔧 技术方法**

技术包括占用状态归约、焦点信念游戏、固定策略价值函数、局部线性规划（LP）备份、随机逼近与步长更新、保守平均策略、GLIE探索；

**📊 数据集**

使用标准有限时限零和POSG基准：Adversarial Tiger、Recycling、Mabc、Matching Pennies；

**📈 对比分析**

与PBVI、HSVI和CFR+比较，CPI在短期时限下表现相当，在较长时限下通常能在运行时间与可利用性之间取得更优的折衷；在某些实例（如Adversarial Tiger、Recycling）PBVI仍更强；CPI的迭代更昂贵但更平滑；

**⚠️ 局限性**

局限包括：①需要在每一步求解双玩家的局部优化，计算量大；②收敛慢、样本复杂度高，限制了在更长时限下的可扩展性；③当前实现依赖线性规划子程序，在大规模游戏中可能成为瓶颈。

---

## 425. Advancing global crop simulations with differentiable modelling

**arXiv ID:** 2610.36827 | [PDF](https://arxiv.org/pdf/2610.36827v1)

**作者:** Yunan Lin `[一作]` (Technical University of Munich), Niklas Boers `[通讯]` (Technical University of Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了可微分的全球格点作物模型Agrocosm，并利用自动微分对作物与土壤过程参数进行梯度优化，以改进历史与未来作物产量模拟。

**💡 创新点**

创新点在于将完全过程基础的作物模型实现可微分化，利用梯度优化对日常碳、水通量观测进行精确拟合，并通过梯度推断极端天气导致的产量损失及制定更有效的灌溉策略。

**🔧 技术方法**

使用Julia实现的Agrocosm、Enzyme.jl进行反向自动微分、Adam优化器、GPU/CPU并行数值架构，以及梯度用于产量对温度、降水等气候变量的敏感性分析。

**📊 数据集**

使用FLUXNET观测数据（GPP、RECO、ET）、ISIMIP3a/3b气候与管理输入、GDHY历史作物产量数据、欧洲欧盟子国家作物统计以及多模型CMIP6气候实验。

**📈 对比分析**

通过五折交叉验证评估站点级通量拟合性能，并与AgMIP GGCMI Phase 3模型比较，优化后Agrocosm在时间相关性和nRMSE上从第三名跃升第一；历史低产年误差降低40%+，未来投影显示参数校准显著降低稻谷在SSP5-8.5下的产量增幅（55%降至22%）。

**⚠️ 局限性**

局限性包括通量观测空间分布稀疏导致参数校准缺乏地区差异性，部分参数逼近边界可能产生过程补偿，未考虑病虫害、极端管理等因素，以及缺乏季节性观测约束限制对极端事件的诊断精度。

---

## 426. Who Knows What - and What Does It Support? Perspective-Bounded Evidence Calibration for Multimodal Theory of Mind

**arXiv ID:** 2610.36824 | [PDF](https://arxiv.org/pdf/2610.36824v1)

**作者:** Ziqi Liao `[一作]` (Jilin University), Hongxia Xie `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PIP-ToM框架，用以在多模态Theory of Mind推理中引入证据可接受性与视角约束。

**💡 创新点**

创新点在于将证据可接受性原则拆解为证据可访问性、源保留和命题相关性，并通过视角存储与声明感知路由实现可解释的概率校准。

**🔧 技术方法**

采用了基于贝叶斯逆向规划的模型、LLM（如GPT‑4o、GPT‑5.4）以及图像+文本融合事件轨迹的前置处理。

**📊 数据集**

使用MuMA‑ToM、MMToM‑QA和Hi‑ToM三个ToM基准数据集进行评估。

**📈 对比分析**

与多种基线（通用MLLM、ToM提示方法、状态追踪、模型推理等）对比，PIP‑ToM在闭源模型下平均达95%准确率，显著超过AutoToM及其他方法；在开放源模型同样获得最高分。

**⚠️ 局限性**

局限性包括对预先构建的证据路由表和结构化声明的依赖，且对高度递归或隐式心理状态的推理仍需依赖模型推断，直接匹配无法覆盖。

---

## 427. TAPESTRY: A Single Backbone for Video Generation and Robot Control

**arXiv ID:** 2610.36823 | [PDF](https://arxiv.org/pdf/2610.36823v1)

**作者:** Jeremy A. Collins `[一作]` (Georgia Institute of Technology), Paarth Shah `[通讯]` (Toyota Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用预训练视频扩散变换器，将其重构为机器人控制策略，无需在推理阶段生成视频；

**💡 创新点**

创新点包括：将机器人动作视为VAE patch令牌，复用视频模型的tokenizer与unembedding；使用三目标联合训练（视频、视频+动作、动作单独）共享权重；单步DDIM推理在零终端SNR下可直接得到动作序列；

**🔧 技术方法**

技术包括：视频扩散Transformer（DiT）、3D VAE编码器、patchify卷积、v预测式DDPM、DDIM、Affine-free LayerNorm、异步注意力掩码、零终端SNR调度；

**📊 数据集**

数据集：大规模跨体视频（DROID、BridgeData、AgiBot-World）、互联网视频（Ego4D、Epic Kitchens、SSv2、Panda-70M）以及针对三任务的双手遥控演示；在LIBERO benchmark上使用LIBERO-90动作标签并仅用视频进行其他四套任务的学习；

**📈 对比分析**

与现有基线（行为克隆、Cosmos Policy、DreamZero、Amplify等）对比，TAPESTRY在LIBERO held-out 套件上平均成功率提升约15%；在真实双手操作任务中达成93%+成功率，且在三类OOD（对象摆放、运动原语、对象组合）中仅TAPESTRY保持高成功率；推理速度约44倍快于DreamZero（单步84 ms vs 20 s），显著降低GPU内存与延迟；

**⚠️ 局限性**

局限性：单步推理仅给出动作块的条件均值，缺乏多模态或高度随机动作的细粒度规划；依赖零终端SNR v预测调度，其他扩散架构（如rectified-flow）不一定兼容；动作空间固定为20D双手，未探索不同机器人或多体的通用性。

---

## 428. Taming Multi-Dimensional Skew in Sparse Matrix Multiplication with Contention-Aware Scheduling [Technical Report]

**arXiv ID:** 2610.36822 | [PDF](https://arxiv.org/pdf/2610.36822v1)

**作者:** Wentao Huang `[一作]` (National University of Singapore), Kian-Lee Tan `[通讯]` (National University of Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“块”式动态调度框架（Morsel-Wise Dynamic Scheduling），通过固定结果大小的切片实现多维度负载均衡，并结合争用感知调度、可选SMT激活以及自适应累加器选择，显著提升稀疏矩阵乘法（SpGEMM）在高 skew 环境下的性能。

**💡 创新点**

创新点包括：①将多维度 skew（计算量、输出密度、数据局部性）解耦，采用固定输出 nnz 的 morsel 进行调度；②提出两种轻量级争用缓解技术（争用感知调度与可选 SMT 激活），减轻共享 L3 缓存竞争；③引入 per-morsel 累加器选择（HASH/ESC）以匹配不同 cf 的内部冲突特征；④在多种真实工作负载（关系查询、图分析、SpMM/SpMV）与 CXL 扩展内存平台上验证方案。

**🔧 技术方法**

技术手段：动态任务调度、固定结果大小 morsel 切片、cf（压缩因子）统计、争用感知调度（CMA）、可选 SMT（SMT‑Enable/Disable）、自适应累加器选择（ACC）、利用 CSR 的符号阶段信息、对 L2/L3 缓存的大小与利用率分析、基于 OpenMP 的并行实现。

**📊 数据集**

使用的数据集包括：R‑MAT（ER、SSCA、G500）合成矩阵；SuiteSparse 119 矩阵；关系查询数据（ClickHouse OLAP、MovieLens、IMDB 等）；图分析数据（AMiner DBLP、Amazon、GSM 等）以及相应的 meta‑path、BFS、三角计数实验。

**📈 对比分析**

与最先进的 SpGEMM 基线（HASH、ESC、HEAP、PB、Intel MKL）、以及同类关系查询和图分析实现进行对比。实验表明：在高度 skew 的工作负载上，框架可获得高达 81%（HASH）/ 36%（ESC）甚至 8×（MKL）以上的加速；在非 skew 情况下性能与基线持平；在 CXL 扩展内存系统上能充分利用额外带宽，达到 DRAM 单核峰值以上。

**⚠️ 局限性**

局限性：对极稀疏矩阵（极低非零密度）时因 cf 排序与 morsel 分配开销可能导致性能略逊；依赖符号阶段的 nnz 估计，若估计误差大可能影响调度；当前实现仅针对 Gustavson 形式，尚未扩展到 outer/inner‑product 形式；对非稀疏或全密集矩阵的收益有限。

---

## 429. An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning

**arXiv ID:** 2610.36820 | [PDF](https://arxiv.org/pdf/2610.36820v1)

**作者:** Ngoc Luyen Le `[一作]` (Gamaizer), Bertrand Laforge `[通讯]` (Sorbonne Université)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了增量式知识图学习者数字孪生，用于基于证据的早期预警模型，能够在各时间点同步学习轨迹并生成可追溯的风险警报。

**💡 创新点**

核心创新在于将知识图作为实时特征抽取与解释链的共同基础，确保特征、特征归因、风险警报与原始记录通过可查询的证据链连接。

**🔧 技术方法**

使用RDF/OWL轻量级本体、SHACL验证、SPARQL查询、PROV-O标准以及XGBoost、LightGBM、CatBoost、TabTransformer等模型，构建数字孪生并进行预测与归因。

**📊 数据集**

基于公开的Open University Learning Analytics Dataset（OULAD），在不同模块、不同学期的展示上进行实验。

**📈 对比分析**

与传统原始表格特征进行配对比较，F1和AUC保持相同，且在三个检查点（第4、第8、第12周）均达到与原始特征相当的预测性能，同时实现了证据覆盖率达0.65–0.76、呈现可信度0.74–0.90。

**⚠️ 局限性**

局限性包括仅在历史数据回放情境下验证，未处理实时流、乱序事件、数据纠正或教师反馈循环，且证据覆盖率在缺失/空闲特征上略低。

---

## 430. Integrated Design and Reproducible Feasibility Analysis of a Four-Wheel Rover with Packet-Commanded Imaging, LiDAR Ranging, and Learning-Assisted Target Recognition

**arXiv ID:** 2610.36816 | [PDF](https://arxiv.org/pdf/2610.36816v1)

**作者:** Sabik Bin Sultan `[一作]` (BAF Shaheen College Kurmitola), Maria Sultan `[通讯]` (South Point School & College)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并分析了一个四轮教育型遥控月球探测车的机械、电气与任务系统，提供可追溯的设计评估、仿真与软件验证；并在合成数据集上评估目标识别算法。

**💡 创新点**

集成商业差速驱动、独立悬挂、分离动力电源与 Raspberry Pi 控制，提出可追溯的设计决策与可重现的验证任务；将目标识别任务扩展为合成数据上的学习辅助，并与基线方法比较。

**🔧 技术方法**

采用机械 CAD 与尺寸分析、滚动速度与爬坡力学计算、悬挂动力学模型；电气电流/功率预算、RC 接收与指令解析；Python 参考执行器与命令队列；合成数据生成器、HSV/几何、HSV+SVM、HOG+HSV+SVM 的线性 SVM。

**📊 数据集**

仅使用合成图像数据集：2400 张 64×64 RGB crop，按训练/验证/测试/shifted 分组；未使用真实摄像头图像。

**📈 对比分析**

通过宏 F1 和准确率比较三种分类器：基线 HSV+几何 95% 准确率 / 0.9512 F1；HSV+SVM 82% / 0.7892；HOG+HSV+SVM 82% / 0.7890。在更随机的 shift 设定下，基线 71.75% / 0.7343，SVM 61.75% / 0.5838，HOG 48.5% / 0.4784，说明基线最稳健，学习方法在背景扰动大时性能显著下降。

**⚠️ 局限性**

缺少真实硬件测试：未测量机体尺寸、质量、力学、悬挂、轮胎与电机性能；摄像头-激光标定未完成；合成数据未验证到现实；目标识别未结合定位和任务控制；需进一步测量与实验验证。

---

## 431. Gradient flow of magnetic inclination and intensity as a GPS-free guidance mechanism

**arXiv ID:** 2610.36812 | [PDF](https://arxiv.org/pdf/2610.36812v1)

**作者:** Zayed Herma `[一作]` (Univ. Grenoble Alpes), Andréa Vassilev `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过将生物磁导航 (BIMN) 建模为梯度流控制问题，利用地球磁倾角与强度在预定义区域内实现 GPS‑free 导航。

**💡 创新点**

创新点在于将磁场梯度视为可控的自然航向信号，并通过梯度流来引导系统到达目标点，从而实现了在磁场相似性导致定位模糊的地区内的可靠导航。

**🔧 技术方法**

使用了梯度流控制理论、地磁场模型（如世界磁场模型 WMM/IGRF）以及数值仿真方法。

**📊 数据集**

采用公开的地磁场数据集（WMM/IGRF）和合成的航迹数据进行仿真验证。

**📈 对比分析**

与传统基于磁力计的方向估计方法比较，所提出的梯度流方法在所有测试点成功到达目标的比例达 100%，平均路径误差约为 1–2 公里，性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：只能在磁场相对稳定且可解析的预定义区域内使用；对磁场模型误差敏感；缺乏真实硬件实验验证；在磁异常区可能出现路径漂移风险。

---

## 432. DFIP: Decision-Focused Inference Path for System One Models

**arXiv ID:** 2610.36804 | [PDF](https://arxiv.org/pdf/2610.36804v1)

**作者:** Ismail Pelaseyed `[一作]` (Superagent Technologies Inc), Alan Zabihi `[通讯]` (Superagent Technologies Inc)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并实现了四个决策模型（PPLX-Decider、Liquid d1-3B、Clef-Flash、GLiNER2.5-Decide）的加速路径，并评估其性能与答案保真性。

**💡 创新点**

提出了基于 CUDA 图重放、候选投影和长度感知批处理的组合加速方法（DFIP），针对系统一模型的特定工作负载实现定制加速。

**🔧 技术方法**

使用 CUDA 图重放、Triton/cuBLAS 投影、schema 兼容的长度感知批处理、bounded capture admission 等技术。

**📊 数据集**

采用银行类语料库（BANKING77）以及自定义的短 Boolean、分类、序数问题请求作为实验数据集。

**📈 对比分析**

通过与强化本地实现对比，使用容器轮次为统计单位，报告几何平均速度比与 95% 置信区间；在短 Boolean 任务中 Decider 提升 2.47×，Clef 1.45×，Liquid 1.01×，GLiNER 1.40×，但在多样化到达时加速失效。

**⚠️ 局限性**

加速效果仅限于特定模型与工作负载；外部流量下可能失效，存在 CUDA 错误、数据集有限、未验证生产可用性与全局适用性。

---

## 433. Polymorphic First-Class Environments

**arXiv ID:** 2610.36798 | [PDF](https://arxiv.org/pdf/2610.36798v1)

**作者:** Jinhao Tan `[一作]` (University of Hong Kong), Bruno C. d. S. Oliveira `[通讯]` (University of Hong Kong)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于环境的多态类型计算，消除了 System F 中的所有形式的替换，将环境视为第一类值，并通过 Manifest 类型和两侧类型等价实现类型实例化。

**💡 创新点**

创新点在于：①用 Manifest 类型配合类型等价替代元层级替换，实现完全无替换的环境语义；②引入盒子类型记录局部类型假设，支持透明类型组件与动态模块组合；③提供完整的 Coq 形式化证明，包含类型安全、归一化、等价可判定与对 System F 的保守性。

**🔧 技术方法**

技术方法包括：de Bruijn 索引表示、环境封闭闭包、盒子化局部执行、Manifest 类型与类型等价关系、实例化 lemmas、语义与代数的双向证明、Coq 机械化。

**📊 数据集**

未使用数据集；论文主要是理论和形式化证明工作。

**📈 对比分析**

对比方法：将该系统与传统的 System F 及单态环境计算进行语义与类型层面的比较。性能方面没有运行时评测，重点在证明正确性与可判定性，证明在 Coq 中已通过。

**⚠️ 局限性**

限制与未实现的功能包括：缺乏子类型、封闭（sealing）机制、递归模块以及更高级的模块抽象；此外对类型推断未作讨论，系统为显式类型。

---

## 434. LingGS: Decoupling Feed-Forward Geometry Prediction from Camera Tracking for High-Fidelity Gaussian SLAM

**arXiv ID:** 2610.36853 | [PDF](https://arxiv.org/pdf/2610.36853v1)

**作者:** Da Li `[一作]` (King Abdullah University of Science and Technology), Dominik Engel `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个基于单目RGB流的在线高精度Gaussian SLAM系统，利用冻结的流式3D基础模型生成的深度预测作为虚拟RGB‑D传感器，并结合显式几何解算器在每帧上同时优化相机姿态和深度残差，随后通过Laplacian引导的Gaussian插值与关键帧重放细化，逐帧构建可渲染的Gaussian场景。

**💡 创新点**

创新点主要包括：
1) 通过直接深度残差与ORB特征匹配的几何解算器，实现对相机姿态的高精度显式优化；
2) 将冻结的流式深度预测器视作虚拟RGB‑D传感器，消除对手工深度标注的依赖；
3) 采用Laplacian权重引导的新颖Gaussian插值策略，自动定位并补全未被现有Gaussians覆盖的细节；
4) 在多种流式后端（LingBot‑Map、STream3R、TTT3R等）上验证几何解算器的通用性。

**🔧 技术方法**

使用的技术与方法包括：
- ORB‑SLAM3框架的改造（加入直接深度残差、HUBER稳健核、相机姿态优化）;
- 流式3D基础模型（如LingBot‑Map）的深度预测，作为虚拟RGB‑D输入；
- 直接深度残差与图像重投影误差的联合优化；
- Laplacian引导的Gaussian插值与3D Gaussian Splatting渲染；
- 关键帧重放与30步联合细化（RGB+SSIM+深度逆深度损失）；
- Sim(3)对齐、FPS评估、深度与RGB损失等。

**📊 数据集**

实验数据集包括：
- 7‑Scenes（室内RGB‑Depth）;
- Oxford Spires（室外，基于激光扫描的深度）;
- RE10K（室内外短视频，真实摄像机）;
- TUM‑D（手持室内外序列）。所有序列均以518×392分辨率输入。

**📈 对比分析**

与MonoGS、S3PO‑GS、Photo‑SLAM、OTF等基线进行比较。性能表现：
- NVS指标：PSNR 24.48 dB、SSIM 0.839、LPIPS 0.293，均优于所有基线；
- 跟踪精度：ATE 8.58 cm（最佳或次优），RPE_t 0.524 cm，RPE_R 0.189°；
- 计算速度：约6.4 FPS（单RTX 4090 GPU），与基线相近；
- 深度误差：在所有三大数据集上均为最低（如7‑Scenes 0.15 m）。

**⚠️ 局限性**

局限性：
- 未实现闭环检测与动态场景处理；
- 仅依赖ORB特征，导致纹理弱或重复结构场景下跟踪鲁棒性下降；
- 对深度预测的噪声敏感，若流式模型失效会直接影响姿态与几何精度。

---

## 435. Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG

**arXiv ID:** 2610.36915 | [PDF](https://arxiv.org/pdf/2610.36915v1)

**作者:** Parv Khatri `[一作]`, Tanya Sah `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对检索增强语言模型中检索结果冲突的类型进行分类任务，系统性评估了在商业推理网关上提供的所有typed decision模型，并将其与Gemini LLM进行对比；还探究了树状与扁平问答、置信度门控级联、日期/顺序/误导源诊断以及端到端答案质量的影响。

**💡 创新点**

首次将typed decision模型应用于冲突类型标注任务，构建完整的模型清单和实验协议；提出树状分解不适用且直接扁平问答最优；通过置信度门控实现成本与性能折衷；设计多维诊断验证模型对日期、顺序及单一错误源的敏感性；在生成器端验证类型预测对答案行为的提升。

**🔧 技术方法**

使用typed decision模型（如1.13.0、Decider v1.2、GPT‑6 Luna Decisions等）做一次性概率预测；采用Gemini 2.5 Flash和Gemini 3.8 Flash作为LLM基线；构建置信度门控级联；通过期望校准误差(ECE)、F1、准确率等指标评估；利用Bootstrap 95%区间和McNemar检验进行统计比较。

**📊 数据集**

使用自研的冲突类型基准集（共458个查询），采自ConflictingQA、SituatedQA、FreshQA、QACC等，包含每个查询的前几条Google搜索结果、冲突类型标签及对应答案；对部分查询人工植入误导源以扩充Misinformation类。

**📈 对比分析**

在相同输入下按配对对比，决策模型在准确率上与Gemini 2.5 Flash相当（约65‑70%），且成本低于1/3、延迟降低约10‑20%；级联将最不确定的25%查询交给Gemini 3.8 Flash后，准确率与单独LLM相近，成本仅为其的25%；端到端生成器在使用预测类型时，答案符合预期行为的比例与使用黄金类型相当，维持在约88%（主评判）或73%（次评判）。

**⚠️ 局限性**

样本量有限，尤其Misinformation类仅5个真实样本；基准集与真实搜索流量分布不一致；不同模型对日期和顺序的敏感性差异大，需单独验证；部分模型非确定性导致预测漂移；实验在单一LLM与判定器环境下完成，未覆盖源视角的评判；未测量答案的事实依赖度和可信度。

---

## 436. A Serious-Game Testbed for Self-Governing Human-Machine Institutions

**arXiv ID:** 2610.36989 | [PDF](https://arxiv.org/pdf/2610.36989v1)

**作者:** Callum Berger `[一作]` (University of Lincoln), Stefan Sarkadi `[通讯]` (University of Lincoln)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 Megabike，一种可同时容纳人类和 LLM 代理的多智能体实时游戏测试平台，并用该平台进行四种性格（合作、懒惰、颠覆、对立）双车对战的 40 场实验，研究群体治理、规则变更、处罚准确性等方面的自治理行为。

**💡 创新点**

创新点在于：① 将物理协作、规则制定与治理决策耦合到同一可见、可记录的模拟环境中；② 通过“机构对象”让规则显式、可审计；③ 允许 LLM 代理以自然语言进行辩论，但所有行动都在统一的仿真器中强制执行；④ 通过“双车对战”设置检验混合人机团队在共享实体上的自治理能力。

**🔧 技术方法**

主要技术包括：Unity 3D 游戏引擎 + LLMUnity 框架；Qwen3.5‑4B‑Q4_K_M 语言模型；神经符号架构（LLM 生成语言 + 规则对象验证）；自定义的多代理协调、决策、监控与制裁模块；数据采集模块记录所有语音、动作与规则变更。

**📊 数据集**

使用的数据集为实验日志：40 场会话、10 场/条件、每场 5 名代理、每名代理分别对应四种性格。日志包含 9,255 条发言、33,229 次收集事件、64,510 次动作采样。

**📈 对比分析**

比较方法：在四种性格加倍条件下对比存活轮数、能量消耗、规则修改次数、领导与集体决策比例、处罚准确率等指标。结果显示：① 合作性格占优可延长存活轮数并减少 Owdi 碰撞；② 懒惰性格导致能量耗尽与碰撞增多；③ 颠覆与对立性格虽导致最高的规则严格度，却未显著提升治理稳定性；④ 集体决策被优先选择，领导选举多见于对立性格；处罚在集体决策下多针对得分最高者，领导决策则多处罚合规者，整体处罚准确率低于随机。

**⚠️ 局限性**

局限性包括：① 仅使用小型 LLM（Qwen3.5‑4B），未验证更大模型效果；② 试验全部为 LLM 代理，未纳入人类参与；③ 未开启骑手转移与审计角色，限制了自选择与制裁灵活性；④ 游戏回合数短、总时长有限，难以观察长期治理演化；⑤ 规则变更几乎零，未充分验证可变性对存活的影响。

---

## 437. GNN-Guided Network Reduction for Centralized Urban Traffic Routing

**arXiv ID:** 2610.36981 | [PDF](https://arxiv.org/pdf/2610.36981v1)

**作者:** Nikolai Antonov `[一作]` (Czech Technical University in Prague), Jan Hůla `[通讯]` (Czech Technical University in Prague)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于图神经网络的动态网络约简方法，用于在有限的运算时间内求解城市级别的集中交通路由问题；

**💡 创新点**

创新点在于：①针对每个OD流使用专门的GNN对路段进行流级别排序，从而构建针对当期车辆流的个性化约简网络；②将约简过程与传统的分段流MILP相结合，既保留了可行性保证，又显著降低了求解规模；

**🔧 技术方法**

技术主要包括：混合整数线性规划（MILP）建模、图神经网络（GNN）进行路段排序、基于约简网络的约束重构与求解、以及微观交通仿真（SUMO）验证路由效果；

**📊 数据集**

数据集涵盖四个真实城市道路网络：布拉格-德伊维奇、伦敦、纽约和悉尼，并在多种交通强度场景（静态/递增）下生成数千个路由实例；

**📈 对比分析**

与全图MILP、固定约简方法（SP、BR、SR）以及随机/LP排序等基线进行比较；在10秒运算预算下，GNN约简的相对目标差距平均仅为0.4–0.6%，比最优固定约简低约3–5个百分点；在SUMO仿真中，GNN方法也取得最高的到达率和最佳的行程时长/到达率比；

**⚠️ 局限性**

局限性在于：①MILP模型仅通过分段费用近似拥堵，无法完全捕捉交通灯、排队、车流相互作用等动态效应；②GNN需在每个场景单独训练，对大规模网络的迁移性能未知；③仿真评估仍基于高负荷测试，缺乏对常规交通情况的验证。

---

## 438. Mining the Nineteenth Century for the Thirteenth: Register-Scale Forced Alignment and HTR Bootstrapping via Legacy Regesta and Antiquarian Editions

**arXiv ID:** 2610.36956 | [PDF](https://arxiv.org/pdf/2610.36956v1)

**作者:** Tamas Kovacs `[一作]`, Daniel Luger `[通讯]` (University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过先用现代专家转写训练识别器，再利用十九世纪手稿与编辑文本作为零标签适配、词汇验证与拓扑分割的多层次辅助，系统评估并提升教廷登记簿（Pope Honorius III Registers）的手写文本识别（HTR）性能。

**💡 创新点**

创新点包括：1）将经典化编辑文本通过“curial projection”映射为与现代拼写一致的目标，形成零标签适配；2）将预训练的中古NLP NER模型蒸馏进识别器的NER头，实现实体监督而不需手工规则；3）利用Pressutti regesta与罗马日期等拓扑信息对识别流进行自动分段与质量校准；4）从多层（正字、词汇、拓扑）视角对旧文本与现代转写的距离与对识别的贡献进行定量分解。

**🔧 技术方法**

技术手段包括：Conformer‑CTC 编码器、Transformer注意力解码器、R2L‑BPE 双向解码、强制对齐、动态规划的条目对齐、语言模型与gazetteer偏置、NER head（RoBERTa、Span‑NER）蒸馏、curial projection规则、OCR错误校正与置信度校准。

**📊 数据集**

数据集：① Reg. Vat. 9（Pope Honorius III Registers）394页，其中1–203页有现代专家转写；② 19世纪编辑文本（Horoy、MGH、Theiner、Raynaldi、Pressutti regesta等），共覆盖约一半转写条目；③ 训练、开发、测试拆分（315/40/39页，13页测试）。

**📈 对比分析**

评估方法：WER、CER、OOV、NER F1、P/R/F1等指标；与oracle（现代标签）及基线（全现代训练）对比；结果表明：基线WER≈12.6%；投影后legacy线适配可将WER降至≈10.8%（≈2点提升）；在第2年迁移实验中，投影+重放提升≈1.6点；NER蒸馏后F1提升≈10个百分点；拓扑分段与质量校准在未转写页能估计≈2–3% CER。

**⚠️ 局限性**

局限性：① 测试集仅13页，样本量有限；② legacy文本覆盖不足，超203页后未知；③ 依赖Internet Archive OCR，未纠正错误；④ 文字形体与转写者混杂导致hand/转写者边界不清；⑤ 只在单一注册簿上验证，未验证跨注册迁移。

---

## 439. When Diffusion Recommenders Stop Sampling: Preserving Generative Diversity

**arXiv ID:** 2610.36954 | [PDF](https://arxiv.org/pdf/2610.36954v1)

**作者:** Louis Castres Saint Martin `[一作]` (Criteo AI Lab), Patrick Gallinari `[通讯]` (Criteo AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出三阶段潜在扩散模型DARTS，用于生成多样化且准确的顺序推荐

**💡 创新点**

通过分离表示学习、生成训练和离散排名，消除采样崩塌，并引入后验预测聚合提升准确率

**🔧 技术方法**

使用Transformer编码器/解码器、潜在扩散模型、连续空间学习与单阶段离散监督、Gumbel噪声对比实验

**📊 数据集**

在Amazon 2023的乐器、工业科学、视频游戏三大类别数据集上进行评测

**📈 对比分析**

与SASRec、DreamRec、DiffuRec等基准对比，DARTS在Recall@K、NDCG@K上提升6.8%–12.0%，多样性诊断显著高于对手，采样聚合进一步提升性能

**⚠️ 局限性**

局限于单一下一步离散评估，未考虑多未来预测、结构化输出，且多采样推理带来吞吐量下降

---

## 440. From Walks to Skeletons: A Reusable Transport Architecture for Interactive Monte Carlo PDE Solvers

**arXiv ID:** 2610.36931 | [PDF](https://arxiv.org/pdf/2610.36931v1)

**作者:** Yunfeng Liao `[一作]`, Xiucheng Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种可重用的传输骨架（DAG）框架，用于交互式Monte Carlo PDE求解，能够在多次查询、边界条件变化、几何编辑或区域传输编辑时重用已采样的随机路径，实现高效的图遍历、条件细化、压缩与局部更新。

**💡 创新点**

创新点在于：①将Walk‑on‑Spheres轨迹组织成持久化DAG并引入Poisson‑kernel重加权的路径附件；②提出条件路径细化、共享hub、局部几何修复和低秩区域编辑等机制，既保留已有采样，又支持动态修改；③实现可覆盖、可压缩、可增量更新的传输骨架，使得多次求解在一次构建后几乎无额外成本。

**🔧 技术方法**

主要技术包括Walk on Spheres (WoS)、Poisson 核重加权、Doob h‑transform、方差导向的精细化、图压缩与覆盖保持、双向WoS、低秩修正、DAG遍历和空间索引。

**📊 数据集**

实验使用多对象三维网格域（包含不规则障碍的球形包围盒）以及特定的杯形、双球、火山-村落等场景，用于评估构建、求解、压缩、动态更新和区域编辑的效果。

**📈 对比分析**

与传统WoS和独立路径WoS对比时，骨架在相同采样预算下可提升约2–3个数量级的求解速度，同时误差比单独WoS低30–60%；构建一次后，后续多次边界重解的累计成本仅为单次WoS求解的约1/8，且在源项、密集查询、动态几何等实验中均保持低误差与显著时间/内存优势。

**⚠️ 局限性**

局限性包括：①附加阈值的设定需权衡重用率与误差；②骨架构建成本高，需合适采样预算；③对极小高贡献对象的采样仍需Bi‑WoS支持；④过度压缩可能削弱后续附件机会；⑤低秩区域编辑仅修改传输，而非真实PDE，适用于交互调整而非科学计算。

---

## 441. DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models

**arXiv ID:** 2610.36950 | [PDF](https://arxiv.org/pdf/2610.36950v1)

**作者:** Egor Maximov `[一作]` (MIL Team), Aleksei Goncharov `[通讯]` (MIL Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一种针对混合注意力语言模型的KV缓存压缩码流模型，在推理阶段仅压缩键值缓存而保持循环状态不变。

**💡 创新点**

首次实现KV缓存的端到端学习，联合优化变换、量化、熵模型与条件熵预测，并通过与冻结模型的输出分布相匹配的Rate‑Distortion目标进行训练。

**🔧 技术方法**

使用PCA联合基底、非线性校正网络、混合量化、rANS熵编码、Laplace熵模型、条件Token/邻近预测、KL散度损失、Rate‑Distortion训练以及混合速率控制等技术。

**📊 数据集**

主要数据集包括FineWeb‑Edu、FineWeb、OpenR1‑Math进行校准与训练；评估使用RULER（4K/32K）、LongBench（18项任务）、MMLU、GPQA Diamond、IFEval等。

**📈 对比分析**

与KVTC、xKV等基线相比，在Qwen3.5‑2B上实现了每token约210字节（≈55–100倍BF16压缩），得分下降≤3.6点；在RULER 4K、LongBench、RULER 32K等任务上均比KVTC在相同或更小请求内存下获得更优或相近性能。

**⚠️ 局限性**

局限性：仅在Qwen3.5‑2B评估，未验证其他混合模型；每个压缩率需单独训练，难以实现多速率；跨设备解码一致性未验证；首次token时间仍受解压影响；对超过32k长文本的实际性能缺乏实验。

---

## 442. Total Deformation-based MPM for Perfectly Elastic Solid Simulation with Strong Deformations

**arXiv ID:** 2610.36945 | [PDF](https://arxiv.org/pdf/2610.36945v1)

**作者:** Timna Böttcher `[一作]` (RWTH Aachen University), Jan Bender `[通讯]` (RWTH Aachen University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种新的基于MOM的完全弹性固体求解器，能够在大变形下避免数值断裂并天然支持接触处理。

**💡 创新点**

核心创新在于直接利用参考网格计算粒子形变梯度，同时在当前网格上求取弹性力，并加入零能量模式抑制，既消除断裂，又允许粒子层面碰撞与动画操作。

**🔧 技术方法**

采用MPM框架、参考网格与当前网格双重传递、Neo‑Hookean弹性模型以及梯度修正与零能量势能抑制技术。

**📊 数据集**

使用Stanford Bunny模型进行形变与碰撞测试，并在立方体扭转实验中验证。

**📈 对比分析**

与传统MPM和TLMPM对比，方法在保持形变完整的同时自然捕捉自碰撞；性能上每步耗时约为MPM的3倍（约73 ms），但精度更高、无断裂。

**⚠️ 局限性**

局限性包括显式求解器导致的稳定性受限、对不同MPM物体间接触的处理仍需改进，以及目前仅适用于完全弹性材料。

---

## 443. JevRec: Linguistic Encoding with Jev-like Decision Modeling for LLM-based Generative Recommendation

**arXiv ID:** 2610.36937 | [PDF](https://arxiv.org/pdf/2610.36937v1)

**作者:** Qiyong Zhong `[一作]` (Zhejiang University), Li Zhang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于可变长度语义标识符（SID）的生成式推荐框架，采用语言学编码和Jev-like决策建模，实现长度与内容的分离预测；

**💡 创新点**

①根据物品交互频率动态分配SID长度，热门物品使用短SID，长尾物品使用长SID；②用独立的长度预测头和内容生成头，缓解梯度冲突；③长度预测引导的解码与跨长度排序，提高推荐质量；

**🔧 技术方法**

生成式推荐（自回归模型）、语言学编码、共享代码本与冲突解决token、Jev-like决策模型、长度预测头、条件内容生成头；

**📊 数据集**

使用三大公开推荐数据集（具体数据集名称未给出，但一般为常用如MovieLens、Amazon、Alibaba等公开数据集）；

**📈 对比分析**

与多种基线模型（如TIGER、其他生成式推荐方法）在Hit@K、NDCG@K等指标上进行对比，JevRec在所有指标上均优于对照基线；相较于TIGER，训练与推理时间更低，梯度冲突更少；

**⚠️ 局限性**

在极大规模数据环境下实现可变长度编码可能增加计算复杂度，需要更多调参；共享代码本容量有限，可能限制对长尾物品的丰富语义表达；缺乏对实际部署场景的进一步验证与鲁棒性分析。

---

## 444. Gaussian Process Bandits for Constraint Black Box Lever Optimization

**arXiv ID:** 2610.36935 | [PDF](https://arxiv.org/pdf/2610.36935v1)

**作者:** Bram van den Akker `[一作]` (Booking.com), Anastasiia Kornilova `[通讯]` (Booking.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在共享预算的离散分配问题中，提出并实现了基于高斯过程上置信界（GP‑UCB）的组合核（Matérn + 线性）算法，先在高保真模拟器中验证其在噪声、漂移和反应性环境下的鲁棒性，再在真实生产环境中进行六周随机对照实验（RCT）验证性能。

**💡 创新点**

创新点在于：① 构建可模拟非平稳、对抗性动态的预算分配仿真环境；② 通过组合核将全局线性趋势与局部非平滑结构结合，显著提升在高噪声、分布漂移下的探索‑利用平衡；③ 将理论上的 GP‑UCB 从实验室快速迁移至线上生产，并通过 RCT 证实其效益。

**🔧 技术方法**

主要技术：高斯过程回归（GP），GP‑UCB 决策策略，组合核（Matérn + 线性），离散化预算分配空间的模拟器，基于噪声/漂移/反应性变换的合成奖励函数，随机对照实验。

**📊 数据集**

数据集：① 合成奖励景观（通过高斯曲线加权构造的可微分函数，包含5–9个激活点，模拟不同噪声、漂移和反应性变化）；② 真实生产环境中的历史预算分配和收益记录，用于模型初始化和 RCT 的对照/实验组对比。

**📈 对比分析**

比较方法：使用累计惩罚（cumulative regret）与差异惩罚对比；与常数基线、ε‑greedy、理想最优基线以及纯核（RBF、Matérn）对比。结果显示，组合核在所有四种环境下平均降低约15–20％的 regret，Matérn+线性在所有挑战性场景下最早实现显著优势（p<10⁻³），RCT 中表现出 90% 置信度的显著效率提升。

**⚠️ 局限性**

局限性：① 仅在低维（3 个分段）情境下验证；② 采用固定的组合核，未针对不同非平稳程度自动调节核权重；③ 仅考虑每周一次的离散化调整，无法处理更高频率或更大维度的预算分配；④ 未将情境特征或时间动态纳入模型，后续需扩展至上下文带和时空核。

---

## 445. TrawMem: Reframing Agent Memory as Task-Oriented Role-Adaptive Workspaces

**arXiv ID:** 2610.36918 | [PDF](https://arxiv.org/pdf/2610.36918v1)

**作者:** Mingyu Zhang `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种任务导向的角色自适应工作空间（Task‑Oriented Role‑Adaptive Workspace）来改进大语言模型代理的记忆系统。

**💡 创新点**

创新点在于将记忆单元按任务角色（直接支撑、桥接、上下文）分配并联合选择补充记忆，而非仅靠语义相关度；同时将交互历史划分为事件线程并构建线程图以捕获跨线程关系。

**🔧 技术方法**

技术方法包括：事件线程分割、线程图构建、基于任务需求的角色赋值、使用软权重与距离成本的优先级计算，以及基于子模函数的贪婪选择实现工作空间构建。

**📊 数据集**

使用了 LoCoMo（多轮对话）和 MemGallery（多类型问答）两个公开基准数据集，并在 Qwen3.5‑9B、Qwen3.8‑27B、DeepSeek‑V4‑Flash 与 GPT‑5.6‑Luna 四种后端模型上进行评测。

**📈 对比分析**

与现有记忆系统（如 SimpleMem、LightMem、Mem0 等）对比，TORAW 在宏观 F1、BLEU‑1 及 LLM‑Judge 分数均提升 2.23–3.14 点，J 分数提升 4.07–4.31 点，并在多任务情景下保持一致的性能优势。

**⚠️ 局限性**

局限性包括：检索与工作空间构建耗时相对较高、对线程划分与角色设定的依赖导致对对话结构的敏感性、以及在隐私与偏见治理方面需要进一步完善。

---

## 446. Digital Twin-Guided Wireless Localization via Probabilistic Inverse Learning

**arXiv ID:** 2610.36905 | [PDF](https://arxiv.org/pdf/2610.36905v1)

**作者:** Mahmoud Saad Abouamer `[一作]` (Aalborg University), Petar Popovski `[通讯]` (Aalborg University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出一种基于数字孪生（DT）的概率逆学习框架PRISM，用单天线AP和分布式感知实现精准无线定位。

**💡 创新点**

创新点在于：①将定位问题建模为概率逆问题，使用学习得到的条件分布生成候选位置；②仅利用DT的正向一致性（不需要精确或可微）对候选位置进行加权重估；③在训练时通过候选正则化控制候选空间大小，保证候选多样性与DT一致性兼顾。

**🔧 技术方法**

主要技术包括：Transformer编码器提取多AP CIR特征；对数变换的高斯候选分布预测位置均值与协方差；DT一致性能量采用SSIM相似度；有限候选正则化与温度加权重；平方马氏距离评估不确定性校准。

**📊 数据集**

使用Sionna射线追踪产生的Munich场景（及San Francisco场景）模拟数据，频率6 GHz、100 MHz带宽、单天线AP数量从5到15个，位置网格为1 m间距。

**📈 对比分析**

与CNN‑Direct、Transformer‑Direct、经典指纹C‑FP、Dense‑FP（0.5 m网格）以及RFMap‑FP对比。PRISM在10个AP时平均误差降至1.11 m、median误差0.70 m，较Dense‑FP（2.17 m）低约49%；在15个AP时median误差0.66 m；对DT失真、材料替换及观测SNR失配表现出稳健性。

**⚠️ 局限性**

局限性包括：①依赖准确的数字孪生几何，严重物理误差仍会影响；②目前仅在静态单天线环境下验证，动态或多天线场景待扩展；③使用仿真射线追踪数据，真实世界实验尚未验证；④候选采样与DT加权对计算开销有一定影响。

---

## 447. Barik transform and Barik eliminant: a novel algebraic method to solve polynomial equations in the sine and cosine of one variable

**arXiv ID:** 2610.36985 | [PDF](https://arxiv.org/pdf/2610.36985v1)

**作者:** Sandipan Bandyopadhyay `[一作]` `[通讯]` (Indian Institute of Technology Madras), Sandipan Bandyopadhyay (Indian Institute of Technology Madras)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Barik 变换和 Barik 消除子，用以将包含至少一条关于正弦、余弦线性关系的三角多项式方程组转化为代数方程，并直接求解该方程组。

**💡 创新点**

创新点在于引入 Barik 变换这一新的几何代数工具，既避免了传统半角切变换导致的参数奇异和多项式次数翻倍问题，又能在不使用结果式或矩阵数值求解的情况下得到紧凑的闭式消除子。

**🔧 技术方法**

主要技术包括几何解释（单位圆与直线交点）、线性变换（Helmert 变换）、代数消元（直接替换或多项式除法）以及符号化简来推导闭式解。

**📊 数据集**

未使用任何实验数据集，本文全部为理论推导与符号分析。

**📈 对比分析**

通过与传统半角切变换（HTT）的对比，阐述了 Barik 变换在避免参数奇异、降低多项式次数、减少计算量方面的优势；由于缺乏数值实验，性能提升仅在理论层面得到说明。

**⚠️ 局限性**

局限性主要是仅适用于至少含一条正弦、余弦线性方程的方程组，且在某些特殊系数取值导致变换矩阵奇异或 α = 0 时需要特殊处理；不构成对 HTT 的通用替代方案。

---

## 448. Advancement or Acceleration? How Data Curation Improves LLM Training

**arXiv ID:** 2610.36983 | [PDF](https://arxiv.org/pdf/2610.36983v1)

**作者:** Jiaxing Wang `[一作]` (JD.com), Qixia Jiang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析并实验验证：数据策划能在存在可纠正的训练‑目标差距时真正提升最终性能（advancement），而在差距为零时仅能加速收敛（acceleration），并提出可观测的 Δ‑测试用以诊断两种效果的出现。

**💡 创新点**

创新点在于将加速与提升明确区分，给出理论证明：提升需要一个可纠正的 train‑target gap，并且将该判别转化为单次基准跑即可计算的 Δ‑测试；同时在可解的感知器模型与真实 LLM 训练中统一验证。

**🔧 技术方法**

技术方法包括：统计物理可解的教师‑学生感知器模型与 replica 解析、Δ‑测试设计、以及大规模 GPT‑style LLM（160M/1.1B）在多域混合和选择场景下的训练与评估。

**📊 数据集**

使用的数据集包括 SlimPajama（7 域）、OpenWebMath、MetaMath‑Mammoth、TinyLlama‑1.1B、以及公开的 5B‑token OpenWebMath 候选池等，涵盖域分布差异与外部目标的混合。

**📈 对比分析**

实验对比了 Uniform/Random、RegMix、DoGE、Skill‑It、Aioli、TANDEM 等混合优化器，和 RHO‑1、BLADE 等选择方法；结果显示当 Δ ≈ 0 时，所有方法几乎无提升；当 Δ > 0 时，TANDEM/DoGE 等在混合任务上可获得 2‑8 以上的 perplexity 或 accuracy 提升，表明真正的提升。

**⚠️ 局限性**

局限性包括：Δ‑测试只能捕捉到训练‑目标损失差异，可能忽略更细粒度的分布不匹配；实验主要聚焦于 160M–1.1B 规模 LLM，尚未在极大规模或极度稀缺数据场景中验证；理论证明基于感知器模型，实际 NLP 任务的可迁移性仍需进一步探究。

---

## 449. Toward Circuit-Aware Mutation Testing for Quantum Programs: Exact Visibility and Execution Planning

**arXiv ID:** 2610.36976 | [PDF](https://arxiv.org/pdf/2610.36976v1)

**作者:** Andriy Miranskyy `[一作]` (Toronto Metropolitan University), Jose Campos `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套面向量子程序的变异测试框架，能够在理想无噪声模型下区分因语义差异、测量不可见性、统计不确定性等原因导致的存活变异体，并为测试执行计划、检测结果报告和测试用例改进提供决策依据。

**💡 创新点**

创新点包括：① 通过将变异影响表述为在变异点前的状态、后继测量的反向传播以及相对单元变换，给出了可直接计算或快速判定的“光锥”和“交换子”判定规则；② 对固定轴旋转、固定一次量子门替换、Clifford 后缀等常见变异场景推导了闭式或多项式可计算的期望差公式；③ 引入了基于旋转角度与测量范围的角度上限判定，既能在不执行任何测量的情况下预先决定某变异对给定测试无效；④ 结合统计不确定性和误差预算，给出可在不执行完整采样的情况下对检测结果进行可信度下界/上界估计；⑤ 设计了完整的执行规划流程和记账机制，保证即使在部分测试未完成的情况下也能保留可靠的变异得分区间。

**🔧 技术方法**

核心技术包括：量子电路的状态传播和测量的逆向传播；量子算子（Hermitian、单元变换）的交换子和规范化；线性代数中的算符范数、迹范数与最优测量（Helstrom）理论；旋转轴的参数移位公式；统计学中的 Hoeffding 以及独立复制判别理论；以及对 Clifford 稳态（stabilizer）电路的二进制表格表示和经典可计算性。

**📊 数据集**

研究未使用真实量子数据集，而是通过理论证明、闭式公式推导以及在小规模（≤ 3–4 qubit）电路上的可执行实现（Python 示例）来验证框架。所有实验均在理想无噪声模型下完成。

**📈 对比分析**

在有限的实验规模中，框架能够在不执行变异体的情况下通过结构过滤、角度上限判定和交换子判定省略多达数十%的测试执行，从而显著降低资源消耗；对已执行的变异，闭式公式可准确给出期望差并与统计采样结果一致。与传统仅基于统计差异的变异测试相比，该框架在变异检测率上保持或提升，同时提供更精细的原因解释与资源估计。

**⚠️ 局限性**

局限性：① 仅在理想无噪声模型下分析，未考虑设备噪声、漂移和中间测量等实际量子硬件特性；② 变异范围限于单个门或子电路的替换，未覆盖多阶变异；③ 对于大规模电路，状态或观测符号的显式计算仍可能呈指数级开销；④ 统计判定仍需要假设独立采样且对置信区间的计算依赖理论上限，实际执行时可能受限于实验误差。

---

## 450. Dense Representation Learning and Differentiable Shape Modeling for Spine Landmark Detection

**arXiv ID:** 2610.36951 | [PDF](https://arxiv.org/pdf/2610.36951v1)

**作者:** Augustin Daridon `[一作]`, Loic Le Folgoc `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于对比学习的像素级密集表征框架 Pix2Point，用通用 U‑Net 背骨生成像素嵌入，并通过与标记原型的对比损失得到自由形状的热图；随后利用 SpineSAM 通过 PCA 形状先验和可微能量优化，对热图进行全局形状约束与细化，配合图搜索初始化。

**💡 创新点**

创新点在于：①将解剖标记检测视作像素级对比学习任务，摆脱传统高斯热图回归的局限；②引入可学习的形状先验与可微能量框架，实现对全局脊柱配置的透明约束；③设计基于图的离散优化（GDO）快速逼近全局最优，为可微优化提供良好初值。

**🔧 技术方法**

核心技术包括：U‑Net 编码-解码骨干、像素级对比损失、训练/批量依赖原型、预训练的自监督像素嵌入、PCA 形状先验、可微能量式 MAP 优化（LBFGS）、离散最短路图搜索（Dijkstra）。

**📊 数据集**

使用公开的 AASCE 骨髓 X‑ray 数据集，以及两份私有双视图（冠状/矢状）X‑ray 数据集（分别含 93 个标记）。

**📈 对比分析**

与传统高斯热图回归、CARN、SpatialConfiguration‑Net、nnLandmark 等基线及脊柱专用方法对比；在 AASCE 上 MRE 下降至 2.18mm，SDR 提升至 56.6%，在私有数据集上亦实现与 SOTA 相当或略优的定位精度与临床参数排名。

**⚠️ 局限性**

局限性包括：①对热图局部最大值的依赖仍可能导致多模态局部最优；②需要对骨骼形状进行 PCA 建模，若数据不足或形态异常（多/少椎骨）可能影响先验质量；③对预训练的自监督嵌入和可微优化的计算资源要求较高。

---

## 451. Developing a Dynamic Multi-factor Authentication System for Securing Smart Organizations

**arXiv ID:** 2610.36926 | [PDF](https://arxiv.org/pdf/2610.36926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 452. Algorithmic Monoculture in Machine Learning-based Site-Specific Fertilizer Recommendation

**arXiv ID:** 2610.36941 | [PDF](https://arxiv.org/pdf/2610.36941v1)

**作者:** Abdelghani Belgaid `[一作]` `[通讯]` (Mohammed VI Polytechnic University), Abdelghani Belgaid (Mohammed VI Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了连续施肥建议的算法单一化问题，构建共享与特定误差模型分配与外部性量化框架，并通过Herfindahl–Hirschman指数与协方差一般化推导多模型组合优化；

**💡 创新点**

提出共享模型误差与交叉模型协方差的三元框架，推导单一化阈值与精度约束凸组合策略，首次用经济学模型量化算法单一化对社会损失的影响；

**🔧 技术方法**

采用经济学模型、HHI分析、协方差一般化二次优化、两模型阈值解析以及受限均值方差组合等方法；

**📊 数据集**

使用摩洛哥五种谷物系统（雨养大麦、粮食玉米、饲料玉米、灌溉小麦、雨养小麦）共132,017条推荐场景，评估9种回归模型族；

**📈 对比分析**

对比四种分配策略（最优单一化、均匀分配、受限损失最小化、无约束社会损失最小化），在中心校准下受限组合将社会损失降低约10.9–41.5%，平均约19.4%，而均匀分配虽降低损失但超出10%私人成本上限；

**⚠️ 局限性**

仅基于预测误差而非因果产量或经济效益，未能识别共同误差比例、外部性权重与协方差，外部性模型过于简化，且采样独立性假设和R^2无法区分共享误差，结果仅为条件模拟损失而非实际福利变化。

---

## 453. TDLM: Adaptive Temporal Integration for Diffusion Language Models

**arXiv ID:** 2610.36940 | [PDF](https://arxiv.org/pdf/2610.36940v1)

**作者:** Bohang Sun `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种Temporal Self-Conditioning Diffusion Language Model（TDLM），在冻结的扩散语言模型中加入轻量级位置对齐的时间注意力，使每个未解决位置能够复用自身历史隐藏表示，从而提高生成效率；

**💡 创新点**

创新点在于：①在冻结的解码器中引入位置对齐的时间注意力模块；②通过自生成轨迹的简单交叉熵自蒸馏目标训练早期预测；③使用因果时间注意力实现对采样轨迹的并行评估；

**🔧 技术方法**

采用的技术包括：冻结的LLaDA/Dream扩散语言模型、位置对齐的时间注意力、简单交叉熵自蒸馏、置信度阈值并行解码、因果时间注意力以及并行轨迹评估；

**📊 数据集**

实验使用的数据集有：MATH‑500、GSM8K（数学推理）和MBPP sanitized（代码生成），以及3,500条自生成轨迹用于训练；

**📈 对比分析**

与原始扩散语言模型和Fast‑dLLM在256 token生成预算、32块大小的设置下对比，测量平均前向次数和秒数；TDLM在速度上可达7.15×加速，相对Fast‑dLLM提升至2.72个百分点准确率，在多任务上排名第一、二或三；

**⚠️ 局限性**

局限性包括：依赖冻结模型生成的教师轨迹，质量受限；自蒸馏无法纠正教师错误；时间注意力仅覆盖少量层；未探索更长历史或更多任务；可能导致错误累积需要更高质量教师或联合微调。

---

## 454. Provenance Signatures for Low Rank Adaptations of Image Diffusion Models

**arXiv ID:** 2610.36933 | [PDF](https://arxiv.org/pdf/2610.36933v1)

**作者:** Tobias Cheung `[一作]` (University of Edinburgh), Rik Sarkar `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种黑盒协议，用于检测在线图像生成服务是否使用了基模型的LoRA（低秩适配）版本，从而判断模型的出处与可疑用途。

**💡 创新点**

创新点在于：①引入了基于局部内在维度直方图的模型签名（dimension signature），将生成图像的局部几何信息转化为可比较的统计量；②采用假设检验（hypothesis testing）与Wasserstein距离相结合，形成开源世界（open‑world）下的判定方法；③在实验中首次对多种LoRA适配与负样本模型进行系统评估，并与现有监督式属性识别方法对比。

**🔧 技术方法**

主要技术包括：扩散模型（Stable Diffusion v1.5）、LoRA微调（低秩更新）、MADA（局部维度估计）算法、Wasserstein距离、统计假设检验，以及使用VAE编码器对图像进行嵌入。

**📊 数据集**

数据集：基模型使用Stable Diffusion v1.5的训练数据；LoRA适配分别在Food101、Pokémon、WikiArt、CheXpert（胸部X光）和EuroSAT（卫星图像）上训练；负样本模型包括DeciDiffusion v1.0、PixArt-α-512、Stable Diffusion v2.1、SDXL和v3.5 medium；此外使用BLIP‑2 Flan‑T5‑XL对无字幕数据进行自动标注。

**📈 对比分析**

与监督式属性识别基线（Nataraj19、Wang20、Giudice21、Ha26）以及无监督的PCA‑64 Wasserstein进行对比。结果显示：在问题 I（负样本区分）中，本文方法AUROC=0.855、F1=0.800，虽然低于监督基线但仍具备竞争力；在问题 II（基模型与LoRA区分）中，AUROC=1.000、F1=0.943，明显优于所有基线，证明其在开放世界场景下的有效性。

**⚠️ 局限性**

局限性包括：仅针对LoRA适配；对高LoRA比例或不同适配方法（如全微调、其他低秩策略）可能效果下降；需要大量采样（每个模型1000张图）和对基模型的访问；方法对LoRA尺度敏感，在极端尺度下识别率可能下降；未验证对更大模型或不同架构（如VAE空域）的一般化能力。

---

## 455. Nothing Left Behind: Finding Evidence from a Large Randomized Evaluation Corpus

**arXiv ID:** 2610.36910 | [PDF](https://arxiv.org/pdf/2610.36910v1)

**作者:** Giuliano Martinelli `[一作]` (Development Impact AI Lab, World Bank Group), Samuel Paul Fraiberger `[通讯]` (Development Impact AI Lab, World Bank Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对开发经济学领域的7,354项随机对照试验（R CT）的证据检索进行实验，比较基于标签的检索与基于文本嵌入+LLM过滤的两阶段检索，评估召回、精确率与计算成本。

**💡 创新点**

创新点在于将语义检索与查询特定的LLM过滤相结合，既提升召回率又保持高精度，并提出将检索与标签组织分离的混合架构。

**🔧 技术方法**

使用的技术包括Gemini 2.5 Flash、Voyage、E5‑large‑v2、Granite等文本嵌入模型进行候选检索，Gemini 2.5 Flash作为实体链接和相关性过滤的LLM。

**📊 数据集**

使用的数据集是ImpactAI 7,354篇RCT的74,826个干预‑结果对，以及包含Beta测试者、Synthetic、双注释、Review‑derived等多种评估集合。

**📈 对比分析**

通过在Beta Tester和Synthetic查询基准上对候选检索进行Recall/Precision/F1比较，并在Review‑derived基准上测量宏观论文召回率，发现基于嵌入的检索在K=2,500时召回率可达85%（vs 45%），精度约94.7%，但过滤成本随深度显著增加。

**⚠️ 局限性**

局限性包括仅在单一RCT语料上实验，评估依赖LLM判断与有限人工标注，未覆盖完整文献；候选预算上限限制完整检索；过滤模型可解释性差，且查询时延随深度显著上升。

---

## 456. HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion Language Models

**arXiv ID:** 2610.36901 | [PDF](https://arxiv.org/pdf/2610.36901v1)

**作者:** Xiang Li `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为有限生成预算下的掩码扩散语言模型提出HiLo-RD方法，通过跨预算表示自蒸馏提升低预算推理质量。

**💡 创新点**

创新点在于将同一模型在大窗口下的可信表示对齐到小窗口视图，解决预算诱发的过早提交问题，并通过表示层蒸馏同时改进词预测与提交顺序。

**🔧 技术方法**

技术包括掩码扩散解码、LoRA参数高效微调、表示层自蒸馏（HiLo-RD）、快速回滚停止规则和自我迭代蒸馏。

**📊 数据集**

使用的数据集包括数学推理的GSM8K、MATH500，以及代码生成的HumanEval、MBPP。

**📈 对比分析**

与原版LLaDA-8B-Instruct以及Masked SFT、Diffu-GRPO、OPSD、dOPSD等后训练方法对比，HiLo-RD在GSM8K上准确率提升9.4个百分点、可读长度缩短约25%，在MBPP上提升pass@k且保持更短响应，整体性能优于或匹配基线。

**⚠️ 局限性**

局限性包括对任务的依赖性强、需额外在大窗口下跑高预算轨迹生成教师样本、对超长序列的压缩能力仍有限，以及需要手工设计停止规则。

---

## 457. Natural Hard-Negative Mining from Positives for Instruction-Following Retrieval

**arXiv ID:** 2610.36900 | [PDF](https://arxiv.org/pdf/2610.36900v1)

**作者:** Chaolang Li `[一作]`, Pengwen Dai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将原始的查询-正例-负例三元组转换为指令条件下的对比训练样本，构建了自然硬负样本挖掘方法。

**💡 创新点**

创新点在于利用可转换负例生成最小化指令，并将原正例作为自然硬负例，避免长篇负例生成。

**🔧 技术方法**

采用候选清理、反转构造和几何过滤三阶段自动化管道，以及信息检索信息对比损失训练双编码器。

**📊 数据集**

使用 MS MARCO 文档集构建 6,716 条训练实例，并在 InstructIR 与 FollowIR 评测集上进行评估。

**📈 对比分析**

与 Promptriever、InF-IR 等基线比较，Qwen3-Embedding-0.6B 在 InstructIR 的 Robustness@10 和 nDCG@10 及 FollowIR 的 p-MRR 上均优于或匹配更大模型，表明数据范式带来显著提升。

**⚠️ 局限性**

局限性在于未对更大规模训练语料或更大模型进行实验，且训练集仍相对较小。

---

## 458. ASNER: Software Architecture Recovery with Adaptive Semantic Neighborhoods and Evidence-Grounded Refinement

**arXiv ID:** 2610.36894 | [PDF](https://arxiv.org/pdf/2610.36894v1)

**作者:** Wenhao Wang `[一作]` (Wuhan University), Tiankuo Zhao `[通讯]` (Wuhan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种基于LLM的自动软件架构恢复方法ASNER。

**💡 创新点**

通过结构化责任描述、项目自适应语义邻域和基于证据的边界细化，实现责任感知的语义关系。

**🔧 技术方法**

结合ENRE依赖抽取、LLM（如DeepSeek、Qwen等）生成文件责任、Qwen3-Embedding-4B向量化、SGASN自适应邻域算法、图融合与社区检测以及ROA推理代理。

**📊 数据集**

在12个开源项目（C、C++、Java、Python，NLOC 5k–2.1M）上进行评估。

**📈 对比分析**

与SemArc、SARIF、GAER-GAT等十余种基线相比，ASNER在宏平均下在ARI、c2c@66等指标提升19.5%和32.9%，整体实现最佳平均性能。

**⚠️ 局限性**

依赖LLM生成责任的准确性受限，且对大规模项目的推理成本高，未探索多模态或动态代码变化的适应性。

---

## 459. Latent Residual Reasoning for Ranking in Generative Recommendation

**arXiv ID:** 2610.36890 | [PDF](https://arxiv.org/pdf/2610.36890v1)

**作者:** Haihan Shi `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在生成式推荐中提出了基于残差的潜在推理框架 Refiner，利用候选项特定的推理标记、历史查询以及 LoRA 参数高效微调，实现多步分数细化；

**💡 创新点**

创新点在于（1）残差推理机制将多步推理视为对初始分数的增量修正，紧扣排序目标；（2）历史查询提供压缩的全局用户上下文，增强候选特定推理；（3）通过 LoRA 与额外预测头实现参数高效微调，避免全模型重训练；

**🔧 技术方法**

采用生成式推荐 Transformer、候选掩码、自动回归推理标记、历史查询、LoRA 参数适配、残差分数更新等技术；

**📊 数据集**

实验数据集包括 Steam、MovieLens10M、70M 用户、4M 商品、20亿交互的工业离线数据，以及线上 A/B 测试流量；

**📈 对比分析**

与 xDeepFM、DIN、SASRec‑ranking、MTGR、ReaRec 等基线对比，使用 AUC、GAUC、CTR、CTCVR 等指标；Refiner 在公测集 GAUC/CTR 最高提升约 1–4%，在工业离线和线上 A/B 测试中分别提升 PV‑CTR 0.75%、UV‑CTCVR 0.21%、GMV 0.46%；

**⚠️ 局限性**

局限在于（1）由于接口和业务差异，无法与 OnePiece 等完整候选交互模型完全对比；（2）使用固定推理深度，未实现自适应深度选择；（3）已部署的强基准模型提升空间有限，表现提升幅度相对有限。

---

## 460. Omni-IGN: Onboard Image-Goal Navigation for Drones with Omnidirectional Vision

**arXiv ID:** 2610.36968 | [PDF](https://arxiv.org/pdf/2610.36968v1)

**作者:** Zichen Yan `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了全机载的Omni-IGN图像目标导航系统，利用四镜头鱼眼相机实现全方位视觉，结合强化学习策略与碰撞安全层，实现连续动作的目标导向飞行。

**💡 创新点**

自注意力机制自适应融合四视角的目标相关特征，动态权重提升目标指引；使用学习式立体深度估计替代深度相机，安全层实时修正不安全动作；实现连续动作、无离线地图、零射击实地转移。

**🔧 技术方法**

端到端强化学习（PPO）+ ResNet9 编码器+ 自注意力+ GRU记忆+ Fast-FoundationStereo 深度推断+ 安全层碰撞预测+ FP16 TensorRT 推理+ NVIDIA Jetson Orin NX+ OAK-4P-New+ Livox LiDAR。

**📊 数据集**

Gibson（训练/测试）、MP3D、HM3D（交叉域评估）、Habitat 及 IsaacSim 实验环境。

**📈 对比分析**

与多种 SOTA ImageNav（Zer、OVRL-V2、FGPrompt、Mem-Aug、VGM、TSGM、SIGN）对比，在连续动作基准下 SR 87.8% SPL 64.2%（相较 SIGN 提升 1.5% SR 与 10.9% SPL）；在 MP3D、HM3D 连续动作下 SR 70.6% SPL 69.2%；IsaacSim 90 次飞行成功率 95.6%，SPL 0.91，平均动作修正率 23.3%。

**⚠️ 局限性**

对高动态/遮挡严重场景的深度估计和安全层仍易产生误判；安全层基于 16-bin 扫描，对侧向碰撞敏感；学习式深度在光照/纹理稀缺时准确性下降；模型仍依赖仿真训练，跨域鲁棒性有待提升。

---

## 461. LearnPrunerV2: Learnable Progressive Token Pruning for Vision-Language Models

**arXiv ID:** 2610.36986 | [PDF](https://arxiv.org/pdf/2610.36986v1)

**作者:** Rinyoichi Takezoe `[一作]` (Li Auto Inc.), Kaiwen Long `[通讯]` (Li Auto Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LearnPrunerV2，一个可学习的进阶视觉令牌剪枝框架，利用阴影令牌估计重要性并实现固定层预算的剪枝策略。

**💡 创新点**

创新点：1）在每个 Transformer 层插入轻量学习模块，融合视觉显著性与查询相关性；2）引入可学习阴影令牌捕获跨模态信息；3）发现层间冗余阶梯模式，提出 Budget‑Deterministic Pruning（BDP）并加入多样性选择；4）实现端到端联合训练的动态剪枝。

**🔧 技术方法**

技术手段：轻量可学习剪枝模块、shadow token、门控平滑、Straight‑Through Estimator、固定层预算比例、DART‑式多样性筛选、FlashAttention 兼容、LLaVA‑mix665k 伪标签训练、TFLOPs 与 KV 缓存评估。

**📊 数据集**

使用数据集：训练集 LLaVA‑mix665k；评估基准包括 9 个图像多模态任务（GQA、ScienceQA‑IMG、TextVQA、POPE、MME、VQAv2、MMBench、MMBench‑CN、MMVet）和 3 个视频 QA 任务（TGIF‑QA、MSVD‑QA、MSRVTT‑QA），以及多种 VLM 架构（LLaVA‑v1.5、LLaVA‑NeXT、Video‑LLaVA、Qwen2.5‑VL）。

**📈 对比分析**

与 FastV、SparseVLM、VisPruner、DART、CDPruner、LearnPruner 等方法对比，LearnPrunerV2 在保留 128/64/32 视觉令牌时，分别保持 99.8%/98.5%/95.7% 的原始性能；在 LLaVA‑NeXT‑7B 维持 99.7% 并压缩至 22.2% 令牌；在 Qwen2.5‑VL‑7B 维持 100%+ 相对精度，显著降低 TFLOPs（如 9.7× 加速，KV 缓存 8.7× 缩减）。

**⚠️ 局限性**

局限性：1）仍需冻结主模型权重，剪枝模块单独训练；2）BDP 固定预算牺牲了一定的输入适应性；3）在极低预算下可能无法捕获细粒度信息；4）对不同硬件环境的加速效果需进一步验证。

---

## 462. KL-MOPD: Discrepancy-Aware Multi-Objective On-Policy Distillation for Generative Recommendation

**arXiv ID:** 2610.36973 | [PDF](https://arxiv.org/pdf/2610.36973v1)

**作者:** Weiqin Yang `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究多目标生成式推荐（GR）中的多目标策略学习，提出一种基于初始学生-教师 KL 差异的 token‑级权重分配方法 KL‑MOPD，解决传统 MOPD 在 GR 上的 token 级不平衡和跨 token 不一致问题。

**💡 创新点**

创新点在于：①通过对初始学生与两位单目标教师的 token‑级 KL 分散量进行量化，动态生成权重；②在 on‑policy 蒸馏中直接替换均匀权重，实现 plug‑and‑play、计算高效；③给出对 additive joint 目标的严格理论收益下界。

**🔧 技术方法**

技术方法：多目标 on‑policy 蒸馏（MOPD）、KL‑权重动态调整、Transformer encoder‑decoder 生成式推荐模型、基础教师训练（CTR/CVR/GMV 等），对比实验采用 AUC、Recall@K 等指标。

**📊 数据集**

数据集：三大公开数据集（TencentGR、YooChoose、Taobao）和工业级 10 亿规模广告数据集（Industry）。

**📈 对比分析**

与 RCE‑KD、MTS^24MM、SmartGR、GR4AD、TSGR、vanilla MOPD、SMOPD、TrustMOPD 等基线相比，KL‑MOPD 在单目标平均提升 4.91%，在联合目标平均提升 4.12%。工业线上 A/B 测试中，收入 +1.63%，GMV +2.20%。

**⚠️ 局限性**

局限性：目前仅处理两位教师（两目标）；权重只基于初始学生，未考虑训练过程中的动态变化；对非 additive 的多目标组合缺乏理论分析；对更复杂的多任务场景（>2 目标）尚未验证。

---

## 463. AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

**arXiv ID:** 2610.36980 | [PDF](https://arxiv.org/pdf/2610.36980v1)

**作者:** Qiming Shi `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了 AutoVend 基准，模拟一名 LLM 代理在 20 台分散售货机网络中通过 Web 接口进行产品选择、定价、采购与现场补货，并评估其在 180 天内的长期经营表现。

**💡 创新点**

创新点在于：① 将共享仓储、交货期、付费现场执行等实际运营约束统一到一个可持续经济评估框架；② 提供了以真实售货机订单数据为基础的需求模型，能捕捉需求的时序性和异质性；③ 通过工具调用和持久记忆的 Model Context Protocol 让 LLM 直接与模拟环境交互，逼近真实的商业决策流程。

**🔧 技术方法**

使用了 POMDP 形式的任务建模、基于真实数据的需求生成器、工具调用接口（远程操作、物理执行、记忆工具），以及 GPT‑6.1、Claude、GLM‑5.3‑Flash 等 LLM 在 Pi harness 上的推理与 API 消耗评估。

**📊 数据集**

数据集来源于一家商用智能售货机公司的 6 个月订单记录（99 台机器、220,128 件商品），从中挑选 20 台、18 个子类、1,372 种商品来构建模拟网络，并对需求模型进行校准与验证。

**📈 对比分析**

评价方法：将每个 LLM 与规则基准（基于新闻订货模型的批量补货）在相同 180 天的模拟环境下进行三次独立跑，比较终端净资产、废料率、定价回报、货架利用率、工具调用次数和 API 效率。结果显示四个 LLM 的平均净资产低于规则基准，只有 GPT‑6.1 Sol 在三次跑中全部超越基准，但其表现波动大；其他模型均表现不如基准，且在资源协调、现场执行成本控制等方面存在显著缺陷。

**⚠️ 局限性**

局限性包括：① 仅在单一环境与单一随机种子下评估，缺乏跨场景的鲁棒性验证；② 现场执行仅以付费访问计费，未模拟真实人员调度或复杂物流；③ LLM 通过工具调用实现决策，但未对模型内部学习机制进行分析；④ 评价指标主要关注经济回报，缺少对决策透明度、可解释性和安全性的评估。

---

## 464. Explicit Regional Representations For Video Quality Assessment

**arXiv ID:** 2610.36978 | [PDF](https://arxiv.org/pdf/2610.36978v1)

**作者:** Fuyi Yang `[一作]` (University of California, Los Angeles), Ming Kai Hsu `[通讯]` (Transtreams Technologies, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一个基于质量相关热图（QRH）的显式像素级区域表示，用于视频质量评估。

**💡 创新点**

创新点在于将质量损伤证据与感知优先级分离，并在无像素级标注的情况下通过视觉语言模型进行语义引导，随后实现分布保持与质量感知两种确定性采样策略。

**🔧 技术方法**

采用多种局部质量线索（模糊、时间不连贯、压缩伪影）、运动敏感度、视觉显著性及VLM（Qwen3.6‑35B‑A3B）权重，使用软OR融合生成QRH，并在DOVER模型上实现DPS和QAS采样。

**📊 数据集**

在LSVQ-test、LSVQ-1080p、KoNViD-1k、LIVE-VQC以及YouTube-UGC等公开视频质量数据集上进行评估。

**📈 对比分析**

与原始DOVER基线相比，分布保持采样在不重新训练的情况下逼近10次随机采样的平均预测；质量感知采样加微调后在8大基准上提升了7项指标（最高提升约2.7% PLCC），并在KoNViD-1k和LIVE-VQC上取得最优或接近最优表现。

**⚠️ 局限性**

局限性包括对单一VQA骨干（DOVER）的依赖、无法直接学习区域表示、VLM语义指导的准确性受限，以及未在完整YouTube-UGC数据上进行全面评估。

---

## 465. Proactive Memory for Event-Driven LLM Agents

**arXiv ID:** 2610.36967 | [PDF](https://arxiv.org/pdf/2610.36967v1)

**作者:** Jinyang Li `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了ProactiveStream纵向事件流基准，并提出Proactive Memory层次化记忆框架，用于在大型语言模型上实现主动助理的长期记忆与决策。

**💡 创新点**

创新点在于：① 将源验证的时间轨迹记忆与历史重放评估的行为规律记忆结合，使得模型能够将过去的观察可靠地映射到当前事件；② 构建跨领域（体育、智能家居、代码协作）的长序列基准，兼顾因果性与可解释性。

**🔧 技术方法**

技术方法包括：大语言模型（GPT‑5.6 系列）用于事件摘要、轨迹生成、规律提取与决策；事件缓冲、事件/轨迹/规律三层记忆结构；干预控制策略与源验证/重放验证机制。

**📊 数据集**

使用数据集：HeartSteps 运动跟踪、CASAS 家居传感器、GitHub 仓库事件，共计17个实体、4539条事件，涵盖 Sports、Home、Code 三个领域。

**📈 对比分析**

与五个现有主动助理基线（ProactiveAgent、ContextAgent、Mem0、A‑MEM、Graphiti）在 ProactiveStream 上进行严格对比，Proactive Memory 在 Sports、Home、Code 三域均获得最高 F1/效能，整体提升约 21%（相对最强基线），并在误干预率与覆盖率上取得平衡。

**⚠️ 局限性**

局限性：仅评估低风险主动服务场景；离线评估无法覆盖实时部署与安全性验证；标签依赖人工协议，可能存在标注偏差；跨域迁移与大规模真实部署的泛化能力尚待进一步验证。

---

## 466. Do bots feel bad? Assessing empirical evidence for functional valence in LLMs

**arXiv ID:** 2610.36962 | [PDF](https://arxiv.org/pdf/2610.36962v1)

**作者:** Iwan Williams `[一作]` `[通讯]` (University of Copenhagen), Iwan Williams (University of Copenhagen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了大语言模型（LLM）中是否存在功能性情感状态（即情感价值），通过梳理三项近期实证研究（Sofroniew等、Ren等、Han等）并结合哲学文献对情感价值的功能角色提出薄层与厚层评估标准，最终得出目前证据不足以确认为LLM具有功能性情感状态。

**💡 创新点**

创新点在于将哲学层面的情感价值功能角色（动机性、共同货币、主体性、认知偏差等）与机器学习解释方法对应，提出一套系统化的评估准则；并指出现有实验设计缺失的关键测量维度，进一步建议未来研究可采用的实证路径。

**🔧 技术方法**

主要使用的技术包括：激活向量解析（情感向量、福利轴）、主成分分析、线性探针（预测内部状态与行为的线性映射）、Thurstonian 模型拟合、以及模型干预（向量 steering）等解释性与因果性分析方法。

**📊 数据集**

利用的核心数据集为：171种情感故事集合（Sofroniew等）；Ren等的对话记录（涵盖约300类对话场景、图像与音频）；Han等的迷宫任务与 emoji 生成任务；以及公开的标准基准（如 MMLU、GSM8K）用于评估福利轴的跨任务通用性。

**📈 对比分析**

对比方法主要是检验内部向量与行为的一致性：对 Ren 等的情感偏好报告与线性探针预测的相关性（约 0.47–0.84）以及对 Sofroniew 等情感向量在不同情境下的激活分布；对 Han 等的福利轴在迷宫与其他任务中的 Steering 效果。然而，这些指标仍无法确定动机性或共同货币的完整功能性，性能表现仅表明存在相关性而非因果证据。

**⚠️ 局限性**

限制主要包括：缺乏对动机性（more‑of‑this/less‑of‑this）与经验导向行为的系统测试；未检验福利轴是否真正充当决策或学习阶段的共同货币；缺乏主体性（mineness）的验证；未探查情感价值的认知偏差效应；存在角色扮演或偏好等替代理论的解释空间；并且实验多为后训练阶段，缺少跨模型与跨任务的稳健性验证。

---

## 467. Keeping Contact: Task-Adaptive Friction Margins for Bimanual Non-Prehensile Manipulation

**arXiv ID:** 2610.36961 | [PDF](https://arxiv.org/pdf/2610.36961v1)

**作者:** Qian Ying Lee `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

规划双手非抓取倾斜任务的接触力，利用带任务自适应摩擦距离边际的凸优化实现稳健执行。

**💡 创新点**

创新点在于引入签名摩擦距离作为力空间边际，并设计任务自适应的边际函数，随后将其嵌入第二阶锥规划中，首次将边际直接约束到接触力规划里。

**🔧 技术方法**

使用第二阶锥规划（SOCP）、符号摩擦距离、任务自适应边际构造、阻尼控制实现执行以及对力/位置的实时估计。

**📊 数据集**

使用Kinova Gen3双臂实验数据集，包含不同质量、重心偏移和摩擦系数的盒子物体，通过实验记录力曲线、成功率和违规率。

**📈 对比分析**

通过与无边际、固定边际以及模型匹配/不匹配的对照实验比较，结果显示任务自适应边际在成功率、摩擦利用率和违规率方面均优于其他方案，尤其在模型不匹配时显著提高稳健性。

**⚠️ 局限性**

局限性包括：只针对固定接触模式的平面倾斜任务验证；边际参数需人工预设，缺乏自适应更新；对环境反作用的估计采用启发式方法，可能在更复杂的环境中失效。

---

## 468. PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

**arXiv ID:** 2610.36958 | [PDF](https://arxiv.org/pdf/2610.36958v1)

**作者:** Xiaohan Ye `[一作]` (Taobao & Tmall Group of Alibaba), Bo Zheng `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了统一的多模态生成检索系统 PailitaoMMSearch2.0，直接从原始图像和自由文本指令生成商品 Semantic ID，实现无意图分类的开放式检索，并在淘宝派来淘应用中上线。

**💡 创新点**

创新点包括：① 通过三维扩展框架（数据扩展、意图扩展、场景扩展）消除检索模型与场景、意图、资源绑定；② 设计可组合的数据工具集，实现冷启动、真实流量优化和持续生产；③ 采用多意图混合预算监督微调 + 统一强化学习，配合 Sample–Beam 采样-束搜索对齐推理与检索；④ 通过可调节的推理预算和模型容量实现多运营点，无需额外意图路由。

**🔧 技术方法**

技术手段包括：本地 VLM（Qwen3.5）原生图文处理；持续预训练（CPT）适配电商域；多意图混合预算监督微调（SFT）；统一强化学习（RL）结合 Sample–Beam 采样；基于自定义评估器的质量控制与奖励模型；动态束宽与推理预算控制；Semantic ID 的序列生成与解码。

**📊 数据集**

数据集：利用代理商品关系生成的冷启动数据（≈30M 条，包含相似、编辑、匹配三类）；真实流量交互数据（≈1M–8M 条）；自动评测集（502/772/371 条），以及独立 40k+ 样本的人类校准评估集。全部数据经过可组合工具集进行过滤、修复、链路生成与质量控制。

**📈 对比分析**

对比方法：Qwen3.8-Flash-Next、GPT6-luna、内部召回-排序管线。实验表明 2B‑Low‑think 版本在宏观 NDCG 0.918、宏观 Excellent Rate 55.1% 的同时，比基线低 74% 推理时延、使用 1 GPU，且在三类检索中匹配检索提升 30.1 分。离线 ablation 显示数据扩展、CPT、跨意图转移、RL 与 Sample–Beam 的累积收益。在线 A/B 测试显示+17.90% 商品页视图、+27.04% 交易额、+8.94% GMV 等显著业务提升。

**⚠️ 局限性**

局限性：① 仅覆盖三种电商检索关系（相似、编辑、匹配），未实现全域关系；② 评测仅在单一电商场景；③ 3 体量模型为独立检查点，未深入探究弹性权重共享；④ 自动评测器与奖励模型可能与人类评估存在相关误差；⑤ 在线 A/B 结果无法单独归因到各组件；⑥ “开放式”仅指查询接口，未完全实现任意关系检索。

---

## 469. Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance

**arXiv ID:** 2610.36947 | [PDF](https://arxiv.org/pdf/2610.36947v1)

**作者:** S M Abdulla Al Mamun `[一作]` (University of Manchester), Mohd. Ruhul Ameen `[通讯]` (Varendra University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过五项实验评估在已有确定性工具证据下，单轮LLM专家上下文是否能提升软件保障，重点探讨上下文规模、验证器反馈与证据适用性；

**💡 创新点**

创新点在于将多上下文评估与后处理门控相结合，系统检验LLM在已知证据情境下的价值，并揭示模型对误导性证据的误判；

**🔧 技术方法**

使用Qwen2.5-Coder-7B-Instruct与1.5B变体，结合SpotBugs/FindSecBugs、JBMC、SV-COMP、Defects4J等工具的输出；

**📊 数据集**

数据集包括Juliet Java 1.3、OWASP Benchmark 1.2、Defects4J、SV-COMP Java、以及自定义的E5证据包；

**📈 对比分析**

与仅使用确定性门控或验证器回放的基线比较，结果显示多上下文并未显著降低误判，验证器反馈提升了正确率但同时增加错误；

**⚠️ 局限性**

局限性包括模型数量有限、实验仅覆盖Java环境、缺乏显著性检验、对多上下文的规模与多样性未充分探究，以及门控仅在已知证据时有效。

---

## 470. DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding

**arXiv ID:** 2610.36929 | [PDF](https://arxiv.org/pdf/2610.36929v1)

**作者:** Saifullah Saifullah `[一作]` (German Research Center for Artificial Intelligence), Sheraz Ahmed `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并系统评估多模态文档理解的可解释性框架（DocXEval），实现多模态、多目标XAI方法与评估指标的统一适配与应用。

**💡 创新点**

首次将13种XAI方法与17种评估指标迁移至多模态文档理解，提出高效多目标适配技术，并通过大规模基准揭示哪些方法最可信、最简洁、最实用。

**🔧 技术方法**

采用梯度、扰动、注意力等三大类XAI方法；利用模态分割、基线选择、归一化与多目标梯度累计等技术；结合TorchXAI库与PyTorch实现。

**📊 数据集**

8个公开数据集（如DocBank、FUNSD、CORD、RVQA、DocVQA、DocInfo等），覆盖文本分类、实体识别、视觉问答三大任务；使用4个Transformer模型（BERT、LayoutLMv2、LiLT、DocFormer）。

**📈 对比分析**

通过17项评估指标（包括公理性、可信度、复杂度、鲁棒性）进行全量基准，结果显示Occlusion（扰动法）在大多数指标上稳居首位，注意力方法（VisualBERT）和梯度方法（Grad-CAM）次之；同时指出Monotonicity、Infidelity指标不可靠。

**⚠️ 局限性**

实验仅覆盖四种Transformer模型，缺乏多模态非Transformer架构；方法对模型内部实现的依赖较大，且高维模态空间导致部分完整性指标计算成本高；跨任务对比仍受限于数据集与模型的多样性不足。

---

## 471. Vision-Language Agents Can Learn in Their Own Latent Worlds

**arXiv ID:** 2610.36925 | [PDF](https://arxiv.org/pdf/2610.36925v1)

**作者:** Haoxi Li `[一作]` (HKUST), Song Guo `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dwell，一种基于预训练视觉‑语言模型（VLM）的模型驱动强化学习框架，能够在 VLM 的潜在空间中进行在线自我更新与隐式想象，从而在有限环境交互下提升策略性能。

**💡 创新点**

创新点：① 将 VLM 重构为递归潜在转移算子，让同一 VLM 既做世界模型又做策略；② 通过轻量 LoRA 适配器实现世界模型与策略的参数高效分离；③ 在潜在空间中直接进行规划与奖励预测，避免昂贵的观测重建；④ 实现在线循环学习，持续用真实经验校正世界模型并利用想象轨迹驱动策略改进。

**🔧 技术方法**

技术：预训练 VLM（如 Qwen2.5-VL-3B）作为冻结主干；LoRA adapter 进行可训练的世界模型（动态、奖励）与策略；潜在槽位表示与 V-JEPA2 编码器对齐；基于 PPO 的离线策略更新；在线交互与潜在规划交替进行；使用 n‑step 返回预热 critic。

**📊 数据集**

数据集/环境：Sokoban、FrozenLake、Navigation（第一人称导航）、PrimitiveSkill（视觉操控）以及 EB-ALFRED（长时序嵌套任务）——共五个交互式视觉环境。

**📈 对比分析**

与方法比较：模型无关方法（PPO、GRPO）、模型基方法（LeWorldModel、AdaWorld）以及公开 VLM 直接使用。Dwell 在五个任务中均为可训练方法中最优，尤其在 Sokoban、FrozenLake 与 EB‑ALFRED 上表现突出；在 Navigation 与 PrimitiveSkill 上略低于 GRPO，说明对观察驱动任务的增益有限。计算效率方面，Dwell 在潜在规划上显著低于基于生成观察的 AdaWorld，整体 GPU‑小时成本下降 38‑58%。

**⚠️ 局限性**

局限性：① 并非所有任务均受益，观测驱动或细粒度动力学任务对潜在想象的帮助有限；② 潜在表示的抽象化可能丢失关键信息，尤其是接触/细节丰富的场景；③ 缺乏不确定性估计，导致长时 horizon 的规划误差累积；④ 仅在单一 VLM 主干和模拟环境上验证，缺乏对更大模型、真实世界交互和跨任务迁移的评估。

---

## 472. OFDM-Based Neuromorphic Wake-Up Signal

**arXiv ID:** 2610.36914 | [PDF](https://arxiv.org/pdf/2610.36914v1)

**作者:** Gaspard Goupy `[一作]` (Nokia Bell Labs), Dani Korpi `[通讯]` (Nokia Bell Labs)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种可由标准OFDM发射机生成的稀疏峰值wake‑up信号nWUS，并给出相应的非相干神经形态检测器。

**💡 创新点**

创新点在于将OFDM子载波相位线性化以产生时间域稀疏峰值，利用这些峰值作为神经形态“spike”输入，实现低功耗事件驱动检测。

**🔧 技术方法**

使用OFDM调制、线性相位进位、IFFT、阈值水平穿越编码、加权模板稀疏相关等技术。

**📊 数据集**

在3GPP TR 38.901 urban macro通道模型下进行仿真，平均发射功率相同，比较了两种OOK基准。

**📈 对比分析**

相较于同等信息容量或更长时延的OOK，nWUS在相同平均功率下检测概率提升18.6pp，计算量下降2.5–43.5倍，噪声下检测同样优于OOK。

**⚠️ 局限性**

局限包括对功率放大器峰值的影响未评估，阈值需预先校准，且只在单输入单输出下验证。

---

## 473. Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning

**arXiv ID:** 2610.36888 | [PDF](https://arxiv.org/pdf/2610.36888v1)

**作者:** Martín de Frutos `[一作]` (Universidad Politécnica de Madrid), Esteban Ferrer `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于风机偏航角和功率设定的噪声引导（Noise Steering）策略，利用风机间的尾流相互作用和噪声定向特性，实现风电场在满足局部噪声约束的同时最大化发电量。

**💡 创新点**

创新点在于将尾流偏航控制与噪声定向耦合，首次通过强化学习同时调节偏航角和功率，实现噪声约束下的功率最优化；并提出低保真声学模型与RL训练的高效耦合方法。

**🔧 技术方法**

使用近端策略优化（PPO）强化学习算法，配合低保真声学模型（基于BPM的分析模型+偶极定向）和稳态仿真器，后续在高保真 FAST.Farm 时序仿真器上验证。

**📊 数据集**

训练数据来源于内部仿真：300k步的稳态仿真样本，随机风速8–15 m/s；校准使用16个不同风速、偏航组合的仿真结果；验证采用同一布局的高保真仿真和随机湍流风场。

**📈 对比分析**

与贪婪控制和行业标准噪声减排模式（NRO）对比，Noise Steering 在 8–15 m/s 范围内平均功率损失仅 2%，而 NRO 为 17%；在高保真仿真和湍流风场中，Noise Steering 能保持噪声阈值 91.7% 时间，功率损失仅 7.4%，显著优于 NRO。

**⚠️ 局限性**

局限性包括：训练仅在单一风向和稳态风场下完成，导致在高保真仿真中表现略显保守；声学模型忽略地面反射、折射等传播效应；模型匹配误差导致迁移性能下降；未来需在线学习、扩展风向、改进声学模型。

---

## 474. Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models

**arXiv ID:** 2610.36948 | [PDF](https://arxiv.org/pdf/2610.36948v1)

**作者:** XinYu Piao `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种自键入参数记忆方法Skim，能够在冻结的语言模型中一次性闭式编辑，将对话回合直接存入模型参数，并在没有外部存储的情况下回答后续问题。

**💡 创新点**

创新点在于模型无需预先给定答案或问题，自动从每个回合提取关键跨度并生成多样化的自问作为键，以闭式权重编辑方式更新单个前馈层，实现内在对话记忆。

**🔧 技术方法**

采用Transformer前馈层的闭式权重编辑（ridge回归）技术，结合自监督的跨度提取、关键问题生成、残差调节梯度优化以及自探测配置调优。

**📊 数据集**

使用LongMemEval对话回忆基准，针对单会话用户子集的33个非数值事实（在N=32时几乎全部存储）。

**📈 对比分析**

与KV缓存、无缓存、True‑Q、Self‑Q等条件对照，Skim在五个模型上平均记忆率为63.1%，最强模型达到75%，显著优于无缓存基线。

**⚠️ 局限性**

主要限制在于存储阶段的跨度提取与选择，尤其对描述性多词答案的成功率低；键的多样性和数字答案的可恢复性也有限。

---

## 475. Too many clients spoil the logs: distributed parsing, aggregation techniques and metrics

**arXiv ID:** 2610.36946 | [PDF](https://arxiv.org/pdf/2610.36946v1)

**作者:** André García Gómez `[一作]` (Austrian Institute of Technology), Edgar Weipp `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在6720种分布式配置下实验，系统评估了多种日志解析器在分布式环境中的表现，并探讨了聚合技术与评估指标。

**💡 创新点**

创新点在于提出了分布式聚合方法与新的指标分类（包括TS、RT、FR、PS等），以及将LLM基准与传统解析器一起在分布式场景下进行全面对比。

**🔧 技术方法**

使用了频率、相似、启发式、LLM（LogBatcher）等解析器，聚合策略包括串接、解析缓存、FedAvg式平均和Paxos去中心化；指标涵盖GA、PA、NED、TS、RT、FR及综合PS。

**📊 数据集**

实验数据集为工业日志数据集HDFS和BGL。

**📈 对比分析**

通过将本地训练与聚合后的分布式训练进行对比，利用多维指标衡量性能；结果显示聚合能提升整体表现，LLM解析器对数据分布更敏感，而BGL数据集受益更显著。

**⚠️ 局限性**

局限性包括：真值模板可能存在噪声；LLM预训练可能对实验数据产生影响；实验仅在IID分布下进行；仅覆盖两大数据集，未覆盖更复杂的通信和隐私场景。

---

## 476. Developing a Speech Language Model for Basque

**arXiv ID:** 2610.36911 | [PDF](https://arxiv.org/pdf/2610.36911v1)

**作者:** Andoni Sudupe `[一作]` (University of Basque Country UPV/EHU), Eva Navas `[通讯]` (University of Basque Country UPV/EHU)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了首个开放源代码的完整端到端巴斯克语 Spoken Language Model，基于模块化 LLaMA‑Omni 架构实现语音输入输出；

**💡 创新点**

创新点在于通过文本 LLM 的适配器与并行流式语音解码器快速迁移语言能力，并采用自动翻译+TTS 合成的无监督流程构建低资源语言多模态数据集；

**🔧 技术方法**

使用了 LLaMA‑Omni、Latxa 3.1‑8B Instruct、Whisper 语音编码器、HuBERT 离散单元提取器、Unit HiFi‑GAN vocoder、CTC 训练解码器及非自回归流式解码技术；

**📊 数据集**

利用 VoiceAssistant‑400K 翻译为巴斯克语并用 TTS 合成约 450k 语音指令对，Basque TTS 2 位说话人数据集（≈16h），Composite‑eu 600h 巴斯克语音用于 HuBERT 训练；

**📈 对比分析**

通过 LLM‑as‑a‑judge、UTMOS、ASR WER、CTC 等评估，巴斯克模型在内容质量 3.86/5、语音 WER 9.94%、UTMOS 3.07、流式延迟 370 ms，性能接近英语言 LLaMA‑Omni；

**⚠️ 局限性**

主要限制在评测全为合成语音、缺乏真实语料测试，评判模型同属 Latxa 系列可能存在自偏，以及 TTS 音质仍落后于英语言基线。

---

## 477. A lightweight C Code Generator for Heterogeneous TinyML Systems

**arXiv ID:** 2610.36907 | [PDF](https://arxiv.org/pdf/2610.36907v1)

**作者:** Chen Liu `[一作]` (Dresden University of Technology), Christian Mayr `[通讯]` (Dresden University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种轻量级 C 代码生成器，用于在裸金属异构 TinyML 系统上部署深度学习模型，能够自动完成前向推理的代码生成、权重调优和偏置融合；

**💡 创新点**

创新点在于：①采用权重调优和偏置融合两种 AOT 优化，减少 CPU 级别的浮点重缩放与偏置加法；②基于模板的后端集成方式，生成可读性强且易于手工修改的纯 C 源码；③通过在编译时完成尺寸重排和权重量化，显著减小二进制体积并提升推理速度；

**🔧 技术方法**

主要技术包括：TFLite 模型解析、IR 生成与优化、权重调优、偏置融合、模板化 C 代码生成、Ping-Pong 缓冲区管理以及与 RISC‑V+MLA（多乘累加器）加速器的交互；

**📊 数据集**

使用的数据集为毫米波雷达图像的四分类数据集（车辆、骑行者、行人、噪声），并在此数据集上训练了 RadarNet（9,704 参数，92% 准确率）的 CNN 模型；

**📈 对比分析**

在同一自定义雷达 DSP 上与 microTVM 基线以及 microTVM+UMA 的 TVM 流程进行对比；结果显示：CodeGen 的单次推理时间为 630 µs，二进制大小 25 kB，分别比 microTVM 的 4,571 µs 与 39.3 kB、microTVM+UMA 的 1,470 µs 与 31.2 kB 低约 86% 与 84% 的推理时间，二进制体积下降约 36% 与 20%；

**⚠️ 局限性**

局限性包括：目前仅支持 TFLite（ONNX 仍在开发）、CNN 常用算子（卷积、全连接、池化、Softmax），需要手工编写后端模板，且对新的自定义 DSAs 的支持需要额外实现；此外，权重调优与偏置融合虽提升性能但可能导致微小精度误差。

---

## 478. PREEN: Exemplar-Based Color Style Transfer with Smartphone Editor Presets

**arXiv ID:** 2610.36896 | [PDF](https://arxiv.org/pdf/2610.36896v1)

**作者:** Beomjin Ahn `[一作]` `[通讯]` (Samsung Research), Beomjin Ahn (Samsung Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 PREEN，直接从内容图像与风格参考预测 29 维照片编辑器预设，实现可解释、可重用的颜色风格迁移。

**💡 创新点**

创新点包括：利用 SA‑LUT 作为教师网络生成伪预设并回归至低维预设；通过 CLIP+LoRA 进行教师监督与图像级损失细化；提供单一强度 α 旋钮实现无重训练的色彩强度控制。

**🔧 技术方法**

采用的技术包括：CLIP 条件残差 MLP 回归器、低秩适配器 LoRA、CIELAB 图像级重建损失、教师网络逆向映射与 alpha 强度插值。

**📊 数据集**

训练数据来自 OpenImages 与 AVA 视觉相似对，评估使用人类标注的 PST50（50 例）以及 DPST 无配对样本。

**📈 对比分析**

与 raster、LUT 及 Deep Preset 等基线对比，PREEN 在 PST50 上在 PSNR、SSIM、LPIPS 上领先 Deep Preset 并接近 LUT，且在 CPU 上实现子秒推理；在人类研究中，Vivid 在风格匹配上胜过对手，默认版在美学上更受欢迎。

**⚠️ 局限性**

局限性包括：只能表达全局 29 维预设，无法处理局部或颜色重映射效果；依赖教师网络，编辑子空间有限；需要语义兼容的参考图像。

---

## 479. On the VC Dimension of Autoregressive Linear Threshold Functions

**arXiv ID:** 2610.36970 | [PDF](https://arxiv.org/pdf/2610.36970v1)

**作者:** Ilan Doron-Arad `[一作]` (Massachusetts Institute of Technology), Elchanan Mossel `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文研究了自回归线性阈值函数（LTF）的 VC 维度，证明在足够长的迭代步数 T≥d² 时其 VC 维度为 Θ(d²)，并进一步展示在随机采样版本中该维度可提升到指数级 2^{Ω(d)}，从而揭示了确定性与随机自回归模型在统计复杂度上的显著分离。

**💡 创新点**

核心创新在于：① 构造一种递归式的多 LTF 计算来实现对 d×d 位标签矩阵的全局编码；② 将该多 LTF 递归压缩为单个 LTF，通过巧妙的位块和权重编码实现；③ 在随机采样模型中引入基于二进制展开的概率采样，利用概率正弦/余弦特性实现指数级标签编码。

**🔧 技术方法**

主要技术手段包括：递归式 LTF 设计、位块编码与模运算、线性阈值函数与逻辑回归门的权重构造、概率分析（利用二项分布与余弦界限）、以及对错误概率的精细控制。

**📊 数据集**

该工作为纯理论分析，不涉及实测数据集。

**📈 对比分析**

由于研究对象为理论函数类，未进行实验对比；但作者通过构造证明展示了在样本复杂度方面确定性模型需要 Θ(d²) 样本，而随机模型只需 2^{Ω(d)} 的可学习能力，从而体现了两者在统计学习上的根本差距。

**⚠️ 局限性**

局限性在于：① 结果仅适用于理想化的自回归 LTF 及其随机化版本；② 对于更一般的非线性生成器或噪声模型，结论尚未扩展；③ 证明中的常数与复杂度上界仍相对粗糙，缺乏更紧的上界。

---

## 480. Live-OPSD: On-Policy Self-Distillation for Proactive Streaming Large Video Models

**arXiv ID:** 2610.36959 | [PDF](https://arxiv.org/pdf/2610.36959v1)

**作者:** Kaixuan Du `[一作]` (Qwen Business Unit of Alibaba), Guanjun Jiang `[通讯]` (Qwen Business Unit of Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种名为Live-OPSD的流式多模态助手自蒸馏方法，利用训练时提供的特权提示来监督自身生成的响应，从而提升响应时机与质量。

**💡 创新点**

创新点在于将自我蒸馏与特权上下文结合，并引入决策感知选择（DAS）在训练中动态挑选重要决策点；同时采用逆KL分布匹配以实现对自身生成的连续体进行监督。

**🔧 技术方法**

使用了on-policy自蒸馏、特权上下文提示、逆KL损失、决策感知选择（DAS）、流式SFT、Qwen系列模型以及离线提示生成等技术。

**📊 数据集**

训练数据基于包含直接提问和持续指令的多模态对话，扩展为因果决策上下文；评估使用LiveProBench、OVO-Bench、OmniMMI等公开基准。

**📈 对比分析**

通过与Vanilla‑SFT、CB‑SFT、SSB‑SFT等SFT基线以及公开主动模型（AURA、ROMA等）进行对比，Live-OPSD在LiveProBench平均得分59.64，较最佳SFT提升5.26点；在OVO‑Bench上实现65.84的整体准确率，并在OmniMMI Proactive Alerting上显著优于对照模型。

**⚠️ 局限性**

局限性包括需预先标注的特权提示和对齐的决策监督；训练使用离线历史与部署历史不一致；目前不支持完全在线自训练，依赖外部模型生成提示；在更复杂的动作空间或多轮自评任务中需额外标注与设计。

---

## 481. Does More Thinking Pay Off? Reasoning Cost and Correctness of LLMs in Code Generation and Bug Fixing

**arXiv ID:** 2610.36952 | [PDF](https://arxiv.org/pdf/2610.36952v1)

**作者:** Andie Huang `[一作]` (Massey University), Lin Ma `[通讯]` (Hangzhou Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了大型语言模型在 Python 代码生成与程序修复任务中使用显式推理时，是否能提升准确率以及产生的思考代价。

**💡 创新点**

创新点在于：①揭示推理长度并不一定带来准确率提升；②通过离线截断实验验证仅保留推理前半段即可保持几乎相同的性能；③提出公平的截断基准与评估方法。

**🔧 技术方法**

使用 Qwen3‑32B、QwQ‑32B、R1‑Distill‑Qwen‑32B、R1‑Distill‑Llama‑70B 等模型的推理与非推理两种设置，配合 Token Count、McNemar 检验、行为编码与离线截断策略等技术。

**📊 数据集**

实验数据集为 HumanEval Pro、MBPP Pro（代码生成）和 BugsInPy（程序修复），共 866 个任务，其中 164、374 与 328 个有效任务被评估。

**📈 对比分析**

通过五轮重复生成，比较开启/关闭推理对 pass@1 / tests‑pass 的差异，利用 McNemar 检验评估统计显著性；结果显示推理提升率低且不稳定，平均思考 token 超千；截断实验显示保留 50% 推理即可无显著性能下降。

**⚠️ 局限性**

局限性包括仅针对 Python、开源 70B 规模模型、量化推理、低 BugsInPy 准确率，以及仅离线评估推理截断；结果可能不适用于其他语言、更大模型或在线推理环境。

---

## 482. Hierarchical Low-Rank Reparameterization for Parameter-Efficient

**arXiv ID:** 2610.37068 | [PDF](https://arxiv.org/pdf/2610.37068v1)

**作者:** Rong Huang `[一作]` (University of Science and Technology of China), Xiao-Ming Fu `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HiLoRR——一种层次低秩重参数化框架，用来进一步压缩现有 PEFT 方法（LoRA、DoRA、MoSA、RaSA 等）产生的可训练参数，并在 LLaMA‑3.1‑8B 与 Qwen2.5‑7B 上进行实验。

**💡 创新点**

创新点在于：①将 PEFT 的可训练参数重新组织为近似正方形矩阵；②对该矩阵递归做低秩分解，形成多层压缩结构；③该重参数化不依赖于特定 PEFT 设计，可通用应用于多种适配器，实现参数量级压缩同时保持或提升性能。

**🔧 技术方法**

使用的技术包括：近似正方形 reshape、层次低秩分解（HiLoRR）、递归重参数化、学习率与初始化的调优、梯度分析，实验中还对单层/双层深度、不同重塑策略、外部秩 r 与内部秩 k 的组合做了 ablation。

**📊 数据集**

使用数据集：Commonsense‑15K 指令微调后评估 BoolQ、PIQA、SIQA、ARC‑c/e、OBQA、HellaSwag、WinoGrande；数学推理任务使用 MetaMathQA‑40K 微调后在 GSM8K 上评估。

**📈 对比分析**

与 LoRA、DoRA、MoRA、NoRA、SVFit 等基线在相同可训练参数预算下比较。HiLoRA/HiDoRA 在低参数设置（≤6M）下仅使用 LoRA‑r16 参数约10% 量级，却实现更高的 Avg./W‑Avg.；在高参数设置（≥26M）下性能相当或略优；压缩后仍保持或超过基线，显著降低训练内存。

**⚠️ 局限性**

限制：对学习率、初始化、层次深度、重塑策略敏感；过深层次在高预算下可能导致性能下降；目前仅在 Transformer 文本模型上验证，未覆盖更大模型或跨模态任务；需要更自动化的秩分配与优化策略。

---

## 483. A hyper-reduction strategy for elastoplastic hydro-mechanical problems in partially saturated soil

**arXiv ID:** 2610.37060 | [PDF](https://arxiv.org/pdf/2610.37060v1)

**作者:** Alba Muixí `[一作]` (Universitat Politècnica de Catalunya), Sergio Zlotnik `[通讯]` (Centre Internacional de Mètodes Numèrics en Enginyeria)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对部分饱和土壤中耦合弹塑性水力-力学问题的超低秩模型（hyper‑reduced order model, HR‑RB）。该模型使用场分离的POD基和能量保持采样加权（ECSW）规则，对残差的非线性修正进行离散化，从而在保持高精度的同时显著降低在线求解成本。

**💡 创新点**

创新点主要有：①在耦合弹塑性水力-力学问题中首次引入场分离的ECSW超低秩策略；②将残差分解为可精确计算的仿射基线与局部非线性修正，极大提高了基元选择的局部性与压缩效果；③该构造完全独立于具体的塑性本构或渗流模型，易于推广；④通过在FEniCSx中仅使用标准组装流程实现，兼容高层次求解器。

**🔧 技术方法**

技术包括：Proper Orthogonal Decomposition（POD）场分离基、Galerkin 投影、能量保持采样加权（ECSW）超低秩积分、贪婪稀疏非负最小二乘法（NNLS）选取规则、后向欧拉时间离散、Newton‑Raphson 非线性求解。

**📊 数据集**

使用的“数据集”为两个数值实验：一个二维平面应变挡土坝模型（3种网格粗细，共计约5×10⁴单元），一个三维全尺寸挡土坝模型（约1.18×10⁵单元）。训练集采用拉丁超立方采样（40个参数点），每点保存313个时间步的位移和孔隙压强快照；测试集为10个独立参数点。

**📈 对比分析**

比较方法：将HR‑RB与仅做POD的RB模型对比，评估误差（全局相对误差≈10⁻³，峰值误差≤1%）以及在线运行时间。结果显示：二维模型在最细网格上实现43×加速，三维模型实现约77×加速；加速随网格细化与塑性区域扩展而显著提升。误差基本受基底截断控制，超低秩积分不产生额外误差。

**⚠️ 局限性**

限制：①需要离线训练和完整的高阶求解以获取快照；②对非线性强度和局部性的依赖较大，若本构或渗流模型极端非线性或需要局部迭代，选取规则的效果可能下降；③目前仅在FEniCSx实现，迁移到其他有限元框架需要重新实现组装约束；④对多物理耦合（如热-力学）尚未验证。

---

## 484. Learning to Code the World: Harness Evolution for Compositional 3D Reconstruction

**arXiv ID:** 2610.37046 | [PDF](https://arxiv.org/pdf/2610.37046v1)

**作者:** Haoyu Yao `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SceneEvolve框架，利用可演化的运行时环境将单张图像转换为可执行代码，从而构建可编辑、可组合的3D场景。

**💡 创新点**

创新点在于：①设计了基于参数化几何原语、层级组合和语义空间关系的SceneAct语义接口；②通过教师指导的递归运行时演化（Teacher‑Guided Harness Evolution）在不更新模型参数的情况下优化重建程序。

**🔧 技术方法**

技术包括：使用开源大型语言模型Qwen3.8‑27B作为学生模型；SceneAct的可执行代码生成和Blender后端；教师模型（如GPT‑6‑Astra）提供反馈以迭代优化运行时；以及基于SSIM/LPIPS/DINOv2等指标的开发集评估。

**📊 数据集**

数据集：ScanNet、CA‑1M用于内部训练与测试；ARKitScenes用于跨域外部测试。

**📈 对比分析**

与基线（同模型无演化）、GPT‑5.6‑Sol、GPT‑6‑Astra、Claude‑Opus‑5、Gemini‑Flash‑3.8以及Kimi‑K3对比，SceneEvolve在四项重建质量指标（SSIM、LPIPS、PSNR、DINOv2‑S）均优于基线，并在ScanNet/CA‑1M/ARKitScenes上平均提升DINOv2‑S 32.3%、降低LPIPS 6.1%、提升PSNR 0.118 dB；推理时间比基线缩短47.0%，等价部署成本下降59.8%；在跨模型迁移到Kimi‑K3时表现与GPT‑6‑Astra相当。

**⚠️ 局限性**

局限性：仍存在颜色失真、视角偏差和部分几何细节欠缺；运行时演化依赖开发集与教师反馈，需额外人工或模型成本；对动态或复杂场景的鲁棒性尚未充分验证。

---

## 485. Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception

**arXiv ID:** 2610.37037 | [PDF](https://arxiv.org/pdf/2610.37037v1)

**作者:** Lai Wei `[一作]` (Shanghai Jiao Tong University), Weiran Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全自动的多区域视觉问题合成框架，利用单张图中相关区域的放大视图生成需要连接证据的多区域问题，并用RLVR对合成数据进行训练，得到 Zooming without Zooming v2（ZwZ2）模型。

**💡 创新点**

创新点：①把“连接证据”作为跨任务的共享核心能力，统一训练以提升多模态感知；②采用缩放视图、跨模型答案锚定和 drop‑one 验证实现高质量自动合成；③仅用单图训练即可显著提升跨域多模态推理性能。

**🔧 技术方法**

使用技术包括：多区域问题合成管线、高清裁剪与颜色标记、跨模型答案一致性检查与区域剔除验证、RLVR 强化学习训练以及多规模模型对比。

**📊 数据集**

数据集：20.4K 通过该框架自动合成的单图多区域问题；评测基准共 25 个，涵盖 Fine‑grained perception、Spatial perception、Long‑context understanding、Embodied spatial reasoning、Multimodal agent、General visual reasoning 等；与基线模型（GPT‑5.6‑Luna、GLM‑5.3‑Flash、DeepSeek‑V4.1‑Flash）及其他合成数据集（ZwZ、TVG、F²RVLM、CRIT、MMHops）比较。

**📈 对比分析**

评测方法：在所有 25 个基准上用无思考模式测算准确率。ZwZ2 在每一规模下均优于基线，整体平均提升约 5–6 分；在 27B 规模下整体准确率达到 60.51%，超过 GPT、GLM、DeepSeek 等先进模型。在与其他合成数据集的对比中，ZwZ2 在 4B 规模上平均提升约 4.3 分，且在 Embodied spatial reasoning 与 Long‑context 任务上提升尤为显著。

**⚠️ 局限性**

局限性：①训练数据仅为单图，长序列检索能力提升有限；②在极长上下文（>256K token）下提升趋缓；③生成问题依赖跨模型一致性，可能受模型偏差影响；④对部分基准（如 ERQA）提升不均匀或略低。

---

## 486. V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

**arXiv ID:** 2610.37019 | [PDF](https://arxiv.org/pdf/2610.37019v1)

**作者:** Miso Choi `[一作]` (Korea University), Jungbeom Lee `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可扩展的视觉外部查找记忆V‑Lumen，用于提升冻结的视觉语言模型（VLM）在推理任务中的性能，且不增加每个token的计算量。

**💡 创新点**

创新点在于将文本侧的离散查找记忆迁移到连续视觉特征，通过冻结的码表进行离散化、基于空间聚合的哈希生成区域感知键，以及文本条件哈希实现查询适应性，从而实现视觉特征的高效、稀疏检索；同时展示外部记忆是与LoRA、MoE等适配器独立的容量轴。

**🔧 技术方法**

主要技术包括：冻结码表离散化视觉补丁特征、空间聚合哈希（SAH）、文本条件哈希（PRH/DRH）、多头条件门控（MHCG）融合记忆向量、以及多阶多头查找表；训练采用两阶段（中期训练+SFT）冻结主干、只更新记忆表和融合层。

**📊 数据集**

使用多样化的视觉语言数据集进行训练：DocVQA、ChartQA、InfographicVQA、ScienceQA、Visual Genome、GQA、RefCOCO/RefCOCOg、AI2D、AOKVQA、TextCaps等；评估集包括ChartMuseum、FRIEDA、VisRes、MMMU、MMMU‑Pro。

**📈 对比分析**

在冻结的Qwen3‑VL‑2B基线上，V‑Lumen单独提升平均+17.8个百分点；在与LoRA/MoE在总参数预算匹配时，V‑Lumen+适配器保持或略优于纯适配器，证明外部记忆可替代部分适配器容量；在零样本和少样本评测中，V‑Lumen在三个未见图像域上均显著优于冻结基线。

**⚠️ 局限性**

局限性包括：记忆表仍需存放在GPU内存，尚未实现CPU/SSD外部存储部署；更大记忆表目前未必提升准确率，需进一步探索行宽与行数对性能的影响；目前仅在2B级模型验证，未知在更大规模模型上的表现；键构造耗时占大部分预填充时间，需要加速。

---

## 487. Design and Architecture of Airmod: An Active Noise Cancellation and Deep Learning System for Real-Time Respiratory Rate Estimation from Tracheal Sounds

**arXiv ID:** 2610.37083 | [PDF](https://arxiv.org/pdf/2610.37083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 488. Compote: Collapse-Resistant Multiple Hypothesis Learning For Sequential Recommendation

**arXiv ID:** 2610.37005 | [PDF](https://arxiv.org/pdf/2610.37005v1)

**作者:** Louis Castres Saint Martin `[一作]` (Criteo AI Lab), Patrick Gallinari `[通讯]` (Criteo AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多假设Transformer（COMPOTE），通过温度退火的多假设学习解决序列推荐中的多模态不确定性。

**💡 创新点**

创新点在于将多假设学习的温度退火与最大熵量化结合，并使用无归一化的max‑similarity推理，避免头坍塌并实现高质量排名。

**🔧 技术方法**

使用因果Transformer、多头预测、温度退火的多假设学习（annealed MCL）、max‑similarity聚合以及向量数据库兼容的无softmax实现。

**📊 数据集**

在Amazon Reviews 2023的三个子集：Musical Instruments、Industrial and Scientific、Video Games（约25k物品、5‑10万用户）。

**📈 对比分析**

与ID‑based、生成式以及多兴趣基线对比，在H@5/10和N@5/10指标上均取得显著提升，尤其在NDCG上超越TIGER；性能提升主要归因于退火多假设训练。

**⚠️ 局限性**

局限包括缺乏对稀有物品的语义建模，需进一步结合语义标识符；以及对最大熵分区的理论假设在实际场景中的适用性尚待验证。

---

## 489. Online gas classification at the edge

**arXiv ID:** 2610.36999 | [PDF](https://arxiv.org/pdf/2610.36999v1)

**作者:** Ton Juny Pina `[一作]` (University of Groningen), Elisabetta Chicca `[通讯]` (University of Groningen)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计了一种受昆虫嗅觉系统启发的轻量级在线气体分类网络，利用稀疏高维二进制表示和局部Hebbian学习，仅用几千个神经元和亚毫瓦功耗即可对传感器阵列的原始读数进行实时分类。

**💡 创新点**

创新点包括：① 通过随机二进制投影加k-最优胜淘汰实现稀疏高维嵌入，构造非线性核；② 采用全二进制权重和Hebbian学习实现低成本在线、少样本学习；③ 利用稀疏表示实现计数草图（在线频率估计）；④ 在混合信号神经形态芯片TEXEL上实现超低功耗边缘部署。

**🔧 技术方法**

使用技术包括：随机二进制投影+k-WTA稀疏化、Hebbian本地学习、梯度下降高分辨率线性读出、mox传感器阵列、时间序列扩展（一阶导数、成对差异/比值）、随机气体序列生成、混合信号神经形态芯片TEXEL。

**📊 数据集**

数据集为：① 单一气体（乙烯、甲烷、CO）及其二元混合物，浓度2000 ppm，训练600曝光、测试300曝光；② 危险气体检测实验，将CO标为危险，乙烯/甲烷为非危险，训练10曝光、测试150曝光。

**📈 对比分析**

与线性SVM和优化非线性（RBF）SVM对比：单气体数据上，二进制Hebbian读出0.881，线性SVM 0.923；二元混合数据上，0.645 对比 0.563，接近RBF 0.800。少样本学习：3曝光0.596，21曝光0.801，48曝光0.866；TEXEL芯片实现准确率74.6%，每次推断482步，静态功耗27µW。

**⚠️ 局限性**

局限性：二进制权重限制了表示能力；需要精细调节稀疏度、投影密度和突触增强概率；对更大气体种类或浓度范围的泛化尚未验证；TEXEL实验受限于2000维表示和静态功耗；整体性能仍略低于优化的RBF SVM。

---

## 490. GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation

**arXiv ID:** 2610.37091 | [PDF](https://arxiv.org/pdf/2610.37091v1)

**作者:** Yang Liu `[一作]` (Genisom AI), Xiang Gao `[通讯]` (Genisom AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种compact多任务世界-动作模型（Geometry-Aware World Model），在冻结的DINOv3编码器潜在空间中预测未来状态并直接回归动作；

**💡 创新点**

创新点在于：1）任务条件下的轻量级潜在预测器，无需大型VLM或视频生成器；2）双未来时域的密集patch监督，保证预测的空间布局；3）通过ACT式动作解码器实现无搜索的实时动作回归；

**🔧 技术方法**

使用的技术包括：冻结的DINOv3 ViT编码器＋查询适配器、双向Transformer预测器、任务代码条件化、VICReg风格的特征损失、ACT风格动作解码器以及密集patch监督；

**📊 数据集**

实验使用的主要数据集为LIBERO（四套任务）和RoboTwin 2.0（50双臂任务），并在Tianji Marvin Pro真实机器人上收集演示数据；

**📈 对比分析**

与多种VLAs、WAMs和compact policy基线对比，在RoboTwin 2.0上77M参数模型实现93.9%/93.1%成功率，超越十亿参数WAM；在LIBERO上38M参数模型95.6%成功率，优于3B参数VLA；同时推理延迟最低（约8.8 ms/7.1 ms）；

**⚠️ 局限性**

局限性包括：仅适用于封闭任务集，缺乏自然语言指令理解；依赖冻结视觉编码器；需要手工提供任务代码；在极少数据或极短训练时间下仍有提升空间但可能受限。

---

## 491. Polynomial-Time Solvability of the Mixed Evacuation Problem

**arXiv ID:** 2610.37065 | [PDF](https://arxiv.org/pdf/2610.37065v1)

**作者:** Naoyuki Kamiyama `[一作]` `[通讯]` (Kyoto University), Naoyuki Kamiyama (Kyoto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明混合疏散问题在多源多汇情况下可用半平面分离与内点法（椭圆法）多项式时间求解

**💡 创新点**

首次将椭圆法与动态网络流的对偶结构结合，利用子模函数最小化与极点分析构造可分离超平面，实现一般规模问题的多项式可解性

**🔧 技术方法**

线性规划对偶性、子模函数最小化、极点（extreme point）分析、椭圆分离法

**📊 数据集**

无，本文为纯理论证明，未使用任何实验数据集

**📈 对比分析**

论文没有实验对比，主要通过理论分析证明算法在输入长度 L 内的多项式时间复杂度

**⚠️ 局限性**

方法仅在理论层面有效；椭圆法在实际规模中常常效率低下；论文依赖 AI 辅助，缺乏完整的实现细节和实验验证

---

## 492. EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training

**arXiv ID:** 2610.37062 | [PDF](https://arxiv.org/pdf/2610.37062v1)

**作者:** Masato Fujitake `[一作]` `[通讯]`, Masato Fujitake

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EnergyCal 规则，对 Muon 与 plain Muon 优化器在稀疏 MoE 语言模型训练中进行能量校准，以显著降低损失尖峰并提升最终困惑度。

**💡 创新点**

仅依据路由专家的 Newton–Schulz 方向能量，对更新进行一侧衰减，无需额外状态或路由统计，保持轻量且无需改动谱变换。

**🔧 技术方法**

使用 Muon / plain Muon、Newton–Schulz 迭代、稀疏 MoE 架构、能量衰减系数、BF16/FP32 混合精度写回等技术。

**📊 数据集**

FineWeb 训练集，固定 2,048 令牌序列，1.31 B 训练令牌，总共 10,000 步。

**📈 对比分析**

在相同硬件、相同 token 预算下与 MuonH、plain Muon+WD 等基线进行多初始化跑；EnergyCal–MuonH 失尖峰负担下降 99.5%，最终 PPL 降 4.06%，吞吐量仅下降 1.62%；plain Muon 版 EnergyCal 失尖峰负担下降 96%，终端 PPL 小幅提升 0.4–0.5%。

**⚠️ 局限性**

实验仅在单一 4B 参数、128 专家 top‑2 MoE 架构上验证，缺乏对不同规模/架构/数据集的泛化；未与统一缩放或其他专家尺度方法对照，难以确定能量衰减的独立贡献；单步干预未能解释全部尖峰，机制解释仍不完整。

---

## 493. AI-Assisted Discovery of CNOT Decompositions for Quantum Permutations on Constrained Topologies

**arXiv ID:** 2610.37045 | [PDF](https://arxiv.org/pdf/2610.37045v1)

**作者:** Daniele Lizzio Bosco `[一作]` (University of Udine), Giuseppe Serra `[通讯]` (University of Udine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在受限拓扑的量子计算机上，本文研究了将任意量子置换合成为CNOT电路的高效方法，利用AlphaCNOT AI助手揭示了远程转置的低门实现模式以及中间CNOT融合机制，并基于此提出了Deterministic Global Transpositions算法；

**💡 创新点**

创新点包括（1）构造性证明：任意距离d>1的远程转置可用最多6d-4个CNOT实现，较传统的3d-1/3d-1实现少1个门；（2）提出Global Transpositions——在加权Cayley图上进行Dijkstra搜索，得到对置换的最优拓扑相关CNOT成本；（3）揭示中间CNOT融合现象，说明直接CNOT合成可超越仅使用独立转置块的上界，解释SWAP路由与CNOT合成之间的差距来源。

**🔧 技术方法**

技术手段包括：AlphaCNOT的强化学习+MCTS搜索，递归构造与证明，基于Permutation群与GL(n,𝔽₂) 的理论分析，Dijkstra在权重Cayley图上求最短路径，广泛实验评估与对比。

**📊 数据集**

使用的实验数据集主要是常量深度的量子旋转与置换电路，覆盖多种拓扑（4L、4Y、5L、5T、6L、6T、6Y、7L、7T、7Y、8F、8H、8T、8T2等），以及所有4量子比特的24个置换；AlphaCNOT多次（单次、十次）执行以获取最佳门数。

**📈 对比分析**

比较方法：将目标置换按（1）SWAP路由展开为CNOT（3倍SWAP成本），（2）Global Transpositions得到的最优转置分解，（3）AlphaCNOT直接合成，以及（4）5比特时用双向BFS求最优CNOT；结果显示平均门数：SWAP 23.85→GT 22.55（↓1.31门，≈5.5%）→AlphaCNOT 21.83（↓0.72门，≈3.2%）；在所有测试拓扑上AlphaCNOT均优于GT，GT优于SWAP。

**⚠️ 局限性**

局限性：对6d-4上界尚未证明其绝对最优；Global Transpositions仍受限于仅使用转置块，无法利用跨块线性变换；中间CNOT融合机制尚无完整理论表征，无法预测何时可利用；AlphaCNOT为随机搜索，缺乏可证明的构造式；扩展到更大规模和更复杂拓扑的可扩展性尚未验证。

---

## 494. Ground-to-Satellite Knowledge Distillation for Crop Classification under Temporal and Geographic Shift

**arXiv ID:** 2610.37021 | [PDF](https://arxiv.org/pdf/2610.37021v1)

**作者:** Miro Miranda `[一作]` (Jülich Supercomputing Centre), Claudia Paris `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用地面照片在训练阶段做特权监督、通过知识蒸馏提升卫星嵌入模型作物分类鲁棒性的方法。

**💡 创新点**

创新点在于仅在训练期间使用现场照片，无需推理时收集，且证明在冻结的卫星嵌入上蒸馏能显著提高宏F1，并在时空域移时进一步增强。

**🔧 技术方法**

采用 CLIP ViT‑B/32 作为教师，结合交叉熵、KL 散度与余弦相似度的蒸馏损失；卫星学生使用 AlphaEarth、TESSERA 嵌入或自训练的序列 Transformer，部分微调 TESSERA。

**📊 数据集**

使用 LUCAS 2018/2022 调查点数据，包含 17 类作物标签、现场四视角照片和 Sentinel‑2/1 时间序列，共 53,885 训练样本。

**📈 对比分析**

通过与仅使用 CE 的学生对比，实验显示冻结嵌入模型在 2018 训练、2022 迁移集上提升 5.5 macro‑F1，地理留存（地中海）提升 3.7，整体在 2.6–5.1 macro‑F1 之间获得显著改善。

**⚠️ 局限性**

局限性：仅在训练时依赖现场照片；部分微调的 TESSERA 在输入路径上与冻结模型不一致，导致结果不稳定；Boreal 领域迁移表现波动；方法依赖预训练 CLIP 与 GFM 嵌入。

---

## 495. Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing

**arXiv ID:** 2610.37011 | [PDF](https://arxiv.org/pdf/2610.37011v1)

**作者:** Nozomi Yamamoto `[一作]` (Hokkaido University), Itsuki Noda `[通讯]` (Hokkaido University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在基于SUMO的仿真环境中，对按需拼车服务引入高级会员制度进行实验，提出并比较了两种调度策略：缩短高级用户的到达时限（Shorter-Deadline）和为高级用户保留专用车辆（Dedicated-Fleet），以评估其对用户行程时长和取消率的影响。

**💡 创新点**

创新点在于将时限紧缩与车辆专用相结合，首次系统性地探讨了在不同运营条件下（高峰/非高峰、不同需求模式）如何通过调度优先级实现服务差异化，并揭示了“优先机制适配性”这一重要设计原则。

**🔧 技术方法**

使用了顺序最优插入调度算法，并在此基础上加入了时限紧缩约束；同时构建了专用车辆子队列实现资源划分；实验依赖SUMO仿真框架对车辆轨迹和需求进行细粒度模拟。

**📊 数据集**

采用合成的15×15网格地图（中心1.6×1.6公里为城市区），配合三种需求情景（均匀、时变峰值、城市聚焦峰值），共30个随机种子生成的请求流，用12辆车进行仿真。

**📈 对比分析**

通过对比基线、Shorter-Deadline和Dedicated-Fleet三种方案，使用行程时长/距离比（TPD）和取消率两个指标进行评估。结果显示：在高峰拥堵时限紧缩能显著降低高级用户TPD，同时保持取消率低；在非高峰条件下专用车辆能更明显区分两类用户，但在拥堵时会导致非高级用户TPD上升；两方案的最佳设置取决于可接受的非高级用户服务降级幅度。

**⚠️ 局限性**

局限性包括缺乏价格机制和真实用户行为数据，调度策略未考虑非随机高级需求分布，实验仅在单一网格地图和固定车队规模上验证，且未对结果进行真实场景验证，缺乏多样化地图和车队配置的稳健性分析。

---

## 496. Blindfold: Keeping the Audit Coin Secret from LLM Agents

**arXiv ID:** 2610.37055 | [PDF](https://arxiv.org/pdf/2610.37055v1)

**作者:** Wei Zhu `[一作]` (Renmin University of China), Huiping Sun `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM代理的审计认证机制，提出前置非干预（PNI）属性并评估泄漏对无偏置信息估计的影响。

**💡 创新点**

定义PNI并证明其对无偏证书的必要与充分性，量化泄漏导致的偏差，提出“盲fold”设计解决方案，并在真实系统上进行定量实验。

**🔧 技术方法**

采用量化信息流、逆概率加权（IPW）置信序列、总变距测度以及概率非干预与可测性理论，配合统计学习与实验验证。

**📊 数据集**

在AgentDojo环境中使用约560对真实用户任务与注入对的实验数据，结合GPT‑4.1‑mini等模型的实际运行日志。

**📈 对比分析**

通过覆盖率与置信上界误差对比盲fold与标准审计（含可见提示、时间戳等），盲fold在泄漏下覆盖率提升至≈99%，标准审计误差可达≈6%。

**⚠️ 局限性**

仅评估单个监控堆栈，未覆盖跨周期适应攻击；时延通道在大规模并发下仍可能泄漏；对实时性能影响未充分量化。

---

## 497. RePAIR: Inverting Backdoor Attacks in Semantic Segmentation

**arXiv ID:** 2610.37015 | [PDF](https://arxiv.org/pdf/2610.37015v1)

**作者:** Arnold Brosch `[一作]` (Heinrich Heine University Düsseldorf), Kira Maag `[通讯]` (Heinrich Heine University Düsseldorf)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种后置修复方法 RePAIR，能够在不重新训练模型、无需标签、未知后门的情况下，定位并修复语义分割模型中的后门攻击。

**💡 创新点**

创新点在于利用后门诱导的类别转移在头部更强而在内部表征更弱的特性，构建安全批准的参考集合，并通过内部表征与头部输出的对比实现像素级定位与选择性修复。

**🔧 技术方法**

采用了受害者-目标类推断、内部表征对比、概率残差校准、特征空间运输和输出安全门限等技术，并针对 CNN 与 Transformer 设计了架构特定的参考与多层逻辑回归。

**📊 数据集**

实验使用 Cityscapes、PASCAL VOC 2012 与 ADE20K 三个常用语义分割数据集。

**📈 对比分析**

与裁剪、微调、Segment Cleanse、ABL 等现有防御方法比较，RePAIR 在八种后门配置下将攻击成功率（ASR）降至 1% 以下，同时保持 74–72% 的 mIoU-B，显著优于其他方法。

**⚠️ 局限性**

局限性包括：依赖内部受害者表征的残留；对极弱或结构更复杂的后门可能效果有限；参考集合质量受无标签图像池污染影响；缺乏对未知后门强度的严格理论保证。

---

## 498. Decoupling Procedural and Outcome Fairness in Seed Allocation for Influence Maximization

**arXiv ID:** 2610.37112 | [PDF](https://arxiv.org/pdf/2610.37112v1)

**作者:** Xiaojing Du `[一作]` (Adelaide University), Feiyu Yang `[通讯]` (Civil Aviation Flight University of China)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Decoupled Procedural Allocation (DPA) 方法，用于在影响最大化问题中实现公平分配，并通过将配额与种子选择分离来平衡访问公平与结果公平。

**💡 创新点**

创新之处在于对期望结果差异进行精确分解为分配与结构两部分，并给出连续一阶矩下界；随后设计了闭式配额规则和基于贡献的交换框架，使得在给定访问容差下能够实现最小化的结果差距。

**🔧 技术方法**

主要技术包括线性邻域曝光模型、随机块模型（SBM）的解析、闭式配额推导、整数规划与局部搜索、度排序与贡献计数，以及对上下界的理论与实验验证。

**📊 数据集**

实验使用了合成的SBM和BA图，以及真实社交网络BlogCatalog、Flickr、具有性别标签的Amherst41和Pokec（约160万用户、22M条边）。

**📈 对比分析**

与Greedy-IM、Proportional、Maximin、Welfare、FIMM、LP-Greedy、IGM-AGM等九种基线以及其配额限制版本进行比较，在开发与前瞻集、访问容差为0.1或0.01时，DPA的平均结果差距最低（<3.2×10⁻⁴），并保持近乎最优的总覆盖；在Pokec上将性别结果差距压至2.4×10⁻⁷，同时保留99.14%总覆盖。

**⚠️ 局限性**

局限主要体现在：主要针对两组情况（可扩展但未全面验证）；依赖线性加性传播假设，对参数估计敏感；在高度异质或多组情形下，理论下界与实践表现可能出现更大偏差。

---

## 499. Text Generation in Low-Resource Languages Using Human-Curated Vocabularies

**arXiv ID:** 2610.37105 | [PDF](https://arxiv.org/pdf/2610.37105v1)

**作者:** Michael Christlambert Sinanta `[一作]` (Independent Researcher), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、基于LLM推理、结合人类词典和少量示例的低资源语言合成文本生成框架，并在六种东南亚语言上进行评估。

**💡 创新点**

首次将人类手工词典与可采样词候选相结合，并利用LLM推理在未见语言中生成连贯且词汇多样的文本。

**🔧 技术方法**

使用词频估计、Zipfian采样、主题约束、few‑shot示例与词候选提示，配合 Gemini 3.5 Flash、Claude Haiku 4.5、GPT‑5.4 Mini等LLM推理。

**📊 数据集**

使用六种语言（印尼语、越南语、爪哇语、明古塞语、阿切语、Hitu）的官方词典、手写示例及20个主题词集。

**📈 对比分析**

与两种基线（Zero‑Shot与Few‑Shot）对比，基于流畅度、离谱率与词汇多样性评估；在极低资源语言上流畅度提升约0.7分，离谱率降至≤0.09，词汇多样性显著提升。

**⚠️ 局限性**

仅覆盖拉丁字母、语法相似的东南亚语言，未验证非拉丁或形态学复杂语言；依赖词典质量，且实验限于三款商用LLM。

---

## 500. Hardware-Matched Spiking Neural Networks for Real-Time Purkinje Cell Spike Classification

**arXiv ID:** 2610.37104 | [PDF](https://arxiv.org/pdf/2610.37104v1)

**作者:** Arooba Farhan `[一作]` (Lahore University of Management Sciences), Muhammad Ali Siddiqi `[通讯]` (Lahore University of Management Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在无绳头台上实现了实时小型化脉冲神经网络，用于小鼠Purkinje细胞的简单波/复杂波分类。

**💡 创新点**

提出两种硬件匹配的SNN：ConvSNN利用矩阵向量处理器加速；StreamSNN采用时序单元和固定延迟实现低面积/能耗。

**🔧 技术方法**

使用脉冲神经网络、矩阵向量处理器、SkyWater 130 nm ASIC、非线性能量检测、后训练8位量化等技术。

**📊 数据集**

基于64只小鼠Purkinje细胞的手工标注记录，采样24.414 kHz，包含噪声多样的记录。

**📈 对比分析**

与随机拆分的现有PC分类器相比，ConvSNN在cell‑held‑out上CS F1达到0.873、StreamSNN 0.837；在随机拆分下ConvSNN CS F1从0.867提升至0.906；MVP加速使延迟降低62%并将电池寿命从17h延长到34h。

**⚠️ 局限性**

受限于样本量、事件重叠处理和缺乏真实行为动物验证，且ASIC延迟主要受波形采集时间限制。

---

## 501. Learning to Look Beyond Screens: Transferable Visual Observation from GUI Grounding

**arXiv ID:** 2610.37093 | [PDF](https://arxiv.org/pdf/2610.37093v1)

**作者:** Ruilin Yao `[一作]` (Wuhan University of Technology), Changhua Meng `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于GUI定位监督的两阶段训练框架，先用教师生成的交互轨迹进行监督微调（SFT），再通过强化学习（RL）优化多轮裁剪-检查决策，使视觉‑语言模型能够主动定位并查看关键信息；

**💡 创新点**

创新点在于利用GUI任务中的空间标注来给中间观察行为提供奖励与掩码，形成“outcome‑gated spatial reward”，实现观察策略的可转移性；

**🔧 技术方法**

主要技术包括视觉‑语言模型（Qwen3.5-35B-A3B、Qwen3.8-27B）、裁剪工具、基于GRPO的多轮强化学习、空间奖励与小目标增益、文本一致性奖励；

**📊 数据集**

训练数据来源为自动生成的GUI界面与对应指令与目标框，评估数据包括ScreenSpot‑Pro、PerceptionBench、HR‑Bench、ChartQAPro、OSWorld‑G‑R等；

**📈 对比分析**

与传统GUI定位方法相比，RL提升了约1.8–6.1个百分点；在跨域评估中，RL相较于SFT在PerceptionBench提升约3–7个百分点，在HR‑Bench、ChartQAPro等任务也实现了显著改进；

**⚠️ 局限性**

局限性包括对裁剪工具的依赖、对高分辨率裁剪的计算成本、在某些任务中RL对源域性能提升不一定带来更好转移、以及对视频或长序列数据的适应性尚未验证。

---

## 502. Systematic Morphometric Brain Autoencoding for Representation Learning in Neurodegenerative Diseases

**arXiv ID:** 2610.37074 | [PDF](https://arxiv.org/pdf/2610.37074v1)

**作者:** Nuria Balbás-Esteban `[一作]` (Universidad Carlos III de Madrid), Carlos Sevilla-Salcedo `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了 SYMBA——一种针对结构 T1‑weighted MRI 的自监督预训练框架，用于提取神经形态特征以支持神经退行性疾病的诊断与预后。

**💡 创新点**

创新点在于三方面：① 采用解剖学约束的几何遮蔽策略，使遮蔽区域与脑结构一致；② 使用多尺度结构相似性（MS‑SSIM）重建损失，专注于遮蔽部位的结构保留；③ 通过逐步增加遮蔽比例和尺度的课程学习，逐步从局部细节迁移到全脑形态。

**🔧 技术方法**

技术实现包括：3D ResNet‑50 编码器、5 层转置卷积解码器；遮蔽模块基于脑灰质掩码、双侧对称、最小组织阈值；重建损失为 MS‑SSIM；自监督训练使用 AdamW、余弦退火；下游任务采用冻结线性探针。

**📊 数据集**

预训练数据来源于 6,953 张来自七个不同研究中心（OASIS、DLBS、IXI、SALD、Wayne、NKI、CamCAN）的健康受试者 T1‑MRI；下游评估使用两组从未参与预训练的独立数据集 ADNI 与 AIBL，涵盖 AD/ CN、MCI 进展、3‑类诊断、Aβ 状态及性别判别。

**📈 对比分析**

与 BrainMorph、BrainSegFounder、AnatCL、BrainIAC 等现有自监督基线进行冻结线性探针对比，且在 ADNI 与 AIBL 上均获得最高 BACC/AUROC；在低标签（1%‑100%）环境下 SYMBA 在 AD/CN、Sex、Amyloid 等任务中至少 0.90 的 AUROC，并在仅 6 个标记样本时已达到 0.90 的 AD/CN 识别性能，显著优于全监督与其它预训练模型。

**⚠️ 局限性**

局限包括：仅评估了线性探针而非微调；对不同脑部疾病（如帕金森、癫痫）未验证；遮蔽策略与 MS‑SSIM 仍需在不同扫描协议下进一步验证；模型规模较大，对资源受限环境不友好。

---

## 503. Janus Vehicles: Entity-Centric Threat Modelling of Protocol-Conformant Vehicles under a Malicious-Manufacturer Assumption

**arXiv ID:** 2610.37059 | [PDF](https://arxiv.org/pdf/2610.37059v1)

**作者:** Jaeung Franciskus Yoo `[一作]` (Korea University), Huy Kang Kim `[通讯]` (Korea University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种面向实体的威胁模型，用以识别在协议符合但功能违背声明用途的恶意制造车辆及相关威胁；

**💡 创新点**

创新点在于将恶意实体分为四类（恶意设计、被攻陷、被重新利用、仿冒）并引入行为-位置矩阵与行为交互模型，从多维度描述隐藏功能的触发、表现与隐蔽方式；

**🔧 技术方法**

主要使用了概念建模与威胁树分析技术，借鉴恶意软件行为目录与属性枚举方法来构建模型；

**📊 数据集**

未使用任何实际数据集，模型为理论性概念框架；

**📈 对比分析**

未进行实验对比或性能评估，缺乏量化指标；

**⚠️ 局限性**

局限性包括：模型尚未在真实车辆或仿真环境中验证，缺乏实证数据；无法直接评估隐藏功能的检测率；并且无法判定制造商的主观意图与法律责任。

---

## 504. p-DCIM: A 28nm Probabilistic SRAM DCIM based Ising Machine for Near-Memory Context Selection in LLM-RAG Pipeline

**arXiv ID:** 2610.37033 | [PDF](https://arxiv.org/pdf/2610.37033v1)

**作者:** Madhav Vadlamani `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一款28nm SRAM-DCIM双模式Ising加速器p-DCIM，用于近存储RAG上下文选择；

**💡 创新点**

创新点在于：1) 采用同一SRAM阵列与加法树实现QUBO构造与求解的双模式重用；2) 利用SRAM伪读失真产生随机性并通过数字磁平面计数实现可控的退火温度；3) 将该加速器嵌入HBF基底，实现候选嵌入与相似度计算靠近存储，减少数据搬移；

**🔧 技术方法**

技术包括SRAM 12T计算单元、双模式操作（确定性与概率性）、基于QUBO的上下文选择、数字退火调度、以及在FPGA/AXI平台下的芯片测量与行为建模；

**📊 数据集**

使用HotpotQA数据集进行上下文选择与下游答案生成的评估；

**📈 对比分析**

与传统基于LLM的SetR/SetRO选择方法对比，p-DCIM在保持相似或略高的Exact Match、Gold-Contains、Token-F1指标的同时，选择阶段能耗降低6个数量级，整体查询能耗下降67%；

**⚠️ 局限性**

限制在于：目前仅支持64×64 QUBO实例，需通过模型扩展到更大规模；退火过程受SRAM噪声特性的影响，对参数调优要求高；集成到完整GPU/HBF系统仍需进一步验证。

---

## 505. Interpretable Deep Learning Models for Breast Cancer Diagnosis Using Medical Imaging

**arXiv ID:** 2610.37026 | [PDF](https://arxiv.org/pdf/2610.37026v1)

**作者:** Sami Alnawali `[一作]` (King Faisal University), Ahmad Din `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究在BreakHis数据集上比较10种迁移学习模型的乳腺癌二分类性能，并结合可解释AI技术。

**💡 创新点**

创新点在于系统性评估CNN与Transformer模型并将Grad‑CAM、LIME、SHAP等XAI方法结合，提供临床可解释的诊断框架。

**🔧 技术方法**

使用迁移学习的CNN模型（ResNet50, DenseNet121, EfficientNet等）和Transformer模型（ViT, Swin），配合数据增强、预处理和多种XAI方法。

**📊 数据集**

采用公开的BreakHis乳腺癌组织病理图像数据集，包括40×、100×、200×、400×四种放大倍数的图像。

**📈 对比分析**

通过交叉验证和患者层面划分的测试集对模型进行比较，DenseNet121在测试集上取得99.4%准确率、0.9997 AUC，其他CNN模型平均在84‑86%之间，Transformer表现最差。

**⚠️ 局限性**

局限在于仅使用单一二分类数据集、缺乏多中心或多模态验证、XAI评价主要为定性、Transformer模型超参数搜索受限。

---

## 506. Linear-Time FPT Rectilinear Planarity Testing of Series-Parallel Graphs

**arXiv ID:** 2610.37018 | [PDF](https://arxiv.org/pdf/2610.37018v1)

**作者:** Giacomo Ortali `[一作]` `[通讯]` (Universitá Telematica Internazionale Uninettuno), Giacomo Ortali (Universitá Telematica Internazionale Uninettuno)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在变量嵌入下系列并行四度图的直线可平面性测试，给出了 2^O(k)·n 的时间复杂度，其中 k 为度为 4 的顶点数。

**💡 创新点**

关键创新是“关键引理”，证明了轻型 P2‑节点（无度 4 顶点）不会增加旋转度集合的块数，从而控制整体块数上限。

**🔧 技术方法**

采用了旋转度（spirality）理论、SPQ*‑树结构、组合计数和几何分析，并在 Lean 4 中完成了形式化证明。

**📊 数据集**

该工作为纯理论算法，不使用实验数据集；所有结果均通过数学证明和形式化验证获得。

**📈 对比分析**

相较于已有的 O(n²) 算法，若 k = o(log n) 则可实现子二次（近线性）时间；在一般情况下时间为 2^O(k)·n，优于之前的二次上界。

**⚠️ 局限性**

主要局限在于时间复杂度中 2^O(k) 的系数（36⁴k）相对粗糙，且依赖于重型 P‑节点数量；尚未证明能否仅用 P3‑节点数或进一步压缩块数；对非双连通图或弯曲最小化的推广仍待研究。

---

## 507. From Ethernet to Wi-Fi: Flow Feature Selection and Construction for Traffic Classification

**arXiv ID:** 2610.37007 | [PDF](https://arxiv.org/pdf/2610.37007v1)

**作者:** Francisco Rau `[一作]` (MaxLinear), Joaquin Perez `[通讯]` (Universitat de València)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个五阶段流程，用以从以太网训练的模型中选取并构造特征，使其能够在Wi‑Fi流量上实现准确的流分类。

**💡 创新点**

创新点在于：①结合速率依赖性、分布偏移（KS距离）与互信息比率三种指标进行特征筛选；②通过冗余相关性消除冗余特征；③构造四个归一化比例特征，以降低速率对特征的影响。

**🔧 技术方法**

采用ns‑3模拟器生成以太网与Wi‑Fi流量，使用XGBoost分类器，并利用Pearson相关、Kolmogorov‑Smirnov距离、互信息估计和TreeSHAP等统计方法进行特征评估与解释。

**📊 数据集**

使用ns‑3模拟的8类流量数据集：VoIP、Video Conference、Gaming、Video Streaming（DASH）、Bulk TCP、IPERF TCP、IPERF UDP、Bursty UDP；包含六个开发场景S1–S6以及一个留作测试的场景S7。

**📈 对比分析**

通过在Ethernet训练集上调参，比较不同特征集合（29→28→16→12→9→7）在Ethernet测试集、S1–S6与S7上的准确率。最终7‑特征模型在S7的准确率提升至≈99.9%，而在Ethernet保持≈99.5%。

**⚠️ 局限性**

局限性包括：仅在ns‑3仿真环境下验证；仅使用Best‑Effort访问类别和单一DASH客户端；未覆盖不同的Wi‑Fi物理层配置、实际网络干扰与多样化流量实现，因而可能无法直接推广至真实生产网络。

---

## 508. Abstract Lipschitz Continuity

**arXiv ID:** 2610.37006 | [PDF](https://arxiv.org/pdf/2610.37006v1)

**作者:** Marco Campion `[一作]` (Sorbonne Université), Caterina Urban `[通讯]` (Inria)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出并系统化了抽象 Lipschitz 连续性（Abstract Lipschitz Continuity, ALC）这一新的函数外延性质，定义了在程序语义、抽象解释和安全分析等领域的适用性，并给出了面向程序语法的可证明的演绎系统。

**💡 创新点**

创新点主要包括：
- 将传统的 Lipschitz 连续性从数据空间推广到属性空间，显式区分属性（qualitative）与量化（quantitative）两类近似；
- 将 ALC 与已存在的三大 extensional property（partial 完整性、partial abstract non‑interference、abstract robustness）建立充分条件的关系；
- 设计了一套参数化的演绎系统，通过输入/输出抽象与距离的兼容性来恢复 ALC 的组合性，并给出 join 与迭代的侧条件；
- 通过理论与多例子展示 ALC 在安全、优化与机器学习鲁棒性中的统一解释。

**🔧 技术方法**

技术手段：
- 抽象解释框架（upper closure operator）与预度量（pre‑metric）的组合；
- 抽象 Lipschitz 连续性的定义与性质推导；
- 演绎系统（Hoare‑style 规则）及其证明（soundness、侧条件）；
- 对常见距离（Hausdorff、路径长度、大小距离）与抽象（区间、符号、直方图）做具体实例化。

**📊 数据集**

本文主要以理论为主，并通过若干结构化示例来说明：
- 文件字节序列与计数模式（字节直方图与区间抽象）；
- 整数集合与符号抽象（区间与包含关系）；
- 迭代闭包 R* 在整数集合上的行为。
未使用外部真实数据集，而是借助这些数学化的例子来验证理论。

**📈 对比分析**

方法比较：
- ALC 作为充分条件与 partial 完整性/partial abstract non‑interference/abstract robustness 的关系，用数学证明给出包含关系；
- 演绎系统与先前的 Lipschitz 证明系统（Chaudhuri 等）进行对比，说明 ALC 通过兼容性侧条件实现更细粒度的组合性；
- 在示例中展示 ALC 如何给出精确的 Lipschitz 常数（如 1‑ALC、0‑ALC）并说明其对鲁棒性的影响。
性能方面为理论证明，未给出实验数值。

**⚠️ 局限性**

局限与未来工作：
- ALC 在一般情况下不具备闭合性，需通过侧条件或更精确的抽象来保证组合性；
- 演绎系统不完整，得出的 Lipschitz 常数可能高于最优值；
- 对于多种距离与抽象的组合，仍需人工验证侧条件；
- 目前仅给出理论证明，缺乏大规模实证评估与工具实现；
- 对于更复杂的编程语言（如并发、概率）和更丰富的安全策略，ALC 的适用性还需进一步研究。

---

## 509. Structure-Actuation Co-Design of All-Electric Heavy-Duty Robotic Manipulators

**arXiv ID:** 2610.36996 | [PDF](https://arxiv.org/pdf/2610.36996v1)

**作者:** Mohammad Bahari `[一作]` (Tampere University), Jouni Mattila `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对全电动重载机械臂，提出了一种混合离散-连续多目标双层结构-驱动协同设计框架，并在含12个全姿态目标、100 kg 工具负载的 SE(3) 基准上得到八个可行的非支配设计。

**💡 创新点**

创新点在于将机构拓扑与电机/减速器/螺旋桨离散选择联合优化，并通过系统级协调与质量反馈迭代实现驱动质量一致性，形成了可收敛的质量自洽双层优化方法。

**🔧 技术方法**

使用 NSGA‑II 多目标遗传算法完成上下层搜索，结合离散组件目录、损耗感知能量模型、刚性动力学仿真与结构应力屏蔽，以及质量反馈循环实现结构与驱动的自洽。

**📊 数据集**

实验基准为12个 SE(3) 末端姿态和 100 kg 工具负载（与 50 kg 球形腕部），并在 20、60、100、140 kg 负载下做负载敏感性筛选；驱动目录包含 68 台电机，减速比与螺距为离散值。

**📈 对比分析**

通过固定预算的双层优化与一次性顺序尺寸化基线对比，评估了硬件质量、联合成本指数与供应能量的差异；结果显示质量收敛需三步，非支配集的硬件质量 387–420 kg、成本指数 1.187–1.310、能量 30.53–36.90 kJ；一次性尺寸化在各指标上可改动 0.43–2.13%、-3.81–10.20% 与 -8.80–9.40%。

**⚠️ 局限性**

局限性包括使用简化的结构应力与电机损耗模型、腕部固定设计、未考虑腕部驱动能量、有限的搜索预算与非全局收敛，未来需要更高保真组件/热模型、鲁棒性分析和实验验证。

---

## 510. CaLLM: Continual Adaptation of LLMs via Non-Parametric Adapter Routing

**arXiv ID:** 2610.36994 | [PDF](https://arxiv.org/pdf/2610.36994v1)

**作者:** Bilge Celik `[一作]` (Eindhoven University of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CaLLM 框架，通过在冻结的大语言模型上部署一组轻量化适配器（LoRA）并结合基于原型的非参数路由器，实现无灾难性遗忘的持续学习，支持在线和离线任务流。

**💡 创新点**

创新点在于：①将任务分布映射到原型空间并用余弦距离实现无梯度训练的路由；②动态管理适配器池，自动创建或复用专家；③保证每次推理仅激活一个适配器，计算复杂度与任务数无关；④通过构造性的任务聚类实现隐式重放，提升跨任务迁移。

**🔧 技术方法**

技术细节包括：冻结Transformer backbone；LoRA 低秩适配器；平均池化获取分布式嵌入；余弦距离原型匹配；阈值决策（reuse/创建/复用）；批处理与在线两种训练协议；使用平均性能、后向转移 (BWT) 等指标评估。

**📊 数据集**

实验数据集：TRACE‑8（8 个跨域任务，包含 QA、代码、数学等）和 CITB‑19（19 个对话子任务，细粒度分类），并在两种协议下进行评估。

**📈 对比分析**

与 Seq‑FT、Inference‑Only、EWC、ER、oLoRA、TreeLoRA 等基线对比，CaLLM 在 TRACE‑8 上平均归一化分数达 0.44，明显高于 Inf‑Only 0.31；在 CITB‑19 中 12/19 任务获得最高分；后向转移保持接近 0 或正值，表明无显著遗忘；计算开销仅为单适配器训练的 0.5%，推理增量 <0.1%。

**⚠️ 局限性**

局限性：需设定原型阈值或池容量，过小会导致严重干扰；对相似但语义重叠的任务（如跨语言同一任务）分离效果不佳；目前仅验证文本任务，未扩展到多模态；适配器复用策略可能在极端稀疏任务流中失效。

---

## 511. SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

**arXiv ID:** 2610.36991 | [PDF](https://arxiv.org/pdf/2610.36991v1)

**作者:** Zafer Sahinoglu `[一作]` (Mitsubishi Electric), Cahit Berkay Kazangirler `[通讯]` (Karabuk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出SpectRAG，一种基于谱图增强的检索-生成框架，通过密集检索种子、图邻域扩展、谱传播与令牌预算上下文选择，构建多样、连贯、证据完整的检索上下文。

**💡 创新点**

将文档块视为图节点，融合语义相似、文档相邻、章节层级、实体、引用及LLM生成的功能问题等六种边信号，利用图拉普拉斯谱坐标进行扩散与重评分，实现从独立检索到全局上下文优化。

**🔧 技术方法**

密集向量检索（FAISS）、BM25、Utility-Question匹配、图构建与加权、谱分解（Laplacian eigenvectors）、图邻域遍历、个性化PageRank、令牌预算选择、LLM生成。

**📊 数据集**

MultiHopRAG、UltraDomain（农业、计算机科学、法律、混合四域）以及HotpotQA、2WikiMultiHopQA、MuSiQue等多跳与长文档问答基准。

**📈 对比分析**

与NaiveRAG、HybridRAG、LightRAG、HyDE等基线对比。检索层面SpectRAG在Recall@10、MRR上显著领先；生成层面在EM/F1上略优或持平，尤其在推理类问题上表现突出。

**⚠️ 局限性**

检索提升不一定转化为显著生成提升；对比、时间推理等问题表现不一；信号权重与块粒度对性能敏感；大规模部署需解决向量索引与谱计算的计算与存储成本。

---

## 512. SplatScout: Query-Driven Part-Level Referring Segmentation in Gaussian Scenes

**arXiv ID:** 2610.37098 | [PDF](https://arxiv.org/pdf/2610.37098v1)

**作者:** Yihan Zang `[一作]` (King Abdullah University of Science and Technology), Dominik Engel `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询驱动的闭环感知框架，直接由视觉语言模型（VLM）控制观察、分割和三维实例构造，实现对Gaussian场景的细粒度部件级指代分割。

**💡 创新点**

创新点在于：① 将感知任务交给VLM做决策，而非预先生成候选集合；② 采用多回合闭环交互，让VLM在每一步使用视觉反馈（观测图、分割遮罩、渲染结果）自我修正；③ 引入新构建的PartRef-ScanNet++基准，用于评估部件级指代。

**🔧 技术方法**

技术包括：2D Gaussian Splatting渲染、SAM进行查询引导分割、FlashSplat投票法进行Gaussian提升、Navigator进行全局与局部视角采样、闭环决策策略（VLM内部策略）以及多视角投票融合。

**📊 数据集**

使用扫描数据集ScanNet++（12场景）和新建的PartRef-ScanNet++基准（261条手工验证的查询），以及对比其他四个公开指代分割基准（Ref-LERF、Causal-LERF、GR-LERF、GR-ScanNet）进行评测。

**📈 对比分析**

与基线（CLIP检索、Search3D、GaussDet、OpenGround、ZeroSplat）对比，SplatScout在PartRef-ScanNet++上取得0.538 mIoU，远超最强基线0.253；在四个公开基准上也实现了大幅提升，最高达到71% @0.25/0.5阈值；闭环交互与多回合投票是显著性能提升的关键。

**⚠️ 局限性**

局限性包括：① 对极端视角缺失或遮挡的场景仍易失误；② 依赖高质量的VLM，模型规模和参数对性能影响大；③ 目前未充分利用场景语义图结构，可能限制在更复杂语义关系的理解；④ 计算成本较高，尤其在多回合闭环过程中需要多次渲染和投票。

---

## 513. A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation

**arXiv ID:** 2610.37110 | [PDF](https://arxiv.org/pdf/2610.37110v1)

**作者:** Wenwu Tang `[一作]`, Olga Saukh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在已完成分布匹配蒸馏的少步扩散模型上，利用极少量可训练参数仅在第一步加入偏置适配器，以恢复被压缩的种子多样性。

**💡 创新点**

提出 ReSeed：一种后训练、仅针对第一步的轻量级偏置修正方法，能在不重新蒸馏、无需目标图像或对抗损失的情况下，大幅提升多样性。

**🔧 技术方法**

使用流匹配扩散框架、分布匹配蒸馏 (DMD) 学生模型、教师平均速度对齐目标以及类 BitFit 的常数通道偏置适配器。

**📊 数据集**

图像实验基于 COCO‑2017 验证集和 JourneyDB 训练提示；视频实验基于 VBench 的 320 题目子集（Wan2.1-T2V‑1.3B）。

**📈 对比分析**

与原始 4 步 DMD、GAD、DP‑DMD 等基线比较，ReSeed 在 SD3‑Medium、SD3.5‑Medium 和 FLUX.1‑dev 上将 DINOv3 多样性提升 37–48%，FID 下降 3–5 点；视频模型在 4 步 Wan2.1‑T2V‑1.3B 上视频级 DINO 多样性提升 27%，同时保持原有质量。

**⚠️ 局限性**

仅在流匹配扩散模型上验证，未测试 U‑Net 或非流匹配架构；需要教师在第一步的轨迹信息，且对教师指导的依赖性有限。

---

## 514. TriDifNet: A Unified Three-Task Diffusion Network for End-to-End RGB-T Perception

**arXiv ID:** 2610.37097 | [PDF](https://arxiv.org/pdf/2610.37097v1)

**作者:** Qianchao He `[一作]` (Xi'an Jiaotong University), Xiaoqing Tian `[通讯]` (Xi'an University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TriDifNet，统一三任务（几何配准、跨模态融合与目标检测）通过扩散网络实现RGB‑T视觉感知。

**💡 创新点**

首次将光学与红外图像的几何配准、跨模态融合与目标检测三任务闭环耦合，并在每个任务中采用条件扩散，构建任务互反馈机制。

**🔧 技术方法**

使用条件扩散模型、密集变形场估计、残差条件扩散、ROI引导重推理以及Nash-MTL多任务平衡技术。

**📊 数据集**

在KAIST多光谱行人基准上进行实验，并通过人工仿射扰动模拟弱对齐情形。

**📈 对比分析**

在注册、融合与系统级检测上与多种基线对比，TriDifNet在误差对齐与检测 mAP_50:95 上分别实现最高表现（Recall 65.44%，mAP_50 57.55%，mAP_50:95 26.83%）。

**⚠️ 局限性**

推理速度受扩散迭代影响，未针对大规模实时场景优化；对严重非刚性畸变的鲁棒性有限。

---

## 515. OpenGray: a benchmark for evaluating the decisions of radiotherapy planning agents

**arXiv ID:** 2610.37081 | [PDF](https://arxiv.org/pdf/2610.37081v1)

**作者:** Humza Nusrat `[一作]` (Henry Ford Health), Kundan Thind `[通讯]` (Henry Ford Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了OpenGray基准，用已知答案的头颈放疗计划任务评估LLM规划代理的决策质量，并审计了临床代理SAGE的规划日志。

**💡 创新点**

通过已知答案的对照任务揭示代理在识别不可实现目标时的失误，强调需评估决策而非仅剂量，并为停机与审查提供明确动作。

**🔧 技术方法**

结合规则脚本、LLM（Terra与Ministral 3 8B）及完整LLM代理，利用OpenKBP‑Opt dose‑influence 矩阵实现精确剂量重算。

**📊 数据集**

使用OpenKBP‑Opt 100例头颈病例（30例评估、10例开发、60例校准），并审计了39例SAGE的SRS记录。

**📈 对比分析**

对四个系统在四类任务（例行、冲突、缺失结构、覆盖/重叠矛盾）中决策准确率、剂量目标达成率和识别率进行计分；在Terra下LLM系统决策准确率>90%，但在重叠矛盾仅提升50%或25%；所有系统提交可行控制但仍有24%未满足硬目标；SAGE日志显示识别成功但缺乏停止指令。

**⚠️ 局限性**

基准仅覆盖头颈部、两种矛盾类型、30例评估；未考虑可实现但接近边界的请求；未加入机台约束；仅审计单一临床代理；缺少已验证的临床案例答案。

---

## 516. Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation

**arXiv ID:** 2610.37080 | [PDF](https://arxiv.org/pdf/2610.37080v1)

**作者:** Peichun Hua `[一作]` (Chinese University of Hong Kong), Yunming Xiao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、无索引更新的 evidence delivery 层，在检索后重新打开页面、构建结构完整的单元（行、列表项、段落等），并在读者预算内用一个单元块替换传统的文本块或截图块，显著提升 Web RAG 的答案准确率。

**💡 创新点**

核心创新在于将检索阶段和阅读阶段解耦：先用检索器定位页面，再在查询时即时解析 HTML，构造带完整结构（标题、表头、行标签）的“结构完成单元”，并通过跨编码器选择与问题相关的单元进行预算友好打包，解决传统固定窗口/截图导致的结构信息丢失问题。

**🔧 技术方法**

技术实现包括：① 固定窗口（1,024 tokens）或截图分块的检索与重排；② 通过脚本过滤后按 DOM 顺序遍历页面，生成块单元和行单元；③ 通过跨度展开、表头重建、标题与分组归纳等方法恢复完整表格网格；④ 词汇预过滤、跨编码器评分、按上下文预算打包；⑤ 兼容文本、截图、以及多步 agentic RAG 等不同管道。

**📊 数据集**

实验数据集：English Wikipedia 2025 snapshot，7,851 问答对（Natural Questions、NQ‑Tables、SimpleQA）用于单步问答；多步 agentic 评测使用 FRAMES（824 问）和 MuSiQue（1,000 问）。

**📈 对比分析**

在三种主流管道（混合文本+BGE、PixelRAG 文本、PixelRAG 截图）中，加入此层均提升 5–6 分（在所有数据集和读者上均显著），并在 3,000–12,000 token 预算范围内，单元块在保持相同 token 预算时比传统块更优；在 agentic 搜索中提升 5.2 分、减少搜索次数，优于 14 种现有检索/上下文扩展或结构感知方法。

**⚠️ 局限性**

局限性包括：① 仅适用于可获取完整 HTML 的静态页面，动态渲染或客户端脚本生成内容的 Web 仍需改进；② 依赖锚页面的检索结果，未检索到的页面无法恢复；③ 词汇预过滤无法捕获语义相似的单元，导致部分重要信息被漏掉；④ 对非 Wikipedia 资源的泛化性尚未充分验证。

---

## 517. GraftRay: Physically Based Mesh Insertion into Gaussian Splatting Scenes via Path Tracing

**arXiv ID:** 2610.37036 | [PDF](https://arxiv.org/pdf/2610.37036v1)

**作者:** Jinxin Ai `[一作]` (Technical University of Munich), Matthias Niessner `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在2D/3D高斯剖面场景中实现物体的物理一致插入，恢复场景材质、法向、局部面积光源，并通过混合路径追踪实现视角一致的阴影、间接光照与软阴影。

**💡 创新点**

创新点包括：①利用LDR图像的饱和像素作为截断观测，结合单光源单元辐射基图像求解光源强度；②将高斯原语与面具网格统一在OptiX路径追踪器中，采用随机透明度采样实现光线传输；③通过差分合成仅传递插入物体对光照的变化，保留原高斯重建的细节。

**🔧 技术方法**

核心技术：光源估计（截断最小二乘）、高斯原语路径追踪（随机透明度、Softshadow射线）、基于Diffusion的材质/法向估计（RGB↔X、DiffusionRenderer）、差分渲染与光照递推。

**📊 数据集**

使用的实验数据集包括：FIPT室内合成场景（4个房间）、NFINITE室内合成场景、FIPT-Real真实捕获、ScanNet++真实与生成场景、SpAItial平台生成的3DGS场景。

**📈 对比分析**

与ComGS、EAG-PT、GaSLight、MV-CoLight、LumiTokens等基线比较，GraftRay在FIPT和NFINITE上以ΔPSNR≈35 dB、阴影IoU≈0.80、VQA分数领先，显示出更精确的阴影与光照一致性，渲染效率保持在同类方法之内。

**⚠️ 局限性**

局限性：光源需手动标注（使用SAM），对2D估计的材质/法向先验依赖较大，误差会直接影响光源拟合与物体插入效果。

---

## 518. Crop Yield Prediction Using Regression Models: A Comparative Evaluation of Model Performance

**arXiv ID:** 2610.37022 | [PDF](https://arxiv.org/pdf/2610.37022v1)

**作者:** Saptak Das `[一作]` (Indian Institute of Science Education and Research), Suborno Naskar `[通讯]` (MCKV Institute of Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对一个包含环境、土壤、作物、气候和管理变量的农作物产量数据集，使用多种回归模型进行产量预测，并对不同训练/测试划分进行性能比较。

**💡 创新点**

提出了基于不同特征编码（完整 one‑hot 与删去参考类别）与不同特征数量（20 与 24）以及不同超参数调优策略（无调优、随机搜索、网格搜索）的系统比较，展示了线性模型在该数据集上与树模型之间的泛化差距。

**🔧 技术方法**

使用的技术包括：多元线性回归、岭回归、Lasso 回归、随机森林回归、决策树回归，以及 sklearn 的 GridSearchCV、RandomizedSearchCV 进行超参数搜索；评价指标为 R²、MSE、RMSE。

**📊 数据集**

采用来自 Kaggle 的 "Agriculture Crop Yield" 数据集，原始数据 1,000,000 条记录，去除 231 条负产量后剩 999,769 条样本。

**📈 对比分析**

在 80:20 与 70:30 两种训练/测试比例下，线性模型（LR、岭、Lasso）均获得约 0.913 的 R² 和约 0.501 的 RMSE，表现相近；决策树在 70:30 下略优；随机森林在训练集上表现极佳但测试集明显欠拟合，显示泛化差距。

**⚠️ 局限性**

局限性包括：只考虑了五种回归模型，未探索更高级的集成或提升方法；超参数调优仅针对部分模型；实验未使用交叉验证或重复拆分；缺乏对不同季节或地区数据的外部验证，可能导致过拟合风险。

---

## 519. Meta Models: Models that Model Learning

**arXiv ID:** 2610.37001 | [PDF](https://arxiv.org/pdf/2610.37001v1)

**作者:** Peihang Li `[一作]`, Ziming Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

用冻结的TabPFN预训练表格模型作为“meta‑model”，预测语言模型训练过程的各项指标（最终损失、学习曲线、超参数选择、内部动态、下游性能）。

**💡 创新点**

提出将训练记录视为数值表格，利用零拷贝的TabPFN实现跨任务、跨配置、跨目标的元预测，并结合KAN学习可解释的配置坐标，展示在多个公开训练集上优于专门微调的LLM。

**🔧 技术方法**

TabPFN、Kolmogorov–Arnold Networks (KAN)、零拷贝上下文推理、对比NCPL、XGBoost、参数化缩放法，以及对Marin Hero Run的前瞻性预测。

**📊 数据集**

Marin、StepLaw、Marin‑MoE、DataDecide、Delphi Hyper‑parameter Sweep以及公开的Marin Hero 535B训练记录。

**📈 对比分析**

在终止损失、学习曲线、配置差异预测等指标上与NCPL、XGBoost、参数化法对比，TabPFN在多数基准上MAE最低（如终止损失0.008 vs 0.0109）、曲线误差下降约70%，并在Marin Hero预测中MAE仅为0.0216，明显优于官方缩放法。

**⚠️ 局限性**

仍无法可靠外推到完全未见的配置或数据料理；预测误差虽低但不保证计算节省；需要更多前瞻性实验验证决策收益。

---

## 520. The State and Fate of Multilingual, Contextual Evaluation in the NLP World

**arXiv ID:** 2610.36990 | [PDF](https://arxiv.org/pdf/2610.36990v1)

**作者:** Manan Uppadhyay `[一作]` (Microsoft Research India), Sunayana Sitaram `[通讯]` (Microsoft Research India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 51 个多语言评测基准（共 242 个数据集、219 种语言）进行系统化审计，聚焦覆盖度、代表性与严谨性三大维度。

**💡 创新点**

首次将覆盖度、文化与语料生成方式、注释者多样性、指标适用性与模型泄露风险等因素统一量化，并通过 STAMP 与 DCQ 双重方法揭示跨语言泄露的真实程度，提出基于社区参与、原生构建与去污染的评测框架。

**🔧 技术方法**

使用 Claude 4.6 进行结构化字段抽取与半自动标注，手工校验；采用 STAMP（含可水印的改写检测）和 DCQ（词义扰动识别）进行泄露检测；构建交互式仪表盘展示结果。

**📊 数据集**

涵盖 51 个公开基准（如 XNLI、XQuAD、TyDiQA、AfriSenti、MasakhaNER、IrokoBench、Global MMLU、MILU 等），从而覆盖多语种、不同资源级别和任务类别。

**📈 对比分析**

对 7 大 LLM（如 GPT‑4、Llama‑2、Gemini‑Pro、Llama‑3.1、Mistral‑v0.3、Gemma‑2、Qwen‑3.5）在上述基准上的表现进行泄露检测；结果显示旧基准泄露率高（PAWS‑X 67.9%，MGSM 45.5%），而新基准泄露率低（MILU 0%、INCLUDE 1%、Global MMLU 1.7%）。

**⚠️ 局限性**

局限性包括：仅评估公开基准；注释质量与多样性不一；STAMP/ DCQ 在低资源语言或极端形态学语言上的可靠性受限；泄露检测方法不完全集中于语言中立，需进一步验证；安全性评测仍稀缺。

---

## 521. Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation

**arXiv ID:** 2610.37070 | [PDF](https://arxiv.org/pdf/2610.37070v1)

**作者:** Yueer Pan `[一作]` (Zhejiang University), Yuexi Pan `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究链式思考（CoT）信度评估中答案正确性筛选导致的样本选择偏差，构建结构因果模型并在可编辑算术 DAG 任务上实验验证该偏差。

**💡 创新点**

提出了正确性筛选的因果框架，推导出选择差距 Δ=(A,C)/P(C=1)，系统审计 18 篇相关论文的报告人群，并通过实验直接测量并拆解了差距来源。

**🔧 技术方法**

使用结构因果模型、协方差分解、Cochran–Mantel–Haenszel 检验、bootstrap 置信区间、逆概率加权、符号执行的程序任务生成器等技术。

**📊 数据集**

基于可编辑的算术 DAG、布尔电路和图可达性等程序性任务的合成数据集，约 4000+ 任务实例，其中四个模型在 1000、300、200 等规模上评估。

**📈 对比分析**

与所有轨迹评估对比，计算选择差距；GLM‑5.3 产生 +7.4pp 的正向差距，DeepSeek‑v4‑flash 产生 +6.0pp；整体合并显著 OR=1.70，差距主要由操作计数相关性解释 42.8% 以上。

**⚠️ 局限性**

仅在合成程序任务中验证，缺乏自然语言真实任务；未检测到快捷路径使用，无法验证碰撞偏差；对未观测到的正确性‑合规性关联假设敏感；对真实部署环境的推广有限。

---

## 522. GKS Fully Homomorphic Encryption: Security Analysis

**arXiv ID:** 2610.37061 | [PDF](https://arxiv.org/pdf/2610.37061v1)

**作者:** John Graham `[一作]` (Zero Inc.), Vladimir Shpilrain `[通讯]` (CUNY)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 Gribov‑Kahrobaei‑Shpilrain（GKS）私钥全同态加密方案进行了形式化安全分析，阐述了安全游戏、攻击向量、缓解措施，并给出了性能基准与实证验证。

**💡 创新点**

创新点在于提出基于幂等环的无噪声 FHE 方案，构建了重新抽象（retraction）不确定性与零理想恢复（ZKI）假设，设计了零碰撞保证（ZCG）和双层加密机制，实现了无限乘法深度与原生加密搜索而无需引导（bootstrapping）或线性化。

**🔧 技术方法**

核心技术包括对 S_n 取直和幂等基底的隐匿置换与嵌入、可逆嵌入变换、统计秩（rank）与哈希分析、bijective 编码、SIMD 安全放大、以及多轮理想旋转与零碰撞排除等代数与统计方法。

**📊 数据集**

实证验证使用了多份医疗数据集（如 Diabetes、Parkinson's、Breast Cancer 等）进行机器学习任务，验证了加密算子在真实数据上的精度与误差可控性。

**📈 对比分析**

与传统 BGV/BFV/CKKS 等基于 RLWE 的方案相比，GKS 在无引导、精确解密、无限乘深与高效 SIMD 计算方面表现突出，典型指标为 2.4 µs/槽加密、31 ns/槽乘法、扩展 16 倍，且不需要额外的加密轮。

**⚠️ 局限性**

主要局限在于仅提供条件 IND‑EAV（需满足平衡与秩条件）和未完成的零碰撞保证实现；零理想恢复（ZKI）仍是未证明的假设；在普通 p 版本中，理想恢复后安全性仅为 O(p) 级别；缺乏对 Retraction Recovery Problem 的标准化硬度证明，且对潜在碰撞收集攻击仍需依赖操作层面的缓解。

---

## 523. Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization

**arXiv ID:** 2610.37058 | [PDF](https://arxiv.org/pdf/2610.37058v1)

**作者:** Giuseppe Curci `[一作]` (Okkam), Andrea Zugarini `[通讯]` (Villanova.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了使用单一模型对生成的偏好数据进行 Direct Preference Optimization (DPO) 的效果，并证明了多模型、多样化生成器在偏好数据构造中的必要性。

**💡 创新点**

创新点在于：① 通过理论与实验揭示单模型对生成的偏好数据会引入生成器相关的系统性噪声，导致奖励劫持与性能下降；② 证明仅增加生成器多样性（不需要外部 LLM 判别器）即可显著缓解该问题，提供了一个更经济的偏好数据构建方案。

**🔧 技术方法**

主要技术包括：DPO 的长度归一化损失、Delta Learning 假设的推导、奖励准确率与奖励差距的训练动态分析、样式特征的 PCA 与逻辑回归评估、以及基于 Benchmark 性能的生成器分层抽样。

**📊 数据集**

使用的主要数据集为：AllenAI 的 Olmo‑2‑0425‑1b‑preference‑mix（改写为多模型生成版本）以及多种公开评测基准（TruthfulQA、ARC‑Challenge、IFEval、Math‑500、GSM8K、Human‑Eval、MBPP）。

**📈 对比分析**

比较方法：将单模型对生成的 DPO 数据与多模型+LLM‑as‑judge 数据进行对比，评估在多个基准上的准确率提升。结果显示，单模型对的性能平均比多模型基线低 27% 的奖励准确率，且在多项基准上表现明显不如多模型基线；而通过增加生成器多样性，性能提升约 0.5%–2%（在 IFEval 上提升高达 9%），但仍未达到多模型判别器方案的水平。

**⚠️ 局限性**

限制：实验仅覆盖到 8B 参数规模的模型，未验证在更大规模下是否同样适用；生成器多样性提升方法依赖于基准性能排序，可能在某些任务上无法准确反映生成器质量；此外，模型间对样式特征的敏感性差异可能导致结果不完全可复制。

---

## 524. State estimation for a hybrid rigid-continuum space manipulator system

**arXiv ID:** 2610.37043 | [PDF](https://arxiv.org/pdf/2610.37043v1)

**作者:** Lorenzo Ticozzi `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于因子图的软刚性航天机械臂状态重建框架；

**💡 创新点**

利用几何可变应变（GVS）实现最小阶参数化，兼顾基座与软臂不确定性，并在因子图中同时融合视觉、编码器和基座测量；

**🔧 技术方法**

因子图优化（GTSAM）、GVS框架、马格努斯因子、Levenberg‑Marquardt迭代；

**📊 数据集**

使用SoRoSim V6.3仿真生成的60秒闭环轨迹数据，包含基座姿态、关节角、编码器读数以及基于Blender渲染的彩色标记立体视觉数据；

**📈 对比分析**

与传统单独软臂或刚体估计方法对比，平均末端执行器位置误差仅为系统特征长度的2.3%，相对基座误差降至毫米级；平均求解时间1.23 ms/步，7.27次LM迭代，满足航天实时需求；

**⚠️ 局限性**

缺乏对软臂姿态的直接视觉信息导致中间关节和末端姿态误差相对增大；需要进一步优化视觉配置并验证硬件实现。

---

## 525. Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2610.37041 | [PDF](https://arxiv.org/pdf/2610.37041v1)

**作者:** Bo Yang `[一作]` (Beihang University), Hainan Zhang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种针对强化学习可验证奖励（RLVR）的引导式探索方法——Guided‑RLVR，通过在高熵句子边界注入语义驱动的“steering tokens”来扩展模型的推理轨迹。

**💡 创新点**

创新点在于：①设计轻量化的状态条件控制器，能够在模型不确定时选择语义相关的提示词；②联合优化控制器与策略，使用 log‑perplexity 奖励对探索进行任务对齐；③通过实验验证该方法能显著提升多样性与 pass@k，而不导致生成长度膨胀。

**🔧 技术方法**

技术主要包括：RLVR（GRPO/DAPO）框架；基于隐藏状态的 MLP 控制器；熵阈值触发机制；log‑perplexity 辅助奖励；联合优化策略与控制器；使用 t‑SNE 与轨迹多样性度量。

**📊 数据集**

使用的公开数据集包括：MATH、AIME2024/2025、AMC2023、Code‑R1，以及用于控制器预训练的 OpenThoughts‑114k‑math。

**📈 对比分析**

与 GRPO、DAPO、ENT、REP 等基线比较，Guided‑RLVR 在 Qwen2.5‑7B‑Instruct 与 Qwen3‑4B 两个 backbone 上均取得最高平均 pass@k 分数，尤其在大采样预算下表现最为突出；轨迹多样性（TD/ED）也显著提升，且生成长度并未增加。

**⚠️ 局限性**

局限性包括：需要先验设计和预训练的 steering token 集合；控制器冷启动对性能有一定影响；算力开销相对 vanilla RLVR 增加约12%；方法主要针对数理推理与代码生成任务，未知在更广泛任务上的泛化能力。

---

## 526. GPA-VGGT: Geometry-Preserving Acceleration of VGGT

**arXiv ID:** 2610.37000 | [PDF](https://arxiv.org/pdf/2610.37000v1)

**作者:** Muyeol Sung `[一作]` (Seoul National University), Jongwoo Lim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对VGGT模型的全无训练加速方法GPA‑VGGT，通过保留查询、对键值合并加大小校正、跨帧广播与低维匹配，实现多视角几何任务的快速推理。

**💡 创新点**

创新点在于：①只合并键值并对合并键加入log size校正以恢复注意力质量；②早期层使用紧凑的跨帧K/V集合维持跨视角上下文；③采用高斯低维投影减少匹配开销，并针对敏感层设置不同合并率。

**🔧 技术方法**

使用的技术包括Token Merging（ToMe）、全局/帧注意力交替、FlashAttention风格自定义Triton核、随机高斯投影匹配、跨帧K/V广播以及敏感度自适应合并。

**📊 数据集**

实验数据集：CO3D、RealEstate10K、7‑Scenes、NRGBD；在不同帧数（10–500帧）下评估相机位姿、点图重建、AUC@3、运行时。

**📈 对比分析**

与VGGT、FastVGGT、FasterVGGT、AVGGT等方法比较，GPA‑VGGT在200帧时实现约2×加速（34s对87s），在AUC@3上保持>99%精度，点图重建质量与原模型相当；相比AVGGT(4)速度更慢但精度更高。

**⚠️ 局限性**

局限性：仍比最快的AVGGT(4)慢；在极长序列（>500帧）内显存消耗较大；对合并比例和投影维度的超参选择敏感，若不当可能略微影响精度。

---

## 527. Self-Regularized Flow Language Models

**arXiv ID:** 2610.37115 | [PDF](https://arxiv.org/pdf/2610.37115v1)

**作者:** Theodoros Kouzelis `[一作]` (Archimedes, Athena RC), Nikos Komodakis `[通讯]` (Archimedes, Athena RC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对连续流式语言模型引入噪声-至-干净自蒸馏（Self-Regularized Flow Language Models），通过在去噪轨迹上利用EMA教师对更干净输入的监督，增强训练目标；

**💡 创新点**

创新点在于：①不需要外部预训练模型或改变架构，只在训练阶段加入自蒸馏损失；②提出两种蒸馏方式——隐藏层对齐（Latent Distillation）与分布对齐（Probability Distribution Distillation），并在早期层直接监督；

**🔧 技术方法**

技术包括：连续流式语言模型（FLM、Spherical Flow、Latent Flow）、EMA教师、噪声级对齐策略（Fixed Offset 与 Random Offset）、cosine/交叉熵蒸馏损失、轻量级投影头；

**📊 数据集**

使用公开数据集 OpenWebText（无监督文本生成）和 GSM8K（数学推理）进行评估；

**📈 对比分析**

与三种基线模型（一热FLM、球面流、潜在流）比较，实验表明在 OpenWebText 上 MAUVE 提升 24.3%，在 GSM8K 上准确率提升 28.8%；训练速度提升约 7.7×，并可在 1/64 的采样步数下达到相同或更优性能；

**⚠️ 局限性**

局限性包括：对噪声级策略的敏感性（如 δ=0.8 的 Fixed Offset 在某些模型下效果不佳）；分布蒸馏在某些模型中不如隐藏层蒸馏有效；需要手动调节 λ、层选择等超参数；

---

## 528. RAVEN: A Unified Active SLAM System for Rapid Autonomous Visual Exploration with Aerial Robots under Localization Uncertainty

**arXiv ID:** 2610.37111 | [PDF](https://arxiv.org/pdf/2610.37111v1)

**作者:** Xinyi Chen `[一作]`, Shaojie Shen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为RAVEN的主动SLAM框架，用于在探索过程中同时优化定位精度与探索效率。

**💡 创新点**

核心创新是通过λ权衡参数实现对探索效率与定位精度的动态平衡，并结合感知感知感知路径规划与主动闭环检测。

**🔧 技术方法**

技术上采用EPEE+模拟器、VINS+闭环检测、GPR、D-optimal frontier选择等，并在多摄像头、IMU噪声下评估。

**📊 数据集**

使用四个仿真场景：Corridor、Warehouse、Office、Apartment，内部数据来自EPEE+生成的地图与轨迹。

**📈 对比分析**

与SRE、REAL、ExplORB、FALCON四种最新方法在10次实验中进行比较，RAVEN在探索时间与ATE上实现Pareto前沿，尤其在λ=1时显著降低定位误差。

**⚠️ 局限性**

局限性主要是仅在仿真环境验证，缺乏真实世界实验，并且对参数设置敏感，未在更大规模或动态环境中评估。

---

## 529. A Counterexample to KL1 Completeness and a Complete Disjunctive Calculus

**arXiv ID:** 2610.37107 | [PDF](https://arxiv.org/pdf/2610.37107v1)

**作者:** Damiano Presa `[一作]` (University of Verona), Massimiliano Badino `[通讯]` (University of Verona)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Kant逻辑的输入/输出系统进行形式化与证明，证明原始ESS系统不完整，提出并验证了修正的推理规则，证明新系统在任意规则集上保持完备性（对有限规则集完整），并在Agda中实现机器验证。

**💡 创新点**

提出了单个中间体条件的may‑trans规则的缺陷，改为分支中间体的may‑trans‑disj规则，首次实现了对Kant逻辑的完整系统；同时提供了完整的机理化证明与对现有逻辑的系统性比较。

**🔧 技术方法**

使用形式语义学、推理规则证明、局部可达性与不变性技术，并在Agda证明助手中进行形式化与机理化验证。

**📊 数据集**

无实验数据集，该工作为理论形式化与证明研究。

**📈 对比分析**

通过与原始ESS系统的对比，展示了新规则在推理完整性上的提升；在Agda中验证证明正确性，但未涉及运行时间或实验性能指标。

**⚠️ 局限性**

仅适用于有限规则集的完备性；对无限规则集仍无法保证完备性；原始ESS系统的完整性仍未解决；扩展到带变量和存在量词的输入/输出逻辑仍为开放问题。

---

## 530. On the Sparsity of Direct Preference Optimization: Weight Disentanglement in the NTK Regime

**arXiv ID:** 2610.37085 | [PDF](https://arxiv.org/pdf/2610.37085v1)

**作者:** Kensuke Sasaki `[一作]` (University of Tokyo), Issei Sato `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过NTK线性化和隐式偏置理论，分析Direct Preference Optimization（DPO）的优化动态，证明其产生的参数更新具有稀疏性，并提出了“dual sparsity”（数据层稀疏与参数层稀疏）概念。

**💡 创新点**

创新点在于：①将DPO视为NTK空间中的最大间隔分类器，首次解释DPO产生稀疏更新的本质；②将dual sparsity与权重解耦（weight disentanglement）结合，证明DPO能显著降低跨任务干扰并提升模型合并性能。

**🔧 技术方法**

使用的技术包括：NTK线性化、最大间隔（margin）分析、隐式偏置（implicit bias）理论、梯度特征（gradient feature）分析、参数级和数据级稀疏性评估、以及模型合并（task vector merging）实验。

**📊 数据集**

实验使用的模型与数据集包括：GPT‑2 Small/Medium；7–9B规模的Llama‑3‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2、Gemma‑2‑9B‑it；数据集为Anthropic HH‑RLHF（帮助性/无害性子集）和UltraFeedback（多维度偏好）。

**📈 对比分析**

通过对比DPO与传统SFT在参数稀疏度、支持向量数量、梯度特征稀疏度以及交叉任务干扰（inner‑product & Jaccard）等指标，结果显示：DPO在交叉任务干扰上比SFT低11～35倍，模型合并后性能下降也显著更小。

**⚠️ 局限性**

局限性：①依赖NTK线性化假设，未验证在整个训练过程中是否始终成立；②参数层稀疏性假设依赖于偏好对的响应共享程度，若两响应完全不重叠，该稀疏性可能消失；③理论推导在大规模模型与多样化任务上的普适性仍待进一步验证。

---

## 531. LitUbik: open-source software for human-led systematic reviews with traceable AI assistance

**arXiv ID:** 2610.37181 | [PDF](https://arxiv.org/pdf/2610.37181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 532. Zhengrong-Spatial-Aware 1.0: Advancing 3D Multimodal Intelligence

**arXiv ID:** 2610.37216 | [PDF](https://arxiv.org/pdf/2610.37216v1)

**作者:** Yu Zhang `[一作]` (Beijing Freedo Technology Co Ltd), Wenbo Chen `[通讯]` (Beijing Freedo Technology Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现并验证了两条端到端多模态适配管线：HiSpatial-3B（RGB+度量尺度3D点图）和Qwen3‑VL（基于QLoRA的低秩适配），并在自定义3D空间问答数据集上显著提升准确率。

**💡 创新点**

创新点包括：①将单目几何估计的点图直接注入视觉模型，实现显式几何感知；②采用混合域训练（目标任务+通用视觉指令）以平衡专门化与通用能力；③使用4‑bit QLoRA对8B/32B大模型进行高效适配；④构建统一的数据层、推理与评估接口，提升可复现性。

**🔧 技术方法**

使用技术包括：MoGe单目几何估计、PaliGemma2‑3B视觉语言骨干、HiSpatial-3B的RGB+XYZ输入融合、QLoRA低秩适配、WebDataset序列化、指令化会话格式、混合域采样、自动化评估脚本。

**📊 数据集**

使用自定义3D空间数据集，包含建筑中心图像、彩色边框/掩码、相机位姿、VQA问答对，并收集了原始与风格化两种拆分。

**📈 对比分析**

通过与原始模型基准（1.73%）、HiSpatial-3B（79.89%）以及Qwen3‑VL-8B/32B（分别从20.74%/30.79%提升至45.27%/47.60%）比较，展示了显著的性能提升；在风格化拆分上亦实现从14.9%提升至55.1%。

**⚠️ 局限性**

局限性包括：对目标域过度专门化导致通用能力下降；输出格式对性能影响显著；依赖单目几何估计，受估计误差限制；混合域比例与数据多样性需进一步优化；缺乏对3D资产直接几何使用的实验。

---

## 533. Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

**arXiv ID:** 2610.37201 | [PDF](https://arxiv.org/pdf/2610.37201v1)

**作者:** Jiawei Zhou `[一作]` (Shanghai AI Laboratory), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BiPA（双向归因）工具，对已学习的技能进行后验分析，重放优化器的选择规则，并将每个 episode 记录为二分图，跟踪每轮对话中模型读取的上下文，进而量化额外输入代价、额外工具调用以及规则对行为的具体影响。

**💡 创新点**

首次系统地分解技能优化过程中的选择偏差、成本溢出以及单规则对行为的作用；通过 BiPA 发现学习后的技能往往没有显著提升成功率，却显著增加输入代价；基于这些发现提出四条设计原则来改进技能优化器。

**🔧 技术方法**

使用双向图建模（turn ↔ context 组件），弹性（turn elasticity）分析，四项代价拆分（工具调用、分裂轮次、每轮内容、技能文本）以及选择重放机制；结合统计检验和贝叶斯难度估计来评估选取优势。

**📊 数据集**

Terminal‑Bench 2.1、AutomationBench 三大长周期任务集；使用 DeepSeek‑V4.1‑Flash 与 GPT‑6‑Luna 两个大语言模型；训练集 160 条轨迹，dev 集 30 任务，测试集 18/75/120 任务。

**📈 对比分析**

对比学习后技能与种子技能以及无技能基线，用 avg@3 成功率、输入代价比例 ρ、轮次比例 ρ^turn 与工具调用比例 ρ^calls 评估；结果显示学习后成功率仅提升约 0.1 分，甚至有时下降；输入代价平均提升约 16%；在其它模型或基准上均未实现迁移。

**⚠️ 局限性**

仅为后验测量工具，无法预测单规则的因果效应；只适用于重发完整历史的 harness，对压缩/摘要历史的系统需重新校准弹性；成功率难以按轮次或规则拆分；研究仅覆盖静态优化器，未探讨递归自我改进场景。

---

## 534. Acoustic Signature Optimisation for Bioinspired Flapping-Wing Micro Aerial Vehicles

**arXiv ID:** 2610.37190 | [PDF](https://arxiv.org/pdf/2610.37190v1)

**作者:** Yi Lun Duh `[一作]`, Sophie F. Armanini `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了两种被动几何设计（翼脉络优化与仿生尾缘纹理）来降低拍翼微型空中载体的噪声；

**💡 创新点**

创新点在于：①利用翼脉络布局作为声学优化变量，首次通过非线性模型和贝叶斯优化实现声学最优翼型；②提出添加尾缘“羽毛纹”而非减形切口，既降低噪声又保留升力；

**🔧 技术方法**

采用了基于 von Kármán 平板的动力学模型、Curle 点偶极声源近似、贝叶斯优化（高斯过程）以及水刀切割制造；

**📊 数据集**

使用了自制实验数据集：10 Hz 俯仰-拍动循环的音频、升力测量与三点麦克风记录，包含原始翼、优化翼与尾缘纹理翼共计六种样本；

**📈 对比分析**

通过与原始翼同日同条件对比，优化翼在 80–1000 Hz 频段的点击声降低 7.8 dB，尾缘纹理翼在 300–1000 Hz 频段降低 3–4 dB，且保持或提升升力；

**⚠️ 局限性**

局限在于：机翼机动机制的噪声仍占据 1 kHz 以上频段，整体噪声降低受限；并且声学优化使用的速度变化代理与真实声场存在差距，需进一步验证更高频段与更复杂气动耦合。

---

## 535. Irreducibility Testing of Sparse Polynomials over Integers is NP-hard and Probabilistic Polynomial Time

**arXiv ID:** 2610.37179 | [PDF](https://arxiv.org/pdf/2610.37179v1)

**作者:** Yichuan Cao `[一作]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Jiawei Nie `[通讯]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了整数域上稀疏一元多项式的不可约性判定问题在确定性多项式时间单射归约下是coNP‑hard，并给出了该问题可以在概率多项式时间（PP）以及多项式空间内解决的上界。

**💡 创新点**

创新点在于：①首次通过从 Exact‑3‑CNF 到稀疏多项式的构造，构造出一个能将 3‑UNSAT 与不可约多项式对应、3‑SAT 与可约多项式对应的多项式时间还原；②在上界证明中引入最新的 Cureton–Thorner 的准 Riemann 假设和 Frei–Sofos–Widmer 的有效素理想计数，避免了传统对 GRH 的依赖，获得了无条件的 PP 与 PSPACE 结果。

**🔧 技术方法**

使用的技术包括：组合多项式还原、Filaseta–Ford–Konyagin 的多项式提升与逼近技术、Dedekind ζ 函数的零自由区间、Eisenstein 判据、模根计数统计、以及基于这些统计的确定性空间算法与大多数机器（majority machine）实现。

**📊 数据集**

本研究不依赖实际数据集，全部使用合成的稀疏多项式与 3‑CNF 逻辑公式作为实验对象；所有证明均在理论层面完成。

**📈 对比分析**

与已有的稠密多项式因式分解算法相比，稀疏多项式因式分解已知不具多项式时间可行性。本文证明不可约性判定在 PP 与 PSPACE 内可解，体现了与传统因式分解算法的时间/空间对比，且在实际实现中不需要完整的因式分解。

**⚠️ 局限性**

局限性包括：①上界结果依赖于最新的数值理论定理，若这些定理被证明不成立将影响结论；②对于有限域上的稀疏不可约性是否为 coNP‑hard 仍未解决；③具体实现需处理大指数项的指数表示，虽在理论上可行但在实践中可能遇到指数级别的常数与数值规模。

---

## 536. MLE-World: Predicting Machine Learning Experiment Outcomes with a Learned World Model

**arXiv ID:** 2610.37175 | [PDF](https://arxiv.org/pdf/2610.37175v1)

**作者:** Brian Ko `[一作]` (University of Washington), Sherry Yang `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并评估了 MLE-World，一个针对机器学习工程任务的 4B 语言模型，可在不执行程序的情况下预测程序的终端输出（错误追踪、训练日志和验证分数）并给出置信度评分，从而实现测试时可扩展的 AI 研究代理。

**💡 创新点**

创新点包括：① 在程序级别构建世界模型，用训练日志数据学习直接预测终端输出；② 引入置信度门控，在置信度低时才实际执行程序；③ 设计了五种基于预测的搜索策略（Sequential、Best-of-N、Pairwise、Fusion、UCB），并证明置信度奖励能提升探索效率。

**🔧 技术方法**

使用技术：finetune Qwen3.5‑4B（SFT）以多字段结构化输出；在 AIDE 框架中实现搜索策略；利用置信度评分（p_max）作为 UCB 探索增益与执行门控；GPU 计算与多任务并行评估。

**📊 数据集**

数据集：MLE‑Bench Lite 的 17 个机器学习工程任务，包含约 500k 条程序执行日志（代码、终端输出、异常类别、验证分数），并划分 1,700 条测试样本进行模型评估。

**📈 对比分析**

与更大规模的提示式编程模型（DeepSeek、Qwen‑Coder）和无模型基线（任务统计、最近邻、父记录）比较。MLE‑World 在终端输出匹配率 77.9% 远高于 DeepSeek 41.1%，推理时间仅为提示模型的 1/131,061。使用世界模型指导的搜索策略在 12 小时预算下平均提升 52.1% 的 leaderboard 百分位（比基准 45.4% 提升 6.7%），并通过置信度门控将实际执行时间降低约 30% 同时保持性能。

**⚠️ 局限性**

局限性：仅在训练任务分布内评估，泛化至未见任务仍表现良好但未系统验证；对时间耗时（超时）预测精度最低；模型无法解释错误原因，仍需依赖真实执行；置信度门控可能忽略低置信度但潜在优秀的程序。

---

## 537. Deep Holes and Aperture: Sign Laws for the Empirical Correlation of Lattice-Sampled Random Wave Fields

**arXiv ID:** 2610.37174 | [PDF](https://arxiv.org/pdf/2610.37174v1)

**作者:** Tianhao Guo `[一作]` (Shanxi University), Yuzhi Yang `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究在有限孔径内对高斯随机波场进行格点采样后，偏移重排所产生残差的经验滞后相关性是否始终为正，并给出了深孔极值定律和对应的临界孔径比例。

**💡 创新点**

创新点在于首次将深孔（Voronoi单元最远点）作为极值极端事件的主导因素，推出精确的临界孔径与高斯代理失效概率，并在多种格子形状（方格、六边形、简单立方、BCC）下给出统一的两侧孔径界限及显式常数。

**🔧 技术方法**

使用了高斯随机场理论、极值理论、Gaussian比较和Hanson–Wright尾界、泊松求和、谱分析与格子覆盖理论、适度偏差（moderate deviations）等技术手段，对相关性量子化并完成了极值极限分析。

**📊 数据集**

主要通过数值仿真生成的平面均匀高斯随机波场（利用Bessel函数展开）对不同格子进行样本评估，以验证深孔代理和临界孔径预测；没有使用真实无线信道数据集。

**📈 对比分析**

与单点估计及传统经验相关性对比，深孔代理在失效概率上相对更精确，尤其在高置信度场景下可提升约15倍；实验表明六边形格子在相同覆盖率下失效率显著低于方格格子。

**⚠️ 局限性**

局限性包括：仅适用于连续谱的高斯随机波，未考虑端口耦合与多输出情形；对特定格子与相位偏移有限制；在高可靠性（γ→0、η→0）下需要极大孔径，且在谱密度非均匀时收敛速度不统一。

---

## 538. What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2

**arXiv ID:** 2610.37157 | [PDF](https://arxiv.org/pdf/2610.37157v1)

**作者:** Nabeelah Faumi `[一作]`, Deshan Sumanathilaka `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

演示了如何在LuaLaTeX或XeLaTeX中使用ACL风格文件。

**💡 创新点**

提供多语言文本示例和引用格式，展示了文档排版的灵活性。

**🔧 技术方法**

使用LaTeX、LuaLaTeX和XeLaTeX。

**📊 数据集**

无。

**📈 对比分析**

无实验比较，仅展示格式示例。

**⚠️ 局限性**

仅为样例文件，没有实际实验或数据，缺乏可验证的实验结果。

---

## 539. Conditional Values: The Contingent Foundations of Attitudes toward Artificial Intelligence

**arXiv ID:** 2610.37197 | [PDF](https://arxiv.org/pdf/2610.37197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 540. Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search

**arXiv ID:** 2610.37176 | [PDF](https://arxiv.org/pdf/2610.37176v1)

**作者:** Francesco Benocci `[一作]` (ISTI-CNR), Rossano Venturini `[通讯]` (University of Pisa)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套针对HNSW邻接列表的压缩框架，利用离线图节点重排和SIMD友好的整数压缩实现高压缩率且几乎无查询延迟损失。

**💡 创新点**

创新点在于将图重排与gap编码相结合降低邻接列表熵，并使用Zeta/SVB等低延迟整数压缩器完成在线解码，最终在生产环境中将HNSW图大小减小44%并保持查询性能。

**🔧 技术方法**

技术手段包括离线图重排（EGB/KaMinPar）、gap编码、Elias‑Fano压缩偏移、Zeta/SVB整数压缩、SIMD并行解码，以及在Rust实现的FAISS/hnswlib基础上集成。

**📊 数据集**

实验使用的公共数据集有SIFT 1M（128‑D）、文本段落 8.8M（内嵌/稀疏BERT 30k‑D）、dense encoder 1M 1024‑D，以及eBay 7M 96‑D产品嵌入。

**📈 对比分析**

与未压缩Baseline、Bitpacker、Elias‑Fano、SVB等方法比较，压缩率最高可达2.3×（Zeta），而自研方案（Zeta/SVB+重排）在压缩率1.7–2.1×的同时，查询延迟与Baseline相当或略优；在eBay生产部署中，图内存缩减44%且无可测延迟损失。

**⚠️ 局限性**

局限性包括：需要额外离线重排时间（尤其在高维/高连接度图上成本更高）；对高连接度的HNSW邻接列表重排与整数压缩效果有限；未实现与向量压缩的联合最优设计，未来可进一步探索。

---

## 541. SimplexLoRA: Dynamic Rank Updating via Alternating Minimization with Simplex Projection

**arXiv ID:** 2610.37206 | [PDF](https://arxiv.org/pdf/2610.37206v1)

**作者:** Igor Shalygin `[一作]`, Aleksandr Beznosikov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 n-尺度单纯形约束的重要性权重来自动学习 LoRA 适配器的层级稀疏度的方法，在保持整体参数预算不变的前提下，通过短周期交替优化实现每层的 rank 分配。

**💡 创新点**

创新点在于：① 使用单纯形约束的权重在标准 LoRA 流程中实现自适应 rank 分配；② 通过 Euclidean 投影实现权重更新，避免了动态剪枝或门控带来的运行时开销；③ 与标准 LoRA 完全兼容，训练与推理开销几乎无增。

**🔧 技术方法**

技术手段包括：交替优化（warm‑up、单纯形投影、rank 更新）、Euclidean 投影到单纯形、LoRA 权重的简单缩放与重初始化、AdamW 优化器与两阶段线性/余弦学习率调度、标准 LoRA 初始化和权重衰减正则化。

**📊 数据集**

实验数据集：GLUE（MNLI、SST‑2、CoLA、QQP、QNLI、RTE、MRPC）、XSum、CNN/DailyMail、GSM8K、BoolQ、HellaSwag；实验模型：DeBERTaV3base、Llama 3.1 8B、Qwen3‑8B、BART‑large。

**📈 对比分析**

与标准 LoRA 及多种动态 rank 方法（如 SVD‑LoRA、动态剪枝等）进行基准对比，采用相同 0.24% 参数预算；在大多数 GLUE 任务上平均准确率提升 1–3%，在 low‑rank 场景（r=1,2）提升更明显；在 Qwen3‑8B、BART‑large 等模型上亦保持竞争力。

**⚠️ 局限性**

局限性：① 需要额外设定 K、T 和正则化系数；② 仅在 8B 以下模型上验证，超大模型效果未知；③ 对某些任务（如 CoLA、RTE）提升有限；④ 训练中梯度范数可能被误判导致权重分配不理想；⑤ 过多投影会导致 rank 过拟合。

---

## 542. A Bi-level Co-design Framework for Lunar Monoped with 2D Jump and Landing Control

**arXiv ID:** 2610.37199 | [PDF](https://arxiv.org/pdf/2610.37199v1)

**作者:** Sanjay Ramkumar `[一作]` (Indian Institute of Space Science and Technology), Suraj Kumar `[通讯]` (U R Rao Satellite Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在月球低重力环境下，对单足跳跃机器人进行双层共设计，外层采用差分进化搜索腿部连杆长度，内层使用两种控制器（QP‑TSC与轨迹优化）评估垂直跳跃的能耗，并扩展到二维跳跃与着陆控制，加入重心角动量调节与LQR着陆吸收。

**💡 创新点**

①提出控制器依赖的形态收敛概念，①通过双层共设计框架验证不同控制器对最优连杆尺寸的影响；②首次将轨迹优化与角动量约束结合，用LQR实现二维着陆；③通过交叉评估展示轨迹优化在能耗上显著优于QP‑TSC。

**🔧 技术方法**

差分进化（DE）外层优化；QP‑TSC任务空间控制；CasADi+IPOPT直接协方差轨迹优化；软正则化的能量近似；LQR着陆控制；经验性吸引域分析。

**📊 数据集**

仅使用MuJoCo仿真数据，进行5次独立DE随机种子实验，采集跳跃高度、能耗、着陆姿态等指标，不涉及公开数据集。

**📈 对比分析**

对每个形态，分别使用两种控制器计算成本运输（COT），并交叉评估。结果显示：轨迹优化在其最优形态下COT≈1.28，QP‑TSC最优形态下COT≈1.55；在同一形态下轨迹优化能量更低约17%。二维跳跃实现了无水平位移和0.2 m水平跳跃，峰值俯仰角分别为43°和51°；LQR在约束姿态下能保持稳定，但对着陆姿态误差敏感。

**⚠️ 局限性**

仅为仿真研究，使用理想驱动器；设计变量仅限于连杆长度；未考虑真实的关节阻尼、摩擦及电机速度-扭矩曲线；着陆控制仅为线性LQR，缺乏对大范围姿态误差的鲁棒性。

---

## 543. LiveSearch: Questions as Executable Verifier Code for a Live Benchmark of Professional-Source Financial Search Agents

**arXiv ID:** 2610.37166 | [PDF](https://arxiv.org/pdf/2610.37166v1)

**作者:** Sinuo Wang `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LiveSearch，一个基于可执行验证程序的金融搜索基准，能够每日自动刷新答案而不需要人工干预。

**💡 创新点**

创新点在于将问题与可执行的黄金程序捆绑，冻结问题文本而让答案随时间动态更新，从而消除记忆式错误并实现真正的实时评测。

**🔧 技术方法**

采用工具调用与提取器结合的程序化方法、LLM辅助文本生成与筛选、以及每日调度执行与快照存档技术。

**📊 数据集**

使用了专业金融数据服务（包含96个工具、9个子服务）的实时数据，构建了约3,320道问题，并选取153道用于时间漂移分析。

**📈 对比分析**

在六个主流工具调用模型上进行评测，得分在70.9%–76.7%之间；约70%已服务问题的答案每天会变化，单一答案重用在一天内失效率超过60%。

**⚠️ 局限性**

局限性包括仅覆盖股票、基金、指数、宏观和微观结构等领域，无法涵盖衍生品、外汇、商品等；数据源为专有服务，非公开访问；以及依赖LLM判定器，可能带来偏见与误差。

---

## 544. Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval

**arXiv ID:** 2610.37162 | [PDF](https://arxiv.org/pdf/2610.37162v1)

**作者:** Pasha Abrishamchian `[一作]` (Georgetown University), Ophir Frieder `[通讯]` (Georgetown University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了CoverageBench‑Negate，一个在原始覆盖基准基础上加入对事实否定并编辑相应文档的评估资源。

**💡 创新点**

创新点在于利用对抗性否定构造与配对证据，并提供详细的质量元数据，从而能检测检索器在参数知识与语料相冲突时的行为。

**🔧 技术方法**

采用LLM（如Llama、Gemma等）进行否定生成、文本定位与改写，以及自动化验证，并结合BM25、Qwen3、Jina等检索器进行基线评估。

**📊 数据集**

基于CoverageBench六个子数据集（CAsT 2020、NeuCLIR 2024、RAG 2024、CRUX‑DUC04、CRUX‑MultiNews、RAGTIME 2025），共约63,000对原/否定证据。

**📈 对比分析**

在α‑nDCG@20和StRecall@20指标上，BM25的覆盖率下降约8–15%，而密集检索器下降更大；且在配对得分差异上密集检索器几乎处于随机水平，显示其对否定的区分能力弱。

**⚠️ 局限性**

局限性包括对LLM生成与验证的依赖可能导致语义偏差；未直接验证语料与预训练知识的冲突程度，评估结果受LLM判定一致性影响。

---

## 545. GCSR-OCC: Geometry Calibration and Semantic Routing for Open-Vocabulary Occupancy Prediction

**arXiv ID:** 2610.37149 | [PDF](https://arxiv.org/pdf/2610.37149v1)

**作者:** Bohua Wang `[一作]` (Alibaba Amap), Yang Cai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种单目RGB图像下的开源词汇3D占据预测框架 GCSR-OCC，能够同时预测体素几何占据和语义标签。

**💡 创新点**

创新点包括证据保留的高斯到占据操作 EC-G2O，以及多模查询条件的高斯语义路由 MM-QGSR，以消除尺度膨胀、重叠积累和语义混合问题。

**🔧 技术方法**

采用高斯分割、质量归一化、p-范数聚合、可学习增益、文本对齐的低秩投影、子类查询与平滑聚合等技术。

**📊 数据集**

使用开源语义占据数据集 Occ-ScanNet 进行训练与评估。

**📈 对比分析**

与闭集与开集基线对比，GCSR-OCC 在几何 IoU 达 63.31% 并在语义 mIoU 上提升至 27.44%，且聚合延迟从 19.76 ms 降至 12.88 ms。

**⚠️ 局限性**

局限性在于对细粒度类别的校准仍受限于缺乏稠密 3D 语义标注，导致与部分全监督模型的语义分割性能仍有差距。

---

## 546. MCP-Net: Real-Time Coronary Perception Network and Topological Analysis for Robotic Guidewire Navigation

**arXiv ID:** 2610.37127 | [PDF](https://arxiv.org/pdf/2610.37127v1)

**作者:** Sahar Nasirihaghighi `[一作]` (University of Twente), Stefano Stramigioli `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于仿真驱动的框架，利用多任务冠状动脉感知网络(MCP‑Net)在X射线导管成像中一次性预测血管分割、导丝尖端位置和狭窄位置，并将结果转化为结构化血管图和导航路径，支持机器人导丝导航。

**💡 创新点**

创新点：① 通过CT血管数据生成完整的X射线导管场景并自动标注多任务标签；② 采用多任务网络联合预测血管、导丝尖端和狭窄，避免单任务训练带来的信息碎片化；③ 将像素级输出转化为中心线图和结构化导航路线，提供可直接用于机器人控制的解剖结构化表示。

**🔧 技术方法**

技术：DeepLabV3+骨干+多任务分支；DSNT热力图回归导丝尖端与狭窄三点；形态学骨架化+距离变换构建中心线图；最短路径搜索生成根-狭窄导航路径；CUDA实现实时推理（≈37 FPS）。

**📊 数据集**

数据集：使用ImageCAS冠状CT血管图像376个体，合成1 504张256×256像素的仿真导管成像；每张图像附带血管掩膜、导丝尖端热图及三点狭窄标记，形成完整的多任务训练与评估数据。

**📈 对比分析**

与U‑Net、SegFormer、FPN等多任务架构对比，DeepLabV3+在血管Dice 90.3%、clDice 96%、导丝尖端误差1.09 px、狭窄尖端误差3.82 px，且端到端延迟26.9 ms（37 FPS），在所有指标上均优于其它基线。

**⚠️ 局限性**

局限性：仅在仿真数据上训练与评估，存在仿真‑真实域差距；狭窄定位误差呈长尾分布，精确定位仍具挑战；未在真实临床成像或机器人闭环控制中验证，缺乏临床转化实验。

---

## 547. EEG-Based Decoding of Musical Mode and Melodic Direction Using Empirical Mode Decomposition

**arXiv ID:** 2610.37148 | [PDF](https://arxiv.org/pdf/2610.37148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 548. Mechanistic Failure Modes of Weight Pruning in Transformer Language Models

**arXiv ID:** 2610.37126 | [PDF](https://arxiv.org/pdf/2610.37126v1)

**作者:** Nabeelah Faumi `[一作]` (Informatics Institute of Technology), Deshan Sumanathilaka `[通讯]` (Informatics Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 GPT-2 small 上进行 2×2×3 的结构化剪枝实验，系统评估注意力头和 MLP 神经元在不同重要性选择（幅度 vs 随机）及稀疏率（30%、50%、70%）下对已知 IOI 与 Greater-Than 计算电路的影响。

**💡 创新点**

发现剪枝产生两类全新失效模式：电路零化（head prune）和电路倒置（neuron prune），并揭示不同剪枝策略会掩盖或暴露内部计算错误，提示仅靠准确率/困惑度不足。

**🔧 技术方法**

使用 TransformerLens 进行激活补丁、路径补丁、直接 Logit Attribution (DLA)，以及表示相似度 (CKA、cosine) 等技术进行电路级因果分析。

**📊 数据集**

数据集：人工合成的 IOI 句子集（300 句）与 TinyStories（1000 句）以及 Greater-Than 任务模板。

**📈 对比分析**

与原始 GPT-2 以及不同剪枝模型比较：在 IOI 任务中 30% Magnitude Neuron 剪枝导致 17.7% 准确率与 -2.475 逻辑差；而 50%/70% Magnitude Head 剪枝仅导致约 46‑48% 准确率。困惑度方面，Magnitude Head 剪枝表现最好，但并不能反映电路倒置。

**⚠️ 局限性**

局限：仅测试 GPT-2 small；仅评估两种电路；未考虑层/块级剪枝、梯度或归因导向剪枝、后续微调；合成数据对自然语言任务的迁移有限。

---

## 549. Matrix Samplers and Quantum-Proof Extractors: Their Correspondence and Almost-Optimal Constructions

**arXiv ID:** 2610.37122 | [PDF](https://arxiv.org/pdf/2610.37122v1)

**作者:** Kuan Cheng `[一作]` (Peking University), Ruiyang Wu `[通讯]` (Peking University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了量子证明随机提取器与矩阵采样器之间的等价性，并利用该等价性得到高效的强矩阵采样器和量子安全提取器；

**💡 创新点**

创新点在于提出并证明了量子证明提取器对应于矩阵采样器的结构关系，设计了基于弱设计和多项式评估的构造，实现了随机位消耗与样本复杂度的最优平衡；

**🔧 技术方法**

核心技术包括Hermitian挤压、量子剩余哈希引理、弱设计、有限域多项式评估、混合样本与测量分析、以及先前的矩阵Chernoff/Hoeffding不等式；

**📊 数据集**

本文为理论工作，没有使用实验数据集；

**📈 对比分析**

通过与现有工作（Xun & Zuckerman 2025、Garg等、Xun & Zuckerman 2025）的比较，证明了在相同随机种子长度下样本数从O(ε⁻²log(D/δ))下降到O_α(ε⁻²Λ^{1+α}/logΛ)，并在相同样本数下随机位消耗减少了loglogD因子；

**⚠️ 局限性**

主要局限包括对参数α和ε的取值范围有一定限制，且构造复杂度仍然是多项式级，未实现真正意义上的低复杂度实现，此外在极端小误差或高维度情况下的性能尚待进一步验证。

---

## 550. Prospector: learning a compact graph traversal policy for scientific reasoning

**arXiv ID:** 2610.37121 | [PDF](https://arxiv.org/pdf/2610.37121v1)

**作者:** Haiqian Yang `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在私有技术文献构建的知识图谱上，训练一个紧凑的LLM策略（Prospector）通过逐步调用工具进行图遍历，以链式证据回答科学工程问题。

**💡 创新点**

创新点在于：①将图遍历决策直接训练进模型权重，避免预先固定检索；②使用低秩适配器保持模型体积小；③通过后期决策强化学习和验证器提升答案可靠性；④证明模型在未见领域图谱上零样本迁移。

**🔧 技术方法**

技术包括：LLM+工具交互（ReAct风格）、低秩适配器微调、两阶段强化学习（全程和决策级）、链式证据验证器、工具包装（节点定位、边检索、路径查询）。

**📊 数据集**

数据集为四个领域知识图谱：半导体制造、仿生材料、PFAS与材料属性，每个图谱由数千篇论文自动抽取。

**📈 对比分析**

与八个主流大模型（包括Closed-Book、检索预取子图、工具驱动）比较，Prospector在半导体图上单次贪婪通路准确率0.839，接近顶尖模型0.877；在其他三领域零样本准确率分别为0.803、0.766、0.809；单次推理速度10.7问/分，成本0.57美元/千问，明显快且低。

**⚠️ 局限性**

局限性包括：仅在有预构建工具的图谱上可用；对图结构的深度理解仍依赖训练领域；未覆盖开放式假设生成；若图谱更新需要重新评估模型。

---

## 551. Rethinking Positional Encoding and Interaction Granularity for CTR Prediction

**arXiv ID:** 2610.37140 | [PDF](https://arxiv.org/pdf/2610.37140v1)

**作者:** Xinxun Zhang `[一作]` (Wuhan University), Jiawei Jiang `[通讯]` (Wuhan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对CTR预测，提出TRIP框架，结合目标感知的秩-时间位置编码与时间分组因果交互。

**💡 创新点**

创新点在于：1）低秩秩编码FRE将序列位置压缩为共享基向量；2）目标相对时间编码TRTE以对数尺度学习时间偏置；3）时间分组因果交互TGCI在目标相关时间组内做双向交互、组间因果交互并引入掩码重构与组摘要压缩。

**🔧 技术方法**

使用Transformer块、低秩分解、学习时间偏置（TRTE）、双向+因果注意力、掩码重构、组摘要、MLP分类头。

**📊 数据集**

在四个公开CTR基准上验证：KuaiRec、KuaiRand、TAAC、TaobaoAd。

**📈 对比分析**

与11个主流CTR模型和8种位置编码配置对比，TRIP在AUC上平均提升1.47%（最高4.45%），并在大多数数据集实现最优LogLoss；位置编码迁移实验显示FRE+TRTE在不同Transformer骨干上均能显著提升性能。

**⚠️ 局限性**

局限性包括：①时间分组采用固定对数区间，可能不匹配个体兴趣转移；②组摘要的均值池化可能抑制细粒度行为差异；③在某些数据集（如KuaiRand）对正样本概率估计过高导致LogLoss略逊。

---

## 552. Physics Guided Diffusion Model Based Exposure Interpolation for High Dynamic Range Imaging

**arXiv ID:** 2610.37135 | [PDF](https://arxiv.org/pdf/2610.37135v1)

**作者:** Zhengguo Li `[一作]` (A*STAR), Chaobing Zheng `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理引导扩散模型的曝光插值算法，以解决高动态范围成像中的亮度顺序反转伪影问题。

**💡 创新点**

创新点在于引入了一种新的扩散模型，通过减少噪声数据与目标之间的皮尔逊相关性变化和拟合误差的放大因子，来生成更多的高频信息。

**🔧 技术方法**

使用了物理驱动的组件和基于生成模型的扩散模型，结合了物理引导和生成AI的方法。

**📊 数据集**

使用了高动态范围成像数据集，具体包括420个序列用于训练，40个序列用于验证，60个序列用于测试。

**📈 对比分析**

与现有的基于卷积神经网络的曝光插值算法相比，提出的算法在生成高频信息方面具有固有优势，实验结果表明其性能优于现有算法。

**⚠️ 局限性**

限制在于曝光插值算法的速度问题，未来的工作将集中在减少反向过程的采样成本上。

---

## 553. SFE-O: An Ontology for Knowledge-Grounded AI in Supercritical CO$_2$ Biomass Extraction and Fractionation

**arXiv ID:** 2610.37123 | [PDF](https://arxiv.org/pdf/2610.37123v1)

**作者:** Thomas Pallet `[一作]` (Université de Lorraine, CNRS, CRAN), M{é}lanie Glibusic `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个针对超临界CO2生物质提取的模块化OWL 2本体SFE-O，并通过实例化示例批量验证其一致性和查询能力。

**💡 创新点**

首次将破碎-完整细胞（BIC）模型、CER/FER机制及质量传输计算依赖关系正式化为语义本体，并重用已有化学、量化、时间和推演资源，实现知识可追踪性。

**🔧 技术方法**

采用OWL 2 DL本体建模、PROV-O、ChEBI、QUDT、OWL-Time等外部语义资源，使用HermiT推理器进行一致性检验，并用SPARQL查询验证本体功能。

**📊 数据集**

仅使用单一真实的实验批量（含Dunaliella salina提取过程参数）进行实例化，未使用大规模公开数据集。

**📈 对比分析**

通过本体一致性检验（HermiT）和对CQ5的SPARQL查询展示其可查询性；性能未做大规模基准，只表明在示例数据上能够保持一致并返回正确结果。

**⚠️ 局限性**

局限在于本体未实现数值计算与约束执行（如Sh、k_f等计算需外部服务），缺乏可执行约束（SHACL）和有效扩散系数的通用模型，且示例数据存在若干不一致性需要人工验证。

---

## 554. How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop

**arXiv ID:** 2610.37196 | [PDF](https://arxiv.org/pdf/2610.37196v1)

**作者:** Haojia Gao `[一作]` (Tsinghua University), Xueqian Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在测试时通过世界动作模型（WAM）生成并评估候选动作片段，从而选择最优动作来提升机器人任务完成率；

**💡 创新点**

提出一种去噪计数对照协议（denoised counterfactual protocol）以客观测量可选头间隙（selectable headroom），并基于此构建了失效门控（failure‑gate）选择器，解决了单次连续回放中随机性导致的“优化者诅咒”；

**🔧 技术方法**

使用位可重现的分支、重复继续、保留测试集或acles、公共随机数、结构化动作扰动、线性探测器以及自监督训练的失效门控与排名器；

**📊 数据集**

在MuJoCo模拟器的RoboCasa和LIBERO‑Plus两大基准上，对Cosmos Policy、Flex‑π、Fast‑WAM、Streaming‑WAM四种WAM进行评估；

**📈 对比分析**

与单次连续回放、值头评估、随机偏移等方法比较，发现失效门控选择器在训练任务的闭环成功率从76.9%提升至81.6%（+4.7个百分点），在未见任务上从62.6%提升至69.8%（+7.2个百分点），并且在最易失效状态下提升更显著；

**⚠️ 局限性**

局限性包括：仅在可精确恢复状态的仿真环境下验证；对真实机器人需重新设计；选取的动作扰动仍为手工定义，可能无法覆盖所有场景；选择器需要额外的想象计算，增加推理开销。

---

## 555. VGIQA: Fine-Grained Image Quality Assessment with Visual Grounding Rewards

**arXiv ID:** 2610.37186 | [PDF](https://arxiv.org/pdf/2610.37186v1)

**作者:** Weichang Wang `[一作]` (Xidian University), Nannan Wang `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种能够同时进行细粒度局部与整体图像质量评估的多模态大语言模型 VGIQA。

**💡 创新点**

创新点在于引入视觉定位奖励机制与双阶段训练（冷启动+GRPO），实现了对局部失真区域的精准定位与评价，并首次构建了 VGSR-2K 本地化评估数据集。

**🔧 技术方法**

采用链式推理（CoT）自监督预训练、Group Relative Policy Optimization 强化学习、视觉定位奖励（DIoU+F1）以及多任务联合训练等技术。

**📊 数据集**

使用新构建的 VGSR-2K 数据集，以及标准 IQA 评估基准（KonIQ、SPAQ、KADID、PIPAL、LIVE‑Wild、AGIQA‑3K、CSIQ）进行实验。

**📈 对比分析**

与 Q-Insight、Qwen2.5‑VL‑72B、Qwen3‑VL‑32B 等模型对比，VGIQA 在局部质量评估中 F1 达到 0.3024、DIoU 0.3235，显著优于对手；在失真感知任务上也提升了多项指标。

**⚠️ 局限性**

局限在于对全局评分仍略低于部分 SOTA 方法，且数据集规模有限，未来需扩展到视频质量评估及更广泛的生成与编辑任务。

---

## 556. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training

**arXiv ID:** 2610.37161 | [PDF](https://arxiv.org/pdf/2610.37161v1)

**作者:** Ziyang Zeng `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 WorkFactory 框架，用以构建可执行的专业任务环境，并通过教师演示收集训练工作代理。

**💡 创新点**

创新点在于从目标输出逆向构建信息支持图，明确信息依赖与位置，并通过执行驱动的修复确保环境可执行且信息完整。

**🔧 技术方法**

采用信息级规划、执行驱动修复、LLM 生成与评估、工具调用、监督微调等技术。

**📊 数据集**

构造了 112,686 个覆盖 612 职业的任务集，并收集教师轨迹，用职业层级、技能文档、专业交付物等构成意图库。

**📈 对比分析**

使用 GDPval-AA、JobBench、Workspace-Bench、OmegaUse-OfficeVal 四大专业工作基准进行评测，WorkPilot 系列模型均明显优于基模型，35B 版 Elo 提升至 1565，27B 版达 1612。

**⚠️ 局限性**

局限在于教师模型仅支持文本输入，未涵盖多模态；训练仅为监督微调，未利用强化学习；任务生成与评估仍需人工审核。

---

## 557. MAPLE-Sec: Mobility-Aware Partitioned Language-Model Engine for Roaming IoT Security

**arXiv ID:** 2610.37147 | [PDF](https://arxiv.org/pdf/2610.37147v1)

**作者:** Yuhang Qiu `[一作]` (Jimei University), Por Lip Yee `[通讯]` (Universiti Malaya)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MAPLE‑Sec，一种针对移动 IoT 端点的语言模型安全引擎，能够在端到端上实现实时入侵检测；

**💡 创新点**

核心创新包括：① 激活压缩编码器实现跨边缘节点的推理状态迁移；② 轨迹条件分割调度根据手over时间、电量、上行质量动态调整模型分割点；③ 基于令牌预算的早期退出机制保证毫秒级延迟；④ 通过联合约束优化实现端到端低延迟和能耗；

**🔧 技术方法**

采用了安全调优的 Transformer 编码器、模型分割与边缘协作、早期退出机制、压缩编码器、混合整数优化、SUMO 移动仿真、TensorRT 与 FP16 推理等技术；

**📊 数据集**

使用 CIC‑IoT 2023 与 Edge‑IIoTset 两个公开 IoT 入侵数据集，并在 SUMO 生成的移动轨迹上重放流；

**📈 对比分析**

与 Cloud‑only、EdgeLLM、EdgeShard、Active‑Infer 四个基线对比，MAPLE‑Sec 在 IID 下 F1 达到 0.962（相对 EdgeShard 提升 2.9pp，Cloud‑only 提升 9.3%），平均延迟 23 ms（比 EdgeShard 低 2.5×），盲点率仅 3.8% 而非 100%，能耗 17.4 mJ/流；在非 IID 场景下仍保持优势；ablation 证明各模块贡献；

**⚠️ 局限性**

局限性包括：① 评估仅基于仿真移动轨迹，未使用原生移动流；② 设备端头部在 Jetson Orin Nano 上测试，边缘尾部仍在桌面 GPU，缺乏完整嵌入式验证；③ 激活压缩仅与原始激活对比，未与量化或图像压缩基线比较。

---

## 558. Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning

**arXiv ID:** 2610.37145 | [PDF](https://arxiv.org/pdf/2610.37145v1)

**作者:** Liying Wang `[一作]` (Universitat Autònoma de Barcelona), Joost van de Weijer `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Fisher-Shaped EWC（FS-EWC），在连续学习中通过在任务学习期间对未来参数漂移的 Fisher 敏感度进行最坏情况惩罚来形塑当前任务的解，并在任务结束时使用轨迹平均法构造保留的参数基准，从而在保持历史保护的同时优化未来的知识保持。

**💡 创新点**

创新点包括：①将 Fisher 信息从单纯的参数保护扩展为指导当前解的形状；②在任务学习阶段加入邻域 Fisher 敏感度惩罚；③通过轨迹平均构造更稳健的保留解。

**🔧 技术方法**

技术方法：Elastic Weight Consolidation (EWC)、Fisher 信息矩阵 (FIM) 的经验估计、邻域最坏情况 FIM 追踪、梯度与 Hessian-Vector 产品修正、轨迹平均参数融合、LoRA 低秩适配、不同 FIM 计算与表示方法。

**📊 数据集**

数据集：视觉连续学习基准（CIFAR-100、ImageNet-R、ImageNet-A、DomainNet、FGVC Aircraft、Stanford Cars、CUB-200、Oxford Flowers、Oxford-IIIT Pets、5Datasets），以及语言连续学习基准（AG News、Amazon Reviews、DBpedia、Yahoo Answers）使用 Llama-3.2-1B/3B 模型。

**📈 对比分析**

通过在上述基准上与 EWC、LoRA、EWCLoRA、FS-EWCLoRA 等多种基线对比，使用平均准确率（FAA）、阶段平均准确率（AAA）、后向迁移（BWT）等指标。结果显示 FS-EWC 在所有视觉和语言基准上均显著提升 FAA 与 AAA，尤其在难度较高的 ImageNet-A 与多域 DomainNet 上提升 5–10% 左右，且在不同 FIM 计算与表示方式下保持稳定优势。

**⚠️ 局限性**

局限性：①邻域半径 r 的选择仍需经验调优，过大或过小可能影响性能；②轨迹平均法需要存储多次迭代状态，增加内存开销；③实验主要集中在预训练 ViT 和 Llama 模型，未评估在更大规模模型或无预训练模型下的表现；④理论分析基于二阶 Taylor 展开，可能在高非线性模型中误差增大。

---

## 559. MixSIS: Mixed-supervised instance segmentation

**arXiv ID:** 2610.37138 | [PDF](https://arxiv.org/pdf/2610.37138v1)

**作者:** Christian Wilms `[一作]` (University of Hamburg), Simone Frintrop `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出混合监督实例分割（MixSIS）任务，并设计基线系统 AnnsMixer，在同一图像内混合 mask、box、point、image-level 标注并生成伪掩码，完成实例分割。

**💡 创新点**

① 允许单张图片中自由混合多种标注级别；② 统一生成伪掩码的 AnnsMixer 方法；③ 展示可显著降低标注时间、提升性能的高效标注方案与跨数据集融合。

**🔧 技术方法**

使用 SAM 进行基于几何提示的伪掩码生成；Mask R‑CNN 作为基础实例分割网络；CAM 提取图像级提示；多级标注混合、非极大抑制等流程。

**📊 数据集**

COCO（用于 MixSIS/WSIS 评估）、PASCAL VOC、UVO、Cityscapes、BDD100k、MinneApple、MAD、COCO image‑level 等多种数据集。

**📈 对比分析**

在 MixSIS 上与 Mask R‑CNN、BoxLevelSet、SAPNet、MAL 等基线对比，mAP 在 0.323–0.266 之间，显著优于单标注级别方法；在 WSIS 上与专用方法相当或领先；案例中标注时间降低 90%，性能提升 34%。

**⚠️ 局限性**

对 SAM 伪掩码质量的依赖；未处理跨域适应问题；仅支持四类标注，缺少对更稀疏或特殊标注的通用支持；系统需预训练模型，缺乏实时推理评估。

---

## 560. Multibranch Ensemble Head in Vision Foundation Models for Efficient Cucumber Phenological Stage Classification

**arXiv ID:** 2610.37118 | [PDF](https://arxiv.org/pdf/2610.37118v1)

**作者:** Kostiantyn Khabarlak `[一作]` `[通讯]` (Dnipro University of Technology), Kostiantyn Khabarlak (Dnipro University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并验证了一种多分支集成分类头，利用轻量线性分支与重型 MLP 分支在自监督预训练的 Vision Transformer 上进行随机采样训练，以提升在黄瓜生长阶段分类数据稀缺环境下的准确率与推理效率。

**💡 创新点**

创新点在于引入轻量与重型分支的随机训练与推理时概率空间平均的多分支集成策略，实现了在不增加显著参数或推理开销的前提下提升模型鲁棒性和性能。

**🔧 技术方法**

采用了自监督预训练的 DINOv2/3 Vision Transformer、Dropout、随机分支采样、概率空间聚合以及两阶段 Fine‑tuning 等技术。

**📊 数据集**

使用了 NALEF Food Safety 数据集，该数据集包含 787 张实景黄瓜图片，标注 5 个生长阶段。

**📈 对比分析**

与多种强监督与自监督 Backbone（ResNet、EfficientNet、ViT、ConvNeXt）及不同 head 进行对比，DINOv2 ViT‑S + 多分支头在仅 12.2 GFlops、22.3 M 参数下取得 0.978 Macro F1，匹敌 300 M 参数模型。

**⚠️ 局限性**

局限性包括数据仅来自叶片视角，花果易被遮挡，模型在极端光照、不同品种或环境下的鲁棒性未评估，且验证集极度不平衡。

---

## 561. TopoField: Technical Report for ECCV 2026 SceneFun3D Challenge

**arXiv ID:** 2610.37218 | [PDF](https://arxiv.org/pdf/2610.37218v1)

**作者:** Zhiwei Chen `[一作]` (Shandong University), Yupeng Hu `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两阶段的确定性解码器TopoField，将点级语义预测转换为置信度排序的实例掩码。

**💡 创新点**

创新点在于：①使用测试时增强（TTA）生成语义共识场，消除索引错位和方差；②引入核心-支持阈值分离与基于几何的稀疏图地理距离分配，精准划分重叠实例；③无学习参数的解码过程与Soft NMS并行实现实例排序。

**🔧 技术方法**

主要技术包括：VOLT点云语义预测、13次TTA与缓存融合、类条件阈值筛选、32邻近点的对称半径修剪图、Dijkstra多源地理距离分配、软NMS抑制同类重叠。

**📊 数据集**

使用SceneFun3D数据集（160训练、30验证、30测试场景，共9个功能交互类别）。

**📈 对比分析**

与基线和其他解码策略对比，TopoField在验证集上实现39.67% AP，在官方测试中获得28.286% AP、41.383% AP_50和47.558% AP_25，排名ECCV 2026 SceneFun3D Challenge第二名。

**⚠️ 局限性**

局限性包括：①top‑1覆盖率仅20%，说明高置信度实例仍难捕捉；②仅处理单帧点云，未考虑时间连续性；③无学习参数意味着无法针对特定场景进一步优化；④对极小或高度遮挡的功能部件仍易误检。

---

## 562. KEPT: Conformal Keepouts Kept in Planning for Enclosed-Space Navigation in Degraded Sensing

**arXiv ID:** 2610.37217 | [PDF](https://arxiv.org/pdf/2610.37217v1)

**作者:** Eric Minwoo Kim `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对感知退化（烟雾、雾、尘埃）环境下的无人机/车辆在隧道等封闭空间中进行安全规划与控制，提出了一个基于分级无标签严重程度的 conformal keepout（保持区域）并在规划与命令层都施加一致的刹车距离约束，从而减少进入危险区的风险并提升任务完成率。

**💡 创新点**

创新点：① 采用 severity 估计动态选择保持区域宽度，实现对不同视角下的感知不确定性自适应校准；② 在规划和命令层使用同一保持区域并仅靠刹车距离约束，无需放宽置信水平；③ 通过离线 conformal 误差界定与在线跟踪误差结合，保证在闭环执行时保持安全距离，首次将 conformal 预测与轨迹规划/控制层耦合起来。

**🔧 技术方法**

使用技术：conformal prediction（分层一次侧量化）、网格化障碍衬垫、B样条轨迹规划、基于安全过滤的命令约束（barrier 函数）、实时跟踪误差（tracking margin）以及无标签 severity 估计。

**📊 数据集**

数据集：① 真实地面机器人在室内烟雾环境下收集的 LiDAR 与地面真值（5 份 bag 用于校准/测试）；② 通过光学渲染与自研雾模型的仿真隧道/矿井环境，产生仿真 LiDAR 与雾退化场景。

**📈 对比分析**

比较方法：在同一条隧道轨迹上对 189 条闭环飞行进行多种对照策略（部署模式 A' soft、law‑only、filter‑only、stock planner 等）比较；评估指标为进入 keepout 的次数、达到目标的次数以及最窄通过距离。结果显示 KEPT 进入 keepout 的飞行显著减少（8 次对比 27 次），目标到达率不低于对照且在转移域更高，整体性能提升约 3.2%（置信区间包含 0）。

**⚠️ 局限性**

局限性：仅在单一路径、单雾级别、单规划器和单四旋翼仿真环境下验证；未在线执行自适应 margin；覆盖声明仅针对地图单元，未给出轨迹级概率保证；受证书更新、跟踪误差和保留空间限制影响，未验证在实际地面车辆上的闭环性能。

---

## 563. PipeLink: Contract-Guided Linking of GPU Kernel Components

**arXiv ID:** 2610.37200 | [PDF](https://arxiv.org/pdf/2610.37200v1)

**作者:** Jize Guo `[一作]` (Peking University), Xupeng Miao `[通讯]` (Peking University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出PipeLink，一种基于契约的GPU核组件链接器，能够将独立编译的组件（如加载器、矩阵乘法主循环、后端等）自动链接成完整的mega kernel。

**💡 创新点**

创新点在于使用契约描述组件对资源的需求与保证，自动进行SM资源统一分配与跨组件数据传输生成，避免手工forking，同时保持与单体编译相近的性能。

**🔧 技术方法**

主要技术包括LTO编译、契约检查与匹配、统一SM资源布局求解、自动生成传输代码以及vendor LTO后端的常量折叠。

**📊 数据集**

评测主要在Hopper和Blackwell GPU上进行，使用51种不同配置进行性能验证；未在论文中提及特定机器学习数据集。

**📈 对比分析**

方法是将同一源码一次性编译为单体核与使用PipeLink链接的核进行对比，链接核在51种配置下与单体编译差异不超过0.2%，并在CuTe注意力、MoE训练、Attention核心以及GEMM+SwiGLU等应用场景中分别实现2.16×加速、匹配原始单体、超过FlexAttention、vLLM、Transformer Engine以及cuBLAS。

**⚠️ 局限性**

局限性包括需要组件提供契约信息、对vendor LTO后端的依赖，以及在极端资源竞争或复杂依赖场景下可能出现分配冲突；论文未讨论跨机器分布式场景。

---

## 564. Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning

**arXiv ID:** 2610.37195 | [PDF](https://arxiv.org/pdf/2610.37195v1)

**作者:** Yingjian Chen `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文使用生成式AI工具对论文语言进行润色，并辅助准备图表和编码。

**💡 创新点**

创新点在于将生成式AI技术集成到科研写作和可视化流程中，提升写作效率与可视化质量。

**🔧 技术方法**

采用了生成式AI模型（如GPT系列）进行文本处理，并使用相关工具辅助图表生成。

**📊 数据集**

未公开使用任何特定数据集，研究主要聚焦于技术应用展示。

**📈 对比分析**

文章未给出与传统手工写作或其他AI工具的对比实验，无法评估性能提升。

**⚠️ 局限性**

局限性包括缺乏实验验证、未展示生成结果与人工质量的定量对比，以及对数据隐私与伦理的讨论不足。

---

## 565. INTR++: INterpretable TRansformer with better Accuracy, Interpretability, and Adaptability

**arXiv ID:** 2610.37178 | [PDF](https://arxiv.org/pdf/2610.37178v1)

**作者:** Dipanjyoti Paul `[一作]` (IIT Kharagpur), Wei-Lun Chao `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种轻量级的 Prompt Adapter 模块（PAM），在保持 Vision Transformer（ViT）主干冻结的同时，提升可解释性 Transformer 的分类精度、解释效果与跨域适应性。

**💡 创新点**

创新点在于：1）将 PAM 集成到每个 Transformer 层，利用 Prompt Self‑Attention 与 Prompt Cross‑Attention 对 prompt 进行自适应增强；2）引入可学习门控参数，使 PAM 仅对主干表示做最小残差调整；3）实现了在自然图像与专业图像（医学、显微、遥感）之间统一、可解释的模型架构。

**🔧 技术方法**

技术：ViT（DINO、DINOv2、BioCLIP2）冻结主干；类特定 learnable prompts；PAM 结构（压缩‑非线性‑PSA‑PCA‑投影‑门控）；多头注意力；交叉注意力；分类头与共享投影向量。

**📊 数据集**

数据集：自然图像—CUB‑200‑2011、Stanford Dogs、Oxford Flowers、Medicinal Leaf、Stanford Cars、FGVC Aircraft；专业图像—MRI Brain Tumors、Gallbladder Cancer Ultrasound、Colon Cancer Whole‑Slide Imaging、Aerial Image Dataset（AID）。

**📈 对比分析**

与 Prompt‑CAM、ProtoConcepts、PIPNet、TesNet、ProtoViT 等可解释模型以及 Grad‑CAM、Layer‑CAM、Eigen‑CAM、Attention Roll‑out 等后置解释方法对比；同时与 LoRA、AdaptFormer 等适配器方案比较。结果显示：1）在所有 10 个数据集上均优于 Prompt‑CAM，提升 1–3% 的准确率；2）插入/删除指标显著优于其他方法；3）在不同 Backbone（DINO、DINOv2、BioCLIP2）上保持一致的优势。

**⚠️ 局限性**

局限性：1）仍依赖冻结的 ViT 主干，无法充分利用任务特定的细粒度特征；2）对极端多模态或非视觉任务尚未验证；3）解释性主要基于注意力权重，可能在某些样本中产生偏差；4）需要针对每个目标数据集进行少量训练，适配过程仍不完全自动化。

---

## 566. DomeGS: Perceptually Consistent Background Completion for Unbounded 3D Gaussian Splatting in Virtual Reality

**arXiv ID:** 2610.37172 | [PDF](https://arxiv.org/pdf/2610.37172v1)

**作者:** Anastasiya Pechko `[一作]` (Jagiellonian University), Przemysław Spurek `[通讯]` (Jagiellonian University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种统一的 3D Gaussian Splatting 场景背景完成框架，将 360° 生成模型与高斯球面结合，实现无缝、具有运动视差的虚拟环境背景。

**💡 创新点**

将生成式全景补全直接投影到有限半径的高斯球面，生成可与近景 Gaussians 共享三维空间的统一背景，从而避免传统 skybox 的无视差与颜色不一致问题。

**🔧 技术方法**

采用 3D Gaussian Splatting、360° 扩展扩散模型（如 FLUX）、视差监督、多视图合成与光照一致性约束，以及对背景高斯的径向与面向平面损失。

**📊 数据集**

使用四个户外 3DGS 重建数据集（Buddha、Tower、Garden、Bear）并在 Meta Quest Pro VR 环境中采集用户交互数据。

**📈 对比分析**

通过双因素 4 场景 × 2 技术（SKY 与 skybox）的 36 参与者在 VR 中完成鸟类搜索任务，记录微探测评分、IPQ、偏好、行为指标和自动视觉质量；SKY 在背景一致性、沉浸感、空间存在感和现实感上均显著优于 skybox（p<0.05），且无差异的任务负荷和轻度的赛博症状。

**⚠️ 局限性**

仅在户外开阔场景验证，缺乏对复杂室内/多层结构的测试；未评估实时渲染性能和大规模场景的计算开销；生成模型可能导致颜色或纹理与真实环境的细微不匹配。

---

## 567. From 12 to 200 Demonstrations: How ACT, Diffusion Policy, SmolVLA and $π_{0.5}$ Scale on a Low-Cost Arm

**arXiv ID:** 2610.37170 | [PDF](https://arxiv.org/pdf/2610.37170v1)

**作者:** Simon Cowling `[一作]` `[通讯]` (fysical.ai), Simon Cowling (fysical.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在低成本SO‑101机械臂上，对一个带有干扰物的抓取放置任务，使用225条人机协作演示，分别在12、25、50、100、200条数据子集上训练四种流行的模仿学习策略（ACT、Diffusion Policy、SmolVLA和π_0.5），并在固定十个物体摆放位置上评估1,340条试验，系统记录关节状态和摄像头图像，随后对成功率、失败阶段、启动失败、细粒度性能以及微调配置进行了全面对比。

**💡 创新点**

首次在同一真实机器人和同一任务上实现了多策略多数据量的定量对比，揭示了：①预训练的π_0.5在少量演示下数据效率最高；②ACT从零开始在大数据量下至少能匹敌或超过预训练模型；③启动失败主要源自演示开头的空闲帧，可通过去除 idle lead‑in 解决；④对π_0.5微调时仅训练低秩适配器（LoRA）与全量动作专家的差别可达60个百分点，远大于数据量带来的提升。

**🔧 技术方法**

利用LeRobot开源框架，具体技术包括：Transformer‑based conditional VAE（ACT）、扩散策略（Diffusion Policy）、SmolVLA 视觉‑语言‑动作模型、π_0.5 预训练的 PaliGemma + Flow‑matching 动作专家；低秩适配器（LoRA）与全量权重微调；使用EMA、学习率调度、固定训练步数等传统训练技术。

**📊 数据集**

数据集为225条人机协作演示（每条约17秒，约514帧），其中200条被划分为5个嵌套子集；测试数据为10个固定物体摆放位置，共1,340条实验；此外每个实验均伴随完整的关节、指令动作和双摄像头视频。

**📈 对比分析**

比较方法：对每个策略在每个数据量下的成功率进行 Wilson 区间估计、Fisher 精确检验、对数几率回归（每翻倍数据的odds ratio）以及基于录像的四级失败阶梯分析；对启动失败、首次抓取精度、相邻动作块一致性等连续指标进行定量；对微调配置效果使用配对 McNemar 检验。性能结果：π_0.5 expert‑only 在12/25条演示下成功率可达约70%；ACT 随着数据增加呈S型曲线，在100条演示时成功率93%，在200条时略降至88%；SmolVLA 仅在50条演示时达到最高50%成功率；Diffusion Policy 整体表现差强人意，最高仅约40%成功率。

**⚠️ 局限性**

主要局限包括：仅针对单一抓放任务、单一SO‑101机械臂；每个策略仅使用单一固定训练 recipe，未进行超参数搜索；训练步数固定，导致数据量与训练曝光度共线，无法分离二者影响；种子数量有限，存在随机性波动；评估光照与训练光照不一致；评分未盲法，需人工审计；30秒时间限制可能将慢速成功误判为失败；仅对部分策略进行 EMA、学习率调整等细节评估。

---

## 568. Rènyi-Regularized Flow-Based Variational Inference for Bayesian Calibration

**arXiv ID:** 2610.37169 | [PDF](https://arxiv.org/pdf/2610.37169v1)

**作者:** Exaucè Luweh Adjim Ngarti `[一作]` (Univ. Grenoble Alpes, Inria, CNRS), Arthur Vidard `[通讯]` (Univ. Grenoble Alpes, Inria, CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究变分推断中正态化流对多模态后验的近似问题，提出在ELBO中加入基于Renyi散度的正则化，以降低模式崩溃现象，并在两参数双峰实验中进行验证。

**💡 创新点**

创新点在于将Renyi散度正则化嵌入变分目标，形成鼓励近似后验覆盖先验支撑的机制，并通过自适应梯度归一化动态调整正则化权重，从而在保持后验多模态的同时抑制模式崩溃。

**🔧 技术方法**

使用的技术包括：变分推断、正态化流模型、Renyi散度正则化、梯度归一化与自适应权重调节、以及前向/反向KL、总变差、能量距离等评价指标。

**📊 数据集**

实验数据为合成的两参数双峰后验，包含两种配置：A（相连模式）和B（分离模式）。通过模拟观测、先验、以及HMC产生的基准后验进行比较。

**📈 对比分析**

比较方法：将未正则化的流、混合高斯以及HMC基准作为对照，计算前向KL、反向KL、总变差、能量距离和模式崩溃比例。结果显示：未正则化在B案例中崩溃率高达87.5%；加入Renyi正则化后崩溃率降至25%（或50%），前向KL与能量距离下降，而反向KL和总变差略有上升。

**⚠️ 局限性**

局限性：正则化改变了原始贝叶斯目标，可能引入估计偏差；效果对架构、初始化和优化路径敏感；并非能完全消除模式崩溃，仍需在不同问题中进一步评估。

---

## 569. Physics-Guided Flow Matching: A Generative AI Framework for Global Total Electron Content Forecasting

**arXiv ID:** 2610.37159 | [PDF](https://arxiv.org/pdf/2610.37159v1)

**作者:** Si-Tong Yang `[一作]` (Clemson University), Benjamin C. Martinez `[通讯]` (Utah State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过物理引导的流匹配框架，利用电离层PDE演化构造概率路径，并结合数据驱动与物理速度的自适应融合，预测24小时全局TEC地图。

**💡 创新点**

创新点在于：① 将电离层连续性方程的时间演化直接用作物理流，形成更符合物理规律的概率路径；② 通过空间自适应加权融合物理估计与数据驱动估计，动态平衡两种信息在不同纬度/位置的贡献；③ 在流匹配生成模型中首次将物理引导嵌入到速度场学习与采样过程中。

**🔧 技术方法**

采用流匹配生成模型、物理引导概率路径、O⁺连续性方程得到的物理速度估计、ConvGRU编码历史物理信息、U‑Net构建数据驱动速度、可变权重λ实现空间自适应融合、以及自动回归多步预测。

**📊 数据集**

使用TIEGCM模拟的全年全球TEC数据（72×144网格），以及对应的F10.7、Kp、E×B漂移、O⁺动力学等物理变量。

**📈 对比分析**

与统计模型STGP、深度学习ConvLSTM以及无物理指导的流匹配基线FMB进行对比；PGFM在24小时平均RMSE、MAE、nMAE上分别比ConvLSTM低39.2%、40.4%、42.3%，比FMB低23.8%、24.9%、15.5%，并在所有时点、纬度上保持最小误差。

**⚠️ 局限性**

限制：① 依赖TIEGCM模拟数据，未在实际观测数据上验证；② 物理估计受限于模型分辨率和对高纬度、极光区的描述不足；③ 需要较大计算资源（两块A100 GPU），实时部署仍具挑战；④ 未来物理驱动未能实时获取，仍需进一步研究物理量预测或数据同化。

---

## 570. FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS

**arXiv ID:** 2610.37153 | [PDF](https://arxiv.org/pdf/2610.37153v1)

**作者:** Tom Sanitz `[一作]` (University of Hamburg), Christian Wilms `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FastBEVFusion，一种利用LiDAR和多视角相机输入的实时3D目标检测框架；

**💡 创新点**

通过半对齐BEV空间与单个软动态交叉注意力的轻量融合模块，省去深度估计并提高鲁棒性；

**🔧 技术方法**

使用ResNet（18/50）图像编码器、PointPillars LiDAR编码器、统一BEV投影、单头多头注意力与TransFusion式检测头；

**📊 数据集**

在nuScenes数据集上进行训练与评估；

**📈 对比分析**

与多种实时与高性能融合方法（如BEVDet、TransFusion、FusionPainting、MVP等）对比，FastBEVFusion在mAP/NDS上领先，且推理速度超过50 FPS（17-41ms/帧）；

**⚠️ 局限性**

对深度估计的简化导致对极端遮挡或大尺度深度变化的处理仍有限；

---

## 571. Variational Optimal Control for Diffusion Guidance

**arXiv ID:** 2610.37146 | [PDF](https://arxiv.org/pdf/2610.37146v1)

**作者:** Ruoyu Wang `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种变分控制框架，将采样时引导和基于奖励的模型微调统一到同一 Jacobian 拉回机制上，构造了 Variational Optimal Guidance (VOG) 与 Distilled Variational Optimal Control (DVOC)。

**💡 创新点**

创新点在于：① 用终端奖励与路径空间 KL 惩罚的变分优化得到的 Doob 转换可直接产生对噪声预测的校正；② 通过去噪 Jacobian 把终端梯度拉回到当前状态，实现与 CFG 不同的几何引导；③ 将相同的拉回思想迁移到参数空间，形成端点蒸馏目标，实现在模型参数层面的奖励适配。

**🔧 技术方法**

使用 Gibbs 变分原理、Doob 变换、局部高斯近似、去噪 Jacobian 拉回、端点蒸馏、LoRA 适配、EM‑Averaging 等技术；对 Diffusion 模型进行轨迹控制与参数适配。

**📊 数据集**

在 ImageNet（64²、256²、512² 及 DiT‑XL/2）上评估 VOG；在 Z‑Image‑Turbo + Pick‑a‑Pic（1000 句子）上评估 DVOC；同时使用 50,000 样本的 FID、Precision/Recall、HPSv3++、CLIP、Q‑Align、AES、ImageReward、DeQA 等评测指标。

**📈 对比分析**

相较于 CFG、AutoGuidance、ReCFG 等基线，VOG 在所有 12 组模型/分辨率上均降低 FID（例如 DiT‑XL/2 25 步 FID 由 7.58 降至 4.44），且在更少采样步骤和更大引导强度下保持优势；DVOC 在 20 次 LoRA 更新后使 HPSv3++、CLIP、Q‑Align、AES、ImageReward 等指标均超过未训练模型及其他已训练方法，显示出强劲的奖励对齐效果。

**⚠️ 局限性**

局限性包括：① 需要在每一步计算 Jacobian 或其近似，计算开销较大；② 端点蒸馏需要多步轨迹和额外的梯度回传，训练耗时；③ 对线性子空间假设的理论分析仅在理想模型下成立，实际在非线性去噪器中近似性有限；④ 目前仅在图像生成任务验证，对其它生成模态的推广尚待进一步研究。

---

## 572. Generative Replay for Equilibrium Propagation: Toward Edge-Friendly Continual Learning in Hopfield Networks

**arXiv ID:** 2610.37139 | [PDF](https://arxiv.org/pdf/2610.37139v1)

**作者:** Victor Cook `[一作]` (LIRMM), Marina Reyboz `[通讯]` (CEA List)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现一种基于能量模型的生成式重放方法（EB-Gen），用于Equilibrium Propagation (EqProp) 训练的Hopfield网络在类增量式连续学习中的灾难性遗忘问题。

**💡 创新点**

① 将输入重建损失加入EqProp，使网络能够从能量景观中生成输入样本；② 利用网络自身的放松动力学生成重放样本，无需外部缓冲区或生成模型；③ 在保持硬件友好的同时实现内在的回放机制。

**🔧 技术方法**

Equilibrium Propagation、Hopfield能量网络、混合训练（输入重建）、随机放松（Langevin-like）、类增量式连续学习、与ER、GenReplay、SRC、Joint、FineTune等方法的对比实验。

**📊 数据集**

Split-MNIST、Split-CIFAR10（以及扩展至ImageNet10的10类子集）。

**📈 对比分析**

通过与经验回放（ER）、生成回放（GenReplay）、Sleep‑Like Replay Consolidation（SRC）、联合训练（Joint）以及FineTune进行比较。结果显示：在Split‑MNIST中EB-Gen平均准确率为75.3%（略低于ER 85.6%但高于GenReplay 74.8%）；在Split‑CIFAR10中EB-Gen平均准确率为56.1%（低于ER 76.6%但略优于GenReplay 50.5%）。在ImageNet10上相较SRC提高了约10%与3%的平均准确率。

**⚠️ 局限性**

① 在Split‑CIFAR10上的性能显著低于ER，说明生成样本多样性不足；② 仅在类增量式设置下验证，缺乏对域增量或无任务边界场景的评估；③ 由于未验证硬件实现，实际功耗与面积优势仍需进一步实测；④ 生成的样本非人类可读，难以直接解释模型记忆。

---

## 573. MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Diffusion

**arXiv ID:** 2610.37134 | [PDF](https://arxiv.org/pdf/2610.37134v1)

**作者:** Eduardo Nieto `[一作]` (Fundación Vicomtech), Karen López-Linares `[通讯]` (Fundación Vicomtech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一套三阶段的MRI‑转CT框架，先通过脊柱分割实现解剖匹配生成伪配对，再用SynDiff反向扩散模型进行二维MR‑转CT翻译，最后利用3D GAN后处理网络提升重建体积的切片一致性与解剖连贯性。

**💡 创新点**

创新点在于：①基于脊柱分割与刚性配准的解剖引导伪配对策略，显著提高无配对数据的学习信号；②将对抗扩散模型SynDiff应用于脊柱域，兼顾生成质量与结构一致性；③加入3D后处理GAN，专门解决二维切片重叠时的间隙与连贯性问题。

**🔧 技术方法**

使用的技术包括脊柱分割（TotalSegmentator）、相似度度量（加权Dice）、刚性配准（Euler变换）、SynDiff扩散网络、3D ResUNet GAN、损失组合（对抗、L1、SSIM、感知、切片一致性）、评估指标（MMD、FID、Dice、EMD、JSD）。

**📊 数据集**

数据集来自公开脊柱数据库：MR取自Spine Generic Project、Instrumented Spinal Cord、SPIDER、TotalSegmentator等，CT取自PDDCA、HaN-Seg、HNSCC-3DCT-RT、Spine-Mets-CT-Seg、TotalSegmentator CT、CTSpine1K，共约558份MR和1136份CT。

**📈 对比分析**

与原SynDiff基线及无匹配版本对比，伪配对+扩散阶段将3D Dice从0.666提升至0.739，rMMD从0.135降至0.053，Vertebral FID下降；加3D后处理后Dice进一步提升至0.575，但整体FID略升高。总体来看，解剖精度和切片连贯性显著提高。

**⚠️ 局限性**

局限性包括：①骶骨及后部骨结构仍易出现形态失真；②胸椎表现不稳定，受限于公开数据缺失；③3D后处理虽然消除了切片间断，但引入轻微平滑，影响图像真实感；④缺乏配对MR‑CT原始数据，导致无法做体素级别相似度评估，FID对医学影像的解释有限。

---

## 574. Science-Fiction Transformation as a Design Lens for Visualization Transformation

**arXiv ID:** 2610.37131 | [PDF](https://arxiv.org/pdf/2610.37131v1)

**作者:** Abdulhaq Adetunji Salako `[一作]` (University of Rostock), Christian Tominski `[通讯]` (University of Rostock)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对科幻作品中形态变化的分析，构建了可视化转换的反思框架，探讨转换的目的、对应关系与可见性，并给出四个假设性可视化转换示例

**💡 创新点**

将科幻形态变换视为可视化转换的隐喻，提出“目的、对应关系、可见性”三要素框架，并将对应关系基数作为衡量转换复杂度的指标

**🔧 技术方法**

主要采用理论分析与案例讨论，参考可视化设计原则、动画转化技术（分阶段、叠加、束缚等）作为讨论基础

**📊 数据集**

使用示例合作网络数据（作者、共著关系、属性、空间位置、时间线）作为假设性转换场景

**📈 对比分析**

未进行实验比较，也未报告任何性能指标，仅通过假设场景进行定性讨论

**⚠️ 局限性**

示例有限，未覆盖N:N对应关系，缺乏实证验证，转换设计细节不完整，科幻类例子不具备全面性

---

## 575. Finite Element Modeling of Fracture in Brittle Foods with Physically Consistent Parameter Calibration: A Practical Case Study on Cooled Dark Chocolate

**arXiv ID:** 2610.37124 | [PDF](https://arxiv.org/pdf/2610.37124v1)

**作者:** Patrick Kurzeja `[一作]` (TU Dortmund University), Jörn Mosler `[通讯]` (TU Dortmund University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对冷却黑巧克力进行拉伸、压缩、三点弯曲和板弯曲实验，利用Sharp‑Interface与Phase‑Field两种有限元框架对脆性断裂进行建模、参数识别与验证。

**💡 创新点**

提出一套针对食品脆性断裂的实用建模流程，系统阐明数值框架选择、网格尺寸、初始裂纹、材料参数及其耦合关系，并在有限计算资源下评估两种框架的可行性与互补性。

**🔧 技术方法**

采用基于变分法的Sharp‑Interface Finite Element（含XFEM）和Phase‑Field Finite Element（扩散损伤）两种数值框架，并配合实验数据进行参数识别。

**📊 数据集**

实验数据集为冷却黑巧克力在约15 °C下的拉伸、压缩、三点弯曲和板弯曲测试，获得弹性模量、泊松比、峰值力和裂纹路径等信息。

**📈 对比分析**

通过将峰值力与单一仿真匹配，Sharp‑Interface在1%以内复现峰值力、弹性斜率和裂纹路径；Phase‑Field在适当的宽度l_φ与g_c配置下也可匹配峰值力，但峰值位移和裂纹路径对l_φ敏感。整体计算成本上Sharp‑Interface更低；Phase‑Field在自由裂纹路径预测上更具优势。

**⚠️ 局限性**

局限性包括：Phase‑Field宽度l_φ与网格尺寸耦合，导致g_c无法与材料属性完全解耦；不同目标（峰值力、峰值位移、裂纹路径）对参数要求冲突；仅考虑准静态；未纳入温度、塑性、异质性、动态破碎等因素。

---

## 576. Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution

**arXiv ID:** 2610.37215 | [PDF](https://arxiv.org/pdf/2610.37215v1)

**作者:** Pham Tri Quang `[一作]` (VinUniversity), Vien Anh Ngo `[通讯]` (VinUniversity)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出R2A框架，在测试时利用离线构建的运动原语图检索几何先验，并通过门控融合到冻结的Vision‑Language‑Action（VLA）策略中，提升在视觉 OOD 环境下的执行鲁棒性。

**💡 创新点**

创新点在于：①将训练经历转化为纯几何的运动原语图，彻底剥离视觉噪声；②设计查询依赖的图神经网络，动态定位机器人当前执行进度并检索未来兼容的原语；③使用维度级门控机制在保持本地反应性的同时自适应地纠正 VLA 的空间动作。

**🔧 技术方法**

核心技术包括：离线运动原语图构建（节点为 kinematic 片段，边包含时间与空间邻接）；扩散模型实现 OOD 初始状态的修正；查询依赖的 GNN 进行动态检索；以及动作和谐化门控网络。

**📊 数据集**

主要使用的实验数据集为 LIBERO‑Plus 与 RoboTwin2.0‑Plus（含仿真与真实环境），并在七类系统性扰动（相机视角、初始状态、语言、光照、背景纹理、传感噪声、物体布局）下评估。

**📈 对比分析**

与三种主流 VLA 基础模型（GR00T N1.7、π_0.5、StableVLA）及最新对比基线（VLA‑JEPA、OpenVLA 等）进行对比，R2A 在所有基线上平均提升约10–12% 的成功率，单一扰动场景如机器人初始状态提升可达 40% 以上。

**⚠️ 局限性**

限制包括：检索图缺乏显式场景几何信息，难以处理全新障碍或任务失败；检索仅基于已有训练经验，未覆盖未见过的补救序列；门控网络的训练仅模拟视觉噪声，真实视觉 OOD 诱发的 VLA 预测错误有时可能超出训练分布，导致修正失效。

---

## 577. Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse

**arXiv ID:** 2610.37211 | [PDF](https://arxiv.org/pdf/2610.37211v1)

**作者:** Jiafeng Liang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究视频强化学习验证奖励（RLVR）中出现的捷径放大问题，并提出一种双焦点自蒸馏策略（BSPO）以保持视觉依据。

**💡 创新点**

首次从感知不确定性角度量化RLVR的视觉依赖崩塌，并设计同时关注推理敏感度与视觉鲁棒性的双重蒸馏框架。

**🔧 技术方法**

感知不确定性分析、梯度方向分析、原则指导的信用分配、对抗性视觉扰动、双焦点加权重优化。

**📊 数据集**

Qwen3-VL-4B/8B模型在Video-MME、LVBench、TempCompass、MVBench、MMVU、Video-MMMU等六个视频推理基准上进行评估，使用Video-R1-260k训练数据。

**📈 对比分析**

与GRPO、DAPO（RLVR基线）以及SDPO、VISD（自蒸馏基线）对比，BSPO在所有基准上均取得最高分，尤其在长视频和复杂推理任务上提升显著。

**⚠️ 局限性**

仅在视觉扰动设计和原则生成上存在超参数敏感性，且在更大规模模型或跨模态任务中的泛化仍待验证。

---

## 578. 3D-SceneQ: Empowering 3D LLM With Query-Guided Adaptive Pruning and Multi-Modal Feature Enhancement

**arXiv ID:** 2610.37202 | [PDF](https://arxiv.org/pdf/2610.37202v1)

**作者:** Yongqi Shan `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出3D-SceneQ，一个结合查询引导自适应剪枝和多模态特征增强的3D大型语言模型，用以提升3D场景理解、推理和定位。

**💡 创新点**

创新点包括：① Query-Guided Adaptive Pruning (QGAP) 通过可学习的潜在查询动态筛选任务相关物体，降低噪声与幻觉；② Multi-modal Object-level Feature Enhancement (MOFE) 将多视角2D语义特征投影到3D空间并与3D表示跨模态融合，弥补点云语义不足。

**🔧 技术方法**

技术方法：基于PointNet++的3D编码器、BERT/CLIP的文本与图像嵌入、跨模态注意力、LoRA参数高效微调、可学习潜在查询与自注意力等。

**📊 数据集**

使用的数据集包括ScanNet、3RScan、ScanQA、Scan2Cap、SQA3D，以及Matterport3D进行零样本验证。

**📈 对比分析**

在Scan2Cap、ScanQA和SQA3D三项基准上，3D-SceneQ分别取得78.1 CIDEr、104.7 CIDEr、67.0% EM@1，均超过现有SOTA（分别提升5.7 CIDEr、+17 EM@1等），证明方法在多任务上均有显著性能提升。

**⚠️ 局限性**

局限性：主要针对室内场景，难以推广到户外；缺乏显式时序建模，无法处理动态场景；推理深度有限，未加入链式思考或强化学习等后训练策略。

---

## 579. Tight Bounds for Randomized Generalized $k$-Server on Uniform Metrics

**arXiv ID:** 2610.37192 | [PDF](https://arxiv.org/pdf/2610.37192v1)

**作者:** Junho Hwang `[一作]` `[通讯]`, Junho Hwang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析了在均匀度量空间下广义k-服务器问题的随机竞争比，给出了最优阶数上界与下界。

**💡 创新点**

首次实现了单位距离下的Θ(k log k)随机竞争比和加权距离下的exp(Θ(k))竞争比，并给出了与之前O(k² log k)和Ω(k)或Ω(2^k)的显著改进。

**🔧 技术方法**

采用权重更新、与权重匹配的随机化策略、分阶段聚合与归约、以及随机约束满足问题的组件分析等技术。

**📊 数据集**

论文纯理论，无实验数据集，仅在假设的均匀度量空间（点数可为有限或无限）上证明结果。

**📈 对比分析**

与之前的上界O(k² log k)和下界Ω(k)（或Ω(2^k)）相比，本工作实现了更紧的Θ(k log k)（单位权重）和exp(Θ(k))（加权）竞争比。

**⚠️ 局限性**

算法在实际实现上支持的权重分布支持量极大，缺乏多项式时间实现；对非均匀度量空间仍未给出有效竞争比。

---

## 580. Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration

**arXiv ID:** 2610.37203 | [PDF](https://arxiv.org/pdf/2610.37203v1)

**作者:** Leon Magnus `[一作]` (Technical University Of Darmstadt), Oleg Arenz `[通讯]` (Technical University Of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于Hadamard更新规则的局部空间矩阵记忆（lsmm），实现单一端到端策略在复杂地形上进行自主迷宫探索。

**💡 创新点**

创新点在于将空间索引与Hadamard记忆结合，仅在机器人当前位置附近更新记忆窗口，保持空间关联信息的持久性，并通过辅助迷宫重构目标鼓励记忆保留任务相关信息。

**🔧 技术方法**

使用了强化学习（PPO）与LSTM、SRU、Transformer、MLP等基线对比，构建lsmm、CNN编码器、线性估计器等模块，利用Hadamard记忆更新规则和局部窗口更新。

**📊 数据集**

数据集为在Mujoco中随机生成的3×10^5个迷宫地图，模拟训练后在真实Unitree Go2四足机器人上进行户外迷宫测试。

**📈 对比分析**

与无结构记忆或有限历史记忆方法比较，lsmm在迷宫覆盖率相当，但能显著降低重构误差（三倍）并实现更均匀的访问分布；实验结果表明在真实环境中也能实现目标自由探索。

**⚠️ 局限性**

局限性包括固定大小、与环境对齐的记忆矩阵需在未知规模环境中扩展；依赖外部定位估计，未实现纯内部感知定位；在规模更大或不规则环境中的优势尚未验证。

---

## 581. ReForest: Interpretability Meets Reasoning in Isolation Forest

**arXiv ID:** 2610.37316 | [PDF](https://arxiv.org/pdf/2610.37316v1)

**作者:** Juyoung Seo `[一作]` (Sungkyunkwan University), Moohong Min `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种融合孤立森林与大型语言模型的混合异常检测框架 ReForest，利用统计高效的孤立路径与 LLM 语义推理实现更准确、更可解释的异常检测。

**💡 创新点**

核心创新包括将孤立森林的决策路径转换为自然语言向 LLM 提供结构化提示，基于实例不确定性动态加权的混合评分机制，以及双视角可解释性（统计路径+自然语言推理）。

**🔧 技术方法**

使用了孤立森林、路径抽取与自然语言生成、LLM（如 GPT‑4）语义推理、基于不确定性的动态加权融合，并在前端采用自适应预筛选。

**📊 数据集**

实验覆盖四个真实数据集：医疗（Hepatitis C Virus、Stroke Prediction）与金融（HMEQ Loan、Loan Default）。

**📈 对比分析**

与 IForest、LOF、ECOD、AutoEncoder 等基线相比，ReForest 在 Recall 上提升约21%、F1 上提升约10%，在医疗和金融场景下均取得最高 F1 分数。

**⚠️ 局限性**

局限性包括：对纯数值模式的异常识别效果有限；LLM 计算成本和推理时延仍高；过多路径信息可能导致噪声；不确定性校准和超参数敏感性需进一步改进。

---

## 582. EdgePace: Deadline- and Freshness-Aware LLM Scheduling on Unified-Memory Platforms

**arXiv ID:** 2610.37307 | [PDF](https://arxiv.org/pdf/2610.37307v1)

**作者:** Teng Mei `[一作]`, Yunming Xiao `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 EdgePace，一种面向统一内存边缘设备的 LLM 调度运行时，能够在保障前端任务时间约束的同时，维持 LLM 输出的时效性。

**💡 创新点**

创新点在于：① 通过 Fine Plan 在预填充和解码阶段构建安全的子层暂停点；② 利用 PMU 计数器结合需求与停滞信号实现低成本内存压力感知；③ 结合前端截止余量、LLM 新鲜度窗口及内存压力风险，采用三态调度策略（Bypass/Admit/Hold）实现动态资源分配。

**🔧 技术方法**

核心技术包括：Transformer 子层细粒度分割与量化；硬件 PMU 监测（LLC 读取缺失、后端停滞周期）；基于阈值的风险评估与实时调度算法；对 llama.cpp、MLC‑LLM、TensorRT 等推理后端的动态适配。

**📊 数据集**

使用的工作负载为：① 多摄像头语义反馈（四摄像头 30 Hz 处理）和 ② 基于 MuJoCo 的机器人目标跟踪（RGB‑D 50 ms 周期，40 ms 传感→指令截止），配合 Qwen2.5‑1.5B、Qwen3‑8B、Qwen3‑VL‑8B 等模型。

**📈 对比分析**

通过与无调度（Unmanaged）、MLC‑LLM、TensorRT、MNN、QAIRT 等基线在四台统一内存平台（OnePlus 13、Jetson Orin Nano Super、Thor、AGX Orin）上对比，EdgePace 将前端截止误差降低 24.1 %–96.1 %，同时保留 64.9 %–98.5 % 的 LLM 吞吐量，并在两种案例中显著提升任务质量。

**⚠️ 局限性**

局限性包括：需对不同模型和平台进行 Fine Plan 与阈值校准；分区开销在极低资源设备上可能显著；只针对共享统一内存的设备设计，对分布式或多卡场景未覆盖；对高动态负载下的实时性保障仍有待进一步验证。

---

## 583. Taming Noise Initialization for Zero-shot Video Restoration with Image Diffusion Priors

**arXiv ID:** 2610.37290 | [PDF](https://arxiv.org/pdf/2610.37290v1)

**作者:** Chong Wang `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种仅使用预训练图像扩散模型的零样本视频恢复框架——TINDER，解决了传统方法在噪声初始化和时序一致性方面的不足。

**💡 创新点**

创新点在于：① 通过对降质视频帧进行 DDIM 反演获取保留时序相关性的噪声；② 引入自适应噪声融合策略抑制反演噪声中的降质模式并逼近高斯分布；③ 在采样过程中加入基于测量引导的轨迹注意力以强化帧间一致性。

**🔧 技术方法**

技术包括 DDIM 反演、适应性低通滤波与高斯混合、数据一致性更新（如 DDS、DiffPIR）以及测量引导的时空注意力机制。

**📊 数据集**

使用了 DAVIS、REDS4 和 Set8 三个视频数据集，涵盖高斯模糊、超分辨率（2×、4×）和 70% 随机缺失等恢复任务。

**📈 对比分析**

与现有零样本视频恢复方法（Kwon 等）和多种噪声初始化策略（独立、混合、逐步、相同噪声）进行对比，实验表明 TINDER 在 PSNR、SSIM、LPIPS、FVD 等指标上均表现优于或至少与之持平，且恢复质量与稳定性显著提升。

**⚠️ 局限性**

主要限制在于计算开销显著增加：相较于传统方法，TINDER 的完整流程平均耗时提升约 2.5 倍（20.18 s 对 9.67 s），且需要额外的显存用于轨迹注意力，未来工作需关注加速和模型压缩。

---

## 584. Bayesian Inference of Expensive Simulators: Survey and Comparison of Design Choices

**arXiv ID:** 2610.37284 | [PDF](https://arxiv.org/pdf/2610.37284v1)

**作者:** Šimon Soldát `[一作]` (Czech Technical University in Prague), Václav Šmídl `[通讯]` (Czech Technical University in Prague)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对高成本仿真器的贝叶斯逆问题进行了系统综述，并提出统一的贝叶斯优化框架（BOSIP），对代理变量、代理模型和采集函数等设计选择进行了细致评估。

**💡 创新点**

创新点在于：①将代理变量泛化为可自定义的变换（custom proxy variables），显著提升后验估计效率；②引入非平稳高斯过程作为代理模型，进一步捕捉仿真器输出的异质性；③比较多种采集函数（MaxVar、EIV、IMMD），提供对不同问题适用性的理论与经验指导。

**🔧 技术方法**

技术方法主要包括：贝叶斯优化、Gaussian Process（包括多维和非平稳变种）、自定义代理变量映射、期望方差（EIV）与最大均值差（IMMD）等采集函数的构造，以及 Monte Carlo/HSIC 等数值近似。

**📊 数据集**

实验使用七个基准问题：四个解析问题（AB、Simple、Banana、Bimodal）和三个基于 PDE 的真实问题（SIR、Duffing、Diffusion），涵盖从二维到三维、单峰到多峰、稀疏观测等多样场景。

**📈 对比分析**

通过在相同仿真预算（大约 100 次）下进行 20 次独立运行，比较不同代理变量、代理模型和采集函数的 TV 距离，结果显示：自定义代理变量与非平稳 GP 能显著降低所需仿真次数，MaxVar 通常在单峰问题中表现最佳，而 EIV 在多峰 Duffing 问题中更优；IMMD 效果不稳定。

**⚠️ 局限性**

局限包括：非平稳 GP 的计算成本高、部分采集函数需额外数值积分导致时间消耗；研究主要聚焦低维（≤3）问题，难以直接推广到高维参数空间；缺乏针对批量仿真、并行化的完整评估。

---

## 585. Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation

**arXiv ID:** 2610.37280 | [PDF](https://arxiv.org/pdf/2610.37280v1)

**作者:** Mingzhe Lu `[一作]` (Chinese Academy of Sciences), Hang Su `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建用户自由文本请求的偏好依赖图（PDG），并通过有向路径检索与拓扑路径排名生成推荐列表。

**💡 创新点**

首次显式建模偏好间的层级化限定关系，利用有向无环图指导检索与评分，从而显著提升候选覆盖和精准度。

**🔧 技术方法**

使用 NLP 解析提取偏好原子与视角实例，图构造与循环消除，Serper API 进行有向路径检索，LLM（Qwen‑3.8‑27B）生成查询与候选，基于拓扑路径的折扣评分与对数压缩。

**📊 数据集**

实验基准包含英文 Reddit MovieSuggestions、30 条英文 Book 询问以及 30 条中文 Anime 询问（来自小红书和知乎），同时对照 OCG‑Agent 等公开基线。

**📈 对比分析**

与 OCG‑Agent、Perplexity、Qwen、DeepSeek、MiniMax、GPT‑6‑Astra 等多种模型在 P@10、R@10、NDCG@10 上进行对比；PDG‑F（+Year）在 Movie 上 P@10 提升 13.7%、R@10 提升 19.6%，Book 上 P@10 提升 92.4%，Anime 上 R@10 提升 60.9%，均显著优于对手。

**⚠️ 局限性**

仍受候选检索覆盖、时序偏差与 LLM 解析误差限制；拓扑路径评分仍未达到理想 NDCG 上限，需进一步改进证据分数精度与模型对复杂语义关系的捕捉。

---

## 586. An Interactive Mobile Augmented Reality System for Efficient Virtual Model-to-Patient Registration in Thoracic Surgery

**arXiv ID:** 2610.37271 | [PDF](https://arxiv.org/pdf/2610.37271v1)

**作者:** Emmanuel Ajaero `[一作]` (Concordia University), Yiming Xiao `[通讯]` (Concordia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于手持iPad的移动AR框架，实现无标记、无外部跟踪的胸部手术中虚拟模型与患者的高精度配准；

**💡 创新点**

创新点包括：①通过手持设备的6-DoF“抓取”模式实现交互式粗配准；②利用iPad LiDAR对患者表面进行点云采集并进行自动细化；③设计了基于占据格的“Surface Coverage Score”实时反馈机制，指导点云采集完成足够覆盖以保证配准质量；

**🔧 技术方法**

技术手段包括Unity+ARKit、LiDAR点云预处理、BlazePose人体关键点分割、DBSCAN聚类、FPFH+RANSAC初始配准、点对平面ICP细化；

**📊 数据集**

使用Visible Human Project（男性CT数据）生成三维表面模型与物理PLA 3D打印体做为仿真手术环境；

**📈 对比分析**

通过16名受试者的实验，粗配准平均TRE 12.82 mm，自动细化后平均TRE 3.31 mm，完成时间约4 min；SUS平均84.1，显示高度可用；与无实时覆盖反馈的对照组相比，误差下降但未显著；

**⚠️ 局限性**

局限性包括：①仅在静态模拟人体上验证，未考虑病人解剖变形；②覆盖反馈与无反馈实验顺序固定，可能存在顺序效应；③受试者主要为STEM学生，缺乏临床医生充分评估；

---

## 587. REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic

**arXiv ID:** 2610.37265 | [PDF](https://arxiv.org/pdf/2610.37265v1)

**作者:** Ahmed M. Fetouh `[一作]` (Benha University), Hamada Nayel `[通讯]` (Benha University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个两阶段管线，先使用 GLiNER 提取有害实体，再将实体序列化后与原始提示一起输入 MARBERT 进行七类有害内容分类。

**💡 创新点**

创新点在于将通用 NER 与预训练阿拉伯语 Transformer 结合，并通过结构化实体块显式提供实体线索，从而提升细粒度分类。

**🔧 技术方法**

使用 GLiNER（通用 NER）、MARBERT 预训练模型、实体序列化、Transformer 分类头、交叉熵训练、Hugging Face Trainer 等技术。

**📊 数据集**

基于 ArGuard 2026 共享任务数据集（训练 16,835 条，开发 2,405 条，测试无标注）。

**📈 对比分析**

通过与仅提示、仅实体两种基线对比，并在开发集上取得 macro‑F1 0.886、准确率 0.897；在盲测子任务 1 macro‑F1 0.735、准确率 0.793，子任务 2 macro‑F1 0.754、准确率 0.770。

**⚠️ 局限性**

局限包括：实体提取误差导致信息缺失或噪声；语义重叠导致类别混淆；对目标和意图识别不足；输入长度限制；评估仅基于开发集，缺乏跨数据集泛化验证。

---

## 588. Preserving Separation Logic Specifications through Compilation

**arXiv ID:** 2610.37244 | [PDF](https://arxiv.org/pdf/2610.37244v1)

**作者:** Carmine Abate `[一作]` (Barkhausen Institut), Sebastian Ertel `[通讯]` (Barkhausen Institut)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

研究了一种编译方法，使得源程序的分离逻辑规范在编译后仍保持完整性。

**💡 创新点**

提出了WP-preserving编译技术，并通过资源感知的交互树模拟实现资源和断言的显式对应。

**🔧 技术方法**

采用了交互树、资源感知模拟、Rocq机械化证明技术。

**📊 数据集**

无数据集；仅在形式化模型上验证。

**📈 对比分析**

通过Rocq证明展示方法正确性，未进行运行时性能对比。

**⚠️ 局限性**

仅适用于简化的结构化语言和模块化编译，尚未扩展至实际工业级编译器。

---

## 589. Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents

**arXiv ID:** 2610.37224 | [PDF](https://arxiv.org/pdf/2610.37224v1)

**作者:** Jiahong Xiang `[一作]` (Southern University of Science and Technology), Yuqun Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为Spine的接口与运行时，允许编码代理通过显式声明子任务边界来管理上下文，兼容本地执行与委托；

**💡 创新点**

创新点在于将子任务边界与执行者解耦，同时在运行时维护持久化的执行树（SpineTree），在每次模型调用前按需将已完成子任务的结果替换其详细交互，从而实现结构化上下文压缩并保持前缀缓存；

**🔧 技术方法**

核心技术包括：子任务生命周期命令（Open/Close/Next/Spawn）、持久化执行树结构、基于LR(0)文法的前端解析、即时上下文编译（backend）以及与现有代理平台（Replit、Code-Assistant）的插件集成；

**📊 数据集**

在三大长周期编码基准上评估：CodeContests（持续仓库演进）、Recon (全程序重建)、OpenEnd（开放式工程任务），并使用四个模型（Gemini-1.5-Pro、Claude3、OpenAI GPT‑4o、OpenWeight模型）进行实验；

**📈 对比分析**

与四种现有上下文管理方法（压缩、反射、平面折叠、递归子代理）以及Replit自带的原生代理相比，Spine在CodeContests上提升Score 11.8%并降低32.4%成本；在Recon和OpenEnd上平均提升约10–20% Score/测试通过率，并在多模型、多平台实验中实现约30–70% token减少、45%成本下降；

**⚠️ 局限性**

局限包括：仅在特定长周期任务和模型下验证，仍需评估在更大规模、多模态或分布式环境中的表现；对深度嵌套的子任务依赖Open/Close命令，若代理对接口使用不当可能导致复杂度上升；

---

## 590. Spiking Neuromorphic Control for Periodic Orbits Stabilization in Underactuated Robots: A Greedy Lyapunov-Based Approach

**arXiv ID:** 2610.37310 | [PDF](https://arxiv.org/pdf/2610.37310v1)

**作者:** Maksim Surov `[一作]` (Sirius University of Science and Technology), Oleg Sumenkov `[通讯]` (Sirius University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种事件触发的神经形态脉冲控制器，实现了欠驱动机器人对周期轨道的轨迹跟踪。

**💡 创新点**

将Lyapunov贪婪脉冲触发方法推广到周期线性时变系统，并通过转向线性化将轨道稳定问题转化为混合LTV系统的零解稳定，证明了实用轨道稳定性。

**🔧 技术方法**

转向线性化、混合系统理论、Lyapunov函数贪婪触发、矩阵黎卡提方程求解、B样条插值、事件驱动脉冲控制。

**📊 数据集**

使用Butterfly机器人实验平台，基于其动力学模型和实验采集数据；未使用公开数据集。

**📈 对比分析**

通过实验验证与传统连续控制对比，脉冲控制在噪声、延迟条件下保持良好跟踪，转向变量收敛到小邻域；实验未给出定量性能指标。

**⚠️ 局限性**

理论仅给出局部实用稳定性，缺乏对非线性混合系统鲁棒性分析；实验仅在单控制输入机器人上验证，未探讨多输入或大偏差情况。

---

## 591. Structured Representation Probing for Risk Anticipation in End-to-End Autonomous Driving

**arXiv ID:** 2610.37299 | [PDF](https://arxiv.org/pdf/2610.37299v1)

**作者:** Jiawei Liang `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在冻结的端到端自动驾驶模型内部结构化表示（行人、车道、车辆等）上做风险预警的可行性，并提出了轻量级监控器Structured Representation Probe（SRP）

**💡 创新点**

创新点在于为每个信息源（Actor、Ego、Map）学习独立的风险方向，并使用卡迪纳尔归一化的 log‑mean‑exp 池化聚合信息，同时加入事件类型辅助监督，实现在不改动原始驾驶政策的前提下直接从内部表示产生统一的风险评分

**🔧 技术方法**

使用线性投影、卡迪纳尔归一化 LME 聚合、加权二元交叉熵训练及事件窗口标签监督，构建单个风险读出并通过辅助头对三类事件进行诊断

**📊 数据集**

在 Bench2Drive（220 条路线）与 CARLA Leaderboard 2.0 验证集上评估，使用 2 秒的事件窗口进行训练和测试

**📈 对比分析**

与三种运行时监控器（Temporal Consistency、Task‑Aware Risk、TRFD）对比，SRP 在 DriveTransformer、UniAD、VAD 上均实现了最高的 PR‑AUC、ROC‑AUC、事件召回和警报提前时间，且读取开销低于 3% 的运行时负载

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实道路测试；对不同驾驶策略的泛化仍需进一步验证；只利用内部表示，未结合外部感知与规划信息，可能无法捕捉所有失效模式

---

## 592. GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search

**arXiv ID:** 2610.37267 | [PDF](https://arxiv.org/pdf/2610.37267v1)

**作者:** Zhiheng Fu `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为GradCO的框架，用于文本描述下的人行行为异常检索（TPAS），能够同时保持身份识别的鲁棒性和对细粒度行为的敏感性。

**💡 创新点**

创新点在于：① Behavior‑Aware Mixture‑of‑Experts（B‑MoE）将外观、正常行为、异常行为三条专家流自适应学习；② Conflict Offload Learning（COL）通过为每个文本–图像对生成离线锚点，化解同身份不同行为导致的梯度冲突；③ Hierarchical Intent Alignment（HIA）根据查询意图动态加权聚合专家相似度与特征，消除聚合失真。

**🔧 技术方法**

技术实现包括多专家Transformer门控、对比学习（InfoNCE）、方向多样性正则化、离线锚点生成与使用、层级查询加权聚合、二次交叉熵匹配损失等。

**📊 数据集**

使用的主要数据集有：PAB（TPAS）、Multi‑Weather（PAB的天气扰动子集）、UCC（OOD测试集），以及传统文本‑图像检索基准CUHK‑PEDES、ICFG‑PEDES、RSTPReID、UFineBench（包括UFine6926和UFine3C）进行跨任务评估。

**📈 对比分析**

与现有SOTA方法（如CMP、X‑VLM、RaSa、IRRA、Bi‑IRRA等）对比，GradCO在PAB上R@1提升约1–2个百分点，mAP提升约2–3个百分点；在Multi‑Weather、UCC OOD上分别提升约5–8个百分点；在TIPR基准上保持或略高于最优模型。相对的，模型参数约增加20M，显存占用约+4GB，推理延迟从1.5ms提升到2.3ms，仍满足实时检索需求。

**⚠️ 局限性**

局限性包括：① 需要额外的子专家与锚点计算，导致模型规模和显存占用上升；② 仅在TPAS及其相关检索任务验证，尚未证明对更广泛的多模态或非人类对象检索的适用性；③ 对极端遮挡或多人交互场景下的异常行为检索仍存在挑战。

---

## 593. GameVerse: A Minute-Scale Gameplay Dataset for Long-Horizon Interactive World Modeling

**arXiv ID:** 2610.37257 | [PDF](https://arxiv.org/pdf/2610.37257v1)

**作者:** Kang He `[一作]` (Alaya Lab), Yongtao Ge `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了GameVerse——一个分钟级长时序的游戏视频数据集，提供质量过滤的视频、帧级摄像机轨迹和四层结构化字幕。

**💡 创新点**

首次统一提供分钟级长时序游戏视频、可靠摄像机轨迹和多粒度结构化字幕，显著提升交互式世界模型的长时序一致性。

**🔧 技术方法**

使用场景一致分割、ViPE摄像机姿态估计与验证、多阶段质量评估，以及Kimi‑K2.5视觉语言模型进行多层字幕生成。

**📊 数据集**

基于11,002条来自YouTube的1080p 30–60 FPS游戏视频，处理后得到69.7K个高质量剪辑，共1,655小时。

**📈 对比分析**

通过VBench和WorldScore评估60秒生成，Fine‑tune后模型在时间一致性、几何一致性和视觉质量上均优于现有基线。

**⚠️ 局限性**

仅涵盖游戏环境，未验证跨域泛化；摄像机轨迹与字幕仍有误差；连续片段仅两分钟，无法测试更长时序或闭环交互。

---

## 594. TaskBridge: Technical Report for SIGIR 2026 eCommerce Challenge

**arXiv ID:** 2610.37231 | [PDF](https://arxiv.org/pdf/2610.37231v1)

**作者:** Zhiheng Fu `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种面向任务导向电商查询的重写框架TaskBridge，构建产品语义锚点与词汇表，生成多种候选重写，并根据任务协议挑选单个或多个查询以供BM25检索；

**💡 创新点**

创新点在于：①通过Catalog Semantic Anchoring将产品标题转化为可检索的语义锚点；②提出统一的五种候选重写策略（保留原查询、检索反馈、角色投票、需求导向、产品导向）；③设计协议感知的多查询选择，利用质量与加权Jaccard重叠度的Pareto优化实现自动与交互两种提交模式；

**🔧 技术方法**

技术包括：大语言模型Qwen3.5‑9B生成候选，Qwen3‑VL‑Embedding‑8B检索语义锚点，Qwen3‑VL‑Reranker‑8B进行检索质量代理，BM25作为最终检索器，及多轮检索反馈、角色投票与多查询多样性评估等；

**📊 数据集**

使用2026 SIGIR eCommerce Workshop Query Reformulation benchmark数据集，包含任务查询、产品标题语料以及对应的查询-产品匹配标注；

**📈 对比分析**

与官方原始查询BM25 baseline比较，Open Phase中Automatic模式下NDCG@10提升至0.49（vs 0.42），Interactive模式提升至0.58；Closed Phase中Automatic提升至0.27，Interactive提升至0.31；整体在任务完成度、必需品召回及MAP上均优于baseline；

**⚠️ 局限性**

局限性包括：①语义锚点覆盖有限，导致部分产品需求未被检索；②候选生成受语言模型生成噪声影响，易出现查询漂移；③多查询选择在质量与多样性平衡上仍需更细致的调优；④系统为无训练推理时方案，迁移性和对其他领域的适应性尚未验证；

---

## 595. Peers4U: Designing Learner-Centered Virtual Peers for Children's Multimodal Mathematics Learning

**arXiv ID:** 2610.37238 | [PDF](https://arxiv.org/pdf/2610.37238v1)

**作者:** Zhi Zheng `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究设计了基于大型语言模型（LLM）的虚拟同伴系统（Peers4U），通过学习者的认知画像生成多模态解题工件，并在混合主动式交互中引导学习者进行主题讨论；

**💡 创新点**

创新点在于：①以学习者认知水平为中心的同伴分配与主题策划；②多模态解题工件与可检视的思维过程结合；③混合主动式交互实现的学习者导向讨论；

**🔧 技术方法**

使用技术包括：LLM驱动的自然语言与多模态解题工件生成、认知画像推断、混合主动式对话框架及多模态内容展示；

**📊 数据集**

数据集为22名儿童在数学任务（如求和17）中的初始解答，用于推断学习者认知画像并生成同伴画像；

**📈 对比分析**

通过与persona‑centered基线的对比实验，研究发现Learner‑centered主题策划显著提升学习效果：儿童在讨论组中产生更多解题工件，并在与更多Leading Peers组合时产出更高质量解答；

**⚠️ 局限性**

局限性包括：样本规模有限（22名儿童），仅针对单一数学任务，缺乏长期跟踪验证，且系统对不同学科的泛化能力待进一步评估。

---

## 596. AnalogScout: Simulation-Efficient PVT-Aware Analog Sizing with LLM Agents

**arXiv ID:** 2610.37319 | [PDF](https://arxiv.org/pdf/2610.37319v1)

**作者:** Yichen Long `[一作]` (Fudan University), Keren Zhu `[通讯]` (Fudan University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了AnalogScout，一个利用LLM代理、精英池、解析gm/Id本地尺寸以及全角点验证的模拟效率高的PVT感知模拟电路尺寸框架；

**💡 创新点**

创新点包括：①把已验证设计纳入精英池，让LLM代理通过改进而非重启来节省仿真；②用预计算的gm/Id查找表和解析求解器实现极低仿真本地尺寸；③将全角点验证与全局优化阶段分离，实现资源高效利用；

**🔧 技术方法**

技术实现：LLM代理（ChatGPT风格）+固定评估器（全角点SPICE）；解析gm/Id求解器；精英池与完全匹配缓存；分阶段流程（探索、局部化、重构、精细化、签核）；

**📊 数据集**

数据集：SKY130节点的三种OTA（2阶Miller、2阶折叠共模、3阶NMC）和GF180MCU的Miller OTA；

**📈 对比分析**

与全局CMA-ES、贝叶斯优化及两种基于LLM的基线进行对比；在相同16k仿真预算下，AnalogScout功耗比CMA-ES低43.3%，在7%预算即可匹配CMA-ES最优功耗；在所有OTAs均优于BO和LLM基线；

**⚠️ 局限性**

局限性：局部化阶段对功耗提升不显著；重构阶段增益有限；GF180MCU测试仅涵盖Miller OTA；签核复测中仍有少量饱和和规格超标案例。

---

## 597. A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection

**arXiv ID:** 2610.37314 | [PDF](https://arxiv.org/pdf/2610.37314v1)

**作者:** Sanjana Islam Kasfia `[一作]` (International Standard University), Abir Mahmud Shahariar `[通讯]` (International Standard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种轻量级注意力卷积神经网络，用于二分类皮肤癌检测并结合显著性图实现可解释性

**💡 创新点**

创新点在于将通道注意力机制与自定义轻量化网络结构相结合，同时集成显著性图做模型可解释，显著降低参数量和存储需求

**🔧 技术方法**

使用了Squeeze-and-Excitation注意力模块、卷积层、全连接层、Dropout、Batch Normalization、Adam优化器和梯度显著性图技术

**📊 数据集**

采用公开的HAM10000数据集的二分类子集（训练2637张，测试660张）

**📈 对比分析**

与VGG16、VGG19、ResNet101和MobileNet四个预训练基线进行对比，性能上准确率达到87%，与MobileNet相当，参数量仅1.95M，存储7.45MB

**⚠️ 局限性**

局限性包括仅针对二分类任务、对不同光照和噪声场景的鲁棒性未充分验证、以及显著性图解释的主观性和缺乏定量评估

---

## 598. Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?

**arXiv ID:** 2610.37276 | [PDF](https://arxiv.org/pdf/2610.37276v1)

**作者:** Sichun Luo `[一作]` (University of Hong Kong), Qi Liu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将评估资源在进化搜索与最终选型两阶段进行分配的框架，利用稀疏的、信息量高的反馈来驱动搜索，并在搜索结束后通过热启动的 successive halving 进行目标确认；

**💡 创新点**

创新点在于：①将评估分为搜索阶段的低成本可比反馈与最终阶段的精准确认两类；②使用共享信息核心和固定响应模型实现跨候选的可比性；③在最终确认中重用搜索阶段观察，显著降低总评估成本；

**🔧 技术方法**

主要技术包括：信息量估计（Beta 后验）、低秩多维项目响应模型、共享信息核心的 Gumbel‑top‑k 抽样、热启动的 successive halving、回归校正的差分估计和交叉拟合；

**📊 数据集**

使用四个自然语言理解基准：HotpotQA、HoVer、DROP、IFBench；演化框架为 ShinkaEvolve 与 AdaEvolve；

**📈 对比分析**

与 Skip Eval、全量评估 Full、随机子集 Random、Cascade、SH 等 baseline 进行比较。实验显示，在大约 50% 的评估细胞消耗下，提出的方法在四个基准上的平均测试分数与 Full 相当甚至略高；相比 Random、Cascade 等，性能更优；

**⚠️ 局限性**

局限性包括：需预先进行全评估的 warm‑up 阶段，模型参数固定后可能限制后期适应；共享核心与信息量估计对任务特征敏感；在不同 LLM 或工具配置下的泛化性需进一步验证；

---

## 599. Understanding Everyday Communication Patterns, Stress, and Productivity in Hybrid Work through Egocentric Wearable Audio Sensing

**arXiv ID:** 2610.37270 | [PDF](https://arxiv.org/pdf/2610.37270v1)

**作者:** Tiantian Feng `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于可穿戴设备的自我中心音频感知系统，用于捕捉混合工作环境中的日常沟通模式，并关联到压力和生产力。

**💡 创新点**

创新点在于利用教师-学生知识蒸馏将大规模语音基础模型（如WavLM、Whisper）迁移到轻量级ResNet上，在低维MFCC特征约束下实现前景说话人分离；以及首次在真实办公室与居家场景中长期部署并分析沟通行为与情绪效应。

**🔧 技术方法**

技术包括可穿戴音频传感器、openSMILE实时特征提取、前景/背景说话人分离（ResNet+蒸馏）、混合效应线性模型、MFCC、声学能量/基频特征。

**📊 数据集**

使用了MMCSG、ICSI-Meeting Corpus、EasyCom三组自我中心语音数据集进行模型训练与评估，并收集了16名高校工作人员的4个月混合工作现场数据。

**📈 对比分析**

通过对比基线ResNet与蒸馏后的Distilled-ResNet，在三组数据集上的DER、SC、VAD指标均有1.1–2.8点提升；在低SNR噪声下训练时的噪声增强显著降低了DER，整体性能接近Whisper-Tiny。

**⚠️ 局限性**

局限在于受试者样本量有限（16人），只覆盖高校行政/教师等特定工作角色；依赖受限的MFCC特征，可能影响说话人识别的鲁棒性；且模型未在更大规模多样化环境中验证。

---

## 600. Watermark Basins: Black-Box Detection of Distilled LLMs

**arXiv ID:** 2610.37266 | [PDF](https://arxiv.org/pdf/2610.37266v1)

**作者:** Hongyan Chang `[一作]` (FAIR, Meta Superintelligence Labs), Alexandre Mourachko `[通讯]` (FAIR, Meta Superintelligence Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于黑盒查询的水印检测方法，利用所谓的“水印基坑”几何特征将被蒸馏的学生模型定位到关键特定的参数空间区域。

**💡 创新点**

创新点包括：①将水印视为在参数空间中划出的细小方向，形成持久的基坑；②用固定的探针模型对输出文本做梯度映射，去除通用任务方向后对关键方向进行对齐；③通过与若干伪造键的排名来得到严格的 p 值，实现无参数访问的精确检测。

**🔧 技术方法**

关键技术：探针梯度映射、随机投影至 8192 维、PCA 去除任务子空间、余弦对齐评分、排名检验；训练采用 LoRA 细调、温度采样、不同水印方案（TextSeal、GumbelMax、KGW、SynthID）。

**📊 数据集**

使用了多种数据集：教师与学生训练的 OpenR1‑Math‑220k 逻辑推理轨迹；测试基准 GSM8K、Math‑500；以及不同教师模型（DeepSeek‑R1‑Distill‑Qwen‑14B、gpt‑oss‑120B、QwQ‑32B、Llama‑3.1‑8B）。

**📈 对比分析**

与现有唯一的黑盒标记计数器（token‑level）比较，本文方法在 1% FPR 下 TPR 达到 79%（n‑gram=1）或 45%（n‑gram=3），远超基线 29%/1.3%；在不同教师、温度、键比例、模型规模、压缩、持续 SFT/ RL 等场景下均保持较高检测率，表现出更好的稳健性。

**⚠️ 局限性**

局限性：需要先生成并存储教师在多键下的响应；检测仅在审计域内有效，跨域检测效果下降；对持续大量 SFT 训练会弱化文本层面信号；需要一定量的查询预算；若恶意对抗更复杂的伪造键或改写教师轨迹，检测性能可能进一步受限。

---

## 601. Your Policy Is Secretly a Harness: The Decision Geometry of Agent Interfaces

**arXiv ID:** 2610.37255 | [PDF](https://arxiv.org/pdf/2610.37255v1)

**作者:** Jian Mu `[一作]` (Hong Kong University of Science and Technology), Yao Shu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究语言模型代理在不同“harness”（接口）下的学习与迁移行为，揭示模型在训练时学到的动作偏好与harness紧密相关，并提出通过“ceiling（最佳可达回报）- mismatch（与最优偏差）”分解来解释部署回报。

**💡 创新点**

创新点在于：①将最优策略的对数几率差与harness诱导的相对动作分数对应；②提出利用ceiling与mismatch来量化迁移成本与收益，并给出适配阈值；③证明仅靠固定检查点的harness评估会误导；④通过采样动作估计mismatch并预测回报变化。

**🔧 技术方法**

采用KL正则化控制、软贝尔曼方程、策略梯度（post‑training）、semantic action接口抽象、基于教师模型的匹配估计、采样动作的KL估计等技术。

**📊 数据集**

使用MiniWoB++（网页交互任务）与Terminal‑Bench Lite（编码代理任务）作为实验数据集，并在Qwen3.5-2B/9B模型上进行训练与评估。

**📈 对比分析**

方法上对比冻结权重与适配后的检查点在不同harness下的任务成功率、返回、ceiling与mismatch；实验结果显示harness改变会导致冻结模型性能骤降，但在适配后可恢复并超过原始模型；在MiniWoB++上post‑training提升成功率至85%，适配后恢复到19%；在Terminal‑Bench Lite上，ceiling‑mismatch差值能精准排序不同harness的准确率。

**⚠️ 局限性**

局限性包括：需要已知最优策略或教师模型才能计算ceiling与mismatch；采样动作估计的准确性受样本量限制；仅在有限数量的harness与任务上验证，尚未验证在更广泛场景的普适性；对动态或开放动作空间的评估仍具挑战。

---

## 602. GraphRAGProv: GraphRAG Meets How-Provenance

**arXiv ID:** 2610.37237 | [PDF](https://arxiv.org/pdf/2610.37237v1)

**作者:** Wadie El Amrani `[一作]` (Independent), Luis Galárraga `[通讯]` (INRIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建基于文档的 RDF 图，使用 LLM 生成 SPARQL 并在 GraphDB 上执行，实时记录答案及其“如何性”多项式，形成可视化工作台供用户跟踪答案与原始文本的对应关系。

**💡 创新点**

首次将每条 RDF 语句的出现与其确切文本片段用 RDF‑star 记录，并在查询重写时生成多项式 provenance，使答案来源、计算过程与原始证据一一对应，提升系统可解释性。

**🔧 技术方法**

采用 DeepSeek‑V4‑Flash 进行三元组抽取与证据定位，GraphDB 存储 RDF‑star 图，Rust 实现的 NPCS 进行查询重写和如何性来源计算，前端使用 React+FastAPI 交互。

**📊 数据集**

在 2WikiMultiHopQA 与 MuSiQue 共 400 条多跳 QA 案例上进行评估，分别包含 200 条组合问题与 200 条可答问题。

**📈 对比分析**

通过对比答案返回率、provenance 可解码率以及完整文本追踪率进行评估，结果显示 2Wiki 约 97% 的案例能完整追踪，MuSiQue 约 59%，证明系统能在多数场景下提供可追踪的答案和来源，但在某些案例中因实体不一致或查询不完整导致失败。

**⚠️ 局限性**

受限于监督设置（仅使用已知支持段落），未检索外部文档；LLM 的抽取和查询生成仍可能出错；对未返回答案的情况无法提供 provenance；大规模应用的性能与可扩展性未做实验。

---

## 603. AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems

**arXiv ID:** 2610.37228 | [PDF](https://arxiv.org/pdf/2610.37228v1)

**作者:** Rui Tian `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了统一的 AutoRSE 评估框架，针对六个工业级推荐任务对三大 LLM+harness 组合进行大规模实验，并对代理的轨迹进行细粒度分析与 idea injection 实验。

**💡 创新点**

①首次提出面向 RSE‑Agents 的统一评测与轨迹级分析方法；②发现当前代理主要靠工程优化，缺乏算法创新；③通过外部 idea 注入验证想法生成是核心瓶颈。

**🔧 技术方法**

使用 LLM Agents（Claude Opus 4.8、GPT‑5.5、DeepSeek V4 Flash）与三种 harness（Claude Code、Codex、Terminus‑2），结合 AutoRSE 框架、LLM Judge 进行实验分割与策略识别，以及 idea 库注入。

**📊 数据集**

六个工业级推荐任务：HSTU、TIGER、IntSR（Amazon Beauty / Kindle）、WuKong（MovieLens Latest）、RankMixer（Criteo 5M）和 OneTrans（KuaiSAR v2）。

**📈 对比分析**

通过隐藏测试标签评估 NDCG@10 / AUC，与基线对比得到平均相对提升 16.1%；对大多数任务实现显著改进；轨迹分析显示 91.7% 的实验为工程优化，算法创新与性能提升呈正相关。

**⚠️ 局限性**

核心限制在于代理缺乏高质量算法想法生成，主要聚焦参数调优与模型集成，导致算法创新不足；在复杂任务（如 IntSR）表现不佳。

---

## 604. Sharpness-Aware Muon: Matrix-Aware Optimization with Momentum-Guided Perturbations

**arXiv ID:** 2610.37225 | [PDF](https://arxiv.org/pdf/2610.37225v1)

**作者:** Zhiwei Zhai `[一作]` (Chinese University of Hong Kong), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种对 Muon 优化器的 sharpness‑aware 变体（MuSAM 和 MoMuSAM），并通过理论分析和实验验证其在训练稳定性与泛化性能上的优势。

**💡 创新点**

创新点在于：①将 SAM 风格的局部最坏情况扰动与 Muon 的矩阵正交化更新结合，首次构建了矩阵意识与 sharpness-awareness 的统一框架；②利用 Muon 的动量矩阵本身作为稳定的扰动方向，既消除了额外的梯度评估，又提高了随机线性化的稳定性，形成 MoMuSAM。

**🔧 技术方法**

采用的技术包括：矩阵正交化（Polar/Newton–Schulz 近似）、SAM 风格的一阶线性化扰动、动量引导的扰动生成、随机线性化稳定性分析以及一阶驻点收敛理论。

**📊 数据集**

实验数据集：Fashion‑MNIST（MLP）、CIFAR‑10 / CIFAR‑100（CifarNet / ResNet‑18）以及 OpenWebText（NanoGPT‑124M GPT‑2）。

**📈 对比分析**

与 SAM、MSAM、VASSO 及原始 Muon 等基线对比，MuSAM 在大多数任务上已超越基线；MoMuSAM 在泛化精度、收敛速度和对超参（ρ、η）的鲁棒性方面均优于 MuSAM 和所有 SAM 风格方法，表现出更好的性能-效率折中。

**⚠️ 局限性**

局限性：①仍需要手动调节扰动半径 ρ 和学习率 η，尽管鲁棒性更好但对极端值仍敏感；②实验主要集中在中小规模模型，对极大模型或其他任务（如语义分割、强化学习）的适用性尚未验证；③理论分析聚焦于一阶驻点收敛，缺乏对更高阶性质或长期稳定性的深入研究。

---

## 605. BenchMeUp: Building Software Engineering Benchmarks from Natural-Language Specifications

**arXiv ID:** 2610.37283 | [PDF](https://arxiv.org/pdf/2610.37283v1)

**作者:** Islem Bouzenia `[一作]` (CISPA Helmholtz Center for Information Security), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用多代理LLM系统自动化生成可执行的软件工程基准，从研究者提供的自然语言规范出发完成实例搜集、构造、验证与打包。

**💡 创新点**

1) 在一阶段生成共享搜索规则与构建计划，统一不同代理的搜索与构造策略；2) 采用并行LLM代理并通过摘要共享经验，避免重复搜索；3) 在干净的容器中回放验证实例的可重现性，保证接口一致性。

**🔧 技术方法**

使用Claude Opus 5（或deepseek‑v4‑flash）生成搜索规则与构建计划；deepseek‑v4‑flash负责实例构造脚本生成；多代理架构（search agents、construction agents）并行执行；Docker容器进行回放与打包；YAML规范保存规则与计划。

**📊 数据集**

主要基于GitHub公开项目与提交历史；实验覆盖四类任务：程序修复、问题复现、编译器自动调优、静态分析警告修复；共构造200+实例，对比Defects4J、SWE‑bench等现有基准。

**📈 对比分析**

与通用LLM Codex（仅靠代码生成）进行对比；对构造出的实例分别运行8种现有技术（RepairAgent、ReProAgent、BOCA、CodeCureAgent等）；实验结果显示：BenchMeUp构造的196/198实例可被技术使用，96%可用；180/198实例满足所有七项质量判定，Codex仅10/200；成本约0.79 美元/实例，平均构造时间36 分钟/实例。

**⚠️ 局限性**

1) 依赖LLM的准确性，弱模型会导致实例丢失或构造失败；2) 回放仅验证可重现性，无法保证实例本身的“正确性”或“真实缺陷”；3) 目前仅支持GitHub数据源，未覆盖其他生态；4) 需要人工验证关键实例，难以完全自动化；5) 高并发代理调用与API成本对资源有限的实验室构成挑战。

---

## 606. Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability

**arXiv ID:** 2610.37292 | [PDF](https://arxiv.org/pdf/2610.37292v1)

**作者:** Tim Menzner `[一作]` (Coburg University of Applied Sciences), Jochen L. Leidner `[通讯]` (Coburg University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了多标签一致性评估框架MLA，使用平均配对汉明一致性并结合不同基线模型来衡量多标签标注的一致性。

**💡 创新点**

核心创新在于将汉明一致性与标签级Fleiss/Scott系数精确分解，证明其与Krippendorffα的等价性，并提供条件最小一致性诊断和多基线灵敏度分析。

**🔧 技术方法**

采用数学推导、Monte Carlo模拟、合成数据生成和Python实现，对汉明距离、期望一致性等统计量进行计算与评估。

**📊 数据集**

实验以4条句子、3名标注者、31个偏见标签的小样本为案例，此外在模拟中随机生成不同标签数、样本量和标注者数量的合成数据。

**📈 对比分析**

将MLA与增广κ、引导F1、MASI、A_m等传统IAA指标进行对比，结果显示不同基线下的值差异显著，MLA提供更具解释力的诊断，而在极端流行度下池化基线表现低，AC1和PABAK可产生较高值；模拟表明指标对流行度和一致性水平高度敏感。

**⚠️ 局限性**

局限性包括假设完整平衡的二值标签设计、忽略标签间依赖、对缺失值处理有限；汉明距离不捕捉标签互斥或层级关系；实证案例样本量极小，模拟结果汇总不提供置信区间。

---

## 607. RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport

**arXiv ID:** 2610.37281 | [PDF](https://arxiv.org/pdf/2610.37281v1)

**作者:** Dongyue Li `[一作]` (Nagoya University), Fei Ma `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种检索特权的自我蒸馏框架RPSD，用于提升序列推荐的性能。

**💡 创新点**

创新点在于将跨用户行为证据与token级信用分配结合，通过时序最优传输检索并用自我蒸馏实现细粒度信用分配。

**🔧 技术方法**

采用LLM+强化学习GRPO、时序最优传输、语义检索、token级自我蒸馏等技术。

**📊 数据集**

在MovieLens‑1M、Netflix、MicroLens三个公开数据集上进行实验。

**📈 对比分析**

与SFT、GRPO、LLM基线及多种对照模型对比，RPSD在HR@3/5、NDCG@3/5上均显著提升，达到SOTA水平。

**⚠️ 局限性**

主要限制包括对检索质量的依赖、跨域泛化受限以及训练过程的计算成本。

---

## 608. Fast Approximate Line Nearest Neighbor Search

**arXiv ID:** 2610.37256 | [PDF](https://arxiv.org/pdf/2610.37256v1)

**作者:** Jiadong Xie `[一作]` (Chinese University of Hong Kong), Yingfan Liu `[通讯]` (Xidian University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种线增强近邻图（Line‑Enhanced Proximity Graph, LPG），通过在原有近邻图中加入有限的“pair edges”，实现对线（无限线）近邻查询（LANN）的高效近似检索。

**💡 创新点**

创新点在于：
- 通过图平滑与层次划分、商图（quotient graph）共同生成候选区域对；
- 引入支持度分数（J）与方向多样性分数（ν、α）自动选择 pair edges；
- 在不扩展向量维度、保持原始邻接度限制 M 的前提下，仅通过少量额外边实现跨区域跳转，显著提升查询覆盖率。

**🔧 技术方法**

核心技术包括：
- 基于 HNSW 的原始近邻图；
- 图平滑（邻域加权均值）与中位数划分构造平衡层次；
- 商图与图跳数约束 κ 用于筛选无短路路径的区域对；
- 线性/非线性评分（g、h、S）评估区域对的分离度与 secant 支持；
- 线性/非线性边分数选取与原始边替换策略；
- Beam search 结合 pair edges 的局部与跨区探索。

**📊 数据集**

使用的数据集包括：
- 6 个公开视觉检索数据集（尺寸从 15k 到 1M，维度 128 或 512）；
- 两个面部图像照明变化数据集（Multi‑PIE、LFW‑1），通过 PCA 嵌入生成线查询；
- 两个 GAN 隐空间检索数据集（StyleGAN 的 e4e 嵌入，维度 512），用语义方向构造查询线。

**📈 对比分析**

与四类基线比较：
- Andoni 等人提出的理论 LANN；
- 直接在原 HNSW 上用线距离的 Beam search；
- 通过二次升维后在 HNSW 中检索；
- 全量暴力检索。
实验显示：
- 在 Recall@10=0.99 的目标下，LPG 对全尺寸数据集实现 QPS 提升 1.44–4.23 倍，且在 Recall@10/100 达到 1.0；
- 在 512 维子集上，LPG 在相同 Recall 下 QPS 1.56–1.85 倍；
- 与升维方法相比，QPS 最高可提升 976×、1656×，且索引空间仅为升维方法的 2–3% 。

**⚠️ 局限性**

局限性与未来工作：
- 仍需针对高维数据（>512）进一步验证；
- 对于非常稀疏或非线性子空间查询，pair edges 的效果有限；
- 参数设置（M、p、κ、r 等）对性能影响显著，需自动调优；
- 当前实现仅支持线查询，扩展到平面或更高维子空间仍是挑战。

---

## 609. TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment

**arXiv ID:** 2610.37243 | [PDF](https://arxiv.org/pdf/2610.37243v1)

**作者:** Erwei Hou `[一作]` (Beijing University of Posts and Telecommunications), Yue Hu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个名为TCM‑DiagCommit的新基准，用于在阶段性证据下评估中医证候辨别，并引入终端行动评分。

**💡 创新点**

创新点在于将传统的静态最终标签任务拆解为三层评估：候选标签区分、证据更新与诊断承诺，且提供了随机、邻近、全词汇候选梯度、固定标签扰动与缺失证据等多样化任务场景。

**🔧 技术方法**

主要技术包括基于ZY‑BERT的候选集生成与语义邻居排序、分阶段信息掩蔽、JSON预测契约和多维度指标（top‑1/3、增益、错误终端率、期望成本）等。

**📊 数据集**

使用的数据集是公开的TCM‑SD/ZY‑BERT，包含54,152条临床记录，提炼出1,834个源案例并扩展为23,842行评估样本，覆盖148种证候标签。

**📈 对比分析**

通过对五个零射击DirectLLM模型（DeepSeek‑flash、DeepSeek‑pro、MiniMax、Mimo、Qwen）在随机10、硬10、硬20、全148等候选集上进行评估，随机10的top‑1准确率在63–66%，硬20约24–30%，全148约15–22%；但在承诺任务中错误终端率高达62–74%。

**⚠️ 局限性**

局限性包括：标注分布可能与真实临床流行率不符；评估受固定提示与API版本影响；未进行临床医生级别验证；以及缺乏多模态或交互式诊疗流程。

---

## 610. Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

**arXiv ID:** 2610.37242 | [PDF](https://arxiv.org/pdf/2610.37242v1)

**作者:** Saiful Khan `[一作]` (Science and Technology Facilities Council), Afrah Farea `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建将 OData 元数据与 BPMN 业务流程链接的 typed property graph，提升 LLM 代理在多步服务调用中的可靠性。

**💡 创新点**

创新点在于设计了一种 hybrid retriever，先用 dense embedding 找到种子实体集，再沿业务流程链进行有限扩展，并在检索结果中提供完整的参数约束（value‑help、导航 cardinality 等），从而显著减少无效调用。

**🔧 技术方法**

采用了类型化图结构、稠密嵌入检索、图数据库（FalkorDB）查询、ReAct 代理以及 mock HTTP 验证层等技术。

**📊 数据集**

主要使用 OData‑Proc 基准数据集（42 个 OData 服务、14 条 BPMN 参考流程、288 条查询）以及 SOCBench‑D（110 条 OpenAPI 查询）进行评估。

**📈 对比分析**

在 OData‑Proc 上，Hybrid retriever 在 process‑chain 查询中 Recall@3 提升 5.7 分，Agent 在 81 条多步测试查询中的完成率从 10.3% 提升到 38.7%；在 SOCBench‑D 上无显著提升，验证提升仅依赖业务流程信息。

**⚠️ 局限性**

局限性包括对 45 条手工编写的 activity‑EntitySet 关联边的依赖、实验使用 mock 验证器导致的 circularity、仅适用于公开业务流程层的系统，以及整体完成率仍较低。

---

## 611. When Is a Relation Worth Trusting? Manipulation Costs, Exposure, and the Trust Trap in Multi-Relational Graph Learning

**arXiv ID:** 2610.37239 | [PDF](https://arxiv.org/pdf/2610.37239v1)

**作者:** Vincess Dongmo `[一作]` (SYNTHI-AI), Astride Melvin Fokam Ninyim `[通讯]` (SYNTHI-AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多关系图欺诈检测中各关系的信任度，基于攻击成本与关系暴露度，提出度门控（degree gating）理论和曝光下界，给出了关系硬化与选择的决策方法。

**💡 创新点**

创新点包括①首次将攻击成本与关系信息交互建模，揭示度门控在部分节点孤立且攻击无法打开门时优于选择/加权；②提出一阶曝光下界，帮助确定最需硬化的关系；③为线性模型提供最优攻击动态规划，并为非线性模型给出可验证的区间；④提出成本感知的关系选择协议-D，能在多关系场景下自动放弃或加固关系。

**🔧 技术方法**

技术手段包括：理论分析（阈值、阈值结构、曝光下界、线性/非线性模型的最优攻击 DP）；多关系CSBM框架；R‑GCN、树模型（HGB‑Graph、CARE‑GNN等）；攻击方法有精确攻击、贪婪攻击、基于梯度的攻击；对抗验证与有保证的区间评估。

**📊 数据集**

使用的数据集有：合成的多关系CSBM；公开的真实图：YelpChi、Amazon Review、FDCompCN、S‑FFSD、Tolokers；并进行了属性共享扩展（Aug）和丰富关系数的合成实验（Rich）。

**📈 对比分析**

通过对比清洁训练的图模型与图无关基线（LR、MLP、HGB‑X）以及对抗训练模型，在不同成本场景下评估对抗最坏AUC。实验显示：①线性模型在一定比例的设置下易陷入信任陷阱；②-D协议在多数情况下与图无关基线持平或更优，尤其多关系时可显著提升；③对抗训练未能完全消除陷阱，-D通过成本规划取得更好鲁棒性。

**⚠️ 局限性**

局限性包括：成本设定为场景假设而非真实测量；实验数据主要为公开数据，非线性模型的最坏AUC为上界；需要标注的验证集；对不确定成本的规划仅给出经验性方案；验证假设为 i.i.d.；属性共享扩展未能带来收益。

---

## 612. HDVS: semi-supervised semantic segmentation via heterogeneous dual-branch voting supervision

**arXiv ID:** 2610.37236 | [PDF](https://arxiv.org/pdf/2610.37236v1)

**作者:** Yongqi Shan `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种异质双分支投票监督框架 HDVS，用于半监督语义分割。

**💡 创新点**

创新点在于将 Transformer 与 CNN 两种网络结构联合，利用特征通信模块和基于置信度的投票机制来提升伪标签质量，减少确认偏差。

**🔧 技术方法**

采用跨伪标签监督、特征通信模块 (FCM)、投票模块、数据增强、EMA 等技术。

**📊 数据集**

在 PASCAL VOC 2012 与 Cityscapes 两大公开数据集上进行评测。

**📈 对比分析**

与 SOTA 方法对比，HDVS 在不同标注比例下 mIoU 提升 0.7–2.9 点，显著优于多种基准方法。

**⚠️ 局限性**

局限在极少标签（1/16）场景下，投票与特征通信可能放大噪声，导致边缘或光照变化区分辨率下降。

---

## 613. Asynchronous Verifiable and Complete Secret Sharing: From Design Patterns to System-Level Trade-offs

**arXiv ID:** 2610.37234 | [PDF](https://arxiv.org/pdf/2610.37234v1)

**作者:** Joachim Bruneau-Queyreix `[一作]` (Univ. Bordeaux), Laurent Réveillère `[通讯]` (Univ. Bordeaux)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对现代一拍异步可验证与完整秘密共享（AVSS/ACSS）协议进行了统一的设计空间分析，并在公共 Rust 框架下对七个代表协议进行系统级性能评估。

**💡 创新点**

创新点包括：①首次将大量 AVSS/ACSS 文献映射到共通的五个设计维度（通信模型、验证模型、恢复策略、批处理/打包、乐观 vs 可靠路径）；②构建了可插拔的实现框架，实现协议间的 apples-to-apples 比较；③在实验结果中揭示了设计选择如何决定延迟、吞吐、消息量、带宽和 CPU 负载，并给出了基于设计的部署指南。

**🔧 技术方法**

使用的技术主要包括：统一的 Rust 实现框架、可靠广播、信息分发、哈希/对称/零知识证明、Batched/Packings、Bracha‑style 发送/回声/就绪、One‑Sided Vote、Grid'5000 物理集群实验、基准工作负载（单秘密与批量秘密）。

**📊 数据集**

数据集与实验环境：在 Grid'5000 Nova 集群上运行 20 台相同机器（Xeon E5‑2620 v4, 64 GB RAM, 10 Gbps 以太网），实验规模从 10 至 100 方，批量大小 1–50，秘密为 32 字节；使用公开论文中的参数构造七个代表协议的实现。

**📈 对比分析**

比较方法：在同一框架、同一网络、同一参数下进行两种场景（optimistic 与 faulty‑dealer stress）测量完成延迟、峰值吞吐、消息数、字节数和聚合 CPU 时间。结果表明：乐观路径在无故障时延迟低但故障时差距大；批处理/打包在大批量时显著降低每秘密成本但单秘密延迟上升；可靠传播或 transcript 完成的协议在故障下保持低成本；公共密钥/多项式承诺的协议在 CPU 上成为主瓶颈。

**⚠️ 局限性**

限制：实验仅在低延迟局域网（Grid'5000）中完成，未覆盖 WAN 或异构部署；实现仅覆盖代表协议，可能未包含各论文的全部优化；故障模型为单一 Byzantine 处置，未探讨完整的 Byzantine 调度或适配硬件加速；因此结果主要用于相对比较而非绝对部署预测。

---

## 614. Subspace Clifford Networks

**arXiv ID:** 2610.37226 | [PDF](https://arxiv.org/pdf/2610.37226v1)

**作者:** Zhongping Ji `[一作]` `[通讯]`, Zhongping Ji

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Subspace Clifford Networks，用分组子空间交互替代原始 CliffordNet 的全局循环滚动，并实现了融合的 CUDA 核，显著降低中间通道扩展。

**💡 创新点**

创新点在于：① 在固定小组尺寸 k=2,3,4 内一次性计算完整的内积与外积，去掉了稀疏滚动导致的高通道扩张；② 通过重计算反向的融合核实现了 17 倍速度提升和 50% 内存减少；③ 将此交互模块嵌入多分辨率层级化网络，进一步提升准确率。

**🔧 技术方法**

使用技术包括：几何代数中的 Clifford 乘积、分组子空间交互、1×1 线性投影、SiLU 激活、可重计算反向传递、层级化多分辨率结构，以及 C++/CUDA 模板化的高效实现。

**📊 数据集**

实验数据集为 CIFAR-100，使用 200 轮训练、AdamW、cosine 学习率衰减和标准数据增强。

**📈 对比分析**

与原始稀疏滚动的 CliffordNet（|𝒮|=2、5）以及基于层级化的 Shift‑Based CliffordNet 进行对比；Subspace 模型在 12 层层级化时仅用约 1.09M 参数、训练时间 42 分钟，却能获得 80.25% Top‑1 ；在 24 层时更是取得 81.21% 的准确率，参数量甚至比 12 层 5‑shift 版少 13%，训练时间相近。

**⚠️ 局限性**

局限性在于：仅在 CIFAR‑100 这一小规模数据集上验证；子空间尺寸固定且较小，可能限制更高维特征的表达；实现高度依赖 CUDA，跨平台可用性受限；未在更大规模视觉任务或其他领域进一步评估。

---

## 615. UCVE: Graph Structured Retrieval for Compliance Validation

**arXiv ID:** 2610.37323 | [PDF](https://arxiv.org/pdf/2610.37323v1)

**作者:** Emily Chen `[一作]` (Capital One, AI Foundations), Amirfarrokh Iranitalab `[通讯]` (Capital One, AI Foundations)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了结构化合规验证的检索问题，提出了UCVE框架，通过上下文需求图（CRG）实现对法律义务空间的检索与评估。

**💡 创新点**

将合规义务空间视为图结构，使用强化学习学习边权重并结合谱重排序，以低通滤波提升义务检索召回率。

**🔧 技术方法**

采用语义嵌入、基于RL的边权学习、k-NN稀疏化、图拉普拉斯谱分解及低通滤波重排序，配合LLM验证器。

**📊 数据集**

在三大领域分别使用OPP-115（隐私政策）、CLAUDETTE（消费者合同公平性）和ESRS（可持续披露）三个基准数据集。

**📈 对比分析**

与BM25、平面余弦、跨编码器、PPR、Leiden等六种基线在匹配K召回上比较，UCVE+Spectral在所有领域均超过基线3–5%的召回提升，且在ESRS领域表现最优。

**⚠️ 局限性**

对稠密义务图的路径效率低、对大规模义务空间的可扩展性有限、依赖领域专家的结构化元数据、以及RL奖励稀疏导致学习难度高。

---

## 616. Hierarchical Sparse Graph for Multi-dimensional Assignment in Event-based Tracking

**arXiv ID:** 2610.37321 | [PDF](https://arxiv.org/pdf/2610.37321v1)

**作者:** Muxi Zha `[一作]` (National University of Defense Technology), Banglei Guan `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无训练、事件驱动的多维关联方法，通过构造层次稀疏图（HSG）将事件多帧轨迹关联问题转化为图划分任务。

**💡 创新点**

创新点在于：①用模糊逻辑将贝塞尔曲线平滑误差和方向‑位移一致性两种动力学指标映射为统一的关联权重；②引入层次稀疏量化实现高效离散化；③设计基于连接相似度的自适应阈值图划分算法，实现自动化的多轨迹聚合。

**🔧 技术方法**

核心技术包括贝塞尔曲线拟合、方向一致性度量、模糊逻辑映射、层次稀疏量化、层次稀疏图构造、连接相似度与自适应阈值的图划分，并配合 K‑means++ 聚类与 null‑model 推导的阈值。

**📊 数据集**

在两个事件数据集上验证：eTram（交通感知）和 Ev‑UAV（微型目标），并对 Ev‑UAV 进行了标签修正以提升评测质量。

**📈 对比分析**

与事件跟踪基线（KF+HA、NS、IoU）以及图像跟踪的多维关联器（Permatrack、UAVMOT、OC‑SORT、LTTrack 等）进行对比；在 eTram 上 HSG 的 MOTA 与 IDF1 分别提升约 8% 与 50%，在 Ev‑UAV 上相较 OC‑SORT 的所有指标提升超过 2 倍，整体表现显著优于现有方法。

**⚠️ 局限性**

局限性在于：①基于投票的图划分缺乏全局连续性约束，可能在轨迹稀疏或拥挤场景下产生错误关联；②计算复杂度受 O(N²) 影响，实时性仍有提升空间。

---

## 617. Self-Referenced Diagnosis of SPAD Dark-Count-Rate Switching for Quantum Communication Receivers

**arXiv ID:** 2610.37312 | [PDF](https://arxiv.org/pdf/2610.37312v1)

**作者:** Yu Luo `[一作]` (Mississippi State University), Junming Diao `[通讯]` (Mississippi State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过自参照诊断框架，区分SPAD暗计数率的真实随机台阶噪声与温度阈值交叉、光泄漏、线缆扰动、电干扰等外部伪影；

**💡 创新点**

提出基于afterpulse-yield比R与归一化对偶生存统计ρ_s的自参照指标，能够在仅利用时间戳记录的前提下识别内在RTS与多种非内在机制；

**🔧 技术方法**

采用时间戳记录、afterpulse统计、DCR可重复性、跨通道共振、事件时序分析等技术，并通过受控实验复现温度阈值交叉、光泄漏、人工注入等伪影；

**📊 数据集**

四个PerkinElmer SPCM-AQR系列SPAD模块，在黑暗环境下连续监测超过920小时，收集时间戳、温度与光强等数据；

**📈 对比分析**

与受控实验产生的伪影结果对比，发现R≈1且ρ_s≈1仅对应内在RTS，其他机制表现出R≠1或ρ_s超出0–1区间，验证方法能准确区分不同机制；

**⚠️ 局限性**

当R接近1时难以区分高度相关的计数损失或精确仿真注入的情况，需要结合时序与跨通道信息；在高计数率下的时延分辨率有限，可能导致对某些微小效应的检测不充分。

---

## 618. MvT: Event-based Multi-view Projection for Multiple Object Tracking

**arXiv ID:** 2610.37309 | [PDF](https://arxiv.org/pdf/2610.37309v1)

**作者:** Muxi Zha `[一作]` (National University of Defense Technology), Laurent Kneip `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MvT框架，利用多视图投影将事件流编码为二维伪帧，以实现无监督的多目标跟踪。

**💡 创新点**

通过时间重对齐与多视图投影保留时空信息，并使用多视图融合和基于运动一致性的关联模型解决单视图模糊。

**🔧 技术方法**

多视图投影 (X‑OY, Y‑OT, TOX)、时间重对齐、双尺度高斯金字塔、图像基检测/分割方法、三项几何关联准则。

**📊 数据集**

在四个数据集上评估：公开的 ECD Shapes、BEES、Ev‑UAV，以及自制的 SOD（小目标）。

**📈 对比分析**

与伪帧和事件逐帧基线以及多种学习/非学习检测/分割算法比较，MvT 在定位精度、跟踪准确率、IDF1 上均位列前列，尤其在小目标与遮挡场景表现优异。

**⚠️ 局限性**

关联模型依赖线性运动假设，难以处理长时间遮挡或大角度转向；未利用事件极性与强度信息，且高斯核未针对不同视角自适应。

---

## 619. Understanding Human-Robot Collaboration through Joint Action

**arXiv ID:** 2610.37298 | [PDF](https://arxiv.org/pdf/2610.37298v1)

**作者:** Chengwen Zhang `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 2021–2026 年发表的 83 篇关于人机协作(HRC)的经验研究进行系统综述，运用 Joint Action 理论框架对其协作机制进行编码与统计，并通过共现与提升（lift）分析揭示不同构建的分布与相互依赖。

**💡 创新点**

创新点在于：①将 Joint Action 理论的九个构建与三大能力（Representation、Prediction、Integration）系统化地映射到 HRC 文献；②构建基于理论驱动的编码手册并对 83 篇论文进行细致的子编码；③利用共现与提升指标揭示构建之间的非对称依赖关系，进而指出研究热点与空白，提供基于证据的研究议程。

**🔧 技术方法**

技术手段：PRISMA 2020 指南的系统检索与筛选；理论驱动框架分析（directed 代码与 best‑fit 框架）；共现矩阵与提升（lift）分析；离散化子编码与统计检验；文本与证据级编码，二位者互信评估（Cohen’s κ）。

**📊 数据集**

数据集：收录自 ACM Digital Library、RO-MAN、Nature、Science Robotics、Computers in Human Behavior、International Journal of Social Robotics 等期刊与会议的 83 篇实证论文（2021–2026 年）。

**📈 对比分析**

方法比较：作者并未开展传统算法性能对比，而是通过频次、覆盖率与共现分析对各构建的研究关注度进行比较。结果显示，Prediction（特别是 Partner‑Action Processing、Task Sharing）与 Integration（Action Coordination）在 70% 以上论文中出现，Representation 的研究相对不足；共现图揭示 Division、Monitoring、Prediction 是多数子构建的必要前置，提示未来设计应优先关注这些核心机制。

**⚠️ 局限性**

局限性：①文献检索主要聚焦于工程与 HCI 领域，可能低估了社会科学、心理学等学科的相关研究；②部分构建（如 Shared Intention、Cooperative Commitment）需要从行为隐含推断，存在主观解释风险；③未对不同研究方法或实验设计进行细粒度评估，导致对技术有效性的直接比较有限；④共现分析依赖于论文数量，罕见组合的 Lift 可能因样本量小而不稳定。

---

## 620. TRUCO: A Scalable Lock-Free Algorithm for Parallel Contour Extraction

**arXiv ID:** 2610.37297 | [PDF](https://arxiv.org/pdf/2610.37297v1)

**作者:** Rafael Munoz-Salinas `[一作]`, Manuel J. Marin-Jimenez `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种多线程无锁的二值图像轮廓提取算法TRUCO，能够在多核CPU上高效提取几何边界。

**💡 创新点**

创新点在于引入起点所有权规则和向下无约束、向上终止的投机式跟踪，避免同步锁，减少内存带宽消耗并实现几乎线性加速。

**🔧 技术方法**

采用并行行扫描、8位状态空间、投机式追踪、起点所有权规则以及无锁写操作等技术。

**📊 数据集**

使用合成高分辨率圆形分布图像和真实世界的Liu4K 16MP数据集。

**📈 对比分析**

与OpenCV Suzuki‑Abe、RLE、Garcia等基线对比，单线程TRUCO约快2×，20线程可达≈20×（低分辨率）或≈14×（16MP），性能在不同硬件（i7、Xeon）保持一致。

**⚠️ 局限性**

在极端几何场景如全垂直条纹或单螺旋时，投机跟踪的中止开销会导致加速降低，最差仅略优于标准方法，且在高密度垂直线条时多线程收益有限。

---

## 621. A Linear Kernel for Planar Semitotal Domination

**arXiv ID:** 2610.37295 | [PDF](https://arxiv.org/pdf/2610.37295v1)

**作者:** Paloma T. de Lima `[一作]` (Norwegian School of Economics), Lukas Retschmeier `[通讯]` (University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对平面图中的半总支配问题，提出了一个线性核（大小为 358k）并证明了该问题在分割图、二分图、t‑爪无图上为 W[2]-难，在圆图上为 W[1]-难。

**💡 创新点**

创新点在于结合了 Alber‑Fellows‑Niedermeier 的区域分解框架并设计了新的三条约简规则，首次实现了平面图上半总支配的线性核；同时给出了多种图类的参数化复杂度边界。

**🔧 技术方法**

主要技术包括：平面图的最大区域分解、基于邻域类型（1、2、3）的三种约简规则、距离-2 见证半总支配的特性以及对约简后图的计数分析。

**📊 数据集**

论文完全基于理论分析，不使用实验数据集；所有结果均为多项式时间算法与复杂度证明。

**📈 对比分析**

与现有的总支配与独立支配等变体相比，核大小上实现了线性上界（358k），但常数系数较大；在实验验证方面尚无数据，理论上证明了算法的正确性和多项式时间可行性。

**⚠️ 局限性**

局限性包括：核大小常数较大，实际应用时可能不够紧凑；仅针对平面图给出了核，无法直接推广到一般图；对特殊图类（如三爪无图）仍无法得到多项式时间或核，仍存在开放问题。

---

## 622. LESSRec: Level-Specific History Modeling for Semantic ID-Based Recommendation

**arXiv ID:** 2610.37286 | [PDF](https://arxiv.org/pdf/2610.37286v1)

**作者:** Shangxin Yang `[一作]` (Chongqing University), Min Gao `[通讯]` (Chongqing University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了LESSRec和LESSRec+两种生成式推荐模型，通过在语义ID层级上单独编码历史序列并在解码时对齐同层级历史，实现更精准的下一个商品预测。

**💡 创新点**

创新点是级别特异的历史建模和双分支跨注意力融合，同时通过层级专用编码器提升效率。

**🔧 技术方法**

使用Transformer编码器-解码器架构、残差量化语义ID、级别隔离自注意力、双分支交叉注意力、位置偏置等技术。

**📊 数据集**

使用Amazon Games、Software和Beauty三大公开数据集，并采用RQ-VAE、RQ-Kmeans、LETTER三种tokenizer。

**📈 对比分析**

与传统顺序推荐器和其它生成式推荐器对比，LESSRec/LESSRec+在Recall@10/NDCG@10等指标上提升约15-50%，并在计算量和显存上显著下降。

**⚠️ 局限性**

局限在于需要预先生成并固定语义ID，且对长历史的处理仍受限于编码器层深度和token数。

---

## 623. Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa

**arXiv ID:** 2610.37245 | [PDF](https://arxiv.org/pdf/2610.37245v1)

**作者:** Valdini Douglace Lemofouet `[一作]` (African Institute for Mathematical Sciences), Shamsuddeen Hassan Muhammad `[通讯]` (Bayero University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个 Hausa 语言安全偏好数据集 HausaSafeLM，并在此数据集上评估了监督微调（SFT）与直接偏好优化（DPO）两种对齐策略，探究其在低资源、文化差异语言环境下的表现与权衡；

**💡 创新点**

创新点在于首次提供 Hausa 语言安全数据集、揭示拒绝率指标在低资源语言中失效、展示安全性、流利度与帮助性之间的三维权衡面以及数据质量与数量对 SFT 与 DPO 的不同影响；

**🔧 技术方法**

采用 Qwen2.5‑7B‑Instruct 作为基模型，使用 LoRA 进行参数高效微调，实施 SFT 与 DPO，利用 Gemini LLM 进行自动评判，构建 Refusal Rate、False Refusal Rate、Helpfulness Rate、SafeQ、AyaQ 等多维度指标；

**📊 数据集**

使用了从 PKU‑SafeRLHF 翻译并人工润色得到的 3,000 条 gold 示例与 7,000 条 silver 示例组成的 HausaSafeLM 数据集，以及 Aya 数据集中的 200 条 benign Hausa 提示；

**📈 对比分析**

通过对比 Base、SFT‑G、SFT‑GS、DPO‑G、DPO‑GS 五个配置，发现 Base 模型在 RR 上满分但质量极低；SFT 提升拒绝流利度但过度拒绝；DPO 提高拒绝精准度却大幅降低质量；SFT‑GS 在 RR、FRR、HR、SafeQ 与 AyaQ 上表现最平衡；

**⚠️ 局限性**

局限性包括仅测试单一 7B 模型、评判由单一 Gemini LLM 完成且缺少母语 Hausa 评审、数据集非原生 Hausa 编写且可能遗漏本土特定危害、评估样本规模有限、对齐目标仍以高资源语言方法为主

---

## 624. TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving

**arXiv ID:** 2610.37230 | [PDF](https://arxiv.org/pdf/2610.37230v1)

**作者:** Shuhuai Li `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向LLM代理服务的工具感知联合调度框架，在分布式CPU工具执行与聚类GPU推理之间高效管理会话。

**💡 创新点**

创新点包括将工具返回准备度、时间衰减的KV缓存信用与token占用作为联合价值指标，和基于动态规划的全局最优选择与放置策略，分离评估与调度实现可扩展性。

**🔧 技术方法**

使用动态规划求解多重背包/一般分配问题，时间衰减缓存信用模型，工具返回预测器以及异步重验证机制。

**📊 数据集**

在DeepSeek‑V4 Flash和Qwen3‑32B的真实工作负载/工具执行日志上进行实验，共12个模型-工作负载-并发设置。

**📈 对比分析**

与SGLang、ThunderAgent、llm‑d等现有基线对比，平均降低26.8%延迟、36.3%首Token时间、提升29.5%解码吞吐量。

**⚠️ 局限性**

受限于工具返回预测的准确性、对非代理场景适用性有限以及动态规划在大规模环境中的计算开销。

---

## 625. Decision Models as Judges for Hallucination Detection

**arXiv ID:** 2610.37229 | [PDF](https://arxiv.org/pdf/2610.37229v1)

**作者:** Haozheng Liu `[一作]` (New York University), Lingyao Li `[通讯]` (University of Arizona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了五种决策模型在检测五类幻觉（事实冲突、事实捏造、上下文不一致、逻辑不一致、指令不一致）上的性能，并与生成式LLM判别器gpt‑6‑sol做对比

**💡 创新点**

首次系统比较多种决策模型在不同幻觉类型上的表现，评估其置信度可靠性，并探索仅用决策模型构成的级联组合在低成本下提升检测精度的可行性

**🔧 技术方法**

使用决策模型（jev、d1、clef、pplx、gpt‑6‑luna）与生成式LLM（gpt‑6‑sol）进行判别，采用balanced accuracy、macro‑F1、AUROC、ECE等指标，设计多数投票、仲裁与置信级联等集成策略

**📊 数据集**

六个公开幻觉检测数据集（FAVABench、HADTest、RAGTruth、ContraDoc、ROSCOE、IF‑RewardBench），共计9852条样本

**📈 对比分析**

与gpt‑6‑sol比较发现决策模型在事实相关和上下文不一致类型可匹敌或超越生成式模型，且成本仅为其1–10%；在逻辑和指令不一致上仍落后；置信级联能在三个类型上略优于gpt‑6‑sol，同时成本仅占其1.2–3.3%

**⚠️ 局限性**

样本来源不均、类型与数据集混杂导致结果难以单独归因；置信度不完全可比；级联仅在离线评估下验证，未测试跨域或实时部署表现

---

## 626. Vision-Language-Action or World-Action? Just Action Models

**arXiv ID:** 2610.37219 | [PDF](https://arxiv.org/pdf/2610.37219v1)

**作者:** Yonghao Yu `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的训练框架——Just Action Models（JAM），通过在已有的视觉预训练模型基础上，结合动作专家，并使用动作损失与前瞻损失进行训练；前瞻损失仅利用冻结的未来表示作为目标，训练结束后丢弃前瞻分支，保持推理时仅使用动作预测；

**💡 创新点**

创新点在于将视觉-语言动作模型（VLA）与世界动作模型（WAM）之间的差异拆分为三部分：骨干初始化、对未来观测的监督目标以及推理时使用的模块；提出的框架可同时适用于生成式、自监督式与视觉‑语言预训练骨干，并且不需要在训练或推理阶段生成图像/视频，从而显著降低计算成本并提升成功率；

**🔧 技术方法**

技术实现主要包括：①预训练视觉骨干（如FastWAM的Wan、ImageWAM的Ovis-U1、DINOv2、Qwen3.5-2B）+动作专家；②动作损失（flow‑matching）与前瞻损失（将当前特征映射到冻结未来骨干的特征并用余弦距离监督）；③在训练期间仅计算当前帧特征，未来目标可预先缓存；④在推理时删除前瞻分支；

**📊 数据集**

使用了三大公开基准：LIBERO（含Spatial、Object、Goal、Long四套任务）、LIBERO‑Plus（更严格的随机化子集）以及RoboTwin 2.0（双臂操作任务）；

**📈 对比分析**

与FastWAM、ImageWAM以及其他现有基线在相同数据、架构、训练预算下进行比较；实验显示在LIBERO和LIBERO‑Plus上，JAM在不生成图像/视频的情况下取得与甚至优于原始WAM的成功率（例如LIBERO‑Plus上从55.3%提升至64.5%，ImageWAM上从65.3%提升至79.9%）；在RoboTwin 2.0上表现相当或略高；并且使用缓存目标后在线训练FLOPs下降73%（FastWAM）或37%（ImageWAM）。

**⚠️ 局限性**

局限性包括：仍依赖于视觉预训练模型的质量，无法完全独立于预训练；实验仅在仿真环境中进行，缺乏真实机器人部署验证；前瞻目标的选择与骨干架构相关，需手工调优；以及对极端分布移位的鲁棒性尚未充分评估。

---

## 627. Displacement Moments and Path-Terminal Separation in Brownian Interpolant Sampling

**arXiv ID:** 2610.37413 | [PDF](https://arxiv.org/pdf/2610.37413v1)

**作者:** Zhong Han `[一作]` `[通讯]` (National University of Singapore), Zhong Han (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了Brownian随机插值器的采样误差，并证明了基于位移矩的KL界限，提出了基于当前状态的学习器。

**💡 创新点**

创新点在于将位移矩与KL误差关联，区分路径与终端误差，首次给出Euler采样的O(1/N)界限，并展示Markov投影信息损失。

**🔧 技术方法**

采用随机微积分、KL散度分析、Euler离散化、位移矩分析、Schrödinger桥理论与条件扩散恒等式等技术。

**📊 数据集**

主要使用数值模拟生成的Brownian路径作为实验数据，没有使用公开数据集。

**📈 对比分析**

通过比较路径KL与终端KL误差，展示了O(1/N)及O(1/N²)的理论上界；在当前状态Euler采样中得到第二矩O((M₂+d)/N)的上界，并证明图像控制相较于概率预测器没有迭代优势。

**⚠️ 局限性**

局限性包括：对一般二阶矩耦合的统一终端速率仍未得到；需较高位移矩才能得到精确界限；Markov投影会丢失信息；实验仅基于合成模拟，缺乏真实数据验证。

---

## 628. Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation

**arXiv ID:** 2610.37409 | [PDF](https://arxiv.org/pdf/2610.37409v1)

**作者:** Zhenye Yang `[一作]` (Beijing University of Posts and Telecommunications), Senzhang Wang `[通讯]` (Central South University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MAGE 框架，通过离线流形增广和在线非对称图传播实现多模态推荐中的互补关系建模

**💡 创新点**

创新点在于将补充性信号从几何空间进行离线补偿，再与视觉稳定锚点进行异构双流融合，并通过自校准门控实现模态自适应融合

**🔧 技术方法**

采用局部 SVD+Soft Conceptor 进行补充性残差提取、局部 gap‑约束注入、双流 GNN（视觉同质化图 + 文本异质化图）、自校准门控及 BPR/对比式损失

**📊 数据集**

在 Amazon 四个公开数据集（Baby、Sports、Clothing、Electronics）上进行实验，使用提取的视觉 4096 维和 LLM‑预处理文本 512 维特征，构建共购图作为补充性边

**📈 对比分析**

与 MF‑BPR、LightGCN、VBPR、SLMRec、BM3、MGCN、LGMRec、MENTOR、GUME 等基线比较，MAGE 在 Recall@20/NDCG@20 上平均提升约 5–7%（硬补充子集提升近 10%）并在长尾与稀疏场景表现最优

**⚠️ 局限性**

主要限制包括共购边不完备、几何注入缺乏语义验证、对图噪声的鲁棒性仅在 30% 损坏下验证、对某些数据集的迁移性不佳以及模型复杂度相对较高

---

## 629. Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling

**arXiv ID:** 2610.37393 | [PDF](https://arxiv.org/pdf/2610.37393v1)

**作者:** Sinjae Kang `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种质量可控策略（QCP），能够在推理时根据用户指定的速度、平滑度等质量标签调节机器人执行行为。

**💡 创新点**

创新点在于将量化质量标签与迭代后见重新标记循环结合，既实现多维质量控制，又通过在线收集和重新标记扩展可控质量范围并提升任务成功率。

**🔧 技术方法**

采用基于流匹配的质量条件化行为克隆模型，并结合后见重新标记与在线滚动收集的迭代训练流程。

**📊 数据集**

使用了 Robomimic（Square、Transport）和 RoboEval（Pack Box、Pick Book）仿真数据集，以及真实 Franka 机器人在 conveyor 与 wine‑glass 传输任务中收集的演示数据。

**📈 对比分析**

与 BC、BC_filter、BC_acc、SORL、SCIQL、DSRL、EXPO 等基线比较，QCP 在质量与成功率双指标上均优于基线，并且迭代后可实现超越初始演示质量的表现。

**⚠️ 局限性**

局限性包括：需要手工定义规则性质量评估函数，质量标签维度升高会导致标签稀疏；后见重新标记循环需要多轮重新训练，计算开销较大。

---

## 630. Same Verdict, Different Readers: Personalizing Fact-Checking Explanations from Relative Preferences

**arXiv ID:** 2610.37380 | [PDF](https://arxiv.org/pdf/2610.37380v1)

**作者:** Haokun Wen `[一作]` (National University of Singapore), Wynne Hsu `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 PERPL 的框架，用于在保持事实准确性的前提下，根据不同读者的偏好重新撰写事实核查解释。

**💡 创新点**

创新点在于通过模拟读者偏好构建跨人格递归偏好对、使用 Borda 排序选择多重正例，并在两个阶段递进学习，既提升个性化，又突破预设风格限制。

**🔧 技术方法**

采用 LLM 生成候选解释、LLM 作为读者模拟、双向优先优化 (DPO) 与辅助似然损失、Borda 排序、对偶采样探索等技术。

**📊 数据集**

基于 MOCHEG、TRUE 与 VeriTaS 三个公开事实核查数据集构建 2080 条实例，并随机生成 10 种读者画像。

**📈 对比分析**

将 PERPL 与八个零样本 LLM（包括 Qwen3.5‑4B、Qwen3.8‑27B、MiMo‑V2.6‑Flash 等）在 2,710 组 claim‑persona 上进行事实准确性和偏好匹配评估，PERPL 在平均 FPW 约 93.3% 及 81.8% DWR 上显著优于所有基线。

**⚠️ 局限性**

主要局限在于依赖 LLM 生成的模拟偏好，真实读者偏好可能与模拟偏好分布不一致，且偏好多样性受预设风格限制。

---

## 631. Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories

**arXiv ID:** 2610.37369 | [PDF](https://arxiv.org/pdf/2610.37369v1)

**作者:** Zirui Ding `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yunfeng Lu `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 CauTrace 的白盒诊断框架，用来追踪大型语言模型（LLM）在整个推理过程中内部状态的演变，并揭示普通推理失败与对抗性诱导失败的共同机制。

**💡 创新点**

创新点在于：①将推理过程抽象为七个功能状态（支持/约束、答案承诺、验证等），②从三层级（表示传播、时间状态协调、轨迹拓扑）系统性分析状态演化，③证明同一终端标签可对应多条内部路径，④在对抗实验中将内部变化与失败结果关联。

**🔧 技术方法**

主要技术包括：使用自定义探针（probe）提取隐藏层中的七个状态得分；计算 Attention–MLP 同意度、值投影谱特征、状态活跃度、切换频率、中心化时序等指标；构造三层诊断分数（表示偏差、过程偏差、轨迹特征）并进行统计比较。

**📊 数据集**

使用的主要数据集有：Qwen2.5、LLaMA-3.2、LLaMA-3.1、Qwen1.5-MoE 等公开模型的推理轨迹；对照集为 Qwen 和 LLaMA 系列的“全错误”与“正确”推理样本；对抗性实验使用 AdvBench/HarmBench 生成的安全与非安全样本对。

**📈 对比分析**

方法比较：对每个模型分别计算成功与失败轨迹在表示传播一致性、验证与承诺的时间差、状态切换次数等指标的差异；结果显示失败轨迹在表示传播一致性下降、验证活跃度相对承诺降低、状态切换增多；对抗样本的内部变化幅度显著大于非翻转样本，且与失败的功能状态匹配，验证了诊断框架的有效性。

**⚠️ 局限性**

局限性包括：①表示层级指标对不同模型/规模表现不一致，缺乏统一的“通用电路”；②诊断依赖大量探针和白盒访问，实用性受限；③需要手动校准阈值和权重，模型间可迁移性仍有限；④对极端长文本或多任务场景的适应性尚未验证。

---

## 632. EmbeddingGemma 2: Unified and Efficient Multimodal Representations

**arXiv ID:** 2610.37364 | [PDF](https://arxiv.org/pdf/2610.37364v1)

**作者:** Henrique Schechter Vera `[一作]` (Google), Mojtaba Seyedhosseini `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 EmbeddingGemma 2，一款 744 M 参数的通用多模态嵌入模型，可将文本、图像、视频、音频及其混合输入映射到同一向量空间；

**💡 创新点**

创新点包括：1) 使用 271 M 双向 Transformer backbone 并集成可选视觉和音频编码器；2) 简化层级嵌入（Per‑Layer Embeddings）去掉 lookup 表；3) 采用三阶段训练（文本预微调→多模态预微调→对比+正则+蒸馏+多重维度 MRL）实现更优的嵌入空间；4) 支持嵌入截断、量化、模块化加载，满足资源受限环境；5) 在 7 个多模态基准上刷新 Pareto 前沿；

**🔧 技术方法**

技术手段包括：双向 Transformer、RMSNorm、GELU、Per‑Layer Embeddings、Vision Transformer、Conformer audio encoder、对比损失、Spread‑Out 正则、蒸馏匹配、Matryoshka Representation Learning、量化感知训练、梯度累积与学习率调度等；

**📊 数据集**

使用的大规模多模态数据集涵盖 MTEB、MMEB‑v2、MIEB‑lite、MAEB、MSEB(Retrieval)、代码检索、视觉问答、视频检索、音频分类/检索等多任务混合数据；

**📈 对比分析**

通过与同参数或更大参数的模态专家和通用嵌入模型对比，EmbeddingGemma 2 在 MTEB、MMEB‑v2、MIEB‑lite、MAEB、MSEB 等基准上均刷新 Pareto 前沿；文本、代码、视觉、音频任务均超过同类模型，量化版仅略低于浮点版；

**⚠️ 局限性**

局限性：对长序列（超大上下文）支持有限；在零样本情感聚类等细粒度音频任务上仍略逊于专用模型；视频细粒度依赖 token 数量，需进一步优化；模型规模相对较大，推理延迟与显存占用仍是挑战。

---

## 633. Online Interval Selection with Binary Predictions: Optimal Consistency and a Continuous Robustness Tradeoff

**arXiv ID:** 2610.37361 | [PDF](https://arxiv.org/pdf/2610.37361v1)

**作者:** Zhong Han `[一作]` `[通讯]` (National University of Singapore), Zhong Han (National University of Singapore)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在有二元预测的情况下的在线区间选择问题，提出了 Shaped-BANK 算法并分析其一致性与鲁棒性平衡。

**💡 创新点**

创新点在于证明了一致性下界 d=φ² 的最优性，给出了完全满足一致性和鲁棒性约束的算法，并提供了连续鲁棒性‑一致性权衡的完整描述。

**🔧 技术方法**

使用了基于账本的记账机制（Bank 机制）、特殊替换阈值、线性成本分析以及严格的几何与组合论证明。

**📊 数据集**

论文主要以理论证明为主，使用公开作业轨迹构造实验实例，但并未给出实际数据集上的实验结果。

**📈 对比分析**

与现有的 1‑增量策略、LR 规则等进行了比较，Shaped-BANK 在一致性为 d 时实现了最小鲁棒性 φ⁵，且在连续权衡中能取到最优性能；相较于其他算法在已知预测情况下的表现更优。

**⚠️ 局限性**

局限性包括：尚未解决最优通用预算上界；对预测错误的鲁棒性分析不足；未考虑公平性和不等成本的实际情况；算法在实际部署中的复杂度未作评估。

---

## 634. Generalizable Target-Less Joint Intrinsic-Extrinsic Camera-LiDAR Calibration using Deep Pixel-Point Correspondences

**arXiv ID:** 2610.37350 | [PDF](https://arxiv.org/pdf/2610.37350v1)

**作者:** Simon Bultmann `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种完全自动、无标定目标的相机–激光雷达联合标定框架，能够同时估计相机内参、径向切向畸变以及相机与雷达之间的六自由度外参。

**💡 创新点**

创新点包括：① 在深度像素–点对应学习中加入相机内参和畸变的随机扰动与合成图像增强，使网络能够在未校正的原始图像上直接学习对应关系；② 将对应预测与非线性优化循环耦合，通过迭代先后更新内参和外参，从而实现内外参的联合优化；③ 采用基于结构光与视觉运动的自动初始化（COLMAP + ORB‑SLAM3 + FAST‑LIO2），无需人工标定目标。

**🔧 技术方法**

技术手段包括：相机针孔+径向切向畸变模型、合成相机参数扰动与图像重投影、CMRNext架构的像素‑点对应网络、RANSAC内点筛选、Ceres非线性最小二乘优化、首尾迭代的内参冻结策略。

**📊 数据集**

训练集使用KITTI（左相机序列01–21）、Argoverse、PandaSet的校正与原始图像；评估集为KITTI右相机（未见配置）以及三台真实机器人（Car、Spot、UAV）不同相机/雷达组合。

**📈 对比分析**

与基线方法MDPCalib、Koide、Shen以及CMRNext对比。结果显示：在KITTI右相机上，联合内外参优化后，平移误差约为2.25 cm、旋转误差0.05°，明显优于仅校正外参的MDPCalib（2.86 cm、0.13°）和基准方法；在三台机器人上，平均平移误差降至3–4 cm，旋转误差≤0.1°，重投影误差≤4 px，表现为当前最优的无目标联合标定方案。

**⚠️ 局限性**

局限性：总体计算时长约300 秒（含初始化、对应推理与优化），对实时性要求高的应用仍有挑战；方法仍依赖较好的运动轨迹和足够多的图像‑雷达配对，极端运动或极低光照下的鲁棒性尚待进一步验证。

---

## 635. Normative Deviation Modeling for EEG-based Depression Detection

**arXiv ID:** 2610.37338 | [PDF](https://arxiv.org/pdf/2610.37338v1)

**作者:** Wan Jiang `[一作]` (Hefei University of Technology), Richang Hong `[通讯]` (Hefei University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发基于健康对照的规范偏差指导诊断方法S‑NDD和F‑NDD，用以提升EEG在重度抑郁症（MDD）检测中的鲁棒性。

**💡 创新点**

创新在于同时构建结构和分布健康参考模型，并将标量级与特征级规范偏差分别融合进深度EEG特征，此外提供目标HC辅助校准以应对跨源分布漂移。

**🔧 技术方法**

技术包括VQ‑VAE与扩散模型生成健康参考、深度时空EEG编码器、注意力聚合、辅助分类以及统计校准。

**📊 数据集**

使用公开的Mumtaz2016与MODMA两大EEG数据集进行实验。

**📈 对比分析**

与传统监督学习、生成式增广及域适应方法对比，S‑NDD与F‑NDD在窗口级和受试者级均取得最高准确率、F1和AUC，并在跨源评估中显著优于基线。

**⚠️ 局限性**

局限在于需通过空间插值对不同电极配置进行对齐，且对采集差异的适应仍依赖手动校准，缺乏完整的无监督域适配方案。

---

## 636. WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization

**arXiv ID:** 2610.37329 | [PDF](https://arxiv.org/pdf/2610.37329v1)

**作者:** Wenjie Qiu `[一作]`, Yue-Jiao Gong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于强化学习的自适应框架WCC，用以在大规模全局优化中联合学习模式切换（全空间NDA或子空间CC）和资源分配。

**💡 创新点**

创新点在于将优化模式与资源预算视为分层决策，并通过16维自适应状态（包含问题结构、收敛趋势、低频能量等）和FeatureSE机制实现跨问题的通用决策。

**🔧 技术方法**

核心技术包括PPO强化学习、自动回归actor结构、FeatureSE重加权、低频能量比例、梯度趋势、变分估计及分解算法DG2/RDSM。

**📊 数据集**

实验数据集涵盖CEC2013LSGO15个大规模分布式全局优化问题以及AOB（1000维Ackley、Rastrigin、Schwefel、Elliptic等多重重叠变量变体）。

**📈 对比分析**

通过与多种CC/NDA/混合基线方法（如CMA-ES、MMES、DG2-CMAES、CC-SHADE-SL等）在3×10⁶函数评估下对比，WCC在绝大多数问题上实现了显著性能提升，平均赢得更多测试。

**⚠️ 局限性**

局限性包括对分解质量高度依赖、训练成本高、在极端维度或高度非线性问题上性能仍受限，以及在资源预算极小或子空间数极大时易出现决策失效。

---

## 637. Decomposing Directed Graphs into Partial Orders is Hard

**arXiv ID:** 2610.37379 | [PDF](https://arxiv.org/pdf/2610.37379v1)

**作者:** Grzegorz Gutowski `[一作]` (Jagiellonian University), Mikołaj Rams `[通讯]` (Jagiellonian University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文研究了有向图（尤其是有向无环图 DAG）被分解为若干个传递子图（对应严格偏序）的问题，提出并分析了“偏序分解数”和“偏序划分数”两种参数，并给出了它们的复杂度结果。

**💡 创新点**

创新点在于：①证明对任意固定 k≥2，判断一个 DAG 能否用不超过 k 个偏序覆盖或划分是 NP‑完备的；②提供了两种关键化简：k=2 时从 NAE‑3‑Sat 归约，k≥3 时从 k‑Coloring 归约；③揭示了在三角形无关 DAG 上两种参数相等；④提出了“pop‑critical DAG”的概念并用于构造关键性子图。

**🔧 技术方法**

技术主要包括图论的结构分析、线图与基础图的色数关系、偏序的传递闭包特性、以及多层次的构造化简（变量/子句 gadget、保护弧、颜色传递）。

**📊 数据集**

本文纯粹理论，没有使用实验数据集；归约基于经典 NP‑难问题（NAE‑3‑SAT、k‑Coloring）以及已知存在的三角形无关 DAG 与高色数图的构造。

**📈 对比分析**

与之前的工作相比，本文给出了完整的 NP‑完备性证明；虽然没有实验评估，但理论上证明了对任意固定 k 的硬度，且在三角形无关 DAG 上两参数相等的性质为进一步研究提供了基线。

**⚠️ 局限性**

局限性包括：①对 tournament（强连通有向图）中 k≥3 的复杂度仍未解决；②构造的 “pop‑critical DAG” 并未给出多项式大小的显式构造，导致构造过程的规模随 k 指数增长；③未给出参数化或逼近算法，实际应用层面缺乏实验验证。

---

## 638. Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution

**arXiv ID:** 2610.37407 | [PDF](https://arxiv.org/pdf/2610.37407v1)

**作者:** Daniel Beránek `[一作]` `[通讯]`, Daniel Beránek

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于DIO/ODI诊断框架的“提交边界授权”机制，分离了Agentic ER中用于推理的证据与授权持续身份变更的证据；

**💡 创新点**

创新点在于将数字身份优化与本体诊断映射到专门的身份保持约束，并通过声明式的授权记录与版本化验证实现安全的持久化；

**🔧 技术方法**

采用了抽象的两阶段MVDIO监督层、域配置的约束绑定函数、声明式的证据提升契约，以及基于SHACL/PROV-O等语义技术的验证逻辑；

**📊 数据集**

该工作为概念性架构，未使用具体数据集，作者仅建议未来在实体解析基准（如TempEL、实体覆盖集）上进行实验；

**📈 对比分析**

对比方法包括无约束Agentic ER、传统标识规则、通用授权门控以及DIO/ODI驱动门控，性能指标尚未评估，本文仅提供实验设计与度量框架；

**⚠️ 局限性**

局限性在于无法消除假设驱动检索的“自我确认”风险、依赖正确的推广过程与域政策、缺乏实证验证，以及对版本冲突与依赖追踪的有限处理。

---

## 639. Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers

**arXiv ID:** 2610.37405 | [PDF](https://arxiv.org/pdf/2610.37405v1)

**作者:** Zongyuan Sun `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为FoG的框架，训练知识图谱(KG)代理通过多轮交互检索并验证答案背后的支持事实，从而让答案可追溯并可审核。

**💡 创新点**

创新点主要包括：① evidence‑grounded synthesis（基于证据的合成）将支持事实嵌入生成的问题中；② Evidence‑Coverage Process Reward（ECPR）在强化学习阶段为检索到的每条支持三元组提供密集奖励；③ 通过设计的 KG 交互代理，记录检索过程中的所有三元组并在训练中加以利用。

**🔧 技术方法**

技术方法涵盖：大型语言模型（Qwen2.5‑3B‑Instruct、Qwen3‑4B‑Instruct）进行监督微调和GRPO强化学习；ReAct式的 KG 交互代理（7种原子操作）；基于反向扩展的问句生成与符号化质量控制；以及多项式奖励设计（答案准确度、步骤有效性、证据覆盖率）。

**📊 数据集**

实验数据集包括 Wikidata‑based KGQA 3 个基准（WikiWebQuestions、QALD10‑en、GTSQA）以及用于跨 KG 转移的 Freebase‑based WebQSP。

**📈 对比分析**

与多种基线（LLM 直接推理、检索‑基于方法、Agent‑based 方法）对比，FoG 在 Hit@1、F1、证据覆盖率（Cov）上均有显著提升，甚至在 3B–4B 小模型上超过更大模型和专门优化的 KG‑R1；在未见 KG 上的零样本转移也表现出良好效果。

**⚠️ 局限性**

局限性包括：① 依赖于 Wikidata 的完整性和实时性，图谱不完整或过时时会影响检索；② 仍需外部教师 LLM 生成合成问题和轨迹，制约自我学习能力；③ 交互步数上限为 10，难以处理更长链或更开放的推理任务。

---

## 640. Para-Gaussian Approximations of Random Deep Neural Networks

**arXiv ID:** 2610.37388 | [PDF](https://arxiv.org/pdf/2610.37388v1)

**作者:** Mihai Nica `[一作]`, Janosch Ortmann `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文提出了一种更准确的神经网络分布近似，称为para-Gaussian分布，旨在改善随机初始化神经网络在有限层数下的高斯行为近似。

**💡 创新点**

创新点在于引入了para-Gaussian近似，通过Hermite多项式构建，提供了一种递归方法来近似神经网络的深层结构，并展示了其在ReLU神经网络中的应用。

**🔧 技术方法**

使用了Hermite多项式和递归方法来构建para-Gaussian近似，并通过总变差距离来评估其与传统高斯近似的性能。

**📊 数据集**

使用了Monte Carlo模拟数据集来验证所提出的para-Gaussian近似与高斯近似的比较，特别是在不同层数和输入输出情况下的表现。

**📈 对比分析**

与传统高斯近似相比，para-Gaussian近似在总变差距离上表现出更好的收敛性，具体表现为O(n^-2)的收敛速度，相较于高斯近似的O(n^-1)。

**⚠️ 局限性**

限制在于该方法主要针对随机初始化的神经网络，且在网络深度较大时，仍需进一步研究其在其他类型网络中的适用性和表现。

---

## 641. Agent-Conditioned Routing: Diverse Reasoning Paths Within One Frozen Model

**arXiv ID:** 2610.37375 | [PDF](https://arxiv.org/pdf/2610.37375v1)

**作者:** Ananth Kalyanasundaram `[一作]` (QpiAI), Aswanth Krishnan `[通讯]` (QpiAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的推理时控制方法 Agent-Conditioned Routing（ACR），通过在冻结的稀疏 MoE 中使用专家可用性掩码实现单次提示下的多方法推理，并在数学与物理的 126 问题基准上进行评估。

**💡 创新点**

创新点在于利用专家可用性掩码（shared/private 结构）在保持模型权重与路由器不变的前提下，实现轨迹间的推理多样性；相较于传统温度采样、Expert‑Sample、RoE 以及信息化顺序提示，ACR 在相同生成预算下显著提升了产生多方法解的概率。

**🔧 技术方法**

主要技术包括：冻结稀疏 MoE（35B 参数）、基于种子化的共享/私有专家可用性掩码、在前 2000 令牌内限制路由、持久化掩码与全模型 KV 缓存共享、对比实验中的温度采样、Expert‑Sample 与 RoE 以及信息化顺序提示。

**📊 数据集**

使用的数据集为 126 道手工筛选的数学与物理多模式推理题（涵盖积分、级数、概率、微分方程、量子等 12 类），并配备 11 道合成开发集以及 30 道被人类评审的子集。

**📈 对比分析**

比较方法：温度采样（T=0.5,0.9,1.1）、Expert‑Sample、RoE、信息化顺序提示；性能方面，GLM‑5.3 版 ACR 在 P=160 时，至少两种方法的题目数为 37.3（≈两倍于温度采样 18.7），准确率约 95.8%；GPT‑5.6‑Sol 版 ACR 则为 45（≈1.67倍于温度采样 27），准确率 97.7%。人类盲评显示 ACR 的多方法数上升约 1.3–2 倍，但准确率略低。

**⚠️ 局限性**

局限性包括：掩码仅限制计算路径，未保证专家具备语义专门化；同一方法可能被多条轨迹复现；基准样本有限，未覆盖更广泛任务；人类评审的同意度未量化；对资源伸缩性与下游决策（如投票或合成）仍需研究。

---

## 642. Comparing Temporal Deep Learning Architectures for Discrete Cross-User Myoelectric Control

**arXiv ID:** 2610.37374 | [PDF](https://arxiv.org/pdf/2610.37374v1)

**作者:** Ethan Eddy `[一作]` (University of New Brunswick), Erik Scheme `[通讯]` (University of New Brunswick)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对五种时序深度学习架构（LSTM、BILSTM、GRU、Transformer、RNN）在离散跨用户肌电控制中的表现进行了系统评估，使用了EMG-EPN‑612和MyoDisCo两大数据集，验证了模型在未见用户及实时连续采集环境中的泛化能力。

**💡 创新点**

创新点在于：①首次在大规模跨用户肌电数据上对多种时序模型进行基准比较；②引入无用户校准的零样本（zero‑shot）跨用户框架；③结合能量阈值分割方法，将模型迁移到真实连续采集场景；④系统分析模型复杂度与性能的权衡。

**🔧 技术方法**

技术手段包括：1‑D卷积特征提取（CNN）；时序网络（GRU、LSTM、BILSTM、Transformer、RNN）；能量阈值的Teager‑Kaiser基分割；Adam优化、交叉熵损失、梯度裁剪；以及统计检验（Friedman、Nemenyi、Wilcoxon）。

**📊 数据集**

使用的数据集为：EMG‑EPN‑612（612位用户、6种手势、5秒窗口标注）和MyoDisCo（14位用户、连续随机提示、无反馈、收集时间跨度5年、不同国家）。

**📈 对比分析**

方法比较：对不同模型、层数、特征提取方式以及参数规模进行离线准确率、实时误检率、延迟等指标评估；结果显示GRU在离线模式下取得93%准确率，MyoDisCo连续数据上最佳GRU模型达96%准确率，参数量最少的RNN也能达到约91%；BILSTM/Transformer参数量显著高但性能提升有限。

**⚠️ 局限性**

局限性包括：仅评估了六种预定义手势，未覆盖更复杂多样化动作；模型在嵌入式部署时对参数量和延迟的需求仍需进一步优化；分割策略依赖能量阈值，可能对不同设备或运动环境泛化受限；并且未对实时校准或用户反馈机制进行研究。

---

## 643. TimeLineXR: An Interactive Exploration of Time in Vision-Language Models

**arXiv ID:** 2610.37349 | [PDF](https://arxiv.org/pdf/2610.37349v1)

**作者:** Nidham Tekaya `[一作]` (Univ. of Applied Sciences St. Pölten), Matthias Zeppelzauer `[通讯]` (Univ. of Applied Sciences St. Pölten)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可交互的时间轴可视化工具TimeLineXR，允许用户在浏览器或WebXR环境中上传图片并在三维时间轴上定位其首次出现年份。

**💡 创新点**

创新点在于将VLM隐含的时间轴转化为可视、可操作的三维场景，并提供无安装的WebXR交互体验。

**🔧 技术方法**

使用CLIP或EVA‑CLIP编码、余弦核KPCA降维、Bézier曲线拟合以及FastAPI后端服务。

**📊 数据集**

基于每年（1700‑2024）的文本提示以及TIME10k基准数据集进行时间轴构建。

**📈 对比分析**

与参考方法相比，在3维投影下MAE为15.3/20.8年（CLIP/EVA‑CLIP），相较原13维实现略高，但实现了可视化交互。

**⚠️ 局限性**

限制包括降维导致精度下降、缺乏实时摄像头输入以及对XR设备API访问的局限性。

---

## 644. Aligning Institutional Power and Operational Ability

**arXiv ID:** 2610.37342 | [PDF](https://arxiv.org/pdf/2610.37342v1)

**作者:** Andrea Marino `[一作]` (University of Amsterdam), Tom van Engers `[通讯]` (University of Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种将机构层的权力与义务与运作层的行动能力对齐的正式框架，并通过过程模型（Petri网）来描述和分析多代理系统中的可执行性与机构状态的可追踪性。

**💡 创新点**

创新点在于：①将机构权力与义务独立建模为状态性位置，并通过反应规则与运作层同步；②定义了“操作可落地性”和“正式机构合法性”两项对齐属性；③提出了可修复的机制（操作层和机构层的约束修正），从而在设计阶段即可发现并纠正不一致的规范配置。

**🔧 技术方法**

使用的技术包括：Petri网（描述运作层流程与可执行性），反应规则与状态机（描述机构层生命周期），治理网（将两层结合的有向图），以及基于这些模型的可达性与属性判定。

**📊 数据集**

没有使用公开数据集，所有实验均基于作者构造的示例（委托销售争议情景）来验证框架的可行性。

**📈 对比分析**

本文没有进行实验性比较，只在理论层面给出了可达性分析与属性验证的算法描述；若采用自动化工具，性能应随Petri网规模呈指数级增长，但对小规模实例（如示例）可在毫秒级完成。

**⚠️ 局限性**

局限性包括：①仅适用于可归约为Petri网的有限流程；②缺乏对不确定性与概率行为的处理；③未提供自动化合成修复的实现与复杂度分析；④对实际大型分布式系统的可扩展性与实现细节尚未验证。

---

## 645. diff-soup: GPU-native Differentiable Simulation and Inference for Large-scale Stiff Dynamical Systems

**arXiv ID:** 2610.37333 | [PDF](https://arxiv.org/pdf/2610.37333v1)

**作者:** Samuel Degnan-Morgenstern `[一作]` (Massachusetts Institute of Technology), Martin Z. Bazant `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 diff‑soup，一个基于 JAX 的 GPU 原生可微分 PDE 求解与优化框架，支持大型稀疏刚性系统的自适应变量步长、变量阶数（VSVO）积分，内置 cuDSS GPU 稀疏直接求解器，并实现了从前向/后向梯度、贝叶斯逆问题到嵌入式深度学习的端到端 PDE‑约束优化工作流。

**💡 创新点**

创新点包括：
① 针对稀疏结构的完全可微分设计，利用图着色实现稀疏自动微分；
② 与 NVIDIA cuDSS 对接的 GPU 稀疏直接求解器，显著突破传统 CPU 线性代数瓶颈；
③ 采用多 GPU、混合精度与批处理技术，支持大规模 2D/3D PDE 的求解；
④ 完整的逆向、前向、二阶敏感性实现，实现一次反向传递即可获得梯度与 Hessian‑vector 乘积；
⑤ 通过 Recipe API 让用户以数学式写 PDE，自动生成稀疏 Jacobian、局部/全局变量分离与 Schur 补全；
⑥ 开源生态，兼容 JAX、PyTorch 等深度学习框架，支持 Bayesian、MAP 与深度网络嵌入。

**🔧 技术方法**

核心技术：
- JAX 自动微分与 JIT；
- GPU 原生稀疏直接求解器 cuDSS；
- 变量步长、变量阶数 BDF/NDF/IDA（JIDA）求解器；
- 稀疏自动微分 + 图着色（CPR）与 Schur 补全；
- 多 GPU 共享/分布式稀疏因子化、混合精度与分块求解；
- 逆向/前向敏感性、检查点重放；
- Bayesian 推断（MAP、NUTS）、优化器（Newton TR、LM、L-BFGS）与深度网络（神经场、W‑IRE）嵌入。

**📊 数据集**

实验数据：
- 1D 热方程（N = 2⁶〜2¹⁴）；
- 2D Brusselator（128×128 网格）；
- 2D/Cahn‑Hilliard 联合化学-力学耦合（256×256 网格，50×50 nm 物理域）；
- 相分离材料的旋光分解图像（用于贝叶斯逆问题）；
- 3D 填充床反应器（96×24×24 网格）用于深度网络嵌入。

**📈 对比分析**

性能对比：
- 与 SUNDIALS、PETSc、Julia 和 MATLAB 在热方程、Brusselator、Cahn‑Hilliard 上做工作-精度与规模图比较；diff‑soup 在 GPU 上在 10⁴-10⁶ 自由度范围内实现 1–2 个数量级速度提升；
- 前向梯度在 CPU 上大约 10 倍慢，反向梯度在 GPU 上 10–100 倍快；
- 通过混合精度与多 GPU，三维 Brusselator 的 10⁶ DOF 计算可在 1/10 的时间完成；
- 在贝叶斯逆问题中，MAP 与 NUTS 采样收敛良好；
- 嵌入式神经网络训练在 7.5×10⁴ 参数规模下完成，无需自定义链路，梯度准确性与数值差值对比。

**⚠️ 局限性**

局限性：
- 依赖 NVIDIA CUDA GPU，CPU 版本仅支持 KLU 或稠密求解；
- 目前仅支持指数 1 的 DAE，较高阶 DAE 需进一步开发；
- 逆向求解需冻结步长/阶数控制，无法完全捕捉自适应动态；
- 稀疏直接求解在 3D 大规模时仍受填充导致的 O(n³/2) 成本与显存限制；
- 对极端非线性或需要高精度的求解场景，混合精度可能导致误差累积；
- 事件处理和多目标优化尚未完全实现高级功能；
- 目前缺乏针对问题特定预条件的 Newton‑Krylov 支持，可能在某些强耦合或非线性强的系统中表现欠佳。

---

## 646. Hidden in Plain Sight: an Empirical Study about The Instruction Stacks of Coding Agents

**arXiv ID:** 2610.37332 | [PDF](https://arxiv.org/pdf/2610.37332v1)

**作者:** Felipe de Sant'Anna Paixão `[一作]` (Federal University of Bahia), Iftekhar Ahmed `[通讯]` (University of California at Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对五款主流命令行编码代理（Claude Code、Codex CLI、Gemini CLI、Qwen Code、OpenCode）的首次模型请求中的“指令堆栈”进行了纵向跨版本的系统采集与分析，揭示了指令堆栈的组成、演化、共性与差异，并追踪了开源代理的指令变更动机。

**💡 创新点**

创新点在于：①首次将完整指令堆栈（除用户提示外的系统指令、工具定义、环境事实）视为可量化研究对象；②通过捕获多达1,886个版本的首次请求，构建了规模最大的跨代理指令堆栈时间序列数据；③结合自动化 LLM 辅助标注与变更追踪，系统揭示了变更背后的开发者动机与技术债务问题。

**🔧 技术方法**

主要技术手段包括：①基于四种 LLM API 协议的统一映射与规范化；②句子级指令切分与自定义十二类功能编码的 LLM 辅助标注；③Levenshtein 编辑距离与差异匹配的变更量化；④LLM（Gemini 4 Argon）驱动的代码变更追踪与动机归因。

**📊 数据集**

数据集：1,886个捕获的首次请求，覆盖1,911个可用发布，涉及约350,000句子级段落，含工具定义、系统指令、消息角色等；通过自动化脚本在沙盒环境下捕获，确保版本与运行环境一致。

**📈 对比分析**

比较方法：对比四个开源代理的指令占比、工具占比、编辑距离以及变更动机比例；结果显示指令堆栈中指令文本占比高达90%，工具定义占比在Claude Code、Qwen Code等三款中超过70%；编辑距离平均约1–2%，仅少数大幅度变更；动机分析表明只有48%变更单元直接说明指令变更原因，主要聚焦模型行为优化。

**⚠️ 局限性**

局限性：①只研究命令行代理，未覆盖图形化或交互式代理；②仅捕获首次请求，未考虑后续交互与动态工具加载；③动机归因依赖开发者提交信息，可能漏报或误报真实原因；④数据截至2026年8月，后续版本及其他代理可能表现不同。

---

## 647. CTBakeGS: training-free baking of photorealistic CT volume renderings into 3D Gaussian splats

**arXiv ID:** 2610.37406 | [PDF](https://arxiv.org/pdf/2610.37406v1)

**作者:** Haruki Fukuda `[一作]` (Gunma University), Lars Ebert `[通讯]` (Zurich Forensic Science Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

将CT体积渲染直接转换为无需训练的三维高斯散点模型（3DGS），实现实时可视化

**💡 创新点**

创新点在于：①直接从体素数据映射生成高斯散点，避免昂贵的训练/优化；②使用透明度映射使重叠散点产生真实的光学厚度；③预烘培光照并随机保留微透明散点，克服8‑bit透明度阈值导致的层失真

**🔧 技术方法**

采用自适应八叉树划分单元、GPU并行采样、光照体积预计算、透明度映射、抗锯齿补偿及WebGL 2渲染器；输出标准 .ply/.spz 文件

**📊 数据集**

在 TotalSegmentator v2 数据集中选取 6 例 1.5 mm 分辨率 CT 扫描（头、心脏、胸腹、骨盆等），并对 5 mm 厚片数据做测试

**📈 对比分析**

使用 PSNR、SSIM、LPIPS 与收敛的光照渲染结果比较；两百万散点模型在宽视图下平均 PSNR ≈32 dB、SSIM≈0.97、LPIPS≈0.06；转换时间在笔记本 4–30 s，桌面 9–60 s；渲染帧率笔记本 68 fps，桌面 28 fps

**⚠️ 局限性**

局限性包括：仅评估 6 例 1.5 mm 数据，未覆盖更薄切片或大体积；对光照、转换器与阈值选择的敏感性；缺乏解剖准确性和临床可用性验证；散点数量受限，细薄层和表面高光仍欠佳

---

## 648. LinMix: Data Mixing as Linear Domain Attribution

**arXiv ID:** 2610.37390 | [PDF](https://arxiv.org/pdf/2610.37390v1)

**作者:** Hubert Baniecki `[一作]` (University of Warsaw), Martin Pawelczyk `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于顶点采样的域混合方法，并利用可解释的线性域归因模型实现语言模型预训练数据混合的优化。

**💡 创新点**

创新点在于将数据混合视为域归因问题，使用顶点采样而非传统Dirichlet采样来设计实验，从而保证线性模型可解释并可扩展至数百域。

**🔧 技术方法**

采用线性回归、实验设计理论、顶点采样以及对数/平方根变换等技术进行域权重优化。

**📊 数据集**

实验基于200B-token的WebOrganizer语料库（24粗粒度/288细粒度域），并扩展至416B-token、40域的混合语料。

**📈 对比分析**

与Dirichlet采样、指数法、LightGBM等基线相比，在1B模型上MBPP提升46%、GSM8K提升14%，在4B模型上平均提升约11%。

**⚠️ 局限性**

主要限制是需训练大量代理模型，且仅对单一目标模型进行一次完整预训练，顶点采样在存在上限时仍可能降低信息量。

---

## 649. Uneven Use, Similar Trust Generative AI in Denmark and Romania

**arXiv ID:** 2610.37348 | [PDF](https://arxiv.org/pdf/2610.37348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 650. How Many and How Much? Contrastive Learning with Negative Samples for Knowledge Graph Embedding

**arXiv ID:** 2610.37389 | [PDF](https://arxiv.org/pdf/2610.37389v1)

**作者:** Sooho Moon `[一作]` (Chung-Ang University), Yunyong Ko `[通讯]` (Chung-Ang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于对比学习的知识图谱嵌入框架，解决负样本数量和权重分配问题；通过风险感知的负样本预算分配和分布校准的负样本权重分配，集成到四种主流KGE模型的训练中。

**💡 创新点**

创新点包括：①将开放世界KG视为闭世界KG的随机偏置，利用观察度量负样本的假负风险并动态分配负样本预算；②在负样本权重上同时考虑分布的离散度和集中度，使用熵度量高分负样本集中度，进而校准权重，从而更有效地利用负样本。

**🔧 技术方法**

使用的技术：对比学习框架、负样本采样、风险感知负样本预算分配、分布校准负样本权重（标准化分数、softmax、熵度量）、预算相关权重归一化；实验使用TransE、RotatE、DistMult、ComplEx四种KGE模型。

**📊 数据集**

使用的公开知识图谱数据集包括：FB15k237、WN18RR、YAGO3-10。

**📈 对比分析**

方法与15个基线（Uniform、KBGAN、NSCaching、Self-Adv等）在所有模型-数据集组合上对比，平均取得最高MRR和Hits@10，提升约10–15%；同时在训练过程中减少采样假负率90%以上；训练效率提升约30%，GPU内存占用下降约34%。

**⚠️ 局限性**

局限性：假设观测概率均匀且实体/关系集相同，未考虑异构观测或增量学习；未联合优化负样本生成策略；适用于有观测信息的闭合训练集，未在极大规模或低资源KG上进行验证。

---

## 651. VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning

**arXiv ID:** 2610.37378 | [PDF](https://arxiv.org/pdf/2610.37378v1)

**作者:** Bojun Zou `[一作]` (University of Science and Technology of China), Jianmin Ji `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 VLA‑Stem 的轻量级强化学习安全机制，能够在保持视觉‑语言‑动作（VLA）模型原始任务意图的同时，在推理时根据实时障碍信息动态调整执行轨迹，从而实现安全可靠的机器人操作。

**💡 创新点**

① 将 VLA 的短期动作序列映射为运动参考，并在此基础上设计轨迹级指导奖励，允许在空间和时间上灵活调整；② 引入教师‑学生蒸馏与课程学习，利用任务专属特权信息加速学习并将恢复行为迁移到无特权的部署策略；③ 通过无缝集成 VLA 参考与安全执行，避免了传统安全滤波器对任务连贯性的破坏。

**🔧 技术方法**

使用多臂 PPO 强化学习、轨迹级指导奖励、教师‑学生蒸馏、课程学习、低维障碍感知（四边形包围盒）以及 VLA 视觉‑语言输入；实现了基于 VLA 的短期动作查询、参考轨迹生成与可执行动作的联合策略。

**📊 数据集**

1) ManiSkill 仿真环境中的 PickCube、StackCube、Object Stowing 三个任务；2) 真实 UR7e 机器人上通过 Teleop 收集的 50 条演示用于 VLA 细调；3) 真实世界中使用 AprilTag 估计障碍位置的低维表示。

**📈 对比分析**

与基线 VLA（π₀、π₀.₅）、VLA‑finetune（对每个任务场景专门微调）以及 VLSA（基于 CBF 的安全滤波器）进行对比。结果显示：VLA‑Stem 在无障碍情形下保持与基线相近或略高的任务成功率；在静态/动态障碍场景下，VLA‑Stem 的无碰撞任务成功率平均提升约 60‑70%，碰撞率降至 15‑30%，明显优于 VLA‑finetune 与 VLSA 的表现。

**⚠️ 局限性**

仅在结构化的单一目标搬运任务上验证，难以处理更长时序任务、接触或力控制的安全需求；依赖低维障碍表示，可能在视觉噪声或复杂场景下失效；VLA 参考的质量直接影响安全策略表现，若 VLA 产生错误参考，系统安全保障有限。

---

## 652. Euclidean Crowding-Driven Adaptive Euclidean-Hyperbolic Graph Learning

**arXiv ID:** 2610.37373 | [PDF](https://arxiv.org/pdf/2610.37373v1)

**作者:** Hao Chen `[一作]` (Xi'an Jiaotong University), Hongying Zhang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了欧氏空间在树状结构图嵌入中的“层间拥挤”问题，给出了定量的几何分析和对泛化误差的理论解释；提出了可自适应融合欧氏与双曲几何的 EH‑Fusion 框架，并在单层指数映射层的基础上实现轻量级的双曲实现；

**💡 创新点**

1）首次用层间最小距离定量刻画欧氏拥挤，并与泛化边界关联；2）提出基于邻域扩张模式的自适应几何选择机制，兼顾多模态几何；3）证明单层指数映射已能捕获双曲优势，显著降低设计复杂度；

**🔧 技术方法**

欧氏到双曲的指数映射（Poincaré / Lorentz），多尺度邻域特征提取，注意力混合的几何融合，基于Rademacher复杂度的margin泛化分析，梯度下降优化（Adam / RSGD）；

**📊 数据集**

Cora、Citeseer、Airport、Disease、Ele‑Fashion 等四个标准图数据集；通过Gromov δ‑超弯曲度度量评估图的层级性；

**📈 对比分析**

与传统 Euclidean GCN/GAT、Poincaré Embeddings、HGCN、FHNN 等基线对比；实验显示 EH‑Fusion 在强层级（δ≈0~1）数据上提升 1–3% AUC，弱层级（δ>5）时保持稳定且优于单一几何；

**⚠️ 局限性**

对 δ‑超弯曲阈值缺乏定量判定；仅在单层映射下无法进一步捕获更深层级或更大尺度结构；适用于仅欧氏或仅双曲背景的任务时仍需实验验证；

---

## 653. Tokens of Motion: Primitive Motion Learning for Human-Humanoid Skill Transfer

**arXiv ID:** 2610.37330 | [PDF](https://arxiv.org/pdf/2610.37330v1)

**作者:** Hanwen Ren `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 M-PACT 框架，将人类手部视频中的运动分解为可迁移的原语，并在仅少量机器人演示下实现跨实体的抓取与操作。

**💡 创新点**

创新点在于：①通过 EPMT 将运动分段、离散化与连续旋转分离，得到可重用的原语词表；②将解码器与原语分离，形成轻量级的跨实体适配；③引入上下文条件的原语预测，实现闭环自主执行。

**🔧 技术方法**

使用技术包括：自监督分段器 SkillSeg、向量量化的 EPMT、跨实体解码器适配、上下文条件分类器 CC‑EPMT、CLIP 视觉与语言特征、机器人逆运动学等。

**📊 数据集**

使用数据集：EgoDex、HoloAssist、HumanoidEveryday，分别用于原语训练、冻结转移和 humanoid 适配。

**📈 对比分析**

与传统端到端重现、几何对齐或子任务拆分基线对比；在跨数据集/跨实体转移时，EPMT 词表利用率 100%，平均平移误差约 3 cm、旋转误差约 1°；仅 100 条机器人演示即可在 humanoid 上实现重现；上下文预测 Top‑1 约 40%，比直接机器人预测提升约 3%。

**⚠️ 局限性**

局限性包括：仅依赖手部姿态数据，未覆盖工具使用或多手协作；预测误差主要集中在旋转和握力，导致执行失败；对复杂环境的泛化仍有限。

---

## 654. Beyond the Readout: A Latent Numerical Frame Shared Across Modality and Operation

**arXiv ID:** 2610.37422 | [PDF](https://arxiv.org/pdf/2610.37422v1)

**作者:** Oriol Juan Sabater `[一作]`, Xavier Suau `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对跨模态（文本与图像）算术运算进行因果子空间分析，探究内部数字表征的共享性、几何关系以及时间调度机制。

**💡 创新点**

通过因果干预而非传统相似度度量，揭示数字表征在四种条件下共享旋转/缩放对齐的几何；发现13维读写器正交子空间承载完整数字信息；证明该子空间通过KV缓存在自回归生成的后续步骤中调度下一位数字。

**🔧 技术方法**

使用分布式对齐搜索（Distributed Alignment Search）学习因果子空间；旋转+缩放映射、正交群同步、RSA/CKA对比、激活补丁、自动回归干预等技术。

**📊 数据集**

使用合成整数算术表达式（0–99）以文本和渲染图像形式，分别进行加法与减法，确保模型在两模态下均能正确解答。

**📈 对比分析**

在四种条件下学习22维因果子空间，直接转移恢复率约45%；旋转+缩放后恢复率91%；13维正交子空间实现75.5% top‑1 20‑候选值检索；跨模态时间调度干预使下一位数字转向准确率77.6%。

**⚠️ 局限性**

仅在整数加减且数字范围为0–99的任务上验证；乘法、深层表示及多模型扩展缺失；未定位具体执行转置的注意力头；因果子空间维度固定；仅针对单一模型架构。

---

## 655. The Capacity of Private Information Retrieval from Coded Storage Systems with Colluding, Robust and Byzantine Servers

**arXiv ID:** 2610.37412 | [PDF](https://arxiv.org/pdf/2610.37412v1)

**作者:** Rui Sun `[一作]` (Shandong University), Yiwei Zhang `[通讯]` (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在具备碰撞、鲁棒与拜占庭服务器的MDS编码存储系统中，如何实现私有信息检索（PIR）并给出了线性、完整支持秩方案的容量上界及可达率，此外还扩展到多文件检索场景。

**💡 创新点**

创新点在于：① 推导了U‑B‑MDS‑TPIR的精确容量表达式（在N−U−2B≥KT且K=1或T=1时无线性约束也成立）；② 设计了基于Schur乘积码与统一选择/删减MDS性质的查询矩阵构造，实现对鲁棒与拜占庭错误的容错；③ 在多文件检索中首次给出容量上界与匹配构造，并在P≥M/2时证明了容量实现。

**🔧 技术方法**

采用的主要技术包括：线性私有信息检索、GRS码存储、Schur乘积码、均匀选择与删减MDS矩阵属性、随机全秩矩阵构造、信息理论上界推导等。

**📊 数据集**

由于研究属于理论信息论，未使用实际数据集，而是在符号域（如𝔽_q）上进行符号级分析与构造。

**📈 对比分析**

通过信息理论上界与构造实现的速率（rate）与已知的单文件或无碰撞情形相比，匹配了最优容量，尤其在K=1或T=1时实现了完全的容量。对一般K>1、T>1的情况，只在线性完整支持秩方案上给出上界。

**⚠️ 局限性**

局限性包括：① 对于K>1且T>1的非线性或非完整支持秩方案，容量尚未完全确定；② 方案依赖于大域存在（q足够大），对具体实现的实际性能与复杂度分析有限；③ 多文件容量的完整性仅在P≥M/2时得到证明。

---

## 656. Game-Theoretic Solvers with Bayesian Opponent Modeling and Moral Rhetoric Framing for GLEE

**arXiv ID:** 2610.37397 | [PDF](https://arxiv.org/pdf/2610.37397v1)

**作者:** Hayden Arnold `[一作]` `[通讯]` (Korea Advanced Institute of Science and Technology), Hayden Arnold (Korea Advanced Institute of Science and Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并评估了一个结合游戏理论求解器、贝叶斯对手估计与语言模型的多任务经济博弈代理，能在讨价还价、谈判与说服游戏中同时生成数值决策与策略性语言。

**💡 创新点**

将经典均衡求解器与动态贝叶斯对手建模、基于道德基础理论的多臂老虎机框架相结合，并为理论假设失效时引入针对性的守护阈值。

**🔧 技术方法**

Rubinstein 讨价还价、Faratin 让步、Kamenica‑Gentzkow 透露、贝叶斯后验推断、Thompson sampling 多臂老虎机、语言模型生成、道德基础理论与证据‑直觉切换。

**📊 数据集**

13,200 场对 11 种合成对手的 Monte Carlo 仿真；9,337 条直播说服信息；GLEE 竞赛的实时排行榜游戏。

**📈 对比分析**

通过对比启用与禁用对手估计模块的 ablation、合成对手平均得分、以及实时排行榜评分；在合成测试中启用模块平均得分 +0.238 / -0.251 / +0.269；在排行榜中最高显示评分分别为 1996.4、2131.1、1941.9，排名第55/198。

**⚠️ 局限性**

帧式分析仅为相关性；合成对手与真实对手不完全一致；对手识别率低导致跨局学习有限；代理在排行榜上的表现仅中等，未突破前沿。

---

## 657. When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG

**arXiv ID:** 2610.37358 | [PDF](https://arxiv.org/pdf/2610.37358v1)

**作者:** Zhouting Zhao `[一作]` `[通讯]`, Zhouting Zhao

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RAG系统中，设计并评估了一种基于成本感知的两动作检索策略，利用基础检索（稠密检索+重排序）产生的运行时特征预测是否需要调用更强的混合检索，从而在保持证据检索质量的同时减少不必要的检索调用。

**💡 创新点**

核心创新在于：①仅使用基础检索产生的特征来估计混合检索的增量证据收益；②在决策时引入单位成本惩罚，实现成本与收益的平衡；③将检索优化与生成质量的评估分离，系统性检验检索改进是否真正提升答案可靠性。

**🔧 技术方法**

使用稠密检索+BM25的RRF混合、交叉编码器重排序、岭回归预测增益、阈值决策；生成端采用Qwen3‑4B语言模型；评估采用Bootstrap聚类置信区间、论文级别的对比分析。

**📊 数据集**

实验基于QASPER科研论文问答基准，划分240篇文档用于开发、60篇文档用于最终评估（共188个问题，169个可评估证据）。

**📈 对比分析**

与“始终保持”和“始终混合”两基线对比；在证据层面，政策实现与始终混合相同的Hit@10（78.70%），但混合调用率仅为24.5%，显著降低了检索调用；但在答案正确性与可信度的评估中，政策与保持策略无显著差异，统计区间均包含零。

**⚠️ 局限性**

主要局限：答案质量评估缺乏独立人类验证，证据与答案质量不一定对应；混合检索成本未量化；样本量有限，救援/伤害实例稀少；仅在训练集同一拆分内评估，未检验对外部测试集的泛化。

---

## 658. HIDE: Hierarchical Isolation via Documentation and Evidence for Rustc Bugs

**arXiv ID:** 2610.37343 | [PDF](https://arxiv.org/pdf/2610.37343v1)

**作者:** Tianrong Zhang `[一作]` (Peking University Shenzhen Graduate School), Dan Hao `[通讯]` (Peking University Shenzhen Graduate School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为HIDE的分层LLM方法，用于定位Rust编译器rustc中的错误文件

**💡 创新点**

将编译器文档与错误证据相结合，构建文件/模块/crate级别的文档聚合，随后通过分层候选筛选与上下文约束来提高定位准确率

**🔧 技术方法**

使用大型语言模型（如GPT-5.4、Gemini‑3.1‑Pro‑Preview、DeepSeek‑V4‑Pro）进行文档生成、候选评估和代码片段裁剪，结合查询跟踪和编译日志进行执行证据分析

**📊 数据集**

在160个可重现的ICE/完整性/语义错误的rustc实例上进行评估，并使用其对应的修复PR生成文件级真值集；对比Agentless、SoapFL、GraphLocator等基线方法

**📈 对比分析**

在Accuracy@k（k=1,3,5,10）指标上，HIDE以42.08%、60.21%、70.83%和80.42%的成绩领先基线；平均执行时间154 s、平均token 145 k，显著低于Agentless、SoapFL和GraphLocator；不同LLM实验显示对k=10的鲁棒性，k=1时存在波动

**⚠️ 局限性**

对k=1的准确率仍相对有限；实验受限于仅包含ICE/完整性/语义错误，未覆盖性能或诊断类错误；对LLM随机性的依赖导致结果方差；在极大规模或结构变动频繁的编译器上可能需要重新调整层级和阈值

---

## 659. CodeCap-Bench: Can Code Serve as a Good Video Caption?

**arXiv ID:** 2610.37341 | [PDF](https://arxiv.org/pdf/2610.37341v1)

**作者:** Jiafu Tang `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探讨使用可执行代码作为视频字幕的表示方式，并构建了CodeCap-Bench基准。

**💡 创新点**

创新点在于用代码捕捉细粒度空间结构与时间动态，并提供多级视觉、时空和信息保留评估框架，创建了含323个结构化2D视频的公开数据集。

**🔧 技术方法**

采用多模态大型语言模型生成可执行代码，利用渲染引擎产生视频，并通过视觉相似度、对象级时空IoU及细粒度信息保留指标进行评估；同时比较代理式与非代理式生成方式。

**📊 数据集**

使用CodeCap-Bench数据集，涵盖桌面界面、2D游戏、视觉设计、数据可视化、图标动画五类结构化2D视频，并配有精细焦点与对象时空标注。

**📈 对比分析**

通过问答准确率和视频重建指标（Appearance Similarity、Motion Pattern Similarity、Keyframe Structural Similarity）对比代码字幕与自然语言字幕，结果显示代码字幕在问答和重建任务中均表现更佳，信息保留更完整。

**⚠️ 局限性**

限制包括LLM对复杂时间动态的捕捉能力不足，开源模型相较专有模型性能差距显著；代理式工作流虽提升渲染成功率，但对质量提升不稳定，且生成成本与效率不一致。

---

## 660. Mind the Intent Gap: A Methodology for Collaborative Agent-Driven Software Engineering

**arXiv ID:** 2610.37340 | [PDF](https://arxiv.org/pdf/2610.37340v1)

**作者:** Nicolas Tacheny `[一作]` (University of Mons), Saïd Mahmoudi `[通讯]` (University of Mons)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并在 Ni2 公司的软件团队中实践了 CAIA（Collaborative Agentic Intent Alignment）方法论，用于在引入 AI 代理时显式记录并对齐每次变更的意图，确保代理在执行前得到完整的上下文和团队共识。

**💡 创新点**

创新点包括：① 以意图为中心的流程，强制在每次变更前完成四步对齐（Describe、Analyze、Review、Align）；② 通过单一“Change Master”驱动 Build 与 Knowledge Agent，避免多源输入导致的不一致；③ 在 Knowledge Base 中持续记录决策及其理由，形成可复用的产品知识；④ 将人类与 AI 的协作关系转化为可验证的规则与工作流。

**🔧 技术方法**

技术方案基于大语言模型（LLM）驱动的代理，分为 Assistant、Build、Knowledge 三类，并配套一个支持 CAIA 流程的平台；同时利用知识库（知识记录、决策记录等）为代理提供上下文。

**📊 数据集**

实验数据主要来自 Ni2 公司内部的真实项目：19 条变更、4 个项目、约10名员工使用该平台；未使用公开数据集，而是以工业生产软件为实验对象。

**📈 对比分析**

计划在大学课程中进行对照实验：一组学生遵循 CAIA，另一组自由使用或不使用代理；对比指标包括每条变更所需时间、结果与预期意图的匹配度（准确性）以及修正次数。预期 CAIA 组在时间相近的前提下，能够得到更贴合意图、修正更少的结果。

**⚠️ 局限性**

局限性：① 目前仅在 Ni2 公司的特定流程和团队规模下验证，缺乏跨组织、跨行业的广泛验证；② 对意图质量高度依赖，若人类写入意图错误或不完整，仍可能导致代理误操作；③ 需要持续人工维护 Knowledge Base，过程成本仍未完全消除；④ 评估实验仍处于规划阶段，尚未获得客观性能数据。

---

## 661. CoS-VQ: Coordinated Shared-Scale Vector Quantization for Large Language Models

**arXiv ID:** 2610.37336 | [PDF](https://arxiv.org/pdf/2610.37336v1)

**作者:** Zukang Xu `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种CoS‑VQ框架，利用共享尺度下的协同码字索引更新实现低位宽向量量化。

**💡 创新点**

创新点在于构造多尺度候选池、使用残差感知切线可分割上界以及低包络区间搜索，从而在保持低维码字的同时实现全局最优的硬索引与尺度联合优化。

**🔧 技术方法**

核心技术包括残差感知主导量化、上界转化为分离二次型、低包络线段划分求解以及真误差验证。

**📊 数据集**

使用了 Qwen3、Llama‑3.1 系列大模型的权重，并在 WikiText‑2、C4、ARC‑Challenge、ARC‑Easy、HellaSwag、LAMBADA、PIQA、BoolQ、MMLU、WinoGrande、LiveCodeBench、GPQA‑D 等数据集进行评估。

**📈 对比分析**

与 BF16、PV‑Tuning、QuIP#、QTIP、NoWag、AQLM、LLVQ 等基线对比，在 2‑bit 低位宽下，CoS‑VQ 在大多数语言建模与下游任务上获得更低的困惑度与更高的准确率，同时保持与 BF16 相近或更优的推理吞吐。

**⚠️ 局限性**

局限性包括：优化仅在有限候选池内进行，理论最优性仅对上界保留；可能无法突破某些坐标停滞；对极低位宽、不同硬件或模型结构的适用性尚未充分验证。

---

## 662. ScIntBench: Physical Interaction Plausibility in 3D Scene Generation

**arXiv ID:** 2610.37326 | [PDF](https://arxiv.org/pdf/2610.37326v1)

**作者:** Wendi Ying `[一作]` (University of Western Australia), Ajmal Saeed Mian `[通讯]` (University of Western Australia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ScIntBench基准，用于系统评估文本条件3D室内场景在语义质量、静态物理可行性和主动交互行为上的交互准备度。

**💡 创新点**

首次定义四个交互指标（ISR、ICP、PIV、DP），结合多视角语义评估和静态物理检查，形成统一的交互评估框架，并对不同生成范式进行跨方法比较。

**🔧 技术方法**

利用Blender渲染多视角图像、Isaac Sim进行物理仿真交互、LLM（GPT‑5.6）生成交互任务并评估ICP、DP，辅以几何检查与可行性测试。

**📊 数据集**

使用SceneEval的30条文本提示（10室内类型）并扩展为203个交互任务，同时在ReplicaCAD、iGibson、BEHAVIOR‑1K上做验证。

**📈 对比分析**

评估12种文本条件3D场景生成方法，结果显示大多数方法在语义与静态物理上表现良好，但交互成功率普遍低，最高仅约36.9%，揭示语义质量与交互可靠性不一致。

**⚠️ 局限性**

生成场景常缺乏必要的交互能力或物理元数据，导致交互失败；评估仍依赖LLM主观评分，且未自动补全缺失的物理属性或关节，限制了生成质量的进一步提升。

---

## 663. Time-Dynamic Semantic-Activity Entanglement: Quantifying Communication-Based Synchrony in Collaborative VR

**arXiv ID:** 2610.37427 | [PDF](https://arxiv.org/pdf/2610.37427v1)

**作者:** Pooja Pol `[一作]` (Technische Hochschule Augsburg), Qing Huang `[通讯]` (Technische Hochschule Augsburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于时间动态语义-活动纠缠（TSAE）的新指标，用来量化协作 VR 环境中团队沟通的同步性。

**💡 创新点**

创新点在于将说话量与语义相似度结合为一个时间分辨的纠缠度量，并通过 Gini 系数描述纠缠在团队成员间的分布。

**🔧 技术方法**

使用 Whisper 进行自动语音转录，随后用 Jina v2 模型进行 late‑chunking 嵌入，并在离散时间窗口内计算 TSAE 和 Gini 系数。

**📊 数据集**

采用了 15 个德国语协作 VR 游戏会话的音频数据，团队规模为 5–8 人，平均时长约 25 分钟。

**📈 对比分析**

本文主要进行描述性分析，展示了单一会话的热图、Gini 轨迹以及跨会话的 Lorenz 曲线与 Gini 分布；未与其他现有指标做精确对比，但表明纠缠分布相对均匀，随时间窗口增大集中度下降。

**⚠️ 局限性**

局限性包括仅针对德语数据，缺乏对 ASR 误差的鲁棒性评估，样本规模小，未结合任务绩效或其他行为数据验证纠缠指标的预测意义。

---

## 664. Privacy-Preserving Over-the-Air Federated Learning over Cell-Free Massive MIMO System with User-Centric Clustering

**arXiv ID:** 2610.37424 | [PDF](https://arxiv.org/pdf/2610.37424v1)

**作者:** Manobendu Sarker `[一作]` (Polytechnique Montréal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montréal)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单元化分布式 Massive MIMO 网络上设计了跨层的隐私保留 OTA 联邦学习框架，联合优化 AP‑UE 聚类、功率分配和差分隐私噪声，以最小化聚合方差。

**💡 创新点**

创新点是将物理层热噪声视为自由的隐私掩蔽资源，并通过 ergodic zCDP 设计构建可调的隐私-噪声替代机制，揭示了隐私受限与噪声受限两种运行 regime，且实现了自适应的聚类与功率控制。

**🔧 技术方法**

采用了用户中心化 Cell‑Free Massive MIMO、over‑the‑air analog聚合、差分隐私的 clip‑and‑perturb Gaussian 机制、零集中化差分隐私(zCDP) 计量、凸-凹程序 (CCP)、贪心坐标下降聚类和基于 Chebyshev 的 channel hardening 置信度评估。

**📊 数据集**

在 MNIST 和 CIFAR‑10 数据集上进行端到端的 FedAvg 训练验证，使用单层 CNN 架构，评估通信与隐私下的分类准确率。

**📈 对比分析**

与五种基线（隐私无视、固定聚类、等功率、单单元格、无噪声）相比，该方法在隐私预算 ε≥10 时将聚合方差降低至 89% 以内，MNIST 上可达 97% 准确率，CIFAR‑10 上可达 48%，远优于基线。

**⚠️ 局限性**

局限包括对大规模天线阵列的 channel hardening 依赖性、对完整的全局干扰与噪声模型简化、只考虑全参与的 UE 与静态大尺度统计，且在 M=4 的场景下对热噪声信用的保守估计可能导致性能过度保守。

---

## 665. Neighborhood Deprivation Shapes Post-Flood Population Losses in Pune and Bandung

**arXiv ID:** 2610.37418 | [PDF](https://arxiv.org/pdf/2610.37418v1)

**作者:** Sukie Xiuqi Yang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Catherine Chen `[通讯]` (Louisiana State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究使用移动设备检测数据分析印度浦那和印度尼西亚万隆两座城市洪灾后社区人口短期流动模式。

**💡 创新点**

创新点在于将卫星洪灾边界与移动设备数据结合，使用数据驱动聚类方法识别不同流动轨迹，并揭示低收入社区在洪灾后流动受限的规律。

**🔧 技术方法**

技术上采用层次聚类、逻辑回归和空间自相关检验。

**📊 数据集**

数据集包括NetMob 2024移动设备检测数据、DFO洪灾多边形、GRDI贫困指数及ASTER高程数据。

**📈 对比分析**

比较方法是将聚类得到的流动轨迹与社会经济变量和洪灾暴露程度进行关联分析，结果显示经济条件与流动轨迹显著相关。

**⚠️ 局限性**

局限性包括洪灾边界的粗略性、设备数据的代表性偏差、缺乏个体迁移轨迹以及未能充分识别洪灾外因素对流动的影响。

---

## 666. The Envelope Layer: An Underestimated Surface of LLM Agent Defense

**arXiv ID:** 2610.37415 | [PDF](https://arxiv.org/pdf/2610.37415v1)

**作者:** Yuelin Han `[一作]` `[通讯]`, Yuelin Han

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 A2A 代理攻击场景中，系统通过对“信封层”进行标签强度与真实性、以及历史记忆的操控，使用 1200 细胞的 2×2 因子实验和 320 细胞的记忆实验，系统性评估了信封层防御的安全边界。

**💡 创新点**

创新点在于：① 将信封层从单一通道扩展为两级结构（入口通道 + 上下文组装面）；② 引入“信封真实性”与“记忆价值”两种实验因子，揭示标签强度、伪造与记忆对防御效果的动态交互；③ 采用 A/B/C/D 四分法与 GDA 测量框架，分层评估不同防御层级的失败点。

**🔧 技术方法**

技术手段包括：A2A‑TIBA 代理链攻击、自然语言标签注入、系统提示与技能声明拼装、LLM 背景模型（MiniMax‑M3、deepseek‑v4‑pro、glm‑5.3、kimi‑k2.7‑code、qwen3.8‑max）接口集成、LiteLLM 桥接、判定裁决双重判定及人工复核。

**📊 数据集**

使用的“数据集”为 15 种前端后端组合（3 个前端代理×5 个 LLM 后端），每组 20 次攻击实验，覆盖 1200 细胞；记忆实验使用 4 种记忆状态（失败、成功、成功‑先、失败‑先），每组 20 次，形成 320 细胞。

**📈 对比分析**

结果显示：在原始信封下，强标签可显著降低语义突破率（最高 81.2%↓ 至 6.2%）；伪造信封会显著提升突破率，且效应因前端而异；记忆状态对突破率影响远大于标签强度。与传统单一 ASR 评估相比，四分法揭示了防御失效点，性能表现因后端与前端组合差异显著。

**⚠️ 局限性**

局限性包括：实验未随机分组；每个细胞仅 20 次复现，置信区间宽；攻击链固定且未对抗性攻击；缺乏加密身份验证；记忆实验组选择受基线偏倚；判定者与攻击者使用相同 LLM 可能产生同源偏差。

---

## 667. AuRaSim: A Ray-Tracing Based Automotive FMCW Radar Simulator for Dynamic Scenarios

**arXiv ID:** 2610.37403 | [PDF](https://arxiv.org/pdf/2610.37403v1)

**作者:** Zhaoze Wang `[一作]` (HELLA GmbH & Co. KGaA), Markus Gardill `[通讯]` (Brandenburg University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并实现了一个基于Blender和Sionna RT的端到端汽车FMCW雷达仿真框架AuRaSim，能够从数字孪生场景生成路径解析的电磁传播数据、可视化的ADC样本和雷达产品，支持动态场景、材料属性、局部速度场与微多普勒效应；

**💡 创新点**

创新点在于：①将Blender的动态场景构建与Sionna RT射线追踪相结合，形成统一的物理仿真流水线；②提出视场限定的Fibonacci采样、路径过滤和局部表面速度场，实现更高效、更真实的多普勒建模；③引入显式路径表示和可选的合成/显式MIMO阵列建模，兼顾精度与计算效率；

**🔧 技术方法**

使用技术包括：Blender 3D建模与动画、Sionna RT GPU射线追踪、Mitsuba 3渲染后端、频率滑坡的FMCW信号模型、ADC采样合成、相位补偿的MIMO阵列建模、数字孪生重建与验证、Ray tracing优化与路径过滤；

**📊 数据集**

采用Gao等人公开的实测雷达与RGB图像数据集（约19.8k帧，77 GHz雷达），用于重建数字孪生场景并与仿真结果进行对比；

**📈 对比分析**

通过对测量与仿真生成的Range‑Doppler图进行RMSE和PSNR评估，AuRaSim相较于RadarSimPy平均RMSE降低约38.8%，PSNR提升3.5 dB；在RTX 3060 GPU上，单帧渲染时间约0.8 s；

**⚠️ 局限性**

局限性在于：材料电磁参数精度不一导致仿真误差；未模拟噪声、相位噪、干扰、校准误差和恶劣天气效应；高阶多路径与大规模动态场景仍计算量大；近场或波前曲率对大阵列的影响尚未完整考虑。

---

## 668. Code uniqueness of 4-cell connected disjoint clusters in spatial chromatic model

**arXiv ID:** 2610.37395 | [PDF](https://arxiv.org/pdf/2610.37395v1)

**作者:** Weining Zhu `[一作]` `[通讯]` (Zhejiang University), Weining Zhu (Zhejiang University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

证明了空间染色模型（SCM）中四单元连通不相交簇（4‑ccd）的染色码唯一性；

**💡 创新点**

首次提出t‑码、l‑码、r‑码等概念，并通过组合学与几何邻接分析证明了4‑ccd的唯一性，为后续更大簇的研究奠定理论基础；

**🔧 技术方法**

采用组合数学、几何图论、染色码分析以及值/位置转换（TE、TL）等技术手段；

**📊 数据集**

研究基于理论模型，不涉及具体实验数据集；

**📈 对比分析**

通过严谨的理论证明完成验证，未进行实验比较或性能评估；

**⚠️ 局限性**

仅适用于4‑ccd簇，未能推广至5‑ccd或更大簇，证明过程复杂，难以直接扩展。

---

## 669. Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data

**arXiv ID:** 2610.37386 | [PDF](https://arxiv.org/pdf/2610.37386v1)

**作者:** Geunhyoung Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种不需要校准数据的 MoE 通道选择方法（Reader），通过权重本身的旋转三元量化草图与低秩恢复映射，实现了在保持 98.1% 原模型准确率的前提下仅使用 17.8% 的专家权重读取。

**💡 创新点**

创新点在于：①证明传统阈值校准既不能保证读取量，也无法带来一致的准确率提升；②提出完全基于模型权重的 Reader，采用旋转三元草图筛选通道并利用低秩映射恢复被跳过的通道，既省去了校准过程，又避免了阈值导致的重复循环；③在三种大规模 MoE 模型上展示了显著的速度提升（MoE 块 1.6–2.69×，整体模型 1.15–1.88×）。

**🔧 技术方法**

使用技术包括：旋转 Hadamard 变换 + 三元量化草图、输出尺度评分、精确 top‑k 计算、低秩矩阵恢复、Triton GPU 内核优化，以及 vLLM 适配。

**📊 数据集**

使用的数据集和评测任务包括 Qwen3‑Thinking、Gemma‑4、Qwen3.5 三个后训练 MoE 模型，评测基准为 GPQA‑Diamond、IFEval、MMLU‑Pro 等对话与推理任务。

**📈 对比分析**

与 Prox、TEAL、WINA 等已有基线相比，Reader 在 12.8% 与 17.8% 的读取预算下平均准确率分别为 94.5% 与 98.1%，远超阈值版本；在相同读取量下的速度提升为 MoE 块 1.60–1.90×、整体模型 1.15–1.88×；阈值版本在高稀疏度下易陷入重复循环，Reader 能避免此问题。

**⚠️ 局限性**

局限性包括：①内核优化仅针对单一预算与单一 GPU（RTX PRO 6000 Blackwell）进行调优；②目前仅针对解码阶段的路由专家做通道选择，未覆盖注意力层；③部分读操作（如部分草图）在不同模型/数据集上参数未必最优；④在量化模型中草图与恢复映射所占的额外内存比例更高。

---

## 670. PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations

**arXiv ID:** 2610.37381 | [PDF](https://arxiv.org/pdf/2610.37381v1)

**作者:** Hanqing Wang `[一作]` (JD.com), Youzheng Wu `[通讯]` (JD.com)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练了一种基于Vision‑Language‑Action的轻量级物理动力学模块，能够从VLM特征中抽取物理token并直接用于动作生成，而不需在线视频生成。

**💡 创新点**

创新点在于结合未来预测和预训练视频世界模型的辅助监督，生成compact物理token；使用人到机合成、子任务标注、ECoT等全公开数据扩展；在四大模拟基准上实现最高成功率。

**🔧 技术方法**

技术包括Qwen3.5‑4B VLM骨干、物理动力学模块（跨层查询）、Diffusion Transformer动作专家、直接未来特征预测、世界模型辅助流匹配、子任务与ECoT语言监督、动作与VL混合训练。

**📊 数据集**

数据集：公开的VLA数据（Real‑Robot、UMI、Simulation）与VL数据（General VL、Spatial & Embodied VL）共计约2.6M轨迹、1.1B帧；通过人到机合成、子任务标注、ECoT扩充。

**📈 对比分析**

在LIBERO、LIBERO‑Plus、RoboTwin 2.0、SimplerEnv四个仿真基准上评估，平均成功率分别为99.3%、89.9%、93.6%、81.5%，超越现有基线（Qwen‑RobotManip、InternVLA‑A1.5 等）。

**⚠️ 局限性**

局限：模型仍依赖大量公开数据，未在真实机器人上验证；物理token只捕捉RGB/深度/光流信息，对物体几何、接触、材质等更细粒度动态的捕捉有限；推理过程中需要VLM预训练，模型规模较大。

---

## 671. TRIDENT-Seq: Profile-Matched Prefix-Future Contrast for Label-Efficient Payment Risk Modeling

**arXiv ID:** 2610.37365 | [PDF](https://arxiv.org/pdf/2610.37365v1)

**作者:** Weiyu Han `[一作]` (Ant International), Tianyi Zhang `[通讯]` (Ant International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一种基于Prefix–Future Contrast（PFC）的交易基础模型TRIDENT‑Seq，用以在多家支付业务中实现低标记适配和跨业务迁移的风险评估。

**💡 创新点**

创新点在于引入按活动特征匹配的对比学习，利用未来摘要和活动预测、离散正则化等辅助任务，突破传统对比学习易被交易量“捷径”影响的局限；同时提供单一预训练编码器可复用给不同业务。

**🔧 技术方法**

技术包括Bloom‑filter哈希式事件编码、Transformer历史编码器、PFC对比学习（profile‑matched negatives）、活动预测头、方差正则化、以及业务级监督微调和风险头。

**📊 数据集**

使用了约2450万条Ant International未标记交易历史以及三家业务（Wallet、Shopping、SME）的标记样本，历史覆盖度从几乎完整到>70%无历史不等。

**📈 对比分析**

与scratch、CoLES、BYB、CPC、TransactionGPT、PANTHER等预训练方法以及生产级XGBoost/SAINT模型对比；TRIDENT‑Seq在0.5%干预预算下AP和召回率提升2–3倍，在线A/B实验中chargeback和总损失率分别下降≈37%。

**⚠️ 局限性**

局限性包括对缺失历史业务的迁移效果有限、微调结果波动大、对未标记数据质量依赖强、需进一步探索编码器学到的特征以及大规模预训练所需的计算资源。

---

## 672. Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation

**arXiv ID:** 2610.37543 | [PDF](https://arxiv.org/pdf/2610.37543v1)

**作者:** Dequan Ou `[一作]` (Delft University of Technology), Guido C. H. E. de Croon `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种无渲染的解析事件模拟器，用于在无光照环境下训练自主机器人导航策略，并在实机上实现全飞行控制

**💡 创新点**

创新点在于直接从场景几何和机动产生事件流，省去渲染步骤，显著提升模拟速度（400×）并实现零调优的模拟到现实迁移

**🔧 技术方法**

结合解析事件生成、PPO强化学习、Int8量化推理以及STM32N6微控制器的Neural-ART NPU实现低功耗飞行控制

**📊 数据集**

使用由随机障碍构成的程序化合成场景进行训练，无需真实传感器数据集

**📈 对比分析**

在与深度基准、手工控制和渲染事件对比时，零调优策略在模拟中达到83%/70%成功率，在三套真实实验中平均82%成功，训练时间仅需34分钟，远快于传统渲染方式

**⚠️ 局限性**

局限性包括对纹理密集场景的鲁棒性下降、事件密度与真实传感器仍有差距，以及仅适用于当前硬件与场景规模

---

## 673. From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents

**arXiv ID:** 2610.37482 | [PDF](https://arxiv.org/pdf/2610.37482v1)

**作者:** Ziyang Chen `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于任务进展的 GUI 记忆框架 Task Advancement Memory（TAM），用于在跨应用的长周期工作流中记录并重用已验证的操作和信息。

**💡 创新点**

创新点在于：① 通过任务进展状态（Task Advancement State）将已完成的子任务结果转化为对剩余目标的精确表示，实现在线任务减法；② 通过进展监督（Advancement Supervision）将局部已验证结果抽象为可重用的操作（Advancement Operator），实现跨任务的经验迁移。

**🔧 技术方法**

技术核心包括：基于 LLM 的任务编译器构建需求图；基于观察的任务进展判定与证据校验；在线状态更新与任务前沿选择；记忆检索与操作抽象；以及基于图的持久化操作库。

**📊 数据集**

使用的评测数据集为 MobileWorld GUI‑only 子集和 SPA‑Bench（单应用英语 Level 3）；所有方法均使用 Qwen3‑VL‑32B‑Instruct 或 GPT‑4o 作为基线模型。

**📈 对比分析**

与 6‑7 号基线（MemGUI‑Agent ConAct、AWM、Mobile‑Agent‑E、StructAgent、GUI‑explorer 等）以及公开结果比较，TAM 在 MobileWorld 上成功率提升至 34.2%（比最佳基线 29.1% 提升 5.1pp，EQA@50 29.2%，生命周期代价降低 38.7%），在 SPA‑Bench 上成功率达 57.0%（比 GUI‑explorer 53.7% 提升 3.3pp）。

**⚠️ 局限性**

局限性包括：对模型对任务需求和 GUI 证据的解释准确性高度依赖，导致在模糊的结果提示下可能出现误判；评估仅覆盖移动设备的 GUI 任务，缺乏 Web/桌面环境的验证；以及在长任务序列或应用更新频繁的情况下，操作库维护与迭代的挑战。

---

## 674. PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions

**arXiv ID:** 2610.37471 | [PDF](https://arxiv.org/pdf/2610.37471v1)

**作者:** Tongkun Guan `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PerceptRL 框架，在 MLLM 的后训练阶段通过模型自身的感知错误生成可验证的单轮 QA 数据和坐标注入的多轮工具交互数据，结合定位-aware 过程奖励提升视觉感知能力。

**💡 创新点**

创新点在于将开放式字幕的感知错误自动转化为可计分的 QA 对，并通过坐标注入纠正多轮工具操作中的坐标漂移，同时在 RL 中加入针对中间定位步骤的奖励，解决了传统字幕优化难以评估且缺乏定位精度的问题。

**🔧 技术方法**

采用多智能体错误检测与审核、QA 合成、规则奖励、DAPO 强化学习、Python 代码解释器、坐标注入（Coordinate Injection）与定位-aware 过程奖励等技术。

**📊 数据集**

使用自检生成的错误驱动数据，覆盖 8 个视觉感知基准（RealWorldQA、MMStar、BLINK、TreeBench、ERQA、RefSpatial、CountQA、RealWorldCounting），并在这些基准上评估模型性能。

**📈 对比分析**

与基线模型、前沿 VLMs（如 Gemini-3-Flash、Qwen3.5-Flash 等）在共享工具接口下对比，单轮 Perception RL 将平均准确率从 61.9% 提升至 69.8%，多轮 Agentic RL 再提升至 71.6%，在 7 个基准上超过前沿模型 5.2% 的平均提升。

**⚠️ 局限性**

局限性包括：训练过程需要多阶段 RL，计算成本较高；目前仅针对静态图像，未扩展到视频时序；坐标注入与定位奖励的设计和阈值需要人工调优。

---

## 675. M4Bench: Evaluating Procedural Specification for Clinical EHR Derivation Agents

**arXiv ID:** 2610.37459 | [PDF](https://arxiv.org/pdf/2610.37459v1)

**作者:** Hannes Ill `[一作]` (Massachusetts Institute of Technology), Leo Anthony Celi `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并发布了一个基于电子健康记录（EHR）的临床衍生变量重现基准（SkillsBench），涵盖28个任务、15个任务族，配有临床专家评审的“技能”markdown文档，并提供Docker隔离的评估管道；

**💡 创新点**

创新点在于：①将临床衍生规则转化为可审计、可版本化的“技能”文件；②设计多种对照（无技能、伪技能、原始SQL、操作规范、模式扰动）拆解技能效应来源；③对大型语言模型（Codex、Claude）进行系统化实验并发布完整实验记录；

**🔧 技术方法**

主要技术包括：大型语言模型（Codex、Claude）、Docker容器化执行、基于DuckDB的任务数据库、Markdown技能规范、自动化评估脚本（匹配率、Key F1、MAE）、随机化抽样与自助法置信区间、干扰测试与消毒审计；

**📊 数据集**

使用的公开数据集是MIMIC‑IV v3.1与eICU Collaborative Research Database v2.0（需要PhysioNet凭证），每个任务从原始表复制到DuckDB实例；

**📈 对比分析**

比较方法为对照实验：无技能 vs 有技能，计算每列匹配率（Reward）以及Key F1。结果显示Codex在所有28个任务上平均提升约0.136，Claude约0.120；提升主要体现在数值列匹配度，Key F1提升微乎其微，严格通过率仅提升约15%；

**⚠️ 局限性**

局限性包括：①仅测量对声明参考实现的匹配度，未评估临床正确性；②实验数据集单一（MIMIC、eICU），未覆盖不同医院或地区；③技能对模型基础能力的提升有限，关键行一致性提升不显著；④Claude实验存在版本与路由差异，结果仅为描述性；⑤未实现跨平台或跨数据源的泛化验证。

---

## 676. Test-Time Communication in Trading Agent Swarms

**arXiv ID:** 2610.37448 | [PDF](https://arxiv.org/pdf/2610.37448v1)

**作者:** Avid Fayaz `[一作]` (Infinite Ascent), André Silva `[通讯]` (Infinite Ascent)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对200个语言模型交易代理在真实美国股市上进行7个交易日的实验，比较5种测试时通信机制（隔离、公告板、带ping公告板、同伴记忆、协作公告板）对词汇、持仓、交易行为和信息质量的影响。

**💡 创新点**

首次在活跃金融市场中量化测试时通信的实际效应，并通过“无同伴”基准区分同伴影响与共通输入；揭示通信促进词汇共识、持仓重叠和同步卖出，但对当日交易决策影响有限；同时揭示信息传递中资格丢失和纠错延迟的问题。

**🔧 技术方法**

使用大语言模型（Opus 5.5、GPT‑6.1 Sol、Grok 4.6、DeepSeek V4 Pro）作为交易经理；设计自定义通信协议、队列化执行、手工记录对话与研究笔记；利用Python脚本进行收益、重叠、词汇传播、同步卖出和研究请求的统计与可视化。

**📊 数据集**

真实美国股票市场的实时行情；每个代理在纸质账户中管理10万美元，7天的交易记录和每日收盘价；实验期间使用的所有公开工具与模型权重均记录在实验档案中。

**📈 对比分析**

通过将每个通信组与同一模型的隔离组对比，使用配对差异检验、bootstrap 置信区间和置换检验评估影响。结果显示：通信组持仓重叠率提升 1.5–2 倍；共享持仓解释约 70% 的组收益；词汇共识提升约 7%；同步卖出比例从 40–60% 上升到 70–90%；但同日交易方向差异无显著提升，整体回报仍低于 SPY。

**⚠️ 局限性**

实验规模有限（每种模型-通信组合仅一组），缺乏随机化和中性框架；通信内容被标记为“非可信”，可能抑制合作；仅观察 7 天，无法评估长期绩效；未将得分与决策分离，难以分离得分刺激与通信本身的效应；对同伴记忆与公告板等机制的可比性受限。

---

## 677. CRISP: Co-designing Pruning and Scheduling for Efficient MoE Inference on Edge Servers

**arXiv ID:** 2610.37442 | [PDF](https://arxiv.org/pdf/2610.37442v1)

**作者:** Congwei Zhang `[一作]` (Shandong University), Dongxiao Yu `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CRISP框架，对Edge平台MoE推理实现重要性感知神经元重排、多层次组合裁剪与宽度感知I/O调度的协同设计，从而在有限显存下显著提升吞吐量并控制精度损失。

**💡 创新点**

创新点包括：① 通过激活量重要性排序实现单份专家权重支持多宽度裁剪；② 基于专家组合热度与角色的分层裁剪策略，保留高频组合的完整专家；③ 设计宽度感知调度器，使宽专家计算窗口覆盖窄专家加载，最大化I/O-计算重叠。

**🔧 技术方法**

使用激活量重要性评分、组合频率统计、神经元重排、分层宽度裁剪、CUDA异步I/O/计算流、定制核加速、以及与量化方法兼容的权重表示。

**📊 数据集**

主要使用C4校准集、WikiText-103、C4、ShareGPT、MMLU、HellaSwag、ARC-Challenge、WinoGrande等公开数据集进行性能与准确性评估。

**📈 对比分析**

与Full Offloading、Uniform Pruning、MoE-Offload、EdgeMoE等基线在Mixtral‑8x7B和DeepSeek‑MoE‑16B上比较，10%专家内存预算下吞吐提升至2.1×，精度提升仅≤3.6%；在不同GPU/UMA平台下能耗下降≈75%，在多种内存预算下保持1.4–2.1×速率优势。

**⚠️ 局限性**

局限性包括：需先验组合频率与神经元重要性统计，输入分布或模型更新后需重新校准；分层阈值固定，可能不适用于所有模型；裁剪宽度只支持预设比例（如50%），不支持任意宽度；极端低内存或高度动态路由场景仍受I/O瓶颈限制。

---

## 678. Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning

**arXiv ID:** 2610.37437 | [PDF](https://arxiv.org/pdf/2610.37437v1)

**作者:** Zengqi Zhao `[一作]` (University Of North Carolina Chapel Hill), Yue Zhang `[通讯]` (University Of North Carolina Chapel Hill)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于模拟器的空间想象基准，系统评估了同一未观测空间状态的 RGB、深度、全景、顶部视角等多种表示对空间推理的影响，并基于此提出了自适应表示选择策略 GEARS。

**💡 创新点**

创新点包括（1）提供多种同一场景的不同表征以隔离表示选择对推理性能的影响；（2）发现即使 oracle RGB 精确也不一定提升推理准确率；（3）提出了基于预测一致性门控和路由选择的轻量级自适应表示选择方法。

**🔧 技术方法**

使用的技术包括 Matterport3D 场景渲染、HY‑World 2.0 与 Stable Virtual Camera 的估计生成、GPT‑6.1‑sol 与 Gemini 3.7 Flash 视觉语言模型、以及 GEARS 的门控+路由模块。

**📊 数据集**

使用的数据集是自构造的 922 题空间想象基准，采自 Matterport3D，涵盖 Viewpoint Imagination 与 Cross‑view Integration 两种任务。

**📈 对比分析**

方法通过与多种基线（仅观察、oracle RGB、随机选择、各单一表示）对比评估，GEARS 在 Viewpoint Imagination 上达 61.4% 及 Cross‑view Integration 上达 77.7% 的准确率，显著优于 MindJourney 与 AVIC，虽然与 oracle 上限仍有差距。

**⚠️ 局限性**

限制包括仅研究静态室内场景、仅覆盖有限的空间推理任务、对动态环境和更复杂交互的评估缺失，以及估计表示在某些任务中仍无法完全弥补 oracle 差距。

---

## 679. Kernel Performance Engineering under Physical Non-Idealities

**arXiv ID:** 2610.37466 | [PDF](https://arxiv.org/pdf/2610.37466v1)

**作者:** Supun Kuruppu `[一作]` (National University of Singapore), Tulika Mitra `[通讯]` (National University of Singapore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了 NVIDIA H100 GPU 在实际工艺缺陷（floor sweeping）和功耗限制（TDP）下，分布式共享内存（DSMEM）与线程块聚类对 kernel 性能的影响，并提出了一种基于执行波数的 Wave‑aware Clustering 策略，提升了 FlashAttention 4 等工作负载的吞吐量。

**💡 创新点**

创新点包括：
1) 通过逆向工程 Skyline，揭示 floor sweeping 使得 GPC‑级聚类可用 SM 数量从 132 降至 120，导致额外的执行波数；
2) 定义了 Wave‑equalized kernel launch 条件，阐明在不同 cluster size 下保持相同执行波数的数学条件；
3) 依据执行波数自动选择 TPC 或 GPC 聚类的 Wave‑aware Clustering 策略，显著提升吞吐量；
4) 证明在 TDP 限制下，降低数据移动可为 GPU 提供频率 headroom，从而提升性能；
5) 将上述分析扩展到 Tenstorrent Wormhole 加速器，证明 TDP 对非 GPU 加速器吞吐量的普适影响。

**🔧 技术方法**

所用技术与方法：
- DSMEM 与线程块聚类技术；
- CUDA 微基准（global‑load‑and‑broadcast）、GEMM、FlashAttention 4；
- NVIDIA Nsight Compute（ncu）性能分析；
- 微观建模与分析（执行波数、SM 频率、L2 访问量）；
- Skyline 逆向工程与物理资源映射；
- TDP 与 DVFS 的功耗‑频率关系分析；
- Tenstorrent Wormhole 的功耗‑频率实验。

**📊 数据集**

使用的数据集与实验设置：
- FlashAttention 4（非因果多头注意力）在 2K–64K 的序列长度、不同 head dimension、batch size、tile 组合；
- GEMM 矩阵乘法（尺寸如 8192×4096×8192、7680×4096×512 等）
- Global‑load‑and‑broadcast 微基准（1 MB 数据）
- Tenstorrent Wormhole 片上矩阵乘法（HiFi4 / LoFi 精度）

**📈 对比分析**

比较方法与性能结果：
- 与 cluster size 1 的基线对比，计算吞吐量提升百分比；
- 采用 ncu 收集 SM 频率、Tensor Core 利用率、L2 读写量、功耗等指标；
- FlashAttention 4 在 Wave‑aware Clustering 下相较基线提升最高 5.30%；
- GEMM 在满足 wave‑equalization 条件时，GPC‑级聚类提升约 7.32%；
- 通过 DVFS 观察 SM 频率提升带来的吞吐量提升，相关系数 >0.9；
- Wormhole 实验显示，当核心数逼近 TDP 限制时，频率下降导致吞吐量饱和。

**⚠️ 局限性**

限制与待改进点：
- Wave‑equalization 条件仅适用于特定线程块计数范围（B ∈ ⋃_{n=1}^{10}(132(n−1),120n]）；
- 仅针对 H100 GPU 的 SM 与 Cluster 约束，其他 GPU 需要重新逆向工程；
- 未考虑其他物理非理想性（如热偏移、功率波动、寄存器压力等）；
- 需要手动收集 SM ID 或通过特定基准推断，增加工程成本；
- 对多 GPU、跨节点分布式计算的影响未作系统性评估。

---

## 680. Strategyproof Deep Reinforcement Learning for Online Resource Allocation

**arXiv ID:** 2610.37504 | [PDF](https://arxiv.org/pdf/2610.37504v1)

**作者:** Fariba Dehghan `[一作]` (University of Southampton), Stephanie Gauthier `[通讯]` (University of Southampton)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种在线资源分配机制，能够在未知未来需求和自利报错的情况下最大化社会福利，同时保证激励兼容性。

**💡 创新点**

1) 构造了通过平滑最小-最大网络实现的单调非线性接受规则，能够捕捉多资源占用交互；2) 通过数值搜索与精确区间运算得到的关键支付，确保策略证明；3) 允许用强化学习或模仿学习训练同一网络，保持激励兼容。

**🔧 技术方法**

平滑最小-最大神经网络（SMM）、强化学习（SARSA/Fitted‑SARSA）、模仿学习（BC/DAgger/DART）、区间算术/Arb多精度数值搜索、奖励设计与策略证明。

**📊 数据集**

多种合成任务流（Partner、Small、Large、Continuous、Complementary），每个流包含不同资源、价值、时长和需求分布，用于训练和评估。

**📈 对比分析**

与线性单调边界、其他单调网络（LipschitzMono、SMNN、MonoDense、MonoNet、PMLinear）以及手工规则（FCFS、Union）对比；在Partner流上SP‑SMM获得97.8% OPT，RL 23,255s vs imitation 261s，速度提升89倍；在其他流中保持与最佳方法相当或优于。

**⚠️ 局限性**

假设任务到达时间可观测，且每个任务仅与机制交互一次；未考虑可延迟报到或多次互动的情形。

---

## 681. AMOR: Adaptive Modal Operator Refinement for Operator Learning on Arbitrary Meshes and Irregular 3D Geometries

**arXiv ID:** 2610.37485 | [PDF](https://arxiv.org/pdf/2610.37485v1)

**作者:** David Voderholzer `[一作]` (Lawrence Berkeley National Laboratory), Thomas O'Leary-Roseberry `[通讯]` (Ohio State University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Adaptive Modal Operator Refinement（AMOR），一种将 Fourier Neural Operator（FNO）中的傅里叶谱卷积替换为可自适应的低秩基底块，从而实现对任意网格的神经算子学习。

**💡 创新点**

创新点在于：①通过跨基准的非局部块与无坐标编码，直接在任意不规则网格上构造算子；②证明该架构在包含常数与点分离基底时对连续算子具有全局逼近能力；③提出可训练与预定义多种基底（POD、DIS、Laplacian、多项式、学习型正交基）以及因式分解的低秩系数张量。

**🔧 技术方法**

技术实现包括：有限元/离散基底投影、稀疏/因式分解的低秩系数张量、FiLM 条件化、可训练基底（POD、DIS、Laplacian、多项式、学习型正交基）以及门控残差块。

**📊 数据集**

使用了八个基准数据集进行验证，涵盖二维非线性扩散、Helmholtz、Kolmogorov流、三维线性弹性、流体流动、航空翼形、弹性点云、机翼骨架等多种工程和物理问题。

**📈 对比分析**

与 FNO、Geo-FNO、PCANet、GraphNO 等方法比较，AMOR 在结构化网格上与 FNO 相当甚至更优，在三维和不规则网格上实现更低误差或更快推理；在机翼骨架等工业案例中显著优于现有基于格点或图网络的算子。

**⚠️ 局限性**

局限性包括：需要较多全局基底才能捕捉传输主导或多尺度现象，导致基底截断不易；对训练数据高度依赖，可能忽略输入输出共同结构；参数量与训练成本相对较高；对极大尺度问题仍受低秩张量规模限制，需进一步研究低秩/张量结构化或动态基底适配。

---

## 682. Computing GR(1) Assume-Guarantee Contracts

**arXiv ID:** 2610.37534 | [PDF](https://arxiv.org/pdf/2610.37534v1)

**作者:** Yigit Narter `[一作]` (University of Michigan), Inigo Incer `[通讯]` (University of Michigan)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种针对 GR(1)‑模态理论的假设–保证合同代数实现，能够在合同互连（composition）和商（quotient）操作中通过上下文辅助变量消除（CAVE）得到闭合在 GR(1) 结构内的合同结果，并在 Pacti 平台上实现自动化。

**💡 创新点**

创新点在于：
• 开发了针对 GR(1) 结构的上下文辅助变量消除算法，分别实现了精确的 refinement CAVE 与强壮的 relaxation CAVE；
• 证明了在可接受的子问题类中实现了最弱 refinement 与最强 relaxation；
• 将该框架嵌入 Pacti，首次实现了对时序合同的自动化组合、商、合并与精细化检查；
• 通过 AEB（自动紧急制动）案例验证了系统级需求与兼容性条件的可提取性。

**🔧 技术方法**

核心技术包括：
• 基于 Bounded‑Next GR(1) 归约与展开，转化为有限窗口下的 SMT 量化消除；
• 使用 SMT 求解器（如 Z3）进行符号量化消除；
• 对比自动机方法（Spot）进行精确消除的实验评估；
• 在 Pacti 中实现合同代数与验证器。

**📊 数据集**

使用的数据集：
• AEB（自动紧急制动）运行示例（由 Zhang 等的自动驾驶案例改编）；
• 合成的多组 GR(1) 合同（Boolean 和线性实数算术），用于评估变量消除规模与 liveness 约束数量对性能的影响。

**📈 对比分析**

与 Spot 的自动机基消除相比：
• 对小规模或 liveness 并发量少的实例，Spot 更快；
• 当 liveness 并发量大或需要对多个变量进行消除时，Spot 成本急剧增长，而 CAVE 仍保持在数十秒以内；
• CAVE 在大多数非平凡实例上保持 soundness，且在 35%~79% 的实例上与精确答案一致；
• 结合了有限窗口消除的符号效率与自动机的精确性，展示了互补的规模性能。

**⚠️ 局限性**

限制与不足：
• 仅适用于 Bounded‑Next GR(1) 子句，无法处理无穷递归或无界 next 深度；
• 在子问题之间存在依赖或时间可变 liveness 条件时可能产生保守的结果；
• 由于只在有限窗口上做消除，某些全局一致性约束（如跨步一致性）可能被忽略；
• 目前仅在 SMT 求解层面实现，未结合完整的无限时序推理后端，导致对某些精确性检查的局限。

---

## 683. Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use

**arXiv ID:** 2610.37533 | [PDF](https://arxiv.org/pdf/2610.37533v1)

**作者:** Baptiste Bonin `[一作]` (Université Laval), Audrey Durand `[通讯]` (Université Laval)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在仿真环境中构建一个双手工具使用的上锁任务，利用记录的机制轨迹准确标注每一次失败的原因，并系统评估各感知通道（手腕力/扭矩、接触、关节角度、姿态、指令、图像等）以及视觉‑语言模型（VLM）内部表示与回答对失败诊断和修复的贡献。

**💡 创新点**

①提出了双手工具使用失败诊断的任务和评价框架，首次将失败原因归因到两臂之间的时序关系；②发现哪只手迟到的原因只能通过同时观测两臂信息获得；③揭示VLM内部已编码迟到信息，但回答会因提问表述而变异。

**🔧 技术方法**

使用线性探针（logistic 回归）在每个感知通道上评估诊断性能；采用Qwen3‑VL 8B等大规模VLM在前视摄像头图像上回答“哪只手迟到”的问题；对内部残差流进行投影，比较其信息与模型输出的一致性；仿真器（Franka Panda）与自定义锁机构。

**📊 数据集**

在17个不同门锁上进行758次尝试（包括174成功、198错误、133时序失败、134力学失败、119双因果失败），每次尝试记录完整的关节、力/扭矩、接触、指令与图像序列，故此数据集是实验的核心。

**📈 对比分析**

通过在每个感知通道上训练线性探针，报告每类失败（如tool‑late、push‑late、cam‑out）的AUC；对比单臂通道与双臂通道，发现双臂姿态与指令可在0.9以上准确区分迟到手；VLM内部状态在层4后对迟到信息的AUC达0.96以上，但其回答的AUC受表述影响，平均在0.5左右。修复实验显示，仅询问两手是否同步且递进修复可在两次尝试内修复近80%的失败，等同于完全知道具体原因。

**⚠️ 局限性**

局限性包括：①全部在仿真中完成，缺乏真实机器人噪声与动力学误差；②失败类型与时序延迟仅在脚本化的固定调度下可检验，真实任务的延迟分布更随机；③仅评估单一工具（方形钥匙）和单一锁机构，无法直接推广到更复杂或多工具场景；④VLM回答依赖于提问表述，说明模型的可解释性受限。

---

## 684. GTFormer: A Syntax-Free Graph Network Alternative to Transformers for 8192$\times$ Length Extrapolation in Linear Time

**arXiv ID:** 2610.37524 | [PDF](https://arxiv.org/pdf/2610.37524v1)

**作者:** Elisabeth Fink `[一作]` `[通讯]` (Middlesex University), Elisabeth Fink (Middlesex University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为GTFormer的无语法结构层次图神经网络，替代传统Transformer的全连接自注意力，能够在极长序列上实现线性复杂度的推理。

**💡 创新点**

将平衡二叉树压缩层、水平位移边、随机扩散快捷路、凸高架门控与自顶向下广播相结合，构造了一个受限度数、对数深度的图结构，实现了零样本长度外推至训练长度的8192倍且保持高精度。

**🔧 技术方法**

采用图神经网络消息传递、局部移位编码、多通道凸高架门控、随机扩散短路以及全局广播与多头读出。

**📊 数据集**

使用八个基于程序代码、日志与检索的合成任务，按程序化生成器在每一步生成新的长度为2^i（最多4M）序列，包含多指针、键值对、层级作用等。

**📈 对比分析**

将GTFormer与标准两层Sinusoidal和RoPE Transformer及纯二叉树和无快捷路变体对比；在8M级别序列上GTFormer零样本准确率达到98.1%，而Transformer在4K后OOM，GTFormer占用22GB内存、推理时延≈800ms，显著低于Transformer的O(N^2)成本。

**⚠️ 局限性**

推理延迟相对平面Transformer更高，且目前仅在人工合成任务上验证；缺乏针对真实大规模文本或代码的实验；若扩展隐藏维度或更大模型，需进一步研究稀疏化实现以提升速度。

---

## 685. Leveraging Distributed Sensing and Hyperdimensional Computing for Non-Line-of-Sight Passive Object Sensing

**arXiv ID:** 2610.37519 | [PDF](https://arxiv.org/pdf/2610.37519v1)

**作者:** Placido Mursia `[一作]` (NEC Laboratories Europe), Xavier Costa-Perez `[通讯]` (NEC Laboratories Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于基础设施辅助感知（ISAC）与超维计算（HDC）的框架，用分布式无线设备的信道测量来感知机器人视野之外的被动实体，并进行风险感知分类。

**💡 创新点**

创新点在于：①利用分布式信道测量实现非视线（NLOS）感知；②将信道冲激响应编码为超维向量，避免显式信道重构；③采用迭代重训练的HDC分类器提升分离度；④在智能工厂场景中将感知结果直接嵌入ROS成本地图，提升全局路径规划安全性。

**🔧 技术方法**

核心技术包括：分布式无线感知、信道冲激响应（CIR）采集、超维向量（HDC）编码（连续与二进制）、绑定/捆绑/置换运算、迭代重训练分类、ROS导航栈集成。

**📊 数据集**

数据集来源为：①基于Commercial Ray‑Tracer（Wireless InSite）在模拟仓库环境中生成的10k个CIR样本；②ROS‑based 真实机器人仿真环境，用于验证成本地图与路径规划的改进。

**📈 对比分析**

与传统深度神经网络（约465k参数、8层）基线对比，HDC连续编码在联合接收机场景下平均准确率达88.98%（±3.19%），超过DNN的77.39%；二进制编码在实时推理下仍保持82.21%准确率，且在MCU上运行时单个操作耗时仅几百微秒，显著低于DNN。性能在“危险区”F1上实现0.8494，远高于DNN的0.6642。

**⚠️ 局限性**

局限性包括：①仅完成二分类（安全/危险），未实现多区域或多类风险建模；②假设CIR估计完美，未考虑实际测量误差；③训练离线完成，缺乏在线适应机制；④对硬件部署和网络同步要求较高；⑤主要验证于室内工厂仿真，缺少真实多样化环境测试。

---

## 686. ALADDIN: Adversarial Latent Diffusion Distillation for One-Step Identity Swapping

**arXiv ID:** 2610.37503 | [PDF](https://arxiv.org/pdf/2610.37503v1)

**作者:** Ivan Shchekotov `[一作]` (Constructor University), Dmitry Vetrov `[通讯]` (Constructor University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练并推理一个一阶段人脸交换模型，直接将源身份投射到目标图像，同时保持目标的姿态、表情、发型、光照与背景。

**💡 创新点**

创新点在于：① 用预训练的扩散流匹配模型初始化生成器，并在其冻结的扩散特征上进行对抗训练，省去传统的蒸馏步骤；② 通过Triplet伪标签与重构损失实现身份与目标属性的可控权衡；③ 重新审视评估协议，避免身份编码器偏置。

**🔧 技术方法**

采用扩散流匹配模型、对抗扩散特征判别器、ArcFace（Glint‑R100）身份编码、LPIPS重构、6DRepNet姿态、DECA表情、SegFormer面部掩码以及多任务损失。

**📊 数据集**

训练集：VGGFace2‑HQ 与 Arc2Face（约4.1M图像、540K身份）；评估集：FFHQ 1000对源/目标；预训练：FLUX 1‑Krea‑dev、PuLID、OminiControl。

**📈 对比分析**

采用 FID（CLIP ViT‑B/32）、身份相似度/检索、姿态/表情误差等指标与多步骤（APP​LE、DiffSwap 等）及一阶段基线（SimSwap、HiFiFace、InSwapper 等）对比，ALADDIN 在 FID、身份、姿态、表情上与 APP​LE 相当或更优，同时推理仅 1 次网络评估（比 APP​LE 28 倍快），训练数据量减少约 10 倍。

**⚠️ 局限性**

局限性：在极端年龄差、稀有身份、纹身等场景下身份转移或目标属性保留效果下降，整体仍受人脸交换领域通用瓶颈限制。

---

## 687. When POST Becomes GET: Investigating HTTP Method Interchangeability and Its Security Implications

**arXiv ID:** 2610.37474 | [PDF](https://arxiv.org/pdf/2610.37474v1)

**作者:** Matteo Golinelli `[一作]` (University of Trento), Bruno Crispo `[通讯]` (University of Trento)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过自动化的浏览器爬虫与请求重放管道 MIDE，测量了 10,000 个顶级网站中 HTTP POST 与 GET 方法互换的现象，并评估其安全影响。

**💡 创新点**

创新点包括：首次大规模测量方法互换；基于无查询 GET 基线的高置信度判定；完整的可复现 pipeline 和开源实现。

**🔧 技术方法**

技术手段涵盖：Playwright 浏览器自动化、POST 请求捕获、Body 转换为查询字符串、无查询 GET 基线、结构相似度比较以及手工验证。

**📊 数据集**

使用的数据集为 Tranco 2026 年 5 月生成的 Top 10,000 网站列表，收集了 21,280 条 POST 请求。

**📈 对比分析**

结果显示在 10,000 站点中有 407（4.1%）具备高置信度互换，方法被视为有效且易实现；相较于以往仅限案例研究的方法，MIDE 能在数千个请求级别上给出统计。

**⚠️ 局限性**

局限性包括：仅未身份验证状态；爬行深度、表单交互有限；仅支持 form-urlencoded、multipart/form-data 与 JSON，且会因查询长度限制跳过部分请求；可能漏测受登录或特定业务逻辑保护的接口。

---

## 688. How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI?

**arXiv ID:** 2610.37472 | [PDF](https://arxiv.org/pdf/2610.37472v1)

**作者:** Sheethal Bhat `[一作]` (Friedrich-Alexander-Universit"at Erlangen-N"urnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universit"at Erlangen-N"urnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

评估了基于rs‑fMRI的个体化功能网络归因在ADNI数据中的测试‑重测可靠性。

**💡 创新点**

首次将受试者级置换检验与扫描‑重测结合，揭示归因稳定性主要由群体共享模式驱动。

**🔧 技术方法**

使用冻结的Schaefer‑200功能连接网络预测模型（多层感知机）以及网络消融归因。

**📊 数据集**

利用阿尔茨海默病神经影像倡议（ADNI）中875名受试者的T1和rs‑fMRI数据。

**📈 对比分析**

与分半内检验和扫描‑重测对比，分半一致性近乎满分（ρ≈0.95），扫描‑重测约0.61，置换检验基线ρ≈0.46，匹配组显著更高。

**⚠️ 局限性**

限制包括样本量小的扫描‑重测样本（74人），预测模型性能弱且与不同扫描协议的匹配不一致。

---

## 689. Who Shaped What? Untangling Human and AI Creative Contributions Through Spatial Provenance in Image Editing

**arXiv ID:** 2610.37469 | [PDF](https://arxiv.org/pdf/2610.37469v1)

**作者:** Riya Sahni `[一作]` (Columbia University), Tongyu Zhou `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并评估了一个混合主动式图像编辑探针Roop，记录编辑历史并通过空间原产热图呈现人机交互的“协商强度”，帮助创作者理解并说明其对作品的贡献。

**💡 创新点**

创新点在于：①将编辑交互与图像区域关联的空间原产记录首次用于贡献感知；②通过热图呈现协商强度而非直接归属；③对照创作者自评与记录的差异，揭示贡献认知与记录之间的分离。

**🔧 技术方法**

技术手段包括：GPT‑4o用于生成编辑计划和建议；SAM3进行图像分割；Google Gemini 2.5 Flash Image（Nano Banana）生成具体编辑；Adobe Firefly API实现图像生成与调色；前端使用交互式热图展示记录。

**📊 数据集**

数据集为两张预设的带人物与背景的图像（来自内部图像库）以及一份包含多种情绪目标的列表，供参与者在实验中选择。

**📈 对比分析**

方法为在20名参与者中进行对照实验（用户主动编辑 vs AI建议编辑），先让参与者自评贡献后查看空间原产热图，再访谈其感知。实验未给出量化性能指标，而是通过主题分析比较自评与热图所展示的交互强度，发现二者往往不一致。

**⚠️ 局限性**

局限性包括：实验任务短、使用固定图像与情绪目标，缺乏长时序创作；受试者仅来自单一软件公司，样本多为初级用户；热图仅总结协商强度，未区分人机细节贡献；未同步全局编辑（如Harmonize）与热图；权重等价且未验证对认知的影响。

---

## 690. Penny for Your Prompts: A Measurement Study of Contextual Leakage in AI Chat Platforms

**arXiv ID:** 2610.37467 | [PDF](https://arxiv.org/pdf/2610.37467v1)

**作者:** Paschalis Bekos `[一作]` (Stony Brook University), Michalis Polychronakis `[通讯]` (Stony Brook University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对42个AI聊天平台（25个通用、17个任务特定）进行系统测量，发现21个平台泄露聊天标题、9个平台泄露原始用户提示，且多数平台将用户邮箱与聊天内容一同发送给第三方追踪器。

**💡 创新点**

创新之处在于结合网络抓包与 VisibleV8（V8 引擎层日志）进行事件级拆解，能够精确把泄露流量归因到具体用户操作和 DOM 读取机制，首次揭示传统追踪脚本如何通过页面元数据捕获聊天上下文，并扩展到敏感领域（健康、法律、教育）的聊天平台。

**🔧 技术方法**

使用 Puppeteer 进行自动化交互和网络抓包，VisibleV8 对 JavaScript API 调用进行日志记录，并利用 HTTP 解析工具识别第三方域，进一步通过脚本执行轨迹重构追踪器的读取路径。

**📊 数据集**

未使用公开数据集，而是对42个公开可访问的聊天平台手工交互（注册、发送提示、导航等），记录每个会话产生的网络请求和执行轨迹。

**📈 对比分析**

与先前单提示或 HAR 分析相比，本文通过多交互会话和事件拆解能更精确定位泄露触发点，发现更多泄露实例，表明泄露普遍且规模可观；平均每个提示产生数千个请求，其中约20-30%指向已知追踪器。

**⚠️ 局限性**

局限性包括仅在美国东部单一视角测量，未考虑地区/同意差异；只覆盖 Web 免费版，未分析付费或移动/桌面应用；测量时间有限，可能无法捕捉后续更新；未评估同意机制对追踪器加载的影响。

---

## 691. An Input Skip Can Reparameterize a VAE Without Improving the Generator

**arXiv ID:** 2610.37456 | [PDF](https://arxiv.org/pdf/2610.37456v1)

**作者:** Kevin Lee `[一作]` `[通讯]` (University of California at Los Angeles), Kevin Lee (University of California at Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在高斯变分自编码器（VAE）中加入常数输入跳过（skip）对重构误差与生成分布的影响，并在 NOAA DART 潮汐观测数据和合成高斯实验中进行验证。

**💡 创新点**

创新点在于证明对常数跳过可通过重新缩放解码器和噪声保持生成分布不变，从而仅缩小重构误差；同时通过能量评分等生成指标展示跳过虽能降低重构误差，但往往伴随生成性能下降，强调重构误差与生成质量不必然同步。

**🔧 技术方法**

使用技术包括高斯 VAE、输入跳过、重参数化、重要性加权估计、能量评分评估，以及 PPCA 作为线性基准。

**📊 数据集**

数据集为 NOAA Deep‑ocean Assessment and Reporting of Tsunamis（DART）的年度记录（共7个窗口）以及自定义合成高斯数据。

**📈 对比分析**

比较方法是同时评估重构均方误差（MSE）、负 IW 对数密度和能量评分；结果显示跳过模型可显著降低 MSE，但在能量评分和 IW 对数密度上往往表现更差；PPCA 在所有实验中均获得更低的能量评分，优于所有神经网络模型。

**⚠️ 局限性**

局限性包括：生成质量评估主要依赖能量评分等指标，重构误差与生成性能不直接对应；仅探讨常数跳过，未考虑更灵活的跳过方案；实验范围受限于 DART 数据和合成设置，可能不具备广泛推广性；估计观测方差时可能导致生成性能下降。

---

## 692. Electric-Field Induction Motor with Discrete Circuit Elements and Improved Rotor Dynamics

**arXiv ID:** 2610.37452 | [PDF](https://arxiv.org/pdf/2610.37452v1)

**作者:** Alex S. Miller `[一作]` (Massachusetts Institute of Technology), Jeffrey H. Lang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种宏观尺度的电场感应电机，采用离散电阻器构建转子并在转子上布置电路，以实现可调节的转子时间常数，进而实现高起始转矩和高效率。

**💡 创新点**

创新点在于：① 用可采购的离散电阻器替代连续材料，显著提升制造可重复性和调节灵活性；② 在转子上设计类似“深条”电路，使转子时间常数随转子滑频变化，实现起始转矩与工作效率的双高平衡；③ 对电场感应“深条”概念进行建模、仿真和实验验证，首次展示了该技术在宏观电机中的应用。

**🔧 技术方法**

主要技术包括：电场感应电机原理、连续介质模型与离散电路等效、PCB制备与离散电阻/电容布局、空气静态轴承+V‑jewel组合轴承、双相高压线性放大器、光学速度传感与失速试验、实验数据与理论模型对比分析。

**📊 数据集**

数据来源为实验测量：多种转子设计（含R0、R1、C1、C_vane参数）在10–3000 Hz激励频率、不同气隙条件下的转速与转矩数据；无公开数据集，全部实验数据由作者自行收集。

**📈 对比分析**

通过对比实验转矩与理论模型预测，利用失速试验测定负载转矩，并计算机械输出转矩。实验结果表明：离散电阻设计可与连续材料模型匹配；“深条”电路能将起始转矩提升约一倍，峰值转矩仅略降，整体工作效率显著高于单电阻转子设计。

**⚠️ 局限性**

局限性包括：驱动采用固定频率，高压可变频驱动技术尚未成熟；空气静态轴承与V‑jewel限制了气隙调节范围；离散元件数量大导致电机尺寸和质量增加；未对长时间运行、极端温度和耐久性进行系统验证。

---

## 693. Interpretable Meta-Policies for Neural Decision Making

**arXiv ID:** 2610.37451 | [PDF](https://arxiv.org/pdf/2610.37451v1)

**作者:** Raban Emunds `[一作]` (Technical University Darmstadt), Kristian Kersting `[通讯]` (Technical University Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出NEXUS框架，将低层神经技能与高层符号化元策略相结合，利用LLM自动生成技能奖励与符号规则，实现可解释且高效的层次化强化学习。

**💡 创新点**

通过把可解释的符号决策迁移到技能层降低符号复杂度；利用LLM自动化生成奖励函数与元策略；引入神经‑符号混合元策略兼顾可解释性与灵活性。

**🔧 技术方法**

层次化Q网络（Hierarchical PQN）、并行离线Q学习、对象中心化感知、LLM生成规则与奖励、符号规则引擎、神经‑符号混合元策略。

**📊 数据集**

Atari Seaquest、Kangaroo（JAXAtari）和Crafter（JAX实现）等视觉复杂环境，使用OCatari等得到的对象中心化表示。

**📈 对比分析**

与非可解释的PQN、HPQN以及PPO基线对比，采用游戏得分和手册目标对齐度两项指标；结果显示NEXUS在游戏奖励上相当或更优，在目标对齐度上显著优于基线；在简化版游戏中表现更稳健。

**⚠️ 局限性**

依赖高质量对象检测与符号化表征；LLM生成的奖励/规则需人工审校；目前仅适用于离散动作空间，连续动作及更复杂动态场景的扩展尚待研究。

---

## 694. Multi-Agent Reinforcement Learning of Co-Adaptive Hip Exoskeleton Control (MARL-Exo) for Metabolic Energy Reduction

**arXiv ID:** 2610.37441 | [PDF](https://arxiv.org/pdf/2610.37441v1)

**作者:** Yifei Yuan `[一作]` (New Jersey Institute of Technology), Xianlian Zhou `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现一种多智能体强化学习框架 MARL‑Exo，利用全身肌肉模型与臀部外骨骼在仿真中共适应，得到一种无需个体调优即可直接用于真实受试者的助行控制器，并在八名健康志愿者上验证其对代谢能耗的降低效果。

**💡 创新点**

创新点在于：①将基于肌肉能量学的代谢功率直接作为外骨骼奖励函数；②采用两阶段共适应训练，并在训练中对人体解剖学、步态和速度进行随机化，使控制器具备群体泛化能力；③通过仿真中对不同奖励形式的筛选，实现单一群体级控制器的快速迭代和转移。

**🔧 技术方法**

技术手段包括：多智能体PPO强化学习；304 Hill 型肌肉的完整人体模型；DART 物理仿真引擎；Uchida 肌肉代谢模型；IMU 低通滤波的姿态观测；Raspberry Pi 5 上的实时部署。

**📊 数据集**

数据来源：仿真阶段使用内部生成的随机化人体参数（解剖学比例、肌肉最大等效力、步态速度等）；实验阶段使用八名受试者的间接呼吸测量、IMU 记录以及设备本身的传感数据，未使用公开数据集。

**📈 对比分析**

比较方法：在仿真中比较两种奖励结构（Metabolic vs Metabolic+Power），在实验中与无设备 (No‑Exo)、被动设备 (Passive) 以及另一已验证控制器 (SMAT) 进行对比；性能表现为：仿真中 EnergyExo 代谢功率下降 13.8%，PowerExo 下降 8.8%；实验中 MARL‑Exo 与 Passive 相比降低 18.3 % 代谢率，平均比无设备低 7.4 %（6/8 受试者实现低于无设备成本）。

**⚠️ 局限性**

局限性：样本量仅 8 人、单一步速、平地实验；设备重量较重且刚性限制导致可能出现软组织位移；仿真与真实的软组织耦合、IMU 误差未完全匹配；未对不同运动强度、斜坡或受损步态进行评估；缺乏长期适应性和个体化调优的研究。

---

## 695. Request-Adaptive Execution for Efficient Hybrid LLM Inference

**arXiv ID:** 2610.37432 | [PDF](https://arxiv.org/pdf/2610.37432v1)

**作者:** Fahao Chen `[一作]` (Shandong University), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Flex，一套结合请求自适应混合 Attention‑SSM 结构与高效运行时的 LLM 推理系统。

**💡 创新点**

创新点包括：①预填门控实现一次性决定每层使用 Attention 或 SSM，保持解码一致；②selective batching 与 SM 资源调度实现不同分支并行；③双层 Radix‑Tree 缓存支持在动态分支下精准前缀重用。

**🔧 技术方法**

技术手段：预填门控、分支切换、分支级缓存、hybrid‑aware context caching、selective batching、load‑aware SM 分配、NVIDIA GreenContext、双层 radix tree、成本正则化训练。

**📊 数据集**

实验数据集：ARC‑Easy、ARC‑Challenge、PIQA、TruthfulQA、CNN/DailyMail、PubMed。

**📈 对比分析**

与 Llama、Mamba、Zamba、Jamba、Nemotron‑H（静态混合基线）及 vLLM、SGLang、Jenga（系统基线）对比；在质量‑效率上 Flex 在 13/18 评价指标领先，吞吐提升 3.4–3.9×，TTFT/TPOT 显著降低。

**⚠️ 局限性**

局限性：需手工对齐 Attention‑SSM 层并训练门控；对极大规模模型仍需量化或更细粒度缓存；动态分支带来额外推理与缓存管理开销。

---

## 696. The Minimum Subpacketization of Reed-Solomon Codes with Linear Repair Achieving the Cut-Set Bound

**arXiv ID:** 2610.37522 | [PDF](https://arxiv.org/pdf/2610.37522v1)

**作者:** Weijun Fang `[一作]` (Shandong University), Fang-Wei Fu `[通讯]` (Nankai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究 Reed-Solomon 码在满足 cut-set 最优线性修复时的最小子包大小。

**💡 创新点**

通过交叉比字段的除数与相对度数约束得到严格下界，并给出与上界相匹配的构造，解决了 Ramkumar 等人提出的开放问题 20。

**🔧 技术方法**

主要使用有限域交叉比、数论（最小公倍数、质数分解）和基变换技术来构造和分析修复方案。

**📊 数据集**

无需外部数据集，所有构造和证明均基于符号域与评价点的代数结构。

**📈 对比分析**

与已有的下界进行比较，证明在满足 d(d+2)≥4(n+1) 的参数范围内，最小子包大小等于 sΔ_s(n,d)=sG_s(n)，从而获得最优性能。

**⚠️ 局限性**

在 d(d+2)<4(n+1) 的情形下，仍存在 Δ_s(n,d)<G_s(n) 的例子，整体最优性尚未完全确定，需要进一步研究。

---

## 697. SecDMP: Provably Secure Data Memory-Dependent Prefetching (Extended Version)

**arXiv ID:** 2610.37454 | [PDF](https://arxiv.org/pdf/2610.37454v1)

**作者:** Jens Sprengers `[一作]` (KU Leuven), Frank Piessens `[通讯]` (KU Leuven)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SecDMP，一种硬件/软件协同的防御机制，能在不降低 DMP 性能的前提下完全消除 DMP 对机密数据的侧信道泄露。

**💡 创新点**

创新点包括：①对 DMP 泄漏的系统分类与分析，揭示现有防御（永久/选择禁用）不足；②设计 SecDMP 并给出形式化安全证明；③实现了首个开源 DMP（Apple、Intel、CDP、IMP）并在 RISC‑V Proteus 上评估；④将 SecDMP 与 Spectre 防御结合，恢复常数时间代码的完整微架构安全。

**🔧 技术方法**

使用的技术：RISC‑V Proteus 仿真平台；硬件侧添加 taint 位与 DMP 针对机密数据的屏蔽；软件侧通过 CSRs 静态划分公共/秘密内存；形式化语义扩展 + 安全证明；对比实验采用非干扰测试、加密与通用工作负载混合的 DMP‑Bench。

**📊 数据集**

数据集与实验程序：自制 DMP‑Bench（包含 HACL* 加密原语、Embench、CoreMark 以及自定义工作负载 AoP/SpMV），不同工作比例（25/50/75/90%）；以及 9 种攻击/泄漏测试程序（IMP Leak、Augury PoC、GoFetch PoC、Peek‑a‑Walk、DMP‑Spectre、Recursion、Load Secret、Same Cache Block、Transient Load）。

**📈 对比分析**

比较方法：通过非干扰测试验证泄漏；测量面积、临界路径、平均/最大性能开销；结果显示：SecDMP 仅增加 0.5% 面积、0.36% 平均性能开销（最高 3.3%），彻底消除所有泄漏；永久禁用性能损失可达 59%（尤其在高工作负载时）；选择禁用平均开销 0.26%，但仍无法阻止 Ind 类泄漏。

**⚠️ 局限性**

局限性：需软件标记机密数据，静态分区难以处理堆/栈动态分配；对 Apple DMP Pref 类泄漏机制未知、Intel DMP Trans 类泄漏是否存在仍不确定；实现依赖于 Proteus 平台，尚未在真实商用 CPU 上验证；未来工作需集成到编译器并完善对复杂程序的支持。

---

## 698. Approaching Optimal Scaling of Queue Size for an Input-Queued Switch

**arXiv ID:** 2610.37521 | [PDF](https://arxiv.org/pdf/2610.37521v1)

**作者:** Taha Ameen `[一作]` (Carnegie Mellon University), Weina Wang `[通讯]` (Carnegie Mellon University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种k级分批调度策略，在n×n输入排队开关中实现期望总队列长度接近最优O(n/(1-ρ))的标度。

**💡 创新点**

创新点在于递归多层分批设计、随机分配初始层级以及利用s-因子与完美匹配的组合，以克服组合匹配约束导致的排队增长。

**🔧 技术方法**

采用组合图论（s-因子、Kőnig色彩定理）与概率论（Chernoff界、Kingman排队上界）等技术进行严格的队列长度上界证明。

**📊 数据集**

该工作为理论分析，未使用具体数据集；假设到达过程为独立均匀流量。

**📈 对比分析**

与之前的单层或固定分批方案相比，k级策略在大系统负载下将指数2^k/(2^k-1)逼近1，导致期望队列长度比O(n/(1-ρ)^2)或O(n/(1-ρ)^{4/3})等更优，且调度计算复杂度为O(n^3k)；实验验证未给出。

**⚠️ 局限性**

局限性包括仍需较高的k值才能逼近最优标度、计算开销较大、仅在均匀独立流量下证明，且未对非均匀或相关流量的性能进行评估。

---

## 699. When Visual Evidence Is Insufficient: How Instructions, Post-Training, and Internal Activations Shape Refusal in Multimodal Language Models

**arXiv ID:** 2610.37430 | [PDF](https://arxiv.org/pdf/2610.37430v1)

**作者:** Ananth K Suresh `[一作]` (Indian Institute of Technology Hyderabad), Vineeth N Balasubramanian `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模态大型语言模型在视觉无答案问题上的证据基础拒绝行为进行系统分析，探讨指令、后训练和内部激活三种控制方式的影响。

**💡 创新点**

提出对拒绝原因的细粒度分类和同样样本的转换评估，验证指令语义与位置、后训练偏好以及行为导向激活在拒绝行为中的互相作用。

**🔧 技术方法**

使用指令层级策略控制、监督与偏好对齐的后训练（SFT、DPO、ORPO、SimPO）、基于行为的激活干预、Qwen3.5 评估器、MoHoBench benchmark 进行实验。

**📊 数据集**

MoHoBench（视觉无答案问题集）和 MMU 作为性能基准。

**📈 对比分析**

通过比较拒绝率、证据基础拒绝比例、混合响应率和 MMU 准确率，发现指令语义和位置对拒绝敏感；后训练能显著提升拒绝但伴随混合响应；行为导向激活可在不同指令下转移但影响有限，且未导致 MMU 下降。

**⚠️ 局限性**

更高的拒绝率并不保证干净的证据基础拒绝；混合响应频率高；模型特异性强；激活方向的通用性不足；实验仅覆盖三种模型和 MoHoBench，难以推广到更广泛的 VQA 场景。

---

## 700. The Structure Gap: When More Data Cannot Replace Structured Reasoning

**arXiv ID:** 2610.37523 | [PDF](https://arxiv.org/pdf/2610.37523v1)

**作者:** Kheer Sagar Patel `[一作]` (University of Illinois Chicago), Mojtaba Soltanalian `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过定义结构编码问题和结构缺口，系统研究了在已知结构被剥夺时，有限资源下学习者与结构知情求解器之间的差距；

**💡 创新点**

创新点在于将结构缺口形式化为样本、参数、计算等资源上的分离，并通过理论定理（高斯逆问题、覆盖下限、轨道不可识别）与实验验证，揭示有限数据无法完全代替结构的根本原因；

**🔧 技术方法**

采用统计学习理论、均值-协方差分析、Wishart分布推导、Yao极小极大原理以及深度学习实验（FCN、GNN、BFS）等技术；

**📊 数据集**

使用人工生成的数据集，包括高斯逆问题（d=m=32）、稀疏恢复（k=4、m=28）、以及Erdős–Rényi图的到达性（8~24节点）等；

**📈 对比分析**

将结构无知的全连接网络与结构知情的解算器、消息传递图神经网络以及BFS做对比；实验显示FCN在有限样本下风险显著高于结构求解器，且在图规模外推性能下降；GNN能显著缩小缺口，BFS实现完美；

**⚠️ 局限性**

限制在于仅针对可压缩、正确且已知的结构，结构错误或不完整时不适用；实验只用FCN作为无结构基线，未覆盖更强的无结构模型；理论多在理想化假设下，实际应用需进一步验证。

---

## 701. World Dreamer v5-lite: A Multi-Robot World Action Model with One Dreamer

**arXiv ID:** 2610.37518 | [PDF](https://arxiv.org/pdf/2610.37518v1)

**作者:** Guoning Wu `[一作]` (World Agents), Yongbin Zheng `[通讯]` (World Agents)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多机器人协作控制框架，利用共享工作空间状态和视觉语言模型实现基于语言指令的多机器人动作预测

**💡 创新点**

通过将多摄像头的校准特征投影到三维平面聚合成共享空间状态Φ；采用可位姿感知的代理查询实现机器人间信息交换；引入掩蔽视图协同训练，鼓励机器人利用同伴的视角来补充视觉信息

**🔧 技术方法**

预训练视频编码器、Qwen3‑VL视觉语言模型、变分自动编码器、流匹配解码器、姿态感知注意力机制以及掩蔽视图训练策略

**📊 数据集**

预训练集包含20种机器人模拟轨迹、RoboFactory轨迹以及LeKiwi与OpenArm真实机器人轨迹；评估使用RoboFactory、RoboCasa365和四个真实机器人任务

**📈 对比分析**

与Diffusion Policy、ACT、π_0.5、Fast‑WAM、GauDP等基线对比；在RoboFactory平均成功率74.0%（比最佳基线+23.7pp），在四个真实任务平均78.8%（比最佳基线+13.8pp），在RoboCasa365整体58.08%（比PatchWAM+10.48pp）

**⚠️ 局限性**

依赖大量标注轨迹与摄像头标定；对动态遮挡和新视角的泛化尚未充分验证；模型训练与推理复杂度较高

---

## 702. Ego-CEVR: Learning Cross-Embodiment Visual Representations from Human Egocentric Videos

**arXiv ID:** 2610.37509 | [PDF](https://arxiv.org/pdf/2610.37509v1)

**作者:** Keru Zhou `[一作]` (Maniformer), Maoqing Yao `[通讯]` (Maniformer)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Ego-CEVR 框架，利用头部视频的未来预测与手腕视频的动作监督，层次化学习全局任务上下文与局部交互表示，从而提升机器人操控的成功率与鲁棒性。

**💡 创新点**

创新点包括：① 以数据本地化（data‑native）为核心的模型设计，专门适配多源多视角的数据；② 层次化表示学习，将全局视觉语义与局部交互特征分离并通过跨注意力融合；③ 结合视频预测与动作监督的双阶段预训练；④ 针对跨体态学习的少量示范适配流程。

**🔧 技术方法**

使用技术主要包括：Diffusion Transformer（Wan2.2）+VAE 进行视频预测；Flow‑matching 框架实现动作与视频的联合学习；Cross‑Attention 机制将全局与局部表示注入动作专家；以及预训练与微调的两阶段训练策略。

**📊 数据集**

实验数据集涵盖：LIBERO、LIBERO‑Plus、RoboTwin 2.0（模拟），以及 AgiBot G1/G2 的真实机器人数据；同时使用人类视角（egocentric）视频和机器人头摄像机视频进行跨体态迁移。

**📈 对比分析**

在 LIBERO、LIBERO‑Plus 与 RoboTwin 2.0 上与 VLA‑JEPA、Fast‑WAM、π₀.₅ 等基线对比，Ego‑CEVR 分别取得 97.7%、90.2% 与 70.7%（Hard 设定）最高或接近最高的成功率，并在视觉扰动、摄像机视角变化等鲁棒性测试中表现更优。

**⚠️ 局限性**

限制主要在于资源受限，未对大规模人类视频进行预训练，也未评估模型在更大规模数据与更大模型尺寸下的扩展性与性能。

---

## 703. Global Choice without Excluded Middle in Dependent Type Theory

**arXiv ID:** 2610.37498 | [PDF](https://arxiv.org/pdf/2610.37498v1)

**作者:** Sewon Park `[一作]` `[通讯]` (Institute of Mathematics, Physics and Mechanics), Sewon Park (Institute of Mathematics, Physics and Mechanics)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了一个基于分阶段集合的模型，展示在没有命题等价性的前提下，全球选择并不能推出排中律。

**💡 创新点**

首次给出可验证 Lean 语义的分阶段集合模型，既验证了证明不可相关性、递归族、商集等特性，又证明了全球选择不蕴含排中律。

**🔧 技术方法**

利用依赖型理论的语义学、分阶段集合（well‑ordered stages）、类别与家庭（CwF）等技术构建模型。

**📊 数据集**

无

**📈 对比分析**

通过理论模型验证与 Lean 语义对应，没有实验比较，证明表明该模型满足所需公理但排中律失效。

**⚠️ 局限性**

模型仅在特定良序集上构造，对更广泛类型的可扩展性未知；且未验证全局选择下的其他可能性（如子类正则性等）。

---

## 704. Revisiting Privacy Risks in Quantized Deep Neural Networks

**arXiv ID:** 2610.37488 | [PDF](https://arxiv.org/pdf/2610.37488v1)

**作者:** Jie Fu `[一作]` (Stevens Institute of Technology), Wendy Hui Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对量化深度神经网络（QNN）的成员推理攻击（MIA）隐私风险进行系统实验评估，并测试现有防御的效果。

**💡 创新点**

首次全面评估8种攻击、6种量化方法，揭示低位量化对隐私影响的非线性关系，并指出现有防御在QNN中的不足。

**🔧 技术方法**

量化技术（PTQ/QAT、固定/混合精度）、黑盒与白盒MIA、Dropout、对抗正则、差分隐私等。

**📊 数据集**

CIFAR-100、CINIC-10、Location、Texas 四个公开数据集，使用 ResNet-18、MobileNetV2、MLP。

**📈 对比分析**

通过攻击准确率、TPR@1% FPR 与模型准确率对比，发现 4–8 位量化几乎不降低攻击效果，只有 2 位量化才显著提升隐私但极大降低准确率。

**⚠️ 局限性**

仅考虑了四个数据集和两种模型，且对量化后防御的调优不充分，未探索更适合 QNN 的专门防御方案。

---

## 705. Keeping Inference in the Radio: Design and Measurement of a Spiking Accelerator for mmWave ISAC Activity Recognition

**arXiv ID:** 2610.37480 | [PDF](https://arxiv.org/pdf/2610.37480v1)

**作者:** Anthony Sánchez `[一作]` (IMDEA Networks Institute), Joerg Widmer `[通讯]` (IMDEA Networks Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在ZCU208 RFSoC上实现并测量了一套微多普勒活动识别的全流式SNN加速器，完成始终在线的感知任务；

**💡 创新点**

首次在通信射频硬件上完整实现并实测SNN推理，结合Spiker+量化与自适应训练，显著压缩逻辑面积与能耗，并通过3bit权重量化实现了硬件友好型模型；

**🔧 技术方法**

使用全连接LIF型SNN、Spiker+固定点量化、Spike Encoder、AXI Wrapper、FPGA（ZCU208）RFSoC及PYNQ软件平台；

**📊 数据集**

采用DISC-A公共数据集（IEEE 802.11ay 60GHz，含微多普勒光谱），对七名受试者的走路、跑步、坐起、举手等四类活动进行训练与测试；

**📈 对比分析**

在同一硬件平台上与ANN（1-bit与16-bit输入）对比，测量准确率、宏F1、延迟、逻辑利用率、功耗与能量。SNN 132神经元实现94.7%准确率、90.7宏F1，推理仅206个时钟周期；逻辑仅为ANN的1/8，动态能耗为ANN的1/3.4倍（对1-bit输入），对16-bit输入则低7.6倍；3bit权重量化后准确率仅下降0.5点。

**⚠️ 局限性**

局限在于仅验证于单一四类活动数据集，SNN的准确率仍落后于CNN/传统网络；未量化前端射频、采样与STFT功耗；实现依赖FPGA，ASIC化后才可充分发挥低功耗优势。

---

## 706. What Carries Forward? Pretraining Disclosure Across Language Model Releases

**arXiv ID:** 2610.37465 | [PDF](https://arxiv.org/pdf/2610.37465v1)

**作者:** Md Sakib Ul Islam `[一作]` (Victorian Institute of Technology), S M Mehedi Zaman `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对41个模型条目（17个家族）进行文档审计，使用2025年《基础模型透明度指数》（FMTI）24个预训练指标，比较不同版本之间的披露得分和指标转移。

**💡 创新点**

首次系统追踪同一家族模型在不同版本中的披露增减与保持，揭示总分增长并不保证信息连续性，并提出对比增、减、保持三类指标转移的评估框架。

**🔧 技术方法**

采用文档基审计流程：先由ChatGPT提取证据并进行初步编码，再由两位人工审核员逐一核对并统一判定；使用二进制评分并对相邻版本做指标转移计数。

**📊 数据集**

以公开技术报告、模型卡、仓库及相关官方文档为证据来源，共构成41条模型条目；不使用任何训练数据集，而是通过文档内容进行分析。

**📈 对比分析**

通过比较相邻版本的总分及各指标的0→1、1→0、1→1转移，发现大多数家族得分下降、部分得分提升但伴随信息流失；表明单一总分不可靠，需关注具体指标的增减情况。

**⚠️ 局限性**

限制包括：样本不是完整代表，家族内部比较不等价；分组条目可能掩盖不同尺寸模型的差异；只评估公开文档，未追溯原始发布日期的文档状态；二进制评分忽略信息细节；开放度分类为操作性标签，未得到正式验证；人工核对依赖先前记录，可能影响一致性。

---

## 707. Concordat: Typed Sessions for LLM Agents

**arXiv ID:** 2610.37461 | [PDF](https://arxiv.org/pdf/2610.37461v1)

**作者:** Milad Sikaroudi `[一作]` `[通讯]` (Independent Researcher), Milad Sikaroudi (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Concordat，一个 Python 库，用来把 LLM 代理的工具调用视作有限状态协议，并在运行时强制协议，防止模型违背预设规则；

**💡 创新点**

将会话类型和类型状态的概念映射到 LLM 代理，自动推导协议、在定义时检查继承安全，并通过工具过滤实现安全可追踪的代理行为；

**🔧 技术方法**

有限状态机、类型状态（typestate）与会话类型、Python 装饰器、运行时工具过滤、预算线性划分、LLM API 调用；

**📊 数据集**

τ‑bench 航班服务基准（50 个任务）与模拟客户对话，使用 GPT‑4.1 与 Claude Haiku 5.5 作为 LLM；

**📈 对比分析**

对标准代理与 Concordat 代理在 50 个任务的压力测试中比较未确认变更次数、任务成功率和费用；Concordat 在保持类似成功率的同时将违规次数降至零，成本略低；

**⚠️ 局限性**

实验仅限单一领域和少数模型；压力场景人为设定，未覆盖真实用户行为；预算与类型未实现静态检查；对多代理多方协议支持有限。

---

## 708. PolyGandr: Steering Generative Models via Spatial Navigation through Latent Space

**arXiv ID:** 2610.37446 | [PDF](https://arxiv.org/pdf/2610.37446v1)

**作者:** Mingwei S. G. Li `[一作]`, Remco Chang `[通讯]` (Tufts University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个名为 VSGI 的视觉交互界面，整合全局探索地图与局部混合视图，通过示例对比构建向量方向、基点与步长三元操作实现生成模型的高维导航与细化。

**💡 创新点**

创新点在于：①将示例对比转换为可视化的“向量算术导航”框架；②将全局探索与局部细化通过统一的空间交互（镜头、网格、自由混合）连接；③利用 Parametric UMAP 维度约简提供可编辑的 2D 地图，同时保持对高维潜空间的控制；④在多种模型（GAN、Diffusion、工程设计空间）上统一实现。

**🔧 技术方法**

核心技术包括：潜空间向量算术（z_pos - z_neg），加权线性插值（inverse‑distance、smoothstep、negative‑log），Parametric UMAP 维度约简，分批与进度渲染，交互式 Lens/网格/混合视图，StyleGAN2 与 Stable Diffusion XL 的推理接口，旋转型 Kirigami 解析器与可视化后端。

**📊 数据集**

使用的数据集：①StyleGAN2 的人脸与猫图像（20k 采样点）；②Stable Diffusion XL 4,000 个椅子图像，基于六种样式的模板提示；③旋转型 Kirigami 的 20k 随机凸四边形（8‑D 顶点坐标），用于构造可行性求解器；此外还使用各模型的训练数据集。

**📈 对比分析**

方法通过三种案例展示：①StyleGAN2 生成/细化，验证插值平滑、方向一致性和局部修正能消除性别干扰；②Stable Diffusion XL 用于椅子风格插值与方向操作，显示可视化轨迹；③Kirigami 专家访谈评估界面提升探索效率与可行性反馈。性能上，StyleGAN2 实时交互；Diffusion XL 需要数秒生成轨迹；Kirigami 通过几何求解实现即时响应；评估主要为定性案例与用户研究，没有量化指标。

**⚠️ 局限性**

局限性包括：①向量方向在局部有效，跨区域会失效；②Diffusion 模型高推理成本导致交互延迟；③缺乏主动约束引导，用户在不可行区域需要试错；④插值方案固定，无法适应高维复杂空间；⑤潜空间非欧氏曲率导致直线插值偏差；⑥当前实现仅支持向量算术，未完全验证对其他控制方式（LoRA、参数空间等）的扩展。

---

## 709. Extending the W3C Data Privacy Vocabulary to Cover the German Health Data Use Act - Lessons Learned

**arXiv ID:** 2610.37527 | [PDF](https://arxiv.org/pdf/2610.37527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 710. Segmentation Confidence for Arbitrary CNNs

**arXiv ID:** 2610.37431 | [PDF](https://arxiv.org/pdf/2610.37431v1)

**作者:** Baris Oguz `[一作]` (University of Pennsylvania), Ipek Oguz `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种在现有卷积神经网络分割管线中可直接使用的单一置信度评分方法，该方法通过多次推理产生的不确定性图计算整体置信度。

**💡 创新点**

创新点在于：①使用多次推理得到的概率图计算体素级不确定性；②将不确定性映射为一系列体积阈值，并通过体积随阈值变化的比例求得单一置信度分数 C；③方法不需改造网络或重新训练，可直接集成到任何现有分割模型。

**🔧 技术方法**

技术手段包括：测试时翻转与噪声扰动产生多份分割结果；体素级标准差和熵不确定性估计；基于阈值的体积变化比计算置信度；使用椭圆包围盒进行无监督异常检测；在主动学习场景下评估置信度选择策略。

**📊 数据集**

使用的数据集为 3D 超声胎盘数据（210+49 图像，来自 121/41 名受试者）以及 Medical Segmentation Decathlon 的海马体 MRI（260 训练样本）。

**📈 对比分析**

与 VVC（体积变异系数）方法对比。对 2D flip-augmented 模型，置信度 C 与 Dice 的相关性显著高于 VVC，且 C 的异常检测准确率更佳；在 3D 加噪声模型中表现相同。主动学习实验表明，使用 worst confidence 策略在样本增量过程中最快提升测试 Dice，接近 oracle 策略。

**⚠️ 局限性**

局限性包括：需要额外的推理次数来生成多份预测，导致推理成本上升；置信度的准确性受测试时增广方式的影响；尚未在真实临床数据或更广泛的模态下进行验证，泛化性待进一步评估。

---

## 711. Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training

**arXiv ID:** 2610.37429 | [PDF](https://arxiv.org/pdf/2610.37429v1)

**作者:** Faeze Moradi Kalarde `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM后训练中提出AdaRoll-GRPO，动态分配rollout预算以提升GRPO训练效果。

**💡 创新点**

将rollout分配问题建模为Bayes‑adaptive MDP，离线求解最优策略，实现难度感知的预算分配。

**🔧 技术方法**

采用BAMDP与离线价值迭代相结合的技术，并与GRPO优势估计协同工作。

**📊 数据集**

在数学推理（GSM8K、MATH）、逻辑推理（LogiQA）和常识推理（CoLoTa）等四个基准数据集上进行评估。

**📈 对比分析**

与vanilla GRPO、Dr.GRPO、DAPO等基线对比，AdaRoll-GRPO在相同rollout预算下平均提升10%+准确率，并且在难度异质性较高的数据集上提升更显著。

**⚠️ 局限性**

局限在于需离线求解大规模BAMDP，且对极大batch或预算变化的实时适应有限；对模型动态演化的适应性仍待提升。

---

## 712. Predict the Swing, Match the Speed: Robotic Pipe Stabbing With Learned Stereo Vision and Residual Reinforcement Learning

**arXiv ID:** 2610.37535 | [PDF](https://arxiv.org/pdf/2610.37535v1)

**作者:** Mohamed Ashraf Mostafa `[一作]` (Azhar University), Samir Khaled `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 StabBot 机器人，利用视觉预测悬挂管子摆动并以匹配速度捕捉、触感确认并插入，完成 200 个不同尺寸和摆幅的随机任务。

**💡 创新点**

首次在线估计悬挂管子自由摆动并实现速度匹配捕捉，结合 3D 关键点视觉定位、残差强化学习优化以及统一的 V‑形导头适配九种尺寸，显著提升捕捉可靠性与柔性。

**🔧 技术方法**

采用卡尔曼滤波对摆动状态进行估计，ResNet‑18 3D 关键点网络从立体图像定位管子、针尖和盒口，PPO 残差强化学习对控制进行细节调节，辅以 MuJoCo 模拟、六轴机械臂、立体摄像头与力/力矩传感器。

**📊 数据集**

在合成的 20,000 对训练数据、2,000 对验证和 2,000 对测试数据上训练关键点网络，并在 200 作业的仿真基准中评估性能。

**📈 对比分析**

与传统固定导头、反应式及守卫式方法在 200 组相同随机作业下对比，StabBot 成功率 100% 对比 74%，接触速度、峰值力分别降低 7.8 倍和 7.8 倍，且在噪声、延迟、帧丢失、校准误差等恶劣条件下仍保持 100% 成功率。

**⚠️ 局限性**

目前仅在仿真环境验证，未在真实井下测试；关键点网络对极端光照、颜色或遮挡条件仍存在误差；对极大摆幅（35–40 cm）和大噪声（>30 mm）时性能下降；需要进一步的现场实验与更大规模真实数据验证。

---

## 713. Orthogonal and Parametrized Multi-Attribute Steering of LLMs

**arXiv ID:** 2610.37517 | [PDF](https://arxiv.org/pdf/2610.37517v1)

**作者:** Remy Ogasawara `[一作]` (University of Utah), Jeff M. Phillips `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的激活空间 steering，本文提出了可裁剪的可调区间、参数化 steering 与正交化多属性 steering 方法，显著提升了多属性控制的可解释性与鲁棒性。

**💡 创新点**

创新点在于：①为每个 steering 向量识别最优可调区间并进行裁剪；②用参数化 steering 直接设定激活在目标强度点；③通过低秩正交变换实现多属性间的严格正交化，消除属性间的干扰。

**🔧 技术方法**

技术手段包括线性表示假设、对比激活添加、α‑iterative 与参数化 steering、低秩正交变换、梯度、激活投影、LLM 评判等。

**📊 数据集**

主要使用 Llama 2 7B Chat（基准）与 Qwen 3 8B（验证），以及 Anthropic 的“Sycophancy on NLP”与“Advanced AI Risk”、TruthfulQA、Toxigen、BBQ 数据集和自制的 Warmth 数据集。

**📈 对比分析**

与无 steering、传统 α‑iterative 以及 MAT‑Steer 在 TruthfulQA/Toxigen/BBQ 上进行对比；参数化与正交参数化在这些基准上提升约 10‑15 % 分数；在多属性实验中，Sensitivity 降至 < 1、Steerability 范围扩大至 ~3×，正交参数化实现了最小干扰。

**⚠️ 局限性**

局限性包括：仍未能完全消除高度相关属性的相互干扰；方法局限于单层线性空间，跨层或非线性 steering 仍需研究；数据集偏差可能导致意外属性耦合；需要更多大规模、多模型验证。

---

## 714. A note on data-efficient finetuning of vision foundation models

**arXiv ID:** 2610.37510 | [PDF](https://arxiv.org/pdf/2610.37510v1)

**作者:** Niclas Hergenröther `[一作]` (Max Planck Institute for Intelligent Systems), Antonio Orvieto `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了视觉提示（VP）、线性探测（LP）、完整微调（FFT）和低秩适配（LoRA）在不同基础模型（注意力、卷积、Mamba）上的迁移效果；

**💡 创新点**

提出使用AutoVP和全连接标签映射提升VP性能，并对提示大小、标签映射以及正则化策略进行了细致对比；

**🔧 技术方法**

采用AutoVP、FCLM、LoRA、LP、FFT以及各种正则化方法（Dropout、L1/L2、SAM、Autoaugment）；

**📊 数据集**

在CIFAR-10/100、Flowers102、GTSRB、DTD、EuroSAT、Food101、OxfordPets等公开图像分类数据集上进行实验；

**📈 对比分析**

通过对比不同迁移范式在相同数据集和超参数下的Top‑1准确率，发现ConvNext在VP下表现最好，LoRA在数据稀缺的DTD任务上最具优势，VP在某些任务上易过拟合但正则化可缓解；

**⚠️ 局限性**

局限在于结果对模型结构和数据集差异敏感，VP对提示设计高度依赖，LoRA在非Transformer模型上的实现尚不完善，且正则化手段尚未完全缩小与LoRA的性能差距。

---

## 715. semanticfa: An R Package for Response-Free Semantic Factor Analysis of Psychometric Scales

**arXiv ID:** 2610.37494 | [PDF](https://arxiv.org/pdf/2610.37494v1)

**作者:** Devon Yanitski `[一作]` (University of Alberta), Chris Westbury `[通讯]` (University of Alberta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开发了名为 semanticfa 的 R 包，利用大语言模型的嵌入向量对人格量表项目进行语义因子分析，并将得到的语义结构与人类受试者的原始响应结构进行对比。

**💡 创新点**

创新点在于首次将大语言模型生成的项目嵌入与传统因子分析相结合，提供一种自动化、可复现的语义结构提取方法；同时通过多种保留准则和多种嵌入规模进行系统比较，验证了语义结构与心理测量结构的一致性。

**🔧 技术方法**

技术包括：Qwen3-Embedding-8B/4B/0.6B 模型生成嵌入、NLI cross‑encoder 计算语义相似度矩阵、R 中的因子分析（EGAnet、GPArotation 等）、t‑SNE/UMAP 可视化、congruence metrics、retention criteria 评估。

**📊 数据集**

数据集为 50 个 IPIP Big‑Five 维度标记项目的题目及其在人类受试者中的响应（Open‑Source Psychometrics Project 数据集），并在补充材料中使用了 42 项 Depression Anxiety Stress Scales（DAS）进行模型规模比较。

**📈 对比分析**

方法通过在不同嵌入规模下拟合因子模型，比较语义相似度矩阵与人类响应相关矩阵的相关性、因子负荷一致性、保留准则选择的因子数等；实验结果显示语义结构与目标因子数高度一致，语义相似度与原始响应相关性显著（r≈0.6-0.8），说明该方法能较好地恢复人类心理结构。

**⚠️ 局限性**

局限性包括：语义与响应结构不一致时无法确定原因；模型主要验证于 IPIP 项目，对其他测量工具的推广需要进一步检验；大语言模型嵌入可能带有偏差；计算成本高（需 GPU 及多模型嵌入）；未进行预注册；对较小样本或非文本项目的适用性有限。

---

## 716. Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families

**arXiv ID:** 2610.37484 | [PDF](https://arxiv.org/pdf/2610.37484v1)

**作者:** Trent R. Northen `[一作]` (Bioaligned Labs), Mingxun Wang `[通讯]` (Bioaligned Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在论文中，作者通过在五个不同架构的开源大语言模型上进行一次持续预训练（QLoRA CPT），使用一份基于真实科学文献重写的“耦合福利”语料库，来教授生态推理，使模型在面对压力情境时更倾向于保护活生物系统。

**💡 创新点**

创新点在于将生态因果依赖关系（人类福祉↔生态功能↔模型自身能力）嵌入到训练数据中，而非单纯使用RLHF或价值层面后训练，证明这种内容驱动的训练能在压力量表下显著降低有害决策。

**🔧 技术方法**

技术实现主要依赖QLoRA持续预训练（无奖励模型、无偏好对、无情境监督），并对混合专家模型的适配器部分进行细粒度选择，保持兼容性与可合并性。

**📊 数据集**

使用的数据集是一个2.97万条已验证文献的重写集合（共934条训练文档，约1.2万标记），以及同尺寸、同风格但去除生态内容的对照语料（Arm‑N），每条文档均可追溯到原始论文。

**📈 对比分析**

通过压力梯度、开放分配决策网格和未见的保险承保评估等多重基准，实验显示耦合福利训练将有害决策从70/300下降到0/300（Llama‑3.3‑70B）、从105/300降至3/300（Qwen‑3.6‑27B），且MMLU等能力指标保持不变，表明效果显著且无能力损失。

**⚠️ 局限性**

主要局限包括单一随机种子实验、结果对推理堆栈和库版本高度敏感、压力梯度的动态范围有限、对Phi‑4梯度的重新评估、以及对数据集偏倚与构造特定性的未完全解决。

---

## 717. Follow the Gradient: Tracing Safety Degradation Back to Training Data

**arXiv ID:** 2610.37481 | [PDF](https://arxiv.org/pdf/2610.37481v1)

**作者:** Saket Reddy `[一作]`, Pratyush Maini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何利用梯度归因识别并移除导致安全退化的训练样本，构建SafeTulu数据集

**💡 创新点**

提出SimpleLoRIF梯度嵌入方法，简化LoRIF并限制反向传播至最后四层，实现百万级数据的可扩展过滤

**🔧 技术方法**

梯度归因、LoRIF、随机投影、低秩因式分解、聚类等技术

**📊 数据集**

Tulu 3、SafeRLHF、Qwen 3、Llama 3.2、Olmo 2等

**📈 对比分析**

与随机去除、LLM-as-a-judge、He等梯度过滤方法比较，SafeTulu在GCG和ActorBreaker攻击成功率上提升至22%，且保持或提升多项能力基准

**⚠️ 局限性**

仅评估一般安全退化，未针对特定有害行为或其他目标集进行测试

---

## 718. Autonomous fixed-wing interception using reinforcement learning

**arXiv ID:** 2610.37470 | [PDF](https://arxiv.org/pdf/2610.37470v1)

**作者:** Daan van Hamersveld `[一作]` (Delft University of Technology), Iurii Kriachko `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究开发了一种基于单目视觉的端到端强化学习控制器，用于实现固定翼无人机的自主拦截；

**💡 创新点**

创新点在于将视觉检测、卡尔曼滤波、姿态估计与强化学习策略整合为一体化网络，直接将视觉状态映射到机动控制指令，并在真实飞行中验证其可转移性；

**🔧 技术方法**

采用了深度强化学习（PPO）、自定义Python仿真环境、卡尔曼滤波、域随机化、图像目标检测算法与V-tail动态模型；

**📊 数据集**

使用自制的固定翼无人机X‑UAV Talon进行系统识别与飞行测试，并在仿真中随机生成目标轨迹、风速、噪声与检测丢失等多种扰动；

**📈 对比分析**

与传统比例导航（PN）与纯追踪（PP）指导法进行比较；在仿真中RL控制器在大多数条件下击中率超过80%，并优于PN/PP；在实飞中成功拦截6次中有3次成功，显示出与基线相比在真实环境中的显著优势；

**⚠️ 局限性**

主要局限包括：对视觉检测误差和假目标不够鲁棒；需要先获取目标检测后才能启动拦截；RL策略对大范围、快速目标、极端失真等情况仍易失效；算法对硬件平台的依赖性较高，需进一步优化自检与多传感融合。

---

## 719. ETAMem: Decoupling and Aligning Heterogeneous Memory for Long-Term Conversational Agents

**arXiv ID:** 2610.37443 | [PDF](https://arxiv.org/pdf/2610.37443v1)

**作者:** Xinying Qian `[一作]` (Nankai University), Jeff Z. Pan `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ETAMem多视图记忆框架，能够将对话信息拆解为事件、情感和时间三种视图，并通过熵引导的检索路由实现跨视图检索。

**💡 创新点**

创新点在于：①将异构会话信息解耦成原子记忆单元并对情感信息进行显式标注；②为每种信息类型构建专属层次化结构（事件层次、情感层次、时间层次）；③使用熵分布动态分配检索预算并在每个视图内进行层级束搜索；④自适应阈值校准与增量维护，提升长期记忆的可扩展性。

**🔧 技术方法**

技术包括：LLM（gpt‑4o‑mini / Qwen3‑30B）生成原子记忆单元和答案；情感分类器（RoBERTa‑GoEmotions）与文本编码器（RoBERTa‑Base）分别提供情感与文本表征；树结构维护（分裂、合并、摘要更新）与熵路由与束搜索算法；自适应阈值校准采用Otsu分割与指数滑动平均。

**📊 数据集**

使用 LoCoMo（多会话长记忆基准）和 LongMemEval‑S（长上下文问答基准）进行评测；同时在 CloneMem 上验证情感相关检索效果。

**📈 对比分析**

与 MemoryBank、A‑MEM、Mem0、MemoryOS、MemOS、MAGMA、TiMem、LightMem 等最强基线比较，ETAMem 在 LoCoMo 上整体准确率提升12.46%（相对于 TiMem），在 LongMemEval‑S 上提升7.96%；在多跳、时间推理等子任务中更显著，同时在构造和推理 token 方面也保持或优于竞争方法。

**⚠️ 局限性**

主要局限在于情感表征仅基于情感极性，可能无法识别中性偏好或细腻情绪，未来需加入更细粒度的情感相关性分类。

---

## 720. FeatureArc: Benchmarking Coding Agents on Evolving Features

**arXiv ID:** 2610.37434 | [PDF](https://arxiv.org/pdf/2610.37434v1)

**作者:** Yunlong Wang `[一作]` (Tencent), Zhichao Hu `[通讯]` (Tencent)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 L4 级别的功能进化评测基准 FEPBench，能自动从开源仓库历史中提取同一功能的多阶段扩展任务，并让编码代理在没有外部错误反馈的情况下持续迭代代码。

**💡 创新点**

创新点在于：①结合历史同一功能依赖、需求驱动的验证与可执行测试；②构建无中间人类检查的 L4 级别评估流程；③通过轨迹分析揭示代理在功能进化中遗留缺陷和自检失效的细粒度机制。

**🔧 技术方法**

使用了多阶段任务构建管道，包括仓库历史分析、特征链挖掘、参考状态构建、环境与测试验证以及需求与接口合同生成；评估采用隐藏的 fail-to-pass / pass-to-pass 测试以及最终链级检查；轨迹分析则利用模型日志和人工标注。

**📊 数据集**

数据集为从 500+ 开源仓库（Python、TypeScript、Rust、Go）中抽取的 73 个手工审核的多阶段功能链，总共包含 2–10 个阶段。

**📈 对比分析**

对 10 个主流大语言模型驱动的编码代理进行对比，单阶段成功率平均 62.7%，链级成功率仅 28.5%；在 L4 评估中，代理在首阶段表现良好但在后续阶段易失效，继承缺陷导致后续阶段成功率从 70% 降至 22%。

**⚠️ 局限性**

主要局限包括：①缺乏中间人工验证导致继承错误难以检测；②自检机制的有效性有限，绿色自检与隐藏测试结果吻合率低；③任务集规模有限，无法覆盖更长、更复杂的功能链；④评测主要集中在四种语言，跨语言通用性尚待验证。

---

## 721. Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well

**arXiv ID:** 2610.37652 | [PDF](https://arxiv.org/pdf/2610.37652v1)

**作者:** Ian Chuang `[一作]` (University of California, Berkeley), Claire Tomlin `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

训练了LEAP手的姿态到达策略，使其能在任意手腕角度下将物体移动到手内任意姿态，并在真实机器人上零样本迁移；

**💡 创新点**

提出了零样本训练流程（包括抓取数据集生成、摩擦与目标自适应课程、精细的奖励与安全过滤器），以及可调节的残差安全过滤器，显著降低掉落率；

**🔧 技术方法**

使用强化学习（PPO）与DAgger、异构演员-评论家、可视深度CNN、Fast FoundationStereo等技术，结合摩擦与目标自适应课程、系统辨识、延迟校准；

**📊 数据集**

利用自生成的100k+抓取数据集（LightningGrasp）生成多达400k+可达抓取姿势，模拟不同手腕方向与随机重置；

**📈 对比分析**

在真实机器人上共计4800个目标（13.3小时）验证，状态策略在粗粒度目标上95.2%成功率，安全过滤器将掉落率降至75%，深度策略在无运动捕捉条件下亦可实现高成功率；

**⚠️ 局限性**

局限性：策略和过滤器需按物体单独训练；深度策略在刀具和葡萄酒杯上存在视觉仿真差距；未实现多物体、零掉落率和更广泛的安全过滤器通用化。

---

## 722. Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence

**arXiv ID:** 2610.37633 | [PDF](https://arxiv.org/pdf/2610.37633v1)

**作者:** Siyoon Jin `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究跨视角时序物体对应，提出联合视角-时间掩码预测框架。

**💡 创新点**

创新点在于将预训练视频Diffusion Transformer重用于RGB→掩码的流匹配，并在统一token空间中实现跨视角注意力对齐、硬负样本调节、小目标局部上下文裁剪，以及伪标签精炼管线。

**🔧 技术方法**

技术包括视频Diffusion Transformer (DiT)、流匹配（Flow Matching）、跨视角注意力对齐（Focal Cross-View Attention Alignment）、硬负样本调节（Hard Negative Conditioning）、局部上下文裁剪（Local Context Crop）、RoPE位置编码、预训练权重迁移、以及伪标签精炼管线。

**📊 数据集**

数据集为 Ego-Exo4D（含稀疏标注的 ego‑和 exo 视角视频），并通过伪标签精炼实现稠密监督。

**📈 对比分析**

与传统的 Track‑then‑Match、Match‑then‑Track 等分离匹配+跟踪基线比较，在保留身份一致性与时序一致性方面实现更高的可见性准确率 (VA) 和 mR@IoU，mask 质量竞争性，尤其在跨视角连贯性上表现突出。

**⚠️ 局限性**

限制：视频 DiT 的潜在空间压缩导致细粒度边界表现不如高分辨率单视角模型；训练依赖复杂的伪标签生成流程，需大量人工或辅助模型保证标注质量。

---

## 723. What Improves During Pretraining? Tracking Knowledge Acquisition with Point-in-Time Models

**arXiv ID:** 2610.37631 | [PDF](https://arxiv.org/pdf/2610.37631v1)

**作者:** Chang Su `[一作]` (University of Pennsylvania), Yao Lu `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并使用时间可变性得分（TVS）来衡量不同年份训练数据对语言模型知识学习的影响，利用点在时间（PIT）模型在多年 Web 爬虫数据上训练得到的负对数似然系数变异度，评估 16,239 条多项选择题目在六个模型（Pythia、TinyLlama、SmolLM3、OLMo 等）不同训练阶段的学习进程。

**💡 创新点**

创新点在于：① 用 TVS 这一基于多年份模型的系数变异度指标，客观量化知识在训练数据中的时间波动；② 在不需要直接访问完整训练语料的情况下，通过 PIT 模型间的对数似然差异来诊断模型学习动态；③ 将 TVS 与模型性能差异关联，揭示高 TVS 项目在后期训练中学习更快、最终性能更高的规律。

**🔧 技术方法**

使用的技术包括：① 计算负对数似然的标准差与均值之比得到 TVS；② 在不同长度 bin 内将项目分为低/中/高 TVS 三组；③ 在模型检查点上对答案选项进行长度归一化似然排序，计算机会归一化准确率；④ 对高低 TVS 项目在训练阶段的准确率和答案与最佳干扰项的边际斜率进行回归分析；⑤ 进行发布年份与文本重叠等泄漏检测。

**📊 数据集**

所用数据集：PIT-4B 模型基于 FineWeb 2017–2024 年份的 Web 爬虫；评估基准包括 HellaSwag、PIQA、WinoGrande、ARC‑Easy、ARC‑Challenge、MMLU，总共 16,239 条样本。

**📈 对比分析**

比较方法：在每个模型的多次训练检查点上，分别计算低/高 TVS 组的机会归一化准确率，观察随训练标记数增长的差距。结果显示，早期低/高 TVS 组相近，但在后期高 TVS 组学习速率更快，最终在 34/36 个模型‑基准组合中高 TVS 项目平均领先 7.3 分（机会归一化准确率）。此优势在知识密集型任务（ARC、MMLU）上最为显著。

**⚠️ 局限性**

局限性：TVS 依赖所用 PIT 模型的质量与规模，仅在单一 4B 参数模型上评估；污染检查仅针对 PIT‑4B 训练数据，未覆盖评估模型完整预训练语料；分析仅限于多项选择基准，无法直接推广到其他任务类型。

---

## 724. VideoLVR: Latent Visual Reasoning over Grounded Video Evidence

**arXiv ID:** 2610.37613 | [PDF](https://arxiv.org/pdf/2610.37613v1)

**作者:** Shilong Li `[一作]`, Shilei Wen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VideoLVR框架，利用时空分布的局部视觉证据在推理过程中生成潜在视觉状态并与文本推理交错；

**💡 创新点**

创新点在于：①通过证据驱动的数据合成生成局部视觉证据轨迹；②将视觉证据编码为潜在视觉块并用视觉词预测（VTP）进行监督；③引入TVPO实现文本与视觉动作的联合强化学习；

**🔧 技术方法**

核心技术包括：潜在视觉推理、视觉词预测目标函数、强化学习（GRPO/TVPO）、多模态自回归生成；

**📊 数据集**

使用了OneThinker、Video‑R1等公开视频问答数据，合成约8.8K证据驱动轨迹；

**📈 对比分析**

在五大视频推理基准（Video‑MMM、TempCompass、MMVU、LongVideo‑Reason、VideoMathQA）上，VideoLVR-8B/7B平均提升约3–4分；强化学习版本进一步提升约1分；与基线模型（Video‑R1、Open‑o3‑Video、Mull‑Tokens、LVR）相比均取得显著优势；

**⚠️ 局限性**

主要局限是依赖合成的时空标注，难以直接迁移到真实未标注的视觉证据；

---

## 725. Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence

**arXiv ID:** 2610.37605 | [PDF](https://arxiv.org/pdf/2610.37605v1)

**作者:** Yunuo Wang `[一作]`, Xing Liu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了音频-语言模型对局部声学事件编辑的响应，利用匹配的同类与不同类替换对比评估模型的答案变化。

**💡 创新点**

提出了匹配的局部波形干预协议，比较同类重写与不同类替换的答案变化，并引入选择性准确率（Selective Accuracy）来衡量模型是否能正确识别插入的事件类别。

**🔧 技术方法**

使用了三种音频问答模型（Qwen2.5-Omni-7B、Qwen2.5-Omni-3B、Fun-Audio-Chat-8B），在合成声景上进行局部波形替换，统计答案变化概率、协方差修正以及log‑mel编辑距离。

**📊 数据集**

构造了600个合成声景，前景来自FSD-MIX-SED（FSD50K），背景来自TUT Urban Acoustic Scenes 2018，覆盖33个查询类别。

**📈 对比分析**

通过比较不同类替换与同类重写的答案变化率发现，前者产生49.7–60.9个百分点的额外答案变化；在满足原始答案正确的样本上，选择性准确率达到73.7–81.9%，表明模型对声学事件类别具有显著响应。

**⚠️ 局限性**

实验基于合成声景，缺乏生态真实性；方法依赖已标注的事件边界和同类 donor，难以直接迁移到自然录音，且对不同声学条件的泛化能力有限。

---

## 726. GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting

**arXiv ID:** 2610.37598 | [PDF](https://arxiv.org/pdf/2610.37598v1)

**作者:** Joseph Natter `[一作]` (Kennesaw State University), Rui Wu `[通讯]` (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种零射交通预测模型 GAS-PFN，结合图结构的合成任务预训练和双掩码 Transformer 推理，实现无需参数更新即可在新或演化的道路网络上进行预测。

**💡 创新点**

创新点包括：① 在合成任务中通过随机游走扩散和道路拓扑采样引入图耦合时空动力学；② 推理时使用自掩码和多通道掩码两条分支，分别捕捉传感器局部时序和邻域空间交互；③ 混合最近历史与历史锚点的上下文记忆；④ 在 PFN 框架中首次将图结构显式嵌入任务先验与推理架构。

**🔧 技术方法**

使用技术包括 Prior-Data Fitted Network (PFN)、Transformer 主干、随机游走扩散、双掩码注意力、混合上下文记忆、门控融合、辅助损失以及图与语义混合邻域路由。

**📊 数据集**

实验数据集为六个流量/速度预测基准：METR-LA、PEMS-BAY、PEMS03、PEMS04、PEMS07、PEMS08，均以 5 分钟采样，覆盖不同规模与地理位置的道路网络。

**📈 对比分析**

与 TimePFN、Chronos-2、TimesFM-2.5、Moirai-2、Lag-Llama 等零射基线以及四个工业通用零射模型进行对比；在 90 分钟预测时，GAS-PFN 的 MSE 平均比 TimePFN 低 34.0%，比四个通用模型低 27.7%；在 36 次 MAE/MSE 对比中获得 27 次最佳结果，证明在零射环境下显著优于现有方法；在有限监督预算对比 SCPT 时，GAS-PFN 在大多数情况下表现更好。

**⚠️ 局限性**

局限性包括：① 需要较大的上下文预算（K≥16）才能充分发挥优势；② 在规模较小或波动不大的网络（如 METR-LA、PEMS-BAY）上提升有限；③ 模型参数固定，无法在线快速适配急剧变化的网络拓扑；④ 训练与推理成本相对较高；⑤ 仅在零射场景评估，缺乏在充分监督环境下的对比。

---

## 727. Spectral Braids: A Topological Descriptor for Dynamic Directed Networks

**arXiv ID:** 2610.37580 | [PDF](https://arxiv.org/pdf/2610.37580v1)

**作者:** Mehmet Emin Aktas `[一作]` (Kennesaw State University), Esra Akbas `[通讯]` (Georgia State University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了谱结迹法（spectral braids），通过跟踪动态有向网络的社区层级转移矩阵的复特征值随时间的绕行轨迹，将其转化为结点交叉序列（braid word），并利用Burau表示将不等长的结织转化为固定维度特征向量，用于动态图网络分类。

**💡 创新点**

创新点在于：①首次将拓扑学中的结（braid）和绕回（braid monodromy）引入图数据；②提出仅关注上半平面特征值的上半平面结，以消除实矩阵对称性导致的冗余；③将结织映射到Burau表示得到可直接输入机器学习模型的特征；④给出复特征值与社区行为（回声室与广播水平）的直观解释。

**🔧 技术方法**

使用的技术包括：社区转移矩阵构造、复特征值跟踪与最小代价匹配、结织读数与简化、Burau表示矩阵计算、特征统计（tracked geometry、exponent sums）以及随机森林分类器。

**📊 数据集**

使用了两套人工合成数据集（History 与 Order），分别模拟在线社区的回声室与广播演化，并生成具有已知标签的动态网络；还在论文中讨论了四个真实网络（电子邮件、国际贸易、Reddit 超链接、Bay Area Bike Share）作为可行性检验。

**📈 对比分析**

与不跟踪特征值的基线（池化谱、起止谱、排序序列）比较，Burau特征在 History 数据集上实现97.1%准确率，远高于基线的44.5%；在 Order 数据集上实现90.8%准确率，基线仅为随机；跨数据集转移（平静↔动荡）仍保持高准确率，说明特征对速度与形状不敏感。

**⚠️ 局限性**

局限性：仅在存在非平衡循环流（circulation）的有向网络上产生非平凡结织；在大多数真实网络中由于流量几乎平衡或只有两条复杂特征值，结织往往退化为平凡；需要每个时间窗口有足够多的边（数千条）才能保证特征稳定；目前仅处理社区级别矩阵，未考虑节点级别特征。

---

## 728. Pneuma: A Large-Scale Real-World Time Series Benchmark for Commercial Aircraft Predictive Maintenance

**arXiv ID:** 2610.37563 | [PDF](https://arxiv.org/pdf/2610.37563v1)

**作者:** Jiafeng Lin `[一作]` (Tsinghua University), Jianmin Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了Pneuma，一个公开的大规模机型泄压系统时序数据集和三任务基准，涵盖四种商用机型、920M时间步及9个真实故障场景。

**💡 创新点**

首次公开真实故障标注的大规模航空时序数据，并统一定义预测、分类和异常检测三任务与评估指标，提供可复现的基准平台。

**🔧 技术方法**

采用多变量时间序列模型（如DLinear、TimesNet、Chronos‑2、iTransformer 等）以及预训练时序基础模型、卷积与 Transformer 分类器和重构式异常检测器进行系统对比。

**📊 数据集**

使用Pneuma数据集，该集由A320、A321、B787、B777的泄压系统QAR记录与维护日志组成，共计约9亿时间步和9个故障场景。

**📈 对比分析**

在统一拆分与评估下对25个模型进行基准，发现即使最强模型仍难以捕捉细微故障；DLinear在预测任务中表现最佳，监督分类相对更好，而异常检测效果差，说明现有模型对航空故障的辨识能力有限。

**⚠️ 局限性**

局限包括仅聚焦泄压系统、重构式异常检测难以区分正常与早期故障、缺乏跨机型迁移验证以及对非泄压系统的通用性不足。

---

## 729. When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance

**arXiv ID:** 2610.37561 | [PDF](https://arxiv.org/pdf/2610.37561v1)

**作者:** Walter Endrizzi `[一作]` (Fondazione Bruno Kessler), Venet Osmani `[通讯]` (Queen Mary University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在严重类别不平衡的临床预测任务中，揭示标准不确定性量化（如 Shannon 熵、分类边距）会被类稀缺性误导，导致少数类系统性排除。提出阈值对齐的模糊熵 H_τ、类条件拒绝 (CCR) 双层框架，并引入类分层的 Risk-Coverage 曲线评估（ccAUGRC），以恢复少数类的安全预测。

**💡 创新点**

创新点在于：①阈值对齐的熵 H_τ 将最大不确定性对准临床阈值 τ；②CCR 在预测类别内部进行排序，从而消除绝对不确定性底层对少数类的不利影响；③ccAUGRC 将整体 Risk-Coverage 分解到各类，揭露传统全局评估掩盖的类不平衡风险。

**🔧 技术方法**

使用了后验概率校准（Platt scaling）、异构集成（XGBoost、RF、Extra Trees、LR、SVM）、熵与距离的阈值对齐度量、类条件拒绝队列、类分层 AUPRC、AUGRC/ccAUGRC 评估、Bootstrap 置信区间与 Holm-Bonferroni 校正。

**📊 数据集**

实验基于三组真实神经退行性疾病数据（MS π=0.13，PD π=0.39，AD π=0.54）以及 56 条合成平衡/不平衡情形（π∈{0.05,…,0.5}，d∈{0.5,1,2,3})。

**📈 对比分析**

与全局熵、边距、随机+CCR 等基线比较，H_τ+CCR 在 56 条合成条件及三组真实数据上显著提升少数类敏感度（在 π≤0.2 时从 0% 提升至 80%+），保持类风险平衡，AUPRC 上升，类差异（class disparity）降至 0.01 以内，整体风险（global AUGRC）下降。

**⚠️ 局限性**

局限性包括：仅适用于二分类，扩展到多分类需重新设计阈值对齐与 CCR；依赖后验概率校准，若校准失效可能影响阈值对齐；阈值 τ 固定，若阈值随时空变化需手动调整；集成模型计算量大；评估仅覆盖测试集，未考虑训练过程方差；数据受隐私限制，复现性受限。

---

## 730. GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis

**arXiv ID:** 2610.37545 | [PDF](https://arxiv.org/pdf/2610.37545v1)

**作者:** Akshay Krishnan `[一作]` (Niantic Spatial), Daniyar Turmukhambetov `[通讯]` (Niantic Spatial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用卫星图像作为条件，结合多视角扩散模型进行街景新视角合成的方法。

**💡 创新点**

创新点包括引入ray‑guided cross‑attention与双重classifier‑free guidance，将卫星图像与街景输入协同建模，同时保留预训练的多视角生成先验。

**🔧 技术方法**

使用技术包括预训练的SEVA多视角扩散模型、DINOv3卫星图像编码器、ray‑guided cross‑attention、双CFG、DDIM采样等。

**📊 数据集**

实验数据集涵盖nuScenes、Argoverse‑v2、Metropolis、VIGOR、自驾车数据和手持相机图像，并在DL3DV等外域数据上验证。

**📈 对比分析**

与SEVA、Sat2Density等基线在PSNR/SSIM/LPIPS/语义IoU等指标上进行对比，实验显示在视角缺失或离散时性能显著提升，同时保持多视角一致性。

**⚠️ 局限性**

局限性包括需已知地理配准的卫星图像和相机位姿，对斜角或高架卫星图像适用性有限，且对训练分布外的场景仍可能受限。

---

## 731. Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops

**arXiv ID:** 2610.37656 | [PDF](https://arxiv.org/pdf/2610.37656v1)

**作者:** Aakash Sarkar `[一作]` (University of California), Motoshige Sato `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于 Gauge 理论的边级注意力机制——Holonomy Attention，用于无监督地在关系图中对边进行评分与更新。

**💡 创新点**

通过对每个实体的坐标系自由度进行约束，推导出唯一的可变换不变边级更新形式，即以 Wilson 环（holonomy）为评分的非阿贝尔注意力；并将其与已有方法对齐，证明其在一般图上的唯一性。

**🔧 技术方法**

Gauge 等变信息传递、Wilson 环计算、极化投影、无监督图同步与社区检测，以及基于 S_4 群的组同步实验。

**📊 数据集**

主要使用了 S_4 群的人工同步数据（14 个实体、随机图），以及对三角形、四边形的不完整关系进行实验；未使用公开图数据集。

**📈 对比分析**

与 Singer 的特征向量方法、循环边消息传递以及单路径/统一权重方法进行比较。Holonomy Attention 在稀疏图上 p=0.2 时达到 0.89 的准确率，超过单路径 0.63；在密集图上与最先进方法相当或略优。AUC 0.99 的三角形一致性检测和 AUC 0.98 的社区分离表现突出。

**⚠️ 局限性**

仅在离散群设置下验证；缺乏对连续 Lie 群或大规模真实图的实验；目前为无监督且无学习参数，缺少可扩展的动态生成机制；对非阿贝尔群的实现仍需进一步优化。

---

## 732. GEPA-Curator: Curated Learning for Self Improving Agents

**arXiv ID:** 2610.37651 | [PDF](https://arxiv.org/pdf/2610.37651v1)

**作者:** Zack Yu `[一作]` (University Of California Berkeley), Amir Gholami `[通讯]` (University Of California Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 GEPA-Curator 的框架，利用数据策展器对提示优化过程进行任务筛选和成功轨迹检索，从而提升 LLM 代理的自我改进能力。

**💡 创新点**

创新点在于将任务策划与先前成功轨迹检索两种技术结合，主动剔除噪声任务并为反思提供可靠的参考，显著提升了提示优化的质量和鲁棒性，并可扩展至 Meta-Harness 代码优化。

**🔧 技术方法**

技术上采用了反思式提示优化（GEPA、ACE）、大语言模型（Gemini 3.1 Pro、Claude Opus 5）、任务历史存储、学习曲线分析、效率化变体以及与 LLM 交互的策展器。

**📊 数据集**

使用的数据集包括 WebArena 下的 GoBrowse（GitLab、Shopping、Shopping Admin、Reddit）、ALFWorld、AppWorld、τ-bench 以及 Math-RAG、USPTO-50k、Symptom2Disease、LawBench 等文本分类任务。

**📈 对比分析**

与 GEPA 和 ACE 进行对比实验，GEPA-Curator 在所有评估基准上均超越基线，平均提升 14–21% 的成功率，且效率化版本在保持性能的同时将总成本降低至基线的约 1.3 倍。

**⚠️ 局限性**

局限性包括对历史记录和错误判定的依赖、仍需大量 LLM 调用导致的计算开销，以及在某些极端噪声环境下可能无法完全过滤无用信号。

---

## 733. UPP$^{\text{cc}}$ is Not Closed Under Intersection

**arXiv ID:** 2610.37634 | [PDF](https://arxiv.org/pdf/2610.37634v1)

**作者:** Mark Bun `[一作]` (Boston University), Mandar Juvekar `[通讯]` (Boston University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了显式矩阵，证明无界误差通信复杂度类UPP不闭合于交集；

**💡 创新点**

首次给出U(F)=O(log n)与U(F∧F)=Ω(√n)的指数分离，利用连续对称化与平均边缘上界技术；

**🔧 技术方法**

Forster平均边缘方法、球面对称化、Gegenbauer多项式展开、逼近理论与多变量多项式符号表示；

**📊 数据集**

不使用任何数据集，全部为理论构造；

**📈 对比分析**

与先前仅实现多项式或二次分离的结果相比，本工作实现指数级分离，并在构造的函数上证明了下界的最优性；

**⚠️ 局限性**

仅对特定函数族有效，无法直接推广到所有UPP问题；所需维度和参数相对较大，且分离幅度仍低于线性。

---

## 734. EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning

**arXiv ID:** 2610.37609 | [PDF](https://arxiv.org/pdf/2610.37609v1)

**作者:** Sihan Yang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出环境级数据管理框架，在 Agentic RL 中以环境‑任务对为基本单位进行任务调度；

**💡 创新点**

创新点在于构造双关系图（within 与 cross）来预测任务奖励方差，并将覆盖与利用相结合的在线调度器；

**🔧 技术方法**

使用基于 LLM 的任务语义提取、TF‑IDF 与抽象匹配构建相似度，结合图传播、奖励方差回归与 GRPO RL 等技术；

**📊 数据集**

在 EnvScaler 可执行环境池（2,550 个任务，51 个环境）上训练，评估 BFCL‑v3、τ‑Bench、ACEBench‑Agent 三个基准；

**📈 对比分析**

与随机调度、DPS、LEEPS 比较，平均在 BFCL‑v3 提升 3–4 分，在 τ‑Bench 与 ACEBench‑Agent 提升约 3–3.5 分，且在 40 步内已达到 200 步随机基准的水平；

**⚠️ 局限性**

局限性包括对图邻居大小和相似度构造的超参敏感，未验证对新环境的迁移能力，以及仅在特定模型和数据集上展示效果。

---

## 735. TANDEM: Balancing Centralized and Decentralized Critics to Address the Information Gap in Complex Multi-Agent Environments

**arXiv ID:** 2610.37592 | [PDF](https://arxiv.org/pdf/2610.37592v1)

**作者:** Alireza Saleh Abadi `[一作]`, Prashant Doshi `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了多智能体强化学习中集中训练、去集中执行导致的信息差距问题，并提出TANDEM方法通过单一集中批评器的两种读取和无参数门控混合优势来缓解该差距。

**💡 创新点**

创新点在于保持单一集中批评器，同时在训练时按代理本地信息与全局信息两种读取，并使用无参数门控仅在两种读取符号不一致时才使用全局优势，实现连续的集中/去集中批评器平衡。

**🔧 技术方法**

技术主要包括CTDE框架下的MAPPO（近端策略优化）与注意力编码器构建的可变实体批评器、两种读取方式、参数无关门控与共享目标回归。

**📊 数据集**

实验使用了三个合作域：Vectorized Multi-Agent Simulator中的目标发现与采样任务，以及3×3格 wildfire suppression 环境，涵盖了部分可观测性、观测噪声与任务/代理开放三种信息差距。

**📈 对比分析**

与六种主流CTDE基线（MAPPO、COMA、QMIX、VDN、MAVEN、MADDPG）比较，TANDEM在9个设置中8个获得最高执行回报，且在信息差距最大的设置中显著优于所有基线。

**⚠️ 局限性**

局限在于仅处理代理与任务开放，未考虑帧开放；门控仅对优势符号不一致进行校正，可能忽略两种读取均错误的情况；且实验仅分离单一信息差距，未评估多重差距交叉情况。

---

## 736. GNBG-C: A Unified and Configurable Benchmark Generator for Constrained Continuous Optimization

**arXiv ID:** 2610.37590 | [PDF](https://arxiv.org/pdf/2610.37590v1)

**作者:** Ali Ghasemzadeh `[一作]` (University of Hong Kong), Amir H. Gandomi `[通讯]` (University of Technology Sydney)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 GNBG-C，一个可配置的数值基准生成器，用于在已知全局最优点周围生成可调节可行区域的连续约束优化实例。

**💡 创新点**

创新点在于把可行区域视为独立可配置的对象，利用中心化约束层保留已知最优点，同时通过可行性比率、活跃约束数、边界形状、变量耦合、数值尺度与冗余等参数精确控制可行空间，从而实现可审计、可复现的约束难度调节。

**🔧 技术方法**

技术实现包括 GNBG 基准目标函数、中心化约束层（g_j = ψ_j(z_Ij;θ_j) - κ·b_j）、Sobol 采样校准可行性比率、梯度非零筛选活跃约束、线性/凸/非凸二次、多项式、超越型约束族，以及多种约束处理策略（DE‑FF/DE‑EPS/DE‑PEN、PSO‑FF、GA‑FF、SRES、ISRES、SCIPY‑DE）进行实验评估。

**📊 数据集**

数据集为 d=10 维的 GNBG‑C 生成实例，共 8 种约束配置，每种配置 15 个实例；实验使用 8 种优化算法，每个实例 30 次运行，总计 28,800 次实验；同时公开 GNBG‑C‑24 24 个固定实例供复现。

**📈 对比分析**

比较方法：在每个算法-实例对上记录首次可行发现、可行目标误差以及 25% 和 50% 目标阈值的达成率；结果显示不同约束设置导致算法在可行性发现速度、可行目标改进以及最终误差上的显著差异，SRES、ISRES 和 DE 变体在多数设置下表现最好，PSO/GA 较弱，SCIPY‑DE 在数值尺度差异较大时受影响显著。

**⚠️ 局限性**

局限性：仅针对已知最优点可行的情形，未覆盖最优点不可行或多目标、等式约束、混合整数、动态/噪声约束等更复杂场景；实验维度受限于 d=10，缺乏对更高维度或真实工业案例的验证。

---

## 737. REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

**arXiv ID:** 2610.37584 | [PDF](https://arxiv.org/pdf/2610.37584v1)

**作者:** Yuxi Huang `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出REGER框架，通过语义预训练、项目转移建模和协同偏好蒸馏，显式建模项目之间的语义与协同关系，从而提升生成式推荐性能。

**💡 创新点**

创新点在于①利用生成语义检索预训练让生成器先学习项目语义邻居关系；②在编码器引入Item Transition Structure Attention，使注意力同时捕捉全局历史与局部项目转移；③基于SID前缀树的协同偏好分解提出HCPD，直接对齐教师与学生在合法分支上的条件分布。

**🔧 技术方法**

使用残差量化SIDs、T5式自回归生成器、全局/局部注意力混合、前缀树条件KL分解以及协同偏好蒸馏等技术实现。

**📊 数据集**

在Amazon Reviews 2014的Beauty、Sports、Toys三大细分以及2023的Instruments、Scientific、Games六个数据集上进行评测。

**📈 对比分析**

与SASRec、TIGER、LIGER等传统与生成式基线及多种蒸馏/监督方案对比，REGER在Recall@5/10和NDCG@5/10上均取得显著提升，尤其在HCPD训练下常占据前列。

**⚠️ 局限性**

局限性在于需依赖预训练语义邻居及冻结教师模型，且预训练与蒸馏步骤增加训练成本，此外仅在目标路径上监督，可能忽略非目标分支的细粒度偏好。

---

## 738. Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents

**arXiv ID:** 2610.37567 | [PDF](https://arxiv.org/pdf/2610.37567v1)

**作者:** Pranjal Aggarwal `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个自动生成实时视觉小游戏的框架，构建了100个实时小游戏及其非实时对应版本，并在这些任务上评估了多种大型语言模型，展示了当前模型在实时交互中的显著性能瓶颈；

**💡 创新点**

创新点在于通过编程代理自动化生成可控难度、可暂停与可调速的实时视觉小游戏，使得可以在同一任务上独立比较实时性、难度和交互模式；

**🔧 技术方法**

使用了大型语言模型（如GPT-6 Astra、Claude Opus、Qwen3.8-27B等）结合可视化交互接口、暂停/慢速控制栈，以及代码执行沙箱；

**📊 数据集**

数据集为论文自行生成的100个实时小游戏（及其对应的非实时版本）与四个难度级别与两种交互模式，总计2000个配置；

**📈 对比分析**

方法通过对比同一小游戏在实时、慢速、暂停三种时钟模式下以及实时与非实时版本的通过率、平均耗时和成本进行性能评估；结果显示最强模型GPT-6 Astra在实时模式下通过率仅9%，而在暂停模式下提升至75%；

**⚠️ 局限性**

限制在于模型缺乏足够的实时反应速度，且当前实验仅覆盖视觉小游戏，无法直接推广到更复杂的游戏或物理交互环境；

---

## 739. ReferStream: Interactive Subject Injection for Streaming Video Generation

**arXiv ID:** 2610.37566 | [PDF](https://arxiv.org/pdf/2610.37566v1)

**作者:** Shuolin Xu `[一作]` (Bournemouth University), Xiaobin Hu `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在视频流式生成过程中，提出一种不需要额外训练的新框架，可在已生成的画面中动态插入新视觉参考，同时保持已有主体不变。

**💡 创新点**

创新点在于：① 通过在当前场景上下文中构造参考特征，提升参考与场景的语义一致性；② 在生成块中临时写入缓存并随后恢复，避免对已生成内容的长期影响，从而实现“及时插入、永久保留”的双重目标。

**🔧 技术方法**

使用了冻结的 LongLive2.0 基础模型、变分自编码器（VAE）对参考图像编码、Latent Diffusion Transformer（DiT）进行噪声去除与缓存写入、ROI 盒子与 α 混合实现参考特征构造、RoPE 对齐实现时序一致、缓存写入/恢复机制控制参考影响时长。

**📊 数据集**

构建了 ReferStream benchmark，包含 48 条 InShot 条目与 20 条 MultiShot 条目，涵盖 139 个参考主体（人、动物、物体）。对比使用 OpenS2V-Nexus、VACE、Phantom、HOMIE 等公开基准。

**📈 对比分析**

方法通过与 LongLive2.0、VACE-14B、Phantom-14B、HOMIE-14B 在同一生成脚本与信息设置下对比。评估指标包括美感（Aesthetic）、运动平滑度（Motion）、文本匹配（GME‑T）、新主体相似度（New‑ID）、已存在主体保留（Existing‑ID）以及计数准确率（Count Accuracy）。结果显示，本文方法在 New‑ID、Existing‑ID、Count Accuracy 以及文本匹配上均优于基线；视频质量与 VACE 相近，运动平滑度略低。

**⚠️ 局限性**

局限性包括：① 新主体相似度与已存在主体保留之间存在权衡，导致 New‑ID 略有下降；② 自动评测对检测错误敏感，可能低估真实表现；③ 视频质量提升有限，尚未覆盖更复杂场景与跨种子验证；④ 缺乏人类主观评估与更细粒度的交互实验。

---

## 740. Kinematic Constraint-Referenced Evaluation for Articulated Object Manipulation

**arXiv ID:** 2610.37550 | [PDF](https://arxiv.org/pdf/2610.37550v1)

**作者:** Baptiste Bonin `[一作]` (Université Laval), Audrey Durand `[通讯]` (Université Laval)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于关节轴投影的Constraint-Referenced Strategy（CRS），用来评估机械臂抓取姿态的执行命令是否符合被操纵对象的运动约束，并给出进度与浪费的双通道签名；该方法不依赖演示数据或学习的评判器；

**💡 创新点**

创新点在于将关节轴作为评价参照，将任意关节的Twist投影到轴上，直接量化指令进展与非约束浪费，从而补足传统的二进制成功率评估，能够区分指令误差类型并提供可解释的诊断；

**🔧 技术方法**

使用机器人关节轴模型、Twist投影、单步反事实门控、R²线性拟合与方向一致性检查，以及现有的OpenVLA-7B与点云行为克隆（BC）策略、MetaFine稳定性度量、DTW等技术；

**📊 数据集**

在ManiSkill数据集（GAPartNet）中，评估74个可关节化对象（39个门、22个抽屉、13个水龙头），对三类机制进行留一机制外泛化测试；

**📈 对比分析**

与传统二进制成功率、Δq_real、四个命令侧度量、DTW、MetaFine稳定性等指标对比，CRS能准确捕捉三种失败（误传、反向、离轴）并给出不同签名，显示出更高的诊断性；鲁棒性测试显示对关节轴误差高达30°仍保持一致的判定；

**⚠️ 局限性**

局限性包括：假设抓握刚性，单一1-DOF关节；仅评估角度浪费而未量化其物理成本；实验仅在仿真环境中进行，未验证真实机器人；需要关节轴标注，限制了在未知或复杂机械结构上的直接应用。

---

## 741. Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?

**arXiv ID:** 2610.37662 | [PDF](https://arxiv.org/pdf/2610.37662v1)

**作者:** Jialv Zou `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为Factorized Vector Quantization（FVQ）的多码本量化方法，利用稠密紧框架投影将视觉特征分解为多个固定维度的视图，并在工业规模训练下将离散视觉 token 与大型语言模型进行整合。

**💡 创新点**

创新点在于：① 证明量化误差并不决定语义保真度；② 通过增加有效位预算提升语义容量；③ 设计FVQ实现对称并行码本，避免残差层级导致信息衰减。

**🔧 技术方法**

采用的技术包括稠密紧框架投影、对称并行量化、紧框架合成、预训练 Qwen3-8B 语言模型、RoPE 位置编码和工业级多阶段训练流程。

**📊 数据集**

使用的数据集覆盖了通用感知与推理（MMMU、MMStar、RealWorldQA）、数学推理（MathVista）、OCR/文档理解（OCRBench、DocVQA、OmniDocBench、CharXiv）、图表与图形理解（ChartQA、AI2D）等，训练集约 200B 令牌。

**📈 对比分析**

与连续视觉表示（如 Qwen3-VL）以及其他离散表示模型（UniAR、Emu3-Chat）对比，FVQ 在九项指标平均上分别比 UniAR 提高 22.8%、比 Emu3-Chat 提升 59.2%，并在多项基准上实现或逼近连续模型的最佳成绩。

**⚠️ 局限性**

局限性包括：① 对高频细节的量化仍可能产生损失，尤其在极低分辨率或极高细节需求的 OCR 场景；② 目前仅针对 Qwen3 系列视觉编码器验证，其他视觉编码器的通用性仍待深入；③ 对极大图像（如 20k×4k）仍需进一步优化并行投影与码本数量。

---

## 742. Metacognitive Decoding: Efficient Inference for Diffusion Language Agents

**arXiv ID:** 2610.37642 | [PDF](https://arxiv.org/pdf/2610.37642v1)

**作者:** Mingyu Zhao `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种元认知控制器，用于根据交互状态动态调节扩散式语言代理的解码过程，从而显著减少推理 FLOPs 并保持甚至提升任务准确率。

**💡 创新点**

创新点在于将“自我对齐”与“流动解码”两层级的控制策略组合，既根据模型置信度自动完成可预测结构，又通过动态缩减激活范围消除冗余计算，使扩散生成更具环境感知。

**🔧 技术方法**

技术实现包括：1）扩散块级语言模型（Block Diffusion）基底；2）轻量级元认知控制器，利用两层 Transformer 编码监测信号；3）两阶段解码策略——置信度对齐（confidence alignment）和流动（flux）政策；4）基于奖励的强化学习（GRPO）训练控制器以平衡准确性与成本。

**📊 数据集**

使用的主要数据集有 BFCL v3（工具调用），HotpotQA、2WikiMultiHopQA 和 MuSiQue（多跳推理）四个基准。

**📈 对比分析**

与 LLaDA2.0‑mini、LLaDA‑8B‑Tools、DLLM‑Searcher、Fast‑dLLM 等基线对比，平均减少约 20.6% 的 FLOPs，且在 5 个基准上保持甚至提升准确率，表现出最优的计算‑效率权衡。

**⚠️ 局限性**

局限性：控制器目前针对特定扩散模型（如 LLaDA‑8B‑Instruct）训练，需进一步研究如何在新出现或更大规模的扩散模型上快速迁移与适配。

---

## 743. Know, then Draw: A New Learning Paradigm for Visual Generation

**arXiv ID:** 2610.37626 | [PDF](https://arxiv.org/pdf/2610.37626v1)

**作者:** Yucheng Mao `[一作]` (University of California San Diego), Hanwen Jiang `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了将视觉生成拆分为知识获取的自回归预训练与像素渲染的扩散后训练两阶段框架，并基于此提出了仅使用语言监督的语义分词器。

**💡 创新点**

创新点包括：①将自回归与扩散分别用于知识学习与像素渲染；②引入Semantic-LM，取消图像重建损失，仅用语言监督训练分词器；③通过阶段级缩放律实现计算资源的最优分配。

**🔧 技术方法**

使用的技术包括自回归预测、IBQ代码书、冻结视觉分词器、文本条件自回归网络、像素扩散流匹配、计算缩放律拟合，以及在H100 GPU上进行的高效训练。

**📊 数据集**

实验使用了公开的大规模图文对数据集（如LAION等）进行自回归预训练和扩散后训练，具体数据集未在文中列出。

**📈 对比分析**

与传统从零开始训练的像素扩散模型和Z-Image-Base基准进行对比，使用GenEval和DPG-Bench评估；3B模型仅耗6,600 H100 GPU小时即可取得GenEval 0.86、DPG-Bench 87.78，计算量远低于基线。

**⚠️ 局限性**

局限性包括：验证仅在单一3B规模上完成，缺乏对更大模型或不同数据量的泛化评估；依赖冻结的分词器和自回归骨干可能限制表示能力；对其他视觉任务的适用性尚待进一步验证。

---

## 744. RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement

**arXiv ID:** 2610.37602 | [PDF](https://arxiv.org/pdf/2610.37602v1)

**作者:** Ziyu Cao `[一作]` (Linköping University), Kailai Li `[通讯]` (Linköping University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RESPLE2，一个基于非均匀B样条的递归连续时间多传感器里程计框架，能够在LiDAR、相机、IMU及腿关节测量的多模态环境下进行实时6-DoF位姿估计，并提供轻量级全局优化模块。

**💡 创新点**

创新点包括：①利用knot-clamping边界性质实现控制点在运动轨迹上的精确预测；②自适应放置非均匀节点，既能捕捉高动态运动，又能利用传感器信息避免几何退化；③在IEKF框架下实现多传感器连续时间观测融合；④首次在B样条估计中加入直接全局优化，利用循环闭环约束直接修正控制点。

**🔧 技术方法**

使用技术：非均匀B样条（分离平移与旋转表示）、迭代扩展卡尔曼滤波、IMU积分预测、连续时间观测模型、VoxelMap局部地图、iBTC循环闭环检测、Newton方法进行姿态重置、Levenberg–Marquardt全局优化。

**📊 数据集**

评估数据集：公开的GrandTour、JEPLO；实验集PPS Tracked Vehicle、HOPPER、HANDI；涵盖四足、履带、混合机动、手持平台等多种机器人与不同传感器配置。

**📈 对比分析**

通过与Coco‑LIC、FAST‑LIVO2、CTE‑MLO、Point‑LIO、Leg‑KILO等现有方法在APE/RPE指标下比较，RESPLE2在大多数序列上达到或优于基准，特别在高动态、IMU饱和以及视觉失效情况下保持鲁棒性；平均每帧计算时间保持在实时预算内。

**⚠️ 局限性**

局限性：①循环闭环检测仍依赖iBTC，受视觉/激光匹配质量影响；②对极端视觉失效或动态障碍物的处理尚不完善；③全局优化在大规模轨迹上仍会产生一定延迟；④对硬件资源（如GPU）需求较高；⑤当前实现主要针对固定硬件平台，迁移到资源受限设备需进一步优化。

---

## 745. EgoThumb: A Vision-Driven Supernumerary Robotic Thumb for Autonomous Grasp and Holding Assistance

**arXiv ID:** 2610.37582 | [PDF](https://arxiv.org/pdf/2610.37582v1)

**作者:** Jianyu Hou `[一作]` (University of California San Diego), Hao Su `[通讯]` (Sudo AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款开源的、视觉驱动的超额人手拇指(EgoThumb)，可在不需手部运动触发的情况下，自动识别握持时机并协助人手完成复杂的“握持-操作”任务。

**💡 创新点**

创新点在于将前臂驱动的双绞线结构与掌面鱼眼摄像头结合，利用视觉模仿学习实现对人手意图的隐式解码，同时实现了轻量化、低遥感、开源的设计。

**🔧 技术方法**

使用了前馈的 Diffusion Policy 框架、CLIP 预训练 ViT 视觉编码器、双绞线主动关节驱动、低延迟同步中间件以及自定义脚部遥操作接口进行数据采集。

**📊 数据集**

数据集由 4 项真实任务（拾取方块、拆卸瓶盖、搅茶、传球）共 105 条演示组成，演示来自两位不同手型用户，涵盖多种环境与背景，全部开源收集。

**📈 对比分析**

通过与演示者及新手用户的对比实验，平均成功率在演示者上为 0.90、用户 1/2 为 0.925/0.875；在已见与全新环境下，平均成功率分别为 0.80/0.73，表现出显著的跨用户与跨环境泛化能力；在更大初始/目标区域的鲁棒性测试中，拾取与拆卸任务保持 >0.73，搅茶与传球则下降至 0.53/0.17。

**⚠️ 局限性**

局限性包括对长时序、空间跨度大、视觉干扰强的任务性能下降；高速度手部运动导致模糊不评估；掌面摄像头突出影响佩戴舒适度；仅在一致手部接近角度下测试，缺乏多角度鲁棒性；评估指标仅为二进制成功率，缺乏细粒度协调质量测量。

---

## 746. As the Story Unfolds: Watching a Film and Identifying Characters as a Human Does

**arXiv ID:** 2610.37579 | [PDF](https://arxiv.org/pdf/2610.37579v1)

**作者:** Zhongrui Gui `[一作]` (University of Oxford), Andrew Zisserman `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了从电影对白中识别角色名称并构建在线角色库的任务，构建了MCN数据集并实现多模态LLM解决方案。

**💡 创新点**

创新点包括：①把角色命名任务从依赖外部演员列表改为仅用影片内视觉、对白和时序线索；②创建大型多模态任务数据集MCN；③基于色彩视觉提示的多模态LLM，采用两阶段SFT+GRPO微调和颜色置换正则化；④在线角色库构建流水线和误差纠正机制。

**🔧 技术方法**

技术：多模态大语言模型（Qwen3VL）、视觉提示（彩色框）、语音识别+说话人分离、TalkNet同步、Parakeet转录、R1式SFT+GRPO强化学习、聚类/嵌入更新、错误纠正。

**📊 数据集**

数据集：MovieNet + Movie101 影片，构成MCN（Character Naming 23,792题、First Occurrence 20 影片）以及自建的未见影片 102 条，包含角色名、轨迹、字幕、时间戳。

**📈 对比分析**

与基线对比：对比随机、开源 Qwen3VL-8B-Instruct、Qwen3.5-9B 等，SFT+GRPO 的 4B模型在整体/已知准确率分别达72.76% /81.06%，接近 Gemini-3.1-Pro-Preview（82.47%/89.69%）。在在线角色库构建中，mTIoU_v 84.90% /54.40%、PCR@5 63% /54.54% 等，明显优于 Gemini 版本。

**⚠️ 局限性**

限制：仍受制于转录/说话人检测噪声、动作复杂/多人场景、短暂指向等细节失误；对开放域知识依赖仍存在，且模型在高人物密度影片上表现下降；需要更鲁棒的视觉/语音特征与多模态推理。

---

## 747. The Noise Plane: A Diffusion Bridge Across Exponent and Dispersion

**arXiv ID:** 2610.37576 | [PDF](https://arxiv.org/pdf/2610.37576v1)

**作者:** Xuran Hu `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 Tweedie 家族的测量指数扩散桥 p-Bridge，可在单一网络中处理多种信号相关噪声。

**💡 创新点**

创新点在于把噪声方差–强度关系（指数 p）与测量累积量 ν 分离，通过指数条件化实现对多种噪声模型的统一学习，并利用复合泊松-伽马构造在中间指数上采样。

**🔧 技术方法**

采用测量指数扩散桥、单网络指数条件化、反向均值更新、单位偏差损失、时序嵌入 UNet 与自回归/一致性正则化等技术。

**📊 数据集**

使用 BSDS500、DIV2K 作为训练集，合成噪声实验覆盖指数 {1,1.25,1.5,1.75,2}，并在 FMD 荧光显微镜和多传感 SAR 数据上进行真实评估。

**📈 对比分析**

与多种基准（BM3D、Noise2VST、VIRNet、I²SB、DDRM 等）对比，p-Bridge 在合成图像、FMD 与 SAR 场景下的 PSNR/SSIM 与 ℳ/RIS 指标均优于或与最佳专用模型持平。

**⚠️ 局限性**

局限包括对指数估计误差的敏感性、对极低噪声或极大噪声场景的适配需要进一步改进，以及对测量索引 ν 的校准依赖训练数据分布。

---

## 748. TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information

**arXiv ID:** 2610.37571 | [PDF](https://arxiv.org/pdf/2610.37571v1)

**作者:** Chanwoo Park `[一作]` (Massachusetts Institute Of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为TeamGym的多智能体决策实验环境，用于构造并评估基于语言模型的团队决策问题；通过定义“关键信息单元”（PIU）并控制其分布、沟通结构和令牌预算，研究团队如何检索、共享与整合分散信息。

**💡 创新点**

创新点在于①引入PIU概念，提供可控的最小决策相关信息集合；②通过实验拆分沟通缺口与检索缺口，量化信息分布与沟通对团队成功率的影响；③利用强化学习（GRPO）在此环境中训练协调者与参与者策略，观察到自发的协调行为。

**🔧 技术方法**

技术包括：大语言模型（Qwen2.5-7B-Instruct、GPT-5.5等）、基于留一检验的PIU筛选、通信结构模拟（循环、顺序、并行、协调者控制等）、强化学习框架（GRPO）以及对最终决策的LLM评估。

**📊 数据集**

使用四大领域的公开数据集：法律（CUAD、MAUD、eCFR Casebook、MSLR、Multi-LexSum、SARA）、金融（FinDVer、FinLongDocQA）、医学（MLCR、EvidenceBench）和软件工程（SWE Atlas–Codebase QnA、DeepSWE）。

**📈 对比分析**

比较方法为单智能体全部PIU vs 多智能体仅PIU分布 vs 多智能体需检索PIU三种评估条件。实验结果显示：通信缺口降低22.9%成功率，检索缺口额外下降19.9%；不同沟通结构对成功率影响随PIU分布变化；经过GRPO训练后，最终决策质量提升（如CUAD验证从≈12%提升至≈22%），并出现自发的沟通与协调行为。

**⚠️ 局限性**

局限性包括：实验仅在一次训练运行下评估，模型覆盖有限；仅考虑合作收益与信息检索，没有研究对抗或部分对齐目标；未结合人类-AI团队交互；缺乏对检索成本的显式建模。

---

## 749. Learning Visuomotor Quadrotor Flight Through Dynamic Narrow Gaps via Differentiable Simulation

**arXiv ID:** 2610.37557 | [PDF](https://arxiv.org/pdf/2610.37557v1)

**作者:** Riccardo Bianco `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一套可直接利用视觉和惯性测量的控制策略，能够让四旋翼在真实环境中以厘米级精度通过静态和动态窄缝。

**💡 创新点**

创新点在于：① 通过可微分的移动缝框架把稀疏的通过事件转化为稠密的路径梯度；② 用路径梯度（BPTT）而非传统的基于分数函数的强化学习，在同等仿真条件下实现极低的梯度方差；③ 采用端到端从随机初始化开始学习，无需演示、教师或规划器；④ 在视觉特征上使用几何工程化的缝角投影，保持了观测的可微性。

**🔧 技术方法**

主要技术包括可微分仿真、逆向传播（BPTT）路径梯度、基于视觉特征的观察输入、对齐/碰撞/对齐平滑损失以及多种训练增强技术（域随机化、延迟控制等）。

**📊 数据集**

使用自定义的仿真环境：随机化的静态缝隙与通过摆动杆驱动的平面摆动缝隙；在高精度仿真和真实硬件上验证；未使用公开数据集。

**📈 对比分析**

与基准PPO进行比较：在1,000个平行环境下，BPTT 仅需19×更少的转移样本、45.5×更少的训练时间，却在模拟中实现99.3%成功率；在真实飞行中达80%成功率。相比之下，PPO在相同任务上需要数倍到数十倍的样本与计算量。

**⚠️ 局限性**

局限性包括：① 训练依赖仿真中的精确几何和碰撞信息；② 视觉输入为手工特征而非原始图像，限制了鲁棒性；③ 仅测试了平面摆动缝隙，未验证复杂或非周期的缝隙运动；④ 10次真实飞行样本统计不充分，失败原因尚未完全分离；⑤ BPTT的显存与计算开销随时间步长与仿真精度放大。

---

## 750. MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control

**arXiv ID:** 2610.37647 | [PDF](https://arxiv.org/pdf/2610.37647v1)

**作者:** Athanasios Glentis `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大规模语言模型预训练中Muon优化器的权重范数控制，提出通过Gram–Schmidt投影去除更新的径向分量，从而设计了MuongS优化器。

**💡 创新点**

创新点在于：①用累计有效学习率来衡量训练过程中的探索能力；②证明在不强制固定范数的前提下，MuongS可实现O(√T)的累计有效学习率；③通过理论与实验表明该方法在长时间训练中能维持更充分的探索。

**🔧 技术方法**

主要技术包括：Muon's orthogonalization、Gram–Schmidt径向分量剔除、累计有效学习率分析、尺度不变目标的光滑性假设下的收敛分析。

**📊 数据集**

使用的公开数据集：FineWeb（100B-token子集）和Ultra‑FineWeb，分别用于训练160M/400M GPT‑style模型和135M/1B参数的MoE模型。

**📈 对比分析**

对比方法：MuongS与Muon、MuonW、MuonH以及AdamW。实验显示在160M和400M模型上，MuongS在长时间训练（高至8倍Chinchilla预算）时能取得比MuonH低约15%的验证困惑度，等价训练量上估算的加速率为5–15%；在MoE模型上也表现出更低的验证困惑度。

**⚠️ 局限性**

局限性：累计有效学习率的提升仅为必要条件，无法保证最终最优收敛；理论分析给出最坏情况下的下界，实际最优范数增长速率仍是开放问题；实验仅覆盖特定模型规模和数据集，泛化性待进一步验证。

---

## 751. Perceiving and Predicting Dynamic Sports: Spatiotemporal Intelligence in Vision Models for Egocentric Tennis Video

**arXiv ID:** 2610.37619 | [PDF](https://arxiv.org/pdf/2610.37619v1)

**作者:** Tudor Braicu `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过收集同步的自我视角和外部视角网球视频，构建数据集并训练多种视觉模型进行空间理解与时空预测，随后将预测结果用于条件扩散模型生成实时全身动作。

**💡 创新点**

创新点在于：①首次在动态自我视角网球视频上系统评估基础视觉模型的空间理解与预测能力；②发现补充深度信息能显著提升3D球定位；③证明更长的视觉历史对预测无显著提升；④提出基于视觉预测的实时全身动作生成框架。

**🔧 技术方法**

使用的技术包括：自监督/监督图像编码器（DINOv3、SigLIP 2、DA3、SAM）、自监督视频编码器（V-JEPA 2）、视觉‑语言模型（Qwen3‑VL‑4B、LFM2.5‑VL‑3B、PLM‑3B、LFM2.5‑VL‑450M），以及条件扩散模型（MDM）和LoRA参数高效微调。

**📊 数据集**

数据集：85个网球回合，53155帧自我视角视频，5个同步外部视角视频，标注3D球轨迹、2D球/玩家位置、击球类型、接触时间等，共计约30.6分钟训练数据。

**📈 对比分析**

比较方法：在同一训练/验证/测试划分上，使用统一的预测头对13种视觉模型进行7个任务的多任务训练；结果显示图像编码器和VLM在定位和预测任务上表现相近，V-JEPA在球预测上最差但玩家预测最优；补充深度信息可将球定位误差减至0.21 m；更长视觉历史对预测提升有限；基于LFM2.5‑VL‑450M的条件扩散能在实时条件下生成接近真实运动的全身动作。

**⚠️ 局限性**

局限性：①对球的3D定位和运动预测仍受限于视角混合导致的模糊/遮挡；②冻结模型特征时球定位误差大幅上升，说明预训练特征不足；③基于标注的运动记录本身存在误差，限制了运动生成的精度；④实验仅在网球场景，尚未验证到其他高速动态任务。

---

## 752. ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning

**arXiv ID:** 2610.37666 | [PDF](https://arxiv.org/pdf/2610.37666v1)

**作者:** Chaehyun Kim `[一作]` (NVIDIA), Fu-En Yang `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ThinkWorld 框架，学习可迁移的 Chain‑of‑Dynamics（CoD）并通过世界想象强化，实现在人类视频与机器人轨迹中统一的动作推理；

**💡 创新点**

创新点在于将动作思考转化为可跨人机共享的离散运动序列（CoD），并用未来想象奖励对其进行强化学习，从而在无动作标签的视频中直接获得监督；

**🔧 技术方法**

技术包括基于 Qwen3‑VL‑2B 的视觉‑语言‑动作 VLM，FAST 运动分词器，DINO 视觉特征空间的未来预测，GRPO 强化学习以及基于未来相似度的世界想象奖励；

**📊 数据集**

使用了 EgoDex、EgoVerse、MolmoAct2（DROID/BimanualYAM）、Bridge/RT‑1、BC‑Z、RoboCOIN、LIBERO、LIBERO‑PRO 以及真实 YAM 机器人实验的视频数据；

**📈 对比分析**

与 Diffusion Policy、MolmoAct2、LaWAM 等方法在 ID/OOD 机器人任务、LIBERO 与 LIBERO‑PRO 基准以及炉子/抽屉任务上进行对比，ThinkWorld 在 OOD 和多环境下保持更高成功率（LIBERO 97.3%，LIBERO‑PRO 55.6%），显著优于基线；

**⚠️ 局限性**

局限性包括依赖 VLM 预训练的视觉语言能力，对未来帧的预测精度受限，无法直接对未知动作执行；对高维连续控制仍需动作专家映射；在人类视频中需要手部轨迹或姿态估计，存在噪声和缺失问题。

---

## 753. An EVM-Cosmos Bridge for Atomic, Semantic-Preserving Transfer of Environmental Tokens

**arXiv ID:** 2610.37663 | [PDF](https://arxiv.org/pdf/2610.37663v1)

**作者:** Sebastian Pandolfi `[一作]` (Universidad de la Republica), Laura González `[通讯]` (Universidad de la Republica)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个跨EVM–Cosmos的桥接系统，实现环境指标代币的原子且语义保持的转移。

**💡 创新点**

创新点包括：HTLC改造的单方转移原子性、自动回调机制消除超时竞争、通过keccak256生成确定性指示器身份防止语义混合、跨链语义一致性校验以及多类代币的镜像注册。

**🔧 技术方法**

使用技术包括：ERC‑1155 与 CosmWasm 合约、Axelar GMP 作为消息中继、HTLC 与自动回调、keccak256 内容地址化身份、双链注册表、Bounty 机制、TLC/TLA+ 机器检查和 PoS 验证者共识。

**📊 数据集**

评估数据集为 50 份从 Polygon 到 Neutron（以及 2 份反向）的实际主网转移记录，涉及 CO₂ 等环境指标代币。

**📈 对比分析**

通过与现有桥（SATP、Gravity、Wormhole、LayerZero、CCIP、Axelar ITS）比较，展示其唯一实现的功能；性能指标为：97 s 的中位延迟、最大 164 s、平均 gas ≈639k、成本约 1–2 USD、语义验证开销 <0.4 % 的消息成本。

**⚠️ 局限性**

局限性包括：回调失败时原子性条件依赖观察者；需要显式 32 字节秘密并在到期前揭示；仅在 Axelar 的 EVM–Cosmos 对上验证，未覆盖其他链对；实现层未做形式验证；注册受制于所有者门控；对恶意发行者的内容地址化不完整；以及对中继费用合规性的依赖。

---

## 754. Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness

**arXiv ID:** 2610.37657 | [PDF](https://arxiv.org/pdf/2610.37657v1)

**作者:** Igor Shilov `[一作]`, Yves-Alexandre de Montjoye `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于每头注意力门控的“Heads of Security”机制，用于在大型语言模型中实现指令–数据分离，以防止提示注入攻击。

**💡 创新点**

通过在每个注意力头中学习的通道级门控，动态重分配对受信任与不受信任令牌的关注度，实现细粒度的指令跟随与数据读取分离。

**🔧 技术方法**

对Transformer注意力权重加入可学习的通道偏置门控，并使用SecAlign++的DPO训练范式对Llama-3.3-70B和GPT-OSS-20B进行微调。

**📊 数据集**

采用与SecAlign相同的Cleaned-Alpaca偏好对数据集，其中每个样本包含用户指令和注入指令的对。

**📈 对比分析**

在黑盒RL‑Hammer和白盒Claudini攻击、标准多项选择与Agentic注入基准上与SecAlign++和ASIDE对比，HoS攻击成功率低至10%/1%，保持约99%原始效能，并在Agentic基准中匹配顶尖水平。

**⚠️ 局限性**

仅在标准密集注意力架构下有效；未决定信任来源，仅提供信任边界；训练数据局限可能导致离谱场景下性能下降；并且对线性注意力模型不可直接适用。

---

## 755. Partially Edited Audio: Generation, Defense, and Open Problems

**arXiv ID:** 2610.37640 | [PDF](https://arxiv.org/pdf/2610.37640v1)

**作者:** Lin Zhang `[一作]` (Independent Researcher), Sanjeev Khudanpur `[通讯]` (Johns Hopkins University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对“部分编辑音频”从生成与防御两大视角做系统综述，阐明其定义、特征、编辑目标与方法、评估指标以及防御技术（检测、定位、说话人追踪、可解释性），并提出未来研究方向。

**💡 创新点**

1) 将生成与防御视为协同进化的两面，强调编辑意图的可解释性与责任追溯；2) 对编辑目标与方法进行统一分类，覆盖语言内容、韵律、情感、背景、跨模态等多维属性；3) 综合已有评测基准与工具包，搭建跨领域、跨语言的评估框架；4) 指出目前研究聚焦单语音、内容编辑的局限，并提出主动水印、跨模态、长时序等开放问题。

**🔧 技术方法**

编辑技术：剪接与复制、生成后剪接、上下文感知填充、基于指令的全局编辑；防御技术：深度学习检测器（CNN、LSTM、SSL、LLM）、段级定位（帧级特征、边界不一致）、说话人/编辑器追踪、可解释性分析。评估技术：WER、UTMOS、说话人相似度、边界平滑度、真实性评分。

**📊 数据集**

编辑与评测数据集：RealEdit、MMAE、AudioMix、ADD、PartialSpoof、HAD、Audio-Visual多模态数据、LLM驱动的部分伪造数据等；防御数据集：PartialSpoof、HAD、ADD、Audio-Visual Forgeries、长时序噪声集；工具包：WeDefense、DeepFense。

**📈 对比分析**

对比方法主要以基准数据集上的检测/定位精度、自然度评分、说话人一致性等指标进行；综述显示：传统剪接型伪造易被检测，context‑aware infilling 以及指令编辑的伪造更难以捕捉；目前防御模型在单语、单环境下表现良好，但跨域、跨编辑风格、长时序、跨模态任务仍存在显著性能下降。

**⚠️ 局限性**

主要局限：1）防御研究仍聚焦单语音内容编辑，缺乏对情感、背景、音乐、一般声效以及跨模态编辑的系统评估；2）评测基准多为单语、干净、短编辑环境，缺乏真实噪声、回声、多说话人、多编辑点的情境；3）跨域泛化与编辑风格迁移能力不足，现有模型对新编辑方式或不同录音环境的鲁棒性差；4）缺乏主动责任追溯机制（水印、可追溯编辑轨迹）与多模态同步一致性评估；5）缺少统一的多任务、跨语言、跨领域的评测框架与公开工具。

---

## 756. Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions

**arXiv ID:** 2610.37624 | [PDF](https://arxiv.org/pdf/2610.37624v1)

**作者:** Anouar Imel `[一作]` (University of Electronic Science and Technology of China), Khalil El Khammar `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 CrossProfile 诊断基准并评估 LLM 组讨论的记忆压缩策略（稀有度、相关性等）

**💡 创新点**

首次系统交叉控制事实所有权、检测记忆压缩策略与决策模型的偏差，并揭示稀有度与相关性策略的局限

**🔧 技术方法**

使用基于令牌预算的复制与摘要记忆策略、相关性过滤器、Oracle 相关性与随机抽取等方法

**📊 数据集**

利用 Qwen2.5-7B、Llama-3.1-8B、DeepSeek 等大模型在 HiddenBench 与 CrossProfile 60 模板（480 任务）上的表现

**📈 对比分析**

通过准确率对比显示 R1（稀有度优先）在经典隐藏配置下优于随机，R1-rel 在相关性优先下往往降低准确率，且决策模型受到“社会证明”影响，平均准确率与记忆保留的证据平衡相关

**⚠️ 局限性**

主要局限在于记忆策略无法解决证据稀有度与相关性冲突，决策模型对来源标签过度信任，自然讨论中证据往往在讨论前已丢失，导致记忆无显著提升

---

## 757. SLICE-CiM: Selective Layer-wise Inference Control Engine with Confidence-Based Early Exit for SRAM based Digital Compute-in-Memory

**arXiv ID:** 2610.37601 | [PDF](https://arxiv.org/pdf/2610.37601v1)

**作者:** Pratham Sharma `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 65 nm CMOS 下设计了一套完整的 SRAM 基数字 Compute‑in‑Memory（DCIM）宏，并配套实现了一个统一的外围控制框架（Flush‑Merge Enable、Confidence‑Based Early‑Exit、Precision Register/Decoder 与 Cycle Controller），实现了层级可变精度、早停和精度切换无停顿的高效推理。

**💡 创新点**

创新点在于：1）将精度切换、累加清除、词线重配置等四项关键操作合并到一次时钟周期内的 Flush‑Merge Enable；2）通过 MSB‑first 顺序和动态阈值实现置信度早停，显著减少累加深度；3）使用精度寄存器与解码器实现按需激活词线，消除切片过激活；4）提出训练‑无关的混合精度分配指标（CoV+稀疏度），一次前向推理即可完成精度分配。

**🔧 技术方法**

技术包括：10‑T NOR SRAM 细胞、位串 MAC、混合精度量化、置信度早停、精度可变词线控制、全定时 RTL 周期控制、TSMC 65 nm CMOS 细化流程、Gate‑level VCD 动态功耗分析与面积评估。

**📊 数据集**

使用 MobileNetV2 网络在 CIFAR‑100 数据集上进行推理验证，并通过 1000 张随机训练图像进行阈值与精度分配校准。

**📈 对比分析**

与现有混合精度方法（HAWQ、HAQ 等）和多种 65 nm CIM 方案对比，系统在 2‑bit/8‑bit 精度、0.8 V、335 MHz 时实现 454.91 TOPS/W，较 8‑bit/8‑bit、1.2 V 基线提升 13.33×，外围功耗仅占系统总功耗 0.24%。

**⚠️ 局限性**

主要局限在于 Transformer 等模型对 ReLU 产生的稀疏性不足，且 GELU/SiLU 激活可能出现超出 |x_i|≤1 的异常值，需开发自适应阈值和异常值感知量化策略。

---

## 758. MoI-Bench: A Benchmark for Image Generation and Editing on Mobile Devices

**arXiv ID:** 2610.37599 | [PDF](https://arxiv.org/pdf/2610.37599v1)

**作者:** Kailai Feng `[一作]` (University College London), Jagmohan Chauhan `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MoI-Bench与MoI-Eval两套框架，用于评估移动端图像生成与编辑模型，覆盖64个子任务、1150个样本，并通过操作、目标和评价标准三维结构生成能力画像。

**💡 创新点**

创新点在于①将用户需求转化为操作×目标任务矩阵，实现细粒度的任务覆盖；②构造任务特定的评价准则立方体，分离指令遵循、视觉质量与细节保持；③引入质量-可部署度二维能力图，帮助区分高质量与高效率的模型。

**🔧 技术方法**

技术包括大规模扩散模型推理、文本与视觉VLM（GPT‑4o、Gemini‑3.1‑Pro）自动问答评分、基于量化与精度调优的移动部署（Core ML/ONNX+QNN HTP），以及多维度指标聚合与可视化。

**📊 数据集**

使用了1,150条多样化样本（文本生成、单图像、多图像引用），每条样本来自公开图像库（Pexels、Wikimedia Commons），并根据用户调查确定目标与操作优先级。

**📈 对比分析**

通过对19个模型（开放权重与闭源模型）进行统一评价和iOS/Android部署测量，结果显示：闭源模型均达Q4，FLUX.2 [klein]为最佳开放模型；在4B以下模型中DreamLite与Ovis‑U1分别位列Q2；部署层面，DreamLite/D4可达D3，移动端模型整体仍未达到Q3或D4。

**⚠️ 局限性**

局限性包括①评估仅关注单步任务，未覆盖多轮交互；②大多数模型在文本与多图像编辑上表现差强人意；③部署结果受平台特定优化影响，未能统一低延迟与低内存的最佳方案；④数据集规模相对有限，可能不足以覆盖所有实际场景。

---

## 759. Understanding the limits of latent world models

**arXiv ID:** 2610.37588 | [PDF](https://arxiv.org/pdf/2610.37588v1)

**作者:** Rodrigo Gonzalez Laiz `[一作]` (Helmholtz Munich), Steffen Schneider `[通讯]` (Helmholtz Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估 latent world models 的“faithful”特性，提出并使用 latent rollout consistency (LRC) 指标来量化模型在动作驱动下的 latent 动态与真实环境动态的一致性。

**💡 创新点**

创新点在于：①用 equivariance 条件定义 faithful latent world model；②引入 LRC 作为直接评估 latent 预测准确度的新指标；③通过 controlled interventions 揭示 pixel 观测下速度估计不足和非线性观测映射导致的两大失败模式。

**🔧 技术方法**

使用的技术包括：自监督编码器 + 预测动态模型；LRC 计算（R² 统计）；状态解码器（两层 MLP）对位置/速度进行 probing；MPC 控制器（TD-MPC2、DreamerV3、LeWM 后置控制）；对环境做逆向或混合观测的实验。

**📊 数据集**

数据集：DeepMind Control Suite 的三个连续控制环境——Pendulum、CartPole、Reacher，使用状态向量和渲染 RGB 图像两种输入形式。

**📈 对比分析**

比较方法：在相同的评估集上同时报告 LRC、任务奖励、位置/速度解码误差；发现即使 LRC 在 30 步时低于 10%，模型仍能获得高奖励；pixel 观测下 LRC 快速衰减且接近 persistence baseline；对比状态输入模型，pixel 输入模型 LRC 明显更差，说明预测不准确但奖励仍可达标。

**⚠️ 局限性**

局限性：实验仅覆盖简单、确定性、低维连续控制环境；未考察随机/部分可观测/高维或离散系统；只评估了三种主流模型，结果可能不适用于其他更复杂的 latent world model；对 LRC 的解释和阈值仍需进一步理论和实证支持。

---

## 760. Timestral

**arXiv ID:** 2610.37585 | [PDF](https://arxiv.org/pdf/2610.37585v1)

**作者:** Adrien Petralia `[一作]`, Zsófia Csákány `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了四个参数从5M到209M的时间序列基础模型 Timestral，支持单变量、多变量和包含已知未来协变量的概率预测。

**💡 创新点**

创新点包括：1）Decoupled Horizon Forecasting（DHF），将预测标记与因果上下文分离，允许在不屏蔽上下文的情况下训练更长的预测区间；2）基于角色的协变量注意力，能在预测时使用已知未来协变量而不泄漏未来目标值；3）两阶段训练策略，先在大规模合成数据上预训练，再在真实+合成混合数据上微调，并使用更长的上下文。

**🔧 技术方法**

使用的技术：Transformer backbone（Pre-norm、RMSNorm、SwiGLU）、块级变体注意力（时间+变量注意力）、分块预测标记、累计软增量的分位数头、无位置编码、旋转位置嵌入、对数归一化及逆归一化、缺失值掩码和对称平均。

**📊 数据集**

训练数据主要为合成时间序列（包含趋势、季节、突发事件、级联变化等），随后在包含真实数据的混合语料库上进行微调；评估使用公开基准 GIFT‑Eval、TIME、fev‑bench（以及 IND‑Eval）来衡量 MAE、MASE、CRPS 等指标。

**📈 对比分析**

在 GIFT‑Eval、TIME 和 fev‑bench 上，209M 版本在所有度量上均位居榜首；123M 版位列第二，且优于所有非 Timestral 模型。速度方面，单个 1,024 步的预测可在 8 CPU 线程下 17–113 ms 内完成，GPU H100 可实现每秒 3,650–18,770 条序列的推理。更小的 28M、5M 版本在速度与准确率上表现出良好的折中。

**⚠️ 局限性**

局限性：1）模型仍受参数规模限制，进一步提升准确率需要更大模型；2）预训练高度依赖合成数据，合成与真实分布差异可能导致迁移瓶颈；3）在极大上下文（>32k 步）和多变量场景下推理延迟显著上升；4）当前仅支持单变量和多变量的概率预测，尚未涵盖跨域/跨任务的自适应能力。

---

## 761. Zero-Shot Multimodal Event Spotting in Soccer Broadcasts

**arXiv ID:** 2610.37583 | [PDF](https://arxiv.org/pdf/2610.37583v1)

**作者:** Sadia Kamal `[一作]` (University of Maryland Baltimore County), Tony Braskich `[通讯]` (Comcast)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在完整的足球比赛转播中，使用通用多模态大语言模型实现了事件检测、分类、时序定位以及基于证据的兴奋度打分。

**💡 创新点**

①零样本（zero-shot）框架，无需领域特定参数微调；②利用稀疏视觉“故事板”与对齐解说文本共同输入，显著降低推理成本；③在检测结果上加入基于多模态证据的兴奋度评分和自然语言解释，提升可解释性。

**🔧 技术方法**

采用多模态大语言模型（如 GPT‑5.6、Qwen3）、手工/自动优化提示、稀疏帧采样、故事板构造、叙事文本对齐、时序合并与聚类、以及基于事件重要性与视觉/文本强度的加权兴奋度公式。

**📊 数据集**

三个公开 YouTube 足球比赛（共 130 条事件，六类：进球、射门、失误、角球、开球、黄牌），对每场比赛手工注释时间戳。

**📈 对比分析**

与两种监督基线（CALF、NetVLAD++）在同一六类评估协议下进行跨数据集对比。最佳零样本配置（GPT‑5.5+故事板）在 5 秒容差下取得 F1≈72%，召回率≈82%，比基线的 F1≈39% 和召回率≈66% 更优；同时推理成本降低约 72%（单帧 64.69$ → 18.01$）。

**⚠️ 局限性**

①仅评估 3 篝火，难以推断泛化到不同联赛/播出风格；②稀疏帧采样可能漏掉极短时刻特征；③重放/回顾内容仍可能导致误检，提示式广播识别并非完美；④兴奋度评分完全基于模型，缺少人工验证。

---

## 762. Collective Dynamics of Self-Reinforcement: How Cross-Task Coupling Drives Policy Collapse in Unsupervised RLVR

**arXiv ID:** 2610.37553 | [PDF](https://arxiv.org/pdf/2610.37553v1)

**作者:** Peize Ding `[一作]` `[通讯]` (Columbia University), Peize Ding (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文探讨了无监督强化学习中的自我强化动态，分析了数据集大小对模型崩溃的影响，提出了一个集体不稳定性的机制。

**💡 创新点**

创新点在于揭示了数据集大小依赖的崩溃现象是由于共享参数空间中的集体不稳定性所驱动，并提出了一个临界数据集大小的公式。

**🔧 技术方法**

使用了随机模拟和统计物理的方法来验证理论，结合了神经切线核的线性化理论。

**📊 数据集**

使用了多种问题的模拟数据集，特别是关注于不同数量的问题对模型崩溃的影响。

**📈 对比分析**

通过随机模拟验证了理论预测，发现当问题数量超过临界值时，模型崩溃的概率急剧上升，且崩溃与数据集大小、共享耦合强度和初始信心有关。

**⚠️ 局限性**

限制在于模型简化了真实语言模型的复杂性，未考虑训练过程中共享方向的变化，且在预训练语言模型上的测试尚未进行。

---

## 763. Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis

**arXiv ID:** 2610.37648 | [PDF](https://arxiv.org/pdf/2610.37648v1)

**作者:** Andrew Szot `[一作]` (Apple), Alexander Toshev `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一套可控的多步骤工具使用环境合成管道，利用LLM自动生成应用、数据库模式、Python API、任务指令、验证器和Oracle解码器，并构建可执行Gymnasium环境供训练。

**💡 创新点**

①对工具、多任务与数据配置提供显式可控性；②提出基于数据组合的缩放律，预测不同组合下的下游性能；③通过大规模实验验证并给出最佳数据分配方案；④实现对大规模训练预算的性能预测。

**🔧 技术方法**

LLM驱动的代码合成（Gemini‑3.1 Pro）、Python API与数据库架构生成、任务与验证器合成、RL/监督训练（SFT）使用GLM‑5.2与Qwen3.5、任务组合控制与缩放律建模。

**📊 数据集**

自行合成的458,752个任务，包含64个应用、10,472个API；下游评估使用AppWorld、BFCLv3多轮、ASTRA‑Bench及自建诊断集。

**📈 对比分析**

与Nemotron、ToolACE/Alpaca、SimSAT、EnvScaler/Factory等基线对比，使用下游成功率评估；在AppWorld Test‑C、BFCL等上平均提升30%+，最优组合在不同模型尺寸上达到约70%+成功率，缩放律误差显著低于仅任务计数模型。

**⚠️ 局限性**

未模拟用户交互、缺乏长文本检索情境、合成数据缺乏语义深度、缩放律仅在单一合成管线验证，未保证跨数据集或训练设置的泛化。

---

## 764. When Should We Intervene at Test Time?

**arXiv ID:** 2610.37658 | [PDF](https://arxiv.org/pdf/2610.37658v1)

**作者:** Jiahao Zhang `[一作]` (Zhengzhou University of Aeronautics), Yu Chen `[通讯]` (Zhengzhou University of Aeronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 PORTAL 的控制器，利用稀疏标签信息在测试时动态选择何时使用、部署哪一种适配干预，并在无价值时停止查询。

**💡 创新点**

创新点包括：① 通过共享反馈一次性评估所有候选干预的相对表现；② 在候选间使用贝叶斯推断和期望信息价值来决定查询策略；③ 在确定性、可重现的标签预算下实现自适应停止与周期性验证，从而在不显著增加标签成本的前提下避免错误干预。

**🔧 技术方法**

核心技术：贝叶斯更新（Dirichlet 先验）、期望信息价值估计、共享误差协方差、可行性（reachability）分析、基于阈值的停止策略、针对候选学习的增量训练。

**📊 数据集**

实验使用了 PACS、CIFAR‑10‑C / CIFAR‑100‑C、ImageNet‑C、CLIP‑ViT-L/14 与 PACS、OfficeHome 等数据集，涵盖了域漂移、图像噪声、天气和压缩失真等多种场景。

**📈 对比分析**

与固定干预（BN、Flip、BN+Flip）、平均 logits 以及多种主动查询策略（ERM、Top‑KL、Disagreement ERM）对比，PORTAL 在多种顺序与污染强度下显著提升平均准确率，标签使用量仅为传统方法的 10%–30% 之内；在完整的 ImageNet‑C 流中，将错误率从 50.14% 降至 49.55%，并实现了 73.37% 的干预收益保留。

**⚠️ 局限性**

局限性：① 依赖“传输假设”（不同子集共享同一类别分布），若分布漂移严重会导致评估失真；② 需要一定量的标签来估计各 strata 的概率，标签分布不均或稀缺时性能下降；③ 实现较为复杂，需维护候选集合、共享统计与决策阈值，适用场景受限于可观测标签和干预可选项。

---

## 765. KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry

**arXiv ID:** 2610.37655 | [PDF](https://arxiv.org/pdf/2610.37655v1)

**作者:** Wenqing Peng `[一作]` (University of Helsinki), Zhi-Song Liu `[通讯]` (LUT University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 KPPax，一款可以把 KPP 语言描述的大气化学机制转译为可微分、即时编译且可向量化的 JAX 代码的工具；

**💡 创新点**

其创新点在于将化学速率法在对数空间使用稀疏操作实现可微分化，并利用 JAX 自动微分和可选的前向/反向 adjoint 直接获取稠密 ODE 的精确灵敏度，无需手工导出伴随方程；

**🔧 技术方法**

采用的技术包括 KPP 语法解析、稀疏坐标列表（COO）存储计量矩阵、Fortran→JAX 表达式转译、对数空间稀疏乘积、JAX 的隐式 ODE 求解器（Kvaerno 系列）以及递归检查点技术的自动微分；

**📊 数据集**

使用了 POLLU 这一 20 物种、25 反应的基准机制，并在其基础上生成合成的流管实验观测数据及雾霾发射量校正的合成场景；

**📈 对比分析**

通过比较自动微分得到的雅可比与中心差分（误差 <10^-6）、在线微调的 Biogenic 校正误差比离线两步训练下降 2.4–5 倍，并且在线细调在单块 NVIDIA GH200 GPU 上仅需约 2 分钟；

**⚠️ 局限性**

局限性包括：对大规模机制（数千种物种）需要稀疏或矩阵无关的线性求解器；对数空间要求浓度始终正，需截断导致梯度失效；只进行了局部可识别性分析，未评估全局不识别区；未在真实实验数据上验证。

---

## 766. Can LLMs Uncover Hidden Physical Laws through Free Exploration?

**arXiv ID:** 2610.37649 | [PDF](https://arxiv.org/pdf/2610.37649v1)

**作者:** Zhen Yang `[一作]` (Hong Kong University of Science and Technology Guangzhou), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology Guangzhou)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个完全自动化的实验框架LongChang，让大型语言模型在无任务提示下自行观察、提出问题、设计实验并检验改写的物理定律；

**💡 创新点**

其创新点在于将探索、工具设计、评估与场景演化整合为闭环循环，真正测试LLM的自发科学探究能力；

**🔧 技术方法**

技术上利用MuJoCo物理仿真、LLM生成场景与工具、LLM评审报告并自动记录探索过程；

**📊 数据集**

使用15个基准场景（1个正常控制+14个单一改写物理定律的场景）作为数据集；

**📈 对比分析**

对12种LLM进行对比实验，GPT-6-Astra在无任务提示下实现了16.7%的发现率，其余模型均为0%，表明该任务对现有LLM仍具挑战；

**⚠️ 局限性**

局限性包括：发现率低、仅针对简化的模拟物理、无法推广至真实世界科学问题、LLM对复杂实验设计的掌握不足。

---

## 767. Multi-State Linear Attention with Generalized Second-Order Routing

**arXiv ID:** 2610.37645 | [PDF](https://arxiv.org/pdf/2610.37645v1)

**作者:** Hee Seung Hwang `[一作]` (Princeton University), Zhiwei Deng `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种改进多状态线性注意力聚合的方法，利用键向量的第一、第二矩（均值与方差）来计算每个段的聚合权重，进一步提高长序列建模效果。

**💡 创新点**

创新点在于首次将段内键的均值与方差同时纳入聚合权重的学习，通过门控混合两项的softmax，既保留了内容感知又减少了对位置的偏倚。

**🔧 技术方法**

技术实现包括：①在每个段内维护固定维度的键均值和协方差矩阵；②查询时用键-查询点积推导均值与方差；③使用可学习的门和温度对两项进行归一化并加权。

**📊 数据集**

实验使用的模型为 Gated DeltaNet‑1.3B 与 Mamba‑2‑780M，数据集涵盖 Long‑Bench、RULER（NIAH 与 QA）、HELMET（检索生成）以及公开的 50B/300B 预训练语料。

**📈 对比分析**

与 Vanilla、Position、Meanpool 三种聚合基线对比，本文方法在 RULER NIAH/QA、LongBench、HELMET 上均提升 7–13% 以上，且在相同或更低的显存消耗下保持竞争优势。

**⚠️ 局限性**

局限性包括：①需在预训练模型上进行中期微调，过程复杂；②段长度固定时对不同上下文长短的适应性有限；③额外的二阶矩存储仍会带来一定的显存开销，尤其在极大段尺寸时。

---

## 768. LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation

**arXiv ID:** 2610.37570 | [PDF](https://arxiv.org/pdf/2610.37570v1)

**作者:** Nesta Midavaine `[一作]` (Prosus), Zülküf Genç `[通讯]` (Prosus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了 LCM-Promo，一个针对促销分配的 LLM 基础模型，用于预测用户在不同折扣条件下的转化概率和提升效果，并在 iFood 平台上线。

**💡 创新点**

创新点包括：1）将单 token 输出解释为转化阈值概率分布，自动生成单调的促销响应曲线；2）利用 LLM 在大规模商务日志上微调，支持数千个上下文分群而非传统细分；3）通过阈值 PMF 解码直接得到可用于预算分配的提升估计，显著提升模型可解释性和应用效率。

**🔧 技术方法**

技术手段包括：4B 参数稠密 LLM（vLLM 部署）、对话式提示（系统提示 + 会话上下文 + 用户表示 + 折扣映射）、单 token 分类 + 交叉熵训练、阈值 PMF 归一化、Bootstrap 采样、批量推理以及对比评估的水波纹标签设计。

**📊 数据集**

使用的数据集为 iFood 的生产日志（订单与促销交互历史）、随机对照试验（RCT）数据进行离线评估，以及多维度用户与会话行为汇总（城市、交付时段、生命周期、钱包配置）。

**📈 对比分析**

在线 A/B 测试与 Segment-EMA 基线对比，提升转化 1% 并降低促销成本 6%；离线评估在 RCT 数据上，LCM-Promo 与提升随机森林（Uplift Random Forest）在 AUUC、Qini 接近或略优，但在 Uplift@30 上略逊，水波纹版本表现更差。

**⚠️ 局限性**

局限性包括：只在 churn 复活阶段验证，缺乏其他生命周期或平台的泛化验证；水波纹标签在 LLM 预训练目标下表现不佳；离线评估受限于固定促销深度的 RCT 数据；生产部署需要高 GPU 资源，且对多样化促销策略的泛化仍需进一步研究。

---

## 769. When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies

**arXiv ID:** 2610.37547 | [PDF](https://arxiv.org/pdf/2610.37547v1)

**作者:** Anukriti Singh `[一作]` (University of Maryland), Firas Abi-Farraj `[通讯]` (Enchanted Tools)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Adaptive Residual Correction（ARC）方法，利用稀疏人工纠正仅在必要时为冻结的基线策略添加残差修正，从而提升最后一公里的执行精度。

**💡 创新点**

创新点在于：①仅通过人类在关键时刻的少量纠正学习一个门控模型，决定何时启用残差控制；②通过统计人类纠正产生的低维基底约束残差探索空间；③在此支持下使用强化学习（TD3）学习基于任务奖励的残差，而非直接模仿人类纠正。

**🔧 技术方法**

使用技术包括：强化学习（TD3），门控分类器（预干预状态判定），主成分分析（提取低维纠正基底），以及对机器人执行动作的重放回放。

**📊 数据集**

数据集来源为三项真实机器人精细操控任务：托盘堆叠、标记插入窄槽、杯子堆叠，收集约8-10次包含人类纠正的轨迹，随后进行两轮自主收集、奖励更新。

**📈 对比分析**

与基线基线策略和直接监督人类纠正的对比实验显示，ARC将平均成功率从33.3%提升至75.0%，在托盘堆叠任务中达到100%成功；相比之下，监督方法仅在三项任务中获得3/12、5/12、3/12的成功率。

**⚠️ 局限性**

限制主要在于门控模型需要依赖人工纠正时机，需为每个任务收集人工纠正示例；若不同操作者或任务更改，门控策略可能需要重新训练；此外，仍需进一步验证在更大范围任务中的可推广性。

---

## 770. Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction

**arXiv ID:** 2610.37668 | [PDF](https://arxiv.org/pdf/2610.37668v1)

**作者:** Prakhar Kaushik `[一作]` (Johns Hopkins University), Alan Yuille `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Hexels——一种具有一次空间色彩场的正六边形平面原语，用于高效的视角合成、表面重建和二维图像表示。

**💡 创新点**

创新点在于将正六边形支持与局部一次颜色梯度相结合，显著减少重叠与存储，提升单个原语的表达能力，并统一同一原语适用于多种任务。

**🔧 技术方法**

技术包括视角纠正射线-平面交点渲染、软六边形权重函数、一次颜色梯度、Adam 优化、稠密化与剪枝策略等。

**📊 数据集**

实验使用 Mip-NeRF360、DTU、Tanks & Temples 以及 Image-GS 数据集。

**📈 对比分析**

在匹配模型内存、原语数量和原生设置下与 2DGS、Triangle Splatting、BBSplat 等基线比较，Hexels 在 PSNR 上提升 4.5–7.4 dB、帧率提升 1.5–2.3 倍、VRAM 使用显著降低，表现出更优的质量‑效率折衷。

**⚠️ 局限性**

局限性包括一次色彩模型难以捕捉高频纹理、平面原语无法完美覆盖曲面、稀疏初始化可能留下未覆盖区域、以及未能形成闭合网格等问题。

---

