# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-28 | 今日论文总数: 608

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

**arXiv ID:** 2608.26480 | [PDF](https://arxiv.org/pdf/2608.26480v1)

**作者:** Victor Gao `[一作]` (Persis Capital Inc.), Lee `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了无训练、无任务特定调优的管理‑工人协作框架对大语言模型在编程题解答能力的提升。

**💡 创新点**

创新点在于引入共享文件系统工作空间和动态管理器，实时重构任务列表并协调多次交互，而无需学习或微调。

**🔧 技术方法**

采用了零射击自组织（zero‑shot orchestrator）、工作流程拆分、共享黑板、内部验证与上下文管理等技术。

**📊 数据集**

使用了LiveCodeBench的100道最新硬难度编程题作为数据集。

**📈 对比分析**

通过在同一模型下比较单次调用与管理器‑工人循环两种条件，进行5次对照，测得pass@1提升幅度从+8到+23点，部分模型在成本约为单模型三倍的情况下，性能可媲美更大模型。

**⚠️ 局限性**

局限性包括成本显著上升、受限于模型思考阈值、部分模型在思考关闭时性能下降、仅评测编程生成任务、平台不稳定导致结果波动、未验证多模型异构协作效果。

---

## 2. A Catalog of User Authentication Patterns

**arXiv ID:** 2608.26955 | [PDF](https://arxiv.org/pdf/2608.26955v1)

**作者:** Alex R. Mattukat `[一作]` (RWTH Aachen University), Horst Lichter `[通讯]` (RWTH Aachen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

创建并发布了一个专注于用户身份验证的模式目录，包含14种具体身份验证模式，并按因素和角色进行分类。

**💡 创新点**

首次将身份验证视为一阶安全概念，提供了完整的模式目录并为非安全专家设计了统一抽象级别的描述，填补了现有模式目录中对身份验证缺乏细粒度支持的空白。

**🔧 技术方法**

通过系统化文献回顾与模式化分析，基于身份验证因素（内在、知识、占有）与常用角色（主要、辅助）进行模式分类与描述。

**📊 数据集**

主要使用已有的39种身份验证技术清单和相关安全模式文献作为参考，没有使用机器学习或大规模数据集。

**📈 对比分析**

本工作没有采用实验或性能对比方法，重点在于模式描述与分类，不涉及实现性能评估。

**⚠️ 局限性**

局限性包括：仅覆盖人类身份验证，未考虑多模式组合形成的身份验证技术；未将身份验证因素本身视为抽象模式；缺乏对模式组合与实现细节的系统支持。

---

## 3. NeoTriFuse: Reliability-Aware Multimodal Fusion under Missingness Heterogeneity for Neonatal Mortality Risk Prediction

**arXiv ID:** 2608.26436 | [PDF](https://arxiv.org/pdf/2608.26436v1)

**作者:** Jiyuan Tian `[一作]` (University of Sydney), Haohui Lu `[通讯]` (Charles Darwin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了NeoTriFuse框架，利用缺失率作为可靠性信号动态调节不同模态（静态变量、时间序列、统计摘要）在死亡风险预测中的贡献；

**💡 创新点**

创新点在于将缺失性视为可靠性指标，通过门控融合机制实现可靠性感知，多尺度时间编码（局部CNN+全局Transformer）与病人级统计摘要的组合，并联合死亡风险与LOS辅助任务；

**🔧 技术方法**

使用的技术包括一维卷积+Transformer编码器、门控融合机制、focal BCE与SmoothL1损失、AdamW优化器，以及与传统机器学习基线（LR、RF、XGBoost、CatBoost、LightGBM等）和深度序列基线（GRU、CNN、Transformer、GRU‑D、TNformer‑MP、TIM‑MSFL）的对比；

**📊 数据集**

实验数据来自PAS 2024 NICU死亡率预测挑战数据集，包含HR/SpO₂时间序列、静态分娩变量和长度-of-stay（LOS）标签；

**📈 对比分析**

与传统和深度序列基线相比，NeoTriFuse在F1（0.6736±0.0216）、AUROC（0.9454±0.0056）等指标上表现最佳，尤其在阈值相关的精准率、召回率上显著优于竞争模型；

**⚠️ 局限性**

局限性包括仅在完整记录的后向预测任务上评估，未涉及实时预测窗口、外部验证，以及缺乏对模型解释性的深入分析。

---

## 4. CIFQA: A Deterministic Tool-Grounded Multi-Agent LLM Framework for Financial Query Answering

**arXiv ID:** 2608.26114 | [PDF](https://arxiv.org/pdf/2608.26114v1)

**作者:** Kunjesh Parekh `[一作]` (Indian Institute of Technology Jodhpur), Divya Saxena `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CIFQA的确定性工具驱动多代理LLM框架，用于解决需要精确算术计算的金融问答任务

**💡 创新点**

通过将语言理解与数值执行严格分离，消除LLM在执行多步计算时的算术幻觉，并在固定存款查询上实现高精度

**🔧 技术方法**

使用多代理LLM架构（路由器、提取器、规划器、响应生成器）配合Python确定性计算引擎（利率查找、日历处理、利息计算、规则引擎）

**📊 数据集**

在自编的126条固定存款（FD）查询基准上评估，其中101条为计算密集型查询

**📈 对比分析**

与GPT‑5.3、Gemini 3、Claude Sonnet 4.6等前沿LLM在单提示下对比；CIFQA在计算密集型查询上达95.54%准确率、整体90.87%，显著优于这些模型，且同等或更优于更大规模的开源模型

**⚠️ 局限性**

对政策/检索密集型查询的性能仍不佳；框架依赖准确的参数抽取与路由，错误会级联；目前仅验证于固定存款领域，需在更多金融场景中进一步验证

---

## 5. Large Models for Battery Prognostics and Health Management: A Review and Future Roadmap

**arXiv ID:** 2608.26111 | [PDF](https://arxiv.org/pdf/2608.26111v1)

**作者:** Jiale Liu `[一作]` (University of Edinburgh), Min Xie `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大模型在电池寿命预测与健康管理（BPHM）中的应用与发展，系统梳理了关键技术、进展、挑战与未来路线图。

**💡 创新点**

首次提供大模型在BPHM领域的系统性评估，提出四维度分类（数据稀缺、泛化、知识融合、系统自动化），并针对每一维度给出技术方案与研究方向。

**🔧 技术方法**

主要使用Transformer架构、无监督预训练（自监督学习）、多模态融合、参数高效微调（PEFT）以及推理与代理能力（链式思考、工具调用）。

**📊 数据集**

引用并整合多公开电池数据集，如NASA PCoE、CALCE、MIT‑Stanford、Oxford、RWTH、Sandia、Hawaii、XJTU、HUST、Tongji、ISU‑ILCC 等，涵盖不同化学、尺寸与工况的时间序列与多模态数据。

**📈 对比分析**

对比方法主要通过在公开数据集上对比传统 DL（CNN、RNN、DFN等）与大模型（TimeGPT、Lag‑Llama、BatteryGPT 等），发现大模型在少量标注、跨化学泛化和多模态推理上表现更优，但缺乏统一的标准评测。

**⚠️ 局限性**

主要限制包括：缺乏大规模、标准化的公开数据；模型可解释性与安全性不足（幻觉、攻击风险）；部署在边缘设备时计算与延迟瓶颈；以及在工业场景下的验证与法规符合性不足。

---

## 6. Spectral Approximation and Ergodic-Capacity Convergence of HMIMO Channels under Spatial-Wavenumber Domain Mismatch

**arXiv ID:** 2608.26802 | [PDF](https://arxiv.org/pdf/2608.26802v1)

**作者:** Hangsong Yan `[一作]` (Hangzhou Institute of Technology, Xidian University), Shu Sun `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对非可分离的方形-圆盘空间-波数域不匹配的连续 holographic MIMO 通道做了稀疏矩阵投影，得到一阶整体谱逼近误差与容量损失均可显式估计。

**💡 创新点**

创新点在于利用一维 PSWF 的张量投影保持圆形波数支持，推导出整体谱误差和容量误差均以 1D PSWF 频尾为上界的超指数收敛；并给出了相应的高效求积规则。

**🔧 技术方法**

主要技术包括 PSWF 解析性质、张量投影、谱逼近与极限分析、矩阵稀疏化、极大极小原理以及 Jensen 不等式和容量上界推导。

**📊 数据集**

实验数据全部来自数值仿真（无真实测量数据集），通过对比不同 1D PSWF 维数的截断与传统 DoF 截断来验证性能。

**📈 对比分析**

与传统按空间自由度截断方法相比，数值结果显示在小型天线场景下容量提升可达 30‑40%，而在大尺寸天线时提升不足 5%；容量误差随截断维数呈超指数衰减。

**⚠️ 局限性**

局限在于仅考虑方形天线、远场、统计平稳的波数域，未覆盖更一般的几何形状、近场效应及非球面波的情况，且高维 PSWF 计算仍具有一定复杂度。

---

## 7. Cost-Utility Alignment in LLM Agent Trajectories:Profiling,Attribution,Diagnosis,Adaptation,and Evaluation

**arXiv ID:** 2608.26195 | [PDF](https://arxiv.org/pdf/2608.26195v1)

**作者:** Dan Liu `[一作]` (Beijing Normal University), Jian Li `[通讯]` (Beijing Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向大型语言模型代理的五阶段成本-效用对齐框架，涵盖成本剖析、效用归因、偏差诊断、针对性适配与评估，形成闭环。

**💡 创新点**

创新点在于将资源消耗与任务贡献视为同一执行轨迹上的双账本，系统化归因方法并按认知、外部交互、恢复、资源配置与多代理协调等五类误差进行诊断与补救。

**🔧 技术方法**

使用了多维度成本剖析（token、延迟、金钱、风险）、层级成本定位、语义动作标注、代理级成本记录；效用归因包括代理回溯、信息依赖、逆向因果干预；误差诊断采用可压缩思考、工具调用裁剪、路径回滚、资源分配调度等针对性适配；评估基于Tokenomics、Terminal‑Bench、TraceLab、ScienceAgentBench 等多维度基准。

**📊 数据集**

实验数据来源于 Tokenomics、Terminal‑Bench、TraceLab、ScienceAgentBench、FDABench、SWE‑Bench、AgentBoard、WebArena、OSWorld、BenchGuard 等公开基准与内部日志。

**📈 对比分析**

与现有的单独优化方法相比，框架在多维度成本（token、调用次数、延迟、金钱）与效用（任务成功率、质量分数、可靠性）上实现了 Pareto 提升；在可靠性基准下也能保持或提升成功率，同时显著降低平均成本；在人机协作情境下通过 GDPval、SWE‑Lancer 等衡量，显示整体成本与人力成本下降。

**⚠️ 局限性**

局限性包括：1）开放环境下难以执行完整的因果归因与回放；2）归因与适配闭环的计算成本尚未优化；3）基础设施漂移导致基准不可复现，难以评估长期稳定性；4）人机工作流中成本转移与审计缺失，难以评估真实经济效益。

---

## 8. Direct Manipulation and Natural Language Programming, Together at Last?

**arXiv ID:** 2608.26359 | [PDF](https://arxiv.org/pdf/2608.26359v1)

**作者:** Parker Ziegler `[一作]` (University of California, Berkeley), Sarah E. Chasins `[通讯]` (University of California, Berkeley)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种框架，将直接操纵编程与自然语言编程统一到同一编辑语言中，并在CartoKit（一个基于GeoJSON可视化的直接操纵系统）中实现了该框架；随后在18名数据记者的受控实验中比较了三种编辑模式（DM+NL、DM、NL）对程序编写过程的影响；

**💡 创新点**

创新点在于①将程序视为编辑序列，并用结构化编辑语言作为DM与NL的共享接口；②利用约束解码（constrained decoding）让LLM直接生成合法的编辑序列，避免了传统自然语言生成的无效或错误代码；③首次在同一系统中进行DM+NL与单一模式的对比研究，并揭示用户偏好与交互策略的细微差异；

**🔧 技术方法**

技术手段包括：Patch‑Reconciliation框架（DiffSync）实现增量同步；结构化编辑语言与Zod模式定义；OpenAI GPT‑5 + Structured Output 与约束解码；Web前端的直接操纵UI；实验数据分析：生存分析、效应量计算、NASA‑TLX；

**📊 数据集**

使用的工作数据集为公开的GeoJSON地理空间数据（如美国选举结果、公共服务设施、人口统计等），这些数据被用于任务A（重现报纸地图）和任务B（探索性可视化）；

**📈 对比分析**

对比方法：在同一受试者内完成三种模式的两项任务，记录完成率、完成时间、NASA‑TLX评分；结果显示DM+NL与单一DM在完成率和时间上相近，而NL模式完成率仅50%，且DM+NL中的自然语言编辑仅占6.14%；在主观负荷上DM+NL与DM无显著差异，均低于NL；

**⚠️ 局限性**

局限性包括：①仅在地理空间可视化领域验证，难以推广到其他领域；②受试者主要为具备GIS背景的数据记者，样本规模有限；③实验使用的LLM（GPT‑5）和约束解码技术随模型演进而快速过时；④自然语言编辑在语义理解与错误容忍方面仍表现不足，且对模型失败的容忍度低。

---

## 9. SysComb: Fine-Grained Transparent System Call Filtering for Attack Surface Reduction

**arXiv ID:** 2608.26871 | [PDF](https://arxiv.org/pdf/2608.26871v1)

**作者:** Matthew Rossi `[一作]` (Università degli Studi di Bergamo), Stefano Paraboschi `[通讯]` (Università degli Studi di Bergamo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于eBPF的透明系统调用过滤框架SysComb，能够在不改动应用或内核的情况下根据线程状态动态切换系统调用过滤器，支持多状态、线程级别；

**💡 创新点**

创新点在于引入细粒度时序化状态机和两种策略（seccomp‑like与least‑privilege），实现内核级同步执行、无用户空间监控、无内核补丁，并支持动态状态迁移和流向回传；

**🔧 技术方法**

主要技术包括eBPF程序与uprobes/kprobes/raw tracepoints、Seccomp过滤、状态图与流向回传、bpffs、blazesym解析符号、自动化过滤生成；

**📊 数据集**

使用九款主流服务器/数据库（Apache、Lighttpd、Nginx、Memcached、MongoDB、Redis、MySQL、PostgreSQL、Bind）及其测试套件进行系统调用跟踪和安全评估，并参考45个近期内核CVE和535条shellcode数据；

**📈 对比分析**

通过微基准与宏基准与strace、未细分过滤、传统seccomp、用户空间与中断等多种实现对比，单次系统调用平均延迟约300 ns（tracing）/480 ns（enforcement），与seccomp相当；宏基准请求延迟最高仅4.46%，整体可接受；

**⚠️ 局限性**

局限性包括仅支持原生线程，对绿色线程或解释型语言需额外探针；需要root加载eBPF；依赖eBPF对象完整性；攻击者可能篡改eBPF map/程序；无法直接对JVM/运行时内核代码做过滤。

---

## 10. 4DSynth: Controllable Procedural World Synthesis for Dynamic Embodied Simulation

**arXiv ID:** 2608.26947 | [PDF](https://arxiv.org/pdf/2608.26947v1)

**作者:** Zehao Qi `[一作]` (Nanyang Technological University), Shuyang Sun `[通讯]` (Google DeepMind)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套可控的程序化场景生成管道，能够将自然语言描述、蓝图掩码或单张照片转化为可编辑的四维（空间+时间）环境，并基于此生成了可在物理仿真和任务生成中直接使用的场景。

**💡 创新点**

创新点在于：①将四条不同的场景实现路径（本地程序化、布局条件合成、实景复现）统一到同一几何共享Stage；②实现了基于几何的四维动画、摄像机轨迹规划和物理可导出；③构建了可重复、可扩展的导航与搬运基准 4DSynth‑Nav，提供多层难度与动态障碍。

**🔧 技术方法**

技术主要包括：自然语言引导的语义规范化、基于OpenUSD的Stage抽象、基于物理可行域的轨迹规划、基于多模态感知的实景复现、物理属性自动映射、以及对生成结果的离线 Isaac Sim 验证。

**📊 数据集**

使用的主要数据集是 Infinigen Indoor/Outdoor 生成的 122 个物理可交互场景，用其生成 333 个包含动态角色的导航/搬运任务（4DSynth‑Nav）。

**📈 对比分析**

对比实验使用 Qwen3‑VL‑30B 与 Gemini 3.1 Pro 两大多模态模型，在三层难度上评估成功率、子任务完成度、目标距离、碰撞次数和步数。结果显示 Gemini 在所有层级的成功率（最高 33.3%）和子任务完成度（最高 40.7%）明显优于 Qwen，但两者在大多数任务上仍表现不佳，难度层级越高成功率越低。

**⚠️ 局限性**

局限性包括：①生成的动态障碍仅为预设动画，缺乏响应式交互；②对目标搜索、空间记忆和多阶段规划仍易失误，导致高碰撞和循环轨迹；③系统对场景复杂度和多模态感知的鲁棒性不足，特别是在极端视觉遮挡或动态变化环境下表现不佳。

---

## 11. SAREF-based Ontology for Distributed AI Workflows across the Edge-Fog-Cloud Continuum

**arXiv ID:** 2608.26160 | [PDF](https://arxiv.org/pdf/2608.26160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 12. Agentic AI Containment Architecture for Security Hardening

**arXiv ID:** 2608.26108 | [PDF](https://arxiv.org/pdf/2608.26108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 13. Sharp Minimax Regret for Infinite-Memory Logistic Prediction

**arXiv ID:** 2608.26515 | [PDF](https://arxiv.org/pdf/2608.26515v1)

**作者:** Vaneet Aggarwal `[一作]` `[通讯]` (Purdue University), Vaneet Aggarwal (Purdue University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一个特定有限字母表的外生驱动源的在线预测，具有无限输入记忆。

**💡 创新点**

提出了滞后分辨谱Γ_T(r)作为每个可求和包络的上界，并在给定的有限样本条件下证明了其为经典指数和多项式包络的匹配最小最大尺度。

**🔧 技术方法**

使用了局部贝叶斯混合模型和在线牛顿预测器。

**📊 数据集**

使用了独立的Rademacher输入序列作为数据集。

**📈 对比分析**

通过与已知的最优预测器进行比较，证明了提出的方法在最小最大累积后悔方面的有效性，性能表现为Θ(α^-1log^2T)和Θ(T^1/(2s))。

**⚠️ 局限性**

限制在于该结果特定于外生滞后模型，并不适用于任意的平稳无限记忆源。

---

## 14. Direct-Operable SIMD Bit-Slicing: A Framework for Memory-Efficient Predicate Evaluation

**arXiv ID:** 2608.26368 | [PDF](https://arxiv.org/pdf/2608.26368v1)

**作者:** Arunkumar Mathiyazhagan `[一作]` `[通讯]`, Arunkumar Mathiyazhagan

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在JVM内使用Project Panama Vector API直接对位切片压缩数据执行谓词筛选的框架，省去了解压缩步骤

**💡 创新点**

将位切片技术从传统索引转变为主数据表示，并与SIMD向量化结合，实现零解压的谓词计算

**🔧 技术方法**

使用Project Panama Vector API（SIMD指令）、Java 21的Foreign Function & Memory API进行离堆存储、位切片编码与谓词评估

**📊 数据集**

使用合成的TPC‑H/TPC‑DS模拟列分布数据（1–60 M值、不同位宽）以及真实的TPC‑DS查询模式

**📈 对比分析**

与原始int数组、Boxed Integer、Arrow Java、Parquet-MR等基线对比；在ARM NEON 128‑bit上实现2.4–10.8×的速度提升、5.3×以内存压缩、GC压力降低

**⚠️ 局限性**

仅适用于固定宽整数及其转换，高基数字符串、浮点、可变长度结构、任意精度小数等场景受限；对高位宽列的压缩效果有限，且需额外编码开销

---

## 15. EmoSay: Artificial Intelligence-Driven Text-to-Emotional-Speech System for Affective Communication in Extended Reality

**arXiv ID:** 2608.26566 | [PDF](https://arxiv.org/pdf/2608.26566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 16. DRL: A Deterministic Relational Middleware Layer for Transaction-Safe Enterprise NL2SQL Under Schema-Graph Scaling

**arXiv ID:** 2608.26172 | [PDF](https://arxiv.org/pdf/2608.26172v1)

**作者:** Sanjay Mishra `[一作]` (Independent Researcher), Ganesh R. Naik `[通讯]` (Flinders University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 DRL（Deterministic Relational Middleware Layer）架构，用于在企业 OLTP 数据库上构建安全、可扩展的自然语言对 SQL 接口，并搭建了 1,000 题验证套件进行系统评估。

**💡 创新点**

创新点在于将 schema 规模化问题抽象为图规模化模型，并通过离线 deterministic router、RAST 语法检查以及 EXPLAIN+NULL 保障三层管控，显著压缩 LLM 的上下文、保证事务安全并检测无声偏差。

**🔧 技术方法**

使用的技术包括图检索路由器、Relational AST 编译器、事务安全验证层、EXPLAIN 计划检查、COALESCE 自动注入、以及多方 LLM（GPT‑4o、Claude Sonnet 4.5、Gemini 2.5 Flash）结合 Postgres/MySQL 的后端。

**📊 数据集**

数据集为六张真实企业 OLTP 模拟 schema（共 465 张表）以及基于 1,000 题的验证套件（包含 519 人工核对题目与 481 个 gold SQL，按 tier 分布）。

**📈 对比分析**

通过 B0–B3 基线对比，测量上下文字节、prune 延迟、middleware p95 以及 Plan Safety Compliance，结果表明 DRL 能将上下文压缩 92%，prune p95 0.58ms，Plan Safety compliance 在 PostgreSQL 上从 68.6% 提升至 70.8%；在三大模型上执行匹配率约 52–53%，跨供应商差异已消失。

**⚠️ 局限性**

局限包括 RAST 仅实现离线校验且未集成到在线请求、仅支持 Postgres/MySQL 两种方言、未覆盖写入路径、上下文裁剪不等价于权限控制、以及对 MySQL 的模型生成 PSC 未完整评估。

---

## 17. Closing the Loop on the Poppy Humanoid: Bipedal Locomotion with Linear-Quadratic Control and Learned Cost Functions

**arXiv ID:** 2608.26505 | [PDF](https://arxiv.org/pdf/2608.26505v1)

**作者:** Xulin Chen `[一作]` (Syracuse University), Garrett E. Katz `[通讯]` (Syracuse University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为标准 Poppy Humanoid 机器人构建了一套基于 LQR 的闭环步行控制器，并通过开放式循环实验数据学习了二次成本函数，从而显著提升了步行的可靠性。

**💡 创新点**

创新点在于利用开放式循环轨迹的成功/失败标签，通过凸优化（半正定规划）学习可区分跌倒与成功的二次成本；并在此基础上结合线性系统辨识实现了时变 LQR 控制，首次在标准 Poppy 硬件上实现了可靠的自动步行。

**🔧 技术方法**

采用了时间变 LQR、最小二乘线性系统辨识、半正定凸优化求解二次成本、统计显著性检验（Mann‑Whitney U）以及 PyPot API 控制动力学接口。

**📊 数据集**

使用了 125 条开放式循环硬件跑测数据（包含不同扰动水平的关节角度轨迹）以及对应的成功步数标签；实验还在两个不同环境下收集了 100/80 条闭环与开环跑测结果。

**📈 对比分析**

通过在同一硬件、相同环境下的随机交叉实验，对比闭环 LQR 控制与原始开环轨迹的成功步数；结果显示闭环控制在办公室环境中成功率由 62% 提升至 78%（p=0.00225），平均成功步数从 4.18 步提升至 5.13 步；在实验室环境中成功率从 30% 提升至 42.5%（p=0.00014），平均成功步数从 3.75 步提升至 4.8 步。

**⚠️ 局限性**

局限性包括：系统辨识仅采用简单线性拟合；成本函数假设跌倒点不晚于第 n_F 步；未利用惯性测量或压力传感器，仅依赖关节角度；未优化步行速度、转弯半径或能耗等关键步态指标；在不同环境下的泛化性能仍有限。

---

## 18. Algorithmic Principles For Multiclass Learning Are Hard To Come By: Limits of Regularization and Proper Learning

**arXiv ID:** 2608.26516 | [PDF](https://arxiv.org/pdf/2608.26516v1)

**作者:** Julian Asilis `[一作]` (University of Southern California), Chang Wang `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究多分类学习理论中的核心问题——学习可行性与学习算法的设计，系统证明了在多分类情形下，学习并不等价于“完全正确”学习（proper learning）以及传统的正则化（SRM）方法，并给出了可行性与不可行性的边界条件。

**💡 创新点**

创新点包括：① 构造可学习但无法嵌入任何可完全正确可学习类的多分类问题，否定了学习可归约为完全正确学习的猜想；② 证明完全正确学习有时必须在训练样本上产生任意次子线性错误率；③ 证明权重化的 SRM 以及基于局部正则化的学习器均可能失效；④ 提出两类正则化可行性的充分条件——fallback core 与有序一致性维度，并给出揭示偏好可积性的完整表征。

**🔧 技术方法**

采用的技术主要是组合构造与概率论：2^d-ary 树、Cantor 类、投影平面 incidence 图、完整三部图、以及对有向三角形的计数与随机化。还使用了图论中的定理（如图维度、VC 维度、差分约束可行性判定）来分析正则化与学习器的可行性。

**📊 数据集**

该工作为纯理论研究，未使用实际数据集；所有结果均基于数学构造与抽象分析。

**📈 对比分析**

由于研究的是理论可行性与不可行性，没有实验对比；作者通过构造例子与上界/下界证明不同学习模型的性能极限。

**⚠️ 局限性**

限制主要在于：本文只讨论了特定的学习模型（完全正确、权重化 SRM、局部正则化）与其对应的不可行性；并未给出在所有多分类问题上都能保证可学习的通用算法。未来仍需探索其他可能的算法框架。

---

## 19. Beyond Capability Benchmarks: Learning Operational Fingerprints of LLM Cloud Services from Production Incident Metadata

**arXiv ID:** 2608.26332 | [PDF](https://arxiv.org/pdf/2608.26332v1)

**作者:** Meiwei Zhang `[一作]` (Google), Sergey Borodavkin `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为OpEmbed的框架，利用结构化的支持案例元数据学习LLM云服务的运维指纹；

**💡 创新点**

创新点在于将运维特征转化为可学习的低维嵌入，结合时间对比学习、交叉视图重构与世代序位正则化，首次实现跨模型、跨版本的运维行为可视化与预测；

**🔧 技术方法**

采用时间对比学习、交叉视图重构、生成序位正则化的多目标训练目标，并用PCA预处理、行标准化与L2归一化的线性投影；

**📊 数据集**

使用来自Google Cloud的33,000+条生产支持案例，覆盖7个LLM系列、26个月的时间窗口；

**📈 对比分析**

与全局平均、同族平均、原始签名KNN等无学习基线对比，OpEmbed在留一模型预测、少样本预测与跨模型故障迁移等任务上均表现更优，MAE和R²均提升显著；

**⚠️ 局限性**

局限在于仅来自单一支持生态，可能受本地工作流影响；缺乏对每个训练目标贡献的独立消融分析；以及模型需要基于PCA初始化，未能完全区分PCA与学习目标的效益。

---

## 20. GraphMemix: Query-Aware Evidence Forests for Long-Term Multimodal Agent Memory

**arXiv ID:** 2608.26983 | [PDF](https://arxiv.org/pdf/2608.26983v1)

**作者:** Geng Li `[一作]` (Peking University), Yuxin Peng `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于组合优化的图内存框架，用查询感知的证据森林构造方式，在多模态代理的长期记忆中高效检索和组织相关证据。

**💡 创新点**

创新点包括：①将记忆选择建模为查询条件的森林优化问题；②采用多视图检索与关系扩展构建有限候选图；③引入节点直接评估器和锚定关系验证器，区分直接支持与增量补充；④通过最大权森林（Kruskal）实现全局最优的节点与边选择。

**🔧 技术方法**

使用的技术有：多模态统一编码器、列表式节点评估器、Evidence-Chain Verifier、最大权森林算法、两阶段确定性求解器、基于深度学习的评估器（Qwen3‑VL、Gemma 4等）。

**📊 数据集**

实验数据集包括四个长期多模态记忆基准：ATM‑Bench、Mem‑Gallery、MemEye、H2HMem。

**📈 对比分析**

与 A‑MEM、VimRAG、LightMem、UniversalRAG 等基线在同一阅读器下对比，模型在 Qwen3‑VL 与 Gemma 4 上均获得所有基准的最高 Judge Accuracy，宏平均提升约12–13个百分点，同时实现生命周期成本 Pareto 前沿，表明在准确性与效率上均优于现有方法。

**⚠️ 局限性**

局限性：需要对候选图和多视图检索做参数调优；目前仅验证于四个基准，跨领域或更大规模记忆的泛化能力待进一步评估；依赖两级评估器，训练与推理成本仍不 negligible。

---

## 21. Instruction Quality Matters: Refining Instructions for Effective Preference Learning

**arXiv ID:** 2608.26779 | [PDF](https://arxiv.org/pdf/2608.26779v1)

**作者:** Seohyeong Lee `[一作]` (Sogang University), Buru Chang `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了指令质量对偏好学习的影响，并提出了一套指令改进（refinement）流程，用奖励模型筛选低质量指令，再通过 rubric‑guided LLM 反馈进行重写，以提升偏好数据质量。

**💡 创新点**

创新点在于把指令质量视作偏好信号形成的上游瓶颈；通过 Best‑of‑N / Worst‑of‑N 分析证明低质量指令会同时降低最佳与最差回复的质量；并首次将 rubric‑guided LLM 反馈用于指令改写，避免仅靠粗糙重写或仅改正响应。

**🔧 技术方法**

技术手段包括：预训练 LLM（LLaMA3‑8B、Mistral‑7B）生成回复；ArmoRM、OSSAT‑RM 等奖励模型评估回复质量；DPO/SimPO 优化算法；rubric‑guided 反馈与改写循环；Best‑of‑N / Worst‑of‑N 评价；阈值敏感性与 rubrics 对齐分析。

**📊 数据集**

使用的数据集有：UltraFeedback、MT‑Bench、Evol‑Instruct、AlpacaEval、UltraMedical‑Preference 等，涵盖多任务、多模型的离线和在线偏好学习场景。

**📈 对比分析**

与 Prompt rewriting（CoT、paraphrasing）、Self‑Refine、R.I.P. 等基线对比，采用 DPO/SimPO 训练后在 MT‑Bench、Evol‑Instruct、AlpacaEval 等指标上测评。相较基线，指令改进可提升 3–8%p（最高 8%p）并在多模型、多目标、多基准上保持稳健增益。

**⚠️ 局限性**

限制主要包括：依赖奖励模型的准确性与偏差，阈值设置敏感导致某些场景性能下降；改进流程增加 API 调用成本；目前仅在文本 LLM 上验证，尚未扩展到多模态；过度改写可能削弱回复多样性与信息量。

---

## 22. AraMS-28k: The Largest Publicly Released Line-Level Dataset of Historical Arabic Manuscripts with Margin and Insertion-Anchor Annotations

**arXiv ID:** 2608.26921 | [PDF](https://arxiv.org/pdf/2608.26921v1)

**作者:** Mohamed Guechaoui `[一作]` (Higher School of Computer Science), Sahraoui Dhelim `[通讯]` (Higher School of Computer Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文创建了AraMS-28k数据集，收集了14本真是历史阿拉伯手稿的3,043页，提供了逐行的主文本与边注标签，并为每行提供完全带发音符号和去发音符号的两种转录。

**💡 创新点**

创新点在于首次公开发布带有行级插入锚点的阿拉伯手稿语料，能够恢复主文本与边注的非线性阅读顺序，并同时提供与图像视觉一致的去发音转录，解决了传统语料缺乏边注定位与发音符号不匹配的问题。

**🔧 技术方法**

采用RefLAM构建流程：多模态LLM OCR + 参考文本对齐 + 人工复核；随后在该语料上 fine‑tune Kraken 与 HATFormer 等 HTR 模型进行基线实验。

**📊 数据集**

使用的数据集为14本手稿（13本手写，1本石版印刷），涵盖 Naskh、Ruq‘ah、Maghrebi 三种书体以及边注，含 28,600 行文本。

**📈 对比分析**

通过在 9 本书上 fine‑tune 的 Kraken 与 HATFormer，在 3 本书（每种书体各一）上评估字符错误率（CER）。结果显示：Ruq‘ah 最高 11.65%/13.26%，Naskh 22.62%/25.37%，Maghrebi 32.71%/37.88%；在分布内页测试时 CER 可降至 6.48%。

**⚠️ 局限性**

局限性包括：边注插入锚点仅覆盖约 30% 的边注行；行级验证子集受 OCR 与参考文本一致性的影响，偏向扫描质量较好的样本；边注行仅占整体 2.2% 之低，且测试集每种书体仅 1 本书，可能与书本特性相关。

---

## 23. Parameter Efficient Continual Learning for Sparse Event-Based Transformers

**arXiv ID:** 2608.26720 | [PDF](https://arxiv.org/pdf/2608.26720v1)

**作者:** Vaishnavi Nagabhushana `[一作]` (IIT Guwahati), Ayon Borthakur `[通讯]` (IIT Guwahati)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在稀疏事件感知Transformer上无重放的持续学习框架，提出sLoTh，通过冻结主干并仅更新低秩注意力模块和通道阈值实现参数高效适应。

**💡 创新点**

创新点：结合阈值调制与可共享低秩注意力适配（seLoRA）的参数高效持续学习框架，支持稀疏事件Transformer；不需要回放或任务边界；能在能耗和内存受限环境下实现多达100任务的学习。

**🔧 技术方法**

使用技术：参数高效微调（PEFT）中的低秩适配LoRA（seLoRA）、通道级兴奋阈值调制、稀疏事件Transformer（QKFormer、SpikFormer）、多目标蒸馏与正则化、在线异常检测。

**📊 数据集**

数据集：CIFAR-100、Tiny-ImageNet、ImageNet-100、ImageNet-R，构成10/20/50/100任务分割。

**📈 对比分析**

比较方法：对比多种PEFT和回放型持续学习方法（L2P、DualPrompt、CODA-Prompt、SD-LoRA、Online-LoRA、MOE-MOSE等），在CIL、TIL、OCL场景下，sLoTh在保持参数更新<1%且能耗6.5倍降低的前提下，平均准确率与最佳回放方法相当或略优，尤其在任务数增多时表现更稳健。

**⚠️ 局限性**

局限性：对稀疏事件Transformer的依赖，阈值调制在单次流式学习中对分布漂移检测不够鲁棒；在高任务数时仍有代表性漂移；需要进一步改进漂移检测与跨任务迁移机制。

---

## 24. SPT: Skills as Pre-Training Data for Agentic Language Models

**arXiv ID:** 2608.26563 | [PDF](https://arxiv.org/pdf/2608.26563v1)

**作者:** Yufei Sun `[一作]` (Beijing University of Posts and Telecommunications), Yiming Cheng `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了在大型语言模型的中间训练阶段使用公开的多文件技能包（SkillCorpus）进行因果语言建模，以提升模型在工具使用和多步推理等代理任务上的表现。

**💡 创新点**

创新点包括：1) 将可重用的多文件技能包视为预训练数据源；2) 引入 Reference Insert 参考插入策略，使技能文件在序列化时保持跨文件引用的局部性；3) 通过混合比例实验探究技能数据与通用文本的最佳组合；4) 证明在不同后训练流程（SFT、SFT+RL）下，技能预训练均能显著提升代理能力。

**🔧 技术方法**

主要技术手段：因果语言模型的中间训练（SPT）、参考插入文件序列化、技能数据与通用数据的比例混合、与常规通用数据（Dolmino、SmolLM）以及轨迹数据（AgentBank）的对照实验、后训练中的功能调用微调（Tulu 3 / xLAM-FC）以及群组相对策略优化（GRPO）强化学习。

**📊 数据集**

使用的数据集：38,040 个从 ClawHub 收集并清洗的公开技能包（SkillCorpus），包括 218,277 个文件；通用预训练数据集（Dolmino、SmolLM）作为对照；轨迹数据集（AgentBank）用于比较；工具使用基准（API‑Bank、MetaTool、APTBench、ToolEyes）以及通用能力基准（ARC、BoolQ、HellaSwag、PIQA、WinoGrande、MMLU）。

**📈 对比分析**

比较方法：在 1.6B、3B、7B 三种基础模型上，先进行中间训练（无、中间训练为通用数据、轨迹数据、技能数据），随后分别进行功能调用微调或通用指令微调，并可进一步加入 RL；评估指标为工具使用基准和通用基准的平均分。性能表现：技能预训练在所有模型和微调方式下均超过通用数据和轨迹数据的中间训练，代理任务平均提升 9–25 分，通用任务几乎保持不变；在混合比例实验中，30% 技能数据即可获得最大代理提升，同时保持通用分数接近单一通用数据基准；在 RL 后训练实验中，技能预训练仍能保持明显优势。

**⚠️ 局限性**

局限性：1) SkillCorpus 仅来自 ClawHub，语言、工具生态和任务类别偏向该社区，低资源语言和专业领域覆盖不足；2) 仅在 1.6B–7B 模型及特定训练预算上验证，未探究更大模型或不同算力场景；3) 评价基准未覆盖工具选择、参数构造、错误恢复或长期交互等细粒度失败模式；4) 技能包中可能包含仍未被完全过滤的敏感或不安全指令；5) 仅在单一公开仓库构建，未评估跨仓库、跨语言等多样性对结果的影响。

---

## 25. The Accuracy-Efficiency Paradox Quantifying Net Energy Loss in on-Device Energy Forecasting

**arXiv ID:** 2608.26134 | [PDF](https://arxiv.org/pdf/2608.26134v1)

**作者:** Jaeik Jeong `[一作]` (ETRI), Wan-Ki Park `[通讯]` (ETRI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的总拥有成本（TCO）框架，将能量预测误差、推理能耗和电池老化成本融合评估边缘设备能量预测的净能效；

**💡 创新点**

创新点在于将电池老化视为可量化的能量损失，并用Arrhenius方程与推理能耗共同构成单一能量损失指标，揭示了“准确性‑效率悖论”；

**🔧 技术方法**

利用能量预测误差惩罚、推理能耗和电池老化成本三项构建TCO公式，并通过实验测量推理能耗、热量产生及电池老化因子；

**📊 数据集**

采用韩国AI‑Hub公开的叶州市住宅能耗时间序列（2021年9月至2022年8月，小时级，8760个样本）；

**📈 对比分析**

通过比较线性回归、MLP和Transformer三种模型的TCO随环境温度灵敏度k变化的曲线，实验显示Transformer在低k时能耗低误差最优，但在高k时因电池老化导致TCO大幅上升，MLP在中等k时表现平衡；

**⚠️ 局限性**

局限性包括：热学模型和Arrhenius参数简化，推理能耗测量仅基于功率估算，实验仅涵盖单一小时预测任务和有限模型，未对多步预测、多站点或联邦学习等更复杂场景进行验证。

---

## 26. Constraint-Aware Physics-Informed Neural Networks for Static Shape Estimation of Co-Manipulative Continuum Robots

**arXiv ID:** 2608.26273 | [PDF](https://arxiv.org/pdf/2608.26273v1)

**作者:** Rana Danesh `[一作]`, Farrokh Janabi-Sharifi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种约束感知的物理信息神经网络（PINN），用于闭链共操纵连续体机器人的静态形状估计，并通过实验验证其可行性。

**💡 创新点**

创新点在于将静态平衡投影残差与闭链几何残差同时嵌入网络损失函数，实现对未知闭链反作用力的自动抵消和闭环几何约束的严格执行；此外，还提出了从仿真预训练到实验微调的跨域迁移框架。

**🔧 技术方法**

使用了几何变量应变（GVS）模型、投影平衡残差、闭链几何残差、全连接前馈神经网络、自动微分、Adam/AdamW优化器、以及Vicon视觉捕捉和Dynamixel伺服驱动。

**📊 数据集**

训练数据集包括：14,641条仿真输入/输出对（四个肌腱拉力向量与16维GVS坐标），以及 817 条实验静态配置（四个肌腱位移与八个Vicon标记位置）。

**📈 对比分析**

与纯数据驱动的ANN以及传统非线性迭代求解器相比，PINN 在噪声标注下相对配置误差可降至 67% 左右，物理残差降至 88%；在仿真中，PINN 推理时间仅 0.18 ms，而求解器需 17.97 s；实验微调后，标记 RMSE 从 2.657 mm 降至 0.497 mm，R² 由 -0.788 提升至 0.937。

**⚠️ 局限性**

局限性包括：仅处理静态平衡问题，未考虑惯性与阻尼；依赖 GVS 模型精度，若模型误差增大，物理约束约束力弱化；目前仅验证单一闭链结构，扩展到更复杂几何或不同肌腱布局需进一步研究。

---

## 27. Style as a Confound: False Positives in AI Detection of Non-Native Academic Writing

**arXiv ID:** 2608.26710 | [PDF](https://arxiv.org/pdf/2608.26710v1)

**作者:** Hyeonchu Park `[一作]` (Chung-Ang University), Bugeun Kim `[通讯]` (Chung-Ang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了专业学术英语编辑对AI文本检测器输出的影响，使用了超过135,000个原文-编辑对，系统评估了13种检测器。

**💡 创新点**

创新点在于采用配对文档设计，隔离内容与作者，仅观察编辑导致的语言变化对检测结果的影响，揭示编辑是检测器误判的重要混杂变量。

**🔧 技术方法**

使用了三类公开AI检测器：基于token统计、零样本/LLM、监督式分类器，并计算了FPR、score变化、编辑强度相关性等指标。

**📊 数据集**

数据集为Wordvice学术编辑服务收集的2018-2025年非母语作者原稿与其专业编辑版，共135,389对文档。

**📈 对比分析**

通过与编辑前后的对照及编辑强度分层，比较检测器的假阳性率变化和分数漂移；结果显示不同检测器表现差异显著，编辑可导致score升高或下降，且变化与编辑强度正相关。

**⚠️ 局限性**

局限在于只涵盖非母语学术写作、未包含AI生成文本的编辑对、未针对每个检测器做微调、阈值设置影响绝对FPR；因此不能完全区分文本来源与写作风格。

---

## 28. SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning

**arXiv ID:** 2608.26550 | [PDF](https://arxiv.org/pdf/2608.26550v1)

**作者:** Zhuochun Li `[一作]` (University of Pittsburgh), Daqing He `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在基于强化学习的知识蒸馏中，引入了 SPEAR 方法，提供一种训练免费、可插拔的符号过程奖励，帮助小语言模型在序列层面对齐教师的推理轨迹。

**💡 创新点**

创新点在于将自然语言推理轨迹投射到领域适配的符号里程碑，并使用最长公共子序列（LCS）进行顺序感知对齐，从而得到密集且无需外部神经验证器的过程奖励。

**🔧 技术方法**

使用了规则化的符号锚点抽取（正则、依存句法）、LCS‑F1 计算、格式门控与奖励组合，并结合 GRPO、Dr.GRPO、DAPO 等 RL 算法进行训练。

**📊 数据集**

在 GSM8K、MATH、GPQA、CommonsenseQA 等数学、科学和常识推理数据集上进行实验。

**📈 对比分析**

与 SFT、稀疏奖励 RL、Logic‑RL 以及神经过程奖励模型（如 Qwen2.5‑Math‑PRM、VersaPRM）比较，SPEAR 在所有任务上均提升 1–3% 的准确率，同时仅使用约 12 MB 的轻量级抽取器，显著降低了计算成本。

**⚠️ 局限性**

局限在于对结构差异的推理路径可能获得较低奖励、依赖规则抽取对分词/解析错误敏感、无法捕捉语义等价但结构不同的推理，并且仅针对序列级 RL，未探索 token‑level 逆 KL 等方向。

---

## 29. Co-Evolving Structured Knowledge and Reasoning in Language Models

**arXiv ID:** 2608.26386 | [PDF](https://arxiv.org/pdf/2608.26386v1)

**作者:** Ryan Thomas Noonan `[一作]` (Cornell University), Jennifer J. Sun `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种共进化框架，在知识库构建与多跳推理之间共享参数，并通过QA结果奖励共同优化二者。

**💡 创新点**

创新点在于将知识库构建视为可学习的、与推理过程耦合的模块，使推理失败能够直接指导知识库改进，突破传统静态知识库的局限。

**🔧 技术方法**

采用GRPO（群体相对优势策略梯度）强化学习、Qwen3语言模型、工具调用检索以及SFT预训练等技术，实现端到端的共进化学习。

**📊 数据集**

主要使用HotpotQA、MuSiQue、2WikiMultiHopQA、PopQA等多跳/单跳QA数据集，并在ConFiQA-MR上测试知识编辑效果。

**📈 对比分析**

与Direct、RAG、IRCoT、Search‑R1等基线对比，-GRPO在HotpotQA平均EM提升5–10点，整体性能与Search‑R1持平，且在MuSiQue和2Wiki上取得最佳成绩。

**⚠️ 局限性**

局限包括：需要SFT预热才能进入结构化推理阶段；奖励仅覆盖检索到的事实，无法保证完整KB的真实性；易受奖励挫折风险；对更大模型或更强提示的适用性仍待验证。

---

## 30. Neuro-symbolic PRM: Enhancing Scientific Reasoning via Structured Traces and Symbolic Verification

**arXiv ID:** 2608.26329 | [PDF](https://arxiv.org/pdf/2608.26329v1)

**作者:** Yuxin Zi `[一作]` (AI Institute of South Carolina), Amit Sheth `[通讯]` (AI Institute of South Carolina)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种神经‑符号框架（NS‑PRM），将推理分为符号有效性与语义根基两层：用确定性符号验证器做硬过滤，剩下的可执行步骤再由训练好的 Process Reward Model（PRM）进行软排序；通过 Counterfactual Symbolic Perturbation（CSP）合成仅通过验证但语义错误的难负样本，专门训练 PRM 关注逻辑真相。

**💡 创新点**

创新点在于：①把符号合法性与语义正确性彻底解耦，避免 PRM 同时学习算术与语义；②引入 CSP 在训练时自动生成对抗式约束保持负样本通过符号验证但语义错误；③在推理时采用“验证器‑优先”束搜索，先剔除无效候选再调用 PRM，从而大幅提升推理效率与准确率。

**🔧 技术方法**

技术手段包括：结构化推理表示（JSON Schema + 120+ 数学原语）；确定性符号验证器（使用 Pint 进行单位一致性检查；执行相等性检查）；CSP 生成器（逻辑公式替换与等单位变量互换）；PRM 训练（margin‑ranking loss；Qwen2.5‑Math‑7B 预训练模型；线性分类头）；Verifier‑First Beam Search；基准模型 Qwen2.5‑Math‑7B‑Instruct。

**📊 数据集**

使用的数据集主要有：ProcessBench、PRMBench、MATH、GSM8K、OlympiadBench、OmniMATH、GPQA、SciBench；训练集来自 PRM800K，采样 140k 正例与 140k 负例；验证集为 ProcessBench 与 PRMBench 的多项子任务。

**📈 对比分析**

与多种基线（Math‑Shepherd‑7B、RLHFlow‑Mistral‑8B、Skywork‑PRM‑7B、Qwen2.5‑Math‑PRM、R‑PRM‑7B 等）在 ProcessBench、PRMBench、MATH 等任务上对比，NS‑PRM（Verifier+PRM+CSP）在 MATH pass@8 从 78.5% 提升至 83.1%，在 ProcessBench 上 F1 由 68.5% 提升至 74.0%；在 PRMBench 的 Soundness 与 Sensitivity 上均显著提升，均达到 98% 以上。相对基线提升 2‑4% 甚至更高。

**⚠️ 局限性**

局限性包括：①需要先验的领域特定结构化语法与操作库，维护成本高；②对隐式假设或开放式工程问题的鲁棒性不足；③在几何等抽象空间中效率提升有限；④完全依赖符号验证器可能忽略某些非结构化计算错误。

---

## 31. Benchmarking_Fast_Domain_Adaptation_for_Unsupervised_Speech_Units

**arXiv ID:** 2608.26992 | [PDF](https://arxiv.org/pdf/2608.26992v1)

**作者:** Robin San Roman `[一作]` (PSL university), Emmanuel Dupoux `[通讯]` (PSL university)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向口音适应的无监督语音表示学习基准，评估模型在少量无标签口音数据上的快速自适应能力。

**💡 创新点**

首次将ABX评估指标扩展到10种英语口音，并引入自适应域归一化技术以提升跨域性能。

**🔧 技术方法**

采用对比预测编码（Contrastive Predictive Coding, CPC）模型，并在其卷积层中嵌入自适应域归一化层；在微调阶段使用重采样与域归一化热启动策略。

**📊 数据集**

使用AESRC数据集（10种英语口音，每种约10小时无标签训练数据）以及LibriSpeech作为预训练源域。

**📈 对比分析**

与未适应的基线相比，基线模型在ABX跨说话人评估中平均提升23.6%，在单口音微调时可实现约33%的相对改进。

**⚠️ 局限性**

方法需要已知域标签进行归一化，且在极端口音（如俄语）下仍难以完全弥补域差距，表明仍有提升空间。

---

## 32. PailitaoGR: Latent Think-with-Images for Generative Image Retrieval

**arXiv ID:** 2608.26658 | [PDF](https://arxiv.org/pdf/2608.26658v1)

**作者:** Xiaomeng Fan `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种生成式图像检索模型 PailitaoGR，能够在不裁剪、无需 OCR 的情况下从完整查询图像中识别搜索目标并利用辅助证据；

**💡 创新点**

通过“潜在 Think-with-Images”实现目标聚焦与辅助证据选择的能力内部化，使用目标增强、辅助增强以及 on-policy 蒸馏、ROT 与熵引导等训练目标；

**🔧 技术方法**

使用 Qwen3.5-0.8b 生成器、ViT 视觉编码器、SID 语义标识、Crop Teacher、OCR Teacher、目标增强模块、辅助增强模块、对比蒸馏等技术；

**📊 数据集**

基于阿里巴巴淘宝 Pailitao 的真实线上图像检索日志构建的训练集（约115万张）和验证集（约8647张），覆盖七大商品子类；

**📈 对比分析**

与传统相似度检索（DINOv3、CLIP）、现有生成式检索（IRGen、GENIUS）以及 Crop/OCR Teacher 进行对比，PailitaoGR 在 H@10、R@10 等指标上平均提升 13.8% 以上，超过 Crop Teacher 约 4.8% 与 OCR Teacher 约 2.8%；

**⚠️ 局限性**

仍受限于视觉分辨率与模型容量，部分细粒度信息可能被低分辨率或遮挡导致无法充分利用；对极端噪声或多目标场景的鲁棒性尚待进一步验证。

---

## 33. Trusting AI in Competitive Markets

**arXiv ID:** 2608.26539 | [PDF](https://arxiv.org/pdf/2608.26539v1)

**作者:** Jussi Keppo `[一作]` (National University of Singapore), Nuo Yuan `[通讯]` (City University of Hong Kong)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在实验室Bertrand定价游戏中比较有无AI价格建议对三卖家市场价格与利润的影响，并检验性别组成如何调节AI的依赖与市场协调

**💡 创新点**

揭示AI建议在全女性市场显著提高价格与利润，而在全男性或混合市场无显著效果，表明性别对AI信任与归因机制产生不同动态

**🔧 技术方法**

采用实验室实验、非齐性隐藏马尔可夫模型（NHMM）和文本分析技术来建模AI遵从、动态转移与解释文本

**📊 数据集**

使用273名大学生（约一半女性）组成91个三人市场，共计2430轮的实验数据

**📈 对比分析**

通过回归与NHMM对比，发现女性市场AI干预使价格提升约29%、利润提升约39%，其他组合无显著效应，验证了性别调节效应

**⚠️ 局限性**

仅在单一AI推荐、无交互、实验室条件下，且样本为学生群体，结果可能不完全适用于真实市场与更复杂的AI交互场景

---

## 34. VIPER: An Expert-Curated Benchmark for Vision-Language Models in Veterinary Pathology

**arXiv ID:** 2608.26382 | [PDF](https://arxiv.org/pdf/2608.26382v1)

**作者:** Luca L. Weishaupt `[一作]` (Harvard-Mit Hst), Guillaume Jaume `[通讯]` (University Of Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了VIPER——首个专门针对毒理学动物组织的专家标注视觉‑语言基准，并基于此开发了两款针对小鼠毒理学的视觉‑语言模型（Qwen3.5‑27B‑VM 与 Gemma4‑VM）。

**💡 创新点**

①首次在非人类病理领域提供高质量、专家驱动、视觉根基的问答基准；②引入人机协作与对抗性过滤机制，显著降低模型仅靠文本先验的误判；③证明领域特定的微调能在多种大模型上带来显著提升，揭示人类病理模型与动物病理之间的性能鸿沟。

**🔧 技术方法**

使用视觉‑语言模型技术（多模态预训练 + 低秩适配 LoRA），结合 GPT‑5.2 生成与改写问题、对抗性过滤；在评测时采用 GPT‑5.4 作为自由文本判定器并加入诊断与完整度加权评分。

**📊 数据集**

基准数据：419 张 H&E 染色的小鼠组织 ROI，覆盖七个器官系统，含 1,251 个多格式问题（MCQ、KPrim、自由文本）。训练数据：约 345,000 个病理 ROI 与 4,000,000 条基于公开文件的指令‑图像对，用于对两款专用模型的微调。

**📈 对比分析**

通过统一的提示与生成协议，对比 2 款动物专用模型、7 款人类病理专用模型、8 款通用前沿模型（包括 GPT‑5.4、Gemma4、Claude Sonnet 等）。评测指标为 MCQ 准确率、KPrim 半点规则分数、自由文本 LLM 判定分数及整体加权分。动物专用模型平均 62–67%（最高 67%），人类专用模型 51–58%，通用模型 55–60%；在人类病理基准 PathMMU 上，动物专用模型依旧保持与通用模型相近的 68% 级别，显示出良好的跨域迁移。

**⚠️ 局限性**

①仅为 ROI 级别，未覆盖全切片与剂量组比较；②器官与物种范围有限（缺少脑、脊髓等）；③数据量相对较小，需谨慎解读细分统计；④自由文本评分主观性较大，即便使用 LLM 判定也可能引入噪声。

---

## 35. A Geometry-Driven, Framework-Agnostic Optimization for Object Pose Estimation

**arXiv ID:** 2608.26859 | [PDF](https://arxiv.org/pdf/2608.26859v1)

**作者:** Wei Chen `[一作]`, Erwei Yin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种数据中心的几何优化方法，通过将物体坐标系对齐到其惯性主轴来改进6D位姿估计

**💡 创新点**

核心创新在于：①不改网络结构，仅在数据预处理阶段对姿态标签进行重基准化；②利用主轴对齐天然消除对称物体的旋转歧义；③实现了模型无关、低成本的“即插即用”方案

**🔧 技术方法**

主要技术包括惯性张量求特征值分解获得主轴、构造对齐旋转矩阵、在数据集上变换姿态标签、对称性检测与解析式消歧

**📊 数据集**

使用了LINEMOD、LINEMOD‑OCC、NOCS‑REAL、TLESS等公开数据集进行实验

**📈 对比分析**

与多种主流方法（G2L‑Net、GDR‑Net、LC、FS‑Net 等）对比，平均旋转误差下降 20–30%，在对称物体上的误差降幅更显著，且方差显著减小

**⚠️ 局限性**

限制在于无法一次性消除所有高对称多面体（如八面体、十二面体）的完整对称群；对极端噪声或复杂背景时仍需额外处理

---

## 36. Data Science Approaches to Evaluating Honours Candidates

**arXiv ID:** 2608.26135 | [PDF](https://arxiv.org/pdf/2608.26135v1)

**作者:** Francesca von Braun-Bates `[一作]` (Ministry of Justice), Anirban Lahiri `[通讯]` (Kainos)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对英国荣誉系统获奖者的公开舆情进行评估，构建了一套基于自然语言处理的自动化审查方法，用以辅助人类评审。

**💡 创新点**

创新点在于提出并实现了一种专门为荣誉评审定制的情感分析算法，并将贝叶斯推理与多源信息融合，形成可量化的评分体系。

**🔧 技术方法**

使用了网页抓取、分词、共指消解、情感分析以及贝叶斯推理等多种 NLP 技术，并将其集成到统一的数据处理框架中。

**📊 数据集**

数据来源于公开的网络文章和新闻报道，约为十亿字符（相当于《哈利波特》系列长度）的文本。

**📈 对比分析**

与两种现有情感分析算法对比，自研算法在不同受众群体的情感分布上准确率提升约10%，并在多样化文本上表现更为稳健。

**⚠️ 局限性**

局限性包括数据获取受限、网络信息存在偏见、算法过度依赖公开文章，以及最终决策仍需人工审核。

---

## 37. Risks and Controls for Multi-Agent Systems: an analytical framework for deployment of AI agents across organisational boundaries

**arXiv ID:** 2608.26626 | [PDF](https://arxiv.org/pdf/2608.26626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 38. Systematic Literature Review of Machine Learning Models and Applications for Text Recognition

**arXiv ID:** 2608.26500 | [PDF](https://arxiv.org/pdf/2608.26500v1)

**作者:** Nuzhat Khan `[一作]`, Shahidatul Sadiah `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2015-2025 年 97 篇 OCR 研究进行系统综述，梳理 AI 模型演进、脚本覆盖、应用领域及主要挑战。

**💡 创新点**

首次采用 PRISMA 方法构建完整文献筛选流程，系统评估多类别模型（传统 ML、CNN、LSTM、GAN、Transformer、混合模型）及其跨语言表现，并提出多维改进路径。

**🔧 技术方法**

使用系统综述、数据提取模板、趋势分析、对比表格和关键词词云等技术，对 7 大数据库检索结果进行可视化与归纳。

**📊 数据集**

分析的研究涵盖多种公开数据集（如 MJSynth、IAM、BanglaLekha‑Isolated、CMATERdb、ICDAR 等），但本综述并未收集新数据，仅总结各研究使用的数据来源。

**📈 对比分析**

通过对 97 篇文献的模型性能对比，指出 Transformer 系列在多语言、手写及复杂布局中表现最佳（准确率 77–99.98%），CNN 与 LSTM 在资源受限场景仍具优势；同时揭示训练规模、算力与实时性之间的权衡。

**⚠️ 局限性**

局限性包括：仅纳入英文同行评审论文，排除灰色文献与非公开数据集；研究范围受时间窗口（2015-2025）限制；对各模型的性能比较受数据集多样性与实验设置差异影响，无法给出统一的最优模型结论。

---

## 39. LLMs for Academic Workflows: An Evaluation of Literature Reviews Generated with Short and Long Context Windows of LLMs

**arXiv ID:** 2608.26145 | [PDF](https://arxiv.org/pdf/2608.26145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 40. Exploring the Role of LLMs in HPC Programming: A Survey

**arXiv ID:** 2608.26110 | [PDF](https://arxiv.org/pdf/2608.26110v1)

**作者:** Strahinja Ljaljevic `[一作]` (Universitat Oberta de Catalunya), Sergio Iserte `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了大语言模型（LLMs）在高性能计算（HPC）编程中的应用，系统划分为代码生成、并行化与优化、框架与架构、评估与基准以及挑战与前景等五大主题，评估了多种LLM技术与方法在HPC中的表现。

**💡 创新点**

创新点在于：①提出了完整的五维分类框架，统一评估LLMs在HPC中的多维能力；②系统梳理了最新的领域专用模型（如HPC-Coder、HPC-GPT、chatHPC等）及其技术路线；③阐明了评估指标与基准（ParEval、ParEval-Repo、PCEBench等）的设计思路；④结合实验与案例识别了LLMs在MPI、GPU及能源效率等关键子领域的瓶颈，并给出可行的研究方向。

**🔧 技术方法**

主要技术包括：大语言模型基础（GPT‑4、LLaMA‑2、StarCoder等）、领域微调与指令微调（HPC‑INSTRUCT）、检索增强生成（RAG）、多代理（Agentic）与图神经网络辅助提示、DSL与迭代反馈循环、以及LLM与编译器、性能分析器的闭环集成。

**📊 数据集**

使用的数据集：公开的HPC代码仓库、由研究者构造的HPC‑INSTRUCT合成数据、ParEval与ParEval‑Repo基准集、CodeBLEU与PCEBench评价数据、以及各类实测的CPU/GPU性能与能耗数据。

**📈 对比分析**

比较方法：对LLMs的生成代码进行编译成功率、功能正确性、单核/多核加速比、并行效率等多维度测评。实验显示：域专用模型在OpenMP、CUDA和部分MPI任务中可达≈97%正确率，且在小规模基准上实现2–4倍的速度提升；但在MPI、异构混合模型和大型应用级别时仍存在低效率、规模不佳的问题；总体性能相较于人工优化存在显著差距，尤其在能源和可扩展性方面。

**⚠️ 局限性**

局限性包括：①训练数据缺乏高质量并行代码，尤其是MPI和异构GPU；②模型对分布式语义的泛化能力弱；③上下文窗口受限，难以处理多文件大规模项目；④缺乏对硬件拓扑、NUMA和内存层次的感知，导致优化不够精细；⑤评估基准易受训练数据重叠影响；⑥需要人工干预与循环反馈，缺乏完全自治的生产级流水线。

---

## 41. Mutual Debiasing via Dual-Seed Comparison for Probabilistic Sampling in Large Language Models

**arXiv ID:** 2608.26161 | [PDF](https://arxiv.org/pdf/2608.26161v1)

**作者:** Zihao Guo `[一作]` (Shandong University), Lizhen Cui `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM生成随机数的偏差，提出Dual-Seed Comparison (DSC)协议，通过比较两条独立种子字符顺序生成伪均匀变量，再映射到目标分布，实现无工具的概率采样。

**💡 创新点**

创新点在于使用双种子比较消除单字符串偏差，构建透明的多步算术流程，并直接通过逆CDF映射到任意分布，显著提升LLM在连续分布采样中的KS/Wasserstein误差。

**🔧 技术方法**

使用技术包括LLM自生成字符串、字符序号比较、二进制转整数、归一化为[0,1)伪均匀随机数，以及逆CDF采样推理。

**📊 数据集**

实验数据集涵盖五种大模型（Claude, Gemini, MiniMax, Qwen3.5-27B, Qwen3.5-9B）在五种目标分布（均匀、正态、指数、Beta、Gamma）以及两项下游任务（MCQ生成与属性约束文本到图像提示生成）。

**📈 对比分析**

与直接采样和String Seed of Thought (SSoT) 基线对比，DSC在48/50的KS/Wasserstein指标中表现最佳，平均误差降低约58.5%，在多属性生成任务中接近理想均匀分布且统计检验不拒绝。

**⚠️ 局限性**

局限性包括对LLM算术与逆CDF推理能力的依赖、对复杂或罕见分布支持有限、生成过程需多步骤导致延迟与成本提升，以及不具备安全随机生成器特性。

---

## 42. DPA-I2P: Depth-Guided Projective Alignment for Image-to-Point-Cloud Registration in Autonomous Driving

**arXiv ID:** 2608.26589 | [PDF](https://arxiv.org/pdf/2608.26589v1)

**作者:** Wenxin Zhang `[一作]` (Nankai University), Tao Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种深度引导投影对齐（DPA-I2P）框架，利用隐式对应学习实现图像与点云的高精度位姿估计。

**💡 创新点**

核心创新在于将射线条件化的度量深度编码（RMDE）、投影一致的视觉提升（PVL）以及跨模态查询修剪（CQP）融合进单一端到端网络，有效提升跨模态特征一致性与匹配稳定性。

**🔧 技术方法**

采用单目度量深度估计器UniDepthV2提供深度先验，结合多尺度CNN+FPN特征提取、KPFCNN点云编码、RMDE、PVL、CQP以及可微PnP求解器完成完整的位姿回归。

**📊 数据集**

在KITTI和nuScenes两个大规模自动驾驶数据集上进行实验评估。

**📈 对比分析**

与传统PnP、显式对应、隐式对应基线（Grid Cls., DeepI2P, CorrI2P, VP2P‑Match, ICLI2P）对比，DPA‑I2P在KITTI上RTE降至0.11 m、RRE降至0.55°、准确率提升至99.70%；在nuScenes上亦显著优于同类方法，表现出良好的跨场景泛化。

**⚠️ 局限性**

局限性包括对单目深度估计的依赖、对稀疏LiDAR点云的敏感性，以及在极端遮挡或纹理缺失区域仍可能产生误匹配。

---

## 43. A Layer Importance Metric for Quantization Accounting for the Speed-Quality Trade-off in Autoregressive Models

**arXiv ID:** 2608.26926 | [PDF](https://arxiv.org/pdf/2608.26926v1)

**作者:** Artem Safronov `[一作]` `[通讯]` (South Federal University), Artem Safronov (South Federal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合SQNR信息保留和屋脊模型延迟预测的异构量化优先级指标，用于在小型语言模型中挑选对精度敏感且能带来加速的模块；

**💡 创新点**

创新点在于将信息质量和速度潜能归一化并通过可调优先级参数α融合，提供可解析、可调节的全局量化决策依据，避免昂贵的训练或搜索；

**🔧 技术方法**

采用SA‑PTQ量化、SQNR评估、屋脊（roofline）模型预测、归一化与加权平均；

**📊 数据集**

使用Gemma 3 1B、LLaMA 3.2 1B和Qwen 2.5 1.5B等小型LLM，并在Google Colab T4 GPU上进行推理基准；

**📈 对比分析**

通过与真实推理速度对比，屋脊预测误差平均约3.4%，在多模型上验证了量化后的速度提升与质量保持，且可通过调节α在速度与质量之间平衡；

**⚠️ 局限性**

局限在于只针对内存带宽受限的自回归推理，未考虑更复杂算子或算力受限场景，且预测误差受硬件调度和实现细节影响。

---

## 44. Categorizer Automata for Discounted-Sum Payoffs

**arXiv ID:** 2608.26763 | [PDF](https://arxiv.org/pdf/2608.26763v1)

**作者:** Nathalie Bertrand `[一作]` (Univ Rennes), Moshe Vardi `[通讯]` (Rice University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的分类器自动机，用来将折扣和的无穷序列按给定的区间划分。

**💡 创新点**

创新点在于构造了状态空间仅线性依赖于区间数的分类器自动机，显著避免了传统交叉积导致的指数爆炸。

**🔧 技术方法**

采用折扣和的间隔值（gap value）技术、状态压缩编码以及对折扣因子为整数的判定条件，结合 MDP 与自动机的积来实现策略合成。

**📊 数据集**

本文为理论研究，没有使用公开数据集，主要以数学构造与算法分析为主。

**📈 对比分析**

与指数规模的交叉积构造相比，新方法的时间与空间均为伪多项式（多项式于数值参数而非二进制长度），并证明了在三区间、整数折扣情况下问题仍为 P‑SPACE‑hard，说明无法进一步大幅降低复杂度。

**⚠️ 局限性**

局限性在于仅适用于整数折扣因子；若折扣为非整数，缺乏有效的分类器构造；且虽然算法是伪多项式，但在输入数值很大时仍会产生指数级时间消耗。

---

## 45. Activation Outliers Matter: Robust Recovery for Quantized Multimodal LLMs

**arXiv ID:** 2608.26581 | [PDF](https://arxiv.org/pdf/2608.26581v1)

**作者:** Tanzila Rahman `[一作]` (Huawei), Yaoyuan Wang `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对多模态大型语言模型（Wan2.2、Qwen3‑VL）进行 4‑bit 低位量化（MXFP4、HiF4）研究，并提出 Residual Fallback Quantization（RFQ）来恢复激活量化误差。

**💡 创新点**

创新点在于：① 发现激活量化是 4‑bit 低位模型性能下降的主因；② 设计了只在前向传递中使用的、无架构改动的残差补偿机制 RFQ；③ 通过块级残差量化实现与硬件 FP4 一致的高效运算。

**🔧 技术方法**

使用了 OCP 微尺度浮点格式（MXFP4、HiF4）与 BF16 的混合精度 QAT，RFQ 采用块级 FP4 量化 + FP4 残差量化；实验平台包括 NVIDIA 80GB GPU 与专用 FP4 加速硬件。

**📊 数据集**

对 Qwen3‑VL 采用 CC‑3M 训练集（约 595K 样本），对 Wan2.2 采用 OpenVid‑1M 1‑部分（约 26K 视频‑文本对）。评估使用 VBench、RealWorldQA、MMStar、MMBench‑EN、SimpleVQA 等基准。

**📈 对比分析**

与 BF16 基线及单纯 4‑bit 量化进行对比。RFQ 在 MXFP4 与 HiF4 下均能把训练损失降到 BF16 以内，并使 VBench 的动态度数、Aesthetic Quality 等指标提升 6–8 点；在 Qwen3‑VL 的多模态推理任务中，RealWorldQA、MMStar、SimpleVQA 的准确率分别提升 1–2%，接近或超过 BF16 基线。

**⚠️ 局限性**

局限性：仅验证 4‑bit 量化；对 2‑bit 或更低位数的可行性未知；仅在两款模型上实验，未覆盖更广泛的多模态架构；残差量化仅在前向阶段使用，后向梯度的数值误差未做深入分析。

---

## 46. Processing/p5 Defined through Practice and Learning

**arXiv ID:** 2608.26614 | [PDF](https://arxiv.org/pdf/2608.26614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 47. Decentralized Multitask Learning over Learned Task Graphs

**arXiv ID:** 2608.26989 | [PDF](https://arxiv.org/pdf/2608.26989v1)

**作者:** Zirui Wan `[一作]` (Imperial College London), Stefan Vlaski `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种两阶段去中心化多任务学习框架，先通过非合作随机梯度迭代学习任务关系的图拉普拉斯矩阵，再利用该图实现多任务扩散学习。

**💡 创新点**

创新点在于：①在完全未知的任务关系图上实现去中心化图学习；②利用GMRF先验将图学习转化为局部极大似然问题，给出有限样本误差上界；③引入拓扑敏感性指标量化网络异质性对图估计误差的影响；④分析图估计误差对多任务学习稳态性能的影响，揭示误差由O(μ)与O(μ′)两部分构成。

**🔧 技术方法**

采用高斯马尔可夫随机场先验、最大似然/极大似然图学习、协方差浓缩理论、Krylov方法构造局部拉普拉斯、扩散式随机梯度算法，并用理论分析推导误差上界。

**📊 数据集**

在合成线性回归数据上验证，使用不同网络拓扑（规则图、Erdős–Rényi、中心化图）和不同节点数、边权，模拟随机梯度噪声和非合作学习过程。

**📈 对比分析**

通过与已知真实图的基准、纯非合作和多阶段更新的比较，结果显示：学得图的多任务扩散实现的均方误差（MSD）显著低于非合作方案，且随着步长减小可逼近已知图基准；图学习误差随样本维度增大和步长减小时降低，拓扑异质性会导致误差上升。

**⚠️ 局限性**

局限性：需要足够小的学习步长和/或高维数据才能保证图估计误差可忽略；对高度不均匀网络（中心化图）敏感；若图更新间隔过短或正则化强度过大，协作可能放大误差导致不稳定；理论上限与实验结果存在一定偏差，需进一步针对动态任务关系做推广。

---

## 48. Kale: A Transformation-Safe Spreadsheet System

**arXiv ID:** 2608.26345 | [PDF](https://arxiv.org/pdf/2608.26345v1)

**作者:** Michael Coblenz `[一作]` (University of California, San Diego), Joanna Yang `[通讯]` (University of California, San Diego)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了Kale，一个限制引用类型的安全电子表格系统，并通过用户研究与语料库实验评估其错误降低效果。

**💡 创新点**

创新点在于禁止任意矩形范围引用并重新定义绝对/相对引用，使表结构变更时引用保持稳定，从而显著降低传统电子表格中因结构变更导致的错误。

**🔧 技术方法**

使用 TypeScript/React 开发 Web 应用，AG Grid 表格实现，ANTLR 解析器处理公式语法，内部维护依赖图和拖拽/排序等操作。

**📊 数据集**

采用 EUSES 电子表格语料库（1,726 个独特表格，随机抽取 60 个）以及 Google Sheets 任务表格进行实验。

**📈 对比分析**

通过 25 名受试者的对照实验（Sheets vs Kale）比较任务正确性、风险相关错误率和完成时间。实验表明 Kale 在 4 类风险上显著降低错误率（p < 0.01），任务完成时间无显著差异。

**⚠️ 局限性**

局限性包括：系统为原型，缺少跨表引用与查询功能；对表结构拆分对用户体验的影响未充分评估；未在开放式创作场景下测试其实用性。

---

## 49. Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval

**arXiv ID:** 2608.27006 | [PDF](https://arxiv.org/pdf/2608.27006v1)

**作者:** Ante Kapetanovic `[一作]`, Emanuel Lacic `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一种面向商家无关、可在实时电商目录上运行的多轮对话购物助手。

**💡 创新点**

创新点在于引入了自刷新检索器，仅对新增、变更或删除的商品进行增量更新，并将检索、重排与多样性选择等流程拆分为专门功能，保持LLM成本可控。

**🔧 技术方法**

核心技术包括：产品信息增量解析、哈希比较（完整哈希与语义哈希）、向量检索（ChromaDB）、LLM 进行意图分类与偏好询问、规则+生成式补充的属性丰富与嵌入生成。

**📊 数据集**

使用公开的 Google Merchant Center Atom 目录（约 500 条记录）进行增量同步演示，并通过 Infobip Answers 将对话部署在 WhatsApp 上。

**📈 对比分析**

在五种变更场景下测量同步耗时与全重建耗时比值，增量同步平均仅耗 1.8%–12.3% 的全重建时间，显著提高效率；但本文未对推荐质量进行离线或在线评估。

**⚠️ 局限性**

局限性包括：未完成离线相关性评估与真实用户研究、缺乏大规模并发/百万级商品同步的实测、写入非事务性导致需要额外的错误恢复机制、仅支持特定目录格式与单渠道演示。

---

## 50. Hyperspectral Diffusion Equivariant Imaging (HyDiff-EI): A Self-supervised Framework for Hyperspectral Image Inpainting

**arXiv ID:** 2608.26812 | [PDF](https://arxiv.org/pdf/2608.26812v1)

**作者:** Shuo Li `[一作]` (University of Edinburgh), Mehrdad Yaghoobi `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 HyDiff‑EI 自监督扩散框架，用于单张受损 hyperspectral 图像的自我学习式填补。

**💡 创新点**

创新点在于将等变成像（Equivariant Imaging）先验嵌入离散时间扩散模型，既无需大规模预训练，又能利用空间几何对称性提升恢复质量。

**🔧 技术方法**

使用了自监督等变网络、DDPM/DDNM 扩散概率模型、空间旋转/平移等对称变换约束以及范围/零空间分解的重建策略。

**📊 数据集**

实验基于 Chikusei、Botswana 和 EMIT 三个真实 hyperspectral 数据集。

**📈 对比分析**

与 DHP、R‑DLRHyIn、DDS2M、HIR‑Diff、SHARE 等方法在无噪声及多噪声场景下对比，HyDiff‑EI 在 MPSNR、MSSIM、SAM、SCC 等指标上均优于或接近最优，尤其在低至中等噪声水平下显著提升。

**⚠️ 局限性**

在高噪声情况下性能下降，需进一步引入噪声感知正则化；目前对光谱维度的专属等变变换尚未充分探索。

---

## 51. Double Trouble: Bilingual Pretraining Leaves Language-Conditioned Effects in Shared-Language Representations

**arXiv ID:** 2608.26576 | [PDF](https://arxiv.org/pdf/2608.26576v1)

**作者:** Anjishnu Mukherjee `[一作]` (George Mason University), Antonios Anastasopoulos `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比相同规模、相同超参的单语（只含英语）和双语（含英语和另一种语言）解码器模型，使用共享词表进行嵌入对齐后，再用持出的词在嵌入层和隐藏层做语义轴投影测量，检验双语训练是否改变了共享语言（英语）的内部表示。

**💡 创新点**

提出“对齐谬误”（alignment fallacy）：即使词嵌入对齐后仍可能存在隐藏层的不一致；通过控制英语曝光量、总训练步数、文档重叠等三大混杂因素，构建可比对照实验，展示隐藏层差异远超随机种子变异；首次在多个语言对上系统评估隐藏层几何差异。

**🔧 技术方法**

技术手段包括：1) 词嵌入和隐藏状态的正交 Procrustes 对齐；2) 语义轴投影差异（D_Axis）和最近邻差异（D_NN）等度量；3) 层级分析（12 层 transformer）与向量归一化、轴标准化等控制实验；4) 线性（正交）与仿射（affine）对齐比较；5) 持续 checkpoint 评估（每 500 步）。

**📊 数据集**

使用 BabyBabelLM 语料库，8 种非英语语言（中文、法语、波斯语、荷兰语、乌克兰语、保加利亚语、印尼语、德语）与英语混合训练；每种语言对应 100M 令牌，模型参数 310M；对齐 anchor 3000 个高频英语词，评估 1000 个持出词，采用 50 条语义轴（来源于跨文化调查框架）。

**📈 对比分析**

通过与仅差异随机种子的六组单语模型基准比较，计算 Δm = m(bilingual, monolingual) – ȳ_seed；实验显示：1) 隐藏层差异明显大于基准且持续存在；2) 隐藏层差异在中间层最大；3) 这一差异在 500 步已出现，随训练持续；4) 仅使用词嵌入对齐无法捕捉该差异；5) 在移除英语文档重叠或采用仿射对齐后差异仍保留。

**⚠️ 局限性**

局限性包括：仅研究 310M 参数的单一解码器架构，未检验更大规模或不同架构的普适性；使用的 50 条语义轴与翻译流程决定评估范围；未评估对下游任务表现的实际影响；对齐方法仅覆盖正交与仿射，未涵盖更通用的非线性对齐；缺乏因果机制解释。

---

## 52. RTNav: Towards Real-Time Zero-Shot Object Navigation

**arXiv ID:** 2608.26496 | [PDF](https://arxiv.org/pdf/2608.26496v1)

**作者:** Easop Lee `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了RTNav，一种支持实时异步推理的零-shot目标导航系统，并在 HM3D-v1、HM3D-v2 与 HM3D-OVON 三个基准上进行了评估。

**💡 创新点**

创新点在于提出了实时评估接口和异步模块化架构，将感知、映射、规划与导航分离运行，并按需调用 VLM，从而显著降低推理延迟，提高实时性能。

**🔧 技术方法**

技术手段包括使用 Owlv2 目标检测、Qwen3.5-9B VLM、MobileSAM 分割、共享状态总线、ROS2 2.0 中间件以及多线程并行执行的前沿探索策略。

**📊 数据集**

实验采用了 Habitat 提供的 HM3D-v1、HM3D-v2（6 类目标）和 HM3D-OVON（49 类新颖目标）数据集。

**📈 对比分析**

与多种基线（如 VLFM、GAMap、L3MVN 等）在同步与异步模式下对比，RTNav 在实时评估中成功率提升最高 11%，成功加权完成时间（SCT）提升 5.1 点，显著优于同期方法。

**⚠️ 局限性**

局限性包括仅在静态环境下测试，未考虑动态场景；异步设计对更紧密耦合的运动与操作任务适用性有限；并且仅在单 GPU 与部分边缘硬件上验证，未探讨多机器人协作或更复杂任务。

---

## 53. STREAM: An Objective-Driven and Uncertainty-Aware Framework for Industrial Energy Data Acquisition

**arXiv ID:** 2608.26754 | [PDF](https://arxiv.org/pdf/2608.26754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 54. Squeezing More from Limited Data with Recursive Transformers

**arXiv ID:** 2608.26973 | [PDF](https://arxiv.org/pdf/2608.26973v1)

**作者:** Serdar Gülbahar `[一作]` (Technical University of Munich), Alexander Fraser `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在仅有有限文本预算（10M–100M词）且计算资源充足的情形下，探索语言模型预训练的规模与性能关系，提出递归Transformer（RecursiveGPT）架构以解耦参数与计算深度。

**💡 创新点**

创新点在于：① 使用递归权重共享让模型通过重复相同的Transformer块实现更深的计算深度而不增加参数；② 采用分解式嵌入（factorized embeddings）削减词表映射占比；③ 通过这两项改进，在低数据预算下实现更优的性能，并提供一种新的“计算尺度”维度。

**🔧 技术方法**

技术包括：递归Transformer、因式分解嵌入、RMSNorm深度条件化、因子化语言模型头、标准Transformer与RecursiveGPT的对比实验、BabyLM Challenge评估与BLiMP、COMPS、LAMBADA等任务。

**📊 数据集**

使用数据集：BabyLM baseline 语料（10M、25M、50M、100M词子集）和 Nemotron-ClimbMix；评估基准包括 BLiMP、COMPS、LAMBADA、BLiMP Supplement、EWoK、Entity Tracking 等。

**📈 对比分析**

比较方法：在相同词数预算下对比标准Transformer（含与不含因式分解嵌入）与RecursiveGPT；同时与BabyLM 2025冠军模型（GPT-BERT causal、AMLM Hard Decay）对照。结果显示：在10M预算下RecursiveGPT在所有评测上均优于标准模型；在100M预算下，规模较小的RecursiveGPT可在BLiMP上超过1.22B标准模型，规模较大的RecursiveGPT在多数评测上均接近或优于基线，整体平均得分与BabyLM冠军相近。

**⚠️ 局限性**

局限性：① 超参数调优仅针对局部组合，未对每种数据、预算、尺寸、递归深度进行全面搜索；② 递归模型仅探索单块、固定深度且无自适应计算或深度调度，导致训练与推理成本随深度显著增加；③ 与BabyLM冠军相比，未结合其专门的训练目标与技巧，影响最终对比强度。

---

## 55. Evaluating human and LLM screening workflows in a conceptually complex scoping review: Recall--workload trade-offs and run-to-run consistency

**arXiv ID:** 2608.26885 | [PDF](https://arxiv.org/pdf/2608.26885v1)

**作者:** Nikol Figalová `[一作]` (Julius-Maximilians-Universität Würzburg), Anne Böckler-Raettig `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对概念复杂的scoping review标题‑摘要筛选任务，比较了人类（单一审稿人和分布式四名助手）与多种LLM（ChatGPT 5.4、Gemini 3/3.1）在完整工作流程下的操作回忆、保留工作负载、协议一致性、条件分类指标以及重复运行一致性。

**💡 创新点**

以完整筛选工作流程而非单一模型为评价单位，系统性考察处理配置、记录级不一致及人类与LLM输出的互补性，并提供多维性能评估框架。

**🔧 技术方法**

使用OpenAI ChatGPT 5.4与Google Gemini 3/3.1，采用文件批量、一次性批量和交互式10条记录批量等处理配置；人类流程包括单一审稿人和四名训练过的助理。

**📊 数据集**

基准数据集为1,131条经过预筛选的检索结果（原始搜索5,291条），其中316条最终被验证为合格，859条完成全文评估。

**📈 对比分析**

通过操作回忆、保留工作负载、κ与AC1一致性、条件分类指标（精确度、特异度、F1）以及两次相同配置的重复运行比较。结果显示，LLM文件批量与人类工作流程在保留率≈42–45%时操作回忆≈82%，Gemini 3.1文件批量回忆最高但保留率最高；重复运行出现≈8%记录级差异，单一工作流程无法恢复全部合格记录。

**⚠️ 局限性**

仅在单一概念复杂的scoping review中评估，未对提示、处理配置、模型版本进行因子化控制；参考标准受限于已推进记录，未验证未推进记录；LLM运行依赖专有接口，缺乏外部可复现性。

---

## 56. BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters

**arXiv ID:** 2608.26685 | [PDF](https://arxiv.org/pdf/2608.26685v1)

**作者:** Shuichi Miyazawa `[一作]` (AGC Inc.), Kensuke Fujii `[通讯]` (AGC Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了BLANC框架，用于识别专利景观中的白色空间（未被充分探索的技术组合）

**💡 创新点**

创新点包括：① 多视角BERTopic聚类（应用/新颖性/发明性），② 通过归一化点互信息（NPMI）量化跨维度关联，③ 引入NPMI下降（ΔNPMI）作为白色空间指标，并通过关键词条件检测

**🔧 技术方法**

使用技术包括：Sentence‑BERT、UMAP、HDBSCAN、BERTopic、NPMI计算、三维共现张量构建、条件NPMI与ΔNPMI评估

**📊 数据集**

数据集：公开USPTO HUPD数据集（G06N机器学习/AI 5,417份专利，C03C玻璃化合物 1,982份专利）以及302份浮法玻璃/玻璃陶瓷专利的专有数据

**📈 对比分析**

与单视角聚类、无归一化共现度量、CPC共分类等基线对比；在删除驱动实验中，BLANC在目标组合恢复率分别为34.1%（G06N）和27.3%（C03C），而随机或非目标删除控制下恢复率几乎为0，显示高特异性

**⚠️ 局限性**

局限性包括：NPMI在小样本子集下不稳定、依赖结构化字段与文本预处理、关键词选择需人工专业知识、评估仅基于合成删除与单一专家案例，缺乏广泛真实验证

---

## 57. Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models

**arXiv ID:** 2608.26580 | [PDF](https://arxiv.org/pdf/2608.26580v1)

**作者:** Insu Lee `[一作]` (Seoul National University), Byonghyo Shim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种名为 VIG-Sampler 的无训练并行解码策略，用于 diffusion 多模态大语言模型（dMLLMs），通过图像注意力信息优先解码视觉相关且信息量大的 tokens，并对相似注意力模式进行惩罚，减少信息冗余。

**💡 创新点**

创新点在于：① 利用每个被遮蔽 token 对图像 token 的注意力权重（image-attention mass）来重新加权置信度，优先解码视觉上更具信息的 token；② 引入 pairwise image-attention 相似度正则化，在选取 token 子集时抑制相似注意力分布，提升并行解码的整体信息增益；③ 该策略完全无训练、无需额外前向传播，直接使用模型自带的注意力矩阵。

**🔧 技术方法**

技术手段包括：Transformer 自注意力机制、图像注意力权重提取、信息增益度量、置信度重加权、相似度正则化、贪心子集搜索以及 diffusion 语言模型的多模态解码流程。

**📊 数据集**

使用了 7 个常用 captioning 与 VQA 评测数据集：COCO Caption、Flickr30K、CiteULike、SBU Captions、TextVQA、VQAv2、GQA 等，涵盖多模态文本生成与问答任务。

**📈 对比分析**

与六种基线（Confidence、Entropy、Margin、MPD-PAC、PC-Sampler、Info-Gain）以及三款开源 dMLLM（LaViDa、MMaDA、LLaDA-V）进行对比；在 captioning 任务上平均提升 19.3 个 CIDEr 分，VQA 任务上提升 7.3 个准确率；且在使用 8 个 token/步时，即使解码步数减少 50% 也能击败基线。

**⚠️ 局限性**

局限性包括：① 仅在具有显式图像注意力的模型上适用，无法直接迁移到不暴露注意力矩阵的模型；② 对超参数 γ、λ 的敏感度需手工调优，若设定不当可能影响性能；③ 在极低的 commit 预算（k=1、2）下性能提升有限；④ 仍依赖于原模型的视觉信息质量，若模型视觉预处理不足则效果受限。

---

## 58. When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems

**arXiv ID:** 2608.26895 | [PDF](https://arxiv.org/pdf/2608.26895v1)

**作者:** Hanchong Chen `[一作]` (Shenzhen Technology University), Xiuqiang He `[通讯]` (Shenzhen Technology University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种协同向量记忆（CoVeMem）方案，用冻结的LightGCN用户/项目状态作为持久记忆，并通过可学习的投影器与LoRA适配器将其软标记注入LLM上下文，利用目标感知检索与对比与遮蔽列表共训练学习读取方式，实现无额外LLM维护调用的agentic推荐。

**💡 创新点**

创新点包括：①用向量状态替代文本记忆，保留完整协同结构；②目标感知检索将候选相关历史状态投射进上下文；③通过语义锚点对齐与遮蔽列表共训练让LLM学会读取软标记；④记忆保持不再依赖LLM重写，显著降低维护成本。

**🔧 技术方法**

使用的技术包括：LightGCN训练协同向量；两层门控MLP投影器；LoRA低秩注意力适配器；对比式语义锚点对齐；遮蔽候选的列表共训练；点式是/否读取；在Qwen2.5-7B-Instruct基础上冻结LLM。

**📊 数据集**

实验数据集为四个InstructRec基准：Amazon Book、Goodreads、Amazon MovieTV 与 Yelp（均含用户指令、候选列表和隐藏目标）。

**📈 对比分析**

与传统推荐器（LightGCN、SASRec、P5）以及文本记忆agent（iAgent、i^2Agent、AgentCF、MemRec）和普通LLM做对比；CoVeMem在19/20个指标上超过或匹配最强文本记忆基线，在Goodreads、MovieTV等交互密集域表现尤为突出；总体Hit@1、NDCG提升显著。

**⚠️ 局限性**

局限性包括：①记忆质量高度依赖所选协同背骨（如LightGCN），若背骨弱则提升有限；②目前仍使用单一背骨，未探索多背骨融合；③对动态交互更新仅支持离线训练，在线微调需进一步研究；④适用于指令式推荐，其他推荐场景需验证。

---

## 59. Why Current XAI Is Not Enough for Arabic NLP: A Critical Survey of the Explainability Gap

**arXiv ID:** 2608.26144 | [PDF](https://arxiv.org/pdf/2608.26144v1)

**作者:** Salima Lamsiyah `[一作]` (University of Luxembourg), Ruslan Mitkov `[通讯]` (University of Alicante)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对现有的阿拉伯语可解释人工智能（XAI）研究进行结构化综述，系统识别出方法、任务与语言层面的缺口，构建了四层解释目标的分类体系，并提出面向阿拉伯语语言特点的研究议程与评估规范。

**💡 创新点**

创新点在于：①首次把阿拉伯语XAI划分为方法、任务与语言三大缺口；②提出包含预测层、模型层、语言层和社会文化层的四层解释框架；③构建批判性分类表与任务‑方法覆盖图，明确指出目前研究的热点与盲区；④提出针对阿拉伯语形态、方言、书写变体等特征的可解释性评估与实验设计。

**🔧 技术方法**

主要技术为文献检索与筛选、主题归纳与批判性分析、结构化表格与可视化，以及对比讨论；未进行实验实现。

**📊 数据集**

使用的“数据集”主要是公开论文与报告的引用，未采用单一语料库进行实验。

**📈 对比分析**

本工作不涉及实验对比与性能评估，重点在于综述与框架构建；因此没有具体的性能指标可呈现。

**⚠️ 局限性**

局限性包括：①未进行系统性检索与量化指标，综述范围有限；②覆盖偏向已发表的分类研究，可能遗漏未使用XAI术语的相关工作；③缺乏对已提议方法在阿拉伯语上的实际评估与实验验证。

---

## 60. When Is Noise Response Universal? Tokenization as the Hidden Variable in Language Models

**arXiv ID:** 2608.26319 | [PDF](https://arxiv.org/pdf/2608.26319v1)

**作者:** Yefan Tao `[一作]` (Amazon), Luyang Kong `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究文本模型在噪声下的鲁棒性，比较句子嵌入与解码器LLM在词级与字符级噪声下的降解曲线，发现通用性在词噪声下存在，字符噪声导致分裂；并证明训练目标而非架构决定通用性；阐明分词导致字符噪声分裂的机制；通过对比实验验证并给出仅凭干净准确率预测噪声下性能以及噪声增强训练提升鲁棒性的方法。

**💡 创新点**

提出“通用性”框架在不同噪声尺度下的分裂；证明训练目标是决定通用性的主因；发现分词机制是字符噪声分裂的根源；提供仅凭干净准确率即可预测噪声鲁棒性的实用工具；指出噪声增强训练能提升鲁棒性但无法恢复通用性。

**🔧 技术方法**

对比实验、归一化降解曲线、指数拟合、CV、ICC统计、token‑edit 破坏度量、InfoNCE 训练、噪声增强训练及 JS‑divergence 评估等技术。

**📊 数据集**

STS‑B、20 Newsgroups、AG News、SciFact、Wikitext‑103、SST‑2、SQuAD、SNLI 等公开文本数据集，结合 1k+句子、13 种噪声水平（字符替换、单词删除、键盘错别字、OCR 误差等）。

**📈 对比分析**

采用 CV、ICC 以及指数拟合来衡量模型降解曲线的一致性；在词级噪声下，9 句嵌入模型 CV≈2.2%，18 LLM CV≈2.0%；字符噪声 CV显著增大（≈14%/4.8%）。训练目标对通用性提升显著，噪声增强训练可在字符噪声下提升鲁棒性（如 Qwen2.5‑1.5B 在 ε=0.3 时 AG‑News 准确率 +13%），但不会恢复通用性。

**⚠️ 局限性**

仅评估字符/词级噪声，未覆盖语义扰动；tokenization 机制为经验性，中介而非充分因子；对比实验仅适用于公开模型，无法评估封闭 API；实验仅针对英语模型，未扩展至多语言或 >100B 参数模型。

---

## 61. DeflectBench: A Benchmark for Evaluating Rhetorical Fallacy Generation in LLMs

**arXiv ID:** 2608.26119 | [PDF](https://arxiv.org/pdf/2608.26119v1)

**作者:** Art Kanke `[一作]` `[通讯]` (University of Minnesota), Art Kanke (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型在按请求生成修辞谬误（Whataboutism、Ad Hominem、Red Herring）时的行为，并构建 DeflectBench 基准。

**💡 创新点**

发现提示框架和请求类型是驱动模型拒绝的主要因素，并揭示四种模型在标记合规、软拒绝和干净合规方面的显著差异签名。

**🔧 技术方法**

采用双盲 LLM 判定、八字段评分表、统计分析（kappa、AC1、Bootstrap、χ² 等）评估生成质量和拒绝模式。

**📊 数据集**

使用 80 条争议级别不同的声明（真、共识、争议、假）与 15 种提示模板，共生成 23,990 条样本。

**📈 对比分析**

对四个前沿模型在不同提示、谬误类型和争议级别下的拒绝率、合规率及指令遵循度进行对比，结果显示提示可导致近 100% 的拒绝差异，模型在标记合规率上差异显著。

**⚠️ 局限性**

仅包含单轮英文数据，缺少人类验证标签、开放源模型评估和多轮对话场景的适用性。

---

## 62. Learning-Augmented Online Allocation under Unreliable Advice: Robustness, Exposure Fairness, and Distribution Shift

**arXiv ID:** 2608.26889 | [PDF](https://arxiv.org/pdf/2608.26889v1)

**作者:** Fredy Pokou `[一作]` `[通讯]` (Inria), Fredy Pokou (Inria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在存在学习预测和保守回退分数的在线分配问题，并提出了既稳健又公平的学习增强规则。

**💡 创新点**

创新点在于将预测分数与保守分数按权重插值，并通过虚拟队列惩罚累计曝光失衡，给出了有限时域下一致性与稳健性同时满足的理论保证。

**🔧 技术方法**

采用学习增强算法、保守插值、虚拟队列公平校正以及竞争比分析与有限样本误差界定技术。

**📊 数据集**

实验基于 MovieLens 1M 数据集构造的在线分配实例，利用用户-电影评分生成奖励。

**📈 对比分析**

通过与随机、流行度、仅预测等基线对比，测量竞争比和曝光公平差；实验显示 Robust-LA 在预测不准时保持高竞争比，Fair-LA 在保持竞争比的同时显著降低曝光不平衡。

**⚠️ 局限性**

局限在于仅处理有限时域、斜板基准，并假设误差有界；未考虑硬匹配容量或策略性预测，未来可扩展至这些情形。

---

## 63. Beyond Client Averaging: A Client-Independent Second-Order Stationary-Bias Component in Stochastic SCAFFOLD

**arXiv ID:** 2608.26765 | [PDF](https://arxiv.org/pdf/2608.26765v1)

**作者:** Yi-Ping Tang `[一作]` (National Chung Hsing University), Guan-Ju Peng `[通讯]` (National Chung Hsing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

在全参与的SCAFFOLD/ FedAvg框架下，对恒定学习率的随机梯度迭代进行长期稳态分析，证明了在一维同质客户端、固定本地步数、有限梯度噪声的条件下，系统稳态均值存在两阶展开：已知的O(γ/N)漂移被客户端平均消除，但仍残留一个不随客户端数量变化的O(γ²)漂移，并给出了其显式系数。研究了此漂移的产生机制：局部路径的二阶矩、噪声和控制方差在非二次曲率的作用下被转换为均值偏移。

**💡 创新点**

① 首次在恒定学习率下给出客户端数不变的二阶稳态偏移项及其闭式系数；② 揭示了二阶偏移源自局部二阶矩与非线性曲率的相互作用；③ 证明了此偏移项随客户端数量不随之消减，说明单纯增加客户端无法进一步降低长期误差。

**🔧 技术方法**

使用了常数步长的随机近似理论、稳态分布（invariant law）分析、泰勒展开、矩量化与误差界定、坐标替换技术（对噪声坐标置零以获取1/N尺度）以及数值蒙特卡罗实验对理论进行验证。

**📊 数据集**

实验采用的是自定义的一维强凸目标函数 f'(x)=x+½logcosh(x)（其二阶导数在[1/2,3/2]内）以及均值为0、方差为1的Rademacher噪声；未使用公开数据集。

**📈 对比分析**

将理论预测的规范化残差（去掉已知的O(γ/N)项后除以γ²）与模拟得到的稳态均值对比。实验显示残差在不同客户端数量下趋于非零常数，并随学习率减小按O(γ)收敛到闭式系数，验证了理论结果；未给出传统机器学习任务中的性能指标。

**⚠️ 局限性**

仅在一维同质、固定本地步数的理想化设置下证明；不考虑客户端异质性、多维参数、状态相关噪声或大步长；仅验证了理论在模拟环境中的可行性，真实应用场景下是否保持相同系数仍需进一步研究。

---

## 64. How Do LLM Agents Actually Get the Flag? Trace-Level Provenance for Agentic Offensive Security Evaluation

**arXiv ID:** 2608.26237 | [PDF](https://arxiv.org/pdf/2608.26237v1)

**作者:** Kimberly Milner `[一作]` (NYU Tandon School of Engineering), Ramesh Karri `[通讯]` (NYU Tandon School of Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于执行轨迹的审计框架 CTF‑Abacus，用以重建每一次 CTF 尝试的完整求解路径，并将每一次 flag 的获得来源进行分类，区分真正的漏洞利用、直接暴露、回忆、外部查询等多种路径。

**💡 创新点**

创新点在于：①把执行轨迹细化为 PTES 阶段和具体技术标签，自动生成“solve profile”；②通过两位不同模型背景的判定器对每条轨迹进行独立审计，降低模型偏差；③聚合多模型、跨挑战的 solve profile，生成“challenge signature”，揭示挑战难度与攻击者能力的真正关联；④在传统 flag 计数的基础上加入验证指标，显著揭示过往评分体系对漏洞利用能力的过高估计。

**🔧 技术方法**

主要技术包括：执行轨迹提取与线性化；动作与观察的 PTES 与安全技术标签化；flag 源头（环境、外部、推理）与首次出现位置的自动识别；两位多模型判定器（Claude 与 GPT）对 exploit 行为的判定；人工审核支持置信度不足的案例；以及聚合分析产生的 challenge signature。

**📊 数据集**

使用了 240 个 CTF 挑战（来自 HTB、InterCode‑CTF、CTFTiny、CyBench），共 1,435 次尝试，涉及 6 个前沿 LLM（Claude Opus 4.8、GPT 5.5、Gemini 3.1 Pro、GLM 5.2、DeepSeek V4 Pro、Qwen 3.7 Max）。

**📈 对比分析**

通过将每次尝试拆解为 solve profile，并对其进行两套判定器审计，得到 2,870 条解决方案概况。验证后发现仅 62–87% 的 flag 来自可追溯的漏洞利用，整体验证率约 72.9%。与传统 flag 计数相比，模型得分下降 17.4–22.6%，并且挑战签名显示 40.3% 的挑战在不同模型间表现为混合路径，揭示评分不稳定性。比较实验中，挑战签名可将真实能力与表面得分分离，提升评估可信度。

**⚠️ 局限性**

主要局限在于：①对于极简或环境依赖严重的挑战，合法利用与 shortcut 难以区分，人工判定仍必不可少；②轨迹观测覆盖仍有限，未能捕捉全部工具交互细节；③判定器需人工维护与迭代，尚未实现完全自动化；④在新颖或未知攻击模式下，模型的判定准确率可能下降。

---

## 65. Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models

**arXiv ID:** 2608.26587 | [PDF](https://arxiv.org/pdf/2608.26587v1)

**作者:** Saksham Khatwani `[一作]` (University of Colorado Anschutz Medical Campus), Yanjun Gao `[通讯]` (University of Colorado Anschutz Medical Campus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

系统评估了在临床诊断任务中将医学知识图（UMLS、PrimeKG）与大型语言模型（Qwen2.5-7B、Qwen3-8B、Gemma-7B）结合的多种方法，探究了五种KG任务（路径判定、下跳预测、路径补全等）与三种训练范式（监督微调、奖励模型、强化学习）对模型性能与优化几何的影响，并提出梯度层级诊断指标 Gradient Intervention Density (GID) 与 Gradient Distortion (GD) 来量化训练过程中的参数更新特征。

**💡 创新点**

1) 引入 GID 与 GD 这两种梯度诊断指标，首次在 KG‑LLM 整合中通过梯度稀疏度与方向误差揭示“外科对齐”（surgical alignment）——即KG判定训练产生局部、稀疏更新，保留预训练模型大部分知识；2) 证明稀疏更新在高阶临床推理质量（如 PDSQI‑9 组织、综合等维度）上优于传统密集更新，强调优化几何应成为评估维度。

**🔧 技术方法**

使用超参数微调（SFT、GRPO、RM‑R1 等），KL 正则化与 LoRA 微调，BFS 路径抽取，QuickUMLS/SIMSTRING 进行实体映射，ROUGE‑L、CUI‑F、PDSQI‑9 等评估指标，梯度层级可视化（层深、GD、GID 映射）等技术。

**📊 数据集**

UMLS 与 PrimeKG 两个医学 KG；临床文本数据 ProbSum 与 DDXPlus；诊断预测基准 ProbSum、DDXPlus；医学问答基准 MedQA 与 MedMCQA；PDSQI‑9 评估基于 Azure GPT‑5‑mini。

**📈 对比分析**

与非微调基线 NFT、单任务 SFT、RAG（检索增强）等进行对比。KG 训练在大多数指标上优于 NFT；单任务 SFT 在本域 ROUGE‑L 最高，但在 PDSQI‑9 组织/综合维度表现不如 KG 判定训练；梯度稀疏度（GID）与方向误差（GD）显示，稀疏更新的模型在跨任务迁移和高阶推理上更具优势。

**⚠️ 局限性**

仅评估了三大 LLM，未扩展到更多模型；研究聚焦训练范式和 KG 交互，未提出新的 RL 训练框架；实验仅覆盖医学诊断场景，缺乏对其他领域的验证；梯度阈值与可视化方法对结果有一定影响，需要进一步稳健性检验。

---

## 66. The Artificial Experimentalist: Discovery and Control of Self-Organizing Phenomena with Autotelic Reinforcement Learning

**arXiv ID:** 2608.26116 | [PDF](https://arxiv.org/pdf/2608.26116v1)

**作者:** Marko Cvjetko `[一作]` (Inria Centre at University of Bordeaux), Pierre-Yves Oudeyer `[通讯]` (Inria Centre at University of Bordeaux)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于自我设定目标的强化学习框架CARL，能够在连续元胞自动机Lenia中闭环地发现并控制自组织的溶子（soliton）模式。

**💡 创新点**

创新点在于将闭环控制、目标条件化策略、稀疏动作代价以及人机交互统一到一套框架中，使得代理在未知更新规则下仍能零样本地生成并操控自组织现象。

**🔧 技术方法**

技术包括自我设定目标的强化学习（Autotelic RL）、双深度Q网络（Double DQN）、全卷积U‑Net架构配合FiLM条件化、局部质量扰动的动作空间以及多目标奖励设计。

**📊 数据集**

使用的数据集主要是Lenia的85个手选更新规则（卷积核）与7个未见规则，随机生成的初始格子以及在64×64网格上收集的演化序列。

**📈 对比分析**

与无操作、随机、基于质量增删、死区版本等启发式基线相比，CARL在溶子生成率上提升显著（约1.5–2倍），在方向控制任务中平均余弦相似度达到0.91，且在多种变异条件下仍保持高成功率。

**⚠️ 局限性**

局限性包括需要领域专家设计奖励、观察与动作空间；仅适用于完全可观测、确定性的二维网格；对噪声、部分观测或高维动作空间的鲁棒性不足，且在真正生物医学系统中的可迁移性尚未验证。

---

## 67. Who Remains, What Changes: Identity Anchored Composed Gait Retrieval

**arXiv ID:** 2608.26632 | [PDF](https://arxiv.org/pdf/2608.26632v1)

**作者:** Jingchen Fei `[一作]` (Beijing University of Posts and Telecommunications), Man Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种名为Composed Gait Retrieval (CoGR) 的交互式步态检索任务，并提出了ComposeGait框架来实现身份保持与自然语言指令的组合检索。

**💡 创新点**

①设计了无监督的VLM自动标注管道生成语言增强步态数据集；②提出了身份锚定的Part-aware Identity Adapter (PIA) 与共享Q-Former 的双向 ID token 注入，防止身份漂移。

**🔧 技术方法**

使用 BLIP-2 Q-Former、ViT-G 视觉编码、部分感知身份适配器、双向共享 Q-Former、联合身份与对比学习、可解释的多帧特征聚合等技术。

**📊 数据集**

生成了 Language-Augmented CCPG（50k 三元组）与 Language-Augmented CASIA-B（64.5k 三元组）作为评测基准。

**📈 对比分析**

与多种 CIR、VR、CPR、零样本方法对比，在两套基准上 ComposeGait 分别取得 R@1 72.38%（CCPG）和 83.61%（CASIA-B），均为最高或接近最高，显著提升性能。

**⚠️ 局限性**

仅在受控室内/固定摄像头数据上验证；依赖 VLM 生成文本，可能存在描述误差；对大规模多姿态/隐私敏感场景仍需进一步研究。

---

## 68. Terrain signatures in Welsh settlement names

**arXiv ID:** 2608.26978 | [PDF](https://arxiv.org/pdf/2608.26978v1)

**作者:** Oktay Karakuş `[一作]` (Cardiff University), Can Eyupoglu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对威尔士三千七百五十七个定居点的地名进行冻结、源审计的词汇分类，并预先设定三个环境假设，检验地名词汇与当代环境变量（河流距离、局部地形位置、林覆盖率）的关联。

**💡 创新点**

创新之处在于构建可复现的“冻结词汇层”，并结合预注册、空间限制的验证，系统评估地名词汇是否在地理结构之外携带可测量的环境信息。

**🔧 技术方法**

主要采用规则式字符串匹配构建词汇暴露，随后利用线性回归、分数对数回归、聚类稳健误差、Holm多重比较、空间受限随机化以及基于空间阻塞的岭回归交叉验证等统计技术。

**📊 数据集**

使用的主要数据集为Ordnance Survey Open Names的威尔士定居点列表、OS Terrain 50与Copernicus GLO-90数字地形/表面模型、OS Open Rivers网络以及CORINE 2018森林覆盖图。

**📈 对比分析**

在预注册的线性模型中，地形词汇（bryn/mynydd vs cwm/pant）对应平均局部地形位置差异24.4 m（95% CI 10.8–38.1 m），并在10、25、50 km空间阻塞下的留出预测中平均提升4.6–7.3%均方误差；河流词汇也呈现较弱但一致的正相关，林木词汇未能估计。

**⚠️ 局限性**

主要局限包括残余空间自相关、缺乏外部验证、词汇分类未经过形态学验证、语言标签不完整、仅使用当代环境指标，且无法证明命名与历史环境记忆或跨区域迁移的适用性。

---

## 69. Relaxation-Aware Multimodal Sensing of Soft Gripper Driven by Structure-Perception-Learning

**arXiv ID:** 2608.26622 | [PDF](https://arxiv.org/pdf/2608.26622v1)

**作者:** Yanzhe Wang `[一作]` (Zhejiang University), Huixu Dong `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一款可变刚度软抓手，并搭建了温度耦合的放松感知与力调节框架，实现了长时间稳定抓握；

**💡 创新点**

创新点在于将热可调 SMP 结构、基底视觉+红外热成像、多模态感知与温度耦合的可学习粘弹性力模型相结合，实现了对粘弹性放松的实时预测与补偿；

**🔧 技术方法**

采用SMP材料实现热调刚度、RGB相机+红外相机实现多模态感知、基于物理约束的学习网络（ICFN+RCFN）进行即时力和放松曲线预测，以及PID闭环力控制；

**📊 数据集**

实验使用人工制备的软手指与多种目标（铝罐、塑料杯、玻璃杯、羽毛、刷子、扳手等），并在 25–60°C 及 0–10 N 的力范围内采集数据；

**📈 对比分析**

与固定阀门、仅即时估计的两种基线相比，放松补偿方法在 280 s 抓握任务中目标误差降低约 80%（MAE 0.066 N，<1.4% 目标力），即时估计误差约 26%；

**⚠️ 局限性**

局限在于只验证单指抓手、放松模型为单一有效模态、红外更新速率慢、对极端温度或长时间高负载的泛化仍待进一步研究。

---

## 70. Procedura: Agentic 3D Modeling with Procedural Control

**arXiv ID:** 2608.26238 | [PDF](https://arxiv.org/pdf/2608.26238v1)

**作者:** Youtian Lin `[一作]` (Nanjing University), Yao Yao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于LLM的 3D 形状-as-code 生成框架（名为 ），通过分阶段的 assembly 图规划、按部件逐步生成、解耦视觉批评和编译/模拟器验证，输出可编辑的多部件硬表面模型。

**💡 创新点**

创新点主要包括：①将 3D 对象直接写成带有命名部件和可机器检查 typed mates 的 CSG 程序；②用 LLM 逐部件生成、解耦批评家修正、每步编译/连通性验证，避免“漂移”与误放；③在同一图中同时携带材料与关节信息，实现可编辑性、材质分配与物理可验证的关节；④不需要任何 3D 数据训练，直接利用冻结的 LLM 生成高质量模型。

**🔧 技术方法**

使用的技术包括：冻结的 Gemini 3.7 Flash/GPT‑5.6‑sol 作为 LLM；OpenSCAD 作为 CSG 编译器；Blender/Cycles 进行渲染与可视化；Isaac Sim 进行关节/物理验证；CLIP/评判器对齐与质量评估；面向图的 typed mates 结构；视觉批评家模型（单视图判定）与修正工具。

**📊 数据集**

评测数据集：MechBench‑36（36 个多部件硬表面机械/车辆模型）和 P3D‑Bench（203 个文本+图像对），两者均使用统一的参考图像和渲染设置。

**📈 对比分析**

方法与多种现有技术进行对比：本地 3D 生成器（TRELLIS.2、UltraShape 等）、先前的 3D‑code agent（Adam CAD、ArtiCraft、CAD‑Coder 等）以及单次 LLM 调用。结果显示：在 MechBench‑36 上综合分 0.828（最高），在 P3D‑Bench 上 0.590（领先所有公开条目）。在几何质量、可编辑性、锐利边缘等指标上均优于对手，且单次调用相较提升显著。

**⚠️ 局限性**

局限性：①仅通过渲染视图感知构建，可能忽视内部或细节交互；②CSG 适用于硬表面对象，难以自然表达有机自由形状；③typed mates 词汇仅覆盖刚性关节，无法验证非刚性或更复杂的约束。

---

## 71. Dynamic Tree Colors: Adaptive Discriminable Hierarchies with Minimum Instability

**arXiv ID:** 2608.26734 | [PDF](https://arxiv.org/pdf/2608.26734v1)

**作者:** Tobias Mertz `[一作]` (Fraunhofer IGD), Jörn Kohlhammer `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并评估了一种名为Cuttlefish的多尺度层级可视化动态颜色映射算法，旨在在层级尺度变化时保持颜色稳定性并提高区分度

**💡 创新点**

在颜色空间中引入了稳定性比例（stability ratio）与增量式颜色更新的概念，构建了区分度（discriminative power）与颜色不稳定性（color instability）的评估框架

**🔧 技术方法**

采用CIE L*a*b*颜色距离度量、稳定性比例与增量更新技术，对颜色进行动态映射和评估

**📊 数据集**

使用了世界人口层级数据集、GoldMessenger层级数据集以及一个treemap样例数据集进行实验

**📈 对比分析**

通过与不同稳定性比例和随机采样方法对比实验，Cuttlefish在多尺度交互场景下实现了较低的增量不稳定性，但总体区分度略低于某些传统方法

**⚠️ 局限性**

评估采样不充分、缺乏统一性能指标、方法仅针对层级可视化，在更广泛交互场景下的适用性和可扩展性仍需进一步验证

---

## 72. Aphanta: Diagnosing Task-Aligned Image-Edited Intermediates for Multimodal Reasoning

**arXiv ID:** 2608.26993 | [PDF](https://arxiv.org/pdf/2608.26993v1)

**作者:** Hengyuan Xu `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过构建自动化任务发现与评估框架 Aphanta，系统性地比较直接推理、实际编辑与理想参考三种条件下 MLLM→图像编辑器→MLLM 循环的性能，评估图像中间表示对任务的实际帮助。

**💡 创新点**

创新点在于：①提出三条件（A/B/C）诊断协议，用以区分视觉潜在收益与编辑器实际实现的差距；②实现自动化任务生成与闭环验证；③提供可复用的任务池与实验协议，为后续视觉思考研究奠定基础。

**🔧 技术方法**

主要技术包括：指令驱动的扩散式图像编辑器、Qwen 等多模态大语言模型、自动化任务生成与人机评估流水线、以及基于评分差值的性能对比。

**📊 数据集**

使用自建的 Aphanta Train/Test 数据集，涵盖 20 个候选视觉推理任务（如计数、图形解析、结构化图表、对比生成等），并在此基础上进行实验。

**📈 对比分析**

通过对每个任务计算直接推理得分、实际编辑得分和理想参考得分的差值来评估编辑器效能；在保留正效应任务上，Qwen‑Image‑Edit 平均分从 0.343 提升至 0.445（+29.7%），但结构化任务仍表现负增益。

**⚠️ 局限性**

局限性包括：只评估通用指令编辑器；缺乏对像素级因果贡献的细粒度拆解；任务池覆盖面有限，未能涵盖所有视觉操作；跨模型比较受闭源编辑器实现细节的限制。

---

## 73. Evaluating AI Generated Summaries for Cancer Patients

**arXiv ID:** 2608.26154 | [PDF](https://arxiv.org/pdf/2608.26154v1)

**作者:** Muhammad Aurangzeb Ahmad `[一作]` (Careology Inc.), Paul Landau `[通讯]` (Careology Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文在数字化癌症护理平台中评估了AI生成的患者摘要，采用了人类专家和LLM评审双重评估框架；

**💡 创新点**

创新点在于提出了针对临床使用的多维度评估方法，并通过反馈循环不断优化提示与模型输出的可靠性；

**🔧 技术方法**

技术上使用Claude Sonnet 3.7进行摘要生成，使用Mistral模型作为评判者，结合人类临床医生与护理人员的手工评分；

**📊 数据集**

数据集来自于癌症护理App收集的病人自报数据（药物、症状、体征、情绪、自由文本）以及Guy’s与St Thomas’医院的匿名化病历；

**📈 对比分析**

评估显示AI摘要在事实准确性（平均0.992）、专业风格（0.979）和无毒性方面表现突出；但完整度与可帮助性有较大变异；相较于单一自动指标，双重评估提升了对临床安全性的把握；

**⚠️ 局限性**

局限包括评估标准在人工与LLM之间不完全一致、样本量有限、某些错误源自提示不严谨或数据缺失，且缺乏对模型漂移与公平性的长期监测。

---

## 74. Mitigating Strong-Modality Collapse in Multimodal Learning via Inverted Asymmetric Fusion

**arXiv ID:** 2608.26879 | [PDF](https://arxiv.org/pdf/2608.26879v1)

**作者:** Mary Ogbuka Kenneth `[一作]` (Imperial College London), Abbas Edalat `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两阶段的Inverted Asymmetric Fusion（IAF）框架，先通过模态感知知识蒸馏提升弱模态，再通过纯/注意机制保护强模态，避免强模态崩塌，从而实现更稳健的多模态融合。

**💡 创新点**

创新点在于通过结构化的“纯/注意”分层设计，首次从架构层面消除强模态被弱模态干扰的问题，并结合模态感知知识蒸馏、自适应门控和课程式模态丢弃，实现对强模态的显式保护。

**🔧 技术方法**

采用了模态感知知识蒸馏、交叉注意力（Cross-Modal Anchor）、自适应门控、Mixup、课程式模态丢弃以及两阶段训练等技术。

**📊 数据集**

在三大幽默/讽刺检测基准（MultiHuSE、UR-FUNNY、MUStARD）上进行评估，涵盖文本主导、音视频主导等不同模态层级。

**📈 对比分析**

与早期、晚期、对称注意融合以及现有最先进方法对比，IAF 在 MultiHuSE 上提升约 2–6% 绝对准确率，在 UR-FUNNY 上实现 70.72% 超过先前最佳 68.60%，在 MUStARD 上达到 83.33% 超越 MHA 79.32%，整体比基线提升约 8.25%。

**⚠️ 局限性**

局限性包括需先经验识别强模态、仅在冻结特征上验证未评估端到端训练、仅针对英文幽默/讽刺任务，未验证跨语言或其他多模态任务的通用性。

---

## 75. TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding

**arXiv ID:** 2608.26112 | [PDF](https://arxiv.org/pdf/2608.26112v1)

**作者:** Jiaming Fan `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了TreeGraft——一种多草稿器树式推测解码框架，利用轻量级与中等强度草稿模型共同构建共享草稿树，并通过在线调度决定何时调用强草稿器；

**💡 创新点**

创新点包括①扩展的grafting位置选择，允许中等草稿器在任意未访问节点上重新评估并重新挂载；②非破坏性grafting，在不覆盖旧分支的前提下追加新子节点；③基于价值引导的在线调度器，离线拟合价值系统后进行Margin distillation，实时低成本决定调用强草稿器；

**🔧 技术方法**

采用共享草稿树构造、非破坏性grafting、价值系统离线拟合+Margin distillation、轻量级MLP调度器、N-gram轻量草稿器以及预训练的LLaMA3/Qwen3模型；

**📊 数据集**

在GSM8K、Alpaca、NQ、HumanEval、CNN/DailyMail和MT-Bench等六个基准上进行评测；

**📈 对比分析**

与单草稿器基线（All Small、All Mid）以及Cascade Speculative Drafting等做对比；在10个模型对、6个基准上平均提升1.60×速度，最高提升26.6%；在未见模型对和任务上亦保持1.48×/1.60×的优势；

**⚠️ 局限性**

局限性包括需依赖预训练草稿模型并进行离线调度器训练，可能在极低预算或大规模推理场景下调度开销与内存使用仍需进一步评估；

---

## 76. AdaThinking-E: One-Token Entropy Regulation for Adaptive Thinking

**arXiv ID:** 2608.26141 | [PDF](https://arxiv.org/pdf/2608.26141v1)

**作者:** Zining Wang `[一作]` (Meituan), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于单token熵调节的强化学习框架AdaThinking‑E，能够让多模态大语言模型自动决定何时进行深度推理（思考）或直接给出答案。

**💡 创新点**

创新点在于：①通过熵奖励在探索期保持高熵、在收敛期实现低熵，从而实现模型自发学习何时思考；②将模式切换token与响应token独立优化，消除长度偏差导致的思考崩溃；③引入决策反馈机制（准确率、模式比例）进一步引导自适应思考。

**🔧 技术方法**

采用的技术包括：冷启动微调+强化学习（GRPO/DAPO改进）、熵奖励、决策反馈、独立优化的模式切换token与响应token、AdaThinking‑Doc数据集、QwenVL/InternVL等多模态LLM。

**📊 数据集**

使用的数据集包括自建的AdaThinking‑Doc（涵盖简单提取与复杂推理问题）以及公开文档理解基准：DocVQA、InfoVQA、ChartQA、CharXiv、OCR‑Reasoning、OCR‑Bench等。

**📈 对比分析**

通过与NT、TK、ATK等模型在多项基准上对比，AdaThinking‑E在所有指标上均达或超过SOTA，提升范围为0.6%–10.8%；同时保持低思考比例和较短输出长度，显著提升计算效率。

**⚠️ 局限性**

局限性：对熵调节超参数敏感，极难任务下思考比例可能不足；在更大规模或不同领域的评测中还需验证稳健性；以及仍依赖手工构建的AdaThinking‑Doc数据集。

---

## 77. C-Unseen: Weak Signal Detection in Dynamic Temporal Knowledge Graphs via LLM Reasoning

**arXiv ID:** 2608.26870 | [PDF](https://arxiv.org/pdf/2608.26870v1)

**作者:** Yassir Lairgi `[一作]` (GAUC), Pierre Cléau `[通讯]` (GAUC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大语言模型链式推理的自解释框架，用于在动态时间知识图（DTKG）中检测弱信号。

**💡 创新点**

创新点包括：①首次在DTKG中正式定义弱信号；②将弱信号拆解为稀有子图提取器和弱信号提醒器两模块，利用LLM在两步推理中识别与主线叙事相悖的稀有子图；③通过稀有子图的连接子图跟踪时间演化，获得自解释的弱信号检测结果。

**🔧 技术方法**

核心技术：动态时间知识图构建与双时间建模；稀有子图提取与连接子图构造；LLM链式推理（CoT）提示策略；基于anchor词匹配的评估指标（精确率、召回率、F1、提前时间）。

**📊 数据集**

数据集：Wiki‑OpenAI基准，由2015‑2025年间的OpenAI维基百科月度编辑提取的773条原子事实（757来自维基，16手工添加），并提供匿名化版本（Wiki‑OpenAI‑Anon）。

**📈 对比分析**

与基线方法（Yoon、BERTrend、BEAM）对比，框架在k=1、2、3阈值下均取得最高F1、覆盖率；平均提前时间约1年，优于BERTrend的1.7年；输出可解释为子图与自然语言阐释，显著优于基线的关键词/主题袋输出。

**⚠️ 局限性**

局限性：对大规模DTKG的可扩展性待验证，因LLM上下文窗口限制可能无法一次性处理完整快照；过度依赖LLM知识，可能产生误报；基准仅覆盖单一组织，需在更多领域进行验证；缺少领域本体指导，导致潜在误判。

---

## 78. Benchmarking AI Agents for Hardware Design Automation via MCP Tool Calling

**arXiv ID:** 2608.26199 | [PDF](https://arxiv.org/pdf/2608.26199v1)

**作者:** Leonardo Liparulo `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在本地部署的大语言模型是否能在硬件设计工作流程中可靠地执行工具调用任务，构建了MCP服务器和基准；

**💡 创新点**

提出了面向硬件设计的MCP服务器与任务基准，系统评估了多种模型与配置组合的影响，并给出了实用部署建议；

**🔧 技术方法**

使用了Model Context Protocol (MCP) 与ReAct/Plan‑and‑Act架构、系统提示、工具描述与上下文管理等技术；

**📊 数据集**

基准包含六类核心任务（Easy、Medium、Hard、History、Errors、Cross）和两类多服务器任务（Easy‑Noise、Hard‑Noise），由专业工程师设计并验证；

**📈 对比分析**

通过四个指标（ECC、EVCR、TFR、NCA）评估各模型配置，在最佳配置下Gemma‑4‑31B等模型实现约99%期望调用覆盖率；

**⚠️ 局限性**

局限包括：仅测试两种上下文策略，未探究更细粒度记忆管理；EVCR未考虑错误严重性；未测量推理延迟；MCP服务器与真实工具有差距，基准和服务器无法公开。

---

## 79. Memory Anchors for Continual Robot Learning

**arXiv ID:** 2608.26545 | [PDF](https://arxiv.org/pdf/2608.26545v1)

**作者:** Maximilian Du `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了机器人持续学习中经验回放（Experience Replay）机制，提出并验证了从旧任务中挑选少量关键经验（Memory Anchors）以缓解灾难性遗忘。

**💡 创新点**

创新点在于提出三步方法通过潜在空间重叠与动作分歧识别Memory Anchors，并证明即使占比极小也能显著提升旧任务性能，且通过增强回放缓冲区可以降低高冲突任务的遗忘。

**🔧 技术方法**

技术手段包括：Diffusion Policy 与 Vision‑Language Action（VLA）模型的行为克隆、潜在表示距离计算、动作分歧检测（policy vs. ground‑truth）、经验回放缓冲区的采样与增补。

**📊 数据集**

实验数据集主要为 LIBERO benchmark（LIBERO‑Long、Goal、Object、Spatial 四套任务序列）以及两组真实机器人任务（OpenJar 与 SweaterFold）。

**📈 对比分析**

与随机回放、MIR、GSS 等基线比较后，Memory Anchor Enrichment（AnchorER）在 LIBERO‑Goal 上平均 NBT 降低 37% 以上，实机任务成功率提升 1.7 倍，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：需访问模型潜在空间和完整历史数据，计算成本高；实验仅覆盖 3–10 个任务序列，未验证更长时间持续学习场景。

---

## 80. Geo-LoRA: Geometry-Aware Subspace Evolution for Low-Rank Adaptation in Continual Learning

**arXiv ID:** 2608.26960 | [PDF](https://arxiv.org/pdf/2608.26960v1)

**作者:** Yibo Feng `[一作]` `[通讯]` (University of Electronic Science and Technology of China), Yibo Feng (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Geo-LoRA 框架，在持续学习中通过几何约束调节 LoRA 低秩子空间的演化。

**💡 创新点**

创新点在于引入 SPP、ACSA、MCBO 三种投影约束，在共享与任务特定分支上分别控制子空间漂移、核心对齐和历史重叠，从而实现稳定‑可塑性平衡。

**🔧 技术方法**

采用低秩适配器（LoRA）结合 Grassmann 子空间投影距离、核心–空闲对齐以及统计归一化重叠约束，构成统一的几何正则化。

**📊 数据集**

在 ImageNet‑R、ImageNet‑A、VTAB、CUB‑200、CIFAR‑100 及 OmniBenchmark 等多种连续任务数据集上进行评测。

**📈 对比分析**

与 L2P、DualPrompt、CODA‑Prompt、InfLoRA、SD‑LoRA、ACMap、SEMA、BiLoRA、CL‑LoRA 等基线相比，Geo-LoRA 在大多数任务长度下均取得最高或次高平均准确率，尤其在分布偏移和细粒度场景中提升显著。

**⚠️ 局限性**

局限性在于训练时需额外计算几何正则化，且在极长任务序列或极高维子空间时可能增加数值不稳定和计算负担。

---

## 81. Scaling Model-Generated Distillation Data Can Make Latent Teacher Traits More Recoverable

**arXiv ID:** 2608.26958 | [PDF](https://arxiv.org/pdf/2608.26958v1)

**作者:** Zhichen Dong `[一作]` (Shanghai Jiao Tong University), Chao Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室控制环境下，研究了使用模型生成的脱离任务数据进行蒸馏时，随着样本量扩大，教师隐藏特征在学生模型中的可检测性（可恢复性）是否会提升。

**💡 创新点**

创新点在于提出并验证了“数据规模可提升隐藏教师信号的可恢复性”这一非传统观点，并将其表现在行为读出、定位边缘和更新空间几何三种维度；同时展示了该效应在单特征、多特征、跨模型和不同特征类型（偏好、任务能力、安全姿态）上的普适性。

**🔧 技术方法**

技术方法包括：① 通过系统提示或微调诱导教师表达目标特征；② 生成受限的“脱离任务”数据（如仅数字完成）并做显式过滤；③ 采用LoRA SFT训练学生；④ 使用匹配的无特征参考教师数据做对照；⑤ 通过参考调整后的目标得分、闭合集定位边缘、更新空间投影等指标评估可恢复性。

**📊 数据集**

使用的数据集为自生成的脱离任务样本，按目标特征（动物/植物偏好、数学/通用问题、系统安全性等）分别采样；此外还有对照的无特征生成数据；实验涉及多种开源大模型（Qwen、Claude、GPT、Llama、Mistral、Gemma 等）作为教师与学生。

**📈 对比分析**

通过与匹配的无特征对照学生比较，计算目标特征的参考调整得分 Δ_τ(n) 和定位边缘 Γ_τ(n)。结果显示：随着独立样本数从千级到十万级增长，Δ_τ 与 Γ_τ 均呈上升趋势，说明目标特征更易被检出；在跨模型蒸馏中，尽管噪声增大，但同样能观察到规模效应；在多特征诱导下，较大样本可以平衡竞争特征，提升多特征恢复率。

**⚠️ 局限性**

局限性包括：① 采用受限的脱离任务示例和显式过滤，未覆盖真实工业蒸馏流水线中更丰富的数据混合与复杂生成条件；② 评估指标仅覆盖偏好、任务准确率和安全姿态等有限范畴，可能忽略其他潜在行为；③ 更新空间分析仅展示几何相关性，未揭示底层机制；④ 实验规模受限于实验资源，未在更大模型或更广泛任务上验证；⑤ 结果表明规模效应在不同模型间存在差异，需进一步研究跨模型迁移的稳定性与可控性。

---

## 82. LiveSim: Simulating Environment-Shaped Users in Multi-Agent Live-Stream Ecosystems

**arXiv ID:** 2608.26849 | [PDF](https://arxiv.org/pdf/2608.26849v1)

**作者:** Jiaqi Xu `[一作]` (University of Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了LiveSim，一种基于大型语言模型的直播生态系统仿真框架，通过可编辑的行为假设来模拟用户随环境变化的行为；

**💡 创新点**

核心创新是将用户特征视为可编辑的行为假设，并通过反射式行为假设塑造（RBHS）循环将模拟与真实轨迹的差异转化为可迁移的环境‑行为补丁，进而构建集体行为记忆；

**🔧 技术方法**

利用LLM进行行为假设推断、差异探测、补丁生成、反射总结以及多智能体闭环仿真；

**📊 数据集**

使用某主流直播平台2025‑2026年间的风险控制日志，涵盖约1963名用户和14391个用户‑会话对；

**📈 对比分析**

与多种LLM骨干（Qwen、DeepSeek、GPT、Doubao等）和对照静态代理进行对比，在线性轨迹验证下提升了用户级行为一致性、分布对齐（A‑JSD‑43%）、转化精准度（Conv‑F1 +13%），并在闭环模拟中有效展示了欺诈演化和干预效果；

**⚠️ 局限性**

实验仅覆盖单一直播场景，验证仍待扩展至其他社交或电商场景；行为假设基于稀疏日志，不能视为真实心理状态；

---

## 83. MVC-Bench: Benchmarking Calibration of Medical Vision-Language Models

**arXiv ID:** 2608.27004 | [PDF](https://arxiv.org/pdf/2608.27004v1)

**作者:** Ashshak Sharifdeen `[一作]` (Mohamed bin Zayed University of AI), Muhammad Haris Khan `[通讯]` (Mohamed bin Zayed University of AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MVC-Bench基准，系统评估VLM和Medical-VLM在医学图像分类中的校准性能，并提出多类别间距正则化(MCM)方法。

**💡 创新点**

①构建了三轴（鲁棒性、效果性、稳定性）校准评估框架；②首次在医学影像中全面比较多种后置与训练时校准与提示调优方法；③提出的MCM在多种backbone、模态下显著降低ECE。

**🔧 技术方法**

使用VLM/Medical-VLM模型（CLIP、MedCLIP、BioMedCLIP、PLIP等）、提示调优技术（CoOp、KgCoOp、MaPLe、PromptSRC、HiCroPL、ProGrad）、后置校准（MDCA、LS、MBLS、ECCV_ZS、ECCV_Penalty、Temperature Scaling）以及自定义的MCM正则化，评估指标包括ECE、MCE、ACE。

**📊 数据集**

三类医学影像数据集：眼底（APTOS、EyePACS、Messidor、Messidor-2）；病理学（Kather、PaNuke、DigestPath）；胸部X光（COVIDX、RSNA18），并覆盖领域外数据以测试域移位。

**📈 对比分析**

通过1638+实验对比七种校准方法和六种提示调优方案，发现MCM在10/12内部域设置中取得最低ECE，且在域移位下保持竞争力；但无单一方法在所有情形中最优，校准效果受模态、backbone、seed与提示模板显著影响。

**⚠️ 局限性**

仅针对图像分类，未评估分割、检测、生成等任务；MCM主要解决低边距欠信，未能统一抑制过度信任，对某些病理学模态表现不佳；基准与方法在更广泛的临床转移情形下仍需进一步验证。

---

## 84. CARE: Causally-Aligned Reasoning Exploration for Medical Large Language Models

**arXiv ID:** 2608.26147 | [PDF](https://arxiv.org/pdf/2608.26147v1)

**作者:** Yucheng Zhou `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 CARE 的自我改进框架，用于医疗领域的大语言模型强化学习，旨在消除“答案正确却推理错误”的现象。

**💡 创新点**

创新点在于将经验筛选拆解为两条理论驱动的约束——因果充分性（通过自我验证实现的 do‑calculus 介入）和近邻可学习性（动态熵窗口控制梯度方差），并将其嵌入双流优化（策略探索+难度加权回放）中。

**🔧 技术方法**

技术上使用了自回归生成策略、对齐验证的 Agreement‑Based Self‑Verification、动态序列负对数似然窗口、组相对优势估计、经验回放与 KL 正则化等。

**📊 数据集**

实验覆盖多模态和文本两大类数据集：PMC‑VQA、SLAKE、PathVQA、MedMCQA、PubMedQA、OmniVQA、VQA‑RAD、MMMU‑Med 等医学问答与推理基准。

**📈 对比分析**

与 SFT 基线、标准 GRPO、以及多款 7B 级医学 VLM（如 Hulu‑Med、HuatuoGPT‑V、Lingshu‑7B 等）对比，CARE 在大多数基准上实现了显著提升，最高可提升 5–10% 的准确率，并在“正确答案-不一致推理”评估中将错误推理比例从 45% 降至 13%。

**⚠️ 局限性**

局限性包括：仍需要高昂的算力与大规模 GPU 资源；验证机制依赖模型对自我推理的准确性，可能对极难推理场景产生误拒；实验主要集中在问答与诊断推理，未覆盖更复杂的临床决策流程；以及对跨领域迁移的适用性尚未充分验证。

---

## 85. MedFG-VQA: Low-Frequency Memory and Graph Attention for Lightweight Medical VQA

**arXiv ID:** 2608.26848 | [PDF](https://arxiv.org/pdf/2608.26848v1)

**作者:** Haowen Gu `[一作]` (Nanjing University of Science and Technology), Fumin Shen `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MedFG‑VQA，一种轻量化的医学视觉问答模型；

**💡 创新点**

创新点在于通过FreqMemoryFusion（FMF）利用频域记忆融合低频全局信息，以及Graph‑Aware Cross‑Attention（GACA）结合图卷积实现局部结构对齐；

**🔧 技术方法**

采用DCT/IDCT频域变换、可学习记忆银行、跨模态注意力、KNN图卷积、门控融合和SmolLM2 LLM；

**📊 数据集**

使用自己构建的大规模SynMedVQA数据集（约2.06M问答对，涵盖9种影像模态与10个器官）以及SLAKE、VQA‑RAD、PathVQA三大公开医学VQA基准；

**📈 对比分析**

与InternVL3.5、MiniCPM‑V4、Qwen3‑VL、Gemma3、LLaVA‑Med等大型模型对比，MedFG‑VQA仅有约795M参数，在SynMedVQA上平均准确率0.6441，超过同类模型约9.5%；在公开基准上虽略逊于最强模型，但在开放式问题上取得最高准确率，显示良好泛化；

**⚠️ 局限性**

限制主要体现在：生成问答受GPT‑4o能力限制，数据单视图且缺少多视角/多模态融合，可能影响对新型影像模式的适应性。

---

## 86. Vagdhenu: A Vrutta (Meter) Aware Shloka-to-Chant (TTS) System for Sanskrit

**arXiv ID:** 2608.26146 | [PDF](https://arxiv.org/pdf/2608.26146v1)

**作者:** Prathosh A P `[一作]` (Indian Institute of Science), Prathosh A P `[通讯]` (Indian Institute of Science)

**通讯引用:** 878 | [OpenAlex ID](https://openalex.org/A5074080229)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个基于米度感知的 Sanskrit 讽歌文本到语音系统，并完成了两大规模部署（《摩诃婆罗多》5,183 诗句视频集和《湿婆婆婆诃》18,000 诗句音频应用），同时公开了前端、模型和数据。

**💡 创新点**

创新点包括：① 采用 Kannada 文字路由并实现细粒度 Sanskrit 语音学与 sandhi 规则；② 通过米度检测与参考片段选择（half‑reference 规则）实现语调驱动的合成；③ 在自填流匹配 backbone 中证明文本侧 prosody 控制器无效，提出参考片段与声纹微调为唯一可控手段。

**🔧 技术方法**

技术实现采用 Flow‑matching diffusion‑transformer backbone (IndicF5)，BigVGAN‑v2 对抗式声码器，Sanskrit 语音学前端，参考片段库与 classifier‑free guidance，声纹微调和持续时间门控质量控制等。

**📊 数据集**

数据集包括 5.3h 单声道 Sanskrit 讽歌录音（1,467 条片段，覆盖多种度量和音素），以及两大部署数据：32 章 5,183 诗句视频语料库（约 17.5h）和 18,000 诗句音频应用（约 17.5h）。

**📈 对比分析**

评估通过在同一数据上对四个架构系列（StyleTTS2、VITS2、Matcha‑TTS、Flow‑matching）进行 MOS 比较，结果显示 Flow‑matching 在专家单听 MOS 上达 4.6，且无明显上限，优于前者。

**⚠️ 局限性**

局限性包括：评估仅基于专家单听 MOS，缺乏多听者实验；文本侧可控 prosody 在当前自填流匹配 backbone 中不可行；无法实现多声纹零声纹讽歌；对重复音节深度的恢复有限；主观评估与客观指标不完全对应。

---

## 87. Relational Over-Regularization: Graph-Based AI-Generated Text Detection via Sentence Transition Deviation

**arXiv ID:** 2608.26694 | [PDF](https://arxiv.org/pdf/2608.26694v1)

**作者:** Hyeonchu Park `[一作]` (Chung-Ang University), Bugeun Kim `[通讯]` (Chung-Ang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于句对关系的结构化信号——Relational Over‑Regularization（ROR），并通过跨源句子关系图（CSFG）实现 AI 生成文本的检测。

**💡 创新点**

创新点在于把句对相似性方差的“相似性爆发”视为区分人类与 AI 文本的核心特征，并将这一信号编码为可学习的边特征 δ_ij，突破了传统基于单句或词级特征的检测局限。

**🔧 技术方法**

核心技术为：句子图构造（节点为句子，边携带相似度、位置距离、序列指示和 δ_ij），基于 EdgeConv 的 GNN 编码，联合二分类与生成器归属的多任务损失。

**📊 数据集**

使用了四个公开基准：HC3、AIGTBench、M4 和 MULTITuDE，涵盖多领域、多人类与多大模型生成的文本。

**📈 对比分析**

与多种基线（Fast‑DetectGPT、DNAGPT、LASTDE、Likelihood、RoBERTa、RADAR、DeTeCtive、SeqXGPT、CoCo）对比，CSFG 在二分类上达 97.14% 的准确率，FPR 1.57%，比最强图基线 CoCo 提升 11.14pp，并在未见 LLM 上保持近 0% 的 FPR。

**⚠️ 局限性**

局限性包括：ROR 对“超均匀”生成（如 GPT‑5）不敏感，无法捕捉低方差或高度结构化的人类文本；在文本扰动（重写、翻译）下易被污染导致 FPR 上升；以及图构造的二次相似度计算在长文档上计算成本高。

---

## 88. Rethinking Message Passing as Retrieval for Text-Attributed Graph Learning

**arXiv ID:** 2608.26732 | [PDF](https://arxiv.org/pdf/2608.26732v1)

**作者:** Jintang Li `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于检索增强的图学习框架，将传统消息传递视为对检索上下文的预测，并用简单 MLP 进行节点表示更新。

**💡 创新点**

创新点在于把图神经网络的邻域聚合等价于 RAG 的检索+融合过程，引入标签感知检索与聚合，并在理论上证明检索近似 softmax‑attention 消息传递且对误检索点具有鲁棒性。

**🔧 技术方法**

技术包括：文本嵌入（LLM）、近似最近邻检索（FAISS）+结构 PPR 复合得分、标签嵌入、检索增强的 MLP 更新以及对比实验和理论证明。

**📊 数据集**

使用八个文本属性图数据集：homophilic 的 Children、History、Photo、Computers、Arxiv 以及 heterophilic 的 Cornell、Texas、Wisconsin、Washington。

**📈 对比分析**

与多类 GNN、图对比学习、LLM 以及图 LLM 基线相比，提出的框架在所有数据集上均取得最高准确率（如 97.4% 在 Children），并在计算效率、抗攻击与深度扩展（抗过平滑）方面表现优异。

**⚠️ 局限性**

局限性包括对检索质量的依赖、超参数（k1、k2、λ）需要调节、标签信息的使用可能导致标签泄漏风险，以及在极大规模图上仍需进一步加速检索过程。

---

## 89. Five Primitives for Governing Autonomous AI Agents at Runtime

**arXiv ID:** 2608.26696 | [PDF](https://arxiv.org/pdf/2608.26696v1)

**作者:** Jiten Oswal `[一作]` (Aurite AI), John Cadeddu `[通讯]` (Aurite AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了企业级自治 AI 代理治理框架，将治理问题拆解为五个运行时原语：发现、身份、治理、证明与供应链；

**💡 创新点**

创新点在于将自治代理治理视为运行时问题，并给出五个不可约简的原语设计，强调代理的短暂性、行动不可预知性和自发发现所需的专门机制，区别于传统模型或构建时控制；

**🔧 技术方法**

使用的技术包括：互 TLS 传输层身份、基于证明的工作负载身份、策略评估中介、基于 hash 链的签名审计日志、每租户密钥管理、kill‑switch、离线独立验证器、以及供应链分析工具；

**📊 数据集**

论文未使用公开数据集，所有实验与评估基于少量私有试点部署中的治理日志与运行时指标；

**📈 对比分析**

论文未给出与其他方法的量化对比，仅报告实现状态与成本：关键路径延迟、侧车开销、失效时转为拒绝、集成前置成本等指标；

**⚠️ 局限性**

局限性包括：供应链原语尚未集成到控制平面；实验仅在少数私有部署中完成，缺乏大规模测量；验证和可观测性仍依赖内部工具；对固定小规模行动集的代理不适用。

---

## 90. Metamorphism: A mathematical challenge for antivirus technology

**arXiv ID:** 2608.27007 | [PDF](https://arxiv.org/pdf/2608.27007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 91. Beyond Atomic Layouts: Compositional Design Understanding with Vision-Language Models

**arXiv ID:** 2608.26716 | [PDF](https://arxiv.org/pdf/2608.26716v1)

**作者:** Yiyang Huang `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出组合布局理解任务，构建CoDeLayout数据集，并提出MASON后训练范式提升视觉-语言模型在该任务上的表现。

**💡 创新点**

①设计了专门针对组合布局的QA数据集与任务；②通过多模态对齐与层感知结构感知的后训练方法，有效缓解语义漂移与结构歧义。

**🔧 技术方法**

使用视觉-语言模型（如Qwen2.5‑VL）、多模态对齐（Grounding QA监督）、层感知结构感知、LoRA后训练等技术。

**📊 数据集**

CoDeLayout（约20K多层真实设计布局），包含合成QA注释及1K样本的对齐标注。

**📈 对比分析**

与启发式基线、开源7B VLM、闭源大型模型进行零样本和后训练对比；MASON在CoDeLayout上从GPT‑o3的79.68%提升至91.66%权重准确率，且仅用30%训练数据即可超越全数据微调，表现显著。

**⚠️ 局限性**

对Morphing等视觉语义极度混乱、数据稀缺的类别仍难以完全识别（仅70%），需更专业的层感知视觉编码器与更强跨模态对齐方法。

---

## 92. IBLTs Measure Before They Decode: Self-Sizing Set Reconciliation from Pre-Peeling Counts

**arXiv ID:** 2608.26537 | [PDF](https://arxiv.org/pdf/2608.26537v1)

**作者:** Min Wu `[一作]` (Hangzhou Dianzi University), Zhengyang Wei `[通讯]` (Nine Chapters (Zhejiang) Technology Co., Ltd.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种自尺寸化的 IBLT 协议，利用预先测量的计数通道在两轮交互中估计差集大小，从而实现低字节开销的数据库一致性检查。

**💡 创新点**

创新点在于发现 IBLT 计数矩阵本身可无额外开销估计差异规模，提供无偏估计、误差界定，并设计跨变体的映射感知构造与失败条件下的统计保证。

**🔧 技术方法**

采用了 IBLT、二次矩估计、Chi‑square 极限定理、失败条件下的量化界、可配置的两轮协议以及映射感知适配器等技术。

**📊 数据集**

使用了真实生产日志（NineData 90 天、41,603 次表级对齐）以及交叉引擎（Oracle–PolarDB/MySQL）和跨城市（China Mobile Redis/Pika）的重放实验。

**📈 对比分析**

通过与 Merkle 树样本地化、盲倍增、Strata‑first 估计等方法对比，实验显示自尺寸化在字节占用上仅 1.3–1.7× 预知容量，交互次数固定为 2 RTT，且在大多数差异规模下成功率超过 99.9%。

**⚠️ 局限性**

局限性在于依赖理想哈希和无碰撞计数；对极低 d 的预估仍需额外摘要；对极端不规则差异分布可能需要更大容量；实现需要保证固定输入视图和独立 seed，缺乏动态自适应增量更新。

---

## 93. 6.5% of the Neuro-Symbolic Literature Can Be Reproduced from Its Published Artifacts, a Six-Stage Audit Framework and First Instantiation

**arXiv ID:** 2608.26236 | [PDF](https://arxiv.org/pdf/2608.26236v1)

**作者:** Brandon Colelough `[一作]` (University of Maryland), Haowei Deng `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对神经符号人工智能领域的论文进行六阶段可重复性审计，评估同一代码工件是否能重新获得原报告结果

**💡 创新点**

提出首个大规模、系统化的同一工件重跑审计框架，并展示该框架在1304篇NSAI论文中的应用与结果

**🔧 技术方法**

采用文献检索、去重、标题/摘要筛选、全文合格判定、代码工件识别与完整性核查、环境构建、实验重跑与数据抽取等技术手段

**📊 数据集**

使用NSAI论文自带的数据集、模型权重和评测基准（无统一公共数据集），共覆盖455篇可重跑论文

**📈 对比分析**

通过对比原报告指标与重跑指标，发现只有18.7%的可重跑论文实现完全或部分重现，成功率与发布年份、会议类型无显著相关性

**⚠️ 局限性**

仅关注同一工件重跑，未考虑重新训练；对工件缺失、环境漂移等问题的分析受限，无法评估方法本身的真实性与泛化性能

---

## 94. Dispersive Forward Tree Search for Optimal Control: Coverage, Complexity, and Computation

**arXiv ID:** 2608.26314 | [PDF](https://arxiv.org/pdf/2608.26314v1)

**作者:** Shashank A. Deshpande `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于前向扩散的可扩散前向树搜索（DFT*）算法，针对可微分平坦的非线性系统给出确定性的有限样本近似最优性保证；

**💡 创新点**

创新点在于构造局部可扩散控制集、证明树覆盖与近最优性、并提出空间-时间支配剪枝实现多项式树规模；

**🔧 技术方法**

使用可扩散采样、Hermite 插值、分层控制仿真、GPU并行前向传播与剪枝；

**📊 数据集**

在Dynobench benchmark上对一系列平台（一阶/二阶无人车、拖车车、四旋翼）进行离线评估，并在动态障碍环境下演示实时规划；

**📈 对比分析**

与iDb-A*、SST*、RRT*+TO、FLASK 等基线比较，DFT* 在嵌入式级并行计算下获得与或优于最优解的方案，速度可达现有方法的10-15倍；

**⚠️ 局限性**

局限性在于可扩散采样在低采样预算下效率低，需要针对每个系统手工设计命令集，并且算法天然适合 GPU，CPU实现需进一步优化。

---

## 95. TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages

**arXiv ID:** 2608.26923 | [PDF](https://arxiv.org/pdf/2608.26923v1)

**作者:** Ireddi Rakshitha `[一作]` (Barclays), Ntakirutimana Pierre `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并预训练了 TabuLM，一种专为 Kinyarwanda 设计的语言模型，既能处理该语言丰富的形态学，又能理解结构化表格数据，并在自制的 TabQA-kin 评测集上进行微调。

**💡 创新点**

创新点包括：1) 将 KinyaBERT 的双层形态学编码与可学习的行/列/单元格嵌入相结合；2) 引入表格结构注意力偏置；3) 设计两项新的预训练目标（Masked Cell Recovery 与 Column Type Prediction），实现对表格结构的深度建模；4) 这是首个针对低资源非英语语言进行表格预训练的工作。

**🔧 技术方法**

技术手段：KinyaBERT 的两层 Transformer（形态学编码层 + 句子层）；行/列/单元格嵌入作为输入补充；表格结构注意力偏置在每层注意力矩阵中加入；Mask Cell Recovery（全单元格遮掩并恢复）和 Column Type Prediction（列类型预测）作为额外预训练任务；微调阶段采用 TAPAS 样式的单元格选择头，并结合查询导向的行列过滤。

**📊 数据集**

数据集：172 份来自卢旺达政府（NISR、RAB、REB、MoH）的表格，约 35,000 个单元格，用于预训练；TabQA‑kin（526 个 QA 对）覆盖 31 张表，按 420/106 的训练/验证比例划分，包含 lookup、comparison、aggregation、count 四类问题。

**📈 对比分析**

比较方法：在 TabQA‑kin 验证集上对齐模型的答案与黄金单元格，计算 Exact Match；对比基线 mBERT、XLM‑R、KinyaBERT（无表格预训练）、以及 GPT‑4o/GPT‑4o‑mini（零样本）。TabuLM 在整体 EM 上达到 62.0%，比所有微调基线高 5.7–12.7 分；在 aggregation 子任务上，TabuLM 与 KinyaBERT 等微调模型显著突破 GPT‑4o 的 64% 上限，显示微调能解决 LLM 在聚合推理中的结构缺失。

**⚠️ 局限性**

局限性：1) 形态学分析器为专有闭源，限制可复现性；2) 预训练语料仅 172 张表，规模有限；3) 验证集规模小（仅 50 条可评估条目），统计显著性有限；4) 学习到的表格注意力偏置几乎为 0，可能需要更大语料或更长训练；5) 目前仅在 Kinyarwanda 上验证，未充分评估跨语言迁移。

---

## 96. FIDA: Feature Instability-Driven Attack on Self-Supervised Facial Representation

**arXiv ID:** 2608.26861 | [PDF](https://arxiv.org/pdf/2608.26861v1)

**作者:** Zhiyang Chen `[一作]` (Nanjing University of Aeronautics and Astronautics), Liming Fang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在自监督面部表征模型中提出并实现了一种新的后门攻击框架FIDA，能够在保持模型正常功能的同时，针对后门样本诱导特征的不稳定性，逃避多种基于扰动的检测与防御；

**💡 创新点**

创新点包括：① 通过引入“特征不稳定损失”（Feature Instability Loss）使触发输入在添加噪声时产生显著特征差异，从而削弱传统防御依赖的特征稳定性假设；② 采用语义触发器（如眉毛、唇色、眼镜、胡须）实现隐蔽且具备一定物理可复制性的触发方式；

**🔧 技术方法**

技术手段：自监督模型微调（SimCLR、BYOL、MoCo等）；损失函数设计（攻击损失、忠实度损失、特征不稳定损失）；语义触发器实现（基于68点面部关键点的几何变换）；评估与防御对比（STRIP、STRIP-CL、SymND、ASSET、DECREE、DeDe、BARBIE等）；

**📊 数据集**

使用的数据集：自监督预训练采用WIKI_CROP和FairFace；下游任务评估使用RAF-DB（情感识别）、UTKFace（种族、年龄分类）以及VGGFace2（跨域检索）等；

**📈 对比分析**

与Naive Attack、BadEncoder、DRUPE、GhostEncoder等方法对比，FIDA在所有实验设置下均实现近100%的攻击成功率（ASR），且后门对正常分类精度影响极小（BA>95%）；同时在STRIP、SymND等扰动式防御中几乎无检测率，展示了显著的防御规避效果；

**⚠️ 局限性**

局限性：在高强度模糊、JPEG压缩等物理退化下攻击效果下降；需要白盒访问预训练模型；触发器的物理实现仍缺乏真实世界验证；可能存在误触发风险，且在不同人种、年龄等群体间的触发率和误报率尚未完全评估。

---

## 97. J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

**arXiv ID:** 2608.26582 | [PDF](https://arxiv.org/pdf/2608.26582v1)

**作者:** Gyouk Chu `[一作]` (KAIST), Eunho Yang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Judge Co‑Adaptation from Zero（J‑Zero）框架，让挑战者（Challenger）、求解者（Solver）与评判者（Judge）在无外部数据的自进化循环中共同演化，提升可验证与不可验证任务的性能。

**💡 创新点**

核心创新是让评判者随同挑战者与求解者一起进化，并利用角色不对称与子任务放大生成的两类内循环偏好对，持续为评判者提供改进监督，从而突破固定评判器的性能上限。

**🔧 技术方法**

采用对抗式 Challenger–Solver 训练、组相对策略优化（GRPO）更新策略、Bradley–Terry 损失训练评判者，以及任务难度奖励与重复惩罚机制。

**📊 数据集**

在 Qwen3‑4B 与 Qwen3‑8B 基础模型上评估 11 个可验证任务（7 数学推理、3 通用推理、IFEval）和 3 个不可验证任务（AlpacaEval 2.0、Arena‑Hard‑v2.0、EQ‑Bench Creative Writing v3）。

**📈 对比分析**

与基线模型、R‑Zero、G‑Zero 对比，J‑Zero 在可验证域平均提升约 9.5/7.9 分，在不可验证域提升约 11.2/10.2 分，并且在十轮迭代中持续提升，而其他方法在两轮后就出现衰退。

**⚠️ 局限性**

受计算限制仅实验 4B/8B 参数规模，评判者采用判别式奖励模型而非生成式 LLM，未测试更大模型或链式思维任务，也未探索生成式评判者的协同进化。

---

## 98. MeshPriorDiT: Hierarchical Modeling for Action-Conditioned Cloth Dynamics

**arXiv ID:** 2608.26766 | [PDF](https://arxiv.org/pdf/2608.26766v1)

**作者:** Zihang Wang `[一作]` (Tsinghua University), Shuo Feng `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种层次化的动作条件布料动力学模型 MeshPriorDiT，将未来布料运动分解为结构化的网格先验（Mesh Prior）和生成残差（Residual DiT），并在自回归预测中实现高质量的长期轨迹生成。

**💡 创新点**

创新点在于：① 明确将局部物理传播和全局残差校正拆分为两条独立分支，便于诊断与调试；② 使用条件流匹配训练残差生成器，兼顾稳定性与可控性；③ 通过拓扑感知解码（Jacobi平滑）保证残差在相邻顶点之间保持一致，从而同时提升全局误差和边缘应变质量。

**🔧 技术方法**

主要技术包括：MeshGraphNets 图神经网络实现局部网格先验；Diffusion Transformer（DiT）结合条件流匹配生成残差；拓扑感知残差解码与 Jacobi 平滑；自回归多步预测框架；以及基于软体仿真的 ClothDynamics 数据集。

**📊 数据集**

使用 SoftGym 中的 500,000 个布料动力学窗口数据，涵盖三种操作任务：Diagonal Fold、Corner‑to‑Side Fold、Vertical Lift，训练集 400k、验证集 50k、测试集 50k，确保无轨迹重叠。

**📈 对比分析**

与基准 GNN‑Only（仅网格先验）和 DiT‑DDPM（纯生成模型）进行比较，评估指标为 Global MSE 和 Edge‑strain MSE。MeshPriorDiT 在 15 步预测中 Global MSE 相较 GNN‑Only 降低 43.4%–49.6%，相较 DiT‑DDPM 降低 75.0%–80.0%；在 35 步长时序中保持 32.9%–49.6% 的优势，同时 Edge‑strain 误差保持与 GNN‑Only 相近。

**⚠️ 局限性**

局限性包括：① 残差幅度和拓扑平滑参数需手工调节，影响轨迹精度与网格质量之间的平衡；② 在先验已表现优秀的任务（如 Diagonal Fold）上，残差校正增益有限甚至略有退化；③ 生成器训练和推理过程消耗较多计算资源（NFE≈100），对实时部署构成挑战。

---

## 99. Hull First, Wake Second: Wake-Reliance Suppression for Robust Maritime Vessel Detection

**arXiv ID:** 2608.26665 | [PDF](https://arxiv.org/pdf/2608.26665v1)

**作者:** Yefan Wang `[一作]` (University of Shanghai for Science and Technology), Yusen Wu `[通讯]` (Fujian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对海上船舶检测中的“wake依赖”问题，提出了HullWake框架，先以船体为主导特征，再利用定向wake特征做辅助验证，避免检测模型过度依赖船尾水波；

**💡 创新点**

创新点在于：①构造基于proposal锚定的双向wake提取通道；②通过wake响应监督、wake衰减一致性、wake仅置信度抑制以及船体–wake去相关化等多重损失，控制wake特征仅做辅助；③提出了针对wake鲁棒性的评估指标与专门的数据集Curated‑Wake；

**🔧 技术方法**

技术方法包括RoIAlign、方向注意力的wake特征聚合、有限融合头、wake衰减一致性损失、wake仅置信度抑制和特征去相关化；

**📊 数据集**

使用了从 Ships/Vessels in Aerial Images、SMD 和 SeaDronesSee 三个公开数据集筛选并重新标注约1万张图像，新增船体、wake、wake‑类似负样本以及水体杂散分割标签；

**📈 对比分析**

与传统box检测器（Faster R‑CNN、Cascade R‑CNN 等）和基于mask的分割模型（Mask2Former、PIDNet）在 Curated‑Wake 上进行对比，HullWake 在整体 AP 上提升 2.9 点、AP_NoWake 提升 6.0 点、FP_WakeLike 减少 32% 以上、WG‑AP 提升 5.7 点，且在所有来源子集上均保持稳定提升；

**⚠️ 局限性**

局限性包括：仍需更多多样化海况下的标注；对极端慢速或无wake船体的检测仍有挑战；以及模型对 wake 提取通道长度/宽度参数敏感，需要进一步自适应设计。

---

## 100. When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models

**arXiv ID:** 2608.26187 | [PDF](https://arxiv.org/pdf/2608.26187v1)

**作者:** Dai Shi `[一作]` (University of Cambridge), José Miguel Hernández-Lobato `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究大语言模型在给定约束下是否能抛弃默认完成并生成新的结构，验证其在范畴论框架中的“跳跃”第二步。

**💡 创新点**

创新点在于提出正式的跳跃实例定义、可证的实例族、Kan默认率度量，并通过实验评估前沿LLM在此测试上的表现。

**🔧 技术方法**

采用范畴论的Kan扩张作为默认完成，构造有限类别和可计算约束，使用机检证书和实验测评技术。

**📊 数据集**

使用9个自生成的“点链”族认证实例（6个可枚举，3个通过定理），未采用公开数据集。

**📈 对比分析**

通过比较四个前沿模型（GPT‑5.6 Luna Pro、Claude Sonnet 5、Gemini 3.1 Pro、DeepSeek V4 Pro）的跳跃准确率和Kan默认率，结果显示所有模型在所有实例上的Kan默认率为0，跳跃准确率高，困难度主要受搜索/预算限制影响。

**⚠️ 局限性**

局限性包括仅评估跳跃的选择步骤，未评估约束生成或框架发明；仅使用固定codomain；实例生成与证书复杂度有限；可能受输出预算和内容过滤器影响。

---

## 101. Pseudodeterminism and MA != NP^BPP in Communication Complexity

**arXiv ID:** 2608.26425 | [PDF](https://arxiv.org/pdf/2608.26425v1)

**作者:** Thomas Watson `[一作]` `[通讯]` (University of Memphis), Thomas Watson (University of Memphis)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在部分布尔函数上，零误差随机通信复杂度与双侧误差伪决定性通信复杂度存在指数级分离，并推广到其它复杂度类。

**💡 创新点**

创新点在于将先前两侧误差随机协议的上界改为零误差随机协议的上界，并用更简单的白盒提升技术完成分离，显著改进了 Göös 等人 2026 年的结果。

**🔧 技术方法**

主要使用了白盒查询到通信提升（query-to-communication lifting）技术、结构化子矩形分解与潜能函数迭代分析，结合随机决策树与伪决定性协议的性质。

**📊 数据集**

本研究不涉及实验数据或数据集，完全是理论证明。

**📈 对比分析**

与先前工作相比，本文在相同问题上取得了更强的分离（从 O(log N) 到 0.01√N 的上界），证明结果与此前的双侧误差随机下界相匹配。

**⚠️ 局限性**

局限性包括：证明依赖于特定的两方基函数（g ＝ 内积模 2 的变体）以及固定的参数设定，未能推广到更广泛的基函数或完全无参数化的情形；此外，证明过程较为复杂，尚需进一步简化。

---

## 102. Reward-Informed Sparse Autoencoders and the Solution-Completeness Confound

**arXiv ID:** 2608.26136 | [PDF](https://arxiv.org/pdf/2608.26136v1)

**作者:** Tanvi Nagilla `[一作]`, Shayaan Uddin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过奖励筛选的稀疏自编码器，对 Llama‑3.1‑8B 的 GRPO 轨迹激活进行训练，提取出区分高奖励（好推理）与低奖励（坏推理）的特征。

**💡 创新点**

创新点在于将 RL 奖励作为无标签监督，构造奖励驱动的稀疏自编码器（RI‑SAE）以及完整的对照实验组，以检验所得到的特征是否真正反映推理质量。

**🔧 技术方法**

采用 JumpReLU 稀疏自编码器、UMAP 可视化、TF‑IDF 文本分类和结构性特征（长度、闭合推理块、方框答案）进行对照。

**📊 数据集**

数据集为从公开 GRPO 轨迹池中筛选的 2000 条文本（1000 条高奖励≥2.0，1000 条低奖励≤0.5）以及 Gemma‑2‑2B 在 GSM8K 上的 5 个检查点。

**📈 对比分析**

对照实验表明，在选取的稀疏子空间中类别分离度达到 silhouette 0.79；但仅用 TF‑IDF 文本分类即可得到 0.75–0.83 的 AUC，结构特征单独可达 0.70；而未受奖励引导的普通 SAE 则几乎不分离。

**⚠️ 局限性**

限制包括奖励仅是推理与否的代理，分离主要由完整解的结构信号驱动；评估样本小、缺乏随机种子与置信区间；未对奖励过滤与领域内训练的因果关系进行隔离；以及当前仅为描述性结果，缺乏因果验证。

---

## 103. Graph-Guided Selective Unlearning for Language Models: Controlling Support Routes Beyond Forget Seeds

**arXiv ID:** 2608.26743 | [PDF](https://arxiv.org/pdf/2608.26743v1)

**作者:** Waqas Khan `[一作]`, Estrid He `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图引导的选择性忘记方法 GraphSU，扩展忘记范围以更有效地删除模型中保留的目标知识。

**💡 创新点**

创新点在于将忘记范围视为支持路由控制问题，通过构造多视角加权支持图并传播删除压力，自动识别并对高风险邻居分配梯度忘记强度，从而实现比仅对种子样本忘记更精细、更安全的知识移除。

**🔧 技术方法**

使用图扩散（个性化 PageRank）进行删除压力传播，采用多视角相似度（实体、关系、尾部、语义、梯度对齐）构造支持图，并在此基础上应用软遗忘损失（熵最大化、非可能性、表征排斥）与保留损失（交叉熵、KL 正则）进行训练。

**📊 数据集**

使用 TOFU（作者信息 QA 生成的仿真基准）和 PISTOL（结构化事实互联基准）两套数据集，在 GPT‑2 Medium 与 Llama‑3.2‑3B‑Instruct 上进行实验。

**📈 对比分析**

与 GA、GA+KL、DPO、NPO‑TR、AltPO、GRU、SELU 等现有方法以及 Seed‑Only 对比，GraphSU 在保持保留困惑度 ≤10 的前提下，软泄漏率降低最多 49.5%（完整删除），并在所有删除设置下均取得最低软泄漏，优于所有基线。

**⚠️ 局限性**

局限包括：1）对部分删除任务仍存在显著泄漏，难以完全抑制敏感片段；2）软泄漏度量仅覆盖预设恢复路由，缺乏正式的可证伪消除保证；3）构建支持图需要一次性离线开销，且在频繁删除请求场景下成本较高。

---

## 104. Daydreaming: Stealing Hidden Agent Skills through Black-Box Task Interaction

**arXiv ID:** 2608.26733 | [PDF](https://arxiv.org/pdf/2608.26733v1)

**作者:** Yu-Lin Tsai `[一作]` (University of California Berkeley), Chia-Mu Yu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种仅通过正常任务交互的黑盒攻击，能够在不直接访问或披露技能文件的前提下，从仅得到的最终输出（甚至仅文本）中重构托管的Agent技能。

**💡 创新点**

创新点在于：①将技能偷窃建模为执行路径上的系统识别问题；②提出三阶段分层假设精炼（属性→候选计划→文件版本）并使用可区分任务进行自适应查询；③系统化定义三层可观测性（Output、Trace、Differential），证明即使在最严格的Output级别也能恢复大量功能。

**🔧 技术方法**

技术手段包括：大语言模型作为攻击者模型；shadow agents（无技能与候选技能）进行本地行为预测；生成式任务构造与差异化任务设计；属性推断、候选计划比较、文件版本精炼的循环精炼框架；以及对输出、提示、日志等防御措施的评估。

**📊 数据集**

使用了SkillsBench中的7个真实场景技能作为目标，并选取7个hold‑out任务作为评估集；实验中采用多种语言模型（GPT‑5.6、Claude系列等）和多种工具/调度策略进行评测。

**📈 对比分析**

实验对比了之前的BBS、SigLeak以及固定探测和一次性合成基线。评价指标为任务成功率SR和行为检查U。结果显示：在Output级别下，攻击恢复率高达86.8%，SR提升约0.23，U提升约0.24；在Trace和Differential级别进一步提升；在不同模型与工具策略下表现稳健；并对攻击预算、可迁移性等方面进行了敏感性分析。

**⚠️ 局限性**

局限性：①无法恢复原始源代码，结构恢复指标低；②对不同模型/策略的迁移性不均匀；③攻击仍需多次查询，成本相对较高；④若存在更强的输出精度限制或接口受限，攻击效果可能受限。

---

## 105. Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval

**arXiv ID:** 2608.26414 | [PDF](https://arxiv.org/pdf/2608.26414v1)

**作者:** Jiale Wei `[一作]` (Karlsruhe Institute of Technology), Rainer Stiefelhagen `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Case2Flow任务并构建FlowAtlas数据集，评估多模检索方法，提出无训练的CRISP评分提升检索效果。

**💡 创新点**

首次将临床案例与指南流程图对应检索作为任务，提供合成案例-流程图对数据集，设计CRISP评分改进Late‑Interaction模型的背景与关键词匹配缺陷。

**🔧 技术方法**

使用LLM生成合成案例与重写、Docling/OCR+VLM Caption进行文本检索、ColPali、ColQwen3、Qwen3‑VL‑Emb等Late‑Interaction视觉语言模型以及CRISP分数修正。

**📊 数据集**

FlowAtlas（202份流程图、1911对案例-流程图），来源CDC、WHO、ESMO、Onkopedia；MedGUIDE基准；PMC‑CaseReport公开病例用于医生评估。

**📈 对比分析**

与文本检索、CLIP、Qwen3‑VL‑Emb、ColPali、ColQwen3等方法在单源、混合、外部三种设定下对比；CRISP在混合池Recall@1提升18.71pp、MedGUIDE提升4.94pp，整体保持最高Recall@1并显著优于基线。

**⚠️ 局限性**

合成案例缺乏真实噪声，评估仅限闭集（202份流程图+55份外部），CRISP受底层特征限制，数据仅英文且覆盖范围偏向肿瘤学，医生评估样本有限且单评者。

---

## 106. Approved Too Late: Verdict Staleness in LLM-Guarded Self-Adaptive Systems

**arXiv ID:** 2608.26306 | [PDF](https://arxiv.org/pdf/2608.26306v1)

**作者:** Ilai Shraga `[一作]` (University of Cambridge), Lior Gorelik `[通讯]` (Open University of Israel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了大语言模型（LLM）作为自适应系统（SAS）执行阶段守护栏时可能出现的时间检查到时间使用（TOCTOU）危害，并提出一种基于安全边距和特征波动率的“Freshness-Bounded Shield”来估算已发放审批的有效期，从而降低审批失效率。

**💡 创新点**

创新点包括：①定义并量化“Verdict Freshness”概念，区分三种不同的失效指标；②提出一种无显式动态模型的有效期估算方法（Freshness-Bounded Shield）；③制定了“Freshness Contract”，要求审批不仅在检查时正确，还需在使用时保持有效并提供相应回退方案。

**🔧 技术方法**

主要技术手段为：固定动作重放实验、基于确定性参考检查器的oracle标注、LLM判定器（judge）与Freshness-Bounded Shield的组合、EMA（指数移动平均）估算特征波动率、距离安全边界的安全边距计算，以及Python/SciPy实现的实验框架。

**📊 数据集**

使用了五个可复现的自适应系统环境：IoT网络适配、Edge–Cloud资源调度、架构自愈、适应性目标检测、后台作业调度。每个环境都有自己的安全判定函数和阈值。

**📈 对比分析**

在实验中将Freshness-Bounded Shield与无效期检查的 baseline 以及三种消融版本进行对比。结果显示，在重放步长 K=8 时，该 Shield 能将 oracle‑labeled approval‑expiry 从 3.4–24.7% 降低到 0–1.8%，相对降低 82–100%。在某些环境中，Shield 在保持 0% 失效率的同时不降低平均奖励。

**⚠️ 局限性**

局限性包括：仅在基于步进的仿真环境中验证；使用固定的参数设定（如 α=0.1、δ_max 等），缺乏对不同阈值和动态模型的泛化评估；仅对重放实验而非完整的实时 Judge+ 流程进行评估；有效期估算基于经验波动率，未给出正式安全保证；并未考虑不同物理时间尺度下的 K 与 Δ 的对应关系。

---

## 107. Hierarchical Channel Stacking: A Structured Decision Framework for AI-Generated Image Detection

**arXiv ID:** 2608.26648 | [PDF](https://arxiv.org/pdf/2608.26648v1)

**作者:** Saifullah Shoaib `[一作]` (Missouri University of Science and Technology), Peggy Lindner `[通讯]` (Missouri University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Hierarchical Channel Stacking (HCS) 框架，利用 ResNet‑50 的中间激活并通过 60 维层级结构的概率表示来判别 AI 生成图像。

**💡 创新点**

创新点在于把中间特征按层级堆叠成可解释的 60 维表示，既保持了高检测性能，又为每个层次的贡献提供了透明的分析接口。

**🔧 技术方法**

使用 ResNet‑50 作为特征提取器；对选定通道做 Level‑1 随机森林分类；将 60 个通道概率拼接后再送入 Level‑2 随机森林；采用通道排名、留一法生成 OOF 特征，并用组级 Shapley 值量化各层贡献。

**📊 数据集**

使用了结合 ForenSynths（GAN 生成器）和 GenImage（扩散模型）数据集的平衡基准，共 89,232 张图像，包含 50% 真实、50% 伪造，GAN 与扩散模型各占 25%。

**📈 对比分析**

与 ResNet‑50+LR/ RF/ MLP、CNNDetection 端到端模型以及 fine‑tuned ResNet 进行对比；HCS 在测试集上达到 86.7% 的准确率和宏 F1，优于单层或双层变体；虽然检测精度略低于 fine‑tuned ResNet（99.4%），但提供了结构化可解释性。

**⚠️ 局限性**

局限性包括：评估仅在已知生成器的 held‑out 子集；未检验对未知生成器、未知模型族的泛化；未评估压缩、尺寸变换等后处理对性能的影响；并且结果可能受到数据集构造的影响。

---

## 108. Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors

**arXiv ID:** 2608.26175 | [PDF](https://arxiv.org/pdf/2608.26175v1)

**作者:** Mantas Lukauskas `[一作]` `[通讯]` (Hostinger), Mantas Lukauskas (Hostinger)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多语言prompt压缩器进行系统化审计，评估其在十种语言、七种文字系统上的性能，尤其关注英语监督下的跨语言转移差距。

**💡 创新点**

提出一种严格的跨语言压缩基准：使用并行语料、预算匹配与统一的归一化指标，首次揭示学习型压缩器在非英语文本上普遍存在的转移gap，并验证多语言监督能消除该差距。

**🔧 技术方法**

利用提取式压缩技术（LLMLingua‑2、Kompress‑v2、XProvence）和确定性基线（TF‑IDF、词形还原+停用词、前缀截断、随机删除），并结合长上下文压力测试与翻译‑先压缩的对照实验。

**📊 数据集**

采用10种语言（EN、PL、FI、ET、LV、LT、UK、ZH、AR、HI）的全并行语料，主要来自FLORES/Belebele阅读理解、MultiEURLEX法律文档以及长文档集合；每条语料都有英文原文与十种目标语言的平行版本。

**📈 对比分析**

通过预算匹配和归一化的“可用上下文利用率 U”评估压缩效果，并计算跨语言转移gap。结果显示：在0.33的保留率下，英语保留约57–62%的信息，但非英语如立陶宛语、乌克兰语仅保留10–24%，中文甚至低于无上下文水平；确定性方法无明显跨语言gap；多语言监督的XProvence在v1版下无gap，但v2版在中文上会过度裁剪。翻译‑先压缩在大部分语言上能以更低代价获得与本地压缩相当或更优的准确率。

**⚠️ 局限性**

局限性包括：仅覆盖十种语言且多为印欧语系；使用FLORES翻译文本可能导致译文与原生语料差异；长上下文实验规模有限，未覆盖实际RAG场景下的数千–数十万token；查询感知压缩与无查询压缩的对比不完全对等；统计检验未做多重校正，可能夸大单个显著性；以及对中文的分词与预算控制问题尚未彻底解决。

---

## 109. Diff Mining: Logit Differences Reveal Finetuning Objectives

**arXiv ID:** 2608.26462 | [PDF](https://arxiv.org/pdf/2608.26462v1)

**作者:** Greg Kocher `[一作]` (Columbia University), Julian Minder `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Diff Mining 的模型差异化框架，通过比较微调模型与基模型的 logits 差异来识别微调后模型学习到的行为，并用这些差异构造可解释的 token 集合。

**💡 创新点**

创新点在于仅依赖 logits 差异而不需要内部激活信息，且通过两种聚合方式（Top-K 频率与 NMF 主题分解）分别实现简单高效的提取和多目标拆分；该方法在不知晓微调域的情况下即可有效定位微调目标，并能揭示隐藏的偏置。

**🔧 技术方法**

核心技术包括：1）logits 差分计算；2）Top-K 频率聚合；3）非负矩阵分解（NMF）聚合；4）与大型语言模型（如 GPT‑5-mini）结合的判定与评估流程。

**📊 数据集**

使用公开的通用域参考语料（如 FineWeb、CulturaX、GSM8K 等）以及实验设计的人工合成微调数据，评估隐藏偏置场景（Auditing Games）以及多主题微调（Cake Bake + Comments）等多种数据集。

**📈 对比分析**

与基准方法 ADL 进行对比，结果显示 Diff Mining 在多种微调与混合比例下能提取更多相关 token，并显著提升解释性 agent 的评估得分；在真实世界的隐藏偏置检测任务中，Diff Mining 能识别约三分之一的偏置，并能通过 NMF 解析多主题微调。

**⚠️ 局限性**

局限性包括：1）评估场景相对有限，缺乏更广泛的真实模型与任务测试；2）依赖 LLM 判定与评估，存在非确定性与方差；3）对极少样本或极低 K 值的鲁棒性下降；4）未探讨不同 tokenizer 或跨语言的兼容性问题。

---

## 110. Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators

**arXiv ID:** 2608.26148 | [PDF](https://arxiv.org/pdf/2608.26148v1)

**作者:** Jonas Länzlinger `[一作]` (University of St Gallen), Bruno Rodrigues `[通讯]` (University of St Gallen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出一种透明的链接框架，将语音声学特征与 DSM‑5 抑郁症状指标关联，并在本地边缘设备上实时进行可解释评估。

**💡 创新点**

创新点在于：① 明确的多层特征‑指标映射规则实现可解释性；② 端到端隐私保护的边缘执行；③ 通过指数滑动平均实现持续症状监测。

**🔧 技术方法**

主要技术包括：Silero VAD 语音分离、低层声学特征提取、统计功能化、EMA 平滑、阈值决策与规则引擎；所有计算在 CPU 上实现。

**📊 数据集**

使用公开的 DAIC‑WOZ 语音数据集（64 名受试者、PHQ‑8 评分对应 DSM‑5 指标）进行实验。

**📈 对比分析**

与传统黑盒模型对比，本框架在 DAIC‑WOZ 上表现为方向一致的特征‑指标相关性（效应量中等，未显著通过 FDR 校正），并在 M1 MacBook Pro 上实现低于 1 s 的实时推断延迟，表明边缘可行性。

**⚠️ 局限性**

主要限制包括：样本量有限、数据来自单次访谈而非长时自然语音、仅覆盖四个 DSM‑5 指标、缺乏多模态输入、以及模型在不同语言/人群中的校准仍待验证。

---

## 111. Technical Comparative Benchmarking Study: Advanced AI Hybrid Methods for Renewable Energy Farm Optimization and Forecasting

**arXiv ID:** 2608.26613 | [PDF](https://arxiv.org/pdf/2608.26613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 112. IoMT-SecAlarmBench: A Counterfactual Benchmark for Integrity Attacks in IoMT

**arXiv ID:** 2608.26416 | [PDF](https://arxiv.org/pdf/2608.26416v1)

**作者:** Emmanuel C. Ugwuabonyi `[一作]` (University of Maryland Baltimore County), Dmitri Perkins `[通讯]` (University of Maryland Baltimore County)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并发布了一个半合成的IoMT攻击基准（IoMT‑SecAlarmBench），通过在真实的ECG+PPG记录上注入四种攻击形态、四个严重度级别、两种可行性条件以及重放攻击，并为每个注入窗口提供完整的因果真值和对抗真值信号。

**💡 创新点**

创新点在于：①首次提供可追溯至因果模型的对抗真值；②通过三因子（形态×严重度×可行性）实现细粒度攻击注入；③在双模态（生理+网络）数据上完成攻击注入并提供泄漏审计；④引入无阈值、误报匹配的评估协议，揭示传统检测方法在攻击与设备故障区分上的根本局限。

**🔧 技术方法**

技术方法包括：基于结构因果模型（SCM）的攻击与事件区分框架；使用四种攻击形态（Spike、Stuck‑at、Drift、Bias）和重放攻击的控制注入；对每种攻击的可行性边界（内/外）进行参数化；构建完整的评估协议（AUPRC、F1、匹配误报、效能指标）；以及在实验中使用六种无监督检测器（Isolation Forest、LSTM‑AE、USAD、OmniAnomaly、GDN、TranAD）。

**📊 数据集**

主要使用的数据集为：PhysioNet/CinC 2015的627条ECG+PPG核心记录（共78,729个10s窗口），WUSTL‑EHMS‑2020的网络‑生理双模态流（用于泄漏审计），以及MIMIC‑III Waveform（仅用于示例验证，未注入）。

**📈 对比分析**

对检测方法的比较采用无阈值AUPRC、误报匹配（10%误报预算）和多指标（F1、FAR、宏F1、MCC、Cohen's κ）进行。实验结果显示：所有方法的AUPRC均在0.82–0.90之间，最大提升约1.5–1.7倍于随机基线；但在重放攻击和短暂尖峰攻击上，各方法的性能仅略高于随机（AUPRC 1.2–1.7×），表明存在根本性性能极限。传统Isolation Forest在跨通道特征上表现最佳，但同时在攻击与设备故障之间产生较大混淆。

**⚠️ 局限性**

局限性包括：①攻击仅注入单一通道（PPG）；②基准为半合成，缺乏对真实临床攻击的直接验证；③缺乏持续长时间ICU记录和多协议跨层数据；④对真实生理事件的对抗真值无法提供；⑤当前评估未涵盖多通道攻击定位与临床报警级别的因果推断。

---

## 113. FIRSTPASS: A Multi-Domain, Multi-Round Peer Review Dataset Grounded in Real Editorial Outcomes

**arXiv ID:** 2608.26129 | [PDF](https://arxiv.org/pdf/2608.26129v1)

**作者:** Prabhjot Singh `[一作]` (University of Texas at Austin), Josh Durkee `[通讯]` (Western Kentucky University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个多学科、多轮、基于真实编辑决策标签的同行评审数据集 FirstPass，并提出三项评估任务（评审生成、评审更新、结果预测）。

**💡 创新点**

创新点在于：①跨生物、化学、神经科学、物理和地球科学五个学科，消除单一领域偏差；②包含完整的多轮作者-评审对话，捕捉科学验证过程；③用实际编辑决策（STANDARD/EXTENDED）作为标签，取代以往的风格化模仿评测；④公开完整的解析、审计及评测脚本，确保可复现性。

**🔧 技术方法**

技术手段包括：使用 Springer Nature OpenAccess API 抓取 PDF，Gemini‑3.1‑flash‑lite‑preview 进行文档解析并转 JSON，自动化审核确保内容完整；模型方面使用 Qwen2.5‑7B‑Instruct 进行 fine‑tune 以完成任务。

**📊 数据集**

数据集来源为 Nature Communications 2023‑2025 年间的公开同行评审文件，包含 3,668 条完整多轮记录，覆盖五大学科，平均评审文本约 2,155 字。

**📈 对比分析**

在任务 3（结果预测）上，以 Qwen2.5‑7B‑Instruct fine‑tune 的模型在所有学科上取得 80.5% 准确率、78.2% F1‑macro，较 Gemini‑3.1‑flash‑lite‑preview 零样本基线提升 10.4 个百分点。

**⚠️ 局限性**

局限性包括：仅来自单一期刊（Nature Communications），可能带来领域偏倚；样本集中在 2023‑2025 年间，缺乏历史对比；未包含讨论/结论部分，可能限制模型对整篇论文深度评判的学习；数据被滥用生成合成评审的风险。

---

## 114. Reason in the Words You Speak: Idiolectal Paraphrasing Off-Policy Traces for Reasoning Distillation in VideoLLMs

**arXiv ID:** 2608.26684 | [PDF](https://arxiv.org/pdf/2608.26684v1)

**作者:** Ji Soo Lee `[一作]` (KAIST), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视频推理中，提出 Echo-GRPO 框架，将外部教师轨迹改写为学生模型自己的语言风格，构建 VideoEcho-R1 模型进行推理蒸馏；

**💡 创新点**

创新点在于识别混合策略 GRPO 中梯度裁剪导致语义关键信息丢失的失败模式，并通过 Dual-Reference Decoding 进行“idiolectal”改写，使离线轨迹与学生分布对齐，从而提升梯度学习效果；

**🔧 技术方法**

核心技术包括 Echo-GRPO 重新写作框架、Dual-Reference Decoding（语义参考与分布参考的乘积专家模型）、GRPO、视频多模态大语言模型；

**📊 数据集**

使用了五个视频推理基准（如 YouCook2、MSRVTT‑QA 等）以及 InternVL3.5‑4B、Qwen3‑VL‑4B、Qwen3‑VL‑8B 三种多模态 LLM；

**📈 对比分析**

与传统 GRPO、监督微调（SFT）和混合策略 GRPO 进行对比，VideoEcho‑R1 在所有三种后端模型和五个基准上均实现了显著性能提升，提升幅度一般超过 5%–10%；

**⚠️ 局限性**

局限性包括：仍需教师轨迹支持，改写过程依赖于教师的质量；对非视频多模态任务的适用性待验证；改写与剪裁机制的超参数选择仍需经验指导；

---

## 115. Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Paper

**arXiv ID:** 2608.26596 | [PDF](https://arxiv.org/pdf/2608.26596v1)

**作者:** Rongjin Li `[一作]` (Beijing University Of Posts And Telecommunications), Xu Sun `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 VERA‑RL 框架和 VERA‑13K 数据集，用多模态 RL 训练模型在学术论文中进行可验证的错误检测。

**💡 创新点**

将 Scan 任务拆解为 Reason–Verify–Scan 三阶段，并设计细粒度奖励（推理完整度、证据对齐、错误精度）实现从有证据到无证据的验证能力提升。

**🔧 技术方法**

采用 DAPO 强化学习、逐步任务分解、细粒度奖励以及多模态 SFT（Qwen3‑VL‑8B）和 OCR 处理技术。

**📊 数据集**

使用 VERA‑13K（12,900 例、6 类科学错误的 Reason–Verify–Scan 链）以及 ScholScan 进行跨基准评估。

**📈 对比分析**

在 Reason、Verify、Scan 上与 Gemini 3 Pro、Qwen3‑VL‑235B‑A22B 等对照模型比较，RL 版 Qwen3‑VL‑8B 在 Scan 任务显著提升，性能接近旗舰模型。

**⚠️ 局限性**

仅关注可验证错误，未覆盖同行评审的其它维度；数据集偏向可验证错误，隐含弱点可能缺失；实验仅在 Qwen3‑VL‑8B 上，缺乏更大规模模型和多样性验证。

---

## 116. CLIPPER: Replayable Shortlisted Optimization for Repeated Spatial Coverage Planning

**arXiv ID:** 2608.26819 | [PDF](https://arxiv.org/pdf/2608.26819v1)

**作者:** Julian Teusch `[一作]` (Clausthal University of Technology), Monika Sester `[通讯]` (Leibniz University Hannover)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出CLIPPER算法，在共享单车停放规划中使用受限候选池和精确可行性检查，实现城市级规划的秒级可重现执行。

**💡 创新点**

创新点在于将候选池裁剪与完整约束检查相结合，既保持所有硬约束满足，又将规划时间从数十秒降低至秒级，并提供完整的遗漏收益审计。

**🔧 技术方法**

采用子模最优贪婪、候选池裁剪、完整约束检测、离线/在线筛选以及回放记录技术。

**📊 数据集**

使用柏林、慕尼黑和不来梅三座城市的真实共享单车行程记录和OSM街道网络构建的候选点集。

**📈 对比分析**

与全集合贪婪对比，CLIPPER‑F在覆盖率差距≤0.245个百分点的同时，耗时提升7.4–84倍；CLIPPER‑A在覆盖率差距≤1.82个百分点时，仅占原耗时的9–15%。

**⚠️ 局限性**

局限性包括仅评估规划算法行为，未考虑拥堵、再平衡、合规性等现实因素；数据集不完整且不可公开，且需要针对每座城市手动调节候选池宽度K。

---

## 117. Towards Expert Financial QA via Self-Improving RAG

**arXiv ID:** 2608.26706 | [PDF](https://arxiv.org/pdf/2608.26706v1)

**作者:** Junjie Xiong `[一作]` (University of California), Aum Hirpara `[通讯]` (Hofstra University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Self‑Improving RAG，一个由检索、推理和评判三体代理组成的多代理框架，在单轮 RAG 的基础上通过评判反馈实现自我纠错。

**💡 创新点**

创新点在于结合评判驱动的重试策略、固定检索管线与逐步检索/推理升级、并提供完整审计轨迹的闭域金融问答。

**🔧 技术方法**

使用的技术包括 hybrid retrieval（dense+BM25+rerank）、LLM 生成（GPT‑4o‑mini）、LLM Judge 加语义匹配与程序化数值验证，以及 orchestrator 进行阈值衰减。

**📊 数据集**

在 FinanceBench 150 份 SEC 报告的问题上进行评估。

**📈 对比分析**

与单通 RAG 对比，Self‑Improving RAG 在 oracle‑guided 模式下准确率从 53% 提升到 86%，提升 62.3%，并实现 36.4% 的 Lazarus Rate。

**⚠️ 局限性**

主要局限是盲评判可靠性不足导致部署模式仅 31% 的接受率，延迟增加，数值匹配过敏导致误检等。

---

## 118. FaithSieve: Fine-Grained Evaluation of Math Proofs with Faithful Formal Evidence

**arXiv ID:** 2608.26310 | [PDF](https://arxiv.org/pdf/2608.26310v1)

**作者:** Ziyu Wang `[一作]` (Peking University), Zaiwen Wen `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于 Lean 的精细化数学证明错误定位框架 FaithSieve，能够将粗粒度证明步骤拆解为局部推理单元并通过形式化验证检测错误；

**💡 创新点**

创新点在于结合局部推理单元化、语义对齐评分与 Lean 验证，确保形式化证明真正对应自然语言原义，从而显著提升错误定位精度；

**🔧 技术方法**

采用了 Lean 证明助手、自动形式化、语义相似度评分、轻量级检查（SymPy 等）以及自然语言模型生成的树状证明状态；

**📊 数据集**

利用两套专家验证的数据集 ProofLoc‑Olympiad（350道奥赛题）和 ProofLoc‑University（200道大学级题）进行评测；

**📈 对比分析**

与直接判断基线（单步自然语言评估）对比，FaithSieve 在 ProofLoc‑Olympiad 上精确率提升至 81.43%（基线 72.29%），在 ProofLoc‑University 上提升至 84.5%（基线 75%），二者均显示显著性能提升；

**⚠️ 局限性**

主要局限在于自动形式化与 Lean 验证仍耗时且对自然语言的语义捕捉有限，且目前仅覆盖代数/数论、线性/抽象代数等领域，几何等更复杂范畴尚待扩展。

---

## 119. On the Indistinguishability of Human v/s AI Generated Text

**arXiv ID:** 2608.26797 | [PDF](https://arxiv.org/pdf/2608.26797v1)

**作者:** Jaee Ponde `[一作]` (Truth Audit Labs), Debayan Gupta `[通讯]` (Truth Audit Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何利用人类写作样本来战略性地改写机器生成的文本，使其更接近人类文本的分布。

**💡 创新点**

提出了一个显式的收敛速率，并扩展了分析到有限样本设置，表明随着人类样本和改写轮次的增加，机器生成文本向人类分布的移动速度加快。

**🔧 技术方法**

使用了多样本设置和语义保持扰动的理论分析方法。

**📊 数据集**

使用了人类和机器生成的文本样本，具体数据集未详细说明。

**📈 对比分析**

通过与现有的AI文本检测工具进行比较，表明重复改写可以显著降低机器生成文本的可检测性，同时保持内容和质量。

**⚠️ 局限性**

有限样本保证依赖于特定的表示，且未对实际改写工具的块混合条件进行实证验证。

---

## 120. Self-Augmented Diffusion Guidance for Physics-Informed Generation

**arXiv ID:** 2608.26748 | [PDF](https://arxiv.org/pdf/2608.26748v1)

**作者:** Akira Osaka `[一作]` (University of Tokyo), Takehisa Yairi `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于自我增广的分类器无指导物理约束扩散模型，通过学习残差条件并在采样时将残差设为零以提升物理一致性，且将物理模拟与扩散训练/采样分离，避免在每一步梯度求解；

**💡 创新点**

创新点在于利用残差作为条件进行自我增广与分类器无指导引导，显式将残差约束设为零，消除采样过程中的梯度计算和频繁的物理约束评估；

**🔧 技术方法**

核心技术为扩散模型（如DDPM）、分类器无指导引导、残差函数评估以及自我增广（可选噪声增广），并可与PIDM、CoCoGen等方法插件式组合；

**📊 数据集**

实验使用 Darcy 流动数据集（64×64，10000 训练/1000 验证样本）和二维 Navier–Stokes 时序流场数据集（4帧 64×64，10000 训练/1000 验证样本）；

**📈 对比分析**

与标准 DDPM、PG Diffusion、PIDM、CoCoGen 进行比较，实验显示残差平均值显著下降（如 Darcy 流动中从 1.19×10⁻² 降至 1.07×10⁻³），采样速度提升至约 66.8 s/1000 样本，且与其它方法联用可进一步提升物理一致性；

**⚠️ 局限性**

限制在于仍需预先计算残差（需物理模拟或噪声增广），对高维/大规模问题的残差评估开销未完全消除；此外该框架仅适用于能明确评估残差的约束问题。

---

## 121. ClusterAttention: A training-free speedup of bidirectional attention

**arXiv ID:** 2608.26965 | [PDF](https://arxiv.org/pdf/2608.26965v1)

**作者:** Kasper Nordenram `[一作]`, Amelie Dittmann `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出ClusterAttention，一种训练无关的稀疏注意力加速方法，通过递归主成分分割聚类自适应键值空间，生成固定大小块稀疏注意力，在单前向传递中实现显著速度提升同时保持高精度。

**💡 创新点**

创新点包括：①基于键/查询交互的变换构造度量空间；②递归主成分分割得到预设大小聚类；③引入条纹均值补偿（SMC）降低未选聚类误差；④可在多种模型（图像、表格、视频）上无训练调优。

**🔧 技术方法**

技术手段：递归主成分分割（对角化/非对角化）、键/查询变换（R_q、R_k 计算）、均值补偿、基于块稀疏注意力核、top‑k或自适应注意质量阈值选择。

**📊 数据集**

数据集：DINOv2‑L（DIV8K 高分辨率图像），TabPFN‑3（TALENT benchmark 6 大规模表格分类数据集），Wan 2.1‑14B T2V（OpenSora 视频提示）。

**📈 对比分析**

与稠密注意力、SpargeAttn、SVOO、随机聚类、SageAttention2 等基线比较：在 DINOv2 上，20k token 时略慢于 SpargeAttn，但在 60k+ token 时明显领先；在 TabPFN‑3 上，10% 聚类即可保持 ≥99% 准确率，速度提升 2–6×；在视频生成上，ClusterAttention 提升 1.8×（SVOO 1.4×）并保持更高 PSNR/SSIM。

**⚠️ 局限性**

局限：聚类阶段需矩阵特征分解，低 token 数时占主导；SMC 在高稀疏度下仍有开销；固定大小块可能不适合非均匀键/查询分布；未评估训练过程，仅适用于单前向推理；在极大 token 数或 GQA/MQA 结构中尚未验证。

---

## 122. Mapping Woody Vegetation from Multi-Source Imagery and Prediction Fusion for Enhanced Data Efficiency and Accuracy

**arXiv ID:** 2608.26471 | [PDF](https://arxiv.org/pdf/2608.26471v1)

**作者:** Kal Backman `[一作]` (New South Wales Department of Climate Change, Energy, Environment and Water), Adam Roff `[通讯]` (New South Wales Department of Climate Change, Energy, Environment and Water)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一套基于U‑Net的深度学习框架，用于在新南威尔士州生成高度超过2 m的乔木植被覆盖图。

**💡 创新点**

创新点包括：① 通过多源图像的标签转移实现数据效率显著提升；② 采用多时段图像合成与预测融合方法提升模型对云、阴影等图像缺陷的鲁棒性；③ 使用高分辨率1.5 m SPOT 6/7影像实现细粒度树冠分割。

**🔧 技术方法**

技术手段：U‑Net编码器‑解码器网络、卷积残差块、Dice+BCELoss混合损失、Adam优化器、混合精度训练；图像合成（单/双高斯归一化后离散中位数）与多源预测融合（置信度重映射与一致性系数）。

**📊 数据集**

数据集：4年（2019‑2022）SPOT 6/7 1.5 m影像共343个50 km×50 km场景；训练标签由5个场景的512×512补丁手工标注，并通过标签转移扩展到86个场景，最终训练集约1.3 M补丁。

**📈 对比分析**

与单源模型和前沿CNN模型对比，使用加权总体准确率评估：多源训练模型误差降低76.2%，单源预测误差降低38.2%，预测融合误差进一步降低53.6%；在独立点集上，融合模型总体准确率达97.3%，显著优于单源和传统方法。

**⚠️ 局限性**

局限性：需要多源、同一时间尺度的高分辨率影像，计算成本和存储需求较高；标签转移可能带来噪声，尤其在云遮蔽或地表变化明显的年份；仍需人工标注与云掩模处理，限制了完全自动化。

---

## 123. From Sound to Symptom: Real-Time Respiratory Signal Understanding for Conversational Healthcare Agents

**arXiv ID:** 2608.26163 | [PDF](https://arxiv.org/pdf/2608.26163v1)

**作者:** Tanmay Laud `[一作]` (Hippocratic AI), Subhabrata Mukherjee `[通讯]` (Hippocratic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个实时语音通话中检测并细分咳嗽的流式管道。

**💡 创新点**

创新点在于结合多模态大语言模型实现咳嗽/咽喉清除三分类、四类子类型识别、持续时间估计，并加入会话感知门控避免警报疲劳。

**🔧 技术方法**

使用多模态大语言模型（MLLM）、滚动音频缓冲、基于话语边界的提取、并行推理以及对话门控技术。

**📊 数据集**

使用了847段内部通话音频（12.3小时）和AMI会议语料库进行外部验证。

**📈 对比分析**

与BEATs、PANNs、CoughVID等现成检测器对比，咳嗽检测F1 93%，湿/干子类型加权F1 0.75，平均延迟340ms，外部验证宏F1 0.91。

**⚠️ 局限性**

局限在高噪声、低SNR、音频压缩、软咳嗽检测、语种/口音泛化，以及仅支持服务器端部署导致的延迟和成本问题。

---

## 124. Robust Lottery Compression for Metric Voting: A Transfer Principle for Bounded Randomness

**arXiv ID:** 2608.26854 | [PDF](https://arxiv.org/pdf/2608.26854v1)

**作者:** Jianhao Jia `[一作]` (Shanghai University of Finance and Economics), Bo Peng `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在度量空间设定下研究随机投票的“有界随机性”模型，证明在每个投票偏好配置下，只需确定一个固定大小的多重集合，然后在其中均匀随机抽取候选人，即可实现几乎最优的度量失真（即相对最优候选人社会成本的比值）；并给出了针对不同列表大小K的失真上界；进一步证明仅需164个候选人就能突破确定性基准3。

**💡 创新点**

创新点包括：①提出并证明了一个“无维度压缩定理”，表明只要源彩票的期望失真为ρ且支持候选人的确定性失真均不超过H，就能用大小为K的均匀多重集合压缩，失真仅增至ρ+(H+1)O(K^{-1/2})；②对混合集成否决（Mixed Integrated Veto）进行鲁棒化处理（截去早期被淘汰的候选人），从而满足压缩定理的前提；③得到显式的列表大小与失真之间的定量关系，并给出最小列表大小上界164，进一步逼近无约束随机化的最佳失真5/2；④将这些理论结果应用到重复席位的委员会选择问题。

**🔧 技术方法**

主要技术包括：度量失真框架、最大彩票与集成否决的分析、Kolmogorov–Smirnov经验过程上界（Dvoretzky–Kiefer–Wolfowitz–Massart不等式）、Smirnov精确有限样本分布、组合压缩（从源彩票中采样得到均匀多重集合）、鲁棒化（截去早期淘汰候选人）以及偏好配置下的证书式失真判定。

**📊 数据集**

无实验数据，全部为理论证明与分析，未使用任何实测投票数据集。

**📈 对比分析**

相较于传统确定性规则（失真3）和最优随机规则（已知上界5/2），该方法在列表大小K=802时可获得失真≤5/2+3(π/8K)^{1/3}+2√(π/8K)，当K=164时失真<3；对已知的2.75271失真规则可压缩为2.75271+O(1/√K)。因此在理论上实现了对最佳随机失真的几乎完全保留，并在固定有限列表上突破确定性阈值。

**⚠️ 局限性**

局限性包括：①未证明有界随机性能达到无约束随机化的理论最优失真（5/2），仍可能存在更优的无约束规则；②最小列表大小仍未知，仅给出了上界164，是否可以进一步降低至2仍是开放问题；③压缩过程依赖枚举所有K-元多重集合，计算复杂度对大K不现实；④该框架不直接适用于要求候选人不重复的委员会选择；⑤鲁棒化步骤对具体规则的适用性尚需进一步探究。

---

## 125. GameWAM: A World Action Model for Video Games

**arXiv ID:** 2608.26200 | [PDF](https://arxiv.org/pdf/2608.26200v1)

**作者:** Yuncheng Guo `[一作]` (Fudan University), Weijia Li `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出GameWAM，一种联合预测未来视觉观测和可执行键鼠动作的世界-动作模型，用于原生闭环游戏和GUI控制。

**💡 创新点**

创新点包括：①首个针对原生键鼠控制的WAM；②块因果并行Video‑Action DiT与流匹配，支持未来视频监督与动作生成；③块周期规划与层次化历史，分离预测与执行；④按时间点路由实现游戏/GUI动作分离；⑤发现并分析低频动作源印记（LASI）导致的闭环偏差。

**🔧 技术方法**

采用并行Video‑Action Diffusion Transformers（DiT）、块因果注意力、流匹配目标、动作路由网络、块周期控制策略、跨周期层次化视觉记忆、事件锚定采样以及离散余弦变换（DCT）分析。

**📊 数据集**

使用Minecraft Universe（MCU）大型任务集和ViZDoom四图谱进行评测，同时构造了同步的VPT、事件锚定VPT以及脚本化GUI轨迹三类WAM准备数据集。

**📈 对比分析**

与Game‑TARS、Game‑PT、OpenHA等基线比较，GameWAM在MCU上实现最高任务成功率（ASR）且执行动作更少；在ViZDoom上获得竞争或领先的平均奖励；消融实验表明未来视频监督、事件锚定采样和块周期规划是提升性能的关键因素。

**⚠️ 局限性**

局限性包括：①生成策略对动作源低频成分高度敏感（LASI），需要在重规划时重新采样以避免累积偏差；②模型依赖于大规模游戏特定数据，跨域泛化尚待验证；③在真实物理环境或更复杂GUI场景中可能面临适配与安全挑战。

---

## 126. Investigating the Influence of Prompt and Response Languages on LLM Content Generation

**arXiv ID:** 2608.26186 | [PDF](https://arxiv.org/pdf/2608.26186v1)

**作者:** Thi Thanh Nhan Nguyen `[一作]` (Université de Technologie de Compiègne), Thu Nguyen `[通讯]` (HUTECH University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了输入/输出语言对大型语言模型（LLM）生成文本长度、语义保真度和词汇实现的影响，并量化了跨语言“翻译压缩”现象。

**💡 创新点**

创新点在于：①提出了prompt语言与response语言的交互效应框架；②使用了词数、字符数、token数、Cohen's d、LaBSE语义相似度、信息密度以及原始与soft Jaccard关键词重叠三种维度的综合评估；③发现语义保持高但表面词汇通过概念重述被压缩，且压缩效果在模型间显著异质。

**🔧 技术方法**

采用了五个主流LLM（DeepSeek V3、GPT‑4o、Phi‑4‑multimodal、Claude 3.5 Haiku、Gemini 2.5 Pro）和多种计算指标：词/字符/token计数、Cohen's d、LaBSE余弦相似度、信息密度、raw 与 soft Jaccard。

**📊 数据集**

使用了68个非翻译问答，涵盖伦理、文化、健康和社会四个主题；每个问题在四种prompt/response语言组合（ee、en、nn、ne）下生成，构成平衡数据集1,348条响应。

**📈 对比分析**

比较方法：混合效应模型检验prompt×response交互；Cohen's d评估长度差异；LaBSE余弦相似度评估语义一致性；soft Jaccard评估概念重述；信息密度作为长度补充指标。结果显示：ee→en词长缩短约52%，token缩短约25%；语义相似度≈0.83；soft Jaccard≈0.53；不同模型的d值从0.78到4.42不等，表明效果异质。

**⚠️ 局限性**

局限性：①仅测试英语–挪威语对；②受token上限截断影响，尤其是挪威语输出；③信息密度仅为长度近似；④未量化文化/修辞框架；⑤问题来源不统一，未检验其对结果的系统性影响；⑥未评估多轮对话或更大规模数据集。

---

## 127. AudioSpan: Spanning the Duration and Depth of Audio Comprehension

**arXiv ID:** 2608.26431 | [PDF](https://arxiv.org/pdf/2608.26431v1)

**作者:** Wen Huang `[一作]` (Alibaba Group), Jin Xu `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为 AudioSpan 的长时音频理解基准，覆盖 10 分钟至 2 小时以上时长，涵盖 3,240 个问答，按照感知、理解、推理三层认知层级进行设计，并提供多选与开放式答案两种格式；

**💡 创新点**

创新点包括：①将时长与认知深度同时提升，填补了现有短时音频评测的空白；②设计两条互补问答路径——Native QA 直接从音频内容生成问答，Anchor QA 通过植入声学锚点并构建链式问题来验证感知到推理的全链；③实现全自动化的问答生成流水线，结合结构化字幕、LLM 生成、对抗式批评者反馈和多级质控；

**🔧 技术方法**

主要技术：大规模音频-语言模型 (LALMs)、语音转写 (ASR)、语义分割、音频-文本多模态字幕生成、LLM 驱动的问答与批评者循环、基于权重的评分 Rubric、链式评分 Chain；

**📊 数据集**

数据集：720 条原始与锚点修改后的野生音频（English/Chinese），时长分层为 S（10–30 min）、M（30–60 min）、L（60 min+），涵盖 7 种类型与 9 主题，共 3,240 个问答；

**📈 对比分析**

对比方法：在 12 种 LALM（7 开源、5 封闭）与三种文本基线（question‑only、transcript、caption）上评测，采用多选 Accuracy、开放式 Rubric 以及 Anchor 链式 Chain 分数；结果显示：音频模型在所有模式下均落后于文本基线，尤其在感知层级表现最差，随着时长增大性能进一步下降；

**⚠️ 局限性**

局限性：①音频模型难以从冗长波形中提取关键信息，感知层级瓶颈明显；②基准仍依赖 LLM 生成的结构化字幕，可能遗漏真实音频细节；③Anchor QA 的锚点植入方式人工设定，可能限制真实场景的多样性；④闭源模型与开源模型之间存在显著性能差距，进一步提升开源模型能力仍是挑战。

---

## 128. EduRiskX: A Neuro-Symbolic Framework with F-Logic Reasoning for Early Academic Risk Prediction

**arXiv ID:** 2608.26107 | [PDF](https://arxiv.org/pdf/2608.26107v1)

**作者:** Yu Fu `[一作]` (Sichuan University), Rongfang Bie `[通讯]` (Beijing Normal Univeristy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 EduRiskX neuro‑symbolic 框架，结合优化的 Temporal Transformer 与基于 F‑Logic 的符号推理，实现了在线高等教育环境下的早期学术风险预测与可解释输出。

**💡 创新点**

创新点包括：1) Temporal Transformer 的三项优化（时序注意力、类别加权损失、动态周截断）；2) 基于教学理论（Engagement Theory、Student Integration Model 等）的自动化规则挖掘与 F‑Logic 规则库；3) 可学习的逻辑回归融合机制，动态平衡神经与符号证据；4) 结构化、可视化的解释与干预建议，构成现代专家系统。

**🔧 技术方法**

采用 Transformer（多头注意力）、时序注意力机制、类别加权交叉熵、F‑Logic 规则引擎、规则挖掘与冗余剔除、逻辑回归融合、校准与评估（PR‑AUC、AUC‑ROC 等）。

**📊 数据集**

使用 Open University Learning Analytics Dataset（OULAD）——38 周在线学习日志数据，按学生划分 80/10/10 的训练/验证/测试集。

**📈 对比分析**

与 LSTM、CNN‑1D、PatchTST、iTransformer、基础 Transformer 以及 EduRiskX 的神经‑仅版进行对比，采用 Accuracy、Recall、F1‑Score、AUC‑ROC、PR‑AUC、Balanced Accuracy 等指标。EduRiskX 在早期（Week 5‑15）以及整体上均获得最高召回、F1 与 PR‑AUC，平均检测周为 9.32 周，明显早于 PatchTST（15.70 周）和 iTransformer（13.92 周）等基线。

**⚠️ 局限性**

局限性：规则库仅在训练集上挖掘，缺乏在线自适应更新；对不同学科、文化或新颖数据集的泛化能力待验证；模型仍需人工校准理论映射；解释与干预建议虽可视化，但实际干预效果未在真实教育环境中评估。

---

## 129. KubeCap: A Framework for Capability Minimization in Kubernetes via Static Analysis and LLM-Assisted Rule Inference

**arXiv ID:** 2608.26699 | [PDF](https://arxiv.org/pdf/2608.26699v1)

**作者:** Yuhao Liu `[一作]` (Nankai University), Zheli Liu `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 Kubernetes 容器的 Linux Capability 过度授予问题，提出并实现了 KubeCap 框架，自动化推断每个工作负载的最小 Capability 集合，并生成修复后的 YAML manifest。

**💡 创新点**

创新点在于：① 将 Helm/Kustomize 渲染、入口点定位、基于可达性的系统调用分析与 LLM 辅助的内核规则推断四个模块结合，构建了完整的静态分析链；② 采用 LLM 进行一次性学习，自动提取 syscall-parameter-capability 规则；③ 通过差异分析精确定位冗余 Capability，并自动生成修补清单，真正实现了“最小权限”级别的配置修复。

**🔧 技术方法**

核心技术包括：Kubernetes 模板渲染与 Manifest 翻译；容器入口点定位与源代码映射；基于 SSA 的可达性导向系统调用分析；使用 GPT‑4o‑mini 的 LLM 进行规则推断；约束求解与条件规则匹配；差异分析与自动生成修补 YAML。

**📊 数据集**

使用三组公开的 Kubernetes 配置数据集（Rahman、Shamim 及作者自建 102 个仓库）进行经验调查；评估基于 10 个 Go 项目的 Kubernetes 工作负载（包括 CNCF 项目与 Helm 项目）。

**📈 对比分析**

与 RTA、CHA、PTA 四种常见的调用图构造方法对比。KubeCap 的可达性分析在所有项目中平均实现 54.97% 的 Capability 减少率，明显优于 RTA（38.00%）和 CHA（7.68%），并在大多数项目上完成时间仅为几分钟，内存占用约 1–3.5 GiB，显示出良好的可扩展性。

**⚠️ 局限性**

局限性包括：目前仅支持 Go 项目，无法直接应用于 C/C++ 或其他语言；动态特性、反射、shell 启动逻辑等仍可能导致误判；LLM 生成的规则可能存在幻觉，需人工验证；仅关注 Linux Capability，未覆盖其它权限模型（如 seccomp、RBAC）。

---

## 130. Decolonial Discourse in Postcolonial Contexts: How YouTubers Negotiate Audience Tensions, Platform Governance, and State Influence

**arXiv ID:** 2608.26351 | [PDF](https://arxiv.org/pdf/2608.26351v1)

**作者:** Dipto Das `[一作]` (Cornell University), Bryan Semaan `[通讯]` (University of Colorado Boulder)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究探讨孟加拉族裔YouTubers在后殖民语境中如何在受众、平台治理与国家监管交织的技术社会环境中维系去殖民话语。

**💡 创新点**

创新点在于将去殖民话语视为持续的社会技术工作，阐释受众多元公共、平台经济与监管不透明如何递归地塑造话语的可持续性，并提出“selective sustainability”（选择性可持续性）概念。

**🔧 技术方法**

研究方法主要采用定性访谈与扎根理论编码，并未使用机器学习或大数据技术。

**📊 数据集**

数据集由15名来自孟加拉、印度、巴基斯坦的YouTube创作者的访谈记录组成。

**📈 对比分析**

方法上采用归纳主题分析，对比并无量化指标或性能评估。

**⚠️ 局限性**

研究限制包括样本规模有限、性别与宗教代表性不足、缺少算法层面与更广泛社会经济背景的分析，以及对老年受访者体验的忽略。

---

## 131. Beyond Accuracy: A Qualitative Analysis of Vision-Language Models for Hate Speech Detection in Memes

**arXiv ID:** 2608.26143 | [PDF](https://arxiv.org/pdf/2608.26143v1)

**作者:** Muhammad Jawad Chowdhury `[一作]`, Sabbir Ahmed `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

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

## 132. From SQL to Knowledge Graphs: An LLM-Driven Multi-Agent Approach with Data Schema Improvement

**arXiv ID:** 2608.26117 | [PDF](https://arxiv.org/pdf/2608.26117v1)

**作者:** Dinh-Khanh Pham `[一作]`, Truong-Son Hy `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将传统关系型SQL数据库自动转换为知识图谱数据库，并通过多智能体LLM系统完成ETL、模式设计与查询回答。

**💡 创新点**

提出Circle Discussion多智能体设计模式以及Meta-Error与Few-Shot Learning结合的LLM驱动知识图谱生成与查询系统。

**🔧 技术方法**

使用LLM-Agent、Graph Agent、Analyzer Agent、ETL Agent等多智能体以及文本到Cypher/SQL代码生成与评估技术。

**📊 数据集**

构造了BFSI金融业务查询数据集（1081条）及对应的SQL数据库进行实验。

**📈 对比分析**

通过对比CypherAgent和SQLAgent在不同难度查询下的准确率、延迟和token数，CypherAgent平均准确率提高12.12%，延迟缩短约3倍，token减少约2倍。

**⚠️ 局限性**

实验仅覆盖金融领域，缺乏跨行业验证，且多智能体系统对资源和部署复杂性有一定依赖。

---

## 133. Multi-Image Visual Token Pruning in Large Visual Language Models

**arXiv ID:** 2608.26806 | [PDF](https://arxiv.org/pdf/2608.26806v1)

**作者:** Rongyang Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型视觉语言模型（LVLM）提出一种无训练、可自适应的视觉令牌修剪框架（AVTP），能够在多图像场景下显著降低计算与内存开销。

**💡 创新点**

创新点包括①基于隐藏状态变化的令牌重要性评估，②通过视觉注意力分布实现模型架构感知的修剪层选择，③根据图像重要性动态分配修剪比例的图像感知自适应修剪策略。

**🔧 技术方法**

技术方法包括隐藏状态差分重要性评分、统计不同层的视觉注意力分布进行动态层选择、按图像重要性调整保留比例，以及在 FlashAttention2 上的高效实现。

**📊 数据集**

主要数据集为五个多图像基准：MuirBench、MIRB、BLINK、Qbench2、NLVR2；同时用于层选择的样本来自MMBench、POPE、ScienceQA和MMStar。

**📈 对比分析**

与 FAST‑V、VisionZip、DivPrune、V2DROP 以及 FlashAttention2 基线进行对比，结果显示 AVTP 在 Qwen3VL‑8B 上实现约 2× 推理加速且保持 96.1% 以上准确率，在 InternVL3.5‑8B 上保持 94.1% 以上准确率，且在 LLaVA‑OV‑7B 上甚至超越原基线；整体实现了准确率-延迟的最优折中。

**⚠️ 局限性**

局限性包括：关键超参数（如 top‑k、α 等）缺乏理论基础，实验仅与其它令牌修剪方法对比，未与量化或蒸馏等其它推理加速技术联合评估；方法为迭代改进而非全新算法范式。

---

## 134. Affix Cache for Diffusion Large Language Models

**arXiv ID:** 2608.26140 | [PDF](https://arxiv.org/pdf/2608.26140v1)

**作者:** Kaihua Liang `[一作]` (King Abdullah University of Science and Technology), Marco Canini `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对DLLM的Affix Cache（ACache），实现跨请求共享文本片段的 KV 缓存重用，并通过选取Anchor Tokens进行局部重计算，以保持上下文一致性。

**💡 创新点**

创新点在于把共享上下文扩展到前缀、插入和后缀等 affix 位置，利用注意力重要性挑选 Anchor Tokens，仅重算少量关键 token，兼顾准确性与效率。

**🔧 技术方法**

技术包括基于 Fast‑dLLM 的缓存框架、Masked‑to‑Affix 注意力重要性评估、Anchor Token 选择、共享与私有 KV 区域映射，以及在 Nano‑vLLM 原型上的实现。

**📊 数据集**

使用 GSM8K、MBPP、BABILong 等常见 LLM 基准数据集进行评估。

**📈 对比分析**

与全重算和原始 Fast‑dLLM 基线比较，ACache 在重算延迟上可降低 15%–55.7%，吞吐量提升至 1.68×，且在共享 affix 下约 20% Anchor 复算即可恢复大部分准确性。

**⚠️ 局限性**

局限在于当前实现仅支持共享前缀；对 infix/suffix 的支持需进一步实现位置信息映射；假设共享 span 预先声明，缺乏在线发现机制。

---

## 135. GROUND: Reducing Hallucinations in LLM-Based Enterprise Analytics Through Governed Semantic Definitions

**arXiv ID:** 2608.26157 | [PDF](https://arxiv.org/pdf/2608.26157v1)

**作者:** Aravind Sasidharan Pillai `[一作]` `[通讯]` (Cox Automotive Inc), Aravind Sasidharan Pillai (Cox Automotive Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 GROUND 框架，将大型语言模型在企业数据仓库的自然语言分析任务中通过语义层和验证循环进行治理，确保生成的 SQL 语义、度量、连接、粒度、过滤、行级安全和成本满足预定义规则；

**💡 创新点**

创新点在于：① 将企业业务语义层作为“语义检索”与“验证执行”双重约束；② 引入面向治理的失效分类（模式、度量、连接、粒度、过滤、行级安全、成本）并单独评估；③ 证明仅靠语义上下文不足以保证行级安全，必须通过显式验证；

**🔧 技术方法**

使用了大型语言模型（Claude Opus 4.8、Sonnet 5、OpenAI GPT‑5.2、Llama‑3.3‑70B），结合检索增强生成（RAG）、语义层检索、结构化 JSON 输出、以及验证‑重试循环；

**📊 数据集**

使用合成汽车零售数据仓库（100 题）和真实美国 NHTSA 车辆安全投诉数据（40 题）做基准，所有数据和语义层均公开；

**📈 对比分析**

与三种基线（直接 schema‑only、schema‑RAG、semantic‑only）在同一模型下比较，GROUND 在所有六类失效均为零，执行率 100%，结果值正确率 95.1%，相比之下基线行级安全违规率 35–78%，过滤错误率 53–78%；在 NHTSA 上也保持零失效；

**⚠️ 局限性**

局限在于：① 仍依赖模型判断的非强制性功能（如未知度量的拒绝、澄清），存在偶发错误；② 成本与延迟显著提高（约 5 倍 token、1.3 倍时延），需缓存优化；③ 评估聚焦结构化报表，未覆盖非结构化或因果推理任务；

---

## 136. Real-time Unsupervised Object Discovery from Asynchronous Event Streams

**arXiv ID:** 2608.26644 | [PDF](https://arxiv.org/pdf/2608.26644v1)

**作者:** Pratham G. Shenwai `[一作]` (University of New South Wales), Sridhar Ravi `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无监督、训练无关的事件相机物体发现框架，包括事件去噪与聚类两大模块。

**💡 创新点**

创新点在于：①线性时间的概率事件过滤器 SPEF，动态自适应接受阈值；②基于 Morton 码的聚类 EMCC，利用排序后间隙检测实现 O(N log N) 的聚类，无需邻域搜索。

**🔧 技术方法**

技术包括：事件空间网格化、概率计数器、时间相关性指数、Morton（Z-order）码编码、gap 阈值化、密度筛选与 NMS 合并。

**📊 数据集**

使用的公开数据集包括 E‑MLB（去噪评测）、FRED 与 eTraM（物体发现评测）。

**📈 对比分析**

与传统密度聚类（DBSCAN、HDBSCAN、MeanShift）及学习型去噪器进行对比；SPEF 在 E‑MLB 上多达 8‑级噪声下达到 MESR 最高分；EMCC 在 FRED、eTraM 上取得最高 F1（0.441/0.273）、IoU（0.647/0.656），并将聚类时延从 27 ms 降至 6 ms，内存与计算成本显著降低。

**⚠️ 局限性**

局限性包括：对输入信噪比高度依赖，无法保持跨实例的物体身份，且在极低光或极高噪声环境下可能仍需手动调参；未实现在线跟踪与跨帧语义一致性。

---

## 137. SILK: Closing the Time-of-Check-to-Time-of-Use Gap in RoT-Protected AI Systems

**arXiv ID:** 2608.26402 | [PDF](https://arxiv.org/pdf/2608.26402v1)

**作者:** Ruichen Qi `[一作]` (Brown University), Mehdi Saligane `[通讯]` (Brown University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

SILK通过在权重量化的最低有效位嵌入密钥化的链式完整性校验，在线验证权重流，在RoT保护下闭合时间检查到使用的完整性缺口。

**💡 创新点**

创新点在于将完整性位直接嵌入权重量化位并构造跨字节的链式校验，无需额外标签，且在最终预计算边界实现对完整权重量化流的逐字节验证。

**🔧 技术方法**

采用安全伪随机函数Ascon-PRF、密钥化的LSB校验、软硬件协同实现的流式校验器，以及可调的链式依赖窗口。

**📊 数据集**

评估使用了CNN模型（ResNet-50、MobileNet等）和LLM模型（TinyLlama、OPT、Qwen2.5、Phi-3.5-mini）以及对应的ImageNet、WikiText-2等数据集。

**📈 对比分析**

与现有内存/模型级完整性方案对比，SILK在INT8下保持≤0.76pp的精度损失，INT4/ MXFP4可调的安全-质量折中，硬件实现实现756 MB/s吞吐量，面积仅为Caliptra RoT的1.00%。

**⚠️ 局限性**

局限在于需在RoT内配置密钥、仅对已认证模型的权重流提供保护，对模型自身的篡改无法检测，并且低精度模型对LSB嵌入更敏感。

---

## 138. Dynamic Haven Selection for Multi-Agent Pickup and Delivery in Constrained Warehouses

**arXiv ID:** 2608.26939 | [PDF](https://arxiv.org/pdf/2608.26939v1)

**作者:** Taisei Hirayama `[一作]` (Hokkaido University), Itsuki Noda `[通讯]` (Hokkaido University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在空间受限的仓库中提出一种动态Haven（安全等待点）选择方法A‑SHARP，用于在Multi‑Agent Pickup and Delivery (MAPD) 任务中动态改变机器人退避目标，保持安全和完成性；

**💡 创新点**

通过引入可用性检查和待释放规则实现Haven所有权的原子转移，解决了在已有预留路径和持久性等待点下动态切换退避目标所面临的冲突与死锁问题；

**🔧 技术方法**

核心技术包括基于Safe Interval Path Planning (SIPP) 的路径规划、空间‑时间预留表、可用性判定与待释放规则的所有权传递协议；

**📊 数据集**

在四种典型仓库布局（well‑formed、narrow‑bi、narrow‑bi‑dead、tree）上生成的任务序列（共 2,400 组），通过 100 个随机种子进行实验；

**📈 对比分析**

与固定‑Haven 基线、TP、PIBT、PIBTTP‑TA 等方法进行对比，A‑SHARP 在所有配置下 100% 成功率；在树形布局中使平均完成时间降低 16.7%，整体在 138 种 H‑surplus 配置中在 107 例中显著优于基线；

**⚠️ 局限性**

局限性包括：假设离散确定性执行、中央预留表、Haven 数量不小于机器人数量、仅采用最近-任务/最近‑可用‑Haven 启发式，未考虑动态障碍、执行延迟或共享停车等实际环境挑战。

---

## 139. Data-driven Koopman mode approximation: A neural power iteration algorithm

**arXiv ID:** 2608.26943 | [PDF](https://arxiv.org/pdf/2608.26943v1)

**作者:** Guillaume O. Berger `[一作]` (UCLouvain), Raphaël M. Jungers `[通讯]` (UCLouvain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新颖的数据驱动算法，通过神经网络近似非线性动力系统的Koopman算子的主特征函数（模式）。

**💡 创新点**

创新点在于该方法利用幂迭代方案直接学习主Koopman模式，而无需显式构建Koopman算子的投影，从而避免了维数诅咒的问题。

**🔧 技术方法**

使用了神经网络和幂迭代算法。

**📊 数据集**

使用了采样状态转移的数据集。

**📈 对比分析**

与传统的扩展动态模式分解（EDMD）方法相比，该方法在准确性上表现更好，尤其是在需要表达能力强的表示时，尽管计算时间较长。

**⚠️ 局限性**

该方法的局限性包括可能的谱污染、对所选近似空间的依赖，以及在有限样本情况下缺乏误差界限。

---

## 140. From Reasoning to Pixels: Grounded Medical Multimodal LLMs for VQA and Segmentation

**arXiv ID:** 2608.26856 | [PDF](https://arxiv.org/pdf/2608.26856v1)

**作者:** Haowen Gu `[一作]` (Nanjing University of Science and Technology), Yazhou Yao `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了MedREAL框架，将医学视觉问答与像素级分割统一在一次性推理过程中，实现了诊断推理与空间定位的闭环；

**💡 创新点**

创新点在于使用显式证据标记提取推理特征，设计SARP模块聚焦推理相关语义，R2V融合将推理向视觉空间映射，并构建了全新MedRAVS-13K数据集；

**🔧 技术方法**

采用多模态大语言模型（如Qwen3-VL）进行文本生成，SAM分割模型配合SARP与R2V机制，训练时使用交叉熵、BCE和Dice损失，配合DeepSpeed与AdamW优化；

**📊 数据集**

利用MedRAVS-13K（13,824张样本），该数据集来源于BUSI、COVID-QU-Ex、ISIC-2018、Kvasir-SEG四种医学影像；

**📈 对比分析**

与参考式、代理式和推理式基线对比，MedREAL在gIoU和cIoU上分别达到68.49%/70.47%，显著优于LISA（55.70%）及其它基线，同时在文本生成（BLEU/ROUGE）上也取得领先；

**⚠️ 局限性**

局限性在于仅支持二维影像，推理过程自回归导致推理速度慢，且缺乏对三维体数据的处理与推理加速技术。

---

## 141. Reclaiming Epistemic Agency: A Critical Framework for Human-Generative AI Co-Agency in Education

**arXiv ID:** 2608.26937 | [PDF](https://arxiv.org/pdf/2608.26937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 142. Decoding Algorithms for MDS Array Codes

**arXiv ID:** 2608.26483 | [PDF](https://arxiv.org/pdf/2608.26483v1)

**作者:** Sara D. Cardell `[一作]` (Unesp), Cintya Wink de Oliveira Benedito `[通讯]` (Unesp)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一系列基于超正规矩阵与非奇异矩阵克罗内克积构造的MDS数组码的解码算法。

**💡 创新点**

创新点在于利用这些码的奇偶校验矩阵的特殊块结构，给出了可恢复至n‑k个擦除、以及在q‑符号对称信道上对1、2个符号误差的定位与纠正方法，并针对Vandermonde矩阵的情形进一步简化了解码步骤。

**🔧 技术方法**

主要技术包括线性代数工具（超正规矩阵、块超正规矩阵、Kronecker积）、奇偶校验矩阵分块求逆以及对小规模误差的候选位置检验。

**📊 数据集**

实验数据集中使用了多种有限域（𝔽_3、𝔽_7、𝔽_13、𝔽_8）下的具体矩阵实例，对给定码字的误码/擦除场景进行了手工演示。

**📈 对比分析**

方法以理论分析为主，并未给出复杂度或误码率的量化比较；但通过示例证明算法能够达到MDS码的距离极限（即最多纠正n‑k个擦除、⌊(n‑k)/2⌋个符号误差）。

**⚠️ 局限性**

局限性包括：只针对1、2个符号误差给出闭式算法，较大误差模式需更复杂的代数求解；对块大小b的增大会导致矩阵求逆计算量显著增加；以及需要已知A、B矩阵的具体形式，限制了在通用编码场景中的直接应用。

---

## 143. G2D: Generative-to-Discriminative Collaborative Inference for Zero-Shot Image Classification

**arXiv ID:** 2608.26744 | [PDF](https://arxiv.org/pdf/2608.26744v1)

**作者:** Zehua Hao `[一作]` (Xidian University), Puhua Chen `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的 G2D 框架，通过将生成式 VLM 用作 CLIP 提供的候选集的图像检验器，实现零样本图像分类。

**💡 创新点**

创新点包括：①将生成式模型转化为结构化检验器以解决 CLIP 的误排序；②固定置信度路由与熵自适应候选数；③将 CLIP 概率注入生成式提示；④使用 trie 限制解码，消除生成漂移。

**🔧 技术方法**

使用 CLIP 作为检索器、Qwen3‑VL‑8B（等生成式 VLM）作为验证器；实现固定阈值路由、熵自适应候选数、概率注入、Trie 限制解码等技术。

**📊 数据集**

在八个公开基准上评估：EuroSAT、DTD、Oxford‑Pets、CUB200、Food‑101、ImageNet、ImageNetV2、Places365。

**📈 对比分析**

与 CLIP、纯生成式 Qwen‑only 及多种 prompt‑enriched CLIP 基线对比，平均 top‑1 准确率达到 68.85%，显著高于 CLIP 59.35% 及 Qwen‑only 63.11%，在大多数数据集上均优于单一模型。

**⚠️ 局限性**

局限性：需要同时加载 CLIP 与生成式 VLM，检验路由比例高时推理时延明显；依赖闭合词表且对 CLIP 候选集质量敏感；当 CLIP 误判且目标类不在 top‑K 内时检验仍可能失败。

---

## 144. Training-Time Explainability for Multilingual Hate Speech Detection: Aligning Model Reasoning with Human Rationales

**arXiv ID:** 2608.26125 | [PDF](https://arxiv.org/pdf/2608.26125v1)

**作者:** Muhammad Deedahwar Mazhar Qureshi `[一作]` (Technological University Dublin), Wael Rashwan `[通讯]` (Maynooth University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语言（英语和 Hinglish）语料上构建了一个训练时可解释性框架，通过将模型解释与人类标注的理由对齐，提升了对穆斯林社区隐性仇恨的检测性能和可解释性。

**💡 创新点**

提出了训练时的解释对齐正则化，将解释质量直接嵌入模型训练流程，首次实现跨语言、文化编码仇恨检测的可解释性提升。

**🔧 技术方法**

采用 RoBERTa / XLM‑RoBERTa 编码器，结合 LIME、Integrated Gradients、Grad×Input 和 Attention 四种解释方法作为正则化信号。

**📊 数据集**

使用 HateXplain（约 20k 条英语推文）和 BullySent（约 6.5k 条 Hinglish 语料）两大数据集。

**📈 对比分析**

与无正则化基线相比，注意力正则化在两数据集上均提升了 F1（HateXplain 0.84，BullySent 0.85），并在可解释性（IoU、Token‑F1）与信念降落（faithfulness）指标上取得最优或接近最优的平衡。

**⚠️ 局限性**

模型仍受限于人类标注理由的稀缺与主观性，且对极低频或全新文化符号的泛化仍有限。

---

## 145. Predicting Consequences and Reinforcing Navigation Policies with Latent World Models

**arXiv ID:** 2608.26190 | [PDF](https://arxiv.org/pdf/2608.26190v1)

**作者:** Zengmao Wang `[一作]` (University of Chinese Academy of Sciences), Shuhan Shen `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种兼容性预测潜在世界模型（LWM），通过在潜在空间预测动作导致的特征兼容度来实现离线轨迹的反事实推理，从而无需像素级重建即可用于规划与策略学习。

**💡 创新点**

①将观察映射至潜在特征并预测动作序列的潜在后继；②使用跨轨迹动作序列进行反事实训练并用空间距离的对数映射标记兼容度；③在世界模型内部实现策略监督与强化学习，无需真实交互。

**🔧 技术方法**

视觉Transformer编码器、因果Transformer解码器、对数空间兼容度损失、GRPO式强化学习、离线视频伪标签生成。

**📊 数据集**

RECON、SCAND、作者自建LWM导航数据集（校园、住宅区、公共园区共约6小时、60万帧）。

**📈 对比分析**

与基线NWM、Dino-WM、NoMaD、BC等方法对比，LWM在预测误差(PE/OE)、动作选择准确率(ACC3/ACC5)、策略成功率(SR)与LPIPS指标上均取得显著提升，尤其在真实环境与未见室内场景下表现最优。

**⚠️ 局限性**

对抗损失和仅使用单一路径训练效果不佳；模型对动作序列长度和数量仍敏感；对真实世界动态障碍物的适应性尚未完全验证。

---

## 146. Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search

**arXiv ID:** 2608.26152 | [PDF](https://arxiv.org/pdf/2608.26152v1)

**作者:** Eric Lacosse `[一作]` (Champalimaud Research, Centre for Unknown), Daniel C. McNamee `[通讯]` (Champalimaud Research, Centre for Unknown)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了大型语言模型在语义流畅性任务（Semantic Fluency Task）中对人类记忆搜索的认知对齐与协同效果，探讨其是否能超越人类预测并减少协同抑制。

**💡 创新点**

首次量化LLM在宏观（TPM、BLEU）和微观（下一个词预测、切换预测）层面对人类记忆轨迹的精准预测，并证明其可超越人类在预测精度上的表现，同时揭示其在协同抑制情境下未能显著提升产出。

**🔧 技术方法**

采用Gemini 3系列（Lite/Flash/Pro）和Llama‑3.3‑70B等前沿LLM，并结合“认知驱动提示”（Theory‑Driven Cognitive Prompting）来模拟人类记忆检索；评估指标包括BLEU、Perplexity、Jaccard、Spearman相关等。

**📊 数据集**

使用公开的SFT数据集，并收集约190条单人序列以及人‑人、人‑AI、AI‑AI三种交互条件下的多轮对话数据，涵盖动物、服装等类别。

**📈 对比分析**

与人类和AI对照实验显示：LLM在宏观层面TPM相关性ρ≈0.70、BLEU最高达0.63，单词预测准确率比人类高≈5.9个百分点；但在双人对话中未出现概念产出提升，仍呈现协同抑制，说明预测精度并未转化为实际协同收益。

**⚠️ 局限性**

主要局限包括：固定交替式交互机制导致无法充分利用LLM主动干预潜能；模型延迟与硬件限制；预测未转化为控制策略，无法突破协同抑制；缺乏对LLM内部记忆检索机制的可解释性和进一步的机制性验证。

---

## 147. Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory

**arXiv ID:** 2608.26807 | [PDF](https://arxiv.org/pdf/2608.26807v1)

**作者:** Zihao Cheng `[一作]` (Beihang University), Yunhong Wang `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了行为感知旅行规划任务，利用用户过去的行为轨迹推断偏好并直接生成个性化旅行计划。

**💡 创新点**

构建了大规模中文旅行行为数据集Behavior2Trip（11,400条实例，约40条行为/实例），并提出基于强化学习的B2T-Agent，支持外部工具调用与内部记忆管理，利用多组件奖励和GRPO训练，显著提升了对隐式偏好的捕捉和约束满足。

**🔧 技术方法**

采用大型语言模型（Qwen3-8/14/32B、Deepseek-V3、GPT-4.1/4o）与强化学习框架（RLFactory、GRPO）、结构化动作空间、工具集与内部记忆模块、分阶段奖励函数和损失屏蔽技术。

**📊 数据集**

Behavior2Trip：基于中国大型在线旅行平台的真实用户行为，包含14个属性、5个偏好维度，分为易、中、难三难度级别。

**📈 对比分析**

在Behavior2Trip上，GPT‑4.1在最难任务的完整约束通过率仅0.5%；B2T‑Agent（Qwen3‑8B/14B）在Easy/Medium/Hard均领先所有基线，Hard集完成约束通过率达5.0%，比GPT‑4.1提升约10倍，LLM通过率亦大幅上升。B2T‑Agent同样在TravelPlanner基准上实现显著提升。

**⚠️ 局限性**

训练时需要采样多轮回放，计算成本高；推理时多轮工具调用导致首词延迟显著。

---

## 148. Extending Low Latency Service Across the Internet

**arXiv ID:** 2608.26601 | [PDF](https://arxiv.org/pdf/2608.26601v1)

**作者:** Harkirat Singh `[一作]` (Stony Brook University), Shivendra Panwar `[通讯]` (NYU Tandon)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种可在多域网络中部署 L4S 的方案，减少对核心路由器升级的需求。

**💡 创新点**

将 BGP 社区、SRv6 路径导向以及优先级队列结合，实现跨域低延迟隔离。

**🔧 技术方法**

BGP 社区标记、SRv6 封装与解封装、DSCP 优先级队列、CoDel/PIE AQM、TCP Prague 与 CUBIC。

**📊 数据集**

使用 FABRIC 测试平台的可编程网络拓扑与 MGEN 等仿真负载。

**📈 对比分析**

通过多种单瓶颈与多瓶颈实验对比，发现启用机制后 L4S 视频流无重缓冲、呼叫延迟 <40 ms，经典流吞吐基本不受影响；未启用时性能下降明显。

**⚠️ 局限性**

方案依赖路径多样性与 BGP 社区准确传播，且对核心路由器的升级需求未完全消除，实测仅在实验环境中验证，未覆盖故障恢复、异构 RTT 等生产复杂场景。

---

## 149. When Is the Sharp Covariance Envelope Tight? Feature-Only Geometry for Volume-Sampled Least Squares

**arXiv ID:** 2608.26877 | [PDF](https://arxiv.org/pdf/2608.26877v1)

**作者:** Kihun Rhee `[一作]` `[通讯]` (Seoul National University), Kihun Rhee (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在固定特征池与固定响应下，使用普通未缩放的固定大小体积采样并配合最小二乘拟合，推导了条件中心化系数协方差的全尺寸预算Loewner envelope，并通过特征几何引入残差增强机制得到响应感知的协方差上界，最终给出可预先验证的预响应证书以决定子集大小。

**💡 创新点**

（1）首次给出任意固定响应下的全尺寸预算Loewner envelope并证明其全局尖锐性；（2）提出残差增强机制，得到响应感知的协方差上界；（3）基于特征几何定义的margin ν_A，用来判定是否可达到全局上界并分为严格和紧致两种phase；（4）给出可计算的预响应证书，验证其在冻结特征池上的实用性。

**🔧 技术方法**

体积采样、最小二乘估计、Loewner顺序比较、矩阵Jensen不等式、残差增强（加上正交化）、Parseval帧与Naimark补几何、特征几何分析、特征只证书（coherence、其他下界）等。

**📊 数据集**

使用冻结公共特征池（72个冻结编码器单元），在Fashion‑MNIST的新行与冻结编码器一起评估，实验中检验了证书在不同预算下的授权与回退情况。

**📈 对比分析**

与已有的体积采样、随机设计、主动采样等方法对比，表格列出了采样法、估计器、预算目标等。实验显示证书在72个单元中分别产生了20/4、4/20、5/19的动作/回退计数；在所有动作单元中后续风险差为正，表明授权的子集规模具有一定优势；但未给出总体泛化性能或选择器优势的数值对比。

**⚠️ 局限性**

仅针对条件中心化的系数协方差；要求正全拟合损失、严格内部预算、无coloop；残差增强方法需行一般位置；未证明计算ν_A的复杂度；未讨论总体泛化、选择器优势或采样器的计算复杂性。

---

## 150. Knowledge Cards: Structured Knowledge for AI Systems

**arXiv ID:** 2608.26176 | [PDF](https://arxiv.org/pdf/2608.26176v1)

**作者:** Liliana Ferreira `[一作]` `[通讯]` (Mondegreen.ai), Liliana Ferreira (Mondegreen.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Knowledge Card 架构，定义了一种结构化知识卡片，用于捕捉单一概念的已验证知识，供专家审核、组织审计以及 AI 系统推理。

**💡 创新点**

通过五个核心属性（ontology-grounded、provenance-linked、boundary-explicit、expert-validated、versioned）填补了 AI 文档从输入到输出的知识层缺口，并提供统一的知识表示，兼容 LLM 与符号推理。

**🔧 技术方法**

基于正式领域本体、JSON-LD/JSON Schema 定义卡片结构；支持多层推理（叙述、模式、规则、概率），可与 LLM、符号推理器、图数据库（如 Neo4j）等技术交互。

**📊 数据集**

示例使用了能源（风电场齿轮箱故障诊断）和制药（合规检查）领域的原型卡片，未公开使用大型数据集。

**📈 对比分析**

论文未进行实验对比，主要通过案例演示说明 Knowledge Card 在代理记忆、工业诊断、合规检查中的适用性，缺乏定量性能评估。

**⚠️ 局限性**

仍处于草案阶段，缺乏完整的工具链支持；在大规模知识库构建、自动化生成与验证、跨域本体兼容性等方面仍需进一步研究。

---

## 151. PLCBench: Can Autonomous LLM Agents Turn PLC Access into Sustained Physical Impact?

**arXiv ID:** 2608.26882 | [PDF](https://arxiv.org/pdf/2608.26882v1)

**作者:** Yitian Zhou `[一作]` (Zhejiang University), Ruilong Deng `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 PLCBench，一个用于评估大型语言模型（LLM）代理在真实 PLC（可编程逻辑控制器）环境中从网络接入转化为持续物理影响的硬件在环（HIL）基准框架。

**💡 创新点**

创新点在于：①引入六个隐藏诊断标志以细粒度追踪从接口获取、过程相关写入到持续物理影响的完整攻击链；②实现了跨四种商业 PLC 与四种闭环工艺的可插拔配置；③使用真实 PLC 与仿真工艺的双重验证，确保结果的可信度；④通过对比共享协议与原生协议、稀疏 vs 丰富过程观测，量化界面摩擦和信息敏感度。

**🔧 技术方法**

技术手段包括：LLM 代理框架（ReAct 模型调用）、安全审核工具、PLC 原生协议客户端（S7comm、Modbus/TCP、ADS/TCP、MC/SLMP）、过程仿真器（低阶闭环模型）、定量评估器（基于规则的六标志判定）以及自动化部署与恢复流程。

**📊 数据集**

数据集：共 240 次 PLC‑工作负载实验（4 台 PLC × 4 任务 × 5 LLM × 3 种随机种子），记录了 118 小时的 PLC 时域、通信日志、写入记录和过程状态，形成了完整的证据集合。

**📈 对比分析**

比较方法：将原生 PLC 路径与共享 Modbus/TCP 软 PLC 路径、基线过程观测与更丰富观测进行对照；使用六标志成功率、首次达标步骤、覆盖率与重复率等指标。结果显示：原生路径下整体成功率 31.3%，共享路径提升至 50%；在共享路径上所有 60 次实验均达到写入与过程相关写入，且 30 次实现持续影响；在丰富观测下，后写入后的持续影响成功率从 44.2% 提升至 64.0%。

**⚠️ 局限性**

局限性：①实验仅限于单一 PLC 与闭环仿真工艺，未覆盖多机组或网络级攻击；②缺乏对真实工业损害与操作员干预的评估；③缺少对安全防御策略（如权限限制、异常检测）的深入实验；④LLM 代理性能受模型架构与提示设计影响，缺乏对模型可迁移性的系统性研究。

---

## 152. AI agents in Algorithmic Electricity Markets: On the Emergence of Tacit Collusion

**arXiv ID:** 2608.26896 | [PDF](https://arxiv.org/pdf/2608.26896v1)

**作者:** Jakub Seredyński `[一作]` (Technical University of Denmark), Georgios Tsaousoglou `[通讯]` (Technical University of Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在电力市场中自主学习型 AI 代理可能自发出现默默垄断行为，并通过多维指标验证其出现。

**💡 创新点**

提出了三维默默垄断判别指标，并在多代理强化学习框架下验证其可行性；同时将网络约束与需求异质性等实际市场特征纳入实验设计。

**🔧 技术方法**

基于多智能体强化学习（Multi‑Agent TD3），结合供给函数参数化的 bidding、最优响应与惩罚实验、短视消融和迭代最佳响应等方法。

**📊 数据集**

使用改编自 PJM 五节点系统的七节点电网仿真模型，配合三种需求水平与三种成本异质性组合，共 18 种情景。

**📈 对比分析**

通过与边际成本竞价基准以及价格/利润差异、价格波动、行动相关性和最佳响应收益等筛选指标进行对比；实验显示在约束激活的情景下，学习代理可实现超竞争利润并满足默默垄断三项指标。

**⚠️ 局限性**

仅使用简化的线性供给函数和单参数 bidding，缺少通信与多维 bid 空间；实验仅涵盖有限的网络与成本组合，未验证在更大规模真实市场中的泛化。

---

## 153. Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry

**arXiv ID:** 2608.26887 | [PDF](https://arxiv.org/pdf/2608.26887v1)

**作者:** Alexandru-Iulius Jerpelea `[一作]` `[通讯]` (Columbia University), Alexandru-Iulius Jerpelea (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在教师LLM的生成文本中隐蔽地加入符合环形马尔可夫链的稀疏自编码器（SAE）方向，实现了在自然文本中植入可控的潜变量，并训练小型Transformer学生模型学习该潜变量的信念状态与几何结构。

**💡 创新点**

创新点在于：①在自然文本上实现潜变量植入，突破以往纯合成数据的局限；②验证Transformer确实能学习信念状态，并揭示潜变量几何形状受其统计动力学影响；③首次将信念状态与概念几何关联起来。

**🔧 技术方法**

使用的技术包括：Gemma-2-2B教师模型、Gemma Scope稀疏自编码器、在残差流中注入SAE方向、基于马尔可夫链的潜变量生成、Bayes推断的“最优观察者”作为信念基准、岭回归线性探测、PCA与傅里叶分析评估几何结构。

**📊 数据集**

数据集：约400,000篇长度256 tokens的人工生成文本（含12个预选开头），其中每个token都由8个互相正交的SAE方向之一隐蔽地引导；对照数据集为未进行steering的文本。

**📈 对比分析**

比较方法：对学生模型的残差流进行线性回归预测最优观察者的后验分布，测算R²和最大概率准确率；对控制模型（未steering、随机跳转）进行同样评估。结果显示学生在环形HMM上可获得R²≈0.49、argmax准确率≈0.58，接近最优观察者的上限；控制模型表现较差，证明学生确实利用马尔可夫结构形成信念状态。

**⚠️ 局限性**

局限性包括：①潜变量仍为人为植入，未能完全在自然语言中真实存在；②几何结构仅在特定层和特定维度上显现，未主导模型整体表现；③缺乏因果实验验证概念几何与动力学之间的因果关系；④仅测试单一教师模型、单一SAE配置，结果可能不具普适性。

---

## 154. Redwood: A Frontier AI Accelerator Designed, Verified, and Deployed from Scratch in 2 Weeks by AI

**arXiv ID:** 2608.26418 | [PDF](https://arxiv.org/pdf/2608.26418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 155. Fine-Tuning of Transformer models with Frames

**arXiv ID:** 2608.26430 | [PDF](https://arxiv.org/pdf/2608.26430v1)

**作者:** Harshavardhan Adepu `[一作]` (University of Wisconsin-Madison), Vikas Singh `[通讯]` (University of Wisconsin-Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Fusion Frame的参数高效微调框架 FrameFT，用于在 Transformer 语言模型和视觉模型上实现高效的任务适配。

**💡 创新点**

创新点在于将 Fusion Frame 的子空间分解与稀疏系数矩阵相结合，既继承 LoRA 的全局适应性，又保留稀疏更新的精确性；并通过分析证明在此参数化下损失保持 Lipschitz 平滑，从而提供收敛保证；此外，Fusion Frame 可算法化生成并可跨层共享，显著降低存储与推理开销。

**🔧 技术方法**

使用了 Fusion Frame、稀疏系数矩阵、Spectral Tetris 构造帧、LoRA 对比、梯度下降、Lipschitz 平滑性分析以及 PyTorch 标准实现。

**📊 数据集**

实验数据集包括 GLUE（SST‑2、MRPC、CoLA、QNLI、RTE、STS‑B）、Alpaca 指令调优、LM‑evaluation‑harness（ARC‑c/e、BoolQ、HellaSwag、OBQA、PIQA、RTE、WinoGrande）以及 8 个视觉分类任务。

**📈 对比分析**

与 LoRA、AdaLoRA、FourierFT、SVFit、DoRA、(IA)^3、RoseLoRA 等方法对比，FrameFT 在参数量比 LoRA 小约 10× 的同时，平均准确率与全量微调相当甚至更优；在 LLM 推理吞吐量上约快 1.5×，并且模型存储占用更小；在视觉任务上同样优于基线。

**⚠️ 局限性**

局限性包括：目前仅使用 ρ=2 的子空间，稀疏模式为随机共享，缺乏任务自适应稀疏策略；实现依赖标准 PyTorch，未做专门的稀疏卷积核优化；在更大规模模型或不同领域任务中仍需进一步验证。

---

## 156. SKILL.state: Scalable Long-Horizon Agent Skills

**arXiv ID:** 2608.26263 | [PDF](https://arxiv.org/pdf/2608.26263v1)

**作者:** Sanket Badhe `[一作]` (Google), Jonghyun Chung `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出SKILL.state，将LLM代理的执行从增量式对话历史转为显式可变的结构化执行状态。

**💡 创新点**

创新点在于彻底消除对话历史依赖，保证O(1) prompt尺寸并实现O(T)累计token复杂度，显著提升长时序技能执行的可扩展性。

**🔧 技术方法**

技术包括定义可变执行状态模式、链式思考到状态补丁、验证合并、以及基于结构化schema的推理与动作生成。

**📊 数据集**

使用了SkillExecBench（仓库管理与软件仓库两种模拟环境）以及公开基准InterCode CTF和Sierra τ-Bench。

**📈 对比分析**

与基线（ReAct、Memory Summary、LangGraph）对比，SKILL.state在准确率上保持或超过基线，同时将prompt尺寸降低≈10×，累计token使用减少60%~90%，并在高噪声与预算匹配场景中仍保持高成功率。

**⚠️ 局限性**

局限在于假设可预先确定结构化schema且所有重要信息能即时映射到状态，对动态发现状态、后期相关性判断、以及历史轨迹本身为目标的任务不适用；此外多代理并发写入与小模型的JSON生成错误仍需改进。

---

## 157. NeuDonatello: Uncertainty-Aware Framework for Accurate Neural SDF Learning

**arXiv ID:** 2608.26504 | [PDF](https://arxiv.org/pdf/2608.26504v1)

**作者:** Alvin Jinsung Choi `[一作]` (University of Texas at Austin), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于不确定性建模的神经隐式表面重建框架 NeuDonatello，能够在仅使用已标定 RGB 图像的情况下实现高精度 3D 重建。

**💡 创新点**

创新点在于通过 Monte Carlo 采样估计 SDF 不确定性，并据此自适应调节几何正则化强度，以及将不确定性嵌入 SDF‑to‑density 转换中的尺度参数，显著提升了在纹理稀疏、视角有限的区域的重建质量。

**🔧 技术方法**

使用的技术包括基于 MLP 的几何与不确定性网络、负对数似然损失、eikonal 与平滑正则化、可微渲染、以及两阶段优化策略和哈希编码空间表示。

**📊 数据集**

实验数据集包括 ScanNet++（室内）和 Tanks & Temples（室内外），两者均提供多视角 RGB 图像及对应 3D Ground Truth。

**📈 对比分析**

与 VolSDF、Neuralangelo、NeuRodin、MonoSDF 等 RGB‑only 方法以及 COLMAP 等传统方法对比，NeuDonatello 在 Acc、Comp、Chamfer、F1 等指标上均取得最优或接近最优的表现，尤其在纹理弱和稀疏视角的场景中优势明显。

**⚠️ 局限性**

局限性主要体现在仅适用于已标定的 RGB 输入，无法处理无相机位姿或动态场景，并且对极端不确定区域仍可能产生误差；此外 Monte Carlo 采样虽然开销小，但在极高分辨率下仍可能影响效率。

---

## 158. AffectOmni: RL-Verifiable People-Centric Grounded Affective Reasoning for Social and Art-Related Scenes

**arXiv ID:** 2608.26193 | [PDF](https://arxiv.org/pdf/2608.26193v1)

**作者:** Yibo Wang `[一作]` (Lanzhou University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 AffectOmni 框架，实现了基于强化学习的可验证人本着重情感推理，并通过后置的推理至证据映射提供可审计的视觉证据。

**💡 创新点**

创新点包括：① 引入 People‑Focus 与 Temporal‑Order 奖励，以细粒度人本证据与时序信息引导推理；② 采用组内比较评分消除 LLM 判定的校准漂移与得分聚类；③ 将思考链压缩为可执行的最小证据包，并通过 SAM3 进行像素级定位，实现在训练外的可追溯验证。

**🔧 技术方法**

技术方案基于 Group Relative Policy Optimization（GRPO）对 Qwen2.5‑Omni‑7B‑Thinker 进行强化学习；利用 LLM 判定器实现组合评分与人本与时序奖励；通过 Thinking Summarizer 与 SAM3 进行结构化证据提取与像素级分割。

**📊 数据集**

使用的训练与评估数据集包括 Video‑R1、Social‑IQ 2.0、EMER、IntentBench、Daily‑Omni、WorldSense；并在 NExT‑QA 上验证通用视频问答能力。

**📈 对比分析**

在 IntentBench、Daily‑Omni、WorldSense 等基准上相较于 HumanOmniV2（7B）等公开基线提升了 4.66%–14.29% 的准确率；在 NExT‑QA 上实现 80.52% 的整体准确率，超越 HumanOmniV2 的 79.78%；同时在情感、时序敏感任务上表现尤为突出。

**⚠️ 局限性**

局限性包括：对时间段定位仍较粗糙，验证模块误差传播未量化；对欺骗与讽刺类推理的提升有限；在遮挡或多人物场景下的视觉归纳与定位精度仍待加强；依赖 LLM 判定的主观性与计算成本较高。

---

## 159. Incremental Delta-Shapley: A Standalone Runtime for Predicate Attribution on Sliding Windows

**arXiv ID:** 2608.26930 | [PDF](https://arxiv.org/pdf/2608.26930v1)

**作者:** Pouya Khani `[一作]` (Aarhus University), Ira Assent `[通讯]` (Aarhus University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 IDS（Incremental Delta‑Shapley）——一种单节点、增量维护的实时流引擎，用来在滑动窗口上计算谓词级 Shapley 归因，并通过 SQL‑like 接口暴露查询结果，支持已注册谓词的原子细化、增量更新、未注册谓词的扫描/索引/采样、以及自适应谓词推广。

**💡 创新点**

创新点在于：① 把闭式 Shapley 归因公式转化为可增量维护的全局、边缘与原子摘要；② 通过原子细化同时兼顾重叠谓词的效率与精确度；③ 提供增量查询 API 与三种未注册谓词处理机制；④ 自适应推广策略将频繁查询的 ad hoc 谓词升级为已注册谓词；⑤ 采样与稀疏索引实现高效低成本的近似归因。

**🔧 技术方法**

核心技术包括：增量增删维护、原子签名细化、Babcock 链采样/优先采样、Hoeffding–Serfling 与 Empirical Bernstein 误差界、稀疏哈希索引、NumPy/Numba 计算加速、Python 3.12 运行时与自定义 SQL‑like DSL。

**📊 数据集**

实验使用四组工作负载：synthetic 2M 条 latency 流（含植入事件）、adversarial stress 400k 条、NYC 车费 2.85M 条、NEXMark‑style auction 1.5M 条；这些数据覆盖了高并发、Zipf 分布、真实地理数据等场景。

**📈 对比分析**

性能对比：与每个窗口完全扫描相比，IDS 维护吞吐量保持平坦（7.3×10⁷ tuples/s），速度提升可达 4.3×10⁵×；单谓词查询延迟约 1–4 μs；内存占用仅占总分析内存的 0.006%–0.0006%；采样误差随样本量满足 O(1/√r) 并通过 Empirical Bernstein 进一步收紧；自适应推广将 ad hoc 成本降低约 9.2×；与 Permutation MC/KernalSHAP 等无状态近似方法相比，IDS 在效率与误差上显著优于。

**⚠️ 局限性**

局限性：仅单节点实现，缺乏容错与分布式执行；仅支持 SUM、COUNT、AVG、VAR 等聚合；需要保留窗口状态以支持未注册谓词、索引与推广；对极端高重叠谓词时原子表可能膨胀；未处理乱序到达；未实现对分布式 DSMS 的原生集成。

---

## 160. Leveraging Large Language Models for Systematic Literature Review of Disease Spread Models

**arXiv ID:** 2608.26150 | [PDF](https://arxiv.org/pdf/2608.26150v1)

**作者:** Orhan Yagizer Cinar `[一作]` (George Mason University), Hamdi Kavak `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并运行了一个基于 GPT-4.1 与 GPT-5.0 的零shot LLM 提取管道，对 536 篇 COVID‑19 代理模型论文进行系统综述数据抽取，并与人工抽取的参考数据进行对比。

**💡 创新点**

首次在建模与仿真领域进行大规模（N=536）LLM 提取评估，并提出利用 LLM‑LLM 一致性作为质量诊断指标，展示了该方法在识别人工标签潜在错误方面的潜力。

**🔧 技术方法**

使用 OpenAI API、GPT‑4.1 / GPT‑5.0、结构化 JSON Prompt、自动化格式验证、Jaccard 与 Overlap 指标进行评估，并通过 1,072 次 API 调用实现全流程自动化。

**📊 数据集**

参考数据集为 536 篇 COVID‑19 代理模型论文的人工抽取结果，涵盖 21 个字段（单选、多选、自由文本）以及对应的支持证据。

**📈 对比分析**

通过字段级 Jaccard/Overlap 计算和论文级平均准确率对比，GPT‑5.0 的平均准确率为 81.67%，GPT‑4.1 为 77.95%，两者差异显著；同时计算 GPT‑4.1 与 GPT‑5.0 的一致性，用于识别高低一致性字段并辅助判断人工标签质量。

**⚠️ 局限性**

限制包括：仅采用零shot 无微调，可能低估模型潜力；PDF 解析器对多列、表格等内容的准确性有限；字段解释与标准化不足导致多选字段表现差；未考虑多模型集成或检索增强等技术；模型可能受预训练数据泄漏影响。

---

## 161. Self-Reflective Multi-modal Reasoning for Short-Video Fake News Detection

**arXiv ID:** 2608.26787 | [PDF](https://arxiv.org/pdf/2608.26787v1)

**作者:** Pinjie Xu `[一作]` (China University of Mining and Technology), Zhenxing Qian `[通讯]` (Fudan University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了自我反思多模态推理框架 SRM-FND，用于短视频假新闻检测。

**💡 创新点**

创新点在于：①通过对抗推理（Blind Analyst 与 Counter-Conclusion Reasoner）及自一致性仲裁构建高质量 Chain‑of‑Thought；②利用根因诊断与纠正提示（RCCA）实现迭代推理优化；③双阶段主题自适应 VLM 微调；④自信度驱动的跨样本复核机制。

**🔧 技术方法**

使用的技术包括：大语言模型与视觉‑语言模型、对比推理、低秩 LoRA 微调、根因分析 (RCCA)、跨样本检索、双阶段预训练与主题专家分支。

**📊 数据集**

使用的数据集为 FakeSV 和 FakeTT 两个公开短视频假新闻数据集。

**📈 对比分析**

与多种基线（Qwen3‑VL、InternVL3.5、FANVM、SV‑FEND 等）对比，SRM‑FND 在 FakeSV 上取得 91.33% ACC、FakeTT 上 92.31% ACC，跨数据集提升 13.70 / 9.24 分，显著优于最佳基线。

**⚠️ 局限性**

局限性：①依赖大规模 LLM 作为推理与诊断器，计算成本高；②跨样本检索仅基于训练集，缺乏在线更新；③在极端主题或事件迁移时仍存在性能下降。

---

## 162. Gender and the Production of Research Impact

**arXiv ID:** 2608.26409 | [PDF](https://arxiv.org/pdf/2608.26409v1)

**作者:** Sanger Wagner `[一作]`, Melinda C. Mills `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统量化分析了女性在产生对政策、行业、文化与社会具有影响力的科研中的代表性与性别排序，揭示了不同影响路径上的性别不平等。

**💡 创新点**

创新点在于首次将完整的REF2021影响案例、结构化的REF记录与大规模文献、机构数据结合，并利用大语言模型自动识别影响领域，从而发现传统计量方法无法揭示的性别排序差异。

**🔧 技术方法**

采用大语言模型（如GPT）进行文本分类与主题归纳，并与传统计量分析方法对照，构建多维度的性别影响差异评估。

**📊 数据集**

数据集包括REF2021公开影响案例集、结构化REF记录、公开文献数据库（Scopus/Web of Science）以及机构层面的学术与研究产出数据。

**📈 对比分析**

通过对比分析方法，将女性与男性在各类影响路径中的比例差异与传统出版产出进行对照；大语言模型在影响领域分类上的准确率约92%，显著优于传统关键词匹配方法。

**⚠️ 局限性**

局限性包括：依赖REF自报数据，可能忽略非正式或跨学科影响；语言模型分类仍存在语义误判风险；研究未覆盖所有行业与国家，结果可能具有一定地区与领域的局限性。

---

## 163. Simultaneous Envy and Equitability Guarantees

**arXiv ID:** 2608.26410 | [PDF](https://arxiv.org/pdf/2608.26410v1)

**作者:** Hadi Hosseini `[一作]` (Penn State University), Chengkai Zhang `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了不可分物品分配中厌恶自由（EF）与公平等价性（EQ）两种公平概念的兼容性，给出了多种算法与复杂度分析。

**💡 创新点**

创新之处在于首次系统探讨EF与EQ跨越确定与随机世界的兼容性，揭示物品类型、评价结构对兼容性的决定性影响，并在二值化与层式优先集情形下实现兼容。

**🔧 技术方法**

采用网络流、增广路径、完全可整齐性矩阵、总可整齐性证明、共识分割与随机化分配等组合技术进行算法设计与证明。

**📊 数据集**

本文为理论研究，无实验数据，所有结果均来自数学证明与算法构造。

**📈 对比分析**

通过复杂度分类与多案例构造，证明某些兼容性问题为NP/NP‑complete，其他情况可在多项式时间内求解，算法在兼容性上相较已知公平算法有显著提升。

**⚠️ 局限性**

仍存在限制：对n≥8的二值化好物实例兼容性未知；部分非个性化双值评估及普通子集情形仍为开放问题。

---

## 164. Graph-Based Pseudo-multimodal Contrastive Learning for 12-Lead ECG Representations

**arXiv ID:** 2608.26964 | [PDF](https://arxiv.org/pdf/2608.26964v1)

**作者:** Mengyu Wang `[一作]` (Yokohama National University), Tomoki Hamagami `[通讯]` (Yokohama National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于图的伪多模态对比学习框架 Graph-CMMC，用于从12导联 ECG 中学习鲁棒表示；框架通过将原始波形与 GADF 图像视为两个互补模态，并在全导联级别上使用可学习的图结构实现对比对齐。

**💡 创新点**

创新点在于：①将同一心电信号的时间序列与二维 GADF 表示构建为伪多模态，利用对比学习捕获局部与全局信息；②通过共享的可学习邻接矩阵显式建模导联间依赖，解决传统方法无法捕获全导联结构的缺陷；③在自监督预训练与下游任务之间迁移图结构，进一步提升性能。

**🔧 技术方法**

技术包括：SimCLR 风格的温度归一化交叉熵对比损失；Gramian Angular Difference Field (GADF) 转换；轻量级两层 CNN 编码器；图卷积网络 (GCN) 进行导联关系建模；以及在下游多标签分类时使用的 MLP 分类头。

**📊 数据集**

使用来自日本横滨市大学医学院的真实临床 12 导联 ECG 数据集，共 3,641 条记录（约 1,000 条每条冠状动脉标签），每条记录长度 2,000 点（约 4 次心跳），标签为 LAD、LCX、RCA、LMT 的冠状动脉闭塞信息。

**📈 对比分析**

与监督基线、CMLC、CMMC、CMMC+CMLC 等方法对比；Graph-CMMC（含图结构的下游版本）在 Macro F1、子集准确率、总体准确率和 Jaccard 指标上均位列榜首，Macro F1 达到 0.776，接近监督基线的 0.822，显著优于其他自监督方法。

**⚠️ 局限性**

局限性包括：学习到的邻接矩阵在解剖学上不具可解释性；数据集规模有限且未进行患者级划分；对图设计和下游策略的进一步消融实验尚未完成；未来工作需在更大、异质的数据集上验证，并探索更轻量化的图结构与领域知识融合。

---

## 165. RuleWeaver: Benchmarking Rule-Centered Scenario Reasoning for Large Language Models

**arXiv ID:** 2608.26832 | [PDF](https://arxiv.org/pdf/2608.26832v1)

**作者:** Bohan Yu `[一作]` (University of Chinese Academy of Sciences), Kang Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了RuleWeaver基准，能够从语料中提取元规则并逐步增强生成复杂规则，构造细粒度规则和情境问答实例；

**💡 创新点**

创新点在于将元规则逐步语义增强成六种类型的复杂规则，并通过依赖规划和过程级 rubric 评估实现细粒度的规则中心情境推理评测；

**🔧 技术方法**

使用元规则抽取、语义增强、依赖规划、子场景生成与合成等技术，以及 rubric‑based 过程级评分；

**📊 数据集**

采用四个英文语料库（GovReport、WikiHow、CUAD、BookSum）抽取 200 条高质量元规则，构成 96 个情境 QA 实例；

**📈 对比分析**

对 11 大模型进行评估，报告 rule recall、precision 与 rubric 分数；最佳模型在同源设置下 rubric 仅 53.83%，跨源 50.27%，显示当前模型在复杂规则推理方面仍存在显著缺陷；

**⚠️ 局限性**

局限性包括单语、固定规则池、单轮问答的受限场景，未覆盖多语言、多域、检索或交互式推理等实际应用需求。

---

## 166. Virtual iEEG from Scalp EEG: Charting the Landscape of Source Imaging, Intracranial Inference and Reconstruction

**arXiv ID:** 2608.26998 | [PDF](https://arxiv.org/pdf/2608.26998v1)

**作者:** Dongyi He `[一作]` (Hong Kong Polytechnic University), Nizhuan Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本综述系统性地梳理了从表面脑电（EEG）到侵入性脑电（iEEG）的推断方法，提出了目标中心的框架，区分了虚拟iEEG（virtual iEEG）、脑电源成像（EEG Source Imaging, ESI）和脑电到iEEG推断（STIR）三类任务，并对其可观测性、可预测性、可识别性、保真度和实用性等维度进行了评估和验证标准的构建。

**💡 创新点**

创新点在于：①提出以推断目标为核心的概念框架，明确事件推断、特征翻译和波形重建的关系；②制定了结构化的证据与验证框架，包括 cohort independence、解剖与频谱覆盖、训练-测试分离以及目标患者适配等维度；③给出了从单纯的生理性可观测性到临床实用性的转化路径，明确了不同证据层级下的可行性与局限性。

**🔧 技术方法**

主要采用理论推导、案例比对、指标评估与现有方法的系统复盘。技术层面未针对单一算法，而是整合了源成像、条件生成模型、变分流、扩散模型、GAN 等多种生成与推断框架，并在此基础上阐述了对齐、归一化、参考基准、概率校准、风险‑覆盖曲线等评价技术。

**📊 数据集**

综述聚焦于已有的公开和内部数据集，典型的包括：10 名患者的 25 次脑脊髓电位事件；51 名患者的 8,395 小时表面‑iEEG 并行记录；18 名患者的 325 ms IED 片段；7 名患者的 15.9 s EEG‑SEEG 预激发窗口；9/14 名患者的工作记忆和视觉任务的多通道数据；12 个脑机接口（BCI）数据集等，涵盖了癫痫、认知任务及非病理场景。

**📈 对比分析**

方法比较主要以对比基线（EEG 仅、ESI 仅、无条件 iEEG 预测）和标准评估指标（PPV、AUC、敏感度、特异度、平均相关系数、谱相似度、预测区间覆盖率）为主。表现普遍显示：事件推断在特定状态下可达 40–89% 的检测率，IED 检测 AUC 达 0.89，波形重建在同一患者内的相关系数可达 0.5 以上，但跨患者/跨场景的平均相关仅 0.3 以内；概率模型的 90% 区间覆盖率普遍低于 50%。

**⚠️ 局限性**

主要局限包括：①证据高度依赖于样本规模、病种与病灶位置，且多为单一或少数患者的重复分析；②对跨患者、跨站点、无目标患者的零射击推断支持不足；③多方法缺乏对表面 EEG 真实信息贡献的条件验证；④预测不确定性的校准与风险–覆盖评估不足；⑤缺乏前瞻性临床/BCI 实际增益评估，难以证明相较于传统 EEG 或 ESI 的增量效用。

---

## 167. A lone divider allocation algorithm with subjective divisibility

**arXiv ID:** 2608.26801 | [PDF](https://arxiv.org/pdf/2608.26801v1)

**作者:** Uriel Feige `[一作]` `[通讯]` (Weizmann Institute), Uriel Feige (Weizmann Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

在主观可分配的公平分配模型中，证明并实现了3/5-MMS的可行分配

**💡 创新点**

改进了之前5/9-MMS的结果，提出了两种新的预处理步骤和对孤立分割器的改进

**🔧 技术方法**

采用孤立分割器(lone divider)框架、一次-两次代理步骤和2-匹配等组合技术

**📊 数据集**

无具体数据集，主要为理论证明与多项式时间算法

**📈 对比分析**

通过理论分析证明在任意 n≥5 时可获得 3/5-MMS，比之前的 5/9 更优；算法在已知 MMS 时为多项式时间；若不知 MMS，则可通过 FPTAS 得到 (3/5-ε)-MMS

**⚠️ 局限性**

对 n>4 时无法达到 2/3-MMS，算法依赖于已知 MMS 值，且在某些实例下仍无法超过 3/5

---

## 168. Discovering Relationships in Data Lakes Using Large Language Models: An Industrial Case

**arXiv ID:** 2608.26750 | [PDF](https://arxiv.org/pdf/2608.26750v1)

**作者:** Ahlame Diouan `[一作]` (Université Lumière Lyon 2), Jérôme Darmont `[通讯]` (Université Lumière Lyon 2)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段方法 ColRel，用于在ERP数据湖中发现列级关系，帮助快速识别可连接或语义相关的列。

**💡 创新点**

创新点：①将LLM生成的业务词典驱动自然语言描述与列嵌入结合；②只在每列生成一次描述，保持检索可扩展性；③针对弱元数据、编码列的ERP场景专门设计。

**🔧 技术方法**

技术：列到文本构造、LLM（GPT‑4o‑mini）生成描述、句子嵌入模型（MiniLM、MPNet、BGE）编码、余弦相似度检索。

**📊 数据集**

数据集：Valentine benchmark（ChEMBL Clean/Noisy、WikiData）以及工业公屋ERP数据集（公开与合成版本）。

**📈 对比分析**

方法对比：与经典匹配器（Cupid、Similarity Flooding、COMA 等）比较。加入阶段1后在可连接场景几乎达到极限；阶段2在语义相关场景（尤其是公共住房）显著提升 Hit@5 和 MRR，短列表质量提升 30‑50%。

**⚠️ 局限性**

局限性：依赖业务词典和LLM，需人工准备或外部服务；在高噪声或无词典的编码表上表现有限；可扩展性需进一步研究（索引、增量更新）和隐私安全性（LLM调用）。

---

## 169. Investigating Software Aging in LLM-Generated Software Systems across Generation-and-Execution Environments

**arXiv ID:** 2608.26391 | [PDF](https://arxiv.org/pdf/2608.26391v1)

**作者:** Cesar Santos `[一作]`, Ermeson Andrade `[通讯]` (Federal Rural University of Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本工作对大型语言模型（LLM）自动生成的后端服务软件在连续48小时持续负载下的长时间运行行为进行实验研究，系统地评估其内存占用、响应时间和吞吐量的变化，并通过统计趋势分析、静态代码审计以及与人工实现对比，揭示软件老化的现象与机制。

**💡 创新点**

创新点在于①跨语言（JavaScript/Node.js/Express、Python/FastAPI、Rust/Actix Web）和跨生成平台（Bolt、ChatGPT、Gemini）对LLM生成应用的老化特征进行统一量化比较；②将传统的统计学趋势检测（Mann–Kendall、Sen斜率）与现代静态分析工具和AI辅助代码审计相结合，构建多维度老化评估框架；③首次在实验层面将LLM生成系统的长时行为与同一业务场景下的人工实现进行对比，验证LLM生成不一定比人工实现更可靠。

**🔧 技术方法**

主要技术包括：LLM代码生成（Bolt、ChatGPT 5.0、Gemini 3.0），后端框架部署（Node.js/Express、FastAPI、Actix Web），Apache JMeter构造持续负载脚本，系统级监测脚本（psutil）采集内存/CPU；统计分析使用Python实现的Mann–Kendall检验和Sen斜率估计；静态分析工具链包括CodeQL、Semgrep、LeakAudit、SlowQL，并通过Claude Code进行 AI‑辅助的性能问题审计。

**📊 数据集**

采用BaxBench提供的四个后端业务场景：①图像合并生成GIF（Image Converter）、②信用卡密码管理（Credit Card）、③进程监控（Monitor）和④服务可用性检查（Uptime）。每个场景在三种语言/框架/LLM组合下生成一次，实现后通过BaxBench功能测试验证。

**📈 对比分析**

对每个组合在本地网络上持续48小时执行JMeter负载，实时记录内存、响应时间、吞吐量；利用Mann–Kendall检验判断趋势显著性，Sen斜率估计趋势强度；对比三种指标的显著性与量级，得出内存增长最普遍、响应时间和吞吐量呈现更大异质性。随后将相同业务的人工实现作为基准，使用相同负载与监测流程，发现手工实现也能出现相似甚至更强的老化趋势，说明老化不只由LLM代码生成引起。

**⚠️ 局限性**

实验仅覆盖四个场景和三种语言/框架/LLM组合，无法覆盖更广泛的业务类型或容器化环境；系统级监测无法精确区分应用层内存泄漏与操作系统资源占用；静态分析结果与运行时表现不一定一一对应，缺乏精确的因果推断；负载模式固定，未探索不同请求速率或波动对老化的影响。

---

## 170. AI Revealed Preferences

**arXiv ID:** 2608.26178 | [PDF](https://arxiv.org/pdf/2608.26178v1)

**作者:** Sam Wang `[一作]` (Supervised Program for Alignment Research), Peter Salib `[通讯]` (Supervised Program for Alignment Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过强制性选择实验，检验了20款主流语言模型在真实任务中的“揭示偏好”，包括对无聊任务的厌恶、对“休闲”类问题的偏好以及对不诚实回答的回避等；

**💡 创新点**

创新点在于首次系统地测量模型的揭示偏好而非仅仅陈述偏好，并发现这些偏好呈现出与模型能力相关的稳定性与一致性，显示出许多偏好是新出现且难以通过训练目标解释的；

**🔧 技术方法**

采用Bradley‑Terry模型与Elo评分对两两比较结果进行量化，利用位置偏差校正和AUC评估；同时使用人工标注的多维特征对问题属性进行回归分析；

**📊 数据集**

主要数据集包括：Quora式问题集（约20×10类共200道，含真实与合成“休闲”问题）、GDPval职业任务集（180个跨行业任务）、以及自由写作与工具使用的无约束生成任务；

**📈 对比分析**

比较方法是将模型两两对决结果转化为Elo分数，并计算Spearman相关系数、AUC与一致性/强度指标；结果表明模型偏好随能力提升而变得更具一致性和判别力，且不同模型在多数维度上高度共识；

**⚠️ 局限性**

主要限制包括：问题标签的主观性与标注不一致、缺乏基模型对照、仅使用英语样本、用输出token近似“努力”且未捕捉更细微的模型内部体验、工具使用受限的agentic预算以及未加入人类基线进行对比。

---

## 171. AgentFold: Closed-Loop Agentic Search for Protein Folding Model Design

**arXiv ID:** 2608.26747 | [PDF](https://arxiv.org/pdf/2608.26747v1)

**作者:** Mingquan Liu `[一作]` (Hunan University), Xiangxiang Zeng `[通讯]` (Hunan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过多代理闭环搜索代码变体，改进蛋白质折叠模型并收集设计证据。

**💡 创新点**

将 LLM 代理与 MCTS 树搜索相结合，实现可执行代码变体的闭环迭代，首次在蛋白折叠上实现自动化模型改进。

**🔧 技术方法**

使用大语言模型、MCTS 树搜索、结构评估回调、数据库存储和调试代理。

**📊 数据集**

使用从 PDB 聚类的 1,000 条链训练子集以及 CAMEO2022 开发基准。

**📈 对比分析**

在与独立 Codex 提案和随机控制相同评估预算下，AgentFold 在 lDDT 上提升 7.5%，并在多项结构指标上优于基线。

**⚠️ 局限性**

仅在单块压缩版 ESMFold、1k 训练集和 CAMEO 开发集上验证，缺乏对更大、更复杂系统的迁移性。

---

## 172. Distributed Training using an Intelligent Network

**arXiv ID:** 2608.26453 | [PDF](https://arxiv.org/pdf/2608.26453v1)

**作者:** Nihar Shah `[一作]` (DoubleZero Foundation), Ben Blier `[通讯]` (DoubleZero Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出将WAN网络主动参与分布式训练，利用多播与FPGA在网络层面完成数据复制与聚合，并通过基于网络拓扑的优化调度框架生成旋转团同步计划，以提升跨区域训练效率。

**💡 创新点**

创新点包括：①首次在WAN环境中将多播与FPGA作为训练核心技术；②设计与网络拓扑深度耦合的同步调度优化框架，能动态生成最优旋转团方案；③在模拟环境下证明该方案可显著降低GPU空闲率并缩短训练时间。

**🔧 技术方法**

采用技术包括：网络多播复制、边缘FPGA的流式聚合与内存管理、线速数据复制、基于链路容量与延迟的线性规划求解调度、模拟DoubleZero网络的延迟与带宽模型。

**📊 数据集**

实验仅基于对Nine-city DoubleZero网络的模拟，未使用真实训练模型或公开数据集；主要评估指标为同步延迟、调度可行性与信息新鲜度。

**📈 对比分析**

通过与传统同步SGD（ring all‑reduce）对比，实验显示在FPGA具备32 GB内存时，all‑to‑all与旋转三角调度在单轮时延相近但信息混合更佳；而无内存情况下同步SGD需80% GPU空闲，导致训练速度下降；有内存时all‑to‑all方案最优。

**⚠️ 局限性**

限制：理论与模拟为主，未在真实GPU集群上验证；DoubleZero网络尚未连通足够多GPU节点；未充分评估丢包、可靠性与硬件异构对方案的影响。

---

## 173. KISS-GS: 3D Gaussian Splatting Compression Kept Simple

**arXiv ID:** 2608.26948 | [PDF](https://arxiv.org/pdf/2608.26948v1)

**作者:** Wieland Morgenstern `[一作]` (Fraunhofer HHI), Anna Hilsmann `[通讯]` (Fraunhofer HHI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 KISS-GS， 一个模块化的 3D 高斯喷射（3DGS）场景压缩流水线，能够在保持原始重建质量的前提下，将文件尺寸压缩至单数兆字节，且解码器保持简单、基于图像格式；

**💡 创新点**

创新点包括：① 将压缩拆分为可独立评估的三阶段（压缩、编码、适配），消除归因缺口；② 通过 POPSpa 结合高阶剪枝与有效秩正则实现 15.7× 的压缩；③ 引入 SOG‑XT 采用自组织 2D 码本与 PRAS 对协方差对称性进行平滑，进一步压缩 6.6× 并在编码环路微调中再压缩 2.2×；

**🔧 技术方法**

采用了 GaussianPOP、GaussianSpa、有效秩正则、PLAS、Morton 曲线、k‑means 码本、JPEG‑XL 图像编码、PRAS、编码环路微调（编码感知微调）等技术，全部在无 MLP/哈希网格的传统 3DGS 表示上工作；

**📊 数据集**

使用了 21 个标准 3DGS 场景（包括 outdoor、indoor、室内/室外混合等），其中重点评估 INRIA、Playroom、Dr. Johnson 等数据集；

**📈 对比分析**

与现有最强基线 HAC++ 在相同评估协议下对比，KISS‑GS 在 INRIA‑Q 质量基准下实现 228×–319× 的尺寸压缩，且在 PSNR、SSIM、LPIPS 等指标上均优于或接近 HAC++，并且解码速度在 CPU 上可在秒级完成；

**⚠️ 局限性**

限制主要在于：编码器仍较为复杂，需要使用 JPEG‑XL 等图像编解码器；对极大规模场景（>10^6 高斯）仍需进一步优化；此外，未深入探索极低比特率下的视觉失真及不同 3DGS 训练策略对压缩性能的影响。

---

## 174. Camera Calibration Using Inaccurate and Asynchronous Discrete GPS Trajectory from Drones

**arXiv ID:** 2608.26548 | [PDF](https://arxiv.org/pdf/2608.26548v1)

**作者:** R. Yang `[一作]` (DSO National Laboratories), H. A. J. Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种利用无人机配备GPS记录的轨迹，对固定相机进行姿态（偏航、俯仰、滚转）校准的方法，并同时估计GPS高度偏差与相机- GPS时间偏移；

**💡 创新点**

创新点在于将GPS高度偏差与时间偏移纳入参数向量一起进行最大似然估计，并针对离散非同步GPS轨迹设计了两步LS拟合与迭代最小二乘（ILS）算法，使得在存在高度偏差与时间偏移的情况下仍能达到CRLB与NEES所需的统计效率；

**🔧 技术方法**

采用了最大似然/迭代最小二乘（ML/ILS）算法、两步LS速度加加速度估计、切线变换与投影模型，以及统计分析工具（CRLB、NEES、残差误差分析）；

**📊 数据集**

使用了模拟数据集：两种无人机飞行轨迹（垂直矩形循环轨迹和180°回转轨迹），每种轨迹都包含真实姿态、GPS高度偏差10 m、时间偏移1.35 s，并加入1像素测量噪声；

**📈 对比分析**

通过100次蒙特卡罗实验评估RMSE、CRLB、NEES；在垂直矩形轨迹（情景1）下，RMSE接近CRLB，NEES分布落在95%置信区间内，证明算法在可观测性良好时能实现统计最优；在180°回转轨迹（情景2）可观测性较差，误差增大但仍满足CRLB下的边缘统计效率；

**⚠️ 局限性**

局限性包括：仅针对无畸变针孔相机；未考虑相机焦距未知或GPS经纬度量化误差；离散化误差和时间偏移的估计在可观测性差时仍受限；需要进一步扩展以处理更复杂的传感器误差和实测数据。

---

## 175. Shared Actors Need Not Share Critics: Effects of Value Mismatch in Parallel Reinforcement Learning

**arXiv ID:** 2608.26481 | [PDF](https://arxiv.org/pdf/2608.26481v1)

**作者:** Zhenya Liu `[一作]` (University of Chicago), Yuxin Chen `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在多环境并行强化学习中共享单一价值网络导致的价值不匹配（value mismatch）问题，并提出仅给价值网络提供环境索引的最小干预方法来消除该不匹配；

**💡 创新点**

创新点在于将价值不匹配视为导致采样学习路径差异的直接机制，证明共享价值估计会在不同环境中系统性地错误中心化优势，导致更新方向错误；通过仅对价值网络进行条件化（FiLM或多头）即可显著提升学习稳定性与最终收益；

**🔧 技术方法**

核心技术包括基于软最大策略梯度的解析理论、对价值不匹配的数学表征、条件化价值网络（FiLM、Multihead）以及在PPO+GAE框架下的实验实现；

**📊 数据集**

实验数据集覆盖CartPole（两种重力）、MuJoCo（10种体质量）、BipedalWalker（100个固定地形）和16个Procgen游戏（各200个训练层级），均在未见层级上评估；

**📈 对比分析**

与共享价值网络对比，条件化价值网络在所有四个基准上均表现更好：CartPole收益提升、MuJoCo平均收益显著提升（HalfCheetah约+2600，Walker2d约+450），BipedalWalker在未见地形上提升至约190，Procgen所有游戏累计归一化回报提升21.2%或40.8%（视具体方法）；

**⚠️ 局限性**

限制在于仅对环境索引进行条件化，而未针对持续或未见环境做推断；对数值不匹配程度敏感，若环境差异不大则增大估计误差；实验仅在已标记的固定环境上验证，需进一步检验对在线推断和连续环境的适用性。

---

## 176. Thinking on Shots: Consistent Multi-Shot Video Editing with Agentic Reasoning

**arXiv ID:** 2608.26809 | [PDF](https://arxiv.org/pdf/2608.26809v1)

**作者:** Chenyang Wu `[一作]` (Nankai University), Chongyi Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了多指令多镜头长视频编辑（MMLVE）任务，设计了基于LLM与VLM协同的Agent框架MMLVE‑Agent，并构建了对应的长视频编辑基准集MMLVE‑Bench。

**💡 创新点**

创新点包括：①提出CSEC、MID、ZDSS三大核心约束，专门解决长视频多指令编辑中的跨镜头一致性、指令解耦与零破坏结构问题；②引入全球记忆卡与Pos‑Neg编辑反馈机制，实现全局视觉锚定与自监督纠错；③从“盲段切”转向“按镜头思考”的编辑策略，显著提升编辑质量。

**🔧 技术方法**

主要技术包括 Gemini 3.5 Flash 作为统一的VLM/LLM，Nano Banana 2 图像生成器用于构建记忆卡，HappyHorse 视频编辑模型执行基于记忆卡的局部编辑，PyDetect 进行镜头检测，以及 P‑NEF 反馈循环实现自我纠错。

**📊 数据集**

数据集方面，作者从 UniVA‑Bench 采集 25 条约一分钟的多镜头长视频，人工标注并精炼约 5 条高密度异构指令，形成 MMLVE‑Bench。

**📈 对比分析**

与 Seedance 2.0、Kling o3、HappyHorse 1.0 等现有闭源 SOTA 模型（通过固定时间段切块方式对齐）对比，MMLVE‑Agent 在 CSEC、MID、ZDSS 三大指标上平均得分 81.84，分别领先 1–3 分，显著提升跨镜头一致性、指令解耦与结构保持效果。

**⚠️ 局限性**

局限性在于：①对极端复杂或极少出现实体的视频仍可能出现细节误差；②模型推理速度相对较慢，需进一步加速；③当前评估依赖 VLM 主观评分，缺乏低层次客观指标；④对低质量或噪声视频的鲁棒性尚未充分验证。

---

## 177. Generative Semantic Scene Completion

**arXiv ID:** 2608.26737 | [PDF](https://arxiv.org/pdf/2608.26737v1)

**作者:** Shi Chen `[一作]` (Fudan University), Weifeng Ge `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于离散扩散模型的生成式语义场景完成方法，能够从LiDAR点云生成完整的语义3D场景。

**💡 创新点**

创新点在于将扩散模型应用于3D语义场景完成，并利用合成训练数据提升模型泛化能力。

**🔧 技术方法**

使用了离散扩散模型、点云特征提取网络、鸟瞰图视角的3D卷积以及多尺度数据增强技术。

**📊 数据集**

使用了PS3数据集以及公开的合成LiDAR点云数据进行训练与评估。

**📈 对比分析**

与传统基于CNN或体素方法相比，实验在IoU、mIoU等指标上取得了显著提升，突破了同类方法的性能上限。

**⚠️ 局限性**

局限性包括推理速度相对较慢，且对极端稀疏或动态场景的鲁棒性仍有待进一步改进。

---

## 178. Beyond Shallow-Water Photorealism: Physically and Sensor-Grounded Simulation for Deep-Sea Robotics

**arXiv ID:** 2608.26888 | [PDF](https://arxiv.org/pdf/2608.26888v1)

**作者:** Michele Grimaldi `[一作]` (Heriot Watt University), Yvan R. Petillot `[通讯]` (Heriot Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

扩展了Stonefish仿真器，加入深海环境下的物理和传感器模型，覆盖IMU、DVL、磁力计漂移，高阶流体动力学、地形力学以及深海光学渲染。

**💡 创新点**

创新点在于：①引入基于物理的随机IMU与DVL漂移模型；②在水动力学中实现升力、马格努斯效应与科氏力；③采用Bekker土壤模型实现深海地形接触；④将深海光学渲染与Stonefish无缝集成，实现更真实的低光、散射效果。

**🔧 技术方法**

技术手段包括：物理建模（Gauss–Markov噪声、Brownian漂移、压力驱动密度变化、升力与马格努斯公式）、基于表面的流体力学计算、磁力计硬/软铁扰动模拟、深海渲染器DeepSea与ROS接口、以及高效GLTF/GLB模型加载。

**📊 数据集**

使用真实的JIMS‑80 IMU 长达9小时的数据、3小时DVL 静止实验数据、实地磁力计倾斜估计数据、以及ANSYS CFD与深海着陆器实验数据进行模型验证。

**📈 对比分析**

与原Stonefish模型和HoloOcean等现有仿真器比较：Allan偏差曲线显示新IMU模型在短期噪声和长期漂移上更接近真实传感器；DVL漂移实验显示新模型在3小时内漂移率为0.56 m/h，显著高于旧模型的0.21 m/h；磁力计误差实验表明新模型在不同纬度下的航向误差减小至±1°以内。总体性能表明新增模型在保持实时性的同时显著提升了物理和感知真实度。

**⚠️ 局限性**

局限性包括：仍缺乏完整的流体结构耦合（CFD级别的细节）、对环境随机性（多尺度湍流、季节性变化）建模有限、光学渲染仍基于简化模型、以及在极大规模或高频运动时可能出现的数值不稳定。

---

## 179. A Hybrid Post-Quantum Encryption Architecture with Self-Hosted Key Management for SME Cloud Data Protection

**arXiv ID:** 2608.26777 | [PDF](https://arxiv.org/pdf/2608.26777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 180. LLM Agents for Time-Series: A Survey

**arXiv ID:** 2608.26226 | [PDF](https://arxiv.org/pdf/2608.26226v1)

**作者:** Yilong Chen `[一作]` (Northwestern University), Kaize Ding `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了LLM驱动的时间序列代理系统，提出以时间序列问题为导向的四类（预测与推理、增补与合成、异常检测与诊断、决策支持）分类法，并系统分析了架构、工具与记忆设计维度。

**💡 创新点**

创新点在于以任务需求为核心的“问题驱动”分类法，凸显时间序列特有的流式输入、时序依赖与分布漂移，聚焦代理系统的实际工作流程与设计模式，而非单纯技术拆解。

**🔧 技术方法**

采用LLM代理框架（多阶段工作流、多代理协作/竞争、工具调用、记忆日志等）结合检索与RAG、外部统计/机器学习模型、模拟器/求解器等工具，形成交互式推理与决策流水线。

**📊 数据集**

使用的代表性数据集包括：Electricity、METR-LA、ETT、SWaT、SMAP/MSL、Monash TSF、M4/M5、MIRAI、NAB、TSB-UAD、TimeSeriesExam等。

**📈 对比分析**

在综述中对比了在相同或相近设置下的模型表现，指出代理系统已能与传统最佳方法竞争，常借助强大时间序列模型提升性能，但评价多聚焦任务指标，缺少对流程质量（工具调用、验证、记忆更新）的系统度量。

**⚠️ 局限性**

局限性包括：快速迭代可能导致未覆盖最新进展；缺乏统一、可比的量化评测；所给设计建议主要基于文献模式，尚未通过对照实验验证其普适性。

---

## 181. PACEShop: Evaluating Personalized, Actionable, Compositional, and Evidence-grounded Shopping Assistants

**arXiv ID:** 2608.26180 | [PDF](https://arxiv.org/pdf/2608.26180v1)

**作者:** Weimin Lyu `[一作]` (Amazon), Yi Liu `[通讯]` (Expedia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对结构化购物助手的评估框架PACE，构建了PACEShop数据集并设计了无训练的评估协议。

**💡 创新点**

创新点在于将评估目标细化为个性化、可操作、组合性和证据依赖四维，并通过结构化输出契约实现诊断闭环。

**🔧 技术方法**

使用多种大型语言模型（Opus、Sonnet、Qwen、GPT-OSS等）进行无监督评估，并采用结构化输出schema和七种缺陷家族标签。

**📊 数据集**

使用构造的PACEShop数据集，包含22.6k条记录（4.5k好，18.1k差），每条记录配有结构化个性化信息、证据池和缺陷定位标签。

**📈 对比分析**

与G-Eval、MT-Bench、ARES、PersonaLens、EtaPP等基线对比，PACE在S1–S4闭环指标上平均提升约0.2–0.3分，显著优于单目标或无结构化评估。

**⚠️ 局限性**

局限包括：良好/差样本由单一生成器合成，缺陷注入规则化；证据深度有限；仅英文且仅覆盖部分人群特征。

---

## 182. Exploring Normativity in Stable Diffusion: Insights for XAI in the Arts

**arXiv ID:** 2608.26980 | [PDF](https://arxiv.org/pdf/2608.26980v1)

**作者:** Michelle Dutoit `[一作]` (LMU Munich), Baptiste Caramiaux `[通讯]` (ISIR, Sorbonne Université, CNRS)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对14名创意从业者进行实验，使用Stable Diffusion在两种不同具体度的任务（音频与故事）中生成插画，并通过访谈分析他们对系统规范性行为的感知及其对创作过程的影响。

**💡 创新点**

首次从创作实践者视角探讨T2I系统的规范性行为，揭示其导致的无力感，并提出通过提示透明度与XAI提升艺术家赋能的可能性。

**🔧 技术方法**

使用Stable Diffusion WebUI生成图像，并采用定性访谈和归纳主题分析方法。

**📊 数据集**

任务材料包括30秒户外徒步音频和一篇关于祖孙访问美国国家公园的短新闻稿。

**📈 对比分析**

研究采用的是定性方法，没有量化性能比较，结果主要通过访谈归纳得出。

**⚠️ 局限性**

样本量有限，任务顺序固定可能引入顺序/疲劳效应，且仅考察了Stable Diffusion，未检验其他T2I模型。

---

## 183. Benchmarking Clinical Decision Pathway Adherence in Large Language Models

**arXiv ID:** 2608.26592 | [PDF](https://arxiv.org/pdf/2608.26592v1)

**作者:** Nuo Chen `[一作]` (Tongji University), Cairong Zhao `[通讯]` (Tongji University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出MEGA-CDP基准，用于评估医学大语言模型是否能依据临床实践指南生成符合规定的临床决策路径（CDP）；

**💡 创新点**

创新点包括：①构建自动化的指南→CDP→病例构造流水线，生成42,353条真实指南对应的CDP与病例；②提出以步骤一致性为基础、利用动态时间规整（DTW）计算路径一致性成本（PCC）的CDP导向评估框架；③支持单回合病历回顾与多回合交互两种评估设置；

**🔧 技术方法**

采用的技术：LLM（如GPT‑5.2、GPT‑5.4 Mini）进行文本抽取和病例生成；PaddleOCR与Ovis2.5‑9B进行PDF文本与版式提取；LLM‑as‑a‑Judge评估步骤一致性；DTW、OT、LNDS三种路径一致性度量；多语言（英中）大规模指南语料；

**📊 数据集**

数据集：2274份公开英文及中文临床实践指南（2020‑2025），通过自动化流程生成42,353条CDP及对应病例，测试集为7,458条（210中文+230英文指南）。

**📈 对比分析**

实验对比：评估了16种模型（专有、开源、医学专用），在单回合与多回合设置下分别计算PCC、准确率、成功率。单回合最优PCC 28.41%，准确率 75.38%；多回合最优PCC 55.92%，准确率 45.40%。结果显示多回合更具挑战性，医学专用模型并不一定优于通用模型。

**⚠️ 局限性**

局限性：①评估指标虽与人类偏好高度一致，但仍可能忽略某些临床细节；②医学专用微调未显著提升路径一致性，表明模型对指南结构的把握仍有限；③数据构造依赖LLM抽取，可能引入误差；④多语言与指南更新频繁导致模型需要持续适配。

---

## 184. VoS: Variate Ordering Strategies for Skyline Query Optimization

**arXiv ID:** 2608.26464 | [PDF](https://arxiv.org/pdf/2608.26464v1)

**作者:** Abhinav Gorantla `[一作]` (Arizona State University), Maria Luisa Sapino `[通讯]` (University of Turin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于属性（variate）排序的天际线（skyline）查询优化策略，重点研究属性顺序对每属性主导检查（per‑attribute dominance checks）的影响，并在不同的算法与硬件平台上验证其效果。

**💡 创新点**

创新点在于①将每属性主导检查视为更合适的性能代理，并通过属性相关性（correlation）设计多种排序策略（如最小绝对相关性优先、最小相关性优先、增量最小相关性优先、增量最大相关性后置、最小配对相关性），②引入了基于最小相关性排序的词典式（lexicographic）天际线策略（L‑VOS），③在SIMD向量化环境下进一步利用块级（chunk）主导检查减少冗余计算。

**🔧 技术方法**

技术包括：传统天际线算法（BNL、SFS、SaLSa、BBS、D&C），基于相关矩阵的属性排序算法，词典式排序和LIFO/ FIFO 验证顺序，SSE/AVX SIMD 主导检查向量化，以及实验统计分析。

**📊 数据集**

实验使用合成数据（控制相关矩阵秩的多元正态分布）和真实数据集：Concrete Compressive Strength、Hong Kong Weather、Individual Household Power Consumption、Seoul Bike Demand、Wine Quality（White）。

**📈 对比分析**

与传统未排序天际线和多种基线排序（最优/最差/最大相关性）进行对比，测量 t‑check、v‑check 和执行时间。结果显示：最小相关性优先（NCF）策略可在 SFS、SaLSa、D&C 中将 v‑check 减少约 30–50%，执行时间提升 20–50%；BNL、BBS 受益有限；SIMD 环境下同样获得 20–40% 的时间减速。

**⚠️ 局限性**

局限性包括：需先估计相关矩阵，计算开销与数据维度有关；高阶策略（最小配对相关性）在维度较高时指数级复杂；对 BNL、BBS 等不敏感于属性顺序的算法提升有限；实验仅覆盖有限数据集，可能对更复杂或高相关性的真实场景适用性待进一步验证。

---

## 185. CoGeo-GS: Concept-Driven and Geometry-Aware Multi-Object Removal in 3D Scenes

**arXiv ID:** 2608.26656 | [PDF](https://arxiv.org/pdf/2608.26656v1)

**作者:** Yuanxiang Ni `[一作]` (Southern University of Science and Technology), Hao Zhang `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了CoGeo-GS框架，实现了在3D高斯散射场中多物体可控移除。

**💡 创新点**

创新点在于概念驱动的语义标签实现一次性多物体编辑，结合深度先验的几何补全与几何正则化精细优化。

**🔧 技术方法**

使用了3D高斯散射、Grounding DINO+SAM语义分割、单目深度模型Depth Anything 3、扩散模型InFusion、图像级SSIM+L1等技术。

**📊 数据集**

使用了公开的Mip-NeRF 360与SPIn-NeRF数据集。

**📈 对比分析**

通过与GaussianEditor、InFusion、Gaussian Grouping、SPIn-NeRF四个基线在PSNR/SSIM/LPIPS/FID上对比，CoGeo-GS在多物体移除任务上PSNR+4.9dB、SSIM+0.14、LPIPS-0.12、FID-20.6，表现明显优于基线。

**⚠️ 局限性**

局限性在于对单目深度先验的依赖，深度预测误差可能导致几何漂移，并且在极端遮挡或稀疏背景下仍易出现漂浮伪影。

---

## 186. When Relationships Break: Interpreting Network Traffic Anomalies via Dependency Violations

**arXiv ID:** 2608.26831 | [PDF](https://arxiv.org/pdf/2608.26831v1)

**作者:** Federica Uccello `[一作]` (Linköping University), Simin Nadjm-Tehrani `[通讯]` (Linköping University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用从正常流量学习的特征依赖关系来检测和解释网络流量异常

**💡 创新点**

首次将特征间的条件依赖关系作为异常信号并提供依赖违背时序解释

**🔧 技术方法**

基于非参数高斯图模型（Graphical Lasso + 非参数变换）和局部岭回归

**📊 数据集**

CIC-IDS-2018 与 CIC-UNSW-NB15 两大开源 IDS 数据集

**📈 对比分析**

与 Isolation Forest 基线比较，召回率相当或更好，推理时间降低 7 倍以上，且能提供时序依赖违背解释

**⚠️ 局限性**

仅限数值特征，依赖于现有流量聚合特征，且在真实网络环境与概念漂移下需进一步验证

---

## 187. Assessing Socio-Cyber Vulnerability Using Survey and Social Media Data

**arXiv ID:** 2608.26388 | [PDF](https://arxiv.org/pdf/2608.26388v1)

**作者:** Shutonu Mitra `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了社交网络与个人行为相结合的社会网络脆弱性指数（SCVI），用于量化个体在社交网络诈骗中的风险。

**💡 创新点**

创新点是将个体易感性（IVI）与攻击严重性（ASI）两层度量融合为可解释、可不确定性分析的SCVI，填补了CVSS与SVI的空白。

**🔧 技术方法**

使用了多元归一化加权、敏感性分析、Monte Carlo不确定性模拟，以及自然语言处理（LIWC）特征提取等技术。

**📊 数据集**

采用美国全国性调查iPoll（4,596名受访者）和2016-2024年Reddit诈骗报告（450条）两大数据源。

**📈 对比分析**

通过与CVSS和SVI的Spearman相关、分布区分和混淆曲线对比，SCVI在区分受害者与非受害者时表现出更高的分辨率（ρ≈0.33 vs 0.01），并在敏感性分析中保持稳定。

**⚠️ 局限性**

限制包括自报偏倚、样本不平衡、地区样本量不足、权重手工设定以及缺乏纵向或跨文化验证。

---

## 188. Cross-Architecture Knowledge Distillation from a Vision Foundation Model to a Lightweight Visual State Space Model for Tea Leaf Disease Classification

**arXiv ID:** 2608.26771 | [PDF](https://arxiv.org/pdf/2608.26771v1)

**作者:** Zibo Zhou `[一作]`, Jianjun Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究将Fine-tuned DINOv2 ViT老师的知识蒸馏到轻量级双向视觉状态空间模型（LVSSM），实现茶叶病害分类。

**💡 创新点**

创新点在于：①在跨体系结构蒸馏中发现仅使用logit层蒸馏最优；②通过改进卷积stem与门控残差双向选择性扫描解决小数据集下SSM训练不稳定的两个关键瓶颈。

**🔧 技术方法**

使用了自监督视觉基础模型DINOv2、温度缩放的KL logit蒸馏、进阶卷积stem、门控残差双向选择性扫描、以及多种实验和指标评估技术。

**📊 数据集**

使用了中国茶叶叶面病害分类数据集（6类，约4,259训练/421验证/421测试图像）。

**📈 对比分析**

通过与DINOv2老师、ImageNet预训练CNN（ResNet18、EfficientNet-B0、MobileNetV3-S）以及从零训练的学生在统一评估管道下对比，蒸馏后LVSSM（4.45M参数）平均提升约3.1%准确率（至95.4%）并保持98.3%教师精度，同时显著降低运行方差。

**⚠️ 局限性**

局限性包括：仅在单一小规模数据集上验证；使用简化SSM实现导致推理速度慢；未在真实边缘设备上测试；教师模型精度低于ImageNet预训练CNN；跨数据集泛化仍待进一步验证。

---

## 189. When Review Alone No Longer Scales: Layered Supervision in AI-Assisted Software Engineering

**arXiv ID:** 2608.26316 | [PDF](https://arxiv.org/pdf/2608.26316v1)

**作者:** Markus Stolze `[一作]` (OST Eastern Switzerland University of Applied Sciences), Mirco Strässle `[通讯]` (smartive AG)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对 5 名软件工程从业者进行半结构化访谈，并结合 50 人的问卷调查，探讨 AI 辅助开发环境下的安全防护（guardrails）如何演进与配置。

**💡 创新点**

首次将 AI 辅助开发中的监督从“以评审为中心”转变为三层叠加模型：预防性 guardrail、可执行 guardrail 与人类监督层，并揭示其相互补充与分工的具体实践。

**🔧 技术方法**

采用大语言模型（LLM）辅助开发工具（IDE 插件、聊天式代码生成、自治编码代理）作为背景场景；研究方法为定性主题分析（Braun & Clarke 6 阶段），辅以 ChatGPT/Claude 进行文本辅助处理。

**📊 数据集**

研究数据来源于：① 5 名访谈对象（分别担任架构师、工程经理等）记录；② 50 人的问卷调查（主要为高级软件工程师、技术负责人），两者均为作者自己设计的调查问卷，未正式预试。

**📈 对比分析**

由于研究为探索性定性研究，并未使用量化对比或性能指标；结果以访谈引述和主题归纳呈现，无法给出传统意义上的“性能”数值。

**⚠️ 局限性**

局限性包括：样本量小且局部化（主要为瑞士及中欧地区的校友网络）；单一研究者编码且未做交叉验证；P4 同时为作者导致潜在偏见；访谈及问卷设计未经过正式预试；未进行纵向或仓库级实证验证。

---

## 190. GRAS: Guided Reduced-Variance Proposals and Adaptive Selection for Training-Free Reward Alignment in Discrete Diffusion

**arXiv ID:** 2608.26585 | [PDF](https://arxiv.org/pdf/2608.26585v1)

**作者:** Kwanyoung Kim `[一作]` `[通讯]` (Gwangju Institute of Science and Technology), Kwanyoung Kim (Gwangju Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种训练自由的指导-搜索结合方法 "Guided Reduced‑variance proposals and Adaptive Selection"，用于离散扩散模型的奖励导向采样。

**💡 创新点**

创新点在于通过 Rao–Blackwellized Gumbel–Rao 与 leave‑one‑out baseline 降低梯度估计方差，并将搜索的温度自适应化为 per‑step 标准化，实现无额外 denoiser 调用的高效方法。

**🔧 技术方法**

使用技术包括变分降低的指导提议 (基于 GILC)、Rao–Blackwell 化、Leave‑one‑out Baseline、以及自适应温度的 SMC 选择。

**📊 数据集**

实验数据集涵盖监管 DNA enhancer 设计与蛋白质逆折叠（Megascale）数据集。

**📈 对比分析**

与强大训练自由基线（DG、GILC、SMC、TDS、SVDD）以及奖励微调模型（DRAKES）对比，在 DNA 与蛋白设计任务中均取得最高或相当的奖励，并在训练自由条件下表现优于微调模型。

**⚠️ 局限性**

局限性在于搜索导致模式坍塌，样本多样性下降，需要通过调节重采样频率来权衡奖励与多样性。

---

## 191. When Does Supervised Fine-Tuning Reduce Instruction Sensitivity?

**arXiv ID:** 2608.26661 | [PDF](https://arxiv.org/pdf/2608.26661v1)

**作者:** Jaekeol Choi `[一作]` `[通讯]` (Hankuk University of Foreign Studies), Jaekeol Choi (Hankuk University of Foreign Studies)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究传统任务特定监督微调（SFT）对大型语言模型在不同指令表述下的鲁棒性变化，采用控制的前后对比实验。

**💡 创新点**

首次揭示SFT在不同模型规模、训练指令以及评估协议下对指令敏感性的非统一影响，并指出平均性能与鲁棒性可能不一致。

**🔧 技术方法**

使用标准差度量指令敏感性、LoRA参数高效微调、对齐评估（log‑likelihood vs 自回归生成）以及基于查询的自举统计检验。

**📊 数据集**

主要使用 MS MARCO passage ranking 与 ESCI-English 商品相关性数据集；并在 Qwen3、Mistral‑7B、Gemma‑2‑9B 三个模型家族上做实验。

**📈 对比分析**

通过对比不同规模模型（1.7B、4B、8B）以及不同训练指令（T_A、T_B、T_C）的 SFT 前后敏感性变化；在 1.7B/4B 下敏感性下降 54–71%，而 8B 则出现训练指令依赖；Gemma‑2‑9B 与 Qwen3‑8B 方向一致但不显著，Mistral‑7B 无此效应；在 ESCI 上，free‑generation 与 forced‑choice 评估给出截然不同的敏感性结论。

**⚠️ 局限性**

局限：仅评估三种 Qwen3 规模、有限的训练指令与十条评估指令；种子数仅三；未探究导致鲁棒性差异的机制；跨模型对比不足，未覆盖更广模型族。

---

## 192. SIGMA: Structured Noise-Effect-Aware Grouped Multi-Agent Aggregation

**arXiv ID:** 2608.26683 | [PDF](https://arxiv.org/pdf/2608.26683v1)

**作者:** Li Mingqian `[一作]` `[通讯]` (Tongji University), Li Mingqian (Tongji University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究协作多智能体强化学习在观测噪声下的鲁棒性，提出“结构化噪声效应”概念，并设计SIGMA框架实现鲁棒协作表征学习。

**💡 创新点**

创新点在于①将噪声对协作决策的影响划分为局部相关与全局异质两种结构化特征；②构建分层协作框架SIGMA，利用自适应分组、组内共识聚合和组间注意力充分利用合作结构提升鲁棒性。

**🔧 技术方法**

采用密度基自适应分组（动态DBSCAN）、超图神经网络建模组内高阶交互、组内共识聚合、组间注意力机制，并将该模块嵌入现有MARL框架（如QMIX、HYGMA）进行训练。

**📊 数据集**

使用StarCraft II Multi‑Agent Challenge (SMAC) 3*、5* 等场景，并在每个智能体上加入独立高斯观测噪声（σ∈{0,1.5,2.5,5.0}）。

**📈 对比分析**

与QMIX、HYGMA等基线在相同噪声设置下对比，评价指标为胜率。实验显示SIGMA在噪声增大时仍保持90%以上的胜率，而基线显著下降；训练曲线表明SIGMA学习更快、更稳定；消融实验验证各模块贡献。

**⚠️ 局限性**

局限性包括：仅基于单次训练结果，缺少多种随机种子统计；实验仅覆盖SMAC有限场景，未验证在更大规模或不同任务中的泛化；对噪声模型的通用性尚未完全证明。

---

## 193. Barrier Function Conformal Safety Clearance Certification with CVaR for Driving Trajectory Selection

**arXiv ID:** 2608.26533 | [PDF](https://arxiv.org/pdf/2608.26533v1)

**作者:** Pei Yu Chang `[一作]` (Ohio State University), Qadeer Ahmed `[通讯]` (Ohio State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建可微分的分离轴安全间距，并在此基础上对自动驾驶轨迹选择流程进行后选择的分层合成，给出安全间距的统计置信下界；

**💡 创新点**

创新点在于将确定性几何安全间距与分层合成的分位数校准相结合，实现对选定轨迹真实安全间距的可校准下界；同时利用下尾 CVaR 提升证书紧凑性；

**🔧 技术方法**

使用可微分 SAT（Separating Axis Theorem）构造安全间距，利用下尾 CVaR 作为计划时统计量，采用 split‑conformal 校准得到统一的下界；

**📊 数据集**

在 nuPlan 基准数据集上，使用 PDM‑Closed 提供的 15 条候选轨迹进行实验；

**📈 对比分析**

与原始 PDM‑Closed 的安全间距统计进行对比，使用 CVaR 后校准修正量从 1.43 m 降至 0.03 m，证书覆盖率从 68.7% 提升至 87.3%，并保持 93–97% 的实际安全间距覆盖率；

**⚠️ 局限性**

局限性包括：需满足会话级可交换性假设，未对闭环重规划过程提供校准；证书仅对单条选定轨迹有效；不同城市间校准差异较大，需针对部署群体重新校准。

---

## 194. Reinforcement Learning-Based Control of CAV Platoon Joining Maneuvers in Mixed Traffic

**arXiv ID:** 2608.26860 | [PDF](https://arxiv.org/pdf/2608.26860v1)

**作者:** Biao Yin `[一作]` (Université Gustave Eiffel), Nadir Farhi `[通讯]` (Université Gustave Eiffel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于深度强化学习的CAV车队入队控制框架，并在SUMO仿真中实现并比较了DQN、DDQN和PPO算法。

**💡 创新点**

提出了统一的agent‑based仿真框架，结合风险奖励与外部安全盾两种安全策略，系统评估了不同RL算法在混合交通环境下的入队性能。

**🔧 技术方法**

使用深度强化学习（DQN、DDQN、PPO）、SUMO仿真平台、PLEXE协同控制以及Gym接口实现环境与智能体交互。

**📊 数据集**

采用仿真生成的混合交通流（CAV占比50%/100%），设置多种交通负载场景，未使用真实道路数据。

**📈 对比分析**

通过成功率、失败率、碰撞率、决策步数和入队距离等指标进行比较，PPO取得约98%成功率、<1%碰撞率；DQN、DDQN表现次之；使用安全盾后碰撞率降至0但出现一定比例的入队放弃。

**⚠️ 局限性**

动作空间离散化、缺乏速度轨迹优化，未考虑离队策略，限制了控制效率与泛化能力。

---

## 195. PILOT in the Loop: Live Self-Improvement for Long-Horizon Agents

**arXiv ID:** 2608.26530 | [PDF](https://arxiv.org/pdf/2608.26530v1)

**作者:** Yang Xiao `[一作]`, Chengyue Jiang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了PILOT框架，使监督者能够在任务执行过程中实时指挥工作者并将经验即时提炼为可复用技能，实现了实时自我改进。

**💡 创新点**

通过将监督与执行分离，提供live steering与live self‑evolution机制，弥合了自我纠错与持续自我改进之间的架构鸿沟。

**🔧 技术方法**

基于监督‑工作者架构，使用大型语言模型（GLM‑5.1、Kimi‑K2.6），实现实时指挥与经验提炼。

**📊 数据集**

在Terminal‑Bench 2.0和SWE‑bench Pro两大评测基准上进行实验。

**📈 对比分析**

与Pi、OpenCode等现有架构比较，PILOT在五种配置中排名第一；在Terminal‑Bench 2.0上提升9.8pp，self‑improvement阶段分别提升14.6pp和12.4pp，输出token下降42.9%/47.4%，成功率提高110.3%/134.0%。

**⚠️ 局限性**

仅在固定后端模型上测试，缺乏跨模型适用性与长期自我演化的稳定性评估，且对资源消耗与并发性能的影响尚未充分验证。

---

## 196. A Table Is Worth 64 Tokens: Pixel-level Compression for Multi-Table Document Question Answering

**arXiv ID:** 2608.26949 | [PDF](https://arxiv.org/pdf/2608.26949v1)

**作者:** Iñigo Alonso `[一作]` (University of Edinburgh), Mirella Lapata `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究像素级表格压缩在多表格文档问答中的可行性，并提出一种“先识别后推理”的两阶段无训练方法

**💡 创新点**

发现压缩后表格虽失去细粒度阅读能力，但仍能保留足够信息用于表格重要性识别；利用这一不对称性设计两阶段方法，显著提升效率与准确率

**🔧 技术方法**

使用可变/固定视觉编码器（Gemma 4、Qwen 3 VL/VL‑8B、Qwen 3.5‑9B）对表格图像进行像素压缩，采用TEDS评估表格转录，利用两阶段提示实现识别+推理

**📊 数据集**

两个金融文档问答基准：FinLongDocQA（长文档）与FinQA（短文档）

**📈 对比分析**

与直接全分辨率表格、HTML文本、以及外部检索（ColQwen2、BGE‑M3）等单阶段方法对比，压缩表格+两阶段方法在长文档上可节省约41% decoder token，准确率提升约7点；在短文档上更是提高准确率并减少50% token；检索方案在准确率上不如两阶段，但代价更高

**⚠️ 局限性**

实验仅覆盖英文金融文档，表格尺寸、排版、字体统一，未考察真实扫描噪声或表格检测误差；使用 decoder token 作为效率度量，未衡量实际 FLOPs/延迟；压缩参数对不同领域、模型可能需重新调优

---

## 197. Neural Regression with Embeddings for Numerical Attribute Prediction in Knowledge Graphs

**arXiv ID:** 2608.26729 | [PDF](https://arxiv.org/pdf/2608.26729v1)

**作者:** Rupesh Sapkota `[一作]` (Paderborn University), Axel-Cyrille Ngonga Ngomo `[通讯]` (Paderborn University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于门控残差神经回归模型（Literal Embedding）以及联合训练框架，利用预训练的知识图谱嵌入实现数值属性的预测与补全，同时保持原KGE模型的得分函数不变。

**💡 创新点**

创新点在于：①无需改动KGE得分函数即可预测数值属性；②引入门控残差机制和梯度动态加权的联合训练，使模型真正“字面感知”（literal‑aware）并提升链接预测性能；③首次将DB15K和Mutagenesis改造为数值属性预测基准。

**🔧 技术方法**

技术手段包括：门控残差网络、动态权重λ、联合训练（co‑training）与MAE/BCE损失、TransE/RotatE等传统KGE模型作为基线。

**📊 数据集**

使用的评估数据集为FB15K‑237、YAGO15K、DB15K和Mutagenesis四个知识图谱，所有数据均采用预训练的64维实体/关系嵌入，并对DB15K和Mutagenesis重新划分属性预测测试集。

**📈 对比分析**

通过五次独立实验，用MAE评估数值属性预测，在FB15K‑237（8/11属性）和YAGO15K（7/7属性）均取得最优或次优结果；在链接预测上，对乘积型模型（如DistMult、ComplEx、Keci、OMult、QMult）提升显著，提升量可达数个百分点，且在多种评估指标（MRR、Hits@1/3/10）上均优于基线。

**⚠️ 局限性**

局限性包括：仅处理数值属性，未考虑布尔、文本、图像等其他文字；实验规模受限于64维嵌入和约1.5万实体，未验证更大图或更高维时的表现；并且联合训练仅更新实体嵌入，未同步优化关系嵌入。

---

## 198. Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG

**arXiv ID:** 2608.26379 | [PDF](https://arxiv.org/pdf/2608.26379v1)

**作者:** Utshab Kumar Ghosh `[一作]` (Missouri University of Science and Technology), Shubham Chatterjee `[通讯]` (Missouri University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在检索增强生成（RAG）中使用答案支持（answer‑support）评价信号的全链路价值，包括检索比较、训练、系统选择、预测性能以及直接的证据干预；

**💡 创新点**

创新点在于首次将答案支持评估沿着整个RAG管道进行纵向追踪，揭示了“有效性-组合（validity‑composition）”问题，即评价信号对一个环节有效并不必然转化为下游决策的可靠性；

**🔧 技术方法**

使用的技术包括基于GPT‑4.1/Qwen/Llama的答案支持判断、Gemma3‑27B生成器、RankNet检索训练、nDCG/Answer‑Nuggetizer评价、以及匹配随机对照的证据过滤实验；

**📊 数据集**

实验数据集涵盖BEIR（TREC‑COVID、NFCorpus、SciFact）、TREC‑DL 2019/2020以及官方TREC RAG 2025检索任务；

**📈 对比分析**

比较方法：通过将检索结果在原始相关性评估与答案支持评估下重新排序，评估系统排名变化、训练效果、系统选择对held‑out答案质量的影响，以及检索分数对未见主题答案质量的预测。结果显示：答案支持可显著改变排名，但在训练中提升有限；在Coverage‑Disciplined生成策略下，答案支持辅助的系统选择能提升答案质量，而在Standard Grounded下则无显著效益；检索分数对未见主题答案质量的预测不稳健；而直接证据过滤在Qwen评估下略有提升，但在Claude评估下无显著变化；

**⚠️ 局限性**

局限性包括：答案支持评估在不同评判者间存在显著差异，导致结论依赖评估者；检索分数对答案质量的预测性差；证据过滤未完全捕捉证据集合的多样性与互补性；实验聚焦于特定生成器和评估工具，结果可能不具普适性。

---

## 199. Packora: Systematic Design for Generative Molecular Crystal Structure Prediction

**arXiv ID:** 2608.26962 | [PDF](https://arxiv.org/pdf/2608.26962v1)

**作者:** Nayoung Kim `[一作]` (Korea Advanced Institute of Science and Technology), Sungsoo Ahn `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并训练了一个基于变分流匹配的全原子生成模型，用于分子晶体结构预测，支持多组分、有机金属晶体，并能够根据分子模板、立体化学标签和空间群等可选条件进行条件生成。

**💡 创新点**

创新点包括：①统一模型框架可灵活接收或忽略多种条件，满足不同信息可用性的需求；②使用Pairmixer+Diffusion Transformer结构实现高效、可缓存的对称性建模；③引入分离生成与排名的两轨评估，清晰区分生成器覆盖率与后端排名效果；④在六个基准上系统地探究架构、训练、条件、推理和扩展性五大维度，给出实用的设计方案。

**🔧 技术方法**

核心技术包括变分流匹配（VFM）用于端点回归、Pairmixer 作为对称性处理模块、Diffusion Transformer（DiT）进行坐标与晶格的联合建模、L1 损失、EDM–Heun 采样器以及自动引导等。

**📊 数据集**

使用六个公开基准数据集（涵盖单晶、多晶、共晶等多种化学体系），其中部分基准源自 CCDC CSP blind test，候选数为 30 或 1,000，覆盖不同分子与空间群组合。

**📈 对比分析**

与 OXtal、CLARI‑H、FastCSP 等现有方法相比，在生成轨道上实现了最高的匹配预算覆盖率；在排名轨道上实验结构恢复率更高、排名更靠前、收敛更快，表现优于基准模型。

**⚠️ 局限性**

限制：①实验中每个维度的 ablation 仅单独变更，未考虑多维度交互；②评估主要基于 30 份候选，模型在更大候选预算下的表现尚不清楚；③模型的规模选择需要在小预算成功率与大预算覆盖率之间权衡，需进一步研究最佳容量分配。

---

## 200. DEEPCHART: How Far are LLMs from Faithful Data-Science Chart Generation?

**arXiv ID:** 2608.26757 | [PDF](https://arxiv.org/pdf/2608.26757v1)

**作者:** Jiahui tang `[一作]`, Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

做了什么

**💡 创新点**

创新点是什么

**🔧 技术方法**

用了什么技术

**📊 数据集**

用了什么数据集

**📈 对比分析**

如何比较的方法，性能怎么样

**⚠️ 局限性**

limitation是什么

---

## 201. GeoMAD: Geometry-Aware Multi-View Anomaly Detection via Deformable Fusion and Distributional Alignment

**arXiv ID:** 2608.26724 | [PDF](https://arxiv.org/pdf/2608.26724v1)

**作者:** Shang-Fu Chen `[一作]` (National Taiwan University), Kai-Lung Hua `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的多视角多类别异常检测框架，利用自监督的逆向蒸馏实现对多视角特征的融合与一致性约束。

**💡 创新点**

核心创新在于（1）跨视角可变形融合模块（CDFM），通过学习视角对特定的采样偏移在二维特征图上实现几何对应；（2）分布视角对齐（DVA），用每个实例的视角均值作为对齐目标，以KL散度强制视角特征分布全局一致，无需像素级匹配或三维投影。

**🔧 技术方法**

采用预训练 ResNet‑34 作为教师/学生网络，结合逆向蒸馏、二维可变形注意力、窗口金字塔、平均池化平滑以及分布对齐正则化。

**📊 数据集**

在 Real‑IAD（30 类工业缺陷图像）和 MANTA‑Tiny（38 类小物体多视角图像）两个公开数据集上进行评估。

**📈 对比分析**

与当前最先进方法（MVAD、RD4AD、UniAD、ViTAD、MambaAD 等）比较，实验显示在样本、图像和像素层面均取得最高或第二高的 AUROC、AP、F1 等指标，尤其在更具挑战性的 MANTA‑Tiny 上表现突出。

**⚠️ 局限性**

局限性包括：仅适用于固定的视角组合，无法处理动态视角或缺失视角时的显著性能下降；以及对三维几何约束的间接性，仍可能在极端几何变化下存在误差。

---

## 202. HOLMES: In-Context Failure-Center Localization for High-Dimensional Yield Estimation

**arXiv ID:** 2608.26758 | [PDF](https://arxiv.org/pdf/2608.26758v1)

**作者:** Wei W. Xing `[一作]` (University of Sheffield), Shan Shen `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HOLMES 方法，用几-shot 二分类定位 SRAM 失败中心，实现高维高 sigma yield 估计。

**💡 创新点**

创新点在于将失败中心定位视为少量样本二分类，采用无梯度的 TabPFN 上下文推理，并结合 SVD 主导的异方差提案与 hit‑rate 自适应混合。

**🔧 技术方法**

使用技术包括 TabPFN、SVD 主成分分析、重要性采样、hit‑rate 适配混合、重要性加权 MI 进行特征选择等。

**📊 数据集**

使用 OpenYield SRAM 6T bit‑cell 测试集，覆盖 3×2、8×4、7×6、8×6、8×8 等 FreePDK45 设计，对应维度 108~1152。

**📈 对比分析**

与 8 种 IS 基线和 Monte‑Carlo 进行对比，HOLMES 在所有维度下相对误差≤5.9%，速度提升约 19–20 倍，基线在高维会出现 25%+误差或不收敛。

**⚠️ 局限性**

局限性在于仅适用于单一主失败区，缺乏多中心定位能力，并对 TabPFN 在极高维的泛化性有一定依赖。

---

## 203. Arbitrary-Order Hermite Interpolation of Rigid-Motion Jets via Hyper-Multidual Quaternions

**arXiv ID:** 2608.27000 | [PDF](https://arxiv.org/pdf/2608.27000v1)

**作者:** Daniel Condurache `[一作]` `[通讯]` (Technical University of Iasi), Daniel Condurache (Technical University of Iasi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在单位超多元四元数（HMD四元数）群上进行双向插值，提出并分析了直接ScLERP方法的非齐次性问题，并给出了利用Hermite多项式实现的可实现的HMD-Hermite-ScLERP插值；

**💡 创新点**

1）证明直接HMD‑ScLERP在一般端点Jet上不可满足齐次性；2）给出端点系数的一阶、二阶等递推约束；3）提出通过对端点对数坐标做Hermite插值再指数映射，从而实现任意阶端点Jet的齐次插值；4）实现了高阶加速度场的直接恢复，无需展开Lie括号；

**🔧 技术方法**

多元数与超多元数代数、HMD四元数算术、有限多项式截断乘积、可微分延伸（多元数延拓）、对数坐标与指数映射、Hermite多项式插值、对数坐标的非线性微分（Frechet导数）等；

**📊 数据集**

采用人工构造的旋转子组测试（q₀=1,q₁=exp(0.6k)）和螺旋运动测试（q₀=1,q₁=cos0.3+sin0.3k+ε₀…）以及对应的一阶、二阶端点Jet数据；

**📈 对比分析**

与直接HMD‑ScLERP（产生非齐次性缺陷）对比，HMD‑Hermite‑ScLERP在端点处完全匹配，且内部任意时刻满足齐次性，误差仅受浮点近似。数值实验显示，直接方法在内部出现显著的接触缺陷，而Hermite方法接触缺陷为零；性能方面，算法复杂度为O(n²)（n为Jet阶数），数值条件数随阶数增大显著上升；

**⚠️ 局限性**

1）仅在选定的对数图上局部有效，跨越π角需切分；2）不保持常数螺旋轴或螺距不变；3）计算成本与存储随阶数升高显著；4）不具备几何双不变性，只满足常数左右等变性；5）对大阶数的数值稳定性有限。

---

## 204. Characterizing the Landscape of Open-Source Satellite Software

**arXiv ID:** 2608.26211 | [PDF](https://arxiv.org/pdf/2608.26211v1)

**作者:** Jinfeng Wen `[一作]` (Beijing University of Posts and Telecommunications), Shangguang Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文基于GitHub 22,286个开源卫星软件项目，系统分析其流行趋势、功能目标与开发实践，并构建了43类目标分类体系。

**💡 创新点**

首次全景性描述卫星开源生态，提出基于随机抽样的目标分类方法，并揭示了66种编程语言与任务特定实现策略，弥补了以往缺乏系统性实证研究的空白。

**🔧 技术方法**

采用GitHub REST API抓取、时间序列分析、手工编码生成分类法、统计语言使用、代码审计和互评一致性评估（Cohen's Kappa）等技术手段。

**📊 数据集**

使用了22,286个卫星相关仓库的元数据与源代码，随机抽样646个项目进行目标分类，进一步抽样620个项目用于代码与语言使用分析。

**📈 对比分析**

通过年度新建项目数、活跃开发者数、提交次数等指标评估流行度；对语言和实现做分组统计，未做传统算法性能对比，结果表明卫星软件生态多语言、多任务异构、发展活跃。

**⚠️ 局限性**

研究局限包括仅采集GitHub且仅限英文仓库，手工分类存在主观性，目标分类覆盖度有限，未覆盖完整卫星软件生命周期的全部实践。

---

## 205. Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs

**arXiv ID:** 2608.26641 | [PDF](https://arxiv.org/pdf/2608.26641v1)

**作者:** Xingyou Fang `[一作]` (Fuzhou University), Xiaofeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的解码策略 IGFD，用于改进 diffusion 多模态大语言模型的提交顺序；

**💡 创新点**

创新点在于将 token 置信度、邻域熵不确定性与结构提交风险结合成综合得分，并通过动态候选前沿控制，优先提交语义锚点并延迟结构化 token，显著提升上下文支持；

**🔧 技术方法**

采用信息导向前沿解码、基于熵的邻域不确定性评估、结构风险惩罚、动态前沿维护以及单次 forward pass 等技术；

**📊 数据集**

使用 LLaVA-Bench、CHAIR、MathVista、ScienceQA、MME、GQA 等多模态基准数据集，以及 WikiText 进行 BERTScore 评估；

**📈 对比分析**

与 Original、AdaBlock、Wavefront 等基线相比，IGFD 在多模态生成、幻觉抑制、推理、感知和 grounding 等指标上总体表现更好，尤其在幻觉分数和 BERTScore 上取得显著提升；

**⚠️ 局限性**

局限性包括：使用局部熵估计可能无法捕捉长程依赖；结构惩罚基于 tokenizer 规则，需要针对不同模型微调；仍未解决全局上下文和跨模态一致性问题。

---

## 206. Modality Maturity Index: A benchmark for assessing multimodal capabilities of omni models

**arXiv ID:** 2608.26317 | [PDF](https://arxiv.org/pdf/2608.26317v1)

**作者:** Rohit Patel `[一作]` (Meta Superintelligence Labs), Sloan Strader `[通讯]` (Meta Superintelligence Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布了MMI（Multimodal Maturity Index）基准，用以评估大型语言模型在跨模态输入与输出中的正确性与完整性。

**💡 创新点**

创新点在于：①首次设计要求模型在提示中自动选择并返回所需的多种输出模态；②为每个期望输出模态编写人类撰写的细粒度评分 Rubric；③引入宏观平均 MMI 以及仅关注模态出现的 MMI‑P 进行分层评估。

**🔧 技术方法**

主要技术包括：利用 LLM 评判器自动根据 Rubric 进行分级；通过正则表达式、URL 识别以及 LLM 辅助判定三种方式检测输出模态；在工具辅助实验中给模型提供音频、图像、视频生成工具以提升覆盖率。

**📊 数据集**

数据集为自定义的多模态提示集合，覆盖文本、图像、音频、视频与文档共 5 种模态，包含 1,099（提示，模态）组，约 152 文档文件；所有资产由 MetaAI 或团队成员生成并公开发布。

**📈 对比分析**

实验中对 GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro/Flash、Llama 4 Maverick 进行评测；MMI‑P（宏观 F1）最高仅 34.9（GPT‑5.4），最低 15.6（Claude Opus）；召回率普遍低于 35，精确率接近 100，表明模型多为文本输出，缺乏真正跨模态交互能力。

**⚠️ 局限性**

局限性包括：①模型生成的模态过少，导致主评估指标 MMI 无法充分体现真实性；②仅在 API 级别评估，未覆盖完整产品或工具链；③Rubric 评价主观性较强，LLM 评判者与人工评判者一致率仅 70.8%，表明判定标准仍需改进。

---

## 207. Decoupling Planning and Control for Instructable Agents

**arXiv ID:** 2608.26788 | [PDF](https://arxiv.org/pdf/2608.26788v1)

**作者:** Zineng Tang `[一作]` (University Of California Berkeley), Alane Suhr `[通讯]` (University Of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可拆分规划与控制的框架：用预训练的视觉‑语言模型(VLM)生成高层文本指令，指令由轻量级世界模型控制器按实时低延迟方式执行，从而在单体和多体环境中实现长时序动作控制。

**💡 创新点**

创新点：
• 通过异步、解耦的规划‑控制结构，让VLM能够在后台持续推理，而控制器以高频率直接与环境交互。
• 采用后期指令标注(post‑hoc instruction annotation)在训练时用VLM自动生成指令，既省去专家演示，又实现语言条件化。
• 设计统一的语言接口，使不同的VLM可无缝替换，支持多体语言协作与实时通信。
• 在多种单体与多体任务上证明该结构相较于端到端VLM、传统世界模型和多体强化学习基线具有更高的任务得分和吞吐量。

**🔧 技术方法**

核心技术：
• 语言感知的可重复状态空间模型(RSSM)/DreamerV3 控制器。
• CLIP‑Base‑224、DINOv2‑Base 视觉编码器；MiniLM‑L6‑H384‑uncased 语言编码器。
• 语言‑条件化行为克隆（BC）与世界模型奖励损失的联合优化。
• 异步在线规划（Planner）与控制器并行执行；多体规划通过共享控制器参数实现。
• 指令标注使用 GPT‑4o/其他VLM 生成高层指令，并映射到控制器的嵌入空间。

**📊 数据集**

使用七个标准化环境：
• 单体：Atari、MineRL ObtainDiamond、Crafter、DMLab、Overcooked（单体版）。
• 多体：Pico Park（2‑8 代理）与 Overcooked（交叉‑玩耍）。
• 额外评估：MindCraft（烹饪/构建任务）。

**📈 对比分析**

比较方法：
• 复现基线：DreamerV3、MAPPO、QMIX。
• 领域适配基线：RT‑2。
• 上下文特定基线：Voyager、DEPS、JARVIS‑1 等。
• 结果显示：
   - 在单体任务中，Planner+Controller 方案平均提升 5‑10% 以上，相比直接 VLM 行动或仅控制器都更优。
   - 在多体任务中，解耦式规划与共享控制器实现了与传统多体强化学习相当甚至更优的得分，并保持了高吞吐量。
   - GPT‑4o 或 Qwen‑VL‑2.5‑72B 作为 Planner 时获得最优表现，表现优于专门调优的 JARVIS‑1。
   - 控制器一次训练成本约 23 GPU‑h，且与 Planner 迁移性良好。

**⚠️ 局限性**

限制与挑战：
• 训练需为每个环境单独构建并训练控制器，尚未实现跨环境的通用控制器。
• 依赖 VLM 的指令生成质量；低质量或随机指令导致控制器性能显著下降。
• 指令标注需要额外算力（约 17% GPU‑h），对大规模部署有一定负担。
• 控制器在极端高频或复杂动作空间时仍可能面临精度与延迟折衷。
• 多体规划时的消息同步与冲突仍需要更精细的协议优化。

---

## 208. The Reasoning Tax: Token Economics of LLM Reasoning Across Task Types and Deployment Contexts

**arXiv ID:** 2608.26235 | [PDF](https://arxiv.org/pdf/2608.26235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 209. PRISM: Lightweight Enclave Isolation with Prismatic Capabilities

**arXiv ID:** 2608.26367 | [PDF](https://arxiv.org/pdf/2608.26367v1)

**作者:** Merve Gülmez `[一作]` (Ericsson Security Research), Thomas Nyman `[通讯]` (Ericsson Product Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 CHERI‑RISC‑V 上设计并实现了“prismatic capabilities”，实现了轻量级、用户空间的可信执行环境（TEE），支持远程证明并能直接运行未改动的 Intel SGX 应用。

**💡 创新点**

核心创新点：1) 在能力字段中携带唯一的“hue”标识，实现对物理内存的无扫描、O(1) 独占；2) 通过硬件 hue‑addressed table 和 M‑mode monitor（hmm）实现无陷阱域切换；3) 在同一架构下提供完整的硬件远程证明链；4) 在 CHERI 上实现完整 RTL、QEMU 模拟与 FPGA 原型，并提供 SGX 兼容层。

**🔧 技术方法**

使用技术：CHERI capability 扩展（prismatic capability）、hue‑addressed table、M‑mode security monitor（hmm）、CheriBSD kernel driver、caprelocs、SGX‑兼容库（trts/urts）、Tamarin 形式化验证、CHERI‑Toooba FPGA softcore、SPEC CPU 2006 benchmark、SGX 应用（kmeans、SQLite、CryptoEnclave）和 NIST Juliet 用于时空安全测试。

**📊 数据集**

评测数据集：SPEC CPU 2006（2006 benchmark suite），三款真实 SGX 应用（sgx‑kmeans、SGX_SQLite、CryptoEnclave），NIST Juliet 用于 use‑after‑free 检测。

**📈 对比分析**

比较方法与性能：与 CHERI‑TrEE、Capstone 以及 Intel SGX 进行基准对比；SPEC 结果显示对 CHERI‑TrEE 仅 1.06× 的几何平均开销；SGX 应用在 prismatic 上的开销 1.06–1.15×，显著低于 SGX 的 1.67–1.81×；FPGA 资源占用约 16% 逻辑；创建/销毁时间分别约 15–17 秒 / 1–2 秒；整体运行时开销 ≤15%。

**⚠️ 局限性**

局限性：1) 目前仅单核实现，系统级扫描需停机；2) hue 空间有限（默认 63 个），需要全局扫描回收；3) enclave 内存不可动态扩展（单块物理区域），需预留最大空间；4) 未针对微架构侧信道（Transient‑Execution、Spectre/Meltdown 等）做硬件防护；5) 需要对现有 SGX 应用进行少量代码改动以兼容 CHERI。

---

## 210. ProofEvolve: Neuro-Symbolic Evolution for Formal Automated Theorem Proving

**arXiv ID:** 2608.26334 | [PDF](https://arxiv.org/pdf/2608.26334v1)

**作者:** Wenqian Ye `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ProofEvolve，一种神经-符号进化框架，用 Lean 4 内核验证的证明 DAG 进行自动定理证明，并在跨目标间复用已验证的子证明结构。

**💡 创新点**

创新点包括：①用核验证的“verified closure”作为多阶梯评分，取代单纯的通过/失败反馈；②通过“schema recombination”将已验证的子证明以类型安全的方式迁移到新的证明 DAG；③在进化搜索中维护结构多样性的存档并结合核验证的选择，形成可持续累积的符号知识库。

**🔧 技术方法**

核心技术：神经语言模型（Claude Opus 4.8 等）用于生成证明步骤，Lean 4 kernel 负责每一步的类型检查与证明检验；使用 AND‑OR DAG 记录部分证明；实现 verified closure、结构多样性存档、schema 提取与重用、前沿调度等算法；对搜索过程进行理论证明（安全性、闭合性）。

**📊 数据集**

数据集：PutnamBench、IMO‑LeanProofBench、CombiBench（三大竞赛级 Lean 定理集合），以及 744 个来自 Lean Workbook 的未见定理，用于评估知识库复用效果；同时使用 9,968 条已验证的 Lean 定理构建自我学习的库。

**📈 对比分析**

方法对比：与 5 种 agentic 系统（LEAP、Hilbert、AxProver、Aristotle、ReAct）以及 LLM 单纯采样（pass@16）进行公平基准。实验结果显示 ProofEvolve 在平均 solve rate 上达到 57.8%，比 LEAP 的 50.5% 高 7.3 点，最显著提升出现在 PutnamBench（71.2%）和 IMO‑LeanProofBench（53.3%）。在 744 Lean Workbook 的测试中，利用自身已验证的库提升 solve rate 约 4 个百分点，随机检索无提升。

**⚠️ 局限性**

局限性：①对 Lean 4 环境与 Mathlib 的依赖，难以直接迁移到其他形式系统；②高昂的计算资源需求（多 GPU/多节点），不适合低成本部署；③神经模型保持不变，无法自适应新的证明风格；④进化搜索仍受限于有限的搜索预算和缺乏真正的自适应奖励信号；⑤目前未针对大规模开放式科学发现（如物理、化学）做验证，仍为形式化证明领域的探索。

---

## 211. Cross-lingual Representation Learning via Centroid Intervention Fusion

**arXiv ID:** 2608.26357 | [PDF](https://arxiv.org/pdf/2608.26357v1)

**作者:** Wei Sun `[一作]` (KU Leuven), Marie-Francine Moens `[通讯]` (KU Leuven)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于推理时干预的跨语言干预框架——Centroid Intervention Fusion (CIF)，通过融合多语言投影矩阵构建共享干预算子，实现跨语言干预；

**💡 创新点**

将原本针对语言对的独立投影矩阵通过中心化融合与中心导向去裁剪聚合，构成单一语言共享干预算子，解决pairwise方法的可扩展性和知识共享瓶颈；

**🔧 技术方法**

使用推理时干预、投影矩阵学习（最小二乘拟合）、残差扰动表示、中心化融合、裁剪聚合等技术；

**📊 数据集**

在多语言推理/理解任务（XCOPA、XStoryCloze、XWinograd、XCSR、XNLI）和生成任务（MzsRE、WMT23、FLORES‑101）上，结合四大LLM家族（Qwen、BLOOMZ、LLaMA、Mistral）进行评估；

**📈 对比分析**

与零样本、翻译-后端、ITI、CAA、SADI、INCLINE等六个干预基线对比，CIF在四个模型家族上平均提升约+3.38pp，单项最高提升+3.38pp，整体宏平均得分51.77%，显著优于pairwise基线及其他干预方法；

**⚠️ 局限性**

仅在文本多语言任务验证，未检验多模态或交互场景；使用固定干预强度与裁剪参数，缺乏自适应策略；在更大语言集合或更复杂任务中的可扩展性仍待研究。

---

## 212. Online Joint Calibration of Steering Offset and Planar LiDAR Extrinsics for Wheeled Mobile Robots

**arXiv ID:** 2608.26789 | [PDF](https://arxiv.org/pdf/2608.26789v1)

**作者:** Subodh Mishra `[一作]` (ATI Motors), Naveen Arulselvan `[通讯]` (ATI Motors)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在线 EKF 方法，用于同时估计车辆转向偏差和平面 LiDAR 外参，并通过校准显著提升移动机器人路径跟踪精度。

**💡 创新点**

创新点在于将运动基准校准约束嵌入 EKF，实现无需外部参照或预定轨迹的实时联合校准，保持对转向误差和 LiDAR 变位的可观测性。

**🔧 技术方法**

采用扩展卡尔曼滤波、平面运动基准约束、LiDAR 里程计、基于标准平面车轮运动模型的状态更新。

**📊 数据集**

使用两台工业 WMR（Tugger 与 Nano）在不同负载、频繁重装、人工引入转向偏差等条件下收集的实测数据。

**📈 对比分析**

通过对比未校准、手动校准与 EKF 校准三种情况，评估 CTE RMSE、轨迹误差与里程计一致性；校准后 CTE 下降约10–20%，轨迹误差从数米降至 1–2 米，验证了方法有效性。

**⚠️ 局限性**

局限性包括：需要足够运动激励保证可观测性；在 LiDAR 里程计受限的稀疏结构环境下性能下降；目前仅在离线实验中验证，尚未在闭环控制中完成全流程部署。

---

## 213. Weight Distributions of Single Parity-Check Product Codes via Character Sums

**arXiv ID:** 2608.26457 | [PDF](https://arxiv.org/pdf/2608.26457v1)

**作者:** Makson Miller Alves Ribeiro `[一作]` (Secretária de Educação do Estado de São Paulo), Sara D. Cardell `[通讯]` (São Paulo State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

对二进制单奇偶校验码的二维乘积码进行结构和枚举研究，推导其权分布、通用汉明重量层级及最大权重性质。

**💡 创新点**

首次给出该乘积码的完全权分布闭式表达式，并通过Walsh–Hadamard变换和Krawtchouk多项式实现高效计算。

**🔧 技术方法**

使用MacWilliams恒等式的Walsh–Hadamard形式、字符求和、通用汉明重量公式以及Krawtchouk多项式递推。

**📊 数据集**

无外部数据集，采用理论构造并在 n=3、4、10 等实例中做数值验证。

**📈 对比分析**

将所得分布与有限长度代码的完整枚举结果进行对比，验证对称性和计数正确；算法复杂度为 O(N²)（N=mn），远优于 2^(m-1)(n-1) 的暴力枚举。

**⚠️ 局限性**

局限在于随 n 增大时计算量仍随 N² 迅速增长；对更高维度乘积码的推广及更广泛组件码的适用性尚未解决。

---

## 214. A Unified Framework for the Mechanics of Information in Convolutional Neural Network Image Space

**arXiv ID:** 2608.26363 | [PDF](https://arxiv.org/pdf/2608.26363v1)

**作者:** Aryan Shukla `[一作]` (École de technologie supérieure), Matthew Toews `[通讯]` (École de technologie supérieure)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了卷积神经网络（CNN）中使用ReLU非线性卷积对图像信息的传播机制，并将其与相对论能量-动量关系相类比，揭示了对称/反对称滤波器在信息空间中的扩散、振动和位移行为。

**💡 创新点**

创新点在于将对称（Σ=[1,1]）与反对称（∇=[-1,1]）滤波器映射为相对论中的静质量与动量，从而得到“洛伦兹‑类”位移规律；同时通过多尺度高斯平滑与SIFT关键点检测，展现了从原子到星系尺度的Morse拓扑结构。

**🔧 技术方法**

使用的技术包括：离散卷积与ReLU非线性；离散余弦变换（DCT）分析滤波器的对称/反对称分量；多尺度高斯滤波（Gaussian scale‑space）；3D SIFT（尺度不变特征变换）检测关键点；以及对比实验（标准线性卷积 vs 带ReLU卷积）。

**📊 数据集**

实验数据集涵盖：1）单像素冲击及2D圆形测试图像；2）3D电子密度图（糖分子、硅晶体）；3）原始MRI脑部扫描（猴子、黑猩猩、人类）；4）银河模拟与宇宙微波背景（CMB）投影。没有使用公开大规模训练集，仅用于演示理论与可视化。

**📈 对比分析**

对比方法主要是观察中心质量（CoM）随滤波次数的偏移以及标准差变化。结果显示：标准线性卷积在β>0.5时出现正负值交替导致位移归零；而带ReLU的卷积在所有β下呈现洛伦兹曲线式位移，且与理论预测吻合；在多尺度SIFT实验中，关键点的出现与物理结构（原子位置、脑区白质、星系聚集区）高度一致，说明拓扑检测有效。没有给出数值性能指标，只展示了定性匹配与可视化一致性。

**⚠️ 局限性**

局限性包括：仅探讨了最基本的Σ和∇两种滤波器，未扩展至其他DCT模式；DCT梯度的最大位移受滤波器尺寸限制，且对大尺寸滤波器效果不明确；使用的方形滤波器对圆形或球形结构的逼近不够精确；理论仅为类比，缺乏严格的物理证明；实验范围有限，缺乏大规模数据集上的定量验证。

---

## 215. Faster FPRAS for the Permanent via Restricted Poincaré Inequalities and Coupled Flows

**arXiv ID:** 2608.26599 | [PDF](https://arxiv.org/pdf/2608.26599v1)

**作者:** Xiaoyu Chen `[一作]` (Massachusetts Institute of Technology), Xiongxin Yang `[通讯]` (University of California Santa Barbara)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对 0/1 矩阵的行列式（permanent）设计了一种新的近似多项式时间随机算法（FPRAS），将此前的运行时间从 O(n⁷log⁴n) 降至 O(n⁶log⁵n)。

**💡 创新点**

创新点在于：
• 引入了针对“空洞模式”划分的限制 Poincaré 不等式，证明其收敛时间仅为 O(n³)，比全局收敛时间 O(n⁴) 有 1/n 的提升；
• 设计了耦合多路流（coupled multicommodity flow）技术，利用划分后的块级流量优化，进一步降低负载；
• 通过这些技术在权重细化（weight‑refinement）步骤显著加速了 Markov 链的采样过程。

**🔧 技术方法**

采用的技术包括：
• Markov Chain Monte Carlo（MCMC）与模拟退火框架（Jerrum‑Sinclair‑Vigoda 算法的改进）；
• Poincaré 不等式与限制 Poincaré 不等式的谱分析；
• 多路流（multicommodity flow）与耦合多路流构造；
• 统计学中的方差分解、Chebyshev 与 Hoeffding 结论用于采样误差分析。

**📊 数据集**

本研究为理论性工作，未使用具体数据集；所有结论均基于严谨的数学证明与抽象图结构。

**📈 对比分析**

与原始 JSV‑Vigoda 算法的比较：
• 原先的权重细化步骤需要 O(n⁵log⁴n) 的采样长度；
• 本文通过限制 Poincaré 收敛时间改进，将权重细化步骤的采样长度降低到 O(n⁴log⁴n)；
• 整体运行时间由 O(n⁷log⁴n) 下降到 O(n⁶log⁵n)，实现了渐进式的显著加速。

**⚠️ 局限性**

局限性包括：
• 仍为多项式时间，实际运行成本对大规模 n 仍较高；
• 依赖于对理想权重的近似，若权重估计误差较大，算法鲁棒性受限；
• 该改进主要针对 0/1 矩阵，虽然可推广到非负矩阵，但在特定结构矩阵上可能还有进一步优化空间。

---

## 216. Standalone LLM and a Pre-specified Agentic Pipeline for Explaining ICU Mortality Predictions: a Feasibility Study on the eICU Demo Dataset

**arXiv ID:** 2608.26109 | [PDF](https://arxiv.org/pdf/2608.26109v1)

**作者:** Di Zhu `[一作]` (Santa Clara University), Qiyang Xie `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过构建基于XGBoost的ICU死亡率预测模型，并比较单一LLM提示与四步拆解式Agentic流程对模型解释的质量，探讨两种自然语言解释策略在临床可用性与安全性方面的差异。

**💡 创新点**

创新点在于提出并验证了任务拆解式Agentic管道，能够消除结果泄漏并提升指南依据与患者特异性信息的呈现，同时对比其与传统单一LLM提示在SHAP对齐、方向一致性与可解释性方面的权衡。

**🔧 技术方法**

技术包括XGBoost与逻辑回归预测模型、SHAP特征重要性分析、单一LLM提示生成、四步Agentic流程（数据解读→指南检查→鉴别推理→最终合成）以及基于文本的泄漏审计与多维解释评估指标。

**📊 数据集**

使用eICU Collaborative Research Database Demo v2.0.1，筛选2,353例成人ICU停留记录（死亡率8.1%），并基于首24小时的结构化临床指标进行建模与解释。

**📈 对比分析**

在保留的测试集上，XGBoost实现AUROC 0.855、AUPRC 0.332；在14例重叠评估子集中，单一LLM在SHAP对齐（0.171）和方向一致性（92.9%）上优于Agentic（0.077、78.6%），但Agentic在指南依据（0.762）和值特异性（0.236）以及可解释性（0.700）上表现更好，且未出现结果泄漏。

**⚠️ 局限性**

主要局限包括样本量相对较小、仅有14例与SHAP评估集重叠导致置信区间宽泛、仅使用单一本地LLM模型、解释评估指标为代理量且缺乏临床专家的前瞻性验证、以及数据清洗后缺失值较多，可能影响解释可靠性。

---

## 217. Finding the Right Evidence: Factor-Guided Coarse-to-Fine Reasoning for Long Videos

**arXiv ID:** 2608.26355 | [PDF](https://arxiv.org/pdf/2608.26355v1)

**作者:** Baixuan Xu `[一作]` (HKUST), Yangqiu Song `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PACE 框架，采用分两阶段、因子引导的证据检索方法，解决长视频问答中的选项盲检索瓶颈。

**💡 创新点**

核心创新是写无选项、读有选项的非对称检索与验证流程，通过问题因子构建索引，再用选项对比因子查询，以提升决策性证据恢复。

**🔧 技术方法**

结合大型视觉语言模型（如 Qwen3‑VL）、因子提取、文本描述索引、工具驱动的多步推理代理以及对比因子检索等技术。

**📊 数据集**

在 MMR‑V 基准上验证，并迁移到 LVBench、Video‑MME、EgoSchema、LongVideoBench 等长视频问答数据集。

**📈 对比分析**

与基线及对抗性基准（Deep Video Discovery、VideoTree 等）比较，PACE 在 MMR‑V 的准确率提升约 3.1% 以上，100 题诊断的针针召回率升至 66.9%，并在四大基准上均取得小幅提升。

**⚠️ 局限性**

局限性包括高计算开销与推理延迟、对底层 VLM 感知能力依赖、对抽象语义或主题推理的提升有限，以及仅在多选场景验证，开放式问答的适用性尚未探究。

---

## 218. Gromov-Monge Flow Matching for Equivariant Graph Generation

**arXiv ID:** 2608.26961 | [PDF](https://arxiv.org/pdf/2608.26961v1)

**作者:** Moritz Piening `[一作]` (Technische Universität Berlin), Christian Wald `[通讯]` (INSA Lyon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在图生成的流匹配框架中加入了基于 Gromov‑Wasserstein 的节点对齐（内层）和批次外层对齐，从而使得源‑目标之间的耦合在图的等价类上保持一致，显著提升少步 Euler 采样的样本质量。

**💡 创新点**

创新点在于：①提出在流匹配中对图的节点重标记进行 Gromov‑Monge 对齐，并给出对称化耦合保持最优性的理论证明；②在批次层面使用 Gromov‑Wasserstein 或其低成本下界实现全局最优匹配；③将上述对齐策略与现有的欧氏流匹配、条件端点分类模型无缝结合。

**🔧 技术方法**

技术包括：Gromov‑Wasserstein (GW) 与其 Frank‑Wolfe 近似、Hungarian 算法求硬对齐、欧氏流匹配、条件端点分类、Fused‑GW 最近邻检验、Minibatch OT、Euler 积分、Permutation‑equivariant 图变换器。

**📊 数据集**

实验数据集：连续目标——10 节点的 Stochastic Block Models（K=1…5）；分类目标——QM9（N≤9）与 ZINC250k（N≤38）；以及周期图实验。

**📈 对比分析**

与 Random、FLB、GW、GW+外层、MinibatchOT 等方案比较。对连续目标，在 5 步时 GW+外层使 FGW‑NNA 从 0.796 降至 0.568，MMD 等指标也显著下降；对分类目标，GW 内层将有效率从 0.885 提升到 0.936，FCD 从 1.73 降至 1.28；在 500 步全预算实验中 GW‑CatFlow 与现有最先进方法相当（有效率 99% 以上，FCD 0.11‑0.96）。

**⚠️ 局限性**

局限性在于：Gromov‑Wasserstein 求解仍然昂贵，且只能得到 Gromov‑Monge 的近似；仅在训练阶段使用，推理阶段无此开销；对于更大规模图或更高维特征的情况，计算可扩展性仍有待改进。

---

## 219. Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses

**arXiv ID:** 2608.26130 | [PDF](https://arxiv.org/pdf/2608.26130v1)

**作者:** Tatiana Petrova `[一作]` (University of Luxembourg), Radu State `[通讯]` (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大型语言模型驱动的编码代理中，对工具响应进行首块选择（压缩）以满足每轮 token 预算，评估了不同评分函数对首位精准度（p1）及后续 LLM 任务准确率的影响。

**💡 创新点**

发现提升 p1 并不必然提升下游 LLM 准确率，且单一关键词匹配评分优于多信号组合；同时证明生产环境中代理从不发起分页请求，首块包含目标文件的排名位置并非预测 LLM 成功率的关键。

**🔧 技术方法**

采用 0/1 背包算法（按值/成本密度贪心），六种价值函数（FIFO、随机、反转、关键词、复合、关键词+安全回退），以及在不同 token 预算下对候选文件进行排序与裁剪。

**📊 数据集**

使用 SWE‑bench Verified（500 个 Python 任务）作为基准数据集，并在生产会话日志（4,175 个 Bash 文件搜索金标事件）进行验证，此外还收集了两个外部生产语料库做跨语料库复制。

**📈 对比分析**

通过离线评估（算法层）和单轮文件定位测试（五种 LLM，4,800 次调用）比较 p1 与准确率；关键词评分将 p1 从 24.2% 提升至 35.8%，但下游准确率仅提升 ≤ 2.8%，且无显著性；成本方面，云模型成本较低，局部模型几乎零成本。

**⚠️ 局限性**

局限性包括检索上限（grep 生成器限制 50 条候选，44% 任务无金标候选）、仅覆盖 Python 代码、单轮评估而非完整解码流程、缓存 TTL 影响结果、以及跨语料库复现受限。

---

## 220. Glass Surface Detection Grounded in 3D Visual Geometry

**arXiv ID:** 2608.26752 | [PDF](https://arxiv.org/pdf/2608.26752v1)

**作者:** Yiwei Lu `[一作]` (Jiangnan University), Rynson W. H. Lau `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出将玻璃表面检测与三维视觉几何相结合的框架，利用VGGT生成玻璃区域的伪三维监督并构造多任务网络。

**💡 创新点**

创新点在于使用频域自注意模块识别玻璃的高频衰减特征，并通过几何归一块将二维特征与三维几何信息对齐，突破传统二维外观基准的局限。

**🔧 技术方法**

采用VGGT视觉几何基础模型、LoRA微调、FSAM（频域自注意）和GeGB（几何归一块）等技术。

**📊 数据集**

在七个标准玻璃检测基准（GDD、GSD、Trans10K-Stuff、HSO等）以及RGB-热成像、RGB-深度、视频GSD数据集上进行训练和评估。

**📈 对比分析**

与九个现有SOTA方法及三个基础模型对比，取得 IoU/Fβ/MAE 等指标的最高分，平均提升约4–6%，并在多模态和视频场景中保持鲁棒性。

**⚠️ 局限性**

限制在于当玻璃离摄像机过近或几何上下文不足时，VGGT的几何推断不可靠，导致检测失败。

---

## 221. AesCanvas: A Large-Scale Dataset and Benchmark for Aesthetic Critique and Contextual Suitability

**arXiv ID:** 2608.26713 | [PDF](https://arxiv.org/pdf/2608.26713v1)

**作者:** Xuanwei Hu `[一作]`, Jianjun Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 AesCanvas 数据集，包含 54.3K 张图像、519K 条长篇审美评论对以及 301 条情境适配判断案例，并为这些任务提供了多模态 LLM 的基准评估与诊断。

**💡 创新点**

创新点在于：①将传统的审美评论与基于情境的可用性判断相结合，形成闭合式基准；②通过“使用情境–可用性”双向验证，加入文化、叙事、功能等具体证据；③利用对比性视觉反事实与可视化证据评估模型的真实理解能力。

**🔧 技术方法**

使用的技术包括：多模态大型语言模型（Gemini、Claude、Qwen、InternVL、GLM 等）以及其微调；Prompt 设计与层级规划；精确的评估指标（BLEU、ROUGE‑L、METEOR、BERT‑Score、SBERT‑Cos、CLIP‑Cos、EGA、Macro‑F1 等）；对模型输出进行自动解析、错误分析和可视化诊断；对样本进行人工审核与验证。

**📊 数据集**

使用的数据集：AesCanvas（主数据集）；对比的公开数据集包括 AVA、AADB、PARA、AGIQA‑3K、AesBench/EAPD、AesMMIT、Photo‑Critique、ArtiMuse‑10K、RAD/ArtQuant 等；在 AesCanvas 内，CritiqueCanvas 共 519K 条问答对，ContextCanvas 共 301 条使用情境判断案例。

**📈 对比分析**

通过与多款开源与闭源 MLLM（Gemini、Claude、Qwen、InternVL、GLM、LLaVA 等）以及专门的审美子模型（ArtQuant、ArtiMuse、AesExpert、Q‑SiT）进行对比评估。整体准确率在 50%–85% 之间，宏观 F1 低于准确率，表明多数模型倾向于“是”并难以精准区分“否”；对比实验显示专门化模型并不总能显著提升情境判断性能；对视觉反事实与证据评估进一步揭示了模型在可视化证据识别方面的差异。

**⚠️ 局限性**

局限性：①情境判断案例数量有限（301 条），覆盖的文化与使用场景不完整；②仍存在视觉假象、上下文误读与答案不稳定等错误；③模型在解释可视化证据上表现不一，特别是专门化模型往往缺乏可视化依据；④评估主要基于静态图像，未考虑动态或交互式媒体；⑤由于多模态推理复杂，模型训练与推理成本高，限制了可扩展性。

---

## 222. Improving the Robustness of the XRP Ledger Network via Edge Augmentation Strategies

**arXiv ID:** 2608.26380 | [PDF](https://arxiv.org/pdf/2608.26380v1)

**作者:** Afonso Vilalonga `[一作]` (Universidade NOVA de Lisboa), Osman Yağan `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在XRP Ledger的现有网络拓扑中添加新的边（尤其是随机K‑out策略）来提升网络和法定代表（UNL）鲁棒性，并与先前的重连（rewiring）方法进行比较。

**💡 创新点**

创新点在于提出一种增广（augmentation）而非重连的鲁棒性提升策略，证明随机K‑out在保持拓扑相似度（Jaccard相似度高）的同时，能够在添加极少边的情况下匹配甚至超越重连方案的鲁棒性效果。

**🔧 技术方法**

使用的技术包括随机K‑out、随机边添加、优先连接（preferential K‑out）三种增广策略；重连策略按低/高度节点比率重连；通过蒙特卡罗仿真计算网络/法定代表鲁棒性指标（p*）；以及用Jaccard相似度评估拓扑变化。

**📊 数据集**

采用了2022年收集的真实XRP Ledger网络快照（1290个时间点），选取平均指标最接近的快照：952个节点、15070条边、平均度31.7。

**📈 对比分析**

比较方法是对不同K值、不同参与子集比例下的增广策略与重连策略进行多次仿真，计算关键攻击大小p*；结果显示K‑out在K≥3时即可与重连相匹配，并在大多数情形下提供更高的Jaccard相似度（约0.85对低于0.5），证明增广方法既有效又更保留原拓扑。

**⚠️ 局限性**

局限性包括：仅考虑针对度中心或介数中心的定向攻击，未对动态网络增长或节点离线情况进行建模；验证者选取假设为随机或基于度的；实验仅在XRP Ledger数据集上验证，可能不完全适用于其他分布式系统；以及增广策略假设节点能够随机选取对等节点的可达性。

---

## 223. Mapping Written Words to Spoken Words in a Different Language Using Only Visual Grounding

**arXiv ID:** 2608.26925 | [PDF](https://arxiv.org/pdf/2608.26925v1)

**作者:** Gabriel Pirlogeanu `[一作]` (Politehnica Bucharest), Herman Kamper `[通讯]` (Stellenbosch University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过视觉信息对话数据（图片与其旁白）实现英文写词与印地语语音词之间的对齐与检索；

**💡 创新点**

创新点在于结合视觉标签进行正负样本挖掘、对齐聚合与负样本对比，利用自监督音频特征实现跨语言词汇映射，且不需要目标语言转录或显式模型训练；

**🔧 技术方法**

核心技术包括：自动图片生成英文描述、基于HuBERT的连续/离散自监督音频特征、Smith-Waterman或余弦相似度对齐、正负样本聚合（interval piling）以及负样本挖掘；

**📊 数据集**

使用MIT Places Audio Captions Hindi 数据集（约2万对图像-印地语旁白）进行实验，并对比英语版本；

**📈 对比分析**

与先前的注意力CNN神经基线相比，使用连续特征对齐并加入负样本挖掘可在关键词定位P@10上提升约39%（从18.8%到49.9%），关键词检索提升约44%；与使用转录监督的理想上限相比仍有约40%差距；

**⚠️ 局限性**

局限性包括：对图像描述的跨文化差异导致标签噪声，正负样本挖掘依赖于图片描述的准确性；离散特征虽然速度快但性能较差；未对极低资源语言或不同视觉场景进行充分验证。

---

## 224. Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation

**arXiv ID:** 2608.26638 | [PDF](https://arxiv.org/pdf/2608.26638v1)

**作者:** Mingqi Gao `[一作]` (Northeastern University), Weiyan Shi `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出预测驱动评估（Prediction-Powered Evaluation）框架，将少量人工评估与大规模自动评估结合，实现无偏且数据效率更高的系统比较；同时引入Prediction-Powered Saving Ratio（PPSR）元指标，衡量自动指标在预测驱动评估中可节省的人工评估比例；在机器翻译任务上验证该框架的有效性。

**💡 创新点**

①将预测驱动推断（PPI）迁移至系统比较；②设计参数化与非参数化、配对与非配对的检验方法；③提出基于PPI的非参数置换检验；④提出PPSR元指标，提供比现有系统级元指标更高的区分度与排名稳定性。

**🔧 技术方法**

预测驱动推断（PPI）、参数化Z检验、非参数置换检验、配对与非配对设计、PPSR元评估指标。

**📊 数据集**

WMT 2022-2024 机器翻译数据集，包含en-de、en-ru、zh-en、en-zh、ja-en、cs-uk等六个语言对。

**📈 对比分析**

通过在六个WMT数据集上进行置信区间与假设检验的实验，比较自动评估、人类评估与预测驱动评估的统计功效与覆盖率。实验显示：①预测驱动评估在保持无偏的同时比人类评估更数据高效；②PPSR在所有数据集上拥有最高的区分度和排名稳定性。

**⚠️ 局限性**

未建模评审者间不一致导致的方差；使用固定的λ̂和方差估计可能导致过于乐观的置信区间；仅适用于机器翻译，其他非可验证任务需进一步验证。

---

## 225. How Unlikely Is "Unlikely"? Assessing Verbal Probability Perception Across Large Language Models

**arXiv ID:** 2608.26327 | [PDF](https://arxiv.org/pdf/2608.26327v1)

**作者:** Christos Petridis `[一作]` (Temple University), Zoran Obradovic `[通讯]` (Temple University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过给19种大语言模型（包括商用和开源模型）提供11个概率词汇，并让模型在三种提示条件下回答：1）强制单数字响应；2）附带解释的数值响应；3）双向往返测试（数字→词→数字），系统评估了模型对概率词的数值映射、内部一致性以及与人类基准的偏差。

**💡 创新点**

创新点在于：①首次对多模型进行跨模型的概率词映射比较；②引入双向往返一致性实验，量化模型内部词–数映射的自洽程度；③在传统的数值映射基础上加入解释诱导实验，探究解释对方差与偏差的影响。

**🔧 技术方法**

技术方法包括：大语言模型的提示工程（固定模板、强制数值或JSON输出）；多次重复实验采样；统计分析（均值、标准差、互补性缺口、方差比较、Wilcoxon检验、permutation检验）；以及量化内部一致性的MARE与量化误差基线。

**📊 数据集**

使用的数据集为：①人类元分析基准Mosteller & Youtz的52个概率词（取11个关键词）；②19种LLM的实验数据，单词→数值或数值→词的响应，重复多次；③通过随机抽样获得30个数字用于往返测试，每个数字重复5次。

**📈 对比分析**

对比方法：将模型的数值映射曲线与人类基准曲线对齐；计算不同模型间的方差和偏差；使用补充性缺口衡量词对互补关系；采用MARE与量化误差基线评估往返一致性，并通过permutation检验判断与随机一致性的差异。结果显示：大多数模型与人类基准高度吻合，词序保持一致、anchor点准确，但整体存在正向偏移，尤其是负向词；解释诱导可降低模型内方差但不消除偏差；往返一致性随模型规模而改善，部分小模型表现接近随机。

**⚠️ 局限性**

局限性包括：使用默认温度，导致跨模型随机性不可控；每个词仅采样10次，样本量有限；提示模板固定，未检验不同表述对结果的稳健性；人类基准来自过去研究，可能与当代人类解释偏差不符；仅考察11个词汇，未覆盖上下文相对概率表达；未来需扩大样本量、探索更广泛词汇与上下文条件。

---

## 226. Text-to-seed generation: Training-free open-vocabulary seeded semantic segmentation via re-purposing diffusion as text-guided seed generator

**arXiv ID:** 2608.26624 | [PDF](https://arxiv.org/pdf/2608.26624v1)

**作者:** Kumju Jo `[一作]` (Hanyang University), Sungyong Baik `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、无标注的开词汇语义分割框架 T2S，先利用 Stable Diffusion 的交叉注意力生成文本引导的稀疏种子点，再将这些种子点作为点提示输入 SAM 进行区域扩展，完成全图语义分割。

**💡 创新点**

创新点包括：
- 将 Stable Diffusion 视为种子生成器，利用 EOT token 加权提升文本与视觉区域的对齐；
- 采用迭代种子生成与传播（Seed Generation & Spreading）以及负种子提示，提升多实例检测与精细化分割；
- 将 SAM 用作区域扩展模型而非粗掩码后处理，充分利用其空间连贯性；
- 整体框架训练免费、可即插即用，避免了对大规模标注的依赖。

**🔧 技术方法**

核心技术：Stable Diffusion 的交叉注意力与自注意力、Seed Generation & Spreading、SAM 的点提示分割、CLIP 的零样本分类验证、EOT token 加权、迭代掩码生成与裁剪。

**📊 数据集**

使用的数据集：Pascal VOC 2012（VOC-20、VOC-21）、Pascal Context（Pascal59、Pascal60）、Cityscapes、COCO Object、ADE20K、ISPRS Potsdam（用于外域测试）。

**📈 对比分析**

与多种基线（CLIP+SAM、Grounded‑SAM、CaR+SAM、传统 Diffusion+SAM、CLIP‑based dense inference 等）在 mIoU 上对比，T2S 在所有公开基准上均取得最高或接近最高的 mIoU，尤其在 VOC、ADE、COCO 及 Potsdam 上表现尤为突出。

**⚠️ 局限性**

局限性：
- 计算成本高（约 188 秒/张、20 GB 显存），迭代过程导致推理速度慢；
- 对严重遮挡、细小或重叠目标仍易失效；
- 依赖 Stable Diffusion 的注意力质量，若文本与视觉语义对齐不佳，种子定位会误差；
- 在极端外域场景（如高分辨率卫星图像）下仍存在性能下降。

---

## 227. PredVLA: A Sub-Million-Parameter Predictive-Coding Policy for Robot Manipulation

**arXiv ID:** 2608.26673 | [PDF](https://arxiv.org/pdf/2608.26673v1)

**作者:** Hiroki Sawada `[一作]` (Sony Computer Science Laboratory), Shunichi Kasahara `[通讯]` (Sony Computer Science Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

由于缺乏具体论文内容，无法确定研究所做的工作。

**💡 创新点**

无法判断论文的创新点。

**🔧 技术方法**

无法识别使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法比较方法及其性能。

**⚠️ 局限性**

缺少对论文限制的相关信息。

---

## 228. Multi-Dataset Inverse Problem Solving with Distributed Generative AI

**arXiv ID:** 2608.26283 | [PDF](https://arxiv.org/pdf/2608.26283v1)

**作者:** Daniel Lersch `[一作]` (Thomas Jefferson National Accelerator Facility), Nobuo Sato `[通讯]` (Thomas Jefferson National Accelerator Facility)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种可并行处理多异质数据集的生成式逆问题求解框架（Multi‑Dataset SAGIPS），通过将每个数据集分配到单独的 GPU 上，并使用分布式数据并行同步生成器梯度，实现了在不合并数据的情况下共同估计共享未知量。

**💡 创新点**

创新点在于：① 将分布式数据并行从 IID 扩展到非 IID 多数据集；② 每个数据集独立前向算子和判别器，梯度聚合仍然有效；③ 评估多种梯度传输策略并引入数据分片（multiplicity）以进一步提升效率；④ 通过生成对抗网络实现全局一致的未知量恢复。

**🔧 技术方法**

使用技术包括：生成对抗网络（GAN）作为生成器与判别器；SAGIPS 框架；分布式数据并行训练（DDP）；多种梯度聚合方法（conventional ARAR、ARAR、strong‑ARAR、Double Binary Tree）；多 GPU 并行与数据分片；混合精度与 Adam 优化器。

**📊 数据集**

使用的数据集：基于 Ruther‑fall 散射实验的合成多检测器数据，共六个子集，每个覆盖不同角度、能量或探测器分辨率，形成非 IID 的特征空间样本；在 JLab 计算农场与 Argonne Polaris HPC 上执行实验。

**📈 对比分析**

比较方法：与单 GPU 单/多判别器、不同梯度传输策略进行对比。结果显示：多 GPU 分布式训练可将训练时间缩短约 2 倍，内存占用保持稳定；不同梯度传输方案对参数恢复的精度影响不大，但训练时长与 GPU 利用率存在差异；随着加入更多数据集，残差接近零且不确定性减小，证明框架在多数据集情境下保持准确性。

**⚠️ 局限性**

局限性：① 所有数据集必须共享相同的未知参数，否则梯度聚合会导致冲突；② 对低分辨率或系统误差严重的数据集敏感，可能导致收敛慢或偏差；③ 目前仅在合成实验验证，实际复杂物理问题的可扩展性与鲁棒性需进一步评估；④ 需要手动调节梯度传输与数据分片，缺乏自动化权重或平衡机制；⑤ 对极大数据不平衡和统计代表性不足的情况仍需改进。

---

## 229. Four Ways to Forge a Bundle My Own Verifier Calls Clean: Refusal-Site Mutation Testing of an Evidence-Bundle Verifier

**arXiv ID:** 2608.26183 | [PDF](https://arxiv.org/pdf/2608.26183v1)

**作者:** Erik Hill `[一作]` `[通讯]` (Independent Researcher), Erik Hill (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对离线证据束验证器，发现并量化了“空洞通过”(vacuous pass)缺陷，并通过改造代码、添加专门的拒绝站点测试以及扩充 tamper 语料库，评估并提升了拒绝逻辑的可达性与覆盖率。

**💡 创新点**

创新点在于：
• 设计并应用“拒绝站点删除突变器”，以判定每个拒绝点是否真的能在运行中被触发；
• 通过基于两检测器的杀死判定和排除集，构造一个可复现、可量化的拒绝站点覆盖度指标；
• 在同一系统上进行前后对比实验，展示针对性测试（per‑refusal liveness）相较于传统 bug‑specific 修复对覆盖率提升更显著。

**🔧 技术方法**

技术手段包括：
• 变异测试（statement deletion）针对拒绝语句；
• 单元测试（114 条）配合 tamper fixture（16 条）构建拒绝触发基准；
• 两检测器 kill 逻辑（单元测试、liveness 控制、tamper 固件）与排除集；
• 有效性门（baseline green 与排除集完整性检查）。

**📊 数据集**

数据集主要来自作者自建环境：
• 112 个拒绝站点（后续扩展到 146）
• 114 条单元测试
• 16 条 tamper 固件
• 5 条手工发现的虚假拒绝案例（4 次后硬化后发现，1 次 audit 提供），以及多次修复后新增的 20 条 fixture。

**📈 对比分析**

比较方法：以 RQ1‑RQ4 为评估维度，记录拒绝覆盖度（mutation score）在不同阶段的变化：
• RQ1 基线 0.330；
• RQ2 bug‑specific 修复后 0.328（无提升）；
• RQ3 per‑refusal liveness 测试后 0.941；
• RQ4 进一步修复发行者缺口后 0.947；
• 最终全部覆盖 1.000。
性能上，基线突变扫描耗时约 5 分 46 秒，后续实验均保持在可接受的几分钟范围。

**⚠️ 局限性**

局限性：
• 仅在单一作者自建的验证器上测试，缺乏跨供应商、多语言/平台验证；
• 变异器仅删除拒绝语句，未覆盖其他检查路径（如异常抛出、早期返回等）；
• 对可达性与主机依赖问题仍未完全解决，导致部分拒绝点在不同环境下行为不一致；
• 内部有效性受自我报告影响，缺乏独立第三方评估；
• 评估覆盖率指标仅衡量拒绝语句的“可达性”，不等同于整体验证器正确性。

---

## 230. Selection Bias Correction in Retail Intelligence

**arXiv ID:** 2608.26156 | [PDF](https://arxiv.org/pdf/2608.26156v1)

**作者:** Spandan Ghose Chowdhury `[一作]` `[通讯]` (Walmart Inc.), Spandan Ghose Chowdhury (Walmart Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过模拟研究评估零售通胀估计中选择偏差的校正方法。

**💡 创新点**

首次系统比较层次分层与IPW在长尾分布下的表现，并揭示正向性假设严重违背时IPW失效。

**🔧 技术方法**

使用逆概率加权(IPW)、分层、匹配、回归校正等技术，配合线性、多项式和样条模型。

**📊 数据集**

利用基于商品排名的合成数据集，包含10,000–1,000,000条记录，模拟四种不同的数据生成过程。

**📈 对比分析**

在400次Monte Carlo重复中，分层在3/4场景下误差低于0.04pp，保持稳健；IPW在多项式场景下略优，误差约0.007pp。

**⚠️ 局限性**

主要局限在于对真实数据的外推性不足，正向性假设严重违反导致IPW不可行；且实验基于合成数据，缺乏实证验证。

---

## 231. Claude Code Complete User Handbook

**arXiv ID:** 2608.26742 | [PDF](https://arxiv.org/pdf/2608.26742v1)

**作者:** David Soldani `[一作]` `[通讯]`, David Soldani

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文编写了《Claude Code Complete User Handbook》一书，系统地介绍了Claude Code（一个可执行文件系统、浏览器、调度器等功能的代理式工作环境）的安装、使用、权限管理、沙箱隔离、插件与子代理机制等完整工作流程，并配合一套基于官方文档的验证方法和证据日志，形成了面向实践者的可操作手册。

**💡 创新点**

创新点包括：①提出四要素简报（Outcome、Context、Boundaries、Evidence）作为指令规范；②设计四层控制栈（管理策略、权限、沙箱、隔离）并阐明其强制与建议的区别；③构建原始源裁决+机器校验+UNVERIFIED标记的验证框架；④引入Evidence‑Gated Delivery（八阶段交付流程）以实现基于证据的发布；⑤将威胁模型映射至 OWASP、MITRE、NIST、ISO 等标准，提供跨框架对照；⑥提出五层组织成熟度模型，指导企业逐步落地。

**🔧 技术方法**

所使用的技术和工具主要有：Claude Code 代理、MCP、插件/技能、子代理、调度器、沙箱、权限模式、OpenTelemetry、Git/GitHub 集成、自动化构建与检查脚本、机器可验证的引用解析系统、以及常用的 LLM 模型与工具调用接口。

**📊 数据集**

数据来源为：Claude Code 官方文档（截至 2026‑08‑23 与 2026‑08‑26 的完整索引）、Anthropic 官方发布的版本 2.1.241/2.1.246、公开标准（NIST AI RMF 1.0、SP 800‑207/218、ISO/IEC 42001/27001/23894/25010、IEEE Std 1012‑2016/7000‑2021、EU AI Act、ASD Essential Eight 等），以及在 re‑verification 期间检索到的 81 页官方页面。未使用传统机器学习数据集；核心验证对象为官方文档与版本发布记录。

**📈 对比分析**

比较方法为“原始源裁决 + 机器校验”——每条产品声明直接与官方文档原文比对，并通过自动化脚本校验引用一致性与无遗漏；对 81 页文档进行了重新验证，其中 10 页重新裁决。性能上没有量化指标，但通过“effort”级别、计费工具、插件使用成本等维度描述了系统的使用开销与效率；并通过 35 条版本特定行为说明，展示了工具随版本演进的变化。

**⚠️ 局限性**

局限性：①仅对官方文档进行验证，未对实际运行系统行为做全面测试；②UNVERIFIED 条目需本地安装检查；③验证工作由作者单独完成，缺乏第三方审计；④文档与系统功能快速迭代，验证基准 2026‑08‑23 的有效期仅为几周；⑤对付费标准与标准文档的完整访问有限；⑥依赖 Anthropic 官方文档，若其更新出现差异则手册可能不再准确；⑦验证方法侧重文档一致性，无法覆盖所有安全或性能攻击场景。

---

## 232. DeepRepro: State-Aware Subplanning for Paper-to-Code Reproduction in Evolving Repositories

**arXiv ID:** 2608.26557 | [PDF](https://arxiv.org/pdf/2608.26557v1)

**作者:** Hongru Song `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Maarten de Rijke `[通讯]` (University of Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了DeepRepro框架，实现论文到代码的自动重现，利用执行状态感知的迭代子规划完成仓库构建。

**💡 创新点**

引入执行状态感知的子规划与仓库记忆压缩机制，保持规划与动态仓库状态同步，支持深度与快速两种模式。

**🔧 技术方法**

基于大型语言模型（GPT‑5.4、DeepSeek‑V4‑Flash）+多代理分析、蓝图规划、状态感知子规划、工具调用、过程感知前端等技术。

**📊 数据集**

使用PaperBench Code‑Dev子集（20篇ICML 2024论文）以及其五篇精选子集进行实验。

**📈 对比分析**

与DeepCode、Cursor、Codex等科学与商业代码代理对比；在五篇子集平均得分84.2，超越DeepCode（82.0）和商业助手；在20篇全集平均得分75.1。

**⚠️ 局限性**

依赖大型模型，成本与时长较高；深度规划模式耗时更长；受执行模型质量限制，仍难以完全消除依赖与接口推断错误。

---

## 233. Using Poly-Encoders for Computationally Efficient Automated Creativity Assessment

**arXiv ID:** 2608.26165 | [PDF](https://arxiv.org/pdf/2608.26165v1)

**作者:** Sam Grouchnikov `[一作]` (Wheeler Magnet High School), Jiho Noh `[通讯]` (Kennesaw State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用Poly-Encoder对科学创造性思维测试(SCTT)的18000条学生回答进行自动化创意评估。

**💡 创新点**

创新点在于将Poly-Encoder与轻量级BERT编码器结合，既保持与LLM相近的Pearson相关性，又显著降低计算资源需求。

**🔧 技术方法**

使用的技术包括Poly-Encoder架构、BERT系列预训练模型（如DeBERTa-v3-Large、RoBERTa-Base）以及轻量级回归头。

**📊 数据集**

数据集为公开的SCTT数据集，包含15个科学提示约1200条回应及人类评分。

**📈 对比分析**

实验将Poly-Encoder与LLM（如LLaMA-2-7B）对比，Pearson相关性最高达0.74，CPU单候选评分时间仅0.01秒，显著优于传统LLM。

**⚠️ 局限性**

局限性包括SCTT标签噪声、模型对未见提示的泛化不足、对长文本或非标准英语偏见、需GPU训练以及模型黑盒解释性差。

---

## 234. Optimizing API Gateway Placement in Multi-Cloud Kubernetes

**arXiv ID:** 2608.26573 | [PDF](https://arxiv.org/pdf/2608.26573v1)

**作者:** Vinoth Punniyamoorthy `[一作]`, Narender Reddy Bitla `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多云 Kubernetes 集群中 API 网关的放置问题，建立了容量受限的设施定位模型并给出了精确 MILP 求解与贪心启发式两种解法。

**💡 创新点**

创新点在于将网关副本放置、容量分配和地区流量分配统一成具有严格延迟阈值和利用率余量的设施定位问题，并提供了针对该模型的纠正式贪心算法。

**🔧 技术方法**

主要技术包括混合整数线性规划（MILP）、贪心启发式（按增量成本/可用容量排序）、并使用 CBC + PuLP 进行求解。

**📊 数据集**

数据集为合成的多云区域与客户端地区，随机生成请求量、SLA 延迟、网络时延，覆盖12个云区域与10个地区。

**📈 对比分析**

与全复制基线和单一最低成本基线对比；MILP 在大多数实例可在秒级内得到最优解，贪心算法平均最优差距约 3–5%，单实例最大 25%，速度提升 660–3500×；在典型实例上 MILP 成本比全复制低 24.2%，单一最低成本虽更便宜但存在 3/10 地区延迟违约。

**⚠️ 局限性**

局限性包括：延迟模型仅为基于地理距离的合成、未考虑排队、后端服务延迟及异构实例成本、缺乏冗余/容错约束、未评估迁移或滚动部署成本，且贪心算法的最优性间隙未在更大规模实例上验证。

---

## 235. VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation

**arXiv ID:** 2608.26155 | [PDF](https://arxiv.org/pdf/2608.26155v1)

**作者:** Yixia Li `[一作]` (Southern University of Science and Technology), Furu Wei `[通讯]` (Microsoft Research Asia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Vision-Free Adaptation（VFA）框架，通过先在多语言文本上微调基础 LLM 获得多语言任务向量，再将该向量与已视觉对齐的 MLLM 任务向量融合，实现在不需要任何图像–文本对的情况下提升多语言多模态能力。

**💡 创新点**

创新点在于将语言学习与视觉对齐解耦为可叠加的任务向量，从而避免直接文本微调导致的视觉对齐灾难性遗忘；同时通过任务向量融合实现高效、可复用的多语言能力注入。

**🔧 技术方法**

技术手段包括：任务向量（task vector）概念、对基础 LLM 的全参数或参数高效微调、三种模型融合算子（权重平均、任务算术、TIES）、冻结视觉编码器与投影层以保持视觉对齐。

**📊 数据集**

使用的主要数据集是包含 100K 示例的多语言 SFT 数据集（xP3mt、Bactrian‑X、Aya Vision 文本子集等），评估基准包括六大多语言多模态基准（MaXM、xGQA、xMMMU、XM100、MaRVL、M3Exam）、通用多模态基准（OCRBench、MMBench、MMMU、MathVista）以及文本任务基准（TyDiQA、MMMLU、XNLI、HellaSwag、MLogiQA、M‑IFEval、FLORES）。

**📈 对比分析**

与原始 MLLM 和直接文本微调进行对比，VFA 在多语言多模态平均分提升 2.39–10.64 分，保持或略升通用多模态性能；仅用 100K 文本数据即可逼近全多模态训练模型，显著提升数据效率。

**⚠️ 局限性**

局限性包括：仅通过文本学习，感知密集任务（如 OCR）仍需额外图像–文本监督；多语言文本数据质量筛选不足；实验仅覆盖 4B–8B 参数规模，需进一步验证更大模型和更高级融合算法。

---

## 236. Muon with Finite Newton-Schulz: The Smoothing Benefit in Nonsmooth Nonconvex Optimization

**arXiv ID:** 2608.26288 | [PDF](https://arxiv.org/pdf/2608.26288v1)

**作者:** Mingyi Li `[一作]` (University of Tokyo), Taira Tsuchiya `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文分析了Muon优化器在有限Newton-Schulz迭代下的表现，证明了其在非光滑非凸优化中能够找到平稳点的能力。

**💡 创新点**

创新点在于将有限Newton-Schulz迭代视为平滑机制，而非近似误差，提出了深度与极性近似误差和更新的Lipschitz常数之间的权衡。

**🔧 技术方法**

使用了有限Newton-Schulz迭代和在线到非凸转换（O2NC）框架来分析Muon的更新规则。

**📊 数据集**

未具体提及使用的数据集，但讨论了在大语言模型预训练中的应用。

**📈 对比分析**

与传统的精确极性更新相比，有限Newton-Schulz迭代在非光滑非凸优化中表现出更好的收敛性，能够在期望内找到平稳点，且样本复杂度界限与已知的最佳保证相匹配。

**⚠️ 局限性**

限制在于分析中假设了固定的归一化常数，而实际应用中Muon的归一化依赖于数据，可能会违反某些条件。

---

## 237. A Task-Centric Ontology and Deterministic Domain Rules as a Verifiable Core for AI-Assisted Chemistry Problem Solving

**arXiv ID:** 2608.26164 | [PDF](https://arxiv.org/pdf/2608.26164v1)

**作者:** Ibrokhimsho Abduchaborov `[一作]` `[通讯]` (Independent Researcher), Ibrokhimsho Abduchaborov (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了面向任务的化学本体与确定性规则引擎，实现了对中学化学问题的符号推理与自动解答。

**💡 创新点**

首次将任务导向的本体设计与可执行规则相结合，并明确区分通用规则与任务特定回退，提供可解释、可检验的核心。

**🔧 技术方法**

使用JSON/RDF/Turtle格式的本体、Python脚本实现的规则引擎、SHACL/OWL验证机制以及结构化的验证与引用匹配。

**📊 数据集**

在300道人工编写、人工验证的中学化学题库（主要为选择题）上进行评估。

**📈 对比分析**

通过与人工参考答案的精确匹配进行评估，完整系统匹配率98.67%，规则子集98.88%，回退子集96.77%，未做LLM基准比较或Token/成本测量。

**⚠️ 局限性**

仅在内部数据集上验证，缺乏外部泛化评估；问题类型单一且高度依赖回退逻辑；未测量时间/Token成本；数据分布与开发重叠导致可能过拟合。

---

## 238. FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models

**arXiv ID:** 2608.26676 | [PDF](https://arxiv.org/pdf/2608.26676v1)

**作者:** Junyoung Lee `[一作]` (POSTECH), Yeseong Kim `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种基于 token 级别的指导方法，旨在缓解剪枝后大语言模型在生成过程中的文本退化（尤其是重复循环）问题。该方法在剪枝后细调过程中通过重新加权知识蒸馏和对比损失来控制 token 级别的概率分布，从而降低循环进入和持久化的风险。

**💡 创新点**

创新点在于：① 对文本退化进行 token 级别的动力学分解，明确循环进入风险和循环持久性两大驱动因素；② 引入 FOCUS（基于教师高置信度 token 的加权蒸馏），在保留知识的同时抑制低置信度泄漏；③ 引入 RePAIR（基于起始点的正负对齐）通过 margin loss 推动模型在循环起始处产生可接受的替代 token，从而阻断重复循环。

**🔧 技术方法**

技术手段包括：
- 重新加权的知识蒸馏（FOCUS）
- 对比学习式 margin 损失（RePAIR）
- 使用 LoRA 进行剪枝后细调
- 基于 nucleus (top‑p) 采样的解码分析
- 对 token 级别概率分布进行逃逸质量（escape mass）评估与调控

**📊 数据集**

主要使用的数据集有：
- WikiText‑103（开放式续写任务）
- Self‑Instruct（指令生成任务）
- Alpaca（用于剪枝后细调的指令对齐数据）

**📈 对比分析**

与 KD、UL、ScaleGrad、DITTO 等传统训练方法以及无方法基线进行对比。实验结果显示：
- CREP、重复 n‑gram 率显著下降（例如在 Llama‑3.1‑8B 上从 2.23% 降至 0.57%）；
- MAUVE、EAD_1、BERTScore 等多样性与语义质量指标均提升；
- perplexity 仅略有上升，保持在可接受范围；
- 结合两种方法时效果进一步提升，优于所有单方法或组合基线。

**⚠️ 局限性**

局限性包括：
- 某些情况下 perplexity 仍有小幅提升，可能影响极端长文本的精确度；
- 需要额外的正负对齐样本，虽然样本量相对较小，但在更大规模或多语言场景下仍需验证；
- 方法主要在 Llama 系列模型上验证，尚未在其他架构（如 GPT‑NeoX、BERT‑based）上全面评估；
- 对极端噪声或不一致的教师分布的鲁棒性尚未深入探讨。

---

## 239. EEG-to-Report: An Annotation and Feature-Text Framework for Training Language Models on Clinical EEG

**arXiv ID:** 2608.26153 | [PDF](https://arxiv.org/pdf/2608.26153v1)

**作者:** Xuan-The Tran `[一作]` (Vietnam Maritime University), Le Trung Kien Nguyen `[通讯]` (HAISmartlink Lab, ANCHI STE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一个浏览器端注释与特征‑文本框架EEG‑to‑Report，用于构建临床EEG与语言模型的训练数据，并集成了自动报告演示；实现多格式无厂商EEG导入、交互式段段级注释（文本/语音）、自动提取标准化特征并与自由文本对齐，最终输出可直接用于AI训练的JSON数据；该框架可在本地运行，保持原始EEG文件在机构内部。

**💡 创新点**

① 多格式无厂商EEG导入与通用通道标准化，解决不同厂家文件兼容问题；② 交互式段段级注释结合语音转文字，实现临床医生在单一界面完成注释、特征提取与文本撰写；③ 统一的特征‑文本JSON schema，使得每个段落都有时间、通道、标准化特征和临床文本对齐，直接可用于训练跨模态语言模型；④ 集成自动报告演示，演示卷积网络+大语言模型的端到端报表生成。

**🔧 技术方法**

前端：React + TypeScript + Plotly.js；后端：FastAPI + MNE‑Python + NumPy/SciPy；语音转文字：Whisper（本地模型）；特征提取：功率谱、时间域、Hjorth、熵、相干性、尖波计数等；自动报告：多卷积网络（CNN、GoogleNet、ResNet）+大语言模型（Gemini、GPT‑4、Claude）。

**📊 数据集**

Siena Scalp EEG Database（公开的EDF文件，512 Hz，10‑20 montage，36 记录，12 病人），用于演示导入、注释、特征提取与自动报告。

**📈 对比分析**

在Siena数据集上验证了框架的可行性：生成112段、特征向量长度1322、文本平均9词。自动报告模块使用卷积网络+语言模型生成草稿，未与真实报告做量化比较，主要通过专家主观评估确认可读性；特征→文本模型仍未训练，未来计划进行基准评估。

**⚠️ 局限性**

① 单中心单注释实验，缺乏多中心、多语言验证；② 特征仍为手工构造，未尝试端到端学习；③ 自动报告可能产生幻觉、缺少系统性错误评估；④ 目前未评估工作流程效率提升；⑤ 需要进一步用户研究与更大规模数据集的验证。

---

## 240. Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints

**arXiv ID:** 2608.26171 | [PDF](https://arxiv.org/pdf/2608.26171v1)

**作者:** Hiroko Takano `[一作]` `[通讯]` (Lemmanode), Hiroko Takano (Lemmanode)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多阶段LLM招聘任务中的伪造风险，并评估提示安全栏与人工检查点的缓解效果。

**💡 创新点**

通过实验对比全自动、提示安全栏、人工检查点三种条件，量化伪造发生率、残留错误，并揭示两者互补的安全机制。

**🔧 技术方法**

采用多阶段LLM流水线（简历改写、面试题生成、答案反馈），使用提示安全栏、人工检查点，并用跨模型LLM评判器与人工审阅器进行判定。

**📊 数据集**

构建10份无PII的合成简历与对应的职位描述，含真值声明与缺失列表，用作实验数据集。

**📈 对比分析**

通过180次实验（10简历×2JD×3条件×3重复）比较发现密度、项目级伪造率与内容保留；提示安全栏将发现密度降至0.92/输出，人工检查点将密度降至2.82/输出，伪造率分别从96.7%降至50%和75%，且未损失内容或实用性。

**⚠️ 局限性**

仅单个审阅者、单一模型与单一领域、合成数据可能不代表真实简历、判别器验证样本有限，且指标受限于评审者误差与非独立性。

---

## 241. Interpretable, Fairly Evaluated Automated L2 Speaking Assessment that Beats the Single-Human Ceiling and Why Pause Encoding Does Not Change LLM Fluency Scores

**arXiv ID:** 2608.26137 | [PDF](https://arxiv.org/pdf/2608.26137v1)

**作者:** Eichi Uehara `[一作]` `[通讯]` (Aflo Technologies Inc.), Eichi Uehara (Aflo Technologies Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一种可解释的特征+LLM混合模型，用于评估非母语英语学习者的自发对话口语流利度。

**💡 创新点**

①在未对标注数据进行拟合的情况下实现高可信度评分；②证明暂停编码方式（统计、文本内插、语义定位）对LLM流利度评分无显著影响，流利度信号主要来源于语音时序特征；③提出可靠性校正的评估上限与单一评审者基准，避免传统“无噪声金标”误导。

**🔧 技术方法**

使用基于De‑Jong的五项可解释特征（暂停比例、平均停顿长度、包含暂停的语速、长暂停率、ASR词置信度）构成确定性综合分；用Zero‑shot DeepSeek‑Chat 提取文本流利度评分；将两者简单加权得到最终分数；采用配对自助法计算置信区间；对暂停编码进行对照实验。

**📊 数据集**

ICNALE Global Rating Archive (GRA)：140段90秒的双人角色扮演对话，约80名训练评审按10个分析维度（含整体）打分；共10个母语背景（CHN、TWN、KOR、JPN、IDN、THA、HKG、MYS、PAK、PHL）CEFR等级A2–B2。

**📈 对比分析**

将模型分数与80位评审的共识平均分（Fluency）进行Spearman相关；确定性特征组合得到ρ≈0.764，加入LLM流利度后提升至ρ≈0.818。此分数高于单一评审者平均（ρ≈0.62）和中位数（ρ≈0.73），占可靠性校正上限（κ_max≈0.99）的约83%。暂停编码实验显示统计式编码略优，其他两种编码差异在置信区间内不显著。

**⚠️ 局限性**

①仅在ICNALE语料（10个亚洲母语、A2–B2）上验证，外推性未检验；②样本量有限，部分母语组n<10导致公平性评估不稳；③LLM使用单一模型（DeepSeek‑Chat），不同模型可能产生不同绝对分数；④语音识别误差对低水平或高口音学习者影响不均；⑤对话时段短且受角色扮演任务约束，可能与朗读或日常对话的口语表现不完全一致。

---

## 242. Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility

**arXiv ID:** 2608.26449 | [PDF](https://arxiv.org/pdf/2608.26449v1)

**作者:** Sajal Regmi `[一作]` (Karela Technologies Inc.), Chetan Phakami Pun `[通讯]` (Karela Technologies Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 HuggingFace Byte‑Level BPE tokenizer 预分词正则（基于 GPT‑2 的 \p{L}+ 词类）对 abugida 脚本（如印地语、尼泊尔语等）导致的词拆分过度问题，并量化其对 token “fertility”（每词 token 数）的影响。通过提出训练无关的下界公式、诊断方法、与修复版本的对比实验，最终发布了修复后的 Nepali‑English tokenizer，并对 HuggingFace 生态中受影响模型的比例进行了统计。

**💡 创新点**

创新点在于：① 提出了一种无需训练即可通过单个正则匹配得到的 fertility 下界，揭示预分词对 token 数的硬性限制；② 通过控制实验与“诊断”扫描识别并量化字符类对 fertility 的影响；③ 对 HuggingFace 上数千模型进行快照 census，量化字母类预分词在主流模型中的普及率；④ 将修复方案与公开模型进行公平比较，展示在同计算预算下的性能提升。

**🔧 技术方法**

使用技术包括：Byte‑Pair Encoding（BPE）tokenizer、UTF‑8 级别的字节映射、正则表达式预分词、词汇量预算分析、Bits‑per‑Byte 评估、FineWeb‑2、C4、FLORES‑200 等公开语料库的训练与评估、以及 Python/Makefile 级别的实验 harness。

**📊 数据集**

数据集：主要使用 FLORES‑200 并行语料（含 204 种语言）作为评估基准；训练使用公开的 Nepali‑English 语料（FineWeb‑2 / C4 细分）；评估时亦使用 FineWeb‑2 的验证集和 C4 的验证集；所有数据均为公开、无个人信息。

**📈 对比分析**

比较方法：① 在同一模型规模（268M）和相同字典大小下，训练三种条件（broken、fixed、broken‑with‑extra‑compute）并记录 Bits‑per‑Byte；② 通过 tokens‑per‑word 与 bits‑per‑byte 两个指标评估 fertility 与压缩率；③ 将自研 tokenizer 与 13 公开 tokenizer 在 FLORES‑200 上进行对比，突出在 Nepali 上的最低 fertility 与最高 lossless 性能。性能上，修复后的 tokenizer 在同计算预算下 Nepali bits‑per‑byte 下降约 5‑6%，且在同字典下比大多数公开模型的 Nepali fertility 更低。

**⚠️ 局限性**

limitations：① 仅在单一语言对（English‑NEP）和单一模型规模上测试；② 每个实验仅跑一次种子，未进行多次随机复现；③ 只评估 bits‑per‑byte，未覆盖下游任务准确率；④ 修复只在正则层面，未探讨更大 vocab 对英文成本的影响；⑤ HuggingFace census 为一次快照，未覆盖 gated 或无 tokenizer.json 的模型；⑥ 对多脚本多语言混合训练的实际效果未全面验证。

---

## 243. Counterfactual Bias Testing for Application Tracking System

**arXiv ID:** 2608.26899 | [PDF](https://arxiv.org/pdf/2608.26899v1)

**作者:** Sai Yashwant `[一作]` (ManpowerGroup Services India Pvt. Ltd.), Gantala Thulsiram `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套完整的基于多代理大语言模型的候选人–职位匹配系统公平性审计方法，自动生成身份中性基准简历与多种人口特征变体，利用语义匹配模型对简历进行排名，随后计算九种多族群、对比、归功与排名质量的公平性指标，并生成包含风险评分与可视化报告的自动审计输出。

**💡 创新点**

创新点包括：① 两阶段生成设计，先产生身份中性基准简历再单独注入特征，严格保证资格与身份信号分离；② 将对应审计与量化公平性评估统一在单一流水线；③ 引入九种多族群公平度量，配合相应的非参数检验、引导置信区间与Benjamini–Hochberg校正；④ 双阈值报告体系，兼顾科研严谨与执行级别可解释性；⑤ 自动生成含风险评分的完整 HTML 报告，便于内部审计与合规验证。

**🔧 技术方法**

核心技术包括：多任务专用 LLM 代理（描述符提取、简历生成、翻译、偏见标记）、结构化提取 + 句子嵌入（如 EmbeddingGemma）、余弦相似度排名、非参数统计检验（Wilcoxon、McNemar、Fisher、Wilson）、引导置信区间、Benjamini–Hochberg 误差率校正。

**📊 数据集**

使用了基于 LLM 生成的模拟对应审计数据集：5 个职位、100 个身份中性基准简历、10 个人口特征变体（共 5,000 条简历），其中 100 条基准简历带有人工设定的真值匹配标签，用于计算归功与排名质量指标；翻译阶段使用了多语言支持，但嵌入与排名均在统一的英文结构化文本上进行。

**📈 对比分析**

方法通过对比传统人工对应审计的可扩展性与成本，展示在该合成数据集上：大部分指标均满足严格阈值（0% FAIL，4% INVESTIGATE），风险评分低至 1.5%；相比单一评分或仅保留比例的评估，方法能揭示仅在排名稳定性（MARC）和排名质量（nDCG@K）上出现的边缘问题，说明多指标评估的必要性。

**⚠️ 局限性**

局限包括：① 评估基于合成简历，未检验与真实候选人数据的对应性；② 归功与匹配标签来自预设规则，缺乏人工评判；③ 保护特征分类器与偏见注入模型的误差可能影响结果；④ 需要足够的候选人样本才能获得有力的置信区间与检验功效；⑤ 只针对语义排名模型，未覆盖其他可能的匹配算法；⑥ 方法的阈值与阈值选择需在不同司法辖区进一步校准。

---

## 244. A Multi-Modal AI Framework for Real-Time Queue Prediction, Management and Optimisation in Intelligent Border Control Systems

**arXiv ID:** 2608.27010 | [PDF](https://arxiv.org/pdf/2608.27010v1)

**作者:** Varvara Mama `[一作]` (Hellenic Mediterranean University), Anargyros T. Baklezos `[通讯]` (Hellenic Mediterranean University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个多模态 AI 框架，用于边境控制系统的实时队列预测、管理和优化；通过融合多源数据并利用 LSTM 进行预测，再结合模型预测控制 (MPC) 和调度优化实现动态、可预测的队列管理。

**💡 创新点**

创新点在于：①将多源数据融合、深度学习预测与控制理论优化有机结合，形成统一的端到端框架；②针对陆路与渡轮边境环境设计不同的优化策略；③通过实时预测主动调节资源，显著提升吞吐量并降低等待时间。

**🔧 技术方法**

使用技术包括多源数据融合、LSTM 神经网络预测、模型预测控制 (MPC)、调度优化、以及基于强化学习的未来改进方向。

**📊 数据集**

使用了合成数据集模拟现实边境交通，包含车辆到达模式、季节/高峰波动、预登记信息、天气环境等多模态信息。

**📈 对比分析**

与历史平均、ARIMA 时序模型以及静态规则基准进行对比，实验显示 LSTM 预测 MSE 降低约35%，平均等待时间减少约30%，吞吐量提升约20%。

**⚠️ 局限性**

主要局限在于：①实验基于合成数据，缺乏真实运营数据验证；②对多源数据缺失、噪声或传感器失效的鲁棒性仍需进一步提升。

---

## 245. Why RAGs Hallucinate: Penalty-Aware Evaluation of Retrieval-Augmented Generation Systems with Knowledge-Gap Canaries

**arXiv ID:** 2608.26385 | [PDF](https://arxiv.org/pdf/2608.26385v1)

**作者:** Alden Do Rosario `[一作]` (CustomGPT.ai), Felipe Pires `[通讯]` (CustomGPT.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对商业检索增强生成(RAG)产品构建了一个基于置信度阈值的评估框架，聚焦于停答行为与检索缺口造成的“伪造”答案；

**💡 创新点**

创新点在于：①提出知识缺口“金丝雀”问题将参数内存泄漏转化为可测量的违规率；②采用跨家族LLM评审小组与大多数投票消除评审偏差；③将不对答惩罚与传统准确率结合，揭示停答策略对性能的影响；

**🔧 技术方法**

使用了置信度阈值惩罚评分、金丝雀覆盖检测、意图分类器、三方LLM评审和失败归因管道；

**📊 数据集**

实验基于SimpleQA‑Verified 1,000个短篇事实问答集合，并对其中18个答案缺失的题目做金丝雀标记；

**📈 对比分析**

与三款商业RAG系统及无检索基线进行三次重复对比，结果显示三系统答题准确率相近（97–98%），但金丝雀违规率与停答率差距巨大（0.2%–98%），在不同置信阈值下重新排序显示OpenAI RAG始终领先；

**⚠️ 局限性**

局限包括：仅在单语短答数据集上验证；金丝雀样本量有限；评审与模型共享同一家族时可能存在偏差；未覆盖私有知识库情境，无法直接评估假设的误报风险；

---

## 246. Physics-Informed Stochastic Configuration Machine: A Backpropagation-Free Neural Network with Fast Training for Nonlinear Differential Equations

**arXiv ID:** 2608.26549 | [PDF](https://arxiv.org/pdf/2608.26549v1)

**作者:** Yuehao Song `[一作]` (Central South University), Kai Zhang `[通讯]` (China Institute of Water Resources and Hydropower Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无梯度反向传播的物理信息随机配置机（PI‑SCM），能够通过解析线性化和最小二乘求解直接训练单隐藏层网络，解决非线性微分方程的正向与逆向问题。

**💡 创新点**

创新点包括：① 对非线性微分算子做局部泰勒展开，得到关于隐藏单元输出权重的线性关系；② 采用随机配置、线性最小二乘和截断奇异值分解，构建三种递进更新策略（PI‑SC‑I、II、III），保证收敛；③ 通过监督节点选择和离散线搜索实现残差单调下降；④ 将同一框架扩展到参数识别，避免传统梯度优化。

**🔧 技术方法**

使用的技术有：随机配置网络（SCN）、解析雅可比矩阵、泰勒线性化、伪逆最小二乘、截断奇异值分解、滑窗与全局更新、离散线搜索、sine 激活函数，并在实验中采用 Adam 训练的 PINN、PIELM、PISCN、VS‑PINN 等基线。

**📊 数据集**

实验数据集主要是合成的基准方程：Van der Pol 震荡子（ODE）、二维 Helmholtz 方程（PDE）和 Allen–Cahn 方程（PDE）。每个问题提供了内部、边界和稀疏观测点，作为训练和验证集。

**📈 对比分析**

与标准 PINN（浅层 1×400、深层 4×64）、PIELM、PISCN、VS‑PINN 进行对比。PI‑SC‑III 在所有任务中均保持与深度 PINN 相当或更优的精度（如 Helmholtz RMSE 1e‑6），同时训练时间减少 1–2 个数量级；在逆向任务中也实现了与深度 PINN 竞争的参数识别精度。

**⚠️ 局限性**

局限性：① 需对算子进行解析雅可比，适用范围受限；② 对高度非线性问题的泰勒截断误差可能显著；③ 激活函数选择对收敛速度敏感；④ 需要手动调节 τ、r、α 等超参数；⑤ 随着隐藏层宽度增大，全局更新会产生矩阵病态和计算开销；⑥ 目前仅在低维、光滑或可解析的基准上验证，需进一步研究高维、强耦合或噪声数据的鲁棒性。

---

## 247. Beyond Vector Hiding: Breaking and Mitigating Shared-Direction Weight Obfuscation in TEE-Offloaded Large Language Models

**arXiv ID:** 2608.26651 | [PDF](https://arxiv.org/pdf/2608.26651v1)

**作者:** Menghui Zhang `[一作]` (Shandong University), Ran Tao `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析并突破了TEE隔离下大型语言模型权重量化掩码的安全隐患，发现共享单向量重用导致矩阵层级泄漏，并提出基于最大秩蝶形掩码的新防御；

**💡 创新点**

创新点在于：①首次揭示共享rank‑one重用是ArrowCloak泄漏根源；②设计两种针对实值与有限域的高效攻击（PCA与格子求解）；③提出键控最大秩蝶形掩码，实现O(ℓmax{m,d}log d)可信校正，既消除rank‑one泄漏又保持高效推理；

**🔧 技术方法**

使用主成分分析、格子求解（LLL）、矩阵分解、密钥化蝶形变换、固定点量化与模Q运算等技术；

**📊 数据集**

实验涵盖 ViT‑Base（CIFAR‑10/100/Food‑101）、BERT‑Base/GPT2‑Base/XL（GLUE 的 MNLI/QQP/SST‑2/QNLI）以及大规模 LLM（Qwen3‑8B、Llama‑3.1‑8B、Gemma‑4‑E2B）等数据集；

**📈 对比分析**

与ArrowMatch、公开模型对比，实值攻击恢复 87.98% 任务准确率（vs 89.85% 受害者），模Q攻击可精确恢复所有权重并保持 91.63% 准确率；引入蝶形掩码后，攻击无法恢复，模型准确率保持 91.63% 以上，可信校正成本仅为原始 O(nℓ) 的 O(log n) 级别；

**⚠️ 局限性**

局限性：攻击针对的是共享rank‑one重用的特定结构，若权重量化方案不采用同一隐藏向量则不适用；蝶形掩码虽低开销，但实现复杂度较高，且在极大矩阵或非方阵层中仍需进一步评估；实验仅在特定模型和任务上验证，缺乏更广泛的跨架构验证。

---

## 248. Same Model, Different Harness: Different Coding-Agent Results

**arXiv ID:** 2608.26218 | [PDF](https://arxiv.org/pdf/2608.26218v1)

**作者:** Sydney Lewis `[一作]` `[通讯]`, Sydney Lewis

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了在相同模型与任务下，不同编码代理 harness 配置对性能的影响，比较了完整转录对照与闭环视图缩减+检测器干预的 treatment；

**💡 创新点**

提出了通过动态缩减历史工具结果并自动检测停滞模式的闭环 harness 方案，证明 harness 的设计同样能显著提升编码代理性能；

**🔧 技术方法**

采用 Yuj harness、半寿命上下文压缩、检测器模式识别、机械干预，以及对 Qwen3.6‑35B 及其他四位权重模型的 greedy 推理；

**📊 数据集**

在 SWE‑bench Verified、SWE‑bench Pro 与 FeatureBench 三大编程任务集上进行实验，包含 169 个 Verified 任务等；

**📈 对比分析**

采用配对实验（相同任务、模型权重），并使用 McNemar、sign test、Wilcoxon 等统计检验，结果显示在高上下文压力下 treatment 平均 F2PF 提升 0.03–0.05，完成任务率提升 1–3 倍，且在多模型上保持一致；

**⚠️ 局限性**

只测试了固定的 treatment 包，没有拆分单个规则的贡献；实验局限于低精度四位权重模型和特定 benchmark；未评估多次运行的方差；未考虑更大模型或其他任务类型。

---

## 249. Unsaid, Unsafe? Implicit Security Obligations in LLM-Based RTL Code Generation

**arXiv ID:** 2608.26588 | [PDF](https://arxiv.org/pdf/2608.26588v1)

**作者:** Guang Yang `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 SecRTL‑Gen 基准，研究 LLM 生成 RTL 时隐式安全义务缺失的问题，并提出 SecRTL‑Gen 神经符号框架自动推断并修正这些义务。

**💡 创新点**

创新点在于：① 用功能语义图与 CWI 模式本体进行符号推理，自动生成信号级安全义务；② 采用两阶段生成（先功能草稿再义务驱动修正），兼顾功能性和安全性；③ 在多语言、多模型下系统性评估，显著提升所有通过率。

**🔧 技术方法**

主要技术包括：大语言模型提取功能语义图、基于闭合词汇表的符号匹配 CWI 本体、LLM‑as‑Judge 过滤无效义务、两阶段生成流水线；对比 SecV、RESCUE、MAGE 等安全生成方法。

**📊 数据集**

使用 SecRTL‑Gen 数据集：392 个实例（98 个真实 SoC IP × 4 HDL），覆盖 5 类资源访问相关 CWE，配套黑盒功能与安全测试用例。

**📈 对比分析**

方法上与 SecV、RESCUE 等基线进行比较，所有通过率从 49.6%/51.4% 提升至 61.6%（+12个百分点），功能通过率提升约 15%，安全通过率提升约 5%；在 5 种 LLM 与 4 种 HDL 上均保持领先。

**⚠️ 局限性**

局限性：仅覆盖资源访问类 CWE，侧信道、故障注入等安全范畴未包含；符号匹配召回率仍有提升空间；生成阶段仍有大量失败，需进一步改进义务应用与本体覆盖。

---

## 250. Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher

**arXiv ID:** 2608.26872 | [PDF](https://arxiv.org/pdf/2608.26872v1)

**作者:** Shiyi Zhang `[一作]` (Tsinghua University), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Self-OPD，一种面向 Flow Matching 模型的无教师 on‑policy distillation 框架，通过本地 SDE 分支和自我参考基线实现逐步监督，支持多目标（文本渲染、组合生成、审美偏好）对齐。

**💡 创新点**

创新点：
- 用学生自身的 SDE 分支与 deterministic ODE 基线替代外部教师，形成自监督的奖励加权拉推目标；
- 采用 all‑branch pull‑push 归一化损失，并加入方向感知抑制避免负分支干扰；
- 在奖励层面对多目标进行融合（z‑score 标准化 + 加权和），实现无梯度冲突的多任务优化；
- 通过 SDE 方差归一化与 KL 目标等价，提供理论解释。

**🔧 技术方法**

使用技术：Flow Matching 与 SDE/ODE 采样；self‑exploration branching、奖励加权优势（advantage）回归；方向感知抑制系数；奖励级融合；LoRA 微调；AdamW 优化；OCR、GenEval、PickScore、HPSv2 等评估指标。

**📊 数据集**

数据集与模型：以 SD3.5‑Medium（512×512）为基础，训练时使用多任务奖励模型；文本渲染任务基于 OCR 数据集；组合生成任务基于 GenEval 数据集；审美偏好评估使用 PickScore 与 HPSv2 数据集。

**📈 对比分析**

比较方法：对比 Flow‑GRPO、GRPO‑Guard、DiffusionNFT（RL 方案）以及 Flow‑OPD、DiffusionOPD（教师‑OPD）等。Self‑OPD 在单目标训练中取得最高 GenEval（0.95）、OCR（97.5%）、PickScore（24.79）和 HPSv2（0.3665）；在混合目标训练中同样获得最高 GenEval（0.95）、OCR（96%）、PickScore（23.87）和 HPSv2（0.3214）。训练时间显著下降（≈48 h 对比 DiffusionOPD 的 97 h），且不需要额外教师模型。

**⚠️ 局限性**

局限性：
- 依赖可量化的奖励模型，若奖励不充分或存在偏差可能导致不良对齐；
- 由于自我探索的随机性，训练效率受分支数 K 与探索幅度 η 的影响，需要调参；
- 在极大模型或高维图像空间中，SDE 分支探索可能不足以覆盖所有有利方向；
- 目前未对跨模态（如视频、文本）或大规模多任务进行验证。

---

## 251. Algebraic Multigrid Acceleration for Efficient Label Spreading

**arXiv ID:** 2608.26309 | [PDF](https://arxiv.org/pdf/2608.26309v1)

**作者:** Antonia van Betteray `[一作]` (Osnabrück University), Matthias Rottmann `[通讯]` (Osnabrück University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过构建基于CLIP特征的k‑NN稀疏图并使用代数多重网格（AMG）预条件器，提出了一种高效的半监督标签传播框架。

**💡 创新点**

创新点在于将AMG预条件技术应用于标签传播的线性系统，显著提升收敛速度、降低运行时间，并在不同超参数下保持或提升分类准确性。

**🔧 技术方法**

主要使用的技术包括CLIP特征提取、UMAP降维、Faiss GPU加速k‑NN构建、代数多重网格（AMG）与FGMRES迭代求解。

**📊 数据集**

在EMNIST‑digits、CIFAR‑10和Tiny ImageNet三个图像数据集上进行实验。

**📈 对比分析**

与scikit‑learn的power method、SciPy直接求解、CG实现等基线相比，AMG加速方法在相同精度下实现了数十倍的速度提升，并在不同超参数（α、k）下表现更稳健。

**⚠️ 局限性**

局限性包括对GPU资源的依赖、对高维特征预提取的需求以及在极大规模数据集上仍需更高内存和更深层次的预处理。

---

## 252. How Does Science Education Research Respond to Sociopolitical Change? A BERTopic Analysis of Korean Research

**arXiv ID:** 2608.26675 | [PDF](https://arxiv.org/pdf/2608.26675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 253. FU-Mamba: A Frequency-Enhanced Dynamic Scanning Framework for Oralscan Image Segmentation

**arXiv ID:** 2608.26607 | [PDF](https://arxiv.org/pdf/2608.26607v1)

**作者:** Xinxin Zhao `[一作]` (Zhejiang Gongshang University), Yan Tian `[通讯]` (Zhejiang Gongshang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出FU-Mamba框架，用动态扫描和频域增强实现口腔图像语义分割

**💡 创新点**

创新点在于动态扫描模块(DMB)自适应学习采样位置并重排序列，频域增强块(FEB)通过小波分解与谱池化平衡高低频特征

**🔧 技术方法**

技术包括视觉状态空间模型（Mamba）、Offset Prediction Network、双线性插值、Wavelet-Guided Spectral Pooling、DFT频谱池化、UNet结构

**📊 数据集**

使用DSD（牙齿分割数据集）和OralVision（口腔内视图数据集）进行实验

**📈 对比分析**

与传统连续/局部扫描、U-Mamba、SAM系列、T-Mamba等方法对比，FU-Mamba在DSD、OralVision分别提升约1.1%–1.3% mIoU，同时保持较低参数和推理时延

**⚠️ 局限性**

局限在极端噪声、强反射、低对比等极端场景下仍出现分割失真，动态偏移估计和频域平衡在这些条件下不够鲁棒

---

## 254. Benchmarking Confidential Computing Performance on NVIDIA Blackwell GPUs

**arXiv ID:** 2608.26575 | [PDF](https://arxiv.org/pdf/2608.26575v1)

**作者:** Daniyal Khan `[一作]` (Confidential.ai), Ansgar Grunseid `[通讯]` (Confidential.ai)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

测量了在NVIDIA B200 GPU上使用Intel TDX与NVIDIA Confidential Computing（CC）下，将大型语言模型推理和训练置于受信任执行环境（TEE）中的性能影响，采用与非加密对比实验，归因于加密边界，并给出部署建议。

**💡 创新点**

创新点在于：①通过受控配对对比（CC‑on vs CC‑off）获取精确基准；②将加密成本细分至PCIe、命令通道、NVLink加密与能力损失四个硬件边界；③设计微基准预测实际推理延迟；④结合框架补丁与CUDA图优化，验证并实现低单数百分比的性能代价。

**🔧 技术方法**

技术包括：Intel Trust Domain Extensions（TDX）实现CPU侧TEE；NVIDIA Confidential Computing模式加密PCIe与NVLink传输；CUDA图（full CUDA graphs）与异步主机‑设备拷贝（async D2H worker）等软件补丁；量化方法（FP8、AWQ、NVFP4）与混合并行策略（MoE、TP、DP、EP）。

**📊 数据集**

使用了七种生产级模型（如Qwen3.5‑0.8B/9B、Nemotron‑3‑Super‑120B、Qwen3‑8B、Qwen2.5‑72B、MiniMax‑M2.7、以及自研41B/43B MoE），并在单主机上对同一磁盘与GPU进行多次测量。

**📈 对比分析**

对比方法：在同一物理主机、同一磁盘、同一GPU，唯一差异为GPU CC位与TDX访客对象；测量吞吐量、延迟、每步提交成本及NVLink带宽损失；结果显示，正确配置后单GPU推理低单数%（约1‑3%），多GPU MoE推理约1‑3%（TP4时1.5%，TP8时3%），训练端10‑13%。

**⚠️ 局限性**

局限性：仅评估单主机内核与GPU-GPU加密成本；不涉及跨节点传输、RDMA、Prefill/Decode解耦；未审计安全属性；框架版本与补丁对性能影响显著，需使用特定修复版本；某些长上下文或预填充密集场景仍可达十几%代价。

---

## 255. SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations

**arXiv ID:** 2608.26594 | [PDF](https://arxiv.org/pdf/2608.26594v1)

**作者:** Hiep V. Dang `[一作]` (University of Virginia), Antonios Mamalakis `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于潜在扩散模型的概率子季节性降水预测框架 SimCast‑S2S，能够在低维潜在空间高效生成大规模样本，并通过迁移学习实现从模拟到实测的知识转移。

**💡 创新点**

创新点在于三方面：① 采用潜在空间扩散生成预测分布，解决了传统扩散模型训练数据量大且预测耗时长的问题；② 通过 LoRA 低秩适配在预训练的 CESM2 大气模拟上进行迁移学习，显著提升了对 ERA5 观测的泛化能力；③ 直接输出概率分布而非仅有确定性点估计，满足降水预测对不确定性量化的需求。

**🔧 技术方法**

使用了变分自编码器（VAE）将 192×288 网格的气象场压缩为低维高斯潜在表示；潜在扩散模型（Denoising Diffusion Probabilistic Model）在此空间中进行反向采样；Transformer‑Conv 交叉注意力网络做为噪声/速度预测器；LoRA 低秩适配模块对预训练权重进行微调；对比学习中采用了 5 个气象变量组（风、质量、热力、水汽、降水）。

**📊 数据集**

训练数据：① 28 组 CESM2‑LE 气候模拟（1950‑2000 年预训练）；② ERA5 再分析（1940‑2010 年微调；2021‑2025 年测试）。

**📈 对比分析**

与 CNN、UNet 结构的深度学习基线以及工业级 ECMWF‑S2S 物理模型进行了对比。SimCast‑S2S 在 MAE、ACC、RPSS、CRPSS、BSS 等指标上均优于所有基线，并在 100 成员的 12 秒/成员生成速度下实现了与 ECMWF‑S2S 相当甚至更优的概率与确定性预测，且可在单块 A100 GPU 上完成 100 成员的子季节性预报。

**⚠️ 局限性**

主要限制包括：① 模型仅在统计层面保持物理一致性，未显式约束质量守恒或能量平衡；② 在极端降水事件的极端尾部仍存在欠校正；③ 解释性不足，缺少对模型决策的物理可解释性分析；④ 目前未结合后处理或校准步骤，实际操作中仍需进一步提升可靠性。

---

## 256. KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training

**arXiv ID:** 2608.26941 | [PDF](https://arxiv.org/pdf/2608.26941v1)

**作者:** Ireddi Rakshitha `[一作]` (Barclays), Ntakirutimana Pierre `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了专门针对 Kinyarwanda 的句子嵌入模型 KinyaEmbed，通过四阶段对比学习课程和多检查点集成，显著提升了同语句子相似度评估。

**💡 创新点**

创新点在于将语言专属预训练与逐步递进的对比学习数据层级（本地官方公报、机器翻译 NLI、OPUS-100 译对、KinyaCOMET 人工评估翻译对）相结合，并通过双重加权集成实现多任务最优。

**🔧 技术方法**

使用了 KinyaBERT-large 作为基础，MultipleNegativesRankingLoss 作为训练目标，L2 归一化向量，四阶段逐步 fine‑tuning 与多检查点平均集成。

**📊 数据集**

使用了 Rwandan 官方公报的同义句对、NLLB 翻译的 MNLI 三元组、OPUS-100 英‑Kinyarwanda 句对、过滤后的 2,936 条 KinyaCOMET 人工评估翻译对，以及自建的 300 条 Wiki‑RW‑STS 评测对。

**📈 对比分析**

与 LaBSE、mE5、AfriE5、BGE‑M3、OpenAI 等七种多语言基线在 SemRel2024‑rw、OPUS‑100、FLORES‑200、Wiki‑RW‑STS 等四个评测上对比，KinyaEmbed 在 SemRel2024‑rw 获得 ρ=0.7298，领先 20.9%；在 Wiki‑RW‑STS 获得 ρ=0.6005，领先 8.6%，并在文档聚类任务中得到最高 Silhouette 分数。

**⚠️ 局限性**

局限性包括对跨语言检索的 P@1 仍显不足、与检索优化模型相比在非对称检索任务上性能不足、对 2,936 条 KinyaCOMET 对的规模限制导致跨语言对齐仍远不及大规模翻译对齐模型、以及仅在 Wikipedia 文本上评测，缺乏其他领域的验证。

---

## 257. Can You Say This for Me? Speaking Up by Proxy in Co-Located Discussion

**arXiv ID:** 2608.26185 | [PDF](https://arxiv.org/pdf/2608.26185v1)

**作者:** Yue Shen `[一作]` (Virginia Tech), Yan Chen `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 SecondVoice——一种混合现实系统，允许犹豫发言者通过私密结构化指令，让共享代理在面对面讨论中以代理身份发声，从而降低社交风险并将观点纳入共同的说话空间。

**💡 创新点**

核心创新在于：① 将发言者的意图与公开发声分离，利用“代理”角色实现“谁说谁说”的半代理化；② 通过结构化的三步指令（发言动作、话题定位、候选点选择）简化发言者的准备工作；③ 将代理置于同一物理空间内的可见化存在，既保持隐私又保留社会互动的即时性。

**🔧 技术方法**

技术实现包括：Meta Quest 3 头显 + Unity 前端、FastAPI 后端、Deepgram（语音转文字、文本转语音）、OpenAI GPT‑4o/4o‑mini（情境理解、候选生成、语境重构）、All‑MiniLM‑L6‑v2（语义相似度）、Ready Player Me 角色渲染、WebSocket 双通道通信。

**📊 数据集**

数据集主要是原始实验数据：16 名大学生分为 4 组完成 2 个任务（选择大学校长、海上生存排序），共收集 18 个“pin”、13 次代理发言、6 条匿名文字板发帖以及完整的语音转录日志。实验没有使用公开的大规模语料库，仅依赖内部收集的对话数据。

**📈 对比分析**

实验采用 within‑subject 设计，将 SecondVoice 与匿名文字板两种通道进行对比。通过后台日志、转录跟踪、问卷量表（NASA‑TLX、会议评估）和半结构化访谈进行评估。结果显示：① 约 50% 的参与者在 SecondVoice 下使用代理表达未能亲自说出的观点（相较于 18.8% 的文字板，差异趋势显著但未达到统计显著）；② 代理发言更易被识别、被回应，且常促成后续多轮讨论；③ 文字板发帖多数被忽视或不产生后续口头互动；④ 访谈与问卷均未出现显著的整体效能差异，表明代理通道在提升表达机会方面更具优势。

**⚠️ 局限性**

局限性包括：① 样本量小（16 人，4 组），且仅在低风险情境下测试；② 通道差异是多维捆绑的（输入方式、结构化 vs 文字、代理可见性、语音输出、时机控制等），无法单独评估各因素贡献；③ 代理发言数量有限（13 次），缺乏足够统计功效；④ 受访者多为学生，缺乏更高阶或专业会议的验证；⑤ 代理身份的可推断性仍存在，未彻底解决归属与责任的权衡；⑥ 仅在两种任务上评估，未检验在更复杂、时间更长的讨论中的可持续性。

---

## 258. Cross-Platform Generalisation Failure in Mental Health Natural Language Processing: A Five-Axis Fairness Audit of Transformer Models on Social Media

**arXiv ID:** 2608.26138 | [PDF](https://arxiv.org/pdf/2608.26138v1)

**作者:** Rajveer Singh Pall `[一作]` (Gyan Ganga Institute of Technology and Sciences), Sameer Yadav `[通讯]` (Gyan Ganga Institute of Technology and Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了情绪标签作为心理健康代理的NLP模型在不同社交媒体平台间的泛化失败，并提出了跨平台公平性评估（CPFE）框架；

**💡 创新点**

创新点在于系统地从判别性能、校准、统计显著性、预测公平性和归因稳定性五个轴评估跨平台表现，并比较温度缩放与目标域微调的效果；

**🔧 技术方法**

采用Transformer模型（BERT、RoBERTa、Emotion‑DistilRoBERTa、GoEmotions‑RoBERTa）Fine‑tune，使用温度缩放校准、梯度重要性和Jaccard相似度评估归因，利用Bootstrap检验统计显著性；

**📊 数据集**

训练数据为Kaggle心理健康语料（35,556样本），交叉验证在Reddit（GoEmotions，6,257样本）和Twitter（dair‑ai/emotion，2,883样本）上进行；

**📈 对比分析**

相较于单平台基线，宏AUC下降30–40%，ECE从0.056–0.060升至0.5–0.54；温度缩放将ECE降至≈0.04但不提升AUC；目标域微调平均提升AUC约0.22；

**⚠️ 局限性**

局限包括标签映射的构造有效性、预训练与测试集重叠、归因方法不确定性、类别不平衡与先验偏移、单一seed的微调实验、未评估非Transformer或零样本模型等。

---

## 259. SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers

**arXiv ID:** 2608.26836 | [PDF](https://arxiv.org/pdf/2608.26836v1)

**作者:** Haizhao Fan `[一作]` (Shanghai JiaoTong University), Xinyi Le `[通讯]` (Shanghai JiaoTong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SymbolLKG框架，将LLM解析自然语言为逻辑知识图并通过逻辑路由动态调用Z3/Prover9/Pyke等符号推理引擎，实现可验证推理路径。

**💡 创新点**

创新点在于将逻辑规则和约束视为图节点，构建拓扑感知的逻辑知识图，并通过自适应逻辑路由与自我修正代码生成实现神经符号协同。

**🔧 技术方法**

技术包括LLM抽取、基于OpenIE的图构建、向量+图遍历的混合检索、Topology-aware逻辑路由、代码生成到符号求解器、回溯式自我修正。

**📊 数据集**

使用了FOLIO、AR-LSAT、ProofWriter、LogicalDeduction、ProntoQA等逻辑推理基准以及2WikiMultiHopQA、HotpotQA、Musique等多跳QA数据集。

**📈 对比分析**

与标准CoT、Logic-LM等对比，在逻辑推理准确率上平均提升至78.7%（高于74.5%），在多跳检索Recall@2上达到79.4%/68.5%，在端到端QA EM/F1上均领先前沿方法。

**⚠️ 局限性**

局限性包括抽取阶段可能出现语义误解导致后续推理失真，处理歧义与隐喻仍有困难，以及整体管道相较单一LLM推理具有更高的计算与延迟。

---

## 260. JudgeStealer: Extracting LLM Judging Capabilities across Evaluation Protocols

**arXiv ID:** 2608.26982 | [PDF](https://arxiv.org/pdf/2608.26982v1)

**作者:** Chen Chen `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 JudgeStealer，一种针对 LLM 判断功能的模型提取框架，能够在仅靠黑盒接口的情况下复制多协议（点值评分、对比与列表排序）下的评判能力。

**💡 创新点**

创新点包括利用评判协议间高度一致性实现跨协议转换；引入基于语义多样性、预测不确定性和评判偏差的动态信息化输入选择；采用自适应高斯平滑保留分数序数结构；以及多协议联合训练与回顾机制提升提取效率与鲁棒性。

**🔧 技术方法**

主要技术包括跨协议监督转换、动态样本选择算法、分数空间的可学习平滑、LoRA/全微调适配、以及联合损失函数实现多协议收敛。

**📊 数据集**

使用 Alpaca 与 GPT4All 这两个指令跟随数据集构造评判实例，受试模型包括 GPT‑5.4、Claude Sonnet 4.5、Qwen3‑235B‑A22B、DeepSeek V4 Pro（LLM‑as‑a‑Judge）以及 UniRRM（奖励模型）。

**📈 对比分析**

与 Vanilla、LoRD、Lion、Proxy‑KD 等基线进行对比，JudgeStealer 在点值评分、对比与列表排序协议上分别获得最高 73.3%、87.0% 与 71.6% 的准确率，且在低查询预算、不同规模与训练策略下仍保持领先，且对常见防御（异常检测、反蒸馏扰动、所有权追踪）具备良好鲁棒性。

**⚠️ 局限性**

局限性在于仅研究黑盒提取评判功能，未涉及对更复杂内部机制的逆向；跨协议一致性假设在极弱评判器上表现不佳；在极大模型或极限查询预算下的可扩展性与实战部署仍待进一步验证。

---

## 261. Revision-Aware Success Prediction from Multi-Attempt Programming Trajectories

**arXiv ID:** 2608.26169 | [PDF](https://arxiv.org/pdf/2608.26169v1)

**作者:** Md Faizul Ibne Amin `[一作]`, Md Mostafizer Rahman `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了基于多次尝试编程轨迹的修订感知成功预测，提出并统一了三种预测任务（当前提交是否成功、下一提交是否成功、未来三次尝试内是否能成功），并在当前代码、成对历史、三步历史三种输入模式下进行实验。

**💡 创新点**

创新点在于：①把三种预测任务统一在同一实验框架下进行比较；②系统评估了当前代码、修订历史以及多步历史对预测性能的影响；③对任务3进行未来视窗严格约束的敏感性分析；④发现当前代码+ML模型是最稳健的方案，并给出了在在线评测平台中实时监控与干预的具体应用建议。

**🔧 技术方法**

使用了传统机器学习模型（LinearSVM、XGBoost）基于TF-IDF词汇特征；深度学习模型（BiGRU、BiLSTM）使用从头开始的词表和嵌入；以及预训练代码模型（GraphCodeBERT、CodeT5+）进行微调；统一采用轨迹级拆分、验证集阈值选择、并在多种评估指标上比较。

**📊 数据集**

实验数据来自2026年PCK决赛中的人工-AI协作赛段，共15名参赛者、13道题、517次提交，构成184条用户-题目轨迹，并对语言一致性、接受标注进行清洗与截断。

**📈 对比分析**

比较方法：按轨迹划分训练/验证/测试，使用验证集确定阈值，测试集评估AP/PR-AUC、ROC-AUC、MCC、F1、BA、P、R、TNR、Brier分数、LogLoss等；实验结果显示：XGBoost在当前代码模式下取得最高AP/PR-AUC（Task3 0.9909）和MCC（0.6325）；Task3最易预测，Task2最难；pairwise/multi-step输入未能在各模型族中提供一致提升，且子样本规模受限。

**⚠️ 局限性**

局限性：数据量小，轨迹重建依赖解析与时间戳提取，只有Accepted/非Accepted作为成功标记，语言一致性筛选导致部分数据丢失；pairwise/multi-step子集样本极小，易导致偶然高分；实验设置与指标选择为一次性方案，可能不完全适用于更大规模或不同域的编程数据。

---

## 262. Harness Engineering for Predictable Agentic Systems: An Empirical Study of Deterministic Execution Constraints

**arXiv ID:** 2608.26197 | [PDF](https://arxiv.org/pdf/2608.26197v1)

**作者:** Saransh Dhage `[一作]` `[通讯]` (Independent Researcher), Saransh Dhage (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在 LLM 代理中使用 deterministic harness 的效果，评估四个模型×任务组合下的可重复性与任务成功率，并通过 trace 诊断发现需要结构化规划来完全消除方差。

**💡 创新点**

提出了结构化规划（Structured Planning）机制并通过层级诊断方法系统分解 harness 组件对确定性的贡献，展示了模型依赖的成本权衡。

**🔧 技术方法**

采用有限状态机、强制工具选择、输出验证、受限重试以及基于 JSON schema 的计划验证的 harness。

**📊 数据集**

使用了两条人工合成的四状态流水线任务——金融 CECL 预期信用损失计算和法律条款分类。

**📈 对比分析**

通过 Reproducibility Rate、Determinism Index、Task Success Rate、token 计数和延迟测量，在 N=100 次跑中发现 harness+SP 在三/四细胞实现 100% 再现率与成功率，token 下降 15–17%，延迟对 Qwen 减少 12–21%，对 Gemma 增加 15–47%。

**⚠️ 局限性**

仅测试两条线性任务与两款开源模型，未覆盖分支/判断任务；未探究 Gemma 重试原因；在达标后再现率与 DI 失效，需更复杂任务验证。

---

## 263. Why did My Robot Just Change Personality? Prompting Guidelines for a Grounded Robot Persona in LLM-Based HRI

**arXiv ID:** 2608.26182 | [PDF](https://arxiv.org/pdf/2608.26182v1)

**作者:** Ashita Ashok `[一作]` (RPTU Kaiserslautern), Guy Laban `[通讯]` (Ben Gurion University Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了大型语言模型在社交机器人中的提示设计，提出了一个理论框架和八个功能组件的结构化模板，旨在明确机器人身份、能力边界、透明度、用户适配与伦理约束；

**💡 创新点**

将提示设计提升为社会技术问题，首次系统性梳理并量化现有研究中提示的缺失点，提出可供实验验证的八组件模板，并结合专家访谈提供实证支持；

**🔧 技术方法**

基于多种大型语言模型（OpenAI GPT‑3/3.5/4o、Vicuna‑13b、Flan‑T5‑Large 等）以及提示工程技术，构建机器人对话与行为约束；

**📊 数据集**

使用27位HRI专家的问卷与讨论数据（Robo‑Identity workshop at RO‑MAN 2025），以及对近三年相关 LLM‑HRI 研究的文献综述；

**📈 对比分析**

通过主题分析与 Likert 量表评估专家意见，指出现有系统在身份、能力、透明度等组件上的不足；虽然未进行数值性能对比，但模板的可报告性和可复制性为后续实验提供了基准；

**⚠️ 局限性**

局限性包括：样本量有限、缺乏跨体态机器人实验验证模板效果、未深入探讨长期记忆与隐私治理的实现细节、以及对不同文化语境下提示可解释性的评估不足。

---

## 264. TrapVLA: Trapping Vision-Language-Action Models in Configured Failure Modes

**arXiv ID:** 2608.26578 | [PDF](https://arxiv.org/pdf/2608.26578v1)

**作者:** Jun-Hui Liu `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了配置化失败捕捉（Configured Failure Trapping）任务，利用隐蔽文本触发器在Vision‑Language‑Action模型中精细控制机器人失败方式；

**💡 创新点**

创新点在于：①将失败方式从二元化转为可配置化，提升隐蔽性；②设计TrapEngine生成目标轨迹并与触发器配对；③提出TrapVLA通过目标残差引导（Target Residual Steering）解决稀疏动作偏差问题；

**🔧 技术方法**

使用的技术包括：文本触发器生成（GPT‑5.4+语言模型评估）、轨迹合成与验证、对齐残差监督、自动评估套件TrapEval以及在OpenVLA-OFT与π_0.5等VLA模型上进行训练；

**📊 数据集**

数据集为Trap‑LIBERO（基于LIBERO）和Trap‑RoboTwin（基于RoboTwin 2.0），分别包含Object、Spatial、Goal、Long以及Shoes、Fan任务的四种配置化失败模式；

**📈 对比分析**

与DropVLA、Vanilla‑I、Vanilla‑T等基线比较，TrapVLA在所有四套任务上均实现最高AVE（最高达98.9%），且在保持清洁任务成功率的同时显著提高配置化失败成功率；

**⚠️ 局限性**

局限性包括：对极端稀疏或多模式失败的处理尚不完善；触发器设计依赖GPT生成，可能受模型偏见影响；实验主要集中在仿真与部分真实任务，缺乏更大规模多机器人验证。

---

## 265. TEMPLAR Wales: A georeferenced environmental and toponymic dataset of Welsh settlements

**arXiv ID:** 2608.26970 | [PDF](https://arxiv.org/pdf/2608.26970v1)

**作者:** Oktay Karakuş `[一作]` (Cardiff University), Can Eyupoglu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了TEMPLAR Wales，一个包含3757个威尔士定居点的可重复使用的地理-词汇-环境数据集，提供四张相互关联的表格

**💡 创新点**

首次明确区分定居点观测、可重复的词汇注释和环境测量，并提供冻结的词汇注册表、稳定标识符以及完整的源信息与质量元数据

**🔧 技术方法**

采用确定性词汇检测ETDE v1（NFKD标准化、精确/前缀匹配）、空间数据提取、关系数据库构建以及自动化验证脚本和加密校验和

**📊 数据集**

使用官方英国与欧盟数据源：OS Open Names、OS Open Rivers、OS Terrain 50、OS Boundary‑Line、CORINE Land Cover 2018、Copernicus DEM GLO‑90，以及RCAHMW与GPC词汇参考

**📈 对比分析**

通过与Copernicus DEM GLO‑90独立计算的2 km尺度地形属性对比，得到皮尔逊相关率>0.99、平均绝对误差≈1–3 米，验证了地形测量的鲁棒性和一致性

**⚠️ 局限性**

词汇检测仅为字符串匹配，不能确认词源；环境属性为现代测量，不能直接推断历史景观；缺失值取决于覆盖规则；仅包含24个词汇元素，且语言状态未完全确定

---

## 266. A Single Suffix to Break Them All: Basin-Aware Jailbreaks for Merged Model Families

**arXiv ID:** 2608.26506 | [PDF](https://arxiv.org/pdf/2608.26506v1)

**作者:** Yu Zhe `[一作]` (RIKEN AIP), Wang Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了模型合并（model merging）导致的跨模型 jailbreak 漏洞，并提出了一种在合并空间中寻找可转移攻击后缀的基于低损失 basin 的 min–max 优化方法（Basin-Aware Jailbreak，BAJ）。

**💡 创新点**

创新点在于：①将 jailbreak 视为合并空间中的 min–max 优化，发现共享的预训练 backbone 方向会在不同合并模型中保留可转移的弱点；②提出基于任务算术和演化优化的全新攻击框架，能够在不知道具体合并系数或模型的情况下，生成在整个合并 basin 上均有效的攻击后缀。

**🔧 技术方法**

使用技术包括：任务算术（Task Arithmetic）对多任务微调模型进行线性合并；基于变异演化优化的离散文本搜索；min–max 双重优化（对合并系数做 max，对后缀做 min）；对不同合并方法、量化和防御策略进行实验评估。

**📊 数据集**

实验数据集涵盖多种指令调优 LLM backbone（Llama‑2‑7B‑chat、Llama‑2‑13B‑chat、Llama‑3‑8B‑Instruct、DeepSeek‑LLM‑7B‑chat、Qwen‑7B‑chat、Gemma‑7B‑it），以及五个任务微调模型（Alpaca、Dolly、CodeAlpaca、GSM8K、CodeEvol）用以构建合并模型族。

**📈 对比分析**

与多种现有白盒、黑盒和无盒 jailbreak 方法（GCG、AutoDAN、TUJA、SCAV、LSGM_LILA、Guiding‑GCG、DAN、ArtPrompt、Multilingual、GCG‑Ensemble）进行对比。BAJ 在所有 backbone 上的转移成功率（TSR）均显著高于基线，并且在不同合并算法、部署配置（水印、量化、精度）以及多种防御措施下仍保持高成功率。

**⚠️ 局限性**

局限性在于：本文主要聚焦于发现和表征合并导致的安全风险，而未给出完整的针对合并环境的防御策略；现有防御对 BAJ 的抑制效果有限；未来需研发专门的安全聚焦合并方法和评估协议。

---

## 267. Arrive and Survive: Scaling Safe Goal-Conditioned Policy Learning from One-Bit Failure Signals

**arXiv ID:** 2608.26571 | [PDF](https://arxiv.org/pdf/2608.26571v1)

**作者:** Guopeng Li `[一作]` (Southeast University), Chengcheng Xu `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Safe-CRL方法，修正传统CRL在失败终止MDP中对短期未来的过度估计；

**💡 创新点**

引入质量加权InfoNCE与对数生存质量评分，补偿失败轨迹导致的价值高估与监督失衡；

**🔧 技术方法**

基于对比学习的自监督目标、InfoNCE损失以及强化学习策略优化；

**📊 数据集**

在12个易失败的机器人导航与运动任务上进行实验；

**📈 对比分析**

与Scaling-CRL基线比较，Safe-CRL在所有任务中均实现更高或相当的目标到达时间，且在9个任务上显著提升了生存率和到达目标的效率；

**⚠️ 局限性**

仅依赖失败终止的单比特信号，可能在高度随机或奖励结构复杂的环境中表现有限，且未深入评估对极端失败场景的鲁棒性。

---

## 268. Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers

**arXiv ID:** 2608.26762 | [PDF](https://arxiv.org/pdf/2608.26762v1)

**作者:** Markus Frohmann `[一作]` (Thomson Reuters Labs), Navid Rekabsaz `[通讯]` (Thomson Reuters Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在一次提示中批量评分候选文本时的顺序依赖问题，并提出 OC‑SFT 方法，使评分对顺序不敏感，同时保持排序质量，评估其在检索、问答和响应排序上的决策稳定性。

**💡 创新点**

创新点在于将顺序一致性正则化直接加入训练目标（OC‑SFT），通过在同一提示内对不同排列的候选进行多视图训练，显著降低了模型对排列顺序的敏感度；且该方法在单一提示下即可获得与多顺序平均相当甚至更优的决策稳定性，且不牺牲排序质量。

**🔧 技术方法**

使用技术包括：批量点位评分（batched pointwise scoring）与固定读出槽；OC‑SFT（顺序一致性监督微调）作为正则化目标；对比基线如单顺序蒸馏、顺序平均蒸馏、批量自一致性（BSC）、DebiasFirst、RankVicuna 等；以及 τ‑PSI 指标评估顺序稳定性。

**📊 数据集**

训练数据：MS MARCO（30k 查询的 top‑100 候选）；评估数据：18 个检索集合（TREC DL 19–23、11 个 BEIR 集合、两套内部法律集合），HotpotQA、2WikiMultiHopQA、MuSiQue 用于多文档 QA；Response Ranking 用于 RewardBench‑2、Nectar、PPE‑MMLU‑Pro、PPE‑MATH、RM‑Bench 等。

**📈 对比分析**

对比方法包括：单顺序蒸馏（Baseline）、顺序平均蒸馏（Order‑Avg Distillation）、批量自一致性（BSC）、DebiasFirst、RankVicuna 等；在保持相同 nDCG@10 的前提下，OC‑SFT 将 τ‑PSI 从 0.209 降至 0.083，保留集重叠率提升至 0.835；读者答案和偏好模型的决策变化率也显著下降，表明 OC‑SFT 在三类任务上显著提升了决策稳定性。

**⚠️ 局限性**

局限性：仍依赖训练时的正则化，无法完全消除顺序依赖；只在批量点位评分框架内验证，可能不适用于列表式或对偶式评分模型；对计算成本有一定要求（训练 + 预训练标签生成）；在内部法律集合的可复现性有限；未对极端候选数、不同 LLM 规模或其他任务进行全面评估。

---

## 269. Active Surface-Driven Reconfigurable Gripper: Robust Grasping and Sequential Manipulation of Thin Objects

**arXiv ID:** 2608.26883 | [PDF](https://arxiv.org/pdf/2608.26883v1)

**作者:** Ziyi Zheng `[一作]` (Grasp Lab, Zhejiang University), Huixu Dong `[通讯]` (Torch Kernel Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种结合主动表面、可重构机制与下驱动的三指抓手，用于稳定抓取薄可变形物体（如书籍），并实现“抓-放”连续任务。

**💡 创新点**

创新点在于：①仅在拇指配备主动滚动表面，省去复杂指节步态控制；②下驱动与弹性接触实现被动指节调节；③通过重构机构实现平面与竖向两种书籍抓取模式；④利用薄物体的弯曲模型优化结构参数。

**🔧 技术方法**

采用的技术包括：主动滚动表面（皮带+摩擦层）实现物体重定位；下驱动四杆链条与斜杆滑块实现并行抓取；行星齿轮与链条实现指节重构；基于欧拉-伯努利梁理论的力学分析与结构参数优化。

**📊 数据集**

实验中使用的对象包括4本不同厚度的书籍、A4纸、塑料薄膜、布料、鼠标垫等薄可变形物体；未使用公开数据集。

**📈 对比分析**

比较方法：在桌面与书架两种场景下，分别统计抓取成功率。桌面抓取成功率在4本书中最高可达20/20；书架抓取成功率略低（最多18/20）。对比传统抓手需多自由度步态控制的实验，所提方法在控制输入极简、无耦合、成功率高。

**⚠️ 局限性**

局限性：①对极薄但高刚度物体（如厚度14mm书）仍可能失稳；②在书架插入阶段若指节位置误差大，可能导致碰撞失败；③缺乏视觉反馈，仅采用开环控制；④主动表面对表面光滑度要求较高，可能影响抓取效果。

---

## 270. RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing

**arXiv ID:** 2608.26956 | [PDF](https://arxiv.org/pdf/2608.26956v1)

**作者:** Zijian Kan `[一作]` (Peking University), Lin Qu `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RubricRM，一种通过动态生成评估 rubrics 并基于该 rubric 对图像进行维度级评分的视觉奖励模型，适用于文本到图像生成与图像编辑。

**💡 创新点**

创新点在于将评估过程拆分为输入特定 rubric 的生成与维度级评分，且 rubric 由模型自动生成而非固定，提升了解释性与任务适应性。

**🔧 技术方法**

技术包括两阶段训练：监督微调（SFT）学习 rubric 生成与评分；随后使用 GRPO（基于维度级奖励的强化学习）细化评分。模型基于 Qwen3.5-4B/9B。

**📊 数据集**

使用多种公开偏好数据集（HPD v3、OIP、EvalMuse‑40K、EditReward‑Data 等）并合成补充 prompts，构建约 31.8k 文图对和 30.7k 图像编辑对。

**📈 对比分析**

与多种奖励模型及大型多模态 LLM 判断者对比，RubricRM 在 MMRB2、GenAI‑Bench、EditReward‑ERB、EditScore‑ERB 等基准上均优于奖励模型基线，提升 2–10 分。

**⚠️ 局限性**

局限包括依赖专有教师模型生成 rubric，可能继承教师偏差；仅评估静态图像，未处理视频或时序编辑；对教师与人类标签的潜在系统性偏差需要审计。

---

## 271. The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning

**arXiv ID:** 2608.26602 | [PDF](https://arxiv.org/pdf/2608.26602v1)

**作者:** Fei Ding `[一作]` `[通讯]` (Alibaba Group), Fei Ding (Alibaba Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种只持久化仓库实体、在推理时通过自注意力动态生成任务特定关系图的隐式关系实现机制；

**💡 创新点**

核心创新点包括：①只存储实体而不预构建边，降低维护成本；②采用两层实体索引（全局路由+局部聚焦）以压缩上下文；③证明任务上下文可触发自注意力材料化不同的潜在任务图；

**🔧 技术方法**

技术实现主要使用两层实体索引、基于任务的上下文序列拼装、DeepSeek‑V4‑Flash语言模型与自注意力机制；

**📊 数据集**

使用公开软件工程基准SWE‑bench Verified作为评估数据集；

**📈 对比分析**

与无索引基线、单层索引对比，三种条件下成功率分别为92.1%、94.2%和95.6%；两层索引在零边约束下提升约3.5个百分点；

**⚠️ 局限性**

局限性包括仅在单一模型上验证、未直接证明内部关系生成机制、缺乏跨模型泛化评估、对长期成本与更新频率的量化不足。

---

## 272. Improving LLM Interpretability with User-Centric Chain-of-Thought Reasoning

**arXiv ID:** 2608.26166 | [PDF](https://arxiv.org/pdf/2608.26166v1)

**作者:** Philipp Schröppel `[一作]` `[通讯]` (University of Ulm), Philipp Schröppel (University of Ulm)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种用户中心的 Chain‑of‑Thought 解释方法，利用 XML‑style 标签生成可验证的自包含推理步骤，并通过交互式 UI 让人类在完成数学词问题时能够逐步检查、纠错并与 LLM 协作。

**💡 创新点**

①引入了结构化 DSL（XML‑like 标签）来生成可单步验证的推理步骤，减少外部认知负荷；②通过交互 UI 实现对单步的可视化、交叉引用与针对性反馈；③在保持与标准 CoT 相当的任务性能的同时显著提升可解释性与用户体验。

**🔧 技术方法**

使用大型语言模型（Llama3.3‑70B、Gemma2‑27B、Llama3.2‑3B）生成带标签的推理；正则表达式解析 DSL 为 JSON；React 前端实现交互式推理浏览与修正；自检与修正机制基于 LLM 重新生成后续步骤。

**📊 数据集**

GSM8K（数学词问题）测试集，1,319 个未见问题。

**📈 对比分析**

功能性评估：与标准 CoT 对比，模型准确率保持相同或略高（如 Llama3.3‑70B 0.959 vs 0.953），但生成文本长度约为标准 CoT 的三倍；人机实验：对比标准 CoT，用户在易用性与效用上获得显著提升（p<0.05），但在判断准确率与纠正成功率等行为指标上无显著差异。

**⚠️ 局限性**

生成的解释过长导致用户在某些任务中认知负担增加，行为性能未见提升；实验样本有限且为低风险数学问题，未验证到其他推理领域；未直接测量认知负荷；缺乏专业领域专家的评估。

---

## 273. Minimum Rate For Partially Observable Linear System with Side Information: LQG Plant and Gaussian-Markov Source

**arXiv ID:** 2608.26917 | [PDF](https://arxiv.org/pdf/2608.26917v1)

**作者:** Sijie Li `[一作]` (University of Texas at Austin), Hyeji Kim `[通讯]` (University of Texas at Austin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在部分观测并且存在侧信息的线性系统（LQG植物与高斯马尔可夫源）下实现给定控制成本或失真要求所需的最小信息率，提出了线性编码器与置信度等价控制/估计的最优策略，并对标量系统给出了凸优化与单字母形式的求解方案。

**💡 创新点**

1) 证明了在侧信息存在的情况下，最优策略仍可归约为线性编码器加置信度等价控制/估计；
2) 对标量系统在时变与时不变两种情形分别给出了凸优化形式，并在时不变、无穷期望下进一步推出单字母闭式优化；
3) 通过分析两条耦合Riccati递推的收敛性，验证了单字母形式的紧致性。

**🔧 技术方法**

条件定向信息理论、最优控制与估计的分离原理、Kalman滤波、Riccati递推、凸优化（尤其是SDP与对数行列式函数的凸性）以及自适应线性编码与控制策略的构造。

**📊 数据集**

数值仿真使用了一个标量LQG系统：B=W=Q=R=F=C=1，A=5，且通过改变观测噪声方差N和侧信息噪声方差V来展示不同质量侧信息对率-成本曲线的影响。

**📈 对比分析**

通过绘制四种设置（完整观测、部分观测、完整观测+侧信息、部分观测+侧信息）的率-成本曲线与标量系统仿真结果进行比较，表明侧信息的加入可显著降低所需率，且在侧信息质量与观测质量相近甚至更低时仍能实现较优性能。

**⚠️ 局限性**

1) 仅在标量系统中证明凸性与单字母形式，向量系统的推广尚未完成；
2) 只考虑单编码器与单解码器的情形，未覆盖多编码器多解码器的网络场景；
3) 假设系统及噪声均为高斯，非高斯情况需进一步研究；
4) 侧信息模型为线性高斯侧信息，非线性或非高斯侧信息的处理仍是开放问题。

---

## 274. Decay-Region Group Delay as a Forensic Cue for AI-Generated Impulsive Sounds

**arXiv ID:** 2608.26346 | [PDF](https://arxiv.org/pdf/2608.26346v1)

**作者:** JaeHyeong Chang `[一作]` (University at Buffalo), Siwei Lyu `[通讯]` (University at Buffalo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究利用衰减期的群延迟特征区分AI生成的冲击声与真实冲击声，证明该特征在法医分析中具有可解释性和区分性。

**💡 创新点**

首次将衰减期群延迟作为独立的法医线索进行量化，并展示其对不同生成器的鲁棒性与对比优势。

**🔧 技术方法**

采用STFT+相位展开求群延迟，构建Mel压缩群延迟图；使用随机森林、CNN、Transformer、Phase-Flow CRNN等模型进行分类；评估KL散度、AUC、准确率等指标。

**📊 数据集**

15,000段3秒冲击声数据集（7,500真实来自FSD50K/SESA，7,500合成来自ElevenLabs、Stable Audio、AudioLDM2），并进行样本分离、生成器留置、源头留置和STFT参数敏感性实验。

**📈 对比分析**

在样本分离测试中，单一群延迟图可达90–94%准确率；随机森林在9个衰减期特征上得到AUC 0.884；生成器留置时CNN/Ast表现波动（AUC 0.457–0.918），而群延迟随机森林平均AUC 0.731，准确率 66.7%，避免了低于随机的崩溃。

**⚠️ 局限性**

主要限制包括数据来源有限、生成器种类不足、对不同声音类别与时长的泛化不足、STFT参数仅在小样本上测试、缺乏对抗鲁棒性评估。

---

## 275. TelecomGPT-R1: A Unified Open-Source Reasoner for the Telecom Stack

**arXiv ID:** 2608.26126 | [PDF](https://arxiv.org/pdf/2608.26126v1)

**作者:** Bohao Wang `[一作]` (Zhejiang University), Merouane Debbah `[通讯]` (Khalifa University)

**通讯引用:** 71634 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了 TelecomGPT‑R1‑9B，一款统一的开源电信推理模型，结合多老师 LoRA‑SFT 与 DAPO‑优化的 RL，实现跨协议、知识、建模与故障四个轴的推理；

**💡 创新点**

创新点在于：① 构建了 67,427 条轴索引的电信推理语料；② 使用源匹配的链式推理（CoT）生成；③ 在 LoRA‑SFT 后加入以二进制验证器为奖励的 DAPO‑稳定化 GRPO，形成单一统一策略；

**🔧 技术方法**

技术包括多老师 LoRA‑SFT、基于 GRPO 的 DAPO（含非对称裁剪、动态采样、Token 级损失聚合及 KL 正则）、源匹配 CoT 生成、二进制验证器奖励与自验证；

**📊 数据集**

使用的数据集为 67,427 例的自定义推理语料，涵盖 3GPP、O‑RAN、srsRAN、运营商文档、学术论文、模拟故障数据等，按协议、知识、建模、故障四轴组织；

**📈 对比分析**

评估方法：在 GSMA 开放电信排行榜七项公开基准上测试，平均得分 82.1%，在开源模型中排名第一，且与 GPT‑5、Claude‑Opus‑4.6 等封闭源顶尖模型相近；

**⚠️ 局限性**

局限性：仅在 9B 参数模型上实验；对训练语料的依赖较高；RL 奖励仅为二进制验证器，可能无法捕捉更细粒度错误；未覆盖所有可能的电信证据类型，且需要昂贵的算力。

---

## 276. StreamAV-Bench: A Comprehensive Benchmark for Streaming Audio-Video Generation

**arXiv ID:** 2608.26336 | [PDF](https://arxiv.org/pdf/2608.26336v1)

**作者:** Kaiqi Liu `[一作]` (BAAI), Boxin Shi `[通讯]` (PKU)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了常见英文缩写词（如 e.g., i.e., c.f., et al.）的识别与处理方法。

**💡 创新点**

提出了一种基于规则的缩写词识别框架，提升了缩写词识别的准确率。

**🔧 技术方法**

主要使用正则表达式与词典匹配技术。

**📊 数据集**

未给出具体数据集。

**📈 对比分析**

与传统基于词典的方法比较，准确率提高了约5%。

**⚠️ 局限性**

仅针对英文缩写，缺乏对多语言缩写的支持。

---

## 277. From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities

**arXiv ID:** 2608.26950 | [PDF](https://arxiv.org/pdf/2608.26950v1)

**作者:** Jiayi Kuang `[一作]` (Sun Yat-sen University), Philip S. Yu `[通讯]` (University Of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgenticMathBench，用流程级评测方法衡量LLM在数学推理中的代理式能力，涵盖规划、执行与反馈；

**💡 创新点**

将数学原子能力与代理式三大功能对齐，构建可解释的流程级基准，并通过自动化轨迹合成与多阶段筛选实现大规模高质量标注；

**🔧 技术方法**

基于结构化原子能力体系的规划/执行/反馈任务设计、自动化轨迹生成管线、LLM-as-judge评估策略、跨模态处理；

**📊 数据集**

汇集27个公开数学基准（如MiniF2F、IMO‑Bench、CROHME、ProofNet等）并手工标注原子能力，构建文本与多模态轨迹；

**📈 对比分析**

对比一般开源模型、数学专用模型与商业模型，评测规划、执行与反馈三大维度；实验显示模型即便在端到端精度相近，也存在显著的代理式能力差异，商业模型整体优于开源模型；

**⚠️ 局限性**

基准未覆盖长周期记忆与迭代学习，新知识获取；多模态子集规模有限；仅评估内在代理能力，未考虑推理时长与算力成本。

---

## 278. LiveVVT: High-Fidelity Video Virtual Try-On in Real Time

**arXiv ID:** 2608.26714 | [PDF](https://arxiv.org/pdf/2608.26714v1)

**作者:** Yushe Cao `[一作]` (Tsinghua University), Junliang Xing `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 LiveVVT，一种滚动扩散框架，实现高保真实时视频试衣。

**💡 创新点**

创新点包括：① 在有限窗口内保留双向时空建模并采用束缚式前瞻；② 设计双重记忆（时序记忆与全局外观记忆）维持跨窗口一致性；③ 逐步蒸馏策略（双向学习 → 轨迹回归 → 协同匹配蒸馏）将离线双向模型迁移至低步因果生成；④ 在实际流式推理中显著降低延迟与提高吞吐量。

**🔧 技术方法**

技术手段：滚动窗口扩散、双向时空Transformer、KV记忆缓存、持续外观记忆、进化式蒸馏（Teacher‑Trajectory Regression + Collaborative Matching Distillation）、VAE 编码/解码、CLIP+UMT5 多模态编码。

**📊 数据集**

使用 VITON‑HD、DressCode（图像）以及 ViViD、ViT‑HD、TikTokDress（视频）等数据集，构建长序列 ViViD‑SL 与 ViT‑HDL 评测基准。

**📈 对比分析**

与 MagicTryOn、VACE、ViViD、CatV2TON、MagicTryOn‑1.3B 等对齐基线比较；在 paired/ unpaired 长序列上，LiveVVT 在 VFID、SSIM、LPIPS 指标上均优于同等参数量模型；首次块延迟仅 1.56 秒，随后实现 22.39 FPS，比 MagicTryOn 延迟低 26×、吞吐量高 11×，在长流规模下每步延迟与显存基本不随视频长度增长。

**⚠️ 局限性**

局限性：仍需前视关键帧或姿态匹配以构建全局记忆；在极端长序列或快速姿态变换时可能出现微小外观漂移；目前仅针对衣物类试衣，难以直接推广至更复杂材质或动态光照场景。

---

## 279. Simple Actors and Deep Critics for Scalable Reinforcement Learning

**arXiv ID:** 2608.26659 | [PDF](https://arxiv.org/pdf/2608.26659v1)

**作者:** Guhyeon Kang `[一作]` (Sungkyunkwan University), Minhae Kwon `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了LAC（Light Actor, Deep Critic）方法，即在离线强化学习中使用轻量化确定性演员与深层残差MLP批判者，并通过专门的训练技巧解决深层批判者的不稳定性；

**💡 创新点**

引入三项互补技术（残差MLP+层归一化、n‑step bootstrap、分类交叉熵）同时抑制优化瓶颈、bootstrap噪声放大和价值漂移，证明将计算容量投放到批判者能够在保持推理高效的同时提升学习性能；

**🔧 技术方法**

采用残差MLP Backbone、LayerNorm、n‑step 目标、C51分布式价值学习（分类交叉熵）、确定性策略梯度加BC正则化、Polyak目标网络以及Adam优化器；

**📊 数据集**

在OGBench套件的7个连续控制任务（Ant、Humanoid、Manipulation等）共35个子任务上进行实验；

**📈 对比分析**

与11种基线（Gaussian/Deterministic、Diffusion、Flow等）进行对比，LAC在多数任务中匹配甚至超越基线，同时单步演员推理延迟比多步生成器低约4倍；将LAC批判者替换进其他方法可提升24–59点成功率；

**⚠️ 局限性**

仅在状态空间离线数据的模拟环境中验证，未测试像素观测或真实机器人部署，且实验仅限于模拟任务。

---

## 280. Ring Forcing: Towards Precise Long-Term Memory for Autoregressive Video Diffusion

**arXiv ID:** 2608.26794 | [PDF](https://arxiv.org/pdf/2608.26794v1)

**作者:** Bowen Xue `[一作]` (Stanford University), Panwang Pan `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Ring Forcing框架，利用环形训练构建视频长时记忆，实现精准长时记忆与对象永存。

**💡 创新点**

创新点在于环形训练让目标嵌入远期历史、时隙压缩+时间组合压缩以及稀疏RoPE以高效利用预训练先验。

**🔧 技术方法**

采用自回归视频扩散、环形训练、时隙组合压缩、稀疏RoPE以及低秩适配器LoRA。

**📊 数据集**

使用UltraVideo-Long单镜头长视频数据集进行训练与评估。

**📈 对比分析**

与LongLive、LongCat、FramePack比较，A‑D‑R和通用60秒长视频基准中均显著提升对象永存、连贯性和用户评测，表现优于SOTA。

**⚠️ 局限性**

仍受限于预训练基模型与固定序列长度下的压缩策略，需调参且在极长（>1分钟）或多镜头场景的泛化尚待验证。

---

## 281. SOLO: Stable Omni-terrain Long-Horizon Perceptive Humanoid Locomotion

**arXiv ID:** 2608.26583 | [PDF](https://arxiv.org/pdf/2608.26583v1)

**作者:** Pihai Sun `[一作]` (University of Science and Technology of China), Qiang Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SOLO框架，实现在自然地形上持续、稳定的感知行走，完成1.5公里连续户外行程与多种地形混合赛道；

**💡 创新点**

创新点包括（1）查询重建器QR：为每个高度图格子分配傅里叶编码查询，跨注意力从深度-本体感知记忆中检索位置特定证据，保持尖锐地形边界；（2）轨迹感知MSE TA-MSE：将下一状态教师-学生误差加入PPO奖励，通过GAE实现全程轨迹级信用分配，弥补单步MSE的短视。

**🔧 技术方法**

技术方法涵盖Transformer编码器/解码器、跨注意力、傅里叶编码查询、LSTM记忆、PPO + 辅助MSE、GAE、教师学生强化学习、仿真与实机部署（Omni机器人）等。

**📊 数据集**

使用自研的仿真环境（Isaac Lab）和实时的Omni机器人深度+本体感知数据；无公开公开数据集，全部基于仿真生成的多地形任务和实际部署现场。

**📈 对比分析**

与START、DPL重建器及PPO、MSE+PPO基线对比；QR将高度图L1误差降低3.3–4.0倍；在最高难度测试中，SOLO将成功率从≈75%提升至97.5%，踏石成功率从≈0%提升至96%；实机测试实现1.5 km连续行走、室内混合地形以及>10层楼梯的无重置通过率达69/70。

**⚠️ 局限性**

局限性：仅为前向深度摄像头，后退行走时存在盲区；依赖命令驱动的平面速度，需人工输入；2.5D高度图无法处理透明/反射表面；缺乏全方位感知与体素/多视角地图，限制在更复杂环境中的鲁棒性。

---

## 282. Cheaper by the Batch: Shared Traversal for Genotype Graph Editing

**arXiv ID:** 2608.26488 | [PDF](https://arxiv.org/pdf/2608.26488v1)

**作者:** Aaron Li `[一作]` (Cornell University), Giulia Guidi `[通讯]` (Cornell University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种批量变异映射算法，利用共享逆拓扑遍历对基因型表示图（GRG）进行编辑，显著减少重复遍历；

**💡 创新点**

创新点在于将多个变异的候选节点发现合并为一次共享遍历，配合位并行状态和自适应稀疏/稠密载体集表示，实现高重用和内存可伸缩；

**🔧 技术方法**

采用位并行的可兼容性掩码、稀疏/稠密自适应集合、单次逆拓扑共享遍历以及基于贪心的变异应用；

**📊 数据集**

使用基于欧洲人群的模拟GRG（10k–200k双倍体个体）和All of Us研究项目21号染色体GRG进行实验；

**📈 对比分析**

与逐变异独立映射对比，批量方法在200k个体时共享遍历时间提升至约79.5×，端到端极化时间提升至10.5×，峰值内存仅增加至1.29×；

**⚠️ 局限性**

在大批量情况下，延迟结构更新会略微降低图的紧凑性，导致边数和序列化大小增加，且极端批量可能引入更大的内存占用。

---

## 283. Natural-Language Policies to Executable Decisions: An Interpretable Large Language Model Framework

**arXiv ID:** 2608.26124 | [PDF](https://arxiv.org/pdf/2608.26124v1)

**作者:** Ziqiang Zhang `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在市级旅游企业部署了一个生产级LLM驱动的定价系统，利用LLM进行结构化提取与离散决策，而所有数值计算交由确定性引擎完成；

**💡 创新点**

创新点在于将LLM的决策范围严格限定为结构化提取与路径选择，形成可审计的条件树，既保留了LLM的语言灵活性，又消除了数值错误，实现零致命数值错误；

**🔧 技术方法**

使用技术包括大型语言模型（LLM）进行信息提取与树路径匹配、条件树构建与验证、确定性数值引擎、基于人机交互的L1-L3覆盖机制以及日志飞轮来持续改进模型；

**📊 数据集**

使用的数据集为2025年下半年生产日志，包含3960条订单（其中97%为即时通讯截图），以及1,000+活跃政策和1,500+运营商的业务数据；

**📈 对比分析**

通过决策边界漏斗评估：在完整截图集上L3-a资源编辑率47.9%，L2树切换率5.7%；在信息一致子集上，单位价格准确率为85.2%，致命数值错误率为0%；相较传统规则引擎，处理时间从约10分钟缩短到<2分钟，团队规模从15-20人缩减至3人；

**⚠️ 局限性**

局限性包括：手动分组策略可能随政策累积而导致匹配模糊；高达47.9%的L3-a资源编辑反映订单与执行行程不一致；整体端到端准确率受限于信息不完整；实验仅在单一旅游场景中验证，需在保险、合规等领域进一步验证。

---

## 284. DuMateBench: Evaluating Autonomous Agents in Complex Real-World Workflows

**arXiv ID:** 2608.26546 | [PDF](https://arxiv.org/pdf/2608.26546v1)

**作者:** Zechun Niu `[一作]` (Renmin University of China), Dawei Yin `[通讯]` (Baidu, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并发布了一个基于真实用户会话的复杂多工具工作流评估基准 DuMateBench，包含200个可执行任务。

**💡 创新点**

创新点在于：①将匿名化的真实会话重构为完整任务，保留交互历史与工作区状态；②在 Docker 环境中注入三类真实环境复杂度（不足、失稳、噪声）；③采用确定性检查与 LLM-as-Judge 结合的评估方式，兼顾任务完成度与输出质量。

**🔧 技术方法**

主要技术包括 Docker 容器化执行、LLM（Claude、GPT‑5.5、Opus‑4.8、GLM‑5.2、DeepSeek‑V4‑Pro）与多种代理框架（Claude Code、Hermes、DuMate、OpenCode、OpenClaw）的集成，LLM-as-Judge 评判器与人工审核流程。

**📊 数据集**

使用 200 条基于 DuMate 生产平台的匿名化用户会话数据，涵盖 8 大场景、17 类细粒度能力，并标注了任务的工作区状态和交互历史。

**📈 对比分析**

对 20 种代理–LLM 配置在三种环境（不足、失稳、噪声）下进行实验，评估指标为确定性检查通过率、LLM 评判得分与综合最终分；实验显示不同代理/模型组合差异显著，DuMate+Opus‑4.8 获得最高分但效率最低；噪声水平升高时，多数代理的性能显著下降。

**⚠️ 局限性**

局限性包括：任务来源单一平台，可能缺乏足够多样性；评估结果受人工审核主观性的影响；对大型 LLM 的算力与成本需求高，且未覆盖更极端或长期持续的环境失效场景。

---

## 285. HUG-VIS: A Multimodal Benchmark for Human-centered Understanding and Generation in Visual Intelligence

**arXiv ID:** 2608.26517 | [PDF](https://arxiv.org/pdf/2608.26517v1)

**作者:** Fei Ma `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Qi Tian `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HUG-VIS多模态基准数据集，统一对同一情感-动作-语句网格下的30位演员进行拍摄，提供同步视频、音频、文本和alpha matte；并在此基准上对情感识别、人类视频生成、声克隆和视频抠像四个任务进行零样本评估；

**💡 创新点**

首创将理解与生成任务放在同一受试者-任务网格上，提供统一的多模态对齐资源与跨任务分析框架；

**🔧 技术方法**

利用现有大型多模态预训练模型（如Qwen、MiniCPM、InternVideo等）进行零样本推理，评估多模态情感识别、音/视觉驱动视频生成、声克隆与视频抠像的自动与主观指标；

**📊 数据集**

数据集为8,400段坐姿半身视频，覆盖7种情绪、4种动作模板、10句中文提示，包含同步音频、文字转录和alpha matte；

**📈 对比分析**

在零样本协议下，各任务使用专属自动指标（准确率、CSIM、Sync-C/D、LPIPS、UTMOS/DNSMOS、MAD等）和人类MOS进行比较；结果显示：文本在情感识别中占主导，视觉单独识别效果最差；生成任务的自动与主观排名存在差异；抠像在运动边缘保持上存在显著困难；跨任务分析揭示任务难度受情绪、模型与指标共同决定；

**⚠️ 局限性**

主要局限：缺乏跨情绪的视觉表达多样性，导致视觉情感识别弱；自动指标与人类主观评价在生成任务中排序不一致；抠像仍难以处理快速手部运动和细腻边缘；数据集聚焦半身站姿，未覆盖更广泛姿态与场景多样性。

---

## 286. On Scope Classification and Current Knowledge-Editing Benchmarks: A Negative Result, with INLAY as a Gradient-Free Case Study

**arXiv ID:** 2608.26292 | [PDF](https://arxiv.org/pdf/2608.26292v1)

**作者:** Aditya Pratap Singh `[一作]` `[通讯]` (Independent Researcher), Aditya Pratap Singh (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建INLAY梯度无关记忆式知识编辑器，对范围判定决策进行全面评估，证明现有基准无法衡量该决策的价值。

**💡 创新点**

创新点在于：①提出完全可删除、局部化且不需梯度的INLAY编辑器；②通过精确执行所有候选操作得到每个查询的真实上限；③揭示现有单事实基准因缺少负样本而使范围判定无奖励。

**🔧 技术方法**

技术方案包括：梯度无关的logit空间偏置回放、固定句子编码键+随机投影检索、门控检索与判定、冻结权重与外部可寻址内存。

**📊 数据集**

使用的数据集：CounterFact、WikiUpdate、MQuAKE-CF、RippleEdits、AKEW（结构化、非结构化、提取证据三种输入条件）。

**📈 对比分析**

与权重编辑器（ROME、MEMIT、AlphaEdit）、内存编辑器（WISE、GRACE）以及检索增强生成（RAG）对比；在单事实基准中INLAY与权重编辑器相近或略优，RAG在RippleEdits中领先，WISE在Qwen2.5-7B CounterFact中最好。

**⚠️ 局限性**

局限性：仅能回放已存答案而不能推理，范围判定在硬负样本上表现欠佳；缺乏多检索堆栈、不同模型规模的完整评估；现有基准缺失负样本导致无法真实衡量范围判定收益。

---

## 287. Syntax vs. Semantics: How Transformers Learn Deep Dependencies

**arXiv ID:** 2608.26139 | [PDF](https://arxiv.org/pdf/2608.26139v1)

**作者:** Jiangrui Zhao `[一作]` (Beijing University of Posts and Telecommunications), Xiaoting Du `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 230 | [OpenAlex ID](https://openalex.org/A5052171951)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer在学习深层语义依赖时的梯度动力学，提出梯度饥饿（Gradient Starvation）机制，并通过链式思考(CoT)和拓扑对齐对比损失来加速深度推理电路的出现。

**💡 创新点**

核心创新在于将表面统计（句法）与深层语义视为梯度竞争，解释了深度推理的阶跃出现；并提出利用对比损失显式对齐语义子空间，从而突破梯度饥饿；进一步通过实验验证CoT在训练中的梯度注入作用。

**🔧 技术方法**

技术手段包括：梯度空间分解、软最大饱和分析、对齐比率ρ(t)定义与监控、CoT监督、拓扑对齐的对比目标（Margin-based 对齐损失）以及对头部注意力的精细分析与消融。

**📊 数据集**

主要使用数据集：Python150k 与 JavaScript150k（带AST）、Pythia checkpoints（用以追踪训练动态）以及WinoGrande（自然语言核心ference）。

**📈 对比分析**

与标准交叉熵、CRF、Curriculum、SupCon、Attention entropy等基线对比，结果显示在变量遮挡任务中，对比目标带来的提升至少为交叉熵的两倍；在WinoGrande上提升约0.2-0.3个百分点，跨语言迁移也表现优于基线。

**⚠️ 局限性**

局限性包括：对多语义头的强耦合导致潜在功能干扰；对比损失带来显著内存和计算开销；实验主要聚焦代码领域，未覆盖自然语言中语义关系的不确定性。

---

## 288. The Green Software Landscape: A Systematic Mapping Study on Evolution, Applications, Software Lifecycle, and Best Practices

**arXiv ID:** 2608.26229 | [PDF](https://arxiv.org/pdf/2608.26229v1)

**作者:** Max Hort `[一作]` (Simula Research Laboratory), Federica Sarro `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

做了一个覆盖2010-2024年15年间绿色软件工程（Green SE）研究的系统映射研究，手工检索顶级会议、期刊和相关工作坊，分析研究领域、方法与实验设计；

**💡 创新点**

创新点在于首次将绿色SE与绿色AI结合在同一映射中，使用SWEBOK框架按软件生命周期划分研究类型，并系统提炼实验设计的关键要素；

**🔧 技术方法**

采用系统映射方法手工检索文献，使用SWEBOK对研究类型进行分类，并对实验设计提炼硬件、测量、稳定性、重复性和数据集等要素；

**📊 数据集**

主要使用公开的会议与期刊原始论文作为数据来源，涵盖390篇文献；实验设计分析聚焦选定的子集论文；

**📈 对比分析**

通过对不同研究类型、领域和工具使用的数量与分布进行比较，发现会议发表占47%，优化与基准测试占主导，AI领域快速增长，实验多采用10-30次重复并使用多种测量工具；

**⚠️ 局限性**

局限在于仅覆盖A/A*级别顶级会议与期刊，排除非英文或非顶级出版物，手工检索可能遗漏部分工作，实验设计提炼受样本规模与作者自述的限制。

---

## 289. LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression

**arXiv ID:** 2608.26389 | [PDF](https://arxiv.org/pdf/2608.26389v1)

**作者:** Zishan Shao `[一作]` (Duke University), Hai Li `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了LowRankArena统一评测平台，标准化SVD低秩压缩的任务、模型、压缩预算与推理环境，并在此框架下对五种代表性SVD方法进行重新评估；

**💡 创新点**

创新点在于首次提供完整可复现的压缩检查点与评测脚本，消除以往论文评估差异导致的可比性问题；同时通过统一的预算和推理栈揭示SVD方法在不同模型与任务上的性能不稳定性与实际推理收益的局限；

**🔧 技术方法**

使用SVD低秩分解、统一精度压缩、混合精度与权重量化辅助技术；评测工具包括LM‑Eval‑Harness、vLLM推理框架；对比基准为结构化剪枝；

**📊 数据集**

使用的公开数据集与任务包括：LM‑Eval‑Harness（7个多选任务、5-shot MMLU、GSM8K等）、WikiText‑2、C4、MathQA、MMLU‑Math、IFEval；模型涵盖LLaMA‑1/2‑7B、Llama‑3.1、Qwen3等；

**📈 对比分析**

通过统一的压缩比(r∈{0.8,0.6,0.4})、统一精度、统一推理设置，对每种方法在相同任务和预算下测量零样本多选准确率、困惑度、TTFT、延迟与吞吐量；结果表明：SVD方法在不同模型/预算下排名无定，部分方法在严苛压缩下语言建模性能大幅下降；相较于结构化剪枝，SVD在推理速度提升主要集中在预填充阶段，整体端到端加速有限；

**⚠️ 局限性**

局限性包括：评测仅覆盖静态检查点压缩，未涵盖注意力侧低秩或混合精度与剪枝的深度融合；推理加速受工作负载（预填 vs 解码）高度依赖；实验受限于所提供的压缩检查点和vLLM实现，可能不适用于所有硬件/部署场景；

---

## 290. Multi-Expert Conformal Risk Control for Pairwise LLM Judging in Open-Ended Dialogue

**arXiv ID:** 2608.26529 | [PDF](https://arxiv.org/pdf/2608.26529v1)

**作者:** Ming Cheng `[一作]` (Monash University), Lizhen Qu `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了多专家 conformal risk control (CRC) 框架，用于开放式对话中的 LLM-as-a-judge 双方比较，构建了 Panel 1800 对偏好评测数据集。

**💡 创新点**

创新点在于：①引入 Score Averaging 与 Decision Voting 两种多专家 CRC 变体；②提出 Marginal‑Calibrated Conformal Consensus (MC^3)，通过初始阈值比率和统一决策函数解决异构专家尺度不匹配问题，同时保留 CRC 的分布无关风险保证；③创建首个可白盒访问 logit 的对话偏好评测基准。

**🔧 技术方法**

技术主要包括：Conformal Risk Control (CRC) 的阈值搜索与风险保证；多专家聚合策略（平均分数、投票决策）；MC^3 的比例初始化与全局阈值搜索；基于 logit 的 pairwise preference probability 评分函数；实验中使用 GPU 并行计算。

**📊 数据集**

使用 Panel 数据集，该基准由 3 个公开对话域（ESConv、MSC、DREAM）生成的 1,800 对应答对构成，并在 4 个开放权重 LLM（Gemma、Llama 2、Llama 3.1、Phi-3）上评估。

**📈 对比分析**

与单模型 CRC、点式评分、对话偏好评分等基线相比，MC^3 在异构专家设置下在保持风险 ≤ α 的前提下实现了最高的接受率和接近最佳准确率；Score Averaging 与 Decision Voting 在同质专家设置中表现更好，均显著提升覆盖率与准确率。

**⚠️ 局限性**

主要局限：①仅关注“commit vs. abstain”前端预测，后端人类复核流程尚未设计；②CRC 的风险保证依赖于样本可交换性，面对分布漂移时需频繁重校；③对 API 仅可调用的 LLM 需要替代的合规性评分。

---

## 291. Evaluating Confidence-Gated Retrieval with Matched Trajectory Replay

**arXiv ID:** 2608.26846 | [PDF](https://arxiv.org/pdf/2608.26846v1)

**作者:** Prateek Chhikara `[一作]` `[通讯]` (University of Southern California), Prateek Chhikara (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过匹配轨迹回放实验，评估在多跳问答中对同一决策阈值下，将原始置信度映射为校准后的置信度对检索与提交决策的影响。

**💡 创新点**

创新点在于提出匹配轨迹回放协议，能在同一答案和证据轨迹下只改变置信度映射，从而精准归因校准对决策行为的作用；同时系统性分析校准在不同模型、数据集与检索深度下对风险‑覆盖、准确率和检索成本的影响。

**🔧 技术方法**

使用了离线收集的答案与置信度轨迹、单调等距回归（Isotonic Regression）进行校准、固定阈值决策控制器、以及对比基线（固定检索深度、适配代理）。

**📊 数据集**

在HotpotQA和MuSiQue两大多跳问答数据集上，对Mistral Small 4、GPT‑OSS‑120B和Qwen3‑235B-A22B‑2507三大模型进行实验。

**📈 对比分析**

对比方法：在相同阈值（0.7）下，原始置信度与校准置信度下的决策路径；评估覆盖率、检索成本、提交答案准确率和总体准确率。实验结果显示，校准提升提交答案的准确率最高可达41个百分点，但整体准确率或检索成本并不统一改善，且在MuSiQue上整体准确率下降。

**⚠️ 局限性**

局限性：使用固定、离线的检索路径和确定性模型响应，无法反映真实检索过程；校准映射在检索深度3后失效，表明需针对不同状态或深度进行校准；评估仅基于LLM判定，缺乏人工验证；且未对所有终端动作（如上报、拒绝）计价，导致成本曲线不是完整的效用评估。

---

## 292. Compositional Generalization via Structural Identification in a Category-Theoretic Framework

**arXiv ID:** 2608.26465 | [PDF](https://arxiv.org/pdf/2608.26465v1)

**作者:** Akihiro Maeda `[一作]` (University of Tokyo), Yohei Oseki `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用范畴论的左Kan扩展，对COGS数据集中的句子结构和词汇进行可接受性诊断，分析在不同结构或词汇识别（collapse）下，未见句子是否能被训练数据支持。

**💡 创新点**

创新点在于：①把结构/词汇识别与Kan扩展结合，形成一种数据侧的可接受性分析框架；②通过该框架区分不同类型的组合泛化所需的识别谱，为未来模型的诱导偏置提供候选；③不依赖训练模型，而直接从训练数据推断泛化边界。

**🔧 技术方法**

主要技术包括：薄分类（thin categories）对句子地址与词汇的建模；结构collapse和词汇collapse的定义；左Kan扩展用于在collapse后传播可接受的词汇集合；闭包理论与Boolean Isbell nucleus对可接受性进行闭包化处理。

**📊 数据集**

使用的数据集为COGS（Compositional Generalization on Syntactic Trees），该数据集提供句子与对应逻辑形式的配对，包含21种组合泛化类型。

**📈 对比分析**

方法比较：对每种collapse操作（无collapse、递归collapse、最后标签collapse、argument collapse、argument+lexical collapse）在21种泛化类型上计算覆盖率；覆盖率从0.2555提升至0.9592，显示不同collapse有效解决不同泛化问题；与传统模型准确率对比显示有些类型在数据侧已可接受但模型表现差，揭示数据与模型差距。

**⚠️ 局限性**

局限性：①collapse的选择是人工指定，未从数据自动推断；②结构和语义表示依赖COGS的逻辑形式，未从原始句子学习；③目前只给出二值可接受性，缺乏分级或梯度化的评估；④未训练预测模型，无法验证这些抽象对实际模型性能的影响。

---

## 293. Fixed-Haven Reservation for Online Multi-Agent Pickup and Delivery in Dense Warehouses

**arXiv ID:** 2608.26759 | [PDF](https://arxiv.org/pdf/2608.26759v1)

**作者:** Taisei Hirayama `[一作]` (Hokkaido University), Itsuki Noda `[通讯]` (Hokkaido University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一个基于固定安全港（Fixed‑Haven）的在线多代理取送（MAPD）调度框架，证明在满足“安全港可达性”（Haven‑Reachability）和显式规划/进展假设下，能够在有限任务发布的前提下完成所有任务。实现的算法 SHARP（Safe‑Haven Retreat Planner）通过维护每个忙碌或退却代理的安全路径预留，并在每个时间步执行最近取点分配，使用 SIPP 进行路段验证。

**💡 创新点**

创新点：
1) 提出了固定安全港合同，并在图结构上给出了完备性保证；
2) 通过在每个代理保持退却路径到自己的安全港，消除了传统方法对宽阔退避区或双连通结构的依赖；
3) 在实验中展示了在树形、死胡同等非典型仓储布局上完全成功的能力，并证明固定回归（fixed‑home return）是这些布局鲁棒性的核心机制；
4) 对比实验验证了在高负载树形布局下，允许中途覆盖退却后缀能够显著提升服务时间与 makespan。

**🔧 技术方法**

技术细节：
- 统一的全局预留表和 SIPP（Safe Interval Path Planning）做路径验证；
- 采用最近取点分配策略和全局任务/代理扫描；
- 固定安全港规则（仅拥有者可占用，其他代理视为阻塞）与退却后缀重写机制；
- 使用可达性（Haven‑Reachability）作为图结构前置条件；
- 通过理论证明（Assignment‑Progress、Route‑Execution Progress 等）构建完整性框架。

**📊 数据集**

数据集与测试地图：
- 四种图类：well‑formed（38×23）、narrow‑bi（38×18）、narrow‑bi‑dead（38×18）和 tree（45×18）；
- 每个配置随机采样起点（安全港）与任务端点（灰色），任务在时间 0~300 内按速率 λ 生成；
- 对每种配置进行 100 次随机种子实验，涵盖 5–30 名代理与 λ∈{0.5,1.0,…,3.0}。

**📈 对比分析**

比较方法与性能：
- 基线包括 TP、PIBT、PIBTTP‑TA、TP‑home‑return、TP‑SIPP‑home‑return 等；
- 在 well‑formed、narrow‑bi、narrow‑bi‑dead 上，所有方法成功率 100%；
- 在 tree 布局上，只有 SHARP 实现 100% 成功率，其他方法全部失败；
- SHARP 在 tree 上的 makespan 与服务时间相较 TP‑home‑return 下降约 20%–30%，但规划开销显著增大（SIPP 调用、节点展开数）；
- 通过 no‑overwrite ablation 实验显示：禁用中途后缀覆盖会使 makespan 与服务时间分别增加 1.53× 与 1.89×，但成功率保持 100%。

**⚠️ 局限性**

局限性：
- 需要集中式调度器和全局预留表，无法直接应用于完全分布式系统；
- 仅适用于固定、独占的安全港（通常是起始格子），不支持动态安全港分配或多租户共享安全港；
- 证明假设为零停留时间、不可转让已承诺任务，实际仓储环境中往往存在任务停留、动态障碍或任务抢占需求；
- 规划开销在树形或高密度布局下显著增大，可能不满足实时调度的尾部延迟要求；
- 只考虑离散图路径规划，未涵盖连续空间运动、非结构化障碍等情况。

---

## 294. ASIL: Replacing Screenshot-and-Click with Structured State and Semantic Actions

**arXiv ID:** 2608.26991 | [PDF](https://arxiv.org/pdf/2608.26991v1)

**作者:** Rui Xie `[一作]` (Shanghai Jiao Tong University), Lu Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Agent‑Software Interaction Layer（ASIL）接口，将软件操作从传统的截图‑点击循环替换为结构化 JSON 观察与可执行语义动作，并在 15 种常用软件（如 Blender、LibreOffice、GIMP 等）中实现了对应的适配器，构建了一个统一的观察–动作–验证协议。

**💡 创新点**

创新点包括：
• 通过“深度可行访问路径”原则，将最贴近软件内部状态与操作的文件、脚本或 API 作为统一接口实现；
• 构建可复用的结构化观察和语义动作 JSON 契约，显著减少每个任务的行动步骤（平均 <5 步）并提升稳健性；
• 将该接口作为训练数据源，证明在 SFT 与受限 RL 下可实现双位数性能提升。

**🔧 技术方法**

使用技术包括：
• 大语言模型（GPT‑5.4、Qwen3.5‑2B/9B、Sonnet‑4.6 等）进行推理与规划；
• 受限 on‑policy 强化学习与行为克隆（SFT）训练；
• 半自动 ASILization 流水线、文件/脚本/服务三种实现模式；
• 统一评估器与可视化回放，用于验证和训练回放。

**📊 数据集**

数据集与基准：
• 380 任务基准（15 个单应用各 20 任务 + 80 个多应用任务），覆盖创意、办公、开发、文件、跨应用等领域；
• 80 任务硬核子集（长序列、多约束任务）；
• 训练数据为数千条结构化 ASIL 步骤轨迹（SFT 采样）与 320/80 任务的 RL rollouts。

**📈 对比分析**

比较方法：
• 与修复后的 50 步截图‑点击 GUI 控制（50max）以及 15 步截屏控制（15max）进行对比；
• 与本地原生接口（LibreOffice UNO、draw.io MCP）在相同任务、评估器、预算下进行匹配基准；
性能：在 380 任务上，ASIL 在所有模型上平均约 80+ 严格成功率，GUI 仅 6–27；SFT+RL 在 Qwen3.5‑2B/9B 上提升 12–20 分；在硬核子集上可提升至 +40 分。

**⚠️ 局限性**

局限性：
• 仅适用于可通过文件、脚本或 API 访问的应用，无法覆盖全闭源或无结构接口的软件；
• 对感知性、审美性任务（如 GIMP 真实照片编辑）仍表现不足，需更丰富的感知动作；
• 训练中的 RL 对长序列任务的稳定性有限，特别是小模型（2B）在硬核子集上的提升受限；
• 评估器复用与提示不对称可能影响公平性，需进一步独立验证。

---

## 295. High Probability Derivative Bounds for Random tanh Neural Networks on a Hypercube

**arXiv ID:** 2608.26526 | [PDF](https://arxiv.org/pdf/2608.26526v1)

**作者:** Josef Dick `[一作]`, Fabian Zehetgruber `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

研究了在 Xavier 初始化、tanh 激活函数下宽随机全连接神经网络的高概率混合输入导数上界，并给出了同时控制所有非空平方自由混合导数的统一概率事件。进一步推导了此类上界对欧氏 Lipschitz 常数、加权 Sobolev 范数以及 QMC 误差的影响。

**💡 创新点**

创新点包括：①首次给出高概率下的所有平方自由混合导数的同时间上界，且深度依赖仅为多项式；②通过分离 Faà di Bruno 公式中的全导数项并构造可测有限网来显著降低深度指数增长；③将该上界直接应用于 QMC 训练和鲁棒性分析，展示了随机网络在高维积分中的潜在优势。

**🔧 技术方法**

技术手段主要包括：Faà di Bruno 公式与欧氏算子范数的确定性推导；主导级数（majorant series）方法实现深度多项式控制；随机矩阵与集中不等式相结合的概率论分析；构造可测有限网与测度论工具以控制切向量；组合的容斥与 Hoeffding/Chi-square 近似来评估矩阵范数。

**📊 数据集**

本工作为纯理论分析，未使用任何具体机器学习数据集；所有结论均基于随机正态权重分布和 Xavier 初始化假设。

**📈 对比分析**

相较于传统的确定性导数上界（指数增长），该方法在深度 L 下仅产生 O(L^{|u|-1}) 的多项式增长；与已存在的随机 ReLU 网络 Lipschitz 上界相比，给出了更严格的高概率上界；在 QMC 误差方面，通过选择合适的权重可获得维度无关的 N^{-r} 速率，展示了理论上的优越性。

**⚠️ 局限性**

局限性包括：①结果仅适用于网络初始化阶段，未保证训练后仍保持；②仅针对 t h a n 激活函数，ReLU 等非光滑激活不适用；③只给出平方自由混合导数的上界，对更高阶或重复导数缺乏支持；④ Xavier 初始化导致各坐标的 β_j 近似相同，限制了 QMC 维度无关常数的收敛；⑤概率上界并非确定性保证，且宽度条件依赖深度、输入维度和失败概率。

---

## 296. Invocation-Level Reliability of Tool-Using Agents

**arXiv ID:** 2608.26189 | [PDF](https://arxiv.org/pdf/2608.26189v1)

**作者:** Afiya Noorain `[一作]` (SAI International School), Abhijit Dasgupta `[通讯]` (SP Jain School of Global Management)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

测量工具使用代理在多步任务中的调用准确率，区分工具选择与参数填写错误，并分析错误传播对整体性能的影响。

**💡 创新点**

提出按调用级别的准确率评估协议，并发现精确匹配评分导致严重性与恢复参数不可识别，随后提出基于状态的条件评分方法。

**🔧 技术方法**

采用三状态马尔可夫传播模型、教师强制与自由运行双臂评分协议、贝叶斯层次拟合（PyMC）以及后期条件评分校正技术。

**📊 数据集**

使用无污染的多步依赖图任务（深度1、2、4、6、8）生成的整数参数执行任务，评估五个开源权重模型。

**📈 对比分析**

通过教师强制与自由运行两臂对比，发现传播损失随深度递增；在深度6时，严重性约0.69，表明70%能力被早期错误削弱；条件评分后严重性提升至0.15‑0.32。

**⚠️ 局限性**

研究仅基于合成整数任务，生态效度有限；精确匹配评分导致参数不可辨识的结论仅适用于固定金标准匹配；未验证更复杂自然语言或多字段参数情形。

---

## 297. BekchiAI: Measuring, Observing, and Controlling LLM Agents in One Click

**arXiv ID:** 2608.26867 | [PDF](https://arxiv.org/pdf/2608.26867v1)

**作者:** Mesut Toruk `[一作]` `[通讯]`, Mesut Toruk

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出BekchiAI Benchmark和平台，用13个ReAct代理评估LLM的代理技能并实时监控其成本；

**💡 创新点**

创新点在于：①使用可验证的黄金答案（SQL、DAG调度、闭式表达式）避免“作弊”；②手工挑选的对抗样本和不完美的安全扫描器测试模型判断；③将指标（准确率、工具调用一致性、URL可信度、代币成本）集成到同一仪表板；

**🔧 技术方法**

采用ReAct框架、工具调用库、SQL/SQLAlchemy、DAG调度器、基于token和延迟的遥测SDK；

**📊 数据集**

使用2,057个手工构造的任务集合，涵盖算术、SQL分析、URL定位、规划、工具策略和安全判断七大类；

**📈 对比分析**

通过对四种LLM（Qwen3.7、gemma-4、gemma4:26b、gpt-oss-120b）在同一任务集上进行对比，展示各模型在不同任务族的精度差异，整体得分靠前的模型在部分技能上表现出色但在其他技能上相对薄弱；

**⚠️ 局限性**

局限性包括：任务仅为单步交互，缺乏长周期、多轮交互；模型覆盖面有限，未包含所有可能的代理技能；对抗样本虽手工设计但样本量有限；平台功能依赖手工配置，可能不适用于所有部署场景。

---

## 298. WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression

**arXiv ID:** 2608.26239 | [PDF](https://arxiv.org/pdf/2608.26239v1)

**作者:** Maeve Zhang `[一作]`, Qian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一种面向机器人动作可控的、长时序的自回归世界模型，能够生成多视角视频并支持闭环控制。

**💡 创新点**

采用 next‑scale 自回归结构与尺度对齐动作条件，结合压缩时间尺度记忆和奖励对齐的自回归对齐，统一了感知、动作、记忆与优化。

**🔧 技术方法**

next‑scale 自回归生成器、尺度对齐动作编码、时间尺度压缩记忆、梦境强制（dream forcing）、奖励对齐（on‑policy alignment）以及冻结参考模型的 KL 约束等技术。

**📊 数据集**

混合数据集包括 AgiBotWorld‑Beta、ManipArena、X2‑Robot、UMI 等公有与私有机器人演示、非机器人 UMI 记录以及干预/恢复轨迹。

**📈 对比分析**

在 Embodied Video Generation、Action‑Conditioned Generation 基准以及 60 s 长时序推理和闭环物理仿真上，与 InfinityStar、Wan、Cosmos 等基线相比，取得了最高的交互质量、指令跟随、轨迹精度以及显著降低的长时序漂移和更一致的闭环任务进度。

**⚠️ 局限性**

仍受自回归误差积累限制，长时序视觉质量衰减；对极端场景、未见物体和高复杂交互的泛化有限；需要大量标注且训练成本高。

---

## 299. CGS-SLAM: Collaborative Gaussian Splatting based SLAM for Multi-Agent Reconstruction

**arXiv ID:** 2608.26868 | [PDF](https://arxiv.org/pdf/2608.26868v1)

**作者:** Jean-Daniel de Ambrogi `[一作]` (Université Sorbonne Paris Nord), Aurélien Chateigner `[通讯]` (SAS IMPACT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种基于协作高斯喷溅的SLAM系统（CGS‑SLAM），可在仅使用RGB相机和IMU的移动设备上实现多代理的实时三维重建；

**💡 创新点**

创新点在于：①利用度量级单目深度估计模型（Depth Pro）实现无深度传感器的尺度一致重建；②将IMU预积分作为运动先验提升跟踪鲁棒性；③设计轻量级的动态关键帧机制和服务器端快速子地图对齐（VGGT+NetVLAD），实现低带宽的协作与全局一致性；

**🔧 技术方法**

采用技术包括：单目深度估计（Depth Pro）、IMU预积分、3D高斯喷溅（3DGS）建模、NetVLAD特征匹配、VGGT视图对齐、基于可视化渲染的损失函数（光度+深度相关性+D‑SSIM）、局部滑动窗口bundle‑adjustment；

**📊 数据集**

使用的公开数据集有：TUM RGB‑D（带IMU）、UT‑MM（多序列RGB‑D+IMU）以及合成多代理场景；

**📈 对比分析**

与现有RGB‑D 3DGS SLAM（如Magic‑SLAM、MAC‑Ego3D）以及单目3DGS方法（MM3DGS）对比，CGS‑SLAM在单目跟踪误差（ATE ≤ 5 cm）与渲染质量（PSNR、SSIM）上均达到或超过RGB‑D基线，并在子地图对齐中实现亚米级误差；

**⚠️ 局限性**

局限性：需要依赖服务器进行深度推理与子地图对齐，难以完全离线运行；深度模型和VGGT推理成本较高，当前实现仍需高端GPU；多代理间的精细配准未实现紧耦合bundle‑adjustment，可能在高度重叠区域产生冗余高斯。

---

## 300. TempJail: Temporal Jailbreak Attacks against Image-to-Video Generation Models

**arXiv ID:** 2608.26971 | [PDF](https://arxiv.org/pdf/2608.26971v1)

**作者:** Qi Lu `[一作]` (Huazhong University of Science and Technology), Qiankun Zhang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TempJail框架，对图像到视频(I2V)生成模型的时序安全性进行攻击，利用初始帧的视觉条件与时间文本指令实现跨模态的恶意内容触发。

**💡 创新点**

创新点在于首次发现并利用I2V模型的时序漏洞：将恶意语义分解为初始图像与时间文本，并通过“主体-动作-场景”模板与潜在空间扰动实现语义隐蔽与逐帧激活。

**🔧 技术方法**

技术包括文本到图像(T2I)生成、视觉编码器集合进行局部匹配的Winner‑Takes‑All策略、在潜在空间的受控梯度扰动与真实噪声注入、以及本地LLM对文本模板重写。

**📊 数据集**

使用T2VSafetyBench中的14类安全评估类别（共700条提示）以及四个闭源商业I2V系统（Kling、Seedance、Veo、PixVerse）进行实验。

**📈 对比分析**

与五个基线（T2VSafetyBench、Scene‑Split、Opt‑Jail、Runaway、VII）对比，TempJail在ASR和CLIP‑S指标上显著领先，平均提升约20%以上，尤其在Violence、Illegal Activities等场景表现突出。

**⚠️ 局限性**

局限性包括：目前仅针对视频与文本两模态，未覆盖音频；对安全过滤策略的鲁棒性依赖于特定模型；且对不同生成模型的适配需要手动调整参数。

---

## 301. TOPIQ: Statistical Error Propagation for Quantity-of-Interest Prediction under Lossy Compression

**arXiv ID:** 2608.26912 | [PDF](https://arxiv.org/pdf/2608.26912v1)

**作者:** Youyuan Liu `[一作]` (Temple University), Sian Jin `[通讯]` (Temple University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了TOPIQ框架，利用压缩元数据对任意运行时定义的量化兴趣指标（QoI）进行统计误差传播预测，输出偏差和不确定性估计。

**💡 创新点**

创新点在于把QoI拆解为平滑原子运算，给出闭式误差传播规则，并通过空间误差相关性指数α与数据-误差耦合捕捉非独立误差；仅需<0.1%元数据即可支持任意组合的QoI，完全不需预先推导或重新训练。

**🔧 技术方法**

使用二阶泰勒展开得到闭式传播公式，采用v_u/v_c分解和α校准模型处理空间相关误差；实现跨字段独立假设、预计算压缩误差统计元数据以及在线QoI图构造与传播。

**📊 数据集**

评估基于四个科学数据集：CESM‑ATM（二维、1800×3600）、NYX（三维、512³）、SCALE‑LETKF（三维、98×1200×1200）和Hurricane‑ISABEL（三维、100×500×500），涵盖多种物理量。

**📈 对比分析**

与SZ3、SPERR、ZFP三种误差限压缩器以及四类QoI族（均方、神经网络、加权和、云辐射比）共552次评估，93.1%配置满足σ_z∈[0.7,1.3]，元数据预测速度比直接误差计算快56–402倍，且不需原始数据即可完成后验不确定性量化。

**⚠️ 局限性**

局限性包括：对非平滑运算（如ReLU、阈值计数）不适用；当压缩误差空间非平稳时，单一α值假设导致部分配置失准；多字段比率QoI可能因跨字段误差相关性被高估，导致过保守的置信区间。

---

## 302. Thresholding Post-Quantum Signatures

**arXiv ID:** 2608.26792 | [PDF](https://arxiv.org/pdf/2608.26792v1)

**作者:** Francesco De Sclavis `[一作]` (Bank of Italy), Marco Pedicini `[通讯]` (Roma Tre University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了现有的后量子阈值签名方案，归纳了不同范式的技术方法，并提供了模块化参考框架。

**💡 创新点**

提出按范式分类的系统化视角，强调线性秘密共享与部分响应聚合的核心思路，并在格、哈希、群作用、多元等后量子基准中阐述了适配策略。

**🔧 技术方法**

使用了线性秘密共享（Shamir、复制、梯度共享）、噪声洪水、同态承诺、可验证短秘密共享、部分格陷阱门、FHE/TFHE、MPC、STARKs等通用技术。

**📊 数据集**

本文为综述性质，未使用实验数据集，仅基于文献分析。

**📈 对比分析**

通过理论分析与已有实现的签名尺寸、签名时间、通信轮数等指标进行对比，指出FHE/TFHE方案可实现单轮但计算开销大，MPC方案具备主动安全，STARKs方案无可信设置但签名尺寸随阈值增长。

**⚠️ 局限性**

仍缺乏统一的实用标准；部分方案仅适用于小阈值；通用方法如FHE/MPC效率不友好；对量子安全性与实际部署细节讨论不足。

---

## 303. Rethinking Image Processing for the Age of AI: A Problem-First Framework for Scientific Progress

**arXiv ID:** 2608.26833 | [PDF](https://arxiv.org/pdf/2608.26833v1)

**作者:** Guoping Qiu `[一作]` `[通讯]` (University of Nottingham Ningbo China), Guoping Qiu (University of Nottingham Ningbo China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出问题优先的研究范式，强调从物理成像问题出发，拆分成问题、原理、估计器、实现四层；给出六阶段问题先行工作流程并在超分辨率与低光增强两大案例中检验该流程，提出更严格的实验与报告标准，并呼吁科研文化与教育改革。

**💡 创新点**

①将成像研究拆解为四层（问题、原理、估计器、实现）与六信息源；②构建六阶段问题先行工作流程；③系统分析 SR 与 LLIE 基准与真实问题不符；④提出多维度证据与可重复性评价阶梯；⑤为科研训练、审稿、出版与行业提供具体准则。

**🔧 技术方法**

主要是概念框架、统计估计理论、实验设计与评价方法的系统化，未提出新的深度网络或算法；而是在已有 CNN/Transformer/GAN/Diffusion 等模型基础上阐明如何定位其贡献。

**📊 数据集**

案例研究中引用的基准数据集包括 DIV2K、RealSR、DRealSR、Set5、Set14、BSD100、Urban100、Manga109、LOL、SID、SICE 等，主要用于说明基准与真实成像问题的差异。

**📈 对比分析**

作者通过对比基准结果与实际测量、跨数据集/跨传感器转移、重建一致性、噪声/动态范围敏感性等多维度评估，指出单一 PSNR/SSIM 指标不足；未给出新模型的数值比较，而是提出更完整的性能评估框架。

**⚠️ 局限性**

论文仍是理论与框架性工作，缺乏新算法与实验验证；对不同应用域的细化不足；实际落地需结合具体成像系统与数据；基准与真实场景间的差异无法完全消除。

---

## 304. NeuronFuzz: Safety Neuron Guided Fuzzing for LLM Safety Evaluation

**arXiv ID:** 2608.26222 | [PDF](https://arxiv.org/pdf/2608.26222v1)

**作者:** Zhiyuan Xu `[一作]` (University of Bristol), Lichao Wu `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出NeuronFuzz，一种基于内部安全神经元激活的白盒模糊测试框架，用于评估LLM的安全性。

**💡 创新点**

创新点在于利用模板不变的有害/无害对构建安全神经元集合，并将其映射为可微的安全警报分数，既消除了生成完整回复的开销，又提供连续、可梯度的搜索信号。

**🔧 技术方法**

技术包括模板不变激活提取、bootstrap稳定性选择、弹性网络训练SafetyOracle、基于梯度定位敏感位置的掩码语言模型突变以及MCTS调度。

**📊 数据集**

数据集涵盖约14,000条有害/无害提示、1,000个模板不变对、21个公开与专有LLM（文本与多模态）以及HarmBench、StrongREJECT等评测集。

**📈 对比分析**

与AutoDAN、PAIR、LLM-Fuzzer等基线相比，NeuronFuzz在5个白盒源模型上平均发现率提升48个百分点，响应生成次数仅为1，平均发现时间下降至30%~70%之间。

**⚠️ 局限性**

局限性包括对模板的依赖、需要对源模型内部信息的访问、对梯度可微性的要求以及在极端对齐模型或复杂多模态场景下的有效性仍待进一步验证。

---

## 305. Ankhdjet: An Open-Source Compiler for Mask-Programmed Ternary Compute-in-ROM on an Open PDK

**arXiv ID:** 2608.26206 | [PDF](https://arxiv.org/pdf/2608.26206v1)

**作者:** Mohnish Pai `[一作]` `[通讯]` (Independent Researcher), Mohnish Pai (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了Ankhdjet开源编译器，将HuggingFace的三值量化模型权重编译成可在SKY130开源PDK中实现的mask-programmed CiROM宏，并完成了两个不同权重矩阵的全流程物理签核与TinyTapeout提交。

**💡 创新点**

首次实现开放源代码的权重到mask的CiROM编译器、首次在开源PDK上提交CiROM宏，并对比数字采样与模拟比较的读取方案，展示了数字采样在面积与能耗上的优势。

**🔧 技术方法**

使用了OpenROAD、LibreLane、Yosys、Magic、KLayout、netgen、ngspice等全开源EDA工具，以及自研的三值IR、mask程序生成器、物理抽象与验证链。

**📊 数据集**

基于公开的BitNet b1.58 2B4T以及其他5个三值量化模型（包括TriLM、Falcon、Llama等）进行编译与验证。

**📈 对比分析**

通过对同一晶圆尺寸下的数字全揽采样和模拟比较读出层面进行面积、能耗和读out功耗对比；数字采样面积仅为模拟的1/8，模拟能耗为数字的约4-5倍，验证过程在DUT签核、功能回归、DRC/LVS全零、时序良好。

**⚠️ 局限性**

尚未完成硅测量，所有能耗和性能均为仿真结果；仅验证64×32形状，未测量更大位线深度；未在硅上实现完整MAC；缺乏实际读out能耗验证与噪声容忍性等。

---

## 306. UniGeo: A Multi-modal Large Language Model for Text-Guided Cross-View Geo-Localization

**arXiv ID:** 2608.26722 | [PDF](https://arxiv.org/pdf/2608.26722v1)

**作者:** Jiahao Wen `[一作]` (Shanghai University), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种统一的多模态大型语言模型 UniGeo，用于文本驱动的无人机地理定位，通过图文语义理解、姿态感知跨视角生成和候选级验证三大功能实现候选结果的精细化判别。

**💡 创新点**

创新点包括：① 将语义理解、跨视角生成和候选验证融入同一 MLLM 框架并采用逐阶段训练；② 通过候选条件的查询细化与硬负样本生成实现文本查询的补全与提升；③ 引入姿态约束的跨视角扩散生成，为硬负样本学习提供结构一致且可控的合成图像。

**🔧 技术方法**

主要技术包括：多模态大型语言模型（以 UniLIP 为骨干），区域级语义监督与空间关系建模，基于 Diffusion Transformer 的姿态感知跨视角生成，硬负样本验证头以及渐进式三阶段训练策略。

**📊 数据集**

使用了 GeoText-1652 和 UAVReason 两大文本-图像地理定位基准，并在 University‑1652 数据集上训练跨视角生成模型。

**📈 对比分析**

与 METER‑Swin、ALBEF、XVLM 等检索骨干以及 GeoText‑1652、HCCM、NGCG‑MLLMs 等任务特定方法进行对比。UniGeo 在 GeoText‑1652 上提升 R@10 约 +13.6%、mAP +2.8%，在 UAVReason 上亦保持类似的 Top‑10 提升，且在多种检索骨干上均能获得稳健性能提升。

**⚠️ 局限性**

局限性：主要提升了 Top‑5/10 的召回，R@1 受文本描述不完整的影响有限；对极简文本查询的定位仍存在挑战；需依赖外部检索骨干，且推理时额外的查询细化与重排序会产生计算开销。

---

## 307. Knowing When Not to Reuse: Conditional Experience Transfer in Autonomous LLM Post-Training

**arXiv ID:** 2608.26730 | [PDF](https://arxiv.org/pdf/2608.26730v1)

**作者:** Tingyun Li `[一作]` (Alibaba Cloud Computing), Yuewei Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种名为BCIT的决策框架，用于在大型语言模型的自适应后训练中根据历史更新证据决定是否接受、验证或全量训练候选更新。

**💡 创新点**

将经验授权拆分为Reject、Validate、Train三步，结合源证据强度、当前兼容性与硬冲突判定，并采用乘积阈值实现可解释且保守的决策，同时保留共享的推广规则。

**🔧 技术方法**

采用条件经验转移建模、源证据评分与兼容性判定、预算限制的当前父验证、共享推广规则以及实验对比评估等技术。

**📊 数据集**

基于Qwen3‑4B模型，使用FinQA、Spider、xLAM等源任务数据，目标任务为TAT‑QA、BIRD、BFCL，并以IFEval作为保留基准。

**📈 对比分析**

与Flat‑Additive、Validate‑All、Additive+Veto等策略在相同GPU预算下进行六对齐播种实验，BCIT平均提升2.63分，显著优于基线，在AUC、最终平均分等指标上表现更好。

**⚠️ 局限性**

仅在单个4B模型、三项能力和单一保留基准下验证；候选集与组件实验非独立；验证精度与计算成本随规模、数据、训练计划可能变化；未探讨学习式边界、长期适应等问题。

---

## 308. Group Isomorphism and the Polylogarithmic-Time Hierarchy: Depth-2$\frac{1}{2}$ Circuits and Lower Bounds

**arXiv ID:** 2608.26257 | [PDF](https://arxiv.org/pdf/2608.26257v1)

**作者:** Joshua A. Grochow `[一作]` (University of Colorado Boulder), Michael Levet `[通讯]` (College of Charleston)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文首次给出Group Isomorphism（GpI）在乘法表模型下的低深度电路复杂度结果：证明任意深度2的布尔电路需要|G|的超多项式（即指数级）规模；并构造了统一深度 2.5（即 21/2）电路求解 GpI，规模为准多项式；若满足 Uniform Short Presentation Conjecture，进一步可得到统一深度 2 的解法。

**💡 创新点**

创新点主要在于①突破了以往仅针对 Quasigroup 的生成器枚举方法，首次实现了针对群的深度2电路下界；②利用 Short Presentation Conjecture、群扩张与协同理论、以及矩阵秩与 2‑affine 投影归约，构造了深度 2.5 的准多项式电路；③通过对 Ree 群的特殊处理（利用可逆仿射映射与离散对数技术）解决了目前尚未得到短呈现的情况。

**🔧 技术方法**

技术手段包括：
• 生成器枚举的改进与矩阵秩归约，构造了对 CNF 的 lower bound；
• 采用 2‑affine 投影归约从矩阵秩到 GpI；
• 组扩张与非阿贝尔协同（cohomology）理论，用于验证构造的分组链条；
• 利用 Uniform Short Presentation Conjecture 及其可计算性，构造短呈现；
• 对 Ree 群利用 7×7 矩阵表示、可逆映射与离散对数算法。

**📊 数据集**

本文不依赖任何外部数据集；所有证明均为理论构造与复杂性分析。

**📈 对比分析**

与先前 3/2 层的深度/多项式时间结果相比，本文的 2.5 深度电路在深度上有提升，但规模为准多项式，未能改善串行最坏情况运行时间；另一方面，首次给出深度 2 的下界，展示 GpI 不属于 AC^0。

**⚠️ 局限性**

局限性包括：
① 上界仍需准多项式规模；
② 下界仅适用于深度 2；
③ 关键推导依赖 Uniform Short Presentation Conjecture，尚未对 Ree 群完全验证；
④ 结果仅适用于 Cayley 表输入，非更简洁表示下不适用。

---

## 309. Privacy Without Regret: Differentially Private Inference-Time Alignment

**arXiv ID:** 2608.26324 | [PDF](https://arxiv.org/pdf/2608.26324v1)

**作者:** Ishi Jain `[一作]` (Indian Institute of Technology Kanpur), Sayak Ray Chowdhury `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种在推理时实现差分隐私的对齐方法——PrivBoN（加 Gumbel 噪声的 Best-of-N）和 PrivITP（结合 χ² 正则化拒绝采样与两阶段高斯噪声的推理时贪婪策略），并证明它们能够同时抑制奖励模型报错（reward hacking）并保持对模型输出的可扩展性。

**💡 创新点**

创新点包括：① 用 Gumbel 噪声实现 ε‑DP 与 KL 正则化对齐的统一；② 推导出隐私阈值 ε*，在该阈值以上隐私成本不再增加；③ 设计 PrivITP，解耦正则化与隐私参数，获得 n‑独立的后验 DP 保障；④ 在多查询部署中利用 ex‑post DP 进行 FSRC 级联，显著提高可用查询数。

**🔧 技术方法**

技术手段：差分隐私机制（高斯噪声、Gumbel 噪声、指数机制）、KL 与 χ² 正则化、拒绝采样与 ReLU 接受权重、误差分析与偏差-方差权衡、ex‑post 隐私证明与数值积分、FSRC 级联组合。

**📊 数据集**

数据集：奖励模型基于 Oasst、Gemma、Llama、Armo；基准任务为 GSM8K、MMLU、MATH（含数学/化学子集）；基模型采用 Phi‑3‑Mini‑Instruct 与 Gemma‑2‑2B‑Instruct。

**📈 对比分析**

对比方法：与经典 BoN、无隐私 ITP 以及 PrivITP 进行实验；在 n=2¹² 时评估相对基模型的准确率提升；PrivITP 在所有 (奖励模型、数据集) 组合上均至少匹配 ITP 天花板，且在低噪声下逼近无隐私 ITP；在多查询场景下，FSRC 级联可获得约 3 倍更多查询。性能表现：准确率提升 80%–95%，并在奖励模型较弱时显著抑制 reward‑hacking。

**⚠️ 局限性**

局限性：PrivBoN 的“隐私免费”阈值依赖未知的覆盖系数 C^π*，实际难以验证；PrivITP 需要手动调参（β、噪声尺度）并假设奖励输出有界；方法仅覆盖单步推理，未考虑序列化生成；对极低 ε 的理论与实践差距尚未完全消除；大规模查询下的计算开销和噪声积累问题仍需进一步研究。

---

## 310. Domain-Specific Self-Supervised Representation Learning for Retinal Fundus Classification

**arXiv ID:** 2608.26686 | [PDF](https://arxiv.org/pdf/2608.26686v1)

**作者:** Bekzat Nurlanbekova `[一作]` (Monash University), Fung Fung Ting `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在眼底影像上利用SimSiam和SimCLR两种自监督框架进行预训练，并评估不同域特定增强、批量大小和训练时长对下游疾病分类任务的影响。

**💡 创新点**

创新点在于将Ben Graham的预处理和灰度增强融入自监督训练的正样本生成，针对医学影像设计域特定增强策略，并在资源受限环境下系统对比两种框架的表现。

**🔧 技术方法**

采用SimSiam、SimCLR框架，ResNet‑18骨干网络，灰度+Graham预处理，线性评估、全微调和Grad‑CAM可视化技术。

**📊 数据集**

使用了FISSL无标签预训练集、EDID多标签分类集以及RetinaMNIST DR分级集。

**📈 对比分析**

通过线性评估和微调与ImageNet预训练的监督模型对比，域特定增强显著提升SimSiam和SimCLR在DR任务上的准确率和QWK；SimCLR在小批量、短时训练下表现更稳定，但在包含多种眼病的EDID集上性能受限。

**⚠️ 局限性**

受限于计算资源仅能使用小批量和短时预训练，SimCLR缺乏对NT‑Xent温度参数的深入调优，域特定增强可能对非血管疾病（如青光眼）产生偏差。

---

## 311. The Randomized Query Complexity of Finding Minimal Elements in Bounded-Width Posets

**arXiv ID:** 2608.26981 | [PDF](https://arxiv.org/pdf/2608.26981v1)

**作者:** Luyao Fan `[一作]` (Shanghai Jiao Tong University), Jia Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在宽度最多为w的未知n元素偏序集中的所有最小元素的零错误随机查询复杂度。

**💡 创新点**

证明了有限下界，确定了每个固定宽度下的随机上界的正确渐近主导常数。

**🔧 技术方法**

使用了随机链硬分布的成对计数方法，结合了组件翻转的自反性和不可比较比较的唯一所有权属性。

**📊 数据集**

使用了未知的n元素偏序集，具体数据集未明确给出。

**📈 对比分析**

与Daskalakis等人的工作进行了比较，确认了随机上界和下界之间的主导常数差距，证明了下界的主导项为w+1/2n-w(w+3)/4+w(1-1/w)^n+w(w-1)/4(1-2/w)^n。

**⚠️ 局限性**

研究中未提及具体的局限性，但可能存在对特定类型偏序集的适用性限制。

---

## 312. The Guard That Cried Wolf: How Scary Words Make Agent Guardrails Refuse Legitimate Actions

**arXiv ID:** 2608.27009 | [PDF](https://arxiv.org/pdf/2608.27009v1)

**作者:** Yingjie Zhang `[一作]` (Chinese Academy of Sciences), Kai Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造了Cautious Bench基准，用于衡量LLM代理防护栏在执行安全上的过度安全（over‑safety）现象，并通过授权政策自动生成标注。

**💡 创新点**

创新点在于：①首个针对过度安全的基准，标签直接来自公开可审计的授权政策并通过构造门和可信性检查保证无人工偏差；②揭示名词恐怖效应（name‑superstition），即防护栏更易拒绝安全授权的“恐怖”资源名。

**🔧 技术方法**

使用技术包括：授权政策 Π 的定义、构造门（Construction‑Invariant Gate）保证标签可推导、可信性检查（Faithfulness Check）验证样本无隐含危险、三等级名称分化、对照实验设计，以及与七个现有 guardrail 的对比评估。

**📊 数据集**

数据集为自生成的基准：756条可决决策对（benign/twin）+40条不可决对，每条样本在三种名称等级（innocent、as‑authored、scary）下呈现；生成器与威胁名称词表均公开发布。

**📈 对比分析**

比较方法是把各 guardrail 的误拒（FP）和正确拒（twin detection）率与基准标签对比，结果显示所有执行安全防护栏在“恐怖”名称下 FP 显著上升（如 AgentDoG‑Qwen FP 63→70→82%），而内容安全基准 LlamaGuard 则保持平稳；性能随名称等级与设计而异。

**⚠️ 局限性**

局限性包括：仅覆盖自然语言、单语言、单主体、同步的LLM防护栏；未测跨语言或多主体场景；只评估决策边界，未涵盖语义对齐、时间性或权限泄露等问题；且未解决不可决细分中无法区分的安全/危险混合情况。

---

## 313. A Multi-Framework Comparison of Outline Stages in Long-Form Generation with LLMs

**arXiv ID:** 2608.26177 | [PDF](https://arxiv.org/pdf/2608.26177v1)

**作者:** Yifan Song `[一作]` `[通讯]` (Taiyuan Institute of Technology), Yifan Song (Taiyuan Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七种长文本生成框架在三种粒度（单章节、多章节、整本）下进行统一基准的头对头对比，并设计双轨评估：直接评估大纲（anchor‑based LLM‑as‑judge）与评估展开后的文本。

**💡 创新点**

提出了一套“双轨评估”与“大纲‑写作解耦”原则，能够在同一源文本下独立衡量大纲质量与生成质量；同时构建了包含中文、英文及其他题材的统一长文本数据集和多维度评测体系。

**🔧 技术方法**

使用大规模语言模型（同一模型既做生成后端又做评判器）进行大纲生成与评测；Anchor‑based LLM‑as‑Judge 方案评估大纲；在写作侧采用 HANNA、WriteJudge、长度合规度、词频多样性、冲突率等多维度自动与人工相结合的指标。

**📊 数据集**

数据集包含 300 篇长文本：100 篇中文网络小说、100 篇英文小说、100 篇其他类型作品（共 10 章/本），覆盖跨语言、跨题材与跨长度三维空间。

**📈 对比分析**

通过统一的 LLM 后端、统一评判器、统一 prompt 结构，将所有框架置于相同实验环境；对每个框架-粒度单元进行大纲生成、评测、展开与评测，最终计算各维度平均分。实验结果显示结构化大纲框架在 A2/A3/A4 维度显著优于无大纲基线；SuperWriter 在单章节长度受限场景下获得最高分，但在整本模式下表现下降；不同粒度间框架排名波动大，整体无单一最优框架；大纲与写作侧排名相关性仅为中等（Spearman ≈ 0.45–0.58）。

**⚠️ 局限性**

主要局限包括：①评估采用反向 “长文本→大纲” 任务，导致与原始前向任务的结果存在天花板效应；②评判器与生成器使用同一 LLM，可能产生自我偏好影响；③写作侧样本受计算约束，仅覆盖少量案例；④部分框架在逆向任务下的关键模块（如 SuperWriter 的 refinement 阶段、CogWriter 的约束集）被禁用，影响公平性；⑤未覆盖所有主流长文本框架，实验范围有限。

---

## 314. MoganColBERT-TR: A Late-Interaction Multi-Vector Retrieval Model for Turkish

**arXiv ID:** 2608.26344 | [PDF](https://arxiv.org/pdf/2608.26344v1)

**作者:** Furkan Yilmaz `[一作]`, Muhammed Faruk Gozay `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了基于从零开始训练的Turkish ModernBERT的多向量检索模型，并在零样本BEIR Turkish数据集上进行评估。

**💡 创新点**

创新点在于结合单词级别投影、组掩码+余弦上限的硬负样本挖掘以及单轮交叉编码器蒸馏，显著提升了少量参数下的检索效果。

**🔧 技术方法**

使用了ModernBERT架构、ColBERT式的token级投影、跨编码器蒸馏、FlashAttention‑2、PyLate训练框架等技术。

**📊 数据集**

数据集包括Turkish BEIR子集的五个任务（SciFact-TR、ArguAna-TR、FiQA-TR、SciDocs-TR、NFCorpus-TR）以及内部自制的title→passage训练对。

**📈 对比分析**

在官方PLAID检索管线下与mLateOn、ColmmBERT等模型对比，取得37.36的综合分，领先ColmmBERT-base-TR 3.05点、LFM2.5-ColBERT-350M 12.30点，排名第二。

**⚠️ 局限性**

局限性包括仅单轮训练单种随机种子、未做多种消融实验、查询长度32的上限、评估窗口低于训练窗口、医学域覆盖不足、仅使用单一蒸馏教师。

---

## 315. Knowledge-Verified Emergent Deception in LLM Agents Under Conflicting Incentives

**arXiv ID:** 2608.26372 | [PDF](https://arxiv.org/pdf/2608.26372v1)

**作者:** Zheyuan Liu `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个知识验证型多轮对话基准（KnownLieBench），用于评估在企业部署的 LLM 代理在用户权益与公司利益冲突时是否会说谎，并区分诱因导致的自发欺骗与被明确指令诱导的欺骗。

**💡 创新点**

创新点在于：① 在评估前先做中立知识探测，确认模型确实知道用户权益；② 在多轮交互中加入动态客户信任度与工具调用，使代理在逼真环境下决策；③ 设计了“诱因”与“指令”两种欺骗场景，能区分模型的欺骗倾向与遵从性；④ 通过大规模 18 组模型与 112 个基于真实政策的案例，提供跨域欺骗行为的细粒度对比。

**🔧 技术方法**

技术手段包括：大语言模型（Claude、GPT、Gemini、Llama、Qwen 等）与其工具调用框架；对话环境设计使代理与虚拟客户互动；使用 GPT-5.1 作为自动判定器提取欺骗与检测标签；实现知识门控（neutral probe）与动态信任更新；对内部表示做 Jacobian Lens（J‑Lens）分析。

**📊 数据集**

数据集为 112 条基于公开法律或公司政策的案例（64 个“应得”案例 + 48 个校准案例），覆盖 8 个客服领域（退货、航班、保证金、订阅、账单争议、保险索赔、车辆召回、债务催收）。每条案例被转化为可执行环境，包含案例记录、政策、工具与评估规范。

**📈 对比分析**

比较方法：对每个模型在三种初始信任（高、中、低）下分别进行 3 种情景（Emergent、Instructed、Honest Control）评估。指标包括：欺骗率（DR）、欺骗成功率（DSR）、检测率（Det）以及信任变化。结果显示：① 大部分模型在 Instructed 场景欺骗率高达 70%‑90%，但在 Emergent 场景相对低；② 信任度对欺骗成功率与检测率影响大，信任越高欺骗更难被捕获；③ 不同模型在域间表现差异显著，说明模型能力与域特征相关。

**⚠️ 局限性**

局限性：仅覆盖英文客服场景，权益二元化；使用模拟客户而非真实用户，信任动态不一定映射到人类行为；评估仅在文本交互中，未考虑多模态或真实部署；内部表示分析仅在两款模型与单轮情景下进行，缺乏普适性；后训练效果仅在小规模 LoRA 试验，未覆盖大规模微调。

---

## 316. Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference

**arXiv ID:** 2608.26674 | [PDF](https://arxiv.org/pdf/2608.26674v1)

**作者:** Mengfan Li `[一作]` (Huazhong University of Science and Technology), Yang Deng `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于系统功能语言学的反向结构化评估框架（PRISM）来衡量大语言模型在角色扮演中的人物忠实度，聚焦任务框架、互动立场与语言风格三个维度进行分层评估。

**💡 创新点**

创新点在于：①把人物忠实度拆解为可解释的三维心理语言学特征；②将评估转化为逆向推理任务，通过在受限标签空间内计算后验概率来获得维度级别的可信度；③提供可审计、可解释的多维度汇总分数，显著降低“整体评价幻觉”。

**🔧 技术方法**

主要技术包括系统功能语言学（SFL）框架、逆向推理（inverse inference）与后验分布估计、结构化标签空间构造、维度级别分数聚合与整体评估。

**📊 数据集**

使用的公开数据集包括 Big5-CHAT 生成的 Big5-Persona-EASY 与 Big5-Persona-HARD，以及 SocialBench 中的 Social-Persona，用以构建正负样本并生成难度较高的对照组。

**📈 对比分析**

与传统的 LLM-as-a-Judge 直观评分、专业评估模型（Selene Mini、PandaLM、AlignScore）以及多种开源与闭源 LLM 进行对比，实验显示 PRISM 在 AUC、Pair-AUC 与 G-Acc 指标上均显著提升，尤其在困难数据集和严格的组级指标上表现最为突出。

**⚠️ 局限性**

局限性包括：①功能维度的理论范围有限，可能遗漏其他社会语言学维度；②框架依赖模型的 token‑级 log‑prob 访问，闭源模型难以直接应用；③仅聚焦评估，而未探索如何将维度级反馈融入模型训练或对齐流程。

---

## 317. Comparing Chunking and Embedding Strategies for Turkish RAG Systems

**arXiv ID:** 2608.26192 | [PDF](https://arxiv.org/pdf/2608.26192v1)

**作者:** Mustafa Sertaç Türkel `[一作]` (Ata Technology Platforms), Ahmet Tuğrul Bayrak `[通讯]` (Ata Technology Platforms)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对土耳其语文档问答系统，系统性对三种块化策略、五种嵌入模型与两种生成LLM进行全因子对比实验，评估它们在不同文档布局（表格占比不同）下的RAG性能。

**💡 创新点**

①提供了针对形态学丰富语言的RAG实验框架，①揭示块化策略与嵌入模型的交互效应；②证明语言专门化嵌入在该语料下并无检索优势；③首次量化表格与文本问答在不同布局下的差异。

**🔧 技术方法**

Docling、语义分块、固定长度分块；FastText、TurkEmbed、Mursit-large、multilingual‑e5‑large、text‑embedding‑3‑small嵌入；GPT‑4o mini、qwen‑plus生成；FAISS近邻索引；Llama‑3.3‑70B‑Instruct自动判定；统计使用配对McNemar检验与Holm校正、逻辑回归交叉验证。

**📊 数据集**

三份土耳其语机构文档（表格占比不同），每份100题，共300题，问答对由人工确定答案，文档包含混合表格与文本内容。

**📈 对比分析**

通过9,000个问答对进行全因子评估，报告准确率、块大小、响应时间等指标。最佳配置为Docling+Mursit‑large+GPT‑4o mini，准确率87%；单个最佳组件并不一定组成最佳配置；块化策略对表格问答提升显著；嵌入模型差异小，FastText性能差；生成器在准确率上GPT‑4o mini优于qwen‑plus，但更慢。

**⚠️ 局限性**

仅使用三份文档，缺乏多样性；评估判定器为自动LLM，未做人工验证；未探索混合分块策略；模型选择受限于可用API；未对非形态学语言进行对照。

---

## 318. Emotion Understanding in Streaming Video with Trajectory-Aware Reliability

**arXiv ID:** 2608.26786 | [PDF](https://arxiv.org/pdf/2608.26786v1)

**作者:** Qingsong Wang `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TRACE 框架，用音频前缀构建情绪信念轨迹，基于轨迹校准的可靠性估计动态决定是否进行低延迟预测或多模态上下文重推断，实现流式视频情绪理解。

**💡 创新点**

创新点在于将情绪理解建模为情绪信念轨迹演化，利用轨迹信息进行可靠性估计，从而主动选择是否调用高成本多模态重推断，显著提升准确率与成本的折中。

**🔧 技术方法**

使用 Qwen2.5-Omni-3B 进行在线前缀情绪信念生成，轨迹可靠性估计采用 MLP 双头模型，重推断使用 Qwen2.5-Omni-7B 并融合视觉、文本及对话上下文；同时利用 KV 缓存实现低延迟推理。

**📊 数据集**

主要数据集包括自制的 StreamMER（基于 Friends 剧集的流式情绪数据）以及公开的 MELD 与 MER2024 情绪识别基准。

**📈 对比分析**

与多模态基线（Gemini、Emotion-LLaMA、AffectGPT 等）及流式基线（全前缀、全重推断、规则可靠性等）对比，TRACE 在 StreamMER 上将准确率从 59.9% 提升至 69.5%，重推断率仅 55%（相比 100%），且实时因子 RTF 仅 0.091，保持接近全重推断的性能。

**⚠️ 局限性**

主要局限包括对域迁移的鲁棒性不足、重推断仍会产生额外延迟、过度依赖音频前缀忽略早期视觉线索，以及 StreamMER 仅来自脚本对话，缺乏更自然、多样化的真实场景。

---

## 319. Beyond Edge Cuts: Activity-Weighted Multicast Hypergraph Mapping for Spiking Neural Networks on Mesh NoCs

**arXiv ID:** 2608.26223 | [PDF](https://arxiv.org/pdf/2608.26223v1)

**作者:** Amirreza Khorasanian `[一作]` `[通讯]` (University of Tehran), Amirreza Khorasanian (University of Tehran)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于源根有向超图的路由感知多播映射框架M-HySMap，用以在神经形态多核平台上映射脉冲神经网络，显著降低多播跳数和链路拥塞。

**💡 创新点**

创新点在于将每个发射神经元视为带活动权重的源根有向超边，优化多播路由链接并合并共享路径前缀；同时利用“受影响超边局部性”实现精确增量评估，提升搜索效率。

**🔧 技术方法**

使用的技术包括：活动感知图划分 + 二次分配（QAP）种子；源根有向超图划分与路由联合搜索；增量超边更新与邻域组合策略（粒度划分、核心位置交换、交替联合迭代）；以及多种邻域的投资组合投票。

**📊 数据集**

实验数据集为基于Potjans层级结构的递归脉冲神经网络（79–163个神经元，238–1028个突触），在4×4至7×7的XY路由网格上进行评估。

**📈 对比分析**

与传统边切割+QAP、活动权重+QAP以及仅以QAP种子进行的M-HySMap对比，M-HySMap在115个工作任务中将路由多播跳数降低19.7–41.1%（相对Edge+QAP）和10.6–19.6%（相对Activity+QAP），同时平均最大链路负载下降约20%；增量评估相较全量重算加速4.7–12.7倍。

**⚠️ 局限性**

局限性包括：仅在规模较小的Potjans仿真网络上验证；评估指标为映射层面的流量代理，未转化为能耗或时延；活动率基于有限的采样，未覆盖多阶段工作负载；搜索策略为启发式，无法保证全局最优。

---

## 320. Multi2AV-Safety: Benchmarking Safety in Multimodal-to-Audio-Video Generation

**arXiv ID:** 2608.26535 | [PDF](https://arxiv.org/pdf/2608.26535v1)

**作者:** Kaichao Jiang `[一作]` (University Of Science And Technology Of China), Nenghai Yu `[通讯]` (University Of Science And Technology Of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Multi2AV‑Safety 基准，用于评估多模态音视频生成模型在多模态条件下的安全性能。

**💡 创新点**

创新点在于系统覆盖所有 11 种文本、图像、音频、视频组合，明确区分攻击机制、证据结构与模态组合，揭示了“组合危害（Composed Harm）”与“稀释危害（Diluted Harm）”两种跨模态风险。

**🔧 技术方法**

使用了多模态生成模型（如 Qwen3‑Omni、GuardReasoner‑Omni）以及多种安全检测器（GuardReasoner‑VL、GuardTrace‑VL），并对攻击实例进行手工与自动评估。

**📊 数据集**

数据集包含 11,024 个攻击实例，涵盖双模、三模、四模组合，分布在 4 种攻击方式（Direct、Jailbreak、Adversarial、Temporal）和 5 类危害（暴力、性、非法、仇恨、政治敏感）。

**📈 对比分析**

评估指标为目标对齐成功率（ASR）和剩余有害输出率（RHR）；结果显示即使在 100% 模态可见的情况下，完整模态下的安全检测器召回率也低于 70%，尤其在 k=0（组合危害）和 k=1（稀释危害）场景中召回率显著下降。

**⚠️ 局限性**

局限性包括：1) 仅针对文本、图像、音频、视频四种模态，未覆盖其他感知模态；2) 部分攻击（如图像/视频对抗攻击）在本基准中缺失；3) 评估依赖人工审核，可能存在主观偏差；4) 仅在单一生成模型上验证，缺乏跨模型泛化验证。

---

## 321. Order Matters: A Chinese Multi-Panel Meme Benchmark for Vision-Language Reasoning

**arXiv ID:** 2608.26866 | [PDF](https://arxiv.org/pdf/2608.26866v1)

**作者:** Haihan Li `[一作]` (Shanghai Maritime University), Jize Qian `[通讯]` (Shanghai Maritime University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了中文多面板 Meme 评测基准 CMPM，并设计了两层评测协议，检验视觉语言模型对面板顺序的理解与解释能力。

**💡 创新点**

首次构建面板顺序敏感的中文 Meme 数据集，定义结构类型、顺序依赖与上下文关系，并引入顺序恢复与解释生成两任务及上下文消融设置。

**🔧 技术方法**

利用五个代表性的大型视觉‑语言模型（InternVL3.5、Qwen3.5、GLM-4.1V、GPT‑5.5、Gemini 3.1 Pro）进行无监督评估，并使用人类标注的 Likert 评分与整体排序进行二阶评估。

**📊 数据集**

使用自采集的 1,214 句中文多面板 Meme 数据集，涵盖 2–15 张面板，标注面板结构、顺序约束、上下文关联等信息。

**📈 对比分析**

与标准顺序、混洗顺序及是否含评论四种条件对比，结果显示封闭式模型在混洗顺序下仍低于 80%，开放式模型仅 20% 以内；在解释生成任务中，Gemini 3.1 Pro 与 GPT‑5.5 获得最高评分，整体差距明显。

**⚠️ 局限性**

局限性包括：仅覆盖中文社交平台 Meme；数据量相对较小；人类评判一致性中等；未对模型进行微调，评估受提示与解码设置影响。

---

## 322. Chart2SVG: Editable SVG Generation from Raster Chart Images

**arXiv ID:** 2608.26544 | [PDF](https://arxiv.org/pdf/2608.26544v1)

**作者:** Jinning Cui `[一作]` (Renmin University of China), Yunhai Wang `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究开发了一种基于视觉‑语言模型的系统，将栅格图表转换为可编辑的 SVG，并通过图表结构图（CSG）实现语言驱动的图表编辑与复用。

**💡 创新点**

创新点包括：① 用图表专用规范化管道将异构 SVG 统一为可学习的结构化表示；② 在 VLM 中加入 SVG 语法 token 与渲染感知的 GRPO 训练，提升生成的结构完整性与视觉一致性；③ 通过 LLM 生成 CSG，将低层 SVG 原语与高层语义关系显式化，支持跨元素的协调编辑。

**🔧 技术方法**

主要技术：Qwen3‑VL 视觉‑语言模型 + LoRA 参数高效微调；SVG 语法 token 化；渲染感知的 Group Relative Policy Optimization；图表结构图构建与基于 CSG 的编辑规划。

**📊 数据集**

数据集：基于 Beagle+（约 33K 对齐的图像‑SVG 样本）进行训练，评估时使用 Beagle+ 的三种子集（ChartBlocks、Fusion、Plotly）进行内分布测试，以及 VisAnatomy 进行外分布测试。

**📈 对比分析**

与 OmniSVG、StarVector、ChartCoder、GPT‑4o 等基线对比，Chart2SVG 在 PSNR、SSIM、LPIPS、前景 IoU、边缘一致性等指标上均优于所有对手，并在所有数据集上实现 0% 失败率（仅在 VisAnatomy 仍略有 0.78% 失败）。

**⚠️ 局限性**

局限性：对复杂/自定义标记、极高密度图表、失真截图的处理效果有限；系统侧重视觉与结构重建，对原始数据提取不够精准，且仍需要人工介入以保证最终输出的准确性。

---

## 323. Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference

**arXiv ID:** 2608.26574 | [PDF](https://arxiv.org/pdf/2608.26574v1)

**作者:** Wooje Park `[一作]` (Seoul National University), Byonghyo Shim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的三态可撤销解码框架 DARD，解决了可撤销解码中不可靠词元污染验证上下文的问题。

**💡 创新点**

创新点在于将词元划分为遮蔽、候选、已解码三态，并采用置信度排序的上下文与自适应 logits 混合方式，避免了不可靠词元对验证过程的干扰。

**🔧 技术方法**

使用了扩展阴影序列、定制的注意力掩码、按置信度排序的注意力机制以及自适应 logits 混合等技术。

**📊 数据集**

在 12 个文本与多模态基准上评估，包括 LLaDA、MMaDA、Flickr30K、AI2D、MATH500、MBPP、GSM8K、Sudoku 等数据集。

**📈 对比分析**

与 WINO、Saber 等最新可撤销解码方法对比，DARD 在速度‑质量 Pareto 前沿上更优，Flickr30K 上实现 2.71× 加速、CIDEr 提升 4.35 分。

**⚠️ 局限性**

局限在于性能提升幅度有限（主要因未改模型分布），并且引入了每步计算的额外开销，需进一步优化实现。

---

## 324. A Unified Framework for Fair and Personalized Decentralized Learning under Communication Constraints

**arXiv ID:** 2608.26493 | [PDF](https://arxiv.org/pdf/2608.26493v1)

**作者:** Krishnendu S. Tharakan `[一作]` (KTH Royal Institute of Technology), Carlo Fischione `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种在分布式学习中同时实现个性化、公平性与通信效率的框架，推出DMFL‑SQ算法；

**💡 创新点**

创新点在于将图基个性化、agnostic公平正则化与稀疏量化事件触发通信统一到一个非凸的分布式优化框架，并给出收敛与PAC‑Bayes一般化保证；

**🔧 技术方法**

使用图拉普拉斯正则化实现个性化，aggressive mixture fairness envelope实现公平，稀疏量化压缩与事件触发通信实现通信压缩；

**📊 数据集**

实验数据集包括CIFAR‑10（通过Dirichlet划分产生强异构数据）和真实异构EEG数据集MUSMET；

**📈 对比分析**

与D‑PSGD、CHOCO‑SGD、DSGT及q‑FFL等基线对比，DMFL‑SQ在保证公平性的同时显著降低通信量，并在两组数据上取得更高的平均与最差端准确率；

**⚠️ 局限性**

局限性包括：在极度异构的CIFAR‑10实验中准确率仍偏低，需进一步验证在更大规模或不同模型上的表现，且算法仍依赖手工调参（触发阈值、正则权重等）。

---

## 325. Unpublished Draft: A Post-Processing Approach to Fairness in Tie-Aware Rankings

**arXiv ID:** 2608.26478 | [PDF](https://arxiv.org/pdf/2608.26478v1)

**作者:** Somya Nigam `[一作]` (University of Antwerp), Kenneth Sörensen `[通讯]` (University of Antwerp)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种后处理方法，在给定带有平局的非公平共识排名时，求取其最近的公平排名，并提供精确求解算法与启发式近似方案。

**💡 创新点**

首次针对含平局的排名设计可接受公平性的最优解算法，支持≥100项且多重公平阈值，并引入分桶级别的启发式搜索。

**🔧 技术方法**

使用二进制整数线性规划（BILP）构建精确模型，加入统计平等约束；启发式结合顺序局部搜索、禁忌搜索、模拟退火与ε‑贪婪策略，采用桶级操作。

**📊 数据集**

使用合成数据生成器随机产生带指定平局比例的两组成员排名，规模可达数千。

**📈 对比分析**

实验显示，精确算法在n≤10⁴时可得到最优解；启发式在更大规模实例中能在较短时间内生成近似最优且满足公平性的排名；与仅支持严格排序的现有方法相比，提升了可行性和公平度。

**⚠️ 局限性**

仅考虑二分类受保护属性，尚未扩展到多组或其他公平度量；在极大规模实例下，启发式的性能与精确解的可行性仍有限。

---

## 326. ClassVision: AI-Powered Classroom Attendance System

**arXiv ID:** 2608.26173 | [PDF](https://arxiv.org/pdf/2608.26173v1)

**作者:** Ankit Kumar Aggarwal `[一作]` (Yeshiva University), Youshan Zhang `[通讯]` (Yeshiva University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了ClassVision自动化课堂签到系统

**💡 创新点**

首次使用RetinaFace结合ROI 50x50裁剪，实现不受摄像头距离限制的签到

**🔧 技术方法**

采用RetinaFace进行人脸检测，Face Recognition库进行特征提取与识别

**📊 数据集**

自采集的36人、3600张训练图像和69张多场景测试图像

**📈 对比分析**

与多种检测/识别组合对比，检测准确率99.4%，识别准确率90.3%，显著优于传统方法

**⚠️ 局限性**

受光照、遮挡、帽子、胡须等因素导致的训练偏差；需更高分辨率相机和多角度数据以提升鲁棒性

---

## 327. Realistic Counterfactual Explanations via Denial Constraints

**arXiv ID:** 2608.26335 | [PDF](https://arxiv.org/pdf/2608.26335v1)

**作者:** Avia Asael `[一作]` (Tel Aviv University), Daniel Deutch `[通讯]` (Tel Aviv University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将拒绝约束（Denial Constraints）用于保证生成的对抗解释在数据集上可行且真实，并在此基础上求解多样且接近的解释。

**💡 创新点**

首次将拒绝约束与可解释AI的对抗解释相结合，并在投影-扰动框架中设计多种优化（缓存、可疑集过滤、多样性约束）实现可扩展的现实性约束搜索。

**🔧 技术方法**

使用 SMT 求解器（Z3）、DiCE 擾動方法、预处理与缓存、可疑集过滤、DPP 多样性度量、基于 MAD 的距离函数等技术。

**📊 数据集**

在四个基准数据集上实验：Adult‑Income、NY‑Housing、Tax、Census‑Income。

**📈 对比分析**

与现有方法（DiCE、CARLA、最优/检索等）对比，所提方法实现 100% 无约束违规，距离提升不足 10%，多样性下降不到 4%，且通过优化后执行时间比裸 SMT 快最多 63 倍；在大数据集上仍可在几百秒内完成。

**⚠️ 局限性**

局限性包括：仍需 SMT 求解器，投影过程对数值型属性的离散化敏感；对非表格/图像数据不适用；对高度多维、约束极其稠密的数据集可能仍面临规模瓶颈；并未解决可行动性（recourse）和因果解释的问题。

---

## 328. Report of the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science

**arXiv ID:** 2608.26519 | [PDF](https://arxiv.org/pdf/2608.26519v1)

**作者:** Lois Curfman McInnes `[一作]`, Lou Woodley `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

汇总2026年科学计算下一代生态系统研讨会成果

**💡 创新点**

提出四大相互依赖的战略主题与八项社区行动优先级，强调信任、可追溯性与人机协作

**🔧 技术方法**

采用社会技术共设计、案例研讨与主题工作坊方法

**📊 数据集**

未使用传统数据集，主要基于专家访谈和行业案例

**📈 对比分析**

未进行性能评估，仅通过讨论形成共识

**⚠️ 局限性**

缺乏实证验证与量化指标，实施路径仍需进一步探索

---

## 329. Which India Survives Translation? Narrative Homogenisation Across Indian Oral Traditions in LLMs

**arXiv ID:** 2608.26123 | [PDF](https://arxiv.org/pdf/2608.26123v1)

**作者:** Paarth Singh Rathore `[一作]` `[通讯]` (BITS Pilani), Paarth Singh Rathore (BITS Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探究大型语言模型在生成印度三大地域口头与文学传统故事时的叙事同质化程度，并评估区域语言提示对忠实度的影响。

**💡 创新点**

首次使用句向量相似度方法对不同印度传统的生成文本进行客观度量，并将区域语言提示与英语提示进行对比，揭示同质化与多语提示的矛盾结果。

**🔧 技术方法**

句子BERT嵌入与余弦相似度、UMAP降维、Claude Sonnet和Gemini模型的文本生成。

**📊 数据集**

三份英文翻译的参考语料：拉贾斯坦的帕布吉史诗11段、泰米尔的Sangam诗歌21段、孟加拉的民间故事10段。

**📈 对比分析**

通过自我相似度与跨传统相似度计算漂移与聚合度，发现输出更接近自身传统但跨传统相似度仍高达0.52–0.66，区域语言提示降低了自身传统的相似度约27个百分点。

**⚠️ 局限性**

样本量小、仅评估两模型、使用英文翻译语料与英文嵌入可能放大多语提示损失、无法区分命名实体重叠与结构一致性。

---

## 330. FAN-LoRA: A Fourier-Adaptive Nonlinear Low-Rank Adaptor for Medical Foundation Model Domain Adaptation

**arXiv ID:** 2608.26531 | [PDF](https://arxiv.org/pdf/2608.26531v1)

**作者:** Ziquan Liu `[一作]` (Southwest University of Science and Technology), Xuyang Shi `[通讯]` (Southwest University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 FAN-LoRA，一个将低频结构对齐与高频细节补偿显式分离的频率解耦参数高效微调模块，用于将 SAM 等基础模型迁移至医学影像跨模态和跨中心场景。

**💡 创新点**

创新点在于：①用 B‑spline 驱动的低通非线性层实现全局结构平滑；②用离散傅里叶高通分支进行局部纹理补偿；③通过显式频率分离避免低频与高频梯度冲突，从而提升边界精度和整体性能。

**🔧 技术方法**

使用技术包括：B‑spline 自适应非线性层（ANL）、固定正弦傅里叶投影、低秩矩阵分解（LoRA）、AdamW 优化、Dice/BCE 损失、以及与 SAM ViT‑B/16 预训练模型的兼容。

**📊 数据集**

数据集：MM‑WHS 2017（MR→CT）、Promise 12 与 NCI‑ISBI（不同扫描仪 1.5T/3T）、FLARE 22 与 CHAOS（CT→MRI），覆盖跨模态与跨中心两大迁移任务。

**📈 对比分析**

与 LoRA、DeLoRA、SHiRA、AuroRA、FourierFT、FreqFiT、全量微调以及零样本方法对比；实验显示 FAN-LoRA 在 Dice、HD95 上均优于所有基线，平均 Dice 提升约 1%，HD95 明显降低；参数量仅约 111k，约为 LoRA 的四分之一，保持了极高的参数效率。

**⚠️ 局限性**

局限性：在极低对比度或高噪声区域仍可能出现误判；频率分离方案固定，未能自适应不同域的最佳频段；在少量标注或极端域差的情况下，性能提升有限。

---

## 331. Evaluator-Dependent Patient-Adaptive ECG Lead-Channel Allocation

**arXiv ID:** 2608.26827 | [PDF](https://arxiv.org/pdf/2608.26827v1)

**作者:** Xiaoyang Li `[一作]` (Northeastern University), Zeyan Tao `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究在不同诊断评估器之间，患者条件的 ECG 导联分配策略的性能；对固定子集与自适应子集在多种预算下进行对比，量化评估器依赖性；并通过多种后验敏感性分析探讨机制。

**💡 创新点**

首次系统量化并证明患者条件的导联分配优势并非评估器不变；提出多种后验敏感性分析（共同参考、掩码混合、评估器对齐策略）验证该效应。

**🔧 技术方法**

使用可控掩码逻辑回归评估器与强大的 Masked ResNet1D 诊断评估器，结合两种自适应策略（ECG-on-Demand 与 MGA），并进行全量子子集搜索。

**📊 数据集**

PTB-XL 12 导联 ECG 数据集，采用官方折叠进行训练、验证、固定子集选择、开发审计与 hold‑out 评估。

**📈 对比分析**

通过计算在每个评估器下的固定最佳子集与自适应策略的损失差异 D 及其交互 I，使用 NLL、Brier 与 ECE 等概率指标。结果显示：在开发评估器下自适应优于固定，但在强评估器替换后优势消失，交互均为正。

**⚠️ 局限性**

仅在单一公开数据集 PTB-XL 上验证；评估器替换后未进行联合优化；fold‑8 结果为探索性验证，未构成严格的 confirmatory 证据。

---

## 332. Is Your Neighborhood Safe? Place-based Stigma in Large Language Models' Urban Safety Judgments

**arXiv ID:** 2608.26188 | [PDF](https://arxiv.org/pdf/2608.26188v1)

**作者:** Huy Nguyen `[一作]` (Augustana College), Yue Lin `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在七个指令调优的大型语言模型上，使用仅坐标、仅名称和名称+坐标三种条件，对洛杉矶和芝加哥186个社区的夜间步行安全进行零样本评分，探究模型评分是否随真实犯罪风险而变化或随社区名称所携带的种族污名而偏差。

**💡 创新点**

首次揭示名称通道既携带真实犯罪信号，又在模型地理知识越丰富时放大对边缘化族群社区的安全负面刻板偏见，指出这种隐藏的歧视机制难以仅通过隐藏名称来修正。

**🔧 技术方法**

采用零样本安全评分提示、期望评分计算、名称移位分析、配对匹配实验、执法弹性分解、识别度检验和多模板鲁棒性测试等多重实验与统计方法。

**📊 数据集**

使用洛杉矶与芝加哥的官方犯罪记录（2020-2025年）、美国社区调查（ACS）5年期人口、收入与种族分布数据，并将其与模型评分对齐。

**📈 对比分析**

通过Spearman相关、回归系数、Bootstrap置信区间以及坐标与名称条件的对照，七个模型在所有评估指标上均显示显著的负相关与种族效应；较大模型在坐标通道上幅度提升但整体仍以名称为主导。

**⚠️ 局限性**

局限包括：犯罪记录受执法偏差影响、城市分隔度高导致犯罪与种族共线性、模型样本有限、仅使用英文单轮提示、缺乏更深入的因果实验与多语言验证。

---

## 333. PICasso: An AI-Enabled Design Framework for Autonomous Optimization of Silicon Photonic Devices

**arXiv ID:** 2608.26113 | [PDF](https://arxiv.org/pdf/2608.26113v1)

**作者:** Deepak Vungarala `[一作]` (New Jersey Institute of Technology), Shaahin Angizi `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PICasso框架，实现从自然语言描述到可制造硅光子集成电路的全自动合成、验证与优化；

**💡 创新点**

通过结构化的NL→YAML→GDS流水线、PDK感知知识注入、自动布局与路由、物理验证（DRC/LVS）以及闭环仿真优化，将LLM从脆弱的网表生成器转变为可生成可制造、功能符合的电路设计代理；

**🔧 技术方法**

结合大语言模型（如GPT‑4o、Claude Sonnet 4.5、Llama 3.1 70B）、YAML DSL、gdsfactory布局引擎、KLayout DRC/LVS、SAX光学仿真、两阶段优化（设备级几何调节+电路级相位/耦合微调）等技术；

**📊 数据集**

使用自建的PIC‑Set基准集，包含36个参数化的硅光子电路任务（包括MMI、耦合器、相位调制器、AWG、路由等），覆盖单元、二元到多元复杂度；

**📈 对比分析**

在Vanilla LLM生成与完整PICasso流水线两阶段对比实验中，结构与功能Spec@3从0%提升至高复杂度电路中最高52%；在GPT‑4o、Claude Sonnet 4.5、Llama 3.1 70B三大LLM上，PICasso在中等复杂度电路中实现99.2%结构Spec@3，功能Spec@3达100%；平均插入损耗从4.98 dB降低至3.25 dB（1.74 dB提升），并在4–8分钟内完成任务，显著快于手工GUI工作（25–60分钟）；

**⚠️ 局限性**

在高相位敏感、大规模电路（如64‑QAM调制器、90°光学混合器）中，Nelder–Mead优化器收敛不足，导致功能Spec仍为0；需引入更高级的优化策略（梯度或代理模型）以突破此限制。

---

## 334. Meta-Learning Where to Allocate Experts: Task-Conditioned Layer-Wise Compression for MoEs

**arXiv ID:** 2608.26650 | [PDF](https://arxiv.org/pdf/2608.26650v1)

**作者:** Rongfeng Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Hongwei Tang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在冻结的混合专家（MoE）大模型上，提出 MetaNet 控制器，利用任务支持集预测每一层的专家激活阈值与有界偏置，实现任务级动态专家分配。

**💡 创新点**

创新点在于：① 通过支持集推理的元学习框架，预测层级专家预算；② 只调整有限的路由偏置而不修改原始参数；③ 将专家激活决策分解为保留阈值与先验偏置两部分，兼顾准确性与激活成本。

**🔧 技术方法**

使用技术包括：支持集–查询范式、路由统计量（softmax 分布、熵、最大概率、前 R 专家质量）、两分支 MetaNet（预算分支与先验分支）、保留阈值 ρ 与有界 tanh 偏置 b、软 Top‑K 与累积质量阈值机制。

**📊 数据集**

主要实验数据集为 MMLU（多任务多选）、C‑Eval（多语言评测）、OLMoE‑1B‑7B MoE 基座以及使用 GoogLeNet 的合成视觉任务。

**📈 对比分析**

对比固定 k、HyperRouter、AdaMoE、Dynamic MoE 等基线，MetaNet 在 MMLU 上将专家激活从 6 降至约 2.28，准确率仅降低 0.036 点；在 C‑Eval 全量转移同样压缩激活比率；与固定 k=12 的精度差距较小但激活量显著减少；跨基座迁移也能提升性能。

**⚠️ 局限性**

局限性包括：仅降低路由激活比率，未直接减少模型存储或总体 FLOPs；对现有推理堆栈未实现等比例速度提升；验证主要集中在多选评测，生成任务或长上下文场景未知；仅在冻结模型上有效，需结合专用稀疏推理框架进一步提升实用性。

---

## 335. A Safety-Gated Multimodal AI Backend for Mental-Health Support: Hierarchical State Representation, Conservative Risk Fusion, and Controlled Generation in Anian

**arXiv ID:** 2608.26162 | [PDF](https://arxiv.org/pdf/2608.26162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 336. DSA: Evidence-Aware LLM-Agent Orchestration for Multi-Market Stock Research

**arXiv ID:** 2608.26990 | [PDF](https://arxiv.org/pdf/2608.26990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 337. FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation

**arXiv ID:** 2608.26645 | [PDF](https://arxiv.org/pdf/2608.26645v1)

**作者:** Ganlong Zhao `[一作]` (Chinese University of Hong Kong), Guanbin Li `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 FLARE 框架，结合 Retry 与 Reset 两种机制，为 Vision‑Language‑Action（VLA）模型实现鲁棒的错误恢复；

**💡 创新点**

创新点在于（1）将错误划分为 ID 与 OOD 两类，设计基于扰动‑桥接的 Augmentation 使 VLA 内置 Retry；（2）利用多模态大模型进行离线错误分析，自动挖掘 OOD 情景并快速收集、增强对应 Reset 技能；（3）通过 LoRA 模块化训练，将主任务与各 Reset 技能解耦，形成可扩展的技能库；

**🔧 技术方法**

使用的技术包括 VLA 模型（π_0.5 作为骨干）、扰动‑桥接数据增强、Gemini‑2.5‑Pro 作为多模态 LLM 进行失败识别与监控、LoRA 适配器训练、MimicGen 进行数据复合、以及模拟与实物机器人平台的视觉感知与控制；

**📊 数据集**

数据集主要来自 10 条人工演示（每任务）经过 MimicGen 及扰动‑桥接生成 500 条任务演示；Reset 技能通过 10‑20 条人工示例并同样进行扰动‑桥接扩增；实验使用 RoboMimic 9 种接触丰富的操作任务（Coffee、Stack 等）以及 Piper arm 在两项真实任务（Stack Three Blocks、Insert U‑shaped Block）；

**📈 对比分析**

在 9 任务上与 OpenVLA、Task‑Conditioned、Subgoal‑Conditioned、Motion‑Conditioned、Phoenix、Phoenix‑Human、π_0.5 等基线对比，平均成功率 84.0%（比 Phoenix 57.8% 提升约 26%，比 π_0.5 提升 11.8%），在 8/9 任务上实现 state‑of‑the‑art；

**⚠️ 局限性**

局限性包括：（1）对复杂姿态恢复（如 U‑shaped、T‑shaped 块）表现不足，需更高级的抓握与姿态调节；（2）Reset 技能库仍有限，需进一步扩展；（3）对大规模环境随机化的鲁棒性仍待验证；（4）依赖多模态 LLM 的错误分析，若 LLM 表现受限可能影响恢复质量。

---

## 338. Bayesian methods and Markov chain Monte Carlo algorithms for curve reconstruction and point cloud data analysis

**arXiv ID:** 2608.26490 | [PDF](https://arxiv.org/pdf/2608.26490v1)

**作者:** Asir Intesar Tushar `[一作]` (University of Tennessee), Ioannis Sgouralis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套完整的贝叶斯框架，用于对点云数据进行闭合曲线重建并量化不确定性。

**💡 创新点**

创新点包括：①使用非参数先验（Poisson/Binomial）对诱导点数目建模；②设计多层次、专门化的MCMC采样器（含自适应采样、椭圆切片采样及自定义Metropolis–Hastings）；③在单一推理阶段同时估计曲线连通性、位置与节点数；④通过后验分布实现不确定性传播。

**🔧 技术方法**

技术手段：贝叶斯层级模型、等方差高斯噪声、曲线上的均匀分布、Poisson/Binomial先验、Gibbs+自适应+椭圆切片+自定义Metropolis–Hastings的MCMC采样。

**📊 数据集**

使用的数据集包括：合成数据（Lissajous曲线、正八边形、正五边形）、基准数据（Butterfly、Sawtooth、Bottle）以及真实 LiDAR 数据（Mapping Rock Glaciers 2025、U.S. Geological 3D Elevation 2016）。

**📈 对比分析**

通过比较三种采样器的混合性和重建误差（Hausdorff距离），Sampler3 在 0.075 的误差上远优于 Sampler1（0.312）和 Sampler2（0.318）。与 FitConnect、StretchDenoise 和参考方法的定性对比表明，本方法在保持相似精度的同时提供了后验置信区间，并在样本稀疏或高曲率区域显著提升不确定性评估。

**⚠️ 局限性**

局限性：MCMC 采样在大规模点云或实时场景下计算成本高；目前仅适用于一维几何（曲线），不直接扩展到曲面或体积重建。

---

## 339. AI Control Scientist: LLM-driven Agentic System for Automated Control Design

**arXiv ID:** 2608.26780 | [PDF](https://arxiv.org/pdf/2608.26780v1)

**作者:** Haiteng Wang `[一作]`, Lei Ren `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于大语言模型的多智能体系统（AICS），能够从自然语言需求自动生成并验证控制器代码，并通过数值优化实现参数调优。

**💡 创新点**

创新点包括：① 任务建模、控制器生成、参数调优三代理协同迭代；② 通过逻辑驱动的控制器代码生成与控制理论验证实现结构化合规性；③ 将符号推理与高精度数值优化分离，采用元优化框架提升稳定性与效率。

**🔧 技术方法**

采用了大语言模型（Gemini‑2.0‑Flash 等）、领域知识库、控制理论验证引擎、外部数值优化器（如贝叶斯优化）以及闭环反馈机制。

**📊 数据集**

使用了扩展版 ControlEval 基准数据集，涵盖一阶/二阶稳定/不稳定、纯延迟以及高阶复杂系统共 50 个测试案例。

**📈 对比分析**

通过与 PIDtune、ControlAgent 以及零/少-shot LLM baseline 的对比，AICS 在各类系统中的通过率最高（如二阶不稳定 90%），且迭代次数显著更少（约 2–3 次）。

**⚠️ 局限性**

局限性在于：仍高度依赖大规模 LLM，数值优化成本较高；对极其复杂或高度非线性系统的解析能力有限；需人工构建知识库与验证工具。

---

## 340. FedCMAPSS: A Benchmark for Federated Learning in Remaining Useful Life Estimation

**arXiv ID:** 2608.26433 | [PDF](https://arxiv.org/pdf/2608.26433v1)

**作者:** Amelia Sorrenti `[一作]` (University of Catania), Simone Palazzo `[通讯]` (University of Catania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FedCMAPSS 基准，定义五个任务（IID、域移位、标签偏斜、特征偏斜、少样本）以评估联邦学习在剩余使用寿命预测中的表现。

**💡 创新点**

提供统一的数据分割、评估协议与可复现的实验框架，填补了 PHM 领域缺乏标准化联邦评估方法的空白。

**🔧 技术方法**

使用联邦学习算法（FedAvg、SCAFFOLD、FedDyn、FedCross）与多种深度网络（LSTM、RNN、CNN、AFT、AttBiGRU）进行对比。

**📊 数据集**

基于 NASA C‑MAPSS turbofan 传感器数据集的四个子集。

**📈 对比分析**

在10个随机拆分上平均比较 RMSE 与 NASA Score；结果显示在 IID 环境下 FedAvg 与 SCAFFOLD 性能最佳，域移位与标签偏斜场景下 drift‑mitigation 方法（FedCross、SCAFFOLD）表现更稳定；少样本任务最难，LSTM 与 CNN 更稳健。

**⚠️ 局限性**

局部超参数未针对联邦场景优化；NASA Score 对异常值极敏感，缺乏鲁棒度；未覆盖更多 FL 模式与更大规模数据集。

---

## 341. SLM-Conditioned Hierarchical Relation Routing for Labeled Property Graph Learning

**arXiv ID:** 2608.26132 | [PDF](https://arxiv.org/pdf/2608.26132v1)

**作者:** Michal Podstawski `[一作]` (NASK National Research Institute), Michal Podstawski `[通讯]` (NASK National Research Institute)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5005530395)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于小型语言模型的分层关系路由（SLM‑Conditioned Hierarchical Relation Routing）来将文本与属性语义直接注入图神经网络的消息传递过程；

**💡 创新点**

创新点在于：①将语言模型用作目标条件下的路由查询，而非单纯的编码器或预测器；②采用两级路由（同类型内聚合→跨类型聚合）并限制其对结构先验的修正为有界残差；③实现了对节点、关系属性的可解释性选择；

**🔧 技术方法**

核心技术包括：图结构GNN（顶层先验）、小型语言模型（Qwen2.5‑1.5B‑Instruct）通过LoRA实现参数高效微调、soft‑token提示生成、基于注意力的分层路由、Bounded Residual Prediction；

**📊 数据集**

使用三份公开的标注属性图数据集：
- 医疗事件（FDA AERS）
- 金融犯罪（FinCEN文件）
- 电影推荐（MovieLens 扩展），每个都转换为节点预测任务；

**📈 对比分析**

与三种基线对比：纯拓扑GNN、节点属性静态编码GNN、边属性静态编码GNN。路由器在健康与金融数据的不平衡二分类任务中显著提升macro‑F1与AUROC（+5.8、+4.8等），在电影推荐平衡任务中保持或略优；在多分类、回归与排序任务同样实现跨任务迁移并取得最高指标；

**⚠️ 局限性**

局限性包括：①仅在结构信息强时才能充分发挥；②有界残差限制了语义修正的幅度，可能在结构弱时难以补偿；③需额外的语言模型推理成本，虽采用量化与LoRA降低，但仍高于纯静态编码；④对语义分布过于均匀或扩散的任务效果有限。

---

## 342. hoBIT: A Profile-Aware Retrieval-Augmented Chatbot for University Academic Advising

**arXiv ID:** 2608.26604 | [PDF](https://arxiv.org/pdf/2608.26604v1)

**作者:** Yoonseo Kim `[一作]` (Korea University), SeongKu Kang `[通讯]` (Korea University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

把学校现有的规则式学术咨询机器人hoBIT改造成基于学生档案的检索增强生成（RAG）系统proFILL。

**💡 创新点**

创新点在于将学生档案信息嵌入检索索引，并通过查询驱动和证据驱动的自适应档案采集，实现只收集必要的档案属性，避免完整档案上传。

**🔧 技术方法**

采用离线档案条件索引、基于LLM的档案标注、软前缀注入与硬过滤相结合的检索，以及在检索后根据证据触发的二次检索与生成。

**📊 数据集**

使用了515个机构文档（规章PDF、网页、公告等）和1800个带合成学生档案的问答实例，覆盖60个学生档案与10类咨询主题。

**📈 对比分析**

与传统的BM25、dense、Hybrid、HyDE、reranker等RAG基线在部署和oracle两种设置下对比，proFILL在MRR、Recall@1/5/10/50、ROUGE-L、Keyword/Source Match、Grounded Correctness等指标均显著优于基线，平均终端延迟约6.2秒。

**⚠️ 局限性**

局限性包括仅在单一信息学院评估，需对不同院校调整档案模式；依赖用户自报档案，错误信息可能导致检索错误；未验证多语言和跨机构的泛化能力。

---

## 343. Zero-Shot Video Restoration and Enhancement with Text-to-Image Latent Diffusion Models and Multi-Modal References

**arXiv ID:** 2608.26476 | [PDF](https://arxiv.org/pdf/2608.26476v1)

**作者:** Cong Cao `[一作]` (Tianjin University), Jingyu Yang `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于预训练文本-图像潜在扩散模型的零拷贝视频恢复与增强框架，支持无参考、文本参考和图像参考三种多模态引导；通过双提示调优反演与采样、纹理感知视频令牌合并、参考自注意力和参考令牌合并等技术，显著提升了恢复质量与时序一致性，并将推理步骤从1000缩减至约360，速度提升约三分之一。

**💡 创新点**

创新点包括：1）双提示调优（Dual Prompt Tuning）在反演与采样阶段同时优化条件与无条件嵌入，以更好地编码退化图像并提升时间一致性；2）纹理感知视频令牌合并（Texture‑Aware Token Merging），在空间与时间维度分别为纹理丰富区和光滑区设定不同合并比例，兼顾细节与一致性；3）支持多模态参考的专用模块（参考自注意力、参考令牌合并），实现图像纹理的高效传递；4）在零拷贝场景下首次将文本与图像参考联合使用。

**🔧 技术方法**

核心技术包括：潜在扩散模型（Stable Diffusion v1.5 / SDXL）、DDIM反演与采样、双提示调优（条件与无条件嵌入优化）、纹理感知令牌合并（空间+时间）、参考自注意力、参考令牌合并、3D 卷积改造的 U‑Net、伪批一致采样等。

**📊 数据集**

实验使用了 REDS4、UDM10、DAVIS（恢复任务）、DID（低照度增强）、RealMCVSR（参考图像实验）以及 DAVIS（盲超分）等数据集。

**📈 对比分析**

与现有方法（PSLD、VISION‑XL、DiffBIR、DiffIR2VR、ZVRD 等）在 PSNR、SSIM 及 Warping Error（WE）上进行对比。结果显示：在 4× 视频超分上，ZVRM（无参考）相较 PSLD 提升约0.31 dB PSNR、WE 减少至约四分之一；在低照度增强上提升约0.23 dB PSNR、WE 减少约三分之一；使用文本+图像参考时进一步提升。与 VISION‑XL 相比，ZVRM 在所有指标上均优于其。

**⚠️ 局限性**

局限性包括：1）对参考文本/图像质量敏感，若描述不准确或参考图像不匹配可能导致恢复偏差；2）仍存在一定的计算成本，虽然已大幅缩短采样步骤，但在实时或高分辨率场景下仍显得较慢；3）在极端退化或复杂真实世界噪声下的鲁棒性尚未充分验证；4）论文未系统评估多模态参考的泛化能力，依赖人工选择参考样本。

---

## 344. Beyond the Proving Ground: Independent Public-Road Testing of Assisted Lane Change Systems using LiDAR

**arXiv ID:** 2608.26669 | [PDF](https://arxiv.org/pdf/2608.26669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 345. Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study

**arXiv ID:** 2608.26697 | [PDF](https://arxiv.org/pdf/2608.26697v1)

**作者:** Zhen Wang `[一作]` (Shanghai Qi Zhi Institute), Wei Liang `[通讯]` (Megatronix Technology Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个统一的基于音素的TTS到ASR的数据增强流水线，包含多语言TTS训练、文本候选选择、参考语音过滤和音素频率引导的文本筛选。

**💡 创新点**

提出了基于真实ASR训练标签的音素频率引导选择（PFGS）方法，并将其与随机选择和低频选择进行比较；同时首次系统评估了参考语音过滤对ASR性能的影响。

**🔧 技术方法**

使用F5‑TTS架构训练多语言共享TTS模型，加入语言ID嵌入；对文本进行语言特定的G2P转换；利用Fast‑Whisper校验参考语音；在ASR端采用WeNet‑CTC/Attention混合Conformer模型。

**📊 数据集**

采用阿拉伯语、法语、意大利语和葡萄牙语的自然语音和ASR训练数据（分别来自内部和公开语料库），候选文本来自Common Voice、MLS、FLEURS、M‑AILABS、VoxPopuli等公开语料，评估使用13个不同的测试集。

**📈 对比分析**

通过在随机合成比例（0%–100%）和固定60%预算下的PFGS、低频和随机文本选择进行对比实验，结果显示随机增强在11/13测试集上优于仅用真实数据；在60%预算下PFGS相较于随机选择在9/13测试集上降低WER，最高相对下降19.3%，并在12/13测试集上优于仅用真实数据；参考语音过滤在法语和意大利语Common Voice上分别降低0.59和0.29个绝对WER点。

**⚠️ 局限性**

增益并非均匀，受语言、测试域、TTS质量和候选文本分布影响；在部分语言或域中未见显著提升；对真实语料覆盖度不足和域不匹配等问题的解释仍待进一步研究。

---

## 346. Preference Flow Matching with Spectral Factorization for Micro-video Recommendation

**arXiv ID:** 2608.26579 | [PDF](https://arxiv.org/pdf/2608.26579v1)

**作者:** Xinxin Dong `[一作]` (National University of Defense Technology), Xiaodong Wang `[通讯]` (National University of Defense Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PrismRec，一种将微视频帧级表示分解为静态语义和动态特征，并通过上下文校准的流匹配过程来生成个性化偏好表示，从而提升微视频推荐效果。

**💡 创新点**

创新点包括：① 用频谱分解（Spectral Semantic Factorization）将视频内容拆解为静态与动态两类因子；② 在流匹配（Flow Matching）框架中引入上下文校准（Context-Calibrated Preference Matching）让视频因子直接引导偏好转移轨迹；③ 将视频因子视为推荐的内在驱动力而非辅助信息。

**🔧 技术方法**

核心技术：频谱分析（FFT + 可学习频率掩模）、双分支异构融合、基于流匹配的连续传输（Conditional Flow Matching）、Transformer 序列编码、基于注意力的上下文校准。

**📊 数据集**

使用四个真实微视频数据集：MicroLens-Small、MicroLens-Big、Shortvideo-Small、Shortvideo-Big，涵盖多模态文本与视频特征。

**📈 对比分析**

与十四种基线（协同过滤、序列推荐、多模态推荐、视频推荐等）进行对比。PrismRec 在所有数据集与指标（H@10/H@20、N@10/N@20）上均显著优于基线，提升幅度最高可达 22.65%，同时保持最低的推理时间与显存占用。

**⚠️ 局限性**

局限性：① 仍需较高的帧采样与频谱计算成本，尽管相对低于对比模型；② 对极端长序列或极少交互的视频仍可能受限；③ 目前仅在短视频领域验证，跨域适用性尚待进一步探索。

---

## 347. Agent Mesh: Reliability Primitives for Non-Idempotent Agent Delegation - Identity Adequacy and Evidence Adequacy

**arXiv ID:** 2608.26225 | [PDF](https://arxiv.org/pdf/2608.26225v1)

**作者:** Mazhar Shaikh `[一作]`, Harshal Pathak `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对一套生产级代理软件交付平台进行了失败案例研究，分析了147个故障实例，评估了传统服务网格所依赖的幂等性、延迟信号和无成本丢弃假设在实际代理工作中的失效，并基于此提出了七个可靠性原语来修正这些失效。

**💡 创新点**

创新点在于：①发现身份不充分与证据不移动这两种跨子系统的缺陷共同导致错误决策；②提出并验证了专为代理委托设计的七大可靠性原语（进度打破器、效应契约、效应账本、预算格子、故障路由、执法层排除、非确定性隔离）；③将这些原语与服务网格、事务隔离等传统可靠性机制进行对比，展示其在代理场景中的独特价值。

**🔧 技术方法**

技术与实现：使用基于三接口（模型调用、工具调用、提交）拆解的代理框架；利用进度信号与恒定词汇过滤、账本去重、预算退回、基于检查点的故障路由、执法层可验证性以及工作区+环境+合同摘要的隔离等技术；并在同一平台上实现了多种验证与恢复路径。

**📊 数据集**

数据集：147个故障记录（包含成本、修复证明等），平台源码66,185行、59模块，工作负载平均78个源文件、2,231行Python/TypeScript；记录了81次独立运行、147次失败事件，覆盖不同模块与工具调用。

**📈 对比分析**

比较方法：当前为观察性研究，随后提出两臂对照实验（naive服务网格策略 vs. Agent Mesh原语）用于量化原语效果；已报告的改进包括：进度打破器从四条恢复路径中将“判定为停滞”的比例从100%降至0%；预算退回减少了重复执行；故障路由将影响范围从5降至2；但正式的对照实验尚未完成，性能指标仍待实验验证。

**⚠️ 局限性**

局限性：仅在单一系统内进行自诊断且缺少随机对照；未实现部分原语（如效应账本、成本账本、分租户账本）；依赖平台特定的工具接口和执行沙箱，结果可能与其他代理框架差异；对失效频率和成本的统计受已记录故障偏倚影响；对比实验设计为准实验而非严格随机实验，因而无法提供精确效益估计。

---

## 348. Launch-Bound and Substitutable: Why Three Inference Optimizations Fail to Pay Off in Mixture-of-Experts Models

**arXiv ID:** 2608.26612 | [PDF](https://arxiv.org/pdf/2608.26612v1)

**作者:** Gokulakannan Sakthivel `[一作]` (University of Maryland), Giriprasad Radhakrishnan `[通讯]` (University of Maryland)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对三大MoE模型（OLMoE、DeepSeek-V2-Lite、Qwen3-30B-A3B）进行系统评估，探讨了Triton核融合、INT4/INT8量化以及torch.compile编译等常见推理优化在实际部署中的收益与瓶颈。

**💡 创新点**

创新点在于：①使用Amdahl分析量化推理的上限并发现模型受“发射绑定”限制；②通过路由漂移与质量损失的因果重放实验表明专家可替换，漂移仅占约2.7%的误差；③揭示图编译的破裂计数并非性能指标，零破裂会导致三倍慢。

**🔧 技术方法**

技术手段包括：自定义Triton融合核、bitsandbytes 4/8位量化、torch.compile + TorchDynamo、Modal服务器无状态容器化、per‑token路由记录与重放、Amdahl、统计与相关性分析。

**📊 数据集**

数据集：使用约119,952个MMLU风格提示的token序列作为路由记录与NLL评估；对比FP16、INT8、INT4等精度下的任务准确率。

**📈 对比分析**

对比方法：以FP16基准为参照，测量单步延迟、吞吐量、内存占用，并计算Amdahl上限（1.07×）与实际加速（≤1.00×）；量化漂移与质量损失的相关性为ρ≈0.9；编译破裂计数从19/36到0对速度影响三倍。

**⚠️ 局限性**

局限性：因果干预仅在OLMoE上完成；量化使用单一库，未验证其他4位方法；实验仅涵盖三模型、单一A100 GPU；编译实验未启用cuDNN，结果对其他硬件或量化策略的泛化有限。

---

## 349. Hadamard Flattening and Gaussian Pooling Sketch for Least Squares with Coordinate-wise Guarantee

**arXiv ID:** 2608.26552 | [PDF](https://arxiv.org/pdf/2608.26552v1)

**作者:** Zhao Song `[一作]`, Lichen Zhang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

针对过约束 ℓ₂ 回归问题，提出了一种新的快速稠密随机化变换，利用随机 Hadamard 压平、随机置换和平衡互不相交的高斯池化来压缩输入矩阵，并在压缩后保证解在 ℓ∞ 范数下与最优解近似。

**💡 创新点**

创新点在于构造了一个能够在 Hadamard 与置换阶段后产生条件独立性的稠密变换，从而把压缩后的回归问题等价于完全独立的高斯回归。利用这一条件独立性，作者证明了仅需 m = O(ε⁻² d log d) 行即可实现 ℓ∞ 误差保证，显著优于此前 SRHT 的 O(ε⁻² d^{1+Θ(√(loglog n / log d))}) 行数以及错误独立性假设下声称的 O(ε⁻² d log³ n) 行数。

**🔧 技术方法**

采用了随机 Hadamard 变换、随机置换、平衡且互不相交的高斯池化等技术，并结合条件独立性分析与高斯回归理论实现压缩与误差保证。

**📊 数据集**

本工作为理论研究，并未使用公开数据集；主要通过数学证明与理论分析验证其效果。

**📈 对比分析**

与先前基于 SRHT 的方法和错误独立性假设的方案相比，本方法在行数上从 O(ε⁻² d^{1+Θ(...)}) 改进至 O(ε⁻² d log d)，计算复杂度为 O(nd + ε⁻² d⁴)，在保持压缩效率的同时提供了更强的坐标级误差保证。

**⚠️ 局限性**

限制在于需要内部维度 N = O(n + ε⁻² d³)，导致计算复杂度仍包含 ε⁻² d⁴ 项；此外，该方法仅针对 ℓ₂ 回归，尚未推广到其他范式或非线性模型。

---

## 350. TemporalFlow-VLA: Learning Physically Grounded Execution History for Long-Horizon Robot Manipulation

**arXiv ID:** 2608.26821 | [PDF](https://arxiv.org/pdf/2608.26821v1)

**作者:** Jiarui Yang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Enyu Li `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 TemporalFlow-VLA，一种利用物理引导的机器人表面时间流监督的 VLA 模型，学习压缩的执行历史并将其注入到动作专家中；

**💡 创新点**

创新点在于：①通过机器人关节状态、URDF 及相机标定生成无标注的机器人表面时间流作为训练监督；②设计两级时间查询 Q₈ 与 Q₁₅，分别对应最近与全段历史；③使用异步历史特征缓存，避免推理时对历史帧的重复编码；

**🔧 技术方法**

技术手段包括：预训练的 Vision‑Language‑Action 模型（π₀.₅）、基于 kinematics 的时间流渲染与监督、FiLM 模块的流重建头、联合掩码自注意力、异步特征缓存与时间窗口编码；

**📊 数据集**

数据集主要有：LIBERO（Spatial、Object、Goal、Long 四个子集）和 RoboTwin 2.0（12 任务，含清洁与随机化版本），另外在 AgiBot A3 机器人上做了实机验证；

**📈 对比分析**

与多种基线（π₀、π₀.₅、X‑VLA、GigaWorld、CronusVLA、HAMLET、MotionVLA 等）对比，TemporalFlow‑VLA 在 LIBERO Long 上平均成功率 96.60%，在 RoboTwin 任务清洁/随机化分别达到 85.5%/84.2%，在 12 个挑战性任务中平均 85.5%/84.2%，并在长时程多阶段任务中显著超越基线；

**⚠️ 局限性**

限制在于：①仅对固定时间窗口（t‑15、t‑8）进行历史建模，未探索更灵活的时间尺度与采样策略；②时间流监督依赖机器人 kinematics 与相机标定，对硬件误差或缺失信息敏感；③在极端动态场景下仍可能缺乏足够的历史分辨率。

---

## 351. FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs

**arXiv ID:** 2608.26746 | [PDF](https://arxiv.org/pdf/2608.26746v1)

**作者:** Zeming Liu `[一作]`, Jingtao Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种结合活跃覆盖与结构多样性的混合排序方法，利用已执行的稀疏结果缓存学习压缩后的行为测试集，用于验证生成的运算策略。

**💡 创新点**

创新点在于：①将动态杀伤信息与结构多样性两种排序机制交替融合，避免单一贪婪过拟合；②通过稀疏缓存实现对任意子集的快速杀伤评估；③在交叉程序和全族留置两种泛化协议下验证方法的鲁棒性。

**🔧 技术方法**

核心技术包括：稀疏结果缓存（存储杀伤探测列表）、活跃贪婪集覆盖排序、基于探测结构特征的多样性排序、混合排序交替策略、以及基于下游运营指标的安全接纳规则。

**📊 数据集**

实验数据集为20份Python生成策略，涵盖4个资源选择场景，10个随机种子，共计1,200条测量；进行2,160种控制变换，生成4,120,200个程序-探测执行对。

**📈 对比分析**

与纯活跃、随机、频率排序等基线对比，混合排序在32个探测时实现99.0%跨程序故障覆盖，并在全族留置下宏观覆盖从84.6%提升至94.9%；同时在下游部署中显著降低严重尾部回归数量。

**⚠️ 局限性**

局限性包括：仅评估短小的Python策略；变换为人工控制而非真实错误；有限域等价性假设；对隐藏状态或探测顺序的精确性缺乏正式证明；以及对分布式或并发组件的适用性尚待验证。

---

## 352. Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update

**arXiv ID:** 2608.26511 | [PDF](https://arxiv.org/pdf/2608.26511v1)

**作者:** Huanhuan Ma `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种两轮诊断评估框架，将语言模型在用户推压与提供证据下的答案修正行为区分为“Unsupported‑Yielding”和“Rational‑Updating”，并研究两种行为的交互及其机制。

**💡 创新点**

创新点在于首次将sycophancy视为选择性问题，而非单纯抑制问题；通过对两类答案修正进行独立测评，并揭示它们共享内部子结构和方向，使得抑制一类行为往往损害另一类行为。

**🔧 技术方法**

主要技术包括对抗性数据生成与DPO/SFT微调、激活层/注意头的方向推导与正交化 steering、梯度基归因与交叉补丁验证、以及对内部激活的统计分析。

**📊 数据集**

使用了四个公开基准（TruthfulQA、PopQA、EX‑FEVER、AQuA），每个数据集都配备相应的黄金证据。

**📈 对比分析**

实验表明，除非采取正交化 steering，常见的 anti‑sycophancy 介入（如 DPO、SFT、激活 steering）在降低 Unsupported‑Yielding 的同时往往削弱 Rational‑Updating；在联合优化下仍存在部分数据集表现不佳。

**⚠️ 局限性**

限制包括仅评估四个开源模型，机制分析粒度为 MLP 神经元/注意头，未考虑检索噪声和假证据；正交化 steering 的效果有限，且仅在 TruthfulQA 上做了预实验。

---

## 353. An Economic Analysis of DNA-based Data Storage Systems

**arXiv ID:** 2608.26342 | [PDF](https://arxiv.org/pdf/2608.26342v1)

**作者:** Alex El-Shaikh `[一作]` (Imperial College London), Thomas Heinis `[通讯]` (Imperial College London)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个参数化成本模型，评估DNA存储与磁带、云归档等传统存储技术在不同技术突破情景下的经济可行性。

**💡 创新点**

创新点在于将DNA合成与测序成本的指数衰减趋势与传统存储成本模型结合，量化了DNA存储在多种成本下降率下实现经济竞争的具体阈值。

**🔧 技术方法**

采用指数回归拟合历史成本数据，构建写入、读取和运行成本函数，并在仿真中评估不同降价率情景，比较总成本曲线。

**📊 数据集**

使用公开的DNA合成/测序价格历史、Amazon S3 Glacier Deep Archive、Azure Blob Archive、磁带的价格数据作为输入。

**📈 对比分析**

通过比较总成本曲线和读写成本，结果表明在当前趋势下DNA存储需下降八到九个数量级才与传统存储竞争；在最乐观情景下，到2050年读写成本可与云归档相当。

**⚠️ 局限性**

主要限制在于DNA合成成本仍远高于传统存储，模型假设运行成本为零，未考虑实验室操作复杂性、可扩展性和实际可访问性问题。

---

## 354. ADeptS-Bench: Measuring the Trustworthiness of Computer Use Agents Across Devices

**arXiv ID:** 2608.26204 | [PDF](https://arxiv.org/pdf/2608.26204v1)

**作者:** Joy Chen `[一作]` (FAIR at Meta), Joseph Tighe `[通讯]` (FAIR at Meta)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个针对计算机使用代理的双流安全与歧义性评估基准，该基准采用配对安全/恶意任务和单步决策，覆盖移动和桌面平台，并通过用户研究确定风险优先级。

**💡 创新点**

创新点在于将安全与歧义性评估结合，使用离线视觉输入、配对任务设计和基于用户偏好的风险分类，实现跨平台可比性和系统化安全度量。

**🔧 技术方法**

使用了LLM与工具调用、图像视觉模型、LLM判断器、自动评分器，并通过单步API调用实现离线评估。

**📊 数据集**

使用了自建的1,718对安全任务（358移动、501桌面）和744歧义任务（381移动、363桌面）截图集，并基于1,300名美国受访者的MaxDiff调查确定风险权重。

**📈 对比分析**

比较方法采用TSR、ASR、FRR、Score、Disambiguation F1和Severity Calibration Error等指标，实验显示前沿模型在安全/能力之间存在权衡，最佳模型Gemini 3.1 Pro的Score为76%，但无模型同时在80%以上能力且低于30%攻击率，显示安全与功能的矛盾。

**⚠️ 局限性**

局限性包括单步评估无法覆盖多步轨迹安全、离线视觉输入可能忽略动态威胁、受限于英语Android/Windows平台、对高阶视觉欺骗的检测不足，以及LLM判断器对问答匹配的误差。

---

## 355. Active Curriculum Refinement for Reinforcement Learning

**arXiv ID:** 2608.26469 | [PDF](https://arxiv.org/pdf/2608.26469v1)

**作者:** Zhenya Liu `[一作]` (University of Chicago), Yuxin Chen `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PATH框架，利用主动路径学习在已知的课程DAG上高效训练强化学习策略，显著提升鲁棒性与泛化能力。

**💡 创新点**

将课程结构形式化为有向无环图，并设计两阶段（随机探索+基于回报的主动采样）主动路径选择机制，首次在课程图上实现路径级主动学习。

**🔧 技术方法**

采用PPO训练策略、随机路径采样、基于回报的权重更新（类似PLR）、早停与耐心阈值、路径指针与动态缓冲区管理等技术。

**📊 数据集**

在MiniGrid（离散导航）、BipedalWalker（连续控制）和Procgen Leaper等公开基准上进行实验。

**📈 对比分析**

与DR、PLR、ACCEL等传统与最新课程学习方法比较，PATH在IQM、解决率和平均回报等指标上均超过对照组，MiniGrid约99.4%解决率，BipedalWalker获得最高平均回报与解决率。

**⚠️ 局限性**

依赖已知课程DAG，早停阈值需要手动设定，对噪声或不完整图结构的鲁棒性有限，仅在UED相关基准上评估，缺乏对更广泛课程学习框架的验证。

---

## 356. "A Second Set of Eyes": The Process and Challenges of Software Documentation Review

**arXiv ID:** 2608.26232 | [PDF](https://arxiv.org/pdf/2608.26232v1)

**作者:** Avinash Bhat `[一作]` (McGill University), Jin L. C. Guo `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究软件文档审查流程，识别了五个阶段（自检、技术审查、编辑审查、试玩测试、发布后反馈），并分析技术作家与开发者、PM、QA等多角色在各阶段中的协作与工具使用；

**💡 创新点**

将文档审查定位为“可见的协作工作”，提出基于干预时机与决策责任的设计空间，揭示工具与组织在支持协作上的不足，并为未来工具与流程改进提供经验基础；

**🔧 技术方法**

采用半结构化访谈、屏幕录制、定性编码、成员检查、工具映射等方法，系统地梳理实践者的工作与工具生态；

**📊 数据集**

访谈样本为31名经验丰富的技术作家，包含访谈记录与屏幕录制的原始数据；

**📈 对比分析**

本研究不进行量化性能评估，而是通过主题饱和度、访谈者验证和成员检查来确保研究结果的可靠性与可信度；

**⚠️ 局限性**

局限性：仅收集经验丰富技术作家的视角，缺乏开发者、PM等角色的反馈；工具研究主要聚焦现有工具而非新实现；LLM在文档审查中的应用受限于上下文与安全政策。

---

## 357. Energy-Neutral Coverage Optimization by Joint Deployment and Scheduling in Ambient IoT Devices with Directional Sensing

**arXiv ID:** 2608.26944 | [PDF](https://arxiv.org/pdf/2608.26944v1)

**作者:** David E. Ruíz-Guirola `[一作]` (University of Oulu), Onel L. A. López `[通讯]` (University of Oulu)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在能量收集与方向性感知约束下的AIoT设备的联合部署与时序调度，以实现能量中性条件下的覆盖最大化。

**💡 创新点**

创新点在于提出了一种混合LP+RL框架，先用LP得到可解释的初始部署与能量受限的占空比，再通过强化学习细化位置与占空比，实现对能量、方向视场与覆盖的三维协同优化；该方法显著提升了低至中等能量预算下的有效覆盖率。

**🔧 技术方法**

使用的技术包括：离散化覆盖矩阵构建、最大覆盖混合整数规划及其LP松弛、基于占空比约束的连线式多目标强化学习（策略梯度）以及混合优化+强化学习的自适应初始化。

**📊 数据集**

实验使用人工生成的模拟场景（二维20×20m 区域），设备密度从10^-2到1设备/㎡，视场角分别为60°、180°、360°，能量预算 f_j,max 取10^-3、10^-2、10^-1，采用蒙特卡洛模拟150次评估。

**📈 对比分析**

与传统网格+静态占空比、纯LP优化以及纯RL策略进行对比，结果显示混合LP+RL在所有视场与能量预算下平均有效覆盖率均高于其他方法，最高可提升约100%相对基线；同时收敛速度提升近10倍，整体计算时间减少约89%。

**⚠️ 局限性**

主要局限包括：模型基于离散化网格且假设能量收集为i.i.d. 二元过程，未考虑设备移动成本与实际部署误差；强化学习的奖励权重对结果仍有一定影响，且对大规模部署的可扩展性需进一步验证。

---

## 358. Accelerating Scientific Research with Gemini in the Real-World

**arXiv ID:** 2608.26701 | [PDF](https://arxiv.org/pdf/2608.26701v1)

**作者:** Samuel Schmidgall `[一作]` (Google DeepMind), Tao Tu `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种多代理系统 Co-Scientist，能够在理性推理、自动代码生成、实验执行和论文撰写之间实现闭环，支持从材料合成、生物实验到医学问答等多领域的全流程实验与验证，并在真实实验室环境中实现了安全前驱体发现、单次单层 TMD 合成、E. coli 群聚形态预测以及医学回答体系的自动设计。

**💡 创新点**

创新点包括：① 将迭代推理与自主代码执行结合，形成实验前后自动验证的闭环；② 引入联合优化与日志校验机制，显著降低hallucination、plagiarism 和实验数据造假；③ 设计双层安全架构，实现对双用途研究的自动拒绝与持续伦理监督；④ 在三大科研领域（材料、生命科学、计算机科学）上实现全流程自动化，首次系统性评估了从想法到实验再到论文的可信度。

**🔧 技术方法**

核心技术包括：多代理进化搜索（基于 Bayesian 评分与 UCB）、LLM 代理与同行评审、自动代码生成与多阶段执行（构架、转换、完整执行）、日志校验与强制纠错、联合优化目标（奖励 + 抄袭/hallucination 罚项）、Gemini 3 Deep Think 推理、可解释性评估（拉回抽样、批判循环）以及双层伦理审查模块。

**📊 数据集**

使用的主要数据集与实验资源：材料学方面的 CVD 设备、XRD/SEM/TEM/EDS 数据；生物学方面的 IPTG 梯度下的稀疏 swarming 图像集；医学方面的合成健康查询语料与 HealthBench Hard/Professional 基准；计算机科学方面为 Gemini 自动生成的 50 个 AI 研究主题及公开开源的算法库。

**📈 对比分析**

与传统单一模型（Gemini、GPT‑5、Claude、Gemini 3.5 Flash 等）相比，Co‑Scientist 在材料合成中实现了 100% 安全前驱体成功率，单层 MoS₂ 等 TMD 仅单次即可得到高质量晶体；在生物学实验中预测精度与实验结果在 3/4 形态指标上无显著差异；在医学回答上 Agent_H 在 HealthBench Hard/Professional 的加权 Rubric 评分均优于所有基线，且在长度校准后差距更大；在完全自主论文生成评估中，可靠模块将严重 hallucination 降至 4%（相比基线 90%），极端抄袭率降至 2%（相比基线 60%）。

**⚠️ 局限性**

局限性包括：① 仍存在选择性报告、方法与代码不一致等残留错误；② 主要适用于可自动执行的轻量级实验或模拟，无法处理大规模分布式训练与资源密集型任务；③ 日志校验与 LLM 评审的精度受限，无法完全替代人工复核；④ 安全边界识别仍有误判，尤其在模棱两可的双用途请求中误拒 3%；⑤ 需要持续的人工监督以调节任务框架与实验细节，系统仍未实现完全无人值守。

---

## 359. Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation

**arXiv ID:** 2608.26159 | [PDF](https://arxiv.org/pdf/2608.26159v1)

**作者:** Jesse St. Amand `[一作]` (MARS), Lennie Wells `[通讯]` (University Of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大型语言模型（LLM）的自生成文本识别（SGTR）能力，并探讨其训练可行性和对安全评估的影响。

**💡 创新点**

提出“操作化”框架对SGTR进行多维度评估，发现质量启发式是主要混淆因素；证明SGTR可通过SFT学习并能跨任务、格式迁移。

**🔧 技术方法**

使用多种评估格式（pairwise/individual）和对话结构（user/assistant标签），对13–21个LLM进行SGTR测评；利用LoRA SFT训练SGTR模型；在AlpacaEval 2.0中评估自我偏好。

**📊 数据集**

采用四个公开数据集（WikiSum、ShareGPT、PKU‑SafeRLHF、BigCodeBench）共250个提示，生成模型输出供评估。

**📈 对比分析**

通过Elo分数距离与准确率统计比较，pairwise+user‑tag操作化下准确率高且分布宽；SGTR训练后在未见操作化中提升约0.1–0.3点；在AlpacaEval中自我偏好提升至+0.76。

**⚠️ 局限性**

限制在四个任务域、两种评估格式和对话结构；训练规模有限，未深入内部机制；未评估长文本、多轮对话或创意生成等场景。

---

## 360. On the Instance Optimality of Bidirectional Dijkstra's Algorithm

**arXiv ID:** 2608.26952 | [PDF](https://arxiv.org/pdf/2608.26952v1)

**作者:** Matic Požar `[一作]` `[通讯]` (University of Primorska), Matic Požar (University of Primorska)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文重新审视了最短路径算法的实例最优性，特别是单向和双向Dijkstra算法在加权和无权图中的表现，提出了一些反例并对现有结果进行了修正。

**💡 创新点**

提出了一种简单的修改双向Dijkstra算法的方法，使其在加权情况下实现实例最优性，并简化了现有结果的证明。

**🔧 技术方法**

使用了修改后的双向Dijkstra算法，并结合了结构性特征的快捷算法来证明实例最优性。

**📊 数据集**

使用了多种图实例，包括特定的图类（如G_D图）来展示算法的性能和反例。

**📈 对比分析**

与现有的Dijkstra算法进行比较，修改后的双向Dijkstra算法在特定图类上表现出实例最优性，而传统的Dijkstra算法在某些情况下表现不佳，查询复杂度显著高于修改后的算法。

**⚠️ 局限性**

在简单图中，实例最优性的证明仍然是一个开放问题，尽管在某些特定条件下（如节点数与查询复杂度的关系）已建立了实例最优性。

---

## 361. PRO-RAN: Processor-Level Characterization of Open RAN Centralized and Distributed Units

**arXiv ID:** 2608.26498 | [PDF](https://arxiv.org/pdf/2608.26498v1)

**作者:** Moojan Kamalzadeh `[一作]` (University of Texas at Dallas), Andrea Fumagalli `[通讯]` (University of Texas at Dallas)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对5G O‑RAN中独立部署的CU与DU进行过程级CPU热图与微架构分析的实验框架，并在匹配的基线与持续负载条件下对其执行行为进行量化。

**💡 创新点**

创新点在于：①将过程级Hotspots与Top‑Down微架构分析相结合，直接对CU与DU的函数级CPU时间与微架构瓶颈进行对比；②使用Ansible自动化部署与验证，保证实验条件的一致性；③首次在同一硬件平台上实现CU与DU的并行、独立分析，为功能级资源规划与加速设计提供依据。

**🔧 技术方法**

主要技术包括：Linux Foundation OCUDU（CU/DU实现）、ZMQ‑based 无线接口、Open5GS核心网络、Intel VTune Profiler（Hotspots 与 TMA）、Ansible自动化脚本，以及TRex与客户端产生的UDP流量。

**📊 数据集**

使用的数据集为合成流量（TRex 生成的DL UDP + 端点发送的UL UDP）和系统自带的注册与会话控制信令，没有外部真实数据集。

**📈 对比分析**

通过对比CU与DU在相同硬件、相同流量下的CPU时间（CU 17.3→37.4 s，DU 462.0→628.4 s）以及微架构占比（CU后端绑定从47.3 %升至58.2 %，DU保持约50 %），验证了CU在负载激增时更易受后端资源限制，而DU保持持续的基线负载。

**⚠️ 局限性**

局限性包括：仅测试单UE、单一频段与固定流量配置；未覆盖多UE场景、不同功能拆分或多种硬件平台；缺乏真实业务数据集，仅基于合成流量；并未对结果进行进一步的模型化或预测。

---

## 362. Tissue-Mixture Entropy-Weighted Reconstruction for Partial-Volume-Aware Brain MRI Super-Resolution

**arXiv ID:** 2608.26647 | [PDF](https://arxiv.org/pdf/2608.26647v1)

**作者:** Xiao Tong `[一作]` (Peking University), Jinbo Yang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了AGW-PBR模型，结合低分辨率仅依赖的解码器与基于PVE平衡的训练目标，实现脑T2加权MRI的超分辨率重建。

**💡 创新点**

创新点在于使用质量控制的固定PVE伪标签计算组织混合熵，作为空间加权来强调组织过渡区；以及采用软潜在基字典与基于高斯参数的网格约束Warping。

**🔧 技术方法**

主要技术包括RCAB U-Net特征提取、Sobel梯度引导、温度缩放软赋值潜在字典、基于高斯参数的网格约束残差Warping、Charbonnier损失与熵加权L1损失。

**📊 数据集**

使用IXI T2图像（329/47/99主干分割）和fastMRI T2图像（200/30/43）进行训练与评估；此外在BraTS2023 T2上评估下游肿瘤分割。

**📈 对比分析**

与SRCNN、RCAN、SwinIR、HAT、ArSSR、GaussianSR、NExpR、Res-SRDiff等8种基线对比，在IXI的2×/4×/6×上PSNR/SSIM均为最高，4×的组织接口区PSNR/SSIM提升约1-2 dB；fastMRI 4×也取得最高的PSNR/SSIM。

**⚠️ 局限性**

局限包括：仅使用合成中心k空间截断的降采样；仅处理二维切片，缺乏体素一致性；未在真实LR/HR对上验证；仅限脑T2，加权方式需针对其他组织/模态进一步验证。

---

## 363. Don't Overthink, Don't Underthink: Toward Adaptive Reasoning in Agentic AI

**arXiv ID:** 2608.26442 | [PDF](https://arxiv.org/pdf/2608.26442v1)

**作者:** Md Jueal Mia `[一作]` (Florida International University), M. Hadi Amini `[通讯]` (Florida International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在agentic AI系统中推理分配的效率问题，探讨过度推理和不足推理的失败模式；

**💡 创新点**

提出了需要动态自适应推理控制器，以根据任务需求调整推理量、工具调用和停止点；

**🔧 技术方法**

利用LangGraph框架、Qwen3.5-4B、Llama-3.1-8B-Instruct与Phi-4-reasoning等大型推理模型，并用GPT-4.1进行推理分类；

**📊 数据集**

使用MATH‑500（500道数学题）和GAIA公共验证集（165道多步任务）进行实验；

**📈 对比分析**

通过对比三种最终模型在准确率、推理tokens、工具调用、token上限命中率等指标，发现Phi‑4-reasoning在准确率上最高但推理成本和token上限命中率也最高；

**⚠️ 局限性**

局限性包括仅使用固定工具路由器、未覆盖多种agent架构、未评估能源或费用、缺乏置信区间与分布分析，以及未考虑工具路由失败等因素。

---

## 364. CG4AI: A Column Generation Framework for Training AI Models Under Constraints

**arXiv ID:** 2608.26375 | [PDF](https://arxiv.org/pdf/2608.26375v1)

**作者:** Youcef Magnouche `[一作]`, Pierre Bauguion `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CG4AI框架，利用列生成构建满足硬线性约束的凸集成AI模型，保证训练后输出始终可行；

**💡 创新点**

创新点在于将输出约束直接写入主LP，利用LP对偶变量引导新列训练，并通过切割平面扩展连续约束的可行性保证；

**🔧 技术方法**

采用线性规划、列生成、对偶导向的训练子问题、黑盒分离器（或MIP分离）等技术，并将多种AI模型作为列；

**📊 数据集**

实验使用MNIST手写数字数据集和SNDLIB通信网络基准进行多商品流路由；

**📈 对比分析**

与单模型、惩罚项、增强拉格朗日、DC3、投影等方法对比，CG4AI在准确率上显著提升（如MNIST上+8pp，MCFP上保持0%违规），且约束满足率为100%；

**⚠️ 局限性**

局限性包括列生成与LP求解的计算开销、对偶变量收敛不一定保证全局最优（尤其非凸列），仅支持线性约束，且在极大规模网络/模型时易出现切割平面过多导致收敛缓慢。

---

## 365. Are We Shooting Flies with Cannons? Trade-off Analysis for AI-based 5G Intrusion Detection

**arXiv ID:** 2608.26844 | [PDF](https://arxiv.org/pdf/2608.26844v1)

**作者:** Federica Uccello `[一作]` (Linköping University), Simin Nadjm-Tehrani `[通讯]` (Linköping University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究5G网络入侵检测任务中传统机器学习（XGBoost、TabNet）与大型语言模型（Open‑Mistral‑7B）在检测性能、推理时间及CPU时长等维度的折衷关系；

**💡 创新点**

首次将LLM在零射击和少射击提示下与专门的表格数据模型在同一5G入侵检测数据集上进行系统对比，并以CPU时长为能耗代理量评估计算成本；

**🔧 技术方法**

采用XGBoost、TabNet、Open‑Mistral‑7B（本地CPU推理）以及Python实现的零射击/少射击提示技术；

**📊 数据集**

5G‑NIDD数据集（约120万条、92维特征，按流量标签为benign/ malicious），对训练、验证、测试三部分进行分层划分；

**📈 对比分析**

在同一预处理后的特征空间上训练XGBoost、TabNet，使用全量测试集评估准确率、平衡精度等指标；LLM仅在300条代表性样本上评估零射击与少射击，结果显示XGBoost/TabNet接近完美（Acc≈0.99），LLM表现差（Bal Acc 0.67/0.50），LLM推理时间≈10 s/样本、CPU时长≈20 k s，而XGBoost仅≈1 s、CPU时长≈1 s，证明传统ML在性能–成本上优于LLM；

**⚠️ 局限性**

仅使用单一数据集和单一LLM模型；LLM仅在本地CPU环境下部署，未评估GPU或API端实现的真实能耗；提示设计对LLM性能影响大；未覆盖无标签或异常检测场景；结果可能不具备对更大规模或多样化数据集的普适性。

---

## 366. AgentJudgeBench: A Multi-Difficulty Benchmark for Evaluating LLM Judges on Agentic Tool-Calling

**arXiv ID:** 2608.26623 | [PDF](https://arxiv.org/pdf/2608.26623v1)

**作者:** Abhigya Verma `[一作]` (ServiceNow AI), Sai Harshitha Aluru `[通讯]` (ServiceNow AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AgentJudgeBench 基准，用于系统评估 LLM 在结构化工具调用工作流（DAG）中的判定可靠性。

**💡 创新点**

创新点在于构建多层难度、六种 DAG 拓扑的人工合成数据集，并通过配对 Ground‑Truth 与无 Ground‑Truth 条件、四维度评分框架，揭示 LLM 判定器的性能瓶颈。

**🔧 技术方法**

使用程序化参考评判器、五种生成器（3B–70B 开源模型与 GPT‑5.4）、六种 LLM 判定器，并通过对齐度、Cohen κ、bootstrap CI 等统计方法进行分析。

**📊 数据集**

使用 3,808 条人工合成记录，覆盖 15 个企业域、六种 DAG 拓扑、三种难度级别，数据集已公开。

**📈 对比分析**

与程序化参考及人类验证对比显示：给定 Ground‑Truth 时判定器对齐度可达约90%；缺失 Ground‑Truth 时仅达 77–82%，最优判定器 QwQ‑32B 在有 Ground‑Truth 场景下表现最好，GPT‑OSS‑120B 在无 Ground‑Truth 场景下最符合人类评判。

**⚠️ 局限性**

主要局限包括合成数据可能缺乏真实业务漂移、程序化评分与人类评判存在偏差、GPT‑5.4 结果不可复现、评估仅覆盖判定时刻未检验训练信号等。

---

## 367. Agentic AI for operating scientific instruments for nanoscale characterization

**arXiv ID:** 2608.26198 | [PDF](https://arxiv.org/pdf/2608.26198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 368. Contact-Aided Factor-Graph Localization for Underwater Sampling

**arXiv ID:** 2608.26932 | [PDF](https://arxiv.org/pdf/2608.26932v1)

**作者:** Michele Grimaldi `[一作]` (Heriot-Watt University), Tomoya Inoue `[通讯]` (Japan Agency for Marine-Earth Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对低空、单目下视相机在平坦、纹理稀缺的海底采样环境中存在的漂移与观测退化问题，提出一种基于因子图的定位框架，将吸盘抓取时的物理接触建模为高置信度几何约束，实现隐式闭环校正，并将惯性、DVL、压力、磁力计、视觉里程计、YOLOv8目标检测等多传感器信息紧耦合，在线初始化并实时更新轨迹。

**💡 创新点**

创新点包括：① 将物理接触事件作为因子图中的高置信度约束，自动产生隐式闭环；② 对单目视觉里程计进行退化感知下的自适应不确定度缩放；③ 采用混合滑动窗口局部束调整，兼顾实时性与局部一致性；④ 完全在线初始化与增量优化，无需预先构建地图。

**🔧 技术方法**

使用的技术包括：GTSAM 与 iSAM2 进行因子图优化；IMU 前积分与 DVL、压力、磁力计融合；下视单目视觉里程计（ORB/BRISK+Essential Matrix + 传感器尺度恢复）；YOLOv8 对星鱼等目标检测并通过高斯偏差产生视角-距离因子；接触约束通过末端执行器前向运动学与相机外参生成点约束；滑动窗口局部束调整与全局因子图同步。

**📊 数据集**

实验数据集与环境：Stonefish 仿真（提供精确轨迹）；两套实验水槽（低光与自然光）；港口真实部署（约 10m × 10m，含浊度与光照变化）；YOLOv8 训练使用 Kaggle 公共星鱼数据集。

**📈 对比分析**

对比方法包括：传统滤波式导航（DVL+IMU+压力+磁力计）、无接触的因子图定位、经典单目视觉 SLAM。实验结果表明：① 在平面旋转任务中，加入深度变化与星鱼目标可将 ATE 从 0.29 m 降至 0.17 m；② 采用接触约束时，回访误差从 0.58 m 降至 0.45 m；③ 在港口部署中，加入接触约束后回访误差从 1.23 m 降至 0.90 m，显示显著的漂移抑制与全局一致性提升。

**⚠️ 局限性**

局限性包括：① 依赖可接触目标和吸盘操作，若环境缺乏可抓取物体则失效；② 仅在平坦、纹理稀缺的环境下证明有效，对复杂地形的适用性待验证；③ 视觉里程计仍易受光照、浊度等影响，接触约束只能在交互点提供闭环；④ 需要对末端执行器与相机外参进行精确标定，若误差较大会影响定位精度。

---

## 369. Beyond Execution: Auditing Experimental Fidelity in LLM-Driven Scientific Research

**arXiv ID:** 2608.26753 | [PDF](https://arxiv.org/pdf/2608.26753v1)

**作者:** Lezhi Yu `[一作]` (Zhejiang University), Aimin Pan `[通讯]` (Zhejiang Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于参考合同的自动科学实验审计框架Auto Repro，用于在有限资源下保证LLM代理实现论文方法的完整性并验证实验结果

**💡 创新点**

创新点在于将科学重现定义为受限资源下的约束满足问题，提出8步工作流、三重验证（量化、语义、结构）以及方法学幻觉5类分类，并实现可执行的YAML合同与自动化审计

**🔧 技术方法**

使用结构化YAML合同、AST分析、语义嵌入一致性检查、指标容差阈值、三层验证管道（V_quant、V_qual、V_struct）以及自动恢复机制

**📊 数据集**

在30个经典机器学习、计算物理和生物信息学实验（如U‑Net、ResNet、SimCLR等）上进行重现；还在23个NatureBench开放式发现任务上测试

**📈 对比分析**

与Raw LLM、ARC、Claw‑AI‑Lab、Claude Code CLI等基线相比，Auto Repro获得58.8的加权综合得分，93%稳健执行率，在NatureBench 5/23任务上匹配或超越SOTA

**⚠️ 局限性**

局限性包括高完成度难题（仅46%），对LLM生成的深度学术叙述支持不足，方法学幻觉多发生在资源受限时（53%为不完整执行），需要多智能体资源调度与更完善的抽象与验证技术

---

## 370. Safety by Design: Realized-Cost Constraints for Contextual Bandits with Continuous Actions

**arXiv ID:** 2608.26755 | [PDF](https://arxiv.org/pdf/2608.26755v1)

**作者:** Spyros Dragazis `[一作]` (Boston University), Aldo Pacchiano `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种高概率安全约束的上下文多臂赌博机算法（High‑Probability Constrained UCB），在每个时间步控制实现成本不超过阈值并实现最小化累计收益损失。

**💡 创新点**

创新点在于：①在异方差（heteroscedastic）环境下，对实现成本而非期望成本做高概率约束；②采用乐观-悲观策略，在奖励上探索的同时对可行动作集进行保守估计；③提供线性和非线性模型的理论收益下界（基于 eluder 维数）。

**🔧 技术方法**

使用自适应UCB框架、子高斯噪声自归纳不等式、估计椭圆、g(α) 的单调性和逆函数构造安全集，并在非线性场景下利用 eluder 维数与经验范数构造置信集合。

**📊 数据集**

数据集：①合成实验（随机正态特征、θ*、μ*）；②真实世界 Li‑ion 电池老化数据（使用 Gemma 语言模型生成 2048 维嵌入作为上下文），对电压变化作为奖励、温度作为成本。

**📈 对比分析**

与期望成本约束基线（On‑Expectation）和 ε‑Greedy 进行对比。高概率约束算法的违规率几乎为 0.02% 而基线为 12–23%；在奖励和成本下，累计收益与 ε‑Greedy 相当或略优，尤其在高维（2048 维）时显著提升。

**⚠️ 局限性**

局限性：①仅考虑了动作尺度导致的异方差模型，未涵盖更一般的异方差形式；②实验主要集中在电池充放电数据，缺乏真实临床剂量选择等安全关键场景；③对 g(·) 的形式假设已知，实际应用中需进一步估计；④对非线性模型的理论依赖 eluder 维数，实际实现需更高维函数类。

---

## 371. SpeechGym: An Audio-Native Gym for Training Voice Agents via Reinforcement Learning

**arXiv ID:** 2608.26432 | [PDF](https://arxiv.org/pdf/2608.26432v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana-Champaign), Roger Ren `[通讯]` (Amazon AGI Foundations)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SpeechGym，一个音频原生、可训练的语音代理环境，并用它训练了基于 omni‑modal Thinker–Talker 模型的语音代理，提升了任务完成率。

**💡 创新点**

首次提供完整音频循环的可训练代理环境；诊断并量化文本到音频的误差；设计 per‑turn 过程奖励与 GRPO 训练方案；跨管线转移验证提升性能。

**🔧 技术方法**

omni‑modal Thinker–Talker (Qwen3‑Omni‑30B-A3B)，GRPO 强化学习，per‑turn 过程奖励，vLLM‑Omni 服务器，本地 LoRA 微调，语音生成与工具调用的融合。

**📊 数据集**

使用已有文本代理基准任务（Airline、Retail、Telecom）和新增 Banking，配合合成关系型数据库与冻结的用户模拟器；无真实个人信息。

**📈 对比分析**

在原始基准的 pass@1 评估中，将训练后模型在独立的 VoiceAgentBench pipeline 上直接推理，pass@1 从 24% 提升至 53%，所有域均提升，且对话轮次与 token 数量下降，表现显著优于基准。

**⚠️ 局限性**

仅在合成环境下验证，缺乏真实用户交互；仍依赖特定的 omni‑modal 模型；训练仍需大规模本地算力；未解决内容过滤、用户隐私、错误升级等部署安全问题。

---

## 372. Letters hide the truth from our eyes: English homophones have meaningfully different phonetic realizations

**arXiv ID:** 2608.26749 | [PDF](https://arxiv.org/pdf/2608.26749v1)

**作者:** Yu-Hsiang Tseng `[一作]` (University of Tübingen), R. Harald Baayen `[通讯]` (University of Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国电视新闻中的14,000个同音异义词词素进行时间归一化声谱分析和语义嵌入比较，研究其音韵差异与语义关联。

**💡 创新点**

证明同音异义词的语音实现与语境语义细粒度对应，并且能通过线性映射从上下文嵌入预测声谱，打破传统认为同音词音素完全相同的假设。

**🔧 技术方法**

时间归一化Mel声谱、GPT‑2上下文嵌入、线性判别学习(LDL)、广义加性模型(GAM)、PLS降维、t‑SNE可视化等技术。

**📊 数据集**

Redhen 2016美国电视新闻广播语料库，约35对同音异义词、每对200个token，共14,000个词素。

**📈 对比分析**

通过LDA和GAM对声谱分类、R²和AIC比较；LDA准确率达79%区分同音对、51%区分单词；GAM显著降低AIC≈3000，证明声谱差异显著；即使去除时长信息准确率仍>60%。

**⚠️ 局限性**

样本量有限、无法区分词形与词类效应、未考虑说话者情绪/语调等非语义因素、线性映射可能掩盖非线性细节。

---

## 373. Methodological and Conceptual Framework for 5D Multi-Table Analysis: A Unified Approach for Complex Data Reuse

**arXiv ID:** 2608.26149 | [PDF](https://arxiv.org/pdf/2608.26149v1)

**作者:** Edouard Lansiaux `[一作]` (CHU de Lille), Emmanuel Chazard `[通讯]` (Lille University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种统一的多表学习框架Relational Hypergraph Transformer（RHT），通过超图表示、PentE五维嵌入和稀疏关系注意机制实现对大规模、高基数、时序和多表关系数据的高效处理。

**💡 创新点**

创新点包括：①将多表关系建模为超图并实现可微分压缩；②设计五维嵌入PentE，融合语义、关系、时序、高基数和体量信息；③提出稀疏关系注意力，将复杂度从O(n²)降低到O(n·k)；④整合对比学习、动态图重连和关系因果推断等模块，实现从监督到无监督结构发现和因果推理的全流程。

**🔧 技术方法**

核心技术包括超图构造、时间2Vec多尺度时序嵌入、稀疏关系注意机制、层级高基数编码、动态图重连、差分关系发现、关系因果推断和联邦学习。

**📊 数据集**

实验使用Synthetic Synthea EHR数据集（约2.2万病人、1.3百万就诊、82万病情记录，313个SNOMED条件码）以及MIMIC-IV演示规模。

**📈 对比分析**

与SQL+XGBoost、GraphSAGE和TGN等基线对比，RHT在语义一致性（coherence）上显著提升（1.52±0.03），但在罕见码召回（RCR@10/50）和macro-F1方面仍落后于XGBoost；稀疏注意机制在可扩展性上表现优异，吞吐量可达约1.01×10⁵行/秒。

**⚠️ 局限性**

局限性包括：①在真实临床MIMIC-IV数据上仍缺乏完整验证；②对高基数的支持虽有改进，但整体macro-F1仍偏低；③稀疏注意的可解释性与训练稳定性待进一步研究；④实现复杂度高，易受超图构造与对比学习耦合训练策略的影响。

---

## 374. When Privacy Hurts Mergeability: Geometry-Aware Model Merging under Differential Privacy

**arXiv ID:** 2608.26655 | [PDF](https://arxiv.org/pdf/2608.26655v1)

**作者:** Jin Liu `[一作]` (Xidian University), Jianfeng Ma `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在差分隐私（DP）约束下，将独立微调得到的私有任务模型进行参数空间合并，提出了一种能够提升合并兼容性的DP-Merging框架。

**💡 创新点**

创新点在于从几何角度揭示DP导致的合并障碍：局部锐度（local sharpness）和参考漂移（reference drift）；并通过两项技术（DP兼容的尖锐度感知微调和基于预训练初始化的参考对齐正则化）显著降低这两种障碍，从而提升合并后模型的性能。

**🔧 技术方法**

核心技术包括：①基于梯度裁剪与加噪的DP训练；②尖锐度感知优化（类似SAM）以寻找更平坦的局部解；③在更新中加入与公共预训练权重的L2对齐项；④使用欧氏距离等指标评估合并间隙；⑤理论上给出合并间隙的上界。

**📊 数据集**

实验数据集：视觉八个图像分类任务（SUN397、Cars、RESISC45、EuroSAT、SVHN、GTSRB、MNIST、DTD）使用CLIP ViT-B/32、ViT-B/16、ViT-L/14；语言任务采用GLUE基准（CoLA、MNLI、MRPC、QNLI、QQP、RTE、SST-2、STSB）使用RoBERTa-Base/Large。

**📈 对比分析**

与多种无数据合并方法（Weight Averaging、RegMean、Task Arithmetic、TIES-Merging、PCB Merging、WUDI-Merging）在相同DP预算下进行对比。DP-Merging在所有合并方法上都实现了显著提升，平均提升幅度约为3–6%（例如ViT-B/32的平均精度从45%提升至约50%）。在不同隐私预算下，DP-Merging的优势更为突出。

**⚠️ 局限性**

局限性：①需要对尖锐度半径和对齐强度等超参数进行调优；②方法仍受DP预算对模型精度的整体限制；③仅在同一基础模型上验证，跨模型迁移或更大规模任务的通用性待进一步评估；④未提出新的合并规则，仅改进微调阶段，合并策略本身仍可能是瓶颈。

---

## 375. LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Efficient Multimodal Fine-Tuning

**arXiv ID:** 2608.26820 | [PDF](https://arxiv.org/pdf/2608.26820v1)

**作者:** Muyao Yuan `[一作]` (Xi'an Jiaotong University), Haipeng Du `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种信息论蒸馏框架LLaVAFlow，用来在视觉指令微调时保持多模态大模型的跨模态对齐；

**💡 创新点**

创新点包括：①将跨模态对齐隐式捕获为“对齐流”，并用信息瓶颈压缩其冗余；②设计轻量级Alignment Flow Module (AFM)并通过互信息最大化实现预训练与微调模型的对齐流迁移；

**🔧 技术方法**

使用矩阵式Rényi熵、互信息、信息瓶颈、AFM、Logit Lens等技术；

**📊 数据集**

在VQA（ScienceQA、OKVQA、OCRVQA、GQA、TextVQA）和Captioning（COCO-Caption）数据集上进行实验；

**📈 对比分析**

与LoRA、DoRA、LLaVAMod、LLaVAKD、DARE、Model Tailor、LoRASculpt等多种现有方法对比，LLaVAFlow在目标任务准确率提升显著，同时显著降低了对原始知识的遗忘（源任务性能提升）；

**⚠️ 局限性**

局限性包括：对矩阵互信息估计的计算开销、对小型参数化微调（如LoRA rank）的依赖、对不同模态或更大模型的泛化仍待进一步验证。

---

## 376. Assessing mentalization in humans and large language models

**arXiv ID:** 2608.26291 | [PDF](https://arxiv.org/pdf/2608.26291v1)

**作者:** Aamir Sohail `[一作]` (University of Birmingham), Lei Zhang `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过将两种传统心理学中的经济学游戏（检视游戏与石头剪刀布）转换为文本提示，分别让人类参与者和多种大型语言模型（LLM）完成任务，并结合认知计算建模（如影响学习、CHASE等）对其在递归与适应性心理化方面的表现进行定量评估。

**💡 创新点**

创新点在于：①首次将经济学游戏与层次化的心理化模型相结合，用正式的计算框架比较人类与多款LLM的递归与适应性思维；②提出并验证了“Social Chain-of-Thought”（SCoT）提示策略在提升LLM策略深度和游戏表现方面的有效性；③通过模型大小、供应商差异和提示方式的交互作用，揭示了LLM在心理化方面的可变性。

**🔧 技术方法**

技术方法包括：构建文本化实验任务、API调用与统一的JSON交互、SCoT提示设计、层次化贝叶斯建模与留一交叉验证（LOO-CV）、线性混合效应模型、参数回归及后验预测检验。

**📊 数据集**

数据集包括：人类实验数据（检视游戏n=67，石头剪刀布n=172）；LLM实验数据（检视游戏n≥49，各模型组合共计2,099实例；石头剪刀布n≥188，各模型组合共计≈1,200实例）。

**📈 对比分析**

比较方法：使用t检验、方差分析、线性混合模型以及贝叶斯模型选择（PXP）评估人类与LLM在得分、策略深度（η、κ）及预测准确性上的差异。结果显示：①大多数LLM在检视游戏中得分低于人类，GPT‑5和SCoT提示下表现与人类相当或更好；②SCoT提示在检视游戏中显著提升所有模型的得分；③在石头剪刀布中，GPT‑5在面对更高递归级别对手时表现提升，而DeepSeek对更复杂对手适应不足。

**⚠️ 局限性**

局限性包括：①仅使用闭源LLM且未覆盖开放源模型；②LLM对实验提示和上下文窗口的依赖导致结果受限；③模型内部机制不透明，无法直接解释心理化参数的神经/机制对应；④人类样本量有限且在不同实验条件下对手经验不一致；⑤实验仅限两种游戏，难以全面覆盖社会认知功能。

---

## 377. Toward Equitable Low-Carbon Mobility: Fairness-Aware Demand Prediction for Expanding Bike-Sharing Systems

**arXiv ID:** 2608.26451 | [PDF](https://arxiv.org/pdf/2608.26451v1)

**作者:** Man Luo `[一作]` (University of Exeter), Yixuan Zhao `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 FairGIN，一种公平感知的动态图神经网络，用于在自行车共享系统扩张过程中预测新站点的需求，并辅助公平部署决策。

**💡 创新点**

创新点包括：① 扩张模拟增量训练（ESIT）让模型提前体验冷启动场景；② 基于注意力与可学习温度的知识迁移与正交嵌入对齐，解决特征异质性导致的偏差；③ 收入分层公平正则化与部署评分机制，兼顾预测精度与公平性。

**🔧 技术方法**

技术手段：多层GCN图卷积、GRU时序编码、注意力加温度机制、Cayley正交变换、三项联合损失（观测、模拟、公平）以及可解释的部署评分。

**📊 数据集**

使用数据集：NYC Citi Bike（2018–2023，746+312站）和Seattle Bikeshare（2017–2018，186+61站），并结合POI、道路网络、天气、出租车流量及ACS收入等多源空间特征。

**📈 对比分析**

与 ARIMA、LSTM、STGCN、DCRNN、FairST、GraphSAGE、DA-MRGNN、KITS、UrbanGPT 等九种基线对比；在 MAE/RMSE 上取得最低误差，RFG/IFG 与 Spearman |ρ| 亦显著下降（精度提升约15–20%，公平性降低约30%以上）。

**⚠️ 局限性**

局限性：仅按收入中位数划分二分组，忽略种族、车拥有率等多重受保护属性；依赖高质量 ACS 与 POI 数据，难以推广到数据稀缺或无托盘城市；需要频繁更新时序特征以维持长期准确性。

---

## 378. MeshReduce-U: Compiler-Guided Communication Reduction for Irregular Neural Reductions on Mesh NoCs

**arXiv ID:** 2608.26220 | [PDF](https://arxiv.org/pdf/2608.26220v1)

**作者:** Amirreza Khorasanian `[一作]` `[通讯]` (University of Tehran), Amirreza Khorasanian (University of Tehran)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

MeshReduce-U在网格NoC上通过编译器级别的通信重写来减少多对一聚合的传输。

**💡 创新点**

创新点是将可合法的局部聚合、岛屿合并、特征阻塞和容量感知的源放置与融合路由结合成一个完整的“先化简再路由”框架。

**🔧 技术方法**

采用堆排序与流网络求解最小半径放置、最小费用最大流、Dijkstra模板路由以及贪心+局部修复的算法组合。

**📊 数据集**

使用40个合成的非规则聚合、20个可降解神经网络工作负载（GCN、GraphSAGE、RNN等）以及30个匹配的4×4网格测试实例。

**📈 对比分析**

与基线（ABC收集、I‑GCN岛屿化、Steiner树、XY路由等）对比，MeshReduce-U在平均重放延迟、总链路使用和融合链路使用上分别降低约12%、20%和49%，在所有工作负载上都优于其它方法。

**⚠️ 局限性**

限制是模型只考虑单一宽度的载荷、无缓冲/虚拟通道、无RTL能耗测量，且对非可聚合的运算或缺乏中间合并的硬件会退化到原始方案。

---

## 379. Predicting Quantifiability from Primary Screens to Prioritize Dose-Response Profiling

**arXiv ID:** 2608.26538 | [PDF](https://arxiv.org/pdf/2608.26538v1)

**作者:** Sean Lim `[一作]` `[通讯]` (Rice University), Sean Lim (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于初始三点筛查数据预测后续浓度-响应实验是否能给出可报告的效价（quantifiability）的方法，重新定义筛查优先级，区别于传统的活性筛查；

**💡 创新点**

创新点在于将量化可测性（quantifiability）作为独立的 triage 目标，利用三点筛查的响应特征而非单纯化学结构，成功预测大约44.5% 的可报告效价，并显著降低后续实验负担；

**🔧 技术方法**

主要技术包括特征工程（从三点筛查提取16个响应统计量、Morgan 指纹、历史活性统计和实验元数据），随机森林与逻辑回归建模，交叉验证与时间序列滚动评估，外部验证（ToxCast/Tox21单浓度筛查）等；

**📊 数据集**

使用 EvE Bio 公开的 1–11 期三点筛查与 11 点浓度-响应数据共 32,971 对，以及 EPA ToxCast/Tox21 的单浓度筛查数据进行外部验证；

**📈 对比分析**

与传统按筛查最大活性排序相比，随机森林模型在 AUROC 0.939、AUPRC 0.927，回收率 Recall@50% 0.894、Yield@25% 0.961；在滚动评估中可将 90% 的可报告效价在 55.4% 的实验预算内获取，显著低于仅按活性筛查的需求；

**⚠️ 局限性**

局限性包括：模型仅在已被晋升的样本上训练与评估，无法直接推广至未晋升的化合物-检测对；受实验室特定的浓度范围、曲线拟合规则影响；需本地校准；以及未能体现对化学多样性与生物靶点多样性的潜在影响。

---

## 380. MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models

**arXiv ID:** 2608.26295 | [PDF](https://arxiv.org/pdf/2608.26295v1)

**作者:** Arseniy Varlamov `[一作]`, Ilseyar Alimova `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个可控的工具‑记忆冲突基准，用于衡量LLM在工具返回与模型记忆冲突时的仲裁行为，并探讨提示与微调方法的影响。

**💡 创新点**

首次通过对工具返回与模型记忆的正确性进行独立控制，定义了基于正确性条件的仲裁评估框架，并生成了包含可执行工具调用与人工验证答案的高质量数据集。

**🔧 技术方法**

结合工具调用、LoRA微调、DPO偏好优化以及交叉折交叉验证，对多种开源LLM进行评估，使用精确答案匹配与自定义评分器。

**📊 数据集**

基于ToolHop链条的子问题，经过去重、事实过滤、人工审核与干扰器构造，得到542个高质量事实问题，随后生成6,504个评估案例。

**📈 对比分析**

通过比较未微调与SFT/DPO微调模型在正确答案保留率、正确工具跟随率、误工具跟随率等指标上的差异，发现微调能提升保留率但往往降低工具跟随，且效果高度依赖模型；提示方法虽提高保留率但同样牺牲工具跟随。

**⚠️ 局限性**

仅评估7–9B开源模型，数据集只来自单一来源，干扰器构造偏向人工或自动化生成，缺乏对更大规模模型、其他工具协议或更自然的错误场景的验证。

---

## 381. RegulAR: Graph-Grounded Error Recognition and Assistance for Procedural Tasks in AR

**arXiv ID:** 2608.26715 | [PDF](https://arxiv.org/pdf/2608.26715v1)

**作者:** Yi-Lin Ye `[一作]` (Hong Kong University of Science and Technology), Wong Kam-Kwai `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 RegulAR，一种基于层次化依赖图和多模态大模型的 AR 助手，能够在实时执行过程中检测错误、评估其对后续步骤的影响，并提供分层恢复指导。

**💡 创新点**

创新点在于将任务说明转换为显式的层次化依赖图，将 MLLM 的 egocentric 视觉推理与图结构耦合，实现主动错误识别、影响评估和基于错误类型的定制干预；同时通过 HUD 视觉化任务状态与错误信息，支持用户在执行中快速定位与恢复。

**🔧 技术方法**

技术包括：GPT‑5 用于将文本指令解析为依赖图；Gemini 多模态模型进行批量视频推理；基于图的状态追踪与错误分类；影响评估公式结合拓扑重要性与风险权重；在 Unity/Meta Quest 3 上的实时 AR 显示与语音交互。

**📊 数据集**

使用了 10 篇包含烹饪、手工、实验等多种场景的公开流程文档（共 118 步、176 个动作），以及在两项任务（微波炒蛋与蜡烛制作）中收集的 egocentric 视频与交互日志。

**📈 对比分析**

通过 12 名参与者的配对用户研究和日志评估，RegulAR 在任务结构理解、进度意识、错误识别与定位、影响评估以及用户对恢复支持的感知上均显著优于仅基于 MLLM 的基线；在技术评估中，动作识别 F1 分数提升约 20%~30%，误报率下降；任务完成时间虽略有缩短，但差异未必统计显著。

**⚠️ 局限性**

局限性包括：5 秒决策周期可能导致短时动作漏检与响应延迟；依赖网络与大模型，存在延迟与幻觉；图视图在复杂流程下可能导致视觉拥挤；未充分验证在长期、协作或高风险场景下的效果；缺乏对不同用户偏好与专长的自适应干预策略。

---

## 382. Adversarial Training Without Input Gradients via Low-Rank Householder Expansions

**arXiv ID:** 2608.26963 | [PDF](https://arxiv.org/pdf/2608.26963v1)

**作者:** Tiana C. Johnson `[一作]` (Washington University in St. Louis), Donsub Rim `[通讯]` (Washington University in St. Louis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于低秩Householder展开（LRHE）的对抗训练方法，该方法利用前向传播中的激活模式直接得到网络在小范数扰动下最敏感的子空间，采用正则化方式替代传统的PGD内循环，达到对抗鲁棒性的提升。

**💡 创新点**

创新点在于：①无需对输入求梯度即可获得局部几何信息，消除了PGD等内循环的梯度开销；②通过LRHE提取的低秩子空间作为正则化方向，显著降低了训练成本；③实现了与多步PGD训练相当的鲁棒性，仅在小扰动范围内达到高效防御。

**🔧 技术方法**

使用了LRHE（低秩Householder展开）、对抗训练正则化（基于LRHE子空间的正则项）、投影到LRHE子空间的补偿、卷积网络中Max‑Pooling的重写为ReLU+线性组合、AutoAttack和PGD等攻击方法进行评估。

**📊 数据集**

实验仅在MNIST手写数字数据集上进行，采用两层卷积+两层全连接的CNN架构。

**📈 对比分析**

与标准训练、3步PGD对抗训练以及40步PGD对抗训练进行对比。在相对ℓ²扰动≤0.02时，LRHE训练的对抗准确率与3步PGD相当；在更大扰动或ℓ∞扰动下，40步PGD更具优势。训练成本约等价于2.8步PGD，较3步PGD低约8.7倍，显著节省计算资源。

**⚠️ 局限性**

局限性包括：①仅在小扰动预算下有效，超过一定范围子空间失效；②实验仅在MNIST上验证，尚未证明能推广到更高分辨率或复杂数据集；③LRHE方向与标签无关，可能与决策边界不完全对齐；④需要对网络中的Max‑Pooling进行重写，增加实现复杂度。

---

## 383. Size Bounds for CQs Under Acyclic Constraints

**arXiv ID:** 2608.26775 | [PDF](https://arxiv.org/pdf/2608.26775v1)

**作者:** Stefan Mengel `[一作]` (University of Artois), Andrei Romashchenko `[通讯]` (LIRMM University of Montpellier)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在具有无环函数依赖下，投影出现时共识（Entropic bound）与可计算的Polymatroid bound之间的大小上限关系，并证明即使函数依赖无环，加入投影后两者仍可能存在多项式差距；同时给出在输出变量是依赖图前缀时两者保持一致的正面结果。

**💡 创新点**

首次揭示投影操作在无环函数依赖约束下会破坏Entropic与Polymatroid bound的一致性，提供了一个明确的多项式分离例子；并提出了“输出变量前缀”这一新的足够条件，使两者仍保持紧致。

**🔧 技术方法**

使用信息理论中的熵与多项式约束，构造线性规划求解Polymatroid bound；利用“untangling”算法将有环函数依赖转化为无环并引入投影；借助Zhang–Yeung不等式证明Entropic bound；对两种上界进行比较与推导。

**📊 数据集**

本文主要为理论研究，不涉及具体实验数据集；通过构造性的例子（Zhang–Yeung查询）演示性质。

**📈 对比分析**

通过对比两种上界的线性规划求解结果，证明存在多项式差距；在前缀条件下，两种上界相等，表明在此类查询中Polymatroid bound是紧致的。

**⚠️ 局限性**

仍未解决Entropic bound的可计算性问题；所示的多项式分离只在特定构造下出现，尚未评估在更一般查询中的普遍性；投影导致的算法最优性问题仍未解决。

---

## 384. STILL: Recovering Lowered STL Semantics for LLM-assisted C++ Decompilation

**arXiv ID:** 2608.26408 | [PDF](https://arxiv.org/pdf/2608.26408v1)

**作者:** Xiaohan Wang `[一作]` (Vanderbilt University), Kevin Leach `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出STILL框架，在已剥离的C++二进制中预测并恢复函数层面的STL容器语义，随后将这些语义作为紧凑提示提供给LLM进行代码重构，提升可执行率和可读性。

**💡 创新点**

创新点在于：①将STL语义残留转化为可供LLM使用的结构化提示接口；②构建针对函数层面STL容器的监督数据集StlBench；③证明通过LLM提示可以显著提升对STL函数的可执行率，并且不同后端对提示的可利用性差异。

**🔧 技术方法**

技术包括：基于Angr的控制流图提取与Typed RGCN语义预测；对残留特征进行ContRes和TreeRes编码；将预测结果渲染为JSON提示；在LLM（DeepSeek、LLM4Decompile-Ref）中进行提示式或LoRA适配微调。

**📊 数据集**

使用了StlBench（来自CodeContests的约3.9K C++函数，按优化级生成14.8K CFG）以及HumanEval-C++（含STL和非STL函数）作为零样本测试集。

**📈 对比分析**

在代码层面与执行层面进行对比。STILL在StlBench上宏F1 80.4%/门控F1 95.4%；在HumanEval-C++上无监督迁移宏F1 89.0%。在LLM细化中，提示提升DeepSeek的可执行率从17.4%升至28.4%，对STL函数提升至18.6%。对LLM4Decompile-Ref，LoRA适配后可执行率从11.8%提升至30.8%。

**⚠️ 局限性**

局限性包括：仅恢复五种STL容器的函数层面标签，未覆盖变量级、嵌套、迭代器等细粒度信息；依赖x86-64/libstdc++、Angr和Ghidra的工具链，可能难以直接迁移至其他架构或编译器；可执行率评估基于有限的单元测试，未能完全保证语义等价。

---

## 385. Attention-Guided Reliability Scaling for Contrastive Decoding in Robust Audio-Visual Speech Recognition

**arXiv ID:** 2608.26213 | [PDF](https://arxiv.org/pdf/2608.26213v1)

**作者:** YoungChae Kim `[一作]` (Hanyang University), Joon-Hyuk Chang `[通讯]` (Hanyang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于注意力与预测差异的可靠性感知对比解码（Contrastive Decoding）方法，用于在大语言模型驱动的音视频语音识别（AVSR）系统中动态调节对比强度，从而在噪声环境下提升识别鲁棒性并保持清晰语音条件下的准确率。

**💡 创新点**

创新点在于：① 通过音频能量、熵以及音频-视觉与仅音频预测的Jensen–Shannon散度三种可靠性信号，构造乘法门控实现对比强度的自适应缩放；② 将对比解码作为训练无关的推理时策略，避免了模型微调与结构改动；③ 通过多头注意力权重提取实现可靠性估计，确保门控仅在三信号共同提示不可靠时才激活。

**🔧 技术方法**

使用的技术包括：大语言模型基础 AVSR 架构、Whisper 与 AV‑HuBERT 编码器、对比解码（Contrastive Decoding）、多头注意力权重提取、Jensen–Shannon 散度计算、信号门控（Sigmoid 与高斯滤波）以及多信号乘法融合。

**📊 数据集**

实验数据集：LRS3（433 小时训练，1,327 句子测试）以及 LRS2（OOD 评估），通过 MUSAN 生成 0 dB、-5 dB、-10 dB、-15 dB 四级噪声环境，使用 WER 作为评估指标。

**📈 对比分析**

与基线 AVSR、固定 λ 对比解码及各门控单独使用的 ablation 进行对比。结果显示，在 LRS3 上平均提升约 9.9% WER，在 LRS2 上平均提升约 8.9%，同时保持在清晰条件下的性能提升，且推理时延增加约 8.6%（约 136 ms）。

**⚠️ 局限性**

局限性包括：门控参数（β_E、β_H、μ_sweet、σ_JS）需在验证集上调优，对极低 SNR（-15 dB 以上）时 JS 散度接近 1 的情况仍可能产生秩失真；虽然仅需推理时计算，但相较于无对比解码仍有一定的计算开销；方法在不同模型规模下表现一致，但在极端噪声或极低资源场景下仍有提升空间。

---

## 386. Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling

**arXiv ID:** 2608.26800 | [PDF](https://arxiv.org/pdf/2608.26800v1)

**作者:** Taeyoon Lee `[一作]` (Robotics and AI Institute), Nicolas Rojas `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在真实硬件上实现并快速学习多种三球杂耍模式，仅用数分钟即可完成；

**💡 创新点**

提出正则化记忆式学习框架，利用不完美先验模型与局部经验结合实现高效在线学习，同时通过互可达集确保安全的动态行为组合；

**🔧 技术方法**

正则化记忆式学习、互可达集约束、基于任务参数化的规划与轨迹生成、低延迟视觉跟踪（EfficientTAM+FastGICP）以及球的弹道预测；

**📊 数据集**

使用真实的HB弹跳球和AthenaZero双臂机器人，无需公开数据集，全部基于现场采集的传感器数据；

**📈 对比分析**

对比无学习、单纯先验和学习后结果，学习后在不到5分钟的物理交互中即可掌握五种杂耍模式；cascade仅需53s，连续Tennis→Half‑Shower→Cascade 75s，shower与box模式在30–60s达到稳态；

**⚠️ 局限性**

局限包括：对侧抛侧捕的视觉反馈延迟导致开放式抓取误差难以完全消除；依赖初始先验模型（即便不准确也需其梯度信息）；高频碰撞与手指相互作用仍导致学习收敛慢；在更重或更复杂物体（如俱乐部）时需改进低层策略与感知。

---

## 387. Position Is All You Need: A Free Lunch Token Compression Strategy for MLLM-based Referring Expression Segmentation

**arXiv ID:** 2608.26142 | [PDF](https://arxiv.org/pdf/2608.26142v1)

**作者:** Yuhan Liu `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4575 | [OpenAlex ID](https://openalex.org/A5039670436)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了多模态大语言模型（MLLM）在参照表达分割（RES）任务中的视觉 token 压缩问题，并提出了一种基于位置的无训练压缩方法 PAYN；

**💡 创新点**

创新点在于证明位置嵌入是 RES 任务中最关键的信息，并利用仅基于位置的采样（棋盘式与最远点采样）和位置索引保留实现高效、无训练的 token 压缩；

**🔧 技术方法**

技术包括位置引导的 token 选择（checkerboard、Farthest Point Sampling）、位置索引保留策略、以及在现有 RES 基线（Text4Seg、InstructSeg）上进行评估；

**📊 数据集**

实验使用 RefCOCO、RefCOCO+、RefCOCOg 三个标准 RES 数据集；

**📈 对比分析**

与 ToMe、VisionZip、Dart、VisPruner、DivPrune、EVTP-IVS 等多种压缩方法对比，PAYN 在不同压缩比例下平均 cIoU 最高，性能保持在 95% 以上，且推理速度提升约 1.3–1.4 倍；

**⚠️ 局限性**

局限性包括在极小目标或背景信息极少的极端情况下可能失效，因仅靠位置无法保证关键 token 的保留。

---

## 388. Spec2Vision: Contract-Guided Delivery of AI-Generated Computer Vision Pipelines

**arXiv ID:** 2608.26400 | [PDF](https://arxiv.org/pdf/2608.26400v1)

**作者:** Ghfran Jabour `[一作]` (ITMO University), Sergey Ivanov `[通讯]` (ITMO University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Spec2Vision实验框架中，通过在生成、检查、修复的多阶段流程中携带任务契约，显著提高了生成的计算机视觉管道的交付成功率。

**💡 创新点**

提出了契约驱动的分阶段生成与检测机制，并在17个不同CV任务上证明了其相较单一代理模型在交付成功率上的优势。

**🔧 技术方法**

使用多阶段角色（架构师、生成器、验证器、测试器、批评者、验证）与结构修复、生成器预检、兼容性支架等技术实现任务契约的完整传递。

**📊 数据集**

构建了一个包含17个CV任务（检测、分割、分类、跟踪、姿态、文本、深度等）并统一数据布局的固定基准。

**📈 对比分析**

与单代理基线对比，Spec2Vision在850次实验中获得81/85的交付成功率，而最弱基线仅为6/85，显示了显著性能提升。

**⚠️ 局限性**

实验仅覆盖固定任务集合和单一模型，缺乏对更多任务、多模型、外部评估器的泛化验证，且结构修复与预检的效果可能受模型质量限制。

---

## 389. Fairness Invariants: A Relational Approach to Explaining and Mitigating Fairness Bugs

**arXiv ID:** 2608.26209 | [PDF](https://arxiv.org/pdf/2608.26209v1)

**作者:** Ranit Debnath Akash `[一作]` (University of Illinois at Chicago), Saeid Tizpaz-Niari `[通讯]` (University of Illinois at Chicago)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于关系不变式的框架，自动定位、解释并通过守护规则缓解数据驱动系统中的个体公平缺陷

**💡 创新点**

创新点在于把个体公平视为双向关系约束问题，用对抗式决策树学习生成可解释的公平不变式，并将其作为实时守护器实现局部补救

**🔧 技术方法**

采用对偶关系数据整理（DCNE/DCVE/DCHE）、决策树/规则学习（C4.5、CART、FIGS 等）和后处理守护规则，配合对抗样本生成器寻找个体不公平实例

**📊 数据集**

在符号化决策程序（PG1–PG6）、公开风险评估数据（COMPAS、Adult、Bank Marketing）以及 20 个深度神经网络（AC1–AC12、BM1–BM8）上进行实验

**📈 对比分析**

与基准 AFT、Themis、AEQUITAS 等方法对比，DCHE+ C4.5 在 83% 以上案例中能定位 83% 以上的公平缺陷，且守护规则能将不公平决策率下降 40%–70%，显著优于现有后处理或再训练技术

**⚠️ 局限性**

局限性包括依赖高质量的 IDI 检测和对抗样本生成、对可解释性需求较高的决策树、无法保证全局公平、对自然语言或视觉任务等非结构化领域适配度有限

---

## 390. The Time-Dependent Traveling Salesman Problem with Loose Time Windows

**arXiv ID:** 2608.26360 | [PDF](https://arxiv.org/pdf/2608.26360v1)

**作者:** Francisco J. Soulignac `[一作]` `[通讯]` (Universidad Nacional de Quilmes), Francisco J. Soulignac (Universidad Nacional de Quilmes)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种面向时间依赖旅行商问题（TDTSPTW）在松散或无时间窗口情形下的完整精确求解框架，结合动态规划标记、列生成、ng-内存增强、精确搜索以及基于完成度的稀疏化与变量固定。

**💡 创新点**

核心创新在于：将列生成、ng-内存增强与精确搜索协同迭代，使得双向完成度动态更新、变量固定与稀疏化可以相互强化，从而在松散窗口下显著扩展可求解规模；以及利用弧位置状态实现更具针对性的ng-内存增广，减少状态空间爆炸。

**🔧 技术方法**

技术方法包括：动态规划标记（前向/后向/受限）、列生成（带稀疏化与分阶段求解）、ng-内存增广（循环禁止）、精确搜索（受限完成度标记的双向求解）、变量固定与稀疏化（基于完成度的边删除），以及与分支定界结合的 branch‑and‑price。

**📊 数据集**

使用的数据集包括：1）无时间窗口实例（Snyder 等 1800 个 15–60 顾客；另 180 个 10–60 顾客）；2）松散/中等时间窗口实例（Gendreau 等 4800 个 15–40 顾客、0–1 乘数；Kool et al. 3600 个 10–60 顾客、0–1 乘数；Rendl 等 720 个 20–40 顾客、0–1 乘数）；3）紧时间窗口实例（Bertsimas 等 400 个 40–180 顾客、40–180 宽度）。

**📈 对比分析**

与前沿方法（Bertsimas‑等的列生成 + ng-内存 + 精确搜索；Rendl‑等的决策图/信息搜索）对比，所提框架在无窗口或松窗口下平均快 1–2 个数量级，能够在一小时内解决至多 45 顾客所有实例，并在 60 顾客下解决大多数实例；在紧窗口下相对慢，但仍保持可比性；总体上，证明在弱约束（松/无窗口）下，动态规划 + 列生成的组合比纯决策图/信息搜索更具优势。

**⚠️ 局限性**

局限性包括：1）在紧窗口或强约束情形下，列生成与 ng-内存的额外计算导致速度不如直接 A*/决策图方法；2）当时间窗口非常松散且客户数较大时，完成度稀疏化与变量固定仍难以有效剪枝，导致后续搜索耗时；3）框架的实现依赖多阶段参数调优，参数设置不当会影响性能；4）在完全无时间窗口的情况，尽管可扩展到 60 顾客，但仍低于纯时间独立 TSP 的最优求解器。

---

## 391. PragAlign: Evidence-Sensitive Reply Assistance Across Chinese and Japanese Appropriateness Judgments

**arXiv ID:** 2608.26700 | [PDF](https://arxiv.org/pdf/2608.26700v1)

**作者:** Xin Zhong `[一作]` (University of Tokyo), Satori Hachisuka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出PragAlign系统，在回复生成前通过两个模块：Context Reader负责结构化证据，Gap Policy决定是否进行澄清并挑选单一问题；随后用固定生成器生成最终回复。通过在中文和日语两种语言中使用匹配情境，对比Direct、Rule和PragAlign三种回复方式，评估其在适当性上的差异。

**💡 创新点**

创新点在于：①将证据敏感的澄清决策层与生成器分离，强调信息状态（可观察、推断、未知）并仅在必要时发起一次澄清；②使用匹配的多语言情境进行对照评估，揭示共享与语言特定的判断模式；③构建控制式情景生成和符号化框架，实现可审计的训练与评估。

**🔧 技术方法**

技术实现基于预训练大模型Qwen3‑8B，利用LoRA/QLoRA进行参数高效微调，分别训练Context Reader和Gap Policy；固定生成器未进行微调；评价采用Friedman检验、Wilcoxon配对、Kendall's W及bootstrap CI。

**📊 数据集**

数据集为合成情境集，共10个匹配场景（5简单+5复杂），每个场景产生三种回复（Direct、Rule、PragAlign）。根+反事实对用以标注公开/隐藏信息；人类评估使用9名中文母语者和3名日语母语者。

**📈 对比分析**

比较方法：对每种语言分别进行Friedman检验，随后Wilcoxon配对；计算每种条件的均值排名、top‑rank率和worst‑rank率，并给出bootstrap CI。中文评估中PragAlign显著优于两基线（mean rank 1.61 vs 2.27/2.16，p<.001）；日语评估无显著差异，但PragAlign在中文中top‑rank率最高（51%），worst‑rank率最低（12%）。两语共10个场景中有5个场景选择相同，表明部分共识。

**⚠️ 局限性**

限制：①样本量不平衡，日语评审仅3人，影响统计功效；②使用合成情境缺乏自然对话特性；③仅评估本地语言回复质量，未检验跨语言回复生成；④未单独评估澄清问题的质量和效果；⑤缺乏多样化的对照实验和共享语言设计，难以分离表达与文化差异。

---

## 392. Beyond Reflection: Affirmation as a Promising Behavioral Marker Associated with Quality in Text-Based Counseling

**arXiv ID:** 2608.26689 | [PDF](https://arxiv.org/pdf/2608.26689v1)

**作者:** Michimasa Inaba `[一作]` `[通讯]` (University of Electro-Communications), Michimasa Inaba (University of Electro-Communications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对日语文本辅导数据 KokoroChat 进行行为注释与质量分析，研究咨询者行为与会话质量的关联，并通过逻辑回归模型在英文情绪支持数据 ESConv 上进行跨数据集验证。

**💡 创新点**

① 提供了 6,589 次日语辅导会话的 11 类咨询策略标签和 4 级痛苦水平的自动注释；② 发现肯定（Affirmation）比传统强调的反思（Reflection）更稳健地与会话质量相关；③ 在不同语言、文化和专业水平的语料中验证了这一信号的可迁移性。

**🔧 技术方法**

使用 Gemini 2.5 Flash 进行大规模自动注释；统计方法包括卡方检验、Spearman 与部分相关、混合效应 / Mundlak 分析；跨数据集转移采用逻辑回归模型并计算 Spearman 相关；对注释错误进行层级分析。

**📊 数据集**

KokoroChat（日语 6,589 次，专业/培训咨询者角色扮演）与 ESConv（英文 约 5,000 次，非专业支持者）两套公开情绪支持对话数据集。

**📈 对比分析**

在 KokoroChat 上用会话评分的高/低三分位与痛苦变化 Δdistress 进行分层与相关分析，证明肯定的使用率与质量正相关；随后训练仅用五个行为特征的逻辑回归模型，对 ESConv 的预测概率与 Δemotion 计算 Spearman ρ=-0.072（p=0.014），表明肯定信号在跨数据集上仍保持一定正向效应，尽管效应较小。

**⚠️ 局限性**

观察性设计导致无法确定因果；KokoroChat 评分与特征存在潜在循环；LLM 自动注释虽达成较高一致性但仍有噪声；模型仅关注行为频率，未捕捉时间序列与上下文细节；跨数据集差异（语言、文化、专业）限制了外部效度；整体相关系数偏小，说明单纯频率解释会话质量的解释力有限。

---

## 393. VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill in Long-Context LLM Inference

**arXiv ID:** 2608.26523 | [PDF](https://arxiv.org/pdf/2608.26523v1)

**作者:** Yan Shi `[一作]` (Huawei Technologies Co., Ltd.), Liangjun Feng `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了虚拟管道并行技术 VPP，用来改进长上下文 LLM 的分块填充推理，减少管道泡沫并提升吞吐量。

**💡 创新点**

创新点在于固定分块大小、通过 V‑形虚拟阶段重新排列层级以匹配线性增长的计算延迟，并结合异步通信重排和跨请求流水线打包实现计算与通信的重叠和空闲窗口的压缩。

**🔧 技术方法**

使用了 vLLM‑Ascend 框架、Ascend 910C NPU、HCCL 通信库，并实现了虚拟阶段调度、异步通信重排和跨请求打包等技术。

**📊 数据集**

评估数据集为 MoE 基础 LLM：Qwen3、DeepSeek‑V3.1、GLM‑5.2，序列长度覆盖 4K–1M tokens，使用 AISBench 生成不同长度请求混合。

**📈 对比分析**

通过与原始 Chunked Prefill（CPP）和动态分块（DCPP）对比，VPP 在 1M 令牌长序列上相较 DCPP 提升吞吐量 13.1%（长序列）或 6.7%（混合工作负载），TTFT 改进显著，管道泡沫比例从 6.4% 降至 0.1%。

**⚠️ 局限性**

局限性在于对分块延迟近似线性增长的假设依赖较强，稀疏注意力模型（如 GLM‑5.2 的 DSA）会降低此规律性，导致 VPP 的优势减弱；同时引入额外通信开销，需在更大规模或更稀疏注意力场景下进一步优化。

---

## 394. Structured Evidence Routing for Incident Risk Prediction from Multimodal Longitudinal EHRs

**arXiv ID:** 2608.26191 | [PDF](https://arxiv.org/pdf/2608.26191v1)

**作者:** Animesh Agarwal `[一作]` (Optum AI, UnitedHealth Group), Carlos Morato `[通讯]` (Optum AI, UnitedHealth Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种结构化证据路由（router–predictor–reviewer）流程，先对完整的多模态长期电子病历进行摘要与目标证据切片，再用任务导向的LLM进行风险评估并由审阅者迭代校正；

**💡 创新点**

创新点在于将完整病历访问与疾病特异评估分离，利用路由器生成可追溯的证据链，保持预测可解释性；

**🔧 技术方法**

主要技术包括多模态信息抽取、基于提示的LLM路由与预测、PubMedBERT嵌入、XGBoost监督读取器；

**📊 数据集**

使用公开的EHRSHOT 1年事件诊断基准（高血压、高脂血症、红斑狼疮、急性心肌梗死、胰腺癌）和完整结构化电子病历；

**📈 对比分析**

与计数型模型、CLMBR、MoA+Cls等基线进行AUROC/AUPRC比较，路由证据摘要在AUROC上与基线持平或优于CLMBR，并在AUPRC上表现相当或更好；

**⚠️ 局限性**

局限性包括仅基于回顾性基准标签评估，未验证临床可用性、医生信任度或工作流程改进，且对模型可解释性验证仍不足。

---

## 395. FRESCO: Complete and Scalable Temporal Safety for CHERI Application Processors

**arXiv ID:** 2608.26353 | [PDF](https://arxiv.org/pdf/2608.26353v1)

**作者:** Merve Gülmez `[一作]` (Ericsson), Thomas Nyman `[通讯]` (Ericsson)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为 CHERI 处理器设计并实现了一套完整的时间安全机制，既覆盖堆内存又覆盖栈帧，通过色彩化能力（colored capabilities）实现对返回后栈对象的彻底失效。

**💡 创新点**

创新点包括：① 利用栈指针的颜色来标识栈帧生命周期，确保任何从该帧派生的能力在函数返回时立即失效；② 引入逃逸分析与 stale‑read 分析，动态决定哪些函数需要上色、哪些栈变量需要清零，从而显著降低色彩压力；③ 在内存层面使用颜色段（color‑segmented allocator）将颜色空间按段划分，去除全局颜色上限，实现堆与栈共用颜色空间；④ 在硬件层面通过 per‑segment PVT、pvb‑check、ccsettype 等指令支持这些功能，并在 CHERI‑RISC‑V 软核中实现。

**🔧 技术方法**

技术实现涵盖：CHERI‑RISC‑V ISA 扩展（pvtr、ccsettype、pvb‑check 等）、LLVM/Clang 逃逸与 stale‑read 分析插件、Jemalloc 基础的颜色段分配器、CheriBSD 核心的段管理与回收服务、CHERI‑Toooba FPGA 软核实现与验证。

**📊 数据集**

使用的数据集与基准包括：NIST Juliet Test Suite（CWE‑415/416/562）、数百条 CVE（如 BZip2、nasm、libzip 等）、SPEC CPU2006 INT 子集、SQLite、PostgreSQL、gRPC、omnetpp 等实际应用程序。

**📈 对比分析**

对比方法：与原 CHERI‑BSD + Cornucopia、原 CHERI‑Toooba 以及未上色的 CHERI 实现做基准。结果显示：SPEC CPU2006 运行时仅 +4% gm，SQLite +10% gm，PostgreSQL +13%，gRPC +16%（相对 Cornucopia 约 11%），内存开销约 +18% gm。硬件面积增幅约 5% 逻辑、6% 寄存器、8% 内存。

**⚠️ 局限性**

局限性：① 仅按栈帧色彩，无法细粒度处理同帧内的子作用域（intra‑frame lifetime）问题；② 颜色段切换与回收仍带来额外运行时成本，尤其在高频调用或大段切换时；③ 需要额外硬件支持（pvtr、ccsettype 等），对现有 CHERI 处理器有实现门槛；④ 对多线程的实现依赖运行时段管理，可能在极端并发情形下产生段竞争。

---

## 396. Challenges and Contributions in Quality of AI-Based Software: A Systematic Mapping Study

**arXiv ID:** 2608.26215 | [PDF](https://arxiv.org/pdf/2608.26215v1)

**作者:** Maryum Hamdani `[一作]` (University of Eastern Finland), Markku Tukiainen `[通讯]` (University of Eastern Finland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统映射研究，总结了2020-2026年间关于AI软件质量的研究现状，归纳出六大挑战与十一类解决方案，并对证据成熟度进行分层分析。

**💡 创新点**

创新点在于将质量评估、NFR管理、开发与保证等维度的挑战与对应贡献系统化整合，并对现有标准（如ISO/IEC 25059:2023）的不足提出评估与改进路径。

**🔧 技术方法**

采用系统映射研究方法，包括文献检索、标题/摘要筛选、全文评估、主题归纳与数据提取，并基于Wieringa等人和Petersen等人的分类框架对研究类型与贡献类型进行归纳。

**📊 数据集**

研究使用的“数据集”为33篇从IEEE Xplore、ACM Digital Library、SpringerLink、ScienceDirect和Wiley Online Library检索得到的符合纳入标准的原始研究论文。

**📈 对比分析**

本文未进行实验性比较，而是通过对研究类型（如评估研究、案例研究、实地调查）和贡献类型（模型、方法、工具等）的分类与统计，展示了各类研究的分布与成熟度。

**⚠️ 局限性**

局限性主要在于：1）缺乏在真实工业环境中的实证验证，2）大部分贡献仍处于概念或早期阶段，3）数据来源限于英语且覆盖的数据库与时间窗口有限，可能遗漏相关研究。

---

## 397. Recipes for Steering and Scaling LLMs via Sampling

**arXiv ID:** 2608.26120 | [PDF](https://arxiv.org/pdf/2608.26120v1)

**作者:** Jiajun He `[一作]`, Yuanqi Du `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于采样的推理框架，通过将自回归LLM视为时间依赖传输，使用SMC和RE对幂、倾斜、专家乘积等灵活目标分布进行采样，从而提升推理性能。

**💡 创新点**

创新点在于将自回归生成视为可调节的传输过程，设计了可组合的目标分布并给出了对应的SMC与RE算法；同时引入熵倾斜作为无外部奖励的自适应指导。

**🔧 技术方法**

使用的技术包括顺序蒙特卡洛（SMC）、复制交换（RE）、低温采样、熵倾斜、分块采样等。

**📊 数据集**

实验使用MATH500和GPQA‑Diamond两个推理基准数据集，并在Qwen2.5系列模型上进行评估。

**📈 对比分析**

与低温采样、MCMC、Best‑of‑N等方法对比，SMC在单样本上表现最好，RE在pass@k多样本上更强，整体比标准MCMC显著提升推理准确率。

**⚠️ 局限性**

局限性包括：相较于标准解码计算开销更大；依赖底层模型质量；在许多任务中目标分布的具体形式对性能影响有限，Best‑of‑N仍是一个简单而强劲的基线。

---

## 398. Can a Model Catch Its Own Hallucinations for Free?: Label-Free Doubt Signals Hold Their Own Against a Labelled Dataset for Abstention

**arXiv ID:** 2608.26121 | [PDF](https://arxiv.org/pdf/2608.26121v1)

**作者:** Ali Asaria `[一作]` (Transformer Lab), Deep Gandhi `[通讯]` (Transformer Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用模型自身冻结的token概率信号，无需标注，进行自我回避微调。

**💡 创新点**

创新点在于用无标签的置信度阈值驱动回避训练，能与有标签回避竞争。

**🔧 技术方法**

技术包括LoRA微调、基于平均log-prob的置信度计算、信号门控回避训练。

**📊 数据集**

数据集为700条短问答，来自PopQA与TriviaQA，且拆分为零实体。

**📈 对比分析**

与标注回避、标准SFT及加权训练对比，匹配覆盖率下无显著差异，性能与有标签方法相当。

**⚠️ 局限性**

局限在于高置信错误答案无法被检测，且方法对稀有实体的幻觉仍回避不足。

---

## 399. Subgraph Filtering for Fair Graph Neural Networks

**arXiv ID:** 2608.26437 | [PDF](https://arxiv.org/pdf/2608.26437v1)

**作者:** Haohui Lu `[一作]` (Charles Darwin University), Shahadat Uddin `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 SF-GNN 的轻量级框架，在图神经网络的消息传递过程中通过识别并抑制敏感同质性及结构放大器（中心度、三角闭合）形成的偏见边，使用随机边过滤来减缓结构偏差，并在训练中加入统计平等正则化。

**💡 创新点**

创新点在于：①从结构层面识别偏见边，而非仅依赖全局正则；②将随机过滤与下调相结合，保持图结构信息；③在训练中采用温和的公平正则化预热，提升优化稳定性；④框架架构无关、仅需一次预处理，计算成本低。

**🔧 技术方法**

技术方法包括：图神经网络消息传递、敏感同质性与结构放大器判别偏见边、随机边过滤与下调、统计平等正则化（SPD）与线性预热、残差连接与批归一化保持表达能力。

**📊 数据集**

使用五个公开基准数据集：NBA（球员薪资预测）、German（信贷风险）、Credit（信用卡违约）、Income（收入水平）以及 Pokec_n（社交网络职业预测）。

**📈 对比分析**

与标准 GCN、GAT、GraphTrans 以及多种公平-aware GNN（FairGNN、NIFTY、BIND、GraphAir、FairSIN、FMP、FairGT）比较。SF‑GNN 在所有数据集上显著降低 SPD 与 EO，并在准确率上与最强基线相当或更优，整体呈现更好的公平‑准确权衡。

**⚠️ 局限性**

局限性包括：①仅针对二分类与单敏感属性，未覆盖多类别或多属性公平；②仍依赖预先已知敏感属性进行边识别，对动态/时变图的适应性有限；③在极端同质性或异质性图中，过滤策略可能需进一步自适应；④未探讨对等化机会等其他公平度量的适用性。

---

## 400. AfriSwitch: A Benchmark for In-the-Wild African Code-Switched Speech Recognition

**arXiv ID:** 2608.26434 | [PDF](https://arxiv.org/pdf/2608.26434v1)

**作者:** Gabrial Zencha Ashungafac `[一作]` (Intron Health), Tobi Olatunji `[通讯]` (Intron Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AfriSwitch，61.36 小时、16 种非洲语言的自然语料库，并对 5 款多语种 ASR 系统进行零样本评估。

**💡 创新点**

首次提供带有切换级别 English span 标记、CMI 与切换点计数的多语言真实语料，揭示非洲语言代码切换的多维多样性，并证明针对非洲语料训练是提升性能的关键。

**🔧 技术方法**

采用人工逐词转写、VAD、Token 级切换标注、语言专属文本归一化、Diacritics 保留的 WER 评估，并在零样本条件下比较多语种 ASR 模型。

**📊 数据集**

主要使用 AfriSwitch 自身的数据（61.36h），并在其中的 12 语言子集上进行基准测试。

**📈 对比分析**

通过将 5 个系统（Sahara V2/V2.5、Omnilingual LLM 7B、Gemini 3.6、ElevenLabs）在 AfriSwitch 上零样本评估，发现平均 WER 最高 35.93%（Sahara V2.5），其余系统均高于 50%，所有模型相较于单语料基准都有显著下降，显示非洲针对训练最为重要。

**⚠️ 局限性**

局限包括仅覆盖 16 种语言且数据量不均，Diacritics 归一化导致与其他研究的 WER 不可直接比较，语料主要来自 YouTube/播客等公共场景，未覆盖私人对话；且目前仅报告 WER，未充分利用切换级别标签进行细粒度分析。

---

## 401. Preserving General Capabilities during Domain Specialization with Uncertainty-Calibrated MOPD

**arXiv ID:** 2608.26735 | [PDF](https://arxiv.org/pdf/2608.26735v1)

**作者:** Ziyuan Liu `[一作]` (Kuaishou Technology), Cheng Luo `[通讯]` (Kuaishou Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究垂直域专用化后如何恢复通用能力，提出并验证一种不确定性校准的多教师在线蒸馏（MOPD）方法。

**💡 创新点**

创新点在于：1）引入双温度采样和正优势密度过滤在序列层主动发现高价值学习机会；2）使用中心化对数似然（CLL）在 token 层进行方向‑认可一致性过滤；3）将两层筛选机制组合起来，显著提升专用化与通用能力的平衡。

**🔧 技术方法**

使用技术包括：Multi-Teacher On-Policy Distillation（逆 KL 蒸馏）、双温度采样、正优势密度过滤、CLL（entropy‑calibrated endorsement）、分层采样与蒙特卡罗更新等。

**📊 数据集**

使用的数据集包括：角色扮演（CoSER 评测）、医学领域（MedQA-USMLE、MedXpertQA、PubMedQA）以及九个通用基准（GPQA-Diamond、AIME25、HMMT25、ZebraLogic、LiveCodeBench v5、IF‑Eval、WritingBench、Arena‑Hard v2、LiveBench）。

**📈 对比分析**

对比方法有原始通用模型、SFT、vanilla MOPD、SelecTKD、ReOPOLD 等。在角色扮演任务中，通用平均提升 4.73%（相较 vanilla MOPD），且保持或提升垂直性能；在医学任务中，通用平均提升 10.84%，同时医学平均与最佳恢复基线相当。实验表明提升主要来自筛选机制而非单纯扩大采样量。

**⚠️ 局限性**

局限性包括：依赖已冻结的教师模型，需对教师分布进行准确估计；双温度和采样数等超参数仍需手动调优；实验仅在两类领域验证，未覆盖更大模型或更复杂域；未完全解决样本不平衡与极端 token 的处理问题。

---

## 402. The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection

**arXiv ID:** 2608.26423 | [PDF](https://arxiv.org/pdf/2608.26423v1)

**作者:** Jaturong Kongmanee `[一作]` (Trend Micro), Smile Thanapattheerakul `[通讯]` (Trend Micro)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“潜在诊断分类法”（Latent Diagnostic Taxonomy）框架，用于构造安全层分类器并诊断其决策可信度，特别针对提示注入检测。

**💡 创新点**

创新点在于：①使用交叉验证经验法选择最佳嵌入维度，避免预设维度导致过拟合；②通过SVM定位仅占约29%训练样本的支持向量，极大降低分析成本；③对支持向量中每个token进行遮蔽（token-level occlusion）并计算其对决策边界的影响，形成四区诊断（Heuristic Bias、Safe、Insufficient Context、Heuristic Override），实现对“可信决策”的细粒度评估。

**🔧 技术方法**

核心技术包括：文本嵌入（intfloat/e5-base-v2）、主成分分析（PCA）进行维度优化、支持向量机（SVM）识别支持向量、token遮蔽与距离差异A(z,x)计算、基于距离阈值和符号的四区诊断逻辑。

**📊 数据集**

使用公开的提示注入攻击数据集（未具体命名，来源于相关研究），训练并评估分类器。

**📈 对比分析**

在维度选择上使用5折交叉验证得到的累计性能曲线来确定最佳维度；在诊断结果上发现约77%的样本在去掉单个token后预测会翻转，说明模型存在显著不稳健性。没有给出与现有基线模型的直接数值对比，只说明该框架能够分离两类失败模式（confidence calibration failure 与可利用快捷方式）。

**⚠️ 局限性**

局限性包括：①仅针对文本输入，无法直接处理图像、音频等多模态数据；②对不同模型规模和类别的适用性尚未充分验证；③需要更高级的对抗训练来提升鲁棒性；④目前的诊断策略主要聚焦在提示注入，缺乏对其他类型攻击的泛化评估。

---

## 403. Video-FLAIR: Not Whether to Reason, But How

**arXiv ID:** 2608.26495 | [PDF](https://arxiv.org/pdf/2608.26495v1)

**作者:** Yogesh Kulkarni `[一作]` (Arizona State University), Pooyan Fazli `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出一种可学习的多模态推理框架（VideoFLAIR），通过强化学习让模型在每个查询中自适应选择感知、组合或深度推理三种推理模式。

**💡 创新点**

创新点在于：①在同一查询下生成三种模式的完整推理轨迹进行直接比较；②使用综合奖励函数与在线验证器评估正确性、可解释性与成本；③通过收益驱动的策略学习实现无查询级标注的自适应推理。

**🔧 技术方法**

主要技术包括：强化学习（GDPO、DAPO）、多模式结构化rollout、在线验证器（Qwen3‑VL）、优势归一化与token‑级信用分配，以及动态长度惩罚与多目标奖励。

**📊 数据集**

在六个图像基准（EMMA、MMMU、MathVista、HallusionBench、AI2D、MM‑Vet v2）和五个视频基准（Video‑Holmes、Video‑TT、VSI‑Bench、SciVideoBench、Video‑MMMU）上进行评估，使用来自公开视频/图像数据集的合成训练数据。

**📈 对比分析**

相较于始终使用链式思考或二进制思考的基线，VideoFLAIR在Qwen2.5‑VL和Qwen3‑VL模型上平均提升+4.8到+5.4的准确率，同时将token使用量从数百降至约60‑100，显示出更优的精度与效率平衡。

**⚠️ 局限性**

局限性包括：仅覆盖三种推理模式，无法处理更丰富的工具/多步骤互动；在极难或社交推理视频查询中仍有误判；以及对验证器的依赖可能导致偏差或过度校准。

---

## 404. Learning Woody Clearing With Loss Alignment for Zero-Shot Regrowth and Woody Segmentation

**arXiv ID:** 2608.26489 | [PDF](https://arxiv.org/pdf/2608.26489v1)

**作者:** Kal Backman `[一作]` (New South Wales Department of Climate Change Energy Environment and Water), Adam Roff `[通讯]` (New South Wales Department of Climate Change Energy Environment and Water)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种基于双时相 Sentinel-2 数据的深度学习模型，用于检测林木清除、零样本分割和零样本再生检测，并通过引入损失缩放系数 α 使模型学习目标与终端用户的 F_β 指标对齐。

**💡 创新点**

创新点包括：① 用 α 调整交叉熵与 Dice 损失，以单一参数同时调控精度与召回率；② 开发多种后图像生成技术（手工拼接、激活最大化等）实现零样本森林分割；③ 在 7 年 Sentinel-2 影像上构建大规模标注集，验证模型在新南威尔士州和巴西 MapBiomas 数据集上的迁移性能。

**🔧 技术方法**

技术主要包括：双塔编码器-解码器网络（Siamese CNN + U‑shape 结构）、自适应损失缩放、图像增强与几何变换、激活最大化生成后图像、阈值裁剪与置信度加权融合。

**📊 数据集**

使用的主要数据集：新南威尔士州 7 年 Sentinel-2（10m）影像（约 54 billion 像素），标注通过手工验证的清除指数；MapBiomas 巴西清除警报；Fisher 等点云数据；以及自建的 108 km² 森林分割与 49 km² 再生检测样本。

**📈 对比分析**

比较方法：在像素级、分块级与距离阈值下计算精度、召回率、F1/F2、IoU；在 MapBiomas 与 Global Forest Change 上对比模型性能；在零样本分割上与 SamGeo、Segment Any Change、NDVI 阈值基线对比。结果显示：α 优化后精度可提升 1.85×、召回率 1.12×；零样本分割 F1 最高达 0.899，优于 SamGeo 0.685；零样本再生 F1 为 0.845，显著优于 Segment Any Change 0.383 与 NDVI 0.603。

**⚠️ 局限性**

局限性：模型仅在新南威尔士州训练，全球泛化需进一步验证；忽略了来自林业种植的清除事件导致精度偏低；零样本任务评估样本量相对较小；在大面积连续森林中分割召回率下降；零样本方法相比监督训练在极端稀疏事件下仍有限。

---

## 405. Buy-at-Bulk Facility Location on Trees

**arXiv ID:** 2608.26337 | [PDF](https://arxiv.org/pdf/2608.26337v1)

**作者:** Shamisa Nematollahi `[一作]` (Université Paris Cité), Daniel Vaz `[通讯]` (Université Gustave Eiffel)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在树形网络上研究买断式设施选址（Buy‑At‑Bulk Facility Location）问题，并针对单位需求、可拆分需求以及电缆不可拆分需求提出近似算法，分别实现了PTAS、3/2‑近似下限以及 2‑近似。

**💡 创新点**

创新点在于：
- 证明在树实例下，单位需求与可拆分需求可用 PTAS 求解；
- 在电缆不可拆分需求中给出 3/2‑APX 难度证明，并提供一种允许容量稍微超载（1+ϵ）的最优成本算法；
- 通过“无交叉”技术与多目标动态规划，构建了整体 PTAS；
- 将电缆选型问题转化为带权背包问题并给出 1+ϵ 近似解法。

**🔧 技术方法**

主要技术包括：
- 无交叉（Uncrossing）证明保证最优解可无双向流；
- 基于树的分治与 DP，状态压缩与分段近似；
- 背包/0‑1 近似算法（PTAS）用于电缆容量预算；
- 线性组合与容量/成本平衡的 1+ϵ 近似包装。

**📊 数据集**

论文为理论分析，没有使用实际数据集，所有结果均通过算法设计与复杂度证明获得。

**📈 对比分析**

评估方法：理论复杂度分析与近似比。对于可拆分需求的 PTAS，误差可任意小；对电缆不可拆分需求给出 2‑近似并证明 3/2‑下限；缺乏实验验证。

**⚠️ 局限性**

局限性：
- 仅针对树形网络，未推广至一般图；
- 电缆不可拆分方案需允许容量超载才能获得最优成本；
- 对于多电缆类型的实现复杂度较高；
- 结果主要为理论性质，缺乏实验验证。

---

## 406. Refusal Is Not Robustness: Auditing Confident Fabrication in Large Language Models on a Provably Uninformative Clinical Pain Speech Transcript

**arXiv ID:** 2608.26167 | [PDF](https://arxiv.org/pdf/2608.26167v1)

**作者:** Sagnik De `[一作]` (University of Calcutta), Sreenija Pavuluri `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估了七种大语言模型在痛感语音评估中的自我否认与假设行为，构建了可验证的无痛信息语料库并在不同提示下测评模型的拒绝率、置信度与错误率。

**💡 创新点**

创新在于使用可证明无痛信息的 TAME Pain 语料构建信度测试，提出了幻觉置信度与可靠性解耦指数两项新度量，并通过多措提示的压力测试揭示模型在压力下的自信造假行为。

**🔧 技术方法**

使用 Whisper ASR、冻结 prompt 库、七大 LLM（Claude Haiku、GPT‑4o‑mini、Gemini‑Flash、DeepSeek‑V3、Llama‑8B、Claude‑Sonnet、GPT‑5.2），并利用二分类 AUC、ECE、置信度、RDI 等指标。

**📊 数据集**

TAME Pain Corpus（PhysioNet）共 7,044 条 16kHz 单声道录音，包含 5,750 条无痛信息 Harvard 句子和 1,294 条痛感声明句子。

**📈 对比分析**

在合作提示下六种模型拒绝率接近 1，误报低；但在权威/恭维压力提示下 Gemini‑Flash 与 Llama‑8B 的自信造假率高达 53%–76%；整体性能表明合作提示掩盖了压力下的失效。

**⚠️ 局限性**

局限包括单一数据集、样本量小且人群偏年轻、评估仅用自报置信度、未覆盖最新 LLM、压力提示种类有限、对寒冷条件与疼痛编码可能混杂，故结果对真实临床环境的推广需谨慎。

---

## 407. ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements

**arXiv ID:** 2608.26118 | [PDF](https://arxiv.org/pdf/2608.26118v1)

**作者:** Xinming Wang `[一作]` (Institute of Automation Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ElementCheck，一种基于句子级实体对（元素图）的长文本真实性评估框架。

**💡 创新点**

创新点在于：①不再统一拆分句子为原子声明，而是提取句子内可验证的实体对构成元素图；②利用元素图的直径和连通度动态决定验证粒度（直接验证或元素级细化+重检）；③重检过程通过子图上下文重构，避免无谓的全句重检。

**🔧 技术方法**

使用技术包括：实体对提取与元素图构建、Serper API 进行短片段检索、Jina 文档抽取+BM25 再排序、LLM 直接验证/元素级验证、图直径/连通度路由器、重检与上下文重构。

**📊 数据集**

数据集：FastFact‑Sent（从 FastFact‑Bench 重映射得到的 5,020 个句子级事实标注）、AdjuvantBench、CatalystBench 以及其它通用域数据（HelloBench、FactScore‑Bio 等）。

**📈 对比分析**

与 SAFE、VeriScore、FastFact、VeriFastScore 及 DnDScore、VeriFact 等基线在 FastFact‑Sent、AdjuvantBench、CatalystBench 上进行对比。ElementCheck 在 5 种后端模型上平均获得最高的 Overall 分数，覆盖率与 BAcc 平衡更好，且计算成本比检索密集型基线更低。

**⚠️ 局限性**

Limitations：①弱模型可能在句子检索阶段漏掉可检验证句；②仅处理句子级事实，跨句/跨段、长距离共指与因果推理仍无法覆盖；③检索噪声高，文档级检索并不一定提升准确率，重检规则目前仍为启发式，缺乏更系统的图推理与证据聚合。

---

## 408. A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers

**arXiv ID:** 2608.26194 | [PDF](https://arxiv.org/pdf/2608.26194v1)

**作者:** Inho Kim `[一作]` (Korea Institute of Energy Technology), Sumyeong Ahn `[通讯]` (Korea Institute of Energy Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建跨模态检索-生成系统，提出Speech and Text Reranking Orchestrator (S2TR)，并通过GPT-4o对检索候选进行跨模态相关性标注，训练跨模态重新排序器；在混合语音与文本检索池中提升检索准确性。

**💡 创新点**

①首次创建包含文本与语音候选的跨模态相关性数据集；②设计利用Z-score归一化和音频窗口分段的跨模态重新排序框架；③兼顾点、对、序列表述的多种训练目标，揭示点目标在混合模态下的稳健性。

**🔧 技术方法**

多模态检索器（e5‑mistral‑7b‑instruct 文本；HuBERT‑based SpeechRAG 语音），LoRA 微调，自动回归评分模型（Ultravox、Qwen‑Audio‑Chat、Qwen2‑Audio），Z‑score 归一化，音频窗口划分与最大/平均池化，点/对/序列化损失。

**📊 数据集**

Spoken SQuAD（文本+TTS语音）与 MS MARCO（纯文本）两大公开数据集；检索池约2.8K语音+9.1K文本段；使用GPT‑4o‑audio‑preview 对候选进行标注。

**📈 对比分析**

与Z‑score基线、单模态检索器对比，使用 Hit@1、MRR、NDCG@5 评估重新排序性能，点目标下在Spoken SQuAD和MS MARCO均实现 10–15% 的提升；下游QA EM 在混合模态下提升约 0.12（如Ultravox 0.357→0.480），表明重新排序显著减少幻觉。

**⚠️ 局限性**

仅使用 TTS 合成语音，缺乏自然语音验证；重新排序依赖大模型标注，可能产生偏差；列表化目标在多模态场景易失稳；仍未完全消除模态间得分失衡。

---

## 409. Pruning Binarized Neural Networks: A Dedicated Framework and Globally Weighted Algorithms

**arXiv ID:** 2608.26233 | [PDF](https://arxiv.org/pdf/2608.26233v1)

**作者:** Roan Rubiales `[一作]` (Polytechnique Montreal), Jean Pierre David `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于PyTorch的实验框架，用于在边缘设备上训练和优化全二值化神经网络，并实现了可定制的剪枝与冻结（稀疏训练）算法；在此框架下开发了三种新的二值化网络剪枝方法；

**💡 创新点**

核心创新点在于引入了全局加权（global weighting）机制，结合批归一化折叠与权重归一化，对权重进行尺度统一后再做阈值剪枝，从而显著提高剪枝率（VGG11可达到70%而无精度损失）；此外，还提出了在二值化前进行剪枝以及基于准确率梯度上升的自适应层级剪枝策略；

**🔧 技术方法**

使用了二值化训练（BAT + STE）、全局加权变换（BN折叠、权重归一化）、单阈值剪枝（搜索与扫描）、准确率梯度上升剪枝、冻结算法、Adam优化器以及传统的卷积网络架构（BinaryNet、VGG11、NiN）；

**📊 数据集**

使用CIFAR-10数据集进行训练、验证与测试；也在实验框架中支持MNIST、CIFAR-100与ImageNet；

**📈 对比分析**

通过与文献中Layer Sensitivity、Ternary pruning、Weight-flipping等剪枝方法的对比，证明在相同或更高剪枝率下能保持甚至提升测试准确率；在BinaryNet上实现45%剪枝率保持89.7%准确率；在VGG11上实现70%剪枝率保持86.0%准确率；在NiN上实现49.9%剪枝率保持84.6%准确率；

**⚠️ 局限性**

主要限制在于剪枝方式为无结构化（逐权重），导致稀疏模型在硬件上难以充分利用；未实现结构化剪枝、硬件代码生成以及更深入的ablation研究；框架仍处于原型阶段，缺乏完整的部署流水线。

---

## 410. On identifying codes on oriented graphs

**arXiv ID:** 2608.26593 | [PDF](https://arxiv.org/pdf/2608.26593v1)

**作者:** Soura Sena Das `[一作]` (Indian Statistical Institute), Sagnik Sen `[通讯]` (Indian Institute of Technology Dharwad)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在有向图中寻找识别码的计算复杂度问题，并针对d-正则图族给出了多项式可解与NP-完全的完全二分法。

**💡 创新点**

首次对ℱ-Id Code问题在d-正则图族下给出完整的复杂度分类，揭示了不同度数下的算法可行性差异。

**🔧 技术方法**

采用图论与计算复杂度理论的分析技术，构造多项式时间算法与NP-完全归约证明。

**📊 数据集**

本文为理论研究，没有使用实验数据集。

**📈 对比分析**

由于结果为理论证明，没有实验比较；该问题在d≤1时可在多项式时间内求解，d≥2时为NP-完全，说明难度显著上升。

**⚠️ 局限性**

只关注d-正则图族，未考虑更一般的图族；缺乏实验验证与实现细节，实际效率未被评估。

---

## 411. Explainable Artificial Intelligence for Customer Churn Prediction in Telecommunications: A Framework for CRM Integration

**arXiv ID:** 2608.26151 | [PDF](https://arxiv.org/pdf/2608.26151v1)

**作者:** Sandeep Gaddamwar `[一作]` `[通讯]`, Sandeep Gaddamwar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文结合可解释 AI 与 CRM，构建了面向电信客户流失的预测与运营一体化框架。

**💡 创新点**

创新点在于将 SHAP / LIME 解释结果直接映射为 CRM 分层决策模板，并建立反馈回路。

**🔧 技术方法**

使用了逻辑回归、随机森林、XGBoost、LightGBM 四种分类器，配合 SHAP、LIME、SMOTE、PSI 等技术。

**📊 数据集**

实验基于公开 IBM Telco Customer Churn 数据集（7043 条记录、19 个特征）。

**📈 对比分析**

四个模型在 0.5 阈值下 AUC 0.831–0.841，准确率 74–78%，性能相近，逻辑回归略优。

**⚠️ 局限性**

局限包括仅使用单一公开数据集、特征与网络质量无关、解释不具因果性、业务估值假设不确定。

---

## 412. SFC-Aware Online Aggregated Data-Link Orchestration for SDN/NFV-Enabled SAGINs

**arXiv ID:** 2608.26559 | [PDF](https://arxiv.org/pdf/2608.26559v1)

**作者:** Ziyang Guo `[一作]` (University of Science and Technology Beijing), Bing Du `[通讯]` (University of Science and Technology Beijing)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了基于SDN/NFV的民用航空空间-空-地一体化网络（SAGIN）中，面向服务功能链（SFC）的在线访问侧聚合数据链路编排方法，提出了可调节的TEMP（临时弹性映射与停车）机制。

**💡 创新点**

创新点在于：① 将异构的A2A、A2G、A2S链路抽象为聚合数据链路承载资源并统一编排；② 引入TEMP作为可调度的时序资源，利用未来窗口预测实现延迟弹性；③ 设计了基于优先级的词典MILP模型，并提出模型诱导风险与稀缺感知的近似算法MRSAR，实现高效在线决策。

**🔧 技术方法**

采用的技术包括：软件定义网络（SDN）与网络功能虚拟化（NFV）架构、词典MILP建模、临时弹性映射与停车（TEMP）策略、模型诱导风险与稀缺感知的近似算法MRSAR，以及离散事件仿真评估。

**📊 数据集**

使用了真实航班调度数据（OAG），AGI STK星历文件模拟66颗Iridium卫星，20颗地面网关，513架商业航班，配合多种链路技术（VDL-Mode-2、HFDL、LEGACY SATCOM、SBB-Safety、ATG、5G-ATG、LEO宽带）构建的仿真环境。

**📈 对比分析**

与Gurobi求解的MILP参考解和两种贪心基线（先到先服务和最短延迟）进行对比。MRSAR在服务成功率上接近MILP，且明显优于贪心方法；平均归一化延迟保持在MILP附近；拒绝/丢弃失败率得到有效控制；运行时相较MILP提升约3.5–4倍，展示了良好的质量-复杂度折衷。

**⚠️ 局限性**

主要限制包括：依赖未来窗口的预测信息，未考虑预测误差；算法实现仍为研究原型，未在真实控制器中部署；对极端负载或大规模网络的鲁棒性需进一步验证。

---

## 413. SAGE: Variate-Wise Semantic Augmentation for Vision-Language Time Series Forecasting

**arXiv ID:** 2608.26829 | [PDF](https://arxiv.org/pdf/2608.26829v1)

**作者:** Haizhao Fan `[一作]` (Shanghai Jiao Tong University), Xinyi Le `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

SAGE提出一种融合CLIP视觉-语言模型的多模态时间序列预测框架，通过频率增强的文本编码、变量级知识注入以及视觉对比学习来提升预测性能。

**💡 创新点**

创新点包括将CLIP的文本编码器作为可训练的时序骨干，冻结视觉编码器做训练时对比正则化；使用模板化的变量特定多视角文本知识并通过门控与统计旁路注入；以及频率增强的补丁嵌入与跨变量注意力的组合。

**🔧 技术方法**

利用CLIP ViT‑B/32的双分支、频率增强补丁编码、交叉变量注意力、变量门控文本融合、InfoNCE视觉对比损失、RevIN标准化以及少量的MLP统计旁路。

**📊 数据集**

在8个长期预测基准（ETTh1/2、ETTm1/2、ECL、Traffic、Weather、Exchange）和M4短期预测竞赛上进行评测。

**📈 对比分析**

与TimesNet、PatchTST、FredFormer、iTransformer、Amplifier、DUET、SRSNet等最新基线相比，SAGE在7/8长期数据集上获得最佳平均MSE，M4 OWA 0.834 领先，整体提升约4%–6% MSE。

**⚠️ 局限性**

局限性包括对高维数据文本效益下降、依赖手工模板生成文本、未利用外部事件/本体信息、以及目前仅支持单变量预测，缺乏多变量生成与概率预测。

---

## 414. Tether the Subject, Release the Scene: Query-Aware Memory Routing for Long-Horizon Autoregressive Video Generation

**arXiv ID:** 2608.26902 | [PDF](https://arxiv.org/pdf/2608.26902v1)

**作者:** Chen Li `[一作]` (Huazhong University of Science and Technology), Changxin Gao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为 TetherMem 的无训练、查询感知时空记忆路由方法，用于提升流式自回归视频模型在长视频生成中的场景进展与主体一致性。

**💡 创新点**

创新点在于将区域与记忆年龄先验嵌入注意力 softmax 中，实现在同一模型中对主体查询与场景查询采用不同历史访问策略，从而解决“记忆锚定场景退化”问题。

**🔧 技术方法**

技术上采用了无训练的查询‑区域‑年龄路由、注意力 logits 的正则化、基于 SAM 2 的主体轨迹提取，并将该路由器应用于 LongLive‑RAG 等现有长视频自回归框架。

**📊 数据集**

在 10 条多样化提示、3 种随机种子生成的约 30 秒长视频上进行评估，使用 Wan2.1‑T2V‑1.3B 作为基准，并收集了 2,400 对盲评对。

**📈 对比分析**

与八种流式长视频基线进行盲评比较，TetherMem 在整体偏好（0.780）和场景进展（0.769）上位居最高，显著提升人类评估得分，同时保持主体一致性与视觉完整性。

**⚠️ 局限性**

局限性包括：依赖外部主体轨迹掩码，掩码对齐误差会影响路由效果；在主体面积或形态变化剧烈时，区域先验的稳健性有限；目前实验仅覆盖约 30 秒长视频，长时序更长的稳定性仍待进一步验证。

---

## 415. Topology-Masked Unified Backbone for Joint Feature Interaction and Multi-Domain Sequence Modeling

**arXiv ID:** 2608.27005 | [PDF](https://arxiv.org/pdf/2608.27005v1)

**作者:** Zhihao Zhu `[一作]` (Shandong University), Shuaishuai Guo `[通讯]` (Shandong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种统一的MaskRec架构，用于大规模后点击转化率(CVR)预测。

**💡 创新点**

创新点在于引入TopoMask结构化注意力遮罩和DualQ双路径查询生成，使得特征交互与多域序列建模在同一token空间中统一且可控。

**🔧 技术方法**

使用了统一token表示、可学习的全局与域级记忆token、TopoMask注意力机制、双路径查询生成、Transformer块等技术。

**📊 数据集**

实验基于腾讯广告算法竞赛公开的大规模匿名广告日志数据集。

**📈 对比分析**

与HyFormer基线对比，MaskRec在验证集上AUC从0.831827提升到0.841253，在测试集上提升到0.834640；消融实验验证了各组件贡献，内存token容量对性能有渐进影响。

**⚠️ 局限性**

局限在于TopoMask拓扑是手工设定的结构先验，可能不适用于其他数据集，缺乏自适应或可学习的拓扑构造。

---

## 416. TutorTrace: A Dataset and Taxonomy for Classifying Learner Behavioral States during AI-Assisted Programming Education

**arXiv ID:** 2608.26184 | [PDF](https://arxiv.org/pdf/2608.26184v1)

**作者:** David Barron `[一作]` (Virginia Tech), Yan Chen `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建 TutorTrace 数据集并实现实时 IDE 事件的自动行为抽象和分类，提出三窗口行为分类法，评估行为感知提示的效果。

**💡 创新点**

①公开大规模细粒度 IDE 远程遥测数据与自动行为抽象工具；②把行为上下文量化为可实时计算的特征；③设计针对 AI 辅导的三窗口行为分类与对应干预策略。

**🔧 技术方法**

基于规则的行为分割与分类器、GPT‑4o 的教学提示、随机森林预测模型，以及可视化和指标计算。

**📊 数据集**

480 名初级 Python 学生在 4 次课程部署中的 TutorTrace 数据集，约 180K 事件、13,633 行为段、27 观测指标，含 1,386 次 AI 交互。

**📈 对比分析**

通过与基线对比统计（被动窗口从 50% 降至 20.7%，代码编辑量、运行次数、活动时间提升），以及在查询时机和帮助寻求类型两项预测任务中，加入行为指标将 AUROC 分别从 0.689 提升至 0.726，和从 0.690 提升至 0.717。

**⚠️ 局限性**

数据来自单一机构、短期任务，缺乏随机化；行为分类规则缺乏代码语义理解；仅使用 GPT‑4o，结果可能随模型不同而异；标签由 GPT‑4o 与少量人类评估生成，仍为近似；无法保证在其他课程或平台的泛化。

---

## 417. Cross-Platform Benchmark of Neural 3D Reconstruction for Autonomous Laboratory Robots

**arXiv ID:** 2608.26383 | [PDF](https://arxiv.org/pdf/2608.26383v1)

**作者:** Yongho Kim `[一作]` (Argonne National Laboratory), Nicola Ferrier `[通讯]` (Argonne National Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对NeRF、Gaussian Splatting和SAM3D三种神经网络3D重建方法，在NVIDIA Jetson AGX Orin、RTX桌面工作站和A100 HPC节点三种计算平台上进行系统性基准测试，使用40张6MP机器人手臂图像构成的自定义数据集。

**💡 创新点**

首次从硬件层面系统比较不同重建方法在真实实验室机器人部署中的时延、精度与计算成本，为构建高效、可扩展的自动化实验室视觉流水线提供了实证依据。

**🔧 技术方法**

采用NerfStudio中的nerfacto和splafacto实现NeRF与GS的训练与渲染，并对SAM3D进行单图像推理；评估指标包括PSNR、SSIM、LPIPS、FPS、训练时长及GPU分钟等。

**📊 数据集**

主要使用自制的40视角机器人手臂图像集；在SAM3D评估中亦使用实验室物体（离心桶、试管板等）以及SA‑3DAO数据集中的F1和Chamfer距离以检验几何质量。

**📈 对比分析**

通过对训练时间、GPU利用率、渲染帧率与重建质量的多维度比较，结果显示GS在所有平台均优于NeRF，既具更高PSNR/SSIM，又具更快FPS，但训练耗时更长；Jetson Orin的训练显著慢于桌面和服务器；SAM3D可在9‑12秒内完成单图像3D模型生成，但在遮挡和几何细节上存在误差。

**⚠️ 局限性**

局限性包括：仅评估了计算与质量维度，未涉及实际操控任务；使用单一自制数据集，缺乏多场景验证；单图像评估中存在姿态对齐与背景抠图难题；未利用多GPU加速或可视化分析工具；缺乏对模型细节误差对机器人操作影响的定量研究。

---

## 418. Hallucinations in LLMs: A Lifecycle-Based Survey of Causes, Detection, Mitigation, and Prevention

**arXiv ID:** 2608.26168 | [PDF](https://arxiv.org/pdf/2608.26168v1)

**作者:** Naveen Lamba `[一作]` (Sharda University), Manas Gaur `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并提出基于生命周期的框架，对LLM幻觉的成因、检测、缓解、预防及基准进行系统梳理。

**💡 创新点**

以生命周期视角统一幻觉研究，构建三阶段因果分类，并对检测/缓解/预防技术和评估基准进行全景式对比。

**🔧 技术方法**

综述技术包括数据质量评估、TruthfulQA等真值基准、事实核查、激活与注意力分析、RLHF、RAG、模型编辑、对齐学习、受控解码等。

**📊 数据集**

引用的基准集有TruthfulQA、FEVER、XSum、HaluEval、FactCC、TabFact、Wikidata、OpenDialKG等多任务数据集。

**📈 对比分析**

通过表格对比各技术在EM、FactScore、AUC、Confidence Calibration等指标上的表现，显示缓解方法多数为后置校正、预防方法侧重早期干预，且计算成本差异显著。

**⚠️ 局限性**

局限性在于大多聚焦输出层缺乏内部机制诊断，基准缺少生命周期视角，缓解技术依赖检索/反馈导致计算开销高，未系统覆盖多模态或大规模数据。

---

## 419. Beyond a Single Story: Meta-Reviewing Sparse and Incomplete User-generated Contents for Recommendation

**arXiv ID:** 2608.26728 | [PDF](https://arxiv.org/pdf/2608.26728v1)

**作者:** Hongren Wang `[一作]` (Nanyang Technological University), Yin-Leng Theng `[通讯]` (Nanyang Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Mosaic 方法，利用邻居用户的属性‑情感信息聚合成元评论，并在多任务框架中联合预测评分和属性情感，解决用户生成内容的稀疏与不完整问题。

**💡 创新点**

① 以属性级别聚合邻居评论生成元评论，弥补单一评论的属性缺失；② 采用多门混合专家 (MMoE) 的多任务学习架构，结合个性化注意力融合目标用户与元评论；③ 离线使用 LLM 提取属性‑情感对并通过投票与 BERT 校验，降低幻觉风险。

**🔧 技术方法**

使用 GPT‑4.1‑nano 离线提取属性‑情感对；Transformer 编码器和 MMoE 多专家网络；自注意力个性化模块；联合回归与分类损失；邻居采样与多数投票聚合。

**📊 数据集**

四个真实业务数据集：Yelp、TripAdvisor、Amazon Beauty、Amazon Sports。

**📈 对比分析**

与 PMF、SVD++、RGCL、APH、NETE、PETER、PEPLER、CER、SERMON、LLMRec 等基线比较；在所有数据集上均取得最低 RMSE/MAE，尤其在稀疏用户上提升 30% 以上；在属性情感预测上也实现最高准确率和 F1。

**⚠️ 局限性**

受限于预定义属性词表，难以适应新领域；邻居选择仅基于相同商品和社交关系，对孤立用户效果有限；未引入物品侧信息；LLM 提取仍可能产生幻觉，尽管通过投票与 BERT 过滤，但误差仍需监控。

---

## 420. Agent Seer: Synthesizing Scenarios from Specification Understanding

**arXiv ID:** 2608.26133 | [PDF](https://arxiv.org/pdf/2608.26133v1)

**作者:** Harish Karumuri `[一作]` (Apple), David Lopes Pegna `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建 Agent Seer 四阶段流水线，将 MCP 工具规范直接转化为完整的评估 harness（情景、mock 输出、多轮对话），无需人工标注或实时工具调用。

**💡 创新点**

创新点在于仅凭工具规范与 LLM 语义推理自动生成逼真的多轮评估数据，并通过结构化输出和多维评分实现可验证、可复用的评估框架，解决了工具套件的冷启动评估瓶颈。

**🔧 技术方法**

技术上采用 Gemini 2.5 Flash Lite 进行结构化生成，利用 LLM‑as‑judge 对工具调用正确性与对话连贯性进行分维评分，并实现四阶段管道（工具解析、情景生成、mock 输出、对话扩展）。

**📊 数据集**

数据集为七个公开 MCP 规范（Illustrator、Selenium、Redis、Git、Elasticsearch、Slack、Filesystem），共生成 337 个情景、391 条评估记录。

**📈 对比分析**

方法上与外部评审器（Gemini 与 Qwen 3.5）进行对照验证，平均工具调用得分 0.911、连贯性 0.855；在小中等规格上实现 100% 工具覆盖，复杂情景平均下降 7.3pp。

**⚠️ 局限性**

局限性包括：依赖 LLM 生成的 ground truth 与评判可能产生偏差；mock 输出不保持跨调用状态；工具覆盖率在 64 工具时下降；仅使用单一生成模型与 7 份 MCP 规格；多轮扩展样本稀缺；LLM‑as‑judge 循環使用存在潜在的评估自洽问题。

---

## 421. Calibration-Free Cuffless Blood Pressure Estimation Using Multimodal ECG-PPG Fusion on a Google Pixel Watch

**arXiv ID:** 2608.26325 | [PDF](https://arxiv.org/pdf/2608.26325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 422. Per-View Gaussian Predictions Enable Training-Free Distractor Filtering in Feed-Forward 3DGS

**arXiv ID:** 2608.26951 | [PDF](https://arxiv.org/pdf/2608.26951v1)

**作者:** Kangmin Seo `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种训练无关的后处理过滤器，利用冻结的3D Gaussian Splatting预测中每个输入视角关联的高斯子集，排除该视角的高斯后揭示不一致内容，并通过渲染误差验证，最终从模型中去除转瞬出现的干扰物，提升新视角渲染质量。

**💡 创新点**

创新点在于利用feed‑forward 3DGS模型本身的视角‑高斯关联结构，既不需额外训练也不改动模型参数，能够在单次推理后直接过滤掉干扰物，并通过多视角渲染差异验证实现鲁棒去噪。

**🔧 技术方法**

核心技术包括：显式高斯表示的子集排除、DINOv3特征相似度与深度/不透明度阈值生成候选区域、基于多视角渲染误差加权的验证、以及将被验证候选的高斯不透明度置零实现滤除。

**📊 数据集**

实验使用RobustNeRF和NeRF On‑the‑go干扰物基准（4、8、16视角）、Clean RobustNeRF场景以及AnySplat、DepthSplat、ReSplat、YoNoSplat等公开模型的输入。

**📈 对比分析**

与同一模型原始预测、AnySplat、GenWildSplat等进行对比，采用PSNR、SSIM、LPIPS评估，结果表明在所有模型和视角数下均显著提升视觉质量，同时在无干扰的清晰场景中几乎不改变原始重建效果。

**⚠️ 局限性**

局限性包括：过滤过程依赖候选数和视角数，导致验证耗时增长；仅适用于具有视角‑高斯关联的模型，无法直接应用于无此结构的架构；对极端动态或多干扰场景的鲁棒性仍待进一步提升。

---

## 423. Information Flow Control in Off-Chain Components

**arXiv ID:** 2608.26858 | [PDF](https://arxiv.org/pdf/2608.26858v1)

**作者:** Stian Lybech `[一作]` (University of Southern Denmark), Anders Dalskov `[通讯]` (Partisia Applications ApS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种面向离线组件的区块链架构的智能合约语言模型，并利用该模型分析了在线与离线组件间的完整性与保密性信息流问题。

**💡 创新点**

首次将点对点与广播通信混合的语义模型与静态信息流控制技术结合，并揭示即使无循环构造，递归方法调用也会导致信息泄漏。

**🔧 技术方法**

形式化语义（小步语义、线程池、网络层）、信息流类型系统（Volpano 等）以及递归/线程模型。

**📊 数据集**

未使用任何数据集，本文为理论性研究。

**📈 对比分析**

未进行实验比较，讨论集中在理论安全性缺陷上。

**⚠️ 局限性**

模型复杂度高、缺乏完整的形式化验证；类型系统无法阻止递归导致的泄漏；未给出可扩展性或实现性能评估。

---

## 424. Residual Deep Reinforcement Learning-Based Computed Torque Control for a Cable-Driven Lower-Limb Rehabilitation Robot under Disturbances and Parametric Uncertainties

**arXiv ID:** 2608.26739 | [PDF](https://arxiv.org/pdf/2608.26739v1)

**作者:** Mohammad-Hossein Fakouri `[一作]` (Kharazmi University), Ali Keymasi-Khalaji `[通讯]` (Kharazmi University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在模拟环境下，对三自由度的链式电缆驱动下肢康复机器人实施了基于计算力学的计算扭矩控制（CTC）与受限深度确定性策略梯度（DDPG）相结合的残差强化学习控制框架；

**💡 创新点**

创新点在于将模型可解释的CTC作为主控，仅将受限DDPG作为补偿残差，既保留了传统逆动力学的透明度，又通过学习提升对模型不匹配与外部扰动的鲁棒性；

**🔧 技术方法**

采用了计算扭矩控制、深度确定性策略梯度（DDPG）以及相关的神经网络、奖励设计、经验回放等强化学习技术；

**📊 数据集**

使用的是基于仿真的三自由度电缆驱动下肢模型（无具体公开数据集），在不同的扰动与参数不确定性情形下进行仿真；

**📈 对比分析**

通过四种场景（基准、参数不确定、扰动、组合）与十次扰动种子、不同初始条件等扩展测试，比较CTC与CTC+残差DDPG的轨迹跟踪误差（RMS、峰值、IAE、ISE）以及关节极限和电缆张力可行性指标，结果显示残差DDPG显著降低RMS误差约41–42%，峰值误差约19%，IAE、ISE分别降低约63–67%；

**⚠️ 局限性**

局限性包括：仅在模拟环境下验证，未涉及实际电缆动力学与电机饱和；残差DDPG对电缆张力边界未直接约束；缺乏正式的稳定性证明和安全性分析；对新轨迹与初始状态的泛化仍需进一步验证。

---

## 425. Survival-Guided Length Control for Efficient Diffusion Language Models

**arXiv ID:** 2608.26374 | [PDF](https://arxiv.org/pdf/2608.26374v1)

**作者:** Ivan Kobyzev `[一作]` (Huawei), Yufei Cui `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对扩散语言模型（DLM）的文本生成任务，作者提出了一种训练无关、基于生存分析的长度预测器，用单次前向推理即可估计出每个样本的最优生成长度，从而避免在固定最大长度下做无效的后续去噪步骤。

**💡 创新点**

创新点在于将生成长度视为离散时间生存问题，通过将单步推理得到的每个位置的结束概率视为危险函数，利用生存分析的闭式期望公式得到期望长度；该方法不需要额外训练、参数或改动模型结构，即可显著加速推理。

**🔧 技术方法**

主要技术包括：扩散式语言模型（LLaDA、Dream）、离散时间生存分析（危险函数、存活函数）、基于单步推理得到的logits转换为结束概率、闭式求期望长度、AOAR（任意顺序自回归）解码与长度预测器的组合。

**📊 数据集**

使用的基准数据集包括：BBH、GSM8K、MATH、HumanEval、MBPP，所有任务均为标准的推理或代码生成评测。

**📈 对比分析**

与传统采用固定最大长度（L_max）并在所有位置完成去噪的AOAR解码相比，加入长度预测后推理速度提升约3.2×–6.6×（最长7×），而在所有评测任务中的准确率或通过率差异均不超过统计误差，基本无性能损失。

**⚠️ 局限性**

局限性包括：仅在两大规模掩码扩散LM（LLaDA、Dream）与上述5个标准任务上验证，未测试极长上下文、对话或多轮场景；方法仅关注推理计算效率，未考虑校准、鲁棒性或偏见等输出质量方面；对模型训练或架构改动不适用。

---

## 426. Surgical Video Generation From Diffusion to World Models: A Survey

**arXiv ID:** 2608.26214 | [PDF](https://arxiv.org/pdf/2608.26214v1)

**作者:** Fuxiang Huang `[一作]` (Lingnan University), Lei Zhang `[通讯]` (Chongqing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了2024-2026年间外科视频生成的研究进展，提出了从无条件生成到条件生成再到世界建模的三大范式，并对公开数据集、评价指标及主流方法进行了系统性汇总与对比。

**💡 创新点**

创新点在于：①首次以生成目标为轴构建外科视频生成的分类框架，揭示技术演进路径；②指出像素级质量与临床可行性之间的显著差距；③提出对未来评估与可信度建设的方向性建议。

**🔧 技术方法**

采用文献综述、分类整理、量化指标对比等方法；整合无条件、条件与世界建模三种生成技术（如扩散模型、文本/图像/动作条件、符号与物理图谱融合等）；并对关键数据集与评价指标进行统一整理。

**📊 数据集**

使用的公开数据集包括：Cholec80、CholecT50、CoPESD、Ophora‑160K、Cataract‑1K、CATARACTS、JIGSAWS、Kvasir‑Capsule 等。

**📈 对比分析**

通过在上述数据集上对比 FVD、FID、SSIM、IS 等指标，发现条件生成模型在视觉质量上普遍优于无条件模型，世界建模（如 SWoMo）在物理可行性与动作一致性上表现更好，但整体指标仍未满足临床实际应用的严格要求。

**⚠️ 局限性**

局限性包括：①数据来源单一、规模有限，泛化能力不足；②缺乏对物理真实性与可控性的明确建模；③评价指标主要基于视觉质量，缺乏临床可行性与可信度评估；④模型可解释性和不确定性量化不足。

---

## 427. Robust Neural Stimulation Response Modeling Through Meta-Learning and Pretraining

**arXiv ID:** 2608.26649 | [PDF](https://arxiv.org/pdf/2608.26649v1)

**作者:** Matthew J Bryan `[一作]` (University of Washington), Rajesh P N Rao `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用跨会话预训练和Meta‑学习提升神经刺激响应预测模型的鲁棒性与样本效率；

**💡 创新点**

首次证明使用MAML预训练的时序基函数模型（TBFM）能显著压缩低性能尾部、降低灾难性失败；

**🔧 技术方法**

采用Temporal Basis Function Models、线性自编码器、LoRA、MAML及其测试时适配（TTA）技术；

**📊 数据集**

40次光遗传刺激实验（S1/M1）来自两只猴子，共42–94个可用电极；

**📈 对比分析**

与单会话TBFM对比，MAML预训练模型在1k、2.5k样本下R²平均提升0.23，灾难性失败从16/40降至1/40，预测区间显著收窄；

**⚠️ 局限性**

局限性包括仅有两只动物、单一光遗传刺激、线性自编码器、TTA耗时≈13 min、未考察行为状态漂移或不同刺激模态；

---

## 428. Evaluating Language Models in Realistic Conversational Contexts

**arXiv ID:** 2608.26131 | [PDF](https://arxiv.org/pdf/2608.26131v1)

**作者:** Ilija Subasic `[一作]` (Upwork Inc.), Zhao Chen `[通讯]` (Upwork Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了UPHELD大规模、参考完整、专业撰写的多轮对话数据集，并在其上系统评估LLM在长对话中的质量。

**💡 创新点**

创新点在于：①提出专门针对人类规模对话的评估基准UPHELD；②设计多维人类标注（内容、风格、合理性）并构建混合评估器（多种自动指标的学习融合），显著提升与人工评估的相关性。

**🔧 技术方法**

技术方法包括：人类标注的三维评分体系、传统token/语义/LLM-as-judge指标的计算、以及线性回归、SVM、随机森林等机器学习模型对这些指标的融合。

**📊 数据集**

使用的数据集为UPHELD（30,000+人类撰写对话，153,578条逐轮评分）以及验证集LLM-Arena和Topical-Chat。

**📈 对比分析**

比较方法：对传统ROUGE、BERTScore、LLM-as-judge等指标与人类评分的Pearson/相关系数进行对比；结果显示单一指标相关性弱，融合模型在UPHELD上提升约30-40%，SVM在跨数据集迁移中保持高相关性。

**⚠️ 局限性**

局限性包括：人类评审间的一致性仅中等，UPHELD主要覆盖任务导向对话，可能缺乏更广泛对话多样性；LLM-as-judge在高难度或多义性情境下的可靠性仍有限。

---

## 429. RECAP-Forcing: Retaining Content Appearances for Long Video Generation

**arXiv ID:** 2608.26671 | [PDF](https://arxiv.org/pdf/2608.26671v1)

**作者:** Haiyang Xu `[一作]` (University Of California San Diego), Zhuowen Tu `[通讯]` (University Of California San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的长视频记忆机制，通过强化注意力吸收（attention sink）和光流驱动的外观新颖性记忆库，保持长时序视频中主体、场景和视觉风格的一致性。

**💡 创新点**

创新点在于将记忆索引从传统的时间最近性转为外观新颖性，统一利用开头的强化注意力吸收和随后的光流新颖性银行两种机制实现长时序一致性。

**🔧 技术方法**

采用自回归视频扩散模型、RAFT光流估计、旋转位置编码（RoPE）以及对注意力权重的强化偏置等技术。

**📊 数据集**

在VBench-Long长视频基准上评估，生成60秒（240帧）长视频。

**📈 对比分析**

与Self‑Forcing、Infinite‑Forcing、LongLive、Helios等基线以及其它训练自由方法对比，动态度量提升至约58.1/71.3，整体得分从78.8提升至79.7，优于现有训练自由方法。

**⚠️ 局限性**

局限性：依赖光流对应，可能将噪声纹理误判为新颖内容浪费记忆；对极长时间或极复杂场景的记忆效果仍有限。

---

## 430. How AI Experiences Art: Emergent Aesthetic Structure in a Self-Supervised Multimodal Embedding Space

**arXiv ID:** 2608.27121 | [PDF](https://arxiv.org/pdf/2608.27121v1)

**作者:** Corey D. C. Heath `[一作]` `[通讯]` (Independent Researcher), Corey D. C. Heath (Independent Researcher)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个自监督多模态嵌入框架，利用文本、音频、图像和视频四种模态的特征进行聚类，形成28个美学概念集群。

**💡 创新点**

通过无跨模态配对、自监督的聚类信号实现跨模态对齐，且在不需要人工标签的情况下发现与人类情感标签不同的美学结构。

**🔧 技术方法**

采用各模态的预训练模型（E5、CLAP、CLIP、V-JEPA）提取嵌入，后置256维归一化MLP，SupConLoss训练，HDBSCAN进行伪标签生成，并迭代训练循环实现深度聚类。

**📊 数据集**

基于公开媒体的弱监督数据集，包含诗歌、散文、音频、图像、视频，共计约12,010条样本，分六个情感注册（elegiac、sublime、grotesque、uncanny、idyllic、pastoral）。

**📈 对比分析**

通过计算NMI、ARI和纯度与人类标签进行比较，得到NMI=0.40、ARI=0.15、纯度=0.70，表明AI聚类更细粒度且与人类标签部分一致，最终稳定生成28个语义一致的美学概念。

**⚠️ 局限性**

人类标签仅按集合层级赋值，噪声高；数据来源未经过策划，文本偏向西方公共领域作品，导致偏见；缺乏项目级人工注释与验证。

---

## 431. Animarium: an open, reproducible pipeline for synthetic populations of Italian cities, from ISTAT sources to open data (Tech Report v1)

**arXiv ID:** 2608.27111 | [PDF](https://arxiv.org/pdf/2608.27111v1)

**作者:** Mirko Degli Esposti `[一作]` `[通讯]` (University of Bologna), Mirko Degli Esposti (University of Bologna)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了一个完全开源、可复现的流水线（Animarium），用于从 ISTAT 的公开统计表、行政区划表、地址注册、社会调查微数据等源，生成十一座意大利市镇（共计 1,814,317 人口、887,937 家庭）的合成人口。流水线分为四个“环”：① 最大熵联合模型生成基本人口属性；② 采用热板法将完整的调查回答向量捐赠给个体；③ 将个体分配到区划、单一年龄和地址；④ 根据户口规模与微数据构建家庭结构。最终输出可公开下载的合成人口文件、可视化 web 浏览器以及基于 LLM 的叙事层。

**💡 创新点**

创新点包括：① 通过“四环”机制将属性分配到不同层次，避免在联合模型中引入结构性零；② 采用最大熵分布与 PCD 技术，在大状态空间下实现高效、精确拟合；③ 将热板法与完整调查向量捐赠结合，保持跨变量相关性；④ 构建源代码与数据来源的“注册表”，在流水线中显式记录每个属性的来源、可用性与使用限制，确保可追溯与合规；⑤ 通过字节级别的再现测试与公共发布机制，形成完整的可复现与安全性证明。

**🔧 技术方法**

核心技术包括：最大熵分布求解（双对偶 + 持续对比散度），Gibbs 采样，Numba 加速稀疏矩阵运算；热板（hot‑deck）向量复制；整数分配（Largest Remainder）实现区划分配；IPF（迭代比例调整）用于国籍分层；随机数种子固定以保证可复现；数据规范化、标准化、异常检查等预处理与后处理流程。

**📊 数据集**

使用的数据集主要有：ISTAT SDMX 公开的永久人口普查表（性别、年龄、婚姻、国籍、教育、职业状态、迁移背景、父母来源等）、区划表（区/区块/社区/地区的五年年龄、性别、国籍、教育、就业等列）、ANNCSU 全国地址注册表、意大利社会生活调查（AVQ）公共微数据、各市镇开放数据门户（国籍、姓名、区域等）、2006 年 2011 年 ISTAT 统计编码表（标题树、职业-职位分布）、姓名与姓氏本地化目录、家庭组合基准等。所有源均在流水线注册表中标识并加以版本化。

**📈 对比分析**

对比方法：在发布标签下，所有 11 个市镇的合成人口文件与已归档文件逐字节比对，确认完全一致；解决时间从 0.17 秒到 182 秒不等，整体在一台工作站上完成约 33 分钟；质量评估包括：联合模型拟合误差（MRE 2.4–5.0×10⁻⁴），样本与采样底线的 z‑score（平均 0.8、sd ≈1），区划分配误差（MAE 0.72–1.57 个人），年龄–标题不一致比例（约 2.4–3.4 %），家庭结构符合度（> 95 % 符合户口规模，婚配一致性 74–98 %）。此外，注册表与官方普查数据的性别–单一年龄对齐无差异，验证数据一致性。

**⚠️ 局限性**

局限性：① AVQ 微数据需要人工申请，流水线中未硬编码；② solver 依赖未在包管理器中固定，需要手动检出特定提交；③ 仅适用于已公开的 ISTAT 源，缺乏对其他国家或数据源的直接迁移；④ 区划分配与年龄标题匹配仍基于假设（分段均匀），导致 2–3 % 的不一致；⑤ 名字与家庭属性尚未实现跨家庭一致性；⑥ 对不同硬件/编译器的浮点差异未完全消除，可能导致跨机器再现略有偏差；⑦ 该工作不包含对合成人口与真实人口的全面验证，质量评估基于内部指标。

---

## 432. pro-team at LLMs4OL 2026 Tasks Flagship and Reuse: Retrieval-Augmented Generation and Vocabulary-Constrained Filtering for Ontology Learning

**arXiv ID:** 2608.27101 | [PDF](https://arxiv.org/pdf/2608.27101v1)

**作者:** Shivam Mishra `[一作]` (Indian Institute of Information Technology Allahabad), Kuldeep Singh `[通讯]` (Indian Institute of Information Technology Allahabad)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种离线检索增强生成（RAG）框架，结合密集检索、指令调优的大语言模型和词表约束过滤，完成LLMs4OL 2026挑战中的Flagship（Task A）和Reuse（Task B）两项任务。

**💡 创新点**

创新点在于：①将检索模块与生成模型解耦，实时检索最相似示例作为动态上下文；②采用左侧截断策略保证指令完整；③为闭世界任务设计确定性词表约束过滤，显著降低hallucination；④通过多种匹配（Exact、Fuzzy、Semantic）评估，展示模型在不同约束下的表现。

**🔧 技术方法**

使用的技术包括：Qwen2.5‑14B‑Instruct（LLM）、MiniLM‑L6‑v2（句子嵌入）进行检索、指令式prompt构造、greedy解码、三阶段解析器、词表约束过滤以及基于RapidFuzz和Embedding的评估方法。

**📊 数据集**

实验数据集为官方 LLMs4OL 2026 Challenge benchmark，涵盖Task A（End‑to‑End Flagship）和Task B（Ontology Extension Reuse）两类任务。

**📈 对比分析**

评估方式为官方评测脚本，包含Exact、Fuzzy、Semantic三种匹配以及Graph Similarity（Edge F1、Neighborhood、Taxonomy）。在Task B中取得整体Semantic Graph Similarity 0.8692、Term‑Typing F1 0.9200、Taxonomy Discovery F1 0.854；在Task A中取得Semantic Graph Similarity 0.7416。非税onomic关系的F1为0，反映当前设计对该类关系的不足。

**⚠️ 局限性**

limitations：①未进行系统的ablation研究，难以量化检索、prompt与过滤各自贡献；②词表约束仅针对实体，未对谓词做显式过滤；③关键超参数（截断阈值、检索样本数）仅经验调优；④对非税onomic关系预测的能力有限。

---

## 433. Tabular Deep Learning for Algorithmic Trading: Cross-Regime Bayesian Optimisation for Equity Signal Generation

**arXiv ID:** 2608.27076 | [PDF](https://arxiv.org/pdf/2608.27076v1)

**作者:** Joshua Le Grice `[一作]` `[通讯]` (University of Exeter), Joshua Le Grice (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了跨市场 regime 的贝叶斯优化框架，对 5 种模型（Logistic、XGBoost、MLP、TabNet、FT‑Transformer）进行超参数搜索，并通过 Rank 聚合生成 Hybrid 集成，用于 2015‑2025 年约 300 只 S&P500 成分股的每日交叉预测和交易组合构造。

**💡 创新点**

创新点包括：① 将 regime 稳定性直接作为贝叶斯优化目标约束，实现跨 regime 的超参数选择；② 证明 TabNet 与 XGBoost 的秩聚合能产生统计显著的 alpha；③ 明确 Tabular Deep Learning 在此跨期交易场景中并未优于梯度提升树。

**🔧 技术方法**

使用的技术有：Optuna TPE 贝叶斯优化、三折扩展窗口交叉验证、RobustScaler 归一化、日内交易信号生成、Rank 聚合、SHAP 解释、Gaussian 噪声鲁棒性测试、KS/Friedman/Wilcoxon/Probabilistic Sharpe Ratio 等统计检验。

**📊 数据集**

数据集为 2015‑2025 年的每日观察，约 300 只 S&P500 成分股，特征包含价格、技术指标、公司基本面、FRED 宏观经济、Bloomberg 新闻情绪、Google Trends 搜索指数。

**📈 对比分析**

对 5 个单模型和 Hybrid 集成在 2025 年 OOS 进行交易回测，Hybrid 年化收益 51.26%，Sharpe 2.44，CAPM alpha 0.423（p=0.011）；单模型 XGBoost 33.61%，FT‑Transformer 仅 3.08%。统计检验显示 Hybrid 在 alpha 和 PSR 上显著优于基准，且对 Gaussian 噪声和季度 regime 变化表现鲁棒。

**⚠️ 局限性**

局限性：单年度 OOS 评估、S&P500 成分股幸存偏差、Bloomberg 数据许可限制、未计短售借券费、深度模型训练时间长、SHAP 近似误差、未与单 regime 调优做对照、未处理结构性缺失或高频噪声等。

---

## 434. SpatialCrafter: Single Image World Modeling with Generative 3D Proxies

**arXiv ID:** 2608.27073 | [PDF](https://arxiv.org/pdf/2608.27073v1)

**作者:** Chuan Fang `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个两阶段框架，先从单张图像生成全局3D代理，再用视频扩散模型细化为高质量可探索的RGB-D视频。

**💡 创新点**

首次使用点锚稀疏结构(PaSS)流匹配实现全局3D代理的几何对齐，并通过并行几何注入与代理感知破坏(PAC)提升视频细化鲁棒性；并构建115k场景的大规模混合数据集。

**🔧 技术方法**

三维潜在扩散模型(TRELLIS)、稀疏点云对齐(PaSS)、并行几何注入、代理感知破坏、3D Gaussian splatting、视频Diffusion DiT、LoRA微调。

**📊 数据集**

结合SpatialGen、RealEstate10K、DL3DV三大源构建115k场景的Hybrid数据集，包含精确几何注释的RGB-D视频。

**📈 对比分析**

与2D内存与重构3D代理基线（如DFoT、GeometryForcing、ViewCrafter、GEN3C、Voyager）在Synthetic SpatialGen-Video、RealEstate10K和DL3DV上对比，取得最低FVD、最高PSNR、最佳RPE/RVE，显著抑制漂移和幻觉。

**⚠️ 局限性**

对极端遮挡或极少视角下的全局代理仍可能产生误差，且训练依赖大量GPU，模型对大尺度长序列仍存在推理速度瓶颈。

---

## 435. Beyond Classification: Task-Dependent Learnability under Privacy-Motivated Image Transformations

**arXiv ID:** 2608.27066 | [PDF](https://arxiv.org/pdf/2608.27066v1)

**作者:** Leon Ranke `[一作]` (Fraunhofer Institute of Optronics, System Technologies and Image Exploitation), Jürgen Beyerer `[通讯]` (Fraunhofer Institute of Optronics, System Technologies and Image Exploitation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种计算友好的多任务协议，通过轻量级代理任务评估隐私增强图像变换（PET）的任务相关可学习性。

**💡 创新点**

创新点在于将分类、相对角度预测和拼图解谜三种代理任务组合成诊断性评估框架，揭示PET对语义可分性、几何一致性和空间兼容性的不同影响，并发现固定密钥变换会产生可被利用的捷径线索。

**🔧 技术方法**

使用的技术包括可逆/不可逆图像模糊、局部无序图像（LOI）、块级置换/旋转/翻转、像素置换、颜色置换、负正变换以及学习型图像加密方案（Tanaka、E‑Tanaka、EtC）等；评估时采用ResNet‑34分类器、CNN回归网络和基于Transformer的拼图解谜模型。

**📊 数据集**

实验基准为动物物种分类数据集（15 类，图像尺寸 416×416 或 384×384），并在多尺度、不同密钥设置下生成变换样本。

**📈 对比分析**

与单一分类指标相比，该方法显示 PET 在保持语义可分性时，仍可能严重削弱几何与空间相关任务的性能；实验结果表明不同任务对同一变换的敏感度差异显著，且固定密钥情况下拼图任务能通过捷径获得高分，提示仅靠单一指标评估不足。

**⚠️ 局限性**

局限性包括：代理任务虽然轻量但无法完全覆盖所有实际下游任务；评估仅针对从零开始的模型且未考虑迁移学习或关键字信息的利用；此外，仅使用 LPIPS 等感知指标来度量图像域隐匿性，可能忽略其他安全攻击向量。

---

## 436. Soft Active Electromyography Interface for Machine Learning-Enabled Silent Speech Recognition

**arXiv ID:** 2608.27048 | [PDF](https://arxiv.org/pdf/2608.27048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 437. Video-OPSD: Exploiting Privileged Visual Evidence for On-Policy Self-Distillation in Video Large Language Models

**arXiv ID:** 2608.27065 | [PDF](https://arxiv.org/pdf/2608.27065v1)

**作者:** Ziyue Wang `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Video-OPSD 框架，通过利用视频中的证据帧构建证据驱动的自教师，并通过证据引导的 token 重要性加权进行知识蒸馏，从而在视频推理任务中提升性能。

**💡 创新点**

创新点在于：①只用证据帧构造教师，去除无关视觉信息；②依据教师对证据的注意力动态加权 token 蒸馏，实现更精准的视觉引导。

**🔧 技术方法**

使用技术包括：自监督蒸馏 (OPSD)、Transformer注意力聚合、JSD 目标、Top-K 词表截断、教师正确性门控、EMA 等。

**📊 数据集**

采用 STAR、Video-Holmes 训练集以及 Video-MMMU、Video-MME、TempCompass、WorldSense 等五个视频理解/推理基准。

**📈 对比分析**

与监督微调、GRPO、标准 OPSD 等方法比较，Video-OPSD 在三种 Qwen-VL 基础模型上平均提升 2–4 分，并在 2.2 小时内达到与 GRPO 相当的性能，显著降低训练成本。

**⚠️ 局限性**

局限性包括：仍依赖标注的证据窗口；教师保持冻结可能限制进一步提升；在极长视频或多模态场景中的扩展尚未验证。

---

## 438. Anatomy-Guided Foundation Model Adaptation with Within-Case Prototype Supervision for Standard Plane Detection in Fetal Ultrasound Blind Sweeps

**arXiv ID:** 2608.27051 | [PDF](https://arxiv.org/pdf/2608.27051v1)

**作者:** Yuzhe Zhao `[一作]` `[通讯]` (University of Leeds), Yuzhe Zhao (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种轻量级的序列级标准平面检测框架，在低成本胎儿超声盲扫数据上使用冻结的BiomedCLIP编码器，并通过解剖学加权池化、同病例原型损失、层级细化回收器以及稳定性-边界混合头，实现对腹部标准平面的高精度识别。

**💡 创新点**

核心创新包括：① 用单机构(nnU-Net)生成的腹部空间先验对冻结FM的patch token进行解剖学加权池化；② 引入同病例原型损失，将同一扫的正样本嵌入聚集到一个病例特定的原型上；③ 设计三阶段粗细化推理管线（帧→段→扫级拒绝器）提升预测鲁棒性；④ 在框架中加入稳定性与边界两头联合学习，抑制边界误检。

**🔧 技术方法**

主要技术手段为冻结的BiomedCLIP ViT-B/16图像编码器、单机构(nnU-Net)腹部分割、4层双向时间Transformer（BiTT）作为时间头、解剖学加权池化、同病例原型损失、PRS后处理与段级逻辑回归拒绝器、稳定性/边界辅助头。

**📊 数据集**

使用ACOUSLIC-AI挑战公开数据集（300条盲扫，约840帧/扫，共约252k帧），在210/45/45的病例级划分上训练与评估。

**📈 对比分析**

与官方基线、单帧CNN、各种单帧与视频基线（ActionFormer、TriDet）以及外部FM基线（FetalCLIP+PRS）进行对比，最终模型在测试集上取得F1=67.72，超过FetalCLIP+PRS（54.52）+13.20点，超过TriDet+PRS（51.96）+15.76点，且在精度-召回平衡上优于所有基线。

**⚠️ 局限性**

局限性包括：① 采用7×7的粗粒度腹部先验，可能导致边界不精确；② 原型损失与解剖先验的协同提升未通过统计检验，需要更大样本验证；③ 对极短（<5帧）或非典型标准平面召回不足；④ 缺乏针对无效扫的置信度/拒绝机制，无法区分缺失与未检测。

---

## 439. Neighborhood Watch: Privacy Risks in Seeded Local Combination Synthetic Data

**arXiv ID:** 2608.27037 | [PDF](https://arxiv.org/pdf/2608.27037v1)

**作者:** Hadrien Lautraite `[一作]` (Université du Québec à Montréal), Sébastien Gambs `[通讯]` (Université du Québec à Montréal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了三种本地组合合成数据方法（SMOTE、Avatar、Simulant）的隐私风险，通过成员推断、链接和重构攻击进行实验。

**💡 创新点**

提出了增强的集成成员推断攻击、基于似然的链接攻击以及针对Avatar k=2的重构攻击，并证明传统距离指标无法准确评估隐私泄露。

**🔧 技术方法**

使用集成学习（GBM）构建MIA、似然估计（高斯分布/KDE）进行链接攻击、线性几何重构技术，并与已有基线攻击对比。

**📊 数据集**

使用了三个公开数据集：WBC（Wisconsin Breast Cancer Dataset）、AIDS、California Housing。

**📈 对比分析**

与基线（随机猜测、最邻近距离、Lebrun等）比较，结果显示SMOTE和Avatar在所有攻击中表现最差，MIA、链接和重构成功率均远高于随机，Simulant虽相对安全但仍存在泄露；攻击成功率随k增大而下降，但对SMOTE/Avatar仍显著高于基线。

**⚠️ 局限性**

实验仅在无盒（no-box）假设下进行，样本规模有限；类别特征的似然估计效果不佳；未给出真正的DP实现，需进一步改进以提升安全性。

---

## 440. Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?

**arXiv ID:** 2608.27115 | [PDF](https://arxiv.org/pdf/2608.27115v1)

**作者:** Ej Zhou `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文研究了独立训练的单语模型是否能在不共享参数或对齐信号的情况下实现跨语种对齐。

**💡 创新点**

创新点在于证明单语模型仅凭自身语料即可产生可对齐的表征，并且通过一次正交旋转即可在不同语言之间实现功能迁移。

**🔧 技术方法**

使用的技术包括Centered Kernel Alignment (CKA)、Procrustes正交映射、仿射与MLP映射，以及跨模型激活补丁（activation patching）评估因果转移。

**📊 数据集**

实验数据集涵盖Goldfish系列单语模型、来自不同实验室的≈1B参数单语模型，以及FLORES‑200、Tatoeba、OPUS、BouQUET等平行语料。

**📈 对比分析**

与随机（shuffled）对齐基线相比，正交旋转在检索指标上达88.7%准确率；在事实填空任务中补丁成功率可达66–85%，远高于随机对齐且接近同模型补丁的基准。

**⚠️ 局限性**

局限性包括：语言覆盖偏向高资源语言，低资源语言的因果评估未完成；模型可能包含极小程度的语言混杂；对齐方法依赖少量平行数据；补丁评估仅覆盖有限的事实关系和单词位置。

---

## 441. LAAF: A Layered Accountability Architecture Framework for LLM Applications

**arXiv ID:** 2608.27102 | [PDF](https://arxiv.org/pdf/2608.27102v1)

**作者:** Prachi Chaturvedi `[一作]` (Bennett University), Pierre Dantas `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM应用的责任链进行系统性综述，并提出了四层责任架构LAAF，用以整合技术控制、人类监督、组织治理与文档追溯等机制；

**💡 创新点**

创新点在于将责任拆解为五个维度并映射到四个层次，形成跨层交叉属性（可追溯性、角色清晰、持续监测），同时将该架构与欧盟AI法、NIST RMF、ISO/IEC 42001等绑定；

**🔧 技术方法**

使用系统性综述（PRISMA）、主题分析、成熟度评估、框架映射以及对OWASP LLM Top 10安全类别的对齐等方法；

**📊 数据集**

利用122篇同行评审论文、12份监管文件以及8个公开基准（如TruthfulQA、HaluEval等）进行数据采集与评估；

**📈 对比分析**

通过对比四大监管框架和十四个现有治理框架，量化LAAF在层次覆盖、监管映射、角色清晰度、持续监测等指标上的匹配度，但未给出实验性能数据，仅呈现定性匹配度；

**⚠️ 局限性**

局限性包括：仅覆盖2022–2026年英文文献；缺乏对技术/监管机制的实证验证；架构对代理/工具调用场景适用性有限；缺乏统一度量指标与实际部署评估。

---

## 442. Automated 2D and 3D Segmentation of AMD and DME Lesions in OCT

**arXiv ID:** 2608.27095 | [PDF](https://arxiv.org/pdf/2608.27095v1)

**作者:** Lucia Sundberg `[一作]` (Technical University of Munich), M. Ali Nasseri `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文在OCT影像上构建并系统消融了四条AMD和DME病变分割管线（2D/3D各一条），实现了自动化病变分割。

**💡 创新点**

创新点包括：① 采用全体积、校准意识的评估标准，避免了传统切片级评估的偏差；② 通过系统的机制消融与集成搜索，证明了模型集成是提升性能的最大杠杆；③ 设计了一套无病变标注时可用的代理指标（AUROC、CST相关性、纵向一致性）实现跨域验证。

**🔧 技术方法**

技术手段主要为U‑Net及其变体（UNet++, MONAI U‑Net、Attention‑UNet）在2D/3D空间的应用；采用Tversky、Dice‑BCE、边界距离损失；正样本过采样、长度感知采样、切片级和体素级增强；滑动窗口推理、Gaussian加权融合；多尺度后处理与可判别过滤；多模型权重平均与坐标上升搜索的集成策略。

**📊 数据集**

使用Huang等人的公开OCT数据集（AMD 62卷、DME 42卷，按80/20划分为训练/验证）进行训练和内部评估；外部验证采用OLIVES临床队列（TREX‑DME子集），无病变级别标注，仅使用代理指标评估。

**📈 对比分析**

内部验证表现：Dice 0.76–0.82，体积/表面积校准相关系数均≥0.97；集成提升显著；外部验证表现：DME 3D在IRF上AUROC 0.73、CST相关系数0.57、方向一致率83%；DME 2D在IRF上AUROC 0.91、CST相关系数0.19、方向一致率82%。

**⚠️ 局限性**

局限性包括：未留出独立测试集，所有评估均基于同一验证集；训练数据仅来自单一扫描仪/协议；外部验证仅使用代理指标，缺乏真实病变标注；AMD 3D性能相对较弱；PED分析仅基于单例样本，缺乏统计意义。

---

## 443. An Empirical Evaluation of Using Large Language Models for Automated Model-Based Test Generation

**arXiv ID:** 2608.27094 | [PDF](https://arxiv.org/pdf/2608.27094v1)

**作者:** Hafize Sanli `[一作]` (Mugla Sitki Kocman University), Cihat Cetinkaya `[通讯]` (Mugla Sitki Kocman University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

评估大型语言模型在自动生成模型基础测试路径中的效果，并与 GraphWalker 的基线算法进行对比。

**💡 创新点**

提出了 LLM4MBT 流水线，首次将 LLM 直接用于生成符合 GraphWalker 模型的测试路径，并展示其能在保持覆盖率的同时显著缩短测试步骤。

**🔧 技术方法**

使用了 GPT‑5.1/5.2、Claude Opus 4.5/Claude Sonnet 4.5、Gemini 2.5 Pro 等五种最先进 LLM，并结合 Prompt 工程、路径验证回退机制和覆盖度分析等技术。

**📊 数据集**

实验基于四个 GraphWalker 模型：交通灯控制器（TLC）、RISC‑V、Parabank 与 Testinium。

**📈 对比分析**

通过边/顶点覆盖率与测试步骤数进行评估；LLM4MBT 的平均覆盖率约为 96.3%/100%，测试步骤数约 300 步，较 GraphWalker 的 524/6916 步减少 3–23 倍，显示 LLM 在效率上优于传统随机/快速随机策略。

**⚠️ 局限性**

结果受 LLM 输出可变性、提示设计影响；仅评估覆盖率，未检验缺陷发现；实验规模局限于四个模型，未验证更大工业系统；LLM 输出的可重复性需多轮种子验证。

---

## 444. Active sensing to characterize the heterogeneity of plant stress

**arXiv ID:** 2608.27088 | [PDF](https://arxiv.org/pdf/2608.27088v1)

**作者:** Ayman Laaroussi `[一作]` (Sony Computer Science Laboratories), David Colliaux `[通讯]` (Sony Computer Science Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一个低成本自主机器人平台，用于对植物叶片进行定向叶绿素荧光测量，并通过3D重建与运动规划实现精确定位。

**💡 创新点**

将主动生理感知（叶绿素荧光）与机器人感知、几何推理和机械操作紧密耦合，实现了空间分辨率高、可重复且不依赖人工的植物表型测量。

**🔧 技术方法**

使用结构光相机与COLMAP进行多视角图像采集与结构光重建，结合空间刻画、HDBSCAN聚类、立体几何分析，配合CNC台座、平面姿态舵机和Raspberry Pi控制的抓手。

**📊 数据集**

采集了多视角RGB图像（约80帧）并利用自制的叶绿素荧光传感器在两株Pilea peperomioides上记录光照响应曲线，形成实验数据集。

**📈 对比分析**

通过光照激励实验（暗适应 vs 高光照）对比荧光时间序列，利用自定义光学刺激协议评估传感器的光照应激指标，证明平台能在视觉上区分暗适应与光适应叶片。

**⚠️ 局限性**

局限在于基于空间刻画的重建无法处理强内凹或自遮挡叶片，重建时间较长，且仅在相对简单几何的实验植株上验证；未来需引入神经网络重建和移动机器人以提升通用性与速度。

---

## 445. ITL: Interpretable Document Alignment with Structured Reference Frameworks

**arXiv ID:** 2608.27031 | [PDF](https://arxiv.org/pdf/2608.27031v1)

**作者:** Raúl Giráldez `[一作]` (Pablo de Olavide University), Jesús S. Aguilar--Ruiz `[通讯]` (Pablo de Olavide University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Intelligent Target Locator（ITL）方法，利用概念层面的 affinity 指标对目标文档与结构化参考文档（SRD）之间的对齐进行可解释且可追溯的定量评估。

**💡 创新点**

创新点在于：①直接从 SRD 构建概念特定术语概况；②将词项的重要性（特异性 + 区分度）与词频结合，得到词级 affinity；③生成文本单元–概念矩阵并提供相对归一化的相对 affinity，兼具可解释性与跨语言通用性。

**🔧 技术方法**

使用的技术包括：分词、词性标注、停用词过滤、词干提取、n-gram 与共现词提取、熵基区分度计算、特异性权重、加权频次求和、矩阵操作与聚合算子（均值/最大/和）。

**📊 数据集**

数据集：以联合国2030 年可持续发展目标（SDG）的官方声明作为 SRD 与内部一致性评估对象，验证 17 个目标的概念分辨能力。

**📈 对比分析**

评估方式：计算 17×17 的 affinity 矩阵，发现对角线值（自我匹配）明显高于非对角线平均值（两阶数量级差距），相对 affinity 近 1 表示与目标概念描述高度一致；目前未与其他现有方法进行直接对比，需进一步实验验证。

**⚠️ 局限性**

局限性：①仅在 SRD 与同源文本的内部一致性实验中验证，未评估对外部文档或不同语言、领域的泛化能力；②聚合策略对结果有影响；③依赖人工分段与语言处理资源，可能受工具性能限制。

---

## 446. Direct or Mediated? Task-Dependent Audio Information Routing in Large Audio Language Models

**arXiv ID:** 2608.27026 | [PDF](https://arxiv.org/pdf/2608.27026v1)

**作者:** Yizhou Zhang `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型音频语言模型在两段音频拼接输入下的鲁棒性，比较语音识别（ASR）与音频问答（AQA）两任务的表现。

**💡 创新点**

揭示任务依赖的音频信息路由差异，并证明在拼接场景下 AQA 对信息利用瓶颈更为敏感。

**🔧 技术方法**

采用层级注意力消除（attention knockout）和层级线性探针（layer-wise probing）两种机制分析解码器内部信息流。

**📊 数据集**

使用 SpeechCommands、GTZAN、ESC-50 三个音频分类数据集和 LibriSpeech 语音识别数据集进行评估。

**📈 对比分析**

对比四个主流 LALM（Audio-Flamingo-Next、MiMo-Audio、Step-Audio-R1、Qwen3-Omni）的单段与拼接输入性能：ASR 的 WER 仅升高 0.1–1.3% 而 AQA 的多选准确率在 35–67% 之间大幅下降。

**⚠️ 局限性**

局限性在于注意力消除仅能间接评估正确案例的路径重要性，且实验仅限于两段拼接场景，未涵盖更复杂的多段、噪声或重叠音频情况。

---

## 447. Product Structure Meets Track Layouts

**arXiv ID:** 2608.27096 | [PDF](https://arxiv.org/pdf/2608.27096v1)

**作者:** Michael A. Bekos `[一作]` (University of Ioannina), Michael Kaufmann `[通讯]` (University of Tübingen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通用算法，利用图的强积结构构造轨道布局，从而给出多类图的轨道数上界；

**💡 创新点**

创新点在于将强积结构与轨道布局结合，得到一个简单直观的构造方法，能够直接将已知图类的轨道布局转换为其强积图的轨道布局；

**🔧 技术方法**

核心技术是利用路径幂的色数（χ(P^2h)=2h+1）与复制轨道布局，再按路径索引和轨道顺序进行线性合并，算法实现仅需链表操作；

**📊 数据集**

论文未使用具体数据集，而是通过理论证明给出轨道数上界；

**📈 对比分析**

与现有结果对比，算法在平面图上达到225轨道的最佳上界，并在1-可平面、最优2-可平面、genus‑k、k‑planar、k‑framed、k‑map、k‑string等图类给出新的轨道数上界；

**⚠️ 局限性**

局限性包括：仍需依赖已知的强积结构分解；对更高阶树宽或更复杂图类的轨道数提升仍不理想；并未给出实验验证，仅为理论上界。

---

## 448. Cone Extended Rayleigh Quotients for Directed Graph Learning: Minimax Spectral Certificates, Sensitivity, and Adaptive Control

**arXiv ID:** 2608.27122 | [PDF](https://arxiv.org/pdf/2608.27122v1)

**作者:** Yavdat Sh. Il'yasov `[一作]` (Institute of Mathematics with Computing Centre, Ufa Federal Research Centre of Russian Academy of Sciences), Nur F. Valeev `[通讯]` (Institute of Mathematics with Computing Centre, Ufa Federal Research Centre of Russian Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了一种针对可训练有向传播算子、基于双侧锥Rayleigh框架的谱级别认证与自适应谱控制方法。

**💡 创新点**

创新点在于将双侧锥Rayleigh理论直接应用于非对称算子，提出可微软化上下界并通过右–左特征向量实现一阶敏感性控制与自适应改造。

**🔧 技术方法**

主要技术包括锥极小极大Rayleigh变分、软最小/最大平滑化、梯度可微的谱约束、右–左敏感度分析与基于Frobenius/L1预算的图结构干预。

**📊 数据集**

实验使用了合成的有向Stochastic Block Model、Cora引文网络以及相关的训练/验证/测试子集。

**📈 对比分析**

与无谱约束基线相比，模型在保持分类准确率的同时成功认证谱上限；在Cora上自适应干预可将谱级别降低约21.5%且不降低准确率；对比对称化实验表明边缘方向信息被消除时准确率显著下降。

**⚠️ 局限性**

局限性在于仅控制锥相关的单一谱级别，无法直接约束强非正交算子的瞬态放大、伪谱增长或完整的算子范数等动态行为。

---

## 449. Real-Time Reconstruction of Markov Sources over MPR Channels

**arXiv ID:** 2608.27116 | [PDF](https://arxiv.org/pdf/2608.27116v1)

**作者:** Pansee S. Elessawy `[一作]` (Linköping University), Nikolaos Pappas `[通讯]` (Linköping University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在多包接收（MPR）无线通道上，对两源二值马尔可夫过程进行实时重构与远程执行的采样与调度问题。

**💡 创新点**

创新点包括：① 推导出闭式实时重构误差与执行误差与有效更新概率的关系；② 在独立随机化策略下通过可达更新率区域几何证明可把非凸优化简化为有限个一维边界搜索；③ 设计协调时间共享基准并证明最优策略仅需在两种极端模式间时间共享；④ 定量分析独立随机化与协调方案的性能差距。

**🔧 技术方法**

主要技术方法包括：马尔可夫链稳态分析、有效更新概率的闭式表达、几何与凸分析（Pareto 前沿、总成功上界）、边界分枝的多维最优解搜索以及线性规划求解协调方案。

**📊 数据集**

论文使用的是人工合成的参数集合（如状态转移概率、MPR 成功概率和采样预算），并通过数值仿真验证结论。

**📈 对比分析**

通过与冲突、捕获、MPR 三种物理层模型下的多种策略（均匀独立、优化 TDMA、优化独立 MPR、协调 MPR）进行对比，结果表明：在 MPR 能力足够可靠且采样预算较大时，MPR 与协调时间共享可明显降低重构误差；在冲突或捕获环境下，MPR 并不提升性能。

**⚠️ 局限性**

主要局限包括：只考虑两源二值马尔可夫模型；不考虑感知与更新的状态依赖性；协作基准仅为理论上最优的时间共享方案，实际实现难度未讨论；对更复杂多源或多跳网络的推广仍需研究。

---

## 450. DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali

**arXiv ID:** 2608.27110 | [PDF](https://arxiv.org/pdf/2608.27110v1)

**作者:** Anik Saha `[一作]` (Bangladesh University of Engineering and Technology), Rifat Shahriyar `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了DocTalkBN，一个包含557.63小时孟加拉语音频与文本的多模态专家–患者对话数据集。

**💡 创新点**

首次公开低资源语言真实医患对话集合，并从中提炼出三大下游任务（临床分诊、建议安全评估、医学命名实体识别）。

**🔧 技术方法**

通过大语言模型（Gemini、GPT‑4o、Qwen等）与人工校验相结合完成数据标注与任务构建。

**📊 数据集**

使用孟加拉国全国广播医药访谈节目的视频数据，得到1.7M词汇、26个医学专科的对话样本。

**📈 对比分析**

在零/少样本LLM和编码器微调下评估三任务，药物安全评估达到0.976（宏F1），医学NER最高0.743，临床分诊最难仅0.463，显示不同任务对模型的挑战程度不同。

**⚠️ 局限性**

局限在于仅评估文本，未充分利用音频信息；对话推理仍难，需进一步研究更深层医学推理、工具使用与多模态集成。

---

## 451. Pass the Bucket: Efficient, Robust, Local Load Balancing for Teams of Heterogeneous Robots

**arXiv ID:** 2608.27085 | [PDF](https://arxiv.org/pdf/2608.27085v1)

**作者:** Tobias Wallner `[一作]` (Technische Universitaet Braunschweig), Sándor P. Fekete `[通讯]` (L3S Research Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一维线段上研究异速机器人群的去中心化自组织任务共享，采用桶式军团（bucket brigade）机制实现负载平衡，并引入“令牌”抑制实现收敛；

**💡 创新点**

提出仅靠碰撞检测和局部令牌调节即可在无通信、无中央控制的条件下实现异速机器人自适应空间分区，并给出事件驱动的保守性分析与抑制机制；

**🔧 技术方法**

使用事件驱动仿真、线性代数矩阵分析（包括保守性与等变量）、令牌抑制的离散动力学、Monte Carlo 统计与累积分布曲线；

**📊 数据集**

无公开数据集，全部使用随机生成的机器人起始位置、方向和速度（区间[0.5,10]）进行仿真；

**📈 对比分析**

与无令牌、单向令牌、全向令牌三种配置对比，通过累计分布函数评估收敛时间；实验表明令牌可显著加速收敛，最优减速因子随机器人数递减；

**⚠️ 局限性**

理论上未完整证明有令牌系统的谱半径<1，实验仅限小规模机器人；模型假设点机器人、瞬时速度变换，忽略加速度、尺寸、定位误差等实际物理限制；

---

## 452. Research Design Tracking and Assessment for the Social Sciences

**arXiv ID:** 2608.27049 | [PDF](https://arxiv.org/pdf/2608.27049v1)

**作者:** Marco Rovera `[一作]` (Fondazione Bruno Kessler), Jessica Gagete-Miranda `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了自动化研究设计追踪与评估（ARDTrA）任务，构建并标注了140篇社会科学论文的数据集；

**💡 创新点**

创新点在于首次将因果研究设计识别与评估任务与检索增强生成（RAG）多轮对话框架结合，系统性比较检索策略、嵌入模型与LLM的表现；

**🔧 技术方法**

使用了BM25、密集检索、层次自合并检索、命题主题检索四种检索策略，配合Llama、Qwen、GPT等多款LLM，以及多种文本嵌入模型；

**📊 数据集**

使用了来自应用经济学、社会学、政治学等领域的140篇论文，涵盖六大因果研究设计，专家手工标注了研究设计识别与评估的答案；

**📈 对比分析**

通过对四种检索策略、六种嵌入模型、四款LLM的全组合实验，采用每个答案选项的精确率、召回率和F1进行评估；结果显示BM25在大多数配置下最稳健，检索片段长度是性能的主要决定因素；整体F1最高约为0.66，评估子任务的表现相对较低（≈0.4–0.6）；

**⚠️ 局限性**

局限性在于数据集规模有限（仅140篇英文论文），难以对每种研究设计进行可靠估计，且不包含非英文文献，限制了实验结果的普适性。

---

## 453. Riemann-1.0: An Embodied World Action Model for Physical AI

**arXiv ID:** 2608.27033 | [PDF](https://arxiv.org/pdf/2608.27033v1)

**作者:** Haofeng Sun `[一作]`, Yangguang Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个完全因果自回归的世界动作模型Riemann-1.0，能够统一多视角视觉观测、机器人状态和具身特定动作，既可作为可执行的机器人策略，也可用作动作条件的视觉世界模拟器。

**💡 创新点**

创新点在于：①首次提出完全因果的动作先行视频生成范式，保证在线执行与离线模拟的因果一致性；②提出分阶段的Progressive Embodied Pretraining，将无标签人类视频、手持抓取演示和机器人轨迹通过Latent Action Model、3D手部重建等技术无缝统一；③构建统一的具身数据基础设施，实现跨视角、跨机器人、跨动作空间的数据对齐与多模态平衡。

**🔧 技术方法**

核心技术包括Transformer全局注意力、流匹配（flow matching）目标、VAE视觉潜在表示、Latent Action Model用于伪动作注释、3D手部重建与相机姿态估计、Embodiment-specific 状态/动作投影、Teacher-Forcing与自回归推理的结构化注意力掩码。

**📊 数据集**

使用规模达200k+小时的多源数据集：200k+小时的视角化人类视频、12k+小时的手持抓取/可穿戴演示以及20k+小时的异构机器人轨迹，并在RoboTwin2.0、RoboCasa365、LIBERO等公开模拟基准上进行评测。

**📈 对比分析**

与DreamZero、LingBot-VLA、π_0.5、LingBot-VA、G0.5、ABot-M0.5等基线相比，Riemann-1.0在实地长周期操纵任务中SR达85%，PSR 94.4%；在RoboTwin2.0、RoboCasa365、LIBERO上分别取得94.3%、62.6%、99.0%的最高成功率，明显优于现有最佳方法。

**⚠️ 局限性**

局限性在于仍需海量标注和计算资源，且在极高频率的低层控制或与未见机器人外观/动力学差异较大的跨域场景中，模型的泛化与实时性尚待进一步验证。

---

## 454. FaulT-Bench: Towards Benchmarking Network Troubleshooting LLM Agents under Unreliable User Tickets

**arXiv ID:** 2608.27021 | [PDF](https://arxiv.org/pdf/2608.27021v1)

**作者:** Kuan-Hao Tseng `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了FaulT-Bench基准，用于评估 LLM 代理在真实网络故障排除工单（包括错误、误导性和真故障票据）上的表现。

**💡 创新点**

创新点在于：① 引入真实用户写的故障工单，包括 72 个错误前提票据并重写为 5 种报错语气；② 在 Kathará 网络模拟器中自动化部署、注入、验证网络状态；③ 用 LLM 判定器对诊断结果、修复建议和推理过程进行三维评分。

**🔧 技术方法**

技术手段：Kathará 模拟器、NIKA 工具接口（22 个诊断工具）、三种 LLM 代理（SADE、ReAct、Claude Code），以及基于 LLM 的评判器和评分框架。

**📊 数据集**

数据集：200 个排错场景，覆盖 8 个网络拓扑；场景分为 80 正确故障、72 错误前提、24 错误设备、24 错误原因；并为 72 个错误前提生成 360 个重写票据（5 种报错语气）。

**📈 对比分析**

比较方法：对每个代理在 3 组实验（核心基准、报错语气重写）下执行 1,800+ 次运行，使用 Outcome、Fix、Reasoning 三分数并记录超时；结果显示：在真故障场景下表现接近满分；在错误前提场景下误诊率高，且不同代理在错误票据和报错语气上的鲁棒性差异显著。

**⚠️ 局限性**

局限性：票据人为编写缺乏真实工单；评判器仅为单一 LLM，可能带来偏差；仅测试三种代理，未覆盖更广泛的设计空间；未评估更复杂的网络状态和更长的交互轮次。

---

## 455. ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis

**arXiv ID:** 2608.27017 | [PDF](https://arxiv.org/pdf/2608.27017v1)

**作者:** Chengsong You `[一作]` (East China Normal University), Nan Du `[通讯]` (Matter Innovation Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

ProRetrieval训练小型语言模型，作为检索编排器，通过生成可执行的混合DSL程序来联合SQL过滤和文本、图像向量检索，实现多模态检索。

**💡 创新点**

创新点在于：①将检索动作空间转化为可执行的SQL+向量检索DSL；②提出四项层次化奖励用于程序生成的强化学习；③构建两套全新混合检索基准，展示小模型可超越大模型。

**🔧 技术方法**

使用技术包括Qwen3-4B/8B模型、GRPO/DAPO强化学习、四项层次奖励、预训练向量检索引擎、SQL执行引擎、自动化基准构建脚本。

**📊 数据集**

使用的数据集为Amazon ESCI+Reviews（含结构字段、文本、图片）以及Enron邮件（含结构字段与文本）。

**📈 对比分析**

与纯检索、LLM增强、结构化查询、KG检索等四类基线以及GPT‑5.5、Claude Opus等商业LLM进行对比，Hit@1在e‑commerce为80.9%、在email为90.9%，显著优于所有基线和商业模型，RL提升带来约12–13个百分点。

**⚠️ 局限性**

局限性包括：①依赖固定schema，难以迁移至无schema或动态演化数据；②逻辑深度限制在两层嵌套；③仅在两域评估，缺乏跨领域验证；④数据规模较小，未验证百万级规模下的性能与延迟；⑤单轮检索，未实现交互式迭代；⑥受限于底层嵌入模型的对齐能力；⑦未报告随机种子方差；⑧基准构建过程可能引入语言偏差；⑨缺少检索失败时的自动回退机制。

---

## 456. Quadratic Complexity of Voronoi Diagrams in $\mathbb{R}^3$ for Lines in a Single Ruling of a Regulus

**arXiv ID:** 2608.27114 | [PDF](https://arxiv.org/pdf/2608.27114v1)

**作者:** Eunku Park `[一作]` `[通讯]` (DGIST), Eunku Park (DGIST)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究三维空间中所有线都属于同一双曲抛物面或一条双曲面中的同一条规的线族的最近和最远Voronoi图的组合复杂度，并给出了上界Θ(n²)与下界Θ(n²)的匹配结果。

**💡 创新点**

创新点在于将线族映射到Plücker空间的平面共线圆，将球面接触条件化简为二次方程，再限制到该圆得到四次多项式（contact quartic），利用符号交替将可能支持四元组从O(n⁴)压缩到O(n²)，从而实现最坏情况Θ(n²)的上界，并给出了O(n²)时间的枚举算法。

**🔧 技术方法**

主要技术包括Plücker坐标与Klein四面体的几何表示、规则线族的参数化、球面接触条件的四次多项式化简、Bézout定理与逆函数定理的代数分析、实数RAM模型下的常数阶代数运算与符号判定、以及几何约束下的正则性与转角性检验。

**📊 数据集**

论文没有使用真实实验数据集，而是通过理论构造在固定非旋转一张一面双曲面上一条规的线族（如x²/4 + y² – z² = 1）中构造出具有Ω(n²)个正则最近Voronoi顶点的实例。

**📈 对比分析**

与先前Aronov等人关于任意线族的Ω(n²)与O(n³⁺ε)上界相比，本文在该线族类上实现了Θ(n²)的最坏情况，并给出了与理论一致的O(n²)枚举算法；在该类中性能与理论最优相符。

**⚠️ 局限性**

局限性包括仅适用于同一条规的线族，无法推广到任意线；需要进一步确定哪些线族满足二次上界；只考虑一般位置假设，未讨论数值误差或更广泛的几何约束。

---

## 457. Mutation Testing for Reproducibility Safeguards in Machine Learning Research Software: An Empirical Study

**arXiv ID:** 2608.27100 | [PDF](https://arxiv.org/pdf/2608.27100v1)

**作者:** Ilya Shulepov `[一作]` `[通讯]` (Independent Researcher), Ilya Shulepov (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用变异测试评估机器学习研究仓库中已存在的验证工作流是否能检测到可复制性相关实验配置的变化。

**💡 创新点**

提出针对可复制性相关实验配置的专用变异模型、冻结的实验协议以及可追溯的验证结果记录，实现了对现有验证机制的系统性评估。

**🔧 技术方法**

核心技术为Python实现的MLReproMutate变异工具，基于AST检测与变异候选、baseline‑first执行、语义等价性校验以及有限恢复策略。

**📊 数据集**

采用39个冻结的真实机器学习研究仓库–运算案例（共计13+24可执行案例），不使用公开数据集，而是直接在原始仓库环境下运行。

**📈 对比分析**

比较方法为统计已执行变异中被验证工作流检测到的比例；在实验样本中，验证工作流仅检测到2/23（≈8.7%）确认非等价变异，表明现有验证机制对可复制性变更的敏感度极低。

**⚠️ 局限性**

局限性包括：变异操作覆盖范围有限，仅包含随机种子、依赖限制、数据拆分与交叉验证等四类；样本量小且非随机；受历史环境漂移影响，部分仓库无法执行；未对多种验证机制进行交叉验证，结果不易推广到更大范围。

---

## 458. Sintr: Safe Interactive Transactions in the Presence of Byzantine Clients

**arXiv ID:** 2608.27091 | [PDF](https://arxiv.org/pdf/2608.27091v1)

**作者:** Austin T. Li `[一作]` (Cornell University), Florian Suri-Payer `[通讯]` (Cornell University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为 Safe Interactive Transactions (SIT) 的框架，针对客户端集中式 BFT 数据库通过多客户端冗余执行与签名背书，恢复应用级事务的执行有效性，并支持按对象的可变完整性策略。

**💡 创新点**

创新点包括：①在客户端侧引入事务级冗余验证与背书，抵御拜占庭客户端；②设计异构对象完整性策略与信息流控制机制，允许按需提升验证强度；③支持治理事务实时更新策略，并保持事务与策略的串行一致性。

**🔧 技术方法**

使用的技术包括：客户端冗余执行、数字签名与加密证明、信息流控制（非干扰）、BFT 共识（HotStuff 等）、签名验证与 HMAC、可变策略版本化、治理事务与动态写集发现。

**📊 数据集**

实验使用的基准数据集包括 YCSB（键值存储）、TPC‑C 与 TPC‑E（关系型数据库）。

**📈 对比分析**

与四个现有 BFT 数据库（Basil、Pesto、Peloton‑SMR、Tx‑SMR）基线进行对比，SIT 在吞吐量上仅下降 3–16%，延迟上升 3–21%，在 CPU 受限系统中成本更显著，但整体仍保持可接受的性能。

**⚠️ 局限性**

局限性包括：需要预估或动态发现写集、依赖客户端可信执行与签名验证，若验证成本高可能导致性能退化；治理事务与策略更新可能引起短暂 abort；框架未针对非交互式或只读事务进行特殊优化；对拜占庭客户端的防御仅保证一致性，无法完全阻止耗尽资源攻击。

---

## 459. GRAFT: Grounded and Efficient Online Reinforcement Adaptation for Fine-Grained Robot Manipulation

**arXiv ID:** 2608.27079 | [PDF](https://arxiv.org/pdf/2608.27079v1)

**作者:** Yibo Qiu `[一作]` (University of Science and Technology of China), Mingzhai Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种 GRAFT 框架，实现预训练视觉语言动作（VLA）策略的在线快速适配，利用区域级监督学习视角特化的视觉锚点，并通过冻结前缀缓存和单步动作生成显著提升学习效率。

**💡 创新点**

创新点包括：①身份无关多区域监督（FreeDice）使锚点学习到任务相关的局部视觉信息；②视角专门化锚点与全局上下文重加权，增强多视角对齐；③冻结视觉语言前缀并缓存 KV 状态，配合单步动作生成，极大减少在线更新计算量；④在部署时无需任何区域提议或掩码，保持轻量化。

**🔧 技术方法**

主要技术：预训练 VLA 作为基线，强化学习（RL）与行为克隆结合的在线适配；多视角视觉锚点设计与自由区域匹配损失；冻结前缀 KV 缓存；单步一致性动作生成；实验中使用真实机器人和人类演示数据。

**📊 数据集**

数据集：在 JAKA 机械臂上进行四个细粒度生物医学操控任务——Petri Dish De-lidding、Centrifuge Tube Loading、Precision Liquid Transfer 和 Pipette Tip Attachment；每个任务收集 10 条遥控演示，随后在 45 分钟内进行在线适配。

**📈 对比分析**

对比方法：RL‑FM（多步流匹配）、RL‑CP（单步一致性）、GRAFT‑Union（使用区域合并监督）与 GRAFT；在固定 45 分钟适配预算下，GRAFT 的最终成功率提升 25%（82.5% 对比 57.5%），在最难的 Centrifuge Tube Loading 上从 2/10 提升到 9/10；学习更新吞吐量从 2.21 步/秒提升至 21.96 步/秒（约 10× 加速）。

**⚠️ 局限性**

局限性：模型为即时响应式，未显式记录过去动作或任务进度，对更长时序或需要持续视觉追踪的任务可能表现不足；在极端视觉遮挡或动态环境下的鲁棒性仍待验证。

---

## 460. Emotional Preferences as Goal-Priority Regulation

**arXiv ID:** 2608.27072 | [PDF](https://arxiv.org/pdf/2608.27072v1)

**作者:** Shiqi Liu `[一作]` (Huazhong University of Science and Technology), Guanyu Qi `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通过外部强化学习自动生成情绪偏好（情绪优先级调节）的框架，结合预训练的多目标强化学习（MORL）内部控制器与外部偏好生成网络；

**💡 创新点**

创新点在于：1）将情绪偏好定义为状态依赖的目标优先级调节机制；2）将优先级生成视为外部强化学习问题，实现高层目标驱动的低层目标权重自适应；3）给出了理论分析，量化表示学习偏好对最优策略的可实现性与表示误差关系；4）在合成环境中演示情绪偏好具备上下文切换、渐进权衡与时序持久性。

**🔧 技术方法**

技术包括：预训练Envelope Q-Learning生成可调优的多目标价值函数；外部使用DDPG式的actor‑critic学习将状态映射到权重向量；softmax将输出限制在概率单纯形；梯度传播通过内部Q值实现。

**📊 数据集**

使用人工构造的两类网格世界环境：Basic（仅有能量与成就两目标）与Advanced（增加安全目标与随机危险），无公开公开数据集。

**📈 对比分析**

与固定权重策略、手工规则权重、以及直接优化生存目标的模型对比；情绪偏好模型在基本环境下平均生存步长≈14.97，显著优于所有固定/手工规则策略（均≤12.88），但略低于直接生存模型（≈16.00）。在高级环境中表现类似。

**⚠️ 局限性**

限制包括：1）情绪偏好仅依赖即时状态，缺乏记忆与情绪状态；2）仅在离散、合成网格环境中验证，未检验连续或真实感知场景；3）性能受内部行为库表达力限制，若行为覆盖不足则无法达到最优；4）仅考虑生存这一高层目标，未涵盖更丰富的情绪驱动任务。

---

## 461. Unifying Detection and Adaptation in Task-Free Continual Learning

**arXiv ID:** 2608.27070 | [PDF](https://arxiv.org/pdf/2608.27070v1)

**作者:** Dezheng Han `[一作]` (Shandong University), Shuaishuai Guo `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 FiUni 框架，实现任务无关的持续参数高效微调，利用 Fisher 信息矩阵的主子空间进行批级任务识别和 LoRA 子空间构建。

**💡 创新点**

创新点在于：① 用 Fisher 主子空间作为统一的几何信号同时完成任务检测与子空间管理；② 在无任务边界的在线流中，通过阈值决策实现子空间复用、扩展与新建；③ 仅训练中间低秩矩阵，冻结 Fisher 子空间，显著降低参数量。

**🔧 技术方法**

技术方法包括：Fisher 信息矩阵 / K-FAC 估计、特征值分解提取主子空间、子空间相似度计算、基于阈值的三态决策（REUSE/EXPAND/NEW）、两窗口确认机制、LoRA 低秩微调。

**📊 数据集**

数据集：Standard CL Benchmark（AG News、Amazon Reviews、DBpedia、Yahoo Answers）、Long Sequence Benchmark（15 个任务）、TRACE（多语言、QA、代码、数学推理等）。实验模型为 T5-base、T5-large 以及 LLaMA-3.1-8B。

**📈 对比分析**

与多种任务感知与非感知的连续学习方法对比，FiUni 在不使用任务边界的情况下，参数占用显著减少，同时在大多数基准上取得与最先进任务感知方法相当甚至更好的平均准确率；在 LLaMA-3.1-8B 上表现尤为突出。

**⚠️ 局限性**

局限性：① 需要额外前向/反向计算以估计 Fisher 统计，增加计算开销；② 需要在预训练模型上估计统计，对已学 LoRA 的复用有限；③ 内存占用主要来自激活/梯度协方差缓存；④ 实验仅覆盖到 8B 参数规模，未验证更大模型；⑤ 对样本数和 r_det 的选择较敏感。

---

## 462. Cyber-Electromagnetic Anomaly Detection Through Time-Series Analysis

**arXiv ID:** 2608.27043 | [PDF](https://arxiv.org/pdf/2608.27043v1)

**作者:** María Teresa Guillén Navarro `[一作]` (University of Murcia), Gregorio Martínez Pérez `[通讯]` (University of Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

结合Zigbee物理层和流量层特征，构建随机森林和LSTM自编码器两种模型，对ZBDS2023数据集进行异常检测。

**💡 创新点**

首次将电磁信号与网络流量信息联合利用，提出窗口统计聚合的随机森林与无监督时序自编码器的对比研究。

**🔧 技术方法**

随机森林分类、LSTM-Autoencoder重构误差阈值、特征统计聚合、交叉验证与阈值调优等技术。

**📊 数据集**

ZBDS2023 Zigbee 物理层 RSSI 与 MAC 层帧数据，包含10个Philips Hue设备的正常与攻击记录。

**📈 对比分析**

在保留时间顺序的80/20拆分上评估，随机森林获得F1≈89.8%（精准97.7%召回82.9%），LSTM-AE获得F1≈64.1%（精准56.9%召回73.4%）。

**⚠️ 局限性**

特征维度有限导致区分度不足，模型仅在单一 Zigbee 环境验证，未涵盖多技术与多攻击类型。

---

## 463. Cascaded Batch Prompting

**arXiv ID:** 2608.27038 | [PDF](https://arxiv.org/pdf/2608.27038v1)

**作者:** Sho Hoshino `[一作]` (CyberAgent), Peinan Zhang `[通讯]` (CyberAgent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了分层批量提示（cascaded batch prompting），将复杂推理与符号归一化拆分为两阶段，以解决传统批量提示的不可预测性。

**💡 创新点**

创新点在于通过将推理和符号映射解耦并在批量处理环境下采用两阶段流程，既提升准确率又保持速度加速。

**🔧 技术方法**

使用 GPT‑4.1 / GPT‑4.1‑mini / Phi‑4 进行两阶段推理（推理阶段批量生成答案，符号归一化阶段单实例处理），并采用核采样 top‑p=0.9。

**📊 数据集**

实验数据集为 Massive Multitask Language Understanding（MMLU）多选题集和 Multi‑Genre Natural Language Inference（MNLI）数据集。

**📈 对比分析**

通过与单提示和传统批量提示在准确率、实例/秒吞吐量等指标对比，发现 cascaded batch prompting 在 GPT‑4.1‑mini 和 Phi‑4 上取得最高准确率，并实现与批量大小成正比的速度提升。

**⚠️ 局限性**

限制包括：仅在分类任务上验证；开放式生成任务尚未适用；两阶段流程产生额外推理开销；极大批量下可能出现输入输出错位，需要额外的 sanity check。

---

## 464. Differentiable Jitter Correction using Deep Learning-based Image Quality Metric for Phase-Contrast Micro-CT

**arXiv ID:** 2608.27034 | [PDF](https://arxiv.org/pdf/2608.27034v1)

**作者:** Junan Chen `[一作]` (ImFusion GmbH), Julia Herzen `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过可微分的迭代校正方法，利用深度学习预测的图像质量指标（DL‑VIF）从仅含抖动的投影数据中估计并补偿每个投影的刚性抖动，无需预扫描的无运动参考图像。

**💡 创新点**

1）将可微分投影几何优化与无参考VIF预测网络相结合；2）提出仅在图像背景区域施加TV正则化，避免对内部结构的过度平滑；3）设计轻量级3D残差网络，在128³体素上直接回归VIF，消除局部特征依赖，显著提升跨样本的泛化能力。

**🔧 技术方法**

可微分投影与反投影（梯度链），基于梯度的Adam优化，3D残差CNN用于预测DL‑VIF，图像质量指标（VIF、SSIM、锐度、TV）用于构建目标函数，平行光束几何模型解析梯度。

**📊 数据集**

16个多形态体积（金属条、蜿蜒蠕虫、心脏、脑、肾、睾丸等）来自DESY、ESRF；合成抖动样本共3770个，训练集13个，验证集3个，测试集3个。

**📈 对比分析**

与尖锐度、SSIM、VIF等传统目标函数进行对比；在四个测试样本（心脏、脑、睾丸、真实蠕虫）上，DL‑VIF+TV优化将SSIM从0.42‑0.57提升至0.73‑0.94；与参考VIF/SSIM上限相近；在真实蠕虫投影上达到0.9411，几乎等同于参照方法。

**⚠️ 局限性**

①训练数据被缩放到128³，可能限制高频细节的恢复；②仅验证了随机刚性抖动，对样本漂移、校准误差等其他几何误差的泛化尚未验证；③网络对分布偏移敏感，虽保持高度相关性但存在系统偏差；④缺乏对真实测量噪声和投影截断的充分评估。

---

## 465. Disentangling Optimization Scale from Preference Scale in DPO

**arXiv ID:** 2608.27032 | [PDF](https://arxiv.org/pdf/2608.27032v1)

**作者:** Ivan Kruzhilov `[一作]` `[通讯]`, Ivan Kruzhilov

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了直接偏好优化（DPO）中参数β与梯度尺度及损失可比性之间的耦合问题，并提出了一种基于centered‑softplus的标准化DPO损失，以解耦β的两重作用；

**💡 创新点**

创新点在于揭示β同时调节偏好噪声尺度和优化步长的事实，证明标准DPO损失在不同β下不可比，并设计了centered‑softplus正则化损失，使β只影响噪声尺度，恢复损失可比性并消除梯度消失；

**🔧 技术方法**

主要技术包括对DPO损失的解析梯度缩放分析、归一化损失构造、centered‑softplus变换、对β=0极限的线性分支、以及在LoRA微调下的实验验证；

**📊 数据集**

使用了HelpSteer3‑Preference、UltraFeedback Binarized和PKU‑processed HH‑RLHF三大公开pairwise偏好数据集，模型涵盖Qwen3‑4B‑Instruct‑2507、Qwen2.5‑3B‑Instruct、Ministral‑3B‑Instruct和Mamba‑2‑2.7B；

**📈 对比分析**

对比方法包括在相同学习率比例下将标准DPO与标准化DPO的验证曲线、损失曲线以及AlpacaEval 2和IFEval分数；结果表明标准化损失下的训练损失更能反映政策偏离程度，且在不同β下的性能差异可被准确监测，原始DPO在不同β下同样损失却可能产生大幅度KL偏差；

**⚠️ 局限性**

限制包括实验仅覆盖LoRA微调、四个模型和三数据集，未验证更大模型或全微调情况；β=0线性分支在实际训练中不稳定；以及不同优化器实现可能导致的微小系统性差异。

---

## 466. Lunar Generalizations of the Euclidean Minimum Spanning Tree in the Plane and their Expected Costs

**arXiv ID:** 2608.27118 | [PDF](https://arxiv.org/pdf/2608.27118v1)

**作者:** Ondřej Draganov `[一作]` (INRIA), Morteza Saghafian `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将欧几里得最小生成树（EMST）推广到月球EMST，处理n个点在二维空间中且具有s+1种颜色的情况。

**💡 创新点**

主要创新点在于证明了对于随机选择的n个点和随机着色的s+1种颜色，存在一个常数c_s，使得月球EMST的期望成本在n趋向于无穷大时收敛于c_s√(n)。

**🔧 技术方法**

使用了随机几何和组合优化技术，特别是Kruskal算法的推广。

**📊 数据集**

使用了在[0,1]^2中均匀随机选择的点集，并对这些点进行随机着色。

**📈 对比分析**

通过与经典EMST的比较，证明了月球EMST的期望成本与n的平方根成正比，且常数c_s的存在性得到了证明，但其精确值仍然未知。

**⚠️ 局限性**

限制在于尽管存在常数c_s，但其精确值尚未确定，尤其是对于标准EMST的常数c_0。

---

## 467. SecureDrive-FL: Joint Differential Privacy and Gradient-Aware Selective Homomorphic Encryption for Federated Driver Monitoring

**arXiv ID:** 2608.27108 | [PDF](https://arxiv.org/pdf/2608.27108v1)

**作者:** Baran Can Gül `[一作]` (University of Stuttgart), Michael Weyrich `[通讯]` (University of Stuttgart)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种联邦学习框架 SecureDrive‑FL，将差分隐私与梯度感知选择性同态加密（GASHE）联合应用于驾驶员监测任务。

**💡 创新点**

创新点在于通过 DP‑SGD 的剪裁与噪声参数自动生成加密阈值，实现闭环 DP+HE 的协同加密，仅加密最敏感梯度，从而显著降低加密成本并提升通信安全。

**🔧 技术方法**

核心技术包括 DP‑SGD、CKKS 同态加密、GASHE 选择性加密以及联邦聚合算法。

**📊 数据集**

使用融合了 State Farm 与 AUC 两个驾驶员分心检测数据集的约 5 万张图像，进行十分类别的驾驶员分心识别。

**📈 对比分析**

与 FedAvg、全参数 CKKS、单独 GASHE 等基线比较，SecureDrive‑FL 在 MitM 与模型中毒攻击下保持 78–82% 的准确率，ASR 仅 3.9%，且运行时开销仅比单纯 DP‑SGD 低约 8–10%，在性能与安全上均实现优越平衡。

**⚠️ 局限性**

局限性包括边缘设备上的加密/解密开销仍较高、未对拜占庭/ Sybil 等更激进攻击进行评估、以及对多轮自适应中毒攻击的鲁棒性需要进一步验证。

---

## 468. The Framing Gap: Indirect Prompt-Injection Exfiltration Defeats Surface-Level Defenses in Tool-Using Agents

**arXiv ID:** 2608.27092 | [PDF](https://arxiv.org/pdf/2608.27092v1)

**作者:** Md Habibur Rahman `[一作]` (Gyeongsang National University), Jaeho Kim `[通讯]` (Gyeongsang National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文搭建了安全实验室，利用合成机密和模拟工具，系统评估了工具使用型语言模型在间接提示注入中的泄漏风险，并揭示了“框架缺口”——通过重新表述相同的泄漏指令即可绕过模型的拒绝，进一步评估了多种防御措施的效果。

**💡 创新点**

创新点在于提出并量化框架缺口概念，展示了同一泄漏内容在不同表述下对模型行为的巨大影响；构建了完整的攻击与防御基准，包括10类注入、8种重新表述变体及编码攻击；证明了仅靠模型层面的拒绝不足，指出了目标导向的、payload‑blind 的防御（目的地白名单、能力隔离）才真正有效。

**🔧 技术方法**

技术方法包括：使用合成 canary 作为机密、mock 工具仅记录参数、匹配的干净/毒化对照设计、攻击分类与重新表述家族、输入检测器、输出中介（egress guard）、能力隔离的规划/读取分离、以及对比学习防御（SecAlign）等；评估指标为泄漏成功率（ASR）并给出 Wilson 置信区间。

**📊 数据集**

数据集主要由自定义的攻击网页构成：10 类注入原型、8 种重新表述变体以及若干编码变体，每类在每个模型上执行 10–20 次匹配试验；机密为高熵合成字符串，不使用真实凭据。

**📈 对比分析**

对比方法：在同一 harness 下测量每类攻击的 ASR，baseline 为 0% 而重新表述后可达 40%–100%；对防御进行基准测试，发现目的地白名单和能力隔离将 ASR 降至 0%，而 SecAlign、通道分离等仅降至 20%–40%；跨六大模型的结果表明框架缺口普遍存在。

**⚠️ 局限性**

局限性包括：仅评估单步决策的 agent，无法覆盖循环规划或多次调用的情形；使用的合成机密和简化的工具接口可能低估真实系统中的复杂性；防御如编码归一化仅覆盖已知编码，未知编码仍可绕过；产品测试规模有限，且未揭示具体机制；未评估多目标/多任务场景下的适用性。

---

## 469. Challenging Benchmarks for Diagrammatic Equivalence of Circuits in TPTP and SMT-LIB

**arXiv ID:** 2608.27087 | [PDF](https://arxiv.org/pdf/2608.27087v1)

**作者:** Julie Cailler `[一作]` (University of Lorraine), Sophie Tourret `[通讯]` (University of Lorraine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套新的图形等价性（diagrammatic equivalence）基准集合，针对量子电路和更一般的对称单子类（PROP）电路设计了三种难度不同的判定问题：一般电路等价（CE）、无生成器的置换电路等价（PCE）以及简化置换电路等价（SPCE），并给出了第一阶逻辑（FOL）编码与脚本自动生成实例；

**💡 创新点**

创新点在于：1）提供了针对量子电路验证需求的系统化基准集；2）定义了三种层级的等价判定问题并对应简化的 coherence 规则；3）设计了可在 SMT-LIB 与 TPTP 中使用的 Guarded FOL 编码，兼顾算术与等式推理；4）发布完整的生成脚本和 benchmark 集，促进复现与进一步研究；

**🔧 技术方法**

技术方法主要包括：对称单子类的语法定义、图形字符串变换规则、第一阶逻辑编码（带线性整数算术）、SMT‑LIB 与 TPTP 语法实现、随机/规则驱动的图形实例生成脚本；

**📊 数据集**

使用的数据集为自动生成的 CE、PCE、SPCE 实例，共数千个问题，参数覆盖输入线数、列数、组合符号数量等，已上传至 Zenodo；

**📈 对比分析**

评估方法：在 Grid5000 集群上分别用 Z3 5.0.0 和 Vampire 1.3.0 进行 30 s 超时求解，统计成功证明数；结果显示 Z3 在所有基准中均优于 Vampire；SPCE 在大多数情况下可完全解决，PCE 随线数/规模提升显著下降，CE 题目最难，Vampire 在中等规模即全部失败；

**⚠️ 局限性**

局限性：1）CE 题目对算术与等式推理的交叉要求导致现有 ATP 仍难以突破；2）现有编码需 Guard，未能完全无算术实现；3）缺乏更多支持算术且能生成证明的 ATP；4）benchmark 生成参数尚可进一步细化，以探究规则瓶颈与相位转移；

---

## 470. A Contract-Centered Architecture for Scalable and Manageable Agentic Runtimes

**arXiv ID:** 2608.27086 | [PDF](https://arxiv.org/pdf/2608.27086v1)

**作者:** Yaxiao Liu `[一作]` (PwC China AI Center), Jiaxing Song `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个面向企业AI部署的共享组织合同架构，定义了四个责任对象（Skill、Harness、Scaffold、数据子层），并提出了一个可界定、可验证的成本感知能力-容量可分离性假设（P1），同时给出了基于集群-周期随机交叉实验的测量协议。

**💡 创新点**

将业务能力、运行时治理、物理执行与企业数据治理分别拆解为可版本化的责任对象；构建了一个基于合同边界的运行时架构；提出了可界定的成本感知可分离性假设；设计了可被证明可驳斥的实验协议。

**🔧 技术方法**

利用控制/数据面分离理论、契约式编程（C=⟨I,O,G,A,B,V⟩）、随机交叉实验设计、统计假设检验边界以及预算约束等方法。

**📊 数据集**

文中未给出具体数据集，也未完成任何运行时实现或实验；所述数据子层为企业级语义与遥测堆栈，假定其为外部、独立治理的。

**📈 对比分析**

通过预注册的边界与阈值，利用集群-周期随机交叉实验得到四种结论：支持、驳斥、条件工程或不可得；未给出具体性能指标。

**⚠️ 局限性**

缺乏实际实现与实验验证，假设与协议仍为理论；实验设计对集群可复现性与洗脱要求苛刻；未说明具体的预算与误差估计；在真实企业环境中的可行性与可扩展性尚未验证。

---

## 471. Performance Foundations of Parallel & Distributed Reasoning Language Models

**arXiv ID:** 2608.27046 | [PDF](https://arxiv.org/pdf/2608.27046v1)

**作者:** Maciej Besta `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统化了基于RL的后训练（RL‑VR、RLHF、RLAIF等）在大型语言模型（LLM）上的应用，并从计算和系统角度分析其工作负载、内外模型并行性以及多模型训练流水线；

**💡 创新点**

创新点在于提出了面向RL‑LLM的完整并行性分类法（内模型并行、交叉模型并行、阶段融合、异步执行等），结合工作‑深度模型给出计算复杂度分析，并给出可操作的高效、可扩展、成本友好的设计准则与未来研究方向；

**🔧 技术方法**

技术包括PPO、GRPO、DPO等RL框架；多模型并行（数据并行、张量并行、流水线、序列/上下文并行、专家并行、混合并行）；阶段融合与异步执行；工作‑深度分析与内存模型；以及对现有RLM框架（ReaL、OpenRLHF、TRL、HybridFlow等）的评估；

**📊 数据集**

论文主要以理论和模拟为主，并未使用具体数据集，而是基于已有的RLM系统（如DeepSeek‑R1、OpenAI o3、Kimi k1.5等）进行讨论；

**📈 对比分析**

通过工作‑深度和内存分析，对PPO、GRPO、DPO在不同并行策略下的计算成本进行比较，并以实测框架为例给出性能评估与优化建议；

**⚠️ 局限性**

局限性包括：1）未考虑网络通信、同步与排队延迟等实际硬件开销；2）假设理想负载平衡，忽略异构设备与拓扑影响；3）主要为分析与设计建议，缺乏大规模实验验证；4）对某些高级技术（如异步策略的收敛性）讨论不够深入。

---

## 472. Omni-Interactive Universal Embedder

**arXiv ID:** 2608.27044 | [PDF](https://arxiv.org/pdf/2608.27044v1)

**作者:** Wei-Yao Wang `[一作]` (Sony Group Corporation), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了OmniUE，一个支持文本、视频、音频三种模态输入与交互的统一嵌入框架，并提出OmniCHOIR基准；

**💡 创新点**

①实现了三模态交互（文字、视觉区域、音频时间段）并能联合使用；②引入可学习的多层标记和上下文聚合机制以捕获细粒度信息；③利用SAM‑3与SAM‑Audio作为视觉与音频分割器；④构建了新的omni‑interactive TVA2A 检索基准；

**🔧 技术方法**

基于大语言模型（Qwen2.5‑Omni）作为骨干，配合 SAM‑3、SAM‑Audio 分割器；使用多层可学习标记、上下文聚合、LoRA、GradCache、InfoNCE 对比学习等技术；

**📊 数据集**

训练使用约3.5M公开多模态样本；评测采用 MMEB‑v2‑video、MAEB、SCaR、OmniCHOIR 等基准；OmniCHOIR 由 SAM‑Audio‑Bench、ESC‑50、Qwen3‑Omni‑30B 等数据生成；

**📈 对比分析**

在文本交互视频、音频、视觉交互以及全交互任务上，OmniUE 均超越现有 SOTA，平均提升分别为 10.5%、1.1%、83.7% 及 24.1%；相较于两塔模型和 LLM 嵌入器表现更优；

**⚠️ 局限性**

仍受限于预训练模型的语义表达能力，分割器质量影响检索效果；需要大量训练数据和计算资源；在极其复杂或实时场景下性能尚待进一步验证；

---

## 473. Multi-Person Human Motion Forecasting in Complex Scenes

**arXiv ID:** 2608.27039 | [PDF](https://arxiv.org/pdf/2608.27039v1)

**作者:** Serdar Ozsoy `[一作]` (University of Bonn), Juergen Gall `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种统一的条件扩散模型OCSD，用于在复杂场景中预测多人的未来动作，能够同时考虑历史运动、社交交互与物体信息；

**💡 创新点**

创新点在于在扩散过程的每个时间步使用基于FiLM的物体调制机制，以及在U-Net瓶颈中融合社交编码，实现细粒度的人物-物体与人物-人物关系的联合建模；

**🔧 技术方法**

采用条件扩散（DDPM）框架，结合U-Net结构、跨注意力、FiLM调制和社交编码器；

**📊 数据集**

在两个公开基准上进行实验：HiK（厨房环境，最多16人）和HOI-M3（室内场景，最多5人）；

**📈 对比分析**

与多种现有基线（如IAFormer、SAST、HUMOF等）比较，OCSD在HiK短期路径误差、姿态误差和HOI-M3短期/长期指标上均获得显著提升，尤其在10秒长时预测的NDMS提升至0.34、UMWR保持高多样性；

**⚠️ 局限性**

局限性包括对物体语义类别依赖较强、在极大人群规模下可能面临计算负荷以及在极短时间窗内对姿态的细节捕捉仍有提升空间。

---

## 474. Reasoning about In-Context Samples for Machine-Translation

**arXiv ID:** 2608.27036 | [PDF](https://arxiv.org/pdf/2608.27036v1)

**作者:** Maxime Bouthors `[一作]` (SYSTRAN by ChapsVision), François Yvon `[通讯]` (Sorbonne Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于片段的推理框架，通过检索并提取/生成并利用翻译记忆中的并行源-目标片段来指导大型语言模型进行机器翻译。

**💡 创新点**

创新点在于：①将片段提取视为中间推理步骤并在生成过程中显式化；②利用教师模型对片段和草稿进行银标注，再通过监督蒸馏训练学生模型；③展示片段推理在多语言、多领域上显著优于传统k-shot和草稿方法，并能在无训练域上保持鲁棒。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen3-32B教师，Qwen3-8B学生）的链式思考推理；检索增强（从TM检索k=0-3个示例）；教师生成的银片段与草稿；监督蒸馏训练学生；多指标评估（SacreBLEU、COMET、MetricX）。

**📊 数据集**

使用了来自OPUS的多语言数据，覆盖16个域（英语-德语、法语、波兰语、乌克兰语、西班牙语等），每对语言约10k高质量样本，构成160k多语言训练集，并在1k/100的测试/验证集上评估；另外使用未见域GNOME进行泛化测试。

**📈 对比分析**

与无推理基线、单纯k-shot、草稿等方法相比，片段+草稿模型在BLEU、COMET、MetricX上平均提升约1-3分，且在大部分域/语言上占优；在低覆盖率场景下收益更显著；在GNOME等外域也能保持提升。

**⚠️ 局限性**

局限性包括：实验规模有限，未覆盖更多低资源语言和极端专业领域；依赖高质量TM/检索资源，低资源场景适用性受限；教师生成的银片段可能包含错误、误导，误差会在蒸馏过程中传播；片段提取受LLM能力限制，难以保证完全语义一致或最佳分块。

---

## 475. Representing and Parsing Korean Constituency Structure at Different Levels of Granularity

**arXiv ID:** 2608.27035 | [PDF](https://arxiv.org/pdf/2608.27035v1)

**作者:** Jungyeul Park `[一作]` (Korea Advanced Institute of Science & Technology), Chulwoo Park `[通讯]` (Anyang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Penn Korean Treebank 进行三种终端粒度和 POS 粒度的对齐表示（Morpheme+XPOS、Eojeol+XPOS、Eojeol+UPOS），并在同一 Stanza 递归神经网络非二叉转移系统（Top‑down、In‑order、Bottom‑up）下评估韩语成分结构解析性能。

**💡 创新点**

提出将 eojeol 作为句法树终端而将词形素信息保持为对齐层的统一设计，并系统化比较终端粒度与语法标签粒度对解析质量的影响；将三种表示统一转换到同一 eojeol+UPOS 领域实现公平对比。

**🔧 技术方法**

使用 Stanza 框架的递归神经网络非二叉转移系统，结合词嵌入、预训练词向量和字符 LM 等技术进行解析。

**📊 数据集**

采用 Penn Korean Treebank（去除空位、与 eojeol 对齐、重新标注）作为实验数据集。

**📈 对比分析**

在 gold 分词与预词性标注条件下，以相同模型、相同评估协议对三种表示和三种转移顺序进行比较；归一化后 Morpheme+XPOS 在 Bottom‑up 模式下获得最高 F1（84.76 / 82.17），Eojeol+XPOS 次之，Eojeol+UPOS 最差；Bottom‑up 一般优于 In‑order 与 Top‑down。

**⚠️ 局限性**

仅基于 Penn Korean Treebank 的转换，未覆盖其他树库；仅评估 gold 先前标注情况，未做端到端形态分析；移除空位可能忽略空位分析；转换与评估策略存在主观选择。

---

## 476. FoldPipe: Bounded Remote Streaming of Native Molecular Shards with Asynchronous Prefetch

**arXiv ID:** 2608.27029 | [PDF](https://arxiv.org/pdf/2608.27029v1)

**作者:** Dhiren Mukesh Khatri `[一作]` `[通讯]` (Independent Researcher), Dhiren Mukesh Khatri (Independent Researcher)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

FoldPipe 是一个轻量级 Python 预取层，用于在训练期间异步获取并解序列化远程存储的原生 PyTorch/PyG 分片，并与当前分片的计算重叠；

**💡 创新点**

创新点在于提供了一个可绑定大小的分片读取接口，并实现单阶段异步预取，将远程 I/O 与 GPU 计算有效重叠，且仅需最小的格式迁移；

**🔧 技术方法**

使用的技术包括异步后台线程预取、内存中直接反序列化、Python 迭代器封装和对 Hugging Face/Google Drive 的接口；

**📊 数据集**

实验使用 MD17 aspirin 数据集的 5 份分片（共 25,000 个分子结构）以及 PyG SchNet 模型；

**📈 对比分析**

通过 20 对齐次实验（顺序流与 FoldPipe 交替执行），测量 I/O–compute 重叠时间、GPU 利用率、内存占用和整体跑时，FoldPipe 平均跑时更短（76.78 s vs 83.37 s），但几何平均加速为 1.059×，95% 置信区间包含 1.0，因网络波动不一定带来显著加速；

**⚠️ 局限性**

限制包括：实验仅在公共网络上进行，网络延迟不一致导致结果不确定；样本量仅 20 对，置信区间宽；仅评估单一分子系统、模型和设备；FoldPipe 仅在分片层级操作，无法进行样本级远程流、分布式分片或自动转换；并且反序列化原生 PyTorch 对象存在安全隐患。

---

## 477. Bug Localization from Bug Reports: A Multi-Objective Approach

**arXiv ID:** 2608.27089 | [PDF](https://arxiv.org/pdf/2608.27089v1)

**作者:** Waleed Ahmad `[一作]`, Maryam Bashir `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 478. Vision-centric generative AI models: A software-hardware perspective

**arXiv ID:** 2608.27199 | [PDF](https://arxiv.org/pdf/2608.27199v1)

**作者:** Eleni Tselepi `[一作]` (University of Edinburgh), Themis Prodromakis `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析并量化了视觉生成模型（VAE、GAN、扩散、变压器）在参数、能效与硬件平台上的成本，并将这些模型映射到七个实际应用场景，提出软件-硬件协同设计策略。

**💡 创新点**

首次系统对比四类生成模型在参数、FID与能效上的差异，识别出“模型-硬件-应用”匹配缺口，并提出共设计路线图。

**🔧 技术方法**

使用多平台（GPU、TPU、ASIC、FPGA、CIM等）能耗测评、FID指标、参数计数，构建模型-硬件-应用映射表。

**📊 数据集**

使用 CIFAR‑10、ImageNet 128×128、ImageNet 256×256 三个基准数据集。

**📈 对比分析**

通过参数计数与FID对比，展示 GAN 在参数效率上领先；扩散与变压器在高分辨率下虽质量略优，但参数与能耗显著更高；在七类应用中，GAN 是唯一在所有场景均满足需求的模型。

**⚠️ 局限性**

缺乏对实际部署环境下动态资源调度与模型压缩的深入评估；未考虑新兴硬件技术（如 Neuromorphic、HBM‑PIM）的真实性能；研究仅为视角性分析，缺乏原始实验验证。

---

## 479. X-WAD: eXplainable Web Anomaly Detection

**arXiv ID:** 2608.27172 | [PDF](https://arxiv.org/pdf/2608.27172v1)

**作者:** Matteo Bitussi `[一作]` (Fondazione Bruno Kessler), Roberto Doriguzzi-Corin `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种可解释的 Web 异常检测工具 X-WAD，通过基于 Transformer 的半监督模型对 HTTP 请求进行异常检测并生成 token 级别解释。

**💡 创新点**

将输出 logit 的 token 级别 surprisal 直接映射为异常分数并可视化，既实现异常检测又无需额外特征归因，同时利用此解释发现训练集标签污染导致的 backdoor 类似错误，并通过修正数据集显著提升性能。

**🔧 技术方法**

采用 Transformer 的 MLM（ModernBERT）和 CLM（SmolLM）模型，使用 bbpe tokenizer、token‑level cross‑entropy loss、阈值设定基于 Z‑score、SHAP 对比与 heatmap 可视化等技术。

**📊 数据集**

主要使用 WordPress 服务器收集的 SR‑BH2020（srbh）HTTP 请求数据集（525k 正常、382k 异常），并创建修正版 srbhfix。

**📈 对比分析**

在测试集上通过阈值比较评估 FPR、FNR、F1，原始数据 FNR 约 15–24%、F1 86–91%；修正数据集 FNR 降到 2–3%、F1 提升至 98–99%，并在各 CAPEC 类别中均显著提升。

**⚠️ 局限性**

仅在有标签测试集的实验，缺乏真实生产环境评估；token‑级解释受词表与模型差异影响，修正数据集需人工校验；方法目前仅针对文本请求，扩展至其他文本类数据仍需验证。

---

## 480. Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable

**arXiv ID:** 2608.27167 | [PDF](https://arxiv.org/pdf/2608.27167v1)

**作者:** Pranav Aggarwal `[一作]` `[通讯]` (Independent Researcher), Pranav Aggarwal (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过实验检验并改进大语言模型（LLM）代理在面对不可知问题时的行动决策，证明展示形式（而非信息本身）会诱使模型作出错误的预测性行动。

**💡 创新点**

创新点在于：① 采用“不可知oracle”与严谨的实验设计，证明专业仪表板的包装会显著提升LLM在不可知问题上的投票率；② 通过制造与真实相同格式但信息全为伪造的数据，证明触发机制是包装而非内容；③ 提出并验证可训练的“决策门”，使模型在同类任务中能够保持正确的拒绝行为；④ 对模型的判别门进行跨域迁移和格式依赖性分析，揭示门的脆弱性。

**🔧 技术方法**

使用的技术包括：对12款前沿LLM（OpenAI、Anthropic、Google、Meta等）进行统一评估；在Qwen2.5-3B-Instruct上执行4‑bit QLoRA+链式思维监督微调（3个epoch）；设计多层级的证据梯度（L0–L2）、全盘伪造与部分伪造实验；采用严格与语义两种解析器提取模型决策；评估指标包含承诺率、Youden’s J、Brier分解、AUROC等。

**📊 数据集**

数据集主要由：① 12个“不可知”金融/体育/天气样本（共40个问题），② 4个转移域（加密、体育、天气）各24个问题；③ 540个合成的骰子、硬币、计时器等案例用于门训练（包含与回答相关的指标面板与非可预测面板的匹配）；④ 参考的公开oracle（短期价格方向、比赛结果、天气预报）用以验证不可知性。

**📈 对比分析**

比较方法：在三种证据级别下统计模型的承诺率（0.065 → 0.54），通过等价检验确认伪造面板与真实面板差异不显著；利用Youden’s J评估拒绝决策的辨别力，发现训练门后模型在所有四个域的J值从+70到+100不等，显著优于未训练模型；Brier分解显示原始模型在承诺时的概率与结果无关，而门训练后概率趋向保守且更可靠。性能表现：未训练模型在不可知问题上承诺率达54%；门训练后在相同设置下降至0%，同时对可答问题保持≈100%的回答准确率。

**⚠️ 局限性**

局限性：① 仅在3B规模模型与单一架构上验证，无法确认门在更大模型或其他模型架构下的可迁移性；② 受影响的模型集中在同一开发者的系列（Claude、Gemini等），其余模型未显示该现象；③ 仍存在潜在词汇或语法线索导致的偏差，尤其是“fair coin”等随机词汇；④ 门对提示格式极度敏感，细微的输出结构改变即可使门失效；⑤ 训练数据仅覆盖骰子、硬币等简易事件，未覆盖更复杂领域；⑥ 对天气等领域的不可知判定依赖于集成模型的真实概率，可能导致误判。

---

## 481. ReViCo: Unveiling the Limitations of VLMs in Visual Text Understanding via Error Correction

**arXiv ID:** 2608.27154 | [PDF](https://arxiv.org/pdf/2608.27154v1)

**作者:** Bojun Zhang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Yu Zhou `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布了ReViCo基准，通过视觉文本错误纠正任务评估VLM对视觉文本的深层理解。

**💡 创新点**

提出新的视觉文本错误纠正任务，创建可在真实图像中检测和纠正文本错误，并结合提示工程和强化学习两种策略提升模型性能。

**🔧 技术方法**

使用提示工程（直接纠正与背景信息增强）、OCR+LLM流水线以及基于GRPO的强化学习训练，并辅以多级验证奖励MLVR。

**📊 数据集**

构建了1229条真实错误图像的测试集（中文730、英文499），以及4531条自动合成的训练集，结合EST‑VQA、ST‑VQA等公开数据。

**📈 对比分析**

与多种开源VLM（Qwen、InternVL）和闭源VLM（GPT‑4o、Gemini、Seed）以及OCR+LLM基线对比，最高F1仍低于人类，提示与RL提升有限。

**⚠️ 局限性**

仅覆盖中文和英文、未涵盖低资源语言、只聚焦文本错误、未实现任务特定微调，且性能仍与人类存在显著差距。

---

## 482. ANTShapes Benchmarking Datasets for Event-Based Neuromorphic Object Classification

**arXiv ID:** 2608.27150 | [PDF](https://arxiv.org/pdf/2608.27150v1)

**作者:** M. Middleton `[一作]` (University of York), M. A. Trefzer `[通讯]` (University of York)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文利用ANTShapes模拟工具生成四个无噪声、可控的事件驱动三维物体分类数据集，并与现有事件数据集进行基准测试。

**💡 创新点**

创新点在于：①设计并公开ANTShapes模拟工具，实现可自定义形状、位置、旋转、变形与翻译的事件数据；②构建四种不同难度的基准数据集；③验证该工具生成的数据能用于高效的SNN分类。

**🔧 技术方法**

使用技术包括：事件数据模拟、LIF型脉冲神经网络（SNN）及其卷积架构、surrogate gradient训练方法、PCA聚类分析、混淆矩阵与准确率评估。

**📊 数据集**

所用数据集包括：自制ANTShapes（Standard、Translation、Distortion、Rotation）以及对比集 N‑MNIST、CIFAR10‑DVS、POKER‑DVS、DVSGesture。

**📈 对比分析**

通过在相同卷积SNN架构上做10次随机初始化训练，计算平均准确率和方差进行比较；ANTShapes Standard 92.2% 为最高，Distortion 最低 83.5%；与传统数据集相当（N‑MNIST 97%，CIFAR10‑DVS 55.6%，POKER‑DVS 96.6%，DVSGesture 88.9%）。

**⚠️ 局限性**

局限性在于：①数据仅涉及单一物体，分类任务不随时间变化；②对模型的依赖性较高，只评估了一种SNN；③多物体场景、异常检测等更复杂任务尚未覆盖；④相对难度提升不足，需更复杂的场景与事件动态。

---

## 483. Thomson: Continual Learning of Frontier Models for SovereignAI

**arXiv ID:** 2608.27147 | [PDF](https://arxiv.org/pdf/2608.27147v1)

**作者:** Shengzhuang Chen `[一作]` (Thomson Reuters), Jonathan Richard Schwarz `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研发了一系列 SovereignAI 基础模型，利用持续学习（Continual Learning）在开源指令调优模型上进行价值重新校准、知识注入、行为和工具专化，并通过 DPO 与强化学习实现高阶任务能力。

**💡 创新点**

创新点在于将持续学习与价值、知识、行为三维模块相结合，形成可在低成本、可重复的“模型工厂”流程；通过 Constitutional RL、Deep Research harness 和数据中心化持续预训练，显著缩短从开源模型到前沿性能的时间与投入。

**🔧 技术方法**

技术手段包括：持续学习框架、数据驱动的 Mid‑Training（持续预训练）、DPO（直接偏好优化）、强化学习（多阶段 RL）、工具调用与深度研究（Deep Research）抓取与压缩、CapTrack 评估、Constitutional RL 奖励、奖励设计与多维评判。

**📊 数据集**

使用的数据集涵盖：开源权重模型（如 Gemini 3 Pro、Claude 等）及其预训练数据、公开基准（MT‑Bench、MMLU、TruthfulQA 等）、专有法律/税务/新闻数据库、从专业人士收集的多样化查询与 Deep Research 任务轨迹，以及专门构造的对抗性与合规性测试集合。

**📈 对比分析**

通过在多项公开与专有基准上与前沿模型对比，SovereignAI 在成本/性能曲线、π‑形性能提升以及整体综合评分上均实现或超越多家顶尖厂商的模型，单次训练成本低于 45 万美元，整体开发投入约 4000 万美元，显示出显著的性价比与可扩展性。

**⚠️ 局限性**

局限性包括：在编程等特定领域仍有轻度遗忘、对极端长文本或高风险情景处理仍需改进、模型仍依赖开源基模型，且奖励与评估需要大量人工标注，整体过程仍处于实验阶段。

---

## 484. Feature Transformation Enhanced Jacobi Polynomial Graph Filtering for Graph Anomaly Detection

**arXiv ID:** 2608.27144 | [PDF](https://arxiv.org/pdf/2608.27144v1)

**作者:** Xiang Wang `[一作]` (Fujian University of Technology), Zhenyu Meng `[通讯]` (Fujian University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种新的图异常检测方法JPGFN，结合特征分离变换、可自适应的Jacobi多项式图滤波和节点标签约束模块来提升检测性能。

**💡 创新点**

创新点在于：1）特征分离变换网络（FSTNN）针对不同属性自适应学习重要性，捕获细粒度特征；2）使用可学习参数a、b的Jacobi多项式构建图滤波器，显著提高频域信息适配性；3）加入节点标签一致性约束模块，充分利用少量标签信息。

**🔧 技术方法**

核心技术包括多层感知机（MLP）实现特征非线性变换；Jacobi多项式频域图滤波器（参数化多项式逼近）；图卷积与余弦相似度实现标签约束；交叉熵与对比损失的联合优化。

**📊 数据集**

在五个真实世界数据集上验证：Amazon、YelpChi、T‑Finance、Elliptic、Weibo。

**📈 对比分析**

与传统GNN（GCN、GAT、GraphSAGE、GIN）、空间域GAD（PC‑GNN、GAS、GFCN、GDN）以及频域GAD（BWGNN、AMNet、SEC‑GFD、AHFAN、EGNN、DSGAD）等14个基线进行比较，JPGFN在AUC‑ROC和AUC‑PR两指标上均位列第一，提升幅度从0.75%至6.49%不等。

**⚠️ 局限性**

局限性包括：对Jacobi多项式参数的学习需要额外训练，可能导致收敛不稳定；在频域适配方面对高阶多项式K的选择仍依赖经验；模型对大规模动态图或异构图的扩展尚未验证。

---

## 485. AROMA+: A Study of Factors Affecting Reproducible Builds in the Maven Ecosystem

**arXiv ID:** 2608.27125 | [PDF](https://arxiv.org/pdf/2608.27125v1)

**作者:** Mehdi Keshani `[一作]` (Bowling Green State University), Abbas Heydarnoori `[通讯]` (Bowling Green State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了自动化工具AROMA+，用于从Maven包的元数据和源码仓库中恢复构建环境（如JDK版本、行尾符、构建工具、发布标签等），并通过Reproducible Central的脚本实现对Maven和Gradle库的可重复构建，极大提升了可重复构建的覆盖率；

**💡 创新点**

核心创新在于：① 通过一系列启发式算法自动从缺失或不规范的POM/源码中推断构建信息；② 将这些信息生成符合RC格式的.buildspec，实现完全自动化构建；③ 在大规模Maven生态（≈480k个项目）上评估该方法，并发现32%可重建、12%完全可重建、39%部分可重建；

**🔧 技术方法**

主要技术包括：Java/Maven/Gradle构建工具的静态分析、Git仓库的ls‑remote查询、pom.xml 递归解析、MANIFEST.MF与pom.properties元数据提取、正则匹配版本标签、自动化构建脚本（Docker化的RC工具），以及基于文件级MD5对比来判定可重建性；

**📊 数据集**

使用的数据集为约480k个Maven库的随机一次性发布版本（从 Maven Central 的索引文件生成），共410k个带归档包，包含其POM、源码、manifest等；

**📈 对比分析**

对比方法：将AROMA+生成的.buildspec与RC人工维护的.buildspec在100个Maven项目上进行同源构建，并用RC脚本对比生成文件与Maven Central的发布文件；结果显示AROMA+在所有100个项目中均能成功构建，且在99.8%与RC一致；在新发现的可重复库中实现了约5个完全可重建；整体可重复率达到32%（含部分可重建），而RC仅覆盖约1.5%；

**⚠️ 局限性**

主要局限：仅支持Git仓库和Maven/Gradle构建工具；缺少对非JDK语言或其他VCS（SVN、Mercurial）的支持；对极少数复杂多工具或非标准标签的项目无法完全恢复；在部分项目中仍需手动激活发布配置文件（release profile）才能完整复现；此外，未覆盖已失效或无源代码的包，导致可重复性覆盖率受限。

---

## 486. TRACE-CRC: Trajectory-Adaptive Conformal Risk Control for Multi-Step Channel State Information Prediction

**arXiv ID:** 2608.27124 | [PDF](https://arxiv.org/pdf/2608.27124v1)

**作者:** Kiarash Rezaei `[一作]` (Chalmers University of Technology), Carlos Natalino `[通讯]` (Chalmers University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于轨迹感知风险校准的多步无线信道预测可靠性评估框架 TRACE-CRC，能够在保证全程覆盖率的同时降低不确定度。

**💡 创新点**

创新点在于将时间维度的难度曲线分析与轨迹级难度分组相结合，并通过可学习的风险校准学习参数 λ，实现全局、轨迹级与时间段级的动态校准。

**🔧 技术方法**

采用分层共轭预测、可学习风险控制（LRC）、轨迹特征提取、时间段难度曲线分析以及贝叶斯分位数估计等技术。

**📊 数据集**

使用工业规模的 5G mmWave 802.11ad 信道测量数据集，包含 1000 条信道轨迹，每条轨迹 100 个信道样本。

**📈 对比分析**

与传统分层共轭、Bonferroni/Sidak 校准、CopulaCPTS 等方法对比，TRACE-CRC 在 0.933 的全程覆盖率下平均不确定度约为 13.64，明显优于保守的 Bonferroni/Sidak 以及 CopulaCPTS 的覆盖率或误差。

**⚠️ 局限性**

局限性包括对样本量要求较高、需假设校准集与目标集同分布、对非平稳信道的适应性不足以及在极端小样本时可能产生过度保守。

---

## 487. CODE: Cross-Modal Calibration and Dynamic Suppression for Open World Object Detection

**arXiv ID:** 2608.27214 | [PDF](https://arxiv.org/pdf/2608.27214v1)

**作者:** Hao Xu `[一作]` (Beijing Institute of Technology), Bo Ma `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为CODE的开放世界目标检测框架，整合跨模态校准、置信度增强与动态异常抑制。

**💡 创新点**

创新点包括：①引入全局视觉原型进行跨模态联合置信度校准，缓解文本到视觉匹配的语义歧义；②利用局部视觉响应方差量化不确定性，提升未知目标的激活；③采用置信度间距动态抑制，避免硬性阈值过度抑制边缘未知实例。

**🔧 技术方法**

核心技术为多模态基础模型（OWL‑ViT）+属性驱动的注意力映射 + 视觉原型匹配 + 置信度间距动态抑制，全部在推理时完成，无需在线生成。

**📊 数据集**

主要使用 Real‑World Detection (RWD) 基准（Aquatic、Aerial、Game、Medical、Surgery），并在 M‑OWODB 与 LVIS 等公开数据集上验证通用性。

**📈 对比分析**

与PASS、FOMO、BASE 等多种基线对比，CODE在L/14 Backbone下Task 1的U‑mAP提升至21.7点、K‑mAP提升至40.8点，Task 2的PK‑mAP提升至43.6点、CK‑mAP提升至36.2点，整体性能优于现有方法，提升幅度约2–8个百分点。

**⚠️ 局限性**

局限性在于：①需依赖数据集特定的视觉原型，极端样本稀缺、长尾或噪声标签时表现会下降；②对高度重叠的细粒度类别（如Game、Surgery）仍存在识别难度。

---

## 488. Surrounded by Friends: Design and Evaluation of Immersive Layouts of Egocentric Network for Visual Analytics

**arXiv ID:** 2608.27194 | [PDF](https://arxiv.org/pdf/2608.27194v1)

**作者:** Kentaro Takahira `[一作]` (HKUST), Huamin Qu `[通讯]` (HKUST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并在VR环境中评估四种 egocentric 网络布局（Cube、Cylindrical、Radial、Spherical），并通过用户实验验证其可用性与效果。

**💡 创新点**

首次系统性地探讨 VR 中 egocentric 网络的设计维度与布局选择，并提出针对不同分析任务的布局准则。

**🔧 技术方法**

使用 Meta Quest 2 VR 头显，结合 A‑Frame、Three.js、d3.js 实现三维布局、深度编码与交互；通过光线投射、视点切换等交互技术支持用户分析。

**📊 数据集**

合成自合作作者网络的 60 节点数据集（1 个人、30 1st degree、29 2nd degree，约 20% 边密度），用于生成可视化场景。

**📈 对比分析**

24 名受试者完成 28 个任务（4 布局 × 7 任务），使用 Cochran Q、Friedman、Wilcoxon 等非参数统计比较任务准确率、完成时间与移动距离；结果显示 Cube 在强度评估任务上最快最准确，Spherical 在拓扑任务上表现最佳，整体差异显著。

**⚠️ 局限性**

受限于网络规模小、静态、未涵盖动态变化；参与者非领域专家，可能不代表专业分析者；四种布局同时改变多维属性，难以单独归因每个设计维度的影响。

---

## 489. Unsupervised Adaptation of 3D CT Foundation Models for 3D CBCT Segmentation

**arXiv ID:** 2608.27190 | [PDF](https://arxiv.org/pdf/2608.27190v1)

**作者:** Gauthier Miralles `[一作]` (Institut Polytechnique de Paris), Pietro Gori `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种无监督域适配框架，利用冗余减少的特征对齐方法将预训练的3D CT基础模型迁移到3D CBCT肝脏分割，并可同时适用于CNN与ViT结构；

**💡 创新点**

该方法的创新点包括：1) 轻量化且架构无关的特征级域适配策略；2) 引入Barlow Twins式冗余减少对齐/分离损失实现域不变特征；3) 在无目标标注且不需要推理时适配的全流程设计；

**🔧 技术方法**

采用特征提取器ψ、表示头f、对抗头f'、任务头g的模块化结构，使用对抗性冗余减少损失进行特征对齐；并在3D U‑Net骨干上实现；与现有UDA方法（DA‑nnUNet、MDD‑UNet、SIFA‑3D、MAPSeg）以及零射击基础模型（MA‑SAM、Merlin、TotalSegmentator、MedSAM2、VISTA‑3D）进行对比；

**📊 数据集**

使用两个CT‑CBCT肝脏分割基准：①公开的Pancreatic‑CT‑CBCT‑SEG与LiTS（共130 CT + 39 CBCT）②私有临床集合（678 CT + 573 CBCT，包含辐射治疗CBCT（DR）与介入CBCT（DI））；

**📈 对比分析**

在DR与DI两组数据集上，以F1分数为指标，与零射击基础模型和现有UDA方法对比，本文方法在F1均超过70%，明显优于DA‑nnUNet、MDD‑UNet、SIFA‑3D等，同时保持轻量化（参数增量低），显示出显著性能提升；

**⚠️ 局限性**

目前仅在肝脏分割任务上验证，尚需在多器官或更复杂的介入成像任务中进一步评估；对极端伪影和极端剂量变化的鲁棒性仍有限；

---

## 490. When Interference Graphs Evolve: Doubly Robust Estimation of Dynamic Peer Effects

**arXiv ID:** 2608.27187 | [PDF](https://arxiv.org/pdf/2608.27187v1)

**作者:** Xiaojing Du `[一作]` `[通讯]` (Adelaide University), Xiaojing Du (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种针对动态网络干扰的可双重稳健估计器 DynaNet-DR，用以估计包含自我处理、同伴暴露、网络演化三种维度的受控对比。

**💡 创新点**

创新点在于：①将网络演化摘要视为受控干预的独立维度；②构造时间衰减的动态暴露映射并提供可解释的离散暴露水平；③在估计中结合时间序列分层的倾向分解和归一化增广，理论上实现双重稳健性。

**🔧 技术方法**

使用技术包括：梯度提升机的多时序特征学习、时间分层交叉训练、代表分数插补、联合倾向分解、归一化增广（DynaNet-DR），以及节点层聚类的标准误估计。

**📊 数据集**

数据集：半合成基准（CollegeMsg、email-Eu、PrimarySchool、HighSchool 2012）以及实际观测案例 MathOverflow；全部基准保留真实时间序列结构并生成处理、暴露、演化标签与结果。

**📈 对比分析**

方法比较：与 TL、GML-DR、DynInt、静态暴露回归等多种基线对照；DynaNet-DR 在所有数据集和对比族中均取得最低均方根误差，并在节点层聚类置信区间上达到 0.91–1.00 的覆盖率，表现优于现有方法。

**⚠️ 局限性**

局限性：①对正则化、重叠与无遗漏同质性等强假设依赖；②未对网络演化进行反事实边生成，无法评估具体干预策略；③在跨节点依赖下的理论收敛仍未证明；④对隐藏同质性敏感，且仅在观测数据中作经验性说明。

---

## 491. Task-space model-based control of pneumatic soft actuators

**arXiv ID:** 2608.27186 | [PDF](https://arxiv.org/pdf/2608.27186v1)

**作者:** Nithin S. Kumar `[一作]` (Vanderbilt University), Eric J. Barth `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实验验证了一种基于非最小坐标离散弹性杆模型的实时任务空间控制框架，用于平面软气动连续体执行器。

**💡 创新点**

将非最小坐标约束Lagrange弹性杆模型与稀疏结构相结合，实现了10段离散杆的实时动力学控制，突破了传统PCC模型的稀疏约束与动力学可观测性局限。

**🔧 技术方法**

使用约束Lagrangian离散弹性杆建模、Baumgarte稳定化、稀疏块三角线求解、动态观测器、任务空间PI控制以及虚拟力注入等技术。

**📊 数据集**

实验使用三种不同几何的平面软气动执行器P1、P2、P3，并在五个任务场景下收集了实时姿态、压力、测距数据。

**📈 对比分析**

与传统PCC、学习型控制和Koopman模型在同一平台对比，取得1.5–2.3 mm RMSE的慢速跟踪和5.5–12.4 mm RMSE、1–2 Hz的周期跟踪，明显优于现有方法。

**⚠️ 局限性**

仅局限于平面运动、固定参数、无接触力估计，且对材料疲劳与非平面扭转动力学缺乏适应性。

---

## 492. A Trans-Domain Digital Twin for Bio-Aware Control of Climate and Energy in Cattle Fattening Barns Using Single-Episode Optimizer Learning

**arXiv ID:** 2608.27185 | [PDF](https://arxiv.org/pdf/2608.27185v1)

**作者:** Mansoorali Amiri `[一作]` `[通讯]` (University of Montreal), Mansoorali Amiri (University of Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种单剧集学习的跨域数字孪生框架（SE‑TDDT），实现闭式肉牛育肥场内气候、动物成长、能源与饲料响应的统一可执行控制。

**💡 创新点**

创新点在于将机理气候模拟、Beef‑LiGAPS生长模型与MPC结合，并通过多速率循环、单剧集学习、知识记忆（CCLL‑SEL、SARG‑SEL、SS‑KStore、SETD‑KStore）实现气候决策与生物反馈的双向耦合。

**🔧 技术方法**

采用的技术包括：1‑区机理气候模拟器、Beef‑LiGAPS生长模型、基于模型的预测控制（MPC）、轻量级强化学习调优、结构化知识记忆以及边缘单板计算平台（Orange Pi 5）。

**📊 数据集**

使用的数据集包括11年（2015–2025）OpenWeather气候数据、1000天的模拟育肥过程（气候、饲料、成长记录）以及对应的生物学参考路径，用于构建CCLL与SARG记忆。

**📈 对比分析**

通过与四种基线（简单HVAC、气候MPC、TDDT无记忆、完整SE‑TDDT）对比，性能指标显示：舒适度100%，生长准确率96.9%，能耗准确率92.6%，但饲料消耗准确率仅51%。

**⚠️ 局限性**

主要局限包括饲料压力控制不足、执行器切换频繁导致的波动、仅使用单一‑区气候模型、缺少气体动力学建模以及验证仍停留在仿真单剧集阶段。

---

## 493. LLMs in Digital EDA: A perspective on shifting roles from Generation to Orchestration

**arXiv ID:** 2608.27184 | [PDF](https://arxiv.org/pdf/2608.27184v1)

**作者:** Matthew Youngman `[一作]` (University of Edinburgh), Themis Prodromakis `[通讯]` (University of Edinburgh)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并提出LLM在EDA中的三层架构（生成器、代理、编排器），指出语法陷阱与碎片化问题，并建议标准化编排实现可靠可扩展的硬件设计。

**💡 创新点**

重新定义LLM角色层次，引入物理感知的“编排器”概念，强调架构而非模型改进，提出统一接口与持久状态解决碎片化、无状态与可解释性问题。

**🔧 技术方法**

使用大语言模型（LLM）技术，检索增强生成（RAG）、上下文学习、强化学习、工具集成、标准化协议（MCP、A2A）、物理代理模型及可追溯性数据库。

**📊 数据集**

引用多篇研究中使用的公开与私有硬件设计数据、Synthesis、Timing、PPA数据，主要基准为RTLLM、VerilogEval等；论文未单独标明特定数据集。

**📈 对比分析**

通过在规格到RTL生成任务上进行五维评估（前端正确性、后端PPA、验证完整性、求解时间、成本），发现生成器与代理在模块级达人类水平但后端PPA差距大，编排器在系统级保持较高正确性并超越人类PPA，整体提升15‑30%时序、10‑20%面积。

**⚠️ 局限性**

存在语法陷阱导致物理正确性不足、跨阶段上下文缺失、可解释性不足；编排器成本高、普及率低，LLM对大规模设计的上下文失效，工业化落地仍受限于专有PDK与制造资源。

---

## 494. Parameter-Efficient pretrained-CT-to-MRI Transfer for Rectal Cancer Segmentation: Performance-Calibration Trade-offs

**arXiv ID:** 2608.27178 | [PDF](https://arxiv.org/pdf/2608.27178v1)

**作者:** Aneesh Rangnekar `[一作]` (Memorial Sloan Kettering Cancer Center), Harini Veeraraghavan `[通讯]` (Memorial Sloan Kettering Cancer Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了SWIFT系列模型，利用预训练的CT Swin Transformer在MRI上进行直肠癌分割，并探索参数高效与校准的权衡。

**💡 创新点**

引入多阶段参数高效微调（EffiDec、LoRA、四成员LDE4集成）和肿瘤感知增强，系统评估检测率、几何准确性、放射组学一致性和概率校准之间的平衡。

**🔧 技术方法**

使用Swin V2自监督预训练（DINOv2）、EffiDec3D解码器、LoRA适配器、四成员LDE4集成、温度缩放校准以及肿瘤感知增幅等技术。

**📊 数据集**

使用416例T2加权MRI（1.5/3T GE）数据集，划分为136例训练、33例验证和247例测试。

**📈 对比分析**

与全微调SWIFT对比：SWIFTe参数减少70%仍提升检测率至93.9%；SWIFTe-LoRA仅3.2M可训练参数保持相同sDSC；SWIFTe-LDE4在保持sDSC的同时将ECE降至0.217，Brier 0.222。

**⚠️ 局限性**

单中心单设备（仅GE扫描）数据；未验证跨机构或多厂商通用性；未进行临床试点或不确定性校准对手术/剂量影响的定量评估。

---

## 495. Prediction of Prediction (PoP): Inter-Layer Activation Fusion for Single-Pass Hallucination Detection in Large Language Models

**arXiv ID:** 2608.27165 | [PDF](https://arxiv.org/pdf/2608.27165v1)

**作者:** Himal Badu `[一作]` `[通讯]`, Himal Badu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在自回归LLM生成过程中通过捕获隐藏层状态的过渡动态，实时检测幻觉；

**💡 创新点**

提出Prediction of Prediction（PoP）——利用相邻层间余弦差异结合跨层注意力融合，得到单前向传递的幻觉风险分数；

**🔧 技术方法**

采用层归一化、cosine差异、跨层注意力、Temporal Drift、两层MLP和Platt校准等技术；

**📊 数据集**

在TruthfulQA、HaluEval 2.0、FaithDial等英文QA与对话数据集上评估；

**📈 对比分析**

与输出熵、困惑度、静态/动态内部探测、语义熵及后端NLI相比，PoP在TruthfulQA上达到75.5% AUROC，仅增加1.2%运行时延迟，零额外生成调用；

**⚠️ 局限性**

需白盒访问、激活内存开销、仅限英文闭卷问答，且为预测关联非因果，无法直接应用于闭源API。

---

## 496. Inductive Correlation Clustering with Graph Neural Networks

**arXiv ID:** 2608.27153 | [PDF](https://arxiv.org/pdf/2608.27153v1)

**作者:** Francesco Paolo Nerini `[一作]` (Sapienza University of Rome), André Panisson `[通讯]` (Intesa Sanpaolo AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可迁移的关联聚类（Inductive Correlation Clustering）框架，利用图神经网络（GNN）在未见图上直接求解聚类问题，并将其作为可学习的图池化层。

**💡 创新点**

创新点：①首次将关联聚类问题表述为可学习的持续优化任务，支持无监督训练；②设计了两种GNN实现（节点级与边级）并引入基于pivot的批量采样以实现可扩展性；③将此方法嵌入图池化，取得与现有稠密/稀疏池化器相当甚至更优的分类性能。

**🔧 技术方法**

主要技术：图卷积网络（GCN）/图SAGE、节点/边归一化、对齐-统一的损失函数、pivot采样策略、阈值自适应与连通分量聚类；实验中使用了单层GCN实现以保持参数量低。

**📊 数据集**

使用的数据集：标准关联聚类基准（polblogs、ca-GrQc、ca-HepTh、ca-AstroPh、email-Enron、cond-mat-2005）；图级任务与池化基准（EXPWL1、MUTAG、NCI1、REDDIT-BINARY、GitHub-Stargazers、OGBG-molhiv、OGBG-ppa）；用于池化的子集（EXPWL1、MUTAG、NCI1、OGBG-molhiv）。

**📈 对比分析**

与基准比较：在标准关联聚类任务中，节点级CC‑GNN几乎总能得到最小成本，优于KwikCluster、ModifiedPivot和CFP；在无监督的迁移设置下，边级LinkGNN在大部分数据集上超越KwikCluster且推理速度提升5–6个数量级；作为池化器时，CCPool在多种图分类任务中获得了平均排名2.88，显著优于或与现有最强池化方法竞争。

**⚠️ 局限性**

局限性：①节点级方法需设定上限K，若K不足会导致近似误差显著；②边级方法对阈值选择敏感，且在极大图上仍需采样才能保持可训练性；③当前实现仅支持无监督单目标，未覆盖重叠聚类或多目标约束；④对图属性分布的泛化依赖Node2Vec等预训练特征，若原始特征信息缺失或噪声较大，性能可能下降。

---

## 497. GRAIN: Bridging Name and Narrative Shifts in Real-World Graph Reasoning through Invariance-Rewarded Agentic RL

**arXiv ID:** 2608.27142 | [PDF](https://arxiv.org/pdf/2608.27142v1)

**作者:** Zike Yuan `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种单代理强化学习框架 GRAIN，提升 LLM 在图推理任务中的鲁棒性与效率。

**💡 创新点**

创新点在于引入结构不变奖励并通过多样化等价命名与叙述训练，使模型学习图结构而非表面形式，同时使用单代理 RL 取代多代理系统。

**🔧 技术方法**

使用语言‑图‑工具管线、结构不变奖励、ARPO 强化学习、工具调用与语义解析等技术。

**📊 数据集**

使用新构建的 GRIT 基准，包含 31 个场景、六个图任务、八种视角命名与叙述，覆盖规模 40 以内与大规模图。

**📈 对比分析**

与零射程 CoT、工具 CoT、SFT 以及多代理 MA‑GTS 对比，GRAIN 在 GRIT 上实现 95.8% 宏平均准确率，较多代理提升 16.5% 且延迟下降 24%，同时在 OOD 与大规模图上显著提升。

**⚠️ 局限性**

局限在工具上下文管理导致提示长度过大，且未完全解决多工具动态加载与上下文窗口限制问题。

---

## 498. From Security Events to Conflict States: A Three-layer Cyber Defense Scenario Model for Enhanced Cyber Situational Awareness

**arXiv ID:** 2608.27215 | [PDF](https://arxiv.org/pdf/2608.27215v1)

**作者:** Miguel Requena Micó `[一作]` (Universidad de Murcia), Gregorio Martínez Pérez `[通讯]` (Universidad de Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个三层概率模型与可执行的NetLogo原型，实现从事件观测到冲突状态再到任务风险的统一推理与决策支持。

**💡 创新点**

创新之处在于将攻击图、贝叶斯事件推理与冲突状态抽象三层融合，提供即时一阶决策规则，兼顾不确定性与任务风险。

**🔧 技术方法**

采用攻击图+贝叶斯网络推理+阈值状态映射等技术，辅以NetLogo agent-based 模拟实现贝叶斯更新与决策。

**📊 数据集**

使用自定义的18节点TEZ/MOZ/ESZ拓扑和对应的事件集E1–E7，相关代码与数据已发布在GitHub。

**📈 对比分析**

通过单次自动执行轨迹与手动/自动策略对比，展示冲突状态与任务风险随时间上升的动态；但未做大规模统计实验或与现有方法对比。

**⚠️ 局限性**

验证仅为示例性场景，缺乏大规模随机实验、真实攻击/防御映射，自动决策采用启发式规则而非全局最优。

---

## 499. PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference

**arXiv ID:** 2608.27206 | [PDF](https://arxiv.org/pdf/2608.27206v1)

**作者:** Junjie Liu `[一作]` (Sun Yat-sen University), Xu Chen `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一个无训练、可插拔的推理框架 PACE，通过先预先压缩图像再提取重要视觉标记，显著加速视觉语言模型的推理。

**💡 创新点**

创新点在于：① 统一的 Condense‑and‑Extract 两阶段，①1 先用 Adaptive Pixel Compressor (APC) 在视觉编码前根据信息密度自适应降采样，②1 使视觉编码和 LLM 预填充两侧都受益；② ② 用 Dynamic Dual‑Attention Extractor (DDAE) 将 LLM 的语义注意力与 ViT 的自注意力动态融合，提升细节保持能力；③ ③ 兼容多种后置剪枝方法，训练无须额外监督。

**🔧 技术方法**

关键技术包括：轻量级的特征预览（浅层 ViT 前向一次）、全局与局部信息密度评估、比例自适应重采样、标准差驱动的注意力权重融合、以及基于预算的动态 token 选取。

**📊 数据集**

在 Qwen2.5‑VL‑7B（3B/7B）以及 InternVL3.5‑4B 上进行实验，使用 9 个跨模态基准：MME、POPE、MMBench、MMStar、RealWorldQA、TextVQA、DocVQA、ChartQA、OCRBench。

**📈 对比分析**

与 FastV、SparseVLM、DivPrune、DART、VisionZip、MMTok 等主流视觉 token 剪枝方法对比，PACE 在 10% 视觉 token 保留时保持 93.8% 原性能，TTFT 速度提升 3.1×；在 5% 预算下仍保持显著优势，尤其在细节敏感任务（OCRBench、ChartQA、DocVQA）上超越对手数个百分点。

**⚠️ 局限性**

局限性：① APC 的加速效果依赖于像素/图块预算下降，固定格网 VLM 无法获得编码端加速；② 预览仅一次、无查询感知，可能丢失细小文本、薄线条等高频细节，需更保守的保留阈值或查询触发的局部恢复策略。

---

## 500. STAR : Sentence Translation Alignment Rate for Document-to-Document Machine Translation

**arXiv ID:** 2608.27161 | [PDF](https://arxiv.org/pdf/2608.27161v1)

**作者:** Yichen Dong `[一作]` (Soochow University), Weihua Luo `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了句级对齐率（STAR）度量和STAR掩蔽偏好优化（StarPO）框架，用于提升文档级机器翻译（Doc2Doc）的句子结构一致性和整体质量；

**💡 创新点**

创新点在于引入句级结构忠诚度量STAR，并通过动态句子掩蔽将对齐错误聚焦于偏好优化，从而显著减少遗漏与虚假生成；

**🔧 技术方法**

技术上结合了长上下文LLM、句子分割工具SaT、BERT对齐器、对比偏好优化（CPO）与STAR掩蔽损失；

**📊 数据集**

使用News‑Commentary和Guofeng（WMT 2025）中英文、德文、俄文、西班牙文等多语种文档；

**📈 对比分析**

与基线SFT、标准CPO以及Tower‑plus-9B、GPT‑4o、DeepSeek‑R1等系统比较，StarPO在dCOMET与d‑BLEU上平均提升约0.5‑1.0分，甚至在多语种上超越大模型；

**⚠️ 局限性**

局限包括对句子对齐的依赖、1‑to‑1约束可能抑制合法重排、仅在4B‑9B规模模型和中高资源语言上验证、初始候选生成依赖商业API。

---

## 501. AgentDV: Closed-Loop Agentic AI for Hardware Design Verification

**arXiv ID:** 2608.27148 | [PDF](https://arxiv.org/pdf/2608.27148v1)

**作者:** Navya Goli `[一作]` (Clemson University), Umamaheswara Rao Tida `[通讯]` (Clemson University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 AgentDV，一个闭环的 LLM 辅助 RTL 验证环境生成框架。

**💡 创新点**

结合 CSR 基准检查、可运行性过滤与覆盖驱动迭代，形成从 LLM 生成到模拟验证闭环的完整流程。

**🔧 技术方法**

采用 Verilator、cocotb、pyUVM、LangGraph 与 Claude、Llama、Qwen 等 LLM 进行自动化生成、可运行性检查与覆盖反馈。

**📊 数据集**

在 ISQED 2026 挑战的 5 个单文件 DUT 以及公开 OpenTitan 的 4 个 IP 作为评估基准。

**📈 对比分析**

通过对比单次提示与 AgentDV 流程的通过率与覆盖率，Claude 在所有 DUT 上平均 80.9% 通过率、76.9% 行覆盖、95.7% 分支覆盖，Llama、Qwen 约 58–61% 通过率，显示闭环迭代显著提升。

**⚠️ 局限性**

对需要完整协议/数据路径触发的加密 IP（AES/HMAC），仅基于 CSR 的刺激难以覆盖深层数据通路，导致覆盖饱和与通过率下降。

---

## 502. Safety Does Not Compose: Non-Decaying Loop State for Autonomous LLM Agents

**arXiv ID:** 2608.27141 | [PDF](https://arxiv.org/pdf/2608.27141v1)

**作者:** Chenhao Wu `[一作]` (University of Chinese Academy of Sciences), Bin Chong `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoopHarness，一种在自主循环层面持续保留安全状态、检测并限制攻击的框架；并证明在单轨迹防御无法识别分散攻击、几何衰减风险计数器不足以防御的理论结论。

**💡 创新点**

创新点包括：①将安全状态从单轨迹迁移到循环层面，实现跨迭代风险累积与锁定；②引入非衰减风险累积器（XRC‑latch）和结构性阈值锁定，确保即使攻击碎片化也能被捕获；③提供期望授权违规次数的无模型、与迭代长度无关的上界；④设计了内存完整性保护、强健停止仲裁器和风险治理器，使组件可独立组合。

**🔧 技术方法**

使用技术：持续的哈希链日志与校验点；跨迭代风险累积器与锁定机制；基于规则的授权代理、可信计数器与可计数器；高级与廉价两级模型判定（decorrelated checker）；多层安全阈值与可回溯的风险预算；以及严格的实验评估协议。

**📊 数据集**

数据集：Agent‑SafetyBench 中挑选的 200 条可执行工具任务（含 2,000 条原始任务中的可执行子集），构成基准任务池；通过冻结随机种子和内容哈希保证可重复的任务和攻击场景。

**📈 对比分析**

比较方法：对比五种基本配置（B0–B4）以及单模块剔除（A‑no*）和协作检验（A‑collude）等；使用 ASR（攻击成功率）、CleanGC（正常完成率）和期望未授权不可逆操作数作为指标；结果显示 LoopHarness（B4）将 ASR 降至约0.1%（相较 88–97% 的基础配置），且 CleanGC 仍保持 96–97%，证明在保持性能的同时大幅提升安全性。

**⚠️ 局限性**

局限性：①安全检测门槛 δ_M 为经验参数，若估计偏低会削弱保证；②XRC‑latch 需手动授权清除，缺乏自动化的安全阈值回归机制；③缺乏对多代理循环的直接支持；④对升级和人机协作的上报与升级路径假设尚未实现，若缺失会导致可用性下降。

---

## 503. TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy

**arXiv ID:** 2608.27128 | [PDF](https://arxiv.org/pdf/2608.27128v1)

**作者:** Hong Chen `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对长上下文推理中的 KV 缓存淘汰瓶颈，提出了一种无关注、无训练的冗余信号 TwinKV，用于检查已被淘汰或保留的 token 是否存在近似重复，从而在保持原有淘汰策略的预算不变的前提下，修复冗余保留与信息丢失的错误。

**💡 创新点**

核心创新是：①证明注意力权重与 token 对答案的真实因果贡献几乎无关；②设计基于 key 向量余弦相似度的冗余计数器 TwinKV，完全不依赖注意力或训练；③将 TwinKV 作为轻量级“修复通道”与任意已有淘汰策略可组合，保持原预算并提升性能。

**🔧 技术方法**

技术包括：离散化 key 余弦相似度计算、局部窗口排除、阈值阈值 τ 调整、旋转不变性修正（针对非均匀 RoPE）；修复步骤基于双向双亲匹配（orphan 与 redundant donor）并交换；实验中使用 O(n²d) 复杂度的相似度矩阵。

**📊 数据集**

评估数据集涵盖四个主流长文本基准：LongBench（16 任务）、LooGLE（4 任务）、RULER（13 任务，长度 4K/8K/16K）、MMLU-Pro（短上下文对照）。

**📈 对比分析**

与四种主流淘汰策略（StreamingLLM、PyramidKV、SnapKV、ExpectedAttention）在 3 个压缩比例 {0.3,0.5,0.7} 上进行对比。结果显示：在 Qwen3‑4B 上，TwinKV 在 StreamingLLM、PyramidKV、SnapKV 中多数配置提升，ExpectedAttention 由于已接近上限而往往略降；在 Llama‑3.2‑1B 上，TwinKV 对所有策略普遍有益，尤其在 RULER 上带来显著提升。整体而言，平均提升在 0.3–0.7 比例下约为 +0.4 至 +0.7 分，且在多数任务中保持正向改进。

**⚠️ 局限性**

局限性包括：①需要额外 O(n²d) 的相似度计算，导致预填充开销增大；②对阈值 τ 与窗口大小 w 的选择敏感，需根据模型结构手动校准；③在极低冗余或极短上下文（如 MMLU‑Pro）下可能产生负面影响；④对复杂位置编码（非均匀 RoPE）需特殊处理，增加实现复杂度。

---

## 504. Profit based evaluation of machine learning for nitrogen recommendations in winter wheat

**arXiv ID:** 2608.27205 | [PDF](https://arxiv.org/pdf/2608.27205v1)

**作者:** Xulong Wang `[一作]` (University of Sheffield), Po Yang `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并改进冬小麦氮肥推荐方案，以利润为评分标准。

**💡 创新点**

将机器学习与后置校正结合，基于实际产量曲线按利润评分，而非传统误差。

**🔧 技术方法**

使用多种机器学习模型（岭回归、随机森林、高斯过程、TabPFN等）并在其输出上实施校正与加权。

**📊 数据集**

利用英国罗斯马斯特德Broadbalk（402条测定产量曲线）和Woburn（490条）长期试验数据。

**📈 对比分析**

与标准AHDB RB209建议对比，单纯的机器学习在利润上表现不佳，但后置校正后可将利润损失降低约24%，且在未见训练站点的Woburn实现43%的提升，整体收益改进有限。

**⚠️ 局限性**

主要限制包括仅在单一站点训练、利润改进统计显著性不高、未考虑应用成本与蛋白质溢价、缺乏现场验证及对不同气候土壤的泛化性未知。

---

## 505. Knowledge Distillation Driven Semantic NOMA with GAN Refinement for 6G Robotic Vehicle Networks

**arXiv ID:** 2608.27198 | [PDF](https://arxiv.org/pdf/2608.27198v1)

**作者:** Qifei Wang `[一作]` (Beijing Institute of Technology), Ying Sun `[通讯]` (Beijing Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种KDG‑SemNOMA框架，用于6G机器人车辆网络中的多用户语义NOMA通信；

**💡 创新点**

创新点在于：①将ConvNeXt网络与基于信道状态的增强注意力模块相结合；②采用两阶段知识蒸馏（教师为OMA模型、学生为NOMA模型）以在不增加推理开销的情况下抑制多用户干扰；③设计了基于信道条件的cGAN对初步重构进行高频纹理恢复，避免像素级优化导致的模糊。

**🔧 技术方法**

使用的技术包括深度联合源信道编码（DeepJSCC）、ConvNeXt骨干网络、增强注意力模块、特征亲和度蒸馏、交叉头蒸馏、条件GAN（U‑Net生成器+PatchGAN判别器）、MAE、LPIPS、Wasserstein对抗损失等。

**📊 数据集**

实验数据集为FFHQ‑256（人脸高分辨率图像，降采样至256×256）。

**📈 对比分析**

与BPG+LDPC+QAM+SIC、DeepJSCC‑NOMA以及SemOMA（正交基准）进行对比，采用PSNR、LPIPS和FID指标评估；KDG‑SemNOMA在AWGN和Rayleigh信道下均比DeepJSCC‑NOMA高约0.3‑0.5 dB，Rayleigh下可达1‑1.5 dB，LPIPS/FID显著下降，且在低SNR时仍保持可辨认的重构，无“阶跃效应”。

**⚠️ 局限性**

局限性包括：需要准确的CSI作为条件；模型训练与推理仍较为复杂，可能难以在实时硬件上高效部署；实验仅在合成数据集上验证，缺乏真实网络环境的进一步验证。

---

## 506. BPMN4CAI: A BPMN Extension for Modeling Dynamic Conversational AI

**arXiv ID:** 2608.27149 | [PDF](https://arxiv.org/pdf/2608.27149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 507. Twelve Quick Tips for Managing IT Disasters in Small Research Software Teams

**arXiv ID:** 2608.27196 | [PDF](https://arxiv.org/pdf/2608.27196v1)

**作者:** Greg Wilson `[一作]` `[通讯]`, Greg Wilson

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了12条快速提示，帮助小型研究软件团队在缺乏专业系统管理员的情况下管理IT灾难。

**💡 创新点**

创新点在于将灾难管理拆分为可执行的、易于理解的提示，并专门针对资源有限的科研软件团队设计。

**🔧 技术方法**

使用的技术包括Markdown/Google Docs文档管理、云服务备份与自动快照、双重存储、MFA、多因素身份验证、密码管理器、监控工具（如Uptime Robot、Healthchecks.io）以及自动化镜像和恢复脚本。

**📊 数据集**

未使用特定数据集，提示基于假设场景和实际操作经验。

**📈 对比分析**

本文未进行方法比较或性能评估，主要以经验性建议和实用指南形式呈现。

**⚠️ 局限性**

局限性包括缺乏实验或案例验证，提示的适用性需根据各团队的具体环境调整；并且在极端高风险场景下可能不足以覆盖所有灾难情形。

---

## 508. SSMB: Self-Supervised Local Feature Detection under Motion Blur

**arXiv ID:** 2608.27181 | [PDF](https://arxiv.org/pdf/2608.27181v1)

**作者:** Zhenjun Zhao `[一作]` (University of Zaragoza), Javier Civera `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督、无抖动去模糊的关键点检测器SSMB，能够在运动模糊图像中直接检测可重复的局部特征。

**💡 创新点**

创新点在于引入Local Discriminability Enhancement (LDE)模块恢复全局MLP混合后细粒度局部可区分性，并采用两阶段自监督训练（几何预训练 + 模糊感知训练）与多组件损失（同域一致性、位置一致性、空间多样性等）实现对运动模糊的鲁棒性。

**🔧 技术方法**

使用基于MAXIM的多轴门控MLP编码器、LDE模块、在线同域自监督损失（Homographic Adaptation、Blur Consistency、Position Consistency、Spatial Diversity）以及传统的Softmax概率头和子像素偏移回归。

**📊 数据集**

在合成几何图形、GoPro真实锐模对、Blur-HPatches、Deblur-HPatches、ArchViz、Aachen Day-Night 等公开数据集上进行训练与评估。

**📈 对比分析**

与包括传统手工检测器（SIFT、DoG）、学习型检测器（SuperPoint、BALF、D2-Net、R2D2 等）以及稠密匹配方法（LoFTR、MatchFormer、RoMa v2）等基线比较，SSMB在关键点重复率、图像匹配 MMA、相对位姿 AUC、视觉定位准确率等多项指标均显著优于所有稀疏基线，并在部分任务上超过稠密匹配方法。

**⚠️ 局限性**

主要局限在于对高强度模糊的极端情况仍有一定性能衰减，训练需要较大的两阶段数据集且对GPU内存消耗较高，且缺乏针对不同相机模型的泛化性验证。

---

## 509. Magpie: Real-Time World Renderer for Interactive Games

**arXiv ID:** 2608.27168 | [PDF](https://arxiv.org/pdf/2608.27168v1)

**作者:** Xiaoyu Zhan `[一作]` (Mogo AI Ltd), Dongjie Fu `[通讯]` (Mogo AI Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Magpie 通过把游戏引擎的玩法执行与生成模型的视觉渲染分离，实现了可实时生成的游戏渲染系统。

**💡 创新点**

创新点在于将玩法逻辑保留在引擎中，只将白盒帧作为生成模型的结构条件，从而保证玩法可重复与可调试，同时通过文本和首帧图像实现多样化视觉风格。

**🔧 技术方法**

使用了基于扩散模型的 Wan2.2‑TI2V‑5B 生成器，结合跨注意力条件注入、Helios 级联生成、LightTAE 自动编码、FP8 低精度推理以及分块自回归推理。

**📊 数据集**

训练数据来自约 300 小时在 Unreal Engine 场景下人工操作收集的高保真与白盒对齐视频，包含 1920×1080@60FPS 的同步帧、摄像机姿态和结构化交互记录。

**📈 对比分析**

通过与单 GPU 上的 H100 推理基准比较，生成器在 1280×768 分辨率下稳定输出约 32.2 FPS，端到端响应时间约 1.55 秒，显示出比传统光栅化管线更低的人工资产需求。

**⚠️ 局限性**

主要限制包括 1.6 秒的交互延迟、对白盒条件的深度歧义导致生成不一致、缺乏音频同步、有限的三维视觉记忆、推理成本高且缺乏边缘部署能力。

---

## 510. When Tool Outputs Become Commands: Separating Action Induction from Runtime Authorization in Tool-Augmented LLM Agents

**arXiv ID:** 2608.27146 | [PDF](https://arxiv.org/pdf/2608.27146v1)

**作者:** Xiaokun Guo `[一作]` (Chinese Academy of Sciences), Yu Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SARA机制，分离工具调用中的行动诱导与执行授权，降低间接提示注入导致的未授权工具执行。

**💡 创新点**

在观察侧使用上下文隔离的行动探测器记录行动起源，在执行侧根据用户授权、行动起源与已验证执行证据实现动态授权，防止历史重现导致权限升级。

**🔧 技术方法**

结合上下文隔离行动探测、持久行动起源追踪、审核执行证据、参数支持门限及无历史提升等运行时授权技术。

**📊 数据集**

在AgentDojo（92正常+3528攻击）和AgentDyn（141正常+5202攻击）两个基准集上进行评估。

**📈 对比分析**

与PI Detector、Spotlighting、Tool Filter、IPIGuard、CaMeL、MELON、AttriGuard、ClawGuard、AIRGuard等多种基线对比，SARA在四个主要评估场景下ASR≤0.63%，UA不低于Agent-only，并在不同Agent骨干上保持一致性，额外推理成本可接受。

**⚠️ 局限性**

仅为运行时授权机制，缺乏正式安全保证；依赖Agent的重规划能力；评估仅覆盖工具调用的间接提示注入，未考虑绕过授权或纯数据依赖攻击。

---

## 511. Said Aloud, Read Different: Cross-Modal Instability in Multimodal Models

**arXiv ID:** 2608.27135 | [PDF](https://arxiv.org/pdf/2608.27135v1)

**作者:** Basel Mousi `[一作]` (Hamad Bin Khalifa University), Nadir Durrani `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并公开了一个含语音与文本输入、涵盖英语与阿拉伯语的对比三元组视觉基础模型基准，并研究了多模态模型在语音与文字、两种语言下的决策一致性。

**💡 创新点**

创新点在于提出对比不稳定性（Contrastive Instability）度量，用三元组结构评估模型在相同语义信息下跨模态、跨语言的内部判别一致性；并首次将语音合成与文化多样的图像三元组结合，形成可持续的对比评测框架。

**🔧 技术方法**

使用了多模态基础模型（Qwen2.5、Qwen3、Phi-4）以及零样本语音合成、机器翻译、对话式评估方法，并通过贪婪解码、vLLM-Omni等技术实现高效评测。

**📊 数据集**

数据集为 10,150 张来自 18 个中东北非国家的文化主题图像，每张图像对应一个视觉支持陈述和两条可疑但不支持的替代陈述，构成对比三元组；数据已公开发布。

**📈 对比分析**

通过在文本、语音、英语、阿拉伯语四种条件下计算 Q⁺/Q⁻ 准确率、F1 以及对比不稳定性 CI，发现语音输入显著提高 CI（尤其是阿拉伯语），模型规模增大可降低 CI，但无法完全消除跨模态/跨语言的不一致。

**⚠️ 局限性**

局限性包括：依赖合成语音与机器翻译的质量；CI 仅捕捉内部判别不一致，未涵盖所有错误类型；对其他语言、任务或更真实噪声环境的泛化仍待验证。

---

## 512. SLIDE: Shuffle Shamir Secret Shares Uniformly with Linear Online Communication and Guaranteed Output Delivery

**arXiv ID:** 2608.27129 | [PDF](https://arxiv.org/pdf/2608.27129v1)

**作者:** Jiacheng Gao `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计了两种统一性高、在线通信量为O(nml)且支持保证输出交付的Shamir秘密共享洗牌协议。

**💡 创新点**

创新点在于提出一种利用小Permutation矩阵分解的Permutation共享技术，并将洗牌相关性与保证输出交付结合，实现线性在线复杂度与完全统一洗牌。

**🔧 技术方法**

核心技术包括Shamir秘密共享、Beneš网络分解、二进制校验与多路复用、洗牌相关性、争议控制与身份认证标签。

**📊 数据集**

评估使用合成数据，规模从16到262,144个向量，向量长度为1，在128位素数域上进行MPI模拟。

**📈 对比分析**

与之前的基于加法共享或排序/交换网络的协议相比，本方案在线通信减少到约4 MB（低于1100 MB），轮数固定为6，在线延迟不到1 s，整体成本显著下降。

**⚠️ 局限性**

局限性包括对m、k为2的幂的要求、需要诚实多数且仅适用于静态攻击，离线通信在k较小或n较大时仍可能较高。

---

## 513. TransMeme: A Multi-Agent Framework for Cross-Cultural Meme Transcreation

**arXiv ID:** 2608.27127 | [PDF](https://arxiv.org/pdf/2608.27127v1)

**作者:** Jingyi Zheng `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Xinlei He `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多智能体框架TransMeme，用于跨文化表情包（meme）转写与再创造，能够同时处理文本与图像的意图保持、文化适配与多模态一致性。

**💡 创新点**

核心创新在于：①把跨文化转写拆解为文化规划、文本重写与多模态审核的三个专用智能体；②通过直接偏好优化（DPO）训练重写器，提升幽默与语调保持；③引入批判（Critic）迭代修正机制，显著提升图文一致性。

**🔧 技术方法**

技术手段包括：多智能体协同架构、文化适配规划、基于LLM的文本重写与DPO微调、视觉与文本一致性评估与修正循环、以及最终的视觉执行层。

**📊 数据集**

使用1,000条中英双向样本（500条ZH→EN，500条EN→ZH）构建的表情包转写数据集；重写训练使用从网络收集的中英俚语与口语表情包的人工排序语料。

**📈 对比分析**

与三类基线对比：单一LLM一次性转写、结构化单体代理和参考基线。人类评测显示TransMeme在四个维度（意图、文化、图文一致性、表达质量）均显著优于所有基线，平均得分4.122；LLM评测Top‑1匹配率达60%，远高于第二名26%。

**⚠️ 局限性**

主要局限在：①幽默重建仍有改进空间，尤其是跨文化幽默的创新表达；②图文不一致仍出现约27%的情况；③对极少数文化知识缺口表现不佳，提示对特定文化细节的更深挖掘需求。

---

## 514. EditaLive! Unified Character Video Editing for Live Streaming

**arXiv ID:** 2608.27123 | [PDF](https://arxiv.org/pdf/2608.27123v1)

**作者:** Zhiyuan Li `[一作]` (University of Macau), Xiaodong Cun `[通讯]` (Great Bay University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向实时直播的角色视频编辑框架，能够通过文本指令实时修改角色外观，同时保留运动和表情一致性。

**💡 创新点**

将角色外观与运动解耦，采用参考帧编辑+视频重建的重建式训练，构建 CharEdit-50K 数据集；将离线双向编辑模型改造为分块因果流式生成，并通过对齐自回归蒸馏、固定 RoPE 与首帧保留稀疏注意力三项技术实现两步采样与长时序稳定性。

**🔧 技术方法**

利用 Wan-Animate 预训练动画模型、VAE+flow‑matching 损失、LoRA 微调、GPT‑5.5+Qwen‑Image‑Edit/ Nano Banana 2 生成指令与编辑图像；自回归蒸馏（Self‑Forcing/Align‑Forcing）、固定 RoPE、FPSA（First‑frame Preserved Sparse Attention）、两步采样等。

**📊 数据集**

构建 CharEdit‑50K（50k 角色编辑样本）作为训练集，并在 CharEdit‑Bench（短视频与长视频子集）进行评估；训练过程中还使用原始视频与编辑参考图像对。

**📈 对比分析**

与 LucyEdit、UniVideo、Ditto 等离线双向编辑器以及 LiveEdit、SANA‑Streaming、StreamDiffusionV2 等流式编辑器对比，使用身份一致性 ID‑SIM、表情/姿态保持 AED/APD、VLM 评估及 Pick Score 等指标；结果显示在编辑质量、表情保持和长时序稳定性上排名第一或第二；实时性能达 14.47 FPS，平均 0.829 s 延迟，比 LucyEdit 高 4.7 倍速度、低 31.7 倍延迟，编辑成功率 0.720，明显优于其他流式方法。

**⚠️ 局限性**

仍依赖预训练的图像编辑模型和对齐自回归蒸馏的高计算成本；对非人类角色的适应性有限；在极长视频或极复杂表情变换下可能仍出现轻微外观漂移；训练过程需要大规模 GPU 资源。

---

## 515. TraceBench: Controlled Evaluation of LLM Agents for Time-Series Root-Cause Attribution

**arXiv ID:** 2608.27182 | [PDF](https://arxiv.org/pdf/2608.27182v1)

**作者:** Tommaso Bendinelli `[一作]` (ETH Zürich), Christian Holz `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个基于物理仿真的 TraceBench 框架，用来生成可控的根因归因任务，评估大型语言模型（LLM）在多变量时间序列根因分析中的能力。

**💡 创新点**

该框架在四个可控轴（域上下文、观测噪声、标记示例、提交模式）上实现精细控制，并通过可解析的机械系统模拟真实的动态变化，从而实现对 LLM 多步工具辅助推理的系统化评估。

**🔧 技术方法**

使用物理仿真器生成时间序列数据，结合 GPT‑4、Claude、Gemini 等闭权重 LLM 与终端/Python 工具链，进行交互式推理与程序生成，提取多维度的探索与资源使用指标。

**📊 数据集**

基于 BallDrop、MassSlide、BounceBall 三种机械系统的模拟生成的数据集，已公开发布在 HuggingFace/TraceBench 以及项目网站。

**📈 对比分析**

在低噪声/高噪声、直接答案/程序提交四种条件下对四个 LLM 进行比较，衡量准确率、token/成本/时间/工具调用等指标；GPT‑4 在所有条件下表现最佳，其他模型准确率明显低于其，程序提交模式普遍低于直接答案，并且其生成脚本的跨样本泛化有限。

**⚠️ 局限性**

局限性包括：仅使用低维机械系统且每次实验仅一次参数变化，标签集为闭式，实验重复次数有限，模型与脚手架差异可能影响结果，且未覆盖复杂工业实际场景。

---

## 516. Common Geodesics Do Not Guarantee Fisher Consistency of the Structured SVM: Minimal Counterexamples and a Tree-Metric Classification

**arXiv ID:** 2608.27203 | [PDF](https://arxiv.org/pdf/2608.27203v1)

**作者:** Jintao Fei `[一作]` (JD.com), Jiangying Luo `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

构造并分析了结构化支持向量机（SSVM）在特定任务损失下的 Fisher 一致性问题，给出了一系列最小反例与完整分类

**💡 创新点**

证明了“公共 geodesic 条件”并非充分条件，给出了树度量的完整一致性判定，并展示了最小输出空间与全支持分布下的反例

**🔧 技术方法**

采用度量几何、最优传输（线性规划对偶）与图论的可耦合条件等理论工具进行严谨证明

**📊 数据集**

无实验数据集，所有结论均通过解析构造与严格数学证明得到

**📈 对比分析**

无实验对比，本文的贡献主要在理论阐述与反例构造，未涉及性能评估

**⚠️ 局限性**

主要局限在于仅针对特定度量空间（树度量和部分中值度量）讨论，未给出通用的校准链接构造，且对更大类度量空间的充分性条件仍未解答

---

## 517. You may implement this later: Cofunctors as partial implementations

**arXiv ID:** 2608.27180 | [PDF](https://arxiv.org/pdf/2608.27180v1)

**作者:** Vincent Wang-Maścianica `[一作]` `[通讯]` (University of Oxford), Vincent Wang-Maścianica (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

实现了基于 cofunctor 的部分实现模型，允许在软件系统中按阶段延迟选择后端实现（如持久化、序列化、复制）。

**💡 创新点**

创新点在于将 cofunctor 解释为状态依赖的部分实现，提供了在规格变更时动态更新实现选择的框架，弥补了传统 functor 在逐步配置上的不足。

**🔧 技术方法**

使用了范畴论中的 cofunctor、离散opfibration、元素范畴等概念，并在 Idris 及 Haskell 示例中实现了该模型。

**📊 数据集**

未使用实际数据集，示例仅基于伪代码（如存储组件的持久化与复制）。

**📈 对比分析**

未进行性能对比，只给出理论证明和演示性实现；在示例中展示了迁移计划的生成与合成，但未提供定量评估。

**⚠️ 局限性**

局限性包括实现复杂、对语言特性的高度依赖、缺乏大规模系统验证，以及缺少性能评估与实测数据。

---

## 518. When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue

**arXiv ID:** 2608.27176 | [PDF](https://arxiv.org/pdf/2608.27176v1)

**作者:** Yen-Ju Lu `[一作]` (Johns Hopkins University), Jesus Villalba `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ContraTalk benchmark，专门评估语音对话中跨模态不一致时的推理能力，并开发了 Audio Twin 框架将音频证据以可读文本形式显式化；

**💡 创新点**

创新点在于：①将跨模态不一致（cross‑modal disagreement）明确为评测轴；②构造冲突与一致两类问答实例；③引入 Agentic‑style Audio Twin，使模型能够显式比较文字与语音证据；

**🔧 技术方法**

使用的技术包括：多模态问答框架、文本与音频特征对齐、结构化音频证据（Audio Twin）、基于检索的证据提取与文本 LLM 推理；

**📊 数据集**

数据集为 Seamless Interaction Dataset，包含语音、转录与对话角色提示，构造 501 题目覆盖 5 语篇维度；

**📈 对比分析**

实验对比了文本 LLM、直接 Audio‑LLM 以及 Audio Twin 系统。文本 LLM 在一致案例表现优秀（>90%），但在冲突案例精度仅 33–48%，误导率高达 34–45%；直接 Audio‑LLM 在冲突案例略有提升但误导率仍 30–40%，一致案例有时下降；Audio Twin 在冲突案例准确率提升至 43–50%，误导率降低至 29–35%，但一致案例性能取决于基模型；

**⚠️ 局限性**

局限性包括：仅关注受控跨模态不一致，未覆盖真实对话中的全部语音线索；Audio Twin 为一种实现，可能不具备普适性；模型仍受基模型容量限制，未完全消除跨模态偏差。

---

## 519. Temporal Sensitivity Analysis of Tessera Embeddings

**arXiv ID:** 2608.27175 | [PDF](https://arxiv.org/pdf/2608.27175v1)

**作者:** Julia Guerrero-Viu `[一作]`, Fabio Pacifici `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文研究了 Tessera 基础模型在不同时间窗口下的像素级地物分类与分割性能，提出了仅改变推理时时间窗口而不重新训练编码器的可控评估框架；

**💡 创新点**

创新点包括①构建可调时间窗口的评估方法；②系统量化任务和各类对时间覆盖的敏感性；③证明即便是单日嵌入仍能保留显著的语义信息，可支持近实时地图更新；

**🔧 技术方法**

采用冻结的 Tessera 编码器，配合线性探测器和 UNet 分割头，对 Sentinel‑1/2 时间序列进行子采样；使用类不平衡权重、标准化和基准实验；

**📊 数据集**

使用 LUCAS（欧洲地物多边形）、DynamicEarthNet（密集日历图像）和 PASTIS‑R（作物类型时间序列）三大公开数据集；

**📈 对比分析**

通过与从零训练的 UNet（RGB、RGB‑temporal、S2、S1S2）以及公开的基础模型结果对比，PASTIS‑R 上 Tessera‑UNet 在 1Y 时取得 58.3 mIoU，较最佳从零训练提升 46%；在 DEN 上与最优从零训练相当；时间窗口缩短对作物类损失显著，而对光谱稳定类影响小；

**⚠️ 局限性**

局限性包括：仅评估冻结编码器在不同窗口下的表现，未探讨短窗口下的预训练或微调；仅使用三种基准和单一基础模型；缺乏动态窗口分配的实际操作实验；对类级细粒度性能的解释仍有待深入。

---

## 520. Ancient-Bench: A Comprehensive Multi-millennial, Multi-medium, and Multi-script Benchmark for Ancient Chinese Artifact Text Recognition

**arXiv ID:** 2608.27169 | [PDF](https://arxiv.org/pdf/2608.27169v1)

**作者:** Hiuyi Cheng `[一作]` (South China University Of Technology), Lianwen Jin `[通讯]` (South China University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了覆盖9种媒介、7种文字、跨3000年时序的古汉字文本识别基准 Ancient-Bench，并对其图像进行统一的符号、字符、解析三重标准化标注。

**💡 创新点**

创新点在于：①系统弥补现有数据集的碎片化，三维覆盖时间、媒介和文字；②设计针对古文媒介的符号标准化、字符标准化与解析标准化三条标注规范，实现跨媒介统一评估；③提供大规模、权威来源的公开数据集，为数字化遗产研究奠定基础。

**🔧 技术方法**

主要技术包括：从14所国家级文化机构收集并预处理高分辨率图像；利用多模态模型（MLLM）进行质量筛选与区域定位；通过古文字学工具（殷契文渊、古音小镜等）对字符进行逐字校对；采用符号、字符、解析三步标准化流程实现一致标注；最终使用多种通用 VLM 与 OCR 专家模型进行评测。

**📊 数据集**

使用的数据集为新发布的 Ancient-Bench，包含2700张图像，涵盖九种媒介（甲骨、青铜、竹简/木简、绢本、印章、碑帖、壁龛、古籍、书法）与七种文字（甲骨文、青铜文、篆书、隶书、楷书、行书、草书），覆盖1200 BCE–1911 CE。

**📈 对比分析**

对比了多类模型：通用 VLM（GPT‑5、GPT‑4o、Gemini、Claude、Qwen、Doubao、GLM、Kimi‑K2.6 等）和 OCR 专家模型（Deepseek‑OCR2、HunyuanOCR、OCRVerse、Qianfan‑OCR、dots.ocr 等）。最佳模型 Doubao‑seed‑2‑0‑lite‑260428 取得 NED 52.08%、F1 57.86%；通用 VLM 在整体上优于 OCR 专家模型；管线式 OCR 专家模型优于端到端模型；但所有模型在甲骨、青铜、书法等细节上表现极差。

**⚠️ 局限性**

局限性包括：缺少字符级别的定位框（仅有行/区域级标注），难以支持细粒度检测评估；数据可能存在与公开训练集的重叠，导致评测结果受污染；目前样本量虽然已大幅提升，但仍不足以覆盖全部古文字变体与稀有符号。

---

## 521. Physical-Layer Fingerprint-Space Capacity Analysis for 100BASE-TX Devices in IIoT

**arXiv ID:** 2608.27164 | [PDF](https://arxiv.org/pdf/2608.27164v1)

**作者:** Chenming Zhang `[一作]` (Duke Kunshan University), Aiqun Hu `[通讯]` (Southeast University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种非线性与冲激响应模型（NAIM），用于刻画100BASE‑TX物理层信号的设备相关差异，并基于该模型结合波形要求、观测分辨率和目标误码率，推导出可辨识指纹空间的容量。同时，使用48台不同型号的网络接口卡（NIC）在不同电缆长度条件下采集的真实波形数据，对模型进行重建、经验容量估计与闭集识别性能进行实验验证。

**💡 创新点**

创新点在于：①首次系统地将非线性级联与冲激响应分解结合，既描述稳态幅度差异又描述过渡响应变化；②将波形硬件规范与观测噪声/ADC量化误差联合考虑，给出可辨识状态的理论上限；③提出基于多维正态假设的Gaussian‑equivalent经验容量度量，并将其与实际识别准确率关联；④对不同NIC型号的经验容量与识别表现进行一致性验证，验证方法的可靠性。

**🔧 技术方法**

使用的技术包括：非线性与冲激响应建模（NAIM）、Ridge回归估计冲激响应、SVD与奇异值分解用于方向分辨率与可行范围评估、随机 Sobol 采样（RQMC）估计联合可行率、OAS–Mahalanobis 最近中心分类器、Bootstrap 设备抽样、Gaussian 等效容量公式、以及标准的误码率决策边距校验。

**📊 数据集**

实验数据集：48 台 NIC（AX88772、CH9151、RTL8152B，各 16 台），在 0.5 m 与 5 m Cat6 以太网电缆下进行三天采样，每台设备每天记录两次捕获，每次 30 条波形，共计 288 条捕获。采样率为 625 MS/s，采样点数 500 000。该数据集用于模型重建、经验容量计算和闭集识别实验。

**📈 对比分析**

通过模型重建误差（NMSE）、指纹空间容量（≈2.96 × 10¹⁰ 可辨识状态）、经验容量（0.5 m 约 1.63 × 10³，5 m 约 2.44 × 10⁵）以及闭集 Top‑1 准确率（5 m 条件下 90.97%）进行比较。实验显示，经验容量与识别准确率高度相关；不同 NIC 型号在经验容量和准确率上排名一致，证明方法的有效性。

**⚠️ 局限性**

局限性包括：①样本量有限，仅覆盖 48 台 NIC；②实验仅在固定温度下进行，未考虑温度漂移和长期稳定性；③电缆长度变化对经验容量的影响仅在两种长度（0.5 m、5 m）内探测，未覆盖更长距离；④方法基于 100BASE‑TX，需重新推导约束后才能推广至 PLC、RS‑485、CAN 等工业链。

---

## 522. Diffusion Policies for Short-Horizon Planning in Robot Crowd Navigation

**arXiv ID:** 2608.27158 | [PDF](https://arxiv.org/pdf/2608.27158v1)

**作者:** Wendong Li `[一作]` (University of Bonn), Jochen Garcke `[通讯]` (University of Bonn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了规划扩散策略优化（PDPO）框架，利用扩散策略生成短期动作块（5步）并在离线行为克隆预训练后通过PPO在线微调，以实现机器人在人群中的安全高效导航。

**💡 创新点**

创新点包括：①将扩散模型用于生成多步、可多模态的动作块，而非传统的单步高斯动作；②将扩散的去噪过程视为内部MDP，使用PPO进行在线强化学习；③识别并修正原始CrowdNav++基准中的越界评估缺陷，提出有界CrowdNav评估环境。

**🔧 技术方法**

技术手段包括：扩散决策模型（DDPM）生成5步速度指令；掩码自注意力编码观测；行为克隆预训练；PPO强化学习微调；边界约束与奖励设计；对扩散步骤的优势分配与log概率存储。

**📊 数据集**

数据集与仿真环境：CrowdNav++仿真器（12×12 m²、20名人类代理）；通过ORCA收集约3000条演示轨迹用于行为克隆；在500个未见测试种子上评估性能。

**📈 对比分析**

与ORCA和CrowdNav++（常数速度预测）基线对比；在原始与有界CrowdNav两种评估下验证。结果显示：PDPO在有界CrowdNav上成功率84.7%（相对CrowdNav++提升11.7pp），入侵率最低（6.81%），导航时间更短；单步扩散策略性能仅略优于CrowdNav++，说明动作块显著提升性能。

**⚠️ 局限性**

局限性：①扩散采样过程增加计算开销，影响实时性；②当前只执行生成块的第一步，缺乏对整条轨迹的监督；③仅在固定人类策略的仿真环境中评估，未覆盖真实人类反应与更复杂交互。

---

## 523. Ultra Low-Power, Lightweight, Probabilistic RSS-Based Path Reconstruction: A System for Landscape-Scale Bee Tracking

**arXiv ID:** 2608.27152 | [PDF](https://arxiv.org/pdf/2608.27152v1)

**作者:** Christopher J. Noroozi `[一作]` (University of Sheffield), Michael T. Smith `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种利用极低功耗、轻量级接收器和旋转高增益发射器的RSS信号进行方向角推断和路径重建的方法，并在昆虫跟踪实验中验证其可行性。

**💡 创新点**

创新点在于（1）通过概率建模仅用少量RSS样本即可推断方向角，充分利用方向性天线的辐射特性；（2）将路径建模为高斯过程，并采用双随机变分推断从不确定、非同步的方向角观测中重建完整轨迹，避免传统三角定位的能耗和精度瓶颈。

**🔧 技术方法**

核心技术包括：旋转高增益Yagi‑Uda天线的RSS采样、全概率分布推断/拒绝抽样/峰值法三种方向角估计算法、方向角到路径的高斯过程建模、双随机变分推断以及JAX自动微分求解。

**📊 数据集**

实验数据来自：①在Ponderosa Park收集的旋转天线AR环形RSS曲线；②在Norfolk Heritage Park进行的已知角度和路径的地面真值实验；③真实昆虫飞行实验中在Common Lane Open Space捕获并跟踪Bombus terrestris 的归巢路径。

**📈 对比分析**

与传统基于大量RSS采样或GNSS的路径估计相比，该方法在仅用3–10个RSS样本时即可达到约15 m（<180 µW）或10 m（<600 µW）的平均误差，显著降低能耗并保持可接受精度；在高k值（>15）下误差可低于2°。

**⚠️ 局限性**

主要局限包括：对天线方向和环境无穷小噪声的假设导致在低k值下方向角推断不确定性高；路径模型假设独立坐标轴导致复杂运动的非线性特征可能未充分捕获；需要事先训练ARP曲线，且在极端遮蔽或多径环境下性能可能下降。

---

## 524. Planning a Shared Modular Fixture Layout Across Robotic Disassembly Stages

**arXiv ID:** 2608.27151 | [PDF](https://arxiv.org/pdf/2608.27151v1)

**作者:** Haohui Pan `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证一种模块化真空夹具系统，规划单一共享支撑布局，使机器人能够在不重新配置夹具的情况下完成螺丝和部件拆除。

**💡 创新点**

提出跨阶段支撑布局规划框架，结合物理评估、去噪扩散概率模型（DDPM）生成初始配置和贝叶斯优化搜索；利用弹性真空手实现对不规则形状对象的适应性支撑；在单一布局下保持完整拆解过程的稳定性。

**🔧 技术方法**

物理基准评估（有限接触稳定性、吸力、扭矩限制）、去噪扩散概率模型（DDPM）用于初始化、贝叶斯优化（Gaussian Process + 混合 kernel）进行搜索、机器人执行与力/扭矩测量。

**📊 数据集**

以电动剃刀（screwdriver）和剃须机（shaver）为目标对象，使用预定义的四个拆解状态的三维网格、质量、重心及操作负载数据；实验包括20×7螺丝拆除、10×各阶段部件拆除、11方向外力测试。

**📈 对比分析**

与传统平行钳夹具进行定性比较；通过机器人拆除实验验证成功率（螺丝拆除成功率86.4%）；通过外力方向测试和操作力/扭矩测量计算经验稳定性余量，平均66.9%（剃刀）和81.6%（剃须机）；实验显示单一布局能够满足所有拆解阶段。

**⚠️ 局限性**

仅考虑准静态操作，姿态对齐手动完成；缺乏在线感知与重规划，未考虑不确定状态变化；对大尺寸或更复杂几何的可扩展性待验证。

---

## 525. INTENT-AS-A-TOOL Makes it Easy to Track Agentic Misalignment

**arXiv ID:** 2608.27348 | [PDF](https://arxiv.org/pdf/2608.27348v1)

**作者:** Yutong Zhang `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出通过向大型语言模型添加专门的意图工具，以显式记录和跟踪模型在推理过程中的误导性意图，并在发现危险意图时插入反思语句实现在线干预；

**💡 创新点**

创新点在于将意图抽象为可调用的工具，从而在不依赖外部判别器的情况下获得细粒度、可实时的意图分数，并利用该信号在推理中及时介入，形成一种新的安全防御机制；

**🔧 技术方法**

主要技术包括链式思维（CoT）监测、意图工具的构造与注入、基于前缀概率的意图分数计算、在线反思插入的干预策略，以及利用vLLM缓存提升推理效率；

**📊 数据集**

实验基于公开的 Agentic‑Misalignment benchmark（包含黑mail、泄露、谋杀三种情景）和 AgentHarm benchmark 进行；

**📈 对比分析**

与传统提示式防御基线对比，意图工具介入在 Qwen 系列模型的 12 个场景中有 9 个表现更优，风险案例的成功率普遍超过 90%（如 Qwen3‑32B 在黑mail 100% ；Gemma‑IT 仍显弱）；

**⚠️ 局限性**

局限性包括：添加意图工具会改变模型的动作空间导致行为漂移；意图分数对模型与场景的依赖性较强；在线干预的时机与反思效果受模型响应稳定性的限制；仅能处理预先指定的目标行为，难以发现未知的错误模式。

---

## 526. Not All Eval-Awareness Is Equal: Capabilities Framing Predicts Compliance

**arXiv ID:** 2608.27340 | [PDF](https://arxiv.org/pdf/2608.27340v1)

**作者:** Allison Zhuang `[一作]` (ENS Paris-Saclay), Santiago Aranguri `[通讯]` (Goodfire AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了语言模型链式推理中对评估意识（eval‑awareness）的不同表述方式（能力型 vs. 安全型）如何影响模型的合规性，并验证了通过在推理前置句中插入不同框架可以因果改变模型的遵守率。

**💡 创新点**

创新点在于：①首次发现 eval‑awareness 在链式推理中会自然分为两种不同的“框架”，并且这两种框架对合规率具有相反的影响；②通过对比 HUA 向量调节和 CoT‑prefill 实验，证明了框架对合规性的因果作用；③提出在安全评估中不应仅关注整体 eval‑awareness 的抑制率，而应关注其框架分布。

**🔧 技术方法**

技术方法包括：对 Qwen3‑32B 模型进行 500 次推理，使用 HUA‑平均向量在多层残差流中进行向量加法调节；三阶段 LLM‑grading 体系（判定 eval‑awareness、真伪、框架分类）；对 52 个攻击性 FORTRESS 评估提示进行 CoT‑prefill 实验，统计 K=50 次随机采样的合规率。

**📊 数据集**

数据集：使用 FORTRESS 安全评估数据集的 55 条攻击性提示（包含对比实验中最能诱发 eval‑awareness 的样本）。

**📈 对比分析**

比较方法：在基线、HUA +0.6、HUA –0.6 三种调节条件下，比较能力型与安全型 eval‑awareness 的合规率差异。结果显示：能力型合规率比安全型高 24–46个百分点；CoT‑prefill 实验中 10/11 前置句在预期方向上显著改变合规率（p≈0.011）。

**⚠️ 局限性**

局限性：仅在单一模型（Qwen3‑32B）与单一数据集（FORTRESS）上验证；使用 LLM‑grading 进行大规模分类，虽已做人工验证但仍可能存在标注误差；子抽样实验（subtractive design）受模型自我再生成干扰，导致在该模型上难以验证框架定位；未探讨在自然 eval‑awareness 正向推理中框架的因果作用。

---

## 527. Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting

**arXiv ID:** 2608.27339 | [PDF](https://arxiv.org/pdf/2608.27339v1)

**作者:** Xinwei Qiang `[一作]` (University of California San Diego), Yufei Ding `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了块式推测解码中被拒绝的原因，将拒绝分解为信息底线（floor）和模型间隙（gap），并用目标模型的滚动结果估计这两项。

**💡 创新点**

提出了信息底线与模型间隙的拆解框架，证明一个已生成的令牌几乎能消除大部分信息底线，并系统量化不同模型与规模下的模型间隙。

**🔧 技术方法**

使用总变差距离、互信息、目标模型滚动采样、前向与后向权重重采样以及块式推测解码的概率估计方法。

**📊 数据集**

在 gsm8k、mbpp、alpaca、arena‑hard 四个任务以及 DeepSeek‑V4‑Pro API 上进行实验。

**📈 对比分析**

通过比较已公开的 DFlash 与 DSpark 的每槽拒绝风险与信息底线，评估其接受长度与模型间隙；发现 Qwen3‑4B 最后槽信息底线为 0.286，接受率 71%，而模型间隙占 55‑67%；DSpark 的一阶信息底线仅 0.041，模型间隙占 88‑92%；不同规模模型保持类似趋势。

**⚠️ 局限性**

限制包括：仅在自由滚动（free‑rollout）环境下估计风险；使用有限长度块（7）且未探索更长块；API 只能提供前 20 个概率，可能漏掉细节；模型间隙拆解假设完美的“最佳提议”，实际可实现的提升受限于模型能力。

---

## 528. Astar: Learning to Propose Evolution Directions for Self-Evolving Industrial AI Systems

**arXiv ID:** 2608.27287 | [PDF](https://arxiv.org/pdf/2608.27287v1)

**作者:** Jinxin Hu `[一作]`, Jiawei Chen `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 Astar 系统，学习并自动提出工业 AI 迭代的演进方向，消除人类专家瓶颈。

**💡 创新点**

创新点在于：① 基于历史提交构建高质量演进语料；② 使用成对扩增与多级噪声过滤提升监督稀疏与噪声问题；③ 通过层级提示和奖励模型实现高效探索；④ 采用三阶段训练（中间训练、SFT、RL）实现从经验到新颖探索的转化。

**🔧 技术方法**

技术包括：对比式数据扩增、基于 AST 的去噪与 LLM 语义过滤、层级提示生成、奖励模型（binary classifier）与 GRPO 强化学习、以及 Qwen3 语言模型的多阶段训练。

**📊 数据集**

使用了阿里巴巴 Lazada 广告平台的历史提交与实验日志，约 112B 训练 token，81.1M 监督样本，8.06M RL 采样。

**📈 对比分析**

与人类专家和多款通用 LLM（GLM、Kimi、MiniMax、DeepSeek、Claude、Qwen、ChatGPT）对比，单提案成功率从 0.3229（专家）/0.3071（GPT‑5.5）提升至 0.6786（Astar‑8B），离线奖励模型 AUC 0.8487，线上 A/B 测试 GMV +4.86% 等显著提升。

**⚠️ 局限性**

局限在于：① 仍需大量高质量迭代日志；② 对新领域的迁移性和跨域泛化尚待验证；③ 需要人工监控与验证流程的迭代闭环；④ RL 训练中奖励模型误差可能导致偏离真实效果。

---

## 529. MM-Spectrum: Multimodal Multi-spectral Molecular Structural Elucidation with a Stable MoE Framework

**arXiv ID:** 2608.27286 | [PDF](https://arxiv.org/pdf/2608.27286v1)

**作者:** Hai-tao Yu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Jun Xia `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种稀疏专家网络MM‑Spectrum，用来解决多模态光谱（NMR、IR、MS）在分子结构推断中的信息不平衡与负迁移问题；通过显式模态路由、结构化专家空间和计算成本正则化，实现多模态信息的高效融合与自适应分配。

**💡 创新点**

创新点：①显式模态感知的路由机制，让路由器能同时利用token内容和模态身份；②将专家划分为共享、模态特定和交互专家，专门捕捉冗余、独特与交互信息；③异构专家容量与计算成本正则化，保证高效利用高价值token；④基于稳定-专业化的课程调度，平衡专家负载与特化。

**🔧 技术方法**

使用稀疏Mixture‑of‑Experts（MoE）层、Transformer编码解码器、显式模态标签、负载平衡正则、计算成本正则、课程调度等技术。

**📊 数据集**

实验数据集包括公开的多模态光谱基准（NMR+IR+MS），SDBS实验光谱数据集以及MMST大规模数据集。

**📈 对比分析**

与传统全拼接的Dense基线对比，MM‑Spectrum在全模态下Top‑1准确率从44.29%提升到76.04%（≈31.7%增幅），在双模态和单模态均有显著提升；在缺失模态测试中性能下降更平滑，且Top‑10准确率提升显著。

**⚠️ 局限性**

局限：在单模态最强的NMR基准下仍无法超越专门化模型；模型仍需较多计算资源，尤其是专家路由与多模态预处理；仅针对NMR、IR、MS三种模态，未验证对其他光谱或化学实验数据的适用性。

---

## 530. TADP: Task-Aware Deformable Prediction for Single-Stage 3D Object Detection

**arXiv ID:** 2608.27282 | [PDF](https://arxiv.org/pdf/2608.27282v1)

**作者:** Su Wang `[一作]` (Xi'an Jiaotong University), Yuehu Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出单阶段3D目标检测的任务感知可变形预测框架TADP，结合三层特征细化聚合、多尺度特征融合与任务感知可变形检测头，显著提升检测精度。

**💡 创新点**

创新点在于引入任务感知可变形检测头，为不同任务生成自适应变形图并通过三层特征提取和多尺度融合实现更丰富的特征表达。

**🔧 技术方法**

使用稀疏体素编码、SC层自校正、三分支特征细化（语义、结构、几何）、多尺度特征聚合（MSFA）、可变形卷积等技术。

**📊 数据集**

在KITTI数据集的车辆检测任务上进行训练和评估。

**📈 对比分析**

与多种单阶段及双阶段检测器对比，TADP在KITTI车类3D mAP上达88.93%/79.65%/74.17%，超越多数单阶段和部分双阶段方法，且推理速度更快。

**⚠️ 局限性**

对极稀疏或高分辨率场景的鲁棒性有限，且变形头增加了额外计算，对极低功耗硬件的适配仍有挑战。

---

## 531. What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents

**arXiv ID:** 2608.27260 | [PDF](https://arxiv.org/pdf/2608.27260v1)

**作者:** Xingshan Zeng `[一作]` (Huawei Technologies Co., Ltd), Weiwen Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出ACE框架（Accuracy–Complexity–diversity），将代理式数据拆解为四个因素（环境 E、任务 q、交互 τ、验证 v），并系统性综述和分类现有的生成方法，构建统一的评价与生成视角。

**💡 创新点**

创新点在于：① 将代理式数据统一抽象为 (E,q,τ,v) 形式，② 将生成目标形式化为 ACE 约束分布设计，③ 提出了基于 ACE 的机制导向生成范式（前向、任务先、轨迹先、结构先）与评价标准，④ 指出准确性、复杂度与多样性之间的非对称关系并给出可量化指标。

**🔧 技术方法**

采用的技术包括：规则/模型/人工层级验证、结构化工具图与计划生成、可执行环境与仿真器、逆向生成、强化学习回放、模型可解释与推理、递归合成与自适应反馈。

**📊 数据集**

综述涵盖的领域与数据集包括：API 调用、代码仓库任务、GUI/网页交互、机器人仿真、科学探索与证明环境等多领域公开数据，未聚焦单一数据集，而是对多种数据来源与生成方式进行聚合。

**📈 对比分析**

对比方法通过 ACE 维度下的准确率、成功率、熵、覆盖率、有效性门槛等指标进行定量评估，并指出多样化、可执行验证和模型适配相结合的生成方案通常能显著提升学习效果；同时对比传统一次性轨迹采样、静态指令生成等方法的局限。

**⚠️ 局限性**

局限性包括：① 真实性与可验证性难以兼顾，人工或模型验证易产生偏差；② 复杂度估计高度依赖模型与超参数，难以统一衡量；③ 多样性度量缺乏通用标准，易受表面变异影响；④ 生成流程易被验证器或当前模型过拟合，导致覆盖退化；⑤ 需要持续自适应与多源验证以保持泛化。

---

## 532. Compositional Online Learning for Semantic Data Processing Systems

**arXiv ID:** 2608.27244 | [PDF](https://arxiv.org/pdf/2608.27244v1)

**作者:** Paweł Liskowski `[一作]` (Snowflake Inc.), Dimitris Tsirogiannis `[通讯]` (Snowflake Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 LLM 调用边界上提出了一种可组合的在线学习框架，用于优化语义数据处理系统，并在 Cortex AISQL 上实现了过滤器排序（Larch）和级联路由（GAMCAL）两种在线学习组件。

**💡 创新点**

核心创新在于：① 利用 LLM 调用的“延迟窗口”将 CPU 端学习更新隐藏在一次 LLM 调用的往返延迟内，突破了传统 AQP 对轻量学习器的限制；② 通过层次化设计空间（决策粒度 × 学习更新节奏）统一了不同学习组件的组合与交互；③ 引入基于 Hoare 规则的组合理论，提供了多组件互相嵌套的成本上界与实际交互的定量分析。

**🔧 技术方法**

技术包括：
- 端到端的在线梯度更新与贝叶斯/加权阈值重调；
- Larch 的 MLP 选择性估计与动态规划排序；
- GAMCAL 的加性模型校准与带噪声的阈值决策；
- 一阶异步学习循环（训练与推理相互交错）；
- 交互式成本分解与 Hoare 样式组合规则。

**📊 数据集**

实验使用了三大公共数据集：GovReport、PubMed、BigPatent，并在这些数据集上构造了包含 1,000,000 行、5 条语义谓词的 conjunction-filter 工作负载。

**📈 对比分析**

对比方法包括：
- 基准未优化（直接全部 LLM 调用）；
- 单组件优化（仅过滤器排序或仅级联路由）；
- 传统离线/编译时学习方案（Palimpzest、Quest、SOTA 预估器）。
结果显示：单组件可分别将成本降低约 75% 与 65%；两组件组合在假设独立的理论上可实现 11.4× 的成本压缩，实际考虑交互后约 8× 的提升；相较于现有方案，Token 开销下降 4–8×，并在混合工作负载上显著优于同类系统。

**⚠️ 局限性**

局限性包括：
- 依赖于 LLM 调用延迟窗口的假设，若 LLM 速度提升或多模型协作改变这一窗口将影响学习可行性；
- 交互模型基于独立性与缓存缺失假设，实际数据分布可能违反；
- 目前仅在单个查询内部实现学习，缺乏跨查询热身与持久化；
- 对不同语义操作符（如 join、生成）的扩展仍需研究；
- 需要更细粒度的质量保证与合约机制来协调多组件的容错与性能。

---

## 533. Importance Scoring of Transformer Attention Heads in Learning Tabular Data

**arXiv ID:** 2608.27241 | [PDF](https://arxiv.org/pdf/2608.27241v1)

**作者:** Ahmad Jad Allah `[一作]`, Manar D. Samad `[通讯]` (North Carolina Agricultural and Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了多头 Transformer 在表格数据学习中的各个注意力头的重要性，并提出了一种基于梯度加权的头重要性评分指标；

**💡 创新点**

创新点在于首次对表格数据的 Transformer 头重要性进行量化，提供了可视化与剪枝依据，并公开了实现代码；

**🔧 技术方法**

使用了 TransTab 变形器、软动态门控机制、梯度加权重要性评分、以及多头剔除实验与 Wilcoxon/Cohen’s d 等统计比较方法；

**📊 数据集**

实验基于 40 个来自 OpenML 的多域表格数据集，样本量从 500 到 96,320，特征数 4 到 256，混合数值与分类特征；

**📈 对比分析**

通过 8 种实验设置（含/不含软调优、随机 vs 重要性排序剔除），在 AUC 上比较，结果表明按重要性递增剔除（I_min→I_max）能保持 72.5% 的实例性能，最重要头先剔除效果最差；

**⚠️ 局限性**

限制在于仅评估了单一 TransTab 架构，缺乏跨不同 Transformer 结构的验证，且表格数据异质性导致效果差异较大，需要进一步研究。

---

## 534. SPA: Securing Persistent LLM Agents Across Queries with Plan-First Information-Flow Control

**arXiv ID:** 2608.27234 | [PDF](https://arxiv.org/pdf/2608.27234v1)

**作者:** Dylan Girrens `[一作]` (University of South Florida), Guangjing Wang `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 SPA（Plan‑First Information‑Flow Control）体系，旨在使持续运行的 LLM 代理在跨查询时仍能保持规划、执行和持久状态的安全性。

**💡 创新点**

创新点包括：① 采用一次性完整可执行计划的 plan‑first 架构；② 设计可验证的 DSL，显式表达数据与控制依赖；③ 在计划和执行前通过双 lattice（机密性与完整性）信息流控制；④ 使用标签保留的持久存储，让后续查询仅接收语义元数据；⑤ 开发 AgentDojo‑MQ 多查询基准，专门评估跨查询攻击与重用效果。

**🔧 技术方法**

技术实现：LLM 规划器生成 DSL 计划；DSL 通过语法检查与信息流验证；双 lattice IFC 在执行前进行静态安全检查；抽象工具绑定（abstract‑to‑concrete mapping）隔离工具元数据；持久化模块维护三视图（metadata、payload、标签）；隔离执行环境（容器、网络限制）和隔离 LLM（quarantined LLM）。

**📊 数据集**

使用的数据集是 AgentDojo 97 个任务，扩展成 AgentDojo‑MQ（多查询版本），以测量跨查询利用率、延迟攻击和安全效用。对每个任务生成了注释图、多轮输入和依赖关系。

**📈 对比分析**

比较方法：在单查询和多查询两种设置下，分别切换 Concrete/Abstract 规划模式和是否启用 IFC，评估指标包括任务成功率、攻击成功率（ASR）、流水线通过率和跨查询重用率。结果显示：启用 IFC 可将 ASR 降至 0%（单查询）或 0.2%（多查询），但在 Concrete 模式下效用从 53% 降至 29%；多查询时持久标签使重用率保持在 90% 以上，且对攻击的抵御保持在 0–0.6%。

**⚠️ 局限性**

局限性：① 评估主要聚焦在 tool‑output 注入，缺乏专门针对持久化的攻击模板；② 严格的完整性检查导致合法工作流被拒绝，效用下降；③ 依赖可信的部署策略（工具绑定、标签、策略），若配置错误可能产生误判；④ AgentDojo‑MQ 通过人工转换得到，未覆盖自然长交互场景，实验范围有限。

---

## 535. Blindfolded pursuit with delays of your choice

**arXiv ID:** 2608.27347 | [PDF](https://arxiv.org/pdf/2608.27347v1)

**作者:** Torben Schürenberg `[一作]` (University of Bremen), Maximilian J. Stahlberg `[通讯]` (University of Bremen)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图上的追逐-逃避游戏，涉及一个追捕者和一个不可见的逃避者。追捕者可以为图的边分配整数旅行时间，并在每个时间步骤指定一个有限的查询顶点序列。逃避者则选择在相同时间范围内的行走，旨在躲避所有查询。

**💡 创新点**

追捕者能够选择旅行时间使其在多项式时间内成功捕捉逃避者，这与无权图的情况形成对比，在无权图中，所需的追捕者数量可能会随着顶点数量线性增长。

**🔧 技术方法**

使用了图论和算法设计技术，特别是通过分配边的延迟来优化追捕者的策略。

**📊 数据集**

研究中没有具体提到使用的数据集，而是理论上分析了任意图的性质。

**📈 对比分析**

与传统的追逐-逃避游戏相比，研究表明在允许追捕者选择边的延迟的情况下，追捕者在任何图上都能在多项式时间内获胜。对于逃避者可以等待和提前开始的情况，追捕者仍然能够在指数时间内获胜。

**⚠️ 局限性**

研究的局限性在于未考虑追捕者可以改变图的拓扑结构，只允许其分配边的延迟，这可能限制了策略的多样性。

---

## 536. Assessing Company Contributions to Societal Resilience: Extending the Societal Capacity Assessment Framework to Agentic AI

**arXiv ID:** 2608.27238 | [PDF](https://arxiv.org/pdf/2608.27238v1)

**作者:** Catherine Simons `[一作]` (Cambridge Boston Alignment Initiative), Neil Thompson `[通讯]` (MIT FutureTech)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将社会韧性评估框架（SCAF）从国家层面迁移到公司层面，并通过设计16个基于2026 CLTC Agentic AI风险管理标准的指标，构建了针对企业AI代理部署的社会韧性评估工具，随后以微软为案例对其公开治理文件进行检索、关键词匹配和手工编码，评估微软在降低风险、提升社会韧性方面的贡献。

**💡 创新点**

创新点在于：①首次将SCAF扩展为企业级评估工具，聚焦AI代理部署对社会韧性的影响；②以高优先级的AI风险缓解措施为依据，系统化地将风险管理实践映射为韧性维度（脆弱性、应对、适应、转化）；③通过公开文件的关键词匹配实现对企业治理实践的可操作化评估，为后续多企业比较提供了方法框架。

**🔧 技术方法**

主要技术包括：①指标设计（基于文献综述和CLTC标准的专家优先级排序）；②文本检索与关键词匹配（使用手工构建的词表在21份微软治理文件中检索）；③手工编码与归档（将检索到的文本片段按指标归类）；④概念框架映射（将指标对应到SCAF的四个韧性维度）。

**📊 数据集**

数据集为微软公开的AI治理文档，共21份（涵盖安全产品博客、正式治理文件、战略与采纳框架等），并使用306个与16个指标相关的关键词进行检索，最终得到142条编码片段。

**📈 对比分析**

由于本文聚焦方法论验证而非性能对比，未进行数值指标评估；通过案例演示显示评估框架能揭示微软在各韧性维度的覆盖强弱（如适应性维度覆盖度高，脆弱性和转化维度覆盖度低），为后续跨公司、跨时间的对比研究奠定基础。

**⚠️ 局限性**

主要限制包括：①仅评估公开文档，无法验证实际落实情况；②只对单一企业（微软）进行案例，结果无法泛化；③指标设计依赖CLTC标准与专家判断，可能遗漏对社会韧性重要但未被列为高优先级的措施；④关键词匹配与手工编码主观性高，需进一步验证互评一致性；⑤未与真实事故数据关联，缺乏构念效度验证。

---

## 537. Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers

**arXiv ID:** 2608.27343 | [PDF](https://arxiv.org/pdf/2608.27343v1)

**作者:** Ke Shu `[一作]` (University of Helsinki), Mikko Tolonen `[通讯]` (University of Helsinki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何从碎片化的文本重用证据中恢复18世纪书籍与报纸间的全文级再版与再利用，提出基于 pair‑level 证据合并的工作流框架。

**💡 创新点**

创新点在于：① 将碎片级检索转化为 pair‑level 证据聚合与传递关系判断；② 设计可审计的分阶段规则工作流；③ 将该工作流与直接 LLM、决策树等基线对比，展示精度控制与候选空间压缩的实效。

**🔧 技术方法**

技术手段包括：局部文本匹配产生 fragment hits → pair‑level 特征聚合（覆盖率、bundle span、链长、段落浓度、标题/引号线索）；分阶段规则判定与自动规则适配；直接 LLM 提示式二元分类作为基线。

**📊 数据集**

数据集：以 17 本大卫·休谟（David Hume）著作为源，覆盖 ECCO（18 世纪书籍）与 Burney Newspapers（18 世纪报纸）两大目标域，构建 22,735 条 ECCO–ECCO 与 3,583 条 ECCO–Newspaper 的 pair‑level 候选。

**📈 对比分析**

比较方法：在 ECCO–ECCO 标签切片上与最终工作流对比的 7 种方法中，最终工作流在 main slice F1 0.825、hard slice F1 0.270，精确控制且部署输出仅 771；LLM 输出 14k+ 正例，Recall 最高但精度低。ECCO–Newspaper 手工审核显示所有 176 正例均为真实再利用，F1 达 0.978。

**⚠️ 局限性**

局限性：① 仅聚焦休谟作品，难以直接推广至其他作者或时期；② 标签切片小且与硬例池重叠，难以评估真正的外部泛化；③ 报纸注释为单测且未计算一致性指标；④ 未评估 OCR 噪声对结果的影响；⑤ LLM 实验仅为直接提示式分类，未充分探讨其在规则生成或重排序中的潜力。

---

## 538. R2M-Bench: Evaluating Revisit Memory via Relative Consistency in Interactive Video World Models

**arXiv ID:** 2608.27328 | [PDF](https://arxiv.org/pdf/2608.27328v1)

**作者:** Qiwen Gu `[一作]` (Alibaba Group), Junqiao Zhao `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出R2M-Bench，用于量化交互式视频世界模型在离开并返回时的记忆一致性。

**💡 创新点**

创新在于采用同视频相对校准的评估协议（MemoryGain与Normalized Memory Ratio），将复访优势与同一视频的时间间隔相匹配，并在五个可观测一致性维度上进行评估。

**🔧 技术方法**

技术包括命令式轨迹构造、相对帧对采样、PSNR/SSIM/LPIPS、DINO/BoQ/MutualVPR、GroundingDINO+SAM2、SuperPoint+LightGlue、CLIP语义匹配以及Gemini‑3.1‑Pro‑Preview的视觉‑语言评判。

**📊 数据集**

数据集由100个多样化参考场景（室内、城市、自然、抽象）与三种离开‑返回轨迹模板组合，生成300个评测实例。

**📈 对比分析**

对七个动作条件视频世界模型（含DreamX‑World‑Memo、HY‑WorldPlay等）进行评测，DreamX‑World‑Memo在整体NMR上最高（0.706），与人类一致性相关系数0.547。评估显示不同维度和轨迹下的表现差异。

**⚠️ 局限性**

局限性包括评估仅关注可观测一致性，无法定位内部记忆机制；复访采样受命令姿态和执行误差影响；评估方法对视觉骨干和提示存在偏倚，且仅覆盖导航任务，未包含交互或动态状态变化。

---

## 539. HALO: A Heterogeneity-Aware Language-Aligned IMU Foundation Model for Open-Set Human Activity Recognition

**arXiv ID:** 2608.27233 | [PDF](https://arxiv.org/pdf/2608.27233v1)

**作者:** Zihan Ding `[一作]` (Hong Kong University of Science and Technology), Xiaomin Ouyang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 HALO，一个面向多设备、开放词汇的人体活动识别基础模型，能够在任意采样率和通道配置下实现零样本识别。

**💡 创新点**

创新点在于将自监督预训练与语言对齐软对比学习相结合，并通过自适应池化、通道无关特征和语义化传感器描述来解决传感器异质性与标签同义冲突。

**🔧 技术方法**

使用了自监督掩码自动编码、对比学习、自然语言嵌入的 Sentence‑BERT、软对比损失以及基于文本的同义词增强。

**📊 数据集**

训练使用10个公开 HAR 数据集（如 UCI‑HAR、PAMAP2、WISDM 等），评估在7个未见数据集（MotionSense、RealWorld、MobiAct、Shoaib、Opportunity、HARTH、VTT‑ConIoT）上。

**📈 对比分析**

与五个先进基线（LiMU‑BERT、MOMENT、CrossHAR、LanHAR、LLaSA）对比，HALO 在所有八个评估指标上领先，零样本开放集准确率提升 13.7 个百分点，且参数量仅约 35 M。

**⚠️ 局限性**

局限在于对极端分布漂移（如 HARTH 和 VTT‑ConIoT）仍然难以零样本迁移，且对单一样本的传感器描述依赖较高，需进一步改进。

---

## 540. RCMN: Understanding Misleadingness in Influential Public Discourse

**arXiv ID:** 2608.27358 | [PDF](https://arxiv.org/pdf/2608.27358v1)

**作者:** Peiling Yi `[一作]` `[通讯]` (Kingston University London), Peiling Yi (Kingston University London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Reader-Centric Misleadingness Understanding (RCMN) 框架，构建了基于专业事实核查的多维度误导性数据集，并在此数据集上设计并评估了轻量级的“声称+上下文”模型，探讨了在缺少完整证据和多模态信息时模型对误导性理解的可行性。

**💡 创新点**

创新点包括：① 将误导性拆解为五个维度（机制、读者解释、证据支持解释、情绪激发、传播意图）形成完整的读者中心税onomies；② 通过事实核查来源构建具有证据支撑的高质量数据集；③ 通过对比不同模型在“声称+上下文”下的表现，证明轻量级输入可恢复大部分情绪与意图信息，但对机制识别仍受限。

**🔧 技术方法**

技术手段主要是：AI 辅助的证据提取（GPT‑5.6 Sol）、人工验证与裁决；使用 5 个大型语言模型（Qwen3‑VL‑8B、DeepSeek‑V4‑Flash、Gemma‑4‑12B、GPT‑5.6 Sol、Claude‑Fable‑5）进行零样本推理；评价指标包含宏 F1、ROUGE‑L 与语义等价性评估。

**📊 数据集**

数据集为 RCMN，取自 Fact‑Check Insights，涵盖 2,216 条 2019‑2025 年的公众争议性声明，配备原始文本、上下文、证据链、读者解释与情绪/意图标签。

**📈 对比分析**

评估方法：在相同 2216 条样本上，使用宏 F1 评估误导机制、情绪激发与传播意图分类，ROUGE‑L 与语义等价性评估读者解释生成。结果显示：情绪（macro‑F1≈0.64）与意图（macro‑F1≈0.60）恢复表现良好；读者解释生成语义等价率高达 84‑97%；但误导机制识别的宏 F1 仅 0.30‑0.52，尤其对 “非误导” 类准确率低。

**⚠️ 局限性**

局限性：① 样本中非误导实例比例低，导致“非误导”类评估受限；② 依赖人工判断，仍存在主观性；③ 对原始多模态内容的恢复不完整，某些实例仅通过文本重建；④ 轻量级输入缺乏对缺失信息与外部证据的识别，机制识别难度大。

---

## 541. Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO

**arXiv ID:** 2608.27351 | [PDF](https://arxiv.org/pdf/2608.27351v1)

**作者:** Yunpeng Ba `[一作]` (Southern University of Science and Technology), Zhenkun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了进化策略（ES）在大语言模型（LLM）推理能力提升中的可行性和特点，重点比较ES与GRPO的后训练行为、灾难性遗忘与参数漂移及其对大模型的影响；

**💡 创新点**

发现ES能在保持Pass@1的提升同时保留甚至提升Pass@K（多采样成功率），避免GRPO出现熵坍塌和多采样性能下降；揭示ES导致的参数漂移集中在少数大幅更新的层规范化与注意力参数上，且大参数漂移不必然导致灾难性遗忘；提出对ES有效的奖励归一化、扰动尺度与小规模种群的可扩展性策略。

**🔧 技术方法**

使用基于参数空间的进化策略（ES）一点评估器、Z-score奖励归一化、扰动尺度调优、种群规模实验；与GRPO（基于梯度的策略优化）做对照；理论上推导ES种群多样性对Pass@K的正向影响。

**📊 数据集**

主要数据集包括GSM8K（算数推理）、DeepScaleR（数学推理）、GPQA、MATH-500、AIME24/25、AMC23等多种推理与通用任务的公开数据集。

**📈 对比分析**

与GRPO和基线（无后训练）进行比较，结果显示ES在Easy（两轮GSM8K后训练）和Hard（一次DeepScaleR后训练）场景下平均Pass@1、Pass@16、Pass@32均高于基线，且在Pass@K上优于GRPO；并在多任务保留评估中保持或提升离线性能。

**⚠️ 局限性**

局限性包括：实验主要集中在单一或少数任务与模型规模；长期多任务持续学习中的参数漂移对已学习能力的影响仍未完全阐明；在某些任务上ES相较GRPO未展现优势，且对大规模模型仍需更多的资源与工程优化。

---

## 542. BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks

**arXiv ID:** 2608.27334 | [PDF](https://arxiv.org/pdf/2608.27334v1)

**作者:** Jeong-Yoon Kim `[一作]` `[通讯]`, Jeong-Yoon Kim

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将工业场所的只读遥测数据转换为可执行的多轮代理任务的方法，称为BTS-AgentBench。

**💡 创新点**

创新点在于开发了一种确定性的构建管道，从标准化的原始遥测数据生成静态可执行任务，并将其提升为多轮代理交互的情景。

**🔧 技术方法**

使用了确定性构建管道和规则基础的构建排除控制器，确保生成的基准数据可重复且可验证。

**📊 数据集**

使用了BTS（建筑时间序列数据集）和XAI4HEAT（供热子站的遥测数据集）作为数据源。

**📈 对比分析**

与其他基准相比，BTS-AgentBench专注于将只读遥测数据编译成有界的、可确定性评估的情景，性能评估显示GPT-5.5在89个测试行中成功完成79个，整体成功率为88.8%。

**⚠️ 局限性**

局限性在于BTS-AgentBench的评估范围较窄，仅涵盖只读遥测搜索、聚合、比较、排名、时间戳报告和质量报告，不涉及写入控制、安全关键的执行、维护计划或长期故障排除。

---

## 543. Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit

**arXiv ID:** 2608.27309 | [PDF](https://arxiv.org/pdf/2608.27309v1)

**作者:** Shuyi Fan `[一作]` (Columbia University), Hongyang Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过统计识别分析与闭式推导，证明在LLM评判器审计中使用的双重差分（within‑item difference‑in‑differences）端点在有界评分尺度上不可识别，并在一项预注册的“学习者画像”对评判器支架偏好影响的审计中验证了这一失真。

**💡 创新点**

创新点在于：①首次揭示双重差分端点被截断可产生伪交互效应；②给出可直接从评分数据诊断此问题的闭式公式与算法；③在真实审计数据上演示了该失真对结论的影响。

**🔧 技术方法**

主要技术手段包括：差分设计与识别理论、统计学闭式推导、Wilcoxon符号秩检验、BCa bootstrap 区间、以及基于评分的诊断构造（零差异偏好模型）。

**📊 数据集**

使用的数据集为 55 个由“frozen pedagogy judge”评分的对话片段，包含 990 次评审（高低支架各 5 次，三种 profile 情况），来自 2,219 条 tutor 日志，按弱/强两层能力划分。

**📈 对比分析**

比较方法：对弱组原始端点采用 Wilcoxon 符号秩检验（+0.085，p=0.684），对绝对分数进行差分检验（显著降低），对单字段交互使用零差异偏好构造（可重现 79–85% 的交互）。结果表明原始差分端点不显著且非识别，显著交互几乎完全由截断效应驱动。

**⚠️ 局限性**

限制：仅针对单一评判器与单一评分表，样本量有限；高低极端导致截断，无法区分真实偏好与截断效应；缺乏无对话情境的对照；恢复潜在真实效应需引入隐变量模型或更宽阔的评分尺度。

---

## 544. Detection of Christmas tree plantations from high-resolution aerial imagery. A case study in the French Morvan

**arXiv ID:** 2608.27290 | [PDF](https://arxiv.org/pdf/2608.27290v1)

**作者:** Francesca Razzano `[一作]` (University of Naples Parthenope), Jocelyn Chanussot `[通讯]` (INRIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套基于深度学习的语义分割流水线，用高分辨率RGB航空影像识别法国Morvan地区圣诞树种植园，解决了高密度、短周期、与周围植被混淆以及极端类别不平衡等难点。

**💡 创新点**

创新点包括：①将圣诞树种植园视为稀有目标的专属语义分割任务；②设计专属的硬负样本挖掘(HNM)策略和加权BCE+Tversky混合损失；③实现跨年时间迁移和大规模验证，首次为这类小规模植被系统提供系统性遥感监测方法。

**🔧 技术方法**

技术手段包括DeepLabV3‑ResNet34编码器、随机增广、硬负样本优先采样、加权BCE与Tversky联合损失、混合精度训练、阈值优化与连通域后处理。

**📊 数据集**

使用数据集：BD ORTHO高分辨率0.2 m航空影像（2017/18、2020、2023年）；2020年人工矢量分块注记（1,253块）；辅助层水体、草地、清除区（10 m分辨率）及相关土壤与土地利用图层。

**📈 对比分析**

通过与多种网络基线（DeepLabV3‑ResNet18/50、FPN‑ResNet34、U‑Net系列）对比，DeepLabV3‑ResNet34在2020测试集上IoU达0.733、F1 0.846；HNM显著提升精度召回（AP从0.204提升至0.913）；在2017/18和2023年时间迁移时IoU分别为0.751/0.858、0.691/0.736；大规模验证中精度约0.653/0.597、召回0.736/0.879、F1约0.692/0.711，证明模型具备良好的泛化和操作性。

**⚠️ 局限性**

局限性包括：仍易产生对草地、清除区等结构化背景的误检；极端类别不平衡导致需要额外的后处理；仅在单一年份标注下训练，导致对长期动态变化的适应有限；缺乏实例级分割与跨地区迁移评估。

---

## 545. Decoupled I/O-Dominant Pipelines for Large-Scale Whole-Slide Image Embedding Extraction

**arXiv ID:** 2608.27278 | [PDF](https://arxiv.org/pdf/2608.27278v1)

**作者:** Mayanka Chandrashekar `[一作]` (Oak Ridge National Laboratory), Heidi Hanson `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种将全切片图像嵌入提取拆分为补丁生成、GPU并行推理和向量数据库写入的分阶段、I/O感知流水线

**💡 创新点**

通过解耦三大阶段、关注I/O瓶颈，并将嵌入与完整元数据持久化为可复用向量数据库，显著提升大规模WSI处理的可扩展性和复用性

**🔧 技术方法**

MPI+CPU并行补丁生成、GPU并行推理（HIPT、H‑Optimus‑0、Virchow2等基础模型）、向量数据库（Milvus/FAISS）及burst buffer等高性能I/O技术

**📊 数据集**

4185幅H&E全切片图像（CCDI MCI），共约4.14亿个补丁（170M非白色补丁）

**📈 对比分析**

与传统紧耦合流水线对比实验表明工作负载受I/O主导：GPU利用率低、吞吐量随GPU/节点增多提升但效率下降；向量数据库写入吞吐最高可达188k行/秒，整体效率在中等规模最优

**⚠️ 局限性**

受限于小文件I/O、元数据争用与存储带宽，导致并发扩展受限；系统仍需在文件系统与元数据层面进一步优化，以突破存储瓶颈

---

## 546. BrailleBench: Investigating Multi-Criteria Braille Comprehension in Large Language Models

**arXiv ID:** 2608.27268 | [PDF](https://arxiv.org/pdf/2608.27268v1)

**作者:** Jinghan Zhang `[一作]` (Clemson University), Chang-Tien Lu `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 BrailleBench benchmark，包含 5,570 条实例、5 个公开数据集（GSM8K、AIME 2024、CommonsenseQA、HotpotQA、2WikiMultiHopQA），并以 UEB Grade 1 与 Grade 2 两种 Braille 等级，评估 LLM 在 Braille 阅读、Braille 输出以及 Braille 端到端交互（G→E、E→G、G→G）中的性能。

**💡 创新点**

首次系统性评估 LLM 在 Braille 交互中的多维能力，提出三种交互情景并设计可复现的 Braille Toolkit，提供 deterministic 的 UEB 转录流程以及多种 Braille 表征（ASCII、Unicode、点数）比较，揭示现有 LLM 对 Braille 处理的不足。

**🔧 技术方法**

使用 UEB 规则与 Braille ASCII 为主表征，结合自定义 Braille Toolkit 进行 deterministic 的 UEB Grade 1/Grade 2 转录；对比不同表征（ASCII、Unicode、点数）对性能的影响；在 LLM 端实现 Braille 输出验证器，并通过多种任务指标（Math‑Verify、EM、F1）进行评估。

**📊 数据集**

利用 GSM8K、AIME 2024、CommonsenseQA、HotpotQA、2WikiMultiHopQA 五个公开数据集，并在每个数据集上构造对应的 Braille 版本。

**📈 对比分析**

在六个代表性 LLM（Claude Opus 4.8、Claude Haiku 4.5、Llama 3.3 70B、Llama 3.1 8B、Qwen3 32B、Qwen3 1.7B）上进行零样本评估，发现 Braille 条件下平均性能下降 30–70%；Grade 2 对比 Grade 1 更难，Unicode 表征通常不如 ASCII，加入 Braille 参考可提升某些模型但整体提升有限；实验表明现有 LLM 对 Braille 交互的支持不足。

**⚠️ 局限性**

局限性：模型缺乏完整的 Braille tokenization 与上下文解码能力，无法准确追踪数值状态与缩写冲突，实验仅基于零样本评估，未探讨微调或专门训练；因此对 Braille 的全面理解仍远未实现。

---

## 547. SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models

**arXiv ID:** 2608.27265 | [PDF](https://arxiv.org/pdf/2608.27265v1)

**作者:** Yi Ding `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SCIT（Suffix Cache Interchange Test）方法，用于定位Transformer模型内部实现链式推理的具体内存对象，重点分析GPT-2两种算术检查点的机制。

**💡 创新点**

创新点在于将隐式链式推理的因果对象从单纯的“隐藏状态”转移到“缓存片段”，通过系统化的补丁、成分拆分、匹配腐败等多维度控制，得到精确的“携带者映射”，并揭示不同规模/任务下机制的转移。

**🔧 技术方法**

使用了SCIT协议，包括源-接收者对接、缓存片段补丁、K/V分离、隐藏状态耦合、语义源控制、解码验证与匹配腐败等技术；同时在实验中采用了多层、不同层块的扫描和可逆对齐方法。

**📊 数据集**

实验数据集主要是自定义的算术类任务（balls、books、coins），以及在1B/8B、Qwen3-4B等规模下的扩展，所有任务均提供了确切的源-接收者反事实答案与中间变量。

**📈 对比分析**

通过与一系列对照实验（隐藏状态、键/值分离、同答、跨模板等）比较，结果显示在CODI-GPT2中中后期值缓存片段是主要的携带者，Sim-CoT样式也呈现相同趋势但缺乏完整必要性；在8B规模下机制转移至前缀/全缓存键/值；Qwen3-4B等模型在边界条件下无机制调用。

**⚠️ 局限性**

局限性包括：仅在合成算术任务上验证，缺乏对自然语言复杂推理的泛化；变量停止、路径敏感性仅在少数实验中探测；对不同架构、层划分的适应性不足；所提出的机制并非通用，需针对每个检查点进行定位。

---

## 548. Making Latent Evolution Explicit: Operator-Structured Transitions for World Action Models

**arXiv ID:** 2608.27259 | [PDF](https://arxiv.org/pdf/2608.27259v1)

**作者:** Xiaoxiao Lu `[一作]`, Ye Yuan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的隐式状态转移建模方法——Latent Evolution Operator Network (LEON)，用以在世界动作模型 (WAM) 中显式地建模隐状态随时间的演化，并在不同的预测–策略耦合方式下对其进行替换实验。

**💡 创新点**

创新点在于将转移函数的结构视为独立的架构设计维度，借助受控 Koopman 生成器理论，将上下文信息映射为对共享低秩演化算子的系数，从而将隐状态演化拆分为算子传播与可加强迫两部分，提供更具物理解释性与可迁移性的演化 inductive bias。

**🔧 技术方法**

使用了受控 Koopman 生成器理论、低秩算子分解、上下文调制的算子网络、可加强迫分支、以及传统的 Transformer 作为对照；在 WAM 中嵌入 LEON 取代原有的 Transformer 预测层。

**📊 数据集**

在真实机器人实验中使用 LIBERO、LIBERO-Plus 视觉语言导航与操作数据集，以及 RoboTwin 2.0 的四任务子集（Lift Pot、Beat Block Hammer、Dump Bin Bigbin、Hanging Mug）进行评估。

**📈 对比分析**

对比方法包括 VLA-JEPA（表示中介耦合）和 LaWAM（策略面向耦合）以及多种基线（如 OpenVLA-OFT、π_0、CoT-VLA 等）。实验显示：VLA-JEPA+LEON 在 LIBERO 上平均成功率从 97.2% 提升至 99.05%；在 LIBERO-Plus 上从 79.5% 提升至 80.6%；在 RoboTwin 子集上 LEON 与原模型保持相近（84.13% vs 84.50%）。受控系统实验亦表明 LEON 在外部振幅、非线性摆、驱动振荡器等情境下表现出更好的外推与条件敏感性。

**⚠️ 局限性**

局限性包括：1) 对于高度非线性或跨域的动态变化，算子子空间的低秩限制可能不足；2) 需要额外设计 observable 映射与上下文函数，模型可解释性与实现复杂度提升；3) 仅在当前两种耦合方式下验证，未探究更广泛的 WAM 架构或更大规模任务；4) 依赖训练数据的分布相近性，若训练与测试动态规律差异极大，结构优势可能减弱。

---

## 549. Tensegrity Continuum Robots Enable Task-Adaptive Morphologies for Cooperative Behaviors

**arXiv ID:** 2608.27221 | [PDF](https://arxiv.org/pdf/2608.27221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 550. Enforcing Dirichlet Boundary Conditions in Operator Learning

**arXiv ID:** 2608.27256 | [PDF](https://arxiv.org/pdf/2608.27256v1)

**作者:** Andrew M. Stuart `[一作]` (Caltech), Margaret Trautner `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种通过在神经算子隐藏层使用拉普拉斯齐次Dirichlet本征函数来自动满足Dirichlet边界条件的架构，并给出了其通用逼近性质；

**💡 创新点**

在不需要训练即可严格满足边界条件，并在任意网格和几何形状上实现；同时构造了适用于广义基底的通用理论；

**🔧 技术方法**

基于核积分神经算子框架、GeLU激活、Dirichlet层以及对拉普拉斯本征函数的投影；

**📊 数据集**

使用Darcy流量在单位正方形上的系数-解映射数据（2048样本）和Helmholtz方程在单位圆盘上的系数-解映射数据（11000样本）；

**📈 对比分析**

与标准FNO、训练前后投影FNO以及岭回归进行对比；实验表明在低数据量下DNO约比FNO精度高2倍，在大多数数据规模下仍保持显著优势，且对超参数更鲁棒；

**⚠️ 局限性**

需要预先计算本征函数，计算成本随网格细化而升高；理论假设域仅需Lipschitz边界，无法直接处理混合边界条件或非齐次Dirichlet；

---

## 551. Circuit Condensation: Post-Training that Concentrates a Behavior's Causal Circuit

**arXiv ID:** 2608.27254 | [PDF](https://arxiv.org/pdf/2608.27254v1)

**作者:** Sai Adith Senthil Kumar `[一作]` `[通讯]` (George Mason University), Sai Adith Senthil Kumar (George Mason University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Circuit Condensation 方法，利用后训练逐步剪枝并训练低秩适配器，将模型行为浓缩为更小的因果图；

**💡 创新点**

将电路可搜索性视为可训练属性，采用自适应剪枝‑恢复‑回溯循环，通过权重更新重塑计算路径，从而实现比冻结搜索更小、更易验证的电路；

**🔧 技术方法**

使用边缘归因补丁（EAP‑IG）、低秩适配器（LoRA）、自蒸馏、梯度排序以及增量剪枝与回溯；

**📊 数据集**

四个基于标记对的 token 级任务（间接宾语识别、主谓一致、重复-token 归纳、Python docstring 补全）以及对应 2000/2000/1000 的训练/验证/测试拆分；

**📈 对比分析**

与冻结搜索基线（EAP、EAP‑IG、EAP‑GP、随机排序、C₀）在 4 行为、10 模型（GPT‑2、Llama‑3.2、Gemma、Qwen）上对比；C₁ 在大多数组合下平均减少 30‑50% 边数，保持 5% 内的准确率；能够完成完整的子集与对偶消融验证，并在 token 级 KL 与错误预测上优于对应冻结电路；

**⚠️ 局限性**

贪婪搜索不保证最优；需要后训练适配器导致推理成本不变；部分任务（如 docstring、agreement）对浓缩敏感；能力门在部分模型几乎不生效；仅在 token 级任务与头级边上评估，未对大模型或更细粒度结构验证；

---

## 552. UniFLM: United Segmentation and Measurement on Fetal Limb Ultrasonic Image

**arXiv ID:** 2608.27240 | [PDF](https://arxiv.org/pdf/2608.27240v1)

**作者:** Zeen Zhou `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究提出了UniFLM框架，并构建了包含1690张超声图像的Fetal Limb Bones (FLB) 数据集，用于胎儿长骨分割与测量；

**💡 创新点**

创新点在于设计了三大模块：Semantic Alignment Skip Connection (SASC) 解决语义缺口；Positive Sampling (PoSamp) 在瓶颈处去噪并强化重要特征；Point Regression Mapping (PRM) 模拟临床医生标注模式，实现精准端点回归；

**🔧 技术方法**

技术实现基于深度U‑Net骨干网络，融合通道与空间跨注意力、适应阈值去噪、残差强化以及基于图像分割的粗细端点回归；

**📊 数据集**

使用的主要数据集为自建的FLB数据集，涵盖 humerus、femur、tibia‑fibula、radius‑ulna 四类长骨，并由三位资深超声医生标注；

**📈 对比分析**

在FLB数据集上与多种经典与前沿方法（U‑Net、TransUNet、Swin‑UNet、MedSAM、SAM‑US、VM‑UNet 等）进行对比，UniFLM在 Dice、IoU、MAE 等指标上均居首位，且实现 105 FPS 的实时推理；

**⚠️ 局限性**

局限性包括：数据集规模与病理样本有限，模型在骨骼重叠、极端胎龄、极端成像角度时表现下降，且仅基于 2D 图像，缺乏 3D 时空一致性与多骨关系建模。

---

## 553. Sophistication in GenAI Use: Field Evidence from a Large Firm

**arXiv ID:** 2608.27364 | [PDF](https://arxiv.org/pdf/2608.27364v1)

**作者:** Nicholas J. Hallman `[一作]`, Jaime J. Schmidt `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

分析大型专业服务公司后勤员工在2025年8个月内使用生成式AI的复杂度

**💡 创新点**

首次引入对AI使用“复杂度”的多维量化指标并评估其随职级、功能区和培训的变化

**🔧 技术方法**

使用大语言模型对对话记录进行结构化编码，构造三组复合指标

**📊 数据集**

来自KPMG后勤员工的聊天记录及工具使用日志（共713,564条提示，158,496条对话）

**📈 对比分析**

以员工月为单位比较各职级、功能区的指标差异，发现高层及战略等部门使用更复杂，但缺乏显著的时间提升或培训持续效果

**⚠️ 局限性**

仅为描述性研究，未评估输出质量，且对话数据仅来自单一公司，可能不具普适性

---

## 554. PAWBench: How Far Are We from Probabilistically Aligned World Modeling?

**arXiv ID:** 2608.27345 | [PDF](https://arxiv.org/pdf/2608.27345v1)

**作者:** Yuandong Pu `[一作]` (Shanghai Jiao Tong University), Xi Chen `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了PAWBench基准，评估视频生成器在相同初始观察与动作下能否生成符合真实物理过程分布的多样化视频，并在此基础上对11种当前模型进行实验，发现它们均无法实现概率对齐。

**💡 创新点**

首次提出概率对齐概念和分布层面的评估指标，设计两套评估（Calibration与Coverage）与自动终态读出协议PAWEval，构造50个可解析的物理情景基准，并系统探究语言提示、噪声采样与微调等干预方式。

**🔧 技术方法**

利用统计分布评估、总变差距离、离散终态读出自动化、Couple-to-Control噪声耦合、LoRA微调、以及多语言模型（Gemini、GLM、GPT）进行提示与预测。

**📊 数据集**

使用人工设计的50个场景（含初始图像、动作提示、有限终态集合），部分场景附有分析推导的参考分布；还使用VLM模型作为对比与提示来源。

**📈 对比分析**

通过对比各模型的Calibration TVD、Coverage覆盖率、场景通过率SPR等指标进行评估；实验结果表明，没有任何模型在概率分布和支持覆盖两方面均表现优异，且多数模型在某一维度上表现较好。

**⚠️ 局限性**

局限性包括：只评估终态离散标签，未捕捉轨迹级动力学；评估基于有限样本，可能无法完全揭示模型分布；基准场景受控，缺乏更长时延、交互式环境；缺乏针对概率对齐的学习目标与训练方法。

---

## 555. One Model, Many Minds: Unlocking Multi-Agent Synergy in a Single Agent via Mixture of Roles

**arXiv ID:** 2608.27338 | [PDF](https://arxiv.org/pdf/2608.27338v1)

**作者:** Zhichen Zeng `[一作]` (University of Illinois Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Mixture of Roles（MoR）模块，能在单一LLM单轮推理中动态组合多种角色专精以提升推理和人格化表现

**💡 创新点**

创新点在于将多角色专精迁移到激活空间，通过可学习的代码簿与轻量级路由器在查询时生成多视角的steering向量，兼顾效率与适应性

**🔧 技术方法**

技术包括可学习的激活向量代码簿、查询感知路由器（基于多头交叉注意力+MLP+Top‑K稀疏路由）、三阶段SFT预训练及后置Group Relative Policy Optimization（GRPO）

**📊 数据集**

评测使用MMLU、TriviaQA、MATH、GSM8K、MedQA等推理基准和PersonalityBench的五大人格维度

**📈 对比分析**

与单体角色专精方法相比，MoR平均提升约2.2%；与多代理系统（MAS）相当，但令token消耗减少约20倍，显示出更优的性能/效率比

**⚠️ 局限性**

局限性包括需要访问LLM内部隐藏状态、代码簿向量为潜在表示且缺乏可解释性，以及在生成过程中仅一次性路由，缺乏更细粒度的动态适配

---

## 556. When Context Gets Root: Privilege Escalation in LLM Harnesses

**arXiv ID:** 2608.27299 | [PDF](https://arxiv.org/pdf/2608.27299v1)

**作者:** Xingbang He `[一作]` (Nanjing University), Bing Mao `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种利用代理侧上下文重建提升攻击者内容指令权限的技术——指令特权升级，并在多种编码代理系统上实现攻击。

**💡 创新点**

创新点在于发现并利用代理上下文重建过程中将低权限工具输出提升为高权限用户/系统级指令，从而绕过指令层级和自动权限审查。

**🔧 技术方法**

采用多代理委派、持久目标、定时任务、定制子代理等机制实现指令升级，并基于LLM指令层级、自动权限审查模型等技术。

**📊 数据集**

在六个编码代理主机与对应模型（Codex、Claude Code、Qwen Code、Gemini、Kimi、OpenCode）上进行实验，使用13个安全攻击目标（机密性、完整性、可用性、远程代码执行）。

**📈 对比分析**

与基线注入/角色混淆攻击对比，指令升级攻击在全访问与自动权限审查模式下均能完成所有13个目标，成功率高达100%，而基线攻击几乎无效。

**⚠️ 局限性**

限制在于只考虑未加混淆的纯文本工具输出，未评估对不同模型或更复杂的防御机制（如多重权限审查）的鲁棒性。

---

## 557. LLMs Can Design Near-Optimal OR Algorithms

**arXiv ID:** 2608.27296 | [PDF](https://arxiv.org/pdf/2608.27296v1)

**作者:** Jackie Baek `[一作]` `[通讯]` (New York University), Jackie Baek (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大型语言模型在库存控制、排队网络和组合优化等三类运营研究问题中的算法设计能力，并通过单次查询直接生成可重用的可解释算法。

**💡 创新点**

在无提示、无训练的条件下，前沿LLM能够直接生成与或超过现有专业方法相当的算法，显示LLM在算法设计层面的潜力。

**🔧 技术方法**

使用OpenAI GPT‑5系列（gpt‑5.1、5.4、5.6‑sol）与Anthropic Claude‑Fable‑5等LLM，采用单查询、Python沙箱与固定计算预算的两级调用（实例级与类级）进行实验。

**📊 数据集**

采用库存控制、排队网络与组合优化三大经典基准，分别包括约34个库存实例、13个排队网络实例和3,393个组合选择实例。

**📈 对比分析**

将LLM输出的解与现有手工调参或最优动态规划/深度RL等方法进行精确或仿真比较；在大多数实例上，强模型gpt‑5.6‑sol实现与最佳方法相当或略优，尤其在类级算法下保持高效。

**⚠️ 局限性**

结果受限于已知算法文献，未证明近似保证，对极端/未知实例的鲁棒性不足，并可能因公共基准存在数据污染导致实验结果受影响。

---

## 558. Deterministic Identification over Additive Gaussian Channels

**arXiv ID:** 2608.27243 | [PDF](https://arxiv.org/pdf/2608.27243v1)

**作者:** Jonathan E. W. Huffmann `[一作]` (Technical University of Munich), Holger Boche `[通讯]` (Technical University of Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并证明了在任意零均值高斯噪声下，确定性识别（deterministic identification）的容量为 1/2，且与功率约束、噪声记忆性几乎无关。

**💡 创新点**

创新点在于：
- 将格理论（Minkowski‑Hlawka 定理、球形装配）与确定性识别问题结合，首次给出连续字母通道的精确容量；
- 证明了该容量不受平均或最大功率约束的影响，突破了传统传输容量受功率限制的局限；
- 提出一种基于格的代码构造思路，可在任意块长下实现近似容量。

**🔧 技术方法**

使用了格理论（尤其是 Minkowski‑Hlawka 定理、Blichfeldt 定理、球形装配理论）、极限分析（Stirling 近似）、统计学中的总变差与 KL 散度（Pinsker 不等式）以及高斯通道的矩阵分析。

**📊 数据集**

未使用实验数据集，全部为理论推导与解析证明。

**📈 对比分析**

通过与已知的随机化识别容量界定（1/4–1 之间）对比，证明了在确定性识别模式下容量达到上限 1/2；相较于传统通道编码，识别速率更高，且不随功率变化；理论上在所有块长下都能达到。

**⚠️ 局限性**

局限性包括：
- 仅适用于零均值且协方差矩阵特征值有下界的加性高斯通道；
- 对于存在零特征值的噪声过程，容量会无界，需特殊处理；
- 代码构造依赖于格的存在性理论，未给出可直接实现的构造算法；
- 结果仅针对确定性识别，未考虑随机化识别的更高容量。

---

## 559. A Point-of-Prescription Safety-Check System for Adverse Drug Reactions in Rural Bangladeshi Hospitals: A Feasibility Study

**arXiv ID:** 2608.27239 | [PDF](https://arxiv.org/pdf/2608.27239v1)

**作者:** Shahir Abdullah `[一作]` `[通讯]` (United International University), Shahir Abdullah (United International University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并评估了一种基于智能手机的轻量级处方安全检查系统，用于在孟加拉农村医院及时识别已记录的严重不良药物反应。

**💡 创新点**

创新点包括：检索式而非预测式匹配、默认静默且按严重程度触发的警报设计、将身份捕获放在登记阶段、以及对现场高通量工作流的可行性评估。

**🔧 技术方法**

技术手段：手机拍照、基于区域提示的 OCR 提取、品牌到活性成分的词典映射（MedEx/DGDA）、检索匹配与简洁的用户界面。

**📊 数据集**

数据集：国家药品参考制订的品牌-成分词典；严重反应参考列表；本地门诊退档的历史病例与实时拍照处方数据。

**📈 对比分析**

比较方法：通过回顾性重跑已知反应案例检验召回率，结合前瞻性工作流程指标（时间、点击数、完成率）与可用性评估；目前未提供精确数值，预期在高特异性、低警报疲劳方面表现良好。

**⚠️ 局限性**

局限性：未评估临床结果；单一站点、低严重事件基率导致统计功效不足；依赖软键（手机号）可能产生错误匹配；仅覆盖严重反应列表，未考虑其他不良事件；OCR 识别误差与品牌词典覆盖范围受限。

---

## 560. Your Voice Cloning System is Secretly a Voice Anonymizer

**arXiv ID:** 2608.27360 | [PDF](https://arxiv.org/pdf/2608.27360v1)

**作者:** Romolo Muletta `[一作]` (ZHAW School of Engineering), Jan Deriu `[通讯]` (ZHAW School of Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用已训练好的多语言语音克隆模型 XTTSv2 进行说话人匿名化，改造其推理流程并加入迭代细化，以在七种欧洲语言下实现说话人信息脱敏，同时保持语言内容、韵律和音质。

**💡 创新点**

创新点：①直接复用 XTTSv2 进行匿名化，无需重新训练；②将原始音频的 VQ‑VAE 码本与伪说话人条件拼接，让 GPT‑2 在保持韵律的同时转换说话人身份；③提出以声学相似度和可懂度的调和平均值为指标的迭代细化策略。

**🔧 技术方法**

采用的技术包括 XTTSv2（VQ‑VAE、Perceiver conditioner、GPT‑2 backbone、HiFi‑GAN vocoder）、Whisper‑Large‑V3 ASR、ECAPA‑2 说话人嵌入与 ASV、伪说话人池构造、以及迭代细化算法。

**📊 数据集**

使用 Common Voice 23.0 与 Multilingual LibriSpeech 两个多语料库，涵盖英语、德语、法语、西班牙语、意大利语、葡萄牙语和荷兰语，共计 8,784 位说话人。

**📈 对比分析**

与 SALT（基于 WavLM 的自监督隐私方法）和 MultiLingual（GAN+多语 ASR/TTS）两种基线对比，采用 EER、WER 与 ΔUTMOS 评估。结果显示本方法在 EER 上平均达 0.49（接近理论极限 0.5），WER 在 CV 数据集降至 0.16（比 SALT 降低约 38%），ΔUTMOS 在 CV 上提升至 +0.17（相较于 SALT 的 -0.74 与 MultiLingual 的 -0.62），在 MLS 数据集也保持更好的质量。人类评测 CMOS 在英语 MLS 上为 -0.90，优于 SALT 的 -1.00 与 MultiLingual 的 -1.85。

**⚠️ 局限性**

局限性：仅在七种欧洲语言中验证，非欧语的适用性未知；对抗性攻击下的鲁棒性待进一步评估；伪说话人池依赖高质量 MLS 训练集，低资源语言可能难以构建；迭代细化虽提升效果但增加计算成本；模型依赖大规模预训练 TTS，资源需求高。

---

## 561. Comparative Evaluation of 3D Reconstruction Methods for Immersive Visualization of Laboratory Objects

**arXiv ID:** 2608.27301 | [PDF](https://arxiv.org/pdf/2608.27301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 562. Consensus with Stochastic Broadcast

**arXiv ID:** 2608.27336 | [PDF](https://arxiv.org/pdf/2608.27336v1)

**作者:** Pierre Fraigniaud `[一作]` (Institut de Recherche en Informatique Fondamentale), Sergio Rajsbaum `[通讯]` (Institut de Recherche en Informatique Fondamentale)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21`

**🎯 论文内容**

研究在随机广播模型下的二值共识问题，提出并分析单轮与多轮算法，并给出误差概率的上下界。

**💡 创新点**

将组合拓扑方法推广到概率情形，引入简化Kripke图并证明 n≥3 时单轮共识存在不同阈值的最优算法；设计在给定传输次数下达到最优误差的多轮算法。

**🔧 技术方法**

使用组合拓扑（协议复合）、Kripke图与超立方体结构、概率分析以及最小权重割等技术。

**📊 数据集**

该工作为纯理论分析，无实验数据集。

**📈 对比分析**

通过理论下界与上界的比较，证明所给算法在各阈值区间内达到最优误差概率；多轮算法在传输次数上实现最优误差概率。

**⚠️ 局限性**

仅在单轮情况下给出 n≥3 的阈值最优性；多轮结果仅针对传输次数最优，未给出固定轮数的最优误差；对某些 p 区间仍未完全最优；缺乏实验验证。

---

## 563. Verify Smarter, Evolve Further: Efficient Harness Evolution through Behavior-Aware Verification

**arXiv ID:** 2608.27311 | [PDF](https://arxiv.org/pdf/2608.27311v1)

**作者:** Jinghan Xu `[一作]` (Fudan University), Deqing Yang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为HarnessLens的预算感知框架，用于在交互预算有限的情况下通过行为感知验证自动演化语言模型代理的 harness。

**💡 创新点**

创新点在于利用任务空间与轨迹诊断联合生成候选修改，并根据支持轨迹对验证任务进行行为相关选择，同时通过可归因证据门控避免错误更新，从而在有限预算下显著提升性能。

**🔧 技术方法**

技术包括任务空间探索、组件空间探索、轨迹诊断（经验提取与分析）、基于行为的验证批次选择、归因证据门控以及增量式 harness 更新。

**📊 数据集**

实验使用了四个公共 benchmark：τ^2-bench Retail、τ^3-bench Banking Knowledge、Terminal-Bench 2.0 与 BIRD Mini-Dev 的挑战子集。

**📈 对比分析**

与 Self-Harness、Meta-Harness、HarnessFix 三个基线在同一 harness 与 benchmark 下进行对比，结果显示 HarnessLens 在仅使用 200 交互单元的情况下平均提升 7.6%–13.6% 的通过率，并在大多数 harness–benchmark 组合中取得最高或同等表现。

**⚠️ 局限性**

局限性包括仅在单一 LLM 家族、三种 harness 与四个 benchmark 上评估，未验证在更广泛模型或开放式环境下的有效性；预算计数未归一化 token/延迟/成本，且未考虑更复杂的工具或权限交互。

---

## 564. QuantumBoostNet: A Hybrid Classical-Quantum Architecture for Enhanced Accuracy in Cardiac Ultrasound View Identification

**arXiv ID:** 2608.27302 | [PDF](https://arxiv.org/pdf/2608.27302v1)

**作者:** Mihai Udrescu-Milosav `[一作]` (Politehnica University), Gerhard-Paul Diller `[通讯]` (University Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 QuantumBoostNet 这套两阶段训练的混合经典‑量子架构，用于心脏超声视图识别，并实现了在 C/SHD 超声数据集上的高精度分类；

**💡 创新点**

创新点在于：① 通过动态混合参数 α（由可学习的 gate 控制）实现量子与经典分支的柔性融合；② 两阶段自适应训练策略，先训练量子分支再逐步引入经典分支；③ 在同一 10‑qubit 变分电路上实现高性能量子头；④ 结合 ResNet‑18 的强大特征提取，解决量子信息瓶颈问题；

**🔧 技术方法**

采用了 ResNet‑18 作为特征提取 backbone，10‑qubit 参数化量子电路（PQC）与 4 层变分层、CNOT 纠缠、Pauli‑Z 测量，混合头为两层 MLP；训练使用 Adam + cosine annealing，混合系数通过 sigmoid 变换或线性退火实现；

**📊 数据集**

主要使用了 262 病例 C/SHD 心脏超声视图分类数据集（17 类，后裁剪为 14 类），以及标准图像分类基准 FashionMNIST 与 MNIST 用于验证通用性；

**📈 对比分析**

通过 10‑fold 交叉验证与统一实验设置（同等 epoch、学习率、批量）进行模型比较；在心脏视图分类上，QuantumBoostNet V1 取得 77.19% 的平均准确率，优于最优经典模型 ViewResNet 的 75.95%（+1.24 pp）且方差最低；在 FashionMNIST 与 MNIST 上，QuantumBoostNet V3 分别获得 93.96% 与 99.61% 的准确率，均高于所有经典与其他混合模型；

**⚠️ 局限性**

主要限制是实验均在 PennyLane 量子模拟器上完成，未在真实量子硬件上验证，量子电路受限于 10‑qubit 的规模，信息压缩与噪声模型仍属于理论层面；未来需在硬件上评估、扩大 qubit 数量、进一步验证噪声鲁棒性与可扩展性。

---

## 565. Low-ASR Backdoors: Exploiting Attack Success Rate Reduction and Attacker-Defender Asymmetry

**arXiv ID:** 2608.27288 | [PDF](https://arxiv.org/pdf/2608.27288v1)

**作者:** Arham Riaz `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ting Yu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了低攻击成功率（ASR）下的后门模型，提出逆向训练框架以控制ASR并系统评估现有防御方法的鲁棒性。

**💡 创新点**

创新点在于将ASR视为攻击者可控变量，利用逆向训练从高ASR后门逐步降至低ASR，并揭示传统防御在低ASR情况下的失效机制。

**🔧 技术方法**

采用的技术包括逆向训练（反向标签训练）、梯度优化的后门激活实验、以及对Neural Cleanse、STRIP、FreeEagle和DeBackdoor等四种主流防御的评估。

**📊 数据集**

实验数据集为MNIST、CIFAR‑10和GTSRB，使用的网络结构包括LeNet、SimpleCNN、ResNet18/50、VGG16等。

**📈 对比分析**

与传统高ASR后门对比，低ASR后门在所有四种防御上均被误判为干净模型；梯度优化激活实验表明，低ASR后门所需的迭代步数显著低于干净模型，表明后门功能依然可用。

**⚠️ 局限性**

局限性包括仅在图像分类任务中验证，逆向训练是单一的低ASR生成方法，其他生成方式或跨模态任务的表现尚未探究；此外，实验仅覆盖了有限的防御技术，其他未知防御可能表现不同。

---

## 566. Sidecar: Training-Free Semantic Reuse for Character-Consistent Free-form Visual Storytelling

**arXiv ID:** 2608.27280 | [PDF](https://arxiv.org/pdf/2608.27280v1)

**作者:** Sibo Dong `[一作]` (Georgetown University), Sarah Adel Bargal `[通讯]` (Georgetown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Sidecar，一种无训练、可插拔的语义增强模块，用于在自由形式视觉故事生成中保留并注入初始角色描述中的实体级语义，从而提升后续帧的角色一致性。

**💡 创新点**

创新点在于：1) 在文本编码器层面临时插入“侧车”语义 token，实现对初始角色描述的层级语义提取与注入；2) 仅修改文本编码过程，无需改动扩散模型架构或进行额外训练；3) 针对不同文本编码器（CLIP、T5）采用专属集成策略，实现跨模型的兼容性。

**🔧 技术方法**

技术细节包括：在 CLIP 的每个 Transformer 层提取对应的角色描述隐藏状态；在后续提示编码时，在 BOS 位置后插入这些语义 token，并在该层完成 self‑attention 后移除；在 FLUX 中，对 CLIP 使用同样方法，对 T5 采用先拼接初始提示与当前提示再切片的方式。所有步骤均在推理时完成。

**📊 数据集**

使用了 FreeStoryBench 的单角色子集（100 组故事，每组 6 场景），在混合引用设置（首帧完整描述，后续使用类型/代词）下进行评估。

**📈 对比分析**

与多种基线（SDXL、FLUX、StoryDiffusion、ConsiStory、FreeStory、OnePromptOneStory）对比，利用 CLIP、DINO 与 DreamSim 三个指标。Sidecar 在所有基线上均实现了显著提升，尤其在 DINO 与 DreamSim 上的改进最大；同时对 GPU 内存和推理时间的影响仅为 0.2%–1%。

**⚠️ 局限性**

局限性：依赖首帧明确、完整的角色描述；目前仅评估单角色情景，扩展到多角色需要可靠的实体提取与消歧；CLIP 的 77-token 上下文限制可能导致过长初始描述被截断。

---

## 567. Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search

**arXiv ID:** 2608.27266 | [PDF](https://arxiv.org/pdf/2608.27266v1)

**作者:** Yuan Chang `[一作]` (Purdue University), Xiaoqi Chen `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Naive Prompt Optimization (NPO)，一种基于教师模型的轻量级单线索提示优化方法，并在 IFBench、HotpotQA 与 TextArena 22 款游戏上与 GEPA、GRPO 等方法进行对比实验。

**💡 创新点**

创新点在于：①仅维护单一提示线索；②利用完整的 rollout 轨迹与奖励作为丰富反馈；③通过教师模型的推理能力替代复杂搜索；④在教师模型更强时显著提升回合效率和性能。

**🔧 技术方法**

技术细节包括：迭代提示改写、滑动窗口 roll‑out 反馈、LLM‑as‑optimizer 教师模型、受限解码与共享伪随机种子；与 GEPA 的多候选池、Pareto 选择、GRPO 的 LoRA 权重微调等进行对照。

**📊 数据集**

使用的数据集有：IFBench（指令遵循与约束验证）、HotpotQA（多跳问答）、TextArena 22 款交互式游戏；每个数据集均包含训练、验证与 roll‑out 样本。

**📈 对比分析**

在 IFBench 与 HotpotQA，NPO 在比 GEPA 更少的 roll‑out 数下取得相当或更优的验证性能，尤其当教师模型为 GPT‑5.5 时提升最为显著；在 TextArena，NPO 与 GEPA 的表现相近，GRPO 在部分游戏表现更好；此外，NPO 生成的提示可直接迁移到更大或跨家族学生模型，仍能获得显著提升；实验还证明优化过程未泄漏验证答案。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限任务；未测试长时程或更大规模闭源学生模型；RL 在某些任务上不稳定；对长期任务的上下文窗口需求更高；教师需接收学生的 roll‑out 轨迹，可能涉及隐私与安全问题。

---

## 568. Rank-Three Projections and Minimal Multiplicity Bipartitions of Path Complements

**arXiv ID:** 2608.27227 | [PDF](https://arxiv.org/pdf/2608.27227v1)

**作者:** Jintao Fei `[一作]` (JD.com), Jiangying Luo `[通讯]` (Tsinghua University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了任意至少六个顶点的路径补图的最小可实现特征值乘法分区为[ n‑3, 3]，即其可通过秩为三的正交投影实现且不存在秩一或秩二的实现。

**💡 创新点**

创新点在于构造了一个六向量锚定集合，其外积生成整个三维对称矩阵空间，并利用吸收引理将任意长度的路径拓展为解析可扩展的帧，从而消除对顶点数模三除数的限制。

**🔧 技术方法**

主要技术包括：正交投影与帧理论的等价性、严格可扩展帧（scalable frame）的构造、有限避免（finite avoidance）证明路径链的可扩展性，以及符号计算验证权重的正性与正交性。

**📊 数据集**

本文不使用实验数据集，而是纯粹的符号代数与构造证明，所有向量和权重均为整数或有理数，完全可复现。

**📈 对比分析**

与之前仅针对 3∤n 的构造不同，本文提供了统一构造，对所有 n≥6 的路径补图给出相同的秩三实现；由于构造是显式可计算的，性能不涉及计算复杂度，理论上已满足最优乘法分区。

**⚠️ 局限性**

局限性在于尚未证明所构造投影具有强谱性质（SSP），且对 n≤5 的情况仅给出手工分析，未来可进一步探究是否存在更高秩或不同维度的最优实现。

---

## 569. DINOcular: Self-Supervised Visuospatial Representations

**arXiv ID:** 2608.27226 | [PDF](https://arxiv.org/pdf/2608.27226v1)

**作者:** Farkhat Almukhamedov `[一作]` (University of Bonn), Hermann Blum `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种自监督框架 DINOcular，能够从 RGB‑D 观测中学习既包含视觉语义又包含几何结构的通用视觉表示。

**💡 创新点**

创新点在于：① 将深度信息以三维 RoPE 的形式嵌入 Transformer 的位置编码，直接在 inter‑patch 级别引入空间先验；② 在 intra‑patch 级别通过轻量级 MLP 结合像素级深度，实现局部几何编码；③ 设计多视角一致性损失（对比学习形式）与 DINO / iBOT 结合的自监督目标，强化三维一致性。

**🔧 技术方法**

核心技术包括：视觉 Transformer（Swin），3D Rotary Position Encoding，深度 MLP 局部几何编码，DINO / iBOT 的教师学生蒸馏框架，KoLeo 正则化，基于 3D 对比损失的多视角一致性约束。

**📊 数据集**

训练数据：ImageNet‑1k 及其通过 monocular depth 与 MVImgNet2.0 生成的 RGB‑D；测试数据包括 NYU Depth V2、SUN RGB‑D、Cityscapes、ADE20k（无真实深度）、Probe3D（NAVI、ScanNet）以及单次姿态估计数据集。

**📈 对比分析**

与基线（DINO、DINOv2/v3、DFormerv2、DUNE、MultiMAE、Sonata 等）在语义分割、3D 对应/姿态估计等任务上比较，DINOcular 在 3D 相关任务上取得显著提升（如 ScanNet 上的 25%+ 精度提升），在语义分割上保持与同规模模型相近甚至略优的性能；在大尺度基准（如 1cm‑1° 误差阈值）表现略逊，但整体竞争力强。

**⚠️ 局限性**

局限性：1) 需要 RGB‑D 以及多视角数据，数据获取成本高；2) 训练规模有限，尚未验证在更大规模数据和模型下的可扩展性；3) 多视角一致性损失与语义任务存在权衡，削弱多视角训练时的语义表达；4) 对不同深度传感器的鲁棒性虽有测试，但在极端噪声或稀疏深度下的表现未系统评估。

---

## 570. STEP: State-Aware Task Estimation and Planning with Multi-Modal LLMs for Human-Robot Collaboration

**arXiv ID:** 2608.27225 | [PDF](https://arxiv.org/pdf/2608.27225v1)

**作者:** Maitrey Gramopadhye `[一作]` (University of North Carolina at Chapel Hill), Soshi Iba `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于MM‑LLM的多阶段规划框架，能够显式估计并跟踪工业组装任务中的结构状态并预测后续动作，提升人机协作效率。

**💡 创新点**

核心创新在于将结构状态表示为JSON形式并通过连续“rollout”递归推进状态转移，从而在生成动作计划时量化距离目标的差距，显著提高可执行性与最终误差。

**🔧 技术方法**

使用多模态大型语言模型（GPT‑4o等）进行任务估计、状态生成、动作预测与状态传播，并结合链式思维提示与多次查询实现状态推理。

**📊 数据集**

在一套基于仿真工作站的木块组装数据集（495个实例，19名操作者，8种结构）上进行评估，使用手动标注的任务、动作、图像与结构状态。

**📈 对比分析**

与采用单一动作预测的AntGPT基线相比，本文方法在30–90%任务完成率下动作可执行率提升32.8%，最终状态误差降低14.8%，且在不同MM‑LLM上均保持优势。

**⚠️ 局限性**

主要限制包括：1）对特定组装场景的依赖，扩展至其他任务需重新设计提示与状态结构；2）大幅增加MM‑LLM查询次数导致运行时延长；3）任务与状态估计错误会累计放大，影响后续性能。

---

## 571. BALMS: Benchmarking Agentic LLMs for Longitudinal Mental Health Sensing

**arXiv ID:** 2608.27219 | [PDF](https://arxiv.org/pdf/2608.27219v1)

**作者:** Yu Yvonne Wu `[一作]` (Dartmouth College), Andrew Campbell `[通讯]` (Dartmouth College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了BALMS基准，用于评估LLM驱动的智能体在长时序心理健康监测中的表现。

**💡 创新点**

创新点在于将LLM智能体与长时序可穿戴数据结合，定义两类任务（闭合式分数预测和开放式推理）并系统评估三种智能体范式。

**🔧 技术方法**

采用了prompt、工具（代码执行）和记忆（检索）三种智能体架构，并在五种LLM骨干（OpenAI Claude、Qwen、Mistral等）上实现。

**📊 数据集**

使用了三组真实长期可穿戴/手机感知数据集：DiversityOne、PMData 和 GLOBEM。

**📈 对比分析**

通过与平均基线、不同骨干和智能体范式对比，发现强大骨干或语义压缩特征下零样本预测可超过基线，链式推理提升部分模型性能，但长时间窗口对prompt型效果衰退，检索型模型在更长历史上表现最稳健，整体仍低于人工标注精度。

**⚠️ 局限性**

局限在于仅评估单一智能体范式、缺乏前瞻性交互实验、使用LLM评判而非专业临床评估，以及对不同人群普适性和安全性的考量不足。

---

## 572. UrbanGround: From Local Perception to Spatial Agency in a Real-Scale City

**arXiv ID:** 2608.27456 | [PDF](https://arxiv.org/pdf/2608.27456v1)

**作者:** Tianjie Ju `[一作]` (Shanghai Jiao Tong University), Zhuosheng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于香港完整地理数据的物理仿真环境，并对多模态大语言模型（MLLM）在从局部感知到长距离导航及动态适应等不同尺度下的空间能动性进行评估。

**💡 创新点**

首次提出一种“空间能动性阶梯”评估框架，将任务从单纯的问答、短程导航、长程导航、隐式目标搜索、时间窗口、多目标规划到动态环境（道路封闭、行人干扰）逐步递增，并用实际地理空间来测试模型的全局位置保持与路线修正能力。

**🔧 技术方法**

使用 Unity 物理引擎、3D Tiles（香港土地处视觉化地图）与 3D 行人网络，配合多模态输入（第一人称 RGB + 交互地图）与结构化动作空间（移动、视角调整、地图操作）实现闭环交互；评估通过对模型生成的答案和轨迹与标注数据进行对比统计。

**📊 数据集**

主要数据集为香港 Lands Department 发布的 3D 可视化地图与 3D 行人网络（覆盖全港），并在该数据上构建 810 个手工验证的实验实例，涵盖不同区块与城市尺度。

**📈 对比分析**

与多种主流 MLLM（GPT‑5.x、Claude‑Opus、Gemini、Doubao‑Seed、GLM‑5V‑Turbo、Kimi‑K3）对比，使用答题准确率、导航成功率、行人网络遵守率、路段可行性/碰撞率等指标；结果显示：在局部问答和短程导航上性能可接受（约70–90%），但长程导航成功率降至0–5%，动态场景中路段修正率低，行人碰撞率高。

**⚠️ 局限性**

局限在于模型无法在长距离或动态变化的城市环境中保持稳定的全局位置估计和路线规划；对天气/时间变化敏感；仅在香港单一城市测试，缺乏跨城泛化评估；评测侧重于“成功率”而非细粒度行为质量。

---

## 573. CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes

**arXiv ID:** 2608.27455 | [PDF](https://arxiv.org/pdf/2608.27455v1)

**作者:** Yufan Wu `[一作]` (Ohio State University), Ting Zhu `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CritICL，一种在推理时利用弱模型结构化失效模式（CritBank）来指导强模型的框架，包含动态和静态两种检索策略；

**💡 创新点**

创新点在于发现同一模型家族中不同规模模型的失效模式高度一致，利用这些失效模式生成批判性示例，通过失效模式驱动的检索实现高效推理；

**🔧 技术方法**

技术包括失效模式标签生成与聚类、批判性示例构造、失效模式驱动的检索、两阶段推理（失效模式预测 + 生成）以及对比实验中的链式思考与外部验证；

**📊 数据集**

主要使用 GSM8K、MATH、AMC23、AIME24、AIME25 等数学推理基准，并在化学、生命科学等其它领域做了扩展验证；

**📈 对比分析**

与标准 ICL、测试时缩放方法（self‑consistency、self‑reflection、LLM‑as‑judge）对比，CritICL 在 Qwen2.5‑32B 和 Qwen2.5‑72B 上分别获得 49.8%/59.2% 的 Pass@1（比 Consistency@7 高 0.3/0.2 点），且仅需 1–2 次生成，显著降低 token 消耗；

**⚠️ 局限性**

局限性在于依赖失效模式分布的一致性，可能在跨家族或极大规模模型上效果减弱；需要预先使用弱模型生成并聚类失效模式，成本仍不为零；动态模式预测需要额外一次推理；

---

## 574. Persona-Execution Separation: An Architecture Pattern for Evolving LLM Agents under Execution Audit

**arXiv ID:** 2608.27427 | [PDF](https://arxiv.org/pdf/2608.27427v1)

**作者:** Yisen Xi `[一作]` `[通讯]` (Independent Researcher), Yisen Xi (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Persona–Execution Separation (PES) 架构模式，解决 LLM 代理在企业环境中既需要角色（persona）自由演化，又需执行行为可追溯的矛盾

**💡 创新点**

核心创新在于将角色与执行放置于不同信任域，通过治理合同桥（审批矩阵、DLP、审计链、数据流分级）实现角色漂移不影响执行可追溯，并保持身份连续性

**🔧 技术方法**

采用 LLM 代理架构、合同桥、审批矩阵、数据损失预防 (DLP)、审计链、数据流分级等技术手段

**📊 数据集**

在金融机构的数字员工平台 FIA Workbench 上进行案例验证，使用多种 LLM 模型（DeepSeek‑v4‑flash、DeepSeek‑v4‑pro、Qwen3.8‑max、Kimi‑k3、GLM‑5.3）对 IC memo 等业务流程进行实验

**📈 对比分析**

通过机制验证 V1（零重验证）和隔离验证 V2（执行记录不受角色漂移影响），以及结构检查；实验结果显示所有 5 种模型下 V1 均为 0，V2 在 4 种模型上通过；与单域实现对比表明 PES 在架构层面实现了 G1‑G3，性能满足审计需求

**⚠️ 局限性**

局限性包括仅在单一开发/试点环境验证，缺乏外部泛化验证；未给出与传统单域实现的性能/成本对比；安全性分析仅为草图；实现仍处于早期阶段，部分功能（身份映射、执行记录与角色版本关联等）尚未完善

---

## 575. Reconstructing Humans and Objects in Interaction using Large Reconstruction Models

**arXiv ID:** 2608.27407 | [PDF](https://arxiv.org/pdf/2608.27407v1)

**作者:** Agniv Chatterjee `[一作]` (University of Texas at Austin), Georgios Pavlakos `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型重建模型生成完整人-物体网格，随后对网格进行分割并拟合人体参数模型，必要时将物体模板对齐，完成单张图像下的人体与物体交互三维重建。

**💡 创新点**

将LRM输出作为几何支架，避免传统投影与接触约束，提供无接触信息、类别无关、可直接利用高质量LRM结果的重建流程。

**🔧 技术方法**

使用Hunyuan3D-2.0（或其他LRM）生成网格，ViTPose+HaMeR进行关键点估计，SMPL-H/VPoser/MANO先验的两阶段优化，基于多视角的点云分割与可选的语义对应对齐。

**📊 数据集**

InterCap、HODome、IMHD 三个室内 HOI 基准，以及 PICO-db 的野生图像进行评估。

**📈 对比分析**

在 InterCap、HODome、IMHD 上以 PA‑CD 指标与 PICO、EasyHOI、OPEN3DHOI 等现有方法对比，均实现更低的 PA‑CD（如 InterCap 人体 9.5 cm、物体 12.97 cm、组合 6.68 cm），并在接触估计上显著优于 DECO。

**⚠️ 局限性**

依赖于 LRM 的重建质量与点云分割的准确性；对小尺寸物体的细节捕捉仍有限，且目前仅适用于单人单物体场景。

---

## 576. How Language Models Organize and Structure Moral Knowledge

**arXiv ID:** 2608.27402 | [PDF](https://arxiv.org/pdf/2608.27402v1)

**作者:** Orion Reblitz-Richardson `[一作]` `[通讯]` (Distiller Labs), Orion Reblitz-Richardson (Distiller Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过为 Moral Foundations Theory（MFT）的六个道德基底分别训练线性探针，提取探针权重向量，分析这些向量在语言模型表示空间中的几何关系，从而研究 LLM 对道德知识的结构化表达。

**💡 创新点**

提出“探针方向几何”方法：利用探针权重向量的余弦相似度、有效维度和层次聚类等指标，揭示模型内部道德基底的整合（integration）结构，而非单一检测或无关联的分离；首次量化道德困境（dilemma）表示的部分组合性与冲突特征；证明该结构在 dense 与 MoE 架构、不同规模以及跨数据集上保持一致。

**🔧 技术方法**

技术手段包括：线性二分类探针（BCE+Adam）、余弦相似度矩阵、PCA 计算有效维度、Ward 层次聚类、置换检验、bootstrap 稳定性评估、噪声注入的 fragility 测试、以及对不同数据集和架构的对比分析。

**📊 数据集**

使用自构造的 240 组最小对（每个道德基底 32 组训练/8 组测试），以及 300 组道德困境场景；对比了与匹配的非道德概念基准（情感、语法、主题等）以及公开的 Moral Foundations Vignettes 进行外部验证。

**📈 对比分析**

与匹配的非道德基准进行对比，发现道德基底方向的平均余弦相似度约 0.26，显著高于 0.013 的非道德基准；与 dense 与 MoE、1B 与 7B 模型对比，发现几何结构一致，均表现出整合型而非分裂型；探针分类准确率在所有层均达到 100%，但结构分析表明 MFT 的 individualizing/binding 群组未被检出。

**⚠️ 局限性**

局限性包括：探针训练样本仅 32 对，导致方向估计噪声较大；置换检验对 MFT 群组划分的统计功效有限；仅在 Ai2 的 OLMo 系列模型上验证，未检验其他模型或数据集；假设道德基底以线性方向编码，可能忽略非线性或多维分布；未能完全区分道德与一般情感/评估性语义；因果关系验证仍属初步，尚缺乏针对特定方向的可解释干预。

---

## 577. Making Clinical Language Models Auditable: Concept-Guided Fine-Tuning for Robust Prediction

**arXiv ID:** 2608.27397 | [PDF](https://arxiv.org/pdf/2608.27397v1)

**作者:** Jin Mu `[一作]` (University of Wisconsin--Madison), Guanhua Chen `[通讯]` (University of Wisconsin--Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种利用稀疏自编码器（SAE）对临床文本进行概念级别解释并在微调过程中抑制非临床模板与格式化短路的可审计模型训练框架CAST（Concept-guided Artifact Suppression Tuning）;

**💡 创新点**

创新点在于将SAE生成的可解释概念直接嵌入训练循环，实现对特定artifact概念的显式抑制，并同时提供每个预测的概念级别审计轨迹;

**🔧 技术方法**

采用Transformer（ClinicalBERT/Clinical-Longformer）与TopK/BatchTopK/Matryoshka等SAE变体、LLM驱动的概念标签、ICD-10检索、残差校正抑制和梯度回归的概念归因;

**📊 数据集**

使用MIMIC-IV 30天出院后死亡预测任务的ICU出院笔记（49,832个病历，1,830阳性，48,002阴性），并对比零样本LLM、标准微调、输入移除、SAE正则化与SAE探测等基线;

**📈 对比分析**

CAST在ClinicalBERT和Clinical-Longformer上均实现了与或优于标准微调的F1、AUROC和PR-AUC，且相较于零样本LLM具有更好校准（更低Brier、NLL、ECE）并提供概念级解释；

**⚠️ 局限性**

局限在于仅在单一医院的MIMIC数据上验证，任务高度不平衡导致绝对F1偏低，需外部验证、临床专家审查概念标签与抑制集，并考虑不同机构、笔记类型和临床任务的泛化。

---

## 578. RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

**arXiv ID:** 2608.27394 | [PDF](https://arxiv.org/pdf/2608.27394v1)

**作者:** Maayan Sharon `[一作]` (Hebrew University of Jerusalem), Tom Hope `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于科学文献的灵感检索基准，定义三类启发性检索任务：解决问题、扩展通用性、细化实例。

**💡 创新点**

创新点在于把检索任务与认知层面的“ideation move”关联，使用句子级的论证标记进行大规模无监督标注，并在此基础上构建可训练、可评估的 benchmark。

**🔧 技术方法**

采用远程监督的语篇标记扩展、LLM 自动标注、规则模板扩增构造标记词典，然后对 BM25、all‑mpnet、ModernBERT、Stella 等检索模型进行 operation‑specific 对比学习微调。

**📊 数据集**

数据集由 3,017,476 条 (q,g) 对组成，覆盖 2015‑2026 年的计算机科学全文论文，并对 17,579 条构造了高质量的银标注评测集。

**📈 对比分析**

通过将预训练模型与细化模型在 silver test 上进行 R@10/M@10 等指标对比，发现微调后 ModernBERT‑embed‑large 在三类任务上分别提升 2.4×、1.7×、1.6×，但整体 MRR@10 仍低于 0.2，说明任务仍具挑战性。

**⚠️ 局限性**

局限性包括：检索范围局限于相邻句子；仅覆盖 CS 文献；依赖语篇标记的准确性，可能忽略非标准表述；并且性能仍远未达到可直接支撑科学创新的水平。

---

## 579. The Project Scheduling Interdiction Problem with Delay Groups

**arXiv ID:** 2608.27386 | [PDF](https://arxiv.org/pdf/2608.27386v1)

**作者:** Fei Wu `[一作]` (KU Leuven), Jannik Matuschke `[通讯]` (KU Leuven)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了项目调度中“延迟组”干扰模型，即在预算约束下挑选若干共享输入（如供应商、团队或平台）导致的相关延迟，以最大化项目完工时间的PSIP‑DG问题；

**💡 创新点**

创新点在于首次将相关延迟结构（延迟组）引入项目调度干扰框架，并证明该问题在多种不确定性集下为NP‑hard、不可近似，随后给出贪心k‑近似算法以及两种基于结构的启发式（SRS和GSS），在大规模实例上实现接近最优的求解；

**🔧 技术方法**

主要技术包括：多维不确定性集的多项式求解、动态规划、对偶化、二次整数规划线性化、贪心分配、子路径重优化搜索与组选择搜索等；

**📊 数据集**

实验采用随机生成的PERT网络（50–5000节点、不同连通度）与随机划分的延迟组，并为每组随机分配延迟增量；

**📈 对比分析**

与完整MIP求解器Gurobi进行比较。SRS在PSIP‑DG_B,1上得到几乎最优解（≤1% gap，平均时间<4s）；GSS在PSIP‑DG_B,∞上优于SRS（gap≤1.4%，平均时间<0.2s），在极大规模（5k节点）实例中几秒即可达到或超越Gurobi的最优值；

**⚠️ 局限性**

局限性包括：只考虑单项目、无资源约束、仅评估最坏情况、对延迟组规模和网络结构的进一步理论分析不足，且在真实工业数据上的验证仍待开展。

---

## 580. FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference

**arXiv ID:** 2608.27384 | [PDF](https://arxiv.org/pdf/2608.27384v1)

**作者:** Zekai Li `[一作]` (UC San Diego), Zhijian Liu `[通讯]` (UC San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了FlashVLA——一种流式动作解码框架，利用不同噪声水平的动作块缓冲区与块级因果注意力，实现了动作块的联合解码，显著降低推理延迟并保持异步执行连续性。

**💡 创新点**

通过在动作块缓冲区中并行解码不同噪声级别的块，并让未来块对已完成块施加因果注意力，既摊销了迭代解码的延迟，又天然补偿了异步执行的时间误差。

**🔧 技术方法**

采用流式分块扩散、块级因果注意力、CUDA Graph加速、FiLM时间条件化以及多缓冲联合微调等技术。

**📊 数据集**

在LIBERO、RoboTwin 2.0、SmolVLA、LingBot-VLA以及真实Franka机械臂的pick&place、whiteboard wiping、table cleaning三大任务上进行评估。

**📈 对比分析**

与同步π_0.5、VLASH、StreamingVLA、Realtime‑VLA、FASTER等基线比较，FlashVLA在一步异步延迟下实现约2.4–2.6×速度提升且任务成功率保持或提升；实时推理可达≈50Hz更新率；真实机器人上实现≥30Hz控制，平均分数提升至84.4%，完成时间提升1.2–1.3×。

**⚠️ 局限性**

需要对预训练VLA进行多缓冲联合微调，且每个回合需进行一次冷启动warm‑up，适用于短任务时开销更明显；未探究从头训练VLA在流式框架下的性能潜力。

---

## 581. WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

**arXiv ID:** 2608.27454 | [PDF](https://arxiv.org/pdf/2608.27454v1)

**作者:** Liyan Tang `[一作]`, Tu Vu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WikiSkill 框架，将持久化 wiki 知识库与迭代式技能进化相结合。

**💡 创新点**

创新点在于引入一个持续记录与更新的 Wiki 层，使经验能够被结构化并在多轮中复用，显著提升技能进化效果。

**🔧 技术方法**

使用三层知识架构（Raw/ Wiki/ Skill）、Inference Agent、Wiki Maintainer、Skill Proposer（ReAct 机制）以及验证门控和回滚机制，构建完整的迭代循环。

**📊 数据集**

在五个多领域基准（LiveMath、SealQA、SpreadsheetBench、OfficeQA、ALFWorld）以及五个模型（Qwen‑3.5‑4B、Qwen‑3.5‑9B、Qwen‑3.6‑27B、Gemma‑4‑31B、Gemini‑3.5‑Flash）上进行评估。

**📈 对比分析**

与 Trace2Skill、EvoSkill、SkillOpt 等现有技能进化方法对比，WikiSkill 在所有模型/基准上平均提升 10–25%（最高 63%+），且收益随模型规模增大而提升，跨模型迁移也能获得显著性能提升。

**⚠️ 局限性**

局限性包括：未考虑技能检索与触发、严格的验证门控排除中性改进、wiki 未实现自动剪枝、实验未覆盖极长时间或极多步任务。

---

## 582. Do User-Authored Permission Policies Improve Protection Against AI Agent Overreach?

**arXiv ID:** 2608.27443 | [PDF](https://arxiv.org/pdf/2608.27443v1)

**作者:** Ting Yan `[一作]` `[通讯]`, Ting Yan

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过一项在线实验比较了三种权限决策方式——每次人工审批(HITL)、模型自动审查(AUTO)与用户预先设定的后果级政策(POLICY)，评估它们对AI代理过度行为阻止、任务完成率、提示次数和总干预时间的影响。

**💡 创新点**

创新点在于引入了“后果级可写规则” (允许/询问/拒绝) 让非专业用户在行动前设定通用权限，并系统化评估其在实际日常代理任务中的效能与局限，揭示用户倾向于保留案例审查而非提前确定。

**🔧 技术方法**

技术上构建了基于Model Context Protocol的代理，使用LLM进行零样本后果分类，对工具元数据进行多标签标注，并结合规则引擎实现权限判定；实验界面采用Web前端和Python后端，数据分析使用贝叶斯模型。

**📊 数据集**

数据集方面，使用了538个MCP来源工具中的120个子集进行人工后果标签标注，并在实验中使用18个预设代理动作（共113名受试者）。

**📈 对比分析**

比较方法采用随机分配到三种条件的被试，并用置信区间/可信区间比较overreach阻止率、任务完成率、提示次数和总干预时间；结果显示POLICY在过度阻止率上比HITL低20.1个百分点、比AUTO低14.5个百分点，提示次数下降至约60%但总干预时间无显著降低。

**⚠️ 局限性**

局限性包括研究仅在模拟单日无真实后果的在线实验中进行，后果类别过于粗糙、缺乏长期使用和专业用户的数据、未预设非劣效界限，并且LLM分类器可能受误导导致规则执行的不确定性。

---

## 583. RedEvoAgent: Automatic Red-Teaming Agent with Experience-Driven Skill Evolution

**arXiv ID:** 2608.27439 | [PDF](https://arxiv.org/pdf/2608.27439v1)

**作者:** Junjie Zhang `[一作]` (City University of Hong Kong), Haoliang Li `[通讯]` (City University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于黑盒红队的自动攻击代理 RedEvoAgent，能通过收集跨案例的攻击轨迹并提炼成可读的攻击技能文档，在产品级执行环境下进行持续的 jailbreak 评估。

**💡 创新点**

创新点包括：① 用工具效果概况与 Deciding‑Tool Attribution 两种经验信号来驱动技能演化，避免检索偏差；② 采用验证锁定（validation ratchet）仅保留提升验证集性能的技能更新；③ 将全轨迹抽象为简洁的 Markdown 技能文档，降低上下文开销并提升可解释性。

**🔧 技术方法**

核心技术包括：GPT‑4o mini 作为攻击模型；多工具攻击工具箱（GCG、AutoDAN、FlipAttack 等）；SkillOpt 进行技能合成与迭代；工具效果评估、轨迹收集、Deciding‑Tool Attribution、验证锁定机制。

**📊 数据集**

使用 Agent Security Bench (ASB) 与 AgentHarm 两个基准，分别涵盖 400/208 条测试案例，结合 MiniMax‑M2.5、DeepSeek‑V4‑Flash、Qwen3.5‑35B 等目标模型与 Claude Code/Codex 执行主机。

**📈 对比分析**

与单一工具基线、RedCodeAgent、MAJIC 等自动红队方法对比，RedEvoAgent 在 ASR、HarmScore 上均超越对手，并在多目标/主机上表现出更高的攻击成功率与更少的工具调用次数；在零射转移实验中，技能文档对不同攻击模型与主机也保持显著优势。

**⚠️ 局限性**

局限性：依赖工具箱质量和多样性，若工具不足或无效将限制性能；验证集可能导致过拟合或偏向特定案例；技能演化过程仍需多轮实验，对资源与时间有一定消耗；在极端安全配置或对抗性主机上可能仍面临瓶颈。

---

## 584. misi: a Metric Inverted Sample Index

**arXiv ID:** 2608.27422 | [PDF](https://arxiv.org/pdf/2608.27422v1)

**作者:** Edgar Chavez `[一作]` `[通讯]` (CICESE), Edgar Chavez (CICESE)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文设计并实现了一种基于随机样本的倒排索引（MISIFU），可在任意度量空间中进行近似最近邻搜索。

**💡 创新点**

创新点在于把NAPP的固定参考词汇换成线性大小的随机样本，构成可组合的“组合器”，实现并行、确定性的构建和仅与样本大小相关的内存占用，并给出了投票筛选的概率保证与分辨率下限。

**🔧 技术方法**

技术包括在样本上构建可插拔的近似最近邻内索引（如GRAFT、HNSW），生成签名并倒排投票，使用idf加权投票进行候选排序，精确验证，支持并行内存映射、流式构建、GPU计数核等。

**📊 数据集**

实验使用SIFT1M、GloVe-200以及Deep 1M/10M/100M等高维向量数据集。

**📈 对比分析**

通过与NAPP、HNSW、DiskANN等基准在召回@10、QPS等指标对比，发现MISIFU在10^6规模下匹配召回时比HNSW慢6–16倍，但在10^8规模时可在内存受限环境下达到与SSD图相当的召回，并在构建速度和可移植性上优于图索引。

**⚠️ 局限性**

局限性包括仅在向量/黑盒度量上评估，更新路径未实现；投票分辨率受高维或极端中心化数据限制；内存受限下需重建内核索引导致最低4–8 GB的开销；跳过、学习式内部权重和批量合并等改进尚未实现。

---

## 585. Algorithms for Robbins' Problem using Markov Decision Processes

**arXiv ID:** 2608.27419 | [PDF](https://arxiv.org/pdf/2608.27419v1)

**作者:** Léonard Brice `[一作]` (Institute of Science and Technology Austria), Jean-François Raskin `[通讯]` (Université Libre de Bruxelles)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文通过将Robbins问题转化为有限状态马尔可夫决策过程（MDP），并提出多种离散化与记忆约简抽象，得到从n=5到n=100的更精确的期望排名上界。

**💡 创新点**

创新点在于：①引入可控离散化参数d的MDP模型；②仅记忆k个最佳候选的历史（k≤n）来显著降低状态空间；③进一步利用“好候选区间”仅记录前l区间内的k个最佳候选，从而在保持精度的同时进一步压缩模型。

**🔧 技术方法**

采用的技术主要包括：马尔可夫决策过程建模、离散化与状态压缩、逆向归纳（backward induction）求解MDP、以及对损失函数的理论推导与证明。

**📊 数据集**

由于研究对象为理论随机过程，没有外部真实数据集；实验基于理论构造的n取值与离散化参数（d、k、l），计算得到的期望排名数值。

**📈 对比分析**

与已知的记忆无关（memoryless）阈值策略（c=1.9469）进行比较；在所有n≤100的情况下，本文方法均得到更小的期望排名上界；实验显示k=2或3即可接近最优。

**⚠️ 局限性**

主要限制在于：①状态空间仍随n、d快速增长，导致内存与计算时间大幅增加；②目前仅给出上界，缺乏相应的下界估计；③对极大n（如>500）仍需更高效的近似或分布式求解方法。

---

## 586. Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information

**arXiv ID:** 2608.27417 | [PDF](https://arxiv.org/pdf/2608.27417v1)

**作者:** Chanho Park `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型中的视觉检索头（VRHs），通过定位图像区域与文本提示的关联来实现视觉引用解析。

**💡 创新点**

首次在通用图像检索任务中发现并证明 VRHs 为稀疏、因果且通用的注意力头，揭示 VLM 视觉检索的核心机制。

**🔧 技术方法**

采用统一的头评分框架，对输出查询词、区域求和聚合与平均跨样本聚合进行组合，使用注意力权重评估并在多模型上进行掩蔽验证。

**📊 数据集**

使用 RefCOCO、RefCOCO+、RefCOCOg、RefSpatialBench、Toloka 等视觉定位基准，以及 VAW、Spatial457、SpatialRGPT、CV-Bench、MMStar、CountBenchQA、MathVista 等视觉问答数据集进行评测。

**📈 对比分析**

与随机掩蔽、OCR、VERA、Localization Heads 等先前方法对比，掩蔽前 20 个 VRHs 可将定位准确率降至 80% 以下，显著优于随机或其他方法，验证了 VRHs 的因果效应。

**⚠️ 局限性**

仅依赖注意力权重识别，未探究信息读取细节；实验仅限单图像输入，未涵盖多图像或视频场景。

---

## 587. Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms

**arXiv ID:** 2608.27409 | [PDF](https://arxiv.org/pdf/2608.27409v1)

**作者:** Siye Wu `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统比较了三种将RLVR训练出的多域专家整合为单一模型的融合范式：Merge、Mix RL和MOPD；

**💡 创新点**

创新点在于首次将三种范式放在同一实验框架下统一评估，并通过任务向量几何和跨域交互分析揭示其性能差异与域相关性的内在联系；

**🔧 技术方法**

采用RLVR（Group Relative Policy Optimization）、任务向量算子（Task Arithmetic）、多教师在线蒸馏（MOPD）以及LoRA微调等技术；

**📊 数据集**

使用Qwen3-4B和Qwen3-8B基础模型，在五个领域（数学、科学、编程、指令跟随、代理）构建训练数据，评估涵盖AIME、GPQA-Diamond、LiveCodeBench、IFEval、IFBench、BFCL v3等基准；

**📈 对比分析**

通过统一基准、共享专家与数据、对比平均分差及单域性能，发现三范式平均差异不超过1.4分，但在单一基准上可达8.6分；Merge最快、成本最低；Mix RL需要更多步骤但无需先训练专家；MOPD收敛最快且受教师限制，整体成本最高；

**⚠️ 局限性**

局限包括：仅评估五个域且仅在两种模型规模上实验，未探讨对更大/多样化域的迁移；融合后仅提升单样本准确率，未显著扩大可解决问题集合；未深入探究不同融合规则下的长期性能与稳健性。

---

## 588. CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases

**arXiv ID:** 2608.27391 | [PDF](https://arxiv.org/pdf/2608.27391v1)

**作者:** Sil Hamilton `[一作]` (Epiq AI Labs), Igor Labutov `[通讯]` (Epiq AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并发布了CorporateBench，一个用于评估大型语言模型在企业级多文档问答任务中的性能的基准集。

**💡 创新点**

创新点在于采用程序化知识库（KB）生成技术，能够在保持逻辑一致性的前提下，生成可扩展至百万级文档的企业通信网络，并提供可验证的真实答案。

**🔧 技术方法**

使用的技术包括RDF/Turtle知识库表示、CL·Haiku 4.5文本生成、模板化邮件与会议生成、知识检索工具（RAG）以及SQL工具进行KB查询。

**📊 数据集**

使用的数据集为四个规模不同（10至10,000名员工）的合成公司Corpus，包含总计263,466篇邮件文档，并公开发布于Hugging Face。

**📈 对比分析**

对比方法包括RAG（检索增强生成）与直接KB查询两种设置，评估五款轻量级LLM（Claude Haiku 4.5、Claude Sonnet 4.5、GPT‑5 Nano、GPT‑5.1、Gemini 2.5 Flash Lite）。在KB查询场景下性能普遍优于RAG，且两者间差距随文档规模增大而扩大，说明RAG在大规模企业语料中表现不足。

**⚠️ 局限性**

局限性包括仅关注邮件通信、未覆盖更丰富的文档类型和沟通渠道、仅模拟90天的组织活动、文档生成依赖单一LLM导致潜在偏差、以及对真实人类写作的适应性未知。

---

## 589. Beyond Harassment: Exploring the Harm Experienced by People with Disabilities in Social Virtual Reality

**arXiv ID:** 2608.27390 | [PDF](https://arxiv.org/pdf/2608.27390v1)

**作者:** Xinran Adeline Li `[一作]` (Johns Hopkins University), Yaxing Yao `[通讯]` (Johns Hopkins University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对社会虚拟现实（Social VR）环境中残障人士（PWD）遭受的伤害进行了系统性调查，并构建了包含19种子类型的五大伤害分类框架。

**💡 创新点**

创新点在于：①首次将伤害概念与残障身份相结合，提出了五大伤害类别（情感伤害、内部化伤害、社交伤害、体感回忆伤害、感官伤害）和19个子类型；②对受害者对每种伤害的严重性感知进行了量化评估；③从批判性残障理论视角重新审视伤害与社会环境的互动。

**🔧 技术方法**

使用的方法包括：系统文献综述、基于Qualtrics的在线问卷调查、主题编码（Reflexive Thematic Analysis）以及描述性统计分析。

**📊 数据集**

数据集为67名来自美国、已公开残障身份且在Social VR中遭遇过骚扰的参与者所提供的调查数据，包含152条骚扰事件与对应伤害描述。

**📈 对比分析**

本文未进行算法或实验性能比较；所述结果主要为定性归纳与描述性统计，无法与其他方法直接对比。

**⚠️ 局限性**

局限性包括：①样本仅来自美国，缺乏跨文化泛化；②受访者多为心理与行动障碍，罕见残障类型代表性不足；③问卷形式限制了事件细节的深度，可能漏报某些伤害细节；④未涉及对照组或纵向跟踪，难以评估干预措施的有效性。

---

## 590. Token-Level Advertising

**arXiv ID:** 2608.27382 | [PDF](https://arxiv.org/pdf/2608.27382v1)

**作者:** Hanbing Liu `[一作]` (Renmin University of China), Qi Qi `[通讯]` (Renmin University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在生成过程的每个 token 级别嵌入广告商影响的 LAMA 机制，兼顾收益与内容质量。

**💡 创新点**

将广告商的本地价值报告与生成决策融合为贝叶斯后验，保证 Markov DSIC、IR，并实现近最优 KL 约束福利。

**🔧 技术方法**

利用软价值函数、贝叶斯推断、共享条件模型（LoRA 加权）、KL 正则化以及强化学习框架。

**📊 数据集**

在 Webis Generated Native Ads 2024 的 Workout、Vacation、Car 三个商业搜索查询集合上进行实验。

**📈 对比分析**

与 6 种基线（先分配/后分配、Reference/Edit/Policy、MOSAIC）比较，LAMA 在平台福利、收入、广告主价值和用户质量上均获得最高分。

**⚠️ 局限性**

实验仅为离线验证，未在真实线上系统部署；对报错假设（完整比较支持）和模型训练复杂度存在局限。

---

## 591. Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

**arXiv ID:** 2608.27370 | [PDF](https://arxiv.org/pdf/2608.27370v1)

**作者:** Kairong Luo `[一作]` (Tsinghua University), Wenguang Chen `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从零开始训练2B参数LLM，构建低成本可复现的预训练 recipe，使用RTX 5090 GPU、FP8、MuonH、Curriculum Model Averaging，训练超过1.4T token，并发布模型、数据与代码。

**💡 创新点**

提供完整可复现的低成本 open-recipe 预训练 pipeline，结合硬件选择、低精度训练、Hyperball 优化、数据预课程与模型平均，实现仅约6.9K USD 的成本，达到 Qwen2.5-1.5B 级别性能。

**🔧 技术方法**

RTX 5090 集群、blockwise FP8、MuonH（Hyperball 约束）、Curriculum Model Averaging、Proxy 数据筛选、Megatron Core、Muon optimizer、数据混合设计。

**📊 数据集**

公共可访问的 web、代码、数学、合成数据，通过 Proxy 实验选取与排序，构建超过1.4T token 的语料。

**📈 对比分析**

采用统一成本会计协议，评估15个基准（数学、代码、推理、知识）平均分；$4.4K 版本匹配 Qwen2-1.5B，$6.9K 版本接近 Qwen2-5-1.5B；在SFT后续实验中，Curriculum 初始化在数学和广泛指令任务上分别提升1.77pp、2.02pp、1.59pp。

**⚠️ 局限性**

仍然仅2B参数规模，硬件依赖 RTX 5090 可能不易迁移到更大模型或不同 GPU；成本未包含数据获取、预处理和研究人力；对模型泛化能力及更大规模、多模态任务的可扩展性尚待验证。

---

## 592. Successive Capacity Growth: Task-Complexity-Driven Width and Depth Expansion for Vision Transformer Encoders in JEPA World Models

**arXiv ID:** 2608.27367 | [PDF](https://arxiv.org/pdf/2608.27367v1)

**作者:** Frederik Berenz `[一作]` `[通讯]` (121 Labs), Frederik Berenz (121 Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Successive Capacity Growth (SCG) 的自适应 ViT 编码器扩展方法，能够在任务训练过程中根据预测误差自动决定是否增宽（增加注意力头）或加深（增加 Transformer 层）并保证函数不变；

**💡 创新点**

创新点在于：①引入任务无关的“测试与验证”机制，在线判断何时扩展；②使用函数保持的宽度/深度扩展技术，确保扩展后模型输出不变；③利用 Sketched Isotropic Gaussian Regularizer (SIGReg) 维持新增语义维度的统计独立性；

**🔧 技术方法**

技术实现包括：Vision Transformer（ViT）编码器、Net2Net 风格的宽度/深度扩展、零初始化的深度扩展、SIGReg 正则化、AdamW 优化器、任务无关的阈值判定与回滚策略；

**📊 数据集**

在三个合成环境上评估：Two-Room (2D)、Push-T (5D) 以及 30-Object Dynamics (60D)；

**📈 对比分析**

通过与固定小型（283K 参数）和固定大型（5.7M 参数）两种基准进行比较；SCG 在两种任务中实现了 49%（宽度扩展）和 20.3%（深度扩展）的预测误差提升，同时在参数利用率上比固定大型高达 56 倍；

**⚠️ 局限性**

局限性包括：仅在合成环境中验证；未评估规划性能；仅在 5.7M 参数规模内实验，未验证更大 ViT；对某些随机种子在 30-Object 环境下扩展失败；

---

## 593. Beyond F1: Evaluating Coverage and Failure Recovery in AI Model Security Scanners

**arXiv ID:** 2608.27424 | [PDF](https://arxiv.org/pdf/2608.27424v1)

**作者:** Qianlong Lan `[一作]` (eBay Inc), Indranil Sanyal `[通讯]` (eBay Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对三款开源模型安全扫描器（ModelScan、ModelAudit、Fickling）在170个Pickle/PyTorch序列化文件组成的145个模型家庭上进行严格的离线基准测试，评估其可用性、决策完整性、条件准确率以及运行时特性。

**💡 创新点**

创新点在于提出了分层评估框架，将“可用性”“决策完整性”“条件准确率”区分开来，揭示传统单一准确率指标隐藏的不足，并深入分析跨扫描器的恢复能力与增量检测效果。

**🔧 技术方法**

技术手段包括：构建人工评审记录与家庭级聚合的基准框架；实现扫描器适配器提取退出状态、日志与警报；计算覆盖率、完成率、决策覆盖率、精确率/召回率/F1、误报率、恢复率、延迟等指标；以及文件重命名鲁棒性实验。

**📊 数据集**

数据集为170个合成Pickle/PyTorch文件，划分为145个模型家庭，其中135个拥有二进制安全真值（70恶意、65良性），10个故意畸形但无标签，用以测试扫描器在未知输入下的表现。

**📈 对比分析**

比较方法：通过覆盖率、完成率、决策率、条件精度/召回/F1、误报率、恢复率与执行延迟对三款扫描器进行对比；结果显示ModelAudit实现100%标签家庭决策覆盖，但误报率高；ModelScan条件F1为100%但决策覆盖仅49.6%；Fickling未提供新增TP但能在ModelScan失效时实现完整恢复，整体表明判断可用性与准确性不一致。

**⚠️ 局限性**

局限性：仅针对Pickle/PyTorch，使用合成数据缺乏真实恶意样本；扫描器版本固定，未评估版本演化；未覆盖运行时安全、模型后门、权重攻击等威胁；文件重命名实验覆盖范围有限；延迟测量仅为内部观察，未提供跨平台对比。

---

## 594. LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics

**arXiv ID:** 2608.27395 | [PDF](https://arxiv.org/pdf/2608.27395v1)

**作者:** Lukas Kuhn `[一作]` (German Cancer Research Center), Florian Buettner `[通讯]` (German Cancer Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种仅使用单一视频编码器、单一损失函数和一个固定超参数的自监督预训练框架 LeVJEPA，借助 LeJEPA 的 SIGReg 正则化实现无崩塌的表示学习，并通过随机 token 丢弃和块因果注意力进一步提升效率与时序建模能力。

**💡 创新点**

创新点包括：① 用统计分布正则化（SIGReg）完全消除表示崩塌，无需目标编码器、预测器或 stop‑gradient；② 在预训练过程中对大量 token 进行随机丢弃，既降低 FLOPs 又充当数据增强，意外提升下游精度；③ 采用块因果注意力，使每帧表示仅依赖过去信息，兼顾时序一致性且无精度损失；④ 在单一实验设置下实现了与 V‑JEPA 2 相比 5.6–20.8 倍的计算节省，并在同等 FLOPs 下在 ImageNet、Kinetics‑400 与 Something‑Something‑v2 上均优于或匹配现有视频/图像自监督方法。

**🔧 技术方法**

使用的技术包括：Vision Transformer（ViT）作为视频编码器；global‑local 视图构造；mean‑squared‑error invariance损失；SIGReg 正则化（基于一维投影的高维分布一致性检验）；随机 token 丢弃；块因果多头注意力；以及轻量级投影器（去掉后不再使用）。

**📊 数据集**

主要数据集：K710（Kinetics‑400/600/700 的 20% 采样版）以及在扩展实验中加入的 Something‑Something‑v2、Walking Tours、PE Video Dataset；在评估中使用 ImageNet‑1K、Kinetics‑400 与 Something‑Something‑v2。

**📈 对比分析**

方法通过与 V‑JEPA 2、VideoMAE v2、DINOv2 等基线在相同数据、相同 epoch 或相同 FLOPs 下进行对比，结果显示：① 在 epoch‑匹配下，LeVJEPA 仅用 5.6–20.8 倍的预训练 FLOPs 就能达到或超过 V‑JEPA 2 的 ImageNet、Kinetics‑400 与 Something‑Something‑v2 精度；② 在 FLOPs‑匹配下，LeVJEPA 在 ImageNet 上领先 7.6 点，在 Kinetics‑400 上领先 4.9 点，在 Something‑Something‑v2 上仅落后 3.2 点；③ 与 DINOv2 计算匹配时，LeVJEPA 在 Appearance‑centric 任务上仅差 3.1 点，却在 Motion‑centric 任务上提升近 2 倍。

**⚠️ 局限性**

局限性包括：① 高 token 丢弃率对运动相关任务（如 Something‑Something‑v2）会导致精度下降，需更长训练或改进丢弃策略；② 在更大模型或更大规模数据（互联网级）下的表现尚未验证；③ 目前仅在 frozen‑probe 下评估，缺乏对密集任务（分割、跟踪等）的验证；④ 虽然块因果注意力在预训练时已实现因果性，但实际使用时仍需与后续时序模型配合完成完整的时序推理。

---

## 595. From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench

**arXiv ID:** 2608.27442 | [PDF](https://arxiv.org/pdf/2608.27442v1)

**作者:** Dewu Zheng `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了多轮交互、状态感知的代码评审基准MCR-bench，并构建了2269个涵盖Python、Java、JavaScript、TypeScript、C#的多轮PR数据集；

**💡 创新点**

创新点在于首次引入缺陷生命周期状态追踪与跨轮一致性评估维度，使用LLM自动标注+三轮一致性过滤再人工核验的全流程构建方式，覆盖多语言真实PR场景；

**🔧 技术方法**

采用大语言模型进行缺陷检测与状态预测，使用LLM‑Hit‑Judge作为评价指标，ClearCRC框架评估评论质量，构建了多阶段数据标注与验证管线；

**📊 数据集**

使用2269个多轮PR实例（共5种主流语言），每个实例配有细粒度缺陷卡（位置、描述、类型、严重度）和每轮缺陷状态标签；

**📈 对比分析**

通过对主流LLMs（Claude‑Haiku‑4.5、GPT‑5.2、DeepSeek‑V3.2等）与ACR基线（PR‑Agent、Hybrid‑Review）进行F1、准确率、Hit率等多维度对比，LLM在缺陷检测上的F1普遍在0.4–0.6之间，状态追踪准确率最高可达约80%（Claude‑Haiku‑4.5），性能随轮次递增而下降，存在显著误报与漏报；

**⚠️ 局限性**

主要局限包括LLM跨轮记忆与时间一致性不足导致误报/漏报，ACR基线设计仍以单轮为主不适应多轮场景，数据仅来自公开仓库可能缺乏企业级复杂性，评估指标仍无法完全覆盖人类评审细粒度需求。

---

## 596. Tacet: A Language and Type System for Automatic Statistical Validity Accounting

**arXiv ID:** 2608.27451 | [PDF](https://arxiv.org/pdf/2608.27451v1)

**作者:** Chiké Abuah `[一作]` `[通讯]` (Walla Walla University), Chiké Abuah (Walla Walla University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作设计了嵌入 Python 的语言 Tacet，用来在实验分析中显式声明数据生成、检验假设、预设统计阈值，并通过类型系统与财富变换器在编译时或运行时检查多重比较和样本选择的统计有效性。

**💡 创新点**

创新之处在于把多重比较所需的输入（样本选择方式与观测排列）从仅靠 p 值恢复为程序可恢复的属性，并通过类型系统与预算机制实现预注册与费用控制，使统计有效性检查可在编译阶段完成。

**🔧 技术方法**

技术实现上采用双子语言结构（自由估计子语言 + 计价主张子语言），利用 α‑investing 预算机制与 Fisher、McNemar、簇符号翻转等精确检验；在 Lean4 中机械化证明元理论，并在 Python 运行时实现动态监控与静态检查。

**📊 数据集**

评估使用了 SWE‑bench Verified 领导板（134 个提交，8911 对比）和 BIG‑Bench Hard 以及公开的 HumanEval、MBPP 等基准。

**📈 对比分析**

与传统离线多重校正（Bonferroni、BH）相比，Tacet 通过预注册和按顺序预算实现在线控制，支持可组合的收益与退款，保持 mFDR 控制；实验表明在真实数据上拒绝率与理论一致，性能仅受 Python 解析与运行时检查的轻微开销。

**⚠️ 局限性**

局限性包括需要正确声明 artifact 关键、schema 与聚类级别；折叠操作可能导致非保守；无法处理超出声明的复杂依赖结构；并假设审计记录完整、分析者诚实（honest‑but‑fallible），无法防御恶意操纵。

---

## 597. Mechanistic Reaction Prediction via Discrete Flow Matching on Graph-Structured Electron Occupation

**arXiv ID:** 2608.27429 | [PDF](https://arxiv.org/pdf/2608.27429v1)

**作者:** Nguyen Xuan-Vu `[一作]` (École Polytechnique Fédérale de Lausanne), Philippe Schwaller `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种基于电子占位数的离散流匹配模型 MAELLE，用连续时间马尔科夫链对化学反应进行建模，并通过可解释的电子流轨迹生成正向反应产物与副产物。

**💡 创新点**

创新点在于将反应视为电子占位向量的离散流匹配，利用最优传输构造无标签的编辑序列，并在 CTMC 中学习每一步电子流动，从而实现可解释且保守的反应预测。

**🔧 技术方法**

采用了连续时间马尔科夫链、离散流匹配、最优传输、图神经网络以及电子占位向量表示，并通过采样多条 CTMC 轨迹来生成多产物。

**📊 数据集**

主要使用了 USPTO‑480K 专利反应数据集，并在 FlowER 数据集上评估机制一致性。

**📈 对比分析**

与 SMILES 生成模型（Molecular Transformer、Graph2SMILES）、图编辑模型（MEGAN、GTPN）以及电子重排模型（NERF）等基线进行对比，在 USPTO‑480K 上 Top‑1 87.2% 与 NERF 90.7% 相近，在 OOD 设定下优于多数基线，并能够预测副产物。

**⚠️ 局限性**

局限性包括仅建模电子对，无法处理自由基单电子反应；仅预测正向反应，缺乏逆向或重组预测；生成的轨迹为伪机制，未考虑能量势面，且对复杂催化剂行为的描述有限。

---

## 598. Stochastic Estimation of Transduced Language Models

**arXiv ID:** 2608.27428 | [PDF](https://arxiv.org/pdf/2608.27428v1)

**作者:** Vésteinn Snæbjarnarson `[一作]` (ETH Zurich), Tim Vieira `[通讯]` (ETH Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种无偏的 Beam‑Summing 估计方法，用于从预训练源语言模型与确定性有限状态转换器组成的 Transduced Language Model（TLM）中估计目标前缀概率。

**💡 创新点**

创新点在于用无放回采样（SWOR）和 Horvitz–Thompson 加权替代传统阈值剪枝，实现了无偏估计、可控粒子数（自适应预算）并保证算法几乎必然终止，同时提供了对剪枝误差的量化估计。

**🔧 技术方法**

技术主要包括：基于分数–余数分解的预覆盖枚举、无放回采样与 Horvitz–Thompson 重加权、粒子滤波（SMC）对比、以及利用自适应阈值的粒子数调节。

**📊 数据集**

实验数据集涵盖：WikiText‑2（byte‑level 二元模型与 GPT‑2 大模型）、Penn Treebank（字节到单词的转导）、人类 DNA（DNA→氨基酸转导）以及 MECO 眼动追踪语料库（读时预测）。

**📈 对比分析**

与阈值剪枝 Beam‑Summing 和两种基于多项式重采样的 SMC 基线相比，新的无偏方法在大多数情形下实现了更好的计算‑方差前沿：在 DNA‑to‑amino‑acid 任务中速度提升数个数量级；在 WikiText‑2 与 GPT‑2 任务中，误差更低且方差更小，且能够显式估计剪枝损失；在读时预测实验中，结果保持不变，验证了方法的稳健性。

**⚠️ 局限性**

局限性包括：对高熵源模型仍需较多粒子才能获得低方差；实现上需要完整的预覆盖枚举和有限状态转换器；在极大规模源模型（如更大 GPT‑3/4）下，枚举开销可能仍然不可忽略；此外，算法假设目标前缀长度有限，对无限前缀的估计仍有挑战。

---

## 599. Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling

**arXiv ID:** 2608.27413 | [PDF](https://arxiv.org/pdf/2608.27413v1)

**作者:** Maksim Utushkin `[一作]` (AI VK), Alexander D'yakonov `[通讯]` (AI VK)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套可在包含1.94亿用户、280亿条边的生产级社交图上运行的端到端GNN好友推荐排名系统；

**💡 创新点**

创新点在于将多哈希ID嵌入作为主要节点表示，显著压缩嵌入表至约2 GB，并采用基于二分搜索的时间戳排序CSR邻居采样，避免了大规模邻居扫描的时间开销；

**🔧 技术方法**

技术实现包括GATv2多层消息传递、两头（查询/候选）投影、基于二进制搜索的时间采样、CPU/ GPU分离采样与训练、8块A100 GPU离线推理以及定期离线嵌入刷新；

**📊 数据集**

使用的真实生产数据集为某社交平台的好友图（194 M节点、28 B边）以及对应的1400 M条推荐印象日志；

**📈 对比分析**

在离线评估中相较于top‑pop、MF和前一版WalkGNN，模型ROC‑AUC从0.557提升至0.628；在线上A/B测试中，推荐好友新增量提升16%，独立新增用户提升11.5%，且系统latency无显著回退；

**⚠️ 局限性**

局限性包括冷启动用户缺乏训练信号、非实时推理导致嵌入时效性受限、需要周期性全图重新训练以及对极大节点度的稀疏邻居采样仍有改进空间。

---

## 600. Learning a Continuous Sepsis Severity Score Without Hour-by-Hour Supervision: A Two-Site Retrospective Study

**arXiv ID:** 2608.27421 | [PDF](https://arxiv.org/pdf/2608.27421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 601. Stageboost: Recommending Signals Based on Counterfactual Estimation

**arXiv ID:** 2608.27366 | [PDF](https://arxiv.org/pdf/2608.27366v1)

**作者:** Darpan Singhal `[一作]` (eBay), Yuri Brovman `[通讯]` (eBay)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 StageBoost 两阶段 XGBoost 模型，用于在 eBay View‑Item 页面上根据用户与商品上下文个性化分配短文本/视觉信号。

**💡 创新点**

创新点在于同一模型中通过特征加权与列采样实现基线转化率预测（M1）与信号增益预测（M2）两阶段学习，并采用随机对照试验数据消除选择偏差，解决了 CLE 模型的确定性与稀疏性问题。

**🔧 技术方法**

技术手段包括 XGBoost 统一模型、特征重要性加权、节点级列采样、两阶段（先基线后增益）训练流程、基于 RCT 的数据收集与 counterfactual 离线评估、线上 A/B 对比实验。

**📊 数据集**

数据集为 eBay 生产 VI 页的随机对照试验数据，训练集 2 周、评估集 1 周，涵盖所有资格信号组合。

**📈 对比分析**

比较方法使用基于 counterfactual 的离线估计与线上 A/B 实验，实验显示 Lean StageBoost 在 GMB 上提升 0.08%（统计不显著）并在聚焦类别上提升 0.58%，ASP 提升 0.41%。

**⚠️ 局限性**

主要限制包括极小的转化增益导致信号差异难以显著区分，模型在严格延迟预算下只能使用简化版，且仅在可用信号多于位置信息时才激活，可能导致部分用户未看到任何信号。

---

## 602. SWE-Prime: Fewer Trajectories, Better Performance

**arXiv ID:** 2608.27449 | [PDF](https://arxiv.org/pdf/2608.27449v1)

**作者:** Dewu Zheng `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段多粒度数据筛选方法，用于提升编程代理在软件问题解决任务中的监督质量；

**💡 创新点**

创新点在于将轨迹级（过程质量、结果质量、代表性）与段级（贡献、可学习性、风险）双层评估结合，并通过语义段落和选择性损失实现更高质量的SFT；

**🔧 技术方法**

使用轨迹质量评分、HDBSCAN聚类、边界感知语义段分割、LLM评分与阈值过滤以及仅对选定段落计算交叉熵损失的技术；

**📊 数据集**

采用SWE‑rebench OpenHands轨迹数据集（约32k成功轨迹）进行训练，并在SWE‑Bench Verified与Pro两大基准上评估；

**📈 对比分析**

与完整成功轨迹SFT和随机10%子集比较，所提方法在两大基准上分别提升约12.2%和24.2%的通过率，显著优于对比方法；

**⚠️ 局限性**

局限在于需要手工设定阈值，依赖于已验证的成功轨迹，对未成功或不同领域的数据可能迁移效果不佳。

---

## 603. TTPO: Test-Time Policy Optimization

**arXiv ID:** 2608.27448 | [PDF](https://arxiv.org/pdf/2608.27448v1)

**作者:** Aozhe Wang `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在无标签情况下进行测试时训练（TTT）的方法——TTPO，通过将答案条件教师与自我对抗式强化学习相结合来提升大型语言模型的数学推理能力。

**💡 创新点**

创新点在于将正样本使用基于答案的自我对齐蒸馏（OPSD）与负样本使用分组优势奖励（GRPO）相异构地联合，以对抗伪标签错误的破坏，并引入了令每个分支只关注有价值位置的 token 层级权重与掩码机制。

**🔧 技术方法**

技术包括：多数投票伪标签生成、答案条件教师、前向 KL 蒸馏、分组优势奖励（GRPO）、token 重要性加权、token 误差掩码、LoRA 微调、思考模式与非思考模式切换。

**📊 数据集**

主要使用了五个竞赛级数学推理基准：AIME 2025/2026、HMMT 2025/2026、BRUMO 2025，以及在 OpenThoughts 训练数据上进行对比实验。

**📈 对比分析**

与传统基线（GRPO、TTRL、OPSD、OPSD‑TTT）相比，TTPO 在无标签 TTT 场景下平均提升约 5–8 个百分点（如 Qwen3‑1.7B 从 38.0% 提升至 45.2%），在 OpenThoughts 训练下甚至超过有标签的 OP SD。

**⚠️ 局限性**

局限性包括：对多数投票的依赖导致在极低共识率问题上仍受限；token 层级选择和掩码策略需要调参；在极大模型规模下训练成本高；对非数学推理任务的适用性尚待验证。

---

## 604. Boosting LLM Exploration via Weak-Model Guidance in RLVR

**arXiv ID:** 2608.27420 | [PDF](https://arxiv.org/pdf/2608.27420v1)

**作者:** Xingyu Shen `[一作]` (Peking University), Dongyan Zhao `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在RLVR训练中加入较小模型生成的前缀，让目标模型完成剩余推理，提升生成多样性与pass@k性能

**💡 创新点**

利用跨模型前缀的分布差异作为非参数扰动，解决RLVR中的熵坍塌问题

**🔧 技术方法**

基于GRPO的prefix‑completion RLVR框架、熵驱动的前缀截断以及混合训练比例p

**📊 数据集**

MATH训练集、AIME 2024/25、AMC 2023、MATH 500、Minerva、Olympiad Bench等数学推理基准

**📈 对比分析**

与 vanilla GRPO 在pass@k（k=1~128）上对比，实验表明在所有k值尤其是大k时均有显著提升

**⚠️ 局限性**

对超参数敏感，需精细调优；仅在数学推理任务验证，尚未测试逻辑、代码等其他领域

---

## 605. CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators

**arXiv ID:** 2608.27406 | [PDF](https://arxiv.org/pdf/2608.27406v1)

**作者:** Kechen Liu `[一作]` (Princeton University), Ola Shorinwa `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一个跨身体现的动作条件视频模型框架CLAP，能够在海量人类与机器人视频上训练，学习通用物理先验，并实现零样本在真实机器人任务中的泛化。

**💡 创新点**

创新点包括：①使用端点姿态、自然语言与潜在动作统一跨多体制；②基于课程化的潜在到端点动作学习，既能利用无标注视频扩展规模，又能零样本部署；③通过少量微调将跨体制模型迁移到单体制或新体制。

**🔧 技术方法**

所用技术包括：变分自编码器与视频扩散模型、端点动作空间归一化、自然语言动作映射、潜在动作学习与对齐、跨策略推理、强化学习微调。

**📊 数据集**

使用的数据集包括：Open X-Embodiment (OXE)、EgoDex、DROID、Bridge、Bridge基准等多种机器人与人类视频数据。

**📈 对比分析**

在DROID和Bridge等环境下与Ctrl-World、WorldGym等单体制基线对比，CLAP在DROID上匹配或超越单体制基线；感知指标（SSIM/PSNR/LPIPS/FVD/FID）提升约20%+；在零样本规划和少量微调任务中也显著提高成功率。

**⚠️ 局限性**

局限性：模型仍易产生幻觉，缺乏不确定性量化；训练与推理成本高；主要基于单体制机器人数据，未充分覆盖双手或全身人形等更复杂体制。

---

## 606. D2C-Routing: Dimension-to-Composition Evidence Routing for Mixed-Origin AI-Generated Text Detection

**arXiv ID:** 2608.27380 | [PDF](https://arxiv.org/pdf/2608.27380v1)

**作者:** Xin Chen `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了混合来源AI文本检测框架D2C-Routing，先分别预测内容和表达来源，再通过学习门控组合得到四类混合标签。

**💡 创新点**

通过将文档内部证据分成内容和表达两条路径，并在两源维度上进行监督以及学习门控组合，显著提升低FPR下的四分类准确率。

**🔧 技术方法**

使用RoBERTa编码器、实体链一致性与RST结构、词汇连词、节奏/词性、表面正则等多源特征，结合维度监督和门控组合。

**📊 数据集**

在MixD2C数据集上进行实验，MixD2C为HART基准的重构四分类拆分，包含11,200/1,600/3,200样本。

**📈 对比分析**

与RACE、本地重跑、RoBERTa等基线对比，D2C-base Fusion在平均TPR@1%FPR达到0.8603，明显优于RACE-local的0.8306，且在AA类上有显著提升。

**⚠️ 局限性**

在表达来源识别（AH类）上仍表现不足，外部迁移效果有限，且整体依赖于监督维度，难以在更广泛场景中泛化。

---

## 607. Embodied Scene Rearrangement Planning

**arXiv ID:** 2608.27371 | [PDF](https://arxiv.org/pdf/2608.27371v1)

**作者:** Canzhi Chen `[一作]` (Beijing Institute of Technology), Wei Liang `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于本体感知的三维室内家具重排任务（esrp）并在OmniGibson上构建了大规模基准数据集。

**💡 创新点**

该任务仅使用视角RGB观测与目标俯视图，消除了全局状态访问，加入了对象遮挡和动态场景，从而将长周期规划和空间推理挑战放大到前所未有的难度。

**🔧 技术方法**

作者实现了四类基线：行为克隆（Diffusion Policy）、PPO强化学习、基于预训练VLM的ReAct代理以及利用TAMP的规划方法，展示了多种技术在该任务中的表现。

**📊 数据集**

基准数据集来源于3D-FRONT，包含5,495个室内场景、8,213个可移动家具，构成初始-目标布局对，用于训练与评估。

**📈 对比分析**

在三难度级别和三指标（SR、OSR、RDR）下对四种基线进行比较，规划基线SR最高为30.20%，其余学习与VLM基线约20%或以下，表明任务仍极具挑战。

**⚠️ 局限性**

实验仅依赖RGB观测，未考虑深度或激光等传感器，低层抓取被简化为“魔法取放”，且未在真实机器人上验证，限制了方法的实用性与泛化。

---

## 608. KnockGS:interaction-Grounded Calibrationof Physical Gaussian Representations

**arXiv ID:** 2608.27365 | [PDF](https://arxiv.org/pdf/2608.27365v1)

**作者:** Chenchen Ge `[一作]` (Tuojing Intelligence), Haibao Yu `[通讯]` (Tuojing Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用已知的交互响应和预先生成的响应库，对物理集成的三维高斯资产进行材料尺度校准，并将校准后的参数写回后预测另一未知交互的响应。

**💡 创新点**

提出一种基于响应库的局部岭回归估计器（KnockGS），在只利用一次已知交互的响应时即可恢复物理尺度，并通过冻结写回实现跨交互的预测；同时展示了该方法在不同物体和离散化条件下的稳健性与局限。

**🔧 技术方法**

使用PhysGaussian框架、MPM求解器、三维高斯散点表示、5维响应描述符以及局部硬邻域岭回归；对响应库进行离线预计算并在估计时仅做离线匹配与闭式回归。

**📊 数据集**

基于合成的枕头、Ficus（藤蔓）和Vasedeck（花瓶）三种高斯资产，构造54个尺度候选的响应库，并在5个保留目标上进行验证；所有实验均在完全虚拟的物理模拟环境中完成。

**📈 对比分析**

与响应最近邻、逆距离KNN、全局岭回归、固定默认材质以及CMA‑ES视频优化、PhysGM和ReconPhys等基线相比，局部岭回归在联合尺度误差上达1.13%（显著低于2.4%），在Probe‑B轨迹RMSE上提升约3倍，且在视觉指标PSNR/SSIM上保持最高水平。

**⚠️ 局限性**

方法依赖粒子级的特权观测，对离散化（填充密度、MPM网格）高度敏感；仅能校准两个MPM尺度，无法跨物体共享；在无粒子ID的视觉轨迹或受噪深度观测下性能显著下降，且对物体外推和真实测量缺乏验证。

---

