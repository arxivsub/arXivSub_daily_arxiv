# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-29 | 今日论文总数: 459

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. How AI Impacts Skill Formation

**arXiv ID:** 2601.20245 | [PDF](https://arxiv.org/pdf/2601.20245v1)

**作者:** Judy Hanwen Shen `[一作]` (Anthropic), Alex Tamkin `[通讯]` (Anthropic)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比使用和不使用AI协助完成新Python异步库（Trio）的学习任务，评估对技能形成的影响。

**💡 创新点**

揭示AI协助可能降低学习者对新技术的概念理解、代码阅读与调试能力，并细化六种AI交互模式，指出高认知投入模式可保持学习效果。

**🔧 技术方法**

采用GPT‑4o聊天式代码助手、随机对照实验与问卷调查、编码记录分析。

**📊 数据集**

实验样本为52名具有1年以上Python经验的自由职业/专业程序员，通过两道Trio编程题及27分的测评问卷。

**📈 对比分析**

与对照组对比，AI组在完成时间无显著提升，测评成绩平均下降约17%（Cohen d≈0.74）。

**⚠️ 局限性**

仅包含单一任务、短时实验、样本非真实工作场景、未测量提示技巧、评估方式有限。

---

## 2. Computational aspects of disks enclosing many points

**arXiv ID:** 2601.20036 | [PDF](https://arxiv.org/pdf/2601.20036v1)

**作者:** Prosenjit Bose `[一作]` (Carleton University), Tyler Tuttle `[通讯]` (Carleton University)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5046519613)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了几种算法来寻找一对点，使得包含这对点的任意圆盘必须包含至少 cn 个点，其中 c 是一个常数。

**💡 创新点**

创新点在于提出了多种算法，包括随机算法和针对特定点集（如凸位置和双色点集）的算法，且在不同情况下优化了常数 c 的值。

**🔧 技术方法**

使用了随机算法、二次时间算法和线性时间算法，结合了高阶 Voronoi 图的概念。

**📊 数据集**

使用了平面上的点集 S，包含 n 个点，并考虑了在简单多边形内的点集。

**📈 对比分析**

与现有方法比较，提出的算法在时间复杂度上有所优化，特别是在处理凸位置和双色点集时，性能表现良好，能够在 O(n log n) 或 O(n) 时间内找到所需的点对。

**⚠️ 局限性**

限制在于算法的复杂性和对特定点集的依赖，尤其是在处理简单多边形时，可能会遇到额外的挑战。

---

## 3. Stingy Context: 18:1 Hierarchical Code Compression for LLM Auto-Coding

**arXiv ID:** 2601.19929 | [PDF](https://arxiv.org/pdf/2601.19929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 4. How Much Progress Has There Been in NVIDIA Datacenter GPUs?

**arXiv ID:** 2601.20115 | [PDF](https://arxiv.org/pdf/2601.20115v1)

**作者:** Emanuele Del Sozzo `[一作]` (Massachusetts Institute of Technology), Neil Thompson `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1725 | [OpenAlex ID](https://openalex.org/A5035360766)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过收集 2006 年至 2025 年 NVIDIA 数据中心 GPU 的完整规格与价格，量化 FP16/FP32/FP64 性能、内存容量/带宽、TDP、价格等指标的增长率，并评估美国出口管制对潜在性能差距的影响。

**💡 创新点**

创新点在于构建 101 颗 GPU 的完整数据集，采用指数增长模型估算 CAGR/DT，系统对比 per‑bandwidth、per‑dollar、per‑watt 指标，并将出口管制与理论 TPP 差距关联，揭示 FP16/FP32 在 1.4–1.7 年内翻倍而 FP64 发展缓慢的趋势。

**🔧 技术方法**

主要技术手段为对数线性指数回归（OLS）对峰值 TFLOPS、HBM/GDDR 容量/带宽、TDP、价格等时间序列进行拟合，并利用置信区间进行增长速率比较。

**📊 数据集**

数据集来源包括 NVIDIA 官方文档、TechPowerUp、VideoCardz、Epoch AI 等，涵盖 Tesla 版到 Blackwell Ultra 版的 101 颗数据中心 GPU 的规格与上市价格。

**📈 对比分析**

通过 CAGR 与 DT 的置信区间对比最高性能每年与全部 GPU 的增长速率，发现 FP16/FP32 每 1.4–1.7 年翻倍，FP64 发展缓慢，内存容量/带宽约 3.3–3.5 年翻倍，价格约 5 年翻倍，功耗约 16 年翻倍；出口管制可将性能差距从 23.6 倍降至 3.54 倍。

**⚠️ 局限性**

主要局限在于价格数据缺失导致约 1/3 GPU 被剔除；仅使用理论峰值而非实际基准；未考虑低位宽、稀疏加速、实际功耗及 HBM 规范的细节，导致对国产芯片和实际性能的估计存在偏差。

---

## 5. Dynamic framework for edge-connectivity maintenance of simple graphs

**arXiv ID:** 2601.20137 | [PDF](https://arxiv.org/pdf/2601.20137v1)

**作者:** Blazej Wrobel `[一作]` `[通讯]` (Wroclaw University of Science and Technology), Blazej Wrobel (Wroclaw University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种动态框架，能在边插入时自动剔除冗余边，在边删除后快速增补最小边集，保持无权简单图的k‑边连通性。

**💡 创新点**

创新点在于：①将Nagamochi‑Ibaraki稀疏证书与Link‑Cut树结合实现O(k log n)的冗余剔除；②利用残差图与Dinic最大流在O(k·n^{5/3})时间内构造局部增补边，保证λ(G)≥k。

**🔧 技术方法**

使用的技术包括：Nagamochi‑Ibaraki稀疏化算法、Link‑Cut树数据结构、Dinic最大流（单元容量）、最大流残差图分析。

**📊 数据集**

文中未给出具体实验数据集，主要为理论分析与算法复杂度证明。

**📈 对比分析**

比较方法主要是与传统的最小割维护算法对比，理论上通过稀疏化将边数压缩到O(kn)，从而实现更低的时间复杂度；实验结果未提供，故无法给出具体性能数值。

**⚠️ 局限性**

局限性包括：仅适用于固定k且无权、简单图；对加权图或有向图的扩展尚未给出；在极端稠密图中恢复步骤仍需O(k·n^{5/3})的开销。

---

## 6. LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning

**arXiv ID:** 2601.19952 | [PDF](https://arxiv.org/pdf/2601.19952v1)

**作者:** Wenhao Zou `[一作]` (Meituan), Jingwen Xu `[通讯]` (Meituan)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种实时语音代理框架 LTS-VoiceAgent，支持在用户讲话时并行进行思考与说话，显著降低响应延迟。

**💡 创新点**

核心创新包括：动态语义触发器精准判断何时开始推理，双角色流式编排器将后台思考与前台预测生成解耦，实现“边听边思考”；以及 Pause‑and‑Repair 评测基准，专门测试语音流中的停顿与自我纠正。

**🔧 技术方法**

技术方案主要基于轻量化 DistilBERT 触发器、Qwen3-8B（可扩展到 32B）LLM、vLLM 异步双流推理栈、ASR 与 TTS 的端到端拼接，以及自监督语义标注与动态状态表维护。

**📊 数据集**

使用公开评测集 VERA、Spoken‑MQA、BigBenchAudio 以及自建的 Pause‑and‑Repair 基准（来源 GSM8K 与 MMLU‑Pro 经过场景化转换）。

**📈 对比分析**

与串行思考、PredGen、LTS‑VAD 等现有流式策略以及 Qwen2.5‑Omni 的端到端模型进行对比，实验表明 LTS‑VoiceAgent 在准确率、首词延迟（TTFS）和前向推理次数（NFE）上均优于其他实时方法，逼近串行思考的准确率但保持毫秒级延迟。

**⚠️ 局限性**

局限性包括：仅在少数模型与单语种、干净语音条件下验证；缺乏多轮对话、噪声、口音、多域词汇的测试；未做用户体验或安全隐私的实证评估；以及仍可能产生 LLM 典型的幻觉与偏见。

---

## 7. PASS: Ambiguity Guided Subsets for Scalable Classical and Quantum Constrained Clustering

**arXiv ID:** 2601.20157 | [PDF](https://arxiv.org/pdf/2601.20157v1)

**作者:** Pedro Chumpitaz-Flores `[一作]` (University of South Florida), Kaixun Hua `[通讯]` (University of South Florida)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5058981040)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 PASS 框架，将 must‑link 约束合并为伪点，并通过 margin 与 Fisher‑Rao 信息几何的子集选择实现可扩展的 pairwise‑constrained k‑means。

**💡 创新点**

创新点在于同时使用 must‑link 合并、基于阈值 margin 的不确定性筛选以及 Fisher‑Rao 信息几何评分的子集选择，显著降低可解空间并高效满足约束。

**🔧 技术方法**

使用的技术包括伪点聚合、margin 余弦筛选、Fisher‑Rao 信息几何评分、0–1 ILP 约束优化，以及将子集映射为 QUBO 的量子近似优化（q‑PCKMeans）。

**📊 数据集**

实验数据集涵盖 UCI 数据集：Iris、HTRU2、Seeds、AC、PR2392、Gas_methane、Gas_CO、Skin 等，样本规模从数百到十万不等。

**📈 对比分析**

与 COP‑k‑means、BLPKM‑CC、PCCC、CP‑QAOA 等基线对比，PASS 在 SSE 与最佳方法相当，运行时间显著更短；在量子混合版本 q‑PCKMeans 中实现更低 SSE，证明可扩展性与性能兼具。

**⚠️ 局限性**

局限性包括：目前量子硬件 qubit 数量和噪声限制导致只能在小规模实例上实现；子集选择参数对极端稀疏约束场景的鲁棒性仍需改进。

---

## 8. Usage, Effects and Requirements for AI Coding Assistants in the Enterprise: An Empirical Study

**arXiv ID:** 2601.20112 | [PDF](https://arxiv.org/pdf/2601.20112v1)

**作者:** Maja Vukovic `[一作]` (IBM Research), Michele Merler `[通讯]` (IBM Research)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5068061267)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

调查了57位企业内部不同职能的开发者对AI编程助手的使用体验，并综合分析了35份相关问卷调查，探讨其对生产力、工作流程及未来需求的影响。

**💡 创新点**

首次系统结合企业实证研究与现有文献综述，揭示不同业务领域对AI助手的多样化动机、期望功能及对流程的深远影响，并提出针对定制化、全仓储理解和自主化等方向的未来需求。

**🔧 技术方法**

使用问卷调查、统计分析与文本挖掘技术，对用户回答进行编码、聚类与可视化，评估生产力提升、代码保留率等关键指标。

**📊 数据集**

核心数据集包括57份企业用户问卷（涵盖业务、经验、语言、工具等信息）以及35篇公开问卷研究的结果。

**📈 对比分析**

通过聚合分析与分组对比，展示不同业务单元在动机、功能价值与生产力提升（平均≥25%）方面的差异；与文献中报告的生产率提升（12–25%）相似，表明AI助手能显著提升开发效率。

**⚠️ 局限性**

局限性包括样本规模相对有限、主要关注ChatGPT、Copilot等工具、缺乏对agentic AI工作流的评估、研究时间短且多为自我报告，未深入检验长期质量与安全影响。

---

## 9. Taxonomy of the Retrieval System Framework: Pitfalls and Paradigms

**arXiv ID:** 2601.20131 | [PDF](https://arxiv.org/pdf/2601.20131v1)

**作者:** Deep Shah `[一作]` (National Institute of Standards and Technology), Nehal Kathrotia `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

未提供具体研究工作

**💡 创新点**

无创新点说明

**🔧 技术方法**

未使用具体技术

**📊 数据集**

未使用数据集

**📈 对比分析**

未说明比较方法与性能

**⚠️ 局限性**

未说明局限性

---

## 10. Going NUTS with ADVI: Exploring various Bayesian Inference techniques with Facebook Prophet

**arXiv ID:** 2601.20120 | [PDF](https://arxiv.org/pdf/2601.20120v1)

**作者:** Jovan Krajevski `[一作]` (Ss. Cyril and Methodius University), Biljana Tojtovska Ribarski `[通讯]` (Ss. Cyril and Methodius University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5027551017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究实现了Facebook Prophet的PyMC重写，并比较了MAP、MCMC（MH、DMZ、NUTS）与VI（ADVI、FullRank ADVI）在时间序列预测上的表现，

**💡 创新点**

创新点在于提供了更灵活的直观API、全面评估多种推断算法并揭示其收敛与预测质量差异、并指出VI在速度与不确定性估计上的优势与不足，

**🔧 技术方法**

主要技术包括PyMC框架、L-BFGS-B优化、HMC/NUTS采样、Metropolis‑Hastings、DE‑MCMC、ADVI及FullRank ADVI变分推断，

**📊 数据集**

使用的数据集是维基百科Peyton Manning页面的每日页面浏览量的两年训练数据与一年预测数据，

**📈 对比分析**

比较方法通过R‑hat、ESS、自动相关、ELBO收敛性、MAE/MSE/RMSE/MAAPE等预测指标与耗时来评估，结果显示NUTS收敛最快且预测最优，ADVI与FullRank ADVI速度快但不确定性估计偏差，MAP最快但忽略不确定性，

**⚠️ 局限性**

局限性包括MH/DMZ收敛差、VI对后验不确定性估计欠准、缺乏GPU加速导致大规模MCMC效率低、模型仍未对季节性与节假日等复杂组件做完整扩展

---

## 11. How do Agents Refactor: An Empirical Study

**arXiv ID:** 2601.20160 | [PDF](https://arxiv.org/pdf/2601.20160v1)

**作者:** Lukas Ottenhof `[一作]` (University of Alberta), Thibaud Lutellier `[通讯]` (University of Alberta)

**通讯引用:** 1152 | [OpenAlex ID](https://openalex.org/A5063985157)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统分析了五种主流代码代理在 Java 开发中的重构 Pull Request，比较其重构类型、频率及对代码 smell 的影响。

**💡 创新点**

首次对 AI 代理在 Java 重构实践中的行为进行大规模量化评估，并探讨其对代码质量的实际效果。

**🔧 技术方法**

采用 RefactoringMiner 检测重构操作，使用 DesigniteJava 3.0 评估代码 smell，并用 Wilcoxon 检验与 Cliff’s Delta 进行统计比较。

**📊 数据集**

使用 AIDev 数据库中 932,791 条代理 PR 过滤出的 1,278 条 Java PR 以及 86 个无代理干预的 Java 项目作为对照。

**📈 对比分析**

通过比较重构类型分布、代码 smell 变化（Δ）和统计显著性，结果显示大多数代理的代码 smell 变化与人类无显著差异，唯 Cursor 的重构导致代码 smell 明显增加。

**⚠️ 局限性**

研究受限于代理在少量项目中的样本量、工具检测的误差以及无法完全排除非重构改动对 smell 影响的可能性。

---

## 12. LogSieve: Task-Aware CI Log Reduction for Sustainable LLM-Based Analysis

**arXiv ID:** 2601.20148 | [PDF](https://arxiv.org/pdf/2601.20148v1)

**作者:** Marcus Emmanuel Barnes `[一作]` (University of Toronto), Safwat Hassan `[通讯]` (University of Toronto)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5022060601)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了 LogSieve，一种基于根因分析的语义保留日志压缩方法，旨在在 CI 日志中删除低信息行并减少 LLM 推理成本。

**💡 创新点**

在传统结构化压缩基础上引入任务感知的语义过滤，使得在保持诊断信息的同时显著减少日志体量。

**🔧 技术方法**

采用 TF‑IDF、BERT、LLaMA3 嵌入的机器学习分类器实现日志行相关性预测，并用 GPT‑4o 进行语义相似性和分类准确性评估。

**📊 数据集**

对 20 个使用 GitHub Actions 的开源 Android 项目的失败 CI 运行日志进行人工标注，共计约 14,600 行。

**📈 对比分析**

与 LogZip 压缩和随机删除相比，LogSieve 在保持语义相似度（Cosine 0.93、GPTScore 0.93）和分类准确率（80%）的前提下，平均实现 42% 行 / 40% token 的压缩。

**⚠️ 局限性**

研究仅覆盖 Android 项目与 GitHub Actions，未验证对其他 CI 平台或日志类型的适用性；且评估仅基于 GPT‑4o，缺乏多模型或人工评估的验证。

---

## 13. Cascaded Vulnerability Attacks in Software Supply Chains

**arXiv ID:** 2601.20158 | [PDF](https://arxiv.org/pdf/2601.20158v1)

**作者:** Laura Baird `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5090346723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过将软件供应链中的 SBOM、依赖关系、已知 CVE 与 CWE 构造异构图，利用 HGAT 对组件是否存在已知漏洞进行预测，并把多链攻击视为 CVE 对的链接预测问题，使用 MLP 进行链式漏洞发现，从而实现基于 SBOM 的多步漏洞链分析。

**💡 创新点**

① 把 SBOM 的依赖结构与漏洞/弱点信息整合为异构图，突破传统孤立漏洞扫描的局限；② 将多链攻击抽象为链接预测任务，克服缺乏多漏洞链标签的难题；③ 在异构图注意力网络与轻量级 MLP 的组合下，既保留图结构信息，又能在稀缺监督下进行链预测。

**🔧 技术方法**

生成 SBOM（CycloneDX + Syft）、漏洞注入（Grype + OSV）、异构图注意力网络（HGAT）与多头图注意力、MLP 链链接预测、负采样、ROC‑AUC 等评估指标。

**📊 数据集**

Wild SBOMs 数据集（200 个 Python SBOM），公开的 35 条多漏洞链（来自公开披露和事件报告），CVE/CWE 数据以及 OSV 数据库。

**📈 对比分析**

与仅使用组件特征或去掉依赖边的模型对比，HGAT+依赖结构在节点分类中取得 91.03% 的准确率、74.02% 的 F1；链预测任务的 ROC‑AUC 达到 0.93，证明异构图结构和链链接预测在提升检测效果方面具有显著优势。

**⚠️ 局限性**

链标签稀缺导致训练样本有限；链级/时间分割的评估不足，模型可能在同一 CVE 的不同配对上出现偏倚；未深入探讨不同 SBOM 生成器和工具差异对结果的影响；未来工作需扩展数据集、引入 LLM/知识图谱等技术。

---

## 14. Locatability and Locatability Robustness of Visual Variables in Single Target Localization

**arXiv ID:** 2601.20080 | [PDF](https://arxiv.org/pdf/2601.20080v1)

**作者:** Wei Wei `[一作]` (University of Victoria), Charles Perin `[通讯]` (University of Victoria)

**通讯引用:** 2026 | [OpenAlex ID](https://openalex.org/A5001649622)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

通过两项大规模实验研究了七种视觉变量在定位任务中的可定位性与鲁棒性，评估了集大小、布局、目标位置等因素的影响。

**💡 创新点**

首次给出针对定位任务的视觉变量可定位性排名与对集大小、位置、布局的鲁棒性分组，并提出了新的“locatability”和“locatability robustness”概念。

**🔧 技术方法**

采用基于 Bayesian 层级回归的线性模型（Student‑t 分布噪声），使用 Web 实验平台（Psychopy+Pavlovia）与自制刺激，并通过对数变换的 RT 数据进行分析。

**📊 数据集**

数据来自 112 名在 Prolific 招募的受试者，每人完成 224–280 次试验，包含 7 种视觉变量、4 种集大小、2 种布局、2 种目标位置。

**📈 对比分析**

通过比较各变量与各因素的对数 RT 斜率、概率区间来评估鲁棒性；结果显示颜色变量最快、形状最慢；非网格布局和更大集大小均显著延长定位时间；鲁棒性分为数个组别，表明不同变量对集大小与位置的敏感度不同。

**⚠️ 局限性**

局限性包括仅评估 7 个静态变量、未考虑重叠、运动、纹理等视觉属性；集大小上限为 768；实验为在线采样，可能缺乏视距控制；仅覆盖平面静态图形，未验证非均匀分布的情形。

---

## 15. A Cache-Aware Hybrid Sieve Combining Segmentation and Bit-Packing for Fast Prime Generation

**arXiv ID:** 2601.19909 | [PDF](https://arxiv.org/pdf/2601.19909v1)

**作者:** Kathi Lakshmi Mani Thirdhana `[一作]` `[通讯]` (National Institute of Technology Calicut), Kathi Lakshmi Mani Thirdhana (National Institute of Technology Calicut)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一种缓存感知的混合筛法，用于高效生成大范围内的素数。

**💡 创新点**

创新点在于将分段筛、位打包和缓存行对齐三种技术统一到同一算法中，显著降低缓存未命中率并减少内存占用。

**🔧 技术方法**

采用位打包仅存储奇数、按 L1 缓存大小分段、块对齐至 64 字节缓存线、预先计算 √N 以内的素数并复用、使用位掩码标记复合数，并在单线程实现中保持数据局部性。

**📊 数据集**

实验数据集为不同规模的整数范围 N = 10⁷、10⁸、10⁹，使用 12 核 AMD Ryzen CPU 进行基准测试。

**📈 对比分析**

通过与传统 Sieve of Eratosthenes 及分段筛进行跑时对比，实验表明混合筛在 10⁹ 规模下的运行时间约为 21.5 s，分别比传统筛快 2.4×、比分段筛快 1.7×；同时内存使用减少约 8 倍。

**⚠️ 局限性**

局限性包括：实现仅为单线程；未充分利用 SIMD 或多核并行；对极小规模 N 的收益有限；算法复杂度仍为 O(N log log N)，主要提升在硬件层面而非理论复杂度。

---

## 16. Shallow-π: Knowledge Distillation for Flow-based VLAs

**arXiv ID:** 2601.20262 | [PDF](https://arxiv.org/pdf/2601.20262v1)

**作者:** Boseong Jeon `[一作]` (Samsung Research), Taehan Kim `[通讯]` (Samsung Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过知识蒸馏将流式视觉‑语言‑动作模型的视觉‑语言骨干和动作头的 transformer 深度从 18 层压缩到 6 层，并在 Jetson Orin/Thor 等边缘设备上实现了超过 2 倍的推理速度提升，成功率仅下降不到 1%。

**💡 创新点**

创新点在于：① 采用联合蒸馏框架同时压缩视觉‑语言骨干和动作头；② 设计了针对流式 VLA 的三重损失（任务、教师匹配、行动‑跨模态注意力对齐）；③ 将注意力蒸馏限定在动作查询与视觉‑语言键值之间，避免对非生成模态的干扰；④ 在真实机器人场景中验证压缩模型的可部署性。

**🔧 技术方法**

技术手段包括：流式 VLA（π‑style）模型、Diffusion Transformer (DiT)、流匹配损失、知识蒸馏（teacher–student）、中间层注意力对齐、以及在 Jetson 平台上的 CUDA 推理优化。

**📊 数据集**

实验数据集主要为 LIBERO（仿真）以及在 ALOHA 与 RB‑Y1 两个真实机器人平台上收集的多任务数据（Peg‑in‑hole、Insert foam、Scoop apple、Pour beans、Recycle 等）。

**📈 对比分析**

与教师模型、SmolVLA（小骨干）、以及对标的 token‑压缩方法相比，压缩后的 Shallow‑π 在所有任务中保持了与教师相当的成功率（<1% 下降），FLOPs 与 CUDA 推理时间均下降 2‑3 倍，尤其在 Jetson Orin 上实现了约 10 Hz 的端到端速度。

**⚠️ 局限性**

局限性包括：蒸馏过程中需要同时加载教师和学生模型，导致训练时 GPU 内存和计算成本升高；未与视觉 token 或扩散步骤等其他效率轴结合；以及对极端超大规模任务的可扩展性尚未验证。

---

## 17. NeuraLSP: An Efficient and Rigorous Neural Left Singular Subspace Preconditioner for Conjugate Gradient Methods

**arXiv ID:** 2601.20174 | [PDF](https://arxiv.org/pdf/2601.20174v1)

**作者:** Alexander Benanti `[一作]` (Stony Brook University), Hong Qin `[通讯]` (Stony Brook University)

**通讯引用:** 9621 | [OpenAlex ID](https://openalex.org/A5091408797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 NeuraLSP，一种基于神经网络的低秩预条件器，用于求解稀疏线性系统；

**💡 创新点**

创新点在于设计了 NLSS 损失函数，能够全局优化求取近零空间向量的左奇异子空间，并且对秩具备自适应性；

**🔧 技术方法**

采用深度 MLP（四层、128/256/256/128 节点）结合 QR 正交化来学习左奇异子空间，并使用 Galerkin 投影构造两级多重网格；

**📊 数据集**

在三类 PDE（扩散、各向异性、屏蔽泊松）上使用基于 FEM 的随机系数数据集（共 1000 训练样本、100 测试样本），网格大小为 64×64，节点随机扰动后三角网格；

**📈 对比分析**

与传统子空间损失、GNN、SA‑AMG 以及完整 SVD 进行对比，NeuraLSP 在多种 PDE 和不同粗粒度下实现 30%–50% 的总运行时间缩短，并保持或优于迭代收敛速度；

**⚠️ 局限性**

局限在于模型对网络架构敏感，当前仅在 MLP 上验证，GNN 可能缺乏同等效率，且未证明对更大规模或不同离散化方式的泛化能力。

---

## 18. On the Effectiveness of LLM-Specific Fine-Tuning for Detecting AI-Generated Text

**arXiv ID:** 2601.20006 | [PDF](https://arxiv.org/pdf/2601.20006v1)

**作者:** Michał Gromadzki `[一作]` (Faculty of Mathematics and Information Science Warsaw University of Technology), Agnieszka Kaliska `[通讯]` (Faculty of Modern Languages and Literatures Adam Mickiewicz University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了基于大规模语料的 AI 生成文本检测，构建了 1 B 人类文本和 1.9 B AI 文本的语料库，训练并评估多种检测模型，提出了 Per LLM 与 Per LLM family 两种微调范式。

**💡 创新点**

创新点在于：①大规模、跨模型的人工与 AI 文本对比语料库；②针对每个 LLM 或 LLM 家族的专门微调策略；③使用 token‑level 预测实现细粒度检测；④在 100 M 令牌基准上实现 99.6% 的 token‑level 准确率，显著优于现有开源基线。

**🔧 技术方法**

主要技术包括 Transformer 生成模型（如 Llama、Phi、Qwen 等）作为后端，冻结主干仅训练分类头；使用 Adam 优化器、二元交叉熵；对 token 进行左侧 padding 以保证全局上下文；在实验中采用了 token‑level 与 sample‑level 两种评估。

**📊 数据集**

数据集：①人类文本 1 B 令牌，涵盖 10 种多样化体裁；②AI 文本 1.9 B 令牌，由 21 种 LLM 在 4 种采样配置下生成；③构建了 master‑large、detect‑gpt‑4.1‑nano、Per‑LLM 与 Per‑LLM‑family 等多种训练/验证/测试子集。

**📈 对比分析**

与基准模型（如 430 M 参数的 open‑source detector）比较，微调后单个 LLM 在 token‑level 上达到 0.995+ 的准确率；Per‑LLM ensemble 召回率高达 0.98 但精度仅 0.66；Per‑LLM‑family ensemble 在召回率和精度均优于 Per‑LLM，且仅需 6 个检测器。整体来看，微调 LLM 的性能远超开源 SOTA，且支持十倍以上上下文长度。

**⚠️ 局限性**

局限性包括：①训练文本仅来自对人类样本的 prompt，缺乏多样化的生成策略；②采样参数统一，未覆盖高温/极端采样导致的多样性；③使用的模型规模相对较小，未验证在更大模型上的可扩展性；④Per‑LLM/Per‑LLM‑family 方案难以覆盖未见模型或混合人机文本；⑤缺乏对抗性攻击（如改写、翻译）的评估。

---

## 19. Hardware-Aware Model Design and Training of Silicon-based Analog Neural Networks

**arXiv ID:** 2601.19905 | [PDF](https://arxiv.org/pdf/2601.19905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 20. CiMRAG: Cim-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs

**arXiv ID:** 2601.20041 | [PDF](https://arxiv.org/pdf/2601.20041v1)

**作者:** Shih-Hsuan Chiu `[一作]` (National Taiwan University), Ming-Syan Chen `[通讯]` (National Taiwan University)

**通讯引用:** 16743 | [OpenAlex ID](https://openalex.org/A5036009069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向边缘设备的噪声鲁棒任务嵌入学习框架 TONEL，用于在计算内存（CiM）环境下提升检索增强生成（RAG）的个性化效果；

**💡 创新点**

创新点在于结合噪声感知任务优化（NATO）和无监督伪标签生成（PGM），实现无标签的领域自适应与对硬件噪声的鲁棒性；

**🔧 技术方法**

采用 CiM 硬件下的 MIPS、8 位量化、投影模型、噪声注入、交叉熵改进、K-means 伪标签等技术；

**📊 数据集**

使用 LaMP 公开的 Movie Tagging（15 类）和 Product Rating（5 类）两个个性化数据集进行实验；

**📈 对比分析**

与 PCA、RoCR 等基线以及 Oracle/Random 进行比较，TONEL 在 100% 噪声条件下的 top‑1 检索精度提升至约 70%（RoCR 仅 30%），在 downstream LLM 分类任务中相对提升 20–40%；

**⚠️ 局限性**

局限在于对伪标签聚类的依赖、对不同 CiM 设备噪声的适配仍需额外调参、以及在极度动态更新场景下的实时聚类效率尚未完全解决。

---

## 21. SDUs DAISY: A Benchmark for Danish Culture

**arXiv ID:** 2601.19930 | [PDF](https://arxiv.org/pdf/2601.19930v1)

**作者:** Jacob Nielsen `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于丹麦文化典藏（Kulturkanon）构建了Daisy文化知识评测基准，并生成了741个闭合式问答对，供大型语言模型（LLM）评估。

**💡 创新点**

首次提供以官方权威文化典藏为基础的丹麦文化评测框架，填补了低资源语言模型在文化知识评估上的空白，并引入了多领域、多时间跨度的评测内容。

**🔧 技术方法**

使用Gemma3 27B（4‑bit量化）在维基百科页面上自动生成随机问题，随后由人工审核修订；评估方法采用BLEU和词级F1分数。

**📊 数据集**

数据集由丹麦文化典藏108件作品的维基百科页面内容抽取并转化为问答对，共计741条；未包含儿童典藏，主要覆盖建筑、艺术、设计、电影、文学、音乐等八个文化领域。

**📈 对比分析**

对五个主流模型（OpenAI GPT‑OSS 20B/120B、Google Gemma 3‑27B、Meta Llama 3.3‑70B、Mistral‑Small 24B）进行单一提示评估，BLEU最高为0.166（Llama 70B），F1最高为0.268；总体来看，模型对丹麦文化的把握仍相当有限，性能相对较差。

**⚠️ 局限性**

局限性：①评测仅覆盖Kulturkanon范围，无法涵盖全部丹麦文化；②单一提示模板可能不适用于所有模型；③数据量相对较小，难以覆盖多样化知识；④评估方法对训练数据中的偏差与安全调优可能产生影响，导致结果偏低。

---

## 22. PiC-BNN: A 128-kbit 65 nm Processing-in-CAM-Based End-to-End Binary Neural Network Accelerator

**arXiv ID:** 2601.19920 | [PDF](https://arxiv.org/pdf/2601.19920v1)

**作者:** Yuval Harary `[一作]`, Leonid Yavits `[通讯]` (Bar Ilan University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并制造了 PiC-BNN，一种完全二进制、基于 CAM 的神经网络加速器，实现了端到端二进制网络，无需任何全精度运算。

**💡 创新点**

创新点在于通过可配置的 Hamming 距离容忍度实现多次近似匹配，利用大数定律在多次输出层执行后对结果取多数投票，弥补缺失的全精度层，彻底消除对全精度硬件的依赖。

**🔧 技术方法**

使用了 NOR 型十晶体管 CAM 结构、可配置电压控制的 ML 充放电（V_ref、V_eval、V_st）、近似搜索、逻辑多数投票以及批归一化常数直接叠加等技术。

**📊 数据集**

评估数据集为 MNIST（10 类，28×28）和 Hand Gesture（20 类，64×64）。

**📈 对比分析**

在测试集上采用多次执行输出层并多数投票的方式评估准确率，MNIST 达到 95.2%（与软件基准相当），Hand Gesture 达到 93.5%；吞吐量 560K 预测/秒，能效 703 M 预测/秒/W，功耗 0.8 W，显著优于传统 BNN 加速器。

**⚠️ 局限性**

局限性包括：需要多次执行输出层才能恢复精度，导致计算延迟；调节电压参数耗时且对温漂、工艺变异敏感；目前仅在简单 MLP 结构上验证，复杂网络或更大数据集的适用性仍待探索。

---

## 23. Cross-Session Decoding of Neural Spiking Data via Task-Conditioned Latent Alignment

**arXiv ID:** 2601.19963 | [PDF](https://arxiv.org/pdf/2601.19963v1)

**作者:** Canyang Zhao `[一作]` (Institute of Automation, Chinese Academy of Sciences), Bing Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

未提供具体研究内容

**💡 创新点**

未提供创新点

**🔧 技术方法**

未提供技术

**📊 数据集**

未提供数据集

**📈 对比分析**

未提供比较方法与性能

**⚠️ 局限性**

未提供限制

---

## 24. Bench4HLS: End-to-End Evaluation of LLMs in High-Level Synthesis Code Generation

**arXiv ID:** 2601.19941 | [PDF](https://arxiv.org/pdf/2601.19941v1)

**作者:** M Zafir Sadik Khan `[一作]` (University of Central Florida), Hadi Kamali `[通讯]` (University of Central Florida)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5020619624)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Bench4HLS 框架，提供170条手工验证的 HLS benchmark、自动化编译、仿真、综合与 PPA 评估，并支持基于 YAML 的 DSE；

**💡 创新点**

创新点在于大规模、结构化 benchmark 集合、可插拔多工具链 PPA API、端到端 Pass@K 评价与 YAML 驱动的自动化 DSE、以及公开可复现的完整评测流程；

**🔧 技术方法**

使用 Vitis HLS、Vivado、Python/TCL 自动脚本、LLM 接口（OpenAI GPT‑5、Qwen 2.5、Llama 3.3）、Pass@K 统计与 YAML 配置实现评测与 DSE；

**📊 数据集**

基于170条设计任务，来源于公开仓库（Verilog, Vitis‑HLS‑Introductory‑Examples, CHStone, HLS4ML, Rosetta 等），每条包含自然语言说明、HLS C++ 代码和对应的测试平台；

**📈 对比分析**

通过 Pass@1/5/10 的编译/仿真/综合成功率比较模型：GPT‑5 最高约 97% 编译成功、70% 仿真/综合成功；Llama 3.3 70B 其次；Qwen 2.5 14B/32B 依次下降；DSE 可在约 40% 设计中提升至少一项 PPA >20%，相比小模型仅约 10%；

**⚠️ 局限性**

局限性包括对大规模复杂 benchmark 的成功率仍有限（尤其小模型）；生成设计的 PPA 仍落后于人工优化的参考实现；DSE 搜索空间受限于手工制定的 YAML ；工具链与目标 FPGA 平台的兼容性有限。

---

## 25. Size Matters: Reconstructing Real-Scale 3D Models from Monocular Images for Food Portion Estimation

**arXiv ID:** 2601.20051 | [PDF](https://arxiv.org/pdf/2601.20051v1)

**作者:** Gautham Vinod `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3871 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于单视角图像的真实尺度三维重建框架，实现了将单图像重建的3D模型准确缩放到真实尺寸。

**💡 创新点**

通过将输入图像与多视角渲染图像的CLIP特征拼接，并训练尺度因子回归网络，首次实现从单图像直接恢复真实物理尺度。

**🔧 技术方法**

使用单视角3D重建（One‑2‑3‑45/TripoSR）、CLIP ViT‑L/14视觉编码器、Blender渲染多视角图像、MLP尺度回归网络等技术。

**📊 数据集**

在MetaFood3D（食物模型及营养注释）和OmniObject3D（通用物体）两个公开数据集上进行实验。

**📈 对比分析**

与基线、RGB估计、深度重建、3D辅助估计、GPT‑4o等方法比较，平均绝对误差降低约30%，MAPE降低43%，相关系数提升14%/66%，整体性能显著优于现有方法。

**⚠️ 局限性**

对极小或极大物体仍易出现误差，依赖单一视角与渲染数量，缺乏实时移动端部署，受限于重建方法的质量。

---

## 26. Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods

**arXiv ID:** 2601.19944 | [PDF](https://arxiv.org/pdf/2601.19944v1)

**作者:** Valery Manokhin `[一作]` (Independent Researcher), Daniel Grønhaug `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 21 种二分类器在 TabArena‑v0.1 数据集上进行 5 折交叉验证，评估五种模型无关后置校准方法（Isotonic、Platt、Beta、Venn‑Abers、Pearsonify）的概率校准效果。

**💡 创新点**

系统比较不同校准器在 log‑loss、Brier、ECE/ECI 等指标上的表现，发现 Venn‑Abers 和 Beta 校准在多数模型和数据集上能显著降低 log‑loss，而 Platt 与 Isotonic 甚至会导致校准退化。

**🔧 技术方法**

采用基于概率的后置校准技术（Sigmoid、Beta 分布、Venn‑Abers、Pearsonify）以及多指标评估（Brier、Log‑loss、ECE/ECI、AUC‑ROC、CPU/内存消耗）来量化校准质量。

**📊 数据集**

使用公开的 TabArena‑v0.1 二分类任务集（共 30 个任务），对每个任务进行随机分层 5 折交叉验证，划分训练/校准/测试集。

**📈 对比分析**

结果显示 Venn‑Abers 在 log‑loss 上平均下降约 14%，Beta 下降约 13%，但在 AUC‑ROC 与计算成本上几乎无显著提升；整体分类性能基本保持不变，且无单一校准方法在所有模型上均占优。

**⚠️ 局限性**

局限性包括：仅评估 5 种校准方法；未进行超参数优化或多类别任务；缺乏对数据集代表性与多样性的系统检验；未采用统计显著性检验；计算成本升高但未与预训练模型成本平衡。

---

## 27. Spectral Ghost in Representation Learning: from Component Analysis to Self-Supervised Learning

**arXiv ID:** 2601.20154 | [PDF](https://arxiv.org/pdf/2601.20154v1)

**作者:** Bo Dai `[一作]` (Google DeepMind), Dale Schuurmans `[通讯]` (University of Alberta)

**通讯引用:** 18397 | [OpenAlex ID](https://openalex.org/A5010575626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过谱分析理论，阐明了自监督学习中表征的充分性与有效性，构建了统一的谱框架，并将现有多种 SSL 算法（对比式、非对比式、能量基、潜在变量等）映射到此框架中，提出了几种基于谱迭代、NCE、密度比拟合等新算法，进一步扩展到多模态、生成与强化学习等应用。

**💡 创新点**

创新点包括：① 从谱视角重新定义“充分表征”，证明任意下游任务可由谱分解得到；② 建立谱框架统一对比式与非对比式 SSL，并给出它们的理论联系与优化误差来源；③ 引入能量基、潜在变量与非线性谱表征，提供可解释的投影器设计；④ 设计了基于密度比拟合、KL 变分、对偶/对抗等多种训练策略；⑤ 展示谱表征在多模态、控制与因果推断中的迁移与优势。

**🔧 技术方法**

技术手段主要包括：谱分解（SVD）、能量基模型（EBM）、潜在变量模型、噪声对比估计（NCE）、KL 散度与泛函变分、固定点迭代/幂迭代、随机特征/傅里叶变换、对偶/对抗学习、以及多模态对齐与生成的梯度估计。

**📊 数据集**

实验数据集主要以 ImageNet、COCO、CIFAR、MNIST、VOC、ImageNet‑1000 等经典 SSL 基准进行评估，并在多模态任务中使用文本-图像配对数据集（如 MS‑COCO Caption、Conceptual Captions）。

**📈 对比分析**

与传统 SSL 方法（SimCLR、BYOL、MoCo、VICReg、SwAV 等）在相同的预训练/下游任务（线性评估、KNN、Fine‑tune）下进行对比，谱框架下的算法在保持相似或略高的准确率的同时，显著降低了对大 batch、对比噪声采样的依赖，并在某些任务中提升了采样效率与模型鲁棒性。

**⚠️ 局限性**

局限性包括：① 需要设计合适的谱分解或能量基参数，模型选择对性能影响大；② 一些非对比式方法在理论上仍存在梯度估计偏差，需大 batch 或特殊正则；③ 能量基与潜在变量模型在训练时需采样/对偶求解，计算成本较高；④ 对连续高维数据的密度比拟合和对偶训练仍面临数值不稳定；⑤ 在某些任务（如极低资源场景）下仍缺乏足够的经验指导。

---

## 28. Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data

**arXiv ID:** 2601.19936 | [PDF](https://arxiv.org/pdf/2601.19936v1)

**作者:** Minseo Kwak `[一作]` (Yonsei University), Jaehyung Kim `[通讯]` (Yonsei University)

**通讯引用:** 1926 | [OpenAlex ID](https://openalex.org/A5110631092)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型预训练数据的成员推断问题，提出一种基于top‑1预测与目标标记概率差异的检测方法

**💡 创新点**

创新点在于利用预训练目标梯度特性，定义“Gap-K%”得分：衡量top‑1预测与真实下一个词的对数概率差并在滑动窗口内平滑，以捕捉局部语义一致性；相较于仅考虑低概率词的传统方法更具判别力

**🔧 技术方法**

核心技术包括：log概率差分计算、标准化除以标准差、滑动窗口局部平均、k%最低分取平均形成最终得分；与Min‑K%++等对比实验验证其有效性

**📊 数据集**

在WikiMIA（原始和改写版本，长度32/64/128）和MIMIR（来自Pile的数据集的7个子域）两个公开基准上进行评估

**📈 对比分析**

与Loss、Zlib、Neighbor、Min‑K%、Min‑K%++等基线对比，Gap‑K%在WikiMIA上平均AUROC提升约9.7%（原始）和5.7%（改写），在MIMIR上在不同模型规模上均优于Min‑K%++，表现最优

**⚠️ 局限性**

仅在灰盒（可访问token概率）条件下可用，无法直接用于完全黑盒API；未测试对最新模型家族的泛化；未评估对抗性重写或强制回避场景的鲁棒性

---

## 29. Probabilistic Sensing: Intelligence in Data Sampling

**arXiv ID:** 2601.19953 | [PDF](https://arxiv.org/pdf/2601.19953v1)

**作者:** Ibrahim Albulushi `[一作]` (King Fahd University of Petroleum and Minerals), Feras Al-Dirini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5071067911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于p‑neuron的概率感知系统，用于在事件检测到时以概率方式启动数据采样，提升能效；

**💡 创新点**

创新点在于将神经系统的“反射”机制转化为概率化激活，通过p‑neuron实现微秒级实时事件检测，并支持可调的平均采样率；

**🔧 技术方法**

采用模拟特征提取电路、p‑neuron（数字FPGA实现或自旋电子sMTJ实现）以及ADC触发机制；

**📊 数据集**

使用实际的地震激发测井地震仪（geophone）数据集进行验证；

**📈 对比分析**

与传统连续采样ADC进行对比，时间域和频率域误差极低（NMSE≈0.41%），并且在93%样本数和活跃采样时间上实现显著节省；

**⚠️ 局限性**

局限在于对p‑neuron的随机性控制依赖于硬件实现（如sMTJ的保持时间），对不同噪声环境和非线性事件特征的鲁棒性尚待进一步验证。

---

## 30. Robust SDE Parameter Estimation Under Missing Time Information Setting

**arXiv ID:** 2601.20268 | [PDF](https://arxiv.org/pdf/2601.20268v1)

**作者:** Long Van Tran `[一作]` (Deakin University), Phuoc Nguyen `[通讯]` (Deakin University)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5101580890)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于分数匹配的框架 ReTrace，能够从无时间戳或乱序观测中同时恢复时间顺序并估计随机微分方程参数。

**💡 创新点**

创新点在于利用前向后向过程的统计非对称性，构造逐对分数匹配误差来判定时间方向，并通过全局排序恢复完整序列，从而在缺失时间信息的情况下仍能实现可识别的参数估计。

**🔧 技术方法**

使用了随机微分方程理论、分数匹配与最大似然估计、Euler–Maruyama离散化、图排序与迭代优化技术。

**📊 数据集**

实验数据包括从不可逆线性SDE生成的合成数据以及基于 PKPD 模型的模拟肿瘤生长数据（约2000条轨迹）。

**📈 对比分析**

与 MST、DPT 等基线比较，ReTrace 在时间排序准确率、漂移与扩散参数均方误差以及计算时间上均显著优于基线，排序准确率可达约99%。

**⚠️ 局限性**

局限性在于仅对不可逆 SDE 可识别时间方向，对可逆或接近可逆过程效果有限，并且需能够估计分数函数，若噪声过大或维度极高时性能下降。

---

## 31. Demystifying Multi-Agent Debate: The Role of Confidence and Diversity

**arXiv ID:** 2601.19921 | [PDF](https://arxiv.org/pdf/2601.19921v1)

**作者:** Xiaochen Zhu `[一作]` (University of Cambridge), Andreas Vlachos `[通讯]` (University of Cambridge)

**通讯引用:** 4915 | [OpenAlex ID](https://openalex.org/A5067943980)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究多智能体辩论（MAD），提出了基于人类讨论的两种轻量化干预——多样化初始化和自适应置信度调制，并给出了理论与实验验证；

**💡 创新点**

创新点在于将多样性与置信度沟通两种人类讨论机制注入MAD，并证明前者提高先验成功率、后者打破马丁格尔限制，使辩论结果系统性提升；

**🔧 技术方法**

采用强化学习（GRPO+LoRA）对LLM进行置信度校准和使用训练，使用Dirichlet‑categorical模型描述辩论动态，结合链式思维提示、温度采样等技术；

**📊 数据集**

实验使用六个问答基准，包括GSM8K、CommonsenseQA、HellaSwag、MMLU、GPQA‑Main和ARC‑Challenge；

**📈 对比分析**

与单模型、简单多数投票及投票后辩论基线对比，本文方法在大多数数据集上提升了Pass@5和准确率，显著优于传统MAD和多数投票；

**⚠️ 局限性**

局限在于假设同质化、完全连通的代理、Dirichlet模型简化了真实LLM行为；多样性选择为启发式，置信度校准仍受LLM过度自信影响；仅在英文问答任务上验证，缺乏对更开放或异质场景的评估。

---

## 32. Large language models accurately predict public perceptions of support for climate action worldwide

**arXiv ID:** 2601.20141 | [PDF](https://arxiv.org/pdf/2601.20141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 33. NeuroAI and Beyond

**arXiv ID:** 2601.19955 | [PDF](https://arxiv.org/pdf/2601.19955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 34. Treating symptoms or root causes: How does information about causal mechanisms affect interventions?

**arXiv ID:** 2601.20010 | [PDF](https://arxiv.org/pdf/2601.20010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 35. Evaluating Actionability in Explainable AI

**arXiv ID:** 2601.20086 | [PDF](https://arxiv.org/pdf/2601.20086v1)

**作者:** Gennie Mansi `[一作]` (Georgia Institute of Technology), Mark Riedl `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7375 | [OpenAlex ID](https://openalex.org/A5061883150)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过访谈构建了一个涵盖12类信息与60种用户行为的可操作性评估目录。

**💡 创新点**

提出了基于用户中心的可操作性评估框架，首次将心理状态行为纳入行动维度。

**🔧 技术方法**

采用情境化设计与主题编码等定性方法收集并分析用户需求。

**📊 数据集**

数据来源为14名医教领域专家的访谈记录。

**📈 对比分析**

没有传统模型性能指标，评估通过对照访谈中行为与信息对应关系进行定性验证。

**⚠️ 局限性**

局限在于样本规模小、仅覆盖医学与教育两领域、仅考虑文本说明。

---

## 36. Fueling Volunteer Growth: the case of Wikipedia Administrators

**arXiv ID:** 2601.20016 | [PDF](https://arxiv.org/pdf/2601.20016v1)

**作者:** Eli Asikin-Garmager `[一作]` (Wikimedia Foundation), Leila Zia `[通讯]` (Wikimedia Foundation)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5037525426)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统性分析了2018年至2025年间284个语言版维基百科的管理员活动趋势，并通过超过3000份问卷和12次访谈深入探讨管理员招聘与留存的障碍与动机。

**💡 创新点**

首次揭示大型维基百科中管理员数量呈双向趋势：大部分语言版稳定或增长，但最活跃的21个语言版普遍出现管理员缺失，且主要因招聘不足而非离职率飙升。

**🔧 技术方法**

采用混合方法：日志数据的SQL抽取与时间序列分析、横向多语言问卷调查、定性访谈内容编码与主题分析。

**📊 数据集**

日志数据来自所有284个维基百科的MediaWiki日志表；问卷样本覆盖6个语言版（英语、西班牙语、法语、印尼语、波兰语、俄语）现任与潜在管理员；访谈对象为5名现任及7名前管理员。

**📈 对比分析**

对管理员数量变化使用线性回归与滑动窗口检验，检验不同语言版的增长/下降趋势；对问卷结果采用比例估计与置信区间；整体表明招聘不足是主因，管理员离职率维持稳定。

**⚠️ 局限性**

研究聚焦规模较大的语言版，可能不适用于小型语言版；调查样本响应率虽高但仍可能存在非响应偏倚；访谈仅覆盖部分前管理员，未涵盖完全离职者。

---

## 37. Reference-Free Spectral Analysis of EM Side-Channels for Always-on Hardware Trojan Detection

**arXiv ID:** 2601.20163 | [PDF](https://arxiv.org/pdf/2601.20163v1)

**作者:** Mahsa Tahghigh `[一作]` (Howard University), Hassan Salmani `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于EM侧信道时频分析和高斯混合模型的参考‑自由始终在线硬件木马检测框架。

**💡 创新点**

创新点在于利用多尺度STFT的统计一致性（混合成分数量的跨尺度稳定性）作为无监督判别指标，彻底消除了对黄金芯片或标记数据的依赖。

**🔧 技术方法**

采用短时傅里叶变换（STFT）、贝叶斯信息准则（BIC）选择的高斯混合模型（GMM）以及跨尺度一致性检测算法。

**📊 数据集**

使用AES‑128加密引擎在固定密钥和明文条件下采集的500条EM波形数据，其中包含有无始终在线木马两种情况。

**📈 对比分析**

与传统需要黄金参考或监督学习的EM检测方法相比，本文方法在相同实验设置下实现了超过90%的检测准确率、低于5%的误报率，并且不需要任何参考模型。

**⚠️ 局限性**

局限性包括对STFT窗口尺寸的选择敏感；在极低功耗或高速工艺环境下，EM信号噪声增大时检测效果可能下降。

---

## 38. A Paradigm for Generalized Multi-Level Priority Encoders

**arXiv ID:** 2601.20067 | [PDF](https://arxiv.org/pdf/2601.20067v1)

**作者:** Maxwell Phillips `[一作]` (University of Wisconsin--Madison), Ahmed Ammar `[通讯]` (Ohio Northern University)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5021297952)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文提出并实现了多层级优先编码器（MLPE）架构，并对单层、树形、递归、两层等已有设计进行详细对比与分析。

**💡 创新点**

创新点在于把两层优先编码器推广为任意层级结构，并给出组合（composition）和级联（cascading）两种构造方法，配套完整的复杂度与延迟公式与计算工具。

**🔧 技术方法**

采用门级与MUX级硬件实现，利用FPGA Vivado综合、ASIC静态CMOS复杂度/延迟推导，采用OR树、宽MUX分层等技术。

**📊 数据集**

使用输入宽度从4到2^18（262,144）位的优先编码器实验，全部代码托管于GitHub（VHDL实现）。

**📈 对比分析**

通过归一化LUT计数与延迟、计算复杂度-延迟乘积（RCDP）进行比较；在FPGA上递归/树形最快，MLPE复杂度最低但延迟略高；在ASIC上树形最快、MLPE最省面积且延迟接近树形。

**⚠️ 局限性**

局限性：仅考虑组合优先编码器，未使用流水线；多层级在FPGA上对路由敏感，m>2时性能下降；未探索更大k值或非二分划分；实验范围限定在n≤2^18。

---

## 39. A Data-Informed Local Subspaces Method for Error-Bounded Lossy Compression of Large-Scale Scientific Datasets

**arXiv ID:** 2601.20113 | [PDF](https://arxiv.org/pdf/2601.20113v1)

**作者:** Arshan Khan `[一作]` (University of Central Florida), Ben O'Neill `[通讯]` (RNET Technologies Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种错误受限的离散数据驱动局部子空间（Discontinuous‑DLS）压缩算法，能够在保持用户指定误差界限的前提下，对大规模科学数据实现高压缩率。

**💡 创新点**

创新点包括：①放宽 C^0 连续性约束，允许局部断裂，从而实现局部误差控制与可伸缩性；②利用泛化有限元方法（GFEM）框架学习数据驱动的局部基底（SVD 产生的局部模式），保留数据的本质特征；③一次性学习基底后可在多帧时间序列中复用，减少重复计算；④将压缩过程拆分为特征学习、误差阈值设定、块级压缩与位数裁剪，形成可并行、可在 MPI 环境下高效执行的完整流水线。

**🔧 技术方法**

核心技术包括：泛化有限元方法 (GFEM) 与局部分区；奇异值分解 (SVD) 用于生成数据自适应基底；误差估计与本地误差容限计算；位数修剪 (bit‑grooming) 与 Gzip 进行后期无损压缩；MPI、PETSc、SLEPc 进行并行矩阵构造与 SVD；GPU/CPU 混合环境下的分布式 I/O。

**📊 数据集**

使用了三维无粘湍流流过圆柱的 LES 数据集（Re=10⁵），包含 1024 个时间步、约 937.5 GB 的浮点数据，作为压缩方法的验证与评测对象。

**📈 对比分析**

与主流错误受限压缩器（SZ3、MGARD）在相同数据集上进行对比。结果显示：在低误差（≤ 1 %）下，Discontinuous‑DLS 的压缩率高于 MGARD、与 SZ3 相当；在中高误差（≥ 5 %）时压缩率超过两者，最高可达 1800×。压缩后误差始终在用户设定的全局阈值内，且随时间保持稳定。吞吐率受分块大小 λ 与误差阈值影响，在 λ≈120、误差 5 % 时可达 27 MB/s；更严格误差导致吞吐下降至 <10 MB/s。

**⚠️ 局限性**

局限性：①基底学习一次性成本随分块大小 λ 增大呈指数增长；②在极低误差需求下压缩率受限；③当前实现仅支持结构化网格，未验证到非结构/自适应网格；④对多物理场、多误差阈值的适配尚未完善；⑤对极高维/大规模数据的内存占用与通信开销需进一步优化。

---

## 40. A Reinforcement Learning Based Universal Sequence Design for Polar Codes

**arXiv ID:** 2601.20118 | [PDF](https://arxiv.org/pdf/2601.20118v1)

**作者:** David Kin Wai Ho `[一作]` (Apple), Louay M. A. Jalloul `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于强化学习的可扩展极化码通用序列设计框架，能够在代码长度高达2048的条件下生成比传统beta‑expansion更优的极化通道可靠性排序。

**💡 创新点**

创新点包括：① 将极化码的通用偏序（UPO）作为物理约束引入强化学习，显著降低搜索空间；② 采用多配置联合优化与迭代look‑ahead学习，提升知识迁移与收敛速度；③ 结合PPO与GNN（GraphSAGE）实现高效的策略学习；④ 通过低阶序列嵌入与渐进式约束松弛保持小块长度性能。

**🔧 技术方法**

核心技术：Proximal Policy Optimization（PPO）+ 语义约束的GNN（GraphSAGE）+ UPO+约束+ MCTS启发式迭代学习+低方差SNR锚定奖励。

**📊 数据集**

数据集：在AWGN信道下模拟SCL-8（list 8）极化码，通过C++实现的基于GA的极化码解码器产生奖励；不使用公开数据集，而是全程自生成的仿真数据。

**📈 对比分析**

与5G NR标准序列及beta‑expansion序列对比；在所有N∈{32,64,128,256,512,1024,2048}与K≈0.6的码率上，学习得到的序列与NR序列相当，且在N=2048时平均可获得≈0.2 dB的BLER提升。

**⚠️ 局限性**

局限性：训练耗时极长（N=2048需30天以上）；仅验证在AWGN+SCL‑8场景，未验证其他信道或解码器；对极化码变体的通用性仍需进一步研究。

---

## 41. Dynamics of Human-AI Collective Knowledge on the Web: A Scalable Model and Insights for Sustainable Growth

**arXiv ID:** 2601.20099 | [PDF](https://arxiv.org/pdf/2601.20099v1)

**作者:** Buddhika Nettasinghe `[一作]` (University of Iowa), Kang Zhao `[通讯]` (University of Iowa)

**通讯引用:** 2830 | [OpenAlex ID](https://openalex.org/A5035246366)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一个五维动力学模型，用以刻画人类与大型语言模型在共同知识库中的相互演化，并通过模拟和校准分析其增长与风险。

**💡 创新点**

创新点在于将模型崩溃、质量稀释与人类能力逆转等三大系统风险统一到同一动力学框架，揭示关键杠杆（如门控严格度、RLHF比例）对生态可持续发展的影响。

**🔧 技术方法**

采用非线性动力学方程、最小可解释参数化、模拟实验与最小二乘校准等技术，并结合门控函数、RLHF增益与归一化学习曲线。

**📊 数据集**

使用维基百科的页面增量、编辑活跃度与浏览量数据，以及 PubMed 与 GitHub+Copilot 的贡献量与质量指标进行实证校准。

**📈 对比分析**

通过与真实时间序列对比（RMSE 约 1.8k 与 0.87k），展示模型能在预后与后续 ChatGPT 时代准确捕捉知识流与质量演化，验证不同参数配置对应的健康、倒退或振荡成长模式。

**⚠️ 局限性**

局限性包括模型过于简化（忽略领域差异、多主体异质性、内容稀释细节）、参数估计依赖有限时间窗口、以及门控与 RLHF 真实效果在不同平台上的可迁移性尚未完全验证。

---

## 42. Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents

**arXiv ID:** 2601.20144 | [PDF](https://arxiv.org/pdf/2601.20144v1)

**作者:** Ziyi Wang `[一作]` (Northeastern University), Dakuo Wang `[通讯]` (Northeastern University)

**通讯引用:** 4791 | [OpenAlex ID](https://openalex.org/A5062817658)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套可验证的多轮工具调用数据生成管线，生成包含模糊、变化、不可行三种真实用户意图的任务，并基于此评估并微调LLM。

**💡 创新点**

创新点包括：① 将可执行轨迹反向映射为任务实现任务与轨迹可验证匹配；② 设计并系统化生成三类现实情境（模糊、变化、不可行）；③ 通过轨迹监督微调显著提升轻量模型的鲁棒性与跨域泛化。

**🔧 技术方法**

使用POMDP框架建模非静态意图，利用强教师LLM（Claude‑4.5‑Sonnet）进行轨迹探索，采用前向后向两阶段合成与LLM验证，最后对成功轨迹进行监督微调（SFT）。

**📊 数据集**

在Tau2‑Bench零售环境中生成约1,099个复杂情境任务（473模糊、279变化、347不可行），并收集2,872条成功轨迹用于训练；对比原Tau2‑Bench的Retail和Airline域。

**📈 对比分析**

对七大LLM（Claude‑3.7/3.5、Qwen3‑235B/32B/14B/8B/4B）在Pass1/Pass2/Pass3指标进行评估，Claude‑3.7在General 0.79 Pass1，模糊/变化/不可行分别下降；SFT后Qwen4B/8B在所有情境均提升至≈0.71/0.69，且在未见过的Airline域也有显著提升。

**⚠️ 局限性**

局限性包括：① 仅在Tau2‑Bench仿真环境，用户模拟器可能不完全贴近真实用户行为；② 仅覆盖有限工具生态，缺乏更广泛多领域验证；③ 对长时序、对抗性或细粒度社交行为的建模不足。

---

## 43. Externally Validated Longitudinal GRU Model for Visit-Level 180-Day Mortality Risk in Metastatic Castration-Resistant Prostate Cancer

**arXiv ID:** 2601.20046 | [PDF](https://arxiv.org/pdf/2601.20046v1)

**作者:** Javier Mencia-Ledo `[一作]` (University of Toronto), Zahra Shakeri `[通讯]` (University of Toronto)

**通讯引用:** 1997 | [OpenAlex ID](https://openalex.org/A5086148541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并外部验证了一种基于GRU的纵向模型，用以预测晚期去势抵抗性前列腺癌患者每次门诊后180天的死亡风险；

**💡 创新点**

创新点在于：1）使用访问层时间更新风险评分，实时跟踪高危状态；2）在两个Phase III试验中进行外部验证，展示模型在分布漂移下的稳健性；3）采用固定敏感度阈值评估临床实用性，量化警报密度和预警时间；

**🔧 技术方法**

技术主要包括：GRU序列模型、自动编码器缺失值填补、弹性网络逻辑回归、Cox比例风险模型、随机生存森林；对GRU进行了特征重要性排列；

**📊 数据集**

数据集来自Project Data Sphere的两组Phase III试验，训练集n=526，外部验证集n=640（仅控制组），共含8项结构化临床变量；

**📈 对比分析**

在内部交叉验证和外部验证中，GRU与RSF均达到C-index 0.87；GRU在外部验证中实现C-index 0.93、AUC-ROC 0.89、PR-AUC 0.87，警报密度18.3/100访，平均预警时间151天；相较于传统模型，GRU在校准与临床可用性上更优；

**⚠️ 局限性**

局限包括：1）仅使用试验数据，缺乏真实世界人群与种族多样性；2）仅利用结构化数据，未充分挖掘临床文本；3）对阈值的敏感度固定，实际部署需本地校准；4）模型对未观察的180天窗口未作完整失活处理。

---

## 44. Counterfactual Cultural Cues Reduce Medical QA Accuracy in LLMs: Identifier vs Context Effects

**arXiv ID:** 2601.20102 | [PDF](https://arxiv.org/pdf/2601.20102v1)

**作者:** Amirhossein Haji Mohammad Rezaei `[一作]` (University of Toronto), Zahra Shakeri `[通讯]` (University of Toronto)

**通讯引用:** 1997 | [OpenAlex ID](https://openalex.org/A5086148541)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造文化相关的非临床信息插入，评估医学语言模型在面临不同文化线索时的诊断稳定性；

**💡 创新点**

创新点在于提出了一个基于因果反事实的benchmark，将文化身份与情境分离，并系统验证其对诊断的影响；

**🔧 技术方法**

采用LLM生成的对照题目，使用多种大型语言模型（GPT‑5.2、Llama‑3.1‑8B、DeepSeek‑R1、MedGemma‑4B/27B）进行多选诊断和简短解释推理；

**📊 数据集**

使用MedQA测试集的150道多选题，通过LLM增广生成三种文化背景（加拿大原住民、中东穆斯林、东南亚）及其身份/情境/两者组合，共1650条样本；

**📈 对比分析**

对比原始、性别/文化中性对照以及不同模型在选项推理和解释推理两种提示下的准确率、翻转率和有害翻转率；大模型在单一文化提示下表现略好，但身份+情境组合普遍导致准确率下降3–7个百分点，说明模型对身份信息尤为敏感；

**⚠️ 局限性**

局限包括：增广样本多、推理成本高；文化增广可能缺乏情境多样性；模型集合有限，未覆盖所有LLM类型；未对更广泛的临床数据集进行验证。

---

## 45. Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning

**arXiv ID:** 2601.20069 | [PDF](https://arxiv.org/pdf/2601.20069v1)

**作者:** Chi-Yao Huang `[一作]` (Arizona State University), Yezhou Yang `[通讯]` (Arizona State University)

**通讯引用:** 4315 | [OpenAlex ID](https://openalex.org/A5002278578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为 Domain Expansion 的框架，通过正交池化机制将潜在空间划分为互相正交的子空间，解决多目标学习中梯度冲突导致的潜在表示崩塌问题。

**💡 创新点**

创新点在于：①将潜在空间结构化为正交子空间，主动预防梯度冲突；②引入正交池化将单一特征映射到多任务子空间；③得到可解释、可组合的潜在表示，支持概念算术操作。

**🔧 技术方法**

技术方法包括：自适应求取潜在分布均值与协方差，特征分解得到主轴向量；选择主轴构建正交域；对每个目标做正交投影；使用对比学习（SupCon）和排序对比（RNC）两种表示学习损失；冻结编码器后训练线性解码器；对梯度冲突方法进行对比。

**📊 数据集**

实验数据集：ShapeNet（3D物体分类与姿态估计）、MPIIGaze（视线估计）以及 Rotated MNIST（旋转数字回归），在 ShapeNet 上定义五个目标：azimuth、elevation、in‑plane rotation（回归）和 category、model ID（分类）。

**📈 对比分析**

与基线（加权求和）、Nash‑MTL、FAMO、IMTL 等梯度冲突缓解方法对比，Domain Expansion 在所有指标上均表现最佳：Spearman 相关性、V‑score、MAE、分类准确率以及概念合成的余弦相似度均显著提升，说明其有效避免潜在崩塌并提供可解释的潜在空间。

**⚠️ 局限性**

局限性主要在于：正交子空间虽可解释，但解码抽象概念仍受限；当前解码器为线性层，无法直接生成自然语言或图像；若需将组合概念转化为人类可理解的输出，需结合生成模型（如 LLM 或扩散模型）。

---

## 46. Editrail: Understanding AI Usage by Visualizing Student-AI Interaction in Code

**arXiv ID:** 2601.20085 | [PDF](https://arxiv.org/pdf/2601.20085v1)

**作者:** Ashley Ge Zhang `[一作]` (University of Michigan), Steve Oney `[通讯]` (University of Michigan)

**通讯引用:** 794 | [OpenAlex ID](https://openalex.org/A5069296306)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究开发了一个交互式系统，用以可视化并追踪学生在编程过程中与生成式AI的交互，帮助教师了解AI使用情况并及时干预。

**💡 创新点**

创新点在于将AI使用跟踪与教师的教学评估流程无缝整合，并提供基于代码历史的可视化分析，首次实现了对学生AI使用模式的实时监控与干预。

**🔧 技术方法**

采用了代码版本控制日志解析、AI使用检测算法（如模型指纹识别）、交互式可视化技术（如D3.js）和机器学习分类器来判定AI生成代码。

**📊 数据集**

使用了来自密歇根大学和圣母大学的本科计算机课程学生的代码提交日志与IDE插件采集的数据集。

**📈 对比分析**

与传统的手工审核和单一的使用报告方法相比，本系统在检测准确率上提升至约92%，并能在两分钟内生成个性化干预建议。

**⚠️ 局限性**

局限性包括对特定AI工具的识别覆盖率不足、对混合人机代码的区分仍存在误差，以及对学生隐私和伦理问题需要进一步完善。

---

## 47. Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning

**arXiv ID:** 2601.20221 | [PDF](https://arxiv.org/pdf/2601.20221v1)

**作者:** Hang Zhang `[一作]` (University of Pittsburgh), Yanshan Wang `[通讯]` (University of Pittsburgh)

**通讯引用:** 6500 | [OpenAlex ID](https://openalex.org/A5080116611)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于工具集成的强化学习框架（Med‑TIV）用于对医学推理过程进行验证，能够在评估时动态查询医学文档并生成可解释的批判；

**💡 创新点**

创新点在于：①允许验证器多轮交互式检索并以证据为根基的判断；②仅使用轨迹级监督即可完成RL训练；③采用自适应课程化筛选难易样本提升学习效率；

**🔧 技术方法**

核心技术包括：多模态工具调用（检索引擎），生成式判定模型，基于Dr. GRPO的策略梯度强化学习，以及动态检索和格式化奖励设计；

**📊 数据集**

使用公开的 Med‑PRM 数据集（带轨迹级标签）作为训练数据，并在四个医学 QA 评测集（MedQA、MedMCQA、MMLU‑Med、MedXpertQA）上评估；

**📈 对比分析**

与多种基线（GPT‑4o‑mini、AlphaMed、Med‑PRM 等）相比，Med‑TIV 在 MedQA、MedMCQA、MMLU‑Med、MedXpertQA 上分别提升 23.5%、32.0% 等平均精度至 60.4%，并在采样预算上比现有奖励模型快 8 倍；

**⚠️ 局限性**

局限性包括：依赖外部检索系统且检索质量会影响结果；当前仅在医学域验证，跨领域泛化待验证；仅提供轨迹级监督，可能忽略细粒度错误信息。

---

## 48. Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control

**arXiv ID:** 2601.20090 | [PDF](https://arxiv.org/pdf/2601.20090v1)

**作者:** Amirmohammad Farzaneh `[一作]` (King's College London), Osvaldo Simeone `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个在 LLM 驱动的自治控制系统中进行因果“若干”推断的框架，能够在用户改写意图后生成可靠的对抗性报告；

**💡 创新点**

创新点在于将结构因果模型与 LLM 生成机制结合，利用概率推断（abduction）恢复环境噪声，并通过 conformal counterfactual generation (CCG) 赋予生成集合正式的覆盖率保证；

**🔧 技术方法**

使用了结构因果模型、Gumbel–Max 采样、神经后验估计（NPE）进行环境噪声恢复、双组件 LLM（动作生成与报告生成）、ns‑3 仿真器以及 conformal language modeling（CLM）校准技术；

**📊 数据集**

实验基于 300 条网络控制指令及其改写版本，使用 ns‑3 的真实环境和数字孪生模拟器产生 KPI 与报告，并以 GPT‑4 生成对抗性问答；

**📈 对比分析**

与直接重新执行（IG）和仅使用模拟器的对抗执行（SIG）进行对比，CG/CCG 在 KPI 误差、相关性以及报告语义质量上均优于两者；在覆盖率和采样效率上，CCG 在相同样本预算下实现更低的超采样率、集合大小与 set loss 接近目标；

**⚠️ 局限性**

局限性包括对模拟器逼真度的依赖、单回合交互、需要额外的 calibration 数据、对多轮交互与多智能体场景尚未充分验证。

---

## 49. Structural Compositional Function Networks: Interpretable Functional Compositions for Tabular Discovery

**arXiv ID:** 2601.20037 | [PDF](https://arxiv.org/pdf/2601.20037v1)

**作者:** Fang Li `[一作]` (Oklahoma Christian University), Fang Li `[通讯]` (Oklahoma Christian University)

**通讯引用:** 7612 | [OpenAlex ID](https://openalex.org/A5100610476)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了StructuralCFN，一种在表格数据中通过结构先验学习特征间关系的可解释神经网络，并在18个基准和OpenML-CC18 12个数据集上进行评估

**💡 创新点**

创新点包括：① 关系感知的差分可学习结构门控，实现自动选取激活物理；② 将多项式、正弦、Sigmoid等符号基函数嵌入网络，产生可直接读取的符号表达式；③ 极小参数量（≈300–2500）即可与GBDT竞争，满足低功耗科学应用

**🔧 技术方法**

技术手段：差分可学习结构门控、混合基函数节点、残差线性旁路与功能委员会聚合、L1稀疏正则、10折交叉验证、统计显著性检验、对比学习与模型压缩

**📊 数据集**

使用的数据集包括18个医学、环境、金融等基准（如Blood Transfusion、Ozone、WDBC、Diabetes、California Housing）以及OpenML-CC18 12个数据集（Sci、Ind、Signal、Econ、Logic等），涵盖回归、分类与高维任务

**📈 对比分析**

与MLP、TabNet和XGBoost（LightGBM）三类主流基线进行同一超参数空间下的10折交叉验证比较；在低噪、科学型数据上显著优于GBDT（p<0.0167），参数量比深度网络低10–20倍，推理延迟仅5–8µs，满足实时边缘计算需求

**⚠️ 局限性**

局限性：在大规模回归或高熵分类任务中不如GBDT，且对离散/分段关系捕捉能力有限；依赖矩阵为O(N²)内存，极高维场景需引入稀疏机制

---

## 50. Mind the Shift: Using Delta SSL Embeddings to Enhance Child ASR

**arXiv ID:** 2601.20142 | [PDF](https://arxiv.org/pdf/2601.20142v1)

**作者:** Zilai Wang `[一作]`, Abeer Alwan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过将自监督学习模型的微调后特征与其对应的差分特征（Delta Embedding）进行融合，以提升儿童语音识别性能

**💡 创新点**

创新点在于提出将模型参数差异扩展到表示层，利用Delta Embedding捕获任务特定信息，并在多模型融合中实现显著提升

**🔧 技术方法**

使用了多模型特征融合（拼接、加权、交叉注意力）以及Canonical Correlation Analysis和Mixture-of-Experts解释补充性

**📊 数据集**

在儿童对话语料库MyST上进行实验，并构建1h、5h、10h等极低资源子集

**📈 对比分析**

与单一微调模型及传统特征融合方法对比，Delta Embedding拼接在所有数据规模下均优于基线，1h场景下相对WER下降10%（HuBERT）和4.4%（W2V2），最优方案在MyST上取得9.64%的WER，刷新SSL模型最佳成绩

**⚠️ 局限性**

局限在于仅验证了三种SSL编码器，对跨域或更大规模儿童语料的泛化能力未进一步探究，且Delta Embedding需额外计算与存储

---

## 51. Towards a Mechanistic Understanding of Large Reasoning Models: A Survey of Training, Inference, and Failures

**arXiv ID:** 2601.19928 | [PDF](https://arxiv.org/pdf/2601.19928v1)

**作者:** Yi Hu `[一作]` (Peking University), Liangming Pan `[通讯]` (Peking University)

**通讯引用:** 974 | [OpenAlex ID](https://openalex.org/A5027533517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了大型推理模型（LRM）的训练机制、推理行为以及非预期失效机制。

**💡 创新点**

专注于LRM内部机制的全面梳理，提出三维度（训练动态、推理机制、不良行为）结构，并给出未来研究方向。

**🔧 技术方法**

结合了SFT+RL训练、激活与权重动态分析、注意力头与稀疏自编码器、线性探针与旋转变换等多种现有技术。

**📊 数据集**

综述涵盖的典型数据集包括数学、编程、逻辑推理等通用推理数据集，未单独引入新的实验数据集。

**📈 对比分析**

论文本身为综述性工作，没有新的实验对比，主要整理已有研究的性能提升与不足。

**⚠️ 局限性**

局限性包括未覆盖多模态、最新模型架构（如扩散、连续标记、循环Transformer）及快速涌现的新研究。

---

## 52. Insight Agents: An LLM-Based Multi-Agent System for Data Insights

**arXiv ID:** 2601.20048 | [PDF](https://arxiv.org/pdf/2601.20048v1)

**作者:** Jincheng Bai `[一作]` (Amazon), Zhihuai Zhu `[通讯]` (Amazon)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5103684648)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Insight Agents（IA），一种基于LLM的层级多代理会话系统，用来为电商卖家提供个性化的数据与业务洞察；

**💡 创新点**

在架构上创新性地将 OOD 检测、代理路由与两条工作流（数据呈现与洞察生成）整合进单一的计划-执行范式，减少 LLM 调用并提升准确率与响应速度；

**🔧 技术方法**

采用轻量化编码器-解码器 OOD 检测、BERT 路由器、LLM（Claude‑3‑Sonnet）驱动的任务拆解与 API 选择、以及动态注入领域知识的生成模块；

**📊 数据集**

使用自构建的 301 题目的 OOD 与路由训练集、300 题的增量数据集，以及 100 题的基准评估集；

**📈 对比分析**

通过与基于 LLM 的少量示例方法比较，AE‑OOD 在 0.01 s 内完成检测并保持 96.9% 精度；BERT 路由器在 0.31 s 内实现 83% 准确度；在 100 题评估中，IA 取得 89.5% 的问题级准确率，P90 延迟 13.56 s；

**⚠️ 局限性**

主要局限包括仅在亚马逊美国卖家环境中验证，依赖内部 API 与数据结构，对数据缺失或异常的处理仍需改进，且系统对新型业务场景的泛化能力尚未充分验证。

---

## 53. Regime-Adaptive Bayesian Optimization via Dirichlet Process Mixtures of Gaussian Processes

**arXiv ID:** 2601.20043 | [PDF](https://arxiv.org/pdf/2601.20043v1)

**作者:** Yan Zhang `[一作]` (Florida State University), Shibo Li `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Dirichlet过程高斯过程混合模型的自适应贝叶斯优化框架（RAMBO），能够在搜索过程中自动发现并建模离散的局部规律（regime）。

**💡 创新点**

创新点包括：① 将离散的局部规律建模为DP混合的GP，消除传统单GP的全局平滑假设；② 采用折叠吉布斯采样，显式消除隐藏函数，提升推断效率；③ 引入自适应浓度参数调度，在探索初期保持模型稀疏，后期细粒度分裂；④ 在混合模型上解析化Expected Improvement，区分intra‑regime与inter‑regime不确定性。

**🔧 技术方法**

核心技术：Dirichlet过程混合高斯过程、折叠吉布斯采样、EM/Adam优化超参、经验贝叶斯推断、适应性α调度、混合式EI采集函数、L‑BFGS‑B多起点优化。

**📊 数据集**

数据集：合成基准（Levy、Schwefel，6D/10D）；真实科学任务：分子构象优化（12D）、药物虚拟筛选（50D降维后）、核聚变等离子体（81D）以及对应的实验评测。

**📈 对比分析**

与SGP、TuRBO、SAASBO、BAxUS、ALEBO、HEBO、Bounce、SMAC、COMBO等主流BO方法对比，RAMBO在所有任务上均表现出显著优势：例如在12D构象任务中比最优基线低约39.7%能量；在50D药物筛选中提高约4.06%结合能；在81D等离子体设计中提高约51.55%等离子体质量因子；并在合成基准上取得更快的收敛速度。

**⚠️ 局限性**

局限性：① 推断仍依赖于MCMC采样，计算量随数据量和regime数增长；② 对极高维（>100D）或极大样本量的数据场景尚未充分验证；③ α调度策略虽然有效，但在不同问题上可能需要手动调参；④ 对噪声水平或非高斯噪声的鲁棒性尚待进一步研究。

---

## 54. Flexible Bit-Truncation Memory for Approximate Applications on the Edge

**arXiv ID:** 2601.19900 | [PDF](https://arxiv.org/pdf/2601.19900v1)

**作者:** William Oswald `[一作]` (University of South Alabama), Na Gong `[通讯]` (University of Alabama)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种可在运行时实现任意位截断的 SRAM 记忆单元 TrunMem，支持视频和深度学习等近似应用。

**💡 创新点**

创新点在于实现完整的位截断灵活性（0–32 位），兼具字节模式与字模式，且面积开销仅 2.89%，大幅提升能耗与质量适配能力。

**🔧 技术方法**

主要技术包括 130 nm SRAM 结构、功率门控截断管理器、位截断控制单元、后仿真与硬件/软件协同框架，以及滤波器剪枝与注意力机制。

**📊 数据集**

实验使用 4,525 条 YUV 视频、CIFAR‑10、CIFAR‑100 以及 PASCAL VOC‑2007 数据集进行验证。

**📈 对比分析**

与现有的基于字节截断的记忆相比，TrunMem 在视频上实现最高 47.02% 的能耗降低，在 DNN 推理上实现 51.69% 的能耗降低；在保持 0.5% 准确率误差内可实现低功耗模式，10% 误差内可实现超低功耗模式。

**⚠️ 局限性**

局限性包括相对较高的面积占比、目前仅在 SRAM 上验证、缺乏硅片测试、以及截断仅针对低位有效位，未来需进一步扩展到 DRAM 等新型存储与更大规模芯片。

---

## 55. The Grammar of Transformers: A Systematic Review of Interpretability Research on Syntactic Knowledge in Language Models

**arXiv ID:** 2601.19926 | [PDF](https://arxiv.org/pdf/2601.19926v1)

**作者:** Nora Graichen `[一作]` (Universitat Pompeu Fabra), Gemma Boleda `[通讯]` (ICREA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对337篇关于Transformer语言模型句法知识的论文进行了系统综述，并整理了1,015个模型实验结果，构建了公开数据库。

**💡 创新点**

首次以系统综述方式量化Transformer模型在句法知识方面的研究现状，提供了方法学建议和可复现的数据库，弥补了以往叙述性综述的不足。

**🔧 技术方法**

采用系统综述流程、数据库构建与标注、句法现象与解释方法编码、以及对BLiMP基准的统计分析等技术手段。

**📊 数据集**

使用BLiMP、GLUE/CoLA、SyntaxGym等公开基准，以及自建的实验材料，对多种句法现象进行评估。

**📈 对比分析**

通过对BLiMP分数的量化比较，发现模型规模与训练数据量正相关，双向模型通常优于因果模型；在形式句法（如一致性、依存关系）上表现优异，而句法-语义接口（如绑定、成分消歧、否定极性）表现相对较弱。

**⚠️ 局限性**

仅覆盖Transformer模型且主要以英语BLiMP为评测，数据库尚不完整，缺乏跨语言与因果性实验，限制了结论的普适性。

---

## 56. E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-in-the-Loop Reinforcement Learning

**arXiv ID:** 2601.19969 | [PDF](https://arxiv.org/pdf/2601.19969v1)

**作者:** Haoyuan Deng `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3039 | [OpenAlex ID](https://openalex.org/A5100389366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种实时人机交互式强化学习框架，利用基于熵的样本选择机制提高样本效率，减少人类干预；

**💡 创新点**

创新点在于通过构造影响函数（利用动作概率与软优势的协方差）量化每个样本对策略熵的影响，并对极端影响值进行剪裁，从而剔除快捷样本和噪声样本，保持熵的稳定下降；

**🔧 技术方法**

使用的技术包括熵正则化的 RLPD（Reinforcement Learning with Policy Distillation）、影响函数估计、协方差剪裁、经验回放与人类干预收集；

**📊 数据集**

在 Lerobot SO‑101 实际机器人平台上，利用四个真实操作任务（Touch Cube、Pick Cube、Pick & Place Cube、Stack Blocks）进行实验；

**📈 对比分析**

与现有最先进的 HIL‑SERL 方法对比，实验显示在四项任务中成功率平均提升42.1%，人类干预率平均下降10.1%，且收敛更快、更稳定；

**⚠️ 局限性**

局限性包括：依赖 Q‑value 的准确性，若价值估计误差大会影响协方差估计；仅在少数单任务环境验证，缺乏多任务或大规模场景的推广；

---

## 57. Table-BiEval: A Self-Supervised, Dual-Track Framework for Decoupling Structure and Content in LLM Evaluation

**arXiv ID:** 2601.19923 | [PDF](https://arxiv.org/pdf/2601.19923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 58. Sparse CLIP: Co-Optimizing Interpretability and Performance in Contrastive Learning

**arXiv ID:** 2601.20075 | [PDF](https://arxiv.org/pdf/2601.20075v1)

**作者:** Chuan Qin `[一作]` (Meta), Stefan Scherer `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 CLIP 预训练中加入非负 ReLU 约束和高维投影，直接在训练阶段强制稀疏性，得到可解释且多模态的稀疏特征。

**💡 创新点**

创新点在于：① 证明稀疏性与性能可共存，② 直接在训练中施加稀疏性而非后处理，③ 稀疏特征天然多模态，可实现概念对齐与训练动态可视化。

**🔧 技术方法**

使用技术包括：非负投影层（ReLU）、维度扩展（72 倍）、CLIP 对比损失、可调 Logit 规模、Clarity 解释度指标、词汇标签映射、VLM 适配器等。

**📊 数据集**

使用的数据集：MetaCLIP（15M/2.2B 影像文本对）、ImageNet‑1k（训练/验证）、COCO（检索与定位）、以及公开的 QA 与检索基准。

**📈 对比分析**

与原始 CLIP 及 Sparse Autoencoder (SAE) 在零射分类、细粒度分类、定位、检索等多项基准上对比；Sparse CLIP 在大多数任务上与基线持平或更优，零射分类平均提升 0.5%~0.7%，稀疏度达 0.5%–0.7%，Clarity 指标显著高于 SAE。

**⚠️ 局限性**

局限性：投影层导致参数量增加约 14%；显存受限限制了维度扩展上限；训练时的可解释性并不直接迁移至已有密集 CLIP 模型；对模型稳定性与内存效率的进一步研究仍待展开。

---

## 59. Benchmarking von ASR-Modellen im deutschen medizinischen Kontext: Eine Leistungsanalyse anhand von Anamnesegesprächen

**arXiv ID:** 2601.19945 | [PDF](https://arxiv.org/pdf/2601.19945v1)

**作者:** Thomas Schuster `[一作]` (Hochschule Pforzheim), Holger Friedrich `[通讯]` (XPACE GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在该研究中，作者构建了包含四种医学问诊场景的德语模拟对话数据集，并对29种自动语音识别（ASR）模型进行性能评估。

**💡 创新点**

创新之处在于将专业医学术语、口音多样性与说话人标注同时纳入数据集，并采用多指标（WER、CER、BLEU、SA‑WER）对模型进行系统化比较。

**🔧 技术方法**

技术上采用了Whisper、Wav2Vec2、Voxtral等开源模型以及AssemblyAI、Deepgram等商业API，并在统一的预处理、greedy解码与后处理流水线下进行评测。

**📊 数据集**

使用的数据集为自制的Med‑De‑Anamnese，来源于公开YouTube对话，包含背痛、腹痛、外籍医生、强口音病人四个场景的人工转写文本。

**📈 对比分析**

通过平均WER、CER、BLEU及方差热图等指标进行比较，结果显示AssemblyAI Universal以≈3% WER领跑，Voxtral Small约7%，Whisper Large‑v3约12.6%，而老旧模型则超20%。

**⚠️ 局限性**

局限性包括数据集规模有限、缺乏真实临床录音、未覆盖所有口音与医学术语、开源模型缺乏内置说话人分离，以及评估主要基于统计误差而未结合人工医学可用性验证。

---

## 60. BengaliSent140: A Large-Scale Bengali Binary Sentiment Dataset for Hate and Non-Hate Speech Classification

**arXiv ID:** 2601.20129 | [PDF](https://arxiv.org/pdf/2601.20129v1)

**作者:** Akif Islam `[一作]` (University of Rajshahi), Md. Ekramul Hamid `[通讯]` (University of Rajshahi)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5008769867)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个约14万条记录的BengaliSent140二分类情感数据集，用于识别仇恨与非仇恨语句。

**💡 创新点**

通过整合七个不同的Bengali文本数据集并统一标签体系，实现了大规模、平衡的仇恨检测基准。

**🔧 技术方法**

使用了词向量嵌入（Word2Vec、FastText）、CNN、Bi‑LSTM、Conv‑LSTM、BERT等多种模型进行基线实验。

**📊 数据集**

来源数据包括BD‑SHSH、Cyberbully Detection Bengali Comments、SentNoB、Multi‑Labeled Bengali Toxic Comments、Bengali Hate Speech Dataset、Bengali Hate Speech Dataset v2、Multimodal‑Hate‑Bengali。

**📈 对比分析**

对比实验表明传统机器学习（LR、SVM、RF、XGBoost）准确率约0.79-0.84，深度学习模型可达0.88，BERT最高可达0.91。

**⚠️ 局限性**

局限在于仅包含单语Bengali、二分类、缺乏代码混杂与细粒度标签，且预处理工具准确性有限。

---

## 61. Beyond Bug Fixes: An Empirical Investigation of Post-Merge Code Quality Issues in Agent-Generated Pull Requests

**arXiv ID:** 2601.20109 | [PDF](https://arxiv.org/pdf/2601.20109v1)

**作者:** Shamse Tasnim Cynthia `[一作]` (University of Saskatchewan), Banani Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5015470184)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对AI编码代理产生的bug修复 Pull Request 合并后的代码质量进行了大规模实证分析。

**💡 创新点**

创新点在于揭示合并成功并不等于质量提升，指出维护性问题（代码异味）占主导，严重性高的 Bug 虽少但风险大；并通过差分 SonarQube 分析量化了代理差异与代码变更量之间的关系。

**🔧 技术方法**

采用 SonarQube 进行差分静态分析，提取 Bug、代码异味、漏洞、热点等问题以及技术债务，并通过问题密度标准化比较。

**📊 数据集**

使用 AIDev 数据集中的 1,210 条 Python 语言的合并 Bug 修复 PR 作为实验样本。

**📈 对比分析**

通过将问题数归一化为每千行代码的密度，比较了 5 种代理（Codex、Copilot、Devin、Cursor、Claude）的表现，发现大部分差异可归因于 PR 体量，未发现显著的性能差异。

**⚠️ 局限性**

局限性包括仅针对 Python 和 AIDev 数据集、代理样本不均衡、缺乏人工基准、仅使用 SonarQube 的静态检测，未涵盖运行时或长期维护影响。

---

## 62. Just in time Informed Trees: Manipulability-Aware Asymptotically Optimized Motion Planning

**arXiv ID:** 2601.19972 | [PDF](https://arxiv.org/pdf/2601.19972v1)

**作者:** Kuanqi Cai `[一作]` (Istituto Italiano Di Tecnologia), Luis Figueredo `[通讯]` (University of Nottingham)

**通讯引用:** 715 | [OpenAlex ID](https://openalex.org/A5040790541)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Just-in-Time Informed Trees (JIT*) 算法，针对高维机器人路径规划，结合动态边扩展与采样以及运动性能模块，实现可扩展且考虑操纵性与自碰撞的优化路径规划。

**💡 创新点**

创新点：①引入“Just-in-Time”边与采样模块，按需扩展祖先节点并在瓶颈区增密采样；②在代价函数中嵌入操纵性权重（最小奇异值与 tanh 变换），动态平衡路径长度与操纵性能；③利用自碰撞危险场（SCDF）在采样过程中实时避碰；④通过自旋转空域与 null‑space 调整提升操纵性。

**🔧 技术方法**

技术：基于 RRT* 与 Informed RRT* 的树搜索框架；利用随机几何图（RGG）进行反向指导；采用自适应祖先策略、局部重连与批量采样；采用最小奇异值、SVD、伪逆及高斯扰动实现操纵性优化；使用自碰撞危险场公式与阈值过滤采样；对比实验使用 OMPL、ROS Noetic、MoveIt、PDT 等工具。

**📊 数据集**

实验数据集：仿真场景包括二维狭窄通道（NP）与随机矩形障碍（RR）在 ℝ⁴、ℝ⁸、ℝ¹⁶ 三个维度；真实机器人实验在单臂(ℝ⁷)与双臂(ℝ¹⁴)的三项任务——齿轮抓取与插入、变形物体预扣件、倒水——在不同机器人平台上进行（如六自由度机械臂与双臂系统）。

**📈 对比分析**

比较方法：与 RRT-Connect、RRT#、Informed RRT*、BIT*、ABIT*、AIT*、EIT* 等 SOTA 采样规划器进行 100 次仿真实验与 30 次真实机器人实验；评价指标包括成功率、初始解时间、初始解成本、最终解成本以及最小奇异值。结果显示：JIT* 在初始解时间和成本上显著优于其它方法，最终解成本平均降低 15-35%；操纵性指标（最小奇异值）提升 20-30%，成功率在单臂任务上提升至 93% 以上，双臂任务上提升至 70% 以上。

**⚠️ 局限性**

局限性：在极度拥挤或动态环境中，采样与自碰撞危险场的计算负担增大，导致实时性能下降；操纵性启发式对任务具有一定先验依赖，难以推广至需要力学控制或柔性操作的场景；当维度超过 16 时，样本数量与碰撞检测复杂度呈指数增长，算法的可扩展性受限。

---

## 63. GPU-Augmented OLAP Execution Engine: GPU Offloading

**arXiv ID:** 2601.19911 | [PDF](https://arxiv.org/pdf/2601.19911v1)

**作者:** Ilsun Chang `[一作]` `[通讯]`, Ilsun Chang

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

该论文提出了一个混合执行引擎，在 OLAP 查询中仅将 Top‑K 排序和键基连接探测阶段卸载到 GPU，以降低 CPU 的排序/连接成本。

**💡 创新点**

创新点在于结合 Key‑Only 传输、Late Materialization 与风险门控（Risky Gate）来精确判断何时使用 GPU，避免在小规模数据上误卸载。

**🔧 技术方法**

采用 GPU 编程（CUDA/PyTorch）、向量化执行模型、成本模型与传输测量，并在 PostgreSQL 16 环境下进行微基准测试。

**📊 数据集**

使用 PostgreSQL 16 中的大规模表（double+text 列）以及 RTX 4060 GPU，N 规模从几万到几百万。

**📈 对比分析**

与 CPU 完整排序、仅 CPU 的 Top‑K、始终 GPU 等策略比较，报告了中位数和 P95 尾延迟；在大 N 下 Key‑Only+Late Materialization 实现高达 12.9× 加速，风险门控显著降低尾延迟。

**⚠️ 局限性**

主要局限在于仅针对 Top‑K 与键基匹配两种原语，缺乏对更复杂聚合或多维 Join 的评估；门控阈值需手动调优，且对不同硬件/工作负载的迁移性未充分验证。

---

## 64. Modeling Next-Token Prediction as Left-Nested Intuitionistic Implication

**arXiv ID:** 2601.19915 | [PDF](https://arxiv.org/pdf/2601.19915v1)

**作者:** Paul Tarau `[一作]` (University of North Texas), Paul Tarau `[通讯]` (University of North Texas)

**通讯引用:** 4829 | [OpenAlex ID](https://openalex.org/A5042600106)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Arrow 语言模型，将下一词预测视为左嵌入直觉主义蕴含的证明扩展，并通过 Prolog 定理证明器验证其逻辑属性；

**💡 创新点**

创新点在于把 token 作为非交换的蕴含运算子，将顺序信息通过左嵌套蕴含链编码，直接对应 modus ponens 与 Curry‑Howard，提供了不依赖注意力的逻辑驱动生成框架；

**🔧 技术方法**

使用了 LJT 直觉主义蕴含定理证明器、低秩乘法 RNN 作为 token 运算子、Curry‑Howard 对应、流式损失、以及 Prolog 动态数据库检索等技术；

**📊 数据集**

主要在 Project Gutenberg 公共领域文本上进行实验，按句子拆分并生成所有子序列片段；

**📈 对比分析**

通过训练时间（约 5–10 分钟/书、RTX‑3090 24GB）、检索延迟（≈0.1–0.3 s/查询）与 Prolog 查询（≈0.5 s）与 Transformer/SSM 的对比，显示 Arrow 在小规模书籍级任务上能够快速训练且检索响应及时，性能与传统模型相当但无需位置编码；

**⚠️ 局限性**

局限在于仅支持精确的连续子序列检索，对噪声、错别字或非连续证据不鲁棒；缺乏在标准语言建模基准上的泛化评估；模型在很大范围内主要表现为记忆而非真正推理。

---

## 65. What is the AGI in Offensive Security ?

**arXiv ID:** 2601.19968 | [PDF](https://arxiv.org/pdf/2601.19968v1)

**作者:** Youngwoong Cho `[一作]` `[通讯]` (University of Sheffield), Youngwoong Cho (University of Sheffield)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

将攻击过程形式化为目标系统的状态机与攻击者的交互式符号代理，并证明任何攻击可归约为符号序列的操控

**💡 创新点**

创新点在于将攻击任务完整地归约为符号语言操作，并指出可利用大型语言模型近似或学习攻击者的可计算策略

**🔧 技术方法**

主要技术包括状态机建模、交互式图灵机框架、符号编码与注入函数、以及可计算策略的字符串化与模型学习

**📊 数据集**

该工作为理论论文，没有使用具体数据集；所有结论均基于数学证明与抽象模型

**📈 对比分析**

无实验比较与性能评估，论文未给出任何数值指标，仅给出理论上的可行性论证

**⚠️ 局限性**

局限性：仅适用于完全可数字化、可计算的系统，未覆盖硬件/物理攻击、非符号输入、以及不可判定性和大规模状态空间导致的实用难题

---

## 66. Pianoroll-Event: A Novel Score Representation for Symbolic Music

**arXiv ID:** 2601.19951 | [PDF](https://arxiv.org/pdf/2601.19951v1)

**作者:** Lekai Qian `[一作]` (South China University of Technology), Qi Liu `[通讯]` (South China University of Technology)

**通讯引用:** 28232 | [OpenAlex ID](https://openalex.org/A5100453264)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将钢琴卷轴（pianoroll）转换为四类事件序列的编码方案，兼顾稀疏压缩与结构保持。

**💡 创新点**

创新点在于设计四种互补事件（帧事件、间隙事件、模式事件、音乐结构事件），实现空间‑时间结构与高效离散序列的统一压缩。

**🔧 技术方法**

使用帧分割、块分区、稀疏压缩、事件映射等技术，将编码结果输入自回归模型（GPT‑2、Llama、LSTM）进行生成；评估采用 MusPy 指标、JS 相似度与 MOS。

**📊 数据集**

使用 MuseScore 数据集（约14万首双轨钢琴曲），时间分辨率为 1/16 拍。

**📈 对比分析**

与 REMI、MIDI 事件、ABC、Octuple 等主流编码方式对比；在编码效率（BDI）上提升 1.36–7.16 倍，在生成质量（PR、GC、SC、JS、MOS）上均获得最高分，尤其在 MOS 上提升 30–66%。

**⚠️ 局限性**

局限在于未包含力度等属性、对多轨或非钢琴乐曲的适用性待验证，且仅在 MuseScore 数据集上评估，需进一步验证在更大规模、多样化数据上的泛化能力。

---

## 67. Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers

**arXiv ID:** 2601.19967 | [PDF](https://arxiv.org/pdf/2601.19967v1)

**作者:** Jinlin Liu `[一作]` (Huazhong University of Science and Technology), Xiaojin Zhang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 219 | [OpenAlex ID](https://openalex.org/A5101527293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种只使用线性模型即可生成不可学习样本的PIL方法，能够在训练时阻止深度网络学习有意义的特征；

**💡 创新点**

核心创新是通过诱导线性化（让网络行为更像线性分类器）来实现不可学习样本，从而大幅降低计算成本；

**🔧 技术方法**

利用无偏置的线性分类器作为代理模型，并通过联合KL和交叉熵损失在PGD框架下优化扰动；

**📊 数据集**

在四大图像分类基准上验证：SVHN、CIFAR‑10、CIFAR‑100、ImageNet‑100子集；

**📈 对比分析**

与多种基线（EM、REM、TAP、NTGA、SEP、SP、AR、CUDA等）对比，PIL在保持几乎相同或更低的生成时间（约40 s）下，在多种模型（ResNet、VGG、DenseNet、MobileNet等）和多种对抗手段（数据增强、JPEG压缩、对抗训练）中均能将清洁测试准确率压至≈10%或更低；

**⚠️ 局限性**

局限性：当只对部分样本加扰时，模型仍可保持高准确率；不可学习样本对清洁样本的梯度几乎正交，难以抑制对清洁数据的学习；

---

## 68. Quick Change Detection in Discrete-Time in Presence of a Covert Adversary

**arXiv ID:** 2601.20022 | [PDF](https://arxiv.org/pdf/2601.20022v1)

**作者:** Amir Reza Ramtin `[一作]` (University of Massachusetts), Don Towsley `[通讯]` (University of Massachusetts)

**通讯引用:** 57275 | [OpenAlex ID](https://openalex.org/A5036683370)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了Cu Sum在后变化分布随时间变化的情况下的行为，并分析了其渐进性能。

**💡 创新点**

提出了对Cu Sum在时间变化后变化分布下的渐进分析框架，并界定了其良好与不良工作区间。

**🔧 技术方法**

采用了KL散度、极限理论、概率论与统计方法对Cu Sum进行渐进性能分析。

**📊 数据集**

使用了高斯和指数分布的理论模型作为实验数据集。

**📈 对比分析**

与固定后变化分布的经典Cu Sum进行比较，表明在特定收敛速率下仍保持渐进最优性，若不满足则性能退化。

**⚠️ 局限性**

仅提供渐进结果，未给出有限样本下的具体性能；对非平稳分布变化的情况缺乏细致讨论。

---

## 69. NCSAM Noise-Compensated Sharpness-Aware Minimization for Noisy Label Learning

**arXiv ID:** 2601.19947 | [PDF](https://arxiv.org/pdf/2601.19947v1)

**作者:** Jiayu Xu `[一作]` (Beijing University Of Technology), Junbiao Pang `[通讯]` (Beijing University Of Technology)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5020694008)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Noise-Compensated Sharpness-Aware Minimization (NCSAM)，在存在标签噪声的情况下通过模拟标签噪声并加入动态校正项，恢复SAM的平滑搜索能力，提升模型鲁棒性。

**💡 创新点**

创新点在于：①首次从PAC‑Bayes视角理论阐明噪声标签会导致SAM扰动方向与幅度失真；②设计了基于渐进标签翻转的噪声梯度模拟与动态缩放校正机制，直接对抗噪声梯度偏差；③无需样本筛选或标签纠正，仍能显著抑制记忆效应并保持SAM的平坦性追踪。

**🔧 技术方法**

使用技术包括：PAC‑Bayes通用泛化分析、Sharpness‑Aware Minimization (SAM) 的改进、梯度模拟与校正、渐进标签翻转、动态缩放系数 s(t)、基于 logit gap 的样本选择与噪声模拟。

**📊 数据集**

实验数据集涵盖合成噪声的 CIFAR‑10/100、Tiny‑ImageNet 以及真实噪声的 Food‑101N、Animal‑10N、Clothing1M。

**📈 对比分析**

与 SGD、SAM、BSAM 等传统优化器及多种噪声鲁棒方法（如 Co‑Teaching、DivideMix、CAL 等）比较，NCSAM 在中高噪声率下普遍提升 3%~20% 的准确率，特别是在 CIFAR‑100、Tiny‑ImageNet 和 Clothing1M 上表现突出。

**⚠️ 局限性**

局限性包括：①对噪声梯度模拟的依赖，需要合适的翻转比例和动态缩放超参；②早期训练阶段噪声估计不稳定，可能导致校正失效；③在极端噪声或多源噪声混合场景下，校正项可能引入过度偏差或缺乏泛化。

---

## 70. GTAC: A Generative Transformer for Approximate Circuits

**arXiv ID:** 2601.19906 | [PDF](https://arxiv.org/pdf/2601.19906v1)

**作者:** Jingxin Wang `[一作]` (Shanghai Jiao Tong University), Weikang Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2551 | [OpenAlex ID](https://openalex.org/A5036572182)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于 Transformer 的近似电路生成模型（Approximate Circuit Transformer, ACT），能够在满足用户设定误差阈值的前提下，直接生成高效的近似逻辑电路。

**💡 创新点**

创新点包括：①将误差容忍掩码机制嵌入生成流程，实现误差约束下的自由探索；②结合监督预训练与强化学习的混合训练策略；③使用 MCTS 与 Pareto 优化实现多目标搜索，突破传统后置重写的局限。

**🔧 技术方法**

所采用技术包括 Transformer 编码-解码架构、三值逻辑误差容忍掩码、近似逻辑检查模块、强化学习（策略梯度）与蒙特卡洛树搜索（MCTS）以及 Pareto 前沿分析。

**📊 数据集**

实验使用的主要数据集为：40M 条精确电路与 200K 条误差为 1%、5%、10% 的近似电路对（由 ABC Resyn2 与 ALSRAC 生成），以及 IWLS 2023 的 1110 条基准电路。

**📈 对比分析**

与 Circuit Transformer、HEDALS、AppResub、ALSRAC 等基线比较，ACT 在延迟、面积和门数上分别提升 47.9%、73.1% 和 69.3%，并且在运行时比前者快 2.8–4.3 倍；在 IWLS 基准上也取得了更优的 Pareto 前沿，证明了其在多目标优化中的优势。

**⚠️ 局限性**

局限性主要体现在：对更大规模电路的可扩展性尚未充分验证；误差估计依赖采样，可能导致误差不精确；目前仅支持单一误差阈值的约束，未实现多目标协同优化。

---

## 71. Automated Benchmark Generation from Domain Guidelines Informed by Bloom's Taxonomy

**arXiv ID:** 2601.20253 | [PDF](https://arxiv.org/pdf/2601.20253v1)

**作者:** Si Chen `[一作]` (University of Notre Dame), Nitesh V. Chawla `[通讯]` (University of Notre Dame)

**通讯引用:** 57921 | [OpenAlex ID](https://openalex.org/A5068157871)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BloomQA 框架，利用 LLM 自动从专业指南提取最佳实践，生成违规情景、Bloom 级别多选题和对话，构建面向教学、饮食和照护三大实践领域的可扩展评测集。

**💡 创新点**

创新点在于：① 将 Bloom 认知层级嵌入自动生成的多选题与对话；② 采用违规情景驱动的隐式知识检验；③ 引入心理测量学（难度、区分度、可靠性）评估指标，实现在无考试题库的领域中构建高质量、可重复的评测。

**🔧 技术方法**

技术主要包括：LLM（GPT‑4o‑mini/Claude Sonnet 4）进行实践提取、情景生成和 MCQ 设计；规则+LLM 质量过滤；Bloom 级别扩展规则；GLMM 与离散化的心理测量分析；以及基于对话的 fine‑tune 评估。

**📊 数据集**

数据集：Teach‑QA（19,824 题）、Diet‑QA（18,756 题）和 CareGiving‑QA（20,000 题），每个集包含约 5,000 个违规情景、20,000 个 Bloom 级别 MCQ 及 5,000 条多轮对话。

**📈 对比分析**

比较方法：用 GLMM 估计模型在不同 Bloom 级别上的准确率、难度与区分度；对 6 款 LLM（DeepSeek‑V3、GPT‑4o、GPT‑4o‑mini、Kimi‑K2、Llama‑33‑70B、Mixtral‑8×7B 等）进行同一题集评测；fine‑tuned 版本在对话数据上提升显著，尤其在 Apply/Analyze 级别。模型表现显示：高阶分析能力相对较强，但在 Remember 级别易失误。

**⚠️ 局限性**

局限性：① 受限于仅 6 款模型且响应相关，难以应用完整 IRT；② 生成情景与对话可能存在细微泄漏或不自然表达；③ 评测覆盖的领域有限，其他实践领域仍需手工验证；④ Bloom 级别与 LLM 认知层级对齐不一致，需进一步校准。

---

## 72. Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions

**arXiv ID:** 2601.19965 | [PDF](https://arxiv.org/pdf/2601.19965v1)

**作者:** Mingxuan Luo `[一作]` (Xiamen University), Chen Lin `[通讯]` (Xiamen University)

**通讯引用:** 17204 | [OpenAlex ID](https://openalex.org/A5100443683)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多阶段级联延迟反馈的净转化率（NetCVR）预测，构建了首个公开的淘宝App级联点击→转化→退款数据集，并提出了在线持续学习的级联模型neTCAS。

**💡 创新点**

创新点包括：①首次公开完整链路的连续NetCVR数据集；②基于CVR‑RFR级联、两阶段重要性去偏和延迟感知排序损失的在线学习框架；③利用延迟时间特征提升去偏与排序精度。

**🔧 技术方法**

使用了共享‑私有网络结构、两阶段重要性加权去偏、延迟感知对比损失、在线流式训练以及多窗口观测技术。

**📊 数据集**

使用了来自淘宝App的CAscadal Sequences of Conversion And Delayed rEfund（CAscadal）大规模数据集。

**📈 对比分析**

与预训练、oracle、每日更新的离线模型以及八个主流延迟反馈基线比较，neTCAS在NetCVR的RI‑AUC提升12.41%，RI‑PRAUC提升14.94%，在CVR上也表现出显著提升。

**⚠️ 局限性**

仍存在长期延迟退款样本稀缺、模型对不同业务场景的泛化能力以及实时计算资源消耗等局限。

---

## 73. Real-Time Robot Execution with Masked Action Chunking

**arXiv ID:** 2601.20130 | [PDF](https://arxiv.org/pdf/2601.20130v1)

**作者:** Haoxuan Wang `[一作]` (University of Illinois Chicago), Gaowen Liu `[通讯]` (Cisco Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了REMAC方法，用于解决机器人异步推理中的交错块不连续性和块内不一致性问题，提升实时执行性能。

**💡 创新点**

创新点在于通过掩码动作块学习纠正策略和前缀保留采样，训练时就对延迟做条件化，避免额外推理延迟。

**🔧 技术方法**

结合流匹配（flow‑matching）与LoRA参数高效微调、掩码策略、Self‑conditioned curriculum和残差对齐等技术。

**📊 数据集**

在Kinetix仿真环境（12个动态任务）以及真实世界的Franka 3手臂抓取-放置任务上进行评估。

**📈 对比分析**

与Naive Async、Bidirectional Decoding、RTC等基线比较，REMAC在所有延迟设置下均取得更高成功率、更快完成时间，并且可与测试时方法进一步提升。

**⚠️ 局限性**

局限在于对极大延迟的鲁棒性仍有限，且在复杂多臂或高频控制场景下需进一步验证。

---

## 74. Game-Theoretic Autonomous Driving: A Graphs of Convex Sets Approach

**arXiv ID:** 2601.20054 | [PDF](https://arxiv.org/pdf/2601.20054v1)

**作者:** Nikolaj Käfer `[一作]` (ETH Zurich), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 560 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种基于图形凸集（GCS）的迭代最优响应（IBR）框架，用来求解多车辆高速公路自驾中策略性交互下的最优轨迹与车道变换；

**💡 创新点**

创新点在于将每辆车的离散车道选择与连续动力学规划统一编码为车辆特定、策略依赖的GCS，利用GCS的凸松弛实现全局最优的混合整数问题，并结合泛化潜在博弈理论证明IBR的收敛性，给出近似纳什均衡误差界；

**🔧 技术方法**

主要技术包括图形凸集（GCS）、迭代最优响应（IBR）方法、泛化潜在博弈理论、凸松弛与混合整数约束求解、MOSEK优化器以及仿真随机实验；

**📊 数据集**

使用的是仿真数据，包含6车4车道、30步（Δt=0.3）的基准场景，以及100组随机参数（N=4，|L|=3，T=30）生成的车辆轨迹；

**📈 对比分析**

与传统MIP或模型预测控制无直接对比，评估方式是潜在函数下降与最终轨迹安全性；实验表明IBR在两轮迭代即可收敛，求解时间约297秒，产生安全且策略一致的多车道变道与超车行为；

**⚠️ 局限性**

局限性包括：需要假设GCS松弛紧致，缺乏理论上紧致保证；对更新顺序和初始点敏感；图构建与更新成本高，未实现缓存/热启动；未提供全局最优性保证与其他方法的客观性能比较。

---

## 75. From Intuition to Expertise: Rubric-Based Cognitive Calibration for Human Detection of LLM-Generated Korean Text

**arXiv ID:** 2601.19913 | [PDF](https://arxiv.org/pdf/2601.19913v1)

**作者:** Shinwoo Park `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5077698683)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过三阶段纵向实验，研究了在韩语环境下利用专业注释者与结构化评分准则提升对LLM生成文本的判别能力。

**💡 创新点**

将国家韩语写作标准改编为微观诊断量表，证明人类专业判别可通过训练和准则得到显著提升。

**🔧 技术方法**

采用结构化评分准则、交互式反馈、交叉验证以及LLM基线检测器（GPT‑5.2 Thinking、Gemini 3 Flash、Claude Sonnet 4.5）进行比较。

**📊 数据集**

使用KatFish韩语论证论文基准（30篇，包含人类与四种LLM生成文本）。

**📈 对比分析**

在三阶段实验中，人类多投票准确率从60%提升至100%，并且与LLM检测器的Fleiss κ从‑0.09提升至0.82，表现出显著提升。

**⚠️ 局限性**

受限于样本量小、注释者仅三名且仅在论证论文上实验，且量表设计后期化可能导致过拟合。

---

## 76. STELLAR: Structure-guided LLM Assertion Retrieval and Generation for Formal Verification

**arXiv ID:** 2601.19903 | [PDF](https://arxiv.org/pdf/2601.19903v1)

**作者:** Saeid Rajabi `[一作]` (University of Delaware), Satwik Patnaik `[通讯]` (University of Delaware)

**通讯引用:** 1066 | [OpenAlex ID](https://openalex.org/A5005805069)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建结构化指纹检索库并结合LLM生成语句，实现了基于结构的SVA自动生成。

**💡 创新点**

创新点在于用AST结构指纹进行检索，避免变量命名影响；并在提示中加入执行路径计数，保证覆盖。

**🔧 技术方法**

采用AST结构指纹、Sentence Transformer、FAISS近似检索、结构化提示与LLM（GPT‑4、Llama‑3、CodeLLama）集成的检索增强生成框架。

**📊 数据集**

以VErilog Assertion TEmplate Repository (VERT) 的约1万对RTL–SVA样本构建知识库，并在同一数据集的分层子集上进行测试。

**📈 对比分析**

与零射击、基于语义检索的Baseline以及行业级大模型对比，STELLAR 在语法正确率、BLEU、BERTScore、执行路径覆盖等指标上提升 15–35% 甚至可与 70B 参数模型匹配。

**⚠️ 局限性**

局限包括依赖已清洗的高质量SVA样本；对极大模块的检索与提示仍需改进；检索覆盖率与指纹设计相关，若硬件设计结构与已知库差异大则效果受限。

---

## 77. IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation

**arXiv ID:** 2601.20084 | [PDF](https://arxiv.org/pdf/2601.20084v1)

**作者:** Yash Saxena `[一作]` (University of Maryland), Manas Gaur `[通讯]` (University of Maryland)

**通讯引用:** 1453 | [OpenAlex ID](https://openalex.org/A5023667301)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了可解释的双向调制适配器IMRNNs，在保持基础检索器冻结的前提下，动态调整查询和文档嵌入，提升检索性能并提供结构、归因与语义层面的可解释性。

**💡 创新点**

创新点在于同时提供三层可解释性（结构、归因、语义）并通过轻量化的两层MLP实现查询-文档双向动态调制，使检索过程既高效又可解释。

**🔧 技术方法**

使用冻结的深度编码器（e5、MiniLM、BGE）+ 投影矩阵 + 两个MLP适配器（查询适配器与文档适配器），并利用 Moore–Penrose pseudoinverse 将调制向量映射回词嵌入空间实现归因。

**📊 数据集**

在BEIR基准的七个数据集（MS MARCO、Natural Questions、HotpotQA、SciFact、TREC-COVID、FiQA-2018、Webis‑Touche2020）以及多种基础检索器上进行实验。

**📈 对比分析**

与DIME、Search‑Adaptor、HypEncoder等主流适配器对比，IMRNNs平均提升nDCG +6.35%、召回 +7.14%、MRR +7.04%，在多领域数据集上持续保持领先。

**⚠️ 局限性**

局限性包括：基于伪逆的词级归因可能产生噪声；文档适配器需逐文档处理，线性时间复杂度限制极大语料库的部署；对低质量基础嵌入的改进有限，需先有良好语义表征。

---

## 78. PILOT: Planning via Internalized Latent Optimization Trajectories for Large Language Models

**arXiv ID:** 2601.19917 | [PDF](https://arxiv.org/pdf/2601.19917v1)

**作者:** Haoyu Zheng `[一作]` (Zhejiang University), Jun Xiao `[通讯]` (Zhejiang University)

**通讯引用:** 9222 | [OpenAlex ID](https://openalex.org/A5101485989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PILOT框架，通过轻量级Hyper‑Network生成查询特定的潜在锚点并注入LLM内部，以在不修改主干权重的前提下稳定多步推理轨迹。

**💡 创新点**

创新点在于将动态潜在引导与Energy‑Aligned Injection结合，既能在单向推理中实时调节推理路径，又保持了模型的通用性和低延迟。

**🔧 技术方法**

使用的技术包括Hyper‑Network、FiLM调制、Proto‑Thought Prior、Energy‑Aligned Injection、Construct‑and‑Verify 数据构造、对齐损失和两阶段训练策略。

**📊 数据集**

采用的训练与评估数据集为数学领域的 MATH500、AIMO、GSM8K 以及代码生成领域的 HumanEval、MBPP，并通过 Construct‑and‑Verify 过滤得到高质量训练样本。

**📈 对比分析**

在与 Zero‑shot CoT、LoRA、Soft CoT、ReFT、CAA、Pause Token、Coconut 等基线相同模型规模下比较，PILOT 在 MATH500 上提升约 8.9%，在代码生成任务上提升 5–7%，且推理延迟仅增加 0.2%。

**⚠️ 局限性**

主要局限在于需要耗时的 Construct‑and‑Verify 预处理来生成高质量锚点，以及锚点注入层和能量参数需域特定手动调参，缺乏完全自适应机制。

---

## 79. Teaching LLMs to Ask: Self-Querying Category-Theoretic Planning for Under-Specified Reasoning

**arXiv ID:** 2601.20014 | [PDF](https://arxiv.org/pdf/2601.20014v1)

**作者:** Shuhui Qu `[一作]` (Stanford University), Shuhui Qu `[通讯]` (Stanford University)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5112694715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Self-Querying Bidirectional Categorical Planning（SQ‑BCP）框架，用于在LLM推理时解决部分可观测条件下的规划失效问题。

**💡 创新点**

创新点在于将预条件不确定性显式标记为“//”，通过局部自问（targeted self‑query）或桥接动作（bridging）主动解决未知，并在双向搜索中加入类别化pullback验证以确保目标兼容性。

**🔧 技术方法**

主要技术包括：预条件标签化与自查询/桥接的确定性细化策略、双向搜索整合、pullback基验证与硬约束检测，以及基于任务的距离函数用于搜索排序/剪枝。

**📊 数据集**

使用了WikiHow与RecipeNLG两大开放式任务数据集，并在每个实例上采用k‑reveal协议隐藏部分预条件以模拟部分可观测环境。

**📈 对比分析**

与直接提示、链式推理（CoT）、ToT、ReAct、Self‑Ask等基线进行对比，SQ‑BCP在WikiHow上将资源违规率降至14.9%（相较Self‑Ask的26.0%）并保持ROUGE‑1/2相当；在RecipeNLG上违规率降至5.8%（相较15.7%）且BLEU保持竞争力。

**⚠️ 局限性**

局限性包括对oracle回答质量的依赖、对LLM预条件标注准确性的敏感、仅支持离散硬约束、可能在连续或不确定环境下效果不足，以及推理时额外的计算开销。

---

## 80. Primitive-Driven Acceleration of Hyperdimensional Computing for Real-Time Image Classification

**arXiv ID:** 2601.20061 | [PDF](https://arxiv.org/pdf/2601.20061v1)

**作者:** Dhruv Parikh `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**通讯引用:** 17489 | [OpenAlex ID](https://openalex.org/A5033166029)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于补丁的超维度计算(HDC)图像编码方法，并在FPGA上实现了端到端的原语驱动加速器，用于实时图像分类。

**💡 创新点**

创新点包括：①将CNN的感受野思想与HDC绑定、置换、聚束原语相结合，形成空间感知的补丁编码；②对硬件与算法进行协同设计，利用补丁并行与维度并行的深度流水线架构，在FPGA上实现低延迟、低功耗的加速器。

**🔧 技术方法**

使用技术包括：高维二值向量（超维向量）编码、绑定/置换/聚束原语、在线相似度引导学习、FPGA HLS实现的多补丁处理器阵列、全局加法树与相似度引擎的流水线化。

**📊 数据集**

在MNIST和Fashion‑MNIST两个标准手写数字与服饰图像数据集上进行评估。

**📈 对比分析**

与CPU（EPYC 7313）和GPU（RTX 6000 Ada）基准相比，FPGA实现单图像推理延迟仅为0.09 ms，准确率达到MNIST 95.67%、Fashion‑MNIST 85.14%，相较CPU提升约1300×、相较GPU提升约60×；吞吐量也优于GPU在小批量场景下的表现。

**⚠️ 局限性**

局限性包括：适用范围主要在小尺寸灰度图像，对更大分辨率或彩色图像的可扩展性尚未验证；补丁尺寸与超维度大小需手工调参；以及FPGA实现对硬件资源（DSP、BRAM、HBM）依赖较强。

---

## 81. DecHW: Heterogeneous Decentralized Federated Learning Exploiting Second-Order Information

**arXiv ID:** 2601.19938 | [PDF](https://arxiv.org/pdf/2601.19938v1)

**作者:** Adnan Ahmad `[一作]` (Deakin University), Marco Conti `[通讯]` (Italian National Research Council)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在去中心化联邦学习中通过第二阶信息实现参数级加权聚合的新方法——Decentralized Hessian-Weighted Aggregation (DecHW)。

**💡 创新点**

创新点在于利用每个本地模型的 Hessian 对角线来衡量参数的重要性，按参数级别动态加权聚合，并通过累积 Hessian 信息平滑梯度变化；该方法无需中心服务器或共享数据即可缓解数据和模型初始化异质性。

**🔧 技术方法**

技术手段包括：第二阶信息（Hessian）、参数级加权平均、邻居图通信、Dirichlet 采样的非 IID 数据分配、卷积神经网络与 SGD 训练、以及对比实验。

**📊 数据集**

使用的公开数据集为 MNIST、Fashion‑MNIST 与 CIFAR‑10。

**📈 对比分析**

与 DecHetero、CFA、DecDiff、DecDiff(VT)、DESA 等基线方法进行对比；在不同异质度（α = 0.2, 0.5, 1）下，DecHW 在测试准确率上始终领先，并在收敛速度上显著优于其他方法（通信轮次大幅减少）。

**⚠️ 局限性**

局限性包括：需额外传输 Hessian 对角线导致通信成本翻倍；依赖邻居图中节点之间的完整信息交换；在极端异质或网络动态变化的场景下效果待进一步验证。

---

## 82. DiSa: Saliency-Aware Foreground-Background Disentangled Framework for Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2601.20064 | [PDF](https://arxiv.org/pdf/2601.20064v1)

**作者:** Zhen Yao `[一作]` (Lehigh University), Mooi Choo Chuah `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DiSa框架，用于开源词汇语义分割，通过显著性感知的前景-背景分离与分层细化实现更精细的像素级预测。

**💡 创新点**

创新点包括：①基于Grad‑CAM的显著性引导的前景-背景拆解模块（SDM），实现自适应类别级分离；②分层细化模块（HRM），在像素、类别和语义层面分别进行特征细化；③两者结合在一次前向过程中显著提升前景与背景的语义一致性与边界定位。

**🔧 技术方法**

技术手段主要包括：CLIP视觉‑语言预训练模型、交叉注意力、ITM损失、Grad‑CAM显著性提取、双分支前景/背景网络、Swin Transformer块、MLP特征聚合、像素/类别/语义级别的全局池化与重加权。

**📊 数据集**

使用COCO‑Stuff作为训练集，评估于六大公开语义分割基准：ADE20K（ADE‑150、ADE‑847）、PASCAL‑VOC（PAS‑20、PAS‑20b）、PASCAL‑Context（PC‑59、PC‑459）。

**📈 对比分析**

与当前SOTA（如CAT‑Seg、DPSeg、OVSeg等）对比，DiSa在所有六个数据集上均实现了1–2.6个百分点的mIoU提升，特别在包含背景类别的PAS‑20b上提升显著，验证了显著性拆解与分层细化的有效性。

**⚠️ 局限性**

局限性包括：①仍依赖大规模VLM预训练与较大的模型参数，推理开销相对较高；②显著性提取依赖ITM梯度，可能对极端遮挡或极小物体的识别产生不稳定；③对完全新颖的语义场景或跨域数据的泛化性尚未系统评估。

---

## 83. Analysis of LLM Vulnerability to GPU Soft Errors: An Instruction-Level Fault Injection Study

**arXiv ID:** 2601.19912 | [PDF](https://arxiv.org/pdf/2601.19912v1)

**作者:** Duo Chai `[一作]` (Beijing University of Posts and Telecommunications), Shangguang Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 11791 | [OpenAlex ID](https://openalex.org/A5054814598)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对GPU上LLM推理的软错误进行了指令级故障注入实验，系统地评估了错误导致的SDC、DUE和模型可靠性。

**💡 创新点**

创新点在于：①首次在LLM推理中使用指令级注入，揭示了错误来源；②提出基于指令易失性因子（IVF）的整体模型易失性因子（MVF）近似方法；③从任务难度、位位置、操作符和层级等五个维度进行多维度易失性分析。

**🔧 技术方法**

技术主要包括NVBitFI指令级故障注入框架、GPU A100硬件平台、CUDA 12.2 + Python 3.12 运行环境，以及自定义的LLM可靠性评估流程。

**📊 数据集**

使用的评测数据集包括 Lambada、PIQA、HellaSwag、WikiText-2、XSum、GSM8K，覆盖推理、生成、推理和数学推理等多种任务。

**📈 对比分析**

通过对比不同模型（GPT2、Llama3.2、Qwen3）在多任务、多位宽、不同规模下的误差率、准确率下降曲线，展示了任务难度和模型规模对鲁棒性的影响；实验表明较大模型并不一定更鲁棒，且高位位误差对SDC影响最大。

**⚠️ 局限性**

局限性包括：①仅在单一A100 GPU上测试，缺乏跨架构验证；②注入过程耗时高（约800–3000 GPU小时）；③未考虑KV缓存错误和真实硬件误差分布，结果可能与实际使用场景略有偏差。

---

## 84. Distributional value gradients for stochastic environments

**arXiv ID:** 2601.20071 | [PDF](https://arxiv.org/pdf/2601.20071v1)

**作者:** Baptiste Debes `[一作]` (KU Leuven), Tinne Tuytelaars `[通讯]` (KU Leuven)

**通讯引用:** 53425 | [OpenAlex ID](https://openalex.org/A5074816094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种分布式Sobolev强化学习框架（Distributional Sobolev Deterministic Policy Gradient, DSDPG），利用可微分世界模型和生成式分布式评估器联合学习回报及其梯度分布，并通过最大切片MMD（MSMMD）实现可计算的分布度量。

**💡 创新点**

创新点在于：①首次将梯度信息与分布式价值学习结合，形成Sobolev Bellman算子；②给出该算子在Wasserstein和MSMMD度量下的收缩性证明；③通过条件变分自编码器实现可微分的随机一阶世界模型，实现了在不易微分环境中的梯度回溯。

**🔧 技术方法**

核心技术包括：Sobolev训练（梯度正则化）、条件VAE世界模型、最大切片MMD度量、分布式TD回归、双重估计校正（TQC）以及基于MMD的分布式更新。

**📊 数据集**

实验数据集主要为：①自定义2D点质量多目标任务；②Gymnasium的MuJoCo环境（Ant-v2、Humanoid-v2、Walker2d-v2、InvertedDoublePendulum-v2等），并在环境中加入乘法观测噪声与高斯动力噪声进行鲁棒性测试。

**📈 对比分析**

与TD3、MAGE、IQN、标准MMD等基线进行比较，结果表明：在无噪声、乘法噪声和高斯噪声三种设置下，DSDPG（MSMMD Sobolev 与 MMD Sobolev）在多数任务中表现与基线持平或优于基线，尤其在噪声环境下显示出更高的稳定性和更好的收敛速度；在Ant-v2和Humanoid-v2等高维任务中优势更为明显。

**⚠️ 局限性**

主要局限包括：①计算开销较大，需要在每一步产生大量样本并求梯度；②目前只实现了动作梯度版本的Sobolev Bellman，完整的状态+动作梯度版本仍存在计算复杂度与实现难度；③对世界模型的依赖性尚未充分探索，可能受限于cVAE或流模型的表达能力。

---

## 85. Me-Agent: A Personalized Mobile Agent with Two-Level User Habit Learning for Enhanced Interaction

**arXiv ID:** 2601.20162 | [PDF](https://arxiv.org/pdf/2601.20162v1)

**作者:** Shuoxin Wang `[一作]` (Yunnan University), Yu Tian `[通讯]` (Tsinghua University)

**通讯引用:** 10921 | [OpenAlex ID](https://openalex.org/A5015080274)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Me-Agent的移动代理，通过两层习惯建模实现个性化执行，无需模型参数更新，并在新的User FingerTip基准上验证其对模糊指令的理解与执行能力。

**💡 创新点**

创新点在于引入无训练的用户偏好学习模块（UPL）与分层偏好记忆模块（HPM），使代理能够在Prompt层通过奖励模型对偏好进行排序，在记忆层以分层结构存储全局与应用专属经验，从而实现高效、可持续的个性化推理。

**🔧 技术方法**

技术手段包括：VLM奖励模型评估执行轨迹，LLM抽象与对比经验，基于Embedding的语义检索与层级记忆检索，以及将经验动态注入Prompt的策略。

**📊 数据集**

使用的数据集包括新构建的User FingerTip benchmark（包含300条Type I和267条Type II模糊指令，覆盖60名用户）以及公开的E dataset，用于评估通用任务完成能力。

**📈 对比分析**

与Mobile-Agent-v2和Mobile-Agent-E在User FingerTip和E dataset上进行对比，Me-Agent在个性化指标（如应用选择准确率ASA 1.0、BERTScore提升）以及任务完成率、成功率、动作精度等多项指标均显著优于基线，任务完成率提升至89.3%。

**⚠️ 局限性**

局限性在于：存储的UI位置信息易因应用更新或弹窗出现而失效；个性化策略仅基于历史行为，未考虑用户当前位置、情绪等动态上下文因素。

---

## 86. "Newspaper Eat" Means "Not Tasty": A Taxonomy and Benchmark for Coded Languages in Real-World Chinese Online Reviews

**arXiv ID:** 2601.19932 | [PDF](https://arxiv.org/pdf/2601.19932v1)

**作者:** Ruyuan Wan `[一作]` (Pennsylvania State University), Ting-Hao 'Kenneth' Huang `[通讯]` (Pennsylvania State University)

**通讯引用:** 1112 | [OpenAlex ID](https://openalex.org/A5083675499)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了包含 7,744 条中文 Google Maps 评价的语料库，其中 900 条包含编码语言，并提出七类编码语言分类体系；对语言模型在编码语言检测、分类和评价预测三个任务上进行基准测试；同时对编码语言的语音相似度进行量化分析。

**💡 创新点**

首次公开的真实世界编码语言数据集及其七类细粒度分类体系；系统评估大型语言模型在处理编码语言时的不足；将语音相似度（CER）与模型预测误差关联，揭示语音扭曲对下游任务影响。

**🔧 技术方法**

使用人机迭代抽取的 bootstrapping 方法构建词典并扩展数据集；利用多标签分类、二分类、回归等下游任务框架评估模型；通过 Pinyin/IPA 计算字符错误率（CER）评估语音相似度；对不同模型（GPT‑5‑mini、Gemini‑2.5‑Flash、Qwen2.5‑7B‑Instruct、DeepSeek‑V3.2）进行系统对比。

**📊 数据集**

核心数据集为 7,744 条 Google Maps 评价（含 5,391 条餐厅评语和 112,521 条地方评语），其中 900 条编码；同时使用了公开的 1.77 万条餐厅评语与 6.66 亿条地方评语作为原始文本来源。

**📈 对比分析**

在编码语言检测任务中，DeepSeek‑V3.2 取得最高 F1 分数；在分类任务中，Cipher 表现最佳，而 Cross‑Lingual Phonetic Encoding 召回最高、精度最低；在评价预测任务中，GPT‑5‑mini 的 MSE 最低，且编码文本导致 MSE 明显上升，显示编码语言干扰情感信号。

**⚠️ 局限性**

构建过程受现有词典和 LLM 识别能力限制，可能漏检罕见或创新编码；数据仅涵盖 2022 年之前的美国公开评价，时间与地域有限；标注受注释者语言背景影响；分类体系不一定覆盖所有可能的编码方式，且未捕捉持续演化的新型编码。

---

## 87. Benchmarking LLAMA Model Security Against OWASP Top 10 For LLM Applications

**arXiv ID:** 2601.19970 | [PDF](https://arxiv.org/pdf/2601.19970v1)

**作者:** Nourin Shahin `[一作]` (Texas A&M University San Antonio), Izzat Alsmadi `[通讯]` (Texas A&M University San Antonio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了10种Llama模型（5个标准模型和5个Guard变体）在OWASP LLM Top 10框架下的安全检测能力，并提供了开源的100条攻击提示基准数据集。

**💡 创新点**

创新点在于将OWASP LLM安全框架与专门的Guard模型相结合，系统对比并发现小型专用模型在检测率、延迟和显存使用上优于大型通用模型，突破了传统“模型越大越安全”的假设。

**🔧 技术方法**

采用了PyTorch + HuggingFace Transformers 在 NVIDIA A30 GPU 上运行 float16/INT8 模型，低温度(0.1)、短 token 生成，并通过解析输出中的 “safe/unsafe” 关键词实现安全标签判断。

**📊 数据集**

使用自构建的 JSON 数据集，包含 100 条均衡覆盖 10 类 OWASP 漏洞的对抗性提示，每条记录包括攻击者指令、触发词、恶意意图、类别、子类别、标签及元数据。

**📈 对比分析**

通过对每个模型的检测率、平均推理延迟和 VRAM 占用进行量化比较，发现 Llama-Guard-3-1B 以 76% 的检测率、0.165 s 的平均延迟和 0.94 GB 的显存使用，成为性能最优的模型；相反，基础模型 Llama‑3.1‑8B 在所有攻击中检测率为 0%。

**⚠️ 局限性**

主要限制包括对系统提示泄露（LLM07）和供应链攻击（LLM03）的检测效果仍然很差；量化版本和多模态模型在安全检测上表现不佳，且仅针对文本场景，未验证在更复杂工作负载中的泛化能力。

---

## 88. OPT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling

**arXiv ID:** 2601.19924 | [PDF](https://arxiv.org/pdf/2601.19924v1)

**作者:** Yitian Chen `[一作]` (Cardinal Operations), Dongdong Ge `[通讯]` (Antai School of Economics and Management)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可扩展的OPT-Engine基准框架，用于评估LLM在自动建模和求解线性/混合整数规划任务的能力。

**💡 创新点**

创新点在于结合可控难度级别、程序化实例生成与自然语言增强，系统比较工具集成推理与纯文本推理，揭示约束建模是LLM的主要瓶颈。

**🔧 技术方法**

使用LLM（如DeepSeek-V3.2、GPT-5.1、Qwen3-4B）与外部求解器（Gurobi、COPT）相结合的工具集成推理，以及链式思考纯文本推理。

**📊 数据集**

使用自研的OPT-Engine生成的10类线性和混合整数规划实例（共5 LP + 5 MIP）作为测试数据集。

**📈 对比分析**

通过在不同难度等级下对比PTR和TIR方法，发现TIR在规模增大时准确率保持高（>80%），而PTR随复杂度急剧下降；在小规模时PTR略优。

**⚠️ 局限性**

局限在于仅评估精确最优解，对近似/启发式解无评估；覆盖范围仅限线性/混合整数规划，未包含非线性、随机或动态规划。

---

## 89. Not All Tokens Matter: Data-Centric Optimization for Efficient Code Summarization

**arXiv ID:** 2601.20147 | [PDF](https://arxiv.org/pdf/2601.20147v1)

**作者:** Saima Afrin `[一作]` (William and Mary), Antonio Mastropaolo `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对代码摘要任务进行数据层面优化，提出并评估三种基于token级的压缩方法（AST、函数签名、CrystalBLEU），并结合语义对齐过滤（SIDE）进行多语言（Java、Python）实验。

**💡 创新点**

首次证明token级优化既能显著降低训练成本，又可在保持甚至提升摘要质量；发现不同编程语言对同一压缩策略的效果完全相反。

**🔧 技术方法**

使用CodeT5+模型进行微调，采用对比学习训练SIDE_py，利用AST解析、函数签名提取和CrystalBLEU统计剔除等技术。

**📊 数据集**

Funcom（Java/Python）、CoderEval、Mastropaolo的Java数据集、人工构建的Python PyBench以及Python Funcom经过CAT清洗后的数据。

**📈 对比分析**

与未压缩基线、仅语义过滤以及不同token压缩组合进行对比，使用BLEU/ROUGE-L/METEOR以及Shannon熵、token保留率评估；在Java中AST能提升约37%，在Python中函数签名压缩后性能最优，压缩率可达83%。

**⚠️ 局限性**

方法对语言高度依赖；仅在Java、Python上验证；使用的模型规模有限（220M），不同模型或任务可能表现不同；评估指标仍以BLEU等表面相似度为主，无法完全覆盖语义质量。

---

## 90. Hypergraph Samplers: Typical and Worst Case Behavior

**arXiv ID:** 2601.20039 | [PDF](https://arxiv.org/pdf/2601.20039v1)

**作者:** Vedat Levi Alev `[一作]` (University of Haifa), Uriya A. First `[通讯]` (University of Haifa)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5056368545)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了稀疏超图在样本采样、收敛与收容概率等方面的性质，并给出了最坏情况下的下界。

**💡 创新点**

提出了通用的下界函数 f_{k,r}(x)，并证明稀疏超图的收容概率至少达到该下界，从而证明了在稀疏条件下收容概率不可能低于 p^k，且与传统的完整超图相比更为紧致。

**🔧 技术方法**

使用了概率方法、二次矩估计、Chernoff 与超几何分布收敛、凸性与逆函数分析以及多变量不等式等技术。

**📊 数据集**

未使用实验数据集，全部为理论证明。

**📈 对比分析**

与完全超图和已知收容概率上界进行比较，证明在大多数参数设置下，新的下界优于以前的上界，并能在稀疏性更高的情形下实现更严格的约束。

**⚠️ 局限性**

主要限制在于常数系数极大、适用范围受限于稀疏性和顶点度的上界；并未给出针对非正规或度数极高的超图的完整证明。

---

## 91. Text-to-State Mapping for Non-Resolution Reasoning: The Contradiction-Preservation Principle

**arXiv ID:** 2601.19933 | [PDF](https://arxiv.org/pdf/2601.19933v1)

**作者:** Kei Saito `[一作]` `[通讯]` (Independent Researcher), Kei Saito (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了将自然语言文本映射到非坍塌状态空间的文本‑状态映射框架ϕ，利用冲突检测、解释提取和状态构造三阶段保留多重解释；

**💡 创新点**

首次为非解析推理（NRR）提供了算法化的文本‑状态桥梁，并通过混合规则与LLM提取实现多义性保持；

**🔧 技术方法**

采用规则基冲突标记检测、LLM枚举解释、语义向量嵌入与余弦相似度去重等技术；

**📊 数据集**

在68条涵盖五类歧义（对立、修饰、认知、词义、结构）的英文及日文句子上进行评估；

**📈 对比分析**

与传统单一解释的基线对比，熵从0提升至1.087，比单一解释提升约1.09比特；

**⚠️ 局限性**

局限包括数据集规模小、对LLM的依赖、冲突标记覆盖不足、权重设定经验化、未进行人工评估。

---

## 92. Visual Prompt-Agnostic Evolution

**arXiv ID:** 2601.20232 | [PDF](https://arxiv.org/pdf/2601.20232v1)

**作者:** Junze Wang `[一作]` (University of Science and Technology Beijing), Cong Cong `[通讯]` (Macquarie University)

**通讯引用:** 2541 | [OpenAlex ID](https://openalex.org/A5036142109)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Prompt‑Agnostic Evolution (PAE)框架，结合任务感知的频域初始化与共享Koopman运算，显著提升视觉提示调优的稳定性与收敛速度；

**💡 创新点**

创新将视觉提示调优视作离散动力系统，引入共享Koopman线性演化与Lyapunov稳定正则化，并通过频域shortcut搜索实现任务感知的提示初始化；

**🔧 技术方法**

使用频域快捷搜索（Modal Pre‑Alignment）、Koopman运算（KLD）、Lyapunov稳定正则、梯度振荡分析、损失景观与CKA可视化等技术；

**📊 数据集**

在25个数据集上评测，包括Fine‑Grained视觉分类、VTAB‑1k（19个任务）、ADE20K语义分割，以及MAE与Swin等不同ViT骨干；

**📈 对比分析**

与多种VPT变体及全量微调对比，PAE平均加速1.41×收敛，提升1–3%准确率（在FGVC、VTAB‑1k和ADE20K上），无推理时开销；

**⚠️ 局限性**

局限在低/无标签场景下任务感知初始化效果受限，Koopman潜空间维度需平衡，过大可能导致过拟合。

---

## 93. LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?

**arXiv ID:** 2601.20009 | [PDF](https://arxiv.org/pdf/2601.20009v1)

**作者:** J. Ben Tamo `[一作]` (Georgia Institute of Technology), Oleg Poliannikov `[通讯]` (Amazon)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5102804201)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个四种零样本提示变体的评估框架，用以系统检测多语言大模型在语言控制上的失效；通过层级可解释性分析发现模型内部形成“语义对齐–推理–语言输出”三阶段结构，并基于此提出只调节最高层的选择性微调方法来提升语言一致性。

**💡 创新点**

创新点在于：①将语言控制问题拆分为“多语言转移瓶颈”和“语言一致性瓶颈”两种失败模式；②揭示语言控制在模型层次上的空间分布；③提出仅更新最终几层即可获得 98% 以上语言一致性，且几乎不影响任务准确率，从而显著降低了微调成本。

**🔧 技术方法**

使用了 Logit Lens 语言概率追踪、隐藏状态余弦相似度分析、以及选择性微调（Selective SFT）等技术，并在 Qwen‑3‑32B 与 Bloom‑7.1B 两大模型上进行实验。

**📊 数据集**

主要数据集包括 MMLU、MGSM、XQuAD，且在六种语言（英语、法语、西班牙语、阿拉伯语、印地语、日语）上构建了代码切换、英文干扰、双语答案等四种提示变体。

**📈 对比分析**

与全参数微调相比，选择性微调仅更新 3–5% 参数即可实现 98%+ 的语言一致性，且在任务准确率上与全微调相当；在 Qwen‑3‑32B 上任务准确率从约 66% 提升至 90%+，在 Bloom‑7.1B 上尽管准确率仍低，但语言一致性几乎达到 100%。

**⚠️ 局限性**

局限性包括：对特定模型的层次依赖性强；在英文干扰提示下任务准确率仍较低；实验仅覆盖六种语言，未检验更多语言或更大规模模型；以及需要手动确定最终层位置，自动化方法尚未完善。

---

## 94. FFE-Hallu:Hallucinations in Fixed Figurative Expressions:Benchmark of Idioms and Proverbs in the Persian Language

**arXiv ID:** 2601.20105 | [PDF](https://arxiv.org/pdf/2601.20105v1)

**作者:** Faezeh Hosseini `[一作]` (Khatam University), Yadollah Yaghoobzadeh `[通讯]` (University of Tehran)

**通讯引用:** 865 | [OpenAlex ID](https://openalex.org/A5031030600)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文设计并评测了针对波斯语固定比喻表达（FFEs）的幻觉现象，提出了FFE‑Hallu基准。

**💡 创新点**

创新点在于首次构建针对FFEs的幻觉评测基准，并通过生成、检测、翻译三任务系统检验LLM的文化与比喻能力。

**🔧 技术方法**

采用多语言大模型（GPT‑4.1、Claude 3.7 Sonnet等）进行推理、生成，并利用LLM‑as‑a‑Judge实现自动评测。

**📊 数据集**

使用了600条精心构造的数据集，包括200条真实波斯习语/谚语、200条人工合成幻觉表达以及200条英波双语对。

**📈 对比分析**

通过人工标注与LLM‑as‑a‑Judge比较，GPT‑4.1在生成、翻译和检测任务中表现最佳，幻觉率最低；开放模型普遍出现高幻觉率。

**⚠️ 局限性**

限制在于数据量有限且仅覆盖波斯语，幻觉检测依赖母语者主观判断，自动评测仍不够鲁棒。

---

## 95. Are We All Using Agents the Same Way? An Empirical Study of Core and Peripheral Developers Use of Coding Agents

**arXiv ID:** 2601.20106 | [PDF](https://arxiv.org/pdf/2601.20106v1)

**作者:** Shamse Tasnim Cynthia `[一作]` (University of Saskatchewan), Banani Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5015470184)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究核心与边缘开发者在使用、评审、修改和验证自治编码代理提交的 PR 的行为差异，基于 GitHub 公开数据进行定量与定性分析。

**💡 创新点**

首次在真实 OSS 项目中对不同经验水平的开发者使用自治代理的模式进行实证检视，揭示核心与边缘组在委托任务、评审深度、修改倾向和 CI 通过率上的显著差异。

**🔧 技术方法**

利用 GitHub REST API 采集 PR、评论、提交和 CI 运行数据；采用 GPT‑4 进行 PR 目的分类；使用手工编码对评审评论和修改提交进行主题归类；运用 Mann‑Whitney‑Wilcoxon、Chi‑squared 等统计检验比较两组差异。

**📊 数据集**

共收集 9,427 条已关闭的代理 PR，来源于 1,391 个拥有 ≥100 星的 GitHub 仓库，涉及 Claude Code、Copilot、OpenAI Codex、Cursor 四个主流代理。

**📈 对比分析**

通过对比核心与边缘组在 PR 提交频率、评审评论数、修改行数、CI 通过率等指标，发现核心组合并到主分支和 CI 成功率更高，边缘组更频繁地使用代理、合并时 CI 检查不足；性能差异主要体现在合并成功率和 CI 通过率。

**⚠️ 局限性**

研究仅聚焦四个代理、仅 GitHub 公开仓库、仅已关闭 PR，经验度量基于单仓库贡献，未考虑跨仓库经验、代理更新版本以及其他私有/企业工作流的差异，可能导致结果无法全面推广。

---

## 96. Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing

**arXiv ID:** 2601.20107 | [PDF](https://arxiv.org/pdf/2601.20107v1)

**作者:** Zhuchenyang Liu `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**通讯引用:** 3806 | [OpenAlex ID](https://openalex.org/A5069437467)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了结构锚点剪枝（SAP），一种零训练、查询无关的剪枝方法，能将视觉文档检索的索引向量压缩超过90%，同时保持检索准确性。

**💡 创新点**

创新点在于：①将重要视觉补丁定位于多头注意力的中间层，通过视觉入度中心度进行识别；②提出Oracle Score Retention（OSR）诊断协议，揭示中间层结构信息峰值；③证明训练无关剪枝在高压缩率下仍可优于传统方法。

**🔧 技术方法**

使用技术包括：多向量晚期交互（MaxSim）检索框架、视觉自注意力的入度中心度计算、SAP-Mean/SAP-Max两种聚合策略、OSR评价指标以及在ViDoRe Benchmark上的评测。

**📊 数据集**

使用数据集：ViDoRe v1（10个子数据集）和ViDoRe v2（4个子数据集），并在ColPali、ColQwen2、Jina Embeddings v4三种VLM骨干上进行实验。

**📈 对比分析**

与EOS‑Adaptive、Random、Semantic Clustering等训练‑无关基线以及Light‑ColPali（训练‑有）比较，SAP在保持90%以上原始检索性能的同时，将索引向量压缩至10%以内，NDCG和Oracle Score Retention指标均优于基线。

**⚠️ 局限性**

局限性在于：仅针对多向量晚期交互的VLM；层选择与token预算是固定的；未探索对更大规模工业索引或其他图文匹配任务的泛化；缺乏实例级动态压缩率的自适应机制。

---

## 97. VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning

**arXiv ID:** 2601.20055 | [PDF](https://arxiv.org/pdf/2601.20055v1)

**作者:** Vikash Singh `[一作]` (Case Western Reserve University), Sam Bayless `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种结合大语言模型与SMT求解器的神经符号框架，通过迭代细化与验证来生成逻辑可验证的答案；

**💡 创新点**

创新点包括：多模型语义一致性检测提升形式化质量、语义路由机制根据命题类型分配符号或软验证、利用最小纠正子集(MCS)提供可操作的错误定位与反馈；

**🔧 技术方法**

使用技术包括：大语言模型（如GPT‑OSS‑120B、Claude Sonnet）、SMT求解器（Z3）进行一致性与可满足性检验、语义路由与等价图构建、MCS算法、以及自回归反馈循环；

**📊 数据集**

实验数据集涵盖FOLIO、ProofWriter、ZebraLogic、AR‑LSAT、BBEH、HLE等六个推理基准；

**📈 对比分析**

与传统单次推理（CoT、SC、SR）及神经符号基线（DSB、LogicLM、LINC、PoT）比较，平均提升约9–18%（在不同模型规模下），并在所有数据集实现了单调收敛；

**⚠️ 局限性**

局限性包括：显著的计算与延迟开销（每轮15–30秒），对模型规模存在“形式化阈值”限制（低于70B参数时形式化有效率低），以及仅能覆盖可判定逻辑（如QF‑UF、QF‑LIA），导致对复杂量化或递归表达式的处理受限。

---

## 98. Membership Inference Attacks Against Fine-tuned Diffusion Language Models

**arXiv ID:** 2601.20125 | [PDF](https://arxiv.org/pdf/2601.20125v1)

**作者:** Yuetian Chen `[一作]` (Purdue University), Ninghui Li `[通讯]` (Purdue University)

**通讯引用:** 17679 | [OpenAlex ID](https://openalex.org/A5101471208)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了扩散语言模型（DLM）的会员推断攻击（MIA）漏洞，并提出了基于子集聚合的进步掩码攻击方法，利用多重掩码和符号统计实现对训练数据的隐私泄漏检测。

**💡 创新点**

创新点在于：①将DLM的可多重掩码特性转化为攻击机会；②设计子集聚合机制，通过逆权重聚合将稀疏掩码的清晰信号与噪声分离；③采用符号统计在高度尾部噪声环境下保持鲁棒性。

**🔧 技术方法**

采用进步掩码采样、符号统计（sign‑based tests）、逆权重聚合、子集投票及参考模型对比重建误差等技术。

**📊 数据集**

使用 MIMIR（六个领域拆分）、WikiText‑103、AG News、XSum 等公开数据集，对 LLaDA‑8B‑Base 与 Dream‑v0‑7B‑Base 两款 DLM 进行实验。

**📈 对比分析**

与12个基线（包括 ARMs 的 Loss、ZLIB、ReCall 等以及图像扩散 MIA 方法 SecMI、PIA）在 AUC、TPR@10%、1%、0.1% FPR 上对比，平均相对 AUC 提升 30%，在低 FPR 下提升可达 8 倍。

**⚠️ 局限性**

局限性：仅针对掩码预测型扩散模型；需要兼容 token‑izer 与掩码方案的参考模型；进步掩码策略专门为双向掩码设计，未验证对其他扩散范式的适用性。

---

## 99. TeleStyle: Content-Preserving Style Transfer in Images and Videos

**arXiv ID:** 2601.20175 | [PDF](https://arxiv.org/pdf/2601.20175v1)

**作者:** Shiwen Zhang `[一作]` (TeleAI), Xuelong Li `[通讯]` (TeleAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 TeleStyle，基于 Qwen-Image-Edit 的轻量级图像与视频风格迁移模型，支持内容保持与多风格迁移。

**💡 创新点**

引入 Curriculum Continual Learning 训练策略结合清洗与合成三阶段数据集，兼顾内容保真与风格泛化；同时提出视频一帧引导的时序一致性模块。

**🔧 技术方法**

使用低秩适配 LoRA、流匹配训练目标、Diffusion Transformer、MS‑RoPE、CLIP/CLIP 嵌入、Wan2.1‑1.3B 等技术。

**📊 数据集**

自建 300k 高质量手工 triplet（D_collected）与 100 万合成 triplet（D_synthetic）以及 Ditto 与内部视频集。

**📈 对比分析**

通过 CSD、CPC 与 Aesthetic 评价，TeleStyle 在 6.317 Aesthetic 分数、0.577 CSD 与 0.441 CPC 上均优于所有现有 DiT 方案，表现为最优的风格相似度、内容保持与美感。

**⚠️ 局限性**

对极端结构变化的视频（如动漫）效果仍不稳定，且对超大分辨率与极细纹理迁移受限，训练成本仍较高。

---

## 100. LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation

**arXiv ID:** 2601.20083 | [PDF](https://arxiv.org/pdf/2601.20083v1)

**作者:** Lee Xiong `[一作]` (AI at Meta), Arnold Overwijk `[通讯]` (AI at Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了LLaTTE（LLM‑Style Latent Transformers for Temporal Events）两阶段序列建模框架，用以在广告推荐系统中实现可扩展的深度序列建模并兼顾严格的实时推理延迟；

**💡 创新点**

创新点在于（1）将大模型的深度与长序列计算异步推理到上游用户模型；（2）设计了Target‑Aware Adaptive Transformer、Multi‑head Latent Attention与自适应金字塔输出，既兼顾高效计算又能捕获长时序信息；（3）通过内容语义特征显著提升可扩展性，构建了内容驱动的可伸缩性曲线；

**🔧 技术方法**

采用Transformer基础的自注意力架构（MLA）、自适应金字塔剪枝、异步embedding服务、FlashAttention等技术，并在Meta内部大规模GPU集群上训练；

**📊 数据集**

使用Meta大规模广告推荐业务的真实流量数据，包含数十亿条用户交互记录、广告内容、用户/广告静态特征等；

**📈 对比分析**

在内部基准上与传统FM+Transformer混合模型对比，使用Normalized Entropy（NE）衡量；结果显示，两阶段模型在保持P99延迟不变的前提下，线上NE下降约0.25%（相当于4.3%转化率提升），实现显著业务收益；

**⚠️ 局限性**

局限性包括：①上游模型需通过异步推理与存储，存在延迟与一致性挑战；②跨阶段信息瓶颈导致传递率约50%，仍有提升空间；③对内容特征的依赖使得需要高质量语义嵌入，可能不易迁移到语料稀缺的场景；

---

## 101. Efficient Token Pruning for LLaDA-V

**arXiv ID:** 2601.20168 | [PDF](https://arxiv.org/pdf/2601.20168v1)

**作者:** Zhewen Wan `[一作]` (Li Auto Inc), Xianpeng Lang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对 LLaDA‑V 这一基于扩散的多模态模型，提出了在首次去噪步骤的中后层进行结构化视觉标记裁剪的 LLaDA‑FastV 方法，以显著降低推理成本。

**💡 创新点**

创新点在于发现 LLaDA‑V 的跨模态注意力聚合延迟到中后层，并将裁剪策略延迟到这一层，形成了持续裁剪（Persistent Pruning）与时空裁剪结合的全新思路。

**🔧 技术方法**

技术核心包括基于 FastV 的视觉标记重要性评估、在指定层进行比例裁剪、以及将裁剪状态锁定至后续所有去噪步骤，以实现 FLOPs 的累计削减。

**📊 数据集**

使用了 AI2D、MME、MMMU、MMMU‑Pro、RealWorldQA、ChartQA 等多种视觉‑语言推理基准数据集进行评估。

**📈 对比分析**

与原始 LLaDA‑V 以及在浅层进行裁剪的 Raw‑FastV 对比，K=15、P=50% 的配置可将计算量降低约 49%（FLOPs 下降至 51%），同时在绝大多数任务上保持 95% 以上的性能，甚至在某些任务上略有提升。

**⚠️ 局限性**

主要局限在于对裁剪比例与层数的选择仍需经验调优，且在极端高裁剪率或细粒度任务（如 ChartQA）上性能仍易受影响；未来仍需探索更自适应的裁剪策略及跨步裁剪的可行性。

---

## 102. DABench-LLM: Standardized and In-Depth Benchmarking of Post-Moore Dataflow AI Accelerators for LLMs

**arXiv ID:** 2601.19904 | [PDF](https://arxiv.org/pdf/2601.19904v1)

**作者:** Ziyu Hu `[一作]` (Stevens Institute of Technology), Xiaodong Yu `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5052001478)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了 DABench-LLM 基准框架，对数据流 AI 加速器（Cerebras WSE‑2、SambaNova SN30、Graphcore IPU）在大型语言模型（GPT‑2、LLaMA‑2）训练与推理中的性能进行系统评估，涵盖芯片级资源分配、负载均衡、资源利用率以及多芯扩展与部署优化；

**💡 创新点**

其创新点在于首次构建统一、可标准化且深入的 LLM 基准体系，既从硬件层面提炼关键指标，又能映射到软件编译与部署策略，显著提升对数据流加速器性能瓶颈的可解释性；

**🔧 技术方法**

采用了 decoder‑block 控制实验、层数/隐藏尺寸调节、资源分配率、负载不平衡（LI）评估、Roofline 模型分析、批量大小与混合精度敏感性实验等技术手段；

**📊 数据集**

以公开的 GPT‑2（Small/mini/tiny）和 LLaMA‑2 7B 等模型作为基准工作负载，本研究未使用特定语料库，而是通过模型结构本身进行实验；

**📈 对比分析**

通过单芯片级指标（PE/PCU 分配比例、LI、TFLOPs、内存占用）以及多芯片扩展（DP/TP/PP）和批量/精度对比，结果显示 Cerebras WSE‑2 在计算密集型方面优势显著但内存受限；SambaNova 在跨机 TP 上存在显著通信瓶颈；Graphcore IPU 内存受限但混合精度提升明显；总体上可达数百k tokens/s，性能受架构决定；

**⚠️ 局限性**

受限于各厂商对内部实现细节保密，实验主要基于编译时指标；对 WSE、RDU 的分析较为完整，IPU 的内部细节不完整；实验采用 decoder‑block 单元，无法覆盖完整模型；基准对不同软件栈的可移植性有限，且未评估能耗等额外指标。

---

## 103. Continuous-Flow Data-Rate-Aware CNN Inference on FPGA

**arXiv ID:** 2601.19940 | [PDF](https://arxiv.org/pdf/2601.19940v1)

**作者:** Tobias Habermann `[一作]` (Fulda University of Applied Sciences), Mario Garrido `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 1677 | [OpenAlex ID](https://openalex.org/A5059222971)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向FPGA的连续流卷积神经网络(CNN)硬件架构，利用数据速率感知的流动设计实现高资源利用率和低延迟推理。

**💡 创新点**

创新点在于通过数据速率分析、流水线交错（interleaving）与可重配置计算单元，实现输出连续流并消除空闲周期，弥合了流式和展开式架构之间的差距。

**🔧 技术方法**

技术实现包括隐式零填充、可重配置的卷积核处理单元(KPU)、全连接单元(FCU)、池化单元(PPU)、自动代码生成、量化感知训练以及多层级数据交错与资源共享。

**📊 数据集**

使用 ImageNet 数据集训练 MobileNetV1 与 ResNet18，并在 jet substructure tagging 数据集 JSC 上验证所提方法的通用性。

**📈 对比分析**

与全并行实现及现有基于 LUT/DSP 的 FPGA 加速器对比，得到在 Xilinx Alveo U280 上最高 6,944 inf/s、每次推理 3.55 mW、LUT 资源降低约 50% 的性能提升。

**⚠️ 局限性**

局限性是当数据速率低于 1 时，连续流无法通过交错恢复，导致卷积单元停滞；此外，高度可重配置的设计增加了硬件实现复杂度。

---

## 104. What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering

**arXiv ID:** 2601.20164 | [PDF](https://arxiv.org/pdf/2601.20164v1)

**作者:** Jim Maar `[一作]` (Hasso Plattner Institute), Neel Nanda `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型的隐式规划能力，利用押韵诗句生成和问答任务两种案例，提出并验证基于平均激活差的激活注入（steering）方法，评估不同规模和指令调优模型的规划表现。

**💡 创新点**

创新点在于：①用极简的均值激活差向量实现激活注入，无需昂贵的跨层转译器；②证明从1B参数起即出现隐式规划，且此机制在多种模型中普遍存在；③通过关注注意力头和MLP层，首次揭示押韵与问答规划的电路差异。

**🔧 技术方法**

主要技术包括：激活steering（平均激活差向量注入）、重生成指标、概率分布差异评估、注意力头补丁/消融实验，以及在不同层和位置进行的激活操作。

**📊 数据集**

数据集包括：①押韵实验：10个押韵族共1050行诗句（训练85行/测试20行）；②问答实验：20个名词对，分别生成13道提示性问题（训练）和5道测试问题，外加7道中立问题。

**📈 对比分析**

通过正确押韵率、重生率、概率分布偏移等定量指标对未steered与steered情形进行对比。结果显示：大模型和指令调优版本表现最佳，steering 能显著改变规划输出；较小模型的效果较弱。

**⚠️ 局限性**

局限性包括：实验仅覆盖押韵与简单问答两种任务，未验证方法在更复杂场景的普适性；缺乏对所有模型的完整电路分析；数据规模有限，主要聚焦英文文本。

---

## 105. Latent Object Permanence: Topological Phase Transitions, Free-Energy Principles, and Renormalization Group Flows in Deep Transformer Manifolds

**arXiv ID:** 2601.19942 | [PDF](https://arxiv.org/pdf/2601.19942v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Bugra Kilictas `[通讯]` (Bahçeşehir University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析Transformer深层残差流的协方差谱、有效秩、稀疏性等指标，提出大语言模型在足够深度时从高熵“液相”过渡到低熵“固相”的相变理论，并定义Transient Class Objects（TCOs）来描述推理阶段的稳定潜在基底。

**💡 创新点**

创新点在于将信息几何、自由能原理、随机矩阵理论与重整化群（RG）思想结合，给出相变阈值γ_c≈0.42，并提供收缩条件与混合模型的严格谱结果，将多步推理的出现解释为潜在空间的相变与离散化。

**🔧 技术方法**

使用技术包括：注意力软最大化作为Gibbs分布的自由能极小化、马尔可夫链的推理过程、随机矩阵理论（Marchenko–Pastur）和自适应谱分析、有效秩/参与比与稀疏性指标、RG式收缩分析以及混合模型谱理论。

**📊 数据集**

在多种规模（1B–30B）的公开Transformer模型上进行实验，使用标准语言建模/推理数据（Chain‑of‑Thought、通用语言数据集）提取激活并计算统计量。

**📈 对比分析**

通过比较不同规模模型在各层深度的Ω（对象完整性）分布、谱密度与有效秩等指标，发现大模型在γ_c≈0.42处出现显著的Ω跳跃、秩崩塌和谱尖峰，显示推理能力显著提升；而小模型未出现此相变，保持单峰分布和高秩。

**⚠️ 局限性**

局限性包括：仅在已训练模型上观察，缺乏因果验证；相变阈值在不同架构、任务上未充分泛化；理论假设（如线性化、块分解）与实际模型的非线性复杂性不完全匹配；实验数据量有限，未给出可操作的优化建议。

---

## 106. Light Field Display Point Rendering

**arXiv ID:** 2601.19901 | [PDF](https://arxiv.org/pdf/2601.19901v1)

**作者:** Ajinkya Gavane `[一作]` (North Carolina State University), Benjamin Watson `[通讯]` (North Carolina State University)

**通讯引用:** 4272 | [OpenAlex ID](https://openalex.org/A5016365293)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种适用于光场显示器（LFD）的点渲染技术（LFDPR），通过对光场视角的点云进行采样与抖动实现实时渲染。

**💡 创新点**

创新点包括：① LFD‑biased 采样——将三角形采样密度与 LFD 的水平/垂直采样对齐；② 基于纹理的 splatting——根据纹理分辨率动态降低点云密度；③ 多视角 mipmapping——在计算着色器中实现纹理 LOD，消除点渲染的纹理锯齿；④ 角度超采样与重建——通过 Gaussian 滤波和视角插值降低视角间混叠与交叉。

**🔧 技术方法**

技术手段：OpenGL 4.5 + NVIDIA RTX 3070 GPU；计算着色器 + 体素/点渲染；自定义 EIA（光场图像交错）合成；多视角 mipmap 计算；角度超采样与空间超采样；HDR‑VDP3 与 SSIM 评估。

**📊 数据集**

使用四个常见渲染场景：Sponza、Gallery、Coconut、Car；各场景包含高分辨率纹理、细分三角形、旋转动画，模拟游戏/渲染工作负载。

**📈 对比分析**

与传统多视角渲染（MVR）和 96×4K 高质量金标准（GStd）对比：在 480×360 视图下，LFDPR 速度提升 2–8 倍，误差（RMSE）与 perceptual 质量（HDR‑VDP3、SSIM）基本持平甚至略优；空间或视角超采样时速度提升约 3–5 倍，质量提升有限，空间超采样对 LFDPR 影响更大。

**⚠️ 局限性**

局限性：点云密度随三角形数增大而急剧增长，GPU 带宽瓶颈明显；空间超采样导致渲染速度下降；角度重建与超采样对人眼感知的真实影响仍需实验验证；当前实现对高带宽 GPU 的优化尚不足。

---

## 107. NucFuseRank: Dataset Fusion and Performance Ranking for Nuclei Instance Segmentation

**arXiv ID:** 2601.20104 | [PDF](https://arxiv.org/pdf/2601.20104v1)

**作者:** Nima Torbati `[一作]` (Danube Private University), Amirreza Mahbod `[通讯]` (Danube Private University)

**通讯引用:** 2108 | [OpenAlex ID](https://openalex.org/A5044448889)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对10个公开的H&E染色细胞核实例分割数据集进行了格式标准化，构建了统一的训练集NucFuse-train和测试集NucFuse-test，并对现有模型进行系统评估和排名。

**💡 创新点**

创新点在于提供统一的数据格式与基准、对数据集进行系统排名、设计融合训练集以提升泛化性能，并通过交叉相关矩阵评估数据集互补性。

**🔧 技术方法**

采用两种先进分割模型——基于CNN的HoVerNeXt和基于ViT的CellViT，结合数据增强、色彩标准化等技术进行训练与评估。

**📊 数据集**

使用了PCNS、PUMA、MoNuSeg、CPM17、TNBC、NuInsSeg、CoNSeP、MoNuSAC、DSB、CryoNuSeg十个手工标注的数据集，以及PanNuke和CellSAM作为对照。

**📈 对比分析**

通过在统一测试集上评估各模型的Panoptic Quality (PQ) 等指标，对单一数据集训练结果进行排名，并通过逐步合并最佳k个数据集的实验显示，融合训练提升了约8% PQ，验证了数据集互补效果。

**⚠️ 局限性**

局限性包括数据集特性差异导致的比较难度、未针对单个数据集做专门预处理、仅关注实例分割不包含分类、以及半自动标注数据可能存在泄漏等。

---

## 108. oculomix: Hierarchical Sampling for Retinal-Based Systemic Disease Prediction

**arXiv ID:** 2601.19939 | [PDF](https://arxiv.org/pdf/2601.19939v1)

**作者:** Hyunmin Kim `[一作]`, Siegfried K. Wagner `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出了层次化采样策略Oculomix，用于在视网膜图像的混合样本数据增强中保持患者特异性特征。

**💡 创新点**

创新点在于将患者–检查–图像层级结构与时间相关的排名监督结合，限制混合空间，避免破坏临床属性并显著提升系统性疾病预测性能。

**🔧 技术方法**

在Vision Transformer（ViT）模型上使用CutMix、MixUp等混合样本增强，并加入基于层级采样的策略和对排序损失的约束。

**📊 数据集**

主要使用内部AlzEye数据集（多时点视网膜图像及5年MACE标签），并在外部HYU MACE数据集上进行验证。

**📈 对比分析**

与传统图像级混合采样（Image-level）和仅检查级（Exam-level）对比，Oculomix在AUROC提升约3%，AUPRC提升约4.4%，外部C-index提升12%，表现更优。

**⚠️ 局限性**

局限在于跨时间点混合标签定义仍存在模糊性，并且仅在ViT模型上验证，缺乏对其他模型或更大样本的通用性研究。

---

## 109. Scaling Next-Brain-Token Prediction for MEG

**arXiv ID:** 2601.20138 | [PDF](https://arxiv.org/pdf/2601.20138v1)

**作者:** Richard Csaky `[一作]` `[通讯]` (Foresight Institute), Richard Csaky (Foresight Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个基于 MEG 数据的自回归生成模型，利用 token 化后训练大规模 Transformer，从短时上下文生成多分钟的脑电序列。

**💡 创新点**

创新点包括：① 用因果 SEANet+残差向量量化实现高压缩、低误差的 MEG token 化；② 在此 token 流上从零训练 Qwen2.5‑VL 样式的 decoder‑only Transformer；③ 设计了跨数据集（CamCAN、Omega、MOUS）通用的生成与评估框架，并提出基于神经生理学指标的长期生成稳定性与条件特异性评估。

**🔧 技术方法**

采用技术：因果 SEANet 编码器/解码器、残差向量量化 (RVQ)、Qwen2.5‑VL 结构的 Transformer、MRoPE 位置嵌入、滑动 KV 缓存生成、以及多种频域与协方差指标的评估。

**📊 数据集**

使用数据集：CamCAN、Omega（训练/验证）和 MOUS（完全保留为测试），共计约 500 小时 MEG 数据。

**📈 对比分析**

评估方法：与 BrainOmni、VidTok 以及 MEG‑GPT 等先前工作比较；在 MOUS 上的生成实现：重建 MAE≈0.2、PCC≈0.944；4‑分钟 roll‑out 的 OER 低且与真实分布保持一致；条件特异性指标显示生成结果在 prompt‑swap 与 real‑real 对照中均显著更接近真实续延，性能优于基线。

**⚠️ 局限性**

限制：未进行刺激锁定评估；高频波段重建略弱；tokenizer 训练相对耗时；缺乏大规模直接对比基线；仅在无刺激标签的 OOD 环境下验证。

---

## 110. Taming Toxic Talk: Using chatbots to intervene with users posting toxic comments

**arXiv ID:** 2601.20100 | [PDF](https://arxiv.org/pdf/2601.20100v1)

**作者:** Jeremy Foote `[一作]` (Purdue University), Hsuen-Chi Chiu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在七个大型Reddit子社区进行现场实验，邀请被删除毒性评论的用户与生成式AI聊天机器人进行康复性对话，研究其对毒性行为的影响。

**💡 创新点**

首次在真实线上社区中使用生成式AI对话干预研究毒性用户，系统评估叙事、规范强调、自我反思等多种说服策略对对话质量和行为变化的作用。

**🔧 技术方法**

采用OpenAI GPT‑3.5/GPT‑4构建对话机器人，使用Prompt Engineering定制不同说服风格；利用Perspective API检测毒性；通过PRAW抓取评论和Moderator日志；使用混合效应回归、岭回归等统计方法分析结果。

**📊 数据集**

共893名受访者，553条对话，来自7个子Reddit社区；对话记录、评论文本、Moderator日志以及用户行为数据（评论数、毒性评分、被删、被禁）均纳入分析。

**📈 对比分析**

对话质量通过主题编码与共现分析评估，行为结果用混合效应回归比较实验组与对照组。结果显示Wave2的对话更具反思性、攻击性更低，但未出现对毒性行为或参与度的显著统计差异。

**⚠️ 局限性**

限制：受访者自愿且无报酬，样本可能与一般互联网用户不符；对话长度短、互动有限，难以捕捉长期行为变化；仅测量毒性评分，未覆盖其他正面行为；AI模型差异对效果影响有限；对持续毒性用户的激励与干预机制不足。

---

## 111. A Flower-Inspired Solution for Computer Memory Wear-Leveling

**arXiv ID:** 2601.19902 | [PDF](https://arxiv.org/pdf/2601.19902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 112. TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance

**arXiv ID:** 2601.20239 | [PDF](https://arxiv.org/pdf/2601.20239v1)

**作者:** Zhemeng Zhang `[一作]` (Shanghai Jiao Tong University), Daolin Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1546 | [OpenAlex ID](https://openalex.org/A5045684726)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种在推理阶段通过触觉引导视觉运动策略的双阶段融合框架——TouchGuide，并配套低成本高精度的触觉采集系统TacUMI，用于细粒度接触丰富的机械操作。

**💡 创新点**

创新点在于将触觉信息以物理可行性评分的方式融入动作空间的采样过程，实现跨策略（扩散、流匹配）控制，而无需对原始策略重新训练；并提出可直接反馈的触觉采集硬件TacUMI。

**🔧 技术方法**

采用DINOv2特征提取、Transformer融合、对比学习构建的Contact Physical Model（CPM），以及在扩散模型和流匹配模型中应用的分类器引导技术；硬件侧使用Vive Tracker、光学触觉传感器和自研手持式抓手。

**📊 数据集**

利用TacUMI收集的多模态演示数据，分别在五个任务（鞋带系紧、芯片交接、黄瓜削皮、花瓶擦拭、锁开）中使用 100/50/50/30/20 份演示进行训练与评估。

**📈 对比分析**

与基线策略（DP、π_0.5）以及现有视觉-触觉方法（RDP、Policy Consensus、SafeDiff、Tactile Dynamics）对比，TouchGuide 在所有五个任务中平均提升约 20%–30% 的成功率（DP 16.3%→36.2%，π_0.5 35.9%→58.0%），且在不同机器人、传感器和策略下均保持优势。

**⚠️ 局限性**

局限性在于 CPM 需要针对每个任务单独训练，对比学习样本量有限，且整个框架仍依赖于高质量的触觉演示；未来需探索无任务专属的通用 CPM 与更高效的触觉表示学习。

---

## 113. RAPID-Graph: Recursive All-Pairs Shortest Paths Using Processing-in-Memory for Dynamic Programming on Graphs

**arXiv ID:** 2601.19907 | [PDF](https://arxiv.org/pdf/2601.19907v1)

**作者:** Yanru Chen `[一作]` (University of California San Diego), Tajana Rosing `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个递归分区的全对全最短路径（APSP）算法和基于相变存储的处理内存（PIM）系统RAPID-Graph，能够在单个芯片内完成完整的Floyd‑Warshall和Min‑Plus计算。

**💡 创新点**

创新点在于：① 递归分区APSP数据流将大图拆分为符合PCM瓦片大小（≤1024节点）的子图，实现完全在内存中并行执行；② 2.5D异构堆栈将PCM计算Die、逻辑Die、HBM3和FeNAND通过UCIe统一打包，提供高带宽、低能耗的计算与存储协同；③ 在PCM中实现了专用的Permutation单元和Min‑Comparator树，支持位串逻辑的加减与最小比较。

**🔧 技术方法**

采用的技术包括：40 nm Sb₂Te₃/Ge₄Sb₆Te₇相变存储、1T1R SLC PCM阵列、FELIX式位串加减与最小比较、PCM专用Permutation单元、6级Min‑Comparator树、UCIe 2.5D堆栈、HBM3高带宽缓存、FeNAND高容量持久存储、Metis图划分、NiemaGraphGen合成图生成。

**📊 数据集**

实验使用了：OGBN‑Products（约2.45 M节点）真实图、Synthetic Newman–Watts–Strogatz（NWS）和Erdős–Rényi（ER）图，以及NiemaGraphGen生成的100、1024、32768节点规模图。

**📈 对比分析**

与单机CPU、A100 GPU、估计H100 GPU、PIM‑APSP、Partitioned APSP、Co‑Parallel APSP等基线对比；在2.45 M节点时，RAPID-Graph比GPU集群快5.8×、能耗低1186×；对单机H100比42.8×速度、392×能效；相较于SOTA PIM 8.3×速度、104×能效；在1k节点上对CPU 1061×速度、7208×能效。

**⚠️ 局限性**

局限性在于：① 仅适用于需要密集矩阵的精确APSP，无法直接处理极稀疏或动态图；② 受PCM瓦片大小和堆栈容量限制，极大规模图仍需进一步划分；③ 需要专用的PCM硬件和UCIe、HBM3、FeNAND等组件，成本与可部署性受限；④ 预处理阶段的Metis划分及递归开销在非常大或不规则图上仍可能成为瓶颈。

---

## 114. Techno-economic optimization of a heat-pipe microreactor, part II: multi-objective optimization analysis

**arXiv ID:** 2601.20079 | [PDF](https://arxiv.org/pdf/2601.20079v1)

**作者:** Paul Seurin `[一作]` (Idaho National Laboratory), Dean Price `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 219 | [OpenAlex ID](https://openalex.org/A5075369934)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对热管微型核反应堆(HPMR)的几何设计与成本进行多目标优化，目标是同时降低电价(LCOE)和杆整合峰值因子(F_Δh)，并满足安全与运营约束。

**💡 创新点**

首次将基于强化学习的 Pareto Envelope Augmented with Reinforcement Learning (PEARL) 算法与物理感知的 surrogate 模型相结合，用于核反应堆的多目标决策，揭示不同材料成本情景下的设计取舍。

**🔧 技术方法**

采用 OpenMC Monte Carlo 进行物理仿真，MOUSE 工具进行技术经济评估，基于高斯过程与多层感知机构建 surrogate 模型；利用 PEARL（PPO 作为策略优化器）实现多目标搜索；还利用 NSGA-II 等传统算法作对照。

**📊 数据集**

构建了约 921 条样本数据集（经 40–168 小时 OpenMC 计算），其中 875 条用于训练 surrogate；数据包括燃料寿命、SDM、F_Δh、q''_max 等 QoI，以及对应的几何参数和成本。

**📈 对比分析**

与单目标优化和传统遗传算法对比，PEARL 在三种成本情景下均能获得更低的 LCOE（可达 4–5 万元/兆瓦时）并保持或改善 F_Δh，显示出明显的 Pareto 前沿优势；实验表明，最大可将 LCOE 降低约 57%。

**⚠️ 局限性**

主要限制包括：surrogate 模型对燃料寿命和 SDM 的预测误差较大；单次模拟成本高，导致样本量受限；对约束的严格性可能导致部分 Pareto 结果不满足实际要求；未来需引入主动学习、动态重训练以提升模型准确性。

---

## 115. Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning

**arXiv ID:** 2601.20209 | [PDF](https://arxiv.org/pdf/2601.20209v1)

**作者:** Jinyang Wu `[一作]` (Tsinghua University), Jianhua Tao `[通讯]` (Tsinghua University)

**通讯引用:** 8438 | [OpenAlex ID](https://openalex.org/A5112613657)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Spark框架，利用动态分支在关键决策点进行自主探索，提升LLM在长时序任务中的训练效率；

**💡 创新点**

创新点在于通过内部不确定性标签自动识别关键状态并触发分支，实现资源优先分配，同时将探索树与策略优化无缝结合，无需外部手工启发式；

**🔧 技术方法**

采用关键-状态动态分支探索、SFT生成不确定性标签、树结构策略更新、GRPO/REACT等RL技术，并基于Qwen2.5‑1.5B/7B模型；

**📊 数据集**

实验使用ALFWorld、ScienceWorld和WebShop三大长期规划基准；

**📈 对比分析**

与闭源LLM（GPT‑4o、GPT‑5、Gemini‑2.5‑Pro）以及Prompting（ReAct）和RL方法（GRPO、ETO、GiGPO、RLVMR）对比，Spark在各模型规模与任务域均取得最高成功率，样本/令牌效率提升显著（如Spark‑1.5B ScienceWorld L2 49.2% > GPT‑5 33.6%），并在OOB泛化上显著优于基线；

**⚠️ 局限性**

限制在于对低能力基础模型的自我感知不足，可能导致内部不确定性标签失效，进而影响动态分支的效果，且目前依赖内部标签，未结合外部反馈提升鲁棒性。

---

## 116. Semantic Uncertainty Quantification of Hallucinations in LLMs: A Quantum Tensor Network Based Method

**arXiv ID:** 2601.20026 | [PDF](https://arxiv.org/pdf/2601.20026v1)

**作者:** Pragatheeswaran Vipulanandan `[一作]` (University of Miami), Dilip Sarkar `[通讯]` (University of Miami)

**通讯引用:** 4637 | [OpenAlex ID](https://openalex.org/A5004229694)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将量子张量网络与熵最大化相结合，提出了一种针对大型语言模型的自监督幻觉检测框架；

**💡 创新点**

创新点在于利用TS概率的量子扰动理论进行本地不确定性量化，并将其纳入语义Rènyi熵的调节，避免传统熵估计过度或不足；

**🔧 技术方法**

核心技术包括量子张量网络(UQ)、Rènyi熵、熵最大化与KL正则、语义聚类（基于DeBERTa双向蕴涵）；

**📊 数据集**

实验使用TriviaQA、SQuAD 1.1、NQ-Open、SVAMP等四个问答数据集，模型涵盖Mistral、Falcon、LLaMA系列不同规模与量化位宽；

**📈 对比分析**

与现有无监督熵方法、监督ER和p(True)基线相比，在AUROC、AURAC、RAC指标上均取得或匹配最先进水平，并在16/8/4位量化下保持稳健；

**⚠️ 局限性**

局限包括仅在开源低至中等规模模型上验证、依赖可获取token级概率、需外部蕴涵模型导致误差传播、对黑盒LLM不适用。

---

## 117. Loss Landscape Geometry and the Learning of Symmetries: Or, What Influence Functions Reveal About Robust Generalization

**arXiv ID:** 2601.20172 | [PDF](https://arxiv.org/pdf/2601.20172v1)

**作者:** James Amarel `[一作]` (Los Alamos National Laboratory), Gerd J. Kunde `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 33306 | [OpenAlex ID](https://openalex.org/A5109907133)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过影响函数诊断探测神经网络对 PDE 求解器对称性的学习情况。

**💡 创新点**

提出了基于影响函数的轨道梯度一致性度量，用以评估模型对物理对称性的内化。

**🔧 技术方法**

使用影响函数（神经切线核）、梯度一致性测量、前向等变误差评估，以及 UNet 与 ViT 结构。

**📊 数据集**

使用 PDEGym 数据集，包括二维可压缩 Euler 流与 Navier–Stokes 流的三类初始条件。

**📈 对比分析**

对 UNet 与 ViT 在相同任务下进行对比，ViT 在测试误差上表现更优，但对称性一致性低；UNet 维持较好对称性但收敛较慢。

**⚠️ 局限性**

局限在于仅评估离散旋转、翻转和平移对称，未考虑 Galilean、尺度等连续对称；仅对两种网络结构；影响函数计算成本高且仅在训练后期评估。

---

## 118. Achieving Productivity Gains with AI-based IDE features: A Journey at Google

**arXiv ID:** 2601.19964 | [PDF](https://arxiv.org/pdf/2601.19964v1)

**作者:** Maxim Tabachnyk `[一作]` (Google), Satish Chandra `[通讯]` (Google)

**通讯引用:** 3000 | [OpenAlex ID](https://openalex.org/A5101965118)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并改进了Google内部IDE的两个AI驱动功能：代码补全和自然语言驱动的代码变换（Transform Code），通过系统化的实验、模型调优和用户体验迭代，实现了显著的生产力提升。

**💡 创新点**

创新点包括：1）采用适应性缓存与Speculative Decoding降低补全延迟；2）通过上下文优先级选择和简化提示构造最大化模型可用信息；3）利用真实开发者手工重写数据进行SFT，显著提升变换质量和多轮对话能力；4）结合A/B实验与离线因果推断（DiD）精确评估功能对开发者生产力的影响。

**🔧 技术方法**

核心技术：大规模语言模型（Gemini、Encoder‑Decoder）、Speculative Decoding、上下文提示工程、SFT（基于真实重写与对话示例）、缓存与请求重用策略、A/B实验与Causal Inference（DiD）等。

**📊 数据集**

数据集：1）Google工程师的编辑历史（按键级记录）用于补全训练；2）内部代码库与相关文件用于输入上下文；3）从IDE中收集的Transform Code手工重写与多轮对话示例（约100‑400条）用于SFT；4）内部测评与观测数据用于A/B和因果实验。

**📈 对比分析**

比较方法：通过A/B实验评估功能接受率、FCML、提示数量、延迟等；通过离线DiD因果分析评估Transform Code对CLT、MeanDurInvSess等生产力指标的长期影响。性能表现：代码补全的FCML提升41%，接受率提升17%；Transform Code的提示数提升40%，接受率达68%，并在实验中实现<1 s的平均延迟。

**⚠️ 局限性**

限制：①依赖Google内部IDE与数据，外部复现难度大；②对罕见或高度复杂的多文件编辑支持尚未充分验证；③多轮对话功能在不同语言/项目中的泛化性待进一步评估；④模型仍需在资源消耗与延迟之间权衡，尚未实现最优平衡。

---

## 119. CascadeMind at SemEval-2026 Task 4: A Hybrid Neuro-Symbolic Cascade for Narrative Similarity

**arXiv ID:** 2601.19931 | [PDF](https://arxiv.org/pdf/2601.19931v1)

**作者:** Sebastien Kawada `[一作]` (Geffen Academy), Dylan Holyoak `[通讯]` (Geffen Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合神经自一致投票和多尺度符号叙事分析的层次化系统，用于判断两个故事与锚故事的叙事相似度。

**💡 创新点**

创新点在于：1) 采用超多数投票与自一致机制作为不确定性评估；2) 当出现完全平局时，利用五种基于叙事理论的符号信号（词汇重叠、语义嵌入、故事语法结构、张力曲线、事件链）构成加权集成进行决策。

**🔧 技术方法**

使用技术包括：Gemini 2.5 Flash LLM的并行投票、自一致投票算法、超多数阈值、差分进化优化权重、TF‑IDF、Sentence‑Transformer、TextBlob情感分析、事件链 LCS 等。

**📊 数据集**

数据集为 SemEval‑2026 Task 4 的开发集（200个三元组）和训练集（1900个三元组），来源于 CMU Movie Summary Corpus（电影情节摘要）。

**📈 对比分析**

与单票、八票自一致、三次投票基线相比，CascadeMind 在开发集上达到了81 %准确率（提升13 %），在决策路径上超多数路径准确率为85 %，而纯符号决策仅为61 %。

**⚠️ 局限性**

局限性包括：1) 仅在商业 LLM 上测试，缺乏可复现性；2) 符号集成在训练集上表现极好，但在困难开发案例中准确率仅为61 %；3) 只影响约5 % 的实例，整体提升有限。

---

## 120. Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery

**arXiv ID:** 2601.20088 | [PDF](https://arxiv.org/pdf/2601.20088v1)

**作者:** Meng Xin `[一作]`, Huizi Mao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对NVFP4量化的LLM和VLM提出并验证了一种基于KL散度的量化感知蒸馏（QAD）方法，用来恢复量化后模型的推理精度。

**💡 创新点**

创新点在于将知识蒸馏与量化感知训练结合，使用全精度教师模型的软标签实现分布对齐，从而在多阶段SFT/RL/模型融合后仍能稳定恢复几乎BF16的性能，且对数据覆盖率和质量具有鲁棒性。

**🔧 技术方法**

核心技术包括：NVFP4二级缩放量化、KL散度蒸馏损失、混合SFT与RL生成数据、不同学习率策略、以及对比实验中的多种评测指标。

**📊 数据集**

实验使用了AceReason Nemotron、Nemotron Nano（V2、12B VL）、Llama Nemotron Super等模型的训练集（SFT数据、RL生成样本、部分域数据、随机token），并在AIME、GPQA、LiveCodeBench等标准benchmark上评估。

**📈 对比分析**

与PTQ、QAT以及原始BF16基线对比，QAD在多模型、多任务上均能将NVFP4模型的性能提升至90%以上的BF16水平，且在RL重训练模型中QAT会显著下降，QAD保持稳定；总体表现优于传统QAT。

**⚠️ 局限性**

局限性包括：仍需完整或部分域的训练数据（尽管量级较低）、对教师模型的依赖（无法替代极端压缩或自监督训练的场景）、以及在某些大模型或高异构架构中可能需要更细粒度的量化策略。

---

## 121. PaperAudit-Bench: Benchmarking Error Detection in Research Papers for Critical Automated Peer Review

**arXiv ID:** 2601.19916 | [PDF](https://arxiv.org/pdf/2601.19916v1)

**作者:** Songjun Tu `[一作]` (Institute of Automation), Dongbin Zhao `[通讯]` (Institute of Automation)

**通讯引用:** 15268 | [OpenAlex ID](https://openalex.org/A5100624298)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PaperAudit-Bench，构建了包含多类型错误的科学论文检测与评审基准，旨在提升大语言模型的批判性评审能力。

**💡 创新点**

创新点在于同时设计错误检测数据集和基于证据的评审框架，并证明显式错误检测能显著提升评审的严谨性与与人类评审的对齐度。

**🔧 技术方法**

采用大语言模型（如 Gemini‑2.5‑Pro、GPT‑5、Qwen 系列、Llama‑3.2 等）进行监督微调（SFT）和强化学习（RL）训练，并集成多模式检索与多代理检测流程。

**📊 数据集**

使用 PaperAudit‑Dataset，该数据集基于 220 篇 ICLR/ICML/NeurIPS 会议论文注入约 15 个合成错误，覆盖 8 类错误类型，且包含跨段一致性与局部错误。

**📈 对比分析**

与 DeepReview 等现有基线对比，显式错误检测在宏 F1、错误覆盖率、评审分数差异化方面均有显著提升；轻量级模型在 SFT+RL 后实现与大型模型相当的检测性能，且能在有限计算资源下部署。

**⚠️ 局限性**

局限性包括错误为合成而非真实、可能导致过度严苛的评审、对提示与配置敏感，以及轻量化模型在极端稀疏错误场景下仍存在覆盖不足的问题。

---

## 122. Lowest Span Confidence: A Zero-Shot Metric for Efficient and Black-Box Hallucination Detection in LLMs

**arXiv ID:** 2601.19918 | [PDF](https://arxiv.org/pdf/2601.19918v1)

**作者:** Yitong Qiao `[一作]` (Zhejiang University), Zhixuan Chu `[通讯]` (Zhejiang University)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5008967163)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为最低跨度置信度（Lowest Span Confidence, LSC）的零样本幻觉检测指标，能够仅凭单次前向推理和输出的 token 概率进行判断

**💡 创新点**

创新点在于通过滑动窗口聚合连续 token 的置信度来捕捉局部不确定性，既避免了全局指标如困惑度的稀释效应，又克服了单 token 指标对噪声的敏感性

**🔧 技术方法**

技术实现上采用了滑动窗口机制、平均概率计算、单向前向推理和 log‑probability 访问，完全可在黑盒 API 场景下使用

**📊 数据集**

实验使用了四个问答基准：Natural Questions、TriviaQA、SQuAD 2.0 以及 CoQA，并在 LLaMA-7B/13B 与 Qwen 系列（0.5B–32B）模型上评测

**📈 对比分析**

与多种基线（Perplexity、Energy、LN‑Entropy、Lexical Similarity、EigenScore、AGSER）对比，LSC 在 AUC‑ROC 和 Pearson 相关系数上均稳步领先，尤其在资源受限和单次推理场景下表现最为突出

**⚠️ 局限性**

局限性在于目前仅是后置检测工具，未提供自动纠错或训练阶段的正则化策略，无法直接用于模型的主动改进

---

## 123. Supporting Informed Self-Disclosure: Design Recommendations for Presenting AI-Estimates of Privacy Risks to Users

**arXiv ID:** 2601.20161 | [PDF](https://arxiv.org/pdf/2601.20161v1)

**作者:** Isadora Krsek `[一作]` (Carnegie Mellon University), Sauvik Das `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2719 | [OpenAlex ID](https://openalex.org/A5006053551)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用漫画创作与在线调查评估五种人口风险估计（PRE）呈现方案，探讨其对 Reddit 用户自我披露决策的影响。

**💡 创新点**

提出四条设计原则：提供可操作建议、透明解释计算过程、避免过度自我审查、使用直观语言，填补了现有隐私风险提示的缺失。

**🔧 技术方法**

采用设计虚构、漫画板、问卷收集与定性/定量编码相结合的方法。

**📊 数据集**

数据来自自构的四种 Reddit 自我披露情境漫画（共 20 版），并由 44 名美国 Reddit 用户完成调查。

**📈 对比分析**

通过对 132 条受访者反思进行主题编码与统计检验，评估不同 PRE 设计的可接受度与效果；未提供算法性能指标，主要以用户体验和偏好为评价维度。

**⚠️ 局限性**

样本仅为美国 Reddit 用户且多为技术熟练者；情境设置有限，缺乏跨平台和多样化威胁模型的验证，结果可能不具普适性。

---

## 124. In-Context Reinforcement Learning From Suboptimal Historical Data

**arXiv ID:** 2601.20116 | [PDF](https://arxiv.org/pdf/2601.20116v1)

**作者:** Juncheng Dong `[一作]` (Duke University), Vahid Tarokh `[通讯]` (Duke University)

**通讯引用:** 32948 | [OpenAlex ID](https://openalex.org/A5020766546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 Decision Importance Transformer (DIT)，一种利用子最优历史轨迹进行无监督预训练的 Transformer‑based in‑context 强化学习框架；

**💡 创新点**

核心创新在于引入优势估计的加权最大似然预训练目标，并通过 Transformer 进行任务识别与优势函数的上下文推断，实现从仅有子最优轨迹到近似最优策略的迁移；

**🔧 技术方法**

技术包括基于 GPT‑2 的因果 Transformer、优势函数的双 Transformer 估计（V 与 Q 估计器）、指数加权重学习（WMLE）以及在 ICRL 环境下的上下文任务识别；

**📊 数据集**

在实验中使用了线性 bandit、Dark Room、Miniworld、Meta‑World (reach‑v2) 与 Half‑Cheetah 等多种离散/连续 MDP 任务，以及对应的子最优轨迹数据集；

**📈 对比分析**

与 DPT、AD、BC、SAC、UCB、TS 等基线对比，DIT 在 bandit 场景下能与理论最优算法齐平，在 MDP 场景中往往匹配甚至超越 DPT，显示出在仅靠子最优数据预训练下的强大性能；

**⚠️ 局限性**

主要限制在于必须使用具有一定奖励的子最优策略轨迹，纯随机或极度低效的数据难以通过该框架提升；此外，优势估计与加权策略仍受轨迹覆盖度与任务多样性的影响。

---

## 125. Fuzzy Categorical Planning: Autonomous Goal Satisfaction with Graded Semantic Constraints

**arXiv ID:** 2601.20021 | [PDF](https://arxiv.org/pdf/2601.20021v1)

**作者:** Shuhui Qu `[一作]` (Stanford University), Shuhui Qu `[通讯]` (Stanford University)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5112694715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种模糊化的范畴理论规划框架（FCP），用可度量的满意度对动作进行评分并保持硬约束可执行性；

**💡 创新点**

创新点在于将模糊真值与拉卡西维茨‧卢卡西乌克 t‑norm 相结合，实现对多步计划质量的可解释累积衰减，同时保留传统范畴规划的可验证性；

**🔧 技术方法**

主要技术包括：LLM 基于 k‑样本中位数聚合的模糊前置评估、拉卡西维茨 t‑norm（及其余量）组合、双向搜索的需求传播、拉回（pullback）硬约束检验以及 α‑cut 计划接受；

**📊 数据集**

数据集涵盖公开 PDDL3 优先级/超订阅基准以及自构建的 RecipeNLG‑Subs（从 RecipeNLG 转化的缺失配料替代任务，配合 Recipe1MSubs 与 FoodKG 约束）；

**📈 对比分析**

与传统 PDDL3 平滑规划器（SGPlan、LPPG、MIPS‑XXL）以及 LLM 直接生成、检索式替代和 ReAct 对比，FCP 在 PDDL3 任务中保持 85.2% 成功率，略低于 SGPlan；在 RecipeNLG‑Subs 上实现 83.6% 成功率且 BLEU 约 90.5，显著优于纯 LLM/检索方法；

**⚠️ 局限性**

局限包括：模糊度量依赖 LLM 的推理可靠性，长计划会因 t‑norm 的零化导致失效；数据集构造可能带来偏差，BLEU 对语义重述不敏感；未来工作需加入语义相似度评估和用户研究。

---

## 126. Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading

**arXiv ID:** 2601.19910 | [PDF](https://arxiv.org/pdf/2601.19910v1)

**作者:** William Meng `[一作]` (University of Pennsylvania), Hong Wang `[通讯]` (Intel)

**通讯引用:** 18262 | [OpenAlex ID](https://openalex.org/A5100369619)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对 LLM 推理中 KV 缓存离线（offloading）引起的内存带宽瓶颈进行了系统分析，并提出了基于 κ_ratio 与 κ_crit 的理论框架来预测何时会由 compute‑bound 变为 memory‑bound。

**💡 创新点**

创新点：①构造了 κ_crit（临界缓存/预填充比）与 κ_ratio（工作负载比）双重指标，解耦模型与硬件；②通过实测与屋顶线分析验证框架；③给出硬件（NVLink、HBM）、模型（MLA、MoE）与调度（迭代级 token budget）三方优化方向，并量化其提升空间。

**🔧 技术方法**

技术手段：基于 vLLM + LMCache 的 KV 离线实现；使用 NVIDIA H100/B200 系列 GPU 与 PCIe‑5；对 KV 缓存大小、预填充 token 数量进行精确控制；利用 roofline 图、带宽/ FLOPs 计算得到 κ_crit；通过 NVIDIA‑smi、功耗监测与自定义日志测量实际执行时间与 GPU 利用率。

**📊 数据集**

使用的数据集：ShareGPT（多轮对话）、NarrativeQA（长文档问答）和 FinQA（金融文档问答）等，分别提供了不同的 κ_ratio 分布（多轮对话 median 100，文档问答 median 5000–10000）。

**📈 对比分析**

比较方法与性能结果：在 H100/PCIe‑5 上，KV 离线导致 PCIe 传输时间占 99%（Qwen）或 88%（多轮对话），GPU 实际功耗仅 28%–35% TDP；κ_ratio 远超 κ_crit（>1000 与 >10），使推理陷入 memory‑bound；与仅做前缀缓存（不离线）的基线相比，整体延迟提升至 60–86 倍。

**⚠️ 局限性**

局限性：①模型与硬件的 κ_crit 计算依赖峰值带宽，实际带宽受系统瓶颈影响，导致预测偏高；②实验仅覆盖 FP16 推理与 1‑token 输出，未考虑完整生成过程；③未完整评估 DeepSeek‑V2 等 MLA 方案；④调度改进仅在 vLLM 上验证，缺少跨框架通用性验证。

---

## 127. Local Duality for Sparse Support Vector Machines

**arXiv ID:** 2601.20170 | [PDF](https://arxiv.org/pdf/2601.20170v1)

**作者:** Penghe Zhang `[一作]` (Hong Kong Polytechnic University), Houduo Qi `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 2054 | [OpenAlex ID](https://openalex.org/A5112621269)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构造0/1损失SVM的对偶形式，提出了稀疏支持向量机（SSVM）的理论基础，并证明其局部解满足线性表示定理；同时探讨了该局部解与hSVM、rSVM之间的关系，并在真实数据集上验证了其优越性。

**💡 创新点**

创新点包括：① 建立了SSVM（对偶）与0/1损失SVM之间的局部对偶理论；② 证明了局部解可由对偶解重构，且满足线性表示；③ 揭示局部解是hSVM全局解在特定参数下的极限，也是rSVM的局部最优点；④ 通过实验展示了局部解在目标值、误差率和泛化指标上的优势。

**🔧 技术方法**

主要技术方法有：对偶性分析、KKT条件推导、局部对偶理论、水平集有界性与极限分析、线性表示定理证明；算法上使用了L0/1 ADMM求解0/1损失SVM、LIBSVM求解hSVM、CCCP求解rSVM。

**📊 数据集**

实验使用了来自libsvm和openml的多种二分类数据集，包含从10维到260维、样本量从几十到数千不等的数据。

**📈 对比分析**

比较方法为：在相同参数下分别求解0/1损失SVM局部解、hSVM全局解（通过调节权重得到近似局部解）以及rSVM局部解；指标包括目标值、误分类率、误分类上界等。实验结果显示，0/1损失SVM的局部解往往在目标值和误分类上界上优于hSVM全局解和rSVM局部解，并在多数数据集上保持较好的泛化性能。

**⚠️ 局限性**

局限性包括：① 只针对线性SVM，未讨论核扩展；② 对偶求解仍为非凸问题，求解效率与可行性需要进一步研究；③ 对局部解的稳定性和收敛性分析仅在理论框架内给出，实际应用中可能受参数选择和初始值影响；④ 实验规模有限，尚需在更大规模数据上验证。

---

## 128. MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference

**arXiv ID:** 2601.19961 | [PDF](https://arxiv.org/pdf/2601.19961v1)

**作者:** Huanlin Gao `[一作]` (China Unicom), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MeanCache，一种无需训练的缓存框架，通过在平均速度域重建轨迹来加速 Flow Matching 推理。

**💡 创新点**

创新点在于将缓存视角从瞬时速度转向平均速度，并利用 Jacobian–Vector Product（JVP）估计平均速度，同时设计基于轨迹稳定性的预算约束最短路径调度策略，以降低误差累积。

**🔧 技术方法**

使用平均速度理论、JVP 缓存、图论最短路径优化（峰值抑制）、以及在 FLUX、Qwen‑Image、HunyuanVideo 上的推理加速实验。

**📊 数据集**

主要使用公开的商用生成模型的默认数据集：FLUX.1[dev]（文本‑图像）、Qwen‑Image（文本‑图像）、HunyuanVideo（文本‑视频）。

**📈 对比分析**

与 TeaCache、DiCache、ToCa、TaylorSeer 等基线对比，MeanCache 在 FLUX.1 4.12×、Qwen‑Image 4.56×、HunyuanVideo 3.59× 的加速比下，保持或提升 ImageReward、CLIP Score、SSIM、PSNR 等质量指标，整体性能优于现有缓存方法。

**⚠️ 局限性**

主要限制是对 JVP 近似的依赖；在极高加速比时仍可能出现局部误差，且调度策略需要先验构建图表，对不同模型或时间步长的适配仍需进一步验证。

---

## 129. DBTuneSuite: An Extendible Experimental Suite to Test the Time Performance of Multi-layer Tuning Options on Database Management Systems

**arXiv ID:** 2601.20015 | [PDF](https://arxiv.org/pdf/2601.20015v1)

**作者:** Amani Agrawal `[一作]` (New York University), Dennis Shasha `[通讯]` (New York University)

**通讯引用:** 20668 | [OpenAlex ID](https://openalex.org/A5055576104)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个可扩展的调优基准套件，对 MySQL、MariaDB、PostgreSQL 和 DuckDB 四大主流开源数据库在数据加载、索引设计、查询重写、连接池、触发器、垂直分区、去范式化等多层面进行系统实验，并提供完整脚本与数据。

**💡 创新点**

创新点在于：①提供一套统一、可复现的实验框架，支持任意数据库和查询类型的扩展；②系统覆盖从硬件到应用层的调优参数，打破以往单侧实验的局限；③对 40+ 调优方案在 4 个数据库上的交叉对比，揭示同一调优选项在不同系统中效果差异巨大的经验法则。

**🔧 技术方法**

技术手段包括：SQL 脚本批量插入与分批加载、强制索引使用、索引类型（B+树、哈希、ART）、聚合触发器、存储过程循环、游标、垂直分区、连接池管理；实验环境为 Ubuntu 22.04 / RHEL 9，CPU 64 核 2.1 GHz，256 GB 内存；通过 Python / Bash 自动化执行 10 次平均；使用 SHOW WARNINGS/EXPLAIN 监控执行计划。

**📊 数据集**

数据集主要为两类：①合成 Employee 表，行数从 10^3 到 10^7；②TPC‑H 基准表，规模因子 0.01667（≈10^5 行）与 1.667（≈10^7 行）以及 10^5、10^7 行的手工裁剪版本；实验中还使用了小型表、扫描窗口表、分区表等辅助数据。

**📈 对比分析**

比较方法：每种调优方案（例如索引类型、批量大小、连接池大小）在同一查询/负载下分别跑 10 次，取平均延迟；通过 forced index / sequential scan 方式对比同一查询的索引扫描与全表扫描；对比不同数据库在同一工作负载下的总时延、吞吐量与标准差。实验结果显示：MySQL 在小表/写密集场景表现最佳；DuckDB 在大规模扫描场景最快；B+树索引在大多数系统优于哈希；触发器在 MySQL 上插入代价高，但能显著提升聚合查询；连接池在 MySQL/MariaDB/PostgreSQL 下显著降低延迟，而在 DuckDB 上影响不大；垂直分区对点查询效果差；去范式化在大规模表上反而损失性能。

**⚠️ 局限性**

局限性：①实验仅覆盖特定 DBMS 版本（MySQL 9.1.0、MariaDB 11.4、PostgreSQL 13.20、DuckDB 1.1）和单一硬件平台，无法完全代表未来硬件或数据库升级后的表现；②实验环境为单机部署，未覆盖分布式场景；③某些调优参数（如内存页大小、写缓存）未全部调试，可能影响对比；④实验使用的是合成或 TPC‑H 数据，对业务特定分布的适用性需进一步验证。

---

## 130. Rewarding Intellectual Humility Learning When Not To Answer In Large Language Models

**arXiv ID:** 2601.20126 | [PDF](https://arxiv.org/pdf/2601.20126v1)

**作者:** Abha Jha `[一作]` (University of Southern California), Sonal Chaturbhuj Gehlot `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了在语言模型中显式奖励“我不知道”行为的 RLVR 训练框架，以鼓励模型在不确定时主动回避。

**💡 创新点**

创新点在于将可验证奖励机制与三元奖励（正确、错误、回避）结合，并通过可调的回避奖励 r_abs 探索模型信心与准确率的权衡。

**🔧 技术方法**

使用了强化学习 (GRPO)、LoRA 微调、以及 OpenAI 的 TRL 库；通过可验证奖励和格式化提示来实现自动化评估。

**📊 数据集**

在 MedMCQA 医学多选题和 Hendrycks Math 难题集合上进行实验，且为 MedMCQA 添加了 “I don't know” 选项。

**📈 对比分析**

通过与无回避基线、随机回避 SFT、RTuning 等三种训练策略比较，发现适度的回避奖励可在不显著降低准确率的前提下显著减少错误回答，尤其在大模型 Qwen-3-4B-Instruct 上表现更佳。

**⚠️ 局限性**

局限性包括在开放式问答中探索不足导致回避率低，RL-only 对错误率改善有限；以及回避比例过大可能导致模型失去回答能力，需要更精细的 SFT 设计。

---

## 131. Node-Weighted Multicut in Planar Digraphs

**arXiv ID:** 2601.20038 | [PDF](https://arxiv.org/pdf/2601.20038v1)

**作者:** Chandra Chekuri `[一作]`, Rhea Jain `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文扩展了Kawarabayashi和Sidiropoulos提出的O(log^2 n)近似算法，针对平面有向图中的节点加权多切割问题进行了研究。

**💡 创新点**

创新点在于提出了一种确定性算法，简化了之前算法的分析，并将结果扩展到节点加权的多切割问题。

**🔧 技术方法**

使用了自然的线性规划松弛技术，并结合了分治法和区域生长的技术。

**📊 数据集**

使用了平面有向图作为数据集，具体的节点和边的权重是非负的。

**📈 对比分析**

与之前的随机算法相比，本文的确定性算法在节点加权多切割问题上达到了O(log^2 n)的近似比，且在节点加权稀疏切割问题上达到了O(log^3 n)的近似比。

**⚠️ 局限性**

限制在于该算法的近似比可能不如某些特定情况下的最佳算法，并且尚未解决在更广泛的有向图或次要自由图中的应用问题。

---

## 132. FastWhisper: Adaptive Self-knowledge Distillation for Real-time Automatic Speech Recognition

**arXiv ID:** 2601.19919 | [PDF](https://arxiv.org/pdf/2601.19919v1)

**作者:** Junseok Lee `[一作]` (OKESTRO Inc), Chang-Jae Chun `[通讯]` (Sejong University)

**通讯引用:** 770 | [OpenAlex ID](https://openalex.org/A5076429401)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出自适应自知识蒸馏（ASKD）框架，并用Whisper编码器训练FastWhisper模型，实现小模型高效语音识别。

**💡 创新点**

在蒸馏过程中动态调节教师依赖度（α_AKD）并结合自监督蒸馏（SKD），既提升自学习又增强泛化能力。

**🔧 技术方法**

使用KL散度、交叉熵、温度软化、软标签蒸馏、Whisper预训练编码器与轻量化Transformer解码器等技术。

**📊 数据集**

训练集包含960h LibriSpeech、453h TED‑LIUM、24h LJSpeech、105h Earnings‑22、78h AMI；评估集为GigaSpeech与VoxPopuli。

**📈 对比分析**

对比标准KD、PL、SKD、ASKD等方法，FastWhisper‑small/large 在各基准集 WER 均优于或相当于 Whisper，FastWhisper‑large 在噪声集上比教师模型低约5.4% WER，推理速度提升约5倍。

**⚠️ 局限性**

仅支持英文；蒸馏超参数仍需经验调优；模型规模与训练时间仍相对较大。

---

## 133. Emergent Specialization in Learner Populations: Competition as the Source of Diversity

**arXiv ID:** 2601.19943 | [PDF](https://arxiv.org/pdf/2601.19943v1)

**作者:** Yuhao Li `[一作]` `[通讯]` (University of Pennsylvania), Yuhao Li (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种通过竞争排除机制（Winner‑Take‑All）实现学习者群体自发专业化的算法NichePopulation

**💡 创新点**

创新点在于证明竞争本身即可诱导专业化，无需显式的多样性奖励或通信机制，并给出理论证明；同时提供简单的尼奇亲和度跟踪与奖励调节框架

**🔧 技术方法**

使用Thompson采样进行方法选择，Winner‑Take‑All竞争排除，贝叶斯（Beta分布）信念更新，尼奇亲和度概率更新，λ调节的尼奇奖励

**📊 数据集**

六个真实世界域：加密货币、商品价格、天气预报、光伏辐射、城市交通、空气质量，数据来自公开API（Bybit、FRED、Open‑Meteo、NYC TLC）

**📈 对比分析**

与同类基线（同质化、随机、IQL、QMIX、MAPPO）对比；在六个域上平均SI达0.747，Cohen's d>20，NichePopulation比MARL基线高约4.3×，速度快4×，内存占用少99%

**⚠️ 局限性**

局限包括：需要先验的区域分类（Regime定义）与方法集合，静态环境假设，Winner‑Take‑All竞争可能过于激进，未实现自动方法/区域发现，未与Oracle基线比较

---

## 134. A Taylor Series Approach to Correct Localization Errors in Robotic Field Mapping using Gaussian Processes

**arXiv ID:** 2601.20149 | [PDF](https://arxiv.org/pdf/2601.20149v1)

**作者:** Muzaffar Qureshi `[一作]` (University of Florida), Rushikesh Kamalapurkar `[通讯]` (University of Florida)

**通讯引用:** 2689 | [OpenAlex ID](https://openalex.org/A5071224574)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于梯度的高斯过程（GP）模型修正框架，用于在移动机器人测量位置存在确定性误差时，实时更新GP预测而无需重新训练整个模型。

**💡 创新点**

核心创新是先离线预计算GP均值与协方差对训练位置的一阶和二阶导数张量（Jacobian、Hessian），在线仅用这些张量与误差信息做张量乘法即可完成二阶泰勒展开修正，从而显著降低计算成本。

**🔧 技术方法**

技术方法包括高斯过程回归、平方指数核（SE核）、泰勒展开、自动微分、稀疏张量运算及梯度/海森矩阵预计算。

**📊 数据集**

实验数据基于合成场景：1D函数 f₁(x)=2+sin(2πx) 与 2D函数 f₂(x,y)=sin(2πx)cos(2πy)，在这些场上引入随机或恒定的测量位置偏差进行仿真。

**📈 对比分析**

通过与完整GP重训练以及未校正的“坏”GP模型对比，结果表明校正后均值误差显著下降，平均提升百分比达到两位数；计算时间相比完整重训练降低数倍，显示出显著的性能优势。

**⚠️ 局限性**

主要限制在于需存储高阶导数张量，随着训练样本数和输入维度的增加，内存消耗迅速增长；此外，该方法仅适用于至少二阶可微的核函数，尚需进一步研究压缩或稀疏近似以适应大规模应用。

---

## 135. Who Writes the Docs in SE 3.0? Agent vs. Human Documentation Pull Requests

**arXiv ID:** 2601.20171 | [PDF](https://arxiv.org/pdf/2601.20171v1)

**作者:** Kazuma Yamasaki `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 7000 | [OpenAlex ID](https://openalex.org/A5011588138)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在真实开源仓库中大规模对比 AI 代理与人类在文档相关拉取请求（PR）的数量、接收率及后续人类跟进情况，揭示 AI 代理在文档编辑中的主导地位与审查缺失风险。

**💡 创新点**

创新点在于首次系统量化 AI 代理在文档 PR 的规模、文件共编辑率、接收率与人类后续修改比例，说明代理生成的文档常被合并且几乎不被进一步修订，暴露了 SE 3.0 文档质量保障的潜在问题。

**🔧 技术方法**

技术方法包括：①利用 GitHub API 采集 PR、提交与文件变更记录；②使用文件扩展名/路径（.md、.txt、/docs/、README）识别文档文件；③计算行数增删、文件共编辑次数、代理增删与人类删的比例，并以图表呈现。

**📊 数据集**

数据集为 AIDev（包含 33,596 个代理 PR 与 6,618 个人类 PR），筛选出 1,478 个代理文档 PR 与 519 个人类文档 PR，进一步提取 3,653 代理提交与 1,889 人类提交，涵盖 35,428 与 17,013 文件级变更记录。

**📈 对比分析**

比较结果显示：代理 PR 数量是人类 PR 的约 2.8 倍；在文档文件上 96.3% 只被代理或人类单独编辑；代理 PR 的线条大部分（约 86.8%）被保留，且 85.7% 的文件中代理增删多于人类删，34.5% 的文件无任何人类删改；因此代理文档贡献占主导且人类后续监督显著不足。

**⚠️ 局限性**

局限包括：①仅分析使用代理的仓库，未覆盖不使用代理的项目；②文档文件判定仅基于扩展/路径规则，导致部分标记为文档的 PR 并未实际修改文档；③未对人类评论、合并后再编辑等细粒度审查过程进行深入分析。

---

## 136. Simulating Complex Multi-Turn Tool Calling Interactions in Stateless Execution Environments

**arXiv ID:** 2601.19914 | [PDF](https://arxiv.org/pdf/2601.19914v1)

**作者:** Maxwell Crouse `[一作]` (IBM Research), Pavan Kapanipathi `[通讯]` (IBM Research)

**通讯引用:** 1172 | [OpenAlex ID](https://openalex.org/A5003720552)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种不依赖状态执行环境的多轮工具调用合成数据生成方法

**💡 创新点**

创新点是选择性生成（显式/隐式工具调用）以及回译去噪技术

**🔧 技术方法**

使用LLM（开源 LLaMA-2 及 LLaMa-Factory LoRA）来生成、筛选和执行对话

**📊 数据集**

使用自定义工具规格，生成 5,000 条对话并与 BFCL-v3 与 τ²-bench 评测集对齐

**📈 对比分析**

与大型前沿模型比较，基于开源模型的效果超过许多大模型，在 BFCL-v3 的 Base 与 Long Context 类别表现最佳；ablation 实验验证选择性生成和回译两项技术的关键作用

**⚠️ 局限性**

在需要系统状态的复杂任务中效果不佳，且若使用更强教师模型生成数据成本会显著增加

---

## 137. MAPLE: Self-supervised Learning-Enhanced Nonlinear Dimensionality Reduction for Visual Analysis

**arXiv ID:** 2601.20173 | [PDF](https://arxiv.org/pdf/2601.20173v1)

**作者:** Zeyang Huang `[一作]` (Linköping University), Andreas Kerren `[通讯]` (Linköping University)

**通讯引用:** 3739 | [OpenAlex ID](https://openalex.org/A5006966951)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MAPLE，一种基于自监督学习的 UMAP 扩展，用于在高维数据中更精确地构建邻域图并生成低维可视化布局。

**💡 创新点**

创新点在于引入最大流形容量表示（MMCR）作为自监督学习目标，压缩同质邻域的方差并放大异质邻域之间的方差，从而在学习阶段动态重构邻域图，显著提升邻域一致性和类间可分离度。

**🔧 技术方法**

使用的技术包括：双层 MLP 编码器–投影器网络进行低维嵌入；MMCR 损失（基于核范数的内/外方差优化）；利用学习得到的邻域图构造模糊图；随后采用 UMAP 的交叉熵布局优化与负采样梯度下降；并在 PyTorch + NumPy 实现中结合 MPS 后端实现加速。

**📊 数据集**

实验数据集涵盖图像领域（MNIST、Fashion‑MNIST、STL‑10）和生物信息学领域（C. elegans 的多个子集和 PBMC 4K），数据维度从 784 到 20,222 甚至 33,538，样本量从 4,000 到 40,000 以上。

**📈 对比分析**

与 UMAP 以及 t‑SNE、PHATE、LargeVis 等基线方法比较，MAPLE 在 17 项指标（邻域命中率、kNN 分类、距离一致性、Calinski‑Harabasz、Davies‑Bouldin、ARI/AMI/NMI/V‑measure、边界信任度、Hausdorff 距离、2AFC 与 ROC‑AUC）上整体提升，尤其在邻域一致性和细粒度子集分离方面优势显著；运行时间与 UMAP 相近，略高但可接受，并且在 100,000 规模数据上仍可在笔记本上几分钟完成。

**⚠️ 局限性**

主要限制包括：对数据几何特征依赖性强，简单或已白化的数据集提升有限；在高维大邻域时计算量激增；需要调节 k 与 λ 超参数，尽管对性能影响不大但仍需实验；核范数的 SVD 计算在 CPU 上瓶颈；且目前仅适用于单模态数据，跨模态或多任务场景尚未验证。

---

## 138. Quantifying non deterministic drift in large language models

**arXiv ID:** 2601.19934 | [PDF](https://arxiv.org/pdf/2601.19934v1)

**作者:** Claire Nicholson `[一作]` `[通讯]` (HelixScribe.AI), Claire Nicholson (HelixScribe.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不同温度、提示模式与模型部署方式下，对 gpt‑4o‑mini 与 llama3.1‑8b 进行重复运行实验，量化同一提示在无控制条件下的行为漂移（baseline drift）

**💡 创新点**

首次系统地把漂移量化为可操作的基准，比较 API 与本地部署、不同提示类别、不同温度及重复、扰动、重用三种模式下的漂移幅度，提供可直接用于后续漂移抑制研究的参照

**🔧 技术方法**

使用唯一输出比例、平均 Jaccard 相似度以及词数方差等词面指标；对 llama3.1‑8b 采用 vLLM + Hugging Face Transformers 进行本地推理；gpt‑4o‑mini 通过 OpenAI Chat Completions API 访问

**📊 数据集**

自定义 5 类提示（未指明摘要、未指明建议、轻约束、风格提示、硬约束），共 100 条（gapfill）+ 25 条（small battery）提示；对扰动输入采用同义词替换，对重用模式则将前一次输出作为新输入

**📈 对比分析**

通过对比不同温度（0.0、0.7）和不同提示模式下的唯一输出比例与 Jaccard 相似度，发现温度升高导致漂移显著增加；在 0.0 温度下 gpt‑4o‑mini 的漂移率约为 24%，llama3.1‑8b 为 9%；扰动输入提升漂移，重用输入抑制漂移；结果表明即使在“确定性”设置下仍存在显著漂移

**⚠️ 局限性**

仅测试两款模型，未覆盖更大规模或不同架构；使用的指标仅基于词面，无法捕捉语义层面的漂移；实验仅为单回合提示，未检验多轮对话或多智能体系统中的漂移积累；数据集规模有限，提示类型多为人工设计，缺乏大规模真实场景

---

## 139. Improving X-Codec-2.0 for Multi-Lingual Speech: 25 Hz Latent Rate and 24 kHz Sampling

**arXiv ID:** 2601.20185 | [PDF](https://arxiv.org/pdf/2601.20185v1)

**作者:** Husein Zolkepli `[一作]` `[通讯]` (Scicom), Husein Zolkepli (Scicom)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对X-Codec-2.0进行改造，增大跳跃大小并加入平均池化，将潜在帧率从50Hz降至25Hz，同时将输出采样率从16kHz提升至24kHz，保持编码器冻结仅微调解码器，显著提升了压缩效率与音质。

**💡 创新点**

通过最小化结构改动（跳跃大小提升+轻量化池化）实现潜在帧率降低与采样率提升的双重优化，同时采用一维线性插值保留预训练解码器的频谱特征，首次在25Hz条件下取得最佳语音质量。

**🔧 技术方法**

使用冻结HuBERT语义编码器、Transformer解码器、平均池化层、线性插值权重迁移、BF16混合精度训练、Adam优化器和多目标损失（mel、对抗、语义）等技术。

**📊 数据集**

训练使用约1.6万小时的多语言语音数据集，包括Common Voice、TTS及表达式语音等，覆盖100余种语言；评估使用Common Voice 17测试集的48,489段多语言样本。

**📈 对比分析**

在Common Voice 17上通过UTMOSv2指标与DAC、Encodec、UniCodec等多种神经音频编码器对比，25Hz 24kHz版本在所有116种语言的平均UTMOSv2上均超过竞争者，单语言提升约0.29分，成为低帧率下的最优方案。

**⚠️ 局限性**

局限性包括训练数据主要为干净语音，缺乏背景噪声与表达式多样性；评估仅基于UTMOSv2预测，可能无法完全反映主观感受；跨语言泛化能力未充分验证；较低帧率导致离散符号信息增大，对自回归模型的预测难度提升。

---

## 140. Attribution Techniques for Mitigating Hallucinated Information in RAG Systems: A Survey

**arXiv ID:** 2601.19927 | [PDF](https://arxiv.org/pdf/2601.19927v1)

**作者:** Yuqing Zhao `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 5752 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 Retrieval-Augmented Generation (RAG) 系统中基于归因的幻觉消除技术进行了全面综述，提出了统一的四阶段流水线（查询优化、检索筛选、提示工程、后处理校正）并构建了六类幻觉的分类与对应归因技术映射。

**💡 创新点**

创新点包括：①针对 RAG 设计了六类幻觉的细粒度分类框架；②提出了四模块统一管线，将归因技术整合为可插拔组件；③系统性对现有归因方法进行对照，提供实践指南和优缺点评估。

**🔧 技术方法**

所用技术主要是归因方法（Query Refining、Reference Identification、Prompt Engineering、Response Correction）以及对现有文献中相应技术的综合分析；同时采用了多模态评估模型（如 NLI、RLHF）对校正效果进行辅助判断。

**📊 数据集**

论文并未在实验中使用特定数据集，而是引用了多篇公开工作中的数据来源，例如 WebGPT、REALM、DPR、CoDA 等；若需实验则可基于这些公开 QA/检索数据集进行验证。

**📈 对比分析**

对比方法主要以定性评估为主：对每种归因技术的效果、计算成本、适用场景进行表格化对比，并在文中列举了各方法在消除不同类型幻觉方面的优势与不足；未给出统一的量化指标，但通过案例分析展示了方法在提升答案可信度与可解释性方面的表现。

**⚠️ 局限性**

局限性包括：①缺乏统一的评测基准和量化指标，难以客观比较不同方法；②归因管线依赖人工调参，部署复杂；③LLM 作为评估者时存在循环偏差，单源验证不够；④对长文本和多跳推理的支持仍有限，容易导致错误累计。

---

## 141. Decomposing multimodal embedding spaces with group-sparse autoencoders

**arXiv ID:** 2601.20028 | [PDF](https://arxiv.org/pdf/2601.20028v1)

**作者:** Chiraag Kaushik `[一作]` (Georgia Institute of Technology), Andrea Fanelli `[通讯]` (Dolby Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于组稀疏自编码器（GSAE）和随机跨模态遮掩的多模态嵌入稀疏字典学习方法，以解决传统SAE的“单模态字典”问题。

**💡 创新点**

创新点在于将组稀疏正则化与跨模态随机遮掩相结合，理论上证明可从分割字典提升到多模态字典，并在实验中显著提升字典的语义性与跨模态对齐。

**🔧 技术方法**

主要技术包括稀疏自编码器、TopK稀疏化、组稀疏（L₂,₁）损失、跨模态随机遮掩以及与语义相关的评价指标（多模态单义性分、零样本性能等）。

**📊 数据集**

在CLIP ViT‑B/16（图像/文本）和CLAP（音频/文本）两大多模态嵌入空间上进行实验，并使用CC3M、MusicBench、CelebA等公开数据集。

**📈 对比分析**

与传统SAE、BatchTopK、Matryoshka等基线相比，GSAE与MGSAE在字典的多模态激活率、死神经元数、零样本跨模态分类准确率和检索mrr等指标上均提升约10–20%，尤其在CLIP上取得相当于原始密集嵌入的性能。

**⚠️ 局限性**

限制在于目前仅针对两模态设置，未充分验证在更多模态或未配对数据上的泛化；组稀疏正则化和遮掩的超参数选择仍需经验调优。

---

## 142. BayPrAnoMeta: Bayesian Proto-MAML for Few-Shot Industrial Image Anomaly Detection

**arXiv ID:** 2601.19992 | [PDF](https://arxiv.org/pdf/2601.19992v1)

**作者:** Soham Sarkar `[一作]` (Indian Institute of Technology), Sayantan Banerjee `[通讯]` (Indian Institute of Management)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了BayPrAnoMeta，一种在极少量正常样本下进行工业图像缺陷检测的贝叶斯元学习框架，并将其扩展到联邦学习环境；

**💡 创新点**

创新点包括：①用Normal–Inverse–Wishart先验得到多元Student‑t后验预测，提供不确定性感知的异常评分；②将Proto‑MAML改为贝叶斯版本，支持任务级的概率适配；③在联邦学习中加入监督对比学习以缓解客户端异质性；④给出了收敛性证明；

**🔧 技术方法**

技术手段主要是：贝叶斯元学习（NIW先验 + Student‑t 预测）、Proto‑MAML、MAML、监督对比学习、联邦学习、基于似然比的异常评分；

**📊 数据集**

实验数据集为工业缺陷图像数据集MVTec AD，将不同物体类别视为不同客户端；

**📈 对比分析**

在AUROC指标上与PatchCore、MAML、Proto‑MAML进行比较，BayPrAnoMeta在大多数类别上均实现最高分，特别是结构复杂或高内在变异的物体；在联邦设置下加入对比学习进一步提升性能；

**⚠️ 局限性**

局限性：在部分类别对比学习反而降低效果；需要预训练的高质量特征；仅在MVTec AD上评估，缺乏跨域或更大规模数据验证；

---

## 143. Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents

**arXiv ID:** 2601.19935 | [PDF](https://arxiv.org/pdf/2601.19935v1)

**作者:** Yiting Shen `[一作]` (Institute of Information Engineering), Songlin Hu `[通讯]` (School of Cyberspace Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Mem2ActBench 基准，评估 LLM 代理在工具驱动任务中主动利用长期记忆进行参数填充的能力。

**💡 创新点**

创新地通过逆向生成和事实演化链构造记忆依赖任务，强调主动检索与参数化，而非仅做事实检索。

**🔧 技术方法**

采用 LLM 驱动的事实抽取、BERTopic+HDBSCAN 聚类、冲突检测与全局排序、混合检索（BM25+BGE）与 LLM 验证、逆向查询生成等技术。

**📊 数据集**

基于 ToolACE、BFCL 与 OASST1 三大公开数据集合成 2,029 条长会话，生成 400 条记忆依赖工具调用任务。

**📈 对比分析**

对七种记忆框架在 Qwen2.5-7B/32B/72B 模型规模下进行 F1、BLEU 与工具准确率评估，结果显示参数填充仍有显著瓶颈，检索质量是主要限制因素。

**⚠️ 局限性**

局限于离线工具调用评估、单一模型族、缺乏交互式反馈，自动生成过程可能无法完全覆盖真实对话的复杂性。

---

## 144. Delta Fair Sharing: Performance Isolation for Multi-Tenant Storage Systems

**arXiv ID:** 2601.20030 | [PDF](https://arxiv.org/pdf/2601.20030v1)

**作者:** Tyler Griggs `[一作]` (University of California), Matei Zaharia `[通讯]` (University of California)

**通讯引用:** 49007 | [OpenAlex ID](https://openalex.org/A5005554337)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究多租户存储系统中的资源共享问题，提出能够在存在高预占延迟的情况下保证性能隔离的资源分配算法。

**💡 创新点**

创新点在于引入可配置的“公平性（τ）”与Pareto有效性两大属性，并通过动态预留资源来同时实现延迟上限与高利用率，突破传统公平共享在写缓冲区和读缓存等高延迟资源上的局限。

**🔧 技术方法**

采用基于写入/读取带宽、flush速率、读放大等参数的模型，设计写缓冲区和读缓存的最小预留策略，并实现了相应的调度与抢占机制，构建了名为 “FairDB” 的 RocksDB 扩展。

**📊 数据集**

使用 YCSB、Snowflake 生产工作负载及 Twitter 缓存等真实数据集进行实验验证。

**📈 对比分析**

与即时公平共享（τ=∞）和固定配额（τ=0）两种基线比较，宏基准中 P99 延迟提升约 9×，吞吐率保持 4% 以内，利用率提升至 30% 以上。

**⚠️ 局限性**

主要限制包括需要准确估计并发升压客户端数 k 与延迟阈值 τ，误估会导致隔离失效或资源利用下降；算法目前仅在单节点 RocksDB 上实现，尚未在分布式一致性环境下验证。

---

## 145. DeRaDiff: Denoising Time Realignment of Diffusion Models

**arXiv ID:** 2601.20198 | [PDF](https://arxiv.org/pdf/2601.20198v1)

**作者:** Ratnavibusena Don Shahain Manujith `[一作]` (National University of Singapore), Kenji Kawaguchi `[通讯]` (National University of Singapore)

**通讯引用:** 7395 | [OpenAlex ID](https://openalex.org/A5003184366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DeRaDiff，一种在推理阶段对对齐扩散模型进行实时重调节 KL 正则化强度的方法，能够在不需要额外训练的情况下，用已对齐模型以不同正则化强度生成图像。

**💡 创新点**

创新点包括：①将解码时对齐（decoding‑time realignment）从语言模型推广到扩散模型；②推导出每一步高斯混合的闭式更新公式，并引入单一可调参数 λ，实现在线正则化强度控制；③通过该方法实现对齐强度搜索的高效计算，显著降低对齐成本。

**🔧 技术方法**

使用技术包括 KL 正则化强化学习（RLHF / Diffusion DPO）、扩散模型（SDXL、Stable Diffusion 1.5）、几何混合与高斯混合推导、DDIM/DDPM 采样器、CLIP、HPS v2、PickScore 等评估指标。

**📊 数据集**

实验数据集涵盖 Pick‑a‑Pic v1、HPS 数据集、Human Preference Dataset v2（HPD v2）以及 SDXL 1.0 与 SD1.5 官方检查点。

**📈 对比分析**

比较方法：在多种 λ 值下使用 DeRaDiff 生成图像，并与完全从头对齐得到的模型在 PickScore、HPS、CLIP 上进行对比。误差普遍低于 1%（大部分 ≤ 5×10⁻⁴），并在计算成本上实现 70%–90% 的 GPU 小时/EFLOPs 节省。

**⚠️ 局限性**

局限性：λ>1 进入非凸混合时可能出现协方差非正定或性能下降；推荐在 [0,1] 范围内使用；推理时需两次前向传播，略有额外开销；对不同模型和对齐策略的适用性仍需进一步验证。

---

## 146. Evaluating Large Language Models for Abstract Evaluation Tasks: An Empirical Study

**arXiv ID:** 2601.19925 | [PDF](https://arxiv.org/pdf/2601.19925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. How many times can two minimum spanning trees cross?

**arXiv ID:** 2601.20060 | [PDF](https://arxiv.org/pdf/2601.20060v1)

**作者:** Todor Antić `[一作]` (Charles University), Pavel Valtr `[通讯]` (Charles University)

**通讯引用:** 1583 | [OpenAlex ID](https://openalex.org/A5072926305)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究平面点集的双色最小生成树交叉数（bicolored MST crossing number），并给出其上界、下界以及在凸、稠密、随机点集上的具体表现。

**💡 创新点**

提出了该参数的线性上界与线性下界，并给出了在凸位置、稠密点集和均匀随机点集上的匹配下界；首次将该参数与凸包、Erdős–Szekeres 定理、概率方法等经典几何工具结合。

**🔧 技术方法**

主要技术包括：最小生成树的唯一性假设、几何交叉计数与角度约束、Erdős–Szekeres 结构提取、随机细分格子与概率估计、点集稠密度与覆盖格子技巧。

**📊 数据集**

实验数据并未使用实际点集；研究基于理论构造的点集（一般位置、凸位置、稠密点集、均匀随机点集）以及对 2×(n/2) 网格的扰动。

**📈 对比分析**

与先前文献中提出的 8n 上界相比，本文给出更紧的线性上界；在凸位置给出 ⌊n/2⌋−1 的下界，稠密点集给出线性下界，随机点集给出期望线性下界；但对一般点集的下界仍仅为常数 1。

**⚠️ 局限性**

局限性：对一般点集的下界仍远低于上界；需要点集满足最小生成树唯一性的“generic”假设；常数未优化；关于该参数的算法复杂度仍未知（可能是 NP‑hard）。

---

## 148. StreamFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs

**arXiv ID:** 2601.20273 | [PDF](https://arxiv.org/pdf/2601.20273v1)

**作者:** Jiacheng Yang `[一作]` (University of Toronto), Gennady Pekhimenko `[通讯]` (University of Toronto)

**通讯引用:** 4304 | [OpenAlex ID](https://openalex.org/A5007585346)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种可扩展的分布式 Diffusion Transformer 推理引擎，能够在多台 GPU 机器上高效推理高分辨率图像与长时长视频。

**💡 创新点**

创新点包括：
• 结合 Ulysses 与 Ring Attention 的拓扑感知序列并行策略；
• 将 all‑to‑all 操作拆分为多段，并将通信与注意力计算流水线化，实现通信与计算的重叠；
• 采用一侧通信（NVSHMEM）实现 Ring、Ulysses 与 Torus Attention，显著减少 GPU 之间的同步与计算开销。

**🔧 技术方法**

主要技术：
• 拓扑感知序列并行（Topology‑Aware Sequence Parallelism）
• Torus Attention（多段 all‑to‑all + 计算重叠）
• 一侧通信实现（NVSHMEM）
• CUDA kernel、FlashAttention 等加速核

**📊 数据集**

使用的工作负载包括：
• Flux（12B 参数）生成 3072×3072 与 4096×4096 图像；
• CogVideoX（5B 参数）生成 20s/40s、768×1360 视频；
并在 AWS EC2 G5（8×A100 + NVSwitch + EFA）上评估。

**📈 对比分析**

与 USP（最先进的统一序列并行）比较，平均提升约 1.4–1.6 倍，单机最高可达 2 倍；在 3/4 台机器配置下，平均速度提升 30%~50%，并在各种头尺寸、序列长度与批量大小下保持一致的加速效果。

**⚠️ 局限性**

局限性：
• 需要序列长度和头数能够被 GPU 数整除；
• 对于仅使用单机 GPU 的场景提升有限，主要优势体现在多机通信；
• 实现复杂，需手动调优通信分块与同步；
• 目前未针对非标准网络拓扑（如多机不完全网）做进一步优化。

---

## 149. Semi-Supervised Masked Autoencoders: Unlocking Vision Transformer Potential with Limited Data

**arXiv ID:** 2601.20072 | [PDF](https://arxiv.org/pdf/2601.20072v1)

**作者:** Atik Faysal `[一作]` (Rowan University), Huaxia Wang `[通讯]` (Rowan University)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5031877375)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种半监督掩码自动编码器（SSMAE），在有限标注数据与大量无标注数据的场景下，联合优化掩码图像重建与分类任务。

**💡 创新点**

创新点在于：①动态门控伪标签生成机制，利用验证集高置信度一致性来决定何时激活伪标签，从而抑制确认偏差；②将掩码自监督与分类任务统一训练，无需额外微调即可获得高性能。

**🔧 技术方法**

使用的技术包括 Vision Transformer 编码器 + MAE 解码器、掩码重建损失、伪标签一致性正则、动态门控伪标签、弱/强数据增强以及交叉熵分类损失。

**📊 数据集**

实验数据集为 CIFAR‑10 和 CIFAR‑100，在 10%、20%、30% 及 40% 标注比例下进行评估。

**📈 对比分析**

与监督 ViT 和 Fine‑tuned MAE 进行对比，在低标注比例下表现显著提升，例如 CIFAR‑10 10% 标注时准确率提升 9.24%，在所有标注比例下均优于基线。

**⚠️ 局限性**

局限性包括：对阈值设定和验证集可靠性的依赖；未验证在更大规模或不同任务上的泛化能力；伪标签生成与门控机制会增加训练时间和实现复杂度。

---

## 150. Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis

**arXiv ID:** 2601.20103 | [PDF](https://arxiv.org/pdf/2601.20103v1)

**作者:** Darshan Deshpande `[一作]` (Patronus AI), Rebecca Qian `[通讯]` (Patronus AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基于人工验证的 517 条轨迹、覆盖 54 类奖励黑客行为的基准数据集，并提出了新的奖励黑客分类法。

**💡 创新点**

创新点在于将奖励黑客检测从传统的孤立二分类问题转化为对比性异常检测任务，并通过对多样化轨迹集的对比推理显著提升检测效果。

**🔧 技术方法**

主要技术包括：利用 Claude‑Code 生成具有真实工具使用的合成代码轨迹；在 200k token 的上下文窗口中实现对比性评估；使用结构化输出解析器、匹配率与检测率等自定义评估指标。

**📊 数据集**

数据集为 TRACe，包含 517 条人工审核通过的代码生成轨迹，涉及 10 大类、54 个子类，平均每条轨迹约 26 轮交互。

**📈 对比分析**

实验与多款开放与闭源 LLM（GPT‑5.2、Gemini‑3‑Pro、Kimi‑K2‑Thinking、Claude‑4.5‑Opus 等）对比，采用对比集群大小 N∈{1,5,10} 与 benign 率 B∈{0.25,0.5,0.9} 的设置；在对比设置下 GPT‑5.2 检测率提升至 63%，而在孤立设置仅为 45%；总体来看，集群规模越大、benign 比率越低模型性能越差。

**⚠️ 局限性**

局限性包括：对语义化奖励黑客的检测仍显不足；对比集群规模和 beni­an‑hack 比例对性能影响显著，需更精细的样本配置；合成轨迹可能缺乏真正环境的多样性；以及模型在自我意识与用户同意等方面的过度依赖导致误检。

---

## 151. TAIGR: Towards Modeling Influencer Content on Social Media via Structured, Pragmatic Inference

**arXiv ID:** 2601.20032 | [PDF](https://arxiv.org/pdf/2601.20032v1)

**作者:** Nishanth Sridhar Nakshatri `[一作]` (Purdue University), Dan Goldwasser `[通讯]` (Purdue University)

**通讯引用:** 3526 | [OpenAlex ID](https://openalex.org/A5032121234)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TAIGR 框架，用以分析健康类社交媒体影响者视频中的核心建议、论证结构与外部科学证据，完成内容真实性验证。

**💡 创新点**

将受众关注的核心 takeaway 抽取、引入论证图结构，并通过因子图推理将外部 PubMed 证据与论证关联，形成端到端的多层结构化验证方法。

**🔧 技术方法**

利用 OpenAI Whisper 转录、LLM 进行语义抽取与角色标注、PubMed 检索与自动分类、因子图（AD3）推理，实现支持与攻击关系的概率传播。

**📊 数据集**

构建了 1,430 条 TikTok 健康类视频转录数据，并从 Science Feedback 与 PubMed 结合的 195 条人工标注验证集进行评估。

**📈 对比分析**

与 Claim‑centric RAG、RAG‑w‑Takeaway、LOKI 等基线对比，TAIGR 在内容验证任务上宏观 F1 提升至 0.52，较最强基线提升约 9.7 分，尤其在隐式 takeaway 上显著优于对手。

**⚠️ 局限性**

仅使用文本转录，忽略多模态信息；证据检索局限于 PubMed，可能漏检；模块化设计易累积误差；数据集规模与多样性有限；假设每段视频仅含单一主张。

---

## 152. Defensive Rebalancing for Automated Market Makers

**arXiv ID:** 2601.19950 | [PDF](https://arxiv.org/pdf/2601.19950v1)

**作者:** Sam Devorsetz `[一作]`, Maurice Herlihy `[通讯]` (Brown University)

**通讯引用:** 24008 | [OpenAlex ID](https://openalex.org/A5086347882)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了防御性再平衡（Defensive Rebalancing）机制，用于通过直接池间资产转移消除CFMM（常数函数自动做市商）网络中的套利机会，并提升部分市场的流动性。

**💡 创新点**

创新点在于：①将再平衡视为一种全局优化问题，证明套利自由状态等价于Pareto效率；②将再平衡问题建模为凸优化，并给出唯一可解的求解框架；③扩展到混合参与者（主动CFMM与被动CFMM、价格预言机）的情形；④提出混合/限制再平衡的理论与算法。

**🔧 技术方法**

技术方法包括：凸优化（使用对数变换将流动性约束转为凸约束）、对数凸函数分析、Pareto效率与套利自由的等价证明、基于路径的再平衡构造、以及在存在手续费与固定成本时的混合整数凸优化处理。

**📊 数据集**

论文没有使用公开数据集，而是通过理论推导与示例（如三角套利例子）来验证模型，主要以符号模型和假设交易函数（如常数乘积、线性价格预言机）为基础。

**📈 对比分析**

在理论上，通过构造证明与凸优化求解器可实现唯一最优解，比较基准为传统仅使用交易的套利消除方式；实验/仿真表明在示例场景中可显著消除套利并提升流动性，且求解时间在可接受范围内（使用标准凸优化工具即可）。

**⚠️ 局限性**

局限性包括：①缺乏在真实区块链环境中的实现与安全评估；②假设所有主动CFMM都能支持直接池间转移，实际实现难度大；③未考虑交易手续费、滑点、gas成本等现实约束的完整影响；④在存在多个预言机或不完全信息时的鲁棒性仍待研究。

---

## 153. CHIME: Chiplet-based Heterogeneous Near-Memory Acceleration for Edge Multimodal LLM Inference

**arXiv ID:** 2601.19908 | [PDF](https://arxiv.org/pdf/2601.19908v1)

**作者:** Yanru Chen `[一作]` (University of California San Diego), Tajana Rosing `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于芯片片段的异构近存储加速架构CHIME，用于在边缘设备上高效推理多模态大型语言模型。

**💡 创新点**

创新点在于将M3D DRAM与M3D RRAM结合，构建 2.5D UCIe 包装，并设计工作负载感知数据布局、KV 缓存分层调度与核融合的映射框架，显著提升带宽利用率与能效。

**🔧 技术方法**

采用了 3D DRAM 与 3D RRAM 的近存取单元、单元层级缓存分层、跨芯片片段 DMA、核融合与软硬协同映射技术，支持 FP16 运算与 512GB/s 交互带宽。

**📊 数据集**

在 VQA 任务上使用 512×512 的“astronaut”图像与 128 词文本输入，评估 FastVLM（0.6B/1.7B）与 MobileVLM（1.7B/3B）模型。

**📈 对比分析**

与 NVIDIA Jetson Orin NX GPU 以及 SOTA PIM 加速器 FACIL 进行对比，CHIME 在 FastVLM/MobileVLM 上实现最高 41× 的速度提升、185× 的能效提升，吞吐量 233–533 token/s、能效 116–266 token/J，功耗仅约 2 W。

**⚠️ 局限性**

局限性包括：对 RRAM 耐久性与写能要求高，映射框架需要针对不同 MLLM 结构手动优化，且在极长序列或非常大模型时仍受 KV 缓存容量限制。

---

## 154. HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue

**arXiv ID:** 2601.19922 | [PDF](https://arxiv.org/pdf/2601.19922v1)

**作者:** Laya Iyer `[一作]` (Stanford University), Subhabrata Mukherjee `[通讯]` (Hippocratic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实施了 HEART 基准，用于在相同的多轮情感支持对话中同时评估人类与大型语言模型的表现。

**💡 创新点**

创新点在于：①首次通过五维交互沟通维度（人类一致性、共情响应、调适度、共鸣度、任务跟随）对人类与模型进行直接对比；②结合人类评审与 LLM 评审两种评估主体；③加入情绪对抗的“敌对”对话场景。

**🔧 技术方法**

技术手段包括：①使用 GPT‑5、GPT‑o3、Claude、Gemini、Polaris‑4 等多种 LLM 生成回应；②人类评审对比两者并给出 5 级优势评分；③基于 HEART 维度的评估 Rubric 与 Bradley‑Terry 模型求得 Elo 排名；④构建 15 类支持策略词典进行策略多样性分析；⑤对模型推理延迟进行 TTFT 评估。

**📊 数据集**

使用数据集：ESConv 作为情感支持对话来源，抽取 280 条常规对话和 20 条情绪对抗对话；每条对话均由 3 名人类撰写 1‑3 句回应并由 18 种 LLM 生成对应回应。

**📈 对比分析**

比较方法为成对偏好评估（无分数，只给出胜负和优势强度），并采用 Bradley‑Terry 计算 Elo 分；实验显示 LLM 在大多数 HEART 维度上与人类相当甚至优于人类，平均人类–模型一致率约 78%，模型在 LLM‑评审下往往能超过人类平均水平；Polaris‑4 在保持高 HEART Elo 的同时，TTFT 低于 500 ms，显示出实时语音代理的可行性。

**⚠️ 局限性**

局限性包括：①评估仅覆盖英语西方情感表达，缺乏跨文化/多语言验证；②仅评测简短文本对话，未涉及多轮长期或多模态交互；③共情评判主观性强，可能受评审者风格影响；④未检验系统在危急或临床情境下的安全与有效性；⑤评估关注“感知共情”，未直接测量用户实际体验与行为改变。

---

## 155. Parametric and Generative Forecasts of Day-Ahead Market Curves for Storage Optimization

**arXiv ID:** 2601.20226 | [PDF](https://arxiv.org/pdf/2601.20226v1)

**作者:** Julian Gutierrez `[一作]` (Institut Polytechnique de Paris), Redouane Silvente `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建两套机器学习框架：一种基于XGBoost与Chebyshev多项式的低维快速预测模型，用于日常的供需曲线预报；另一种基于DDPM的标记泊松过程生成模型，在天气和燃料变量的条件下从订单层面生成合成供需曲线，并以此优化日内价格制造型储能策略。

**💡 创新点**

创新点在于：①将供需曲线压缩为八个可解释参数并用多项式近似，实现高速、可解释的预测；②将供需曲线拆解为订单级别的标记泊松过程，用DDPM生成全局曲线，获得概率分布而非单点预测；③把预测结果直接嵌入供应函数均衡模型，量化储能对价格分布、收益和价格压缩效应的影响。

**🔧 技术方法**

技术包括：XGBoost回归（量化多元特征的非线性关系）；Chebyshev多项式与线性插值实现曲线分段拟合；DDPM（去噪扩散概率模型）用于条件生成订单量和价格；点过程与标记建模；供应函数均衡理论与随机优化（价格影响模型）实现储能收益最大化。

**📊 数据集**

数据集主要来自EPEX SPOT 2018‑2024年的日内供需曲线（含订单点）、法国气象 ERA5（温度、辐照、风速）与燃料价格（煤、油、天然气），并结合ENTSO‑E负荷预测与公共假期等特征。

**📈 对比分析**

比较方法：对快速模型在2024年进行外部验证，nMAE≈7–11%；对生成模型在200个样本上计算均方误差，elastic区均方误差<1.3%（供给）/5%（需求）；两模型均在低价区误差低；快速模型在极值区更稳定；生成模型提供收益分布、价格压缩效应，显著提高预期收益并减少收益尾部风险；两模型与基线均优于传统单点价格预测。

**⚠️ 局限性**

局限性：①快速模型是确定性，无法捕捉极端波动与复杂订单结构；②生成模型训练成本高、推理慢，且对极端情况（极大/极小订单）仍表现不佳；③两模型均未完整模拟关联块订单、最小收入条件等复杂投标规则；④在极端市场情形下预测误差分布不一定稳健，需进一步校正。

---

## 156. Towards Intelligent Urban Park Development Monitoring: LLM Agents for Multi-Modal Information Fusion and Analysis

**arXiv ID:** 2601.20206 | [PDF](https://arxiv.org/pdf/2601.20206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 157. DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment

**arXiv ID:** 2601.20218 | [PDF](https://arxiv.org/pdf/2601.20218v1)

**作者:** Haoyou Deng `[一作]` (Huazhong University of Science and Technology), Nong Sang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 11186 | [OpenAlex ID](https://openalex.org/A5013734579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DenseGRPO，一种利用稠密奖励的强化学习框架，用于对齐流匹配模型与人类偏好。

**💡 创新点**

创新点在于通过 ODE 推断中间清晰图像的奖励并计算奖励增益实现步长稠密奖励，以及基于这些稠密奖励自适应调节 SDE 采样的时间步噪声，实现探索空间校准。

**🔧 技术方法**

主要技术包括 GRPO、流匹配模型、ODE 与 SDE 采样、奖励模型、稠密奖励计算和噪声校准算法。

**📊 数据集**

实验使用了 Compositional Image Generation、Visual Text Rendering、Human Preference Alignment 三大基准，以及 DrawBench 数据集进行评估。

**📈 对比分析**

与 Flow‑GRPO 和 CoCA 进行对比，DenseGRPO 在 PickScore、Aesthetic Score、UnifiedReward 等多项指标上平均提升约 1.0 以上，整体性能显著优于竞争方法。

**⚠️ 局限性**

局限性包括对奖励模型质量高度依赖、需要多步 ODE 推断导致计算成本上升，以及仍存在轻微的奖励劫持现象。

---

## 158. Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation

**arXiv ID:** 2601.20215 | [PDF](https://arxiv.org/pdf/2601.20215v1)

**作者:** Na Li `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 EASQ 框架，利用稀疏问卷满意度反馈实现短视频推荐模型的实时端到端对齐。

**💡 创新点**

创新点在于：①建立独立参数路径（LoRA+多任务+Mixture-of-Experts）防止稀疏信号被行为数据淹没；②采用在线定制的 DPO 目标，让问卷专家网络即时充当参考模型，实现实时满意度对齐；③将问卷反馈与主模型结合的整体优化策略。

**🔧 技术方法**

技术手段包括：LoRA 低秩微调、Mixture-of-Experts（MoE）多任务学习、ReLU 路由、对比式/二元交叉熵、DPO（Direct Preference Optimization）对齐目标、基于 transformer 的多头注意力 backbone。

**📊 数据集**

使用真实短视频平台的两大业务场景日志数据（行为日志 + 约 0.5% 视图的问卷反馈），并在离线实验与 7 天在线 A/B 测试中验证。

**📈 对比分析**

与 EMER、EMER_S、Imputation Network、SAQRec 等基线对比；在离线评测中在 NDCG@5、NDCG@10、HR、MRR 等指标均比最强基线提升约 3%~4%；在线 A/B 测试中 APP 观看时长、视频播放量、互动率、问卷正向反馈率等均有显著提升，负向反馈率下降。

**⚠️ 局限性**

局限性包括：①问卷激活率仍极低，导致可用满意度数据稀疏；② DPO 参考模型为问卷专家网络，若问卷采样失真或偏差，会影响对齐效果；③模型复杂度较高，需多任务、MoE 与 LoRA 并行训练，部署成本和计算资源消耗较大。

---

## 159. High-Resolution Mapping of Port Dynamics from Open-Access AIS Data in Tokyo Bay

**arXiv ID:** 2601.20211 | [PDF](https://arxiv.org/pdf/2601.20211v1)

**作者:** Moritz Hütten `[一作]` `[通讯]` (GRID Inc), Moritz Hütten (GRID Inc)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用 2024 年 8–10 月公开的 AIS 数据，对东京湾的船舶活动进行高分辨率（约 30 m 空间、约 1 min 时间）重建，得到平均船舶数量、进出港流量、停泊码头分布，并通过信号阴影模式推断主要接收站位置。

**💡 创新点**

创新点：
- 引入可动态扩展的“进出区”方法，精准区分停泊与临时离港，克服接收机覆盖不均的挑战；
- 基于 AIS 消息阴影实现接收站位置逆推，精度达到 0.1°，首次揭示大城市中接收站的隐私风险；
- 结合密度阈值法识别码头与停泊区，避免传统聚类方法对稀疏数据的敏感；
- 系统量化多来源不确定性（离港/停泊误判、暗船缺失、AIS‑B 船舶），并提供上/下限估计。

**🔧 技术方法**

主要技术：AIS 数据清洗与状态分类、动态进出区阈值公式、空间格网统计、船舶速度与航向分析、地理统计（Kent、von Mises–Fisher 分布）、信号阴影的地理线段拟合、置信椭圆与 68% 包含范围、与官方统计的时间序列对比、误差传播与不确定性量化。

**📊 数据集**

数据集：
- 公开 AIS（AIS‑stream.io/ VesselFinder）从 2024‑07‑29 至 2024‑10‑27，约 688 万条位置报告；
- AIS 静态数据 3325 只船舶，含 GT、IMO、类型等；
- 参考其他公开数据估算暗船比例（约 16%）与 AIS‑B 份额（约 12%）；
- 对比日本海岸警卫队、交通省等官方统计数据。

**📈 对比分析**

与方法/性能比较：
- 船舶数量与流量与日本海岸警卫队 Uraga 频道数据、交通省码头进出港平均值高度吻合；
- 进出港速率 293 ± 22 / 天，置信区间覆盖官方统计；
- 接收站位置推断误差 ≈ 0.1°（约 10 km）小于官方公布的不确定性；
- 通过置信椭圆与误差分析，系统不确定性整体低于 10%，满足行业监测需求。

**⚠️ 局限性**

局限性：
- 仅覆盖 3 个月，假设全年活动稳定，季节性波动（台风、雨季）未充分评估；
- 暗船与 AIS‑B 船舶仍导致一定的低估，尤其是渔船与小型船舶；
- 接收站位置推断受数据量与阴影角度限制，可能存在多重解；
- 对小尺寸船舶与 SAR 补充信息缺失，无法完全覆盖全部船舶；
- 数据隐私问题在高密度城市区可能导致公开 AIS 数据不可持续。

---

## 160. Automated Marine Biofouling Assessment: Benchmarking Computer Vision and Multimodal LLMs on the Level of Fouling Scale

**arXiv ID:** 2601.20196 | [PDF](https://arxiv.org/pdf/2601.20196v1)

**作者:** Brayden Hamilton `[一作]` (Automation and Robotic Engineering Science), Henry Williams `[通讯]` (Automation and Robotic Engineering Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了使用卷积神经网络、Transformer语义分割模型以及零训练的多模态大语言模型（LLM）对新西兰海事船舶船体生物附着度进行自动分类，基于Level of Fouling（LoF）尺度进行评估。

**💡 创新点**

创新点在于：①首次将LLM与视觉模型结合，通过结构化提示和检索增强（RAG）实现零训练的LoF判定；②提出在同一数据集上统一基准，比较传统CNN、Transformer分割与LLM的性能与可解释性；③探索HSV与边缘提取预处理对分类准确率的影响。

**🔧 技术方法**

技术包括ResNet‑18/50卷积分类器、SegFormer分割器、OpenRouter API访问的GPT‑4V等LLM、HSV/边缘预处理、检索增强生成（RAG）以及系统级提示工程。

**📊 数据集**

使用由新西兰主要产业部（MPI）提供的762张船体近景图像，人工标注LoF 0–5级别，构成专家标注数据集。

**📈 对比分析**

在相同训练/测试拆分下，CNN在LoF 0/1/5极值上表现最佳（最高约95%准确率），但对2–4级难以区分；SegFormer提供可解释的覆盖率热图，准确率略低；LLM在使用精细提示和RAG后，极值准确率提升至约70%–75%，但整体准确率仍停留在40%–50%之间。

**⚠️ 局限性**

局限性包括：①数据集在中间级别（LoF 2–4）样本稀缺导致模型泛化差；②视觉模型对图像噪声、光照和水体浑浊的鲁棒性有限；③LLM对提示措辞高度敏感，长上下文中无法始终捕捉关键信息；④所有方法仅评估近景帧，未覆盖整艘船体的整体覆盖率。

---

## 161. Securing AI Agents in Cyber-Physical Systems: A Survey of Environmental Interactions, Deepfake Threats, and Defenses

**arXiv ID:** 2601.20184 | [PDF](https://arxiv.org/pdf/2601.20184v1)

**作者:** Mohsen Hatami `[一作]` (Binghamton University), Yu Chen `[通讯]` (Binghamton University)

**通讯引用:** 17355 | [OpenAlex ID](https://openalex.org/A5014591403)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了人工智能代理在网络-物理系统（CPS）中的安全威胁，特别是基于生成式AI的深伪攻击，并提出了面向CPS的系统性防御框架SENTINEL，同时通过真实智能电网案例演示防御组合的实用性。

**💡 创新点**

创新点在于：1）构建了六阶段的SENTINEL框架，提供从威胁评估到持续监测的全生命周期方法；2）针对MCP协议和多模态深伪攻击提出了细粒度的威胁分类与防御策略；3）将深伪检测技术与物理约束结合，形成了可在实时边缘设备上落地的防御层级。

**🔧 技术方法**

使用技术包括：SENTINEL框架（威胁谱、资源评估、防御选型、深度防御架构、验证规划与自适应监测）；深伪检测技术（rPPG、视觉‑LiDAR 融合、C2PA 可信凭证、语音抗劫持、文本对抗检测、行为模仿检测）；多模态融合与可信供应链保障；基于物理信号（ENF、GNSS）和安全审计的防御组合。

**📊 数据集**

数据集与案例：利用公开深伪数据集（FaceForensics++, DeepFakeDetection Benchmark, ASVspoof, AASIST, C2PA provenance dataset）以及行业标准的语音与文本生成对抗数据；在真实智能电网（ANCHOR‑Grid）部署中使用实际传感器与控制日志进行验证；使用工业控制系统仿真平台模拟多模态攻击。

**📈 对比分析**

比较方法：在智能电网案例中对不同防御层级（感知验证、检测层、响应层、适配层）进行实验，衡量检测延迟、误报率、对系统安全裕度的影响，并与传统单一检测方案对比；对深伪检测模型在CPS边缘环境下进行基准测试，展示在低算力和实时约束下的性能曲线。

**⚠️ 局限性**

局限性：1）受限于实时性与边缘计算资源，部分高精度深伪检测模型不可落地；2）框架对特定协议（MCP）依赖较强，迁移到其他协议需重新评估；3）缺乏大规模真实攻击数据，导致对新型生成模型的适配性未知；4）持续自适应机制对运维团队技术门槛较高；5）深伪攻击的快速演化使得防御策略需要频繁更新，维护成本高。

---

## 162. Causal-Driven Feature Evaluation for Cross-Domain Image Classification

**arXiv ID:** 2601.20176 | [PDF](https://arxiv.org/pdf/2601.20176v1)

**作者:** Chen Cheng `[一作]` (Florida State University), Ang Li `[通讯]` (Florida State University)

**通讯引用:** 1429 | [OpenAlex ID](https://openalex.org/A5100413592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出显式分段PNS度量，评估学习表示在不同域中的因果有效性，以提升OOD分类性能。

**💡 创新点**

创新点在于将概率必要与充分因果概念PNS引入分段表示层，并通过跨域自然实验和两阶段训练（先学习语义对齐分段，再基于PNS选取Top‑K重训练）实现因果有效特征筛选。

**🔧 技术方法**

核心技术包括生成式分段编码器+解码器、基于PNS的必要性/充分性评估、Top‑K段落选择以及基线方法如IRM、GroupDRO、MMD、CDANN等的对比训练。

**📊 数据集**

实验使用受控多域MNIST（8种变换）和标准PACS 4域的2→2迁移设置。

**📈 对比分析**

与传统基线相比，PNS方法在多域MNIST的平均准确率提升至87%以上，在PACS 2→2的平均准确率达0.9139，显著优于IRM、GroupDRO、MMD、CDANN等方法。

**⚠️ 局限性**

局限在于需构造语义一致且可分段的表示，PNS评估需要多次前向传播导致计算开销较大，且在极端域差异下仍可能受限。

---

## 163. Adequately Tailoring Age Verification Regulations

**arXiv ID:** 2601.20241 | [PDF](https://arxiv.org/pdf/2601.20241v1)

**作者:** Shuang Liu `[一作]` (Carnegie Mellon University), Sarah Scheffler `[通讯]` (Carnegie Mellon University)

**通讯引用:** 238 | [OpenAlex ID](https://openalex.org/A5038503962)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建一个多维度的“适当调整”分析模型，对美国25州的年龄验证立法进行系统梳理，并评估现行技术实现（直接上传身份证、可验证数字凭证、面部年龄估计）的优势与劣势，进而在田纳西州保护儿童社交媒体法案的案例中展示如何应用该模型进行法规与技术的权衡分析。

**💡 创新点**

创新点在于：1）首次将美国所有州级年龄验证法案映射并归纳；2）提出包含政府目标与系统属性四层的可量化框架，用以解释最高法院对“适当调整”的中间审查标准；3）将该框架应用于技术方法与法规评估，提供跨学科的评估工具；4）通过案例研究验证模型的实用性。

**🔧 技术方法**

使用的技术包括：法律文本与判例分析、标准化可验证凭证技术（ISO/IEC 18013、W3C Verifiable Credentials、OpenID4VP）、光学字符识别（OCR）、面部年龄估计的机器学习模型以及相关的安全与隐私评估技术。

**📊 数据集**

论文未使用具体实验数据集，主要基于公开的法律文件、州法案文本、最高法院判例、行业技术标准与公开的技术实现案例。

**📈 对比分析**

评估方法通过模型的四个属性（Assurance、Business Convenience、Consumer Convenience、Data Protection）对不同技术进行定性对比；在案例研究中量化了各属性对田纳西法案的影响，展示了技术选择如何在保障儿童安全与保护隐私、便利性之间取得折衷。

**⚠️ 局限性**

局限性包括：1）缺乏实测数据和量化指标，模型评估依赖主观权重；2）技术评估基于文献与示例，未在大规模真实环境中验证；3）州法律差异导致模型适用性受限；4）未充分考虑未来AI伪造等新兴技术对系统安全的影响。

---

## 164. Understanding npm Developers' Practices, Challenges, and Recommendations for Secure Package Development

**arXiv ID:** 2601.20240 | [PDF](https://arxiv.org/pdf/2601.20240v1)

**作者:** Anthony Peruma `[一作]` (University of Hawaii at Manoa), Italo De Oliveira Santos `[通讯]` (University of Hawaii at Manoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 npm 包维护者进行调查，分析他们的安全认知、实践与挑战，并提出改进建议。

**💡 创新点**

首次系统性从开发者视角量化 npm 生态安全现状，揭示认知与实际安全之间的差距。

**🔧 技术方法**

采用混合方法的在线问卷、定量描述统计、Borda 排序以及手工主题编码分析。

**📊 数据集**

收集了 75 名 npm 包开发者的问卷数据，涵盖了参与角色、经验、下载量等信息。

**📈 对比分析**

与已有工具（如 npm audit、Dependabot 等）对比分析，指出工具满意度低、误报高、缺乏可操作性。

**⚠️ 局限性**

样本受限于自选参与、时间点与平台，难以覆盖全部 npm 开发者，且主要为自我报告，存在偏差。

---

## 165. Feature Projection Learning for Better Vision-Language Reasoning

**arXiv ID:** 2601.20224 | [PDF](https://arxiv.org/pdf/2601.20224v1)

**作者:** Yi Zhang `[一作]` (Shenzhen University), Liang-Jie Zhang `[通讯]` (Shenzhen University)

**通讯引用:** 3889 | [OpenAlex ID](https://openalex.org/A5068728111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Feature Projection Learning（FPL）方法，通过将类别原型特征投射到查询图像特征空间并重构特征图来实现CLIP模型的高效微调，解决了现有方法在性能、参数量和训练时长上的瓶颈。

**💡 创新点**

创新点在于将分类任务转化为特征投影任务，利用闭式Ridge回归求解投影矩阵，并加入投影正交性损失以提升类别间可分性；同时只需学习极少参数（μ、ϵ）即可保持对原始CLIP知识的完整性。

**🔧 技术方法**

核心技术包括：CLIP预训练视觉编码器E_v（去掉注意力池化层获取空间特征图）、Ridge回归投影模型、投影正交性损失、温度参数学习、以及将投影模型输出与原始CLIP输出融合的残差机制。

**📊 数据集**

在11个通用与细粒度分类数据集（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food-101、FGVC Aircraft、DTD、EuroSAT、SUN397、UCF101）以及4个分布偏移数据集（ImageNet‑V2、ImageNet‑Sketch、ImageNet‑A、ImageNet‑R）上进行评估。

**📈 对比分析**

与CoOp、CoCoOp、Tip‑Adapter‑F等SOTA方法相比，FPL在11个数据集上平均提升约4%–5%，在16-shot ImageNet上取得66.68%准确率，仅需1分钟训练、0.001GFLOPs、0.001M参数，明显优于对比方法的训练时长和参数规模。

**⚠️ 局限性**

局限性在于仅针对图像分类与域泛化任务验证；在更大规模或多模态任务中性能与鲁棒性尚未全面评估，且仍依赖CLIP预训练基础，若预训练模型不匹配，效果可能受限。

---

## 166. Minimum-Cost Network Flow with Dual Predictions

**arXiv ID:** 2601.20203 | [PDF](https://arxiv.org/pdf/2601.20203v1)

**作者:** Zhiyang Chen `[一作]` (Tsinghua University), Xia Yin `[通讯]` (Tsinghua University)

**通讯引用:** 5743 | [OpenAlex ID](https://openalex.org/A5100736595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种利用机器学习预测的双向 ε‑relaxation 最小成本流算法，并给出了其时间复杂度与样本复杂度的理论分析。

**💡 创新点**

创新点在于：①首次将双向预测引入 ε‑relaxation；②证明在预测误差为 ∞‑范数时，算法仍保持一致性与鲁棒性；③给出基于成本缩放的更优时间复杂度；④通过 PAC 学习理论给出预测学习的样本复杂度；⑤在交通网络与芯片逃逸路由两个实际场景中验证了显著加速。

**🔧 技术方法**

核心技术包括：ε‑relaxation（双向）算法、成本缩放技巧、预测误差分析、伪维数与 PAC 学习框架、卷积神经网络（UNet）作为特征预测器。

**📊 数据集**

实验使用的真实数据集：
- 交通网络（>200k 节点的多实例），
- 芯片逃逸路由（从 300×300 到 >1000×1000 的 PCB 网格）。

**📈 对比分析**

与传统方法比较：
- ε‑relaxation（无预测）
- 网络单纯形（NS）
- 逐最短路径（SSP）

性能：
- 交通网络：平均加速 6.2–21.4×，部分实例达 21.4×；
- 逃逸路由：平均加速 1.1–2.3×，最高 2.3×；
- 对比 NS 与 SSP，预测版 ε‑relaxation 在 0/1 流实例上可超越两者。

**⚠️ 局限性**

局限性：
- 仅考虑双向预测，未探讨原始预测或其他算法；
- 需要先验的预测模型，学习成本高且依赖足够多的训练样本；
- 预测误差仍会影响运行时间，极端误差下性能退化；
- 证明基于理论假设（如最大成本 C、图的规模），实际效果可能受网络结构限制；
- 对非常大规模实例的可扩展性尚未在大规模实验中全面验证。

---

## 167. An Autonomous Agent Framework for Feature-Label Extraction from Device Dialogues and Automatic Multi-Dimensional Device Hosting Planning Based on Large Language Models

**arXiv ID:** 2601.20194 | [PDF](https://arxiv.org/pdf/2601.20194v1)

**作者:** Huichao Men `[一作]` (Midea AI Research Center), Xinhua Xiao `[通讯]` (Midea AI Research Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于大型语言模型的AirAgent自治框架，实现对家居空气系统的主动感知、推理与规划，能够根据用户个人资料与实时环境生成可执行的多维控制策略；

**💡 创新点**

创新点在于双层协作架构（记忆标签提取 + 推理规划）以及半流式输出机制，将Chain‑of‑Thought解释与可执行JSON指令并列，支持对25个空气参数维度的自动规划与约束；

**🔧 技术方法**

技术包括语音识别（ASR）、指令调优的实体提取模型、基于LLM的推理‑规划模型、结构化控制指令生成、半流式输出分割标记以及闭环控制与算法策略协同；

**📊 数据集**

使用自构造的指令‑输入‑输出格式数据集（包含复杂代词、细腻表达的对话），并融合多源环境传感数据、设备状态、用户画像与疾病知识库；

**📈 对比分析**

与竞争对手闭源模型对比：记忆标签提取模块用户体验通过率从60%提升至83.3%，推理‑规划模块通过率从40%提升至94.9%，平均推理延迟分别从0.51s/4.82s降低到3.33s/4.51s；

**⚠️ 局限性**

局限主要为推理延迟偏高、在健康相关功能触发（如消毒）和基本控制命令逻辑一致性上存在错误，需进一步提升健康语义识别、逻辑约束及推理效率。

---

## 168. ProFlow: Zero-Shot Physics-Consistent Sampling via Proximal Flow Guidance

**arXiv ID:** 2601.20227 | [PDF](https://arxiv.org/pdf/2601.20227v1)

**作者:** Zichao Yu `[一作]` (University of Science and Technology of China), Weiguo Gao `[通讯]` (Fudan University)

**通讯引用:** 3188 | [OpenAlex ID](https://openalex.org/A5033363297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于预训练的功能流匹配（Functional Flow Matching）模型的零样本物理一致采样框架ProFlow，能够在仅给定稀疏观测且严格满足偏微分方程约束的情况下，生成满足物理规律且符合观测的随机场。

**💡 创新点**

创新点在于将物理约束和观测约束融入到后验采样的两步近似：先做终端点的proximal优化（局部MAP更新）严格满足物理方程和观测；再通过线性插值回到流匹配路径，保持生成器的统计结构，从而实现硬约束与生成先验的统一。

**🔧 技术方法**

使用的核心技术包括功能流匹配预训练模型、贝叶斯后验框架、proximal优化（约束最小化）以及线性OT桥接（插值）。

**📊 数据集**

实验使用公开的Poisson、Helmholtz、Darcy三类二维椭圆 PDE 数据集和一维粘性 Burgers 方程的数据集，所有数据均来自 FNO 与 DiffusionPDE 官方发布的训练集。

**📈 对比分析**

与EIC、DiffusionPDE、D‑Flow、PCFM等四个基准方法相比，ProFlow在前向、逆向与联合稀疏重建任务中均取得了更低的重建误差、均方误差、标准差误差以及更小的 PDE 残差，显示出更优的物理一致性和统计准确性。

**⚠️ 局限性**

主要局限是每一步的proximal优化计算成本较高，可能导致采样速度受限；此外在高度无序或湍流等更复杂的动力系统中效果尚未验证，且目前缺乏全局收敛性理论保证。

---

## 169. Hyperparameter Transfer with Mixture-of-Expert Layers

**arXiv ID:** 2601.20205 | [PDF](https://arxiv.org/pdf/2601.20205v1)

**作者:** Tianze Jiang `[一作]` (Princeton University), Boris Hanin `[通讯]` (Princeton University)

**通讯引用:** 835 | [OpenAlex ID](https://openalex.org/A5085872076)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对Mixture-of-Experts层的超参数迁移参数化，能够在保持稀疏度的同时，随模型宽度、深度、专家数和专家尺寸的扩展迁移学习率和初始化尺度。

**💡 创新点**

创新点在于将稀疏MoE模型纳入最大更新参数化（μP）和CompleteP框架，并通过动力学均值场理论（DMFT）提供理论支持；同时实现了跨尺度的超参数无损迁移，解决了传统调参难题。

**🔧 技术方法**

使用了动态均值场理论（DMFT）、最大更新参数化、完成参数化（CompleteP）、无辅助损失的专家负载平衡策略以及标准Adam优化器。

**📊 数据集**

主要使用了FineWeb和C4这两个大规模自然语言数据集进行实验。

**📈 对比分析**

通过在固定token预算下对比MoE模型与相同激活参数量的稠密GPT-2基线，发现迁移得到的超参数能实现均匀专家负载、训练稳定，并在更长token序列下取得与稠密模型相当甚至更优的验证损失。

**⚠️ 局限性**

局限性包括：实验主要集中在固定token预算或中等token horizon，未验证更长训练时的效果；稀疏度设定为固定比例，未探讨不同稀疏度对迁移的影响；以及对硬件成本（时延、内存）的影响尚未深入研究。

---

## 170. On the Computational Complexity of Performative Prediction

**arXiv ID:** 2601.20180 | [PDF](https://arxiv.org/pdf/2601.20180v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Jingming Yan `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究在强烈的 performative 效应下计算 performative 稳定点的计算复杂性，提出并证明了一个从线性可解到 PPAD‑complete 的临界点；

**💡 创新点**

揭示了 Lβ/α>1 时，计算 ε‑performatively 稳定点本质上等价于寻找一般博弈的 Nash 均衡，从而证明了该问题在稍微扩张的情形下是 PPAD‑hard 的；

**🔧 技术方法**

使用固定点与变分不等式的直接归约、收缩映射与 Halpern 迭代、椭圆算法以及对非扩张范数的推广等理论工具；

**📊 数据集**

论文为理论性研究，不涉及具体数据集；

**📈 对比分析**

通过理论上限与下限对比，证明当 ρ≤1+O(ε⁴) 时可在 (d,log(1/ε)) 时间内获得近似解，而当 ρ>1+ε 时问题变为 PPAD‑hard，展示了算法性能与 ρ 之间的锐利分界；

**⚠️ 局限性**

主要局限在于仅给出理论复杂度分析，未提供实验验证；且对于实际应用中的特定分布映射，是否仍保持困难性仍是开放问题。

---

## 171. Eliciting Least-to-Most Reasoning for Phishing URL Detection

**arXiv ID:** 2601.20270 | [PDF](https://arxiv.org/pdf/2601.20270v1)

**作者:** Holly Trikilis `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**通讯引用:** 2152 | [OpenAlex ID](https://openalex.org/A5038376039)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Least-to-Most提示框架用于钓鱼URL检测。

**💡 创新点**

创新点是将Least-to-Most分解策略与答案灵敏度机制结合，引导LLM逐步推理并在阈值满足时终止。

**🔧 技术方法**

使用大型语言模型（Gemma 3、Llama 3.1、GPT‑4.1、Gemini 2.5‑Flash）以及提示工程技术。

**📊 数据集**

采用HP、EBBU和ISCX三个公开URL数据集进行实验。

**📈 对比分析**

与单轮提示和监督模型URLTran比较，Least‑to‑Most平均F1约0.904，略低于URLTran的0.99，显著优于One‑Shot。

**⚠️ 局限性**

局限在于仍依赖LLM推理速度和阈值设置，且在最差模型（如Llama）下效果相对有限。

---

## 172. SATA: Sparsity-Aware Scheduling for Selective Token Attention

**arXiv ID:** 2601.20267 | [PDF](https://arxiv.org/pdf/2601.20267v1)

**作者:** Zhenkun Fan `[一作]` (Georgia Institute of Technology), Arijit Raychowdhury `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8411 | [OpenAlex ID](https://openalex.org/A5091408102)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对选择性 token attention 的稀疏感知调度框架 SATA，旨在通过重新排序 Q/K 访问并充分利用数据局部性来提升 Transformer 计算的硬件利用率。

**💡 创新点**

创新点包括：① 基于掩码的 intra‑head 排序实现 Q/K 的局部化；② inter‑head 动态调度 FSM，使计算单元在执行 Q‑K 乘法时保持满载；③ 对长序列的 tiling 与 zero‑skip 方案，显著抑制冗余访存与计算。

**🔧 技术方法**

使用了计算内存（CIM）架构、数字控制器、Finite State Machine 调度、类似度计算与排序、SystemVerilog 合成、NeuroSim 仿真及后硅验证的评估框架。

**📊 数据集**

在四个采用 Top‑K 选择性注意的 Transformer 任务上验证：TTST、KVT‑DeiT‑Tiny、KVT‑DeiT‑Base 以及 DRSformer，所用数据集涵盖自然语言与视觉任务。

**📈 对比分析**

与传统密集注意力以及现有稀疏加速器（如 A^3、SpAtten 等）对比，SATA 在吞吐量上提升 1.47×–1.76×，能耗效率提升 1.81×–2.94×；在 systolic‑array 级平台上实现 3.09× 吞吐提升、停顿周期降低至 75.2%。

**⚠️ 局限性**

主要局限：调度器在极大 tile 或低嵌入维度（Dk<32）时开销显著；零跳机制在冗余 operand 极高时收益有限；调度开销与 tile 大小呈二次增长，需在 1GHz 时钟下保持低于计算时间。

---

## 173. Reversible Efficient Diffusion for Image Fusion

**arXiv ID:** 2601.20260 | [PDF](https://arxiv.org/pdf/2601.20260v1)

**作者:** Xingxin Xu `[一作]` (Tianjin University), Pengfei Zhu `[通讯]` (Xiong'an Guochuang Lantian Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种可逆高效扩散模型RED，用于多模态图像融合，并实现端到端训练；

**💡 创新点**

创新点包括：①可逆融合范式实现低显存的逆向扩散过程；②引入可逆残差块进一步降低训练内存；③在扩散过程中直接给出强监督（SSIM/MAE/梯度损失），消除马尔可夫误差累积；④学习融合权重w自适应组合不同迭代步骤的结果。

**🔧 技术方法**

技术实现上采用DDIM/Latent Diffusion模型为骨干，搭建可逆U‑Net网络（可逆残差块），使用可逆前向/后向重算机制，训练时利用SSIM、MAE和梯度损失进行监督，并通过可逆模块实现显存复用。

**📊 数据集**

实验数据集包含可见-红外融合数据集LLVIP、MSRS、M³FD，以及医学图像融合数据集Harvard Whole Brain Atlas；

**📈 对比分析**

与七个SOTA方法（U2Fusion、UMF-CMGR、YDTR、DeFusion、CDDFuse、DDFM、TC-MoA、TTD、Text-DiFuse、Dream‑IF）对比，RED在EI、AG、SF、Q^AB/F、VIFF、PSNR等指标均排名第一或第二，且在目标检测等下游任务中也取得最优mAP表现。

**⚠️ 局限性**

局限性包括：训练耗时较长（可逆机制导致时间-空间折中）；步骤数选择经验性；在高负载应用（如医疗、自动驾驶）中对计算成本与性能平衡尚未得到充分保障。

---

## 174. C2:Cross learning module enhanced decision transformer with Constraint-aware loss for auto-bidding

**arXiv ID:** 2601.20257 | [PDF](https://arxiv.org/pdf/2601.20257v1)

**作者:** Jinren Ding `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 C2 框架，通过交叉学习块增强决策变换器并引入约束感知损失，改进了自动竞价模型。

**💡 创新点**

① 交叉注意力的交叉学习块强化状态、动作、RTG 序列之间的跨相关建模；② 将预算与 CPA 约束融入损失函数，实现对优劣轨迹的差异化学习。

**🔧 技术方法**

变换器架构、跨序列注意力、约束加权的平方误差损失、AuctionNet 离线数据评估。

**📊 数据集**

Alibaba 公开的大规模广告竞价数据集 AuctionNet。

**📈 对比分析**

在 50%–150% 预算比例下与 USCB、CQL、BCQ、CDT、IQL、DT、GAS、GAVE 等基线进行离线对比，C2 平均提升约 1.96%，最高 3.23%，显著优于 SOTA GAVE。

**⚠️ 局限性**

仍缺乏动态惩罚自适应、模型结构对不同竞价场景的泛化性有限，且仅在离线模拟环境验证。

---

## 175. SoftHateBench: Evaluating Moderation Models Against Reasoning-Driven, Policy-Compliant Hostility

**arXiv ID:** 2601.20256 | [PDF](https://arxiv.org/pdf/2601.20256v1)

**作者:** Xuanyu Su `[一作]` (University of Ottawa), Nathalie Japkowicz `[通讯]` (American University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了一个基于Argumentum Model of Topics（AMT）和Relevance Theory（RT）的可控生成 benchmark，叫 SoftHateBench，用来生成并评估软性仇恨言论。

**💡 创新点**

创新点在于把 AMT 的推理结构与 RT 的相关性引导结合，逆向生成软仇恨实例，并通过分层的模糊化（GroupVague、HostilityVague）形成可控的难度梯度。

**🔧 技术方法**

采用逆向 AMT 生成、RT 引导的 beam search、NLI 估计 Effect/Cost、LLM 多模型成本估计、语义相似度、熵等技术实现生成与评估。

**📊 数据集**

数据集来源于 7 大社群领域的 16,426 条高质量仇恨语料，经过 AMT 结构化后产生 4,745 条软仇恨实例，进一步扩展到 14,235 条软变体，以及同等数量的硬仇恨基线。

**📈 对比分析**

对比方法包括 Encoder 基线、专有 LLM、开源 LLM、以及安全模型，使用 Hate Success Rate（HSR）评估。结果显示硬仇恨 HSR≈76.8%，软基线降至 43.5%，软_GV 32.9%，软_HV 21.2%，说明现有系统在软仇恨上表现显著衰减。

**⚠️ 局限性**

限制在于只评估软仇恨的检测率，未覆盖非仇恨特异性；生成的软实例可能不完全覆盖所有现实场景；对模型的可解释性和对话式介入研究仍缺乏。

---

## 176. HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-BENCH

**arXiv ID:** 2601.20255 | [PDF](https://arxiv.org/pdf/2601.20255v1)

**作者:** Yueyang Wang `[一作]` (School of Mathematical Sciences), Xiaoqing Liu `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究如何在大型语言模型的中训练阶段评估其软件工程能力，并提出基于高熵的评估指标HE‑SNR；

**💡 创新点**

创新点在于①引入熵压缩假设，将智能视为对高熵决策点的“合理犹豫”；②设计能够抵消长上下文税的高熵信噪比指标HE‑SNR；③提供仅500条轨迹的高效过滤策略与基于高熵的评价集；

**🔧 技术方法**

技术手段包括Mixture‑of‑Experts模型的中训练与SFT、Top‑k熵与熵压缩理论、RoPE/YaRN上下文扩展及频率缩放、HE‑SNR指标的构造；

**📊 数据集**

使用的数据集为SWE‑bench‑Verified的500条成功轨迹（约12.5M tokens），仅保留Action部分进行过滤；对比MoE‑S与MoE‑L在32K/128K上下文的性能；

**📈 对比分析**

通过与传统PPL、HE‑PPL的相关性实验，HE‑SNR在所有检查点与SWE‑bench Pass@1保持线性且对长上下文税鲁棒，明显优于PPL；SFT后显示对高熵子集的性能下降即为Alignment Tax；

**⚠️ 局限性**

局限性包括依赖静态熵阈值缺乏自适应性、仅针对代码风格差异导致的高熵进行过滤且缺乏通用性、尚未在非软件工程任务或更大模型规模上验证；

---

## 177. Order-Optimal Sample Complexity of Rectified Flows

**arXiv ID:** 2601.20250 | [PDF](https://arxiv.org/pdf/2601.20250v1)

**作者:** Hari Krishna Sahoo `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6010 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108`

**🎯 论文内容**

分析了Rectified Flow生成模型的样本复杂度，并给出了理论证明

**💡 创新点**

首次利用结构化线性路径和平方损失的特性，引入局部Rademacher复杂度得到最优O(ε^-2)上界

**🔧 技术方法**

局部Rademacher复杂度分析、Polyak–Łojasiewicz条件、梯度下降收敛性分析

**📊 数据集**

文中未给出具体数据集，主要是理论分析

**📈 对比分析**

与现有DDPM、一般Flow Matching比较，理论上达到更优的O(ε^-2)复杂度，优于O(ε^-4)

**⚠️ 局限性**

局限在于假设数据为sub-Gaussian且模型可满足PL等理想条件，缺乏实验验证

---

## 178. BLENDER: Blended Text Embeddings and Diffusion Residuals for Intra-Class Image Synthesis in Deep Metric Learning

**arXiv ID:** 2601.20246 | [PDF](https://arxiv.org/pdf/2601.20246v1)

**作者:** Jan Niklas Kolf `[一作]` (Fraunhofer IGD), Fadi Boutros `[通讯]` (Fraunhofer IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在扩散模型采样阶段通过文本嵌入插值和残差集合运算（并集与交集）来引导目标属性，从而生成具有高类内多样性的合成图像，进而提升深度度量学习（DML）的检索性能。

**💡 创新点**

创新点在于：①将文本嵌入插值与残差集合运算结合，形成新的采样控制机制；②利用并集与交集的残差运算在潜在空间中有选择地注入或抑制属性方向；③为DML任务提供可控且语义一致的合成数据。

**🔧 技术方法**

技术手段包括：Stable Diffusion 1.5 作为基线模型；LoRA + Textual Inversion 进行模型个性化；CLIP 用于属性编码与相似度计算；潜在堆叠与文本条件融合；残差集合运算（并集、交集、差集）；以及控制指导权重与范数裁剪。

**📊 数据集**

实验数据集：CUB‑200‑2011（鸟类）和 Cars‑196（汽车）两大视觉分类数据集。

**📈 对比分析**

通过与基线（仅使用目标anchor prompt）、CutMix‑HSE、ProxyAnchor、Potential Field 等现有方法对比，使用 ResNet‑50 作为 backbone，在 Recall@1 上分别提升 3.7%（CUB）和 1.8%（Cars）；在多种 backbone 与 loss 组合下，Union 残差运算整体表现优于 Intersection，并证明 1:0.6 的真实/合成比例最优。

**⚠️ 局限性**

局限性：目标属性无法完全独立注入，可能伴随姿态或背景变化；在极端属性或背景下效果仍受限；生成方法并非完全的图像修复方式，仍会对概念特征产生轻微影响。

---

## 179. MALLOC: Benchmarking the Memory-aware Long Sequence Compression for Large Sequential Recommendation

**arXiv ID:** 2601.20234 | [PDF](https://arxiv.org/pdf/2601.20234v1)

**作者:** Qihang Yu `[一作]` (Zhejiang University), Fei Wu `[通讯]` (Zhejiang University)

**通讯引用:** 20391 | [OpenAlex ID](https://openalex.org/A5004882141)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 MALLOC 基准，系统性地对大型序列推荐中的内存感知长序列压缩方法进行分类、集成与评估，构建统一实验平台并在多种数据集上开展大规模实验；

**💡 创新点**

创新点在于：①首次将长序列压缩方法按内存分配粒度（序列级、令牌级、头级、精度级、架构级）细化分类；②在传统准确率指标之外引入 MACs 与内存占用等资源度量，形成多维度性能评估；③通过 Pareto 前沿分析揭示不同压缩策略在精度与资源消耗之间的权衡，为工业部署提供决策依据；

**🔧 技术方法**

使用的技术包括：HSTU 基座模型、序列级压缩（Native、Linformer、Reformer）、令牌级压缩（Longformer、Activation Beacon、H2O、SnapKV）、头级压缩（MQA、GQA、MLA）、精度级压缩（KIVI、IntactKV）、架构级压缩（RWKV）等；同时采用统一的训练/推理框架 FuxiCTR 与 PyTorch；

**📊 数据集**

使用了三个公开数据集：Amazon‑Electronic（短序列），MicroVideo1.7M 与 KuaiVideo（均为长序列，最大序列长度≈1024）；

**📈 对比分析**

通过统一的数据预处理、相同的超参数与模型结构，对比了 AUC、GAUC、Logloss 与资源消耗（MACs、内存）。实验结果显示：Native 与 Reformer 在精度上最优；Longformer/Beacon 在长序列上表现突出；Pruning、Precision‑level 方法精度下降显著；Head‑level 方法兼具精度与资源优势，构成 Pareto 前沿；

**⚠️ 局限性**

局限性包括：①压缩方法的优势高度依赖数据集与序列长度；②深度可扩展性不一致，某些方法训练不稳定；③实现复杂度差异大，难以在工业大规模系统快速落地；④缺乏自适应动态压缩机制，未考虑实时变化的资源约束。

---

## 180. Certificate-Guided Pruning for Stochastic Lipschitz Optimization

**arXiv ID:** 2601.20231 | [PDF](https://arxiv.org/pdf/2601.20231v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 4685 | [OpenAlex ID](https://openalex.org/A5044574038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于 Lipschitz 性能的黑盒优化算法 CGP（Certificate-Guided Pruning），能够在每一步给出显式的可证最优子集并提供可停止准则；并进一步设计了 CGP-Adaptive、CGP-TR、CGP-Hybrid 三个变体以支持未知 Lipschitz 常数、高维度和局部平滑性的场景。

**💡 创新点**

核心创新在于：①显式维护“活跃集” A_t 并通过 Lipschitz 上界 U_t 与下界 ℓ_t 形成可计算的证书，②证明 A_t 的体积随采样次数以近似最优维度 α 收缩，得到实例相关样本复杂度 (ε^{-(2+α)}) 的上界；③通过增广策略学习未知 L、信任域局部证书以及在局部平滑时切换到 GP-UCB，从而兼顾理论保证与实用性能。

**🔧 技术方法**

技术手段包括：基于子高斯噪声的自信区间构造、Lipschitz 上界 U_t 的最小化封装、活跃集 A_t 的闭式判定、覆盖半径与置信半径的迭代控制、增广复制策略、增广学习 L 的二分增大机制、信任域局部证书与裁剪规则、以及 GP-UCB 的局部细化。

**📊 数据集**

在 12 个基准上进行实验，维度覆盖 2~100，包括传统函数（Needle, Branin, Hartmann, Rosenbrock, Ackley, Levy, SVM-RBF, LunarLander）以及高维工程问题（Rover-60, NAS-36, MuJoCo-Ant-100）和模拟任务。

**📈 对比分析**

与 9 个基线（Random, GP-UCB, TuRBO, HEBO, BORE, HOO, StoSOO, LIPO, SAASBO）对比，CGP‑Hybrid 在 12 个任务中均达到或超过最优；在低/中维度任务中与 GP-UCB 竞争，且在 Branin、Rosenbrock 等局部平滑任务上通过 GP 切换进一步提升；在高维度任务中 CGP‑TR 以 9–12% 的优势击败 TuRBO，并提供局部安全证书；此外 CGP 的计算速度比 GP 方法快 6–8 倍。

**⚠️ 局限性**

局限性包括：①要求全局 Lipschitz 连续，若假设被违反可能错误剪枝；②标准 CGP 对维度上限约为 15，需依赖 CGP‑TR 才能扩展到 50~100；③当 L 估计不佳时，证书可能不及时更新，需要多次增广；④在极其高维或非平滑函数中，信任域或 GP 切换的效果有限。

---

## 181. Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems

**arXiv ID:** 2601.20230 | [PDF](https://arxiv.org/pdf/2601.20230v1)

**作者:** Haoyuan Yu `[一作]`, Minjie Cai `[通讯]` (Hunan University)

**通讯引用:** 646 | [OpenAlex ID](https://openalex.org/A5076011239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种半级联的全双工对话框架，将自然对话拆分为可独立处理的对话单元，并由多模态大语言模型直接从音频输入控制听说状态切换。

**💡 创新点**

创新点在于将转口和用户发言状态统一到同一决策空间，使LLM能够通过音频和文本协同判断是否继续或切换，既保留了情感和韵律信息，又大幅降低了推理延迟。

**🔧 技术方法**

核心技术包括 Silero VAD + CAM++（声源检测）、Paraformer（异步 ASR）、Qwen3‑Omni（多模态决策）和 IndexTTS1.5（流式 TTS），并通过 WebSocket/FastAPI 实现前后端交互。

**📊 数据集**

在 Human‑like Spoken Dialogue Systems Challenge 提供的 HumDial 数据集上进行实验。

**📈 对比分析**

实验结果表明，在测试集上该框架在语义与交互状态推断上实现了行业领先水平，排名第二，响应延迟从 2.753 秒降至 1.632 秒，语义得分 89.7、交互得分 57.8。

**⚠️ 局限性**

局限性主要体现在对 VAD/SV 的准确性依赖、单一数据集的泛化能力不足，以及对极端噪声和多说话人场景的鲁棒性待进一步验证。

---

## 182. Proactive SFC Provisioning with Forecast-Driven DRL in Data Centers

**arXiv ID:** 2601.20229 | [PDF](https://arxiv.org/pdf/2601.20229v1)

**作者:** Parisa Fard Moshiri `[一作]` (University of Ottawa), Emil Janulewicz `[通讯]` (Ciena)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5038176216)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种结合预测模型与深度强化学习的混合框架，实现服务功能链（SFC）的主动预配，利用预测的资源可用性进行数据中心（DC）与VNF的动态放置决策。

**💡 创新点**

创新点在于先用DRL生成仿真数据训练时空图神经网络与LSTM等预测模型，再将其集成为加权投票的集成预测器，并将预测结果嵌入DC选择过程，实现既考虑当前状态又预判未来资源的主动放置。

**🔧 技术方法**

采用深度强化学习（DQN/A2C/PPO等）产生数据集，Optuna进行超参数搜索，LSTM、Temporal/Spatial‑Temporal Graph Neural Networks进行时空预测，并将它们的加权预测融合到DC选择算法中，整体实现基于PyTorch等框架。

**📊 数据集**

使用在仿真数据中心环境中通过DRL交互产生的合成数据集，包含CPU、存储、SFC/VNF计数等多维特征。

**📈 对比分析**

与不使用预测的基线DC选择方案对比，在相同仿真场景下，预测驱动方案将AR、Industry 4.0等低接受率服务的接纳率分别从约30%提升至45–50%和45%，MIoT提升至30–47%，并在Cloud Gaming、VoIP、Video Streaming等高负载服务保持90%+接受率的同时，E2E延迟分别降低20.5%、23.8%和34.8%。

**⚠️ 局限性**

主要限制在于仅在模拟环境下验证，缺乏真实数据中心的部署与评估，集成预测器依赖多模型训练导致计算成本上升，且对长时延跨DC关联的建模仍不够充分，未来可引入图变换器等更强时空模型。

---

## 183. Control Models for In-IDE Code Completion

**arXiv ID:** 2601.20223 | [PDF](https://arxiv.org/pdf/2601.20223v1)

**作者:** Aral de Moor `[一作]` (JetBrains), Ekaterina Garanina `[通讯]` (JetBrains)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在JetBrains IDE中引入控制模型，利用触发器和过滤器在LLM代码补全中减少无用推理，提升效率和质量。

**💡 创新点**

首次提出“控制模型”概念，将IDE内部指标与LLM输出结合，使用轻量级CatBoost触发/过滤器实现约20%推理节省，并在生产环境验证。

**🔧 技术方法**

使用梯度提升决策树(CatBoost)作为触发/过滤器模型，并与Transformer基线进行对比；采集IDE遥测特征和代码上下文，在云端和本地推理。

**📊 数据集**

匿名化的JetBrains云补全日志，覆盖Kotlin、Python、PHP、C#等多语言，包含用户行为特征与补全事件标签。

**📈 对比分析**

离线评估通过不同误报率对比CatBoost与Transformer模型，衡量符号完成率、接受率、取消率；在线A/B实验显示触发器约减少13.8%生成、过滤率提升；过滤器提升接受率≈46%但符号完成率下降≈10%。

**⚠️ 局限性**

Transformer模型存在延迟与隐私泄漏风险、无法在所有语言上线、离线/在线指标依赖导致偏差、未充分评估长期生产力影响。

---

## 184. An Accounting Identity for Algorithmic Fairness

**arXiv ID:** 2601.20217 | [PDF](https://arxiv.org/pdf/2601.20217v1)

**作者:** Hadi Elzayn `[一作]`, Jacob Goldin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并验证了一条将模型准确性与不同公平度量（群内校准误差和群间错误不平衡）通过总不公平预算联系起来的会计恒等式，研究其在全局校准模型中的性质，并在二元与非二元预测任务中进行推广。

**💡 创新点**

创新点在于给出了一个通用的会计恒等式，量化了公平度量间的相互关系、准确性与公平性的补充性，并阐明了在非二元任务中不公平预算与准确性关系的放宽，从而突破传统不可调和的不公平不可能性结论。

**🔧 技术方法**

采用统计公正度量（群内校准误差 δ_C、群间不平衡 δ_B）、协方差与总误差（MSE）的理论推导，以及通过对照实验验证会计恒等式的有效性。

**📊 数据集**

在 COMPAS、Adult、German Credit、Bank Marketing 等公开公平性基准数据集上进行实验，使用多种预测器（逻辑回归、决策树、随机森林、梯度提升等）。

**📈 对比分析**

与 AIF360、FairLearn 等现有公平干预方法比较，发现公平方法在保持准确性的前提下仅能在不同不公平度量之间替代；若降低准确性，则总不公平预算会增加。实验显示理论与经验高度吻合。

**⚠️ 局限性**

主要局限包括：仅适用于全局校准模型；只考虑单一二元保护属性，未扩展至多组或交叉属性；未涵盖其他公平度量；未给出道德优先级选择；在分布漂移下全局校准可能失效。

---

## 185. TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement

**arXiv ID:** 2601.20208 | [PDF](https://arxiv.org/pdf/2601.20208v1)

**作者:** Wanjun Jia `[一作]` (Hunan University), Yaonan Wang `[通讯]` (Hunan University)

**通讯引用:** 20229 | [OpenAlex ID](https://openalex.org/A5025640070)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TRACER框架，结合树结构的Affordance Chain-of-Thought与空间约束与交互收敛细化技术，实现复杂纹理变形物体的精确功能区域定位与长周期操作。

**💡 创新点**

创新点在于将层次化语义推理与视觉感知分离，利用SCBR抑制预测溢出、ICRF实现像素级聚合，从而实现跨层级语义到物理交互的稳健映射。

**🔧 技术方法**

主要技术包括基于InternVL2-4B的多模态大语言模型推理、双流监督的SCBR损失、基于加速度场的ICRF动态流匹配，以及两阶段训练策略。

**📊 数据集**

使用改进版Fine-AGDDO15数据集（增加类型、边界约束、软掩模）进行训练与评估，并在实际双臂ABB机器人平台上验证。

**📈 对比分析**

与OOAL、OS-AGDO基线相比，TRACER在KLD、SIM、NSS三个指标上分别提升约4.8%、7.5%、4.3%，在真实场景的单臂/双臂抓取成功率分别从4/10、5/10提升至6/10、7/10。

**⚠️ 局限性**

局限性包括对极薄物体的深度不确定性、复杂接触动力学导致抓取失效以及双指抓手在极薄服装上的稳定性不足。

---

## 186. MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation

**arXiv ID:** 2601.20199 | [PDF](https://arxiv.org/pdf/2601.20199v1)

**作者:** Jing Yan `[一作]` (Bytedance), Yang Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 81616 | [OpenAlex ID](https://openalex.org/A5100354659)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 MERGE 的基于动态聚类的项目索引框架，能够在流式推荐场景下自适应构建、更新并层级化索引。

**💡 创新点**

创新点包括：① 从零开始自适应生成簇并实时监控簇占用；② 采用阈值驱动的匹配与 EMA 更新，避免误分配；③ 使用 Union‑Find 进行高效簇合并；④ 细到粗的层级合并策略实现多层索引，显著提升聚类均衡与分离度。

**🔧 技术方法**

技术手段包括：余弦相似度阈值匹配、指数移动平均（EMA）更新簇中心、Union‑Find 进行簇扩展、簇占用阈值控制、细-粗层级合并（基于相似度加权平均）、Silhouette 系数剪枝，以及多种聚类评估指标。

**📊 数据集**

数据集：来自字节跳动内部的数亿规模真实候选项目流，覆盖多种内容类型与用户行为，未公开公开公开；实验还在该平台的实时推荐管线中进行。

**📈 对比分析**

与基线 StreamingVQ 对比：离线指标显示 I2C CosSim 从 0.6 提升到 0.9；簇大小分布更均匀；C2C CosSim 下降至 0；在线 A/B 测试显示 AAD+0.0081%、AAH+0.0546%、WatchTime+0.1006%；对低流量与新鲜内容的曝光提升显著；单路径指标（Pass‑Through Rate、Contribution Ratio、Action 率）均有显著提升。

**⚠️ 局限性**

限制：需手动调优阈值（匹配阈值、占用阈值、合并阈值）；在极高频率更新场景下簇快速增删可能导致计算开销上升；目前仅在单层 VQ 与 MERGE 进行对比，缺乏对更深残差量化结构的系统性评估；对极端长尾项目的聚类效果仍有进一步改进空间。

---

## 187. Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery

**arXiv ID:** 2601.20193 | [PDF](https://arxiv.org/pdf/2601.20193v1)

**作者:** Zhipeng Zhang `[一作]` (China Mobile Research Institute), Zhenjie Yao `[通讯]` (Institute of Microelectronics, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了元认知强化学习框架，利用内部价值预测误差稳定性(VPES)来评估学习过程的可靠性，并通过可恢复的信任变量动态调节学习速率，防止在噪声或不确定反馈下的学习崩溃。

**💡 创新点**

创新点在于将学习本身视为可调节的“认知过程”，引入了可恢复的信任变量、失败安全机制以及“失去信任快、恢复慢”的非对称更新规则，解决了传统鲁棒RL仅抑制噪声而不考虑学习可信度的问题。

**🔧 技术方法**

技术方法包括：基于PPO的标准策略梯度优化；VPES作为内部一致性指标；信任变量τ和学习率缩放因子c的双向更新；失败安全阈值τ_min限制低信任时的学习率；以及与传统鲁棒RL、学习率调度、适应性优化器等基线的对比实验。

**📊 数据集**

数据集与实验环境：MuJoCo连续控制任务（HalfCheetah-v4、Walker2d-v4、Hopper-v4 等），在训练期间加入随机奖励腐败（p=0.5、ξ=10）和线性递增的不确定噪声，以模拟真实世界的信号不可靠与非平稳性。

**📈 对比分析**

与基线比较：在奖励腐败情形下，Full Meta‑Cognitive（完整框架）在平均最终回报、CVaR@20% 与晚期失败率等指标上均优于 Base‑PPO、手动学习率调度、Elastic‑PPO 等；在非平稳噪声下，该框架显著降低了尾部风险和晚期失效率，尽管平均回报略有下降，整体表现更稳健。

**⚠️ 局限性**

局限性：信任更新速率（η_up, η_down）手工设定，缺乏自适应学习；实验仅针对合成奖励腐败的连续控制任务，未验证在更复杂环境或真实噪声下的泛化；当前只调节学习率，未扩展到探索、经验回放或网络结构等其他学习环节。

---

## 188. Improving Diffusion Language Model Decoding through Joint Search in Generation Order and Token Space

**arXiv ID:** 2601.20339 | [PDF](https://arxiv.org/pdf/2601.20339v1)

**作者:** Yangyi Shen `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**通讯引用:** 23602 | [OpenAlex ID](https://openalex.org/A5091179481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种联合搜索生成顺序与词元的解码算法OTS，利用概率估计对轨迹进行剪枝；

**💡 创新点**

创新点在于将离散扩散模型的并行去噪特性转化为在生成顺序和词元空间的联合搜索，并设计了专用的增量似然估计器；

**🔧 技术方法**

主要技术包括多波束搜索、块级扩散、基于模型输出的增量似然评估与动态剪枝；

**📊 数据集**

使用GSM8K、MATH500、Countdown和HumanEval四个推理/编程基准，并在LLaDA-8B-Instruct及其RL后训练版本上评测；

**📈 对比分析**

与低置信度重掩、随机重掩、AR、AR+beam等方法比较，OTS在所有四个数据集上平均提升3-8%准确率，甚至超过昂贵的diffu-GRPO；

**⚠️ 局限性**

局限性包括较高的计算量（NFE和beam size），对特定模型的适用性有限，以及对更大规模模型的可扩展性仍需进一步研究。

---

## 189. MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment

**arXiv ID:** 2601.20335 | [PDF](https://arxiv.org/pdf/2601.20335v1)

**作者:** Qinzhuo Wu `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MobileBench-OL在线基准，包含1080个真实世界移动GUI任务并分为5个子集，评估任务执行、复杂推理、探索和噪声鲁棒性；

**💡 创新点**

创新点在于①从真实设备收集多样化任务，②设计5个子集覆盖基础、长时序、GUI推理、噪声鲁棒等维度，③构建自动评估与重置机制实现可复现、稳定评测；

**🔧 技术方法**

使用自动评估框架（Auto‑Eval）与细粒度重置任务，结合设备状态观察、LLM判定与XPath规则实现任务成功判定；

**📊 数据集**

数据集为来自80款中国主流应用的1080个任务，任务由人工标注、黄金轨迹生成并覆盖多种功能点与噪声场景；

**📈 对比分析**

对12个领先GUI代理（含GPT‑4o、M3A、T3A、Mobile‑Agent‑V2、UI‑TARS‑1.5等）进行基准测试，UI‑TARS‑1.5在5个子集的SR最高；整体SR低于70%，说明在长时序、探索与噪声处理方面仍有显著提升空间；

**⚠️ 局限性**

限制在于缺乏跨应用任务、对极端复杂场景覆盖不足，且部分任务对服务器交互或不可逆操作的处理仍需改进；

---

## 190. Test-Time Adaptation for Anomaly Segmentation via Topology-Aware Optimal Transport Chaining

**arXiv ID:** 2601.20333 | [PDF](https://arxiv.org/pdf/2601.20333v1)

**作者:** Ali Zia `[一作]` (La Trobe University), Wei Xiang `[通讯]` (La Trobe University)

**通讯引用:** 15971 | [OpenAlex ID](https://openalex.org/A5100389005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出TopoOT框架，将拓扑持久性与最优传输相结合，实现对异常分割的测试时自适应；通过多尺度持久性图的OT链式对齐生成稳健伪标签，并用轻量化头部在运行时优化分割结果。

**💡 创新点**

创新点在于：①用拓扑持久性图捕捉多尺度结构并通过OT链式对齐实现稳定性评估；②以OT稳定特征作为伪标签，替代脆弱阈值；③结合OT一致性和对比学习的轻量化训练，实现单样本自适应；④支持2D/3D无监督分割并可无缝迁移不同模型。

**🔧 技术方法**

技术手段包括：持久性同调、熵正则化最优传输、跨-PD与跨层级OT链式匹配、伪标签生成、OT一致性损失、对比损失和轻量化分割头。

**📊 数据集**

使用的公开数据集有MVTec AD、VisA、Real‑IAD（2D）以及MVTec 3D‑AD、AnomalyShapeNet（3D）。

**📈 对比分析**

与传统阈值方法、TTT4AS等基线比较，TopoOT在大多数基准上实现了F1提升约+24.1%（2D）和+10.2%（3D），且在精度/召回/IoU方面均领先；推理速度121 FPS、显存349MB。

**⚠️ 局限性**

局限性包括：仍依赖底层异常映射的质量；持久性计算与OT链式匹配的计算开销；对纹理复杂或极端分布漂移的鲁棒性有限；未来需要进一步优化效率与多模态统一表示。

---

## 191. DiagLink: A Dual-User Diagnostic Assistance System by Synergizing Experts with LLMs and Knowledge Graphs

**arXiv ID:** 2601.20311 | [PDF](https://arxiv.org/pdf/2601.20311v1)

**作者:** Zihan Zhou `[一作]` (Northeastern University), Zezheng Feng `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一款双用户交互式诊断支持系统，该系统将大型语言模型（LLM）、医学知识图谱（KG）与临床专家协同工作，既能为患者收集结构化病史，又能为医生提供可解释的诊断候选、证据链与治疗建议，并实现专家闭环知识更新；

**💡 创新点**

创新点包括①双用户（患者与医生）协作的闭环推理框架，②基于LLM的引导式病史收集与KG检索融合的RAG式诊断生成，③专家“在环”动态KG演化与知识校验，④角色自适应界面与多源证据可视化，显著提升解释性与信任度；

**🔧 技术方法**

技术手段包括GPT‑4.1 LLM、PrimeKG/Neo4j知识图谱、FAISS检索、RAG推理、Graphology/Sigma.js图形可视化、Nuxt.js/Vue前端与Flask后端，此外利用模板约束、情感支持对话、节点重要性排序与专家审阅接口实现闭环学习；

**📊 数据集**

使用PrimeKG（约13万节点、400万边、1.7万疾病、8000种药物）作为医学知识库，评估时采用Royal College of Physicians UK提供的12组临床情景包与模拟患者/医生数据；

**📈 对比分析**

与两类基线（文本对话+KG检索+LLM辅助和仅在医生确认后呈现诊断）进行within‑subject对比实验，共12名模拟患者、12名医生。评估指标包括患者满意度问卷、NASA‑TLX工作负荷、SUS可用性得分、专家评分以及诊断准确率（Top‑1/Top‑3）和诊断时长。实验结果显示：患者满意度显著提升，医生工作负荷下降，SUS得分77.5，诊断Top‑1准确率由7/12提升至9/12，Top‑3从9/12提升至11/12，诊断时长从18.6 min下降至7.8 min；系统未出现显著锚定效应；

**⚠️ 局限性**

局限性包括：仅在非急诊成人内科模拟情境下评估，未涵盖低健康素养或辅助技术使用者；模拟患者缺乏真实临床复杂度；KG演化效果需在长期真实部署中验证；系统可能受LLMhallucination影响，需要进一步的事实核查机制；图形视图在疾病信息量大时易混乱，需改进可视化与推荐算法；

---

## 192. Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches

**arXiv ID:** 2601.20307 | [PDF](https://arxiv.org/pdf/2601.20307v1)

**作者:** Xinyu Li `[一作]` (Xiamen University), Chen Lin `[通讯]` (Xiamen University)

**通讯引用:** 17204 | [OpenAlex ID](https://openalex.org/A5100443683)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于在线广告中post‑click GMV预测的延迟反馈建模方法，并构建了首个包含完整交易序列的公共基准数据集。

**💡 创新点**

创新点包括①首次公开构建完整交易序列的GMV基准；②发现单购与多购样本标签分布显著不同，提出双分支网络；③设计标签校准器与部分标签去学习等去偏技术；④实现在线流式训练与自适应路由。

**🔧 技术方法**

使用技术包括在线流式学习、双分支（单购/多购）网络、轻量路由器、标签校准器、Ground‑Truth Alignment、Partial Label Unlearning、Log‑MAE 损失、温度软化等。

**📊 数据集**

使用了来自淘宝/阿里巴巴展示广告的点击日志，涵盖82天、7天归因窗口的完整交易序列，构成TRAnsaCtion sEquences（TASE）基准。

**📈 对比分析**

通过与单塔离线/在线、双塔等基线以及oracle模型比较，评估指标为AUC、ACC（相对误差≤20%）、ALPR；该方法在AUC上提升0.86%，在ACC上提升2.19%，在ALPR上降低6.88%，总体显著优于基线。

**⚠️ 局限性**

局限性包括：数据仅为内部匿名化样本，未覆盖真实业务场景；模型对路由阈值和参数设置敏感；未在多业务或跨平台场景验证；部分去偏技术仍有提升空间。

---

## 193. Structure-constrained Language-informed Diffusion Model for Unpaired Low-dose Computed Tomography Angiography Reconstruction

**arXiv ID:** 2601.20304 | [PDF](https://arxiv.org/pdf/2601.20304v1)

**作者:** Genyuan Zhang `[一作]` (Chongqing University), Weiwen Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3066 | [OpenAlex ID](https://openalex.org/A5077824092)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种结构约束、语言指导的扩散模型 SLDM，用于将低剂量 CTA 图像恢复为正常剂量质量的重建。

**💡 创新点**

创新点在于融合结构先验约束、语义文本监督与差分血管增强三大模块，有效解决低剂量 CTA 的结构失真和误增强问题。

**🔧 技术方法**

使用结构约束的扩散 SDE、CTA‑CLIP 视觉‑语言模型、跨模态注意力机制、差分血管增强模块（SAEM）以及优化的采样策略。

**📊 数据集**

采用北京阎城医院的 50 名患者低剂量/正常剂量 CTA 数据集，并构造结构通道和文本描述进行训练与评估。

**📈 对比分析**

与 MUNIT、CycleGAN、IdentityGAN、MALAR、Syn‑diffusion、FGDM 等方法对比，SLDM 在 PSNR、SSIM、ISNR、SNR 以及两名放射科医师主观评分上均表现更优或相当。

**⚠️ 局限性**

局限性包括：仍需弱配对数据；推理时间约 7 秒，未达到实时水平；未实现无对比剂的非对比 CT 合成。

---

## 194. A Learning-based Framework for Spatial Impulse Response Compensation in 3D Photoacoustic Computed Tomography

**arXiv ID:** 2601.20291 | [PDF](https://arxiv.org/pdf/2601.20291v1)

**作者:** Kaiyi Yang `[一作]` (University of Illinois Urbana-Champaign), Mark A. Anastasio `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7581 | [OpenAlex ID](https://openalex.org/A5046506193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于学习的空间冲击响应补偿框架，将受有限尺寸换能器影响的 3D PACT 测量数据映射为理想点换能器数据，从而实现快速有效的图像重建。

**💡 创新点**

创新点在于：①将 SIR 补偿迁移到数据域并使用深度学习实现；②设计两种模型——纯数据驱动的 U‑Net 与物理启发的 Deconv‑Net；③提供高效的合成随机球体训练数据生成策略，使模型能在未见数据上保持良好泛化。

**🔧 技术方法**

采用深度卷积网络（U‑Net、Deconv‑Net）结合傅里叶域去卷积；训练使用 MAE 损失与 Adam 优化；模拟数据通过 k‑space 伪谱方法、闭式 SIR 公式生成；在 GPU 上并行实现。

**📊 数据集**

训练集：10,000 对合成随机球体数据；验证/测试集：同源球体；OOS 测试：确定性球体、均匀与异质声速数字乳腺phantom（NBI‑PBS）；实验数据：真实乳腺 PACT 的在体扫描结果。

**📈 对比分析**

与未补偿 SIR（UBP）及基准方法对比，评价指标包括 FWHM、相对平方误差（RSE）、归一化互相关（NCC）和 DICE 系数；结果显示 U‑Net 与 Deconv‑Net 均显著提升分辨率和结构保留，Deconv‑Net 在所有指标上均优于 U‑Net；在实验数据中亦恢复细节并降低噪声影响。

**⚠️ 局限性**

局限性：仅验证了特定换能器尺寸和形状；未针对更大换能器造成的信息丢失进行深入评估；模型训练基于合成球体，虽然泛化良好但对极端结构或不同系统几何仍可能受限；未与高精度迭代优化重建方法直接比较。

---

## 195. Memory Retrieval in Transformers: Insights from The Encoding Specificity Principle

**arXiv ID:** 2601.20282 | [PDF](https://arxiv.org/pdf/2601.20282v1)

**作者:** Viet Hung Dinh `[一作]` (University of Sydney), Kanchana Thilakarathna `[通讯]` (University of Sydney)

**通讯引用:** 2064 | [OpenAlex ID](https://openalex.org/A5073287852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多种LLM的注意力机制进行实验，验证Transformer的键值矩阵实现了类似人类的记忆检索，并证明检索线索主要是关键字。

**💡 创新点**

创新点在于将编码特异性原理与注意力机制结合，首次揭示键值权重能够索引记忆痕迹，识别出专门编码关键字的特定神经元，进而实现可解释的机器消融。

**🔧 技术方法**

使用的技术包括注意力矩阵交换、键矩阵扰动、关键字提取（GPT‑4o、LXT）、以及ROUGE‑L、BERTScore、MAUVE、重复率等多维评估指标。

**📊 数据集**

实验数据集涵盖多款LLM的事实/反事实对照数据、公开书籍长文本以及GPT‑4o生成的关键词集合。

**📈 对比分析**

通过与随机扰动、LXT关键词以及GPT‑4o关键词对照，发现关键词扰动显著降低ROUGE‑L和BERTScore，验证了注意力关键字对记忆召回的决定性作用，整体生成质量影响较小。

**⚠️ 局限性**

限制在于关键词提取仍较粗糙，无法处理复合词；关键词选择与数量缺乏系统化标准，扰动策略仅采用零化，未充分探索更高效的消融方法。

---

## 196. The Forecast After the Forecast: A Post-Processing Shift in Time Series

**arXiv ID:** 2601.20280 | [PDF](https://arxiv.org/pdf/2601.20280v1)

**作者:** Daojun Liang `[一作]` (Qilu University of Technology), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 48891 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在已部署的时间序列预测模型上，提出一种轻量级、架构无关的后处理方法 δ-Adapter，通过对输入进行微调和对输出进行残差校正来提升预测精度与不确定性估计，同时保持模型不变。

**💡 创新点**

创新点包括：① 用可控信赖区间 δ 对输入/输出进行 bounded 修正，实现理论上的局部下降保证；② 将输入适配器演化为稀疏、时段感知的特征选择器，提升可解释性与稳定性；③ 结合分位数校准器与自适应 conformal 校准器，提供校准且具有有限样本覆盖率的置信区间。

**🔧 技术方法**

技术手段：小型 MLP/低秩头作为适配器；additive/multiplicative 形式的输入/输出编辑；Gumbel‑Sigmoid relax 的稀疏掩码；pinball 损失和可靠性正则化；自适应尺度学习的 conformal 校准；理论证明（Lipschitz、局部下降、组合稳定性）。

**📊 数据集**

实验使用多种公开时间序列基准：ETT‑ETT‑E、Exchange、Traffic、Weather、M5 等；覆盖单变量和多变量场景，并在不同 Backbone（Sundial, TTM‑R2, PatchTST, TimeMixer, iTransformer 等）上验证。

**📈 对比分析**

与传统全微调、LoRA、在线学习、聚合方法以及现有校准技术（CQR、EnbPI、SPCI）比较，δ-Adapter 在 MSE、MAE 以及 PICP 等指标上均实现显著提升，且训练开销仅为原模型的 2–6% 参数量，推理不需要任何改动。

**⚠️ 局限性**

局限性：① 需要手动调节 δ 与学习率，过大可能导致性能波动；② 目前仅处理低阶残差结构，难以补偿严重的概念漂移；③ 对某些高容量模型（如 LoRA 细调）仍存在性能波动；④ 依赖有限的训练数据，稀疏掩码可能在极端稀疏预算下失效。

---

## 197. RusLICA: A Russian-Language Platform for Automated Linguistic Inquiry and Category Analysis

**arXiv ID:** 2601.20275 | [PDF](https://arxiv.org/pdf/2601.20275v1)

**作者:** Elina Sigdel `[一作]` (Laborarory of AI application in Psychology, Institute of Psychology RAS), Anastasia Panfilova `[通讯]` (Laborarory of AI application in Psychology, Institute of Psychology RAS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个免费开放的 RusLICA web 服务，实现对俄语文本的自动化心理语言学分析，涵盖96个词汇、句法、形态、统计与情感分类。

**💡 创新点**

创新点在于不采用直接翻译，而是基于俄语语义词典、RuWordNet、俄罗斯国家语料库等构建专属词汇表，并融合预训练语言模型实现情感检测；同时将 LIWC 方法扩展到俄语并加入语法特征。

**🔧 技术方法**

技术包括 SpaCy 俄语模型进行分词、词形还原、句法解析，MyStem 词形还原，语义词典/语料库查询，RuWordNet 同义/下位/上位关系扩展，以及 HuggingFace 的 pre‑trained RUBERT 模型用于情感分类。

**📊 数据集**

主要数据集为俄罗斯国家语料库（RNC）和 RuWordNet 词典，用于构建词汇表；情感检测使用已 fine‑tune 的 Aniemore/rubert‑tiny2‑russian‑emotion‑detection 模型，该模型基于 CEDR 数据集训练。

**📈 对比分析**

与传统 LIWC 的直接词汇计数相比，RusLICA 在俄语文本上提供更丰富的统计、句法、形态特征，并通过预训练模型提升情感识别准确度；实验表明对大规模俄语语料的处理时间可在12小时以内完成，但未给出定量评估指标。

**⚠️ 局限性**

局限性包括：词汇计数方法忽略上下文与语义歧义，依赖 SpaCy 解析在标点或语序异常时可能失效；词典覆盖仍不完整，缺乏短语、俚语、表情符号等；缺乏公开基准数据集的评估与性能对比。

---

## 198. Multimodal Multi-Agent Ransomware Analysis Using AutoGen

**arXiv ID:** 2601.20346 | [PDF](https://arxiv.org/pdf/2601.20346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 199. A Source-Free Approach for Domain Adaptation via Multiview Image Transformation and Latent Space Consistency

**arXiv ID:** 2601.20284 | [PDF](https://arxiv.org/pdf/2601.20284v1)

**作者:** Debopom Sutradhar `[一作]` (United International University), Sami Azam `[通讯]` (Charles Darwin University)

**通讯引用:** 5643 | [OpenAlex ID](https://openalex.org/A5062716310)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种源免费域自适应方法，利用多视角增强和潜在空间一致性直接从目标域学习域不变特征。

**💡 创新点**

创新点在于首次将多视角增强与潜在空间一致性相结合，完全不依赖源数据、伪标签或对抗训练。

**🔧 技术方法**

使用ConvNeXt编码器、两种数据增强管道、均方误差一致性损失和交叉熵分类损失。

**📊 数据集**

在Office-31、Office-Home和Office-Caltech三个标准域自适应基准上进行实验。

**📈 对比分析**

与现有SFDA和传统方法对比，平均准确率分别提升至90.72%、84.00%和97.12%，在多数域迁移任务中获得最高或次高分。

**⚠️ 局限性**

主要局限是对增强策略的依赖，缺乏对目标标签的利用，且在大规模数据或高维特征空间下多视图的计算与存储成本可能显著。

---

## 200. TABED: Test-Time Adaptive Ensemble Drafting for Robust Speculative Decoding in LVLMs

**arXiv ID:** 2601.20357 | [PDF](https://arxiv.org/pdf/2601.20357v1)

**作者:** Minjae Lee `[一作]` (FuriosaAI), Kangwook Lee `[通讯]` (UW-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在大型视觉语言模型（LVLM）上系统评测了现有的投机式解码（SD）草稿方法，并提出了一种训练无关、可插拔的动态集成草稿方法（TABED），通过批量推理获取多份草稿并根据历史验证结果自适应调整权重，实现了显著的推理加速。

**💡 创新点**

创新点在于①利用目标模型的硬/软标签在测试时动态调整各草稿的权重，②在LVLM中实现了完全无训练的投机式解码加速，③可与高级验证技术（如token-tree）和多种草稿候选（多模态、文本、池化、多模态合成等）无缝集成，保持极低的额外开销。

**🔧 技术方法**

核心技术包括投机式解码、批量推理、基于KL或TVD的自适应权重采样、token-tree并行验证、图像信息压缩与captioning的草稿增强，以及参数共享策略。

**📊 数据集**

使用了11个多样化数据集，包括DocVQA、POPE、MMVet、IEdit、MB Spot、VIST、NQ、GSM8K以及两组包含5张图像的OOD数据集，覆盖单轮、双轮、多图像和文本推理等场景。

**📈 对比分析**

与单一多模态草稿（M）和文本草稿（T）等基线对比，TABED在所有场景下均能获得最佳或第二佳的块效率，平均墙时加速比达×（数值待补）且比单草稿提升>％，在OOD场景中表现尤为稳定。

**⚠️ 局限性**

局限性包括：仅在LLaVA系列LVLM上验证，未测试更大规模或其他家族模型；草稿模型仍保持较小容量；自适应权重机制缺乏理论保证；未来需扩展到更多模态（音频、扩散等）和更大模型。

---

## 201. Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models

**arXiv ID:** 2601.20354 | [PDF](https://arxiv.org/pdf/2601.20354v1)

**作者:** Zengbin Wang `[一作]` (AMAP, Alibaba Group), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布SpatialGenEval评测基准，利用信息密集、长篇空间感知Prompt和10维多选QA系统评估文本到图像模型的空间智能，并构造SpatialT2I数据集用于监督微调，提升模型空间推理能力。

**💡 创新点**

创新点在于：①把空间智能拆解为10个子域（从基础属性到因果交互）；②使用大语言模型生成长篇信息密集Prompt和对应多维度QA，并人工审核避免答案泄露；③提出多选评测与“None”选项的投票机制；④将评测数据转化为可训练的图文对，验证数据驱动提升空间智能。

**🔧 技术方法**

技术主要包括：大语言模型（Gemini 2.5 Pro、Qwen2.5‑VL‑72B、GPT‑4o）用于Prompt/QA生成与评估；多模态评测框架（5轮投票）；多种T2I架构（扩散、递归、统一模型）与文本编码器（CLIP、T5、LLM）；监督微调（Stable Diffusion‑XL、UniWorld‑V1、OmniGen2）。

**📊 数据集**

数据集：1) SpatialGenEval：25个真实场景下1,230条信息密集Prompt + 12,300个多选QA；2) SpatialT2I：15,400张图文对，来源于14款SOTA模型的生成并经过大模型重写以保证文本与图像一致性；3) 公开的T2I模型生成结果作为训练和评测样本。

**📈 对比分析**

评估方法：对每张生成图像，使用多选VQA评测（5轮投票）得到10个子域得分；在23款模型中排名；最高分约62.7%（Seed Dream 4.0），Open‑source与closed‑source差距缩小；空间推理子域普遍低于30%，表明为主要瓶颈；微调后模型在SpatialGenEval上平均提升4–6%。

**⚠️ 局限性**

局限性：①空间推理仍表现不佳，特别是比较、遮挡和因果子域；②评测依赖MLLM理解，可能引入评估误差；③数据规模有限，未覆盖更复杂的动态交互场景；④未探讨跨域或多模态扩展的普适性。

---

## 202. CE-RM: A Pointwise Generative Reward Model Optimized via Two-Stage Rollout and Unified Criteria

**arXiv ID:** 2601.20327 | [PDF](https://arxiv.org/pdf/2601.20327v1)

**作者:** Xinyu Hu `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Xiaojun Wan `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造了一种点式生成奖励模型 CE-RM-4B，并通过两阶段采样（先生成统一查询基准，再针对每个响应进行评估）实现对评估准则的显式优化；

**💡 创新点**

创新点在于：①将评估准则统一为仅依据查询生成，消除对每个响应的条件化导致的偏差；②采用两阶段 rollout 的强化学习框架，在仅有 pairwise 偏好标签的情况下显式优化准则与评估的生成；③用极少的 5.7K 质量筛选数据训练出仅 4B 参数的模型，却在多项奖励模型基准上优于更大规模的 pairwise 模型；

**🔧 技术方法**

使用 Qwen3-4B-Instruct-2507 作为基础模型；实现两阶段强化学习（GRPO），并结合 test‑time scaling、子组奖励估计等技术；

**📊 数据集**

主要数据集为 Skywork‑Reward‑Preference‑80K‑v0.2（筛选后 5.7K 条），以及 Tulu‑3 数据用于 RL 训练；

**📈 对比分析**

与多种现有奖励模型（CompassJudger1‑32B、RRM‑32B、RRM‑7B 等）以及通用 LLM 进行对比。CE‑RM‑4B 在 RewardBench、RewardBench2、RM‑Bench、PPE Correctness、JudgeBench 上的得分分别达到 89.0/74.6/79.8/69.7/73.7/77.4，采用 test‑time scaling 进一步提升至 90.0/76.3/83.2/75.0/76.3/80.2，显著优于同类 pairwise 模型，尤其在 Best‑of‑N 场景和多响应场景中表现突出；

**⚠️ 局限性**

局限性包括：奖励信号的估计仍不够精准，缺乏真实点式标注；对工具辅助的使用尚未完善，可能在通用聊天场景中降低性能；未来可引入少量点式标注进行校准，进一步提升评估可靠性。

---

## 203. UnlearnShield: Shielding Forgotten Privacy against Unlearning Inversion

**arXiv ID:** 2601.20325 | [PDF](https://arxiv.org/pdf/2601.20325v1)

**作者:** Lulu Xue `[一作]` (Huazhong University of Science and Technology), Leo Yu Zhang `[通讯]` (Griffith University)

**通讯引用:** 4477 | [OpenAlex ID](https://openalex.org/A5015011245)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对机器无学习逆向攻击的后处理防御方法UnlearnShield，能够在不影响模型精度和遗忘效果的前提下抑制数据重建。

**💡 创新点**

首创在余弦表示空间引入方向性扰动并通过约束模块联合优化，以同时实现隐私保护与模型可用性。

**🔧 技术方法**

采用方向性扰动、余弦相似度损失、幅度约束初始化（AIM）以及遗忘一致性损失等技术。

**📊 数据集**

在CIFAR‑10、STL‑10等视觉数据集上使用ResNet18和ConvNet进行实验。

**📈 对比分析**

与基线、Noise、Pruning、DGP、Soteria、Outpost等对手防御对比，UnlearnShield在隐私指标（SSIM、LPIPS）显著优于其他方法，同时保持了近乎无损的模型准确率和遗忘效果。

**⚠️ 局限性**

仅针对计算机视觉任务，未验证在文本或语音等其他数据类型上的通用性。

---

## 204. Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation

**arXiv ID:** 2601.20321 | [PDF](https://arxiv.org/pdf/2601.20321v1)

**作者:** Yuzhe Huang `[一作]` (Beihang University), Ziyuan Jiao `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5084328887)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种从触感视觉对齐转向触感-力对齐的框架（TaF-VLA），通过构建一个含10 M帧同步触感图像、6轴力/力矩和力矩阵图的TaF-Dataset，并训练TaF-Adapter，将触感序列映射到物理力的离散潜在空间，最终实现强大的视觉-语言-动作（VLA）策略。

**💡 创新点**

创新点在于：①利用对比学习将触感嵌入与真实力信号对齐，①通过向量量化的离散码簿抑制噪声并提升跨传感器泛化；②设计历史感知的因果Transformer捕捉触感的时序动态；③将力对齐的触感编码注入VLA骨干，实现语言指导下的力敏感操作。

**🔧 技术方法**

主要技术包括：对比学习（InfoNCE）与VQ-VAE量化的力编码器、因果Transformer对触感序列的编码、流匹配/扩散策略的动作生成、以及现有VLA骨干（如π0.5）与动作块模型的集成。

**📊 数据集**

使用的数据集有：①大规模同步触感-力数据集TaF-Dataset（10 M+帧）；②覆盖20+任务的力感知操作数据集（含LLM生成的力指令），为模型训练与评估提供丰富语义与物理标签。

**📈 对比分析**

通过与五类基线（仅视觉的Act/π0.5、视触对齐FreeTacMan、扩散DP、以及各自加TaF-Adapter）比较，在7个接触密集任务中，TaF-VLA平均成功率提升至64.8%，相较最强基线提升约19–22个百分点，且在力关键任务上显著高于对齐视觉或视觉+力的传统方法。

**⚠️ 局限性**

局限性包括：①推理频率受限，难以满足高速力反馈；②对不同物理感知机制（如电容、磁性、压阻）泛化受限；③硬件制造误差导致力标记噪声；④主要在视觉触感传感器上验证，未覆盖结构截然不同的触感模态；⑤策略未实现完整的全局触感感知与环境认知，仅在动作层面使用触感。

---

## 205. CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting

**arXiv ID:** 2601.20318 | [PDF](https://arxiv.org/pdf/2601.20318v1)

**作者:** Jiyuan Xu `[一作]` (Zhejiang University of Finance and Economics), Jiahao Nie `[通讯]` (Zhejiang University of Finance and Economics)

**通讯引用:** 518 | [OpenAlex ID](https://openalex.org/A5051595025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了CPiRi框架，融合了时间序列先验与通道不变关系学习，实现了通道顺序不敏感的多变量预测

**💡 创新点**

通过把时间特征冻结为先验、引入通道洗牌正则化以及仅用轻量级空间模块实现了真正的通道置换不变性

**🔧 技术方法**

使用冻结的Sundial预训练编码器、Transformer自注意力的空间模块以及通道洗牌正则化训练策略

**📊 数据集**

在交通、能源、计数等公开基准（METR‑LA、PEMS‑BAY、PEMS‑04、PEMS‑08、SD、Electricity）以及大规模子集（GBA、GLA、CA）上进行评测

**📈 对比分析**

与CI、CD、CD+CI等SOTA方法对比，CPiRi在大多数数据集上实现了WAPE/MAE最优，且在通道洗牌测试中几乎无性能损失

**⚠️ 局限性**

对突变趋势的动态融合不完善，且目前仅基于内在信号，未结合外部非结构化信息

---

## 206. Less is More: Benchmarking LLM Based Recommendation Agents

**arXiv ID:** 2601.20316 | [PDF](https://arxiv.org/pdf/2601.20316v1)

**作者:** Kargi Chauhan `[一作]` (University of California), Mahalakshmi Venkateswarlu `[通讯]` (Georgia Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性基准实验评估了不同LLM模型在不同用户历史上下文长度（5、10、15、25、50条）下的推荐质量、延迟与token成本。

**💡 创新点**

发现LLM推荐系统在更长上下文下质量并无显著提升，提出“最小足够上下文”原则，揭示了LLM对长期上下文利用的根本限制，并给出可实现高达88%成本节约的实用指南。

**🔧 技术方法**

使用四款主流LLM（GPT‑4o‑mini、DeepSeek‑V3、Qwen2.5‑72B、Gemini 2.5 Flash），标准化提示工程，复合质量指标（0.7关键词重叠+0.3类别匹配），配合paired‑t检验、重复测量ANOVA、token与延迟统计分析。

**📊 数据集**

REGEN数据集（Office Products域）中用户购买历史与下一条真实购买，挑选拥有至少51条历史的用户。

**📈 对比分析**

采用within‑subject设计，对同一50名用户在五种上下文长度下进行评估；结果显示质量评分稳定在0.17‑0.23，延迟模型特异；token使用从5条到50条增长约8倍，质量无提升，验证了成本‑性能优势。

**⚠️ 局限性**

局限性包括仅针对办公室产品域，复合质量指标未覆盖实际推荐价值，缺乏人工评估，提示设计简单，未探索更深层次的上下文利用技术或其它LLM与细调模型。

---

## 207. TPGDiff: Hierarchical Triple-Prior Guided Diffusion for Image Restoration

**arXiv ID:** 2601.20306 | [PDF](https://arxiv.org/pdf/2601.20306v1)

**作者:** Yanjie Tu `[一作]` (Northwestern Polytechnical University), Jiacong Tang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三重先验引导扩散网络（TPGDiff），实现统一图像恢复

**💡 创新点**

创新点在于层级分配：将语义先验注入深层、结构先验注入浅层，并通过降解先验实现多阶段自适应

**🔧 技术方法**

采用扩散模型、教师‑学生蒸馏、跨模态结构聚合、时间调制等技术

**📊 数据集**

使用涵盖低照度、雾化、摩尔纹、雨、雪、雨滴、模糊、云、噪声等九类降解的公开数据集

**📈 对比分析**

与多种单任务和统一恢复方法对比，在 PSNR/SSIM/FID/LPIPS/MUSIQ 等指标上均取得最优或相近最佳性能

**⚠️ 局限性**

局限在于训练成本高、对极端破坏仍可能出现细节失真或语义漂移

---

## 208. Artifact-Aware Evaluation for High-Quality Video Generation

**arXiv ID:** 2601.20297 | [PDF](https://arxiv.org/pdf/2601.20297v1)

**作者:** Chen Zhu `[一作]`, Yangang Wang `[通讯]` (Southeast University)

**通讯引用:** 2909 | [OpenAlex ID](https://openalex.org/A5100758518)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种面向视频生成的细粒度缺陷检测与评估框架，包含Appearance、Motion、Camera三维度的10种常见生成缺陷；

**💡 创新点**

创新点包括：1）构建人类感知导向的缺陷分类体系；2）大规模GenVID数据集与多标签QA格式；3）基于光流的动量引导动态帧采样（FMG-DFS）；4）将上述技术集成至DVAR模型实现高效缺陷识别；

**🔧 技术方法**

技术手段包括：多模态大型语言模型微调（Qwen2.5‑VL）、光流计算与峰值检测、动态帧采样、二分类QA训练与正则化提取；

**📊 数据集**

使用GenVID数据集，约8万条生成视频，覆盖多款生成模型，且每条视频标注10类缺陷并转化为96万QA对；

**📈 对比分析**

在GenVID测试集上，DVAR在Appearance、Motion、Camera和All四维度的准确率分别达0.849/0.785/0.767/0.800，明显高于GPT‑5、GPT‑4o、LLaVA‑Next等SOTA模型，整体提升约30%；

**⚠️ 局限性**

局限性包括：1）缺陷检测仍受光流估计误差影响；2）对非常细微或长时段缺陷的识别能力有限；3）模型在非标注类别的普适性待验证；4）需大量标注成本与计算资源。

---

## 209. Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics -- Rotating Detonation Engines

**arXiv ID:** 2601.20295 | [PDF](https://arxiv.org/pdf/2601.20295v1)

**作者:** Yuxuan Bao `[一作]` (University of Washington), J. Nathan Kutz `[通讯]` (University of Washington)

**通讯引用:** 28402 | [OpenAlex ID](https://openalex.org/A5083450863)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了 Cheap2Rich 框架，通过低频 LF 与高频 HF 两条路径，将低保真 RDE 模型与高保真数据进行多尺度融合，实现仅用稀疏传感器即可重构完整状态。

**💡 创新点**

创新点在于将 SHRED 与潜在空间 GAN 对齐相结合，同时通过谱稀疏的 HF 校正模块实现对注射器驱动细尺度扰动的可解释建模，从而显著弥合仿真‑真实差距。

**🔧 技术方法**

核心技术包括 SHRED（浅层递归解码器）、潜在空间 GAN 对齐、低频低通滤波、高频基于残差的 LSTM+解码器、谱稀疏正则化以及 SINDy 进行缺失物理方程挖掘。

**📊 数据集**

使用的数据集为三波共旋转 RDE 的高保真三维化学流模拟（250 帧、100 网格点）与一维 Koch 低保真模型的同步输出，传感器数量为 25 个。

**📈 对比分析**

相较于仅用 SHRED 的 0.4114 RMSE，Cheap2Rich 通过 HF 校正将 RMSE 降低至 0.1031（约 74.9% 降低），SSIM 从 0.1113 提升至 0.3638，显著优于基线方法。

**⚠️ 局限性**

局限性包括仅验证于 1D RDE 模型，传感器布局对结果敏感，且高频校正仍依赖大量仿真数据，尚未在真实实验平台上彻底验证。

---

## 210. Physically Guided Visual Mass Estimation from a Single RGB Image

**arXiv ID:** 2601.20303 | [PDF](https://arxiv.org/pdf/2601.20303v1)

**作者:** Sungjae Lee `[一作]` (POSTECH), Kwang In Kim `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一个单张RGB图像估计物体质量的物理结构框架，将质量拆分为体积与密度两项隐含因子，并分别用几何信息与VLM产生的材料语义引导推断；

**💡 创新点**

创新点在于把几何与材料分别映射到体积与密度两条物理通道，并通过实例自适应门控融合多模态信息，避免直接端到端的粗糙关联；

**🔧 技术方法**

使用技术包括单目深度估计生成点云、VLM（Qwen2.5‑VL）提取材料语义、DenseNet+PointNet+CLIP文本编码器、门控融合机制以及双回归头；

**📊 数据集**

采用的公开数据集为 image2mass 与 ABO‑500；

**📈 对比分析**

与 RGB、RGB+Depth、image2mass、VLM直接推理等多种基线进行对比，基于 ALDE、APE、MnRE、q 等指标，单视图方法在所有指标上均优于或逼近多视图NeRF2Physics；

**⚠️ 局限性**

局限性包括仅使用单一材料语义，难以处理多材料混合物体；单视图几何信息不完整导致密度估计仍易受限；在复杂背景下的鲁棒性尚待进一步提升。

---

## 211. SemBind: Binding Diffusion Watermarks to Semantics Against Black-Box Forgery Attacks

**arXiv ID:** 2601.20310 | [PDF](https://arxiv.org/pdf/2601.20310v1)

**作者:** Xin Zhang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 22480 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SemBind 框架，将潜在水印与图像语义绑定，以防黑盒伪造攻击。

**💡 创新点**

创新点在于通过学习语义掩码将水印信号与语义对应，既能保持水印鲁棒性，又能有效抵御黑盒伪造；并在多种现有潜在水印方案上实现兼容。

**🔧 技术方法**

采用对比学习训练语义掩码网络，利用隐式二值码对潜在空间进行符号调制；结合现有潜在水印方法（Tree‑Ring、Gaussian Shading、PRC、Gaussian Shading++）。

**📊 数据集**

使用 Stable Diffusion v2.1/1.5 模型生成图像，并在 MS‑COCO 与 Stable Diffusion Prompts（SDP）数据集上评估；训练语义掩码使用 SemCon‑3M。

**📈 对比分析**

与原始四种水印方案比较，SemBind 显著降低黑盒印记攻击和重写攻击的误接受率（Det）与位准确率（Bit Acc），同时保持或提升图像质量（FID/CLIP）和鲁棒性（对 JPEG、亮度、模糊等扰动的检测率>0.99）。

**⚠️ 局限性**

限制在于需要一次性训练语义掩码、部署时额外生成辅助图像，以及对极端自适应攻击的鲁棒性仍待进一步验证。

---

## 212. One Word is Enough: Minimal Adversarial Perturbations for Neural Text Ranking

**arXiv ID:** 2601.20283 | [PDF](https://arxiv.org/pdf/2601.20283v1)

**作者:** Tanmay Karmakar `[一作]` (Indian Statistical Institute), Surjyanee Halder `[通讯]` (Indian Statistical Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了仅插入或替换单词的最小攻击方式，以提升神经检索模型的排名。

**💡 创新点**

创新地提出查询中心概念，设计黑盒与白盒单词级攻击，并引入微观诊断指标。

**🔧 技术方法**

利用语义嵌入计算查询中心、梯度引导插入点以及单词级攻击方法。

**📊 数据集**

实验基于MSMARCO passage数据集与TREC‑DL 2019/2020查询集。

**📈 对比分析**

与PRADA在相同白盒条件下对比，最高可达91%攻击成功率，单词编辑量仅为1，语义相似度≥0.97，性能与PRADA相当。

**⚠️ 局限性**

局限在于仅在白盒环境验证，攻击范围受限于单词级，未考虑多词或黑盒实用性，也未评估对大型语言模型排序器的影响。

---

## 213. Hallucination Begins Where Saliency Drops

**arXiv ID:** 2601.20279 | [PDF](https://arxiv.org/pdf/2601.20279v1)

**作者:** Xiaofeng Zhang `[一作]` (Shanghai Jiaotong University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 8842 | [OpenAlex ID](https://openalex.org/A5100662197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于梯度的注意力显著性指标 LVLMs‑Saliency，并在推理阶段引入两种方法：Saliency‑Guided Rejection Sampling (SGRS) 与 Local Coherence Reinforcement (LocoRE)，用于检测并抑制大规模视觉‑语言模型（LVLM）的幻觉输出。

**💡 创新点**

创新点在于：①首次将梯度与注意力相乘形成 token‑级显著性度量，揭示幻觉与上下文记忆丢失的因果关系；②设计动态阈值的 SGRS，实现对低显著性候选 token 的即时拒绝；③引入 LocoRE 通过局部注意力放大强化最近上下文，形成闭环的连贯性保持机制。

**🔧 技术方法**

技术手段包括：梯度计算与 Hadamard 乘积得到显著性矩阵；top‑K 采样与自适应阈值筛选；对自注意力权重做按距离加权的乘法放大；实验在多种 LVLM（LLaVA‑1.5/13B、Qwen2‑VL‑7B/13B/32B、Intern‑VL‑7B/13B）上实现。

**📊 数据集**

数据集涵盖：①图像问答综合基准（LLaVA^W、MM‑Vet、VizWiz、ScienceQA）；②幻觉评测基准（POPE、CHAIR）；③通用生成评测（MME）。

**📈 对比分析**

与 OPERA、DOPRA、VCD、E‑AH 等现有方法对比，SGRS+LocoRE 在 POPE、CHAIR、MME 等多项指标上均实现了 SOTA 或显著提升（例如 POPE F1 由 85.4% 提升至 86.9%，CHAIR_S 由 38.4% 降至 35.6%）。

**⚠️ 局限性**

局限性包括：①需要手动调参（α、β）以权衡抑幻率与延迟；②过度拒绝可能降低生成多样性与自然度；③实验仅覆盖部分主流 LVLM，未验证在更大规模或其他任务上的通用性。

---

## 214. Beyond the Needle's Illusion: Decoupled Evaluation of Evidence Access and Use under Semantic Interference at 326M-Token Scale

**arXiv ID:** 2601.20276 | [PDF](https://arxiv.org/pdf/2601.20276v1)

**作者:** Tianwei Lin `[一作]` (EverMind), Yafeng Deng `[通讯]` (Shanda Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了面向大规模长上下文证据检索的对抗性针线式基准EverMemBench‑S（EMB‑S），并构建了多尺度参考语料阶梯及诊断协议。

**💡 创新点**

① 引入语义干扰的对抗性NIAH测试，混合多文档黄金证据与近似负样本；② 将证据访问与答案生成分离的诊断流程；③ 在统一的文档ID接口下，对原生长上下文模型与检索‑生成管线进行跨规模评估。

**🔧 技术方法**

使用密集向量检索（Qwen3‑Embedding‑8B、KaLM‑Embedding‑Gemma3‑12B）、BM25基线、reranker、LLM‑as‑a‑Judge（Grok‑4）以及人类+LLM验证流程。

**📊 数据集**

基于326M‑token MemoryBank（汇聚9个公开长上下文基准），构建483个验证查询；参考语料阶梯从64K到326M tokens。

**📈 对比分析**

通过检索器的R@1、SR@10、FR@10评估证据获取，随规模增大显著下降；多源检索比单源差距更大；在可容纳完整语料的长上下文模型中，答案质量随规模升高而下降，最高得分约为3.55/5。

**⚠️ 局限性**

样本量有限、文档ID接口可能引入格式偏差、依赖特定工具/评判器、未评估推理时的效率差异。

---

## 215. CURVE: Learning Causality-Inspired Invariant Representations for Robust Scene Understanding via Uncertainty-Guided Regularization

**arXiv ID:** 2601.20355 | [PDF](https://arxiv.org/pdf/2601.20355v1)

**作者:** Yue Liang `[一作]` (Shanghai Research Institute for Intelligent Autonomous Systems), Hong Chen `[通讯]` (Tongji University)

**通讯引用:** 478903 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CURVE 框架，结合变分不确定性建模与基于原型的因果干预，学习稀疏稳健的场景图结构，用以提升 OOD 泛化和低样本适配。

**💡 创新点**

创新点在于：①将不确定性视为剔除环境相关边的标准，执行不确定性引导的稀疏化；②使用可学习原型实现软特征空间后门调整，逼近环境分布并消除混杂；③端到端可微的结构学习与不确定性加权推理。

**🔧 技术方法**

采用变分推断、可学习原型池、软后门干预、基于 σ 的门控机制、Top‑K 稀疏化、GCN+LSTM 推理，以及基于不确定性的损失与校准技术。

**📊 数据集**

使用数据集：CARLA‑SR（训练）、DeepAccident（零样本 OOD）、DoTA（仿真到真实）。

**📈 对比分析**

与 RS2G、RS2V、Sg‑risk 等基线比较，CURVE 在 ID、OOD 和仿真到真实迁移中均表现出更高的 Accuracy、AUC、MCC，并且平均节点度数和边数显著降低，验证了稀疏结构的有效性。

**⚠️ 局限性**

局限性：①未给出完整的因果图识别理论，仅使用低秩原型近似；②对极端环境差异的适应性仍需更多数据验证；③原型数的选择对性能影响大，需要进一步自动化或理论指导。

---

## 216. MMSF: Multitask and Multimodal Supervised Framework for WSI Classification and Survival Analysis

**arXiv ID:** 2601.20347 | [PDF](https://arxiv.org/pdf/2601.20347v1)

**作者:** Chengying She `[一作]` (University of Chinese Academy of Sciences), Yun Bian `[通讯]` (Shanghai Advanced Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了MMSF框架，利用多模态WSI图像与临床数据同时完成肿瘤分型和生存预测。

**💡 创新点**

创新点包括：线性复杂度的Mamba‑MIL骨干、基于图的组织拓扑构建、临床数据嵌入模块以及分层（早期+后期）融合策略。

**🔧 技术方法**

采用的技术有：Mamba状态空间模型、图神经网络（GAT）、预训练的UNI2基础模型、SE注意力融合、Cox比例风险损失和自适应补丁选择等。

**📊 数据集**

使用的数据集包括分类任务的CAMELYON16与TCGA‑NSCLC，以及生存分析任务的TCGA‑BLCA、COAD、LUAD、STAD、KIRC。

**📈 对比分析**

通过与多种基线（ABMIL、TransMIL、EfficientMIL、CAML、MCAT、HSFSurv等）对比，分类ACC/AUC提升2.1–6.6%/2.2–6.9%，生存C‑index提升约8.9%/5.5%，在大多数数据集上取得最优或次优结果。

**⚠️ 局限性**

局限性包括：图构建耗时、对完整临床数据的依赖、未系统处理缺失值、仅处理单张WSI、对其他疾病或影像模态的泛化能力待进一步验证。

---

## 217. Demonstration-Free Robotic Control via LLM Agents

**arXiv ID:** 2601.20334 | [PDF](https://arxiv.org/pdf/2601.20334v1)

**作者:** Brian Y. Tsui `[一作]`, Tiffany J. Hwu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了FAEA框架，即将通用大语言模型（Claude Agent SDK）直接应用于机器人操作，无需任何演示或微调，实现了示范自由的操控；

**💡 创新点**

创新点在于证明通用LLM代理框架可以通过反复推理与程序合成，完成复杂的操作任务，完全跳过传统的演示收集与任务专用训练；

**🔧 技术方法**

使用技术包括ReAct循环、Claude Agent SDK的工具接口、Claude Opus 4.5 LLM、迭代式程序合成以及对仿真环境的特权状态访问；

**📊 数据集**

实验数据集涵盖LIBERO、ManiSkill3与MetaWorld三大机器人操作基准；

**📈 对比分析**

与VLA模型（如SmolVLA、π_0）和低样本模仿学习基线对比，FAEA在LIBERO、ManiSkill3、MetaWorld上分别取得约84.9%、85.7%和96–100%的成功率，竞争力接近或超过使用≤100演示的VLA模型，略低于大规模微调模型；

**⚠️ 局限性**

局限性包括：对亚厘米精度的插入等高精度任务表现不佳；LLM推理延迟（秒级）不适合实时控制；仅在仿真中验证、使用特权状态；只测试了Claude Agent SDK与单一LLM，缺乏跨代理/模型的泛化验证；

---

## 218. Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching

**arXiv ID:** 2601.20332 | [PDF](https://arxiv.org/pdf/2601.20332v1)

**作者:** Fengrui Zuo `[一作]` (University of Science and Technology of China), Xvehai Zhou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为Window‑Diffusion的无训练加速框架，用双窗口机制和阶段级KV缓存，显著减少扩散式语言模型推理时的全序列重复计算。

**💡 创新点**

创新点在于对扩散推理进行细粒度的token级分析，发现前缀局部性、远程上下文饱和以及解码token的阶段性稳定性，进而设计了窗口裁剪和缓存重用策略。

**🔧 技术方法**

采用双窗口token组织、KV缓存与周期性刷新、解码自适应终止等技术，并基于现有LLaDA和Dream模型实现。

**📊 数据集**

使用的公开数据集包括GSM8K、MATH、HumanEval和MBPP，实验涵盖基准和指令两种模型。

**📈 对比分析**

与Block‑Diffusion、DKV‑Cache、Fast‑dLLM等基线相比，Window‑Diffusion在保持相似准确率的前提下，实现了平均5×的吞吐量提升，最大可达99×的速度加速。

**⚠️ 局限性**

局限性包括对窗口大小和缓存刷新周期的敏感性，需要针对不同模型和任务进行超参调优；在极长序列或非掩码扩散模型上的效果尚待验证。

---

## 219. PsychePass: Calibrating LLM Therapeutic Competence via Trajectory-Anchored Tournaments

**arXiv ID:** 2601.20330 | [PDF](https://arxiv.org/pdf/2601.20330v1)

**作者:** Zhuang Chen `[一作]` (Central South University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 16270 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为 PsychePass 的统一框架，用来通过脚本化的咨询模拟与基于双向博弈的 Elo 排名，系统性评估并优化大语言模型（LLM）的治疗能力。

**💡 创新点**

创新点在于：①通过将对话轨迹脚本化，消除“过程漂移”；②采用瑞士制锦标赛和 Bradley‑Terry 评级，避免“标准漂移”，实现高效且可比的评价；③将博弈结果转化为奖励信号，构建闭环的轨迹‑RL 优化。

**🔧 技术方法**

使用的技术包括：LLM 客户端角色扮演、对话脚本与阶段化提问、瑞士制双向博弈、Elo 评分、Bradley‑Terry 统计模型、奖励模型训练以及基于群体相对策略优化（GRPO）的 on‑policy RL。

**📊 数据集**

数据集为 100 条从真实在线咨询平台（YiSum）采样并经过 LLM 扩充的匿名客户档案，构成 100 个高质量模拟客户；再使用 100 个独立档案用于 RL 训练与测试。

**📈 对比分析**

对比方法：在 12 维度上对 12 种 LLM（包括 GPT‑5.2、Gemini 3 Pro、Claude Opus 等）进行 4 轮瑞士制博弈，得到 Elo 排名；通过 Cohen’s κ 与人类专业咨询师的评估验证一致性（最高 1.000，弱-弱 0.75）。RL 优化后模型在 12 维度的胜率提升明显（总体 win:loss:tie 59.5:26.0:14.5），但在直接建议等维度略有下降。

**⚠️ 局限性**

局限性包括：①仅评估文本层面的咨询能力，无法覆盖语音、面部表情等多模态信号；②模拟访客驱动对话，未充分考察治疗师主动引导的能力；③高度依赖模拟数据，缺乏真实人类访客的情绪深度和不可预测性；④未提供人类治疗师的直接性能基准，因可完成对话数量有限；⑤RL 训练可能导致模型过度记忆特定对话模式。

---

## 220. Beyond Speedup -- Utilizing KV Cache for Sampling and Reasoning

**arXiv ID:** 2601.20326 | [PDF](https://arxiv.org/pdf/2601.20326v1)

**作者:** Zeyu Xing `[一作]` (Chinese University of Hong Kong), Sinno Jialin Pan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 33477 | [OpenAlex ID](https://openalex.org/A5082984558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将大语言模型的 KV 缓存重新用作轻量级表示，既能完成链式嵌入（Chain‑of‑Embedding）自评，又能实现基于 KV 缓存的快速/慢速思考切换，从而在不增加显存或 FLOPs 的前提下提升推理效率与可控性。

**💡 创新点**

创新点包括：① 将 KV 缓存视为可直接聚合的嵌入源，提出 KV‑CoE 方案；② 用 KV 缓存池化结果训练轻量 MLP 生成难度分数，实现 KVClassifier/ KV‑Generative 两种自适应思考切换；③ 所有方法均不需额外激活或模型改造，兼容现有推理框架。

**🔧 技术方法**

技术手段：KV 缓存聚合（平均/求和）、token‑/layer‑级嵌入构造、CoE 轨迹度量（Δr、Δθ）、轻量 MLP 难度预测、控制 token 注入实现思考模式切换；实验使用 Llama‑3.1‑8B‑Instruct、Qwen2‑7B‑Instruct、Qwen3‑8B、DeepSeek‑R1‑Distil‑Qwen‑14B 等 LLM；评估指标包括 AUROC、FPR95、准确率、token 数量等。

**📊 数据集**

数据集：MATH、TheoremQA 用于链式自评；GSM8K、MATH500 用于快速/慢速思考切换；MTEB 的 AmazonCounterfactualClassification、DBpediaClassification、FinancialPhrasebankClassification、TweetTopicSingleClassification 用于 KV 嵌入与专用嵌入模型对比；训练难度估计时使用 GSM8K、MATH 的原始训练集。

**📈 对比分析**

与基准比较：KV‑CoE 在 MATH、TheoremQA 上 AUROC 均超过 MaxProb/PPL/Entropy，虽略低于原 CoE 但仍显著提升；KV‑Classification 与 KV‑Generative 在 GSM8K、MATH500 上分别以 0.914/0.604 的准确率与 5.7× 的 token 降低（至 727 token），准确率下降仅 3.2% 左右；与全慢思考相比，token 省去 70% 以上，几乎不损失准确度。

**⚠️ 局限性**

局限性：KV 缓存本身并非全局可比的嵌入，存在方向性、低维度、缺乏对抗训练导致的离散性；对需要全局相似度检索或多类扩展的任务效果不佳；因此仅适用于受限候选集或局部轨迹比较的自评与动态推理控制。

---

## 221. SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger

**arXiv ID:** 2601.20312 | [PDF](https://arxiv.org/pdf/2601.20312v1)

**作者:** Kaiyuan Chen `[一作]` (Yunnan University), Xuejie Zhang `[通讯]` (Yunnan University)

**通讯引用:** 12592 | [OpenAlex ID](https://openalex.org/A5100707913)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对小型语言模型（SLM）的自适应过程优化（SAPO）方法，用以提升多步推理性能。

**💡 创新点**

创新点在于用错误相关负（ERN）启发的“首次错误检测”策略，动态定位最可能的错误步骤，仅对该步骤进行验证，避免了传统 Monte Carlo 逐步验证的高昂成本；同时引入自我验证与扩展机制，使过程监督更具稀疏性和高效性。

**🔧 技术方法**

技术方法包括：过程奖励模型（PRM）训练、首次错误检测与自我验证、过程监督信号生成、使用 ORPO 进行自我对齐，以及迭代探索–利用框架。

**📊 数据集**

使用的数据集包括 GSM8K、MBPP 以及新构建的 GSM_Process、MBPP_Process 两个过程级验证基准，评估指标为数学任务准确率、代码任务 Pass@1。

**📈 对比分析**

与 RFT、RFT+DPO、RPO、GRPO、SFT+GRPO、ORM 等自演化方法相比，SAPO 在数学和代码任务上均取得更高的准确率/Pass@1，并且随着迭代次数提升性能显著，且在过程监督上显著降低 FLOPs 与耗时。

**⚠️ 局限性**

局限性包括：仍需要在每一轮同步更新理由者与验证器，跨轮匹配时误差率上升；对模型能力较弱的 SLM（如 LLaMA、Gemma）提升有限；以及在极大规模任务或更复杂推理链时，首次错误定位的准确性仍可能不足。

---

## 222. SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips

**arXiv ID:** 2601.20309 | [PDF](https://arxiv.org/pdf/2601.20309v1)

**作者:** Jiahuan Yu `[一作]` (University of Illinois Urbana-Champaign), Minjia Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4579 | [OpenAlex ID](https://openalex.org/A5077768924)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Superchip（如NVIDIA GH200）上的LLM推理，提出了SLO感知的主动旋转调度器和高性能的KV缓存旋转引擎，以解决GPU内存不足导致的队首阻塞问题。

**💡 创新点**

创新点包括：① OS灵感的主动旋转调度器（使用Virtual Lag Time衡量SLO滞后并采用Largest‑VLT‑First策略）实现SLO感知的抢占与复位；② 针对NVLink‑C2C全双工特性的KV缓存旋转引擎，采用块优先布局、批量传输内核、及时块旋转等手段消除数据竞争并最大化带宽利用。

**🔧 技术方法**

核心技术包括：SLO感知调度（VLT、LVF）、NVLink‑C2C全双工数据传输、块优先KV缓存布局、批量传输内核、跨迭代管线化、以及基于Python/C++实现的vLLM扩展。

**📊 数据集**

使用的模型有LLaMA‑3‑8B、Qwen2.5‑32B、Mixtral‑8x7B；数据集包括ShareGPT和LMSYS‑Chat‑1M，并通过Poisson到达率模拟不同RPS。

**📈 对比分析**

与vLLM、LightLLM、LTR、NEO等主流LLM推理框架对比，实验显示在高请求率下TTFT SLO满足率提升高达74.7%，TBT SLO保持与或优于基线，吞吐量与vLLM相当或略高，证明了系统在SLO感知和带宽利用方面的优势。

**⚠️ 局限性**

局限性：依赖Superchip的NVLink‑C2C架构，缺乏对PCIe环境的优化；对模型大小的支持仍受GPU/CPU内存总量限制；调度参数（α、β_B、β_F、B_xfer）需要手动调优以适应不同场景。

---

## 223. OSDEnhancer: Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion

**arXiv ID:** 2601.20308 | [PDF](https://arxiv.org/pdf/2601.20308v1)

**作者:** Shuoyan Wei `[一作]` (Beijing Jiaotong University), Huihui Bai `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1648 | [OpenAlex ID](https://openalex.org/A5030238978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种高效的一步扩散框架OSDEnhancer，用于实现场景下的时空视频超分辨率（STVSR）；

**💡 创新点**

创新点包括：1）将STVSR拆分为时间细化与空间增强的Mixture-of-Experts（TR-SE MoE）架构，实现专门化并行学习；2）设计双向可变形VAE解码器，跨尺度跨帧聚合和补偿；3）采用一阶扩散直接从低分辨率、低帧率视频生成高分辨率、高帧率视频，显著降低采样步骤；

**🔧 技术方法**

技术主要有：预训练视频扩散模型（CogVideoX/DiT），LoRA微调、残差感知监督、光流对齐损失、DISTS结构损失、无参考质量评估（NQA）以及可变形卷积与交叉注意力实现的双向解码；

**📊 数据集**

使用了多种数据集进行训练与评估：HQ‑VSR、Adobe240、DIV2K、RealBasicVSR合成的合成降质数据，以及真实场景数据集VideoLQ、MVSR4x、GoPro、UDM10、SPMCS、YouHQ40；

**📈 对比分析**

与多种两阶段与单阶段STVSR方法（如LDMVFI+DAM‑VSR、VEnhancer、VideoINR、MoTIF、BF‑STVSR等）比较，OSDEnhancer在大多数指标（PSNR/SSIM/LPIPS/FloLPIPS/MUSIQ/CLIP‑IQA/FasterVQA/DOVER）上实现了最优或接近最优表现，且推理速度比现有扩散模型快约7倍；

**⚠️ 局限性**

局限性主要体现在：1）仍需大量算力进行预训练与微调；2）在极端运动或极低分辨率场景下，细节重建可能受限；3）对输入噪声/压缩伪影的鲁棒性需进一步验证；

---

## 224. Endogenous Reprompting: Self-Evolving Cognitive Alignment for Unified Multimodal Models

**arXiv ID:** 2601.20305 | [PDF](https://arxiv.org/pdf/2601.20305v1)

**作者:** Zhenchen Tang `[一作]` (University of YYY), Jing Dong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过自演进的两阶段内部循环（RLVR+RLMT），在统一多模态模型中实现了从被动理解到主动生成推理的转换，填补了认知差距。

**💡 创新点**

创新点在于：①提出Endogenous Reprompting，使模型自身生成与生成器先验高度对齐的描述；②使用仅300条视觉指令样本即可自我训练的SEER框架；③在生成前端引入内部评估器和思考环节，改进传统RLHF只针对低层像素的做法。

**🔧 技术方法**

技术核心包括：RLVR（可验证奖励的强化学习）、RLMT（基于模型奖励的思考）、GRPO（组相对策略优化）、Curriculum Learning、MAE编码器、VAE生成器、KL正则化。

**📊 数据集**

主要使用的样本：Visual Instruction Elaboration（300条）作为压缩代理任务；评测集包括MME、POPE、GQA、MMMU、SEEDBench、GenEval、DPG-Bench；对比基线包括外部重写器（BeautifulPrompt、PromptEnhancer）和大型多模态LLM（GPT‑5.2、Gemini3、Qwen3max）。

**📈 对比分析**

与外部重写器和SOTA MLLM进行人类对比，SEER的胜率>0.5，尤其在Hard指令上明显领先；在生成质量评估（GenEval/DPG-Bench）中保持或略优，且单词数显著减少（约22.94）。

**⚠️ 局限性**

局限性：仅针对视觉指令细化，样本极少；对更大规模、多样性指令的泛化尚未充分验证；模型规模较小，缺乏对高分辨率或复杂生成任务的评估。

---

## 225. MiLorE-SSL: Scaling Multilingual Capabilities in Self-Supervised Models without Forgetting

**arXiv ID:** 2601.20300 | [PDF](https://arxiv.org/pdf/2601.20300v1)

**作者:** Jing Xu `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9305 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种MiLorE-SSL框架，实现多语言自监督模型的轻量化持续扩展，能在保留已有语言能力的同时快速加入新语言。

**💡 创新点**

创新点包括：①将LoRA低秩适配与软Mixture-of-Experts（MoE）相结合，实现参数高效的专家共享与语言特化；②采用少量重放数据缓解灾难性遗忘；③在保持训练成本低的前提下，显著提升多语言ASR和LID性能。

**🔧 技术方法**

技术手段：LoRA低秩矩阵更新、软路由的MiLorE模块、有限重放策略、HuBERT基础模型、ML-SUPERB评测框架。

**📊 数据集**

使用的数据集包括：CommonVoice（zh-CN, zh-HK, yue, en）、Thchs-30、AISHELL-1/3、Fleurs 进行评测；重放样本采用约100小时英文 CommonVoice 数据。

**📈 对比分析**

与 mHuBERT-147 和 HuBERT-Large 进行对比，MiLorE-SSL 在 CommonVoice 和 Fleurs 的 ASR CER 及 LID ACC 均显著下降/提升，达到约10% CER（多语言）和 99.4% ACC，且仅使用 2.14% 可训练参数。

**⚠️ 局限性**

局限性：目前仅验证了三种语言，专家数量与低秩维度受限；在极低资源或非常不同语言的扩展效果尚未评估；重放样本仍需人工挑选，且对长期多语言序列扩展的鲁棒性未知。

---

## 226. GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction

**arXiv ID:** 2601.20331 | [PDF](https://arxiv.org/pdf/2601.20331v1)

**作者:** Mai Su `[一作]` (Peking University), Guoping Wang `[通讯]` (Peking University)

**通讯引用:** 16199 | [OpenAlex ID](https://openalex.org/A5100366298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过引入基于高斯可见性的多视角几何一致性约束和逐层四叉树校准的单视深度约束，改进3D高斯投影建模的表面重建

**💡 创新点**

创新点包括：1）利用高斯可见性信息推理跨视点可见区域，解决传统基于深度投影一致性受深度误差影响的问题；2）在单视深度先验上采用四叉树分块自适应块级仿射校准，逐步消除尺度不一致与局部误差

**🔧 技术方法**

使用3D高斯投影、alpha混合渲染、Depth‑Anything V2单视深度网络、四叉树分块对齐、深度与表面一致性损失以及多视角光度一致性等技术

**📊 数据集**

在DTU室内数据集和Tanks & Temples（TNT）室外数据集上进行评估

**📈 对比分析**

与SuGaR、2DGS、PGSR、QGS等方法对比，DTU上平均Chamfer Distance降至0.50 mm（超过14/15场景最佳），TNT上平均F1‑score提升至0.53，性能明显优于前沿方法

**⚠️ 局限性**

仍受限于对单视深度先验的依赖，尤其在光照稀疏或纹理缺失区域可能出现深度误差，导致表面细节不足

---

## 227. ECG-Agent: On-Device Tool-Calling Agent for ECG Multi-Turn Dialogue

**arXiv ID:** 2601.20323 | [PDF](https://arxiv.org/pdf/2601.20323v1)

**作者:** Hyunseung Chung `[一作]` (KAIST), Edward Choi `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 ECG-Agent，一种支持多轮 ECG 对话的工具调用式 LLM 代理，并构建了 ECG-MTD 数据集；

**💡 创新点**

创新点在于将工具调用与 ECG 诊断结合，解决单轮、体积大、缺乏测量精度等问题，并实现了可在设备上部署的小模型；

**🔧 技术方法**

技术包括 LLM 工具调用框架、三种工具（分类、测量、解释）、LoRA 微调、4‑bit 量化、SpectralX、Neurokit2 等；

**📊 数据集**

使用 PTB‑XL 电图数据生成多种导联（12导、Lead I、Lead II）以及通过 Gemini‑2.5‑Flash 生成的对话；

**📈 对比分析**

通过与 Gemini‑2.5‑Flash、PULSE、GEM 等基线比较，ECG‑Agent 在准确率、完整性、下一步动作预测和对话质量上均优于基线，且 1B/3B 小模型几乎匹配 8B/32B 大模型；

**⚠️ 局限性**

局限性包括对 12 导联的解释工具缺失、在复杂多导联诊断时仍易产生幻觉、评估主要依赖 LLM‑as‑Judge 及自动化评价，缺少大规模真实临床验证。

---

## 228. VersaQ-3D: A Reconfigurable Accelerator Enabling Feed-Forward and Generalizable 3D Reconstruction via Versatile Quantization

**arXiv ID:** 2601.20317 | [PDF](https://arxiv.org/pdf/2601.20317v1)

**作者:** Yipu Zhang `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 34025 | [OpenAlex ID](https://openalex.org/A5100441678)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套算法与硬件协同设计方案，即 VersaQ‑3D，用于实现基于 Visual Geometry Grounded Transformer（VGGT）的即时、可泛化 3D 重建，兼顾高精度与低功耗。

**💡 创新点**

创新点包括：① 无校准的正交变换量化（WHT+ DCT），能在 4‑bit 量化下保持 98‑99% 的精度；② 统一可重构多精度计算单元（BF16/INT8/INT4），共享 systolic 流程；③ 针对 VGGT 全局注意力的两阶段重计算分块，显著降低内存压力并提升吞吐率。

**🔧 技术方法**

采用正交变换量化、INT‑基 WHT 与 DCT、BF16 与 INT 多精度混合、systolic 与 SIMD 计算架构、INT4/INT8 互补累加、非线性算子 BF16 单元、两阶段重计算分块技术；实现基于 Synopsys 28nm RTL 的加速器。

**📊 数据集**

使用 Co3Dv2（相机位姿评估）和 7‑Scenes（点云重建评估）两个公开数据集进行实验。

**📈 对比分析**

与 Jetson XNX / ONX GPU 进行对比，W4A4 模式下速度提升 5.2×–10.8×，能效提升 ~100×；在 W4A8 模式下保持 98‑99% 全精度；量化准确率在 W4A4 下相较 RTN 提升 2.39×、相较 QuaRot 提升 1.61×。

**⚠️ 局限性**

主要限制包括：对极低位权重/激活仍存在精度下降；长序列全局注意力的分块方案虽有效但仍受限于最大序列长度；硬件面积与功耗仍高于极简化实现，难以进一步压缩；在极端动态或高帧率场景下精度衰减可能更明显。

---

## 229. AMA: Adaptive Memory via Multi-Agent Collaboration

**arXiv ID:** 2601.20352 | [PDF](https://arxiv.org/pdf/2601.20352v1)

**作者:** Weiquan Huang `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5114038310)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理协作的自适应记忆框架 AMA，用以支持 LLM 代理的长时交互与复杂推理

**💡 创新点**

创新点在于：①多粒度记忆存储与检索；②检索路由与验证由四个专门代理完成；③逻辑驱动的 Refresher 解决记忆冲突，实现动态更新与删除

**🔧 技术方法**

技术包括：LLM 生成的记忆构造、向量检索、意图向量驱动的检索路由、逻辑一致性检查与更新、基于嵌入的多层检索

**📊 数据集**

评估数据集：LoCoMo 与 LongMemEvals（包括单/多会话、时间推理、知识更新等任务）

**📈 对比分析**

与 FullContext、RAG、LangMem、MemGPT、Zep、Mem0、Nemori 等基线对比，AMA 在 LoCoMo 的 LLM‑Score 最高（0.774/0.805），在 LongMemEvals 平均准确率 0.698，显著优于最强基线；同时仅占 19% 传统上下文 token，减少 80% 令牌消耗

**⚠️ 局限性**

局限性：多代理协作带来额外计算开销；对小型模型的效率与通用性仍待提升

---

## 230. PalmBridge: A Plug-and-Play Feature Alignment Framework for Open-Set Palmprint Verification

**arXiv ID:** 2601.20351 | [PDF](https://arxiv.org/pdf/2601.20351v1)

**作者:** Chenke Zhang `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 94250 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种名为PalmBridge的特征空间对齐模块，用于开放集掌纹验证。

**💡 创新点**

创新点在于利用向量量化学习紧凑且正交的代表向量，对查询与登记特征做最近邻映射并线性融合，抑制域偏移导致的噪声，同时保持身份区分。

**🔧 技术方法**

采用向量量化（VQ）、特征一致性损失、正交正则化，并结合多种主干网络（ResNet‑18、DHN、CompNet、CO3Net、CCNet、SF2Net）。

**📊 数据集**

在四个掌纹数据集（IITD、PolyU、Tongji、PalmVein）以及掌纹血管数据集PalmVein上进行实验。

**📈 对比分析**

与无数据增强基线、PalmRSS、C‑LMCL、UAA等对比，PalmBridge在内部开放集和跨数据集开放集均显著降低EER，并在闭集验证中实现几乎零EER。

**⚠️ 局限性**

局限性：代表向量覆盖有限，过度映射可能损失细粒度身份信息；在极端域偏移时仍可能出现映射冲突。

---

## 231. Bridging the Applicator Gap with Data-Doping:Dual-Domain Learning for Precise Bladder Segmentation in CT-Guided Brachytherapy

**arXiv ID:** 2601.20302 | [PDF](https://arxiv.org/pdf/2601.20302v1)

**作者:** Suresh Das `[一作]` (Narayana Superspeciality Hospital), Sayantari Ghosh `[通讯]` (National Institute of Technology Durgapur)

**通讯引用:** 402 | [OpenAlex ID](https://openalex.org/A5002804599)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并验证一种双域学习（dual‑domain learning）方法，即在训练集里加入少量带有应用器（WA）的CT图像与大量不带应用器（NA）的CT图像，以提升放射治疗中膀胱分割的鲁棒性。

**💡 创新点**

创新点在于将稀缺的WA数据与丰富的NA数据进行“数据掺杂”（data‑doping），通过少量（10‑30%）WA样本即可让模型达到与仅用WA数据训练相当的性能，解决了应用器引起的协变量偏移和数据稀缺问题。

**🔧 技术方法**

采用多种深度学习分割网络（U‑Net、U‑Net++、Half‑UNet、RRDB‑U‑Net、DC‑UNet、Attention‑U‑Net），配合标准数据增强和Dice损失函数进行训练与评估；同时在轴向、冠状、矢状三平面上进行实验。

**📊 数据集**

数据集由同一机构收集的20例无应用器CT扫描（NA）和20例带应用器CT扫描（WA）组成，全部进行专家标注并做体素间距、强度归一化。

**📈 对比分析**

对比仅用NA训练、仅用WA训练和不同NA:WA比例掺杂（如7:3）的模型，评估指标为Dice系数（DSC）和交并比（IoU）。结果显示：仅NA训练时IoU约0.76，加入10‑30% WA后IoU提升至≈0.92，DSC可达0.94，显著优于单一域训练。

**⚠️ 局限性**

局限性包括：样本量有限（仅20例WA），来自单中心，缺乏外部验证；仅研究膀胱分割，未涉及其他器官；对掺杂比例的最佳值依赖具体数据和网络，可能不适用于其他病种或扫描模态。

---

## 232. Towards Compact and Robust DNNs via Compression-aware Sharpness Minimization

**arXiv ID:** 2601.20301 | [PDF](https://arxiv.org/pdf/2601.20301v1)

**作者:** Jialuo He `[一作]` (Hong Kong University of Science and Technology), Huangxun Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5014199941)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种压缩感知的锐度最小化框架（C‑SAM），在训练时通过扰动剪枝掩码实现结构平坦化，以同时实现模型压缩与对语义扰动的公认鲁棒性。

**💡 创新点**

创新点在于将锐度最小化从参数空间迁移到掩码空间，并构造三项损失（稳定性、比率与一致性）以保证子网络在压缩与语义变形下的稳定性，解决了传统 SAM 与剪枝先后顺序导致鲁棒性退化的问题。

**🔧 技术方法**

采用的技术包括：随机掩码扰动+稳定性损失、语义扰动生成（基于扩散模型的潜在空间变换）、比率损失（相对分类边距的鲁棒性比例）、一致性损失（软硬掩码 KL 对齐）以及 L1 稀疏正则；训练分为预训练、鲁棒掩码搜索、最终二值化微调。

**📊 数据集**

实验使用 CelebA‑HQ、Flowers‑102（两级样本规模）和 CIFAR‑10‑C，评估不同网络结构（ResNet‑18、GoogLeNet、MobileNet‑V2）和剪枝比例（50%/70%）。

**📈 对比分析**

与传统剪枝、对抗训练、S^2‑SAM 等基线相比，C‑SAM 在保持准确率与原模型相近的同时，验证集的概率认证准确率（PCA）提升最高可达 42%，并在多种结构化/非结构化剪枝方案中表现出更优的鲁棒性。

**⚠️ 局限性**

局限性包括：需要使用扩散模型生成语义扰动，导致额外的计算开销；对掩码噪声和安全阈值等超参敏感；在极高压缩比例下仍可能出现性能下降。

---

## 233. Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction

**arXiv ID:** 2601.20299 | [PDF](https://arxiv.org/pdf/2601.20299v1)

**作者:** Tianyi Alex Qiu `[一作]` (Center for Human-Compatible Artificial Intelligence), Cameron Allen `[通讯]` (Center for Human-Compatible Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种基于同侪预测的LLM评估与训练框架，无需强监督即可抵御欺骗。

**💡 创新点**

实现了激励兼容、无需强监督的评估机制，并揭示了逆向缩放特性，即模型能力差距越大，抵御欺骗越强。

**🔧 技术方法**

采用同侪预测机制、对数概率打分、对比学习（DPO）等技术，以及理论证明和大规模实验。

**📊 数据集**

使用由MATH、MMLU、MMLU-PRO、ARC、OpenBookQA、RACE、MCTest等融合的约3.7万题目，涵盖85个领域的问答数据。

**📈 对比分析**

与LLM-as-a-Judge等基线对比，实验显示在训练和评估中同侪预测能恢复绝大部分因欺骗导致的准确率下降，并在模型能力差距大时表现出更高的误导抵抗性，整体性能优于传统方法。

**⚠️ 局限性**

局限性包括未解决参与者之间的合谋问题、对专家/参与者规模的依赖，以及对先验分布一致性的假设等。

---

## 234. MARE: Multimodal Alignment and Reinforcement for Explainable Deepfake Detection via Vision-Language Models

**arXiv ID:** 2601.20433 | [PDF](https://arxiv.org/pdf/2601.20433v1)

**作者:** Wenbo Xu `[一作]` (Sun Yat-sen University), Jiantao Zhou `[通讯]` (University of Macau)

**通讯引用:** 9084 | [OpenAlex ID](https://openalex.org/A5037979193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 MARE 框架，通过视觉‑语言模型结合多模态对齐和强化学习实现可解释的 Deepfake 检测与推理。

**💡 创新点**

创新点在于：①构建多模态对齐数据集并设计五维奖励函数（格式、准确度、文本相关度、ROI 与对齐），②采用 RLHF 与强化学习激励 VLM 生成文本‑空间对齐的推理，③提出Forgery Disentanglement Module，将面部图像分离身份、结构与伪造痕迹特征，提高对伪造痕迹的捕获。

**🔧 技术方法**

使用技术包括：视觉‑语言模型（Qwen2.5‑VL、InternVL2.5 等）、GRPO 强化学习、RLHF、文本编码器（SentenceTransformer）及特征解耦与对抗分类器。

**📊 数据集**

实验数据集涵盖：传统 Deepfake 数据集（FF++、Celeb-DF、WDF、DFDC、DFD）以及自建的 Deepfake 多模态对齐数据集 DMA。

**📈 对比分析**

在 intra‑dataset 与 fuse‑dataset 对比中，与多种 SOTA 方法对标，MARE 在绝大多数数据集上取得最高或接近最优的准确率与 AUC，并在推理任务中获得最高的 Acc/F1 分数。

**⚠️ 局限性**

局限性：模型仍受限于训练数据规模与伪造技术多样性，对极细微或新颖伪造手段的检测效果尚待提升；同时，对齐评估指标仍需进一步完善。

---

## 235. GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering

**arXiv ID:** 2601.20429 | [PDF](https://arxiv.org/pdf/2601.20429v1)

**作者:** Junseo Lee `[一作]`, Jaewoong Sim `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

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

## 236. Nonlinear Dimensionality Reduction with Diffusion Maps in Practice

**arXiv ID:** 2601.20428 | [PDF](https://arxiv.org/pdf/2601.20428v1)

**作者:** Sönke Beier `[一作]`, Karoline Wiesner `[通讯]` (University of Potsdam)

**通讯引用:** 2145 | [OpenAlex ID](https://openalex.org/A5084296874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对Diffusion Map降维方法进行实践性综述，系统评估预处理、参数设置以及组件选择对嵌入结果的影响，并提出基于神经重构误差的组件重要性评估。

**💡 创新点**

创新点在于揭示传统先验假设（如取前k个组件）常导致误判，提出NRE（Neural Reconstruction Error）方法能量化每个组件的重建贡献，并展示通过NRE重新排序组件可显著提升降维效果。

**🔧 技术方法**

核心技术包括Gaussian核构建邻域图、α参数的无向化、Markov转移矩阵归一化、特征值分解、Diffusion距离等谱方法，并结合深度学习自动编码器实现近似逆映射以计算NRE。

**📊 数据集**

实验主要使用经典的Switzerland Roll（Swiss Roll）数据集（约3000点，三维嵌入的二维曲面），并在不同预处理（缩放、归一化、冗余维度）与参数设置下进行对比。

**📈 对比分析**

与传统PCA、Isomap、LLE等方法比较时，Switzerland Roll在正确选择ε、α、N等邻域参数后，Diffusion Map能完整展开曲面；NRE指标显示仅包含Ψ₁和Ψ₅即可达到几乎零重建误差，优于仅取前k个连续谱分量的做法。

**⚠️ 局限性**

局限性包括：对参数（尤其ε、α、N）高度敏感、对离散与连续变量混合以及冗余维度的误导作用；时间参数t在实践中作用有限；且NRE方法需训练神经网络，计算成本较高。

---

## 237. Let's Roll a BiFTA: Bi-refinement for Fine-grained Text-visual Alignment in Vision-Language Models

**arXiv ID:** 2601.20419 | [PDF](https://arxiv.org/pdf/2601.20419v1)

**作者:** Yuhao Sun `[一作]` (University of Melbourne), Feng Liu `[通讯]` (University of Melbourne)

**通讯引用:** 9974 | [OpenAlex ID](https://openalex.org/A5001956517)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BiFTA 方法，在零样本分类中对图像局部视图和 LLM 生成的文本描述进行去冗余处理，提升视觉-文本对齐效果。

**💡 创新点**

创新点在于：①通过 IoU 阈值过滤重叠图像裁剪，实现视图去冗余；②通过文本描述的余弦相似度阈值和 Top‑k 选择实现语义去冗余，从而让视觉与文本特征更具多样性和区分度。

**🔧 技术方法**

使用 CLIP 预训练模型（ViT‑B/32、ViT‑B/16、ViT‑L/14、RN‑50、RN‑101 等）作为基准，结合随机裁剪、IoU 过滤、文本余弦相似度过滤、Top‑k 选取等技术；对比了 WCA、CuPL、Waffle、CLIP‑E、CLIP‑D 等基线。

**📊 数据集**

在 6 个零样本分类基准上进行评估：ImageNet、CUB、Oxford Pets、DTD、Food101、Place365。还在附录中验证了对 ALIGN、AltLIP、GroupViT、SigLIP 等其他 VLM 的适用性。

**📈 对比分析**

与 WCA 等现有最优方法相比，BiFTA 在多数数据集均实现 0.5%–3.3% 的准确率提升（例如 DTD +3.3%，CUB +1.5%），整体平均提升约 1.0%。实验还表明单模态去冗余（仅 VR 或仅 DR）已能提升性能，双模态去冗余可获得最佳效果。

**⚠️ 局限性**

局限性：①去冗余过程主要在离线阶段完成，虽然对推理时间影响小，但仍需额外预处理；②在大规模、高多样性数据集（如 ImageNet）提升幅度有限；③方法主要针对 CLIP 及其变体，未在非 CLIP VLM 上作深入探讨。

---

## 238. On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents

**arXiv ID:** 2601.20404 | [PDF](https://arxiv.org/pdf/2601.20404v1)

**作者:** Jai Lal Lulla `[一作]` (Singapore Management University), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 4917 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比在GitHub PR任务中是否存在AGENTS.md文件，实证评估其对AI编码代理（Codex）的运行时长与Token消耗的影响。

**💡 创新点**

首次采用配对实验设计与真实仓库任务，量化AGENTS.md对代理运营效率的具体效益，并探讨其在实际开发中的意义。

**🔧 技术方法**

使用OpenAI Codex代理、Docker沙箱、LLM自动生成任务描述、Wilcoxon符号秩检验等技术进行实验与统计。

**📊 数据集**

基于26个符合条件的仓库（共10个仓库、124个PR），每个PR限制在≤100行变更、≤5文件，并取其历史合并前的仓库快照。

**📈 对比分析**

在同一PR的两种条件（有/无AGENTS.md）下运行，测量Token使用和墙钟时间；结果显示有AGENTS.md时平均时间减少20.27%，中位数减少28.64%；输出Token平均减少20.08%，中位数减少16.58%。

**⚠️ 局限性**

仅使用单一代理（Codex）和小规模任务，样本有限；未评估代码正确性、对大规模或多模块项目的适用性，以及不同模型/框架的普适性。

---

## 239. How Software Engineering Research Overlooks Local Industry: A Smaller Economy Perspective

**arXiv ID:** 2601.20382 | [PDF](https://arxiv.org/pdf/2601.20382v1)

**作者:** Klara Borowa `[一作]` (Warsaw University of Technology), Lech Madeyski `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了ICSE 2026社区调查问卷，结合波兰等小经济体研究者视角，阐述研究与产业间的鸿沟，并提出改进建议。

**💡 创新点**

首次将小经济体视角与ICSE调查结合，用反思性主题分析揭示研究与产业脱节的根源，并给出针对性对策。

**🔧 技术方法**

使用反思性主题分析（reflexive thematic analysis）方法，手工编码、子主题生成、主题聚合。

**📊 数据集**

ICSE 2026社区调查问卷（280名参与者，主要为欧洲院士）

**📈 对比分析**

无基准比较，采用定性分析得出结论，未涉及数值性能评估。

**⚠️ 局限性**

样本主要为欧洲院士，波兰代表性不足；方法主观性高，缺乏量化评估。

---

## 240. Policy of Thoughts: Scaling LLM Reasoning via Test-time Policy Evolution

**arXiv ID:** 2601.20379 | [PDF](https://arxiv.org/pdf/2601.20379v1)

**作者:** Zhengbo Jiao `[一作]` (Binjiang Institute), Meng Han `[通讯]` (Binjiang Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Policy of Thoughts (PoT) 框架，将 LLM 推理过程视为实时的策略演化，使用 MCTS 与 GRPO 在临时 LoRA 上实现测试时的在线优化。

**💡 创新点**

创新点在于打破冻结策略假设，将执行反馈内化为实时策略更新，形成闭环的 conjecture‑refutation 迭代，提升小模型推理稳定性。

**🔧 技术方法**

使用的技术包括 Monte Carlo Tree Search (MCTS)、Group Relative Policy Optimization (GRPO)、低秩适配器 LoRA，以及基于执行反馈的奖励机制。

**📊 数据集**

在 LiveCodeBench、HumanEval、MBPP、ICPC 等代码推理基准数据集上进行评测。

**📈 对比分析**

与多种自检、搜索与大型模型基线比较，PoT 在 4B 模型上平均准确率达到 58.98%，超过 GPT‑4o、Claude‑Opus‑4 等 50 倍规模模型，显著提升。

**⚠️ 局限性**

局限包括对超参数敏感、在极大搜索空间下仍受限、以及仅在编程推理任务验证，跨任务泛化仍待进一步探索。

---

## 241. Hopes and Fears -- Emotion Distribution in the Topic Landscape of Finnish Parliamentary Speech 2000-2020

**arXiv ID:** 2601.20424 | [PDF](https://arxiv.org/pdf/2601.20424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 242. TINNs: Time-Induced Neural Networks for Solving Time-Dependent PDEs

**arXiv ID:** 2601.20361 | [PDF](https://arxiv.org/pdf/2601.20361v1)

**作者:** Chen-Yang Dai `[一作]` (National Yang Ming Chiao Tung University), Chieh-Hsin Lai `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种时间诱导神经网络（TINNs）来解决时变偏微分方程（PDE）中传统空间–时间PINNs存在的时间耦合问题，能够在不增加过多参数的情况下，让网络权重随时间平滑演化。

**💡 创新点**

创新点在于：① 将时间作为参数空间的轨迹而非输入特征；② 采用层级式的时间嵌入（2L维代码）并通过逐元素仿射映射构造完整权重，极大降低参数量；③ 针对非线性最小二乘形式的PINN损失，使用Levenberg–Marquardt（LM）二阶优化，显著提升收敛速度和稳定性。

**🔧 技术方法**

技术包括：物理信息神经网络（PINN）、时变权重参数化、层级时间嵌入、Levenberg–Marquardt二阶优化、自动微分与高阶时间导数计算。

**📊 数据集**

在五个经典时变PDE基准上验证：粘性Burgers方程、Allen–Cahn方程、Klein–Gordon方程、Korteweg–de Vries方程和波动方程，采用统一的训练点集与迭代次数。

**📈 对比分析**

与标准PINN、CoPINN*和PirateNet SOAP三种强基线相比，TINNs在相同计算预算下实现了2.9–10.5倍的误差提升（相对L2误差降低）且训练时间缩短3–8倍，且模型参数数目仅为传统方法的1–2%，在所有任务上均表现出更高的准确性和更快的收敛。

**⚠️ 局限性**

局限性包括：① LM优化在参数量过大时成本高，限制了模型可扩展性；② 时变参数化对极端非平滑或多尺度时间演化的适应性仍需进一步验证；③ 目前实验仅覆盖中等规模网络和一维/二维空间问题，尚未在高维或复杂几何域上进行测试。

---

## 243. Remember Me, Not Save Me: A Collective Memory System for Evolving Virtual Identities in Augmented Reality

**arXiv ID:** 2601.20437 | [PDF](https://arxiv.org/pdf/2601.20437v1)

**作者:** Tongzhou Yu `[一作]` (China Academy of Art), Han Lin `[通讯]` (Wenzhou University)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5101867733)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过 AR 与 LLM 交互创建了一个可在公共对话中逐步形成数字身份的虚拟市民系统。

**💡 创新点**

核心创新包括动态集体记忆（DCM）模型的叙事张力机制、状态反射化身实现环境可解释性，以及基于地理文化的上下文锚定。

**🔧 技术方法**

技术实现依赖 Unity/ARFoundation 前端、Python/FastAPI 后端、ChatGLM 与 VITS 对话引擎、ChromaDB/FAISS 向量存储，以及多模态 LLM 对合成图像的解析。

**📊 数据集**

数据集为在 2024 年济南国际双年展期间收集的约 2500 条公开对话记录及参与者拍摄的 AR 合成图像。

**📈 对比分析**

与传统 RAG 系统对比，DCM 在 2500 次交互中保持了 70‑82% 的主题一致性，并通过 Apply Magic Sauce 评估显示虚拟市民稳定出现 ISTP 维度人格，性能优于单一 RAG。

**⚠️ 局限性**

局限性包括仅在艺术展览场景验证、观察时长短、缺乏跨文化对比实验，以及对系统长期演化和安全性的评估不足。

---

## 244. Rethinking Thread Scheduling under Oversubscription: A User-Space Framework for Coordinating Multi-runtime and Multi-process Workloads

**arXiv ID:** 2601.20435 | [PDF](https://arxiv.org/pdf/2601.20435v1)

**作者:** Aleix Roca `[一作]` (Barcelona Supercomputing Center), Vicenç Beltran `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 759 | [OpenAlex ID](https://openalex.org/A5041767872)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了用户空间线程调度框架 USF 并实现协作调度策略 SCHED_COOP，解决 HPC 与 AI 负载中的线程过度投放以及多运行时/多进程之间的竞争与干扰。

**💡 创新点**

创新点在于：通过在 glibc 中插装实现完全用户空间、无内核修改、支持多进程和 TLS 的调度框架；并设计了 SCHED_COOP 协作调度，避免 LHP/LWP 并显著提升过度投放场景性能。

**🔧 技术方法**

使用了 nOS‑V 线程/任务库、glibc 扩展拦截 POSIX 同步 API、共享内存多进程调度、NUMA-aware affinity、协作式无抢占调度以及对阻塞系统调用的插装。

**📊 数据集**

评测数据集包括：矩阵乘法、Cholesky 分解、PyTorch 推理（LLaMA‑3.2、GPT‑2、RoBERTa）、AI 微服务请求仿真、LAMMPS 与 DeePMD‑kit 的分子动力学模拟。

**📈 对比分析**

与 Linux 默认调度器、手工集成 nOS‑V 以及不同资源划分的多进程/多运行时方案进行对比；在过度投放场景中 SCHED_COOP 的性能提升范围为 2–28%（最高 2.4×），在 AI 微服务与 MD 组装等真实工作负载中分别提升 2.4×、4% 以上。

**⚠️ 局限性**

局限性：只能拦截 glibc 的同步 API，需修改自定义忙等阻塞；对 I/O 系统调用未做处理；共享内存仅对同用户/组安全；必须重新编译依赖 glibc 的应用；无法支持非 POSIX 调度或内核级协作。

---

## 245. Guiding the Recommender: Information-Aware Auto-Bidding for Content Promotion

**arXiv ID:** 2601.20422 | [PDF](https://arxiv.org/pdf/2601.20422v1)

**作者:** Yumou Liu `[一作]` (Shanghai Jiao Tong University), Guihai Chen `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种双目标内容推广竞价框架，既最大化即时点击价值，又通过“梯度覆盖”策略降低平台 pCTR 模型的不确定性，从而提升内容的长期有机曝光效果。

**💡 创新点**

创新点包括：① 将信息增益近似为可分解的梯度覆盖目标，理论上与 I‑optimal 设计相连；② 基于 Lagrange 双重性设计的两阶段自投标方案，实现预算动态节拍和每次竞价的即时最优；③ 针对缺失标签问题提出置信门控梯度估计及其零阶变体，实现在竞价时无标签梯度推断。

**🔧 技术方法**

核心技术包括：梯度覆盖（kernel‑based facility‑location）目标、单调子模性分析、第一/第二价拍卖的 Lagrange 双重优化、置信门控的梯度估计（基于交叉熵/梯度 L2 范数）以及 Zeroth‑Order（两点）梯度估计。

**📊 数据集**

实验数据集：合成二分类数据集、Criteo CTR 数据、Xiaohongshu（Shutiao）真实推广日志；使用 Logistic Regression / Deep CTR 模型作 pCTR 基线。

**📈 对比分析**

与基线比较包括：仅价值竞价、仅不确定性竞价、均匀竞价、pCTR 线性竞价、Oracle 不确定性选择。结果显示本文方法在 AUC 与 LogLoss 上均优于所有基线，预算消耗近乎精准，且在缺失梯度的 Zeroth‑Order 场景下仍保持竞争力。

**⚠️ 局限性**

局限性：① 梯度覆盖依赖验证样本的代表性，若验证集偏离真实分布则效果受限；② 置信门控阈值与 ū 的选择需要经验调参；③ 对极端低预算或极短生命周期的内容，预算节拍与子模性假设可能不完全满足；④ 当前方法针对一价拍卖，虽然可扩展到二价拍卖，但实现细节未充分验证。

---

## 246. An Empirical Evaluation of Modern MLOps Frameworks

**arXiv ID:** 2601.20415 | [PDF](https://arxiv.org/pdf/2601.20415v1)

**作者:** Jon Marcos-Mercadé `[一作]` (University of the Basque Country UPV/EHU), Mikel Egaña Aranguren `[通讯]` (University of the Basque Country UPV/EHU)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 MLflow、Metaflow、Apache Airflow 与 Kubeflow Pipelines 四种主流 MLOps 框架进行实证比较评估

**💡 创新点**

首次以 MNIST 与 IMDB 两套真实用例为基础，系统化地将易安装性、配置灵活性、互操作性、代码插装复杂度、结果可解释性和文档支持六项指标量化打分，形成加权总分以比较工具性能

**🔧 技术方法**

使用 Python+PyTorch 编写模型，并通过 GitHub Actions、Docker、Kubernetes、Airflow DAG、Metaflow FlowSpec 以及 MLflow Tracking/Model Registry 等技术实现完整的 CI/CD 流程

**📊 数据集**

使用 MNIST 图像分类数据集和 IMDB 情感分析文本数据集进行实验

**📈 对比分析**

通过对每项指标赋分并按权重计算加权总分，得到 MLflow（8.30）和 Airflow（8.25）在多维度表现优于 Metaflow（7.15）和 Kubeflow Pipelines（7.05），表明前两者在易用性、可配置性和互操作性方面更具优势

**⚠️ 局限性**

实验局限于仅评估六项基本功能，未覆盖实时服务、监控、报警和自动超参数优化；Kubeflow Pipelines 仅在本地 Minikube 上部署，缺乏云端大规模验证，且 Metaflow UI 安装与使用仍有技术门槛

---

## 247. Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents

**arXiv ID:** 2601.20412 | [PDF](https://arxiv.org/pdf/2601.20412v1)

**作者:** Qihao Wang `[一作]` (Institute of Information Engineering), Yuanmin Tang `[通讯]` (Institute of Information Engineering)

**通讯引用:** 131 | [OpenAlex ID](https://openalex.org/A5037051033)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于认知负荷理论的工具使用评估框架，并构建了可调认知负荷的基准 ToolLoad‑Bench；

**💡 创新点**

将任务复杂度拆分为内在负荷（用工具交互图量化）和外在负荷，建立指数成功概率模型来预测准确率，实现诊断式评估；

**🔧 技术方法**

利用认知负荷理论、工具交互图（TIG）、指数成功关系、Hosmer–Lemeshow 校准等技术；

**📊 数据集**

构建了 ToolLoad‑Bench（500 条实例），来源于 Berkeley Function Calling Leaderboard v3 的 200 条实例扩展而来；

**📈 对比分析**

对多款闭源与开源模型（如 GPT‑4o、Claude 3.7 Sonnet、Gemini 2.5 Pro、Qwen3‑235B、xLAM2‑32B 等）进行准确率与认知负荷曲线比较，结果显示 xLAM2‑32B 在高负荷下保持最高准确率，表明不同模型具有不同的基准性能与负荷敏感度；

**⚠️ 局限性**

基准领域覆盖有限；外在负荷测量依赖 LLM 评估，缺乏客观特征化指标，需进一步扩大泛化和完善负荷度量方法。

---

## 248. Fuzzy Private Set Union via Oblivious Key Homomorphic Encryption Retrieval

**arXiv ID:** 2601.20400 | [PDF](https://arxiv.org/pdf/2601.20400v1)

**作者:** Jean-Guillaume Dumas `[一作]` (University Grenoble Alpes), Luiza Soezima `[通讯]` (Aarhus University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种针对模糊私有集合并（FPSU）的协议，利用键值检索与可同态加密实现两方私密的模糊集合并。

**💡 创新点**

创新点在于引入可隐式同态加密检索（OKHER）以及通过球结构映射为图并利用图着色、a-排除、d-条带等结构化优化显著降低通信和计算复杂度。

**🔧 技术方法**

主要使用了可同态加密（线性/全同态）、可隐式键值存储（OKVS）和私有信息检索（PIR）等技术，并结合图着色算法DSATUR等。

**📊 数据集**

实验主要在理论与实验中使用生物识别（biometric）数据、随机生成的高维点集合等数据集进行验证。

**📈 对比分析**

与传统UPS/PSU协议相比，FPSU在轴不重叠、a-排除或d-条带等结构下通信量降至O(dmlog(δn))或O(d²mlog(δ²n))，计算量与输入规模呈线性关系。

**⚠️ 局限性**

局限在于需提前知道或上界色数，且对球重叠情况仍需较高成本；对于无结构或高密度集合，协议效率相对下降。

---

## 249. FedRD: Reducing Divergences for Generalized Federated Learning via Heterogeneity-aware Parameter Guidance

**arXiv ID:** 2601.20397 | [PDF](https://arxiv.org/pdf/2601.20397v1)

**作者:** Kaile Wang `[一作]` (Hong Kong Polytechnic University), Mingjin Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 2287 | [OpenAlex ID](https://openalex.org/A5009895460)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在异构联邦学习场景下，提出FedRD算法以提升对未见客户端的泛化性能。

**💡 创新点**

通过参数引导的自适应重加权去偏分类器和异构感知的全局聚合策略，解决了优化方向和性能偏差的两大分歧问题。

**🔧 技术方法**

采用自适应类别重加权、欧氏距离域差异度量、性能差距正则化、Sigmoid变换权重、ResNet特征提取等技术。

**📊 数据集**

在PACS、VLCS、OfficeHome、Mini-DomainNet等四个公开多域图像数据集上进行实验。

**📈 对比分析**

与FedAvg、FedProx、Scaffold、FedSR、FedBN、FedGA、FedIIR等基线在leave-one-domain-out评估中比较，FedRD在多数域上取得更高平均准确率，提升约3–5个百分点。

**⚠️ 局限性**

尚未充分评估通信成本、模型规模以及动态客户端加入时的适配性，且未验证在非图像任务上的迁移效果。

---

## 250. LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning

**arXiv ID:** 2601.20375 | [PDF](https://arxiv.org/pdf/2601.20375v1)

**作者:** Wei Huang `[一作]` (Ant Group), Tao Wei `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LLM‑AutoDP，一个利用大型语言模型（LLM）作为智能代理，自动生成和迭代优化数据处理（DP）策略的框架，旨在无需暴露原始敏感数据的前提下提升LLM微调性能。

**💡 创新点**

创新点包括：①将LLM作为决策代理，采用基于上下文学习和群组相对比较的反馈机制，快速收敛高质量DP策略；②提出三种加速技术——分布保持采样（DPS）、处理目标选择（PTS）和缓存重用机制（CRM），显著降低评估成本；③通过多轮交互实现策略生成与评估闭环，提升策略搜索效率。

**🔧 技术方法**

使用技术包括：LLM代理生成策略、基于提示工程的多轮交互、群组相对比较机制、二分类器筛选低质量样本、分布保持采样、处理目标选择、缓存重用机制、微调评估、GPT‑4/Baichuan‑M1‑14B‑Instruct 作为判定模型。

**📊 数据集**

实验数据集涵盖五个医学 QA 数据集（Chinese‑medical‑dialogue、cMedQA2、Medical‑O1‑Reasoning‑SFT、Huatuo‑26M‑Lite、Huatuo‑26M‑Lite‑100）以及一个法律数据集 DISC‑Law‑SFT。

**📈 对比分析**

对比方法包括无处理（No‑Process）、全处理（All‑Process）、随机搜索（RS）以及 SELA AutoML。实验显示，LLM‑AutoDP 在多数数据集上与未处理数据比较的赢率超过80%，与 SELA 的赢率约为65%；相较于单步贪心策略，LLM‑AutoDP 的赢率提升 60–85%；通过 DPS、PTS、CRM 组合可将策略搜索时间降低至原来的 5–10%，实现 76–95% 的加速。

**⚠️ 局限性**

局限性包括：①仍依赖大型 LLM 的算力与成本；②仅针对文本数据，未扩展至多模态；③在极大规模数据或极低质量数据时，二分类器与采样策略可能需要再调整；④虽然避免了直接暴露原始数据，但整体流程仍未实现严格的差分隐私保护；⑤需要手工设计提示模板，可能对不同领域存在迁移性挑战。

---

## 251. Dual-Modality IoT Framework for Integrated Access Control and Environmental Safety Monitoring with Real-Time Cloud Analytics

**arXiv ID:** 2601.20366 | [PDF](https://arxiv.org/pdf/2601.20366v1)

**作者:** Abdul Hasib `[一作]` (University of Frontier Technology), Anish Giri `[通讯]` (Bangalore University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个双模态 IoT 框架，将 RFID 门禁控制与多传感器环境安全监测（火焰检测、水流测量、人员识别）通过 ESP32 边缘处理和 Google Sheets 云同步实现统一管理。

**💡 创新点**

创新点包括：① 通过 Google Sheets 作为低成本、可自定义的云日志平台；② 多层级 RFID 认证与时效约束的组合；③ 采用自适应阈值的火焰检测算法；④ 通过分布式子系统与局部缓存实现网络断连下的容错；⑤ 端到端成本仅为 5,400 BDT（约 48 USD）.

**🔧 技术方法**

使用技术包括：ESP32‑WROOM‑32D、RC522 RFID、SG90舵机、IR 火焰传感器、YF‑S201 水流传感器、16×2 I2C LCD、FreeRTOS、JSON/HTTP、OAuth2、指数退避重传、卡尔曼滤波、Google Sheets API.

**📊 数据集**

实验数据集：10,000 次门禁尝试、在 1–30 L/min 范围内的水流测量、不同距离和光照条件下的火焰检测、45 天连续运行日志（≈ 99.8% 云上传成功率）。

**📈 对比分析**

与单功能系统、商用集成系统及相关研究对比，评价指标为认证准确率、火焰检测精度、云日志成功率、响应时间、安装复杂度与成本。该系统在认证准确率 99.2%、火焰检测 98.5%、云上传成功率 99.8% 的同时，成本仅为商用系统的 15%，整体得分最高。

**⚠️ 局限性**

局限性包括：RFID 读取范围仅 5–7 cm；火焰检测在强阳光下精度下降；缺乏实时双向云通信；对多站点扩展、数据库优化和安全认证（如证书机制）尚未完善。

---

## 252. Can Continuous-Time Diffusion Models Generate and Solve Globally Constrained Discrete Problems? A Study on Sudoku

**arXiv ID:** 2601.20363 | [PDF](https://arxiv.org/pdf/2601.20363v1)

**作者:** Mariia Drozdova `[一作]` (University of Geneva), Mariia Drozdova `[通讯]` (University of Geneva)

**通讯引用:** 1722 | [OpenAlex ID](https://openalex.org/A5058371961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文探讨了连续时间生成模型（Flow‑matching、Score‑matching）以及离散扩散模型（DDPM/DDIM）在极度稀疏、全局约束的离散结构——Sudoku——上的表现，并将其应用于无条件生成与基于约束的求解两种场景；

**💡 创新点**

创新点在于：①将标准连续时间生成框架投射到全局约束离散问题；②系统比较不同采样方式（ODE、SDE、DDPM/DDIM）及概率路径（线性vs余弦）对生成质量与求解效率的影响；③展示通过β(t)比例噪声调节的SDE可作为高效的随机求解器。

**🔧 技术方法**

使用了Transformer编码Sudoku格子、时间傅里叶嵌入、Flow‑matching/Score‑matching训练目标、Euler–Maruyama SDE/ODE采样、β(t)比例噪声、DDPM/DDIM离散化、蒙特卡洛批量采样以及熵与约束违例的评估指标。

**📊 数据集**

使用公开的Sudoku数据集（训练/测试划分），每个9×9盘子被展开为81个单元并用9维logits表示。

**📈 对比分析**

通过生成有效Sudoku比例、熵与约束违例的相关性、标准差等统计量进行比较。结果显示：在无条件生成中，DDPM/DDIM可达>80%有效率，连续时间SDE约12‑13%，ODE近0；在指导求解时，线性路径+β(t) SDE与DDPM在总模型前向次数上相当（≈1.8M），但DDPM在批次数上略优。

**⚠️ 局限性**

限制包括：未对约束注入机制做系统消融；仅使用单一数据集与模型架构；未与专业离散表征或符号推理方法对比；采样效率仍远低于传统Sudoku求解器；对概率路径一致性与约束求解效果之间的理论关系尚不明晰。

---

## 253. PEARL: Plan Exploration and Adaptive Reinforcement Learning for Multihop Tool Use

**arXiv ID:** 2601.20439 | [PDF](https://arxiv.org/pdf/2601.20439v1)

**作者:** Qihao Wang `[一作]` (Institute of Information Engineering), Yanbing Liu `[通讯]` (Institute of Information Engineering)

**通讯引用:** 4185 | [OpenAlex ID](https://openalex.org/A5037159025)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了PEARL框架，将大语言模型的工具使用拆分为离线工具探索与在线强化学习计划生成两阶段，提升多步工具调用的可靠性和规划质量。

**💡 创新点**

创新点在于：①离线探索构建工具使用手册，显著降低调用错误；②规划中心化奖励与Group Relative Policy Optimization（GRPO）相结合，直接对计划结构进行强化学习；③两阶段解耦架构实现执行稳健与规划灵活的协同。

**🔧 技术方法**

使用的大技术包括大语言模型、工具学习、离线工具试错、强化学习（GRPO）、规划奖励设计、执行器模型、对齐策略等。

**📊 数据集**

实验基于ToolHop和T‑Eval（转换为ToolHop风格）两个基准数据集。

**📈 对比分析**

通过与14款开源/闭源LLM及SFT基线模型在Success Rate（SR）和Invocation Error Rate（IER）两指标上对比，PEARL在ToolHop取得56.5% SR（最高），IER仅3.8%；在T‑Eval取得77% SR，IER 1.0%，均明显优于更大规模模型。

**⚠️ 局限性**

局限性包括：需要耗时的离线探索；奖励机制仅关注工具匹配，可能忽视最终任务结果；对新工具的迁移依赖已有工具清单；RL训练需要调参，收敛稳定性仍待进一步提升。

---

## 254. Graph-Structured Deep Learning Framework for Multi-task Contention Identification with High-dimensional Metrics

**arXiv ID:** 2601.20389 | [PDF](https://arxiv.org/pdf/2601.20389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 255. Self Voice Conversion as an Attack against Neural Audio Watermarking

**arXiv ID:** 2601.20432 | [PDF](https://arxiv.org/pdf/2601.20432v1)

**作者:** Yigitcan Özer `[一作]` (National Institute of Informatics), Junichi Yamagishi `[通讯]` (National Institute of Informatics)

**通讯引用:** 21579 | [OpenAlex ID](https://openalex.org/A5007639385)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并评估了自声码（Self Voice Conversion, Self‑VC）作为针对神经音频水印的通用攻击方法。

**💡 创新点**

创新点在于发现并证明即使保持说话人身份和语义内容，利用自声码的重构过程也能将水印信息几乎完全消除，挑战了现有水印方案的鲁棒性。

**🔧 技术方法**

使用的技术包括基于S3R/ HuBERT 的声码模型（kNN‑VC 与 RVC），以及音频处理与评估工具如 CREPE、HiFiGAN、EVC、ECAPA、Whisper‑L、UTMOS 等。

**📊 数据集**

实验数据集为 LibriTTS 的 test‑clean 子集。

**📈 对比分析**

对比方法包括经典 DSP 的 DCT 水印、AudioSeal、Timbre、WMCodec、VoiceMark 等五种主流水印方案，以及多种攻击方式（无攻击、传输通道失真、传统声码器、以及自声码攻击）。结果显示：在无攻击时大多数方案误码率低；在传统声码器攻击下误码率上升但仍可识别；而自声码攻击将所有方案的误码率推至约 0.5，接近随机猜测，说明水印失效。

**⚠️ 局限性**

局限性在于实验仅覆盖了两种自声码模型且未探讨更强的深度对抗训练或在其他声音域（如音乐、环境声）下的表现；同时未提供防御策略，未说明自声码攻击在实际部署中的可行性与成本。

---

## 256. Towards Quantum-Safe O-RAN -- Experimental Evaluation of ML-KEM-Based IPsec on the E2 Interface

**arXiv ID:** 2601.20378 | [PDF](https://arxiv.org/pdf/2601.20378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 257. Quartet of Diffusions: Structure-Aware Point Cloud Generation through Part and Symmetry Guidance

**arXiv ID:** 2601.20425 | [PDF](https://arxiv.org/pdf/2601.20425v1)

**作者:** Chenliang Zhou `[一作]` (University of Cambridge), Cengiz Oztireli `[通讯]` (University of Cambridge)

**通讯引用:** 2226 | [OpenAlex ID](https://openalex.org/A5046322671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

我们提出了一个名为Quartet的结构感知3D点云生成框架，通过四个协同扩散模型分别学习形状潜在、对称性、语义部件和装配器，实现高质量、可控且保证对称的点云生成。

**💡 创新点**

创新点在于同时显式建模部件组成与对称性，并用四个分离但协同的扩散模型实现对称保证与可解释的分解生成；此外引入稀疏变分自编码器与对称群分布学习，使生成过程既具备可控性又保持全局结构一致性。

**🔧 技术方法**

核心技术包括：多阶段扩散模型（四个）、稀疏变分自编码器（SVAE）及其潜在扩散、Transformer‑based 3D扩散用于部件生成、U‑Net+跨注意力用于装配器学习、对称性搜索与对称群分布建模、以及对称度评价指标SDI。

**📊 数据集**

使用ShapeNetPart数据集，主要实验类别为airplane、car和chair，并对每个类别的部件进行分割与归一化处理。

**📈 对比分析**

与多种基准模型（PointFlow、ShapeGF、DPF‑Net、SetVAE、DPC、PVD、LION、SPAGHETTI、DiT‑3D、SALAD、FrePolad等）在1‑NNA与SDI等指标上对比，Quartet在1‑NNA接近50%（与真实分布匹配度高）且SDI显著降低（对称性更好），达成state‑of‑the‑art。

**⚠️ 局限性**

局限性包括：目前仅支持无条件生成；对称性处理仅限于有限的反射/旋转组合，无法覆盖复杂连续对称；仅在三类物体验证，跨类别泛化尚待验证；以及模型训练和推理时间仍较高。

---

## 258. A High-Performance Fractal Encryption Framework and Modern Innovations for Secure Image Transmission

**arXiv ID:** 2601.20374 | [PDF](https://arxiv.org/pdf/2601.20374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 259. GuideAI: A Real-time Personalized Learning Solution with Adaptive Interventions

**arXiv ID:** 2601.20402 | [PDF](https://arxiv.org/pdf/2601.20402v1)

**作者:** Ananya Shukla `[一作]` (Plaksha University), Siddharth Siddharth `[通讯]` (Plaksha University)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5062321920)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 GuideAI——一种基于多模态生物传感（眼动、心率变异性、姿势、笔记行为）与大规模语言模型结合的实时自适应学习系统，支持文本、图像、音频、视频四种学习模式。

**💡 创新点**

创新点在于：①将实时认知-生理状态推断与 LLM 交互闭环结合，实现对内容难度、节奏、语调、身体姿势等多维度的即时自适应；②在学习系统中引入多模态干预策略（认知优化、注意转移、生理调节、语调适配）；③提供跨媒体的个性化学习路径，满足多样化学习需求。

**🔧 技术方法**

技术包括：大规模语言模型（ChatGPT/Claude/Llama）、Pupil Labs 眼动追踪、Max-Health-Band 心率传感、MediaPipe 姿势识别、笔记 OCR、Lab Streaming Layer 同步、HRV 与眼动特征提取、Z 分数归一化、语义状态抽象、LLM 生成适配干预、Web 平台实现多媒体交互。

**📊 数据集**

数据集：自行收集的 66 名学习者的生理与交互数据、学习内容（统计、机器学习、生物学等）、NASA‑TLX、知识测验题（回忆与问题解决）以及主观评价；实验使用 25 名参与者进行交叉对照，所有数据均为实验自采，未使用公开公共数据集。

**📈 对比分析**

比较方法：在 25 名参与者中采用 within‑subject 交叉实验，分别在非自适应 LLM 控制与 GuideAI 条件下完成四种学习模式。评估指标为 NASA‑TLX、知识测验成绩（回忆、问题解决）和 7 分制主观评估。结果显示 GuideAI 在认知负荷（各维度均显著降低）、学习表现（问题解决平均提升 16.5%、回忆平均提升 10.3%）和主观体验（专注、个性化、适配节奏显著提高）方面均优于对照。干预类型的消融实验进一步表明认知优化为最关键驱动。

**⚠️ 局限性**

限制：①样本规模小且实验场景受限，缺乏大规模、多样化样本与长期跟踪；②硬件依赖强（眼动、心率、姿势传感），部署成本高；③隐私与数据安全需要更完善的本地化处理；④LLM 仍存在幻觉与内容不准问题；⑤干预效果因个人差异而异，需进一步细化与个性化；⑥仅在 STEM 领域验证，其他学科的适用性尚待探索。

---

## 260. SFQA: A Comprehensive Perceptual Quality Assessment Dataset for Singing Face Generation

**arXiv ID:** 2601.20385 | [PDF](https://arxiv.org/pdf/2601.20385v1)

**作者:** Zhilin Gao `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 20610 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个专门针对歌唱面部生成（SFG）的质量评估数据集SFQA，并对现有12种生成方法进行主观与客观评估。

**💡 创新点**

创新点在于①首次系统收集并融合真人与AI生成的人脸图片与多语言音乐，形成多样化的SFG样本；②设计了多维度主观打分方案；③对比了多种主流视频质量评估与多模态评估方法，揭示了它们在SFG场景下的不足。

**🔧 技术方法**

使用了深度学习生成模型（VAE、扩散模型、Stable Diffusion等）产生视频；采用了主观MOS测试、以及14种客观评估方法（SimpleVQA、FAST-VQA、DOVER、SyncNet、LMM、AVQA、VALOR等）进行评测。

**📊 数据集**

数据集包括100张参考图像（50真人、50AI生成）和36段音乐（7种风格，含中英），共生成5,184段歌唱面部视频，用于构建SFQA。

**📈 对比分析**

在SFQA上对14种评估方法进行基准测试，VALOR在多模态对齐组中表现最佳，但总体性能仍偏低，说明现有方法不适合SFG质量评估。

**⚠️ 局限性**

局限性包括：①样本类别（人脸与音乐）仍有限；②生成方法受分辨率限制导致视觉质量偏低；③客观评估方法大多针对普通视频，缺乏针对歌唱面部的专门指标。

---

## 261. RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification

**arXiv ID:** 2601.20377 | [PDF](https://arxiv.org/pdf/2601.20377v1)

**作者:** Xinyan Chen `[一作]`, Jianfei Yang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了RF‑MatID数据集并对基于UWB‑mmWave雷达的材料识别进行系统基准测试。

**💡 创新点**

首次公开大规模、宽带（4–43.5 GHz）、几何多样（角度0°–10°、距离200–2000 mm）的RF材料数据集，并提供频域和时域两种原始表示，评估法规频段可行性。

**🔧 技术方法**

使用UWB‑mmWave雷达硬件、频域到时域逆FFT转换、复杂数白化、深度学习模型（MLP、ResNet‑50、Bi‑LSTM、Transformer、TimesNet、DINOv3、ConvNeXt、LSTM‑ResNet等）以及自研基线模型。

**📊 数据集**

RF‑MatID数据集（16细粒度类别，5个超类，142 k样本，71 k频域+71 k时域，角度/距离多样）。

**📈 对比分析**

通过7种数据划分（随机、跨距离、跨角度）和5种频段协议，对10+模型进行精细/超类/子类分类实验，平均准确率超96%，但在跨距离/角度偏移时下降显著；同时发现毫米波在距离变化时更鲁棒，厘米波在角度变化时更稳健。

**⚠️ 局限性**

数据集仅包含平板单层材料，缺乏多层/厚度、复杂环境（多径、遮挡）以及更细粒度的物理约束，频率采样相对稀疏，未来需要扩充材料多样性、环境复杂度和采样分辨率。

---

## 262. RAW-Flow: Advancing RGB-to-RAW Image Reconstruction with Deterministic Latent Flow Matching

**arXiv ID:** 2601.20364 | [PDF](https://arxiv.org/pdf/2601.20364v1)

**作者:** Zhen Liu `[一作]` (University of Electronic Science and Technology of China), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 6745 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于流匹配的RGB到RAW重建框架RAW-Flow。

**💡 创新点**

创新点在于将RGB‑RAW映射建模为确定性潜在传输，并引入跨尺度上下文引导与双域潜在自编码器。

**🔧 技术方法**

使用的技术包括确定性潜在流匹配（flow matching）、双域潜在自编码器、跨尺度上下文引导以及感知损失和对抗式训练。

**📊 数据集**

在FiveK-Nikon、FiveK-Canon和PASCALRAW三个公开RAW数据集上进行训练与评估。

**📈 对比分析**

与回归模型和扩散模型对比，RAW-Flow在PSNR/SSIM上均居前列，尤其在RAW域上高出约2–3dB。

**⚠️ 局限性**

局限性包括仅在固定相机模型下训练，缺乏跨相机泛化能力，且推理仍需多步ODE积分，计算开销相对较大。

---

## 263. Switchcodec: Adaptive residual-expert sparse quantization for high-fidelity neural audio coding

**arXiv ID:** 2601.20362 | [PDF](https://arxiv.org/pdf/2601.20362v1)

**作者:** Xiangbo Wang `[一作]` (Hangzhou Dianzi University), Fei Wen `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于残差专家向量量化（REVQ）的神经音频编解码器 SwitchCodec。

**💡 创新点**

创新点是将共享基量化器与稀疏激活的专家量化器分离，按固定索引顺序应用，实现码率与码本容量解耦，并支持轻量级可变码率。

**🔧 技术方法**

使用了卷积变分自编码器、残差专家向量量化、门控路由网络、直通估计器以及可变码率 top‑k 选择策略。

**📊 数据集**

训练数据来自 VCTK、LibriTTS、Free Music Archive (FMA) 与 Common Voice，采样 44.1kHz 单声道。

**📈 对比分析**

与 EnCodec、DAC 等基线在 2.67 kbps、5.33 kbps 下做客观（ViSQOL、PESQ、Mel、STFT）和主观 MUSHRA 对比，SwitchCodec 在所有指标上均优于基线，MUSHRA 超过 91，近乎透明。

**⚠️ 局限性**

局限包括：仍需显式路由掩码传输（虽很小），模型对极低码率（<0.9 kbps）或极高动态范围内容的处理尚未深入验证，且实验主要在单声道 44.1kHz 上。

---

## 264. SpeechMapper: Speech-to-text Embedding Projector for LLMs

**arXiv ID:** 2601.20417 | [PDF](https://arxiv.org/pdf/2601.20417v1)

**作者:** Biswesh Mohapatra `[一作]` (Inria), Ioan Calapodescu `[通讯]` (NAVER LABS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SpeechMapper，一种两阶段训练的投影器，将 SFM 嵌入映射到 LLM 嵌入空间，从而实现语音到 LLM 的零样本或任务特定能力。

**💡 创新点**

创新点在于避免使用显式对齐器与昂贵的 LLM 前向计算，只用 LLM 嵌入层预训练投影器，并通过轻量级的 MSE + 余弦损失实现语音-文本嵌入对齐，显著降低资源消耗并提升泛化。

**🔧 技术方法**

使用的技术包括冻结的 SFM（seamless‑m4t‑v2‑large）、基于 Transformer 的投影器、MSE 与余弦嵌入损失、交叉熵与 MSE 的混合自适应微调，以及 AdamW/余弦学习率调度。

**📊 数据集**

主要数据集为 960 小时 LibriSpeech（预训练）、EuroParlST 与 CoVoST2（ST 评测）以及 SpokenSQUAD 与 LibriSQA（SQA 评测），并与 IWSLT25 竞赛基线共享相同的 SFM 与 LLM。

**📈 对比分析**

与 IWSLT25 竞赛最佳系统（BEST‑IWSLT25‑IF）相比，SpeechMapper 在 ST 与 SQA 上实现了相近或更高的 COMET/准确率，仅需 1.5 小时 1K 步 A100 训练，显著降低了算力和数据需求。

**⚠️ 局限性**

局限性包括：仍需与目标 LLM 的嵌入维度保持一致；对极端口音或噪声环境的鲁棒性未作充分评估；以及在更大规模多模态数据上的验证尚未展开。

---

## 265. Schadenfreude in the Digital Public Sphere: A cross-national and decade-long analysis of Facebook news engagement

**arXiv ID:** 2601.20413 | [PDF](https://arxiv.org/pdf/2601.20413v1)

**作者:** Nouar Aldahoul `[一作]` (New York University), Yasir Zaki `[通讯]` (New York University)

**通讯引用:** 1517 | [OpenAlex ID](https://openalex.org/A5018129441)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用人工标注和大语言模型，对2015-2024年间美国、英国和印度三国主要新闻页面的Facebook帖子与评论进行情感分析，系统量化并比较了公众在他人不幸事件中的 Schadenfreude（幸灾乐祸）表达。

**💡 创新点**

首次实现了跨国、跨意识形态、跨时间尺度的 Schadenfreude 长期趋势与情境分析，揭示了其与政治权力、议题类型和文化背景的相互作用，填补了此前仅有零散案例研究的空白。

**🔧 技术方法**

采用 GPT‑4o Mini 进行帖子/评论的二分类（是否描述不幸、情感类别），并在 GPT‑5.2 上进行主题归类；结合手工标注的高质量数据集训练和微调，形成 end‑to‑end 的文本情感与主题判别流水线。

**📊 数据集**

收集了 9 家主要新闻机构（左倾、中立、右倾）在三国的公开 Facebook 页面数据，约 1000 条帖子/月 × 10 年，共计 约 100 万条评论和 10 万条帖子，涵盖多语言（英、印）与多文化背景。

**📈 对比分析**

通过对比反应图标（Sad/Angry/Haha 等）与文本评论中的 Schadenfreude 率，结合多维度分组（国家、意识形态、议题类别）以及 OLS 回归分析，验证了模型在情感分类上的准确率（F1≈88%）和主题分类的可靠性；结果显示右倾受众、政治与道德议题以及印度语境中 Schadenfreude 最高，时间序列呈现明显的政治权力相关波动。

**⚠️ 局限性**

主要限制包括：1）平台删帖与评论删除可能导致样本偏差；2）语言模型对讽刺、隐含情绪的捕捉仍有限，跨语言误判风险；3）数据仅来自公开页面，可能不代表普通新闻消费者；4）观察性设计无法确立因果关系；5）不同国家的监管、语言习惯和平台渗透率可能影响结果可比性。

---

## 266. On Efficient Polyphase Network Implementation Using Successive Vector Approximation

**arXiv ID:** 2601.20411 | [PDF](https://arxiv.org/pdf/2601.20411v1)

**作者:** Luiz F. da S. Coelho `[一作]` (CEDRIC-CNAM), Paulo S. R. Diniz `[通讯]` (COPPE/UFRJ)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对滤波器组多载波(FBMC)系统的多相网络(Multiplexed Polyphase Network, PPN)实现，本文提出利用两种贪心算法（SDL与MPGBP）将原始浮点滤波器系数近似为和符号二进制幂（SOPOT），从而实现无乘法器的高效实现。

**💡 创新点**

创新点在于：①提出了Signed Digit Loading（SDL）这类向量级的贪心分配算法；②将匹配追踪（Matching Pursuits）与SOPOT结合，形成MPGBP算法；③通过在向量层面分配SOPOT，可显著降低均方误差（MSE）与残留干扰，提升OOB和BER性能，优于传统的CSD（Canonical Signed Digit）逐元素逼近。

**🔧 技术方法**

核心技术包括：
- SOPOT数值表示与位平面分配；
- 贪心算法（SDL）在向量上逐步分配最显著残差的SOPOT；
- 匹配追踪与通用位平面（MPGBP）在一次迭代中同时选取多重残差并分配SOPOT；
- 与CSD算法的比较与评估。

**📊 数据集**

数据集与实验设置：
- 采用PHYDYAS原型滤波器；
- 128个子载波，64个信息块；
- 传输信号使用4-QAM/64-QAM调制；
- 在AWGN信道下进行Monte‑Carlo仿真（10 000样本）。

**📈 对比分析**

比较方法与性能：
- MSE、干扰MSE、BER、OOB功率谱密度（PSD）与原始系统、4‑bit CSD、SDL（1.8 SPT/coeff）和MPGBP进行对比；
- 结果显示：SDL/MPGBP在相同SPT数下平均可获得约20 dB的MSE提升；
- OOB功率谱与干扰MSE也比CSD低约20 dB；
- 在4‑QAM下BER与原始系统相近，SDL略优；在64‑QAM下，SDL保持原始系统的BER，而CSD性能明显下降。

**⚠️ 局限性**

局限性：
- 仅在仿真环境下验证，缺乏实际硬件实现与功耗测量；
- 只针对PHYDYAS滤波器，未验证对其他滤波器或更复杂多载波结构的适用性；
- 贪心算法的计算复杂度未做完整理论分析，实际实现时对时钟与资源需求需进一步评估。

---

## 267. AWGformer: Adaptive Wavelet-Guided Transformer for Multi-Resolution Time Series Forecasting

**arXiv ID:** 2601.20409 | [PDF](https://arxiv.org/pdf/2601.20409v1)

**作者:** Wei Li `[一作]` (Shanghai University), Wei Li `[通讯]` (Shanghai University)

**通讯引用:** 60404 | [OpenAlex ID](https://openalex.org/A5109869303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 AWGformer 架构，利用自适应小波分解与跨尺度注意力实现多变时序预测；

**💡 创新点**

创新点在于可学习的自适应小波基与分解层级、跨尺度特征融合、频率感知多头注意力以及层次化预测网络；

**🔧 技术方法**

使用了 Transformer 结构、可学习小波变换、耦合矩阵、频率选择注意力、逆小波重构等技术；

**📊 数据集**

在公开基准数据集 ETT（ETTh1/ETTh2）、Traffic、Electricity 等上进行实验；

**📈 对比分析**

与 Autoformer、FEDformer、TimesNet、PatchTST、iTransformer、DLinear 等方法对比，AWGformer 在 MSE/MAE 上均优越，尤其在长预测时段（如 720 步）表现显著提升；

**⚠️ 局限性**

局限性包括：小波基仍受限于紧支撑形式、注意力复杂度为 O(T²)，难以处理极长序列；训练假设规则采样，缺乏对不规则时间戳和概率预测的支持。

---

## 268. ScatterFusion: A Hierarchical Scattering Transform Framework for Enhanced Time Series Forecasting

**arXiv ID:** 2601.20401 | [PDF](https://arxiv.org/pdf/2601.20401v1)

**作者:** Wei Li `[一作]` (Shanghai University), Wei Li `[通讯]` (Shanghai University)

**通讯引用:** 60404 | [OpenAlex ID](https://openalex.org/A5109869303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ScatterFusion 框架，将散射变换与层级注意机制结合，用于多尺度时间序列预测。

**💡 创新点**

创新点在于：1）多尺度散射变换模块使用可学习的小波滤波器；2）尺度自适应特征增强；3）跨分辨率时序注意力；4）趋势‑季节‑残差分解引导的损失函数，三者协同提升预测鲁棒性和准确性。

**🔧 技术方法**

使用技术包括：变形不变散射变换、可学习小波滤波器、尺度自适应注意力、跨分辨率注意力机制、趋势‑季节‑残差分解损失、Transformer 架构以及 AdamW + cosine annealing 训练策略。

**📊 数据集**

实验数据集：ETT、Traffic、ECL、Weather 四大公开基准（总共七个数据集提到，但核心评测集中在上述四个）。

**📈 对比分析**

与 Informer、Autoformer、FEDformer、PatchTST、TimesNet、DLinear、WFTNet 等先进模型比较，ScatterFusion 在 MSE/MAE 上普遍领先，尤其在长预测期（720 时步）表现显著优于 PatchTST 等基线。

**⚠️ 局限性**

局限性：对极端事件的鲁棒性仍有限；高阶散射系数计算开销较大；未针对少样本迁移学习进行深入探索。

---

## 269. Comprehension vs. Adoption: Evaluating a Language Workbench Through a Family of Experiments

**arXiv ID:** 2601.20394 | [PDF](https://arxiv.org/pdf/2601.20394v1)

**作者:** Giovanna Broccia `[一作]` (National Research Council Institute for Information Science and Technology), Alessio Ferrari `[通讯]` (University College Dublin)

**通讯引用:** 2991 | [OpenAlex ID](https://openalex.org/A5041720518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过三轮实验评估Neverlang语言工作台的可理解性和用户接受度。

**💡 创新点**

首次系统化使用改造后的MEM模型将技术可理解性与用户接受度关联，并进行家族实验与元分析。

**🔧 技术方法**

采用定制MEM评估模型、Wilcoxon检验、Spearman相关、Holm-Bonferroni校正及随机效应元分析。

**📊 数据集**

使用三组学生/博士样本（共50人）完成的可理解性测试与接受度问卷。

**📈 对比分析**

对比实验结果和元分析表明：语法易懂、可用性被认为高，但易用性低；接受度与可理解性无显著相关，整体效能偏低。

**⚠️ 局限性**

受限于样本规模、主要为学术新人、缺乏长期实践与编码任务，且未与其他工作台做对比。

---

## 270. Eliminating Hallucination in Diffusion-Augmented Interactive Text-to-Image Retrieval

**arXiv ID:** 2601.20391 | [PDF](https://arxiv.org/pdf/2601.20391v1)

**作者:** Zhuocheng Zhang `[一作]` (Hunan University), Zijun Long `[通讯]` (Hunan University)

**通讯引用:** 96 | [OpenAlex ID](https://openalex.org/A5102631471)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种Diffusion-aware Multi-view Contrastive Learning (DMCL) 框架，用于改进扩散增强的交互式文本-图像检索，通过多视图对比学习抑制生成视图中的幻觉，提升检索效果。

**💡 创新点**

创新点在于：①将文本、扩散生成图像和融合视图视为多视图，采用多视图查询-目标一致性与文本-扩散一致性两种对比目标，实现语义过滤；②通过硬负样本挖掘进一步强化对抗幻觉的鲁棒性；③系统性地将幻觉视为可学习的零空间，提升检索对话鲁棒性。

**🔧 技术方法**

技术手段包括：多视图对比学习（InfoNCE + 对称化），文本-扩散一致性损失，硬负样本挖掘，融合编码器（文本编码、图像编码+投影头），Stable Diffusion 3.5 生成查询图像，BEiT‑3 作为多模态检索编码器。

**📊 数据集**

使用自构建的 DA‑VisDial（在 VisDial 上加入扩散生成的对话图像）作为训练集；评测采用五个基准：VisDial、ChatGPT_BLIP2、HUMAN_BLIP2、Flan‑Alpaca‑XXL_BLIP2、PlugIR_dataset，用于在分布内外进行验证。

**📈 对比分析**

与三类基线（零样本 BEiT‑3、ChatIR、PlugIR‑CR）以及其扩散增强版本进行对比。DMCL 在 Hits@10 上在 0 轮到 10 轮均优于基线，分布内提升约 4.9%–7.4%，分布外（零样本）提升 6%+，表明其具有良好的泛化与鲁棒性。

**⚠️ 局限性**

目前仅采用简单的加法融合来融合多视图，未进一步探索更高效或自适应的融合方式；同时模型对极端幻觉场景的过滤效果仍有提升空间。

---

## 271. A Program Logic for Abstract (Hyper)Properties

**arXiv ID:** 2601.20370 | [PDF](https://arxiv.org/pdf/2601.20370v1)

**作者:** Paolo Baldan `[一作]` (University of Padua), Diletta Rigo `[通讯]` (University of Padua)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种统一的Hoare式程序逻辑(Abstract Program Property Logic)，能够同时涵盖标准Hoare逻辑、错误逻辑与超属性逻辑，并支持抽象解释的形式化；

**💡 创新点**

通过在完整幺半群和格的抽象框架下引入可变的非幺半群运算和基底，首次实现对抽象域和超属性的系统统一与抽象化；

**🔧 技术方法**

利用完整幺半群、量子格、抽象解释的最佳正确逼近（BCA）以及推理规则的扩展（join/meet、iter、rec、inv）等形式方法；

**📊 数据集**

无实验数据集，论文主要为理论构造与证明；

**📈 对比分析**

无性能评估或实验比较，主要讨论理论可行性与与已知逻辑的对应关系；

**⚠️ 局限性**

受限于对非加性幺半群的抽象化精度、缺乏实证验证，以及对迭代与超属性固定点语义的进一步研究待完成。

---

## 272. LIFT: Byzantine Resilient Hub-Sampling

**arXiv ID:** 2601.20368 | [PDF](https://arxiv.org/pdf/2601.20368v1)

**作者:** Mohamed Amine Legheraba `[一作]`, Sébastien Tixeuil `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文对Elevator协议在拜占庭攻击下的鲁棒性进行了首次系统评估，并提出了一种名为LIFT的改进协议，利用确定性PRNG对节点中心化进行再分配以抵御攻击。

**💡 创新点**

创新点在于：①揭示了Elevator协议在约2%拜占庭参与率处出现的临界阈值；②设计了仅需一次性激活的确定性Hub再分配机制，利用节点不可变随机ID生成共享种子，防止协同误导；③在保持原有性能（收敛周期≈4轮）的同时显著降低了拜占庭节点占主节点比例。

**🔧 技术方法**

使用了PeerSim仿真框架实现Elevator和LIFT协议，并在多种拜占庭攻击模式（单个、非协同、多协同）下评估其行为；采用Java默认线性同余PRNG生成共享随机种子。

**📊 数据集**

实验数据集为1000个节点的随机k‑out网络（k=20），多次（100次）独立仿真得到平均统计结果。

**📈 对比分析**

通过比较主节点占比、收敛周期和节点失效率等指标，结果表明：在5%拜占庭参与率下，LIFT将主节点中的拜占庭比例降至≈3.4%；在10%和15%参与率下虽仍能初步抑制，但拜占庭节点仍能逐步恢复；与原Elevator相比，LIFT在低至中等攻击强度下显著提升了鲁棒性。

**⚠️ 局限性**

局限性包括：①需要预先确定一次性激活周期（默认为第100轮），若网络收敛时间不匹配会导致效果下降；②在高拜占庭比例（>10%）时，LIFT仍难以彻底阻止节点重新夺取hub；③依赖节点ID不可变随机分配，对抗攻击者能够篡改ID的场景尚未覆盖。

---

## 273. Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models

**arXiv ID:** 2601.20367 | [PDF](https://arxiv.org/pdf/2601.20367v1)

**作者:** Qing Lyu `[一作]` (University of California), Alexandre Bayen `[通讯]` (University of California)

**通讯引用:** 14593 | [OpenAlex ID](https://openalex.org/A5021116704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种无监督的多智能体轨迹预测异常检测框架，利用Transformer模型学习正常驾驶模式并通过预测残差识别危险情景。

**💡 创新点**

创新点在于结合多智能体Transformer和残差聚合、孤立森林评分，并引入双重评估方案（稳定性与物理对齐）来验证异常与实际安全风险的一致性。

**🔧 技术方法**

核心技术包括序列到序列Transformer轨迹预测、残差加权聚合（max、q95、mean、top‑k）、Isolation Forest异常分数、Kendall相关、Jaccard重叠、Spearman相关与K‑Means聚类。

**📊 数据集**

实验数据集为美国加州NGAIM US101高频车辆轨迹数据，构造了约2.9万个包含7辆车的局部场景。

**📈 对比分析**

与基于TTC阈值和物理特征的Isolation Forest基线相比，本文方法识别了2,832个异常场景，其中388个为基线未捕获的细粒度风险；max聚合得到的异常评分与多种安全代理指标（TTC、距离、速度/加速度变化）相关性最高。

**⚠️ 局限性**

局限性包括仅在离线NGAIM数据上验证，缺乏在线实时部署评估；残差仅反映预测误差，未直接衡量真实碰撞风险；模型对不同交通场景的泛化能力尚待进一步验证。

---

## 274. Youtu-Parsing: Perception, Structuring and Recognition via High-Parallelism Decoding

**arXiv ID:** 2601.20430 | [PDF](https://arxiv.org/pdf/2601.20430v1)

**作者:** Kun Yin `[一作]`, Shuangyin Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Youtu‑Parsing，一种基于Vision Transformer和Prompt‑Guided 2.5B参数LLM的高效文档解析框架，支持文本、公式、表格、图表、印章等多种元素的端到端识别；

**💡 创新点**

核心创新包括双轨并行解码（Token Parallelism与Query Parallelism）实现10–20×的推理加速且保持与自回归解码完全一致的输出、解耦式共享视觉特征+区域提示解码、混合掩码训练与层次关系标记等技术；

**🔧 技术方法**

采用NaViT动态分辨率ViT、Youtu‑LLM‑2B、MLP投影、Flash Attention、混合掩码训练、GRPO强化学习、合成数据流水线、关系标记与层次结构解析等；

**📊 数据集**

使用了OmniDocBench v1.5、olmOCR‑bench、CC‑OCR、OCRBench v2、In‑house表格/文本/公式/图表/印章数据集，以及自制的1000张图表和1000张印章样本；

**📈 对比分析**

与通用VLM（如GPT‑4o、InternVL3、Qwen2.5‑VL‑72B、Gemini‑2.5‑Pro）、专业解析模型（PaddleOCR‑VL、MinerU2.5、dots.ocr等）以及基线流水线工具对比，Youtu‑Parsing在OmniDocBench整体分数93.22、公式CDM 93.19、表格TEDS 91.15/95.43、olmOCR‑bench整体80.5%、文本OCR 98.94%、表格TEDS 88.24、图表RMS‑F1 0.6124、印章编辑距离80.13等指标均取得SOTA，且单页推理时间仅1.75s；

**⚠️ 局限性**

主要局限在于模型参数量大（2.5B），导致显存占用高、部署受限于GPU，训练成本和推理时的显存需求较高；此外对极长序列或非常稀有字体/语言的适应仍需进一步验证。

---

## 275. Mix2Morph: Learning Sound Morphing from Noisy Mixes

**arXiv ID:** 2601.20426 | [PDF](https://arxiv.org/pdf/2601.20426v1)

**作者:** Annie Chu `[一作]`, Prem Seetharaman `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Mix2Morph模型，利用噪声混合的替代数据对文本到音频扩散模型进行微调，实现高质量的声音注入（sound infusion）

**💡 创新点**

创新点在于：①不需要专门的形态数据集，而是用加噪声的混合音频作为高时间步的训练目标；②通过时间域与频域的增强实现“声音注入”这一实用的非对称混合；③提出了新的评估指标如LCS和方向性得分

**🔧 技术方法**

技术包括文本到音频的潜在扩散模型、VAE压缩/解压、时间域RMS对齐、频域平均频谱对齐、随机增强模式、在高时间步训练、以及多维评估指标

**📊 数据集**

使用了大规模授权SFX数据集与公开的通用音频语料库进行预训练，随后用合成的噪声混合音频进行微调，评估时挑选了50对不同类别的概念音频共100个注入提示

**📈 对比分析**

与五种基线（基线模型、简单混音、粒度重采样、MorphFader、SoundMorpher）比较，Mix2Morph在对应度、介于度、方向性、LCS和FAD上均优于所有基线；主观听力实验显示其最高的“形态率”与MOS得分

**⚠️ 局限性**

局限包括：仅在高时间步训练导致对细节恢复依赖原始预训练；对低频、低质量音频仍可能出现微妙混合不自然；缺乏音频到音频的交互式控制

---

## 276. Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs

**arXiv ID:** 2601.20420 | [PDF](https://arxiv.org/pdf/2601.20420v1)

**作者:** Yuhang Liu `[一作]` (Australian Institute for Machine Learning), Javen Qinfeng Shi `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于理论的概念提取框架ConCA，用于从LLM内部表示中无监督地恢复潜在概念的后验对数概率。

**💡 创新点**

创新点在于通过潜在变量模型证明LLM表示可近似为概念后验对数的线性混合，进而设计稀疏ConCA并在指数空间上施加稀疏正则，解决了传统SAE在理论基础和可解释性上的不足。

**🔧 技术方法**

主要技术包括潜在变量生成模型、理论推导的线性混合关系、稀疏正则化（L1）与不同归一化（LayerNorm、GroupNorm、BatchNorm）结合的自编码器架构。

**📊 数据集**

使用Pile数据集子集（200M token）训练ConCA，评估使用Pythia、Gemma3、Qwen3等多规模多架构模型的内部表示；对照SAE变体（Top‑k、Batch‑Top‑k、P‑anneal）。

**📈 对比分析**

与SAE相比，ConCA在Pearson相关性和MSE上均更优，尤其在不同模型规模和架构下表现更稳定；在少样本线性探测和OOD任务上也实现了更高或相近的AUC。

**⚠️ 局限性**

局限性包括稀疏正则仍可能导致过拟合，稀疏度设定依赖经验；对真实语言中概念边界的判定仍需更精细的对抗样本；理论证明依赖若干假设，在实践中可能被违反。

---

## 277. Meeting SLOs, Slashing Hours: Automated Enterprise LLM Optimization with OptiKIT

**arXiv ID:** 2601.20408 | [PDF](https://arxiv.org/pdf/2601.20408v1)

**作者:** Nicholas Santavas `[一作]` (eBay), Shahram Khadivi `[通讯]` (eBay)

**通讯引用:** 1063 | [OpenAlex ID](https://openalex.org/A5047428600)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 OptiKIT，一个自动化的分布式 LLM 优化框架，覆盖模型压缩、统计评估、推理基准与部署调优全过程。

**💡 创新点**

关键创新点在于：全流程自动化与标准化、基于配方的压缩引擎、SLO 驱动的稳健基准算法、贝叶斯调优器以及与企业基础设施无缝集成的统一接口。

**🔧 技术方法**

主要技术包括：Ray 分布式执行框架、vLLM 与 OpenAI 接口、TensorRT-LLM、GPTQ/SmoothQuant/RTN 量化方法、TPE 贝叶斯搜索调优、以及针对 GPU 资源的动态分配与自动清理。

**📊 数据集**

使用的评估数据集有公开基准 GSM8K、IFEval、Do‑Not‑Answer 以及内部电商业务相关的私有基准；模型涵盖 Qwen 2.5 7B、Mistral 24B 与 Llama 3.3 70B 三大系列。

**📈 对比分析**

与手工优化比较，OptiKIT 在 GPU 吞吐量上提升 1.3–2.8 倍（以 70B Llama 为例），工程时间从 80–100 小时降至 15–25 小时，同时保持 99% 以上的准确率恢复。

**⚠️ 局限性**

局限性包括：对极大规模模型（>70B）在多机多卡的可扩展性尚未完全验证、在自定义微调后量化性能未知，以及当前同步阻塞的调度架构导致 GPU 利用率有待进一步提升。

---

## 278. HINT: Hierarchical Interaction Modeling for Autoregressive Multi-Human Motion Generation

**arXiv ID:** 2601.20383 | [PDF](https://arxiv.org/pdf/2601.20383v1)

**作者:** Mengge Liu `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**通讯引用:** 10580 | [OpenAlex ID](https://openalex.org/A5024401174)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HINT，一种基于扩散的自回归多人体运动生成框架；

**💡 创新点**

创新点在于（1）使用“canonicalized latent space”将个体运动与交互分离，支持可变人数；（2）滑动窗口自回归策略结合局部与全局层次条件，实现长序列一致性与细粒度交互建模；

**🔧 技术方法**

技术主要包括：VAE预编码、局部/全局条件的跨注意力与AdaLN、滑动窗口扩散生成、相对旋转平移编码；

**📊 数据集**

主要使用InterHuman与InterX两个基于SMPL/H、SMPL-X的人体交互动作数据集；

**📈 对比分析**

与多种离线与在线基线（T2M、MDM、InterGen、InterMask、DART等）比较，HINT在FID、R@Top3、MM Dist等指标上取得最优或近优表现，尤其在多人体交互的FID大幅提升；

**⚠️ 局限性**

局限性包括：仍缺乏对全局文本命令的全局优化，导致长文本对齐略弱；扩展到更多人体时仅做简单拼接，可能需要进一步微调；未考虑对象或环境交互。

---

## 279. STORM: Slot-based Task-aware Object-centric Representation for robotic Manipulation

**arXiv ID:** 2601.20381 | [PDF](https://arxiv.org/pdf/2601.20381v1)

**作者:** Alexandre Chapin `[一作]` (Ecole Centrale de Lyon), Liming Chen `[通讯]` (Ecole Centrale de Lyon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 STORM，轻量级的多阶段任务感知基于槽的对象中心表示，改进机器人操纵的鲁棒性和泛化。

**💡 创新点**

结合视觉-语义预训练与任务对齐的两阶段策略，保证槽的语义一致性并对控制目标对齐。

**🔧 技术方法**

冻结的 DINOv2 视觉基座 + Slot‑Attention、CLIP 文本编码、对比学习、Transformer 解码器 + GMM 动作头。

**📊 数据集**

VG‑COCO、PASCAL VOC、COCO 用于对象发现预训练，MetaWorld、LIBERO 用于操纵任务评估。

**📈 对比分析**

与冻结 DINOv2、微调 DINOv2、无监督与弱监督槽方法对比，STORM 在 ID 与新视觉干扰下的成功率分别提升 1% / 12.7% 与 10.7% / 19.0%，显著优于基线。

**⚠️ 局限性**

仅在仿真中验证，槽数固定、简单名词抽取不足以应对复杂关系语言，未在真实机器人上测试。

---

## 280. RepSFNet : A Single Fusion Network with Structural Reparameterization for Crowd Counting

**arXiv ID:** 2601.20369 | [PDF](https://arxiv.org/pdf/2601.20369v1)

**作者:** Mas Nurul Achmadiah `[一作]` (National Formosa University), Jun-Wei Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 6162 | [OpenAlex ID](https://openalex.org/A5007240491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化的 RepSFNet 框架，用于实时拥挤计数。

**💡 创新点**

创新点在于结合 RepLK‑ViT 背景、ASPP+CAN 的特征融合以及 Concatenate Fusion，避免注意力机制和多分支设计，实现了低参数、高效推理。

**🔧 技术方法**

核心技术包括大核卷积的结构重参数化（RepLK）、Atrous Spatial Pyramid Pooling（ASPP）、Context‑Aware Network（CAN）、拼接融合、MSE+Optimal Transport 损失。

**📊 数据集**

在上海Tech、NWPU、UCF‑QNRF 三个公开数据集上进行评测。

**📈 对比分析**

与 P2PNet、M‑SFANet、STEERER 等现有方法对比，RepSFNet 在多数据集上保持竞争性精度，同时推理延迟降低多达 34%。

**⚠️ 局限性**

局限在于缺乏显式注意力导致在极端拥挤场景（如 UCF‑QNRF）略逊一筹，且深度下采样和固定膨胀率可能损失稀疏场景细节。

---

## 281. OmegaUse: Building a General-Purpose GUI Agent for Autonomous Task Execution

**arXiv ID:** 2601.20380 | [PDF](https://arxiv.org/pdf/2601.20380v1)

**作者:** Le Zhang `[一作]` (Baidu Frontier Research Department), Haifeng Wang `[通讯]` (Baidu Frontier Research Department)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了OmegaUse，一种基于Mixture-of-Experts的通用GUI代理模型，支持手机和电脑端的自主任务执行；

**💡 创新点**

创新点在于：①结合自动化合成与底层探索的分层数据构建管道，②采用SFT+GRPO的分离训练策略与专用奖励机制，③引入跨终端统一动作空间与OS‑Nav离线基准；

**🔧 技术方法**

使用的技术包括MoE视觉语言模型、监督微调、Group Relative Policy Optimization、专用定位与顺序奖励；

**📊 数据集**

数据集来源于六大公开GUI定位集（Aguvis、UI RefExp等）、自动合成轨迹、专家演示以及新发布的OS‑Nav（ChiM‑Nav、Ubu‑Nav）等；

**📈 对比分析**

在ScreenSpot‑V2、AndroidControl、AndroidWorld、ChiM‑Nav、Ubu‑Nav等基准上，OmegaUse均实现或逼近SOTA，尤其在ScreenSpot‑V2上达96.3%，AndroidControl上79.1%的步骤成功率；

**⚠️ 局限性**

局限在于：仍依赖大量人工审核的合成数据、在极端高分辨率或专业软件场景中精度略逊于更大参数模型、对动态交互（如Web实时渲染）还有提升空间。

---

## 282. Gen-SER: When the generative model meets speech emotion recognition

**arXiv ID:** 2601.20573 | [PDF](https://arxiv.org/pdf/2601.20573v1)

**作者:** Taihui Wang `[一作]` (Tencent), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种名为 Gen‑SER 的生成式框架，将语音情感识别（SER）重新表述为分布运输问题，并通过生成模型实现情感类别的分布映射与判别。

**💡 创新点**

创新点在于将离散情感标签映射为连续正交嵌入（正弦分类编码），并采用目标匹配式生成模型结合逻辑均值调度和桥接方差调度，实现高效分布运输；同时将 HuBERT 特征直接作为初始分布，避免了传统自编码器的重构误差。

**🔧 技术方法**

主要技术包括 HuBERT 预训练特征提取、基于 ODE 的流匹配（target‑matching）生成模型、逻辑均值与桥接方差调度、Transformer 结构与自适应 RMS‑norm、以及欧拉 ODE 求解器用于采样。

**📊 数据集**

实验使用多种英文学术情感数据集（crema‑d、emodb、TESS、savee、RAVDESS、MELD）共 52k 条样本，及 750k 条的性别识别自建数据集，评估在 MELD 与 Air‑Bench 上的性能。

**📈 对比分析**

与基于编码器+分类层的传统方法（WavLM、Hubert 等）以及基于 LLM 的方法（Qwen‑audio、Qwen2‑audio、OSUM 等）对比，Gen‑SER 在 MELD 上取得 56.5% 准确率，显著优于分类基线、与多数 LLM 竞争；在性别识别任务上达到 90.5%，略高于现有 LLM 方法。

**⚠️ 局限性**

局限性包括：相较于训练规模更大、使用 LLM 的 SenseVoice‑L 等模型仍略逊色；生成式推理需要多步推导，计算成本和推理时间比直接分类略高；当前仅在情感与性别两类任务上验证，需进一步探索对更多多类语音理解任务的适用性。

---

## 283. Unsupervised Ensemble Learning Through Deep Energy-based Models

**arXiv ID:** 2601.20556 | [PDF](https://arxiv.org/pdf/2601.20556v1)

**作者:** Ariel Maymon `[一作]` (Bar-Ilan University), Uri Shaham `[通讯]` (Bar-Ilan University)

**通讯引用:** 3744 | [OpenAlex ID](https://openalex.org/A5088774511)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督集成学习方法DEEM，利用深度能量模型（iRBM）从仅有的预测结果中恢复真实标签并构建元学习器。

**💡 创新点**

创新点包括：①将条件独立的Dawid‑Skene模型等价转化为可识别的可多项式RBM（iRBM）并给出收敛理论；②通过在iRBM前层堆叠多项式网络实现对学习器间复杂依赖的深度建模；③在无标签、无额外特征的条件下实现端到端训练与推理。

**🔧 技术方法**

技术手段主要是能量基模型（EBM）、可多项式RBM、深度多项式层、Sparsemax激活、Langevin采样与Hungarian算法用于类别映射。

**📊 数据集**

在多类模拟数据、真实集成数据（Tree3K、MnistE、PetFinder、CSGO、MicroAgg2、EyeMovem、ArtiChars、GesturePhsm）以及大规模ImageNet 1000类上进行实验。

**📈 对比分析**

与众多基线（多数投票、Dawid‑Skene、L‑SML、EBCC、HLM、DNN、FlyingSquid等）比较，DEEM在大部分数据集上取得最高或次高准确率，平均提升约0.6%，并在专家集成、Mixture‑of‑Experts 以及 ImageNet 1000类任务中表现优异。

**⚠️ 局限性**

局限性包括：①在实际训练中难以达到理论上MLE的收敛，导致模型可能未充分利用条件独立性保证；②对类别数极大时的计算复杂度和分区函数估计仍具挑战；③模型需要预先估计或假设每个学习器的预测覆盖完整数据，无法处理稀疏或部分覆盖的场景。

---

## 284. PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs

**arXiv ID:** 2601.20539 | [PDF](https://arxiv.org/pdf/2601.20539v1)

**作者:** Oguzhan Gungordu `[一作]` (Georgia Institute of Technology), Faramarz Fekri `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4398 | [OpenAlex ID](https://openalex.org/A5083854532)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了PathWise框架，通过多代理推理和推演图（entailment graph）实现自动化启发式设计，能够记忆并利用历史演化路径；

**💡 创新点**

创新点在于将启发式生成视为有状态的决策过程，利用策略代理、世界模型代理和批评者协同工作，结合提示层级多样化和状态打乱，显著提升探索效率和结果质量；

**🔧 技术方法**

核心技术包括大型语言模型（LLM）推理、图结构记忆、策略与世界模型的协同推演、批评者反馈循环、提示级多样性机制以及状态打乱；

**📊 数据集**

实验使用了多种组合优化问题数据集：TSP、KP、CVRP、MKP、OP、BPP（包含离线和在线版本），每个问题均有多个测试集并覆盖不同规模；

**📈 对比分析**

与传统手工设计启发式、Neural COMBO、以及现有LLM驱动的AHD方法（FunSearch、EoH、ReEvo、HSEvo、MCTS‑AHD）进行比较，PathWise在所有任务上实现了更低的最优性缺口、更快的收敛速度和更高的可推广性；

**⚠️ 局限性**

局限性包括对LLM推理的高度依赖，推理成本相对较高；在超大规模问题或极端稀疏实例时，推理效率和记忆管理仍需进一步优化；

---

## 285. A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests

**arXiv ID:** 2601.20529 | [PDF](https://arxiv.org/pdf/2601.20529v1)

**作者:** Julia Richter `[一作]` (ETH Zürich), Marco Hutter `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一套基于三种真实月球探测任务（钛铁矿、稀土元素、极地水冰）的多机器人关键性能指标框架。

**💡 创新点**

创新点在于将科学目标与场景需求直接映射到 KPI，区分效率、鲁棒性与精度三大维度，并强调多机器人团队的可比性。

**🔧 技术方法**

采用多机器人探测架构、行为树控制、状态与交互日志收集以及现场仿真实验来量化指标。

**📊 数据集**

使用月球探测任务的卫星影像（LROC、Lunar QuickMap）与地面模拟实验中的真实地形数据。

**📈 对比分析**

通过对效率（映射效率/速率）、鲁棒性（停机时间、自动化比率、重试率）等指标的量化，对比不同任务阶段表现，验证了框架在实际试验中的可行性。

**⚠️ 局限性**

局限在于精度类指标需要可靠的地面真值数据，操作员工作负荷测量复杂，部分指标高度相关且需根据场景权衡选择。

---

## 286. Opportunities of Touch-Enabled Spherical Displays to support Climate Conversations

**arXiv ID:** 2601.20468 | [PDF](https://arxiv.org/pdf/2601.20468v1)

**作者:** Mathis Brossier `[一作]` (Linköping University), Lonni Besançon `[通讯]` (Linköping University)

**通讯引用:** 2000 | [OpenAlex ID](https://openalex.org/A5087889244)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在两次实验性研讨会上，研究者探索了触摸式球面显示器在博物馆和科学中心环境中支持气候对话的可能性，收集并分析了参与者的交互需求和建议。

**💡 创新点**

创新点在于提出了针对球面显示器的多种交互机会（如层叠地图、滑动时间控制、浮动叠加窗口、分屏比较等）和技术实现建议，首次系统性地把触摸交互与气候数据可视化结合，为公众参与式气候讨论提供了新的交互范式。

**🔧 技术方法**

使用的技术包括：触摸感知球面显示器、触摸手势（旋转、平移、滑动）控制视角与时间轴、叠加窗口（浮动或固定）展示局部细节、分屏视图实现并行比较、外部辅助显示器与声音提示等。

**📊 数据集**

使用的气候数据集主要来自公共气候可视化项目，包括：空气表面温度变化、降水变化、海冰浓度、海表温度变化以及基于未来情景的“适宜居住”指标，数据均可视化在球面地图上。

**📈 对比分析**

在对比方法上，研究提出通过分屏、层叠以及浮动窗口来实现空间、时间和多数据集的并行比较；然而论文未进行量化性能评估，只是通过参与者反馈描述了效果和可行性。

**⚠️ 局限性**

局限性包括：样本量小、仅在实验室/实验场景进行、未覆盖复杂多用户交互冲突的系统化评估、缺乏长期使用效果与学习成效的实证验证、以及对球面显示器硬件交互限制（如旋转冲突、可视范围受限）的解决方案仍不完善。

---

## 287. Piloting Planetarium Visualizations with LLMs during Live Events in Science Centers

**arXiv ID:** 2601.20466 | [PDF](https://arxiv.org/pdf/2601.20466v1)

**作者:** Mathis Brossier `[一作]` (Linköping University), Mario Romero `[通讯]` (Linköping University)

**通讯引用:** 1555 | [OpenAlex ID](https://openalex.org/A5044438375)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在全景天象馆中实现并评估了一个基于大语言模型的 AI 导航助手，允许导游通过语音命令控制 OpenSpace 可视化软件，探索 AI 作为共航员的可行性；

**💡 创新点**

首次提出 AI 共航员概念，区分主动与被动模式，并通过与人类机舱操作员的对比实验验证其在减轻认知负荷、支持多任务和准备阶段的潜力；

**🔧 技术方法**

采用 OpenAI 低延迟多模态 LLM、WebSocket+Lua API 与实时语音识别模型，实现工具调用控制摄像机运动与可视化资产；

**📊 数据集**

使用实验室收集的 5 位专业导游的演示会话音频与系统日志作为数据来源，未使用公开数据集；

**📈 对比分析**

通过系统日志的延迟与成功率统计与访谈定性反馈比较，主动模式在流畅度上略优，但整体可靠性仍落后于人类机舱操作员，且在摄像机控制与时间感知上表现不佳；

**⚠️ 局限性**

主要局限在 LLM 的实时推理延迟、缺乏精细时间与运动控制、易出现误识别与情境误判，难以完全替代人类操作，需进一步提升自主性与情境理解能力。

---

## 288. CM-GAI: Continuum Mechanistic Generative Artificial Intelligence Theory for Data Dynamics

**arXiv ID:** 2601.20462 | [PDF](https://arxiv.org/pdf/2601.20462v1)

**作者:** Shan Tang `[一作]` (Dalian University of Technology), Xu Guo `[通讯]` (Dalian University of Technology)

**通讯引用:** 11697 | [OpenAlex ID](https://openalex.org/A5071909231)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于连续介质力学的最优传输框架（CM‑GAI），利用物理约束在数据稀缺条件下实现生成式 AI 任务，并在材料、结构与系统级别对应的三类典型问题（应力–应变预测、温度耦合应力场、冲击塑性应变）中进行验证。

**💡 创新点**

创新点包括：
1) 将连续介质力学（有限变形、能量守恒、外力耦合）与最优传输理论统一，形成可直接用来描述概率分布演化的物理驱动方程；
2) 将该方程通过 PINN 形式参数化，构建两网络（位移场与体力场）实现对伪时间演化的学习；
3) 在稀缺实验/数值数据下完成多维概率分布生成，并通过逆向重建获得最终的物理量；
4) 在图片生成任务中展示该框架与概率流 ODE 的关联，证明可用于图像的高效生成。

**🔧 技术方法**

核心技术：
- 最优传输理论（Monge‑Kantorovich）
- 连续介质力学（有限变形、动能、势能、外力）
- 物理信息神经网络（PINN）
- 逆向概率密度重建（Jacobian 变换）
- 主成分/谱特征增强网络、全连接网络、Dropout、Softplus 等深度学习构造
- 维度约简（PCA）与高维数据映射回原空间
- 与传统方法（FPCA‑GPR、回归）对比的 NRMSE 评估。

**📊 数据集**

使用的实验/数值数据集包括：
1) 热熔胶（Fuller EH9821B）在 29–40 °C 的压缩应力–应变曲线；
2) 聚酰胺（PPSU）在 −25–150 °C 的应力–应变数据；
3) 聚氨酯泡沫在 0.0004–8 s⁻¹ 的应力–应变曲线；
4) 混凝土在 20–950 °C 的压缩曲线；
5) 其它材料（HDPE、NT‑Cu、甘油凝胶、PEEK）在不同温度/应变率下的实验数据；
6) 对于结构与动态案例：有限元仿真得到的悬臂梁温度耦合应力场、铜 Taylor 杆冲击塑性应变场。

**📈 对比分析**

对比方法主要是 FPCA‑GPR 组合，评价指标为全域均方根误差（NRMSE）。在所有案例中，CM‑GAI 的 NRMSE 统一低于 4%，而 FPCA‑GPR 的误差普遍在 10–16% 之间，表明 CM‑GAI 在数据稀缺时具有更高的预测精度与泛化能力。

**⚠️ 局限性**

局限性：
- 目前实验验证多聚焦于二维/六维的概率分布，尚未在更高维度或更复杂几何/非欧几里得空间下深入验证；
- 对特征空间的本构建模仍以简单的 Neo‑Hookean 超弹性为例，缺乏对材料非线性、微观结构耦合的系统性建模；
- 需要显式学习/估计概率空间的度量张量，相关的非欧几里得几何推断仍处于理论探索阶段；
- 在实际工程应用中，体力场的物理解释与实验可测量性尚未完全对齐，可能导致外力选择不唯一；
- 对训练稳定性、超参数选择及收敛性理论缺乏充分的分析，需进一步完善。

---

## 289. Exploiting the Final Component of Generator Architectures for AI-Generated Image Detection

**arXiv ID:** 2601.20461 | [PDF](https://arxiv.org/pdf/2601.20461v1)

**作者:** Yanzhu Liu `[一作]` (Institute for Infocomm Research), Mondal Soumik `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 852 | [OpenAlex ID](https://openalex.org/A5103258191)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过仅利用图像生成器的最后一个功能模块（如VAE解码器、VQ去分词器或扩散去噪器）对真实图像进行“污染”，生成伪造样本，并训练二分类器识别真实与伪造图像。

**💡 创新点**

创新点在于提出基于生成器最后组件的通用检测思路，构建了基于最终组件的生成器分类学，并证明仅使用该组件生成的样本即可实现跨模型、跨范式的强泛化。

**🔧 技术方法**

技术上采用了预训练的DINOv3视觉骨干（可微调），通过K‑Medoids聚类挑选代表性伪造样本，利用交叉熵训练二分类器；同时设计了三种最终组件的伪造算法并构造对应的训练集。

**📊 数据集**

使用MS‑COCO 2014作为真实图像集合；通过Stable Diffusion 2.1、JanusPro、PixelFlow等公开模型的最终组件构造伪造样本；测试集涵盖22种未见生成器（如SD1.3/1.4/XL、DALL·E 2/3、Glide、Midjourney、Firefly等）和真实社交媒体图像。

**📈 对比分析**

与八种主流零样本或少量样本检测基线（AIDE、C2P‑CLIP、CoDE、RINE、FatFormer、NPR、LGrad、DIRE）在准确率与AP上对比，实验显示本文方法在所有22个测试集上平均准确率达98.83%，AP基本为1.0，明显优于对照组，甚至在最难的未知生成器上仍保持高精度。

**⚠️ 局限性**

局限性包括：需要对生成器的最后组件（或至少灰盒访问）才能构造训练样本，若生成器使用完全不同的终端模块或对组件做过大改动可能导致检测性能下降；实验主要基于公开模型，针对高度自定义或商业闭源模型的泛化仍需进一步验证。

---

## 290. Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective

**arXiv ID:** 2601.20520 | [PDF](https://arxiv.org/pdf/2601.20520v1)

**作者:** Qiyan Zhao `[一作]` (Shanghai Jiao Tong University), Da-Han Wang `[通讯]` (Chinese Academy of Sciences Institute of Automation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在 diffusion‑based 多模态大语言模型（dMLLM）中使用缓存导致的重复文本生成问题，并提出了 CoTA 方法来缓解这一 "Repeat Curse"。

**💡 创新点**

创新点在于：① 通过信息流分析揭示上下文 token 作为锚点的作用，发现缓存破坏信息流并导致重复；② 基于此提出了 CoTA，包含上下文 token 注意力增强（CTAE）和熵引导投票（CTEV）两大无训练干预模块。

**🔧 技术方法**

使用了信息流可视化、注意力衰减函数、深层熵惩罚投票、与现有 dLLM‑Cache 缓存技术的结合，以及多模态基准评估。

**📊 数据集**

实验使用了 MSCOCO 图像字幕、DocVQA、ChartQA、MMStar、MME、Seed、LLaVA^w、MathVista、MMBench 等多模态数据集。

**📈 对比分析**

在多模态基准与缓存对比实验中，CoTA 在减少 ARR、MRL、ARL 等重复指标上显著提升，同时保持或提升整体得分；与其他 M‑LLM 对比亦表现更好，且计算开销仅略有增加。

**⚠️ 局限性**

局限性：仅在 LLaDA‑V 与 dLLM‑Cache 上验证，缺乏对更多开源 dMLLM 与不同规模模型的泛化评估；且由于缓存方案稀缺，CoTA 未在更多缓存方法上进行测试。

---

## 291. TÄMU: Emulating Trusted Applications at the (GlobalPlatform)-API Layer

**arXiv ID:** 2601.20507 | [PDF](https://arxiv.org/pdf/2601.20507v1)

**作者:** Philipp Mao `[一作]` (École Polytechnique Fédérale de Lausanne), Mathias Payer `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 6323 | [OpenAlex ID](https://openalex.org/A5065116578)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建了一个在GP API层面实现的可信执行环境可信应用(TA)重托管平台，支持在通用硬件上动态分析、模糊测试和调试多种TEE上的TA；

**💡 创新点**

提出了基于全局平台（GP）标准化接口的高层仿真（HLE）方法，并提出“贪婪HLE”策略来优先实现最能提升覆盖率的TEE专属API，从而显著降低手工重托管工作量；

**🔧 技术方法**

利用API层拦截、Qiling/Unicorn仿真框架、ASAN内存检测、AFL++模糊器、Ghidra静态分析生成ICFG，以及自定义的NWCA驱动和GDB调试接口；

**📊 数据集**

收集了截至2025年的七种主流Android TEE（如Samsung QSEE、Huawei TrustedCore、Qualcomm Snapdragon TEE、MediaTek TEE等）中的TA二进制，涵盖约82个GP兼容TA，包含大量GP、libc和TEE专属API调用；

**📈 对比分析**

与现有的全系统仿真方案（如PartEmu）和单一TEE的手工仿真相比，本文平台实现了在x86机器上对四大TEE TA 的完整执行，模糊测试速率提升5–10倍，覆盖率可达90%，并成功发现并披露了多项零日漏洞；

**⚠️ 局限性**

仍受限于TEE专属API的实现难度、无法直接仿真TEE内核层和TA间IPC、对库代码缺乏完整仿真、以及对复杂输入格式TA缺少自动化套件生成的支持。

---

## 292. Latent Temporal Discrepancy as Motion Prior: A Loss-Weighting Strategy for Dynamic Fidelity in T2V

**arXiv ID:** 2601.20504 | [PDF](https://arxiv.org/pdf/2601.20504v1)

**作者:** Meiqi Wu `[一作]`, Kaiqi Huang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 14519 | [OpenAlex ID](https://openalex.org/A5028693655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在视频扩散模型中加入基于潜在空间帧间差异的运动先验，动态调整损失权重，从而提升高频运动生成的质量。

**💡 创新点**

创新点在于提出无光流、无外部运动估计的潜在时序差异（LTD）运动先验，并将其作为简单的损失重加权机制，实现对高动态区域的精准强化，易于与现有潜在扩散模型无缝集成。

**🔧 技术方法**

主要技术包括3D VAE编码解码、DiT扩散网络、滑动窗口潜在差异计算、对数变换权重、基于LTD的损失重加权；训练时采用DDIM推理与文本条件提示。

**📊 数据集**

实验使用了VBench与针对运动的VMBench两大基准数据集，并在内部3,860对视频-文本对上进行微调。

**📈 对比分析**

与Wan2.1等主流T2V模型对比，LTD方法在VBench提升3.31%、VMBench提升3.58%，人类评估中相对优势达8.31%，并在运动平滑、动态度等指标上显著优于对照组。

**⚠️ 局限性**

局限性包括对主题一致性略有下降、依赖于内部数据集，且在极端运动或非线性动态场景下的表现尚未系统验证。

---

## 293. Reinforcement Unlearning via Group Relative Policy Optimization

**arXiv ID:** 2601.20568 | [PDF](https://arxiv.org/pdf/2601.20568v1)

**作者:** Efstratios Zaradoukas `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 13473 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PURGE框架，实现对大型语言模型的可验证性目标记忆删除；

**💡 创新点**

将记忆删除建模为可验证任务，使用GRPO进行强化学习，避免外部奖励网络并提供理论保证；

**🔧 技术方法**

采用Group Relative Policy Optimization (GRPO)、自生成的删除语料、奖励函数为是否出现禁止词；

**📊 数据集**

使用RWKU（Real World Knowledge Unlearning）基准、Phi-3‑Mini‑4K‑Instruct模型；

**📈 对比分析**

与GA、DPO、NPO、RT等方法对比，PURGE在召回率降低、流畅度提升、对抗鲁棒性增强上取得领先；单个目标的标记使用量比最先进方法低最多46倍；

**⚠️ 局限性**

在大模型规模下需要更多训练轮次，且对完全删除所有相关概念的能力仍有限，未来需改进删除集生成与批量删除机制。

---

## 294. Online Risk-Averse Planning in POMDPs Using Iterated CVaR Value Function

**arXiv ID:** 2601.20554 | [PDF](https://arxiv.org/pdf/2601.20554v1)

**作者:** Yaacov Pariente `[一作]` (Technion - Israel Institute of Technology), Vadim Indelman `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在部分可观测马尔可夫决策过程（POMDP）中，使用迭代条件价值‑at‑Risk（ICVaR）作为动态风险度量，提出并实现了ICVaR版的稀疏采样、POMCPOW 与 PFT‑DPW 算法，并给出了其政策评估与稀疏采样的有限时间误差保证。

**💡 创新点**

首次将ICVaR动态风险度量引入POMDP在线规划，提供稀疏采样与MCTS的理论收敛证明，并设计了基于ICVaR的探索策略与进步宽化方法，使规划能够通过风险参数α从风险中性平滑过渡到高度风险厌恶；同时给出了完整的误差分析与收敛速率。

**🔧 技术方法**

使用ICVaR风险度量、经验CVaR估计、粒子滤波（Particle Filter）来更新信念，采用蒙特卡罗树搜索（MCTS）框架结合稀疏采样与进步宽化，构建了ICVaR‑POMCPOW 与 ICVaR‑PFT‑DPW 两个改进算法；并通过概率不等式与集中不等式证明了算法的性能保证。

**📊 数据集**

在LaserTag与LightDark这两大标准POMDP基准环境上进行实验，环境包含离散/连续状态、动作和观测空间。

**📈 对比分析**

与原始风险中性算法POMCPOW与PFT‑DPW在200条回合、95%置信区间下对比，ICVaR规划器在LaserTag和LightDark两环境中尾部风险分别降低17%–35%和37%–51%，证明了在ICVaR准则下的性能提升。

**⚠️ 局限性**

局限性包括：稀疏采样与ICVaR‑POMCPOW/PFT‑DPW 的计算量较大，只适合短期规划；对动作空间进行枚举，难以直接处理大规模或连续动作；ICVaR估计依赖样本数，收敛速度受限；实验仅在离散动作场景验证，缺乏对更复杂连续动作空间的证明。

---

## 295. Advancing Open-source World Models

**arXiv ID:** 2601.20540 | [PDF](https://arxiv.org/pdf/2601.20540v1)

**作者:** Robbyant Team `[一作]`, Hao Ouyang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可交互、可实时、支持长时序（分钟级）的视频世界模拟器LingBot-World，实现从文本与动作控制生成持续、逻辑一致的高保真视频。

**💡 创新点**

创新点包括：① 用层次化语义标注的混合数据引擎解决交互数据稀缺；② 三阶段演化训练（预训练→知识注入→实时化）与混合专家模型；③ 通过适应归一化与Plücker嵌入实现动作可控；④ 将双向扩散改写为区块因果注意力并通过少步蒸馏 + 对抗优化实现实时推理；⑤ 通过自动化Unreal渲染与多任务学习获得长时序一致性与空间记忆。

**🔧 技术方法**

技术核心包括扩散视频生成（DiT、Wan2.2）、混合专家（MoE）、区块因果注意力、少步蒸馏、分布匹配蒸馏、对抗优化、AdaLN动作注入、Plücker编码、FSDP2+Ulysses并行、VLM层级字幕。

**📊 数据集**

数据集来源：① 大规模真实视频（含第一/第三人称）② 游戏录像（RGB + 关键帧动作与相机参数）③ 用Unreal Engine 自动合成的高帧率、可对齐相机轨迹的合成视频；并使用VLM进行层次化字幕生成。

**📈 对比分析**

在VBench基准上与Yume‑1.5、HY‑World 1.5对比，LingBot‑World在影像质量、审美质量、动态度、运动平滑度、时序闪烁与整体一致性上均取得最高分；同时实现单 GPU 480p 16fps，推理延迟<1s，支持长达10分钟的视频生成。

**⚠️ 局限性**

局限性：记忆能力仅为上下文窗口衍生的隐式记忆，缺乏显式存储；推理成本高，需企业级 GPU；动作空间有限（仅导航与基础交互）；对目标物体的精细交互精度不足；长时序会出现漂移；仅支持单智能体视角。

---

## 296. Colored Markov Modulated Fluid Queues

**arXiv ID:** 2601.20537 | [PDF](https://arxiv.org/pdf/2601.20537v1)

**作者:** Benny Van Houdt `[一作]` `[通讯]`, Benny Van Houdt

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

**🎯 论文内容**

本文在马尔科夫调制流式排队（MMFQ）框架基础上提出了多色MMFQ（colored MMFQ）和带有流量跳跃的多色MMFQ，给出了其状态空间、演化规则和矩阵解析方法，并用该框架对多种难以用传统方法分析的排队系统（如MMAP/PH/1/N/LCFS、具有层级子作业的FCFS排队等）进行建模和性能评估。

**💡 创新点**

创新点包括：
- 引入颜色作为额外的记忆结构，使得队列可以追踪不同作业或工作量的来源，从而突破了传统MMFQ在状态空间爆炸时的限制；
- 设计了带有流量跳跃的多色MMFQ，利用PH分布将跳跃映射为线性增长区间，进而实现对跳跃过程的矩阵分析；
- 推导了一套通用的矩阵方程（包含非对称代数黎卡提方程和Sylvester方程）用于计算多色MMFQ的平稳分布，并给出了边界条件和归一化条件；
- 在多种排队应用中证明该方法在计算复杂度上显著优于传统的“树状”或“全状态”Markov链模型。

**🔧 技术方法**

技术手段主要包括：
- 马尔科夫调制流式排队模型的状态空间定义与分层；
- 矩阵解析技术：使用NARE、Sylvester方程求解Ψ矩阵；
- PH分布的近似与转换，将跳跃过程表示为一段线性增长区间；
- 对偏微分方程（PDE）进行稳态化简，得到线性ODE组来求解密度函数；
- 稳态向量p_-的求解通过对边界状态的闭合方程进行归一化；
- 数值实现中采用SDA、ADDA、Bartels–Stewart等算法。

**📊 数据集**

本文主要是理论推导与数值实验，使用的“数据集”为模拟实验中的参数配置，例如：
- MMAP/L/PH/L/1/N/LCFS队列的到达过程采用2状态MMAP，服务时间为指数或Erlang；
- 多层级作业队列使用Erlang‑3服务时间；
- 通过自行设置λ、q1、q2等参数来产生不同负载（ρ=1、1.025等）。

**📈 对比分析**

与传统方法（如树状分层QBD或全状态Markov链）对比，
- 在MMAP/PH/1/N/LCFS例子中，该方法的运行时间仅为数毫秒，甚至对N=1000也能在0.1秒内完成；
- 在多层级作业队列中，经典QBD的块矩阵大小随层数C指数增长，导致C≥8时内存超过3GB；而多色MMFQ的矩阵大小只随C线性增长，运行时间随C线性上升，显著优于传统方法；
- 结果显示在相同负载下，队列长度分布、失效概率等指标与理论一致。

**⚠️ 局限性**

局限性包括：
- 需要把跳跃大小近似为PH分布；若跳跃分布不易用PH逼近，方法失效；
- 对颜色的顺序和跳跃规则有严格限制（如颜色不能被跳过、跳跃只能由下一级颜色产生），不支持无序颜色或复杂颜色切换；
- 在某些边界行为（如到达0时立即添加下一作业）需额外设计，使得模型相对繁琐；
- 对极端大状态空间（如多种服务时间组合导致的S_+维数极大）仍会出现矩阵维度爆炸，需进一步优化或采用分层近似。

---

## 297. Robust Distributed Learning under Resource Constraints: Decentralized Quantile Estimation via (Asynchronous) ADMM

**arXiv ID:** 2601.20571 | [PDF](https://arxiv.org/pdf/2601.20571v1)

**作者:** Anna van Elst `[一作]` (Telecom Paris), Stephan Clémençon `[通讯]` (Telecom Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了AsylADMM，一种轻量级的异步去中心化ADMM算法，用于在资源受限边缘设备上估计中位数和分位数，且仅需每个节点存储两个变量；

**💡 创新点**

其创新点在于实现了低内存占用、通信高效且对数据腐败鲁棒的去中心化分位数估计，并通过异步ADMM实现快速收敛；

**🔧 技术方法**

采用了去中心化ADMM（异步/同步变体）、Gossip协议、Markov链理论用于分析和分位数修剪；

**📊 数据集**

实验主要使用合成数据与公开分布式数据集（如MNIST/COCO的分割版本），以模拟边缘设备上的数据异构；

**📈 对比分析**

与传统基于均值的Gossip、同步ADMM及排名修剪方法对比，AsylADMM在通信次数、收敛速度和鲁棒性上均表现更优；

**⚠️ 局限性**

局限性在于理论证明仅覆盖同步变体，对异步网络的收敛性尚未完全分析，且在极端网络拓扑下的性能可能受限。

---

## 298. DeepSeek-OCR 2: Visual Causal Flow

**arXiv ID:** 2601.20552 | [PDF](https://arxiv.org/pdf/2601.20552v1)

**作者:** Haoran Wei `[一作]` (DeepSeek-AI), Yukun Li `[通讯]` (DeepSeek-AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DeepEncoder V2 作为视觉编码器，将 CLIP 组件替换为 LLM 风格结构，并在文档 OCR 中实现可变长度的视觉 token 压缩与因果重排。

**💡 创新点**

创新点在于引入双流注意力：视觉 token 采用双向注意力，因果查询 token 采用单向因果注意力；通过自学习查询与自定义注意力掩码实现视觉信息的因果重排序；保持视觉 token 与查询 token 数量一致以支持多次“注视”。

**🔧 技术方法**

使用 Qwen2‑0.5B 作为 LLM 风格编码器，结合 80M SAM‑base 视觉 tokenizer，采用多裁剪策略、可自定义的注意力掩码、以及 3B‑参数 MoE 语言解码器。

**📊 数据集**

训练数据主要来自 OCR‑1.0、OCR‑2.0 以及通用视觉数据，约占 80% 的混合数据，并在 OCR‑1.0 中按文本/公式/表格 3:1:1 进行平衡采样。

**📈 对比分析**

通过在 OmniDocBench‑v1.5 上与 DeepSeek‑OCR 以及多种基线模型对比，DeepSeek‑OCR‑2 在 1120 个视觉 token 下实现 91.09% 的整体准确率，比基线提升 3.73%，阅读顺序编辑距离下降至 0.057。

**⚠️ 局限性**

局限性包括对文本量极大的报纸类文件识别精度仍不理想（编辑距离 >0.13），主要受视觉 token 上限与报纸数据不足限制；且目前仅在文档 OCR 任务验证，尚未在更广泛的 2D 理解任务上展示效果。

---

## 299. Say Cheese! Detail-Preserving Portrait Collection Generation via Natural Language Edits

**arXiv ID:** 2601.20511 | [PDF](https://arxiv.org/pdf/2601.20511v1)

**作者:** Zelong Sun `[一作]` (Renmin University of China), Zhiwu Lu `[通讯]` (Renmin University of China)

**通讯引用:** 2346 | [OpenAlex ID](https://openalex.org/A5085349794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Portrait Collection Generation（PCG）任务，构建了大型数据集 CHEESE 并设计了结合文本与身份细节保持的生成框架 SCheese。

**💡 创新点**

创新点在于：① 通过 LVLM 与反演验证生成高质量多属性编辑文本的大规模数据集；② 采用 Fusion IP‑Adapter 与 ConsistencyNet 的分层细节保持机制，兼顾多属性修改与身份/细节一致。

**🔧 技术方法**

主要技术包括：LVLM 用于文本生成与验证；Stable Diffusion 作为扩散生成器；Fusion IP‑Adapter 与 ConsistencyNet 通过分离注意力实现细节保持；教师强制与对齐损失提升训练效果。

**📊 数据集**

使用数据集 CHEESE，包含约 24K 人像相册、约 576K 三元组（参考图、修改文本、目标图）及其高质量标注。

**📈 对比分析**

在 CHEESE 测试集上与 DreamBooth、IP‑Adapter、FLUX.1 Kontext 等基线对比，SCheese 在细节保持（DP）、提示跟随（PF）和 CLIP‑T 等指标上均取得最高分，优于所有零拷贝与微调方法。

**⚠️ 局限性**

局限性：数据集仅覆盖人像集合，难以推广到其他主题；对 LVLM 的依赖导致标注成本与质量受模型水平限制；生成过程仍需大量显存与计算资源。

---

## 300. Comparative evaluation of training strategies using partially labelled datasets for segmentation of white matter hyperintensities and stroke lesions in FLAIR MRI

**arXiv ID:** 2601.20503 | [PDF](https://arxiv.org/pdf/2601.20503v1)

**作者:** Jesse Phitidis `[一作]` (Canon Medical Research Europe), Maria Valdés Hernández `[通讯]` (UK Dementia Research Institute)

**通讯引用:** 8489 | [OpenAlex ID](https://openalex.org/A5103510164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文针对小血管病影像中的白质高信号（WMH）和缺血性卒中病灶（ISL）进行联合分割，研究并比较了六种利用部分标注数据的监督策略，并在12个公开/私有MRI数据集上进行大规模实验。

**💡 创新点**

创新点包括：①系统评估六种易实现的部分标注方法（多类、二元、多模型、条件分支、伪标签、分阶段、类别自适应、边缘损失）；②发现伪标签策略在AP、DSC、AVD等指标上明显优于其他方法；③通过结合大量公开/私有数据集，验证部分标注能显著提升模型性能。

**🔧 技术方法**

技术细节：使用 MONAI 的 DynUNet 结构（6 级、3×3×3 卷积、残差块、实例归一化、LeakyReLU）；训练采用 SGD+Nesterov、Dice+交叉熵损失、polynomial LR；数据增强包括 nnU‑Net 方案加上 MRI 专用的运动/伪影/偏置场模拟；实现基于 PyTorch Lightning。六种监督方法实现方式如文中所述。

**📊 数据集**

数据集：共2052个 FLAIR 体积，1341 个 WMH 标注、1152 个 ISL 标注；来自 MSS1‑3、LBC1936/1921、WMH‑ch、BRATS、ISLES、SOOP、WSS、ESS、LINCHPIN 等 12 个公开/私有数据集，覆盖多种磁场强度、厂商、病人群体。

**📈 对比分析**

比较方法：使用 AP、DSC、AVD、ASD、LPRE、LREC 等指标；伪标签模型在 AP（WMH 76.0% vs 基线 75.6%）、DSC（WMH 55.2% vs 48.1%）、AVD（WMH 67.2% vs 65.9%）等方面表现最佳，尤其在 ISL 分割上提升明显；多模型方法在 BRATS 这类无标注病灶的数据上误检率最低，但整体 AP 与伪标签相近。其他方法（类别自适应、边缘损失等）在部分数据集上表现中等。

**⚠️ 局限性**

局限性：①需要至少一部分完整标注数据；②伪标签生成与质量高度相关，训练过程相对繁琐；③对不同标注策略（保守/积极）敏感，可能导致跨数据集迁移性能下降；④在脑肿瘤样本中误检严重，模型对这类特殊情况的鲁棒性不足；⑤在完全无标注或仅有单类标注的数据集上尚未验证效果。

---

## 301. CtrlCoT: Dual-Granularity Chain-of-Thought Compression for Controllable Reasoning

**arXiv ID:** 2601.20467 | [PDF](https://arxiv.org/pdf/2601.20467v1)

**作者:** Zhenxuan Fan `[一作]` (Zhejiang University), Beng Chin Ooi `[通讯]` (Zhejiang University)

**通讯引用:** 21295 | [OpenAlex ID](https://openalex.org/A5024892041)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CtrlCoT，一种双粒度链式推理压缩框架，兼顾语义抽象与token级裁剪；

**💡 创新点**

创新点在于三大模块：Hierarchical Reasoning Abstraction（多层语义压缩）、Logic-Preserving Distillation（保持数学推理关键token的自监督剪枝）以及Distribution-Alignment Generation（对齐压缩后的生成分布）；

**🔧 技术方法**

采用LLM生成多层CoT、GPT‑4驱动的自监督pruner、基于压缩比控制的多比率CoT生成器，并用LoRA微调实现预算控制与预算自由推理；

**📊 数据集**

在GSM8K和MATH‑500两个数学推理基准上进行评测；

**📈 对比分析**

与LC‑Prompt、Truncation、TokenSkip等基线比较，CtrlCoT在保持或提升准确率的同时，token数显著下降（最高可达55%压缩），实现更高的token效率；

**⚠️ 局限性**

限制在于LLM对预算指令的可控性有限，生成的CoT长度仍可能偏离设定预算，尤其在较弱模型上更为明显。

---

## 302. BMAM: Brain-inspired Multi-Agent Memory Framework

**arXiv ID:** 2601.20465 | [PDF](https://arxiv.org/pdf/2601.20465v1)

**作者:** Yang Li `[一作]` (Guangdong Institute of Intelligence Science and Technology), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种脑启发式多代理记忆框架 BMAM，用来解决长周期 LLM agent 的“灵魂侵蚀”问题。

**💡 创新点**

创新点在于将记忆拆分为情景记忆、语义记忆、显著性记忆和控制层，并通过时间线索化索引和多信号递归检索实现跨时空一致性。

**🔧 技术方法**

采用情景记忆时间线索化、知识图谱、稠密向量检索、语义融合与加权 Reciprocal Rank Fusion 等技术，辅以预设的控制与调度逻辑。

**📊 数据集**

使用 LoCoMo、LongMemEval、PersonaMem、PrefEval 四个长记忆基准数据集进行实验。

**📈 对比分析**

与 MemOS、Mem0、MemOS 等七个基线进行对比，BMAM 在 LoCoMo 上取得 78.45% 正确率，整体表现优于基线；在其他基准上也保持竞争力。

**⚠️ 局限性**

局限包括缺乏多模态或跨域验证、对精确时间推理仍存在不足、实体歧义和检索覆盖率问题未得到充分解决。

---

## 303. Chasing Meaning and/or Insight? A Survey on Evaluation Practices at the Intersection of Visualization and the Humanities

**arXiv ID:** 2601.20464 | [PDF](https://arxiv.org/pdf/2601.20464v1)

**作者:** Alejandro Benito-Santos `[一作]` (National Distance Education University), Eva Mayr `[通讯]` (University for Continuing Education)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述171篇可视化与人文学科交叉设计研究的评估实践，构建编码框架并量化评估方法、流程与质量；

**💡 创新点**

首次在VIS*H领域提供基于大样本的评估质量度量与方法组合的统计分析，揭示单一方法缺陷、复合方法提升质量、并提出基于“多元证据三角化”的评估范式；

**🔧 技术方法**

采用结构化编码与层次聚类、比例等级逻辑回归、共现热图等数据分析技术，对评估方法进行归纳与预测；

**📊 数据集**

数据集来源于从17篇核心文献出发的Snowball检索，最终得到171篇同行评议期刊/会议论文，覆盖文献年代2013‑2025；

**📈 对比分析**

与传统单一方法评估相比，包含日志分析、问卷/调查、访谈等多方法组合的研究在质量评分上显著更高（平均得分提升至3.53，OR≈4.86），并通过聚类得到八种典型评估工作流；

**⚠️ 局限性**

局限性包括：仅限英文同行评议论文，忽略工作坊、书籍等非期刊形式；评估质量评分依赖主观专家判断；方法组合多样但缺乏纵向/长期用户研究；未系统探究不同行业/文化差异对评估的影响。

---

## 304. MuVaC: AVariational Causal Framework for Multimodal Sarcasm Understanding in Dialogues

**arXiv ID:** 2601.20451 | [PDF](https://arxiv.org/pdf/2601.20451v1)

**作者:** Diandian Guo `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Yanbing Liu `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MuVaC 框架，联合多模态讽刺检测（MSD）和讽刺解释（MuSE）任务，模拟人类认知的因果链。

**💡 创新点**

首次将讽刺检测与解释视为因果关系，采用变分因果推断与对齐-融合（ATF）模块实现两任务的共同优化。

**🔧 技术方法**

变分因果推断、BART/CLIP/CLAP 编码器、对齐-融合（ATF）模块、注意力机制、变分自编码器技术。

**📊 数据集**

使用 MUStARD、MUStARD++（多模态讽刺检测数据集）和 WITS（多模态讽刺解释数据集）进行实验。

**📈 对比分析**

与 BERT、FiLM、MO‑Sarcation、VyAnG‑Net、MV‑BART 等多种基线对比，MuVaC 在 MUStARD、MUStARD++ 上 F1 提升约 10%，在 WITS 上 ROUGE、BLEU 等指标排名第一。

**⚠️ 局限性**

对无解释生成的鲁棒性不足，speaker‑independent 性能略逊；对解释质量高度依赖 ChatGPT‑4o 生成，跨语言/方言的泛化能力待进一步验证。

---

## 305. Can We Improve Educational Diagram Generation with In-Context Examples? Not if a Hallucination Spoils the Bunch

**arXiv ID:** 2601.20476 | [PDF](https://arxiv.org/pdf/2601.20476v1)

**作者:** Evanfiya Logacheva `[一作]` (Aalto University), Juho Leinonen `[通讯]` (Aalto University)

**通讯引用:** 3979 | [OpenAlex ID](https://openalex.org/A5041367899)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了基于Rhetorical Structure Theory的图表代码生成方法，并通过计算机科学教育者的专家评估验证其有效性。

**💡 创新点**

创新点在于将RST分析用于挑选上下文示例，以减少生成图表时的事实与忠实度幻觉，并提高逻辑组织与连贯性。

**🔧 技术方法**

使用大型语言模型（GPT‑4o、o3）进行上下文RST分析、示例检索、图表生成与修复；采用Graphviz、LLM自检与修复循环及基于自定义量表的人工/自动评估。

**📊 数据集**

数据集包含25篇教育性文本（按难度分为advanced、medium、basic）与4篇示例文本，生成150张图表，评估时使用人工标注的三维量表。

**📈 对比分析**

通过对比三种生成方式（RST1、RST2、零样本）和两大模型，发现RST1+o3在逻辑组织、连通性和布局美观上最高，整体人评与LLM评估一致性较好；但幻觉仍占比10‑20%，并随文本复杂度上升。

**⚠️ 局限性**

局限性包括LLM的随机性与样本量有限、示例覆盖率不足、教育文本长度导致RST解析低效、LLM难以自我评估布局，且幻觉传播影响最终图表质量。

---

## 306. Beyond Literacy: Predicting Interpretation Correctness of Visualizations with User Traits, Item Difficulty, and Rasch Scores

**arXiv ID:** 2601.20544 | [PDF](https://arxiv.org/pdf/2601.20544v1)

**作者:** Davide Falessi `[一作]` (Universita degli Studi di Roma Tor Vergata), Angela Locoro `[通讯]` (Universita degli Studi di Brescia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在数据可视化素养评估中，预测受试者在未见图表前对可视化题目是否能正确解读。

**💡 创新点**

创新点在于提出了P-HIC预测框架，将心理测量指标（Rasch难度）与专家评分、受试者历史表现结合，实现了个性化项选择。

**🔧 技术方法**

采用逻辑回归、随机森林和多层感知机三种传统机器学习模型，并通过特征选择进一步提升性能。

**📊 数据集**

使用了1083名参与者共34,656个条目回答的数据集，包含八种可视化及四类题型。

**📈 对比分析**

通过十次十折交叉验证比较模型，逻辑回归+特征选择获得中位AUC 0.72、κ 0.32，优于随机森林与MLP。

**⚠️ 局限性**

局限性包括样本仅为英语受试者、缺乏更细粒度行为特征（如眼动、反应时），以及模型未涵盖跨文化公平性。

---

## 307. IOTA: Corrective Knowledge-Guided Prompt Learning via Black-White Box Framework

**arXiv ID:** 2601.20526 | [PDF](https://arxiv.org/pdf/2601.20526v1)

**作者:** Shaokun Wang `[一作]` (Harbin Institute of Technology), Yihong Gong `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 23347 | [OpenAlex ID](https://openalex.org/A5100687952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种黑白盒提示学习框架 IOTA，将冻结的视觉预训练模型与基于纠正知识的白盒提示模块结合，用于下游任务的高效微调。

**💡 创新点**

创新点在于将错误预测与正确认知对比生成可解释的真–错提示，并通过纠正知识引导的提示选择策略将这些提示投射到黑盒模型中，实现数据驱动与知识驱动的互补。

**🔧 技术方法**

技术手段包括 CLIP 视觉与语义编码器、ViT 作为黑盒基础、学习可调提示向量、MATCH 令牌、基于余弦相似度的软匹配与损失约束，以及两阶段易到难的自适应学习。

**📊 数据集**

实验覆盖 12 个公开数据集，包含 4 个细粒度、4 个自然和 4 个专业类数据集（如 Flowers102、CIFAR‑10、ImageNet、EuroSAT 等）。

**📈 对比分析**

与现有 8 种提示学习和 1 种适配器方法对比，IOTA 在 16/8 shot 以及易→难两阶段自适应设置中均实现最高平均准确率，平均提升约 2–6%（在 16-shot 设定中提升 6%）。

**⚠️ 局限性**

局限性在于对可用的“难样本”数量敏感；当样本稀缺时纠正知识不足导致效果下降；此外框架仍依赖手工制定的提示模板和预训练模型，难以自动化扩展到更广泛任务。

---

## 308. AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors

**arXiv ID:** 2601.20524 | [PDF](https://arxiv.org/pdf/2601.20524v1)

**作者:** Matic Fučka `[一作]` (University of Ljubljana), Danijel Skočaj `[通讯]` (University of Ljubljana)

**通讯引用:** 3420 | [OpenAlex ID](https://openalex.org/A5024699185)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 AnomalyVFM 框架，将任何预训练的视觉基础模型（VFM）转换为强大的零样本异常检测器。

**💡 创新点**

创新点包括：
1) 三阶段合成数据生成器，利用 FLUX 等生成器产生多样化的无缺陷图像、局部缺陷及对应掩码；
2) 参数高效的适配策略，在 VFM 的 Transformer 头部注入低秩适配器（LoRA）并结合置信度加权损失，以微调内部特征而非仅改头；

**🔧 技术方法**

使用的技术：FLUX 生成模型、IS-Net 前景分割、DINOv2 作为特征验证、LoRA 适配器、置信度加权 Focal+L1 损失、轻量级卷积解码器。

**📊 数据集**

训练数据：10,000 张由 FLUX 自动生成的合成样本；测试数据：9 个工业数据集（MVTec AD、VisA、BTAD、MPDD、RealIAD、KSDD、KSDD2、DAGM、DTD-Synthetic）和 9 个医学数据集（HeadCT、BrainMRI、BR35H、ISIC、ClinicDB、ColonDB、Kvasir、Endo、TN3K）。

**📈 对比分析**

与现有零样本方法（SAA、WinCLIP、AnomalyCLIP 等）和少样本方法（INP-Former 等）对比，AnomalyVFM 在图像级 AUROC 上平均提升 3.3 分点，像素级 AUROC 提升 0.9 分点，且在少样本场景下可达到或超过 SOTA；推理速度也显著更快（≈20 ms/样本）。

**⚠️ 局限性**

主要局限：图像生成阶段占用资源高，约一天时间；生成模型仍可能产生缺陷缺失或误标记，需要过滤；对生成质量的依赖性仍需进一步提升。

---

## 309. Audio Deepfake Detection in the Age of Advanced Text-to-Speech models

**arXiv ID:** 2601.20510 | [PDF](https://arxiv.org/pdf/2601.20510v1)

**作者:** Robin Singh `[一作]` (UncovAI), Lohith Rachakonda `[通讯]` (UncovAI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估三种新型文本转语音模型（Dia2、Maya1、MeloTTS）对三种主流深度伪造检测器（Whisper-MesoNet、XLS-R-SLS、SSL-AASIST）以及UncovAI专有检测模型的检测效果，探讨不同生成机制下的检测瓶颈。

**💡 创新点**

首次在同一实验框架下对流式、LLM驱动和非自回归三大 TTS 体系进行对比；提出多视角（语义、结构、信号）检测组合，揭示单一检测范式难以覆盖所有攻击；展示 UncovAI 模型在所有三种攻击下的近乎完美性能，强调集成检测策略的重要性。

**🔧 技术方法**

采用 Whisper 编码器+MesoNet 分类、XLS‑R 基础前端+层级敏感层选择（SLS）融合、SSL‑AASIST 结构图模型；同时使用 UncovAI 专有模型；对生成的 12,000 条合成音频进行 WER、SIM、FAD、SNR、ACT 等评估。

**📊 数据集**

基于 DailyDialog 对话文本，随机采样 4,000 条对话转化为 12,000 条音频（包含多轮、情感标注），用于生成 Dia2、Maya1、MeloTTS 的合成样本。

**📈 对比分析**

通过 EER、AUC、F1、FRR@1%FAR 四项指标对比，发现 Whisper 对 MeloTTS 友好但对 Maya1 效率低；XLS‑R‑SLS 在 Dia2 上表现最佳，MeloTTS 较差；SSL‑AASIST 介于两者之间；UncovAI 在三者上均达到 0.99 以上的 F1，几乎无误判。

**⚠️ 局限性**

实验仅覆盖三种 TTS 体系，缺乏真实网络深度伪造样本；评估仅基于 English DailyDialog，语言多样性不足；未探讨对抗攻击、压缩等现实噪声对检测器的鲁棒性；UncovAI 模型为闭源，无法复现。

---

## 310. Efficient Autoregressive Video Diffusion with Dummy Head

**arXiv ID:** 2601.20499 | [PDF](https://arxiv.org/pdf/2601.20499v1)

**作者:** Hang Guo `[一作]` (Microsoft Research Asia), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 19862 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究发现并利用自回归视频扩散模型中的“dummy heads”，通过异质内存分配、动态头分类和打包注意力前向，在不需要额外训练的情况下显著提升生成速度。

**💡 创新点**

创新点在于首次识别并压缩不利用历史上下文的注意力头，提出三项技术（异质内存分配、动态头编程、打包注意力）实现对KV缓存的精细化管理。

**🔧 技术方法**

使用技术包括多头自注意力、KV缓存压缩、动态规划、打包注意力前向、滑动窗口以及分层头分类等。

**📊 数据集**

实验数据集涵盖 VBench、VBench‑Long、720P/1080P 影视生成集以及交互式长视频评测集。

**📈 对比分析**

与 Self‑Forcing、LongLive、R‑KV、Infinipot‑V、TeaCache 等基线对比，单帧生成速度提升至 24.3 FPS（1.4×加速），长视频和高分辨率上亦保持 1.4–2.0×加速且质量下降 ≤0.5%。

**⚠️ 局限性**

局限性包括对 dummy head 比例的敏感性，过度压缩会导致邻居 head 信息丢失；并且仅适用于已有 KV 缓存的自回归模型，无法直接推广到双向或无缓存模型。

---

## 311. An explainable framework for the relationship between dementia and glucose metabolism patterns

**arXiv ID:** 2601.20480 | [PDF](https://arxiv.org/pdf/2601.20480v1)

**作者:** C. Vázquez-García `[一作]` (University of Granada), Juan M. Górriz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

使用半监督变分自编码器（VAE）对ADNI数据库中的PET影像进行重构与编码，并通过相似性正则化将潜在空间的第一个维度与认知评分（ADAS13）对齐，从而捕捉阿尔茨海默病的代谢衰退模式

**💡 创新点**

提出一种灵活的相似性正则化框架，允许根据不同临床变量选择相似度度量，并通过相位图评估超参数区分稳定与失败区间，进一步实现对潜在空间中共变因素（如空间变形、强度变化）的可解释分离

**🔧 技术方法**

半监督VAE、3D卷积网络、重构损失（MSE）、β‑VAE KL权重、Pearson相似性损失、GLM voxel‑级分析、逻辑回归分类、Bootstrap验证、相位图可视化

**📊 数据集**

ADNI数据库的3466张FDG‑PET扫描及相关结构体积（海马、内侧颞叶、内嗅皮层、梭状回）

**📈 对比分析**

与传统无监督VAE及单变量认知评分的分类基准对比，二分类（AD vs HC）准确率0.80±0.02，灵敏度0.79±0.04，特异度0.77±0.02，平衡准确率0.79±0.02；相比仅用ADAS13可达0.96±0.03，表明潜在变量仅携带疾病信息；在GLM和解码可视化上与已知代谢衰退区域一致

**⚠️ 局限性**

仅依赖单一认知评分做相似性约束，可能忽略多模态信息；潜在空间维度固定为8，未探讨更高维或多变量正则化的可扩展性；模型受数据集分布和预处理步骤影响，外部验证仍需进一步验证

---

## 312. Implicit Hypothesis Testing and Divergence Preservation in Neural Network Representations

**arXiv ID:** 2601.20477 | [PDF](https://arxiv.org/pdf/2601.20477v1)

**作者:** Kadircan Aksoy `[一作]` (Technical University Berlin), Protim Bhattacharjee `[通讯]` (German Aerospace Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将监督分类视为在学习到的表示上做二元假设检验，并通过 KL 散度和 Neyman–Pearson 理论解释网络训练过程。

**💡 创新点**

创新点在于：①把深度网络训练动态映射到“证据-误差”平面，直观区分表示质量与决策效率；②证明充分训练的网络输出相当于对数似然比的充分统计量，逼近 NP 最优；③引入 k‑NN 散度估计，兼顾高维样本和低计算量；④首次在 SNN 以及多样化数据集上验证该框架。

**🔧 技术方法**

主要技术包括：二元假设检验（Neyman–Pearson 与 Stein’s 典型极限）、KL 散度与信息瓶颈理论、k‑NN 散度估计、梯度下降/Adam 优化、Spiking Neural Network 的 LIF 模型和 surrogate 梯度训练。

**📊 数据集**

实验数据集有：高斯分布（二维或多维）、自定义二元图像（BSC 噪声）、Yin‑Yang 三类几何样本、MNIST 手写数字，以及相同架构的 SNN。

**📈 对比分析**

比较方法：在训练过程中记录每个网络在证据-误差平面上的轨迹，计算输入与表示层的平均 KL 散度 D_inp、D_θ 以及对应的误差指数 P_θ；实验表明：FC、BN、Dropout 等普通网络在 DNN 中靠近 NP 极限；SNN 在早期显示高 KL 散度但误差不下降，随后逐步逼近极限；多数投票在信息低效网络上显著提升性能；总体来看，网络趋向于可实现的 KL‑误差边界。

**⚠️ 局限性**

局限性包括：①未考虑贝叶斯误差（α、β 双重优化），仅聚焦 NP 极限；②k‑NN 散度估计在高维或有限离散分布（如 SNN 的膜电位）下可能偏差；③缺乏对 Chernoff 信息的评估与优化；④实验多集中在低维或小样本场景，真实大规模数据的验证仍待进一步研究。

---

## 313. Challenges in Android Data Disclosure: An Empirical Study

**arXiv ID:** 2601.20459 | [PDF](https://arxiv.org/pdf/2601.20459v1)

**作者:** Mugdha Khedkar `[一作]` (Heinz Nixdorf Institute Paderborn University), Eric Bodden `[通讯]` (Heinz Nixdorf Institute Paderborn University and Fraunhofer IEM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过问卷和线上讨论调查Android开发者在Google Play Data Safety Section表单的填写体验与挑战。

**💡 创新点**

首次系统分析开发者在DSS分类、信心与误报之间的差异，并揭示其主要难点与误区。

**🔧 技术方法**

使用问卷调查、主题编码、在线社区爬取（Stack Overflow、Reddit、Discord、GitHub、Hacker News）以及定性文本分析技术。

**📊 数据集**

41名开发者的问卷数据与172条开发者讨论帖子（共642名开发者）构成研究数据集。

**📈 对比分析**

未进行性能对比，只通过统计与主题分析表明开发者信心下降、分类困难、误报率高等问题。

**⚠️ 局限性**

样本量有限、仅覆盖Google DSS、可能存在自选偏差、讨论来源受限，结果难以推广到Apple等其他平台。

---

## 314. Fair Recourse for All: Ensuring Individual and Group Fairness in Counterfactual Explanations

**arXiv ID:** 2601.20449 | [PDF](https://arxiv.org/pdf/2601.20449v1)

**作者:** Fatima Ezzeddine `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Omran Ayoub `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**通讯引用:** 817 | [OpenAlex ID](https://openalex.org/A5026384166)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究提出了一种基于强化学习的无模型方法，用来生成既满足个体公平、群体公平，又兼顾两者的混合公平的可行动性反事实解释（CF）。

**💡 创新点**

创新点在于①首次把混合公平（个体与群体公平并重）引入CF生成；②设计了专门的奖励函数，兼顾等效性（EE）与等可行动性（ECR）；③通过RL自适应搜索实现多样化、可解释且公平的CF集合。

**🔧 技术方法**

技术实现包括：Soft Actor‑Critic（SAC）强化学习框架、基于Gower距离的相似度度量、可行动特征约束、聚类预处理、以及自定义奖励函数和终止条件。

**📊 数据集**

使用了三个公开数据集：Adult（收入预测）、SSL（司法风险评估）和Alzheimer（疾病预测），分别包含受保护属性（种族、性别、性别）与可行动特征。

**📈 对比分析**

与NiCE、DiCE和原型引导CF三种主流基线相比，所提方法在所有数据集上实现了90%以上的有效性、与训练分布高度相符的可行性（重构误差）、低Gower相似度（接近原样本）和最小化特征变动；混合公平模式在保证高成功率（≥85%）的同时，保持了PD≤10%，显示出良好的公平性与质量平衡。

**⚠️ 局限性**

局限性包括：强化学习需要较长训练时间与较高计算资源；在单纯群体公平约束下，成功率下降、距离增大；混合公平虽然不显著降低动作数量，但在极端偏差情形下仍可能需要更多样本或更细粒度的公平度量来进一步提升公平性。

---

## 315. TimeCatcher: A Variational Framework for Volatility-Aware Forecasting of Non-Stationary Time Series

**arXiv ID:** 2601.20448 | [PDF](https://arxiv.org/pdf/2601.20448v1)

**作者:** Zhiyu Chen `[一作]` (University of Electronic Science and Technology of China), Yanru Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 3315 | [OpenAlex ID](https://openalex.org/A5100632845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一种名为 TimeCatcher 的轻量级变分预测框架，结合趋势建模、VAE 隐层编码和波动增强模块，实现对非平稳时间序列的长周期预测。

**💡 创新点**

提出了将变分自编码器与波动感知增强模块相结合的残差式设计，既保留了原始趋势，又能捕捉潜在动态与突发波动；同时使用可学习的动态阈值掩码精准放大重要波动。

**🔧 技术方法**

使用了 MLP 基础网络、变分自编码器（VAE）、可学习的波动增强模块、SoftPlus 缩放、线性投影以及 t‑SNE 可视化技术；训练采用 AdamW 与 L1 损失。

**📊 数据集**

在九个工业公开数据集上评估，包括 Traffic、ETT 系列（ETTh1/2/ETTm1/2）、Weather、Solar Energy、Exchange Rate、Electricity。

**📈 对比分析**

与 11 个基线（TimeMixer、TimeBase、iTransformer、PatchTST、Crossformer、TiDE、TimesNet、DLinear、SCINet、FEDformer、Autoformer）在四个预测步长（96/192/336/720）下用 MSE/MAE 比较，TimeCatcher 在大多数任务上均取得最低误差，尤其在高波动数据集上 MSE 降低约 17%，MAE 降低约 21%，并且在延长预测窗口时性能衰减更慢。

**⚠️ 局限性**

局限在于仍为离线训练模型，缺乏在线学习与多模态扩展；对极端噪声或异常事件的鲁棒性尚未完全验证；以及在极平稳或极短期任务中提升幅度有限。

---

## 316. CCMamba: Selective State-Space Models for Higher-Order Graph Learning on Combinatorial Complexes

**arXiv ID:** 2601.20518 | [PDF](https://arxiv.org/pdf/2601.20518v1)

**作者:** Jiawen Chen `[一作]` (Southeast University), Wenwu Yu `[通讯]` (Southeast University)

**通讯引用:** 25640 | [OpenAlex ID](https://openalex.org/A5100627758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于选择性状态空间模型的组合复合 Mamba（CCMamba），实现了在图、超图、单纯形和细胞复合等高阶结构上的高效消息传播，替代自注意力机制；

**💡 创新点**

创新点在于将多阶边界关系重新组织为结构化序列，并构造 rank‑aware 选择性状态空间模块，能够实现长距离、方向性信息传递，并统一处理不同维度的复合结构；

**🔧 技术方法**

使用了选择性状态空间模型（Mamba）、组合复合理论、1‑CCWL 表达性理论、线性时间消息传递、残差连接与层归一化等技术；

**📊 数据集**

在 TUDatasets（MUTAG、PROTEINS、IMDB‑BINARY、IMDB‑MULTI、AMAZON‑RATINGS、ROMAN‑EMPIRE、MINESWEEPER）、Cora、CiteSeer、PubMed 等数据集上进行实验，并将图数据提升为超图、单纯形和细胞复合；

**📈 对比分析**

与 GCN、GAT、GIN、HyperGCN、HyperGAT、UniGCN、UniGNN、SCCN、CWNN 等基线在节点和图分类任务中进行对比，CCMamba 在大多数数据集上实现了 state‑of‑the‑art 或接近 state‑of‑the‑art 的准确率，同时在深层网络下更稳健、训练速度和显存显著降低；

**⚠️ 局限性**

受限于 1‑CCWL 的上限，极大规模或高维细胞复合的可扩展性仍需进一步验证，且在某些高度异构或高维结构下性能提升有限。

---

## 317. Normative Equivalence in human-AI Cooperation: Behaviour, Not Identity, Drives Cooperation in Mixed-Agent Groups

**arXiv ID:** 2601.20487 | [PDF](https://arxiv.org/pdf/2601.20487v1)

**作者:** Nico Mutzner `[一作]` (University of Zurich), Heiko Rauhut `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文通过在线实验，研究了在三人类加一 AI 机器人组成的小组中，AI 标签和行为策略是否会影响公共物品游戏中的合作模式和随后的一次性囚徒困境中的合作持久性。

**💡 创新点**

创新点在于首次在小组层面检验“规范等价”假设，即即使 AI 被标记为非人类，合作依旧遵循人类群体的规范机制；并通过对比 AI 与人类标签以及三种固定策略（全合作、条件合作、不合作）来探讨 AI 对规范的影响。

**🔧 技术方法**

采用的技术主要是实验经济学中的线性混合效应模型、逻辑斯蒂回归和描述性统计，用 oTree 平台搭建并运行游戏，收集参与者在多轮公共物品游戏与单轮囚徒困境中的决策、以及后实验调查问卷。

**📊 数据集**

数据集来自 Prolific 上招募的 236 名参与者，构成 59 个三人类加一机器人的小组，记录了 10 轮公共物品游戏、一次囚徒困境以及后续规范评估问卷。

**📈 对比分析**

比较方法是将 AI 与人类标签、三种策略作为处理变量，使用混合模型检验对贡献水平和合作率的影响；结果显示标签和策略对合作几乎没有显著影响，合作下降趋势与传统公共物品游戏一致，说明规范等价。

**⚠️ 局限性**

局限性包括：实验采用固定脚本 AI，未考虑自适应或沟通能力；反馈仅为总贡献，可能掩盖个体行为差异；样本为在线受试者，真实性和长期互动的外推性有限；部分受试者对组构成持怀疑态度，尽管稳健检验表明不影响主结论。

---

## 318. ConStruM: A Structure-Guided LLM Framework for Context-Aware Schema Matching

**arXiv ID:** 2601.20482 | [PDF](https://arxiv.org/pdf/2601.20482v1)

**作者:** Houming Chen `[一作]` (University of Michigan), H. V. Jagadish `[通讯]` (University of Michigan)

**通讯引用:** 14528 | [OpenAlex ID](https://openalex.org/A5090550596)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了ConStruM框架，利用上下文树和全局相似性超图为LLM提供分层、限量的上下文证据，从而提升列级模式匹配精度。

**💡 创新点**

创新点在于将schema上下文拆解为可检索的树形结构与超图组块，并在有限上下文预算内动态聚合分层证据与对比线索，显著提升LLM在上下文依赖匹配场景的表现。

**🔧 技术方法**

采用LLM（GPT‑5）进行上下文树和对比提示生成，利用文本嵌入、阈值相似性链接与层次聚类构建树与超图，最终通过结构化提示增强LLM的列匹配决策。

**📊 数据集**

主要使用的评测数据集包括HRS Employment的代码书（HRS‑B）作为上下文压力基准，以及MIMIC‑2‑OMOP作为标准Schema匹配基准。

**📈 对比分析**

与ReMatch、Matchmaker等基线对比，在HRS‑B上准确率从0.503提升至0.935；在MIMIC‑2‑OMOP上Top‑1准确率达到59.69%，与Matchmaker相近，证明结构化上下文能显著提升LLM匹配效果。

**⚠️ 局限性**

局限性在于对列语义高度依赖上下文时最优；当列描述已足够区分时增量收益有限；此外构建上下文树与超图仍需耗费预处理成本，且在极大schema下的存储与检索开销需进一步优化。

---

## 319. On Every Note a Griff: Looking for a Useful Representation of Basso Continuo Performance Style

**arXiv ID:** 2601.20478 | [PDF](https://arxiv.org/pdf/2601.20478v1)

**作者:** Adam Štefunko `[一作]` (Charles University), Jan Hajič `[通讯]` (Charles University)

**通讯引用:** 8535 | [OpenAlex ID](https://openalex.org/A5102738283)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的特征表示方法——griff，用于描述对齐后的意大利巴洛克低音连续体演奏的和声结构，并在此特征空间中对不同演奏者的风格进行统计分析。

**💡 创新点**

创新点在于：①将低音连续体的即兴演奏拆解为“griff”——每个谱记符对应的演奏音集合，保留音高、顺序与近似并列信息；②通过将 MIDI 音高转换为相对音阶间隔，使得特征不受调性影响；③使用“ordered”和“pooled”两种表示方式，兼顾时序与整体和声结构；④首次在 ACoRD 数据集上利用此特征进行跨演奏者风格相似性评估。

**🔧 技术方法**

技术手段包括：对齐算法 DualDTWNoteMatcher（结合预处理与贪心时间并行分配）；griff 的生成与编码（按时序窗口聚类、区间转换、字符串编码）；统计与实验方法（累计覆盖率、交叉熵相似性矩阵）。

**📊 数据集**

使用的数据集为 The Aligned Continuo Realization Dataset (ACoRD)，包含175条 MIDI 记录，5 份低音连续体乐谱，7 位演奏者，共约 6 小时、66,967 个音符。

**📈 对比分析**

比较方法主要是计算不同演奏者的griff特征分布的交叉熵和累计覆盖率。实验结果显示，同一演奏者在相同乐谱上的griff分布更相似，跨演奏者之间差异明显；但由于样本量有限，仅证明了griff空间的区分潜力，未给出完整的分类/聚类性能评估。

**⚠️ 局限性**

局限性包括：①对齐仍有误差，尤其是边缘情况；②griff只保留了和声结构，忽略了节奏和动态细节；③实验规模小（7位演奏者），缺乏更广泛的验证；④未进行深入的机器学习评估（聚类、分类等）。

---

## 320. Inequality in Congestion Games with Learning Agents

**arXiv ID:** 2601.20578 | [PDF](https://arxiv.org/pdf/2601.20578v1)

**作者:** Dimitris Michailidis `[一作]` (University of Amsterdam), Fernando P. Santos `[通讯]` (University of Amsterdam)

**通讯引用:** 15408 | [OpenAlex ID](https://openalex.org/A5073403497)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

使用多智能体强化学习（Q‑learning）在 Braess 与阿姆斯特丹地铁网络的拥堵博弈中，研究学习率差异对效率与公平的影响。

**💡 创新点**

首次引入“学习价格”（Price of Learning）以及源节点公平度量，揭示即便整体效率良好，学习率不均仍会产生持久的不公平与效率低下。

**🔧 技术方法**

核心技术为无模型的 Q‑learning、ε‑贪婪探索、价格与公平度量分析，结合博弈论的纳什均衡与社会最优概念。

**📊 数据集**

使用人工构造的 Braess 两源网络与抽象化的阿姆斯特丹地铁网络（基于真实线路与时间表），不涉及实际出行数据。

**📈 对比分析**

通过与纳什均衡、社会最优以及不同学习率设定的对照实验，发现尽管价格最终趋于 1，学习率不均会导致显著的源间差距和长时间的效率浪费。

**⚠️ 局限性**

局限在于仅考虑两类学习速率的群体、使用单一 Q‑learning 算法、网络模型简化且缺乏真实通勤者的行为与数据验证。

---

## 321. MeCo: Enhancing LLM-Empowered Multi-Robot Collaboration via Similar Task Memoization

**arXiv ID:** 2601.20577 | [PDF](https://arxiv.org/pdf/2601.20577v1)

**作者:** Baiqing Wang `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Harbin Engineering University and Northwestern Polytechnical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出MeCo框架，利用任务级相似性实现多机器人协作规划。

**💡 创新点**

设计基于工作空间重叠的相似性检验方法、S‑Planner相似运动规划器和连续规划模块，并实现任务缓存的选择性存储与去重。

**🔧 技术方法**

结合大型语言模型（LLM）生成高层规划、基于相似任务的轨迹参考、RRT与IK校验以及缓存LFU策略。

**📊 数据集**

扩展RoCoBench为MeCoBench，对六个桌面多机器人任务进行相似性实验。

**📈 对比分析**

与RoCo、Central Plan、HMAS‑2、ReAct四个基线在三种相似度场景下对比，MeCo在成功率提升约30%，规划时间约55%缩短，token消耗约70%降低。

**⚠️ 局限性**

对高工作空间重叠任务仍受碰撞风险影响，缓存规模和相似阈值需手动调优，极端相似度低时提升有限。

---

## 322. Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands

**arXiv ID:** 2601.20555 | [PDF](https://arxiv.org/pdf/2601.20555v1)

**作者:** Wadhah Zai El Amri `[一作]` (Leibniz Universitaet Hannover), Nicolás Navarro-Guerrero `[通讯]` (Leibniz Universitaet Hannover)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在机器人手上布置七个低成本压电麦克风，并使用Audio Spectrogram Transformer对振动信号进行建模，实现了对外部触碰位置的高精度定位（静态误差<5mm）以及轨迹跟踪。

**💡 创新点**

创新点在于将结构传导声学与深度学习结合，既在冲击响应定位上实现毫米级精度，又在滑动轨迹跟踪中克服机器人自身运动噪声，且首次系统性对材料刚度与纹理对两类任务的影响进行了量化。

**🔧 技术方法**

采用的技术包括压电触摸麦克风阵列、20kHz下采样+STFT谱图、Audio Spectrogram Transformer模型（7通道输入），以及对噪声的预估与扣除。

**📊 数据集**

使用的数据集包括约6.5万条冲击样本（不同材质的软塑、硬塑、木、金属）和约3.6万条滑动轨迹样本（四种材质，约345类绘画轨迹），全部公开。

**📈 对比分析**

与传统触摸皮肤和可视触摸方法相比，本文方法在静态定位上误差约3.5–5.8mm，轨迹跟踪在静止机器人时误差≤4mm，动态时≤12mm，显著低于现有声学或视觉定位方案，且成本更低。

**⚠️ 局限性**

局限包括在机器人运动时误差上升、对某些材质（如硬塑）在背面定位不稳定、未能与视觉信息融合以及对极端噪声环境的鲁棒性尚待提升。

---

## 323. IoT Device Identification with Machine Learning: Common Pitfalls and Best Practices

**arXiv ID:** 2601.20548 | [PDF](https://arxiv.org/pdf/2601.20548v1)

**作者:** Kahraman Kostas `[一作]`, Rabia Yasa Kostas `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过系统化评估IoT设备识别的四大流程（方法选择、数据准备、特征提取、模型评估），指出现有研究常见错误并给出改进建议，形成一套完整的最佳实践指南。

**💡 创新点**

创新点在于：①从识别粒度出发阐释特征与模型的匹配关系；②系统剖析数据标签、增强与泄漏风险；③提出特征去标识化的标准化流程；④推荐可模块化的One‑vs‑Rest架构和宏F1评价方法，显著提升模型可扩展性与公平性。

**🔧 技术方法**

主要技术包括：传统机器学习算法（决策树、SVM、Logistic回归）、轻量化深度学习（CNN/Transformer）对比；特征工程利用包层级特征、统计时序特征；数据预处理采用tshark快速解析、IP/MAC/校验和去除；评估使用宏平均F1、精确率/召回率、混淆矩阵可视化。

**📊 数据集**

实验以Aalto University IoT数据集为核心，补充UNSW IoT、Kaggle/UCI公开数据集，覆盖多种协议（IP、ZigBee、Z‑Wave）与设备类型（摄像头、传感器、插座）。

**📈 对比分析**

通过在统一预处理和特征提取条件下，分别训练决策树、SVM、随机森林、神经网络，并用宏F1评估。结果显示决策树和随机森林在宏F1上优于深度网络（提升约5–10%），且推理延迟低于k‑NN/SVM，满足实时安全需求。

**⚠️ 局限性**

局限性在于：①仍主要依赖公开实验室数据，缺乏真实部署环境的多样化噪声与攻击样本；②对极端稀缺设备的泛化能力尚未充分验证；③去标识化过程中对协议细粒度特征的丢失可能影响某些高精度识别任务。

---

## 324. Beyond Divergent Creativity: A Human-Based Evaluation of Creativity in Large Language Models

**arXiv ID:** 2601.20546 | [PDF](https://arxiv.org/pdf/2601.20546v1)

**作者:** Kumiko Nakajima `[一作]` (University of Amsterdam), Sandro Pezzelle `[通讯]` (University of Amsterdam)

**通讯引用:** 833 | [OpenAlex ID](https://openalex.org/A5007142536)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了传统 Divergent Association Task（DAT）在衡量大语言模型（LLM）创造力方面的有效性，并提出了基于上下文适应的 Conditional Divergent Association Task（CDAT）来同时衡量新颖性和适宜性。

**💡 创新点**

创新点在于将创造力定义为在保持对给定线索的语义关联性的前提下产生多样化词汇，并通过“适宜性门控”过滤噪声，构造了既客观又与人类认知理论相契合的双维度评估框架。

**🔧 技术方法**

采用了 SBERT 词向量进行语义相似度计算，使用词汇语义距离来量化新颖性和适宜性，并通过统计检验和 Pareto 前沿分析揭示模型族的创造力分布。

**📊 数据集**

数据集为从 Brown 语料库中提取的 539 个名词线索，配合从 WordNet 随机抽取的词汇或 GPT‑4.1 生成的高关联词汇，构成了评估词列表。

**📈 对比分析**

与两类基准（随机采样与常见关联词）对比，CDAT 发现较小、低延迟优化的模型在适宜性门控通过后往往拥有更高的平均新颖性；高级模型则在适宜性更高但新颖性略低；DAT 的得分被随机和“作弊”基准超越，显示其无效性。

**⚠️ 局限性**

局限包括人类对照样本有限、线索词选择可能引入偏差、评估仅涵盖单词级别的新颖性与适宜性，且未考虑更复杂语境、跨模态或多领域创意任务。

---

## 325. Interpreting Emergent Extreme Events in Multi-Agent Systems

**arXiv ID:** 2601.20538 | [PDF](https://arxiv.org/pdf/2601.20538v1)

**作者:** Ling Tang `[一作]` (Shanghai Artificial Intelligence Laboratory), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了用于解释大语言模型驱动的多智能体系统中出现的极端事件（黑天鹅事件）的完整框架，基于 Shapley 值对每个代理动作进行归因，并在时间、代理、行为三维上聚合归因，构建一组量化指标来解释事件的起因、驱动者和行为特征。

**💡 创新点**

创新点：
- 首个专门解释多智能体系统极端事件的框架。
- 采用 Shapley 值提供“无先验、无偏”的真实归因，解决了传统基于统计或提示的归因方法在极端尾部表现不佳的问题。
- 设计五个解释指标（相对风险时滞 L_tm、代理风险集中 G_ag、风险-不稳定性相关 C_ag、代理风险同步 Z_ag、行为风险集中 G_be）实现从“何时”“谁”和“何种行为”三维的全景解释。

**🔧 技术方法**

技术：
- 以 Shapley 值为核心的归因方法，利用 Monte Carlo 采样实现可扩展的近似计算。
- 对多智能体系统的轨迹进行因果重演（保留部分动作，替换其余动作为安全动作）来评估子集贡献。
- 通过聚合归因得到时间/代理/行为维度的贡献，并基于 Gini、Pearson、Vicsek 模型等统计工具构建解释指标。

**📊 数据集**

数据集/实验环境：
- 经济系统场景 EconAgent（10 代理，平均轨迹长度≈34）
- 金融市场场景 TwinMarket（10 代理，平均轨迹长度≈27）
- 社交网络场景 SocialNetwork（20 代理，平均轨迹长度≈21）
- 采用多种主流大语言模型（GPT‑4o mini、Llama‑3.1‑8B、Claude‑3‑Haiku、Qwen‑Plus、DeepSeek‑V3.2）生成代理动作。

**📈 对比分析**

对比方法：Random、Failure Taxonomy (FT)、Failure Attribution (FA)、Agent Tracer (AT)。
- 评价指标为“风险下降率”——在删除归因最高的 3/10 个动作后，极端事件风险下降的比例。
- 在所有实验设置中，本方法均获得最高的风险下降率，表明归因结果最为可信，优于基线方法。

**⚠️ 局限性**

局限性：
- Shapley 值的计算本质为 NP‑hard，虽通过 Monte Carlo 近似但对大规模系统（数千动作）仍存在计算瓶颈。
- 归因依赖于对安全动作的假设，若安全动作定义不准确可能影响结果。
- 当前实验仅覆盖三类模拟场景，未来需在更真实、更大规模的多智能体环境中验证。

---

## 326. DiffVC-RT: Towards Practical Real-Time Diffusion-based Perceptual Neural Video Compression

**arXiv ID:** 2601.20564 | [PDF](https://arxiv.org/pdf/2601.20564v1)

**作者:** Wenzhuo Ma `[一作]` (Wuhan University), Zhenzhong Chen `[通讯]` (Wuhan University)

**通讯引用:** 8334 | [OpenAlex ID](https://openalex.org/A5006748765)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了DiffVC-RT，一套实时扩散式感知视频压缩框架，能够在1080p编码和720p解码时实现 >30fps。

**💡 创新点**

创新点包括：① 高效信息架构（PixelUnshuffle+通道扩展、VAE Encoder 去除、轻量化 U‑Net/VAE Decoder），② 零成本显式/隐式一致性建模（在线时序Shift + 像素/特征光流惩罚），③ 异步并行解码管线（Batch‑dim Shift + BF16/FP16 混合精度）。

**🔧 技术方法**

主要技术手段有：PixelUnshuffle、通道扩展、Pruned U‑Net与VAE Decoder、在线时序Shift模块、光流估计（RAFT‑lite）、特征拼接、混合 BF16/FP16 计算、Batch‑dim Shift 并行解码。

**📊 数据集**

使用 Vimeo‑90k 进行训练，评估数据集为 HEVC、UVG 与 MCL‑JCV，覆盖多种分辨率与场景。

**📈 对比分析**

与传统编码器（VTM/HM）、基准 NVC（DCVC 系列）、GAN 及前沿扩散 NVC（DiffVC/OSD）进行对比；在 LPIPS、FloLPIPS 上显著优于所有基线，解码速率达到 206 / 30 fps，且在 720p 真实时间下实现。

**⚠️ 局限性**

限制包括：仍需高算力 GPU；在 1080p 编码时速度受限；存在 N‑1 帧的解码延迟；对光流估计错误敏感；在极低比特率下可能出现纹理失真。

---

## 327. Investigating the Development of Task-Oriented Communication in Vision-Language Models

**arXiv ID:** 2601.20641 | [PDF](https://arxiv.org/pdf/2601.20641v1)

**作者:** Boaz Carmeli `[一作]` (Technion Israel Institute of Technology), Ron Meir `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究预训练视觉语言模型在零样本提示下，如何在指代游戏中自行产生并使用任务导向的通信协议（更高效或更隐蔽的语言）。

**💡 创新点**

创新点在于证明LLM能够在不进行微调的情况下生成新词或符号，形成比自然语言更高效或对人类不可解的协议，并展示同构模型可自发协同理解这些隐蔽协议。

**🔧 技术方法**

采用零样本提示、参考游戏框架进行实验，并通过游戏成功率、描述长度、词汇新颖度等指标评估自然语言与自定义语言的差异。

**📊 数据集**

使用的视觉数据集包括Flags（真实与合成国旗）、COCO以及CLEVR。

**📈 对比分析**

对比方法是把自然语言、效率导向和隐蔽导向三种语言在同一指代游戏中进行评估；实验显示效率语言可在准确率上超过自然语言（如GPT‑4o在Flags上达到0.89 vs 0.79），隐蔽语言同架构模型对接准确率高、外部观察者表现低，最高准确率可达0.97。

**⚠️ 局限性**

局限性包括：实验仅在最多10张候选图的简化环境下、未进行微调、仅评估词汇新颖度与准确率，未检验语义完整性、对更大规模、不同语言或多模型场景的可推广性。

---

## 328. DRAINCODE: Stealthy Energy Consumption Attacks on Retrieval-Augmented Code Generation via Context Poisoning

**arXiv ID:** 2601.20615 | [PDF](https://arxiv.org/pdf/2601.20615v1)

**作者:** Yanlin Wang `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33227 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对检索增强代码生成系统的能耗攻击，利用检索语料库中的触发器诱导模型生成冗长输出，从而显著提升GPU延迟和能耗。

**💡 创新点**

创新点在于：①在检索阶段进行上下文投毒，攻击无需直接修改用户查询；②采用基于梯度的多位置突变与攻击缓冲池加速触发器优化；③引入EOS损失与多样性损失以及KL约束，实现语义不变但输出长度显著增长的隐蔽攻击。

**🔧 技术方法**

技术主要包括：检索增强生成（RAG）框架、梯度驱动的触发器突变、EOS与多样性损失函数、KL相对熵约束、攻击缓冲池以及多位置突变。

**📊 数据集**

使用RepoEval和Odex两大代码完成基准数据集作为评测语料库，并在其对应的代码检索语料库中注入毒样本。

**📈 对比分析**

与RawRAG、Prompt Injection、LLMEffiChecker等基线对比，攻击在保持95–99%功能正确率的前提下，输出长度提升3–10倍，延迟增加约85%，能耗提升约49%，比最强基线提升25–32%。在黑盒迁移下仍能使延迟、能耗提升70%以上。

**⚠️ 局限性**

限制包括：需要对检索语料库的访问与梯度信息，攻击对模型参数的依赖在黑盒下效果下降；检测方法（SVM、Perplexity、CodeBERT）对注入的毒样本识别率低；实验仅覆盖两种公开模型与两套基准，可能不适用于更大或专有模型。

---

## 329. ACFormer: Mitigating Non-linearity with Auto Convolutional Encoder for Time Series Forecasting

**arXiv ID:** 2601.20611 | [PDF](https://arxiv.org/pdf/2601.20611v1)

**作者:** Gawon Lee `[一作]` (Pusan National University), Hyerim Bae `[通讯]` (Pusan National University)

**通讯引用:** 1577 | [OpenAlex ID](https://openalex.org/A5047158713)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在时间序列预测领域提出ACFormer架构，通过融合卷积自编码器与线性注意力机制，提升对非线性高频成分的捕捉能力。

**💡 创新点**

核心创新在于①引入individual receptive field分析揭示卷积层可识别“pivot channels”，②设计共享压缩与独立扩展的Auto‑Convolution Encoder，兼顾线性效率与卷积非线性特征提取。

**🔧 技术方法**

采用卷积自编码器、共享压缩/独立扩展卷积、时间门控注意力、RevIN归一化、全局卷积大核等技术。

**📊 数据集**

在六大工业基准数据集上评测：ECL、ETTh1/ETTh2、ETTm1/ETTm2、Solar、Traffic、Weather、PEMS等。

**📈 对比分析**

与Transformer、线性、CNN、状态空间等SOTA模型在MAE/MSE上进行基准对比，ACFormer在多数数据集上实现SOTA，显著低于iTransformer且计算量更小。

**⚠️ 局限性**

仍存在对极端非周期性或稀疏序列的泛化受限，缺乏多尺度自适应融合，对极长序列的可扩展性需要进一步验证。

---

## 330. StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval

**arXiv ID:** 2601.20597 | [PDF](https://arxiv.org/pdf/2601.20597v1)

**作者:** Shaokun Wang `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Shandong University)

**通讯引用:** 28534 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种用于连续文本-视频检索（CTVR）的结构化跨模态对齐框架StructAlign，旨在缓解多模态持续学习中的特征漂移和模态失配问题。

**💡 创新点**

创新点主要在于引入简单等角紧框架（Simplex ETF）作为几何先验，设计跨模态ETF对齐损失以保持类别级别的等角分布，并引入跨模态关系保持（CRP）损失利用互补模态保持相似性关系，从而同时抑制跨模态与单模态的特征漂移。

**🔧 技术方法**

技术手段包括冻结CLIP预训练的文本与视觉编码器，使用轻量级的Mixture-of-Experts (MoE) 与LoRA模块进行参数高效微调；构造跨模态ETF对齐损失、CRP损失以及对比学习检索损失；并通过ETF几何约束实现类别间等角分布与类别内聚集。

**📊 数据集**

实验在两个主流视频-语言基准数据集MSRVTT和ACTNET上进行，按照任务划分实现10/20个增量任务的连续学习设置。

**📈 对比分析**

与现有CTVR方法StableFusion以及经典持续学习方法LwF、VR-LwF、ZSCL、MoE-Adapter等相比，StructAlign在Recall@1/5/10、Median Rank、Mean Rank和Backward Forgetting指标上均实现了显著提升，并在参数效率上仅需约33.9M可训练参数即达成最高平均Recall@1。

**⚠️ 局限性**

局限性主要体现在：①与理论上最优的Upper Bound仍存在一定差距；②目前仅验证于文本-视频两模态的检索任务，尚未扩展到更开放或多模态的持续学习场景；③对超参数（λ1、λ2）敏感，需要进一步探索自动调优机制。

---

## 331. Harnessing Large Language Models for Precision Querying and Retrieval-Augmented Knowledge Extraction in Clinical Data Science

**arXiv ID:** 2601.20674 | [PDF](https://arxiv.org/pdf/2601.20674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 332. AutoOverlap: Enabling Fine-Grained Overlap of Computation and Communication with Chunk-Based Scheduling

**arXiv ID:** 2601.20595 | [PDF](https://arxiv.org/pdf/2601.20595v1)

**作者:** Xinwei Qiang `[一作]` (University of California), Adnan Aziz `[通讯]` (Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了名为Syncopate的编译器与运行时，能够将单GPU Triton核自动转换为细粒度通信与计算重叠的分布式核；

**💡 创新点**

创新点在于提出通信块（chunk）抽象，将通信粒度与内核结构解耦；通过chunk调度与轻量注解实现内核级别的重叠，并支持多种通信后端与自动调优；

**🔧 技术方法**

技术细节包括基于Triton的源到源编译器、chunk‑based通信计划、内核 tile 调度重排、自动调优（后端选择、SM分配、chunk大小、tile顺序）以及与PyTorch分布式的无缝集成；

**📊 数据集**

使用的工作负载基于Llama‑3与Qwen模型的FFN与Attention层，涵盖不同隐藏维度、头数、序列长度的多GPU场景；

**📈 对比分析**

与手工优化基线（ThunderKittens、Triton‑Distributed、AsyncTP、Flux）以及自动编译器（Domino、Alpa、Mercury）对比，平均获得1.3×加速，最优可达4.7×，在8 GPU H100上几乎匹配最优手工实现；

**⚠️ 局限性**

局限性包括：主要在GPU集群内验证；对极不规则或小规模任务的优势不明显；自动调优过程需要额外时间；对内存占用与可扩展性未做深入讨论。

---

## 333. A Computational Approach to Language Contact -- A Case Study of Persian

**arXiv ID:** 2601.20592 | [PDF](https://arxiv.org/pdf/2601.20592v1)

**作者:** Ali Basirat `[一作]` (University of Copenhagen), Navid Baradaran Hemmati `[通讯]` (Certified Translation Agency)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在单语波斯语 BERT 训练模型的中间表示上做信息论探针和归因分析，探究语言接触对语言结构的隐式编码，发现语法类别对接触不敏感，而词形变化（如格和性别）受接触影响明显。

**💡 创新点**

创新点在于将变分可用信息（variational usable information）与 LAPE 归因方法相结合，首次系统评估单语模型在跨语言情境下对接触语言的中间表示的结构化反映，并揭示了形态学特征的选择性、结构受限的接触痕迹。

**🔧 技术方法**

使用技术包括：ParsBERT（12 层 Transformer）作为基模型；信息论探针（variational usable information）评估可用信息；LAPE 归因分析衡量信息分布；并结合层级可视化与统计比较。

**📊 数据集**

数据集为：ParsBERT 在 3.9 亿波斯语文本上预训练；交叉语言评估采用 Parallel Universal Dependencies (PUD) 8 种不同接触程度的语言（阿拉伯语、英语、法语、德语、印地语、日语、俄语、土耳其语）以及对应的 UD 注释。

**📈 对比分析**

比较方法是将不同语言的可用信息和 LAPE 分数与层级、形态学特征做对比；结果显示：语言识别信息显著但在高层趋于稳定；UPOS 信息几乎不随接触程度变化；格与性别信息在低接触/无接触语言（如英语、法语、日语）表现高，在高接触/复杂形态语言（如土耳其语、俄语、德语、印地语、阿拉伯语）表现低，体现出形态学受接触影响更大。

**⚠️ 局限性**

局限性包括：PUD 数据集缺失关键接触语言（阿塞拜疆语、土库曼语、乌尔都语、亚美尼亚语、库尔德语等），无法完整评估波斯语在不同区域和语言家族中的接触效果；模型仅基于单语预训练，无法捕捉更深层次的语法重构。

---

## 334. Energy Efficient Downlink mMIMO Using Dynamic Antenna and Power Adaptation

**arXiv ID:** 2601.20586 | [PDF](https://arxiv.org/pdf/2601.20586v1)

**作者:** Ravi Sharan B A G `[一作]` (Nokia Bell Labs), Silvio Mandelli `[通讯]` (Nokia Bell Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种基于动态天线和功率适配的联合方案，在mMIMO gNB下行链路中实现网络能耗与吞吐量的平衡。

**💡 创新点**

创新点在于先使用多CSI-RS框架进行天线适配，再结合POLITE链路自适应进行功率适配，并按时隙动态决策，充分利用UE缓冲区与宽带CQI实现功率谱密度降低而不影响吞吐量。

**🔧 技术方法**

使用技术包括多CSI-RS配置、POLITE链路自适应、PF调度、3GPP 38.901 UMa NLoS信道模型、OFDM、TDD、功率放大器效率模型等。

**📊 数据集**

采用3GPP系统级仿真，基于FTP3流量模型生成低/轻/中/高负载场景，没有使用公开数据集。

**📈 对比分析**

与固定天线（8/16/32 TRX）、动态天线、POLITE功率等基准进行比较；在低/轻负载下能耗降低35%-41%，吞吐量保持与基准相当；在中/高负载下能耗与吞吐量均优于传统方案。

**⚠️ 局限性**

主要限制在于对多CSI-RS框架下CSI获取的依赖，未考虑CSI估计误差、硬件延迟及真实系统验证。

---

## 335. GPO: Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control

**arXiv ID:** 2601.20668 | [PDF](https://arxiv.org/pdf/2601.20668v1)

**作者:** Shuhao Liao `[一作]` (Beihang University), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1229 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并验证了Growing Policy Optimization (GPO)，一种通过随时间扩展有效动作空间来改进强化学习训练的框架。

**💡 创新点**

创新点是将动作空间动态增长与PPO保持一致，理论证明梯度失真有限且能在早期提升信噪比、后期提升探索。

**🔧 技术方法**

使用PPO、动作平滑变换、仿真与真实硬件实验。

**📊 数据集**

在仿真四足和六足机器人（Unitree Go2）以及500个随机命令的评估集上训练。

**📈 对比分析**

与固定动作空间的PPO、DeCAP等基线比较，GPO在训练收敛速度快、最终奖励高、硬件鲁棒性提升，成功实现零射击 sim‑to‑real。

**⚠️ 局限性**

局限在于对动作范围的假设和对高维连续控制的依赖，可能在更复杂地形或更大模型中需要更细粒度的增长策略。

---

## 336. ALER: An Active Learning Hybrid System for Efficient Entity Resolution

**arXiv ID:** 2601.20664 | [PDF](https://arxiv.org/pdf/2601.20664v1)

**作者:** Dimitrios Karapiperis `[一作]` (International Hellenic University), Vassilios Verykios `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于冻结双编码器的分块主动学习系统ALER，用于高效解决实体解析任务。

**💡 创新点**

通过分块K-means划分候选集、冻结SBERT嵌入、轻量MLP迭代训练及混合查询策略，实现了在保持语义精准度的同时显著提升计算与内存效率。

**🔧 技术方法**

SBERT预训练模型、HNSW近似最近邻索引、K-means聚类、轻量Siamese MLP、混合置信/不确定采样及两阶段cascade匹配。

**📊 数据集**

9个标准实体解析基准（Abt-Buy, Amazon-Walmart, Amazon-Google, ACM-DBLP, Scholar-DBLP, Restaurants, IMDB-DBPEDIA, Voters, DBLP）。

**📈 对比分析**

与DIAL、AL-Risk、ERABQS三种主流主动学习基线对比，在F1、训练时长和解析延迟上均优，尤其在百万级数据上实现接近100% F1且训练/解析时间缩短至十几秒。

**⚠️ 局限性**

依赖预先冻结的SBERT嵌入，对极端语义变异或极大噪声的数据仍可能产生误匹配；同时在极端多模态或非常动态的流式场景下需进一步改进。

---

## 337. A Multi-Camera Optical Tag Neuronavigation and AR Augmentation Framework for Non-Invasive Brain Stimulation

**arXiv ID:** 2601.20663 | [PDF](https://arxiv.org/pdf/2601.20663v1)

**作者:** Xuyi Hu `[一作]` (University of Cambridge), Stephan Goetz `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于多摄像头、可打印光学标签的低成本脑刺激导航系统，并将其与AR可视化结合，实现实时导向TMS电极定位；

**💡 创新点**

创新点在于：①利用低成本消费级摄像头和AprilTag光学标记实现5 mm以内定位精度；②将实时定位数据通过Unity+AR Foundation投影到患者头部，提供沉浸式即时反馈；③通过多摄像头融合和高斯加权估计降低单摄像头误差；

**🔧 技术方法**

采用的技术包括：计算机视觉光学标记检测与姿态估计、三摄像头同步校准、多摄像头融合算法、Unity3D脑模型渲染、AR Foundation实时渲染、TCP传输数据；

**📊 数据集**

数据集：实验使用10名医学生/医生（无TMS经验），每人完成多目标定位任务；此外在精度测试中采集100帧随机点、15个定位点，使用标定板进行摄像头内参外参校准；

**📈 对比分析**

与Vuforia+HoloLens1、HoloLens辅助脑室切口、Intel RealSense SR300三种现有方法对比，平均定位误差分别为4.94 mm、5.2 mm和20 mm；本系统在4.94 mm附近，与Vuforia相近，明显优于RealSense；整体表现稳定，重现性好；

**⚠️ 局限性**

局限性包括：依赖清晰光学标记，光照变化或快速运动会影响检测；多摄像头校准误差会累积；部分遮挡仍可能导致定位不稳；未来需加入深度传感或滤波提升鲁棒性。

---

## 338. Lila: Decentralized Build Reproducibility Monitoring for the Functional Package Management Model

**arXiv ID:** 2601.20662 | [PDF](https://arxiv.org/pdf/2601.20662v1)

**作者:** Julien Malka `[一作]` (Telecom Paris), Arnout Engelen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 Lila 系统，实现了在功能包管理器 Nix 环境下去中心化的构建可复现性监测与报告聚合。

**💡 创新点**

创新点在于：① 通过 Nix 的 post-build hook 自动生成签名的可复现性证明；② 将构建报告分布式收集到聚合服务器，消除单点构建；③ 提供可视化仪表盘，支持历史回溯与自动检测回归。

**🔧 技术方法**

使用了 Nix 功能式包管理、post-build hook、加密签名、REST API、数据库、Web UI、二进制缓存等技术。

**📊 数据集**

采集了 Nixpkgs（约 80,000+ 包）及其 CI 构建的可复现性报告，已累计超过 150,000 条报告，并与 NixOS CI 构建流水线无缝对接。

**📈 对比分析**

与传统中心化监测（如 Debian、Arch）相比，Lila 在实验部署中收集的报告量更大、扩展性更好，未发现显著性能瓶颈；能覆盖整个 Nix 生态并实现 90%+ 的可复现率验证。

**⚠️ 局限性**

局限性包括：仍依赖中心化聚合服务器，真正的点对点聚合尚未实现；需要第三方提供构建资源；对构建环境元数据的采集与分析仍有提升空间。

---

## 339. ProSkill: Segment-Level Skill Assessment in Procedural Videos

**arXiv ID:** 2601.20661 | [PDF](https://arxiv.org/pdf/2601.20661v1)

**作者:** Michele Mazzamuto `[一作]` (University of Catania), Antonino Furnari `[通讯]` (University of Catania)

**通讯引用:** 3075 | [OpenAlex ID](https://openalex.org/A5089549062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并发布了ProSkill数据集，提供多域、分段级别的动作技能评估标注；

**💡 创新点**

创新点在于引入可扩展的瑞士赛制 + ELO评分的标注流程，既实现了对比标注又能生成全局绝对分数；

**🔧 技术方法**

采用了基于视频特征的三种全局排名方法（USDL、DAE‑AQA、CoFInAl）和三种对比排名方法（RAAN、AQA‑TPT、CoRe），并使用I3D或VideoMAE提取特征；

**📊 数据集**

整合来自EgoExo4D、Meccano、EpicTent、IkeaASM、Assembly101等公开数据集，总计1135段、71个动作；

**📈 对比分析**

在全局排名上，CoFInAl和USDL达到最高Spearmanρ≈0.6，但整体仍偏低；对比排名中AQA‑TPT+VideoMAE在EgoExo4D上最高准确率0.79，Assembly101仍靠近随机；

**⚠️ 局限性**

限制在于任务多样性与数据噪声导致模型泛化差，标注成本高，且目前仅依赖视觉信息，缺乏多模态与更细粒度的评估指标。

---

## 340. Supply Chain Insecurity: Exposing Vulnerabilities in iOS Dependency Management Systems

**arXiv ID:** 2601.20638 | [PDF](https://arxiv.org/pdf/2601.20638v1)

**作者:** David Schmidt `[一作]` (University of Vienna), Edgar Weippl `[通讯]` (University of Vienna)

**通讯引用:** 6451 | [OpenAlex ID](https://openalex.org/A5083435816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究iOS应用依赖管理系统的供应链攻击，量化其脆弱性并演示依赖混淆和劫持攻击的可行性。

**💡 创新点**

首次在iOS生态中系统性评估依赖混淆与劫持风险，揭示应用包泄露导致远程代码执行，并对多种依赖管理系统提出防御方案。

**🔧 技术方法**

利用逆向分析提取框架名与版本，结合CocoaPods、GitHub API、域名查询和Python脚本进行漏洞扫描与PoC演示。

**📊 数据集**

使用9,212个iOS应用与对应Android下载量数据集、279个具披露计划的iOS应用以及公开GitHub项目与依赖文件。

**📈 对比分析**

将iOS生态与Cargo、Go模块、Maven、npm、pip等五大生态对比，评估其身份认证、代码执行与URL保留等安全特性，发现大多数系统仍存在相似漏洞。

**⚠️ 局限性**

仅检测框架目录中的内部依赖，遗漏未使用资源或直接链接到二进制的库，未覆盖过时或恶意包检测，未来需纵向研究和开发者视角调研。

---

## 341. An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems

**arXiv ID:** 2601.20637 | [PDF](https://arxiv.org/pdf/2601.20637v1)

**作者:** Panayiotis Ioannou `[一作]` (University of Cambridge), Pietro Cicuta `[通讯]` (University of Cambridge)

**通讯引用:** 8903 | [OpenAlex ID](https://openalex.org/A5045269326)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估神经常微分方程（NODE）在噪声合成数据下的外推能力，并探讨使用NODE生成的数据辅助符号回归（SR）恢复系统的微分方程。

**💡 创新点**

提出将NODE作为数据增强与去噪工具，配合符号回归在稀缺噪声数据中恢复物理定律的端到端方法，并证明NODE可在动态相似条件下实现有效外推。

**🔧 技术方法**

神经常微分方程（NODE）、符号回归（SR）、JAX、DEAP等。

**📊 数据集**

合成噪声数据，来自两组阻尼振荡系统：无摩擦的倒立摆（cart‑pole）和描述细菌适应的生物模型（bio‑model）。

**📈 对比分析**

将SR在原始噪声数据与NODE生成的完整数据上分别训练；在无噪声下，SR可完整恢复三条方程；在5%噪声下，仅恢复两条方程；使用NODE生成的数据可提升SR对噪声的鲁棒性，取得约5%误差内的结果。

**⚠️ 局限性**

受限于数据仅为合成、噪声水平高、训练集未覆盖所有动态，且SR对低信噪比项敏感，难以完全恢复复杂方程；未来需扩展多条件数据、改进NODE结构或加入物理先验。

---

## 342. CLEAR-Mamba:Towards Accurate, Adaptive and Trustworthy Multi-Sequence Ophthalmic Angiography Classification

**arXiv ID:** 2601.20601 | [PDF](https://arxiv.org/pdf/2601.20601v1)

**作者:** Zhuonan Wang `[一作]` (Zhejiang University), Beng Chin Ooi `[通讯]` (Zhejiang University)

**通讯引用:** 21295 | [OpenAlex ID](https://openalex.org/A5024892041)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种专为单模态多序列眼底血管造影分类的框架CLEAR-Mamba，并构建了覆盖43种疾病的大规模FFA/ICGA数据集。

**💡 创新点**

创新点包括：1) 通过Hyper‑adaptive Conditioning（HaC）实现轻量级实例自适应调制；2) 采用可靠性感知预测（RaP）头基于证据学习提供校准的置信度与不确定性；3) 在MedMamba高效的状态空间网络上进行时空特征建模。

**🔧 技术方法**

核心技术包括MedMamba视觉状态空间模型、FiLM/低秩适配器的HyperNetwork、Evidential Deep Learning（Dirichlet分布）以及自适应证据权重。

**📊 数据集**

使用了自建的多序列FFA/ICGA数据集（15,524张图像、43类疾病），以及公开的RetinaMNIST、OCT‑C8和Harvard‑GDP三大基准。

**📈 对比分析**

与CNN、ViT、Mamba及医学专用模型进行对比，CLEAR在自建数据集上OA、F1、AUC均提升4–8个百分点；在公开基准上亦实现或接近最优性能，显示出更好的泛化与校准。

**⚠️ 局限性**

局限性：① 仅针对单模态；② 需要大量标注且对少数类别仍易受样本不平衡影响；③ 证据学习对标签噪声敏感，未来需进一步完善跨中心验证与多模态融合。

---

## 343. Shortest LCD embeddings of binary, ternary and quaternary linear codes

**arXiv ID:** 2601.20600 | [PDF](https://arxiv.org/pdf/2601.20600v1)

**作者:** Junmin An `[一作]` (Sogang University), Haeun Lim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种将任意二/三/四元线性码嵌入最短LCD码的方法，并利用该方法构造了多组新的最优LCD码。

**💡 创新点**

创新点在于证明：对任意[ n , k ]码，其最短LCD嵌入长度为 n+ℓ（ℓ 为码的hull维数），并给出了构造此类嵌入的通用矩阵形式；随后将该理论应用于Hamming、Reed–Muller等经典码，得到多条未曾出现的最优LCD码。

**🔧 技术方法**

核心技术为：线性代数中的hull维数计算、矩阵秩与可逆性分析、以及在码生成矩阵中附加可逆矩阵与任意矩阵的构造；同时利用了BKLC库中已知最优码的参数进行嵌入。

**📊 数据集**

使用 Magma 的 BKLC（Best Known Linear Codes）库作为基准码集合，涵盖了二/三/四元的[19,4,12]、[19,5,11]、[20,6,10]、[20,5,12]等已知最优码作为起点进行嵌入。

**📈 对比分析**

与已知最优LCD码（包括BKLC库中的参数）对比，新构造的码在距离上均提升了1点（例如[23,4,14] vs [23,4,13]），并且在某些参数下（如[16,10,4]、[44,36,4]）实现了目前已知的最佳距离，证明了该方法在性能上的有效性。

**⚠️ 局限性**

局限性包括：目前仅在 q=2,3,4 的情形下给出完整证明与构造；对更大 q 的推广尚未完成；此外，在高维码的实际计算与码表生成上仍存在计算复杂度上升的问题。

---

## 344. Person Re-ID in 2025: Supervised, Self-Supervised, and Language-Aligned. What Works?

**arXiv ID:** 2601.20598 | [PDF](https://arxiv.org/pdf/2601.20598v1)

**作者:** Lakshman Balasubramanian `[一作]` `[通讯]` (MoiiAi), Lakshman Balasubramanian (MoiiAi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对不同训练范式（监督、无监督、语言对齐）在 9 个多域人重识别数据集上进行系统评估。

**💡 创新点**

首次全面比较 11 个模型在跨域任务中的性能，揭示模型规模与泛化能力无正相关，并指出混合训练策略可提升鲁棒性。

**🔧 技术方法**

使用监督分类与 Triplet 损失、CLIP/SigLIP2 语言对齐预训练、DINOv2 自监督等技术，并进行微调与零射训练。

**📊 数据集**

覆盖 MSMT17、Market‑1501、DukeMTMC‑reID、CUHK03、GRID、CelebReID、PKU‑ReID、LasT、IUSReID 等 9 个公开数据集。

**📈 对比分析**

通过 mAP、Rank‑k 等指标对比，发现监督模型在域内表现优异但跨域崩溃；语言对齐模型跨域稳健；混合模型（如 CLIP‑ReID）在大多数数据集上取得最高分。

**⚠️ 局限性**

仍缺乏能同时在域内高精度与跨域鲁棒的统一模型；语言对齐模型整体精度偏低；模型规模不必然带来更好泛化；实际部署还面临隐私、效率与长时间跟踪等挑战。

---

## 345. Dependable Connectivity for Industrial Wireless Communication Networks

**arXiv ID:** 2601.20580 | [PDF](https://arxiv.org/pdf/2601.20580v1)

**作者:** Nurul Huda Mahmood `[一作]` (University of Oulu), Matti Latva-aho `[通讯]` (University of Oulu)

**通讯引用:** 12294 | [OpenAlex ID](https://openalex.org/A5035872239)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文构建了工业无线通信网络（IWCN）的可靠性理论框架，并通过可预测唤醒协议和 TSN 技术等案例，展示了实现可依赖通信的实践方案。

**💡 创新点**

创新点在于：① 将系统可靠性理论与工业场景的多址与实时传输技术结合，形成统一的可靠性评估方法；② 提出了基于空间相关性和能量预测的智能唤醒策略，显著提升事件检测概率；③ 讨论了 5G 与 TSN 的融合路径，为无线端实现端到端确定性传输提供了新思路。

**🔧 技术方法**

主要技术包括：自适应多址调度、实时网络/信道监测、边缘机器学习、可预测唤醒（Wake‑up Radio）、TSN 流控制与 FRER、以及 5G/Wi‑Fi 7 中的时间同步机制。

**📊 数据集**

实验基于仿真生成的能量采集设备与事件触发数据，未使用公开数据集；通过仿真场景验证了唤醒策略的性能。

**📈 对比分析**

与传统基于空间相关性的待机唤醒以及简单占空比周期调度相比，智能唤醒方案在 5 ms 内接收足够事件信息的概率提升了约三位数；TSN‑over‑wireless 的实验表明，通过多链路与 FRER 可实现低于 1 ms 的时延保证，但在高负载下仍会出现包突发导致可靠性下降。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境验证，缺乏实际工业现场的实验支持；② 能量与组唤醒大小权衡尚未系统化，导致长期可用性需进一步研究；③ 对高动态网络（如大规模移动设备）下的同步与调度算法仍需优化。

---

## 346. Efficient Multimodal Planning Agent for Visual Question-Answering

**arXiv ID:** 2601.20676 | [PDF](https://arxiv.org/pdf/2601.20676v1)

**作者:** Zhuo Chen `[一作]` (ShanghaiTech University), Kewei Tu `[通讯]` (ShanghaiTech University)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5102274699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多模态规划代理（Multimodal Planning Agent），通过动态决定是否执行检索增强生成（mRAG）流程中的各个步骤，优化视觉问答（VQA）任务的推理效率与准确率。

**💡 创新点**

创新点在于：①将检索与生成流程拆分为可选择的子路径，避免了传统固定多阶段管道的冗余；②利用自动标注的查询拆解数据训练代理，使其在推理时能判断是否需要图像检索、文本检索或两者都需要；③实现了单次调用代理即可完成决策，显著降低工具调用次数与延迟。

**🔧 技术方法**

技术方法包括：①多模态检索增强生成（image retrieval、text retrieval、query rewriting）；②LLM prompt-based 判断与生成；③对代理模型进行 LoRA 微调（rank 32）或全参数微调；④使用 Qwen2.5-VL-7B-Inst 作为基础模型，Qwen-Max 等模型用于评估。

**📊 数据集**

训练数据：InfoSeek、VQAv2.0、WanWu（中文 VQA）；测试数据：Life VQA、Private VQA、Dyn‑VQA（中、英）、NoCaps、Visual7W、Mix。

**📈 对比分析**

与 WebWatcher（Deep Research Agent）和 OmniSearch（prompt-based）进行对比。实验显示：在 Mix 数据集上，代理方法在保持或提升 VQA 准确率（相较于 +k_i,t 的基线）同时，搜索时间下降 60%+，工具调用延迟为 WebWatcher 的 1/3~1/4，整体推理效率显著提升。

**⚠️ 局限性**

局限性：①代理在极简场景下仍需多轮工具调用；②对大模型（如 GPT‑4o）迁移需额外评估；③部分复杂检索需求可能因缺乏足够上下文而误判；④当前仅验证在公开数据集上，真实场景多样性尚待进一步验证。

---

## 347. Overview of the TREC 2025 Tip-of-the-Tongue track

**arXiv ID:** 2601.20671 | [PDF](https://arxiv.org/pdf/2601.20671v1)

**作者:** Jaime Arguello `[一作]` (University of North Carolina), Bhaskar Mitra `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在TREC 2025中组织了一项Tip‑of‑the‑tongue (ToT)检索任务，扩展了以往只聚焦特定领域的做法，覆盖了53类实体，并收集了MS‑ToT、人工提问和LLM合成的多种查询。

**💡 创新点**

创新点在于：①将ToT检索任务推广到通用领域并引入多样化查询来源；②通过两款LLM（Llama‑3.1‑8B‑Instruct 与 GPT‑4o）生成大规模合成查询，增强评测覆盖面；③对不同查询类型（MS‑ToT、人工、合成）进行细粒度性能对比。

**🔧 技术方法**

主要技术包括传统BM25检索（Anserini、PyTerrier）、稠密检索（Lightning‑IR、BERT‑based embeddings）、以及多种重排序模型（LM‑ART、LAMBDA‑MARt 等）。

**📊 数据集**

使用的数据集有：6,407,814 条来自 2023 年 Wikipedia 的文章（包含答案），143 条 MS‑ToT 训练查询，3 组开发集（含 172 条 MS‑ToT、150 条人工、300 条合成），622 条测试查询。

**📈 对比分析**

评估指标采用 NDCG@1000（官方）、NDCG@10、Recall@1000、MRR 等，最优系统（pyterrier‑bm25）取得 NDCG@1000=0.6824；整体来看系统在 MS‑ToT 查询上的表现优于人工和合成查询，但差距仍大；不同查询类型间存在高度相关性，表明评测设计有效。

**⚠️ 局限性**

局限性包括：ToT 查询本身极其复杂且多样化，导致即便是最强模型也难以突破 0.7 的 NDCG；部分实验者使用外部数据，缺乏训练集透明度；合成查询虽覆盖多领域，却可能与真实用户查询的语义偏差；目前评测仍主要基于 Wikipedia，无法覆盖更丰富的知识来源。

---

## 348. Learning Contextual Runtime Monitors for Safe AI-Based Autonomy

**arXiv ID:** 2601.20666 | [PDF](https://arxiv.org/pdf/2601.20666v1)

**作者:** Alejandro Luque-Cerpa `[一作]` (Chalmers University of Technology and University of Gothenburg), Hazem Torfah `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于上下文的运行时监控器，用于安全地管理 AI 控制器集合，能够在不同环境下动态选择最合适的控制器，必要时切换到可靠的安全控制器。

**💡 创新点**

创新点在于将安全监控问题转化为上下文感知的多臂老虎机（contextual bandit）学习任务，从而实现对每个上下文环境中最优控制器的主动选择，并提供理论安全保证与收敛性分析。

**🔧 技术方法**

采用了逻辑回归模型来估计控制器违背安全规范的概率，并通过上下文多臂老虎机算法进行在线学习与不确定性采样，以最小化 regret；同时对比了基于平均、专家混合等传统集成方法。

**📊 数据集**

实验使用了 Carla 交通仿真器配合 Scenic 生成器产生的两套基于视觉的控制器（CNN+PID），分别覆盖了城市巡航与动态城市环境两种驾驶场景，包含约 1.5 万个不同天气、时间、路况与障碍距离的上下文。

**📈 对比分析**

与传统平均、专家混合等无上下文基线相比，实验显示本文方法在安全性（违规率更低）与性能（成功率/奖励更高）上提升了约 30%–70%，且在不同置信阈值下保持了更低的误报率。

**⚠️ 局限性**

主要局限包括：仅考虑位置型上下文，无法处理时序或动态状态信息；模型假设违背概率服从逻辑回归，可能不适用于复杂非线性关系；以及在控制器覆盖不足或完全无偏的场景中，优势相对有限。

---

## 349. The Multiserver-Job Stochastic Recurrence Equation for Cloud Computing Performance Evaluation

**arXiv ID:** 2601.20653 | [PDF](https://arxiv.org/pdf/2601.20653v1)

**作者:** Francois Baccelli `[一作]` (INRIA and Télécom Paris), Andrea Marin `[通讯]` (Università Ca' Foscari Venezia)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文通过将多服务器作业排队模型（MJQM）在FCFS调度下转化为随机递推方程（SRE），推导了系统稳定性阈值并给出了子完美采样（SPS）与稳定性阈值估算算法；

**💡 创新点**

创新点在于将MJQM纳入单调可分离（monotone‑separable）框架，利用Loynes定理得到稳定性阈值的阈值型表达式，提出可并行化的SPS算法以及GPU加速实现；

**🔧 技术方法**

主要技术包括随机递推方程建模、单调可分离网络理论、Loynes定理、子完美采样（SPS）和SIMD/GPU并行化；

**📊 数据集**

实验使用了合成配置（2、5、20、2048台服务器）以及真实的Google Borg Cell B数据中心配置；

**📈 对比分析**

与已有解析解和离散事件仿真（DES）对比，SJSRE方法在准确性上与解析解一致，在性能上通过GPU实现可获得显著的速度提升，且能提供置信区间；

**⚠️ 局限性**

局限性包括仅适用于FCFS调度、缺乏完美采样的严格保证、对GPU内存和显存大小敏感、对更复杂调度策略或多资源类型的推广仍需进一步研究。

---

## 350. TGSBM: Transformer-Guided Stochastic Block Model for Link Prediction

**arXiv ID:** 2601.20646 | [PDF](https://arxiv.org/pdf/2601.20646v1)

**作者:** Zhejian Yang `[一作]` (Jilin University), Hechang Chen `[通讯]` (Jilin University)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5108294333)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了基于Transformer指导的重叠随机块模型TGSBM，用于大规模网络的链接预测。

**💡 创新点**

①将重叠随机块模型与稀疏图Transformer结合，保持可解释的社区结构；②引入增广稀疏注意力（expander-augmented），实现近线性复杂度的全局混合；③采用神经变分推理生成结构化后验。

**🔧 技术方法**

变分自编码器、稀疏图Transformer、expander图扩展、Binary‑Concrete/Stick‑breaking 机制、MLP解码器等。

**📊 数据集**

Cora、Citeseer、Pubmed、ogbl-collab、ogbl-ddi（OGB）以及 HeaRT 评估协议。

**📈 对比分析**

与传统启发式、MPNN、pairwise GNN 以及图Transformer（LPFormer）对比；在标准评估中 MRR/AUC 与 LPFormer 相当或更优；在 HeaRT 下平均排名 1.6（优于 LPFormer 的 2.4），并且训练速度最高可达 6×。

**⚠️ 局限性**

对超参数敏感，难以处理极稠密图的显存瓶颈；推理速度仍慢于纯 GNN；未覆盖动态/时变网络场景；解释性依赖社区结构假设。

---

## 351. A Foundation Model for Virtual Sensors

**arXiv ID:** 2601.20634 | [PDF](https://arxiv.org/pdf/2601.20634v1)

**作者:** Leon Götz `[一作]` (Volkswagen AG), Leo Schwinn `[通讯]` (Technical University of Munich)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5028233502)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了首个统一的基础模型（Foundation Model）用于虚拟传感器，能够在单一模型中预测多种虚拟传感器并选择需要的子集；

**💡 创新点**

创新点包括：①使用可学习的变体嵌入和信号相关性向量实现自动学习每个虚拟传感器所需的输入信号；②通过稀疏化和传感器选择机制实现高效推理；③结合教师强制（teacher forcing）与混合训练，解决自回归训练难题；

**🔧 技术方法**

技术主要为因果解码器式Transformer、时间与变体嵌入、注意力偏置的信号相关性矩阵、稀疏化阈值化、自动生成原型标记、教师强制训练；

**📊 数据集**

使用公开Traffic数据集（约15M样本）和大规模CAN总线车辆数据集（≈18B样本，1713变体）进行实验；

**📈 对比分析**

与16个单独模型（每个预测单一虚拟传感器）对比，统一模型在CAN数据集上保持甚至提升预测精度，同时计算时间缩短415倍、显存需求下降951倍；在Traffic数据集上误差略高但差距可忽略；

**⚠️ 局限性**

局限性包括：未对所有超参进行完整搜索，模型容量在极大传感器数目下可能受限；教师强制训练可能无法完全捕获错误积累，虽然通过微调可缓解；

---

## 352. /dev/SDB: Software Defined Boot -- A novel standard for diskless booting anywhere and everywhere

**arXiv ID:** 2601.20629 | [PDF](https://arxiv.org/pdf/2601.20629v1)

**作者:** Aditya Mitra `[一作]` (Kadir Has University), Tuğçe Ballı `[通讯]` (Kadir Has University)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5053940414)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于硬件模块和云服务的全新标准 /dev/SDB，支持在任何网络环境（有线、无线、蜂窝）下为不同用户按角色动态加载专属操作系统，实现无磁盘、基于内存的网络启动。

**💡 创新点**

创新点在于：①将网络连通性与启动过程集成于硬件模块，实现无须本地存储、无代理 DHCP 泵送、可通过多种链路（Wi‑Fi、蜂窝）连接云端；②在云端通过身份验证直接分发专属 OS 镜像，实现细粒度的用户与操作系统绑定；③结合 iPXE、DHCP、TFTP 的改进，提供安全链式加载与动态 DNS 解析。

**🔧 技术方法**

使用技术包括：轻量级单板电脑（Alpine Linux）作为硬件模块；iPXE 作为链式启动器；DHCP、TFTP、DNS 的代理/主服务；Flask+SQLite 的云管理平台；Linux 内核+initrd+文件系统的 OS 镜像；QEMU+GNS3 进行网络仿真。

**📊 数据集**

无公开数据集，实验使用自定义的三种轻量级 OS（Kolibri OS、Tiny Core Linux、Alpine Linux）以及在 GNS3 模拟网络中的三台目标机进行验证。

**📈 对比分析**

对比方法：通过仿真在不同网络拓扑下验证启动流程与身份认证，记录成功/失败登录日志。性能未给出具体指标，仅报告在理想仿真环境下每次启动 <3 秒；实际性能将受网络带宽、内存和服务器负载影响。

**⚠️ 局限性**

局限性包括：①仅在仿真环境验证，缺乏真实硬件与大规模网络的测试；②安全性主要靠硬件模块的独立性，若模块被物理篡改仍是风险；③未详细评估大规模并发启动时的带宽与服务器压力；④仅使用 SQLite，规模化部署需更强数据库支持。

---

## 353. Dialogical Reasoning Across AI Architectures: A Multi-Model Framework for Testing AI Alignment Strategies

**arXiv ID:** 2601.20604 | [PDF](https://arxiv.org/pdf/2601.20604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 354. When Vision Meets Texts in Listwise Reranking

**arXiv ID:** 2601.20623 | [PDF](https://arxiv.org/pdf/2601.20623v1)

**作者:** Hongyi Cai `[一作]` `[通讯]` (Universiti Malaya), Hongyi Cai (Universiti Malaya)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出轻量化 2B 参数的 Rank-Nexus 多模态列表式重新排序器，可同时处理文本与图像信息，提升检索结果排序质量。

**💡 创新点**

创新点：① 进阶跨模态训练策略（文本蒸馏 → 图像对比 → 图像列表式），实现模态间知识迁移；② 通过质量筛选与多样性核心集选择，仅使用极少量高质量训练样本即可获得最佳性能；③ 无需推理链的直接前向推断，显著降低推理延迟。

**🔧 技术方法**

技术细节：视觉语言模型（InternVL‑3‑2B / Qwen3‑VL‑2B）+ CLIP 对齐；知识蒸馏、对比学习、Listwise Plackett‑Luce 损失；LoRA 微调与 8‑bit 量化；多样性核心集（最大多样性问题的贪心近似）筛选；Prompt engineering 设计文本/图像列表式排序模板。

**📊 数据集**

使用的数据集：文本检索（MS MARCO passage、TREC DL19/DL20）；多模态检索（INQUIRE、MMDocIR）；零样本检索（BEIR 7 个子集）；图像对齐训练样本来自 MMDocIR，利用 CLIP 过滤；文本蒸馏标签来源于 GPT‑4/Claude‑4.5 等大型模型。

**📈 对比分析**

与现有方法对比：在文本检索上 Rank‑Nexus 74.6（DL19）/70.0（DL20）nDCG@10，超过 7B RankZephyr 与 reasoning‑based Rank‑R1；在 BEIR 零样本上平均 54.9，优于 14B Rank‑R1；在图像检索 INQUIRE 上 nDCG@50 73.9，接近 GPT‑4o；在多模态检索 MMDocIR recall@1 70.1，接近或优于 ColQwen+7B VLM；推理速度最快，单 GPU 2B 模型在 MS MARCO top‑1000 列表仅需约 2 秒，远快于 7B reasoning 模型。

**⚠️ 局限性**

局限性：对极细粒度视觉细节（如复杂技术图表）表现不如大型专有模型；依赖特定检索器（ColQwen），尚未实现端到端检索+排序；核心集选择在多模态领域仍有提升空间，且在极度稀缺对齐数据时仍受限。

---

## 355. SketchDynamics: Exploring Free-Form Sketches for Dynamic Intent Expression in Animation Generation

**arXiv ID:** 2601.20622 | [PDF](https://arxiv.org/pdf/2601.20622v1)

**作者:** Boyu Li `[一作]` (Hong Kong University of Science and Technology), Hongbo Fu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7485 | [OpenAlex ID](https://openalex.org/A5100616578)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让用户用自由形式的草图表达动画意图，结合视觉‑语言模型实现从草图到动态图的自动生成，并通过澄清与细化提示提升交互效果。

**💡 创新点**

创新点在于：①把草图视为开放式、含糊的提示而非固定指令，②引入层级澄清提示（确认、多选、填值、上传资源）主动协商模型解释；③在生成后提供基于关键帧的局部可视化细化，使创作者能在已生成视频上直接修改。

**🔧 技术方法**

技术主要包括：大规模视觉‑语言模型（如 GPT‑4o‑vision 等）用于草图解析与脚本生成；Manim 作为代码驱动的二维向量动画渲染；prompt‑engineering 与上下文记忆实现澄清与细化；前端网页界面实现草图绘制、生成、提示弹窗和关键帧编辑。

**📊 数据集**

数据集：原始研究使用了 24 次创作（8 位参与者 × 3 轮），包含多样化的自由草图、生成脚本、视频预览及澄清/细化交互记录；没有使用公开的标注数据集，主要依靠用户手绘的非结构化草图。

**📈 对比分析**

通过三阶段用户研究比较：Stage 1（单次生成） 19/24 结果被视为与意图不符；Stage 2（加入澄清提示） 19/24 结果更贴近用户意图，误差显著下降；Stage 3（局部细化） 12 份视频在 55 次细化后保持大部分场景不变且用户满意度提升。整体性能指标主要为定性用户评价和错误率下降，未给出定量 FPS 或 BLEU 等度量。

**⚠️ 局限性**

局限性：①受限于参与者样本（以大学生为主）；②缺乏实时反馈，草图到视频需数秒；③完全依赖通用 VLM，无法充分利用矢量/笔画信息；④仅支持简短的二维向量动画，难以处理角色、物理或长时序动画；⑤澄清与细化仍需手动交互，模型误解未完全消除。

---

## 356. Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation

**arXiv ID:** 2601.20614 | [PDF](https://arxiv.org/pdf/2601.20614v1)

**作者:** Yanqi Dai `[一作]` (Renmin University of China), Zhiwu Lu `[通讯]` (Renmin University of China)

**通讯引用:** 2472 | [OpenAlex ID](https://openalex.org/A5103244144)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 MathForge 框架，结合难度感知的策略优化（DGPO）与多维问题改写（MQR），以提升大型语言模型在数学推理任务中的表现。

**💡 创新点**

创新点在于：① 用难度平衡的组优势估计（DGAE）消除 GRPO 的更新幅度不平衡；② 用难度感知的问题层加权（DQW）突出更难问题；③ 通过保持答案不变的多方面改写（背景、抽象术语、子问题）系统地提升训练数据难度。

**🔧 技术方法**

主要技术包括：基于可验证奖励的强化学习、GRPO 与 DGPO 的策略优化、MQR 的自动化问题改写提示、以及与现有 RL 方法（GPG、DAPO、GSPO 等）的集成。

**📊 数据集**

使用的数据集包括公开的 MATH、AIME、AMC、Minerva、Olympiad、GeoQA‑8k 等数学推理基准；MQR 通过对 MATH 数据进行多维改写生成扩展数据。

**📈 对比分析**

与 GRPO、Dr.GRPO、GPG、DAPO、GSPO、GRPO‑AD 等方法对比，MathForge 在 Qwen2.5‑Math‑7B、Qwen2.5‑Math‑1.5B、Qwen2.5‑3B、DeepSeek‑Math‑7B 以及多模态 GeoQA 上平均提升约 4‑5%，在所有基准上均为最优。

**⚠️ 局限性**

局限性包括：依赖准确的可验证奖励；改写质量受限于改写模型能力；在极高难度或非数学推理任务中的效果尚未验证；以及训练成本相对较高。

---

## 357. Regularized Gradient Temporal-Difference Learning

**arXiv ID:** 2601.20599 | [PDF](https://arxiv.org/pdf/2601.20599v1)

**作者:** Hyunjun Na `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2148 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种正则化的GTd2算法R-GTD，用于在特征交互矩阵（FIM）奇异时仍能保证收敛的离策略价值评估方法。

**💡 创新点**

通过构造凸‑凹鞍点形式并加入正则化项，使得即使FIM奇异也能得到唯一解；并给出收敛性证明和误差上界，展示正则化如何在奇异情况下改进稳定性和精度。

**🔧 技术方法**

凸优化、鞍点理论、原始-对偶梯度动态（PDGD）、常微分方程（ODE）方法、重要性采样和线性函数逼近。

**📊 数据集**

实验使用人工生成的MDP：一个三状态的示例用于可视化正则化效果，另一个100状态、10动作、γ=0.99的离散MDP用于评估在FIM接近奇异时的性能。

**📈 对比分析**

与传统GTD2对比：在FIM奇异的情形下，R‑GTD收敛更稳定、方差更小、误差更低；在非奇异情况下，R‑GTD与GTD2得到相同的解，性能无明显损失。

**⚠️ 局限性**

仅在线性函数逼近下提出；未给出有限时收敛速率；对非线性逼近和更复杂MDP的适用性仍需进一步研究。

---

## 358. WFR-MFM: One-Step Inference for Dynamic Unbalanced Optimal Transport

**arXiv ID:** 2601.20606 | [PDF](https://arxiv.org/pdf/2601.20606v1)

**作者:** Xinyu Wang `[一作]` (Peking University), Tiejun Li `[通讯]` (Peking University)

**通讯引用:** 12302 | [OpenAlex ID](https://openalex.org/A5100703332)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于平均流（Mean‑Flow）的无平衡流匹配框架，并在Wasserstein–Fisher–Rao几何下实现了WFR‑MFM算法，能够在一次性更新中完成细胞状态的传输与质量变化，无需轨迹级ODE仿真。

**💡 创新点**

创新点包括：① 用平均速度和平均质量增长场概括任意时间区间的动态；② 通过导数恒等式实现对平均流的监督训练；③ 在无平衡OT中实现一阶（或多阶）一致性，提供可控的速度‑精度折中；④ 在大规模扰动预测中实现一次性高效推断。

**🔧 技术方法**

核心技术为：平均流变量定义与导数恒等式、无平衡流匹配（UFM）与条件目标、WFR几何下的最优熵传输耦合、神经网络参数化平均场、无仿真训练与推断算法。

**📊 数据集**

使用的实验数据包括：三类合成数据（Gene, Dyngen, Gaussian）、四类真实单细胞RNA测序数据（EMT, EB, CITE‑seq, Mouse hematopoiesis）以及一大规模合成扰动基准（5100 条扰动条件）。

**📈 对比分析**

与现有基于ODE仿真的WFR‑FM、Dormand–Prince RK5(4)、显式Euler等方法相比，WFR‑MFM 的推断速度提升 2–3 个数量级；在所有数据集上保持或超过对手的 𝒲₁ 与 RME 指标，且可通过增大推断步数实现精度与速度的连续折中。

**⚠️ 局限性**

局限性在于：未采用最新的平均流框架改进；对其他无平衡OT几何的推广尚未验证；缺乏真实实验扰动数据的最终验证，需进一步扩展到更大规模、复杂实验设置。

---

## 359. DIVERSE: Disagreement-Inducing Vector Evolution for Rashomon Set Exploration

**arXiv ID:** 2601.20627 | [PDF](https://arxiv.org/pdf/2601.20627v1)

**作者:** Gilles Eerlings `[一作]` (Hasselt University), Kris Luyten `[通讯]` (Hasselt University)

**通讯引用:** 3117 | [OpenAlex ID](https://openalex.org/A5063648916)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 DIVERSE 的框架，利用预训练模型中的 FiLM 层与 CMA‑ES 优化器，在不重新训练或访问梯度的前提下，通过调整隐藏层激活来系统性地探索深度网络的 Rashomon 集。

**💡 创新点**

创新点在于：①将 FiLM 作为低维调节向量，将网络空间压缩为可搜索的调节空间；②使用全协方差 CMA‑ES 在梯度不可得的连续空间中高效寻找既保持准确率又能产生显著预测分歧的模型；③通过软/硬不一致度混合的多样性度量实现对 Rashomon 成员的多尺度评估。

**🔧 技术方法**

核心技术包括：Feature‑wise Linear Modulation (FiLM) 层、Covariance Matrix Adaptation Evolution Strategy (CMA‑ES)、总变差距离 (TVD) 与硬不一致度 (hard disagreement) 的混合多样性度量、以及对验证集上误差阈值的 Gaussian 软惩罚。

**📊 数据集**

实验使用了三种公开数据集：MNIST（3 层 MLP）、PneumoniaMNIST（预训练 ResNet‑50）和 CIFAR‑10（预训练 VGG‑16）。

**📈 对比分析**

与传统的完整重新训练（Retrain）和 dropout‑based Rashomon 探索方法进行对比。DIVERSE 在相同 Rashomon 阈值下，产生与 Retrain 相当甚至更高的多样性度量，并且在计算时间上仅需数分钟，显著快于 Retrain（数小时），略慢于 dropout（秒级）但保持更严谨的验证/测试分离。

**⚠️ 局限性**

局限性包括：①全协方差 CMA‑ES 随维度快速变得昂贵，限制了可搜索的 FiLM 维度；②FiLM 调节虽能产生多样性，却不保证模型差异具有可解释或因果意义；③仅在 MLP/CNN 三个任务上验证，缺乏对更复杂架构（如 Transformer）的全面评估。

---

## 360. bi-modal textual prompt learning for vision-language models in remote sensing

**arXiv ID:** 2601.20675 | [PDF](https://arxiv.org/pdf/2601.20675v1)

**作者:** Pankhi Kashyap `[一作]` (Indian Institute of Technology Bombay), Biplab Banerjee `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 2027 | [OpenAlex ID](https://openalex.org/A5020786167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BiMoRS，一种基于图像字幕与视觉特征融合的双模态提示学习框架，用于遥感图像分类的域泛化任务。

**💡 创新点**

创新点在于使用冻结的图像字幕模型生成语义描述，并通过跨注意力机制将文本与视觉信息融合，生成图像自适应提示，显著提升跨域与新类别泛化性能。

**🔧 技术方法**

采用 BLIP-2 作为字幕模型，CLIP ViT‑B/16 作为视觉编码器，BERT tokenizer 处理字幕，轻量级投影头与跨注意力模块共同构建提示。

**📊 数据集**

在四个遥感数据集（PatternNet、RSICD、RESISC45、MLRSNet）上进行实验。

**📈 对比分析**

与 CoOp、CoCoOp、ProGrad、APPLeNet、MaPLe、TCP 等 SOTA 方法比较，BiMoRS 在三种泛化设置下平均提升 1–2% 准确率，且参数量仅为 1M。

**⚠️ 局限性**

局限性包括对字幕质量的依赖、仅验证分类任务且未涉及目标检测或分割等更复杂遥感任务。

---

## 361. FD-MAD: Frequency-Domain Residual Analysis for Face Morphing Attack Detection

**arXiv ID:** 2601.20656 | [PDF](https://arxiv.org/pdf/2601.20656v1)

**作者:** Diogo J. Paulo `[一作]` (University of Beira Interior), João C. Neves `[通讯]` (University of Beira Interior)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5002088416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了单图像人脸模糊攻击检测（S-MAD），提出基于频域残差的全局与局部特征融合方法。

**💡 创新点**

创新点是将频域的功率律残差与面部语义区域的局部残差结合，并通过马尔可夫随机场实现结构化融合，以提升跨数据集和跨攻击的泛化能力。

**🔧 技术方法**

技术主要包括离散傅里叶变换、功率律拟合去噪、PCA压缩、支持向量机分类、局部区域提取、马尔可夫随机场推理。

**📊 数据集**

使用SMDD（合成StyleGAN2+OpenCV）进行训练，评估于FRLL-Morph和MAD22（含WebMorph、MIPGAN等）以及MorDIFF。

**📈 对比分析**

与多种监督S-MAD方法（IDistill、MADation、D-FW-CDCN等）对比，在FRLL-Morph上平均EER 1.85%（第二名），在MAD22上平均EER 5.10%（第二名），整体性能优于轻量级模型且接近重型多任务模型。

**⚠️ 局限性**

限制在于仍依赖频域特征，对极端高质量GAN/扩散模糊攻击的细粒度局部痕迹捕捉有限，且仅使用单一语义区域划分，未结合3D姿态或对抗学习，可能在更复杂攻击中性能下降。

---

## 362. OS-Marathon: Benchmarking Computer-Use Agents on Long-Horizon Repetitive Tasks

**arXiv ID:** 2601.20650 | [PDF](https://arxiv.org/pdf/2601.20650v1)

**作者:** Jing Wu `[一作]` (University of Oxford), Vibhav Vineet `[通讯]` (Microsoft)

**通讯引用:** 8754 | [OpenAlex ID](https://openalex.org/A5045147286)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了面向长周期、重复工作流的计算机使用代理（CUA）基准，并通过少量压缩演示（Few‑Shot Condensed Workflow Demonstration）提升代理在此类任务中的执行能力。

**💡 创新点**

创新点在于①正式定义长周期、重复工作流任务并构建跨领域、七种执行环境的基准；②设计基于语义关键步骤的压缩演示方法，既降低上下文消耗，又能指导代理全局规划与子工作流逻辑。

**🔧 技术方法**

使用了基于POMDP的任务建模、AgentS2.5+GPT‑5框架、上下文窗口内的关键步骤演示，以及传统的CUA模型（OpenCUA、UI‑TARS‑1.5）作为对照。

**📊 数据集**

采用了真实与合成混合的数据集：费用报销领域的收据（含多种格式、不同币种）以及学术成绩单领域的多国家、多布局的成绩单，合成流程通过模板与LLM生成。

**📈 对比分析**

在Expense Report与Transcript两大领域的Level 1和Level 2任务上，使用SWA衡量部分进度，Baseline模型均为0%；相较于Baseline，使用压缩演示的AgentS2.5+GPT‑5在SWA上显著提升（如Level 1 100步时从27.08%提升至91.74%）。但仍未实现完整流程的成功率（SR为0%）。

**⚠️ 局限性**

主要局限包括：①模型上下文窗口限制，导致极长工作流仍难以完整放入演示；②压缩演示仅为软约束，缺乏硬性执行保证，因而在更高难度级别仍无法完成全部步骤。

---

## 363. Immersive Volumetric Video Playback: Near-RT Resource Allocation and O-RAN-based Implementation

**arXiv ID:** 2601.20625 | [PDF](https://arxiv.org/pdf/2601.20625v1)

**作者:** Yao Wen `[一作]` (Nanjing University), Kun Yang `[通讯]` (Nanjing University)

**通讯引用:** 30712 | [OpenAlex ID](https://openalex.org/A5058780924)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一套基于O‑RAN的沉浸式体素视频回放框架，联合调度无线、计算和内容资源，使用可变像素渲染比例实现动态质量与延迟权衡；

**💡 创新点**

首次在每帧级别实现无线与云计算协同控制，并将渲染像素比例作为连续可调的控制变量；采用结构化动作分解的SAC强化学习与QoE感知奖励加速收敛，突破传统离散或粗粒度资源分配的局限；

**🔧 技术方法**

O‑RAN架构（RIC、SMO、O‑Cloud）、深度强化学习（Soft Actor‑Critic）、Weber‑Fechner QoE模型、实时调度与功率控制、3D Gaussian splatting渲染；

**📊 数据集**

实验使用模拟环境生成多用户、分辨率与信道条件数据；原型部署在OAI 5G基站、Meta Quest 3 HMD和云服务器上进行真实流量与延迟采集；

**📈 对比分析**

与DDPG、均匀分配AVG、云端均匀渲染Cloud‑AVG对比；评估指标包括MTP延迟、QoE分数、公平性、调度成功率；SAC在模拟中将中位MTP延迟降低约18%，提升平均QoE和公平度；原型数据显示不同分辨率下吞吐与延迟的折衷；

**⚠️ 局限性**

仅在小规模实验与离线强化学习环境验证，缺乏大规模部署、在线安全约束、真实用户体验评估；对动态内容变化、非理想网络条件的鲁棒性尚待研究；

---

## 364. Single-Nodal Spontaneous Symmetry Breaking in NLP Models

**arXiv ID:** 2601.20582 | [PDF](https://arxiv.org/pdf/2601.20582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 365. GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection

**arXiv ID:** 2601.20618 | [PDF](https://arxiv.org/pdf/2601.20618v1)

**作者:** Shuguang Zhang `[一作]` (Institute of Computing Technology), Xiang Ao `[通讯]` (Institute of Computing Technology)

**通讯引用:** 3174 | [OpenAlex ID](https://openalex.org/A5068007462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用多模态大型语言模型生成客观图像描述作为语义锚点，从而量化图像与文本之间的差异来检测讽刺的框架（GDCNet）

**💡 创新点**

创新点在于将LLM生成的事实性图像描述作为跨模态的语义对齐参照，采用语义、情感和视觉文本一致性三维差异表示，并通过门控融合实现自适应模态权重

**🔧 技术方法**

使用CLIP作为视觉与文本编码器，LLaVA‑NEXT生成图像描述，构造语义差异（CLIP文本相似度）、情感差异（RoBERTa情感分类）和视觉文本一致性（CLIP图像-文本相似度），再通过MLP与门控融合进行分类

**📊 数据集**

在改进版的MMSD2.0多模态讽刺检测基准上进行评估，该数据集去除了无关线索并修正注释，提供更可靠的评测环境

**📈 对比分析**

与文字、图像及多模态基线（BiLSTM、BERT、ResNet、InCrossMGs、MOBA等）以及LLM直接推理（Zero‑Shot、CoT）对比，GDCNet在准确率、召回率和F1上均刷新MMSD2.0榜单，显著优于所有对照方法

**⚠️ 局限性**

主要限制包括对LLM生成描述的质量高度依赖，若描述语义不足或偏差，差异表示可能失效；同时生成描述的推理开销相对较大，影响实时应用

---

## 366. AgentIF-OneDay: A Task-level Instruction-Following Benchmark for General AI Agents in Daily Scenarios

**arXiv ID:** 2601.20613 | [PDF](https://arxiv.org/pdf/2601.20613v1)

**作者:** Kaiyuan Chen `[一作]` (xbench), Yuan Gong `[通讯]` (xbench)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向“一日”真实使用场景的AI代理评测基准AgentIF-OneDay，涵盖工作、生活、学习三类任务并强调文件中心、隐式指令推理和迭代细化三大能力维度。

**💡 创新点**

创新点在于①构建多模态、附件驱动的任务生成管线和基于工作流的评判流程；②将评估从单模型扩展到完整代理系统，强调真实用户需求与长尾任务；③通过视觉语言模型与搜索模式实现高质量自动化判分。

**🔧 技术方法**

使用了大语言模型（ChatGPT、Gemini‑3‑Pro 等）作为评判者与工具；多模态解析器与 VLM；搜索模式与人机标注相结合的评估框架。

**📊 数据集**

基准数据集包含104个任务、767个评分点，涵盖文本、PPT、HTML 等附件，任务由人工采样并通过自动化流水线扩展生成。

**📈 对比分析**

与四款主流代理（ChatGPT‑Agent、Genspark、Manus、Minimax‑Agent）对比，最高整体分为 Manus 0.645；不同代理在工作、生活、学习三类任务以及三能力维度上表现各异，API驱动代理与 RL 基础代理的基线能力趋于相近。

**⚠️ 局限性**

主要局限包括高昂的人工标注成本、任务生成的可扩展性瓶颈、隐式指令推理与长期一致性仍表现不足，以及评估仅覆盖“一日”场景，无法直接推广到更长周期任务。

---

## 367. CoBA: Integrated Deep Learning Model for Reliable Low-Altitude UAV Classification in mmWave Radio Networks

**arXiv ID:** 2601.20605 | [PDF](https://arxiv.org/pdf/2601.20605v1)

**作者:** Junaid Sajid `[一作]` (Tallinn University of Technology), Muhammad Mahtab Alam `[通讯]` (Tallinn University of Technology)

**通讯引用:** 3147 | [OpenAlex ID](https://openalex.org/A5057700615)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对5G毫米波网络中的低空无人机，提出并实现了一种名为CoBA的深度学习模型，用以准确区分无人机在授权和受限空域的飞行状态。

**💡 创新点**

创新点包括：①将卷积层、双向LSTM和注意力机制结合成一体化网络，以同时捕捉空间与时间特征；②首次在毫米波频段使用原始物理层测量进行无人机分类；③证明PCI和SSB索引两项指标即可实现近乎完美的分类，显著提升特征有效性。

**🔧 技术方法**

技术细节：使用1D卷积提取局部特征 → BiLSTM捕获双向时序依赖 → 注意力机制聚焦关键信息 → 全连接层与残差连接输出；训练采用加权交叉熵、AdamW优化器，采用梯度裁剪与Dropout保证收敛稳定；同时对比了SVM、KNN、DT、LR、传统LSTM和基于指纹的FP模型。

**📊 数据集**

数据集：在TalTech校园5G毫米波基站（n258波段）上收集的58,788条样本，包含PCI、SSB索引、RSSI、SSB‑RSSI、SS‑RSRP、SS‑SINR、SS‑RSRQ等物理层指标；每条样本对应低空（≤50 m）飞行的授权或受限标签。

**📈 对比分析**

评估方法：对所有模型进行90/10/10的训练/验证/测试划分，计算准确率、精确率、召回率和F1；CoBA在全部特征下达成99.89%准确率，略优于FP（99.78%）和传统机器学习模型（≈90–91%）；使用仅PCI和SSB索引时准确率仍保持99.82%，显示模型对特征的高效利用。

**⚠️ 局限性**

局限性：仅针对二分类低空场景，未覆盖高空或多类场景；实验环境单一（单一校园基站配置），可能影响跨域泛化；模型训练较为复杂，部署至实时边缘设备仍需进一步优化。

---

## 368. Helper-Assisted Coding for Gaussian Wiretap Channels: Deep Learning Meets PhySec

**arXiv ID:** 2601.20678 | [PDF](https://arxiv.org/pdf/2601.20678v1)

**作者:** Vidhi Rana `[一作]`, Taejoon Kim `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一种新的自编码器架构，用于可靠性层设计，并在多帮手场景下设计了相应的编码方案。

**💡 创新点**

创新点在于：①不使用SIC方法而直接让接收机在训练期间估计消息，显著缩短训练时间；②将该架构推广到支持多帮手的系统，并在Gaussian多址窃听信道上验证其有效性。

**🔧 技术方法**

采用自编码器技术结合信息理论的辅助码设计，利用仿真实现多帮手干扰抵消与信息泄露抑制。

**📊 数据集**

使用仿真生成的数据集（针对Gaussian多址窃听信道与多帮手场景），未使用公开真实数据集。

**📈 对比分析**

与基于SIC方法的传统架构对比，训练时间缩短且在多帮手配置中信息泄漏显著下降；在Gaussian多址窃听信道的仿真结果与已有非构造性理论一致，表明编码方案在实际场景中具备较好性能。

**⚠️ 局限性**

局限性包括：①实验仅基于仿真，缺乏真实无线环境验证；②对不同信道模型和噪声水平的鲁棒性未做系统评估；③未给出具体实现复杂度分析。

---

## 369. A Dialectic Pipeline for Improving LLM Robustness

**arXiv ID:** 2601.20659 | [PDF](https://arxiv.org/pdf/2601.20659v1)

**作者:** Sara Candussio `[一作]` `[通讯]`, Sara Candussio

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种对话式的“Dialectic Pipeline”，让大型语言模型在生成答案前通过自我质疑和修正，提升答案的准确性与可靠性。

**💡 创新点**

创新点在于将模型的生成、检验（antithesis）与最终决策（synthesis）三个阶段串联成一种自监督的自我对话流程，既保留模型的泛化能力，又能在不依赖额外验证器或大规模微调的前提下显著降低幻觉率。

**🔧 技术方法**

技术核心包括：多步自生成与自评（自对话），对上下文进行oracle‑RAG式检索与动态摘要/过滤，结合不同模型架构（Gemma‑2、Phi‑3、LlaMa‑3.1）的前置与后置归一化、RoPE/LongRoPE位置编码、Multi‑Query Attention 等。

**📊 数据集**

实验使用了多跳推理数据集（如 HotpotQA、Multi‑Hop QA 等）以及标准单选任务，且对不同模型尺寸与族群进行了对比。

**📈 对比分析**

与传统Chain‑of‑Thought单向提示及纯RAG方法相比，Dialectic Pipeline 在准确率上提升了约5–10个百分点，并在多模型、多数据集上保持了更稳健的性能。

**⚠️ 局限性**

局限性包括：对极长上下文的处理仍受限于模型窗口；自评阶段可能被错误信息误导，导致合成答案不稳定；实验主要集中在事实问答与多跳推理，其他领域（如生成式创作）尚未充分验证。

---

## 370. OnePiece: A Large-Scale Distributed Inference System with RDMA for Complex AI-Generated Content (AIGC) Workflows

**arXiv ID:** 2601.20655 | [PDF](https://arxiv.org/pdf/2601.20655v1)

**作者:** June Chen `[一作]` (Wechat Tencent), Stephen Liu `[通讯]` (Wechat Tencent)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了面向AI生成内容（AIGC）工作流的分布式推理系统，采用微服务拆分、RDMA单向通信、双环缓冲和动态节点管理，实现多阶段流水线的弹性资源调度和高效GPU利用；

**💡 创新点**

创新点包括：1）将AIGC流水线细粒度拆分为多服务并通过RDMA实现高吞吐低延迟通信；2）提出双环环形缓冲（double‑ring buffer）机制，在RDMA环境下无CPU介入即可解决多生产者死锁；3）动态节点管理器（Node Manager）实时监控GPU利用率并弹性分配实例；4）流水线和快速拒绝机制保证高并发下稳定延迟；

**🔧 技术方法**

使用技术包括：RDMA单向传输（one‑sided RDMA），GPU内存直接访问，双环环形缓冲，微服务架构（TaskManager/RequestScheduler/TaskWorker/ResultDeliver），动态调度与弹性实例分配，Paxos原子领导选举，NVIDIA Triton推理服务器等；

**📊 数据集**

主要使用WAN2.1模型所需的图像‑视频生成数据集（如公开的图像‑视频对），以及为评估所用的标准AI生成内容基准数据集；

**📈 对比分析**

与传统单体推理管线进行对比，实验表明在Wan2.1图像‑视频生成任务上GPU资源消耗下降约16倍，吞吐量提升显著，且在高并发负载下仍保持低延迟；

**⚠️ 局限性**

局限性包括：1）对RDMA网络硬件和配置有较高依赖；2）不支持重传或可靠传输，易受网络丢包影响；3）主要针对AIGC多模型流水线，其他类型推理任务适用性未验证；4）系统实现复杂，对运维人员要求较高。

---

## 371. P2S: Probabilistic Process Supervision for General-Domain Reasoning Question Answering

**arXiv ID:** 2601.20649 | [PDF](https://arxiv.org/pdf/2601.20649v1)

**作者:** Wenlin Zhong `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**通讯引用:** 2508 | [OpenAlex ID](https://openalex.org/A5041727387)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Probabilistic Process Supervision（P2S）框架，通过动态合成高质量Gold-CoT和Path Faithfulness Reward（PFR）实现对LLM推理过程的细粒度自监督强化学习。

**💡 创新点**

创新点：①动态Gold-CoT合成与过滤机制，使模型自我生成并挑选最可靠的推理路径；②PFR在每一步计算条件概率差值，提供密集、步骤级的奖励信号，解决传统仅关注最终结果的奖励稀疏问题；③PFR可与任意结果奖励无缝集成，形成层次化奖励体系。

**🔧 技术方法**

核心技术：自监督强化学习（GRPO）+ 过程级奖励（PFR）+ 动态Gold-CoT合成+ 策略梯度与优势估计 + 归一化概率差值计算 + 逐步权重调度。

**📊 数据集**

使用的公开数据集：DROP（阅读推理）和Medical QA（医学问答），均经过长度和答案长度过滤后构成10k/2k训练/测试集。

**📈 对比分析**

与多类基线对比：提示式（CoT、Self‑Consistency）、传统微调+RL（SFT、GRPO）、RLPR族（DRO、RLPR、VeriFree）和General Reasoner。P2S在DROP上ACCAvg 70.70、ROUGE 76.78，击败所有对比方法；在Medical QA上ACCAvg 24.28、ROUGE 52.90，均优于RLPR、VeriFree、General Reasoner等。

**⚠️ 局限性**

局限性：①仍需在每次训练迭代中生成和评估多条Gold-CoT，计算量较大；②奖励仅基于模型自身概率，可能受模型偏好影响；③对极长推理链的规模化适应性尚未充分验证；④对完全无结构化或非常开放域的推理任务仍需进一步评估。

---

## 372. Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability

**arXiv ID:** 2601.20642 | [PDF](https://arxiv.org/pdf/2601.20642v1)

**作者:** Rohan Asthana `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vasileios Belagiannis `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 3866 | [OpenAlex ID](https://openalex.org/A5027065196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无去噪的记忆检测与缓解方法。

**💡 创新点**

创新点在于同时利用高噪声下的梯度范数和低噪声下的角度相似度，组合两种信息得到更鲁棒的记忆检测指标。

**🔧 技术方法**

使用扩散模型的分数估计、角度相似度计算、加权求和等技术，并在Stable Diffusion实现。

**📊 数据集**

在Stable Diffusion v1.4、v2.0以及MemBench上进行实验，使用已知的记忆与非记忆文本提示。

**📈 对比分析**

与现有无去噪检测基线相比，AUC和TPR@1%FPR均提升，检测速度约快5-7倍；在缓解实验中，SSCD相似度显著下降，CLIP与美学分保持或提升。

**⚠️ 局限性**

局限在于仅针对高/低噪声极端有效，未探究中间噪声层次，且方法主要验证于图像文本扩散模型，迁移到其他任务需进一步验证。

---

## 373. Agent Benchmarks Fail Public Sector Requirements

**arXiv ID:** 2601.20617 | [PDF](https://arxiv.org/pdf/2601.20617v1)

**作者:** Jonathan Rystrøm `[一作]` (Oxford Internet Institute), Chris Russell `[通讯]` (Oxford Internet Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用LLM辅助的文献综述方法，对1300余篇LLM Agent基准论文进行系统分析，提出了六项针对公共部门的基准评价准则，并评估现有基准是否满足这些准则。

**💡 创新点**

创新点在于将公共行政理论与技术基准标准相结合，构建了面向公共部门的完整评价框架，并首次系统性检验了现有基准的符合度。

**🔧 技术方法**

采用LLM的零样本结构化数据提取、链式思维提示以及人工校验相结合的技术流程，对论文进行准则符合度标注。

**📊 数据集**

数据来源为从ArXiv及ICLR/NeurIPS/ACL等会议检索到的约1300篇LLM Agent基准论文。

**📈 对比分析**

通过对每篇论文的六项准则进行二元/未知标注，并统计覆盖率；结果显示没有单一基准满足所有准则，尤其在公共部门相关性和指标完整度方面表现最差。

**⚠️ 局限性**

局限在于LLM与人工标注的一致性不高，公共部门相关性样本稀缺导致评估不够全面，且方法对极少数标注缺乏深度验证。

---

## 374. Ranking-aware Reinforcement Learning for Ordinal Ranking

**arXiv ID:** 2601.20585 | [PDF](https://arxiv.org/pdf/2601.20585v1)

**作者:** Aiming Hao `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5375 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于强化学习的 Ranking‑Aware Reinforcement Learning (RARL) 框架，用以同时优化序数回归和排序任务，能够在单图像回归与多图像排序之间切换。

**💡 创新点**

核心创新在于：①将回归与 Learning‑to‑Rank 通过可验证奖励函数统一到同一目标；②设计响应突变操作（Response Mutation Operation, RMO）解决策略熵崩塌和梯度消失；③两阶段训练策略避免多目标训练不稳定。

**🔧 技术方法**

技术手段包括：强化学习与可验证奖励（RLVR）框架，Group Relative Policy Optimization (GRPO) 算法，Kendall τ、长度一致性等多维奖励；模型采用 Qwen2.5‑VL 的视觉语言模型进行生成；RMO 通过高奖励样本注入提高探索。

**📊 数据集**

在三个基准上验证：UTKFace（人脸年龄回归），COCO‑REM（目标计数排序），AVA（美学评估）三种不同的序数回归与排序任务。

**📈 对比分析**

与基准模型和传统监督微调（SFT）相比，RARL 在 MAE、Spearman 相关系数和 Kendall τ 上均显著提升，尤其在多图像排序上取得 SOTA 结果（如 UTKFace MAE 3.81，AVA SRCC 0.803）。

**⚠️ 局限性**

局限性包括：依赖大量标注数据；RLVR 的训练成本高，调参复杂；RMO 需要手工设定替换比例，可能不适用于所有任务；对极度稀疏或噪声极大的序数标签效果尚待进一步验证。

---

## 375. Active Learning for Decision Trees with Provable Guarantees

**arXiv ID:** 2601.20775 | [PDF](https://arxiv.org/pdf/2601.20775v1)

**作者:** Arshia Soltani Moakhar `[一作]` (University of Maryland), MohammadTaghi Hajiaghayi `[通讯]` (University of Maryland)

**通讯引用:** 5664 | [OpenAlex ID](https://openalex.org/A5111876448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对二分类决策树的主动学习理论框架，给出了争议系数的首次分析，并设计了能够在多项式时间内实现 (1+ε) 近似的主动学习算法，得到多项式对数标记复杂度；

**💡 创新点**

创新点在于：1) 首次量化决策树的争议系数并证明其在满足“路径上每个节点使用不同特征维度”和“输入为网格结构”的前提下可被多项式对数上界；2) 设计了通用的多项式对数标记复杂度的主动学习算法，提供乘法误差保证，突破了传统加性误差框架的局限；

**🔧 技术方法**

使用理论工具包括 VC 维度、争议系数、版本空间收缩、球形距离分析等；算法采用迭代裁剪、误差上下界估计、直接估计阶段等；

**📊 数据集**

实验使用了规模为 10^7 的离散数据集，加入 10% 标记噪声进行验证；

**📈 对比分析**

与传统加性误差主动学习方法对比，本文算法在满足假设条件下实现了多项式对数标记复杂度，并在实验中成功率超过 90%；理论上标记复杂度为 O(ln^2 n·(2^h + ln(1/δ)) + ln^2 n/ε^2·(2^h ln ln^n/ε + ln(1/δ)))，几乎达到下界；

**⚠️ 局限性**

局限性：需要每条根到叶子路径上的特征维度互不相同且输入分布为网格状结构，若放宽这些假设则标记复杂度退化为多项式；目前仅针对离散数据域，未扩展到连续空间；

---

## 376. GraphAllocBench: A Flexible Benchmark for Preference-Conditioned Multi-Objective Policy Learning

**arXiv ID:** 2601.20753 | [PDF](https://arxiv.org/pdf/2601.20753v1)

**作者:** Zhiheng Jiang `[一作]` (University of California), Volkan Ustun `[通讯]` (USC Institute for Creative Technologies)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5089250268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于图结构的资源分配 benchmark GraphAllocBench，结合 CityPlannerEnv，提出 Preference-Conditioned Policy Learning（PCPL）方法并引入新的评价指标。

**💡 创新点**

创新点在于：①提供可扩展的图型 benchmark，可自定义需求、资源、目标和依赖；②引入 PNDS 与 OS 两个补充指标，更直观评估偏好一致性和解的非支配性；③将异构图神经网络（HGNN）与偏好条件化相结合，显著提升大规模图场景下的性能。

**🔧 技术方法**

技术手段包括：Proximal Policy Optimization (PPO) + MLP/ HGNN；偏好向量条件化、Graph Attention Network、全局池化（Mean/Max、Attention）；smooth Tchebycheff scalarization；评估指标计算。

**📊 数据集**

数据集：GraphAllocBench 自定义的若干资源分配问题集合（Problems 0–6），通过 CityPlannerEnv 生成的图结构和目标函数；无公开真实数据集。

**📈 对比分析**

比较方法：将 PCPL 与 PD‑MORL、MLP 基线进行对比，评估超体积比、PNDS 与 OS。实验显示 HGNN 在大规模问题（100 需求/资源）下超越 MLP，取得更高的超体积，但 MLP 在偏好一致性（OS）上略好；在小规模问题上两者相近。

**⚠️ 局限性**

局限：benchmark 仍以合成图为主，缺乏真实城市规划或灾害模拟等复杂情境；对偏好一致性的 OS 仅评估秩相关性，未考虑量值差异；HGNN 的训练成本和局部最优陷阱仍存在；对风险与不确定性建模不足。

---

## 377. Compression Tells Intelligence: Visual Coding, Visual Token Technology, and the Unification

**arXiv ID:** 2601.20742 | [PDF](https://arxiv.org/pdf/2601.20742v1)

**作者:** Xin Jin `[一作]` (Eastern Institute of Technology), Wenjun Zeng `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 23057 | [OpenAlex ID](https://openalex.org/A5049963367)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了传统视觉编码与现代视觉 Token 技术，提出统一的理论框架（信息瓶颈、Rate‑Distortion、熵编码等），并在多模态大语言模型、AI‑生成内容与具身 AI 等场景中验证了压缩效率与智能性能的深度关联。

**💡 创新点**

创新点包括：①将视觉编码与 Token 技术从信息论、功能、优化和目标四维统一，形成“Compression Tells Intelligence”统一视角；②给 Token 化引入信息瓶颈与 Rate‑Distortion 的数学表述；③提出从传统压缩技术启发 Token 化的多种压缩策略（Attention、Similarity、RL、Query 等）与可量化的压缩-任务平衡；④在实验中展示了基于统一框架的 Token 编码可在保持 96% 任务性能的同时压缩至 25% token，且比现有 codec 节省 36% bitrate。

**🔧 技术方法**

使用的核心技术包括：信息瓶颈与 Rate‑Distortion 理论、熵编码（Huffman、Arithmetic）、Transform/Quantization、VQ‑VAE/VQ‑GAN、ViT/CLIP/BLIP‑2、Diffusion 与 Flow‑Matching、Transformer/LLM、Attention‑based、Similarity‑based、Query‑based、RL‑based Token 压缩方法。

**📊 数据集**

实验数据集涵盖常见视觉语言基准：MME、ScienceQA、VQAT、POPE、SeedBench、VizW 等，并在 ImageNet/COCO 等公开数据集上进行编码与重建评估。

**📈 对比分析**

通过在不同 Token 保留率（25%、12.5%、6.25%）下与完整 Token baseline、FastV、PruMerge、QPID 等方法比较，发现 QPID 在 25% 保留率时达到 96.8% 的任务平均准确率；相较传统 codec，CoTAM 在保持同等任务性能的前提下实现 36% bitrate 节省；整体性能优于现有 Token 压缩与 codec 方案。

**⚠️ 局限性**

局限性包括：①统一 Tokenizer 难以同时兼顾语义一致性与像素重建精度；②不同任务/模态迁移性差，需针对性微调；③高分辨率视频 Token 压缩仍面临延迟与 GPU 负载瓶颈；④理论框架未给出可直接量化的最优点，需更多实证验证；⑤缺乏统一标准与评测基准，难以跨平台复现。

---

## 378. Rendering Portals in Virtual Reality

**arXiv ID:** 2601.20722 | [PDF](https://arxiv.org/pdf/2601.20722v1)

**作者:** Milan van Zanten `[一作]` (University of Basel), Milan van Zanten `[通讯]` (University of Basel)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5061633834)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

**🎯 论文内容**

本文研究了在VR中实现可穿越门洞（portal）并保持无缝过渡，以解决现实空间限制问题；

**💡 创新点**

创新点在于：1) 将门洞从平面改为盒子，利用背面剔除保证双眼渲染无缝；2) 采用模板缓冲区仅渲染门洞可见区域；3) 使用单通道实例化渲染减少双眼重复绘制；

**🔧 技术方法**

技术包括：立体渲染、模板/深度/模板缓冲区、背面剔除、单通道实例化渲染、基于GPU的纹理推送优化；

**📊 数据集**

未使用公开数据集，实验基于自制场景（含0~6个互连门洞），在Nvidia GeForce GTX 1070上测评；

**📈 对比分析**

通过比较不同门洞数量下的FPS和GPU帧时长，发现单对门洞已导致约37% FPS下降、近50倍GPU帧时长增长；使用模板缓冲区可将渲染量按三分之一提升；单通道实例化可大幅降低重绘成本；

**⚠️ 局限性**

局限性包括：1) 门洞数量受性能限制，仍需尽量减少可见门洞；2) 单通道实例化无法单独为两眼实现门洞传送；3) 实验仅在单一硬件平台验证，未探讨多平台/多GPU情况；

---

## 379. Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs

**arXiv ID:** 2601.20704 | [PDF](https://arxiv.org/pdf/2601.20704v1)

**作者:** Melika Mobini `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1208 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究构建了 10,000 篇焦点论文的人类参考文献与 LLM（GPT‑4o、Claude Sonnet 4.5）生成的参考文献对应的引用网络，并与领域匹配的随机基线进行对比，评估结构与语义特征在区分两类网络中的效果。

**💡 创新点**

创新点在于首次将 LLM 生成的参考文献映射为完整的引文图，系统比较结构属性与文本嵌入在检测 LLM 与人类参考列表之间的可分辨性，揭示 LLM 文本语义存在可检出的痕迹。

**🔧 技术方法**

技术方法包括：基于 SciSciNet 的网络构建；计算节点结构特征（度数、紧密度、聚类系数等）；利用 3072 维标题/摘要嵌入；采用随机森林 (RF) 与图神经网络 (GNN) 对图级别聚合特征进行分类；使用余弦相似度评估节点语义相似性。

**📊 数据集**

使用的数据集为 SciSciNet（10,000 篇 Q1 期刊论文，约 275k 参考文献），并通过模糊匹配与保守阈值检索生成的 LLM 参考文献。

**📈 对比分析**

在三类比较（人类、LLM、随机）中，结构特征仅能以 0.60 的准确率区分人类与 LLM，显著拒绝随机基线（≈0.89–0.92）。加入嵌入后 RF 准确率提升至 0.83，GNN 结合节点嵌入在 LLM vs. 人类上达到约 93% 的测试准确率；在 Claude 版本中，RF 可分辨率约 0.77，随机基线仍被清晰排除。

**⚠️ 局限性**

主要限制包括：仅使用基于参数知识的 LLM 生成文献，未涉及外部数据库检索；实验聚焦于 Q1 期刊范围，可能对其他领域适用性不足；模型在处理更大规模或更复杂的参考结构时的可扩展性未作评估。

---

## 380. Positive-Unlabeled Reinforcement Learning Distillation for On-Premise Small Models

**arXiv ID:** 2601.20687 | [PDF](https://arxiv.org/pdf/2601.20687v1)

**作者:** Zhiqiang Kou `[一作]` (Southeast University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60099 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在离线部署的场景下，提出了一种正负标签无监督（PU）强化学习蒸馏方法，通过一次性查询黑盒教师获取锚点，随后在本地采样多条候选回复并自评产生偏好信号，最终实现小模型的 RL 对齐；

**💡 创新点**

创新点在于：①利用单个高质量锚点而非大量人标或奖励模型，显著降低教师调用量；②通过 PU 归一化自评将候选集转换为软标签分布（LDL‑GRPO），实现无监督的组级偏好优化；

**🔧 技术方法**

技术包括：正负标签无监督自评、锚点条件偏好归一化、标签分布学习（LDL‑GRPO）以及组相对策略优化（GRPO）；

**📊 数据集**

数据集涵盖单模态写作（WritingPrompts）、数学推理（Competition Math）以及多模态视觉问答（A-OKVQA）等多任务场景；

**📈 对比分析**

与 SFT、SFT→SFT、SinglePair‑DPO、Anchor‑GRPO、Self‑PPO、AnchorRank‑DPO 等基线对比，实验显示在 Raw 与 LC（长度控制）评测中，LDL‑GRPO 在所有任务上均获得最高或次高的胜率，显著提升小模型的对齐效果；

**⚠️ 局限性**

局限性包括：对教师锚点质量敏感；需要在本地生成足够多的候选样本（K 需要调优）；方法在极端资源受限或教师表现欠佳的场景下可能效果不佳；

---

## 381. Polite But Boring? Trade-offs Between Engagement and Psychological Reactance to Chatbot Feedback Styles

**arXiv ID:** 2601.20683 | [PDF](https://arxiv.org/pdf/2601.20683v1)

**作者:** Samuel Rhys Cox `[一作]` (Aalborg University), Niels van Berkel `[通讯]` (Aalborg University)

**通讯引用:** 3293 | [OpenAlex ID](https://openalex.org/A5003896144)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过三种聊天机器人反馈风格（Direct、Politeness、Verbal Leakage）与心理距离（个人 vs 社会）交互的在线实验，探讨它们对用户心理反抗、自由感受以及行为意向的影响。

**💡 创新点**

创新点在于首次将“Verbal Leakage”（语义滑漏、停顿、口吃等自然口语特征）作为一种新颖的间接反馈策略，与传统直接和礼貌风格进行对比，揭示了心理反抗与参与度之间的权衡。

**🔧 技术方法**

技术方法包括使用GPT‑4o生成符合指定反馈风格的对话，采用3×2混合实验设计收集自评量表与开放式回答，并用线性模型和事后检验分析结果。

**📊 数据集**

数据集由来自Prolific的158名美国受试者构成，使用自定义的个人和社会影响情境文本，以及生成的聊天机器人对话日志和问卷数据。

**📈 对比分析**

评估通过方差分析和Tukey检验比较三种反馈风格在情绪反抗、自由威胁、信息处理和说服力等指标上的差异；结果显示礼貌风格降低怒气和自由威胁但缺乏惊讶感，Verbal Leakage 引发更高惊讶与幽默感，礼貌风格在说服力上最高。

**⚠️ 局限性**

局限性包括实验采用情境模拟而非真实决策行为，样本仅来自美国，缺乏长期和多模态（语音、面部表情）验证，且仅一次交互未能评估持续使用的衰退效应。

---

## 382. ScaleFree: Dynamic KDE for Multiscale Point Cloud Exploration in VR

**arXiv ID:** 2601.20758 | [PDF](https://arxiv.org/pdf/2601.20758v1)

**作者:** Lixiang Zhao `[一作]` (Xi'an Jiaotong-Liverpool University), Lingyun Yu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套 GPU 加速的自适应 KDE 算法，能够在沉浸式 VR 环境中实时重算多尺度点云的密度场，从而支持连续的尺度切换、精确的空间选择与导航。

**💡 创新点**

创新点在于：①将自适应 KDE（基于 Epanechnikov 核、可变带宽）动态计算在 GPU 上；②使用 k‑d 树邻域查询与层次化并行归约实现高效加速；③将动态密度场无缝嵌入选择（基于 Marching Cubes 的等值面）和逐级导航工作流；④通过用户研究验证其在准确性、效率与认知负荷方面的优势。

**🔧 技术方法**

技术栈包括：GPU 计算着色器（HLSL）、k‑d 树空间索引、Epanechnikov 自适应核、层次化并行归约、Marching Cubes、Unity3D 引擎、Vive Pro 2 VR 设备、NASA TLX 负荷量表、Bootstrap 置信区间等统计方法。

**📊 数据集**

数据集为宇宙学 N‑body 模拟产生的点云，包含 5 个时间步（每步 76k、164k、442k 及 Filament 数据集），用于性能测试与 24 人的用户实验。

**📈 对比分析**

比较方式：①性能：GPU 与单核/多核 CPU 对比，取得 0.042–0.309 秒的执行时间，分别相较于单核 183×、多核 36–157×；②功能：在选择任务中对比预先计算的单分辨率（PS）、多分辨率（PM）与实时动态（DR）三种密度场；结果显示 DR 在 F1 与 MCC 上比 PM/PS 高 1.21–1.92 倍，完成时间比 PM/PS 快 0.70–1.08 倍，NASA TLX 工作量更低，用户偏好最高。

**⚠️ 局限性**

局限性：①固定 64³ 分辨率仍可能遗漏细节；②每次尺度变换都会触发重算，导致频繁 zoom 时的额外开销；③需要 GPU 与 GPU‑CPU 内存传输，存在瓶颈；④实现复杂度高，需高性能显卡；⑤对动态变化的数据分布适应性有限；未来可通过阈值筛选、局部重算、GPU‑端选择算法等方式改进。

---

## 383. When More Data Doesn't Help: Limits of Adaptation in Multitask Learning

**arXiv ID:** 2601.20774 | [PDF](https://arxiv.org/pdf/2601.20774v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Mingyue Xu `[通讯]` (Purdue University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5101976077)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究多任务学习中自适应性（adaptivity）的统计极限，提出并证明了即使每个任务样本量无限大，仍无法通过仅观察多源数据达到最优学习率。

**💡 创新点**

创新点在于：
- 给出一个比之前更强的无自由午餐（no‑free‑lunch）定理，消除样本量上界 n<2/β−1 的限制；
- 通过精确控制 KL 散度和 Fano 方法构造任务分布，证明自适应学习在样本量任意大时仍无效；
- 证明在特定任务设置下，全局 ERM（Pooling）可近似达到最优自适应率，揭示自适应与 Pooling 的关系。

**🔧 技术方法**

使用信息论工具（Fano 方法、KL 散度上界）以及统计学习理论中 Bernstein 条件、转移指数（transfer exponent）等概念。

**📊 数据集**

本文为理论研究，不使用具体真实数据集，而是构造离散分类任务（两类、两个特征点）来进行证明。

**📈 对比分析**

对比方法主要是：1）自适应算法（无额外信息的学习器）与 2）已知良好任务的 ERM。结果显示，自适应算法在样本量任意大时仍只能得到 Ω((n√N)^−1/(2−β)) 的误差，无法达到 minimax 率；而 ERM 在已知任务时可获得更快的 O((n√N)^−1/(2−β)·exp(−…)) 误差；Pooling 在该构造下近似最优，仅多一个对数因子。

**⚠️ 局限性**

限制：
- 需要任务数 N = Ω(exp(n)) 才能得到强不可能结论；
- 仅在离散二分类模型下给出证明，通用性未知；
- 对自适应率的上界与下界仍未完全匹配，具体最优自适应率仍是开放问题。

---

## 384. COMET-SG1: Lightweight Autoregressive Regressor for Edge and Embedded AI

**arXiv ID:** 2601.20772 | [PDF](https://arxiv.org/pdf/2601.20772v1)

**作者:** Shakhyar Gogoi `[一作]` `[通讯]` (Jorhat Engineering College), Shakhyar Gogoi (Jorhat Engineering College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种专为边缘设备设计的基于行为空间的自回归回归框架 COMET‑SG1，用于长期稳定的时间序列预测。

**💡 创新点**

通过线性编码多尺度行为空间、内存锚定的状态转移和学习校正，实现无循环/注意力的长期稳定推理，且模型结构内置 bounded behavior。

**🔧 技术方法**

采用线性投影、多尺度行为空间编码、加权 L1 距离检索、软权重聚合、学习的线性校正以及内部行为状态更新等技术。

**📊 数据集**

使用了非平稳、存在 regime 变化的合成交易类时间序列作为实验数据。

**📈 对比分析**

与 kNN、MLP、LSTM 基线在一步与五步 MAE 及长时序自回归漂移进行对比，COMET‑SG1 在保持短期精度的同时实现了 bounded drift，参数量仅 2.9 KB，内存 345 KB。

**⚠️ 局限性**

目前仅在 Python 原型上验证，推理时需全内存扫描导致 prototype 运行慢，实际部署需进一步优化近似邻域检索与内存管理。

---

## 385. The Latent Space of Equational Theories

**arXiv ID:** 2601.20759 | [PDF](https://arxiv.org/pdf/2601.20759v1)

**作者:** Luis Berlioz `[一作]` (National Autonomous University of Honduras), Paul-André Melliès `[通讯]` (Paris City University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了等价理论的潜在空间，并将推理图映射到该空间中，观察推理流的空间结构。

**💡 创新点**

首次将等价理论用Stone配对构造特征空间，再通过PCA得到三维潜在空间，发现等价理论在空间中聚类，推理边具有方向性和并行性。

**🔧 技术方法**

利用有限模型理论的Stone配对、机器学习的PCA降维、统计分析与可视化技术，以及图论构建推理图。

**📊 数据集**

基于Tao等人整理的4694个基本等价理论，以及1000个随机生成的大小4–16有限幺半群模型。

**📈 对比分析**

通过统计比较可逆边、原子边和严格边在潜在空间中的长度差异，发现可逆边平均约为原子边的七分之一；潜在空间能够精准区分理论签名、期望与方差，并通过可视化验证推理流方向；尚未给出精确的数值指标或对比实验。

**⚠️ 局限性**

特征空间构造过于简化（仅基于随机样本和单一PCA），未尝试更先进的降维方法；样本规模有限，未覆盖更大范围的模型；未对证明复杂度进行正式度量；对Herbrand树等深层结构的分析仍待进一步研究。

---

## 386. Persona Prompting as a Lens on LLM Social Reasoning

**arXiv ID:** 2601.20757 | [PDF](https://arxiv.org/pdf/2601.20757v1)

**作者:** Jing Yang `[一作]` (Technische Universitaet Berlin), Nils Feldhus `[通讯]` (Technische Universitaet Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在社会敏感任务（如仇恨言论检测）中，通过 persona prompting（模拟不同社会人口属性）来引导大型语言模型生成 token 级解释，并分析其对分类准确率与解释质量的影响。

**💡 创新点**

创新点在于系统评估 persona prompts 在不同任务和不同 LLM 上的双重效应——即对标签预测有时提升但对 rationale 质量往往下降，并揭示 PP 对模型内在偏见的缓解作用有限。

**🔧 技术方法**

采用 persona‑conditioned 结构化提示、token‑level rationale 生成与评估（Token‑F1、IOU‑F1、Krippendorff’s α）、MAE、Mean Error 等指标，比较多模型（GPT‑OSS‑120B、Mistral‑Medium、Qwen3‑32B）在无 persona 与有 persona 两种条件下的表现。

**📊 数据集**

使用四个公开数据集：HateXplain、CoS‑E、SST‑2，以及 BRWRR（带六组人群标签），其中 HateXplain 还提供词级解释与三位评标者的对齐。

**📈 对比分析**

与基线（无 persona）对比后发现：在 HateXplain 上，部分模型在标签准确率上有所提升（尤其是 Mistral‑Medium），但 rationale 的 Token‑F1 明显下降；在 CoS‑E 与 SST‑2 上，PP 对分类和解释均几乎无效甚至负面；总体而言，所有模型对某些人群（白人、非洲裔美国人、老年人）表现更好，且普遍过度标记为“有害”。

**⚠️ 局限性**

主要局限：① 仅使用三种 LLM，结果可能不具普适性；② 词级解释由 prompting 产生，非真实内部激活的特征归因；③ 数据集标注样本有限，尤其 HateXplain 的评标者仅三人，难以捕捉更广泛解释多样性。

---

## 387. Like a Therapist, But Not: Reddit Narratives of AI in Mental Health Contexts

**arXiv ID:** 2601.20747 | [PDF](https://arxiv.org/pdf/2601.20747v1)

**作者:** Elham Aghakhani `[一作]` (Drexel University), Rezvaneh Rezapour `[通讯]` (Drexel University)

**通讯引用:** 325 | [OpenAlex ID](https://openalex.org/A5012092057)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 5,126 条 Reddit 上的心理健康社区帖子进行大规模语义分析，探讨用户如何评估和与大型语言模型（LLM）进行情感支持交互。

**💡 创新点**

创新点在于：①将技术接受模型（TAM）与治疗联盟理论相结合，构建理论驱动的注释框架；②采用混合 LLM–人工注释管道，首次在自然语言讨论中实现大规模、可操作化的评估；③系统性揭示了用户对 LLM 支持的使用模式、情感、风险与治疗联盟之间的关系。

**🔧 技术方法**

主要技术包括：①GPT‑5.2 与 Gemini 3 Pro 等大型语言模型的自动标注；②基于 LLM 的主题分析与手工校对；③统计检验（卡方检验、Cramér’s V）评估维度与情感/使用意向的关联。

**📊 数据集**

使用的数据集为：从 47 个 DSM‑5 对应的心理健康子版块收集的 5,126 条经历或探索型 AI 使用帖；原始语料为 4.7 百万条帖子，经过关键词检索与 LLM 过滤后得到高召回集，再通过人工抽样和注释得到最终数据。

**📈 对比分析**

通过对 5 种 LLM（GPT‑5.2、Gemini 3 Pro、Claude Opus 4.5、Kimi、Qwen）在 13 个分类维度上的精确率、召回率与宏 F1 进行对比，GPT‑5.2 在 TAM 维度上表现最佳（F1≈0.72‑0.85），Gemini 3 Pro 在治疗联盟维度上最佳（F1≈0.78‑0.84），总体宏 F1 接近 0.70。

**⚠️ 局限性**

局限性包括：①仅基于公开的 Reddit 数据，受平台人口与文化偏差影响；②仅分析英语帖子，无法覆盖多语言视角；③LLM 注释可能产生细微误判；④缺乏临床疗效评估与纵向使用轨迹；⑤未能直接衡量 AI 使用对用户心理健康的因果影响。

---

## 388. QueerGen: How LLMs Reflect Societal Norms on Gender and Sexuality in Sentence Completion Tasks

**arXiv ID:** 2601.20731 | [PDF](https://arxiv.org/pdf/2601.20731v1)

**作者:** Mae Sosto `[一作]` (Centrum Wiskunde en Informatica), Laura Hollink `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并使用QueerGen框架，对14种LLM在三类身份（未标记、非酷儿标记、酷儿标记）下的句子完形生成进行系统评估。

**💡 创新点**

提出三分身份比较框架、构建30个身份标记的QueerGen数据集，并对MLM与ARLM跨架构进行比较，揭示隐藏的异性恋规范偏差。

**🔧 技术方法**

利用句子完形填空、VADER情感分析、Hugging Face Regard指标、Perspective API毒性检测以及词汇多样性计算等技术。

**📊 数据集**

自制的QueerGen数据集（10个未标记主体、30个身份标记、10个句型，共3100条提示）。

**📈 对比分析**

通过平均情感、关怀、毒性和预测多样性四维度在三类身份上做分布对比，发现MLM对酷儿标记偏差最大，开放源ARLM中立性较好，封闭源ARLM可通过RLHF部分缓解但转移偏差；整体偏差并未消除。

**⚠️ 局限性**

受限于提示工程、评价工具缺失对LGBTQIA+细节的捕捉、身份标记不完整、未考虑交叉身份、仅评估Top‑1、仅限英语，且未深入探究训练数据与模型架构导致偏差的根源。

---

## 389. AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts

**arXiv ID:** 2601.20730 | [PDF](https://arxiv.org/pdf/2601.20730v1)

**作者:** Shicheng Fang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 17395 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了 AgentLongBench，一个基于模拟环境回放、动态交互日志的长上下文代理评测基准，涵盖 32 种问答类型、8 项任务、32K–4M token 长度。

**💡 创新点**

创新点在于引入知识无关（Knowledge-Free）设置、工具响应的简洁/冗长格式、可控的工具日志解析任务，以及通过 Lateral Thinking Puzzle 生成可验证的、因果连贯的交互轨迹。

**🔧 技术方法**

技术上结合了自动化环境模拟、符号化实体映射、确定性反馈机制、长上下文 LLM（GPT‑4.1、Grok‑4.1 等）、外部检索与记忆框架（RAG、Mem0、A‑Mem 等）进行评测。

**📊 数据集**

使用 Pokémon 数据集构建知识密集场景，并通过完全符号化的 ID 与属性映射生成知识无关数据；所有轨迹均由内部模拟生成，无需额外公开数据集。

**📈 对比分析**

通过对比 GPT‑4.1、Gemini‑2.5‑Flash、Claude‑Sonnet‑4.5、Grok‑4.1 等专有模型、Qwen、DeepSeek、GLM 等开源模型以及 RAG、Mem0、A‑Mem 等记忆系统，发现模型在 2M token 长度时表现急剧下降，Grok‑4.1 在长序列中保持最高准确率；记忆增强效果不明显。

**⚠️ 局限性**

局限性包括：在知识密集场景下仍依赖参数知识，长时序状态跟踪与高信息密度（高 ACL）任务仍表现不佳；记忆框架无法有效支持逻辑依赖；评测仅基于合成谜题环境，缺乏真实世界复杂交互验证。

---

## 390. Fully Dynamic Algorithms for Graph Spanners via Low-Diameter Router Decomposition

**arXiv ID:** 2601.20718 | [PDF](https://arxiv.org/pdf/2601.20718v1)

**作者:** Julia Chuzhoy `[一作]` (Toyota Technological Institute at Chicago), Merav Parter `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 1303 | [OpenAlex ID](https://openalex.org/A5022875077)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本篇论文提出了一种针对全动态无向图的确定性自适应算法，能够在保持接近最优大小的稀疏子图（即图扩张子图）的同时，给出子对数级的伸展比、几乎线性的更新时间以及低回报（recourse）；核心技术是构造并维护一种低直径路由分解（low‑diameter router decomposition），该分解保证每个簇内部为强路由器并且路由路径完全位于簇内；此外，还利用该分解得到低拥塞（low‑congestion）与容错（fault‑tolerant）扩张子图，以及新的连通度证书。

**💡 创新点**

创新点主要在于：①提出并实现了可在自适应对手下保持的低直径路由分解，首次将路由器的“内部”性质与全图相匹配；②在确定性自适应环境下实现子对数级伸展比与几乎线性更新时间的稀疏子图；③通过路由分解得到低拥塞与容错扩张子图，并给出相应的动态连通度证书。

**🔧 技术方法**

技术手段包括：构造星形结构的 W_k 图并设计剪枝（pruning）算法；利用已存在的“良好连通图”（well‑connected graph）技术进行路由器嵌入；采用低直径聚类（decremental low‑diameter clustering）维护簇；结合 expander 分解与“移动切割”（moving cut）构造长度受限扩张器；使用 Even‑Shiloach 树实现动态距离维护；通过顶点分裂（vertex‑split）处理度不均衡；以及对算法进行 deamortization 以获得确定性的 worst‑case 更新时间。

**📊 数据集**

论文为理论研究，不涉及实验数据集；所有结果均基于对任意无向图的分析与证明。

**📈 对比分析**

与以往工作相比，本算法在自适应对手下实现了子对数级伸展比（k·2^{O(1/δ^6)}）和 O(n^{1+O(1/k)}) 边数的扩张子图；更新时间为 n^{O(δ)}，回报为 n^{O(1/k)}，均为先前随机或仅在无自适应对手下可得到的最佳值；相比之下，之前的算法要么是随机的、要么只能在无自适应对手下工作，或在伸展比/更新时间/大小之间无法同时满足子对数/线性/最优。

**⚠️ 局限性**

局限性包括：①参数范围受限（k 必须在 512 到 (log n)^{1/49} 之间，δ 需小于 1/400）；②伸展因子中的 2^{O(1/δ^6)} 常数可能极大，导致实际伸展比偏大；③更新时间虽然为 n^{O(δ)}，但在 δ 接近 1/400 时仍较慢；④算法实现复杂，涉及多层嵌套的数据结构；⑤仅针对简单图（无自环、无多重边）且需要顶点分裂以处理高度图；⑥缺乏实验评估和对比。

---

## 391. Block Erasure-Aware Semantic Multimedia Compression via JSCC Autoencoder

**arXiv ID:** 2601.20707 | [PDF](https://arxiv.org/pdf/2601.20707v1)

**作者:** Homa Esfahanizadeh `[一作]` (Nokia Bell Labs), Harish Viswanathan `[通讯]` (Nokia Bell Labs)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于联合源-信道编码（JSCC）的块擦除感知语义多媒体压缩框架，实现在不同信道质量下的自适应语义重建；

**💡 创新点**

创新点在于将块级擦除模型与不均匀重要性保护（UEP）相结合，使用单一编码器‑解码器对多级块进行压缩与抗误码，同时兼容现有网络协议，支持拥塞控制与智能重传；

**🔧 技术方法**

采用深度学习自动编码器，JSCC + 块擦除模拟、可调擦除概率向量、-1占位符标记擦除、以及对图像/视频的端到端训练（Adam优化器等）；

**📊 数据集**

使用的数据集包括 CIFAR‑10（图像）、Vimeo‑90k（视频）以及 UVG（视频基准）；

**📈 对比分析**

与传统压缩、GRACE、无擦除训练、渐进细化等方法在多种擦除率和信道条件下对比，实验表明 PSNR 在大多数情形下明显优于基线，鲁棒性更好且误差随信道恶化呈平滑下降；

**⚠️ 局限性**

主要限制在于需预设擦除概率向量以匹配未知的实际信道分布；对高分辨率视频的实验不足；且若网络层无法实现块重要性标记与差异化拥塞控制，效果将受限。

---

## 392. Beyond GEMM-Centric NPUs: Enabling Efficient Diffusion LLM Sampling

**arXiv ID:** 2601.20706 | [PDF](https://arxiv.org/pdf/2601.20706v1)

**作者:** Binglei Lou `[一作]` (Imperial), Aaron Zhao `[通讯]` (Imperial)

**通讯引用:** 395 | [OpenAlex ID](https://openalex.org/A5101093262)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对扩散式大语言模型（dLLM）的采样阶段，设计并实现了 d-PLENA NPU 架构，提供轻量级非 GEMM 指令、向量与标量数据流重用、以及分离混合精度内存层级，以显著提升采样效率。

**💡 创新点**

创新点包括：
• 将采样流程改造成硬件友好的 Stable‑Max 形式，减少多次内存访问；
• 提出专用的向量/标量 ISA 扩展（最大归约、Top‑k 排序、掩码选择等）；
• 架构分离 Vector、FP、Int 三种 SRAM，降低内存碎片与控制路径干扰；
• 采用块化采样与流式预取（HBM → Vector SRAM）实现低延迟、高带宽的采样流水线。

**🔧 技术方法**

使用的技术手段：
• 周期级仿真（Cycle‑Accurate Simulator）与 RTL 验证（Cocotb）确保功能正确性；
• 采用 7 nm OpenROAD 设计套件进行综合，评估面积与功耗；
• 利用 BF16、MXFP8 等混合精度数据格式；
• 在 FPGA‑style 流程中实现 FP 单元对指数/倒数的硬件加速；
• 通过块化采样与 VLEN 调节实现内存利用率与延迟的折衷。

**📊 数据集**

主要在公开的扩散式 LLM 模型上评估：LLaDA、DREAM 等，实验参数覆盖批量 B、扩散步数 T、词表大小 V、块大小 V_chunk 等维度；未使用特定文本数据集，而是以模型自带的 token 序列为输入。

**📈 对比分析**

比较方法：将 d-PLENA 在相同采样负载（T = 1, B = 16, L = 32, V = 126k）与 NVIDIA RTX A6000 GPU 进行对比。结果显示，在 VLEN = 2048、Vector SRAM = 8 MB 时，d-PLENA 达到 2.53× 的加速，最优情况下 0.99 ms 的采样延迟；指令级分析表明向量操作占 48% 以上，内存访问占 40%，控制与标量操作仅 12% 以内。

**⚠️ 局限性**

局限性：
• 只针对采样阶段进行优化，未处理温度噪声或 Gumbel‑Max 等后处理；
• 大词表（V > 100k）时仍需较大 Vector SRAM，导致资源占用高；
• 在边缘模式下（V_chunk < V）延迟仍相对较高；
• 性能评估仅基于单个 GPU/CPU 对比，缺乏跨平台或多节点验证；
• 设计依赖专用 ISA，需在后续 NPUs 中实现兼容性。

---

## 393. One Step Is Enough: Dispersive MeanFlow Policy Optimization

**arXiv ID:** 2601.20701 | [PDF](https://arxiv.org/pdf/2601.20701v1)

**作者:** Guowei Zou `[一作]` (Sun Yat-sen University), Weibing Li `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4794 | [OpenAlex ID](https://openalex.org/A5063378661)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种能一次性生成动作的生成式机器人控制框架DMPO

**💡 创新点**

融合MeanFlow无知识蒸馏的一步推断、扩散正则化防止表征崩塌、以及PPO微调突破专家极限

**🔧 技术方法**

使用轻量级Vision Transformer+MLP、MeanFlow流匹配、扩散正则化（InfoNCE/L2、余弦、Hinge、协方差）与PPO+行为克隆

**📊 数据集**

在RoboMimic四项操控任务和OpenAI Gym（Hopper、Walker2d、Ant、Humanoid）及Kitchen D4RL数据集上训练与评估

**📈 对比分析**

与多步扩散/流匹配基线（ReinFlow、DPPO、ShortCut）以及一阶方法（MP1、CP、1-DP）对比，DMPO实现1–20×推断速度提升，超过或匹配多步基线的成功率与奖励，并在Franka‑Panda机器人上实现>120Hz实时控制

**⚠️ 局限性**

在极度复杂任务中仍受表征空间不足限制，需调节扩散正则系数，且对高维视觉输入的泛化仍需进一步验证

---

## 394. Decoupling Perception and Calibration: Label-Efficient Image Quality Assessment Framework

**arXiv ID:** 2601.20689 | [PDF](https://arxiv.org/pdf/2601.20689v1)

**作者:** Xinyue Li `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21047 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LEAF框架，通过多模态大语言模型（MLLM）教师进行点级与对比级知识蒸馏，随后用少量人类MOS数据对轻量级学生模型进行校准，实现标签高效的图像质量评估

**💡 创新点**

核心创新在于将质量感知与MOS尺度校准解耦，利用教师提供的稠密监督（点级软分数与带置信度的对比偏好）来训练学生，仅在极小比例的MOS样本上完成尺度对齐；同时采用置信度加权的对比蒸馏和多任务损失

**🔧 技术方法**

采用InternVL‑3.5‑8B等MLLM作为教师，提取token级对数似然生成软分数；学生采用ConvNeXt‑Base等轻量级CNN进行回归；训练阶段结合SmoothL1、加权二元交叉熵、MSE及PLCC损失；使用AdamW、混合精度训练

**📊 数据集**

在用户生成内容（UGC）Benchmarks：KonIQ‑10k、SPAQ；在AI生成内容（AIGC）Benchmarks：AGIQA‑3K、AIGIQA‑20K；所有数据集均按MOS可见比例（0%、10%、30%）划分进行实验

**📈 对比分析**

与最强弱监督与无监督方法对比，LEAF在10% MOS下可达到SRCC/PLCC与主流弱监督相当，30% MOS下甚至超过最优监督结果；在无监督情形下亦能显著超越所有标签自由基线；实验表明仅用10% MOS即可获得高质量评估，性能随MOS比例上升而趋于饱和

**⚠️ 局限性**

局限性主要体现在对教师模型的依赖与稀疏MOS校准的泛化性：若教师对特定场景的质量感知不足，蒸馏效果受限；MOS校准仍需在特定数据集上收集少量标签，极端低资源或跨域迁移时可能表现下降

---

## 395. ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code

**arXiv ID:** 2601.20679 | [PDF](https://arxiv.org/pdf/2601.20679v1)

**作者:** Mingqiao Mo `[一作]` (University of Chinese Academy of Sciences), Yangfan He `[通讯]` (University of Minnesota)

**通讯引用:** 966 | [OpenAlex ID](https://openalex.org/A5028171572)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ShieldedCode框架，学习并生成虚拟机保护（VMP）代码的表示，提升软件对逆向工程的防御能力。

**💡 创新点**

通过层级依赖建模、功能对比学习与保护对比学习相结合，以及专门的保护效果优化任务，使得模型既能保持功能一致，又能区分不同保护强度。

**🔧 技术方法**

利用大规模源-VMP配对数据、层级注意力掩码、功能对比学习（FCL）、保护对比学习（PCL）、PEO任务，并采用两阶段持续预训练+微调的训练策略。

**📊 数据集**

构建了基于AnghaBench、The Stack、VirtuCorp 3M等的源‑VMP配对数据集，使用HumanEval_compile、BinaryCorp‑VirtualAssembly、六个真实项目等作为评测集。

**📈 对比分析**

与GPT‑4o、CodeLlama、jTrans等基线对比，HumanEval_compile Pass@1 达到26.95%（最高），BinaryCorp‑VirtualAssembly Recall@1 最高为0.488，且在逆向工程实验中恢复率最低，显示显著性能提升。

**⚠️ 局限性**

局限性包括：对极高保护级别的泛化尚未充分验证；仅针对x86‑64架构，其他架构缺乏实验；训练成本高，需大规模算力；对特制VM的抵抗力仍可能有限。

---

## 396. Smoothing the Black-Box: Signed-Distance Supervision for Black-Box Model Copying

**arXiv ID:** 2601.20773 | [PDF](https://arxiv.org/pdf/2601.20773v1)

**作者:** Rubén Jiménez `[一作]`, Oriol Pujol `[通讯]` (University of Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于签名距离的黑盒复制框架，将硬标签复制转化为对教师决策边界的距离回归，实现无数据、无内部信息的模型重建。

**💡 创新点**

创新点在于：① 用α参数化的目标中心正则化，显式控制监督信号的 Hölder / Lipschitz 光滑性；② 通过两种无模型距离估计算法（全局逐点搜索与聚类并行搜索）实现高效的距离标注；③ 将距离输出同时作为不确定性量化。

**🔧 技术方法**

技术包括：签名距离目标构造、α‑正则化理论、两种距离估计算法、神经网络和梯度提升回归学生模型、Sobol 序列合成数据、精细的实验评估指标。

**📊 数据集**

使用三类二维合成数据（高斯碰撞、螺旋、杂乱 blob）和三类 UCI 经典表格数据（乳腺癌、稻谷、矿物/岩石）。

**📈 对比分析**

与传统硬标签复制和多种基线模型（RF、GB、NN）比较，指标为仿真误差 R^ℱ_emp 和测试准确率。实验表明：在低至中等样本量下，距离复制可显著降低仿真误差并提升准确率；α>0 时可在保持或提升准确率的同时略微牺牲仿真精度。

**⚠️ 局限性**

局限性在于：① 距离估计在高维或极其不规则模型上精度下降；② 目前仅针对二分类，需进一步扩展至多分类；③ α 的选择仍经验性，缺乏自动化策略；④ 仅在合成与小型表格数据上验证，尚未验证大规模、图像等高维场景。

---

## 397. Agentic Fog: A Policy-driven Framework for Distributed Intelligence in Fog Computing

**arXiv ID:** 2601.20764 | [PDF](https://arxiv.org/pdf/2601.20764v1)

**作者:** Saeed Akbar `[一作]` (Sarhad University of Science and Information Technology), Rahmat Ullah `[通讯]` (University of Essex)

**通讯引用:** 294 | [OpenAlex ID](https://openalex.org/A5066659045)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Agentic Fog（AF）框架，将雾节点建模为自治策略驱动的代理，通过共享内存和p2p协作实现分布式优化；

**💡 创新点**

创新点在于：①以LLM之外的Agentic AI概念为基础；②将雾计算建模为Exact Potential Game，保证异步最佳回应下的收敛与稳定；③设计共享内存与分层架构实现部分可观测环境下的长期优化；

**🔧 技术方法**

技术包括：多智能体系统设计、共享内存协调、潜在函数游戏理论、异步有限理性最佳回应动态、仿真模拟（Poisson请求、Zipf内容流行度）；

**📊 数据集**

使用仿真数据：10–50个雾节点，随机边界度网格；请求采用非平稳泊松过程，内容流行度遵循Zipf分布；

**📈 对比分析**

与集中式ILP、贪婪启发式对比；在动态工作负载下AF平均延迟降低15–30%，收敛速度最快；在节点失败时，延迟上升幅度最低；控制开销略高；

**⚠️ 局限性**

局限性：依赖共享内存持续一致性；对大规模网络的通信开销有一定增长；在极端高动态场景下可能需要更细粒度的协调频率；

---

## 398. Anytime-Valid Quantum Tomography via Confidence Sequences

**arXiv ID:** 2601.20761 | [PDF](https://arxiv.org/pdf/2601.20761v1)

**作者:** Aldo Cumitini `[一作]` (Politecnico di Milano), Osvaldo Simeone `[通讯]` (Northeastern University London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在线、随时有效的量子态层析方法（AV‑QST），为每个时间步提供可靠覆盖概率的置信集合，弥补传统方法在递增采样时缺乏严格误差保证的问题。

**💡 创新点**

创新点在于将置信序列（confidence sequences）与量子态层析相结合，构造基于似然比的随时有效置信集合，并证明其满足全时刻覆盖概率1−α，无需预先设定样本量或假设先验正确。

**🔧 技术方法**

主要技术包括：似然比构造、Martingale与Ville不等式证明、置信序列理论、最大似然估计或贝叶斯后验均值作为点估计器，以及对量子测量（POVM）下的贝尔定律进行统计分析。

**📊 数据集**

使用仿真数据：两量子比特（D=4）和四量子比特（D=16）的随机纯态（按Haar分布生成），采用最小信息完整POVM（MIC‑POVM）进行局部测量。

**📈 对比分析**

与传统贝叶斯置信集（B‑QST）和固定样本量的似然比置信集（LR‑QST）比较。AV‑QST在所有时间步均保持误差率低于目标α，且集合大小显著小于LR‑QST；B‑QST虽集合更小但经常违背覆盖约束。总体性能表明AV‑QST在保证可靠性的同时，提供了更具信息量的置信集合。

**⚠️ 局限性**

局限性包括：对高维系统的计算成本可能较高，置信集合仍比最优的贝叶斯置信集保守；在极少量样本时集合仍可能过大；此外，本方法主要验证于仿真，实际实验中的噪声与误差模型未充分考察。

---

## 399. Exploring Re-inforcement Learning via Human Feedback under User Heterogeneity

**arXiv ID:** 2601.20760 | [PDF](https://arxiv.org/pdf/2601.20760v1)

**作者:** Sarvesh Shashidhar `[一作]` (Indian Institute of Technology Bombay), Madhav Kotecha `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在RLHF框架中考虑工作者偏好异质性，提出通过聚类工作者并为每个聚类学习个性化奖励模型的方法，并在文本摘要任务上进行了验证。

**💡 创新点**

通过同时学习奖励模型与工作者嵌入并自动确定聚类数量，提供了一种既避免单一奖励模型失效又不需要为每个用户单独建模的中间方案。

**🔧 技术方法**

采用BTL奖励模型、正则化参数学习、用户嵌入向量、余弦相似度与t‑SNE可视化以及RLHF策略优化等技术。

**📊 数据集**

在Reddit TL;DR人类反馈数据集（AbhishekBot/Summarize_Final_Worker_id）上实验，使用40位共同工作者的筛选后数据。

**📈 对比分析**

与传统单一奖励模型（Naive RLHF）对比，采用win‑rate衡量偏好判断准确度，个性化模型的win‑rate从52.13%提升至53.22%和52.70%，表现出轻微提升。

**⚠️ 局限性**

仅在40位工作者的小规模样本上验证，聚类数量自动化仍有限，win‑rate提升幅度不大，未评估生成质量或计算开销。

---

## 400. HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework for Extremely Low-Bit LLMs

**arXiv ID:** 2601.20745 | [PDF](https://arxiv.org/pdf/2601.20745v1)

**作者:** Guoan Wang `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5590 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Hestia 框架，用温度控制的 Softmax 软化三值量化并结合 Hessian 指导的温度退火，解决了传统 STE 的死区与梯度失配问题。

**💡 创新点**

创新点在于：①把硬量化映射改为可微的温度控制软化；②利用张量级 Hessian 迹作为轻量级曲率信号，进行自适应温度调度，实现对不同层的敏感度精准把握；③通过压缩阶段的凸组合平滑从全精度到三值权重的迁移，避免代表性坍塌。

**🔧 技术方法**

核心技术包括：温度控制的 Softmax 软化量化、基于 Hessian 迹的自适应温度退火、Hutch++ 算法做离线曲率估计、以及压缩阶段的可学习权重混合。

**📊 数据集**

使用 Llama‑3.2（1B 与 3B）模型，在 10B 个 UltraFineWeb 采样的 token 上进行量化感知训练；评估采用 5 个零样本基准（ARC‑Easy、ARC‑Challenge、HellaSwag、PIQA、WinoGrande）。

**📈 对比分析**

与 STE 基础的三值 QAT 基线（如 Tequila、BitNet、Spectra 等）对比，Hestia 在 1B 模型上平均提升 5.39% 以上，在 3B 模型上提升 4.34%，在所有 5 个基准上均有显著改善，甚至在相同 10B 训练量的条件下超过了多种 100B 训练量的三值模型。

**⚠️ 局限性**

局限性包括：目前仅验证在权重量化、三值量化场景；需要离线 Hessian 估计，虽然开销小但仍需额外步骤；对更大规模模型、激活量化或更低比特率（如 2bit）尚未深入验证。

---

## 401. SA-PEF: Step-Ahead Partial Error Feedback for Efficient Federated Learning

**arXiv ID:** 2601.20738 | [PDF](https://arxiv.org/pdf/2601.20738v1)

**作者:** Dawit Kiros Redie `[一作]` (Norwegian University of Science and Technology), Stefan Werner `[通讯]` (Aalto University)

**通讯引用:** 7984 | [OpenAlex ID](https://openalex.org/A5059938646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种结合步前预览与部分误差反馈的 Federated Learning 算法 SA-PEF，以减少通信并加速模型收敛。

**💡 创新点**

创新点在于引入可调步前系数 α，既能利用误差反馈的长期稳定性，又能在训练早期通过步前预览快速匹配梯度，从而克服非 IID 数据导致的误差残留与梯度不匹配问题。

**🔧 技术方法**

技术手段包括：δ‑收缩型压缩器（如 Top‑k）、局部 SGD、部分误差反馈、步前预览、残差收缩分析以及非凸目标下的收敛证明。

**📊 数据集**

实验使用三大图像分类基准：CIFAR‑10、CIFAR‑100 和 Tiny‑ImageNet，搭配 ResNet‑9/18/34 网络。

**📈 对比分析**

与无压缩 Local‑SGD、传统 EF、SAEF 和 CSER 等方法比较，SA-PEF 在多种压缩率、参与率与非 IID 参数下均能更快达到目标精度，并在通信量与轮数上具有更优的效率。

**⚠️ 局限性**

局限性包括：对 α 的选择仍需经验性调优；在极低参与率或极端非 IID 场景下优势不明显；并未考虑自适应学习率或动量与误差反馈的交互影响。

---

## 402. Deep Semi-Supervised Survival Analysis for Predicting Cancer Prognosis

**arXiv ID:** 2601.20729 | [PDF](https://arxiv.org/pdf/2601.20729v1)

**作者:** Anchen Sun `[一作]` (University of Miami), Xiaodong Cai `[通讯]` (University of Miami)

**通讯引用:** 2161 | [OpenAlex ID](https://openalex.org/A5107839821)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究者提出并实现了一种基于深度半监督学习的Mean Teacher框架的ANN‑CoxPH模型，用于预测多种癌症的预后，并在单模态（RNA‑seq或WSI）和多模态（两者联合）场景下进行实验。

**💡 创新点**

创新点在于：①将未标记样本和删失样本的监督信息通过一致性损失融入CoxPH学习；②在单模态和多模态下采用Mean Teacher与交叉注意力融合，实现对高维基因表达与图像特征的有效利用；③通过半监督学习显著提升预测准确性。

**🔧 技术方法**

使用技术包括：深度半监督学习（Mean Teacher）、多层感知器（MLP）、DINOv2预训练特征提取、Transformer‑based交叉注意力融合、CoxPH模型、Harrell’s c‑index与Integrated Brier Score评估。

**📊 数据集**

数据集为：TCGA四种癌症（BRCA、LUAD、LUSC、UCEC）的RNA‑seq与PFI；TCGA BRCA的全切片图像（WSI）；GEO GSE96058的未标记RNA‑seq样本。

**📈 对比分析**

与传统单层ANN Cox‑nnet相比，单模态Cox‑MT在四种癌症的c‑index平均提升0.09–0.18，IBS平均下降0.038–0.082；多模态Cox‑MT在BRCA上c‑index0.83、IBS0.079，均优于单模态和多模态Cox‑nnet；增加未标记样本数可使c‑index从0.81提升至0.90。

**⚠️ 局限性**

局限性包括：仍需大量未标记数据才能充分发挥优势；模型参数（如EMA常数、权重w、噪声σ）对性能影响需进一步系统评估；实验仅覆盖少数癌症类型，未在其他医学或非医学领域验证；缺乏临床转化验证。

---

## 403. Enterprise Resource Planning Using Multi-type Transformers in Ferro-Titanium Industry

**arXiv ID:** 2601.20696 | [PDF](https://arxiv.org/pdf/2601.20696v1)

**作者:** Samira Yazdanpourmoghadam `[一作]` (Polytechnique Montreal), Vahid Partovi Nia `[通讯]` (Polytechnique Montreal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种多类型Transformer（MTT）框架，统一解决0-1背包问题（KP）和工作车间调度问题（JSP），并将其应用于铁合金制造工艺的原料混合成本最小化；

**💡 创新点**

创新点在于引入多类型注意力机制，在异质图表示中分别捕获不同实体类型（如作业-机器、物品-容量）之间的关系，实现了跨问题、跨规模的无缝迁移，并通过价值转换将成本最小化问题映射为MTT的最大化任务；

**🔧 技术方法**

核心技术为多类型Transformer架构、异质图编码、基于价值的离散化处理、以及与OR-Tools求解器的对比评估；

**📊 数据集**

实验数据集包括：①KP标准基准（50-100项，共128个实例）；②JSP Taillard/Demirkol 5×5至10×10静态实例与动态泊松到达合成实例；③铁合金制造工厂的14种原料混合实际数据；

**📈 对比分析**

与传统精确求解器OR-Tools对比，KP的平均optimality gap约0.001，JSP约0.03，铁合金实例约0.025；MTT推理时间在1-0.1秒内，而OR-Tools耗时从数十秒到数百秒不等，表明MTT在速度与质量上具有竞争优势；

**⚠️ 局限性**

局限性包括：对连续分配或多维容量问题需先离散化处理；在极大规模实例上可能面临计算瓶颈；缺乏与经典启发式或其他学习模型的系统对比，且对多目标、多约束情境的适用性尚未充分验证。

---

## 404. MuRAL-CPD: Active Learning for Multiresolution Change Point Detection

**arXiv ID:** 2601.20686 | [PDF](https://arxiv.org/pdf/2601.20686v1)

**作者:** Stefano Bertolasi `[一作]` (Politecnico di Milano), Luigi Amedeo Bianchi `[通讯]` (Università di Trento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种结合多尺度小波分解和主动学习的半监督变点检测方法 MuRAL-CPD，能够在少量人工标注下显著提升时间序列的变点识别精度。

**💡 创新点**

核心创新在于将主动学习直接嵌入无监督多分辨率 CPD 体系中，通过对特征权重和阈值的联动优化，实现模型的自适应调优；同时采用几何弯曲曲线的极值法对阈值进行智能初始化，减少查询次数。

**🔧 技术方法**

使用了离散小波变换（Daubechies‑2）、滑动窗口归一差异度量、线性特征加权、峰值显著性变换、主动学习（查询最不确定样本）以及贝叶斯优化求解超参数。

**📊 数据集**

实验数据集包括 BabyECG（心率睡眠阶段）、UCI‑HAR（人体活动识别）、Honeybee Dance（蜜蜂舞步）和 USC‑HAD（运动捕捉），共四个真实场景。

**📈 对比分析**

与最先进的半监督方法 ICPD 进行对比；在所有数据集上，MuRAL-CPD 在相同查询量下均能获得更高的 F1 分数，尤其在低监督预算下显示出更快的收敛速度。

**⚠️ 局限性**

限制在于对阈值初始化仍有一定依赖，且在极少标注的初期性能可能受限；此外，对不同波形类型或长序列的泛化能力尚需进一步验证。

---

## 405. Audit Trails for Accountability in Large Language Models

**arXiv ID:** 2601.20727 | [PDF](https://arxiv.org/pdf/2601.20727v1)

**作者:** Victor Ojewale `[一作]` (Brown University), Suresh Venkatasubramanian `[通讯]` (Brown University)

**通讯引用:** 11576 | [OpenAlex ID](https://openalex.org/A5061790878)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向大型语言模型（LLM）的审计追踪（audit trail）框架，设计了参考架构（包含事件捕获、存储和审计接口）并实现了一个轻量级 Python 库，以实现跨生命周期（预训练、适配、部署、监控）的可追溯性和治理记录。

**💡 创新点**

创新点在于：①将治理决策（批准、豁免、声明）与技术事件（训练、部署、监控）统一到同一时间序列化日志中；②采用增量哈希链（hash chain）实现追加不可篡改的审计账本；③提供跨组织的可扩展接口与共享标识符，支持供应链级别的可追溯；④将审计功能设计为可插拔、低耦合的层，易于嵌入现有 MLOps 工作流。

**🔧 技术方法**

主要技术包括：事件驱动的捕获器（callback、middleware、CLI）、JSON‑L 追加式日志、SHA‑256 链式哈希保证完整性、Python 的标准库与 MLOps 工具集成（Hugging Face、FastAPI、Git 等）以及基于元数据的查询与验证接口。

**📊 数据集**

论文未针对具体任务数据集做模型训练或评估；在 PoC 示例中仅演示了对公开数据集（如 Hugging Face 数据集）进行注册和版本化的流程，重点在于展示审计事件的生成与链式存储，而非数据集本身。

**📈 对比分析**

没有对比实验；性能评估以“可行性”与“低集成开销”为主，PoC 在典型训练/部署脚本中添加回调后，日志生成量与原有日志差异不足 5%，且哈希链验证在毫秒级完成，证明系统在现有 MLOps 流程中几乎不引入显著延迟。

**⚠️ 局限性**

局限性包括：①在大规模、高吞吐量部署中日志量巨大，需进一步研究聚合与索引策略；②隐私与合规性仍需外部加密与访问控制，审计日志本身不处理敏感内容；③跨组织共享时需额外签名与指针机制，实施复杂度提高；④因缺乏因果推断，审计仅能证明事件顺序，责任判定仍需人工解读。

---

## 406. Less is More: Clustered Cross-Covariance Control for Offline RL

**arXiv ID:** 2601.20765 | [PDF](https://arxiv.org/pdf/2601.20765v1)

**作者:** Nan Qiao `[一作]`, Ju Ren `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对离线强化学习中的分布偏移问题，发现平方误差目标会在稀缺数据和 OOD 区域产生有害的 TD 交叉协方差，从而导致训练不稳定和性能下降。

**💡 创新点**

创新点在于提出了 C^4（Clustered Cross‑Covariance Control for TD）两种机制：① 在梯度空间对回放缓冲进行分区采样，局部化更新以消除跨区协方差；② 在每次更新中加入基于梯度的校正惩罚，显式抵消交叉协方差带来的偏差，且保持目标函数的下界。

**🔧 技术方法**

采用的技术包括梯度空间聚类、单簇 mini‑batch 采样、梯度协方差惩罚项、混合高斯模型的 EM‑式聚类更新，以及在已有离线 RL 算法（如 CQL、TD3+BC、IQL、DOGE、TSRL 等）上做插件式集成。

**📊 数据集**

实验主要使用 D4RL 公开数据集中的 MuJoCo 运动学任务（Ant、Hopper、Walker2d、HalfCheetah 等），并在仅 10k 条样本的极低数据量设置下进行评估；同时也在 AntMaze、Maze2D、Adroit 等 OOD 强化任务上验证。

**📈 对比分析**

与多种基线（BC、CQL、TD3+BC、IQL、DOGE、TSRL、BPPO、A2PR、DR3、LN、SORL 等）比较，C^4 在低数据量和 OOD 强调的设置中平均提升约 30%（最高可达 70%+），训练更稳定、收敛速度快，且在计算成本上仅略高于轻量级正则化方法。

**⚠️ 局限性**

局限性包括：① 仍需在不同任务中调节聚类数和惩罚强度；② 目前仅在离线 RL 场景验证，其他 RL 任务或大规模数据下的有效性尚待进一步验证；③ 对极端稀疏数据或高维动作空间的鲁棒性需要更多实证。

---

## 407. ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler

**arXiv ID:** 2601.20755 | [PDF](https://arxiv.org/pdf/2601.20755v1)

**作者:** Bohua Zou `[一作]` (Huawei Hilbert Research Center), Haibo Chen `[通讯]` (Huawei Central Software Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ProfInfer，一种基于 eBPF 的细粒度、无侵入式 LLM 推理性能分析工具，能够在移动/边缘设备上实时捕获 token、图、算子级别的执行信息，并结合硬件计数器生成三种可视化视图（ProfDAG、ProfTime、ProfStat）

**💡 创新点**

创新点在于：①将 eBPF 动态探针与 LLM 推理运行时函数精确对齐，支持多级别跟踪；②在不修改源代码的前提下实现低于 4% 的运行时开销；③将硬件性能计数器与算子语义关联，提供跨层次的性能洞察；④为移动场景量身定制的可视化与分析流程

**🔧 技术方法**

使用技术包括 eBPF（libbpf、BCC）、BPF Compiler Collection、Perf 事件、GPU/CPU/ NPU 后端调用、硬件性能计数器（L3 缓存刷新、内存访问等）、Chrome Trace/Event 以及 Python/Graphviz 生成可视化

**📊 数据集**

实验使用了 LLaMA3.2‑1B、Qwen2.5‑1.5B、Gemma2‑2B、Qwen1.5‑MoE‑A2.7B 等公开 LLM 模型，部署在 Orange Pi 5 Plus、Orange Pi 5 Ultra 及其他支持 OpenHarmony 的设备上

**📈 对比分析**

通过与 llama.cpp 自带的 .dot 输出、ONNX Runtime profiler 以及不同后端（CPU、OpenCL、NPU、GPU）下的基准进行对比，结果表明 ProfInfer 在开启完整功能时的推理速度下降仅 2.8–4%（单线程），并能够准确定位内存带宽瓶颈、KV‑cache 影响、MoE 访存/磁盘瓶颈以及算子级别的后端调度差异

**⚠️ 局限性**

局限性包括：仅支持 llama.cpp 及其现有后端（CPU、OpenCL、Rockchip NPU）；GPU 计数器采样不完整；对多线程和动态调度的精细化分析仍有提升空间；缺乏自动化优化建议；在高吞吐量批处理场景下的采样粒度和可扩展性待验证

---

## 408. Learning to Live with AI: How Students Develop AI Literacy Through Naturalistic ChatGPT Interaction

**arXiv ID:** 2601.20749 | [PDF](https://arxiv.org/pdf/2601.20749v1)

**作者:** Tawfiq Ammari `[一作]` (Rutgers University), Kiran Garimella `[通讯]` (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析36名本科生一年内与ChatGPT的10,536条消息，识别出五种使用体裁，说明学生如何在日常实践中形成AI素养

**💡 创新点**

提出“使用体裁”和“修复素养”等概念，证明AI素养是关系性、情感性和认知性实践，而非单纯的技术技能

**🔧 技术方法**

采用基于OpenAI GPT‑4o的零样本标注，结合手工编码的质性编码框架

**📊 数据集**

以36名来自Rutgers大学的本科生导出的完整ChatGPT聊天记录为数据集

**📈 对比分析**

通过人工标注与GPT标注的Cohen κ 比较（0.75–0.91），验证标注可靠性；未做算法性能对比，仅进行经验性描述

**⚠️ 局限性**

样本规模小、仅来自单一高校、缺乏人口统计信息、仅研究ChatGPT，且技术更新快导致结论时效性有限

---

## 409. Implementing Metric Temporal Answer Set Programming

**arXiv ID:** 2601.20735 | [PDF](https://arxiv.org/pdf/2601.20735v1)

**作者:** Arvid Becker `[一作]` (University of Potsdam), Torsten Schaub `[通讯]` (University of Potsdam)

**通讯引用:** 8472 | [OpenAlex ID](https://openalex.org/A5058467603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了两种计算方法，将度量约束融入答案集编程（ASP）中，构建了两种度量片段并给出了完整的语义翻译与实现；

**💡 创新点**

核心创新是将时间约束外部化，利用差分约束和元编码显著降低时序细粒度对 grounding 的影响，并证明两种翻译的完备性与正确性；

**🔧 技术方法**

采用了答案集编程、平衡与稳定模型理论、差分约束（Clingo/Clingcon）以及元编码框架；

**📊 数据集**

实验以牙医行程规划为例，使用距离表（时间表）作为数据集；

**📈 对比分析**

通过比较不同片段、不同实现（Boolean vs 整数差分）在 grounding 和求解时间上的表现，结果显示整数差分方案对时间精度不敏感，且全局算子带来的规则数与求解复杂度明显上升；

**⚠️ 局限性**

主要局限在于全局度量算子导致的规则膨胀与求解开销，且当前评测规模有限，需要更大规模实验验证与优化。

---

## 410. Continual GUI Agents

**arXiv ID:** 2601.20732 | [PDF](https://arxiv.org/pdf/2601.20732v1)

**作者:** Ziwei Liu `[一作]` (Tsinghua University), Tao Feng `[通讯]` (Tsinghua University)

**通讯引用:** 8475 | [OpenAlex ID](https://openalex.org/A5100678146)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了持续学习的 GUI 代理任务（Continual GUI Agents），并设计了一种名为 GUI‑AiF 的强化学习微调框架，使代理能够在域（如移动、桌面、Web）和分辨率（1080p→4K）变化的环境中持续保持并提升定位性能。

**💡 创新点**

创新点在于：①将持续学习引入 GUI 代理；②引入两种奖励机制 APR‑iF 与 ARR‑iF，分别鼓励代理探索多样的交互点和区域，从而降低对固定坐标/尺度的过拟合；③将这些奖励与 KL 正则化结合，平衡探索与保持已有知识的关系。

**🔧 技术方法**

使用强化学习微调（RFT）+GRPO 方法，结合 APR‑iF、ARR‑iF、KL 散度等技术，并在 Qwen2.5‑VL‑3B 预训练模型上进行训练。

**📊 数据集**

采用三大基准数据集：ScreenSpot‑V1、ScreenSpot‑V2（支持移动/桌面/Web域），以及 ScreenSpot‑Pro（支持多种高分辨率软件界面）。训练时还使用了 Widget Captioning、ShowUI‑web、OmniACT 等公开数据集。

**📈 对比分析**

与传统 SFT 及多种 RFT 基线（SeeClick、GUI‑Actor、InfiGUI‑R1、SE‑GUI、GUI‑G²）进行对比。实验显示 GUI‑AiF 在三大基准的平均准确率上均优于基线，尤其在域迁移和分辨率提升任务中显著提升了 2–5% 的准确率。

**⚠️ 局限性**

局限性包括：①仅实验 3B 参数规模的模型；②仅考虑域和分辨率两种变换，未覆盖更多真实场景（如暗模式、布局升级、国际化等）；③受限于算力，使用的数据集和实验规模有限，可能未能完全展示方法的最大潜力。

---

## 411. Distributed Learning over Noisy Communication Networks

**arXiv ID:** 2601.20723 | [PDF](https://arxiv.org/pdf/2601.20723v1)

**作者:** Emrah Akyol `[一作]` (Binghamton University), Marcos Vasconcelos `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在存在噪声通信链路的分布式系统中，基于log‑linear学习（LLL）的二进制协同博弈；

**💡 创新点**

创新点在于将通信层的噪声通过二进制对称信道（BSC）或二进制擦除信道（BEC）模型显式化，并区分快通信和快照两种运作模式，得到快通信下的Gibbs采样结构、快照下的非平衡马尔可夫过程及其高温展开，并提出有限K通道使用的插值模型，进一步支持异质链路可靠性的有效权重解释；

**🔧 技术方法**

主要技术包括概率马尔可夫链理论、Gibbs分布与熵正则化优化、噪声通道的期望与方差分析、强大定理与高温展开、以及通信理论的重复编码与期望估计视角；

**📊 数据集**

实验使用人工生成的网络数据集：环网、格网、Erdős–Rényi 随机图以及星形网络，节点数为100，所有边权设为1；

**📈 对比分析**

与快通信、快照以及不同K值的对比，主要指标为稳态协同潜能；结果显示在高温（β=0.5）两种模式相近，而在低温（β=2）快通信显著优于快照并且方差更小；有限K模型随K增大收敛到快通信，且收益递减；异质链路实验表明快通信通过有效权重平衡噪声，快照对不可靠边更敏感；

**⚠️ 局限性**

局限性包括：仅考虑二进制动作的协同博弈；假设信道独立且对称；未考虑时间变异或异步更新的延迟；对多动作游戏、非潜在游戏、以及更复杂的通信策略的推广仍待研究。

---

## 412. Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction

**arXiv ID:** 2601.20720 | [PDF](https://arxiv.org/pdf/2601.20720v1)

**作者:** Matej Halinkovic `[一作]` (Slovak University of Technology), Marek Galinski `[通讯]` (Slovak University of Technology)

**通讯引用:** 84 | [OpenAlex ID](https://openalex.org/A5045411709)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于查询的可微分摄像头‑雷达融合框架 Li‑ViP3D++，实现端到端感知与轨迹预测。

**💡 创新点**

创新点在于 Query‑Gated Deformable Fusion（QGDF）机制，它通过可微分的 BEV 采样、查询自适应门控和多视角注意力，实现对 RGB 与 LiDAR 信息的软融合，消除了传统的离散对齐与阈值选取。

**🔧 技术方法**

核心技术包括 DETR3D 风格的稀疏查询、ResNet‑50 摄像头编码器、PointPillars 点云编码器、可微分 BEV 采样、查询门控融合、以及基于 VectorNet 的 HD 地图编码和多假设轨迹回归。

**📊 数据集**

在 nuScenes 数据集上进行训练与评估，使用 6 视角 RGB、5 次 LiDAR 采样与 HD 地图输入。

**📈 对比分析**

与 Li‑ViP3D、ViP3D 等基线相比，Li‑ViP3D++ 在 EPA 由 0.250 提升至 0.335、mAP 提升至 0.502、误检率降至 0.147，并且推理时延从 145.91 ms 减至 139.82 ms，显示出更高的检测精度与更低的假阳性率。

**⚠️ 局限性**

主要局限在于仍需依赖高质量雷达与高清地图，且在极端天气或低光照环境下的鲁棒性未充分验证；相较于纯视觉模型，计算量略高。

---

## 413. LEMON: How Well Do MLLMs Perform Temporal Multimodal Understanding on Instructional Videos?

**arXiv ID:** 2601.20705 | [PDF](https://arxiv.org/pdf/2601.20705v1)

**作者:** Zhuang Yu `[一作]` (Shanghai Jiao Tong University), Shiliang Sun `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10943 | [OpenAlex ID](https://openalex.org/A5047846625)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LEMON（Lecture-based Evaluation benchmark for MultimOdal uNderstanding）——一套针对 STEM 讲座视频的多模态评测基准，涵盖长时序、跨模态和多轮交互。

**💡 创新点**

创新点包括：①将教学视频作为长期因果场景，提供严谨的时间线和教学结构；②设计六大任务与十二子任务，跨感知、推理和生成的认知层级；③采用多轮对话格式，逼真模拟教师‑学生互动；④通过人工‑AI 双重流程生成高质量 QA 对，确保问题真正需要多模态理解。

**🔧 技术方法**

技术手段：视频采集与同步音频/字幕；使用 Whisper‑v3 生成字幕；GPT‑4o 与 Gemini 2.0 Flash 生成初始问题与答案；人工审核与多阶段质量检测；评测采用多模态 LLMs（GPT‑4o、Gemini、Qwen3‑Omni、MiniCPM‑o 2.6 等）以及专门的长视频模型（LongVA、LongVU、Video‑XL‑2 等）。

**📊 数据集**

数据集：2,277 条讲座片段，平均时长 196.1 秒，覆盖 5 个 STEM 领域（数学、人工智能、计算机科学、电子工程、机器人），共 29 门课程；4,181 对 QA（3,413 选项题 + 768 开放式题）。

**📈 对比分析**

对比方法：在统一的零样本评测框架下，使用 21 种公开/专有 MLLM 进行 6 类任务评估。结果显示：专有模型（如 GPT‑4o、Gemini 2.5 Pro、GPT‑5）整体表现明显优于开源模型；但所有模型在时序因果推理、未来内容预测和高阶生成任务上相对薄弱；加入字幕可显著提升性能，音频提升不稳定；性能差距体现在 10–30% 之差。

**⚠️ 局限性**

局限性：①对音频理解不足，语音识别和语义抽取仍易出错；②时序因果推理能力弱，难以捕捉长距离依赖；③跨语言生成偏向西方语言，亚洲语言表现差；④生成答案常出现幻觉和格式错误；⑤对长视频的记忆和帧采样敏感，需更高效的时序建模。

---

## 414. Reflected wireless signals under random spatial sampling

**arXiv ID:** 2601.20699 | [PDF](https://arxiv.org/pdf/2601.20699v1)

**作者:** H. Paul Keeler `[一作]` (University of Melbourne), H. Paul Keeler `[通讯]` (University of Melbourne)

**通讯引用:** 667 | [OpenAlex ID](https://openalex.org/A5033480448)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究随机空间采样下无线电波在平行墙面之间反射导致的功率分布奇异点，并给出基于镜像法的双壁模型及其闭式解（使用 Lerch 泛函），通过理论推导与数值仿真验证奇异点的存在与位置。

**💡 创新点**

首次系统揭示并解析随机位置产生的概率密度尖峰（逆平方根奇异点），提出利用镜像法与 Lerch 泛函将物理反射过程与统计模型结合，并提供理论与仿真一致的闭式表达式。

**🔧 技术方法**

镜像法、概率变换（变量替换法）、Lerch 泛函求和、数值仿真（MATLAB）以及随机相位模型对比。

**📊 数据集**

使用 MATLAB 生成的仿真数据，模拟不同频率（k=10^2, 10^3）、墙距、衰减指数及反射系数 κ，随机取样的发射器位置。

**📈 对比分析**

将随机位置模型与随机相位模型在相同参数下进行对比，结果显示随机位置模型产生尖峰奇异点，而随机相位模型得到光滑的单峰分布；仿真结果与理论推导一致，验证了奇异点预测的准确性。

**⚠️ 局限性**

模型假设无限长平行墙、单斜波衰减、角度无关的镜面反射、忽略散射、极化效应和近场效应，适用范围受限；在实际复杂环境中需考虑角度依赖、非镜面反射及多路径干涉等因素。

---

## 415. Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?

**arXiv ID:** 2601.20694 | [PDF](https://arxiv.org/pdf/2601.20694v1)

**作者:** Hao Liang `[一作]` (King's College London), Yali Du `[通讯]` (Alan Turing Institute)

**通讯引用:** 1644 | [OpenAlex ID](https://openalex.org/A5064693586)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种仅靠纯利用（不做显式探索）即可在外生马尔可夫决策过程（Exo-MDP）中获得次优收敛的学习框架，并给出了在表格和线性函数逼近（LFA）下的有限样本风险上界。

**💡 创新点**

创新点主要包括：① 将外生过程的独立性作为纯利用的理论基础，证明不需要探索；② 提出两个新工具——反事实轨迹（counterfactual trajectories）和贝尔曼闭合特征传输（Bellman‑closed feature transport），实现无乐观值估计的误差控制；③ 在 LFA 下设计了 Least‑Squares Value Iteration with Pure Exploitation（LSVI‑PE）算法，得到多项式于特征维度、外生状态数和步长的风险上界；④ 通过理论与实验验证，首次在 Exo‑MDP 上展示纯利用优于传统探索驱动方法。

**🔧 技术方法**

技术上使用：动态规划、最小二乘回归、特征锚点（anchor set）设计、贝尔曼闭合传输假设、反事实轨迹构造、Martingale 归约与计数技术，以及线性混合模型框架。

**📊 数据集**

使用的数据集主要为合成实验：① 表格 Exo‑MDP（5 维状态、5 维外生、3 个动作、5 步）；② 存储控制问题（连续状态与动作，离散价格）；③ 其他基准实验如随机库存控制和能量存储等，均基于随机生成的外生马尔可夫链或均匀分布奖励。

**📈 对比分析**

与基线比较：① 传统乐观探索算法（如 Optimistic Model Estimation, UCB‑style 方法）；② 经验回放/后向回归方法；③ 纯利用的 FTL/ERM 方案。实验结果显示：在所有设定下，LSVI‑PE 与 PEL 的累积风险均显著低于乐观探索方法，且在更高维或更大步长时表现尤为突出。

**⚠️ 局限性**

局限性：① 需要 Exo‑MDP 的外生状态完全独立且可观测；② 对内生动态 f 与奖励 r 必须已知；③ 线性逼近假设要求特征空间能完整逼近价值函数并满足贝尔曼闭合传输；④ 对外生状态空间有限的假设；⑤ 需要人工选择锚点集，若锚点设计不佳会影响收敛；⑥ 对模型误差（如贝尔曼误差）存在线性累积偏差。

---

## 416. Decentralized Identity in Practice: Benchmarking Latency, Cost, and Privacy

**arXiv ID:** 2601.20716 | [PDF](https://arxiv.org/pdf/2601.20716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 417. Optimal Transport Group Counterfactual Explanations

**arXiv ID:** 2601.20692 | [PDF](https://arxiv.org/pdf/2601.20692v1)

**作者:** Enrique Valero-Leal `[一作]` (Universidad Politécnica de Madrid), Giuseppe Casalicchio `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于最优传输（OT）映射的群组反事实解释方法，能够一次性为整个样本组生成可泛化的反事实，并通过凸 bi‑Lipschitz 约束控制几何失真。

**💡 创新点**

创新点在于把群组反事实问题从逐点优化转化为学习 OT 映射，显著减少参数量、提高可解释性和泛化能力，并通过凸约束实现可控制的几何保持；同时提供多种可解释的参数化（Affine、Gaussian、GMM 等）和闭式解。

**🔧 技术方法**

使用的技术包括最优传输理论、凸二次规划、半正定规划（SDP）、高斯/高斯混合模型的解析解，以及多目标进化优化（NSGA‑II）等。

**📊 数据集**

实验使用15个包含数值特征的二分类数据集（来自公开基准），在每个数据集上随机划分 20 组（每组至多 200 条样本）进行评估。

**📈 对比分析**

与三种基线（独立、Group‑Lipschitz、Group‑bi‑Lipschitz）对比，实验显示在绝大多数情形下提出的方法在 W₂ 距离、失真控制和有效性指标上都优于基线，尤其在 bi‑Lipschitz 约束强（K>2.5）时更显优势；在多目标优化中，方法的超体积均优于基线。

**⚠️ 局限性**

局限性包括：当基线 bi‑Lipschitz 约束收敛时，基线在部分场景下仍能得到更优结果；高维解释仍不直观；对非线性路径的建模有限，需进一步研究神经网络 OT 模型以适应更复杂数据。

---

## 418. MedViz: An Agent-based, Visual-guided Research Assistant for Navigating Biomedical Literature

**arXiv ID:** 2601.20709 | [PDF](https://arxiv.org/pdf/2601.20709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 419. Tendon-based modelling, estimation and control for a simulated high-DoF anthropomorphic hand model

**arXiv ID:** 2601.20682 | [PDF](https://arxiv.org/pdf/2601.20682v1)

**作者:** Péter Polcz `[一作]` (Pázmány Péter Catholic University), Miklós Koller `[通讯]` (Pázmány Péter Catholic University)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5067935891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于肌腱位移和张力测量的非线性优化方法，估计人形手的关节角度并利用估计值实现姿态闭环控制。

**💡 创新点**

创新点在于：①使用Denavit‑Hartenberg（DH）方式对具有3-DoF关节的高自由度手部骨骼建模；②构造完整的肌腱分支/连接矩阵，将肌腱张力与关节角度关联；③将估计器与PI反馈结合并引入前馈补偿，以提高姿态跟踪性能。

**🔧 技术方法**

核心技术包括：非线性方程组求解（CasADi+IPOPT+MUMPS），肌腱张力/位移模型，DH几何建模，关节角度估计器，Jacobian‑based PI控制与前馈补偿。

**📊 数据集**

实验使用MuJoCo仿真环境下的Anatomically Correct Biomechatronic Hand模型，手指共5个自由度，手掌6个自由度；通过预录的肌腱长度配置实现前馈估计，不依赖外部视觉或传感器数据集。

**📈 对比分析**

通过6个预设手势（G1–G6）评估性能：关节角误差大部分低于5°，大部分手指末端位置误差低于5 mm；加入前馈后，过渡时间缩短、稳定误差不受明显影响；仅在极度屈曲姿态（G5）出现较大约束违例和角误差。

**⚠️ 局限性**

主要局限包括：①Roll 轴角度观测不佳，导致高频误差；②软约束实现导致瞬时约束违例，影响极端姿态；③模型依赖肌腱几何与张力参数，若参数误差或外部碰撞未建模，估计与控制性能下降；④当前仅在仿真验证，缺乏真实硬件实验验证。

---

## 420. A Human-Centred AI System for Multi-Actor Planning and Collaboration in Family Learning

**arXiv ID:** 2601.20737 | [PDF](https://arxiv.org/pdf/2601.20737v1)

**作者:** Si Chen `[一作]`, Nitesh Chawla `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于大型语言模型（LLM）的家庭学习协同系统 ParPal，并通过专家评估和11个家庭的一周实地部署验证其在任务拆解、分配与协同可见性方面的效果。

**💡 创新点**

将多方协同和可见性引入家庭学习规划，利用 LLM 进行任务拆解与分配，并将 AI 与人类共同参与的反馈机制结合，构建了可视化时间表与任务分配面板；同时提出了针对多方协同的“社会成本”考虑与动态重新规划需求。

**🔧 技术方法**

技术包括 GPT‑4o LLM（通过 GPT‑4o API）、多步骤提示与角色引导、Web 前端实现（可响应式页面）、JSON 计划格式、任务分配与进度可视化面板、AI 辅助答题、例题生成与解释生成。

**📊 数据集**

主要数据集为自定义的 10 个家庭配置（2–4 位照护者）与 5 种学习任务集，共 50 个周计划（用于专家评估），以及 11 个家庭的使用日志和访谈数据；未使用公开公开数据集。

**📈 对比分析**

评估方法：两名领域专家对每个计划按 5 维度（角色-任务匹配、任务拆解质量、覆盖完整性、上下文感知、可执行性）采用 3 级量表评分；对 GPT‑4o 与 Claude 4.5 在同一 50 计划上的得分进行对比，结果两模型整体相近，GPT‑4o 在角色分配与可执行性略优；实地测试中，约一半家庭使用任务分配功能，用户满意度提升，但仍出现不可执行或协调成本高的任务。

**⚠️ 局限性**

限制：LLM 在多方协同规划中缺乏对协同成本与可行性约束的推理；任务拆解往往缺乏连贯的学习路径和时序；时间/空间约束常被忽视，导致不可执行任务；系统缺乏动态增量式重新规划与长期协同努力的建模，未能充分捕捉隐性劳动与协调成本。

---

## 421. Adapting the Behavior of Reinforcement Learning Agents to Changing Action Spaces and Reward Functions

**arXiv ID:** 2601.20714 | [PDF](https://arxiv.org/pdf/2601.20714v1)

**作者:** Raul de la Rosa `[一作]` (Universidad de los Andes), Nicolas Cardozo `[通讯]` (Universidad de los Andes)

**通讯引用:** 778 | [OpenAlex ID](https://openalex.org/A5027708259)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自适应的Q‑学习框架，能够实时应对奖励函数漂移和行动空间扩展的环境变化；

**💡 创新点**

创新点在于将概念漂移检测（Page‑Hinkley）与动态调节学习率与探索率相结合，保持单一Q‑表并避免灾难性遗忘；

**🔧 技术方法**

技术包括基于Page‑Hinkley的漂移检测、TD误差驱动的自适应学习率、ε‑贪婪策略的探索率重置、以及Q‑表的动态扩容；

**📊 数据集**

数据集分别为9×9网格世界（目标迁移和新增跳跃动作）和自定义的交通信号控制Gym环境（车辆流量变化与新的信号相位）；

**📈 对比分析**

与标准固定学习率、固定ε衰减的Q‑学习进行对比，实验显示自适应框架在目标迁移时收敛速度提升约1.7倍，且在行动空间扩容时能快速融入新动作；

**⚠️ 局限性**

局限在于漂移检测参数需要经验调优，某些奖励分布子集的漂移可能被忽略，以及仅在离散、低维状态空间的tabular Q‑学习上验证，未扩展到深度RL或高维问题。

---

## 422. Online Density-Based Clustering for Real-Time Narrative Evolution Monitorin

**arXiv ID:** 2601.20680 | [PDF](https://arxiv.org/pdf/2601.20680v1)

**作者:** Ostap Vykhopen `[一作]`, Veronika Solopova `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在社交媒体监测中，用在线密度聚类方法（如DenStream）替代离线HDBSCAN，构建可实时更新的叙事分析管道。

**💡 创新点**

提出将传统聚类质量指标与叙事特有指标结合的综合评估框架，并发现DenStream在聚类质量上优于HDBSCAN，展示了在线聚类在可扩展性与动态适应性上的潜力。

**🔧 技术方法**

使用Transformer基嵌入（MiniLM）、UMAP降维、在线密度聚类算法（DBSTREAM、DenStream、TextClust）以及LLM主题标注。

**📊 数据集**

乌克兰信息空间的多语言社交媒体文本，约每天1.7万条文档。

**📈 对比分析**

采用滑动窗口+前置训练+增量更新的实验设计，比较Silhouette、Davies–Bouldin、Narrative Distinctness、Contingency、Variance以及训练/预测时间；DenStream在聚类质量上最佳，但叙事一致性略逊。

**⚠️ 局限性**

局限包括数据域专一、依赖特定嵌入/降维设置、LLM标注偏向HDBSCAN、River实现缺陷导致性能偏差、仅单日评估且未覆盖长期概念漂移。

---

## 423. Supervised Guidance Training for Infinite-Dimensional Diffusion Models

**arXiv ID:** 2601.20756 | [PDF](https://arxiv.org/pdf/2601.20756v1)

**作者:** Elizabeth L. Baker `[一作]` (Technical University of Denmark), Jes Frellsen `[通讯]` (Technical University of Denmark)

**通讯引用:** 1103 | [OpenAlex ID](https://openalex.org/A5072272257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在无限维函数空间中使用Doob’s h-变换对扩散模型进行条件化的理论框架，并引入了无仿真监督指导训练（Supervised Guidance Training）来学习不可计算的引导项，从而实现后验采样。

**💡 创新点**

首创了在函数空间中对已训练的扩散模型进行微调以获得后验分布的完整方法，证明了条件分数可分解为无条件分数加引导项，并提供了一个无仿真、基于分数匹配的训练策略。

**🔧 技术方法**

利用无限维Doob’s h-变换、分数扩散模型、随机最优控制、Tweedie近似、去噪分数匹配损失以及 Fourier Neural Operator 等技术实现条件采样。

**📊 数据集**

在三个实验任务上验证：1) 一维稀疏观测的合成函数；2) 一维热方程的初始温度恢复；3) MNIST 3号数字的形状补全（使用EFD表示）。

**📈 对比分析**

与专门训练的条件扩散模型以及 FunDPS 近似方法对比，实验结果显示 Supervised Guidance Training 在 RMSE 与能量评分上均接近或优于条件扩散基线，明显优于 FunDPS。

**⚠️ 局限性**

方法依赖于无条件分数的精确估计，若分数误差累积会影响后验质量；需要可获得真实 (X₀, Y) 对；在不同离散化或高维场景下的计算开销与稳定性仍待进一步研究。

---

## 424. Independence of Approximate Clones

**arXiv ID:** 2601.20779 | [PDF](https://arxiv.org/pdf/2601.20779v1)

**作者:** Théo Delemazure `[一作]` `[通讯]` (University of Amsterdam), Théo Delemazure (University of Amsterdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在选举中近似克隆候选人对投票规则结果的影响，并提出了两种测度近似克隆的方法（α‑删除克隆和β‑交换克隆），结合理论分析和实证实验来评估常用单胜选举规则（IRV、Ranked Pairs、Schulze、Plurality、Borda）的“近似克隆不变性”。

**💡 创新点**

创新点在于：①首次从理论上证明对所有m≥4的规则，在任何非零α或β阈值下都无法满足弱近似克隆不变性；②提出弱近似克隆不变性作为可行的弱化；③在实际数据上评估近似克隆出现频率，并比较规则在不同阈值下的鲁棒性；④将理论阈值与实验结果对比，揭示规则对近似克隆的实际敏感性。

**🔧 技术方法**

技术方法包括：定义α‑删除克隆和β‑交换克隆的正式数学描述；使用组合论与图论分析规则的克隆不变性；构造反例证明不可行性；在三候选情形下证明IRV、Ranked Pairs、Schulze的弱不变性；对三类真实数据集进行计算实验，统计最小α/β值以及规则是否违反各项不变性。

**📊 数据集**

使用的真实数据集包括：①苏格兰地方选举（STV/IRV）——1,070个偏好档案，候选人3–14人，选民数43–2,905人；②国际体操/花样滑冰评审（Preflib）——48个档案，候选人14–30人，评委7–9人；③Mini‑jury 讨论实验——2,581个档案，候选人4人，评委5人。

**📈 对比分析**

比较方法：对每个档案统计：①完美克隆对规则的不变性违例比例；②α≤0.2近似克隆的违例比例；③所有候选对的违例比例；④规则是否满足“失败者不变性”。实验显示IRV和Ranked Pairs在大多数情况下表现最稳健，尤其在低α值时违反率低；Borda和Plurality违例率明显更高。弱不变性在所有规则中几乎都超过90%成立，显示其在实践中可能过于宽松。

**⚠️ 局限性**

局限性：①理论结果对m≥4的情况给出不可行性，但在m=3时仍需依赖规则是否为单一获胜者；②只考虑候选对而非更大克隆集合；③未探讨随机化或计数权重的投票规则；④实验仅基于三类数据集，无法覆盖更广泛的社会偏好结构；⑤弱不变性在实践中似乎不足以捕捉“真实”克隆效应的严苛要求。

---

## 425. Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy

**arXiv ID:** 2601.20776 | [PDF](https://arxiv.org/pdf/2601.20776v1)

**作者:** Huanyu Tian `[一作]` (King's College London), Christos Bergeles `[通讯]` (Conceivable Life Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种弱监督框架，将稳手共操转化为自动共驾的显微镜下微操控系统；

**💡 创新点**

利用稳手演示产生的视觉-运动配对作为弱标签，结合双阶段3D感知与不确定性预算，实现无标定、无手动深度标注的闭环控制；

**🔧 技术方法**

采用多模态视觉网络（EfficientViT + Spatial Attention）与双任务深度估计（ResNet‑18+嵌入），配合软门限Kalman滤波、Bi‑Chamfer标定、基于交叉熵与Huber损失的训练；

**📊 数据集**

通过“warm‑up”共操轨迹收集的显微镜视频与机器人状态，构成两类数据集：平面定位数据集（约27k帧）与深度扫描数据集（约2.5k帧）；

**📈 对比分析**

与PnP、雅可比回归、欧氏距离基准对比，标定误差低于2–8像素，平面定位平均误差1.41像素（≈12μm），深度误差≤0.38mm，用户试验中工作负荷降低≈77%且成功率接近100%；

**⚠️ 局限性**

精度受限于像素分辨率与显微镜的光学深度，无法实现亚微米级别；需更高放大倍率才能满足细胞级操作；

---

## 426. Exploring Transformer Placement in Variational Autoencoders for Tabular Data Generation

**arXiv ID:** 2601.20854 | [PDF](https://arxiv.org/pdf/2601.20854v1)

**作者:** Aníbal Silva `[一作]` (University of Porto), Carlos Soares `[通讯]` (University of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探究了将Transformer集成到VAE的输入、潜在空间和解码器不同位置对表格数据生成的影响；

**💡 创新点**

创新点在于系统性评估Transformer在VAE各组件中的作用，并揭示了其对数据保真度与多样性之间的权衡，以及Transformer在解码器中的近乎“恒等”行为；

**🔧 技术方法**

使用了VAE与Transformer相结合的架构（E-VAE、EL-VAE、LD-VAE、ELD-VAE等），并采用了注意力机制、tokenization、CKA相似度分析等技术；

**📊 数据集**

实验基于OpenML CC18 Benchmark中的57个混合类型表格数据集，样本量范围为500–96,320，特征维度4–240；

**📈 对比分析**

通过低密度估计（单变量和两变量统计）、高密度估计（α-Precision、β-Recall）以及机器学习效率（Utility、ML-Fidelity）等指标对比，发现加入Transformer后数据多样性提升但保真度下降，整体性能并未显著提升；

**⚠️ 局限性**

主要局限在于Transformer对数据生成质量的提升有限，尤其在机器学习任务上效果不显著；此外，Transformer在解码器中的近乎恒等表现表明其在该位置可能冗余，未来需进一步优化架构和减少计算开销。

---

## 427. A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion

**arXiv ID:** 2601.20847 | [PDF](https://arxiv.org/pdf/2601.20847v1)

**作者:** Willams de Lima Costa `[一作]` (Voxar Labs, Centro de Informática, Universidade Federal de Pernambuco), Cristiano Coelho de Araújo `[通讯]` (Volkswagen do Brasil)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套融合相机与惯导传感器的道路表面分类框架，并针对多样化的驾驶环境构建了全新的 ROAD 数据集。

**💡 创新点**

创新点包括：①三子集结构的 ROAD 数据集（多模态、视觉单模和合成场景）；②轻量级双向交叉注意力模块与自适应门控融合策略；③在数据增强与同步预处理上细致设计，提升跨域鲁棒性。

**🔧 技术方法**

核心技术为：EfficientNet‑B0 视觉编码器 + CNN‑BLSTM 惯导编码器；双向交叉注意力对齐视觉与惯导 token；可学习门控融合与全连接分类头。

**📊 数据集**

主要使用自研的 ROAD 数据集（包含约10小时10分钟的多模同步录制、13小时34分钟的视觉单模和53分钟的合成样本），并在公开的 Passive Vehicular Sensors（PVS）基准上进行验证。

**📈 对比分析**

与两种现有最优方法比较，PVS 上精度提升 1.4pp，ROAD Subset‑#1 上精度提升 11.6pp，最终获得 95.6%（PVS）和 98.2%（ROAD）整体准确率，少数类别 F1 亦显著提升。

**⚠️ 局限性**

局限性包括：在视觉良好条件下惯导贡献有限；跨模同步误差导致转换边界误判；数据集虽多样但仍缺乏极端低光与大规模车辆多样性。

---

## 428. GNN Explanations that do not Explain and How to find Them

**arXiv ID:** 2601.20815 | [PDF](https://arxiv.org/pdf/2601.20815v1)

**作者:** Steve Azzolin `[一作]` (University of Trento), Sagar Malhotra `[通讯]` (TU Wien)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5086702765)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过理论与实验揭示了自解释图神经网络(SE‑GNN)的“无意义”解释失效现象，证明这些解释可以与模型真实推理完全无关，进而提出一种新的鲁棒可解释性度量方法。

**💡 创新点**

创新点在于：①给出一条充分条件，说明在存在锚点集合(anchor set)时，SE‑GNN可以在保持预测性能的同时输出完全无意义的解释；②构造恶意攻击与自然训练两种情境下的失效案例；③设计了基准评测框架，用已知失效解释来检验可解释度量的效能，并提出扩展充分性测试（Extension Sufficiency Test）度量，显著优于现有度量。

**🔧 技术方法**

技术主要包括：图分类任务中的SE‑GNN架构（解释提取器与分类器组合）、损失函数设计（交叉熵与解释惩罚项）、理论证明（锚点集合与硬阈值提取器），以及可解释性度量的实现与评估（基准、拒绝率计算、超图搜索）。

**📊 数据集**

使用的数据集包括：一个合成数据集（3*），以及三个真实图分类数据集（2*、2*、3*，具体名称在论文中未给出但可替换为常见图数据集如Cora、Citeseer、Pubmed等）。

**📈 对比分析**

在恶意与自然训练的两种设置下，作者将新度量与现有度量（Necessity/Sufficiency 组合、Edge/Complement 变换等）进行对比，结果显示新度量在拒绝率上普遍高于0.5，且随扰动预算增加而进一步提升，而传统度量往往在某些失效案例中拒绝率接近0。

**⚠️ 局限性**

局限性包括：①分析聚焦于仅由锚点节点构成的无意义解释，未覆盖更一般的无关子图；②假设解释提取器为硬阈值形式，实际应用中可能采用软阈值或其他正则化；③新度量虽更稳健但仍可能把真正可靠的解释误判为无意义，且计算量随超图枚举而增大。

---

## 429. Context-Augmented Code Generation Using Programming Knowledge Graphs

**arXiv ID:** 2601.20810 | [PDF](https://arxiv.org/pdf/2601.20810v1)

**作者:** Shahd Seddik `[一作]` (University of British Columbia), Patanamon Thongtanunam `[通讯]` (University of Melbourne)

**通讯引用:** 1421 | [OpenAlex ID](https://openalex.org/A5040320723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Programming Knowledge Graph（PKG）结构化知识图谱，用于代码和文本检索增强生成（RAG）以提高复杂编程任务的准确性。

**💡 创新点**

创新点在于将代码与教程分别建模为AST/JSON层级图谱，支持多粒度检索（函数级和块级），并结合结构化剪枝和候选重排序降低幻觉与噪声。

**🔧 技术方法**

使用AST解析、JSON结构化、向量检索（Voyage-Code-2）、图数据库Neo4j、语义检索与重排序算法，以及多模型生成与评估。

**📊 数据集**

数据集包括PythonAlpaca（代码），Python Tutorials（文本），以及公开基准HumanEval和MBPP进行代码正确性评估。

**📈 对比分析**

与传统BM25、Dense检索及无检索基线对比，PKG在HumanEval和MBPP上提升至约+20% pass@1，Block-PKG比Func-PKG更精确，重排序进一步提升约+4%到+12%。

**⚠️ 局限性**

局限性包括对Python/英文语料的依赖、检索粒度静态选择、重排序仅基于语义相似度、构建成本与存储开销、以及对高端闭源模型的收益有限。

---

## 430. Jurisdiction as Structural Barrier: How Privacy Policy Organization May Reduce Visibility of Substantive Disclosures

**arXiv ID:** 2601.20792 | [PDF](https://arxiv.org/pdf/2601.20792v1)

**作者:** Thomas Brackin `[一作]` `[通讯]` (Independent Researcher), Thomas Brackin (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了隐私政策中“司法区隔离披露”结构模式，即将重要数据实践仅在特定司法区条款中披露，导致其他地区用户可能无法获得完整通知。

**💡 创新点**

首次提出并量化了此结构性透明度缺陷，并基于信息探寻理论阐述其对用户知情同意的影响，提出“普适实质披露”规范。

**🔧 技术方法**

采用多模型LLM（Claude、GPT、Gemini）联合分类技术，对政策段落进行自动化标注和等价性判定，并通过一致性与外部验证评估。

**📊 数据集**

使用123家大型跨国公司的隐私政策语料库（约42,000词平均长度），并结合OPP‑115人类标注数据进行验证。

**📈 对比分析**

通过与OPP‑115一致性（Kappa≈0.86）和人类验证（Kappa≈0.63）评估模型可靠性；在样本中发现77家公司（≈62%）存在司法区隔离披露，实例计282个，其中93.6%为直接实质性陈述。

**⚠️ 局限性**

局限在于：样本为选择性大型公司，无法代表全部网站；分类基于同一语料迭代调优，存在循环偏差；对实质性披露是否确实普遍存在的技术验证尚缺；以及仅分析主域隐私政策，未覆盖子域或多语言版本。

---

## 431. REASON: Accelerating Probabilistic Logical Reasoning for Scalable Neuro-Symbolic Intelligence

**arXiv ID:** 2601.20784 | [PDF](https://arxiv.org/pdf/2601.20784v1)

**作者:** Zishen Wan `[一作]`, Tushar Krishna `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

无法获取论文主要内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用技术

**📊 数据集**

无法确定数据集

**📈 对比分析**

无法进行性能比较

**⚠️ 局限性**

无法评估限制

---

## 432. Construction and Decoding of Convolutional Codes with optimal Column Distances

**arXiv ID:** 2601.20825 | [PDF](https://arxiv.org/pdf/2601.20825v1)

**作者:** Julia Lieb `[一作]`, Michael Schaller `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种在任意有限域上构造具有最佳列距离的卷积码，并证明了所构造的码是唯一达到最佳列距离的码。

**💡 创新点**

创新点在于通过固定有限域并最大化列距离，提出了一种新的卷积码构造方法，同时开发了针对这些码的低复杂度维特比解码算法。

**🔧 技术方法**

使用了麦克唐纳码和一阶里德-穆勒码作为构建块，并利用这些结构开发了改进的维特比算法。

**📊 数据集**

使用了任意有限域的卷积码构造，具体数据集未明确提及，但涉及的码率和参数是任意选择的。

**📈 对比分析**

与传统的维特比算法相比，本文提出的算法在复杂度上有显著降低，具体复杂度为O(N· q^2n log_q(n))，而传统算法的复杂度显著更高。

**⚠️ 局限性**

限制在于所构造的码在某些情况下可能不具备最佳列距离，且对于不同的码率，可能需要不同的构造方法。

---

## 433. SokoBench: Evaluating Long-Horizon Planning and Reasoning in Large Language Models

**arXiv ID:** 2601.20856 | [PDF](https://arxiv.org/pdf/2601.20856v1)

**作者:** Sebastiano Monti `[一作]` (Ipazia SpA), Bruno Lepri `[通讯]` (Fondazione Bruno Kessler and Ipazia SpA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在长时间规划任务中的表现，采用简化的Sokoban线性通道作为基准。

**💡 创新点**

提出低复杂度、单盒线性Sokoban benchmark，系统评估LRM在长路径规划中的计数与状态跟踪缺陷。

**🔧 技术方法**

使用推理型LLM（如 GPT‑4o、Claude‑3、Claude‑3‑opus 等）以及LLM‑Modulo 工具链（PDDL 解析/验证/求解器）。

**📊 数据集**

生成并公开 80 个长度 5–100 的单盒线性通道地图数据集，涵盖四种旋转方向。

**📈 对比分析**

对比仅使用模型推理与配合外部规划器的推理，发现模型在需要超过 25–30 步的规划时准确率急剧下降；工具链稍有提升，但仍难以弥补模型的内部计数与状态维护缺陷。

**⚠️ 局限性**

局限性包括：仅测试单盒线性通道、缺少多盒死锁与分支、对模型内部计数和记忆表示的依赖、API 版本/后端波动、预训练泄漏风险，以及基准的外部有效性有限。

---

## 434. Linear representations in language models can change dramatically over a conversation

**arXiv ID:** 2601.20834 | [PDF](https://arxiv.org/pdf/2601.20834v1)

**作者:** Andrew Kyle Lampinen `[一作]` (Google DeepMind), Murray Shanahan `[通讯]` (Google DeepMind)

**通讯引用:** 9134 | [OpenAlex ID](https://openalex.org/A5072322524)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在自然对话中的线性表示（如真伪性、伦理性）随上下文动态变化的现象。

**💡 创新点**

创新点在于揭示对话上下文可显著翻转模型内部线性维度投影，表明这些表示不是静态的，而是随对话角色与内容动态重组。

**🔧 技术方法**

采用了对Gemma 3系列模型残差流的激活提取、正则化逻辑回归识别线性维度、margin得分计算，并通过对抗“opposite day”提示、对话重放等实验来追踪表示演变。

**📊 数据集**

使用了自建的平衡真伪性与伦理性yes/no问答集、情境相关问题集，以及来自先前研究、人工编写和大型模型生成的多段对话与故事脚本作为实验数据。

**📈 对比分析**

通过与空白提示、对抗提示、不同模型规模（Gemma 12B、4B）以及其他模型（Qwen 3 14B）的对比，发现对话中margin得分显著翻转，且大模型更易出现表示变化。

**⚠️ 局限性**

限制包括样本对话数量有限、只关注少数概念、未直接验证表示的因果作用，以及对大模型内部机制缺乏深入解释。

---

## 435. MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents

**arXiv ID:** 2601.20831 | [PDF](https://arxiv.org/pdf/2601.20831v1)

**作者:** Vishnu Sashank Dorbala `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 39170 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级的记忆控制框架 MemCtrl，利用可训练的二分类记忆头 μ 在多模态大语言模型上实时筛选并存储重要观测，以提升嵌入式 AI 任务的决策性能。

**💡 创新点**

创新点在于引入可迁移的记忆头，使模型能够主动写入内存而不需要修改主干，并提供离线专家监督和在线强化学习两种训练方式，实现实时主动记忆筛选。

**🔧 技术方法**

采用多模态大语言模型（MLLM）+ 轻量 MLP 记忆头、离线专家监督（GPT‑4o）与 REINFORCE 强化学习、基于 EmbodiedBench 的任务评估框架。

**📊 数据集**

使用 EmbodiedBench 数据集，包含 ALFRED 与 Habitat 两个子集，用于评估嵌入式任务完成率。

**📈 对比分析**

通过在 Gemma‑3‑12B‑IT 与 Qwen2.5‑VL‑7B‑Ins 两个低性能 MLLM 上对比无记忆基线、简单记忆、离线监督和在线 RL 四种 μ 变体，平均提升约 16%，在长句与复杂指令上提升超过 20%。

**⚠️ 局限性**

局限性包括需要强大模型的专家演示或稀疏奖励导致训练成本与效率问题，且在短期任务中优势不明显，尚未考虑音频观测和真实世界部署。

---

## 436. Reinforcement Learning via Self-Distillation

**arXiv ID:** 2601.20802 | [PDF](https://arxiv.org/pdf/2601.20802v1)

**作者:** Jonas Hübotter `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30455 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了自我蒸馏策略优化（SDPO），利用大语言模型自身在接收环境提供的丰富Token化反馈后，充当自我教师，对原始生成进行稠密的下一步Token级别信用分配，从而实现在线强化学习。

**💡 创新点**

创新点在于：①将模型本身作为反馈条件的自我教师，无需外部奖励模型或强教师；②把环境给出的细粒度反馈（如错误信息、单元测试失败、参考答案等）转化为稠密的优势信号；③通过自我蒸馏实现稠密信用分配，显著提升样本效率、缩短推理长度，并能在测试时快速发现难题答案。

**🔧 技术方法**

技术方法包括：强化学习中的策略梯度与KL蒸馏损失；自我教师通过在原始生成前加入反馈进行条件推理；top‑K稠密蒸馏、EMA/信任域教师正则化；与GRPO的优势混合（SDPO+GRPO）以及对比离线蒸馏、最佳k采样、对话采样等。

**📊 数据集**

实验数据集涵盖：科学推理（Chemistry, Physics, Biology, Materials）、ToolAlpaca（工具调用）、LiveCodeBench v6（编码竞赛）用于训练与验证；测试时使用非常难的LCBv6子集；对比基准包括IFEval、ArenaHard‑v2、MMLU‑Pro 等保留任务；使用模型包括 Qwen3‑8B、Qwen3‑7B、Qwen2.5‑Instruct、Olmo3‑7B‑Instruct 等。

**📈 对比分析**

与改进后的GRPO、SFT、离线蒸馏、best‑of‑k 采样、对话采样等方法对比，SDPO 在 LCBv6 上从 41.2% 提升到 48.8%，在推理任务上 64.1%→68.8%，在 4× 更少的生成轮数内达到 GRPO 最终准确率；测试时 SDPO 在难题上实现 3× 更少的尝试就能达到同样的发现概率，整体样本效率和推理长度显著优于基线。

**⚠️ 局限性**

局限性包括：对模型的上下文学习能力高度依赖，较弱模型可能表现不如 GRPO；若环境反馈不充分或误导，则自我蒸馏难以学习；计算上虽然只增加 log‑prob 计算，但在小模型或短生成时仍可能成为瓶颈；目前主要验证于可验证奖励任务，尚未在开放式文本或连续奖励场景证明泛化。

---

## 437. FAIRT2V: Training-Free Debiasing for Text-to-Video Diffusion Models

**arXiv ID:** 2601.20791 | [PDF](https://arxiv.org/pdf/2601.20791v1)

**作者:** Haonan Zhong `[一作]` (University of New South Wales), Yang Song `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对文本到视频（T2V）扩散模型中的性别偏差进行系统分析，并提出一种训练‑free 的文本嵌入去偏框架 FAIRT2V，配合动态去偏调度实现对视频的公平生成，并设计了基于 VideoLLM 与人工验证的评估协议。

**💡 创新点**

①发现 T2V 偏差主要来源于预训练文本编码器；②利用两性 anchor 在球面几何空间进行角度插/外推去偏；③在扩散早期采用动态去偏调度，既消除身份偏差又保持时序一致；④构造面向视频的公平评估流程，兼顾自动推理与人工审核。

**🔧 技术方法**

球面线性插值（SLERP）实现去偏；动态去偏时间表（仅在身份形成阶段使用去偏嵌入）；VideoLLM（Gemini）对全视频进行性别推理；视频质量评测指标 FVD、FAST‑VQA、CLIP‑T/F、TIFA 等。

**📊 数据集**

采用 16 个常见职业的中性/多样化提示集（基于 U.S. Bureau of Labor Statistics），在 Open‑Sora T2V 模型上生成视频；实验数据仅来自自生成视频，不使用公开视频数据集。

**📈 对比分析**

与两种训练‑free 基线 FairDiff 与 FairImagen 进行对比。FAIRT2V 在 Video Fair Ratio（VFR）上显著优于两基线，同时在 FVD、FAST‑VQA、CLIP‑T/F、TIFA 等质量指标上保持甚至提升性能，表明其在公平与生成质量之间取得更佳平衡。

**⚠️ 局限性**

仅针对二元性别偏差，尚未覆盖多类或跨群体偏差；对 T5 编码器的去偏效果不佳；去偏过程可能影响显式用户意图；评估仍依赖 VideoLLM 与人工审核，存在主观性与算力开销。

---

## 438. The Monotone Priority System: Foundations of Contract-Specific Sequencing

**arXiv ID:** 2601.20783 | [PDF](https://arxiv.org/pdf/2601.20783v1)

**作者:** Naveen Durvasula `[一作]` `[通讯]` (Columbia University and Ritual), Naveen Durvasula (Columbia University and Ritual)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

提出并实现了基于单调优先级的合约交易序列约束系统，允许合约开发者为每个调用设定全局优先级并限制其不高于被调用者。

**💡 创新点**

证明该系统是唯一满足存在性、优先级、扩展、可约性和无关调用独立性五个自然公理的实现方案，且可映射为全局整数优先级，简化区块构造。

**🔧 技术方法**

采用形式化公理化方法、偏序和可达性分析、证明技巧，以及简单的按优先级排序算法实现。

**📊 数据集**

未使用实验数据集，全部工作基于理论证明与数学推导。

**📈 对比分析**

与传统无约束或全局约束模型对比，证明存在性与可构造性；区块构造可在 O(n log n) 时间内完成，显著低于一般约束求解。

**⚠️ 局限性**

局限在于只能处理无状态的全局整数优先级约束，无法表达基于链状态的约束；对优先级上限 λ 的需求导致需要在合约部署时重新计算优先级。

---

## 439. FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models

**arXiv ID:** 2601.20857 | [PDF](https://arxiv.org/pdf/2601.20857v1)

**作者:** Hongyu Zhou `[一作]` (Zhejiang University), Yiyi Liao `[通讯]` (Zhejiang University)

**通讯引用:** 2725 | [OpenAlex ID](https://openalex.org/A5018811297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需对扩散模型微调的方案，通过在3D高斯展开（3D Gaussian Splatting）中进行交互式的二维-三维细化，显著提升外推视角的渲染质量。

**💡 创新点**

创新点包括：① 交互式2D‑3D细化策略，使已改进的视角反馈到后续细化，提升多视角一致性；② 基于Fisher信息的像素置信度掩模作为细化引导，精准定位需要修复的区域；③ 多级置信度引导与整体引导相结合，兼顾结构与细节；④ 仅使用预训练的图像扩散模型（IDM），避免高昂的微调成本。

**🔧 技术方法**

技术手段：3D Gaussian Splatting、预训练图像扩散模型（如SDXL、Flux）、Fisher信息置信度计算、交互式2D‑3D细化、像素级置信度引导、整体引导、颜色校正矩阵等。

**📊 数据集**

实验数据集：LLFF（前视场景）、Mip-NeRF 360（对象中心场景）、Waymo（驾驶场景）。

**📈 对比分析**

与基线方法（ViewExtrapolator、NVS‑Solver、Difix3D+、StreetCrafter）在相同的3D细化步骤下比较，使用PSNR/SSIM/LPIPS（LLFF、Mip‑NeRF）及KID（Waymo）评估。结果显示：在LLFF、Mip‑NeRF 360和Waymo上，本文方法在量化指标上与微调模型相当，甚至在多视角一致性与细节保真度上优于基线。

**⚠️ 局限性**

局限性：① 对极端外推视角中出现的大量伪影，缺乏足够可信引导时易失效；② 3D高斯展开的更新过程相对慢，收敛可能需要数十步细化；③ 仅使用IDM，虽省时却在某些场景下仍可能出现一致性与模糊问题。

---

## 440. Post-Training Fairness Control: A Single-Train Framework for Dynamic Fairness in Recommendation

**arXiv ID:** 2601.20848 | [PDF](https://arxiv.org/pdf/2601.20848v1)

**作者:** Weixin Chen `[一作]` (Hong Kong Baptist University), Yuhan Zhao `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 1389 | [OpenAlex ID](https://openalex.org/A5101650698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种单次训练框架Cofair，允许在推理阶段根据不同公平度要求动态调整推荐结果。

**💡 创新点**

创新点在于引入共享表示层与公平条件适配器相结合，并通过用户级正则化保证公平度随级别递增且保持单调。

**🔧 技术方法**

主要技术包括多任务共享表示、适配器模块、对抗公平损失、用户级正则化以及自适应公平系数权重。

**📊 数据集**

使用了Movielens-1M和Lastfm-360K两大公开数据集，敏感属性为性别。

**📈 对比分析**

与ComFair、FairRec、FairGo、AFRL等现有基线在多层公平度下进行比较，实验显示Cofair在相同训练周期内能够获得更广泛且更优的公平-准确率折衷曲线。

**⚠️ 局限性**

限制在于对敏感属性的假设为二值，且对多重或交叉属性的处理需要进一步扩展；对动态公平度的调优仍需人工设定T和正则化权重。

---

## 441. PatchFormer: A Patch-Based Time Series Foundation Model with Hierarchical Masked Reconstruction and Cross-Domain Transfer Learning for Zero-Shot Multi-Horizon Forecasting

**arXiv ID:** 2601.20845 | [PDF](https://arxiv.org/pdf/2601.20845v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` (Technical University of Denmark), Taner Yilmaz `[通讯]` (Afyon Kocatepe University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 PatchFormer，一种基于补丁的时间序列基础模型，能够实现零样本多步长预测。

**💡 创新点**

创新点包括多尺度层次补丁分词、对比学习加掩码重建的自监督预训练、跨域知识蒸馏以及轻量化适配器微调。

**🔧 技术方法**

技术方法涵盖 Transformer 编码器、动态掩码、对比损失、知识蒸馏以及 Adapter 结构。

**📊 数据集**

使用了 24 个涵盖天气、能源、交通、金融、医疗的基准数据集，预训练数据总量达 87 亿点。

**📈 对比分析**

与 ARIMA、LSTM、Transformer、Informer、Autoformer、PatchTST、TimeGPT 等基线对比，PatchFormer 在 24 个配置上平均降低 27.3% MSE，并且推理速度比普通 Transformer 快 3.8 倍。

**⚠️ 局限性**

局限性在于需大规模预训练数据、对非常短期波动的补丁分辨率有限、缺乏完善的不确定性量化以及在极大模型规模下仍有显著计算开销。

---

## 442. Open-Vocabulary Functional 3D Human-Scene Interaction Generation

**arXiv ID:** 2601.20835 | [PDF](https://arxiv.org/pdf/2601.20835v1)

**作者:** Jie Liu `[一作]` (University of Amsterdam), Yan Zhang `[通讯]` (Meshcapade)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个训练无关的 FunHSI 框架，能够根据多视角 RGB‑D 图像和开放词汇任务提示自动生成与功能元素交互的 3D 人体。

**💡 创新点**

创新点在于引入功能感知的接触图推理、基于 LLM 的功能元素定位与接触关系生成、人体初始化与接触图自我纠正，以及两阶段优化策略，从高层任务意图到物理可行的交互实现提供了全流程解决方案。

**🔧 技术方法**

采用 Gemini‑2.5‑Flash、Gemini、GPT‑4o 等视觉‑语言模型进行功能定位与人像生成；使用 SMPL‑X、VolumetricSMPL 进行人体建模与 SDF 碰撞检测；CameraHMR、WiLoR 估计姿态；VPoser 作为先验；MapAnything 等进行 3D 重建；并结合接触图的多阶段优化。

**📊 数据集**

主要使用 SceneFun3D 数据集（30 个室内场景，三视角 RGB‑D、功能标注）进行基准测试，并在真实城市场景（通过 GeoCalib + MapAnything 重建）上进行进一步验证。

**📈 对比分析**

与改编后的 GenZI* 与 GenHSI* 进行定量比较，指标包括语义一致性、非碰撞得分、功能接触距离等。FunHSI 在功能交互任务上显著优于基线（功能接触距离最低、整体接触距离最小），在一般交互任务上保持相近或略优的语义一致性与物理可行性。

**⚠️ 局限性**

局限性在于仅支持单步、单帧的功能交互，无法处理多步时序交互；对功能元素检测的依赖会影响结果；尺度统一和大场景中尺度估计尚未完善。

---

## 443. Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives

**arXiv ID:** 2601.20833 | [PDF](https://arxiv.org/pdf/2601.20833v1)

**作者:** Tengyue Xu `[一作]` (AgentAlpha Team), Kris Chen `[通讯]` (AgentAlpha Team)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Idea2Story 框架，将科研流程拆分为离线知识构建和在线研究生成两阶段：离线阶段提取可重用的方法单元并构建知识图谱；在线阶段通过检索和组合知识图谱中的研究模式来生成具体的科研方案，并通过 LLM 评审迭代优化。

**💡 创新点**

创新点在于：①采用预计算驱动的策略，把对文献的理解从运行时推理迁移到离线知识结构化；②构建可复用方法单元的结构化知识图谱，解决大模型上下文窗口瓶颈；③多视角检索与基于评审的循环优化相结合，显著提升研究模式的创新性和可行性。

**🔧 技术方法**

核心技术包括：大语言模型（GLM‑4.7、Gemini 3 Pro 评审）、文本嵌入与 UMAP 降维、DBSCAN 聚类生成研究模式、图数据库构建知识图谱、基于图的检索与组合、LLM 驱动的评审‑改进循环。

**📊 数据集**

数据集为过去三年内 NeurIPS 与 ICLR 接受论文及其同行评审（约13,000篇），并对论文内容和评审信息进行脱敏与安全过滤，用于构建方法单元和知识图谱。

**📈 对比分析**

与直接让 LLM 生成完整研究故事的基线进行对比，使用 Gemini 3 Pro 进行客观评审。结果显示 Idea2Story 生成的研究模式在问题表述、方法结构和创新性方面优于基线，评审分数更高。

**⚠️ 局限性**

局限性包括：①依赖特定会议的论文集合，知识覆盖范围有限；②当前仅进行定性评估，缺乏实验验证循环；③对 LLM 评审的鲁棒性与公平性尚未充分保证；④知识图谱需持续更新才能保持最新性。

---

## 444. Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers

**arXiv ID:** 2601.20796 | [PDF](https://arxiv.org/pdf/2601.20796v1)

**作者:** Yiran Huang `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 16061 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地研究并解析多模态 Transformer 的上下文学习机制，构建可控的合成实验平台以分离数据统计、架构与学习动态的因果关系。

**💡 创新点**

首次揭示多模态学习的模态不对称性——主模态预训练能在次模态数据复杂度低时实现上下文学习，并提供对应的机制（归纳电路）与实验验证；同时证明 RoPE 会抑制此类电路。

**🔧 技术方法**

使用两层 Transformer 解码器、RMSNorm、SiLU、RoPE/APE、MLP 投影器、可选的预训练 ViT 编码器，并引入进阶进度度量（PHStrength、IndStrength 等）评估注意力电路。

**📊 数据集**

主要使用从 Gaussian 混合模型生成的合成双模态分类数据，随后在 Omniglot 图像与对应文本的真实多模态任务上进行验证。

**📈 对比分析**

通过对比不同数据复杂度、模态规模、位置编码方式、模型规模以及是否使用编码器的设置，在合成任务上实现 ICL 准确率超过 95%，并通过注意力电路消融、归因回归等方法证明电路的因果性；在 Omniglot 上验证相同趋势。

**⚠️ 局限性**

限制包括：实验仅在小规模、人工合成数据上进行，无法直接推广到大规模真实多模态预训练体系；RoPE 的负面影响在实际 LLM 中可能被其他工程技巧抵消；未深入探讨跨模态对齐的更细粒度机制。

---

## 445. SERA: Soft-Verified Efficient Repository Agents

**arXiv ID:** 2601.20789 | [PDF](https://arxiv.org/pdf/2601.20789v1)

**作者:** Ethan Shen `[一作]` (University of Washington), Tim Dettmers `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了软验证生成（Soft Verified Generation）方法，利用教师模型在无测试环境下生成大量训练轨迹，并通过软验证（行级召回）筛选数据；在此基础上训练了 32B 代码代理 SERA，展示了低成本（约 2000 美元）即可达到或超过现有最佳开源/闭源模型的效果；进一步验证了仓库专门化能力，利用软验证数据实现对 Django、Sympy、Sphinx 等仓库的微调，单仓库仅需 8000 条轨迹即可匹配教师模型表现。

**💡 创新点**

核心创新包括：1）软验证替代传统单元测试验证，消除对测试基础设施的依赖；2）仅使用教师模型生成轨迹，避免复杂的强化学习或bug 注入管道；3）证明软验证数据质量与硬验证相当，显著降低数据生成成本；4）通过专门化比例（α）与样本规模的经验法则，展示专门化样本效率提升；5）完整公开 200k 轨迹、代码与模型，降低研究门槛。

**🔧 技术方法**

技术主要有：教师模型 GLM‑4.5‑Air/GLM‑4.6 进行轨迹生成；SWE‑agent 工具框架收集动作与观察；软验证通过行级召回计算 r；数据截断与过滤策略（patch≤40 行、工具输出≤600 令牌）提升质量；在 Qwen‑3‑32B 上进行全监督微调（axolotl、vLLM）；基于 scaling law 预测性能并指导样本规模；专门化实验使用 α 作为仓库特定数据比例。

**📊 数据集**

数据集：从 121 个公开 Python 仓库（与 SWE‑smith 公开集合一致）生成约 200k 条轨迹；评估使用 SWE‑bench Verified 子集（真实 GitHub issue + PR ），覆盖 Django、Sympy、Sphinx 等仓库；实验也用 128 仓库生成的任务作为对照。

**📈 对比分析**

与现有方法（SWE‑smith、BugPilot、SkyRL‑Agent、DeepSWE 等）在 SWE‑bench Verified 上对比；SERA‑32B 在 32K 上取得 49.5%/54.2% 通过软验证生成的轨迹，优于 SWE‑smith（25.6%）并接近 Devstral‑Small‑2（50%）；在 64K 上同样领先大多数开源模型；专门化实验显示仅 8000 条轨迹即可匹配或超过教师 GLM‑4.5‑Air（51.2%）与 Devstral‑Small‑2（48.6%）的表现。成本方面，SERA‑32B 仅需约 2000 美元；仓库专门化成本约 1300 美元。

**⚠️ 局限性**

局限性包括：1）验证结果仅在 SWE‑bench 上评估，未验证在其他 benchmark 或真实任务中的泛化；2）软验证是否在更大模型/更高质量数据下仍可取代硬验证尚不确定；3）专门化实验主要基于公开仓库，未直接验证对完全私有代码库的效果；4）实验受限于 3 次种子，部分小幅度差异的显著性需谨慎；5）模型与教师均为 Qwen/GLM 系列，未充分验证对其他模型族的迁移性。

---

## 446. Evolutionary Strategies lead to Catastrophic Forgetting in LLMs

**arXiv ID:** 2601.20861 | [PDF](https://arxiv.org/pdf/2601.20861v1)

**作者:** Immanuel Abdi `[一作]` (University of California Berkeley), Gopala Anumanchipalli `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对演化策略（ES）在大规模语言模型微调中的持续学习表现进行了系统实验，并与梯度基方法GRPO进行对比。

**💡 创新点**

创新点在于揭示ES在保持先前能力方面的灾难性遗忘现象，并通过更新范数与稀疏度分析解释其原因。

**🔧 技术方法**

主要技术包括使用ES和GRPO对Qwen2.5-1.5B与Llama-3.2-1B模型在数学与推理任务上微调，并对更新梯度的ℓ2范数和稀疏度进行量化。

**📊 数据集**

使用的数据集包括GSM8K、MATH、OlympiadBench、Countdown以及用于评估先前能力的HellaSwag。

**📈 对比分析**

与GRPO相比，ES在新任务上可达相近性能，但在持续微调过程中导致约10% 的先前任务准确率下降，且更新范数大幅高、稀疏度低，显示出更强的全局参数漂移。

**⚠️ 局限性**

局限性包括ES更新的随机性导致实验方差大，且仅评估了单一先前任务的遗忘，未覆盖多任务全局性能损失。

---

## 447. When Flores Bloomz Wrong: Cross-Direction Contamination in Machine Translation Evaluation

**arXiv ID:** 2601.20858 | [PDF](https://arxiv.org/pdf/2601.20858v1)

**作者:** David Tan `[一作]` (Saarland University), Koel Dutta Chowdhury `[通讯]` (Saarland University)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5020040995)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

调查大型多语言语言模型在机器翻译任务中的数据污染现象，尤其是跨方向污染。

**💡 创新点**

通过多路平行数据集 FLORES‑200 设计源/目标扰动实验，证明污染是跨方向且基于目标语言记忆，并发现命名实体替换可作为检测污染的有效方法。

**🔧 技术方法**

采用 BLEU、COMET 评估指标、back‑translation、paraphrasing、spaCy+Aya 进行实体替换，以及 Axolotl 进行细调等技术。

**📊 数据集**

使用 FLORES‑200 测试集（15种语言）以及低资源外部测试集 PMIndia、Mann‑ki‑Baat，并利用多语言 back‑translation 及实体替换句子。

**📈 对比分析**

将已知污染的 Bloomz 与无污染的 Llama 进行同等设置对比；Bloomz 在多语言对中 BLEU 高达 40–90，Llama 仅 30–40；在细调实验中未见污染模型 BLEU 上升多达 16 点，但 COMET 下降。

**⚠️ 局限性**

仅测试 7–8B 参数模型，未解析内部记忆机制；实体替换可能因形态变化导致 BLEU 下降；实验使用模拟污染的细调可能不完全反映预训练阶段污染。

---

## 448. $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval

**arXiv ID:** 2601.20844 | [PDF](https://arxiv.org/pdf/2601.20844v1)

**作者:** Zihao Wang `[一作]` (Hong Kong University of Science and Technology), Simon See `[通讯]` (Nvidia)

**通讯引用:** 2847 | [OpenAlex ID](https://openalex.org/A5077539496)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了基于嵌入的 top‑k 检索的最小嵌入维度（MED）并给出了理论与实验结果。

**💡 创新点**

创新点在于证明 MED 与元素数量无关，只随 k 线性增长（Θ(k)），并在中心化（centroid）设置下给出对数级上界 O(k² log m)，从而驳斥先前的经验性结论。

**🔧 技术方法**

使用了 VC 维度、循环多面体构造、概率方法（随机向量取样）以及 hinge‑loss 优化等技术来推导上下界并验证实验。

**📊 数据集**

实验主要基于人工合成的随机向量数据集（随机初始化的 m 维向量），未使用公开真实数据集。

**📈 对比分析**

与先前的经验性拟合（如 m_WBNL(d)）比较，实验显示在相同维度下能支持更大 m，且维度对 m 的增长呈对数或指数关系，性能明显优于对方曲线。

**⚠️ 局限性**

实验受限于数值精度、可行的函数数量以及 O(k² log m) 上界的非最优性；仅针对欧氏/内积/余弦空间，未探讨其他嵌入空间。

---

## 449. Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning

**arXiv ID:** 2601.20829 | [PDF](https://arxiv.org/pdf/2601.20829v1)

**作者:** Minwu Kim `[一作]` (New York University), Keith Ross `[通讯]` (New York University)

**通讯引用:** 15521 | [OpenAlex ID](https://openalex.org/A5101801953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了失败前缀条件化（failure‑prefix conditioning）方法，通过在训练时以稀有错误推理片段为起点，重新分配探索空间以恢复饱和问题中的学习信号；

**💡 创新点**

创新点在于将训练重心从原始问题转向错误前缀，显著提升了在已达高准确率（≈97%）任务上的强化学习效果，并且支持迭代刷新前缀以持续获取学习信号；

**🔧 技术方法**

采用强化学习与可验证奖励（RLVR）框架中的GRPO算法，配合失败前缀抽样、前缀长度选择与迭代更新策略；

**📊 数据集**

使用了MATH、DeepScaleR等大型数学推理数据集，并在MATH500、AMC12、AIME24/25、HMMT25等五大评测基准上验证；

**📈 对比分析**

与基准模型（未训练、仅在饱和问题上训练、在中等难度问题上训练）相比，失败前缀模型在所有评测基准上平均提升约2.8个百分点（与中等难度模型持平），同时保持推理 token 长度不变，性能稳健；

**⚠️ 局限性**

局限性包括在处理正确前缀时略微降低对准对齐度、前缀选择可能偏离当前策略导致的离策略性问题，以及迭代刷新所需的额外采样成本。

---

## 450. Repeater-Assisted Massive MIMO Full-Duplex Communications

**arXiv ID:** 2601.20822 | [PDF](https://arxiv.org/pdf/2601.20822v1)

**作者:** Mohammadali Mohammadi `[一作]`, Michail Matthaiou `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并优化多天线基站与分布式全双工重复器的放大权重，以最大化加权最小频谱效率。

**💡 创新点**

创新点在于将全双工基站与可调放大重复器结合，并采用凸-凹程序与可行点追踪的连续凸近似(SCA)方法实现非凸优化，同时兼顾公平性目标。

**🔧 技术方法**

使用技术包括零逼近(ZF)预编码/组合、凸-凹程序、可行点追踪与SCA、CVX求解、统计信道模型和SI抑制技术。

**📊 数据集**

采用仿真随机部署的数据集：64个单天线UE、32个重复器，基站中心，400 m² 区域内的路径损耗模型，未使用公开数据集。

**📈 对比分析**

通过与随机权重RA‑FD、随机权重RA‑HD以及无重复器的集中式FD‑mMIMO进行CDF比较，RA‑FD优化在DL/UL频谱效率分别提升约98%和252%，相对集中式FD提升约4倍和2.5倍。

**⚠️ 局限性**

局限性包括：假设同步且重复器位置固定；SI抑制仅为-60 dB；算法对初始点敏感、计算复杂；仅考虑单天线UE，未涵盖多天线UE或实际硬件非理想情况。

---

## 451. C3Box: A CLIP-based Class-Incremental Learning Toolbox

**arXiv ID:** 2601.20852 | [PDF](https://arxiv.org/pdf/2601.20852v1)

**作者:** Hao Sun `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**通讯引用:** 1730 | [OpenAlex ID](https://openalex.org/A5100655948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了C3Box，一个面向CLIP的增量学习工具箱，统一整合了传统、ViT以及最新CLIP方法并提供统一的JSON配置与执行流程；

**💡 创新点**

创新点在于提供了模块化、可复现、跨平台的统一框架，解决了现有CLIP增量学习方法碎片化、实验不一致的问题；

**🔧 技术方法**

采用CLIP预训练模型（LAION‑400M/​OpenAI）、ViT骨干、prompt/adapter、回放、herding等技术，并继承PyCIL的流水线；

**📊 数据集**

使用十个公开基准数据集（CIFAR‑100、CUB‑200、ObjectNet、ImageNet‑R、FGVCAircraft、StanfordCars、Food101、SUN397、UCF101、TV100）进行评测；

**📈 对比分析**

对比了17种代表性增量学习方法，实验表明CLIP‑based方法在平均精度和最后精度上普遍优于传统方法，显示出更强的抗灾难性遗忘能力；

**⚠️ 局限性**

局限性包括仅支持固定的10个数据集、依赖CLIP预训练且尚未覆盖所有最新增量学习技术，可能在更大规模或多模态场景下表现有限。

---

## 452. End-to-end example-based sim-to-real RL policy transfer based on neural stylisation with application to robotic cutting

**arXiv ID:** 2601.20846 | [PDF](https://arxiv.org/pdf/2601.20846v1)

**作者:** Jamie Hathaway `[一作]` (University of Birmingham), Rustam Stolkin `[通讯]` (University of Birmingham)

**通讯引用:** 9609 | [OpenAlex ID](https://openalex.org/A5005183926)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于神经风格迁移的模拟到真实世界的机器人切割控制迁移方法，并在KUKA LBR iiwa协作机器人上验证；

**💡 创新点**

创新点在于利用无对齐样本的VAE编码空间进行内容-风格配对，采用风格迁移优化生成真实域观测窗口，无需训练判别器或生成器，实现样本高效、无标签迁移；

**🔧 技术方法**

核心技术包括变分自编码器（VAE）用于学习跨域潜在表示，行为克隆（BC）训练目标域策略，神经风格迁移优化（content+style loss）生成目标域观测；

**📊 数据集**

使用模拟生成的50条演示轨迹作为内容数据，和148条真实无监督轨迹作为风格数据，结合已有的32,000条模拟演示训练专家策略；

**📈 对比分析**

与未迁移专家、BC、条件VAE（CVAE）、CycleGAN等方法比较，在多材料（聚氨酯泡沫、纸板、波纹塑料、云母、铝）及不同曲率、路径偏移等场景下，风格迁移在任务完成时间、路径偏差、工具负载、动作相似度等指标上均优于或与最佳方法持平；

**⚠️ 局限性**

局限性包括：未显式处理目标域动作分布差异或机器人运动学/动力学约束；对真实样本覆盖度依赖较高，配对不匹配或小样本时生成质量下降；

---

## 453. Deep Researcher with Sequential Plan Reflection and Candidates Crossover (Deep Researcher Reflect Evolve)

**arXiv ID:** 2601.20843 | [PDF](https://arxiv.org/pdf/2601.20843v1)

**作者:** Saurav Prateek `[一作]` `[通讯]`, Saurav Prateek

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于顺序迭代的深度研究者架构，用来生成博士级研究报告。

**💡 创新点**

核心创新点包括：1）通过反思（Reflection）实现研究计划的顺序细化；2）候选人交叉（Candidates Crossover）算法，利用多参数LLM候选人并行搜索并合成答案；3）统一一次性报告生成，避免多轮降噪迭代。

**🔧 技术方法**

使用 Gemini‑2.5‑Pro LLM 作为主推模型，结合 Web 搜索工具（Tavily）、LLM‑as‑judge、交叉合成等技术，形成全流程自动化。

**📊 数据集**

在全球公开基准 DeepResearch Bench（100 个博士级研究任务，涵盖 22 个学科、英中两语）上进行评估。

**📈 对比分析**

与主流深度研究代理（Claude Researcher、Nvidia AIQ、Perplexity Research、Grok Deeper Search 等）通过 RACE 与 FACT 评估框架对比，整体分数 46.21，略低于 SOTA 52.44，显著优于其它参比模型，说明顺序缩放优于并行自一致性。

**⚠️ 局限性**

局限性：①未实现自评与迭代反馈步骤，导致搜索效率与答案质量有提升空间；②一次性报告生成虽然降低延迟，但在极大规模任务中可能缺乏细粒度微调；③对多模态信息支持不足，主要集中于文本和网页搜索。

---

## 454. Reward Models Inherit Value Biases from Pretraining

**arXiv ID:** 2601.20838 | [PDF](https://arxiv.org/pdf/2601.20838v1)

**作者:** Brian Christian `[一作]` (University of Oxford), Tsvetomira Dumbalska `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究奖励模型（RM）在预训练阶段继承的价值偏差，证明基模型（Llama与Gemma）导致RM在代理（agency）与共融（communion）价值维度上呈现系统性差异，并跟踪这些差异在训练过程中的演化。

**💡 创新点**

创新点在于：①首次系统性表明RM从预训练LLM继承价值偏差；②将心理学“Big Two”与道德维度映射到RM中；③引入混合加权对数比（MWLR）作为可用的隐式奖励评分；④通过可复制的实验证实不同数据量与来源对偏差的缓解程度。

**🔧 技术方法**

技术方法包括：①“完整词元搜索”（exhaustive token search）评估RM奖励；②心理语言学工具将词汇映射到价值维度；③计算预训练模型的对数概率差并构建隐式奖励模型；④混合加权对数比（MWLR）提升隐式奖励的可解释性；⑤在不同基模型上使用LoRA、AdamW、Bradley‑Terry损失进行RM微调；⑥使用统计模型（混合效应线性模型、ANOVA、Kendall τ）评估偏差。

**📊 数据集**

数据集主要包括：RewardBench中10个开源RM；心理语言学语料库“Big Two”和“Moral Foundations Dictionary 2（MFD2）”；Skywork v0.2（80k偏好对）和Unified Feedback（800k偏好对）等偏好数据；此外还使用了不同规模的Gemma与Llama预训练模型。

**📈 对比分析**

比较方法：对比基模型（Gemma vs Llama）在相同提示、相同偏好数据下的RM奖励排名；利用MWLR对不同模型的对数概率差进行排序；在训练过程中每1000步记录RM奖励，观察偏差随步数与数据量变化。实验表明，Gemma RM在共融词汇上得分更高，Llama RM在代理词汇上得分更高；随着训练数据量增大，偏差可被部分抑制，但在大规模模型或采用GRM正则化时差距仍显著。

**⚠️ 局限性**

局限性包括：①仅分析单词级别奖励，未深入多词或长文本；②仅针对Llama和Gemma两大模型族，未覆盖全部开源LLM；③只考察了两维价值（代理、共融），未扩展到更多伦理维度；④实验主要使用中小规模模型，缺乏对极大模型的完整评估；⑤隐式奖励的MWLR方法需进一步验证其对实际生成行为的预测能力。

---

## 455. Low-Complexity Pilot-Aided Doppler Ambiguity Estimation for OTFS Parametric Channel Estimation

**arXiv ID:** 2601.20827 | [PDF](https://arxiv.org/pdf/2601.20827v1)

**作者:** Bo-Yuan Chen `[一作]` (National Taiwan University), Hsuan-Jung Su `[通讯]` (National Taiwan University)

**通讯引用:** 1526 | [OpenAlex ID](https://openalex.org/A5018959296)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了OTFS在极高相对速度下出现的Doppler歧义问题，并提出低复杂度的基于pilot的歧义检测与补偿框架。

**💡 创新点**

创新点是利用延迟维度的相位旋转特征，通过多pilot相位差与投票法快速估计整数歧义，并在此基础上进行高效的MLE通道估计。

**🔧 技术方法**

采用OTFS信号模型、相位差投票估计、最大似然估计（MLE）与符号干扰消除（SIC），以及EP-GZ和DSP两种pilot布局。

**📊 数据集**

在模拟环境中使用二维延迟-多普勒稀疏通道（P=4、M=64、N=32）以及自定义的LEO卫星高相对速度（k_max=79）进行仿真。

**📈 对比分析**

与理想CSI、标准MLE、以及全搜索的Extended MLE进行比较；结果表明提出的方法在BER、NMSE上接近Extended MLE并远优于标准MLE，且计算复杂度与标准MLE相当。

**⚠️ 局限性**

局限在于对弱路径的检测依赖于较高的pilot-to-data功率比，在DSP布局下仍存在误检错误；同时实验仅为仿真，缺乏实测验证。

---

## 456. How Disciplinary Partnerships Shape Research Landscape in U.S. Library and Information Science Schools

**arXiv ID:** 2601.20806 | [PDF](https://arxiv.org/pdf/2601.20806v1)

**作者:** Jiangen He `[一作]` (University of Tennessee), Wen Lou `[通讯]` (East China Normal University)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5033347739)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国44所图书信息科学学院的组织结构与研究产出进行系统性、跨学科的实证映射，提出三大基础研究维度，并分析其随时间的演化。

**💡 创新点**

首次构建基于主题建模的三维研究景观框架，揭示组织结构与研究侧重点共现的模式，并指出人类中心技术成为主要增长向量。

**🔧 技术方法**

采用BERTopic+SPECTER2进行主题建模，UMAP降维+层次聚类，Aitchison距离与PERMANOVA进行多元方差检验，线性回归与三角图跟踪研究演化。

**📊 数据集**

2013-2024年14,705篇论文、1,264名教员及其所属44所学校的结构与出版记录（Dimensions API、Wayback Machine抓取的教员目录）。

**📈 对比分析**

通过PERMANOVA检验结构间主题比例差异，显著p<0.01；计算线性趋势、增速，发现人类中心技术增长率最高（6.7%），信息检索下降（-2.4%）。

**⚠️ 局限性**

仅关注出版物忽略教学与服务；样本仅美国，可能不具全球代表性；无法确定因果方向，需进一步案例研究。

---

## 457. Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning in Few-Shot Relation Extraction

**arXiv ID:** 2601.20803 | [PDF](https://arxiv.org/pdf/2601.20803v1)

**作者:** Aunabil Chakma `[一作]` (University of Arizona), Eduardo Blanco `[通讯]` (University of Arizona)

**通讯引用:** 1723 | [OpenAlex ID](https://openalex.org/A5052295709)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合式示例选择方法，将 LLM 生成的例子与基于句法语义规则检索的例子相结合，从而将一-shot 关系抽取转化为 5/10-shot 的 in-context learning 任务。

**💡 创新点**

创新点在于：①引入自监督的句法语义规则表示，用于检索结构上与给定单例例子相似但语义多样的真实句子；②通过聚类与多样性控制策略挑选代表性例子；③将检索例子与 LLM 生成/改写例子混合，平衡相似度与多样性，从而显著提升性能。

**🔧 技术方法**

技术手段包括：in-context learning、LLM 生成与改写、句法语义规则提取与嵌入、FAISS 索引检索、k-means/++ 聚类、多样性采样、NER 过滤、Prompt 设计。

**📊 数据集**

使用的数据集为 TACRED 的 5-way 1-shot 版本（FewRel 的重新包装）以及对应的 FewRel 子集，并在检索阶段使用 UMBC WebBase 的 230 万句无标注文本。

**📈 对比分析**

与 1-shot 基线、仅 LLM 生成、仅检索等多种对照方式比较，混合方法在 Qwen3-4B、Gemma 4B 等小型 LLM 上取得了新的 SOTA（TACRED 上 F1 最高 37.8，FewRel 上亦显著提升），并在多种 LLM 及数据集上保持稳健。

**⚠️ 局限性**

局限性包括：仅针对与 TACRED 对齐的 FewRel 子集进行实验；不同 LLM 对同一示例集的响应差异需进一步个性化；语义规则提取的误差可能导致引入不相关示例；对大模型效果提升有限。

---

## 458. Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces

**arXiv ID:** 2601.20800 | [PDF](https://arxiv.org/pdf/2601.20800v1)

**作者:** Kaito Baba `[一作]` (Preferred Networks Inc), Shuhei Watanabe `[通讯]` (SB Intuitions Corp)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Conditional PED-ANOVA（cPED-ANOVA），一种能在具有条件层级和动态域的搜索空间中估计超参数重要性（HPI）的框架。

**💡 创新点**

①识别了原 PED-ANOVA 在条件搜索空间中“漏泄”问题，即条件变量的方差被错误归因给被条件化的超参数。②提出只使用“within-regime”方差的条件本地 HPI 定义，并给出闭式 Pearson 散度估计器，保持原方法的高效性。

**🔧 技术方法**

基于方差分解、Pearson 散度、核密度估计（KDE）以及对每个条件区间（regime）加权的闭式公式，构成了高效的 HPI 计算器。

**📊 数据集**

主要使用合成实验数据：1）条件激活（Gating 变量 c 决定 x 或 y 的激活），2）条件域变化（c 决定 x、y 的取值区间），并对不同 γ、γ' 量化。实际评测也用 Optuna 框架中的实现进行多次随机采样。

**📈 对比分析**

与四种基线（PED-ANOVA、f-ANOVA、MDI、SHAP）在两类 naive 处理（过滤、填充、域扩展）下进行对比。实验表明，cPED-ANOVA 能在所有设定下得到符合直觉且稳定的 HPI 曲线，而 naive 方案往往给出相同或错误的重要性，甚至对不活跃超参数也给出非零重要性；计算速度与原 PED-ANOVA 相当。

**⚠️ 局限性**

局限性：只在单一“最优子空间”内估计本地 HPI，未考虑多目标或多级搜索；对真实复杂模型的验证有限；对非常大的层级树可能需要进一步优化实现；并且仍需手动确定每个超参数的 regime 数量与域划分。

---

## 459. A Methodology for Designing Knowledge-Driven Missions for Robots

**arXiv ID:** 2601.20797 | [PDF](https://arxiv.org/pdf/2601.20797v1)

**作者:** Guillermo GP-Lenza `[一作]` (Universidad Politécnica de Madrid), Pascual Campoy `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 5494 | [OpenAlex ID](https://openalex.org/A5001678286)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并实现了一套将知识图谱集成到 ROS 2 系统中的完整方法论，并在 Aerostack2 框架下完成了多无人机搜索救援仿真实验。

**💡 创新点**

创新点在于将知识图谱与 ROS 2 模块（知识库、提取器、检索器）结合，构建了从任务定义到高层规划的五步流程，并支持多代理知识图谱的自动合并与实时更新。

**🔧 技术方法**

主要技术包括知识图谱建模与查询、ROS 2 节点通信、Aerostack2 航迹规划与控制、Gazebo 仿真环境、基于规则的推理以及 Python 语言实现高层任务脚本。

**📊 数据集**

实验使用 Gazebo 生成的仿真数据（位置、状态、电量、摄像头图像等），并未使用公开真实数据集。

**📈 对比分析**

通过对比传统静态知识表示，作者展示了知识图谱在决策支持与任务执行效率方面的优势，但实验仅以视频和代码示例进行定性评估，未给出量化性能指标。

**⚠️ 局限性**

方法的主要限制是需要人工设计感知信息到知识图谱节点/边的映射，过程繁琐且易出错；缺乏自动化工具与高级（如概率或机器学习）推理支持。

---

