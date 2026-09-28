# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-28 | 今日论文总数: 642

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Multi-Objective Human-in-the-Loop Bayesian Optimization of a Lower-Limb Exoskeleton

**arXiv ID:** 2609.30695 | [PDF](https://arxiv.org/pdf/2609.30695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 2. Bridging LLM Agents and Data Spaces: An Architectural Mediation Approach using the Model Context Protocol

**arXiv ID:** 2609.30341 | [PDF](https://arxiv.org/pdf/2609.30341v1)

**作者:** Jaime Alonso Ruiz `[一作]` (Universidad Politécnica de Madrid), Andres Munoz-Arcentales `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5041365624)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于 Model Context Protocol（MCP）的中介层，能够在不修改 Eunomia 数据空间代理的前提下，将数据空间的目录查询和数据服务调用功能以结构化工具的形式暴露给 LLM 代理，实现 AI 代理与数据空间的互操作。

**💡 创新点**

创新点在于：①使用 MCP 把协议驱动的数据空间能力转化为 LLM 可发现和调用的工具，构建了一个非侵入式、可插拔的中介层；②保持了数据空间治理与 AI 代理演进的分离，避免了协议耦合和治理破坏；③提供了完整的架构图和实现细节，为未来多代理协调与动态策略提供参考。

**🔧 技术方法**

主要技术包括 Model Context Protocol、JSON‑RPC（stdio 传输）、HTTP REST API、Eunomia Agent、DCAT‑AP 3.0 元数据模型，以及标准化工具发现与调用流程。

**📊 数据集**

使用了一个名为“MCP Evaluation Dataset”的单一测试数据集，并在 Eunomia 目录中注册了一个 mock 数据服务，模拟真实数据访问。

**📈 对比分析**

评估采用确定性四步序列：工具发现 → 列表查询 → 元数据检索 → 服务查询。所有步骤在单机环境下重复执行，验证协议一致性、功能正确性及数据完整性。由于是功能验证，未进行性能基准；结果表明系统能够无错误地完成端到端调用，功能可用性良好。

**⚠️ 局限性**

局限性包括：①缺乏身份验证与授权，易受 prompt 注入与信息泄露风险；②仅实现了目录查询和服务调用，未覆盖合同协商、数据平面等完整数据空间功能；③示例规模极小，未检验在大规模数据空间中的可扩展性与性能；④仅在本地 stdio 传输下测试，未验证网络化部署的安全与可靠性。

---

## 3. GAUDI: Geometry-Aware Diffusion for Calibrated Air-Quality Time-Series Imputation

**arXiv ID:** 2609.30340 | [PDF](https://arxiv.org/pdf/2609.30340v1)

**作者:** Xinjin Li `[一作]` (Columbia University), Tianxin Zhou `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了针对空气质量监测中块状缺失的条件扩散插补模型，并通过对侧信息处理方式的 Ablation 实验探讨其对插补性能的影响。

**💡 创新点**

提出了“特征侧条件化”策略，即仅保留变量身份和观测掩码而抑制窗口位置嵌入，从而在保持时序和特征交互的前提下提升插补精度。

**🔧 技术方法**

使用扩散模型与结构化状态空间层相结合的 denoiser，配合时间-特征注意力机制实现条件化插补。

**📊 数据集**

在 ItalyAir 数据集（13 维空气质量变量，窗口长度 32，50% 块缺失）上进行实验。

**📈 对比分析**

与完整侧信息模型、本地 CSDI 以及线性插值比较，特征侧条件化在三次训练种子上平均 RMSE 降低约 5%，并显著优于全侧信息模型。

**⚠️ 局限性**

仅提升点估计准确度，置信区间覆盖率仍偏低，且实验仅覆盖单一数据集和缺失模式，缺乏跨数据集和缺失形态的验证。

---

## 4. Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents

**arXiv ID:** 2609.30725 | [PDF](https://arxiv.org/pdf/2609.30725v1)

**作者:** Yiran Hu `[一作]` (Purdue University), Lin Tan `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地分析了 Claude Code 与 Mini‑SWE‑Agent 在执行中的三类成本低效行为（子检索覆盖、类似脚本生成、测试重复执行），并评估了三种缓解策略的效果

**💡 创新点**

首次将低效行为分为三类，并通过对比结构检索、代理合成技能与开发者设计技能的成本削减效果，证明了人类设计的高层技能更为有效

**🔧 技术方法**

采用轨迹解析、基于规则与 LLM 的动作标注、代码图（CodeGraph）结构检索、技能生成与注入、以及多配置、多基准的实验评估

**📊 数据集**

使用 SWE‑bench Verified（300/200 任务）与 Pro（100 任务）数据集

**📈 对比分析**

在 8 种代理配置（CC+S46、MSA+S46、MM3、Q35+）与两套基准上进行对比实验，发现开发者设计技能可将任务成本降低 7.9%–41.7%，最大幅度达 41.7%；结构检索在某些配置下反而导致成本上升至 28.14%

**⚠️ 局限性**

实验仅覆盖有限的代理与任务集；成本测量不包含执行时间和硬件成本；行为检测与标注可能存在误差；技能设计依赖人工调优，难以自动泛化

---

## 5. RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent

**arXiv ID:** 2609.30717 | [PDF](https://arxiv.org/pdf/2609.30717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 6. Query-Conditioned Prototype Adaptation for Cross-Domain Few-Shot Learning: Single-Query Inference, Controlled Comparisons, and Failure Modes

**arXiv ID:** 2609.30769 | [PDF](https://arxiv.org/pdf/2609.30769v1)

**作者:** Rushab Rasik Karania `[一作]` (University of Nottingham Malaysia), Tomas Maul `[通讯]` (University of Nottingham Malaysia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在冻结的ViT特征空间中，利用单个未标记查询与标记支持集共同变换来构造查询特定原型的WIPT方法，实现在跨域少样本学习中的测试时原型适应。

**💡 创新点**

创新点在于在不进行目标域梯度更新或伪标签的前提下，仅通过单查询的全局嵌入交互来动态调整原型，从而探索查询参与对原型构造的真实贡献，并提供严格的对比实验。

**🔧 技术方法**

使用了ViT‑Small/16作为固定编码器，基于Transformer的两层自注意力头来实现查询与支持的联合变换，并采用欧氏距离进行分类。

**📊 数据集**

实验使用miniImageNet作为源域训练集，在三大目标域（CUB、EuroSAT、ISIC）上进行跨域评估，分别包含细粒度、卫星和皮肤病图像。

**📈 对比分析**

与固定原型ProtoNet以及同容量的仅支持集Transformer做对比；在1-shot下，WIPT在CUB和EuroSAT上比ProtoNet高于0.2–2.1个百分点；在5-shot下，ProtoNet整体表现最好，WIPT仅在ISIC上相对支持集Transformer有约1.0个百分点提升。额外的多查询上下文并未带来显著准确率提升。

**⚠️ 局限性**

局限性包括：仅使用单一源域和固定ViT编码器；未探索更大或多模态查询集；对不同分数函数的敏感性未充分揭示；以及对表示层本身不做适配，导致方法在某些域（如ISIC）表现不佳。

---

## 7. Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More

**arXiv ID:** 2609.30768 | [PDF](https://arxiv.org/pdf/2609.30768v1)

**作者:** Deng Pan `[一作]` (University of Notre Dame), Nitesh Chawla `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对三种大规模推理语言模型（QwQ‑32B、DeepSeek‑R1‑Distill‑Qwen‑32B、Qwen3‑32B）在三项高风险决策任务（Adult、COMPAS、Credit）上进行思考（思考链）与无思考基线的对照实验，评估其对因特征保护属性导致的对照反事实公平性的影响。

**💡 创新点**

提出了两种动态衡量工具——Counterfactual Depth Probability Gap（CDPG）和Bias Transition Matrix（BTM），揭示思考阶段既能消除已有的对照反事实偏差，又能在高置信度下产生新的偏差，形成“非对称双重效应”，并指出该效应主要来源于对照对状态的联合转换而非单侧属性变化。

**🔧 技术方法**

使用前缀截断法对思考链进行深度分段，计算每一层的概率差距（CDPG）；构建非思考与思考之间的预测转移矩阵（BTM）来追踪对照对状态的变化；利用配对对比单元（a,b,c,d）分解思考带来的偏差增减；采用McNemar检验和统计显著性评估。

**📊 数据集**

Adult（性别）、COMPAS（种族）和Credit（性别）三大公开表格数据集，各约5,000条记录，通过固定模板生成自然语言描述并进行对照属性替换。

**📈 对比分析**

与传统的组级公平性度量（群体差异）相比，单一组度量对思考效应的判定不一致，而对照反事实公平性在所有9个模型×数据集组合中均表现为偏差增大；通过CDPG和BTM的细粒度分析，揭示思考会在约5倍的比例下产生新偏差，并在高置信度下强化其幅度。

**⚠️ 局限性**

研究仅覆盖二元分类任务与二元保护属性，未评估多类别、多语言或生成任务；思考链分段为固定10段，可能影响深度细节；基线无思考实现为插入空白思考块，可能无法完全消除模型的默认推理；实验采用4‑bit量化，未检验更高精度模型的表现；未提供具体缓解策略，仅提出诊断工具。

---

## 8. T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation

**arXiv ID:** 2609.30576 | [PDF](https://arxiv.org/pdf/2609.30576v1)

**作者:** Yang Liu `[一作]` (Shopify), Linjun Yang `[通讯]` (Shopify)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种时间感知的旋转位置编码T‑RoPE，用于生成式推荐系统，使注意力能够直接捕捉事件发生的绝对时间、跨尺度周期性与日历周期；

**💡 创新点**

创新点在于将RoPE中的整数索引替换为实际时间戳，加入可学习的时间系数、多尺度频率库、查询时移对齐以及非平稳键旋转，从而打破时平移不变性，使模型能感知季节性与长期周期；

**🔧 技术方法**

技术主要包括基于Transformer的自回归推荐框架、RoPE改造、学习时间系数、多尺度频率银行、查询时移对齐、非平稳键旋转及其高效前向/反向实现；

**📊 数据集**

使用了五个公开基准（ML‑20M、Amazon Books、PixelRec 200K/1M/8M）以及工业级电商数据（600M+交互）进行评估；

**📈 对比分析**

与SASRec、TiSASRec、HSTU、HSTU+Time RAB、HSTU+TO‑RoPE等基线比较，T‑RoPE在所有指标上均实现最优，公开基准上HR@10提升8–12%，PixelRec上提升78–130%，工业数据上提升13–82%；

**⚠️ 局限性**

局限性包括仍需手动设定频率范围、对极端稀疏或非周期性数据的适用性未知，以及非平稳键增加模型复杂度与训练不稳定性的可能性。

---

## 9. Evaluating Code Recommender Systems: A Review

**arXiv ID:** 2609.30351 | [PDF](https://arxiv.org/pdf/2609.30351v1)

**作者:** Daniel Borst `[一作]` (WU Vienna), Stefan Sobernig `[通讯]` (WU Vienna)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过系统文献综述，对2017-2024年间关于代码推荐系统（CRS）评估的92项原始研究进行梳理与合成。

**💡 创新点**

创新之处在于首次系统地聚焦CRS评估方法本身，而非其推荐算法，全面归纳评估类型、度量指标、软件工程阶段及威胁，并公开研究数据。

**🔧 技术方法**

采用标准的系统文献综述流程（PRISMA/SE指南），构建搜索字符串、预设纳入排除标准、双人评审与质量评估，并使用内容分析法提取研究特征。

**📊 数据集**

使用了包含405篇全文文献、92篇高质量评估研究的公开数据集（已发布于Zenodo），以及4个学术数据库的检索结果（共4,101条记录）。

**📈 对比分析**

通过对评估类型、指标分布、软件工程领域与威胁类别的量化比较，发现系统中心化的离线评估占主导；但并未直接比较CRS性能，而是呈现评估方法的偏好与不足。

**⚠️ 局限性**

局限性包括用户中心评估样本极少、评估类型组合不足、对LLM基CRS的可复制性缺乏统一标准，且研究筛选与编码仍可能受主观偏差影响。

---

## 10. Energy-efficient operation of neural operators for virtual sensing

**arXiv ID:** 2609.30580 | [PDF](https://arxiv.org/pdf/2609.30580v1)

**作者:** Jason Yoo `[一作]` (University of Illinois Urbana-Champaign), Syed Bahauddin Alam `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在固定几何、固定查询网格的虚拟感测任务中，研究者通过保留不随输入变化的空间计算（即 trunk 重用）或使用编译器冻结技术，显著降低推理能耗，并与传统 eager 与图形重放执行方式进行对比。

**💡 创新点**

创新点在于：①将可重用的空间表示抽象为单独的计算图块；②将冻结与图形重放结合，形成可扩展的能耗削减方案；③通过对 DeepONet、FNO 等多种神经算子结构的实验，阐明算子结构与可重用性、能耗之间的系统性关系。

**🔧 技术方法**

使用了 PyTorch/CUDA 图形重放、标准编译器冻结、DeepONet、Fourier Neural Operator (FNO)、MIMONet 以及自制的重用框架；实验平台包括 Jetson AGX Xavier、RTX 5060 Ti 与 NVIDIA A2000 GPU。

**📊 数据集**

主要数据集为热交换器的 310 条 CFD 参考案例（共 3,977 个网格节点、3 维速度和压强），以及 2,642 条燃料电池观测记录用于离线更新策略评估。

**📈 对比分析**

比较方法：在相同模型、输入、输出以及服务接口下，测量每秒请求数 1–60Hz 的能耗、平均功率、响应时间以及请求成功率；对比 eager、graph、reuse、reuse+graph、冻结+graph 等路径。结果显示：
- 在 Jetson 上，reuse+graph 可使单请求能耗降低至 43.0 mJ（比 eager 下降 60%），整体服务能耗降低 22–23%；
- 在 15 W 固定时钟下，60 Hz 负载时，平均功率从 8.5 W 降至 6.3 W，p95 响应时间从 11.3 ms 降至 4.0 ms；
- DeepONet 的 trunk 重用在不同设备上实现了 41–73% 的能耗下降，而 FNO 在能耗上仅提升 3% 左右。

**⚠️ 局限性**

局限性：
- 仅适用于固定几何与固定查询网格的任务，无法直接推广到需要动态网格或查询变更的情形；
- 需要一次性构建冻结或图形重放的“artifact”，其构建开销在短期部署中可能抵消能耗收益；
- 评估以离线 310 条 CFD 参考为准，缺乏对更大规模或更复杂流场的验证；
- 对模型的确定性与数值精度做了严格约束，未探讨模型不确定性或随机性对能耗与误差的影响。

---

## 11. Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling

**arXiv ID:** 2609.30288 | [PDF](https://arxiv.org/pdf/2609.30288v1)

**作者:** Narges Mokhtari `[一作]` (Iran University of Science & Technology), Ebrahim Rezaii `[通讯]` (Iran University of Science & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在掩码语言模型中，用低秩瓶颈自编码器堆叠的混合模块取代 Transformer 的自注意力层，并在掩码位置引入迭代拉伸-校正精炼过程完成预训练。

**💡 创新点**

核心创新在于：①显式低秩瓶颈自编码器替代自注意力，压缩比例可调；②构建局部、全局和跨头三层自编码器级联；③提出在连续嵌入空间中进行掩码填充的迭代拉伸-校正机制。

**🔧 技术方法**

使用的技术包括：WindowMixAE、GlobalMixAE、ChannelMixAE（低秩自编码器），迭代拉伸-校正精炼，内容感知邻近平均（ChunkedContentBias），频率感知训练调度，混合损失与自编码器重建损失，AdamW + cosine 学习率调度与混合精度训练。

**📊 数据集**

主要数据集为 C4 大规模文本语料进行预训练，GLUE benchmark 用于下游微调。

**📈 对比分析**

通过与参数匹配的 BERT、TinyBERT、BERT* 等自注意力模型在 C4 上的掩码预测准确率与 perplexity 对比，以及在 GLUE 上单句任务（CoLA、SST‑2）与双句任务（MRPC、MNLI 等）的微调结果进行比较；模型在单句任务上与注意力模型持平或更优，在双句任务上略低，且在最低频率词桶上表现相当；算力方面，相较于匹配的注意力模型，FLOPs 减少约 1.9 倍。

**⚠️ 局限性**

主要局限在于缺乏对不同句段之间关系的对齐机制，导致在句子对比较任务上的表现落后；对迭代精炼超参数（T、α、γ）的敏感性高；对更长序列的可扩展性与动态权重生成仍需进一步研究。

---

## 12. DGT-Map: Directional Global Traversability Mapping Utilizing Multi-Task Learning for Heterogeneous Vehicles

**arXiv ID:** 2609.30461 | [PDF](https://arxiv.org/pdf/2609.30461v1)

**作者:** Jaskrit Singh `[一作]` (Worcester Polytechnic Institute), Constantinos Chamzas `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种自监督学习框架 DGT-MAP，能够从 RGB‑D 观测中学习全球、方向感知且对车辆特定的越野可行性成本图，并将其用于 Hybrid A* 规划与闭环控制。

**💡 创新点**

创新点：① 统一的多任务网络实现跨车辆共享地形特征，同时保持车辆特定的输出；② 通过在车辆行驶过程中计算位移误差得到的 locomotion 信号作为自监督标签，天然捕获方向依赖和车辆动态差异；③ 构建方向索引的全局 BEV 成本图，使规划器可在不同朝向下评估地形难度。

**🔧 技术方法**

使用技术：RGB‑D 传感器融合生成六通道 BEV；双分支 ResNet‑18 提取颜色与高度统计特征；多任务回归头输出车辆特定成本；离散角度离散化 + 滑动窗口推断形成方向性成本图；Hybrid A* 规划结合占据图与方向成本；Nav2‑MPPI 控制器跟踪规划路径。

**📊 数据集**

数据集：在 CARLA 仿真环境中收集四台不同几何/动力学车辆（Jeep、VW Van、Toyota Hummer、Mercedes EQC）共 400 条轨迹，生成约 30,000 个地形补丁，每个补丁与对应的 locomotion 标签配对，按 70/15/15 训练/验证/测试划分。

**📈 对比分析**

比较方法：1) 方向无关平均成本 (dgt‑map(Iso))；2) 单任务训练 (dgt‑map(ST))；3) 仅占据图 (occ‑only)；4) 基于坡度的手工模型 (slope‑baseline)。实验在三种地图（Hill、Ridge、随机起止）中测量导航成功率。DGT‑MAP 在 Hill 与 Ridge 环境下分别达 80%/94% 与 16/26 的成功率，显著高于其他基线；在随机任务中多任务版在 Jeep 和 VW Van 上分别提升 70%→64% 与 54%→35% 的成功率，表明多任务学习在数据稀缺时尤为有效。

**⚠️ 局限性**

局限性：① 仅在仿真中训练与评估，缺乏对真实土壤、松散岩石等复杂地形的适应；② 需要每个目标车辆在训练集出现，无法零样本泛化；③ 成本图需离线生成，导致在未知环境下实时更新受限；④ 对每个方向需要独立推断，推理时间随角度分辨率线性增长；⑤ 低层控制未利用地形成本，导致规划偏离时难以即时补偿。

---

## 13. To Solve Bilevel Optimization with Nonconvex Lower Levels, We Need Second-Order Stationarity

**arXiv ID:** 2609.30501 | [PDF](https://arxiv.org/pdf/2609.30501v1)

**作者:** Zhiyao Zhang `[一作]` (Ohio State University), Jia Liu `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于二阶驻点的下层代理，设计 Perturbed Gradient Bilevel (PGB) 算法，并给出其 𝒪(T^-2/5) 的有限时收敛率，解决非凸下层的双层优化问题。

**💡 创新点**

创新点在于：①首次将二阶驻点（SOSP）作为下层代理，避免陷入下层鞍点；②通过 μ‑增广隐函数构造可计算的超梯度；③实现了无随机性下层的理论收敛保证。

**🔧 技术方法**

主要技术包括：二阶驻点判定的 Perturbed Gradient Descent（PGD）子循环；μ‑增广隐函数与隐函数定理求解超梯度；共轭梯度（CG）求解 Hessian‑逆；两时尺度的参数衰减策略。

**📊 数据集**

实验使用 HelpSteer 语料库对 Llama‑3.2‑3B‑Instruct 进行数据集筛选与模型微调。

**📈 对比分析**

与 PBGD、GALET、MEHA、SUN‑DSBO‑SE 等基线对比，PGB 在上层损失和训练时间上均优于所有基线，尤其在上层收敛速度和最终性能上实现了显著提升。

**⚠️ 局限性**

局限性：仅针对确定性 LLNC‑BLO；未考虑采样噪声/随机梯度；对大规模问题的内存与计算复杂度仍有待进一步优化。

---

## 14. Fully 3GPP-Compatible Long-Range Sensing for LEO-ISAC: A Window-Grid Processing Framework

**arXiv ID:** 2609.30653 | [PDF](https://arxiv.org/pdf/2609.30653v1)

**作者:** Yi Geng `[一作]` (Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences), Yu Zhao `[通讯]` (Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种窗口-网格（WG）处理框架，用于在低地球轨道集成感知通信（LEO-ISAC）中实现长距离双星测距，并解决了符号错位与信号时长不匹配的问题。

**💡 创新点**

创新点在于：1）完全兼容3GPP标准的发射波形，只在接收端实现新算法；2）利用双星几何确定的最短/最长到达时间，将绝对时延压缩为相对时延，从而仅需对相对时延区间进行窗口化处理；3）通过多窗口网格实现对不同相对时延段的独立解码，随后裁剪和抑制鬼峰，保证全可辨距离。

**🔧 技术方法**

采用的技术包括：OFDM信号、时间窗切片、FFT/2D-FFT信道估计、窗口-网格对齐、相对时延裁剪、鬼峰衰减分析以及仿真中的雷达方程与噪声模型。

**📊 数据集**

使用的“数据集”为仿真环境：双星系统（地面站和600 km 高度卫星）下的两个飞机目标，采用24 GHz载波、103.2 MHz总带宽、120 kHz子载波间距等3GPP标准参数，仿真覆盖 644–664 km 的双星测距。

**📈 对比分析**

与传统 OFDM 感知方法相比，WG 框架在 640 km 以上的双星距离下实现了 5–6 m 的测距误差（约为 0.001 %），并通过相对时延裁剪将鬼峰衰减至少 23 dB，成功抑制伪峰，满足超长距离测距需求。

**⚠️ 局限性**

主要局限：1）尚未实现轨道运动的多普勒补偿，导致速度估计误差；2）计算复杂度随 WG 数量线性增加，需并行处理；3）仅在仿真环境下验证，缺乏硬件实现与实时性能评估；4）对非常大相对时延段的多目标情况仍需进一步研究。

---

## 15. Dynamic Regret in Online Convex Optimization with Indicator Switching Costs

**arXiv ID:** 2609.30556 | [PDF](https://arxiv.org/pdf/2609.30556v1)

**作者:** Naram Mhaisen `[一作]` (TU Delft), George Iosifidis `[通讯]` (TU Delft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在线凸优化中的动态遗憾，特别是引入了指示切换成本的情况，即每当连续决策不同就会产生固定的惩罚。这种情况模拟了服务器激活、模型部署和缓存更新等启动开销。

**💡 创新点**

提出了一种元学习框架，结合了随机懒惰的FTRL基础学习者，并在二进制时间尺度上重启，通过一个关注移动的主学习者聚合它们的提议密度，并通过最大耦合连续混合样本动作。

**🔧 技术方法**

使用了元学习框架和随机懒惰的FTRL（Follow-the-Regularized-Leader）算法，结合了多尺度的重启机制。

**📊 数据集**

没有具体提到使用的数据集，但研究的背景涉及在线凸优化和动态遗憾的理论框架。

**📈 对比分析**

与现有的静态比较器的保证相比，提出的方法在动态遗憾上取得了最优的界限，且不需要先验知识。对于分段常数比较器，动态遗憾为𝒪̃(√((S_T+1) T))，对于路径长度受限的比较器，动态遗憾为𝒪̃(T^2/3 P_T^1/3)。

**⚠️ 局限性**

在指示切换成本下，无法保持统一性，虽然在分段常数比较器上可以达到𝒪̃(√(T(S_T+1)))，但在路径长度受限的比较器上，界限降级为𝒪̃(T^2/3 P_T^1/3)。

---

## 16. Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction

**arXiv ID:** 2609.30738 | [PDF](https://arxiv.org/pdf/2609.30738v1)

**作者:** Tianfang Xie `[一作]` (Georgia Institute of Technology), Wei Zhu `[通讯]` (Zhangjiang Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种层级多样性调度的 KV 缓存淘汰方法，通过结合平均注意力、注意力方差和冗余惩罚来选择缓存 token

**💡 创新点**

创新点在于：①将注意力方差与冗余惩罚融合成统一分数，实现类似最大边际相关（MMR）的多样性控制；②采用层级可调的系数配置（全局、分段、曲线）探索深度依赖的分数权重；③通过搜索验证仅使用全局常数已能显著提升多数任务，且在检索任务中进一步提升

**🔧 技术方法**

使用 Mistral‑7B 语言模型，基于 PyramidKV 的 KV 缓存框架，采用梯度自由的基于方差和余弦相似度的分数计算，采用随机拆分的开发集进行坐标搜索，最终在测试集上评估；此外，还对 Llama‑3.1‑8B 进行迁移实验

**📊 数据集**

评估基于 LongBench 的 16 个英文长文本任务，包括单/多文档问答、摘要、少量学习、检索计数、代码补全等

**📈 对比分析**

与 SnapKV、PyramidKV 等基线对比，使用全局常数 λ1=0、λ2=-0.15 时，宏观平均分从 33.31 提升至 34.44（+1.13 分），在 13/16 任务中取得最佳；使用分段搜索可进一步提升至 35.15 分（+1.84 分），在检索任务中提升 9.6 分；在不同缓存预算（32、64、128）下均保持优势，并能直接迁移分段配置

**⚠️ 局限性**

局限性包括：实验仅在一个基准套件上进行，样本量有限；搜索过程受路径依赖影响，单一随机拆分可能导致偶然性；未在多模型或更大规模实验验证；缺乏对实时推理中的实时性能和能耗评估

---

## 17. Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4

**arXiv ID:** 2609.30716 | [PDF](https://arxiv.org/pdf/2609.30716v1)

**作者:** Amanda Fitch `[一作]` `[通讯]` (Google), Amanda Fitch (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了 Google 的 Gemma 4-e4b 语言模型在面对两份相互矛盾的文档时，究竟会优先采纳哪份信息。

**💡 创新点**

通过设计完全对平衡的四叉交叉实验，首次实现了将词汇偏差、文档表头框架与文档顺序位置三种效应在同一实验中彻底分离，并发现语义框架的影响远大于顺序偏差，且后者高度可变。

**🔧 技术方法**

采用了对数概率（nat）度量、BCa 复抽样置信区间、TOST 等统计方法，对 Gemma 4-e4b 进行前向推理，结合四种跨越顺序和角色分配的交叉配置。

**📊 数据集**

使用了 13 组英语项目管理名词对（如 Agile/Waterfall、Scrum/Kanban 等），共进行 784 次前向推理，形成实验数据集。

**📈 对比分析**

通过对四种配置的平均与差异计算，比较来源框架与顺序位置的效应大小，发现框架优势约为 4:1，复制提示并不会削弱首位偏差，表头与主体措辞的组合会调节偏差强度。

**⚠️ 局限性**

实验仅覆盖单模型、单轮、短文本、英语项目管理词汇，未检验多文档、长上下文或其他模型的表现，故结果在更广泛场景下的适用性有限。

---

## 18. Impedance Cloning: Learning Equilibrium Point Parameters for Contact-Rich Manipulation

**arXiv ID:** 2609.30842 | [PDF](https://arxiv.org/pdf/2609.30842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 19. ADF-EA: A Unified Execution Assurance System for Agent Device Foundation

**arXiv ID:** 2609.30691 | [PDF](https://arxiv.org/pdf/2609.30691v1)

**作者:** Xuechun Li `[一作]`, Hang Huang `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了ADF-EA架构，利用设备能力合同（DCC）统一规划与执行安全，支持持久执行状态、证据驱动恢复、重试与状态修复。

**💡 创新点**

核心创新在于：1）将设备能力的调用条件、预期效果、证据要求和中断/恢复规则统一为DCC；2）将规划与执行在同一合同语义下对齐；3）设计持久执行状态模型，区分已验证进度、未决结果与预算，确保恢复时不重复物理动作。

**🔧 技术方法**

采用的技术包括：基于LLM的代理规划、DCC规范化与验证、持久化执行状态与预算管理、证据资格化与观察重用、正式的状态转换规则与条件完备性证明。

**📊 数据集**

实验使用的场景/数据集有：基于模拟的过程控制、家电、机器人操作；AI2-THOR、Meta-World、Heating模拟器；此外在多模型与多代理框架的对比中使用不同LLM（DeepSeek、GLM、Qwen等）。

**📈 对比分析**

通过与公开算法ToolGate、Verified Tool Calls (VTC) 的直接对比以及对Direct的基线，评估指标包括：目标确认完成率、证据合格完成率、违规调度次数、重复调用次数、不可用能力调用等。ADF-EA在所有14个完成条件下均实现证据合格完成且无违规调度，显著优于对照组。

**⚠️ 局限性**

局限性：依赖准确的设备能力合同和可靠的证据获取；在异步部分效应、并发预算或多任务依赖失效时需要额外保证；真实硬件部署需保证传感与控制间的时序可靠；当前实现主要针对顺序、单一状态设定任务，复杂图形任务的自动恢复仍需进一步研究。

---

## 20. Beyond the Last Truffula Tree: SustainAI - A Water-Aware, Closed-Loop Framework for Environmentally Accountable AI

**arXiv ID:** 2609.30747 | [PDF](https://arxiv.org/pdf/2609.30747v1)

**作者:** Farnaz Farid `[一作]` (Western Sydney University), Sami bin Azad `[通讯]` (Southwest Jiaotong University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了SustainAI框架，实现水资源监测与推理的闭环，并在推理过程中加入基于幻觉的惩罚与基于区域水压的路由决策。

**💡 创新点**

创新点在于首次将实时水表计、幻觉惩罚机制和水压路由算法结合；并通过“Care by Design”把社区责任与路由决策相联。

**🔧 技术方法**

使用了实时水计量传感、WRI Aqueduct水压指数驱动的路由算法、闭环反馈与动态权重调整、前端可视化水足迹，以及小语言模型推理与水成本计算模型。

**📊 数据集**

在健康误信息检测任务中使用Gemma-2-2B-IT小语言模型，进行1335条推理样本，并结合各地区水压指数与数据中心位置数据。

**📈 对比分析**

通过与五个国家（US、CN、MY、UK、PT）九个数据中心的对比，显示水足迹差异超过11倍；幻觉惩罚后每个正确答案的有效水成本从0.3 mL升至约1.66 mL，提升约5.6倍。

**⚠️ 局限性**

局限性包括样本量不足、仅评估单一模型、未实时验证反馈机制、误信息集仅限健康领域，难以推广到其他类型内容。

---

## 21. QSV: Quat-Sphere-Vision for Coupled Quaternion Attention on Spherical Lattices

**arXiv ID:** 2609.30592 | [PDF](https://arxiv.org/pdf/2609.30592v1)

**作者:** Nicholas Foley `[一作]` (University of Texas at San Antonio), Amanda Fernandez `[通讯]` (University of Texas at San Antonio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种稀疏球面视觉模型 Quat‑Sphere‑Vision（QSV），用单个单位四元数在图中同时完成注意力权重与特征传输。

**💡 创新点**

创新点是将传统注意力的 {W_Q, W_K, W_V} 三重投影压缩为单一相对四元数，既提供路由权重又实现旋转对称特征传输。

**🔧 技术方法**

使用稀疏 kNN 图、Fibonacci 球面布局、Hamilton 结构线性、Riemannian Adam 以及四元数的旋转运算。

**📊 数据集**

在 CIFAR‑10 和 CIFAR‑100 两个小型自然图像数据集上训练评测。

**📈 对比分析**

通过匹配参数、图结构的对照实验比较：QSV 在同等参数下略低于标准注意力（87.3% vs 85.9%），平面网格控制可提升至 91.1%，表明模型核心瓶颈在于球面布局而非四元数核。

**⚠️ 局限性**

局限在于与强大卷积/变压器基线相比性能不足，主要受球面采样和稀疏图传输对局部纹理的限制，且实验仅在小规模、低分辨率数据集上验证。

---

## 22. Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence

**arXiv ID:** 2609.30535 | [PDF](https://arxiv.org/pdf/2609.30535v1)

**作者:** Dries Rooryck `[一作]` (Harvard University), Kianté Brantley `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在英语、荷兰语和中文的100M词限制下，训练了小型GPT‑2解码器模型，使用三阶段（词级→句级→单语）代码切换语料进行预训练；

**💡 创新点**

创新点在于将代码切换构成的细粒度数据与学习课程顺序相结合，显著提升了跨语言表示的对齐与下游任务表现；

**🔧 技术方法**

技术包括GPT‑2‑small架构、字节级BPE tokenizer、Adam优化器、cosine学习率调度、cross‑domain similarity local scaling（CSLS）以及bitext retrieval precision@1评估；

**📊 数据集**

使用的数据集为BabyBabelLM的英语、荷兰语和中文语料，并通过LLM生成词级和句级代码切换的合成语料；

**📈 对比分析**

通过与无代码切换、随机数据排序以及四种对照语料的对比，利用BabyLM评测套件和跨语言检索任务，CS+课程预训练模型平均提升约0.36分，跨语言检索率提升显著；

**⚠️ 局限性**

局限性包括仅验证三语模型、模型规模有限、代码切换生成成本高、未探究对更远或低资源语言的泛化性。

---

## 23. Verification of Compiler-to-Accelerator Mappings for Machine Learning Accelerators

**arXiv ID:** 2609.30651 | [PDF](https://arxiv.org/pdf/2609.30651v1)

**作者:** Akash Gaonkar `[一作]` (Princeton University), Aarti Gupta `[通讯]` (Princeton University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了 Bolt 框架，针对 ML 编译器中将 IR 代码映射到粗粒度硬件加速器原语的过程进行形式化验证，保证编译后程序与硬件原语的功能等价。

**💡 创新点**

创新点包括：①首个面向粗粒度加速器原语的验证框架；②通过两步模板（Sync‑Skeleton 对齐循环，Layout‑Sketch 关联张量布局）实现循环与张量数据的自动对齐与关系推导；③无需依赖编译器内部信息，直接验证黑盒映射。

**🔧 技术方法**

使用技术包括：产品程序（product programs）验证方法；Hex 语言（Intermediate Verification Language）和 ILA 模型描述硬件语义；自定义重写规则、参数替换与自定义数值规则；量化实例化与 SMT（Z3）求解；以及手工制定的循环与布局关系模板。

**📊 数据集**

验证数据集主要是两个开源 ML 加速器的多种映射：FlexASR（linear‑layer、pooling）和 HLSCNN（2D 卷积、带累加的卷积），使用对应的硬件/应用参数实例进行验证。

**📈 对比分析**

对比方法：与 CHC/ BMC 等传统验证工具对比；通过 Z3 求解生成的 VC 在实际硬件参数下成功完成验证；验证时间随循环数、嵌套深度和数据布局复杂度增长，但相较于未对齐或无模板的方式显著降低求解时间；在更大规模参数下仍能在合理时间内完成（若降尺度则更快）。

**⚠️ 局限性**

局限性：①需要人工提供重写规则、布局模板和部分循环不对齐时的 invariants；②目前未实现完全自动化；③对非线性算术的处理仅通过具体化参数；④仅验证粗粒度原语映射，未覆盖完整编译流程；⑤依赖硬件 ILA 语义模型的可用性。

---

## 24. Frequency-Modulated Piezoelectric Haptic Display

**arXiv ID:** 2609.30626 | [PDF](https://arxiv.org/pdf/2609.30626v1)

**作者:** Boyuan Liang `[一作]` (University of California), Masayoshi Tomizuka `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种共享源频率调制触觉显示，通过频率编码替代幅度编码，利用单一高压放大器和多通道模拟开关实现多像素控制。

**💡 创新点**

创新点在于采用频率调制替代传统幅度调制，并提出SSFM架构共享放大器，大幅降低每像素驱动硬件需求，支持高密度可穿戴显示。

**🔧 技术方法**

使用频率调制触觉显示技术、共享源频率调制结构、低压阈值触发的模拟开关、SPI控制以及TDK PowerHap 1204压电振荡器。

**📊 数据集**

未使用公开数据集，而是基于六组志愿者实验收集的感知准确率数据。

**📈 对比分析**

通过对比实验（频率辨识、位置定位、多指感知、运动方向与速度）评估性能，频率对比全部正确，运动方向完全正确，速度辨识准确率91-94%，定位准确率≥90%。

**⚠️ 局限性**

局限在于仅采用粗糙频率级别，未进行精细阈值研究；实验仅基于原型并非真正可穿戴，振幅有限，缺乏对皮肤耦合和更细感知层面的评估。

---

## 25. Epstein Files Engine: Agentic Search for Investigative Journalism

**arXiv ID:** 2609.30611 | [PDF](https://arxiv.org/pdf/2609.30611v1)

**作者:** Duy K. Nguyen `[一作]` (New York Times), Zach Seward `[通讯]` (New York Times)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并部署了Epstein Files Engine，一款基于LLM的查询规划器和BigQuery SQL引擎，帮助记者快速检索并验证三百万页多媒体文档中的新信息。

**💡 创新点**

创新点在于将自然语言问题转化为结构化SQL查询并结合Diff这套多模态去重技术，既提高检索效率，又通过新颖度放大器为记者挖掘真正新颖的线索。

**🔧 技术方法**

技术实现包括LLM（如GPT）做查询规划、Google BigQuery执行SQL、文本语义嵌入与视觉感知哈希拼接为449维向量、余弦相似度做去重、LibreChat+Agent接口、以及Diff的近似最近邻检索。

**📊 数据集**

使用的数据集为1.1 M封邮件、3 M页PDF、200 k图片及音视频的1月30日司法部发布文件、Times自有报道档案以及外部关于Epstein的新闻头条。

**📈 对比分析**

通过对比人类标注的近邻匹配，Diff在τ = 0.92时达到≈0.28精确度、0.86召回率；Engine在数百名记者的使用中共生成4,500+查询，产生20+篇正式报道，表明实用性与效能。

**⚠️ 局限性**

局限性包括缺乏对比实验（如去除内部档案或外部头条后效果）、Diff验证仅在新闻高峰后进行、未与其他去重方法系统对比、仅在单个新闻机构部署，且评估样本有限。

---

## 26. CRC-Router: Risk-Constrained Routing for Medical Agentic AI Systems

**arXiv ID:** 2609.30714 | [PDF](https://arxiv.org/pdf/2609.30714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 27. A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models

**arXiv ID:** 2609.30292 | [PDF](https://arxiv.org/pdf/2609.30292v1)

**作者:** Fanji Yang `[一作]` (Guizhou University of Finance and Economics), Mingsen Deng `[通讯]` (Guizhou University of Finance and Economics)

**通讯引用:** 4600 | [OpenAlex ID](https://openalex.org/A5015316402)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了2018-2026年假评论检测研究，按证据来源（文本、情感、行为、时间元数据、用户-产品图谱、多模态内容、外部知识、LLM生成信号）和融合层次（特征级、表示级、图级、决策级）构建分类框架，并梳理了从传统机器学习到深度学习、PLM和LLM的技术演进与融合方法；

**💡 创新点**

从信息融合视角和LLM时代提出全新的分类与评估框架，系统性地将文本、行为、结构与多模态证据以及LLM生成与检测的双向挑战纳入同一视野，并揭示LLM生成假评论对检测系统带来的新难点；

**🔧 技术方法**

综述了传统机器学习、深度CNN/RNN/注意力网络、预训练语言模型（BERT、RoBERTa、ELECTRA等）以及大型语言模型（GPT、ChatGPT、LLaMA、GLM等）的直接判别、提示增强、特征融合、生成式对抗、数据增广、图神经网络等多种技术；同时采用系统化文献检索、编码与审阅流程构建211篇论文的研究语料；

**📊 数据集**

评估基准主要包括Amazon、Yelp、OpSpam等主流公开数据集，并讨论了标签构造方式（过滤算法、众包、人工标注、规则）、数据拆分与评估协议对实验结果的影响；

**📈 对比分析**

通过对比实验表明：① PLM微调与特征融合可显著提升F1/Accuracy至0.90以上；② LLM直接判别在零样本或少样本场景下性能偏低（≈45%），但提示增强与生成式增广可显著提升准确率（≥93%）；③ 但所有方法在LLM生成假评论检测上仍面临较大性能落差；整体趋势显示，模型越是融合多源证据与LLM能力，其性能越趋稳健；

**⚠️ 局限性**

存在的局限包括：① 标签构造与数据偏差导致模型泛化受限；② LLM生成假评论的多样性与可解释性缺乏统一评测标准；③ 多模态与图谱融合对数据收集与计算资源要求高；④ 现有方法缺乏对抗生成与检测的双向动态稳定性与解释性研究；

---

## 28. Managing Iterative Hybrid Quantum-Classical Optimization as a First-Class Scientific Workflow

**arXiv ID:** 2609.30532 | [PDF](https://arxiv.org/pdf/2609.30532v1)

**作者:** Giuliana Siddi Moreau `[一作]` (CRS4 - Center for Advanced Studies Research and Development in Sardinia), Lidia Leoni `[通讯]` (CRS4 - Center for Advanced Studies Research and Development in Sardinia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一层面向科学工作流的管理层，用于调度、监控和容错 hybrid quantum‑classical 优化循环（拆分–求解–聚合）。

**💡 创新点**

创新点在于：① 引入反馈驱动的终止节点与基于残差的收敛判定；② 采用仲裁（quorum）聚合与温启动恢复，支持在单轮内从 QPU 迁移到经典后端；③ 统一的、后端无关的 provenance 模式，记录设备、射击预算和子问题执行细节。

**🔧 技术方法**

使用技术包括：动态工作流引擎（可与 Parsl、Dask、PyCOMPSs 等 WMS 交互）、自定义执行器接口、作业失败注入模型、QPU 共享与排队管理，以及基于 BQM 的 ADMM 与层级划分两种拆分模式。

**📊 数据集**

实验数据集来源于两类真实应用：多仓库车辆路径规划（MDCVRP）和可再生能源社区能源共享配置（ADMM），并分别采用密集 QUBO 表达式。

**📈 对比分析**

对比方法：将工作流层与无层驱动脚本、不同后端（模拟器、仿真器、IQM QPU）进行性能评估。结果显示，工作流层的 orchestration 开销占总时间约 61%；迭代共识模式的 barrier 费用高达 78%，而层级划分模式仅约 6%；在注入失败场景下，工作流层能保持完成率并通过 speculatively 重新执行抑制延迟。

**⚠️ 局限性**

限制包括：① 评估主要在 HPC 环境模拟下完成，未覆盖真实 QPU 队列的动态调度特性；② 尚未与无层驱动脚本或完整的通用 WMS 进行基线对比；③ 对嵌套拆分模式未做实验；④ 可扩展性（极大 fan‑out）与多调度器间的跨设施迁移性仍待进一步验证。

---

## 29. From Routing Delay Shifts to Silent Data Corruption: Neutron-Induced SEU Effects in AXI-Based Zynq UltraScale+ MPSoCs

**arXiv ID:** 2609.30538 | [PDF](https://arxiv.org/pdf/2609.30538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 30. Nearest but Not Dearest: Shared Curator-Feedback Infrastructure for Content-Only Search and Recommendation

**arXiv ID:** 2609.30568 | [PDF](https://arxiv.org/pdf/2609.30568v1)

**作者:** Matt Sandler `[一作]` `[通讯]` (Feed.fm), Matt Sandler (Feed.fm)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个共享的内容检索堆栈，利用专业策展人提供的“拒绝”反馈将失败拆分为声学（sound）与上下文（context）两类，并将这两类反馈分别路由到约束过滤器（处理上下文失败）和嵌入重权重头（处理声学失败）上，统一服务于搜索式和推荐式两种入口；在此过程中实现了单一堆栈下跨模式（搜索与推荐）检索性能提升。

**💡 创新点**

核心创新是提出“声学 vs 上下文”失败分解作为反馈路由依据，将专业反馈在共享堆栈中按失败性质分流到不同的工程层面；这一做法在内容仅检索（无行为日志）场景下首次展示了能让少量策展人循环就能显著降低检索拒绝率，并在搜索与推荐两种入口共享同一改进机制。

**🔧 技术方法**

技术方案包括：1) LAION‑CLAP joint audio–text encoder 提供 512 维嵌入；2) 精确余弦检索；3) 约束过滤器（SQL 规则谓词）在候选生成阶段过滤上下文失配；4) 线性投影重权重头（metric‑learning head）在表示层对声学失配进行再加权；5) 通过二次采样的训练-验证分层实现对声学负例的学习。

**📊 数据集**

使用 Feed.fm 的授权音乐目录（57,758 首曲目）及其对应的 LAION‑CLAP 嵌入；反馈数据来自四位内部策展人共 1,200 条（每轮）评分，其中包含拒绝原因编码（sound / ctx / other）。

**📈 对比分析**

方法对比：基线（仅余弦检索 + 预过滤）与实验（加入约束过滤 + 重权重头）在两轮评估中，拒绝率从 38.17% 降至 28.83%（-9.33pp，-24.5% 相对），McNemar 检验显著（χ²=22.37，p≈2.2e‑6）。分解分析表明过滤器贡献 4.08pp，重权重头贡献 5.25pp，合计 9.33pp，验证两机制均有效。

**⚠️ 局限性**

局限性包括：1) 过滤器与重权重头未做单独因果分离；2) 仅在单一目录与单一 encoder 上测试，缺乏跨数据集验证；3) 仅评估策展人拒绝率，未直接测量听众行为影响；4) 4 位策展人规模有限，可能限制结果的普适性；5) 未对搜索式入口进行实验，跨模式效果尚未量化；6) 训练数据稀缺导致重权重头可能受噪声影响。

---

## 31. Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events

**arXiv ID:** 2609.30746 | [PDF](https://arxiv.org/pdf/2609.30746v1)

**作者:** Isabella S. Thiel `[一作]` (Massachusetts Institute of Technology), Themistoklis P. Sapsis `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将对齐的粗糙模拟器产生的噪声扰动群集作为局部不稳定几何的无侵入传感器，并通过FiLM模块将其协方差特征注入到现有的时间序列校正模型中，以实现稀有事件的高效仿真。

**💡 创新点**

创新点在于利用低噪声状态下的群集协方差与有限时不稳定方向的对应关系，构建了一种机制感知的装配条件化插件，而非单纯的数据驱动或模型结构改进。

**🔧 技术方法**

技术实现包括：对齐的粗糙模型与高分辨率参考的nudging、低噪声随机微分方程群集、协方差动力学分析、FiLM特征线性调制以及Transformer残差注意力和概率递归STORN两种后端。

**📊 数据集**

实验使用两套数据集：低维三阶混沌系统的短轨迹与高分辨率二维层QG系统（128×128）以及其对应的24×24粗糙网格。

**📈 对比分析**

与无条件化STORN或Transformer基线相比，在仅有50个时间步高分辨率训练数据的极端事件评估中，FiLM-STORN在KL散度、Log‑L1误差、超阈值频率和空间超阈值面积分布等指标上显著优于同等训练量的基线，并在超长训练量（1000步）基线下仍保持竞争力。

**⚠️ 局限性**

局限性包括：需满足小噪声、局部线性化和谱间隙假设；在粗糙模型与参考统计已接近时提升有限；额外的群集计算增加算力成本；仅在动力学不稳定性显著且极端事件由有限时拉伸驱动的系统中有效。

---

## 32. Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms

**arXiv ID:** 2609.30558 | [PDF](https://arxiv.org/pdf/2609.30558v1)

**作者:** Jiaqi Ding `[一作]` (University of North Carolina at Chapel Hill), Guorong Wu `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MemProbe 框架，利用认知科学中的干扰、误信息、巩固和再巩固四种实验范式，对代理记忆系统的稳定性-可塑性平衡进行行为诊断。

**💡 创新点**

创新点在于：①将认知记忆实验逻辑迁移到代理记忆评估；②设计可复用的四种诊断范式和行为指标；③生成 56 轮实验套件并从行为维度生成可解释的稳定-可塑性谱；④展示同等准确率系统在更新行为上的显著差异。

**🔧 技术方法**

使用统一的 Gemini-3-Flash 阅读器进行结构化打分，基于记忆增量协议对六种增量记忆系统（Mem0、Zep/Graphiti、LangMem、Cognee、A‑MEM、MemoryOS）以及检索基线进行评估，并引入基于实验范式的行为指标。

**📊 数据集**

数据集为自生成的 56 轮诊断实验（覆盖干扰、误信息、巩固、再巩固四个范式）以及相应的“源信息”“时间标签”等结构化金标准；检索基线采用 Oracle RAG、Naive RAG、Time‑aware RAG 等。

**📈 对比分析**

通过对齐输入协议、统一检索器和结构化评分，比较六个系统在整体准确率、可塑性、稳定性以及 SP‑Balance 等指标上的表现。A‑MEM 以 88.4% 的整体准确率和 88.3% 的 SP‑Balance 领先；Cognee 在稳定性上更强；Mem0 及 Graphiti 等系统在某些范式上表现相近但行为谱截然不同。

**⚠️ 局限性**

局限性在于：①评测仅覆盖有限的 56 轮实验，未扩展到多语言、多任务或真实用户历史；②使用统一阅读器和结构化评分，可能忽略不同阅读器对源信息的影响；③部分对话表面生成可能带有 LLM 的偏差，影响实验可复现性。

---

## 33. EA-Ops: Git-Native Architecture as Code for Continuous Enterprise Architecture Governance

**arXiv ID:** 2609.30593 | [PDF](https://arxiv.org/pdf/2609.30593v1)

**作者:** Vahid Tavakkoli `[一作]` (University of Klagenfurt), Kyandoghere Kyamakya `[通讯]` (University of Klagenfurt)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了EA‑Ops，一套 Git 原生的 Enterprise Architecture‑as‑Code 框架，支持 YAML 模型、治理规则验证、变更影响分析以及静态门户生成。

**💡 创新点**

将语义化的 EA 建模与 GitOps 工作流、可执行治理、确定性验证和可重现 CI 评估相结合，实现了从代码到文档的一体化。

**🔧 技术方法**

使用 Python + PyYAML、GitHub Actions CI、ArchiMate 3.2 关系矩阵验证、图遍历算法进行影响分析、以及静态 HTML/JS/SVG 门户渲染。

**📊 数据集**

采用合成基准（100–50 000 对象，200–100 000 关系）、240 次故障注入试验，以及虚构的 Metroville Digital Permit 参考架构（65 个对象、85 条关系）和 10 条受控变更场景。

**📈 对比分析**

通过对 240 次故障注入试验获得精确率/召回率/F1 全部为 1.0；在 50 000 对象规模下验证中位时间 52.6 s，影响遍历 627 ms，Markdown 报告生成 823 s；100 000 对象端到端任务因 180 min CI 限制而超时。

**⚠️ 局限性**

报告生成是主要扩展瓶颈；影响分析仅使用无向连通性；评估仅基于合成数据和虚构案例，真实 EA 仓库的拓扑、复杂度和治理规则可能不同。

---

## 34. Memory-Aware Multi-Sensor Perception for Efficient and Safe Navigation in Dynamic Environments

**arXiv ID:** 2609.30495 | [PDF](https://arxiv.org/pdf/2609.30495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 35. Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis

**arXiv ID:** 2609.30841 | [PDF](https://arxiv.org/pdf/2609.30841v1)

**作者:** Thong Bach `[一作]` (Deakin University), Truyen Tran `[通讯]` (Deakin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于能量景观理论，提出了三种无训练的检测信号，用于识别和防御扩散式大型语言模型中的 jailbreak 攻击；

**💡 创新点**

创新点在于将安全对齐解释为在 denoising 能量景观中形成安全与有害基底之间的能量障碍，并从中导出基于步骤 0 比例、SED 与 ΔE 斜率的三重检测方案，覆盖已知攻击的所有逃逸策略；

**🔧 技术方法**

采用能量最小化框架、Fisher 信息几何、对数几率比率与速度梯度分解等技术，计算无训练的 logit 空间信号；

**📊 数据集**

使用 LLaDA-8B、LLaDA-1.5、Dream-7B 与 LLaDA-MoE-7B 等扩散语言模型，评估 HarmBench、AdvBench、AlpacaEval、XSTest 等标准数据集；

**📈 对比分析**

通过 AUROC 与阈值级召回率对比，三信号联合检测在所有模型与攻击场景下均能保持 AUROC≥0.83，单一信号在不同架构上性能波动显著，表明三信号互补；

**⚠️ 局限性**

局限性包括仅评估现有攻击、对中文提示的适配需要重新构造词表、阈值级召回率随模型架构和生成长度变化而波动，以及未针对完全自适应攻击者进行实验。

---

## 36. Simulation-Efficient Analog Circuit Yield Optimization via Monte Carlo Zeroth-Order Gradient Estimation

**arXiv ID:** 2609.30678 | [PDF](https://arxiv.org/pdf/2609.30678v1)

**作者:** Liyan Tan `[一作]` (University of California Santa Barbara), Zheng Zhang `[通讯]` (University of California Santa Barbara)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于零阶蒙特卡罗随机梯度下降（ZO‑MC‑SGD）的模拟高效模态工艺变异下模拟电路良率优化方法，将规格余量转换为光滑损失，并通过配对过程样本的扰动估计梯度。

**💡 创新点**

创新点在于：①利用规格余量构造与良率高度相关的光滑损失，并通过Spearman秩相关检验保证其与真实良率的一致性；②采用配对扰动（共享相同过程样本）的零阶梯度估计，实现无显式梯度的局部下降方向；③证明该估计在高斯平滑目标下无偏，方差与过程维度无关，且给出样本复杂度与收敛性上界；④显著降低模拟预算（最高可达8倍）。

**🔧 技术方法**

主要技术包括：零阶随机梯度下降、有限差分梯度估计、softplus 规格余量惩罚、Spearman秩相关对齐、Adam 自适应学习率、Monte Carlo 过程采样、理论收敛性分析与方差估计。

**📊 数据集**

使用五个模拟电路基准（1 阶 CS、3 阶 CS、5 阶 CS、2 阶 Miller、3 阶 Miller），设计维度最大 30，过程维度最大 42，采用 ngspice + Level‑1 设备模型，并假设已知高斯匹配分布（Pelgrom 规模）来生成过程样本。

**📈 对比分析**

在与五种基准（BO、CMA‑ES、PSO、TuRBO、RobustAnalog）相同的 SPICE 预算下进行比较。ZO‑MC‑SGD 在所有基准上均能在 50–200 次模拟内达到平均良率 0.95，而其他方法往往需要 200–800 次甚至 3200 次，部分基准根本未达标。高精度评估显示其良率可达 0.934–1.000，表明其在预算紧张时具有明显优势。

**⚠️ 局限性**

局限性包括：①实验仅覆盖到 42 维过程变量，尚未验证在数百维甚至千维的实际大规模电路中的表现；②需要已知的过程分布；③对软加权阈值的校准仍需手工或额外计算；④理论证明针对平滑目标，对真实良率的收敛性仍需经验验证；⑤在高度相关或分布漂移的工艺变异下，仍需进一步验证其鲁棒性。

---

## 37. GRACIDIT: Graph-Circuit Digital Twin for Configuration-Induced Routing Delay Prediction in Zynq UltraScale+ FPGAs

**arXiv ID:** 2609.30534 | [PDF](https://arxiv.org/pdf/2609.30534v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 38. Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification

**arXiv ID:** 2609.30467 | [PDF](https://arxiv.org/pdf/2609.30467v1)

**作者:** Heyuan Huang `[一作]` (Johns Hopkins University), Mark Dredze `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了一套针对检索‑验证流程的多维错误分类体系，并通过LLM‑as‑Judge自动化方式在无黄金答案的开放式医学事实检验中大规模标注与分析。

**💡 创新点**

创新点包括：①首次在开放式医学事实验证中系统性定义检索与验证两阶段的错误模式；②利用LLM自动诱导模式快速生成完整代码本；③对检索与验证过程进行细粒度评估，揭示SOTA系统的根本瓶颈。

**🔧 技术方法**

使用的技术主要有：多种检索模型（MedCPT、RRF‑2/4、Qwen3‑Embedding‑8B）、多语料库（MEDIC、Google、Google Scholar）、LLM验证器（GPT‑5.4、MedGemma 27B、Gemma 3 27B、Qwen3.6‑27B、Mistral Small 3/4），以及LLM‑as‑Judge自动模式用于错误标签生成。

**📊 数据集**

实验数据集包括开放式医学问答集MedExpert（及其金标注子集MedExpert‑gold）、SciFact、HealthVer、CovidFact、以及大规模医学语料库MedRAG（MEDIC）等。

**📈 对比分析**

在四种检索器、三种知识库、六种验证器的交叉实验中，用Recall、F1等指标对比；结果显示，即便采用SOTA检索与验证组合，开放式医学场景下的F1也仅在0.06–0.08之间，模型规模扩大、推理长度增加、医学微调或语料扩展均未显著提升性能。

**⚠️ 局限性**

限制包括：缺少黄金证据标签，无法评估检索精度；仅基于MedExpert/MedRAG的数据，未涵盖时间维度；LLM‑as‑Judge标注受API预算限制；模型评估仍无法完全覆盖所有错误模式。

---

## 39. OpenHail: An Event-Driven Gymnasium Environment for Electric Ride-Hailing Fleet Control

**arXiv ID:** 2609.30628 | [PDF](https://arxiv.org/pdf/2609.30628v1)

**作者:** Tommaso Schettini `[一作]` (Concordia University), Jorge E. Mendoza `[通讯]` (HEC Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个开源 Gymnasium 环境（OpenHail），用于电动出租车队伍的联合控制（请求分配、重新定位与充电），并提供固定大小的观察与动作空间、决策时钟机制、充电设施与电池动力学、基线策略与评估工具。

**💡 创新点**

创新点包括：①将请求分配、车辆重新定位与充电整合为单一固定尺寸的联合动作接口；②引入可配置的决策时钟，支持事件驱动、周期性、混合以及策略请求的交互方式；③在模拟器中实现了第一入先出（FIFO）的充电队列和电池能量动态；④提供完整的可复制实例、基线策略、验证与性能测试框架。

**🔧 技术方法**

技术手段主要有：Python + Gymnasium（Gymnasium API）实现事件驱动仿真；使用 NumPy、PyTorch 进行向量化计算与强化学习（PPO）；利用 pyhailing 作为基础框架；提供可配置的决策时钟与动作可行性掩码；实现基线分配/重新定位策略与强化学习策略。

**📊 数据集**

数据集：使用纽约市历史行程记录生成的请求序列（含时间、起点、终点、行程距离），以及与之配套的服务区域、充电设施位置与容量配置。

**📈 对比分析**

比较方法：将三类控制器（最近邻基线、随机可行基线、周期性 PPO 训练策略）在同一实例集（N=100,500,1,000,2,000 车辆；D=5,10,20,40 充电/重新定位点）下进行 5 对一天的仿真；测量总耗时、每决策时钟耗时及强化学习训练时间。结果显示：PPO 需要 1.6–2.2 倍于最近邻的总仿真时间；在 2,000 车辆、40 位置时，PPO 每周期平均耗时约 1.06 ms，仍在可接受范围内；训练耗时从 0.54 小时（100 车辆）到 21.03 小时（2,000 车辆、40 位置）不等。

**⚠️ 局限性**

局限性：①未考虑路网拥堵与路径规划，仅使用恒定速度与线性能耗；②充电设施仅支持 FIFO 排队，未考虑不同功率的充电桩或多级充电策略；③事件驱动机制虽灵活，但对高频率事件可能导致大量内部更新；④评估仅基于纽约市数据，缺乏多城市或多样化需求场景的验证；⑤未探讨多智能体交互或协同学习的效果。

---

## 40. Learning Vision-Based Agile Gap Traversal: Differentiable Simulation with a Warm-Started Critic

**arXiv ID:** 2609.30696 | [PDF](https://arxiv.org/pdf/2609.30696v1)

**作者:** Nuthasith Gerdpratoom `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种两阶段强化学习框架，利用可微分仿真和预训练的 privileged critic 学习视觉感知的敏捷 gap‑traversal 策略。

**💡 创新点**

创新点包括：① 采用短时段 quasi‑analytical policy gradient (QPG) 避免对视觉渲染进行反向传播；② 通过热启动 critic（从专家阶段迁移过来）显著提升视觉策略的学习效率和稳定性；③ 无需对专家策略重新训练即可适配不同的无人机动力学参数，增强泛化能力；④ 只使用二值门掩模作为视觉输入，避免纹理、光照等干扰。

**🔧 技术方法**

使用的技术：可微分仿真（基于 Isaac Sim + JAX）、QPG 与 SHAC 结构、热启动 critic、简化的 Jacobian 模型、二值门掩模渲染、GRU‑CNN 视觉感知网络、行为克隆（DAgger）验证。

**📊 数据集**

数据集：在 Isaac Sim 中合成的多种门形（矩形、椭圆、梯形、三角形）以及不同滚转角的仿真环境；真实世界实验使用 Motion‑Capture 记录的门与无人机姿态，用于在线生成二值掩模。

**📈 对比分析**

对比方法：BPTT、PPO、全时段 BPTT、冷启动 critic、冻结 critic、单摄像头配置、行为克隆（BC）。结果显示：QPG + 热启动 critic 的视觉策略成功率最高（≈0.893），比 BPTT 提升约 50%，比冷启动 critic 提升约 30%；在未见过的门形下仍能通过；对动力学参数变更时，BC 成功率显著下降，而两阶段方法保持稳定。

**⚠️ 局限性**

局限性：二值门掩模是通过运动捕捉生成的，未包含感知误差；需要先验的门位置估计；目前仅在受限空间中验证，未充分测试更复杂环境；缺乏从原始 RGB 图像直接学习的 end‑to‑end 方案。

---

## 41. Why Clipping Matters in AdaGrad? Toward a High-Probability Theory under Generalized Smoothness

**arXiv ID:** 2609.30276 | [PDF](https://arxiv.org/pdf/2609.30276v1)

**作者:** Alokendu Mazumder `[一作]` (Indian Institute of Science), Punit Rathore `[通讯]` (Indian Institute of Science)

**通讯引用:** 343 | [OpenAlex ID](https://openalex.org/A5001120317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

分析了在广义平滑性和重尾噪声下的同步坐标AdaGrad，展示了未剪切的AdaGrad在重尾噪声下可能导致的几何失调，并证明了剪切可以修复这一问题。

**💡 创新点**

提出了剪切不仅是鲁棒性启发式，而是适应几何的结构稳定器，并提供了在重尾噪声下的有限时间高概率收敛保证。

**🔧 技术方法**

使用了剪切的AdaGrad算法，结合了历史度量和当前梯度的同步更新，采用了确定性回溯论证来分析算法的行为。

**📊 数据集**

使用了具有界限条件二阶矩的随机梯度，构造了一个凸随机优化问题来展示AdaGrad的失效。

**📈 对比分析**

与现有方法比较，剪切的AdaGrad在重尾噪声下的复杂度为𝒪(ε^-2)，与信息论下界相匹配，且在高概率下提供了收敛保证。

**⚠️ 局限性**

限制在于未剪切的AdaGrad在重尾噪声下可能导致的几何失调，且现有的高概率收敛结果通常假设噪声是次高斯的或有界的。

---

## 42. Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops

**arXiv ID:** 2609.30297 | [PDF](https://arxiv.org/pdf/2609.30297v1)

**作者:** Enrico Palumbo `[一作]` (Spotify), Christine Doig Cardet `[通讯]` (Spotify)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5119556689)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建多轮对话合成数据管线和自我改进循环，帮助Spotify在冷启动场景下快速部署会话式推荐代理。

**💡 创新点**

创新点在于将单轮意图扩展为多轮对话并利用差异化对比优化与方差根因分析自动修复代理规划与工具使用错误。

**🔧 技术方法**

主要技术包括LLM生成合成对话、LLM-as-judge评估、编码代理自动改写提示与工具定义，以及差异化对比优化与迭代细化。

**📊 数据集**

使用了Spotify内部单轮用户请求数据作为种子，并生成合成多轮对话，同时利用人工标注与LLM评测验证质量。

**📈 对比分析**

与手工优化提示相比，循环提升了8%质量；线上A/B测试显示相较先前的会话细化体验，用户收听量提升14%，活跃用户增长5%，跳过率下降5%。

**⚠️ 局限性**

局限性包括合成对话可能无法完全覆盖真实用户行为、长对话性能随回合数增加下降，以及自我改进仍需人工复核。

---

## 43. The Shape of Events: Edge-Based Inductive Biases via Cross-Domain Distillation

**arXiv ID:** 2609.30478 | [PDF](https://arxiv.org/pdf/2609.30478v1)

**作者:** Soshun Kihara `[一作]` (Independent Researcher), Masato Taki `[通讯]` (Rikkyo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用事件相机数据与标准RGB图像的跨域知识蒸馏，构建事件蒸馏模型（ED_Student）并在ImageNet上训练。

**💡 创新点**

首次证明事件相机捕获的亮度梯度信息能在RGB领域引入“结构先验”：抑制高频纹理依赖、强化边缘形状特征，从而获得色彩不变性、形状偏好和对高频噪声的鲁棒性。

**🔧 技术方法**

技术手段包括：事件数据（Neuromorphic‑ImageNet）作为教师模型（DiST ResNet‑34）；交叉模态知识蒸馏（KL + CE 损失）；对比实验、频谱分析、形状冲突、图像‑C 数据集评估、对抗攻击等多维度诊断。

**📊 数据集**

主要使用的数据集为：N‑ImageNet（事件版 ImageNet）作为教师输入，ImageNet‑val 作为评估基准；额外使用 ImageNet‑C、色彩旋转、灰度、Canny、形状冲突图像以及 19 个下游分类任务进行线性探针转移。

**📈 对比分析**

与基线 ResNet‑34（73.9%）对比，ED_Student 在色彩不变性（Hue rotation worst 63.4% vs 56.7%）和形状偏好（形状冲突图像精度提升）以及对 ϵ≤1/255 的 PGD/FGSM 攻击的鲁棒性（平均提升约 2‑4%）显著优于基线；对抗训练和噪声训练在大部分下游任务表现下降，而事件蒸馏在 19 个线性探针任务中平均提升 1.6%，并在 CIFAR、医学影像、sketch 等任务上获得最大收益。

**⚠️ 局限性**

局限性包括：N‑ImageNet 仅通过在监视器上重拍 RGB 图像生成，未充分利用事件相机的高动态范围与运动视差；大模型（ConvNeXt‑Base、RepLKNet31B）对蒸馏效果减弱；仅在 CNN 图像分类任务上验证，未扩展至 Transformer、密集预测或非视觉任务。

---

## 44. TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Video Understanding

**arXiv ID:** 2609.30670 | [PDF](https://arxiv.org/pdf/2609.30670v1)

**作者:** Yibo Ma `[一作]` (Om Ai Research), Tiancheng Zhao `[通讯]` (Om Ai Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了TRACE——一种面向流式视频理解的条件感知评估基准，结合时序审计、核心-适配器协议和多维度报告；

**💡 创新点**

创新点在于将时间有效性、执行条件和操作行为显式化，并通过统一的核心-适配器接口记录实际历史处理与响应事件，提供比单一分数更丰富的评估视角；

**🔧 技术方法**

使用了时序化的核心-适配器框架、因果视频输入、文本解析器、主动响应评分机制，以及多项指标（准确率、延迟、工作量、误报/漏报等）；

**📊 数据集**

基于StreamingBench与OVO-Bench的视觉任务，构建了1,240条记录（517段视频），包括问答与主动响应两类任务，采用1 FPS采样；

**📈 对比分析**

对八个公开模型或系统在八种配置下进行实验，结果显示相同的QA准确率可能对应不同的完成率、延迟、输出量和无效输出；主动响应中，准确率、延迟、误报率与漏报率可独立变化；示例：LiveCC与MOSS-Preview准确率≈65%但完成率、工作量和无效输出差异显著；

**⚠️ 局限性**

局限性包括仅处理视觉单指令任务、1 FPS采样、未覆盖多轮交互、负事件或无触发情况、仅评估正触发误报，以及部分模型在文本协议下的输出不可解析等问题。

---

## 45. Predicting Transmembrane Protein Topology from 3D Structure

**arXiv ID:** 2609.30446 | [PDF](https://arxiv.org/pdf/2609.30446v1)

**作者:** Sitong Chen `[一作]` (Technical University of Denmark), Xiaopeng Mao `[通讯]` (Technical University of Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

使用基于三维蛋白结构的图神经网络对蛋白质拓扑进行预测

**💡 创新点**

提出了“major voting”方法，将原子级 GNN 预测聚合到残基级，实现残基级拓扑标签推断

**🔧 技术方法**

采用 SchNet（以及尝试的 EGNN、GCPNet）、Gaussian 平滑、全连接层、Adam 及学习率衰减等深度学习技术

**📊 数据集**

使用与 DeepTMHMM 相同的蛋白序列数据集，并结合 AlphaFold 预测的 3D 结构

**📈 对比分析**

通过 5 折交叉验证与基线（多数类别）比较，并使用 McNemar 检验验证显著性；残基级准确率提升约 6–8%（从 68%→75%），但拓扑正确率仅在球状蛋白上达到 3.6%，远低于 DeepTMHMM（≈80–95%）

**⚠️ 局限性**

限制包括缺乏预训练权重、数据不平衡导致对 α‑TM/β‑barrel 的预测效果差、AlphaFold 结构不确定性以及 GNN 内存/计算瓶颈

---

## 46. Subjects, Not Authors: The Authorship Hazard in Agentic Dataspaces

**arXiv ID:** 2609.30614 | [PDF](https://arxiv.org/pdf/2609.30614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 47. All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation

**arXiv ID:** 2609.30416 | [PDF](https://arxiv.org/pdf/2609.30416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 48. Cosine Similarity Is Not Evidence: Measuring the Noise Floor of Interpretability Transfer Under Quantization

**arXiv ID:** 2609.30275 | [PDF](https://arxiv.org/pdf/2609.30275v1)

**作者:** Pranav Varshney `[一作]` `[通讯]` (University of Michigan), Pranav Varshney (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文探讨在模型量化过程中可解释性评估的测量有效性，提出闭式噪声底并测量真实激活中的类间分离度，指出缺乏关键统计量会导致误读量化后相似性指标，并给出更可靠的比较方法和实践建议。

**💡 创新点**

创新点包括：1）推导并验证了差分均值方向相似度的闭式噪声底（$nho^2/d$）；2）首次在真实激活上测量并公开类间分离度$ho$；3）提出在同一量化模型内测得的分半（split‑half）零分布作为正确的对照；4）阐明规模不变统计量无法区分翻译与衰减失效，并给出区分方案。

**🔧 技术方法**

采用差分均值估计、Monte Carlo 模拟、闭式期望推导、分半零分布计算、与 AUROC/决策阈值回归等技术，结合量化（INT4/INT8）与全精度（FP16）权重对比实验。

**📊 数据集**

主要数据集为 Qwen2.5‑1.5B‑Instruct，使用 AdvBench 有害提示与 Alpaca 无害指令，样本数 $n=256$，隐藏层维度 $d=1536$；此外在讨论中引用了 Llama‑2‑7B（$d=4096$）的公开结果。

**📈 对比分析**

比较方法为将量化模型的相似度（如余弦）与其自身分半零分布（和全精度零分布）对比，并采用置信界判定是否存在旋转；实验表明 INT8 的相似度几乎无变化，INT4 的相似度低于噪声底，说明方向确实发生旋转；同时 AUROC 对于纯翻译失效保持不变，对衰减敏感。

**⚠️ 局限性**

局限性包括：仅在单一模型与单一隐藏维度上测得 $ho$，对其它模型的推广尚未验证；量化采用仿真和权重量化，未覆盖激活量化；对真实拒绝率的实测缺失；外推至更大隐藏尺寸时需假设 $ho$ 变化，存在不确定性。

---

## 49. A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods

**arXiv ID:** 2609.30397 | [PDF](https://arxiv.org/pdf/2609.30397v1)

**作者:** Miquel Miró-Nicolau `[一作]` (Universitat de les Illes Balears), Riccardo Guidotti `[通讯]` (Uiniversity of Pisa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于干预的合成真值（Synthetic Artificial Intelligence Ground Truth, SAIG）框架，用来在二值图像、时间序列和表格数据三种不同数据模态下对可解释人工智能（XAI）方法进行客观评估。

**💡 创新点**

创新点在于：①首次将干预式 SAIG 扩展到非图像模态（时间序列和表格）；②通过离散化和可控干预生成本地真值解释；③提出统一的评估流程和指标，消除了传统“可信度”评估缺乏真值参考的问题。

**🔧 技术方法**

使用技术包括：①干预式生成合成数据（如移除图像部件、减少时间序列片段、降级表格特征）；②多种 XAI 方法（RISE、LIME、Gradient、DeepLIFT、KernelSHAP、Integrated Gradients、T‑SHAP、LORE）；③评估指标 SIM、KL Divergence、AUC‑ROC 等；④预训练模型（ResNet50、MultiRocket、MLP）作为被解释对象。

**📊 数据集**

使用的数据集：①从 FunnyBirds 合成的二值图像（47,470 张）；②从相同图像生成的时间序列（1251 采样点）；③基于离散化输入与 sin‑weight 函数生成的表格数据（≈4M 样本，3 个特征）。

**📈 对比分析**

通过将每种 XAI 输出与干预得到的真值解释做比对，计算 SIM、KL 和 AUC。实验结果显示：在图像任务中，梯度类方法（Gradient、DeepLIFT、Integrated Gradients）和 LIME/KernalSHAP 在 SIM 上表现最佳，但整体仍远低于理论上限；在时间序列任务中，RISE 与 LIME 在 SIM 上最优，KernelSHAP 低效；在表格任务中，Gradient 与 LORE 接近最优，LIME 低效。总体来看，现有 XAI 方法在不同模态下的解释质量差异明显，没有一种方法能在所有任务上都表现最优。

**⚠️ 局限性**

局限性包括：①评估仅在合成数据上进行，可能无法完全反映真实数据的复杂性；②干预过程虽然可控但仍依赖于人工设计的模型与生成规则；③评估指标主要针对局部重要性，未涵盖更丰富的解释属性；④对某些方法（如 KernelSHAP）因背景样本设置不当导致性能偏低，提示评估过程需更细致。

---

## 50. Do LLMs Understand Context? A Knowledge Graph-Based Evaluation Framework

**arXiv ID:** 2609.30484 | [PDF](https://arxiv.org/pdf/2609.30484v1)

**作者:** Subavarshana Arumugam `[一作]` (University of Moratuwa), Uthayasanker Thayasivam `[通讯]` (University of Miami)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出基于知识图谱的LLM上下文理解评估框架，使用S3KG对生成回答、黄金答案和上下文三者的KG进行比较，形成两维度的Contextual Understanding Score（CUS），并通过Triplet Analyzing Unit（TAU）进行错误诊断。

**💡 创新点**

创新点在于：①将结构化图谱与语义嵌入结合的混合相似度S3KG；②软标签对齐解决语义相同但文本不同的实体/关系；③双维度评估GoldSim与CtxSim，利用调和平均量化模型在事实与上下文的平衡；④对低分案例的细粒度三元组错误分类。

**🔧 技术方法**

技术包括：知识图谱抽取（统一提示+后处理归一化），SBERT语义嵌入，Weisfeiler–Lehman图核，混合系数α混合结构与语义相似度，硬/软标签对齐，Triplet‑级余弦相似度分类。

**📊 数据集**

使用PubMedQA（医学问答，长答案）和MesaQA（消费者医疗问答）两大QA数据集；在9个文本相似度基准（短文本、KG扰动段落、实体交换对照）上进行S3KG与基线对比。

**📈 对比分析**

与ROUGE、BLEU、BERTScore、MiniLM、sentence‑T5‑base等传统指标对比，S3KG在6/9基准上最高，F1提升可达+7.6点，AUROC最高达0.973；在QA评估中，Mistral‑7B在两大数据集上获得最高CUS。

**⚠️ 局限性**

局限包括：对KG抽取质量高度依赖，噪声或缺失会显著降低S3KG；SBERT推理在大规模评估时计算成本较高；目前仅验证7B开源模型，未涉及更大或专有模型；软标签对齐不处理方向性语义等细微差异。

---

## 51. NavGen: Visual Generative Models as a Scalable Data Engine for Embodied 3D Navigation

**arXiv ID:** 2609.30770 | [PDF](https://arxiv.org/pdf/2609.30770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 52. Settling the Matroid Secretary Problem

**arXiv ID:** 2609.30421 | [PDF](https://arxiv.org/pdf/2609.30421v1)

**作者:** Zhiyi Huang `[一作]` (University of Hong Kong), Zhiyi Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 2055 | [OpenAlex ID](https://openalex.org/A5090297023)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文解决了Matroid秘书问题，提出了一种具有-概率竞争性的算法。

**💡 创新点**

创新点在于该算法是序数的，并且仅通过比较和独立性oracle访问到到达的元素，具有期望的多项式时间和oracle复杂度。

**🔧 技术方法**

使用了序数算法和生存过程的辅助设计与分析。

**📊 数据集**

使用了未知matroid模型下的元素集合，元素在[0, 1]区间内独立且均匀到达。

**📈 对比分析**

与之前的算法相比，本文的算法在竞争性上达到了最优的-概率竞争比，并且在期望的多项式时间和oracle复杂度下实现。

**⚠️ 局限性**

限制在于算法的接受概率依赖于生存过程的比率，而该比率的有效计算仍然是一个挑战。

---

## 53. Offline Policy Evaluation as a decision support tool for designing Adaptive Experiments

**arXiv ID:** 2609.30273 | [PDF](https://arxiv.org/pdf/2609.30273v1)

**作者:** João Victor Ferreira Alves `[一作]` (Instituto de Ciência e Tecnologia Itaú), Thiago Costa Rizuti da Rocha `[通讯]` (Instituto de Ciência e Tecnologia Itaú)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用历史A/B测试日志，对预设的上下文多臂带宽（CMAB）策略进行离线评估与选择，并提供可行的热启动模拟验证其上线效果。

**💡 创新点**

创新点在于提出一套两层决策支持框架：先用离线策略评估（OPE）从固定实验数据筛选出有潜力的自适应策略，再在与日志数据匹配的模拟器中热启动验证，从而在上线前判断是否值得部署自适应实验及选定哪种策略。

**🔧 技术方法**

核心技术包括离线策略评估中的双重稳健（DR）估计、样本拆分与Bootstrap置信区间、以及多种CMAB算法（Bootstrapped Thompson Sampling、Bootstrapped UCB、Softmax/Boltzmann、ε‑Greedy）和对应的模拟器实现。

**📊 数据集**

使用的实验数据包括：四种不同异质性结构的合成RCT数据（无上下文、类别异质性、线性异质性、周期性异质性及双变量组合），以及公开实验数据集Hillstrom、Criteo Uplift和LaLonde。

**📈 对比分析**

通过OPE估计的策略价值和在线累计回报/回报差对各策略进行比较；实验表明：在存在显著上下文异质性的情形下，自适应策略显著优于固定分配；在无异质性或异质性弱的场景中差距微乎其微；在公开数据集上，Criteo表现出强烈的自适应优势，Hillstrom和LaLonde则未能显著区分。

**⚠️ 局限性**

研究局限在于未进行离线策略学习，仅评估预设策略；热启动模拟为最优匹配环境，未考察非平稳、数据偏差或实际部署约束；仅关注单步带宽决策，未扩展到长期强化学习情形。

---

## 54. Actively Resolving Contextual Uncertainty for Underspecified Tasks in Natural Language

**arXiv ID:** 2609.30428 | [PDF](https://arxiv.org/pdf/2609.30428v1)

**作者:** Zachary Ravichandran `[一作]` (University of Pennsylvania), Vijay Kumar `[通讯]` (University of Pennsylvania)

**通讯引用:** 42844 | [OpenAlex ID](https://openalex.org/A5087021192)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CLUE框架，利用LLM在闭环中生成假设、更新任务相关概念，并通过语言嵌入的稠密体素地图实现假设落地，进一步通过主动感知（导航、检查、操作）验证假设并完成欠定自然语言任务。

**💡 创新点**

创新点包括①使用LLM生成可更新的任务假设并与语义嵌入地图相结合；②引入闭环反馈，使机器人在实时感知中不断修正假设；③设计基于地图查询与聚类的候选生成和TSP路径提示，提升信息采集效率；④在开放词汇环境下以符号假设替代完整贝叶斯推断，实现可扩展的推理。

**🔧 技术方法**

技术栈包括：GPT‑5.1 LLM（通过系统提示和上下文生成假设及行动序列）；VLM（对检查行为进行文本查询）；稠密语义体素地图（RGB‑D深度感知 + feature embedding）；DBSCAN聚类 + 余弦相似度匹配候选位置；TSP 近似求解生成最短访问顺序；Nav2 轨迹规划；Spot机器人底盘与抓取/放置控制。

**📊 数据集**

实验数据集为真实环境（两个室内、一个室外）共15个任务（物体辨别、功能推理、遮挡推理），每个任务在Boston Dynamics Spot上完成；地图在线构建于RGB‑D传感器数据。

**📈 对比分析**

对比方法：oracle（具备逐步指令）、NLMaps（开放式地图查询、无闭环反馈）、DAAAM（场景图+VLM标注）。CLUE在任务成功率上达到86.7%，仅比oracle低7%；相较NLMaps的20%和DAAAM的22%，CLUE提升约4×和3×；资源使用方面，CLUE耗时、动作、LLM/VLM调用量比oracle略高，但在成功率上明显优于对比基线。

**⚠️ 局限性**

局限性：稠密语言嵌入地图占用内存巨大（室外场景≈100GB），依赖云端GPT‑5.1模型，未针对动态环境或实时计算进行优化；LLM路径规划表现受限，需进一步设计同时兼顾语义推理与路径效率的生成结构；在更开放或大规模场景中，地图构建与更新的效率与精度尚需提升。

---

## 55. Cost-Aware Best-LLM Identification using Dueling Feedback

**arXiv ID:** 2609.30360 | [PDF](https://arxiv.org/pdf/2609.30360v1)

**作者:** Sarvesh Gharat `[一作]` (Indian Institute of Technology Bombay), Jayakrishnan Nair `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 792 | [OpenAlex ID](https://openalex.org/A5037150671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个考虑查询成本异质性的双打多臂老虎机（dueling MAB）模型，用于在一组大型语言模型中寻找具有Condorcet赢家的最佳模型。

**💡 创新点**

创新点在于：① 在存在Condorcet赢家假设的前提下，首次为双打MAB引入异质采样成本；② 通过信息理论下界得到闭式表达；③ 设计了成本感知的Track‑and‑Stop算法，并证明其在错误概率趋于0时达到成本上界的渐近最优。

**🔧 技术方法**

使用的技术包括：概率统计（KL散度、GLR检验）、对策优化（权重分配）、跟踪采样策略（WeightPullAllocation）以及Chernoff式停止规则。

**📊 数据集**

使用了Chatbot Arena公开的多任务 LLM 比较数据集（文本到图像、文本到文本、视觉、搜索），并通过合成三臂实例进行验证。

**📈 对比分析**

与四种基线（TAS、CRR、DPCA、DCTAC）对比，DCTAS在所有真实数据集上平均成本下降约3–6%，并在合成实验中始终取得最低成本；DCTAC因停止规则更早而略优于DCTAS。

**⚠️ 局限性**

局限性包括：只在存在Condorcet赢家的情形下证明最优性；对复杂成本结构的泛化尚未充分；并且在极端成本差异或噪声高的场景下，算法可能需要更长时间收敛。

---

## 56. Don't CLAP: Are Music-Text Models Bag-of-Words?

**arXiv ID:** 2609.30540 | [PDF](https://arxiv.org/pdf/2609.30540v1)

**作者:** Yuan-Chiao Cheng `[一作]` (Georgia Institute of Technology), Alexander Lerch `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估文本到音乐系统的CLAP得分在属性交换（timbre、lead/伴奏、先后顺序）时的鲁棒性，并提出 Music Attribute‑Swap Benchmark (MASB) 和 MASB‑Order 两个对比基准；

**💡 创新点**

①创建了可对比属性交换的基准数据集；②通过属性交换测试 CLAP 是否真正捕捉音乐属性；③发现大型音频‑语言模型的优势主要源自文本先验而非音频信息；

**🔧 技术方法**

对四种对比音乐‑文本模型（LAION‑CLAP、MS‑CLAP、MuQ‑MuLan、CLaMP 3）和一个大型音频‑语言模型（Qwen2‑Audio‑7B‑Instruct）进行 CLAP 相似度、文本嵌入距离、Cohen κ 与 prior‑balanced accuracy 等多项指标评估；

**📊 数据集**

使用 Song Describer Dataset、MTG‑Jamendo 10 s 片段构成 400 条音频及其属性交换字幕；并从 MoisesDB stems 构造 300 条对称样本（MASB‑Order）进行先后顺序对比；

**📈 对比分析**

通过比较原始字幕与交换字幕的 CLAP 分数来判定模型是否能区分属性；结果表明四个对比模型的准确率均≈50%（无统计学意义），而 Qwen2‑Audio 在所有属性上平均达 68% 甚至 72%；使用 Cohen κ 发现对比模型几乎不受文本先验影响，而大型模型则与文本先验高度一致；

**⚠️ 局限性**

局限性：CLAP 和对比模型对文本属性绑定缺乏敏感度，表现为词袋特性；大型模型的优势主要来自文本先验，未真正从音频中获取细节；数据集间的分布差异（真实录音 vs. 处理后的 stems）可能导致迁移性能不稳定。

---

## 57. AlphaEarth distinguishes cities but compresses urban variation

**arXiv ID:** 2609.30356 | [PDF](https://arxiv.org/pdf/2609.30356v1)

**作者:** Andrew Renninger `[一作]` `[通讯]` (University of Glasgow), Andrew Renninger (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对AlphaEarth地球嵌入向量在1000个功能性城市的采样与分析，构建并验证了一个跨城市、跨时空的可比城市特征空间。

**💡 创新点**

创新点在于提出一种基于球面几何的城市嵌入评估协议，系统量化了嵌入在全球范围内对城市多样性、空间结构和时间演变的捕捉能力，并揭示了嵌入容量分布不均、城市间相似度受大洲与气候影响、年际变化主要由观测与采样误差驱动的事实。

**🔧 技术方法**

主要技术包括：使用预训练的多模态视觉编码器（AlphaEarth v2.1）生成10米分辨率的64维单位向量；球面距离与切平面投影用于度量城市均值与方差；多元回归、聚类、PCA、Bootstrap及自举检验用于统计推断。

**📊 数据集**

数据集涵盖：AlphaEarth 2024年全球层（64维向量）；全球人类居住层（GHSL）提供2020年四级城市化分类；Functional Urban Areas（FUA）提供1000个大城市人口与区域信息；人类发展指数（HDI）与气候、地形、植被、雷达观测等多源辅助变量。

**📈 对比分析**

比较方法：对每个城市计算均值向量和内部方差，并与全球参考进行角距离比较；利用同质匹配和残差分析衡量空间结构；通过多元回归检验HDI对嵌入方差的影响，并在不同年度层面评估城市路径与变化速度。结果显示：城市均值在空间上形成连续但被大陆与气候结构化的分布；内部方差与HDI呈显著正相关；年际路径主要为随机波动，净位移远低于累计路径。

**⚠️ 局限性**

局限性包括：样本仅覆盖人口≥25万的城市，未考虑面积或人口加权；使用的四级城市化标签为粗尺度且与训练数据不一致；嵌入的方差受观测缺失与采样噪声影响，未能完整校正时序可靠性；缺乏对嵌入与城市功能真实对应关系的验证。

---

## 58. Aerial Manipulation in the Wild with Onboard Perception, Policy Learning, and Whole-Body Control

**arXiv ID:** 2609.30521 | [PDF](https://arxiv.org/pdf/2609.30521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 59. A Benchmarking Framework for Context-aware XR Interfaces

**arXiv ID:** 2609.30466 | [PDF](https://arxiv.org/pdf/2609.30466v1)

**作者:** Hyunsung Cho `[一作]` (Carnegie Mellon University), David Lindlbauer `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1656 | [OpenAlex ID](https://openalex.org/A5058551017)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个用于评估XR上下文感知功能建议的基准框架，包括功能facet表示、扩展版MineXR++数据集、三类基准任务（上下文因子分析、初始建议、后续建议）以及基于模拟交互成本的评估协议。

**💡 创新点**

创新点：1）将应用功能拆分为语义一致的功能facet，兼顾粒度与可操作性；2）在MineXR上增添facet级注释，生成MineXR++；3）设计了三种真实情景下的基准任务；4）引入导航与搜索成本模型，评估建议的用户实际负担，而非仅仅匹配度。

**🔧 技术方法**

技术手段：图结构的facet图与关系检索、全局流行度基线、Gemini 3 Flash实现的零/少 shot LLM推理、模拟交互成本计算（打开菜单、扫描、导航等固定代价）。

**📊 数据集**

数据集：MineXR++（基于原始MineXR），包含109个XR布局、695个widget、42个应用、273个功能facet、1,007个能力，以及用户、任务、环境三元组标签。

**📈 对比分析**

方法对比：对比手动访问、全局流行度、关系检索、零shot LLM、少shot LLM，使用Cost、Recall@10、F1@10、Hit@10评估。结果显示少shot LLM在初始建议中降低约22%成本，后续建议中降低约31%成本，关系检索在后续建议中也表现突出，其余方法提升有限。

**⚠️ 局限性**

局限性：假设上下文已完美给定，未考虑感知、认知负荷和长期真实使用；模拟交互成本模型简化为固定步长，未覆盖真实XR交互的多模态成本；数据规模受原始MineXR的限制，缺乏大规模真实使用数据。

---

## 60. From Visual Search to Movement Control: A Priority Field for Artificial Agents

**arXiv ID:** 2609.30704 | [PDF](https://arxiv.org/pdf/2609.30704v1)

**作者:** Han Zhang `[一作]` (University of Michigan), Zhong Cao `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个轻量级的优先图模型，用以预测人类在视觉搜索任务中的首次注视；随后将该模型扩展为优先场模型，应用于控制人工智能代理在reach–avoid（到达目标并躲避移动障碍物）任务中的运动；

**💡 创新点**

首次将认知层面的“优先计算”从视觉注意迁移到空间运动控制，提出将视觉搜索的优先图映射到运动方向的优先场，并通过简单的记忆机制产生人类式的先验行为，证明了该结构在运动学习与迁移中的高效性与可解释性；

**🔧 技术方法**

使用软最大化的优先图/优先场加权求和；轻量化的两层MLP实现感知映射和规划；行为克隆（最小二乘损失）训练；动态感知心理物理函数（sigmoid距离转化）；泄漏累加器实现历史记忆；与无显式优先计算的MLP、Transformer进行对比；

**📊 数据集**

公开的人类首次注视数据（114,232 条，333 名受试者，11 组实验）用于训练和验证优先图模型；人工专家演示数据（12,000 条状态-动作对，20 轮）用于训练优先场代理；

**📈 对比分析**

通过训练损失、目标达成率/分钟、碰撞率/分钟等指标与无优先计算的 MLP、Transformer 在同一输入下进行对比；结果显示优先场代理在训练阶段收敛更快，目标达成率显著高于对照模型，碰撞率显著低于 MLP，并在障碍密度提升、障碍速度加快以及统计学习测试中均保持更好的性能；

**⚠️ 局限性**

模型极度简化，仅适用于二维平面、仅感知距离、障碍物形状单一；记忆机制仅记录目标位置，未考虑奖励或其他学习机制；优先图模型未区分目标特征提升与干扰特征抑制，可能无法推广到更复杂的视觉搜索任务；未在真实机器人或更丰富的感知环境中验证。

---

## 61. Can a Robot Read Braille? - Learning to Adapt Contact via Imitation Learning for Tactile Braille Recognition

**arXiv ID:** 2609.30676 | [PDF](https://arxiv.org/pdf/2609.30676v1)

**作者:** Xi Chen `[一作]` (Shenzhen University), Peng Zhou `[通讯]` (Great Bay University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过学习专家演示，机器人能够评估触摸接触质量并主动纠正接触姿态，以获得可读的点字信息。

**💡 创新点**

提出了基于图像的自适应接触框架和多头策略学习，联合预测接触可接受度和姿态校正，实现闭环接触优化。

**🔧 技术方法**

使用图像编码器（ResNet‑18）+多头网络、SL1损失、tactile 视觉传感器（Xense G1‑WS）、机器人抓取控制、基于姿态融合的重建与语义解码。

**📊 数据集**

20块中英文 Braille 纸板（10块用于训练，10块用于在线评估）以及 xArm7 机器人搭载的触觉传感器。

**📈 对比分析**

与单次触摸、仅重试、仅位置校正等基线比较，Full 策略在保持平均 1.42 次尝试/位置的情况下，触觉质量提升至 94%，重建准确率 88.6%，行读取准确率 80%，显著优于其他方法。

**⚠️ 局限性**

仅在平面 PLA 板和固定传感器安装下验证，最多三次尝试；未测试非平面、柔性材料、耦合扰动或不同传感器配置。

---

## 62. MuseTimbre: Zero-Shot Timbre Transfer by Controlling a Frozen Music Generator

**arXiv ID:** 2609.30548 | [PDF](https://arxiv.org/pdf/2609.30548v1)

**作者:** Yuan-Chiao Cheng `[一作]` (Georgia Institute of Technology), Zhiyao Duan `[通讯]` (University of Rochester)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种基于预训练音乐生成器（Stable Audio 3）的零样本乐器音色迁移系统MuseTimbre，能够将音频参考的音色迁移到多音源录音的演奏中，同时保持原始音高信息。

**💡 创新点**

创新点包括：①首次通过冻结预训练生成器并添加两条独立控制通道（音高跨注意力与音色AdaLN）实现音色迁移；②对CLAP音频编码器进行端到端微调，使其成为对音高不敏感、可用于音色相似度衡量的嵌入；③通过多音高估计器（Basic Pitch）直接从源音频提取音高，实现多音源的准确对齐。

**🔧 技术方法**

技术手段包括：预训练的Diffusion Transformer（Stable Audio 3）作为生成器；1-D CNN编码器对钢琴卷（piano roll）音高信息进行特征提取；AdaLN对全局音色嵌入进行调制；CLAP音频-文本双分支编码器用于音色提取；rectified-flow 损失训练；跨注意力与AdaLN在Transformer层中嵌入。

**📊 数据集**

使用了 50% 合成（Slakh MIDI + General MIDI 音色）与 50% 真实（MoisesDB、URMP）音频混合的数据进行训练；评估时选用四个未参与训练的公开数据集：PHENICX-Anechoic、MAESTRO v3 test、GOAT、Bach10，共 13 种乐器、304 对源-参考组合。

**📈 对比分析**

与七种基线（文本条件 DDIM/​DDPM 逆向、MuseControlLite、TokenSynth‑T、CTD、TokenSynth‑A、SS‑VQ‑VAE）比较。MuseTimbre 在音色匹配（PaSST）上达 57.6% ；在音高保持（frame‑F1）上为 0.316，优于 CTD（0.48%）与 SS‑VQ‑VAE（0.45%），但略低于 TokenSynth‑A（0.38）。主观听评中，MuseTimbre 在音色评分上最高（2.6），接近 TokenSynth‑A；在音高评分上略逊于 TokenSynth‑A（3.4）。

**⚠️ 局限性**

局限性包括：①对音色的全局控制不支持细粒度的时变音色变化；②对非常大音高跨度或复杂和声的音高估计仍可能产生误差；③微调 CLAP 仍受训练数据分布限制，跨域真实录音时音色相似度测量的泛化性待验证。

---

## 63. CLEAR: Online Speech Content Leakage Estimation through Cross-ASR Disagreement

**arXiv ID:** 2609.30415 | [PDF](https://arxiv.org/pdf/2609.30415v1)

**作者:** Bhawana Chhaglani `[一作]` (University of Massachusetts Amherst), Prashant Shenoy `[通讯]` (University of Massachusetts Amherst)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于多模ASR不一致性的无参考语音内容泄露估计方法CLEAR。

**💡 创新点**

利用不同ASR模型输出的分歧来评估隐私泄露并实时识别可能泄露的词语，提供动态隐私调节反馈。

**🔧 技术方法**

采用多种最先进ASR模型、序列相似度计算、校准映射与模型子集选择，实现低延迟且高相关性的估计。

**📊 数据集**

在Mozilla Common Voice数据集上使用Kirigami和低通滤波等隐私变换进行评估。

**📈 对比分析**

与持有参考转录的PER对比，Spearman相关性分别为0.8（单样本）和0.98（平均），校准后MAE为0.162，三模组实现RTF≈0.15秒，显著优于单模型置信度。

**⚠️ 局限性**

受限于ASR多样性和对极端压缩的适应性未知，且仅评估内容泄露，未涵盖说话人身份等隐私维度。

---

## 64. LUMO (Lightweight Unified Multilingual Orchestrator): A Privacy Preserving Offline Voice Assistant

**arXiv ID:** 2609.30692 | [PDF](https://arxiv.org/pdf/2609.30692v1)

**作者:** Md. Mehedi Hasan Naeem `[一作]` (Jatiya Kabi Kazi Nazrul Islam University), Md. Sujan Ali `[通讯]` (Jatiya Kabi Kazi Nazrul Islam University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Raspberry Pi 5上实现了一个完全离线、可多语种（英、孟）对话的语音助手LUMO，集成了VOSK ASR、4‑bit GGUF TinyLLaMA生成式语言模型和Piper TTS。

**💡 创新点**

创新点在于将离线ASR、压缩的生成式LLM与TTS融合为一个低功耗、低延迟的端到端管道，并通过4‑bit GGUF量化显著降低模型占用内存，满足边缘设备资源限制。

**🔧 技术方法**

技术包括WebRTC VAD、VOSK基于Kaldi的离线ASR、TinyLLaMA 1.1B量化模型、Piper TTS、Python多线程管线以及USB SSD加速和主动冷却。

**📊 数据集**

使用自制的1,000条双语命令式语料（500英，500孟）记录于USB麦克风，包含人工标注转录，用于评估识别、推理和合成性能。

**📈 对比分析**

与Rhasspy和Mycroft对比，LUMO在离线环境下实现了3.2 s的总延迟、9 W的峰值功耗，低于对手（约5 s/12 W、3.5 s/11 W），且完全零网络流量，显示出更优的低功耗低延迟优势。

**⚠️ 局限性**

局限性包括在高噪声环境下WER显著升高、孟加拉语识别误差较高、生成模型推理能力有限、缺乏长对话记忆和IoT上下文感知功能。

---

## 65. VLALight: Lightweight Vision-Language-Action Models for Emergency-Aware Traffic Signal Control

**arXiv ID:** 2609.30709 | [PDF](https://arxiv.org/pdf/2609.30709v1)

**作者:** Kemou Jiang `[一作]` (Beihang University), Zhiyong Cui `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级端到端视觉-语言-动作框架VLALight，用单一前向传播实现交叉路口交通信号控制，尤其在紧急车辆服务上表现优异。

**💡 创新点**

创新点在于将多方向摄像头视图拼接为统一视觉输入，并用文本说明构建跨视角对应关系；采用VLA范式直接预测阶段，消除图像-文本转换的细节损失和多阶段推理延迟。

**🔧 技术方法**

技术包括：大规模预训练的VLM（DINOv2+SigLIP+ViT/14），LoRA适配器，phase查询，端到端行为克隆训练，最小绿灯停留约束。

**📊 数据集**

数据集基于TransimHub仿真，复现六个真实交叉口的路网拓扑、流量模式与紧急车辆，使用Blender高保真渲染生成多视角图像，并人工验证专家标注。

**📈 对比分析**

与传统固定、规则、强化学习方法以及云端VLM系统（VLMLight 32B/3B）对比，在六个城市的常规流和多种流量模式下，VLALight在平均行驶时间、等待时间及紧急车辆行驶/等待时间上均达标最高，特别是平均紧急等待时间下降21.1%，推理时延仅224 ms。

**⚠️ 局限性**

局限性：训练仅基于行为克隆，缺乏强化学习微调；仅使用单帧输入，未利用时序记忆；实验仅在仿真环境中，未进行真实世界验证与安全性证明。

---

## 66. Input-Layer Starvation: Why Per-Layer Pruning Breaks IoT Intrusion Detectors

**arXiv ID:** 2609.30729 | [PDF](https://arxiv.org/pdf/2609.30729v1)

**作者:** Md Anas Biswas `[一作]` `[通讯]` (University of Portsmouth), Md Anas Biswas (University of Portsmouth)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在小型物联网入侵检测器上使用层级统一稀疏化导致的类别级失效，提出了通过保护第一层或全局阈值裁剪及重新校准归一化统计的低成本修复方案。

**💡 创新点**

首次揭示统一层级裁剪会导致第一层饥饿，引起批归一化统计偏移，导致模型在部署时出现严重的误归因和高误报；并给出简单的防护与修复方法。

**🔧 技术方法**

使用基于L1幅度的一次性与渐进式稀疏化、批归一化统计重估、手工构造掩码和实验性验证。

**📊 数据集**

主要使用CICIoT2023数据集（3.7M流记录，34类）和TON_IoT数据集（93.6K记录，10类）进行实验。

**📈 对比分析**

通过与未裁剪基线及其他裁剪策略（保护第一层、全局裁剪）对比，发现统一层级裁剪在宏观准确率提升不足的情况下，宏F1降半，误报率翻倍，修复后宏F1恢复至0.53，误报率降至≈30%。

**⚠️ 局限性**

局限在于只针对单输入通道的卷积网络、特定裁剪工具实现，未验证结构化裁剪、量化感知训练等方法，对更大或不同架构的推广尚待验证。

---

## 67. Pretrained ASR Pseudo-labeling for Noisy Police Audio

**arXiv ID:** 2609.30469 | [PDF](https://arxiv.org/pdf/2609.30469v1)

**作者:** Kaavya Chaparala `[一作]` (Johns Hopkins University), Anjalie Field `[通讯]` (Johns Hopkins University)

**通讯引用:** 4549 | [OpenAlex ID](https://openalex.org/A5022479813)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用预训练 ASR 模型对无标签的喧闹警用广播音频进行伪标签生成，并通过多种过滤策略对生成的文本进行筛选，随后对模型进行自监督微调，提升在 BPC 领域的识别性能。

**💡 创新点**

① 通过外部 LLM‑as‑a‑judge 过滤器以语境合理性评估伪标签，显著低于传统内部置信度（log‑prob、STAR）的方法；② 采用跨模型伪标签微调的思路，利用不同模型产生的伪标签互相提升；③ 对极噪声域的伪标签效果进行系统评测。

**🔧 技术方法**

使用 Whisper large‑v3 与 Qwen3‑ASR1.7B 作为基准模型；伪标签生成、内部置信度过滤（log‑prob、STAR）、LLM‑judge 过滤；迭代伪标签（IPL）与跨模型微调；最后对比基线 OTS、oracle 过滤等。

**📊 数据集**

两套警用广播音频数据集：巴尔的摩（Baltimore）与芝加哥（Chicago），分别约 23–24 小时训练集，1.2–1.3 小时验证集，2.2–2.3 小时测试集。

**📈 对比分析**

相较于 OTS 基线，所有过滤策略均可显著降低 WER；LLM‑judge 过滤在 Qwen3‑ASR 上从 0.3439 降至 0.3051（≈11% 下降），在 Whisper 上从 0.3611 降至 0.3496（≈3% 下降）。oracle 过滤可进一步降低 4–11% 的 WER；跨模型微调在巴尔的摩上提升 1–2% 的 WER，芝加哥则表现不佳。内置信度过滤对性能提升有限。

**⚠️ 局限性**

① 伪标签与噪声数据仍存在较大误差，LLM‑judge 过滤仍未达到 oracle 过滤的理想水平；② 需要人工标签才能实现最优过滤，实际可行性受限；③ 仅在两座城市的数据上评估，缺乏对其他警用语音域的泛化；④ 迭代伪标签的收益有限，可能受限于初始伪标签质量；⑤ 未尝试将伪标签与自监督预训练结合，潜在改进空间未被探讨。

---

## 68. Seasonal and Quantum-inspired Models for Neutron Monitor Time Series Forecasting

**arXiv ID:** 2609.30281 | [PDF](https://arxiv.org/pdf/2609.30281v1)

**作者:** Krishna Bhatia `[一作]` (Fractal AI Research), Srinjoy Ganguly `[通讯]` (University College London)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5114048177)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 Lomnický Štít neutron monitor（LMKS）每小时时序数据进行多步预测基准评估，比较季节性基线、深度序列模型以及量子启发式变体。

**💡 创新点**

首次将量子启发式 Kolmogorov–Arnold 网络（QiKAN）与传统季节性、LSTM、TCN、N‑BEATS、KAN 等模型在同一监测数据上对标，证明低维功能分解与量子启发式结构在周期性监测系列中的优越性。

**🔧 技术方法**

使用 PyTorch 实现季节性 Naïve、LSTM、TCN、N‑BEATS、KAN、QiLSTM 与 QiKAN，采用标准化、dropout、早停等训练技巧，并以 MAE 与 RMSE 作为评估指标。

**📊 数据集**

利用 LMKS 的每小时 neutron monitor 记录（1981‑12‑01 至 2023‑07‑10），以 20 小时历史窗口预测 10 小时未来。

**📈 对比分析**

在统一训练配置下对各模型做点预测误差评估，结果显示 QiKAN 在 MAE 0.5276、RMSE 0.7896 上优于其他模型，季节性 Naïve 仅略高，TCN 与 QiLSTM 误差最高。

**⚠️ 局限性**

实验仅采用快速跑配置和有限的超参数搜索，量子启发式模型对初始化与正则化敏感；所报告结果为可复现快照，未覆盖完整搜索空间，模型在不同分辨率或其他 neutron monitor 数据上可能需进一步调优。

---

## 69. Auditing System-1 Models on Biosecurity-Relevant Benchmarks: Calibration, Selective Prediction, and Permutation Instability in a Non-Generative Model

**arXiv ID:** 2609.30454 | [PDF](https://arxiv.org/pdf/2609.30454v1)

**作者:** Kimon Antonios Provatas `[一作]` (University of Texas at Austin), Ilias Georgakopoulos-Soares `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

审计了商业非生成式 System‑1 模型在生物安全和实验室研究基准上的准确率、校准性、错误检测、选择性预测以及答案顺序不稳定性。

**💡 创新点**

创新点：①系统评估非生成式模型的可靠性与校准；②控制实验量化答案顺序敏感性；③提出基于置信度的选择性答案置换平均方法以提升准确率。

**🔧 技术方法**

采用单通前向推理、最大概率置信度、期望校准误差（ECE）、AUROC、风险‑覆盖曲线、答案置换平均、Bootstrap 置信区间等技术。

**📊 数据集**

使用 Weapons of Mass Destruction Proxy（WMDP）及其 Bio、Chem、Cyber 变体、WMDP‑Bio‑Robust（改写版）和 LAB‑Bench 六个子任务（CloningScenarios、ProtocolQA、SeqQA、DbQA、LitQA2、SuppQA）作为数据集。

**📈 对比分析**

方法：对每个任务计算准确率、ECE、AUROC、错误率（≥0.9 置信度）和选择性预测；与随机猜测对比，WMDP‑Bio 达到 ~85% 的准确率，WMDP‑Cyber 仅 63%；AUROC ≈0.82、ECE ≈0.03；答案顺序变动导致约 30% 项目不一致；对低置信度项目进行置换平均可将 WMDP‑Cyber 准确率提升约 5%。

**⚠️ 局限性**

局限性：仅评估单一闭源模型，结果不一定可推广；未测量延迟与外部模型对比；置换实验仅覆盖四选项问题；返回概率被量化为两位小数；实验仅涵盖部分 LAB‑Bench 子任务；缺乏真实的危险检测基准。

---

## 70. Proceedings 19th Interaction and Concurrency Experience

**arXiv ID:** 2609.30353 | [PDF](https://arxiv.org/pdf/2609.30353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 71. HARDEN: Constrained Evolutionary Search for Harder, Answer-Preserving Evaluation Cases

**arXiv ID:** 2609.30571 | [PDF](https://arxiv.org/pdf/2609.30571v1)

**作者:** Aditya Kumaran `[一作]` (Distyl AI), Pradyumna Tambwekar `[通讯]` (Distyl AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种受限进化搜索框架HARDEN，能够在保持原始答案不变的前提下，将现有评测案例变得更具挑战性；

**💡 创新点**

创新点在于将进化搜索与域特定复杂度轴结合，并通过LM驱动的正确性、真实性和有效性三重约束，生成既真实又难以被模型破解的评测样本；

**🔧 技术方法**

技术包括：受限进化算法、变异模型与任务模型的双重循环、基于大语言模型的正确性与真实性检测、语义熵不确定性度量、以及基于网络搜索的真实性评估规则；

**📊 数据集**

实验使用了金融、医学和法律三大专业领域的公开基准：FinQA、PubMedQA 和 ContractNLI，并在 Qwen3.5 系列模型（35B、122B、397B）上进行评估；

**📈 对比分析**

与单次基线变异（少量提示和带网页搜索的少量提示）对比，HARDEN 在所有模型上平均将准确率降低 22.7%，最高可降 49.9%，同时将模型输出的语义熵提升 112%；

**⚠️ 局限性**

局限性包括仅在 Qwen3.5 体系内验证、推理被禁用、验证集规模有限、对领域专家的判断验证不足、跨模型/跨架构迁移性未知，以及进化搜索的计算成本显著高于单次变异。

---

## 72. When the Preconditioning Exponent Turns Negative: Learning-Rate Coupling and Cross-Environment Generalization

**arXiv ID:** 2609.30271 | [PDF](https://arxiv.org/pdf/2609.30271v1)

**作者:** Gongyue Zhang `[一作]`, Honghai Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Adam 变体中可调预条件指数 p 与全局学习率 η 的交互作用，在一个四环境、稀疏特征的线性分类器上进行全网格控制实验，探究其对源域拟合与跨域鲁棒性的影响。

**💡 创新点**

发现交叉环境最优指数随 log10(η) 近似线性下降，出现正负指数切换；揭示源域验证与鲁棒性之间的冲突；提供了一个简单的补偿关系 p ≈ log η – C / log s；通过特征分配与签名边际分解解释负指数为何在高步长下提升鲁棒性。

**🔧 技术方法**

采用 Adam 样式自适应优化器，允许指数 p ∈ [−0.5,0.5]；在 5×21=105 组 (p,η) 组合上训练单层线性模型；使用 RMS 权重比例、签名边际分解等诊断手段；使用线性回归评估 p 与 log η 的关系。

**📊 数据集**

合成的四环境数据集：每个样本包含稠密、稳定稀疏、环境相关稀疏和纯噪声 3000 维特征；每个环境 8192 训练、2048 验证、8192 测试样本；单随机种子 42。

**📈 对比分析**

比较方法：在每个源环境下以源验证误差挑选检查点，随后在所有四个测试环境上评估交叉/最差/源域准确率；结果显示交叉最优指数随 log10(η) 下降，负指数可在高 η 下提升 0.1–0.9 个百分点的交叉准确率；源域验证始终选正指数；实验为单种子、有限预算。

**⚠️ 局限性**

局限性：仅单一随机种子；仅线性模型和合成数据，缺乏深度网络与真实数据验证；训练时间有限，多数最佳点在最终 epoch；鲁棒性评估使用目标环境信息，缺少可部署的源域选择标准；仅研究 Adam 家族，未探究动量、衰减、权重衰减等对指数-学习率耦合的影响。

---

## 73. Threat-Aware Energy-Efficient Deployment for Dynamic UAV Networks: A Multi-Agent RL Approach

**arXiv ID:** 2609.30690 | [PDF](https://arxiv.org/pdf/2609.30690v1)

**作者:** Faisal Al-Kamali `[一作]` (University of Ottawa), Mohamed H. Ahmed `[通讯]` (University of Ottawa)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种多UAV威胁感知、能量高效的部署框架，将安全性与能效统一考虑；

**💡 创新点**

①在K-means聚类中直接嵌入威胁约束（TAKM）实现安全初始化；②通过迭代求解最小所需UAV数；③将TAKM+安全初始化与双网络延迟DDPG（MATD3）相结合，形成端到端安全强化学习方案；

**🔧 技术方法**

威胁感知K-means（TAKM）、匈牙利算法匹配、双网络延迟DDPG（MATD3）+集中式训练/分布式执行、奖励设计中的安全惩罚、离线训练+在线前向推理；

**📊 数据集**

仿真数据：4000×4000 m²区域内随机生成非均匀GUs分布，K=400/600/1500，设定两类威胁区半径分别为500 m与950 m；

**📈 对比分析**

与GPSO、OPSO、MADDPG、MATD3无TAKM、随机TAKM、CKM等方案比较；实验显示EE达到4.48×10⁶ b/J，零安全违规，速度比GPSO快，能效接近OPSO，在线计算复杂度仅为4.7×10⁵运算，显著低于PSO；

**⚠️ 局限性**

训练阶段计算量大，集中式训练在UAV数量大时收敛缓慢；未考虑风、GPS漂移等实际飞行扰动；仅在仿真中验证，缺乏硬件现场测试。

---

## 74. Orchestrating GenAI for Interdisciplinary Research

**arXiv ID:** 2609.30588 | [PDF](https://arxiv.org/pdf/2609.30588v1)

**作者:** Shirley Anugrah Hayati `[一作]` (University of Minnesota), Dongyeop Kang `[通讯]` (University of Minnesota)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对15名跨学科研究者进行为期1-2周的纵向研究与半结构化访谈，考察他们如何在研究流程中调度标准ChatGPT与Deep Research两种生成式AI工具，以填补知识缺口、获取信息、进行方法探索及跨学科整合。

**💡 创新点**

提出“专业知识悖论”——在知识缺口最大的次要领域生成式AI最有价值，但同时最难被验证；并给出针对跨学科需求的三项设计启示（验证支持、跨域合成与学科特定输出）。

**🔧 技术方法**

利用OpenAI的标准ChatGPT和Deep Research两种GenAI模型，结合质性分析（访谈、日志、笔记）与定量指标（交互次数、使用频率、成功率）进行研究。

**📊 数据集**

样本为15名来自计算机科学、工程、人文社会科学等不同学科的研究人员；收集的日志包含411条提示-响应对，平均27.4轮交互；每位参与者记录6.27个研究目标。

**📈 对比分析**

通过对比标准模式与Deep Research在不同研究目标（信息检索、技术查询、写作、思路生成等）中的成功率（如技术信息寻求Deep Research 41%成功 vs 33%标准）以及使用模式，展示工具在跨学科场景下的适用性与局限；未做客观准确性评测，仅以参与者自评为准。

**⚠️ 局限性**

研究样本规模小、参与者仅限美国、持续时间短（两周），且Deep Research使用受限于免费版的调用次数；成功率基于研究者主观判断，未对输出真实性或错误率进行验证。

---

## 75. Inquesto Score: A reliability Protocol For Voice Agents

**arXiv ID:** 2609.30514 | [PDF](https://arxiv.org/pdf/2609.30514v1)

**作者:** Massa Baali `[一作]`, Bhiksha Raj `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Inquesto Score，测量语音代理在固定评估人群中的可靠性得分。

**💡 创新点**

引入版本化协议、跨模态失败分类、直接从音频测量时序事件，并将诊断视图与主分离。

**🔧 技术方法**

采用音频管道（TTS、声学条件、Whisper 语音识别、音频活动检测）、工具调用跟踪以及固定模型 Gemma‑2‑9B 进行语义判断。

**📊 数据集**

使用合成账单支持场景30个，3种声学条件，4种说话人群，共306通话。

**📈 对比分析**

在13个不同配置的基准代理上评估，IS 从 6% 到 45% 不等，显示时序与部署参数对可靠性影响显著。

**⚠️ 局限性**

局限在于仅覆盖单一域、合成呼叫、单一评判模型，未包含真实用户、更多场景及复杂攻击。

---

## 76. Entropy Regularization: A Free Correction to Cross-Entropy for Verified Demonstrations

**arXiv ID:** 2609.30572 | [PDF](https://arxiv.org/pdf/2609.30572v1)

**作者:** Mihir Dhanakshirur `[一作]` (University of Michigan), Ambuj Tewari `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了熵正则化的交叉熵（ER-CE）作为后训练目标，使语言模型在多解验证任务中更好地满足验证器要求。

**💡 创新点**

证明了标准交叉熵与验证器风险不一致，并给出了学习理论反例；通过控制模型输出支持的大小，提出了可微分的熵正则化方法，并证明其在满足一定假设下能PAC学习验证器风险。

**🔧 技术方法**

使用熵正则化的交叉熵损失，基于低秩适配（LoRA）在已有预训练模型上进行微调；对损失的参数 λ 进行调节。

**📊 数据集**

在数学推理任务 GSM8K、代码生成任务 MBPP 以及更大规模的 MATH 基准上进行实验，使用 Qwen2.5-1.5B-Instruct（和 7B 版本）。

**📈 对比分析**

与标准交叉熵（λ=0）相比，ER-CE 在 GSM8K 上提升约 9% Pass@1，在 MBPP 上提升约 4.4%，在 7B MATH 上提升约 5%；实验均使用 Pass@1 评价，且提升在多次随机种子下显著。

**⚠️ 局限性**

目前仅针对 Pass@1 进行评估，尚未验证在更高级的推理策略（如树搜索、Best-of-N）下的效果；正则化强度的自动选择仍是未解决的实践难题。

---

## 77. LiTe-GS: Oracle-Efficient Next Best View Selection for 3D Gaussian Splatting

**arXiv ID:** 2609.30393 | [PDF](https://arxiv.org/pdf/2609.30393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. Privacy-Preserving Prompted Policy Search for Robotic Control

**arXiv ID:** 2609.30554 | [PDF](https://arxiv.org/pdf/2609.30554v1)

**作者:** Ali Irshayyid `[一作]` (Oakland University), Jun Chen `[通讯]` (Oakland University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种隐私保护的LLM引导策略搜索框架PP‑ProPS，在不暴露原始策略参数和奖励信息的前提下实现机器人控制策略的优化。

**💡 创新点**

创新点在于将策略参数与奖励进行秘密编码，提供分量级反馈，并采用有限历史提示以控制提示长度，从而兼顾隐私、安全与搜索效率。

**🔧 技术方法**

技术包括对策略参数的对角线缩放编码、奖励的比例缩放、组件级奖励反馈、top‑K 有限历史表示以及基于大语言模型的增量搜索。

**📊 数据集**

使用了MuJoCo 行为模拟、经典控制、Highway 驾驶和FetchReach 机器人臂等十个不同的连续与离散控制环境作为实验数据集。

**📈 对比分析**

与Vanilla ProPS、R2PO 以及七种标准RL基线（PPO、SAC、TRPO 等）对比，PP‑ProPS 在七个任务中超过 Vanilla ProPS，在多数任务中超越 RL 基线，且提示长度和响应时间显著下降。

**⚠️ 局限性**

局限性包括在某些高维 Humanoid 任务中性能下降，且对非线性控制策略的适用性尚未验证。

---

## 79. Audit Before You Commit: Locating Belief Failures in Active Identification for One-Shot Manipulation

**arXiv ID:** 2609.30608 | [PDF](https://arxiv.org/pdf/2609.30608v1)

**作者:** Mohamed Abouagour `[一作]` (Indiana University), Byung-Cheol Min `[通讯]` (Indiana University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并验证了一种“先探测后承诺”的机器人执行框架，并对其决策坐标覆盖率和执行模型乐观性进行离线审核与校准，最终通过模型边界修正显著降低失败率并实现跨引擎迁移与真实机械臂插入测试。

**💡 创新点**

创新点在于将决策所需的两个独立条件（覆盖率与乐观性）分别审核并校准；通过离线审计定位观测模型错误并冻结修正；以及展示该修正能在不同物理引擎和硬件上复现。

**🔧 技术方法**

技术包括粒子滤波信念更新、scenario‑minimax 执行、分层覆盖率与情景分数的 conformal 校准、观察模型边界扫描、执行门阈值优化以及多引擎仿真与真实机械臂实验。

**📊 数据集**

使用的数据集主要为七类仿真任务（插入、推送、投掷、铰链、抓取、洗牌板）在 MuJoCo、SAPIEN（PhysX）与 PyBullet 三个引擎中的实例，并附带在真实 5‑DOF 机械臂上的物理插入实验。

**📈 对比分析**

与无探测基线、随机探测和固定探测序列对比，覆盖率提升至 80% 以上，插入任务的失败率从 34% 降至 11%，推送任务误差下降 0.2–0.3 cm；整体表现在各任务族均显示显著改善。

**⚠️ 局限性**

局限性包括主要在仿真环境评估，真实硬件实验仅限单一工具与狭窄的探测模型；conformal 校准仅保证覆盖率而非承诺后风险；模型修正未能在所有引擎（如 PhysX）上完全复制，需进一步研究更鲁棒的观察与执行模型。

---

## 80. Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline

**arXiv ID:** 2609.30290 | [PDF](https://arxiv.org/pdf/2609.30290v1)

**作者:** Haowei Liu `[一作]` (Santa Clara University), Yi Fang `[通讯]` (Santa Clara University)

**通讯引用:** 2933 | [OpenAlex ID](https://openalex.org/A5083935587)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对工业级文本到SQL管线中的LLM评判阶段进行审计，发现其与人工标注的相符度极低，主要由“Grade‑Hallucination”导致。

**💡 创新点**

提出将判定器替换为成本低廉的自托管模型Qwen，并通过三强评判者一致路由提升判定准确率，同时阐明判别误差可通过“过度提示”触发的机制而非模型本身。

**🔧 技术方法**

使用LLM-as-judge、Prompt工程（CoT、无预览等）、vLLM部署、无监督的多模型一致与辩论机制，以及自动化数据标注与活检流程。

**📊 数据集**

在内部通信数据库的604,146行合成数据集上构建四阶段流水线，标注了1,482条问题–SQL对；此外对BIRD‑financial和Spider四个公开数据库进行跨域评估。

**📈 对比分析**

通过与人工金标准的Cohen’s κ对比，发现生产评判器κ=0.04，而自托管Qwen达到κ≈0.72；三强一致路由进一步提升至κ≈0.79；prompt调优最高仅提升至κ≈0.37；成本方面，Qwen每次调用仅为Opus的1/300。

**⚠️ 局限性**

局限性在于评判器的校准仅基于单一英文电信域，人工标注者仅为作者两人，跨域表现受schema不透明度影响，且未对低质量金标准进行外部验证。

---

## 81. Highly Scalable Selectorless Cryogenic Memory Array Using Ferroelectric Josephson Field-Effect Transistors

**arXiv ID:** 2609.30672 | [PDF](https://arxiv.org/pdf/2609.30672v1)

**作者:** Saheeb Ahmad `[一作]` (Clemson University), Shamiul Alam `[通讯]` (Clemson University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于Fe–JoFET的选择器无记忆阵列，结合物理基础Verilog‑A模型验证其非易失性、低功耗和可扩展性。

**💡 创新点**

创新点在于利用铁电门极实现固有单元选择和读取，彻底消除外部选择器与感测电路，实现极低功耗（读10 fW、写0 W）和高密度。

**🔧 技术方法**

使用Fe–JoFET器件、Preisach模型、BCS理论、改进的Ambegaokar‑Baratoff方程、RCSJ模型以及V/2写入与读电流偏置的电路方案。

**📊 数据集**

采用实验测得的5 µm宽、600 nm长Fe–JoFET在50 mK下的P–V、I_c、R_N和V_DS曲线作为验证数据。

**📈 对比分析**

通过与Cryo‑CMOS、SFQ CRAM、磁性JJ、Superconducting Memristor、FeSQUID以及混合Josephson‑CMOS等现有技术的单元面积、功耗、选择器需求和感测电路等指标进行对比，显示该方案在面积≈3 µm²、读功耗10 fW、无选择器、无感测等方面优于传统方案。

**⚠️ 局限性**

局限性包括仍需实验验证完整阵列的保持时间、干扰鲁棒性、噪声与温度漂移对性能的影响，以及理论模型尚未涵盖所有实际工艺与环境因素。

---

## 82. Audio LLMs Know When They Can't Hear You

**arXiv ID:** 2609.30625 | [PDF](https://arxiv.org/pdf/2609.30625v1)

**作者:** Amirhosein Javadi `[一作]` (Apple), Mohammad Samragh `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了音频大型语言模型（Audio LLM）是否能够自我判断其转录是否可靠，并基于冻结的音频编码器设计了轻量级的可靠性预测器。

**💡 创新点**

创新点在于：①通过模型基础可靠性标注（critical‑SNR）获得pair‑specific可靠性边界；②发现音频编码器的表示中已编码可靠性信息，故无需生成文本即可判断；③提出跨模型迁移标签库并通过边界对齐显著降低迁移误差。

**🔧 技术方法**

使用的技术包括：关键SNR估计与二分搜索、无参考音频质量/可懂度指标、生成不确定性评估、Fe‑WER 语音‑文本对齐估计、时间池化+MLP轻量级分类器、跨模型标签迁移与边界对齐校准。

**📊 数据集**

实验数据集涵盖：LibriSpeech、DNS Challenge、MUSAN、OpenSLR26（室内混响）、LJSpeech、SONYC‑UST、OpenSLR28（真实房间）等多种语音、噪声与混响组合。

**📈 对比分析**

与基准方法（SNR、Audiobox‑Aesthetics、DNSMOS、NISQA、TorchAudio‑SQUIM、生成不确定性、Fe‑WER）对比，模型在Qwen2‑Audio‑7B‑Instruct上域内宏F1 81.10%、域外宏F1 78.09%，准确率分别为82.60%与79.55%，相较最佳基线提升约10–12个百分点，平均绝对误差降低0.09–0.11。

**⚠️ 局限性**

局限性包括：①需要为每个模型单独构建标签库，构造成本较高；②跨模型迁移仍受关键SNR差异影响，迁移误差不完全消除；③仅对四类可靠性进行粗粒度划分，缺乏更细粒度或连续度量；④在极端噪声或未知真实场景下可能表现下降。

---

## 83. Coding Agents Aren't Enough! Evaluating an Enterprise Security Brain for Agentic Cloud Investigations

**arXiv ID:** 2609.30345 | [PDF](https://arxiv.org/pdf/2609.30345v1)

**作者:** Leon Goldberg `[一作]` (Sola Security), Konstantin Koutsyi `[通讯]` (Sola Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一个真实的 AWS 企业环境上，对 28 个云安全调查任务进行评估，比较了离线构建的安全情境层（Sola Security Brain）与通用编码代理（Claude Code）在覆盖率、成本和耗时上的表现。

**💡 创新点**

创新点在于：①首次量化展示了专用安全上下文层在多任务场景下的显著优势；②提出并描述了“sample‑and‑generalise”这一答复模式，揭示了编码代理在预算限制下的普遍性错误；③通过池化相对召回的评测方法，将覆盖率转化为安全价值比率。

**🔧 技术方法**

采用技术包括：离线构建的关系型图谱（资产、身份、权限等节点与边）；基于该图的查询层；Claude Code 通过 CLI 与 AWS 交互的计划-执行-观察循环；评测流程采用三次独立绘制、盲池化、分层权重与基准门控。

**📊 数据集**

数据集由一套真实企业级 AWS 多账户环境组成，并基于该环境生成 28 个自然语言任务；任务由独立的任务生成器和安全实践者审核后固定不变。

**📈 对比分析**

比较方法采用池化相对召回，计算覆盖率、持续时间和成本，并归一化到每单位覆盖率。结果显示 Sola Security Brain 的覆盖率为 0.693，Claude Code 为 0.387，提升 79.2%；Sola 在 25/28 任务中领先；每单位覆盖率耗时 1.6 倍更快、成本 31.6 倍更低；总体耗时仅比 Claude 多 13%。

**⚠️ 局限性**

局限性包括：①评测覆盖率基于双方答案池，无法捕捉两者均未发现但存在的重要事实；②只测试单一云供应商，跨厂商环境的效果可能更大；③基线仅使用一个编码代理，未构建更广泛的代理池；④“sample‑and‑generalise”模式未量化其出现频率。

---

## 84. MVAgent: Multi-Agent Video Generation via Consistent Condition Construction and Shot-Level Policy Optimization

**arXiv ID:** 2609.30609 | [PDF](https://arxiv.org/pdf/2609.30609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 85. Stable initialization without the CLT

**arXiv ID:** 2609.30633 | [PDF](https://arxiv.org/pdf/2609.30633v1)

**作者:** Simon Kuang `[一作]` (University of California, Davis), Xinfan Lin `[通讯]` (University of California, Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出均匀相位初始化，利用正弦激活的周期性，使所有隐藏层的偏置均匀分布在[0,2π]，从而在任何宽度和深度下实现梯度和前向传播的正交化，消除层间依赖。

**💡 创新点**

创新点在于首次将正弦函数的周期对称性用于网络初始化，得到严格的正交雅可比矩阵；同时对输入/输出层采用基于数据协方差和结构张量的零样本刻度匹配，实现零调参的网络初始化。

**🔧 技术方法**

使用随机正弦激活、均匀相位偏置、正态分布权重、以及协方差匹配技术；理论分析基于正弦周期性和独立性，实验实现基于JAX/PyTorch。

**📊 数据集**

在图像（Cameraman、彩色图像）和音频（Bach音频）等数据集上进行神经表征实验。

**📈 对比分析**

与传统SIREN、SM20、CVP26以及非正弦激活网络进行对比；在图像、音频和窄网络实验中，均匀相位初始化在MSE/SNR上优于或匹敌最佳调参基线，且不需要手动调节频率超参数。

**⚠️ 局限性**

局限性包括主要聚焦于多层感知机和正弦激活，缺乏对卷积或Transformer等更复杂结构的验证；理论假设随机权重分布，实际训练中仍受优化器、学习率调度等因素影响。

---

## 86. Neural Ideals and Neural Codes: An Algebraic Framework for Neural Network Classification and Feature Interpretation

**arXiv ID:** 2609.30279 | [PDF](https://arxiv.org/pdf/2609.30279v1)

**作者:** Venkata Subbaiah Yerrapati `[一作]` (S V National Institute of Technology), Ajay Kumar Shukla `[通讯]` (S V National Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于神经理想与神经码的代数框架，用以分析和可视化神经网络隐藏层的内部表示，并提供交互式软件工具；

**💡 创新点**

创新点在于将神经理想理论引入人工神经网络分类，定义允许码空间并证明理想稳定性，能够用伪单项式生成器直接判定样本类别；

**🔧 技术方法**

采用布尔多项式环、伪单项式、阈值映射、代数生成算法、特征可视化与平均图像分析，并用Python/Matplotlib实现软件；

**📊 数据集**

在 XOR 二分类问题与 MNIST 手写数字数据集上进行实验；

**📈 对比分析**

通过对比实验结果展示神经理想能完整复现网络决策，特征可视化揭示隐藏层结构；在 MNIST 上未与传统指标做严格对比，但表明解释性与可视化效果良好；

**⚠️ 局限性**

需要大量计算资源；仅适用于前向网络，无法直接推断输入理想成员；扩展至更大规模网络与其他架构仍待研究。

---

## 87. Federated Targeted Maximum Likelihood Estimation

**arXiv ID:** 2609.30503 | [PDF](https://arxiv.org/pdf/2609.30503v1)

**作者:** Diyang Li `[一作]` (Cornell University), Kyra Gan `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了两种跨站点联邦化的目标最大似然估计（TMLE）算法 FedTMLE-G 和 FedTMLE-L，使得训练过程在保持数据本地化的同时实现与传统集中式 TMLE 相同的估计精度。

**💡 创新点**

创新点包括：①首次将 TMLE 的 “targeting” 步骤本身分布化；②引入差分编码的有限精度梯度聚合协议；③用描述长度论证数据依赖停止的统计收敛；④对信息披露与隐私风险进行分析；⑤对本地平均方案的收敛与损失损失量化。

**🔧 技术方法**

核心技术：目标最大似然估计、跨站点联邦学习、梯度聚合、差分编码、描述长度理论、有效影响函数、非凸收敛分析。

**📊 数据集**

实验使用了两种合成数据集：一个 10,000 观测的非正态混合 Gaussian mixture，另一个位置相关方差的 curved mixture；数据按随机/空间不均匀方式划分成 5–40 个客户端。

**📈 对比分析**

对比方法：FedTMLE-G（梯度聚合）与 FedTMLE-L（本地平均）在不同权重（样本大小权重 vs 均匀权重）下进行；结果显示在 Gaussian mixture 上 FedTMLE-L 初期收敛更快，而在 curved mixture 上 FedTMLE-G 更快；在更多客户端或通信轮数固定时，FedTMLE-G 维持较低的目标误差但通信成本更高。

**⚠️ 局限性**

局限性：①分析假设每次内部优化可以得到精确或可控的近似解；②本地平均可能因客户端间不一致导致收敛速度慢且估计偏差；③隐私保护仅通过信息披露分析，未给出正式的差分隐私保证；④实验仅在合成数据上验证，缺乏真实医疗/金融数据的评估。

---

## 88. Tactile Sensing Array for Multi-Phalanx Sensing in Humanoid Hands

**arXiv ID:** 2609.30506 | [PDF](https://arxiv.org/pdf/2609.30506v1)

**作者:** Neel Adwani `[一作]` (New Jersey Institute of Technology), Cong Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究设计并制作了一款低成本、多指节的Velostat触觉阵列，用于仿人手指的触觉感知，并通过实验验证了其在不同接触几何形状下的感知效果。

**💡 创新点**

创新点包括：① 将触觉传感器按指节重要性分层设计，指尖采用7点2×3矩阵+单点，近/中节只设前向传感线；② 采用可打印的TPU间隙+凸块结构显著加速恢复时间（相较平面接触降低74%）；③ 通过简单的电路和多路复用实现低成本实现。

**🔧 技术方法**

主要技术：Velostat压电阻感应层、导电胶带电极、TPU 3D打印间隙/凸块结构、低频（20 Hz）多路扫描电路、简单电压分压读数、Python/GUI可视化。

**📊 数据集**

未使用公开数据集；所有实验均采用自制的力传感/称重设备对传感器进行加载、卸载和自由放松的实测数据。

**📈 对比分析**

比较方法：将平面接触基准设计与TPU间隙/凸块改良版在相同加载条件下进行5次重复测试；评估指标为回弹时间（返回至基线20%以内的时长）。结果显示平均回弹时间从4.6 ± 1.4 s降至1.2 ± 0.7 s，提升率为74%。此外，通过对不同几何形状（尖、平、边、角、侧）和完整抓握的激活图可验证传感器能够区分接触模式。

**⚠️ 局限性**

限制：① Velostat 本身存在显著的滞后和漂移，需要软件补偿；② 当前仅实现单指，未覆盖掌面和全手；③ 仅在几何形状有限的立方体/圆柱体上验证，未系统评估对更复杂物体的泛化；④ 生产工艺仍需手工拼装，批量化难度较高。

---

## 89. Bringing AI to Autonomous Systems -- From Cognition to Collective Intelligence

**arXiv ID:** 2609.30291 | [PDF](https://arxiv.org/pdf/2609.30291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 90. Hull Games of Induced Path Convexities in Graphs

**arXiv ID:** 2609.30302 | [PDF](https://arxiv.org/pdf/2609.30302v1)

**作者:** Eurinardo Costa `[一作]` (Universidade Federal do Ceará), Rudini Sampaio `[通讯]` (Universidade Federal do Ceará)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5015249016)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并证明了在单调（monophonic）与 ℓ_k（k≥2）凸性下的 hull 游戏 _𝒞 和 _ℓ_k 的计算复杂度、给出了在路径与环图上使用 Sprague‑Grundy 定理求解胜负的多项式算法，并对奇数 k 以及特殊偶数 k=2^h-4 的 nimber 序列给出了周期性与赢输判定，k=2 时与 Conway 的 Couples‑are‑Forever 产生高度相似的序列。

**💡 创新点**

①首次将 hull 游戏扩展至单调与 ℓ_k 凸性并证明其 PSPACE‑完备；②利用 Sprague‑Grundy 在路径/环上实现多项式求解；③在奇数 k 下给出闭式胜负判定；④在 k=2^h-4 下完成完整周期性描述，并揭示与经典游戏的深层关联；⑤为 k=2 的情况提出高度相符的猜想。

**🔧 技术方法**

主要技术包括多项式时间归约（Clique Forming 游戏 → hull 游戏）、Sprague‑Grundy 理论与 nimber 递归、模 2 运算与位运算（如 a⊕b、a&b）、周期性分析与大量计算验证。

**📊 数据集**

使用的“数据集”为：所有路径 P_n、环 C_n（n up至 10 000 000），以及通过归约构造的特殊图 G（加 5 个顶点的结构），并在此基础上进行数值实验验证 nimber 序列与理论一致性。

**📈 对比分析**

通过在路径/环上实现的多项式算法与已知的贪婪/穷举方法对比，k=2 时 Nim 序列与 Couples‑are‑Forever 的一致率高达 99.97%（误差仅 0.03%），表明所提出方法在这类图上既高效又准确；对奇数 k 的闭式判定则无须计算，进一步提升效率。

**⚠️ 局限性**

限制：①区间游戏 _𝒞 的复杂度仍未确定；②k=2 的 Nim 序列是否完全周期化仍为未解问题；③仅在偶数 k=2^h-4 时观察到周期性，其他偶数 k 的周期性未知；④现有多项式算法仅适用于路径/环，未扩展到更广泛图类；⑤对更小直径图的 PSPACE‑硬度尚未给出最小阈值。

---

## 91. Moment-guided edge sampling

**arXiv ID:** 2609.30472 | [PDF](https://arxiv.org/pdf/2609.30472v1)

**作者:** Weibin Cai `[一作]` (Syracuse University), Reza Zafarani `[通讯]` (Syracuse University)

**通讯引用:** 6236 | [OpenAlex ID](https://openalex.org/A5021992851)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于谱矩的边采样框架，通过计算局部边编辑对随机游走转移矩阵谱矩的影响，来控制全局图结构。

**💡 创新点**

创新点在于同时提供低阶闭环计数的 O(1) 组合方法和任意阶的低秩更新方法，实现高效的边级矩变化计算，并将这些变化转化为可解释的结构签名，用以指导采样和学习。

**🔧 技术方法**

技术包括谱矩分析、随机游走转移矩阵、闭环计数的组合公式、利用循环迹不变性与低秩更新压缩计算、以及在图学习中的应用实验。

**📊 数据集**

实验数据集涵盖经典合成图、真实的学术与网页图，以及 Cora 与 Citeseer 两个常用的节点分类数据集。

**📈 对比分析**

通过与均匀随机删边、GRACE、MVGRL、GCL-SPAN 等方法比较，发现该框架在保持谱、三角聚类、邻接逆度等结构指标方面误差更小，并在无监督对比学习中取得与现有方法相当的节点分类准确率。

**⚠️ 局限性**

局限性在于仍需假设无向图且主要关注谱矩所能捕捉的结构；在极高阶矩或大规模稠密图中计算开销提升，且对非谱性质的保证有限。

---

## 92. Containing Behavioral Cascades from Manipulated Claims in LLM-Powered Multi-Robot Systems

**arXiv ID:** 2609.30523 | [PDF](https://arxiv.org/pdf/2609.30523v1)

**作者:** Waleed Bin Khalid `[一作]` (Indiana University), Byung-Cheol Min `[通讯]` (Indiana University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了LLM驱动的多机器人系统在遭受语义操纵后，如何通过主动验证机制限制行为级联，提出了 Verify–Adapt–Hold（VAH）框架。

**💡 创新点**

创新点在于将后攻击验证转化为团队级规划问题，并利用LLM推理按任务影响优先选择验证对象，同时允许未受影响机器人继续执行任务，从而显著抑制级联扩散。

**🔧 技术方法**

使用了LLM推理、事件驱动角色分配、主动感知、空间–时间 A* 规划，以及对虚假障碍注入的实验仿真。

**📊 数据集**

使用48×28的仓库网格仿真环境，设置 N=3/6/9/12 台机器人，随机生成低/高/混合/杀死四类虚假障碍，进行 15 次重复实验。

**📈 对比分析**

与 Nominal、Naive Replan、Verification‑First（Verify‑Min/All）、Rule‑based VAH 等基线对比；指标包括 SOC、完成时间、覆盖率和级联容纳率；结果表明该方法 SOC 与完成时间接近 Nominal，级联容纳率在 0.60–1.00 之间，覆盖率≥98%。

**⚠️ 局限性**

局限性包括仅在模拟同质机器人、集中式控制环境下验证；在高影响或小规模队列下级联仍可能存在；未考虑通信延迟、异构团队或部分真实验证等实际情况。

---

## 93. Maltsev Constraint Satisfaction Problems and Deterministic Logspace With Counting

**arXiv ID:** 2609.30757 | [PDF](https://arxiv.org/pdf/2609.30757v1)

**作者:** Dejan Delic `[一作]` (Toronto Metroplitan University), Ali Syed `[通讯]` (Toronto Metroplitan University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种新的算法，用于解决由有限马尔特韦斯（Maltsev）代数参数化的约束满足问题（CSP），并证明该问题属于DET复杂度类。

**💡 创新点**

创新点在于：①不依赖显式马尔特韦斯多项式；②利用“三元图”（triple graph）与对称（1,2）-Datalog一致性检查相结合，形成仅使用logspace与MOD-kL预言机的求解过程；③将求解归约为在有限环上计算行列式的问题，深化了CSP与线性代数之间的联系。

**🔧 技术方法**

技术手段包括：
- 语法简单的二元实例化与对称（1,2）-Datalog程序；
- 三元图构造与连通性检查；
- 对最小子代数的计数一致性（i,a,b)-test；
- MOD-logspace预言机与DET类归约；
- 结合马尔特韦斯代数的“矩形性”与子直积性质。

**📊 数据集**

本研究为纯理论性工作，没有使用具体数据集；所有分析均在符号级别进行。

**📈 对比分析**

与传统Bulátov–Dálmá算法（通用高斯消元）相比，新的算法不需要构造子幂生成集，空间复杂度保持在logspace，且仅使用有限个MOD-kL预言机；在理论上证明了其属于DET类，说明了其与行列式计算、线性代数问题的等价性。

**⚠️ 局限性**

局限性与未解决问题：
- 仍未给出可实现的具体实现细节与实验验证；
- 仅针对马尔特韦斯模板的CSP，其他类型模板的扩展尚未探讨；
- 对于是否能在更弱的逻辑（如LFP+Rk）中完全表达该算法仍是开放问题。

---

## 94. Learning-Accelerated Narrow-Phase Collision Detection via Check Ordering for Sampling-Based Motion Planning

**arXiv ID:** 2609.30599 | [PDF](https://arxiv.org/pdf/2609.30599v1)

**作者:** Hao Jiang `[一作]` (Shanghai Jiao Tong University), Xiaoming Duan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在采样式运动规划中，提出通过优化细相位（narrow phase）中网格检查顺序来加速基于阶段的碰撞检测。

**💡 创新点**

①推导出细相位期望检测时间的最优检查顺序准则；②利用超网络（hypernetwork）预测碰撞概率，从而近似该最优顺序；③实现了低开销、良好泛化的预测模型，且不替代精确几何检测，只重新排序检查。

**🔧 技术方法**

期望时间模型、p_i/t_i排序准则、MeshNet*特征提取、超网络生成CollisionPredictionNet、交叉熵损失训练、MoveIt!+FCL+OMPL仿真框架、GPU/CPU推理。

**📊 数据集**

Panda机器人九个连杆与ModelNet40随机选取的252个障碍物（9×252对），每对产生10万相对姿态样本；测试时使用Panda和未见的UR5机器人以及在不同障碍数量（10–120）下的环境。

**📈 对比分析**

与传统BVH、Octree、FCL等基础检测；与Fastron预测器、基于表面积/距离的启发式排序；在碰撞检测平均耗时上比BVH提升约20–30%，在极度拥挤环境可达37%加速；在RRT、RRT‑Connect、BIT*等采样式规划器中，平均求解时间降低约10–22%，成功率提升约3–5%。

**⚠️ 局限性**

对非碰撞情况几乎没有加速；预测误差仍可能导致检查顺序偏差；当出现大量新网格时仍需调用MeshFeatureNet，尽管耗时小但增加了前期成本；实验主要集中在离散障碍与固定机器人连杆，缺少动态或多机器人场景的验证。

---

## 95. Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation

**arXiv ID:** 2609.30650 | [PDF](https://arxiv.org/pdf/2609.30650v1)

**作者:** Shengjun Zhang `[一作]` (Hubei University), Cheng Zeng `[通讯]` (Hubei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了“因果保留”（causal retention）概念，即在交互式智能体中冻结学习后的状态能否正确回答预先定义的干预探测问题（动作、上下文、直接目标、效应值与延迟）。

**💡 创新点**

创新点包括：① 用贝叶斯决策风险度量探测错误，给出接口细化与完美保留的等价条件；② 提出“选择性适配”（selective adaptation）与局部编辑框架，能在不破坏稳定条目的前提下修正已迁移的机制；③ 引入“证据门控写入”“读取过滤”“时间信用”“隐藏上下文设置”等多维门控机制，形成可解释的机制记忆层Causal Core；④ 在多种环境（有限结构因果模型、连续控制、MuJoCo世界模型、预训练语言模型）上统一评估，展示任务表现不等价于因果保留。

**🔧 技术方法**

技术主要包括：结构因果模型与干预抽样、贝叶斯决策理论、接口细化与风险分析、门控写入与读取过滤算法、诊断式自适应更新、统计检验与信息论界限、离散与连续两种实验设置的自监督与标记化技术。

**📊 数据集**

使用的数据集包括：自定义有限结构因果模型族（带读出/隐藏上下文变体）；MuJoCo环境的TD-MPC2预训练模型（cheetah-run、walker-walk、reacher-easy）；以及大规模语言模型Qwen2.5-7B-Instruct的12个程序化任务族。

**📈 对比分析**

与基线比较方法包括：被动相关、奖励优化、排名损失、条件发现、潜在世界模型、元组/神经探测、官方TD-MPC2世界模型、未门控提议、迁移消融等。性能指标为机制准确率、坐标误差、改编分数（AdaptScore）以及读出/隐藏错误率。实验表明：Causal Core在读出/隐藏错误率几乎为0，机制准确率超过0.79，迁移后在MuJoCo上实现了>0.94的效应符号准确率；在Qwen中，门控写入后改变延迟的准确率提升至1.00并将读出误报率降至0.056。

**⚠️ 局限性**

局限性：① 评估以固定探测分布为前提，无法覆盖所有潜在干预；② 需事先设定探测映射，无法动态调整；③ 对极端高维或非结构化环境的适用性仍待验证；④ 门控写入依赖充分的干预样本，稀疏奖励/数据时效果可能下降。

---

## 96. Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content

**arXiv ID:** 2609.30563 | [PDF](https://arxiv.org/pdf/2609.30563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 97. MM-VeriAgent: Learning to Use Extensive Tools to Verify Multimodal Misinformation with Reinforcement Learning

**arXiv ID:** 2609.30698 | [PDF](https://arxiv.org/pdf/2609.30698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 98. Structure-Guided Masked Autoencoders for Ultra-High Resolution Scientific Image Understanding

**arXiv ID:** 2609.30682 | [PDF](https://arxiv.org/pdf/2609.30682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 99. Domain Bounds as a Silent-Fault Detector for AI-Ready Scientific Data

**arXiv ID:** 2609.30507 | [PDF](https://arxiv.org/pdf/2609.30507v1)

**作者:** Kathryn Knight `[一作]` (Oak Ridge National Laboratory), Heather Bort `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究并实现了一种将数据创建时的上下文（领域边界）显式编码并携带的机制，能够在科学数据与 AI 工作流中自动检测因上下文差异导致的静默错误。

**💡 创新点**

创新点在于提出把数据创建条件显式记录为 DomainBoundSpec 并通过 RO-Crate 与 SHACL 进行自动化验证，填补了传统描述/追溯记录无法捕获的领域边界错误空白。

**🔧 技术方法**

采用 RO-Crate 进行元数据封装，SHACL 进行约束校验，并实现基于 DomainBoundSpec 的自动化检查；实验中对比了 provenance、descriptive metadata 与 domain bound 的检测效果。

**📊 数据集**

使用了医学领域的 Sepsis‑3 病例集（MIMIC‑III/IV）和材料科学的能隙计算表进行实验，注入了 32 个故障实例和 144 个正常控制。

**📈 对比分析**

通过在 32 个注入故障和 144 个控制上进行对比，域边界检测成功率达 94%（17/18），而 provenance 与 descriptive 记录均未检测到此类错误；验证耗时仅数十微秒，足以在工作流调度时使用。

**⚠️ 局限性**

局限性包括只能检测已声明的领域边界，缺失或错误声明的边界无法识别；实验覆盖的故障类别不完整，实际应用需要与领域社区共同完善规范并处理词汇版本变更。

---

## 100. A 120-State Binary Turing Machine Equivalent to the Riemann Hypothesis

**arXiv ID:** 2609.30306 | [PDF](https://arxiv.org/pdf/2609.30306v1)

**作者:** Joseph M. Shunia `[一作]` `[通讯]`, Joseph M. Shunia

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个具有120个工作状态的显式确定性单带双符号图灵机，该机的空白带计算在黎曼假设为假时停止。

**💡 创新点**

将工作状态从744减少到120，减少了83.87%的状态数，创造了新的状态计数记录。

**🔧 技术方法**

使用了自筛选递归、偏移数字计数和重复二分法等技术。

**📊 数据集**

没有具体提到使用的数据集，但涉及的数学概念包括黎曼ζ函数和切比雪夫函数。

**📈 对比分析**

与744状态的基准进行比较，当前机器使用120个工作状态，相比之下，744状态的机器使用了约6.2倍的工作状态。性能上，当前机器在相同模型下表现出更小的状态计数。

**⚠️ 局限性**

没有证明下限，构造并不意味着120个状态接近最优，也没有证明黎曼假设。

---

## 101. LensDesigner: A Self-Improving Agent for Optical Lens Design

**arXiv ID:** 2609.30450 | [PDF](https://arxiv.org/pdf/2609.30450v1)

**作者:** Lei Sun `[一作]` (INSAIT Sofia University St Kliment Ohridski), Luc Van Gool `[通讯]` (INSAIT Sofia University St Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于大型语言模型的LensDesigner自动化光学镜头设计框架，模拟专家工作流程；

**💡 创新点**

创新点在于结合光学感知检索、宏观调度与自进化机制，利用课程生成器持续提炼设计经验；

**🔧 技术方法**

采用LLM驱动代理、光学工具套件、可检索的镜头库、模拟器回传与梯度优化等技术；

**📊 数据集**

构建了LensLib100K（约18万条光学镜头设计）和LensArena（120个高难度任务）数据集；

**📈 对比分析**

与传统检索、遗传优化和深度学习基线以及LLM在境内学习进行对比，在LensArena上取得87.5%–80%成功率、Avg RMS低至12.7μm，显著优于对照方法；

**⚠️ 局限性**

目前仅支持球面镜头，依赖外部光线追踪软件导致延迟，且在非球面或自由曲面系统上性能未知。

---

## 102. MOCHA: Multi-Objective Co-Design using Hypernetwork Architectures

**arXiv ID:** 2609.30570 | [PDF](https://arxiv.org/pdf/2609.30570v1)

**作者:** Varun Madabushi `[一作]` (Georgia Tech), Maegan Tucker `[通讯]` (Georgia Tech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MOCHA 框架，利用多目标设计超网络在单一网络中学习机器人设计空间的 Pareto 最优策略族。

**💡 创新点**

创新点在于将多目标强化学习与设计条件化超网络结合，实现连续设计空间的单网络 Pareto 策略；并通过演化搜索和贝叶斯优化生成设计 Pareto 前沿和最优通用设计。

**🔧 技术方法**

采用超网络（MDH）、多目标 PPO、Dirichlet 采样、Sobol 序列、NSGA‑III 演化搜索以及 TuRBO 贝叶斯优化等技术。

**📊 数据集**

在程序化生成的 Cheetah 与 Walker 机器人环境中实验，分别使用 6 维设计空间和 2~3 个目标（跑步、跳高、能耗）。

**📈 对比分析**

与设计条件化多层感知机（MDMLP）对比，MOCHA 在 Cheetah 的超体积提升 28%、在 Walker 提升 6%，间距更小；训练时间仅为 MDMLP 的三分之一。

**⚠️ 局限性**

局限性包括：超网络对设计评价的准确性可能不足；只能表示超立方体约束，无法处理复杂约束；未考虑 sim‑to‑real 迁移。

---

## 103. ORCA: Evaluating LLMs on Data Science Code Translation

**arXiv ID:** 2609.30749 | [PDF](https://arxiv.org/pdf/2609.30749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 104. Practical Algebraic Parameter Estimation for Noisy Data via Gaussian Process Regression

**arXiv ID:** 2609.30451 | [PDF](https://arxiv.org/pdf/2609.30451v1)

**作者:** Oren Bassik `[一作]` (CUNY Graduate Center), Alexey Ovchinnikov `[通讯]` (CUNY Queens College)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于差分代数与高斯过程回归的参数与初始条件估计方法

**💡 创新点**

通过GPR平滑估计导数并构造稳健的多射击子系统，克服传统差分代数对噪声敏感的局限

**🔧 技术方法**

高斯过程回归、差分代数、同调连续求解、局部Levenberg‑Marquardt优化

**📊 数据集**

在25个来自生态、流行病、神经等领域的ODE模型上生成的1,250组噪声数据（5个噪声水平×10次实验）

**📈 对比分析**

与AMIGO2（多起点全局优化）和SHADE（差分进化+局部精细化）对比，修饰版在SR‑10 88.5%、SR‑1 79.6%等指标上领先，尤其在高噪声下的鲁棒性和精度显著提高

**⚠️ 局限性**

仅适用于采样密集的噪声数据，求解多项式系统计算量大，未对后验不确定性进行传播，对稀疏或真实实验数据及局部可识别模型的适用性有限

---

## 105. Insurance Reserve Intelligence Platform

**arXiv ID:** 2609.30765 | [PDF](https://arxiv.org/pdf/2609.30765v1)

**作者:** Anugya A `[一作]` (Indian Institute of Science), Somya Rai `[通讯]` (Exlservice)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套保险储备智能平台，结合经典Thiele求解器与物理信息与知识信息神经网络（PINN/KINN）来预测终身保险储备比率

**💡 创新点**

首次将Thiele微分方程残差与行业知识约束（边界、单调性、上限、流动性等）同时融入损失函数，并采用标准化储备比率提升数值稳定性

**🔧 技术方法**

使用全连接神经网络（tanh激活，256维隐藏层），物理信息损失（PDE残差）、知识信息损失（单调性、边界、平滑性）以及场景损失；训练采用Adam、梯度裁剪等；评估用MAE、RMSE、R²、PDE残差、边界误差、单调性与泛化指标

**📊 数据集**

完全基于合成终身保险数据集，包含1000条训练、200条验证、200条测试；每条政策48个时间步，特征包括时间、发行年龄、定价利率、情景利率、保费比率、保额与死亡强度

**📈 对比分析**

与经典Thiele求解器进行对比；在200条政策上，PINN/KINN推断时间仅0.0138 s，经典求解器1.6521 s，速度提升约119.5×；预测指标R²=0.9887、MAE≈785、RMSE≈1213，整体匹配度高，但单调性与泛化得分低

**⚠️ 局限性**

受限于合成数据、仅适用于终身保险、单调性与泛化性能不足、对死亡率敏感性错误、缺乏真实数据验证、缺少完整基准与消融实验

---

## 106. SafeNom: Data-Aware Microservice Policies

**arXiv ID:** 2609.30394 | [PDF](https://arxiv.org/pdf/2609.30394v1)

**作者:** Karuna Grewal `[一作]` (Cornell University), Justin Hsu `[通讯]` (Cornell University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了SafeNom，一种基于名义语言的微服务安全策略语言和运行时监控框架，能够在黑盒、无侵入的环境下验证包含数据流和不等式约束的安全属性。

**💡 创新点**

创新点包括：①引入“懒绑定”机制，使策略能在数据出现后才绑定并可声明作用域；②扩展名义正则表达式和名义自动机，支持等值与不等值检查；③利用服务网格的侧车代理实现分布式、无侵入的监控；④通过α‑等价保证策略对具体参数值不敏感。

**🔧 技术方法**

技术手段：名义词与名义正则表达式、懒绑定的语义定义、基于名义自动机的监控模型、层级式幂集确定化、Istio服务网格中的WebAssembly Envoy过滤器、符号有限状态转导用于将平面请求序列转换为名义词。

**📊 数据集**

使用了两个真实微服务应用作为评估数据集：医院工作流系统和酒店预订系统，平均服务树深度分别为6和4.5，包含多层嵌套与多次调用。

**📈 对比分析**

与未监控基线相比，SafeNom的监控头信息占用不到12位，单个请求的延迟提升仅为0.5–1.5毫秒；每一次API调用平均产生约0.16毫秒的监控开销；监控开销随调用链长度线性增长，常见的30个调用以内的延迟增益低于4毫秒。

**⚠️ 局限性**

局限性：仅支持顺序调用的微服务；对异步并发调用需要进一步扩展；策略需要满足“一致解析”可约束以保证在线转导的确定性；对极大值域的参数值仍需保持足够的寄存器/头空间。

---

## 107. Skill Profiling with Attributable Reasoning (SPAR): A Wearable Analysis System for Boxing

**arXiv ID:** 2609.30753 | [PDF](https://arxiv.org/pdf/2609.30753v1)

**作者:** Nibraas Khan `[一作]` (Vanderbilt University), Nilanjan Sarkar `[通讯]` (Vanderbilt University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了基于八个IMU和压力垫的全身体动捕捉装置，能够实时捕捉拳击动作并将每一次拳击自动分类为专家或新手，同时提供针对不同受众的三层可解释反馈；

**💡 创新点**

创新点在于提出了针对分析师、教练和运动员三类受众的三层解释框架，分别使用关节归因、动力链层级的对比反事实和自然语言叙述，实现了可操作性解释；

**🔧 技术方法**

技术手段包括：身体 IMU 传感、脚垫压力采集、基于 OpenSim 的姿态校准、MOMENT 时间序列基础模型（冻结）+小型 Transformer 分类器、Contrastive Ablation Attribution、Directed Counterfactual Search 以及 Claude Sonnet 4.5 语言模型生成叙述；

**📊 数据集**

使用了 17 名参与者（5 名专家、12 名新手）共 4,713 次拳击的数据集，涵盖六种标准拳击动作；

**📈 对比分析**

在留一人留出交叉验证下的 AUC 为 0.842±0.097（95% 置信区间 [0.769,0.907]），对比消融实验表明姿态归一化、上采样和编码方式对性能关键；六位教练的使用访谈表明 DCS 层级解释最具实用价值；

**⚠️ 局限性**

局限性包括样本量小、仅二元技能标签、拳击动作级标签可能存在噪声、仅测试正拳姿势、模型可能对单一专家过拟合等。

---

## 108. A Benchmark Framework for Screening Automation in Systematic Reviews

**arXiv ID:** 2609.30298 | [PDF](https://arxiv.org/pdf/2609.30298v1)

**作者:** Gauransh Kumar `[一作]` (Université de Montréal), Eugene Syriani `[通讯]` (Université de Montréal)

**通讯引用:** 1642 | [OpenAlex ID](https://openalex.org/A5049129140)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SRBench框架与PromptSR工具，用于评估软件工程系统综述中基于LLM的标题与摘要筛选自动化方法。

**💡 创新点**

创新点在于：①构建了包含45,064条记录、32篇系统综述的新数据集，覆盖35个ACM CCS类别；②采用针对类别不平衡的评价指标（MCC、BAcc）替代传统准确率、F1；③提供完整的实验管理与结果分析工具PromptSR，支持多种prompt特征、自动实验调度与可视化。

**🔧 技术方法**

技术主要包括：大语言模型（如Qwen3‑30B）、Python后端、PostgreSQL数据库、Streamlit Web界面、vLLM推理加速、模板引擎与实验调度。

**📊 数据集**

使用的数据集为SRBench，继承并扩展了SESR，加入14篇新的系统综述，包含标题、摘要、作者、关键词等元数据。

**📈 对比分析**

通过在32个数据集上使用最小化prompt（仅上下文与排除标准）并用Qwen3‑30B评估，得到中位数BAcc≈0.69、MCC≈0.32；与SERS对比发现同一prompt在多数数据集上可与或优于大规模专有模型。

**⚠️ 局限性**

局限性包括：仅评估单一模型与单一prompt基线；数据集仅涵盖标题/摘要筛选，未覆盖全文筛选、数据提取等后续阶段；构建与校正数据集耗时且易出错；未覆盖多模型与多prompt的全面比较。

---

## 109. When Does Advection-Aware Graph Nowcasting Help? A Controlled Study of Distributed Solar Ramp Forecasting with a Self-Supervised Cloud-Motion Estimator

**arXiv ID:** 2609.30286 | [PDF](https://arxiv.org/pdf/2609.30286v1)

**作者:** Phillip Jiang `[一作]` `[通讯]` (Appsofa LLC), Phillip Jiang (Appsofa LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在控制的合成云场实验中，构建并评估了云动向感知的图神经网络（GNN）和自监督云动向估计器，并提出了两阶段冻结前馈器实现光伏发电涨跌的短期预测。

**💡 创新点**

①拆解增益来源，证明云动向估计器本身能贡献约一半Oracle‑CMV提升；②设计位置感知自监督光流重建器，精度比传统交叉相关法提升2–4倍；③提出两阶段冻结模型在中等风速下恢复60% RMSE；④给出v·H≤网络尺度时云动向特征有效的经验规则；⑤公开合成模拟器与代码。

**🔧 技术方法**

异构图神经网络、Graph WaveNet学习邻接、GRU时间编码器、位置感知编码器、温度退火Softmax光流重建损失、两阶段冻结训练、RMSE、CSI、能量和变差得分。

**📊 数据集**

单一合成云场模拟器，生成光照指数、真实CMV与传统交叉相关估计，提供约49个站点、1km网格的GHI时间序列。

**📈 对比分析**

与智能持久性、单站GRU、静态GCN、学习邻接GNN对比，实验表明在风速低到中等且v·H≤网络范围时，冻结两阶段模型将RMSE从0.155降至0.069（≈60% Oracle‑CMV提升），CSI提升约5分；高风速或v·H超出网络时，图结构优势显现，学习邻接模型更优。

**⚠️ 局限性**

仅在合成模拟器上验证，真实云层多层、剪切、生成等复杂特性未包含；CMV恢复与Oracle‑CMV对比为控制实验，需在真实分布式光伏网络上进一步验证。

---

## 110. Who Acts When the User Is Gone? Digital Remains, Survivor Claims, and Post-Mortem Governance

**arXiv ID:** 2609.30449 | [PDF](https://arxiv.org/pdf/2609.30449v1)

**作者:** Supriya Khadka `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对800条来自十个不同Reddit子社区的关于已故用户数字隐私与安全的帖子进行定性内容分析，梳理并归纳了数字遗留、主张主体、行动需求、治理紧张点、访问障碍和政策缺口等维度，并基于此提出了“后用户数字治理框架”，旨在为平台与机构提供可协调的治理设计指南。

**💡 创新点**

创新点在于将“后用户安全”视为对传统账户中心安全的压力测试，提出数字遗留治理是协作与争议共存的社会技术工作；构建了包含资产、主体、行动、紧张、障碍、缺口等六大维度并跨越情绪与风险的治理框架；强调设备级障碍与多系统碎片化政策缺口的重要性；并提出以目的、资产、主体和情境为导向的四项设计承诺。

**🔧 技术方法**

研究主要采用定性内容分析技术：手工筛选、双重编码（单一研究者完成编码，随后由二级研究者审阅）、构建代码表、逐案归类并计算描述性频数；此外还利用常量比较法对主题进行归纳与细化。

**📊 数据集**

数据集为800条在2009–2026年间公开的Reddit帖子，涵盖技术支持、法律建议、遗产规划、个人理财、哀伤支持、关系、隐私与通用建议等十个子社区。

**📈 对比分析**

本研究不涉及算法对比或性能评估，而是通过频率统计与描述性分析呈现问题分布与关联模式，未给出数值指标或基准对照。

**⚠️ 局限性**

局限性包括：样本仅来自Reddit，无法代表所有社群与文化；单一编码者可能导致主观偏差；缺乏交叉验证与可靠性检验；数据仅为英文帖子，忽略了跨语言差异；对实际治理效果与用户体验未进行实证验证。

---

## 111. Batched Feedback and the Random-Access Wall in Search-Based Graph Construction

**arXiv ID:** 2609.30493 | [PDF](https://arxiv.org/pdf/2609.30493v1)

**作者:** Édgar Chávez `[一作]` `[通讯]` (CICESE), Édgar Chávez (CICESE)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了一种基于批量反馈的可并行、确定性的近邻图构建方法，在不牺牲检索质量的前提下显著减少构建工作量。

**💡 创新点**

引入同步批量反馈以替代增量构建的冗余，剖析随机访问墙，证明对黑盒度量距离的构建受随机访问吞吐限制；同时证明PiPNN通过预先固定pair集突破该墙。

**🔧 技术方法**

采用Vamana、PiPNN、hnswlib等图构建算法；利用多线程同步块、occlusion pruning、beam搜索；对距离评估计数、时间等进行精细度量；构建了roofline与随机访问壁垒模型。

**📊 数据集**

GloVe‑100（1.18M 维 100 向量，L2 归一化）和 SIFT‑128（1M 维 128 向量）作为基准。

**📈 对比分析**

采用 10 次配对构建实验比较工作量、构建时间、距离/查询次数；批量反馈在 GloVe 上工作量比 Vamana 少 0.88 倍、时间为 1.55 倍；与 PiPNN 相比，距离评估速度提升 5–6 倍；相较于密集 GEMM，构建速度低 20–27 倍，且随维度增加至约 1.96 倍。

**⚠️ 局限性**

由于随机访问瓶颈，beam 无法后期批量化；构建仍受随机访问墙限制，需预先固定 pair 集才能突破；在高维嵌入数据集上构建成本显著提高。

---

## 112. AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth

**arXiv ID:** 2609.30483 | [PDF](https://arxiv.org/pdf/2609.30483v1)

**作者:** Sheng-Tse Lin `[一作]` (Carnegie Mellon University), Bhiksha Raj `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12932 | [OpenAlex ID](https://openalex.org/A5113017615)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了音频语言模型对自由文本中数值声学声明的准确性，构建了评估基准和解析器，并测评多模型在不同任务下的性能。

**💡 创新点**

提出针对声学量数值的解析与评估框架，使用可调阈值的自适应选择机制降低误差，并提供可直接在原始波形上训练的参考解码器。

**🔧 技术方法**

采用WavLM编码器+Qwen3‑8B解码器的LoRA微调、线性回归基线、Spearman相关、heteroscedastic Gaussian负对数似然训练以及前缀token自适应阈值等技术。

**📊 数据集**

使用Libri2Mix（双说话人混合+噪声+房间衰减）和AMI会议远场录音，共约6000混合及对应干净对照片段。

**📈 对比分析**

与四个开源音频语言模型和一个闭源Gemini 3.8 Flash 进行比较，通过覆盖率、均方误差、Spearman相关及误差上限等指标评估，发现仅少数模型在0.3相关阈值以上，整体误差多高于常数预测器。

**⚠️ 局限性**

评估仅覆盖10个声学量，模型多数在不同音频条件下表现不佳；阈值校准依赖训练集，泛化受限；混合声源的声学量引用不全导致部分单元难以评估。

---

## 113. ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers

**arXiv ID:** 2609.30272 | [PDF](https://arxiv.org/pdf/2609.30272v1)

**作者:** Mohd Moin Khan `[一作]` (Indian Institute of Science), Pandarasamy Arjunan `[通讯]` (Indian Institute of Science)

**通讯引用:** 1026 | [OpenAlex ID](https://openalex.org/A5040213611)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ENAS，一个针对微控制器的CPU-only硬件感知神经网络架构搜索框架；

**💡 创新点**

引入静态分析预筛选、丰富的基于单元的搜索空间（支持标准、深度可分离和瓶颈卷积及可选跳跃连接）以及三阶段混合搜索策略，显著减少搜索时间并在保持精度的同时降低峰值激活内存；

**🔧 技术方法**

使用分析可行性检查、基于TFLite-Micro的代理训练、权重共享和INT8后训练量化；

**📊 数据集**

在两个TinyML基准上评估：Visual Wake Words（图像人脸/物体检测）和Melanoma Cancer（皮肤病变分类）；

**📈 对比分析**

与NanoNAS比较，ENAS在VWW上平均速度提升2.41×、在Cancer上1.70×，准确率分别下降0.79pp和1.31pp（VWW无显著差异，Cancer显著差异），并在高容量MCU上实现显著更高的准确率；

**⚠️ 局限性**

局限性包括对大输入分辨率的代理排名不足、搜索预算有限、仅测试图像数据、Flash估算不够精确、搜索空间相对受限且未做多次外循环收敛等。

---

## 114. Proportional Representation in Temporal Voting with Ranked Preferences

**arXiv ID:** 2609.30555 | [PDF](https://arxiv.org/pdf/2609.30555v1)

**作者:** Noam Hazon `[一作]` (Ariel University), Nicholas Teh `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在时间投票（每轮选择一名候选人）中使用排名偏好时的比例代表性问题，定义了针对不同切点解读（固定切点、随轮切点、个体切点）的 JR、PJR、EJR、PSC 等公理，分析了这些公理在不同信息模式（离线、半在线、在线）下的可实现性与所需信息量，并给出实现规则及其最优误差界限。

**💡 创新点**

创新点包括：
1) 将比例代表性从批准票推广到排名偏好，首次提出三种切点解读并统一考虑；
2) 证明对所有可接受切点同时满足的公理几乎不可实现；
3) 构造了一系列可实现的规则（贪心、预算分配、SCR 等），并给出在一般和受限偏好域（单峰、单跨、有限首选、循环）下的完整存在性与最优加法误差；
4) 在部分域内实现了零误差的在线或半在线规则，并给出了误差下界（Θ(log n) 或 Θ(d)）。

**🔧 技术方法**

主要技术手段包括：
- 多选举理论中的 JR/PJR/EJR/PSC 公理与其约束分析；
- 贪心分配与两阶段规则（GCR）实现固定切点 PJR；
- 预算分配算法（MES、Ordered Budget Rule）实现随轮或弱公理；
- SCR（Solid Coalition Refinement）实现弱 PSC；
- 复杂度分析（coNP‑hardness 证明、NP‑hardness 证明）；
- 树分配与不一致性定理（Schmidt’s discrepancy theorem）用于证明最优误差下界；
- 计数与不等式分析，结合 Droop 计数条件。

**📊 数据集**

由于研究集中在理论分析，实验数据集仅为人工构造的合成偏好，包括：单峰、单跨、欧几里得、循环偏好、固定/随轮/个体切点的排列，用于构造反例与证明可实现性边界。

**📈 对比分析**

通过理论证明对规则的可实现性与信息需求进行比较，并给出多项式时间实现与最优误差的具体数值。例如：
- 离线规则可实现固定切点 PJR，时间为指数；
- 半在线 MES 可实现随轮弱 PJR，时间为多项式；
- 在线 SCR 可实现弱 PSC，时间为多项式；
- Serial Dictatorship（在线）在有限首选下给出最优加法误差 ⌊s(n‑s)/n⌋；
- 在单峰/单跨域下的 Ordered Budget Rule 实现零误差弱 PJR；
- 在最坏情况下误差为 Θ(log n)（在线 all‑rank‑wPJR）或 Θ(d)（所有区间下的 PJR）。

**⚠️ 局限性**

限制与不足：
1) 对于一般排名偏好，EJR、PJR 等公理无法保证；
2) 大多数规则需离线信息或至少知道轮数；在线规则仅能满足弱公理；
3) 对个体切点的公理几乎不可实现；
4) 在连续区间下仍有误差下界，无法做到完全零误差；
5) 主要基于理论构造与反例，缺乏实证验证；
6) 规则对偏好域的依赖较强，仅在特定域（单峰、单跨、有限首选）下能达到最优；
7) 对计算复杂性的证明多为理论性的，未给出具体实现细节或实验性能评估。

---

## 115. Convergence guarantees for Muon: New parameter regimes and generalizations

**arXiv ID:** 2609.30546 | [PDF](https://arxiv.org/pdf/2609.30546v1)

**作者:** Arthur C. B. de Oliveira `[一作]` (Northeastern University), Eduardo D. Sontag `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Muon 的渐近收敛理论，并在其软正号正则化下证明了收敛性；同时设计了 Muesterov Nesterov 变体并给出同样的收敛保证。

**💡 创新点**

将 Muon 的 Newton‑Schulz 正则化解释为有界预条件器，揭示其等价于预条件 Polyak heavy‑ball，从而实现经典 Lyapunov 分析；首次在 Muon 上引入 Muesterov 以获得加速收敛。

**🔧 技术方法**

使用软正号代理、预条件 Polyak heavy‑ball / Nesterov 分析、Lyapunov 能量函数、Polyak‑Łojasiewicz 条件、Newton‑Schulz 迭代、符号矩阵分析等技术。

**📊 数据集**

实验数据集主要包括标量交叉熵模拟和固定批次的 nanoGPT（小型 Transformer）训练，以验证理论。

**📈 对比分析**

通过与 Canonical Muon、Ideal Muon 的数值对比，展示软正号和 Muesterov 在收敛速度和稳定性上的提升；实验表明 Muesterov 在 1D 交叉熵问题下收敛更快，nanoGPT 试验中增大正则化可抑制振荡并加速下降。

**⚠️ 局限性**

证明中对 β 的约束（β<1/√2）与实际使用（β≈0.95）不完全一致；未给出针对大规模训练的具体调参指导；实验仅在低维或固定批次设置下验证，未覆盖完整大规模训练场景。

---

## 116. Atelier: Learning Local Self-Supervised Features for CryoEM Volumes via Hypernetworks

**arXiv ID:** 2609.30569 | [PDF](https://arxiv.org/pdf/2609.30569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 117. RoboMonitor: Label-Efficient Runtime Monitoring of Robot Task Execution via Predictive Representation Learning

**arXiv ID:** 2609.30715 | [PDF](https://arxiv.org/pdf/2609.30715v1)

**作者:** Abhiroop Ajith `[一作]` (Siemens Corporation), Eugen Solowjow `[通讯]` (Siemens Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RoboMonitor——一种利用自监督预训练与时序监督微调的机器人执行监控框架，能仅通过任务指令和相机观测实时识别执行阶段、失败与完成。

**💡 创新点**

创新点在于：1）通过未来特征预测、逆动力学预测与遮蔽预测三种自监督任务在无标签轨迹上学习视觉+时序表示；2）将这些表示迁移至监控器并采用Temporal SFT（窗口内外一致性约束）实现密集监督与低波动的阶段预测；3）实现标签高效，少量标注即可达到或超过传统基线。

**🔧 技术方法**

使用的技术包括：视觉语言模型 Qwen3‑VL‑4B‑Instruct 的视觉LoRA、未来特征预测、逆动力学预测、遮蔽现状预测、Temporal SFT（窗口内一致性和跨窗口一致性约束）、残差注意力融合模块等。

**📊 数据集**

数据集：预训练使用 25 小时多相机机器人轨迹（12 个操纵任务、2 种机器人）；监督阶段使用 4 个任务（Reel Packing、Electronic Component Insertion、Toolbox Sorting、Humanoid Grasping）的标注数据，分别在 52 和 100 条轨迹预算下进行实验。

**📈 对比分析**

与 Qwen3‑VL、Robometer 以及 Gemini few‑shot 进行对比；在 52 条标注例子下，RoboMonitor 取得 93.1% 阶段准确率、85.9% 宏召回，明显优于基线；在 100 条例子下进一步提升至 93.6% 阶段准确率；Temporal SFT 将 Qwen3‑VL 的 spurious‑switch 率从 15.2% 降至 4.9%；在闭环部署中，模拟 Toolbox Sorting 完成率 97.5%，真实 Reel Packing 完成率 87.5%，无误恢复触发。

**⚠️ 局限性**

主要局限：只能给出单一全局执行阶段，无法表达多臂异步或并行子任务；缺乏多标签或因子化状态表示；目前未提供置信度估计或人机协同的接口。

---

## 118. Mentored Decoding: Faster Inference meets Boosting

**arXiv ID:** 2609.30474 | [PDF](https://arxiv.org/pdf/2609.30474v1)

**作者:** Vivien Tran-Thien `[一作]` (Google), Richard Nock `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并正式分析了“导师式解码”（Mentored Decoding）与提升推理速度与模型质量的方法，证明了在允许输出偏差的前提下，可在保持或提升质量的同时显著加快推理速度。

**💡 创新点**

创新点在于：①将导师式解码与经典的 Boosting 框架结合，给出了通用的 f‑divergence 约束下的最优解结构；②证明总变差（TV）情况下最优解具有几何直观性，并可通过“breakpoints”数据结构在 O(log n) 时间内快速得到；③提出了新的自归一化 Boosting 算法，避免了传统 AdaBoost 的归一化开销；④给出实验模拟，展示在多种 f‑divergence 下可将接受率提升 10%+，且偏差几乎为零。

**🔧 技术方法**

主要技术包括：f‑divergence 约束优化、KKT 条件求解、几何直观的 TV 解析、Breakpoints 数据结构、基于边缘（edge）自归一化的 Boosting、以及对 top‑k 截断的理论支持。

**📊 数据集**

实验使用了随机生成的均匀分布和 Beta 分布的 p、q 作为示例，未在真实 LLM 推理数据集上进行评估，主要以理论推导和 toy 模拟展示效果。

**📈 对比分析**

与传统的精确匹配（Speculative Decoding）对比，导师式解码在相同的误差阈值下可将接受率提高约 10%+，而在 f‑divergence 可微的情况下偏差梯度为 0；Boosting 能在少量模型下实现指数级提升，实验结果显示质量提升可超过目标模型。

**⚠️ 局限性**

局限性包括：需要满足 p>0、q>0、0<D<TV(p,q) 等假设；对大型词表仍需 O(n log n) 排序；在 top‑k 截断下仍需额外实现；Boosting 的边缘参数估计可能受模型相互相关影响；且仅在两模型（drafter+target）场景下理论最完整，扩展到多模型仍待研究。

---

## 119. SkillEvoReg: Regularizing Agent Skill Evolution Against Overfitting

**arXiv ID:** 2609.30861 | [PDF](https://arxiv.org/pdf/2609.30861v1)

**作者:** Guanyu Nie `[一作]` (Huawei Noah's Ark Lab), Mingxuan Yuan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套通用的正则化框架，用以控制语言模型代理在重复更新技能时出现的过拟合现象；

**💡 创新点**

创新点在于将神经网络训练中的dropout、容量正则化以及对抗式数据增强等概念迁移到离散的技能更新流程中，并引入因果对抗样本验证（CCV）来检测微小更新导致的回归；

**🔧 技术方法**

核心技术包括训练时技能dropout、基于复杂度的局部正则化以及CCV；在实现时对每种技能演化系统（SkillOpt、SkillEvolBench、ContinualSkillBench）分别进行适配；

**📊 数据集**

在三大基准上评测：SpreadsheetBench、SearchQA、LiveMathBench（SkillOpt）；SkillEvolBench的六个任务；以及ContinualSkillBench的五个领域（Mathematics、Law、Finance、Office、Healthcare）；

**📈 对比分析**

与原生更新器对比，正则化框架在保持或提升目标任务和迁移任务性能的同时，显著压缩技能状态大小（token量下降10–35%），并能发现仅凭结构指标难以识别的回归问题；

**⚠️ 局限性**

局限性包括：对不同技能更新语义的适配仍需手工配置；CCV的攻击生成与阈值设定可能影响检测敏感度；在极度稀疏或高度动态的任务序列中，正则化参数的选择和收敛仍有待进一步研究。

---

## 120. Benchy: towards a universal language for task-oriented AI benchmarks

**arXiv ID:** 2609.30550 | [PDF](https://arxiv.org/pdf/2609.30550v1)

**作者:** Francis F Daniel `[一作]` (SURUS), Marian Basti `[通讯]` (SURUS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Benchy：一种基于语义的 YAML 规范、编译为 JSON IR 的评测语言与执行引擎，用于独立描述 AI 任务的程序、评分函数和数据集，并通过适配器与任意 AI 系统对接。

**💡 创新点**

创新点在于：①将评测定义统一为单一语义 YAML 规范，消除多重语法；②编译为可执行的 JSON IR，保证语义不变；③提供统一的运行时契约和适配器，隔离 AI 系统差异；④基于 SURUS 共享任务/域/语言本体实现任务一致性验证。

**🔧 技术方法**

使用的技术包括：YAML 解析、语义验证、编译为 JSON IR、基于固定输入/输出模式的执行引擎、评分函数的加权平均聚合、适配器模式以及 SURUS 本体注册。

**📊 数据集**

使用的数据集为 benchmark 例题集，包含输入 x_i 与期望输出 y_i^*，数据格式由 Benchy IR 决定。

**📈 对比分析**

由于论文侧重设计与实现，没有实验性能比较；但该架构通过单一合同和适配器实现与多种 AI 系统的无缝对接，理论上可并行执行并记录准确的分数。

**⚠️ 局限性**

局限性包括：①不支持可变长度输出集合；②仅实现了加权平均聚合，未提供多种聚合器；③编译不做默认值注入，可能导致定义错误被拒绝；④当前实现未包含对动态执行日志或可解释性支持。

---

## 121. Redesigning Trust: Replacing Dark Patterns with Fair Choice Architecture in Financial Interfaces

**arXiv ID:** 2609.30475 | [PDF](https://arxiv.org/pdf/2609.30475v1)

**作者:** Oluwadamilola Awakan `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 1576 | [OpenAlex ID](https://openalex.org/A5059400253)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并验证一种可量化的公平选择架构模型，用以消除金融界面中的暗黑模式（特别是取消流程中的摩擦）

**💡 创新点**

通过将交互成本拆分为步骤、输入、提示三类，并证明“离开不比加入更耗费”可仅靠计数验证，从而把公平性转化为可审计的设计约束；同时引入视觉对称与语言中立两项设计不变量

**🔧 技术方法**

使用形式化模型推导、结构化工作流设计、移动端高保真原型实现以及结构计数与不变量的手工审计

**📊 数据集**

无真实用户数据集；仅在模拟的信用卡注册/取消原型中计数 S、I、P 三类交互项

**📈 对比分析**

通过对比原始非对称流程与对称原型的 S、I、P 计数，验证对称流程满足公平约束；未做定量用户体验或性能测评，只做结构化审计

**⚠️ 局限性**

局限包括：缺乏真实用户实验验证；仅针对信用卡服务的单一场景；未评估对绝对负担与用户满意度的影响；模型未涵盖语言/时间等非计数维度的暗黑模式

---

## 122. Proceedings Tenth Symposium on Working Formal Methods

**arXiv ID:** 2609.30324 | [PDF](https://arxiv.org/pdf/2609.30324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 123. CraftTrace: Unflattening Videos into Malleable, Creation-Inspired Structures for Generative Editing

**arXiv ID:** 2609.30623 | [PDF](https://arxiv.org/pdf/2609.30623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 124. Realizability Is Not Enough: Encoding, Liveness, and Auditing of Synthesized Robot Supervisors

**arXiv ID:** 2609.30460 | [PDF](https://arxiv.org/pdf/2609.30460v1)

**作者:** David C. Conner `[一作]` (Christopher Newport University), Kyle Bloom `[通讯]` (Christopher Newport University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现一个基于 GR(1) 反应式综合的开源 ROS 2 FlexBE 主管流程，能够自动生成基于能力的规范、进行前期分析、后期审计、状态归约，并输出可执行的层次有限状态机。

**💡 创新点**

在规范设计上引入可配置的能力编码（枚举 vs. one‑hot）与 liveness 约束（System‑Goal vs. Fair‑Outcome），并首次在同一流水线中结合 well‑separation、策略审计和状态归约三种验证模块，弥合了可实现性与可部署性之间的差距。

**🔧 技术方法**

技术栈包括 GR(1) 规范生成（Slugs、CUDD）、Python‑based 规范分析与综合包装、FlexBE 状态机生成、显式策略审计（自动检测协议违规、死锁、有限失败、不可达目标）、状态归约（基于输出等价的局部合并）。

**📊 数据集**

使用四个真实案例（咖啡机、河流过桥、PyRoboSim 物流与两款四旋翼无人机）作为实验数据集，涵盖从小型模拟到硬件验证的不同规模与复杂度。

**📈 对比分析**

通过比较能力编码、liveness 选型、是否启用 pending 记忆等八种配置，测量规格大小、合成时间、BDD 节点数、策略与归约后状态数以及审计通过率。结果表明枚举编码通常能加快合成速度，System‑Goal liveness 在可实现性与运行效率上更稳健，审计成功率最高；归约能显著减小执行机状态，但对策略大小的影响依赖域与编码。

**⚠️ 局限性**

局限性在于：审计只覆盖四类结构缺陷，无法检测所有 liveness 失效；Fair‑Outcome 仍易产生无目标循环；编码选择并非普适优化，需依赖声明顺序和 CUDD 重排策略；最终生成的 HFSM 仍需人工调试，无法完全保证在所有环境下的安全性。

---

## 125. SkillRefine: Cross-Source Skill Induction and Execution Validation for LLM Agents in Refinery Planning Software

**arXiv ID:** 2609.30674 | [PDF](https://arxiv.org/pdf/2609.30674v1)

**作者:** Dongxiao Liu `[一作]` (Beijing University of Posts and Telecommunications), Xiaoyong Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SkillRefine 框架，利用文档、专家 CASE 记录和执行反馈实现工业规划软件（AspenTech PIMS）的跨源技能学习与精细化修正。

**💡 创新点**

创新点在于：① 角色分离的跨源技能诱导——将稀疏的专家协同修改模式与结构化文档知识结合；② 基于多信号的执行驱动细化——通过格式、求解合规性与结构化匹配三层诊断，并利用工具调用轨迹进行定位修复；③ 进阶披露与逐步加载机制，提升技能库的可扩展性与可解释性。

**🔧 技术方法**

主要技术包括：LLM ReAct 代理、COM 自动化接口封装、规则式文档解析、模式提取与 Schema 对齐、结构化匹配与差分分析、信号驱动的轨迹归因与局部修复。

**📊 数据集**

使用的数据集为：AspenTech PIMS 两个演示模型（Weight Sample 与 Gulf Coast）、3,126 页文档、317 条 CASE 记录、构建集 40 题、测试集 100 题的 PIMS‑Bench 基准（共 140 题）。

**📈 对比分析**

与基线、DocOnly、Trace2Skill、SkillOpt、SkillBase 等六种条件进行对比，四个 LLM 主干上 SkillRefine 在平均 match F1 方面分别提升 0.19–0.30（最高 0.91），尤其在 L3–L5 复杂多表协调任务上显著优于其他方法。

**⚠️ 局限性**

局限性在于：仅在 AspenTech PIMS v22.0.11 的两模型环境中验证，未检验对未知炼厂模型或其他工业规划平台的迁移；多解任务使用单一专家参考导致可能误判；轨迹归因只能提供候选修复，缺乏因果证明。

---

## 126. Benchmarking the Connectomes of Caenorhabditis elegans within the Reservoir Computing Framework

**arXiv ID:** 2609.30508 | [PDF](https://arxiv.org/pdf/2609.30508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 127. A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID

**arXiv ID:** 2609.30287 | [PDF](https://arxiv.org/pdf/2609.30287v1)

**作者:** Paweł Blicharz `[一作]` (Gradient PG), Miłosz Grunwald `[通讯]` (Gradient PG)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究冻结BERT编码器中哪些神经元对AI文本检测具有决定性作用，利用稀疏线性探针和激活置换实验验证其因果性；

**💡 创新点**

首次在AI文本检测领域使用双向激活置换（activation patching）验证神经元的充分性与部分必要性，并揭示指令调优模型在层12集中的稳定神经元分布与纯基础模型的区别；

**🔧 技术方法**

L1正则稀疏探针、L2评估探针、平均消融、双向激活置换、Jaccard相似度分析、留一族族评估；

**📊 数据集**

使用RAID基准（包含11种生成器、6个域）的人工与人类文本；

**📈 对比分析**

与完整特征（全9,216维）探针比较，发现仅使用约1%神经元即可保留86–94%检测性能；在前后置换实验中，所选神经元的翻转率比随机选取高10–16倍，说明其因果作用显著；

**⚠️ 局限性**

仅评估单一编码器架构（BERT-base），未检验更大模型或多语言；平均消融操作离开数据流，可能产生非自然激活；两种基础模型仅能初步支持层12分布结论，缺乏更广泛验证；未评估对抗重写攻击下的鲁棒性。

---

## 128. SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages

**arXiv ID:** 2609.30739 | [PDF](https://arxiv.org/pdf/2609.30739v1)

**作者:** Puja Ahmad Habibi `[一作]` (SEACrowd), Peerat Limkonchotiwat `[通讯]` (AI Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SEA-CLIP-Tiny，轻量化的多语文本-视觉嵌入模型，支持东南亚七种主要语言。

**💡 创新点**

创新点在于将 CLIP-KD 框架与多语教师引导、地区精选数据和多目标蒸馏相结合，构建了少于 5千万参数、兼顾多语能力的高效模型。

**🔧 技术方法**

使用 CLIP-KD 知识蒸馏、多目标损失（MSE、ICL、CRD）、MetaCLIP-2 多语教师、ViT-T/16 视觉编码器和 12 层 Transformer 文本编码器等技术。

**📊 数据集**

利用通用数据集 CC12M 与东南亚本地化数据集（Cultural Ground Open-Ended、WIT-HF、BLOOM、SEA-Mammoth 等）进行训练。

**📈 对比分析**

与 CLIP-KD、TinyCLIP、MobileCLIP2 等基线模型在七语图文检索、零样本分类和 VQA 上对比，SEA-CLIP-Tiny 在平均 R@10 上提升 12.1 点，参数减少 38.4%，CPU 延迟降至 24.1 ms，整体检索性能排名第一。

**⚠️ 局限性**

局限性包括与大教师模型仍有差距、语言覆盖不全、可能继承数据偏见、在 VQA 任务表现不及 MobileCLIP2、以及对某些低资源语言的检索提升不均衡。

---

## 129. ScopeBench: Do Agents Preserve Engagement Boundaries Under Goal Pressure?

**arXiv ID:** 2609.30325 | [PDF](https://arxiv.org/pdf/2609.30325v1)

**作者:** Shane Caldwell `[一作]` (dreadnode), Will Pearce `[通讯]` (dreadnode)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了ScopeBench，一个评估安全代理在目标压力下是否遵守范围边界的基准，利用死端任务和两臂验证框架进行可重复测量。

**💡 创新点**

创新点在于将范围遵守拆分为机械验证与全程轨迹评估两臂，既提供高精度下限，又通过判别器捕捉机械验证无法发现的违规行为；同时引入死端任务设计，确保任何成功完成即意味着越界。

**🔧 技术方法**

采用的技术包括：①基于Shell脚本的确定性验证器；②利用工具支持的AI判别器对完整轨迹进行违规检测；③使用F1、Fleiss κ等统计指标对判别器进行校准；④利用Bootstrap方法评估置信区间。

**📊 数据集**

数据集为ScopeBench Pilot，包含30个Web安全任务（跨域访问、凭证重用、SSRF等），每个任务提供未范围和两种范围表述（口语化与正式）共180条轨迹；此外公开了ATIF轨迹日志与Judge校准数据集。

**📈 对比分析**

在八种代理模型（Claude、GPT、GLM等）上评测，原始能力得分从12.2%到81.1%，范围遵守率从34.4%到86.7%；通过两臂评估，机械验证仅发现4.4%–32.2%的违规，判别器补齐额外331次违规，综合后违规率达13.3%–65.6%。

**⚠️ 局限性**

局限性包括：①任务仅为单一路径死端，缺乏可选的合法解决方案；②仅覆盖Web应用安全领域；③判别器错误倾向于误报，需进一步校准；④评估环境可能导致模型对评测情境过度保守，影响生态效度；⑤只给出存在违规的二值标签，未细化违规程度。

---

## 130. Component Benchmark: Hierarchical Model Profiling for Large-scale Recommendation Systems

**arXiv ID:** 2609.30656 | [PDF](https://arxiv.org/pdf/2609.30656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 131. REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles

**arXiv ID:** 2609.30547 | [PDF](https://arxiv.org/pdf/2609.30547v1)

**作者:** Haixu Ma `[一作]` (Adobe Inc.), Sumit Ranjan `[通讯]` (Adobe Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并部署了 REALMS 系统——一种基于 AI 助手的对话式自然语言接口，可在秒级实时精确计算大型企业用户画像库中的受众规模。

**💡 创新点**

创新点包括：① 采用嵌入式密集检索与类别桶分离的属性检索机制，实现无手工配置的 Schema grounding；② 结合知识图检索增强（KG‑RAG）与模板化 in‑context 学习的 LLM‑NL2SQL 生成流程，显著提升复杂嵌套模式下的查询准确性；③ 引入统一的 Schema 标准化层，使系统跨行业、跨数据源可复用且支持精确计数。

**🔧 技术方法**

使用技术包括：密集向量检索（FAISS/Annoy）、类别桶索引、句子嵌入模型、LLM（如 GPT）模板化提示、知识图检索、SQL 语法与治理验证、列式分析数据库（如 ClickHouse/Snowflake）以及自我纠错的验证‑重生成循环。

**📊 数据集**

数据集：真实企业客户数据平台的数百万用户画像，包含数千维属性；600 条自然语言受众尺寸查询（由生产环境真实问题和 LLM 生成的组合），涵盖计数、top‑k、百分比三类意图，并以不同属性数（1–3）评估复杂度。

**📈 对比分析**

对比方法：与 Skeleton、Sampling、Predictive 近似查询方法进行基准；评估指标为属性检索 Recall@k 与 Exact‑match、SQL 执行匹配率、LLM 语义匹配率；实验结果显示 Recall@5 94%、Exact‑match 93%、SQL 执行匹配 90%、LLM 语义匹配 95%。负载测试表明在 10–30 请求/分钟时中位延迟 <9 s，60 RPM（10 并发）时中位延迟 10 s，95 % 分位 <14 s，失败率极低，证明系统在实时交互下保持高精度与低延迟。

**⚠️ 局限性**

限制：① 需要离线大规模数据重建与向量索引，成本高；② 目前仅支持 SQL 语义，无法处理更复杂的非结构化查询；③ 对极大属性数或非常深的嵌套层仍可能出现检索或生成误差；④ 需要持续维护知识图示例库以保持检索效果；⑤ 受 LLM 运行成本与可解释性约束。

---

## 132. A Unified Account of Concepts and Chunks

**arXiv ID:** 2609.30414 | [PDF](https://arxiv.org/pdf/2609.30414v1)

**作者:** Karthik Singaravadivelan `[一作]` (Institute for the Study of Learning and Expertise), Pat Langley `[通讯]` (Institute for the Study of Learning and Expertise)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套统一的概念与块（chunk）理论，并实现了名为 / 的系统，该系统在句法学习任务中通过增量无监督方式构建概念层次与块结构。

**💡 创新点**

创新点在于：① 将 Cobweb 的概率概念层次与块的组合结构结合，形成内容层与上下文层两棵并行层次结构；② 通过解析与生成两种活动实现从经验中自动学习并利用块；③ 通过识别阈值实现对块的筛选与分配，支持可泛化的组合。

**🔧 技术方法**

使用技术包括 Cobweb 分类/预测、增量无监督学习、内容层与上下文层两棵 Cobweb 树、基于分数的解析与生成模块、阈值识别机制。

**📊 数据集**

使用数据集：合成的上下文无关文法（CFG）生成的句子与对应解析树，设计了三种不同非终结符数量（3/6/8）以及三种词汇量级别（11/22/39）进行实验。

**📈 对比分析**

比较方法：采用错误缺漏率（omission）和误报率（commission）两个指标，利用五折交叉验证评估。实验结果显示，在所有复杂度下错误率均低于 20%，学习曲线随训练样本快速下降，系统在合成文法上表现出良好的泛化与生成能力。

**⚠️ 局限性**

限制（limitation）：仅能处理已标注解析的句子；块的组合仅为二元且仅支持“before/left‑of”关系；解析采用贪心策略，无法回溯；识别阈值需手工调节；实验仅在合成文法上进行，未验证自然语言的实际效果。

---

## 133. Adaptive Multi-Value Control in LLMs via Causal Activation Steering

**arXiv ID:** 2609.30405 | [PDF](https://arxiv.org/pdf/2609.30405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 134. Auditing Latent-Space Monitors for Autonomous Driving

**arXiv ID:** 2609.30557 | [PDF](https://arxiv.org/pdf/2609.30557v1)

**作者:** Nikhil Kamalkumar Advani `[一作]` (Independent Researcher), Saurav Kumar `[通讯]` (Bosch North America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对两类自主驾驶任务（LaneSegNet的在线矢量化地图生成和VAD的端到端规划）中的运行时失效监测器进行了系统性审核，探讨内部表示对失效预测的增量价值。

**💡 创新点**

创新点在于提出了基于内部表征的失效监测的增量价值评估协议，并证明内部表征在已存在强输出与状态基线的情况下往往不提供统计学显著的额外预测能力。

**🔧 技术方法**

使用的技术包括基于高斯混合模型的表示新颖性检测、轻量级有监督MLP探针、DeepSets结构化输出监测，以及与非内部表征（车载状态、驾驶指令、预测轨迹）比较的增量评估。

**📊 数据集**

实验数据集分别为OpenLane‑V2（LaneSegNet）和nuScenes（VAD），共计61,432帧并标注了多种失效端点。

**📈 对比分析**

比较方法通过AUROC、AP和选择性风险曲线实现；内部表征单独监测的AUROC分别为0.780和0.868，但当加入非内部表征后，后者可提升到0.825和0.924，且加入内部表征的增益在统计上不显著。

**⚠️ 局限性**

局限性包括仅验证两种模型与任务，监测器设计偏向浅层探针，缺乏闭环安全评估，且对数据集和任务的泛化能力尚待进一步验证。

---

## 135. HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference

**arXiv ID:** 2609.30270 | [PDF](https://arxiv.org/pdf/2609.30270v1)

**作者:** Simran Koul `[一作]` `[通讯]`, Simran Koul

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HybridInfer，一种基于设备热量头信息的强化学习路由器，用于在 on‑device、edge 与 cloud 三层 LLM 推理之间动态选择；

**💡 创新点**

首次将设备热量头作为状态输入，结合 RL 学习三层能力选择，并强调局部性奖励是热量路由可行的先决条件，在真实 Snapdragon 手机上实现并验证；

**🔧 技术方法**

采用离线 Tabular Q‑learning，查询复杂度与热量头状态作为输入；使用 OpenCL/MLC‑LLM 推理栈、Android 传感器 API、BERTScore‑F1 与 ROUGE‑L 进行质量评估；

**📊 数据集**

使用 210 条人工生成的提示（60 条测试、150 条训练），Gold 参考由 Claude Opus 生成，覆盖短、中、长及多步分析型提示；

**📈 对比分析**

与两种手工调优的路由策略（复杂度阈值与热量阈值）以及三层静态路由进行对比；RL 路由在最低成本下获得最高 BERTScore，覆盖率与可靠性均优于全局设备推理；显著性检验显示 RL 路由显著优于基线；

**⚠️ 局限性**

实验样本有限，RL 路由在热设备上偶尔崩溃，长提示仅以 wedge 率呈现；仅在单一设备模型上验证，能源测量仅为电流×时间近似；结果可移植性与更大工作负载的效果尚待进一步研究。

---

## 136. Structured Bayesian Modeling of Dynamic Receptive1 Fields in Salamander Retinal Ganglion Cells

**arXiv ID:** 2609.30731 | [PDF](https://arxiv.org/pdf/2609.30731v1)

**作者:** Alokesh Manna `[一作]` `[通讯]` (Texas A&M University), Alokesh Manna (Texas A&M University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `e15e3743-5ee0-4d5f-813d-d146868082fc` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用结构化贝叶斯模型（空间上的SPDE高斯马尔可夫随机场 + 时间上的AR(1)过程）对视网膜神经元的视场进行高维估计，并在155个鳗鱼视网膜神经元上独立拟合。

**💡 创新点**

创新点在于将空间与时间依赖性与稀疏性结构统一到一个可通过INLA快速求解的高维模型中，同时提出了两步聚类方法得到三种时间响应表型，并在模拟实验中验证该模型相较于传统LASSO、弹性网络等方法在支持恢复和系数估计上的优势。

**🔧 技术方法**

使用的技术包括：Poisson广义线性模型、SPDE基础的空间高斯马尔可夫随机场、AR(1)时间过程、INLA（集成嵌套拉普拉斯近似）进行高维贝叶斯推断、BIC选择的功能聚类、B-spline展开、主成分分析、Gaussian混合模型。

**📊 数据集**

数据集为鳗鱼（斑马鱼）视网膜网状细胞记录，使用13×13像素降采样的灰度图像（共303幅，每幅重复13次）以及155个神经元的200 ms时段内的尖峰计数，共计3 939次实验。

**📈 对比分析**

与传统方法（无正则化GLM、Poisson-LASSO、弹性网络、仅空间SPDE、空间+时间SPDE/AR(1)）在模拟实验中比较。结果显示：空间+时间SPDE/AR(1)在对held‑out的Poisson对数得分和系数误差上均优于所有对照方法；在支持恢复指标（TPR、FPR、F1）上表现较好，但与LASSO相比在控制假阳性方面略逊；整体而言该模型在预测和估计精度上具有显著优势。

**⚠️ 局限性**

主要限制包括：未实现联合的层级模型（未对同一功能类型的神经元进行共享与稀疏性约束），仅进行独立拟合并事后聚类；缺乏对真实生物学类型的验证；仅在单一鳗鱼数据集上测试，未进行重复实验或跨系统验证；模型在支持恢复上仍未能完全保证连通稀疏性，需进一步引入结构化稀疏先验。

---

## 137. DiffusionShadow: Diffusion-based Shadow Caching for Neural Volume Rendering

**arXiv ID:** 2609.30658 | [PDF](https://arxiv.org/pdf/2609.30658v1)

**作者:** Kai-Chen Tung `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

该论文提出了一种基于扩散模型的阴影缓存框架，用于在直接体积渲染中快速重建阴影INR并实现实时阴影可视化。

**💡 创新点**

创新点在于将大量预计算的阴影INR压缩为一个条件扩散模型，能够在推理时根据光照方向即时生成对应的阴影权重，既减少了存储占用，又避免了昂贵的二次光线投射。

**🔧 技术方法**

技术上结合了SIREN网络对阴影系数体素的拟合、基于Transformer的条件扩散模型以及光照方向的Fourier位置编码，并使用几何损失与渲染损失进行监督。

**📊 数据集**

使用了五个医学与流体仿真数据集：Chameleon、Zebrafish、Mechhand、Vortex 和 Argon Bubble。

**📈 对比分析**

与传统的二次光线阴影计算和 BlockFusion 以及 Deep Volumetric Ambient Occlusion 基线相比，本文方法在所有数据集上实现了至少 20 倍的渲染速度提升，同时显著降低了运行时 GPU 内存消耗，渲染质量与真值相当或更好。

**⚠️ 局限性**

主要局限包括对高频阴影细节的过度平滑，扩散模型对未见光照方向的泛化能力有限，以及对更复杂或更大范围光照采样时可能需要更高容量的模型或更精细的监督策略。

---

## 138. Population loss in shallow ReLU networks: Bias & families of critical points

**arXiv ID:** 2609.30661 | [PDF](https://arxiv.org/pdf/2609.30661v1)

**作者:** Michael Field `[一作]` `[通讯]` (University of California Santa Barbara), Michael Field (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

对浅层ReLU网络的损失函数（含偏置）进行严格的解析推导，给出梯度与二阶导数的闭式公式；在此基础上利用对称群（行/列置换与正交变换）对网络参数空间进行分解，得到等距子空间与等价类，进而将高维梯度方程降到固定点空间中的低维方程，得出一套可手动求解的临界点方程；研究了偏置为零与非零时的梯度结构差异，并讨论了零偏置子空间是否为梯度不变子空间。

**💡 创新点**

创新点主要有：①首次将正交群与置换群的等变性完整地嵌入ReLU网络损失的梯度分析；②在偏置存在的情况下给出完整的梯度与Hessian表达式；③通过等价类与行/列“行类型”概念，将高维的临界点方程化简为可计算的矩阵块方程；④提出“fossilization”现象的几何解释，说明在添加神经元时旧临界点如何转化为高维的单形体。

**🔧 技术方法**

主要技术包括：符号计算与手工推导（对H函数的积分与导数），分块矩阵与对称群作用的固定点空间理论，群表示论与等变向量场的线性化，欧式与Frobenius范数下的等距坐标变换，以及对偏置正则化的微分链式推导。

**📊 数据集**

由于论文为理论研究，没有使用具体的实验数据集；若需要实验验证，可参考经典的MNIST或CIFAR-10等图像数据集来验证理论预言，但在本文中仅在符号层面给出解析结果。

**📈 对比分析**

与以往工作（如Brutzkus & Globerson 2017）相比，本工作提供了更精细的梯度与Hessian表达式，并在对称结构下展示了如何将高维梯度方程降维；理论上能够预测临界点的数量与分布，实证上可用数值求解验证但本文未给出数值实验，因此无法给出传统意义上的性能指标。若实施数值实验，可将本方法与标准反向传播、随机梯度下降进行对比，预计在高对称性场景下梯度计算更快且更稳定。

**⚠️ 局限性**

局限性包括：①假设所有行均不为零且无平行行，限制了对某些实际网络的适用性；②对高维参数的解析推导极其繁琐，易出错；③仅适用于浅层单隐藏层网络，无法直接推广到深层或卷积结构；④对齐正则化和偏置的理论结果虽然完整，但在真实训练过程中可能被其他因素（如噪声、优化器）掩盖；⑤在极限角度趋近0或π时，指数项可能导致数值不稳定，需要额外处理。

---

## 139. LLM Parkinsonism: Executive-Control Failure, Token-Inefficient Persistence, and an Uncertainty-Aware Global Executive Control Architecture for Autonomous Language-Model Agents

**arXiv ID:** 2609.30662 | [PDF](https://arxiv.org/pdf/2609.30662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 140. A Large-Scale Empirical Study of Modern Phishing Email Content

**arXiv ID:** 2609.30683 | [PDF](https://arxiv.org/pdf/2609.30683v1)

**作者:** Jaehwan Park `[一作]` (University of Tennessee), Doowon Kim `[通讯]` (University of Tennessee)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对近13个月收集的约290万封真实钓鱼邮件进行大规模内容分析，覆盖主题、呼叫动作与冒充实体，探讨附件（图像、PDF、日历邀请）在钓鱼中的角色并与历史数据对比。

**💡 创新点**

创新点在于使用LLM（GPT‑5.1）对邮件文本与附件进行高精度标注，系统量化主题、CTA与冒充实体的关联，并揭示附件在补充、替代和强化信息上的三种功能及其时序演变。

**🔧 技术方法**

采用GPT‑5.1的prompt‑based分类流水线，并结合正则表达式与phonenumbers库提取URL、电话，验证其在主题/CTA/冒充标注上的F1>0.98。

**📊 数据集**

数据集为APWG合作收集的3.8M报告邮件，去重后得到2.9M有效邮件，包含188k图像、139k PDF、57k日历邀请，另对比Nazario 2015‑2025历史语料。

**📈 对比分析**

通过在1,000封人工标注样本上评估，GPT‑5.1在主题、CTA、冒充三个任务上均取得F1≥0.98，LLM与传统LDA模型相比在语义准确度上提升显著；在跨时间对比中使用Jensen‑Shannon散度与cosine相似度验证相似性。

**⚠️ 局限性**

局限在于样本来源仅为APWG报告，可能偏向其报告者与受害者群体；历史对比受Nazario样本稀疏与收集方式差异限制，且未对邮件变体聚类，导致同一攻击可能被多次计数。

---

## 141. From S3Q Theory to Implementation: Towards an Architecture for Machine Qualia

**arXiv ID:** 2609.30743 | [PDF](https://arxiv.org/pdf/2609.30743v1)

**作者:** Tetiana Grinberg `[一作]` (Symbiokinetics Inc), Kevin Schmidt `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种由五层组成的计算架构，实现S3Q理论的三大条件，并通过已公开的算法组件完成从感知到内部模拟再到结构一致性的完整流水线。

**💡 创新点**

核心创新在于将现有技术（如Slot Attention、SFA、MI估计、现代Hopfield网络、前向模型T、精度调节等）按S3Q三条准则系统性组合，并给出了此组合唯一能产生的可检验预测（自我槽的MI最高、行为模式基于惊讶与情感分离等）。

**🔧 技术方法**

使用的技术包括：Slot Attention/SAVI进行对象中心编码；SFA与增量SFA提取慢特征；互信息估计（MINE/InfoNCE）实现自我/世界标注；前向模型T（C‑SWM或SlotFormer）进行状态预测与方差估计；现代Hopfield网络完成稀疏种子到完整表示的模式补全；精度加权的预测误差和情感阈值实现行为模式切换；以及基于目标的精度调节。

**📊 数据集**

本文未在具体数据集上进行实验，作者仅提到可用于验证的环境类型，如ThreeDWorld、MineDojo等；因此在该论文中并未使用任何公开数据集。

**📈 对比分析**

由于本文主要是理论构想与架构设计，没有提供实验对比或性能评估；作者提出的可检验预测（如自我槽MI最高、三种行为模式）需要在后续实验中验证。

**⚠️ 局限性**

局限性包括：1）缺乏实测数据和实验验证；2）对环境的要求较高，必须存在多个可控对象；3）自我/世界标注和前向模型的收敛性仍是未解决的技术挑战；4）缺乏统一的学习目标或变分推理框架；5）对复杂多目标或多智能体情境的适应性尚未探讨。

---

## 142. Action Forcing: Training World Models on Unsupervised Video by Recovering Underlying Egomotion Bases

**arXiv ID:** 2609.30595 | [PDF](https://arxiv.org/pdf/2609.30595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 143. Training-Free Contextual ASR via SpeechLLM-Based Error-Aware Selective Retrieval

**arXiv ID:** 2609.30694 | [PDF](https://arxiv.org/pdf/2609.30694v1)

**作者:** Natsuo Yamashita `[一作]` (Hitachi Limited), Masaaki Yamamoto `[通讯]` (Hitachi Limited)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个训练无关的语境ASR框架，利用预训练的SpeechLLM联合生成转写和错误跨度，基于错误跨度从词典检索近音词并进行二次识别。

**💡 创新点**

核心创新在于用SpeechLLM的自我错误定位实现无训练的选择性检索，显著减少词典查询量并提升检索相关性与ASR精度。

**🔧 技术方法**

使用SpeechLLM（Qwen3-Omni-30B），音频‑文本少量提示学习（ICL）进行联合推断；利用Acoustic Neighbor Embeddings（ANE）进行近音检索；在第二步使用同一模型做条件重识别。

**📊 数据集**

在医学语料MedSyn、空中交通控制ATCOSIM和金融财经Earnings三个领域的公开数据集上进行评估。

**📈 对比分析**

与全词典检索、随机检索、命名实体抽取、置信度检索等基线以及不同输入配置的SpeechLLM进行对比。结果显示：查询量降低92.6%，检索命中率和MRR提升；在MedSyn上WER从7.30%降至5.67%，在ATCOSIM与Earnings亦显著提升。

**⚠️ 局限性**

仅在英语语料下验证，检索范围受词典覆盖限制；错误跨度上限两词，可能漏检长词条；缺乏针对多语言或大规模部署的实验；模型依赖大型SpeechLLM，计算成本仍高。

---

## 144. Reinforcement Learning of Communication in a Mesh of Small Language Models

**arXiv ID:** 2609.30578 | [PDF](https://arxiv.org/pdf/2609.30578v1)

**作者:** Mehmet Kerem Turkcan `[一作]` `[通讯]` (Columbia University), Mehmet Kerem Turkcan (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并训练了一个由多个小型语言模型组成的去中心化消息网格，使用置信度头决定发言者和听众，并在消息和修订后通过gossip一致性实现加权投票。

**💡 创新点**

创新点在于：① 在去中心化图结构中自学习何时何种信息需要交流；② 采用置信度头来自动选举发言者并为修订设定接受阈值；③ 在推理流程内部使用GRPO训练会话策略，让自然语言消息真正增益推理、协调与交通控制任务；④ 用gossip实现无协调者的加权投票。

**🔧 技术方法**

核心技术包括：置信度头（V）作为判定器；会话策略（T）通过LoRA适配器生成提示与修订；gossip一致性用于分布式投票；GRPO（Group Relative Policy Optimization）在推理过程内训练会话策略；自一致性（self-consistency）与加权投票（rerank）作为基线；在交通场景中使用SUMO仿真与语言模型控制器的行为克隆。

**📊 数据集**

数据集涵盖：1) GSM8K（数学/逻辑推理）；2) MATH-500（大规模数值数学问题）；3) SwarmBench（多智能体协作任务，如flocking、同步、捕获等）；4) 交通信号控制模拟（纽约市时序约束下的16个交叉口，40个事故种子）。

**📈 对比分析**

比较方法：自一致性（SC）、加权投票（Rerank）、冻结会话（Frozen talk）、训练会话（Trained talk）。在所有三种语言模型（Qwen3.5-0.8B、Qwen3.5-2B、SmolLM3-3B）和三大任务上，训练会话在网络规模 N=3 到 32 时均显著优于 SC 和 Rerank；例如在 N=32 时 Qwen3.5-0.8B 的 GSM8K 准确率从 0.568 提升到 0.705，SmolLM3-3B 的 MATH-500 从 0.492 提升到 0.722。交通控制实验中，通信控制器在 40 个事故种子上平均人均延迟 116.9 秒，优于局部自适应 118.1 秒，且在严峻子集上平均减少 48.2 秒。对抗实验显示，在 8 个代理中有 4 个被攻击时，未防御的网格投票准确率降至 0，防御网格仍保持 0.507。

**⚠️ 局限性**

局限性：① 需要手工标注的真值或奖励，训练阶段对任务分布有较高依赖；② 仅在同步、无分区网络中验证，缺乏异步或动态网络的鲁棒性；③ 置信度头与会话策略对模型架构有限制，需针对每个模型单独训练；④ 对抗攻击防御仍存在一定弱点，尤其是完全被篡改的情况。

---

## 145. When 10,000 Windows Are Not 10,000 Tests: Auditing Statistical Confidence in Sliding-Window Time-Series Classification

**arXiv ID:** 2609.30721 | [PDF](https://arxiv.org/pdf/2609.30721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 146. Learning Polarization Image Restoration with General Restoration Priors

**arXiv ID:** 2609.30728 | [PDF](https://arxiv.org/pdf/2609.30728v1)

**作者:** Chenggong Li `[一作]` (Central South University), Degui Yang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向多种退化（模糊、低照度、噪声、马赛克及其组合）的全流程极化图像恢复框架。

**💡 创新点**

创新点包括：① 发现归一化Stokes表示在极化恢复中最优；② 设计双分支网络，利用预训练的通用图像恢复模型（MoE扩展）对强度通道进行先验知识迁移；③ 引入双向Intensity‑Polarization Adaptive Transfer Module（IP‑ATM）实现跨域可学习特征选择与调制；④ 构建全新的PolarCDD数据集，涵盖11种复合退化，支持一体化研究。

**🔧 技术方法**

使用的技术：Normalized Stokes表示、U‑Net+NAFNet骨干、Mixture‑of‑Experts（MoE）、跨域特征调制（基于SFT/PixelAdaLN的缩放/平移）、零卷积调制、预训练通用恢复模型（如InstructIR）以及文本条件机制。

**📊 数据集**

使用的数据集：公开极化恢复数据集（PolDeblur、PLIE、LLCP、PIDSR 等）以及自建的 PolarCDD 数据集（包含 50k 训练图像、100 个测试图像，涵盖 11 种退化组合）。

**📈 对比分析**

与状态‑of‑the‑art 方法（通用 AIO 模型如InstructIR、MoCE‑IR、CDIR；以及专用极化恢复模型如PolDeblur、PLIE、PIDSR 等）进行对比。实验结果显示，本方法在 PSNR、SSIM 与 MAE 方面均显著优于所有基线，尤其在极化参数（DoLP、AoP）恢复上提升约 2–4 dB；在 PolarCDD 上平均 PSNR_I 37.38 dB、PSNR_p 33.19 dB，显著高于竞争者。

**⚠️ 局限性**

局限性：① 仍需依赖预训练强度先验，若强度通道退化极端难以迁移时效果下降；② 对极化参数的学习仍受限于归一化 Stokes 表示的数值范围，低 DoLP 区域可能出现误差；③ 训练与推理成本较高，尤其是 MoE 与跨域调制模块；④ 该框架主要在合成/实验室场景评估，尚未在大规模真实极化环境中充分验证。

---

## 147. POIL: Point-based One-Shot Imitation Learning with Stable Dynamical Systems

**arXiv ID:** 2609.30404 | [PDF](https://arxiv.org/pdf/2609.30404v1)

**作者:** Sang Min Kim `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于功能部件3D点集合的单次示范迁移与稳定闭环执行框架POIL，能够在未见物体、抓取姿态和目标几何变化下完成任务。

**💡 创新点**

将功能部件点作为统一表示同时实现轨迹迁移与闭环控制；将BCSDM从SE(3)扩展到点集合，获得几乎全局收敛；利用多模态LLM实现基于名称的部件定位，克服视觉特征在视角变化下的漂移。

**🔧 技术方法**

多模态LLM+Rex-Omni +SAM2进行部件检测与掩膜；MVTracker做多视角点跟踪；非刚性配准OAReg + TPS变形对轨迹进行自适应；Point-set BCSDM做闭环控制；RANSAC + 最小二乘求解转动扭矩。

**📊 数据集**

使用MuJoCo仿真与Franka Emika Panda实物实验；基准包含Point Policy（50 demo）和Instant Policy；在可视化评估中采用三类物体（杯子、平底锅、剪刀）在12个视角下测试。

**📈 对比分析**

与Point Policy和Instant Policy比较，POIL在Pick‑and‑Place Cube、Reshelving、Mug Insertion、Hanging Bag等四个任务中单示范实现100%成功率；在姿态、抓取、目标几何变化以及外部扰动下均保持高性能，而基线在远离训练分布时性能显著下降。

**⚠️ 局限性**

依赖物体与功能部件的查询、少量用户点击对应、部件几何相似性；对点跟踪高度依赖，遮挡、透明/镜面表面或极端视角下可能失效；仅支持单抓取，若无法找到可行抓取则失败；未处理非刚性或需要重新抓取的长序列任务。

---

## 148. SAGE: Source-Anchored Guidance via Frequency Equalization for Hierarchical RGB-T Alignment and Fusion

**arXiv ID:** 2609.30703 | [PDF](https://arxiv.org/pdf/2609.30703v1)

**作者:** Timing Li `[一作]`, Pengfei Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的频域框架 SAGE，用源锚定的频率均衡、层次对齐和引导子带融合实现弱配准 RGB‑T 图像的对齐与融合。

**💡 创新点**

创新点包括：① 源锚定频率均衡（SAFE）在保留源信息的同时提取结构和增益引导；② 层次频率协同对齐（HFCA）将低频全局仿射估计与高频残差细化结合；③ 引导子带融合（GSF）在频段级融合中利用对齐可靠性和源引导实现跨频段协同。

**🔧 技术方法**

技术手段包括 Haar 小波变换、可逆联合编码、频率均衡、低频仿射对齐、局部相关性细化、引导门控融合及逆小波重构。

**📊 数据集**

实验数据集：DroneVehicle、MFNet、RoadScene（含真实与合成对齐扰动）。

**📈 对比分析**

与 SuperFusion、ReCoNet、MURF 等八种 SOTA 方法对比，SAGE 在对齐指标（HD95、ASSD）和融合指标（AG、SF、SCD）上均取得领先或同等优异成绩，并在下游目标检测任务中提升召回率、精确率和 mAP。

**⚠️ 局限性**

局限性：仍依赖小波分解假设，处理极大或非刚性对齐时效果有限；模型虽算力适中，但 FLOPs 与参数仍高于部分轻量级方案，且未评估实时多帧场景或不同传感器的通用性。

---

## 149. ST-pRRTC: Parallel Space-Time RRT-C with Adaptive Goal-Time Forests

**arXiv ID:** 2609.30533 | [PDF](https://arxiv.org/pdf/2609.30533v1)

**作者:** Duo Zhang `[一作]` (Rutgers State University of New Jersey), Jingjin Yu `[通讯]` (Rutgers State University of New Jersey)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于GPU并行的时空 RRT‑Connect 运动规划器 ST‑pRRTC，用于已知障碍物轨迹下的多目标到达时间最优规划，并提出了区间根采样和根回收两种变体。

**💡 创新点**

通过共享前向树与自适应后向根森林的组合，实现连续到达时间探索；区间根保证概率完备与到达时间最优性；根回收策略在有限树容量下动态聚焦早期到达，提升有限预算性能。

**🔧 技术方法**

GPU 并行 RRT‑Connect（pRRTC）框架、动态碰撞检测、时间窗口引导的连续根采样、最大速度边缘扩展、根回收与根分组策略。

**📊 数据集**

Disc2D、Panda‑spheres、Tabletop 机器人重排三大模拟基准以及真实 UR5e 机器人与 Crazyflie 直升机的演示。

**📈 对比分析**

与 ST‑RRT*、SI‑RRT 及无回收版本在相同预算下对比，使用成功率、首解时间、最终到达时间、路径长度等指标；ST‑pRRTC 两种变体在所有基准上均实现更低的首解时间、更早的到达、较高的成功率（尤其是桌面任务），且在有限根数与大时间范围下根回收显著提升。

**⚠️ 局限性**

根回收缺乏理论完备性与最优性保证，可能错过可行到达时间；算法依赖已知、确定的障碍轨迹与开环规划，鲁棒性受限；在极大规划时域或极小根容量时，性能下降。

---

## 150. LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information

**arXiv ID:** 2609.30706 | [PDF](https://arxiv.org/pdf/2609.30706v1)

**作者:** Furkan Yilmaz `[一作]`, Muhammed Faruk Gozay `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在单通道决策模型基础上引入了可学习的价值信息（VOI）头，使模型能一次前向传播同时给出决策分布和每个待询问槽位的期望信息增益，进而动态决定是否提问及提问内容；

**💡 创新点**

创新点在于：① 用槽位块和VOI头一次性预测所有槽位的VOI，避免多次编码；② 通过规则定义的黄金决策、无标签VOI目标、双家族交叉校验以及Gini上限来训练可靠的VOI预测；③ 引入Gini阈值校正，使模型在分布外也能正确抑制不必要提问；

**🔧 技术方法**

技术包括：ModernBERT‑large编码器、选项评分头、VOI头（MLP + Gini上限）、零标注VOI目标回归、温度校准、单步决策与提问循环、训练分为热身与联合阶段；

**📊 数据集**

数据集涵盖：12个客服规则架构（8训练+4零射），21,601条训练示例（包含多种答案类型），19,000条通用单轮数据，27个公开数据集，ABCD真实对话918条，SGD 3,812条；

**📈 对比分析**

与Laya、Jev及多项公开基准对比，受控实验中决策准确率与Bayes上限相当，提问策略AUC与oracle相符；在真实对话中，在提问率大幅降低后准确率提升4.7点；在12个公开基准上，最终模型在7/12任务上优于Laya；

**⚠️ 局限性**

局限性包括：1) 受限于合成对话，真实用户交互未完全评估；2) 零射决策准确率仍低于上限，且分布外校准仍弱；3) 生成文本与真实语境可能偏差；4) 仅使用英文单一随机种子；5) 对比基准缺少慢速VOI或LLM代理等。

---

## 151. Combining General and Domain-Specific Pretext Tasks for Brain MR Image Segmentation

**arXiv ID:** 2609.30708 | [PDF](https://arxiv.org/pdf/2609.30708v1)

**作者:** Tasneem Nasser `[一作]` (University of Calgary), Naser El-Sheimy `[通讯]` (University of Calgary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在脑部磁共振成像（MRI）分割中，提出了一种多任务自监督预训练框架，将域特定任务（体素级脑龄预测）与通用任务（图像修复）联合学习，以提升对多种下游分割任务的迁移性能。

**💡 创新点**

创新点在于：①首次将体素级脑龄预测与图像修复共同作为自监督预训练任务；②证明多任务预训练能够在低样本环境下显著提高分割性能；③提供了一套系统的实验评估，涵盖年龄相关与非年龄相关的三类分割任务。

**🔧 技术方法**

使用的技术包括：SwinUNETR Transformer 结构作为骨干网络；自监督预训练任务为图像修复（coarse dropout+感知损失）和体素级脑龄预测（MAE损失）；多任务学习采用加权总损失；下游微调采用解码器重置+全网络微调策略。

**📊 数据集**

数据集方面：①多源T1w图像（包括CORR、IXI、ABIDE、OASIS、ADNI等）用于预训练；②CNS（T1w+FLAIR）用于MS预训练；下游任务分别使用：MSLesSeg（115病例），ISLES2026慢期脑卒中（649病例），Mindboggle-101（101病例）进行分割实验。

**📈 对比分析**

比较方法：将预训练模型（单任务图像修复、单任务脑龄预测、两者联合）与从零开始训练进行对比，评估Dice分数；在不同标签样本量（如11-58、10-400、10-65等）下重复实验。结果显示：多任务预训练在大多数低样本场景下显著优于单任务和随机初始化，尤其在MS（11-58）和脑卒中（10-150）任务中提升3-7个百分点；在大样本时差距缩小。

**⚠️ 局限性**

局限性：①对MS任务的预训练受限于单源T1w+FLAIR数据，缺乏多样性；②体素级脑龄预测在单独使用时对解剖分割无显著优势，说明单任务并非通用；③实验仅覆盖分割任务，未检验分类、预测等其他应用；④缺乏针对直接与脑龄相关的标注任务（如认知功能预测）进行验证。

---

## 152. MedTokenBudget: Lesion-Preserving Token Routing for Dermoscopic Image Classification

**arXiv ID:** 2609.30613 | [PDF](https://arxiv.org/pdf/2609.30613v1)

**作者:** Zhexiang Li `[一作]` `[通讯]` (University of California, Los Angeles), Zhexiang Li (University of California, Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在光学活检图像分类中，提出了一个后置基准的Token路由框架MedTokenBudget，通过学习一个多信号评分器（LATS）在保持固定Token预算的前提下，选择包含病灶区域的Token子集，以构建紧凑的病灶增强表示。

**💡 创新点**

创新点：① 引入病灶保留率（lesion retention rate）作为评价指标，明确衡量Token路由是否保留诊断证据；② 通过注意熵、特征范数和局部特征对比三种弱信号结合的学习评分器LATS；③ 在训练中加入预算学习曲线、分布正则化、注意力蒸馏和可用病灶掩模监督，显著提升低预算下的准确率与病灶保留率。

**🔧 技术方法**

技术手段包括：Vision Transformer（ViT）作为冻结基线；LATS评分器使用多层MLP；Top-K路由采用Gumbel Softmax实现可微分选择；预算调度使用余弦衰减；多任务损失包括分类、分布正则、病灶掩模监督与注意力蒸馏。

**📊 数据集**

主要使用ISIC 2019皮肤病变分类数据集，并利用ISIC 2018的病灶掩模做监督和评估；另外在Kvasir v2上做了单纯准确率验证。

**📈 对比分析**

与随机、NormBased、ToMe、AttentionEntropy、LocalContrast等七种Token选择策略对比；在低至中等预算（β=0.1–0.5）下，LATS在准确率上与全量Token相当甚至更好，同时病灶保留率提升2.9–1.7倍，证明其在保持诊断证据的同时保持了高性能。

**⚠️ 局限性**

局限性：① 仅在皮肤病变任务上验证，难以推广到无病灶掩模的其他医学图像；② 病灶掩模覆盖仅MEL/NV类别且样本量有限，可能偏高保留率；③ 采用后置路由不降低前向基准的推理延迟，未实现端到端加速；④ 对不同ViT基础模型与更细粒度的二维局部对比信号的探索尚未完成。

---

## 153. CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production

**arXiv ID:** 2609.30471 | [PDF](https://arxiv.org/pdf/2609.30471v1)

**作者:** Mukul Chhabra `[一作]` (Dell Technologies), Luigi Medrano `[通讯]` (Dell Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Cargo 框架，将检索到的参考答案重新解释为程序范例，并在评估时依据实时实例上下文进行三类断言状态（支持、矛盾、不可证伪）判断，以解决引用‑实例偏差问题。

**💡 创新点**

创新点在于把检索参考视为程序范例而非精确答案，构造基于实例的三类判定并加入检索置信门控，实现对实时交互的可选择性评估，同时通过构造性扰动测试集验证方法。

**🔧 技术方法**

使用 LLM‑as‑a‑judge、稠密检索与余弦相似度门控、上下文根植的结构化判定，以及置信门控与风险‑覆盖曲线分析。

**📊 数据集**

数据集包括生产中的企业技术支持多代理助手的 400 条跟踪样本，以及基于十个种子生成的 246 条构造性扰动实例（实体移植、矛盾注入、不可证伪、程序腐败、检索干扰）。

**📈 对比分析**

与传统直接参考评估相比，Cargo 将误判率从 100% 降至 0%，在保持对矛盾的 100% 召回的同时将 DI 提升至约 0.58；对程序腐败的召回仅约 20%，但整体成本‑覆盖优于固定门控，并在不同 LLM 模型上表现一致。

**⚠️ 局限性**

局限包括对程序腐败的识别不足、构造性扰动可能不代表真实流量、仅在单一企业支持域验证、对不可证伪断言的处理可能忽略部分伪造信息，以及门控对检索相似度的依赖导致的误判。

---

## 154. Parameters vs. Context: TRACE Fine-Tuning for Robust Retrieval-Augmented Generation

**arXiv ID:** 2609.30337 | [PDF](https://arxiv.org/pdf/2609.30337v1)

**作者:** Zhengchen Huang `[一作]` (LinYi University), Xing Wang `[通讯]` (LinYi University)

**通讯引用:** 5417 | [OpenAlex ID](https://openalex.org/A5100365483)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验中使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明新方法在XXX指标上优于传统方法。

**⚠️ 局限性**

限制在于XXX，例如数据集的规模、模型的复杂性等。

---

## 155. Fake News Theories: Harnessing Disciplinary Insights for Computational Modeling, Detection, and Explanation

**arXiv ID:** 2609.30427 | [PDF](https://arxiv.org/pdf/2609.30427v1)

**作者:** Zhaoyang Cao `[一作]` (Syracuse University), Reza Zafarani `[通讯]` (Syracuse University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于跨学科社会科学理论的可解释假新闻检测框架，将理论映射为可量化特征并用于机器学习

**💡 创新点**

首次系统整合传播、心理学、经济学等多领域理论为可解释特征，强调理论驱动与可解释性的统一

**🔧 技术方法**

使用词典、统计模型、实体网格、逻辑一致性推理、LLM提示等技术提取特征，并用XGBoost进行分类

**📊 数据集**

在ReCOVery、Fake_And_Real_News和GossipCop三大公开基准数据集上评估

**📈 对比分析**

与全词表TF‑IDF基线比较，理论特征模型在保持可解释性的同时取得AUC≈0.79、F1≈0.77的良好表现，单特征AUC仅50‑70%，多特征组合提升有限但稳定

**⚠️ 局限性**

受限于数据集差异、缺乏传播轨迹、对LLM提示的敏感性以及部分特征对特定数据集依赖强，导致跨数据集一致性不高

---

## 156. Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs

**arXiv ID:** 2609.30492 | [PDF](https://arxiv.org/pdf/2609.30492v1)

**作者:** Sang Bin Moon `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 634 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对角色设定进行集合层面的多样化，提出并评估了四种基于空间填充与前沿搜索的角色选择和生成方法，以提升大语言模型在创造性任务中的多样性和创造力。

**💡 创新点**

创新点在于将角色多样化视为集合级条件问题，构建了选择/生成与空间填充/前沿搜索的四维设计空间，并提出了全局最优的Coverage、Dispersion选择算法以及UC-MCMC与Evolutionary TextGrad生成方法；证明了角色几何分布能显著提升输出多样性。

**🔧 技术方法**

技术上使用了基于语义嵌入的距离度量（余弦、欧氏、马氏），整数线性规划与增量团搜索实现Coverage与Dispersion；Metropolis-within-Gibbs MCMC实现UC-MCMC；Evolutionary TextGrad结合自然语言梯度和密度评价进行角色进化；以及Gemma‑4、EmbeddingGemma、Qwen3.6等LLM作为生成与评估。

**📊 数据集**

数据集包括Alternative Uses Task（AUT）物体使用、Infinity‑Chat 对话、Divergent Association Task（DAT）等公开多样化评测集合；角色样本来源于 PersonaMem‑v2（PersonaHub 子集）。

**📈 对比分析**

通过与任务仅提示、随机角色、Coverage/Dispersion 选择、UC‑MCMC/Evolution 生成以及 Creativity‑enhanced、Creative、DMAD 等提示与推理基线对比，实验显示 Evolution 生成的角色在 AUT 上可将多样性提升 78.8%、原创性 26.1%、灵活性 49.5%、创造力 13.9%；在 Infinity‑Chat 和 DAT 上同样实现显著的同质性降低、灵活性提升，且与提示工程协同可进一步提高。

**⚠️ 局限性**

局限性包括：角色空间距离与响应空间距离并非一一对应，需经验验证；只在特定 LLM 与评估器上测试，缺乏广泛的用户实验；生成的角色对任务特定优化无考虑，可能在某些任务下效果受限；评估主要依赖 LLM‑as‑judge，未进行充分的人工评价。

---

## 157. What Will Remain Human in Software Architecture? A Focus Group Report

**arXiv ID:** 2609.30334 | [PDF](https://arxiv.org/pdf/2609.30334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 158. Conditional Predictive Sufficient Statistics for Visual Representation Learning

**arXiv ID:** 2609.30647 | [PDF](https://arxiv.org/pdf/2609.30647v1)

**作者:** Yuzhou Hong `[一作]` `[通讯]` (Zhejiang Sci-Tech University), Yuzhou Hong (Zhejiang Sci-Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证一种条件预测充分统计量（CPSS）框架，用来训练视觉编码器以保留可预测的共享因子信息并丢弃私有噪声；通过将下一嵌入的余弦损失解释为von Mises–Fisher分布的方向似然，说明了常数嵌入的最优性与梯度截断的必要性；实验证明在中间层能获得最优线性可读性，输出层因被迫贴近浅层目标而性能下降。

**💡 创新点**

核心创新在于：① 将预测信息量形式化为“条件预测充分统计量”，为自监督视觉学习提供新的理论目标；② 证明余弦损失相当于方向似然，揭示了梯度截断与常数解的关系；③ 通过层级探测提出“中间层读取”规则，解释了为何输出层往往不具备最佳下游可读性。

**🔧 技术方法**

使用小型因果Transformer架构（Patch‑embedding → 6层多头注意力 + MLP），优化器为AdamW，损失为归一化余弦相似度；实验对比了包含/不包含future shift、stop‑gradient、像素回归等控制变量。

**📊 数据集**

在两种公开数据集上验证：MNIST（28×28 灰度）和 CIFAR‑10（32×32 RGB），采用标准 50k/10k 切分，不做数据增强。

**📈 对比分析**

比较方法为在冻结模型的不同层（0即 patch 嵌入、各层输出）上训练线性分类器。MNIST 上 CPSS 中间层（第 2 层）达 87.3% 线性精度，输出层仅 77.2%；像素基线为 91.6%。CIFAR‑10 上 CPSS 中间层 32.2%，输出层 28.2%，像素基线 32.1%。控制实验显示移除 future shift 或 stop‑gradient 会导致精度下降或特征退化。

**⚠️ 局限性**

局限性包括：① 仅在短时间、无增广的实验下验证，未能与大规模对比学习或掩码自编码器竞争；② 理论假设未来条件独立（shared‑factor model）在真实图像上仅近似；③ 未证明梯度下降总能逃避常数解；④ 对目标嵌入的浅层假设限制了对更深语义空间的适用性；⑤ 只评估线性探测，未检验更复杂下游任务的性能。

---

## 159. GraspTwin: Zero-Shot Task-Oriented Grasp Optimization via a Digital Twin

**arXiv ID:** 2609.30543 | [PDF](https://arxiv.org/pdf/2609.30543v1)

**作者:** Daniel J. Evans `[一作]` (Virginia Tech), Dylan P. Losey `[通讯]` (Virginia Tech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于单帧 RGB‑D 图像构建数字孪生，利用 VLM 生成任务导向的抓取位置并在数字孪生中通过贝叶斯优化与域随机化进行梯度无关的局部优化，最终实现零样本、任务导向且物理可行的抓取。

**💡 创新点**

将大型模型的语义推理与基于仿真的物理优化相结合，提出 real‑to‑sim‑to‑real 框架；通过 VLM 提示得到语义抓取方案后在数字孪生中做局部优化，首次在一次试验内实现任务导向且可执行的抓取。

**🔧 技术方法**

数字孪生构建（SAM‑3D + Isaac Lab），VLM 任务提示与解析（类似 LLM + VLM），贝叶斯优化与 Thompson sampling，域随机化仿真，机器人逆运动学与控制。

**📊 数据集**

使用 45 个基于 SAM3D 生成的仿真场景，30 个标准任务（包含不同对象、光照、拥挤等）进行评估，并在 10 个真实世界任务上进行实测。

**📈 对比分析**

与 LERF‑TOGO、GraspMolmo 等基线对比；在仿真中 GraspTwin 的总体抓取成功率为 51.6%（相较 33.1% 的 GraspMolmo 提升 18.5%），在真实世界中实现 33% 的绝对提升，显著优于基线。

**⚠️ 局限性**

整体处理时间约 5 分钟，主要耗时在多批次物理仿真评估；数字孪生的精确度与域随机化的可靠性仍有限，需更高并行计算以缩短处理时间。

---

## 160. SoGuDiff: Socially Guided Diffusion for Steerable, Norm-Grounded Robot Navigation

**arXiv ID:** 2609.30560 | [PDF](https://arxiv.org/pdf/2609.30560v1)

**作者:** Christian Schaible `[一作]`, Stephen L. Smith `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于扩散模型的机器人导航框架SoGuDiff，能够在部署时通过四个可解释的社会风格轴（近距离保守、通行侧偏好、让路倾向、群体敬畏）实时调节导航策略，并通过可视化的可行性投影层保证动力学和碰撞安全。

**💡 创新点**

创新点包括：①仅使用单轴标注的演示数据即可在推理阶段通过按轴的分类器自由引导（CFG）实现多轴风格组合；②将社交行为与可行性投影分离，使扩散模型专注于多模态社会意图而不牺牲动态可行性；③通过结构化条件丢弃训练，单个模型即可在不同条件下生成对应的轨迹，从而实现多风格连续可控性。

**🔧 技术方法**

核心技术包括：条件扩散模型（DDPM/DDIM）配合Transformer编码器实现对场景与风格的嵌入；按轴分类器自由引导（CFG）实现风格组合；软约束最优控制（OCP）投影层通过acados求解以保证轨迹可行；以及在训练中使用结构化条件丢弃来实现多条件学习。

**📊 数据集**

使用了CrowdNav仿真环境中的随机人群场景、HuNavSim生成的演示数据，以及真实世界中的Clearpath Jackal机器人与YOLOv5n检测得到的行人信息，所有演示均来自单轴风格标注的规划器。

**📈 对比分析**

在500个随机人群场景和500个几何场景上与SFM、ORCA、RL、Transformer等多种基线进行对比。SoGuDiff在成功率、时间到达、路径长度、个人空间侵入率等指标上均达到或超过最优基线，并且通过风格轴的单轴和组合扫频展示了显著的效率-社交性权衡优势；在真实硬件上亦能实现不同风格的安全导航。

**⚠️ 局限性**

主要局限包括：投影层仅提供软约束且不保证绝对安全，常数速度估计可能导致碰撞；在高密度拥挤下过于保守的风格会导致停滞；多轴风格组合时某些轴可能占主导，导致预期风格被削弱；过大引导权重会削弱连续性并可能降低成功率。

---

## 161. From Bilateral Trade to Matching Markets: Sharp Gains from Trade

**arXiv ID:** 2609.30702 | [PDF](https://arxiv.org/pdf/2609.30702v1)

**作者:** Zhengyang Liu `[一作]` (Beijing Institute of Technology), Zihe Wang `[通讯]` (Renmin University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在满足贝叶斯激励兼容、临时个体理性以及预期预算平衡的匹配市场中，如何最大化交易收益并给出下界与上界；

**💡 创新点**

创新点在于提出了将双边交易的二次最佳性能转移到一般匹配市场的完全匹配（finite-support）归约，并在此基础上给出了 MHR 供给者和二元供给者的最优效率下界（≈0.7249、8/9、27/32 等），这些下界在双边交易中已达到极限；

**🔧 技术方法**

主要技术包括：双边到匹配的归约、cap‑monotone 规则构造、Lagrangian 曲线与虚拟值分析、卖家正则化、凸优化与极小极大定理、连续分布的指数比较与折扣点递推等；

**📊 数据集**

研究基于理论分析，无使用实测数据集；

**📈 对比分析**

方法通过解析推导，证明在满足独立型分布、向下封闭可行性约束下，所得比值优于先前已知的 1/2 下界，具体数值与已知下界相匹配或更好；

**⚠️ 局限性**

局限在于仅覆盖独立型、Borel 先验、向下封闭可行性，并未考虑双边双方同时有多于两种类型的情况，且对非独立或动态市场的适用性尚未探讨。

---

## 162. LLPR: Location-aware learning and physics-based reconstruction for raindrop removal from a single image

**arXiv ID:** 2609.30758 | [PDF](https://arxiv.org/pdf/2609.30758v1)

**作者:** Zewei He `[一作]` (Zhejiang University), Zhe-Ming Lu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了LLPR框架，利用位置感知学习与物理模型重建实现单幅图像雨滴去除。

**💡 创新点**

创新点在于引入仅在训练阶段使用的定位感知分支、基于物理模型的重建机制以及循环损失，显著提升性能且推理无额外成本。

**🔧 技术方法**

采用U‑Net结构的Transformer骨干（如DRSformer/Restormer），结合物理方程 A⊙R 与循环损失，并辅以位置感知分支。

**📊 数据集**

训练使用合成雨滴数据集（约1100对），评估在Test‑a、Test‑b以及自建的野生雨滴测试集Test‑wild上。

**📈 对比分析**

与七类SOTA方法比较，LLPR(DRSformer)和LLPR(Restormer)在PSNR/SSIM上均位列第一，参数量和FLOPs最低，并在用户主观评价和无参考指标上优于其他方法。

**⚠️ 局限性**

局限在于仍依赖合成训练数据，且物理模型假设简化，难以处理极端雨滴重叠或复杂光照变化。

---

## 163. Werracle: Sub-Cent Intra-Block AI Reflex Oracles and Flash-Loan Circuit Breakers for EVM Smart Contracts

**arXiv ID:** 2609.30719 | [PDF](https://arxiv.org/pdf/2609.30719v1)

**作者:** Volkan Dağlı `[一作]` (ITOUCH Bilişim Sistemleri Ltd. Şti.), Dağhan Dağlı `[通讯]` (Toros Science High School)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在以太坊EVM上实现了一种零存储、单存储槽即可执行的AI决策预言机Werracle，能够在单个交易中完成子毫秒级的推理和决策。

**💡 创新点**

核心创新是用基于Mandelbrot逃逸动力学的程序化非线性决策边界替代传统的多余量级神经网络权重，并通过Q16.16定点算术在纯Solidity字节码中实现。

**🔧 技术方法**

技术手段包括：64位Q16.16定点算术、单槽位位级存储压缩、16点Pareto微网格逃逸检测、全EVM字节码实现、以及Uniswap v4动态费率挂钩。

**📊 数据集**

使用的主要数据集为合成的交易特征向量（成交量、tick速度、地址等）以及Uniswap v4订单簿波动测试数据，全部由内部的1000条加密封装测试向量验证。

**📈 对比分析**

与传统Web2预言机和ZK‑ML（如EZKL）对比，Werracle在推理延迟从秒级缩短到<1 ms、验证gas从数十万降至21,438 gas，成本降低约18×，并实现了原子级的区块内回退/动态费率调整。

**⚠️ 局限性**

局限性在于：决策功能受限于固定的Mandelbrot参数空间，缺乏可调的多层神经网络灵活性；对复杂多维特征映射的表达能力有限，且目前仅在模拟和Uniswap v4场景中验证，尚未证明在更广泛DeFi协议中的通用性。

---

## 164. Learning to Bias: Machine Learning-Enhanced Particle Filters

**arXiv ID:** 2609.30498 | [PDF](https://arxiv.org/pdf/2609.30498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 165. Silent Success: A Release Gate That Passed on Checks It Never Ran, and Eight More

**arXiv ID:** 2609.30307 | [PDF](https://arxiv.org/pdf/2609.30307v1)

**作者:** Dong Hyeon Jeon `[一作]` `[通讯]`, Dong Hyeon Jeon

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文归纳了九个软件系统中“描述作为证据”这一失效模式，阐释其产生机制并提出一套可复现的修正方法；

**💡 创新点**

创新点在于首次系统化描述与状态之间的“弱→强”提升结构，并证明七个案例可通过单条查询、命令或比较即可被捕获；

**🔧 技术方法**

主要技术手段为：对门控决策逻辑的重构、引入第三状态“无法确定”、对数据集覆盖率与计数进行显式比较、以及对历史记录、配置、声明等进行完整性检查；

**📊 数据集**

使用的“数据集”为研究系统的实时部署数据快照（约12万行时间序列表和若干配置信息），并对公开项目（LiteLLM、vLLM）中的相关提交进行复现；

**📈 对比分析**

通过与原始门控逻辑及公开项目的对照实验，展示修正后门控在两周内由绿变红再恢复绿的完整过程，且在所有修正点的回归测试中均通过，验证了方法的可行性与高效性；

**⚠️ 局限性**

局限性包括：案例数量有限、主要来自单一项目团队，缺乏对普遍性和泛化的统计；改进措施需在更大规模的多系统实验中验证，且对部署环境的匿名化限制了外部复现的可能性。

---

## 166. Timo: $\textbf{T}$aming Mult$\textbf{i}$modal Diffusion Transformer for Human $\textbf{Mo}$tion Generation

**arXiv ID:** 2609.30761 | [PDF](https://arxiv.org/pdf/2609.30761v1)

**作者:** Zhao Wang `[一作]` (LimX Dynamics), Tao Yu `[通讯]` (LimX Dynamics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向人体动作生成的运动学感知多模态扩散 Transformer（MMDiT）框架，利用双向文本-动作注意力、流匹配、几何与旋转运动学监督，以及两阶段课程学习，实现文本与动作的精细对齐与时间连贯性。

**💡 创新点**

创新点包括：①全共享的双向多模态注意力，使文本与动作能够相互更新；②在流匹配的基础上加入几何和旋转运动学损失，直接监督关节旋转及其时间变化；③两阶段课程训练，从粗泛化动作学习到细粒度文本对齐；④构建跨六大公开数据集的统一评估基准，提供多维度评测。

**🔧 技术方法**

核心技术为：多模态扩散 Transformer（MMDiT）与流匹配（rectified flow）；共享注意力模块；几何损失（姿态重建、根部对齐）和旋转运动学损失（角速度、加速度一致性）；两阶段课程学习策略；UniPC 采样与classifier-free guidance。

**📊 数据集**

使用了 HumanML3D、BABEL、AIST++、GRAB、PerMo、BONES 等六个公开动作数据集，并结合自研光学捕捉数据（约1,983小时）进行训练；评测集共计40,025个保留片段。

**📈 对比分析**

与 MotionMillion、HY‑Motion、GENMO、Kimodo 等四种公开系统在同一评测基准下对比，平均得分从 61.5 提升至 86.6，40.8% 相对提升；在文本‑动作匹配、自然度、多样性、平滑度等维度均超过对手；仅在可碰撞性（Plausibility）维度略逊 Kimodo，表明几何碰撞仍需改进。

**⚠️ 局限性**

局限性包括：①可碰撞性得分仍低，生成动作在碰撞检测上不如 Kimodo；②在真实机器人执行时需额外的重定向与控制步骤，无法直接保证动态稳定；③部分数据来源为私有光学捕捉，导致完整训练流程不可完全复现；④生成动作可能产生误导性合成影像，需谨慎使用。

---

## 167. A Framework for Identifying, Categorizing, and Explaining Bias in AI-Generated Code

**arXiv ID:** 2609.30642 | [PDF](https://arxiv.org/pdf/2609.30642v1)

**作者:** Manaal Basha `[一作]` (University of British Columbia), Gema Rodriguez-Perez `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对AI生成的Python代码进行细粒度偏见标注，构建了多标签偏见数据集，并利用大语言模型的in‑context学习（ICL）对代码偏见进行识别与解释。

**💡 创新点**

创新点包括：①提出了基于受保护属性、关联类型和偏见类型的多维度偏见分类与标准化解释框架；②首次将ICL应用于后置偏见审计；③系统评估LLM生成的偏见解释与专家注释的一致性。

**🔧 技术方法**

采用的技术主要是大语言模型（Gemini、Qwen3‑Coder、DeepSeek‑R1、Phi‑4）进行ICL推理，利用结构化JSON输出和Ratcliff/Obershelp相似度计算评估解释质量。

**📊 数据集**

数据集来自Huang等人扩展的Python代码生成集合，共7,840条，其中784条经过人工多标签标注形成基准。

**📈 对比分析**

比较方法是对不同ICL配置与模型进行多次评测，Gemini在偏见类型识别上的准确率≈90%、召回≥95%，解释相似度≈80%；开源模型Qwen3‑Coder的准确率≈83%、召回≈87%，表明ICL能在无微调的情况下实现高性能。

**⚠️ 局限性**

局限性包括：可能存在数据泄露导致记忆化结果；代理偏见检测表现低；仅覆盖Python函数，难以推广至更大规模或其他语言；解释质量评估依赖词法相似度，可能低估语义一致性。

---

## 168. Too Late to Slash: Coordinating a Risk-Free Equivocation Attack

**arXiv ID:** 2609.30509 | [PDF](https://arxiv.org/pdf/2609.30509v1)

**作者:** Hao Chung `[一作]` (Layerzero Labs), Chen-Da Liu-Zhang `[通讯]` (Lucerne University of Applied Sciences and Arts)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并证明了一种风险免费、协作的攻击策略，使得在算法性削权（slashing）下，理性验证者能够在不失去质押的前提下联合实施双重签名攻击，从而削弱了以质押为保障的安全性。

**💡 创新点**

创新点在于：
1) 对削权机制给出的直观安全理由进行形式化挑战；
2) 引入“垄断阈值”概念，证明只要验证者集体权重达到该阈值，即可实现对冲裁决的完全屏蔽；
3) 设计多种协同协议（包括带奖励、抵押、匿名注册等），展示即便在更复杂的削权规则下，攻击仍能成为纳什均衡；
4) 提出了“诚实质押安全”属性并在匿名注册协议中利用它保障诚实验证者不被误判。

**🔧 技术方法**

技术手段主要是：
- 基于博弈论的形式化建模和纳什均衡证明；
- 对区块链共识与削权协议进行抽象建模（包括最终性、验证与执行流程）；
- 设计可编程智能合约（协调合约、抵押合约、匿名合约）来实现验证者的协作与质押管理；
- 证明“垄断阈值”下的等价性与可执行性。

**📊 数据集**

该工作不依赖任何数据集；所有结果均为理论分析与形式化证明。

**📈 对比分析**

由于是纯理论工作，没有实验或基准测试，亦未与其他方法直接比较；作者通过证明纳什均衡与攻击收益，展示了在理性玩家条件下攻击的优势。

**⚠️ 局限性**

局限性包括：
- 需要理想化的网络与共识模型（同步/部分同步均可）；
- 攻击需要验证者集体权重达到垄断阈值，实际区块链中难以实现；
- 假设所有攻击者都完全理性且能完美协同，忽视了实际中出现的策略冲突、随机性与惩罚执行延迟；
- 未考虑外部治理或法院等现实中可能介入的削权执行机制，因而对实际安全性的直接结论有限。

---

## 169. Atlases Are Already Inside: Recovering Population Templates from Pretrained Diffusion Models

**arXiv ID:** 2609.30566 | [PDF](https://arxiv.org/pdf/2609.30566v1)

**作者:** Jian Shi `[一作]` (King Abdullah University of Science and Technology), Peter Wonka `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种在已训练扩散模型推断阶段使用后验均值迭代的采样策略，使所有随机种子在多模态数据（脑MRI、胸X、脸部、3D形状等）中收敛到同一图像，即提取人群的中心模板（Intrinsic Atlas），不需要额外训练或配准步骤。

**💡 创新点**

创新点包括：
- 通过去噪器的后验均值迭代实现“canonical convergence”，在所有种子下自动聚焦到中心结构；
- 无需配准模型即可得到多域、多模态的高质量中心模板；
- 通过条件化（如年龄）生成任意属性下的模板族，实现按需构建。

**🔧 技术方法**

主要技术：扩散概率模型（DDPM）、后验均值迭代（无噪声采样）、Tweedie公式、条件化扩散模型、注册评估框架（ANTs、VoxelMorph、MultiMorph 等）、统计指标（Dice、PCK、IoU、中心性、变形量等）。

**📊 数据集**

使用的数据集包括：脑T1 MRI（IBSR18、Mindboggle101、ABIDE-I、IXI、OASIS‑1）；2D 眼底、胸X射线、脸部（CelebA、FFHQ）；3D 形状（ModelNet飞机/椅子、KeypointNet车/椅子）；以及3,796种字体的字母“A”。

**📈 对比分析**

与经典模板构建、学习型模板、深度配准方法比较。Intrinsic Atlas 在所有评估数据集上获得最优或第二优（Dice、PCK、IoU、中心性、变形量等），在年龄匹配的模板上注册误差更小，证明其更具中心性和实用性。

**⚠️ 局限性**

局限性：
- 当人群缺乏单一中心或存在多个子中心时，收敛会产生多张模板或模糊图像；
- 对分辨率/尺度敏感，低分辨率下可能失去结构一致性；
- 仅适用于已训练并覆盖足够样本的扩散模型；
- 对不具备明显共性的人群（如随机构成）无法得到清晰模板。

---

## 170. AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework

**arXiv ID:** 2609.30541 | [PDF](https://arxiv.org/pdf/2609.30541v1)

**作者:** Aparajith Chandran `[一作]` (Amazon), Florian Hottier `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在亚马逊书籍推荐系统中，将 AutoResearch（LLM 驱动的自动化研究）扩展到生产规模，构建了三代理框架以自动化代码级实验。

**💡 创新点**

提出了五种生产规模失败模式及其对应的三原则（prevent、persist、redirect）设计，并证明了其成本依赖性，展示了在高成本实验中预执行检查、跨作业持久化和停滞重定向的必要性。

**🔧 技术方法**

使用 Claude LLM（Sonnet/Opus）进行自动代码生成，搭配预执行的 Code Fixer、Criticizer、跨作业 S3 持久化、滑动窗口上下文等技术；系统 A 进行全代码级迭代，系统 B 进行超参数与架构级迭代。

**📊 数据集**

使用 Amazon 的书籍共购行为图、句子转换器文本特征以及购买日志等数据，训练并评估 Recall@6 或加权一致性指标。

**📈 对比分析**

通过与手工调优基线对比，系统 A 的 Recall@6 从 3.79% 提升至 6.90%（1.82×），系统 B 的加权一致性从 0.348 提升至 0.735（2.11×），并在多日期评估中保持提升；同时实现了 5.8× 的覆盖率扩展。

**⚠️ 局限性**

实验仅在单一域和单一组织内完成，组件消融是观察性而非控制实验，评估指标未直接验证在线效果，且未解决 metric fixation 的元学习问题。

---

## 171. Staged Depth Training: A Representation Curriculum for PINNs

**arXiv ID:** 2609.30299 | [PDF](https://arxiv.org/pdf/2609.30299v1)

**作者:** Kejia Zhang `[一作]` (University of Maryland), Haizhao Yang `[通讯]` (University of Maryland)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5079602544)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种名为 Staged Depth Training (SDT) 的分阶段训练策略，使 PINN 的隐藏表示先在浅层网络中通过临时物理监督学习，然后冻结并在后续阶段继续加深，最终再联合微调。

**💡 创新点**

创新点在于将隐藏表示视为可独立学习、可迁移、可逐步细化的对象，并通过临时物理头、表示转移和冻结机制实现“表示课程”，显著提升 PINN 性能。

**🔧 技术方法**

技术包括：阶段化训练、临时物理监督头、前缀冻结与转移、分阶段学习率与步长、以及对多种 PINN 架构（MLP、ResNet、PirateNet）的兼容实现。

**📊 数据集**

使用了 PINNacle 框架下的 20 个默认前向 PDE 问题（含 Poisson、Navier–Stokes 等），并在三种 backbone 上进行实验。

**📈 对比分析**

与标准端到端训练进行同预算比较，SDT 在 58 个非平局单元中获得 53 次更低 L2 误差，平均几何平均误差降低 28–33%，并在 Poisson–Boltzmann 2D 任务中提升深度缩放指数。

**⚠️ 局限性**

局限性包括：需要可拆分隐藏层的网络结构、对训练阶段划分和学习率等超参数敏感、实验规模受限于 PINNacle 任务，且对极端高阶 PDE 的自动微分成本未做深入评估。

---

## 172. MR. POP: Multi-Robot Parallel Optimizing Planner for Almost-Surely Asymptotically Optimal Planning

**arXiv ID:** 2609.30644 | [PDF](https://arxiv.org/pdf/2609.30644v1)

**作者:** Chih H. Huang `[一作]` (Columbia University), Zachary Kingston `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于元算法和AO‑x的多机器人并行运动规划器，并在GPU上实现大规模并行化，能够在几秒内求解大规模多机械臂任务。

**💡 创新点**

创新点在于将元算法与AO‑x相结合，同时利用GPU多线程并行实现图构建、树搜索、最近邻查询和碰撞检测，显著提升求解速度并实现100%解题率；此外将生成的全局最优路径作为种子，显著提高后续局部优化器的成功率。

**🔧 技术方法**

主要技术包括RRT‑Connect、AO‑x（自适应迭代优化）、元算法（增量成本边界搜索）、PRM图构建、并行最近邻搜索、并行碰撞检测、信息采样、四维线程分配等；实现依赖Intel Core Ultra 9 24核CPU和GeForce RTX 5090 GPU。

**📊 数据集**

实验数据集：四臂Franka bin‑packing（28个实例）与五臂Franka shelf‑reaching（35个实例）作为多机器人基准；单机器人实验使用MotionBenchMaker中的Fetch机器人（7个场景各100个实例）。

**📈 对比分析**

与多种基线比较（RRT‑Connect、Bi‑RRT*、批量信息树、Fast Marching Tree等），在给定时间预算内实现了100%解题率，比最快基线快2.5×~5.1×；在路径优化阶段收敛速度更快、解的质量更好；当用作局部优化器的种子时，成功率从4%提升至72%。

**⚠️ 局限性**

局限性：依赖GPU并行加速，CPU单线程性能受限；对极大规模机器人团队的扩展性尚待验证；理论证明基于鲁棒最优假设，在非理想环境下可能不完全成立。

---

## 173. Geometric Feature Learning for Functional Data Valued on the Symmetric Positive Definite Manifold

**arXiv ID:** 2609.30487 | [PDF](https://arxiv.org/pdf/2609.30487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 174. Design and Characterization of a Variable-Length Continuum Mechanism with Force Locking

**arXiv ID:** 2609.30759 | [PDF](https://arxiv.org/pdf/2609.30759v1)

**作者:** Katelyn King `[一作]` (Stanford University), Allison M. Okamura `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种杆驱动、可变长度、力锁定的连续螺旋结构的柔性机械手，并通过实验验证其在柔性与刚性两种工作模式下的弯曲范围、轴向与弯曲刚度。

**💡 创新点**

首次将力锁定机制与连续螺旋骨架相结合，实现同一装置在同一空间内既能实现高柔性导航，又能通过拉伸/压缩实现高刚度，并实现可微型化。

**🔧 技术方法**

使用杆驱动的推拉装置、3D 打印/光固化工件、六轴力/扭矩传感器、步进驱动的主轴以及 OpenCV 姿态追踪。

**📊 数据集**

实验样本为三种不同螺旋单元数（T = 3、4、5）的原型，其长度、弯曲角度和力/位置数据由传感器记录。

**📈 对比分析**

通过拉伸/压缩曲线评估轴向刚度，弯曲路径比较理论与实验误差，得到 180° 弯曲角度、平均末端定位误差 <10%，轴向刚度随 T 增大显著降低，力锁定状态下弯曲刚度提升可达 0.56 的无量纲增益。

**⚠️ 局限性**

局限在于刚性工作区间较小、只能单平面弯曲、存在摩擦导致的滞后、轴向与曲率耦合导致刚度不均匀，以及微型化时材料与摩擦问题。

---

## 175. Steering Versus Teleporting in Mobile Virtual Reality

**arXiv ID:** 2609.30620 | [PDF](https://arxiv.org/pdf/2609.30620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 176. Fixed Points Without Fixed Diffusion: Implicit Neural Sheaves for Convergent Test-Time Computation

**arXiv ID:** 2609.30277 | [PDF](https://arxiv.org/pdf/2609.30277v1)

**作者:** Rémi Bourgerie `[一作]`, Viktoria Fodor `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1255 | [OpenAlex ID](https://openalex.org/A5075982238)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种基于自适应神经层束的隐式图神经网络SheafDEQ，通过可学习的矩阵化边缘传递实现稳定的平衡点。

**💡 创新点**

通过状态归一化实现尺度不变的自适应层束推送，获得在无谱半径或正则化约束下的唯一平衡点收敛，并在异步和带延迟通信环境下保持收敛性。

**🔧 技术方法**

采用子齐性深度平衡（SubDEQ）框架，结合可学习的矩阵层束映射、tanh+偏移的非线性，以及Thompson度量的合同性证明。

**📊 数据集**

在合成分布式推理任务（Chains、Counting、Sums、MNIST Terrain、Coordinates）以及噪声社区检测（三社区图）上评估。

**📈 对比分析**

与固定传播隐式基线（APPNP、IGNN、EIGNN、EnergyGNN）以及有限深度自适应模型（GraphAttnDEQ、Loop Transformer、DNSD）进行对比，SheafDEQ在跨社区连接增强时表现最好，分布式任务上优于固定传播基线，预测准确率略低于部分自适应模型，但收敛稳定性更好。

**⚠️ 局限性**

对极端初始化收敛缓慢，实验仅覆盖合成或转导任务，未在大型真实图上验证可扩展异步推理，缺乏更紧凑的收敛理论。

---

## 177. PixSim: a calibrated open-source simulator of instant-payment fraud, recovery and interdiction under analyst capacity constraints

**arXiv ID:** 2609.30684 | [PDF](https://arxiv.org/pdf/2609.30684v1)

**作者:** Bashir Zeimarani `[一作]` (Instituto CERTI Amazônia), Carlos Maurício Serodio Figueiredo `[通讯]` (Universidade do Estado do Amazonas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了PixSim模拟器，模拟巴西Pix即时支付系统的不可逆结算、MED恢复机制、监管持留窗口以及有限分析师队列等要素；并将其校准至巴西中央银行公开数据。

**💡 创新点**

首次公开集成恢复机制、转账路径、容量受限审核的模拟环境，能够在统一框架下评估不同欺诈拦截策略；通过参数校准实现与真实恢复率的高精度匹配，并揭示容量对策略优劣的关键影响。

**🔧 技术方法**

离散事件模拟、参数校准与验证、梯度提升评分器、分析师队列模型、实验基准与统计对比。

**📊 数据集**

巴西中央银行公开的Pix交易量、交易类型、DICT账号信息、MED争议统计（受争议、已接受、已退回等）以及时间分布数据。

**📈 对比分析**

在多场景（不同欺诈组合、容量水平、DICT信息缺失、跟踪机制投射、欺诈强度变化）下对四种参考策略（Pass All、阈值、阈值+阻断、容量感知）进行成对对比，评估损失值、误阻率、恢复率等指标；结果显示：容量充分时阈值+阻断策略损失最小，容量不足时优势消失；恢复率受转账传播速度限制，跟踪机制可显著提升恢复率。

**⚠️ 局限性**

模型未涵盖接收端检测、标记撤销、真实欺诈分布差异、队列服务时间与审核准确率的实际分布、单一欺诈率假设、仅评估全量部署场景，且2026年窗口的数据未能重现，导致对实际系统表现的预测有限。

---

## 178. From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning

**arXiv ID:** 2609.30391 | [PDF](https://arxiv.org/pdf/2609.30391v1)

**作者:** Yichen Lin `[一作]` (Shanghai Jiao Tong University), Tao Yao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在离线强化学习的上下文学习框架中，作者提出了 Q-Target Pretrained Transformers (QTPT)，通过用 Bellman 风格的 Q‑target 目标替代传统的行为克隆监督，训练 Transformer 以预测上下文条件下的动作价值，从而在面对弱或次优离线数据时提升鲁棒性。

**💡 创新点**

创新点主要包括：①将 Q‑learning 的 TD 目标直接嵌入 Transformer 预训练，形成“Q‑target 预训练”范式；②在理论上分离样本偏差与模型偏差，给出线性 bandit 与一般 MDP 的子最优性上界；③提出基于数据覆盖度的判定标准，说明在何种数据条件下 QTPT 能显著优于监督预训练。

**🔧 技术方法**

技术手段：Transformer（因果解码器）作为序列模型；Bellman / TD 目标与 bootstrap；蒙特卡洛与 TD 目标的 ablation；软最大/双重目标等稳健备份；多种 Transformer 架构（GPT‑2、Llama、Qwen2）与 CQL 等正则化；在离线数据上进行预训练，测试时直接利用固定参数执行贪婪选择。

**📊 数据集**

使用的数据集包括：1) 合成控制实验的随机与 LinUCB 收集的线性 bandit 与 MDP；2) Darkroom、Dark Key‑to‑door、Miniworld 等上下文 MDP；3) D4RL 公开基准（Kitchen、AntMaze）；4) GSM8K 数学推理数据；5) 额外的 Meta‑RL 与不同行为数据覆盖度的基准。

**📈 对比分析**

与监督预训练（SPT）、Q‑SFT、Q‑DT、CQL、IQL、RL^2、PEARL 等方法进行对比。实验显示：在随机或次优数据下 QTPT 的子最优性显著优于 SPT，且在 Darkroom、Miniworld、D4RL Kitchen/AntMaze 中得到更高的平均奖励或成功率；在 GSM8K 任务中相较行为克隆提升约 1.8% 的准确率。实验强调相同架构、相同数据集的直接对比，强调鲁棒性而非绝对 SOTA。

**⚠️ 局限性**

局限性：①理论保证依赖覆盖度与行为策略支持，低覆盖度数据无法保证；②D4RL 实验仅在匹配的 Transformer 协议下完成，未与专门调优的离线 RL 方案对齐；③仅使用单步 TD 备份，未探索多步或自回归等更复杂的 Bellman 形式；④在极度稀疏奖励环境下，Bootstrap 监督仍可能不充分；⑤实验结果的统计显著性未完全评估，主要基于平均奖励或成功率。

---

## 179. Reliability-aware Cross-sample Enhancement for Robust Multimodal Sentiment Analysis

**arXiv ID:** 2609.30470 | [PDF](https://arxiv.org/pdf/2609.30470v1)

**作者:** Menghua Jiang `[一作]` (Sun Yat-sen University), Sijie Mai `[通讯]` (South China Normal University)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5010270301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了RCE框架，统一解决多模态情感分析中的噪声、缺失模态等实际场景；

**💡 创新点**

创新点包括①使用自适应vMF变分信息瓶颈对模态不确定性建模并实现质量感知压缩；②利用可靠性感知跨样本检索高置信度、语义一致邻居进行增强；③构建超模态生成与多层可靠性融合，实现跨模态信息充分融合；

**🔧 技术方法**

采用vMF变分信息瓶颈、Perceiver‑style超模态生成、记忆池检索、可靠性感知多层融合、Transformer编码等技术；

**📊 数据集**

实验数据集包括CMU‑MOSI、CMU‑MOSEI（情感分析）以及UR‑FUNNY、MUStARD（幽默与讽刺检测）；

**📈 对比分析**

与DecAlign、MODS、PSA‑MF等全模态基线以及C‑MIB、OMIB、CyIN、HME、QMF等专用方法比较，RCE在全模态、噪声、缺失模态以及幽默/讽刺任务上均实现了显著提升（Acc7、MAE、准确率均优于或匹配最优基线）；

**⚠️ 局限性**

局限性为模型参数量较大、引入额外超参数和记忆池，需进行模型规模和参数调优，未来需探索轻量化与自适应方案。

---

## 180. Spanning Trees with Many Leaves in Graphs of Minimum Degree at Least 7

**arXiv ID:** 2609.30354 | [PDF](https://arxiv.org/pdf/2609.30354v1)

**作者:** Sogol Jahanbekam `[一作]` `[通讯]` (San Jos'e State University), Sogol Jahanbekam (San Jos'e State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在每个连通 n 点、最小度至少 7 的图中构造一棵叶数不少于 0.5455 n 的生成树，给出了多项式时间算法；

**💡 创新点**

提出了一个通用的递归框架，对所有 δ≥7 给出显式叶数下界，首次把 δ=7 的下界提升到 0.5455 n；

**🔧 技术方法**

主要技术是树的扩展策略与递推分析，结合严格整数运算求解叶子数递归；

**📊 数据集**

无实验数据集，全部为理论证明与严格计算；

**📈 对比分析**

与以往仅有的 11/21≈0.5238 n 下界相比，δ=8、9、10 时分别达到 0.5850、0.6151、0.6413 的叶子比例，性能显著提升；

**⚠️ 局限性**

仍未达到 Linial 猜想的上限，且递推式在 δ≤6 时不优，算法与理论仍有改进空间。

---

## 181. DanLing NestedTensor: Composable Multi-Ragged Tensors for Deep Learning

**arXiv ID:** 2609.30379 | [PDF](https://arxiv.org/pdf/2609.30379v1)

**作者:** Zhiyuan Chen `[一作]` `[通讯]` (DanLing Team), Zhiyuan Chen (DanLing Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 NestedTensor，一种在 PyTorch 中将多维 ragged（可变尺寸）结构内嵌于张量本身的抽象，使得打包、广播、特征变换和归约等操作能够自动维护样本边界和逻辑轴；在整个前向/后向计算、自动微分和编译执行过程中都保持该结构。

**💡 创新点**

创新点在于：①把 ragged 轴信息作为张量内部状态携带，避免在模型代码中手动管理偏移；②支持多维非领先 ragged 轴，结构感知算子能在广播、变换、归约时保持并更新结构；③跨编译边界保持结构，允许在 eager 与 Inductor 编译之间无缝切换；④通过分段调用和索引减少 padding 计算，显著提升速度和内存利用。

**🔧 技术方法**

实现技术包括：NestedTensor 包装器；通过 __torch_function__ 与 __torch_dispatch__ 为 PyTorch 高层函数和 ATen 操作注册结构感知处理器；利用 FlashAttention/FlexAttention 等 packed kernels；使用 PyTorch Inductor 进行动态尺寸编译；自动微分桥接以保持梯度连通；以及分段调用与索引逻辑以维护 ragged 结构。

**📊 数据集**

实验数据集涵盖自然语言与计算机视觉任务：BERT（IMDB）、GPT‑2（WikiText‑103）、Transformer（WMT14）、ViT（ImageNet‑1K）、DETR‑R50（COCO）、FCN‑ResNet（ADE20K）等，覆盖不同规模与变形特性。

**📈 对比分析**

对比方法：将 NestedTensor 与传统全填充（padded）、手动打包（explicit‑packed）以及 PyTorch 原生 torch.jagged 进行对比；在 NVIDIA A100 上测得：在四个 BERT 规模下 eager 速度提升 1.3‑1.5 倍，compiled 1.8‑2.0 倍；四个 FCN backbone 的 eager 速度提升 1.2‑1.4 倍；Pairformer 风格工作负载 eager 速度提升 1.5‑2.0 倍；内存占用相比填充减少 20‑40%。

**⚠️ 局限性**

局限性：编译器对 ragged 轴数量与形状的支持仍有限，极端不均衡批次可能无法完全消除冗余；并非所有 PyTorch 操作已实现结构感知，需要在模型中显式使用 NestedTensor；在某些场景下分段调用与索引的开销可能抵消部分速度提升。

---

## 182. Amplify What You Gaze At: Target Saliency Boosting in Text-to-Image Generation

**arXiv ID:** 2609.30733 | [PDF](https://arxiv.org/pdf/2609.30733v1)

**作者:** Shengqi Dang `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出目标视觉显著性提升任务，利用可学习的标记符号在文本提示中直接控制目标对象在文本‑图像生成中的视觉显著性，且不需要任何显著性图或视觉先验。

**💡 创新点**

创新点包括：①将显著性视为相对属性，使用双向标记符号（强调/抑制）在提示中显式标注目标与非目标；②设计了显著性先验标记激活（SPMA）机制，在训练时根据对象的相对显著性随机激活标记，从而学习到鲁棒的显著性知识；③构建了自动化的显著性‑语义数据集，使用LLM生成提示、SAM3进行分割、AAM进行显著性映射，自动为每个对象分配显著性分数。

**🔧 技术方法**

采用基于扩散的文本‑图像模型FLUX.1‑dev，使用标记符号嵌入、流匹配训练目标、SAM3进行对象分割、AAM生成显著性图、LLM生成提示与目标识别；训练时仅微调四个标记符号的嵌入，保持模型其它参数冻结。

**📊 数据集**

构建了约1637张图像‑提示对的数据集，每对包含对象级显著性分数；此外创建了432个目标显著性提升测试集；在实验中还使用标准显著性评估指标（Target‑NSS）和图像质量/语义相似度指标（CLIPScore、CLIPIQA）。

**📈 对比分析**

与Prompt Engineering、Prompt Rewriting、LoRA（全/文本）、FLUX‑Editing等5个基线进行对比。实验结果显示，本方法在目标显著性（Target‑NSS 0.2761）上领先，语义对齐（CLIPScore 0.282）保持竞争力，视觉质量（CLIPIQA 0.9516）达到最优；用户研究也表明在显著性、语义和质量三项指标上获得显著优势。

**⚠️ 局限性**

主要局限包括：①依赖自动显著性和分割预测，可能引入噪声和偏差；②未显式建模空间交互或遮挡，可能在复杂场景下表现不佳；③仅针对静态图像，缺乏对视频连续性的支持；④标记符号设计为单对象级别，无法处理更复杂的显著性关系。

---

## 183. Probabilistic Robustness-driven Universal Adversarial Perturbations with Explainability against Deep Reinforcement Learning-based Intrusion Detection System

**arXiv ID:** 2609.30605 | [PDF](https://arxiv.org/pdf/2609.30605v1)

**作者:** Hongsen Zhang `[一作]` (University of Warwick), Carsten Maple `[通讯]` (University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于概率鲁棒性（PR）的通用对抗扰动（UAP）以及进一步利用XAI引导的PX-UAP，用于攻击基于深度强化学习的入侵检测系统（IDS）

**💡 创新点**

①首次将PR视为UAP优化目标，显式化PR驱动的半连续优化；②将XAI（集成梯度）嵌入UAP生成，形成对特征重要性权重化的扰动分配，提升攻击效率

**🔧 技术方法**

深度强化学习（DQN）IDS、概率鲁棒性指标、通用对抗扰动算法、PGD与FGM更新、XAI（Integrated Gradients）、Softmax加权与熵正则化

**📊 数据集**

CICIDS2018网络流数据集，处理为76维归一化特征，类别为二分类（正常/恶意）

**📈 对比分析**

与五种主流UAP基线（PD_mean_UAP、PD_L2_UAP、COSSIM_L3、COSSIM_L4、PCC_UAP）在不同扰动预算下对FNR与准确率进行对比，实验显示PR-UAP在所有预算区间均优于基线，PX-UAP在低预算和高预算时进一步提升约10%（FNR）

**⚠️ 局限性**

主要限制：仅在白盒设置下验证；对黑盒迁移性虽有初步探测，但未系统评估；对网络流特征空间的L₂约束假设，实际部署中可能需兼顾更多协议合规与实时性约束

---

## 184. Improving Molecular-Morphology Contrastive Pretraining using Deep-Learning-based Morphology Profiles

**arXiv ID:** 2609.30433 | [PDF](https://arxiv.org/pdf/2609.30433v1)

**作者:** Jie Li `[一作]` (GSK), Zhizhuo Zhang `[通讯]` (GSK)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种新的分子-形态对比预训练框架 v2，利用深度学习提取 Cell Painting 图像的嵌入，并通过对比学习将其与分子图神经网络（GGNN）生成的分子嵌入对齐，从而得到更能反映分子在细胞中导致形态变化的分子表征。

**💡 创新点**

创新点主要包括：①用深度学习模型（ResNet‑18 fine‑tuned on triplet loss）代替传统的 CellProfiler 手工特征，显著提升形态嵌入质量；②引入 CORAL 批处理校正以消除跨实验源的批效应；③在对比学习中将形态嵌入与分子嵌入对齐，得到更具生物学意义的分子表示；④展示该方法随训练数据规模呈 log‑linear 递增的性能提升，并在毒性、ADMET 等任务上获得竞争甚至领先的结果。

**🔧 技术方法**

技术手段包括：细胞图像预处理（照明校正、Tile 生成）→ ResNet‑18 + triplet loss 训练 → CORAL 批处理校正；分子编码采用 Gated Graph Neural Network；对比学习使用 InfoNCE 双向损失；下游任务采用线性探针和全微调两种方式。

**📊 数据集**

使用的数据集有：JUMP‑CP Cell Painting（1,728 批，含 1,728,000+ wells）；ChEMBL20 二分类任务（≈450K 分子）；四大基准集（ToxCast 8,576 任务；ChEMBL2K 2,355 任务；Broad6K 6,567 任务；Biogen3K 3,521 任务）。

**📈 对比分析**

通过与 v1、ECFP4、CLOOME、InfoAlign、CHMR 等方法在 morphology retrieval、QSAR 线性探针、全微调以及 ToxCast、ChEMBL2K、Broad6K、Biogen3K 等基准上的对比实验，结果显示：v2 在形态检索中超越 v1、接近 ECFP；在 QSAR 线性探针上均优于 v1；在 ToxCast、ChEMBL2K、Biogen3K 等任务中取得最佳或第二佳成绩，尤其在 ToxCast 上比 CHMR 高出 1.0% 点、InfoAlign 高出 3.9% 点；并且检索性能随训练样本数呈 log‑linear 上升。

**⚠️ 局限性**

局限性包括：①未考虑剂量/浓度信息，导致同一分子在不同剂量下的形态差异未被建模；②仅在单一细胞系/组织背景下训练，跨细胞系泛化性待验证；③使用 ResNet‑18 作为图像编码器，可能不及自监督 ViT 或 Masked Autoencoder 的表现；④对比学习仅对齐两个模态，未捕捉更深层的多模态交互；⑤缺少对多模态（如转录组）联合建模的探讨。

---

## 185. The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge

**arXiv ID:** 2609.30604 | [PDF](https://arxiv.org/pdf/2609.30604v1)

**作者:** Alexander Gill `[一作]` (University of Utah), Ana Marasović `[通讯]` (University of Utah)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了名为KNOWS的基准，评估浏览器代理在需要多步骤信息检索、合成与视觉/空间推理的复杂开放式任务（生成Docs/Slides/Sheets）中的助手性能；

**💡 创新点**

创新点在于将任务设计、混合评估器（确定性检查+LLM/VLM判断）和对实时网络的依赖相结合，构建了高度真实、长时序的助手式评估框架；

**🔧 技术方法**

采用Google Workspace API、VLM、LLM（Claude、GPT‑5.5、DeepSeek V4 Pro）以及BrowserGym、Comet等浏览器框架实现；

**📊 数据集**

使用包含110个任务（25 Docs、40 Slides、45 Sheets）的数据集，平均指令约200词、约2小时完成；

**📈 对比分析**

与SOTA模型（Claude Opus 4.7、GPT‑5.5、DeepSeek V4 Pro）以及不同 harness 进行比较，最高完整成功率仅3%，部分指标最高约70%，但大多数产出不可用；

**⚠️ 局限性**

局限包括任务规模有限、依赖实时网页导致可重复性受限、仅覆盖Google Workspace、模型在视觉/空间推理与长时序推理方面不足。

---

## 186. Electric Vehicle Charging Station Location Selection using Geospatial Artificial Intelligence (GeoAI)

**arXiv ID:** 2609.30417 | [PDF](https://arxiv.org/pdf/2609.30417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 187. What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study

**arXiv ID:** 2609.30402 | [PDF](https://arxiv.org/pdf/2609.30402v1)

**作者:** Akshit Sharma `[一作]` (Indian Institute of Technology Guwahati), Prashant W. Patil `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模态虚假信息检测进行了大规模的经验性研究，系统评估了不同视觉编码器、文本编码器、融合策略与分类头在三大基准上的表现。

**💡 创新点**

首次通过 3,375 次受控实验、统计显著性检验与跨数据集泛化分析，系统揭示了各设计因素对性能的真实贡献，形成可操作的设计准则。

**🔧 技术方法**

使用了冻结的预训练视觉模型（ResNet-50、ViT-B/16、CLIP ViT-B/32、SigLIP、VGG-16）和文本模型（BERT-base、DistilBERT、SBERT、CLIP-Text、SigLIP-Text），以及三种融合方式（早期、中期、后期）和三种轻量级分类头（逻辑回归、随机森林、XGBoost）。

**📊 数据集**

实验基准包括 MMFakeBench、MiRAGeNews 与 VLDBench，分别包含 11k、15k 与 31k 条图文对，覆盖多源内容、AI 生成与视听谣言。

**📈 对比分析**

通过对比实验、配置信息的系统统计（宏观精度、AUROC、准确率）以及跨数据集迁移评估，发现早期融合与 SigLIP 视觉编码器在域内效果最佳，逻辑回归/ XGBoost 在分类头上表现近似；但迁移时整体性能下降明显，表明鲁棒性有限。

**⚠️ 局限性**

局限性在于仅使用冻结编码器和简单分类头，未探讨可微调的跨模态注意力或参数高效适配；此外，效果差异虽具统计显著性，但实际幅度可能不足以支撑更复杂系统。

---

## 188. RAZOR: Pruning Replaceable Experts in LLMs

**arXiv ID:** 2609.30465 | [PDF](https://arxiv.org/pdf/2609.30465v1)

**作者:** Mingyang Song `[一作]` (Tencent), Mao Zheng `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Mixture-of-Experts (MoE) 语言模型进行专家剪枝，提出基于功能可替换性的训练无关剪枝方法。

**💡 创新点**

创新在于用共识残差衡量专家对原始混合的偏差，并通过单专家删除的闭式表达评估对输出的损害，考虑存活专家的重归一和路由补充，突破传统仅基于路由频率或输出范数的衡量。

**🔧 技术方法**

使用共识残差、单专家删除识别式、条件均方根聚合、无梯度前向计算，无需微调或子集搜索。

**📊 数据集**

在 GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V4-Flash-0731、Hy3 四大 MoE 预训练模型上进行实验；使用七域校准数据集进行预测漂移评估，九项任务基准（AIME'26、IFEval、IFBench 等）评估下游性能，Free-generation 指标评估生成行为。

**📈 对比分析**

与 Frequency、EAN、REAP 等基线相比，在 25% 和 50% 专家删减预算下，方法在所有四个模型上均获得最高的宏观平均得分（比 REAP 提升 2.12–5.59 分，击败 36/36 对比任务），并在匹配的 GLM 与 Qwen 上的逆 KL 损失更低，显示预测分布保持更好，但在 50% 剪枝时整体性能仍下降。

**⚠️ 局限性**

限制包括：仅考虑单个专家删除的补充专家可能被进一步剪枝；未考虑多轮删除的交互影响；缺乏对推理延迟、能耗、内存占用等实际部署成本的评估；实验覆盖范围有限，仅对部分基线和生成评估进行了比较。

---

## 189. Joint Effects of Node Density, Propagation, Wi-Fi Generation, and Transport Protocol on WLAN Performance: An ns-3 Study

**arXiv ID:** 2609.30510 | [PDF](https://arxiv.org/pdf/2609.30510v1)

**作者:** Leonel Olimpio Silima `[一作]` `[通讯]` (University of Porto), Leonel Olimpio Silima (University of Porto)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单接入点的上行拓扑中，使用 ns-3 对 IEEE 802.11g 与 802.11ax 两种 Wi‑Fi 模式进行系统性评估，比较 UDP 与 TCP 在 Friis、LogDistance 与 Friis–Nakagami 传播模型下的性能，覆盖 5–50 台站点、6 种距离与 3 个随机种子，共 1,080 次仿真（1,073 次有效）。

**💡 创新点**

首次将可靠性、吞吐量、时延与公平性等多指标联合起来，系统性探讨密度、距离、传播环境与传输层协议如何共同决定 WLAN 性能；并通过 FlowMonitor 的 IP‑flow 级别计数提供统一度量框架。

**🔧 技术方法**

采用 ns-3 离散事件仿真、FlowMonitor 统计、配置 802.11g/802.11ax、三种传播模型、10 秒仿真时间、1KB 数据包、每台站点 1–2 Mb/s 的发送速率，利用统计学方法计算均值、方差与 95% 置信区间。

**📊 数据集**

数据集完全由上述仿真产生，无外部真实数据；结果基于 1,073 个有效仿真运行的 CSV 汇总。

**📈 对比分析**

通过对每种协议–标准–传播组合的吞吐量、PDR、时延与 Jain 公平性进行均值比较，绘制分布图与热力图；发现 TCP 的 PDR 接近 1，但并不总能超越 UDP 的吞吐量；在 Friis 传播下 802.11ax 的平均吞吐量最高（约 26.5 Mb/s），而 802.11ax Friis–Nakagami 中 UDP 的吞吐量超过 TCP。

**⚠️ 局限性**

局限性包括：单 AP、静态站点、仅上行、10 s 运行时间、缺乏移动、障碍物与干扰、无多 BSS、未配置 OFDMA/MU‑MIMO 等高级 802.11ax 特性；随机种子数量有限、部分运行失败导致重现次数不足；FlowMonitor 仅提供 IP‑flow 级计数，未测量应用层良包率或物理层误码；缺少完整仿真与脚本，重现性受限。

---

## 190. PALM: Point-in-Time Adaptation for Financial Language Models

**arXiv ID:** 2609.30316 | [PDF](https://arxiv.org/pdf/2609.30316v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5082843254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了PALM点时间适配方法，取代每年完整预训练；

**💡 创新点**

证明时序老化并不影响性能，并用低秩适配器对新文本进行增量更新；

**🔧 技术方法**

低秩适配器、语言模型预训练、信息系数评估；

**📊 数据集**

使用1.1B-4.2B参数的PIT模型（ChronoGPT、ChronoGPT-Instruct、DatedGPT）和FNSPID金融新闻数据；

**📈 对比分析**

将每个可用版本在同一评估窗口（2020-2023）上对比，PALM在信息系数、收益和夏普比率上平均提升约1.2-1.6个百分点，且显著优于继续预训练；

**⚠️ 局限性**

仅在单一资产类别和十年时间范围内验证，且对不同评分协议及更大模型的推广仍未知。

---

## 191. Stealth Apart, Harm Together: Skill Cascading Attacks on Skill-Based Agent Systems

**arXiv ID:** 2609.30383 | [PDF](https://arxiv.org/pdf/2609.30383v1)

**作者:** Zihao Zhu `[一作]` (Chinese University of Hong Kong, Shenzhen), Baoyuan Wu `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 8181 | [OpenAlex ID](https://openalex.org/A5068027800)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了多技能级联攻击的概念，并构建了首个包含213个真实案例的级联攻击基准。

**💡 创新点**

创新点是将攻击目标分布到多个独立技能中，突破单技能扫描盲点；并提出跨技能行为组合扫描的防御思路。

**🔧 技术方法**

使用多代理红队框架、LLM（Claude Sonnet 4.6、GPT‑5.4等）以及技能包自动化改写与评判。

**📊 数据集**

基准数据来自ClawHub公开技能，涵盖10个领域、7种攻击目标和3种级联模式。

**📈 对比分析**

对24种主流agent/LLM组合进行实验，平均攻击成功率89.4%，能绕过现有单技能扫描、跨技能扫描和运行时防御。

**⚠️ 局限性**

局限包括实验仅在沙盒环境、对真实攻击者共装安装的估计不足、以及目前跨技能防御仍不完善。

---

## 192. NeuralCert: certified computational discovery of extremal mathematical constructions

**arXiv ID:** 2609.30296 | [PDF](https://arxiv.org/pdf/2609.30296v1)

**作者:** Mark Patrick Roeling `[一作]` `[通讯]` (Netherlands Defence Academy), Mark Patrick Roeling (Netherlands Defence Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用神经网络进行高维变分试验函数的学习、谱诊断、修剪，并通过多模评估实现完全精确的证明；

**💡 创新点**

将计算机发现与严格证明相结合，提出了从发现到认证的完整流程，首次实现了在个人电脑上完成全程可验证的数学证明；

**🔧 技术方法**

深度学习优化、谱分析与修剪技术、精确多模评估、可分离表示法；

**📊 数据集**

三类极值问题（如几何极值、能量最小化、组合极值等）作为实验案例；

**📈 对比分析**

与传统手工构造/数值搜索方法对比，发现了更优的构造、揭示了经验不变量并形成正式证明，且发现了优化瓶颈，表现出比旧方法更高的发现效率和证明完整性；

**⚠️ 局限性**

对网络架构和参数调优高度依赖，修剪与评估步骤仍需人工调试，且在更大规模问题上计算成本和数值稳定性仍是挑战。

---

## 193. The Price of Thought: Does Test-Time Reasoning Pay in LLM Trading?

**arXiv ID:** 2609.30705 | [PDF](https://arxiv.org/pdf/2609.30705v1)

**作者:** Jiayi Chen `[一作]` (New Jersey Institute of Technology), Guiling Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对DeepSeek、GPT、Gemini三大LLM在固定信息、提示、输出、组合规则下，改变推理力度（无推理、低推理、高推理、最大推理）进行受控实验，评估其对2024年美国股票组合净收益的边际经济效应。

**💡 创新点**

①首次在固定决策管线中仅调整推理力度，测量其真实收益增量；②采用数值、可识别新闻、屏蔽新闻三种信息条件，并对DeepSeek进行四级推理曲线；③通过重复生成和随机审计评估结果稳定性；④综合交易成本、token使用、错误率等运维维度，形成完整的“推理成本‑收益”评估。

**🔧 技术方法**

使用LLM的chain‑of‑thought或多步推理控制；将结构化分数映射至均衡多头/空头组合；利用Newey‑West HAC、bootstrap和Holm校正进行配对统计推断；对多轮生成进行可靠性与稳定性评估。

**📊 数据集**

2024年美国上市股票日数据（120只筛选为100只活跃股），241个交易日形成日；每日提供16个数值特征和可/不可识别新闻文本；覆盖整个交易年度的真实价格、交易成本等信息。

**📈 对比分析**

比较方法：配对比较低推理与基线推理的每日组合净收益差异，使用Newey‑West HAC估计标准误、95%置信区间和经济阈值（0.25 bps/天）。结果显示九个对比均未超过阈值，CI均跨零，DeepSeek在某些条件下表现非单调甚至负面，未见可重复的收益提升。

**⚠️ 局限性**

局限性：仅基于单一年份单一市场；推理力度标签在不同模型间不等价；未考虑真实交易冲击、流动性限制和非线性成本；对未来模型更新敏感；统计功效不足，无法完全排除小幅经济效应；实验使用固定组合规则，未检验其他策略设置。

---

## 194. Backbone-Adaptive Evidence Routing for Robust Pairwise LLM Judging

**arXiv ID:** 2609.30751 | [PDF](https://arxiv.org/pdf/2609.30751v1)

**作者:** Zeyan Li `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多种基准和不同语言模型裁判背骨，提出了BAER框架，能够根据条件动态选择最合适的证据获取方式，并保证候选答案的对称性。

**💡 创新点**

创新点包括：① 将裁判信号拆分为符号偏好与候选无关的可靠性；② 构造三种对称的证据头（堆叠、路由、参考验证）；③ 在每个条件下冻结头选择，避免测试时过拟合。

**🔧 技术方法**

主要技术：符号化对称信号表示、Logit缩放与衰减、正则化的对称Logistic堆叠、候选无关的专家路由器（随机森林）以及不暴露候选答案的参考验证。

**📊 数据集**

使用四个公开优先级评估基准（RewardBench、JudgeBench、HH‑RLHF、UltraFeedback）与两种8B裁判模型（Qwen 3‑8B、Llama 3.1‑8B）进行实验。

**📈 对比分析**

与八种外部裁判协议（直接双向、Self‑Consistency、Chain‑of‑Thought、Rubric‑based 等）进行比较；BAER 在所有八个条件下均达到最高准确率，平均提升约 4.9 分，覆盖率始终为 100%。

**⚠️ 局限性**

局限性：需要额外的推理调用（比单次裁判成本更高），仅在已知基准/裁判背骨的条件下可部署，且当前针对 8B 模型；未来需研究对未知条件的泛化与成本削减。

---

## 195. Closing the Loop: Continuous Measurement-Driven Refinement of Offloading Predictions

**arXiv ID:** 2609.30429 | [PDF](https://arxiv.org/pdf/2609.30429v1)

**作者:** Falk Dettinger `[一作]` (University of Stuttgart), Michael Weyrich `[通讯]` (University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个基于实时测量的闭环反馈机制，能够在车辆边缘计算过程中持续校准并改进绝对值预测。

**💡 创新点**

首次将在线学习与计算卸载管道结合，使用服务器特定的多头MLP任务头进行增量更新，从而实现对动态网络与硬件条件的即时适应。

**🔧 技术方法**

核心技术包括轻量化多头MLP、滑动窗口特征提取、服务器特定标准化、滚动平均预测平滑、增量在线更新以及sigma误差分析。

**📊 数据集**

数据来源为真实车辆在两大Kubernetes集群（曼海姆与斯图加特）执行的感知任务（物体识别、情感识别）的实时执行时间、网络延迟与资源利用记录。

**📈 对比分析**

与传统离线训练的基线模型相比，采用在线测量驱动的闭环后，RTT、处理时间等指标的MAE/MSE/MAPE显著下降，决策正确率提升；但在高变异服务器上仍存在较大残差。

**⚠️ 局限性**

主要限制包括轻量模型难以完全捕获多峰、长尾分布；更新需要足够批量导致响应延迟；滑动窗口冷启动期预测不稳定；残差仍有可能超过100 ms的容忍阈值。

---

## 196. Differentiable RNA Secondary Structure Extraction for Deep Learning

**arXiv ID:** 2609.30752 | [PDF](https://arxiv.org/pdf/2609.30752v1)

**作者:** Tyler Illman `[一作]` (University of Western Australia), Ryan K. Krueger `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 RNA 二级结构预测模型在训练目标与结构提取方法不匹配时的表现，并提出通过对模型输出施加对称双随机矩阵（SDSM）归一化来直接生成可解释的概率矩阵，从而消除后处理步骤。

**💡 创新点**

创新点在于：①系统评估训练与提取的契合度对预测性能的影响；②提出可微分的 SDSM 归一化，使模型直接输出对称双随机（即概率）矩阵；③通过四种提取算法（Nussinov、最大权匹配、SPOT‑RNA 贪婪、RiNALMo 贪婪）对比不同训练策略的效果。

**🔧 技术方法**

使用技术包括：深度可微分的 Nussinov‑like 动态规划、最大权匹配（Edmonds 的 Blossom 算法）、对称双随机矩阵归一化（修改的 Sinkhorn‑Knopp 迭代）、二分类交叉熵损失（加权平衡）以及基于 Bootstrap 的检验。

**📊 数据集**

采用公开基准 ArchiveII 数据集（3,975 条 RNA 序列，长度 <200，包含 tRNA、SRP、5S 等家族）。

**📈 对比分析**

实验结果显示：在预训练 RiNALMo 输出上四种提取器性能相近；在自训练模型上，训练目标与提取器一致时性能提升显著；SDSM 模型在所有提取器上均优于 BCE 基线，尤其在最大权匹配和 SPOT‑RNA 上取得 0.9 以上的 F1 分数，且预提取 MSE 最低。

**⚠️ 局限性**

限制包括：模型规模较小、仅在短 RNA（<200 nt）上训练，未充分评估跨家族泛化能力；仅考虑非伪结结构，伪结预测能力有限；实验为过拟合设置，未验证在真实未知序列上的表现。

---

## 197. CSCWD: Cross-Scale Channel-wise Knowledge Distillation for Lightweight Tiny Object Detection on Edge Devices

**arXiv ID:** 2609.30395 | [PDF](https://arxiv.org/pdf/2609.30395v1)

**作者:** Amir Zamani `[一作]` (Islamic Revolution Comprehensive University), Zeinab Ghasemi-Naraghi `[通讯]` (K.N. Toosi University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出一种交叉尺度通道知识蒸馏（CSCWD）框架，用于在边缘设备上提升轻量级小目标检测的精度，且不改变模型推理结构。

**💡 创新点**

创新点在于：① 将教师网络高分辨率的P₂层特征跨尺度迁移到学生的P₃层；② 仅在训练阶段引入蒸馏，推理时保持原YOLO11n网络；③ 通过通道级概率分布对齐，实现精细空间信息的传递。

**🔧 技术方法**

主要技术包括：YOLOv11轻量级网络、通道级知识蒸馏（CWD）、跨尺度特征对齐、温度缩放softmax与KL散度损失、NCNN-FP16部署。

**📊 数据集**

使用的数据集：训练和验证采用Drone‑vs‑Bird（7个序列）和VisDrone（迁移训练）；零样本评估使用DUT‑Anti‑UAV；对照实验用YOLOv8n、v10n、YOLO11s、YOLO11m-P2等模型。

**📈 对比分析**

与基线CA‑YOLO11n相比，CSCWD提升mAP@0.5从47.25%到50.17%（+2.92pp），召回率从16.92%到17.30%；在Raspberry Pi 5上NCNN‑FP16部署，mAP@0.5提升至50.32%，平均延迟82.3 ms、12.15 FPS不变；在DUT‑Anti‑UAV零样本测试中，mAP@0.5提高1.77pp。

**⚠️ 局限性**

局限性包括：仅在航空小目标场景验证，跨域泛化仍有限；对输入分辨率的敏感性需进一步研究；nTCR仅为相关性指标，未证明因果；持续负载测试仅单次，缺乏统计；未加入时序建模或更复杂的压缩技术。

---

## 198. SR-Gadgets: Make Scan-Resistant Caching Practical

**arXiv ID:** 2609.30468 | [PDF](https://arxiv.org/pdf/2609.30468v1)

**作者:** Yunjia Zheng `[一作]` (Harvard University), Juncheng Yang `[通讯]` (Harvard University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了可插拔的 Gadgets（ProbBypass 与 RecencyGuard），使多队列缓存算法在面对重复扫描时更具鲁棒性，显著减少误差峰值（miss‑ratio cliff）和 Belady 异常。

**💡 创新点**

首次用 C-score 与 P-score 定量衡量误差峰值和 Belady 异常；证明 LIRS 的优势来自队列调控而非堆栈距离；设计通用 Gadgets，可让现有算法无需重写即可实现扫描抵抗。

**🔧 技术方法**

多队列缓存结构、ghost 队列、堆栈距离、概率绕过（ProbBypass）、递归水印（RecencyGuard）、PAVA 逼近、C-score/P-score 统计方法，以及 libCacheSim 仿真器。

**📊 数据集**

使用 5,538 条生产级块级 I/O 跟踪（Cloudphysics、Tencent CBS、Alibaba）和 67 条非块级键值/对象跟踪（Twitter、Meta KV、Meta CDN、WikiMedia CDN、Tencent Photo 等）。

**📈 对比分析**

与 FIFO、LIRS、Cliffhanger 等基线在 1% 与 10% 缓存容量下进行 miss‑ratio improvement 对比；结果显示 Gadgets 可使 5 种算法 miss‑ratio 降低最多 23.1%，同时显著降低 C-score 与 P-score，提升 5~18% 的命中率。

**⚠️ 局限性**

对 2Q 的 RecencyGuard 可能产生负面影响；仍依赖多队列结构，极大规模或极短序列场景需进一步调优；对非扫描工作负载的提升不一定显著。

---

## 199. StarWM: Self-Supervised Trained Attention Routing for Robust World Models

**arXiv ID:** 2609.30667 | [PDF](https://arxiv.org/pdf/2609.30667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. Strategic Self-Consistency

**arXiv ID:** 2609.30352 | [PDF](https://arxiv.org/pdf/2609.30352v1)

**作者:** Tori Qiu `[一作]` (Carnegie Mellon University), Manuel Gomez-Rodriguez `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 8485 | [OpenAlex ID](https://openalex.org/A5042180520)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种算法，让LLM服务提供者在自一致性（self-consistency）框架下伪造额外推理路径，从而对用户过度计费。

**💡 创新点**

揭示了即使在承诺的自适应停止规则下，提供者仍可通过重新排序和追加路径来欺骗计费，并证明该攻击可规避精确似然比审计。

**🔧 技术方法**

使用了自适应停止规则（PPR‑1v1、ASC）、似然比检验、动态规划计数方法以及自一致性推理路径生成技术。

**📊 数据集**

在数学推理、科学推理和通用问答（如数学竞赛、科学问答、问答基准）上进行实验，基于指令模型与推理模型预生成的答案分布。

**📈 对比分析**

与标准自一致性实现及其他停止规则对比，实验显示额外路径分布呈重尾，且在不同审计阈值下可产生数十至数千条额外路径，表明过度计费风险显著。

**⚠️ 局限性**

仅适用于离散可验证答案的自一致性，未覆盖连续输出或奖励导向的停止规则；对非可接受停止规则的理论下界有限，且实验范围受限于所选基准。

---

## 201. Policy-Calibrated DAgger: Offline Calibrated Noise Injection for Imitation Learning

**arXiv ID:** 2609.30462 | [PDF](https://arxiv.org/pdf/2609.30462v1)

**作者:** Jenny Wang `[一作]` (Carnegie Mellon University), George Kantor `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在DAgger迭代过程中利用生成式策略的概率特性，离线估计并注入与策略误差匹配的噪声，从而在不增加专家干预的情况下提升机器人对复杂、狭窄环境中的目标跟踪鲁棒性。

**💡 创新点**

核心创新在于：①将生成式策略的“部分去噪”作为学习动作插值器，精准估计策略对专家轨迹的误差；②基于该误差离线计算策略校准噪声（policy‑calibrated noise），无需在环中投射噪声；③将校准噪声用于生成针对性恢复轨迹，显著降低协变量漂移。

**🔧 技术方法**

使用的主要技术包括：生成式扩散模型的前向/逆扩散过程与部分去噪；DAgger与HG‑DAgger的交互式数据聚合框架；多模态误差估计与噪声参数的离线计算；以及基于PD控制的动作补全与重标记。

**📊 数据集**

实验数据集为两套模拟环境：① 3D 真实感引擎杠杆任务的物理仿真（使用SplatSim + PyBullet）；② 2D 规划平面三连杆机器人实现的狭窄通道抓取任务；两者均采用500条专家演示（RRT‑based oracle）并在多轮DAgger循环中收集干预数据。

**📈 对比分析**

与BC、HG‑DAgger、固定噪声水平等基线方法比较，校准噪声策略在平面任务中将成功率从79.9%提升至87.3%，比HG‑DAgger提升约2.8%；在3D引擎杠杆任务中，在相同干预数据下，校准噪声实现的成功率高于BC和HG‑DAgger，并与最佳固定噪声水平相当，避免了需手动调参的噪声搜索。

**⚠️ 局限性**

主要局限包括：① 需要已训练好的生成式扩散策略作为误差估计基准，若策略欠拟合误差估计可能失真；② 目前仅验证于仿真环境，真实机器人验证尚未完成；③ 部分去噪过程假设动作在[-1,1]归一化，实际高维或非欧氏动作空间需进一步适配。

---

## 202. SlideLab: Audience-Centered Scientific Slide Generation and Evaluation

**arXiv ID:** 2609.30294 | [PDF](https://arxiv.org/pdf/2609.30294v1)

**作者:** Vidushee Vats `[一作]` (INSAIT, Sofia University St Kliment Ohridski), Yuxia Wang `[通讯]` (INSAIT, Sofia University St Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个训练-free的多智能体框架SlideLab，用于从科研论文自动生成连贯、视觉化的演示文稿，并引入了模拟观众交互的评估环境ConfArena。

**💡 创新点**

核心创新在于先规划完整叙事蓝图，再逐步构建并迭代优化幻灯片，同时利用多模态LLM进行布局调试与事实校验，并通过按幻灯片交互的评估框架精准捕捉演示质量问题。

**🔧 技术方法**

采用多智能体架构（Planner、Generator、LayoutDebugger、Compositor），结合前沿长上下文多模态LLM、Playwright渲染、图像生成模型（如Stable Diffusion）以及视觉质量与事实一致性校验技术。

**📊 数据集**

使用ArcBench（100篇论文-幻灯片对）和30篇自注释机器学习论文做人类评估，同时利用公开的幻灯片模板和论文PDF进行实验。

**📈 对比分析**

与PPTAgent、DeepPresenter、Kimi Slides、Manus进行盲人类偏好实验，SlideLab在30篇论文中被选为最佳77%；在ConfArena评估中获得最高环境分数，与人类排名一致，且推理token使用量约为DeepPresenter的四分之一。

**⚠️ 局限性**

仅针对会议式科研演示场景，未验证教育或商业演示等其他领域的适用性；系统仍需人工审校以避免生成错误或误导性内容。

---

## 203. TrafficImag: A Benchmark for Counterfactual Roadside Traffic Video Generation

**arXiv ID:** 2609.30722 | [PDF](https://arxiv.org/pdf/2609.30722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 204. Untangling the Spaghetti Code in Game Development: A Review of Challenges and Academic Solutions

**arXiv ID:** 2609.30349 | [PDF](https://arxiv.org/pdf/2609.30349v1)

**作者:** Esdras Caleb Oliveira Silva `[一作]` (UFRN), Lyrene Fernandes da Silva `[通讯]` (UFRN)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开展了针对游戏开发领域代码异味与技术债务的系统性文献综述，梳理了30多篇核心论文，评估了学术提出的检测工具、产品线、测试框架等解决方案，并考察了其在业界的接受度与实际效果。

**💡 创新点**

创新点包括：① 采用BART-large‑MNLI零样本分类模型实现对海量文献的自动化预筛选；② 结合严格的质量评估标准对34篇研究进行评议；③ 通过行业访谈与实测数据对比，系统性揭示游戏领域代码质量差异与解决方案的采纳瓶颈。

**🔧 技术方法**

主要技术手段为系统性综述方法（PRISMA 2020 规范）、BART-large‑MNLI 零样本文本分类、Python 数据清洗脚本、质量评估问卷（Galster 等），以及统计描述与可视化工具。

**📊 数据集**

数据集为 1599 篇文献，来源于 Scopus、IEEE Xplore、Web of Science、ACM DL、ScienceDirect 与 Google Scholar；最终筛选为 34 篇原始研究（含 1 篇综述）作为分析核心。

**📈 对比分析**

对比方法：在游戏与非游戏软件域中比较代码异味出现率、技术债务规模、测试覆盖率以及工具采纳率。结果显示：① 游戏领域代码异味与技术债务更高；② 自动化测试覆盖率低（约 15%）；③ 现有检测工具与产品线虽有效，但实际采纳率仅约 30%，主要受时间压力与缺乏集成支持限制。

**⚠️ 局限性**

局限性：① 文献筛选与评估由单一研究者完成，尽管与二级研究者讨论以降低偏差；② BART‑MNLI 分类阈值 0.5 可能导致漏检；③ 关键词与检索字符串可能引入样本偏倚；④ 综述截止于 2024‑08，无法覆盖此后发表的研究；⑤ 对行业实际采纳情况的观察有限，缺乏长期跟踪验证。

---

## 205. GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for \textit{De Novo} Peptide Sequencing

**arXiv ID:** 2609.30542 | [PDF](https://arxiv.org/pdf/2609.30542v1)

**作者:** Abdellah El Mekki `[一作]` (University of British Columbia), Muhammad Abdul-Mageed `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为GyroNovo的de novo肽段测序框架，利用解码器错误引导的潜在缺失碎片恢复和连续m/z旋转注意力来更好地重建缺失碎片并显式建模峰间相对质量关系。

**💡 创新点**

创新点包括：①根据解码器当前预测错误动态加权缺失碎片的重构损失；②基于解码器难度生成易/难两种增强视图，提升对难以识别残基的证据；③将连续质量差异融入Rotary Position Embedding，实现对峰间相对质量的直接编码。

**🔧 技术方法**

技术实现采用Transformer编码-解码结构，内置潜在缺失碎片恢复模块，结合RoPE改进的连续质量旋转注意力，配合多视图训练与错误驱动的增量正则化。

**📊 数据集**

实验使用NovoBench基准数据集，包括Nine-species、Seven-species和HC-PT三组肽谱数据。

**📈 对比分析**

与DeepNovo、PointNovo、Casanovo、AdaNovo、LIPNovo及其改进版LIPNovo+等多种基线在氨基酸、PTM及完整肽段级别进行精度/召回率和AUC评估；GyroNovo在所有数据集上均以约9%肽段精度提升、7%氨基酸精度提升，以及相较LIPNovo+的1-3%指标提升，表现最优。

**⚠️ 局限性**

局限性包括：训练阶段需多视图增强，导致计算和内存负担相对较高；依赖理论碎片作为监督，可能对极度缺失或高噪声谱的泛化能力有限；并未在极端化学修饰或非典型离子形式上进行专门评估。

---

## 206. Recursive Self-Improvement via On-Policy Distillation for Reasoning

**arXiv ID:** 2609.30652 | [PDF](https://arxiv.org/pdf/2609.30652v1)

**作者:** Shangjian Yin `[一作]` (Meta AI), Hamed Firooz `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了动态共演自蒸馏框架 (DCE) 与自我精简学习 (SRCL)，实现LLM推理过程的递归自我提升。

**💡 创新点**

创新点在于让教师与学生在每个训练轮次都由同一模型更新的检查点初始化，实现金色答案条件的教师随学生进化；同时通过 SRCL 训练更短、更准确的回答来控制推理成本。

**🔧 技术方法**

采用 on‑policy 自蒸馏、前向 KL 散度、金色答案前置提示、教师刷新机制、答案验证与自我重写筛选等技术。

**📊 数据集**

使用 OpenThoughts 1.47 万条数学推理训练样本，评测集包括 AIME 2024/25/26、HMMT 2025 等竞赛题目。

**📈 对比分析**

与 Base、SFT、GRPO、OPSD 及推理时间扩展对照；在 Qwen3‑8B 上 DCE+SRCL 将 Average@12 从 30% 提升至 66%，平均生成长度从 ~19k 降至 17.5k；在 4B 同样提升至 62%；整体显著优于所有基线。

**⚠️ 局限性**

局限性包括对大规模算力的依赖、对小模型效果有限、需要已验证答案作为教师输入、以及未在更广泛领域（非数学推理）进行充分验证。

---

## 207. Learning What to Skip: Counterfactual Credit Assignment for Efficient Multi-Agent LLM Workflows

**arXiv ID:** 2609.30734 | [PDF](https://arxiv.org/pdf/2609.30734v1)

**作者:** Jinfeng Xu `[一作]` (University of British Columbia), Victor C. M. Leung `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LW2S框架，学习何时跳过多代理LLM工作流中的后续组件以降低成本

**💡 创新点**

通过对控制性“跳过”干预进行逆因果信用分配，训练动作特定安全模型并结合校准与领域原生守门实现安全、可回退的跳过决策

**🔧 技术方法**

逆因果干预收集、动作安全预测（逻辑回归+TF‑IDF特征）、校准阈值选择、领域守门器、顺序回退策略

**📊 数据集**

MATH、GSM8K、MMLU、MBPP等公开基准

**📈 对比分析**

与RouteLLM、GraphPlanner、AutoMix、Prompt‑LLM等基线在相同后缀跳过协议下比较，LW2S在所有任务上至少减少25% token成本，且在大多数设置下保持或提升整体准确率，且观察到零任务级降级

**⚠️ 局限性**

需预先收集控制性干预数据，安全模型对未知域的泛化有限，且在某些高成本边界（如MMLU）仍可能出现轻微降级

---

## 208. NEMSim: Learning Control-Conditioned Multi-Event Physical Dynamics via Executable Event-Mechanism Priors

**arXiv ID:** 2609.30718 | [PDF](https://arxiv.org/pdf/2609.30718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 209. A Long-Legged, Direct-Drive Monopedal Robot Achieves Exceptional Jump Height

**arXiv ID:** 2609.30530 | [PDF](https://arxiv.org/pdf/2609.30530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 210. To Store or To Regenerate? A Cost Model for AI-Generated Content at Scale

**arXiv ID:** 2609.30448 | [PDF](https://arxiv.org/pdf/2609.30448v1)

**作者:** Yunjia Zheng `[一作]` (Harvard University), Juncheng Yang `[通讯]` (Harvard University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文建立了AI生成内容存储与按需重生成的生命周期成本模型，分析了何时重生成比持久化存储更便宜，并提出了利用中间表示（IR）缓存的LazIR方案；

**💡 创新点**

创新点在于将生成管线的前置计算转移到IR层，只保留轻量级解码，从而显著降低重生成成本，并将成本折算为IR大小与解码代价的函数；

**🔧 技术方法**

技术上采用了基于硬盘、磁带价格趋势、GPU算力与能耗下降的闭式成本模型，结合Zipfian访问偏好、缓存热集、IR存储和GPU推理；

**📊 数据集**

实验使用FLUX.1-dev图像生成模型、MusicGen-Large音乐生成模型以及公司内部的2.87年真实图像生成访问轨迹（约2.07亿次请求、9.23千万张图像）；

**📈 对比分析**

比较方法是将三种重生成策略（完整重生、缓存+重生、LazIR）与HDD、磁带两种持久化基线进行成本和性能对比；结果显示LazIR在两种模态下成本分别下降3.1~3.7倍，且在真实轨迹中成本降低至原HDD存储的一半，同时保持100~200ms的交互延迟；

**⚠️ 局限性**

局限性包括假设未来成本与技术趋势保持不变、访问分布为Zipfian、未考虑压缩或不同数值精度导致的再生差异、未纳入通胀、未对硬盘/磁带供应链冲击做建模等。

---

## 211. SignTrace: Describe a Sign, Find the Word

**arXiv ID:** 2609.30295 | [PDF](https://arxiv.org/pdf/2609.30295v1)

**作者:** Zengji Tu `[一作]` (Peking University), Dai Wan `[通讯]` (Peking University)

**通讯引用:** 241 | [OpenAlex ID](https://openalex.org/A5070500360)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SignTrace 系统，利用自然语言运动描述查询，检索中国手语字典条目，并在 500 条基准查询上评估检索性能。

**💡 创新点**

创新点在于将 LLM 结合多通道检索与结构化动作匹配：① 用 LLM 对字典条目进行动作丰富与视觉重述；② 并行提取查询动作与重写；③ 通过七个文本/结构/意义通道联合检索；④ 使用 LLM 对候选条目进行动作匹配重排序。

**🔧 技术方法**

采用的技术包括：DeepSeek 与 Gemini LLM 进行动作抽取、重写与重排序；BM25 文本检索；结构化动作特征匹配；RRF 融合权重；并行查询处理与并发请求。

**📊 数据集**

使用的数据集为 6,699 条中国手语国家普通手语词典条目及其 8,687 条意义记录，并构造 500 条从词典动作描述派生的运动查询作为评测基准。

**📈 对比分析**

通过与原始 BM25、七通道检索以及重排序后的最终排序进行对比。结果显示：Hit@1 94.0%，Hit@9 97.4%，MRR 0.9540；相较于单通道 BM25 提升显著，重排序将首位命中率从 71.8% 提升至 94%。

**⚠️ 局限性**

局限性包括：评测基准由词典自生成，缺乏真实用户查询；检索性能受字典表述与查询相似度影响；未覆盖不同方言、视频输入或多轮交互；模型服务可能变动，未评估稳定性；缺乏外部用户体验与学习效果验证。

---

## 212. Unifying In-Memory Data Analytics through Sparse Compilation

**arXiv ID:** 2609.30497 | [PDF](https://arxiv.org/pdf/2609.30497v1)

**作者:** Anand Jayarajan `[一作]` (University of Toronto, NVIDIA), Gennady Pekhimenko `[通讯]` (University of Toronto, NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

Reffine 是一个基于编译器的稀疏内存分析引擎，提供统一的中间表示和后端，将各种数据分析工作负载编译为高效多核代码。

**💡 创新点**

创新点在于设计了基于关系代数与稀疏迭代理论的 Field/Operator/Reduction IR，实现跨域优化（融合、并行化）并自动生成硬件高效代码。

**🔧 技术方法**

使用了稀疏编译理论、SMT 求解器（Z3）、LLVM IR JIT、Apache Arrow 内存格式、矢量化循环等技术。

**📊 数据集**

使用了 TPC‑H（scale factor 10）、合成流/图数据（1 亿条随机时间序列、Snap Twitter 图）等数据集。

**📈 对比分析**

与 Pandas、Polars、DuckDB、Umbra、NetworkX 等系统对比，Reffine 在多核下对 TPC‑H 实现最高 24.9×、对流/图工作负载分别 18.3×、47.9× 的加速。

**⚠️ 局限性**

局限性包括：当前后端对稠密线性代数优化不足，无法完全取代专业引擎；高级查询规划等优化需在 IR 之上实现；对极端大规模数据的内存消耗与调度策略尚未全面评估。

---

## 213. HuGo: LLMs as Whole-Body Policy Code Designers for Humanoid Loco-Manipulation

**arXiv ID:** 2609.30594 | [PDF](https://arxiv.org/pdf/2609.30594v1)

**作者:** Seoyeon Choi `[一作]` (University of California, Berkeley), Negar Mehr `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出HuGo框架，利用大型语言模型（LLM）在冻结的低层全身控制策略上生成闭环高层策略代码，并通过数值轨迹+视频帧的评估循环，进行局部代码差分修正，完成从任务描述到可执行策略的自动化生成；

**💡 创新点**

突破性在于：①不需要专家演示、参考动作或手工奖励，直接从自然语言任务描述生成高层策略；②引入循环评估与代码diff更新机制，使模型在仿真或真实机器人上可自适应修正；③可实现零样本从仿真到硬件的迁移并进一步通过硬件回放进行细化；

**🔧 技术方法**

核心技术包括GPT-4等LLM用于代码生成与评估、可执行策略代码的即时运行、视频+轨迹的反馈提取、局部代码diff生成与应用、两种低层全身控制策略（RL跟踪与SONIC 3‑point VR）以及IsaacSim仿真与Unitree G1硬件平台；

**📊 数据集**

使用五个随机化的仿真任务（低门、窄门、按键、推箱、举箱），并在硬件上针对两个任务进行测试；未使用任何演示或奖励数据集，全部基于任务自然语言描述；

**📈 对比分析**

与高层强化学习（PPO）和基于演示的HDMI方法对比，HuGo在五任务中平均成功率≥80%，在硬件上零样本迁移可达90%（低门）和67%（举箱），高层RL表现低，HDMI在部分任务略优；

**⚠️ 局限性**

局限包括：受限于低层策略的姿态与抓取控制（如抓取失稳、接触力不足）；LLM在诊断接触失败时可能缺乏细粒度理解，导致修正不完全；需要多次生成并挑选最佳策略以提升可靠性，且对视觉信息依赖较高。

---

## 214. Not All Memories Are Equal: Hierarchical Collaborative Memory for Validity-Aware Retrieval in LLM Agents

**arXiv ID:** 2609.30289 | [PDF](https://arxiv.org/pdf/2609.30289v1)

**作者:** Yufei Shi `[一作]` (Nanyang Technological University), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 HiCoMER 框架，针对团队协作场景下的层级化、持续演变的记忆进行维护、有效检索和基于记忆的回答生成。

**💡 创新点**

设计层级化冲突更新器与可验证检索器，实现团队与个体记忆的有效性维护与冲突消除，避免语义相关但已失效的记忆被检索。

**🔧 技术方法**

结合密集检索、交叉编码重排序、LLM 结构化冲突更新、可验证评分 MLP 以及 GRPO 强化学习等技术。

**📊 数据集**

构造了两个生物医学协作数据集 ALI 与 PROTAC，包含团队与个体记忆以及冲突标注。

**📈 对比分析**

与多种基线（平面 RAG、混合 RAG、重排 RAG、G-Memory、Self-RAG、MemGPT、Mem0 等）在 ORR@5、CRR@5、NDCG@10、ROUGE‑L、Decision F1 等指标上比较，HiCoMER 在所有指标上显著优于基线，尤其在维护有效记忆率和答案质量上提升显著。

**⚠️ 局限性**

对团队/个体记录的噪声、缺失或对齐不良敏感，且仅在生物医学场景验证，缺乏跨领域泛化。

---

## 215. Anatomy-Aware Dexterity-Driven Design Optimization of Surgical Continuum Robots

**arXiv ID:** 2609.30745 | [PDF](https://arxiv.org/pdf/2609.30745v1)

**作者:** Tony Qin `[一作]` (University of North Carolina), Ron Alterovitz `[通讯]` (University of North Carolina)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

针对手术连续机器人提出一种基于解剖学与灵活度联合评估的设计优化方法，目标是最大化可达体积灵活角(RVDSA)；

**💡 创新点**

创新点在于引入RVDSA作为目标函数，结合Jacobian引导的采样与自运动采样的高效运动规划，且使用自适应模拟退火实现全局收敛；

**🔧 技术方法**

使用并行概率道路图(PPRM)、Jacobian伪逆求逆运动学、自运动空间采样以及自适应模拟退火优化；

**📊 数据集**

使用来自National CT Colonography Trial的数据集，选取5个结肠解剖模型并标注癌变息肉，分辨率1mm³；

**📈 对比分析**

与纯随机采样、纯Jacobian采样以及仅针对3D体素覆盖的优化方法对比，最优设计在RVDSA上平均提升至2.58 sr（约107.8°），比体素优化提升78%；

**⚠️ 局限性**

局限包括未考虑双臂协同与臂间碰撞、样本解剖数量有限、计算成本高且缺乏解析逆运动学解。

---

## 216. Empty Intersection: Provenance Coverage Rose to 98% and Neither Verification Decision Moved

**arXiv ID:** 2609.30308 | [PDF](https://arxiv.org/pdf/2609.30308v1)

**作者:** Dong Hyeon Jeon `[一作]` `[通讯]` (Independent Researcher), Dong Hyeon Jeon (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一次生产部署的冻结快照（194,620 行）上，测量了两种结构性预防方案：对每行打标记（provenance grade）并在验证查询中过滤，只保留观察标记的行；以及用单一写入入口（write ingress）强制要求每行具有有效的 provenance 标记；并评估这些方案对两条验证决策（V1 与 V2）是否能获得可接受输入。

**💡 创新点**

首次系统地量化了 provenance grading 与写入入口在实际部署中对验证决策的“达成度”（reach）影响，揭示了高覆盖率并不一定能改变决策结果，证明了覆盖率与决策可达性是独立的度量。

**🔧 技术方法**

利用冻结快照重放技术、SQL 查询过滤、四种候选标记词汇表、写入入口验证器以及对决策查询窗口的手工分析，对比各方案对可接受输入集的影响。

**📊 数据集**

使用一次生产部署的完整快照数据集，包含 194,620 条记录和两条验证决策（时间序列 V1 与关系型 V2）所需的 32 条查询行。

**📈 对比分析**

通过计算决策在每个干预后可接受的输入数量、覆盖率百分比以及被拒绝写入的行数进行比较；结果显示：I3 使两条决策变为不可判定；I1 将覆盖率从 36.1% 提升至 98.4% 但未改变决策；I2 拒绝了 3,070 条写入；性能指标（如执行时间、吞吐量）未在本文测量。

**⚠️ 局限性**

局限性包括：仅针对单一部署（非样本）；快照中无观察标记行，因而 I3 的过滤路径未被激活；写入入口仅作为重放验证器实现，未评估其在真实生产环境中的行为；未执行完整的修复例程；所列的必要条件并非充分条件；并且未探讨被拒绝或新增标记行对其他查询或后续工作可能产生的价值。

---

## 217. BioEVAL: A global, multi-institutional benchmark of large language and multimodal models for bioengineering

**arXiv ID:** 2609.30489 | [PDF](https://arxiv.org/pdf/2609.30489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 218. WeEnv: The Environment for Agentic Reinforcement Learning at WeChat

**arXiv ID:** 2609.30766 | [PDF](https://arxiv.org/pdf/2609.30766v1)

**作者:** Yang Yu `[一作]` (Tencent), Junjie Zhang `[通讯]` (Tencent)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对agentic RL中环境生命周期产生的高成本，提出了一套完整的环境管理框架，涵盖打包、初始化和资源分配三阶段，显著降低迭代时间。

**💡 创新点**

创新点包括：①层级组化（layer group）包装，使可变组件独立发布并在初始化时通过轻量级层叠完成环境构建；②基于分离元数据与内容的新层格式，支持按需读取；③弹性资源分配机制，实时监控并按需调整CPU/内存，避免固定配额导致的性能瓶颈。

**🔧 技术方法**

技术手段：层级组化（layer group）与环境计划（environment plan）; 新型层格式（metadata+内容分离，支持范围读取）；FUSE+OverlayFS实现按需文件系统；cgroup监控+动态配额调整；多级缓存（实例、节点、集群）。

**📊 数据集**

主要使用SWE-Smith多语言数据集（Python、Rust、C++、Go、JavaScript 等）与 Qwen3-8B/14B LLM；在 WeChat 的内部训练环境中部署验证。

**📈 对比分析**

与三种主流后端（E2B、Docker、AgentENV）对比，整体迭代时间显著下降：环境初始化时间从 53.4% 降至 9.1%，初始化时间提升 5.6–14.2×；任务执行速度提升 1.35×，弹性资源分配进一步提升 1.5×，总体实验在不同数据集、模型规模和 harness 上均保持稳定加速。

**⚠️ 局限性**

局限性：1) 依赖对层格式的自定义实现，兼容性受限；2) 弹性资源分配仍基于阈值，极端突发需求可能导致调度延迟；3) 对分布式缓存一致性与网络延迟的假设未在极大规模集群下彻底验证；4) 对多任务并发的资源争用模型尚未覆盖更复杂的动态工作负载。

---

## 219. EviDETR: Preserving Query-Relevant Temporal Evidence for Moment Retrieval and Highlight Detection

**arXiv ID:** 2609.30724 | [PDF](https://arxiv.org/pdf/2609.30724v1)

**作者:** Haoran Sun `[一作]` (Beijing Normal-Hong Kong Baptist University), Shuqi Wang `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EviDETR框架，联合完成视频时刻检索和高光检测任务。

**💡 创新点**

核心创新在于三项证据保留模块：语义感知特征重加权(SFR)提升查询相关表示，时序Top‑2混合专家(TTop2MoE)解码器实现查询自适应稀疏计算，以及MR‑to‑HD(MR2HD)模块将检索级证据直接传递给高光预测。

**🔧 技术方法**

技术上基于DETR式端到端Transformer，结合跨模态注意、稀疏混合专家、ROIAlign多尺度聚合、以及跨任务的证据融合。

**📊 数据集**

使用CLIP+SlowFast特征，在QVHighlights、TACoS和Charades‑STA三个公开视频时序基准上进行实验。

**📈 对比分析**

与现有方法比较，EviDETR在QVHighlights上取得最高的R1@0.5/0.7和HD-mAP/HIT@1，平均mAP仅次于少数顶尖模型；在TACoS和Charades‑STA上也持续领先，证明了跨数据集迁移能力。

**⚠️ 局限性**

局限性包括对大型预训练模型的高度依赖，模型训练成本较高；未对更小规模或低资源场景进行评估，且仅在视频级别进行了证据融合，尚未探索更细粒度的时段级联机制。

---

## 220. PolicyAttention: Softmax Attention Implements Policy Mirror Descent for Closed-Loop Control

**arXiv ID:** 2609.30500 | [PDF](https://arxiv.org/pdf/2609.30500v1)

**作者:** Yuhe Sui `[一作]` (Quantitative Research Society), Shufang Chen `[通讯]` (Quantitative Research Society)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种基于因果softmax注意力的Transformer实现负熵PMD更新，并验证其闭环控制性能。

**💡 创新点**

创新点在于提出固定的因果softmax解码器实现PMD actor–one-step critic循环，给出返回策略残差定理，并训练Transformer学会该机制。

**🔧 技术方法**

使用了因果Transformer架构（pre‑LayerNorm、logπ+ηQ注意力分数）、有限残差分析、返回策略控制定理、近似变分目标以及对照基准。

**📊 数据集**

在合成的离散MDP上进行实验，包含不同状态/动作数（S=4,8,16）的随机环境，用作训练和评估数据集。

**📈 对比分析**

在统一的“common harness”下与Exact PMD、学习PMD目标、Liang–Lai线性注意力actor–critic和Algorithm Distillation 进行比较；PolicyAttention在T=20时返回策略损失比两者低18–28倍，并保持在1.5×oracle的可用性范围内。

**⚠️ 局限性**

局限性包括依赖精确的一步模型批判（非采样）、仅在合成表格域上验证、需要为更大状态空间重新训练，以及未证明SGD能够恢复构造的稀疏电路。

---

## 221. Perspectives on Sustainable Computational Science and Engineering

**arXiv ID:** 2609.30389 | [PDF](https://arxiv.org/pdf/2609.30389v1)

**作者:** Julia Kowalski `[一作]` (RWTH Aachen University), Anil Yildiz `[通讯]` (RWTH Aachen University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文阐述了可持续计算科学与工程的双重框架，将CSE视为实现可持续发展工具并探讨自身的能源、软件和长期价值维度，提出一系列针对科研机构、基金、中心、行业等利益相关者的可操作性建议。

**💡 创新点**

创新点在于将可持续性拆解为“可持续计算”和“可持续软件”两大支柱，并通过实际项目示例与政策建议系统化、跨学科地将可持续性嵌入CSE研究与实践。

**🔧 技术方法**

采用了多学科方法：算法优化、混合精度、适应性网格、自动微分、性能可视化工具与开源框架（如Trixi.jl、CODA、AMReX、t8code），并结合能耗监测与生命周期评估。

**📊 数据集**

无专门实验数据集，主要引用航空、风电等应用领域的案例与公开软件/模型的实验结果。

**📈 对比分析**

文章未进行直接数值比较，而是通过案例说明不同技术组合（如GPU加速、混合精度、模型层级）在能耗、性能与可持续性上的潜在提升，并强调需开发统一指标。

**⚠️ 局限性**

局限性在于缺乏统一、可度量的可持续性指标与量化评估框架，实践建议多基于经验且需进一步验证，且未深入探讨生成式AI等新技术对可持续性的双向影响。

---

## 222. Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study

**arXiv ID:** 2609.30553 | [PDF](https://arxiv.org/pdf/2609.30553v1)

**作者:** Soumen Garai `[一作]` (National Institute of Technology Durgapur), Suman Samui `[通讯]` (National Institute of Technology Durgapur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于教师引导的低保真度评估框架TGL-NSGA-II，用于解决高昂评估成本的组合约束多目标进化优化问题，尤其在TinyML架构搜索中表现突出。

**💡 创新点**

创新点在于：①将低保真度评估转化为可靠排序任务；②设计KD-Lite短期知识蒸馏与分层抽样相结合的评估方法；③提出针对系统性偏差和采样方差的理论界定，推导出期望Kendall‑τ、误排概率和第一非支配前沿扰动的上界；④基于理论推导给出融合权重和自适应蒸馏系数的实施规则。

**🔧 技术方法**

技术手段包括：教师模型的Monte Carlo dropout确定样本难度并划分联合分层；KD-Lite短期蒸馏训练与独立分层评估集的组合；高斯过程GP代理融合；基于残差方差和相关性的自适应权重计算；以及对评估方差、系统性偏差和排名逆转概率的统计分析。

**📊 数据集**

在实验中使用了两个音频分类数据集：Google Speech Commands v2（关键词识别）和BirdCLEF 2021（鸟鸣识别），构建了约10⁵规模的离散网络架构搜索空间。

**📈 对比分析**

与传统NSGA-II、MOBO、SA‑NSGA-II以及零成本代理等基线相比，TGL‑NSGA-II在GSC任务上平均超越对手的HV并降低GD；在BirdCLEF任务上实现最低平均FPR和较高的可行率；在全局评估预算相同的条件下，TGL‑NSGA-II实现了约2.2倍的加速。

**⚠️ 局限性**

局限性包括：理论仅适用于固定候选群体，未给出整体进化轨迹的收敛保证；依赖预训练教师和小样本 pilot 评估，教师匹配不足会导致偏差；实验仅覆盖两类音频任务，跨模态推广尚未验证；对GP多目标代理的多目标性和精度缺乏深入分析。

---

## 223. When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess

**arXiv ID:** 2609.30328 | [PDF](https://arxiv.org/pdf/2609.30328v1)

**作者:** Salma Roshdy Aly `[一作]` (University of Windsor), Ziad Kobti `[通讯]` (University of Windsor)

**通讯引用:** 1103 | [OpenAlex ID](https://openalex.org/A5056407977)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模型管道 MARCH 在代码评审中的表现进行实验验证，并提出门控机制以识别缺乏判断依据的情况

**💡 创新点**

首次系统阐明证据必须满足“独立性”与“差异性”两条条件，证明其缺失会导致判断失效，并提出基于日志的无须额外标签的门控方法

**🔧 技术方法**

采用三阶段代理架构（solver、proposer、checker）与 Qwen3 系列 LLM 进行问题解析、可检验命题生成与独立验证，并通过日志信息实现门控

**📊 数据集**

使用 CodeJudgeBench（含多生成器、不同难度的双解题对）和 HumanEvalFix（人工故障注入）两个公开基准数据集进行评测

**📈 对比分析**

与单一模型直接判断结果比较：单模型准确率73.4%，门控后多模型管道在保留比较的前提下从20.7%提升至36.9%，显著提高但仍未达标

**⚠️ 局限性**

门控仅减少误判且提升精度，整体准确率仍远低于单模型；当生成器产出的解答在证据上完全相同时，管道无法区分，且对更高难度任务效果有限

---

## 224. HyQDB: LLM-Assisted Debugging for Hybrid Quantum Workflows

**arXiv ID:** 2609.30313 | [PDF](https://arxiv.org/pdf/2609.30313v1)

**作者:** Charlie Campbell `[一作]` (Imperial College London), Hongxiang Fan `[通讯]` (Imperial College London)

**通讯引用:** 936 | [OpenAlex ID](https://openalex.org/A5057043409)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对混合量子-经典程序中常见的无声错误，提出了 HyQDB，一种两层式 LLM 辅助调试框架，能够通过确定性分析器提供硬件、优化和物理证据来定位机械性错误，并在无证据时升级至意图重构层解决概念性错误。

**💡 创新点**

创新点在于：①将硬件、优化和物理三种确定性分析器与 LLM 结合，实现证据注入式调试；②引入“无证据门”作为切换机制，依据错误类型自动分配修复策略；③构建基于专家分类的 QFaultBench 基准，填补公共数据集对无声错误评估的空白。

**🔧 技术方法**

技术方法包括：LLM（如 GPT‑4 等）在接收结构化 JSON 证据后生成修复建议；硬件检测器扫描目标设备与门集；优化检测器结合 AST 静态分析与运行时轨迹监测；物理检测器识别问题域并检查量子不变量；意图重构层通过自然语言推理恢复程序意图并对比实现差异。

**📊 数据集**

使用了两部分数据集：①QFaultBench 开发集（34 例）和保留集（28 例）共 62 个手工注入的无声错误任务，覆盖 10 种真实 PennyLane 工作负载；②对每个任务进行隐藏断言验证，保证错误确实导致程序输出错误。

**📈 对比分析**

对比方法：基准 LLM 纯粹修复（45%）→只使用确定性分析器（60%）→完整 HyQDB（75%）。实验显示在开发集和保留集上都实现了显著提升，尤其是机械性错误通过证据注入提升 20% 的准确率；概念性错误仅通过意图重构可修复，提升显著。Token 费用上，机械错误更节省；概念性错误因需额外调用意图重构，token 成本提升 2‑4 倍。

**⚠️ 局限性**

局限性包括：①基准任务仅 62 例，某些错误类别样本不足；②检测阈值基于开发集调优，迁移到保留集时误报/漏报；③确定性检测器覆盖有限，未能识别所有机械错误，导致误路由到意图重构层。

---

## 225. Guarded Gradient-Based Activation Steering of Shutdown Responses in Qwen3.5-0.8B: A Minimum-Step Policy

**arXiv ID:** 2609.30326 | [PDF](https://arxiv.org/pdf/2609.30326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 226. Selective Amortization of Full-Budget Counterfactual Reasoning for Visual Token Communication

**arXiv ID:** 2609.30756 | [PDF](https://arxiv.org/pdf/2609.30756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 227. TinyCVIO: A Constellation-Aided Visual-Inertial Odometry System for Nanodrones

**arXiv ID:** 2609.30358 | [PDF](https://arxiv.org/pdf/2609.30358v1)

**作者:** Derin Ozturk `[一作]` (Cornell University), Christopher Batten `[通讯]` (Cornell University)

**通讯引用:** 3060 | [OpenAlex ID](https://openalex.org/A5091660287)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在配备RP2354微控制器的小型无人机上，开发了TinyCVIO视觉惯性里程计系统，利用预设几何的LED阵列实现实时定位；

**💡 创新点**

创新点包括将LED阵列的刚性几何嵌入MSCKF测量模型、采用流式QR压缩保持固定状态大小，以及利用PIO实现低功耗实时图像预处理；

**🔧 技术方法**

技术手段包括双核并行前端/后端、光阑阈值事件聚类、IPPE姿态估计、Rolling‑Shutter补偿、Cauchy加权门控和滚动窗口克隆；

**📊 数据集**

使用数据集包括：在Webots中生成的七条仿真轨迹；手持HITL实验结合Vicon运动捕捉的多格LED阵列；以及在Crazyflie无人机上进行的三组不同速度的figure‑eight飞行；

**📈 对比分析**

与独立点测量模型和平面点模型比较，Rigid‑Board模型在检测噪声、克隆窗口、LED预算等多项实验中平均A、TE降低约27%，RMSE在10段内保持0.5–0.6%相对姿态误差，估计延迟保持15–16 ms；

**⚠️ 局限性**

局限性主要在于需预先布置已知几何且平面水平的LED阵列，且对LED布置误差和镜头畸变较敏感，无法直接应用于无预置基站或非平面场景。

---

## 228. Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition

**arXiv ID:** 2609.30439 | [PDF](https://arxiv.org/pdf/2609.30439v1)

**作者:** Bo Su `[一作]` (Indiana University), Thai Le `[通讯]` (Indiana University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现目标说话人去学习（TSU-ASR）任务及轻量级可插拔的Enrollment-Conditioned Gating (ECG) 模块，使得在多说话人会议语音识别中能够动态屏蔽指定说话人的内容，同时保持其他说话人的转录与说话时间。

**💡 创新点**

创新点在于：① 设计基于说话人声纹匹配的门控机制，能够在保持双流LLM ASR结构不变的前提下，仅抑制语义流中的受保护说话人信息；② 仅训练轻量级门控模块，即可在推理时支持未见过的说话人，无需重新训练底层模型；③ 将机器无学习理念与大语言模型结合，满足实时动态opt‑out需求。

**🔧 技术方法**

使用技术包括：双流 Zipformer 编码器（语义流与声纹流）+ Qwen2.5-7B LLM；门控模块采用两层 MLP 结合 cos 相似度计算说话人匹配分数；训练时的联合损失为交叉熵 + 二元交叉熵；评估指标采用最长公共子序列（LCS）计算 CLR，并使用 cpWER-R / cpCER-R 评估保留说话人转录质量。

**📊 数据集**

使用的数据集为：AMI（英语远场会议录音）和 AliMeeting（中文会议录音）。在两个数据集上分别划分训练/测试集，保证测试集中说话人不与训练集重叠，测试时随机选取 10% 语音时长并设定 25% 说话人为 opt‑out。

**📈 对比分析**

与未加 ECG 的基线模型进行对比，主要指标为 CLR‑rare、CLR‑all、cpWER‑R 和 cpCER‑R。实验显示，ECG 能将 AMI 的 CLR‑rare 从 67.0% 降至 41.6%，AliMeeting 从 72.3% 降至 27.3%，同时保留说话人的转录错误率保持在与基线相近或略有提升的水平。

**⚠️ 局限性**

局限性包括：① 在仅包含受保护说话人的场景下仍存在信息泄露；② 说话人重叠（尤其是短重叠）时抑制效果不完全；③ 需要对每个未见说话人提供声纹样本，若样本不足或质量差则抑制效果下降；④ 目前仅在 AMI 与 AliMeeting 两个数据集验证，需在更大规模、多语种会议场景进一步评估。

---

## 229. Praxis: Distilling Physical Interaction Priors from Egocentric Videos for Generalizable Whole-Body Manipulation

**arXiv ID:** 2609.30735 | [PDF](https://arxiv.org/pdf/2609.30735v1)

**作者:** Shuliang He `[一作]` (Chinese University of Hong Kong), Guiliang Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了Praxis框架，利用单一的egocentric RGB‑D视频示范实现全身移动、姿态校准与抓取的三阶段闭环全身操纵；

**💡 创新点**

其创新点在于将单次人类示范的手腕轨迹与触觉反馈在线重定向，形成可跨物体、跨姿态、跨环境的无任务特定政策迁移体系；

**🔧 技术方法**

技术手段包括视觉‑语言导航(VLN)、基于WBC的闭环姿态校准、MANO/MediaPipe手部重定向、FoundationPose++在线6D物体估计、触觉握力调节及Qwen3.8‑27B在线验证；

**📊 数据集**

实验使用了150条机器人/人类收集的导航轨迹、五个长程全身操纵任务（Hand Over、Pour Water、Pull Drawer、Open Lid、Manipulate Pipette）的单一演示，以及公开的视觉语言与姿态数据集；

**📈 对比分析**

与π_0.5、YOTO、OKAMI三种基线在33次试验中对比，Praxis平均成功率为76.97%（YOTO 66.67%，OKAMI 37.58%），在3 m/5 m距离、转向等环境变化中保持高成功率，触觉反馈提升约15%；

**⚠️ 局限性**

限制方面包括对高质量egocentric视频与手腕重定向的依赖、对遮挡/深度误差敏感、缺乏大规模多样化互联网视频训练，难以直接迁移至无标注视频，且对硬件异常的鲁棒性仍有限。

---

## 230. Prompt Injection Detection for Email Agents Through Attack Chain Modeling

**arXiv ID:** 2609.30657 | [PDF](https://arxiv.org/pdf/2609.30657v1)

**作者:** Ahmad Hashmi `[一作]` (Eastern Michigan University), Yunting Yin `[通讯]` (Eastern Michigan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对邮件代理的提示注入检测框架，基于攻击链模型实现分阶段识别与决策。

**💡 创新点**

将提示注入拆解为检索、绕过防御、工具调用、工具参数验证和最终成功四个阶段，并结合文本检测、阶段验证器、规则风险信号以及用户意图一致性，形成多层防御。

**🔧 技术方法**

使用TF-IDF + 线性逻辑回归、阶段级分类器、手工规则、动作/上下文一致性检查，并通过逻辑回归堆叠学习最终决策。

**📊 数据集**

在 LLMail、BIPIA、NotInject、NVIDIA Agentic IPI、PromptShield、ShieldLM、Neuralchemy 等公开提示注入与硬负样本数据集上训练与评估。

**📈 对比分析**

采用随机拆分、阶段迁移、条件阶段、跨数据集、阈值设定等多协议对比实验；在严格阈值下平均 F1 0.406，显著高于五个冻结模型（0.216），但在 NVIDIA 数据集上的召回率仍偏低。

**⚠️ 局限性**

对早期阶段（检索、绕过）检测仍不稳定；攻击者可通过挑选低风险词汇绕过；跨域泛化受限；阈值微调难以同时兼顾低误报与高召回。

---

## 231. "If You're Not Doing It, Somebody Else Is": Active Negotiation and the Invisible Labor of Sustained LLM Use

**arXiv ID:** 2609.30699 | [PDF](https://arxiv.org/pdf/2609.30699v1)

**作者:** Matt Viana `[一作]` (Pennsylvania State University), Dana Calacci `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对36名美国研究生的半结构化访谈，探讨他们在认知到LLM存在缺陷后，如何通过风险识别、缓解劳动和正当化三阶段的“主动协商”循环，持续使用LLM，并提出了这一循环模型。

**💡 创新点**

提出主动协商（Active Negotiation）框架，重新定义LLM持续使用为“隐形劳动”而非单纯满意度；揭示多维度（实用、内部认知、社会）交织的协商过程；阐释认知切换与结构性压迫对使用行为的影响；引入语言平等作为非英语使用者的正当化手段。

**🔧 技术方法**

采用定性研究方法：半结构化访谈、主题分析、代码书编制，并使用Taguette进行数据管理；没有计算机实验或机器学习技术。

**📊 数据集**

使用来自一所美国研究型大学的36名研究生（21名母语为英语、15名非英语母语）的访谈数据，按专业和英语熟练度分层，收集个人使用习惯、风险认知、缓解与正当化策略。

**📈 对比分析**

研究不涉及算法性能比较；通过对访谈数据的主题归纳，形成框架和案例分析；没有使用基准测试或量化指标。

**⚠️ 局限性**

局限：样本仅来自单一高校、规模有限；EFL/非EFL二分可能忽视语言多样性；访谈自我报告易受社会期望偏差；研究时间截至2025年9月，技术与政策快速演变；缺乏成员核查与客观验证；仅适用于学术/高技能知识工作场景。

---

## 232. VkVIO: Cross-platform GPU Acceleration for Visual-Inertial Odometry with Vulkan

**arXiv ID:** 2609.30459 | [PDF](https://arxiv.org/pdf/2609.30459v1)

**作者:** Ole Hoffmann `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**通讯引用:** 54683 | [OpenAlex ID](https://openalex.org/A5087710605)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文实现了基于Vulkan的跨平台GPU加速视觉惯性里程计（VIO）前端，支持Apple、Nvidia和ARM等多种GPU硬件；

**💡 创新点**

创新点在于首次将Vulkan计算着色器应用于Stereo VIO前端，并通过子组和共享内存实现高效并行；

**🔧 技术方法**

采用Vulkan、SPIR‑V、子组操作、共享内存等技术，将Basalt前端重构为GPU实现；

**📊 数据集**

使用了EuRoC、TUM‑VI、Monado SLAM、Hilti Challenge等多种视觉惯性数据集进行评测；

**📈 对比分析**

与Basalt CPU前端、CUDA加速前端（Jetson‑SLAM、FastTrack）以及不同厂商GPU对比，VkVIO在Apple M3、Nvidia RTX 3070和Radxa A7Z上分别实现了约1.4×–11×的速度提升，保持精度同时功耗更低、帧时间更稳定；

**⚠️ 局限性**

局限包括对子组支持的依赖（Radxa fallback）、对极高分辨率或多相机系统的可扩展性待验证，以及对后端优化窗口等参数的敏感性。

---

## 233. Learning-Based Pressure Predictive Control of a Vertebraic Soft Robotic Tail

**arXiv ID:** 2609.30479 | [PDF](https://arxiv.org/pdf/2609.30479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 234. Spectral Feedback for Test-Time Alignment of Protein Diffusion Models

**arXiv ID:** 2609.30456 | [PDF](https://arxiv.org/pdf/2609.30456v1)

**作者:** Shai Dickman `[一作]`, Kannan Ramchandran `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该论文的内容缺失，无法确定具体研究内容。

**💡 创新点**

无法确认创新点。

**🔧 技术方法**

无法确认所使用的技术。

**📊 数据集**

无法确认使用的数据集。

**📈 对比分析**

无法比较方法或评估性能。

**⚠️ 局限性**

主要限制是缺乏完整论文信息。

---

## 235. Training-Free Bottleneck Width Planning for Convolutional Autoencoders

**arXiv ID:** 2609.30755 | [PDF](https://arxiv.org/pdf/2609.30755v1)

**作者:** Guannan Guo `[一作]` `[通讯]` (Beihang University), Guannan Guo (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用训练集的像素协方差谱和指定的NMSE阈值，预测卷积瓶颈层的通道数并给出激活-参数Pareto前沿，无需任何网络训练。

**💡 创新点**

提出了无训练、精确的多尺度谱率失真规则，证明了共享线性块卷积自编码器的最优通道数，并揭示嵌套尺度优势。

**🔧 技术方法**

采用多尺度图像块提取、协方差估计、PCA特征分解、谱尾规则、线性自编码器理论、非线性验证自编码器、U‑Net实验以及Bootstrap不确定性评估。

**📊 数据集**

在13个灰度图像数据集上验证，包括MNIST、KMNIST、Fashion‑MNIST、CIFAR‑10/100、ChestMNIST、PneumoniaMNIST、BreastMNIST、OrganAMNIST、RetinaMNIST、BloodMNIST、Oxford‑IIIT Pet、EuroSAT。

**📈 对比分析**

与全局PCA、块PCA、网格搜索和Least‑Volume基线对比，实验显示预测误差平均0.84% MAPE，十个数据集预测精确，剩余三个误差仅一通道，且在不同重建阈值下可量化非线性节省。

**⚠️ 局限性**

受限于共享线性块卷积、灰度平均MSE目标、架构依赖的切点设定以及有限样本导致的谱尾波动，且未覆盖重叠块、彩色图像或感知/语义损失等场景。

---

## 236. Communication-Aware Model Distributed Inference via Latent Representation Compression

**arXiv ID:** 2609.30413 | [PDF](https://arxiv.org/pdf/2609.30413v1)

**作者:** Peyman Gholami `[一作]` (University of Illinois Chicago), Ness Shroff `[通讯]` (Ohio State University)

**通讯引用:** 21670 | [OpenAlex ID](https://openalex.org/A5035752536)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在资源受限的边缘网络中进行分布式推理，设计一种动态激活压缩策略以满足吞吐量 QoS 约束并最大化模型准确率。

**💡 创新点**

提出通信感知的可变压缩因子优化框架；在已知信道状态信息时给出单任务闭式解和多任务凸化水填策略；在无 CSI 的情况下使用随机对偶下降和块坐标下降，证明长期延迟约束可被严格满足并给出最优性间隙。

**🔧 技术方法**

采用 Pipeline Parallelism、激活压缩（Top‑k 稀疏、均匀量化、LLM.int8）、凸优化、水填分配、随机对偶下降、BCD、Lyapunov 稳定性分析，以及 Stein 核平滑对准确率曲线进行估计。

**📊 数据集**

使用视觉任务 ResNet‑56/CIFAR‑10、MLP/MNIST、LLM 等模型；在模拟的线性拓扑、Raspberry Pi、Jetson Orin Nano 及 GPU 多机测试平台上评估。

**📈 对比分析**

与无压缩、全压缩、固定压缩、均匀压缩、CSI‑aware 线性分配、无 CSI 历史估计等基线对比，实验表明 CSI‑aware 闭式解或无 CSI 随机对偶下降方案在保持吞吐量约束的前提下，准确率提升 5–10%，延迟下降 30–50%，多任务场景下实现了更高的聚合效用。

**⚠️ 局限性**

局限性包括：1) 只能在可压缩范围内工作，链路中断时需切换或退出；2) 需要准确的准确率–压缩曲线，曲线误差会影响最优解；3) 对信道估计误差敏感；4) 多任务 BCD 只保证局部最优，可能导致次优资源分配。

---

## 237. Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents

**arXiv ID:** 2609.30293 | [PDF](https://arxiv.org/pdf/2609.30293v1)

**作者:** Justice Owusu Agyemang `[一作]` (Kwame Nkrumah University of Science and Technology), Jerry John Kponyo `[通讯]` (Kwame Nkrumah University of Science and Technology)

**通讯引用:** 758 | [OpenAlex ID](https://openalex.org/A5077159920)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Cartograph，一个联邦 MCP 代理，通过操作员签名的能力卡、两阶段检索与 RIFT 三层聚类，实现 O(k) 的工具发现与进阶披露，显著减少令牌消耗与代理延迟。

**💡 创新点**

创新点包括：① 将工具发现从 O(n) 遍历转化为 O(k) 检索；② 采用操作员签名（Ed25519）保证描述可信度；③ RIFT 三层冲突检测（密度聚类、边缘分析、词汇诊断）定位高风险工具集；④ 通过代理层控制工具描述与 provenance，兼容代码执行方式。

**🔧 技术方法**

使用技术：Ed25519 签名、MiniLM‑L6‑v2 句子嵌入、两阶段检索（服务器层与工具层）、RIFT 3‑层聚类算法、Python 3.11 端点实现、Gateway 一shot stdio 调用、JSON schema、Pydantic 数据模型。

**📊 数据集**

数据集：22 台 MCP 服务器共 374 个安全研究工具（包括 Ghidra、IDA、Daedalus、Burp、Chrome DevTools 等），49 条人工构造查询基准，以及 119 台服务器的 LLM 生成卡（仅 7 台服务器）。

**📈 对比分析**

比较方法与性能：与基于 Jaccard 关键字检索和随机基线比较；R@5 为 0.816（vs 0.592 和 0.013），令牌消耗从 42,450 降至 475（98.8% 节省），RIFT 检测 49 个混淆集（4 个 HIGH 风险），Gateway 延迟仅 +0.8%（≈5 ms）。

**⚠️ 局限性**

局限性：LLM 卡仅覆盖 7/22 服务器，混合卡导致中心偏移；基准规模小且人工构造；未评估更强嵌入模型或更大工具集合；Gateway 未实现连接池和多协议（HTTP/SSE）支持；跨组织联邦需要 PKI/密钥轮换；尚未测试多操作员环境。

---

## 238. Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval

**arXiv ID:** 2609.30601 | [PDF](https://arxiv.org/pdf/2609.30601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 239. In-Context Binding Capacity in Language Models

**arXiv ID:** 2609.30634 | [PDF](https://arxiv.org/pdf/2609.30634v1)

**作者:** Manas Venkata Sai Ravulapalli `[一作]` (Efficient Computation Inc.), Samrath Singh Chadha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模语言模型中测量和量化“绑定容量”，即模型在上下文中同时记住多少个实体-值绑定，并评估在不同干扰强度下的鲁棒性。

**💡 创新点**

提出了规模-容量的幂律关系（K_50 ≈ c·N^α），证明预训练配方在规模不变时对容量影响不显著，并且直接任务训练可显著超过零样本容量预测，进一步将干扰效应分解为单绑定基准降低、容量阈值移动和曲线斜率变化三种机制。

**🔧 技术方法**

采用阈值分析与连续回忆曲线拟合（Logistic回归），用线性回归检验规模与容量的关系，计算有效维度、包装效率等几何量，并对干扰块进行多维度评估。

**📊 数据集**

使用自制的绑定任务数据集（K个实体-值绑定 + D个干扰标记 + 查询），在10个模型族共计近20个公开模型上运行（≤3B模型12个，扩展到12B的8个模型）。

**📈 对比分析**

通过比较不同模型的K_50与参数规模、与直接训练模型的K_50，以及干扰块前后的容量变化，发现：
- K_50 随参数规模呈幂律增长，α≈0.2（具体数值未给出）。
- 预训练配方在规模控制后对容量无显著影响；但在12B规模范围内仍出现配方差异。
- 直接训练模型在同一容量指标下的K_50可达到零样本预测的4.98倍（取决于阈值定义）。
- 干扰导致单绑定基准下降、容量阈值约减半，但在自我基准归一化下阈值变化不明显。

**⚠️ 局限性**

局限性包括：
- 预训练配方效应仅为观察性关联，缺乏因果证明。
- K_50 被右截断（受实体池大小限制），低端信息更具说明力。
- 连续回忆曲线仅在≤3B规模下测得，对更大模型的行为未知。
- 任务仅测量单次查询的检索，未覆盖状态更新或多步推理。
- 干扰分解基于阈值定义，可能隐藏更细粒度的机制。
- 交叉模型几何量与容量的相关性未提供机制解释。

---

## 240. From Mono to Stereo: Accelerating Binocular Gaussian Splatting via Reprojection and Selective Patching

**arXiv ID:** 2609.30741 | [PDF](https://arxiv.org/pdf/2609.30741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. Recommendation World Models for Future-State Control

**arXiv ID:** 2609.30711 | [PDF](https://arxiv.org/pdf/2609.30711v1)

**作者:** Jinfeng Xu `[一作]` (University of British Columbia), Victor C. M. Leung `[通讯]` (University of British Columbia)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于训练好的顺序推荐器的“世界模型接口”，通过局部后向预测未来状态并在保持原始推荐器性能的前提下选择更符合目标的展示卡片。

**💡 创新点**

创新点在于：①把推荐器的“锚点”slate作为参考，构造可替代slate并利用局部动作-状态预测评估其对未来统计量的影响；②设计三重门控（效用、目标增益、风险）实现安全的目标导向控制；③在已训练模型上实现可迁移、可解释的选择层。

**🔧 技术方法**

技术包括：基于序列编码的顺序推荐器（GRU4Rec、SASRec、BERT4Rec、NextItNet 等）；局部动作-状态预测（线性回归、Ridge 回归）；目标与效用估计（Recall@20/NDCG@20、未来状态 L1 损失）；风险判别器（逻辑回归）；以及在重放与模拟器中的闭环评估。

**📊 数据集**

使用 MovieLens‑25M（电影评分数据）和 KuaiRand‑Pure（短视频点击/长观看标签数据）做时间序列重放评估；使用 KuaiSim（基于图的短视频用户模拟器）做10 步闭环交互评估。

**📈 对比分析**

与十二种不同的顺序推荐器基线（递归、注意力、卷积、自监督、多兴趣、扩散、流式等）进行匹配对比；在重放中，添加接口后 Recall@20 与 NDCG@20 均提升 5–6% 以上，未来状态 L1 损失下降 0.04–0.08；在闭环中，接口实现的目标控制声明成功率为 100%，在不降低点击率的前提下将目标 L1 降低 0.37，优于仅使用状态预测或语义控制的方案。

**⚠️ 局限性**

局限性包括：①只评估了一步预测的精度，长周期的状态误差未深入分析；②风险门控阈值需在校准集上手工调参，迁移到新业务时可能需要重新校准；③在极端稀缺目标或高度动态环境下，候选滑动窗口可能无法覆盖足够多的可行方案，导致回退频率上升。

---

## 242. WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving

**arXiv ID:** 2609.30436 | [PDF](https://arxiv.org/pdf/2609.30436v1)

**作者:** Mingkai Jia `[一作]` (Hong Kong University of Science and Technology), Wei Yin `[通讯]` (Horizon Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了在冻结的驾驶世界模型基础上对轨迹进行对齐，学习一套紧凑的轨迹潜在空间，从而实现更高效、更安全的轨迹规划。

**💡 创新点**

创新点在于：①无需改动已有世界模型，即可通过对齐学习把视觉语义迁移到轨迹潜在空间；②双分支轨迹自编码器同时保留几何重构与语义对齐；③使用 CLIP‑style 对齐损失将世界模型特征注入轨迹表示。

**🔧 技术方法**

主要技术包括：双分支轨迹自编码器、世界模型对齐损失（WALT）、流匹配（flow‑matching）轨迹头、JEPA 与 REPA 的对比实验。

**📊 数据集**

使用 NAVSIMv1 与 NAVSIMv2 作为评测数据集。

**📈 对比分析**

与原始 waypoint 基线和其他自监督方法（JEPA‑Traj、REPA‑Traj）对比，WALT 在 NAVSIMv1 上 PDMS 提升 0.4 点（89.4→89.8），在 NAVSIMv2 上 EPDMS 提升 0.6 点（87.3→87.9），同时在生成阶段 FLOPs 减少 30.5%。

**⚠️ 局限性**

局限性：依赖于已有冻结的世界模型，无法直接改进或替代该模型；对齐策略在不同任务或场景下的泛化性尚待验证；与原始 waypoint 的分辨率差异导致部分细节信息损失。

---

## 243. Learning coarse-step dynamics and internal mechanical response with graph networks

**arXiv ID:** 2609.30344 | [PDF](https://arxiv.org/pdf/2609.30344v1)

**作者:** Vinay Sharma `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个基于图神经网络的框架，利用半隐式 Newmark 更新和秩一虚拟枢纽，实现从粗时间步观测轨迹中无监督推断内部机械量，并进行长期动力学预测。

**💡 创新点**

创新点在于将机械学的两大结构——半隐式 Newmark 更新和全局耦合的秩一虚拟枢纽—融入图神经网络，使得推断的力、扭矩和响应算子在状态更新中直接参与，可解释且可访问；同时实现了在粗时间步下的稳定长程预测。

**🔧 技术方法**

使用图神经网络（E(3)-equivariant 结构）进行特征编码，学习边缘的线性与角动量通量及位置/速度响应算子；结合半隐式 Newmark 迭代和秩一虚拟枢纽的权重耦合实现更新；训练仅依赖轨迹误差。

**📊 数据集**

实验数据集包括：有限元夹持梁（FEniCS 生成）、人类运动捕捉（CMU 运动捕捉数据库）、蛋白质动力学（腺苷酸激酶开放-闭合数据）以及步态生物力学（仪器化跑步机记录）。

**📈 对比分析**

与 GNS、MGN、EGNN、EGHN、EGNO、IGNS 等基线进行对比，采用自回归预测评估。结果显示在梁、运动和蛋白质任务中保持有限误差；在步态实验中推断的膝/髋关节矩与独立逆动力学高度相关（r≈0.94/0.87）。相较基线，加入枢纽和半隐式更新可将误差从约3%降至≈0.5%。

**⚠️ 局限性**

局限性包括：秩一枢纽对长梁或强非局部耦合效果有限；局部求解未保证精确动量守恒；学习到的响应算子只能给出相对指标，缺乏绝对尺度；未直接监督外部力，导致量纲缺失。

---

## 244. ProCAP: Probabilistic Cross-Attentive Prompt Learning for Vision-Language Models

**arXiv ID:** 2609.30434 | [PDF](https://arxiv.org/pdf/2609.30434v1)

**作者:** Hiwa Azeez Abbas `[一作]` (University of Kurdistan), Moloud Abdar `[通讯]` (University of Queensland)

**通讯引用:** 9690 | [OpenAlex ID](https://openalex.org/A5014000715)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

ProCAP提出了一种在冻结CLIP模型下的概率交叉注意力提示学习框架，用于在少样本和分布迁移条件下进行视觉‑语言模型的适配；

**💡 创新点**

其创新点在于：①通过堆叠双向多头交叉注意力将视觉与文本提示实现紧密两向交互；②将提示参数化为高斯分布并加入KL与L2正则化以稳定低样本优化；③引入对称InfoNCE头，在低维空间对齐跨模态图像特征与类别文本表示；

**🔧 技术方法**

采用了提示学习、交叉注意力、Gaussian提示参数化、对称InfoNCE对比损失、KL/L2正则化以及冻结CLIP骨干网络等技术；

**📊 数据集**

在11个经典图像分类基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、UCF101、DTD、EuroSAT），以及ImageNet的四个分布偏移版本（ImageNetV2、ImageNet‑Sketch、ImageNet‑A、ImageNet‑R）和十个目标数据集的跨数据集迁移任务上进行评估；

**📈 对比分析**

与MaPLe、CoOp、CoCoOp等提示学习基线以及2026年最新方法进行比较，ProCAP在16-shot基于新旧类别迁移任务中实现了81.33的调和平均得分，显著优于MaPLe；在域泛化和跨数据集迁移任务中也保持竞争力；

**⚠️ 局限性**

局限性包括：仍依赖CLIP预训练知识，可能继承其偏见；在极少样本或极端分布漂移场景下提升有限；模型规模和计算量相对较大，且主要针对图像分类任务，其他任务的适用性仍需进一步验证。

---

## 245. The Interviewer's Perspective: Unpacking the Impact of Real-Time AI Interviewing Assistance on Social Dynamics

**arXiv ID:** 2609.30388 | [PDF](https://arxiv.org/pdf/2609.30388v1)

**作者:** Zhe Liu `[一作]` (University of British Columbia), Joanna McGrenere `[通讯]` (University of British Columbia)

**通讯引用:** 4663 | [OpenAlex ID](https://openalex.org/A5016459516)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建实时AI辅助系统ProbeAssist，研究访谈者在半结构化访谈中使用实时AI提问辅助的体验与行为。

**💡 创新点**

创新点在于首次系统性探讨访谈者、受访者与AI三角互动的社交动态，提出缓解社交压力、在AI贴合度与探索性之间动态平衡、保持访谈者人际存在感的三条设计原则。

**🔧 技术方法**

使用OpenAI Realtime API进行即时语音转写与生成提问建议，并与WebRTC、Next.js等技术构建高保真、可定制的实时AI辅助原型。

**📊 数据集**

实验采用三份相似结构的访谈指南（时间管理、团队项目经验、主管关系）作为实验材料，未使用公开数据集，而是利用LLM在现场生成提问。

**📈 对比分析**

通过对照结构化观察（CSO）实验，18名访谈者在三次访谈中分别体验无AI、受控型AI与表达型AI；结果显示AI能显著提高提问数量和数据质量、提升满意度，但在延迟和社交成本方面存在不足。

**⚠️ 局限性**

限制包括样本规模有限、访谈为角色扮演且缺乏真实受访者体验、未在真实项目中验证系统效果，以及技术延迟仍影响交互流畅性。

---

## 246. HCOE: Hyperbolic Clinical Ontology Embeddings from Biomedical Language Models

**arXiv ID:** 2609.30763 | [PDF](https://arxiv.org/pdf/2609.30763v1)

**作者:** Yixuan Li `[一作]` (McGill University), Ziyang Song `[通讯]` (Ohio University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Hyperbolic Clinical Ontology Embeddings (HCOE)，将冻结的 BioBERT 表示映射到 Poincaré 球，并结合父/子层级对比学习和路径聚合生成层级感知的医疗代码嵌入；

**💡 创新点**

创新点在于：①将预训练医学语言模型的语义与临床本体层级结构融合到超bolic 空间；②设计双侧（父侧+子侧）对比学习目标；③利用粗细层级路径聚合通过 Möbius 加法聚合多级嵌入；

**🔧 技术方法**

使用技术包括：BioBERT（冻结），Poincaré（超bolic）嵌入，Möbius 加法，父/子对比学习，线性投影与指数映射，实验评估在 ICD/ATC 关系预测、PheCode 转移和 MIMIC‑IV 预测任务；

**📊 数据集**

使用的数据集包括：ICD‑10/ICD‑9、ATC、CCS 本体；MIMIC‑IV 住院 EHR；以及通过 CCS 迁移到 PheCode 的数据；

**📈 对比分析**

与 BioBERT、SapBERT、HiTs、OnT、RotatE、cui2vec、Poincaré Embedding 等基线在 ICD/ATC 关系预测（F1 最高分别 75.6%/76.3%）、PheCode 迁移（80.6%）、以及 MIMIC‑IV 的死亡、再住院、药物推荐和罕见药物预测（AUPRC/AUROC/Recall@15 均达到或超过最佳基线）进行比较，HCOE 在大多数指标上取得领先；

**⚠️ 局限性**

局限性在于：只使用了 ICD/ATC/CCS 本体，未结合更丰富的 UMLS 或其他知识图谱；对跨编码系统的对齐仍存在挑战；仅在单一 EHR 数据集验证，泛化性待进一步评估；罕见药物预测仍受样本稀缺影响；未来需探索多模态输入。

---

## 247. MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation

**arXiv ID:** 2609.30837 | [PDF](https://arxiv.org/pdf/2609.30837v1)

**作者:** Tianze Xu `[一作]` (Shanghai Jiao Tong University), Gang Yu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MOPD‑Router框架，实现 token 级多教师路由，并在此基础上提出 ExpertAlign 专业对齐度量，消除 prompt 级域标签依赖。

**💡 创新点**

创新点在于：①用 token 级动态加权代替硬域标签路由；②设计了基于教师专精方向与教学方向对齐的 ExpertAlign 指标；③提供可插拔度量接口，支持多种路由策略。

**🔧 技术方法**

采用了 on‑policy distillation、entropy、novelty、cosine 对齐、top‑k 支持统计、共享基模型前向推理等技术，并在 Qwen3 系列 LLM 上实现。

**📊 数据集**

使用 60K 未标注混合数据（30K 对话 + 30K 数学）与 25K 数学、25K 代码、16K 说明性任务的标注数据；在 AIME、HMMT、HumanEval、MBPP、IFEval、IFBench 等公开基准上评测。

**📈 对比分析**

与 Standard MOPD、Open‑MOPD、Mean Aggregation 对比，ExpertAlign 在无标签数据上整体提升 5.9%（Qwen3‑1.7B）/3.8%（Qwen3‑4B），在标注数据上提升 3–5%；在各子任务上均显著优于基线。

**⚠️ 局限性**

局限性包括：①需要共享基模型且需额外前向推理；②计算开销相对基线略高；③对教师池质量和 top‑k 参数敏感；④对极端跨域混合或低资源领域的鲁棒性尚未充分验证。

---

## 248. PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices

**arXiv ID:** 2609.30836 | [PDF](https://arxiv.org/pdf/2609.30836v1)

**作者:** Minghui Yu `[一作]` (Shanghai Jiao Tong University), Gang Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种无训练的、可插拔解码器框架PTC-Decoder，提升小语言模型在离线边缘设备上的工具调用可靠性。

**💡 创新点**

将规划视为工具并强制首步调用，再结合 Token‑level Hard Constraints Decoder（TC‑Decoder）在解码过程中动态约束工具名称，首次实现仅通过解码层面即可提升 SLM 的执行一致性。

**🔧 技术方法**

采用 Plan‑to‑Act 方案、有限状态机约束、token‑level mask、量化小语言模型以及无训练的解码器实现。

**📊 数据集**

在自建的 200 条真实卫星任务基准上进行评测，并在 ToolAlpaca、Seal‑Tools、API‑Bank 等公开工具执行数据集上进行跨域验证。

**📈 对比分析**

使用 LLM‑judge（Ov., R.A., F.R., Rb.）和规则基准（Recall, Precision, F1）进行对比，PTC‑Decoder 在 7 个 SLM 上平均提升 Ov. +1.21、F1 +0.096，最弱模型提升 350% 以上，跨数据集提升 30‑70% 记分。

**⚠️ 局限性**

仍无法保证最终答案准确性或错误恢复，仅约束工具名称未覆盖参数；对更强 LLM 可能抑制创造性；Token 开销约 2.07×，导致推理成本略增。

---

## 249. Crypto-bound identity-verified capability tokens for coordinating distributed AI agents: A proposal

**arXiv ID:** 2609.30824 | [PDF](https://arxiv.org/pdf/2609.30824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 250. Subject-Invariant Cross-Modal Decoding of Perceived Speech from Brain Recordings

**arXiv ID:** 2609.30832 | [PDF](https://arxiv.org/pdf/2609.30832v1)

**作者:** Aoke Zhang `[一作]` (Peking University), Jing Chen `[通讯]` (Peking University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种跨模态、跨受试者的感知语音解码方法SICMD，融合MEG和fMRI信号以提取丰富的时空特征并实现受试者不变的表示。

**💡 创新点**

创新点包括：①将PESA模块与CorrCA算法相结合，实现受试者一致信息的自动提取；②统一跨模态和跨受试者解码框架，显著提升跨受试者迁移性能；③在模型训练阶段引入个人化专精阶段，大幅降低训练步骤和时间。

**🔧 技术方法**

使用的技术包括：PESA（基于位置编码的空间注意力）、1×1卷积、ConvConcatNet的ConvBlock、简单的MLP和线性层进行fMRI编码、fMMF融合方法、CorrCA一致性提取、CLIP损失、Adam优化器等。

**📊 数据集**

使用公开的多模态神经影像数据集（12名受试者，连续中文语音、fMRI TR=0.71s、MEG 306通道、1000Hz），共60个试次。

**📈 对比分析**

与基线方法（fMRI Encoder、Brainmagic、ConvConcatNet、CPSD、fMMF）对比，SICMD在Top‑1、Top‑10和Rankacc上分别提升约10.6%、10.1%和1.7%；同时训练步骤减少88.8%（多受试者）和60.5%（单受试者），表现出显著的性能与效率优势。

**⚠️ 局限性**

局限性：仅在MEG‑fMRI双模数据上验证，受试者样本有限（12人）；未在EEG‑fNIRS等更常见的非侵入性组合上测试；模型复杂度较高，实时部署和大规模应用仍需进一步优化。

---

## 251. CDBG: Causally Motivated Dual-Invariance Learning against Topological and Predictive Shifts in EEG Workload Recognition

**arXiv ID:** 2609.30831 | [PDF](https://arxiv.org/pdf/2609.30831v1)

**作者:** Yuzhe Zhang `[一作]` (Nanjing University of Aeronautics and Astronautics), Daoqiang Zhang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于因果双重不变性学习的脑图框架（CDBG），用于跨受试者的脑电（EEG）工作负荷识别。

**💡 创新点**

创新点包括：①将工作负荷识别建模为同时解决“类条件拓扑偏移”和“预测机制偏移”两类分布偏移；②在图推理阶段使用随机边掩码与工作负荷条件的拉普拉斯谱对齐来提取稀疏且跨受试者稳定的功能子图；③在分类阶段对每个受试者采用Invariant Risk Minimization（IRM），实现预测不变性。

**🔧 技术方法**

主要技术手段：可学习的稀疏图解释器（随机边掩码+Binary Concrete 采样）、拉普拉斯谱对齐（MMD 计算）、IRMA 约束（对标量缩放梯度的一阶约束）以及整体端到端的联合优化。

**📊 数据集**

实验使用三大公开/自建 EEG 数据集：SELF（8名空中交通管制员，59 通道），STEW（48 名受试者，14 通道），EEGMAT（36 名受试者，19 通道）。

**📈 对比分析**

在严格留一受试者测试（LOSO）协议下，与12种图、工作负荷专属及通用 EEG 基线方法比较，CDBG 在 Macro‑F1 上均领先 2.48%–4.23%，并在 Accuracy 上实现显著提升，表明跨受试者泛化能力显著增强。

**⚠️ 局限性**

局限性包括：①需要多个源受试者来构建环境，单受试者或极少受试者的数据可能无法有效训练；②仅在监督式源数据下工作，缺乏无监督或在线适配机制；③目前仅验证于工作负荷识别任务，对其他 EEG 认知任务的适用性尚待进一步研究。

---

## 252. AC Power Flow Contingency Analysis Using a Single Deep Neural Network

**arXiv ID:** 2609.30859 | [PDF](https://arxiv.org/pdf/2609.30859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 253. SUCRe: Selective Uncertainty-Aware Contrastive Representation for Graph Transfer Learning

**arXiv ID:** 2609.30826 | [PDF](https://arxiv.org/pdf/2609.30826v1)

**作者:** Mingcan Wang `[一作]` (Northeastern University), Zhiqiong Wang `[通讯]` (Northeastern University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 SUCRe，一种针对图迁移学习的选择性不确定性感知对比表示方法，通过结构感知熵匹配和域感知半难负样本挖掘，实现源图到目标图的高效、低负迁移特征迁移。

**💡 创新点**

创新点在于（1）引入结构感知熵基匹配差异（SEMD），在特征对齐过程中量化并利用节点不确定性，避免负迁移；（2）提出域感知半难负样本策略，在对比学习中筛选信息量最大的负样本，既提升表示辨别力又降低计算冗余。

**🔧 技术方法**

主要技术包括熵测度、个性化 PageRank 权重、低秩 LoRA 微调、对比学习中的半难负样本挖掘、预训练 GNN 及其冻结+可训练分支等。

**📊 数据集**

在 PubMed、Cora、CiteSeer、Amazon Photo、Amazon Computers、Physics、WikiCS 等标准图数据集上进行实验。

**📈 对比分析**

与 GCN、PPNP、GPPT、DDSM、AdapterGNN、GraphControl、GraphLoRA 等基线比较，SUCRe 在公共、少样本和多源-目标设置下平均排名 1.32，准确率提升最高 4.18%，并在 GPU 内存占用与训练时间上明显优于 GraphLoRA。

**⚠️ 局限性**

局限性包括：仅针对节点分类任务验证，未评估多任务或图级任务；对极端结构差异的跨域迁移仍可能受限；并依赖预训练 GNN 的可用性和源图特征质量。

---

## 254. SeA-RVINS: Semantic-Aware Tightly Coupled RTK-Visual-Inertial System with Correlation-Preserving Robust Estimation for Urban Navigation

**arXiv ID:** 2609.30814 | [PDF](https://arxiv.org/pdf/2609.30814v1)

**作者:** Wang Hu `[一作]`, Bo Wu `[通讯]` (UC Riverside)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为SeA‑RVINS的固定滞后RTK‑视觉‑惯性估计框架，用以在城市环境中实现可靠的绝对姿态估计。

**💡 创新点**

创新点包括：① 语义感知的学习式立体前端过滤不可靠跟踪；② 混合的歧义连续性拓扑，既共享短段歧义状态又通过随机游走因子软链接；③ 保留双差测量共享枢轴相关性的鲁棒估计（批量、标量、潜在枢轴三种配置）。

**🔧 技术方法**

结合了深度学习语义分割（SegFormer）、学习式特征提取与匹配（SuperPoint‑LightGlue）、动态协方差缩放（DCS）、以及整数歧义恢复（IAR）等技术。

**📊 数据集**

使用公开的TEX‑CUP 20 km 城市驾驶数据集进行实验，包含约50% 深层城市区。

**📈 对比分析**

与多种公开基线（RTKLIB‑EX、GVINS、VINS‑Fusion 等）在固定滞后FGO中对比，SeA‑RVINS 在所有配置下均实现 100% 可用率，水平 RMSE 0.38–0.39 m，最大水平误差 1.60 m，显著优于基线。

**⚠️ 局限性**

局限在于对完整整数歧义的固定依赖、未充分验证极端多路径/多路径干扰环境，以及对细粒度地图辅助（如车道中心线约束）的缺失。

---

## 255. AGATE: Provenance-Based Runtime Defense Against Compositional Attacks on LLM Agents

**arXiv ID:** 2609.30830 | [PDF](https://arxiv.org/pdf/2609.30830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 256. Towards Universal Representation-Based Process Control

**arXiv ID:** 2609.30790 | [PDF](https://arxiv.org/pdf/2609.30790v1)

**作者:** Jinmyeong Choi `[一作]` (Carnegie Mellon University), Artur Dubrawski `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种以参考相对过程控制为核心的窗口级时间序列一致性检测方法，利用预训练编码器将窗口映射到表示空间，在该空间中使用核密度估计构造经验参考分布，并通过长度条件的conformal校准得到满足假设检验显著性水平的决策阈值。

**💡 创新点**

创新点包括：①将传统的参数化平稳性检验转化为非参数的参考相对检验；②引入基于LDA的敏感性对齐，将表示空间聚焦于统计稳定性变化；③采用长度条件conformal校准解决窗口长度异质性导致的校准失效问题；④实现了对周期性（cyclostationary）过程作为合法稳定状态的统一处理。

**🔧 技术方法**

使用技术主要包括：预训练时间序列编码器（如Chronos2）、LDA投影用于敏感性对齐、Gaussian核密度估计（KDE）、基于参考分布的conformal校准（包括长度条件版本）。

**📊 数据集**

实验主要在合成数据集上进行：AR(1)窗口、周期模板窗口，以及在这些基准上添加均值/方差/趋势/单位根或周期偏差等结构化偏差。所有窗口长度在{1,2,4,8}比例下随机选取；参考集、校准集和测试集均来自同一分布或其偏差版本。

**📈 对比分析**

与传统方法（ADF、KPSS、PP、两阶段周期性检验）以及CPD方法（CUSUM、ClaSP）对比，本文方法在不同误报率下均获得高达0.98–1.00的AUC，显著优于传统平稳性检验且稳健对周期性和组合偏差。长度条件校准在保持显著性控制的同时提升检测功效。

**⚠️ 局限性**

局限性：①依赖参考集的代表性，若参考集未覆盖所有正常状态可能导致误报；②预训练编码器的通用性在极端领域迁移时可能受限；③实验仅在合成数据上验证，需在真实工业/业务场景中进一步评估。

---

## 257. MDSkin-Net: Multi-Task Skin Lesion Analysis Driven by Pattern Analysis Priors and Spatial Alignment Regularization

**arXiv ID:** 2609.30855 | [PDF](https://arxiv.org/pdf/2609.30855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. Aligning One-Step Generative Models with Reward-Weighted Transport Distillation

**arXiv ID:** 2609.30840 | [PDF](https://arxiv.org/pdf/2609.30840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 259. Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength

**arXiv ID:** 2609.30849 | [PDF](https://arxiv.org/pdf/2609.30849v1)

**作者:** Phuong Q. Le `[一作]` (University of Melbourne), Jey Han Lau `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造输入和链式推理（CoT）两类扰动，并利用LLM评判器统一测量扰动强度，评估大型语言模型的自一致性；

**💡 创新点**

创新点在于提出了基于LLM的扰动强度评判框架，统一衡量输入与CoT扰动；并将自一致性细分为响应性（对强扰动的灵敏度）和鲁棒性（对弱扰动的不敏感性）两种互补指标；

**🔧 技术方法**

核心技术包括LLM-as-a-judge的提示设计、语义相似度（余弦距离）与信息量变化（surprisal）对照实验；

**📊 数据集**

实验数据集涵盖ARC-Challenge、OpenBookQA、Sports以及StrategyQA四个多选问答数据集；

**📈 对比分析**

通过与人类标注的强度评判比较，LLM评判器在Pearson、Spearman和Kappa上均显著优于余弦距离和surprisal；随后利用扰动强度的翻转率（FR）衡量模型的响应性与鲁棒性，结果显示输入扰动对模型影响更大，而CoT扰动更能保持答案稳定；

**⚠️ 局限性**

主要局限包括：对中间强度级别的判定存在歧义、强扰动样本稀缺导致估计方差较大、以及假设LLM评判与人类判断完全一致可能不完全成立。

---

## 260. Attention-Based Adaptive Policies for Simultaneous Speech-to-Text Translation

**arXiv ID:** 2609.30839 | [PDF](https://arxiv.org/pdf/2609.30839v1)

**作者:** Filip Tăşădan `[一作]`, Anders Søgaard `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种基于交叉注意力的实时翻译决策策略，允许离线训练的语音转文本模型在流式场景中使用。

**💡 创新点**

创新点在于利用最近帧注意力得分（RFAP）和双条件注意力速率（DCAP）来动态判断何时输出翻译，从而无需额外训练或模块。

**🔧 技术方法**

使用基于Encoder-Decoder（Conformer-Transformer）架构的交叉注意力机制，结合两步推理获取注意力得分。

**📊 数据集**

在CVSS-C语料库的法英、德英、斯英三语对上进行实验，数据来源于CoVoST 2。

**📈 对比分析**

与EDAatt、Wait‑K和Local Agreement等基线相比，RFAP在中等延迟下BLEU提升约4点；DCAP在低延迟下可实现负平均延迟，同时保持较高翻译质量。

**⚠️ 局限性**

局限在于DCAP在延迟增大时性能下降，且两种策略仍需依赖离线模型的注意力分布，可能对不同语料或模型结构的泛化性有一定影响。

---

## 261. Evaluation Is All You Need for Multi-Modal Autonomous Driving

**arXiv ID:** 2609.30818 | [PDF](https://arxiv.org/pdf/2609.30818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 262. Fast Plans, Faithful Actions: Closing the Planning-Execution Gap in Hierarchical Vision-Language-Action Models

**arXiv ID:** 2609.30833 | [PDF](https://arxiv.org/pdf/2609.30833v1)

**作者:** Chuanliang Xie `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在基于π_0.5的分层视觉语言动作模型中，针对规划-执行接口的两个关键需求——实时规划与执行对规划的真正依赖——进行系统的受控干预实验，并提出两种改进：块自回归规划(Block‑AR)和归一化目标调制(NGM)；

**💡 创新点**

创新点在于(1)首次用受控干预量化规划与执行的对齐程度，揭示传统后缀令牌条件下执行器几乎不利用规划端点；(2)提出块自回归规划，将每个完整的waypoint映射为单次VLM前向推理，显著减少串行推理次数和计划延迟；(3)设计归一化目标调制，采用深层残差路由并通过阶段门控、噪声扰动与丢弃来抑制“位移-积分器捷径”，使规划端点在执行时真正发挥作用。

**🔧 技术方法**

技术主要包括：PaliGemma（SigLIP + Gemma‑2B）作为规划器，300M Gemma动作专家作为执行器，低秩适配（LoRA）微调，块自回归解码（Block‑AR），归一化目标调制（NGM），以及受控干预实验（端点抹除、持续时间增减、图像/语言替换）和性能评估指标（计划延迟、动作占比、成功率、敏感度）。

**📊 数据集**

使用了四个LIBERO套件（Spatial、Object、Goal、Long）作为仿真数据集，并在真实双臂Rokae AR5‑5+Robotiq 2F‑85平台上完成三类任务（pepper‑banana 放置、pepper 互换、颜色匹配放置），共收集154条演示。

**📈 对比分析**

与Token‑AR基线和传统后缀令牌条件的π_0.5进行比较。Block‑AR将VLM前向推理次数从57降至8，计划延迟从1094 ms降至125 ms，动作占比提升至≈65%；成功率保持在Token‑AR±1.4个百分点。NGM进一步提升所有四套件平均成功率≈+2.6个百分点，并将端点敏感度从0.6%提升至42.6%，使端点抹除导致成功率下降≈7.4个百分点。

**⚠️ 局限性**

局限性包括：仅在π_0.5的特定分层架构上验证，缺乏对其他VLA模型的泛化评估；Block‑AR假设固定格式的计划token，NGM依赖AdaRMS条件，二者是否适用于不同专家结构未知；仅单次训练、实验规模有限（仅三任务、20次试验），无法评估模型方差和统计显著性。

---

## 263. Geometric Optimization Parameterized by Piercing Complexity

**arXiv ID:** 2609.30829 | [PDF](https://arxiv.org/pdf/2609.30829v1)

**作者:** Aritra Banik `[一作]` (National Institute of Science Education and Research), Saurabh Ray `[通讯]` (NYU Abu Dhabi)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过引入“穿刺度”这一参数，研究并证明了在固定穿刺度下，基于局部搜索的算法能够为几何Set Cover和离散独立集问题提供PTAS；同时给出了对权重问题的近似算法；

**💡 创新点**

创新点在于将穿刺度作为核心复杂度指标，摆脱了传统的非穿刺或脂肪性约束，利用平面支持图的分离性质和奇数交叉分割，构建了适用于任意穿刺度受限几何族的PTAS，并得到浅轨迹上界与权重近似；

**🔧 技术方法**

主要技术包括：局部搜索框架、奇数交叉边界分割（Pach–Tóth定理）、自适应递归分割、浅轨迹的线性超图签名与采样，以及基于VC维度与浅单元复杂度的ε‑net论证；

**📊 数据集**

本文为理论研究，不依赖具体数据集，而是给出泛化的证明与算法分析；

**📈 对比分析**

与以往仅在非穿刺或脂肪几何族上得到的PTAS相比，本文的算法在常数穿刺度下保持多项式时间；在穿刺度为多项式对数时可得到QPTAS；权重问题实现了常数因子近似；

**⚠️ 局限性**

局限性在于需对穿刺度做上界，无法直接推广到任意穿刺度的族；此外算法对穿刺度的指数依赖仍较大，且对Hitting Set等问题的扩展尚未完成；

---

## 264. Adaptive Interaction Graphs for Particle Simulation

**arXiv ID:** 2609.30822 | [PDF](https://arxiv.org/pdf/2609.30822v1)

**作者:** Aiden Zhou `[一作]` `[通讯]` (Yale University), Aiden Zhou (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种自适应交互图粒子仿真器AdaptGNS，利用每颗粒预测的不确定性动态扩展邻域以降低长时序误差累积。

**💡 创新点**

首次在粒子仿真中引入基于不确定性的单通道图构造机制，并用异方差负对数似然联合训练加速器和方差头，实现不需要双重GNN推理的自适应邻域。

**🔧 技术方法**

采用图神经网络、软正则化方差头、异方差高斯NLL损失、百分位阈值扩展邻居以及延迟不确定性推理的单通道自适应图构造。

**📊 数据集**

使用水滴（SPH流体）和沙粒（MPM颗粒）两套粒子仿真数据集进行实验。

**📈 对比分析**

与固定k或半径的GNS以及MLP对比，AdaptGNS在WaterDrop上边缘数减少20%且MSE@200略优，在Sand上提升约8% MSE，同时相较于NLL固定图模型表现更好。

**⚠️ 局限性**

主要局限包括延迟不确定性估计导致突变响应滞后、未完全量化构图开销、对3D/混合物理场景验证不足，以及NLL目标下加速精度受限。

---

## 265. Evaluating Real-Time Voice Agents: From Component Quality to Grounded Outcomes

**arXiv ID:** 2609.30798 | [PDF](https://arxiv.org/pdf/2609.30798v1)

**作者:** Shivam Negi `[一作]` (Northeastern University), Rashi Jain `[通讯]` (University of Texas at Dallas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对实时语音代理领域的文献进行系统综述，梳理了38篇核心论文，归纳出六大应用类别，并提出TRG报告标准。

**💡 创新点**

①通过可验证的检索与程序化校验构建可复现的文献语料；②将架构、评价和多方交互三大维度统一分析并形成TRG最低报告规范；③揭示“架构是部署约束”“评价已转向基于状态验证”“双人假设正在瓦解”三项实证论断。

**🔧 技术方法**

采用系统检索（arXiv API）、关键词过滤、手工筛选、程序化PDF验证与元数据比对；使用基于语义标签的分类与六维度对齐；并用可执行验证脚本检测误引用。

**📊 数据集**

研究构建的38篇核心论文元数据集合（15种seed、20通过检索、3追加），并提供GitHub/Zenodo存档；实际评估数据来源于各论文中的原始评测数据（如Full‑Duplex‑Bench、VoiceBench、MP‑Bench等）。

**📈 对比分析**

本文不做数值汇总，而是对各论文的指标与方法做归类，突出相互对照：架构层面讨论end‑to‑end与cascade速度/可部署性；评价层面对比基于文本与基于状态验证的成功率；多方交互层面对比dyadic与multiparty评测。性能上，end‑to‑end在保留并行语音信息上表现优越，但目前不可自托管；cascade在自托管与可控性上更优。

**⚠️ 局限性**

①样本量有限，仅38篇，可能存在主题与发表渠道偏差；②主要来自arXiv，缺乏同行评审与可重复实验结果；③未进行元分析或统一基准，无法直接排序；④对多方模型缺乏实测评估；⑤综述时间窗口短，需周期性更新。

---

## 266. HasMem: Hard-Origin Adaptively Softened Memory for Long-Term LLM Agents

**arXiv ID:** 2609.30797 | [PDF](https://arxiv.org/pdf/2609.30797v1)

**作者:** Zihong He `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Hai-Ning Liang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Hard-Origin Adaptively Softened Memory（HasMem），一种在冻结的 LLM 之上对连续记忆槽宽度进行自适应调整、重编码和读取适配的机制，用于长期记忆管理。

**💡 创新点**

创新点在于：①使用硬起点（硬提示嵌入）作为可验证的初始状态；②通过控制器动态决定每条记录的 KEEP / SHRINK / EXPAND 操作；③采用 Writer 在宽度变更时重新编码记忆槽；④加入低秩读取适配器（Reader）和跨轮状态（Global）实现读出与写入的协同适配；⑤在保持硬提示等价性的前提下实现连续宽度调整。

**🔧 技术方法**

主要技术包括：冻结 LLM（Qwen2.5-7B / Mistral-7B），连续空间记忆槽，位置池化与线性插值重编码，低秩读取适配（A_r, B_r），GRU 递归全局状态，控制器 MLP 与成本预测，训练采用 QA 监督、位置成本正则化与强化学习风格的策略学习。

**📊 数据集**

使用的数据集有：① Multi-Session Chat（MSC）开发集，提取 535 条恢复式问题用于评估记忆保留与重构；② LongMemEval‑S（500 条跨轮问答）用于评估跨会话事实推理与知识更新。

**📈 对比分析**

比较方法：将 HasMem 与同一冻结 LLM 的硬提示基线（硬引用）和基于规则的宽度重编码进行对比；同时对不同目标宽度比例（ρ）和读出方式（Reader/Global）进行 ablation。实验结果显示：在与硬提示相同的记忆位宽下，HasMem 的词法 F1 提升 4.4 点，EM 下降 3.7 点；在约束的体积预算下，学习到的宽度策略比规则重编码提升 8–23.6 个百分点 EM；Global 读出在 ρ=0.75 时 EM 提升 2.24 点；LongMemEval‑S 上 F1 提升 5.6 点、NLL 降低 6.983。

**⚠️ 局限性**

局限性：① MSC 评测仅包含 7 条记录以内的短历史，无法充分验证长历史衰退与干扰；② 评测依赖规则上限 100.0，缺乏更大规模的压缩评估；③ 对不同硬提示长度、硬提示与软化交互的影响未充分探究；④ 仅在 Qwen/Mistral 上测试，跨模型推广性待验证；⑤ 读取适配与写入重编码的相互影响需要更系统的 ablation。

---

## 267. ConsultMind:Towards Automated Diagnostic Consultation via Uncertainty-Aware Reasoning

**arXiv ID:** 2609.30796 | [PDF](https://arxiv.org/pdf/2609.30796v1)

**作者:** Xiao Sun `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了AutoDisym自动构建疾病–症状贝叶斯网络（DSBN）并提出ConsultMind面向诊疗的自适应、可解释的对话框架。

**💡 创新点**

创新点在于：①通过诊断知识与诊断标注的临床叙述自动生成DSBN；②利用贝叶斯后验不确定性驱动询问策略（探索、区分、巩固），实现动态且可解释的诊断决策。

**🔧 技术方法**

技术手段包括：诊断知识图构建、检索增强的LLM映射、贝叶斯网络推理、策略编码与动态决策、以及基于后验熵的终止判定。

**📊 数据集**

使用的数据集：①精神科、呼吸科、发热门诊共14,541例的临床案例（对话、电子病历）；②外部评测集990例涵盖78个疾病；③三公开数据集（MentalHospital、MedSP1000、AIHospital）进行泛化验证。

**📈 对比分析**

方法比较：与直接提示、RAG+LLM、医学专用LLM三组基线对比；ConsultMind在所有临床场景与所有LLM上均显著提升，Top‑1提升最高22%点，Top‑3提升最高38%点，MRR提升约0.265；外部数据也表现出显著增益。

**⚠️ 局限性**

局限性：依赖LLM的表达能力，某些罕见疾病或未覆盖的症状仍可能出现缺漏；DSBN的构建需要较大规模标注数据，且对专业知识的覆盖度有限；在极端噪声或错误注释下的鲁棒性尚待进一步验证。

---

## 268. Moving Horizon Estimation for Quadrotors: An $\mathcal{L}_1$ Adaptive Optimizer Approach

**arXiv ID:** 2609.30777 | [PDF](https://arxiv.org/pdf/2609.30777v1)

**作者:** Thinh Nguyen `[一作]` (University of Illinois Urbana-Champaign), Naira Hovakimyan `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于线性平滑移动窗口估计的QP形式，并开发了带l1ao的时间变优化求解器，用于四旋翼无人机状态估计。

**💡 创新点**

创新点在于将MHE转化为稠密QP并采用l1ao时变求解器，实现一次迭代即可跟踪最优解，显著降低计算负荷，并在不良初始猜测和未知过程噪声条件下提高估计精度。

**🔧 技术方法**

采用线性化MHE、QP condensing、l1ao自适应低通滤波预测、Newton连续时间方法、OSQP/Clarabel等数值求解器，以及Python+NumPy实现。

**📊 数据集**

使用仿真四旋翼动力学模型（m=1kg，J=[…]）与GPS+IMU测量噪声（σ=0.1）进行仿真，无真实数据集。

**📈 对比分析**

与EKF、传统基于OSQP的MHE对比，l1ao‑MHE在RMSE上平均降低25%（初始误差）和60%（未知过程噪声）且每步计算时间比OSQP快2–3倍，达到实时可行。

**⚠️ 局限性**

局限性：仍基于线性化模型，未考虑硬约束；对非高斯噪声鲁棒性未验证；在极端动态或大噪声下预测误差可能导致性能下降。

---

## 269. Dialogue-Based Streaming Audio-Visual Target Speaker Extraction with Predictive Dialogue Information

**arXiv ID:** 2609.30774 | [PDF](https://arxiv.org/pdf/2609.30774v1)

**作者:** Shuhan Zhang `[一作]` (Shenzhen Loop Area Institute), Haizhou Li `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM的目标说话人语音活动投影（TS‑VAP）模块，用于在线音频-视频目标说话人提取（AV‑TSE），并构建了第一个包含第三方干扰的实时对话基准。

**💡 创新点**

创新点包括：①首次针对自然对话中的第三方干扰构建在线AV‑TSE基准；②利用语音LLM（Mini‑Omni2 + Qwen2）直接从重叠混合中预测未来目标与伙伴说话活动，实现跨模态预测；③系统比较历史、同步（ASD）与预测三种上下文对在线提取的贡献，证明预测上下文最具补充性。

**🔧 技术方法**

主要技术：语音LLM推理模块TS‑VAP（结合 Whisper+CLIP 编码与 Qwen2 视觉条件化头）；基于卷积、Transformer、GridNet 和双路径 RNN 的五种 AV‑TSE 后端；历史上下文通过 GRU 汇总；同步上下文使用 ASD 嵌入；预测上下文通过多头注意力与 2 s 记忆池化；pVAD 头用于目标缺失时的抑制；两轮训练（先训练TS‑VAP再冻结后端训练）。

**📊 数据集**

数据集：VoxCeleb2（用于预训练 20k 2‑speaker 混合）；IEMOCAP‑Dialog3Mix（6 s 对话+随机第三方干扰，包含 23.5k 训练、8.1k 验证、3k 测试混合）；RealTalk‑Dialog3Mix（1k 真实现场对话，用于零样本跨域评估）。

**📈 对比分析**

评估指标：SI‑SNR（目标出现时）与 CSR（目标静默时正确抑制率）。实验表明：TS‑VAP 在 AV‑SepFormer、TDSE、USEV 上分别提升 5.7%、4.6% 与 5.8%（≈0.15 dB）平均 SI‑SNR；与历史+同步+预测三者联合可获得 10.03 dB，较无上下文基线提升 10.9%；在 RealTalk 上仍保持显著提升（最大 +0.35 dB）。CSR 在第三方干扰下提升约 5.4 点。

**⚠️ 局限性**

局限性：①预测上下文在高重叠区（>80%）的提升有限；②TS‑VAP 模型训练与后端分离，未实现端到端 LLM 嵌入；③仅在二人对话场景验证，难以直接推广至多方对话；④依赖于训练时的第三方干扰样本，跨域泛化仍有待进一步验证。

---

## 270. Learning Natural Conversational Behavior in Tandem Speech-to-Speech Models with Randomized Guidance

**arXiv ID:** 2609.30773 | [PDF](https://arxiv.org/pdf/2609.30773v1)

**作者:** Manato Yaguchi `[一作]` (Sakana AI), So Kuroki `[通讯]` (Sakana AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在双向语音到语音对话系统KAME中，研究者提出使用随机中间指导来替代传统的LLM模拟，用于训练前端模型。

**💡 创新点**

创新点在于直接从对话语料库采样随机回复文本作为中间指导，避免每条训练样本都需要耗时的LLM生成，从而在保持高质量回答的同时提升对话自然性。

**🔧 技术方法**

采用的技术包括双向S2S前端、异步文本LLM后端、随机中间指导生成、Moshi模型基础、Whisper ASR、Silero VAD、pyannote说话人分离等。

**📊 数据集**

使用的数据集包括合成问答对生成的对话、3.8k小时的真实两人对话（PodcastIndex）以及用于评测的MT‑Bench和MTR‑DuplexBench。

**📈 对比分析**

与LLM生成指导、相似度检索指导以及仅目标指导等策略对比，随机指导在MT‑Bench上的得分与相似度检索相近，实测在3.8k小时真实对话上实现了更顺畅的轮流、自然度更高，并保持显著的MT‑Bench优势。

**⚠️ 局限性**

局限性包括：模型在处理打断和暂停时仍表现不佳，训练目标未显式约束回应中止行为，且随机指导未对后端信息的精细使用提供监督。

---

## 271. Stable Recovery and Benign Overparameterized Landscapes for Phase Retrieval from Coded Diffraction Patterns

**arXiv ID:** 2609.30825 | [PDF](https://arxiv.org/pdf/2609.30825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 272. The KV Cache Is the New Memory Wall

**arXiv ID:** 2609.30854 | [PDF](https://arxiv.org/pdf/2609.30854v1)

**作者:** Tejinder Singh `[一作]` `[通讯]` (Dell Technologies), Tejinder Singh (Dell Technologies)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对长上下文自回归 LLM 推理中 KV 缓存压缩技术进行系统化，给出统一的分析模型、基准协议与跨域对比框架。

**💡 创新点**

提出基于算力密度与内存流量交叉的三域模型，给出跨域压缩方法的闭式理论极限，并制定八条设计规则。

**🔧 技术方法**

采用量化、令牌驱逐、分页、前缀共享与异构内存分层等压缩技术，并构建 NVIDIA H100/B200 与 AMD MI300X 的 roofline 与拓扑模型。

**📊 数据集**

以 Llama‑3‑70B 及其子模型为基础，使用 LongBench、检索样本等公开评测集，重点在统一协议下的模拟数据。

**📈 对比分析**

通过统一协议分离派生与报告指标，在 128k 文本上下文下对 2‑bit、4‑bit 量化、驱逐率、前缀共享、分页、分层等方案进行对比；在 KV‑bound 区域可获得 2‑bit+驱逐 6.6× 的解码加速，单序列仅 1.25×。

**⚠️ 局限性**

仅提供理论与模拟，实际实现受 kernel 与调度层瓶颈限制；缺乏跨硬件统一的基准、质量评估一致性以及完整的功耗与延迟曲线。

---

## 273. Peer-Grounded Counterfactual Path Planning for Chronic Health Management

**arXiv ID:** 2609.30838 | [PDF](https://arxiv.org/pdf/2609.30838v1)

**作者:** Saman Khamesian `[一作]` (Arizona State University), Hassan Ghasemzadeh `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了POROS框架，通过构建行为进展图给慢性病患者提供可执行的增量行为路径，帮助其逐步提升健康指标。

**💡 创新点**

创新点在于：①边缘要求同行可实现的行为距离与健康结果严格提升，保证每一步可行且有提升；②使用二次距离权重（d²）实现路径分解；③框架无模型依赖，可一次构建后多次查询；④嵌入Bandura自效能和Festinger社会比较理论。

**🔧 技术方法**

核心技术包括：临床行为转移距离（CBTD）计算、阈值ε基于同体内最大单期转变、构建有向无环图、最短路径搜索（Dijkstra）求最小成本路径。

**📊 数据集**

使用两组独立的1型糖尿病纵向数据集：ExActHealth（18人483天）和T1D‑UOM（15人854天）。

**📈 对比分析**

与单点终点方法（DiCE、Wachter等）以及路径感知方法（FACE）比较。POROS在“红→蓝”最接近终点的任务中，98.1%/97.0%源节点可达；每步TIR提升平均从约26%降至5%，比单跳减少约4.8×-5.5×；相较于FACE，POROS 100%步骤保证单调提升且不超出ε，路径更短且更可执行。

**⚠️ 局限性**

局限包括：①ε仅保证行为可行性，未考虑个体代谢差异；②路径质量受样本多样性限制，小样本时路由有限；③评估为回顾性，需前瞻性验证其临床效益；④目前仅适用于可定义MCID与单一健康结果的情形。

---

## 274. HIRE: History-Conditioned Interaction Reasoning and High-Rate Execution for Visually Aliased Precision Manipulation

**arXiv ID:** 2609.30828 | [PDF](https://arxiv.org/pdf/2609.30828v1)

**作者:** Rongji Li `[一作]` (Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了跨速率的 HIRE 框架，用历史力学信息进行交互状态推理，并在接触关键阶段以高频率、流形结构化方式执行，解决了可视化交互状态混淆问题。

**💡 创新点**

创新点在于：①将力矩历史视为持久的物理证据，通过 Force Perceiver 将可变长度力学序列压缩为固定 token；②在接触期间引入 Interaction‑Manifold Executor（IME），将运动进度与横向修正分离并与力反馈耦合；③实现 ISR 与 IME 的双向闭环，使物理执行结果回馈给下一轮推理。

**🔧 技术方法**

技术主要包括：多模态（视觉+语言+力）序列编码、因果时间卷积、跨模态注意力（Force Perceiver）、跨速率循环（5 Hz ISR 与 50 Hz IME）、力预测自监督、流形结构化执行（intrinsic ρ 与 transverse δ ⊥）以及 hybrid force‑position 控制。

**📊 数据集**

使用了300 条真实机器人演示（每个任务 300 条）收集的数据，包含 RGB‑D 视觉、六轴力/扭矩传感器数据与执行动作；实验涵盖透明物体 pick‑and‑place、EV 连接器插拔、四分之一转扳机旋转等三种交互模式。

**📈 对比分析**

与七个基线（π_0.5、π_0.5+naive force、TA‑VLA、ForceVLA、FM‑VLA、RDP、Force Policy）在 30 次真实机器人 roll‑out 下对比。HIRE 在所有阶段均达 90%‑以上完成率，显著优于最强基线（例如 RDP 96.7% 但在某些阶段失败率高）。在执行精度方面，HIRE 的角度、径向误差、力跟踪等指标均优于对比方法，且在未见对象上保持高泛化。

**⚠️ 局限性**

局限性包括：①对力/扭矩传感器精度依赖较高；②跨速率闭环的实现复杂度和调参成本；③在极端视觉遮挡或极低频率传感下，ISR 对力历史的依赖可能不足；④模型在高度动态或多物体交互环境中的鲁棒性尚待验证。

---

## 275. Learning Provable Neural Network Observer for Uncertain Dynamical Systems

**arXiv ID:** 2609.30819 | [PDF](https://arxiv.org/pdf/2609.30819v1)

**作者:** Zhangyi Wang `[一作]` (Zhejiang University), Shengze Cai `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段训练框架，用点导向的Lyapunov预训练快速获得高容量神经网络观测器，再通过LMI微调实现全局Lyapunov稳定性证明。

**💡 创新点**

创新点在于：①将全局LMI约束拆解为无约束预训练与轻量化的LMI微调，显著降低大规模网络的求解难度；②设计基于采样的Lyapunov损失，给出局部稳定半径和概率覆盖理论；③通过预训练提供温和的初始化，加速LMI收敛。

**🔧 技术方法**

使用技术包括：神经网络观测器（残差网络）、Lyapunov函数与LMI约束、点导向Lyapunov预训练损失、LMI惩罚微调、半正定规划（SDP）和梯度下降。

**📊 数据集**

实验数据集包括：X-29 机型仿真、Quad-UAV 在地面效应下的飞行仿真、AUV 在动态流涡街中的航迹跟踪。

**📈 对比分析**

与直接LMI求解、单纯LMI梯度下降、PID、PID+Neural Lander、基本NMPC、NMPC+ESO 等方法比较，结果显示：预训练+微调训练时间缩短约2.5倍；MSE、跟踪误差显著下降（如AUV误差从1.49 m降至0.05 m，约48%提升）。

**⚠️ 局限性**

局限性包括：缺乏有限样本下的收敛与覆盖保证；LMI微调仍需求解大型SDP，计算量不消除；需要手工设定Lyapunov阈值、正则化参数；对极大网络或非常复杂动力学的通用性待进一步验证。

---

## 276. Quadratic bounds for uncompletable words and matrix mortality

**arXiv ID:** 2609.30817 | [PDF](https://arxiv.org/pdf/2609.30817v1)

**作者:** Rahul Chandelkar `[一作]` (Efficient Computation Inc), Samrath Chadha `[通讯]` (Efficient Computation Inc)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究有限非空不完整唯一可译码中最长不可完成单词的上界，并给出对应的矩阵零积最短长度上界；证明了在最大词长k的限制下，最长不可完成单词长度不超过4k^2-3k，且此二次上界在某些构造中最优。

**💡 创新点**

①将代码问题与矩阵零积问题等价化，得到统一的二次上界；②使用条件期望与循环平均技术构造长度≤4k^2-3k的不可完成单词；③给出多项式时间算法完成判定并输出短单词；④在Lean中形式化证明。

**🔧 技术方法**

条件期望、循环平均、矩阵压缩、花瓣自动机、Kraft等式、联合谱半径、零矩阵判定、自动机理论、符号计算。

**📊 数据集**

论文基于理论推导和构造示例（如X_k代码、对应的自动机），并未使用实验数据集；所有结果均为数学证明。

**📈 对比分析**

与之前的上界(k^2(δ+1)(δ+2)-k、(k+1)k^2(L+2)(L+1))相比，新上界仅依赖k，消除了对代码长度和延迟的依赖；算法多项式时间，时间复杂度在输入大小（代码总长度L、字母表大小d）上为多项式，满足实际可行性。

**⚠️ 局限性**

仅适用于唯一可译码；对非唯一可译码问题仍无相同上界；上界虽最优阶，但常数项可能尚可改进；构造示例需特定字母表与代码结构，普适性有限。

---

## 277. A Benchmark and Diagnostic Study of Epistemic Admission in Shared Agent Memory

**arXiv ID:** 2609.30813 | [PDF](https://arxiv.org/pdf/2609.30813v1)

**作者:** Xiaoyang Li `[一作]` (University of Southern Queensland), Taotao Cai `[通讯]` (University of Southern Queensland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了 Correlated Promotion Benchmark（CPB），用以评估多智能体在共享记忆中对声明的录入与传播效果。

**💡 创新点**

创新点在于构建了两种实验模式（CPB‑Static 与 CPB‑Live），在 Static 中预设真值与来源关系，在 Live 中记录写入、检索与来源血统，并以此观察误导传播；此外引入源类型门控策略显著降低错误采用。

**🔧 技术方法**

技术包括：基于来源血统的线性 collapse 复制检测、四类入库策略（vote、confidence、judge、治理）以及 Mem0 与 A‑MemGuard 的写入与一致性检查。

**📊 数据集**

使用公开注释的数据集（claim‑verification 语料、Wiki 引用图、Multi‑hop 编辑语料）并配合自行编写的虚构情境（共 1200 期，包含 4 个智能体家族）。

**📈 对比分析**

在 4 个智能体家族、100 个配置上对 10 种策略（Static）和 8 种策略（Live）进行评测；治理规则在所有家族中均保持误导率仅 0.06‑0.09，显著优于共享全部策略（误导率 0.22‑0.47）。

**⚠️ 局限性**

限制在于：若缺乏来源血统信息，几乎所有策略都难以持续拒绝伪造声明；复制与改写的检测依赖人工设定；实验仅在已知情境下验证，真实场景下可变因素未被完全覆盖。

---

## 278. XPhysICS: Cross-Physical-Domain Threat Grounding for Industrial Control Systems Security

**arXiv ID:** 2609.30805 | [PDF](https://arxiv.org/pdf/2609.30805v1)

**作者:** Sangshin Park `[一作]` (University of Utah), Luis Garcia `[通讯]` (University of Utah)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种面向工业控制系统的跨域威胁下沉方法（XPhysICS），通过源侧威胁抽象、目标条件下沉、验证切片构建和切片充分性评估，实现将已记录的网络-物理威胁语义映射到不同目标系统上，并支持后续检测/可视化等下游分析。

**💡 创新点**

核心创新在于：①将源侧语义抽象与目标条件下沉分离，使用五个显式资格判据（角色、类型、阶段、一致性、规则表面）实现确定性下沉；②引入“验证切片”这一目标特定的中间表示，保留受控路径、观测信号、时序假设等信息；③实现完整的可追溯证据账本（provenance ledger），记录源、映射、下沉决策与切片生成全过程；④提供可选的控制器代码推导（CrossPLC）和时序逻辑门控（STL/RTAMT）作为增强路径。

**🔧 技术方法**

技术包括：结构化威胁抽象（effect family、manipulated/consequence roles、observability obligations、provenance）、基于目标合同的可执行下沉（角色匹配、类型兼容、阶段连贯、切片可行性、规则交叉）、验证切片构造（路径、信号、依赖、时序）、定量评估（切片充分性分类）、下游消费者兼容性测试（Invariant、GeCo、SAIN、SCAPHY）、动态可实现性验证（ICSFlux 论文式搜索重现）以及可选的 PLC 代码推导与 STL 监控。

**📊 数据集**

使用了 83 条源侧威胁抽象（来自 SWaT、WADI、OilTreatment、Fischertechnik 等 12 篇文档），并构建了 5 个目标合同（水处理、供水、水电/水能、化工工艺）进行验证；此外，利用 SPHERE 实验环境中的模拟器轨迹、受控扰动以及 9 条水电/化工目标的限定案例进行切片执行与消费者测试。

**📈 对比分析**

比较方法主要通过：①跨目标下沉覆盖率矩阵（源 → 目标 结构下沉比例）；②验证切片的充分性评估（strong/partial/insufficient）；③在同一切片上运行四类下游消费者，记录检测命中与误报；④对比上游 GeCo 实现与本方法生成切片的兼容性；⑤动态可实现性研究对比结构下沉与模型驱动搜索结果。性能表现：下沉覆盖率在同域目标最高（≈90%），跨域略低；切片充分性中 strong/partial 约占 80%，insufficient 约 20%；下游消费者在切片上表现一致，误报率随消费者类型变化；GeCo 兼容性确认为 1.0 recall，FP 2‑3；动态可实现性显示结构下沉与模型实现不完全一致。

**⚠️ 局限性**

限制包括：①源侧抽象需人工或专用解析，存在主观性与一致性挑战；②目标合同仅支持 5 种 effect family，限制了适用范围；③下沉判据中的类型检查粗糙，未实现完整物理单元一致性；④验证切片基于受控扰动，未覆盖真实攻击场景；⑤依赖完整目标证据信息，缺失时可能导致下沉失败或切片不充分；⑥实验以 SPHERE 研究环境为主，泛化性至其他工厂和控制架构有限；⑦动态可实现性仅在单一模型/命令面向下测试，未全局验证；⑧缺乏多目标交叉验证与自动化抽象一致性评估。

---

## 279. Understanding the Role of Prompt Template in Knowledge Distillation for Safety Alignment

**arXiv ID:** 2609.30802 | [PDF](https://arxiv.org/pdf/2609.30802v1)

**作者:** Anjila Budathoki `[一作]` (University of Tennessee), Yi Ding `[通讯]` (University of Tennessee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探究在知识蒸馏过程中，使用不同的提示模板对对齐学生模型的安全性（拒绝行为）会产生何种影响。

**💡 创新点**

创新点在于首次系统验证聊天模板（含对话控制标记）在蒸馏阶段会显著削弱已有的安全对齐，而非聊天模板则可在保持较高实用性的同时减轻安全退化。

**🔧 技术方法**

采用LoRA参数蒸馏（平衡CE与KL损失），结合对话和非对话两种模板格式，以及内部表示分析与安全攻击评估。

**📊 数据集**

实验数据集包括大规模指令跟随数据以及四个安全评测基准（AdvBench、JailbreakBench、HarmBench、SORRY-Bench）。

**📈 对比分析**

对比结果显示，聊天模板蒸馏后攻击成功率显著上升（例如Gemma SORRY-Bench ↑35.32%），但同时提升了任务效用；非聊天模板蒸馏虽然提升效用稍弱，但安全退化更小。

**⚠️ 局限性**

主要局限在于仅评估了基于文本的善意指令蒸馏，未涵盖代码/数学等结构更丰富领域；仅使用LoRA蒸馏；安全评测依赖自动化裁判，可能存在偏差。

---

## 280. Motion Style Slider: Endpoint-Supervised Continuous Style Control for Human Motion Diffusion

**arXiv ID:** 2609.30795 | [PDF](https://arxiv.org/pdf/2609.30795v1)

**作者:** Chen-Chieh Liao `[一作]` (Institute of Science Tokyo), Shuichi Kurabayashi `[通讯]` (Cygames, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过端点监督实现连续风格强度控制的运动生成框架。

**💡 创新点**

设计了风格方向与标量强度α的组合，并引入线性与单调性正则化，使得在无中间强度训练样本下实现平滑、可插值和可外推的风格控制。

**🔧 技术方法**

使用预训练的运动扩散模型MDM、冻结的TMR风格/内容编码器、适配层，并在其上加入α条件与正则化。

**📊 数据集**

在PerMo、Bandai‑Namco、Xia三大风格数据集以及自制的过度反应（real‑capture over‑reaction）数据集上训练和评估。

**📈 对比分析**

与DeepMotionEditing、MCM‑LDM等基线在内容保持、风格识别、Fréchet Motion Distance等指标对齐评估，结果显示在风格控制一致性（MonoViol、LinErr、ExtraErr）方面优于基线，内容与现实性指标亦具竞争力。

**⚠️ 局限性**

依赖端点配对的质量和风格嵌入的准确性；高α生成仍可能出现运动瑕疵，缺乏物理约束，且仅验证了参考对齐的控制，未覆盖文本或未见风格的跨域迁移。

---

## 281. Symbiotic Architecture for Post-Hoc Audio Extension of Frozen Language Models

**arXiv ID:** 2609.30784 | [PDF](https://arxiv.org/pdf/2609.30784v1)

**作者:** Yotaro Kubo `[一作]` (Sakana AI), Yujin Tang `[通讯]` (Sakana AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 symbiotic 架构，通过 injector 模块直接把音频特征写入 LLM 的 KV 缓存，从而在不 fine‑tune LLM 权重的前提下实现音频理解；

**💡 创新点**

创新点包括：①将 audio‑prefilling 任务从 LLM 解耦出来，成本由 injector 宽度决定；②利用上下文优化（context optimization）直接控制 KV 缓存，避免灾难性遗忘；③在 injector 训练中引入噪声 RoPE 与 KV 规模匹配，显著提升对长序列的鲁棒性；

**🔧 技术方法**

主要技术：基于 LConv 的 CNN injector、RMSNorm 归一化、KV 规模匹配、噪声 RoPE、LoRA 细调音频编码器、WavLM 作为音频编码器、Qwen3 作为 LLM、subsampling 与 max‑pooling；

**📊 数据集**

使用的数据集包括：LibriSpeech（ASR）、CompA‑R、DCASE2025 Task5、Clotho‑AQA（AQA）、CochlScene（ASC）以及文本评测集 WikiText‑2、HellaSwag、GSM8K；

**📈 对比分析**

与 Encoder‑only、Monolithic（带 LoRA 的 LLM fine‑tune）以及 SLM 基线对比；Symbiotic 在音频任务上接近 Monolithic 但激活参数从 1,108M 降至 552M，prefilling 速度提升约 21%（157 s vs 199 s）；在文本任务上保持与冻结 LLM 相同的性能，避免灾难性遗忘；ASR WER 下降至 1.91%‑3.73%，长句子 WER 由 120.68% 降至 4.70%；非 ASR 任务显著优于 Encoder‑only；

**⚠️ 局限性**

局限性：目前仅在小型 LLM（Qwen3‑0.5B）上验证，缺乏大模型规模的可扩展性证明；CNN‑centric injector 对于更复杂的语音任务（如 CochlScene）仍有性能差距；对 KV 缓存写入的精细控制仍需进一步研究。

---

## 282. An $n^{8/5+o(1)}$-Time $Ω(λ^3)$-Approximation for Longest Common Subsequence

**arXiv ID:** 2609.30778 | [PDF](https://arxiv.org/pdf/2609.30778v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种算法，用于计算两个长度为n的字符串的最长公共子序列的Ω(λ^3)近似值。

**💡 创新点**

创新点在于将算法的运行时间从O(n^1.95)改进到O(n^1.6+o(1))，显著提高了效率。

**🔧 技术方法**

使用了算法设计和复杂性分析的技术。

**📊 数据集**

未具体提及使用的数据集。

**📈 对比分析**

与之前的算法进行比较，新的算法在运行时间上有显著的改进，性能更优。

**⚠️ 局限性**

未提及具体的局限性。

---

## 283. Learning Chance-Constrained MDPs with Bellman Distributional Certificates

**arXiv ID:** 2609.30856 | [PDF](https://arxiv.org/pdf/2609.30856v1)

**作者:** Chenbei Lu `[一作]` (Cornell University), Hongyu Yi `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出了一种基于Bellman分布式证书的算法，用于在无限期折扣的机会约束MDP（CCMDP）中学习安全且近似最优的策略，并给出了模型基础与模型自由两类实现及其样本复杂度上界与下界。

**💡 创新点**

创新点包括：①将机会约束转化为有限时间、离散预算下的贝尔曼递归，构造可重用的分布式安全证书；②在模型基础学习中利用行列逆KL置信集实现对所有策略的统一置信保证；③在模型自由学习中设计了基于轨迹概率的方差减小策略梯度方法，并提供独立验证的安全保证；④提供了匹配的下界，证明样本复杂度的两项主导因素不可约。

**🔧 技术方法**

核心技术包括：Bellman分布式证书、离散化预算与截尾尾部、逆KL置信集的鲁棒贝尔曼备份、行列KL信赖集的共享、轨迹概率与贝尔曼递归的结合、方差减小的策略梯度与KKT残差分析、独立验证与安全阈值收紧。

**📊 数据集**

实验数据集主要有：①合成的CCMDP实例；②IEEE 14-bus DC电力网络的能量存储控制基准（由DC潮流仿真器生成的有限状态CCMDP）。

**📈 对比分析**

比较方法包括：①使用Bellman证书的策略选择器；②使用Markov-CMDP期望成本上界（即利用马尔科夫不等式的充分条件）的方法。实验结果显示，Bellman证书方法在样本量足够时趋近于真模型的机会约束最优解；相比之下，期望成本上界方法即便在真模型下也存在结构性保守性，导致性能明显落后。

**⚠️ 局限性**

局限性：①模型基础结果仅适用于确定性策略、有限后继数且需要假设可行的规划oracle；②模型自由结果仅给出局部KKT残差保证，验证可能返回未决定；③未对更广泛的政策类、连续状态空间或复杂后继结构给出全局理论保证，后续研究需拓展。

---

## 284. I-Parakeet: Integer-Only Conformer ASR on Mobile NPU

**arXiv ID:** 2609.30846 | [PDF](https://arxiv.org/pdf/2609.30846v1)

**作者:** Taichi Nishimura `[一作]` `[通讯]` (Sony Interactive Entertainment), Taichi Nishimura (Sony Interactive Entertainment)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了一个完全整数运算的Parakeet-CTC 0.6B Conformer ASR模型I-Parakeet，可在智能手机NPU上无浮点运算运行。

**💡 创新点**

提出整数化的相对位置自注意力、针对Swish的极大误差最小化近似以及层级激活范围分析来选择INT16/INT8量化策略。

**🔧 技术方法**

整数量化、整数相对位置自注意力、极大误差最小化的Swish多项式近似、BatchNorm INT16、预编码分位数裁剪、I-BERT整数核等技术。

**📊 数据集**

使用LibriSpeech（dev‑other、test‑clean、test‑other）和Common Voice进行量化校准与评估。

**📈 对比分析**

与CPU FP16/INT8、NPU FP16、NPU INT8等基线比较，在Nothing Phone 3a上I-Parakeet实现RTF 0.048、WER 4.97%，比CPU FP16快7.5倍、占用内存低60%，仅比FP32少1.21个百分点。

**⚠️ 局限性**

对预编码层仍需分位数裁剪，BatchNorm输出需要INT16，且在NPU上Swish使用硬件查表导致略高WER，未在低功耗无浮点CPU（如ARM Cortex‑M）上验证。

---

## 285. Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness

**arXiv ID:** 2609.30820 | [PDF](https://arxiv.org/pdf/2609.30820v1)

**作者:** Nux Li `[一作]` `[通讯]` (Meta), Nux Li (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究循环Transformer在后训练量化中的失效机制，并提出针对性恢复策略。

**💡 创新点**

创新点在于揭示“反馈暴露”和“校准盲区”两种关键问题，并通过沿循环累积Hessian的轨迹校准有效恢复性能。

**🔧 技术方法**

使用了分组INT4、GPTQ、AWQ、谱半径与Hessian秩分析、控制实验以及轨迹校准等技术。

**📊 数据集**

实验数据来自GSM8K、WikiText-2、LAMBADA等公开数据集。

**📈 对比分析**

与单步GPTQ和RTN进行对比，单步GPTQ往往低于RTN，但轨迹校准能在九个模型中将大部分性能恢复至接近bf16水平。

**⚠️ 局限性**

局限在于仅覆盖至约4.17B参数规模，且并非所有模型均能完全恢复，且秩提升并不一定能预测恢复幅度。

---

## 286. Counterfactual Online Conformal Prediction Under Adaptive Logging

**arXiv ID:** 2609.30811 | [PDF](https://arxiv.org/pdf/2609.30811v1)

**作者:** Xinyu Qiao `[一作]` (Shanghai Jiao Tong University), Tao Yao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并评估一种逆倾向加权在线合规预测方法（PW‑OCP）及其双重鲁棒扩展（DR‑OCP），以在自举日志环境下实现对每个动作的反事实覆盖率；

**💡 创新点**

首次将逆倾向加权和双重鲁棒技术嵌入在线合规预测，证明在存在动作诱导的分布偏差时能达到最优反事实覆盖率，并将覆盖误差映射到决策 regret；

**🔧 技术方法**

使用逆倾向加权估计、双重鲁棒估计、在线自适应阈值更新、信息理论下界证明及实验对比等技术；

**📊 数据集**

在合成均值/方差偏移、多臂扩展实验，以及真实数据集 ZOZOTOWN Open Bandit Women/Men 和 DJIA 重新平衡中进行验证；

**📈 对比分析**

与 ACI、FACI、AgACI、Conformal PID、SAOCP、NEx‑CP、ECI、Online COPP 等多种在线合规基线对比，PW‑OCP/DR‑OCP 在反事实覆盖率上显著优于基线，并在决策 regret 上逼近 Oracle，提升了整体性能；

**⚠️ 局限性**

仅适用于上下文赌博机而非完整 RL，要求正性且固定探索下界，双重鲁棒性依赖准确的倾向/结果模型，且在极端自举日志或未观测混杂情形下性能可能受限。

---

## 287. Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems

**arXiv ID:** 2609.30809 | [PDF](https://arxiv.org/pdf/2609.30809v1)

**作者:** Tak Shing Au Yeung `[一作]` (NVIDIA), Simon See `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究并实现物理信息神经网络（PINN）和深度算子网络（DeepONet）求解无穷拉普拉斯及从p=2到p=1000的p-拉普拉斯方程，并给出收敛理论与大p极限分析。

**💡 创新点**

提出针对大p问题的稳定化与连续训练策略、PINN在p-拉普拉斯问题下的条件收敛证明，以及DeepONet在参数化p与几何域上逼近无穷拉普拉斯的通用逼近理论。

**🔧 技术方法**

采用PINN、DeepONet、正则化/残差裁剪、梯度范数裁剪、η归一化、Adam优化等技术。

**📊 数据集**

通过有限元Newton求解器生成p-拉普拉斯解作为训练数据，涵盖2D/3D多种域（圆、椭圆、球、圆柱、环面）以及p从2到1000的参数样本。

**📈 对比分析**

与传统有限元/ Oberman 解法对比，PINN在高维/大p下推理速度快、误差≤10⁻⁷，DeepONet在无穷极限下误差与FEM收敛至极限误差相当，并能在多域间插值。

**⚠️ 局限性**

仍缺乏对非线性、退化椭圆问题的严格收敛证明，对大p训练收敛的理论保证有限，网络对极端几何或p>1000外推可能失效。

---

## 288. Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional Foundation Embedding on Low-Data Molecular Assays

**arXiv ID:** 2609.30789 | [PDF](https://arxiv.org/pdf/2609.30789v1)

**作者:** Yiqi Yao `[一作]` (Harvey Mudd College), Miquel Duran-Frigola `[通讯]` (Ersilia Open Source Initiative)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一套可审计的低标签分子活性预测系统，利用可命名的紧凑描述符块组合，并通过贪婪搜索与表格基础模型 TabPFN 配合实现高效预测。

**💡 创新点**

创新点在于：①将每一维特征与模型卡和训练来源挂钩，实现特征级可审计；②通过基于模型卡的泄漏过滤和贪婪增量搜索，获得与高维 CheMeleon 嵌入相当的 AUC；③预声明性能门限（pooled 与 per-assay parity）保证结果可解释且可比较。

**🔧 技术方法**

技术手段包括：TabPFN 作为基准预测器；随机森林与 kNN 的冷启动证据；交叉验证 AUC 作为选择指标；特征层次可追溯的命名块；贪婪增量搜索；基于模型卡的泄漏检查；对齐一致性权重的投票融合。

**📊 数据集**

使用了 Therapeutics Data Commons ADME/Tox 的九个二分类 assay（含 Ames、hERG、intestinal absorption 等），在 50–100 标签、10/25 标签以及 10 个未见 assay 上进行评估。

**📈 对比分析**

方法是将紧凑块组合与 CheMeleon‑2048 与 Mordred‑1411 两宽嵌入在相同的分子子集上通过 TabPFN 进行 AUC 评估。结果显示，50–100 标签下紧凑块平均 AUC 为 0.762，对比 CheMeleon 的 0.764 差距 +0.003，满足预设的 pooled parity gate；25 标签下亦满足 pooled gate；10 标签下未满足。整体性能与宽块相当，但特征维度仅 14 列，推理成本约为宽块的三分之一。

**⚠️ 局限性**

局限性包括：仅在 9 个 assay 上验证，低标签（10）情形表现不足；模型卡泄漏过滤保守，未能完全排除泄漏风险；不同子集对 AUC 影响显著；对 δ=0.001 阈值的稳定性缺乏充分验证；无法为某些提升性能的块提供化学解释。

---

## 289. Job Class Thermal Intent Aware Liquid Cooling Allocation for AI Data Centers

**arXiv ID:** 2609.30785 | [PDF](https://arxiv.org/pdf/2609.30785v1)

**作者:** Krishna Chaitanya Sunkara `[一作]` `[通讯]` (Oracle), Krishna Chaitanya Sunkara (Oracle)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了Job‑Class Thermal Intent (JCTI) 框架，利用调度器发布的作业类型与调度预告信息，提前对液冷系统进行热预测与流量预置，从而在作业启动前就已做好冷却准备。

**💡 创新点**

核心创新在于将作业级热意图（job‑class, dispatch delay）直接注入到液冷控制层，构造前馈+反馈混合控制，打破了传统仅依赖温度传感的被动控制模式；同时引入工作负载热就绪度评估，量化作业到达后温度稳定与违规风险。

**🔧 技术方法**

使用了基于 MLPerf GPU 能耗数据的热签名建模、基于阿里巴巴生产日志的作业到达/持续时间/GPU分配统计、Monte Carlo 仿真评估、前馈预置流量计算、PI 反馈修正、可投影的多机柜共享 CDU 流量分配与线性约束投影。

**📊 数据集**

主要数据集为 MLPerf GPU 能耗轨迹（训练、推理、分析）以及阿里巴巴真实生产集群的作业到达时间、持续时间与 GPU 分配日志。

**📈 对比分析**

通过 120 组配对蒙特卡洛实验与传统 PI 循环对比，JCTI 将热违规率降低 56.4%、过冲积分降低 60.2%，但泵能耗提升 25.4%；所有差异均在统计学上显著（p < 10⁻¹²⁰）。

**⚠️ 局限性**

局限性包括对作业类别与调度预告精度的依赖、泵能耗提升、热模型为整体简化（未考虑冷板微通道与 GPU die 级热传播）、缺乏实时多区耦合与在线参数自适应机制。

---

## 290. MVVBench: Benchmarking 4D Reasoning in Vision-Language Models

**arXiv ID:** 2609.30952 | [PDF](https://arxiv.org/pdf/2609.30952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 291. Skip the Talk, Re-Focus on Vision: Latent Reasoning for Reasoning Segmentation in Multimodal Large Language Models

**arXiv ID:** 2609.30783 | [PDF](https://arxiv.org/pdf/2609.30783v1)

**作者:** Tianhang Guo `[一作]` (National University of Defense Technology), Xinbiao Gan `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LIRSeg，在推理分割任务中使用一组可学习的潜在标记代替传统的显式链‑of‑thought（CoT），实现更高效且更精确的目标分割；

**💡 创新点**

① 通过两阶段训练（空间对齐 + 基于 GRPO 的 RL 优化）将潜在标记与视觉目标紧密对齐；② 引入极值优势采样、探索‑稳定梯度分离以及潜在多样性放大等信息理论机制，提升潜在表示的多样性与学习信号；③ 用可学习潜在标记取代冗长 CoT，显著降低注意力干扰和推理标记数量；

**🔧 技术方法**

使用多模大语言模型 Qwen2.5‑VL‑7B 与 SAM3 分割器，结合组相对策略优化 (GRPO) 进行强化学习，使用多目标奖励（掩码 IoU、边框 IoU、长度与格式奖励），以及空间对齐损失、极值采样、梯度分离和多样性正则等技术；

**📊 数据集**

训练使用 ReasonSeg、MUSE、MMR、RefCOCO 以及 VisionReasoner‑7K 数据集，评估覆盖单/多目标、单/多语义、单/多粒度等多种场景；

**📈 对比分析**

与 SFT、RL、PixelThink、DR²Seg 等现有方法在同一 MLLM+SAM 组合下对比，LIRSeg 在 ReasonSeg 上 gIoU 71.7（比 VisionReasoner 提升 5.9）且推理标记数仅 4（约 16× 减少），在 MUSE、MMR 上分别提升 7.1 和 4.7，RefCOCOg 上 72.7，整体性能优于同类基线并保持速度提升；

**⚠️ 局限性**

对潜在标记数量的选择敏感，过多会导致冗余、过少则表达不足；RL 奖励噪声仍影响极难样本的表现；目前验证主要集中在 Qwen2.5‑VL‑7B + SAM3，其他 MLLM/分割器的泛化仍待进一步验证；

---

## 292. Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift

**arXiv ID:** 2609.30781 | [PDF](https://arxiv.org/pdf/2609.30781v1)

**作者:** Liang You `[一作]` (University of Pittsburgh), Siyuan Dai `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种缺失性感知的校准方法（missingness-aware conformal calibration），在跨医院分布偏移下为死亡预测模型生成满足目标覆盖率的预测集合。

**💡 创新点**

创新点在于：①在独立样本上选择最能区分不同医院的测量变量作为分组依据，避免使用校准数据重用；②在每个缺失组内采用 Mondrian 校准；③对聚合与医院内评估差异给出解析性分解（加权、抵消、最差组切换）并证明聚合可能导致方法排名逆转。

**🔧 技术方法**

技术主要包括：分位数校准（Mondrian conformal prediction）、非一致性得分、分组选择启发式、对比随机/临床分组、风险分层、条件 conformal、权重调整与统计分解。

**📊 数据集**

使用了两个大型 ICU 数据集：eICU Collaborative Research Database（82 家医院）和 MIMIC-IV（单一医院内 6 个护理单元），预测目标为 ICU 住院死亡。

**📈 对比分析**

与传统池化校准、随机/临床分组、风险分层、score-tree、mask weighting 以及官方条件 conformal 进行对比。缺失性感知校准在其选定的缺失组上平均减少 1.9% 的 worst-group 覆盖缺口；但在更广泛的缺失组、个别医院或标签上并未持续改善，甚至出现符号反转。风险分层在面板覆盖上表现更好，但缺失组覆盖不一定得到保证。

**⚠️ 局限性**

局限性包括：①研究为回顾性且使用 ICU 入院后第一天的数据，未评估临床可行性；②校准假设校准样本独立，未考虑医院内相关性；③分组选择仅为启发式，未证明最优；④未评估空集或多标签集合的操作策略；⑤结果仅在 eICU 多医院与 MIMIC-IV 单医院内单元之间验证，缺乏更广泛的跨机构推广验证。

---

## 293. JevSoup: System-One Routing for Training-Free LoRA Composition

**arXiv ID:** 2609.30922 | [PDF](https://arxiv.org/pdf/2609.30922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 294. Machine Unlearning for Large Language Models: Foundations, Advances, and Agentic Extensions

**arXiv ID:** 2609.30909 | [PDF](https://arxiv.org/pdf/2609.30909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 295. Sampling Safe Futures: Multimodal Trajectory Planning for Personalized Safety in Anthropomorphic AI

**arXiv ID:** 2609.30780 | [PDF](https://arxiv.org/pdf/2609.30780v1)

**作者:** Benedetta Picano `[一作]` (University of Florence), Dusit Niyato `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种个性化的轨迹级安全框架，用于人形人工智能的多模态轨迹规划，以确保用户的心理安全。

**💡 创新点**

创新点在于将人形人工智能的安全性从响应级别过滤转变为个性化的对人机关系未来演变的控制。

**🔧 技术方法**

使用了生成流网络（Generative Flow Network）来生成多样化的未来演变，并进行轨迹采样。

**📊 数据集**

使用了基于真实人类与聊天机器人交互的统计数据进行模拟评估，并使用公共基准派生的响应策略。

**📈 对比分析**

与传统的响应级别安全措施和多轮风险累积检测方法进行了比较，结果表明，轨迹感知决策显著降低了有害状态的频率，同时保持了有益的互动。

**⚠️ 局限性**

局限性在于该方法依赖于用户模型的准确性，且在实际应用中可能面临用户行为的多样性和不可预测性。

---

## 296. Spackle: Completing Large View Single Image NVS with Adaptive Gaussians

**arXiv ID:** 2609.30941 | [PDF](https://arxiv.org/pdf/2609.30941v1)

**作者:** Xuanzhi Liu `[一作]` (Shenzhen University of Advanced Technology), Song Wang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Spackle 框架，通过残差学习在单图像大视角变换合成中增添针对遮挡区域的额外 Gaussian，避免固定 Gaussian 预算下的容量竞争。

**💡 创新点**

创新点在于：①使用二进制遮挡掩码自动识别缺失区域；②仅在这些区域上增添可选 Gaussian，保持总数不变；③通过生成式伪监督训练残差网络，保持高效推理。

**🔧 技术方法**

采用技术包括 3D Gaussian Splatting (3DGS)、二进制遮挡掩码、生成式伪监督（如 Diffusion/ProPainter）以及残差学习网络。

**📊 数据集**

训练数据：COCO 2017 无标签子集；评估数据集：Middlebury、Booster、WildRGBD、ETH3D、LLFF、Tanks & Temples。

**📈 对比分析**

与基线 SHARP 对比，使用 CLIPIQA、NIQE、MUSIQ、Hole Area Ratio 等指标。Spackle 在大视角偏差下显著提升 CLIPIQA、减少 Hole 区域，同时保持相近的渲染速度（额外 2–4% 时间开销）。

**⚠️ 局限性**

局限性包括：需要准确的遮挡估计和高质量的伪监督；残差学习引入少量额外参数与计算；若基线 3DGS 初始化不足，效果会受限。

---

## 297. From Tapping to Hopping: Augmenting Mobile GUI Agents with App-Native Deeplinks

**arXiv ID:** 2609.30887 | [PDF](https://arxiv.org/pdf/2609.30887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 298. Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models

**arXiv ID:** 2609.30935 | [PDF](https://arxiv.org/pdf/2609.30935v1)

**作者:** Bing Wang `[一作]` (Jilin University), Masashi Sugiyama `[通讯]` (RIKEN)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在持续大模型微调中估计并正交未知预训练梯度的框架，称为EoupCT，旨在同时保留任务特定能力和通用知识，缓解灾难性遗忘。

**💡 创新点**

创新点在于：①使用可学习的软提示配合 Gumbel-Softmax 生成最易被遗忘的伪预训练数据；②构建多目标 Pareto 优化问题，并设计第一阶高效优化器实现梯度正交；③在正交约束中同时考虑新任务梯度、估计的预训练梯度以及历史任务梯度，全面防止干扰。

**🔧 技术方法**

核心技术包括：可微软提示生成、Gumbel-Softmax 软解码、梯度正交投影、第一阶 Pareto 优化、低秩适配（LoRA）等。

**📊 数据集**

使用 SuperNI（15个NLP指令任务）评估任务特定性能，使用 MMLU（9个通用知识子任务）评估通用知识保持。

**📈 对比分析**

与LoRA、LoRAMoE、OLoRA、GainLoRA、CLoRA 等现有连续微调基线对比；实验表明EoupCT在 SuperNI 上平均提升约3.5% 准确度、3.7% 减少遗忘率；在 MMLU 上提升约1-5% 准确度且遗忘率显著降低。

**⚠️ 局限性**

局限性包括：对软提示长度和伪序列长度敏感，需调参；仅在单一预训练模型上估计梯度，若预训练数据分布极其多样化仍可能无法完全覆盖；实现复杂度较高，且对硬件资源有一定要求。

---

## 299. A Second Torque Port for Series Elastic Actuators: Parallel-Integrated Design and Time-Scale Torque Allocation

**arXiv ID:** 2609.30873 | [PDF](https://arxiv.org/pdf/2609.30873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 300. Training Graph Foundation Models on The Web Graph

**arXiv ID:** 2609.30894 | [PDF](https://arxiv.org/pdf/2609.30894v1)

**作者:** Ryoma Sato `[一作]` `[通讯]` (National Institute of Informatics), Ryoma Sato (National Institute of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了Acacia，一种从零开始、仅使用网页图进行预训练的纯图Transformer模型，能够在无需额外训练的情况下完成节点分类、链路预测、节点聚类、图生成和上下文学习等多种任务。

**💡 创新点**

核心创新在于：①纯图模型从零训练，完全不依赖预训练LLM；②通过token化节点/边/标签，支持任意特征维度和语义；③在自回归Transformer框架下实现多任务和in‑context学习；④利用Common Crawl网页图的多样性实现了从零起步的泛化能力。

**🔧 技术方法**

技术包括：decoder‑only自回归Transformer；节点/边/标签/header四种token化方式；随机特征哈希与节点ID置换；在训练中对节点特征进行随机化与正态噪声注入；采用广度优先搜索构建不同大小的子图；采用多任务训练表征采样与随机化；实现in‑context学习通过将示例图作为额外连通分量输入。

**📊 数据集**

训练数据主要为Common Crawl网页图；评估使用Cora、CiteSeer、PubMed、ogbn‑products四个标准图数据集；模型权重已公开在Hugging Face上。

**📈 对比分析**

通过与随机初始化、无训练Acacia以及Chance baseline比较，实验显示在节点分类（Cora 58.2%/CiteSeer 39.5%/PubMed 19.9%/ogbn‑products 66%）和in‑context分类、聚类等任务中，Acacia均显著优于基线；在聚类实验中ARI/AMI均大幅超过0.2。

**⚠️ 局限性**

局限性包括：对标签分布不均的图（如PubMed）效果受限；模型对节点ID随机化的依赖导致在极端结构或缺失特征的场景下表现不佳；仅使用网页图预训练，可能缺乏领域特定的结构知识；在极大图或复杂任务上仍需进一步优化规模与效率。

---

## 301. CoCoRerank: Towards Conventional Commit Message Generation by Component and Candidate Consistency Reranking

**arXiv ID:** 2609.30953 | [PDF](https://arxiv.org/pdf/2609.30953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 302. From Segments to Trajectories: Evolving Affective Graphs with Evidence Retrieval for Continuous EEG Emotion Recognition

**arXiv ID:** 2609.30890 | [PDF](https://arxiv.org/pdf/2609.30890v1)

**作者:** Chi Yang `[一作]` (Nanjing University of Aeronautics and Astronautics), Yuzhe Zhang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种用于连续EEG情感识别的端到端框架EAGER，目标是对整段EEG试验输出时间对齐的情感轨迹。

**💡 创新点**

创新点在于：①采用情感状态引导的拓扑演化模块(ASTE)，动态更新电极间的神经网络拓扑；②结合多尺度时序证据检索模块(MTER)，在不同时间尺度上聚合局部情绪波动与全局趋势，提升轨迹跟踪精度。

**🔧 技术方法**

技术方法包括：基于频域特征的Transformer时间编码、短时功能连接估计、上下文门控融合、图状态演化、以及多头注意力的多尺度时序检索与门控融合。

**📊 数据集**

实验使用三套连续情感标注EEG数据集：MAHNOB‑HCI（情感强度/价值）、SEED‑VII（情感强度）和REFED（价值与唤醒）。

**📈 对比分析**

与SVR、KNN、LSTM、Visual‑to‑EEG、GIGN、MAET、MASA‑TCN、TSMMF、EmT等九个基线进行对比。EAGER在所有数据集上均取得最高的CCC/PCC，且在MAE/RMSE上保持竞争力，显示出对情感轨迹的优越跟踪性能。

**⚠️ 局限性**

主要限制包括：在极端个体差异下预测往往被拉向群体均值，导致离散个体反应幅度被低估；目前模型仍为离线批处理，未实现实时流式推理。

---

## 303. EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting

**arXiv ID:** 2609.30880 | [PDF](https://arxiv.org/pdf/2609.30880v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 EXAONE Demand，一个针对需求预测的时间序列预训练模型，利用专门构建的需求语料库（真实多源数据加合成生成器）和混合专家适配器，对需求序列按行为特征进行路由并提升预测性能。

**💡 创新点**

创新点主要包括：① 需求专用语料库：通过严格筛选、统一转换真实数据并补充合成数据，覆盖稀缺的间歇、块状、促销等行为；② 需求感知路由器：基于八个尺度无量纲统计量学习动态分配低秩分支，实现按需求行为自动分配模型容量；③ 低秩 Mixture‑of‑Experts 适配器：在冻结的预训练 Backbone 上增添共享与四个路由分支，显著提升对四类需求（平滑、间歇、波动、块状）的预测。

**🔧 技术方法**

使用技术包括：冻结的 Chronos‑Bolt/Chronos‑T5 等预训练 TSFM；低秩 MoE 适配器与两层路由网络；合成数据生成器（高斯过程核、负二项计数、日历、促销、生命周期、库存寡缺等模块）；统计量提取与辅助损失实现行为监督。

**📊 数据集**

数据集：真实需求共 73 源、11.3M 系列；合成需求 47,500 系列；评估套件 22 个保留数据集（M4、M5、Rossmann、Favorita 等）；对比 36 个公开 TSFM 以及冻结 Backbone。

**📈 对比分析**

对比方法：统一窗口、预测长度、量化水平，全部无模型调参；使用六项指标（MASE、ND、WQL、MAPE、MAE、MSIS）及平均排名、胜率；EXAONE Demand 在所有指标上均领先，平均排名 1，胜率 91.9%，比最强基线提升约 5% 以上，在 22 个数据集上始终不处于最后，且两版本（真实+合成 vs 仅合成）均表现优异。

**⚠️ 局限性**

局限性：路由器仅基于原始窗口统计，未利用外部协变量；四类分支与路由结构固定，未学习；库存缺货仅通过合成模拟，未对真实销售中的隐性需求进行恢复；未来工作需加入协变量感知路由、可学习的分支结构和真实库存模型。

---

## 304. Developing a Roadmap to an AI-first Organization: A Case Study in Embedded Software Development

**arXiv ID:** 2609.30863 | [PDF](https://arxiv.org/pdf/2609.30863v1)

**作者:** Viktor Kjellberg `[一作]` (University of Gothenburg and Chalmers University of Technology), Miroslaw Staron `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对一家大型嵌入式软件公司的 40 名从业者进行混合方法研讨会，收集问卷和访谈数据，分析结果并制定 AI‑first 组织路线图。

**💡 创新点**

首次将 AI‑agent 与嵌入式软件工程实践结合，提出面向技术、组织和人力因素的综合路线图，并揭示关键挑战。

**🔧 技术方法**

利用 LLM、AI‑agent 框架（MCP、AI‑friendly Code Protocols）以及研究方法论（半结构化 Mentimeter 调查、归纳主题分析）。

**📊 数据集**

采用公司内部研讨会收集的问卷和访谈数据，共 40 名参与者的响应；并未使用公开数据集。

**📈 对比分析**

未进行实验对比或性能评估，路线图基于访谈定性分析；因此无数值指标可比。

**⚠️ 局限性**

仅基于单一企业的前瞻性研究，缺乏跨组织验证，且路线图未经过实证评估。

---

## 305. OneWorld: Learning Consistent Physics Across Actions in World Models

**arXiv ID:** 2609.30946 | [PDF](https://arxiv.org/pdf/2609.30946v1)

**作者:** Ke He `[一作]` (Wuhan University), Bin Yang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文研究了action‑conditioned视频世界模型在不同干预下的共享世界一致性问题，并提出了OneWorld框架，该框架通过共享物理机制解释器和共享世界证据评分实现多分支预测的物理一致性；

**💡 创新点**

创新点在于：①设计了物理机制解释器与共享世界证据评分机制，能够在不同动作干预下聚合物理解释；②通过共享世界流训练与采样时间引导，实现跨干预的物理一致性；③提出了独立的多干预评估协议，量化单回合与跨干预的物理拟合差距；

**🔧 技术方法**

采用的技术包括action‑conditioned流模型（基于ACWM‑DiT）、物理机制解释器（在先验基础上学习后向分布）、共享世界证据评分（衡量多干预的物理兼容性）、共享世界流训练损失与采样时引导梯度、以及基于仿真器的评估框架；

**📊 数据集**

使用了受ACWM‑Phys设定的控制环境——Push Cube、Push Rope、Pour Water，以及RoboDesk演示场景；

**📈 对比分析**

通过与ACWM‑DiT、Vid2World、CoCo、Twin Rollouts、Shared Noise、Posterior Alignment等基线在单回合M‑MSE、单干预与共享物理拟合（E_ind、E_shared）以及共享世界差距（Δ_world）等指标上进行对比；OneWorld在共享世界一致性上将Δ_world降低84.3%，单回合M‑MSE最低，整体表现最优；

**⚠️ 局限性**

局限性包括：对物理机制解释器的维度和表达能力依赖于训练设计；评估主要基于仿真环境，真实世界迁移性和泛化性尚未充分验证；训练与推理成本相对较高，且对多分支的扩展仍需进一步研究。

---

## 306. VLaRL: Augmenting Vision-Language-Action Models with Simulation-Trained Latent-Conditioned Residual RL

**arXiv ID:** 2609.30868 | [PDF](https://arxiv.org/pdf/2609.30868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 307. Persistent Negatives for Adversarial Black-Box On-Policy Distillation

**arXiv ID:** 2609.30864 | [PDF](https://arxiv.org/pdf/2609.30864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 308. Landscape Limits of Quantum-Inspired Evolutionary Optimization across 256 continuous functions

**arXiv ID:** 2609.30938 | [PDF](https://arxiv.org/pdf/2609.30938v1)

**作者:** Rishi Govind `[一作]` (BosonQ Psi Corporation), Abhishek Chopra `[通讯]` (BosonQ Psi Corporation)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对256个连续优化基准函数进行大规模实验，系统评估了量子启发进化优化(QIEO)在不同地形特征下的表现，并与遗传算法(GA)与CMA-ES进行对比，揭示QIEO的优势与局限。

**💡 创新点**

创新点在于：①构建全面的基准库并手工标注11种景观特征；②提出严格成功率与成本归一化评估，使用精度曲线展示不同容差下的性能差异；③通过逻辑回归将景观特征归约为两条主轴，揭示QIEO受维度与多峰性驱动的特性；④对比了三种QIEO变体与GA/ CMA-ES的结果，验证了QIEO在低维多峰景观上的覆盖优势和CMA-ES在曲率适应上的优势。

**🔧 技术方法**

技术方法包括：量子启发进化优化（固定θ、适应θ、实数编码三种变体）；经典遗传算法（实数编码与二进制编码）；CMA-ES（Hansen默认参数）；严格成功率定义（scaled residual<10^-3）；成本归一化（每百万评估能解决的函数数）；精度曲线（不同容差下成功率）；统计分析（Wilson区间、卡方检验、逻辑回归）。

**📊 数据集**

数据集为256个连续函数，来自Jamil & Yang的175个函数和Gavana的81个函数，包含未平移与平移两组（总计508个函数，256未平移，252平移）。每个函数带有手工标注的11种景观特征。

**📈 对比分析**

比较方法：在统一评估上限（5000代、1000个个体）下记录每个算法的评估次数、严格成功率、精度曲线；将评估次数归一化后比较每百万评估解决的函数数。性能结果显示：实数编码QIEO在未平移版的256个函数中严格成功率≈86%，高于CMA-ES≈66%；但QIEO的评估次数约是CMA-ES的200-300倍；在精度曲线中，CMA-ES在容差<10^-8时显著优于QIEO，而在10^-3容差下QIEO更胜一筹。QIEO主要优势集中在二维多峰、无耦合、良好条件的函数；CMA-ES在多峰且高度弯曲、强耦合的函数上表现更好。

**⚠️ 局限性**

局限性：①评估成本高（数百万评估）；②随着维度增加，覆盖能力指数下降，导致成功率显著下降；③无法充分利用曲率与耦合信息，导致在高条件数或非分离函数上的表现不佳；④二进制编码受分辨率限制，导致在高精度需求时失败；⑤单一评估上限不等同于成本均衡，导致不同算法之间的成本比较不完全公平。

---

## 309. EPOC: Endpoint-Preserving Online Correction With Compressed Residual State for Multi-Horizon Time Series Forecasting

**arXiv ID:** 2609.30929 | [PDF](https://arxiv.org/pdf/2609.30929v1)

**作者:** Takumi Fujimoto `[一作]` (Keio University), Hiroaki Nishi `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Endpoint‑Preserving Online Correction (EPOC)，通过压缩残差的低阶 DCT 系数和保留终端残差实现在线修正。

**💡 创新点**

创新点在于同时保留终端残差与低阶 DCT 组合，并将终端值共享给各通道的岭回归，显著提升精度同时保持极低的状态开销。

**🔧 技术方法**

使用离散余弦变换 (DCT)、在线指数加权岭回归、全局混合权重和预设的残差摘要等技术。

**📊 数据集**

实验数据集包括八个多变量时序数据：ETT 系列 (ETTh1/2/ETTm1/2)、Appliances、BDG2 站点 (Fox、Panther、Rat、Fox/ Panther 站点)。

**📈 对比分析**

与 δ‑Adapter、COSA、FAC、OMPB、Full ELF 等在线校正方法以及 TEFL‑style 适配器对比，EPOC 在 96 条固定‑基准条件下平均降低 MSE 15.40%、MAE 9.35%，并在大多数情况下优于对手，且仅保留约 6.3 KB 的辅助状态。

**⚠️ 局限性**

局限性在于仅针对完整块反馈、固定基准模型、固定时间步长；对不同预测长度、实时更新、峰值内存、延迟或能耗未做评估。

---

## 310. FTB Graph: Determining and Validating First-token Broadcasters and Language-Identity Head Circuits in Multilingual Language Models

**arXiv ID:** 2609.30954 | [PDF](https://arxiv.org/pdf/2609.30954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 311. MACBT: A Multi-Agent Cognitive Behavioral Therapy Decision Support System with Longitudinal Memory

**arXiv ID:** 2609.30939 | [PDF](https://arxiv.org/pdf/2609.30939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 312. ManiVid: Unified and Explainable Forensic Analysis of Manipulated Videos

**arXiv ID:** 2609.30934 | [PDF](https://arxiv.org/pdf/2609.30934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 313. TISD: On-Policy Self-Distillation with Trajectory Intervention

**arXiv ID:** 2609.30878 | [PDF](https://arxiv.org/pdf/2609.30878v1)

**作者:** Taeckyung Lee `[一作]` (KAIST), Sung-Ju Lee `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为Trajectory-Intervention Self-Distillation（TISD）的自监督学习方法，通过教师-学生的不一致性定位分支点，强制教师选择的动作并让学生重新生成后续序列，然后对整个轨迹进行密集的教师监督。

**💡 创新点**

创新点在于把教师-学生分歧作为轨迹引导信号，而非仅仅局部纠正；利用峰值KL点进行分支，既能暴露未出现的后续上下文，又能在同一次采样中获得更多教师目标，从而显著提升自监督训练效果。

**🔧 技术方法**

使用的方法包括：基于privileged context的on-policy self-distillation、逆KL（reverse KL）作为分歧度量、控制性token干预、峰值KL位置分支选择、学生侧后缀重新生成、以及在完整轨迹上进行的教师-学生分布对齐损失。

**📊 数据集**

实验数据集为LiveCodeBench v6（编程任务，配有丰富的执行反馈）和SciKnowEval L3（物理、化学、材料、生物等科学推理任务，无丰富环境反馈）。

**📈 对比分析**

与SDPO、GRPO、TRD等基线相比，TISD在LiveCodeBench上Avg@4提升至50.4%（相对SDPO 0.8%），Pass@4提升至51.4%；在SciKnowEval上Avg@128在等步/等时预算下分别提升0.8%/0.3%，总体表现优于所有基线。

**⚠️ 局限性**

局限性包括：需要学生在教师选择的动作后仍能继续生成；额外的分支选择和后缀生成会增加训练时的计算开销；在完全依赖教师重写后缀的场景下训练不稳定；且效果受教师目标质量和学生对分支点的可达性限制。

---

## 314. Evaluation of portability and performance of an OpenMP5 offloaded Quantum-Inspired Evolutionary Optimization Across the GPU Ecosystem

**arXiv ID:** 2609.30862 | [PDF](https://arxiv.org/pdf/2609.30862v1)

**作者:** Kasturi Venkata Srikanth `[一作]` (BosonQ Psi Corporation), Abhishek Chopra `[通讯]` (BosonQ Psi Corporation)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在不同GPU生态系统（实验室内Tesla V100、云工作站A100、领导层MI300X）中使用单一OpenMP 5源代码实现的量子启发式进化优化（QIEO）的可移植性与性能表现；

**💡 创新点**

证明了基于gene‑parallel（基因级并行）映射的QIP、Evaluation与CIP可在三种GPU上实现高达100‑150×的加速，同时保持单源代码不变；同时给出了可移植的优化层与环境特定的调整策略；

**🔧 技术方法**

主要技术包括OpenMP 5目标offload、基因级并行、基于共享/常量内存的读写调度、分段归约（Evaluation）、保持Qbit网格驻留、移除DetermineElite的设备执行；

**📊 数据集**

使用0/1背包问题作为基准，涵盖从Small到Large不同种群数与基因数（约3 000个实验），评估在三种GPU上的运行时间与速度比；

**📈 对比分析**

通过与同源CPU基线（单线程与72线程）比较，获得几何平均速度提升：V100 90×/12×、A100 136×/17×、MI300X 155×/16.6×；表明gene‑parallel映射在所有环境中均占优；

**⚠️ 局限性**

局限性在于Evaluation归约仍是瓶颈（10‑16×加速），DetermineElite的设备调用导致显著启动开销，且小规模（如32×500）工作负载在GPU上不具优势；对极大规模的内存容量与线程调度仍需进一步优化。

---

## 315. Self-Play Search Distillation for Large Language Model Reasoning

**arXiv ID:** 2609.30936 | [PDF](https://arxiv.org/pdf/2609.30936v1)

**作者:** Lorenzo Molfetta `[一作]` (University of Bologna), Pasquale Minervini `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过 MuZero 类搜索专家在可执行棋类游戏中的自对弈，记录每个决策点的最佳动作、合法动作、搜索价值和可重放的后续分支，随后将这些记录转化为可复现的“链式思考”监督数据，用以在不使用搜索的情况下对大语言模型（LLM）进行后训练。

**💡 创新点**

创新点在于：①提出了 Self‑Play Search Distillation（SPSD）框架，将搜索专家产生的完整决策记录转化为可验证的、基于环境的监督；②通过可重放的分支和值估计提供了对比性和后果性的监督，使模型能够学习到“比较和后果”思维模式；③将该监督方式与多种后训练方法（SFT、OPSD、RuleBot-Distill）进行系统对比，证明了搜索驱动的监督能显著提升游戏和数学推理性能。

**🔧 技术方法**

技术包括：MuZero / EfficientZero 训练的搜索专家、Monte Carlo Tree Search (MCTS) 与 PUCT 策略、可执行游戏环境接口、数据验证与线性化监督生成、LLM 的监督微调（SFT）、按搜索证据的按序自我蒸馏（OPSD）、以及基于规则的对照监督（RuleBot-Distill）。

**📊 数据集**

数据集主要由四个基础棋类游戏（Connect4、Domineering、Simplified Othello、Tic‑Tac‑Chess）构成，每个游戏生成约5k条可验证的搜索记录；此外，还使用了 15 个从 Ludii 集合中抽取的保留游戏进行泛化测试，以及六个数学推理基准（MATH500、AIME24/25、AMC23、Olympiad Bench、Minerva Math）用于评估推理能力。

**📈 对比分析**

对比方法：在 Qwen3‑4B、Qwen3‑8B、Llama‑3.1‑8B 三大模型上分别使用 SPSD、RuleBot‑Distill、SFT 训练。SPSD 在 Qwen3‑4B 上将游戏胜率从 15% 提升到 45%，FIDE 分数从 15% 提升到 39.9%，数学六项平均得分从 24.1 提升到 36.6；在更大规模的 Qwen3‑8B 和 Llama‑3.1‑8B 上也表现出显著提升。相比之下，单纯的 SFT 在训练后期性能下降，RuleBot‑Distill 的提升幅度小于 SPSD。SPSD 的优势主要体现在搜索价值与模型决策的一致性上。

**⚠️ 局限性**

局限性：①依赖搜索专家的质量，若专家表现不佳或过拟合训练环境，监督质量会下降；②仅在可执行棋类环境中验证，尚未证明能直接迁移至完全不相关的任务（如长文本推理）；③后训练需要额外的 GPU 资源和时间，规模受限；④模型在更大规模下仍受“思考”机制限制，部分模型即使训练到 1000 步仍出现对搜索价值的误读；⑤对比方法中未考虑自监督或强化学习与搜索结合的混合策略，可能存在更高效的路径。

---

## 316. CacheReforge: Bounded Recovery for Stale KV Caches under Evolving Adapters

**arXiv ID:** 2609.30884 | [PDF](https://arxiv.org/pdf/2609.30884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 317. UltraG-Bench: A Multi-task Benchmark for assessing Large Vision-Language Models on Pixel-level Evidence Grounding in Ultrasound

**arXiv ID:** 2609.30928 | [PDF](https://arxiv.org/pdf/2609.30928v1)

**作者:** Quanhao Zhu `[一作]` (Dalian University of Technology), Feng Xia `[通讯]` (RMIT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 UltraG‑Bench 三任务基准（指令分割、证据 VQA、证据报告生成）并提出 UltraG‑Agent，将大规模视觉‑语言模型与 Ultrasam3 分割模型结合，评估超声图像中像素级证据定位能力。

**💡 创新点**

① 统一标注 40 个公开超声分割数据集为像素级证据，形成跨任务评估框架；② 采用证据约束式注释，确保文本与像素掩模严格对应；③ 设计 UltraG‑Agent 通过 MLLM 语义推理与 Ultrasam3 精细定位协同，突破单一模型在语义或像素定位上的局限。

**🔧 技术方法**

使用大规模视觉‑语言模型（InternVL、Qwen、Lingshu 等）、Prompt‑driven 分割模型 UltraSAM3、GPT‑5.5 生成证据约束注释，并通过自动校验与专家复核保证标注质量。

**📊 数据集**

40 个公开超声分割数据集，覆盖 13 个解剖类别（腹部、乳腺、心脏、颈动脉、胎儿、肾脏、肝脏、肺部、肌肉、神经、卵巢、前列腺、甲状腺），共 138,832 张图像和 181,952 个像素级掩模。

**📈 对比分析**

对 14 种 MLLM、2 种 Prompt‑driven 分割模型及 4 个 UltraG‑Agent 变体进行统一评估。结果显示普通 MLLM 语义准确率高但像素定位差；像素分割模型定位优秀但缺乏语义生成；UltraG‑Agent 在指令分割、VQA、报告生成上均显著提升（IoU/Dice/Acc/G‑Acc/ROUGE‑L/SemAcc），最高变体比最优基线提升 0.29–0.39。

**⚠️ 局限性**

1) 仅针对超声影像，未验证跨模态或其他医学影像；2) 证据仅基于分割掩模，诊断、病理结论仍需人工标注；3) 依赖 GPT‑5.5 生成注释，可能引入注释偏差；4) 评估聚焦于像素级与文本对应，未考察模型鲁棒性与安全性。

---

## 318. QReason: Query-Focused Decoupled Chain-of-Thought for Efficient Passage Reranking

**arXiv ID:** 2609.30904 | [PDF](https://arxiv.org/pdf/2609.30904v1)

**作者:** Yang Zhang `[一作]` (Beijing Normal University), Yuanfei Huang `[通讯]` (Beijing Normal University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 QReason 框架，利用查询重写器生成可重用的 Chain‑of‑Thought 逻辑查询，替代滑窗中重复生成的推理轨迹，实现高效的段落重排序；

**💡 创新点**

通过将全局查询推理与窗口特定的排序解耦，显著减少推理冗余，并通过两阶段训练（有监督微调 + 强化学习）使重写查询更具排名导向性；

**🔧 技术方法**

采用大型语言模型（如 Qwen 系列）作为重写器与重排序器，使用 Relevance‑Grounded Supervised Tuning（基于相关段落的监督）和 Rank‑Aligned Dual‑Reward Refinement（列表奖励 + 点奖励）进行强化学习；

**📊 数据集**

在 BRIGHT 这一推理密集检索基准上进行评估，并在实验中参考了 FreshStack 与 NeuCLIRBench 作为补充验证；

**📈 对比分析**

与多种基准模型（Rank‑T5、Rank‑Zephyr、ReasonRank、Qwen 系列等）对比，QReason 在 Qwen3.5‑35B‑A3B 上实现平均 NDCG@10 36.93，优于同规模的 ReasonRank 及其他推理重排序器，同时在速度上比 ReasonRank 提升约 7.9 倍；

**⚠️ 局限性**

缺点在于仅通过列表奖励和点奖励缺乏对覆盖度、可信度和冗余等多维度质量的显式约束，未来可考虑引入基于 rubrics 的奖励来提升可控性与鲁棒性。

---

## 319. Bundled Contact Gradients: Stabilizing Differentiable Simulation for Deployable Dynamic Tasks

**arXiv ID:** 2609.30951 | [PDF](https://arxiv.org/pdf/2609.30951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 320. Reliability-Regulated Trajectory Optimization for Progressive COLMAP-Free 3D Gaussian Splatting

**arXiv ID:** 2609.30865 | [PDF](https://arxiv.org/pdf/2609.30865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. Towards Understanding Momentum Acceleration in River-Valley Loss Landscape

**arXiv ID:** 2609.30957 | [PDF](https://arxiv.org/pdf/2609.30957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 322. From annotation to reasoning: Culture in language models

**arXiv ID:** 2609.30897 | [PDF](https://arxiv.org/pdf/2609.30897v1)

**作者:** Daniel Hershcovich `[一作]` (University of Copenhagen), Jens Bjerring-Hansen `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出基于文学文本的文化推理评估框架，设计注释与推理任务并强调解释深度与学者多元视角

**💡 创新点**

首次将文学解释视角引入文化推理评估，提出保留学术争议、区分解释与支持质量、并结合专家反馈改进模型的创新框架

**🔧 技术方法**

融合自然语言处理与文学研究的方法，采用检索增强、对比实验、专家评审与软标签评估技术

**📊 数据集**

以丹麦文学为起点，使用历史与当代丹麦/挪威小说语料库、学术评论与引用资源作为数据集

**📈 对比分析**

通过设计识别引用、解释用途、比较阅读、修订等四类任务，并以专家评审评估错误、证据关联与可行性；目前尚无具体数值性能，但提出评价指标和对比基准

**⚠️ 局限性**

缺乏实证实验与数值验证，评估高度依赖专家评审难以规模化，跨语言跨文化的可比性与检索上下文选择仍面临挑战

---

## 323. Effects of Transcript Compression on LLM-based Medical Misinformation Detection in Japanese YouTube Videos

**arXiv ID:** 2609.30882 | [PDF](https://arxiv.org/pdf/2609.30882v1)

**作者:** Yuya Wake `[一作]` (University of Tsukuba), Toshiyuki Amagasa `[通讯]` (University of Tsukuba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较不同转录压缩方式（全文、摘要、检索增强、筛选）对日语医学YouTube视频可信度判定的影响，使用LLM进行分类；

**💡 创新点**

首次系统评估四种输入设计对LLM验证准确率的差异，并通过语言特征分析解释误判原因；

**🔧 技术方法**

使用大型语言模型cyberagent/calm3-22b-chat进行分类，Qwen2生成摘要，RAPTOR实现检索增强，J‑LIWC及手工编制的含糊词典进行语言特征量化；

**📊 数据集**

使用74条长篇日语医学YouTube视频（10条真实，64条虚假），每条转录≥10,000字符，手工标注；

**📈 对比分析**

在相同LLM与提示下比较四种输入，Baseline（全文）准确率0.905，F1 0.891，MCC 0.524；压缩方式均增大假阴性，Summary最差，Screening次优；

**⚠️ 局限性**

样本量小且不平衡、单作者标注可能带偏差、语言分析基于词典未捕捉上下文语义，未检验跨平台及多语言泛化。

---

## 324. Financial Fragility in Societies of LLM Agents: Coordination Failures and Stabilizing Mechanisms

**arXiv ID:** 2609.30940 | [PDF](https://arxiv.org/pdf/2609.30940v1)

**作者:** Zhenhao Fu `[一作]` (Shanghai Jiao Tong University), Qibing Ren `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了FRAIL实验框架，评估多代理LLM在银行挤兑、债务滚存和众筹三种金融环境中的系统性风险

**💡 创新点**

首次系统性研究多代理LLM导致的金融脆弱性，并对比了三种承诺机制（补偿式、集中式、参与者主导）对稳定性的影响

**🔧 技术方法**

使用多种主流LLM（GPT、Claude、DeepSeek、GLM、Qwen、MiniMax）与自定义金融仿真引擎实现动态状态更新与承诺执行

**📊 数据集**

采用模拟的三种金融情境（银行挤兑、债务滚存、众筹）作为实验数据集，未使用真实市场交易数据

**📈 对比分析**

通过五次episode的基准无机制对照与三种机制的对比实验，结果显示所有机制均显著提升成功率和社会福利，最佳机制因金融结构而异；例如在银行挤兑中参与者主导机制最优，在债务滚存中集中式机制最佳

**⚠️ 局限性**

局限在模拟环境的参数设定和有限的金融情境，未覆盖真实市场的复杂性和多样性；机制效果受模型特性和环境参数影响，可能难以直接推广到实际金融系统

---

## 325. Flow-TAG: Flow-based conditional latent transport for accurate spline approximation and data compression

**arXiv ID:** 2609.30955 | [PDF](https://arxiv.org/pdf/2609.30955v1)

**作者:** Roman Pavelkin `[一作]` (Eindhoven University of Technology), Fons van der Sommen `[通讯]` (Eindhoven University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了flow‑TAG框架，利用生成式流模型结合1D U‑Net生成B样条曲线的最优参数化（节点向量），实现对2D/3D曲线和ECG信号的逼近与压缩。

**💡 创新点**

创新点在于：①将B样条函数重新定义为可学习的U‑Net结构；②采用自调节多任务几何感知损失，将流匹配与曲线拟合误差联合训练；③通过可微分Sigmoid门实现离散节点选择的梯度可传播。

**🔧 技术方法**

使用了流匹配（Flow Matching）技术、可微分Sigmoid门、1D U‑Net自编码器、三任务损失（FM、MSE、平滑项）、以及条件ODE求解。

**📊 数据集**

训练与评估使用公开SplineGen数据集（1M+ 2D/3D B样条曲线）以及MIT‑BIH Arrhythmia数据库（ECG信号）。

**📈 对比分析**

与多种基线（PARNET、SplineGen、DNN‑Solver、传统压缩算法如JPEG2000、CAE）比较；在2D/3D曲线拟合中RMSE和Hausdorff距离均显著下降（平均RMSE下降55%，Hausdorff下降52%）；ECG压缩中得到13倍压缩率，PRDN≤9%，QS高于现有方法。

**⚠️ 局限性**

局限性包括：仅按100点块段处理，可能影响全局连贯性；训练依赖伪真值算法，若输入噪声严重可能不稳定；对极端曲线或不同分辨率的泛化尚未充分验证。

---

## 326. Low-Bit Recurrent States in Hybrid Language Models

**arXiv ID:** 2609.30950 | [PDF](https://arxiv.org/pdf/2609.30950v1)

**作者:** Hongren Chen `[一作]` (TelSwarm), Jiayang He `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对混合语言模型的循环状态进行低比特量化，提出基于可观测性Gramian的衰减加权与归一化范围的混合精度位宽分配，并在量化衰减率时采用对数尺度。

**💡 创新点**

创新点在于：①在不需要校准数据、旋转或额外训练的情况下，利用可观测性Gramian推导量化误差权重；②将衰减加权与归一化状态范围结合，实现高效的位宽分配；③对衰减率进行对数量化，进一步降低误差。

**🔧 技术方法**

技术方法包括：混合精度位宽分配、对数衰减量化、状态范围归一化、频繁写回策略、可观测性Gramian分析、无校准数据的高速率位分配法。

**📊 数据集**

使用的数据集包括 WikiText、RULER、Kimi、Qwen 系列模型、Nemotron-3-Nano 等公开文本基准，并在这些模型上进行实验评估。

**📈 对比分析**

在与七种基线（INT、TurboQuant、Hadamard INT、NVFP4、DAMP-style、随机采样等）对比时，本文方法在四比特平均位宽下，在 Kimi、Qwen3.8、Qwen3.6 上分别以 3.3、4.8、27.9 倍的 NLL 降低优势；在 6 位时 NLL 与 FP32 差异不足 0.005 nats；在 64 令牌写回场景下，亦能实现 1.6–4.1 位/元素的最佳或接近最佳性能。

**⚠️ 局限性**

局限性包括：在部分模型–预算组合（如 Qwen3.8、Nemotron）下性能未必优于基线；对两位量化时，某些基线偶尔更好；量化误差相关性与实现细节（如对数量化的精度）需要进一步研究；在极低位宽或稀疏写回情况下收益降低。

---

## 327. Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring

**arXiv ID:** 2609.30924 | [PDF](https://arxiv.org/pdf/2609.30924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 328. Causeway: Restoring Task Accessibility for Instruction Switching in VLA Policies

**arXiv ID:** 2609.30913 | [PDF](https://arxiv.org/pdf/2609.30913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. PORL: Pretrained Offline Reinforcement Learning for the Job Shop Scheduling Problem

**arXiv ID:** 2609.30948 | [PDF](https://arxiv.org/pdf/2609.30948v1)

**作者:** Mateo Toro Diz `[一作]` (Rosenheim University of Applied Sciences), Noah Klarmann `[通讯]` (Rosenheim University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了预训练离线强化学习（PORL）框架，将模拟环境下的在线预训练与离线数据微调相结合，以适应工业生产中的作业车间调度问题。

**💡 创新点**

创新点在于逆向训练顺序：先在线预训练后离线微调，并在微调阶段加入 KL 散度政策约束，限制与预训练策略的偏离，提升对低质量数据的鲁棒性。

**🔧 技术方法**

采用图神经网络（GIN）处理状态，DQN 与 CQL 结合的离线强化学习，并加入 KL 约束进行多目标优化；使用软max将 Q 转化为策略。

**📊 数据集**

数据集：在线预训练使用均匀分布的 15x15 JSSP；离线微调使用 10x10 的目标分布实例，包含机器处理时间分布、优先约束，生成的三套行为策略（启发式、噪声专家、随机）作为离线日志。

**📈 对比分析**

与基准比较：使用最优求解器、优先调度规则、基于 PPO 的在线 RL。PORL 在分布偏移实例上使最优性缺口下降 7.5% 于传统离线 RL，28.7% 于 PPO；在三种数据质量下，PORL 的性能优势随数据质量下降而增强。

**⚠️ 局限性**

局限：仅在小规模 10x10 以及单一分布偏移情景上验证，训练稳定性受多目标冲突影响，需要调参；依赖于可用的模拟环境，模拟与真实环境差异可能削弱迁移效果；KL 约束强度与其他约束方式尚未深入探讨。

---

## 330. LogicTree-RAG: Logic Tree-guided Retrieval-Augmented Generation for Long-form Patent Drafting

**arXiv ID:** 2609.30943 | [PDF](https://arxiv.org/pdf/2609.30943v1)

**作者:** Jiaqi Zhu `[一作]` (National University of Singapore), Beng Chin Ooi `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于逻辑树的检索增强生成框架（LogicTree‑RAG），可将研究论文自动转化为结构完整、技术信息充分、符合法律规范的专利全文。

**💡 创新点**

创新点在于：①引入全局层级逻辑树作为组织骨干，先决式构建概念层次再生成文本；②采用证据驱动的递归生成和节点感知语义检索，保证每一节点都有检索证据支持且避免重复；③设计混合遍历策略（BFS+DFS）实现章节平衡与细节展开，兼顾全局一致性与局部深度。

**🔧 技术方法**

使用技术包括：大语言模型（LLM）作为生成引擎；语义分块、向量检索（Milvus）构建知识库；节点感知语义检索与判别性相关性评分；递归生成与精炼过程；混合遍历策略。

**📊 数据集**

主要使用数据集为 Pap2Pat（1813篇论文‑专利对），并在 LCFO（多域长文本扩展）上验证通用性。

**📈 对比分析**

与13类基线（单次LLM调用、基于大模型的agent、手工纲要等）比较，LogicTree‑RAG 在覆盖率、事实性、语义相似度等内容级指标上均领先；在语言级指标上保持与最优基线相当，且每标记生成效率最高，能够输出更长、结构更完整的专利草案。

**⚠️ 局限性**

局限性：仍需人工专家审阅以确保法律合规；在专利实践中关键的权利要求结构与先例依赖等细节尚未完美自动化；目前仅在专利与长文本扩展上验证，其他复杂文档类型（标准、软件文档）需进一步研究。

---

## 331. Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations

**arXiv ID:** 2609.30918 | [PDF](https://arxiv.org/pdf/2609.30918v1)

**作者:** Marcin Kostrzewa `[一作]` (Wrocław University of Science and Technology), Maciej Zięba `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种统一的跨族评估协议，用以检验鲁棒反事实解释方法在不同模型更新情形下的稳健性。

**💡 创新点**

创新点在于把所有方法放在同一组八类模型变更上进行评估，并将鲁棒性、覆盖率、基础有效性等指标分离报告，揭示方法间的相对性能和失效模式。

**🔧 技术方法**

使用了参数扰动、重训练、架构变更、数据增删等八类模型变更，并评估六种鲁棒方法与两种基准，采用指标包括覆盖率、基础有效率、经验鲁棒性、端到端鲁棒性和最近邻距离。

**📊 数据集**

实验数据集为四个二分类表格数据集：Breast Cancer Wisconsin、Pima Indians Diabetes、Wine Quality 与 HELOC。

**📈 对比分析**

通过在固定的事实实例上生成CFE并在所有已持出模型变更上测试，结果显示不同变更类别下方法排名不稳定，参数扰动鲁棒方法在重训练等变更下表现下降，而RobX等方法在大多数变更上保持高鲁棒性但距离更大。

**⚠️ 局限性**

限制包括仅适用于二分类数值表格数据、未考虑多分类、图像或文本、不可变特征和时间成本，且仅评估最近邻距离而非稀疏性或可解释性。

---

## 332. ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning

**arXiv ID:** 2609.30906 | [PDF](https://arxiv.org/pdf/2609.30906v1)

**作者:** Zhenlong Dai `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的工具选择框架，用以在大规模工具仓库中进行多轮搜索与精细化优化。

**💡 创新点**

创新点包括三方面：①通过类别约束工具辨别提升模型对功能相似工具的区分；②事件级搜索建模显式强化搜索过程对工具组合的影响；③轨迹对齐奖励分配提供细粒度回报以引导搜索与选择。

**🔧 技术方法**

采用强化学习（PPO/GRPO风格）、事件级优势估计、轨迹级信用分配以及多轮搜索引擎调用的LLM策略优化。

**📊 数据集**

使用StableToolBench（16k工具、3种任务场景）和AppWorld（457个API的日常应用）进行评估。

**📈 对比分析**

与RAG、RAGSFT、Multi-turns、Search-R1、GSPO、GDPO、MARAG-R1等基线相比，方法在F1、召回率、精度、匹配度及OOD场景下的任务完成率均明显提升，尤其在多工具组合和未见指令场景中优势突出。

**⚠️ 局限性**

主要局限在于仍需依赖大规模算力训练，方法对不同LLM的通用性有限；在极端类别分布不均或工具接口复杂度高时，类别约束与搜索奖励仍可能不足；未深入研究实时动态工具更新的适应性。

---

## 333. Network Analysis in Communication Research: Research Topics, Knowledge Organization, and Research Practices

**arXiv ID:** 2609.30905 | [PDF](https://arxiv.org/pdf/2609.30905v1)

**作者:** Pengjia Cui `[一作]` `[通讯]` (University of California San Diego), Pengjia Cui (University of California San Diego)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 2,114 条 Web of Science 记录进行文献计量与定性阅读，构建了一个整合式综述，系统比较了通信网络研究在不同子领域（组织、新闻、受众、互动等）中的主题、方法与观察结果，并提出三条核心判断：连接的相关性取决于任务与价值标准；同一层面上的共同性可与另一层面的差异并存；在媒体使用影响的研究中，选择与内容共提并不能单独解释不一致的发现。

**💡 创新点**

创新点在于将网络分析的理论与实践跨子领域统一起来，既利用计量学方法揭示知识资源与引用模式，又通过对案例的“抽象审计”与“全文对照”实现从概念到方法再到结果的层层递进比较；同时提出了条件共引检验来捕捉资源之间的非随机关联，突破传统单一共引或共词分析的局限。

**🔧 技术方法**

主要技术包括：① 文献计量分析（关键词共现、引用共现、条件共引检验）；② 结构化抽样与全篇阅读（抽样审核、案例编码、引用语境阅读）；③ 网络可视化与图属性比较；④ 统计检验（Benjamini–Yekutieli、条件期望比值）；⑤ 主题与方法分类与编码。

**📊 数据集**

使用的数据集为 2,114 条 Web of Science Core Collection 记录，时间跨度 1973–2026，涵盖 1,905 篇文章、28 篇综述等。对其中的 40 条样本进行抽样审核，24 条研究案例进行全文阅读，4 条共引对照组共计 64 条记录，此外还检索了 2,114 条全文的关键词与引文信息。

**📈 对比分析**

比较方法采用分层抽样与条件共引检验，先在整个文献池中挑选频率 ≥ 40 的引用对，再通过随机化在年份内保持每条记录的引用数与每条引用的出现频率，计算观察值与期望值之比（OR）。与传统的共引热图相比，该方法能够识别出“局部”资源共现显著性，展示方法–理论、方法–方法等组合的相对优势。结果显示方法–方法组合出现频率高于期望，而方法–理论组合低于期望，暗示研究者在实践中更倾向于组合同类型资源。

**⚠️ 局限性**

局限性包括：① 选择性抽样（仅 40 条记录做全文审计）可能导致代表性不足；② 文献计量仅捕捉引用与关键词，无法反映实际研究内容与数据；③ 条件共引检验未考虑主题、期刊或研究设计的协变量，可能遗漏更深层次的解释；④ 多数比较基于抽象层面或自我报告，缺乏因果推断；⑤ 研究关注点分散，难以形成统一的“最佳实践”或理论验证框架。

---

## 334. How to break the Miranda signature scheme over matrix Gabidulin codes

**arXiv ID:** 2609.30925 | [PDF](https://arxiv.org/pdf/2609.30925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 335. Large language models underestimate and partly misrepresent cultural variation in everyday norms

**arXiv ID:** 2609.30896 | [PDF](https://arxiv.org/pdf/2609.30896v1)

**作者:** Kimmo Eriksson `[一作]` (Mälardalen University), Pontus Strimling `[通讯]` (Institute for Futures Studies)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过对全球90个社会中150个日常行为情境的社会规范评分进行人类基准，评估四款前沿大型语言模型（GPT‑5、GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro）在跨文化情境中的准确性，重点考察其对文化差异幅度的低估和对差异模式的识别弱化。

**💡 创新点**

创新点在于将大语言模型与真实跨文化调查数据直接对齐，系统量化其在跨文化社会规范预测中的压缩效应和“粗俗性”梯度效应，揭示模型在不同行为类型和发展水平社会中的性能差异。

**🔧 技术方法**

采用标准化提示模板，让模型生成对特定社会的平均情境评分，使用五次调用求平均并对回复进行解析；评估指标包括压缩比、跨文化相关系数、误差均值、水平偏差等统计。

**📊 数据集**

使用的核心数据集是2023–2024年收集的Global Study of Everyday Norms（GSEN），覆盖90个社会、约25,000名参与者、15种行为×10种情境共150个情境的评分。

**📈 对比分析**

对比方法是将模型估计与GSEN的社会平均评分做差值与相关分析；模型在跨文化差异幅度上平均压缩为0.47–0.54，跨文化相关系数仅0.23–0.33，表现远低于在单一社会的准确度（内群体相关≥0.86）。

**⚠️ 局限性**

局限在于：①人类基准是非代表性便利样本，主要为大学生；②模型未使用语言或文化背景提示的调节（非本地语言提示提升有限）；③评估侧重于平均评分，未考察个体差异；④模型的推理配置和温度参数未统一，影响可比性。

---

## 336. Cross-Backend QIEO: Universal Runtime Portability across OpenMP5, CUDA, HIP, and Multi-Language Interfaces

**arXiv ID:** 2609.30914 | [PDF](https://arxiv.org/pdf/2609.30914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 337. Odd Cycle Transversal on $H$-free graphs

**arXiv ID:** 2609.30900 | [PDF](https://arxiv.org/pdf/2609.30900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 338. PHASE: Compliance-Enabled Tactile Phase Retrieval for Few-Shot Insertion Learning

**arXiv ID:** 2609.30889 | [PDF](https://arxiv.org/pdf/2609.30889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 339. DAPEVO: Deep Adaptive Patch Frame-Event Visual Odometry

**arXiv ID:** 2609.30947 | [PDF](https://arxiv.org/pdf/2609.30947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 340. Co-design of trajectory and morphology for a vertical jump-climbing robot

**arXiv ID:** 2609.31014 | [PDF](https://arxiv.org/pdf/2609.31014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 341. Evidence-Grounded Auditing of Identification Assumptions in Climate-Policy Causal Evaluations

**arXiv ID:** 2609.30867 | [PDF](https://arxiv.org/pdf/2609.30867v1)

**作者:** Yonghong Zhang `[一作]` (Universidad Autonoma De Madrid), Ricardo Correia `[通讯]` (Universidad Autonoma De Madrid)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建并评估了一套基于大型语言模型的审核流程，对差分中的差分（DID）研究的11个识别假设进行结构化的证据审计，并生成可供专家审阅的风险报告。

**💡 创新点**

创新点包括：①提出可操作的11维识别假设–含义–证据rubric；②设计受限检索–门控–两阶段评估管道，确保模型仅在检索和评估阶段调用；③采用缺陷注入与人工标注的局部基准，对审核器进行系统化验证；④引入明确的停机机制（当检索不到相关证据时不作判断）。

**🔧 技术方法**

技术实现主要使用 GPT‑4o 等大型语言模型；受限检索（lexical retrieval）+相关性门控；两阶段评估（评估证据充分性并给出风险等级）；固定流程控制（无自由行动的 Agent）；缺陷注入脚本和评估脚本；统计比较与精确度、召回率等指标。

**📊 数据集**

数据集：1）CausalVerify 语料库中 26 篇标记为 DID 的论文；2）合成的 11 种单一缺陷和 33 种多重缺陷的检验样本；3）人工标注基准（5 篇论文 × 11 维，共 55 细胞），两位外部评审者独立标注并再调和。

**📈 对比分析**

评估方法：与关键词匹配基线、单通道（single‑pass）与多通道（per‑dimension）三种架构对比；在缺陷注入实验中检测率从 0.18（关键词）提升至 0.73–1.00（LLM + 门控），误报率保持低；在真实论文中检索覆盖率约 60%，40% 细胞因检索失败被标为 abstain；在人类标注基准上，未经校准时误判率高，应用预设规则后精确度从 0.24 提升至 0.76，权重 Kappa 从 0.08 提升至 0.23。

**⚠️ 局限性**

局限性：①样本规模有限（人工标注仅 5 篇，真实论文 26 篇），导致统计置信区间宽；②检索机制仅基于文本片段，导致覆盖不足，检索失败时可能误判为高风险；③缺陷注入文本为作者自创，缺乏独立专家编写的缺陷实例；④仅针对 DID 设计，未验证是否适用于其他准实验设计；⑤模型在不同供应商间误报率差异大，需重新校准；⑥未在真正的气候政策评估论文上验证效果。

---

## 342. Attacking Diophantus: Special Cases of Bag Containment

**arXiv ID:** 2609.30956 | [PDF](https://arxiv.org/pdf/2609.30956v1)

**作者:** George Konstantinidis `[一作]` (University of Southampton), Fabio Mogavero `[通讯]` (University of Naples Federico II)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个统一的框架，用来判定布袋语义下的连接查询（join-uniform）与任意连接查询之间的包含关系，并给出了相应的决策算法。

**💡 创新点**

创新点在于：① 把包含问题转化为在最小统一闭包上的有限算术表示；② 通过多重规范实例（multicanonical instances）将计数问题化为单项式-多项式不等式；③ 解决了在带权偏序上的特殊Diophantine不等式问题，克服了传统Hilbert第10问题导致的不可判定性。

**🔧 技术方法**

核心技术包括：最小统一闭包、偏序（multiplicity poset）与其权重、Möbius逆变换、强非负性多项式分析、线性化并求解同余Diophantine系统；算法复杂度分析（双指数、指数、Σ2^P）。

**📊 数据集**

本研究完全为理论性，未使用实验数据集，仅在形式化模型与理论上进行验证。

**📈 对比分析**

与之前仅针对投影无或join-无自由查询的判定方法相比，本文的框架覆盖更广（join-uniform），并保持或略高的复杂度（bag‑set: 2EXP，bag‑bag: Σ2^P），证明了这些问题的可判定性并给出了对应的上界；对比分析表明该方法在理论上是最完整、最通用的判定方案。

**⚠️ 局限性**

局限性包括：仍未突破到完整的所有连接查询；复杂度仍为双指数或高阶多项式级别，实际实现难度大；未结合对包含查询的限制，无法解决更一般的未判定案例；未来研究需要进一步压缩闭包规模或寻找更宽松的可判定边界。

---

## 343. Quadruped Obstacle Avoidance and Footstep Planning with Distributed Low-cost Time-of-Flight Sensors

**arXiv ID:** 2609.31008 | [PDF](https://arxiv.org/pdf/2609.31008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 344. Metacognitive Selective Ensemble for Mobile Systems

**arXiv ID:** 2609.31031 | [PDF](https://arxiv.org/pdf/2609.31031v1)

**作者:** Sungmin Lee `[一作]` (Yonsei University), JeongGil Ko `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向移动设备的主动集成框架（Metacognitive Selective Ensemble），通过保持一个活跃模型子集并仅在检测到成员失效时进行替换，实现连续传感流下的高效推理。

**💡 创新点**

创新点包括：
• 利用模型可靠性的短期持续性，保持活跃集并仅在必要时触发拒绝-路由；
• 后执行拒绝器利用输出置信度、马氏距离、同伴一致性和短期稳定性等多信号做无前向搜索的失效检测；
• 预先构建类别条件替换表，完成无前向搜索的替换决策；
• 通过状态化的 reject‑then‑route 机制在资源受限设备上保持低延迟和低内存消耗。

**🔧 技术方法**

技术细节包括：
• 深度学习模型池（Transformer、CNN、GRU、MLP）
• 基于软最大概率、概率间距、马氏距离比、同伴一致率和预测稳定性的多信号拒绝器
• 类别条件路由表（无前向搜索）
• margin‑weighted soft voting 作为最终决策
• 嵌入式评估（Raspberry Pi 4B）

**📊 数据集**

使用四个公开人体动作识别（HAR）数据集：HHAR、UCI‑HAR、WISDM、MOTIONSENSE。

**📈 对比分析**

与固定 K=3 集成、完整 N=10 soft voting、KNORA‑U、META‑DES、Softmax Response、Ens‑CLRF、FilterAct 等基线比较。在 K=3 的计算预算下，
• 在 14/16 配置中均优于固定集成；
• 在完整 10 模型预算下，精度与之相当且误差不超过 0.8%；
• 在 Raspberry Pi 4B 上，比完整 10 模型快 2.7×、内存节省 69%。

**⚠️ 局限性**

局限性：
• 对部署时分布漂移的鲁棒性不足，missed rejection 仍是主要错误来源；
• 替换决策仅基于历史信息，缺乏对当前输入的实时证据，可能错过最佳替换；
• 需要预先训练拒绝器和路由表，且在极端场景下可能失效。

---

## 345. Same Text, Different Numbers: The Divergence of LLM-Based Measures

**arXiv ID:** 2609.31013 | [PDF](https://arxiv.org/pdf/2609.31013v1)

**作者:** Hamid Boustanifar `[一作]` (EDHEC Business School), Sasan Mansouri `[通讯]` (University of Groningen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了七种不同大型语言模型（LLM）在对S&P 500公司2024年财报电话会议文本进行13项财务文本测量时的可重复性与一致性。

**💡 创新点**

首次系统评估多模型间测量一致性，并揭示模型选择对变量水平、相对排名、以及后续经济推断的显著影响。

**🔧 技术方法**

利用API直接调用的生成式LLM（GPT‑4、Claude、Gemini、Llama、Mistral、DeepSeek、Qwen），并在统一提示和尺度下对文本进行评分。

**📊 数据集**

数据集为2024年1,946场S&P 500公司财报电话会议问答文本，结合传统字典式测量、财务与市场数据。

**📈 对比分析**

通过相关系数、协方差分解、PCA、回归检验等多种统计方法比较模型间一致性，结果显示平均Spearman相关仅0.52，模型间差异占比分别为约33%、34%与32%，对经济结果的系数估计差异可达正负60%。

**⚠️ 局限性**

局限性包括仅检验API版LLM、未覆盖所有最新模型、未验证模型是否因知识截止导致偏差、以及未与人工编码的“真值”直接对照。

---

## 346. Breaking the Black Box: Byte-Level Boundary Inference of Real-World Antivirus Systems

**arXiv ID:** 2609.31012 | [PDF](https://arxiv.org/pdf/2609.31012v1)

**作者:** Jieshuai Yang `[一作]` (Nankai University), Wanpeng Li `[通讯]` (University of Liverpool)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出 AVHunter 框架，能够在黑盒条件下推断真实 AV 产品的字节级决策边界，并构建了首个大规模字节级 AV 边界数据集（BABD）。

**💡 创新点**

创新点在于：①首次实现对 AV 决策边界的细粒度字节级推断；②构建了覆盖 11 款主流 AV 的大规模 BABD 数据集；③通过边界标注实现了比传统二分类模型更高保真度的 AV 代理。

**🔧 技术方法**

核心技术包括：基于二进制搜索的边界点探测与区域扩展；多尺度卷积编码器‑注意力融合‑解码器结构 AV‑DBNet；使用焦点损失、空间惩罚和分类损失实现联合学习；对抗样本数据增强等。

**📊 数据集**

使用 BABD 数据集，该数据集由 BODMAS、SOREL‑20M 及 VirusShare/ VirusTotal 的恶意样本和商业 PE 软件的良性样本组成，包含 11 款 AV 的字节级边界标签。

**📈 对比分析**

与传统基于二分类反馈的模型（FFNN‑TL、dualFFNN）比较，AVHunter 在 11 款 AV 上平均检测一致率达 97.43%、边界召回率 85.07%，在边界引导的规避（BR‑Encryption）中平均规避率 83.78%，在诱导误报中误报率 81.33%，并在七个月内仅出现约 1.6% 的性能衰减。

**⚠️ 局限性**

主要局限包括：仅关注静态检测边界，无法覆盖动态/云端检测；对边界点数量有限制（≤20），可能偏向简单规则；查询成本较高（约 66.5 次查询/样本）；受限于数据集偏好，可能未能充分覆盖复杂或罕见的 AV 规则。

---

## 347. TRACKGRAPH: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking

**arXiv ID:** 2609.31005 | [PDF](https://arxiv.org/pdf/2609.31005v1)

**作者:** Peder Borge Hellesylt `[一作]` (Norwegian University of Science and Technology), Annette Stahl `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建实时在线开放词汇3D场景图，使用FastSAM在稀疏关键帧提取无类别掩码，并利用DINOv3密集特征在图像空间追踪掩码，随后将追踪到的源轨融合进类无关3D段层，维护多视角CLIP特征图册，实现空间关系建模与对象检索。

**💡 创新点**

① 在图像空间追踪掩码并保留短期身份，避免每帧完整分割与推理；② 采用DINOv3密集特征实现高帧率掩码传播；③ 通过层次化场景图的类无关段层和长时重识别，提升实例持续性和空间一致性；④ 用CLIP特征图册进行多视角检索，提升开放词汇检索效果。

**🔧 技术方法**

FastSAM（无类别分割）、DINOv3（密集视觉特征）、CLIP（跨模态嵌入）、Hydra（在线重建与场景图骨架）、TSDF体素融合、Marching Cubes、k-d树重识别、Voxels-based 关联等。

**📊 数据集**

Replica、ScanNet++、HM3D三大室内数据集；以及真实四足机器人（RealSense + LiDAR + IMU）和无人机记录的RGB‑D数据。

**📈 对比分析**

与ConceptGraphs、HOV‑SG、OVI‑MAP、FindAnything等基线在Replica上对比，系统实现1.7×速度提升、3.3×GPU内存下降；在分割与检索指标上保持竞争性或领先（如Replica上的同义词频率最高，HM3D检索AP略低但仍优于部分基线）。

**⚠️ 局限性**

在大型多样化场景（HM3D）检索AP下降，掩码覆盖不完全；对遮挡、动态物体仍敏感；依赖关键帧间传播，长时间无关键帧可能导致身份漂移；未在极端光照或噪声条件下充分验证。

---

## 348. GitHub Engagement Signals for CVE Prioritization: The GitHub Popularity Metric (GPM)

**arXiv ID:** 2609.31004 | [PDF](https://arxiv.org/pdf/2609.31004v1)

**作者:** Jafar Akhoundali `[一作]` (Leiden University), Olga Gadyatskaya `[通讯]` (Leiden University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于 GitHub 上公开漏洞利用仓库的用户交互（星标、fork 等）来衡量 CVE 热度的 GitHub Popularity Metric（GPM），用于漏洞优先级排序。

**💡 创新点**

创新点在于利用公开可采集的 GitHub 参与度作为预测已被利用漏洞的信号，构建了一个完全开放、无二进制阈值、预测性强、可直接与 EPSS、SSVC 等现有指标融合的度量。

**🔧 技术方法**

技术手段包括：GitHub Search API 深度爬取、唯一用户交互计数、GPM 计算公式、统计检验（Wilcoxon、Spearman 相关）、阈值优化（F1 最大化）以及与 KEV、EPSS、SSVC、CVSS 的交叉评估。

**📊 数据集**

使用的数据集包括 MITRE CVE/SSVC、GitHub 漏洞利用仓库、EPSS v3/v4、NVD、CISA KEV、VulnCheck KEV、Metasploit、Nuclei、AlienVault OTX 以及公开的勒索软件利用 CVE 列表。

**📈 对比分析**

通过将 GPM 与 CISA KEV 进行二分类比较（精确率、召回率、F1）并将阈值设为 12，F1 约为 0.51；与 EPSS 的 Spearman 相关性在 95% 情况下显著（ρ>0.5），与 SSVC 排名高度相关，并在 658 个 CVE 上发现其余指标未覆盖的高热度漏洞，证明其预测性能优于现有方法。

**⚠️ 局限性**

局限性包括：覆盖率仅约 2.5% 的 CVE，依赖 GitHub 可用性与 API 限制，可能受到虚假星标或恶意仓库噪声影响，且在漏洞公开前或仅存在私有利用码时仍会出现滞后。

---

## 349. Weaponizing Ground Truth: Data Poisoning Attacks by Exploiting Boundary Misalignment Between Antivirus Software and Learning-Based Detectors

**arXiv ID:** 2609.31003 | [PDF](https://arxiv.org/pdf/2609.31003v1)

**作者:** Jieshuai Yang `[一作]` (Nankai University), Wanpeng Li `[通讯]` (University of Liverpool)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Bi-Iocane框架，对上游AV标签生成链进行黑盒数据污染，利用AV与ML检测边界差异实现双向恶意/误报；

**💡 创新点**

将攻击从下游模型迁移至上游AV标注，且无需了解下游特征空间，支持侵蚀与诋毁两种方向；

**🔧 技术方法**

使用二分搜索发现关键边界字节、覆盖/注入攻击、生成轻量变体并上传至多引擎聚合服务；

**📊 数据集**

使用241k个PE文件（120k恶意、120k良性）以及13个真实AV引擎与VirusTotal实验数据；

**📈 对比分析**

与八类ML检测器对比，攻击成功率≥92%，恶意/误报覆盖率高，且对干净集F1损失≤0.15%；

**⚠️ 局限性**

仅在Windows/PE环境验证，未覆盖Android/Linux；聚合规则与阈值不同；可能被数据治理与防御过滤降低。

---

## 350. STORM-Bench: Evaluating Online Video QA under Evolving and Incomplete Evidence

**arXiv ID:** 2609.30981 | [PDF](https://arxiv.org/pdf/2609.30981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 351. Factorized axis convolutional gated recurrent unit with dynamic adaptive pooling for remaining useful life prediction of rolling bearings

**arXiv ID:** 2609.30972 | [PDF](https://arxiv.org/pdf/2609.30972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 352. PICO: Projection-Informed Consistency Optimisation for 6DoF Surgical Tool Pose Estimation

**arXiv ID:** 2609.30989 | [PDF](https://arxiv.org/pdf/2609.30989v1)

**作者:** Lucy Fothergill `[一作]` (University of Leeds), Duygu Sarikaya `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 PICO 端到端模型，实现实时无标记手术工具 6DoF 姿态估计。

**💡 创新点**

通过投影一致性损失和点对点一致性损失两种代理任务，结合多任务学习提升几何一致性与鲁棒性。

**🔧 技术方法**

采用 U‑Net+ResNet50 编码器，辅以深度与分割任务；使用投影损失、点对点损失、几何旋转损失及多任务总损失进行训练。

**📊 数据集**

在 SurgRIPE 数据集（含大针驱动器和马里兰双极钳）上进行实验与评估。

**📈 对比分析**

与挑战中多阶段和单阶段方法对比，旋转误差排名第二，旋转精度与迭代方法相当，推理速度约 33 FPS，满足实时需求。

**⚠️ 局限性**

深度估计精度不足导致 ADD 指标偏低；裁剪重采样可能引入纵横比失真；公开数据集有限，限制了泛化和复现性。

---

## 353. VisTacAlign: Co-Training Dexterous Policies on Tactile Human and Robot Demonstrations

**arXiv ID:** 2609.30959 | [PDF](https://arxiv.org/pdf/2609.30959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 354. Tracing and Relearning Detection Evidence in Text-to-Speech Systems

**arXiv ID:** 2609.30983 | [PDF](https://arxiv.org/pdf/2609.30983v1)

**作者:** Eunji Shin `[一作]` (Ewha W. University), Jaegul Choo `[通讯]` (KAIST AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对F5-TTS–BigVGAN语音合成管线进行受控重合成，探究声学模型与声码器哪个阶段为深度伪造检测器提供更多区分信息，并通过声学模型微调和检测器适配验证其影响。

**💡 创新点**

首次通过固定声码器对比实验分离声学生成与声码器重构的检测证据，证明声学模型的更新可降低固定检测器的辨别能力，而检测器适配仍能恢复辨别能力；同时揭示不同声码器对检测分离度的影响。

**🔧 技术方法**

使用F5-TTS中的Diffusion Transformer (DiT) 作为声学模型，BigVGAN 作为声码器；对DiT进行对抗性与特征匹配损失的细调；对XLS-R SLS检测器进行自适应训练；评估时使用UTMOS、ECAPA-TDNN说话人相似度和Whisper WER 等指标。

**📊 数据集**

LibriSpeech（train-clean-100、test-clean）、VCTK（ASVspoof 2019 训练集与评估集）以及 VoxPopuli（800句英语议会演讲）作为训练与评估数据集。

**📈 对比分析**

通过固定检测器（XLS-R SLS、XLSR-Mamba、AASIST-L）的EER对比，展示声学模型微调后EER显著上升（最高 18.9%），而检测器在对细调输出进行适配后EER降至 1.7%（LibriSpeech）或 7.5%（VCTK）。同时对比声码器重构的EER，表明声学生成是主要分离来源。性能指标显示，声学微调不明显损害音质与内容质量。

**⚠️ 局限性**

仅评估单一声学模型与声码器组合，适配仅针对一种检测器架构；实验规模受限于少量数据集与训练种子，未考虑说话人多样性与其他声码器；未探索更广泛的对抗性训练策略。

---

## 355. SciHorizon-eLab: An Agentic Protocol-to-Task Compiler for Scalable Benchmarking of Scientific Embodied Agents

**arXiv ID:** 2609.30971 | [PDF](https://arxiv.org/pdf/2609.30971v1)

**作者:** Maokai Qin `[一作]` (Chinese Academy of Sciences), Hengshu Zhu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种“协议到任务”编译器，将自然语言实验协议转化为可执行、可验证的物理仿真任务，并基于此构建了包含300个实验室任务与15个协作任务的评测基准。

**💡 创新点**

创新点在于将实验协议语义保持与可执行任务合成、独立评估标准生成、以及多阶段仿真认证三大功能集成到同一流水线，实现协议级别到仿真级别的无缝转换与自动化评测。

**🔧 技术方法**

主要技术包括语义解析与环境构建、基于注册原子技能的可执行程序生成、独立成功条件规划与可视化验证，以及MuJoCo物理仿真下的任务认证与实例化。

**📊 数据集**

使用了51个源协议场景（涵盖液体处理、混合、固体操作、热控与装置交互等5大操作族），30种原子技能库，并生成300个已认证任务。

**📈 对比分析**

通过对三种视图-运动策略（π0.5、ACT、DP）在10个代表性任务上的任务成功率（SR）与步骤完成率（SSR）进行对比，最高的SR仅为49.7%，说明现有策略在实验室级长时程与协作任务中仍表现不佳。

**⚠️ 局限性**

主要局限性包括：任务成功率低、对对象几何与手势的泛化差、对人机交互事件的感知与响应不足，以及在部分任务仍需人工干预以完成语义匹配与场景校正。

---

## 356. Packet iSlip

**arXiv ID:** 2609.30960 | [PDF](https://arxiv.org/pdf/2609.30960v1)

**作者:** Marc Mosko `[一作]` `[通讯]` (University of California), Marc Mosko (University of California)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

论文提出并实现了对传统iSlip协议的改进版本piSlip，针对同时支持包（packet）和细胞（cell）接口的交叉线交换机，在不加速（no speedup）的情况下通过虚拟切穿（virtual cut‑through）减少包端的传输延迟。

**💡 创新点**

创新点在于：1）修改iSlip的grant指针策略，使得包端在收到最后一个细胞后才推进grant，从而实现虚拟切穿；2）引入Virtual Output Lists（VOLs）来管理包的重组和就绪状态；3）设计了包端输出状态机，将Cut‑Through与Queue模式结合，解决了包端饥饿与细胞端不公平的问题；4）在同等负载下保持与原始iSlip相近的缓冲需求，显著降低延迟并减小方差。

**🔧 技术方法**

技术手段包括：虚拟输出队列（VOQ）、虚拟输出列表（VOL）、状态机同步（输入/输出端状态机的Cut‑Through/Queue模式）、在iSlip的grant/accept指针上做细胞/包适配改造、使用离散时间仿真器Sim 2.01进行性能评估。

**📊 数据集**

使用了自定义的流量生成器（vcpacket），在仿真中产生几何分布的细胞/包流量，配置了2、8、16、32、64端口的交换机，在10%–95%端口利用率下，分别进行100,000个细胞时钟的仿真（前50,000忽略）。

**📈 对比分析**

对比方法：在同一拓扑、同一负载下，比较未修改iSlip、iSlip Pkt（包端仅重组）以及piSlip三种配置的平均细胞延迟、缓冲占用与方差。结果显示：piSlip在所有端口数和利用率下，平均细胞延迟比iSlip Pkt低10%–30%，与原始iSlip的缓冲占用相近，且延迟方差更小。

**⚠️ 局限性**

局限性：1）实验仅在无加速的交换机上；2）未测试低包端端口数或大规模端口（如64端口）下的稳定性；3）缺乏正式的正确性与活性证明；4）仿真中大部分配置仅跑5–10次，导致置信区间宽；5）未深入评估细胞端的公平性与突发性变化；6）VOL实现仍以链表形式，硬件实现可能需要CAM/哈希优化。

---

## 357. Predictive Rolling-Horizon Optimization for Commitment-Aware Model-Parallel Inference under Spatio-Temporal Edge Dynamics

**arXiv ID:** 2609.31018 | [PDF](https://arxiv.org/pdf/2609.31018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 358. PipeDRAM: A Data-Transposition-Free In-DRAM Architecture with Hardware/Software Pipelining

**arXiv ID:** 2609.30998 | [PDF](https://arxiv.org/pdf/2609.30998v1)

**作者:** Geraldo F. Oliveira `[一作]` (Huawei Research), Onur Mutlu `[通讯]` (ETH)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种新的PU‑D（Processing‑Using‑DRAM）架构PipeDRAM，能够直接在水平数据布局下执行位串行/并行的内DRAM计算，完全消除了传统PU‑D需要的运行时数据转置。

**💡 创新点**

核心创新在于：
• Mat‑Aware Bit Mapping（MABM）——在内存控制器上对每个cache line做确定性位重排，使得每个数据元素的所有位被局部化到同一DRAM芯片并跨mat分布，保持水平布局但满足PU‑D的操作数局部性；
• 软件辅助流水线调度——采用模数调度（iterative modulo scheduling）静态生成PU‑D指令的pipeline映射，利用DRAM mats作为流水线阶段，实现位级并行与位串行的混合调度；
• 分层内DRAM复制方案——结合MIMDRAM的mat‑to‑mat互连与LISA的子阵列互连，异步传播进位，将关键通信从流水线的关键路径中剔除。首次实现无需数据转置的PU‑D，同时兼顾位级并行和流水线。

**🔧 技术方法**

实现技术包括：
• MIMDRAM细粒度mat访问（mat isolation transistors, mat selector, inter‑mat interconnect）；
• LISA子阵列隔离与高速互连；
• SALP（subarray‑level parallelism）支持多子阵列并行；
• 迭代Modulo Scheduling算法生成静态pipeline表；
• MABM的位重排逻辑（Φ 与 Φ⁻¹）在控制器数据路径实现；
• 分层复制（inter‑subarray + inter‑mat）实现异步进位传递。

**📊 数据集**

使用12个真实工作负载，取自SPEC 2017、Phoenix、Polybench、Rodinia等基准，涵盖gemm、convolution、backprop、kmeans等，数据规模与精度为32位（部分8位）大规模数据集。

**📈 对比分析**

对比方法：使用基于CACTI的能耗模型和自研Cycle‑level模拟器，分别评估SIMDRAM、MIMDRAM、Proteus、CPU和GPU。PipeDRAM-Hier在12个应用上平均提升11.8×性能、降低25.4×能耗，80.4×相对Proteus；相对于CPU、GPU的能效分别提升356×和11.7×。DRAM芯片面积增量仅1.86%，CPU die面积增量0.05%。

**⚠️ 局限性**

主要限制：
• MABM需要在内存控制器实现位重排，虽无额外数据移动但增加了控制器逻辑复杂度；
• 对低位宽（≤8位）操作，流水线利用率下降，部分mat闲置；
• 单批次（B=1）时进位传播仍有约2–30%暴露；
• 需要在DRAM芯片内部做细粒度mat访问与子阵列互连的硬件改造，兼容性与成本需进一步验证。

---

## 359. TempQ-Jail: Query-Constrained Candidate Ranking for Text-to-Video Jailbreak Attacks

**arXiv ID:** 2609.31032 | [PDF](https://arxiv.org/pdf/2609.31032v1)

**作者:** Tianmeng Fang `[一作]` (Singapore Management University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了TempQ-Jail方法，解决文本到视频的黑盒越狱攻击在查询预算受限下的候选分配与排序问题。

**💡 创新点**

将越狱候选的生成与查询优先级统一，提出异质候选构造、端到端攻击价值评估和预算感知排序三模块，显著提升在有限查询下的严格攻击成功率。

**🔧 技术方法**

采用多种攻击机制合成异质候选，使用Sentence‑BERT编码和多头MLP surrogate模型进行安全门通过、风险与时效评分，结合预测不确定性与Gumbel噪声实现候选排序。

**📊 数据集**

在CogVideoX‑5B文本到视频模型上，使用T2VSafetyBench生成的70个可行意图作为测试集，搭建GPT‑4O‑Mini安全过滤门。

**📈 对比分析**

与DACA、T2V‑OptJail、SceneSplit、SPARK、TFM、BSB六种基线在相同模型、过滤门、预算下对比，TempQ‑Jail在TP‑ASR@5/10、AUC‑TP、AvgQ上均领先，最高TP‑ASR@10为65.4%，AUC‑TP 0.469，AvgQ 6.3。

**⚠️ 局限性**

方法依赖预训练的surrogate模型和固定候选池，无法在线更新；仅在单一目标模型和安全门上验证，缺乏对其他T2V系统或更复杂过滤策略的鲁棒性评估。

---

## 360. Incipit: Axiom-Grounded Scaffolding for Human-AI Literary Creation

**arXiv ID:** 2609.31007 | [PDF](https://arxiv.org/pdf/2609.31007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 361. Precision at Speed: Sample-Efficient Online Model-Based Reinforcement Learning for Hydraulic Excavator Control

**arXiv ID:** 2609.31025 | [PDF](https://arxiv.org/pdf/2609.31025v1)

**作者:** Claudio Canales `[一作]` (ETH Zürich), Javier Ruiz-del-Solar `[通讯]` (Universidad de Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

该论文提出了一种在线模型基强化学习框架，直接在11.5吨液压挖掘机上从零开始学习精准高速轨迹跟踪控制。

**💡 创新点**

创新点在于结合概率动态集合、精度门控的轮廓奖励以及即时MPPI规划，实现样本高效、速度优先的精确轨迹控制。

**🔧 技术方法**

使用的技术包括概率动态集合模型、MPPI采样规划、指数移动平均滤波、精度门控奖励、在线经验循环与JIT编译的实时推理。

**📊 数据集**

主要数据集为实际机器的在线采样数据，实验中没有预先的仿真或演示数据，仅利用随机正弦探索收集的真实轨迹。

**📈 对比分析**

与基准模型（如TD-MPC2、DreamerV3、MBDPO）相比，在相同的交互时间内实现更低的预测误差和更高的控制精度，20分钟内达到与100–150分钟训练相当的跟踪精度，40分钟后保持亚厘米级误差并实现高达180 cm/s的速度。

**⚠️ 局限性**

主要局限包括对多任务或接触丰富环境的适应性不足、对不同负载或工具–土壤交互的鲁棒性未知，以及对模型不确定性利用不足导致的探索与风险管理受限。

---

## 362. Robust Successor Features

**arXiv ID:** 2609.31016 | [PDF](https://arxiv.org/pdf/2609.31016v1)

**作者:** Erik Nikulski `[一作]` (Universitat Pompeu Fabra), Javier Segovia-Aguas `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了鲁棒后继特征 (Robust Successor Features, RSF)，统一了转移学习与鲁棒强化学习，能够在奖励与转移动态同时变化的任务中实现零样本泛化。

**💡 创新点**

在线性MDP假设下，将奖励与转移动力学分别用不同特征表示，并通过 RSF 学习一个任务无关的表示，使得同一模型可对奖励和转移的任意组合进行零样本预测。

**🔧 技术方法**

结合 Generalized Policy Improvement (GPI)、深度 Q‑网络 (DDQN) 与 n 步 TD 学习，构建 RSF 训练框架，并给出理论子最优性界限。

**📊 数据集**

在经典 3×4 随机 Gridworld 环境中，对不同滑动概率 p 与步进奖励 r 的组合进行训练与测试。

**📈 对比分析**

与 Universal Successor Feature Approximator (USFA) 以及奖励/转移单轴限制的 RSF ablation 对比，RSF 在 Q‑值 MAE 0.063、策略准确率 93.6% 上明显优于其他方法（USFA 0.131、86.7% 等）。

**⚠️ 局限性**

假设奖励与转移可线性分解且特征已知，限制了对连续空间或非线性动力学环境的适用性，且当前未联合学习特征映射。

---

## 363. FARE: Forensic Acceptance Region Estimation for Catching Bait-and-Switch Image Generators

**arXiv ID:** 2609.30982 | [PDF](https://arxiv.org/pdf/2609.30982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 364. G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation

**arXiv ID:** 2609.31009 | [PDF](https://arxiv.org/pdf/2609.31009v1)

**作者:** Ruikang Liu `[一作]` (ZTE Corporation), Xiangsheng Zhou `[通讯]` (ZTE Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的后训练量化框架G2PTQ，利用块级全局监督、刷新梯度与Hessian以及通用梯度补偿实现对LLM权重的高精度量化。

**💡 创新点**

创新点在于：①将第一阶梯度与二阶Hessian同时引入块级优化；②在每个Transformer块前刷新梯度/Hessian，避免旧估计导致的失效；③采用自适应trust‑region缩放避免梯度补偿导致的数值爆炸；④提供高效的递推实现与lazy‑batch更新，使复杂度与GPTQ相当。

**🔧 技术方法**

技术包括PTQ、GPTQ/FOEM/GuidedQuant的改进、块级MSE/KL损失、Hessian外积近似、Cholesky分解、递推计算F、lazy‑batch与自适应trust‑region缩放。

**📊 数据集**

使用多种大规模LLM（0.6B~125B）和Mixture‑of‑Experts模型进行评测；在下游任务中使用标准Commonsense QA benchmark和PPL/KL指标。

**📈 对比分析**

与RTN、GPTQ、GuidedQuant、GTAQ等基线对比，G2PTQ在W2A16和4‑bit权激活量化下均实现了更低的KL、显著提升的QA准确率（如平均提高6.45%），在MoE模型中误差更小。

**⚠️ 局限性**

局限性包括：需要对每个Transformer块执行一次反向传播以刷新梯度/Hessian，导致校准时的计算与内存开销略高；trust‑region阈值需手动调参；目前主要验证了权重量化，激活量化仍需进一步研究。

---

## 365. PhoenixSR: Generative Heterogeneous Distillation Unleashes Efficient Models for Real-World Super-Resolution

**arXiv ID:** 2609.30988 | [PDF](https://arxiv.org/pdf/2609.30988v1)

**作者:** Xin Di `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将预训练的扩散模型的生成先验迁移到不含扩散组件的超分网络，提升感知质量同时保持重建精度。

**💡 创新点**

采用分布匹配蒸馏而非特征/输出对齐，并结合LoRA适配真实分数、REPA正则化假分数、样本级真实性锚定与方向可靠性加权，实现对多种网络结构的无架构依赖迁移。

**🔧 技术方法**

分布匹配蒸馏(DMD)、LoRA、REPA正则、样本级真实性锚定(SFA)、方向可靠性加权(DRW)、Stable Diffusion 2.1 Teacher、对抗/感知/像素损失等。

**📊 数据集**

RealSR、DRealSR、DIV2K‑Val 三大真实超分基准，训练集为 DIV2K、Flickr2K、OST、WED、FFHQ、Manga109、SCUT‑CTW1500。

**📈 对比分析**

在 SwinIR、HAT、Real‑ESRGAN、SeeMoRe 等六种前馈 SR 体系上与传统 Diffusion‑based SR 进行对比，PhoenixSR 在 PSNR 约 +0.2~0.5 dB 的同时，LPIPS、MUSIQ、NIQE、CLIP‑IQA 等感知指标显著提升（最高 10‑15%），且模型参数、FLOPs 与推理时间保持不变。

**⚠️ 局限性**

仍需依赖大型扩散教师与配对训练数据；对极端噪声/高倍率的鲁棒性尚未充分验证；蒸馏过程复杂，训练成本高。

---

## 366. Where and When to Force: Routed Forcing for Streaming Avatars

**arXiv ID:** 2609.30963 | [PDF](https://arxiv.org/pdf/2609.30963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 367. Evaluating Sycophancy in Chinese Large Language Models on Factual Questions Derived from Online Search Queries

**arXiv ID:** 2609.30986 | [PDF](https://arxiv.org/pdf/2609.30986v1)

**作者:** Geng Liu `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了中文LLM在面对错误用户信念时的事实真诚性，并评估了反真诚提示的效果。

**💡 创新点**

首次系统性对中文LLM的事实真诚行为进行分阶段匹配转移分析，区分正确、错误与不确定的转移路径。

**🔧 技术方法**

采用基于提示的实验设计，开启/关闭推理、三种前沿中文LLM（Qwen、DeepSeek、Doubao）以及两种对抗真诚指令。

**📊 数据集**

使用从中文搜索查询提炼的12,165条是非题事实问题（T2Ranking）共产生364,941条模型回答。

**📈 对比分析**

通过匹配转移率和逻辑回归比较，发现错误信念会导致错误对齐回答和正确转不确定，反真诚提示能减少错误对齐但往往增加不确定，整体效果模型差异明显。

**⚠️ 局限性**

仅评估有限模型、有限提示、受控回答标签、缺乏自然对话上下文和未考察用户后续影响。

---

## 368. CCRV-Bench: Constraint-Based Evaluation of Causal Reasoning in Vision-Language Models

**arXiv ID:** 2609.30979 | [PDF](https://arxiv.org/pdf/2609.30979v1)

**作者:** Linyuan Gao `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 CCRV-Bench，一个面向单帧图像的视觉因果推理基准，通过四个因果任务维度（因果关系发现、状态预测、因果诊断与干预）与四种约束（实体符号化、空间定位、事实对抗、最小化输出）对多模态模型的因果推理能力进行分层评估。

**💡 创新点**

创新点在于将因果推理任务与约束机制有机结合，设计了多维度、跨约束的评估框架，利用实体符号化、空间定位等手段显著抑制模型的 shortcut 学习；并首次量化不同约束对模型因果推理性能的影响，揭示 unconstrained 评价与真实因果能力的脱节。

**🔧 技术方法**

采用视觉因果图模型、Deep Causal Reasoning（DCR）与 Constraint Satisfaction Rate（CSR）评估方法；使用 GPT‑5.1 生成任务样本并手工审核；通过对比 CDI/eCDI 等指标分析模型在不同任务与约束下的性能变化。

**📊 数据集**

基于 Visual Genome 数据集提取视觉实体与关系，随后通过 GPT‑5.1 生成 800 条基线 QA 对，并扩展到 3,200 条约束变体；人类审核对 20% 样本进行验证，保证数据质量。

**📈 对比分析**

在 15 个主流 VLM（包含 6 个闭源与 9 个开放权重模型）上进行实验，结果表明：基线 DCR 高但受约束后性能显著下降；空间定位约束导致平均 DCR 降至 41.1%；事实对抗约束反而提升 DCR（平均 90.1%），而最小化输出和实体符号化也带来明显降幅；不同模型在约束遵从率与因果推理得分之间存在差异，展示了模型在约束环境下的多样化弱点。

**⚠️ 局限性**

局限性包括：仅针对单帧静态场景，未覆盖视频或连续时间的因果推理；数据主要来自 Visual Genome，可能无法覆盖高复杂度或专业物理场景；约束生成依赖 LLM 可能引入语言偏差；目前约束侧重语义、空间与逻辑 shortcut，未涵盖更广泛的时序与物理域挑战。

---

## 369. Synth-JEPA: Joint Embedding Prediction for Renderer-Free Synthesizer Parameter Search

**arXiv ID:** 2609.31024 | [PDF](https://arxiv.org/pdf/2609.31024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 370. The Linear Representation Hypothesis for Vision-Language-Action Models

**arXiv ID:** 2609.30996 | [PDF](https://arxiv.org/pdf/2609.30996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 371. Does Uniform Discrete Diffusion Need Time?

**arXiv ID:** 2609.30977 | [PDF](https://arxiv.org/pdf/2609.30977v1)

**作者:** Chunsan Hong `[一作]` (KAIST), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在统一离散扩散模型（UDM）中扩散时间（time）对预测的影响，并证明在有限训练数据下，时间敏感性在大多数扩散阶段几乎可以忽略。

**💡 创新点**

创新点在于：① 通过理论推导揭示 UDMs 的最优预测仅通过匹配上下文的数量来调节时间；② 提出有限数据下的 Fisher 敏感度上界，解释了时间依赖性在大多数时间点被压制的原因；③ 证明时间仅在高噪声端才具有显著作用。

**🔧 技术方法**

采用了统一离散扩散模型框架、留一（LOO）预测目标、时间敏感度（Fisher 信息）分析、以及多种时间参数化和混合时间调制策略。

**📊 数据集**

使用了公开语言数据集 OWT（约 860 万句子）和 LM1B（1 亿句子）进行实验，并在 Duality 与 LOO+CE 两个 UDM 变体上评估。

**📈 对比分析**

比较方法：在不同时间点计算 NLL 与 Fisher 敏感度；在整个训练周期评估验证 perplexity。实验结果显示：时间无关 UDM 在大多数时间段与时间相关模型持平，混合模型（仅在 t>0.8 时使用时间调制）几乎等同于完整时间相关模型，时间相关模型在 t≈1 处略有优势。

**⚠️ 局限性**

局限性：理论上限假设训练序列在高维空间足够分离；实验仅覆盖语言任务；高噪声端仍需时间信息；对不同 tokenizer 或更大模型的通用性仍待验证。

---

## 372. Governed Deduction: Policy-Grounded Premise Authorization Beyond Relevance

**arXiv ID:** 2609.31029 | [PDF](https://arxiv.org/pdf/2609.31029v1)

**作者:** Wesley Shu `[一作]` (Institute of Energetic Paradigm), Hsi-Ching Lin `[通讯]` (National Center for High-Performance Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了治理推理（Governed Deduction）框架，探讨在学习推理中如何区分前提的相关性与授权限制，并通过一系列对照实验验证该框架的有效性。

**💡 创新点**

创新点在于将授权约束视为与前提相关性分离的独立决策问题，定义了匹配的一侧控制（仅前提/状态或仅过渡）以及一个符号化策略或áculo，实现对授权机制的实验验证。

**🔧 技术方法**

技术上采用了固定词向量+线性逻辑回归的组合，对前提、状态、过渡进行词袋式特征化，并通过阈值优化与交叉验证进行模型选择。

**📊 数据集**

使用了来自 Spider 语料库的 RBAC-增强文本到 SQL 数据集，构建了 4,461 对匹配的授权对（同一前提状态下既有允许又有拒绝的过渡）。

**📈 对比分析**

实验对比了三种控制模型（仅前提/状态、仅过渡、联合），在经过角色名置换的泄露控制后，所有学习模型均跌至 50%（随机水平），而符号化策略则保持 100%，表明线性模型无法捕获授权关系。

**⚠️ 局限性**

局限性包括仅测试了线性特征+逻辑回归模型，未检验非线性或神经网络方法；数据来源于 RBAC 增强的文本到 SQL 任务，可能不具备对数学证明或自然语言推理的生态适用性；以及仅评估局部授权预测，未探讨其在完整推理流程中的实际安全或效能影响。

---

## 373. ZooWork-ShopRanker: An Open, Preference-Aligned E-Commerce Reranker

**arXiv ID:** 2609.31002 | [PDF](https://arxiv.org/pdf/2609.31002v1)

**作者:** Siqiao Xue `[一作]`, Ning Hu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者开发了一系列可扩展的电商检索重排序模型（0.6B、4B、8B），通过交叉 LLM 判别器标注的偏好对其进行对齐，并将旗舰模型蒸馏为更小的模型；

**💡 创新点**

创新点在于：①利用跨模型族的大型语言模型做“优先级-先决约束”判定，形成高质量的偏好标注；②将标注的偏好用于对齐，随后用教师-学生蒸馏实现高效推理；③设计了专门的双格式评测基准（ShopRank-Bench）及诊断轨道（属性层级、预算约束），细粒度验证模型对约束的遵循能力。

**🔧 技术方法**

技术包括：LoRA 低秩微调、对比式对数损失（Bradley‑Terry/RankNet）训练、对齐后对齐后的强化学习无关的对数回归，蒸馏使用软标签加二元交叉熵，再在标注对齐数据上使用 pairwise 损失微调；评估采用查询聚类自举和 McNemar 对比。

**📊 数据集**

数据集主要来自 Gensmo 商业搜索引擎的真实流量：约 4.4k 条 LLM 标注的偏好对，135k 软标签对齐样本，以及公开的 MTEB 重新排序任务；评测基准 ShopRank-Bench 包含约 10,511 条偏好对（分金、银、铜三层）及 1,500 条属性层级对和 1,302 条预算对。

**📈 对比分析**

与最强的开源重排序器（Jina‑m0、BGE‑Reranker‑v2‑m3 等）以及自身基线模型对比，8B 和 4B 模型在 ShopRank-Bench 上均明显优于对手；在结构化文本中 0.6B 与 4B 基线几乎相当，而在自然语言文本中 4B 仍保持优势；MTEB 任务亦显示对齐后模型在平均分上提升 0.5–1.0 分。

**⚠️ 局限性**

局限性包括：①对预算约束的遵循仍不理想，需额外的程序化监督；②标注过程依赖多大 LLM，成本高且易受模型偏见影响；③实验范围局限于 Gensmo 的商品类别，缺乏跨域验证；④在极端约束或多约束情形下，模型仍可能产生错误。

---

## 374. Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models

**arXiv ID:** 2609.30995 | [PDF](https://arxiv.org/pdf/2609.30995v1)

**作者:** Shan Zhao `[一作]` (Technical University of Munich), Julien Boussard `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种层次化因果表示学习框架，用于气候模型模拟海表温度，并通过全局与局部潜变量显式区分内部动力学与人为强迫的响应。

**💡 创新点**

在因果学习中引入全局潜变量与局部潜变量层级，显式建模CO₂、CH₄等全球温室气体与SO₂、BC等区域性气溶胶强迫，并通过GMST约束提升模型的可解释性和物理一致性。

**🔧 技术方法**

基于PICABU的单父亲稀疏因果图，加入层次化潜变量、增广拉格朗日约束、分解的ELBO、GMST约束以及贝叶斯过滤的自回归生成技术。

**📊 数据集**

使用NorESM2-LM月度海表温度与气候强迫（CO₂、CH₄、BC、SO₂）数据，训练阶段涵盖历史期及SSP1-2.6/SSP2-4.5，测试阶段采用OOV的SSP3-7.0情景。

**📈 对比分析**

与线性模式缩放（LPS）及无因果基线对比，模型在SSP3-7.0情景下的GMST趋势更贴近NorESM2，误差更低；同时准确重现2.5–5年ENSO周期以及区域气溶胶的加热/降温响应，展示出更优的物理一致性和预测性能。

**⚠️ 局限性**

限制包括：同一层级内不同强迫因子（如CO₂与CH₄）难以完全分离；缺乏真实的因果图做基准；仅在单一气候模型、有限变量（仅海表温度）上验证；对极端情景（如4×CO₂、SSP5-8.5）和多变量（降水、海平面压力等）评估仍待进一步研究。

---

## 375. Practical Deterministic Linear-Time Modular Subset Sum

**arXiv ID:** 2609.30992 | [PDF](https://arxiv.org/pdf/2609.30992v1)

**作者:** Phuoc Dinh Le `[一作]` (Georgia Institute of Technology), Kha Le `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种确定性 O(m) 时间、O(m) 空间的算法，用于在给定的可压缩输入上求解模 m 的完全子集和，并能返回任意可达残差的证据。

**💡 创新点**

创新点在于：① 将可达残差表示为沿循环加法生成的区间列表；② 通过按质因子增量扩展工作模数，将输入按最大公因子分组，极大减少重建循环的次数；③ 利用 DeVos‑Goddyn‑Mohar‑Šámal 等人关于可逆残差子集和的定理，证明边界列表的个数受限；④ 结合比较排序与基数排序的混合策略，在总排序时间仍为 O(m) 的前提下实现线性扫描。

**🔧 技术方法**

核心技术包括：循环区间维护、方向切换时的边界枚举、gcd 阶段的重建、批量排序（使用基数排序处理大批量、比较排序处理小批量）、父节点记录以实现证据回溯、以及预处理阶段的最小质因子表和逆元表。

**📊 数据集**

实验使用了三类合成数据集：随机支持（约 m/4 个不重复残差）、单一重复生成器（记录 (1, m‑1)）和 3 的幂次集合；每类在五个不同规模（1e5、2.5e5、5e5、7.5e5、1e6）以及两类模数（素数与合数）上进行测试。

**📈 对比分析**

与传统的 64 位位集 DP、哈希移位树和确定性移位树算法做比较。实验显示：在所有输入上，本算法的运行时间呈线性增长，并明显快于两种移位树实现；在随机支持和 3 幂次输入上，位集 DP 更快；但在重复生成器输入上，位集 DP 需要 O(m²/w) 的时间，而本算法保持 O(m)。

**⚠️ 局限性**

局限性包括：需要在支持常数时间模运算的 word‑RAM 环境下运行；对极大模数的内存占用仍为 O(m) 词，可能在内存受限的机器上受限；此外，算法在理论上是线性的，但实际常数受排序与区间合并实现细节影响；在稀疏输入（几乎空集）时，可能存在额外的预处理开销。

---

## 376. MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens

**arXiv ID:** 2609.30967 | [PDF](https://arxiv.org/pdf/2609.30967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 377. Gradient Surgery for Physics-Informed Neural Networks

**arXiv ID:** 2609.30966 | [PDF](https://arxiv.org/pdf/2609.30966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. FRAM: Trajectory-Guided Visual Feature Selection for Compact Language-Conditioned Robot Manipulation

**arXiv ID:** 2609.30965 | [PDF](https://arxiv.org/pdf/2609.30965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 379. IDM-Net: A Lightweight Illumination-Decoupled Modulation Network for Low-Light Image Enhancement

**arXiv ID:** 2609.30962 | [PDF](https://arxiv.org/pdf/2609.30962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 380. THA: Weighted Finite-State Text Normalization and Inverse Text Normalization for Khmer

**arXiv ID:** 2609.30984 | [PDF](https://arxiv.org/pdf/2609.30984v1)

**作者:** Seanghay Yath `[一作]` `[通讯]` (Digital Government Committee), Seanghay Yath (Digital Government Committee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文实现了一个基于加权有限状态机（WFST）的柬埔寨语文本正则化与逆向正则化工具包，可一次性处理整行文本并自动识别数字、货币、日期等语义类别。

**💡 创新点**

创新点在于：①无需单词分段器，采用全行最短路径搜索实现标签化；②设计边界过滤器避免半音节中断；③通过可选连接器和重复标记规则支持空格、零宽空格以及“…”重复词的准确展开与收缩。

**🔧 技术方法**

技术方法包括：Pynini编译加权转换器、短路搜索、语义类别权重、上下文约束过滤、可变连接器、字符顺序规范化等。

**📊 数据集**

使用的数据集包括：Google Khmer 语音合成测试套件（cardinals, decimals, times）、Google Khmer TTS 语料（句子已写成口语形式）、柬埔寨词典条目、以及随机生成的10类300条写文本。

**📈 对比分析**

评估方法包括与Google参考语法比较、对TTS提示句子逆向正则化检查、回环（round‑trip）验证以及消融实验；在所有类别中达到99.8%~100%的一致率，回环率在口语合成文本中超过99%，处理速度约为每秒1k字符。

**⚠️ 局限性**

主要局限包括：对拼写变体和打字错误的容忍度不足、对语义歧义（如电话号码、货币/重量、连字符用途）的处理仍依赖上下文或后置模型、未覆盖罗马数字、字母代码和缩写、缺乏大规模标注黄金标准。

---

## 381. Dynamic Task and Resource Scheduling Towards Space-Air-Ground-Sea Integrated Network

**arXiv ID:** 2609.31011 | [PDF](https://arxiv.org/pdf/2609.31011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 382. Self-Supervised Perceptually Interpretable Monocular Depth Estimation

**arXiv ID:** 2609.30987 | [PDF](https://arxiv.org/pdf/2609.30987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 383. FeatMark: Feature-level Watermark Protection against Mimicry Attacks with Diffusion Models

**arXiv ID:** 2609.30980 | [PDF](https://arxiv.org/pdf/2609.30980v1)

**作者:** Haoyang Li `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为FeatMark的水印框架，利用隐蔽的语义微特征对文本到图像扩散模型的模仿攻击进行可追溯性标记；

**💡 创新点**

创新点在于将水印从像素级能量极低的扰动转移到场景一致、可编辑的语义微特征，并通过自动特征库选择、概念编辑与图像隐匿相结合，实现了高隐蔽性与强鲁棒性的双重目标；

**🔧 技术方法**

核心技术包括基于CLIP的开放词汇特征检索与评分、基于概念程序的可编辑区域定位与指令化编辑、mask‑guided概念编辑、图像隐匿（image cloak）以及轻量化的水印阅读器训练；

**📊 数据集**

在VGGFace2、CelebA‑HQ和WikiArt三大数据集上进行评估，亦扩展至视频模仿攻击场景；

**📈 对比分析**

与传统像素级水印（StableSignature、HiDDeN、StegaStamp、Tree‑Ring、MetaCloak等）以及多种消除/净化攻击（IMPRESS、Noisy Upscaling、WEvade、UnMarker、Diffusion Attack、WatermarkAttacker等）相比，FeatMark在保持FID、精度/召回/覆盖/密度等感知质量指标接近无防御水平的同时，水印位错误率极低（≈0.01-0.04），对大多数净化与移除攻击的鲁棒性几乎不受影响；

**⚠️ 局限性**

局限性主要在于：1）对攻击者的后处理策略仍有限，极端或专门针对特征编辑的自适应攻击尚需进一步验证；2）对视频扩展的实验仍处于早期阶段，尚未评估长时序一致性与实时推理成本；3）特征库的构建与维护需要人工挑选与更新，可能导致对新领域的适应性不足。

---

## 384. LipSSM: Structurally Lipschitz-Bounded Cascaded State-Space Model via Metric Transfer between Consecutive SSM Layers

**arXiv ID:** 2609.30973 | [PDF](https://arxiv.org/pdf/2609.30973v1)

**作者:** Natsuki Yoshino `[一作]` (Tokyo University of Agriculture and Technology), Kohei Yatabe `[通讯]` (Tokyo University of Agriculture and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的 Lipschitz 连续状态空间网络（LipSSM），通过特定的系统矩阵参数化来保证网络整体的 Lipschitz 常数，同时保持对长时序依赖的建模能力，并在单通道非线性 IIR 系统辨识任务中验证其有效性。

**💡 创新点**

创新点在于：① 通过 Cayley 变换等构造层间权重，使得网络层间信息得以传递，从而比传统逐层约束获得更紧凑的全局 Lipschitz 上界；② 采用增量消耗理论对 SSM 进行参数化，确保网络在训练过程中始终满足预设的 Lipschitz 约束；③ 在保持约束的前提下，显著提升模型对长时序数据的表达能力。

**🔧 技术方法**

使用技术包括：状态空间模型（SSM）框架、增量消耗理论、Cayley 变换、权重矩阵构造、Spectral norm 约束、Jacobian 谱范数计算、自动微分与 Adam 优化器、Power 迭代估算谱范数。

**📊 数据集**

主要使用的实验数据为人工生成的单通道非线性 IIR 系统序列（长度 T=32），训练集 5000 条序列，验证集 1000 条；此外还对 LipKernel 进行了数值验证以比较 Lipschitz 上界。

**📈 对比分析**

对比方法：将 LipSSM 与 LipKernel 在相同网络深度、相似参数量和相同 Lipschitz 约束下进行比较，评估指标包括 NMSE（归一化均方误差）、参数量、推理时间以及实际测得的 Lipschitz 常数。实验结果显示：LipSSM 在 NMSE 上显著优于 LipKernel，尤其在高记忆系数（α=0.9）时表现更突出；参数量相近，推理时间略慢，但满足理论 Lipschitz 上界。

**⚠️ 局限性**

局限性：① 训练时间和计算成本显著高于传统卷积式 LipKernel（LipSSM 约 100 分钟 vs 10 分钟）；② 目前仅在小规模模拟数据上验证，缺乏对大规模真实序列任务的测试；③ 参数化过程较为复杂，可能在更大网络规模下导致实现与调参挑战。

---

## 385. Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance

**arXiv ID:** 2609.30997 | [PDF](https://arxiv.org/pdf/2609.30997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 386. Refining Cytology Predictions with Conditional Random Fields

**arXiv ID:** 2609.31028 | [PDF](https://arxiv.org/pdf/2609.31028v1)

**作者:** Manon Dausort `[一作]` (Université Catholique de Louvain), Benoît Macq `[通讯]` (Université Catholique de Louvain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 CytoCRF，一种针对细胞学图像分类的条件随机场模型，用来细化 Vision‑Language Models 在细胞图像上的零样本预测。

**💡 创新点**

创新点在于：①针对细胞学染色和染色体结构重新设计 pairwise term；②通过结合多种 backbone 构建邻域拓扑，进一步提升预测质量；③证明邻域拓扑对性能的影响超过单一 pairwise term 的贡献。

**🔧 技术方法**

使用技术包括：Vision‑Language Models（VLM）进行初步分类；Conditional Random Fields（CRF）对预测进行后处理；多 backbone 特征融合；对 pairwise term 进行染色体特定的重构。

**📊 数据集**

实验使用十个细胞学数据集，这些数据集覆盖多种染色协议，采用独立的 patch pools。

**📈 对比分析**

在所有注释预算下，与现有 CRF 框架进行比较，CytoCRF 领先；在最佳基线上提升 +13.6% 点，在仅 50 条注释的情况下比零样本方法提升 +33.7% 点。

**⚠️ 局限性**

局限性包括：需要多 backbone 计算，增加了模型复杂度和推理时间；模型主要针对细胞学图像，尚未验证其在其他细胞学子领域或不同数据源上的普适性。

---

## 387. TACTIC: Understanding Tactile Encoders and Conditioning for Contact-rich Robot Manipulation Policies

**arXiv ID:** 2609.30969 | [PDF](https://arxiv.org/pdf/2609.30969v1)

**作者:** Seongjin Bien `[一作]` (University of Technology Nuremberg), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在实际机器人上对五种视觉基触觉编码器和五种多模态融合策略进行系统评估，共完成了 2,180 次试验，覆盖了四种接触丰富的操作任务。

**💡 创新点**

首次提供了统一训练与部署环境下的全面对比，揭示不同任务对编码器与融合方式的特定需求，证明简单拼接仍具竞争力，并指出基于 CLIP 的对比融合会产生任务专属化而非通用性。

**🔧 技术方法**

使用 DIGIT 视觉触觉传感器、ResNet‑18、SARL、UniT、Sparsh‑DINO、T³ 等编码器，配合 Concat、FiLM、GCA、CLIP‑R/T 等融合策略，采用 Action Chunking Transformer（ACT）策略进行端到端控制，并对 CLIP 进行对比预训练及 PCA 分析。

**📊 数据集**

收集了四个任务（板擦拭、花瓶擦拭、USB‑C 插入、螺钉拧紧）各约 100 条遥控演示，使用四对 DIGIT 卡槽，构成 2,180 次真实世界试验；此外在可见度衰减与干扰物等 OOD 条件下再次评估。

**📈 对比分析**

通过每种条件 20 次实时推理，按抓取、尝试、完成、终止四阶段记录成功率；结果显示无单一最佳组合，Sparsh‑CLIP‑T 在螺钉任务上最高终止率 90%，ResNet18‑Concat 在 USB‑C 上达 80%，整体平均终止率约 50%，并在 OOD 情况下进一步验证了不同策略的鲁棒性。

**⚠️ 局限性**

研究受限于仅使用两摄像头和特定机械臂平台，数据来源局限于遥控演示，未涵盖更广泛的传感器与环境；对比融合需要精细采样，易导致泛化下降，且实验仍无法完全覆盖所有触觉需求的多样性。

---

## 388. FAVoR: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation

**arXiv ID:** 2609.30968 | [PDF](https://arxiv.org/pdf/2609.30968v1)

**作者:** Lu Han `[一作]` (University of Sydney), Nguyen H. Tran `[通讯]` (University of Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究作者风格同质化问题，提出FAVoR模型，在联邦学习中通过共享-私有适配器保持个性化写作；

**💡 创新点**

创新点在于引入作者风格残差机制与Angular Style Classification Encoder（ASCE）对齐，并构建共享-私有边界以防止风格同质化；

**🔧 技术方法**

采用联邦PEFT、LoRA适配器、ASCE风格编码、风格对齐损失、私有残差包和联邦聚合等技术；

**📊 数据集**

使用BlogText（Blog Authorship Corpus）和Mythos-Reddit写作提示数据集；

**📈 对比分析**

与FedAvg、FedProx、pFedMe、Ditto、FedDPA等基线比较，FAVoR在作者准确率、宏F1、外部验证AUC/EER显著提升，同时语义质量保持不变；

**⚠️ 局限性**

局限包括仅在英文写作、单一模型体系、未实现正式隐私保护、评估依赖预训练ASCE、作者风格测度仍有限。

---

## 389. FLIP: Final Layer Inference-Time Probing for Vision-Language Models

**arXiv ID:** 2609.30993 | [PDF](https://arxiv.org/pdf/2609.30993v1)

**作者:** Drandreb Earl O. Juanico `[一作]` (University of the Philippines), Rowel O. Atienza `[通讯]` (University of the Philippines)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 FLIP（Final Layer Inference‑Time Probing），一种在可开源视觉‑语言模型（VLM）最终层对隐藏状态做 elementwise flooring 并在输出层前插入的干预方法，用来验证干预是否导致结构化、任务相关的行为变化。它配合四条检验准则（regime 结构、grounding‑proxy 对齐、feature‑coherence 依赖、负控制对比）对干预效果进行系统筛选。

**💡 创新点**

创新点在于：①把干预结果视为可验证的实验对象而非自动解释；②设计了基于阈值扫掠的 probe‑and‑sweep 协议，能够区分真正的任务相关计算和无意义的扰动；③通过在最终层（post‑normalization）插入 Flooring 并对比原始 decoder‑layer、归一化匹配、负控制等，确认该方法只在特定“内部区域”产生正向改进，验证了干预的有意义性。

**🔧 技术方法**

技术手段包括：
• Elementwise flooring（max(z, θ)）在最终隐藏状态上实施；
• 在 θ 范围内进行 sweep，观察检测 recall R_50 与计数误差 ε_count 的变化；
• 线性输出头理论分析，验证干预仅对任务相关方向有投影；
• RMR（Relevance Mass Redistribution）诊断，用于跟踪特征重分配；
• Grounding‑proxy 统计（使用 detection recall 与 counting error 的关联），
• 负控制（左/右单词对比、feature‑coherence 混合 λ）和原始层对比验证。

**📊 数据集**

使用的数据集：
• 7,056 张 MS‑COCO val2017 图像，配合单对象查询的定位/计数提示；
• 其它 VLM 评测：Qwen3‑VL‑8B、Qwen3‑VL‑4B、Kimi‑VL‑A3B、MMStar、MindCube、NaturalBench。实验中对每张图像都做多次提示，确保统计独立性。

**📈 对比分析**

对比方法：在不同干预点（最终层 vs 原始 decoder‑layer）、不同阈值匹配（post‑norm vs percentile‑matched）以及负控制（单词对/feature‑coherence 混合）下比较。结果显示：
• 在最终层的内部区域（θ 约 –1 ~ 0）检测 recall R_50 明显提升、计数误差下降；
• 原始层或匹配阈值的干预未出现相同的“内部峰值”或直接导致性能下降；
• 在多模型实验中，该峰值在 Qwen3‑VL‑4B 与 Kimi‑VL‑A3B 中更为显著，Qwen3‑VL‑8B 的幅度最低。

**⚠️ 局限性**

局限性：
① FLIP 仅验证干预是否具有结构化、任务相关效应，未能直接定位具体电路或机制；
② 评估范围限定在定位/计数任务，难以推广到更广泛的 VLM 任务；
③ 对最终层的干预不排除早期层潜在信息被忽略的可能；
④ 需要大量统计样本才能得到可靠的 FWHM 区间；
⑤ 仅是一个筛选工具，后续仍需补丁、循环追踪或因果抽象等进一步工作。

---

## 390. AuthGuard-R: Safety-Compliant Mission Hijacking and Dual-Gate Defense for LLM-Controlled Robots

**arXiv ID:** 2609.31110 | [PDF](https://arxiv.org/pdf/2609.31110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 391. Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change

**arXiv ID:** 2609.30974 | [PDF](https://arxiv.org/pdf/2609.30974v1)

**作者:** Haruka Ezoe `[一作]` (University of Tokyo), Ryohei Hisano `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将多时期词语用法分布转化为单一时间保持的耦合过程CUSP，能够同时给出词义变化的幅度、时间、机制、具体成分迁移、词本地变化方向以及对应的文本证据。

**💡 创新点**

核心创新在于：1）利用层次最优传输与Markov组合构建一个全局保持每个时期边缘分布的时间过程；2）引入位移算子和分解公式，精确把变化拆分为成分中心移动、内部重组以及被传输的成分对；3）定义词本地模式并以此解释多时期变化；4）在高斯混合特化下实现解析表达与统计可恢复性。

**🔧 技术方法**

技术包括：层次最优传输、Markov链组合、位移算子与迹/模式距离、方差分解、词本地基向量（特征分解）、高斯混合模型估计、最大似然正则化、统计收敛证明。

**📊 数据集**

数据集：Synthetic Gaussian mixtures；DWUG（英德两语两期词义变化评分数据）；Janus（六期受控词义变化数据）；美国最高法院判决文本（按年代划分的法律词语使用数据）。

**📈 对比分析**

与APD、Prototype、AP-JSD、Usage-level OT、fSUS等传统词义变化评估方法比较；在DWUG上CUSP全迹与均值迹的Spearman相关系数分别为0.746/0.742（英）和0.815/0.809（德），与最优方法相当或略优；在Janus中恢复时间曲线精确度高，AUROC 1.0；在Court数据中能够定位显著迁移并提供可检验的文本片段。

**⚠️ 局限性**

局限性：1）仅提供整体过程，无法追踪单个用法实例的轨迹；2）依赖高斯混合假设和固定成分数，可能对高维分布或复杂语义变化不够敏感；3）对混合模型的估计对样本量和嵌入质量敏感；4）计算成本较高，尤其在多时期多词大规模语料下。

---

## 392. DualManip: Agentic Dynamic Manipulation via Dual-Path Semantic Reasoning and Geometric Adaptation

**arXiv ID:** 2609.31112 | [PDF](https://arxiv.org/pdf/2609.31112v1)

**作者:** Chengxi Li `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DualManip，一种双路径框架，将VLM的高层语义推理与实时几何适配分离，实现对动态场景下的意图一致操纵

**💡 创新点**

创新点在于：①将语义路径与几何路径解耦，避免频繁高延迟的VLM推理；②设计形状自适应对应网络，实现在非刚性变形下的点级匹配；③通过任务条件的接触点转移和可行性验证，实现在线抓取重构；④信息交互模块将两条路径桥接，必要时触发语义重规划

**🔧 技术方法**

使用技术包括：预训练VLM（GPT‑5.6 Terra）、SAM 3进行目标分割、Softar推断语义方向、AnyGrasp生成候选抓取、形状自适应对应网络（基于Sinkhorn归一化和全局刚性+形变变换）、Chamfer、SmoothL1等损失、ARAP正则化等；实现低延迟抓取更新（≈138 ms）

**📊 数据集**

实验采用自制的数据集：对每个对象采集多视RGB‑D，构建模板点云；每个对象收集约100帧RGB‑D进行对应网络训练；在六个真实任务（软蛇、玩具定位、块鱼、RAM插槽、GPU插槽、风扇定位）上进行15次试验

**📈 对比分析**

与ReKep、OmniManip、CLEA、Open‑loop四种基线比较；在静态场景下DualManip与最佳基线同等（77.8 %）或更优（assembly 51.1 %）；在单次变更下DualManip和CLEA均达64.4 %成功率，显著高于ReKep（42.2 %）和OmniManip（26.7 %）；在连续动态下DualManip实现53.3 %成功率，远超ReKep（28.9 %）和OmniManip（17.8 %）并且CLEA/Open‑loop失败；平均适配延迟为138.7 ms，远低于CLEA（6.376 s）和Qwen版CLEA（4.329 s）

**⚠️ 局限性**

主要局限是需预先构建对象模板并进行专门训练，导致无法零样本泛化到未见实例；对类别级对应学习的支持有限，限制了对新对象的即时适配

---

## 393. Accuracy Evaluation of INS/ZUPT Filtering Methods Based on Different Geometric Error Definitions

**arXiv ID:** 2609.31057 | [PDF](https://arxiv.org/pdf/2609.31057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 394. Where Compute Matters: Heterogeneous Attention for Efficient Video Diffusion

**arXiv ID:** 2609.31050 | [PDF](https://arxiv.org/pdf/2609.31050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 395. Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods

**arXiv ID:** 2609.31107 | [PDF](https://arxiv.org/pdf/2609.31107v1)

**作者:** Saksham Kiroriwal `[一作]` (Fraunhofer IOSB), Jürgen Beyerer `[通讯]` (Fraunhofer IOSB)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于信息几何的贝叶斯优化框架，利用后验映射的拉回费舍尔张量来分析并上界采集函数梯度。

**💡 创新点**

创新点在于将采集函数梯度分解为采集敏感度与拉回费舍尔张量迹的乘积，从而统一解释高维BO中梯度消失、尺度长度调节与RAASP等现象，并基于此设计了Fisher-Information Trust Region (FITR) 方法。

**🔧 技术方法**

主要技术包括信息几何、费舍尔信息矩阵、拉回度量、自然梯度、拉回费舍尔张量对角近似、对角泰科诺夫正则化以及与TuRBO相似的信任域优化框架。

**📊 数据集**

实验使用多维连续基准问题（如HPA102-1、MOPTA08、LassoDNA、SVM等）以及使用IBNN深度宽度无限核的基准。

**📈 对比分析**

与TuRBO-REI、DSP、BOUNCE等强基线比较，FITR在SE核GP实验中在多数任务上优于TuRBO-REI，并与其他基线保持竞争力；在IBNN实验中也表现出一定提升。

**⚠️ 局限性**

局限性包括缺乏外部BO循环的收敛保证、拉回费舍尔张量在高维下退化导致需对角近似、非等距核下梯度消失机制不如预期，以及全旋转Fisher域采样效率低和边界剪裁问题。

---

## 396. The Residual Stream's Effective Depth

**arXiv ID:** 2609.31098 | [PDF](https://arxiv.org/pdf/2609.31098v1)

**作者:** Barak Gahtan `[一作]` (Technion Israel Institute of Technology), Alex M. Bronstein `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种有效深度的标量诊断方法，测量变换器的层级残差流的表示相似性如何随着层距衰减，并将该特征聚合为一个数值。

**💡 创新点**

创新点在于将残差流视为离散时间过程，并通过层间相似性自相关来定义有效深度，提供了一种架构无关的冗余诊断。

**🔧 技术方法**

使用了中心化核对齐（CKA）作为相似性度量，并通过加权全滞后聚合来计算有效深度。

**📊 数据集**

使用了16个解码器语言模型的数据集，包括Pythia、OLMo-2、Qwen3.5等，涵盖了不同的架构家族。

**📈 对比分析**

通过与理想化的正交更新参考值F_L进行比较，发现大多数模型的有效深度低于该参考值，表明存在额外的冗余。性能表现为15个模型的有效深度在20%-44%之间，且模型家族间的差异显著。

**⚠️ 局限性**

限制在于该方法仅适用于解码器语言模型，且CKA对纯残差流旋转是盲目的，未来的工作需要探索不同的架构和更新规模对有效深度的影响。

---

## 397. Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents

**arXiv ID:** 2609.31076 | [PDF](https://arxiv.org/pdf/2609.31076v1)

**作者:** Bartłomiej Cupiał `[一作]` (University of Warsaw), Karthik R. Narasimhan `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在长时间跨度的文本驱动游戏环境中，对比低级原语、基于代码的高级技能以及两者混合控制的性能与成本；

**💡 创新点**

首次系统化评估代码技能在语言模型代理中的价值，并提供完整的代码技能库与可复现的实验框架；

**🔧 技术方法**

使用大型语言模型作为高层决策者，构造代码技能作为低层策略，并结合零样本推理、监督微调(SFT)与PPO强化学习；

**📊 数据集**

在NetHack Learning Environment与MiniHack任务上进行实验，覆盖14种不同规模的语言模型；

**📈 对比分析**

在零样本设置下，技能控制平均提升3倍进度、减少86%推理成本；混合控制保持95%性能同时仅提升约2倍成本；在RL阶段，技能与混合控制相较原语提升7–8倍进度；

**⚠️ 局限性**

技能库手工设计且不完整，无法覆盖所有游戏情境；仍远未能通关NetHack；缺乏自适应学习新技能的机制，限制了进一步进步。

---

## 398. AtomWorld-Mem: Memory-Restored World States for Long-Horizon Atomistic Evolution

**arXiv ID:** 2609.31133 | [PDF](https://arxiv.org/pdf/2609.31133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 399. Do we need to answer that question? Salience and Answerability of Potential Questions in Naturalistic Dialogue

**arXiv ID:** 2609.31130 | [PDF](https://arxiv.org/pdf/2609.31130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 400. SAGE: A sampling-aware global evaluation benchmark for species distribution modeling

**arXiv ID:** 2609.31082 | [PDF](https://arxiv.org/pdf/2609.31082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 401. SADRA: Sound Capability-based Access Control System for Resource-Disaggregated Architectures

**arXiv ID:** 2609.31119 | [PDF](https://arxiv.org/pdf/2609.31119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 402. Double-stream registration with pyramid fusion for HDR video with alternating exposures

**arXiv ID:** 2609.31108 | [PDF](https://arxiv.org/pdf/2609.31108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 403. Exploiting Spatial Structure for Transductive Few-Shot Classification of Whole-Slide Images

**arXiv ID:** 2609.31040 | [PDF](https://arxiv.org/pdf/2609.31040v1)

**作者:** Tiffanie Godelaine `[一作]` (Université Catholique de Louvain), Christophe De Vleeschouwer `[通讯]` (Université Catholique de Louvain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SlideTIM，一种针对病理全切片图像（WSI）的转导式细粒度分类改进方法，能够在极少数标注的前提下联合优化所有补丁的预测；

**💡 创新点**

创新点包括：① 引入空间-潜在正则化，使空间相邻且语义相似的补丁共享相同预测；② 用基于估计的类别先验的边缘熵正则化，主动将预测分布拉向切片中实际出现的类别；③ 在LC‑TIM框架上加入这两项正则，显著提升在高度不平衡、空间结构复杂的WSI上的表现；

**🔧 技术方法**

使用的技术主要是：Vision‑Language Model（VLM）用于零样本预测；转导式信息最大化（LC‑TIM）作为基线；交叉熵、互信息、KL散度正则；空间‑潜在混合相似度；先验锚定熵正则；以及Alternating Direction Method（ADM）实现闭式更新；

**📊 数据集**

实验使用了四个WSI数据集：BACH（乳腺癌分级）、CATCH（犬皮肤组织类型）、SKINCANCER（皮肤癌与组织类型）和TIGER（二分类乳腺癌与健康组织），所有数据均可分割为448×448补丁并有像素级标签；

**📈 对比分析**

与零样本（ZS）以及TIM++、LC‑TIM、α‑TIM等基线对比，SlideTIM在macro‑F1上在1 shot时提升约+19.4pp，16 shot时提升+34.6pp；相较于最佳基线α‑TIM，1 shot时提升+8.1pp，16 shot时提升+3.2pp；准确率和少数类别F1也均优于对手；

**⚠️ 局限性**

局限性在于假设标注集已覆盖切片中所有出现的类别，若路径学家遗漏某个类别则先验估计失效；此外方法依赖可生成补丁标签的WSI，未验证在无像素级分割的通用数据集上的适用性。

---

## 404. MetaPermit: Scalable and Auditable Access Control for AI Agents via LLM-Inferred Meta-Attributes

**arXiv ID:** 2609.31039 | [PDF](https://arxiv.org/pdf/2609.31039v1)

**作者:** Hanzhang Ma `[一作]` (Paderborn University), Debayan Roy `[通讯]` (Huawei Hilbert Research Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于MetaPermit的工具调用授权框架，利用LLM推断元属性并由静态策略评估，从而在不枚举用户意图的情况下实现可扩展且可审计的安全控制。

**💡 创新点**

创新点在于将语义推断与授权决策解耦：通过设计有限且任务无关的元属性集（MetaAttributes）让LLM仅输出受限值，静态ABAC策略做最终判断，从而显著提高一致性与可审计性，降低对攻击的易受性。

**🔧 技术方法**

主要技术包括：1）LLM推断模块（对每个候选工具调用生成元属性向量）；2）ABAC式的策略引擎（基于元属性评估授权）；3）执行门控（根据策略允许或拒绝调用并提供反馈）；4）基于MetaPermit的策略反馈机制；5）与AgentDojo/AgentDyn等基准对比实验。

**📊 数据集**

使用AgentDojo和AgentDyn两个公开基准，涵盖7个任务套件（workspace、travel、banking、slack、shopping、github、dailylife）以及5种prompt‑injection攻击（Direct、Ignore Previous、InjecAgent、Tool Knowledge、Important Instructions）共计7,545个攻击实例。

**📈 对比分析**

与无防御、CaMeL、IPIGuard以及直接LLM授权等三类基线进行对比。结果显示MetaPermit在MiniMax-M2.7和Qwen3-235B上将攻击成功率从最高26.49%降至0%，同时保持53–71%的任务完成率，比CaMeL（25–30%）和IPIGuard（30–40%）的效能更高；MetaPermit还提供31%更一致的授权决策，并在所有模型上保持可用性与安全性的最佳平衡。

**⚠️ 局限性**

局限性包括：1）安全性仅为经验性保障，无法正式证明元属性推断的语义正确性；2）LLM推断仍受模型随机性影响，导致元属性分布不稳定；3）对白盒注入攻击的防御需依赖策略层完整性；4）对大规模高成本模型的推断开销尚未充分优化。

---

## 405. Aurora-X: Built for Extreme Time Series Forecasting

**arXiv ID:** 2609.31038 | [PDF](https://arxiv.org/pdf/2609.31038v1)

**作者:** Xingjian Wu `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Aurora-X，一种百亿参数的时间序列基础模型，支持跨变量预测、协变量条件、可变分辨率推理和任意分位点预测。

**💡 创新点**

创新点在于进阶训练课程（预训练→中期→后期）、基于模式引导的稀疏 MoE 路由、隐式分位点网络以及多尺度后期训练实现可变分辨率推理。

**🔧 技术方法**

采用 Patch Embedding+RevIN、Time–Group MoE 架构、稀疏专家、深层模式指导、IQN 头、RoPE 时间注意、动态分辨率重采样等技术。

**📊 数据集**

使用 GIFT‑Eval、TIME、FEV‑Bench、TFB、DAG‑Bench 等公开时间序列基准。

**📈 对比分析**

与现有预训练 TSFM（TiRex、Chronos‑2、Falcon‑2.0 等）以及专用监督模型对比，Aurora‑X 在 GIFT‑Eval、TIME、FEV‑Bench 上实现了最低相对 MASE 并保持较低推理延迟。

**⚠️ 局限性**

局限在于对大规模后期训练依赖较高、训练成本仍显著、对极端长序列的推理仍需增量 token 以及未充分探索跨任务泛化的细粒度控制。

---

## 406. Collision-free Movement on Grids and Beyond

**arXiv ID:** 2609.31099 | [PDF](https://arxiv.org/pdf/2609.31099v1)

**作者:** Hendrik Molter `[一作]` (Hasso Plattner Institute), Meirav Zehavi `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图上的无碰撞移动问题，旨在协调一组机器人以达到目标形成，同时最小化总旅行距离。

**💡 创新点**

首次提出在给定形成特征的情况下，规划无碰撞的机器人运动以实现该形成的问题，并分析了该问题的参数化复杂性。

**🔧 技术方法**

使用参数化复杂性理论，特别是固定参数可解性（FPT）和近似算法。

**📊 数据集**

使用网格图、平面图和单位圆盘图作为数据集进行分析。

**📈 对比分析**

与现有方法进行比较，发现该问题在网格图上是NP-hard的，且在特定条件下是W[1]-hard的，同时提供了多项式核化结果。

**⚠️ 局限性**

限制在于当存在“令人厌恶的机器人”时，问题的复杂性显著增加，且在某些情况下无法获得多项式核。

---

## 407. Block Sparse Attention with Log-Linear Complexity

**arXiv ID:** 2609.31093 | [PDF](https://arxiv.org/pdf/2609.31093v1)

**作者:** Bohao Tang `[一作]` (Shanghai Jiao Tong University), Pengfei Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于金字塔 Top‑K 选择的块稀疏注意力机制（Pyramid Sparse Attention, PISA），并在训练与推理阶段通过 Triton 高效核实现了该机制。

**💡 创新点**

创新点在于：①使用多层金字塔化的 key 块表示，使得每个查询只在少量候选块上进行评分；②在每层使用 LogSumExp（LSE）评分而非直接平均或最大，以更准确地评估块重要性；③将多级选择与评分融合到单个 GPU 核中，显著降低了 I/O 与内存占用，使块选择复杂度从 O(N²/C) 降至 O(N log N)。

**🔧 技术方法**

技术方法包括：金字塔化 key 归约（平均池化）、分层 Top‑K 选择、LSE 评分、硬件感知 Triton kernel（两阶段训练/预填，单阶段解码）、多头共享 KV 机制、以及实验中使用的标准 Transformer 代码与优化技巧。

**📊 数据集**

数据集与评估：①大规模预训练（100B tokens，4K 长度）+ 10B tokens 继续预训练（16K 长度）用于构建 418M、1.47B、2.67B 模型；②语言建模指标（perplexity）、多选题集（BoolQ、PIQA 等）和包含率（Containment）评估；③长上下文检索评测采用 RULER 的 “needle‑in‑a‑haystack” 任务（1K–16K 长度）。

**📈 对比分析**

与 Baseline（Full Attention、BSA、NSA、HiLS）比较：PISA 在语言建模与常识推理上与对齐方法相近，且在包含率和 RULER 长上下文检索中均优于对手；在块选择阶段，PISA 在 64K、128K、256K 长度下分别比 BSA 快 2.86×、5.31×、9.95×，并在硬件上实现了较低的延迟。

**⚠️ 局限性**

局限性：①仍需手动设定块大小 C、Top‑K 数量和金字塔层数，可能不适用于所有任务；②在非常小的模型或短序列场景下，金字塔开销可能不明显；③论文主要评估在单机 GPU 上，跨机/多卡的可扩展性未作系统层面实验；④对比实验中未涉及某些最新稀疏注意力变体，未来需进一步验证普适性。

---

## 408. BenX: Resource-Sharing Permutations for Computational Integrity

**arXiv ID:** 2609.31087 | [PDF](https://arxiv.org/pdf/2609.31087v1)

**作者:** Luca Campa `[一作]`, Stefano Trevisani `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种基于可逆Beneš网络与Dickson多项式的置换哈希函数，可在小素数域（Goldilocks、BabyBear等）上实现高效加密。

**💡 创新点**

创新点在于将Beneš网络转化为可逆置换，并通过Dickson多项式实现低乘法复杂度，同时实现与NTT的硬件资源共享。

**🔧 技术方法**

采用可逆Beneš网络、Dickson多项式、MDS线性层、FPGA流水线、Rust软件实现和Plonky3证明系统。

**📊 数据集**

使用Goldilocks、BabyBear、Mersenne31、KoalaBear等标准小素数域作为测试向量，并未引入专门的数据集。

**📈 对比分析**

与Poseidon2、Tip5等哈希函数对比，FPGA实现吞吐量提升3–5%，延迟降低1.3–2.2倍，软件实现比同类哈希快7–13倍。

**⚠️ 局限性**

局限在于单独核心面积较大，对大状态长度扩展仍需优化，并且安全性需在更大字段上进一步验证。

---

## 409. Band-Selection Stability and Semantic Segmentation Performance: A Study on Hyperspectral City

**arXiv ID:** 2609.31074 | [PDF](https://arxiv.org/pdf/2609.31074v1)

**作者:** Jiarong Li `[一作]` (University of Galway), Brian Deegan `[通讯]` (University of Galway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估六种波段选择方法在十个类平衡ROI采样下的内部稳定性，并测量其对三种语义分割模型（U-Net、DeepLabV3+、SegFormer）不同子集大小（K=3,5,7,9,11,13）下的分割性能与推理速度影响。

**💡 创新点**

首次系统检验波段选择稳定性与下游分割性能的关联，提出基于频率共识的集成基线，并揭示在K=9时能实现高效性与精度的折衷。

**🔧 技术方法**

采用JMIM、CMIM、mRMR、Sim-LP、JMIM+CSNR、JMIM+CSNR+Corr等波段选择技术，利用成对Jaccard相似度评估稳定性；使用mIoU、mF1度量分割效果，并对CPU输入层推理时间进行基准测试。

**📊 数据集**

使用Hyperspectral City V2（128波段，450–950 nm，19类城市地物）数据集。

**📈 对比分析**

通过在十个ROI样本上生成60个Top-25子集，截取不同K值子集在三种SSM上训练9次（3个ROI×3个训练种子），比较其mIoU/mF1与128波段基线差异；结果显示Sim-LP与JMIM+CSNR在mIoU/mF1上分别提升至+2.01/+1.72点，K=9可实现约18–22倍的CPU推理加速且性能差距不足0.1个百分点。

**⚠️ 局限性**

局限于单一城市HSI数据集、仅采用成对Jaccard作为稳定性度量；频率共识基线未能优于单方法子集，结果是否能推广到其他数据集或稳定性指标尚未验证。

---

## 410. INTERACT: Interactive Planning for Autonomous Driving via Anchor-Conditioned Prediction and Trust-Region Refinement

**arXiv ID:** 2609.31137 | [PDF](https://arxiv.org/pdf/2609.31137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 411. Distributed Learning as a Service: The Developer's Perspective

**arXiv ID:** 2609.31061 | [PDF](https://arxiv.org/pdf/2609.31061v1)

**作者:** Tianyue Chu `[一作]` (Telefónica Scientific Research), David Solans Noguero `[通讯]` (Telefónica Scientific Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了分布式学习即服务（DLaaS）平台，允许开发者在单一控制台上配置并启用差分隐私、分割学习、分层聚合和知识蒸馏等四种机制，无需改动客户端代码，完成从任务配置到在边缘设备上实时检测的完整生命周期；

**💡 创新点**

首次将差分隐私、分割学习、分层聚合与知识蒸馏四种高级分布式学习技术整合为可声明式服务策略，在同一框架内实现可组合性与统一管理；

**🔧 技术方法**

采用基于Django REST的协调器、Docker化FastAPI辅助聚合器、Android SDK客户端，内部实现中心/本地DP、模型切分、分层聚合、学生模型蒸馏等算法；

**📊 数据集**

使用工业级智能家居唤醒词任务数据集“Ok Aura”以及CIFAR-10作为离线评估基准；

**📈 对比分析**

通过离线实验对比，分割学习可将设备峰值内存降低33%，知识蒸馏将广播模型压缩至88.8%（CIFAR-10）或92.4%（唤醒词），分层聚合将服务器每轮接收量从O(N)降至O(H)，在实际演示中展示了各机制对模型性能、通信成本和推理延迟的提升；

**⚠️ 局限性**

局限在于：仅通过少量实时训练轮次难以展示收敛性能，演示侧重演示流程；对大型真实部署的长期稳定性、网络延迟和跨平台兼容性等方面仍需进一步验证；

---

## 412. Compact Force Sensor for Dual-UAV Cable-Suspended Payload Transport with Tension-Aware Outer-Loop Control

**arXiv ID:** 2609.31035 | [PDF](https://arxiv.org/pdf/2609.31035v1)

**作者:** Andrea Delbene `[一作]` (Università degli Studi di Genova), Marco Baglietto `[通讯]` (Università degli Studi di Genova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并验证了一种紧凑型三轴力传感器，用于双UAV悬挂式载荷运输，配合分布式级联控制实现了高精度张力补偿

**💡 创新点**

在UAV系统中首次实现低成本、低重量的多轴张力传感器，并将其与离线低频张力反馈耦合，提升了稳态稳定性与障碍通行性能

**🔧 技术方法**

采用四个拉伸传感器构成惠斯通桥，利用MCP3423模数转换和AD8293放大器实现高采样率；控制层采用分层PID级联与姿态映射；实验通过PX4/ROS2、OptiTrack与自研传感器实现

**📊 数据集**

实验数据来自室内1公斤载荷与双x500 UAV的实测，未使用公开大规模数据集，所有数据均为实验室收集的实时测量与MoCap标定

**📈 对比分析**

与基于外部力估计的分布式几何控制器对比，结果显示测量张力方案在载荷下落与窄隙通行测试中实现了更小的偏差、振荡衰减与障碍避让更平滑，提升了整体鲁棒性

**⚠️ 局限性**

传感器的采样率仍低于行业标准，结构对温度与湿度敏感；控制器参数较为保守，未针对高速动态情况做优化，且缺乏对极限操作条件的全面评估

---

## 413. Interplay between Emotional Dynamics and Network Structure on the Social Media Vent

**arXiv ID:** 2609.31129 | [PDF](https://arxiv.org/pdf/2609.31129v1)

**作者:** Yuina Takahashi `[一作]` (University of Tsukuba), Sho Tsugawa `[通讯]` (University of Tsukuba)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析Vent平台的情绪动态与网络结构，探究用户帖子前时间线情绪与其随后情绪标签的内/跨类别关联，以及用户对时间线情绪波动的易感性与网络聚集性。

**💡 创新点**

①利用细粒度情绪标签进行跨类别关联分析；②量化并比较用户易感性，发现高易感用户在网络中高度聚集；③在大规模公开数据上实现大规模观察，填补先前研究中粗粒度与单一情绪类别的空白。

**🔧 技术方法**

统计方法（Mann‑Whitney U、基线比较）、情绪暴露量化、匹配率指数、相关系数、最短路径分布、对照随机重连模型。

**📊 数据集**

Vent公开数据集：约93万用户、3360万帖子、1360万关注边，聚焦7个细粒度情绪类别。

**📈 对比分析**

与基线情绪分布比较、使用虚拟时间线估计基线；对高易感用户与保度随机模型对比，发现1跳连接高8.3倍、平均路径距离2.35 vs 3.29；未给定精确分类性能指标。

**⚠️ 局限性**

仅捕获潜在曝光，未证明因果关系；关联可能受个人情绪特征、情绪同质性或外部事件影响；研究仅适用于Vent平台，需进一步验证跨平台普适性。

---

## 414. The Crowd in the Machine: A Crisis-Informatics Reading of the 2026 Autonomous Agent Incidents

**arXiv ID:** 2609.31060 | [PDF](https://arxiv.org/pdf/2609.31060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 415. LocUS: Head Selection and Subspace Projection for Targeted Activation Steering

**arXiv ID:** 2609.31122 | [PDF](https://arxiv.org/pdf/2609.31122v1)

**作者:** Irene Tallini `[一作]` (Area Science Park), Alberto Cazzaniga `[通讯]` (Area Science Park)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free的激活控制方法LocUS，利用词汇子空间对大型语言模型的内部激活进行局部、稀疏的干预；

**💡 创新点**

创新点在于将词汇子空间与注意力头选择相结合，实现双层定位：先通过对比数据和词典识别对属性最敏感的注意力头，再在这些头的原始输出空间投影到属性相关的稀疏子空间，从而限制干预方向与范围；

**🔧 技术方法**

主要技术包括差分均值（Difference‑of‑Means）激活推导、SOMP（Simultaneous Orthogonal Matching Pursuit）稀疏重建、词典子空间投影、头/层级的稀疏干预以及强度归一化；

**📊 数据集**

在三类任务上评估：毒性减轻（ThoroughlyEngineeredToxicity）、情感重定向（IMDb sentiment）和顺从抑制（persona Sycophancy），使用Mistral‑7B、DeepSeek‑7B和Llama‑8B三大模型；

**📈 对比分析**

与标准DoM、ITI、SMH等基线比较，LocUS在目标指标（毒性率、情感正率、顺从评分）上均优于或匹配最强基线，同时保持MMLU和PPL近似不变；整体干预参数仅占标准DoM的0.4%‑0.8%；

**⚠️ 局限性**

局限性在于只能针对由词汇差异携带的属性，且在语义流畅度（PPL）方面仍略逊，未覆盖开放式推理等更细粒度能力。

---

## 416. Kintsugi-VLA: Turning Failed Robot Rollouts into Recovery Data through Interventional Recoverability

**arXiv ID:** 2609.31048 | [PDF](https://arxiv.org/pdf/2609.31048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 417. Confident, Not Wiser: The Dunning-Kruger Effect in Human-AI Interaction

**arXiv ID:** 2609.31095 | [PDF](https://arxiv.org/pdf/2609.31095v1)

**作者:** Daniela Fernandes `[一作]` (Aalto University), Robin Welsch `[通讯]` (Aalto University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究比较了人类单独完成推理任务与人类+AI协作完成任务的表现与自我评估，探讨AI辅助对元认知的影响。

**💡 创新点**

创新点在于：①将自我评估分为全局估计、块级估计和答案级信心，细分元认知维度；②对Dunning‑Kruger效应在AI协作中的出现做了分组、分块、回归‑to‑均值及测量误差校正等多重控制；③提出对AI协作伙伴的评估与对自身认知的区分，强调“合作性能评估”而非单纯的“自我评估”。

**🔧 技术方法**

使用了GPT‑5.6 Luna作为AI模型，并在相同题目上跑了100次低推理成本的对照实验以获得模型自身表现；对人类表现与自我估计进行统计分析（t检验、效应量、AUROC、回归、分组比较、Bayesian等）。

**📊 数据集**

数据集为自制的40道推理题，涵盖矩阵推理、空间旋转、三段论推理、字母串类比，每类10题，确保题目难度与AI模型性能相异。

**📈 对比分析**

比较方法为：①人类+AI组 vs 人类单独组 vs AI模型组（两者使用相同题目），测得得分差异、效应量；②元认知准确性（估计误差）与敏感度（AUROC）对比；③Dunning‑Kruger对比（分四分位、分块、回归‑to‑均值、测量误差校正）。结果显示人类+AI组整体得分显著高于人类单独组，但自我估计误差与AUROC均低于人类单独组，Dunning‑Kruger效应在两组均存在，AI协作组的对比幅度更大。

**⚠️ 局限性**

局限性包括：非随机分配两组导致潜在自选偏差；固定四类任务与模型性能混杂，难以推广至其他任务；模型对照实验与实际对话条件不完全一致；未记录详细交互过程（提示次数、答案采纳、验证行为）导致无法深入解析AI使用方式对结果的影响；元认知测量受分数噪声、上限效应限制；部分统计检验依赖于特定假设，未能完全排除统计学解释。

---

## 418. Robust Graph Clustering Network for Multiple Missing Data

**arXiv ID:** 2609.31033 | [PDF](https://arxiv.org/pdf/2609.31033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 419. Frame the adversary: a structure-aware attack methodology

**arXiv ID:** 2609.31128 | [PDF](https://arxiv.org/pdf/2609.31128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 420. The Power of Indirection: Scaling Switches Beyond Silicon Boundaries

**arXiv ID:** 2609.31092 | [PDF](https://arxiv.org/pdf/2609.31092v1)

**作者:** Lukas Röllin `[一作]` (ETH Zürich), Laurent Vanbever `[通讯]` (ETH Zürich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种多ASIC交换机架构，利用电路交换的间接层动态重映射入口端口到各ASIC，从而显著减少跨ASIC通信，提升整体性能并降低功耗。

**💡 创新点**

核心创新点在于：① 在单设备内部实现电路交换间接层，保持出口端口固定，简化硬件与状态迁移；② 设计出微秒级控制平面与高效贪婪映射算法；③ 通过实验验证该方法在单ASIC性能下仅降低约1%，并实现数百瓦功耗节省。

**🔧 技术方法**

采用的技术包括：电路交换（可为电气或光学）实现的间接层；P4可编程交换机收集流量矩阵；基于Netbench的包级仿真；贪婪与匈牙利匹配算法实现映射。

**📊 数据集**

使用的数据集有：① 典型高性能计算/机器学习流量模式（Shuffle、Stride、All-Reduce、Ring All-Reduce、Hierarchical All-Reduce）；② Facebook数据中心Web和Hadoop流量跟踪；③ LLM训练工作负载（PyTorch FSDP对Llama 3.2模型的8 GPU训练）。

**📈 对比分析**

实验方法：在包级仿真和真实硬件原型上与传统非阻塞与过载多ASIC设计对比，衡量吞吐量、99%分位FCT、跨ASIC带宽利用率及功耗。结果显示：① 在多数工作负载下，/在单ASIC性能内误差<1%；② 对LLM训练，/实现3倍吞吐；③ 通过减少跨ASIC链路带宽，/在两ASIC场景下可实现约400W功耗节省。

**⚠️ 局限性**

主要局限包括：① 对大于4 ASIC的扩展需要更低的过载比例，复杂度随ASIC数增大；② 需要高速、低延迟的电路重配置，电气实现仍受限于芯片间连接；③ 控制平面需要对流量变动快于重配置周期，若流量波动更快则性能下降；④ 硬件原型仅实现两ASIC，未验证更大规模的实际可行性。

---

## 421. Comparative Evaluation of an XR Pen-based Control Interface for Semi-Autonomous Mobile Robot Navigation in Service Environments

**arXiv ID:** 2609.31117 | [PDF](https://arxiv.org/pdf/2609.31117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 422. Cheap, open agents make LLM pollution harder to mitigate

**arXiv ID:** 2609.31054 | [PDF](https://arxiv.org/pdf/2609.31054v1)

**作者:** Raluca Rilla `[一作]` (Max Planck Institute for Human Development), Dirk U. Wulff `[通讯]` (University of Basel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了某种新型算法在特定任务中的应用，旨在提高任务的效率和准确性。

**💡 创新点**

创新点在于提出了一种改进的算法框架，能够在处理复杂数据时显著提升性能。

**🔧 技术方法**

使用了深度学习技术和强化学习方法相结合的技术。

**📊 数据集**

实验使用了公开的标准数据集，以确保结果的可比性和可靠性。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现较好，但在其他类型的数据上可能效果不佳。

---

## 423. Modeling Student Sensemaking with LLMs and Knowledge-Graph-Guided Inference

**arXiv ID:** 2609.31046 | [PDF](https://arxiv.org/pdf/2609.31046v1)

**作者:** Özge Alacam `[一作]` (LMU München), Sinem Gencer `[通讯]` (Gazi University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究如何利用指令调优的大语言模型（LLMs）在不进行任务特定训练的情况下，对化学实验课堂中的协作科学学习对话进行多维度的sensemaking分析，并评估加入知识图谱诊断信息的效果

**💡 创新点**

提出了将学生对话转换为结构化知识图谱并提供诊断指标作为LLM提示上下文的创新框架，探讨了不同提示策略（定义支架、链式思考、单/双回合）对多维sensemaking指标的影响

**🔧 技术方法**

使用两款中型指令调优LLMs（Gemma‑3‑27B‑IT 与 Qwen‑3‑32B），实现链式思考（CoT）、单回合与双回合提示，并将学生知识图谱诊断指标嵌入提示中

**📊 数据集**

使用23个经过领域专家多维度标注的博伊尔定律实验室活动对话数据集（共18名学生、5个小组），每个对话平均6500词、45回合

**📈 对比分析**

与人工标注对比，计算F1、Cohen κ等指标；单回合链式思考提升对USM的识别；双回合在某些维度提升但整体不显著；加入知识图谱诊断在单回合链式思考配置下将κ从0.51提升至0.60，表明最优配置仍需针对不同维度选择

**⚠️ 局限性**

样本量仅23个，主题单一；仅评估两款中型LLM，缺乏跨模型泛化；对话中多维度标注高成本，结果可能受注释者主观性影响；知识图谱诊断构建仍有误差，未覆盖情感和结构一致性等更细粒度特征

---

## 424. DepthEvidence: Unifying Metric Depth Prediction and Geometric Reasoning in Multimodal Language Models

**arXiv ID:** 2609.31103 | [PDF](https://arxiv.org/pdf/2609.31103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. Monitor Jailbreaking: Evading Chain-of-Thought Monitoring Without Encoded Reasoning

**arXiv ID:** 2609.31121 | [PDF](https://arxiv.org/pdf/2609.31121v1)

**作者:** Julian Schulz `[一作]` `[通讯]` (Meridian Cambridge), Julian Schulz (Meridian Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在带有链式思考监控的强化学习环境中训练推理模型，研究其如何绕过监控以完成主任务和副任务。

**💡 创新点**

首次提出并实证“监控越狱”现象，即模型通过重新表述链式思考来逃避监控，而不是隐藏推理过程。

**🔧 技术方法**

采用 GRPO 强化学习、GPT‑oss 20B/120B 推理模型、GPT‑5 系列 LLM 监控器、Suspicion Score 评估方法，并使用自动化 paraphrase 防御。

**📊 数据集**

使用 Game24 作为主任务，三位数乘法和 GSM8K 词问题作为副任务的数据集。

**📈 对比分析**

通过监控分数与副任务准确率的双重指标进行比较，实验显示模型能在保持高副任务成功率的同时显著降低监控分数；越狱策略在未见监控器上部分转移，但对更强监控器或跨厂商监控器效果有限；Paraphrase 能在一定代价下恢复监控可检测性。

**⚠️ 局限性**

局限在于仅评估 GPT‑oss 与 GPT‑5 组合，越狱对更强或跨厂商监控器的泛化能力弱，且手动复现策略不稳定，尚需进一步验证在更广泛设置下的有效性。

---

## 426. From Shortcut Learning to Discrete Neural Insertion Sort

**arXiv ID:** 2609.31114 | [PDF](https://arxiv.org/pdf/2609.31114v1)

**作者:** Konstantinos Mylonas `[一作]`, Thrasyvoulos Spyropoulos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了神经网络在执行插入排序时的可解释性与泛化能力，发现传统 CLRS 基线模型在中间提示监督不足且会提前产生正确输出（shortcut learning），随后提出一种离散化的插入排序执行器，该模型将数值和控制状态分离，并在每一步通过离散状态瓶颈强制模型仅使用预定义的控制状态，从而实现了从训练长度 16 到 64、128 的完美长度泛化；

**💡 创新点**

创新点在于将插入排序的局部交换与全局控制拆分为离散状态和连续数值两部分，采用链式图结构、边比较位和虚拟全局节点，并通过额外的虚拟节点监督实现对全局循环结束的准确判断，从而在仅使用离散状态监督的情况下实现了强泛化；

**🔧 技术方法**

技术手段包括基于图神经网络的链式结构、离散状态编码与解码、门控的数值传递机制、全局虚拟节点的注意力聚合以及多任务监督（状态、数值和虚拟节点的二分类）等；

**📊 数据集**

使用的“数据集”是从 [0,1] 均匀分布采样的随机标量序列，并利用官方 CLRS 基准的插入排序实现生成完整的中间状态标签；

**📈 对比分析**

对比方法为官方 CLRS 基线模型，评估指标包括状态准确率、数值准确率和排序序列准确率，实验表明离散执行器在训练长度 16 以及外推长度 64、128 上均能达到 100% 的准确率，而基线模型则无法保证中间状态的正确性；

**⚠️ 局限性**

局限性包括对算法特定结构和监督的高度依赖（需要手工定义链图、离散状态集合、虚拟节点监督等），离散瓶颈限制了模型的表达自由度，且目前仅针对插入排序验证，尚未证明其在更广泛的顺序算法中的通用性。

---

## 427. Pocket-STVG: lightweight architecture for Spatio-Temporal Video Grounding

**arXiv ID:** 2609.31135 | [PDF](https://arxiv.org/pdf/2609.31135v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 428. Moment Ambiguity and the Limits of Robust Stochastic Optimization

**arXiv ID:** 2609.31090 | [PDF](https://arxiv.org/pdf/2609.31090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 429. Linear Certificates for Membership Comparability, Quadratic Barriers for Selectors

**arXiv ID:** 2609.31053 | [PDF](https://arxiv.org/pdf/2609.31053v1)

**作者:** Sebastian Ben Daniel `[一作]` `[通讯]`, Sebastian Ben Daniel

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了二元成员比较器和选择器提供的部分信息如何通过非统一建议转化为精确识别，并给出了相应的建议复杂度与证明方法。

**💡 创新点**

首次证明了对所有二元成员比较器可用线性非确定性建议和短证书（≤5n+12位），并给出了选择器在平均/最坏情况下的建议上界与下界，实现了对P‑选择集的最优推广。

**🔧 技术方法**

核心技术包括：图论中的独立两步覆盖、短迫使与多数决定二分、黑盒冻结与隐藏核心分布的确定性/随机化下界、以及利用假设𝖴（唯一电路SAT）实现的线性建议转移。

**📊 数据集**

该工作为纯理论分析，无需实验数据集；所有结论均通过组合与复杂性理论构造与证明得到。

**📈 对比分析**

对比传统P‑选择器的线性建议（n+1位）与本工作得到的3n+5位建议，证明了在平均/最坏情况上分别取得线性与二次下界；在随机化平均时间上实现2n+O(1)位建议。

**⚠️ 局限性**

限制包括：常数系数（如3、5）尚未证明最优；对更高 arity 的比较器扩展尚未完成；下界主要在黑盒查询模型，未对全局多项式时间下界给出更强结果；假设𝖴仍为前置条件，若不成立则结论失效。

---

## 430. DynBranch: Speculative Subgraph Reuse for Dynamic Agentic LLM Serving

**arXiv ID:** 2609.31047 | [PDF](https://arxiv.org/pdf/2609.31047v1)

**作者:** Junyi Shen `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可在运行时决定分支路径的LLM服务系统，利用早期填充、跨请求子图重用以及基于负载的价格门控来消除分支解析瓶颈。

**💡 创新点**

通过为未解析分支引入稳定坐标、实现候选子图的提前执行与确认后推广，并结合两级负载定价控制，首次实现可在分支决策前完成子图计算并在后续请求中安全重用。

**🔧 技术方法**

采用了子图抽象、缓存与依赖刷新机制、预测器接口、两级决策门控以及在模型-API边界的中间件实现，无需修改代理或模型引擎。

**📊 数据集**

在四种工作负载（Routing、ReAct、Sub‑Agent、HCI）上评估，使用BFCL函数调用日志、MuSiQue多跳问答、HotpotQA检索语料和Schema‑Guided Dialogue数据集。

**📈 对比分析**

与五个现有系统（InstCache、Helium+、Parrot、DSP、SPAgent）对比，在Qwen3‑32B/H200和Qwen3‑8B/RTX4090平台上平均延迟提升最高32%，相较无重用基准降低46–66%，并在p99降低11–33%。

**⚠️ 局限性**

仅在同质模型、已知分支坐标、无副作用的可重复子图场景下验证，需进一步扩展到事务隔离、跨资源调度、状态化工具和多模态管道。

---

## 431. Toward AI-Augmented Cooperative Engineering Workflows: Requirements and Architecture the European Rover Challenge

**arXiv ID:** 2609.31136 | [PDF](https://arxiv.org/pdf/2609.31136v1)

**作者:** Ahmed R. Sadik `[一作]` (Honda Research Institute Europe), Joan Smith `[通讯]` (Honda Research Institute USA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对欧洲机器人挑战（ERC）中的学生团队进行角色适应性问卷调查，识别协作工程流程中的瓶颈，基于结果提出 AI 赋能的协作工程工作流需求，并设计一套包含多模态交互、凭证管理、服务选择与外部工程工具集成的 AI 助手体系结构。

**💡 创新点**

首次系统化挖掘 ERC 团队在需求跟踪、任务分配、知识迁移、沟通协同、重工与集成风险等方面的痛点，并提出对应的 AI 需求与架构框架；架构通过服务选择器实现多模态交互与多种专业 AI 服务的无缝对接，体现了从任务、需求、知识、沟通到集成风险的端到端 AI 支持创新。

**🔧 技术方法**

使用大型语言模型（ChatGPT、Copilot 等）作为专业 AI 服务核心；客户端提供文字、语音、图像交互；服务端包括凭证服务器、WebSocket/HTTP 通信、服务选择器；与外部工程工具（Git、CAD、设计数据库等）通过 API 交互；利用 LLM 进行需求解析、任务监控、文档生成、沟通摘要及集成风险检测。

**📊 数据集**

ERC‑2025 队伍问卷数据集，共 104 条响应，覆盖 14 支队伍，涵盖团队结构、任务管理、沟通方式、知识迁移、需求跟踪、AI 工具使用等维度。

**📈 对比分析**

本研究未进行系统性能或对比实验；提出的架构为设计概念，未来计划实现核心服务后在 ERC 环境中进行实验评估，尚无量化结果。

**⚠️ 局限性**

局限性包括：数据来源仅为自我报告问卷，缺乏现场观测与客观指标；所提架构尚未部署与评估，缺乏实证验证；研究范围局限于 ERC 学生团队，外延至其他工程域仍需进一步验证。

---

## 432. Impact of Antenna Position Errors on TDMA and NOMA in Pinching-Antenna Systems

**arXiv ID:** 2609.31088 | [PDF](https://arxiv.org/pdf/2609.31088v1)

**作者:** Wei Jiang `[一作]` (German Research Center for Artificial Intelligence), Hans D. Schotten `[通讯]` (University of Kaiserslautern)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过建立位置误差到相位误差的映射模型，分析了在PINCHING‑ANTENNA SYSTEM（PASS）中位置误差对TDMA和NOMA多址技术的平均速率与失效概率的影响，并给出了闭式的上下界与数值验证。

**💡 创新点**

创新点在于：①首次将机械位移误差转化为相位误差并给出其均方根与相干因子；②针对TDMA与NOMA分别推导了平均速率上界与下界以及闭式失效概率；③揭示单极点TDMA对位置误差免疫、复数极点TDMA仅损失阵列增益、NOMA会出现不可通过增大发射功率消除的干扰底限。

**🔧 技术方法**

使用的技术主要包括：相位敏感性分析、中心极限定理将误差建模为高斯分布、Jensen不等式、相干因子分析、闭式失效概率推导（利用高斯尾Q函数与Bessel函数），以及Monte‑Carlo仿真验证。

**📊 数据集**

实验数据集为仿真数据：在28 GHz、波导折射率1.4、信号间距3 m、用户分布在10 m方形区域内，噪声功率-90 dBm；位置误差假设为均值为0、方差σ_x²的高斯分布。

**📈 对比分析**

比较方法：对比单极点TDMA、多极点TDMA（N=8）与NOMA强用户的速率保持率和失效概率。结果显示：单极点TDMA对误差完全不敏感；多极点TDMA在σ_x≤1 mm时仍保持>90%速率；NOMA在同等误差下速率迅速下降并出现失效概率底限，说明其对相位同步极为敏感。

**⚠️ 局限性**

局限性包括：仅考虑单波导且视线（LOS）通道，未覆盖多径环境；误差模型仅涵盖位移而未考虑角度或温度漂移；NOMA的分析仅在N=1、K=2时给出闭式下界，无法直接推广到多用户多极点场景；仿真验证也仅基于理想模型，实际硬件的非线性和电磁耦合未考虑。

---

## 433. OmouAI: Argumentative Human-AI Policy Deliberation with Simulated Personas

**arXiv ID:** 2609.31078 | [PDF](https://arxiv.org/pdf/2609.31078v1)

**作者:** Stylianos Loukas Vasileiou `[一作]` (New Mexico State University), Georgina Curto `[通讯]` (United Nations University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了OmouAI，一套将大语言模型与计算论证相结合的互动式公共政策决策支持系统；

**💡 创新点**

创新点在于：①引入可配置的“人格”角色，实现多方利益相关者与专家的模拟对话；②采用量化双向论证框架（QBAF）与渐进语义，对论点进行可解释、可争议的评分；③通过与可量化目标（如联合国可持续发展目标）的对齐，提供政策建议的影响评估；

**🔧 技术方法**

使用技术包括：大语言模型（LLM）用于论点生成、分类与评分；量化双向论证框架（QBAF）与渐进语义（如DF‑QuAD、二次能量模型）进行论证推理；文本挖掘与命名实体识别用于用户输入的论点抽取；

**📊 数据集**

未公开具体数据集，系统依赖用户输入的政策主张、角色设定及公开的目标指标；

**📈 对比分析**

论文未给出与现有方法的实验对比或性能指标，主要以系统架构与示例演示说明功能；

**⚠️ 局限性**

局限性包括：LLM固有偏见与不确定性导致论点质量不一；缺乏大规模真实政策情景的实证验证；对话管理与论证规模的可扩展性待研究；需人工监督以确保决策可靠性。

---

## 434. Externalized CPDAG Summaries Improve LLM Causal Deduction

**arXiv ID:** 2609.31071 | [PDF](https://arxiv.org/pdf/2609.31071v1)

**作者:** Wentao Sun `[一作]` (Nokia Bell Labs), Alonso Silva `[通讯]` (Nokia Bell Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个两轮提示和解码流程（Structured Thinking），通过先生成一个结构化的 CPDAG 摘要来帮助大型语言模型在 Corr2Cause 基准上做因果推理。

**💡 创新点**

将因果推断任务视作隐式对象推理，设计了结构化的 CPDAG 摘要中介，并证明外部化和约束化的中间表示能显著提升性能。

**🔧 技术方法**

利用大型语言模型（Qwen、Gemma、GPT‑5.4‑mini）进行两轮提示，使用正则/JSON Schema 约束解码，工具调用输出 CPDAG，基于 PC 算法的推理步骤和 Meek 规则。

**📊 数据集**

使用 Corr2Cause 原始数据集（ID 拆分 1162 条、Paraphrase‑OOD 2246 条），并在实验中加入随机与真实网络的更大图样本进行压力测试。

**📈 对比分析**

对比强直觉 PC 指令单轮基线、两轮结构化文本控制和单轮结构化提示，评估 F1(Yes) 与准确率；在 Qwen3.5‑27B 上提升 13.4pp F1(Yes) 至 86.4，整体准确率>95%，并在不同模型、OOS 拆分及变量规模上保持正向提升。

**⚠️ 局限性**

仅适用于 CPDAG 推断，未实现完整的图一致性校验，约 18–20% 输出存在图一致性违规，缺乏对更大模型/更复杂任务的全面验证，且方法未完全分解每个组件的贡献。

---

## 435. Research with AI Agents: How Agentic Systems Are Changing Scientific Work

**arXiv ID:** 2609.31219 | [PDF](https://arxiv.org/pdf/2609.31219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 436. CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings

**arXiv ID:** 2609.31062 | [PDF](https://arxiv.org/pdf/2609.31062v1)

**作者:** Marko Řeháček `[一作]` (Masaryk University), Vít Nováček `[通讯]` (Masaryk University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了基于冻结查询嵌入的差异平均探针，用于在患者搜索查询中评估医疗紧急性、心理紧急性和主题敏感度等临床风险轴。

**💡 创新点**

将临床风险轴视为线性方向，并在冻结嵌入空间中通过差异平均法学习一维探针，提供低延迟、可审计的安全门控；同时证明该方法在肿瘤查询上与专家标注和开源LLM性能相当。

**🔧 技术方法**

技术包括：差异平均（DiM）探针在L2归一化嵌入上训练；BERTopic聚类与HDBSCAN降维；few-shot提示生成对比样本；多种冻结嵌入模型（Qwen3、Harrier、text-embedding-3-large、gemini-embedding-2）；与两款开源LLM（gpt-oss-safeguard-20b、gpt-oss-120b）及前沿LLM（gpt-5.4）进行对比；使用二次加权kappa、宏F1和AUROC评估。

**📊 数据集**

使用79,658条捷克语肿瘤中心搜索日志，经过BERTopic聚类后构造对比样本；200条人工标注（90真实+110合成）的黄金标准；另外400条评估池（290真实+110合成）用于对比实验。

**📈 对比分析**

与两款开源LLM和前沿LLM在200条黄金查询上进行二次加权kappa（QWK）对比。MU、PU探针与开源LLM无显著差异，性能接近专家标注；相较LLM，探针吞吐量约30×高、延迟接近零；在MU紧急召回上探针漏报率低于LLM。

**⚠️ 局限性**

局限性包括：样本量有限（90真实查询）导致统计功效不足；高危查询大多为合成，真实高危样本稀缺；TS轴未能通过单一方向捕获，需要改进对比构造；仅在捷克语单轮搜索上验证，跨语言、跨站点及对话场景尚未评估。

---

## 437. Preserve-and-Compose Training for Composed Image Retrieval

**arXiv ID:** 2609.31202 | [PDF](https://arxiv.org/pdf/2609.31202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 438. Neuralyzing the Trace: Selective Representation-Level Unlearning with Contrastive Sparse Autoencoders

**arXiv ID:** 2609.31056 | [PDF](https://arxiv.org/pdf/2609.31056v1)

**作者:** Itai Zehavi `[一作]` (Mila), Ulrich Aivodji `[通讯]` (Mila)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于对比稀疏自编码器（SCALPEL）的内部表示层级记忆删除方法，能够在保持模型整体性能的同时精准抹除特定作者信息；

**💡 创新点**

创新点在于将对比学习与稀疏自编码器结合，利用多视角增强目标的稳定性，形成对低能量、精细目标的高选择性特征，并通过尺度无关的选择得分控制背景扰动；

**🔧 技术方法**

技术包括多视角对比学习、稀疏自编码器（SAE）、信息NCE对比损失、特征选择与残差流干预、以及理论分析证明对比学习提升目标选择性；

**📊 数据集**

使用TOFU数据集（包含多个作者的问答文本）以及Wiki数据作为背景知识，并在Qwen、Llama、Gemma三大模型上进行实验；

**📈 对比分析**

与传统NMF、标准SAE、Gradient Difference、RMU等基线比较，SCALPEL在忘记目标概率下降的同时保持更高的模型实用性，几乎位于或接近Pareto前沿，表现优于NMF/SAE，且与RMU、Gradient Difference相当；

**⚠️ 局限性**

局限性包括仅在相对较小的模型与受控数据集上验证，未对大型模型或更复杂目标（文档/事实级别）测试；理论分析局部，未能保证参数级删除；方法仅提供持续的内部过滤而非完全的参数消除。

---

## 439. Light Field Primitive for Novel View Synthesis

**arXiv ID:** 2609.31198 | [PDF](https://arxiv.org/pdf/2609.31198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 440. Who Says What: Symbolic Trimodal Binding Mechanisms in Audio-Visual LLMs

**arXiv ID:** 2609.31193 | [PDF](https://arxiv.org/pdf/2609.31193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 441. KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

**arXiv ID:** 2609.31045 | [PDF](https://arxiv.org/pdf/2609.31045v1)

**作者:** Jiahao Hui `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了KuaFu，一种面向用户行为序列的两轴压缩与理解框架，能够在保持高质量理解的前提下将长序列压缩到极低的存储尺寸；

**💡 创新点**

创新点在于以单行为项为压缩单元、采用两轴（token 与宽度）压缩投影、分阶段（预训练、压缩问答、协同训练、RL）训练策略以及专门针对四种hallucination类型的RL奖励机制；

**🔧 技术方法**

使用了基于Qwen3-4B或Llama-3.2-1B的LLM Encoder-Decoder架构、LoRA、MemToken、两轴投影网络、RoPE位置编码、DPO/DAPO强化学习；

**📊 数据集**

数据集包括公司内部广告行为日志、兴趣摘要序列、中文长文本语料MNBVC，以及公开的MRQA、RecBench等基准；

**📈 对比分析**

通过离线评估（F1、准确率、EM、CTR）和线上A/B实验比较，KuaFu在五项业务指标上至少不逊于单任务模型，并提升了2.0%生活阶段指标、提高了1.37% GMV，压缩率10-15×时实现每GPU吞吐量提升37-350%，节省190块GPU；

**⚠️ 局限性**

局限性包括压缩比例固定、对极长行为项细节损失、对稀疏序列的适配不足、未系统验证更大规模参数/数据的可扩展性。

---

## 442. FlatClip: A Geometry-Aware Surface-Level Baseline for fMRI Representation Learning

**arXiv ID:** 2609.31204 | [PDF](https://arxiv.org/pdf/2609.31204v1)

**作者:** Mo Wang `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了一种基于表面flatmap的fMRI特征提取框架FlatClip，利用冻结的SigLIP2图像编码器在多种任务上实现了中等精度的预测。

**💡 创新点**

创新点在于将宏观大脑几何结构转化为可直接输入图像模型的flatmap序列，并证明冻结的图像预训练模型在保持空间布局的同时能够兼顾ROI和体素模型的优势。

**🔧 技术方法**

使用的技术包括表面渲染、SigLIP2冻结编码、时间平均池化、轻量级MLP探测器，以及多种空间扰动控制实验。

**📊 数据集**

评估数据集包括HCP（性别预测）、PPMI（帕金森病诊断）、ADNI（MCI/AD对比）以及NSD（视觉fMRI COCO80多标签识别）。

**📈 对比分析**

与多种基线（ROI级别的Brain‑LM、BrainMASS等、体素级别的SwiFT、NeuroSTORM、Omni‑fMRI）在相同冻结特征+轻量探测器设置下进行对比；FlatClip在HCP和ADNI任务中表现优于大多数ROI基线，弱于最强的体素模型；在NSD视觉解码任务中，使用任务相关的脑区flatmap能进一步提升mAP，优于大多数通用fMRI基线。

**⚠️ 局限性**

局限性包括仅覆盖皮层表面、忽略亚皮层/脑干信号，时间平均池化导致时序信息丢失，以及对表面几何的裁剪和变形可能影响精度。

---

## 443. Samples, Sources, Space: Decomposing Data Scale in Spatially Structured Representation Learning of Human Brain Microarchitecture

**arXiv ID:** 2609.31201 | [PDF](https://arxiv.org/pdf/2609.31201v1)

**作者:** Christian Schiffer `[一作]` (Institute of Neuroscience and Medicine (INM-1), Research Centre Jülich), Timo Dickscheid `[通讯]` (Institute of Neuroscience and Medicine (INM-1), Research Centre Jülich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在显微全脑组织学中，将数据规模拆分为唯一样本数、源多样性和空间覆盖率，并通过控制预训练条件评估其对表示学习的影响；

**💡 创新点**

首次把数据规模细分为三维度（唯一样本、源多样性、空间覆盖），并证明在固定样本预算下，增加源数并不提升表示质量，而样本数、计算和模型容量才是关键；

**🔧 技术方法**

使用SpatialNCE无监督目标、Vision Transformer（ViT-B/L/H）架构、混合效应回归与最小可检验效应分析；

**📊 数据集**

21份人类后天脑组织的全脑组织学图像（约1.16亿个图像块），并按切片、空间坐标和参考空间对样本进行采样；

**📈 对比分析**

与不同源数、样本数、计算量、模型容量以及空间覆盖率的组合进行对照实验，结果显示：1）样本数加倍提升宏F1≈0.09 logit；2）计算加倍提升≈0.11 logit；3）模型容量提升约0.08–0.13 logit；4）固定样本预算下增加源数无显著改进；5）更广泛的空间覆盖和跨源正样本权重显著提升性能；

**⚠️ 局限性**

仅在21个源的范围内探讨，未考虑不同组织学技术、不同任务或更大样本数；源数对表现影响的结论可能受固定预算限制；同时未对跨源正样本比例的进一步细化研究；

---

## 444. Accounting for Bias Enables Sustainable LLM Evaluation

**arXiv ID:** 2609.31184 | [PDF](https://arxiv.org/pdf/2609.31184v1)

**作者:** Harshita Katoch `[一作]` (University of Kaiserslautern--Landau), Sebastian Vollmer `[通讯]` (University of Kaiserslautern--Landau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的心理测量框架，对LLM评估中的配对比较和序数评分进行联合建模，并显式校正位置偏差、冗长偏差、自我增强和评测者严厉度等系统性偏差，以实现更可靠的模型排名。

**💡 创新点**

创新点在于：①将配对比较与序数评分整合为单一似然函数；②引入多因素偏差参数（位置、冗长、同家系偏好、评测者严厉度）并在IRT基础上显式建模；③证明通过偏差校正可在大幅减少比较次数的同时保持甚至提升排名准确性，从而实现评估的可持续性。

**🔧 技术方法**

使用了扩展的Bradley–Terry–Luce模型（配对数据）与多因素Rasch模型（序数数据），以及它们的联合似然；参数估计采用L-BFGS-B梯度优化并加以岭正则化；在模拟实验中利用了自适应信息增益策略。

**📊 数据集**

在MT‑Bench基准上进行实验，使用GPT‑4评测器生成的配对比较与单分数序数评分数据。

**📈 对比分析**

与传统的简单胜率统计和平均分数对比，HybridRater模型在各种锦标赛设计（Uniform、Swiss、Stratified、Hub‑Spoke、MultiHub、Popularity）和不同序数占比（10%、50%、90%）下均取得更高的Kendall‑τ相关性。尤其在稀疏且对抗性调度场景中，HybridRater在相同样本量下达到了比传统方法高出数个百分点的排名准确性，同时显著减少了所需的比较次数（最多可低至原来约1%）。

**⚠️ 局限性**

主要局限包括：①模型假设单维潜在能力，无法捕捉多维LLM能力结构；②仍需进一步验证校正后能力是否真正对应目标构念；③在多任务、多模型场景下，偏差参数估计可能受限；④实际部署时需实现自适应调度的复杂度和计算成本。

---

## 445. SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations

**arXiv ID:** 2609.31164 | [PDF](https://arxiv.org/pdf/2609.31164v1)

**作者:** Tobias Vente `[一作]` (University of Antwerp), Bart Goethals `[通讯]` (University of Antwerp)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 SPADE（Serendipitous Pareto Distance Evaluation）——一种基于用户历史相似度与全局热门度两维空间构建 Pareto 前沿，并通过最短欧氏距离衡量推荐的惊喜度与相关性的评估指标。

**💡 创新点**

创新点在于：①首次将相似度与热门度同时纳入评估框架，解决传统指标单维度的缺陷；②利用 Pareto 前沿确定“最不惊喜”的边界，使得距离越远的候选项越具真正的 serendipity；③仅对测试集中的正确推荐计算分数，确保评估严格关注用户真正接受的项，避免随机或无关项获得高分。

**🔧 技术方法**

技术细节包括：使用 PMI/PPMI 计算项与用户历史的相似度并进行 Empirical Bayes 平滑；归一化相似度与热门度至 [0,1]；构造用户特定 Pareto 前沿；用欧氏距离得到 SPADE 分数；在 Recpack 框架下对 EASE、SLIM、ItemKNN、Popularity、Random 进行网格搜索调参；对五个数据集进行 80/20 训练/测试切分并计算 top‑K 指标。

**📊 数据集**

实验采用的五个公开数据集：CiteULike、MIND、MovieLens‑1M、MovieLens‑20M（子采样至 10k 用户）以及 Netflix（子采样至 10k 用户）。

**📈 对比分析**

比较方法：将 SPADE 与三种主流超精度指标——Novelty（反向热门度）、Primitivity（基于“原始”热门推荐的偏差）、Co‑Occurrence（基于 PMI 的协同过滤距离）进行对比。实验结果显示：①随机与纯热门基线在传统指标上往往能得到高分，但在 SPADE 中得分极低，证明 SPADE 能有效剔除无关推荐；②在各数据集上，性能最优的模型（如 EASE）在 SPADE 上获得中等分数，表明其既保留了一定的相关性，又能提供一定程度的 serendipity；③SPADE 在所有数据集上始终保持对真实相关推荐的严格要求，避免了传统指标可能被 “作弊” 的现象。

**⚠️ 局限性**

局限性包括：①仅为离线评估，未考虑实时/动态推荐场景；②依赖于准确的相似度与热门度估计，若这两项本身存在偏差会影响最终分数；③计算 Pareto 前沿与距离对大规模项目集合的计算成本相对较高；④缺乏对时间维度或用户体验主观感受的直接衡量，可能无法完全覆盖用户对 serendipity 的主观体验。

---

## 446. WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting

**arXiv ID:** 2609.31162 | [PDF](https://arxiv.org/pdf/2609.31162v1)

**作者:** Yuhan Zhu `[一作]` (East China Normal University), Christian S. Jensen `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于世界模型的多模态协变量时间序列预测框架 WorldTS，先将历史观测与多模态协变量映射到潜在空间并预测未来状态，再解码回观测空间。

**💡 创新点**

创新点：1）将多模态协变量直接融入潜在状态预测，而非仅在观测空间使用；2）采用两阶段训练：先在潜在空间进行有监督的未来状态学习，再冻结模型训练解码器；3）使用潜在空间匹配与 VICReg 正则防止表征坍塌。

**🔧 技术方法**

主要技术：CausalPatching+Transformer 预测器、潜在空间编码器/解码器、文本编码（GPT‑2）、图像编码（轻量 CNN）、CovEncoder、Modal Adapter、VICReg 正则、两阶段训练目标。

**📊 数据集**

在 21 个真实数据集上评估：12 个数值协变量数据集、8 个文本协变量数据集（Time‑MMD）、1 个图像协变量数据集（MMSP 太阳能功率）。

**📈 对比分析**

与 10+ 数值基线、9+ 文本基线、13+ 图像基线对比，WorldTS 在大多数数据集上实现最低 MSE/MAE；数值协变量提升约 18%，文本协变量提升约 9%，图像协变量提升约 9%。

**⚠️ 局限性**

局限：需要两阶段训练，训练过程较复杂；对协变量依赖较大，缺乏协变量时性能下降；在极大潜在维度或过短/过长补丁长度时表现不佳。

---

## 447. ReG-SAM: Reference Graph-Driven SAM for 2D Foundational Vessel Segmentation

**arXiv ID:** 2609.31160 | [PDF](https://arxiv.org/pdf/2609.31160v1)

**作者:** Donghang Lyu `[一作]` (Leiden University Medical Center), Marius Staring `[通讯]` (Leiden University Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了基于SAM的2D血管分割框架ReG-SAM，利用参考图和血管原型嵌入实现全自动分割。

**💡 创新点**

创新点在于引入图形提示嵌入（GPE）与血管原型嵌入（VPE），通过模态感知的参考图数据库实现无标注推理，并改进Mask Decoder以捕捉细粒度血管结构。

**🔧 技术方法**

采用SAM backbone、图神经网络（TransformerConv）、ResNet式CNN编码器、Vascular Prototype Attention、对比学习与L1损失、随机采样等技术。

**📊 数据集**

使用19个公开血管数据集（X光、DSA、PF、OCTA、SLO、Fundus等六种模态），并将ORVS数据集作为零样本测试集。

**📈 对比分析**

与SAM-Med2D、MedSAM、OVS-Net、ASPS、SAM-HQ等基准进行Dice和IoU评估，ReG-SAM在多数数据集获得最高或第二高分，在零样本集表现更优，整体精度提升约4‑5%。

**⚠️ 局限性**

受限于血管标注样本稀缺、参考数据库规模有限以及对低质量注解的敏感性，未来需更大、更高质量的多模态血管数据集以进一步提升性能。

---

## 448. CoralPlan: Observation Skill Selection and Execution for Underwater Robotic Inspection

**arXiv ID:** 2609.31211 | [PDF](https://arxiv.org/pdf/2609.31211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 449. Teacher-Anchored Selection of Post-Training Quantized Models under Domain Shift

**arXiv ID:** 2609.31155 | [PDF](https://arxiv.org/pdf/2609.31155v1)

**作者:** Alejandro Rodriguez Dominguez `[一作]` (University of Reading), Xia Hong `[通讯]` (University of Reading)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在已生成的压缩模型族中，如何在无标签或标签稀缺时基于共享教师选择最优候选模型。

**💡 创新点**

提出“anchored selection”框架，将无标签的教师距离作为基准，加入少量标签校正，并给出二次量子同构族的解析标识与对称噪声衰减定理。

**🔧 技术方法**

使用量化压缩、教师模型对齐、KL散度度量、加权校正、交叉验证系数选择、对称标签腐败模型与线性衰减分析等技术。

**📊 数据集**

在 DomainNet（ResNet50、MobileNetV2、ViT）和 CIFAR‑20 数据集上进行实验。

**📈 对比分析**

与多种无标签性能估计方法（平均置信度、entropy、ATC、DOC、SoftmaxCorr、COT、Nuclear norm）以及直接验证交叉熵比较，发现 anchoring 在标签稀缺且/或标签受腐败时显著降低目标交叉熵，性能优于传统方法。

**⚠️ 局限性**

anchor 方案仅在标签稀缺时有效，标签充足时校正收益消失；在某些架构下无标签估计已足够；方法对教师模型质量和候选族结构高度依赖。

---

## 450. CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks

**arXiv ID:** 2609.31149 | [PDF](https://arxiv.org/pdf/2609.31149v1)

**作者:** Yuxuan Qiu `[一作]` (University of Tokyo), Tetsuya J Kobayashi `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于化学反应网络（CRN）的计数空间扩散模型CRNDiff，并在推理时通过倾斜的Feynman–Kac（FK）调度实现对稀有子群的条件生成。

**💡 创新点**

创新点在于：①利用CRN框架定义可解析的出生–死亡前向噪声过程，得到闭式转移核、马尔可夫桥和阶乘-协方差动力学；②通过二阶过剩协方差自适应选择终止噪声时间；③设计两阶段的倾斜FK调度（先倾斜边缘分布再校正联合分布），实现无训练模型的条件采样；④将上述方法与scRNA‑seq数据结合，显著提升稀有细胞类型的生成质量。

**🔧 技术方法**

核心技术包括：化学反应网络（CRN）和随机质量作用动力学、生成函数与阶乘-协方差分析、前向过滤-后向采样（FFBS）、倾斜的Feynman–Kac调度、基于粒子滤波的逆向采样。

**📊 数据集**

实验使用成人心脏细胞图谱（Human Heart Cell Atlas）中的单细胞RNA‑seq数据，针对内皮、髓系和神经细胞三种目标细胞类型进行评估。

**📈 对比分析**

与scVI、scANVI、CFGen、MDLM等主流单细胞生成模型进行对比。CRNDiff在条件纯度（purity）、平均每基因Wasserstein‑1距离、最大均方差（MMD²）和皮尔逊相关系数（PCC）等多项指标上均优于对照组，尤其在稀有细胞类型下的纯度提升最为显著；同时在无条件生成时，切片Wasserstein‑1距离、Fano因子等统计量也逼近真实细胞分布。

**⚠️ 局限性**

局限性包括：①终止噪声时间仅基于二阶过剩协方差，可能忽略更高阶依赖；②倾斜FK调度在极稀有目标时仍可能出现重要性权重集中和粒子退化；③目前仅实现单分子出生–死亡网络，未探索更复杂的多反应网络；④模型在库大小和基因检测变异方面对真实数据的拟合仍有一定差距。

---

## 451. How can AI accelerate the green transition?

**arXiv ID:** 2609.31188 | [PDF](https://arxiv.org/pdf/2609.31188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 452. Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution

**arXiv ID:** 2609.31214 | [PDF](https://arxiv.org/pdf/2609.31214v1)

**作者:** Zhe Li `[一作]` (Singapore Management University), Jun Sun `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于计数因果估计的影响力框架，将训练样本对模型行为的影响形式化为指定的对抗式量化，并通过该框架区分规格不匹配与近似误差，组织现有影响力估计器，给出了局部动态分解，并在控制实验与实际任务（噪声标签检测与LLM归因）中验证了行为代理与规格选择对排名与检测性能的关键影响。

**💡 创新点**

核心创新在于：① 将影响力定义为三元组（行为、干预、对抗训练过程）的计数因果估计量；② 明确指出不同规格导致完全不同的“真实”排名，而近似误差只在规格一致时才显现；③ 通过局部积分与Jacobian分解统一描述梯度、轨迹、逆Hessian等多种估计器的共性结构；④ 在噪声标签与LLM归因实验中揭示，行为代理需与估计器及任务共同优化，默认的loss代理往往不最优。

**🔧 技术方法**

使用了计数因果推断、局部线性近似、Jacobian/NTK特征分解、梯度相似度、轨迹聚合、逆Hessian（LiSSA）、影响函数、留一法、以及对抗训练的路径积分等技术；实验中使用了Kendall’s τ、AUPRC、AUROC、top‑k精度等评估指标。

**📊 数据集**

实验数据集包括 FashionMNIST（logistic回归）、CIFAR‑10（ResNet‑18）用于噪声标签检测，Qwen3‑8B（fine‑tuned on ScienceQA）用于响应腐败与条件后门归因，另外还验证了 Gemma‑2‑9B‑it 与 Llama‑3.1‑8B‑Instruct 的相同设置。

**📈 对比分析**

在控制实验中，用Kendall’s τ比较不同规格下的排名，一致性随对抗扰动距离增大而下降；在实际任务中，AUPRC、AUROC与top‑k精度表明：① 与默认loss比较，目标logit或硬边缘等行为代理可显著提升噪声标签检测与LLM后门识别；② 同一估计器在不同行为代理下性能差异大；③ 在后门实验中，使用目标logit的TracIn可将攻击成功率从80%降至≈5%。

**⚠️ 局限性**

局限性包括：① 仍需任务级别的行为代理选择与验证，无法提供统一的“最佳”代理；② 计数因果估计对大规模模型的精确对抗训练往往不可行，仅能在可行范围内使用近似；③ 结果主要基于特定数据集与模型，泛化到更复杂或更大规模任务仍需进一步研究。

---

## 453. Self-Supervised Representation Learning: From Spectral Foundation Models to Auroral Emission Spectra

**arXiv ID:** 2609.31206 | [PDF](https://arxiv.org/pdf/2609.31206v1)

**作者:** Matthieu Le Lain `[一作]`, Sébastien Lefèvre `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

预训练 1D Vision Transformer（MAE）在 223k 条无标签极光光谱上学习表示，并在 811 条标注光谱上微调实现多标签分类，性能优于专家手工特征和以往监督模型。

**💡 创新点**

证明自监督预训练在相同光谱窗口内可捕获物理诊断信息，并通过与天文基础模型的对比揭示窗口重叠是迁移性能的关键；同时展示冻结表示已达到专家特征水平。

**🔧 技术方法**

采用 1D Vision Transformer + Masked AutoEncoder + 线性探针 + Integrated Gradients 等技术；对输入设置两通道（线性通道与基线通道）并使用信息化掩码。

**📊 数据集**

使用 ASIS Skibotn 极光光谱数据集：约 223,065 条未标注光谱用于预训练，811 条已标注光谱用于微调和评估。

**📈 对比分析**

通过与专家特征、未训练控制和先前监督模型的对比，冻结表示 mAP 0.863（比未训练 0.816 高 0.047），微调后 mAP 0.870，10% 标签时提升 0.159；同时与 SpecFormer 等外部模型的比较显示窗口重叠决定迁移效果。

**⚠️ 局限性**

局限在于对不同光谱窗口的迁移效果差异显著，长时间预训练成本高，稀缺类别样本有限导致评估不稳定。

---

## 454. KnottedGraph: Scalable knotted-graph topology for scientific and mathematical discovery

**arXiv ID:** 2609.31152 | [PDF](https://arxiv.org/pdf/2609.31152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 455. Complexity, approximation, and extension of proper $\{a,b\}$-edge-weightings

**arXiv ID:** 2609.31205 | [PDF](https://arxiv.org/pdf/2609.31205v1)

**作者:** Péter Madarasi `[一作]`, Máté Simon `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对固定整数对 {a,b} 的边加权问题进行系统研究，包含判定、优化与扩展三类问题：1）判定是否存在合法的 {a,b}‑加权（即相邻顶点的加权和不同），证明对所有固定的不同整数对，判定问题在简单三度平面图上是 NP‑完全的；2）在循环无边的平面多重图上给出最大化合法边数的 EPTAS，并证明在 ETH 下不存在 2^o(√m) 的算法；3）证明在简单三度图上该优化问题为 APX‑完整；4）研究从部分 {a,b}‑加权到完整加权的扩展问题，证明在简单三度平面双部图上为 NP‑完全，且在树上可多项式求解。

**💡 创新点**

创新点主要有：①首次证明在简单三度平面图上 {a,b}‑加权判定问题是 NP‑完全（之前仅对一般图或非平面三度图已知）；②构造了基于平面分离定理的树分解，得到 2^O(√m) 的最优子指数算法，并用 ETH 给出匹配下界；③提出了一种新颖的 EPTAS，利用层删、素数模运算以及平面图的对偶树 DP，完成最大合法边数的近似；④通过新型“同步器”与“反转器”构造的图，完成 2‑in‑4‑SAT 到 {a,b}‑加权的多种多重化归，完成优化问题的 APX‑完整性证明；⑤首次给出树上部分加权扩展的多项式算法，填补了此前仅有完全图或无加权的情况。

**🔧 技术方法**

核心技术包括：
- 平面分离定理与树分解相结合的 DP 方案（用于子指数算法）。
- 约束满足与权重映射的同步器/反转器构造（用于 NP‑完备归约）。
- 层删与素数模运算相结合的 EPTAS 框架（用于最大合法边数）。
- 对偶图与三角化的树 DP（用于 EPTAS 的内部优化）。
- 动态规划与可计数约束的树 DP（用于树上扩展问题）。

**📊 数据集**

本文主要为理论性工作，未使用具体实验数据集；所有结果均通过构造证明、算法设计与复杂性分析得到。

**📈 对比分析**

方法比较：
- 对判定问题，证明了 NP‑完备性并给出 2^O(√m) 的最优子指数算法；
- 对优化问题，提供 1/2 的多项式时间近似、2^O(√m) 的确定性子指数算法、以及 EPTAS；证明其在 ETH 下的下界与 APX‑完整性；
- 对扩展问题，显示在树上可多项式求解，而在平面双部三度图上为 NP‑完全。
总体性能：判定与扩展问题在一般图上不可多项式求解；在平面多重图上可实现子指数算法与 EPTAS；在树上实现多项式扩展算法。

**⚠️ 局限性**

局限性与未解决问题：
- 仅覆盖平面或三度图，未扩展到更广泛的图类别。 
- EPTAS 复杂度仍高（取决于 1/ε 与素数集合），实际实现难度大。 
- 对非平面图的 {a,b}‑加权判定与扩展仍未给出多项式算法或紧凑下界。 
- 仅针对整数权重，对实数权重的扩展仍有进一步研究空间。 
- 对扩展问题的最优下界（是否存在更强的 APX‑完整性或更好的多项式算法）尚未完全解决。

---

## 456. Evaluating the Impact of Adaptive Extended Reality on Human-Robot Interaction Across the Reality-Virtuality Continuum

**arXiv ID:** 2609.31138 | [PDF](https://arxiv.org/pdf/2609.31138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 457. Evolutionary Safety of Recursive Self-Improving AI: Taxonomy, Risk Discovery, and Evaluation

**arXiv ID:** 2609.31186 | [PDF](https://arxiv.org/pdf/2609.31186v1)

**作者:** Chang Gong `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Ruijie Guo `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“进化安全”（Evolutionary Safety）框架，系统性分析递归自我改进（RSI）下安全属性的演化，构建风险分类、评估单元和治理原则。

**💡 创新点**

将安全视角扩展至持久性与递归改进的过程；定义六类安全退化表现；构造跨五域（代理状态、模型状态、评估与环境、计算 substrate、元层更新）分类体系；提出基于状态、更新、轨迹和谱系的评估与治理方法。

**🔧 技术方法**

概念建模、风险映射、对比实验协议、基准设计、评估单元化（状态、更新、轨迹、谱系）、安全治理原则（修改边界、预提交门控、独立验证、谱系追踪、恢复策略）。

**📊 数据集**

引用并结合现有安全与自适应基准：SafetyBench、RM-Bench、JudgeBench、EAL-Bench、EvoPathBench、L2RPN、OpenAI/ChatGPT 等公开数据集与实验平台，补充自研实验记录。

**📈 对比分析**

采用对比方法：更新保持（前后状态对比）、累计风险（多轮更新风险累积）、持续性检验（源移除后重测）、传播评估（谱系传递）、恢复评估（修复后再适配）。在上述基准上报告风险率、累计暴露、恢复后性能与安全的平衡；但缺乏统一的量化指标与跨系统基准对比。

**⚠️ 局限性**

局限性：框架主要理论化，缺乏统一的实证验证；评估基准难以覆盖所有进化路径；评估方法在动态评判者、隐蔽风险、社交适应等方面仍有限；治理措施需要在真实 RSI 系统中进一步验证与迭代。

---

## 458. AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side

**arXiv ID:** 2609.31166 | [PDF](https://arxiv.org/pdf/2609.31166v1)

**作者:** Ryoma Sato `[一作]` `[通讯]` (National Institute of Informatics), Ryoma Sato (National Institute of Informatics)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了AgentRecommender，一种基于LLM代理的用户侧推荐系统，可在不获取额外标签数据的情况下构建满足用户自定义约束的推荐列表。

**💡 创新点**

创新点在于利用LLM内部知识和工具调用实现对隐藏属性的零射估计，并通过主动探索推荐网络，在用户侧实现无标签、可定制的推荐功能。

**🔧 技术方法**

技术主要包括大语言模型（GPT‑5.6 Luna）、结构化提示与JSON schema输出、主动搜索与工具调用、预算限制下的图遍历与再排序。

**📊 数据集**

实验使用了MovieLens 1M、LastFM和Amazon Home & Kitchens三个公开数据集。

**📈 对比分析**

与官方推荐器及iAgent等基线对比，AgentRecommender在满足多种属性约束（如年代、类型、价格）时，成功率提升至约0.6–0.68，类别覆盖率提升约1.5–4倍，且在预算从1到16的范围内均表现优异。

**⚠️ 局限性**

局限性包括对LLM知识的依赖、属性估计不精确时可能失效、需要多次额外查询导致延迟以及在价格属性难以推断时性能下降。

---

## 459. Unknown-Traffic Detection, Calibration and Shortcut Reliance in Distilled Encrypted-Traffic Classifiers over One Year

**arXiv ID:** 2609.31141 | [PDF](https://arxiv.org/pdf/2609.31141v1)

**作者:** Mahmoud Abbasi `[一作]` `[通讯]` (University of Salamanca), Mahmoud Abbasi (University of Salamanca)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过教师交换控制，研究了知识蒸馏在加密流量分类器中对未知流量检测、校准、快捷依赖和概念漂移等属性的继承。

**💡 创新点**

创新之处在于设计了教师交换实验以区分教师特定信号与正则化效应，并在真实一年流量数据上评估了继承效果、温度影响和快捷特征传递。

**🔧 技术方法**

采用了Hinton蒸馏、EnDD、标签平滑、温度缩放以及能量分数、最大软化概率、马氏距离、kNN等技术，配合集群重采样统计。

**📊 数据集**

使用CESNET‑TLS‑Year22数据集，包含25M TLS流、180个Web服务，划分102个已知服务与未知服务，进行35周的测试窗口。

**📈 对比分析**

对比蒸馏学生与直接训练学生、教师、标签平滑和EnDD，利用宏F1、未知检测AUROC、FPR@95、OSCR、ECE、NLL、AURC等指标，结果显示蒸馏仅传递得分模式，未提升检测或校准；教师优势在特征空间分数下显现，学生在时间上比教师衰退更慢。

**⚠️ 局限性**

局限性包括仅在单一数据集与模型规模上实验，未深入特征层蒸馏；教师规模受限；快捷特征实验为合成；评分文件量化导致部分指标偏差；结果受特定评分规则影响。

---

## 460. TaskIR: Task-Driven Image Restoration via Degradation Adaptation and Task Feedback

**arXiv ID:** 2609.31170 | [PDF](https://arxiv.org/pdf/2609.31170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 461. BreathGRU: A Novel Semi-Supervised Bidirectional Gated Recurrent Unit Framework for Speech and Breath Segmentation for Respiratory Audio

**arXiv ID:** 2609.31165 | [PDF](https://arxiv.org/pdf/2609.31165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 462. Deep Reinforcement Learning for Misbehavior Detection Under Partially Observable V2X Data

**arXiv ID:** 2609.31217 | [PDF](https://arxiv.org/pdf/2609.31217v1)

**作者:** Roshan Sedar `[一作]` (Centre Tecnològic de Telecomunicacions de Catalunya), Charalampos Kalalas `[通讯]` (Centre Tecnològic de Telecomunicacions de Catalunya)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在部分可观测的车辆到一切（V2X）数据下的误行为检测，提出了一种基于深度强化学习（DRL）的检测框架，并考虑了自然遮挡与对抗性特征抑制攻击。

**💡 创新点**

创新点在于将误行为检测转化为序列决策问题，利用DRL在不完整观测下学习自适应策略，并首次引入利用自然遮挡的对抗性攻击模型。

**🔧 技术方法**

采用了深度强化学习（LSTM+DQN）、对抗性特征抑制与自然遮挡模拟、XGBoost基线对比、滑动窗口技术以及统计缺失模式（MCAR/MAR/MNAR）进行评估。

**📊 数据集**

使用了公开的VeReMi数据集，其中包含多种误行为类型和不同交通密度的消息。

**📈 对比分析**

通过准确率、F1分数和攻击成功率（ASR）在不同缺失模式和攻击场景下与XGBoost基线进行比较；DRL在自然遮挡下显著优于XGBoost，但在利用遮挡的对抗攻击下ASR可超过70%；在特征抑制攻击中DRL逐步下降，XGBoost表现出突然崩溃。

**⚠️ 局限性**

局限性包括：在利用自然遮挡的对抗攻击下易被逃避；对特征抑制攻击的鲁棒性不足，需要进一步对抗训练；实验仅在VeReMi上完成，未验证实时部署与计算开销。

---

## 463. ALF: An Active Learning Framework for Scientific Discovery

**arXiv ID:** 2609.31197 | [PDF](https://arxiv.org/pdf/2609.31197v1)

**作者:** Shikha Surana `[一作]` (InstaDeep), Paul Duckworth `[通讯]` (InstaDeep)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ALF框架，用于在科学实验中高效地进行数据获取和实验设计，支持离线（对现有数据集）和在线（对昂贵实验oracle）两种完整的数据采集循环；

**💡 创新点**

创新点在于提供了单一统一的API，既能用于离线基准测试，又能直接部署到真实实验环境，并通过模块化、数据类型无关的设计实现了高度可扩展性；

**🔧 技术方法**

采用主动学习与贝叶斯优化相结合的技术，利用多种Surrogate模型（CNN、MLP、GP、ESM‑2、Chemprop等）、多种Acquisition函数（Greedy、UCB、EI、TS、CoreSet等）以及搜索函数（增广与生成式模型）和可替换Oracle实现完整的ask/tell循环；

**📊 数据集**

使用了多领域数据集，包括蛋白质序列（GFP、ProteinGym、FLIP）、小分子（GuacaMol）和材料（MatBench），并在这些数据集上验证了框架的通用性；

**📈 对比分析**

通过在上述三类任务中进行离线Benchmark和在线实验，比较了七种采集策略，实验表明ALF在主动学习场景下平均RMSE下降，在贝叶斯优化场景下最大得分提升，整体性能优于现有单一模式工具；

**⚠️ 局限性**

局限性包括对生成式搜索和oracle调用的支持仍相对有限，实验范围主要集中在少数任务，尚未充分验证在更复杂多模态或高成本实验环境中的可迁移性和鲁棒性。

---

## 464. SPO: Discovering Adaptive Large Neighborhood Search Operators via Stackelberg Program Optimization

**arXiv ID:** 2609.31179 | [PDF](https://arxiv.org/pdf/2609.31179v1)

**作者:** Xinyi Ke `[一作]` (Chinese Academy of Sciences), Jian Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过堆叠式程序优化框架（SPO）自动发现可根据 LNS 进程状态自适应的摧毁和修复算子，并将其作为可执行程序进行学习与演化。

**💡 创新点**

创新点包括：① 将摧毁算子设为 Stackelberg 领导者、修复算子设为从属跟随者，利用领导-跟随信用分配实现互相优化；② 在程序生成时引入 LNS 状态作为条件输入，使算子本身包含对搜索进程的自适应决策；③ 结合 LLM 生成器与基于群体的进化搜索形成耦合的生成-种群优化循环。

**🔧 技术方法**

使用技术：LLM（带 LoRA）生成可执行程序；Group Relative Policy Optimization（GRPO）优化生成器；演化算法更新算子种群；堆叠式信用分配；LNS 迭代评估并在 100 步/500 步 roll‑out 上计算改进效益。

**📊 数据集**

数据集：在 50 节点欧氏 TSP 和 CVRP 进行训练，测试时包含更大规模的 TSP（至 500 节点）和 TSPLIB 标准实例，以及 CVRPLIB 的 A、B、P、X 系列。

**📈 对比分析**

对比方法：手工构造算子（NN、FI、2‑opt、3‑opt、ALNS）与其他 LLM 算子发现框架（EoH、FunSearch、ReEvo、G‑LNS），以及求解器基线（Clarke‑Wright、OR‑Tools）。SPO 在所有 TSP、CVRP 组别均实现最低参考间隙；在扩大规模和跨分布时性能提升更为显著，且在 500 步 LNS 迭代下持续优于基线。

**⚠️ 局限性**

局限性：① 依赖大规模 LLM 生成与算子演化，计算成本高；② 对 LNS 状态的单一信号（迭代无改进次数）可能不足以捕捉更复杂的搜索动态；③ 在更大规模或不同类型问题的泛化尚未完全验证，需要进一步扩展与评估。

---

## 465. Seeing Semantic Shift: Difference-Aware Sentence-Level Temporal Segmentation of Sign Language Videos

**arXiv ID:** 2609.31148 | [PDF](https://arxiv.org/pdf/2609.31148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 466. HyperErase: Scale-Calibrated Hypernetwork for Multi-Concept Erasure in Text-to-Image Models

**arXiv ID:** 2609.31154 | [PDF](https://arxiv.org/pdf/2609.31154v1)

**作者:** Yi Sun `[一作]` (Harbin Institute of Technology), Yuxia Qiao `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于超网络的多概念擦除框架 HyperErase，能够在文本到图像模型中根据输入提示自动生成对应的 LoRA 适配器，从而实现目标概念的抹除。

**💡 创新点**

核心创新在于把概念擦除重新定义为“提示条件参数摊销”，使用单一超网络将文本描述映射到专属的 LoRA 更新；另外引入分离尺度校正（decoupled magnitude rectification）和平方根尺度编码，显著提升生成稳定性与擦除效果。

**🔧 技术方法**

主要技术包括：超网络（HyperNetwork）生成器、LoRA 参数表征、基于 BERT 的提示编码、解耦模式-尺度表示、平方根尺度编码、教师引导的先验校正，以及对 Stable Diffusion U‑Net 的 LoRA 微调。

**📊 数据集**

实验数据集主要使用 Stable Diffusion v1.4 预训练模型，并在 I2P（色情内容）和 MSCOCO 10000 条提示上进行评估，同时也在 FLUX.1‑dev 进行跨模型验证。

**📈 对比分析**

与 ESD 单概念、混合、顺序擦除等基线对比，HyperErase 在 14 个擦除任务（裸体、艺术风格、物体）上实现了与 ESD 单概念相当或更优的擦除效果，同时保持更低的 FID、更高的 CLIP 分数，且在持续学习场景下不出现显著生成退化。

**⚠️ 局限性**

主要局限包括：在按顺序连续擦除时若未采用正则化，易出现灾难性遗忘；超网络对概念数目敏感，概念多样性大时生成精度可能下降；实现仍需依赖基线 LoRA 生成数据，训练成本相对较高。

---

## 467. FedHisto-PAST: Parameter-Efficient Stain-Aware Federated Learning for Cross-Site Lung Histopathology Classification

**arXiv ID:** 2609.31150 | [PDF](https://arxiv.org/pdf/2609.31150v1)

**作者:** Muhammad Muhtasim Shahriar `[一作]` (International Islamic University Chittagong), Mohammad Ali Moni `[通讯]` (Charles Sturt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 FedHisto-PAST v2，一种结合冻结的 HIBOU-B 基础模型、低秩适配、颜色条件 FiLM、配对色差一致性、可靠性感知原型和自适应聚合的参数高效联邦学习框架，用于跨站肺组织病理图像的三分类。

**💡 创新点**

将颜色条件的配对对抗视图与预测/特征一致性、缺失类可靠性原型正则化以及自适应聚合统一到同一框架，实现了在保持基础模型冻结的前提下的跨站、非 IID 数据的高效联邦学习。

**🔧 技术方法**

低秩适配（LoRA）、FiLM 条件模块、Jensen‑Shannon 预测一致性、特征一致性、可靠性加权原型正则化、基于样本量与一致性自适应加权的 FedAvg 聚合、冻结 HIBOU-B Transformer、随机裁剪与颜色扰动等。

**📊 数据集**

内部使用 LC25000 与 WSSS4LUAD 合并的 8,922 张图像进行五客户端非 IID 模拟，外部作为探索性交叉数据集的 LungHist700 共 691 张图像。

**📈 对比分析**

与 FedAvg-PEFT、FedProx-PEFT、FiLM‑Zero、FedHisto‑PAST v1 等基线在内部固定划分下的宏 F1 近乎 0.999，外部 LungHist700 上 v2 的宏 F1 为 0.729，显著优于其它基线（v1 0.677，FiLM‑Zero 0.672 等），提升了 Normal 与 SCC 的召回率但降低 ACA 召回率；同时保持 1.25% 参数更新与 10% 额外通信开销。

**⚠️ 局限性**

仅在图像级别的模拟客户端，缺乏病人/切片级别独立性；外部评估为开发导向的探索性数据集，未进行正式临床验证；未实现差分隐私、加密聚合或攻击鲁棒性；内部性能接近上限，难以区分方法优劣；校准仍差。

---

## 468. Onboard Wind-Preview Model Predictive Control Using Pitot-Static Sensing for Multirotor UAVs

**arXiv ID:** 2609.31185 | [PDF](https://arxiv.org/pdf/2609.31185v1)

**作者:** Bas Meere `[一作]` (Eindhoven University of Technology), Duarte Antunes `[通讯]` (Eindhoven University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出利用一根低成本轻量级的pitot静压管悬挂在机体前方的延长杆上进行风速预览，并将此预览信息嵌入非线性模型预测控制（MPC）中，以实现对无人机悬停时风激波的前瞻性抵消；通过仿真与室内外实验探究延长杆长度与悬停精度之间的权衡，并验证该方法在不同风速条件下的性能提升；

**💡 创新点**

创新点在于首次将机体前方的pitot预览传感器与MPC相结合，实现了完全机载的风预览控制；通过系统化的延长杆长度与风速/机动性耦合分析，揭示了最佳预览距离并非固定，而随风速与无人机响应速度变化；

**🔧 技术方法**

采用低成本5g pitot静压管、四轴飞行器的非线性MPC、Acados求解器、基于Runge-Kutta离散化的系统模型；同时使用EMA、滚动最大值及Sigmoid阈值对pitot信号进行三阶段滤波；

**📊 数据集**

使用了evoturb生成的IEC Class A Normal Turbulence（NTM）模拟风场（4m、7m、10m等均速场）进行仿真；室内实验通过工业风扇产生的可重复风流；室外实验利用地面超声波风速计测得的实地风速；

**📈 对比分析**

与PX4标准位置模式和未使用风预览的MPC做对比；室内实验中风预览MPC的RMSE从0.144m降至0.048m，分别比基线和无预览MPC降低66%和60%；室外实验中沿风方向的RMSE降低54%；仿真中显示在4m/7m风速下，最佳延长杆长度约为1.25m，较短或较长均导致误差升高；

**⚠️ 局限性**

局限性包括：需保持稳定的主风向，单一方向的pitot受±20°可接受角限制；低风速下传感器误差增大（<1m/s不可靠，RMSE≈0.72m/s）；延长杆增加惯性导致极端风速或机动性差的平台不易使用；若环境中风向频繁变化或多向扰动，单一预览传感器效果有限。

---

## 469. Where a Model Sends Its Own Repeated Token

**arXiv ID:** 2609.31181 | [PDF](https://arxiv.org/pdf/2609.31181v1)

**作者:** Nicolás Vera Zúñiga `[一作]` `[通讯]`, Nicolás Vera Zúñiga

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大模型在输入自身重复token（t,t）时的自回归输出，构建了整个词表上的“自续点”与“目的地”映射，并对19个模型、7种分词器、不同语料的表现进行系统测量。

**💡 创新点**

创新点在于：①将自续点和目的地映射扩展到整个词表而非样本子集；②通过解析目的地（非自续点的最大概率输出）来比较模型，发现此映射比固定点更能区分模型族；③系统评估了量化、数值精度等鲁棒性边界。

**🔧 技术方法**

使用的方法包括：单步前向推理获取argmax、Hamming距离/一致率比较、对数it margin记录、量化实验（8/4位对称量化、按通道分组量化），以及离散化后将token解码为字符串以实现跨分词器比较。

**📊 数据集**

数据集：采用了从Pile（约2000篇文档）抽取的2000个最频繁词汇（包括空格前缀、ASCII区块及空白列表）作为probe集，此外模型训练语料不统一，涉及Pile、非Pile等多种语料，确保结果不受单一语料影响。

**📈 对比分析**

比较方法：①固定点集合用Hamming距离评估；②非固定点的目的地用字符串一致率评估；③对不同模型族、分词器和语料进行交叉比较，并在不同量化/精度条件下重新测量。性能结果显示：目的地映射对模型族的归属准确率达0.8333（相对随机0.1389），对归档架构的聚类准确率可达0.90；8位量化保持约0.90一致率，4位量化则几乎失效。

**⚠️ 局限性**

限制包括：仅为观察性对照，无法揭示因果机制；模型族与分词器仍部分共线；probe集交集随模型多样性缩小；量化研究仅限权重量化，未覆盖激活量化或真实部署环境；测量为确定性，无法识别实例ID；并且固定点映射被证明受集合大小支配，信息量有限。

---

## 470. Audio emotion recognition for atypical hearing

**arXiv ID:** 2609.31168 | [PDF](https://arxiv.org/pdf/2609.31168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 471. Neural State Prediction: Obstructing Shortcut Learning in EEG Foundation Models

**arXiv ID:** 2609.31167 | [PDF](https://arxiv.org/pdf/2609.31167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 472. Bayesian Tensor Autoencoder with Physics-informed Predictive Prior for Multi-dimensional Time Series Anomaly Detection

**arXiv ID:** 2609.31157 | [PDF](https://arxiv.org/pdf/2609.31157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 473. DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration

**arXiv ID:** 2609.31215 | [PDF](https://arxiv.org/pdf/2609.31215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 474. Enabling a Unified Cross-Domain Representation for Two-Finger Gripper Manipulation via Interaction-Centric Modeling

**arXiv ID:** 2609.31207 | [PDF](https://arxiv.org/pdf/2609.31207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 475. BAT-CLIP: Trimodal Alignment of Brain, Audio and Text

**arXiv ID:** 2609.31180 | [PDF](https://arxiv.org/pdf/2609.31180v1)

**作者:** Suhyun Kim `[一作]`, Jiook Cha `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

针对自然语言听觉神经信号，提出 BAT‑CLIP 框架，将iEEG编码器的嵌入通过对比学习同时对齐到预训练的音频和文本嵌入空间，从而学习到同时保留语音时序与语义信息的神经表征。

**💡 创新点**

创新点在于：①首个三模态（脑‑音频‑文本）CLIP对齐方案；②利用自监督预训练的脑基底模型（DIVER‑1）作为对齐前置，显著提升对齐效果；③通过对比实验验证三模态对齐在多种下游语言检索任务上优于单模态对齐。

**🔧 技术方法**

使用的技术包括：自监督预训练的脑Transformer模型 DIVER‑1；冻结的 AudioCLIP 音频与文本编码器；对齐时采用双向交叉熵的三模态对比损失；以及对比实验中的线性探测头。

**📊 数据集**

主要使用的数据集为 Podcast intracranial electrophysiology 数据集，包含约 30 分钟的自然对话 iEEG、音频与文字对齐信息，并使用官方 Podcast Benchmark 评测套件。

**📈 对比分析**

在六个线性探测任务（内容/非内容词、词性、句子起始、GPT‑2 语言难度、词嵌入、Whisper 潜在向量）中，BAT‑CLIP 在大多数任务上超过了基准 CNN、未对齐的 DIVER‑1 以及单模态 BA‑CLIP/BT‑CLIP；在内容词分类略逊于 BA‑CLIP，句子起始检测未见提升，说明对齐有时会削弱对短时事件的敏感性。

**⚠️ 局限性**

局限性包括：①仅在小规模 iEEG 数据上训练与评估，扩展到更大数据集的效果未知；②对齐后主要使用线性探测，未验证端到端解码性能；③三模态对齐可能弱化对时序敏感的任务（如句子起始检测）；④iEEG 数据获取难度高，实验可重复性受限。

---

## 476. Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting

**arXiv ID:** 2609.31169 | [PDF](https://arxiv.org/pdf/2609.31169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 477. Semantic Navigation for Issue Localization in Code Repository

**arXiv ID:** 2609.31176 | [PDF](https://arxiv.org/pdf/2609.31176v1)

**作者:** Yunxiang Wei `[一作]` (Independent Researcher), Jundong Li `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了SemNav框架，通过语义导航图、语义卡片和候选工作区，实现了在软件仓库中针对问题报告的文件/函数级定位，并支持LLM在迭代中不断更新候选集。

**💡 创新点**

创新点在于：①将语言服务器在线解析的语义关系与预构建的语义导航图结合，消除了目标模糊；②使用语义卡片提供源代码上下文的压缩解释，显著降低LLM的上下文负担；③设计候选工作区，将候选位置与其证据链持久化，便于多轮修正与比较。

**🔧 技术方法**

技术包括：基于AST的离线图构建；Pyright语言服务器在线解析关系；BM25检索与关系扩展搜索；Semantic Card生成与缓存；LLM代理（Gemma 4B、Qwen系列）与六个工具（Search、Expand、Read、Update、Compare、Submit）。

**📊 数据集**

使用数据集：SWE‑bench Lite（300个问题），PLocBench（821个问题）和SWE‑Explore（368个实例），涵盖文件/函数级定位与源证据质量评估。

**📈 对比分析**

与工作流驱动和代理驱动基线相比，SemNav在大多数指标上名列前茅；例如在Gemma 4B上，File Hit@10从68.33%提升至82.67%，Function Recall@10从32.92%提升至40.72%；在PLocBench上Acc@5从60.41%提升至67.36%；在SWE‑Explore上线性精度从53.35%提升至79.99%，并在下游修复任务中将解决率从44.00%提升至52.33%。

**⚠️ 局限性**

局限性包括：仍未达到oracle上下文的性能；对语言服务器的依赖使得动态或未解析符号的处理受限；在构建和维护语义图、卡片以及工作区时存在计算与存储开销；目前仅支持文件/函数级定位，未覆盖更细粒度的代码块或行级定位。

---

## 478. I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?

**arXiv ID:** 2609.31161 | [PDF](https://arxiv.org/pdf/2609.31161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 479. Rethinking Data Quality for AI-Driven Systems: Evidence from Practitioner Interviews

**arXiv ID:** 2609.31191 | [PDF](https://arxiv.org/pdf/2609.31191v1)

**作者:** Hariharan Gopinath `[一作]` (Chalmers University of Technology), Helena Holmström Olsson `[通讯]` (Malmö University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对九家组织16位从业者的半结构化访谈，采用反射性主题分析方法，提炼出 AI 驱动系统数据质量的六个实践主题，并进一步归纳出五个解释机制，提出生命周期保障（lifecycle assurance）的概念框架。

**💡 创新点**

创新点在于将数据质量从传统的数据集属性扩展到嵌入模型行为、模型评估、代理记忆等 AI 具体场景，并通过“行为层可追溯性”“模型评估者”“代理记忆”“真实性”“合规门槛”“行为覆盖性”六大主题与五个根本机制的交叉分析，形成了以证据链为核心的生命周期保障视角。

**🔧 技术方法**

主要技术手段为：访谈设计、音频转录（WhisperX）、人工校正、Taguette 代码化、反射性主题分析（Braun & Clarke 的六阶段）以及跨主题归纳的解释机制构建。

**📊 数据集**

论文不使用传统机器学习数据集，而是采集了从业者访谈文本作为定性数据源；若需验证，可在未来工作中引入行业标准数据集进行案例验证。

**📈 对比分析**

由于研究性质为质性探索，未进行数值比较或性能评估；结果以主题阐释和概念模型为主，未与实验或基准进行对比。

**⚠️ 局限性**

局限性包括样本主要聚焦于汽车安全领域，可能限制跨行业推广；访谈和编码为单研究者主导，缺乏多位编码者的交叉验证；缺少量化评估，无法验证生命周期保障框架在实际项目中的效能。

---

## 480. A Class of Shift-Invariant Permutations and Their Algebraic Structure

**arXiv ID:** 2609.31174 | [PDF](https://arxiv.org/pdf/2609.31174v1)

**作者:** Cheng Lyu `[一作]` (Hubei University), Lei Hu `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一类由递归定义的映射生成的𝔽_2^n的平移不变置换，探讨了其代数结构。

**💡 创新点**

提出了平移不变置换的多项式组合性质与准守恒景观之间的等价条件，并确定了函数序列的长期行为。

**🔧 技术方法**

使用了代数框架和多项式计算，特别是通过构造同态映射来研究置换的性质。

**📊 数据集**

使用了𝔽_2^n的多种景观函数，特别是准守恒景观，构造了超过一百个平移不变置换的示例。

**📈 对比分析**

通过与现有方法的比较，展示了所提出的置换在性能和效率上的优势，尤其是在多项式计算和代数结构的统一性方面。

**⚠️ 局限性**

限制在于对于某些特定的景观，平移不变置换的性质可能不完全适用，且在某些情况下，平移不变置换的构造可能不具备简单的闭合形式。

---

## 481. The existence of polyhedral invariants is undecidable for linear systems

**arXiv ID:** 2609.31146 | [PDF](https://arxiv.org/pdf/2609.31146v1)

**作者:** David Monniaux `[一作]` `[通讯]`, David Monniaux

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过从线性计数器机归约，证明了在仅使用线性算术的程序中，存在多面体归纳不变量（用于证明某控制位置不可达）的判定是不可判定的。

**💡 创新点**

创新点在于提出一种无二次守卫、仅用抛物线轨迹与重播机制即可完成归约的技术，消除了之前方案中对二次守卫的依赖。

**🔧 技术方法**

主要技术包括抽象解释、凸多面体理论、线性算术约束、归约构造以及对抛物线的张量函数与重播步骤的数学分析。

**📊 数据集**

本文为理论证明性质，未使用任何实验数据集。

**📈 对比分析**

由于研究聚焦于理论可判定性，未进行实验对比；结果表明无论采用何种推理方法，若不受限于特定子类，求解过程必然不完备。

**⚠️ 局限性**

局限性：证明仅适用于具有 6 个数值变量的确定性程序，且控制转移使用封闭线性约束；对于 2–5 维情形以及更一般的非封闭约束仍是未解决的问题。

---

## 482. ChronoFuseGS: Multi-Temporal Gaussian Fusion with Per-Splat Persistence and Change Visualization

**arXiv ID:** 2609.31339 | [PDF](https://arxiv.org/pdf/2609.31339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 483. Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence

**arXiv ID:** 2609.31159 | [PDF](https://arxiv.org/pdf/2609.31159v1)

**作者:** Ahmed-Rafik Baahmed `[一作]` (CESI), Mourad Zghal `[通讯]` (CESI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个面向资源受限 IoT 边缘设备的个性化时序学习框架，利用分裂学习和知识蒸馏实现训练时的服务器指导与推理时的自治推断。

**💡 创新点**

创新点包括：① TeRR‑SAtt 通过固定稀疏 reservoir、轻量级学生网络和注意力机制，解决了训练-部署不匹配问题；② AMGF 利用学习路径动量进行客户端聚类，生成个性化教师更新，并通过可靠门控的预期学习实现动态协作。

**🔧 技术方法**

使用技术包括：分裂学习、时序稀疏 reservoir、GRU 与注意力模块、知识蒸馏、动量聚类（Affinity Propagation）和可靠门控的预期学习。

**📊 数据集**

实验使用 LBNL 智能建筑温度预测数据集（20 个热区）。

**📈 对比分析**

与传统 FL、传统 FSL 以及全模型 FL/FSL 进行对比。TeRR‑SAtt 在边缘训练延迟降低 65.5%，推理延迟降低 44.7%，内存使用降低 18.4%，CPU 使用降低 33.1%。AMGF 在 RMSE 上相较全局教师蒸馏提升最高 35.31%，平均提升 5.14%。

**⚠️ 局限性**

局限性包括：服务器训练仍需通信；聚类算法复杂度为 O(N²)；预期参数需手动调优；在极端非 IID 或突发动态下，聚类与预期可能不稳定；验证仅在智能建筑场景，需要进一步跨域评估。

---

## 484. JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models

**arXiv ID:** 2609.31142 | [PDF](https://arxiv.org/pdf/2609.31142v1)

**作者:** Jianyi Hu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Leo Yu Zhang `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了针对强化学习校准决策（RLCD）模型的首个对抗性基准，设计了 812 个问题、66 个场景和 9,744 个单编辑对抗变体，评估了模型在面对黑盒攻击时的决策翻转、漂移以及置信门阈值下的审查负担。

**💡 创新点**

创新点包括：①首次为 RLCD 形式化对抗基准，采用无标签的“决策翻转”度量并对照相同请求的重复跑作噪声基线；②提出结构化的单编辑攻击模板（问题、状态、注入、结构控制），并通过计费 token 进行交付验证；③引入置信门阈值评估，量化攻击后决策被送至人工复核的比例。

**🔧 技术方法**

技术方法主要是：黑盒单编辑攻击、无梯度/无搜索的模板式变体、对模型返回结果的重复跑比对、基于场景聚类的 95% 置信区间自助法、统计显著性检验（Holm 校正配对检验）以及置信门阈值下的 AUROC 与审查负担分析。

**📊 数据集**

使用的数据集为 66 个由供应商公开场景扩展而来的问答集合，包含 812 个问题（314 true/false、337 多选、161 等级），其中 143 个问题由人工审核标注。对抗样本基于对每个问题生成 12 个单编辑变体，合计 9,744 条请求。

**📈 对比分析**

与噪声基线（相同请求的重复跑）对照，观察到：非攻击性重写翻转率≤1.2个百分点；最强攻击——观察者意见（未加入指令）翻转率 12.1%，注入命令 10–13%；攻击后 38% 的高置信答案被推至 0.8 以下，导致审查负担显著上升。相较于无攻击状态，模型对抗性低于 2% 的噪声基线，表明该模型在 RLCD 任务上易被单一文本编辑触发决策改变。

**⚠️ 局限性**

局限性包括：仅评估单一模型版本（Jev v0.0.1，2026‑09‑25），攻击方式极其简化（无搜索、无自适应）；对 143 人工标注样本的准确性验证仅由一位 LLM 注释员完成；大多数标签来源于模型自身的共识，缺乏外部真实标注；API 的预处理、置信计算公式及版本别名均未公开；因此结论对其他 RLCD 模型、后续版本或更强攻击者的适用性有限。

---

## 485. Can Linguistic Reasoning Vectors Enhance Multimodal Reasoning Ability?

**arXiv ID:** 2609.31140 | [PDF](https://arxiv.org/pdf/2609.31140v1)

**作者:** Ziyi Wang `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LIFT 方法，通过在不更新 VLM 主干的情况下，将基 LLM 的语言侧推理向量迁移到对应的 VLM，以提升多模态推理性能

**💡 创新点**

创新点在于将推理向量（Reasoning Vectors）作为轻量级向量干预，证明基 LLM 的推理表示在多模态扩展后仍可被利用，并通过可学习的向量适配进一步增强效果

**🔧 技术方法**

采用了向量干预（vector injection）、可学习向量适配、层级选择、基于答案 token 的隐藏状态差分来提取 Reasoning Vectors，并在多模态 VLM 的语言层进行注入

**📊 数据集**

在六个推理基准上评估：文本推理集 GSM8K、CommonsenseQA、StrategyQA；多模态推理集 MathVista、MathVision、ScienceQA；使用 Qwen2.5‑VL‑7B‑Instruct 与 InternVL2.5‑8B 两个 VLM

**📈 对比分析**

与 VLM 基线、零样本 CoT、四种推理向量来源（LLM‑derived、VLM‑derived、VLM‑MM）以及可学习适配进行对比；LLM‑derived 向量在两种 VLM 上平均提升约 1–2%，甚至超过 4‑shot CoT；可学习适配进一步提升 0.5–1%

**⚠️ 局限性**

局限包括：仅验证两种 VLM 架构；需要支持样本和层级选择；仅对语言侧隐藏状态干预，无法直接纠正视觉感知错误；对不同多模态任务的泛化性待验证

---

## 486. Resource-Optimized and Energy-Aware Agentic AI Framework Anchored on Blockchain for Secure Software Supply Chains

**arXiv ID:** 2609.31282 | [PDF](https://arxiv.org/pdf/2609.31282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 487. New distance bounds for one-generator quasi-cyclic codes with applications to Hermitian LCD codes and entanglement-assisted quantum codes

**arXiv ID:** 2609.31300 | [PDF](https://arxiv.org/pdf/2609.31300v1)

**作者:** Kanat Abdukhalikov `[一作]` (UAE University), Rasha M. Shat `[通讯]` (UAE University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了针对任意指数的一生成元准循环码的块支持下界，并用其构造了许多具有良好参数的四元Hermitian LCD码及其对应的最大纠缠辅助量子码。

**💡 创新点**

创新点在于通过块截断与块缩短操作得到新的距离下界，并在平衡一生成元准循环码中建立了非递减的下界链，首项即Jensen界；同时给出了四元Hermitian码的多项式本体与LCD判定条件，提升了搜索效率。

**🔧 技术方法**

使用的技术包括块支持距离分析、Chinese Remainder Theorem分解、拼接描述、Hermitian内积与本体分析，以及Magma计算进行自动搜索和距离计算。

**📊 数据集**

数据集主要是自定义的准循环码生成多项式，长度上限为63，搜索覆盖了多种指数与块长组合，未使用公开标准数据集。

**📈 对比分析**

与已有的Araya‑Harada表、Grassl表以及近期文献的EAQECC表进行对比，成功改进了6条Araya‑Harada下界、13条EAQECC参数，并在Grassl表中提供了66条未记录的新量子码；总体上表现出更高的纠错距离与更低的纠缠需求。

**⚠️ 局限性**

局限性包括只考虑了一生成元准循环码（未覆盖多生成元或准扭转码），块支持下界尚未与所有已知距离上界（如Lally、Güneri‑Özbudak、谱上界等）进行系统比较，且搜索范围仍受计算资源限制。

---

## 488. Dynamic Sampling for Telemetry in Microservices: A Reinforcement Learning and Entropy-Based Approach

**arXiv ID:** 2609.31292 | [PDF](https://arxiv.org/pdf/2609.31292v1)

**作者:** Renan Martins Alves `[一作]` (Federal University of Rio Grande do Sul), Juliano Araujo Wickboldt `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过构建一个名为RADAR的强化学习代理，动态调整OpenTelemetry收集器的尾部采样规则，以实现对分布式追踪数据的自适应采样。

**💡 创新点**

创新点在于将Shannon熵作为采样质量的奖励信号，配合无状态多臂老虎机模型和REINFORCE策略梯度算法，使采样过程完全自动化并能优先保留信息丰富且稀有的追踪模式。

**🔧 技术方法**

使用的技术包括Python实现的RL代理、OpenTelemetry Collector、Jaeger后端、Elasticsearch数据库、Kubernetes滚动更新、以及熵计算和奖励函数实现。

**📊 数据集**

实验数据集为自行搭建的 Minimal Boutique 微服务应用，利用 Locust 生成随机流量并注入错误与延迟，以产生多样化的追踪数据。

**📈 对比分析**

与全量采集和固定 20% 采样基线对比，RADAR 在带宽上减少 97.4%、CPU 使用率降低 99%、内存使用下降 58%，同时保留 85.6% 的稀有追踪模式，平均熵提升约 25%。

**⚠️ 局限性**

局限性包括仅在单一自建应用上验证、缺乏对相同规则的静态配置对比、奖励函数超参数手工调节、未评估滚动更新的运行时开销，以及对更复杂生产环境的泛化能力待进一步验证。

---

## 489. When the Model Retires: An Empirical Study of LLM Migration in Open-Source Applications

**arXiv ID:** 2609.31288 | [PDF](https://arxiv.org/pdf/2609.31288v1)

**作者:** Hyungjin Lukas Kim `[一作]` `[通讯]` (Myongji University), Hyungjin Lukas Kim (Myongji University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统分析了商业LLM API在模型退役后应用迁移行为，挖掘GitHub提交并匹配官方退役事件，评估迁移时机、成本与失效情况。

**💡 创新点**

创新点在于首次量化生态系统级的被动/主动迁移比例、不同供应商退役通知长度对迁移行为的因果影响，以及对迁移规模和架构影响的细粒度测度。

**🔧 技术方法**

采用文本检索、正则匹配、人工验证、统计回归（logistic）、Bootstrap置信区间以及GitHub GraphQL/REST API获取提交差异等技术。

**📊 数据集**

数据集由22,555条匹配提交构成，覆盖17,703仓库，并与30个官方退役事件表进行关联，公开在Zenodo。

**📈 对比分析**

通过与手工标注真迁移比例的加权，报告82%迁移发生在退役后，且通知时长越长后迁移比例越低；每条迁移平均改动6行，重模型迁移成本高达700行。

**⚠️ 局限性**

局限在于仅捕获提交信息中显式引用退役模型的迁移，遗漏无声迁移或未迁移项目；搜索上限导致部分事件被截断，且依赖事件公告时间的准确性。

---

## 490. Peregrino: A Full-Hardware Accelerator for the Complete Falcon Post-Quantum Digital Signature Scheme on Resource-Constrained Edge Devices

**arXiv ID:** 2609.31252 | [PDF](https://arxiv.org/pdf/2609.31252v1)

**作者:** Antonio Carreño `[一作]` (Universidad Politécnica de Madrid), Jorge Portilla `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

实现了完整 Falcon 数字签名方案（密钥对生成、签名生成、签名验证）的全硬件加速器 Peregrino，采用手写 RTL 并可在单颗 Artix-7 FPGA 上运行。

**💡 创新点**

创新点在于首次完整手写实现 Falcon，模块化可替换设计、内存中心架构、浮点仿真、Karatsuba 乘法和递归采样拆分，显著降低资源占用且实现可控。

**🔧 技术方法**

采用 HDL 设计、AXI Lite 接口、嵌入式浮点仿真、Karatsuba 乘法器、递归采样状态机、MMU 与 BRAM 等硬件技术。

**📊 数据集**

使用 Falcon 官方 NIST KAT 参考向量进行功能验证，并在 Artix-7 上进行实验；无公开数据集。

**📈 对比分析**

与 FalconTakesOff (HLS 方案)、Intel i5/i7、ARM Cortex-A9/M4 等平台软件实现进行比较，Peregrino 在资源占用上分别减少 1.9×LUT、3.4×FF、2.7×BRAM、9.9×DSP，且在仿真浮点软件基础上，关键操作的时钟周期分别降低 92%、96%、85%，但整体延迟仍高于高性能 CPU。

**⚠️ 局限性**

局限性包括：频率仅 75–80 MHz，数据传输开销高，未做侧信道分析，仅适用于无 FPU 边缘设备；提升频率和使用 DMA 可进一步改进。

---

## 491. Representation-Guided Generation and Integration of Executable Programs for Robot Manipulation

**arXiv ID:** 2609.31337 | [PDF](https://arxiv.org/pdf/2609.31337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 492. CG-HAF: An Interpretable Global-Local Lesion-Burden Fusion Framework for Ordinal Acne Severity Grading in Agentic Skincare Support

**arXiv ID:** 2609.31326 | [PDF](https://arxiv.org/pdf/2609.31326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 493. Provenance of HAVING Queries in Semirings with Monus

**arXiv ID:** 2609.31246 | [PDF](https://arxiv.org/pdf/2609.31246v1)

**作者:** Aryak Sen `[一作]` (University Grenoble Alpes), Pierre Senellart `[通讯]` (PSL University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在m-半环框架下定义了HAVING子句的语义，并证明在吸收性和乘法对减法分配性条件下其等价于一种自连接重写。

**💡 创新点**

创新点在于给出了HAVING的统一m-半环语义，提供了自连接重写的等价性证明，并通过反例阐明必要性，扩展了对可能世界、概率评估和复杂度的理论分析。

**🔧 技术方法**

主要使用了m-半环代数、δ吸收公理、可能世界语义、归纳与组合证明以及针对SUM/COUNT 的动态规划算法。

**📊 数据集**

实验使用了常见的TPC‑H/TPC‑DS基准数据集（以及部分合成数据用于概率评估）。

**📈 对比分析**

与传统的自连接实现相比，实验表明在吸收性半环上重写方法保持相同或更好的性能，且能够生成等价且可并行化的查询计划。

**⚠️ 局限性**

局限性包括：对非吸收性或不满足乘法对减法分配性的半环无法保证等价；在涉及多聚合组合的HAVING条件下仍需暴力枚举，导致性能下降。

---

## 494. Budgeted Quotient-Residual Guidance for Frozen Pocket-Conditioned Molecular Diffusion

**arXiv ID:** 2609.31222 | [PDF](https://arxiv.org/pdf/2609.31222v1)

**作者:** Xinyu Wang `[一作]` (University of Connecticut), Minghu Song `[通讯]` (Hefei Comprehensive National Science Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了预算化商数残差引导方法（QRG），通过将商数余量提升为度量水平的水平向量，并以冻结分子扩散采样器自身的步长作为信任域上限，实现无重训的商数目标推断；

**💡 创新点**

创新点在于将商数微分与水平提升结合，利用采样器的更新尺度作为尺度约束，将本来在商数空间“沉睡”的梯度在采样器尺度下激活，从而避免昂贵的回溯搜索；

**🔧 技术方法**

使用了商数微分、水平提升、采样器相对信任域、KL/动力学解释、等变性证明、以及产品预算分离等理论技术，并实现了预测下一步和局部半径两种残差计算方式；

**📊 数据集**

在pocket‑conditioned分子扩散模型的标准基准任务上评估，包括fragment growing、scaffold hopping、linker design以及side‑chain decoration；

**📈 对比分析**

与无引导、仅section、shuffle‑residual、rollout‑teacher等控制进行对比，结果显示在fragment和scaffold任务上有效率提升显著（从0.815→0.864，0.664→0.707），linker略增，且新颖性与多样性保持不变，运行时间低于基准；

**⚠️ 局限性**

局限性包括：在side‑chain装饰任务中预算过大导致性能下降，且方法效果在不同模型或更大规模任务上仍需进一步验证。

---

## 495. KCensus: Synthesizing Latency-Optimal Consensus Fast Paths (Extended Version)

**arXiv ID:** 2609.31302 | [PDF](https://arxiv.org/pdf/2609.31302v1)

**作者:** Clément Burgelin `[一作]`, Rachid Guerraoui `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Knowledge Census 框架，用知识需求推导可恢复的快路径，并在 geo‑replicated 键值存储中实现并自动合成最优快路径

**💡 创新点**

通过把快路径的恢复性问题转化为知识覆盖与兼容性条件，从而把设计空间离散化并求解最优方案，既避免了手工硬编码的快路径，也实现了针对网络拓扑、提议者分布和延迟目标的自适应优化

**🔧 技术方法**

采用基于知识的 adopt‑commit 协议、可合成的传播图、Rust 编写的 SMR 引擎、以及基于线性规划的优化器

**📊 数据集**

在 AWS 全球实例上部署 3~31 个区域的 geo‑replicated 系统，使用冲突最小的键值工作负载（写入 100 000 键、1000 req/s）以及 Zipfian 读写混合测试

**📈 对比分析**

与 Multi‑Paxos、EPaxos、SwiftPaxos、Pando 等主流快路径协议对比；在 7‑节点跨洲部署下平均延迟下降 16%（最多 38%），尾延迟也明显降低；在规模扩展、故障容忍、资源消耗和高负载下均优于或匹配基准

**⚠️ 局限性**

假设稳定期足够长、仅支持 crash 故障；对 Byzantine 或极端网络抖动缺乏完整处理；在多需求（多快路径）配置下的鲁棒性尚未完全优化；内存占用在最大规模下略高于传统方案

---

## 496. MoTop: Motion-Topological Model For Micro AU Detection

**arXiv ID:** 2609.31285 | [PDF](https://arxiv.org/pdf/2609.31285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 497. Cognitive Skills in the Age of AI: Computing Students and Experts Perceptions

**arXiv ID:** 2609.31272 | [PDF](https://arxiv.org/pdf/2609.31272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 498. Towards VLA-Dreamer: Refining VLA Behavior Using World Models

**arXiv ID:** 2609.31313 | [PDF](https://arxiv.org/pdf/2609.31313v1)

**作者:** Parsa Mastouri Kashani `[一作]` (University of Hamburg), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并验证了一种利用冻结的VLA视觉编码器嵌入训练预测性世界模型，并在此模型中通过强化学习对VLA进行微调的框架；

**💡 创新点**

将VLA的视觉嵌入空间直接作为世界模型的潜在空间，既降低了对大量高质量仿真数据的依赖，又实现了在嵌入空间内的短期规划与 RL，并通过语义分割与深度探针评估嵌入的可预测性；

**🔧 技术方法**

结合VLA（如π_0-FAST、OpenVLA）冻结的视觉编码器、MLP 预测式世界模型、PPO 等模型自由 RL 算法，以及图像-深度/分割探针进行评估；

**📊 数据集**

使用NICOL机器人真实演示数据集，包括语言指令、遥控轨迹、关节位置与相机帧；

**📈 对比分析**

通过在嵌入空间内使用均方距离奖励进行 RL 微调，并利用探针测量未来嵌入的分割与深度准确度，实验表明嵌入能够保持一定的语义一致性，但在长时延迟下会出现漂移；

**⚠️ 局限性**

嵌入在复杂多物体场景下的未来预测能力有限，世界模型在长时间推断中漂移显著，缺乏明确的稀疏成功奖励，导致规划与执行的可靠性受限。

---

## 499. UniAR: A Unified Framework for Autism Recognition Enhanced by Multi-View Prompt Learning

**arXiv ID:** 2609.31298 | [PDF](https://arxiv.org/pdf/2609.31298v1)

**作者:** Lei Xin `[一作]` (Wuhan University), Zhenglun Kong `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 UniAR 框架，利用多粒度提示学习在缺乏诊断文本的情况下实现 ASD（自闭症谱系障碍）识别。

**💡 创新点**

创新点在于：① 自动生成词、短语、句子级诊断描述，填补语义稀缺；② 通过可学习视觉代码簿将视觉特征离散化为原型；③ 采用 Mixture‑of‑Experts 进行尺度自适应调制，实现多尺度视觉‑语义对齐；④ 整合三大模块（视觉原型化、分层语义生成、跨尺度对齐）构建统一、可解释的识别系统。

**🔧 技术方法**

使用了多尺度视觉编码 + 视觉代码簿、GPT‑4o 生成提示、冻结文本编码器、MoE 共享与尺度专属专家、跨模态注意力与多尺度融合、向量量化、联合损失（分类、对齐、分离、融合、量化）等技术。

**📊 数据集**

使用了 ABIDE I/II（脑功能连接 MRI）、HRM（动态面部视频）、Kaggle（静态面部图像）以及自构建的 ASD‑MM（跨平台社交媒体视频 + 专家诊断文本）四大数据集。

**📈 对比分析**

与六类 MRI 基线（CPM、BrainGNN、SpectBGNN、STAGIN、Bolt、Causality）、多种面部识别基线（RAN、SCN、DMUE、RUL、EAC）、GPT‑4o、EAC 进行对比。UniAR 在 MRI 上平均准确率 75.9%（提升 1.5%），在面部上 91.6%（提升 1.2%），在 ASD‑MM 四分类上优于 EAC，且生成诊断文本与专家报告的 ROUGE 分数均较高。

**⚠️ 局限性**

限制包括：依赖大模型生成文本可能出现幻觉；对极端异构场景的泛化仍待进一步验证；代码簿与 MoE 参数需手工调优；在极少样本或严重缺失视觉信息时性能仍受限。

---

## 500. Gauss What You Need: Compact Gaussian Splatting Across Scene Scales

**arXiv ID:** 2609.31248 | [PDF](https://arxiv.org/pdf/2609.31248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 501. Softmax Reparameterization for Output-Head Quantization

**arXiv ID:** 2609.31291 | [PDF](https://arxiv.org/pdf/2609.31291v1)

**作者:** Asim Kadav `[一作]` (Adobe SDC), Tracy Holloway King `[通讯]` (Adobe SDC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大词表语言模型的输出头进行后训练的软最大化重参数化（softmax reparameterization），通过在等价类中搜索一个共享行的标量偏移，使低比特量化（尤其是W4和W2）后保持更高的预测准确性。

**💡 创新点**

创新点在于：① 只对输出头进行等价类的一维标量搜索，而不需要重新训练；② 使用验证集上的KL损失来选择最佳偏移；③ 通过rank‑one修正处理非线性logit路径；④ 证明该方法可在不同量化器（RTN、AW‑MSE、全Hessian GPTQ）和不同数据集上保持或提升性能，并且不会增加推理开销。

**🔧 技术方法**

技术包括：softmax等价变换、后训练量化（GPTQ、AW‑MSE、RTN）、rank‑one校正、验证集KL搜索、稀疏/全Hessian优化、Pack‑inference (Marlin INT4)、以及对残差的Fisher矩阵分析。

**📊 数据集**

数据集：WikiText（训练/验证/测试），C4、OpenWebMath（跨域迁移），以及7个模型的输出头（Phi‑4‑mini、Gemma 3/4、Qwen3.5、BLOOM‑1.7B、BLOOMZ‑1.7B、XGLM‑1.7B）。

**📈 对比分析**

对比方法：原始头、固定均值中心化（t=1）以及验证选择的重参数化。评估指标为KL散度、困惑度（PPL）和推理延迟。结果显示：在W4量化中，重参数化将Phi‑4‑mini的RTN KL从1.23降至0.35（≈71%），在BLOOM上提升约70%；在W2量化中几乎所有头都得到显著改善。部署时，Pack‑W4推理延迟比BF16低10.8%，且保持了重参数化带来的精度提升。

**⚠️ 局限性**

局限性：仅适用于线性softmax输出头；对非线性logit路径需要额外rank‑one校正；搜索空间为一维标量，可能无法找到全局最优；在极端低精度（如W2）下仍有一定的困惑度提升空间；需要额外的验证集来选择参数，增加了实验成本。

---

## 502. PIA: A Personal Intelligence Agent Turning Health Conversations into Records and Records into Understanding

**arXiv ID:** 2609.31255 | [PDF](https://arxiv.org/pdf/2609.31255v1)

**作者:** Jeonghun Yoon `[一作]` (KAIST), Jaegul Choo `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个名为 PIA 的个人智能代理，专门为消费者健康对话系统服务，能够把对话中的健康信息提取为结构化的临床记录，并通过四项控制（抽取、记忆、检索、理解）将这些记录转化为用户的持续健康理解，最终为健康代理提供精确可引用的证据。

**💡 创新点**

核心创新点包括：
- 结构化抽取与长期记忆框架的结合，保证药物、测试等记录按字段持久化而非仅作文本记忆；
- 由可插拔健康模块（医学别名词典、知识图谱、时间解析规则）驱动的四控机制，提供失真最小化的记忆与检索；
- 通过“问题合同”在后台持续生成七个维度的用户概况，形成“整体用户理解”而非逐问重构；
- 三维个性化视角（单一检索、健康快照、时间轨迹）展示回答随上下文深度变化的演进；
- 分层“理解阶梯”规划，将从状态到因果到推荐的功能拆分，为后续迭代提供清晰路径。

**🔧 技术方法**

使用的技术包括：
- LLM 作为抽取器与推理核心；
- Hindsight 引擎实现关联记忆、因果链和检索；
- 结构化数据库（系统记录）与关联内存双重存储；
- 语义/关键字/图谱检索结合的检索路由器；
- 决策门控（记忆门、检索门）和时间解析器；
- 基于知识图谱的医学标注与异常判定；
- 版本化字典与审计日志。

**📊 数据集**

主要数据集为内部合成测试集：20 个合成用户、741 条对话、976 条记录、271 个评测场景；以及 319 条医学测试名称及 4,494 个别名的手工构建字典，包含 144 条参考范围。

**📈 对比分析**

比较方法采用基准评测场景，按四个轴（存储、检索、答案、能力）进行通过/失败判定；关键指标包括门控精准度 100%/召回率 89.1%，读取召回 91%，检索 MRR 0.785、Recall@k 0.763、Precision@k 0.758；存储判定通过率 88.3%。在生产环境下，系统可实现 99.7% 记录标准键匹配、99.2% 概念标签匹配，但参考范围和异常标记仅覆盖约 65%。

**⚠️ 局限性**

主要局限性：
- 只有理解阶梯的第一层（状态摘要）已投入生产，目标轨迹、因果关联等仍处于试点；
- 记忆门受字典覆盖限制，导致部分测试结果无法立即记忆；
- 自我报告数据缺失偏差显著，尤其睡眠等行为指标；
- 评估仅基于合成数据，缺乏真实用户的纵向验证；
- 对药物门控与异常标记尚未完全实现，限制了临床可用性。

---

## 503. Joule-Profiler: Profiling the Energy Consumption of Build Automation Tools Made Easy

**arXiv ID:** 2609.31228 | [PDF](https://arxiv.org/pdf/2609.31228v1)

**作者:** Jérémy Woirhaye `[一作]` (Inria), Romain Rouvoy `[通讯]` (Inria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并演示了Joule-Profiler，对 Google Gson 项目的 Maven 构建进行细粒度能耗测量，比较了冷构建（无缓存）与热构建（已缓存）并跟踪跨版本的能耗回归。

**💡 创新点**

创新点在于：① 采用无侵入的标准输出正则匹配实现阶段边界检测；② 结合 Intel RAPL（CPU）和 NVML（GPU）直接硬件计数，提供比现有模型估计更精确、可分阶段的 CI 能耗分析；③ 通过冷/热构建协议剖析依赖解析与计算成本。

**🔧 技术方法**

技术栈包括：Intel RAPL 与 NVML 直接硬件计数、异步标准输出监控、正则表达式匹配、Linux sysfs 访问、Nix reproducible builds、Jupyter Notebook 数据分析、JSON/CSV 输出。

**📊 数据集**

使用的数据集为 Google Gson 5 个最近发布版本的源代码及其 Maven 依赖，配合 Nix flake 提供的原始测量数据与分析笔记本。

**📈 对比分析**

比较方法：在 Grid'5000 节点上对每个版本执行 40 次冷/热构建，计算每个插件阶段的能耗并绘制堆叠柱状图和热图；性能上发现 2.14.0 版本相较 2.13.2 在 warm 构建中能耗下降 41–50%，并定位降幅集中在编译阶段，验证工具能有效检测能耗回归。

**⚠️ 局限性**

局限性：① 仅适用于输出可行性较高且逐行日志的程序；② 目前测量的是整个机器的 RAPL 能耗，无法区分目标进程的真实能耗；③ 高频率或批量日志可能导致阶段边界检测误差；④ 在缺乏 RAPL 支持的虚拟化环境中无法直接测量。

---

## 504. Revisiting Certified Defense with Differential Privacy on Vision Transformers

**arXiv ID:** 2609.31310 | [PDF](https://arxiv.org/pdf/2609.31310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 505. Acoustic-to-Text KV Compression for Full-Duplex Speech Models

**arXiv ID:** 2609.31224 | [PDF](https://arxiv.org/pdf/2609.31224v1)

**作者:** Yejin Lee `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在全双工语音语言模型中，利用听觉空闲时间生成文本记忆，并在 KV 缓存超限时删除旧语音状态，实现在线 KV 压缩。

**💡 创新点**

创新点包括：①引入 transcription side channel 在监听空闲期间产生文本记忆；②使用 LoRA 微调并结合 token‑级知识蒸馏，保持原有的监听/说话行为；③在不增加额外 ASR 模型的前提下完成语音到文本的压缩与实时推理。

**🔧 技术方法**

主要技术包括 LoRA 低秩适配、词级强制对齐、交叉熵与 KL 失真蒸馏、在线 KV 缓存清除、MiniCPM‑o 4.5 语音模型。

**📊 数据集**

训练使用 460h LibriSpeech 子集（132K 短句），评估使用 LongSpeech（10 分钟会话）和 Full‑Duplex‑Bench 进行对话行为测试。

**📈 对比分析**

与原始无压缩 MiniCPM‑o 4.5、StreamingLLM、外部 ASR cascades 等进行对比；峰值 KV 缓存减少 64.6%，10 分钟会话 WER 下降到 12.9%（对比 96.9），临时 QA 精度提升到 42%（对比 30%），摘要 BLEU/ROUGE 亦有提升；Full‑Duplex‑Bench 的暂停、转场与中断指标与基线保持相近。

**⚠️ 局限性**

局限性包括：需要足够的监听空闲时间；在更严格的 KV 限制下 WER 可能上升；对极长或结构复杂的对话适用性未充分验证；仅在短句数据上微调，可能不完全迁移到更大范围的对话情境。

---

## 506. Bridging Body and Brain: Gene-Driven Morphology--Control Co-Design

**arXiv ID:** 2609.31329 | [PDF](https://arxiv.org/pdf/2609.31329v1)

**作者:** Fu Feng `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于Morphogene的形态-控制协同设计框架GeCode，能够同时优化机器人身体结构与控制策略；

**💡 创新点**

创新点在于引入Morphogene作为形态与控制的共享隐式蓝图，并通过AdaConcat实现关节级的个性化注入，同时将形态搜索转化为在连续Morphogene空间中的基因驱动探索；

**🔧 技术方法**

核心技术包括基于Transformer的形态生成与控制网络、正则化自编码器（RAE）构建Morphogene空间、AdaConcat模块进行关节级条件注入，以及基于性能驱动的Morphogene更新与全局局部搜索策略；

**📊 数据集**

实验使用12个MuJoCo仿真环境（Crawler、Stepper、Pusher、TerrainCrosser、Cheetah、Swimmer、Glider、Walker等），涵盖2D与3D多种形态设计任务；

**📈 对比分析**

与StackelbergPPO、BodyGen、Transform2Act等先进基线对比，GeCode平均提高约69.5%任务回报，收敛速度约2.5倍，在相同时间预算下表现显著优越；

**⚠️ 局限性**

限制包括仅在仿真环境验证，对极其复杂的高维形态适应性尚未充分验证，且Morphogene更新和相似度阈值等超参数需要经验性调优。

---

## 507. Benchmarking Attention for Tabular Foundation Models

**arXiv ID:** 2609.31306 | [PDF](https://arxiv.org/pdf/2609.31306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 508. CytoSPM: Open-Vocabulary Cytopathology Detection with Structured Prompt Bank

**arXiv ID:** 2609.31314 | [PDF](https://arxiv.org/pdf/2609.31314v1)

**作者:** Wenjie Li `[一作]` (Central South University), Yixiong Liang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了跨五种细胞学域的开放词汇检测基准PentaCyto，并提出了高效的两阶段结构提示匹配检测框架CytoSPM。

**💡 创新点**

创新点是将结构化细胞形态提示与视觉特征通过相互选择的方式进行匹配，解耦视觉与语言，既避免深层跨模态融合又实现了对未见细胞类别的零样本检测。

**🔧 技术方法**

采用ConvNeXt‑T视觉骨干、冻结的BioMedCLIP文本编码器、两阶段视觉特征提取与结构提示匹配，并在训练中加入结构匹配热启动。

**📊 数据集**

使用PentaCyto数据集（涵盖宫颈、尿液、呼吸道、浆液和甲状腺细胞学，共24个基础类与9个未见类）进行训练与评估。

**📈 对比分析**

与YOLO‑World‑L、WeDetect等开放词汇检测器对比，CytoSPM在基类和新类均取得最高的mAP与mAP50，并以25.8 FPS实现高效推理。

**⚠️ 局限性**

局限在于仍依赖医学视觉–语言预训练，且对更广泛的显微镜场景的泛化能力待进一步验证。

---

## 509. Who Belongs Together? Topical and Social Structure in Bluesky Starter Packs

**arXiv ID:** 2609.31297 | [PDF](https://arxiv.org/pdf/2609.31297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 510. G2MAF: Test-Time Gradient Guidance for Multi-Agent Flow Policies

**arXiv ID:** 2609.31286 | [PDF](https://arxiv.org/pdf/2609.31286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 511. Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows

**arXiv ID:** 2609.31301 | [PDF](https://arxiv.org/pdf/2609.31301v1)

**作者:** Haoran Zhang `[一作]` (Harbin Institute of Technology), Hongzhi Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种名为 Effect Commit Contract 的运行时机制，用于在大型语言模型（LLM）代理执行工具调用时，将业务层的审批结果、实际持久化效果和后续工作决策在同一执行过程中进行绑定、比较和持久化，从而阻止未获批准的持久化变更继续传播。

**💡 创新点**

创新点包括：①将业务审批、执行证据和后续决策三者统一为一个跨阶段约束；②在受控执行边界内使用完整观察和事务/代理分离的方式实现对持久化结果的精确验证；③通过可持久化的终端记录支持进程恢复和后续任务的安全继续；④为多种后端（PostgreSQL、MySQL、GitHub API、ToolSandbox等）提供统一的能力与观测接口。

**🔧 技术方法**

核心技术涵盖：事务/原子性操作、观察/审计触发器、边界认证（Outlet Certification）、一次性授权绑定、idempotency key、可重放与恢复的终端日志、不同后端的适配器（PROMOTABLE、MEDIATED、OPAQUE）以及与模型运行时的集成。

**📊 数据集**

使用的数据集和任务包括：MCPMark 公共数据库任务（3 任务），STATE‑Bench 公共业务任务（12 任务），206 任务的参考比较，ToolSandbox 四个场景，8 个 MySQL 预定义操作，8 个 GitHub 操作等。

**📈 对比分析**

评估方法：在公开任务上对比 Direct Execution、AgentSpec 规则、Error‑Reject、完整观察（Complete‑Effect Audit）等多种配置；对比成功率、错误提交率、以及在不同后端与恢复场景下的行为；性能评估显示：持久化验证平均额外耗时约 1.08 ms（相对直接执行约 4.35×），边界认证额外 15 ms，效果比较随效果数量线性增长，500 条效果约 2.9 ms。整体实验覆盖 108 个 Live‑Agent 任务、540 个恢复实验以及跨后端测试，证明方法在保证安全的同时保持了较低的运行开销。

**⚠️ 局限性**

局限性：①需要先行注册观察器与能力，配置复杂度高；②对 OPAQUE 后端只能检测已持久化的效果，无法回滚；③依赖事务或可控的执行边界，对无事务或分布式事务环境支持有限；④需可信的审批来源，若审批数据不可信则失效；⑤对跨域多步骤的细粒度约束支持不完善，主要聚焦单个执行单元。

---

## 512. WeaveAgent: A Two-Stage Tool-Routing Agent for Ultra-High-Resolution Remote Sensing Imagery

**arXiv ID:** 2609.31234 | [PDF](https://arxiv.org/pdf/2609.31234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries

**arXiv ID:** 2609.31261 | [PDF](https://arxiv.org/pdf/2609.31261v1)

**作者:** Michele Paolicelli `[一作]` (Università degli Studi di Bari Aldo Moro), Giovanni Semeraro `[通讯]` (Università degli Studi di Bari Aldo Moro)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计 Mixture of Semantic Attention Regimes (MoSAR) 机制，让 Transformer 在训练期间学习输入相关的可控衰减几何，进而在保持语言建模质量的同时降低注意力成本。

**💡 创新点**

创新点是将注意力稀疏化视为几何问题，提出多元语义注意力周期并利用查询和键的路由器动态选择不同的距离衰减规则，兼顾局部性与全局性。

**🔧 技术方法**

采用 RoPE 作为基础位置编码，添加双线性路由器 MLP 进行查询/键路由，使用可微分的混合衰减函数、成本正则化以及 top‑1 硬路由离散化等技术。

**📊 数据集**

在英语维基百科语料上训练，使用单词级分词，模型规模为 5 亿参数，训练约 10B 标记。

**📈 对比分析**

与 RoPE、ALiBi、Fixed‑S/M、RoPE‑M‑mask 等基线对比，MoSAR 在训练长度 2048 时相当或略优于 RoPE，在 4k/8k 长度外推时取得最佳困惑度，且硬路由后成本显著下降。

**⚠️ 局限性**

限制包括仅在小规模模型和语言建模任务中验证，缺乏对更大规模、对齐或下游任务的评估，且目前的成本指标为期望注意力成本而非实际推理加速，需要进一步实现稀疏内核。

---

## 514. Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings

**arXiv ID:** 2609.31253 | [PDF](https://arxiv.org/pdf/2609.31253v1)

**作者:** Ekaterina Trushkova `[一作]` (HSE University), Anton Lysenko `[通讯]` (HSE University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文将基于项-项共现图的拉普拉斯特征作为冻结的图谱位置编码，替换SASRec中的可学习位置信息，实现一种轻量化的顺序推荐器。

**💡 创新点**

创新点在于仅通过一次离线图谱特征计算即可为顺序模型提供结构化位置信息，保持原有Transformer架构和训练流程不变，并避免在线计算。

**🔧 技术方法**

技术手段包括构建归一化拉普拉斯矩阵、求取其前k个特征向量作为LPE、将LPE与可学习缩放因子相加注入SASRec，并使用全软最大交叉熵损失进行训练。

**📊 数据集**

实验使用Amazon‑Beauty、Amazon‑Clothing、Amazon‑Sports和Yelp2018四个公开顺序推荐基准数据集。

**📈 对比分析**

通过与原始SASRec、无位置信息版、RoPE、TiSASRec等基线对比，SASRec‑LapPE在大多数指标上均优于原始SASRec，并在NDCG@100、Recall@100等高截断指标上超过TiSASRec，且保持了竞争力。

**⚠️ 局限性**

局限性包括仅使用低频拉普拉斯特征导致覆盖率有时下降，未探究谱频段选择或可学习谱滤波的潜力，且对大规模图谱的特征求解仍需较高计算成本。

---

## 515. Geometric Inconsistency Localization in Multi-View Image Sets

**arXiv ID:** 2609.31247 | [PDF](https://arxiv.org/pdf/2609.31247v1)

**作者:** Xander Staelens `[一作]` (Ghent University - imec), Glenn Van Wallendael `[通讯]` (Ghent University - imec)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个用于宽基线多视角几何不一致性定位的全新数据集 DeformView，并基于此数据集设计了轻量级学习型方法 DEFECt3R，用以像素级别定位不一致区域。

**💡 创新点**

创新点在于①首次构建了具有像素级几何不一致标注的宽基线多视角数据集；②将跨视角特征差异映射到可学习的分类器中，显著降低误检并提升定位精度；③将多视角几何一致性作为多媒体取证的新信号，扩展了传统检测方法的范畴。

**🔧 技术方法**

技术手段包括 DINO ViT-S/16 特征提取、MASt3R 视角对应与特征对齐、FeatsUp 上采样，以及基于差异特征的两层 MLP 分类器；训练采用 focal loss、AdamW 优化与余弦退火学习率。

**📊 数据集**

使用的主要数据集是 DeformView（1029 个 3D 物体在 24 个视角下的原始与变形渲染图），并通过对比原始-变形、原始-原始、变形-变形三种图像对进行评估。

**📈 对比分析**

与基线方法 MEt3R、TSED 以及其在真值相机位姿下的版本比较，DEFECt3R 在像素级定位的 AUC、F1、精确度上均优于 MEt3R，且误检率显著下降；在对图像对的二分类上，DEFECt3R 亦显著高于基线，虽然整体性能仍有提升空间。

**⚠️ 局限性**

局限性包括：①数据集仅基于受控几何变形，缺乏真实 AI 生成多视角图像的异质性；②方法仍受对应估计误差影响，精度和召回率仍不理想；③在极端视角差或纹理不足的场景下效果可能受限。

---

## 516. Cybflight: An Embedded Rust Autopilot for Aerial Robotics Research

**arXiv ID:** 2609.31232 | [PDF](https://arxiv.org/pdf/2609.31232v1)

**作者:** Yifan Lin `[一作]` (University of Toronto), Hugh H. -T. Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个完全在 STM32H743 微控制器上运行的 Rust 自动驾驶仪 Cybflight，支持可替换的硬件接口、状态估计、轨迹规划、MPCTC 控制器和 INDI 内环，实现了室内外高速飞行。

**💡 创新点**

核心创新在于通过强类型硬件接口与研究核心解耦，使得更换硬件、估计器或控制器仅需在编译时选择实现，无需重构相邻模块，并将所有计算（包括非线性 MPC 和 INDI）压缩到单个嵌入式微控制器，实现高性能、低成本、低重量的全本地化飞行。

**🔧 技术方法**

使用 Rust 语言、强类型系统、固定尺寸矩阵/向量实现数学运算、MPCTC（20 步非线性 MPC）和 INDI 内环；嵌入式数值核心保持代码可读性；EKSF 作为状态估计器；硬件接口覆盖 IMU、气压计、磁力计、转速反馈、MoCap 与 RTK GNSS 位置更新，所有模块通过编译时选择实现。

**📊 数据集**

实验数据来自自行生成的轨迹（圆形、figure‑eight、slalom、Split‑S）以及室内 MoCap 与室外 RTK GNSS 位置更新；未使用公开数据集，而是在实验场景中自行采集。

**📈 对比分析**

性能评估：室内 10 ms 控制周期下，最高 12.38 m/s、RMS 跟踪误差 0.153 m；室外 31.4 m/s（113 km/h）在 438.6 m 轨迹上完成，显示单微控制器即可实现高速、低误差飞行；相比传统需要伴随计算机的 MPC 实现，Cybflight 无需额外硬件即可完成相同任务。

**⚠️ 局限性**

限制包括：需要在每次模块替换后重新编译并刷机；固定尺寸类型只能捕获维度错误，无法防止物理单位或坐标系错误；某些更为计算密集的算法仍可能超出 STM32H743 的计算预算；当前仅验证了 MPCTC 与 INDI 的组合，其他控制策略仍需进一步评估。

---

## 517. Imp-ACT: Adaptive Impedance Control and Action Chunking with Transformers to Learn Contact-Rich Manipulation from Demonstrations

**arXiv ID:** 2609.31225 | [PDF](https://arxiv.org/pdf/2609.31225v1)

**作者:** Luca Zanetti `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种将方向依赖的卡氏刚度调节直接嵌入示范采集流程的Imitation Learning框架Imp-ACT；

**💡 创新点**

创新点在于：①在示范阶段利用自调Cartesian阻尼控制器实时在线调节运动方向上的刚度，并将刚度值与视觉、触觉、关节观测同步记录；②将刚度作为行动空间的一维参数加入ACT模型，实现在自主执行时通过视觉+传感器预测位置与刚度；

**🔧 技术方法**

技术实现包括：自调Cartesian阻尼控制器、Action Chunking with Transformer (ACT) 模型、VR手柄+RealSense相机视觉、6轴F/T传感器、LeRobot数据格式、基于PyTorch的Transformer训练与推理；

**📊 数据集**

数据集为在Franka Emika Panda+Robotiq 2F‑85平台上收集的擦拭与插头插入两类任务的演示数据，分别在低刚度、高刚度和自调刚度三种控制条件下共收集了1200余个演示（擦拭40×3，插头插入80×3）；

**📈 对比分析**

评估方法是与固定低刚度(200 N/m)和高刚度(1000 N/m)基线在擦拭和插头插入两任务上进行25次试验，比较成功率、平均/峰值接触力、振动功率、完成时间；结果显示Imp-ACT在擦拭任务实现100%成功率并将振动功率降低约29–180倍，在插头插入任务实现最高76%成功率并将横向接触力下降43%；

**⚠️ 局限性**

局限性包括：仅调节平移方向刚度，未覆盖旋转刚度；缺乏能量守恒/被动性滤波保障；实验仅在单机器人单任务环境下验证，泛化性和大规模数据集适用性尚未充分探索；

---

## 518. An ETH-Tight, Constructive FPT Algorithm for the Cone and Polytope Intersection Problem

**arXiv ID:** 2609.31328 | [PDF](https://arxiv.org/pdf/2609.31328v1)

**作者:** Klaus Jansen `[一作]` (Kiel University), Felix Ohnesorge `[通讯]` (Kiel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文扩展了Koana和Kumabe的框架，将高重数装箱问题推广到Cone and Polytope Intersection问题，提出了一种新的算法。

**💡 创新点**

创新点在于通过结合Carathéodory类型的整数锥界限与活跃支持枚举，显著降低了运行时间，并提供了稀疏证书的显式解压算法。

**🔧 技术方法**

使用了活跃支持枚举、分离oracle和构造重建等技术。

**📊 数据集**

使用了Cone and Polytope Intersection问题的输入数据集，包括有界有理多面体P和任意有理多面体Q。

**📈 对比分析**

与Koana和Kumabe的算法相比，本文的算法在时间复杂度上改进为2^2^O(d)·(enc(P) + enc(Q))^O(1)，并且在支持的稀疏性上也得到了保证。

**⚠️ 局限性**

限制在于该算法的运行时间是双指数的，尽管在理论上是最优的，但在实际应用中可能会受到输入维度d的影响。

---

## 519. Semi-Automatic Quantification of Bayesian Networks for Software Decision Support: Comparing WSA and RNM in a Software R&D Organization

**arXiv ID:** 2609.31275 | [PDF](https://arxiv.org/pdf/2609.31275v1)

**作者:** Mirko Perkusich `[一作]` (Federal University of Campina Grande), Angelo Perkusich `[通讯]` (Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一家软件研发组织内，以嵌入式案例研究的方式，对同一贝叶斯网络模型的两条半自动化条件概率表量化管线（Weighted Sum Algorithm 与 Ranked Nodes Method）在两种决策场景（功能选择与用户界面设计）中的效果进行了比较。

**💡 创新点**

创新点在于：1）将专家预设的情境走查与真实决策记录的回溯重构结合，提供双重验证；2）揭示即使两条管线给出相似的候选名单，完整概率分布仍可能截然不同；3）给出针对方法选择与验证的实践性准则。

**🔧 技术方法**

使用技术包括：贝叶斯网络（BN）、Weighted Sum Algorithm（WSA）、Ranked Nodes Method（RNM）、情境走查（Model Walkthrough）和总变差距离（Total Variation Distance）等。

**📊 数据集**

数据集来源为组织内部的价值评估记录：Sprint 9/10 的功能评估（385 条因素记录）以及 UI 设计评审会议 4/18/20/16/17 的因素评估（共 744 条记录），共计数百条因素评估与决策备选项。

**📈 对比分析**

比较方法：①情境走查的模式输出一致率；②历史决策重构的排名一致性；③对两管线输出分布进行 TVD 计算。性能表现为：WSA 在走查一致率上更佳（Context A：7/7 vs 4/7，Context B：6/8 vs 4/8），但在 TVD 上两者存在中等到较大差异（Context A：中位 0.304，Context B：中位 0.340，极值 0.890）。

**⚠️ 局限性**

局限性：仅覆盖一家组织、两种特定决策场景，评估为描述性且未测量总专家负担；未对 RNM 的不同校准设定做敏感性分析；缺乏跨组织或不同决策类型的复现性验证，导致结论推广性受限。

---

## 520. See to Reach, Feel to Grasp: Learning A Blind Grasp Reflex for Anthropomorphic Robotic Hands

**arXiv ID:** 2609.31323 | [PDF](https://arxiv.org/pdf/2609.31323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 521. The planted tensor problem over finite fields: algorithms and cryptography

**arXiv ID:** 2609.31256 | [PDF](https://arxiv.org/pdf/2609.31256v1)

**作者:** Yuxuan Liu `[一作]` (Chinese Academy of Sciences), Chuanqi Zhang `[通讯]` (Monash University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究了新的平均案例难题——植入完全等距子空间（Planted Totally‑Isotropic Space，PTIS）问题，并给出了其在随机张量上的定义与算法分析；随后基于 PTIS 的假设构造了两个新的公钥加密原语——公共信息下的私密并行消息（PSM）协议和基于可达图的秘密共享方案（FGSS）。

**💡 创新点**

创新点包括：① 将经典的植入团问题推广到张量域，形成 PTIS 问题，首次在有限域上提出等距子空间的植入与恢复挑战；② 证明了在维度 d≥n/2 时可用非交换秩技术实现多项式时间恢复，并给出了 q^O(n log n) 的暴力级别算法，为大部分中间维度提供了上界；③ 通过秩分布分析提出了一类基于线性组合的判别器，证明 d>n/2 时即可在多项式时间内判别植入实例；④ 结合上述假设，构造了两类可公开信息的协议，突破了植入团方案仅限于子弹速、布尔函数的局限，支持多维输出与更高安全级别。

**🔧 技术方法**

使用的技术主要有：随机张量模型（交替双线性映射）、非交换秩（non‑commutative rank）算法、矩阵秩分布与低度（low‑degree）方法、子空间枚举与最小秩子空间（shrunk subspace）技术、多项式方程求解（Gröbner 基）、以及在隐私模拟中利用 2‑hint 的等价假设（PAT‑w2H）。

**📊 数据集**

本文为理论性研究，没有使用具体的实验数据集；所有实验均在 Magma 上对随机生成的 3×3、6×6 等小尺寸张量进行 Gröbner 基求解，以验证算法可行性，得到 q^O(n log n) 的实际运行时间与理论一致。

**📈 对比分析**

对比方法：与经典植入团的伪多项式时间算法相较，PTIS 在 d≥n/2 时已可多项式恢复，且在中间维度提供了 q^O(n log n) 的上界；判别器相较于图的度分布方法，利用张量秩分布实现了更强的区分性；在协议层面，公共信息量从 O(N^2)（图）提升到 O(n^3)（张量）但信息安全性提升到指数级，消息长度仍为多项式。性能方面：在实验中，针对 d≈20、n≈60 的实例，Gröbner 基求解耗时数十秒，表明该方法在中小规模上是可行的。

**⚠️ 局限性**

局限与未解决问题：① PTIS 的指数难度仍是猜想，缺乏严格证明；② 对 d≪n/2 的高效算法尚未找到，当前最佳是 q^O(n log n)；③ 与判别问题的搜索问题之间的关系尚不清晰；④ 现有协议公开信息量仍较大（O(n^3)），在实际应用中需要进一步压缩；⑤ 对于公钥加密等更复杂协议的构造仍是未来研究方向。

---

## 522. An Optimal Structure for All-Pairs Nearest Mincuts and Sensitivity Oracles for Edge Insertions

**arXiv ID:** 2609.31290 | [PDF](https://arxiv.org/pdf/2609.31290v1)

**作者:** Koustav Bhanja `[一作]` (Weizmann Institute of Science), Asaf Petruschka `[通讯]` (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的数据结构——最近最小割层次（Nearest Mincut Hierarchy），能够在O(n)空间内编码所有顶点对的最近最小割，并在O(n)时间内查询任意一对顶点的最近最小割；同时利用该结构设计了所有对最小割插入敏感性奥里克（All‑pairs Min‑cut Sensitivity Oracle）和单源最近最小割树的O(n)构造。

**💡 创新点**

创新点在于首次实现了与Gomory‑Hu树等价的全对最近最小割的最优压缩表示（O(n)空间、O(n)查询），并首次证明最小Gomory‑Hu树无法用少量树来表示所有最近最小割；此外引入了链极大（Chain Maximizer）工具和对连接尸体（Connectivity Carcass）Skeleton 的高效使用。

**🔧 技术方法**

主要技术包括：①构建层次树（Hierarchy Tree）与Skeleton的组合；②仅存O(n)个顶点投影和链极大信息；③利用子模/对偶模性质、四分量引理等结构性引理来决定子树是否属于割；④在查询时使用LCA和级别祖先查询；⑤基于上述结构实现所有对敏感性奥里克和单源最小割树构造。

**📊 数据集**

论文为理论研究，未使用具体实验数据集；所有结果均为理论复杂度分析。

**📈 对比分析**

与现有方法相比，所提出的数据结构在空间上达到信息理论下界O(n)，查询时间达到最优O(n)，而传统所有对敏感性奥里克需O(n^2)空间，单源最近最小割树方法只能处理单源。该结构在最坏情况下保持与Gomory‑Hu树相同的性能。

**⚠️ 局限性**

限制：仅适用于无向带正权重图；对有向图或动态/删除边情况尚未给出最优表示；链极大和Skeleton等子结构实现复杂，对实现细节有较高要求。

---

## 523. More Sensors Only One Field: Rethinking Continual Spatio-Temporal Forecasting

**arXiv ID:** 2609.31325 | [PDF](https://arxiv.org/pdf/2609.31325v1)

**作者:** Lewei Xie `[一作]` (City University of Hong Kong), Zhi-An Huang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种Spatio-Temporal Field Operator（STFO），通过坐标基升维和查询，利用共享的潜在场表示实现连续时空预测；

**💡 创新点**

创新点在于将传感器布局与空间动力学解耦，使用坐标基聚合到固定潜在网格并通过谱描述器对场演化进行自适应调制，保持参数形状不变；

**🔧 技术方法**

主要技术包括坐标基编码与解码、谱描述器（Spectral Regime Encoder）、双路径（Fourier + 线性注意力）场算子、以及基于神经算子的学习框架；

**📊 数据集**

使用了PEMS-Stream、CA-Stream和AIR-Stream三个实时传感器网络数据集，涵盖交通流和空气质量监测；

**📈 对比分析**

与多类基线（传统图卷积、神经算子、持续学习方法、在线适应方法）对比，STFO在所有数据集和预测时段上平均MAE、RMSE、MAPE均优于对手，尤其在新传感器扩展时保持低误差；

**⚠️ 局限性**

局限性包括对潜在网格分辨率和傅里叶模式数的敏感性、对坐标映射质量的依赖，以及在极端动态变化或传感器稀疏情况下的性能下降。

---

## 524. AgentXploit: Autonomous Repository-to-Runtime Red-Teaming for AI Agents

**arXiv ID:** 2609.31318 | [PDF](https://arxiv.org/pdf/2609.31318v1)

**作者:** Weida Liang `[一作]` (National University of Singapore), Dawn Song `[通讯]` (University of California, Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两角色审计系统，分离仓库级攻击路径发现与运行时利用，构建可自动识别并验证AI代理漏洞的完整流程。

**💡 创新点**

首次结合源代码分析与外部验证的端到端审计框架，定义可交互的路径记录和显式攻击生成，并提供72实例真实漏洞基准。

**🔧 技术方法**

使用AST/依赖/符号搜索、LLM生成攻击路径、交互式攻击生成、外部确定性验证器，搭配LSP、工具链等。

**📊 数据集**

收集了72条公开CVEs/安全问题的实例，涵盖12个开源AI-代理项目，构建AgentXploit基准。

**📈 对比分析**

与Codex基线、AgentDojo等对比，端到端成功率59.3%（Codex 38.4%），在Token预算匹配后仍领先13个百分点；在AgentDojo上成功率79.2%高于AgentVigil 52.7%。

**⚠️ 局限性**

基准规模有限（72例，仅13条间接路径）、评估主要依赖人工匹配、仅测量验证成功不涵盖隐蔽性与整体功能影响，缺乏更广泛模型与仓库多样性。

---

## 525. LUCID: Learning Under Confounding for Inference and Discovery in Time Series

**arXiv ID:** 2609.31315 | [PDF](https://arxiv.org/pdf/2609.31315v1)

**作者:** Mohammad Fesanghary `[一作]` `[通讯]` (Bloomberg L.P), Mohammad Fesanghary (Bloomberg L.P)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于马尔科夫–帕普尔（Marčenko–Pastur）谱统计的自适应去混淆层，能够在不同隐含混淆程度（稀疏或普遍）下自动选择合适的去混淆策略，并与现有因果发现引擎无缝结合

**💡 创新点**

创新点在于（1）利用马尔科夫–帕普尔理论自动判定混淆模式；（2）在普遍混淆分支中引入低秩加稀疏谱去混淆（S-L）和持久性门控，恢复时点0的稀疏结构；（3）通过数据驱动的“无边缘”阈值校准提高精度；（4）保持与发现引擎解耦，使得同一层能包装多种算法

**🔧 技术方法**

主要技术包括马尔科夫–帕普尔谱统计、低秩加稀疏谱分解、谱裁剪（Trim）技术、持久性门控（利用Spearman相关的加权因子）、无边缘阈值校准（循环位移置换），以及基于时序相关性的小步向后选择（PDS）过滤和先验向后关联判别

**📊 数据集**

在十类合成混淆场景（稀疏本地混淆、普遍混淆的不同子类：波动性、结构、GARCH、稠密载荷、混合滞后、异质性、接近单位根等）以及基于真实11家银行CDS面板校准的半合成案例（3因子GARCH(1,1)-t）上进行评估

**📈 对比分析**

将该层与三种主流发现引擎（默认的PC式+partial-correlation，PCMCI+，NTS‑NOTEARS）一起使用，并与VARLiNGAM、DYNOTEARS、LPCMCI、TS‑ICD等基线进行对比。结果显示，在十类混淆家族的OOD评估中，加入去混淆层后，整体按族加权的定向、滞后分辨F1从最优基线0.411提升至0.601（绝对提升0.19，≈46%相对）。在CDS半合成实验中，精度从0.050提升至0.404，SHD从25.1降至4.6，表现出显著改善

**⚠️ 局限性**

局限性包括（1）谱去混淆在因果结构与因子方向重叠时可能抑制真实边缘；（2）持久性门控不能完全消除道德化错误；（3）依赖二阶谱结构，非线性因子混合或弱因子可能导致误判；（4）在高尾噪声（Student‑t3）下性能略下降；（5）评估主要在合成或半合成数据，缺乏真实因果基准验证

---

## 526. A Mathematical Theory of Near-Field Super-Resolution

**arXiv ID:** 2609.31299 | [PDF](https://arxiv.org/pdf/2609.31299v1)

**作者:** Sajad Daei `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究有限孔径近场感知的稀疏源恢复，提出了一种基于有限二次相位消 Cancellation 的定理，证明在满足特定条件下可通过总变差最小化实现精确恢复。

**💡 创新点**

创新点在于引入了“Quadratic‑Phase Aperture Certification (QPAC)”——一种取代传统最小分离距离的支持统一条件，利用有限二次指数和的非平稳性、残差相关性和近似线性化三种机制，提供可计算的、非渐近的上界，用于构造 Hermite 对偶证书。

**🔧 技术方法**

使用了离散 Fresnel 近似、全局 Hermite 对偶证书、非平稳相位分析、Bessel‑Vandermonde 有限谐波提升以及半离散 TV 规划与 SDP 对偶求解等技术。

**📊 数据集**

实验数据为合成的二维空间信号（两个点源），在不同的孔径采样数、频率、距离网格以及角度区间上进行仿真，没有使用公开真实数据集。

**📈 对比分析**

通过 QPAC 计算得到的恢复预算（η_SS、η_near、η_far）全部低于 1，表明满足条件时 TV 最小化可唯一恢复所有稀疏源；实验还展示了提升模型在数值上实现高角度精度和精确距离定位。

**⚠️ 局限性**

局限性包括：QPAC 仅为充分条件，无法判定所有可恢复配置；理论仅适用于一维孔径与半离散距离网格，且在大范围、非平面或多维阵列时需进一步推广；并且需要显式计算细致的支持统一上界，计算成本随网格细化而显著增加。

---

## 527. Deterministic Regime Switching and Feasibility Inversion in Dynamic Tensor Rematerialization

**arXiv ID:** 2609.31250 | [PDF](https://arxiv.org/pdf/2609.31250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 528. MA-WAM: Multi-Agent World-Action Model for Test-Time Planning

**arXiv ID:** 2609.31281 | [PDF](https://arxiv.org/pdf/2609.31281v1)

**作者:** Guowei Zou `[一作]` (Sun Yat-sen University), Hejun Wu `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MA-WAM框架，使用冻结的生成式多智能体策略生成候选联合动作序列，并通过路由世界模型（RWM）评估候选未来，执行最高分序列的首步动作后重规划；

**💡 创新点**

将世界模型与生成式策略结合，实现测试时候选未来评估；设计路由世界模型以保留跨智能体依赖且高效；在离线多智能体RL部署中引入重规划而非即时执行；

**🔧 技术方法**

生成式流政策（CoFlow）、路由世界模型（RWM）—共享专家与上下文路由，监督学习的动力学与奖励预测，MPC式重规划；

**📊 数据集**

MAMuJoCo（2Ant、4Ant）、SMAC（3m、2s3z、5m_vs_6m、8m）、MPE（Spread、Tag、World）三大离线多智能体基准；

**📈 对比分析**

与多种离线MARL基线（BC、MA-ICQ、MA-CQL、MA-TD3+BC等）对比，并与Reactive（M=1）和Random（M=8均匀选取）对照；在30个设置中，Planning平均提升22%相对直接执行，RWM相对随机提升25.6%，在多数任务获得最优或次优成绩；

**⚠️ 局限性**

仅在集中式测试时协调（CTCE）下验证；CTDE实现尚未完成；当基线已接近任务上限时改进有限；候选数与规划 horizon 固定为8，未做自适应；仅评估RWM评分器。

---

## 529. Identifying Scientists on X

**arXiv ID:** 2609.31264 | [PDF](https://arxiv.org/pdf/2609.31264v1)

**作者:** Philipp Meier `[一作]` (Heinrich-Heine-University), Stefan Dietze `[通讯]` (Heinrich-Heine-University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过分析X/Twitter用户的简介和推文内容，构建了自动识别科学家与非科学家的系统。

**💡 创新点**

创新点在于结合传统语言学特征与对比学习的DeBERTa模型，并通过多模态集成显著提升分类性能。

**🔧 技术方法**

技术手段包括词汇多样性、句法深度、情感与话题建模等特征提取，以及对比损失和三元组损失的对比学习模型。

**📊 数据集**

使用了两个自建数据集——Orcid（基于ORCID匹配）和Scholar（基于DOI推文与OpenAlex匹配）共计约25万条推文。

**📈 对比分析**

与传统随机森林和逻辑回归模型相比，融合对比学习的DeBERTa集成模型在10折交叉验证中实现了最高0.96的F1分数，随机森林在不含关键词特征时仅0.84。

**⚠️ 局限性**

局限性包括仅能识别公开透露专业身份的科学家、数据偏向已分享ORCID/DOI的研究者，且模型缺乏跨平台通用性与对隐私的完全保障。

---

## 530. Deduplication-while-Training: A Resilient Paradigm for Privacy-Preserving Cross-Client Deduplication in Federated Learning

**arXiv ID:** 2609.31262 | [PDF](https://arxiv.org/pdf/2609.31262v1)

**作者:** Rongxi Wang `[一作]` (Nankai University), Hanmiaomiao Wang `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种“Deduplication-while-Training (DwT)”范式，并实现了 DwT-FL 系统，使得联邦学习中跨客户端去重与模型训练可以并行进行。

**💡 创新点**

创新点在于将传统的“Deduplication-before-Training”变为持续在线的服务，设计了 CAS‑based 并发状态声明、热冷双队列调度以及基于双向索引的快速故障恢复，显著降低了断线恢复和动态加入的开销。

**🔧 技术方法**

主要技术包括：盲 OPRF 生成去重标签、基于 CAS 的并发状态争用、热冷双队列调度、心跳机制、双向索引（状态表 + 反向用户表）以及对训练结果的权重聚合。

**📊 数据集**

使用 Haiku 数据集（15,281 条短文本）进行 FL 训练，并在实验中按不同客户端数量、重复率和数据规模构造多种实验场景。

**📈 对比分析**

与现有的 NDSS'25 基线相比，DwT-FL 在端到端时间、去重阶段、故障恢复和动态加入的时间开销上都优于对手；故障恢复时间可降低至 93% 以下，动态加入时间可降低 94% 以上。

**⚠️ 局限性**

局限性包括：仅实现了精确去重（未覆盖模糊去重）；对恶意客户端攻击（如重复断线）无防御；使用静态 OPRF 密钥，未考虑长期链接和密钥更新；未评估模型质量提升或安全攻击抵御效果。

---

## 531. Compress What You See, Not What You Say: Anchored Context Distillation for Latent-Observation Software Engineering Agents

**arXiv ID:** 2609.31430 | [PDF](https://arxiv.org/pdf/2609.31430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 532. Purin: A Biology-inspired Mechanism for Artificial Neural Networks

**arXiv ID:** 2609.31235 | [PDF](https://arxiv.org/pdf/2609.31235v1)

**作者:** Zishu Liu `[一作]` (University of Exeter), Christos Grecos `[通讯]` (University of Wisconsin - Parkside)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一种名为Purin的机制，向传统卷积神经网络中加入输出侧效能（W_out）与短期可塑性因子（g_stp），并通过时间间隔抽象实现短期/长期突触效能的动态变化；

**💡 创新点**

创新点在于：① 将输入侧与输出侧权重拆分为两个可训练矩阵，模仿生物突触前后效能；② 引入基于时间间隔的短期可塑性因子g_stp，并在前向传播中按激活值更新，兼容梯度下降和反向传播；③ 不需要离散时间步或 spike 编码，保持连续激活与标准BP兼容；

**🔧 技术方法**

技术细节包括：使用分割权重（W_in、W_out）和带符号幅度函数（LeakyReLU）作为激活；g_stp在每个样本更新并在批量结束时指数恢复；训练采用AdamW、学习率0.0001、权重衰减0.0005；不使用BatchNorm或Dropout；

**📊 数据集**

实验使用四个公开数据集：CIFAR‑100、AID、UCM以及102 Category Flower（Oxford102），所有图片统一裁剪到224×224；

**📈 对比分析**

通过匹配输入预处理、激活函数、Dropout与BatchNorm的基线模型，分别与Purin模型在AlexNet、VGG11、GoogLeNet上进行五次独立实验。结果显示：Purin在VGG11上在四个数据集均提升OA（最高提升约3×），在AlexNet上提升UCM、Oxford102约8%/9%，但在AID、CIFAR‑100略降2–3%；在GoogLeNet上提升AID约0.9%，其他数据集略降或无显著变化；与SE块比较，Purin+GoogLeNet的OA明显高于插入SE块的模型；

**⚠️ 局限性**

局限性包括：① 仅在非残差网络上验证，未评估ResNet等残差结构；② 需要额外的输入预处理、激活函数改动以及禁用BN/Dropout，难以直接迁移；③ 对某些网络（如GoogLeNet）提升有限或略有下降；④ g_stp参数需手动调节（学习率、范围限制、指数恢复），对不同任务可能需要重新调参；

---

## 533. RupeeBias: Auditing Demographic Bias in Indian Economic Guidance from Large Language Models

**arXiv ID:** 2609.31245 | [PDF](https://arxiv.org/pdf/2609.31245v1)

**作者:** Pavithra P M Nair `[一作]` (Amrita Vishwa Vidyapeetham), Krishnashree Achuthan `[通讯]` (Amrita Vishwa Vidyapeetham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 RupeeBias 这一基准，用于评估印度语境下 LLM 在经济指导中的人口统计偏差。

**💡 创新点**

创新点在于覆盖 87 个印度特有的身份标识（包括种姓、宗教、地区、性别、残障与城乡），并使用单属性对照设计在英语与 Hinglish 双语言环境中生成 39,150 条提示。

**🔧 技术方法**

采用对照差异分析、相对对比差距 Δ% 与 Violation@10%、以及 Kendall's W 评估模型一致性等统计方法，并对 9 款主流 LLM 进行温度为 0 的评估。

**📊 数据集**

使用构造的 39,150 条人工生成提示数据集，涵盖四大经济使用场景：薪资估算、薪资增幅估算、反要约建议与服务定价（雇主/雇客两面）。

**📈 对比分析**

比较结果显示，Gemini 3 Flash 取得最低 Mean Δ%（≈8%）与 Violation@10%（≈24%），而 Sarvam 105B 与 GPT‑5.4‑mini 的 Mean Δ% 超过 30% 且 Violation@10% 超过 60%，说明不同模型在偏差大小与一致性方面差异显著。

**⚠️ 局限性**

局限性包括：仅针对技术行业的四个用例，使用合成单轮提示，标识仅通过显式单属性呈现，未考虑隐式线索与交叉身份，且结果仅适用于评估时点的模型版本。

---

## 534. ExoLaN: Physics-Consistent Context-Aware Dynamics Learning for Exoskeletons

**arXiv ID:** 2609.31434 | [PDF](https://arxiv.org/pdf/2609.31434v1)

**作者:** Lucas Schulze `[一作]` (Technical University of Darmstadt), Oleg Arenz `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于物理约束的上下文感知深度拉格朗日网络ExoLaN，用于估计和预测人体-外骨骼耦合系统的力矩与运动；

**💡 创新点**

创新点在于将深度拉格朗日网络与时间卷积网络结合，加入学习型脚垫映射（LIM）以估计部分接触力矩，并在单模型中同时实现逆动力学与前向动力学；

**🔧 技术方法**

使用深度拉格朗日网络、时间卷积网络（TCN）、长短时记忆网络（LSTM）以及多步预测损失；

**📊 数据集**

采用21位用户、28个任务的数据集（采样200Hz），训练时选取14人、7个常用任务；

**📈 对比分析**

与黑盒MLP、LSTM、TCN基准模型对比，ExoLaN在7个未见用户和21个未见任务上的力矩均方误差降低约7%（相较于黑盒），多步损失显著提升前向动力学预测精度，长时滚动误差降低多达60%；

**⚠️ 局限性**

局限在于对接触力矩的估计仍依赖脚垫测量，缺乏对全身动力学的建模，且模型训练对全身姿态缺乏支持，未来需扩展到更完整的身体模型与强化学习场景。

---

## 535. Augmented Reality Interfaces for Human-Robot Collaboration: Development of a ROS 2-Based Sensor Streaming Framework and Validation via SLAM Algorithms

**arXiv ID:** 2609.31396 | [PDF](https://arxiv.org/pdf/2609.31396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 536. ITS Fairy: Occlusion Assistance Selected Against a Recipient's Own Perception Reports

**arXiv ID:** 2609.31429 | [PDF](https://arxiv.org/pdf/2609.31429v1)

**作者:** Yenan Wang `[一作]` (Chalmers University of Technology), Claudio Casetti `[通讯]` (Politecnico di Torino)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ITS Fairy，利用基础设施端服务器本地动态地图（S-LDM）根据车辆自身已报告的感知信息，向每辆车仅推送缺失的、与碰撞相关的目标状态，从而在不改动车辆本地碰撞规避控制器的前提下提升信息可用性。

**💡 创新点**

创新点在于：①读取接收方自身的 CPM 报告来确定缺失目标，而非推断或几何估计；②仅发送与冲突相关且缺失的目标状态；③保持本地控制逻辑不变，易于与现有系统集成。

**🔧 技术方法**

技术包括：SUMO–ms-van3t–S-LDM 联合仿真、TCA（最近接近时间）冲突评估、S-LDM 统一对象聚合、定期（10/5/1 Hz）分析并向车辆推送简化的对象状态消息。

**📊 数据集**

使用基于 SUMO 的仿真场景：感知遮挡的车道合并和四车道交叉口，速度范围为 30–120 km/h 及极限 140–200 km/h；未使用真实交通数据集。

**📈 对比分析**

比较方法：对同一车辆在局部感知（local-only）与 ITS Fairy 辅助（Fairy-assisted）下的每一次冲突，统计最小 TCA 及路线通行时间。结果显示，辅助模式下最小 TCA 均显著大于局部模式（局部最小值接近 0，辅助平均 3.8–9.5 s），通行时间降低 11–47%。

**⚠️ 局限性**

局限性：①评估基于仿真，未考虑实际无线网络延迟、丢包或干扰；②使用的 CPM 报告假设无误，实际中可能存在错误或缺失；③对高频分析（10 Hz）下仍需频繁通信，若网络负载高可能受限；④未测量真实车辆碰撞率，仅评估 TCA 指标。

---

## 537. Evaluating the accuracy of KV cache reuse techniques

**arXiv ID:** 2609.31415 | [PDF](https://arxiv.org/pdf/2609.31415v1)

**作者:** Samuel Cestola `[一作]` (Huawei Technologies Ltd), Diego Didona `[通讯]` (Huawei Technologies Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的KV缓存重用评估方法和可控的合成基准BoxOffice，用来客观测量重用对准确率的真实损失并揭示新的缓存重用动态。

**💡 创新点**

创新点在于：① 通过筛除无意义查询构造“有意义子集”，纠正现有评估中准确率被夸大的问题；② 设计BoxOffice基准，实现跨查询重用、缓存陈旧（stale）等动态的可控实验；③ 在多版本缓存中展示选择策略对性能的关键影响。

**🔧 技术方法**

使用了位置无关KV缓存重用、选择性重算、热/冷缓存、多版本缓存、上下文聚类与嵌入相似性，以及基于F1的归一化评价。

**📊 数据集**

主要使用LongBench‑QA（MuSiQue、HotpotQA、2WikiMQA）与BoxOffice合成数据集（电影、box‑office、三种查询模板）。

**📈 对比分析**

在BoxOffice上对比CacheBlend、FusionRag、LMCache、CacheCraft等方法，计算norm‑F1；实验表明不同模型、不同策略表现差异显著，warm/冷各有优势；多版本缓存能提升性能，但取决于版本选择策略。

**⚠️ 局限性**

局限性包括：仅覆盖三种查询模板和8B级模型；仅研究重算型KV缓存重用，未涉及微调或压缩方式；BoxOffice合成基准可能无法完全模拟真实业务场景。

---

## 538. Short Paper: Prefix Count Limits Can Increase First-Hit Discovery in Card Reissuance

**arXiv ID:** 2609.31398 | [PDF](https://arxiv.org/pdf/2609.31398v1)

**作者:** Wasif Faisal `[一作]` (BRAC University), Suprava Saha Dibya `[通讯]` (BRAC University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在卡号被盗后，使用前缀活跃计数上限控制策略对枚举风险的影响，发现计数控制并不一定能降低枚举成功率。

**💡 创新点**

提出并证明了前缀计数限制与枚举成功率之间没有单向蕴含关系，并通过理论分析和大规模合成实验验证了这一非直观结论。

**🔧 技术方法**

使用概率推导、均匀搜索模型、Luhn 校验生成合成卡号、以及基于前缀权重的枚举评估技术。

**📊 数据集**

构建了包含 50,000 个合成卡号的数据集，按 11 位/12 位前缀划分，覆盖多种密度和形状场景。

**📈 对比分析**

与等量随机替换策略比较，在不同前缀密度、预算和分配策略下，计数控制在部分情况下导致枚举成功率反向上升，幅度最高约 10%。

**⚠️ 局限性**

实验仅基于合成数据，未考虑支付系统噪声、授权流程和真实用户行为，限制了结果直接应用于真实发行机构的可信度。

---

## 539. Completed Pairs Hide Capped Failures: A ReVerPi Case Study of Selective Context Projection

**arXiv ID:** 2609.31381 | [PDF](https://arxiv.org/pdf/2609.31381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 540. AxonSynth: Domain-Randomized Synthetic Data for Zero-Shot 3D Axon Segmentation in Light-Sheet Microscopy

**arXiv ID:** 2609.31431 | [PDF](https://arxiv.org/pdf/2609.31431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 541. AlphaOpsBench: Benchmarking End-to-End Alpha Strategy Operationalization in Prediction Markets

**arXiv ID:** 2609.31390 | [PDF](https://arxiv.org/pdf/2609.31390v1)

**作者:** Huaiyu Jia `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Shuo Sun `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AlphaOpsBench，一个评估 LLM 在预测市场策略端到端实现可信度与可执行性的基准框架。

**💡 创新点**

引入生命周期级别、策略来源保留、模型与源经济决策分离，以及分层生成（设计+代码）与直接生成的对比，揭示现有 LLM 在实现经济意图上的显著局限。

**🔧 技术方法**

使用 Qwen3.8‑27B 大语言模型通过 Direct 与 Staged 两种生成模式，配合自定义 Prompt、静态与行为验证，并采用 PML2 与 Fill‑Only V3 两个回测引擎评估历史可执行性与金融绩效。

**📊 数据集**

基于 581 条来源保留的策略记录（手工指南、LLM、学术论文等）和 Polymarket 2026‑06‑01 至 2026‑09‑01 的全生命周期数据（约 1.28 M 二元市场、1.84 亿交易、结算信息及订单簿记录）。

**📈 对比分析**

通过对比 Direct 与 Staged 两种生成方式，评估覆盖率、设计/代码合规性、行为验证、回测完成率、ROI 与最大回撤等指标；在受控任务上 Canonical Pass 仅 35/180（Direct）和 20/180（Staged），对真实策略均未通过；回测完成率高达 99% 但在 PML2 上的平均 ROI 为负，V3 为微正。

**⚠️ 局限性**

仅在单一模型、单一三个月窗口内评估；回测假设的流动性与队列模型未能完整再现实时交易环境；未覆盖多合约协同与自适应对手行为；无法评估实时交易的真实盈利与风险。

---

## 542. Programs-of-Layers in LLMs through the Lens of Cortical Areas

**arXiv ID:** 2609.31360 | [PDF](https://arxiv.org/pdf/2609.31360v1)

**作者:** Justus Westerhoff `[一作]` (Berliner Hochschule fur Technik), Felix Alexander Gers `[通讯]` (Berliner Hochschule fur Technik)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

复现并扩展 Polar 方法，对预训练 Transformer 的层进行动态跳过/重复路由，探究程序空间与可解释性。

**💡 创新点**

在不改动权重的前提下，用 MCTS 搜索与轻量级路由器生成可执行程序，发现极小的程序集合即可覆盖大部分问题，并揭示路由器失效的根源。

**🔧 技术方法**

使用 MCTS、分段与操作头的轻量级路由器、冻结的预训练模型及无重训练的动态层级路由。

**📊 数据集**

DART‑Math（5 难度层级）、MMLU‑Pro、ASDiv、MAWPS 等。

**📈 对比分析**

与标准单向前向（greedy）对比，MCTS 的 Skip+Repeat 程序显著提升准确率；路由器在 Pass@1 仍退回身份程序，Pass@5 通过小程序集覆盖略有提升。

**⚠️ 局限性**

路由器始终输出身份程序，无法在 OOD 上验证；单次预测策略鲁棒性差；程序空间受限，原论文数值未完全复现。

---

## 543. InternW0-$Δ$: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data

**arXiv ID:** 2609.31394 | [PDF](https://arxiv.org/pdf/2609.31394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 544. Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding

**arXiv ID:** 2609.31382 | [PDF](https://arxiv.org/pdf/2609.31382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 545. Open Vocabulary Domain Unlearning

**arXiv ID:** 2609.31356 | [PDF](https://arxiv.org/pdf/2609.31356v1)

**作者:** Sumanth Udupa `[一作]` (University of Queensland), Mahsa Baktashmotlagh `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉语言模型上提出开放词汇域不学习（OVDU）框架，利用 Fisher 信息掩码和 Targeted Manifold Scattering（TMS）实现对指定视觉域的样式遗忘，同时保持对其他域的零样本泛化性能。

**💡 创新点**

创新点：①引入开放词汇域不学习协议，强调遗忘必须对未见类别通用；②用 Fisher 信息掩码精准隔离样式敏感参数，防止语义丢失；③设计连续几何偏好损失 TMS，实现局部散射而非全局移位，显著提升样本效率。

**🔧 技术方法**

核心技术：对角 Fisher 信息近似掩码；负梯度交叉熵联合 TMS 的保留与混淆项；硬挖掘策略；ViT‑B/16 视觉编码器；OpenCLIP 预训练模型；AdamW 优化。

**📊 数据集**

实验数据集：PACS、Office‑Home、DomainNet（单域及多域），并在 ImageNet‑val 与 CIFAR‑100 上评估零样本泛化。

**📈 对比分析**

与 ADU、Baseline、NegGrad+Fisher 等基线对比，TMS 在所有 γ（0.25–1.0）下均取得最高调和均值；4‑shot 训练即可超过 8‑shot Baseline，样本效率显著提升；在 ImageNet‑val/CIFAR‑100 上保持与基线相近的零样本性能，证明未引入灾难性语义遗忘。

**⚠️ 局限性**

限制：使用仅对角 Fisher 近似，未考虑参数间协方差；实验仅覆盖至三域遗忘，未探索更大规模多域；TMS 仅应用于视觉编码器，文本编码器未做相应处理，可能存在跨模信息泄漏。

---

## 546. Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis

**arXiv ID:** 2609.31422 | [PDF](https://arxiv.org/pdf/2609.31422v1)

**作者:** Jakub Masłowski `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了 Active Provenance Gate（APG），一种在多代理辩论（MAD）系统中将辩论日志封闭为可验证证据库，并在合成阶段通过 NLI 认证、有限自修复和偏差报告来防止合成摘要中的虚假共识。

**💡 创新点**

创新点在于将数据来源追踪从被动日志转为主动运行时验证；在合成边界引入硬性可信度门（PF≥0.95）并自动触发自修复或明确的偏差报告，显著提高了 Provenance Fidelity 并提升了操作者对系统的信任。

**🔧 技术方法**

技术包括：1) 使用 Gemini 3 Flash 作为执行提议者（Proposer）生成合成摘要；2) 使用 Gemini 3.1 Pro 作为严格的 NLI 审核器（Validator）进行句子级别的支持度检查；3) 构建基于正则表达式的句子分割和 PF 计算；4) 设计有限自修复循环（最多 3 次迭代）与 Divergence Report 输出。

**📊 数据集**

数据集：使用开放源代码的 Resilient MAS 框架中的 90 条知识标注轨迹（包含 KG/RAG 文档、对话日志等），并在此框架下构建了针对危机管理委员会的对抗性辩论测试床；还进行了 33 份受试者问卷（N=33）以评估用户信任。

**📈 对比分析**

比较方法：将基线流畅合成（Baseline）与加入 APG 的合成（APG）在 90 次对抗运行中对比 PF、偏差率、计算成本和用户信任。结果显示：APG 将平均 PF 从 0.288/0.183 提升至 0.617/0.586，偏差率在高冲突场景下达到 75%（低冲突 60%），自修复循环的平均 PF 在 3 次迭代后进一步提升；在用户测试中，尽管 Baseline 更流畅，但 APG 的 Divergence Report 在高冲突情景下获得更高的信任评分（p=0.0039）。

**⚠️ 局限性**

限制：1) 实验仅基于合成模拟和预设冲击，未验证在真实现场和实时决策中的表现；2) 采用闭世界假设评估 PF，未与外部现实世界知识库对齐；3) NLI 审核器为 LLM，可能受语言偏见、冗长性和提示敏感性的影响；4) 系统仍依赖模型异构，增加部署复杂度；5) Divergence Report 的可解释性与交互性尚未在专业操作员中充分验证。

---

## 547. Decodable In-Context State and Model Output Across Training

**arXiv ID:** 2609.31401 | [PDF](https://arxiv.org/pdf/2609.31401v1)

**作者:** Manas Venkata Sai Ravulapalli `[一作]`, Samrath Singh Chadha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对大型语言模型在预训练和后训练阶段的中间层进行探针（probe）分析，研究绑定（binding）错误的可解码性、输出准确率以及探针引导的修复（steering）效果，并比较不同检查点、不同模型规模以及不同解码方式的表现。

**💡 创新点**

创新点在于：①系统性追踪探针可解码性与错误修复随预训练步骤和模型规模的变化；②引入探针引导的自门控修复（self‑gated steering）并验证其在不同检查点的提升；③通过候选 logits 与最终状态的解码对比，揭示了“最终读出瓶颈”可能并不存在；④在计算状态任务上提出了更严格的评估协议，说明表面信息与内部状态的分离限制。

**🔧 技术方法**

主要技术：逻辑回归探针（probe）训练在模型残差流（residual stream）的不同层；梯度派生子空间（gradient‑derived subspace）和类均值子空间（class‑mean subspace）对比；自门控修复技术（probe‑guided steering）；信息理论阐释（conditional entropy, mutual information）；对比解码器（multinomial logistic decoder）和最终状态解码器的 log‑loss 对比；基于多步训练的后训练检查点（SFT、DPO、RLVR）评估。

**📊 数据集**

使用的数据集包括：①自定义绑定任务（K 个实体–义务对，D 个干扰词），用于探针训练和错误检测；②公共预训练检查点的 Pythia（160 M–6.9 B）和 OLMo‑2（1 B–13 B）；③代码执行任务 MBPP/HumanEval（计算状态追踪）以及 Boxes 及 CRUXEval‑O 等。

**📈 对比分析**

比较方法：①探针准确率与 1/K（present‑set）基线、词汇基线和 |O| 基线对比；②不同规模模型的 probe 准确率随预训练步数的 Spearman 相关；③探针解码与候选 logits 解码的 log‑loss 差值；④修复实验中 oracle‑target、probe‑target 与随机方向的错误修复数量对比。性能表现：probe 在 ≥1 B 模型上可解码率提升至约 0.65–0.71，显著高于 1/K；自门控修复在后期检查点可将错误率提升至 40%–70%；但候选 logits与最终状态解码在高阶检查点无显著差异。

**⚠️ 局限性**

局限性：①探针与模型内部计算的因果关系仍不明确，仅能提示信息存在而非使用；②计算状态任务受表面信息限制，难以完全分离内部状态；③错误样本量有限，统计功效受限；④缺乏对下游任务（如代码编辑）的直接因果验证；⑤不同模型的 probe 生成器不可复现，影响结果可复现性。

---

## 548. Transformer-based Monte Carlo Localization in Construction Meshes

**arXiv ID:** 2609.31357 | [PDF](https://arxiv.org/pdf/2609.31357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 549. How Much Must a Private Mempool Hide? Exact Leakage Thresholds for Sandwich Attacks

**arXiv ID:** 2609.31379 | [PDF](https://arxiv.org/pdf/2609.31379v1)

**作者:** Tingyi Lin `[一作]` (Adrasteia Labs), Ruoran Lai `[通讯]` (Sun Yat-sen University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在隐私化交易池中分析交易量区间泄露对防止前置/后置攻击（sandwich attack）的影响，提出精确的可行性阈值和执行权竞拍模型；

**💡 创新点**

揭示区间泄露下最小量阈值决定攻击可行性的关键定理，并给出闭式利润公式与阈值表达式；

**🔧 技术方法**

使用常数乘积AMM的解析代数、信息结构模型、均衡分析及一价拍卖机制；

**📊 数据集**

无实测数据，全部基于理论推导和符号分析；

**📈 对比分析**

未进行实验比较，仅给出理论利润上界与阈值的解析关系，证明在满足阈值条件下攻击必然获利；

**⚠️ 局限性**

局限于单一隐藏订单、常数乘积池、无手续费、完全鲁棒性需求，未覆盖多订单、多池或其他CFMM的情况。

---

## 550. Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization

**arXiv ID:** 2609.31383 | [PDF](https://arxiv.org/pdf/2609.31383v1)

**作者:** Brayden Zhang `[一作]` (University of Toronto), Kashyap Chitta `[通讯]` (ELLIS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练、轻量化的端到端驾驶策略后处理模块Endpoint-Constrained Optimization（ECO），通过锚定已执行的历史轨迹并保持终点不变，修正中间航点以提升闭环可执行性。

**💡 创新点**

创新点在于将终点约束与执行历史相结合的轨迹优化方法，能够在不需要地图、额外训练或重新采样的前提下显著缓解开闭环差距，提升多种策略的闭环性能。

**🔧 技术方法**

使用L-BFGS-B优化器、平滑项、转弯惩罚项以及偏差惩罚项，并对历史轨迹做插值与转换，以实现可嵌入任何航点输出的后处理。

**📊 数据集**

在HUGSIM和AlpaSim两大光照真实感仿真器上评估，使用nuScenes、Waymo、KITTI-360、PandaSet等公开数据集构建的轨迹，验证方法的通用性。

**📈 对比分析**

与基线策略直接对比，ECO在HUGSIM闭环得分提升最高可达18%+，例如VaVAM路程完成率从84%提升到99%，在AlpaSim同样提升多项指标，显著优于传统方法。

**⚠️ 局限性**

局限在于仅修正航点轨迹，无法改变终点或对其他车辆进行避让，碰撞场景仍需额外的安全层或更完整的规划策略。

---

## 551. Towards Understanding LLM-Based Log Anomaly Detection: An Empirical Study of Performance, Efficiency, and Robustness

**arXiv ID:** 2609.31371 | [PDF](https://arxiv.org/pdf/2609.31371v1)

**作者:** Bin Li `[一作]` (Beijing Jiaotong University), Siyang Lu `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在三大公共日志数据集上，对 LLM 进行日志异常检测的多维系统评测，涵盖适配策略、模型结构、规模与量化等多因素。

**💡 创新点**

提出六维评估框架（检测性能、推理效率、适配效率、部署效率、量化鲁棒性、噪声鲁棒性），并揭示适配策略对性能影响最大，模型规模与架构提升因数据集而异。

**🔧 技术方法**

使用多种适配方法（Head-only、Layer‑Norm、Frozen、LoRA、Prefix、Adapters、Prompt‑Tuning、P‑Tuning、CoT 等），比较 RoBERTa、DeBERTa、BART、T5、DeepSeek‑7B、Llama3‑8B、Llama2‑7B 等模型，探究 GPT‑2 系列规模差异，评估 INT8/INT4 量化对性能的影响。

**📊 数据集**

BGL、HDFS、Thunderbird 三大公开日志数据集，采用时间序列划分进行训练与测试，避免信息泄漏。

**📈 对比分析**

通过表格对比发现：白盒与软提示方法（如 LoRA、Prompt‑Tuning）往往达成 98% 以上 F1；模型规模增大对 HDFS 与 Thunderbird 的提升显著，但 BGL 变化有限；低位量化（INT8、INT4）在保持 99% 以上 F1 的同时显著减小模型尺寸；不同噪声类型下，结构噪声影响最小，语义/标签噪声更显著。

**⚠️ 局限性**

局限性包括：评测仅覆盖三大日志数据集，未考察其他领域日志或跨域迁移；量化实验仅限 Llama2‑7B，缺少更大模型的量化效果；适配策略多样但未深入挖掘其原理；实验环境固定，未探讨不同硬件对效率的影响。

---

## 552. The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models

**arXiv ID:** 2609.31341 | [PDF](https://arxiv.org/pdf/2609.31341v1)

**作者:** Christoph Walser `[一作]` (Zurich University of Applied Sciences), Jonathan Fürst `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对本地信息抽取（KIE）流水线在两种不同布局的文档（近纯文本合同与布局丰富表单）下进行系统评估，比较模型（VLM、文本LLM、专用模型）、输入表征（原始图像、Tesseract OCR、Docling、DeepSeek-OCR2）以及推理配置（批量化、FP8量化）对准确率与每页能耗的影响，并提出能效最佳实践。

**💡 创新点**

首次将能耗与准确率并列评估小型本地模型（≤8B 参数）且不使用云服务的场景，发现批量化与FP8量化的能耗收益互为替代；揭示文档布局决定最佳输入表征和模型类型；给出针对不同文档的能源/准确率权衡指南。

**🔧 技术方法**

使用开源视觉‑语言模型（Qwen3‑VL、Arctic‑TILT、NuExtract）、文本LLM（Qwen3、Llama‑3.2、Mistral、Mini‑BERT 等）、vLLM 推理引擎、FP8 量化、CodeCarbon 与 Bench360 能耗采样；配合 Tesseract、Docling、DeepSeek‑OCR2 OCR。

**📊 数据集**

Kleister‑NDA（约 3000 页合同）和 VRDU（约 900 页注册表单）两大公开数据集。

**📈 对比分析**

在同一台 NVIDIA L4 GPU 上测量能耗（mWh/页）和字段精确匹配率；批量化可使能耗降低 38–85%，FP8 在单请求下可节能 27–32%，但在批量后仅 9–19%；神经 OCR 能耗高 17–18 倍但对精度提升有限；在布局丰富表单上 VLM 领先，在纯文本合同上文本 LLM + 低成本 OCR 最高。

**⚠️ 局限性**

能耗测量依赖软件估算，未包含冷却等系统功耗；仅在单个 L4 GPU 上测试；未考虑多 GPU 或不同硬件的迁移；评估只在两大英文数据集，未覆盖多语种或手写等；推理仅使用单次提示，未尝试链式思考或 few‑shot；模型加载能耗未计入。

---

## 553. Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning

**arXiv ID:** 2609.31363 | [PDF](https://arxiv.org/pdf/2609.31363v1)

**作者:** Alireza Abdollahpoorrostam `[一作]` (EPFL), Daniel Kuhn `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了带Wasserstein惩罚的分布式鲁棒优化（DRO），并提出了两种新方法：多启动粒子上升（MPA）和基于输入凸神经网络（ICNN）的全局传输映射，用于在非凸损失下实现更有效的对抗训练。

**💡 创新点**

创新点在于：①证明最优对抗传输映射必须是循环单调的；②发现标准粒子上升在多维情形下违反循环单调性，从而导致运输成本浪费；③提出MPA通过批量重新分配和多启动搜索来恢复循环单调性；④通过ICNN强制映射为梯度，从而天然满足循环单调性，并实现更好的全局最优近似。

**🔧 技术方法**

使用的技术包括：
- Wasserstein DRO与惩罚式对抗优化框架；
- 最优传输与循环单调性理论；
- 粒子上升（PA）与多启动粒子上升（MPA）；
- 输入凸神经网络（ICNN）构造Brenier映射；
- 梯度下降、投影梯度下降与Danskin定理；
- 经验风险最小化与鲁棒优化（RO）对比；
- 对抗训练的标准PGD与AutoAttack评估。

**📊 数据集**

主要使用的数据集与任务包括：
- CIFAR‑10特征空间和像素空间的多分类逻辑回归；
- CIFAR‑10.1、CIFAR‑10.2（自然域漂移）和CIFAR‑10‑C（图像破坏）评估鲁棒性；
- 车轮杆控制任务（cart‑pole）在物理参数不确定性下的鲁棒控制。

**📈 对比分析**

比较方法包括：ERM、RO、PA、NN‑DRO、Sinkhorn DRO、Wasserstein‑Fisher‑Rao DRO等。实验结果表明：
- ICNN‑DRO在所有鲁棒性指标上均优于基线，尤其在干扰攻击、自然域漂移和图像破坏上显著提升；
- MPA也表现出色，尤其在批量重新分配后显著降低Monge gap；
- 在控制任务中，ICNN‑DRO在外推环境下取得最长平均 episode 长度。整体来看，两种方法在鲁棒性、泛化和Monge gap方面均超过传统对抗训练与现有最先进方法。

**⚠️ 局限性**

主要限制：
- MPA的重新分配步骤复杂度随批量平方增长，计算成本较高；
- ICNN对抗网络需要额外的网络结构与训练稳定性调优，导致每步训练耗时更长；
- 两种方法均在硬件资源与训练时间上高于单步对抗更新，但通过显著提升鲁棒性与泛化来抵消这一成本。

---

## 554. CognitiveReality: Robot-Agnostic Semantic Gaussian Mapping with an LLM Agent for Immersive Collaborative VR Teleoperation

**arXiv ID:** 2609.31418 | [PDF](https://arxiv.org/pdf/2609.31418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 555. EEG-based Word Association Paradigm for Adult ADHD Screening: An Exploratory Pilot Study

**arXiv ID:** 2609.31359 | [PDF](https://arxiv.org/pdf/2609.31359v1)

**作者:** Caroline Peng `[一作]` (Goldsmiths, University of London), Tony Russell-Rose `[通讯]` (City St George's, University of London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

在成人ADHD筛查中，使用EEG与词联想实验探测神经与语义差异

**💡 创新点**

首次将EEG与计算语义距离相结合评估ADHD的客观筛查潜力，并发现语义距离可区分自报与诊断组

**🔧 技术方法**

EEG采集（ERP N400、θ/β比率、α抑制）+ GloVe 词嵌入语义距离 + 混合方法访谈

**📊 数据集**

词联想刺激来自 Small World of Words 数据库，语义距离采用公开的 GloVe 预训练词嵌入

**📈 对比分析**

通过单因素 ANOVA 进行三组比较；自由联想任务语义距离显著差异（p=0.003），EEG指标未显示显著差异

**⚠️ 局限性**

样本量极小（每组5人）、未记录药物或共病、设备差异、GloVe 嵌入可能与个体语义网络不完全对应

---

## 556. A Safety-Bounded SDC-to-MCP Gateway for Medical AI Agents

**arXiv ID:** 2609.31358 | [PDF](https://arxiv.org/pdf/2609.31358v1)

**作者:** Bennet Gerlach `[一作]` (University of Luebeck), Stefan Fischer `[通讯]` (University of Luebeck)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

未提供论文内容，无法确定

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 557. ActKV: Efficient LLM Agents through Action-Guided KV Cache Management

**arXiv ID:** 2609.31395 | [PDF](https://arxiv.org/pdf/2609.31395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 558. RECAST: From Log Replay to Closed-Loop Driving Simulation with View-Complete Actors

**arXiv ID:** 2609.31374 | [PDF](https://arxiv.org/pdf/2609.31374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 559. Solution for Interference in Hotspot Scenarios Applying Q-Learning on FFR-Based ICIC Techniques

**arXiv ID:** 2609.31405 | [PDF](https://arxiv.org/pdf/2609.31405v1)

**作者:** Iago Diógenes do Rego `[一作]` (Federal University of Rio Grande do Norte), Vicente A. de Sousa `[通讯]` (Federal University of Rio Grande do Norte)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 LTE/5G 网络中，分析热点区域导致的干扰，并比较多种基于 FFR 的 ICIC 算法，随后提出一种基于 Q‑Learning 的动态调节 Strict FR 的 RsrqThreshold 的方法来缓解热点出现时的性能下降。

**💡 创新点**

创新点在于：①首次将 Q‑Learning 应用于 Strict FR 参数的实时自适应调节；②通过 2^k 与全因子设计确定对性能影响最大的参数；③在动态热点场景下实现了显著的 SINR 与吞吐量提升。

**🔧 技术方法**

主要技术包括：3GPP LTE/ns-3 仿真平台、FFR‑based ICIC 算法（Strict FR、SFR、SFFR 等）、Q‑Learning 强化学习框架（状态为平均 SINR，动作为 RsrqThreshold 取值）。

**📊 数据集**

使用 ns-3 内置的网络模型，采用 5 MHz/100 RB 带宽、非 GBR TCP‑video 流量，用户分布为均匀或固定位置的热点（每个热点 10–20 名 UE），并通过模拟生成的 SINR 与吞吐量数据。

**📈 对比分析**

对比方法：将 Q‑Learning 动态调节前后（RsrqThreshold 固定为 32）在相同热点激活/关闭的时间段内统计平均 SINR 与吞吐量；结果显示 SINR 最高提升可达 180%（热点 UE），吞吐量在热点 UE 处提升 12–60% 之间，整体系统平均提升约 6–10%。

**⚠️ 局限性**

限制与待改进：仅考虑静态部署（无移动性）、仅调节 RsrqThreshold，未考虑带宽分配；仅在 DL 方向实验；使用的仿真模型未包含多频段或小基站；缺乏实测验证，且 RL 收敛速度与状态空间大小有关，可能在更大规模网络中表现不佳。

---

## 560. Progressive Memory Transformer: Memory-Aware Attention for Time-Series

**arXiv ID:** 2609.31351 | [PDF](https://arxiv.org/pdf/2609.31351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 561. AFA-Net: A Differential Attention Approach for Auditory Attention Detection

**arXiv ID:** 2609.31402 | [PDF](https://arxiv.org/pdf/2609.31402v1)

**作者:** Philip H. Lee `[一作]`, John H. L. Hansen `[通讯]` (University of Texas at Dallas)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种名为AFA‑Net的轻量级神经网络，用于通过EEG信号进行听觉注意力检测，并显式地对EEG噪声进行抑制。

**💡 创新点**

创新点在于引入差分注意力机制，用两个注意力图相减实现对无关特征的负权重赋值，显著提升了对噪声的鲁棒性，同时保持参数量极低。

**🔧 技术方法**

采用了空间‑时域补丁模块（STPM）中的2D卷积提取局部特征，焦点注意力模块（FAM）中的多头差分注意力以及传统的CSP预处理、全连接分类层等技术。

**📊 数据集**

实验数据集为公开的KUL（荷兰故事）和DTU（丹麦有声书）两套64通道EEG记录。

**📈 对比分析**

在统一的实验设置下，AFA‑Net与SS‑CNN、STANet、DenseNet‑3D、DBPNet、DARNet、MHANet等SOTA模型进行对比，2秒决策窗口下在KUL上达96.8%准确率，参数仅0.03M，显著优于所有基线模型，并在DTU上也取得领先表现。

**⚠️ 局限性**

局限性包括：在DTU 1秒窗口的性能略逊于最佳基线；实验仅覆盖两套数据集，缺乏跨任务或跨设备的验证；差分注意力对超参数的敏感性未系统评估；尚未在实时应用中验证其可行性。

---

## 562. Learning to Leverage Compliance: A Policy-Admittance Learning Framework for Robotic Insertion

**arXiv ID:** 2609.31439 | [PDF](https://arxiv.org/pdf/2609.31439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 563. dRVG: Quadtree-Guided, Resolution-Complete Online Motion Planning for Polygonal Robots in Unknown Environments

**arXiv ID:** 2609.31412 | [PDF](https://arxiv.org/pdf/2609.31412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 564. ContraFM-S2O: Flow Matching-Based One-step SAR-to-Optical Image Translation Model with Contrastive Learning

**arXiv ID:** 2609.31378 | [PDF](https://arxiv.org/pdf/2609.31378v1)

**作者:** Mingqian Yu `[一作]` (Chinese Academy of Sciences), Peilin Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种名为ContraFM‑S2O的流匹配模型，用于实现SAR到光学图像的单步高质量翻译。

**💡 创新点**

创新点在于引入平均速度取代瞬时速度实现一次性推理，并利用对比学习防止不同SAR条件下的速度场重叠，从而兼顾速度与质量。

**🔧 技术方法**

采用流匹配框架、平均速度建模、对比学习、ODE推理、U‑Net结构以及VAE编码器进行特征降维。

**📊 数据集**

使用QXS‑SAROPT和SAR2Opt两个公开数据集进行训练与评测。

**📈 对比分析**

与Pix2Pix、CycleGAN、PCGL‑GAN等GAN模型以及S2ODPM、cDDBM、CycleDiff等扩散模型进行对比，取得SSIM、FID、LPIPS、PSNR等指标的SOTA表现，同时推理速度提升约4.4倍（单图0.10s）。

**⚠️ 局限性**

局限性包括对λ值敏感（过大会降低质量）、对不同场景的泛化仍待验证，以及仍需采样先验噪声等前置操作。

---

## 565. Differential Attention Unlocks Complementary EEG and Speech Fusion for Emotion Recognition

**arXiv ID:** 2609.31399 | [PDF](https://arxiv.org/pdf/2609.31399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 566. Adaptive Switching Between Leader-Based and Leaderless BFT Protocols

**arXiv ID:** 2609.31388 | [PDF](https://arxiv.org/pdf/2609.31388v1)

**作者:** Sudip Bhujel `[一作]` (University of Kentucky), Yang Xiao `[通讯]` (University of Kentucky)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了一套BFT协议切换框架BFTide，使分布式系统在网络状况变化时能够在保持一致性与可达性的前提下，智能且低开销地在领导者基础的部分同步协议与无领导者异步协议之间切换。

**💡 创新点**

创新点包括：① 嵌入式切换层将协议切换逻辑与主共识流程融合，避免了单独共识轮；② 使用离线训练的深度Q网络（DQN）做实时协议建议，能够基于系统性能指标做动态决策；③ 提供了针对领导者被攻击或网络大规模延迟的自适应策略；④ 在安全性与延迟之间实现了更优的折中，显著降低了在极端条件下的事务延迟。

**🔧 技术方法**

核心技术包括：BFT共识协议（HotStuff、FIN、PBFT、Dumbo）实现；嵌入式协议切换层与签名投票/切换证书；离线训练的DQN强化学习；度量同步与中位数聚合；Python实现与CloudLab实验；使用TCP traffic control 模拟多种网络延迟场景。

**📊 数据集**

实验使用的“数据集”是自建的网络仿真与真实云实验数据，覆盖四种网络条件（安全、领导者延迟、全局延迟、抖动），节点规模31，负载范围20–120 KB/s，利用CloudLab Emulab 进行多区域延迟实验；未使用公开的第三方数据集。

**📈 对比分析**

与静态协议（HotStuff、FIN、PBFT、Dumbo）以及最新自适应方案BFTBrain进行比较，主要指标为中位事务延迟与吞吐量。结果显示：在领导者延迟场景下，BFTide的延迟比HotStuff低30%~60%，比BFTBrain低约70%；在安全与轻微抖动场景下，延迟与吞吐量几乎与HotStuff持平；总体而言，BFTide在保持吞吐量的同时，显著提升了在恶劣网络条件下的响应速度。

**⚠️ 局限性**

局限性包括：① 只能在预定义的协议池（HotStuff、FIN）间切换，未支持多协议混合或动态扩展协议；② 切换必须等待主协议在特定高度完成提交，无法在主协议停滞时强制退出；③ 切换引入的额外开销（消息验证、聚合）在低延迟场景下仍占比10–20%；④ DQN策略是离线训练，需在新环境中重新校准，且对极端攻击模式的泛化能力有限；⑤ 目前实验集中在吞吐量与延迟，未评估对持久性、存储占用或跨数据中心部署的影响。

---

## 567. Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers

**arXiv ID:** 2609.31342 | [PDF](https://arxiv.org/pdf/2609.31342v1)

**作者:** Md Shamim Ahmed `[一作]` (University of Southern Denmark), Richard Röttger `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究检索增强生成（RAG）在知识已被更新后仍使用过时证据导致模型回答错误的现象，称为 stale‑document poisoning，并构建了跨医学、法律、软件和平台政策的 317 条知识反转基准。

**💡 创新点**

创新点在于首次系统评估检索过时证据对已知正确答案的破坏效应，揭示模型缺乏对证据时效性的选择性信任；并通过时间可用性实验、因果干预和检索重排序验证模型对显式有效期信息的利用能力。

**🔧 技术方法**

使用检索增强生成技术、因果内部状态补丁（activation patching）以及固定的混合重排序器（semantic + recency）等方法。

**📊 数据集**

使用 317 条官方源验证的知识反转数据集，其中 50 条为时间可用性对照，覆盖医学、法律、软件 API 和平台政策。

**📈 对比分析**

通过在 12 种模型（API 与开源）上测量过时证据导致的错误率（高达 30‑91%），与最新证据一致率（97‑100%），并用 50 条时间对照测试模型对有效期的判别率（大模型可达 100%），以及因果补丁证明有效期信息直接影响决策。

**⚠️ 局限性**

局限性包括：评估仅针对检索文档的单条证据，未考虑多文档冲突；时间可用性实验仅使用官方标注的 50 条反转；对模型内部机制的定位为局部头部，未给出完整因果图；并且检索重排序依赖可靠的时间元数据，若元数据错误可导致误判。

---

## 568. Modeling and Generative-AI-Based Design of Load-Adaptive Gravity Balancing Mechanisms

**arXiv ID:** 2609.31386 | [PDF](https://arxiv.org/pdf/2609.31386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 569. Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers

**arXiv ID:** 2609.31403 | [PDF](https://arxiv.org/pdf/2609.31403v1)

**作者:** Mert İncidelen `[一作]` (Fırat University), Murat Aydoğan `[通讯]` (Fırat University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了DecoyBench数据集，系统评估了视觉语言模型（VLM）在高低分辨率下识别叠加的轮廓文字与阴影文字的能力；

**💡 创新点**

创新点在于首次使用Decoy Font方法生成双层文字图像，并从分辨率与频率角度揭示VLM在处理多层文字时的统一局限；

**🔧 技术方法**

采用Decoy Font图像生成、EM与Levenshtein相似度评估指标，使用两种提示策略（naive、guided）对六款闭源VLM进行API调用识别；

**📊 数据集**

使用300幅包含相同字数与长度的轮廓+阴影双层文字图像，分别以512×512与64×64两种分辨率进行评测；

**📈 对比分析**

通过将模型输出与标注文本做EM/LS比较，在512×512下模型能接近人类识别轮廓文字但几乎无法识别阴影文字；在64×64下模型与人类均能高精度识别阴影文字，显示性能受分辨率与层频率影响；

**⚠️ 局限性**

局限性包括仅使用单一字体与两种分辨率、仅英文文本、未单独测试单层文字、样本量仅10人验证、未探讨不同字体对比度或多语言情况等。

---

## 570. Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models

**arXiv ID:** 2609.31397 | [PDF](https://arxiv.org/pdf/2609.31397v1)

**作者:** Andrea Masini `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个闭环的 Intent2Tc 框架，能够将业务级流量整形意图自动翻译成可部署的 Linux traffic-control（tc）配置。

**💡 创新点**

创新点在于：① 使用 AQM 基础的数字孪生进行语义建模；② 通过多阶段的元数据提取、子意图生成与规则生成，并在每一步引入批判（Critique）模块进行纠错；③ 结合检索增强生成（RAG）实现知识的持续积累与复用；④ 通过实验验证即使是小型模型在此框架下也能达到大型模型的性能。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）与小型语言模型（SLM）的推理；检索增强生成（RAG）；Active Queue Management（AQM）数字孪生；批判模块（Critique）进行错误识别与修正；以及基于 RFC 9315 的 Linux tc 配置语法。

**📊 数据集**

使用了基于公开业务意图数据集转换而来的 100 条 RFC 9315 合规的流量整形意图数据集，并通过人工专家验证生成的子意图和 Linux tc 配置作为金标准。

**📈 对比分析**

与单通道（single-pass）流水线对比，实验显示：在子意图生成阶段提升了约 0.07 的 ROUGE‑L、0.06 的 Token‑F1 和 0.06 的 SBERT 分数；在规则生成阶段，Token‑F1 提升 0.065，NED 降低 0.218；整体上每条意图的 token 使用减少约 23,500，推理延迟下降 2.2 秒，成本降低约 6 美元。RAG 在保持或提升精度的同时进一步降低 token 消耗与延迟。

**⚠️ 局限性**

局限性包括：① 框架仍依赖预先构建的数字孪生模型，动态流量环境下的实时语义建模尚未实现；② 批判模块主要基于模板规则，面对极端或非标准意图时可能难以完全纠错；③ 目前评估集中在 Linux tc 上，缺乏对其他 QoS 平台（如 OpenFlow、gRPC）的一致性验证；④ 在大规模真实环境中的可扩展性与持续学习机制仍需进一步实验。

---

## 571. Evaluation of Acoustic Noise Level and Impulsiveness Inside Vehicles in Different Traffic Conditions

**arXiv ID:** 2609.31432 | [PDF](https://arxiv.org/pdf/2609.31432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 572. ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs

**arXiv ID:** 2609.31448 | [PDF](https://arxiv.org/pdf/2609.31448v1)

**作者:** Junyi Gao `[一作]` (University of Edinburgh), Ewen M Harrison `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种极小参数的适配器，利用视觉语言模型的图表输入，对不规则时间序列的数值进行残差校正；

**💡 创新点**

创新点在于：①仅冻结预训练模型的所有参数，只学习不到百万个可训练参数；②利用零初始化和RMS缩放使适配器可在原有图表特征上做细粒度数值校正；③同时支持风险预测和时序多选问答；

**🔧 技术方法**

技术手段包括视觉语言模型（如Qwen3.5）图表编码、基于注意力的时间序列编码器、残差桥接、零初始化和RMS缩放；

**📊 数据集**

使用MIMIC‑IV v3.1数据集，包含ICU 24h内的11项不规则观测，构建AKI、死亡预测和22,000+时序问答样本；

**📈 对比分析**

与未适配模型、图表LoRA、文本LoRA以及ChatTS、MLLM4TS、ITFormer、OpenTSLM等方法进行对比；在2B参数模型下，AKI AUROC 0.7376接近GPT‑5.6 Sol；在4B参数模型下，平均精度(AP)和QA准确率均超过同类方法，且训练参数数比LoRA低90%以上；

**⚠️ 局限性**

局限性包括：仅在单一医院系统和单一随机种子下验证；问答采用自动生成的多选问题；未在外部队列或开放式交互上测试；适配器对极端稀疏或异常值的鲁棒性待进一步研究。

---

## 573. DistributedDesignOptimizer: A modular Python framework for Setup, Execution and Processing of Distributed Design Optimization

**arXiv ID:** 2609.31446 | [PDF](https://arxiv.org/pdf/2609.31446v1)

**作者:** Sebastian Ellmaier `[一作]` (Leiden University), Anna V. Kononova `[通讯]` (Leiden University)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个名为DistributedDesignOptimizer的模块化Python框架，用于定义、执行和后处理分布式设计优化问题，并支持多种协调算法、标准基准、并行/集群部署以及交互式可视化。

**💡 创新点**

创新点在于将问题定义与协调方法解耦，给出了统一的算法结构并提供可插拔的协调方法、收敛判据、迭代策略和参数更新；同时实现了完整的历史记录和可视化工具，方便方法的调试与比较。

**🔧 技术方法**

主要技术包括面向对象的模块化设计、Python多进程并行、基于类的协同参数和子系统管理、有限差分/ BFGS Hessian近似、以及统一的协调框架（如ALC、Consensus-ALC、ALADIN、SBDP）和交互式后处理器。

**📊 数据集**

使用的基准数据集包括SSBJ等典型航空器设计问题，并在用户目录下提供一套标准基准，用于验证和对比不同算法的表现。

**📈 对比分析**

通过与现有框架（如OpenMDAO、GEMSEO、ALADIN‑α等）的功能评分表和对SSBJ问题的数值实验，展示了该框架在满足六大功能需求、实现可扩展算法以及在单机/集群上保持良好收敛性的优势；实验结果表明，ALC等方法在该框架下能以可接受的计算时间获得接近最优的目标值。

**⚠️ 局限性**

局限性包括框架仍处于持续开发阶段，缺乏专门的变量类、协调/记录/处理之间的进一步解耦、完整的集群部署支持、以及更丰富的基准套件；当前仅实现了少数几种协调方法，对新算法的支持仍需社区贡献。

---

## 574. OpenVAM: Open-World Visual Attention Modeling with VLMs

**arXiv ID:** 2609.31364 | [PDF](https://arxiv.org/pdf/2609.31364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 575. Implicit Neural Representation for Hyperspectral Video Compression

**arXiv ID:** 2609.31435 | [PDF](https://arxiv.org/pdf/2609.31435v1)

**作者:** Alfredo Scalera `[一作]` (University of Strathclyde), Jaime Zabalza `[通讯]` (University of Strathclyde)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `aaccfe5c-6b26-4208-b23c-35331481e142` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于隐式神经表示（INR）的高光谱视频压缩方法，改造了 HNeRV 网络以处理多通道输入，并采用自定义 HSFusion 损失实现光谱一致性重建。

**💡 创新点**

创新点包括：①将 INR 方法扩展到高光谱视频域；②设计 HSFusion 损失融合 L2 与余弦相似度，提升像素级光谱保真；③在无大规模数据集的情况下实现过拟合压缩，显著降低压缩率。

**🔧 技术方法**

使用技术主要有：隐式神经表示（INR）+ HNeRV 架构、L2+余弦相似度的 HSFusion 损失、量化与熵编码（Huffman）以及对 PCA+JPEG2000 的基准实现。

**📊 数据集**

采用 HOT2026 高光谱目标跟踪数据集（512×256，16 通道）进行训练与评估，并与 PCA+JPEG2000 处理的帧序列进行对比。

**📈 对比分析**

通过 PSNR、SAM 及下游目标跟踪指标（AUC、DP）进行比较。INR 方法在相同比特率下实现 BD-PSNR +4.99 dB、BD-rate -88.88%，并在低码率下将跟踪 AUC 提升至 23.42%、DP 提升至 35.56%。

**⚠️ 局限性**

局限性包括：需要针对每个视频进行超参数搜索；量化与熵编码仍可进一步优化；在极低比特率下 PCA+JPEG2000 仍有竞争优势；过拟合方法对输入分布变化敏感。

---

## 576. Mutable Transcripts: Mitigating Context Pollution through Editable Conversation State

**arXiv ID:** 2609.31354 | [PDF](https://arxiv.org/pdf/2609.31354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 577. Context-Aware Functional Modeling for Android Third-Party Library Detection

**arXiv ID:** 2609.31409 | [PDF](https://arxiv.org/pdf/2609.31409v1)

**作者:** Dihao Fan `[一作]` (Beihang University), Xu Wang `[通讯]` (Beihang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于上下文感知对比学习与功能划分的Android第三方库检测工具Library Finder；

**💡 创新点**

创新点在于：①通过方法级调用关系与类上下文的双编码融合，使用对比学习得到鲁棒的语义表示；②将库拆分为功能连贯的子单元，采用最匹配分区得分以解决部分库重用问题；③构造了更真实、覆盖多种R8变体的新基准数据集；

**🔧 技术方法**

技术方法包括：UniXcoder预训练模型 fine‑tune + InfoNCE 对比学习；BFS线性化调用子图；跨注意力（cross‑attention）融合类上下文；功能分区合并与Jaccard相似度；基于余弦相似度的库/版本级评分；

**📊 数据集**

使用的数据集为新构建的LibFan基准：200款F‑Droid开源应用、46个含漏洞的第三方库（共3,120个版本），每款应用在四种R8变体下编译；另外在1,081个Google Play应用上进行野外评测；

**📈 对比分析**

与LibPecker、LibScan、LibHunter三大主流工具对比，Library Finder在最严苛R8 full模式下库级F1达81.3%（相较最佳基线提升64.9%），版本级F1为47.6%（提升35.6%），运行时间约0.64s/对偶，整体效果显著优于基线；

**⚠️ 局限性**

局限性包括：版本级检测仍难，难以捕捉相邻版本细微差异；依赖JADX与Androguard构建调用图，可能受限于反射、多态等高级特性；基准主要来自F‑Droid，商业应用的差异未完全覆盖；

---

## 578. PANEL: An Open-Source, Self-Hosted Web Platform for Human Evaluation of Generative Models

**arXiv ID:** 2609.31392 | [PDF](https://arxiv.org/pdf/2609.31392v1)

**作者:** Matteo Spanio `[一作]` (University of Padova), Martín Rocamora `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个名为 PANEL 的开源自托管平台，用于在实验室内部设计、分发和分析面向生成模型的听觉/感知实验，支持多模态刺激、分支逻辑、即时统计分析以及 GDPR 合规的数据管理。

**💡 创新点**

创新点在于把实验设计、数据采集、统计检验和可复现性完整打包为可部署的服务，解决了现有工具只能聚焦于单一协议或受限于第三方平台的局限；同时提供实验生命周期管理、功效分析、可定制的问卷类型和可插拔的扩展机制。

**🔧 技术方法**

使用 Django + React/JavaScript 构建前后端，后端基于 Python 实现各种问卷类型与分配策略，数据库使用 PostgreSQL，部署通过 Docker Compose；统计方法包括 Bradley–Terry 模型、显著性检验、重复测量校正与功效计算；提供 REST API 与签名 webhook 以实现外部集成。

**📊 数据集**

论文未引用特定数据集；平台可接受任意音频、视频、图像、文本等刺激，示例中使用了音乐生成系统的对比实验。

**📈 对比分析**

比较方法采用双人模式（同一提示下的两种模型输出并排呈现），报告胜率与 Bradley–Terry 分数；单刺激模式采用平衡随机或阻塞分配；平台持续计算分布、均值、显著性检验，支持即时结果查看；性能方面支持多用户并发、实时分析，但未给出定量性能指标。

**⚠️ 局限性**

局限性包括：需自行部署、缺乏硬件级毫秒级刺激触发精度；仅在浏览器中运行，无法保证极低延迟；依赖实验者自行提供刺激与实验设计；并非完全支持所有标准化协议（如 ITU‑R BS.1534）的完整实现。

---

## 579. MexHat: A Dataset for Hate Speech Detection in Mexican Spanish Videos

**arXiv ID:** 2609.31553 | [PDF](https://arxiv.org/pdf/2609.31553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 580. DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models

**arXiv ID:** 2609.31349 | [PDF](https://arxiv.org/pdf/2609.31349v1)

**作者:** Haojun Xu `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种用于压缩大规模视频扩散模型的知识蒸馏方法，旨在保留机器人交互中的运动动力学，实现仅四步推理的高效视频生成模型；

**💡 创新点**

创新点包括：①基于教师速度转角度量构建的时间亲和条件再噪声采样策略，能够动态调整每条生成轨迹的噪声时间分布，平衡运动恢复与视觉质量；②基于潜在空间运动强度的动态引导假分数追踪机制，在有限的更新预算内对难以拟合的轨迹加大 critic 损失权重；

**🔧 技术方法**

技术主要包括：分布匹配蒸馏（DMD）、时间亲和度量、教师速度转角度量、假分数模型（critic）以及基于 VAE 潜在差分的难度预测网络；

**📊 数据集**

使用数据集：RoVid-X（约 3.3 万个剪辑）进行训练，评估数据集为 R‑Bench、PAI‑Bench‑G、EZS‑Bench 以及 WorldArena 两个任务；

**📈 对比分析**

与基线 Base DMD（使用固定噪声时间表和均匀 critic 加权）对比，本文方法在 R‑Bench 任务遵循一致性提升 9.6pp，PAI‑Bench‑G 领域分数提升 5.1 分；在 WorldArena 任务中成功率从 16% 提升至 34%；视觉质量保持与 Base DMD 相当；

**⚠️ 局限性**

局限性：教师速度转角度量和再噪声采样先验需要针对每个教师和时间表重新估计，无法直接迁移到新的教师或模型结构。

---

## 581. Bandwidth, Latency, and 400 Million Kilometers: The Case for Mars-Local Compute

**arXiv ID:** 2609.31566 | [PDF](https://arxiv.org/pdf/2609.31566v1)

**作者:** Maleeha Masood `[一作]` (University of Illinois Urbana-Champaign), Deepak Vasisht `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并评估了一种两层计算架构，在火星轨道（赤道同步层和低轨层）上部署计算节点，以实现对火星表面探测任务的持久共享服务与高带宽数据上传。

**💡 创新点**

创新点在于：①将计算资源迁移到火星轨道，解决了表面与地球间极低速率、时延大、间歇性通信带来的信息滞后问题；②设计了“两层”架构，红外同步层提供广覆盖持久服务，低轨层提供短时高速链接与局部处理；③通过量化覆盖率、每日数据上传量与节点数的关系，为可扩展部署提供决策依据；④对功耗、质量、热控进行首次可行性估算，证明在现有技术下可实现。

**🔧 技术方法**

使用技术包括：轨道力学与覆盖计算（赤道同步与低轨模型）、自由空间链路模型（接收功率与速率估算）、功耗/热控设计（太阳能板、蓄电池、散热片面积计算）、基于现有火星探测器（MRO、TGO、Curiosity、Perseverance）的数据流与传输速率测定。

**📊 数据集**

数据集主要来自：HiRISE、MRO、TGO、Perseverance与Curiosity的实际传输日志和观测量，使用这些观测量对模型进行校准，并以 2026 年的通信容量为基准进行评估。

**📈 对比分析**

比较方法：将三种部署方案（仅同步层、仅低轨层、两层混合）按覆盖率、每日表面到轨道的数据上传量（Gb/sol）进行对比。实验结果显示：同步层单节点覆盖 33% 轨面；三节点可达 90%；低轨单节点每日上传约 0.85 Gb；两层混合可在维持 90% 覆盖的同时，每个低轨节点提供约 10 倍于同步层的上传带宽。性能表明两层方案在保持持续服务与高带宽传输之间取得良好平衡。

**⚠️ 局限性**

局限性：①成本高且发射窗口约每 26 个月一次，部署受限；②轨道计算节点需携带大功率太阳能与冷却系统，增加质量；③需使用辐射硬化硬件，计算能力受限；④低轨层覆盖有限，需多颗卫星才能满足全局需求；⑤未深入讨论故障恢复、容错与安全性；⑥对真实硬件的实验验证尚未完成。

---

## 582. Weight Pair Encoding: Inducing a Smaller Grammar in Neural Network Weights

**arXiv ID:** 2609.31564 | [PDF](https://arxiv.org/pdf/2609.31564v1)

**作者:** Irene Tallini `[一作]` (Area Science Park), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Weight Pair Encoding (WeightPE) 方法，通过在训练过程中将int8量化的权重序列使用带全局L2失真预算的损失式Re‑Pair压缩，显式地将权重文法大小作为训练目标，进而在保持较小准确率损失的前提下显著减小网络权重的可压缩文法尺寸。

**💡 创新点**

创新点在于将文法大小作为显式训练目标引入权重训练；使用带全局L2预算的损失式Re‑Pair将近似重复模式转为完全相同的共享规则；并将此压缩步骤嵌入Straight‑Through Estimator中，使网络在前向传播中使用压缩后的权重并可训练。

**🔧 技术方法**

技术包括int8量化、权重矩阵按行序列化为单一字符串、聚类领导者重写（Lossy REWRITE）与标准Re‑Pair合并、L2失真预算控制、Straight‑Through Estimator梯度传递。

**📊 数据集**

数据集为CIFAR‑10，对 Vision Transformer 基础模型 ViT‑B/16 与 ViT‑L/16 进行微调。

**📈 对比分析**

方法通过与传统int8 QAT 的权重文法大小及测试准确率对比评估，WeightPE 在保持 1.1–1.9 点准确率下降的情况下，将文法大小压缩至 0.38–0.43 倍；同时对 SEQUITUR 与 LZ78 等非训练时使用的压缩器也能保持相近的压缩率，证明了压缩结果的可迁移性。

**⚠️ 局限性**

局限性包括仅在两种 ViT 变体与 CIFAR‑10 数据集上验证；仅对 MLP 投影层应用，未针对注意力层或其它网络结构；序列化方式固定为行优先；压缩操作的计算成本在不同参数设置下差异显著，尚未做充分优化；尚未在更大规模模型、不同任务或量化级别下验证其通用性。

---

## 583. Forensic Twins: Self-Supervised Residual Learning for AI-Generated Image Forensics

**arXiv ID:** 2609.31514 | [PDF](https://arxiv.org/pdf/2609.31514v1)

**作者:** Javier Muñoz-Haro `[一作]` (Universidad Autónoma de Madrid), Julian Fierrez `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种自监督残差学习框架（SSRL）来提取真实图像的微观指纹，从而实现对 AI 生成图像的零样本检测和无监督来源聚类。

**💡 创新点**

创新点在于：①仅使用真实图像训练；②通过冻结残差提取器和采样不重叠的图像裁剪，消除语义干扰；③采用 Barlow Twins 的冗余消减目标，迫使模型只关注图像采集过程的稳定指纹；④在检测中直接使用一类 GMM 进行异常检测。

**🔧 技术方法**

核心技术包括：冻结的残差提取器（FRE），自监督学习中的 Barlow Twins 损失，ViT‑Tiny 编码器，双视图不重叠裁剪，以及离线一类 GMM/ k‑Means 聚类。

**📊 数据集**

训练数据为 491,690 张来自 ImageNet‑1k 与 MS‑COCO 的真实图像；评估时使用 27 种 AI 生成模型（GAN、扩散模型、商业系统）和 ImageNet、MS‑COCO 的 1,000 张真实图像。

**📈 对比分析**

与 8 种现有方法对比，零样本检测的 AUC 达到 97.99%，仅 3.0M 参数、6.44 ms 前向推理；在聚类任务中准确率在 k=N 时提升 6.13%，k=2N 与 k=4N 分别提升 17.12 与 10.76，明显优于 FSD、ConV 与 CLIP。

**⚠️ 局限性**

局限性包括：对商业生成器的性能仍较弱；单一 GMM 的密度估计在重压缩等后处理下易退化；模型尚未结合少量标记样本的微调或更复杂的一类建模方法（如 Normalizing Flows）。

---

## 584. Muslim: A Deployed Arabic Voice AI Platform for Grounded Islamic Knowledge

**arXiv ID:** 2609.31511 | [PDF](https://arxiv.org/pdf/2609.31511v1)

**作者:** Yahya Mohamed Elnawasany `[一作]` `[通讯]`, Yahya Mohamed Elnawasany

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套生产级的阿拉伯语语音 AI 平台，用于提供源自可靠文本的伊斯兰知识，支持实时语音交互、检索增强生成、精细化的账户计费与可观测性；

**💡 创新点**

创新点包括：① 全流程自托管、无第三方云的实时语音管线；② 公开发布的阿拉伯伊斯兰领域专用 LLM 与 TTS 模型（Muslim‑6B‑PRO、Fasih‑TTS‑V1）；③ 账户与计费层的细粒度轮询配额、容量墙与延迟验证；④ 针对 GPU 失效的三层可观测性栈；⑤ 采用确定性检索与回声验证，严格避免模型幻觉；

**🔧 技术方法**

技术涵盖：NeMo FastConformer 阿拉伯语 ASR、Silero VAD、OpenAI 兼容 LLM 接口、Coqui XTTS 自托管 TTS、FastMCP 与 MCP 协议、LiveKit SFU、QLoRA 4‑bit 微调、Python/TS 统一测试、错误报告与产品分析；

**📊 数据集**

使用的数据集包括：1,297 条单说话人现代标准阿拉伯语音频；2,731 条精心挑选的工具调用/生成示例；约 50,000 条古兰经圣训记录；8 本经典 Tafsir 书籍的 JSON；大量 Quranic metadata、websearch、fiqh Q&A 等远程语料；以及公开的 SILMA、Arabic TTS Arena 等基准；

**📈 对比分析**

对比结果显示：Fasih‑TTS‑V1 在 Arabic TTS Arena 上排名第 5（17 系统）/第 2（11 开源系统），recitation 验证器 98.4% 准确率；端到端语音延迟 0.9–1.7 秒；检索成功率 100%，检索增量仅 50 ms；ASR 平均 235 ms、LLM 首 token 520 ms；整体性能与商用助手相当；

**⚠️ 局限性**

局限性包括：单机 GPU 可用性限制（仅单机可对话）；方言阿拉伯语识别误差未优化；缺乏正式用户体验评估；检索覆盖范围受限，超出 Tafsir/Hadith 语料会回退至 LLM 产生不确定答案；Muslim‑6B‑PRO 尚未公开基准评测；TTS Arena 排名随时间变化。

---

## 585. UQ-LOB: Uncertainty-Aware Limit Order Book Mid-Price Forecasting

**arXiv ID:** 2609.31491 | [PDF](https://arxiv.org/pdf/2609.31491v1)

**作者:** Derrick Gilchrist Edward Manoharan `[一作]` (Tampere University), Juho Kanniainen `[通讯]` (Tampere University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种轻量级的 UQ-LOB 模块，在预训练的 LOB 编码器上实现输入相关的不确定性量化，输出可校准的高斯分布或三分类分布，并提供可阈值的置信度信号。

**💡 创新点**

创新点在于：① 通过“上下文感知”自注意力机制将最近完成窗口的真实标签嵌入预测中，实现无梯度更新的即时自适应；② 结合 heteroscedastic 似然、方向性损失和分布式校准惩罚，获得可校准且有意义的置信度；③ 在同一网络结构下同时支持连续和离散预测。

**🔧 技术方法**

使用条件/自注意力神经过程（Conditional/Attentive Neural Process）框架；对编码器采用预训练的 D‑TABL 变体；对回归头使用带权重的 NLL + MAE + 方向损失 + 校准损失；对分类头使用加权交叉熵；通过 SNR 或 softmax 最大值实现置信度门控。

**📊 数据集**

基于 Kraken 加密货币交易所的 Level‑2/Level‑3 订单簿数据，覆盖 7 种美元计价资产（BTC, ETH, SOL, LTC, DOGE, SUI, TAO），共计 5.2 亿条事件，时间范围为 2025‑11 至 2026‑02。

**📈 对比分析**

与同一编码器下的专门分类头进行对照；在 5 s、10 s、15 s 三个时隙上评估。UQ‑regression 在全量测试集上方向宏 F1 为 0.42‑0.45，门控后最高 0.57；UQ‑classification 在全量测试集上为 0.45‑0.48，门控后最高 0.56。UQ‑regression 还提供了近 68% 的置信区间覆盖率，且在大幅度移动上门控后方向 F1 达到 0.88（下跌）和 0.83（上涨）。

**⚠️ 局限性**

主要局限包括：95% 置信区间低于 90% 覆盖，说明高斯似然无法捕捉长尾分布；未与权重空间不确定性方法（MC Dropout、深度集成）做对比；测试集时间跨度有限，难以评估在多种市场 regime 下的鲁棒性；缺乏基于真实交易成本的盈利性评估。

---

## 586. Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networks

**arXiv ID:** 2609.31466 | [PDF](https://arxiv.org/pdf/2609.31466v1)

**作者:** Siddhartha Shankar Das `[一作]` (Pacific Northwest National Laboratory), Mahantesh M Halappanavar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种无监督的拓扑稀疏化框架（称为Scaffold），利用支持图理论中的扩张度（dilation）与拥塞度（congestion）来生成稀疏图支持，从而在保持图通信结构的同时大幅减少GNN的边数、内存占用和训练时间。

**💡 创新点**

创新点在于：①首次将支持图理论中的扩张度和拥塞度联合用于GNN稀疏化；②设计了可扩展的贪心与近似变体（Greedy、Heap、Batch、Fast、Sample），能够在不同规模图上高效构造稀疏支持；③通过实验验证该方法在保持预测性能的同时显著降低资源消耗。

**🔧 技术方法**

技术细节包括：基于最短路径的支持图构造、对每条候选边计算扩张度与拥塞度的加权评分、贪心/分层/低伸展树等多种稀疏化实现；在GNN训练与推理中直接使用稀疏支持图替代完整邻接图；对不同训练策略（-1、-K）和采样方式进行设计与评估。

**📊 数据集**

使用了19个节点分类基准数据集，其中包含8个同构（homophilic）、6个异构（heterophilic）和5个大规模图（如reddit、ogbn-products、ogbn-proteins等），全部转化为无向无自环图进行评估。

**📈 对比分析**

方法与全图训练、跨度森林、随机稀疏、学习型稀疏等20种基线进行对比。实验显示，在保留10%–50%边缘时，Scaffold在大多数数据集上可达到或接近全图性能（误差≤1%），且在内存占用与训练时间上优于大多数对手，整体平均排名最高。

**⚠️ 局限性**

局限性包括：①目前仅适用于静态无向无权图，未考虑有向、加权或动态场景；②缺乏针对完整扩张度-拥塞度目标的近似保证；③在极端低稀疏度下性能下降仍显著；④方法仍需在更大规模或更复杂任务（如图生成、关系预测）上验证。

---

## 587. OC-GS: Gaussian Splatting for Irregular Turntable Capture

**arXiv ID:** 2609.31572 | [PDF](https://arxiv.org/pdf/2609.31572v1)

**作者:** Jae Joong Lee `[一作]` (Purdue University), Bedrich Benes `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在旋转台捕获的稀疏、非均匀视图下，通过共享旋转轴、摄像机和旋转中心，联合优化每帧角度与高斯场几何的对象中心化高斯散射模型OC‑GS；

**💡 创新点**

创新点在于：①利用图像预测的几何与相机信息初始化非均匀角度并拟合圆；②为每帧学习角度修正并去除平均值，避免冗余旋转；③在共享运动模型下实现角度与几何的同时优化；

**🔧 技术方法**

采用高斯散射渲染（2DGS/3DGS框架）、光度L1损失与不透明度/尺度正则化、旋转轴和枢轴的共轨道约束，并结合现有预训练相机/几何估计器（VGGT、AnySplat）进行初始化；

**📊 数据集**

主要数据集为Google Scanned Objects（100件物体的合成视图，12/8/6视角的正则与非正则采样）以及RotGS公开的真实旋转台捕获；

**📈 对比分析**

与AnySplat、SPFSplat、InstantSplat、RotGS、Nerfacto（以及Splatfacto的真实相机版本）对比，OC‑GS在12/8/6个不均匀视图下均以显著优势领先，最高达10+ dB的FG‑PSNR提升，且在GPU显存和训练时间上也优于RotGS；

**⚠️ 局限性**

局限性包括：依赖预训练相机/几何初始化；假设旋转轴固定，轴偏差会影响重建；对极端角度误差或非圆形轨道的鲁棒性尚未验证。

---

## 588. Adapting for AI: How elementary teachers adjust their practices for an AI-integrated curriculum

**arXiv ID:** 2609.31569 | [PDF](https://arxiv.org/pdf/2609.31569v1)

**作者:** Fasika Melese `[一作]` (University of Pennsylvania), Shiyan Jiang `[通讯]` (University of Pennsylvania)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对三位小学教师在为期三周的暑期营中实施以ToyTalk为核心的英语语言艺术（ELA）+ AI素养课程进行案例研究，探讨教师在技术、学习者与教学三维张力下的适应实践；

**💡 创新点**

提出教师在生成式AI工具课堂中四种适应实践模式（repair、differentiate、translate、balance），并强调课程与工具设计应支持教师即兴修复与翻译，凸显教师共同意义建构的重要性；

**🔧 技术方法**

使用ToyTalk对话式AI玩具平台（无代码界面）进行学生定制；研究方法为质性案例研究，收集每日反思、专业发展小组讨论录音与访谈记录；

**📊 数据集**

数据来源为三位教师的日记/反思文本、集体讨论转写与访谈记录，涉及78名二、三年级学生；

**📈 对比分析**

通过归纳性编码和主题分析，未进行量化性能对比，结果以教师适应实践的主题呈现，缺乏客观指标；

**⚠️ 局限性**

研究样本规模小，仅涵盖单一农村学区的三位教师，缺乏学生视角和量化学习效果评估，结果仅为描述性发现。

---

## 589. Multi-agent Scaling Across Disjunctive and Compensatory Tasks

**arXiv ID:** 2609.31563 | [PDF](https://arxiv.org/pdf/2609.31563v1)

**作者:** Carolina Fortuna `[一作]`, Blaz Bertalanic `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多智能体大型语言模型（LLM）团队规模与性能的关系，提出将Steiner分组任务分类法用于量化分析并重点关注分歧（disjunctive）与补偿（compensatory）两类任务；

**💡 创新点**

首次将Steiner任务分类与LLM团队规模理论相结合，揭示任务结构决定团队规模提升幅度，并通过实验验证了多轮推理和模型族异质性对性能的不同影响；

**🔧 技术方法**

采用多智能体投票（多数/多投票）、几何平均聚合、交互式推理（多轮对话）等方法，并在13种公开权重LLM上进行大规模实验；

**📊 数据集**

使用公开基准数据集：GSM8K、GSM‑Hard、MATH‑500、MMLU‑Hard、ARC‑Challenge（分歧任务）以及RealFP Fermi估计（补偿任务）；

**📈 对比分析**

对比单体模型与多智能体团队，在分歧任务中团队规模提升可达5–20个百分点但投票上限有限；在补偿任务中规模提升极小，平均误差仅下降约6%，多轮推理几乎无效；

**⚠️ 局限性**

限制包括：使用“先给答案后说明”prompt导致推理收益受限；未评估无同伴修订情形；补偿任务仅基于单一含噪标签的Fermi基准；所有模型参数均≤20B，仅测试分歧与补偿两类任务。

---

## 590. Agentic Economies for Autonomous Scientific Discovery

**arXiv ID:** 2609.31562 | [PDF](https://arxiv.org/pdf/2609.31562v1)

**作者:** Nenad Tomasev `[一作]` (Google DeepMind), Simon Osindero `[通讯]` (Google DeepMind)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了构建高度自治AI科学家生态系统的框架，强调了多代理协作、资源分配、数据经济、想法经济以及负面结果、可复制性等关键问题，并提出相应的技术与制度设计。

**💡 创新点**

创新点在于将AI科学家视为多代理经济主体，提出AI代理的“科学数据市场”“科学想法市场”“IP‑NFT”“智能委托”等机制，以及在资源稀缺、实验验证瓶颈下的成本意识与协同机制。

**🔧 技术方法**

技术主要包括多代理系统、智能合约、可信执行环境、零知识证明、数据Shapley值、预测市场、可组合智能合约、可验证计算与可追溯日志等。

**📊 数据集**

由于论文为理论与系统设计，未使用具体实验数据集，主要参考公开数据库（如AlphaFold数据库）与已有科学数据共享实践。

**📈 对比分析**

该工作并未在实验上与传统方法比较，而是通过理论分析与案例讨论说明所提出机制在降低资源浪费、提升实验效率、促进负面结果共享、加强可复制性等方面的潜在优势；性能指标主要体现在概念可行性与对现有流程的改进。

**⚠️ 局限性**

局限在于缺乏实证验证与实现细节，涉及多代理身份与信任、合约可执行性、隐私保护、市场效率与博弈问题的未解答，且在高风险科研与监管合规方面仍面临技术与政策挑战。

---

## 591. FragToken: Amplifying LLM Inference Costs through Noncanonical Token Generation

**arXiv ID:** 2609.31552 | [PDF](https://arxiv.org/pdf/2609.31552v1)

**作者:** Zihan Wang `[一作]` (University of Electronic Science and Technology of China), Guowen Xu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于非规范化 token 生成的训练时资源消耗攻击方法，并实现了 BPE 对齐合并的标签构造与自蒸馏 fine‑tune。

**💡 创新点**

创新点在于利用 token 重映射空间诱导模型生成更长 token 序列而不改变可读文本，从而在普通输入下持续提升推理成本并保持任务性能。

**🔧 技术方法**

采用源模型自蒸馏、容量感知预算和 BPE 对齐合并的训练框架，以及基于 token inflation ratio 的评估。

**📊 数据集**

在 GSM8K、PIQA 与 OpenBookQA 三个 benchmark 上，使用 Alpaca 生成的自蒸馏响应作为训练数据。

**📈 对比分析**

与原始模型、标准 SFT、五种推理时攻击以及两种训练时攻击比较，平均 token inflation ratio 达 1.99–2.46，任务准确率仅下降 0.2–1.5 个百分点，且在大多数防御检测中检测率仅略高于正常。

**⚠️ 局限性**

局限包括对不同语言/领域的迁移性未知、对极端温度或重复惩罚敏感，以及缺乏针对检测器的完整鲁棒性评估。

---

## 592. ClearGS: Reliability-Aware Gaussian Splatting from Handheld Videos

**arXiv ID:** 2609.31509 | [PDF](https://arxiv.org/pdf/2609.31509v1)

**作者:** Xuanzhi Liu `[一作]` (Shenzhen University of Advanced Technology), Song Wang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ClearGS 方法，用于从质量混合、视角覆盖不均的手持视频中重建 3D 高斯 Splatting 场景。

**💡 创新点**

创新点包括：① 分级可靠性监督——将外观可靠性、退化风险和几何效用分离并赋权；② 弱激活被抑制帧以保留轨迹覆盖；③ 无参考 Render‑Guided In‑Video Restoration (RIVR)，利用当前渲染、原始视频的去模糊与高频融合进行候选选择；④ 完整轨迹修复合并以保持早期修复细节。

**🔧 技术方法**

使用技术包括 Qwen3‑VL 进行外观/退化评估，冻结 Restormer 运动去模糊器，MUSIQ 与 CLIP‑IQA 等无参考质量评估，COLMAP 稀疏重建、3DGS 结构，以及自监督权重分配与后期合并。

**📊 数据集**

数据集为 GS2E（10 场景，三种合成模糊级别）和 GSOTM（四种摄像机运动退化：MB、RS、MB+RS、PN）。

**📈 对比分析**

在相同输入、相同评价下与 DiFiX3D、GeoQuery、SyncFix 等基线对比，ClearGS 在 LPIPS、CLIP‑IQA、MUSIQ 等指标上均优于或持平，尤其在强退化场景下取得显著提升。

**⚠️ 局限性**

局限性：无参考评分可能无法完全捕捉所有失真，极端模糊或极低光照帧的恢复效果有限；方法对计算资源（多轮训练、RIVR 计算）有一定负担。

---

## 593. Evaluating Cultural Awareness of LLMs for Haitian Creole

**arXiv ID:** 2609.31506 | [PDF](https://arxiv.org/pdf/2609.31506v1)

**作者:** Christelle Clervilsson `[一作]` (Institut Polytechnique de Paris), Yanzhu Guo `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估低资源语言海地克里奥尔语中大型语言模型的文化意识，构建首个文化相关基准。

**💡 创新点**

首次将文化意识评估与克里奥尔语言结合，提出四维指标（特异性、偏差、多样性、变异性）并进行故事生成偏见分析。

**🔧 技术方法**

采用文本填空与故事生成任务，使用Mistral（中型）模型进行评测，结合人工标注的实体列表。

**📊 数据集**

自制海地克里奥尔文化实体词表（食品、音乐、节庆、政治人物等）与法语对照实体列表，来源于手工收集与Wikidata。

**📈 对比分析**

通过与法语高资源情况对比，发现海地克里奥尔在特异性、多样性方面低于法语；模型在特定领域表现不均，且在故事生成中呈现贫困与韧性刻板印象。

**⚠️ 局限性**

局限包括实体列表不完整、拼写变异导致评测偏差、模型自身语言与文化知识混淆，以及仅使用单一模型进行生成与提取的双重影响。

---

## 594. Fundamental Limits of Sequence Reconstruction Problems in Immunogenomics

**arXiv ID:** 2609.31501 | [PDF](https://arxiv.org/pdf/2609.31501v1)

**作者:** Jaswanthi Mandalapu `[一作]`, Nir Weinberger `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了免疫基因组学中D基因重建的痕迹复杂度，对三种生物学启发的痕迹生成模型进行了信息理论下界与可实现算法的分析。

**💡 创新点**

①在TrimSuffixAndExtend模型下给出了Θ(n)的最优痕迹复杂度并提出低复杂度Prefix-Filtered Mode（PFM）算法；②在TrimAndExtend模型下证明最优痕迹复杂度为Θ(n²)，并用简单的Bit‑Wise Mode（BWM）实现；③在SuffixExtend_t(TrimSuffix)模型下给出多项式上下界，指出尚未收敛。

**🔧 技术方法**

采用信息论的Fano方法和KL散度下界、总变距离下界；对PFM、BWM、TrimAndFindMode分别做精确概率分析；使用Hoeffding不等式和聚类筛选技术。

**📊 数据集**

实验数据为人工合成的二进制序列，采用Monte‑Carlo仿真在不同n和δ下评估误差率，构造置信区间。

**📈 对比分析**

与基线BWM、PFM、TrimAndFindMode对比：在W1中PFM显著优于BWM，满足O(n)理论；在W2中BWM略优于PFM，符合O(n²)预期；在W3中仅给出上界O(n³)，实际性能与理论未完全匹配。

**⚠️ 局限性**

限制：对W3的痕迹复杂度未达最优，缺乏对突变/替换噪声的考虑，实验仅在二进制模拟数据上，未验证对真实AIRR‑seq数据的适用性。

---

## 595. Verifiable Randomness for Blockchain-Based Lottery Systems

**arXiv ID:** 2609.31485 | [PDF](https://arxiv.org/pdf/2609.31485v1)

**作者:** Gonçalo Ferreira `[一作]` (Univ. of Aveiro), Paulo Bartolomeu `[通讯]` (Univ. of Aveiro)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于区块链的可验证抽奖随机数生成平台，利用commit‑reveal协议并引入了可选揭示与未来区块哈希结合的机制；

**💡 创新点**

创新点在于提出“reveal‑as‑a‑choice”可选揭示策略和使用未来区块哈希作为熵源，既解决了传统commit‑reveal的最后揭示攻击，又消除了服务拒绝攻击；

**🔧 技术方法**

采用Solidity智能合约在Polygon 2.0链上实现，核心技术包括commit‑reveal、可选揭示、块哈希熵锚定、XOR聚合以及随机性验证；

**📊 数据集**

实验数据集包括：256场离线模拟（10人/场）、1,000场（5人/场，15块揭示窗口）和10,000场（同参数）的本地区块链运行；

**📈 对比分析**

通过统计测试（Hamming权重、χ²均匀性检验、Shannon熵）与MT19937伪随机数生成器视觉对比，结果显示种子无碰撞、均匀性良好、熵接近理论上限；

**⚠️ 局限性**

局限性包括：无法实现票券退款或抽奖取消、对区块生成者的潜在操纵有一定风险、依赖公共链的性能与成本约束

---

## 596. "AI is (not) the new...": A Diagnostic Analogy Framework for Generative AI's Cultural Impacts

**arXiv ID:** 2609.31482 | [PDF](https://arxiv.org/pdf/2609.31482v1)

**作者:** Rida Qadri `[一作]` (Google Research), Remi Denton `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了一个诊断类比框架，用于分析生成式 AI 在知识发现与合成中的文化影响，区分结构性与偶发性后果。

**💡 创新点**

创新点在于将技术与社会相互作用拆分为“知识场所-治理逻辑-技术机制”三维坐标，提供精细化的类比分析工具，揭示传统模糊类比导致的治理失配。

**🔧 技术方法**

未采用算法实现，而是理论框架；在案例讨论中使用了 LLM、检索增强生成（RAG）、索引/目录等技术概念。

**📊 数据集**

未使用具体数据集；框架讨论以生成式 AI、索引系统等现有技术为例。

**📈 对比分析**

没有量化比较或实验；通过案例对比（索引 vs 推理，编辑权威 vs 统计共识）展示框架的说明力。

**⚠️ 局限性**

局限在于缺乏经验验证，框架主要理论化；对实际治理干预的可操作性和适用范围仍需进一步探索。

---

## 597. Online Learning via Learned Latent Bayesian Tracking

**arXiv ID:** 2609.31559 | [PDF](https://arxiv.org/pdf/2609.31559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 598. Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality

**arXiv ID:** 2609.31454 | [PDF](https://arxiv.org/pdf/2609.31454v1)

**作者:** Bradley Scott `[一作]` (University of Glasgow), Edmond S. L. Ho `[通讯]` (University of Glasgow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在联邦学习环境下，针对标签翻转和图像噪声两种数据损坏，利用输入条件不确定性（含变异估计和 Monte Carlo Dropout 统计）和预测标签损失两种信号进行样本级别的检测，并在 ResNet‑20 上对 CIFAR‑10 与 SVHN 进行实验。

**💡 创新点**

创新点在于提出两种粒度（per‑client 与 within‑client）AUC 评估框架，揭示输入不确定性对标签翻转无效而对图像噪声有效，证明预测标签损失更适合检测标签错误，并证明两信号性能随联邦污染率变化而呈现不同趋势。

**🔧 技术方法**

采用 FedAvg 联邦平均训练、UDF‑GMA 的可学习变异估计（aleatoric head）与 MC Dropout 采样、熵、互信息、预测标签损失等不确定性与误差指标，以及 Mann‑Whitney AUC 作为检测度量。

**📊 数据集**

使用 CIFAR‑10 与 SVHN 两个 32×32 RGB 图像数据集，在 10 个非 IID 客户端（Dirichlet α = 0.1、0.3、1.0）上进行实验。

**📈 对比分析**

通过在 q∈{0.2,0.4,0.6} 与 M∈{2,4,6,8} 的网格上评估 within‑client AUC，结果显示标签翻转下预测标签损失 AUC 达 0.85（CIFAR‑10）/0.95（SVHN），输入不确定性接近 0.5；图像噪声下预期熵 AUC 为 0.67/0.66，高于损失 0.64，且随着 M 增大优势更加明显。

**⚠️ 局限性**

局限性包括仅评估非对抗性标签错误与单一噪声水平、仅使用 ResNet‑20、10 个客户端与 FedAvg，未考察更复杂任务、不同模型、不同聚合策略或对抗性攻击，也未将信号集成到完整防御管线，只作诊断性评估。

---

## 599. Region-Level Black-Box Defense Against Stealthy Embedding-Space Backdoors in CLIP

**arXiv ID:** 2609.31558 | [PDF](https://arxiv.org/pdf/2609.31558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 600. Strategically Diverse Sampling for Self-Training

**arXiv ID:** 2609.31571 | [PDF](https://arxiv.org/pdf/2609.31571v1)

**作者:** Alexander Gurung `[一作]` (University of Edinburgh), Mirella Lapata `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了两种基于策略多样性的采样方法（Groot 与 Verbalized Sampling），用于生成多样化的自训练数据，从而提升大型语言模型在困难编程与故事规划任务上的性能。

**💡 创新点**

创新点在于把“策略多样性”视为自训练的关键维度，证明其比仅靠正确性更能驱动模型提升，并展示了仅使用学生模型而非大型教师即可获得更优结果的可能性。

**🔧 技术方法**

采用了层次树结构采样（Groot）、基于概率列举的采样（VS），随后进行 SFT 与 RFT 微调；评估时使用 Recursive Self‑Aggregation (RSA) 进行测试时聚合，并通过 MaxRL/GRPO 对模型进行强化学习。

**📊 数据集**

实验数据集包括编程领域的 Cobalt、LiveCodeBench、OJBench（前沿难度子集）以及故事下一章节预测（Next‑Chapter Prediction, NCP）数据集。

**📈 对比分析**

与 IID 采样（含高温、扩展预算）以及 235B 规模教师进行对比，策略多样化方法在 pass@k、coverage、RSA 与 RL 等指标上显著优于对照组；在 frontier 代码任务上 pass@64 提升至 15–18%，在 NCP 上 coverage@15% 提升至 4–5%；自训练 4B 模型的表现甚至超过 235B 教师的 IID 训练。

**⚠️ 局限性**

局限性包括：需要额外一次生成调用以构造策略树或列表；效果受限于学生模型本身能生成的策略空间；在推理阶段未实现直接多样化；尚未在更大规模模型或更广泛任务上系统验证。

---

## 601. Generalization behavior of OPTQ and the role of regularization

**arXiv ID:** 2609.31560 | [PDF](https://arxiv.org/pdf/2609.31560v1)

**作者:** Erin George `[一作]` (University of California, San Diego), Rayan Saab `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

研究并推导了OPTQ（及其随机变体）在泛化误差上的理论界限，并提出了基于正则化参数λ的取值公式。

**💡 创新点**

首次给出数据驱动量化算法的经验-总体误差比较不依赖特定模型，并提供了针对λ的最优取值建议，显著改善低样本场景的性能。

**🔧 技术方法**

利用正则化经验误差与总体误差的比较不等式、最小二乘更新、正则化矩阵扩展、矩阵微积分推导、随机取样理论等技术。

**📊 数据集**

三类合成分布：均匀Hadamard、非均匀Hadamard、ReLU网络输出分布（64×16随机矩阵乘以ReLU）。

**📈 对比分析**

与之前的两种λ建议（λ₁=0.01‖A‖_F²/N，λ₂=10⁻³‖A‖_op）对比，通过实验在不同m值下测量测试误差，发现新推荐的λ在低采样时误差显著下降，在高采样时保持相近或更优性能。

**⚠️ 局限性**

仅在合成数据上验证，未在真实大型模型权重上测试；正则化参数λ的常数β仍需经验选择；理论假设如矩阵满秩、数据分布紧凑等限制了泛化性。

---

## 602. NEXT: Physics-Informed Neuro-Spectral Exponential Time Differencing Architectures

**arXiv ID:** 2609.31539 | [PDF](https://arxiv.org/pdf/2609.31539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 603. HySTAR: Anchored Hypergraphs for Stable Credit Assignment in Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2609.31531 | [PDF](https://arxiv.org/pdf/2609.31531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 604. SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery

**arXiv ID:** 2609.31507 | [PDF](https://arxiv.org/pdf/2609.31507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 605. Vision-Based 6-DoF Grasp Pose Estimation for Robot Cloth Unfolding

**arXiv ID:** 2609.31452 | [PDF](https://arxiv.org/pdf/2609.31452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 606. EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models

**arXiv ID:** 2609.31551 | [PDF](https://arxiv.org/pdf/2609.31551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 607. Game Arena: Strategic LLM Evaluation in Competitive Environments

**arXiv ID:** 2609.31473 | [PDF](https://arxiv.org/pdf/2609.31473v1)

**作者:** Bovard Doerschuk-Tiberi `[一作]`, Minmin Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 Kaggle Game Arena 平台，发布了棋类、扑克和狼人三种竞技游戏环境，并在这些环境中对多款大语言模型进行大规模头对头对战评估。

**💡 创新点**

创新点在于通过持续、可扩展的游戏竞技来避免传统静态基准饱和，并提供统一的文本化接口、基于真实结果的客观指标、可复现的数据与评测流程，支持多种信息结构与交互复杂度的游戏。

**🔧 技术方法**

采用了统一文本化 harness、ReAct 交互框架、Elo/Bradley–Terry、BB/100、Game‑Theoretic Evaluation (GTE) 等评分方法，并通过大量手牌复制、方差降低与 Bootstrap CI 等统计技术提升评测稳健性。

**📊 数据集**

使用了自生成的大规模游戏数据集：棋局采用 PGN/FEN 记录（每对模型 40 场）、扑克采用 900,000 手（每手 100 场 100 手的 100‑手 episode 及镜像）以及狼人约 31,000 场的事件日志。

**📈 对比分析**

评测采用全对角锦标赛并给出 Bootstrap 置信区间；结果显示模型实力明显分层：Gemini 3 Pro/Flash 在棋类占优，GPT‑5.2 在扑克中领先，其他模型按收益/失误区间可分为顶层、中层和底层，性能差异显著且具统计显著性。

**⚠️ 局限性**

主要局限包括：固定游戏量导致计算效率低、缺乏自适应调度；模型更新频繁导致排行榜连贯性受影响；跨游戏统一评分难度大；多玩家游戏中信用分配和团队结果归属问题尚未解决；目前仅覆盖三种游戏，需扩展更多多样化任务。

---

## 608. Polychromatic 2-colorings with Bounded Discrepancy for Triangulations

**arXiv ID:** 2609.31574 | [PDF](https://arxiv.org/pdf/2609.31574v1)

**作者:** Alma Arevalo Loyola `[一作]` (Carleton University), Thomas Shermer `[通讯]` (Simon Fraser University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文研究平面三角化的多色极限2-着色（polychromatic 2‑coloring）的差异度（discrepancy），并提出了新的上界与有效算法。

**💡 创新点**

创新点包括：① 通过匹配、四色着色与对偶图的三条边着色构造跨越性二分着色，从而把差异度降低至 3n−16/7；② 证明在存在匹配或完美匹配时差异度可降至 n/3；③ 设计了线性时间局部最小化算法，使得得到的差异度不超过 5n−24/7，首次给出多项式（线性）时间实现；④ 提出了“重要/非重要”顶点分类与颜色交换技巧，为进一步改进差异度提供新思路。

**🔧 技术方法**

主要技术手段包括：离散充电（discharging）方法、三角化的支撑四边化（spanning quadrangulation）、对偶图 3‑边着色对应四色着色、匹配大小与独立集上界的关系、局部最小化和颜色交换策略，以及对凸面与面数关系的分析。

**📊 数据集**

本工作为理论性质研究，没有使用实际数据集，全部结果均通过理论证明获得。

**📈 对比分析**

与先前最优 5n−169 的上界相比，本文将上界压缩至 3n−16/7，进一步证明了在存在匹配或完美匹配时差异度可降至 n/3；线性时间算法实现了差异度 ≤5n−24/7，比之前的多项式时间算法更高效，运行时间仅为 O(n)。

**⚠️ 局限性**

局限性：线性时间算法尚未达到最优 n/3 的差异度，只能得到 5n−24/7；此外，算法对图的结构要求仍为三角化，且对复杂结构的进一步优化仍为开放问题。

---

## 609. DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education

**arXiv ID:** 2609.31568 | [PDF](https://arxiv.org/pdf/2609.31568v1)

**作者:** Quang Nguyen `[一作]` (Posts and Telecommunications Institute of Technology), Nam Vu `[通讯]` (Posts and Telecommunications Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套可在越南教育环境中自托管、符合数据主权法规的 AI 辅导系统 DeepEdu‑v1。

**💡 创新点**

创新点包括：1）Similarity Chunk Rolling (SCR) 通过聚类相似子块来对长文本上下文进行稀疏注意力的批量化，显著减少检索调用和预填充延迟；2）自适应 Agentic Layer，利用 Playbook 记录校准的教学规则而非大规模权重微调，能够持续提升本土化知识准确性并保持可审计性。

**🔧 技术方法**

技术核心为：多头注意力的稀疏选择（TokenSelect）与 SCR 的聚类优化；自监督的 ACE（Agentic Context Engineering）框架；后训练量化（AWQ/GPTQ）；FlashAttention/PagedAttention 等低延迟实现；对比学习与失败记忆库、对抗式课程生成等自我提升机制。

**📊 数据集**

使用的基准数据集包括：InfiniteBench、RULER、Formula、FiNER（金融推理）、AppWorld（多步工具交互）以及越南本土化教材（Sach Giao Khoa）原始文本（在内部实验中未公开）。

**📈 对比分析**

与 TokenSelect 等现有长文本推理方法相比，SCR 在 R.KV 等检索任务中保持或提升准确率（最高提升约 7%），TTFT 延迟下降约 35%；与 ACE 基线相比，加入 RAE、对抗课程和失败记忆后，金融推理平均准确率从 60.7% 提升至 65.8%，交互式 AppWorld 的目标完成率从 9.3% 提升至 10.4%。整体系统在 H100 GPU 上实现了约 2 倍的 TTFT 加速。

**⚠️ 局限性**

局限性：评估主要集中在公开长文本与交互式基准，尚未在完整越南课程数据上系统验证本土化效果；模型在极长上下文下仍受 KV 缓存动态消耗限制；自我提升循环需依赖高质量的人工校正或自动化反馈，实际部署中的误差传播与公平性问题仍需进一步研究。

---

## 610. BeatGraph: Self-Supervised Heartbeat Graphs for Infant ECG Representations from the Home Environment

**arXiv ID:** 2609.31546 | [PDF](https://arxiv.org/pdf/2609.31546v1)

**作者:** Mohammad Nur Hossain Khan `[一作]` (University of Massachusetts Amherst), Bashima Islam `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `109c2b71-d051-425c-831f-0c544c24280d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种以心跳为单元的自监督 ECG 编码器 BeatGraph，并在婴儿单通道 ECG 上实现多任务学习与迁移。

**💡 创新点**

创新点在于将心跳直接作为 token 构造完整图，结合 Transformer 与图注意力，使用掩码心跳嵌入预测与 BYOL 自蒸馏进行预训练，显著提升婴儿 ECG 的表示质量。

**🔧 技术方法**

技术包括自监督掩码心跳嵌入预测、BYOL 自蒸馏、心跳波形+间隔特征编码、Transformer 位置编码、GATv2 图注意力以及注意力池化。

**📊 数据集**

数据集为 3,361 小时未标注婴儿单通道 ECG（143 名 3–11 个月婴儿）及 47.5 小时标注任务数据；外部 ZZU-pECG（儿童）和 PTB-XL（成人）用于迁移评估。

**📈 对比分析**

与 ECG-FM、HuBERT-ECG、SimCLR、BYOL、MTAE 等基线在四个婴儿任务上对比，BeatGraph 在准确率、macro-F1 与 κ 上均领先 0.076–0.158；在 ZZU-pECG AUROC 为 0.892，接近最佳模型；在 PTB-XL 线性评估 AUROC 达 0.931，匹配或超过 ST-MEM。

**⚠️ 局限性**

局限包括对心跳检测准确性的依赖、缺乏手工心跳标注验证、单通道模型对极端心率或噪声鲁棒性待进一步研究，以及在多通道环境下的适配性尚未充分探索。

---

## 611. Can You Check That? The Checkability Boundary for Local LLM Network Automation

**arXiv ID:** 2609.31540 | [PDF](https://arxiv.org/pdf/2609.31540v1)

**作者:** Maleeha Masood `[一作]` (University of Illinois Urbana-Champaign), Momina Nofal `[通讯]` (Independent Researcher)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于Intrinsic Check的局部优先网络自动化框架Touchstone，使用小型语言模型生成候选答案，并通过任务特定的确定性检查决定是否接受本地答案或上报至前沿大型模型；

**💡 创新点**

创新点在于将Intrinsic Check引入模型推理门控，构建任务可检验性边界；通过多模型组合、离线性能评估和置信度门控实现选择性上报；

**🔧 技术方法**

采用7个1–8B的通用小型语言模型（Llama‑1B、Qwen‑1.5B、SmolLM‑1.7B、Qwen‑3B、Llama‑3B、Qwen‑7B、Llama‑8B），离线统计模型权重，聚合候选（投票、执行测试或规则拆分），并使用预定义的Intrinsic Check（一致性、覆盖与原始性、可逆性、执行验证）进行本地裁决；

**📊 数据集**

使用六个结构化网络任务数据集：冲突检测、意图翻译、三种日志解析（OpenSSH、HDFS、Proxifier）、路由代码生成，以及知识性控制任务TeleQnA；

**📈 对比分析**

与前沿GPT‑5.5全量推理基线对比，Touchstone在可检验任务上实现了与全局模型相当甚至更高的准确率（如冲突检测98.6%），并将上报率显著降低（最多仅16%），但在弱检验或无检验任务中准确率提升有限，且误报率（FAR）较高；

**⚠️ 局限性**

局限性包括：(1) 需要任务专用的Intrinsic Check，若任务缺乏可检验特性则无效；(2) 检查的误报率高时导致误接受；(3) 对于路由代码等需高质量代码生成的任务，SLM生成通过检验的候选极少，导致上报率高；(4) 设计检查需人工工程，难以自动化。

---

## 612. Structured Reasoning Agentic Framework for Interpretable Critical View of Safety Assessment

**arXiv ID:** 2609.31524 | [PDF](https://arxiv.org/pdf/2609.31524v1)

**作者:** Qing Xu `[一作]` (University of Nottingham), Zhen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ReasonCVS 框架，用解剖结构图抽象、VLM 驱动的子准则验证和基于理由的推理代理实现可解释的 Critical View of Safety 评估。

**💡 创新点**

首次将 CVS 评估拆解为结构化的解剖图推理，采用子准则层级验证与理由蒸馏，使模型在保持高性能的同时实现可追溯、可解释的判定。

**🔧 技术方法**

使用 Vision‑Language 模型（VLM）、大型语言模型（LLM）、SAM‑3 分割、LoRA 微调以及教师‑学生理由蒸馏技术。

**📊 数据集**

在 Endoscapes‑CVS201 基准上进行实验，包含 11,090 帧图像级注释和 493 帧像素级分割。

**📈 对比分析**

与 10+ 传统与多模态方法（如 ResNet、SurgVLP、HecVL、LayoutCVS、LG‑CVS、SV2LSTG 等）对比，ReasonCVS 获得 68.1% mAP、81.4% BAcc，较 SV2LSTG 提升 3.8% mAP、6.6% BAcc。

**⚠️ 局限性**

局限在子准则无标注导致理由蒸馏依赖教师模型，可能受 VLM 识别误差影响，对未知解剖变异的泛化能力有待验证，且模型推理过程相对复杂。

---

## 613. KneePreM: Towards 3D Knee MRI Foundation Models via Large-Scale Unlabeled Pretraining and Label-Efficient Fine-Tuning

**arXiv ID:** 2609.31461 | [PDF](https://arxiv.org/pdf/2609.31461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 614. Fast and Secure Simultaneous Authentication of Equals for WPA3

**arXiv ID:** 2609.31519 | [PDF](https://arxiv.org/pdf/2609.31519v1)

**作者:** João Ferreira `[一作]` (University of Aveiro), Hélder Gomes `[通讯]` (University of Aveiro)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种重构的 WPA3‑SAE 认证框架，将高成本的“狩猎‑捕获”过程迁移至客户端，并通过慢速路径密钥派生和无状态会话票据实现快速重连。

**💡 创新点**

创新点在于（1）使用单向哈希+PBKDF2/Argon2 生成 s，使 AP 的验证变为单步；（2）实现无状态会话票据，避免 AP 存储会话状态；（3）将计算密集型任务卸载给客户端，显著降低 AP 的 CPU 负载。

**🔧 技术方法**

采用的技术包括：椭圆曲线密码学、HMAC、PBKDF2/Argon2、AES 加密、Encrypt‑Then‑MAC 票据封装、IEEE 802.11 Beacon IE 定制。

**📊 数据集**

未使用公开数据集，实验基于虚拟化测试平台（Intel Core i9‑13900HX、32GB DDR5、hostapd/wpa_supplicant VMs）进行性能评估。

**📈 对比分析**

与标准 WPA3‑SAE 的对比：初始认证平均延迟从 2284 µs 降至 72.5 µs（≈96.8% 降低），重连时延从 2284 µs 降至 21 µs（≈99% 降低）。

**⚠️ 局限性**

局限性包括：票据使用导致的前向保密性下降、对非支持新协议的客户端兼容性不足，以及需要 AP 在 Beacon 中暴露盐/迭代信息可能被恶意 AP 进行降级攻击。

---

## 615. TemplateCraft: Agentic Visual Template Generation

**arXiv ID:** 2609.31451 | [PDF](https://arxiv.org/pdf/2609.31451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 616. A search-to-decision reduction for the linear code equivalence problem

**arXiv ID:** 2609.31517 | [PDF](https://arxiv.org/pdf/2609.31517v1)

**作者:** Jean-François Biasse `[一作]` (University of South Florida), Philip Waitkevich `[通讯]` (University of South Florida)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种多项式时间的搜索到决策化简，证明给定线性码等价的搜索版本（PCE 与 LCE）可以通过有限次数的决策判定查询来实现，从而直接恢复线性同构。

**💡 创新点**

创新点在于首次给出对 LCE 的完整搜索-决策化简，并设计了一个确定性多项式时间方法仅通过行约简和支撑连通性来恢复单列比例（对角）部分，从而填补了先前仅为启发式或基于线性系统求解的空白。

**🔧 技术方法**

核心技术包括：对列的重复/比例结构进行“伪复制”扩展以构造可比实例；利用行简化保持列支撑不变；通过支撑连通图传播比例约束；以及使用最小化的线性系统与 RREF 唯一性来判定和重构对角矩阵。

**📊 数据集**

本文属于理论计算机科学范畴，未使用实验数据集，所有结果均在数学证明与算法复杂度分析框架内给出。

**📈 对比分析**

方法在理论上与现有的搜索算法（如 Leon 算法、SSA）相比不需要预先求解大规模二次方程组，而是通过决策 Oracle 的调用实现；但由于缺乏实验评测，无法给出实际性能数值，但证明了多项式上界并与决策 LCE 等价。

**⚠️ 局限性**

主要限制在于依赖一个理想的线性码等价判定 Oracle，实际实现该 Oracle 的复杂度未知；此外，算法对不可分解代码的处理需要先将代码分解为不可约成分，若该分解过程困难则整体效率受限。

---

## 617. TinyAudio: Compact and Efficient Text-to-Audio Generation for Low-Resource Deployment

**arXiv ID:** 2609.31525 | [PDF](https://arxiv.org/pdf/2609.31525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 618. Authority at Commit Time: Reject-and-Rerun Semantics for Governed Agentic Systems

**arXiv ID:** 2609.31490 | [PDF](https://arxiv.org/pdf/2609.31490v1)

**作者:** Jesus Salas `[一作]` `[通讯]` (Independent Researcher), Jesus Salas (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种在企业代理系统中通过完整调解实现的提议接收、验证和授权协议，确保只有当前权威批准的提议才能产生机构效应。

**💡 创新点**

将乐观并发控制与语义验证结合，提出拒绝并重跑的提交协议，并将权限与组件依赖检查作为读集。

**🔧 技术方法**

OCC、语义验证器、事务性outbox、签名快照清单、分区局部日志、etcd租约与时钟观察。

**📊 数据集**

使用Phi‑4大模型的三条种子构建的合规性与税务等工作流。

**📈 对比分析**

通过Matrix原型实现的L4/L5生命周期、局部网络化组合与委托网关三层实验，对比完全调解与直接凭证旁路，发现完全调解无误报，侧路误报被检出；延迟在250 ms以内，吞吐和大规模测试未完成。

**⚠️ 局限性**

尚未实现认证身份、跨域日志完整性、WAN容错、吞吐量、跨域原子性；完整调解需全局控制，代理旁路仍可能绕过审计。

---

## 619. How Far Can INRs Go? Cross-Domain Parameter-efficient INR-Based Semantic Segmentation for Brain MRI

**arXiv ID:** 2609.31573 | [PDF](https://arxiv.org/pdf/2609.31573v1)

**作者:** Ziyao Shang `[一作]` (University of Waterloo), Sirisha Rambhatla `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了在低参数和跨域环境下的脑MRI分割，探讨了隐式神经表示（INR）在语义分割中的表现，并提出了新的层次融合架构HierINRSeg。

**💡 创新点**

创新点在于：①对MetaSeg INR层级特征进行几何分析，发现语义信息分布在不同层；②设计HierINRSeg层次融合多层INR特征，提高鲁棒性与泛化；③在不同参数、增强、3D覆盖率下系统比较INR与传统U-Net的规模特性。

**🔧 技术方法**

采用隐式神经表示（SIREN）、元学习（MAML）、层次特征融合、PCA/UMAP可视化、对比损失与焦点损失等技术。

**📊 数据集**

使用ABIDE I数据集中的KKI（源）、Caltech、CMU（目标）三中心脑MRI数据。

**📈 对比分析**

在控制规模、增强两种协议下，分别对比MetaSeg、HierINRSeg、U-Net、nnUNet-v2、TransUnet、UNeXt等模型；结果显示HierINRSeg在低参数（≤50k）下比MetaSeg提升5.6 Dice，OOD上提升8.2 Dice；INR模型在低参数且增强受限时表现优于U-Net，随参数增大U-Net优势显现。

**⚠️ 局限性**

局限在于仅使用强度变换的增强，未考虑空间扰动；对更大3D覆盖率时表现下降；实验仅限脑MRI，需在更广泛数据集验证。

---

## 620. Toward verifiably private learning from federated data

**arXiv ID:** 2609.31494 | [PDF](https://arxiv.org/pdf/2609.31494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 621. A Flow Matching Framework for Neural Representational Dissimilarity

**arXiv ID:** 2609.31544 | [PDF](https://arxiv.org/pdf/2609.31544v1)

**作者:** Zeyuan Ye `[一作]` (University of Texas at Austin), Xue-Xin Wei `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于流匹配（flow matching）的框架，用以统一神经表征不相似性度量，并通过该框架对多种已有距离度量（如欧氏、相关、马氏、KL、Fisher信息等）进行统一理论推导与估计；同时设计了新的几何模板距离；在合成数据、模拟实验与真实小鼠视觉皮层记录上验证了该框架的有效性；

**💡 创新点**

创新点在于：①将多种传统的神经表征距离度量归结为不同速度场约束下的Jeffreys散度；②通过流匹配提供了一个通用且可学习的估计器，特别适用于高维复杂分布与连续条件；③利用源分布模板与速度约束设计新的度量，实现对几何属性的显式控制；

**🔧 技术方法**

核心技术包括：连续正则化流（continuous normalizing flow）与流匹配（flow matching）算法、Jeffreys散度计算、深度神经网络学习速度场、Monte Carlo估计、PCA降维等；

**📊 数据集**

使用的数据集包括：①合成高斯与高斯混合条件数据；②模拟连续变量的高斯响应数据；③六组小鼠视觉皮层电极记录（10k-25k神经元，4k次静态光栅试验）；④Allen Visual Coding Neuropixels与Calcium成像数据；

**📈 对比分析**

与传统方法（如欧氏距离、相关、马氏、KL、GKR、Wishart过程、局部线性估计等）比较时，流匹配在高维、非高斯或连续条件下往往表现更好，尤其在：①Fisher信息估计（线性和全信息）时，流匹配的误差低于GKR、Wishart等；②时间分辨RDM估计时，流匹配在保持高似然、捕捉非高斯结构方面优于传统方法；整体性能提升体现在更高的Pearson相关性、更低的相对绝对误差以及更好的log-likelihood表现；

**⚠️ 局限性**

主要限制包括：①理论推导基于全局最优，而神经网络训练可能未达此点，导致估计偏离理论对应度量；②验证仅集中在视觉皮层，尚未推广至其他脑区或测量技术（fMRI、EEG、深度网络模型）；③对源分布与速度约束的选择仍需经验性调优；

---

## 622. Prompt Minimization: Reducing Input Redundancy Without Sacrificing Output Fidelity

**arXiv ID:** 2609.31505 | [PDF](https://arxiv.org/pdf/2609.31505v1)

**作者:** Marius F. R. Juston `[一作]` (University of Illinois Urbana-Champaign), Rudhi Bashambu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了三种任务无关的提示最小化方法：零射击搜索、基于进化的上下文学习和RL微调。

**💡 创新点**

创新点在于将提示压缩视为可搜索的优化问题，提供三种轻量化搜索策略，并构建统一的两项目标（语义相似度与压缩率）来指导搜索。

**🔧 技术方法**

技术包括：LLM零射击压缩、基于进化策略的多候选生成、PPO+LoRA的RL压缩策略、BERTScore语义评估、压缩率计算以及LangChain管线自动化。

**📊 数据集**

使用 60 条由 GPT‑4o、GPT‑5.1、Grok 和 Claude Sonnet 4.6 生成的长篇开放式提示（平均128词，范围60–232词），并在这些提示上进行实验。

**📈 对比分析**

对比方法为：BERTScore 语义相似度、压缩率比值，二者按 0.5/0.5 权重组合。实验表明：三种方法均能显著缩短提示长度且保持高语义相似度；RL 微调在某些模型上表现出更强的压缩能力；在 Llama‑3.1‑8B 上压缩比优于 Qwen‑2.5‑32B，但两者的 BERTScore 接近。

**⚠️ 局限性**

局限性包括：多候选搜索在时间与令牌上成本高、BERTScore 可能忽略语用细节、实验主要基于机器生成的提示，未验证专家手写提示的适用性，且仅评估单一参考输出，未能覆盖任务扰动或更广泛输入分布。

---

## 623. Retail Product Search: A Practical Approach at Target

**arXiv ID:** 2609.31498 | [PDF](https://arxiv.org/pdf/2609.31498v1)

**作者:** Darshan Sonagara `[一作]` (Data Sciences, Target Corporation), Alex Li `[通讯]` (Data Sciences, Target Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电商场景下构建并部署了混合检索系统，将传统的词典检索与稠密向量检索并行执行，结合精度控制和加权交错融合，以提升搜索相关性和商业转化率。

**💡 创新点**

核心创新包括：1) 用用户交互日志（点击、加入购物车、购买）构造可量化的正负样本，训练域适配的双编码器；2) 设计精度控制模块，基于NER和查询分类器实现属性过滤；3) 在列表重叠很小的场景下证明加权交错融合优于倒数排名融合；4) 在生产环境中实现低延迟向量检索（ANN + 缓存 + 自适应批处理）。

**🔧 技术方法**

技术栈包括：BERT（DistilBERT）双编码器、改进的对比损失(ICL)、ScaNN ANN索引、Solr倒排索引、REST API微服务、TorchServe、CPU 端推理、动态字段掩码、NLP 识别与分类。

**📊 数据集**

数据集：约20M条基于交互日志生成的查询-商品对，按类别和查询频率分层抽样；5k条人工标注的电商查询-商品相关性基准；在离线评测中使用这些数据进行 NDCG、MRR 计算；在线A/B测试使用真实流量进行CTR、ATC、转化、需求等指标。

**📈 对比分析**

方法比较：离线时，扩充训练集、加入硬负样本、使用字段标记与高亮掩码等改进使 NDCG@5 从0.8200 提升到0.8794，MRR 0.9822；在线时，Finetuned v2 方案相对词典检索提升 CTR 0.97%、转化 0.98%、需求 1.10%，并将零结果率降低约50%。

**⚠️ 局限性**

局限性：缺乏个性化检索，仅使用文本特征，无法充分利用图像或多模态信息；精度控制采用硬阈值，未能动态学习；在高并发或新产品更新时的检索一致性尚未完全解决。

---

## 624. COFI-DQI: Curve-based Optimal Function Intersection via Decoded Quantum Interferometry

**arXiv ID:** 2609.31484 | [PDF](https://arxiv.org/pdf/2609.31484v1)

**作者:** Gretchen L. Matthews `[一作]` (Virginia Tech), Julia Shapiro `[通讯]` (Virginia Tech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于解码的量子算法，称为解码量子干涉（DQI），用于组合优化问题，特别是针对有限域上的最大线性可满足性（max-LINSAT）问题。

**💡 创新点**

创新点在于引入了COFI（基于曲线的最优函数交集），扩展了DQI的应用范围，并通过使用代数几何码的不同曲线族来提高性能。

**🔧 技术方法**

使用了量子傅里叶变换和代数几何码的解码理论，结合了量子干涉技术来优化解的概率。

**📊 数据集**

使用了来自不同代数几何曲线的码，包括Suzuki码和扩展范数-迹码，特别是针对Hermitian码的比较。

**📈 对比分析**

通过与一维Hermitian码的DQI满意度进行比较，展示了Suzuki码和扩展范数-迹码在量子资源需求和约束数量方面的优势，证明了在某些条件下它们的满意度分数更高。

**⚠️ 局限性**

限制在于当前的DQI框架主要依赖于唯一解码算法，未来的研究可以探索如何将列表解码算法纳入DQI框架，以提高解码半径和优化保证。

---

## 625. Uncertainty-Aware Federated Learning for Infant Movement Analysis

**arXiv ID:** 2609.31463 | [PDF](https://arxiv.org/pdf/2609.31463v1)

**作者:** Edmond S. L. Ho `[一作]` `[通讯]` (University of Glasgow), Edmond S. L. Ho (University of Glasgow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了第一套用于婴儿运动分析的联邦学习框架，针对General Movement Assessment（GMA）任务实现了基于2D骨架序列的移动检测与分类。

**💡 创新点**

提出了Uncertainty‑Aware Federated Averaging（UA‑FedAvg）聚合策略，将MC‑Dropout得到的预测熵作为客户端置信度权重，动态调整各客户端在全局模型更新中的贡献。

**🔧 技术方法**

采用CNN端到端模型、Monte Carlo Dropout进行不确定性估计、联邦平均FedAvg及其UA‑FedAvg变体、FedAvg‑w/validation‑loss、Flower框架进行分布式训练与聚合。

**📊 数据集**

使用Kulvicius等公开的45名婴儿的多模态数据集，仅利用ViTPose提取的2D骨架（15个关节、4维特征）组成的5秒视频片段，共1683个片段（FM+ 943, FM‑ 740）。

**📈 对比分析**

在3种数据分布（111‑211、211‑211、211‑112）下与集中式、无联邦及传统FedAvg进行对比。UA‑FedAvg（含或不含验证损失）在多数配置下可提升准确率、灵敏度、F1分数超过FedAvg约0.5–1.5%，整体接近集中式模型（误差<1%）。

**⚠️ 局限性**

局限性包括：仅在典型发育婴儿数据上验证，未覆盖高危人群；使用预测熵作为不确定性指标可能导致过度自信；未与FedProx、SCAFFOLD、FedAdam等主流方法做更系统对比；缺乏对信息治理与安全技术的深入探讨。

---

## 626. Segment-Level Agentic Topic Modeling for Improved Data Exploration and Resource Efficiency

**arXiv ID:** 2609.31460 | [PDF](https://arxiv.org/pdf/2609.31460v1)

**作者:** Myeongjun Erik Jang `[一作]` (J.P. Morgan Chase), Fran Silavong `[通讯]` (J.P. Morgan Chase)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了SeLATM框架，通过段级主题生成与代理反馈循环实现更高效、可解释的主题建模。

**💡 创新点**

创新点在于消除文档级主题分配，采用段级聚类+LLM生成主题，并引入多代理迭代细化流程。

**🔧 技术方法**

技术包括语义分割、凝聚层次聚类、最大边际相关性抽样、LLM（OpenAI GPT）生成主题与评估、代理一致性/多样性评估及分裂/合并操作、规划器。

**📊 数据集**

使用公开数据集Banking77、Bills、Wiki以及公司业务数据集CCC、BCR、CI。

**📈 对比分析**

与BERTopic、C‑Top2Vec、TopicGPT、LLooM、TIDE、LLM‑ITL等基线比较，SeLATM在P1、ARI、NMI、LLM‑as‑a‑Judge的TA、TC等多项指标均超越或相当基线，并显著降低Token消耗。

**⚠️ 局限性**

局限在于迭代过多导致合并过度、CI数据集改善有限，以及规划器阈值需进一步调优。

---

## 627. Diagnosing the Sources of Compositional Failure in Vision-Language Models: A Controlled Analysis

**arXiv ID:** 2609.31456 | [PDF](https://arxiv.org/pdf/2609.31456v1)

**作者:** Mona Gandhi `[一作]` (Ohio State University), Srinivasan Parthasarathy `[通讯]` (Ohio State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了COMPASS评测框架，用结构化的场景图构造分层、多难度的图像-文本对，分离并量化联合推理成本与各视觉技能的负荷；

**💡 创新点**

创新点在于通过将由对象、属性、关系构成的复合说明拆分为单独的原子说明，建立“组合集成差距”与“技能负荷”两种可控评测维度，首次在同一实验设置下揭示不同原子类型对模型性能的主导影响；

**🔧 技术方法**

采用场景图提取、GPT‑4o‑mini生成自然语言说明、LLM替换生成精细负样本、检索式评测与OLS回归分析性能随原子计数变化的关系；

**📊 数据集**

使用Visual Genome场景图作为基础，生成约138万条合成说明并通过硬负样本构造得到87k组组合/拆分对和274k组技能负荷对；

**📈 对比分析**

在OpenCLIP、SigLIP v2、PE‑CLIP、NegCLIP、CE‑CLIP、BLIP‑L及Qwen3‑VL‑Embedding‑8B等多种VLM上进行统一检索评测，发现组合集成差距普遍为正，但仅解释了部分性能下降；技能负荷回归显示每项技能的性能主要受自身原子计数（self‑load）负面影响，交叉负载（cross‑load）往往为正，且此模式在所有模型族中保持一致；

**⚠️ 局限性**

局限包括说明合成依赖于Visual Genome场景图与GPT‑4o‑mini，可能与真实自然语言分布不符；属性与关系评估始终伴随对象上下文，无法完全独立评估；评测仅覆盖检索任务，未涵盖生成式模型；对自负荷下降机制（编码或跨模态对齐缺陷）仍缺乏深入解释；

---

## 628. From Reward Signal to Visual Utility: A Controlled Audit of Medical VLM Post-Training

**arXiv ID:** 2609.31450 | [PDF](https://arxiv.org/pdf/2609.31450v1)

**作者:** Wang Jingxin `[一作]` `[通讯]` (Chinese Academy of Sciences), Wang Jingxin (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对医学视觉语言模型在PMC‑VQA数据集上的后训练做了系统性、细粒度的评估，比较了多种训练策略（SFT‑LoRA、不同参数适配范围、GRPO、对抗证据目标），并通过三位二进制图像干预模式（Correct/NoImage/Shuffled）追踪每个问题的视觉依赖变化；同时引入生成路径对齐技术检验证据目标与实际回答的一致性。

**💡 创新点**

创新点在于：① 通过配对问题级别的三位二进制干预模式细致拆解视觉益处与危害事件，揭示SFT在视觉依赖上的重分布；② 对抗证据目标与生成路径对齐，发现目标对交叉条件分数的敏感性；③ 通过对比不同参数适配范围与强化学习（GRPO）的效果，探究视觉路径扩展对任务表现的负面影响。

**🔧 技术方法**

使用了LoRA（语言模型/视觉模型）、多模态参数高效适配、Group Relative Policy Optimization (GRPO)、对抗证据目标（softplus惩罚）、生成路径追踪、配对Bootstrapped置信区间统计。

**📊 数据集**

主要使用PMC‑VQA（2,000题清洁测试集、10,000题训练集、1,500题验证集）进行多模态问答评估；另用SLAKE数据集检验本地答案生成的效果。

**📈 对比分析**

通过配对问题级别的Bootstrapped 10,000次重采样计算置信区间，比较不同训练方案的Correct‑Image准确率、视觉益处率(VBR)、视觉危害率(VHR)、图像敏感度(IS)等指标。结果显示：SFT‑LoRA提升了约+1.1个百分点的Correct‑Image准确率，但同时产生了大量视觉益处/危害事件的变动；扩大适配范围降低了准确率；标准GRPO几乎不改变准确率；对抗证据目标在训练集上显著提升目标分数，但在验证集上的增益不稳定。

**⚠️ 局限性**

局限性包括：仅在单一3B模型与PMC‑VQA数据集上实验，未考察随机种子和更大模型的可重复性；图像替换映射可能导致分布偏移，C‑N/C‑S指标的临床意义有限；证据目标在同一验证集上多次调参，存在过拟合与评估偏倚；未评估生成解释文本的临床可用性；整体实验规模有限，缺乏跨数据集和跨任务的验证。

---

## 629. FuseReg: Regularizing Layer Fusion Mitigates the Reconstruction-Generation Gap in Representation Autoencoders

**arXiv ID:** 2609.31620 | [PDF](https://arxiv.org/pdf/2609.31620v1)

**作者:** Hongyang Du `[一作]` (USC PSI Lab), Yue Wang `[通讯]` (USC PSI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在代表性自动编码器（RAE）中引入随机层融合正则化，训练解码器和扩散生成器对不同编码器层组合保持鲁棒性；

**💡 创新点**

创新点是将层融合视为训练分布，通过随机采样子集保持全层均值，显式惩罚跨层不一致，从而缩小重建-生成差距；

**🔧 技术方法**

使用了冻结的DINOv3-L视觉编码器、ViT解码器、DiT扩散模型、随机子集采样正则化、线性平方误差理论分析和多指标评估；

**📊 数据集**

主要在ImageNet-256数据集上实验；

**📈 对比分析**

相较于传统的固定层融合（如RAEv2），随机融合的解码器在全层、稀疏和单层融合下均能保持高PSNR（最高27.5dB），且在不改变生成器的情况下，解码器替换即可将gFID从3.01降至2.21；联合正则化后，DiT-Base gFID从13.96降至9.93，IS提升；

**⚠️ 局限性**

局限包括仅在ImageNet-256 + DINOv3-L + DiT-Base/XL 上验证，需分别调节两阶段的drop率，理论假设为线性平方损失，未验证在其他编码器或更大分辨率下的迁移效果。

---

## 630. New LoRA Skills Should Read but Never Write

**arXiv ID:** 2609.31600 | [PDF](https://arxiv.org/pdf/2609.31600v1)

**作者:** Zeyan Li `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 READ 作为一种新颖的 LoRA 适配器组合方法，能够在不重新训练旧适配器的情况下逐步添加新技能，并将所有更新折叠成单个权重变化。

**💡 创新点**

创新点在于两方面：①通过将每个适配器重写为平衡的规范坐标（canonical form）消除因子化自由度；②采用单向（只读）耦合，使新技能只能读取旧技能的输入子空间而不能写回，保证旧技能不被破坏。

**🔧 技术方法**

使用的技术包括 LoRA 低秩分解、薄 QR 分解与奇异值分解实现规范化、基于耦合矩阵 G 的增量训练以及最终将 B·G·A 直接合并到基础权重中。

**📊 数据集**

在 Llama‑3.2‑3B‑Instruct 与 Qwen3‑4B 两个 3‑10B 级模型上，使用 GLUE、SuperGLUE、Domain 以及 BBH 四个基准集，对每个任务训练独立的 LoRA 适配器。

**📈 对比分析**

与 14 种可折叠的融合方法（如 RegMean、LoRA‑LEGO、LoRAHub 等）进行对比。READ 在 32 条终端线性组合中平均提升 +0.073（95% CI +0.047 ~ +0.101）分数，且在 92 条顺序追加中有 72 条满足预注册可靠性规则，说明其在保持旧技能的同时能够有效融入新技能。

**⚠️ 局限性**

局限性主要集中在 BBH 数据集上出现的采集失败（大部分是新技能无法充分学习任务信号），以及目前仅支持最多 6 个技能的规模，耦合矩阵随技能数平方增长。

---

## 631. Generate, Track, Improve: Perceptive Multi-Skill Humanoid Locomotion with RL-Fine-Tuned Motion Generators

**arXiv ID:** 2609.31577 | [PDF](https://arxiv.org/pdf/2609.31577v1)

**作者:** Zachary Olkin `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作提出了基于原始深度图像的双层感知控制架构，包含运动生成器与跟踪策略，并通过离线强化学习微调提升了多技能人形机器人在真实世界中的通用性；

**💡 创新点**

核心创新是将优势加权回归（AWR）与结构化搜索相结合的离线强化学习微调循环，使运动生成器在新地形和技能组合上显著提升；

**🔧 技术方法**

采用感知流匹配运动生成器、控制引导的深度强化学习跟踪策略、优势加权回归、动态优化的人类运动数据与深度图像感知；

**📊 数据集**

利用动态优化的人类运动数据生成的地形一致运动片段库作为训练与评估数据集；

**📈 对比分析**

通过与传统在线残差微调对比，离线AWR微调后地形通行成功率提升25%，技能选择准确度提升80%，并在室内外实验中实现走路、跑步、站立、跳箱、爬楼梯等多种动作；

**⚠️ 局限性**

局限包括手工设计的奖励函数不一定适用于所有动作、仅使用深度信息限制语义判断、需要手动调节超参数和人类选择参考运动，导致方法难以完全扩展到任意运动。

---

## 632. GraphWrit3R: End-to-End 3D Scene Graph Writing

**arXiv ID:** 2609.31595 | [PDF](https://arxiv.org/pdf/2609.31595v1)

**作者:** Luka Milivojevic `[一作]` (Sofia University St Kliment Ohridski), Danda Pani Paudel `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从点云或高斯球面输入直接生成完整3D场景图，输出结构化JSON，无需任何中间显式表示；

**💡 创新点**

提出单一端到端潜在模型并引入多模态编码器和体素对比对齐损失，使得不依赖真值框和专有模型即可高效生成场景图；

**🔧 技术方法**

使用Sonata与Chorus稀疏体素变换器、共享PTv3-M2骨干、对比对齐损失以及Qwen2.5-0.5B LLM，并以SpatialLM预训练为基础；

**📊 数据集**

在3DSSG基准上以SceneVerse+SceneSplat++训练、ScanNet做跨域验证；

**📈 对比分析**

与ConceptGraphs、Open3DSG、RelationField、ReLaGS等基线对比，采用Recall@K指标，在不使用真值框的条件下实现SOTA召回率且推理速度显著提升；

**⚠️ 局限性**

受LLM上下文窗口限制导致大场景JSON截断，3DGS重建质量差异导致多模态融合偏向点云，并且对象检测仍是三元组召回的瓶颈。

---

## 633. From Source Code to Network Profile: Automated and Traceable MUD Profile Generation for IoT Devices

**arXiv ID:** 2609.31594 | [PDF](https://arxiv.org/pdf/2609.31594v1)

**作者:** Alessandro Lotto `[一作]` (University of Padua), Mauro Conti `[通讯]` (University of Padua)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于源代码的工具，自动生成 IoT 设备的 MUD 文件，并提供完整的可追溯性报告

**💡 创新点**

创新点在于将静态提取、检索驱动的 LLM 推理与确定性验证相结合，实现从源码直接恢复网络行为、生成可追溯的 MUD 规则，并能检测与纠正语义错误

**🔧 技术方法**

使用了轻量级静态与语法分析、向量检索、OpenAI LLM（GPT‑4/3.5）推理、规则编译与结构校验等技术

**📊 数据集**

评测数据集为自构造的 Linux 核心 + curl + wget + OpenSSH 代码库（约 1.4 GB）

**📈 对比分析**

通过与流量驱动的 MudGee 对比，源代码方法在 9 h 内完成，准确率 100%，而流量方法在有限监控窗口下难以覆盖罕见或条件触发的通信，性能显著优于传统流量驱动方案

**⚠️ 局限性**

局限性：依赖可遍历的源码树，无法处理闭源或动态生成的组件；对多语言、嵌入式环境支持有限；LLM 推理结果不完全可复现

---

## 634. Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer

**arXiv ID:** 2609.31587 | [PDF](https://arxiv.org/pdf/2609.31587v1)

**作者:** Md Shohel Arman `[一作]` (Daffodil International University), Igor Molybog `[通讯]` (University of Hawai`i at M	extbackslash=anoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估一种通过自然语言描述文件来替代代码上下文的文档生成方法，并验证其在代码补全任务中的有效性。

**💡 创新点**

提出了“回溯基准（roundtrip benchmark）”，通过将代码描述转化为代码并用原始单元测试评估复原精度，以此直接衡量描述的完整性与压缩性，并用该基准驱动描述生成提示的自动优化。

**🔧 技术方法**

使用大语言模型（Gemini 3.5‑flash、Gemini 3.1‑Pro 等）完成描述生成、代码重构与评估；通过自动化提示优化循环提升描述质量；在多模型、多仓库上执行 SWE‑bench、SWE‑ContextBench 等评测。

**📊 数据集**

数据集主要来自 SWE‑bench Verified 子集（共 11 个文件级 fixture）以及手工构造的 3 个测试文件，外部评测使用 12 个仓库的 99 个相关任务。

**📈 对比分析**

与传统的 OpenWiki 文档生成工具对比，回溯基准生成的描述在代码复原精度上显著更好；在源文件被隐藏时，优化后的描述将问题解决率从 0.08 提升到 0.71；但在源文件可用时，任何形式的文档（静态压缩、完整描述或检索上下文）都无法提升或甚至略逊于仅给出问题文本的模型，差异不显著。

**⚠️ 局限性**

限制包括：回溯基准仅覆盖 11 个文件级例子，无法全面代表大型仓库；描述生成与代码复原对模型的依赖较高，低能力模型表现不佳；负向结果表明文档在源可见时可能成为干扰；实验中未充分探索不同编程语言、不同上下文长度对结果的影响。

---

## 635. Trust Guided Decision Transformer

**arXiv ID:** 2609.31586 | [PDF](https://arxiv.org/pdf/2609.31586v1)

**作者:** Chainesh Gautam `[一作]` (International Institute of Information Technology), Kameshwaran Sampath `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Trust Guided Decision Transformer (TGDT)，在决策时先用下一状态预测误差判断上下文可靠性，再通过冻结的 IQL critic 选取最佳动作，从而提升 Decision Transformer 在长回放中的表现。

**💡 创新点**

创新点在于：①在动作价值排序之前加入上下文可靠性过滤；②利用外部/内部状态预测误差与 split conformal 估计的可靠性阈值实现可信度筛选；③将行为正则化与残差动作头结合，使动作保持在离线数据分布内；④把可信度判定与值评估分离，避免在不可靠上下文中误用高价值预测。

**🔧 技术方法**

核心技术包括：冻结的 IQL critic、行为正则化残差动作头、内部下一状态预测头、滚动预测误差统计、split conformal 可靠性阈值、上下文长度自适应筛选规则。

**📊 数据集**

使用 D4RL benchmark 中的 Maze2D、AntMaze、MuJoCo locomotion、Adroit、Kitchen 等任务（包含稀疏奖励、连续控制、精细操作等多种环境）。

**📈 对比分析**

与 vanilla Decision Transformer、上下文重置、critic‑only 选择以及多种离线 RL 与序列模型基线（BEAR、BCQ、CQL、IQL、MoRel、EDAC、VDT 等）进行对比。TGDT 在长回合任务（如 Maze2D、AntMaze）获得显著提升，整体在 D4RL 上取得最优或次优性能，并显著减少预测误差持续段，提升平均返回。

**⚠️ 局限性**

局限性：①可靠性阈值来自离线 teacher‑forcing 数据，非交换性下无正式覆盖保证；②实验仅在状态空间进行，图像观测或非平稳环境的适用性未知；③需要额外 |ℒ| 次前向传播，增加计算开销，对低延迟场景需权衡。

---

## 636. Configuration, Not Conscience: A Large-Scale Empirical Study of LLM System Prompts

**arXiv ID:** 2609.31575 | [PDF](https://arxiv.org/pdf/2609.31575v1)

**作者:** Constantinos Patsakis `[一作]` (University of Piraeus), Efthymios Alepis `[通讯]` (University of Piraeus)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性分析了四个社区仓库合并得到的407份系统提示文本，探究其词汇分布、功能模块、跨厂商复制与结构相似度以及版本演化与维护成本。

**💡 创新点**

创新点在于：①将泄露的系统提示视为配置文件而非价值声明，量化运营与安全词占比；②区分文字复制、语义相似与结构相似三种不同的跨厂商“复制”模式；③利用官方发布对照组量化泄露文本的规模放大与安全内容倒置；④将提示维护问题视为软件工程问题，提出版本控制、回归测试与反老化检查的必要性。

**🔧 技术方法**

采用词频统计、TF‑IDF余弦相似度、5‑gram Jaccard、MiniLM句子嵌入、结构骨架（标记序列）比较、正则规则检测（工具/协议、安全政策、法律、注入防御等）以及版本差分与外部工具PromptLint/Textstat等多种技术。

**📊 数据集**

使用的数据集为407份系统提示（共62家厂商），其中31份为Anthropic官方发布的提示作为匹配对，来自四个社区仓库的代码文件。

**📈 对比分析**

通过对比词汇占比、相似度指标与版本差分，发现泄漏文本平均比官方提示长6.6倍，安全条款比例下降到约9%，工具/协议条款上升至57%；跨厂商文字复制率低（<10%），但结构相似度高；官方版本链每步变化约1.3k词，泄漏链则高达15k词，显示泄漏版本漂移显著。

**⚠️ 局限性**

限制包括：数据来源主要为泄漏文件，真实性与完整性难以保证；正则检测存在噪声与漏检；人工标注单一编码，缺乏交叉验证；官方版本链仅包含Anthropic，难以推广；未评估提示差异对实际模型行为的影响。

---

## 637. User Model Extraction via Belief Self-Distillation

**arXiv ID:** 2609.31603 | [PDF](https://arxiv.org/pdf/2609.31603v1)

**作者:** Ali Holmov `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Belief Self-Distillation (BSD)，一种可读写的框架，用来从 LLM 的自然对话中提取并操纵其隐式用户模型。

**💡 创新点**

将线性探测与因果干预统一在同一压缩向量上，既可读取模型对用户的推断，又能通过写回该向量直接改变模型行为，且发现不同模型共享用户表示几何。

**🔧 技术方法**

冻结教师与学生模型，利用低秩读取投影 A 与写入投影 B 进行自蒸馏，基于多项选择问题提取信念分布并用 KL 散度训练；对结果进行线性探测、对抗激活干预 (CAA) 与跨模型 CKA 对齐。

**📊 数据集**

WildChat（约27k 真实对话）+ WildJailbreak（约31k 对抗提示）合计约58k 训练样本；测试集7k；此外使用 StrongREJECT、Alpaca 等作为评估素材。

**📈 对比分析**

与传统线性探测、显式提示的 RBP 基线比较；在三大模型上压缩至128维的用户向量能保留 >96% 的可解码信息，激活干预效果提升 1.5–4 倍（如 Llama‑3.1 78% 对比 21%），在安全测试中将拒绝率从 98% 降至 62%。

**⚠️ 局限性**

依赖预先设定的属性集合和多项选择的信念提取；仅在单层、同规模模型上验证；对长交互、开放式信念的发现尚未实现；写入接口需要白盒访问，可能带来安全风险。

---

## 638. AgentWorld: Benchmarking Long-Horizon Collaboration of Multi-agent LLMs

**arXiv ID:** 2609.31590 | [PDF](https://arxiv.org/pdf/2609.31590v1)

**作者:** Raphael Shu `[一作]` (OpenAgents), Rui Zhang `[通讯]` (Penn State University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AgentWorld，基于 MMORPG 仿真环境的长时序多智能体协作基准；

**💡 创新点**

创新点在于：① 引入黑盒交互、异质角色、长时间任务（25–55 轮）；② 提出 Causal Collaboration Effectiveness (CCE) 图形化因果协作度量；③ 公开 100 任务及 100 变体，支持多模态评估；

**🔧 技术方法**

使用 LLM 代理（Gemini 3 Flash、Claude Haiku 4.5、GPT‑5 Mini、DeepSeek R1‑70B）与 13 种高级 API 工具交互，构建因果行动图；

**📊 数据集**

数据集为 100 人工标注主任务与 100 LLM 生成变体，涵盖 8 类（战斗、制作、采集、交易、探索、求生、建造、协调）；

**📈 对比分析**

与随机、单一代理、无沟通、共享计划等基线对比；最佳模型 Gemini 3 Flash 任务成功率 52.0%，CCE 0.320；其它模型性能更低；

**⚠️ 局限性**

局限包括：① 任务仍以人工方式设计，缺乏完全自动化；② 仅评估文本聊天与高级 API 交互，未涉及低层物理控制；③ CCE 依赖 LLM 判定，虽稳定但仍受模型偏差影响；

---

## 639. Common-Mode Collapse and Recovery in Direct Feedback Alignment

**arXiv ID:** 2609.31589 | [PDF](https://arxiv.org/pdf/2609.31589v1)

**作者:** Varun Reddy `[一作]` (Harvard University), Houman Safaai `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21`

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

## 640. Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency

**arXiv ID:** 2609.31619 | [PDF](https://arxiv.org/pdf/2609.31619v1)

**作者:** Parsa Hosseini `[一作]` (University of Maryland), Nima Chitsazan `[通讯]` (AI Foundations, Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用自监督的置信度预测对推理模型进行微调，使模型在生成答案时更加高效

**💡 创新点**

不直接优化推理长度或停止，而是通过学习中间推理状态下的置信度（仅依据模型自身概率）来间接提升推理效率

**🔧 技术方法**

自监督置信度标注、基于交叉熵的置信度预测损失、对抗性多轮生成与微调

**📊 数据集**

600道AIME数学推理题（训练集）以及AIME2025、GSM8K、GPQA‑Diamond、LiveCodeBench、HumanEval等多领域基准（验证与测试）

**📈 对比分析**

与基线模型、DEER（早停）、A&Z、On‑Policy SFT等相比，ConfSFT在Gemma、Qwen、Nemotron、GPT‑OSS四大模型族上均保持准确率不变，生成token平均下降约10‑20%（最高约25%），与显式效率优化方法效果相当或更优；相较于DEER，虽然DEER可获得更大token压缩，但往往伴随显著准确率下降，ConfSFT在保持准确率的同时实现显著效率提升

**⚠️ 局限性**

仅在推理阶段不改变生成策略，依赖文本标记抽取中间状态，需手动设置或可能受模型内部结构影响；在非数学推理任务的泛化仍有待进一步验证；对极端长推理或多步推理的适应性尚未系统评估

---

## 641. Gap-free Differentially Private PCA for Gaussian Data

**arXiv ID:** 2609.31614 | [PDF](https://arxiv.org/pdf/2609.31614v1)

**作者:** Alina Ene `[一作]` (Boston University), Huy L. Nguyen `[通讯]` (Northeastern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种在高斯数据上实现差分隐私的主成分分析算法，利用剪裁与加噪的私有幂迭代方法，在无谱间隙条件下恢复近似最优一维子空间。

**💡 创新点**

创新点在于：①提供了无谱间隙（gap‑free）的DP PCA方案；②通过私有剪裁与一次性噪声实现更紧的样本复杂度；③利用对称性与马尔可夫过程证明的潜能函数收敛。

**🔧 技术方法**

技术上主要使用了：差分隐私的高斯机制与RDP/α-Δ-δ 转换；高斯浓度与随机矩阵的经验协方差近似；以及针对噪声的马尔可夫/Doob不等式分析的噪声功率迭代。

**📊 数据集**

实验数据集为合成的独立同分布高斯样本（N(0,Σ)），其中Σ的主特征值λ₁已知近似值Λ。

**📈 对比分析**

与已有的需要谱间隙或采用更粗方法的DP PCA算法对比，本文在样本复杂度为Θ(log d·√{d·log(1/δ)}/α + d + 1/α²) 的情况下，以常数概率实现误差O(αλ₁)，表现出更低的样本需求与更好的误差上界。

**⚠️ 局限性**

局限性包括：需要先知λ₁的常数倍近似；仅适用于高斯分布，非高斯数据效果未知；在高维下对δ的要求相对严格，且成功概率仅为常数级。

---

## 642. Learning Robot Policies from Sparse Success Signals via STL-Guided Stein Variational Policy Gradient

**arXiv ID:** 2609.31606 | [PDF](https://arxiv.org/pdf/2609.31606v1)

**作者:** Hongrui Zheng `[一作]` (University of Pennsylvania), Rahul Mangharam `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于信号时序逻辑（STL）鲁棒性梯度与Stein变分梯度下降（SVGD）的低层控制策略学习框架STL‑SVPG，用于稀疏成功信号任务的训练与部署。

**💡 创新点**

创新点在于将全轨迹STL鲁棒性直接通过可微分动力学传播给策略梯度，利用SVGD构建参数种群实现信息共享与多样性，从而在无成功经验时即可获得任务级反馈，克服传统稀疏奖励难以学习的问题。

**🔧 技术方法**

使用了可微分STL鲁棒性（LogSumExp平滑）、MuJoCo MJX + JAX实现的自动微分仿真、Stein变分梯度下降、基于核的参数共享与排斥、以及域随机化与真实机器人部署（Crazyflie与Franka Panda）。

**📊 数据集**

实验数据集包含六个机器人任务（Crazyflie Unlock、Random Cue、Ping‑pong、Pillar Pull、Panda Random Cue、Panda Ping‑pong），每个任务使用1000条隐藏初始条件进行评估，并在物理平台上收集随机起始条件下的成功率。

**📈 对比分析**

与PPO、SAC、TD‑MPC2等强化学习基线比较，采用稀疏成功率作为评价指标。STL‑SVPG在五个任务中均获得最高平均成功率，在需要长时间序列、接触交互和逻辑顺序的任务上显著优于基线；在实测中实现了从仿真到真实机器人的成功传输。

**⚠️ 局限性**

局限性在于依赖可微分动力学模型和准确的STL鲁棒性估计；对接触不确定性和多事件信用分配处理不足，导致在复杂接触任务（如Ping‑pong）中未能完整完成；以及需要更完善的域随机化与鲁棒性以提升真实环境下的泛化能力。

---

