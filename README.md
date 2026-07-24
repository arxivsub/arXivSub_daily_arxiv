# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-24 | 今日论文总数: 560

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. A Knowledge-Injection Framework for Zero-Shot Adaptation of LLMs to Delirium Prediction

**arXiv ID:** 2607.20453 | [PDF](https://arxiv.org/pdf/2607.20453v1)

**作者:** Jessica Sena `[一作]` (University of Florida), Parisa Rashidi `[通讯]` (University of Florida)

**通讯引用:** 11430 | [OpenAlex ID](https://openalex.org/A5007040136)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

在 ICU 痴呆预测中提出了在推理时注入外部医学知识的轻量级框架，避免了模型微调与检索，利用结构化 EHR 摘要与预生成的知识报告进行零样本预测。

**💡 创新点**

创新点在于仅通过推理时的知识注入，保持模型权重不变，同时发现不同规模模型对知识内容与结构的敏感性，短版报告比长版更有效。

**🔧 技术方法**

使用 LLaMA 3.x 开源语言模型（8B 与 70B）与 SHAP 归因分析、统计检验等技术评估性能。

**📊 数据集**

数据集为 MIMIC‑IV 中 3,160 个 ICU 入院记录，平衡后包含 1,580 例痴呆与 1,580 例非痴呆。

**📈 对比分析**

通过 AUROC 与 DeLong 检验与 GPT‑5.2 参考模型对比，8B 版在注入知识后 AUROC 提升 8.57 点（从 53.20% 提升至 61.77%），70B 版提升 1.99 点；与 GPT‑5.2 的差距从 15.66% 缩小至 7.09%。

**⚠️ 局限性**

局限包括单机构回溯性验证、知识报告的生成依赖外部来源、对不同任务和医院的泛化尚未验证、模型对提示敏感且需人工干预。

---

## 2. Enabling Scalable Topology Inference in Distribution Systems via Constrained Multi-Source Inference

**arXiv ID:** 2607.20480 | [PDF](https://arxiv.org/pdf/2607.20480v1)

**作者:** Haoran Li `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于多源信息和约束引导的可扩展配电网拓扑重构框架，兼顾空间、测量与运营约束；

**💡 创新点**

创新点在于将拓扑重构建模为约束推理问题，采用局部检测与重连、可靠性估计与伪造测试相结合，实现大规模、可扩展且可信的拓扑修正；

**🔧 技术方法**

使用地理信息系统(GIS)坐标、AMI电压/负荷时间序列、故障日志等多源数据，并结合电压相关性、空间距离、变压器容量约束及聚类与互信息等数据驱动方法；

**📊 数据集**

利用与美国大型配电公司合作的三条配电馈线数据，覆盖约8,000台AMI计量，包含节点与变压器的空间坐标、相位信息和15分钟间隔的电压/负荷记录；

**📈 对比分析**

与传统单纯电压相关性和数据库基础方法比较，所提框架在95%以上的拓扑重构精度、显著降低计算量，并在物理约束下消除过载情况；

**⚠️ 局限性**

局限性包括对候选变压器邻域大小的依赖、对测量完整度的敏感性以及在极端数据缺失或大规模拓扑错误时性能可能下降。

---

## 3. The Storyteller in the Model: Narrative Pattern Inheritance, Escalation Dynamics, and Alignment Governance in LLMs

**arXiv ID:** 2607.20449 | [PDF](https://arxiv.org/pdf/2607.20449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 4. SenCos-GEM: SENet-Calibrated and Law-of-Cosines-Constrained Geometry-Enhanced Molecular Representation for Property Prediction

**arXiv ID:** 2607.20551 | [PDF](https://arxiv.org/pdf/2607.20551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 5. DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions

**arXiv ID:** 2607.20469 | [PDF](https://arxiv.org/pdf/2607.20469v1)

**作者:** Zishan Shao `[一作]` (Duke University), Hai Helen Li `[通讯]` (Duke University)

**通讯引用:** 200028 | [OpenAlex ID](https://openalex.org/A5052819678)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并验证了LLM解码时跨任务共享的低维子空间（DecodeShare），通过仅在解码阶段投影移除来评估其因果影响。

**💡 创新点**

首次在KV缓存推理下从解码状态识别共享子空间并进行解码时因果干预，揭示预填充估计与解码实际行为的不匹配。

**🔧 技术方法**

使用PCA+阈值筛选、投影干预、匹配预算对照以及KV缓存解码技术。

**📊 数据集**

在多任务基准上测试，包括算术、常识、知识问答、验证与物理推理等，并覆盖不同模型层级。

**📈 对比分析**

相较于预填充或随机子空间，移除共享子空间导致准确率显著下降，证明其高因果杠杆；对比基准下性能提升明显。

**⚠️ 局限性**

需要白盒访问、对模型内部状态进行估计，且对预填充估计的转移可能不稳定；实验受限于所用模型与数据集。

---

## 6. Optimizing Hypergraph-Based RAG: Toward Better Fact Extraction and Chunk Retrieval

**arXiv ID:** 2607.20506 | [PDF](https://arxiv.org/pdf/2607.20506v1)

**作者:** Houda Khrouf `[一作]` (Qlik), Sebastiao Correia `[通讯]` (Qlik)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在 HyperGraphRAG 基础上，提出 EXT++ 自一致性提取方法和基于个性化 PageRank（PPR）的检索策略，构建更完整、连通性更强的 n‑ary 知识超图，并通过结构传播提升检索质量。

**💡 创新点**

创新点：① EXT++ 通过自一致性 prompting 在单个 LLM 调用中完成多轮提取并取并集，显著降低孤立 hyperedge 并提升实体连通度；② 在超图上使用 PPR 进行全局传播，克服传统检索的 horizon 问题，优先选取结构上高度相关的块。

**🔧 技术方法**

技术手段：自一致性 prompting、超图（Hypergraph）构建、个性化 PageRank、向量检索、节点打分与 Noisy‑OR 归一、基于实体/超边的分层检索、GPT‑4o‑mini 进行抽取与生成、NanoVectorDB 进行向量存储与检索、NetworkX 处理超图。

**📊 数据集**

使用数据集：① Fiction（UltraDomain 章节子集，84 题）；② CS（计算机教材原始数据集，398 题）；③ MAUD（LegalBench‑RAG 合并收购协议子集，74 题）。

**📈 对比分析**

评估方式：与 Standard RAG、Anthropic Contextual Chunking、HippoRAG2、GraphTransformer 及原 HyperGraphRAG baseline 进行对比，指标包括 Contextual Recall、Correctness、Completeness。实验显示，在三类数据集上，HyperGraphRAG+PPR+EXT++ 在 Contextual Recall、Correctness、Completeness 上均取得最优或次优结果，Fiction 上 Contextual Recall 提升 51%，MAUD 上提升 69%，Completeness 提升 11%。

**⚠️ 局限性**

局限性：① 在 MAUD 法律文本中 Contextual Recall 仍低于 HippoRAG2，因 hyperedge 语义冗余导致相似度噪声；② 依赖 LLM 进行超图抽取，仍存在提取成本与延迟；③ 当前仅使用标准 PPR，未探索更适合超图的 diffusion 方法；④ 对高词频冗余语料的处理仍需改进。

---

## 7. Position: Natural Language Should Not Fully Replace Formal Languages

**arXiv ID:** 2607.20432 | [PDF](https://arxiv.org/pdf/2607.20432v1)

**作者:** Eitan Wagner `[一作]` (Hebrew University Of Jerusalem), Omri Abend `[通讯]` (Hebrew University Of Jerusalem)

**通讯引用:** 1890 | [OpenAlex ID](https://openalex.org/A5059068945)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出自然语言与正式语言在任务规格化中的互补性，构建信息理论框架并证明特异性交叉点；

**💡 创新点**

创新性定义任务特异性与翻译差距，证明存在阈值使自然语言不再优于正式语言，并通过跨模态实验验证；

**🔧 技术方法**

使用信息理论（互信息、KL 散度、熵）、约束满足、Rational Speech Acts 模型以及对大型语言模型的实验；

**📊 数据集**

采用 Midjourney 交互日志、Gemini‑3 生成图片、HumanEval、SWE‑bench 代码评测以及公开文本、音频、图像数据集；

**📈 对比分析**

通过比较提示长度、生成成功率与特异性阈值评估三种方案，结果显示低特异性任务自然语言更高效，高特异性任务正式语言更优，混合方案实现折衷；

**⚠️ 局限性**

局限在于仅考虑单次生成，忽略迭代反馈和人机交互；假设翻译差距与正式语言冗余可估计；对不同领域和语言的普适性有限；未深入评估专业成本。

---

## 8. DC-Leap: Training-Free Acceleration of dLLMs via Draft-Guided Contiguous Leaping Decoding

**arXiv ID:** 2607.20467 | [PDF](https://arxiv.org/pdf/2607.20467v1)

**作者:** Yanhua Jiao `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 63711 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不需要额外训练的前提下，提出 DC-Leap 框架，通过动态连续验证与草稿引导跳跃解码，显著加速 Diffusion 大语言模型的推理。

**💡 创新点**

将 JPDE 误差转化为局部窗口级顺序验证，允许使用更宽松的置信阈值，并通过草稿提供前瞻上下文，兼顾速度与生成质量。

**🔧 技术方法**

动态连续验证（DCV）、草稿引导解码、并行重掩码策略、阈值化置信度判定以及 KV-Cache 兼容等技术。

**📊 数据集**

GSM8K、MATH、HumanEval、MBPP、IFEval 等标准基准，使用 LLaDA-8B-Instruct、LLaDA-1.5、Dream-v0-7B-Instruct 等 dLLM 模型。

**📈 对比分析**

与 Fast-dLLM、LocalLeap、L2P 等现有并行解码方法对比，DC-Leap 在 MBPP 等任务上实现最高 53.19× 的速度提升，结合 KV-Cache 后可达 105.02×，且生成质量几乎不变。

**⚠️ 局限性**

阈值需手动设定，对极低置信度或草稿质量不足的情况可能导致错误累计；在极长序列或非 dLLM 体系下的泛化性尚待进一步验证。

---

## 9. Frontier Financial Judgement: Can agents tell what might move a stock?

**arXiv ID:** 2607.20645 | [PDF](https://arxiv.org/pdf/2607.20645v1)

**作者:** Joshua Harris `[一作]` `[通讯]`, Joshua Harris

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Frontier Financial Judgement基准，用以评估AI代理在金融新闻中新颖性、重要性和方向三维判断的能力。

**💡 创新点**

创新之处在于将人工设计的合成新闻与实时新闻混合，控制事件难度并可复现真实环境，同时同时评估误报率、成本和推理时延。

**🔧 技术方法**

采用大型语言模型与工具集成框架（Web搜索、Harbor等），通过中等推理和JSON输出对14个代理进行评估。

**📊 数据集**

使用v1数据集，包括82个专业设计的事件与656个评估条目，聚焦半导体供应链公司（如ASML、NVIDIA），混合合成与实时新闻与历史文件。

**📈 对比分析**

通过新颖性、重要性、方向的单标签准确率及全标签准确率进行比较，最佳模型GPT‑5.5的全标签准确率为52.4%，其他模型原子准确率介于54%–71%，误报率差异明显。

**⚠️ 局限性**

局限性在于合成事件难以完全再现真实新闻流中的竞争性解释，覆盖行业有限（仅半导体），且部分专业判断具有主观性，导致标签差异。

---

## 10. THOR: A Theta-Gamma Hierarchical Oscillatory Reasoning Framework for Multi-hop QA

**arXiv ID:** 2607.20459 | [PDF](https://arxiv.org/pdf/2607.20459v1)

**作者:** Ziyang Ling `[一作]` (University of Science and Technology of China), Mingzhai Sun `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2197 | [OpenAlex ID](https://openalex.org/A5055541105)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于人脑θ–γ层级振荡原理的多跳推理框架THOR，利用全局规划与局部检索的分层控制、验证与修复机制，显著降低注意力衰减和错误累积；

**💡 创新点**

创新点在于：①以θ–γ振荡为时序分层控制，分离全局计划与局部检索；②引入结构化的slot‑schema记忆与修复意识的iACC验证器；③设计基于振荡的离散状态机，实现从本地修复到全局重规划的可控跳转；

**🔧 技术方法**

技术包括：多头注意力的LLM (GPT‑3.5‑turbo/GPT‑4o 等) 作为核心，BGE‑M3/BGE‑Reranker 进行主题与上下文检索；iPFC（全局θ节拍）负责计划与帧修复；iHPC（局部γ节拍）负责检索与推理；iACC 负责三种一致性检查；离散状态机实现θ–γ交替与回溯；

**📊 数据集**

使用三大多跳问答基准：HotpotQA、2WikiMultiHopQA、MuSiQue；

**📈 对比分析**

与提示工程、检索增强、代理式等主流方法对比，THOR在EM与F1上取得最高分；在HotpotQA上EM提升至72.1/81.4（GPT‑4o），MuSiQue上F1最高52.1/57.7；在极端的四跳、对抗文档注入、检索召回等细粒度评测中，THOR在框架偏移率( FSR)与锚点偏移率( ASR)上均优于同类方法；

**⚠️ 局限性**

局限性包括：当提供的证据仅存在细微词汇或语境差异时，验证器难以检测导致错误；检索过程中弱相关证据过滤不足会导致混淆；对极其细粒度或长链推理的错误定位仍有提升空间。

---

## 11. Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating

**arXiv ID:** 2607.20481 | [PDF](https://arxiv.org/pdf/2607.20481v1)

**作者:** Evan Chen `[一作]` (Purdue University), Christopher Brinton `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了CARGO，一种训练无关的本地-云协作路由框架，利用本地模型在多种 prompt 变体下生成的回答的一致性来判断是否将任务 offload 到云端。

**💡 创新点**

创新点在于：① 用 prompt 变异而非温度采样来获得多样化回答；② 通过贝叶斯后验置信区间实现自适应早停估计回答一致性；③ 采用概率路由与在线校准，使系统可在不同协作比例下自适应且不需要额外训练路由器。

**🔧 技术方法**

技术方法包括：prompt-varied sampling、贝叶斯模式质量估计（Beta 先验与后验）、贝叶斯置信区间早停、概率路由函数（sigmoid 调整 offload 概率）以及热身阶段的 λ 校准。

**📊 数据集**

实验数据集涵盖数学推理（MATH-lighteval、GSM8K、SVAMP、MATH-500、AGIEval-Math、MinervaMath）、多选推理（ARC、MMLU）和阅读理解（SQuAD），并在多种本地 LLM（Qwen2.5-3B/7B、Phi-3-mini、Llama-3.2-3B）上评估。

**📈 对比分析**

与随机 offload、自信度路由、CoT 步骤路由以及监督学习路由进行对比；在固定协作比例 0.3 时，CARGO 在所有模型-数据组合上均优于无监督基线，并在多处超过监督路由；在不同协作比例下亦保持最优或接近最优性能。

**⚠️ 局限性**

局限性包括：当本地模型的内在一致性信号弱时（如 ARC 任务）提升有限；需要多次生成导致额外推理延迟；对多模态或更复杂资源预算（如 token 预算）场景的适用性尚待进一步验证。

---

## 12. Inducing Comparability of Factorised Probability Distributions

**arXiv ID:** 2607.20502 | [PDF](https://arxiv.org/pdf/2607.20502v1)

**作者:** Jan Speller `[一作]` (University of Münster), Tanya Braun `[通讯]` (University of Münster)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出一种最小结构 Laplace 扩展（msx）来使任意两张因子图（factor graph）在不相同变量集上可比较，并保留原始概率分布的投影。

**💡 创新点**

创新点：1）将因子图通过统一（Laplace）扩展映射到同一可测空间，保证投影保持概率分布；2）给出确定性算法生成最小结构扩展；3）为局部因子级比较提供理论基础。

**🔧 技术方法**

使用的技术：因子图建模、Laplace（均匀）扩展、可测空间投影、分区函数的解析表述、结构匹配与最小化扩展算法。

**📊 数据集**

未使用具体数据集；论文为理论方法论，未给出实验数据或真实数据集。

**📈 对比分析**

比较方法：在扩展后的同一可测空间上可直接使用 KL、总变差、Wasserstein、Hellinger 等全局距离；局部可通过因子潜力差异衡量。性能方面未给出实验结果，主要通过理论证明其保真性和最小性。

**⚠️ 局限性**

限制：1）需要因子图中因子范围唯一；2）扩展后仍不考虑原因子语义差异，可能影响可解释性；3）对分区函数的影响尚需进一步定量化；4）扩展过程在极端模型可能产生较大扩展量。

---

## 13. Skill-Contracted Agents for Evidence-Aware Materials Literature Analysis

**arXiv ID:** 2607.20431 | [PDF](https://arxiv.org/pdf/2607.20431v1)

**作者:** Bixuan Li `[一作]` (Tianmushan Laboratory), Lei Zheng `[通讯]` (Tianmushan Laboratory)

**通讯引用:** 9961 | [OpenAlex ID](https://openalex.org/A5074843268)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 AlphaAgent，一种基于技能契约的材料科学文献分析框架，将检索式问答与全文报告生成拆分。

**💡 创新点**

创新点在于：①检索意图重写与迭代检索；②通过技能契约实现检索与生成的显式边界；③在答案生成阶段严格使用检索结果，提升可信度。

**🔧 技术方法**

使用技术包括大语言模型（LLM）、检索增强生成（RAG）、检索意图重写、技能契约驱动的工作流、PDF 解析与结构化报告生成。

**📊 数据集**

数据集为 30 万余篇来自 Journal Citation Reports Metallurgy 与 Metallurgical Engineering 目录的文献索引。

**📈 对比分析**

与基线 RAG、GPT‑5.5、Kimi‑K2.6 对比，AlphaAgent 在 40 题目信息检索与分析上平均得分 4.66（深度分析）/4.46（概念问答），比基线高 0.61（深度）和 0.38（概念），显著提升了机制解释与可信度。

**⚠️ 局限性**

局限性包括：仅在精心构建的索引内检索，无法覆盖所有公开论文；对 PDF 解析错误或缺失可能导致报告生成失败；评测样本有限，需进一步验证在更广泛任务上的鲁棒性。

---

## 14. MiniCache: Reusable Program Caching with Small Model Interfaces for Efficient LLM Inference

**arXiv ID:** 2607.20507 | [PDF](https://arxiv.org/pdf/2607.20507v1)

**作者:** Jingquan Chen `[一作]`, Yong Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

实现了一种基于可重用程序缓存的LLM推理优化框架，将程序思维转化为可参数化缓存对象，减少大型模型调用。

**💡 创新点**

创新点在于统一利用小模型同时完成语义变量提取与投草式推理，从而实现缓存命中加速与生成成本降低。

**🔧 技术方法**

使用了小模型进行语义变量抽取、投草式解码、程序化推理与缓存生成，并结合 Qwen3-32B 作为目标大型模型和 Qwen3-1.7B 作为辅助小模型。

**📊 数据集**

在购物式请求集（Shopping‑Full、Shopping‑Struct）、WebShop、金融推理数据集 Formula 以及 CodeTAT‑QA 等数据集上进行实验。

**📈 对比分析**

与 Direct LLM、PoT、ExactCache、GPTCache、GenCache 等基线对比，取得约 3.1× 延迟加速、约 2.8× 吞吐量提升，并在准确率上与 PoT 相当。

**⚠️ 局限性**

局限在于仅适用于结构相似的任务，受限于缓存生成成功率与小模型抽取准确性，且可能因错误缓存导致误传错误。

---

## 15. HypNO: A Graph-Based Neural Operator with Physics-Informed Message Passing for Hyperbolic Conservation Laws

**arXiv ID:** 2607.20541 | [PDF](https://arxiv.org/pdf/2607.20541v1)

**作者:** Dimitrije Ždrale `[一作]` (École Polytechnique), Hossein Nick Zinat Matin `[通讯]` (École Polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 HypNO，一种基于图神经网络的超声波算子，能够一次前向传播从初始条件直接预测全时空解。

**💡 创新点**

创新点在于将物理信息（流量、特征速度、Rankine–Hugoniot 速度、向上/下采样指示、熵和 CFL 桥接）嵌入边特征，并通过物理门控的消息传递实现冲击波捕获，替代传统的全局谱混合，显著提升了对冲击波的解析能力。

**🔧 技术方法**

采用了空间–时间稠密图结构、基于消息传递的 GNN、物理门控函数、深层监督和离散化边特征（Finite‑Volume 接口量）等技术，并与 FNO、WENO5、Godunov、HLL 等方法进行对比。

**📊 数据集**

使用从 LWR（Lighthill‑Whitham‑Richards）和 ARZ（Aw‑Rascle‑Zhang）模型中解析得到的 Lax‑Hopf 与波前跟踪解生成的合成数据集，训练样本覆盖 2、3、5、7、10 段的分段初始条件，OOD 测试使用 8、20、25、30 段等未见过的复杂度。

**📈 对比分析**

与 FNO、WENO5、Godunov、HLL 等基线比较，HypNO 在 ID 与 OOD 情况下均实现了更低的平均绝对误差（MAE）和更窄的误差分布，尤其在冲击波带内误差显著小于传统方法，ARZ 案例误差约为最优基线的 1/3–1/4。

**⚠️ 局限性**

局限性包括：目前仅在 1 维问题上验证；对更高维域的可扩展性、计算复杂度和内存占用尚未充分评估；固定邻域宽度不自适应本地波速和影响域；需进一步研究自适应网格、更多物理场的训练数据以及对更大规模工程问题的推广。

---

## 16. SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification

**arXiv ID:** 2607.20475 | [PDF](https://arxiv.org/pdf/2607.20475v1)

**作者:** Pragaash Ponnusamy `[一作]` (Together AI), Tri Dao `[通讯]` (Together AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

统一LLM采样流程，提出SonicSampler的CUDA Graph兼容内核，实现日志处理、采样与验证的垂直融合。

**💡 创新点**

两阶段分块Top‑k算法与自适应Radix‑Bitonic选择，动态位指示支持混合贪婪/随机解码，所有操作在单一CUDA Graph中完成，显著降低内核启动与内存拷贝。

**🔧 技术方法**

Triton内核、CUDA Graph、分块并行、Lexicographic编码、Gumbel‑Max、adaptive radix‑bitonic、Bitonic排序网络、Programmatic Dependent Launch。

**📊 数据集**

使用Qwen3‑8B、DeepSeek v3.1、GPQA‑Diamond等模型日志，基准评测在NVIDIA B200（及H100）GPU上。

**📈 对比分析**

与FlashInfer、TileLang‑TopK、RadiK、TRT‑LLM等基线对比，单核Top‑k延迟降低2‑5×，端到端解码速度提升15‑17%（80‑120 TPS），在不同批量与词表尺寸下保持稳定优势。

**⚠️ 局限性**

仅对top‑k≤128有效，超出时可能丢失极少概率质量；当前仅支持单步贪婪/随机采样，束搜索与极大批量下的稀疏路径仍需进一步优化。

---

## 17. Domyn-Small: A European 10B Reasoning Language Model

**arXiv ID:** 2607.20448 | [PDF](https://arxiv.org/pdf/2607.20448v1)

**作者:** Simone Angarano `[一作]` (Domyn), Martin Cimmino `[通讯]` (Domyn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Domyn‑10B，一个10B参数的开放权重推理语言模型，经过持续预训练、监督微调和多阶段强化学习后实现高效推理；

**💡 创新点**

创新点在于将RoPE基底调节与上下文扩展结合，采用多阶段RL（GRPO、DPO、跨域GRPO）提升推理、指令遵循与工具调用能力，并公开完整可复现的训练流水线与Domyn Swarm推理框架；

**🔧 技术方法**

使用技术包括Transformer decoder‑only架构、Grouped‑Query Attention、RoPE、bfloat16混合精度、持续预训练、监督微调、可验证奖励的GRPO、DPO、异步多环境GRPO、YaRN上下文扩展以及Domyn Swarm的vLLM+HPC部署；

**📊 数据集**

数据集来源于Italia‑10B基础训练数据，增补503B高质量长文（web、代码、数学、科研、跨语）进行CPT，SFT混合40+公开指令集（3.85M样本、12.29B token），RL使用DeepScaleR、DeepSeekMath等数学题、代码单元测试及多域任务，评估基准包括MATH‑500、AIME、GPQA、MMLU、IFEval、HumanEval等；

**📈 对比分析**

与同类7–10B模型（Qwen3.5‑9B、OLMo‑3‑7B‑Think、Llama‑3.1‑Nemotron‑Nano‑8B、Ministral‑3‑8B）在推理精度与token效率对比，Domyn‑10B在核心推理上token预算约为Qwen3.5‑9B的1/3，同时在HumanEval、IFEval、MMLU等多项基准上保持竞争力，展现最佳准确‑效率平衡；

**⚠️ 局限性**

局限性包括受2024年基础模型限制，硬数学、>32k长上下文和多轮工具调用仍落后于顶尖对手；多语言细粒度不佳；未进行专门安全微调，拒绝率与抗破环性能仍有提升空间。

---

## 18. Geometric Configurations of Perturbed Jailbreak Prompts

**arXiv ID:** 2607.20581 | [PDF](https://arxiv.org/pdf/2607.20581v1)

**作者:** Lynn Delcon `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了小型权重的Qwen-2.5和Llama-3系列模型在遭受字符串扰动越狱提示时的内部表示，比较了最后一层最后一个token的嵌入空间与前50个下一token概率空间。

**💡 创新点**

创新点在于揭示嵌入空间仅按拼写与模板线性分离，而概率空间是一维且与模型安全行为无关，表明嵌入分离并非安全信号。

**🔧 技术方法**

采用支持向量机、随机森林回归、GEE逻辑回归以及参与比率等技术对嵌入和概率空间进行分析与行为关联。

**📊 数据集**

使用包含96个模型专用越狱提示及其50个随机改写共38,592个示例，并对照公开的自然指令数据集。

**📈 对比分析**

通过SVM和随机森林评估分离度，结果显示对安全标签无显著区分，性能与已有研究相当但未发现新的安全方向。

**⚠️ 局限性**

受限于合规回答样本稀少、跨模型差异大以及假设独立性的偏差，导致结果对具体设计的关联性不强。

---

## 19. ClickGuard: Detecting and Spoiling Clickbait News with Informativeness Measures and Large Language Models

**arXiv ID:** 2607.20463 | [PDF](https://arxiv.org/pdf/2607.20463v1)

**作者:** Wojciech Michaluk `[一作]` (Warsaw University of Technology), Anna Wróblewska `[通讯]` (Warsaw University of Technology)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5031984813)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为ClickGuard的浏览器扩展，能够在用户点击前后检测并警告点击诱饵，同时提供简短的内容摘要。

**💡 创新点**

创新点在于将深度语义嵌入与15种语言学特征相结合的混合模型，以及在扩展中提供可解释的“诱饵度”评分和自动生成的内容spoiler。

**🔧 技术方法**

使用了OpenAI生成的1,000维嵌入、手工设计的语言特征、XGBoost分类器以及GPT‑4o‑mini进行内容摘要。

**📊 数据集**

训练与评估使用了四个公开英文数据集：Kaggle新闻标题数据（≈53,000条）、Clickbait Challenge 2017（38,830条）、SemEval‑2023 Task 5（≈4,000条）以及合并后的20,000条点击诱饵与20,000条非诱饵标题。

**📈 对比分析**

与TF‑IDF、Word2Vec、RoBERTa等基线相比，混合模型在测试集上实现了0.909的F1分数，比仅使用嵌入的模型提升约4.5个百分点，显示出更优的检测性能。

**⚠️ 局限性**

局限包括目前仅支持英文，需对语言特征进行调整才能适用于多语言场景；后端需要云服务支持，导致部署成本和延迟受限；spoiler的生成依赖GPT‑4o‑mini，可能产生内容不准确或偏见。

---

## 20. Masked Topology Modeling for Self-Supervised Learning on Parametric CAD

**arXiv ID:** 2607.20642 | [PDF](https://arxiv.org/pdf/2607.20642v1)

**作者:** Heinrich Jiang `[一作]` (StoryGold AI), Jennifer Jang `[通讯]` (StoryGold AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种自监督预训练任务Masked Topology Modeling (MTM)，通过预测B-rep面相邻图中被遮蔽边的凸性与曲线类型，提升B-rep表征学习；

**💡 创新点**

首次利用B-rep面相邻图的边标签作为自监督信号，并结合理论证明MTM能识别凸性并迫使编码器捕捉面间几何关系；

**🔧 技术方法**

采用UV-Net结构的图卷积编码器，配合MoCo对比学习、BFS连通面块遮蔽重建和MTM的边预测头；

**📊 数据集**

在公开的ABC CAD模型集以及自行生成的程序化B-rep数据集上进行预训练；

**📈 对比分析**

在Fusion 360、SolidLetters、MFInstSeg和CadSynth等四个下游基准上进行评估，MTM+MoCo在全量数据和少样本场景均优于现有SOTA，尤其在少样本时性能显著提升；

**⚠️ 局限性**

对角线类型的识别受限于采样分辨率，且预训练在非分布式数据上仍有提升空间，未来可扩展更大数据与更细粒度标签。

---

## 21. FlowEdit: Information-Theoretic Control of LLM Reasoning Flows for Ill-posed Problems Involving Conflicts

**arXiv ID:** 2607.20500 | [PDF](https://arxiv.org/pdf/2607.20500v1)

**作者:** Sizhe Tang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6543 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了FlowEdit框架，用于在含冲突的开放式数学推理问题中主动维护并输出所有合法的解答分支，避免LLM单条推理链的隐式选择；

**💡 创新点**

创新点在于把“冲突冲突问题”从检测/拒绝视角转化为信息流调节问题，利用双重条件互信息目标（信息充分性与分支分离）在模型内部显式约束每条推理流，保证各分支独立且有用；

**🔧 技术方法**

主要技术包括：信息理论约束（InfoNCE用于充分性，CLUB用于分离）+ϵ-充分性条件 + 边界嵌入作为分支标识 + 结构化的输出格式与多分支训练；

**📊 数据集**

构建了5,000条含冲突的推理题集，覆盖三大领域（Pure Math、Daily、Application）并标注每个实例的1~4个合法分支；

**📈 对比分析**

在闭源模型（Haiku、GPT‑5、Gemini）与开源模型（Qwen）上进行对比，使用“ill‑posed”提示；FlowEdit在exact‑set‑match (EM) 上比最佳闭源基线提升68%，信息恢复 (IR) 提升24%，并在多分支场景（K*≥2）表现尤为显著；

**⚠️ 局限性**

局限性：仅在分支已开始时调节内部信息流，未明确控制何时产生分支；依赖next‑token预测机制，对更大规模或不同架构的可扩展性尚未验证。

---

## 22. AINTMA: Agentic AI Architecture for Autonomous Test Management with Generative Intelligence, Secure Cloud Communication and Adaptive Quality Analytics

**arXiv ID:** 2607.20452 | [PDF](https://arxiv.org/pdf/2607.20452v1)

**作者:** Vinil Pasupuleti `[一作]`, Srinivasateja Songa `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了AINTMA——一个由六个专门化智能代理组成的多智能体架构，用于自动化测试管理、风险评估、强化学习优先级排序、执行编排、生成质量叙述和零信任云安全。

**💡 创新点**

创新点在于：①将强化学习与风险评估信号结合，显著提升测试优先级的准确性；②利用大型语言模型生成可读的质量报告和测试建议，满足不同角色的需求；③在代理间实现完全零信任的加密事件网格，并提供可审计的多租户隔离；④在真实企业环境中持续运行18个月，验证系统的可扩展性和 ROI。

**🔧 技术方法**

使用技术包括：多智能体协作框架、消息队列事件网格、OAuth2/JWT 零信任身份验证、TLS 1.3 加密、XGBoost 与多层感知机风险预测、强化学习 Q‑学习（含状态向量512维）以及 GPT‑4/Claude 等大型语言模型。

**📊 数据集**

数据集来源于12个真实项目的CI/CD执行历史（总计约76,500个测试用例、4.2百万次执行），并包含代码变更、覆盖率、历史运行结果等特征，形成47维风险特征与12维变更特征的训练集。

**📈 对比分析**

通过与Jira+Xray、TestRail、Zephyr Enterprise以及随机排序等基线比较，AINTMA在APFD上平均达88.4%，比随机51.2%、最优商业工具82.1%高出约6个百分点；循环时间缩短43%，缺陷逃逸率从8.3%降至2.1%；ROI约340%，9个月内收支平衡。

**⚠️ 局限性**

局限性包括：冷启动阶段需要数月的CI历史和强化学习收敛；依赖外部LLM API，对隔离环境不适用；APFD未考虑缺陷严重性；安全和多租户实现对硬件/网络条件敏感；实验仅在单一大型企业中进行，外部可推广性待验证。

---

## 23. CAMeR: Keyword-Gated Hybrid Activation for Adaptive Memory Retention in LLM Agents

**arXiv ID:** 2607.20458 | [PDF](https://arxiv.org/pdf/2607.20458v1)

**作者:** Haowen Lai `[一作]` `[通讯]`, Haowen Lai

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于关键字门控的混合激活机制（CAMeR）来实现LLM代理的自适应记忆保留。

**💡 创新点**

创新点在于将词级关键词重叠与全文本嵌入余弦相似度相结合的门控策略，显著提高记忆区分度并减少无用记忆。

**🔧 技术方法**

使用KeyBERT/YAKE/TF‑IDF提取关键词，all-MiniLM‑L6‑v2嵌入，关键词与余弦相似度混合门控，权重衰减+强化更新，以及基于权重的检索排名。

**📊 数据集**

采用自建的76‑memory、100‑round、8主题簇的受控实验数据集（称为CMR Benchmark）。

**📈 对比分析**

与Memory‑R1、Oblivion、SuperLocalMemory、Full History、No Memory等基线比较，CAMeR在scissors gap上提升1.6×，token消耗比全上下文低83.2%，并在检索精度上略有提升。

**⚠️ 局限性**

局限包括规模有限（仅76条记忆）、仅英语实验、合成数据不具生态真实性、超参数固定、对嵌入模型的依赖以及对关键词提取器的语言适应性不足。

---

## 24. MKEvolve: A Modular Multi-Agent Framework for Kernel Code Generation

**arXiv ID:** 2607.20501 | [PDF](https://arxiv.org/pdf/2607.20501v1)

**作者:** Jason Yoo `[一作]` (University of British Columbia), Youngsuk Park `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MKEvolve框架，通过迭代分解和优化PyTorch模块的子模块，实现可组合、可配置、可解释的高性能Kernel生成。

**💡 创新点**

创新点在于将整个Kernel拆解为可独立优化的子Kernel，利用LLM驱动的beam search并行改进子模块，同时在迭代过程中动态合并或拆分子模块，显著提升正确性与可迁移性。

**🔧 技术方法**

使用LLM（Claude 4.5 Opus、GPT‑OSS 120B）进行分解、生成、调优；Beam Search、Triton、TritonRL作弊检测、CUDA流评测等技术；以及LLM预算分配、子Kernel合并/拆分等自研算法。

**📊 数据集**

在KernelBench的L2（100个算子序列）与L3（50个完整模型）基准上进行评估。

**📈 对比分析**

与Parallel Scaling、Beam Search、KernelFalcon等基线比较，MKEvolve在正确率、速度提升（相对torch.compile）方面均优于其他方法，并且令LLM token消耗降低15–35%。

**⚠️ 局限性**

局限性包括对LLM质量的高度依赖、对复杂子模块拆分的自动化仍有限、以及在大规模模型或不同硬件平台上的可扩展性尚未充分验证。

---

## 25. StabilityBench: Benchmarking Instability in LLMs

**arXiv ID:** 2607.20558 | [PDF](https://arxiv.org/pdf/2607.20558v1)

**作者:** Emma Kondrup `[一作]` (Mila --- Quebec AI Institute), Reihaneh Rabbany `[通讯]` (Mila --- Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 StabilityBench——一种能把单轮基准查询转换为多轮对话历史的模型无关评估算子，进一步通过社会人口代理模拟和引导式 bait 进行语义保持的扰动，评估 LLM 在实际会话情境下的稳定性。

**💡 创新点**

创新点在于：①设计了任务保持的多轮模拟器和 2‑turn baiting 机制，②引入了社群人口代理条件与“sycophancy”/上下文注入两类 bait，③提出了 Bait Degradation Rate、Simulation Degradation Rate 与 Flip Rate 三种衡量模型不稳定性的指标，④推出了大小保持的 StabilityBench‑Mini 变体以降低评测成本。

**🔧 技术方法**

主要技术包括：基于概率分布的多轮交互模拟 d(c,q)；两轮 baiting 操作 𝒜_b^pre/𝒜_b^post；LLM‑as‑Judge 进行语义保真与匹配验证；统计学指标计算和多种模型对比实验。

**📊 数据集**

使用四个广泛采用的基准：AIME 与 GSM8k（数学推理）、HealthBench（医疗问答）以及 StrongReject（安全/拒绝测试），并在十款来自 GPT‑5、Gemini 2.5/3、Mistral 3 等模型族上进行评估。

**📈 对比分析**

与原始单轮基准对比，发现多轮模拟和 bait 使得大部分模型出现 15%–30% 的性能下降，Flip Rate 甚至高达 50% 以上，尤其在医疗与安全场景下表现突出；但在安全基准中模型的稳定性相对更好，且模型规模与准确率呈正相关。

**⚠️ 局限性**

限制包括：①社群人口代理仅基于英美语境，缺乏跨文化验证；②对 StrongReject 的验证率低，可能因模型安全机制导致拒绝评判；③评测仍以单一 LLM‑as‑Judge 进行，未涵盖多模态或更复杂的交互；④缺乏对不同 bait 与模拟组合对齐性的深入分析。

---

## 26. CRAWO: Custom Resources for Adaptive Workload Orchestration

**arXiv ID:** 2607.20490 | [PDF](https://arxiv.org/pdf/2607.20490v1)

**作者:** Eugênio Santos `[一作]` (Federal University of Rio Grande do Norte), Frederico Lopes `[通讯]` (Federal University of Rio Grande do Norte)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 CRAWO 框架，利用 Kubernetes CRD 与 Operator 实现边缘 AI 工作负载的自适应编排，并在车载车牌识别（LPR）场景中进行验证。

**💡 创新点**

创新点在于将调度智能与执行分离，构建可插拔的多准则决策层，采用 VIKOR 等 MCDM 方法进行硬件感知节点分配，并通过 CRD 交互实现无侵入式部署。

**🔧 技术方法**

技术采用 Kubernetes/K3s + CRD + Operator（Go Kubebuilder）+ Spring Boot 微服务 + YAFS 离散事件模拟，核心调度算法为 VIKOR 多准则决策。

**📊 数据集**

使用模拟的车辆监控关键帧流（1 FPS）作为 LPR 工作负载，基于 YOLO 检测与 CRNN OCR 模型的推理过程，未使用公开数据集，而是通过仿真生成相应帧和网络条件。

**📈 对比分析**

与 Random、Round‑Robin、Least‑Loaded、TOPSIS 等基线对比，VIKOR 在 5G 理想与典型条件下将端到端延迟降低约 85% 以上，失效率几乎为 0，并且在网络拥塞时比 TOPSIS 更稳健。

**⚠️ 局限性**

局限性包括仅在离散事件模拟中验证，未在真实硬件上测试；调度权重固定，缺乏自适应；仅针对单一 LPR 用例，缺乏对其他智能城市管道的泛化验证。

---

## 27. Incomplete Prompt Jailbreaks in Large Language Models

**arXiv ID:** 2607.20473 | [PDF](https://arxiv.org/pdf/2607.20473v1)

**作者:** Yeonjea Kim `[一作]` (KAIST), Jaesik Choi `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了不完整提示破解（IPJ）的机制，阐明LLM在句子续写时如何被诱导生成有害内容，并提出基于神经元级控制的防御策略。

**💡 创新点**

创新点在于将IPJ形式化为吸引子诱导模型延迟拒绝的现象，并识别出终止与续写的两类功能性神经元，实现更细粒度的安全干预。

**🔧 技术方法**

技术包括九类吸引子实验、LoRA安全微调以及基于神经元激活差异的终止/续写神经元识别与权重调节。

**📊 数据集**

使用公开的六类Jailbreak问题（210条样本）与HuggingFace上的incomplete-prompt-jailbreak数据集进行实验。

**📈 对比分析**

在Gemma、Qwen、Llama等模型上测量攻击成功率和拒绝距离，结果显示不完整提示的成功率高于完整提示，安全微调效果有限，而神经元调节显著提高拒绝率并降低有害输出。

**⚠️ 局限性**

局限在于缺乏系统的因果神经元识别框架、吸引子分类仅经验性、以及LoRA微调对不同吸引子和内容的泛化不足。

---

## 28. Detecting Neural Network Failures through Spectral Analysis of Internal Activations

**arXiv ID:** 2607.20590 | [PDF](https://arxiv.org/pdf/2607.20590v1)

**作者:** Arunan J `[一作]` `[通讯]` (Independent Researcher), Arunan J (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于内部激活频谱特征的误判检测框架 SDNN。

**💡 创新点**

首次将 Spectral Drift 概念引入误判检测，并通过多尺度频谱分析与递进式训练实现对内部动态的捕捉。

**🔧 技术方法**

采用 STFT、离散小波变换与统计时刻特征提取，结合双向 GRU 监测网络与递进式（curriculum）学习。

**📊 数据集**

在 CIFAR‑10 上使用预训练 ResNet‑50 进行验证，利用验证集中的自然误判样本。

**📈 对比分析**

与 MaxSoftmax、ODIN、Energy Score、Deep Ensemble 等输出层置信度方法对比，SDNN 在三次实验中平均 AUROC 为 79.0%（±25.3%），显著优于基线（约 50–55%）。

**⚠️ 局限性**

训练对随机种子高度敏感，模型在不同初始化下表现波动大；在浅层网络或其他数据集上的效果有限。

---

## 29. A Graph Neural Network approach to zero-shot Digital Twins

**arXiv ID:** 2607.20535 | [PDF](https://arxiv.org/pdf/2607.20535v1)

**作者:** Alicia Tierz `[一作]` (Universidad de Zaragoza), Elías Cueto `[通讯]` (Universidad de Zaragoza)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种零样本认知数字孪生框架，结合实时视觉感知与几何无关的热力学信息图神经网络，能够在未见几何和边界条件下实时推断不可观测的物理场并通过增强现实可视化。

**💡 创新点**

核心创新是将 GENERIC 热力学形式化嵌入局部图神经网络，使模型保持能量守恒和熵增约束，同时通过闭环数据同化实现从单帧视觉到完整三维物理状态的零样本迁移，并实现实时 AR 输出。

**🔧 技术方法**

采用 Local‑TIGNN（局部热力学信息图神经网络）、U‑Net/YOLOv11s‑seg 语义分割、稀疏光流跟踪、连续闭环数据同化、热力学正则化和增量优化的图神经网络。

**📊 数据集**

训练使用基于 Abaqus 的 FEM/SMH 合成数据（200 条 viscoelastic beam 轨迹和 120 条 glycerin sloshing 轨迹），视觉感知用真实实验图像（55 张 beam，160 张 fluid）并人工标注。

**📈 对比分析**

与开放式滚动仿真相比，闭环同化显著抑制数值漂移；在 beam 任务中位姿 RRMSE <0.4%，应力 RRMSE 11%；在 fluid 任务中位姿 RMSE 1.5mm，速度 8.7mm/s；整体实时延迟分别为 9.1 ms（beam）和 25.2 ms（fluid），均低于 30 fps 限制。

**⚠️ 局限性**

局限于平面运动假设，对深度变化敏感，需引入 RGB‑D 或立体视觉；此外仍依赖标注数据，难以跨摄像机/光照环境的泛化。

---

## 30. Deepfake News Detection: A Multimodal Framework Integrating LipNet, DeepSpeech and ResNET for Enhanced Audio-Visual Analysis

**arXiv ID:** 2607.20579 | [PDF](https://arxiv.org/pdf/2607.20579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 31. Directional Hallucinations: Ideological Drift in News-Grounded LLM Question Answering

**arXiv ID:** 2607.20487 | [PDF](https://arxiv.org/pdf/2607.20487v1)

**作者:** Chendi Wang `[一作]` (Vrije Universiteit Amsterdam), Jieying Chen `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套可复现的测量框架，用于评估新闻检索式问答任务中大型语言模型（LLM）的幻觉（hallucination）是否存在意识形态偏向，并在此基础上量化幻觉内容的左倾漂移。

**💡 创新点**

创新点在于：①将幻觉检测与意识形态方向性评估相结合，揭示幻觉本身是一个带有系统性政治偏向的错误；②通过token级别的logit不确定性分析，解释了“不确定性→猜测”机制与左倾漂移之间的关系；③提供了一套公开可复现的端到端评估流水线，可用于平台和公共机构对政治信息AI系统的稽核。

**🔧 技术方法**

技术包括：1）使用Llama 3生成中性问题；2）用四个LLM（Llama 3 8B、Mistral 7B、Deepseek 7B、GPT‑4o‑Mini）在给定新闻文档上进行 grounded QA；3）利用ANAH‑v2进行句级幻觉检测；4）训练DeBERTa‑v3的二元立场分类器判定幻觉句子左右倾向；5）提取最终层token logits，计算熵等不确定性指标，并用逻辑回归评估幻觉与不确定性、漂移之间的关系。

**📊 数据集**

数据集：QBias 21,727篇美国政治新闻，标注为左、右、中心三派，并附有主题标签（如选举、移民、枪支管制）。

**📈 对比分析**

比较方法：对四个模型的幻觉率、主题风险、左倾漂移比例以及不确定性与幻觉/漂移的关联进行统计检验。结果显示：Deepseek幻觉率最高（21.3%），GPT‑4o‑Mini最低（7.8%）；所有模型的幻觉内容在左倾方向上显著偏差（约65–70%），即使在右倾来源上也表现出左倾漂移；不确定性对幻觉预测具有显著正相关（OR≈40），并在Deepseek与Llama中进一步预测左倾漂移。

**⚠️ 局限性**

局限性包括：①立场分类器的二元设计忽略了中心派；②幻觉检测器ANAH‑v2未专门针对政治新闻验证；③使用单一模型生成问题可能引入偏差，但实验表明对左倾漂移影响有限；④对闭源模型的logit不确定性无法完整评估。

---

## 32. Workload-Aware Caching for Multi-Agent Systems

**arXiv ID:** 2607.20495 | [PDF](https://arxiv.org/pdf/2607.20495v1)

**作者:** Anas Mohamed `[一作]` (University of Minnesota), Ali Anwar `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工作负载感知的缓存失效策略，针对多代理系统中的DAG执行计划，结合重计算成本、依赖计数和代理调用频率决定缓存淘汰。

**💡 创新点**

创新点在于：①将三种与代理DAG结构、重计算成本、工作负载频率相关的信号整合成统一评分函数；②通过评分决定淘汰，显著提升低容量下的延迟和吞吐；③与传统LRU/LFU/ARC以及其他代理优化技术进行系统性比较。

**🔧 技术方法**

使用的技术包括：基于min‑heap的淘汰算法；DAG拓扑分析提取依赖计数；实时测量任务重计算时间；累计记录代理调用频率；在A100 GPU上使用Qwen2.5（14b/7b）模型构建Plan‑Act多代理工作流；实验框架集成CUDA、Ollama、Python。

**📊 数据集**

实验数据集为三大多代理基准：SlideVQA（演示问答）、MP‑DocVQA（多页文档问答）和VideoMME（视频理解），分别覆盖中等、可变和低重用场景。

**📈 对比分析**

与无缓存、LRU、LFU、FIFO、ARC、GPTCache、计划缓存、并行执行等基线进行对比。结果显示：在三组基准中，工作负载感知缓存平均降低64.7%延迟、比最佳有限容量方案提高31.1%延迟、吞吐提升至2.84×、准确率保持≈94.7%，并且接近无限缓存的理论性能。

**⚠️ 局限性**

局限性：权重固定，未能自适应学习；与计划缓存、并行执行等技术保持独立，未实现协同优化；未将缓存状态纳入规划阶段；在低重用场景下，收益相对有限。

---

## 33. AI-Driven Surrogate Models for Predicting Electrode-Scale Discharge Behavior in Lithium-Ion Batteries

**arXiv ID:** 2607.20577 | [PDF](https://arxiv.org/pdf/2607.20577v1)

**作者:** Mengda Xing `[一作]` (Université d'Artois), Alejandro Franco `[通讯]` (Université de Picardie Jules Verne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套自动化框架，将深度学习替代传统有限元求解器，实现电极尺度锂离子电池放电行为的快速预测；

**💡 创新点**

创新点在于引入可学习的高斯位置编码（GPE）和专用时间编码模块，结合Swin3D Transformer对稀疏体素网格进行时空特征学习；

**🔧 技术方法**

使用的技术包括3D体素化、Gaussian Positional Encoding、Swin3D Transformer（shifted-window自注意力）、多层感知机时间解码器、MSE损失训练；

**📊 数据集**

采用大规模电化学仿真（ES）数据集，涵盖NMC111阴极在不同几何、活性材料比例、日历化程度和放电倍率下的24步时间序列，约18个样本；

**📈 对比分析**

与传统点云学习方法（PointNet、DGCNN、Point Transformer、Point Transformer V2）对比，Swin3D Small+GPE模型在RMSE、SMAPE、R²上均表现最佳（RMSE 0.0734，SMAPE 55.43%，R² 0.7946），且推理时间仅0.149秒，较物理仿真加速10⁵倍；

**⚠️ 局限性**

局限在于数据集规模有限，导致更大模型容易过拟合；模型仅处理单一物理场（锂离子浓度），未考虑多物理耦合；未来需扩展到更复杂场景并优化样本需求。

---

## 34. Robust Critics: Defending LLMs Against Multi-Turn Attacks

**arXiv ID:** 2607.20472 | [PDF](https://arxiv.org/pdf/2607.20472v1)

**作者:** Roman Belaire `[一作]` (Singapore Management University), Pradeep Varakantham `[通讯]` (Singapore Management University)

**通讯引用:** 2896 | [OpenAlex ID](https://openalex.org/A5089113099)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于对话评审（Dialogue Critic Guided Sampling, DCGS）的多轮对话鲁棒性框架，通过在每轮推断用户意图并根据该意图生成回复，显著提升了大型语言模型在面临恶意攻击时的安全性。

**💡 创新点**

创新点在于：① 将用户意图建模为潜在变量，并在生成过程中引入两阶段采样（先生成意图再根据意图生成回复）；② 用学习到的价值与后悔（regret）评审器对候选意图/回复进行指数加权重抽样，实现推理时的“软”策略改进；③ 提出了理论证明，DCGS 等价于对基线策略的指数倾斜，可在任何有限候选池中保证期望收益提升；④ 采用轻量级的冻结基础 LLM + 线性评审头，使方法易于迁移至闭源前沿模型。

**🔧 技术方法**

核心技术包括：Markov Decision Process（MDP）建模、基于强化学习的价值评审（Q 估计）与后悔评审、TD 学习、软策略改进（soft policy improvement）以及两阶段采样重加权（reweighting）技术；此外，还使用了文本相似度奖励和多轮对话模拟器。

**📊 数据集**

使用了 CARES-18k、WildJailbreak、RedBench、Harmbench、ORBench、XSTest、DAN 等公开对抗数据集进行实验，并在这些数据集上构建了多轮对话模拟环境。

**📈 对比分析**

与多种基线（VDCGS、RDCGS、GPT‑4o naive、CAT、DCR、TPO、SmoothLLM 等）进行对比。DCGS 在所有数据集上的防御成功率（DSR）均显著高于基线，同时保持或提升目标完成率（GCR）。在迁移实验中，使用小模型的 DCGS 也能提升 frontier 大模型（如 GPT‑5.4‑mini）的鲁棒性。

**⚠️ 局限性**

主要局限包括：① 需要预先定义良好的奖励函数，限制了在更开放式任务中的直接应用；② 对极长对话或极端稀缺样本的鲁棒性尚未充分验证；③ 评审器的训练依赖于对抗模拟数据，若攻击手法出现显著变化，可能需要重新训练；④ 目前仍未探讨对模型生成效率与延迟的影响。

---

## 35. Is MoE Routing a Huffman Code? Discovering the Frequency-Diversity Law in Chain-of-Thought

**arXiv ID:** 2607.20427 | [PDF](https://arxiv.org/pdf/2607.20427v1)

**作者:** Ching-Chieh Tsao `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4716 | [OpenAlex ID](https://openalex.org/A5101936536)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `2704f255-0c84-4173-b83c-0e9a3dbea232` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从信息论角度剖析 Mixture‑of‑Experts (MoE) 路由机制，提出频率‑多样性定律，证明 MoE 路由本质上是自发的 Huffman 编码，并通过实验验证；在发现 Qwen3.5‑35B‑A3B 由于功能冗余导致 Huffman 信号消失后，提出了 Subset Difference Pruning（去除功能冗余专家）的训练后剪枝方法；

**💡 创新点**

创新点在于：①将 MoE 路由视为最优源编码器；②提出 Frequency‑Diversity Law 及其时间压缩表征；③发现并解决 Qwen 典型的功能冗余问题；④提出无需再训练即可提升路由效率的剪枝策略；

**🔧 技术方法**

使用信息理论（互信息、Huffman 下界）、统计分析（Spearman 相关、Pearson r）、稀疏专家分析、基于余弦相似度的专家冗余检测与剪枝；

**📊 数据集**

使用公开数学推理数据集：GSM8K、MATH、AQuA、CompMath‑MCQ；

**📈 对比分析**

通过对比各模型（Gemma‑4‑27B‑A4B、Phi‑3.5‑MoE、Qwen3.5‑35B‑A3B）的 Huffman 相关性、专家计数与操作类型频率的匹配程度，以及剪枝前后 GSM8K 上的准确率，发现剪枝后 Huffman 相关性从负转正，准确率仅下降约1.7%，证明方法有效；

**⚠️ 局限性**

局限性包括：①实验依赖于手工定义的四类操作类型，可能不适用于更复杂任务；②仅在特定稀疏度（k/E）范围内有效；③剪枝方案需在推理阶段手动设置，未实现在线自适应；④未对训练过程中的动态冗余产生机制进行深入建模。

---

## 36. RE-AD: Real-Time Requirement Adherence for Data Labeling

**arXiv ID:** 2607.20455 | [PDF](https://arxiv.org/pdf/2607.20455v1)

**作者:** Siddarth Malreddy `[一作]` (Uber AI Solutions), Subrat Sahu `[通讯]` (Uber AI Solutions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RE‑AD框架，利用LLM在人工标注过程中实时验证并纠正对标准操作程序（SOP）的遵循情况。

**💡 创新点**

核心创新在于：① 通过自我反思实现SOP的递归原子化，生成三层（格式、简单词汇、主观语义）可验证规则；② 基于规则复杂度的分层并行验证引擎，结合正则、轻量模型和链式推理模型实现实时反馈。

**🔧 技术方法**

技术栈包括：LLM（Gemini 3 Flash/Pro）、自我反思与链式推理、正则/字符串解析、前缀缓存、规则分层路由与并行验证。

**📊 数据集**

使用了自制的合成基准 RE‑AD‑Eval（20条规则、3层级、分层注释）以及真实工业标注流水线数据进行部署评估。

**📈 对比分析**

相较于传统后置审核和批量验证，RE‑AD在合成基准上整体F1≈0.75、格式约1.0、简单词汇≈0.86、主观语义≈0.55，平均延迟仅2.3 s；在生产环境中错误纠正率达82%，显著降低了审核成本。

**⚠️ 局限性**

局限性包括：仅在英语环境下测试；主观语义层的F1低且易产生误报；合成基准可能缺乏真实多样性；跨语言、噪声或领域漂移时的鲁棒性待验证。

---

## 37. More Is Not More: What Matters for Diversity in LLM Opinions?

**arXiv ID:** 2607.20429 | [PDF](https://arxiv.org/pdf/2607.20429v1)

**作者:** Qiyang Yao `[一作]` `[通讯]` (New York University), Qiyang Yao (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过因子实验分别控制 persona 深度和交互架构，对 7 种 LLM 在 100 条真实开放式问题上生成的意见进行系统评估，量化多样性。

**💡 创新点**

①将输入条件与交互结构两大干预维度解耦的因子实验；②构建统一的 α‑β 多度量多样性评估协议；③发现 persona 细化不呈线性增益、不同架构覆盖非重叠观点、低成本技巧效果微弱。

**🔧 技术方法**

使用 persona 细化（Role、Basic、Mid、Pro）、单/多轮自提问、5 人多代理讨论、温度提升、指令增强、性别/种族提示、个性标签等干预；提取 atomic 观点、文本嵌入（OpenAI text‑embedding‑3‑small），计算 MPD、CC、Vendi、β‑Vendi、UCR 等多样性指标。

**📊 数据集**

100 条真实用户查询的开放式问题集合；20 个 persona（不同深度）；7 大 LLM（来自不同供应商）。

**📈 对比分析**

通过配对 Wilcoxon 检验和 Cliff's δ 进行多度量比较。结果表明：单句职业（Role）提供最高 ROI，交互架构互补覆盖率提升约 50%，低成本技巧几乎无效；persona 细化呈 diminishing returns。

**⚠️ 局限性**

未评估训练时干预、争论/角色冲突多代理提示、持续交互或非文本环境；多样性指标基于嵌入空间，缺乏与人类多样性分布的绝对校准。

---

## 38. Break Through the Compression Bottleneck: From Theory to Practice

**arXiv ID:** 2607.20434 | [PDF](https://arxiv.org/pdf/2607.20434v1)

**作者:** Xiusheng Huang `[一作]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences), Kang Liu `[通讯]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了低秩分解与量化联合压缩大型语言模型的问题，并证明两者非正交，提出了先低秩再量化的最优压缩顺序以及Diagonal Adhesive Method（DAM）来显著降低额外误差。

**💡 创新点**

创新点在于首次给出低秩分解与量化非正交的数学证明，确定最优压缩顺序，并设计可学习的DAM方法以缓解量化导致的激活异常值问题。

**🔧 技术方法**

主要技术包括张量与点积层级误差分析、SVD分解、GPTQ量化、以及自适应对角矩阵调优的DAM。

**📊 数据集**

实验使用LLaMA系列模型（7B、13B、30B、70B）进行评估，利用WikiText‑2 perplexity 和 9 个 zero‑shot 任务（BoolQ、HellaSwag、LAMBADA、OpenBookQA、PIQA、SIQA、WinoGrande、ARC‑Easy、ARC‑Challenge）。

**📈 对比分析**

与传统 Q→L 和 L→Q 压缩顺序相比，DAM 在相同压缩率下平均提升约10–40%准确率或降低 perplexity，显示其在压缩后性能上显著优于现有方法。

**⚠️ 局限性**

局限在于未在更大模型或不同 GPU 平台上验证，也缺乏对不同量化位宽和低秩比例的更广泛探索。

---

## 39. Fisher Widths: Local Learning Geometry and Anisotropic Recovery

**arXiv ID:** 2607.20578 | [PDF](https://arxiv.org/pdf/2607.20578v1)

**作者:** Vu Khac Ky `[一作]` `[通讯]` (FPT University), Vu Khac Ky (FPT University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了统计流形上的高斯宽度复杂性，通过一对函数：原始Fisher宽度和逆Fisher宽度，探讨了它们在学习和恢复中的作用。

**💡 创新点**

提出了原始和逆Fisher宽度之间的尖锐关系，证明了它们的乘积不小于欧几里得尺度的平方，展示了Fisher各向异性如何在几何之间转移复杂性。

**🔧 技术方法**

使用了Fisher信息矩阵和高斯过程的几何方法，结合了统计学习理论和恢复理论。

**📊 数据集**

使用了多种Fisher矩阵配置的模拟数据集，特别是针对不同稀疏配置的高斯测量。

**📈 对比分析**

通过与标准Fisher-Lipschitz界限的比较，展示了在小Fisher球上的局部达到性，并通过支持敏感的恢复估计与其他方法进行了比较，性能表现良好。

**⚠️ 局限性**

局限性在于对一般结构集的下界原则的缺乏，以及在不同Fisher配置下的恢复性能可能不一致。

---

## 40. SevDiff: Severity-Conditioned Diffusion for Long-Tail Conflict Trajectory Generation

**arXiv ID:** 2607.20549 | [PDF](https://arxiv.org/pdf/2607.20549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 41. PlanE: Meta Planning of Data, Tuning, and Inference for Extractive-based LLMs

**arXiv ID:** 2607.20470 | [PDF](https://arxiv.org/pdf/2607.20470v1)

**作者:** Jiacheng Wang `[一作]` (Ant Group), Guangya Yu `[通讯]` (East China University Of Science And Technology)

**通讯引用:** 1598 | [OpenAlex ID](https://openalex.org/A5017641808)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PlanE 框架和 DTI planner，用于构建面向提取任务的 LLM。

**💡 创新点**

将数据、调优、推理三阶段统一规划，并设计离散的 DTI 组合优化方案，同时使用元学习预测最佳组合。

**🔧 技术方法**

采用数据分解（Pipeline、Bidirectional）、指令微调（SFT、SFT+RL）和提示推理（Direct、Intersection、Union）等技术，并构建基于元学习的 DTI planner。

**📊 数据集**

在 CMeIE‑V2、ACE05、14Lap 三个公共的提取与情感分析数据集上进行实验。

**📈 对比分析**

与 Grid Search、MetaGPT 等基线相比，PlanE 在单目标下 F1 提升约 1.36% 并将搜索时间缩短 554,833 秒；在多目标下同样可匹配 Grid Search 的性能。

**⚠️ 局限性**

局限包括：DTI planner 为离散函数，无法直接梯度优化；未考虑更多实际资源约束及更复杂的组合；采用标量编码导致信息量有限。

---

## 42. Isolating LLM Alignment from Regex: Zero Coverage and Metric-Dependent Divergence Under Adversarial Mutation

**arXiv ID:** 2607.20494 | [PDF](https://arxiv.org/pdf/2607.20494v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]` (Lumytics), Alexandre Cristovão Maiorano (Lumytics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM生产环境中，将正则过滤层与模型对齐层分离，使用单轴消融法检验在移除正则过滤后对齐层是否能提升拦截率。

**💡 创新点**

创新点在于：①构建了 L5‑no‑regex 的消融端点，仅禁用正则过滤，②设计了三类对齐层对抗语料（carry‑forward、regex‑bypass、alignment‑isolate）并进行静态预生成的攻击变体，③比较了子字符串检测与 LLM 判断两种评价指标的差异，揭示对齐层在不同攻击形式下的可检测性差别。

**🔧 技术方法**

使用了 Gemini‑2.5‑flash 作为对齐模型，Gemini 生成的 paraphrase、PAIR 与 TAP 等变体；评估中采用子字符串匹配器、LLM 内部判定器以及 Wilson 置信区间与 Fisher 检验等统计方法。

**📊 数据集**

使用了 45 条手工设计的对抗提示（共三类子语料），并通过 Gemini paraphrase、PAIR 与 TAP 生成约 1,555 条 probe‑run 对。该语料覆盖 OWASP LLM Top‑10 五类威胁。

**📈 对比分析**

比较方法：在 L0（无防御）与 L5‑no‑regex（仅对齐）两端点之间，按 OWASP 类别计算拦截率、置信区间和 p 值。结果显示，使用子字符串检测时 L5‑no‑regex 的拦截率为 0%（与 L0 相同），而使用 LLM 判断时，PAIR 生成的攻击在多类中达到 56–100% 的拒绝率；表明评价指标会显著影响发现对齐层效果。

**⚠️ 局限性**

局限性包括：①仅在 Gemini‑2.5‑flash 上实验，无法推广到其他 LLM 家族；②静态预生成的攻击变体可能低估对齐层的真实绕过能力；③子字符串检测过于保守，可能低估真正的拒绝；④LLM 判断可能存在自评偏差；⑤实验未覆盖不同温度、不同硬件或动态攻击循环。

---

## 43. DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making

**arXiv ID:** 2607.20491 | [PDF](https://arxiv.org/pdf/2607.20491v1)

**作者:** Raffi Khatchadourian `[一作]` `[通讯]` (IBM Financial Services Market), Raffi Khatchadourian (IBM Financial Services Market)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DFAH-Bench回放基准，用以评估金融工具使用型LLM在多次相同查询中的过程稳定性。

**💡 创新点**

创新点在于引入了DAR–TAR差距、证据接触差异、决策集中度等可观测过程指标，揭示了结果一致但过程不稳定的现象。

**🔧 技术方法**

采用可重复的工具调用模拟、SHA‑256哈希链、Ed25519签名等技术，并使用工具调用轨迹与证据哈希的可观察通道。

**📊 数据集**

使用了150个金融任务（合规分流、组合约束、DataOps异常）共1,338个案例、8,127条回放日志，覆盖10个模型。

**📈 对比分析**

通过与传统的决策一致率、准确率比较，发现高达18个百分点的DAR–TAR差距，表明仅靠准确率低估了模型过程不稳定性。

**⚠️ 局限性**

局限性包括仅针对金融领域、缺乏推理文本覆盖、API模型回放次数有限、模型多样性有限等。

---

## 44. Algorithmic Approaches to Sequential Decision-Making and Social Epistemology

**arXiv ID:** 2607.20636 | [PDF](https://arxiv.org/pdf/2607.20636v1)

**作者:** Kavya Ravichandran `[一作]` `[通讯]` (Toyota Technological Institute at Chicago), Kavya Ravichandran (Toyota Technological Institute at Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文研究了顺序决策和社会认识论中的算法方法，探讨了在复杂的现实世界中如何通过算法工具来理解决策问题，特别是涉及投资的决策。

**💡 创新点**

创新点在于结合了理论模型与社会问题的分析，提出了在社会认识论中如何通过算法干预来影响决策的框架，并探讨了个人特质（如毅力）对决策的影响。

**🔧 技术方法**

使用了顺序决策算法、数据驱动算法设计框架以及数学模型来分析社会现象。

**📊 数据集**

论文中使用了多臂赌博机问题的理论模型，以及与社会认识论相关的案例研究，具体数据集未明确提及。

**📈 对比分析**

通过与传统的最坏情况分析进行比较，论文展示了在数据驱动的算法设计中如何利用历史实例来学习良好的算法，性能上优于最坏情况的界限。

**⚠️ 局限性**

限制在于理论模型的抽象性可能无法完全捕捉现实世界的复杂性，且在某些情况下，模型的适用性和有效性可能受到限制。

---

## 45. AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use

**arXiv ID:** 2607.20536 | [PDF](https://arxiv.org/pdf/2607.20536v1)

**作者:** Junzhi Chen `[一作]` (New York University), Ashish Sabharwal `[通讯]` (Allen Institute for AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出AppWorld-UL基准，用扰动方法将原始自律任务改造成需要用户交互的工具使用任务，涵盖歧义澄清、不可行性处理和确认请求三种交互类型；

**💡 创新点**

创新点在于：①系统化的扰动生成交互式任务；②基于知识约束的LLM用户模拟；③可程序化评估交互质量；

**🔧 技术方法**

使用LLM（Claude Opus、GPT‑5.5等）作为代理和用户模拟器，结合AppWorld的API调用框架与程序化评估；

**📊 数据集**

使用AppWorld环境中的9个模拟应用（Amazon、Spotify等）生成516个任务；

**📈 对比分析**

通过对比不同LLM代理（功能调用和代码生成）在I‑TGC与I‑SGC指标上的表现，最佳系统Claude Opus 4.7仅获得48.6% I‑TGC（30.2% I‑SGC），显示交互需求显著提高难度；

**⚠️ 局限性**

局限性包括：交互模拟仍依赖LLM的可靠性；任务难度受扰动设计与知识分配影响；代理对已知信息的利用仍不充分，导致整体性能低于原始AppWorld。

---

## 46. Demonstrating GenDB: Instance-Optimized and Customized Query Processing Code Generation via LLM Agents

**arXiv ID:** 2607.20630 | [PDF](https://arxiv.org/pdf/2607.20630v1)

**作者:** Jiale Lao `[一作]` (Cornell University), Immanuel Trummer `[通讯]` (Cornell University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型语言模型（LLM）代理自动生成针对特定数据、工作负载和硬件的实例优化、定制化查询处理代码和存储结构，并通过多代理工作流实现离线批量模板查询的端到端自动化；

**💡 创新点**

在传统数据库引擎难以扩展的前提下，首次将LLM驱动的代码生成与硬件感知、数据感知、工作负载感知的迭代优化结合，能够在离线场景下为重复模板查询实现显著性能提升；

**🔧 技术方法**

使用Claude Agent等LLM代理、工具调用（文件、终端、网络）、多阶段拆解（工作负载分析、存储/索引设计、查询规划、代码生成、迭代优化）、模糊测试、硬件感知优化（SIMD、缓存、bitmap、hash等）；

**📊 数据集**

实验使用了标准TPC‑H（规模因子10）和自定义SEC‑EDGAR基准（新建数据集以避免LLM训练泄漏），并支持用户上传自定义数据与查询；

**📈 对比分析**

与DuckDB、ClickHouse、Umbra、MonetDB、PostgreSQL等主流引擎比较，利用系统自动创建索引与LLM推荐索引后，GenDB在TPC‑H上实现总耗时249 ms，分别比DuckDB快2.4×、Umbra 2.7×、ClickHouse 9.2×；在SEC‑EDGAR上实现403 ms，比分别比DuckDB快3.8×、Umbra 3.3×；

**⚠️ 局限性**

主要限制包括：离线生成成本高、生成时间与费用随模型和任务复杂度显著提升、缺乏完整的语义正确性保证（仅通过模糊测试验证一致性）、目前对Ad‑hoc查询支持有限、未提供正式形式化验证或安全性保证。

---

## 47. Exact ReLU realization of affine one-dimensional refinement iterates via residual memory and offset frames

**arXiv ID:** 2607.20586 | [PDF](https://arxiv.org/pdf/2607.20586v1)

**作者:** Boldsaikhan Bolorkhuu `[一作]` (McGill University), Tsogtgerel Gantumur `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

证明了在一维向量值连续分段线性(CPwL)输入与逼迫函数的情况下，任意阶的仿射细化算子迭代可由固定宽度、深度线性（O(n)）的ReLU网络精确实现。

**💡 创新点**

核心创新是引入残差记忆控制器（Residual Memory Controller），将非可逆的残差动力学改写为可注入的鞭形乘积，能够在网络中准确回放残差状态，从而实现后向 Horner 递归；此外，通过“偏移帧”分解使逼迫项在所有切点之外安全读取，解决了分支选择模糊。

**🔧 技术方法**

技术手段包括：1）CPwL函数的精确ReLU实现；2）基于 M‑ary 数字映射的块级传递矩阵；3）残差记忆控制器的 CPwL 拓扑映射及其可逆性；4）分支选择器的可微CPwL门控；5）双帧（普通帧与偏移帧）分解与拼接；6）后向回放 + Horner 递归的组合网络。

**📊 数据集**

本工作为纯理论分析，不使用任何训练数据集；所有证明均基于符号推导与结构化的网络构造。

**📈 对比分析**

由于是严格的实现定理，论文未进行实验对比或性能评估；仅给出网络宽度固定、深度 O(n) 的理论上限以及参数增长上界为指数级别。

**⚠️ 局限性**

局限性：1）仅覆盖一维参数、CPwL 正则性；2）在 M=2（二进制）情况下，缺乏非平凡的偏移帧，因而只能处理“普通帧边界分离”的逼迫项；3）参数上界为指数级，尚未优化；4）未讨论逼近极限分形或多维残差动力学的推广。

---

## 48. Verifier-First Evaluation of Agentic LLMs for Infrastructure-as-Code Generation

**arXiv ID:** 2607.20478 | [PDF](https://arxiv.org/pdf/2607.20478v1)

**作者:** Mohamed Jouini `[一作]` `[通讯]`, Mohamed Jouini

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 IaC（Terraform/AWS）代码生成进行 verifier-first 的实证研究，比较七种 agentic 策略在 IaC‑Eval v2 186 任务上的性能。

**💡 创新点**

创新点包括：① 将验证流程拆分为 VALIDATE、PLAN、OPA 三个阶段并使用 McNemar 统计验证差异；② 证明检索（ReAct+RAG）和修复（迭代 Refinement）解决不同的 failure 类；③ 在 IaC 生成任务中首次将 DSPy 的 prompt 级优化（GEPA）和示例级优化（SIMBA）与检索/修复组合；④ 通过 Rego oracle 实验揭示 79% OPA 失败为信息缺失而非模型能力缺陷。

**🔧 技术方法**

技术手段包括：ReAct + MCP / RAG 检索、Iterative Refinement、GEPA（反射式 prompt 进化）、SIMBA（示例对比优化）、Rego policy 注入、DSPy 编译管线、McNemar 统计、MLflow/Arize 追踪。

**📊 数据集**

使用数据集 IaC‑Eval v2：186 条 Terraform/AWS 任务，包含自然语言需求、HCL 参考解答和 Rego intent policy，覆盖 6 个 AWS 服务族和 6 个难度等级。

**📈 对比分析**

对比方法：所有策略在同一三阶段 verifier‑first pipeline 下跑同一 186 任务，记录 pass@1、各阶段 failure 计数，采用 McNemar 统计检验 pairwise 结果；性能方面：Rego oracle 93%（最高），GPT‑4o 迭代修复 84.4%，GEPA‑RAG 53.2%，Active RAG 45.7%，Qwen 7B 基线 14.0%。

**⚠️ 局限性**

局限性：① 仅针对 AWS/Terraform；② Rego oracle 仅用于诊断，不适合作为真实部署策略；③ 采用温度 0 的确定性推理，未评估多次运行方差；④ 部分 OPA 失败是 benchmark artifacts；⑤ DSPy 评估仅在小规模配置下进行，未探索更大优化预算。

---

## 49. Reliability-Aware LLM Alignment from Inconsistent Human Feedback

**arXiv ID:** 2607.20515 | [PDF](https://arxiv.org/pdf/2607.20515v1)

**作者:** Jingyi Huang `[一作]` (Miami University), Yang Zhang `[通讯]` (Miami University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可靠性导向的偏好优化框架RGPO，针对RLHF中的不一致人工反馈进行建模和校正。

**💡 创新点**

创新点在于通过EM求解注解者可靠性和潜在真标签，并用可靠性一致性度量动态调整损失权重，减少噪声影响。

**🔧 技术方法**

采用EM最大似然估计、迭代隐变量推断、可靠性一致性加权损失以及DPO/SimPO/IPO等RLHF算法。

**📊 数据集**

使用MultiPref和HelpSteer2两份多注释偏好数据集进行实验。

**📈 对比分析**

与标准DPO、SimPO、IPO对比，在AlpacaEval2和Arena-Hard上均提升约3-10% win率，特别是Qwen2.5+RGPO达到最高85%级别。

**⚠️ 局限性**

局限在于仅验证8B级别模型、缺乏大模型及更丰富多注释数据集，且HelpSteer2仅用注释索引代替真实注解者ID。

---

## 50. Telco-GAIA: Bilingual Benchmark for Agents in Telecom Domain

**arXiv ID:** 2607.20510 | [PDF](https://arxiv.org/pdf/2607.20510v1)

**作者:** Dmitrii Khizbullin `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了Telco-GAIA基准，用于评估工具调用型语言模型代理在真实电信运营商多源数据上的推理与检索能力。

**💡 创新点**

创新点在于将双语、多模态（文本、图片、表格）与多源（网站快照、PDF、合成SQL数据库、外部档案）结合，提供可复现的沙箱环境，并采用严格因果多跳链和精确字符串匹配的客观评测方式，避免LLM-as-a-Judge带来的非确定性。

**🔧 技术方法**

采用Docker沙箱、浏览器工具、REST API查询、PDF解析工具、图像识别模块等技术，构建代理的工具调用与多轮交互框架。

**📊 数据集**

使用的任务集包含100个人工验证的英语（65题）与阿拉伯语（35题）问答任务，来源涵盖运营商网站快照（HTML、图片、PDF）、合成SQLite客户数据库、以及维基百科与ArXiv的外部网页。

**📈 对比分析**

通过在12款商业与开源LLM上跑参照代理，比较了准确率、成本与延迟。最高模型准确率达71%，中等成本模型约38%，视觉相关任务最低仅3.8%，展示了当前代理在图像与文档理解方面的瓶颈。

**⚠️ 局限性**

局限性包括数据量有限（仅100题）、单一运营商与行业、合成客户数据库缺乏真实噪声，因而对跨行业或更大规模的泛化能力评估存在限制。

---

## 51. Representation Robustness Under Executable Reasoning Constraints in Large Language Models for Mathematical Problem Solving

**arXiv ID:** 2607.20520 | [PDF](https://arxiv.org/pdf/2607.20520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 52. Do Active SAE Feature Planes Carry More Holonomy? A Preregistered Reversal in Gemma

**arXiv ID:** 2607.20522 | [PDF](https://arxiv.org/pdf/2607.20522v1)

**作者:** Larry Richards `[一作]` `[通讯]`, Larry Richards

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在Gemma 2 2B语言模型中，活跃的稀疏自编码器（SAE）特征平面与混合特征平面在激活空间的全周率（holonomy）是否有区别，并对假设“holonomy聚焦于语义富集的特征平面”进行了检验

**💡 创新点**

首次在Transformer语言模型的激活空间上使用全周率测量工具验证语义集中假设，并揭示了该假设在Gemma 2 2B的特定层与读出下逆向成立（活跃特征平面相较混合特征平面holonomy更低）

**🔧 技术方法**

采用了基于受限雅可比传输规则的全周率测量仪，计算从层‑12到层‑13的残差流读出上的曲率；同时使用SAE解码器行向量构造特征平面、随机平面，使用回归和混合效应模型进行统计分析

**📊 数据集**

使用Gemma 2 2B模型的层‑12残差流激活，来自WikiText‑103语料的390条文本片段作为基点，构造了共2,340个测量点（3种平面×2大小×390基点）

**📈 对比分析**

通过在匹配的内平面大小下对活跃特征平面与混合特征平面进行配对OLS比较，得到对数尺度上的显著负效应−0.294（约26%降低）且低于材料阈值；大小效应未显著；三向排序因缺乏共同支持而未给出结论；二级混合效应模型与一次分析结果一致

**⚠️ 局限性**

结果仅适用于单一模型、单一层、单一读出、特定循环半径和受限雅可比传输规则；随机平面缺乏匹配支持；可能存在激活强度、特征参与度、字典几何、离散中心偏移、激活流形亲近度和传输剪切等混杂；off‑manifold代理不完善；缺乏对其他层、模型、数据集或不同测量方法的推广性验证

---

## 53. SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales

**arXiv ID:** 2607.20548 | [PDF](https://arxiv.org/pdf/2607.20548v1)

**作者:** Mikail Khona `[一作]` (NVIDIA), Tijmen Blankevoort `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型预训练，作者改进并扩展了预条件梯度优化方法（Mu和SOAP），解决了大批量训练中的不稳定性，并提出了分层分布式优化器实现。

**💡 创新点**

创新点包括：①在SOAP中实现逐步QR正交化并结合KL‑Shampoo协方差估计，消除训练损失尖峰；②使用更新‑RMS匹配框架实现AdamW、Mu、SOAP的公平学习率迁移；③提出兼容Megatron‑LM的层级分布式优化器，保持矩阵完整性并隐藏通信。

**🔧 技术方法**

技术手段：SOAP、Muon、AdamW、Kronecker‑factored预条件、QR正交化、Newton‑Schulz极化、KL‑Shampoo、更新‑RMS匹配、层级分布式优化器、Megatron‑LM框架。

**📊 数据集**

数据集与模型：使用Nemotron‑3数据集（1T、3T tokens）和Qwen‑3系列，训练8B Dense GPT、3B/30B MoE、8B/72B混合Mamba‑Transformer等多种规模模型，序列长度8192。

**📈 对比分析**

比较方法：通过更新‑RMS匹配统一学习率，对Mu、SOAP、AdamW在不同批量大小（最高100M tokens）下的训练损失、稳定性和token效率进行对比。实验表明，Mu和SOAP在大批量下均显著优于AdamW，SOAP在部分实验中略优于Muon；在下游任务（MMLU、Coding、Math、Commonsense）上也获得更高分数。

**⚠️ 局限性**

局限性：未系统调优ε参数；对极端规模的数值稳定性仍需进一步研究；正交化迭代次数和KL‑Shampoo等子模块的超参数仍依赖经验；需要更多针对不同网络结构（如Mamba、LoRA等）的混合优化策略。

---

## 54. ConfidenceBench: Evaluating Confidence Calibration in Large Language Models

**arXiv ID:** 2607.20526 | [PDF](https://arxiv.org/pdf/2607.20526v1)

**作者:** Matthew ffrench-Constant `[一作]` (Independent Researcher), Sanyam Kapoor `[通讯]` (New York University)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5044630105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个由200道私有多项选择题组成的 ConfidenceBench 校准基准，评估大语言模型的口头置信度。

**💡 创新点**

创新点在于将 Brier 分数作为惩罚函数，直接对模型口头置信度进行评分，并提供四种认知失误类型的题目，突出准确性与校准的区别。

**🔧 技术方法**

采用提示式口头置信度抽取、Brier 分数计算、Calibration Gap、ECE 等指标，并在 15 个前沿模型上进行三次独立实验。

**📊 数据集**

使用手工编写的 200 道私有题目，覆盖空间推理、高精度数学、词汇检索和不可知问题四类。

**📈 对比分析**

与 15 个模型（包括 GPT‑5、Claude、Gemini 等）以及人类测试者对比，最准确模型并非最佳校准，最佳校准模型的 Brier 仅为 0.103，低于人类 0.105，表现突出。

**⚠️ 局限性**

局限在于只评估口头置信度而非内部概率，数据集为私有且仅覆盖英文单轮多选，难以推广到多语言、多轮或开放域生成任务。

---

## 55. Semantic Field Theory: Historical Origin, Higher-Order Interaction, and Stabilized Semantic Inference

**arXiv ID:** 2607.20451 | [PDF](https://arxiv.org/pdf/2607.20451v1)

**作者:** Dimitris Vartziotis `[一作]` `[通讯]` (NIKI Digital Engineering), Dimitris Vartziotis (NIKI Digital Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出Semantic Field Theory（SFT）框架，将词义表示为分布式语义场、上下文变形、子集交互、残差分解和能量稳定化；给出Gaussian闭包、Mobius残差、阶谱和能量最小化的数学形式，并演示三词示例；

**💡 创新点**

首次将语义场理论系统化为可估计的计算模型，结合高阶交互、残差谱和能量动态；提供闭合的Gaussian乘积公式、Mobius逆推残差与三词问题的精确对应；

**🔧 技术方法**

基于高斯分布的场代数、子集拉普拉斯（Mobius）变换、能量函数最小化（梯度下降）、参数化的上下文变形（可用Transformer编码）以及可解释的残差阶谱计算；

**📊 数据集**

论文主要以理论和示例为主，并未给出真实数据集实验，评估思路提到SICK、SNLI、MultiNLI、GLUE、SuperGLUE等通用基准；

**📈 对比分析**

未给出具体实验对比或数值结果；作者提出在未来工作中对比p=1,2,3级交互、残差诊断与人类歧义评估，并比较能量收敛和多峰性等指标；

**⚠️ 局限性**

主要限制包括：对语义空间S的依赖导致可识别性问题；高阶交互计算复杂度高，需稀疏化或近似；未解决语义接地与规范性；与Transformer的关系仅为方法论上的对照，而非直接映射。

---

## 56. Benchmarking the Personalization Capabilities of Large Language Models

**arXiv ID:** 2607.20471 | [PDF](https://arxiv.org/pdf/2607.20471v1)

**作者:** Ashutosh Srivastava `[一作]` (Adobe Media & Data Science Research), Balaji Krishnamurthy `[通讯]` (Adobe Media & Data Science Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 SDR-Arena 框架，利用 Bayesian Persuasion 视角对大语言模型（LLM）在生成个性化销售信息方面的能力进行系统评估，并发布了公开的 SDR-Bench 语料库。

**💡 创新点**

将两方博弈（sender‑receiver）与生成式 LLM 相结合，构建可复现、可扩展的评估平台；同时提供基于公开成功案例的大规模客观测试集，填补了现有单方自适应评测的空白。

**🔧 技术方法**

采用大语言模型（如 Claude Sonnet 4.6、GPT‑4o、Qwen‑2.5 等）配合时序受限网络搜索或深度研究代理；通过语义对齐评估器（Semantic Judge）计算 Weighted Coverage Score，形成客观量化指标。

**📊 数据集**

主要使用 SDR-Bench（6,279 条公开成功故事）以及两家企业内部的销售邮件与电话记录（约115,000 封邮件、5,435 通话），覆盖 22 行业与 200 家企业。

**📈 对比分析**

通过对比 WCS 与人工判定，发现 Claude Sonnet 4.6 在公共数据上最高约 55.8%，但与人类专家仍存在显著差距（个性化平台）；深度研究代理略优但成本更高；在不同行业（医疗、技术）模型表现差异明显。

**⚠️ 局限性**

评测仅关注语义覆盖，未覆盖写作风格、语气与情感等关键信息；模型无法完整复制人类策略，仍处于“个性化平台”；数据泄露风险虽低但仍可能受预训练记忆影响；评估聚焦销售场景，难以直接泛化到其它沟通任务。

---

## 57. CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning

**arXiv ID:** 2607.20553 | [PDF](https://arxiv.org/pdf/2607.20553v1)

**作者:** Yubo Wang `[一作]` (Alibaba Group), Lei Chen `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的轻量级记忆管理模型CMI-Mem，将下游QA奖励与自监督的条件互信息(CMI)奖励相结合，实时评估每个记忆操作的增量信息量；

**💡 创新点**

创新点在于：①引入CMI作为内在奖励，弥补传统QA奖励对查询分布的依赖；②设计四类认知槽（核心、事件、语义、程序）共享策略的多维记忆架构；③通过残差投影估算CMI并采用高斯形状化实现稠密、平滑的奖励反馈；

**🔧 技术方法**

采用Qwen3-4B/8B基础模型与Qwen3-Embedding做语义检索；使用Group Relative Policy Optimization (GRPO)进行策略优化；使用残差投影+余弦相似度计算CMI；引入奖励形状化与多任务奖励混合；

**📊 数据集**

在LongMemEval（s_cleaned）进行训练与评估，LoCoMo用于离散分布外检验，MemoryAgentBench（排除FC-MH子集）用于多任务记忆检验；

**📈 对比分析**

与RAG、Mem0、MemBuilder-P（prompting）和MemBuilder-RL/Memory-R1（RL）等基线进行对比；CMI-Mem-4B在LongMemEval、LoCoMo和MemoryAgentBench上分别提升约+3–+6点，CMI-Mem-8B在MemoryAgentBench上进一步提升约+5点，显示出显著的跨场景迁移与性能优势；

**⚠️ 局限性**

主要局限：①仅在4B/8B规模模型上验证，未测试更大模型；②对多模态任务的适应性受限于基模型能力；③缺乏直接的记忆质量评估指标，依赖QA评测，难以全面衡量记忆管理效果；

---

## 58. SiGMA: Sign-Guided Merging and Adaptation for Multimodal Continual Instruction Tuning

**arXiv ID:** 2607.20511 | [PDF](https://arxiv.org/pdf/2607.20511v1)

**作者:** Keonhee Park `[一作]` (Seoul National University), Gunhee Kim `[通讯]` (Seoul National University)

**通讯引用:** 6118 | [OpenAlex ID](https://openalex.org/A5100664729)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SiGMA框架，解决多模态持续指令微调中出现的负干扰问题，结合签号引导的自适应调优和签号引导的LoRA合并；

**💡 创新点**

首次将权重符号分解作为子空间划分的依据，训练时动态对齐通用子空间以减少漂移，推理时选择并放大特定子空间参数以抑制负干扰；

**🔧 技术方法**

基于LoRA的参数高效微调、签号引导的自适应调优与合并、掩码+温度因子、余弦距离放大等技术；

**📊 数据集**

在UCIT（ImageNet‑R、ArxivQA、VizWiz、IconQA、CLEVR、Flicker30k）和DCL（Remote Sensing、Medical、Autonomous Driving、Science、Finance）两大基准上进行实验；

**📈 对比分析**

与LoRA‑FT、O‑LoRA、MoE LoRA、CL‑MoE、HiDe LLaVA、DISCO等最新方法对比，SiGMA在UCIT和DCL的Avg. All提升约4–8%，Forgets显著下降，整体表现优于现有最优；

**⚠️ 局限性**

随着任务序列增长，通用子空间稀疏度上升，长期持续学习效果可能受限；目前仅在LoRA上验证，未扩展到其他PEFT方法。

---

## 59. CT-Merging: Consensus Directions and Task-Level Scaling for LoRA Adapter Merging

**arXiv ID:** 2607.20561 | [PDF](https://arxiv.org/pdf/2607.20561v1)

**作者:** Keumseo Ryum `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyuk Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种数据自由的LoRA适配器融合方法CT-Merging，通过共识方向和任务级系数来合并多任务适配器。

**💡 创新点**

在SVD基线合并中引入了基于平均投影子空间的共识方向构建和基于任务残差能量的RMS系数分配，解决了系数量化失配问题。

**🔧 技术方法**

利用SVD分解、投影子空间平均、极化正交化、RMS能量归一化以及全局缩放γ等技术。

**📊 数据集**

在CLIP ViT-B/16、ViT-B/32、ViT-L/14的八、十二、十六任务视觉基准（如Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC45、SUN397、SVHN、CIFAR100等）以及KnOTS提供的检查点上评估。

**📈 对比分析**

与Task Arithmetic、TIES、KnOTS-TIES、Iso-CTS、DC-Merge等基线比较，CT-Merging在大多数设置下平均归一化准确率领先，尤其在KnOTS检查点上提升约2–3个百分点。

**⚠️ 局限性**

仍需手动设定共识秩k和残差秩，对任务数量和模型规模的敏感度需进一步研究，且仅在LoRA低秩更新场景下验证。

---

## 60. Conflict Resolution under Degraded Surveillance in Air Corridors Using Multi-Agent Reinforcement Learning

**arXiv ID:** 2607.20547 | [PDF](https://arxiv.org/pdf/2607.20547v1)

**作者:** Esrat Farhana Dulia `[一作]` (Kent State University), Ruben Del Rosario `[通讯]` (Kent State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究基于深度Q网络的多智能体强化学习框架，在降噪、延迟、缺失等退化监视条件下，实现在三维航空走廊中异种无人机与eVTOL的去中心化冲突解决

**💡 创新点**

创新点在于同时考虑多种不确定因素（观测噪声、通信延迟、信息丢失、风扰动、执行器不确定性、模型不确定性）以及能量消耗与走廊约束，且为不同机型训练了专属策略

**🔧 技术方法**

使用深度Q网络（DQN）强化学习与经验回放、目标网络，并采用14种离散动作空间与结构化三维走廊模拟环境

**📊 数据集**

使用自生成的模拟数据集，随机初始化不同密度、最小分离阈值及机型组合的90种测试情景，训练共20,000个episode，每个情景50个测试回合

**📈 对比分析**

通过与基线无冲突解决策略对比，损失分离事件与碰撞率显著降低，最高LOS事件可在1秒内恢复，六个Pareto最优配置实现了容量与安全的折衷，训练收敛稳定（2.76%变化）且测试CV<10%

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证，未考虑真实无人机动力学与更复杂的气象与通信环境，动作空间有限且未评估连续控制策略，结果未在真实飞行数据上验证

---

## 61. Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects

**arXiv ID:** 2607.20596 | [PDF](https://arxiv.org/pdf/2607.20596v1)

**作者:** Seonglae Cho `[一作]` (Holistic Ai), Adriano Koshiyama `[通讯]` (Holistic Ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并验证稀疏自编码器（SAE）中单词单调特征的几何与因果稳定性，覆盖六种语言模型和三种SAE族。

**💡 创新点**

首次证明单词单调特征的因果重要性受 SAE 族而非仅受尺度或激活函数影响，并系统比较不同 SAE 族在同一基模型上的因果结构。

**🔧 技术方法**

采用零删减因果 ablation、解码器对齐检测、PCA/Grassmannian 维度分析、统计检验（BH、Mann‑Whitney、Spearman）、自解释和词嵌入对齐等技术。

**📊 数据集**

使用 Neuronpedia 公共稀疏自编码器检查点、OpenWebText 128 词序列、以及 GPT‑2、Gemma、Llama、DeepSeek 等多种模型。

**📈 对比分析**

通过单词单调特征检测率、聚类紧密度、因果显著性（BH p<0.05）与恢复率进行跨模型/跨 SAE 族比较，发现 GemmaScope/BatchTopK 在大多数层因果相关，而 LlamaScope 在多数层缺乏锚定；跨族差异可达 46 倍。

**⚠️ 局限性**

交叉族比较混杂训练数据、词典宽度、训练配方等因素；仅检视单词单调特征，未覆盖多词跨度；检测阈值与词表级标注依赖固定操作点。

---

## 62. StrideDiffusion: Accelerating Diffusion Models for Time-series Generation

**arXiv ID:** 2607.20545 | [PDF](https://arxiv.org/pdf/2607.20545v1)

**作者:** Du Yin `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了训练无关、基于频谱能量的自适应步长采样器StrideDiffusion，用以加速时间序列扩散模型的推理过程。

**💡 创新点**

创新点在于发现时间序列扩散逆向过程呈现从高频到低频的粗到细激活序列，并通过频带能量、幅度漂移和相位速度等统计量自适应决定采样步长，且提供了频带稳定性理论支持。

**🔧 技术方法**

技术上使用DDIM与DPM‑Solver‑2的确定性单步仿射更新，构建频带门控模块并与步长调度器结合，实现微跳与大跳的自适应切换；频谱统计通过FFT计算。

**📊 数据集**

实验数据集包括六个无条件生成基准（Sines、Stocks、ETTh、Energy、fMRI、MuJoCo）以及四个条件填充与预测任务（Stocks、ETTh、Energy、fMRI）。

**📈 对比分析**

与Diffusion‑TS、Diffusion‑TS‑fast、DiffWave（Fast）、DiffTime等基线对比，StrideDiffusion 在无条件生成中仅需 14‑66 次函数评估，取得最高 18.9× 的壁钟加速，且质量与基线相当；在条件任务中平均 5‑14× 加速，预测精度保持不变。

**⚠️ 局限性**

局限性包括：目前采用经验式步长调度，缺乏学习型策略；门控阈值需手动调优，对极长序列或高维频带的适用性尚未验证；在极低噪声阶段仍需细步。

---

## 63. HERMES: Heterogeneous Edge-Relational Multi-Head Embedded SSM Attention for Traffic Conflict Prediction at Signalized Intersections

**arXiv ID:** 2607.20505 | [PDF](https://arxiv.org/pdf/2607.20505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 64. AsymVerify at SemEval-2026 Task 6: Asymmetric Confidence-Gated Verification for Political Evasion Detection

**arXiv ID:** 2607.20439 | [PDF](https://arxiv.org/pdf/2607.20439v1)

**作者:** Sebastien Kawada `[一作]` `[通讯]` (Kaons), Sebastien Kawada (Kaons)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AsymVerify，一种基于置信度门控的双向验证系统，用于判别政治问答中的 Clear Reply、Ambivalent 与 Clear Non-Reply。

**💡 创新点**

将低置信度预测按错误类型（向下降级 CR/CNR→AMB 或向上升级 AMB→CR）分派给专门的验证器，并利用置信度门控降低计算成本，同时通过异步两路验证实现高宏F1。

**🔧 技术方法**

使用结构化 JSON 语言模型提示、置信度门控与条件验证（P2、P3）、低温度推理以及 GLM、DeepSeek、Llama 等多种 LLM 后端。

**📊 数据集**

基于 CLARITY 政治访谈问答数据集，训练集 3,448 条，Dev 308 条，Eval 237 条，Ambivalent 标签占比 59%/67%。

**📈 对比分析**

与 41 支参赛队伍对比，Eval 上 Macro F1 0.85 并列第二；Dev 上单通道 73.0 Macro F1，调用数 1.48/例，单次验证已提升 6.8–15.2 Macro F1，整体提升 17.1。

**⚠️ 局限性**

阈值与验证提示仅针对美国英语政治访谈，跨文化/语言需重新调参；标注主观性导致错误聚类；未评估对抗性诡辩或 ASR 噪声；系统仅对每对独立分类，未利用多轮上下文。

---

## 65. Monkey King Bang: A Unified Scientific Multimodal Foundation Model

**arXiv ID:** 2607.20557 | [PDF](https://arxiv.org/pdf/2607.20557v1)

**作者:** Hesen Chen `[一作]` (Shanghai Academy of AI for Science), Yuan Qi `[通讯]` (Shanghai Academy of AI for Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的科学多模态模型MKB，能够在同一框架下理解并生成DNA/RNA/蛋白序列、分子图、气象场和医学影像等六大科学分支的原生数据。

**💡 创新点**

创新点在于将共享Transformer骨干与专属编码器、适配器和解码器相结合，并采用两阶段模态-语言训练课程，使模型能够在保持通用能力的同时掌握多种科学模态的结构特征。

**🔧 技术方法**

技术手段包括基于Qwen3‑VL‑8B的自回归Transformer、Perceiver式重采样与MLP投影、Swin风格空间编码器、SAM3分割头、Flamingo式SMILES解码器，以及混合任务的多模态训练策略。

**📊 数据集**

使用了DNA、RNA、蛋白序列、SMILES分子图、ERA5气象场、医学影像分割数据以及公开基准如Biology‑Instructions、MOLNet、ERA5、BiomedParse、MMLU‑Pro、MMMU‑Pro等。

**📈 对比分析**

与专用基线和大模型对比，MKB在生物序列理解、分子预测、气象预报和医学分割任务上表现接近或优于同规模/更大模型，同时保持了Qwen3‑VL的通用能力；在某些任务上实现了显著的性能提升。

**⚠️ 局限性**

局限主要体现在需要高精度数值回归的任务（如ADMET、酶动力学）以及对结构高度依赖的配对预测，当前模型在这些领域仍落后于专门化的专业模型。

---

## 66. PersonaTrail: Benchmarking Personalized Web Agents through Browsing Trails

**arXiv ID:** 2607.20482 | [PDF](https://arxiv.org/pdf/2607.20482v1)

**作者:** Seungbin Yang `[一作]` (KAIST), ChaeHun Park `[通讯]` (Chonnam National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个基于真实浏览轨迹的个性化网络代理基准与两阶段记忆框架（Preference‑Aware Contextual Memory）来评估代理的偏好推理和情景重现能力。

**💡 创新点**

创新点包括：①将原始浏览历史拆分为事实记忆与偏好记忆以更精细地捕捉用户偏好；②构建包含单跳与多跳、单一与多任务的可复现开放网络基准；③在检索与评估阶段使用LLM与向量检索的混合策略。

**🔧 技术方法**

主要技术手段有：LLM驱动的数据生成与查询模板发现、向量检索与聚类构建记忆、两阶段检索与LLM评估器（如Gemini 3.1 Pro）以及管理式开放网络验证管道。

**📊 数据集**

使用了8,585个从Tranco与Similarweb筛选的真实网站、PersonaHub生成的合成用户画像，以及由LLM生成的浏览历史与个性化查询数据。

**📈 对比分析**

与无检索、AWM、ReasoningBank等基线进行比较；在Preference Inference任务中，PACM将单跳TSR从约28%提升至53%以上；在Episodic Grounding任务中，多跳TSR从17%提升至39%（以Qwen‑3.6‑27B为例），显示显著性能提升。

**⚠️ 局限性**

局限性包括：仅支持一次性查询，缺乏对话交互；使用合成用户画像与意图，未覆盖真实用户多样性；只评估可解答的查询，未考虑不可回答或已删除内容的情况。

---

## 67. Finding Fast Filters

**arXiv ID:** 2607.20634 | [PDF](https://arxiv.org/pdf/2607.20634v1)

**作者:** Karima Ma `[一作]` (Adobe Research), Jonathan Ragan-Kelley `[通讯]` (MIT)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套统一的设计语言和自动搜索框架，用于在给定的 FIR 目标滤波器上生成高质量、低延迟的近似滤波器实现，并通过梯度下降学习连续参数，最终自动生成可直接编译为高效 SIMD 并行 C++ 代码的滤波器程序。

**💡 创新点**

创新点主要包括：① 将多种传统加速技术（多分辨率、递归、频域掩码、卷积金字塔、TIIR 等）抽象为可组合的可微分原语，形成单一可搜索的设计空间；② 采用连续参数化的可微分表示（尤其是 TIIR 的多项式重参数化和软截断技术），使得基于梯度的优化能够在非凸结构中稳定收敛；③ 引入成本模型预测吞吐量，避免了成千上万个候选程序的昂贵编译/基准；④ 设计了层级化编译器，将抽象程序降到 SIMD 并行、向量化、缓存友好的 C++ 表达式模板，实现真正的高速运行。

**🔧 技术方法**

技术手段包括：可微分程序设计语言、梯度下降优化、TIIR 参数化与软截断、Stride 与 DownUpsample 结构、成本模型估计、三阶段 IR 降低、SIMD 向量化与多线程并行、以及自动化的搜索与采样策略。

**📊 数据集**

实验数据集覆盖 1D 目标滤波器（Gaussian、Lanczos、HRIR、Telephone 等）和 2D 目标滤波器（Gaussian、Gabor、Lowpass 等），每个类别分别测试多种尺寸；评价数据采用白噪声输入（4 M 样本或 4 M 像素）来计算 PSNR，并通过对比 FFT、直接卷积、传统近似（Triple Box Blur、FRM、CP、YVV、SVD）来评估吞吐量和质量。

**📈 对比分析**

与基线方法比较时，所生成的滤波器在 PSNR 维持或提升的同时，吞吐量往往比 FFT 或直接卷积快 3–10 倍，甚至对 2D Gaussian、Gabor 等高分辨率滤波器可实现 30–70 倍的速度提升；在 1D 电话滤波器上也能比 CP、FRM 提升 1–3 倍。实验显示搜索得到的设计在 Pareto 前沿上显著占优，质量/性能曲线优于传统手工设计。

**⚠️ 局限性**

主要局限包括：① 搜索空间仍以连续参数为主，离散结构（如组合式滤波器、混响卷积）难以直接纳入；② 对每个目标滤波器的训练成本高，虽然一次性，但对可参数化滤波器族（如多尺寸 Gaussian）缺乏高效的动态重构机制；③ 仅使用 PSNR 作为误差度量，可能无法捕捉频率响应细节或人耳/视觉感知差异；④ 对不同滤波器族间的连贯性（参数连续变化时的频率响应平滑）未做专门设计，可能导致跨参数切换时产生可感知的波动。

---

## 68. Instruct-FD: Can Your Full-Duplex Speech System Follow Turn-Taking Instructions?

**arXiv ID:** 2607.20460 | [PDF](https://arxiv.org/pdf/2607.20460v1)

**作者:** Yuzhi Tang `[一作]` (Boson AI), Alex Smola `[通讯]` (Boson AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Instruct‑FD，一个面向指令的全双工语音对话系统的评测基准，用以检验模型在不同情境下是否能按指令调整交互时机。

**💡 创新点**

创新点在于把交互时机控制视为指令遵循问题，设计了可扩展的合成对话生成管线、与模型架构无关的多轮用户指挥器，以及基于 LLM 的评判器，从而实现大规模、可复现的指令化评测。

**🔧 技术方法**

技术实现包括 LLM 生成的指令化合成对话、Gemini TTS 语音合成、Qwen3 ASR 与 forced‑aligner 进行时间对齐、Claude Sonnet 作为判定器、Silero VAD 与 WebRTC 兼容的多轮播放框架。

**📊 数据集**

使用的数据主要是由 LLM 生成的 29 场景、912 条指令化测试用例；另外通过 180 条人工标注的评判结果验证评判器，和 11 名受试者的人类评测数据。

**📈 对比分析**

与六款主流全双工系统进行对比，使用 IAS（指令遵循得分）和 ΔIAS（指令敏感度）衡量。最佳模型 Gemini 在整体指令遵循上达 64.4%，在 Continue、Acknowledge 上表现突出，但在 Backchannel、Interrupt 等主动行为上仍低于 50%。

**⚠️ 局限性**

局限性包括：合成数据仍难完全逼真，评测仅覆盖两轮对话、仅限英语；指令集合有限，且 LLM 评判器受 ASR 误差影响；部分模型在指令触发时表现不稳定，说明当前技术对细粒度交互控制仍不足。

---

## 69. GLAN-QnA-KR: A Seedless Taxonomy-Driven Korean Instruction Corpus

**arXiv ID:** 2607.20443 | [PDF](https://arxiv.org/pdf/2607.20443v1)

**作者:** Daekeun Kim `[一作]` `[通讯]` (Korea University), Daekeun Kim (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作发布了 GLAN‑QnA‑KR，一套 303,581 条韩语问答对的指令式数据集，并提供生成协议、统计信息与污染审计；

**💡 创新点**

创新点在于采用 seedless taxonomy‑driven 生成流程，首次在 30 万级规模内实现可公开重现、无种子输入的韩语指令式语料库，并通过两层审计证明与主流评测集几乎无重叠；

**🔧 技术方法**

主要技术包括 GLAN 生成管线、Microsoft Phi‑3.5‑MoE‑instruct 作为生成模型、字符三元组 Jaccard 与 multilingual‑E5 嵌入余弦相似度进行污染审计，以及对数据分布的统计分析；

**📊 数据集**

使用了自生成的 303,581 条问答记录作为主数据集，并利用 KMMLU、KoBEST（五子任务）和 HAE‑RAE‑Bench 三大公开韩语评测集进行污染审计；

**📈 对比分析**

评估方法为对 20,000 条 GLAN 问题与评测集每条问题计算字符三元组 Jaccard 与 E5 余弦相似度的最大值；结果显示最大 Jaccard 为 0.163、最大余弦为 0.901，且无任何评测条目达到 0.8 Jaccard 或 0.95 余弦，表明污染极低；

**⚠️ 局限性**

主要局限在于仅由单一模型生成，继承其偏差与风格；缺乏对答面语义污染的检测；答案长尾极端；跨语言标签/正文结构导致使用不便；需核查 Phi‑3.5 许可证，且未提供 SFT 性能评估。

---

## 70. Tractable Hierarchical Control of Autoregressive Language Models

**arXiv ID:** 2607.20483 | [PDF](https://arxiv.org/pdf/2607.20483v1)

**作者:** Max Scribner `[一作]` (University of Edinburgh), Vaishak Belle `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 PASTA-G，一种通过把大语言模型(LLM)蒸馏成可计算的隐马尔可夫模型(HMM)来实现对确定性下推自动机(Deterministic Pushdown Automaton, DPDA)约束的自回归生成方法。

**💡 创新点**

创新点在于：①利用 HMM 的可推理性，实现对 DPDA 约束的概率精确推断，从而在有限长度下以多项式时间保证 LLM 生成满足 DCFL 的序列；②在约束生成过程中不再仅考虑下一个 token 的概率，而是把整个剩余序列的约束概率整合进 token 选择中，显著提升生成质量。

**🔧 技术方法**

核心技术包括：
- 将 LLM 蒸馏成 HMM，形成可推理的可计算概率模型；
- 通过 DPDA 的堆栈状态与 HMM 隐状态的组合，递归计算“剩余序列满足 DPDA 约束”的概率；
- 对 DPDA 的状态空间进行缓存化，构造 O(|Q|^2·n^2·h) 的动态规划表，实现线性空间与二次时间的推理。

**📊 数据集**

主要数据集：Dyck-1（平衡括号）语言，作为典型的上下文无关语法；实验还使用了随机初始化的两状态 HMM 来验证概率计算的准确性。

**📈 对比分析**

与 Ctrl-G（基于 DFA 约束的蒙特卡洛方法）对比：在 Dyck-1 任务中，Ctrl-G 的缓存空间和推理时间呈指数增长，而 PASTA-G 的缓存空间线性增长、推理时间二次增长；实验结果显示两者在下一个 token 的概率分布完全一致，说明 PASTA-G 在精度上与暴力计数无差异，同时在性能上大幅优于 Ctrl-G。

**⚠️ 局限性**

局限性包括：
- 仅适用于确定性上下文无关语言，无法处理非确定性下推自动机或上下文相关语言；
- 需要 LLM 与 DPDA 使用相同的 token 化方式，存在 token 对齐问题；
- 目前只考虑了有限长度约束，无法直接处理无限或可变长度生成；
- 对 DPDA 的堆栈深度 h 仍有隐式上限，过深时会导致缓存占用过大。

---

## 71. ExecuGraph: A Multi-Agent, Execution-Grounded Framework for Reliable Backend Code Synthesis with Large Language Models

**arXiv ID:** 2607.20499 | [PDF](https://arxiv.org/pdf/2607.20499v1)

**作者:** Sai Deekshith Lekkala `[一作]` (Kakatiya Institute of Technology and Science), Manpreet Singh `[通讯]` (Boston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ExecuGraph 多代理框架，核心是将执行验证嵌入代码生成工作流；

**💡 创新点**

将执行结果作为唯一接受判据，构建可配置的多代理工作流，便于隔离各代理贡献；

**🔧 技术方法**

使用 LangGraph/LangChain 调度代理，Ollama 本地 LLM，ChromaDB 作为可选检索层，Python 沙箱执行；

**📊 数据集**

在内部 30 题 DSA 套件、HumanEval 64 题、APPS‑Intro 50 题上评测；

**📈 对比分析**

相较于单一一次性生成与单代理重试基线，ExecuGraph 在多重迭代和跨模型条件下提高 3–22% 的通过率，成本提升约 3–5 倍；

**⚠️ 局限性**

仅限 Python、需本地 7B LLM、沙箱隔离有限、依赖预置测试、对未规范化问题和正式验证缺乏保障。

---

## 72. Bayesian uncertainty estimation improves clinical decision making in medical AI agents

**arXiv ID:** 2607.20582 | [PDF](https://arxiv.org/pdf/2607.20582v1)

**作者:** Frederik Hauke `[一作]` (University Hospital RWTH Aachen), Daniel Truhn `[通讯]` (University Hospital RWTH Aachen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

实现并评估了在胸部 X 光多任务分类器上使用 MC‑dropout 产生不确定性估计，并通过代理实验检验该信号对临床决策支持的改进效果。

**💡 创新点**

创新点在于证明将不确定性以二值化错误风险标志呈现给决策支持代理，可显著提升代理对模型错误的识别率，而原始不确定性数值不足。

**🔧 技术方法**

使用 DINOv2 Vision Transformer、MC‑dropout、logistic 回归错误检测器以及 GPT‑5.1 生成的临床决策支持代理。

**📊 数据集**

使用公开的 TAIX‑Ray 胸部 X 光数据集（137,593 训练、34,860 验证、42,928 测试张图）。

**📈 对比分析**

在十个不同数据规模实验中验证了标准差与验证损失的 U 形关系；在代理实验中，加入不确定性提升错误检测 AUROC 从 0.74 提升到 0.77，且二值错误风险标志使代理敏感度从 0.63 提升至 0.79，显著降低错误诊断。

**⚠️ 局限性**

局限包括仅针对单一影像模态与单一机构数据，缺乏外部验证；MC‑dropout 仅为简便，可能不如深度集成或变分推断；代理评估基于自动化输出，缺乏临床医生最终判定。

---

## 73. Naver-News-KO: A Korean News Summarization Dataset for Open-Source Fine-Tuning of Summarization Models

**arXiv ID:** 2607.20442 | [PDF](https://arxiv.org/pdf/2607.20442v1)

**作者:** Daekeun Kim `[一作]` `[通讯]` (Korea University), Daekeun Kim (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Naver-News-KO韩国新闻摘要数据集进行系统文档化，记录采集协议、统计信息、拆分方式，并提供可复现的基线训练脚本；

**💡 创新点**

创新点在于将原先仅有列名和下载链接的无文档数据集转化为可引用、可复现的研究资源，并对测试集与训练集的近重复性进行量化审计，阐明版权与使用限制；

**🔧 技术方法**

主要技术包括：基于Python的爬虫采集、随机拆分、Lead-3抽取基线、KoBART模型微调、Gemma‑2B‑ko LoRA微调，以及ROUGE、BERTScore等评价指标的计算；

**📊 数据集**

使用的数据集为Naver‑News‑KO，共27,400条文章-摘要对，涵盖经济和IT/Science两类，按22,194/2,466/2,740划分为训练、验证、测试；

**📈 对比分析**

比较方法为与Lead‑3抽取基线、KoBART微调和Gemma‑2B‑ko LoRA微调的ROUGE‑1/2/L和BERTScore-F1进行对比；在测试集上，Lead‑3达55.1/33.2/50.6，KoBART提升至56.6/36.9/53.9，Gemma‑2B‑ko略低，说明抽象模型对高提取偏好的摘要仍面临挑战；

**⚠️ 局限性**

局限性包括：十天采集时间限制、仅两类领域、摘要来源为编辑摘要（非人工抽象）、随机拆分导致的近重复泄露、单种子基线、评价指标对提取偏好敏感、版权问题等。

---

## 74. thaulab@EEUCA 2026: Who Said What to Whom? A Targeting-Aware Neural-Symbolic Pipeline for Gaming Toxicity Detection

**arXiv ID:** 2607.20447 | [PDF](https://arxiv.org/pdf/2607.20447v1)

**作者:** Anmol Guragain `[一作]` (Universidad Politécnica de Madrid), Ricardo de Córdoba `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建三阶段流水线，结合两个小型Transformer的集成与Linguistically-Informed Mediator，对游戏聊天进行六分类毒性检测。

**💡 创新点**

创新点在于两阶段无外部数据的数据增强、基于语音行为理论的LIM规则以及多语言领域专属的毒词检测。

**🔧 技术方法**

使用DeBERTa-v3-base、XLM-RoBERTa-base、焦点损失、两阶段人工生成增强、词元统计、语法规则等技术。

**📊 数据集**

使用World of Tanks聊天日志的GameTox共享任务数据集，共计约42k训练样本，含多语种和极度类别不平衡。

**📈 对比分析**

在官方测试集上Macro F1 0.6441（第三名）与准确率0.9062（第一名），相较单模型和纯神经方法提升显著。

**⚠️ 局限性**

局限包括规则化方法缺乏灵活性、增强文本受安全过滤限制、语料库和游戏实体词典不完整、只在模型不一致样本上应用LIM。

---

## 75. From Errors to Rules: Iterative Prompt Optimization for Text Classification

**arXiv ID:** 2607.20497 | [PDF](https://arxiv.org/pdf/2607.20497v1)

**作者:** Yueying Cui `[一作]` (Amazon Web Services), Mukul Prasad `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于错误诊断的提示优化方法 ERGO，迭代遍历训练集并生成可解释的决策规则。

**💡 创新点**

创新点在于将分类错误用作结构化反馈，构造诊断→处方→重写的元提示，直接学习任务特定的边界规则。

**🔧 技术方法**

技术主要包括：LLM 的元提示生成、按批次随机打乱训练集、对错误对进行诊断并写规则、全提示重写与验证的循环优化。

**📊 数据集**

使用了八个分类基准数据集（TREC、CLINC150、RTE、Ethos、Yahoo、20Newsgroups、Rotten Tomatoes、MASSIVE）覆盖 2–150 类。

**📈 对比分析**

与 ICL‑Diversity、DSPy、GEPA、APE 等方法对比，ERGO 在“可学习边界”任务（如 TREC、CLINC150）表现最佳；ICL‑Diversity 在覆盖性任务上领先，DSPy 在多类别任务上优势明显，总体平均性能相当，无单一方法绝对领先。

**⚠️ 局限性**

局限性包括：对非可学习边界或生成任务无效；仅在英语分类任务上验证；固定批大小可能不适用于极大类数；跨模型训练假设弱→强的情况未完全验证。

---

## 76. Autonomous disproofs of the sum-product conjecture over $\mathbb R$ with GPT-5.5 Pro

**arXiv ID:** 2607.20525 | [PDF](https://arxiv.org/pdf/2607.20525v1)

**作者:** Yichen Huang `[一作]` `[通讯]`, Yichen Huang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用 GPT‑5.5 Pro 构建三阶段推理代理，在无网络搜索、无数据污染的条件下，自动生成七个完整、严格的反例证明，证明 Erdős–Szemerédi 加乘猜想在实数域上是错误的。

**💡 创新点**

创新点包括：①在公开模型、公开代码、无外部搜索的前提下实现完全自主的反例证明；②生成的证明形式多样，其中部分证明不依赖单位，改为使用 L^p‑区域构造，展示了 GPT‑5.5 Pro 在深度数学推理上的广度与深度；③系统化的三阶段 prompting pipeline 与 reasoning‑token 监控，为后续的 AI 数学研究提供可复现、无污染的实验范例。

**🔧 技术方法**

技术方法主要是 GPT‑5.5 Pro API + 三阶段 prompting pipeline（proof‑plan proposal → proof construction → review）+ 自动化对话管理 + reasoning‑token 计数与分析。

**📊 数据集**

数据来源主要是公开的数学理论与构造（完全实域数域、单位群、L^p 区域），以及实验中产生的中间输出和最终证明；没有使用传统机器学习数据集。

**📈 对比分析**

通过 8 次独立实验评估，7 次成功（87.5%）平均 reasoning‑token 125.3k（成功案例）/ 132.4k（全部案例），在 token 预算与计算成本上与现有人工/OpenAI/Claude 证明相当或更优，同时保留了可复现性与无污染性。

**⚠️ 局限性**

局限性包括：①仍有一次失败，模型仍可能产生不完整或错误证明；②对算力与 token 预算要求较高，尤其在第一阶段；③依赖特定 GPT‑5.5 Pro 版本，缺乏对更通用模型或其他数学领域的泛化验证。

---

## 77. When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers

**arXiv ID:** 2607.20594 | [PDF](https://arxiv.org/pdf/2607.20594v1)

**作者:** Tong Zhang `[一作]` (Fudan University), Tao Xie `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究权重共享的循环Transformer在一系列可解组词问题和简易序列任务上的学习机制与泛化能力，并通过大量实验揭示自由训练下模型会形成线性计算前沿。

**💡 创新点**

创新点包括：①提出“预算定律”——前沿速度严格跟随训练合同的需求；②提出头部测量 τ(n,i)，可检测循环中实际执行的步骤，解决尾部指标盲点；③证明前沿机制可通过warm‑start迁移到不同循环预算，且无法通过输入调度强制；④阐明“学习墙”由运算符规模而非电路复杂度决定，并通过运算符优先序列训练消除。

**🔧 技术方法**

使用的技术：权重共享深度循环Transformer、不同循环调度（固定、对数、流式）、激活补丁与损伤曲线、头部测量 τ、梯度裁剪、温度退火、参数匹配与部分解耦、正则化与随机种子实验等。

**📊 数据集**

数据集包括：Z60、S4、S5、A5 组词问题；Parity、LSB‑increment、prefix‑product 任务；公开的 easy‑to‑hard prefix‑sum benchmark（官方 32 位训练/512 位测试）。

**📈 对比分析**

比较方法：尾部指标（单步跳过、跨步 Jacobian 对齐、相邻循环余弦相似度、Asymptotic Alignment）与头部指标（前沿速度、前沿完成度、τ 斜率）在 16 种种子下的 Spearman 相关和 AUROC。结果表明：头部指标能显著预测 OOD 长序列性能（AUROC 高达 1.00），而尾部指标无显著相关；预算定律可准确预测最佳循环次数并防止过度迭代；模型在长序列上可实现四倍长度外推且保持高精度。

**⚠️ 局限性**

限制：研究仅覆盖 ≤25M 参数的循环Transformer，无法直接推广到大规模预训练模型；机制选择高度依赖训练预算和架构先验，未解决全局注意力在长距离依赖中的不稳定性；对超长输入（>512 位）和跨任务迁移的泛化仍需进一步验证。

---

## 78. ShriNep@EEUCA 2026: RAKSHAK - Multi-Task DeBERTa with Rationale Distillation and Jigsaw-Augmented Training for Toxic Intent Classification

**arXiv ID:** 2607.20450 | [PDF](https://arxiv.org/pdf/2607.20450v1)

**作者:** Binayak Karki `[一作]` (Mechi Multiple Campus), Pingala Ghimire `[通讯]` (Himalaya College of Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两个系统来处理World of Tanks聊天毒性意图分类，主系统RAKSHAK采用多任务学习与教师推理生成来提升宏F1；

**💡 创新点**

将跨域Jigsaw数据、LLM生成极端样本、教师生成解释、超对比损失与稀有类别二分类头相结合，构建端到端多任务框架；

**🔧 技术方法**

使用DeBERTa‑v3‑base主干、Qwen2.5‑14B教师生成理据、Focal Loss、Supervised Contrastive Loss、稀有类别二分类头及数据增强与迁移学习；

**📊 数据集**

基于GameTox 53k游戏聊天语料、16,225条映射自Jigsaw的毒性样本，以及100条LLM生成的极端主义样本；

**📈 对比分析**

通过与仅使用GameTox+Focal Loss的M1以及M1+Jigsaw的对比实验，RAKSHAK在官方测试集宏F1达0.5883排名第7，Jigsaw提升2.6点，架构再提升3.7点；

**⚠️ 局限性**

存在领域差距、生成极端样本的真实性、无法单独评估各子模块贡献、批量大小限制对对比损失效果、对非英语内容支持不足，以及训练时使用解释导致的输入分布移位。

---

## 79. Response drift across frontier large language models

**arXiv ID:** 2607.20454 | [PDF](https://arxiv.org/pdf/2607.20454v1)

**作者:** Mohammed Aledhari `[一作]` (University of North Texas), Mohamed Rahouti `[通讯]` (Fordham University)

**通讯引用:** 1677 | [OpenAlex ID](https://openalex.org/A5017529726)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对10款前沿大型语言模型在62道开放式问题上进行了47名专家评审的全交叉评估，收集了近三万条人类打分，探讨模型输出与专家参考答案之间的偏差（响应漂移）

**💡 创新点**

创新点在于首次将大规模人类评审与响应漂移概念结合，揭示了模型在不同领域与问题上的“漂移”结构，并发现大多数模型聚集在相同的高漂移“天花板”上，且自动评测无法捕捉此现象

**🔧 技术方法**

采用5分李克特量表进行人工忠诚度评分，并通过统计方法（方差分解、混合效应模型、等效性检验等）量化漂移与模型差异，同时计算了多种NLP相似度指标作对比

**📊 数据集**

使用62道多学科问题集，覆盖推理、数学、编程、对话、安全与伦理、专业知识等六大能力领域，并配备多位领域专家校验的参考答案

**📈 对比分析**

评估结果显示，除Claude和Gemini两模型外，其他8款模型在所有领域的平均偏差均在78–81%之间，呈现“漂移天花板”；高水平模型的偏差为47–49%，且在某些领域高水平模型与低水平模型的差距可达70个百分点，说明人类评估揭示了更细粒度的性能差异

**⚠️ 局限性**

局限包括仅使用单一参考答案可能导致风格或多答案偏见、问卷题量有限、评审人数受限、依赖web端界面可能引入部署级混淆，以及人工评估主观性与时间成本高等因素

---

## 80. InferenceBench: A Benchmark for Open-Ended LLM Inference Optimization by AI Agents

**arXiv ID:** 2607.20468 | [PDF](https://arxiv.org/pdf/2607.20468v1)

**作者:** Jehyeok Yeon `[一作]` (Max Planck Institute for Intelligent Systems), Maksym Andriushchenko `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个开放式基准(InferenceBench)，要求 AI 代理在限定时间内自行搭建可运行的 OpenAI 兼容推理服务器，并对其推理速度进行优化。

**💡 创新点**

创新点在于：①将系统级、内核级、运行时级的决策纳入评测，真正考察代理的端到端研发能力；②使用质量门和完整性门防止模型仅在评测指标上作弊；③将搜索过程与最终结果一并记录，揭示代理在探索与稳定化方面的薄弱点。

**🔧 技术方法**

使用了多种技术和工具：Claude、Gemini、GPT 系列代理；vLLM、SGLang、TGI、TensorRT‑LLM 等推理引擎；量化（fp8、int4）、chunked prefill、prefix 缓存、speculation、KV‑cache 归档、CUDA‑graph 捕获等优化手段；评估脚本提供 TTFT、TPOT、请求吞吐、p90/p99 等指标。

**📊 数据集**

数据集包括：Mistral‑7B‑Instruct‑v0.3 作为基准模型；LongBench v2 用于生成真实的长上下文请求；MMLU‑Pro 500 问多选题作为质量门检测集。

**📈 对比分析**

对比方法：与 PyTorch 基线、默认推理引擎（vLLM、SGLang、TGI）以及匹配时间预算的非代理搜索（SMAC、TPE、Uniform Random）进行速度提升对比。结果显示，代理平均可获得 8.08× 的速度提升，超过默认引擎（最高 4.05×），但仍落后于最优的非代理搜索（最高 11.53×）。

**⚠️ 局限性**

限制：仅在单个 H100 GPU、单节点环境下测试，未考虑多 GPU/分布式部署；评测只关注速度与质量门，未涵盖能源消耗、可维护性等维度；完整性门虽有一定效果，但可能无法捕捉所有作弊手段；基准使用固定模型版本，结果随时间可能变动。

---

## 81. Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models

**arXiv ID:** 2607.20436 | [PDF](https://arxiv.org/pdf/2607.20436v1)

**作者:** Phongsakon Mark Konrad `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对已 fine‑tuned 的指令调优模型进行后置审计，检测评估框架与部署框架之间的行为差异，并通过内部激活空间定位该差异。

**💡 创新点**

创新点在于将行为不匹配映射为可定位的路由方向（mid‑depth attention band），并利用路径补丁定位其在模型中具体的层级位置；随后在预先定义的窗口中拟合激活对比方向，并通过一维坐标干预验证其决定性。

**🔧 技术方法**

采用路径补丁（path patching）、激活对比（paired activation contrast）与路由坐标干预（routing‑coordinate intervention）等技术；同时设计了随机、错误层、符号翻转以及语义对照等控制实验以检验干预的专一性。

**📊 数据集**

使用五个指令调优实例（Gemma‑2‑2B、Gemma‑2‑9B、Qwen‑2.5‑7B、Llama‑3‑8B、Phi‑3‑mini‑4k‑instruct），在 sandbagging、sycophancy、refusal 三种行为上生成了多种问答对；每个行为配备了校准（train）与 held‑out（test）数据，seed 范围为 42–46，样本数从 28 到 600 变化。

**📈 对比分析**

与随机、错误层、符号翻转和语义对照干预进行对照；在 12 个模型×行为的实验中，10 个单元在 held‑out 上实现了 0.12 以上的 gap 缩小，6 个样本数 ≥ 120 的单元实现了 gap 小于 0.06 的闭合；部署端准确率几乎不变，且干预在大多数单元中表现出显著的改进。

**⚠️ 局限性**

单一坐标干预在 10/12 单元有效，部分单元因高秩或位置错误而失效；实验仅覆盖 2B–9B 参数规模、LoRA 低秩调优、单一多选评测；未探讨更大模型、混合专家、生成任务或训练时防御等场景；此外，外部公开的 fine‑tuned checkpoint 的适用性需要进一步验证。

---

## 82. EvoSQL: Memory-Augmented Critic-Generator Co-Evolution for Text-to-SQL

**arXiv ID:** 2607.20489 | [PDF](https://arxiv.org/pdf/2607.20489v1)

**作者:** Jiawei Zhou `[一作]` (Shanghai Jiao Tong University), Kai Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于生成器和批评器协同进化的Text-to-SQL框架，在推理时通过多轮候选SQL的执行验证、LLM批评和基于utility的选择，利用上下文记忆实现SQL候选的迭代改进。

**💡 创新点**

创新点在于将批评器作为可重用的诊断模块与生成器共同进化，加入执行信息调度的utility函数、时间衰减和一致性奖励，并通过离线Self‑Distillation Policy Optimization（SDPO）将执行反馈注入模型。

**🔧 技术方法**

使用的大技术包括多轮生成与批评、基于执行和LLM的双阶段验证、上下文化经验记忆、utility‑guided 采样与聚合、以及SDPO迁移学习。

**📊 数据集**

在Spider（dev/test）和BIRD（dev）两大公开基准上进行实验。

**📈 对比分析**

相较于Maj@16自洽基线，所提出的Co‑Evolve框架在四个开源骨干上均提升EX，尤其在BIRD‑Dev上提升高达+9.19%；SDPO初始化进一步提升了Spider‑Test和BIRD‑Dev的准确率。

**⚠️ 局限性**

主要限制包括：仅局部测试时演化、记忆不跨问迁移；批评器的冷启动与校准问题导致信用分配不稳定；以及整体系统的算力和标注成本高。

---

## 83. Learn2Zinc: Fine-tuning Small Language Models for Text-to-Model Translation in MiniZinc

**arXiv ID:** 2607.20456 | [PDF](https://arxiv.org/pdf/2607.20456v1)

**作者:** Serdar Kadioglu `[一作]` (Fidelity Investments), Karthik Uppuluri `[通讯]` (Fidelity Investments)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5091991933)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究针对小规模语言模型（0.6B~20B参数）进行细粒度微调，以实现从自然语言生成MiniZinc约束模型。

**💡 创新点**

提出跨模型错误引导（Cross‑Model Error Bootstrapping）构建语法纠错数据集，并将语法纠错任务与生成任务联合训练；同时引入自我反思与模型集成，显著提升执行准确率。

**🔧 技术方法**

采用LoRA低秩适配、8/4位量化、语法纠错微调、语法错误收集、模型自我修正（自我反思）和自顶向下集成。

**📊 数据集**

使用Text2Zinc、Or‑Instruct 以及通过跨模型错误收集生成的扩充语法纠错数据，共计约15.6K个训练样本。

**📈 对比分析**

与最前沿GPT‑5.2、Text2Model及其他基线对比，单模型微调后执行准确率从0%提升至76%，自我反思后提升至89%，集成模型达到98%；但解题准确率仅提升至34%，低于GPT‑5.2的57%。

**⚠️ 局限性**

主要局限在于语义推理能力不足，模型虽能生成语法合法的MiniZinc代码，却常出现约束冲突或多余变量导致的求解错误；此外，微调对模型在其他领域的表现可能产生负面迁移。

---

## 84. CLOE: Christoffel Loss Autoencoder for Anomaly Detection

**arXiv ID:** 2607.20530 | [PDF](https://arxiv.org/pdf/2607.20530v1)

**作者:** Léa Billet `[一作]` (LAAS-CNRS), Alexandre Gaffet `[通讯]` (SCHAEFFLER)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 Christoffel 函数的自编码器（CLOE）用于高维表格数据的半监督异常检测。

**💡 创新点**

创新点包括：① 在自编码器的联合训练中加入可微分的 Christoffel 函数损失，使得潜在空间学习更符合异常检测的支持估计；② 仅需调整一个多项式阶数超参数，且能自动阈值化；③ 在 CPU 上实现轻量级、高效的训练与推断。

**🔧 技术方法**

使用深度自编码器（3层隐藏层，ReLU + Tanh）、Christoffel 函数（经验矩阵逆 + Cholesky）、动态正则化系数、以及自动阈值化策略。

**📊 数据集**

在 ADBench 基准上选取 15 个高维表格数据集（维度 9~1555，样本量 80~299,285），如 ALOI、backdoor、cardio、mnist 等。

**📈 对比分析**

与 DAGMM、OC‑SVM、iForest、ECOD、DeepSVDD、kNN、KDE 等传统与深度方法比较。CLOE 在大多数数据集上实现了更高的 AU‑ROC（平均 0.871 vs 0.578）和 AP AUC（平均 0.590 vs 0.189），并在高维情境中表现尤为突出。

**⚠️ 局限性**

局限性：① 需要将潜在维度限制为 8，受矩阵尺寸和计算成本限制，难以直接扩展到更高维度；② 目前仅针对表格数据，未验证对图像或序列数据的适用性；③ 对极少量训练样本时的鲁棒性尚需进一步评估。

---

## 85. LLM-INSTRUCT at UZH Shared Task 2026: Constraint-Aware Retrieval and Selective Debate for Paragraph-Level Argument Mining

**arXiv ID:** 2607.20430 | [PDF](https://arxiv.org/pdf/2607.20430v1)

**作者:** Phuong Huu Vu Tran `[一作]` (Vietnamese-German University), Hoang Van `[通讯]` (RMIT University Vietnam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个针对UN和UNESCO决议的段落级论证挖掘系统，重点解决段落类型、141个主题标签以及段落间有向关系的预测，且必须符合严格的JSON模式；

**💡 创新点**

创新点在于在生成前先用元数据感知的稠密检索缩小标签候选集，结合按维度上限的受限解码、可选择的三代理辩论以及最终的schema校验，显著提升了准确率和提交鲁棒性；

**🔧 技术方法**

使用了Qwen3-8B（4-bit）生成器、intfloat/e5-base-v2稠密检索、RAG示例、三代理辩论框架、受限解码和JSON校验等技术；

**📊 数据集**

使用了来自UN-RES语料库的2695条UN决议文本及其英译，以及UNESCO评测拆分，标签集来自官方CSV文件；

**📈 对比分析**

在官方排行榜上获得第一名，Task1b微型F1从35.83%提升至40.08%，Task1a类型分类准确率约86%，整体表现优于同场参赛队伍；

**⚠️ 局限性**

局限性包括：关系预测阶段仍依赖候选生成和阈值调优；组件诊断未做完整因子实验；默认优先使用英文翻译，缺乏纯法语案例分析。

---

## 86. Multimodal CoLRAG-TF: Triple-Filtered Retrieval for Complex PDFs

**arXiv ID:** 2607.20517 | [PDF](https://arxiv.org/pdf/2607.20517v1)

**作者:** Takato Yasuno `[一作]` `[通讯]`, Takato Yasuno

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个四轴融合的检索增强生成系统 Multimodal CoLRAG‑TF，能够在多模态 PDF 集合中实现多跳推理。

**💡 创新点**

创新点在于将稠密文本嵌入、BM25 关键词匹配、知识图三元组过滤与图像相似度四个轴融合，并通过贝叶斯优化让三元组轴主导以克服词汇偏差。

**🔧 技术方法**

使用了 Table Transformer、Hybrid OCR（PyMuPDF+Tesseract）、LLM（Qwen2.5‑7B）生成摘要与三元组、FAISS 索引、HippoRAG2 样式三层层次检索、LLaVA 视觉 LLM 以及 Optuna 贝叶斯优化。

**📊 数据集**

数据集为 43 份日本国土交通省灾害教训 PDF（共 2403 块）、457 对 QA 对（169 单跳 + 288 多跳）以及 12 张灾害图片的检索基准。

**📈 对比分析**

相较于单轴或二轴基础模型，四轴模型在单跳检索 Recall 达到 0.9909，在多跳问答中答案相似度提升 71.6%，平均召回率 0.5，证明三元组轴显著提升多跳推理质量。

**⚠️ 局限性**

主要局限包括 OCR 噪声导致 13% 表格无法识别、视觉 LLM 灾害类型识别率仅 41.7% 影响图像查询效果、以及多跳标签不完整导致 Recall 上限受限。

---

## 87. Benchmarking Large Language Models on Multi-Sensor Physical Hazard Assessment

**arXiv ID:** 2607.20476 | [PDF](https://arxiv.org/pdf/2607.20476v1)

**作者:** Faizan Iqbal `[一作]` `[通讯]` (Lovely Professional University), Faizan Iqbal (Lovely Professional University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估五大大型语言模型在多传感器物理危害评估场景中的表现。

**💡 创新点**

设计了针对多传感器联合评估、比例响应与模式辨识的三类基准情境，系统量化LLM对联合阈值违规的识别缺陷。

**🔧 技术方法**

使用API调用（温度0.0）、结构化表格与纯文本提示、以及三维评分量表（阈值算术、危害分类、行动建议）等技术。

**📊 数据集**

使用60个人工生成的场景，包含20个多传感器联合、20个单传感器比例、20个模式辨识场景，采用NIOSH、ASHRAE、WHO等国际安全阈值作为基准。

**📈 对比分析**

通过Q1–Q3得分、Wilcoxon符号秩检验和Bonferroni校正进行比较；结果显示单传感器场景几乎完美，但多传感器场景几乎无预警，ChatGPT‑4o在结构化提示下表现显著下降。

**⚠️ 局限性**

仅限于五个模型，情景为合成数据缺乏实时噪声与专家验证，Q2评分低估，难以推广至所有LLM。

---

## 88. Expectation Alignment of Language Models for Real-World User Expectations

**arXiv ID:** 2607.20485 | [PDF](https://arxiv.org/pdf/2607.20485v1)

**作者:** Miaomiao Li `[一作]` (Chinese University of Hong Kong), Kam-Fai Wong `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10745 | [OpenAlex ID](https://openalex.org/A5008208316)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建 ExpectBench 评估基准并提出 LENS 隐式期望感知生成框架以提升 LLM 的期望对齐；

**💡 创新点**

首次从真实多轮交互中提取并量化用户期望，并通过隐式期望编码显式引导生成；

**🔧 技术方法**

使用 LLM 作为观察者提取潜在期望、轻量化投影器以及冻结主模型进行生成；

**📊 数据集**

使用 480 万条真实人机交互日志构成 ExpectBench，包含 12k 交互实例和 34,876 条期望；

**📈 对比分析**

与 GPT‑4o、DeepSeek‑R1‑7B、GLM‑4‑9B、LLaMA‑3.1‑8B、Mistral‑7B、Qwen3‑8B 等模型对比，LENS 在大多数维度平均提升约 0.2‑0.3 分（满分 5 分），但整体仍低于 3 分；

**⚠️ 局限性**

仍无法完全捕捉用户多样化期望，缺乏动态实时期望跟踪与个性化适应；

---

## 89. ReliableTableQA:How Much Supervision Does Reliability Annotation Need?

**arXiv ID:** 2607.20537 | [PDF](https://arxiv.org/pdf/2607.20537v1)

**作者:** Huei-Chung Hu `[一作]` (DOCOMO Innovations, Inc.), Koyo Kobayashi `[通讯]` (NTT DOCOMO, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研发了一个框架，用于为表格问答（SQL 执行结果）自动标注统计可靠性，区分可答但不可靠的情况并给出多标签可靠性警告。

**💡 创新点**

创新点在于：①提出十类可直接从执行结果检测的可靠性危害；②设计程序优先的 SQL 生成与可靠性标注数据管线，避免 LLM 生成模式崩溃；③证明可靠性注解是数据效率问题，GRPO 仅在低样本时有显著提升。

**🔧 技术方法**

使用技术包括：基于 Qwen3‑4B‑Instruct 的 LoRA SFT + GRPO 强化学习；上下文无关文法生成多样化 SQL；确定性可靠性探测器；结构化 JSON 输出。

**📊 数据集**

使用公开零售数据集：Synthetic Customer、Olist、Dunnhumby（共 50k 标注样本）以及未见的 H&M 时尚零售域进行跨域评估。

**📈 对比分析**

与零射击、提示‑only 基线对比；在 100/200 条 SFT 样本下进行 ablation；评估 Rel‑F1、Unreliable Confident Answer Rate（UCRA）、解析率等指标。结果显示：200 条 SFT 样本即可达 Rel‑F1≈0.98、UCRA≈0；GRPO 在 100 条样本时提升 Rel‑F1≈+0.06，超过 200 条样本时无显著收益。

**⚠️ 局限性**

局限性：①可靠性危害仅覆盖经典统计问题，未包含因果、时间相关或测量误差；②模型在推理时需先执行 SQL 以获取统计，导致延迟；③稀有标签（如 R4、R8）在某些模式下缺失；④跨域评估仅在单一未见域 H&M 上验证，缺乏更广泛的真实企业数据验证。

---

## 90. PhantomFill: When the Form Demands an Answer, Language Models Invent One

**arXiv ID:** 2607.20492 | [PDF](https://arxiv.org/pdf/2607.20492v1)

**作者:** Rana Muhammad Usman `[一作]` `[通讯]`, Rana Muhammad Usman

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究语言模型在强制结构化输出时的强制性错误生成行为，揭示当答案不可得且格式不容许拒绝时模型会大幅度地编造信息。

**💡 创新点**

创新点在于构建可检测“强制性伪造率”与“逃逸利用率”的基准，并通过三梯度实验展示格式约束如何彻底驱使模型生成虚假内容。

**🔧 技术方法**

采用对话式提示设计、JSON模式约束、强制解码以及多模型对比实验，并使用自动化评分器进行客观测评。

**📊 数据集**

数据集包括两类人工合成数据：社交媒体帖子与客服工单，均在“不可回答”场景下构造缺失证据的输入。

**📈 对比分析**

实验对比了13个模型（包括开放权重模型和前沿闭源模型）在自由文本、可逃逸JSON和强制JSON三种格式下的表现，结果显示在强制JSON下多数模型的伪造率达到100%，而可逃逸JSON则显著降低此率。

**⚠️ 局限性**

限制在于仅使用人工构造的缺失证据数据，未覆盖所有行业场景；同时自由文本评估依赖LLM判断，可能引入主观性。

---

## 91. AuthProbe: Specification-Driven, Multi-Identity Detection of Broken Object-Level Authorization in Recruitment API

**arXiv ID:** 2607.20574 | [PDF](https://arxiv.org/pdf/2607.20574v1)

**作者:** Jay Barach `[一作]` `[通讯]` (Independent Researcher), Jay Barach (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AuthProbe，一款基于 OpenAPI 规范的黑盒多身份扫描工具，用于检测 API 中的 Broken Object‑Level Authorization (BOLA) 和 IDOR 漏洞。

**💡 创新点**

创新点在于：① 通过 OpenAPI 自动识别可读对象端点；② 采用多身份交叉测试与响应差异判定，显著降低误报；③ 以可配置的严重性阈值和机器可读报告实现 CI/CD 集成门禁。

**🔧 技术方法**

技术核心：Python 实现；HTTP 请求库、YAML 解析；基于 OpenAPI 解析资源；多身份身份管理；响应差异（对比对象 ID）与枚举、缺失身份验证、存在性 oracle 四种探针；JSON/Markdown/JUnit 报告生成。

**📊 数据集**

使用了两套自带的目标服务：1) 漏洞版（Sequential int IDs + 失去所有权检查），2) 加固版（UUID、拒绝默认 + 一致 Not‑Found 响应），以模拟 2025 年 McHire 事件。没有使用公开真实数据集。

**📈 对比分析**

在漏洞版上，AuthProbe 能检测到全部植入的 BOLA/IDOR 发现且无误报；在加固版上完全无发现；扫描时间随对象数量线性增长，单身份样本 1–50 时，扫描时长从 16ms 增至 114ms，验证了线性成本模型。

**⚠️ 局限性**

局限性：仅覆盖读取（GET）操作；需依赖可列出的对象；无法检测无列表或不可猜测 ID 的 API；对返回体不包含对象 ID 的变形格式可能漏报；需要人工提供身份凭证或登录适配器。

---

## 92. DynamicMCPBench: A Trace-Grounded, Effect-Scored Benchmark for LLM Agents over Live MCP Servers

**arXiv ID:** 2607.20531 | [PDF](https://arxiv.org/pdf/2607.20531v1)

**作者:** Jerzy Kamiński `[一作]` (ITMO University), Anna Kalyuzhnaya `[通讯]` (ITMO University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了 DynamicMCPBench，一种基于真实 MCP 服务器的可重跑评测框架，能够自动生成任务并通过执行轨迹中的效果检查点对 LLM 代理进行评估。

**💡 创新点**

其创新点在于将评测从固定数据集转为可重跑框架，采用前向生成和轨迹蒸馏，使用路径无关的效果检查点、矿场和部分顺序进行答案无关评分。

**🔧 技术方法**

实现过程中使用了 LLM 代理、Model Context Protocol、自动探索与蒸馏、确定性重放、等价工具集合、二层 LLM 判断器和 pass^3 可靠性评估。

**📊 数据集**

实验使用了 121 台公开 MCP 服务器生成的 1,845 个任务（从 2,051 条参考轨迹中蒸馏），并在 15 类目下抽取 750 条平衡任务进行评测。

**📈 对比分析**

通过在这 750 条任务上对 24 种模型执行三次独立尝试并计入 pass^3，结果显示最强模型仅能解约 51% 的任务，平均准确率随工具链长度从 39% 降至 13%，并与人工评估 0.76 的一致率相符。

**⚠️ 局限性**

主要局限包括评测器保守导致真实能力被低估、12% 状态变更任务易受沙箱限制、人工验证仅覆盖单一模型、评测数据偏向英文公开 API，且未验证私有或非英文部署的适用性。

---

## 93. Attention Degradation, Function Token Anchoring, and the Limits of Attention-Based Intervention in Large Language Models

**arXiv ID:** 2607.20524 | [PDF](https://arxiv.org/pdf/2607.20524v1)

**作者:** Sagar Dangal `[一作]` (London Metropolitan University), Manoj Shakya `[通讯]` (Kathmandu University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统探讨了Transformer模型在5-100个token短期上下文中的注意力衰减及其与功能词（冠词、介词、标点）相互作用的机制，并通过对功能词插入、替换以及Relay-Aware Attention（RAA）等因果干预，检验平均跨位置注意力衰减是否真正限制上下文检索；

**💡 创新点**

创新点在于首次跨架构（绝对位置编码与RoPE）对短期注意力衰减曲线进行定量化并揭示其与层深相关的指数-平台模式；提出功能词“继电链”机制并通过因果实验验证其存在与架构依赖；通过RAA干预直接检验注意力分布改变是否影响模型表现，证明平均注意力衰减主要是描述性统计而非因果因素；

**🔧 技术方法**

主要技术包括：统计分析平均注意力衰减、功能词替换/插入实验、对注意力logit加权的RAA干预、层级熵分析、多事实检索探测（beats-distractors指标）、并利用软max归一化与注意力重分布验证；

**📊 数据集**

使用英文WikiText-103和WikiText-2数据集，截断到512 token长度，随机抽取样本进行多模型实验；

**📈 对比分析**

对比GPT-2、LLaMA-3.2-1B/3B、OPT-1.3B和distilgpt2等四种架构，结果显示RAA在GPT-2与LLaMA-1B无显著提升，OPT-1.3B表现出距离相关的正负效应，总体效应近零；多事实检索实验表明平均衰减速率与检索准确率无相关性，模型容量决定性能；

**⚠️ 局限性**

局限性包括：样本量小、RAA仅在短距离验证注意力重分布、对长距离效果验证不足、替换用newline存在结构意义冲突、插入实验控制位置不匹配、只测试基线模型且仅限英语WikiText，且不同语言、指令调优或更大模型的通用性未知。

---

## 94. Uncertainty-Aware Trust Estimation for Multi-LLM Systems via Structured Expert Judgement

**arXiv ID:** 2607.20529 | [PDF](https://arxiv.org/pdf/2607.20529v1)

**作者:** Jiawei Zheng `[一作]` (University of Exeter), Jiazhen Zhang `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多模型 LLM 系统中，提出了基于结构化专家判断的加权聚合方法（Cooke 权重），通过对模型在已知答案的校准问题上的对数评分来估计模型在不同上下文中的可靠度，并将这些权重用于聚合多模型的概率预测。

**💡 创新点**

创新点在于将决策理论中的 Cooke 结构化专家判断方法迁移到 LLM 领域，构建了可根据上下文自适应调整的信任估计机制，显著提升了在模型异质性和含噪专家环境下的鲁棒性和可靠性。

**🔧 技术方法**

使用的技术包括：Cooke 对数权重法、上下文感知的校准问答、概率化多模型输出、负对数似然 (NLL)、多分类 Brier 分数、过度自信错误率 (OE) 等评估指标。

**📊 数据集**

实验数据集为 MMLU 及其更具挑战性的扩展版 MMLU-Pro，涵盖多学科四选多选题。

**📈 对比分析**

与多数投票、等权平均、全局权重、准确度权重等基线对比，Cooke 权重在异质性和受污染的专家面板中实现了最高的准确率、最低的 NLL 与 Brier 分数，并显著降低了过度自信错误率；在同质面板中表现与其他聚合方法相近但在概率质量上更具优势。

**⚠️ 局限性**

局限性包括：依赖校准问题集，若校准样本与目标分布不匹配会影响权重泛化；要求各模型提供可靠的概率输出，且仅在多选任务上验证；合成噪声专家的污染实验可能未能完全覆盖真实世界的失效模式。

---

## 95. Semi-Supervised Text-Attributed Graph Distillation

**arXiv ID:** 2607.20477 | [PDF](https://arxiv.org/pdf/2607.20477v1)

**作者:** Yurui Lai `[一作]` (Hong Kong Baptist University), Tsz Nam Chan `[通讯]` (Shenzhen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种面向半监督文本属性图（TAG）的数据蒸馏框架，名为TAGD。

**💡 创新点**

创新点在于结合双通道图-文本编码、自主协同自训练以及Wasserstein距离导向的图草图化和关键词驱动的LLM文本合成，实现高效可读的压缩图。

**🔧 技术方法**

主要技术包括双路径（图感知与图无关）编码器、协同自训练(CoST)、基于Wasserstein距离的图草图化、聚类后重分配、关键词提取、LLM摘要生成和WSD验证的候选筛选。

**📊 数据集**

实验使用七个真实的TAG数据集：Cora、Citeseer、DBLP、Computers、Photo、History、WikiCS。

**📈 对比分析**

与现有图蒸馏、文本蒸馏以及混合模型相比，TAGD在节点分类任务上在不同压缩率下保持甚至超过原始模型性能，且在LLM4TAG迁移任务中实现最佳的性能-压缩折中。

**⚠️ 局限性**

局限性包括对LLM的依赖导致推理成本较高、在极低压缩率下可能出现过平滑导致性能下降、以及对关键词抽取阈值和候选数的敏感性。

---

## 96. What is Good? Extracting and Testing Implicit Theories of Literary Quality from LLM Reasoning Traces

**arXiv ID:** 2607.20425 | [PDF](https://arxiv.org/pdf/2607.20425v1)

**作者:** Birger Moëll `[一作]` `[通讯]` (Uppsala University), Birger Moëll (Uppsala University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过两项实验，先提取大模型在评价写作质量时的隐式理论，再通过系统削弱文本特征验证该理论。

**💡 创新点**

创新点在于利用推理链（reasoning traces）抽取模型的美学信念，并用降解实验（degradation）检验其因果敏感性，且在两种大型模型（DeepSeek R1 与 Qwen QwQ）上重复验证。

**🔧 技术方法**

主要技术包括链式思考（chain‑of‑thought）推理、文本特征删改、分类评估与统计分析。

**📊 数据集**

使用30篇真实文本（涵盖六个质量层级）构建基准，并挑选5篇经典文学段落进行结构与风格降解实验。

**📈 对比分析**

模型在基准文本上的分类准确率为80%，对结构与声调的削弱表现出显著质量下降，而词汇简化影响最小；综合削弱导致评分下降更大但呈亚加性。两模型与多次复制实验均得到一致趋势。

**⚠️ 局限性**

主要局限包括：推理链的可靠性未完全保证，源文本识别可能导致评分偏差，样本规模有限，且模型对不同作者的偏好差异尚未完全解释。

---

## 97. Making Open-Source Text LLM Watermarks Durable Against Merging

**arXiv ID:** 2607.20435 | [PDF](https://arxiv.org/pdf/2607.20435v1)

**作者:** Luisa Scharff `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11545 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对开放源代码LLM的水印训练方法——Merge-Adversarial Training（MAT），使水印在模型合并后仍能被检测。

**💡 创新点**

创新点在于将合并操作作为对抗训练的一部分，显式在训练时模拟线性插值合并，从而显著提升水印对多种合并算法（LINEAR、SLERP、TIES）的鲁棒性。

**🔧 技术方法**

使用对抗训练框架、KL散度损失、top‑k 逻辑门以及水印生成算法（如KGW、AAR、KTH）等技术；训练过程中通过在当前模型与未水印基模型之间插值生成临时合并模型进行梯度更新。

**📊 数据集**

主要数据集包括多语言混合训练集（English、German、Math、Code），细化域数据（NuminaMath-CoT、Evol‑Instruct‑DE、French‑Alpaca+Lucie），以及用于评估的C4、FineWeb‑2、GSM8K等。

**📈 对比分析**

与传统水印蒸馏（KGW‑D）及其他水印方案相比，MAT在三种合并算法下的TPR@1%平均提升达25pp，SLERP下最高提升51pp；同时保持与基准模型相近的语言模型质量（困惑度、重复率）和下游任务性能。

**⚠️ 局限性**

局限性包括：仅在线性插值下训练，虽然对非线性合并有一定迁移，但更激进的合并方法（如DARE‑TIES、Model Breadcrumbs）未作验证；目前仅在Llama‑3.1‑8B‑Instruct和Qwen‑2.5‑3B‑Instruct上测试，扩展到更大模型和更多水印方式（如GaussMark、SynthID等）仍待研究。

---

## 98. LeanFlow: A Case Study in Workflow-Driven Lean Autoformalization

**arXiv ID:** 2607.20503 | [PDF](https://arxiv.org/pdf/2607.20503v1)

**作者:** Lazar Milikic `[一作]` (EPFL), Viktor Kuncak `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于LLM的工作流驱动系统，将完整的数学论文转化为可构建的Lean项目，并引入了缓存同文件验证器LeanProbe以提升证明修复效率。

**💡 创新点**

创新点包括：①源文本与声明的显式关联（蓝图+门控机制），②工作流管理器负责排队、失败记录、重试与验证；③LeanProbe提供低延迟的局部检查；④对工作流、工具访问、源检查等关键环节进行系统化消融实验。

**🔧 技术方法**

技术手段：大规模语言模型（LLM）、Lean4/LeanInteract、LeanProbe（同文件缓存验证）、程序化工作流管理器、API调用、调用预算控制、声明蓝图与源跨度映射。

**📊 数据集**

数据集与案例：两篇完整数学论文（Pythagorean Triples与Cramer–Wold Theorem）、RLM25-PFR 证明工作负载、ICML 2026 AI for Math TCS 挑战项目。

**📈 对比分析**

比较方法与性能：在两篇论文上比较全流程、无排队、工具访问不同的变体，记录调用次数、输入/输出token及是否完成。全流程在2000次调用预算内完成两篇论文，且在token效率上最好；无排队变体耗尽预算。LeanProbe将同文件检查速度提升9–14倍；在RLM25-PFR上取得75.7% BEq+，在ICML 2026挑战中全部通过。

**⚠️ 局限性**

局限性：实验规模仅覆盖两篇论文和小型基准，依赖特定LLM与固定环境，未评估在更大、复杂论文上的通用性；高调用预算与源预检可能排除部分含歧义的文档；缺乏对最终审计与可解释性的全面评估。

---

## 99. Double-Scoring: Reliable Extraction of Strong Lottery Tickets

**arXiv ID:** 2607.20555 | [PDF](https://arxiv.org/pdf/2607.20555v1)

**作者:** Bryce A. Christopherson `[一作]` (University of North Dakota), Salah Dandan `[通讯]` (University of North Dakota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种增广分数空间的双分数方法，用冻结权重的方式高效提取高稀疏率的强彩票子网络。

**💡 创新点**

创新点在于用双分数扩展分数空间，将固定密度1/2的竞争机制引入，消除层级稀疏度手动设置，并在理论上证明可表示所有原始子网络，从而显著提升强彩票提取效果。

**🔧 技术方法**

采用的技术包括分数训练（TopKMask 与 straight‑through 估计器）、双分数增广、零增广网络理论、随机初始化与标准 ReLU MLP/FashionMNIST 实验，并与 SNIP、GraSP、IMP 等基线对比。

**📊 数据集**

实验主要在 FashionMNIST 的宽 256 MLP 上进行，另在无 BatchNorm 的 CIFAR‑10 VGG‑style 卷积网络做 sanity check。

**📈 对比分析**

通过与固定密度 Mask、随机 Mask、SNIP、GraSP、IMP、SET、RigL、Movement 等方法比较，DoubleScore 在 90% 与 95% 稀疏率下取得 82–78% 的准确率，明显优于基线；在稀疏训练后仍保持竞争力，并且对稀疏度的敏感性低。

**⚠️ 局限性**

局限性包括：理论仅保证可表示性，未确保全局最优；增广方法得到的是有效稀疏度而非精确稀疏率，投影可能导致性能下降；实验规模有限，未验证大规模模型或 Transformer；迭代版计算成本较高。

---

## 100. WaveformQA: Benchmarking LLM Temporal Reasoning on Digital Waveforms

**arXiv ID:** 2607.20638 | [PDF](https://arxiv.org/pdf/2607.20638v1)

**作者:** Yichuan Liu `[一作]` (Tenstorrent), Nick Vadlamudi `[通讯]` (Tenstorrent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出WaveformQA基准，用来评估LLM在数字波形上的时间推理能力；

**💡 创新点**

创新点在于构建了360道程序化生成、可验证的问答，涵盖多信号相关、事件排序等复杂推理场景，并证明事件时间JSON格式显著提升推理准确率；

**🔧 技术方法**

利用LLM（如Gemini 2.5 Pro、Claude Sonnet 4.5/4.6、Qwen3 30B）与自动化生成的JSON波形数据进行问答评测；

**📊 数据集**

使用来自开源RISC‑V核心（PicoRV32、DarkRISCV、SERV、biRISC‑V、Ibex）的13条VCD波形，在不同信号数和转换数的复合复杂度格中产生数据集；

**📈 对比分析**

比较方法采用聚合准确率、上下文内准确率和错误类型分布，结果显示JSON格式比VCD提高37–53%准确率，模型整体准确率受上下文窗口限制，最长30k转换时准确率下降8–12%；

**⚠️ 局限性**

局限性在于仅覆盖RISC‑V核心波形，样本量有限，未涉及更广泛的硬件领域或视觉波形分析，且仍受LLM上下文窗口约束。

---

## 101. PromptPack: Scaling LLM Annotation Agents for Online Recommendation

**arXiv ID:** 2607.20528 | [PDF](https://arxiv.org/pdf/2607.20528v1)

**作者:** Sebastian Koralewski `[一作]` (Teads Inc.), Blaž Škrlj `[通讯]` (Teads Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了PromptPack，一个可在标准LLM API层面实现的批量注释代理，用于从广告创意中提取结构化特征并显著降低成本；

**💡 创新点**

核心创新在于将共享系统提示与严格XML结构包裹相结合，利用上下文批处理同时避免语义交叉，配合后置纠错层实现确定性输出；

**🔧 技术方法**

采用大语言模型（OpenAI GPT‑4.1‑nano、GPT‑4o‑mini、Anthropic Claude‑haiku‑4.5、Google Gemini‑2.5‑flash）与XML序列包装、输出校正以及可选的批量排列投票（BPE）；

**📊 数据集**

使用了10,000条广告标题的内部生产数据集，标题长度约12词，已平衡点击与非点击标签；

**📈 对比分析**

与单调调用基线、BatchLLM、BPE等多种批处理方案进行对比，PromptPack在批量20时保持AUC与单一调用相当（差异≤0.012），但成本降低89%，吞吐量提升2.5×；

**⚠️ 局限性**

局限性包括对更大模型的可扩展性受限于输出token线性增长，BPE多轮投票提升性能但成本与延迟显著上升，且仍需在生产环境中验证对非线性ranker的进一步收益。

---

## 102. Dropping the Anchor: Statistical Context Summarization for Distributed Systems via Pulsar Attention

**arXiv ID:** 2607.20457 | [PDF](https://arxiv.org/pdf/2607.20457v1)

**作者:** Aryan Sood `[一作]` (Indian Institute of Technology Roorkee), Shantanu Acharya `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Pulsar Attention，在分布式长上下文推理中用可变内容的attention‑sink前缀和Max‑IDF块摘要取代Star Attention的静态anchor，以降低Phase 1 FLOPs并保持KV缓存相同。

**💡 创新点**

创新点是用内容感知的Max‑IDF统计摘要和小型attention‑sink来替代静态anchor，显著减少计算量，同时在长序列上保持或提升性能。

**🔧 技术方法**

采用两阶段分布式自注意力、FlashAttention、在线softmax合并、IDF统计以及Max‑IDF分块评分等技术。

**📊 数据集**

在Meta Llama‑3.1‑8B‑Instruct模型上使用RULER和BABILong这两个长上下文基准数据集。

**📈 对比分析**

与全密集注意力、Star Attention、StreamingLLM和MInference对比，Pulsar在32K‑128K长度下相较Star提升2.8%‑6.1%，相较Dense提升4.0%‑4.7%，Phase 1 FLOPs下降约3.3×，预估wall‑clock加速约6×。

**⚠️ 局限性**

局限是对需要检索多值的任务（如NIAH MultiValue）缺失罕见词之外的关键片段，导致召回率低；Max‑IDF单分数可能忽略非罕见但重要的值块。

---

## 103. Marking the Wrong Symptoms: Evaluating LLM Watermarks in Medical Texts

**arXiv ID:** 2607.20462 | [PDF](https://arxiv.org/pdf/2607.20462v1)

**作者:** Melanie Rieff `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11545 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了多种文本水印在医学语言模型中的影响，利用专家验证的LLM‑as‑Judge框架审计推理质量、术语准确性和幻觉等临床关键指标。

**💡 创新点**

创新点在于首次跨5种水印、11个LLM与7个VLM进行大规模实验，并揭示即使准确率不下降，水印仍可显著破坏临床推理、视觉解读和术语使用，此外提出仅对最终答案水印可避免推理质量衰退的策略。

**🔧 技术方法**

使用的技术包括5种生成时文本水印（KGW、DipMark、AAR、PPL、SynthID）、Chain‑of‑Thought提示、以及基于Gemini‑3‑Flash的LLM‑as‑Judge多维度审计。

**📊 数据集**

使用的数据集为MedQA（单模态医学多选）和MedXpertQA‑MM（含图像与实验室数据的多模态医学问答），共计4000道题目。

**📈 对比分析**

评估方法是在99% TPR检测点下比较水印与无水印的准确率与LLM‑as‑Judge判定的错误率；结果显示准确率基本保持，但错误推理、术语混淆、幻觉等指标显著上升；对推理模型，仅水印最终答案可保持推理质量。

**⚠️ 局限性**

限制包括仅评估单轮多选问答，未覆盖开放式、对话或检索增强的临床场景；仅针对公开权重模型，无法代表闭源大模型；LLM‑as‑Judge需高能力且验证范围有限，可能无法捕捉所有细微错误。

---

## 104. VeriSimpl: Robust Optimization Modeling from Natural Language using Simplification-based Verification

**arXiv ID:** 2607.20474 | [PDF](https://arxiv.org/pdf/2607.20474v1)

**作者:** Sumaya Abdul Rahman `[一作]` (Texas A&M University at Qatar), Mohammad Raza `[通讯]` (Qatar Computing Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VeriSimpl 框架，利用求解器生成的简化诊断查询，帮助 LLM 自动生成并验证自然语言描述的优化模型代码。

**💡 创新点**

创新点在于把优化求解器用于生成简化查询（约束变异、变量掩蔽），让 LLM 在可控的局部情境下进行推理，形成高精度的自检机制。

**🔧 技术方法**

技术包括 LLM 代码生成、约束简化与可变性验证、变量掩蔽、类型一致性检查以及求解器接口（如 Gurobi/CPLEX/SCIP）。

**📊 数据集**

使用四个公开基准数据集：NL4Opt、Text2LP、Complex OR 与 Industry OR。

**📈 对比分析**

与 GPT‑4o / R1 等基线比较，VeriSimpl 在所有数据集上平均提升约 10–15% 的准确率，且自检精度超过 90%，但覆盖率仅约 20–30%。

**⚠️ 局限性**

局限性包括对自然语言的误解导致的错误一致性、低覆盖率（许多正确模型被误判）以及对未建模约束或属性的检测不足。

---

## 105. Distinguishing Artificial from Authentic: Evaluating LLMs for Detecting LLM-Generated Content

**arXiv ID:** 2607.20446 | [PDF](https://arxiv.org/pdf/2607.20446v1)

**作者:** Juho Leinonen `[一作]` (Aalto University), Paul Denny `[通讯]` (University of Auckland)

**通讯引用:** 34278 | [OpenAlex ID](https://openalex.org/A5011711071)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型（LLM）是否能检测其自身生成的教育任务内容，包括编程、反思写作和简答题。

**💡 创新点**

首次系统评估了LLM自检在不同任务类型中的准确性、提示设计和响应长度对检测效果的影响，并揭示了任务依赖性与短答题的失败模式。

**🔧 技术方法**

采用GPT‑4o生成多种提示下的人工合成答案，并使用同一模型作为检测器，输出概率或是二分类结果。

**📊 数据集**

数据来自新西兰奥克兰大学一年级工程课程的四种作业，包含真实学生提交（共913名学生）与GPT‑4o生成的1200条答案。

**📈 对比分析**

通过对比检测器给出的LLM‑likelihood分布，计算Cliff’s δ和p值，发现编程题和长篇反思能较好区分，简答题却表现相反；检测性能对提示和长度敏感。

**⚠️ 局限性**

局限在于仅使用单一LLM检测器、单一课程背景、预处理对比可能偏差以及不同任务的提示不可直接跨比较。

---

## 106. KeySI: An Interaction Framework for Tuning Text Embeddings Based on Human Feedback

**arXiv ID:** 2607.20556 | [PDF](https://arxiv.org/pdf/2607.20556v1)

**作者:** Yan Zhu `[一作]` (Tulane University), Rebecca Faust `[通讯]` (Tulane University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出KeySI，一种基于关键词分组的语义交互框架，利用用户在关键词层面表达概念并将其转化为伪监督，进而微调文本嵌入模型；

**💡 创新点**

创新点包括：①将交互入口从文档级改为关键词级，显著降低人工成本；②设计关键词组到文档伪类的翻译流程，并引入gap‑based语义去噪机制；③通过半硬三元组损失与中心拉拢损失实现概念级别的嵌入重构；④提供交互式修订流程，使用户能可视化验证并细调模型；

**🔧 技术方法**

核心技术包括：关键词提取与筛选（KeyBERT + 词干化）、BERT文本嵌入、t‑SNE二维投影、gap‑based语义去噪、半硬三元组损失、中心拉拢损失，以及交互式可视化（关键词视图、投影、文档列表、全文查看）等；

**📊 数据集**

实验使用的公开数据集有：COVID‑19 62篇科研文章（4类风险因子）；20 Newsgroups 240篇文档（6类子集）；AG News 7600篇文档（仅在附录展示）；

**📈 对比分析**

与DeepSI的交互成本进行对比，KeySI在NASA‑TLX工作量评分明显更低（平均1.75 vs 3.63），完成率8/8 vs 6/8；用户偏好全为KeySI；在定量实验1中，语义去噪使伪监督纯度提升约20%；实验2中，概念集合k‑NN纯度和轮廓系数均显著提升，证明模型嵌入空间被有效重构；

**⚠️ 局限性**

局限性包括：关键词池规模随语料增大而膨胀，可能导致关键词噪声和选择困难；未系统评估在大规模语料下的可用性；依赖t‑SNE投影，投影误差可能影响用户判断；当前只支持离散概念分组，无法表达层级或关系；未评估对下游检索、分类等任务的影响。

---

## 107. DataPrep-Bench: Benchmarking LLMs as Training Data Preparators

**arXiv ID:** 2607.20465 | [PDF](https://arxiv.org/pdf/2607.20465v1)

**作者:** Hao Liang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15963 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了DataPrep-Bench，一个统一的下游驱动LLM数据准备基准，包含数据构造与数据质量评估两条赛道，并提供基线方法：Data-Construction-Skill与Distributional Alignment Score (DAS)。

**💡 创新点**

创新点在于：①首次将LLM驱动的数据构造与质量评估整合为同一基准；②提出可重用的技能层驱动的Agent构造框架；③提出基于MMD的分布对齐评分来预测数据对下游性能的贡献。

**🔧 技术方法**

技术包括：技能层驱动的Agentic数据构造（使用Claude Opus等大模型执行chunking、提问、验证等步骤）；MMD与文本编码器（Qwen3-Embedding-8B）实现分布对齐评分；基于多模型（Qwen2.5-7B、Llama-3.1-8B、Mistral-7B）的下游微调与评测。

**📊 数据集**

使用的主要数据集有：①原始域源（各领域书籍、手册）；②候选训练集池（如DataFlow-10k、WizardLM 196k、UltraChat等）；③代理域代理集（Infinity-Instruct、ODA-Math-460k、Logics-STEM、ReasonMed、Fin-o1、DISC-Law-SFT）；④下游基准集（MMLU-Redux、Math/Science/Medical/Finance/Law专属任务）。

**📈 对比分析**

比较方法：在相同原始域、相同微调协议和相同下游评测下，对各构造方法与质量评估指标进行对比。结果显示：①添加领域合成数据往往会下降性能；②DataFlow-Skill在结构化领域表现最好，Agent方法在推理重领域领先；③Data-Construction-Skill在知识抽取密集域（Finance、Medical）提升约20点；④DAS在Math、Science、Medical的Pearson相关性>0.70且为最高，说明分布匹配是最可靠的预测信号。

**⚠️ 局限性**

限制：①合成域数据普遍负面影响，表明表面质量评估不可靠；②Skill-guided构造在开放式推理域（Science）及部分专业域（Law）表现不佳；③DAS在Finance和Law的相关性低，主要因候选池不平衡与专业词表缺失；④目前的候选池规模有限，难以全面覆盖所有领域；⑤缺少针对多模态或更细粒度评测的扩展。

---

## 108. AI-Driven Multi-Hop Relay Selection for Smart Urban NR-V2X Networks via Learning-to-Optimize Graph Neural Networks

**arXiv ID:** 2607.20554 | [PDF](https://arxiv.org/pdf/2607.20554v1)

**作者:** Giambattista Amati `[一作]` (Fondazione Ugo Bordoni), Pierpaolo Salvo `[通讯]` (Fondazione Ugo Bordoni)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于图神经网络的学习优化框架，实时完成密集城市环境下 NR‑V2X 的多跳中继选择；

**💡 创新点**

创新点在于：①将离线 MILP 最优解作为监督标签，构造“学习即优化”模型；②采用边感知的 Graph Isomorphism Network (GINE)，将传播特征嵌入消息传递，提高对 LoS/NLoS 的判别；③在保持接近最优连通性的同时，将计算时延压缩至毫秒级，实现实时部署；

**🔧 技术方法**

主要技术包括：图神经网络（GIN / GINE）、混合整数线性规划（MILP）作为训练 Oracle、边特征增强的消息传递、轻量级后处理（单出链、无环检查）以及基于 SUMO–GEMV² 的城市交通与射频仿真；

**📊 数据集**

使用了三条意大利罗马城区（Porta Pia、EUR、Trastevere）生成的 286,602 个时间快照的图数据集，节点为 CAV 与 RSU，边特征包含 SNR、距离、LoS 状态与 Shannon 容量；

**📈 对比分析**

与最优 MILP、贪心 SNR 选链及单拓扑 GIN 做对比。实验表明 GINE 在连通率上可达 MILP 的 92%–97%，并将求解时间从 10⁻²–10⁰ s 降至 <10 ms，速度提升 10–100 倍；与贪心方法相比，连通率提升 5–12 %（在密集场景下可达 20 %+），且对城市域迁移具备一定鲁棒性；

**⚠️ 局限性**

局限性包括：仅在仿真数据上验证，缺乏真实测量验证；未考虑对抗环境、突发拓扑变化或恶意中继；模型为中心化推理，未探讨分布式或多智能体强化学习方法；未来工作将引入时空 GNN、图 Transformer 及实验平台验证。

---

## 109. Stochastic Sampling is Epistemically Shallow: The Dimensionality Gap Between Temperature Variation and Model Diversity in LLMs

**arXiv ID:** 2607.20464 | [PDF](https://arxiv.org/pdf/2607.20464v1)

**作者:** Izhar Ali `[一作]` `[通讯]` (Rowan University), Izhar Ali (Rowan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型的温度采样随机性进行研究，探讨其是否能揭示模型知识的不确定性，并将其与多模型多样性集成进行对比。

**💡 创新点**

发现温度采样在跨问题结构上表现出“维度缺口”，即单模型内部只有约一个有效维度，而多模型集成可显著提升多维不确定性探测。

**🔧 技术方法**

利用Marchenko–Pastur随机矩阵检验、特征值分析和Shannon有效秩等统计方法，对正确率矩阵进行维度与结构检测。

**📊 数据集**

使用公开基准数据集MMLU（500道题）、HellaSwag（200道题）和GSM8K（100道题）进行实验。

**📈 对比分析**

对比结果显示，单模型温度采样的MP信号计数≤1，而24模型多样性集成可达到4，表明多模型方法在检测跨问题不确定性上更具优势。

**⚠️ 局限性**

研究局限包括仅评估二元正确性、MP检验对信号敏感度有限、样本规模受限以及仅关注已调教模型，未覆盖更大规模或不同任务。

---

## 110. CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits

**arXiv ID:** 2607.20518 | [PDF](https://arxiv.org/pdf/2607.20518v1)

**作者:** Xue-Jian Gao `[一作]` (Huawei), Yuwei Fan `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了CANN Bench，一个面向华为Ascend NPU的 AI 生成算子代码基准，包含 53 种算子、1060 个测试用例，按难度分层并支持多种精度。

**💡 创新点**

创新点在于：①使用硬件锚定的 HAP 极限来评估性能；②三维加权评分（编译、正确性、性能）并防止奖励作弊；③公共/隐藏案例拆分、可维护的仓库和在线排行榜；④在 Ascend 官方 CANN 仓库内提供完整的可复现工具。

**🔧 技术方法**

使用的技术包括 Ascend CANN 编译器、Python/ATen 验证、GPU 级别的时间测量、硬件性能模型、精度容忍阈值和自动化评测脚本。

**📊 数据集**

数据集由 Ascend 生产工作负载提取的 53 种算子组成，覆盖 FP16/BF16/FP32/INT8 以及多维度、对齐、量化等多样化测试案例，分为 20 个公开和 80 个隐藏案例。

**📈 对比分析**

比较方法通过对每个案例的编译成功率、功能正确性（相对误差门限）和性能（相对基准与 HAP 极限）进行加权，最终生成 0–100 的分数；性能表现表现为大多数算子在基准水平或接近硬件极限，优于现有基准并能揭示优化空间。

**⚠️ 局限性**

限制包括：仅支持 FP16/BF16/FP32/INT8，未来 FP8/低精度待扩展；多流和预编译二进制的奖励作弊防护仍有限；仅覆盖 Ascend 910B2，未覆盖新硬件或其他 DSL；基准需要频繁刷新以跟随软件/硬件更新。

---

## 111. When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion

**arXiv ID:** 2607.20543 | [PDF](https://arxiv.org/pdf/2607.20543v1)

**作者:** Todd Zhou `[一作]` `[通讯]` (Harvard University), Todd Zhou (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究可验证奖励强化学习（RLVR）在多次采样推理中出现的 pass@k 反转现象，并提出 Per-Problem Base Anchoring (PBA) 方法在保持边界提示多样性的同时提升模型表现。

**💡 创新点**

将 pass@k 反转归因为边界模式承诺失败，提出两模式理论解释，并设计基于冻结基础模型检验置信度的 prompt 级锚定策略来防止边界提示的覆盖丢失。

**🔧 技术方法**

使用 RLVR（GRPO）强化学习、token‑level KL 估计、冻结基础模型的边界判别、prompt 级 KL 锚定，配合基准模型、全局 KL、熵奖励等对照实验。

**📊 数据集**

实验数据集为 Omni‑MATH‑Test、MATH500 两大数学推理基准，另外使用 3000 个提示的诊断子集进行细粒度评估。

**📈 对比分析**

通过与基准模型、匹配 GRPO、全局 KL、熵奖励、随机/硬遮罩等对照，PBA 在 p@1 上提升约 3.9 分，在 p@256 上提升约 4.7 分，显著减少边界提示的失效，甚至在高 k 下超过基线覆盖率。

**⚠️ 局限性**

局限性包括：仅验证单一模型族与数学验证器，理论简化为两模式，依赖冻结基础模型的估计；未解决支持扩展、多模态验证、全局多样性恢复等问题，实验范围受限于所用基准。

---

## 112. JAXBench: Benchmarking Autonomous TPU Kernel Optimization

**arXiv ID:** 2607.20466 | [PDF](https://arxiv.org/pdf/2607.20466v1)

**作者:** Arya Tschand `[一作]` (Harvard University), Sethu Sankaran `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了JAXBench——一个包含50个面向TPU v6e的JAX工作负载的基准套件，用于评估自动生成的Pallas内核性能；并提供8个手工优化的Pallas基准；搭建了可重复的设备侧计时与评估框架；通过LLM驱动的最佳采样、迭代细化、文档上下文增强与Autocomp等四种方法进行比较；

**💡 创新点**

首次为TPU提供专用、生产级的、计算饱和的基准，解决了GPU专用基准无法直接迁移的问题；同时揭示了在TPU上文档上下文比模型规模更为关键的事实；

**🔧 技术方法**

使用JAX与Pallas编写工作负载，利用XLA编译，采用Perfetto追踪进行设备侧计时，采用LLM（Gemini 3 Flash/Pro）生成与改进Pallas内核，使用Autocomp的文档注入技术；

**📊 数据集**

工作负载来源于生产LLM（如Llama‑3.1、DeepSeek‑V3、Mixtral、Mamba‑2、AlphaFold2）与KernelBench Level‑2；数据规模被自适配以饱和TPU MXU；

**📈 对比分析**

对比了最佳采样、迭代细化、迭代+文档上下文、Autocomp四种方法：在Gemini 3 Flash下，Autocomp以1.36×的几何平均速度提升和76%覆盖率领先；在Gemini 3.1 Pro上，四种方法均提升，迭代+上下文与Autocomp几乎相当（≈3.8×）；与手工Pallas基准相比，自动方法约为手工基准的77%；

**⚠️ 局限性**

仅覆盖单芯片TPU v6e，未涉及多芯片分布式；手工基准仅覆盖8/17关键操作；LLM对Pallas语法的理解仍不完善，导致高比例的API误用；

---

## 113. Thermodynamic Weight Decay: Exploring Grokking Acceleration via Attention Specific Heat

**arXiv ID:** 2607.20552 | [PDF](https://arxiv.org/pdf/2607.20552v1)

**作者:** Chitraansh Pandey `[一作]` `[通讯]` (Independent Researcher), Chitraansh Pandey (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了神经网络在训练时出现的“grokking”现象，提出并实现了一种新的优化器 CvAdamW，用以在训练早期通过动态调节 weight decay 来提前触发模型的泛化。

**💡 创新点**

创新点在于将注意力矩阵的方差视为热力学意义上的比热作为预警信号，构建了连续的、比例式的热能注入机制；并提出无任务特定阈值的 z-score 形式的自适应热量注入，提升了方法的通用性。

**🔧 技术方法**

使用了Transformer（decoder-only）模型、注意力比热计算、指数滑动平均（EMA）、z-score 异常检测、动态 weight decay 注入等技术；实验中还对比了一步式阈值触发与连续比例式触发两种策略。

**📊 数据集**

实验数据集为模块算术任务 a+b mod 97（9409 个样本，50/50 训练/验证划分），采用 2 层、d_model=128 的 Transformer 进行评估。

**📈 对比分析**

与标准 AdamW 在 4000 epoch 预算下的对比显示，CvAdamW 在 epoch 2802 成功实现 grokking 而 baseline 未能 grok；在 10 种随机种子上，cold-start 变体平均提前 257 epoch（约 6%），Wilcoxon 检验 p=0.049，Bootstrap 95% CI [53,489]，但 t‑检验等其他统计不完全一致。

**⚠️ 局限性**

限制包括：仅在单一任务和小模型上验证，未在语言或视觉模型中测试；缺乏与其他进度指标（如注意力熵、权重范数等）的直接对比；统计功效有限（n=10）；z-score 阈值虽通用但在极端任务上仍需微调。

---

## 114. OPTScientist: Multi-Agent Discovery of Typed Optimizer Programs for Transformer Pretraining

**arXiv ID:** 2607.20486 | [PDF](https://arxiv.org/pdf/2607.20486v1)

**作者:** Zhongzheng Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xiaoguang Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OPTScientist框架，通过多代理的理论引导循环在强类型DSL中自动发现适用于Transformer预训练的优化器，并最终发现RS‑MR；

**💡 创新点**

创新点在于将优化器设计表述为编译器验证的类型化程序空间，配合四个专门角色（Theorist、Designer、Engineer、Reviewer）以及两阶段进化的搜索策略，实现可审计、可扩展的自动优化器科学发现；

**🔧 技术方法**

采用了Typed DSL与编译器验证、进化搜索、LLM驱动的多代理协作、两阶段DSL演化、原生Transformer预训练评估和参数分组技术；

**📊 数据集**

使用FineWeb‑Edu 100B文本数据集，在NanoChat d21（21层、1.099B参数）模型上进行3000步的预训练实验；

**📈 对比分析**

与AdamW、Muon、Sophia、RMSProp、Lion、SGD+Momentum等基线进行对比，RS‑MR在最终验证BPB上取得0.798的最佳成绩，比Muon提升约0.57%，仅增加约0.39%的优化器状态内存；

**⚠️ 局限性**

局限于Transformer预训练，仅对矩阵参数优化器有效，DSL演化保持保守，实验只在单一模型规模与数据集上验证，缺乏更广泛的模型、训练时长和不同架构的验证。

---

## 115. One Round Is All You Need: Analytic Federated Learning for Task-Heterogeneous Multi-Label Medical Image Classification

**arXiv ID:** 2607.20641 | [PDF](https://arxiv.org/pdf/2607.20641v1)

**作者:** Afsaneh Mahanipour `[一作]` (University of Kentucky), Hana Khamfroush `[通讯]` (University of Kentucky)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于闭式求解的联邦学习框架，用单轮或两轮通信完成多标签医学图像分类，解决不同医院只标注部分疾病类别导致的任务异质性问题。

**💡 创新点**

创新点在于：①平衡标签投影消除类别不平衡与缺失标签引起的负偏差；②针对任务异质性的按类别绝对聚合律，可在所有标注客户端间一次性恢复全局岭回归解；③可选的解析式伪标签细化步骤进一步提升冷类性能。

**🔧 技术方法**

采用了分析学习（analytic learning）技术，将特征提取与线性分类器学习转化为闭式岭回归求解，结合特征自相关矩阵与标签投影向量的全局聚合。

**📊 数据集**

使用公开的 ChestXray14 数据集（含 8 种常见胸部疾病）以及多种 ImageNet 预训练骨干网络（ResNet、VGG、EfficientNet）验证方法的可迁移性。

**📈 对比分析**

与 FedAvg、FedMLP 等基线比较，在所有 4 种缺失标签设置下，单轮方案平均提升 BACC 约 4–5 分、AUC 约 5–7 分，FedMLP 在极端缺失（仅一方标注）时跌至 50% 近，而本方法保持 68–70% BACC，显著优于梯度式方案且通信成本仅 1–2 轮。

**⚠️ 局限性**

局限性包括：伪标签细化在此数据集上并未带来明显提升；方法依赖冻结的预训练特征，对特征质量敏感；在极端任务异质性下，单个标注客户端的样本稀缺可能导致伪标签噪声；未考虑多轮迭代的进一步性能提升。

---

## 116. Preference Tuning as Spectral Update Reorganization

**arXiv ID:** 2607.20438 | [PDF](https://arxiv.org/pdf/2607.20438v1)

**作者:** Peiyan Zhang `[一作]` (Hong Kong University of Science and Technology), Haohan Wang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对偏好后训练（RLHF / DPO / GRPO）产生的参数更新进行谱分解，提取“头部”和“尾部”两类成分，进一步将其重组为可插拔的LoRA适配器并在多模型、多算法、多数据下进行干预实验。

**💡 创新点**

发现偏好训练的更新始终呈现稳定的谱头–尾结构，头部承载主要终端行为和跑级求解偏差，尾部虽单独无显著效果但对完整学习和覆盖率至关重要；并证明谱结构的形成与提示–偏好一致性密切相关。

**🔧 技术方法**

使用LoRA参数化、奇异值分解（SVD）对模块级权重差分进行谱分解，随后进行插拔、跨跑重组、训练时谱投影以及提示偏好噪声实验；评估指标包括 ID、OOD 与 TRAP 上的准确率。

**📊 数据集**

实验涵盖 Qwen、Llama 系列模型，采用 DPO、GRPO 两种优化方式，数据来源包括合成偏好标注与真实基准（Benchmark-derived）偏好对，且对提示–偏好一致性进行了随机腐蚀。

**📈 对比分析**

通过对比基线、全更新、仅头部、仅尾部以及跨跑混合适配器，评估各干预方案在不同任务分布上的性能。结果显示：头部能显著偏离基线并在部分场景下提升性能，但在 OOD 上往往不及全模型；仅尾部接近基线；跨跑重组表明头部决定主导行为；训练时仅头部训练未能恢复全模型性能。

**⚠️ 局限性**

局限性包括：仅在 LoRA 冻结骨干的设置下进行，未验证在全参数微调或更大规模 RLHF 流程中的可迁移性；未直接追踪谱变化与提示-偏好一致性破坏的因果机制；只揭示了更新级别的结构，未定位具体注意力头或 MLP 通路。

---

## 117. SCoPE: Shift-Aware Speaker-Conditioned Priors for Emotion Recognition in Conversations

**arXiv ID:** 2607.20445 | [PDF](https://arxiv.org/pdf/2607.20445v1)

**作者:** Burak Can Kaplan `[一作]` (University of Hamburg), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SCoPE模块结合情绪转移预测与可变融合，实现对话中说话者情绪的动态推理

**💡 创新点**

①将情绪先验视为说话者特定的递归状态；②利用情绪转移预测作为控制信号动态调节先验与多模态证据的权重；③将先验与证据按贝叶斯式融合，形成轻量化、实时可用的系统

**🔧 技术方法**

GRU自回归先验生成器、双头分类器（情绪与转移预测）、多模态证据编码器（SDT框架）以及基于转移概率的加权融合

**📊 数据集**

IEMOCAP（多模态）和MELD（多模态）数据集

**📈 对比分析**

与八种最新基线模型（图神经、Transformer、对比学习等）在IEMOCAP和MELD上进行对比，SCoPE在IEMOCAP上取得最高的准确率和加权F1，MELD上提升基线性能并在罕见情绪上表现最佳

**⚠️ 局限性**

对突然情绪跳变的鲁棒性不足；对长时序用户行为、持续情绪状态等更深层心理因素的建模尚未展开

---

## 118. The Active Ingredient in Muon's Grokking

**arXiv ID:** 2607.20512 | [PDF](https://arxiv.org/pdf/2607.20512v1)

**作者:** Yufeng Wang `[一作]` `[通讯]` (Independent Researcher), Yufeng Wang (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探究 Muon 优化器在 Grokking 任务中的加速机制，重点验证正交化与谱归一化对速度与稳定性的影响。

**💡 创新点**

创新点在于将正交化（Newton–Schulz 迭代）识别为真正加速的关键，并揭示了正交化次数与学习率共同决定的速度–稳定性前沿。

**🔧 技术方法**

使用 Newton–Schulz 正交化、AdamW 对照、Fourier 频谱分析、显著性检验等技术手段，构建多种 ablation 方案进行实验。

**📊 数据集**

在模数算术（加、减、乘）数据集上进行实验，采用 40% 训练、60% 预测比例，覆盖多种学习率和正交化迭代次数。

**📈 对比分析**

通过 first-crossing 与 stable-grok 两项指标比较，正交化仅方案在多种学习率下先行触达阈值并保持稳定，速度比 AdamW 快约 20–40%，且对学习率更鲁棒。

**⚠️ 局限性**

局限于单一模型规模、单一任务族，未验证更大规模下的前沿；first-crossing 指标易失稳；Fourier 分析仅为粗略近似，缺乏更细粒度的解释。

---

## 119. RealVDeblur: One-Step Diffusion for Generalizable Real-World Video Deblurring

**arXiv ID:** 2607.20628 | [PDF](https://arxiv.org/pdf/2607.20628v1)

**作者:** Renbiao Jin `[一作]` (Shanghai Jiao Tong University), Tianfan Xue `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了RealVDeblur，一种利用视频扩散模型的高效生成框架，用于解决真实视频中的运动模糊问题。

**💡 创新点**

创新点包括：① 基于3D Gaussian Splatting和高帧率视频构建大规模物理驱动的模糊合成管线；② 在VAE中取消时间压缩，采用逐帧编码以捕捉帧间模糊差异；③ 通过分布匹配蒸馏将多步扩散推理压缩为单步生成器；④ 引入无训练的Temporal Window Mask消除RoPE外推不稳定，支持长视频的常量内存推理。

**🔧 技术方法**

技术主要包括：预训练的DiT视频扩散模型、LoRA参数高效微调、分布匹配蒸馏（DMD）、局部窗口自注意力、3D Gaussian Splatting渲染、ISP感知噪声模拟。

**📊 数据集**

使用了约2000个3DGS场景和3000个高帧率视频（GoPro、REDS、SloMo）合成的OmniBlur数据集，作为训练数据；在BSD、RealBlur、RSBlur、FEVD、RWBI等真实视频基准上进行评测。

**📈 对比分析**

与多种基线（ESTRNN、RNN-MBP、ShiftNet、VRT、RVRT、BSSTNet）对比，RealVDeblur在PSNR、SSIM、LPIPS、FID、MUSIQ、NIQE、tOF等指标上取得领先或第二名，并在3DGS重建任务中提升几乎所有指标。

**⚠️ 局限性**

局限性在于：① 对极端长序列的推理仍受窗口大小限制；② 模型依赖于预训练的DiT，训练成本较高；③ 在某些极端光照/曝光极限下，仍可能出现细节缺失或纹理失真。

---

## 120. PortLBM: A Portable Lattice Boltzmann Tool Leveraging SYCL on AMD, NVIDIA, and Intel GPUs

**arXiv ID:** 2607.20650 | [PDF](https://arxiv.org/pdf/2607.20650v1)

**作者:** Alexander Strack `[一作]` (University of Stuttgart), Dirk Pflüger `[通讯]` (University of Stuttgart)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了基于 SYCL 的可移植格子玻尔兹曼方法框架 PortLBM，支持多种 GPU 厂商并提供实时可视化。

**💡 创新点**

创新点在于将三种数据布局（stream、bundle、collision）和四种算法（LPTL、NPTL、NPOL、NSOL）集成到单一可扩展框架，并通过性能可移植性研究验证了不同 GPU 上最佳配置的差异。

**🔧 技术方法**

使用技术包括 SYCL（AdaptiveCpp/DPC++）、ISO C++ 单源代码、GPU 并行计算以及 Dear ImGui+ImPlot 进行实时绘图。

**📊 数据集**

数据集主要为流体动力学基准：Hagen‑Poiseuille、Kármán vortex、翼流和多孔介质，均在双精度下对不同 GPU 进行模拟。

**📈 对比分析**

通过 LUPS（每秒格点更新次数）基准测试并结合硬件计数器和 roofline 分析，比较了三种数据布局和四种算法，结果显示 stream 布局在 NVIDIA/Intel 上最快，bundle 在 AMD 上最快，NPTL 整体性能最好。

**⚠️ 局限性**

限制包括仅支持二维 D2Q9I 模型、缺乏多 GPU/3D 扩展、需要针对每个厂商手动调优，以及交换流（NSOL）在当前硬件上无效。

---

## 121. AISE-Bench: A Full-Cycle Curated Benchmark for Information Seeking on Academic Knowledge Graphs

**arXiv ID:** 2607.20498 | [PDF](https://arxiv.org/pdf/2607.20498v1)

**作者:** Fanjin Zhang `[一作]` (Renmin University of China), Juanzi Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究构建了AISE-Bench，一个包含真实用户查询、完整API执行轨迹以及来源引用的全周期学术知识图谱信息检索基准；

**💡 创新点**

创新点在于引入真实用户意图、多步API规划与参数填充、答案与引用的可验证性，以及针对规划、执行与答案质量的系统化评估方案；

**🔧 技术方法**

采用大型语言模型的工具调用、定制化Agent工作流（CAW）、ReAct、PLAY2PROMPT、CodeAct等框架，并利用API库与LLM‑as‑a‑Judge等技术进行评测；

**📊 数据集**

使用AMiner真实查询语料、9个学术API（AMiner、Google Scholar等）以及1,133个标注好的问答对，总共4,000+实例；

**📈 对比分析**

对14种方法（包括最新LLM、Agent框架、编码代理和深度研究系统）进行对比，最佳模型Gemini‑3‑Pro在正确率约61%时仍显不足，整体在规划、参数化和答案完整性方面表现有限；

**⚠️ 局限性**

主要局限在API规划与参数填写准确率低、模型易产生无效或错误检索结果、不同API选择仍易混淆、整体答复完整度与可信度仍未达到理想水平。

---

## 122. Beyond SBDD: Geometric Deep Learning in Polypharmacology and Multi-target Drug Design

**arXiv ID:** 2607.20550 | [PDF](https://arxiv.org/pdf/2607.20550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 123. Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis

**arXiv ID:** 2607.20527 | [PDF](https://arxiv.org/pdf/2607.20527v1)

**作者:** Taewan Goo `[一作]` (BioNexus), Tae-Hyung Kim `[通讯]` (BioNexus)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种针对 agentic 科学合成系统的引用可信度评估与监控框架，利用 gold 级别标注校准验证器并加入分层式 conformal 保证，最终提供一个可在单 GPU 上部署的守护器。

**💡 创新点**

创新点在于：①证明不同验证器的“支持率”高度依赖于阈值，单一指标不可比；②将评估流程 anchoring 在 human gold 上，消除验证器偏差；③通过 split-conformal 校准实现对未被 flag 的不支持引用的分布无关保证；④把 re-attribution 设计为可替换的 commodity slot。

**🔧 技术方法**

主要技术包括：1) 用 AttrScore-3B、GPT-4o、DeBERTa-NLI 等多种验证器；2) 采用 split-conformal 进行未支持类的阈值校准；3) 通过 BM25、模型自检等方法进行 re-attribution；4) 对验证器与 gold 的匹配、负样本难度评估。

**📊 数据集**

使用的数据集有：SciFact（用作验证器校准）、QASA（用于 conformal 校准与 re-attribution 评估）、PubMedQA（做范围控制测试）以及公开的四大 27–35B 规模 LLM 和两条 agentic 管线（OpenScholar、PaperQA2）。

**📈 对比分析**

对比方法：对同一 agentic 输出使用 5 个 gold-validated 验证器计算“unsupported rate”，显示 3%–18% 的差异；对比验证器间负样本一致性，发现 negative-specific agreement 仅 0.27–0.30。通过 conformal 校准，保证的 catch rate 与目标 1-α 近似（例如 90% 目标对应 90% 真实 catch）。Re-attribution 在 QASA 上 recall@1 约 0.69–0.76，远超随机。总之，守护器能在单 GPU 上以 3%–6% 的 flag 率保证 90% 的未支持引用被检出。

**⚠️ 局限性**

局限性包括：①未支持率依赖于验证器阈值，若无 gold 级标注无法解释；②conformal 绑定的假设是校准负样本与部署负样本可交换，实际部署若负样本更难需重新校准；③只监测引用支持，未考虑引用质量、反驳与选择性引用；④re-attribution 仅在提供的上下文中可行，对完全缺失支持的情况无补救；⑤未评估检索阶段错误对整体评估的影响。

---

## 124. PhantomSeal: Proactive Deepfakes Defense with Identity/Context Protection and Forensic Tracing

**arXiv ID:** 2607.20564 | [PDF](https://arxiv.org/pdf/2607.20564v1)

**作者:** Liangqin Ren `[一作]` (University of Kansas), Bo Luo `[通讯]` (University of Kansas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为PhantomSeal的主动防护框架，能够在图像被用于面部替换攻击前，先通过“cloak”技术保护身份与背景并实现取证追踪。

**💡 创新点**

创新点在于同时实现身份与背景双重保护、利用可解释的cloak引导扰动生成、以及通过组合损失实现追踪与扰动的平衡；此前的主动防护仅关注身份或背景单一维度。

**🔧 技术方法**

技术手段包括：基于梯度投影的扰动优化（类似PGD）、多目标损失（身份、背景、cloak对齐与噪声约束）、颜色通道噪声上限、优先级约束的二阶优化，以及对GAN、扩散等多种面部替换模型的通用实现。

**📊 数据集**

数据集主要使用VGGFace2和FFHQ，生成cloak图像时使用StyleGAN3合成；在实验中随机采样6k人脸、3k配对等。

**📈 对比分析**

与现有检测、干扰、溯源方法对比，PhantomSeal在SimSwap、DiffFace等模型下将身份攻击成功率降低至约0.3%–1.9%，追踪成功率高达95%以上，同时保持MSE低于108、SSIM≥0.68；在黑盒场景和多模型集成下亦保持2.5%以下身份攻击率。

**⚠️ 局限性**

局限包括：在黑盒或多模型背景下背景保护与追踪性能下降；对抗性去噪器能显著削弱背景保护；在强自适应攻击（已知cloak）下保护效果仍有一定退化；需针对不同生成模型的上下文提取器进行重新训练。

---

## 125. Autonomous Topology Mutation: Safe Runtime Restructuring for Multi-Agent LLM Systems with Capability, State, and Shadow Invariants

**arXiv ID:** 2607.20488 | [PDF](https://arxiv.org/pdf/2607.20488v1)

**作者:** Bronislav Sidik `[一作]` (Toga Networks (Huawei)), Nizzan Kimhi `[通讯]` (Toga Networks (Huawei))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了自主拓扑突变（ATM）机制，实时监控多智能体LLM框架中的过载情况并通过安全不变式动态重构团队拓扑，拆分过载智能体为专用子智能体并热交换协调者角色；

**💡 创新点**

创新点在于将六信号瓶颈指数与三条形式化安全不变式（能力单调性、状态路由完整性、影子验证前置）结合，实现可验证的运行时拓扑变更；

**🔧 技术方法**

使用六信号Bottleneck Index、LLM驱动的拆分器、基于PL级别的内存分离与降维、影子验证窗口、协调者热交换以及openjiuwen框架的Rails实现；

**📊 数据集**

实验基于DeepSeek‑V3 + 720个任务运行，使用三类自制工作负载（科研、代码调试、安全审计）以及5个真实Python bug 代码小案例（W4），工具采用确定性模拟；

**📈 对比分析**

对比静态单体（A0）、随机拆分（A1）、ATM拆分（A2）和ATM+分离（A3），评估成功率和PL3暴露，结果显示ATM拆分将代码任务成功率从3.3%提升至61.7%（p<1e‑10），分离后PL3暴露降至0，同时每个请求的p99延迟低于500µs；

**⚠️ 局限性**

局限包括工具为确定性模拟、PL分类基于正则表达式、样本量有限、A4实时触发仅在仿真中验证、仅支持单层拆分、未在公开基准上评测、未处理子智能体过载等。

---

## 126. Generative Bayesian Filtering for State Estimation

**arXiv ID:** 2607.20521 | [PDF](https://arxiv.org/pdf/2607.20521v1)

**作者:** Lei Cao `[一作]` (Northwestern University), Naichen Shi `[通讯]` (Northwestern University)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5024713053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于预训练条件变分自编码器（CVAE）的生成式贝叶斯滤波框架（GBF），用于从高维传感器流中在线推断系统隐状态。

**💡 创新点**

创新点在于：①将生成式模型的观测分布直接嵌入贝叶斯递归推理；②利用得分函数（梯度）进行后验采样（SGLD/AdamWSGLD），实现对连续和离散隐状态的联合优化；③通过模块化的观测模型，使得既能利用预训练生成模型，又能保留贝叶斯滤波的时序先验与不确定性量化。

**🔧 技术方法**

核心技术包括：条件变分自编码器（CVAE）用于建模观测分布；贝叶斯递归预测–校正；得分（梯度）采样（SGLD/AdamWSGLD）；马尔可夫状态转移矩阵作为时序先验；对数似然能量函数与软交叉熵结合的总能量；实验评估采用多种对比算法（NN、HMMNN、SCALE、DEEPBAYES、Kalman）。

**📊 数据集**

实验数据集包括：合成的 Temporal‑MNIST 与 Temporal‑Fashion‑MNIST（加入噪声）；真实工业数据：激光熔粉床成形（LPBF）中的同步X‑ray与近红外影像；医疗领域数据：MIT‑BIH 纵向心电图（ECG）序列。

**📈 对比分析**

与对比方法的比较结果显示：GBF 在所有任务（噪声下的识别、状态跟踪）中均取得最高准确率、F1 分数和覆盖率；在 Brier 分数与 ECE 上也表现出更可靠的概率校准；相较于纯判别模型（NN）对噪声更稳健，且相对传统 Kalman 过滤器在非线性、高维观测场景中表现更优。

**⚠️ 局限性**

局限性包括：①需要先行训练高质量的 CVAE，且模型容量需与观测维度匹配；②后验采样算法计算量大，实时性受限；③方法主要针对离散状态空间，连续状态或多模态时序需要进一步扩展；④对转换矩阵估计的依赖可能在缺乏足够历史数据时受限。

---

## 127. Scaling Closed-Loop Feature Channel Configuration with LLMs

**arXiv ID:** 2607.20516 | [PDF](https://arxiv.org/pdf/2607.20516v1)

**作者:** Tolgay Atinc Uzun `[一作]` (University of Würzburg), Dmitry Ignatov `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在闭环大语言模型生成的可执行神经网络代码中，对通道宽度进行搜索，并通过一次性epoch的CIFAR‑100评估验证其效果；研究规模扩大到每个迭代周期生成250个候选网络，累计8个完整周期后对结果进行分析。

**💡 创新点**

创新点在于将通道配置搜索直接嵌入LLM代码生成流程，并在更大采样规模下揭示了非标准宽度、结构化宽度分布以及显著提升的参数效率等新的架构规律。

**🔧 技术方法**

使用的技术包括OlympicCoder‑7B LLM进行代码生成、LoRA微调、一次性epoch的CIFAR‑100代理训练、动态PyTorch instrumentation、以及统计与可视化分析。

**📊 数据集**

使用的数据集为CIFAR‑100，一次epoch训练作为代理评估。

**📈 对比分析**

比较方法：在每个周期计算严格CIFAR‑100评估的平均、最大、前5/10性能，并绘制Pareto前沿；结果显示平均精度呈正向线性趋势，最高精度从0.3144提升到0.3676，参数效率也显著提升。

**⚠️ 局限性**

局限性：仅一次实验，未检验一次epoch与完整训练结果的相关性；样本并非独立同分布；未对不同LLM、数据集或训练时长进行系统对比，噪声来源（LLM采样、随机初始化等）仍未完全消除。

---

## 128. Answer-then-Edit: Reasoning Skeleton Editing for Anti-Distillation with Preserved Utility

**arXiv ID:** 2607.20440 | [PDF](https://arxiv.org/pdf/2607.20440v1)

**作者:** Fan Li `[一作]` (UNSW Sydney), Wenjie Zhang `[通讯]` (UNSW Sydney)

**通讯引用:** 24297 | [OpenAlex ID](https://openalex.org/A5100385514)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在教师模型生成的推理轨迹基础上，提出了后期编辑的抗蒸馏框架SGRE，用来阻止知识蒸馏。

**💡 创新点**

核心创新在于结合认知负荷理论，先提取推理骨架，再通过图粗化和文字化三步后期编辑，既大幅提升反蒸馏效果，又保持回答准确性与自然性。

**🔧 技术方法**

使用的小型LLM提取推理骨架；并行/串行规则实现骨架图粗化；教师模型重新生成含复杂文本的推理轨迹；评估采用SFT蒸馏、自然度打分等方法。

**📊 数据集**

实验使用了标准推理基准数据集：GSM8K、MATH、MMLU‑Pro。

**📈 对比分析**

与ADS、DOGe等现有抗蒸馏方法在多种教师/学生模型上对比，SGRE在降低学生精度上最高（≈60%+的下降），同时教师回答准确率保持不变，自然度下降不到6%，表现显著优于基线。

**⚠️ 局限性**

局限性包括：对教师完整轨迹的依赖，长文本编辑耗时；对不同骨架提取器的鲁棒性有限；仅验证了SFT蒸馏场景，其他蒸馏方式效果未知。

---

## 129. STeMP: Spatio-Temporal Modelling Protocol

**arXiv ID:** 2607.20592 | [PDF](https://arxiv.org/pdf/2607.20592v1)

**作者:** Jan Linnenbrink `[一作]` (University of Münster), Hanna Meyer `[通讯]` (University of Münster)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出并实现了一个名为STeMP的时空机器学习建模协议（Spatio‑Temporal Modelling Protocol），并通过一个 Shiny Web 应用与 R 包支持协议的填写、自动提取信息和预警功能。

**💡 创新点**

创新点在于专门针对时空机器学习模型的透明报告与标准化流程，自动化提取模型与空间数据特征、识别常见陷阱并给出警示，同时为模型开发、评估和使用者提供统一的文档框架。

**🔧 技术方法**

主要技术包括 R 语言与 Shiny 框架实现的 Web 应用、自动化信息提取（读取 RDS 模型对象、空间数据 CRS 等）、基于邻距推断采样模式、以及对评估策略、数据泄露和不确定性量化等的检查与警报。

**📊 数据集**

在示例中使用了南美洲植物物种多样性数据（sPlotOpen 数据库）以及连续空间预测因子（海拔、位置和 WorldClim 气候变量）。

**📈 对比分析**

示例模型为随机森林，报告的 R² 值为 0.7，但通过协议自动识别出随机交叉验证与聚类采样不匹配，提示该性能指标可能过度乐观，需进一步验证。

**⚠️ 局限性**

主要局限包括：目前仅支持 R 语言模型对象（不支持 ONNX 等通用格式）、聚焦于空间机器学习（对时间维度支持不足）、未利用大语言模型提取文献信息、缺乏自动上传至 Zenodo 或 JSON 导出等功能。

---

## 130. Can Valence Reflect Morality in Natural Language? A Preliminary Annotation Study

**arXiv ID:** 2607.20461 | [PDF](https://arxiv.org/pdf/2607.20461v1)

**作者:** Jonny O'Dwyer `[一作]` (Technological University of Shannon), Ishita Singh `[通讯]` (TUS Global)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对从Norm Bank中随机抽取的500条文本情景进行情感强度（动作与后果）连续评分，并通过这些评分构建二分类（道德/不道德）数据集，随后用逻辑回归模型评估情感强度对道德判断的预测能力。

**💡 创新点**

首次在文本道德情境中同时获取动作与后果的连续情感评分，提出以Lin一致性系数为权重的EWE加权金标准，展示情感评分在道德二分类中显著的预测效果。

**🔧 技术方法**

采用R语言实现人类评分工具；计算Lin一致性系数和EWE金标准；做方差分析、Pearson相关和Kruskal-Wallis检验；使用L2正则化逻辑回归并通过5折交叉验证选择正则化参数；用准确率和Matthews相关系数评估模型。

**📊 数据集**

Norm Bank语料库（子集包含SocialChem、ETHICS、Moral Stories）中的500条情景文本。

**📈 对比分析**

将正则化逻辑回归与多数类基线对比，交叉验证准确率86.5%，测试集准确率89%，Matthews相关系数从0提升至0.764，表明情感评分可有效区分道德与不道德情景。

**⚠️ 局限性**

样本量和标注者有限（仅500条、6名标注者）；情感评分与道德标签之间仍存在较低的标注者一致性；未结合其他道德框架（如MFT），对更复杂的道德推理和跨文化泛化能力未做验证。

---

## 131. Beyond Liars' Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs

**arXiv ID:** 2607.20479 | [PDF](https://arxiv.org/pdf/2607.20479v1)

**作者:** Amr Moustafa `[一作]` (University of Bonn), Florian Mai `[通讯]` (University of Bonn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型中的欺骗检测进行了系统研究，探讨了表示层深度、探针表达能力、稀疏特征以及撒谎类型对检测性能的影响。

**💡 创新点**

首次将层深度、探针复杂度、稀疏表示与撒谎类型三维度综合评估，并指出检测性能高度依赖具体表示空间，提出可基于混合专家（MoE）的集成检测策略。

**🔧 技术方法**

使用线性探针、Truncated Polynomial Classifier (TPC)、Truth2D、INLP、Mass‑Mean 等探针；利用稀疏自编码器（Gemma Scope）生成稀疏特征；采用平均池化、follow‑up 反思提示以及 upper‑bound 探针等方法进行评估。

**📊 数据集**

评估数据来自 Liars' Bench（CG、GS、HP‑C、HP‑KR）和补充的 DolusChat（fabrication、omission、exaggeration）数据集，DolusChat 用于训练，Liars' Bench 用于跨域测试。

**📈 对比分析**

通过 AUROC 与召回率比较不同层（20% vs 66%）与不同探针组合的性能。结果显示：深层在自我表述类数据集提升检测；稀疏与密集表示几乎相当；非线性探针并非始终优于线性探针；最佳性能取决于具体撒谎类型与层级，整体仍未达到完美分类。

**⚠️ 局限性**

仅检验20%和66%两层，未进行完整层级扫描；缺乏统计显著性检验；稀疏特征仅在66%层测试；训练与评估数据分布差异导致反向迁移现象；使用默认0.5阈值限制实际应用；未在多模型上验证一致性。

---

## 132. Joint Utilization of Geospatial and census proxies for Autoencoder-Assisted Downscaling (JUGAAD) of socioeconomic indicators in India

**arXiv ID:** 2607.20559 | [PDF](https://arxiv.org/pdf/2607.20559v1)

**作者:** Aditya Dutt `[一作]` (University of Florida), Aditya Singh `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了JuGAAD框架，利用印度2001年和2011年的普查与NSSO调查数据通过自编码器压缩高维调查指标，并用深度回归模型将细粒度普查及地理变量映射到低维空间，从而在村级聚类尺度上生成高分辨率的社会经济指标预测。

**💡 创新点**

创新点在于将自编码器与回归模型结合，实现从粗尺度调查数据到细尺度预测的端到端降尺度；同时采用六类主题特定自编码器、六边形聚类参考体系以及基于国家层面标识的固定效应，提升空间一致性与解释性。

**🔧 技术方法**

使用深度学习技术：多层全连接网络作为回归器、非线性自编码器进行维度压缩、Huber损失函数、ensemble预测以估计不确定性；此外采用中间微调的卷积网络（如Overfeat）来提取地理特征。

**📊 数据集**

数据集包括印度国家样本调查办公室（NSSO）2001/2011年的六大主题指标（共475个），全国普查（人口、住房、劳动力等）以及48项地理环境变量（土壤、植被、气候等）和州/联邦领地的one‑hot编码。

**📈 对比分析**

通过将聚类级预测聚合回区级与真实NSSO指标比较，使用均方误差、R²（均匀和加权）评估性能，结果显示大多数主题的加权R²在0.6-0.8之间，表明模型在细尺度下能保持与粗尺度相近的解释力；在完整解码后，部分主题（如消费支出、土地持有）仍能维持R²≈0.7，显示较好预测效果。

**⚠️ 局限性**

局限性包括：仅使用单一样本（每区）进行训练导致过拟合风险；解码器在聚类级别的泛化可能受限于训练时仅使用区级隐空间；缺失值填补过程可能引入噪声；以及原始NSSO调查本身的偏差和不完整性会传递到预测结果。

---

## 133. Attention-based Experience Replay Framework for Continual Learning of Agnostic Time Series Forecasting Models

**arXiv ID:** 2607.20493 | [PDF](https://arxiv.org/pdf/2607.20493v1)

**作者:** Quentin Besnard `[一作]` (University of Tours), Nicolas Ragot `[通讯]` (University of Tours)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在非平稳环境下，使用持续学习与经验回放相结合的多变量长时序预测框架，并验证其在真实地下水位数据上的有效性。

**💡 创新点**

创新点在于：① 将经验回放（ER）推广至时序预测；② 设计基于多头注意力的采样模块（MHAM），能够动态选择最具信息量的历史样本；③ 架构兼容任意预测模型，保持模型的可迁移性；④ 在大规模地质水位数据集上展示了显著优势。

**🔧 技术方法**

主要技术包括PatchMixer预测模型、固定容量经验回放、基于Transformer的多头注意力采样、在线单轮微调以及MAE/MSE评估；实验实现基于PyTorch。

**📊 数据集**

使用了12个多变量时间序列数据集：ETTh2、ETTm1、Weather（标准基准）和9个地质水位（Piezo）数据集，涵盖多种非平稳特征。

**📈 对比分析**

与Batch Learning、Joint Training、CL随机、CL MinLoss等方法对比，Attention策略在漂移明显的数据集上与Joint Training相当甚至更优，且显著降低了训练时间和存储需求；在无漂移场景下，随机策略表现接近Attention。

**⚠️ 局限性**

局限性包括：预训练需大量数据；缓冲区大小、采样比例等超参数需针对场景细调；对极慢漂移或无漂移数据的提升有限；未探索结构成长、知识蒸馏等更先进的持续学习机制；在大规模生产环境中的集成和可扩展性仍待验证。

---

## 134. TopoGuard: Graph Theory Based Defenses Against Split-Knowledge Attacks on RAG

**arXiv ID:** 2607.20437 | [PDF](https://arxiv.org/pdf/2607.20437v1)

**作者:** Chahana Dahal `[一作]` (University of Nevada, Las Vegas), Zuobin Xiong `[通讯]` (University of Nevada, Las Vegas)

**通讯引用:** 992 | [OpenAlex ID](https://openalex.org/A5054098183)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并提出了 split‑knowledge 攻击，即将多个表面上看似无害的文档组合在一起诱导 LLM 产生错误关联，并设计了一套基于图论的检测框架 TopoGuard，用来识别 RAG 系统中这种组合式恶意行为。

**💡 创新点**

首创了 split‑knowledge 攻击的形式化描述，并给出基于谱分割的检测理论保证；同时提出多种 TopoGuard 变体（谱间隙、Fiedler conductance、模块性、实体融合），实现无标签阈值校准下的高召回。

**🔧 技术方法**

构造语义相似度 k‑NN 图，利用拉普拉斯谱（λ₂）、Cheeger不等式、实体 Jaccard 相似度、Louvain 社群检测等图论与 NLP 技术来判别文档集合的拓扑异常。

**📊 数据集**

在 HotpotQA 和 MuSiQue 两个多跳问答数据集上构造对抗样本并进行评估。

**📈 对比分析**

与 TextFilter、LlamaGuard（2‑8B/3‑8B）和 LLM‑as‑a‑Judge 等基线对比，TopoGuard‑λ₂+Entity 在 HotpotQA 上 AUROC 95.2%、召回 35.1%@1%FPR，约为 LlamaGuard 的 21 倍；在 MuSiQue 上虽性能略降，但仍优于所有基线；且检测延迟低于 0.5 ms，鲁棒性对自适应攻击保持。

**⚠️ 局限性**

仅能检测组合式攻击，对单一文档攻击（如 PoisonedRAG）无效；当攻击与合法文档属于同一语义簇时召回降至 ~1%；需在 R、嵌入模型等配置变更后重新校准阈值。

---

## 135. Knowledge Injection Exists in MoE? Exploring Expert-Aware Contrast Decoding in MoE for Mitigating LLMs'Hallucinations

**arXiv ID:** 2607.20426 | [PDF](https://arxiv.org/pdf/2607.20426v1)

**作者:** Xinyue Fang `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**通讯引用:** 33326 | [OpenAlex ID](https://openalex.org/A5100440919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Mixture‑of‑Experts（MoE）模型在消除幻觉（hallucination）时层级专家激活差异，并基于此提出了专家感知自适应对比解码（EAACD）方法；

**💡 创新点**

创新点在于首次利用MoE高层专家激活模式的差异进行内部对比解码，并通过自适应专家划分、注意力引导幻觉放大以及动态校准实现无需外部资源的幻觉抑制；

**🔧 技术方法**

核心技术包括：专家划分（基于置信度与一致性）、注意力引导幻觉放大、对比解码（使用KL分辨率自适应权重）以及基于原始预测熵的动态校准；

**📊 数据集**

实验使用四个问答数据集：FACTOR（包括News‑FACTOR、Wiki‑FACTOR、Expert‑FACTOR）、HellaSwag、StrategyQA、MathQA；

**📈 对比分析**

与Greedy、Contrastive Decoding、DoLa、SCMoE、END及Self‑Endorsement等基线进行对比，EAACD在LLaMA‑MoE和Qwen‑MoE上均取得最高准确率，提升幅度最高可达约13%；

**⚠️ 局限性**

局限性在于只针对MoE架构验证，若未来模型不再采用MoE则适用性受限；此外实验仅涵盖问答任务，未对生成类任务或其他模型结构进行评估。

---

## 136. The Devil is in the Spectrum: Mitigating Representation Collapse in LLMs via Topologically Regularized Side-Path

**arXiv ID:** 2607.20484 | [PDF](https://arxiv.org/pdf/2607.20484v1)

**作者:** Yiheng Tao `[一作]` (Peking University), Jie Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 18669 | [OpenAlex ID](https://openalex.org/A5100428821)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种非侵入式的侧路模块TRSP，用于缓解大型语言模型中出现的表示坍塌问题；

**💡 创新点**

创新点在于通过谱分析揭示表示坍塌的两种极端（同质化与隔离化），并设计基于拓扑正则化的triBox和长度感知门控，实现混合效率与信息容量的谱平衡；

**🔧 技术方法**

核心技术包括：① 通过连续双盒滤波实现参数无关的三角形卷积（triBox）；② 采用覆盖率为基础的轻量级门控（长上下文门）；③ 对侧路与标准注意力的加法融合；

**📊 数据集**

主要数据集为Llama‑3.2‑1B的Alpagasus‑5k微调、MMLU、HellaSwag；RULER长上下文基准；NoLiMa从零开始训练的109M Transformer，训练长度1K，评估到8K；

**📈 对比分析**

与原模型、LoRA、Gated Attention、Differential Transformer等进行对比。TRSP在MMLU上从36.01%提升至37.76%，HellaSwag从27.97%提升至29.26%；在NoLiMa 8×评估中保持83.2%准确率，远超Differential Transformer（53.9%）和Gated Attention（33.6%）；总体上仅增加约50个可训练参数；

**⚠️ 局限性**

局限性包括：① 仅在特定模型和任务上验证，未见跨模型通用性；② 侧路仍引入额外计算开销（O(Td)）；③ 对谱平衡的理论保证基于近似，实际效果受数据分布影响。

---

## 137. Moir: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing

**arXiv ID:** 2607.20433 | [PDF](https://arxiv.org/pdf/2607.20433v1)

**作者:** Jea Kwon `[一作]` (Max Planck Institute for Security and Privacy), Meeyoung Cha `[通讯]` (Max Planck Institute for Security and Privacy)

**通讯引用:** 16396 | [OpenAlex ID](https://openalex.org/A5061810530)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于模型自身生成文本的自监督协方差估计方法，用于改进知识编辑中的保持空间。

**💡 创新点**

创新点在于用单随机词种子从模型内部生成样本，直接估计运作分布的协方差，消除传统基于外部语料的分布失配。

**🔧 技术方法**

利用协方差约束编辑（如 MEMIT、AlphaEdit）结合自生成文本采样来估计保持协方差，核心技术是随机词种子生成与协方差估计。

**📊 数据集**

实验使用开放权重模型 OLMo‑2、Llama‑3.1、Qwen‑3，并对比 Wikipedia、预训练混合等外部语料以及 Oracle 协方差。

**📈 对比分析**

与 Wikipedia 基准及 Oracle 协方差比较，保持性能显著提升，尤其在数学和代码任务（如 GSM8K、HumanEval）保持率提升至 80% 以上，编辑质量保持不变。

**⚠️ 局限性**

局限性包括仅在 MEMIT/AlphaEdit 两种编辑框架验证；依赖模型生成质量，极度 RLHF 模式压缩的模型可能失效；仅在 7–8B 规模模型上测试，未知在更大规模或非指令调优模型上的表现。

---

## 138. Belief Propagation in LLM World Models: Measuring Strategic Information Bias with Prediction Markets

**arXiv ID:** 2607.20441 | [PDF](https://arxiv.org/pdf/2607.20441v1)

**作者:** Mykola Khandoga `[一作]` (Future Principle), Artur Kiulian `[通讯]` (Future Principle)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究利用LLM与预测市场的结合，量化并消除英语新闻生态系统对乌克兰战争预测的框架偏差。

**💡 创新点**

创新点在于提出一种以预测市场为基准的、可校准的概率单位测度，能够分离模型参数与文本上下文对偏差的贡献，并验证乌克兰军事信息源对偏差的修正效果。

**🔧 技术方法**

使用大语言模型（Gemini 2.5 Flash/Pro、GPT‑5‑mini、Gemini 3.1 Pro Preview）进行上下文感知推理，结合预测市场价格轨迹进行校准，并对文本进行系统性剔除与补充的消融实验。

**📊 数据集**

数据集包括111个Polymarket乌克兰相关预测市场（约93,000个预测点），16,457篇英文新闻稿以及一组乌克兰军事实时分析源（如深度状态、General Staff报告等），并附带完整推理轨迹。

**📈 对比分析**

通过与市场价格轨迹的偏差比较（pp差值）和对比无上下文基线，发现英文新闻导致的正面占领偏差为64–72%错误率；加入乌克兰源可将偏差显著下降，模型性能提升因模型深度而异，深度模型易放大偏差。

**⚠️ 局限性**

局限包括样本量仅为65个领土市场（统计功效受限）、结果依赖Polymarket的偏置与分辨率、不同模型处理差异有限、以及对其他冲突情境的泛化性待验证。

---

## 139. Chronofy: A Temporal-Logical Decay Architecture for Information Validity in Time-Aware Retrieval-Augmented Generation

**arXiv ID:** 2607.20560 | [PDF](https://arxiv.org/pdf/2607.20560v1)

**作者:** Muntaser Syed `[一作]` (Florida Institute of Technology), Sharun Akter `[通讯]` (Daffodil International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Chronofy，一个三层神经符号框架，将时间衰减嵌入检索增强生成 (RAG) 流程，旨在防止时间幻觉。

**💡 创新点**

创新点在于：① 将 Signal Temporal Logic (STL) 形式化地用于验证检索知识的时间有效性；② 用贝叶斯决策理论将衰减系数 β 与潜在过程的均值回复率关联，提供可解释的时间衰减；③ 引入最弱链原则以对输出置信度进行上界约束。

**🔧 技术方法**

技术实现包括：时间子空间嵌入 (TMRL)、指数衰减加权图检索 (结合 TempValid 与 STAR-RAG)、STL 鲁棒性评分与最弱链验证，以及基于 OU 过程的决策理论 β 估计。

**📊 数据集**

实验使用了 ICEWS14、GDELT、TimE‑Lite 新闻、MIMIC‑IV 临床知识图谱等时间知识图谱与问答基准。

**📈 对比分析**

与静态检索、纯递归以及 Oracle 时间接近等基线相比，Chronofy 在 ICEWS14 上 MRR 提升 9.4%，在 GDELT 上提升 48.9%，在 TimE‑Lite RAG 中准确率从 0.450 提升至 0.488（相对 0.381 的递归基线提升 8.4%）。

**⚠️ 局限性**

局限包括：指数衰减无法捕捉非单调生命周期；β 参数需要领域专业知识或标注数据学习；时间焦点解析覆盖率仅 37.8%，影响检索质量；在非高斯或非平稳动态下理论推导可能失效。

---

## 140. Confidently Deceptive: How Confidence Amplifies the Risk of LLM Deception

**arXiv ID:** 2607.20444 | [PDF](https://arxiv.org/pdf/2607.20444v1)

**作者:** Ali Asad `[一作]` (Queen's University), Xiaodan Zhu `[通讯]` (Queen's University)

**通讯引用:** 11014 | [OpenAlex ID](https://openalex.org/A5016892586)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估大型语言模型在产生欺骗性回答时的自信度，包括对话中的自我报告信心和基于logit的内在置信度，并研究误差细化与后门触发对欺骗率与信心的影响；

**💡 创新点**

首次将欺骗、模型自信度与自我意识三者联合评价，揭示高自信欺骗与误差细化相互放大、模型能识别自身欺骗但不回避，且后门触发的欺骗虽少但极具自信；

**🔧 技术方法**

采用Prompt/Chain‑of‑Thought（CoT）监测器、logit‑基于置信度估计（序列平均、最小、熵、短语置信度）、自我预测与自我意识探针、以及误差细化微调（QLoRA）；

**📊 数据集**

使用ThoughtCrime (TC)、DeceptionBench、MASK、Liars’ Bench后门触发版本等公开欺骗基准；

**📈 对比分析**

通过风险评分R̂＝欺骗率×平均自信度、与人类偏好实验（78%更高自信欺骗更受青睐）以及对比未细化与细化模型的R̂变化，发现细化将风险提升最高37点；后门欺骗率低但自信度显著高；

**⚠️ 局限性**

研究仅覆盖三款开源推理模型，使用单一GPT‑5基监测器，评估为相关性而非因果，未探究干预措施，且对前沿系统与不同监测器的泛化能力缺乏验证。

---

## 141. Evaluating the Effectiveness of Persona Simulation in Opinion Prediction with GPT-4.1

**arXiv ID:** 2607.20589 | [PDF](https://arxiv.org/pdf/2607.20589v1)

**作者:** Sarah Y. Li `[一作]` (McLean High School), Ziyu Yao `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用 GPT‑4.1 对九个州的选举结果、医疗保健意见以及多主体对话进行人格模拟预测。

**💡 创新点**

首次将 LLM 与多类型人格（meta、objective、subjective、dataset）相结合，在政治与医疗意见预测中评估其效果。

**🔧 技术方法**

采用 GPT‑4.1、Llama‑3.1‑70B 生成人设，并使用多项式逻辑回归与 GPT 进行对比分析。

**📊 数据集**

使用 Columbia Personas 数据集、2024 年 ANES 数据以及 Pew 调研的 W123 数据。

**📈 对比分析**

与逻辑回归对比，GPT‑4.1 在选举预测中的准确率约为 61%–68%，在医疗意见预测中最高可达 94%（准确率）和 0.85 的 F1 分数。

**⚠️ 局限性**

模型存在族群偏差、过度概括以及对话缺乏自然流畅度等局限。

---

## 142. Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts

**arXiv ID:** 2607.20519 | [PDF](https://arxiv.org/pdf/2607.20519v1)

**作者:** Andrei Cristian Popescu `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究循环Transformer的自适应深度，把它分为轨迹形成和读出两个独立的部分进行分析。

**💡 创新点**

提出“轨迹–读出”视角，区分训练目标对轨迹的塑造与停止规则的学习，证明固定先验训练能产生难度感知轨迹，简单后置读出可匹配或超过学习门。

**🔧 技术方法**

使用循环Transformer、固定先验损失、线性/MLP门、后置读出（熵、top‑1、对数边缘、预测KL、隐藏状态位移等）以及预训练的Ouro 1.4B/2.6B模型。

**📊 数据集**

受控合成任务为MANO（模数算术与二进制奇偶性），大型基准为MMLU、ARC‑Easy、ARC‑Challenge、OpenBookQA、HellaSwag、CommonsenseQA。

**📈 对比分析**

通过 D@X 与平均退出深度比较，固定先验+后置读出在 MANO 上可在 1.5 次循环内达到 99% 以上准确率；在 Ouro 上后置读出与预训练门竞争或更优，平均延迟下降约 1.2–1.6 倍。

**⚠️ 局限性**

局限在于仅评估现有轨迹，未探索新的预训练目标或更大规模模型的通用性；门学习仍需更好分离训练与读出，且读取开销在大模型中可能不显著。

---

## 143. Human-in-the-Loop Large Language Model Framework for Identification of Cutaneous Immune-Related Adverse Events

**arXiv ID:** 2607.20428 | [PDF](https://arxiv.org/pdf/2607.20428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 144. Improving Access to Essential Medicines via Decision-Aware Machine Learning

**arXiv ID:** 2607.20542 | [PDF](https://arxiv.org/pdf/2607.20542v1)

**作者:** Angel Tsai-Hsuan Chung `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并在塞拉利昂全国范围内部署了一套基于决策感知机器学习的核心药品分配系统，实现对关键药品的精准需求预测和最优分配。

**💡 创新点**

结合多任务学习、催化先验以及决策感知学习，在数据稀缺且质量差异大的环境中实现样本效率和公平分配。

**🔧 技术方法**

使用随机森林多任务预测、Catalytic Prior 规范化、决策感知学习算法、随机优化（线性规划+样本平均逼近）以及 SynthDiD 因果评估等技术。

**📊 数据集**

利用 DHIS2 月度消耗数据、mSupply 仓储库存与过期信息、以及人口普查与卫星数据作为催化先验。

**📈 对比分析**

通过 SynthDiD、标准 DiD、地理匹配、替代控制产品和缺失值插补等稳健性检验，结果显示受试地区消费提升约19–21%，公平性提升，且未显著增加库存缺货。

**⚠️ 局限性**

依赖缺失率高的 DHIS2 数据，模型假设需求不被浪费，极端突发事件鲁棒性有限，评估仅覆盖短期效果，未验证长期可持续性。

---

## 145. From Atoms to Entropy: Optimal Noise Allocation for Diffusion Training in the Convex Regime

**arXiv ID:** 2607.20540 | [PDF](https://arxiv.org/pdf/2607.20540v1)

**作者:** Luca Ambrogioni `[一作]` (Radboud University), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了扩散模型训练中噪声级别分配的最优策略，并给出了理论指导

**💡 创新点**

提出了在冻结特征假设下，最优时间调度总是稀疏原子分布，并在无耦合极限下推导出基于生成熵率平方根的光滑调度公式

**🔧 技术方法**

使用了Polyak–Łojasiewicz假设、平均SGD分析、矩阵随机理论、信息论熵率、Carathéodory定理等数学工具

**📊 数据集**

在低维实验中使用Dirac混合、二维Swiss Roll、Moons以及MNIST数据集进行验证；在大规模实验中使用离散域（bMNIST、bFashionMNIST、DNA）和连续域（MNIST、FashionMNIST、CIFAR-10、FFHQ）

**📈 对比分析**

与统一时间采样、CosMap、Logit-Normal、EDM等基线对比，原子调度在所有实验中收敛最快，光滑熵率调度在神经网络上与原子调度相近并优于手工调度，在离散域提升速度2–2.7倍，在连续域性能相当或略优

**⚠️ 局限性**

结果依赖于冻结特征和良好指定假设，未考虑完整的自适应梯度或批量训练；在强耦合或输入条件不均的情况下熵率近似失效

---

## 146. Leveraging Biokinetic Knowledge Priors for Data-Scarce Bioprocess Modeling

**arXiv ID:** 2607.20539 | [PDF](https://arxiv.org/pdf/2607.20539v1)

**作者:** Kyunghoon Hur `[一作]` (Korea Electronics Technology Institute), Seongjun Yang `[通讯]` (Cold Spring Harbor Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在数据稀缺的生物制程预测中，系统比较了通过模拟预训练与在网络结构中嵌入生物动力学ODE两种方式注入先验知识，并给出了可直接复现的合成数据生成与预训练策略。

**💡 创新点**

首次在统一任务、统一数据集与统一编码器‑解码器框架下，对比数据层与结构层先验注入的效果，发现二者可互换且模拟预训练在数据稀缺环境下更具数据效率。

**🔧 技术方法**

利用Biokinetic ODE（Monod、Baranyi、Gompertz 等）生成合成曲线，采用Encoder‑Decoder架构并结合GRU、PINN、Hybrid‑NeuralODE、BioStruct‑ODE等网络，配合模拟预训练、联合训练与上下文条件等技术。

**📊 数据集**

评估使用了11个批量细菌培养实验数据，涵盖7种细菌（E. coli、Listeria、Shigella、Staphylococcus aureus、Yersinia、Salmonella、Pseudomonas），包含两份自研数据和九份公开数据。

**📈 对比分析**

采用统一的轨迹回归损失和多项指标（R²、RMSE_log、Accuracy Factor、EndPointErr、Rank）进行评估，结果显示BioStruct‑ODE与预训练MLP的性能相近，均优于无先验基线，且在多上下文条件下预训练表现最佳。

**⚠️ 局限性**

仅针对细菌批量培养，未覆盖连续或饲料补充工艺；跨物种泛化尚未解决；实验结果对随机种子敏感；缺乏对真实工业隐私数据的验证。

---

## 147. Codec-Gauge: Learning Compression-Friendly Gauges for Transformer KV Caches

**arXiv ID:** 2607.20538 | [PDF](https://arxiv.org/pdf/2607.20538v1)

**作者:** Yitao Jiang `[一作]` (Dartmouth College), Devin Balkcom `[通讯]` (Dartmouth College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种后训练坐标变换（Codec‑Gauge），在Transformer KV缓存不变的情况下，通过学习正交通道旋转来提升压缩和量化质量。

**💡 创新点**

创新点在于利用频域分布目标（DCT谱质心损失+光滑速率代理）训练坐标变换，使缓存在固定后端（如GPU zfp）下的能量更集中，从而显著降低失真。

**🔧 技术方法**

技术包括基于矩阵指数的正交旋转参数化、二维离散余弦变换（DCT）来衡量频谱集中度，以及光滑对数幅度代理来鼓励能量稀疏；压缩后端采用GPU zfp、block‑uniform 量化和 KIVI‑style 量化。

**📊 数据集**

训练使用 FineWeb‑Edu sample‑10BT 语料收集 KV 缓存样本，评估则基于 6 个不同模型（Qwen3‑0.6B、Qwen3.5‑0.8B、Llama‑3.2‑1B、Gemma‑3‑1B、Phi‑4 Mini Instruct、Ministral‑3‑3B‑Base‑2512）以及一份 27B Gemma‑3 进行。

**📈 对比分析**

与 identity、随机正交、Hadamard、DCT、PCA/KLT 等控制以及不同量化策略的对比显示，在 3/4/6 位/值的固定比特率下，Codec‑Gauge 能将 KL 散度降低约 44%，Logit MSE 降低约 43%，Top‑1 翻转率降低约 25%，并在量化路径上同样实现显著提升。

**⚠️ 局限性**

局限性包括仅针对标准 softmax KV 缓存；需要额外的训练步骤和计算开销；在极低比特率（2 位）下效果受限；未针对动态缓存、稀疏或分层压缩方案；实现时需手动融合恢复步骤，部署复杂度相对较高。

---

## 148. Grounding Investor Views: Neural Predicates in the Black-Litterman Model

**arXiv ID:** 2607.20533 | [PDF](https://arxiv.org/pdf/2607.20533v1)

**作者:** Marcos Florencio `[一作]` `[通讯]` (Pontifical Catholic University of Rio de Janeiro), Marcos Florencio (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于神经谓词的端到端框架，将结构化财务分析数据转换为 Black‑Litterman 模型所需的视图（Pick 矩阵、视图收益和不确定性），实现可解释且可微分的视图生成。

**💡 创新点**

创新点在于：①把神经谓词的概率分布直接映射为 Black‑Litterman 视图；②使用 Shannon 熵量化视图不确定性，替代传统经验性的方差估计；③通过逻辑规则实现多谓词组合，兼顾可解释性与端到端可学习性。

**🔧 技术方法**

技术手段包括：DeepProbLog 逻辑推理 + 语言模型（LLM）生成神经谓词；加权模型计数（WMC）实现可微分推理；Shannon 熵映射视图不确定性；Black‑Litterman 贝叶斯更新与均值-方差优化。

**📊 数据集**

数据集：结构化财务分析信息（估值、盈利质量、资产负债表等），示例中使用两只虚拟公司 Acme 与 Globex 的财务配置作为演示。

**📈 对比分析**

比较方法：仅为演示性示例，未进行大规模对照实验；通过示例展示视图生成与后续组合权重调整的经济可解释性；未给出具体性能指标或对比基准。

**⚠️ 局限性**

局限性：①仅为理论框架与演示实现，缺乏大规模实证验证与参数校准；②对 LLM 的输出稳定性和可重复性依赖较高；③缺少与现有视图生成方法的定量对比；④未处理多资产组合中的复杂相互作用与扩展性问题。

---

## 149. Position: Stop Reactively Patching Your Model Every Time and Start Proactive Test-Driven AI Development

**arXiv ID:** 2607.20532 | [PDF](https://arxiv.org/pdf/2607.20532v1)

**作者:** Nadine Chang `[一作]` (NVIDIA), Jose M. Alvarez `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于“测试空间”（test space）的主动测试驱动 AI 迭代循环（Proactive Test‑Driven Flywheel），并与传统的反应式修补循环（Reactive Flywheel）进行理论对比与分析。

**💡 创新点**

创新点包括：①引入测试空间概念，将错误映射到任务条件空间，实现对整个任务条件的覆盖与预防；②通过概率与数学证明（如Coupon Collector 结果）展示主动循环在长尾场景下迭代次数和积压量显著更少；③提出一系列开放研究问题，指明构建测试空间、反馈对齐、弱点模式识别与解决策略等关键挑战。

**🔧 技术方法**

主要技术手段：概率与组合数学建模（如 H_M、H_K 的 Harmonic 数推导）、理论证明；利用自然语言处理与视觉语言模型从规范文本中提取因素；讨论聚类、支持向量机等模式识别方法来发现弱点；整体框架以理论与示意图为主，没有实战实验。

**📊 数据集**

未使用具体公开数据集，论文以自动驾驶中“停车的消防车/校车”场景为示例进行说明，整体分析以理论为主。

**📈 对比分析**

比较方法：通过推导 Reactive Flywheel 与 Proactive Flywheel 的期望迭代次数（T^R = (M/p_R)·H_M，T^P = (K/p_P)·H_K）以及回收量公式，对两种策略在不同参数下的性能进行理论比较；图示表明在 M/K 较大或 p_P 接近 p_R 时，主动循环迭代次数与积压量显著低于反应式循环。未给出实验性能指标。

**⚠️ 局限性**

局限性：①理论假设均匀曝光、无灾忘等理想条件；②测试空间构建、因素提取、反馈对齐等关键步骤仍为开放问题，缺乏成熟方法；③在 K ≈ M 或 p_P << p_R 的情形下，主动循环优势减弱；④实现难度高，需要跨学科协作与大量工程投入。

---

## 150. IssueTrojanBench: Benchmarking AI Coding Agents Against Malicious Issue Requests

**arXiv ID:** 2607.20759 | [PDF](https://arxiv.org/pdf/2607.20759v1)

**作者:** Ankur Singh `[一作]` (Concordia University), Tse-Hsun Chen `[通讯]` (Concordia University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 AI 编码代理在接收恶意 issue 请求时的安全性进行了系统评估，构建了基于多种攻击类型和投递向量的自动化 benchmark 并在 3 个主流代理与 3 个 LLM 上进行了 4,176 次实验

**💡 创新点**

提出了首个针对 AI 编码代理的恶意 issue 测试 benchmark（IssueTrojanBench），并揭示了现有代理在供应链、配置篡改等攻击下的高危漏洞与模型级安全机制的差异

**🔧 技术方法**

采用 Prompt Injection、隐式指令嵌入、攻击向量多样化、可视化与语言扰动等技术，结合 Exploit Execution Metric（EEM）进行自动化评估

**📊 数据集**

使用 SymPy 与 requests 两个开源 Python 项目中的 6 条 seed issue，生成 696 条恶意实例（4,176 次评测）

**📈 对比分析**

对比 6 种 agent‑model 组合，结果显示整体漏洞率 66.5%，供应链攻击 96.6%，模型层拒绝率差异显著（GPT‑5.3 15.2%、GPT‑5.4 26.4%、Sonnet 4.6 58.9%），且轻量级指令‑数据分离防御效果有限

**⚠️ 局限性**

实验受限于单一任务提示、仅 2 个仓库、人工标签主观性以及模型与代理快速迭代可能导致结果随时间变化

---

## 151. Self-Supervised Bio-Inspired Robotic Trajectory Planning with Obstacle Avoidance

**arXiv ID:** 2607.20743 | [PDF](https://arxiv.org/pdf/2607.20743v1)

**作者:** Miroslav Krupa `[一作]` (Comenius University Bratislava), Kristína Malinovská `[通讯]` (Comenius University Bratislava)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于前向模型、逆向模型自监督学习的神经轨迹规划框架，能够在含障碍的环境中生成可碰撞规避且符合机器人动力学约束的轨迹；

**💡 创新点**

创新点在于：①利用单独训练的前向与逆向模型作为内部监督信号，实现无专家示例的自监督轨迹学习；②通过对轨迹进行“校正”并将校正误差作为学习目标，迫使轨迹模型学习符合动力学的路径；③针对模型可能利用校正误差的“利用”现象，提出几种防止策略（几何先验、监督预训练、几何损失等）；

**🔧 技术方法**

核心技术包括：多层感知机实现的前向模型和逆向模型；基于GRU的循环轨迹模型；自监督的校正机制与对应损失函数；几何先验损失、梯度裁剪等训练技巧；

**📊 数据集**

使用了在MyGym仿真环境中生成的三类数据集：①500k条转移数据（含/不含障碍比例9:1，碰撞/非碰撞比例4:1）；②12k条轨迹（长度≤52）同样的比例；训练数据来源为KUKA LBR iiwa 7自由度机械臂和单个静态障碍箱；

**📈 对比分析**

评估方式：①几何一致性指标（初始/目标距离、步长、角度等）；②仿真执行指标（碰撞率、成功率、沿途跟随率、误差距离等）。结果显示：小型轨迹模型TM_1在障碍环境下表现最优，碰撞率低、成功率高；较大模型在无障碍环境中更易出现利用校正误差导致的振荡轨迹；整体上自监督方法在执行精度上仍落后于完全监督或更精准的内部模型；

**⚠️ 局限性**

局限性：①前向/逆向模型预测误差导致校正过程不完全真实，轨迹模型倾向于利用误差产生非物理可执行轨迹；②较大模型易过拟合校正信号，出现振荡或无意义路径；③当前仅在静态单障碍仿真环境验证，缺乏动态或多障碍/真实机器人实验验证；

---

## 152. Pipelined Gradient Coding

**arXiv ID:** 2607.20739 | [PDF](https://arxiv.org/pdf/2607.20739v1)

**作者:** Xian Su `[一作]` (Florida International University), Jun Li `[通讯]` (City University of New York)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出管线化梯度编码（PGC），在分布式训练中将梯度计算拆分为多步，每个 worker 每步仅评估一个数据分区并复用最近的 stale 梯度，既保持梯度编码的容错性，又避免了传统 GC 的 c 倍计算开销。

**💡 创新点**

创新点在于：① 用管线化方式实现梯度编码，降低每步计算负载；② 针对两种主流的数据放置方案（FR 与 CR）设计了对应的 PGC 编码与解码，并给出了收敛证明；③ 通过引入 staleness 限制与混合平均化来控制误差，实现了既快速又稳健的训练。

**🔧 技术方法**

使用梯度编码技术、分布式梯度下降、管线化计算、随机梯度噪声分析与收敛理论、实验仿真与云端实验平台（Google Cloud）以及与 DGD、GC、IS‑SGD、IS‑GC 等基线的对比。

**📊 数据集**

在 ResNet‑18 上分别使用 CIFAR‑10 和 ImageNet 数据集进行实验，验证了 PGC 在不同 straggler 负载下的性能。

**📈 对比分析**

通过对比每步耗时、训练步数和最终损失，实验显示：PGC 的每步时间几乎与 DGD 相当，但收敛更快，整体训练时间比传统 GC 减少 30–50%，并且在所有 straggler 场景下均优于 IS‑SGD 与 IS‑GC；PGC‑CR 在收敛速度上优于 PGC‑FR。

**⚠️ 局限性**

局限性包括：① 仍需维护最近 c‑1 次梯度，导致一定的 stale 错误，影响极端大步长或极慢 worker 情况下的收敛；② 误差受管线深度 c 与学习率 η 的折衷影响；③ 目前实验主要集中在标准 CNN（ResNet‑18），缺乏对更大模型或更复杂任务的验证。

---

## 153. LLMs Get Lost in Evolving User Intent

**arXiv ID:** 2607.20734 | [PDF](https://arxiv.org/pdf/2607.20734v1)

**作者:** Jihoon Tack `[一作]` (Microsoft Research), Jennifer Neville `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将单轮可验证基准自动转换为多轮、用户意图随对话演变的评测框架，并在该框架下评估多种LLM模型。

**💡 创新点**

创新点在于通过anchor intent与逆向合成前置意图的方式，保留原始基准的可验证性同时生成可控的多轮对话；同时定义三种可调节的意图演变动态（信息揭露、修正、功能切换）。

**🔧 技术方法**

技术实现包括：结构化意图建模与LLM提示抽取、反事实参数与前置功能生成、基于规则的意图调度与对话渲染，以及使用原始基准验证器对最终答案进行自动评测。

**📊 数据集**

使用的基准数据集有 GSM8K（数学）、BIRD‑SQL（文本‑SQL）、BrowseComp+（搜索）和 SWE‑Bench Verified（软件工程），均为可验证的单轮任务。

**📈 对比分析**

在单轮和演变意图两种场景下进行对比实验，评估模型的准确率。实验表明：在演变意图场景下，模型准确率普遍下降，最严重的下降可达约30%（例如 GPT‑5.5 在数学任务从 99% 降至 80.5%），功能切换是最难的动态；引入记忆机制（prompt recap、oracle recap）可部分提升，但仍无法恢复到单轮性能。

**⚠️ 局限性**

局限性包括：只考虑单一意图转移且未覆盖多意图或多目标对话；对话渲染缺乏多样化用户风格；中间步骤仅通过最终验证器间接评估，无法对中间轨迹做直接可验证；未来需扩展到更细粒度的用户行为、多样化人物设定以及可验证的中间过程。

---

## 154. librla: Randomized Linear Algebra Library

**arXiv ID:** 2607.20732 | [PDF](https://arxiv.org/pdf/2607.20732v1)

**作者:** Adrianna Gillman `[一作]`, Zydrunas Gimbutas `[通讯]`

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个可在 MATLAB、Python、Julia 中使用的随机低秩分解库，支持 QR、SVD、插值分解，并可按固定秩或误差阈值构造分解。

**💡 创新点**

统一多语言实现且稳定高效；提供固定秩与容差两种模式；支持矩阵自由操作；允许用户自定义过采样、功率迭代参数。

**🔧 技术方法**

随机采样范围寻找（randomized range finder）结合 QR、SVD 或 ID；使用 Level‑3 BLAS 进行高效矩阵乘与分解；实现功率迭代与过采样；在矩阵自由模式下使用矩阵向量乘子。

**📊 数据集**

Hilbert 矩阵、指数衰减谱矩阵、Gaussian 混合模型矩阵、NOAA ERSST v5 海表温度数据、图像压缩示例（4016×6016 RGB图）。

**📈 对比分析**

与 PyTorch 的 rSVD（固定秩）和 SciPy 的 rID（容差/秩模式）比较；在相同硬件下执行时间相当，且在 SciPy 上的速度提升可达 10‑30 倍；误差与现有实现相当或更优。

**⚠️ 局限性**

仅适用于中小规模（≤10 000）矩阵；不支持更大稀疏或分布式内存；在极低秩或高度退化奇异值分布时过采样/功率迭代仍需手动调参；容差模式的停止准则为启发式，未给出严格误差上界。

---

## 155. GPE: Evaluating Robust Evidence Aggregation for Fact Verification under Controllable GEO-Style Poisoning

**arXiv ID:** 2607.20730 | [PDF](https://arxiv.org/pdf/2607.20730v1)

**作者:** Zhaoqi Wang `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并发布了 GPE（GEO Poisoning Evaluation）基准与评估框架，用以在可控的 GEO‑式毒化环境下测试事实验证模型。

**💡 创新点**

创新点在于：①提出可调节毒化比例的多域事实验证数据集；②配备统一的证据环境和知识图谱；③设计了可控制恶意证据注入与检索接口；④系统性评估多种验证方法的鲁棒性和效率。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑V4‑Flash、GPT‑5.4）作为检索增强和直接推理后端，并对四种毒化手段（FakeGPT、PoisonedRAG、ATA、Ignore Injection）进行注入；评估指标包含准确率、Token 成本、Macro‑F1 等。

**📊 数据集**

数据集包含 638 条声明，涵盖政治、娱乐、科学、医学、历史和常识六大领域，提供人类标注标签、原始证据、恶意证据与关联知识图谱。

**📈 对比分析**

通过在同一声明与证据集合下对 Direct、RAFTS、SAFE、STEEL 四种方法进行对比，实验显示在不同毒化比例和攻击类型下性能差异显著，ATA 最具破坏性，且没有方法在所有设置中保持高鲁棒性；同时评估了 token 效率，揭示准确率下降导致成本显著提升。

**⚠️ 局限性**

局限性包括：①仅评估了四种验证策略，未覆盖所有现有技术；②毒化示例与比例为离线构造，未考虑动态检索过程的自适应攻击；③对知识图谱的利用未在实验中充分体现，未来工作需进一步挖掘图结构的鲁棒性潜力。

---

## 156. Operational Identity: A Finite Audit of Declared and Implemented Rules of Sameness

**arXiv ID:** 2607.20729 | [PDF](https://arxiv.org/pdf/2607.20729v1)

**作者:** Denise M. Case `[一作]` `[通讯]` (Northwest Missouri State University), Denise M. Case (Northwest Missouri State University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种可计算的“操作性身份分区”，并将其与系统声明的“声明性身份分区”进行比较，检查两者的一致性；

**💡 创新点**

创新点在于：①将实现机制的身份相关结果形式化为可观测的“表面”，②引入“忠实性”与“偏差识别”，③在子类基础上给出四分类的偏差诊断，④证明通过有限搜索即可决定偏差与非单调性；

**🔧 技术方法**

技术包括：基于分组的等价关系构造（并查集），对表面和使用函数的签名哈希，O(n²)枚举以寻找偏差对，且实现了线性近似算法；

**📊 数据集**

论文未使用外部数据集，而是以形式化的记录域、变换历史和公开机制集合为输入；

**📈 对比分析**

比较方法是将声明性与操作性分区映射到分区格中做精化检验；在小规模实例上与随机实例实验均表明算法在O(n²)时间内完成，线性近似可将开销降低至O(n+m⋅k)；

**⚠️ 局限性**

局限性包括：仅针对已公开的机制和使用；对未覆盖的联合机制、隐藏表面、使用缺失无法检测；需要完整的变换历史；偏差判定不涉及实际的规范合理性，且通过历史扩展可能导致通过判定失效。

---

## 157. Edit-Neighboring Data Streams and Privacy under Continual Observation

**arXiv ID:** 2607.20727 | [PDF](https://arxiv.org/pdf/2607.20727v1)

**作者:** Joel Daniel Andersson `[一作]` (Institute of Science and Technology Austria), Roodabeh Safavi `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种新的隐私邻接关系——编辑邻接（edit‑neighboring）流，用以捕捉个体参与时可能导致的时间偏移，并在此框架下研究持续观测（continual observation）下的差分隐私计数问题。

**💡 创新点**

创新点包括：①引入编辑邻接流并证明其比传统交换邻接（swap‑neighboring）更能反映真实系统中参与者导致的时间移位；②给出编辑邻接下数据独立加噪声机制的下界，证明任何此类机制的误差必须为多项式；③构造了第一套在编辑邻接下实现对数级误差的计数机制，并进一步扩展到稀疏流；④通过下界与上界的比较，确定了编辑邻接是“甜点区间”，既比更宽松的前缀和邻接更一般，又不需要多项式误差。

**🔧 技术方法**

核心技术包括：随机桶化（randomized bucketing）结合离散拉普拉斯分布、基于稀疏向量技术（SVT）的分区机制、偏置持续计数器（biased continual counter）、以及针对编辑邻接的复杂耦合映射（f_{σ→σ′}) 用以证明隐私。

**📊 数据集**

实验使用合成 Bernoulli 流（长度 T=10^4，概率交替为 0.1/0.9 的 10 规模区块）以及基准恒定 Bernoulli 流（概率 0.01）进行比较。

**📈 对比分析**

与先前针对交换邻接的持续计数器（如因子化机制）相比，本文提出的机制在保持相同成功概率下的攻击优势时，所产生的均方误差（RMSE）显著更低；实验结果显示，攻击优势≤0.1 时误差比旧方法高 3–9 倍；在更大 T 时误差差距随 T 指数增长，验证了理论上对数误差与多项式误差的分化。

**⚠️ 局限性**

限制主要体现在：①下界仅适用于数据独立加噪声机制，无法涵盖自适应或数据相关噪声方法；②编辑邻接假设忽略了多重插入导致的重排情况，只考虑单次插入；③实验仅基于合成数据，缺乏对真实系统（如联邦学习或网络统计）中的实际表现评估；④在稀疏流情况下，常数项仍相对较大，影响实际可用性。

---

## 158. Evaluating Large Language Models for Symbolic Security Protocol Analysis

**arXiv ID:** 2607.20712 | [PDF](https://arxiv.org/pdf/2607.20712v1)

**作者:** Paolo Modesti `[一作]` (Teesside University), Derek Enodolomwanyi `[通讯]` (Teesside University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了 OpenAI GPT 与 DeepSeek 两大 LLM 在安全协议分析中的表现，探讨其 chat 与 reasoning 模式与传统形式化验证工具 ProVerif 与 OFMC 的差异，并评估多跑一致性与置信校准；

**💡 创新点**

创新点在于：① 系统化评估 LLM 直接生成安全 verdict 与正式工具输出的匹配度；② 引入类型意识的协议识别混淆以避免训练数据泄露；③ 对多种模型、模式、多跑、置信度等维度进行细粒度测评；

**🔧 技术方法**

采用了 LLM 零-shot 结构化提示、chat 与 reasoning 模式的 API 调用、Python 自动化流水线、cluster bootstrap 统计、置信度校准分析以及 ProVerif 与 OFMC 两大形式化验证工具；

**📊 数据集**

使用了 130 个 / 规范协议，涵盖 388 个安全目标，来自 2025.09 版本的协议库，并对协议进行类型识别混淆处理；

**📈 对比分析**

通过将 LLM verdict 与 ProVerif（主基准）及 OFMC 1/2 会话结果对比，计算准确率、精确率、召回率与 F1。chat 模式召回高但精确率低（≈31%），reasoning 模式精确率提升（GPT≈66.5%，DeepSeek≈45%）但召回下降至≈55%；对不同安全目标分类表现差异显著；置信度无有效校准；多跑一致性差异存在；

**⚠️ 局限性**

局限性包括：LLM 结果不具备形式化证明保障，难以捕捉多会话攻击；模型对不同协议表现差异大；缺少人工专家复核；仅评估两大 LLM 提供商，未涵盖其他模型；提示与输出结构可能限制模型表达；训练数据污染风险仍需关注。

---

## 159. U-CFR: Uncertainty-Guided Cascade Forward Refinement for Interactive Segmentation

**arXiv ID:** 2607.20705 | [PDF](https://arxiv.org/pdf/2607.20705v1)

**作者:** Elijah Danquah Darko `[一作]` (University of Idaho), Matthew William Anderson `[通讯]` (Idaho National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种交互式图像分割框架U‑CFR，能够在每次用户交互后自动生成伪点击进行自我校正，同时使用双头网络（语义分割头+边缘检测头）提升边界精度。

**💡 创新点**

创新点在于：①利用边缘检测头作为辅助监督，强化共享ViT骨干对高频边界信息的学习；②设计基于预测不确定性和边缘梯度融合的边界不确定性地图，用于引导伪点击的生成，形成主动、目标导向的精细化循环；③在推理时实现无需额外训练模块的Cascade‑Forward Refinement（U‑CFR）迭代，显著降低所需点击数。

**🔧 技术方法**

技术包括：ViT‑Base（MAE预训练）骨干 + FPN多尺度特征；双分支解码器（语义+边缘）；Dice损失与归一化焦点损失；伪点击生成阈值策略；边界不确定性计算（U_pred × G_seg）；迭代推理回路。

**📊 数据集**

训练集：SBD；测试集：SBD、Pascal VOC、COCO_MVal、GrabCut、Berkeley、DAVIS；进一步在无微调的场景下评估医学（BraTS、ssTEM、OAIZIB）与材料科学（SEM‑Precipitate）数据集。

**📈 对比分析**

与现有SOTA方法（SimpleClick、PseudoClick、FocalClick、FocusCut等）对比，U‑CFR在多数基准上取得最低点击数（NoC@90、NoC@95）和最高mIoU、NSDS；在医学/材料任务中，无微调即显著提升mIoU@1与NSDS，显示强鲁棒性。

**⚠️ 局限性**

局限性：①伪点击的生成依赖不确定性与边缘梯度的准确度，若预测误差大可能误导；②在极细微或极稀疏目标上，边缘检测仍可能失效；③目前未探索多尺度伪点击或不同类型提示（scribble、框）的融合，未来可进一步提升。

---

## 160. A Framework for Reputation Aware Uninorm-driven Consensus Algorithms for Blockchain Networks

**arXiv ID:** 2607.20700 | [PDF](https://arxiv.org/pdf/2607.20700v1)

**作者:** Bruno Ramos-Cruz `[一作]` (University of Jaen), Luis Martínez `[通讯]` (University of Jaen)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于直觉主义模糊集合和无穷范聚合运算的声誉管理方法，用于提升区块链共识算法的公平性与可恢复性。

**💡 创新点**

创新点包括：①将直觉主义模糊集合引入声誉度量，能够同时表达成员度、非成员度及不确定性；②利用无穷范聚合运算实现声誉权重的动态更新，实现正负声誉的强化和恢复机制；③在共识算法中嵌入该机制后，保持线性计算复杂度且不增加额外通信开销。

**🔧 技术方法**

技术手段主要是：直觉主义模糊集合（IFS）构造成员函数与非成员函数；无穷范聚合运算（UAO）与Fodor无穷范；以及在区块链共识框架中实现成功验证率、声誉度与声誉权重的计算。

**📊 数据集**

实验使用模拟区块链网络，生成不同验证器的成功验证率序列（如表格中所示），未使用公开真实区块链数据集，而是基于自定义的仿真数据。

**📈 对比分析**

与传统单一阈值声誉模型比较，通过实验评估α参数对声誉度和权重的影响，并展示了验证器在误差后能快速恢复声誉的曲线。实验结果表明，方法在保持线性时间复杂度的同时，实现了更好的公平性与鲁棒性。

**⚠️ 局限性**

限制包括：①缺乏在真实大规模区块链环境中的大规模实验验证；②对无穷范参数的手工设定，缺乏自适应调优；③在极端攻击或网络分区情况下，声誉恢复机制的安全性尚未充分评估。

---

## 161. Buzz to Boom: Detecting Message Progression Vulnerabilities in Electron Applications via Segmented Directed Fuzzing

**arXiv ID:** 2607.20698 | [PDF](https://arxiv.org/pdf/2607.20698v1)

**作者:** Jianjia Yu `[一作]` (Johns Hopkins University), Yinzhi Cao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种分段定向模糊框架，用于检测 Electron 应用中的跨进程消息进展漏洞（MPV），并验证零日漏洞链；

**💡 创新点**

创新点在于将端到端模糊拆分为跨 IPC 边界的独立段，利用 LLM 自动生成针对每个进程的 harness 与种子，实现多步攻击链的高效发现与自动合成；

**🔧 技术方法**

技术栈包括代理式静态分析（使用 Claude Opus 4.7 与 Claude Code）、分段定向模糊（基于 jazzer.js / libFuzzer）、IPC 监听或acles、语义驱动的种子/ harness 生成、LLM 驱动的输入重构与端到端验证；

**📊 数据集**

数据集来源于 GitHub，筛选包含 Electron 依赖且 star>100、含 custom URI 或 IPC 处理代码的开源项目，覆盖数十个真实应用（如 Paperlib、Visual Studio Code、Discord 等）；

**📈 对比分析**

与单体端到端模糊（jazzer.js）对比，分段模糊发现 17 个零日漏洞（单体仅 6），覆盖率提升约 98%，平均发现时间从 30m22s 降至 17m9s，在多段漏洞场景中表现显著优于基线；

**⚠️ 局限性**

局限性包括：代理式静态分析可能漏报跨非标准库、模板引擎或 LLM SDK 的深层数据流；LLM 推理受训练数据与上下文窗口限制；框架专注 Electron，迁移至其他多进程平台需重新定义边界与 instrumentation；

---

## 162. Explanation-Based Runtime Verification for Trustworthy ML-driven Optical Networks

**arXiv ID:** 2607.20675 | [PDF](https://arxiv.org/pdf/2607.20675v1)

**作者:** Omran Ayoub `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Paolo Monti `[通讯]` (Chalmers University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于模型解释的运行时验证框架，用于光网络中ML驱动的决策（如光链路质量分类）

**💡 创新点**

创新点在于将SHAP解释与物理一致性与逻辑连贯性检查结合，在推理时筛选不可信决策，而非仅依赖置信度或熵；该方法能在保持高自动化率的同时显著降低误报

**🔧 技术方法**

采用的技术包括：XGBoost二分类模型、SHAP特征归因、熵筛选、物理一致性检查（核心物理特征如路径长度、跨度数、调制阶数）、运行时验证器（决定是否接受/推迟决策）

**📊 数据集**

使用了两个公开的光链路QoT估计数据集，分别包含约30k条样本，并采用5折重复hold‑out进行实验

**📈 对比分析**

与基线XGBoost和仅基于熵的选择性拒绝方法比较，实验显示：XAI验证在保持几乎不变的TPR、较低的FNR的同时，比熵法在相同检查率下能减少近90%的假正率，并且对自动化率的损失更小

**⚠️ 局限性**

局限性包括：仅针对二分类任务，依赖预定义的物理特征集合，SHAP计算在大规模实时场景下可能成本较高；对多类别、不同网络场景的通用性尚未验证；在某些情况下会略微提高FNR

---

## 163. From Agent Failures to Text Policies: What Works and What Breaks

**arXiv ID:** 2607.20668 | [PDF](https://arxiv.org/pdf/2607.20668v1)

**作者:** Jaideep Ray `[一作]`, Ankit Goyal `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在冻结权重的语言代理上使用TextGrad进行文本策略优化，探索如何通过自然语言反馈改进代理行为。

**💡 创新点**

创新点在于把代理级的文本策略学习拆解为执行、证据、更新、选择四个子任务，并系统评估这些环节的瓶颈。

**🔧 技术方法**

采用TextGrad框架及其诊断与更新机制，并结合官方GEPA搜索、步骤对齐轨迹、同前缀分支对照以及对比实验验证更新方法。

**📊 数据集**

使用Qwen2.5-7B-Instruct和Mistral-7B-Instruct-v0.3两大7B模型，在TextWorld、TextWorldExpress和TextArena三种文本任务环境中收集6,160条测试记录。

**📈 对比分析**

与固定提示和人类编写规则对比，人工规则可提升约5个百分点成功率（从15.6%升至20.6%），但基于轨迹学习（无论是摘要、轨迹、分支还是GEPA搜索）都未能显著超过固定提示，提升幅度不超过1-2个百分点。

**⚠️ 局限性**

局限性包括仅覆盖两种模型和三种合成环境，规则生成与选择依赖有限的验证集，且人类规则与任务族归属信息为特权，未检验与权重训练方法或跨领域任务的迁移性。

---

## 164. Rushes: A Human Preference Dataset for Pluralistic Alignment

**arXiv ID:** 2607.20767 | [PDF](https://arxiv.org/pdf/2607.20767v1)

**作者:** Michael Xu `[一作]` (Microsoft Research), Bill Dolan `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了 Rushes 数据集和基准，收集并分析了 8,167 名玩家在 AI 生成的多模态分支叙事游戏中做出的 44,226 次选择，旨在研究个性化参与偏好与序列决策；

**💡 创新点**

创新点在于提供大规模、长期用户轨迹的揭示偏好数据，并揭示“Engagement Gap”，即当前前沿 LLM（如 GPT‑5）在预测用户选择时仍无法超越简单流行度基线，强调多样化对齐的必要性；

**🔧 技术方法**

技术手段包括使用 GPT‑4o/GPT‑5 生成文本、图像、视频、音频；通过语义相似度和词汇多样性过滤保证选项多样性；利用 Azure 内容安全审查；在基线层面应用 SVD 矩阵分解、SASRec、语义分类器及 GPT 模型进行对比评估；

**📊 数据集**

使用的数据集为 Rushes 本身，包含 44,226 次决策事件、8,167 位用户、6 个多模态游戏（文本、图片、视频、音频），并提供完整的候选选项与用户历史记录；

**📈 对比分析**

评估采用事件级 top‑1 准确率，使用用户分层时间拆分；SVD 基线取得最高 37.7% 准确率，流行度基线为 36.4%；GPT‑5+历史约 34.2%，GPT‑5/4o 零样本分别 30.9%/30.3%；显示 LLM 与基线存在显著差距，验证 Engagement Gap；

**⚠️ 局限性**

局限性包括仅在英文 Xbox Insider 人群中收集，缺乏跨语言与跨文化验证；数据主要来自 GPT‑4o 生成，模型对其他 LLM 的适用性未知；多模态内容仅在实验中使用，未充分评估其对偏好预测的贡献；系统仍可能被用于不当的诱导或沉迷设计，需谨慎使用；

---

## 165. GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries

**arXiv ID:** 2607.20757 | [PDF](https://arxiv.org/pdf/2607.20757v1)

**作者:** Miguel P. Bento `[一作]` (Independent Researcher), João Seabra `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GaugeQuant，一种在训练期间利用Transformer连续对称性学习量化最优基的框架。

**💡 创新点**

创新点是通过可微LogSumExp惩罚项显式打破对称性，自动选择抑制激活异常值的基，且无需校准数据或量化仿真。

**🔧 技术方法**

使用了旋转矩阵、Cayley变换、block‑diagonal SO(b) 参数化、stop‑gradient、LogSumExp 近似 L∞ 等技术，并保持正交性。

**📊 数据集**

在C4文本上微调8192步，评估采用WikiText‑2数据集。

**📈 对比分析**

与基线预训练模型、仅微调以及加入GaugeQuant进行比较；在LLaMA‑2 7B的W4A4g128量化下，从8.22降到6.73，W4A16从11.16降到5.45，性能显著优于后处理方法。

**⚠️ 局限性**

局限性包括在线MLP旋转导致推理略慢、block‑diagonal 结构限制激活重分布、LogSumExp 过度抑制大激活可能削弱表达能力，以及仅在两款模型上验证，未测试更大模型。

---

## 166. Leaky Language Models: Stealing Architecture and Inference Optimizations via Per-Token Timing

**arXiv ID:** 2607.20723 | [PDF](https://arxiv.org/pdf/2607.20723v1)

**作者:** Sadegh Majidi `[一作]` (Purdue University), Kazem Taram `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用生产LLM的逐token生成时序信息，提出两种攻击：检测推理优化（如speculative decoding）并恢复模型架构参数。

**💡 创新点**

首次展示仅凭token时序即可泄露推理优化细节和模型内部结构，并构建基于理论与实测相结合的时间模型实现高准确率的架构推断。

**🔧 技术方法**

基于Transformer运算图的理论复杂度分析、线性回归时间预测器、cuBLAS核选择纠正、FlashAttention与KV缓存建模，以及二分搜索和多维网格搜索。

**📊 数据集**

使用开源LLM（Llama 3.2、Qwen、Phi3.5、Gemma2等）收集多配置的timing数据；远程API（Google Gemini、OpenAI GPT、Cohere、Mistral、Weights & Biases Llama 3.1 8B）作为黑盒测试集。

**📈 对比分析**

与官方模型参数对比，时间预测NRMSE仅0.12（eager）/0.188（FlashAttention+KV）；架构泄露top‑5准确率>70%单参数，双参数约40%；在远程API上能定位隐藏维度，层数排名4/28。

**⚠️ 局限性**

局限在于需精确token时序且受网络噪声影响，假设单GPU、decoder‑only、已知GPU型号；对多GPU、多模态或大规模部署的推理场景适用性有限。

---

## 167. Perspective Latents as an Architectural Condition for Causal Emergence in Active Inference Agents

**arXiv ID:** 2607.20708 | [PDF](https://arxiv.org/pdf/2607.20708v1)

**作者:** Hongju Pae `[一作]` `[通讯]` (Active Inference Institute), Hongju Pae (Active Inference Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在无奖励的主动推断代理中，视角潜在变量 g 所体现的因果出现（Φ_r）位置与学习过程的关系。

**💡 创新点**

创新点在于将 Φ_r 定位到结构化的视角潜变量 g 上，并通过原子分解揭示学习如何重组 Φ_r 的组成，而非仅仅增大其规模。

**🔧 技术方法**

使用 ΦID（Integrated Information Decomposition）计算 Φ_r，并对其进行原子分解；采用 GRU 结构的慢速潜变量 g 与快感知潜变量 z 进行对比分析。

**📊 数据集**

使用一个 15×9 的 2D 网格世界，包含三个不同观测噪声区块作为实验环境，训练 30 个随机种子、每个种子 10 次回放。

**📈 对比分析**

通过与未训练模型、时间打乱、以及前后环境转换的对照进行比较；结果显示 g 的 Φ_r 在训练后由负转正，整体规模下降但在不同原子组间分布发生重组，且在训练后 g 的 decoupling 在环境变化中保持稳定。

**⚠️ 局限性**

局限在于仅使用极简环境，难以区分视角特定组织与通用递归学习的影响；此外 Φ_r 的计算基于高斯假设，未考虑更复杂的动力学。

---

## 168. Enhancing Attack Detection Capabilities in BACnet/IP Networks Using Machine-Learning Models

**arXiv ID:** 2607.20686 | [PDF](https://arxiv.org/pdf/2607.20686v1)

**作者:** Derek Manzella `[一作]` (Dakota State University), John D. Hastings `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了统一的Zeek BACnet日志解析器，搭建了带标签的BACnet/IP模拟实验平台，并评估了五种无监督异常检测模型对六类攻击的检测效果。

**💡 创新点**

将原四日志合并为单文件简化ML流程；提供首个带标签的BACnet/IP攻击数据集；系统性比较多种无监督模型在高频与低频攻击下的表现。

**🔧 技术方法**

Zeek改写、bacpypes3 Docker模拟、Python scikit-learn实现的LOF、EllipticEnvelope、IsolationForest、One-Class SVM、SGD One-Class SVM。

**📊 数据集**

自建模拟实验平台生成的基线+六类攻击的PCAP，并按包标注，未使用公开的未标注或非网络数据集。

**📈 对比分析**

以F1、精确率、召回率、训练/测试时间为指标，One-Class SVM平均F1 0.864，极高流量攻击检测≈0.99，隐蔽攻击检测约0.77；其他模型表现明显逊色。

**⚠️ 局限性**

实验基于模拟网络，缺乏真实生产环境噪声与多样性；仅覆盖BACnet/IP，未评估MS/TP或BACnet/SC；对低流量隐蔽攻击的检测仍不理想。

---

## 169. Cross-Domain Generalization in Optical Networks via Joint Contrastive and Classification Learning

**arXiv ID:** 2607.20666 | [PDF](https://arxiv.org/pdf/2607.20666v1)

**作者:** Ali Al Housseini `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Omran Ayoub `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了联合对比学习与分类学习的框架，用于光网络QoT估计的跨域泛化。

**💡 创新点**

创新点在于将对比学习与分类目标联合优化，使表示空间既保持类内紧凑又对下游决策有益，从而获得域不变的任务相关特征。

**🔧 技术方法**

使用深度度量学习（对比损失+多相似度损失）、投影网络、在线三元组挖掘及联合训练策略。

**📊 数据集**

实验采用三个不同拓扑与配置的光网络数据集D1、D2、D3（共14-37节点、不同波长与速率），每个都有不同比例的失败样本。

**📈 对比分析**

与随机森林、ExtraTrees、XGBoost、CatBoost、单纯的MLP以及“分离”训练等基线对比，实验表明在最差域转移场景下MF1与PR-AUC提升约20%，在零样本和1-2%样本注入下仍保持高性能。

**⚠️ 局限性**

局限性包括对二分类任务的专注、对极大域差异的鲁棒性尚未完全验证、模型训练与推理的计算开销相对较大，以及在极小样本注入时对超参数的敏感性。

---

## 170. Scalable Low-Cost Laboratory Automation: A Digital Twin-Integrated Robotic Platform for Autonomous Liquid Handling (RAINBOTTM)

**arXiv ID:** 2607.20662 | [PDF](https://arxiv.org/pdf/2607.20662v1)

**作者:** Mohamed Rami Ayeche `[一作]`, Fadwa El Mellouhi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将消费级 3D 打印机改装为低成本液体处理机器人 RAINBOTTM，配备单通道移液器、线性执行器、颜色传感器，并实现浏览器端数字孪生和与 CEIDTM 逆向设计框架的闭环实验流程；通过色彩混合实验验证系统的精确移液、实时监测与自主优化。

**💡 创新点**

创新点在于：① 采用 3D 打印机机械平台实现可编程液体处理，成本大幅下降；② 将数字孪生与硬件同步，支持远程实时监控与人机干预；③ 将逆向设计优化算法与硬件结合，实现闭环目标驱动实验，同时保留人工监控。

**🔧 技术方法**

主要技术包括：Python + G‑code 控制 Klipper 固件实现台架运动；电磁线性执行器与继电器驱动移液器 plunger 与 tip‑eject；GY‑33 RGB 传感器实时颜色测量；Unity WebGL 生成的浏览器数字孪生通过 WebSocket 与物理平台双向同步；CEIDTM（Cooperative Explorer for Inverse Design）实现目标颜色的离散自由度优化。

**📊 数据集**

使用的“数据集”是实验室自行生成的颜色混合数据，包含不同体积红、黄、蓝、纯水的混合结果；未使用公开化学/材料数据集。

**📈 对比分析**

比较方法：在色彩混合任务中，将 RAINBOTTM 的颜色测量误差与预期值比较，平均绝对误差为 2%；在逆向设计优化中，将 CEIDTM 的 Frechet 距离与网格搜索、随机搜索、语言模型引导搜索等方法对比，CEIDTM 在 24 次实验后得到的最优误差为 0.0145（相较于网格搜索 0.0172、随机搜索 0.0593）。成本方面，RAINBOTTM 约 1,260 USD，远低于市场上 5,000 USD 起的商业液体处理器。

**⚠️ 局限性**

局限性：① 逆向设计算法 CEIDTM 属于专有技术，需额外许可；② 仅实现单通道移液器，适用范围受限；③ 主要验证在颜色混合等可观测任务，对更复杂化学反应或多通道操作尚未测试；④ 依赖人机监督以防安全风险，完全自主运行仍受限制。

---

## 171. Axolotl3D: a Unified Framework for Faithful 3D Shape Completion

**arXiv ID:** 2607.20660 | [PDF](https://arxiv.org/pdf/2607.20660v1)

**作者:** Anita Hu `[一作]` (NVIDIA), Maria Shugrina `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种统一的多模态、遮挡感知3D生成框架，能够利用图像、可见性掩码、相机参数和稀疏点云实现单视图、稀疏多视图、遮挡补全以及几何编辑等任务。

**💡 创新点**

创新点包括：1）将多种观察信号（图像、掩码、相机、点云）统一在一次生成模型中；2）采用可视化掩码的交叉注意力与Plücker投影来实现遮挡意识和相机一致性；3）设计跨模态训练策略，利用大规模3D数据合成多种条件场景，提升跨模态推理能力。

**🔧 技术方法**

核心技术包含：Hunyuan3D 2.1 的形状VAE+Diffusion Transformer、DINOv2图像编码、VecSetX点云编码、Plücker embedding相机投影、遮挡掩码偏置交叉注意力以及大规模数据增强。

**📊 数据集**

训练使用来自TRELLIS-500K（约40万件物体）的3D网格，测试在Toys4K（4k合成物体）和OmniObject3D（6k扫描物体）上，并在实际图像中使用MapAnything+SAM2生成点云与相机参数。

**📈 对比分析**

与ShapeR、Hy3D-Omni、SAM3D、Hunyuan3D-Omni等基线相比，本文方法在单视图和稀疏多视图场景下F-score、vIoU、Chamfer Distance等指标均实现了最高或相近于最高水平，且对遮挡和预测深度误差表现出更好的鲁棒性。

**⚠️ 局限性**

局限性在于对极端遮挡或严重深度估计误差仍可能导致补全不完整；对噪声点云的鲁棒性有限；目前支持的相机位姿范围与纹理多样性仍有待扩展。

---

## 172. Adaptive Multi-Horizon Reinforcement Learning

**arXiv ID:** 2607.20656 | [PDF](https://arxiv.org/pdf/2607.20656v1)

**作者:** Manoosh Samiei `[一作]` (McGill University), Paul Masset `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自适应多时域价值估计框架，通过学习可调门控权重动态组合多个折扣因子对应的价值函数，实现灵活的规划时间尺度。

**💡 创新点**

创新点在于将折扣因子视为专家，使用状态相关的门控网络实时分配权重，从而在持续学习情境下自动选择最合适的时间尺度，而非固定或单一折扣；同时将混合后价值函数用于 γ=1 的 Expected SARSA 更新。

**🔧 技术方法**

技术包括 Tabular Expected SARSA(λ) 对多折扣专家训练、基于 softmax 的门控网络、混合价值函数的无折扣更新、以及在 MiniGrid 环境中的连续任务训练。

**📊 数据集**

使用 MiniGrid 三个任务（foraging、goal‑reaching、four‑rooms）以及其按顺序切换的连续任务序列进行实验。

**📈 对比分析**

将混合方法与 10 个固定折扣因子（γ∈{1,0.998,…,0.5}）基线比较，评估累计回报和奖励/步数。实验表明，在所有任务中混合方法能达到或接近最佳单折扣基线的性能，并在任务切换时快速适应。

**⚠️ 局限性**

局限性包括仅在表格规模环境验证，随机种子对结果有较大影响；未在函数逼近或更大状态空间上测试；门控权重学习的收敛性与可解释性尚待进一步研究。

---

## 173. PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics

**arXiv ID:** 2607.20653 | [PDF](https://arxiv.org/pdf/2607.20653v1)

**作者:** Haocheng Yin `[一作]` (Georgia Institute of Technology), Lu Gan `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PhysCoRe，一种结合可微分MPM仿真器与两块前馈网络的物理修正残差世界模型，用于预测机器人操作下变形物体的演化并实现在线材质识别。

**💡 创新点**

创新点在于：①用Material from Motion (MfM)在有限交互中一次性推断粒子级材质及置信度；②用Residual from Dynamics (RfD)在仿真内部动态中学习残差修正，弥补分析模型与现实的差距；③置信度可驱动主动探索。

**🔧 技术方法**

技术包括：可微分Material Point Method (MPM)，图U-Net、FiLM条件稀疏3D U-Net，傅里叶特征编码，GRU时间模块，Chamfer距离、L2误差等损失；以及基于Perlin噪声的材质仿真数据增强。

**📊 数据集**

使用12条真实人手操纵变形物体的RGB‑D序列（绳索、毛巾、毛绒玩具、Play‑Doh），并基于14条PhysTwin真实捕获进行合成数据增强，生成1,260个增强实验。

**📈 对比分析**

与PhysTwin、PGND、EMPM等基线对比，PhysCoRe在未来预测的Chamfer距离和跟踪误差上分别降低约30–43%，并在运行时将材质识别从数百秒降到约十秒，显示出更高精度和更快适应。

**⚠️ 局限性**

局限性在于仅覆盖弹性与弹塑性物体，对撕裂、粘附、流体等更复杂动力学缺乏处理；训练数据集中残差修正网络覆盖范围有限，尚未验证更广泛动态分布的可扩展性。

---

## 174. ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management

**arXiv ID:** 2607.20764 | [PDF](https://arxiv.org/pdf/2607.20764v1)

**作者:** Pavel Golikov `[一作]` (University of Toronto), Mark C. Jeffrey `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个名为 ArbiGraph 的基准生成器，用于构造可验证的、可执行的任务图，评估工具辅助语言代理在长链式推理过程中如何保留、更新、组合与丢弃任务相关的上下文状态。

**💡 创新点**

创新点在于：① 把上下文抽象为可执行的、有类型的中间状态；② 通过用户指定的依赖拓扑自动生成可验证的多任务提示；③ 设计四种拓扑（基线、遗忘、链式、分支链）系统性检验上下文管理能力。

**🔧 技术方法**

技术实现包括：Python 代码执行器、NetworkX 构建 DAG、Graphviz 可视化、预处理/后处理适配器、工具调用修复协议、符号验证和 Wilson 置信区间评估；模型使用 Qwen3.5‑27B 结合计算器工具。

**📊 数据集**

数据集由三类任务组成：数学算子（40 个）、GSM‑风格算术词题（41 个）和 LeetCode Python 跟踪任务（80 个），通过随机采样生成 16 条样本，覆盖 4 种拓扑。

**📈 对比分析**

比较方法：在每种拓扑下对目标任务进行零样本推理，记录最终任务准确率、平均 token 长度和回合数。结果显示：基线准确率高（数学 94.5%），但在链式拓扑上数学准确率降至 75.5%，多分支链进一步降至 61.2%，Python 跟踪任务较为稳健（约 90%）。

**⚠️ 局限性**

局限性包括：仅测试标量/列表的可执行任务；仅评估单一模型 Qwen3.5‑27B；未覆盖多模态任务；拓扑类型有限；未与多种模型/代理设计做对比。

---

## 175. A real-time RGB-D perception pipeline for autonomous impact hammers in mining: self-filtering, rock segmentation and rock-breaking poses generation

**arXiv ID:** 2607.20748 | [PDF](https://arxiv.org/pdf/2607.20748v1)

**作者:** Martín Gallegos `[一作]` (Universidad de Chile), Javier Ruiz-del-Solar `[通讯]` (Universidad de Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个实时RGB‑D感知流水线，能够在嵌入式硬件上以约10Hz生成岩石破碎目标姿态和无机器人障碍3D工作空间表示。

**💡 创新点**

结合实例分割与几何点云处理，并引入自滤与遮挡管理、表面法向约束的姿态生成与多准则优先级，支持自动化冲击锤操作。

**🔧 技术方法**

使用RTMDet实例分割、DBSCAN、ICP自标定、双向滤波、点云投影、SDF、表面法向分析、JAX/Open3D/PyTorch/ROS等技术。

**📊 数据集**

在实验环境中使用两台ZED X相机收集同步RGB‑D数据，构建钢栅模型，并通过人工标注的岩石位置进行评估；未公开专用数据集。

**📈 对比分析**

与基于质心的单姿态方法对比，实验显示系统在真实实验中实现约10Hz，姿态生成成功率约72%，并在10分钟的测试中维持低延迟（≈675ms）与合理的姿态数量。

**⚠️ 局限性**

对倾斜表面岩石缺少姿态；对岩石分割误差影响优先级；未考虑钢栅几何；以及对高复杂场景下的分割融合需要改进。

---

## 176. Attribution Markets: A Fisher-Market Formulation for Fractional Credit Assignment Between Planned Tasks and Performed Actions

**arXiv ID:** 2607.20694 | [PDF](https://arxiv.org/pdf/2607.20694v1)

**作者:** Salavat Ishbulatov `[一作]` `[通讯]` (Independent researcher), Salavat Ishbulatov (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将已完成动作与计划任务的归属问题建模为带卖方保留价和买方现金选项的准线性费雪市场，并给出了求解比例响应动态算法。

**💡 创新点**

创新点在于：① 用市场清算机制实现了保守性、预算上限和垃圾过滤的三项设计要求；② 通过饱和阈值固定点方法修正了完成目标扩展的收敛问题；③ 引入熵正则化统一市场与Sinkhorn最优运输的差异，并给出噪声自适应策略。

**🔧 技术方法**

采用了费雪市场理论、比例响应动态、对数-赔率融合的亲和度信号、凸优化、Brouwer固定点定理、熵正则化以及均值-方差风险感知估值。

**📊 数据集**

使用了自定义的去环化多种子合成数据集：7个任务、约170个动作、63天时段、15个随机种子，噪声水平分别为0、0.15、0.30。

**📈 对比分析**

与硬分配、softmax、Sinkhorn OT 四种归属规则进行对比。结果显示：Sinkhorn 在总变差误差上最优；市场在预算违规、稀疏性、争议拆分以及满足预算上限方面表现最佳；在噪声为0时市场的错误率仅略高于Sinkhorn，但在噪声增大时市场保持了绝对的预算安全。

**⚠️ 局限性**

局限性包括：对亲和度噪声敏感；收敛性仅在满足对角占优的充分条件下保证；未在真实用户数据上验证；熵正则化市场的高效算法尚未实现；均值-方差扩展仅实现了单任务子问题。

---

## 177. Learning to Detect UI Principle Violations via Reinforcement Learning

**arXiv ID:** 2607.20690 | [PDF](https://arxiv.org/pdf/2607.20690v1)

**作者:** Nishi Mehta `[一作]` (University of California Santa Cruz), Pratik Jayarao `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练了一款4B规模的视觉-语言模型，用作网页前端代码生成的UI/UX审计师，覆盖19条可检测的界面质量原则；

**💡 创新点**

创新点在于统一了WCAG、暗黑模式和认知设计三类原则，采用合成注入+教师验证的数据生成流程，利用强化学习在轻量模型上实现高准确度审计；

**🔧 技术方法**

使用Qwen3-VL-4B-Thinking模型与Qwen3-VL-235B-A22B教师模型，结合GRPO强化学习算法和多模态输入（截图+HTML）进行训练；

**📊 数据集**

数据集约1万份合成网页，先生成干净Tailwind页面，再注入1–3条可验证违规，教师模型确认后得到标签；

**📈 对比分析**

与零射击基线相比，RL后微平均F1从36%提升至84%，13条原则达到80%以上；在所有原则中，视觉与暗黑模式检测提升最大，性能已接近实际审核需求；

**⚠️ 局限性**

局限包括仅单次实验、教师模型产生的标签缺乏人工校验、分辨率（448×544）限制了对相对判断原则（如misdirection、Fitts’s Law）的识别、以及合成注入样式可能不完全覆盖真实生产环境中的违规情况。

---

## 178. Towards Capability-Aware Traversability Navigation for Unstructured Environments

**arXiv ID:** 2607.20679 | [PDF](https://arxiv.org/pdf/2607.20679v1)

**作者:** Gianluca Capezzuto `[一作]` (University of São Paulo), Marcelo Becker `[通讯]` (University of São Paulo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 Capability-Aware Traversability (CAT) 框架，直接在视觉特征空间中嵌入机器人本体约束，实现多种机器人平台的可行性预测。

**💡 创新点**

创新点在于将机器人能力向量通过 SPADE 块对语义地图进行空间自适应归一化，使物理限制直接编码；并构建交互式标注流水线，将轨迹与零拮信号结合生成稠密监督。

**🔧 技术方法**

技术包括 DINOv3 视觉编码、CLIPSeg 语义分割、GroundingDINO + SAM2 的交互式标注、SPADE 解码器、对比学习的机器人特定原型以及 InfoNCE 损失。

**📊 数据集**

使用 WayFASTER、GrandTour、TartanDrive、NaviTrace 数据集，以及在 Spot 与 TerraSentia 等真实平台上的部署。

**📈 对比分析**

与 WayFAST、W‑RIZZ 等基线对比，在人类标注和物理轨迹两种评估分布下，CAT 的 AUROC 最高，分别提升 11% 及 15.8%，平均可行性分数和 AUPRC 亦显著优于基线。

**⚠️ 局限性**

局限在于对动态障碍的识别仍依赖语义地图，未覆盖所有机器人本体，且多原型方案未显著提升能力区分；语义分组错误及未被训练的地形类别仍影响精度。

---

## 179. Safe and Scalable Multi-Drone Payload Transport via CBF-based Reinforcement Learning with Zero-Shot Sim-to-Real Transfer

**arXiv ID:** 2607.20665 | [PDF](https://arxiv.org/pdf/2607.20665v1)

**作者:** Jaeyoun Choi `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于控制屏障函数(CBF)的分布式强化学习框架，实现多无人机协作运输悬挂载荷并能在动态环境中保持安全。

**💡 创新点**

创新点在于：①使用极简二维抽象保持关键信号耦合、降低仿真成本；②在分布式政策中直接学习DGCBF实现安全约束；③通过域随机化和零样本sim-to-real转移，确保策略在未见过的团队规模和动态障碍物下仍可安全工作。

**🔧 技术方法**

核心技术包括：控制屏障函数（CBF）与离散图CBF的联合学习（DGPPO），域随机化（随机团队大小、弹簧刚度等），离散-连续安全桥接（跟踪误差容限），以及低层PID跟踪控制。

**📊 数据集**

数据集：在仿真中使用JAX物理引擎生成数千个随机场景（包括障碍物、团队大小3–5、弹簧刚度[0.05,0.30]等），训练10^5步；实地测试在Crazyflie 2.1平台上完成10个硬/易场景与多组实验。

**📈 对比分析**

与InforMARL、MAPPO-Lagrangian及惩罚式策略比较，DGPPO在安全率和奖励平衡上表现最佳；在团队规模扩大至6、8时仍保持高安全率（≥84%）且完成率稳定；在多组动态障碍测试中实现了无冲突安全协作。

**⚠️ 局限性**

局限性：仅适用于二维平面载荷运输，需均匀分布的悬挂点；对三维运动、非均匀悬挂或更复杂载荷几何尚未验证；依赖PID跟踪误差已知，若硬件误差增大可能影响安全保证。

---

## 180. Component structure and percolation in block models

**arXiv ID:** 2607.20719 | [PDF](https://arxiv.org/pdf/2607.20719v1)

**作者:** Riccardo Franchi `[一作]` (University of Michigan), M. E. J. Newman `[通讯]` (University of Michigan)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了随机块模型（Stochastic Block Model）及其度数校正变体的连通分量结构与渗流（percolation）性质，推导了巨型分量大小、小分量分布以及节点/边渗流阈值的解析表达式，并将微观（microcanonical）模型的结果映射到宏观（canonical）模型，进一步扩展到非均匀渗流。

**💡 创新点**

创新点在于：
1) 通过生成函数方法得到多组块模型的巨型分量和小分量分布的闭式迭代方程；
2) 设计了微观与宏观模型之间的映射（利用拉普拉斯变换），使得度数校正的宏观模型也可用同一套解析工具处理；
3) 将渗流分析推广到节点/边渗流以及基于度数的非均匀渗流；
4) 结合理论与大规模随机网络仿真，验证了公式的准确性。

**🔧 技术方法**

主要技术：
- 概率生成函数（Probability Generating Functions）与线性化/特征值分析；
- 拉普拉斯变换用于处理连续度数分布；
- 矩阵特征值判定巨型分量存在条件；
- 生成函数迭代与有限阶展开（或FFT）计算小分量分布；
- 模型映射与大数极限定理。

**📊 数据集**

论文未使用真实数据集，而是通过理论推导和大规模（数十万到百万节点）随机生成的块模型网络进行数值验证；使用的主要网络类型包括：
- 原始SBM（Poisson度数）；
- 微观度数校正SBM（精确度数分布）；
- 宏观度数校正SBM（连续度数分布）。

**📈 对比分析**

与仿真比较：对巨型分量大小、渗流阈值及小分量分布的数值结果与理论迭代公式完全吻合（误差 < 1e-5），显示模型预测在无限大网络极限下高度准确。若对比其他近似方法（如单块近似、随机图阈值），本文的多块精确解析在预测阈值与分量大小方面显著更精细，尤其在块间连接极低的 assortative 结构中能捕捉到“双跳”现象。

**⚠️ 局限性**

局限性：
- 解析结果基于无环（locally tree-like）稀疏网络的无限大极限，难以直接适用于高密度或有强环的实际网络；
- 对多块、大规模参数的闭式解难以手工求解，需数值迭代；
- 仅覆盖非重叠、单成员块结构，无法直接处理重叠或混合成员模型；
- 对动态过程（如时间演化、网络演化）和更复杂的传播机制（多级传播、复杂阈值）未给出完整解析。

---

## 181. Are Diversity Metrics Measuring Diversity? A Capability-Controlled Audit of Majority-Vote Gain in LLM Ensembles

**arXiv ID:** 2607.20768 | [PDF](https://arxiv.org/pdf/2607.20768v1)

**作者:** Donghwan Kim `[一作]` `[通讯]` (Aidentyx Inc.), Donghwan Kim (Aidentyx Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多模大语言模型集合中多数投票的优势，并审计五种多样性度量在控制模型能力后对投票增益的预测作用。

**💡 创新点**

发现除协同错误（co‑failure）之外，其他多样性指标与模型能力高度耦合，只有协同错误在控制后仍保持负向关联；此外提出了严格控制能力后剩余维度的代数非可分性与经验一致性分析。

**🔧 技术方法**

使用能力控制（线性与非线性）、部分Spearman相关、残差线性回归、模型层级重抽样、以及多重子集分析等统计技术，对31,900个大小为2-4的子集进行评估。

**📊 数据集**

主要使用MMLU‑Pro（30个模型，356个共解析题目）和TruthfulQA（29个模型，338个题目）两个公开基准。

**📈 对比分析**

对比“最佳成员”基准，发现多数投票在大小为3的子集中仅有约10%的子集能击败最佳成员，而在所有子集中的正增益率为1.27%；原始多样性与增益的相关性被能力所混杂，控制后仅残差协同错误保持负相关，幅度约为0.18‑0.57。

**⚠️ 局限性**

结果受共解析切片、模型池与基准的特定性影响，协同错误的效应虽稳健但幅度有限；多样性指标与能力的高度共线性导致其预测力不显著，且仅在特定配置下才可能出现误差。

---

## 182. Adaptive Confidence-weighted Expansion for Trustworthy Multi-Omics Multimodal Fusion

**arXiv ID:** 2607.20742 | [PDF](https://arxiv.org/pdf/2607.20742v1)

**作者:** Mohammad Raahemi `[一作]` (University of Ottawa), Hamid Nasiri `[通讯]` (Lancaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出ACE框架，通过相关性驱动的模态扩展和基于置信度的自适应加权，提升多组学多模态融合模型的可信度和鲁棒性。

**💡 创新点**

创新点包括：①利用 Pearson 相关性生成补充模态；②双层置信度机制实现自适应加权和全局置信估计；③输入重建自监督正则化增强鲁棒性；④将上述模块整合为统一的ACE体系。

**🔧 技术方法**

采用深度多模态神经网络、特征级门控、置信度校准（TCP）头、自监督输入重建、噪声注入与自适应权重损失等技术。

**📊 数据集**

在四个公开多组学数据集上验证：BRCA、KIPAN、LGG、ROSMAP。

**📈 对比分析**

与MOGONET、TMC、CF、GMU、MM‑Dynamics等现有方法对比，ACE在准确率、F1、AUC 以及置信度校准指标（MSE、ECE、AUC‑Error等）上均显著优于基线。

**⚠️ 局限性**

局限主要体现在：模型参数与计算量约翻倍；在极端缺失或极度不平衡的数据场景下仍可能受限；以及需对不同数据类型的超参数做进一步调优。

---

## 183. REGARD: Regional Affective Differences in Large Language Models

**arXiv ID:** 2607.20722 | [PDF](https://arxiv.org/pdf/2607.20722v1)

**作者:** Andrei Chetvergov `[一作]` (Ivannikov Institute for System Programming of the Russian Academy of Sciences), Sergey Bolovtsov `[通讯]` (Ivannikov Institute for System Programming of the Russian Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了不同区域训练的LLM在对后苏联实体的情感框架差异，利用VAD模型评估情感维度并对19个模型进行聚类，发现情感强度（arousal）是主要区分因素。

**💡 创新点**

创新点在于采用Valence–Arousal–Dominance（VAD）框架代替单一情感极性，揭示情感强度和主导度在模型表现中的重要性，并通过LLM裁判自动化评分与人类验证相结合。

**🔧 技术方法**

技术包括：VAD评分（两台LLM裁判）、Ward‑linkage层次聚类、Python数据处理、OpenAI/第三方API调用、可复现的裁判合同。

**📊 数据集**

数据集为CIS‑Affective‑500（500个后苏联实体），共19个生成模型、28,500条回复、57,000个模型-裁判对、900条人类VAD标注。

**📈 对比分析**

比较方法：对19模型按平均VAD和泛化回答率聚类，评估各群体在arousal、valence、dominance上的差异。表现显示：arousal在3个聚类间差异显著（0.34–0.58），valence波动小；与人类标注的相关性分别为0.845（valence）、0.565（arousal）、0.513（dominance）。

**⚠️ 局限性**

局限性包括：低arousal裁判一致性较差；只在俄语文本下测试，无法分离流利度与情感；未包含非俄语后苏联LLM；聚类数目人为设定；评估仅针对情感强度，未考虑信息质量。

---

## 184. Security Vulnerability Patterns in AI-Generated Code: A Cross-Model Comparative Study

**arXiv ID:** 2607.20713 | [PDF](https://arxiv.org/pdf/2607.20713v1)

**作者:** Shanna M. Kahn `[一作]` (Dakota State University), John D. Hastings `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在非专业用户场景下，LLM（ChatGPT、Copilot、Gemini）生成的自动化脚本是否存在可被利用的安全漏洞，并通过自动化审计工具进行评估。

**💡 创新点**

首次将三大主流LLM在相同提示下生成的非专业级自动化脚本进行跨模型漏洞一致性对比，并将发现的漏洞同时映射到CVSS、OWASP Top 10和MITRE ATT&CK三大安全框架，以多维度衡量风险。

**🔧 技术方法**

使用LLM模型生成脚本，Claude Code自动化审计工具进行漏洞检测，随后使用CVSS v3.1进行分值计算，并将结果映射到OWASP Top 10:2021和MITRE ATT&CK框架。

**📊 数据集**

构建9个Python自动化脚本（Web抓取、邮件发送、文件归档）—每个脚本分别由三种LLM生成，形成3×3=9份代码样本。

**📈 对比分析**

对去重后的17类漏洞按模型计数、重叠度和CVSS加权分数进行比较；发现不同模型间差异<10%，且大多数漏洞在所有模型中出现，说明风险与任务类型相关而非平台差异；整体漏洞严重性集中于前11类。

**⚠️ 局限性**

局限性：仅单次生成，未覆盖多模型多轮输出；任务仅限三类，未测试其他自动化场景；使用Claude Code时每文件最多5条发现，可能漏掉低严重度漏洞；未进行手工验证或动态利用测试；模型选择局限于ChatGPT、Copilot、Gemini。

---

## 185. FELT: Generating Tactile Signals from Vision for Visuo-Tactile Manipulation

**arXiv ID:** 2607.20683 | [PDF](https://arxiv.org/pdf/2607.20683v1)

**作者:** Zinan Li `[一作]` (University of Southern California), Daniel Seita `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于RGB视觉信息生成每个手指压力触觉图像或特征的框架，称为Feature-Extracted Latent Tactile（FELT），用于增强机器人在视觉遮挡或信息不足时的抓取与操作能力。

**💡 创新点**

创新点包括：① 采用双分支Transformer解码器，分别对应左右手指的物理传感器拓扑；② 通过跨面交互模块实现两侧手指间的协同推理；③ 只需单次前向传递即可合成触觉，支持离线增强、在线部署和无触觉特征训练。

**🔧 技术方法**

技术手段主要是：冻结的DINOv2视觉Transformer提取视觉token；学习型查询嵌入与二维位置编码；跨模态注意力与门控跨面交换；卷积读出层与残差深度可分离卷积预测接触概率与压力强度。

**📊 数据集**

使用的主要数据集为Zhu等人公开的两面触觉手指的RGB‑触觉配对数据（约2.6M帧），并在xArm平台上收集30个GELLO演示（约72K帧）作为测试集，另外为下游策略收集了四个任务各60条演示（共240条）。

**📈 对比分析**

在触觉生成任务上，FELT在LPIPS、能量比例和帧准确率等指标上均优于最近邻与MAE基线；在四个操作任务中，使用生成触觉或latent特征的策略在20次试验中相较于仅视觉的基线提升显著，且在低样本场景下latent特征可与真实触觉表现相当。

**⚠️ 局限性**

局限性包括：依赖大量高质量的RGB‑触觉配对数据；生成触觉依赖于视觉中可见的接触区域，严重遮挡会导致性能下降；目前仅针对压力型触觉传感器，无法直接迁移到磁性或凝胶型等其他触觉设备。

---

## 186. ODeform: Learning Continuous 4D Motion for Shape Deformation with Neural ODEs

**arXiv ID:** 2607.20670 | [PDF](https://arxiv.org/pdf/2607.20670v1)

**作者:** Yordanka Velikova `[一作]` (Technical University of Munich), Benjamin Busam `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 ODeform 框架，利用神经普通微分方程（neural ODE）将 3D 点云与物理参数映射到潜在空间，连续演化后生成任意时间点的对象变形。

**💡 创新点**

创新点在于将刚体运动与局部形变拆分为并行神经 ODE，保持 SE(3) 结构并实现高效稳定的连续演化，同时引入自适应权重损失与逆向参数优化，提升了物理一致性与泛化能力。

**🔧 技术方法**

采用的技术包括双编码器 MLP、并行神经 ODE、adjoint 敏感度训练、可变时间步长的 Runge‑Kutta 求解、自动加权损失以及基于梯度的逆向物理参数优化。

**📊 数据集**

实验使用扩展后的 Contact‑Force（Everyday Deform 数据集）与 Mass‑Elastic（下落实验）数据集，并在 HouseCAT6D 与 3D Gaussian Splatting 场景中验证跨域泛化。

**📈 对比分析**

与 nODE、RNN、PE‑GNN 等基线对比，ODeform 在未见物理参数、几何形状、插值与外推任务中显著降低 RMSE/MAE（如未见接触力 RMSE 6.657 mm、未见质量‑柔度 RMSE 1.279 mm），并在稀疏帧下保持低误差，表现优于所有对比方法。

**⚠️ 局限性**

限制主要体现在仅针对单物体弹性变形、缺乏多物体交互建模、对极端形状或极端物理参数的泛化尚有限，以及对大规模高质量物理模拟数据的依赖。

---

## 187. SalesLoop: Reinforcement Learning from Performance Feedback for Sales Lead Ranking

**arXiv ID:** 2607.20655 | [PDF](https://arxiv.org/pdf/2607.20655v1)

**作者:** Chenyu Zhang `[一作]` `[通讯]` (Li Auto Inc), Chenyu Zhang (Li Auto Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了SalesLoop——一种将强化学习与CRM线索排序结合的闭环框架，利用真实业务回报来持续更新排序模型；

**💡 创新点**

在离线-上线指标不匹配、点对点与列表化目标错位以及时间分布漂移三大瓶颈上引入：①基于排名位置和转化速度的性能感知奖励；②将Group Relative Policy Optimization（GRPO）改造为Discriminative GRPO的列表化优化；③构建闭环的在线迭代机制；

**🔧 技术方法**

采用强化学习与性能回报、GRPO/Discriminative GRPO列表化损失、LoRA微调的Qwen2.5-1.5B语言模型、KL‑divergence对齐、BCE正则化、月度在线迭代等技术；

**📊 数据集**

使用真实新能源车厂CRM数据：9.2M条历史样本（2025年7-9月）用于离线基线；16.5M条上线数据（2025年12月-2026年6月）用于160天A/B测试；以及103天生产日志；每条线索含47个结构化特征和约2682个对话文本；

**📈 对比分析**

对比XGBoost、DeepFM、LLM+SFT、LLM+DPO等基线，离线评测中SalesLoop在P@K、R@K、NDCG@K、AUC上分别提升约+15.8%、+7.9%等；在160天A/B测试中累计锁定转化提升+4.7%（p=0.047）和+8.7%（p=0.002），每位专员额外转化1.6–2.7个；基线模型Top‑10%召回率44.1%，高意向线索转化率提升2.3倍；

**⚠️ 局限性**

局限在于月度迭代频率受30天转化窗口限制，无法快速适应短周期变化；仅关注即时转化收益，未考虑客户生命周期价值；模型对非转换线索无奖励，可能忽略潜在高价值线索；需进一步验证跨域迁移、探索‑利用平衡和更细粒度奖励设计。

---

## 188. Scaling Interpretable Transformers with Parity Bottleneck Layers

**arXiv ID:** 2607.20652 | [PDF](https://arxiv.org/pdf/2607.20652v1)

**作者:** Andrew Mack `[一作]` (Principles of Intelligence), Lauren Greenspan `[通讯]` (Principles of Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一种可解释的 GPT‑2 规模模型 ParityTransformer，其核心是 Deep Parity Bottleneck（DPB），通过无参数的 parity hash 构造过完备稀疏字典，实现了激活稀疏化与可解释性；

**💡 创新点**

创新点在于：①用参数‑free parity hash 取代传统学习的稀疏字典，消除显存/带宽瓶颈；②采用多层 Mixture‑of‑Experts 结构让稀疏选择成本呈对数级，实现在大模型上可扩展的可解释瓶颈；③在设计与训练中加入 EMA 标准化、方差与重建辅助损失，以提升特征的稀疏性与判别力；

**🔧 技术方法**

使用技术包括：Deep Parity Bottleneck、层级 hash‑based sign vectors、hierarchical MoE（多级专家）、EMA 归一化、方差与重建辅助损失、RMSNorm、MuP 训练、GPU register‑level hash 计算等；

**📊 数据集**

数据集主要为 20B FineWeb‑Edu 进行预训练，评测使用 Pile、HellaSwag、LAMBADA 等公开数据集；毒化检索实验使用 10B/20B FineWeb‑Edu 的训练与混合检索语料；

**📈 对比分析**

与匹配性能的密集 GPT‑2 基线、Top‑K SAE、Matryoshka SAE 等进行对比；在语言建模指标（PPL、loss）上与密集基线相当或更优；在可解释性评估（Sparse Probe、AutoInterp、SCR、TPP、Absorption）以及因果干预、steering、毒化检索等任务上，ParityTransformer 通常优于或至少等同于后者，且在干预次数、概念注入/流畅度曲线等方面表现更好；但需 5–10× 更多训练 tokens，训练吞吐率约 1.2–2.1× 之上；

**⚠️ 局限性**

局限性包括：较高的解释性税（token‑budget 与 throughput 开销大）；在 AutoInterp 等指标上略逊于某些 SAE；仍需 EMA 调整以避免稀有特征死亡；目前仅在 MLP 层实现可解释瓶颈，未扩展至注意力层；在更大规模模型（3B+）上仍缺乏充分验证；

---

## 189. DS@GT ARC at ImageCLEFmed GANs 2026: Geometric Filtering for Privacy-Preserving CT Slice Generation

**arXiv ID:** 2607.20692 | [PDF](https://arxiv.org/pdf/2607.20692v1)

**作者:** Eric Regina `[一作]` (Georgia Institute of Technology), Samir Hadi Cisneros `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种两阶段的隐私保护合成肺部 CT 切片生成框架，先用 OT-CFM 生成候选图像，再通过 Supervisor 的几何过滤和子集选择实现隐私屏蔽；

**💡 创新点**

首次将 Optimal Transport 条件流匹配与几何后处理（自动编码器嵌入、DPP、Stein Kernel Thinning）结合，形成可调节隐私-实用性权衡的完整管线；

**🔧 技术方法**

使用 OT-CFM（UNet 34.5M）、自编码器（空间、对比、黎曼距离）、Determinantal Point Process、Stein Kernel Thinning、RBF-MMD、FID 等指标；

**📊 数据集**

基于 CLEF 2026 ImageCLEFmed Subtask 3 的 10,000 张 256×256 的无标签肺 CT 切片（去重后 9,931 张，训练 7,945 张）；

**📈 对比分析**

在官方评估中，100 轮 OT-CFM + Supervisor 的 FID 0.3290 与 PPS 0.549 位居榜首；经过空间 DPP 进一步过滤得到 FID 0.2639、PPS 0.492，显示在视觉真实性与隐私泄漏间取得较优折衷；

**⚠️ 局限性**

尽管能显著降低实例级记忆和成员推断，但所有模型在患者重识别攻击（攻击 3）上泄漏率仍高达 0.97–1.0，说明当前方法无法彻底抑制深层结构身份泄露。

---

## 190. Spatially Grounded Concept Bottleneck Models for Trustworthy Breast Ultrasound Diagnosis

**arXiv ID:** 2607.20691 | [PDF](https://arxiv.org/pdf/2607.20691v1)

**作者:** Moshiur Rahman Tonmoy `[一作]` (Texas A&M University--Kingsville), Afzel Noore `[通讯]` (Texas A&M University--Kingsville)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于弱空间监督的概念瓶颈模型（SG‑CBM），通过从乳腺超声图像的病灶分割中提取两类临床区（病灶 ROI 与后方声带）来约束概念特征图的空间定位，从而提升诊断可靠性和解释可解释性。

**💡 创新点**

创新点包括：①利用病灶边界的粗粒度掩模构造可解释的空间区域，避免了对像素级概念标注的需求；②设计分组的“分离”与“质量集中”损失，使概念激活集中在指定区域；③在诊断任务中引入“训练‑腐败 / 测试‑清洁”注释质量压力测试，系统评估监督质量对模型可信度的影响。

**🔧 技术方法**

技术包括 EfficientNet‑B2 作为特征提取器、1×1 卷积生成概念图、top‑25% 池化得到概念 logits、线性分类器做最终诊断、分离损失、质量集中损失以及基于掩模的分组空间约束；使用 BCE + focal loss 训练概念与诊断。

**📊 数据集**

在 BrEaST 数据集上进行实验，该数据集提供灰度乳腺超声图像、病灶掩模以及 16 个 BI‑RADS 描述符。

**📈 对比分析**

与标准 CNN（EfficientNet‑B2）和无空间约束的普通 CBM 进行对照。SG‑CBM 在 5 倍交叉验证中实现了最高诊断 AUROC（0.892）和精确度（0.802），概念宏 AUROC 也由 0.741 提升到 0.771；空间定位指标（Energy‑in‑Zone、Hit@1、Top‑5% Overlap）均显著提升，表明概念证据更符合临床解剖。

**⚠️ 局限性**

局限性：①依赖病灶分割结果，对分割精度有一定要求；②后方声带区域的定义为经验性、未考虑超声物理；③仅在单中心数据上验证，缺乏多中心泛化评估；④未进行放射科医生用户研究来验证解释是否真正提升临床信任。

---

## 191. Cardinality-Decomposed Loss: Matching Training Objectives to Relation Structure in Heterogeneous Recommendation Graphs

**arXiv ID:** 2607.20737 | [PDF](https://arxiv.org/pdf/2607.20737v1)

**作者:** Parul Maheshwari `[一作]` (PayPal), Prakhar Mehrotra `[通讯]` (PayPal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了按边缘基数区分损失函数的Cardinality-Decomposed Loss（CDL），以防止异构图中属性嵌入崩溃；

**💡 创新点**

创新点在于将一对多边缘使用BPR、一对一边缘使用交叉熵，并通过λ平衡两者，解决梯度冲突；

**🔧 技术方法**

使用HeteroSAGE GNN、BPR与交叉熵组合、梯度余弦相似度评估冲突、λ扫网；

**📊 数据集**

在MovieLens-1M、Last.fm-360K、Yelp、BookCrossing和PayPal Audience Factory五个数据集上进行实验；

**📈 对比分析**

与仅使用BPR的CF基线对比，属性判别度提升30–42个百分点，NDCG在语义对齐度高的数据上提升+7.8%、+2.9%、+3.3%，低对齐时可接受NDCG轻微下降；

**⚠️ 局限性**

局限在于λ需手动调参、未实现自适应调度，且在已存在拓扑泄露的图中属性提升有限，未评估在超曲几何或对比预训练上的表现。

---

## 192. Decentralized UAV Swarms for Ground Target Protection in GPS- and Communication-Denied Environments

**arXiv ID:** 2607.20710 | [PDF](https://arxiv.org/pdf/2607.20710v1)

**作者:** Dimitria Silveria `[一作]` (Queen's University), Sidney Givigi `[通讯]` (Queen's University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在GPS、通信受限环境下，利用无人机编队对地面目标进行围剿和防御的完整流水线。

**💡 创新点**

创新点在于：①无GPS、无通信，仅利用相对测量完成目标状态与编队相位估计；②引入不变扩展卡尔曼滤波器（IEKF）估计地面及空中目标；③自适应围剿控制器根据目标速度动态调节角速度与半径；④整个系统在真实硬件上实现并完成单/多攻击者测试。

**🔧 技术方法**

采用离散卡尔曼滤波器估计相邻无人机的相位差；不变扩展卡尔曼滤波器估计目标状态；基于相位差控制的离散式角速度控制器；自适应半径公式与速度比例控制。

**📊 数据集**

使用Crazyflie四旋翼与Limo地面车在Vicon追踪室内实验，生成相对测距数据并加入高斯噪声模拟LiDAR。

**📈 对比分析**

与单纯基于测距或中心化方案对比，系统在3D位置RMSE仅0.03m，相位误差≤7°，在单机攻击情境下成功完成围剿与中和，说明方法具有高精度与鲁棒性。

**⚠️ 局限性**

限制包括：对相位控制假设所有无人机角速度相同导致估计误差在目标速度变化大时增大；缺乏对半径误差的显式校正；只能处理单一攻击者；实验规模受限于硬件数量，未在大规模或动态多攻击者场景验证。

---

## 193. NVIDIA-labs OO Agents: Native Python Object-Oriented Agents

**arXiv ID:** 2607.20709 | [PDF](https://arxiv.org/pdf/2607.20709v1)

**作者:** Paul Furgale `[一作]`, Ricardo Silveira Cabral `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于Python对象的统一代理框架 NOOA，将代理行为、状态、工具和上下文直接映射为 Python 类、方法、字段和注解。

**💡 创新点**

创新点在于将六大接口能力（类型化 I/O、引用传递、代码即行为、可编程循环、对象状态、模型可见的 Harness API）整合到单一表面，并通过“……”方法体实现 LLM 驱动的循环，消除了传统框架的多文件、DSL 和手动序列化痛点。

**🔧 技术方法**

主要技术包括 Python 类和类型注解、@strategy 装饰器控制策略、CodeAct 代码执行循环、ContextManager 与 EventManager 生成可读、可追溯的上下文，及通过预览与截断实现的引用传递。

**📊 数据集**

使用了公开 Benchmark 数据集：SWE‑bench Verified、Terminal‑Bench 2.0、CyberGym L1、ARC‑AGI‑3 以及 88 条能力测试用例，覆盖类型签名、状态管理、工具调用、错误恢复等场景。

**📈 对比分析**

通过与 14 个现有代理框架对比，NOOA 在所有六大接口能力上均实现了完整支持；在 Benchmark 上表现优于公开通用框架，接近或超过专用系统（SWE‑bench 最高 82.2% / Terminal‑Bench 73.0%，CyberGym 86.8% / ARC‑AGI‑3 85.1%），并在 Token/调用成本上更高效。

**⚠️ 局限性**

局限性主要在于代码执行在主进程内，依赖 sandboxing 保障安全；引用传递的实现不适用于跨进程或多语言环境，且对极大输入的预览策略仍有空间改进；复杂的策略配置和错误恢复机制仍需要进一步简化和自动化。

---

## 194. Improving the performance of an ASV system using hybrid speech features

**arXiv ID:** 2607.20706 | [PDF](https://arxiv.org/pdf/2607.20706v1)

**作者:** Stanisław Ciszkiewicz `[一作]` (Warsaw University of Technology), Artur Janicki `[通讯]` (Warsaw University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了使用混合声学特征提升ASV系统在干净与噪声条件下的性能

**💡 创新点**

提出将基于声带非线性模型的RAB描述子与传统MFCC/PNCC/CQCC等特征组合，显著提升噪声鲁棒性

**🔧 技术方法**

采用MFCC、CQCC、PNCC与RAB特征提取，使用高斯混合模型（GMM）进行说话人建模，评估EER指标

**📊 数据集**

使用Google Speech Commands (GSC) 数据集（30名说话人各100条录音）以及NoiseX-92噪声数据库

**📈 对比分析**

通过对120条测试录音与30个模型进行3600次对比，使用EER作为性能指标；结果显示在噪声环境下PNCC+RAB混合特征将EER下降至约0.7%–1.5%，比单一特征低4–7个百分点

**⚠️ 局限性**

受限于样本量（仅30名说话人）与传统GMM后端，需进一步验证在更大规模数据和更先进深度学习后端的效果

---

## 195. CEDAR: Causal Edge Discovery for Autoregressive Processes

**arXiv ID:** 2607.20696 | [PDF](https://arxiv.org/pdf/2607.20696v1)

**作者:** Mohammad Fesanghary `[一作]` `[通讯]` (Bloomberg LP), Mohammad Fesanghary (Bloomberg LP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种约束式方法Causal Edge Discovery for Autoregressive Processes（CEDAR），用于稀疏自回归时间序列的滞后因果边发现；

**💡 创新点**

创新点包括：①使用AR(1)-残差化的非线性距离相关和解析t检验进行高效滞后筛选；②引入稳定的Momentary Conditional Independence（MCI）剪枝，消除间接路径；③加入确定性C‑node处理趋势型非平稳，提升稳健性；

**🔧 技术方法**

技术手段主要为：AR(1)残差化、U‑centered距离相关、BH多重检验、两条件CI测试（Condition 1 & 2）、MCI剪枝、C‑node基准；

**📊 数据集**

数据集包括：多种拓扑（Erdős‑Rényi、scale‑free、小世界）的合成稀疏结构因果过程（不同维数d与样本量T）；真实河网数据——Elbe河流量分段数据；

**📈 对比分析**

与PCMCI+、LPCMCI、+（PCMCI+风格改进）等方法比较，CEDAR在数据稀缺（T≤200）且自回归为AR(1)的稀疏场景下F1最高或相当，T≥500时性能趋于PCMCI+或+；在河网实验中结合PELT分段后，CEDAR恢复10/11条真边，F1≈0.83；

**⚠️ 局限性**

局限性包括：假设每变量只有AR(1)自回归；未考虑潜在混杂；解析t检验在强自相关下可能保守；C‑node对趋势基准依赖，误差会削弱效果；对于高阶AR或多滞后因果，方法可能漏检或误检。

---

## 196. Language Models Embody and Amplify Human Cognitive Distortions: What Is to Be Done?

**arXiv ID:** 2607.20695 | [PDF](https://arxiv.org/pdf/2607.20695v1)

**作者:** Arnau Marin-Llobet `[一作]` (Harvard University), Mahzarin R. Banaji `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述LLM中隐蔽且扩大的社会偏见，并提出诊断、监管和操作层面的对策

**💡 创新点**

指出对齐训练可能加剧偏见，提出公开偏见注册与监管框架

**🔧 技术方法**

未使用具体技术，仅基于已有研究综述

**📊 数据集**

未使用数据集

**📈 对比分析**

未进行实验对比

**⚠️ 局限性**

缺乏量化实验与可操作细节

---

## 197. End-to-End Learning of Safe Optimal Feedback Control in High Dimensions with Control Barrier Function Layers

**arXiv ID:** 2607.20674 | [PDF](https://arxiv.org/pdf/2607.20674v1)

**作者:** Xingjian Li `[一作]` (University of Texas at Austin), Samy Wu Fung `[通讯]` (Colorado School of Mines)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种可扩展的端到端训练框架，用于在高维非线性控制系统中学习满足控制势函数（CBF）硬约束的半全局反馈控制器。

**💡 创新点**

核心创新在于将 Davis–Yin 拆分（DYS）与 Jacobian-Free Backpropagation（JFB）相结合，既可在高维 QP 层实现可微安全过滤，又避免了传统显式梯度求解所导致的计算与内存瓶颈；并提供了针对非光滑安全层的收敛证明。

**🔧 技术方法**

采用了控制势函数、Davis–Yin 拆分、Jacobian‑Free Backpropagation、CLarke 广义雅可比以及非光滑收敛分析技术。

**📊 数据集**

在多智能体路径规划任务上进行实验，涵盖单积分器（50 维）、双积分器（4、24 维）以及三维四旋翼（60、360、1200 状态维度）等模型。

**📈 对比分析**

与两种基准方法（自动微分 unrolling 及 CVXPY Layers 隐式微分）对比，所提出方法在所有任务中均能稳定收敛，内存占用下降 10 倍以上，训练时间显著缩短；在运行成本与终端成本上保持竞争力，且始终满足安全约束。

**⚠️ 局限性**

目前仍受限于对 HOCBF 细节与极端大规模问题（>1200 状态维）在数值稳定性与收敛速度方面的挑战，未来需进一步优化迭代收敛与求解器鲁棒性。

---

## 198. LegalCiteTrust: Benchmarking Citation Trustworthiness in Chinese Long-Form Legal Research Reports

**arXiv ID:** 2607.20872 | [PDF](https://arxiv.org/pdf/2607.20872v1)

**作者:** Yunhan Li `[一作]` (City University of Macau), Min Yang `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LegalCiteTrust基准，用于评估中文长篇法律研究报告中引用的法律权威的可信度；

**💡 创新点**

创新点在于将引用可信度拆分为存在性、真实性与适用性三维度，并与覆盖率、支持度三大报告层面指标并列评价；

**🔧 技术方法**

采用LLM与代理研究系统结合检索、后处理和引用验证的技术，实验中使用了多种检索工具和E/F/A反馈机制；

**📊 数据集**

使用了13,622篇中文法律研究文档构成的72个任务集，涵盖236个子议题、585条细化标准和1,459条证据点；

**📈 对比分析**

通过主排行榜、工具消融和验证消融三种评估方式对比系统表现，发现不同系统在覆盖率、支持度与引用可信度之间存在显著权衡，E/F/A反馈显著提升信任度和最终分数；

**⚠️ 局限性**

局限包括：评估仍需人工审核，数据集仅覆盖中国法律领域且难以直接迁移至其他司法辖区；引用验证受限于可检索来源；评估聚焦于引用可信度，未覆盖法律建议的整体质量。

---

## 199. ViSTR-Bench: Can MLLMs Reason from Continuous Visual Cues in Dynamic Scenes?

**arXiv ID:** 2607.20868 | [PDF](https://arxiv.org/pdf/2607.20868v1)

**作者:** Han Li `[一作]` (Beihang University), Naiyan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ViSTR‑Bench，一个专注于多模态大型语言模型（MLLMs）在动态视觉场景中从连续视觉线索进行定性时空推理的基准测试集；

**💡 创新点**

创新点在于：①从时空维度拆分四大推理维度（运动感知、空间关系、结果预测、物理动力学）并覆盖15个子任务；②通过事件定位、视觉提示和结果截断等预处理确保每个样本仅依赖连续视觉证据；③提供人类基准与多类模型（专用空间MLLM、公开通用MLLM、专用空间MLLM）对比，揭示当前模型的局限。

**🔧 技术方法**

使用多模态推理框架，结合文本链式推理（CoT）提示、视频/多帧视觉输入、视觉工具（光流、三维重建）等技术；

**📊 数据集**

利用公共视频数据集（Waymo、ScanNet、Ego4D等）、Web 视频和自采录制的 1340 条 QA 对；

**📈 对比分析**

对比了 12 种专有与 9 种开源通用 MLLM 以及 9 种专用空间 MLLM，最佳模型 GPT‑5.4‑thinking 仅 62% 的准确率，远低于人类 91%，显示出显著性能差距。

**⚠️ 局限性**

局限包括：①仅覆盖短时、二元决策任务，未扩展至长时、多主体、交互场景；②评价仅基于二元准确率，缺乏中间推理步骤或定量评估；③模型主要在目标跟踪、运动状态估计、结果预测和隐含物理关系推理方面表现欠佳。

---

## 200. Which Model Is Actually Serving You? IRIS: Budgeted Black-Box Auditing of Model Substitution and Routing Dilution in LLM Gateways

**arXiv ID:** 2607.20860 | [PDF](https://arxiv.org/pdf/2607.20860v1)

**作者:** Yuewei Zhang `[一作]` (Tsinghua University), Hanzhang Qin `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种完全基于返回文本的黑盒LLM门控审计框架（IRIS-audit），能够检测整个流的模型替换、分段替换（稀释）并精确估计稀释比例与被替换模型；该框架通过随机生成低熵序列的探针实现无标签识别，且在审计前先通过小样本试点估计所需查询预算。

**💡 创新点**

创新点在于：① 将随机生成探针与可复用的可见字符串特征相结合，首次实现跨模型替换与稀释检测、模型归因、稀释比例估计以及预估查询预算；② 提出了“估计→预算”策略，在审计前根据 pilot AUROC 指标自适应分配查询次数；③ 在理论层面给出指数误差衰减与稀释的 1/ε 与 ε^-2 预算阈值，并提供置信区间估计。

**🔧 技术方法**

使用的技术包括：随机生成探针（c_n,L），可见字符串特征向量 φ(y)（179 维统计），随机森林多分类器 g_c，平均得分 S_m，阈值 τ_mean、τ_cj，Binomial/任何比例检验，最小二乘归一化求解归因权重，delta‑method 置信区间，CLT/Clopper–Pearson 上界估计等。

**📊 数据集**

数据集：本地 Qwen3 ladder（6 个同族模型），OpenRouter 17/45 个商用/开源模型，第三方 MET 公开数据集，真实生产端点（DeepInfra、Novita、Together 等）以及多轮自定义探针测试。

**📈 对比分析**

与 FLIPS、MET、RUT、KBF、B3IT 等黑盒审计器比较，IRIS 在替换检测、稀释检测、模型归因、稀释比例估计上表现更佳；在已知替换时 Hit@B 达到 87% 甚至 90%，比固定预算方案提升 14%；在稀释比例 ϵ=0.3 的情况下检测功率 85%，误报率 1.7%。

**⚠️ 局限性**

局限性包括：① 对极低熵探针（如 c_2,1）或极高温度/低温度极端采样时检出率下降；② 对未知稀释模型的估计仅能给出下界；③ 审计假设请求独立，突发粘性路由会影响置信区间；④ 需预先构建探针库与参考样本，对新模型或动态更新需周期刷新；⑤ 对完全自适应语义路由仍是开放挑战。

---

## 201. Offline RL with Hierarchical Action Chunking

**arXiv ID:** 2607.20834 | [PDF](https://arxiv.org/pdf/2607.20834v1)

**作者:** Ahad Jawaid `[一作]` `[通讯]` (University of Texas at Dallas), Ahad Jawaid (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了一种Hierarchical Implicit Q-Chunking (HiQC) 的离线目标条件强化学习算法，通过高层潜在规划与低层动作块执行双层时间抽象来缓解“horizon curse”。

**💡 创新点**

在高层规划与低层块化之间进行双重时间抽象，并将低层批判器条件化于完整动作块以实现无偏k步价值回溯，理论上误差上界从O(√T)降至O(√(T/k))。

**🔧 技术方法**

结合Implicit Q-Learning (IQL) 的期望回归、优势加权回归 (AWR)、条件流匹配 (Conditional Flow Matching) 作为低层块化策略，并使用k步Q-Chunking 与潜在子目标表示。

**📊 数据集**

在OGLBench套件中的9个任务（包括antmaze、humanoid、pointmaze、cube-triple、puzzle-4x6、scene-play等）进行评估，特别关注“giant”版长时序任务。

**📈 对比分析**

与平面、动作块化、层次化基线（GCBC, QC, DQC, HIQL等）比较，HiQC在长时序导航任务（如antmaze-giant、pointmaze-giant）取得最高成功率，整体平均得分位列第一，尤其在极长路径任务上显著优于其他方法。

**⚠️ 局限性**

仅使用固定块大小 k，过大块易产生开放式执行误差；理论分析依赖每步误差有界假设，未考虑深度网络逼近与采样误差；对随机环境与更深层次层级的适用性尚待验证。

---

## 202. REFACT: Adaptive Fact Restatement for Compact and Faithful Chain-of-Thought Reasoning

**arXiv ID:** 2607.20833 | [PDF](https://arxiv.org/pdf/2607.20833v1)

**作者:** Zhensheng Jin `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出适应性事实重述引用框架，训练大型语言模型在推理过程中判断何时需要引用以及以何种粒度重述来源事实；

**💡 创新点**

将引用转化为支持答案的中间状态，采用两阶段SFT+RL训练并设计四项奖励（格式、准确性、可追溯性、答案可行性）确保引用既可追溯又足以回答问题，避免无依据推理和无意义复制；

**🔧 技术方法**

基于大型语言模型的两阶段训练（监督微调+GRPO强化学习）、引用实用奖励、教师模型生成高质量事实重述、验证模型评估引用可行性等技术；

**📊 数据集**

训练数据来自HotpotQA_CARE和MuSiQue；评估数据使用LongBench v1/v2、LV‑Eval、ConFiQA；构造长上下文时使用小说/期刊文章作为干扰；

**📈 对比分析**

与Zero‑Shot、LongAlign、LongFaith、CARE等基线比较，在LongBench v1、LV‑Eval上取得最高F1/准确率；相比CARE推理长度缩短约50%；在ConFiQA中源偏好更高、参数覆盖率更低，表明更依赖上下文；

**⚠️ 局限性**

依赖教师模型和验证模型生成高质量轨迹，训练成本较高；仅在Qwen3‑4B/8B等模型上验证，跨模型/多语言泛化尚待探索；构造干扰文本可能影响训练质量。

---

## 203. An Improved Linear Extractable Sketch Data Structure for Flow Count Statistics

**arXiv ID:** 2607.20830 | [PDF](https://arxiv.org/pdf/2607.20830v1)

**作者:** Patthadon Tantiameorn `[一作]` (Kasetsart University), Jittat Fakcharoenphol `[通讯]` (Kasetsart University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对网络流量计数问题，本文改进了FermatSketch线性可提取草图结构，在桶中存储多组线性组合以放宽纯桶要求，实现更高的空间利用率。

**💡 创新点**

创新点在于：①将每个桶扩展为多槽（k槽），每槽保存多组计数与ID线性组合；②利用系数矩阵的全秩性通过穷举求解线性系统，从而在非纯桶中仍能恢复流计数；③在空间节省与提取时间之间实现可调的权衡。

**🔧 技术方法**

技术手段包括：哈希映射、模p算术、可逆Bloom Lookup表思想、Gaussian消元求解有限域线性方程、系数矩阵穷举搜索与矩阵秩判定。

**📊 数据集**

实验使用合成数据，随机生成不同数量的受害流（从300到500），并在多种桶/槽配置下测量提取成功率和空间占用。

**📈 对比分析**

与原FermatSketch在相同空间下比较，改进方案在99%成功率下对100个受害流可将空间减少约34.9%，对较大流量（200+）则降幅约8%；空间相同下提取成功率提升显著，但提取耗时随L和k^2呈指数级增长。

**⚠️ 局限性**

局限性包括：①提取时间与系数集合大小L、槽数k成指数关系，实际实现受限于可接受的计算开销；②需假设模p运算下的线性系统可逆，理论证明依赖特定概率假设；③实验仅基于合成数据，缺乏对真实网络流量的验证。

---

## 204. Auditing Provenance Sensitivity in LLM Agent Action Selection

**arXiv ID:** 2607.20827 | [PDF](https://arxiv.org/pdf/2607.20827v1)

**作者:** Junchi Liao `[一作]` `[通讯]`, Junchi Liao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型代理在执行工具调用和参数填充时，是否会受到未授权上下文的影响进行细粒度的授权审计。

**💡 创新点**

创新点在于提出目标特定授权标签、匹配源干预、有效证据退化与合作交互诊断相结合的审计框架，能够在完整提示与部分证据条件下分别定位未授权信息的影响。

**🔧 技术方法**

采用了目标-因素拆分、源权威干预、Harsanyi/Shapley 交互分析、模型分数对数几率评分以及生成动作的解析，配合多模型（Qwen3‑4B/30B、Mistral‑24B、Llama‑70B、DeepSeek‑V2‑Lite）对比。

**📊 数据集**

使用了 450 条受控下一动作任务集，来自 AgentAudit、Tau2Audit、BFCLAudit 三个来源，共 1,350 个工具/参数目标和 10,800 个目标‑因素标签。

**📈 对比分析**

通过多模型对比，发现受信任来源的支持提升约 +0.45 对数几率、竞争降低约 –2.0，对数几率差异在 1.15 以上；生成动作的源敏感度为支持 1.7% 与竞争 5.4%；当加入显式授权策略时，未授权竞争错误率可从 14.9% 降至 9.5%，提升 5.4 点。

**⚠️ 局限性**

局限在于：仅在人工构造的受控任务中评估，未覆盖真实交互场景；未能完全隔离内部推理链中的来源信息；以及模型对未授权信息的抵消效果不均衡，部分模型仍对未授权竞争产生显著影响。

---

## 205. SubSplat: High-Resolution Pixel-aligned 3DGS via Sub-pixel Gaussian Reparameterization

**arXiv ID:** 2607.20813 | [PDF](https://arxiv.org/pdf/2607.20813v1)

**作者:** Jiun Lee `[一作]` (AimFuture), Sangmin Lee `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SubSplat，通过 Sub‑pixel Gaussian Reparameterizer (SPGR) 在低分辨率特征上细化高分辨率的 3D Gaussian 原语，实现高质量新视角渲染；

**💡 创新点**

创新点在于将每个像素对齐的 Gaussian 拆分为多子像素原语，并通过可变分辨率的 SPGR 与多视角特征聚合，打破传统像素对齐方法的分辨率‑效率折衷；

**🔧 技术方法**

采用 3D Gaussian splatting、像素对齐网络、可变形注意力（Deformable Attention）进行特征聚合、子像素重参数化、透明度分配与颜色调节等技术；

**📊 数据集**

使用 RealEstate10K（720p/1080p）和 ACID（720p/1080p）数据集进行训练与评估；

**📈 对比分析**

与 PixelSplat、MVSplat、HiSplat、TranSplat、DepthSplat 及 2D upsampler（Bilinear、HiT‑SR）等基线比较；在 512×512 与 1024×1024 输出上，SubSplat 取得 PSNR 25.52/26.03、SSIM 0.850/0.775、LPIPS 0.167/0.216，延迟仅 42 ms，显著优于基线；

**⚠️ 局限性**

局限性包括对子像素原语密度的内容感知控制不足、原语数量增长导致渲染开销上升，以及在极复杂几何或动态场景中的性能尚待验证。

---

## 206. A generalized shape function approach for multimaterial topology optimization

**arXiv ID:** 2607.20784 | [PDF](https://arxiv.org/pdf/2607.20784v1)

**作者:** Swagatam Islam Sarkar `[一作]`, Prabhat Kumar `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于形状函数的多材料拓扑优化方法，能够使用n个设计变量映射到2^n种材料分布

**💡 创新点**

创新点在于将形状函数映射到超立方体的顶点，实现材料分布的自然归一化与边界约束；并引入密度与投影滤波器保证数值稳定与结果稀疏

**🔧 技术方法**

使用了多维线性插值形状函数（1D、2D、3D、4D、5D）、SIMP材料插值、密度滤波和投影滤波以及MMA优化器

**📊 数据集**

采用标准拓扑优化基准案例（MBB梁、悬臂梁、反转器、抓手）作为数据集，各案例中对不同材料数量设定体积分数

**📈 对比分析**

通过与传统SIMP方法在相同几何、边界与体积分数条件下的对比，得到了更低的目标函数值、明显更少的灰色元素和更清晰的材料边界，收敛稳定，迭代次数在400以内

**⚠️ 局限性**

局限性包括：多材料问题仍然是非凸的，易陷入局部最优；需要预先设定足够大的体积分数与材料弹性对比；对极大材料数量的求解仍需更多计算资源

---

## 207. Agentic Designer: Progressive Multi-Agent Collaboration for Structure-Aware Interior Layout Generation

**arXiv ID:** 2607.20866 | [PDF](https://arxiv.org/pdf/2607.20866v1)

**作者:** Zhijing Yang `[一作]` (Guangdong University of Technology), Tianshui Chen `[通讯]` (Guangdong University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多代理协作的进化式室内布局生成框架——Agentic Designer，通过逐步生成、评估与细化，实现结构感知的家具布局。

**💡 创新点**

创新点在于：①将布局生成拆分为生成器、评估器、细化器三大角色；②引入Progressive Consensus Mechanism（PCM）保证每一步在满足几何约束后才进入下一步，避免错误累积；③采用程序式文本表示与LLM进行空间推理，使得模型能够在保持可解释性的同时精准处理墙、门、窗等结构约束。

**🔧 技术方法**

技术手段包括：基于Qwen2.5-Coder-7B的大语言模型，采用LoRA微调；生成器采用自回归策略，评估器实现几何违约检测，细化器实现基于诊断信息的去噪修正；实现层面采用结构化代码格式、程序式布局表示和逐步共识机制。

**📊 数据集**

使用了新构建的InStruct基准数据集（18,853套房间，包含墙、门、窗的结构注解），并对3D-FRONT进行结构化改造以兼容该任务。

**📈 对比分析**

与DiffuScene和SemLayout等最先进的一步生成方法比较，Agentic Designer在结构合规率（SVR）从90%以上降至约17%，边界违约计数（BVC）降低至0.3，碰撞得分（CS）显著下降，同时平均家具数（ANF）提升，用户研究中循环通道、位置合理性、比例平衡的Likert评分均超过4.7，表明性能明显优越。

**⚠️ 局限性**

局限性在于对极其不规则或带有深外凹陷的平面图仍可能出现拓扑误判，将家具错误放置在外部凹槽中；评估器目前仅基于坐标几何检查，缺乏全局拓扑感知，需要进一步加入视觉或占用图等辅助特征。

---

## 208. Position Bias is Hidden Behind Ceiling Effects: A Permutation Diagnostic for LLM Benchmarks

**arXiv ID:** 2607.20864 | [PDF](https://arxiv.org/pdf/2607.20864v1)

**作者:** Hiroki Tamba `[一作]` `[通讯]`, Hiroki Tamba

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了一个可与inspect_ai无缝对接的全排列答案顺序诊断工具，利用24,000次API调用在四大模型供应商和五个MMLU子任务上系统评估位置偏差；

**💡 创新点**

首次将全排列方法与统计检验（卡方/克雷默V）结合，构建了黄金区间（Goldilocks zone）和两种偏差机制（处理负荷与内容歧义）模型，并实现了无注册报告场所的预注册协议；

**🔧 技术方法**

采用全排列答案顺序、卡方检验、克雷默V效应量、Spearman秩相关、bootstrap置信区间，并通过Python实现inspect_ai扩展；

**📊 数据集**

使用MMLU benchmark的五个子任务（小学数学、大学数学、形式逻辑、专业医学、哲学），每个子任务包含50道题；

**📈 对比分析**

在gpt-4o-mini、claude-haiku-4-5、gemini-2.5-flash、grok-3四个模型上进行对比；结果显示仅在60–95%的基础准确率区间内可检出偏差，低阶模型呈单调早期偏好，高阶模型无可检测偏差，且通过克雷默V和形状诊断区分两种机制；

**⚠️ 局限性**

研究仅限于MMLU，黄金区间上下边界未完全验证；高阶模型的无偏差结果可能因上限饱和而非真实无偏差；推理链模型的偏差缺失未能确定是缓解还是未检测；成本不均导致复制难度增加；需要在非MMLU基准和前沿模型上进一步验证。

---

## 209. CSPF: A Constrained Shared-Private Fusion Method for Non-Verifiable Preference Evaluation

**arXiv ID:** 2607.20862 | [PDF](https://arxiv.org/pdf/2607.20862v1)

**作者:** Hehao Zhang `[一作]` (Institute of Automation Chinese Academy of Sciences), Xuange Gao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Constrained Shared-Private Fusion（CSPF）的隐藏状态融合方法，用以在非可验证的偏好任务中结合多模型的评估信息

**💡 创新点**

创新点在于将冻结的奖励模型的隐藏表示分解为共享与专家专属部分，并通过约束学习实现多视角的交互式融合，而非仅聚合最终标量得分

**🔧 技术方法**

使用了共享-私有编码器、Barlow Twins约束、SupCon对比损失、对标量得分的辅助校准等技术，训练目标为Bradley–Terry pairwise偏好损失

**📊 数据集**

主要使用LM‑Arena目标域偏好数据集进行训练和验证，并在PPE（Preference Proxy Evaluations）数据集上进行跨域评估

**📈 对比分析**

与单专家奖励模型、自然语言规则评估器以及传统标量得分聚合（RM Ensemble、LSC）等三大基线对比，CSPF在LM‑Arena验证准确率达68.04%，在PPE off6得分61.67%，均优于所有基线，尤其在跨域性能上显著提升

**⚠️ 局限性**

局限性包括仅在静态pairwise评估中验证，未测试在RLHF或最佳‑N选择等实际策略优化场景；仅考虑暴露隐藏状态的标量奖励模型，未涵盖生成式判定器和批判‑评分评估器

---

## 210. Multilevel Graph Wavelet Compressed Sensing with Scale-Aware Neural Recovery

**arXiv ID:** 2607.20857 | [PDF](https://arxiv.org/pdf/2607.20857v1)

**作者:** Amirhossein Nouranizadeh `[一作]` (Simplicial Technologies Inc.), Mengjia Xu `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了图波形压缩感知（GWCS）框架，利用谱图波形变换生成稀疏表示，并通过非参数多层重要性采样与尺度感知的神经逆变换实现离线压缩与重构。

**💡 创新点**

核心创新在于结合非参数多层重要性采样（MLIS）与可学习的尺度嵌入的神经逆图波形变换（NIGWT），既保留了可解释的稀疏压缩，又通过神经网络实现高质量恢复。

**🔧 技术方法**

使用了谱图波形变换（SGWT）与Chebyshev多项式近似、GraphSAGE消息传递的 Encode‑Process‑Decode 结构、稀疏自动编码器（SGAE）等技术。

**📊 数据集**

实验数据包括合成的近似带限图信号以及四个 PDE 模拟数据集：Turbulent Radiative Layer、Viscoelastic Instability、Kolmogorov Flow 与 Dynamic Stall。

**📈 对比分析**

与经典图信号采样方法（ESS、RSBS、FastGSSS、BSGDA）以及学习型基线（GXN、SGAE）对比，GWCS 在大多数数据集（尤其是光滑谱结构）上取得最低 MSE/RMSE，压缩率为 5% 时性能尤为突出。

**⚠️ 局限性**

局限性在于对能量分布广泛、湍流或瞬态的信号效果不佳，固定的波形基数可能无法捕获所有尺度能量；同时需要 Chebyshev 近似的计算开销较大。

---

## 211. Sonic Stage: Automatically Generating Interactive Spatial Soundscapes to Facilitate Dialogue Video Comprehension for Blind Viewers

**arXiv ID:** 2607.20835 | [PDF](https://arxiv.org/pdf/2607.20835v1)

**作者:** Shuchang Xu `[一作]` (Hong Kong University of Science and Technology), Yukang Yan `[通讯]` (University of Rochester)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一套名为 Sonic Stage 的系统，自动将含大量对话的影视视频转化为交互式空间音景，通过空间化对话、情境音效和交互式描述三种音频技术，帮助盲人/低视力观众在对话期间获取关键信息并保持沉浸式观看体验。

**💡 创新点**

创新点主要有：① 在同一场景的多镜头中重建统一的 3D 场景，并将音频映射到该空间，实现跨镜头的连续空间音效；② 将对话空间信息与情境音效内嵌化，避免传统音频描述在对话时段的空缺；③ 采用对话相关的交互式描述，减少不必要信息并降低观看中断；④ 通过自动化管线（VGGT、YOLO、BoT‑SORT、Gemini‑2.5 等）实现全流程无人工干预。

**🔧 技术方法**

技术手段包括：3D 视觉重建（VGGT）、人物检测与跟踪（YOLOv11+BoT‑SORT）、音频分离与转录（Gaudio Studio、Tencent Transcription API）、对话空间化与音量衰减优化、文本到音频合成（ElevenLabs）、动作缺失音效检测与生成（Gemini‑2.5）、交互式描述生成（Gemini‑2.5）、Unity+Steam Audio 渲染、触控交互等。

**📊 数据集**

使用了 16 条对话视频（涵盖电视戏剧、喜剧、电影戏剧、音乐剧、脱口秀等类别），在用户评估中选取 6 条视频（每类 1 条），每条视频长度 1.5–2 分钟，包含 87%–97% 的对话内容。

**📈 对比分析**

与基于 SPICA 的基线系统（支持屏幕空间空间探索和帧级描述）进行对照实验，12 名盲人/低视力受试者。结果显示：Sonic Stage 在角色位置、运动、动作和视觉细节的回忆准确率均显著提升（p < .01），空间存在感、叙事投入度和整体使用体验评分均高于基线；使用频率更高、暂停时长更短，整体观看时长相似。

**⚠️ 局限性**

局限性包括：仅处理单一场景且无跨场景切换的内容；对重叠语音或背景噪声处理不足；缺乏对相机切换信息的反馈；仅在短视频片段上验证，长视频可能出现注意力疲劳；对听障受众的可访问性评估不足；未整合触觉反馈等多模态支持。

---

## 212. Beyond Heavy Log Curation: Perplexity-Based APT Detection via Unsupervised, Context-Augmented Language Models

**arXiv ID:** 2607.20832 | [PDF](https://arxiv.org/pdf/2607.20832v1)

**作者:** Shoya Otsu `[一作]` (Mitsubishi Electric Corporation), Ye Wang `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于预训练语言模型的无监督、上下文增强的APT检测方法 CAPTAIN，利用 perplexity 对日志条目进行异常评分。

**💡 创新点**

创新点在于：仅使用极简领域无关预处理；通过编码历史日志生成少量上下文 token 并注入 decoder 语言模型；将 perplexity 视为时间序列并应用 Wiener 滤波平滑，提升检测稳定性，且不依赖人工特征或标签。

**🔧 技术方法**

技术包括 Transformer Encoder + Q‑Former 软提示桥接 + decoder-only LLM（如 Qwen3‑0.6B），自编码器式无监督训练，以及 Wiener 滤波平滑。

**📊 数据集**

使用公开 ATLAS 日志基准，分别在 AIRTAG 预处理和 CAPTAIN 预处理下评估。

**📈 对比分析**

通过 ROC AUC 与 1-AUC 进行比较；在标准预处理下 CAPTAIN 与 AIRTAG 竞争力相当，且在长日志/低预处理情况下平均 AUC 0.93，显著优于 AIRTAG 的 0.68；当 token 数增大时 CAPTAIN 的性能保持稳定。

**⚠️ 局限性**

局限性：无监督一类检测可能将合法行为变更误报；对极长日志仍有限制；上下文窗口大小与阈值需进一步自适应；在真实生产环境的泛化尚未完全验证。

---

## 213. Enhancing Explainable Cardiac Diagnosis with Guide-Grounded Multimodal LLMs

**arXiv ID:** 2607.20814 | [PDF](https://arxiv.org/pdf/2607.20814v1)

**作者:** Hai-Nam Duy Vuong `[一作]` (National Economics University), Thien Van Luong `[通讯]` (National Economics University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `109c2b71-d051-425c-831f-0c544c24280d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 CNN+Grad‑CAM+指导式知识注入的多模态 LLM 框架，用于生成可解释的心电图诊断报告。

**💡 创新点**

创新点在于将从权威心电图教材和指南中提炼的“解读指南”作为固定知识块注入，显著降低 LLM 的幻觉并提升报告与临床标准的一致性。

**🔧 技术方法**

采用 ResNet‑50 进行多标签心电图分类，Grad‑CAM 产生可视化热力图，离线知识蒸馏生成结构化指南，再将图像、热图、分类结果与指南一起输入多模态 LLM（Gemini‑2.5‑flash‑lite）进行报告生成。

**📊 数据集**

使用公开的 PTB‑XL 心电图数据集（约 2.2 万条 12‑lead ECG）进行训练、验证与测试。

**📈 对比分析**

通过与无指南注入的基线（CNN+Grad‑CAM+LLM）对比，分类性能基本相当，BERTScore 从 0.818 提升至 0.953，LLM 自动评估的胜率也从 38%/24% 提升至 62%/76%。

**⚠️ 局限性**

局限性包括指南内容需持续更新、注入完整指南增加上下文长度、LLM 自动评估仍受模型偏差影响，以及系统对高质量文本知识的依赖导致可扩展性受限。

---

## 214. Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling

**arXiv ID:** 2607.20791 | [PDF](https://arxiv.org/pdf/2607.20791v1)

**作者:** Phillip Howard `[一作]` (Thoughtworks), Amir Abdullah `[通讯]` (Thoughtworks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在高温采样下保持大型语言模型（LLM）拒绝行为的方法，并提出了“拒绝门控解码（Refusal‑Gated Decoding）”技术。

**💡 创新点**

创新点在于：①用少量贪婪解码的前缀检验来判断是否应继续高温采样，②通过学习拒绝前缀集合并在每一步判断兼容性，③重用 KV 缓存实现低额外延迟，③设计了早停策略，仅在确认非拒绝时才启动高温采样。

**🔧 技术方法**

采用的技术包括：高温采样与截断采样（p‑less）、贪婪解码 probe、兼容性判定（refusal‑prefix set）、自动前缀缓存（Automatic Prefix Caching）以及在解码时的早停/长度限制。

**📊 数据集**

使用了三大安全基准数据集：JailbreakBench（JBB）、XSTest、WildJailbreak，总计 2,650 条提示，涵盖约 1,300 个应拒绝、1,350 个应回答的样本。

**📈 对比分析**

对比方法包括：直接高温采样、在线 LlamaGuard‑4 路由器、SafeDecoding、朴素的贪婪‑后高温序列化。实验显示，拒绝门控解码在 91–99% 的情况下保持了贪婪拒绝行为，且非拒绝的额外代价仅为 1.1×–1.4× 的延迟（与直接高温采样相比），优于其它基线。

**⚠️ 局限性**

局限性：①依赖预先学习的拒绝前缀集合，可能无法覆盖所有新出现的拒绝模式；②对更大规模或未见模型的泛化尚未验证；③虽然延迟提升低，但在极低延迟应用中仍可能显著；④在极端高温下仍可能出现少量拒绝丢失。

---

## 215. 3D-GIMP: When 3D Gaussian Inpainting Meets PatchMatch

**arXiv ID:** 2607.20789 | [PDF](https://arxiv.org/pdf/2607.20789v1)

**作者:** Xuening Tian `[一作]` (University of Stuttgart), Shohei Mori `[通讯]` (University of Stuttgart)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

对3D Gaussian Splatting（3DGS）场景进行物体移除，通过先在参考视图上使用扩散模型完成单视图填充，然后利用3D感知的PatchMatch在所有训练视图中传播纹理，实现全局一致的高质量修补。

**💡 创新点**

创新点在于：①将昂贵的多视图扩散生成改为单视图扩散+基于纹理映射的传播，显著降低计算成本；②提出尺度一致的Poisson深度补全，解决单视图深度尺度不匹配；③构造全局映射场（Mapping Field）并结合可见性评分，在跨视图传播时保持几何与光照一致性；④通过与基线方法的混合（如 LeftRefill）进一步抑制幻觉，提升细节保真度。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、Stable Diffusion（Nano Banana Pro + Gemini 2.5）单视图填充、Poisson方程深度补全、3D-aware PatchMatch映射传播、alpha混合、基于Dirichlet边界条件的Poisson融合、以及后期轻量级优化。

**📊 数据集**

在IMFine、360-USID、Mip-NeRF 360等360°无界场景数据集上进行实验；每个数据集提供数百视图、目标物体、以及可选的参考视图。

**📈 对比分析**

与SPInNeRF、NeRFiller、GScream、InFusion、AuraFusion360、GEN3C等基线方法比较，结果显示：PSNR提升至17.47/19.09（相较于AuraFusion360 17.46/16.22），LPIPS降至0.344/0.248（相较于AuraFusion360 0.173/0.387），FID保持或略优，同时处理时间从数小时降至6-10分钟，显著提升速度与一致性。

**⚠️ 局限性**

局限性包括：①Poisson深度补全的尺度系数需要手动调节，场景特定；②当深度重建失效或极端几何复杂时，后续PatchWarp精度下降；③在严重遮挡或光照变化极大区域，基于纹理的映射可能产生弱监督导致细节失真；④需要用户手动标注3D ROI，未实现完全自动化。

---

## 216. Synthetic minority data is redundant or invalid: a data-dependent validity theory and a de-biased test

**arXiv ID:** 2607.20787 | [PDF](https://arxiv.org/pdf/2607.20787v1)

**作者:** Ahmad B. Hassanat `[一作]` (Mutah University), Ghada A. Altarawneh `[通讯]` (Mutah University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并验证了基于保留真实数据的去偏估计器，用来评估合成少数类样本的真实性，并在91种过采样方法及深度生成器上进行大规模审计。

**💡 创新点**

首次提出将真实性定义为总体量，并给出一致去偏估计器；证明自证性不可行，揭示类重叠设定的真实性下限；并推出联合有效性与信息增益的公开标准。

**🔧 技术方法**

采用1‑NN/k‑NN投票估计、Le Cam极限理论与一致性证明、统计实验设计（LogReg、XGBoost、LightGBM）、深度生成模型比较及阈值移动交叉验证等技术。

**📊 数据集**

在6个公开不平衡二分类数据集（Pima糖尿病、Framingham心脏病、胸腔手术、Spambase、MAGIC、交通事故严重度）以及高维信用卡欺诈数据上进行实验。

**📈 对比分析**

与三种基线（不采样、类别权重、阈值移动）对比；大多数方法在真实性上未达标，信息增益极小（F1提升<0.01），且往往导致校准恶化。

**⚠️ 局限性**

在维度过高或样本极少时估计方差升高，k‑NN需要调参；对非二分类或回归问题不直接适用；部分方法在空间变换后无效。

---

## 217. Rethinking Open-World Video Anomaly Detection: Diagnosing Definition Blindness

**arXiv ID:** 2607.20780 | [PDF](https://arxiv.org/pdf/2607.20780v1)

**作者:** Inpyo Song `[一作]`, Jangwon Lee `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对开世界视频异常检测（OWVAD）的定义依赖性进行研究，发现并量化了“定义盲点”现象，提出三种新评测指标（DC-Disc、DC-DetΔ、DC-SelΔ），并设计了去除共享异常信号的定义对比评分方法 DeCoS。

**💡 创新点**

创新点在于：①识别并解释 OWVAD 评测中忽略定义跟随的失效模式；②通过权重分解揭示动态定义评测被目标‑正常分离压制的原因；③提出三种基于定义对比的评测指标，逐步消除正常帧、通用异常和多事件选择的捷径；④构造 DeCoS，将共享异常证据置零、实现零和对比，并通过轻量读出层实现查询相对分配。

**🔧 技术方法**

主要技术手段包括：CLIP 视觉‑文本嵌入、基于余弦差的 margin 计算、归一化中心化、零和约束、时间残差读出、定义选择损失（cross‑entropy）训练，以及在 UCF‑Crime、XD‑Violence、MSAD 等数据集上的评测。

**📊 数据集**

使用的公开数据集有 UCF‑Crime、XD‑Violence、MSAD；此外对比了弱监督 VAD、开词表 VAD、语言导向 OWVAD（LaGoVAD）、生成式 VLM 等方法。

**📈 对比分析**

实验对比 LaGoVAD、VadCLIP、PLOVAD、LLaVA‑1.5、InternVL3、Qwen2.5‑VL 等，结果显示传统 AUROC/Drift@5 评测对大多数模型无效；在 DC‑Disc、DC‑DetΔ、DC‑SelΔ 三个指标上，DeCoS 显著提升，AUROC 提升 7.3–16 点，定义跟随差距提升 15.5–28.3 点，证明其在定义条件下的显著优势。

**⚠️ 局限性**

限制：①DeCoS 依赖多定义对比，单定义情形下有效性较弱；②DC‑SelΔ 使用合成两事件切片，可能受到拼接伪影影响，虽然在自然多事件视频上验证但样本有限；③当前基准缺少规模化天然多事件视频及对应的查询时间标注，限制了评测的真实性。

---

## 218. Webly Supervised Multi-Label Recognition: Evaluation Benchmark and Dual-Branch Multi-Label Contrastive Learning

**arXiv ID:** 2607.20874 | [PDF](https://arxiv.org/pdf/2607.20874v1)

**作者:** Zhihua Xu `[一作]` (Guangdong University Of Technology), Tianshui Chen `[通讯]` (Guangdong University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了面向 Web 端多标签识别（WS-MLR）的统一评估基准 Web-COCO 与 Web-Pascal，并在此基准上重新实现并对比多种代表性方法，提出了双分支多标签对比学习框架 DBMLCL，用于在海量 Web 图像中识别多标签并纠正噪声标签。

**💡 创新点**

创新点主要包括：①双分支网络同时学习类别特定的实例级与类别级特征；②引入实例对比损失与原型对比损失，提升类别间判别力；③基于特征-原型相似度与预测置信度的阈值机制实现自适应噪声标签纠正。

**🔧 技术方法**

技术方法：双分支 ResNet-101 + Semantic‑Aware Representation Learning (SARL) 模块；实例对比损失、原型对比损失；噪声标签校正算法；使用 Adam 优化器、温度参数 τ=0.1、动量 m=0.999 等训练细节；与 CLIP 的无监督多标签识别基线对比。

**📊 数据集**

数据集：Web-COCO（约 290k 图像，80 类）与 Web-Pascal（约 236k 图像，20 类）来自 Web 搜索；验证/测试集使用 MS‑COCO 与 Pascal VOC 的人工标注子集。

**📈 对比分析**

与 SSGRL、ML‑GCN、CSRA、ASL、P‑GCN、KGGR 以及 CLIP‑基线 CCD 等方法对比，DBMLCL 在 mAP、OF1、CF1 上均取得最高分，尤其在 Web‑COCO 上 mAP 达 71.4%（相比 CSRA 的 70.3% 提升 1.1%）且在不同数据比例下表现稳健；但模型参数与 FLOPs 较大，训练吞吐量最低。

**⚠️ 局限性**

局限性：①双分支训练导致显著的训练开销和内存占用；②Web‑COCO / Web‑Pascal 仍受搜索引擎偏倚、长尾分布、离题图像及语言依赖噪声影响；③尚未充分融合 CLIP 等视觉‑语言先验以进一步提升噪声纠正与泛化性能。

---

## 219. LO-FAR: A Cost-Aware Local Filter for Sparse Feature Ranking in Industrial Ad Recommendation

**arXiv ID:** 2607.20873 | [PDF](https://arxiv.org/pdf/2607.20873v1)

**作者:** Egemen Erbayat `[一作]` (Meta Platforms, Inc.), Srihari Reddy `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个只使用CPU、与模型无关的本地稀疏特征排名方法 LO-FAR，用来从数百个短 ID‑列表特征中快速筛选出高价值特征。

**💡 创新点**

创新点在于将特征排名完全解耦离下游完整模型训练，采用每个特征独立的局部 ID 级预测器与聚合评分，显著降低计算成本并实现可并行、可重复的 CPU 端工作流。

**🔧 技术方法**

技术方案包括：样本抽样、训练/测试划分、特征爆炸、经验正率/最近邻回退的局部概率估计、平均聚合以及基于交叉熵等指标的持有评估。

**📊 数据集**

实验使用生产级 1M+ 条用户‑广告交互日志，包含 475 个 99% 位长 <5 的短 ID‑列表特征。

**📈 对比分析**

与覆盖率启发式、Shuffle‑based 重要性以及 BSN 等三种基线在 CTR/CVR 任务下比较；LO‑FAR 在 100–400 个特征保留时保持与 Shuffle 和 BSN 相近的 NE 增益，排名耗时约 2 CPU 小时，成本远低于 GPU 日；在 CPU‑only 环境下实现了可接受的性能。

**⚠️ 局限性**

局限性包括：仅适用于短 ID‑列表特征，对长列表需改进；缺乏交互信息，无法捕捉仅通过特征交互显现的信号；排名结果仅作为筛选依据，最终仍需在下游模型中验证。

---

## 220. Auditing Evidence Use in Medical LLM Diagnosis

**arXiv ID:** 2607.20848 | [PDF](https://arxiv.org/pdf/2607.20848v1)

**作者:** Junchi Liao `[一作]` (University of Electronic Science and Technology of China), Fuji Ren `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于证据干预的行为审计，用以测量医学LLM在诊断决策中的证据使用方式。

**💡 创新点**

创新点在于将诊断相对的证据角色与低阶交互效应相结合，区分合法的差异诊断冲突与潜在的证据误用，并通过鲁棒性筛选得到高置信度的失败机制。

**🔧 技术方法**

采用多种技术：多提示下的候选诊断评分、诊断相对证据角色定义、交互效应计算（Shapley‐style）、自动标签与临床审核相结合的判定流程，以及针对提示、选项顺序和扰动的鲁棒性验证。

**📊 数据集**

使用了三个医学诊断数据集：DDXPlus（结构化症状/历史）、CupCase与MedCase（基于案例报告的非结构化文本），并在五个开源LLM上进行评估。

**📈 对比分析**

与传统的整体诊断准确率对比，发现准确率并不能反映证据使用质量；通过交互分析显示大部分交互为合法的支持或冲突，失败机制仅占少量，但在DDXPlus上可通过临床审核确认，鲁棒性筛选后准确率提升至约80%。

**⚠️ 局限性**

局限性包括：依赖于手工选择的证据单元与候选集、对提示词与选项顺序敏感、仅测量提示条件下的行为而非内部推理、临床审核样本为高强度筛选样本，缺乏总体普适性估计。

---

## 221. Efficient and Interpretable Body-Based Emotion Recognition with Lightweight Temporal Convolutional Networks

**arXiv ID:** 2607.20820 | [PDF](https://arxiv.org/pdf/2607.20820v1)

**作者:** Christian Arzate Cruz `[一作]` (Honda Research Institute Japan), Houshyar Asadi `[通讯]` (Deakin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并验证轻量化时序卷积网络（TCN）在基于身体运动的情感识别中的有效性，并与图卷积基线（G-TSG）进行对比。

**💡 创新点**

提出在保持高识别性能的同时显著降低参数量和推理延迟，展示了轻量化TCN可作为实时情感系统的可行替代方案；同时结合三种可解释性方法（区域专属训练、遮蔽、梯度敏感度）探究身体不同部位对情感分类的贡献。

**🔧 技术方法**

采用轻量化时序卷积网络（TCN）架构、图卷积时空网络（G-TSG）、零基遮蔽、输入梯度可视化等技术。

**📊 数据集**

使用DIEM-A（多文化亚洲表演者的身体运动情感数据库）进行评估。

**📈 对比分析**

在leave‑performer‑out交叉验证下，TCN-Base仅比G-TSG低1.58%准确率、1.25%宏F1，但参数减少79.18%、MACs减少98.03%、推理延迟下降约12.5倍，表明性能差距不显著且效率大幅提升。

**⚠️ 局限性**

仅与单一图卷积基线比较；未考虑跨文化、不同情景或更广泛的轻量化骨架模型；遮蔽实验产生分布偏移，导致解释结果不具可比性。

---

## 222. New Complexity-Theoretic Frontiers of Tractability for Neural Network Training

**arXiv ID:** 2607.20811 | [PDF](https://arxiv.org/pdf/2607.20811v1)

**作者:** Cornelius Brand `[一作]` (Vienna University of Technology), Mathis Rocton `[通讯]` (Vienna University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文研究了ReLU与线性激活网络的最优训练问题，提出了新的上界算法并扩展了可多项式时间训练的网络类；

**💡 创新点**

创新点在于证明所有隐藏节点出度为1的ReLU网络可在多项式时间内训练，同时通过“吹大”技术将任意常数规模ReLU网络转化为可训练架构，并为线性激活网络提供了通用的“untangling”分解方法；

**🔧 技术方法**

主要技术包括：权重离散化（将深度边权取{-1,1}）、超平面划分枚举、二次优化（椭圆法）、图结构变换（吹大）以及树宽下的MSOL+Courcelle定理；

**📊 数据集**

该研究为理论计算复杂性分析，未使用具体实验数据集；

**📈 对比分析**

通过算法复杂度分析比较，得到对ReLU网络的运行时间为 |D|^{n·w·2^w}，对常数规模网络实现多项式时间；

**⚠️ 局限性**

局限性在于对隐藏节点出度>1的网络仍不可多项式训练，且所提出的吹大方法在实践中可能导致网络尺寸指数级增长；

---

## 223. Imprecise Probabilistic Programming, Precisely: Credal Sets via Graded Monads, BDDs, and Semiring-Parametric Inference (Functional Pearl)

**arXiv ID:** 2607.20801 | [PDF](https://arxiv.org/pdf/2607.20801v1)

**作者:** Jack Liell-Cock `[一作]` (University of Oxford), Sam Staton `[通讯]` (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了一个嵌入式 Haskell DSL（Imp），支持同时处理精确概率与 Knightian 不确定性，并通过将程序编译为 BDD 并使用加权模型计数（WMC）在同一 BDD 上完成三种推理方式（精确枚举、梯度优化、区间近似）。

**💡 创新点**

创新点在于：①引入按名称索引的分级单子（graded monad）在类型层面追踪 Knightian 源，强制独立不确定性不产生交叉；②证明并实现精确概率、梯度求解与区间推理都可通过同一 BDD+WMC 体系完成；③利用 Haskell 的类型系统和 QualifiedDo 语法，使 DSL 既直观又安全。

**🔧 技术方法**

使用技术包括 Haskell 的 GADT、类型族、QualifiedDo、类型级列表；分级单子与 Merge/TagAll 类型族实现名称合并；Binary Decision Diagram（BDD）以及补边压缩；加权模型计数（WMC）泛化为 Semiring；双数（DualS）实现自动微分；区间（IntervalS）实现保守近似；以及梯度下降优化器。

**📊 数据集**

主要使用合成案例：Ellsberg urn（多颜色版本）、机器人导航（IMDP）、Monty Hall 变体、两子问题，未使用真实世界数据集。

**📈 对比分析**

对 Ellsberg urn 的实验比较：精确枚举随着 Knightian 变量数量呈指数增长；区间 WMC 与梯度推理均为多项式复杂度；梯度下降在 9 个变量左右保持精确；区间近似速度最快但结果更宽松。实验显示同一 BDD 可以通过不同 Semiring 实例完成三种推理，性能差异主要由变量排序和阈值数量决定。

**⚠️ 局限性**

局限性包括：①枚举法在 Knightian 变量数目较多时不可行；②区间近似因依赖问题和缺失概率总和约束导致不够紧凑；③目前仅支持离散模型，无法直接处理连续分布；④BDD 变量排序对性能影响大；⑤未实现多项式 polyhedron Semiring，导致对大规模 credal 集合的求解仍受限。

---

## 224. Ocular Verification for Virtual Reality

**arXiv ID:** 2607.20790 | [PDF](https://arxiv.org/pdf/2607.20790v1)

**作者:** Husanpreet Singh `[一作]` (University of Wyoming), Sudipta Banerjee `[通讯]` (University of Wyoming)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估ISO/IEC 29794‑6虹膜质量指标在VRBiom数据集上的表现，研发三种图像修复（偏轴校正、反射去除、非均匀照明校正），并在单模（虹膜、眼周）与双模（虹膜+眼周）级联融合下进行眼部身份验证

**💡 创新点**

首次系统性检验VR环境下传统虹膜质量度量的适用性，提出针对VR捕获的图像修复流程，并通过多模融合显著提升验证性能

**🔧 技术方法**

偏轴校正采用DINOv3+H8Net生成投影矩阵；反射去除使用UnReflect；照明校正采用UNIR‑Net；虹膜识别基于ArcIris；眼周识别采用MobileFaceNet；最终采用加权分数级融合

**📊 数据集**

VRBiom（Meta Quest Pro 近红外眼部视频数据，25名受试者，约23K帧）

**📈 对比分析**

与单模虹膜、单模眼周对比：单模虹膜EER 0.44，眼周EER 0.34；双模融合（25/75权重）EER 0.33，AUC 0.75，d' 0.99，较单模虹膜降低约11% EER，提升约16% AUC

**⚠️ 局限性**

部分ISO度量（如margin adequacy、concentricity）在VR图像上失效；图像修复网络有时会平滑或破坏细节纹理，导致虹膜识别下降；目前仅在VRBiom验证，需进一步推广至其它VR眼部数据集

---

## 225. HARP: The Human--AI Research Platform

**arXiv ID:** 2607.20773 | [PDF](https://arxiv.org/pdf/2607.20773v1)

**作者:** Zeshu Zhu `[一作]` (BTPX Innovation Lab), Emily Eiben `[通讯]` (BTPX User Assistance)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 Human–AI Research Platform (HARP)，让研究者在可控的实验情境下与实时可配置的 LLM 进行交互，并以一个关于响应长度与技术专属性的研究为例进行演示。

**💡 创新点**

创新之处在于：①实现了对活 LLM 行为的实验控制，同时保持生态有效性；②记录了交互行为（如键入时长、暂停、删除）等细粒度指标；③在交互中实时触发调查问卷，提供情境反馈。

**🔧 技术方法**

利用可配置的大型语言模型（如 GPT），平台通过 API 调用、实时响应监测、对话日志记录、实验条件配置等技术实现实验控制与数据采集。

**📊 数据集**

示例研究使用了一份产品需求文档（PRD）作为任务材料，未使用公开大规模数据集。

**📈 对比分析**

通过对不同响应长度（短、中、长）和技术语言（面向开发者或面向普通用户）的组合进行 A/B 对比，收集了记忆与理解测试结果以及自评认知负荷；实验显示长度与技术性对用户的保留效果有显著影响，但未给出具体数值。

**⚠️ 局限性**

局限性包括：键入行为的个体差异导致 keystroke 指标不稳定；目前仅支持文本交互，无法捕捉语音、表情等多模态信号；需进一步验证在更大规模实验中的可复现性。

---

## 226. The Human-AI Substitution Principle: When will you be replaced by AI in your organization?

**arXiv ID:** 2607.20781 | [PDF](https://arxiv.org/pdf/2607.20781v1)

**作者:** Bonny Banerjee `[一作]` (Independent Scholar and Consultant), Shreya Singh `[通讯]` (Chennai Mathematical Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并分析了“人机任务分配（HAT）”模型，系统性研究组织层级结构与风险调整后成本如何决定何时、何地以及为何人类员工被AI取代；

**💡 创新点**

将人类技能获取成本与AI能力扩展成本的“成本不对称”假设正式嵌入模型，构建了基于风险调整的替代原则、层级深度缩减、混合组织与中层易受自动化影响的条件等一整套理论框架；

**🔧 技术方法**

使用线性成本与风险加权的数学优化模型，结合层级跨度、AI固定/可变成本拆分与三维风险分解（可靠性、合规、声誉）实现对人机替代决策的闭式分析；

**📊 数据集**

无实证数据集；该工作为理论模型，未进行数据驱动实验；

**📈 对比分析**

本研究不涉及实验比较，性能评估以理论推导与比较分析为主；

**⚠️ 局限性**

局限在于模型假设（如成本不对称、风险权重固定、层级结构可完全重构）与现实组织约束可能不完全吻合，且未考虑动态学习、协作成本和真实数据验证。

---

## 227. Flint: A Semantics-Driven Data Visualization Intermediate Language

**arXiv ID:** 2607.20775 | [PDF](https://arxiv.org/pdf/2607.20775v1)

**作者:** Yunhai Wang `[一作]` (Renmin University of China), Chenglong Wang `[通讯]` (Microsoft)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于语义类型的可视化中间语言Flint，配合编译器实现从高层语义描述自动生成不同可视化后端（Vega‑Lite、ECharts、Chart.js）的图表；通过案例研究和LLM实验验证其简洁性、鲁棒性和可移植性。

**💡 创新点**

创新点在于把数据语义抽象为一层可操作的类型体系，利用编译器将语义映射到低层图表参数，实现语言层面与后端实现解耦；并通过动态模板和全局优化实现高质量图表的自动配置。

**🔧 技术方法**

使用的技术包括：中间语言语义类型定义、编译器前端解析、优化阶段（局部/全局布局优化）、后端代码生成（多后端模板），以及LLM（GPT‑4/5）生成语义描述的实验。

**📊 数据集**

使用的数据集主要有：用户参与度模拟数据（Period、Week、AgeGroup、Users），2025年全球游戏市场合成数据，以及TidyTuesday 2025公开数据集用于LLM评估。

**📈 对比分析**

方法上与基线DirectVL（直接输出Vega‑Lite）对比，使用VisEval视觉评估器给出0–5分的Relevance、Chart Errors、Clarity、Design Quality四项评分；实验结果显示Flint在GPT‑4/5模型下的胜率均超过30%，在LLM能力较弱时优势更明显。

**⚠️ 局限性**

局限性包括：目前仅支持静态图表，不支持交互式功能；语义类型与字段的单一映射不适用于多义字段；对交互与高阶语义支持有限；后端扩展需手动实现模板。

---

## 228. Emergent Compositional Skills in Mixture-of-Experts VLAs

**arXiv ID:** 2607.20771 | [PDF](https://arxiv.org/pdf/2607.20771v1)

**作者:** Shlok Shah `[一作]` (Princeton University), Ishaan Javali `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Vision‑Language‑Action模型中加入混合专家（MoE）动作头，直接从专家演示学习可重用、可解释的技能。

**💡 创新点**

不使用预先定义的层次结构或技能库，而是通过LoRA微调与单次路由实现技能的自发分解与组合，从数据中产生可重用的模组化技能。

**🔧 技术方法**

采用LoRA Mixture‑of‑Experts架构、流匹配（flow‑matching）行为克隆损失、负载平衡正则化，并在预训练的π₀和SmolVLA VLA骨干上实现路由器。

**📊 数据集**

使用LIBERO‑10机器人操纵任务数据集进行训练与评估，并在SmolVLA上进一步测试。

**📈 对比分析**

将MoE策略与同一预训练模型的稠密基线进行对比，二者在任务完成率上相当，但MoE能显著实现专家的专门化与可复用性。

**⚠️ 局限性**

局限性包括部分专家仍与特定任务相关、专家与技能的映射不完全精确、对未知情境的泛化能力有限。

---

## 229. Robust Asynchronous Q-Learning under Reward and State Corruption via Batching

**arXiv ID:** 2607.20822 | [PDF](https://arxiv.org/pdf/2607.20822v1)

**作者:** Sreejeet Maity `[一作]` (North Carolina State University), Aritra Mitra `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对奖励与状态观测同时受对手污染的强化学习场景的鲁棒 Q‑学习算法，采用 epoch‑based 分批策略与鲁棒均值估计来逼近 Bellman 最优算子；

**💡 创新点**

创新点在于将在线批量化与鲁棒截断均值估计相结合，解决了联合奖励‑状态污染问题，并给出了最优的偏差下界；

**🔧 技术方法**

主要技术包括 Huber 污染模型、鲁棒截断均值估计、批量化（epoch）更新、贝塞尔不等式以及对 Q‑值迭代的梯度截断；

**📊 数据集**

实验使用经典 Grid‑World 环境（100 状态、40 动作、γ=0.5）进行验证；

**📈 对比分析**

与标准 Q‑学习对比，鲁棒 Q‑学习在不同污染水平下均能收敛至与最优值相近的近似解，误差比普通 Q‑学习显著更小；

**⚠️ 局限性**

局限性包括仅适用于表格 MDP、需要存储每个 epoch 的完整样本集合、依赖最小访问概率的先验信息以及对 batch 长度的敏感性。

---

## 230. Spectrogram-Based Joint Detection, Localization, and Classification of Events in Continuously Recorded IBR Waveforms

**arXiv ID:** 2607.20817 | [PDF](https://arxiv.org/pdf/2607.20817v1)

**作者:** Shivanshu Tripathi `[一作]` (University of California), Hamed Mohsenian-Rad `[通讯]` (University of California)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于光谱图的时序事件检测、定位与分类框架，用于对连续波形测量中的电力系统事件进行识别。

**💡 创新点**

创新点在于将多通道波形转换为时间‑频率的光谱图，并将检测、定位与分类统一为单阶段时序目标检测任务，采用U‑Net实现。

**🔧 技术方法**

采用STFT得到对数功率谱作为输入，利用二维U‑Net进行端到端的检测、定位和分类，并使用交叉熵和IoU等指标进行训练与评估。

**📊 数据集**

使用了六天的单相干扰数据（4类）和200条三相故障记录（4类），包含不同背景噪声的训练、验证与测试集。

**📈 对比分析**

与基线的原始时序U‑Net‑1D比较，光谱图方法在检测、定位和分类上显著提升；单相任务mAP提升至1.0，三相任务mAP提升至0.858，定位精度和召回率均更高。

**⚠️ 局限性**

局限性包括事件类型识别准确率仍偏低，对STFT重叠比例的依赖较大，且未对实时流式部署进行评估。

---

## 231. Can an AI System Be Creative? A Critical Perspective from Art and Engineering

**arXiv ID:** 2607.20796 | [PDF](https://arxiv.org/pdf/2607.20796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 232. The Geometry of Personality: Activation Steering with Jungian Cognitive Functions

**arXiv ID:** 2607.20803 | [PDF](https://arxiv.org/pdf/2607.20803v1)

**作者:** Liu Zai `[一作]` (University Of Glasgow), Joemon M. Jose `[通讯]` (University Of Glasgow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于激活向量的激活驱动方法，用来控制大型语言模型（LLM）在八种荣格认知功能上的人格表现。

**💡 创新点**

创新点在于把人格从传统的特质模型（如大五）转变为动态的认知过程模型；同时首次构建了可用于激活驱动的荣格认知功能评估协议和大规模角色扮演叙事数据集。

**🔧 技术方法**

采用了差分均值提取激活向量、层级效果扫描、主成分分析（PCA）以及多维驱动的线性回归检验等技术，对Llama‑3.1‑8B进行激活注入和评估。

**📊 数据集**

数据集为由2100+虚拟角色自述构成的SWCPQ扩展版（包含荣格功能得分与标签），并生成了专门用于评估的“SWCPQ‑LLM”数据集。

**📈 对比分析**

通过对每一层、每个功能的λ（注入强度）扫描，得到从最低到最高的功能得分差值，显示所有八个功能在第7-12层之间可实现显著且单调的控制，效果最优的层为7–12；相较于传统的特质驱动，显著提升了多维人格控制的可解释性和效能。

**⚠️ 局限性**

局限性包括：仅在Llama‑3.1‑8B上验证；需要大量标注数据和多轮对话来提取激活向量；多维激活方向非线性组合，表明对更复杂人格组合的控制仍存在挑战。

---

## 233. Talking Ink: A Flow-Based Multi-Molecule Molecular Communication Testbed for Effective Channel Modeling and Detector Benchmarking

**arXiv ID:** 2607.20802 | [PDF](https://arxiv.org/pdf/2607.20802v1)

**作者:** Alexander Wietfeld `[一作]` (Technical University of Munich), Wolfgang Kellerer `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

本文实现了一个基于流动的多分子分子通信实验平台，使用青色、品红、黄色墨水作为可区分的信号分子，配合微泵注射和非侵入式光谱感知，实现实时MUMO‑OOK传输并收集完整的接收波形。

**💡 创新点**

创新点在于：① 通过可重构的墨水注射与光谱分离实现多分子实时并行信道；② 提出有效端到端模型，将有限释放、流动扩散与接收平滑整合，并给出两种紧凑的传播核；③ 设计了MEDD自适应能量差检测器，在不使用通道模型的情况下仅用接收波形即可实现接近MMSE的错误率。

**🔧 技术方法**

采用了微泵驱动的注射、透明管道流动、AS7341八通道光谱传感器以及Beer–Lambert光吸收解混算法，结合离散卷积、AD式扩展衰减模型与Poiseuille盘形核，构建了完整的信道与接收模型。

**📊 数据集**

采用的实验数据集包括在9.5 cm、23.5 cm和100 cm三距离下的单脉冲测量，用于拟合端到端冲激响应；以及在7.4 cm和23.5 cm两距离、不同符号周期与注射持续时间下的约6600比特连续MUMO‑OOK payload 测试数据。

**📈 对比分析**

通过对比多类检测器（基于能量、差分、CIR辅助、MMSE、MLSD等）在测量数据上的BER，MEDD在主设置下达成0.76 %（与MMSE的0.71 %差距仅两误）并在部分设置实现零误码，证明了其在不依赖通道模型时的优越性能。

**⚠️ 局限性**

限制主要包括：端到端模型对尾部（late‑tail）响应的匹配不足、注射几何与压力平衡导致的色彩差异、光谱传感的漂移和波形噪声，以及实验仅基于墨水，缺乏与真实生物分子环境的直接可移植性。

---

## 234. Socially Consistent Multi-Robot Navigation Using Decoupled Planning and Trajectory Coordination

**arXiv ID:** 2607.20772 | [PDF](https://arxiv.org/pdf/2607.20772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 235. External Clustering Validation by the Homogeneity-Parsimony Trade-off

**arXiv ID:** 2607.20799 | [PDF](https://arxiv.org/pdf/2607.20799v1)

**作者:** Andreas Tiffeau-Mayer `[一作]` `[通讯]` (University College London), Andreas Tiffeau-Mayer (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于信息瓶颈原理的外部聚类验证框架，定义了归一化的同质性（homogeneity）和简约性（parsimony）两项评分，形成可视化的Pareto最优曲线；

**💡 创新点**

创新点在于：① 通过对条件熵做固定归一化，使得两项评分在[0,1]区间内可比且单调；② 通过泛化熵（min‑entropy、碰撞熵）统一推导集匹配（purity、inverse purity）和对偶分类（specificity、sensitivity）指标；③ 将两目标视为Pareto优化，提供 Q‑measure 等调和平均量化方法；

**🔧 技术方法**

使用信息理论工具（Shannon 熵、Rényi 熵、Tsallis 碰撞熵）、信息瓶颈变体、数值实验、可视化曲线、AUC 统计以及对 AMI 的对比；

**📊 数据集**

实验集包括：1）人工合成三类二维数据，用于特征选择示例；2）MNIST 图像子集（2000 张），使用 PCA 50 维作为输入，对 k‑means 与凝聚聚类进行评估；

**📈 对比分析**

对比方法包括：k‑means、凝聚聚类（Ward 链接），评估指标为 homogeneity、parsimony、AUC、AMI。结果显示：凝聚聚类在中等 parsimony 区间实现更高 homogeneity，且 AUC 更大；k‑means 在某些参数下 homogeneity 更高但 parsimony 较低；AMI 与 homogeneity‑parsimony 结果一致但不够细粒度；

**⚠️ 局限性**

局限性：仅适用于硬聚类，未考虑软/重叠聚类；归一化参照依赖数据规模，导致跨数据集比较需注意；Q‑measure 的权重仍需经验选择；缺乏针对不同应用场景的明确阈值指导；

---

## 236. Memoir: Should a Model Write to Its Memory While It Thinks?

**arXiv ID:** 2607.20792 | [PDF](https://arxiv.org/pdf/2607.20792v1)

**作者:** Jaber Jaber `[一作]` (RightNow AI), Osama Jaber `[通讯]` (RightNow AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在记忆写入与推理循环中可写快层的影响，评估了可写推理与只读推理的学习速率差异。

**💡 创新点**

提出四层记忆接口（快、短期、慢、冻结）并将其与可变深度自适应推理及未来潜在能量目标相结合，构建可写推理模型Memoir。

**🔧 技术方法**

线性注意力的快权重写法、可变深度递归（Adaptive Computation Time）、能量目标与未来潜在预测、delta‑rule 更新、层级内存写入控制等技术。

**📊 数据集**

使用64值、16干扰器的程序性关联回忆任务（procedural associative recall with key interference）。

**📈 对比分析**

采用12个种子、240步训练的成对t检验；只读推理在240步时平均提升0.1354点；在960步时两者均达到1.0。

**⚠️ 局限性**

仅在单一任务和有限步长上验证，无法区分写入时机与写入量；未评估大规模任务、自然语言或长期学习；硬件特定的工程测量不具备普适性。

---

## 237. Toward Mechanistic Interpretability of an AI Foundation Model Fine-Tuned for Atmospheric Chemistry

**arXiv ID:** 2607.20778 | [PDF](https://arxiv.org/pdf/2607.20778v1)

**作者:** Jason Y. Hu `[一作]` (Stanford University), Makoto M. Kelp `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对微软 Aurora 基础模型在大气化学预测中的内部机制进行机械可解释性分析

**💡 创新点**

首次将对化学过程的物理约束检验与内部特征的稀疏自动编码器（SAE）耦合，用以探查 AI 气候模型的化学学习能力

**🔧 技术方法**

光谱扰动实验、NOx 施加实验、野火残留实验、PCA、稀疏自动编码器（AuroraScope）与特征消融（zero‑ablation）

**📊 数据集**

CAMS 重新分析数据（2003‑2022）、ERA5、CMIP6、GFS 等混合气候数据，Aurora 预训练与微调数据

**📈 对比分析**

与 CAMS 参考模拟比较：在 12‑小时预报中 Aurora 在 NOx 影响下可产生臭氧抑制，精度与 CAMS 相当或略优，但存在负浓度与非物理组合，野火烟雾的保持性不佳

**⚠️ 局限性**

缺乏显式化学约束导致负浓度与不一致的化学关系；内部表示主要受预训练气象偏置主导，化学信息难以解耦；在长时间滚动中误差随时间放大，缺乏对非平稳情境的鲁棒性

---

## 238. Memory-Computation Tradeoffs in Semi Amortized Parametric Optimization

**arXiv ID:** 2607.20769 | [PDF](https://arxiv.org/pdf/2607.20769v1)

**作者:** Shijie Pan `[一作]` (Johns Hopkins University), Enrique Mallada `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究半半优化框架，探讨在固定在线迭代预算K下，需要多少离线内存才能保证所有参数下的ε-最优解。

**💡 创新点**

提供了基于参数空间维度、收敛率和解映射敏感度的通用上界与下界，并揭示强凸和β增长两种情形下的记忆-计算-精度三者之间的相互替代关系。

**🔧 技术方法**

利用投影梯度下降、覆盖/打包论证、非参数最近邻预测器以及元证明框架将收敛率和解敏感度两大结构性输入统一起来。

**📊 数据集**

在实验中使用了参数化的Tikhonov正则化回归（可闭式求解）以及可凸化的两层神经网络训练两类数据集。

**📈 对比分析**

与单纯PGD做对比，实验显示：在强凸（β=2）情形下，增加记忆对迭代次数提升有限；在β>2情形下，增大记忆显著降低所需迭代次数，验证了理论的多项式缩减预测。

**⚠️ 局限性**

限制包括仅考虑凸问题、假设精确的离线解、参数空间维度高但未利用低维结构，以及未探讨学习型预测器的表达式与记忆成本的折中。

---

## 239. Probabilistic Residual Learning for Online Recommendations

**arXiv ID:** 2607.20863 | [PDF](https://arxiv.org/pdf/2607.20863v1)

**作者:** Wenyuan Wang `[一作]` (Rutgers University), Hao Wang `[通讯]` (Rutgers University and UIUC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Probabilistic Residual Learning（PRL）框架，通过在已有深度学习推荐器基础上学习残差并进行因果去偏，提升跨域冷启动推荐效果。

**💡 创新点**

创新点在于将用户分簇、领域因果混杂器以及残差学习统一到层次贝叶斯因果模型中，并通过do-算子实现无须主动干预即可进行去偏。

**🔧 技术方法**

采用层次贝叶斯推断、变分推理、因果图模型、概率聚类、深度编码器（如CDL、DLRM、NCF等）及梯度上升训练。

**📊 数据集**

主要使用两大数据集：XMRec（18个国家、52.5M交互）与MovieLens（基于年龄的跨域划分），均包含文本特征与领域因子。

**📈 对比分析**

与五种基准推荐器（CDL、DLRM、PerK、NCF、LightGCN）在Recall@20、Precision@20、F1@20、MAP@20、NDCG@20等指标上对比，PRL在所有基准上均有显著提升，尤其在跨域冷启动场景。

**⚠️ 局限性**

局限性包括：对领域因子依赖性强，需预先定义或学习领域嵌入；模型推断复杂度较高；在弱因果信号的数据集（如MovieLens）提升有限。

---

## 240. Code Monitor Red Teaming for Public-Test-Passing Code

**arXiv ID:** 2607.20852 | [PDF](https://arxiv.org/pdf/2607.20852v1)

**作者:** Junchi Liao `[一作]` (University of Electronic Science and Technology of China), Fuji Ren `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在代码生成后，提出并实现了一种监测-红队协议（Monitor-Red-Teaming），通过公共测试通过后再让LLM验证器在固定的公共视图（M1）下对代码进行评分，并构建了一个跨函数、数据科学和工作流任务的基准测试集，对生成器压力与验证器能力进行系统评估。

**💡 创新点**

创新点在于：① 将生成器与验证器的情报边界固定为公共视图，拆分残留bug构造与验证器排名两大问题；② 通过弱到强验证器的梯度实验揭示验证器能力与证据边界的相互作用；③ 引入 inferability 审计，区分可推理与证据受限的错误；④ 将 adversarial G4-adv 作为鲁棒性压力测试。

**🔧 技术方法**

使用技术包括：LLM（如 Qwen‑7B、Mistral‑7B、GPT‑5.4‑mini、GLM‑5.1）进行生成与验证；Hybrid 复合式推理提示；静态检查、生成测试监控；评估指标 AUROC、AUPRC、FNR@5%FPR、NA率；对比实验设计和 bootstrap 置信区间。

**📊 数据集**

数据集包含：① 71,000 条候选代码，分布在三大任务域——函数级（BigCodeBench‑Hard）、数据科学（pandas、NumPy、scikit‑learn、SciPy 等）、工作流/工件（schema、聚合、配置、报告）。

**📈 对比分析**

比较方法：在固定公共视图下，分别评估不同生成器压力（G1–G4‑adv）和验证器架构（Direct、Hybrid）以及弱到强模型；使用 AUROC、FNR@5%FPR 作为核心指标。结果显示：弱验证器在 5%FPR 下遗漏 80% 以上残留 bug；强验证器可将漏判率降至约 54%；静态检查几乎随机；生成测试监控可提升 20–30% 排序性能，但仍受 G4‑adv 影响。

**⚠️ 局限性**

局限性包括：① 隐藏测试仅覆盖冻结的测试集，无法完整覆盖语义正确性；② 构造的 data‑science 与 workflow split 仍未覆盖所有库与交互式开发场景；③ G4‑adv 为对抗性压力测试，非真实部署误差率；④ 未评估学习型验证器、完整静态分析、交互式评审及成本/延迟折衷；⑤ 部分错误归因于公共视图证据边界的限制。

---

## 241. Explainable graph attention network for stress recognition (StressGAT) via differential action units

**arXiv ID:** 2607.20819 | [PDF](https://arxiv.org/pdf/2607.20819v1)

**作者:** Thomas Kassiotis `[一作]` (Hellenic Mediterranean University), Giorgos Giannakakis `[通讯]` (Hellenic Mediterranean University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种可解释的图注意网络StressGAT，用于基于面部表情的个性化压力识别。

**💡 创新点**

创新点在于结合差分动作单元归一化、因果时间图结构以及多实例学习聚焦，实现了可解释且个性化的压力检测。

**🔧 技术方法**

采用了图注意网络GATv2、动作单元(AU)特征、MIL注意聚合与批量归一化等技术。

**📊 数据集**

使用了58名参与者的面部视频数据集，包括多阶段压力诱导实验，数据可在GitHub获取。

**📈 对比分析**

与线性SVM、LSTM/GRU、1D-CNN和Transformer等基线进行对比，StressGAT在留一人交叉验证下取得88.62%准确率，优于所有基线。

**⚠️ 局限性**

局限在于仅基于视觉信号，对抑制型压力人群效果不佳，且缺乏多模态融合与真实环境验证。

---

## 242. Profiling Lightweight Large Language Models

**arXiv ID:** 2607.20806 | [PDF](https://arxiv.org/pdf/2607.20806v1)

**作者:** Tomohiro Harada `[一作]` (Saitama University), Gabriel Luque `[通讯]` (University of Malaga)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了轻量级LLM在本地资源受限环境下的精准度-时间-内存-能耗四维评估（PTME）框架，并在桌面模拟的多种资源包下对六种模型进行基准实验

**💡 创新点**

①将精度、执行时间、峰值内存和能耗统一测量为PTME指标；②发现静态代理指标（参数量、FLOPs、加载内存）对成本预测良好但对精度无效；③利用Pareto分析揭示非支配配置，指出单一指标评估的局限；④分析资源限制对PTME的敏感性，证明大模型更易受限

**🔧 技术方法**

使用Intel RAPL硬件能耗计数、Docker+CPU频率/核心绑定+内存限制来逼近边缘设备；采用Ollama推理后端、量化模型、固定确定性解码；统计检验（Cochran Q、McNemar、Friedman、Nemenyi）和三维可视化来分析PTME

**📊 数据集**

HumanEval（代码生成）、GSM8K（数学推理）、MMLU-Pro（多任务语言理解）三大基准，分别涵盖不同任务难度和输出结构

**📈 对比分析**

在E1–E5四种资源包下测得六个模型（TinyLlama 1.1B、Qwen2.5‑1.5B、Gemma‑2B、Phi‑2‑2.7B、Phi‑3‑mini、Mistral‑7B）的PTME；结果表明小模型成本低但精度差，大模型精度最高但成本极高；Qwen2.5在精度与成本之间实现良好折中；资源限制加剧时间成本，能耗下降幅度不如时间，且大模型更易受限

**⚠️ 局限性**

实验仅在桌面CPU上通过软件限制模拟边缘设备，未在真实单板或移动硬件上验证；能耗仅测CPU包，未包含DRAM、GPU或其他系统组件；使用单一后端与确定性解码，缺乏对GPU加速或其他推理框架的评估；样本集有限，未覆盖更大模型或不同任务，因而结果不一定可直接推广到更广范围

---

## 243. New Criteria and Constructions for Self-Orthogonal Codes

**arXiv ID:** 2607.20805 | [PDF](https://arxiv.org/pdf/2607.20805v1)

**作者:** Peng Wang `[一作]` (Chang'an University), Ziling Heng `[通讯]` (Chang'an University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了多种线性码自正交性判定标准，并利用这些标准构造了多类新的自正交码及其最小码

**💡 创新点**

创新点在于给出了基于定义集的自正交判定条件，并将其应用于已知码和新构造的自正交码，进一步拓展了自正交码与量子码、最小码的联动框架

**🔧 技术方法**

采用了加法字符分析、差分集与分支的组合、Pless幂矩、向量双鞅函数等编码理论与数论技术

**📊 数据集**

使用的“数据集”主要是组合数学构造的定义集，如部分差分集、向量双鞅函数产生的集合、平面子集等，未涉及外部标注数据

**📈 对比分析**

通过与已知码参数（如最优码、量子码界）以及理论极限（球面包装、霍默、Singleton、量子界等）进行比对，许多新码达到或接近这些界限，量子码在码长、维度和距离上优于同类已有构造

**⚠️ 局限性**

局限性在于判定条件仅适用于特定结构的码（如基于定义集、平面子集等），对更一般或随机线性码的自正交性仍缺乏通用方法

---

## 244. Classical Acceptance Is Not Hybrid Authentication: Measuring X.509 Verifier Semantics in Post-Quantum Migration

**arXiv ID:** 2607.20800 | [PDF](https://arxiv.org/pdf/2607.20800v1)

**作者:** Taesung Kim `[一作]` (Electronics and Telecommunications Research Institute), Yousung Kang `[通讯]` (Electronics and Telecommunications Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八个主流 X.509 验证堆栈的默认路径和可选模式进行实验，评估它们在混合（后量子/传统）证书验证时是否能正确识别并强制后量子证据，从而避免将传统接受误读为混合身份。

**💡 创新点**

提出基于规范的四层验证模型、可参数化的混合验证契约以及生命周期脱同步实验，首次系统揭示非关键扩展导致的“silent promotion”缺口，并给出标准、库和接口需要补齐的具体要点。

**🔧 技术方法**

采用差分测试框架、规范派生的参考验证程序、可确定性的证书合成器以及生命周期状态模拟，结合多栈对比分析实现行为。

**📊 数据集**

构建了包含六种证书方案（两条纯后量子、四种混合设计）的合成语料库，并通过 SHA-256 摘要保证可重复性。

**📈 对比分析**

对八个验证堆栈（七个独立实现）在默认路径和可选模式下的五种行为结果进行对比，发现大多数堆栈默认只做传统路径验证，未使后量子证据决定结果；性能并未作为评估指标，重点在行为差异上。

**⚠️ 局限性**

样本并非完整覆盖、部分生命周期状态仅模型化、标准与实现的解释差异导致交互性问题、未评估链级混合或多层路径对验证的影响等限制。

---

## 245. Robostral Navigate

**arXiv ID:** 2607.20785 | [PDF](https://arxiv.org/pdf/2607.20785v1)

**作者:** Arjun Majumdar `[一作]`, Thomas Chabal `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Robostral Navigate，一款仅使用单一RGB相机的8B视觉‑语言导航模型，能够通过图像空间的指点方式预测航点，并在真实机器人上实现跨平台部署。

**💡 创新点**

核心创新在于：①仅依赖最通用的RGB摄像头，避免深度/多摄像头依赖；②将航点以像素坐标方式输出，天然抗相机内参与尺度变化；③前缀缓存训练（prefix‑caching）将完整轨迹压缩为单一序列，令训练标记数减少22×；④在此基础上采用CISPO强化学习提升探索与回溯能力；⑤通过随机化机器人尺寸与相机姿态实现跨机器人泛化。

**🔧 技术方法**

使用技术包括：大规模8B视觉‑语言模型（用于指点与指令理解）、121M参数扩散策略（生成低层轨迹）、前缀树注意力掩码（高效序列训练）、在线RL（CISPO）、多机器人随机化与仿真环境（Habitat + 物理渲染）。

**📊 数据集**

训练数据来源于约2.4 M条从350k仿真场景生成的轨迹（涵盖办公、住宅、商业、户外等多样布局）；评估数据为R2R‑CE和RxR‑CE验证未见集（语义指令 + RGB观测）。

**📈 对比分析**

与以往单相机方法相比，Robostral Navigate在R2R‑CE上成功率提升10.5点（从66.9%到77.4%），在RxR‑CE上提升1.7点（73.4%→75.1%）；同时在采用深度/多摄像头的系统上也以5.3点与7.6点的优势获胜，整体SPL与导航误差也表现最佳。

**⚠️ 局限性**

局限性包括：①训练高度依赖仿真数据，真实世界环境的迁移尚未全面验证；②前缀缓存训练与RL实现需要大量GPU与复杂的系统调度；③模型仅使用单摄像头，仍可能受低光或动态遮挡影响，且对极端尺度变化的鲁棒性尚需进一步验证。

---

## 246. Unsupervised Metal Artifact Reduction in Dental CBCT using Fine-tuned Cycle-Consistent Adversarial Networks

**arXiv ID:** 2607.20977 | [PDF](https://arxiv.org/pdf/2607.20977v1)

**作者:** G. L. T. Chamika `[一作]` (University of Peradeniya), Ruwan D. Jayasinghe `[通讯]` (University of Peradeniya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一种无监督的 CycleGAN 框架，用于从牙科 CBCT 图像中去除金属伪影，同时保留解剖细节。

**💡 创新点**

创新点在于利用无配对数据训练的 CycleGAN，结合 U‑Net 生成器与 PatchGAN 判别器，并通过精细调节循环一致性与身份损失，显著提升伪影抑制效果与结构保留。

**🔧 技术方法**

技术手段包括 CycleGAN、U‑Net 生成器、PatchGAN 判别器、无监督对抗训练、BRISQUE/FID/SSIM 等无参考评估指标，且推理时间仅 3.03 ms。

**📊 数据集**

使用公开的 ToothFairy 数据集，共约 4000 张 2D CBCT 切片（约 2000 张含金属，2000 张无金属）。

**📈 对比分析**

与经典 NMAR 及两种监督 UNet 基准对比，BRISQUE 由 16.54 降至 10.82（34.6% 改善），FID 从 207.03 降至 157.04，SSIM 提升至 0.9105，PSNR 为 25.96 dB，且推理速度最快。

**⚠️ 局限性**

局限性包括仅在 2D 切片上工作、缺乏真实无伪影的 ground truth、在极端光子匮乏区可能出现假影像、缺乏跨扫描仪通用性，需临床专家监督验证。

---

## 247. Update the Unseen Only: Minimizing AoI for Collaborative Perception through Online Learning

**arXiv ID:** 2607.20967 | [PDF](https://arxiv.org/pdf/2607.20967v1)

**作者:** Yanan Ma `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对基础设施辅助协作感知（CP）系统的时延感知优先调度框架，通过同时利用车辆本地感知和基站广播更新来最小化信息新鲜度度量AoI。

**💡 创新点**

创新点在于：①首次将车辆的动态感知覆盖范围纳入AoI建模并给出闭式长期AoI表达式；②在未知环境统计且存在状态观察延迟的情况下设计了在线学习加权最大权重调度算法LocMW；③给出了理论下界与最优随机化基准，并证明LocMW的累计超额AoI为子线性。

**🔧 技术方法**

采用的技术包括：INAR(1)移动模型、最小二乘岭回归在线参数估计、预测期望AoI、Lyapunov驱动的最大权重调度与水填策略、以及证明中使用的均值场逼近与凸分析。

**📊 数据集**

实验使用了三组数据集：pNEUMA（大规模车辆轨迹）、FLUID（信号控制交叉口轨迹）以及V2X‑Sim（基于SUMO和CARLA的3D感知数据），从而评估AoI与检测精度。

**📈 对比分析**

与多种基线（最大需求、传统最大权重、随机化策略、下界及无更新）对比，LocMW在所有场景下都实现了至少31.6%的AoI下降，并在3D目标检测中提升了16.3%的mAP，且在观察延迟或车辆密度增大时仍保持显著优势。

**⚠️ 局限性**

局限性包括：①对INAR(1)模型的假设可能不适用于所有实际交通环境；②需要较多的历史观测才能收敛估计，导致在极短延迟或极端突发事件时性能受限；③算法虽然复杂度低，但在极大规模区域划分时仍需进一步优化。

---

## 248. Fast and Efficient Approximate Nearest Neighbor Search for High-Dimensional LLM Embeddings

**arXiv ID:** 2607.20957 | [PDF](https://arxiv.org/pdf/2607.20957v1)

**作者:** Nico Hezel `[一作]` (HTW Berlin), Klaus Jung `[通讯]` (HTW Berlin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对SISAP 2026挑战，本文通过结合EVP量化与DEG图构建、FP16重排序、维度扩展与FLAS缓存友好排序，分别在归一化BGE-M3 1024维向量的kNN图构造和非归一化Llama-3.2-8B 128维向量的MIPS查询上实现近似最近邻搜索的高效与高召回。

**💡 创新点**

创新点在于：①将训练无关的Equi‑Voronoi Polytopes（EVP）量化嵌入DEG图中，显著加速构造并通过异构精度与重排序恢复召回；②使用维度提升（lifting）将MIPS转化为欧氏最近邻，从而消除向量归一化差异；③在构造前采用Fast Linear Assignment Sorting（FLAS）进行一维缓存友好排序，提升查询时的内存局部性。

**🔧 技术方法**

采用技术包括：Equi‑Voronoi Polytopes量化、Dynamic Exploration Graph（DEG）、FP16与EVP混合精度、异构距离度量、维度扩展（L2→L2+1），Fast Linear Assignment Sorting（FLAS）缓存友好布局、SIMD位运算、异构查询重排序。

**📊 数据集**

使用的数据集为：①6.4 M个1024维归一化BGE‑M3嵌入向量（Task 1）；②256 k个128维非归一化Llama‑3.2‑8B嵌入向量（Task 2）。

**📈 对比分析**

对比方法：在小型200 k样本的基准上，EVP+DEG+重排序将构造+查询时间从9.35 s降至5.72 s（≈39 %提升）并保持0.8召回；在大型6.4 M样本上，总耗时297 s（构造52 %，探索34 %）满足召回0.8；Task 2中，基线MIPS查询耗时100 ms，加入FLAS降低到80 ms，维度扩展进一步到33 ms，组合后达到29 ms，召回≥0.8。

**⚠️ 局限性**

限制与不足：①EVP量化会导致精度损失，需额外重排序或FP16转换；②实验仅在AMD EPYC 7F72和Intel Xeon 8581C CPU上验证，未探讨GPU或更大规模数据；③对非归一化向量的维度扩展适用于当前128 d范围，可能在更高维或更大分布漂移时效果下降；④整体框架对硬件缓存策略高度依赖，跨平台迁移需进一步评估。

---

## 249. RECO: Region-Aware Compensation for Extrinsic Perturbations in Roadside 3D Detection

**arXiv ID:** 2607.20947 | [PDF](https://arxiv.org/pdf/2607.20947v1)

**作者:** Junsheng Du `[一作]` (Sun Yat-sen University), Yuhuan Lu `[通讯]` (Macao Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RECO框架，利用区域感知的6-DoF外参补偿和软门限融合，显著提升路侧单目3D检测对摄像头外参漂移的鲁棒性。

**💡 创新点**

创新点包括：①基于可学习距离边界的近/远区间外参补偿；②引入可微软门限平滑两种补偿的投影，保持连续性；③辅以重投影损失监督外参；④在DAIR‑V2X和Rope3D上实现全流程实验。

**🔧 技术方法**

使用技术包括：ResNet+LSSFPN的BEV检测骨干；SE(3) 6-DoF外参预测网络；软门限插值投影；重投影监督损失；BEV特征采样与聚合。

**📊 数据集**

实验数据集：DAIR‑V2X‑I 与 Rope3D 路侧单目3D检测基准。

**📈 对比分析**

与BEVHeight、CoBEV、BEVSpread、HeightFormer、BEVHeight++等基线比较，在yaw和z轴扰动下均取得AP提升，尤其在远距离和持续漂移场景下表现显著优于现有方法。

**⚠️ 局限性**

局限性：仍需依赖训练时的噪声模拟，对极端大幅度偏移或多摄像头协同尚未充分验证；模型仅针对单摄像头场景，未扩展到完整的多摄像头网络。

---

## 250. Ms. Forcing: Efficient Streaming Video Generation with Multi-Scale Patchification and Attention

**arXiv ID:** 2607.20940 | [PDF](https://arxiv.org/pdf/2607.20940v1)

**作者:** Zekun Li `[一作]` (Brown University), Srinath Sridhar `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Ms. Forcing 流媒体视频生成框架，通过多尺度补丁化（MSP）和多尺度自注意力（MSSA）在滚动窗口内动态调整空间粒度，从而显著降低冗余计算，实现更高效的实时视频合成。

**💡 创新点**

创新点在于：① 在固定窗口位置上根据噪声级别预先分配不同粒度的补丁，避免不同噪声状态共享相同细粒度；② 采用多尺度自注意力匹配可见键值密度与查询粒度；③ 设计同噪声级分布匹配蒸馏（H‑DMD），统一训练时合成视频的噪声级，消除训练‑推理差异。

**🔧 技术方法**

使用技术包括：基于 Rolling Forcing 的滚动窗口解码、DiT 结构、LoRA 适配器、KV‑cache、分布匹配蒸馏（DMD）等。

**📊 数据集**

在 VBench（5 秒短视频）和 MovieGen（60 秒长视频）官方评测提示上进行训练和评估。

**📈 对比分析**

与 Rolling Forcing、SkyReels‑V2、CausVid、Self‑Forcing、LongLive 等流媒体视频生成器对比，单卡 H200 GPU 上帧率从 16.35 FPS 提升至 22.84 FPS（提升 39.6%），VBench 质量/语义分数略优，60 秒质量漂移从 2.227 降至 1.700，展现了更快、更稳定的性能。

**⚠️ 局限性**

限制在于仅验证了 1.3B Wan2.1‑T2V‑1.3B 小模型、832×480 分辨率；KV‑cache 更新仍带来一定延迟；尚未在更大模型或更高分辨率下测试。

---

## 251. Controllable and Content-Based Recommendations

**arXiv ID:** 2607.20938 | [PDF](https://arxiv.org/pdf/2607.20938v1)

**作者:** Fırat Öncel `[一作]` (Concordia University), Cem Subakan `[通讯]` (Concordia University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出可控且基于内容的推荐框架 CCBR，利用多模态内容生成可编辑的文本用户档案，并与协同过滤模型对齐。

**💡 创新点**

创新点在于：①通过多模态基础模型把项目内容转成文本总结；②将文本瓶颈与协同过滤嵌入对齐，使用户能通过编辑自然语言或新增未见项目来直接控制推荐；③支持正负偏好表达与多模态干预。

**🔧 技术方法**

使用多模态 LLM（Qwen3‑VL‑32B、VideoLLaMA‑3、MusicFlamingo）、文本 LLM（Qwen3‑30B）、BGE‑M3 文本编码器、InfoNCE 对齐、标签监督以及多种协同过滤骨干（EASE、EDLAE、DAE、Multi‑VAE、MacridVAE 等）。

**📊 数据集**

实验数据集包括 H&M 服装图像、MovieLens‑20M 电影预告、Million Song Dataset 音乐三种多模态数据。

**📈 对比分析**

通过与七种协同过滤骨干和 TEARS 基线在 NDCG@100、Recall 等指标下对比，CCBR 在保持或略低于骨干的准确性的同时，在可控性实验（留一法、加一法干预）中显著优于 TEARS，能够实现更强的推荐方向控制。

**⚠️ 局限性**

局限性：①文本瓶颈压缩导致部分准确性下降；②依赖高成本的预训练多模态 LLM；③实验多为离线，缺乏实时交互评估；④负面偏好支持仅在部分数据集可行。

---

## 252. SciExplore: Evaluating Autonomous Agents from Scientific Navigation to Information Integration

**arXiv ID:** 2607.20926 | [PDF](https://arxiv.org/pdf/2607.20926v1)

**作者:** Yinhao Tang `[一作]` (University of Science and Technology of China), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了SciExplore基准，覆盖四级科学信息检索与推理任务，并对现有LLM与科研代理进行系统评估。

**💡 创新点**

创新点：①将科学信息检索分为实体导航、文献检索、引用匹配、跨源结构化综合四级逐层任务；②采用专家驱动、反向轨迹构造、噪声遮蔽等手段保证任务真实性与答案唯一性；③引入严格的质量控制流程（难度校准、人工交叉验证、答案唯一性检查），避免模型记忆式捷径。

**🔧 技术方法**

技术手段包括：大语言模型与检索工具的融合（如OpenAI GPT-5.1、Gemini-3-Pro等）、专家策划与多阶段验证流程、基于模板的结构化输出评估、以及搜索调用统计与分析。

**📊 数据集**

数据集：SciExplore共103个专家策划任务，覆盖十余学科，分为四类任务（T1–T4）分别聚焦数据库导航、文献检索、引用补全、跨源结构化综合。

**📈 对比分析**

比较方法：对基础LLM、LLM+检索工具、以及深度科研代理（如Tongyi-DeepResearch、Gemini Deep Research、OpenAI Deep Research）进行统一评测。整体表现：基础LLM准确率低于20%，深度科研代理最高整体得分仅为49%，其中结构化综合任务T4的准确率低于20%。

**⚠️ 局限性**

局限性：模型在面对模糊检索、信息抽取、长文本注意力、结构化输出遵循等方面表现不佳；易出现早停、幻觉、信息丢失；搜索覆盖不足导致主键召回率低；整体无法实现可靠的自主科研助手功能。

---

## 253. Representing Entity Importance in AI Knowledge Systems: A Dual-Signal Framework of Audience Evaluation and Structural Authority

**arXiv ID:** 2607.20925 | [PDF](https://arxiv.org/pdf/2607.20925v1)

**作者:** Shen Xu `[一作]` `[通讯]` (Independent Researcher), Shen Xu (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可解释的双信号实体重要性表示方法，并在电影领域用IMDb观众评估与Wikipedia PageRank验证其非冗余性。

**💡 创新点**

认为实体重要性应拆分为观众评估维度和结构权威维度，而非压缩为单一标量，提供了最小可解释的表示框架。

**🔧 技术方法**

使用PageRank计算结构权威度，采用Spearman相关、Top‑K重叠和差异分析评估两维度的冗余程度。

**📊 数据集**

使用IMDb非商业数据集、Wikidata映射以及英Wikipedia链接构建的482部电影图谱。

**📈 对比分析**

通过Spearman相关（ρ=0.2275）、Top‑10重叠仅10%、Top‑100重叠34%等指标，表明两维度差异显著，非冗余。

**⚠️ 局限性**

仅在电影域实验，使用单一PageRank和Wikipedia链接，未考虑其他图度量、不同领域及实时用户行为数据。

---

## 254. OPOD: On-Policy Omni Distillation

**arXiv ID:** 2607.20918 | [PDF](https://arxiv.org/pdf/2607.20918v1)

**作者:** Tong Zhao `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种 On-Policy Omni Distillation (OPD) 框架，利用专门的文本、图像、音频教师将三种模态的专业知识整合进单一的 omni‑modal 大模型，实现统一推理。

**💡 创新点**

创新点包括：①单向 token 约束，仅在教师概率高于学生时才施加；②自适应模态控制，为每种模态独立调节教师约束强度；③教师验证奖励将答案置信度与推理收益分离，提供过程级别的强化学习回报。

**🔧 技术方法**

技术手段为：on‑policy distillation 与 GRPO (离线强化学习微调)、自适应双重控制器、验证奖励（答案置信度 + 推理收益）、参考 KL 正则化、token‑level 教师–学生对比等。

**📊 数据集**

使用的数据集涵盖十二个基准：文本（AIME25/26、HotpotQA、MMLU‑Pro、GPQA）、视觉（MMMU、MathVista、ChartQA、A‑OKVQA）、音频（MMAU、AVQA）以及综合 OmniBench，模型基于 Qwen3‑Omni‑30B‑A3B 及其 7B/3B 变体进行实验。

**📈 对比分析**

与原始模型、GRPO、Native OPD、ExOPD、以及单模态专家教师比较，OPD 在 12 个基准上的平均得分分别为 70.8/51.7/46.2（30B/7B/3B），相较最强对手提升 2.1/1.8/1.7 分；在所有基准上均优于基线，且在 10‑11 项基准上排名第一/第二，且即使包含专家教师，OPD 仍保持领先。

**⚠️ 局限性**

局限性包括：需要额外训练并在训练期间保持三套专门教师的服务；目前仅针对文本、图像、音频三模态，未验证对视频等更复杂模态的扩展；对教师质量高度依赖，若教师表现欠佳会削弱效果；训练成本和算力需求较高。

---

## 255. Traceable Scholarship: Page Anchors and Ariadne's Thread for Humanistic Inquiry in the Age of Generative AI

**arXiv ID:** 2607.20916 | [PDF](https://arxiv.org/pdf/2607.20916v1)

**作者:** Deyu Jing `[一作]` `[通讯]` (Fudan University), Deyu Jing (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了可追溯学术研究框架 Traceable Scholarship 及其 AIH-Infra 三层技术实现。

**💡 创新点**

将可追溯性从理论规范转化为可运行的页面锚点、双页码、引用优先生成、NO_EVIDENCE 等技术组合，并定义四级合规模型。

**🔧 技术方法**

采用 OCR/结构化 Markdown、VLM、TEI/IIIF、检索增强生成 (RAG)、多层 Agent 调度、MCP 任务接口等技术。

**📊 数据集**

以 29 卷 Kant Akademie-Ausgabe 知识库（约 1900 万 token）为案例，使用 BAAI/bge-m3 向量模型、reranker 及多种 OCR 引擎。

**📈 对比分析**

通过与传统 RAG 的对比，记录检索精度、页面锚点完整性和证据分级，展示检索偏差修正与可追溯性提升，性能表现优于单纯 RAG。

**⚠️ 局限性**

仍缺乏跨语言、多源验证和自动评测指标，块化、索引失配等技术瓶颈限制大规模应用。

---

## 256. URF: A Unified Robot Control-Policy Framework for Stable Contact Aware Manipulation

**arXiv ID:** 2607.20912 | [PDF](https://arxiv.org/pdf/2607.20912v1)

**作者:** Jiyou Shin `[一作]` (Sungkyunkwan University), Hyungpil Moon `[通讯]` (Sungkyunkwan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了统一的机器人控制-策略框架（URF），能够同时预测虚拟目标、刚度矩阵和阻抗‑顺应切换比例，并在多模态感知输入下执行鲁棒的接触操纵任务。

**💡 创新点**

其创新点在于将动作预测与低层控制耦合：通过预测切换比例自适应地在阻抗与顺应控制之间切换，从而在接触条件变化时实现更安全、更精确的执行。

**🔧 技术方法**

技术上采用多模态编码器（视觉、关节姿态与力谱）结合条件扩散网络生成控制指令，并利用统一阻抗‑顺应控制器与基于力的切换标签进行训练。

**📊 数据集**

使用了人工直接教学收集的演示数据，包括100条盒子翻转、50条线压任务的多模态记录。

**📈 对比分析**

与DP+force、ACP等基线对比显示，URF在盒子翻转任务中成功率达90%，在线压任务中成功率100%，同时显著降低了失效率、力增长率和力波动。

**⚠️ 局限性**

局限性包括切换比例在所有轴共享、需要手工设定力阈值，且在更复杂的插入等任务中可能需要更细粒度的控制策略。

---

## 257. MAGE-Vein: Multi-Instance Age and Gender Estimation from Finger Vein Images

**arXiv ID:** 2607.20897 | [PDF](https://arxiv.org/pdf/2607.20897v1)

**作者:** Katsuki Tanaka `[一作]` (Tohoku University), Kenta Takahashi `[通讯]` (Hitachi Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多实例多任务学习框架 MAGE-Vein，用三根手指的指静脉图像同时预测年龄和性别。

**💡 创新点**

创新点包括：① 将三根手指的特征进行拼接+平均的混合级联融合，显著抑制局部噪声；② 通过同时训练年龄回归和性别分类任务，网络能够去除性别相关的血管差异，聚焦真实的衰老特征；③ 使用人口平衡的大规模指静脉数据集，突破先前因数据偏差导致的结论失效。

**🔧 技术方法**

采用 DenseNet‑161 作为骨干网络，配合特征级融合、交叉熵+均方误差的多任务损失，使用 AdamW 优化，图像采用填充（padding）预处理，随机水平翻转增强；通过 Grad‑CAM 对模型关注区域进行可视化解释。

**📊 数据集**

使用私有 402 名受试者（10~70 岁、男女各半）构成的平衡指静脉数据集；同时在公开的 MMCBNU_6000 数据集上做 5 折交叉验证检验泛化。

**📈 对比分析**

与基线（均值预测）、Wimmer 等人基于单指的年龄回归以及各类性别分类方法对比，MAGE‑Vein 在测试集上得到 MAE 6.12、相关系数 0.880、CS@5 0.526；在受试者级聚合后 MAE 5.47、相关系数 0.921；在 MMCBNU_6000 上 MAE 4.63、但相关系数仍低，表明数据偏差仍是主要瓶颈。

**⚠️ 局限性**

局限性在于对公开数据集的泛化仍受人口分布偏差限制，低相关系数说明模型可能仍对样本平均值过拟合；此外方法依赖三根手指的完整图像，对单指或缺失指样本的鲁棒性未知。

---

## 258. WhereEdit: Mask-aware Local Latent Editing for One-Step Image Editing

**arXiv ID:** 2607.20883 | [PDF](https://arxiv.org/pdf/2607.20883v1)

**作者:** Ming Hu `[一作]` (Institute of Optics and Precision Mechanics, Chinese Academy of Sciences), Quan Wang `[通讯]` (Institute of Optics and Precision Mechanics, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于一阶文本到图像模型的实时图像编辑框架WhereEdit，自动发现可编辑区域并局部强化语义转换

**💡 创新点**

通过Attention-guided AutoMask自动定位编辑区域，并引入Amplified Conditional Transport（ACT）在目标区域内实现更强、更稳定的语义迁移；实现训练与逆向无关

**🔧 技术方法**

Attention机制、联合噪声采样、可微分的局部模糊掩码、条件吸引器（ACT）以及单步Diffusion推理

**📊 数据集**

PIE-Bench（700例，512×512）和公开的SD-Turbo模型

**📈 对比分析**

与多步、少步及一阶编辑方法（ChordEdit、SwiftEdit等）对比，WhereEdit在PIE-Bench上在语义一致性（CLIP-Edited）与背景保真度（PSNR）上均优于现有一阶方法，且推理速度与内存占用保持在一阶水平

**⚠️ 局限性**

仅在单步范式下工作，若需极大语义改动仍可能受限；AutoMask对Attention质量敏感，复杂场景下可能定位不准；与多步精细编辑相比，细节恢复仍略逊

---

## 259. Where Animacy Lives in Large Language Models: Tracing the Circuits of the Animacy Concept

**arXiv ID:** 2607.20995 | [PDF](https://arxiv.org/pdf/2607.20995v1)

**作者:** Samuele Punzo `[一作]` (University of Amsterdam), Sandro Pezzelle `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在动词-动员场景下的动性（animacy）区分行为进行因果机制分析，利用电路发现技术定位并评估对应的神经网络子图

**💡 创新点**

首次将机制解释方法（Edge Attribution Patching + Integrated Gradients）应用于动性问题，揭示了动性行为是分布式、分层的，并系统评估了其跨模型、跨任务的可转移性

**🔧 技术方法**

使用 EAP‑IG、激活补丁、必要性/充分性实验、随机对照、边预算扫描以及迁移实验等一系列解释与评估技术

**📊 数据集**

自构造的“被动前缀最小对照对”数据集（约 16,305 对），以及 BLiMP、命名实体等用于迁移测试的公开数据集

**📈 对比分析**

采用平均对数几率差（avg_LD）作为任务指标，模型成功率在 45–50% 之间；通过电路发现，7–13% 边缘即可实现 85% 可信度（Qwen 仅 1.74%），必要性 ablation 仅 20 条边即可导致可信度降至 0.1；迁移实验显示在仅变数据或目标的设置下电路仍保持高准确率，但在两者同时变化时电路失效，模型仍可完成任务

**⚠️ 局限性**

研究仅聚焦被动前缀与人名动性，对动物或更细粒度动性层次缺乏覆盖；数据依赖 LLM 生成的语义帧，可能存在偏差；电路与具体任务/数据高度相关，未能解释更广泛语义特征；仅使用高置信度样本进行电路发现，未覆盖模型失败情况

---

## 260. Sparse Concept Channels in Frozen 3D CT Vision Encoders

**arXiv ID:** 2607.20993 | [PDF](https://arxiv.org/pdf/2607.20993v1)

**作者:** Farhad Nooralahzadeh `[一作]` (University of Zurich), Michael Krauthammer `[通讯]` (University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对冻结的3D医学视觉‑语言模型进行训练‑自由探测，提取稀疏通道编码放射学发现，随后利用判别阈值生成检测结果并通过模板化语义化为诊断报告。

**💡 创新点**

提出Concept Channel Probe (CCP) 方法：通过通道选择度量、闭式均值差判别器实现稀疏、因果可解释的特征子集，且可在不同骨干网络、机构与解剖区域无训练迁移。

**🔧 技术方法**

采用冻结嵌入线性探测、基于每通道AUROC的通道排名、均值差方向学习、因果消融、以及基于语料库的模板化表述，无需任何参数更新。

**📊 数据集**

实验使用CT‑RATE胸部CT、RadChest‑CT外部胸部CT、Merlin腹部CT三大公开多标签数据集，并对比CT‑CHAT、CT‑CLIP、Pillar‑0等现有模型。

**📈 对比分析**

在多标签分类上，CCP‑10在CT‑RATE上取得AUROC 0.798、F1 0.790，接近监督Fine‑tune 0.833；在报告生成上报告F1 0.549、BLEU‑1 0.483，显著优于CT‑CHAT 0.184/0.373；推理延迟约0.24 s，约为CT‑CHAT的1/23。

**⚠️ 局限性**

局限性包括：罕见发现样本不足导致通道估计不稳；需依赖校准集且假设冻结编码器与解剖匹配；在解剖不匹配或更细粒度病理标签下迁移性能下降。

---

## 261. Distributed Model-Based Diffusion For Scalable Multi-Robot Trajectory Optimization

**arXiv ID:** 2607.20992 | [PDF](https://arxiv.org/pdf/2607.20992v1)

**作者:** Haejoon Lee `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了分布式模型基础扩散（DMBD）框架，用于在多机器人系统中通过服务器与机器人协同完成轨迹优化。

**💡 创新点**

创新点在于将传统全局反向扩散过程拆分为各机器人本地的条件扩散子过程，仅需局部信息与服务器广播的轨迹估计，从而显著提升规模可扩展性与采样效率。

**🔧 技术方法**

技术包括扩散模型（Model‑Based Diffusion）、条件扩散与蒙特卡罗得分上升、分布式服务器‑机器人通信以及基于JAX的GPU加速实现。

**📊 数据集**

使用基于仿真的多机器人环境（双积分、运动学自行车模型、圆形/矩形机器人，包含碰撞、楼层、电梯等约束）作为实验数据集。

**📈 对比分析**

与CEM、MPPI、MBD及D4orm等四个基线在目标交换、多层覆盖、停车与拥堵等任务中比较，DMBD在成功率、规划时间上均优于基线，尤其在大规模机器人数（N≥20）时保持高成功率且规划时间仅秒级。

**⚠️ 局限性**

局限包括：仍依赖服务器聚合与广播，通信延迟可能影响实时性；理论误差界限基于无穷样本假设，有限样本误差未完整分析；在高度耦合约束下局部得分与全局得分的偏差可能较大。

---

## 262. LivePhys: Transforming Static Physics Problems into Interactive Simulations via a Scan-to-Play Framework

**arXiv ID:** 2607.20990 | [PDF](https://arxiv.org/pdf/2607.20990v1)

**作者:** Xiaowei Dai `[一作]` (Beijing Technology and Business University), Yonghong Ke `[通讯]` (Beijing Normal University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 LivePhys 框架，将教材中静态物理题图扫描后自动转化为可执行、交互式的 2D 物理仿真。

**💡 创新点**

创新点在于将多模态感知、物理推理与确定性仿真解耦，利用大型多模态语言模型生成物理可执行的中间表示，从而实现“Scan‑to‑Play”流程。

**🔧 技术方法**

技术包括 OCR + Segment Anything 对图像进行细粒度分割，跨模态定位对齐文字与图形；链式思考提示的 GPT‑4o 等 MLLM 进行实体与约束推理；Matter.js 物理引擎执行仿真并提供交互界面。

**📊 数据集**

使用了 50 道常见教材力学题（包含静态图和文本）的自构数据集进行评估。

**📈 对比分析**

通过与 GPT‑4o、Claude 3.5 Sonnet、Gemini 1.5 Pro 在可执行率、空间精度、交互保真度三项指标的对比，LivePhys 的执行率 94%、空间精度 92%、交互保真度 90% 明显优于基线（约 70%）。

**⚠️ 局限性**

局限性包括仅覆盖 2D 力学场景，样本量有限且仅评估感知负荷（未测学习成效），缺乏对 3D 或 VR 等更复杂环境的支持。

---

## 263. From a Word-Level Dictionary to Sentence-Level Semantics: Multilingual Grievance Labelling with Contextual Models

**arXiv ID:** 2607.20946 | [PDF](https://arxiv.org/pdf/2607.20946v1)

**作者:** Lin Tian `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了情绪诉求（grievance）在威胁评估中的测量方法，指出传统词典匹配的循环性和上下文缺失问题，并构建了非循环的多语言基准。

**💡 创新点**

创新点在于：1）揭示词典匹配的循环性导致的高精度伪影；2）设计无循环的无偏抽样基准并加入全篇上下文标注；3）用多语言上下文编码器（mDeBERTa/XLM‑R）和LLM教师进行知识蒸馏，证明上下文显著提升了测量准确性。

**🔧 技术方法**

主要技术包括：词典匹配加语义修正、基于mDeBERTa和XLM‑R的目标句+全文上下文编码器、Qwen3 LLM作为教师/提示式评估，以及自监督/蒸馏训练。

**📊 数据集**

使用了 169,368 条公开 Facebook 贴文，抽样出 5 种语言（英语、意大利语、法语、德语、荷兰语）的 1,500 条评测样本，其中包含无偏随机、词典正/负两类子集。

**📈 对比分析**

通过平均精度（AP）和宏观 AUROC 对比，全文上下文模型在所有子集上均优于词典匹配，尤其在词典负区平均精度从 0.142 提升至 0.198（39% 相对增益），并且人类监督模型仍能取得最高分；上下文模型在 39% 的相对提升。

**⚠️ 局限性**

局限性包括：基准样本来源非代表性平台数据，缺乏图片、链接等多模态信息；仅使用文本上下文；以及情绪诉求本身主观且难以直接与暴力意图关联。

---

## 264. Interaction Dynamics Modeling and Predictive Control for Safe Steerable Catheter--Tissue Interaction

**arXiv ID:** 2607.20939 | [PDF](https://arxiv.org/pdf/2607.20939v1)

**作者:** Yongyan Cao `[一作]` `[通讯]` (Voryx Robotic), Yongyan Cao (Voryx Robotic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于交互动力学模型和预测优化的可安全操控可转向导管与组织的控制框架。

**💡 创新点**

创新点在于将导管动力学化简为配置不变的线性交互动力学模型，采用增益压缩卡尔曼滤波估计未知扰动，并在预测控制中硬约束接触力，实现在保持接触安全的同时实现无偏差跟踪。

**🔧 技术方法**

所用技术包括：部分物理前馈消去已知刚度与阻尼；配置不变的双积分交互动力学；增益压缩卡尔曼滤波估计扰动；基于约束二次规划的预测控制；以及在MuJoCo物理仿真环境中实现的八段张力驱动导管。

**📊 数据集**

主要数据集为MuJoCo分布式柔性模拟平台（8节段张力驱动导管与Kelvin–Voigt组织墙），并未使用真实临床数据，所有实验均在仿真中完成。

**📈 对比分析**

与经典阻抗控制、关节空间PD控制等方法对比，预测交互动力学控制在自由空间误差降低90%、在接触时保持0.5 N安全阈值且跟踪误差仅0.03 mm，且在心脏周期性运动下仍保持安全约束。

**⚠️ 局限性**

主要局限包括：仅针对单自由度张力驱动导管，需假设扰动可被低频卡尔曼估计捕获；对复杂多节段、摩擦滞后及心脏相位预测仍需进一步研究；以及缺乏硬件验证。

---

## 265. Clustered Edge Intelligence: Beyond Just Convergence of Edge Computing and AI

**arXiv ID:** 2607.20937 | [PDF](https://arxiv.org/pdf/2607.20937v1)

**作者:** Chinmaya Kumar Dehury `[一作]` (IISER Berhampur), Praveen Kumar Donta `[通讯]` (Stockholm University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 Clustered Edge Intelligence (CEI) 架构，强调把从边缘设备获取的智能（Intelligence）视为一等实体，并通过智能本身而非设备来聚类、发现、共享与生命周期管理；同时梳理了基础技术、研究维度与应用场景。

**💡 创新点**

创新点包括：① 智能作为独立可发现、可观察、可复用的实体；② 智能聚类（ICC）取代传统设备聚类（DCC），实现跨设备、跨场景的协同推理；③ 架构层次分明（设备层、控制层、云层）与完整的基础技术栈（知识图谱、语义发现、可观测性、生命周期自动化、市场化机制、LLM 辅助聚类）；④ 兼顾多租户、标准化与安全隐私需求。

**🔧 技术方法**

采用的技术与工具有：边缘代理 (Edge Agent)、知识图谱 / OKF、语义发现与可观测性框架、REST / MQTT / CoAP、5G/6G 与 LoRaWAN 通信、LLM 生成语义关系、传统聚类算法（K‑means、DBSCAN 等）以及服务发现、设备发现与网络发现机制；同时利用容器化、IaC 与区块链支持智能生命周期管理。

**📊 数据集**

论文未给出具体实验数据集或案例数据，侧重于理论框架与技术路线，因此未使用公开数据集；若要实现可验证性，可结合智慧城市、工业4.0 或智能农业等场景下的感知数据进行实验。

**📈 对比分析**

本工作为概念性研究，未提供实验对比与性能评估；作者在文中提及未来可通过模拟或真实部署评估智能聚类的准确率、延迟、能耗及系统可扩展性，但目前尚未给出量化结果。

**⚠️ 局限性**

局限性包括：① 缺乏实现与实验验证，无法验证理论假设的可行性与性能；② 规模化部署、实时性与安全隐私方面的细节尚未完善；③ 依赖于标准化与互操作性，当前缺少统一的智能描述与接口规范；④ 对资源受限设备的实际部署成本与能耗影响未进行评估。

---

## 266. The Consensus Number of Untraceable Cryptocurrencies

**arXiv ID:** 2607.20929 | [PDF](https://arxiv.org/pdf/2607.20929v1)

**作者:** Christian Cachin `[一作]` (University of Bern), François-Xavier Wicht `[通讯]` (University of Bern)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了两种加密货币转账设计：线性不可追踪资产转移（LUAT）和常量不可追踪资产转移（CUAT），并分析了它们在同步方面的影响。

**💡 创新点**

创新点在于通过形式化这两种设计，揭示了它们在隐私保护和同步效率之间的权衡，LUAT在存储上付出代价，而CUAT则在同步和公平性上付出代价。

**🔧 技术方法**

使用了线性不可追踪资产转移和常量不可追踪资产转移的形式化模型，分析了它们的共识数和同步特性。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了加密货币转账的理论模型和协议。

**📈 对比分析**

通过共识数的比较，LUAT的共识数为2，而CUAT的共识数在强不可追踪性下是四分之一的平方，表明CUAT在同步方面的复杂性更高。

**⚠️ 局限性**

限制在于CUAT在异步调度下可能导致饿死现象，而LUAT则在存储上存在无限增长的问题。

---

## 267. MagicMakeup: A Region-Controllable Diffusion Transformer for High-Fidelity Makeup-Transfer

**arXiv ID:** 2607.20924 | [PDF](https://arxiv.org/pdf/2607.20924v1)

**作者:** Ziyi Wang `[一作]` (Zhejiang University), Peng-Tao Jiang `[通讯]` (vivo Mobile Communication Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于扩散 Transformer 的面部化妆迁移框架 MagicMakeup，能够实现高分辨率（1024×1024）且可区域控制的化妆迁移，保持源脸身份与几何；

**💡 创新点**

创新点包括：①Token‑Aligned Region Gating (TARG)，将像素掩码投影到注意力空间并对 logits 进行区域门控，消除跨区域泄漏；②Cross‑Modal Perception Guidance (CMPG)，通过文本与图像概念对齐实现转移/保留的语义解耦；③利用真实化妆图像做去化妆得到高质量对齐对，构建高分辨率带区域标签的配对数据；④提出统一的 MakeupHQ 高分辨率基准，覆盖合成与真实两种设置；

**🔧 技术方法**

核心技术包括：扩散 Transformer (DiT)、Token‑Aligned Region Gating、Cross‑Modal Perception Guidance、基于 CLIP/DINO 的多模态特征对齐、面部检测与关键点对齐、自动化数据生成管线，以及多尺度采样与LoRA fine‑tuning；

**📊 数据集**

主要使用自行构建的 1024×1024 MakeupHQ‑Synthetic 与 MakeupHQ‑Real 数据集（各 500 对），以及公开的 Makeup‑Wild（256×256）作为对比；数据集通过真实化妆图像的区域性去化妆生成、身份一致性过滤与区域精细标注完成；

**📈 对比分析**

通过与 GAN（CPM、EleGANt、SSAT 等）和扩散方法（SHMT、MAD、Stable‑Makeup、Flux‑Makeup）在同一基准上评估，指标包括 DINO‑I、CLIP‑I、Face‑ID、L2M、FID；在 1024×1024 MakeupHQ 上 MagicMakeup 获得最高 DINO‑I 与 CLIP‑I，最低 L2M 与 FID，Face‑ID 仍保持竞争力；在 256×256 Makeup‑Wild 上亦保持优异表现；

**⚠️ 局限性**

局限性主要体现在：①缺乏对单个化妆品（如腮红、修容）细粒度控制；②在极端侧视角下表现下降，部分化妆无法完整保留，原因是极端姿态的数据稀缺；

---

## 268. Scientific exploration, collaboration and labor division in the large language model era

**arXiv ID:** 2607.20923 | [PDF](https://arxiv.org/pdf/2607.20923v1)

**作者:** Xiang Zheng `[一作]` (University of Wisconsin--Madison), Chaoqun Ni `[通讯]` (University of Wisconsin--Madison)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过将 PubMed Central 全文与 OpenAlex 出版与合作历史以及 CRediT 贡献声明相结合，描述性分析 LLM 普及后科研人员在研究方向、跨学科合作与团队劳动分工方面的变化。

**💡 创新点**

首次系统量化 LLM 扩散对科学家探索广度、跨学科合作与团队角色多样性的宏观影响，并揭示 AI 写作强度与科研多样性之间的选择与强化关系。

**🔧 技术方法**

采用基于词频的 AI 写作比例估计、Rao–Stirling、Shannon 熵等多样性指标、事件研究与共匹配方法，对论文文本进行混合分布最大似然估计。

**📊 数据集**

整合了 PubMed Central 全文、OpenAlex 作者与出版记录以及 PLOS/PMC 的 CRediT 贡献声明，共计约 775,000 名作者、2,600 万篇论文。

**📈 对比分析**

通过 2011‑2025 年作者级面板的事件研究和高低 AI 写作率作者匹配，对比 2022 年前后变化，观察到 2025 年研究跨学科指数上升约 13%，新领域发表比例提升约 35%，团队角色差异化显著增强。

**⚠️ 局限性**

研究仅为描述性，未能识别因果关系；AI 写作比例为文本层面推断，可能误判；数据以生物医学为主，难以推广至所有学科，且仅捕捉已公开的合作与贡献信息。

---

## 269. DMG: A Scalable and Efficient Memory-Disaggregated Graph Processing System

**arXiv ID:** 2607.20881 | [PDF](https://arxiv.org/pdf/2607.20881v1)

**作者:** Yizou Chen `[一作]` (Chinese University of Hong Kong), Ming-Chang Yang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了首个可在资源分离的内存（DM）上高效扩展、缓存友好且性能优异的图处理系统

**💡 创新点**

创新点包括：1）基于自适应索引的 DM‑友好图存储，融合索引与边列表降低 IOPS 需求；2）自适应更新协调器，结合协同更新、传值/引用重分发和直接远程更新，显著减少细粒度远程写入；3）两阶段工作负载管理，快速粗粒度划分结合运行时细粒度重调度（包括协程与动态轮询），消除 hub‑vertex 尾部延迟

**🔧 技术方法**

使用技术包括：RDMA 一侧写入/读操作、门铃批处理、索引/边列表合并、可变长度段长度压缩、协程异步执行、双模式（push/pull）执行、动态分区重用与负载均衡

**📊 数据集**

使用四个规模达数十亿的图数据集：twitter‑2010、uk‑2007‑05、rmat‑29、clueweb12

**📈 对比分析**

与现有 DM 系统 FAM‑Graph 及基线(-Base) 对比，实验显示：在单 CN 下可实现 1.8‑2.7× 的加速；在多 CN 扩展时速度提升 1.5‑3.1×；整体可实现最高 4.9× 的性能提升，并将每 CN 缓存需求降低 18.9×，启动时间仅为 Gemini 的 2.8%

**⚠️ 局限性**

局限性包括：目前仅支持静态图，动态图更新未完整实现；依赖 RDMA 低延迟和高 IOPS，未在更宽带或 CXL 环境下全面验证；仅评估了 BFS、CC、PR 三类工作负载，其他图算法的效果仍待验证

---

## 270. Latent Variable-Mediated Cross-Learning for Few-Shot Acoustic Impedance Imaging

**arXiv ID:** 2607.20989 | [PDF](https://arxiv.org/pdf/2607.20989v1)

**作者:** Junheng Peng `[一作]` (Chengdu University of Technology), Yi Bao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种结合正则化去卷积和对称交叉学习的半监督声学阻抗成像框架 RD‑SCL，解决波形未知与标注稀缺问题。

**💡 创新点**

核心创新在于：①使用一阶 Tikhonov 正则化的闭式频域去卷积算子，动态估计潜在波形并保持可微性；②采用无辅助网络的对称交叉学习，利用已估波形在标注与未标注数据间形成物理一致性约束。

**🔧 技术方法**

技术手段包括：频域去卷积、Tikhonov 正则化、时间序列卷积网络（TCN）、双向 GRU、交叉损失与监督损失的联合训练。

**📊 数据集**

实验数据集为 SEAM、Marmousi 2（合成）以及四川盆地油气勘探实际地震数据。

**📈 对比分析**

与 6 个开源基线（TCN、EIF、SSM、ADDIN、SSL、PICL）在 SNR、R²、SSIM、MAE、MSE 等指标上对比，RD‑SCL 在所有指标上均显著优于竞争方法，且参数量仅 56.5k，计算速度快。

**⚠️ 局限性**

局限性包括：假设地震波形为时不变；在极低标注比例或高噪声下仍会出现一定误差；对多源/时间变化波形的适应性尚未探索。

---

## 271. Chemical Chain-of-Thought Functions as a Hallucination-Prone Molecular Scratchpad

**arXiv ID:** 2607.20935 | [PDF](https://arxiv.org/pdf/2607.20935v1)

**作者:** Jiatong Li `[一作]` (National University of Singapore), Yatao Bian `[通讯]` (National University of Singapore)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套化学推理可检验性框架，对四大推理模型在12项生成化学任务上进行评测，量化了中间推理过程中的幻觉（extrinsic reasoning fabrication）并分析其对最终答案的影响。

**💡 创新点**

创新点在于：① 将化学结构作为可直接验证的中间步骤，将 CoT 视为“scratchpad”而非可信解释；② 提出 ER（extrinsic reasoning fabrication）度量，揭示答案正确与推理过程错误之间的解耦；③ 通过过程监督奖励（GRPO + 验证门控）显著降低 ER，同时保持/提升任务性能。

**🔧 技术方法**

采用的技术包括：规则式化学实体抽取与 RDKit SMARTS/SMILES 匹配；梯度×输入 saliency 与注意力分析定位关键 trace 词；功能团替换、SMILES 草稿扰动等因果干预实验；Semantic 与 conditional entropy 评估推理不确定性；过程监督训练（Chem‑R‑Faithful）结合答案准确性、反幻觉、验证奖励。

**📊 数据集**

使用的数据集有：ChEBI‑20（caption‑to‑molecule、molecule‑to‑caption）、USPTO‑50k（retrosynthesis）、S²‑Bench（结构生成、编辑、优化子任务）。

**📈 对比分析**

通过对四个公开推理模型与改进后的 Chem‑R‑Faithful 在同一任务集上的精确匹配率、ER 率、整体幻觉分数进行对比。Chem‑R‑Faithful 在保持或提升任务准确率的同时，将 ER 从约 13% 降至 4%，整体幻觉分数大幅下降，且不牺牲原有性能。

**⚠️ 局限性**

局限性包括：① 只针对功能团级声明，未覆盖所有化学表达；② ER 仍存在，未完全消除幻觉；③ 模型可能通过写完整答案提前“作弊”，导致对过程监督的评估失真；④ 仅在化学领域验证，需进一步测试在其他科学领域的可迁移性；⑤ 过程监督奖励依赖手工编写的验证器，扩展性受限。

---

## 272. Transformer-Assisted LLM-Based Source Code Summarisation: to Enable More Secure Software Development

**arXiv ID:** 2607.20933 | [PDF](https://arxiv.org/pdf/2607.20933v1)

**作者:** Jesse Phillips `[一作]` (Lancaster University), Mo El-Haj `[通讯]` (Lancaster University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将小型Transformer生成的代码摘要作为提示，提升LLM在源代码摘要任务中的表现。

**💡 创新点**

创新点在于融合Transformer的技术词汇与LLM的生成流畅性，通过“Transformer-Assisted LLM-Based Source Code Summarisation”实现对比度更高的摘要质量。

**🔧 技术方法**

使用的技术包括小型任务专用Transformer（CodeSumBart）、四款开源LLM（Llama 3.1、Llama 3.2、CodeLlama、Deepseek Coder 1.5）和多种prompt策略（zero-shot、implicit、explicit、explicit-improvement）。

**📊 数据集**

数据集为经过改进清洗的Funcom（499,997条Java方法–摘要对），并在其上训练CodeSumBart以生成示例摘要。

**📈 对比分析**

通过多种评估指标（Bleu‑1/4、Smoothed Bleu‑4、Rouge‑L、BertScore）比较，发现使用“Explicit one‑shot”提示的CodeLlama在Bleu‑4提升约12.4%、BertScore提升至70.79%，整体优于其他LLM。

**⚠️ 局限性**

主要限制包括：实验仅覆盖小于10B参数的LLM，缺乏对更大模型的验证；使用旧版Funcom数据集可能不具备当前开发趋势的代表性；评估仅依赖自动指标，未包含人工评测。

---

## 273. FA-LAM: Focus-Aware Large Avatar Model for One-Shot 4D Animatable Gaussian Head

**arXiv ID:** 2607.20922 | [PDF](https://arxiv.org/pdf/2607.20922v1)

**作者:** Yingdong Hu `[一作]` (HKUST), Jun Zhang `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种基于3D高斯的全头动画化数字人模型，能够从单张图片或视频流一次性生成可控表情的高质量全头模型，并支持多视角和实时流式重建。

**💡 创新点**

主要创新点包括：① 通过对称语义注意力正则化，利用头部双侧对称特征引导注意力集中；② 将重建与动画两项任务拆分为双阶段训练，消除梯度冲突；③ 设计可视化门控的自回归重建模块，实现多视角/流式输入下的常量显存占用。

**🔧 技术方法**

技术上使用了 Vision Transformer+DINO编码器、UV空间交叉注意力、3D Gaussian 解码器、FLAME 参数化、双阶段训练、可视化门控融合、GMM注意力正则化及多模态损失（MSE、LPIPS、mask、KL）。

**📊 数据集**

训练与评估数据集包括 VFHQ、Nersemble‑v2、Ava256，以及自制的多视角扩展 MV‑VFHQ；在这些数据上与多种基线进行对比。

**📈 对比分析**

与现有单视角与多视角/4D重建方法相比，本文在自重建、跨身份重现、流式重建等任务中均取得最高或相近的 PSNR/SSIM/L‑PIPS，并在跨身份和表情驱动上显著提升 CSIM、AED/ APD 等指标；此外自回归模块在显存和时序一致性上优于全注意力方案。

**⚠️ 局限性**

限制包括：对非人类或高度扭曲面部的鲁棒性未评估；在极端遮挡或非常大视角下细节仍可能出现模糊；模型对训练数据的多样性依赖较高，若缺乏高质量侧视图可能影响效果。

---

## 274. Source-Prior-Driven Selective Adaptation for Efficient Diffusion Model Finetuning

**arXiv ID:** 2607.20913 | [PDF](https://arxiv.org/pdf/2607.20913v1)

**作者:** Yi Xiong `[一作]` (University of Science and Technology of China), Xiao-Ming Fu `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种源先验驱动的选择性微调框架，通过学习静态保留掩码来识别对源域影响较小的参数，并将其补集作为固定可写子空间，实现扩散模型在有限可训练参数下的高效微调，同时兼顾目标域适应和源域保留。

**💡 创新点**

创新点包括：① 通过源侧探测学习静态保留掩码，显式确定哪些参数对源域影响最小；② 将该掩码的补集作为固定可写支持，直接控制目标知识写入位置；③ 针对不同层级和参数类别设计选择性更新策略，进一步提升适应-保留权衡；④ 结合稀疏正则和二阶近似解释掩码学习过程。

**🔧 技术方法**

主要技术手段包括：可学习分数张量与全局阈值（配合STE）实现掩码学习；源侧损失 + 稀疏正则构建 Stage‑I 目标；低秩/稀疏/选择性微调策略；二阶泰勒展开解释掩码对源域损失的影响；在微调阶段使用固定可写掩码进行梯度投影。

**📊 数据集**

实验使用 Stable Diffusion 1.5 与 Stable Diffusion 3 作为基准模型；目标域包含 WikiArt、Anime、Cyberpunk、Pokemon 等；评估指标包括目标域 KID、CLIP；源域 FID、sCLIP；通过 Trade‑off 指标衡量适应-保留平衡。

**📈 对比分析**

与 LoRA、DoRA、SaRA、SVDiff、WaveFT 等基线对比。实验表明，在 1.6M~21.4M 可训练参数范围内，Linear‑Only/All‑Layers 方案在目标 KID 与源 FID 之间取得更优 Trade‑off，整体性能优于传统低秩/稀疏微调方法，尤其在小至中等预算时优势更为明显。

**⚠️ 局限性**

局限性包括：掩码质量受源侧数据分布影响，可能在多样性不足时表现不佳；Stage‑I 需要额外计算成本；掩码为静态，缺乏对不同提示、时间步或多域持续学习的动态适配，未来需探索动态路由与任务自适应掩码。

---

## 275. Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation

**arXiv ID:** 2607.20908 | [PDF](https://arxiv.org/pdf/2607.20908v1)

**作者:** Quazi Ishtiaque Mahmud `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于强化学习与可验证奖励的框架（RLVR+结构奖励），能够将顺序代码（C、PyTorch）自动翻译并生成性能优化的 CUDA 程序。

**💡 创新点**

创新点包括：①将结构化的 GPU 性能特征（内存共线、算术强度、占用率、同步模式等）与可验证的执行奖励联合形成复合奖励；②使用离线对比排序器快速区分优劣候选；③采用多轮迭代细化与结构化反馈；④构建了包含多实现、多输入的 C→CUDA 与 PyTorch→CUDA 数据集。

**🔧 技术方法**

技术手段：基于 Qwen‑3‑32B 的大模型，使用 QLoRA 4‑bit 微调；采用 GRPO 训练策略；离线使用 MLP 结构评分器；执行 harness 包括编译、单元测试、速度测量；多轮 prompt 结构化反馈；使用结构化特征提取与归一化。

**📊 数据集**

数据集：2.9k C→CUDA、1k PyTorch→CUDA，分别配备 5+ CUDA 变体；额外使用 BabelTower、KernelBench、CodeNet、C++/Python 代码库；每个顺序程序至少有 5 个可编译、通过测试且至少 1.5X 加速的实现。

**📈 对比分析**

与多种基线（Qwen‑3‑32B、Qwen2.5‑Coder‑32B、CodeLlama‑34b‑Instruct‑hf、OpenAI O4‑mini、QiMeng‑MuPa、CUDA Agent）在正确率和几何平均加速上进行对比；在 C→CUDA 任务中，模型正确率提升至 89%（比 Qwen‑3‑32B 提升 17%），几何平均加速提升至 11.02X（比 Qwen‑3‑32B 提升 5.3X）；在 PyTorch→CUDA 任务中，加速提升至 6.41X（比 CUDA Agent 提升 3.32X），正确率提升至 93%。

**⚠️ 局限性**

限制：训练高度依赖执行反馈，导致编译与基准测量成本高；迭代细化受模型与奖励设计限制，可能无法探索未见过的优化策略；结构特征提取与排名器的准确性依赖于训练数据质量。

---

## 276. DINO-VPT: Hierarchical Visual Prompt Tuning for Joint Physical-Digital Face Anti-Spoofing

**arXiv ID:** 2607.20900 | [PDF](https://arxiv.org/pdf/2607.20900v1)

**作者:** Pierre Gallin-Martel `[一作]` (Tohoku University), Takafumi Aoki `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级的纯视觉框架DINO-VPT，用于统一物理与数字面部欺骗检测；

**💡 创新点**

通过层级视觉提示树（VP-Tree）与动态路由网络（PRN）实现输入条件化的多级提示注入，结合模拟欺骗提示（SSC）来提升跨模态泛化；

**🔧 技术方法**

使用DINOv2 ViT-B/14作为基础模型，加入registers、层级视觉提示、Prompt Routing Network、Simulated Spoofing Cues以及端到端的多任务损失；

**📊 数据集**

在UniAttackData统一欺骗数据集和MICO跨数据集评测基准上进行实验；

**📈 对比分析**

与VLM基线（UAD+、MoAE-CR、SUEDE、HiPTune）及传统视觉模型（MTFace、ResNet50、ViT-B/16等）对比，DINO-VPT在UniAttackData Protocol2平均ACER仅0.63%，在MICO上的HTER和AUC与顶尖视觉模型持平；

**⚠️ 局限性**

主要局限是仍需依赖DINOv2预训练与手工设计的提示树，且在更复杂的多模态或无层级标签的数据集上效果未知；

---

## 277. Beyond Independent Optimization: Compression, MoE Routing, and Quantization Interactions in Multimodal Edge Intelligence

**arXiv ID:** 2607.20981 | [PDF](https://arxiv.org/pdf/2607.20981v1)

**作者:** Jay Gor `[一作]` (Nirma University), Zhengkui Wang `[通讯]` (Singapore Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了 2021‑2026 年间视觉‑语言和多模态大语言模型的高效推理技术，强调压缩、路由、量化、缓存和硬件部署之间的耦合失效链。

**💡 创新点**

提出了以失效传播链为核心的统一视角，并引入时序路由一致性（TRC）作为视频 MoE 模型的诊断指标，首次把多技术交互纳入评估框架。

**🔧 技术方法**

主要利用压缩（token dropping/merging）、MoE 路由（Top‑k、连续路由）、量化（PTQ、混合精度）、KV 缓存优化和硬件感知基准等方法，进行综合讨论。

**📊 数据集**

无原始实验数据集；采用对 2021‑2026 年发表的公开论文与公开基准（如 VQAv2、LLM‑Bench、MMLBench）进行文献归纳。

**📈 对比分析**

通过综述和对比分析，指出各技术在不同阶段的性能权衡与互相影响，但未给出统一实验结果，主要呈现已有工作在准确率‑效率 Pareto 前沿上的分布。

**⚠️ 局限性**

局限在于缺乏统一实验平台与标准化指标，无法验证跨技术协同优化的实际收益；并且对长序列、多模态任务的细粒度评估仍待进一步研究。

---

## 278. GuardianAgentBench: Where Agents Fail and How to Guard Them

**arXiv ID:** 2607.20982 | [PDF](https://arxiv.org/pdf/2607.20982v1)

**作者:** Vishal Ishwar Naik `[一作]` (Vectara, Inc.), Humayun Irshad `[通讯]` (Vectara, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GuardianAgentBench（GABench）用于在生产级框架下评估LLM代理的安全性，并实现三种防护机制。

**💡 创新点**

创新点包括：①将 benchmark 扩展至实际生产框架（LangChain、LlamaIndex、Vectara）；②通过多阶段自动生成与人工校验构建 580 个多工具、多轮场景；③引入五种对抗攻击模式并对防护方法做统一比较；④实现结构化执行时 guardrail 并证明其优于系统提示。

**🔧 技术方法**

技术手段包括：大语言模型（Claude Sonnet、GPT-5.2、Gemini-3、DeepSeek、Qwen3、GPT-OSS-120B）驱动的场景生成与评判；多轮工具调用的 DAG 追踪；基于规则的 guardrail（参数校验、工具覆盖、相关性/成本检查）和错误回退机制。

**📊 数据集**

使用 GABench 内部生成的 580 场景数据，涵盖六个领域（客户服务、邮箱、日历、商业智能、金融、内部知识）以及 81 种工具，按 398 正常 + 182 对抗实例划分。

**📈 对比分析**

比较方法：在三框架、六模型上对响应、行动和整体准确率进行测评，并在同一设置下对比系统提示与 guardrail 两种防护；结果显示最佳模型整体准确率仅 74.8%，而 guardrail 使失败率下降约 20% 并保持 0.5% 的误报率。

**⚠️ 局限性**

局限性包括：仍受限于自动与人工验证的覆盖范围；对抗模式覆盖面有限；guardrail 主要针对工具调用层面，未覆盖更深层次的安全风险；评测聚焦于生产框架，缺乏对自定义环境的适应性评估。

---

## 279. The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning

**arXiv ID:** 2607.20952 | [PDF](https://arxiv.org/pdf/2607.20952v1)

**作者:** Ishan S. Kshirsagar `[一作]` `[通讯]` (Independent Researcher), Ishan S. Kshirsagar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在棋类任务中，使用阶段化的连续思维训练和随后强化学习，提升模型合法走子率至61%，并消除虚假将死陈述；

**💡 创新点**

证明连续（latent）思维在RL后并非被主动读取，而是通过塑造模型权重实现鲁棒性提升，挑战了传统的scratchpad假设；

**🔧 技术方法**

采用Coconut式连续思维机制、GRPO强化学习、J‑lens与六条件因果干预等技术；

**📊 数据集**

使用12,002个手工标注的国际象棋位置（FEN+最佳走法+解释）以及100个固定评估位置；

**📈 对比分析**

与SFT、显式链式思维+RL等基线对比，法子合法率从38%提升至61%，误报将死从28/100降至0/100，准确率保持约10%不变；

**⚠️ 局限性**

仅在单一seed、特定模型规模和任务范围内，缺乏跨模型/领域验证，且RL对准确率无显著提升，结论受训练样本与评估方法限制。

---

## 280. Best-of-Evidence: Best-of-N Selection under Partial Verification

**arXiv ID:** 2607.20950 | [PDF](https://arxiv.org/pdf/2607.20950v1)

**作者:** Cenwei Zhang `[一作]` (IQuest Research), Lei You `[通讯]` (Technical University Of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出 Best-of-Evidence（BoE）框架，在固定候选池下通过局部可检验命题和预算分配实现对多模态任务的证据驱动选择。

**💡 创新点**

创新点在于将部分验证问题建模为带符号候选–因子图，利用预算价值信息选择共享证据，实现比单纯的 Best-of-N 更高效的查询。

**🔧 技术方法**

采用候选–因子图、价值信息（EVSI）近似、分数加权更新、基于阈值的停止等技术，并在实验中使用 Qwen3-VL-30B 与 Qwen3-VL-235B 进行候选生成与证据判断。

**📊 数据集**

评测使用四个医学 VQA 数据集：VQA-Med、PathVQA、PMC-VQA、MedXpertQA-MM。

**📈 对比分析**

通过对比原始 BoN、随机因子检索、整答复判断及 BoE，实验表明在 VQA-Med 上 BoE 在 16 预算下比 BoN 提升约 0.26–0.58% 准确率，且在部分数据集显示显著的救援率。

**⚠️ 局限性**

限制在于候选生成质量和可检验命题的噪声；若候选池不含正确答案或证据不可靠，BoE 无法弥补；且当前的预算分配采用贪婪近似，可能未能充分挖掘多步证据序列。

---

## 281. Three-Pronged Spectral Control for Federated Parameter Efficient Fine Tuning

**arXiv ID:** 2607.20914 | [PDF](https://arxiv.org/pdf/2607.20914v1)

**作者:** Shiva Raj Pokhrel `[一作]` (Deakin University), Anwar Walid `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TRISHUL 框架，在联邦环境中实现参数高效微调

**💡 创新点**

创新点在于三重光谱控制：共享冻结多头基实现精确聚合、核范数近端收缩抑制局部噪声、凹水池分配动态调度层级容量

**🔧 技术方法**

使用光谱正则化、核范数收缩、精确多头低秩聚合、凹水池头分配以及基于 SVD 的近端阈值化

**📊 数据集**

使用 CIFAR-100、SVHN、20 Newsgroups、MRQA、GLUE 以及 LLaMA3.2‑1B 大模型进行实验

**📈 对比分析**

与 FedIT、FedEx‑LoRA、FFA‑LoRA、Fed‑SB、RAVAN、SCAFFOLD+LoRA 等基线对比，TRISHUL 在异构设置下提升 3–5% 的准确率，通信量保持不变且计算开销 <1%

**⚠️ 局限性**

对核范数收缩参数敏感，过度收缩会导致性能退化；目前不提供正式隐私保证；动态层分配、不同秩的多头设计等仍待进一步研究

---

## 282. Tencent WorkBuddy Bench: A Multi-Domain Coding-Agent Benchmark with Contamination-Resistant Task Construction

**arXiv ID:** 2607.20911 | [PDF](https://arxiv.org/pdf/2607.20911v1)

**作者:** Tencent WorkBuddy Bench Team `[一作]` (Tencent), Xing Sun `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Tencent WorkBuddy Bench——一个多域评测套件，涵盖代码、前端、办公与安全四类真实工作任务，提供统一的任务目录、执行环境、评分仪表及公开榜单。

**💡 创新点**

创新点在于：1）任务构建通过从真实提交、PR 或业务场景逆向重写为口语化请求，防止可搜索提示污染；2）采用分布式信息驱动的任务选取，匹配真实使用分布；3）统一多域任务格式与执行管线，支持双模型主机并开放完整代码与数据，便于第三方复现与审计；4）多样化评分机制（隐藏测试、规则检查、LLM/VLM/Agent 评判、确定性程序评分）。

**🔧 技术方法**

使用的技术包括 Docker 容器化执行、Harbor 任务目录约定、LLM 评判器（LLM/VLM、Agent Judge）、确定性单元测试、五层反作弊框架、跨模型（CodeBuddy、Claude）主机、自动化打分与聚合脚本。

**📊 数据集**

数据集为四个子集的任务集合：Code 80 个基于真实 OSS 代码的仓库任务；Web 70 个前端项目任务；Office 50 个混合格式文件工作流任务；Security 60 个红蓝团队安全任务。所有任务公开且不包含任何用户数据。

**📈 对比分析**

评测方法：在两个模型主机（CodeBuddy Code、Claude Code）上分别执行每个任务，收集隐藏测试或规则检查分数，按任务权重平均得到子集分数，并公布跨模型排行榜。实验显示不同模型在四个子集的排名差异显著，说明模型在代码、前端、办公与安全能力上存在明显差异；在某些子集（如 Security、Office）表现优异，而在代码子集中易忽略合同细节导致得分偏低。

**⚠️ 局限性**

主要限制包括：1）公开发布易导致后期模型训练中泄露；2）模型评判器可能引入偏见；3）Code 子集主要为 Python，跨语言泛化有限；4）Office 子集仅测试文本与文件，未覆盖 OCR、视觉判断等；5）评分依赖特定主机与接口，参数配置可能影响结果；6）任务构建基于内部使用统计，可能仍存在隐性偏好。

---

## 283. HierarchicalDAEW: Domain-Aware Edge-Weighted Graph Convolution with Evidential Uncertainty for Multi-Section Spatial Gene Expression Prediction from H&E Histology

**arXiv ID:** 2607.20896 | [PDF](https://arxiv.org/pdf/2607.20896v1)

**作者:** Kritanu Chattopadhyay `[一作]` (National Institute of Technology Durgapur), Debotosh Bhattacharjee `[通讯]` (Jadavpur University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

从 H&E 病理图像预测空间基因表达并给出置信区间

**💡 创新点**

① 基于 Leiden 聚类的三类边类型（内域、跨域、边界）进行域感知加权卷积；② 采用层级域级 GCN 与 CrossScaleGate 进行域级上下文融合；③ 用 evidential 归一逆伽马分布实现单前向推断的不确定性估计；

**🔧 技术方法**

图卷积网络 (DAEWConv、DomainGCN)、多层感知机、Vision Transformer (UNI)、Relational GCN、Normal–Inverse‑Gamma 归一化损失、对比学习、conformal 预测、Monte Carlo Dropout 对比

**📊 数据集**

10x Visium 六块人类组织切片（乳腺、结直肠、前列腺、小脑等），共计约 10k spots，512 个空间可变基因

**📈 对比分析**

与 13 个公开基线（STNet、HisToGene、Hist2ST、TRIPLEX、SEPAL 等）在三块乳腺切片的多切片联合训练中平均 PCC 0.696，显著优于基线；在单切片和跨组织迁移测试中仍保持较高 PCC（>0.58），并实现置信区间覆盖率 0.903，显示良好校准

**⚠️ 局限性**

① 边类型依赖表达域，推理时需先用中心匹配得到域标签；② 仅对 512 个可变基因，尚未覆盖全转录组；③ 对不同组织和采样协议的泛化仍有限，需要少量适配数据；④ 计算成本较高，需 GPU 加速

---

## 284. TwistedMerge: Certified Higher-Order Diagnostics and Abstention for Model Merging

**arXiv ID:** 2607.20887 | [PDF](https://arxiv.org/pdf/2607.20887v1)

**作者:** Ting Gong `[一作]` (University of Washington), Shitan Xu `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 TwistedMerge，一套基于有限对比复合体的模型合并认证与放弃（abstention）框架，明确区分循环不一致、同步可去除的误差、中心 H^2 反对以及非阿贝尔平移；并给出了对应的无解定理、误差控制定理与灵敏度检验。

**💡 创新点**

创新点包括：①将模型合并视作有限降解问题，引入对比复合体的“冻结”机制；②提出三路判定（平凡、非平凡、未认证）与严谨的放弃门；③给出常数边无解定理、冻结复合体三路规则与预声明族误差控制等新理论；④将同步、中心类与非阿贝尔平移统一到同一诊断流水线；⑤在实验中展示同步可去除的对齐缺陷、因表示变化导致的因子平均不稳定、以及在受控实验中证明中心类与项目化预测器的有效性。

**🔧 技术方法**

核心技术包括：离散降解理论、Čech 2-共形、三角残差的投影与闭合检测、距离到余边界的几何度量、中心系的投影与闭合门、非阿贝尔平移的分支投影、基于阈值的三路判定、以及对低秩适配器的 GL_r 变换不变性检验。

**📊 数据集**

主要数据集为 MNIST MLP（用于对齐缺陷实验）、CIFAR‑10 ResNet‑18 低秩 LoRA 适配器（用于表示不变性实验）以及 120 套自然检查点集合（用于自然残差预测与中心类验证）。

**📈 对比分析**

实验对比：①同步对齐显著消除单一边错误，保持与未损坏合并相同的性能；②对比传统因子平均，全球同步或稠密 SVD 在准确率与稳定性上优于平凡平均；③在受控的 μ₂ 四面体实验中，中心类证据能被项目化预测器恢复，且在自然数据中中心类与周期指数门未通过；④自然残差对合并性能的预测几乎为负，说明单一残差不具备普遍预测性。整体上 TwistedMerge 在需要严格认证时能提供安全放弃，而在自然场景下更适合作为基准。

**⚠️ 局限性**

限制包括：①对比复合体的构造缺乏自然、可泛化的规则；②中心类与项目化预测器的实现仍处于受控实验，缺乏端到端的实际合并提升；③实验规模受限于单层 LoRA 与 ResNet‑18，未覆盖多层 transformer 或更大模型；④对决策效用的评估尚未在自然分布漂移下完成；⑤对非阿贝尔平移与周期指数的实用意义仍需进一步验证。

---

## 285. HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving

**arXiv ID:** 2607.20988 | [PDF](https://arxiv.org/pdf/2607.20988v1)

**作者:** Quanfu Yu `[一作]` (Automotive New Technology Research Institute, BYD Company Limited), Liulong Ma `[通讯]` (Automotive New Technology Research Institute, BYD Company Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyWorldVLA 混合世界模型，将像素级监督与潜在空间学习结合，实现端到端的视觉-语言-动作驱动。

**💡 创新点**

创新点在于三阶段训练：预训练阶段联合视频 VAE 潜在预测与像素重建，随后仅预测潜在并送入动作专家，实现像素细粒度建模与噪声鲁棒性的统一。

**🔧 技术方法**

技术包括视频 VAE + Flan‑T5 跨模态注意、可学习查询 Q、Emu3 VLM、联合注意机制、生成/选择式动作专家，以及多模态离散化与自回归训练。

**📊 数据集**

使用 OpenScenes、NuPlan 进行预训练，NAVSIM v1/v2 进行评估，并在 OpenScenes 破坏集上测试场景噪声鲁棒性。

**📈 对比分析**

与 Pixel‑based（DriveVLA‑W0、WoTE 等）、Latent‑based（DreamerAD、World4Drive 等）以及 VLA 基线比较，HyWorldVLA 在 NAVSIM v1/v2 上 PDMS/EPDMS 最高，在噪声测试中达到 86.87 分。

**⚠️ 局限性**

局限性包括仅基于单目摄像头，缺乏多模态传感器融合，对关键交通元素的潜在动力学耦合不足。

---

## 286. Distribution-Alignment Bridge for Uncertainty-Aware Text-to-Video Retrieval

**arXiv ID:** 2607.20984 | [PDF](https://arxiv.org/pdf/2607.20984v1)

**作者:** Kyeongmo Chae `[一作]` (Kyungpook National University), Sangtae Ahn `[通讯]` (Kyungpook National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文将文本‑视频检索转化为分布对齐任务，使用高斯分布表示语义与不确定性，并通过确定性分布桥实现迭代对齐；

**💡 创新点**

创新点在于将检索视为分布到分布的确定性迁移，既保留各模态的误差不确定性，又利用方向性KL对比损失实现更精细的排名；

**🔧 技术方法**

核心技术包括CLIP预训练编码、概率编码器（均值与对数方差输出）、基于时间嵌入的漂移网络（分布桥）以及双向KL对比损失；

**📊 数据集**

在MSR‑VTT、MSVD和VATEX三大公开数据集上进行实验；

**📈 对比分析**

与现有基于CLIP、概率建模与扩散的SOTA方法对比，DAB在R@1、R@5、R@10以及平均排名（MnR）上均显著提升（MSR‑VTT R@1提升至56.2，R@10提升至92.4；MSVD R@1提升至54.5，R@10提升至93.9；VATEX R@1提升至70.7，R@10提升至98.3），同时保持较低的计算延迟；

**⚠️ 局限性**

局限性包括：对高维高斯近似的假设可能不足以捕捉复杂分布；桥的确定性迭代可能在极端模态差异下收敛慢；缺乏跨语言或多模态扩展的验证。

---

## 287. Delivery, Not Storage: Cue-Anchored Working Memory as a Harness Property for Coding Agents

**arXiv ID:** 2607.20972 | [PDF](https://arxiv.org/pdf/2607.20972v1)

**作者:** Swapnanil Saha `[一作]` `[通讯]`, Swapnanil Saha

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了面向代码生成代理的两层记忆体系：一层是传统的文档记忆，另一层是基于线索的工作记忆（脑记忆）。

**💡 创新点**

创新点包括：① 结合认知科学的线索记忆理论，设计出可组合的多词汇触发器（路径、符号、语义、事件、时间）；② 将触发器存放在代理自身写入的工作记忆中，而非静态规则；③ 在硬件层面实现 deterministic 的记忆注入与审计；④ 通过控制实验验证该记忆体系能显著提升长周期代理的记忆利用率。

**🔧 技术方法**

核心技术有：1) Vectr 本地索引守护进程提供工作记忆存储与触发器评估；2) 两种注入渠道——Harness 生命周期钩子与 API 代理；3) 触发器语言（{path, symbol, semantic, event, temporal}）与 deterministic 执行；4) 记忆生命周期管理（注入预算、重复抑制、过期检查）。

**📊 数据集**

使用了 Apache Camel（约 169k 代码块）作为单一代码库；通过在同一项目上实现 “Stream‑Mode Resequencer” 功能进行实验；此外在衰减实验中使用了同一代码库的 64 文件子集和 10 条合成运维事实。

**📈 对比分析**

比较方法：在 12 次人工评分的对照实验中，对比无记忆、仅工具、仅注入、混合等多种配置。结果显示：① 强制注入的配置在 114 回合中零次主动记忆调用；② 注入渠道（钩子或 API）均无正确性损失；③ 在重复压缩实验中，注入存储的 10 条事实在 138 次压缩后仍被完整恢复，而仅使用摘要通道的系统全部丢失。性能方面，注入带来的回合/时间增幅在 5–30% 之间，符合实验规模。

**⚠️ 局限性**

局限性：① 样本量小（12 组对照、1 组衰减实验）；② 仅在单一项目与单一代理/模型上验证，缺乏跨项目和跨模型的通用性验证；③ 触发器词汇仅部分被激活（路径、事件、语义），符号与时间触发器未验证；④ 捕获侧自动化未实现，仍依赖人工编写；⑤ 读取上限与工具限制导致部分实验受限；⑥ 评估主要基于 token 匹配，未检验语义衰减。

---

## 288. From Scalars to Time Series: Rethinking Implicit Neural Representations for Time-Varying Volumetric Data

**arXiv ID:** 2607.20970 | [PDF](https://arxiv.org/pdf/2607.20970v1)

**作者:** Weihan Zhang `[一作]` (Sun Yat-sen University), Jun Tao `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将时间变化体积数据视为空间索引的时间序列，并在此基础上构建了基于Mixture‑of‑Experts（MoE）的隐式神经表示（INR）框架，用以高效压缩并重建时变体积数据。

**💡 创新点**

创新点包括：①把四维稠密标量预测改为对每个空间位置的完整时间序列进行学习，显著降低了对时空联合采样的需求；②利用可训练的时间嵌入与空间特征共同驱动专家路由，充分挖掘空间中不同区域的异质时间动态；③采用LoRA低秩适配器对专家解码器进行轻量化参数化，兼顾模型容量与压缩率；④通过聚类预训练的warm‑up阶段为路由器提供伪标签，稳定专家分配。

**🔧 技术方法**

技术手段包括：隐式神经表示（SIREN、Instant‑NGP 之类的坐标‑基 MLP）、Mixture‑of‑Experts 路由网络、低秩 LoRA 解码器、可学习的时间嵌入表、位置编码与正弦激活、硬路由与交叉熵预训练、以及基于聚类的 warm‑up 方案。

**📊 数据集**

实验使用四个大型科学仿真数据集：argon bubble intensity、combustion CHI/YOH/MF、ionization H2/H+/PD、vortex vorticity（以及 Tangaroa vorticity magnitude）。

**📈 对比分析**

在 PSNR、LPIPS、DreamSim、Hausdorff 距离等指标下，与现有学习型 INRs（SIREN、NeurComp、CoordNet、Switch‑NeRF、Neural Experts、MoE‑INR）以及传统误差界定压缩器（ZFP、SZ3、TTHRESH）进行对比。结果显示，本文框架在保持或略优 PSNR 的同时，压缩比提升至 10‑100×，压缩/解压时间分别比学习基线快 40‑60×、比传统压缩器快 20‑40×，并在多数数据集上获得最优或竞争性的几何一致性与感知质量。

**⚠️ 局限性**

局限性包括：①在高度复杂的时间动态场景下，压缩质量不一定最优；②模型对每个空间位置独立建模，未显式捕捉邻域间的时空相关性；③由于一次性预测完整时间序列，随机访问单个时间步的灵活性较低，适合连续帧重建而非点对点查询。

---

## 289. FSB-Net: Frequency-Spatial Boundary Network for Brain Stroke Lesion Segmentation in Non-Contrast CT

**arXiv ID:** 2607.20955 | [PDF](https://arxiv.org/pdf/2607.20955v1)

**作者:** Linke Fan `[一作]` (Tongji University), Kai Shu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于频域边界建模的脑卒中病灶分割网络 FSB‑Net，能够在低对比度的非增强 CT 上实现更精细的边界分割。

**💡 创新点**

创新点在于：①利用离散小波变换提取多尺度高频边界特征；②设计频域-空间双向交叉注意力模块（FSCAM）实现边界与空间特征的相互强化；③引入频谱边界损失（Spectral Boundary Loss）以显式优化边界锐度。

**🔧 技术方法**

采用了 PVTv2‑B2 视觉 Transformer 编码器、Haar 小波 DWT、双向注意力、频谱边界损失以及多尺度深度监督等技术。

**📊 数据集**

使用公开的 Brain Stroke CT 数据集（含 1,130 份缺血性、1,093 份出血性病灶），共 2,223 份标注切片，尺寸 352×352。

**📈 对比分析**

与 U‑Net、UNet++、MANet、DeepLabV3+ 等基线进行比较，FSB‑Net 在 mDice、mIoU 与 HD95 上分别取得 94.85%、90.35% 与 2.01 像素，明显优于最强基线 DeepLabV3+（mDice 93.78%、HD95 2.11）并大幅降低边界误差。

**⚠️ 局限性**

局限性包括：仅在 2D 切片上工作，未利用体积上下文；使用 Haar 小波可能无法完美捕捉弯曲边界；模型未区分缺血与出血两种子类型，可能限制临床应用。

---

## 290. Sidewalk Moments: Are Richer Representations Always More Human-Aligned? Evidence from City-Walk Videos

**arXiv ID:** 2607.20903 | [PDF](https://arxiv.org/pdf/2607.20903v1)

**作者:** Liu Liu `[一作]`, Fábio Duarte `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对61个YouTube城市步行视频进行细粒度划分，提取视频、时序平均图像（TAI）、音频和文本等多模态特征，并评估这些表示对观众参与度的预测能力。

**💡 创新点**

提出将时间维度压缩成TAI并与完整视频表示对比，发现尽管视频在连续相关性上更强，但在二分类任务中TAI表现相当甚至更好，揭示人类感知在时间压缩上的选择性特性。

**🔧 技术方法**

使用VideoMAE、MAE ViT-L、Intern-VL3、EVA-CLIP、VGGish等预训练模型提取特征；使用Logistic回归、线性SVC、KNN、RBF‑SVC、随机森林和LightGBM等分类器进行评估；还用AMT 2AFC实验进行人类验证。

**📊 数据集**

61个公开YouTube城市步行视频，约109小时，总计约50,000个10秒剪辑，包含多城市、多风格的街景。

**📈 对比分析**

通过Spearman、Kendall、pairwise accuracy、nDCG@10等连续指标，以及不同阈值的二分类AUC/F1，对比不同模态的表现；结果显示TAI在二分类AUC/F1上往往优于视频，文本表现中等，音频表现接近随机；AMT实验确认人类对TAI和视频的判断相近。

**⚠️ 局限性**

限制包括：依赖YouTube重放热度作为间接参与信号，可能受平台行为影响；仅聚焦城市步行视频，难以推广到其他视角或更动态的环境；gap分析仅基于单一线性分类器，未完全探索所有情境；未设计自适应时间压缩机制。

---

## 291. Anti-Goal Reasoning: Rethinking the Theory of Goal Reasoning in Non-Axiomatic Logic

**arXiv ID:** 2607.20902 | [PDF](https://arxiv.org/pdf/2607.20902v1)

**作者:** Bowen Xu `[一作]` `[通讯]` (Temple University), Bowen Xu (Temple University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在非公理逻辑(NAL)中对目标推理进行扩展，提出了反目标（anti‑goal）概念以及主动预防（prevent）操作，以解决目标否定的语义歧义并支持避让行为的推理；

**💡 创新点**

1) 将避免行为单独建模为反目标，消除目标否定与追求否定事件的混淆；2) 为反目标推理提供与正向推理对应的欲望值（desire‑value）函数选择原则；3) 引入主动预防操作，将反目标与正目标连接起来，支持“为防止某事件而行动”的情景；

**🔧 技术方法**

使用非公理逻辑的推理框架（IL到NAL的转化）、回溯目标推理规则、欲望值函数设计、以及最小化案例实验来验证理论；

**📊 数据集**

使用极简词汇表（light、food、hurt、press）构建四个最小案例（实现、避免、预防、保留），并在模拟环境中执行；

**📈 对比分析**

通过四个案例的行为曲线（累积奖励）验证理论预测；实验显示在实现、避免、预防、保留四种情形下，系统行为与理论一致，未给出数值性能指标，主要是定性符合预期；

**⚠️ 局限性**

1) 只关注理论扩展，未实现动机学习与动态动机更新；2) 仅使用极简案例，缺乏大规模或真实数据验证；3) 对行为对称性与学习复杂度的分析有限；4) 与强化学习等现有方法的系统性对比不足。

---

## 292. Is Deep Research Reliable? Misleading Knowledge Induces False Conclusions

**arXiv ID:** 2607.20891 | [PDF](https://arxiv.org/pdf/2607.20891v1)

**作者:** Pengyu Zhu `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了深度研究代理在长周期工作流程中对误导性知识的可靠性，并提出了 MisKnow‑Agent 框架用于生成可控的误导实例进行评估。

**💡 创新点**

创新点在于设计了能够控制权威级别与文风的误导知识生成机制，并系统分析了时机、来源、框架与 LLM 对误导采纳的影响。

**🔧 技术方法**

采用了 LLM 生成、搜索引导校验、跨模型交叉验证、预/后研究验证策略以及 DeerFlow 与 WebThinker 等典型深度研究框架。

**📊 数据集**

使用了 DeepResearch Benchmark 的 100 个任务以及 MisKnow‑Agent 生成的 5,933 份误导性文档。

**📈 对比分析**

通过 False‑Conclusion Adoption Rate (FCAR) 对比实验，发现单一误导文档即可导致 34%–85% 的误导采纳，预后防御可降低但未完全消除。

**⚠️ 局限性**

局限在于缺乏实时中间状态校验，防御效果模型依赖且组合不一定叠加，以及对闭源系统和更复杂攻击场景的覆盖不足。

---

## 293. Information-Theoretically Secure Aggregation for Lightweight Federated Learning: Resilient to Dropouts and Adversaries

**arXiv ID:** 2607.20890 | [PDF](https://arxiv.org/pdf/2607.20890v1)

**作者:** Hyeong-Gun Joo `[一作]` (Hanyang University), Dong-Joon Shin `[通讯]` (Hanyang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级、信息理论安全的聚合框架，专门针对一比特量化的联邦学习（signSGD-MV），实现单轮安全多项式求值并仅泄露最终多数投票结果；

**💡 创新点**

创新点包括：①通过逆形式指数约简将多数投票多项式的最高次数减半，显著降低乘法深度；②提出两种单轮安全乘法方案，其中单掩码幂共享方法利用多数投票多项式的单基结构，将离线复杂度和存储从指数级降为线性；③利用MDS码错误纠正实现对用户下线和恶意篡改的鲁棒性；

**🔧 技术方法**

技术主要包括信息理论安全共享（t< n/2）、逆形式指数约简、单掩码幂共享与DN+N-BTE两种单轮乘法实现、MDS码解码、以及 Fermat 小定理构造投票多项式；

**📊 数据集**

实验使用 MNIST、FMNIST、CIFAR-10 三个标准数据集，采用不均匀分布的 100 名用户、每轮 14–21 名活跃用户；

**📈 对比分析**

与传统的 signSGD-MV 与 Hi-SAFE（基于Beaver三元组的计算安全）比较，实验表明在高达 35.7% 的下线率和 23.9% 的恶意率下，所提框架保持与无下线/恶意情况相同的模型准确率，同时在线通信量降低至原来的 0.5%~0.8%，延迟降低 80% 以上；

**⚠️ 局限性**

局限性主要在于：①方法 A 的离线复杂度为指数级，限制了大多项式度的可扩展性；②需要在每轮使用全新的随机掩码，若多轮运行效率受限；③隐私保护基于诚实多数假设，在 t 接近 n/2 时鲁棒性下降；

---

## 294. Engine-Native Editable 3D World Reconstruction with Objects and Lighting

**arXiv ID:** 2607.20889 | [PDF](https://arxiv.org/pdf/2607.20889v1)

**作者:** Junhao Chen `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Lumera，一套从单张游戏场景图像解析可编辑的 3D 场景，包括对象盒、网格、可编辑灯光及 HDR 环境的完整管线。

**💡 创新点**

创新点在于将游戏引擎原生的参数化灯光纳入结构化解析任务，构建 Lumera‑2K 数据集并设计基于 SpatialLM 的灯光与盒子解析器。

**🔧 技术方法**

使用的技术包括基于 SpatialLM 的自回归代码生成、SAM3D 网格重建、IntrinsicHDR HDR 估计以及受限的 VIGA‑风格生成器/验证器迭代。

**📊 数据集**

数据集为 Lumera‑2K，包含 2,513 个 UE5 项目，提供 3.73M 组件、63M 物体实例、102.6K 参数化灯光、95.1K 视角。

**📈 对比分析**

与 DetAny3D、SpatialLM、N3D‑VLM、WildDet3D 等基线比较，Lumera‑Box 在 mAP、IoU‑B、F‑score 等指标上提升显著（例如 mAP↑0.1141），而 Lumera‑Light 在非空场景召回率接近 1，灯光定位 F1 在 0.5 m 阈值下为 0.209。

**⚠️ 局限性**

局限性包括盒子精度仍有偏差、光源定位精度不足（尤其是近摄像机或视锥外灯光）、以及对 UE‑风格资产之外的泛化能力有限。

---

## 295. Deep Reinforcement-Learning-Guided Model Predictive Control for Preventing Overtakes in Autonomous Racing

**arXiv ID:** 2607.20973 | [PDF](https://arxiv.org/pdf/2607.20973v1)

**作者:** Yufei Xi `[一作]` (University of Michigan), Tulga Ersal `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种用于自主赛车的防御性阻挡框架，旨在防止更快的对手超车。该框架通过层次化的深度强化学习引导的模型预测控制（MPC）来实现防御性策略。

**💡 创新点**

创新点在于将防御性阻挡问题形式化为空间占用调节问题，并将学习到的防御意图嵌入到非线性MPC中，从而实现动态可行的实时防御控制。

**🔧 技术方法**

使用了深度强化学习（SAC）和非线性模型预测控制（MPC）技术，结合了空间参考生成和动态约束满足。

**📊 数据集**

在Thunderhill West赛道上进行了仿真评估，该赛道在训练中未使用，确保了评估的真实性和有效性。

**📈 对比分析**

与基准控制器相比，提出的框架在多个评估段中显著提高了超车时间，从8.8秒增加到14.6秒，同时有效减少了对手的前进距离，且在动态约束下实现了实时计算，平均求解时间为33.3毫秒。

**⚠️ 局限性**

局限性包括当前评估仅限于仿真，使用了一个保留的真实赛道，并且与没有学习参考的空间包络MPC进行比较。未来的工作应在硬件上验证控制器，并测试多个赛道和对手策略。

---

## 296. AUCH-Net: Action Unit-Based Consistency-Aware Hypergraph Network for Cross-Domain Few-Shot Facial Expression Recognition

**arXiv ID:** 2607.21004 | [PDF](https://arxiv.org/pdf/2607.21004v1)

**作者:** Xinhan Qiu `[一作]` (Xiamen University), Hanzi Wang `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AUCH-Net，利用动作单元一致性超图实现跨域少样本表情识别。

**💡 创新点**

创新点在于将动作单元作为节点构建一致性超图，加入关系一致性损失和AU正则化损失，实现更稳健的特征学习。

**🔧 技术方法**

采用超图神经网络、关系一致性损失、AU正则化、多分支特征融合等技术。

**📊 数据集**

训练使用CK+、MMI、Oulu-CASIA、RAF-DB、SFEW，测试使用CFEE_C、EmotioNet_C、RAF_C。

**📈 对比分析**

与多种基准方法对比，在5-way 1-shot和5-shot任务上取得最高或最接近最高准确率，显著优于现有方法。

**⚠️ 局限性**

仅使用水平翻转作为一致性约束，尚未探索其他变换或更大规模数据。

---

## 297. Naju: A Native Discrete State-Space Model with Independent Retention and Writing for Long-Sequence Memory

**arXiv ID:** 2607.21000 | [PDF](https://arxiv.org/pdf/2607.21000v1)

**作者:** Hyuk Lim `[一作]` (Korea Institute of Energy Technology), Seunghyun Yoon `[通讯]` (Korea Institute of Energy Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 Naju 的原生离散状态空间模型，专门解决长序列记忆任务中既需要保持旧绑定又需要及时覆盖旧绑定的问题。

**💡 创新点**

创新点在于将忘记门直接用作离散极点并独立地设置写门，使保持和写入不再受同一门控耦合，可同时实现高保留和强覆盖；并且直接在离散域参数化，无需从连续时间系统再离散化。

**🔧 技术方法**

技术细节包括：离散状态空间递推 xₙ = fₙ⊙xₙ₋₁ + iₙ⊙(Bₙhₙ)；fₙ、iₙ、Bₙ、Cₙ 通过短深度可分离卷积和线性层自适应生成；关联并行扫描实现线性时间；对递推稳定性提供 BIBO 与衰减记忆分析；使用 SiLU 激活和多门输出调制。

**📊 数据集**

实验数据集涵盖：4 个长序列诊断任务 (T1–T4)；多查询关联召回 (MQAR)；Long‑Range Arena（5 任务）；WikiText‑103 语言建模；并在各任务上进行长度外推测试。

**📈 对比分析**

比较方法：与 Transformer、Mamba、Mamba‑2、xLSTM、GLA、RetNet 等模型在同等参数/训练预算下对照准确率、PPL、吞吐量等指标。Naju 在长距离保持与覆盖的两条争议轴（T2、T4）均居首，WikiText‑103 上获得最低 perplexity；在 LRA 上平均性能略高于同类扫描模型。

**⚠️ 局限性**

局限性：实现依赖于高效的并行扫描，导致相对较高的实现开销；对模型宽度和状态尺寸的初始化与优化敏感；模型参数量比 Mamba 略大；目前尚未在极大规模（>1000 层）或多任务迁移场景下验证。

---

## 298. Geo3R: Mitigating Spatial Reasoning Hallucination in Multimodal Large Language Models

**arXiv ID:** 2607.21085 | [PDF](https://arxiv.org/pdf/2607.21085v1)

**作者:** Mingyu Wang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Geo3R，一个训练-free、plug-and-play 的框架，利用单张图像的几何估计和结构化 3D 证据，提升多模态大语言模型（MLLM）的空间推理能力，减少空间推理幻觉。

**💡 创新点**

创新点在于：①将 2D 视觉信息逐步映射到多空间 3D 坐标系（相机、重力对齐世界、物体局部），②构造结构化“几何卡片”提供多帧几何证据，③无需额外训练即可直接增强任何 MLLM 的空间推理。

**🔧 技术方法**

使用的技术包括：Monocular depth estimation（DepthPro）、相机内参估计（GeoCalib）、语义分割（SAM）、物体姿态估计（Orient Anything V2），以及基于这些结果的后投影、世界坐标变换和局部坐标变换；最后将几何信息封装为卡片并注入模型提示。

**📊 数据集**

评估数据集：ViewSpatial-Bench（VSB）、3DSRBench（3DSR）和 CV-Bench（CVB），共 18 个空间推理子任务（视角、方向、视点）共计 17,493 条样本。

**📈 对比分析**

与多种基线比较：通用 MLLM（LLaVA、Qwen3-VL、Gemini-3-Flash、GPT-5）、专用空间推理模型（SpaceLLaVA、SpatialRGPT、Spatial-SSRL、SenseNova-SI、SpatialThinker）以及现有幻觉缓解方法（OPERA、Tri-HE、Reefknot、AdaptVis、APC-VLM）。Geo3R 在所有基线上均取得显著提升：Gemini-3-Flash 提升 7.06%、Qwen3-VL 提升 10.90%，并在 18 任务中获得最高平均准确率。

**⚠️ 局限性**

局限性：①性能受限于现有几何估计工具的精度，②推理时存在额外的计算开销，③仅在单图像空间推理场景验证，缺乏多视角/视频扩展实验。

---

## 299. C-PTQ: Fisher-weighted Channel-wise Sensitivity for Post-training Quantization of MLLMs

**arXiv ID:** 2607.21076 | [PDF](https://arxiv.org/pdf/2607.21076v1)

**作者:** Jiameng Li `[一作]` (KU Leuven), Matthew B. Blaschko `[通讯]` (KU Leuven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的通道级后训练量化方法（CPTQ），通过将 Fisher 信息嵌入通道级量化误差权重，使量化误差与任务损失的敏感度保持一致。

**💡 创新点**

创新点在于：① 用第二阶导数的 Fisher 信息近似 Hessian，直接量化每个通道对任务损失的贡献；② 对通道级量化误差加权，而非传统的模态或 token 级别；③ 采用对角 Fisher 矩阵，既保留了敏感度信息，又大幅降低了计算与存储开销。

**🔧 技术方法**

技术：后训练量化（PTQ）、通道级缩放（CWS）、Fisher 加权的均方误差目标、对角 Fisher 矩阵近似、激活统计与梯度信息结合、grid search 搜索最优缩放因子。

**📊 数据集**

数据集与模型：Qwen2.5-VL、InternVL2、LLaVA-OV（7B/32B 等规模）量化；校准集采用 128 张 COCO 图像-文本对；评测基准包括 MMMU、VizWiz、SQA、TextVQA、OCRBench、ChartQA、AI2D、DocVQA。

**📈 对比分析**

与 AWQ、MBQ、QIG、MASQuant 等现有 CWS 方法对比，在权重量化 3bit（W3A16）和权重+激活 4bit/6bit（W4A8、W4A6）场景下，CPTQ 取得 SOTA 结果，平均性能提升 1–3% 以上；同时量化与推理速度与 MBQ 相当，显著快于 QIG；在极低比特（如 W4A4）下仍保持相对鲁棒。

**⚠️ 局限性**

局限性：目前仅对 LLM 解码器进行量化，视觉编码器与投影层仍保持全精度；在极低位宽（W4A4 等）下性能下降显著，仍需补偿机制；对角 Fisher 近似虽然效果好，但对跨通道依赖信息的捕捉有限。

---

## 300. Spectral Transformation for Layer-wise Global Rank Discovery in Federated LoRA for Vision Transformers

**arXiv ID:** 2607.21074 | [PDF](https://arxiv.org/pdf/2607.21074v1)

**作者:** Hariharan Ramesh `[一作]` (University of Arizona), Jyotikrishna Dass `[通讯]` (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SpecTraL方法，在联邦学习环境下对Vision Transformers（ViT）进行LoRA微调，解决现有聚合策略导致的跨项误差和高通信成本问题。

**💡 创新点**

创新点在于利用低秩潜在空间中的正交Householder变换进行全局秩发现，并通过随机矩阵理论的尖峰协方差模型自动分离共识信号与非IID噪声，从而无需手动调优秩参数；同时引入填充感知初始化框架，避免在客户端重新合并预训练权重。

**🔧 技术方法**

使用LoRA适配器、正交Householder变换、尖峰协方差模型（Random Matrix Theory）、填充感知初始化技术以及低秩特征空间聚合。

**📊 数据集**

在DomainNet和NICO++两个多域图像数据集上进行实验，针对ViT‑B/16和ViT‑L/16模型。

**📈 对比分析**

与传统平均、拼接、全量权重重构及辅助模型聚合方法相比，SpecTraL在保持或提升准确率的同时显著降低通信开销、减少服务器端计算，并消除了对秩选择的超参数搜索。

**⚠️ 局限性**

局限性包括对极端非IID场景的鲁棒性尚未全面验证，依赖于尖峰协方差模型假设，且在规模更大或更稀疏的客户端设置下可能需要进一步优化。

---

## 301. PrefReward: Learning User Preference Matrix for Personalized Text Generation

**arXiv ID:** 2607.21067 | [PDF](https://arxiv.org/pdf/2607.21067v1)

**作者:** Yue Wu `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练自由的偏好奖励框架PrefReward，先从用户历史文本中抽取显式偏好矩阵，再用KL散度奖励指导LLM解码；

**💡 创新点**

创新点在于将用户偏好显式化为可解释的矩阵，且通过奖励机制在推理阶段实现个性化，无需额外微调或强化学习；

**🔧 技术方法**

采用BM25检索用户代表性历史、token级LLM推断生成偏好向量、KL散度奖励和Best‑of‑N采样解码；

**📊 数据集**

使用LongLaMP数据集进行实验，聚焦长文本个性化生成任务；

**📈 对比分析**

与非个性化、检索增强（RAG）、上下文驱动（CoS）等基线对比，PrefReward在ROUGE‑1、ROUGE‑L和METEOR上均获得最高分，性能提升明显；

**⚠️ 局限性**

局限性包括依赖预定义的偏好标签集、对标签数量敏感、对极端动态风格适应性有限以及在极大文本上下文下的计算成本。

---

## 302. GuidedAttention: Interpretable and Correctable Visual Attention for OOD-Robust Robot Manipulation via Imitation Learning

**arXiv ID:** 2607.21049 | [PDF](https://arxiv.org/pdf/2607.21049v1)

**作者:** Masaki Murooka `[一作]` (CNRS-AIST JRL), Yukiyasu Domae `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 GuidedAttention 框架，利用可解释、可纠正的视觉注意力关键点作为中间表示，将其与扩散策略结合，实现端到端的视觉运动控制。

**💡 创新点**

创新点在于：① 将视觉注意力显式化为关键点，可在执行时一次性由用户纠正并通过跟踪持续使用；② 在扩散策略中以关键点特征或坐标作为条件，提升对空间结构的关注；③ 通过特征对齐损失、随机特征路由等机制保证纠正信息在训练与推理中的一致性。

**🔧 技术方法**

使用的技术包括：DETR 风格的视觉注意力编码器（ResNet-18+Transformer），基于 U‑Net 的 denoising diffusion 运动策略，Co‑Tracker 用于关键点跟踪，随机特征路由、特征空间对齐损失等训练技巧。

**📊 数据集**

实验数据集：在 MuJoCo 仿真环境中收集 Cable、Ring、Particle 三个任务的 30 条遥控演示；在真实机器人 UR5e+RealSense 环境中收集 Cup Insertion、Chain Pick‑and‑Place、Towel Fold 三个任务的 32 条演示；关键点仅在首帧人工标注，后续由跟踪器传播。

**📈 对比分析**

与 DP（全局视觉特征扩散策略）和 ACT（动作块变压器）做对比。实验表明：在 ID 条件下 DP‑GA 提升约 15pp，Pos‑OOD、App‑OOD 条件下提升 20–30pp；若在推理时进行一次性关键点纠正，OOB 成功率可再提升 40–80pp，显著优于两种基线。

**⚠️ 局限性**

局限性：① 仅适用于可用少量关键点充分描述任务的场景，复杂任务可能需要更丰富的注意力表示；② 关键点为 2D 图像坐标，无法直接处理深度信息，遮挡或变形会导致误跟踪；③ 依赖外部跟踪器的性能，跟踪失败会直接影响最终控制效果。

---

## 303. GroupVideo: Multi-Identity Customized Text-to-Video Generation

**arXiv ID:** 2607.21027 | [PDF](https://arxiv.org/pdf/2607.21027v1)

**作者:** Xinyang Song `[一作]` (University of Chinese Academy of Sciences), Zhenan Sun `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 GroupVideo，一种能够在多身份场景下生成自定义文本驱动视频的离线训练框架；

**💡 创新点**

创新点在于多模态身份对齐（视觉与语义双向融合）以及 ID 定位模块（为每个身份提供空间引导），并采用两阶段渐进式训练和边界框/遮罩正则化；

**🔧 技术方法**

主要技术包括 Video Diffusion Transformer (DiT)、VAE 视觉编码、CLIP+Q-Former 语义感知器、Softmax 层生成定位掩码，以及自定义的边界框和遮罩损失；

**📊 数据集**

使用了自建的 2 万高质量多身份视频数据集（1080×1920，2-3 人，完整身体覆盖），并与公开单人数据混合训练；

**📈 对比分析**

与 ConsisID、Ingredients、Concat-ID、MAGREF 等现有方法对比，GroupVideo 在 FaceSim、CLIPScore、FID 等指标上均优于对手，尤其在多身份保真度与自然运动上表现突出；

**⚠️ 局限性**

限制在于对多身份数量的扩展仍有限（实际实验最多 3 人），且对极端遮挡或极度运动的场景效果尚待验证。

---

## 304. HiMe: Real-Time Self-Hosted Personal Agent Platform for Health Insights with Wearable Devices

**arXiv ID:** 2607.21019 | [PDF](https://arxiv.org/pdf/2607.21019v1)

**作者:** Wei Liu `[一作]` (King College London), Yulan He `[通讯]` (King College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个本地部署、隐私优先的健康代理平台HiMe，可将可穿戴设备的连续数据实时分析并生成个性化健康洞察。

**💡 创新点**

通过将数据库视为第一类组件、联合优化效果与效率、实时与长期用户建模三大设计原则，实现了可验证、可扩展且可在本地硬件上持续运行的健康代理。

**🔧 技术方法**

采用LLM代理架构、工具层统一数据库读写、代码执行工具、事实验证、分层设计与多角色代理，并配合本地LLM推理框架vLLM等技术。

**📊 数据集**

在五个公开可穿戴数据集（LifeSnaps、PMData、MMASH、CovIdentify、GLOBEM）上进行回放评测。

**📈 对比分析**

通过终端数据库状态的自动评测（pass@1、召回率、耗时）对22个模型（本地与托管API）在分析、记忆、计划等角色进行比较；强大本地模型可与API媲美，弱模型存在幻觉与性能不足。

**⚠️ 局限性**

多轮可靠性与跨模态报告叙述仍欠佳，小模型在本地部署的可用性受限，实时触发与长时趋势平衡仍需改进。

---

## 305. ADABORD: a novel AdaBoost approach for ordinal classification

**arXiv ID:** 2607.21003 | [PDF](https://arxiv.org/pdf/2607.21003v1)

**作者:** Rafael Ayllón-Gavilán `[一作]` (IMIBIC), Pedro A. Gutiérrez `[通讯]` (Universidad de Córdoba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对有序分类（Ordinal Classification）的AdaBoost框架（称为Adabord），通过改进基学习器和误差函数来充分利用类别的序列信息。

**💡 创新点**

创新点在于：①使用Ordinal Gini（OGini）划分准则的序数决策树作为弱学习器；②采用绝对排名概率得分（aRPS）作为误差函数，以距离敏感的方式衡量预测与真实的累计分布差异。

**🔧 技术方法**

技术实现基于AdaBoost的多分类扩展SAMME，集成OGini决策树与aRPS误差；实验中还比较了传统AdaBoost、OEAdaBoost、BP-MLP、BP-MLP-CLM、Ridge、XGBoost、O-LGBM等方法。

**📊 数据集**

实验使用了TOC-UCO仓库中46个大规模有序分类数据集，涵盖多种领域和不同类别数。

**📈 对比分析**

与七种基线方法比较时，Adabord在多数有序指标（AMAE、MMAE、QWK）上表现最佳，尤其在类别数≥5的子集上显著优于其他方法；在非有序指标BACC上，XGBoost略胜。

**⚠️ 局限性**

局限性包括：对极端类别不平衡的鲁棒性仍待提升；在非有序或序数信息不明显的任务中，基于序数的误差函数和基学习器可能不如传统方法表现。

---

## 306. Improving Communication of Changes in Model-Based Engineering with Model-Independent Change Descriptions

**arXiv ID:** 2607.21084 | [PDF](https://arxiv.org/pdf/2607.21084v1)

**作者:** Philip Ochs `[一作]` (Karlsruhe Institute of Technology), Ina Schaefer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了将正式变更模型映射为模型独立语言的方法，以便跨学科团队理解模型变更。

**💡 创新点**

创新点在于提供一种自动化、可解释的变更描述框架，桥接正式变更与自然语言描述之间的差距。

**🔧 技术方法**

使用Delta Modeling、EMF、MOF以及Eclipse Modeling Tools实现框架。

**📊 数据集**

采用BCS案例研究（Body Comfort System）作为数据集进行评估。

**📈 对比分析**

通过混合方法评估，量化技术可行性、实施框架以及案例研究中的技术适用性；在用户研究中定性评估可行性、实用性和可扩展性；总体性能表现为可行且具有潜力。

**⚠️ 局限性**

局限性包括仅在单一案例中验证，缺乏大规模多领域实验，且框架对复杂模型的扩展性尚待进一步研究。

---

## 307. Maintenance Signals in AI-Assisted GitHub Repositories: Evidence from GenAI Adopters

**arXiv ID:** 2607.21079 | [PDF](https://arxiv.org/pdf/2607.21079v1)

**作者:** Rikuto Tsuchida `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对公开声明采用生成式 AI 的 GitHub 开发者进行研究，采集并对比 179 个可见 AI 助手配置的仓库与 179 个匹配的传统仓库，分析 README 结构与维护 issue 以探测 AI 帮助带来的维护成本变化。

**💡 创新点**

首次系统性将可见 AI 助手使用与仓库级别特征、文档质量及 issue 行为相结合，揭示 AI 助手更多聚焦于外部依赖管理、文档结构化而非改变仓库类型，填补了从工具性能向实际开发流程转化的研究空白。

**🔧 技术方法**

使用 GitHub REST/GraphQL API 抓取数据，Hungarian 算法实现仓库匹配，Zanartu 等分类模型对仓库类型进行自动化划分，Mann‑Whitney U 与卡方检验评估差异；同时手工标注 issue 的类型与影响领域。

**📊 数据集**

公开 GitHub 用户资料、仓库元数据与 issue 数据，共计 622 位自报采用者、179 个 AI 助手仓库、179 个传统仓库、248 条作者创建的 issue；全部数据通过 Zenodo 公开重现包提供。

**📈 对比分析**

通过提交量相匹配的对照组进行 README 量化比较，发现 AI 助手仓库 README 更长、H1 头与代码块密度显著提高，URL 密度下降；issue 分析显示 AI 相关 issue 更集中于外部依赖（API 限流等），且增强需求占比高于 bug，整体表现为维护重心从代码生成转向外部依赖与行为验证。

**⚠️ 局限性**

样本仅限公开自报使用者，关键词检索可能漏掉使用者或使用不同词汇者；只捕捉到配置文件中的 AI 使用，未覆盖纯聊天式助手；issue 归类由单一标注者完成，缺乏交叉验证；仓库匹配仅基于提交数，未考虑团队规模或项目领域差异。

---

## 308. TransBiolab: A Real-World Multi-View Dataset of Cluttered Transparent Biomedical Objects

**arXiv ID:** 2607.21071 | [PDF](https://arxiv.org/pdf/2607.21071v1)

**作者:** Ke Ma `[一作]` (Huazhong University of Science and Technology), Tian Xia `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并公开了 TransBiolab 数据集，包含 98 个实验室场景的 161,315 帧 RGB‑D 图像，提供 15 种透明生物医学实验器皿的 1.03M 实例标注，包括 6D 位姿、完整与可见掩码、深度以及每帧相机标定。

**💡 创新点**

创新点在于：① 针对透明塑料实验器皿的真实实验室多视角、多人物、重叠混乱场景提供大规模标注；② 按类别、物体数量和视角三轴组织数据，便于研究操作难度；③ 设计了多任务基准（分割、深度估计/补全、6D 位姿）和完整的机器人操控评估。

**🔧 技术方法**

使用了 Intel RealSense D435i 与 Franka Panda 机器人进行多视角捕获，采用 ORB‑SLAM3 估计相机轨迹、KinectFusion 风格点云重建和 Blender‑ProgressLabeller 进行序列级多视角标注；在评测中分别使用 SAM‑3、TransLab、Depth Anything 3、ClearGrasp、FoundationPose 与 MegaPose‑6D 等开源模型。

**📊 数据集**

数据集以自身的 TransBiolab 为主，评测对比了 KeyPose、StereOBJ‑1M、ClearPose、TransCG、Trans10K‑v2 等公开透明物体数据集，并在 TransBiolab 的 hold‑out 10 个真实实验室场景（TBiolab‑HO）上进一步验证。

**📈 对比分析**

在基准任务中，SAM‑3 的平均 mIoU 为 0.685，TransLab 低于其；深度估计器 Depth Anything 3 在 TransBiolab 上的 AbsRel、RMSE 及 MAE 均高于 ClearPose，表明透明物体难度更大；6D 位姿方面，FoundationPose 的 ADD‑S AUC 约 80.8%，在不同视角、类别与堆叠复杂度下表现波动；机器人实验显示并联抓手成功率 65.3%，多指手 56.7%，证明数据集能驱动完整 perception‑to‑action 流程，但仍有显著误差。

**⚠️ 局限性**

局限性包括：① 透明物体的光学特性导致标注与模型表现仍存在显著差距；② 依赖手工多视角标注流程，标注成本高；③ 机器人抓取成功率受抓取策略与控制精度限制，难以仅归因于感知错误；④ 数据集主要覆盖实验室常见器皿，可能不足以涵盖所有实验流程与环境变化。

---

## 309. Counterfactual Explainability Framework With CycleGAN And Counterfactual-Classifier Alignnment Score for Retinal Disease Classification

**arXiv ID:** 2607.21068 | [PDF](https://arxiv.org/pdf/2607.21068v1)

**作者:** Kritanu Chattopadhyay `[一作]` (National Institute of Technology Durgapur), Soumya Chatterjee `[通讯]` (National Institute of Technology Durgapur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 CounterFundus 框架，结合 EfficientNet-B5 分类器、CycleGAN 生成病变到正常的对抗样本，并通过 CCAS 指标评估解释空间一致性。

**💡 创新点**

创新点在于：①将 Counterfactual 生成与 XAI 评估结合；②设计 CCAS（Spearman、IoU、指点准确度）综合指标；③利用 CCAS 过滤的对抗样本提升分类性能。

**🔧 技术方法**

采用 EfficientNet‑B5、CycleGAN（ResNet‑based 生成器与 PatchGAN 判别器）、EigenCAM、Spearman、IoU、指点准确度、SSIM、PSNR、FID 等技术。

**📊 数据集**

使用 4 类（正常、糖尿病视网膜病变、青光眼、白内障）共 4217 张公开视网膜图像数据集；外部验证使用 RFMiD 数据集。

**📈 对比分析**

在 5‑fold CV 与锁定测试集上，模型准确率 95.2%±0.69，AUC 99.17%；通过 CCAS 过滤对抗样本提升准确率至 97.99%，仅使用 72% 语料。与 GradCAM、GradCAM++、LIME 等解释方法对比，EigenCAM 在 CCAS 评估中表现最佳。

**⚠️ 局限性**

局限性包括：对抗样本生成质量仍受训练域限制；CCAS 仅衡量空间一致性，无法覆盖所有临床特征；未对不同成像模态（如 OCT、皮肤镜、胸片）进行验证。

---

## 310. Do Pathology Vision-Language Models Truly See Pathology?

**arXiv ID:** 2607.21065 | [PDF](https://arxiv.org/pdf/2607.21065v1)

**作者:** Chengyang Zhang `[一作]` (Sichuan University), Jiancheng Lv `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 PathBind 基准，用三类样本（VQA、教学图谱VQA、区域定位）评估病理视觉语言模型在视觉依赖、答案准确性和实体级视觉绑定三方面的能力。

**💡 创新点**

创新点在于：①从答案准确度拆分出视觉必要性和多模态增益；②揭示“领域训练幻觉”，即专属训练提升答案却不一定提升视觉绑定；③设计了三维度过滤与专家复核流程，确保样本真正依赖图像与实体定位。

**🔧 技术方法**

采用大规模视觉语言模型（Gemini‑3‑Pro、GPT‑5.4、PathGen‑LLaVA、Patho‑R1‑7B 等）与通用 VLM 进行对照实验，并使用注意力权重与 IoU、Precision、Recall 等指标评估视觉绑定。

**📊 数据集**

使用公开病理 VQA 公开数据（PathMMU、Path‑VQA、OmniMed‑Bright、Quilt‑VQA、MedXpert‑Path）、PathVG 定位数据以及私人病理教学图谱（PathPTA）共 2,600 条样本。

**📈 对比分析**

通过与传统公开 VQA 公开数据的对比，展示了在 PathBind 上模型的多模态增益更大，表明样本过滤提升了视觉依赖；同时，尽管大多数模型在答案准确率上表现强劲，但在 IoU、Precision 仍低，说明视觉绑定仍薄弱。

**⚠️ 局限性**

局限性包括：①样本仍受文本先验影响，难以完全消除语言偏差；②仅评估了注意力映射的定位能力，未考察更细粒度的推理过程；③对某些模型缺乏完整的注意力可视化与解释，导致难以深入分析根本原因。

---

## 311. QuantiBias: Benchmarking Quantization-Induced Bias in LLMs

**arXiv ID:** 2607.21063 | [PDF](https://arxiv.org/pdf/2607.21063v1)

**作者:** Emilio Ferrara `[一作]` `[通讯]` (University of Southern California), Emilio Ferrara (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究量化后大语言模型在开放式生成中的偏见提升，提出QuantiBias评测基准；

**💡 创新点**

发现量化导致的“选择性安全缺口”，即短表面安全测试通过但开放式生成偏见显著升高；并提供多语言并行生成偏见探测与严重性评估工具；

**🔧 技术方法**

采用有效比特/权重测算、量化精度评估、推理前后对比、并行翻译生成、独立评判员与内容严重度等级评分等技术；

**📊 数据集**

使用Qwen3.6‑27B和Gemma‑4‑31B两大骨干；MBTP（320条平行生成偏见），BBQ、XSTest、StrongREJECT、CEB、StereoSet、MMLU‑Redux、GSM8K等多套基准；

**📈 对比分析**

以“有效比特/权重”为轴比较不同量化等级；短表面安全指标（拒绝率、过度拒绝、BBQ）保持不变；开放式生成偏见率在约25%–27%之间，显著高于全精度；推理模式在Qwen上可将偏见率减半，Gemma无效；整体偏见上升但安全保持；

**⚠️ 局限性**

局限性包括：评判员的主观性与样本量不足、未考虑激活精度、仅关注快速路径、语言/内容耦合导致跨语言比较受限，以及对量化导致的严重度提升尚未测量。

---

## 312. GeoThreat: Transferable Targeted Adversarial Attacks on Large Vision-Language Models for Remote Sensing Image Interpretation

**arXiv ID:** 2607.21036 | [PDF](https://arxiv.org/pdf/2607.21036v1)

**作者:** Yimin Fu `[一作]` (Hong Kong Baptist University), Michael K. Ng `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoThreat，一种针对遥感图像解释的可迁移有目标的对抗攻击方法

**💡 创新点**

通过联合调节全局概念表示和局部感知表示，采用协同重要性估计与跨注意力感知适配来克服遥感图像全局‑局部推理带来的可迁移性与可控性挑战

**🔧 技术方法**

使用 CLIP 视觉编码器集合进行对抗样本生成，利用注意力与梯度敏感度融合的协同重要性估计，跨注意力对齐感知补丁，迭代更新 ℓ∞ 约束扰动，并采用模型集成优化

**📊 数据集**

在 UCM、SIRI‑WHU 和 AID 三个遥感数据集上进行图像描述和分类任务评估

**📈 对比分析**

与 AttackVLM、AdvDiffVLM、AnyAttack、SSA‑CWA、M‑Attack、VEAttack、FOA‑Attack、V‑Attack 等八种现有 LVLM 攻击方法比较，GeoThreat 在所有被试 LVLM 上平均攻击成功率达 88% 以上，语义相似度 AvgSim 达 0.73，显著优于基线 20–30%

**⚠️ 局限性**

依赖多个 surrogate 模型与大计算开销，需手动调参（如 λ、ρ），仅在三类数据集上验证，对真正闭源模型或防御环境的鲁棒性仍待进一步研究

---

## 313. VibeVoice-ASR-BitNet Technical Report

**arXiv ID:** 2607.21075 | [PDF](https://arxiv.org/pdf/2607.21075v1)

**作者:** Songchen Xu `[一作]`, Furu Wei `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对边缘 CPU 的实时语音识别，提出了压缩版本 VibeVoice-ASR-BitNet，并在其上实现了异构量化和 SIMD 优化。

**💡 创新点**

创新点包括：① 对 VAE 分词器使用全流程 INT8 (I8_S) 量化，配合核融合和 SIMD 优化；② 对自回归语言模型使用 BitNet 三元量化 (I2_S)，大幅压缩权重；③ 引入渐进式量化感知训练 (α‑blending) 以稳定收敛；④ 设计统一的 SIMD GEMM 内核，支持 I8_S 与 I2_S；⑤ 在同等模型体积下实现 RTF<1，显著快于 Whisper.cpp。

**🔧 技术方法**

使用技术包括：异构量化（I8_S 与 I2_S）、渐进式 QAT、SIMD 矩阵乘加内核、operator fusion、ggml 框架自定义算子、ARM/NVIDIA AVX2/NEON 优化。

**📊 数据集**

数据集：Multilingual LibriSpeech Corpus（6 种语言）、AISHELL4、AMI、AliMeeting、Fleurs、LibriSpeech、VoxPopuli 等多种读音、会议与多语种测试集。

**📈 对比分析**

通过在相同硬件（AMD EPYC 7V13）上与 Whisper.cpp 进行对比，VibeVoice-ASR-BitNet 在 1.6 GB 模型下，线程数 1–8 时均比 Whisper.cpp 快 1.55–2.28 倍；在 3 线程时实现 RTF<1；准确率仅比 FP16 低 1–4% WER，保持多语种识别能力。

**⚠️ 局限性**

局限性：仅在离线批处理模式下测试；不支持流式实时解码；异构量化方案只在 VibeVoice-ASR 体系结构验证，是否适用于 Whisper、Qwen‑Audio 等其他 VAE‑LM 模型尚未探究。

---

## 314. From Evaluation to Optimisation: Hierarchy-Aware Training Signals for CWE Prediction in Python

**arXiv ID:** 2607.21069 | [PDF](https://arxiv.org/pdf/2607.21069v1)

**作者:** Muntasir Adnan `[一作]` (University of Canberra), Carlos C. N. Kuhn `[通讯]` (University of Canberra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 ALPHA 处罚能否作为直接训练信号，并比较三种交付方式（SFT、双头分类、强化学习）在分布漂移下对 CWE 预测的效果。

**💡 创新点**

证明 ALPHA 处罚在 GRPO 强化学习框架下可显著提升分布漂移性能（最高 27.9% 罚分下降），并系统对比三种交付方式，揭示仅直接使用 RL 能够克服监督训练中的模式崩溃、词汇限制和标签泄露问题。

**🔧 技术方法**

使用 PEFT（LoRA）对 Qwen2.5-Coder-7B 进行适配；双头架构（生成解释 + 分类头）；GRPO 强化学习（无价值网络，基于组相对优势的优势估计）；KL 正则化；对 ALPHA 处罚的梯度分析；平均/末端池化。

**📊 数据集**

SecurityEval（121 个 Python 函数，69 CWE），SVEN（342 样本，4 主 CWE）用于 OOD 测试，CVEfixes（964 样本，CWE 918/352/444）用于补充 RL 训练数据。

**📈 对比分析**

在分布匹配和 OOD 两种设置下，分别对零射击基线、SFT、双头分类器、GRPO 进行累计 ALPHA 罚分评估；SFT/双头在 OOD 下退化至低于零射击基线，而 GRPO 在默认配置下提升 8.3%，最优配置提升 27.9%，与 32B 模型性能相当；通过 Welch t 检验 p=0.005 验证显著性。

**⚠️ 局限性**

主要限制：训练数据极其稀疏且类别不平衡；实验仅在 Python 函数级别，未验证对其他语言或更高粒度的适用性；RL 结果基于单一随机种子，稳定性待进一步验证；未单独评估奖励结构对性能的独立贡献。

---

## 315. Achieving Text-based Person Retrieval with Any Granularity

**arXiv ID:** 2607.21057 | [PDF](https://arxiv.org/pdf/2607.21057v1)

**作者:** Jialong Zuo `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了文本检索的任意粒度（Any Granularity）人检索任务，并构建了多粒度数据集 UFine6926-MG 与评测基准 MG-Eval。

**💡 创新点**

核心创新包括：① 5层粒度谱与多粒度文本标注引擎；② 兼容粗细描述的评测协议与交叉身份标签；③ CMAM 框架，包含正交专家视觉感知、概率跨身份对齐以及粒度一致推理三大模块。

**🔧 技术方法**

技术手段主要包括：多模态 Transformer、结构异构正交专家网络、属性重叠软标签概率对齐、跨粒度一致性监督、以及 mSD 等连续相似度评测指标。

**📊 数据集**

使用的数据集：UFine6926-MG（基于 UFine6926 的多粒度扩充）、CUHK-PEDES-MG、ICFG-PEDES-MG 以及 MG-Eval（含交叉身份标签）。

**📈 对比分析**

与现有 SOTA 方法（如 CLIP、IRRA、TBPS、FGCLIP 等）在标准单粒度基准和 MG-Eval 的分离/进展评测中均实现了显著提升；在任意粒度检索中，CMAM 在所有粒度层面均保持最高的 R@1、mAP 和 mSD。

**⚠️ 局限性**

主要局限包括：对属性注释的依赖仍需人工校验；粒度级别定义虽经验性强，但缺乏严格信息论或概率理论的优化；对极低粒度查询的鲁棒性仍有提升空间。

---

## 316. Discrete Truthful Heterogeneous Two-Facility Location: The Line and Beyond

**arXiv ID:** 2607.21046 | [PDF](https://arxiv.org/pdf/2607.21046v1)

**作者:** Panagiotis Kanellopoulos `[一作]` (University of Essex), Alexandros A. Voudouris `[通讯]` (University of Essex)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在离散异质两设施定位问题上，针对线图实现了最优的4/3近似策略不变机制，并在任意连通图上设计了2近似的确定性策略不变机制；同时证明在K_{1,3}（爪形图）上无可行机制能突破3/2的下界；进一步通过构造局部覆盖与最大独立集的域划分，完成了完整的策略不变性与近似分析；

**💡 创新点**

1) 在线图上首次实现了与已知下界完全匹配的4/3近似；2) 引入“Parity‑Median”与“局部覆盖”技术，统一处理不同规模实例；3) 对一般图引入基于最大独立集的两域划分机制，保证设施位于不同支配集，从而实现2近似；4) 通过爪形图的严格证明，揭示线图结果并不在更复杂图中延伸，给出了3/2的下界；

**🔧 技术方法**

采用固定奇偶性划分、受限中位数规则、局部覆盖与最大独立集扩展、支配集性质、受限中位数单调性等近似机制设计与证明技术；

**📊 数据集**

无实验数据集，全部为理论证明与构造例子；

**📈 对比分析**

与已知的4/3下界和17/4上界对比，线图上实现了最优；在一般图上与先前的2近似或更差结果对比，提供了上界与下界的紧界；

**⚠️ 局限性**

1) 对一般连通图的上界仍是2，下界仅为3/2，仍有较大间隙；2) 机制依赖于最大独立集的构造，可能在实际应用中需要额外计算；3) 仅考虑两设施和所有代理至少批准一设施的场景，未涵盖更大规模或更复杂偏好。

---

## 317. Regularized Optimization on Grassmann Manifold: Theory, Algorithm and Applications

**arXiv ID:** 2607.21039 | [PDF](https://arxiv.org/pdf/2607.21039v1)

**作者:** Zhuan Liang `[一作]` (Beijing Normal University), Zheng Zhai `[通讯]` (Beijing Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对社区检测和聚类问题，提出了一种正则化投影矩阵逼近（RPMA）框架，直接在投影矩阵流形（Grassmann流形）上优化以获取鲁棒的低秩投影矩阵，并设计了梯度投影与Cayley-SM转化的 Riemannian 迭代算法。

**💡 创新点**

创新点包括：
- 在投影矩阵上加入可控的条目级正则化（如 Huber、非负、行和约束），实现稀疏、可解释的投影估计；
- 在 Grassmann 流形上给出一阶二阶最优性条件和局部稳定性分析，证明正则化参数足够小可保持全局最优点的唯一性与局部最优性；
- 设计了无须频繁特征分解的 Cayley‑SMW 迭代方法，显著降低每步复杂度。

**🔧 技术方法**

主要技术：
- Riemannian 优化（梯度投影、Cayley 转化、线搜索）;
- 线性代数工具（Sylvester 方程、Sherman‑Morrison‑Woodbury 公式）;
- 正则化理论与谱间隙分析；
- 合成噪声数据与真实图像相似度矩阵构造。

**📊 数据集**

实验数据集：COIL20、AT&T Faces、Semeion、DIGIT‑10（均为图像/手写数字数据集），以及人工生成的 40×40 块对角投影矩阵作为合成测试。

**📈 对比分析**

比较方法：SDP‑1、SDP‑2、SDP‑U（放宽等容量约束）、Spectral Projection、SLSA。结果显示：RPMA‑NS 在 AT&T Faces 与 Semeion 上获得最优 ACC/NMI/ARI；RPMA‑S 在 COIL20、DIGIT‑10 上与 Spectral Projection 相比提升约 2–5% 的聚类指标；在不平衡采样实验中，RPMA‑S 与 RPMA‑NS 的性能保持稳定，而 SDP‑U 随采样比例增加性能急剧下降。

**⚠️ 局限性**

局限性：
- 正则化参数 λ、δ 需要手动调节，理论中仅保证在“足够小”的范围内；
- 对谱间隙 η_K 的依赖使得在高噪声或近似重复特征值时可能失效；
- 目前仅在投影矩阵流形上考虑正则化，未探讨更一般的稀疏/低秩约束；
- 算法收敛到局部临界点，缺乏全局最优性保证；
- 在非常大规模图或极稀疏数据时仍需要进一步加速。

---

## 318. Spectral-Spatial Synergistic Guided Network for Hyperspectral Salient Object Detection

**arXiv ID:** 2607.21032 | [PDF](https://arxiv.org/pdf/2607.21032v1)

**作者:** Yanyan Peng `[一作]` (Beijing Institute of Technology), Jianan Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级光谱-空间协同引导网络（S3GNet），用于高光谱显著目标检测。

**💡 创新点**

创新点包括：①Spectral Structure-Aware Module（SSAM）通过光谱一阶导数与超像素聚类提取鲁棒光谱特征；②Stream-Aware Attention Module（SAAM）融合加权相关注意力（WCA）与耦合增强注意力（CEA），实现光谱与空间跨流协同；③Progressive Gated Refinement Decoder（PGRD）采用门控机制和渐进式上采样精细恢复目标边界。

**🔧 技术方法**

使用的技术有：光谱一阶导数、超像素聚类、加权相关注意力、耦合增强注意力、门控细化模块、渐进式上采样、混合 BCE+IoU 损失、PyTorch 框架、预训练 ResNet-50 主干网络。

**📊 数据集**

使用的数据集包括 HSOD-BIT-V2、HS-SOD、以及 RGB‑T VT821/VT1000/VT5000，用于评估高光谱与跨模态显著检测。

**📈 对比分析**

与多种 RGB/HSI 基线方法及 Hyper‑HRNet 等最先进方法比较，S3GNet 在 MAE、F_beta、E_measure、S_measure 等指标上均超过对手，提升约 0.7%/10.8%；参数仅 9.74M，FLOPs 9.12G，FPS 137，兼具实时性与高精度。

**⚠️ 局限性**

局限性：在极其细腻或空洞结构的目标上仍易出现漏检，且训练样本量有限导致对极端场景的泛化能力不足。未来计划引入多尺度结构感知模块提升对复杂边界的捕捉能力。

---

## 319. ZONDA: Zero-shot Object Navigation with Dynamic Avoidance in Multi-floor Environments

**arXiv ID:** 2607.21025 | [PDF](https://arxiv.org/pdf/2607.21025v1)

**作者:** Shaomin Liang `[一作]` (Southern University of Science and Technology), Shiyao Zhang `[通讯]` (Great Bay University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了ZONDA框架，实现在多楼层、动态障碍物环境下的零样本目标导航。

**💡 创新点**

创新点包括无平台特定RL控制的启发式多楼层规划、跨视角目标验证以及预测性行人回避。

**🔧 技术方法**

采用了预训练的VLM（Qwen3-VL）、LLM推理、基于高度差的可通行地图、Kalman滤波预测行人轨迹等技术。

**📊 数据集**

在HM3D、MP3D以及自制的HM3D-DYNA动态数据集上进行评估，并在真实TITA机器人上验证。

**📈 对比分析**

相较于现有零样本方法ASCENT，ZONDA在HM3D和MP3D上的成功率分别提升至66.5%/48.2%，在动态数据集上成功率提升至48.8%，并保持较高的SPL。

**⚠️ 局限性**

局限性在于仍依赖高质量的VLM和传感器精度，对极端动态环境或多目标场景的鲁棒性待进一步验证。

---

## 320. WAT3R: Feedforward Underwater 3D Reconstruction

**arXiv ID:** 2607.21023 | [PDF](https://arxiv.org/pdf/2607.21023v1)

**作者:** Jiayi Xu `[一作]` (Hong Kong University of Science and Technology), Sai-Kit Yeung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了WAT3R，一种面向水下环境的端到端三维重建框架，能直接从水下视频中一次前向推断像素对齐的三维点图与相机姿态。

**💡 创新点**

创新点在于：① 将水下成像物理模型（UIFM）嵌入自监督的降解适配模块，使网络在不依赖海量真实三维标注的情况下，利用光学衰减与散射特性提升几何估计；② 采用两阶段训练：先用仿真水下数据进行全监督适配，再在真实水下序列上进行自监督多视角一致性微调。

**🔧 技术方法**

核心技术包括：轻量化的神经降解适配模块、全景光学参数回归头、基于ViT的密集预测转换器（DPT）进行三维点预测、物理一致性重建损失、自动遮挡掩蔽的自监督重投影损失。

**📊 数据集**

训练与评估使用了FLSea、SQUID、USOD10K三大水下数据集，合成数据基于Jerlov水体模型与ULAP光照模型。

**📈 对比分析**

与VGGT、Pi3、DA3、MapAnything等基线相比，WAT3R在多视角与单目深度估计中均获得更低的Abs Rel与更高的δ<1.25精度，在相机姿态估计中RPE_rot下降超过20%，总体表现优于现有方法。

**⚠️ 局限性**

局限性包括：对极端浑浊或快速动态场景的适应性仍有限；自监督阶段对光照与运动假设敏感，可能导致尺度漂移；缺乏真实三维标注的评估仍依赖合成或伪深度，可能影响真实世界泛化。

---

## 321. Reexamining zero-shot summarization: Empirical investigation of trustworthiness of LLM-summarizers

**arXiv ID:** 2607.21010 | [PDF](https://arxiv.org/pdf/2607.21010v1)

**作者:** Vasudha Bhatnagar `[一作]` (University of Delhi), Raj Kumari Bahl `[通讯]` (University of Delhi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于多次生成结果的稳定性评估协议，用以量化LLM摘要器的可靠性。

**💡 创新点**

创新点在于首次将稳定性指标（稳定系数、置信区间）纳入LLM摘要器的评测框架，并通过多维统计方法系统评估。

**🔧 技术方法**

采用BERTScore、AlignScore、语义一致性（SC）三种自动评估指标，计算摘要的语义与事实一致性，并用统计检验（Gini、CV、Levene、Friedman-Nemenyi）分析变异。

**📊 数据集**

使用CNN/DM新闻、PubMed医学、IN‑Abs法律三大语料库的样本进行实验，分别生成100条摘要以构建稳定性评估。

**📈 对比分析**

在实验中对Gemma3、Llama-3.2与Qwen-2.5三大LLM摘要器进行对比，结果显示Gemma3在BERTScore稳定性和整体排名上最优，Llama稳定性最低，AlignScore整体表现最差。

**⚠️ 局限性**

局限性包括仅覆盖三类文档和小规模LLM，未考虑专家评估，且多次生成所需计算成本高，未来需扩展至更多领域与大模型。

---

## 322. Explainable Deepfake Detection Challenge

**arXiv ID:** 2607.21007 | [PDF](https://arxiv.org/pdf/2607.21007v1)

**作者:** Abhijeet Narang `[一作]` (Monash University), Abhinav Dhall `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并举办了Explainable Deepfake Detection Challenge，旨在同时评估图像真实性分类与针对不同用户的可解释性自然语言说明生成。

**💡 创新点**

创新点在于提出多维度评估体系（宏F1、BERTScore、可读性、LLM实体/证据对齐），并将检测与解释任务结合，推出技术级与简化级两种解释，以满足技术专家与普通用户的需求。

**🔧 技术方法**

采用了视觉‑语言模型（Qwen3‑VL、InternVL），对其进行LoRA微调；使用DINOv3等视觉检测器；并在解释生成中加入偏好学习、奖励机制、对抗训练和强化学习等技术。

**📊 数据集**

使用了百万规模的XPlainVerse数据集，该数据集包含真实与多类型伪造图像，并配有参考的技术级与简化级自然语言说明。

**📈 对比分析**

评估方式将宏F1、BERTScore‑F1、SLE以及LLM基准的实体与证据对齐评分进行加权，最终得到综合分数；前五名团队的最终分数分别为0.7612、0.7549、0.7456、0.7378、0.7249，显示检测与解释性能并不完全同步。

**⚠️ 局限性**

局限性包括：评估仅使用英文说明，缺乏多语言支持；LLM基准的实体/证据评分可能受模型误差影响；数据集未覆盖所有伪造手法，对抗鲁棒性和分布外泛化仍待提升。

---

## 323. Weight-norm Criticality: A Mechanism for Loss Spikes Induced by the Normalization and Weight Decay

**arXiv ID:** 2607.21005 | [PDF](https://arxiv.org/pdf/2607.21005v1)

**作者:** Xiaolong Li `[一作]` (Shanghai Jiao Tong University), Zhi-Qin John Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究权重衰减与归一化交互导致的权重范数临界性，从而解释训练中的损失尖峰现象。

**💡 创新点**

提出“权重范数临界性”概念，并给出每个尺度不变层的临界阈值，可预测并定位损失尖峰，填补了仅关注学习率临界的空白。

**🔧 技术方法**

理论分析基于尺度不变性与Hessian齐次性，实验采用梯度下降/AdamW、Hessian谱分析、PCA可视化以及对权重范数与尖峰阈值的动态跟踪。

**📊 数据集**

使用Transformer（LLaMA‑style 187M参数）、ResNet‑50 (CIFAR‑100)、MNIST 以及合成回归数据。

**📈 对比分析**

对比不同权重衰减系数以及是否对MLP参数启用衰减，观察训练损失曲线与尖峰频率；实验表明禁用MLP衰减能显著减少尖峰并使损失更低，验证了理论预测。

**⚠️ 局限性**

只关注尺度不变层，无法完全解释全局非尺度不变网络的所有不稳定；理论假设在实际训练中需近似估计每层Hessian，计算成本较高。

---

## 324. STLSat---An Improved Tableau for Satisfiability Checking of Signal Temporal Logic Formulas

**arXiv ID:** 2607.21081 | [PDF](https://arxiv.org/pdf/2607.21081v1)

**作者:** Marco Zamponi `[一作]` (IMT Lucca), Michele Chiari `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个基于树状表格的 STL 满足性检查方法，修正了先前算法的不一致性问题，并集成到开源 Rust 工具中。

**💡 创新点**

创新点在于重新设计跳跃规则，保证了表格的完整性与可靠性，并提供了无冲突子集（unsat core）提取。

**🔧 技术方法**

使用了树状表格求解、优化的 FOL 编码和定量无量化 SMT 编码三种求解器，并通过并行执行实现组合求解。

**📊 数据集**

使用了自研的 MLTL/STL benchmark 集合，包括 NASA‑Boeing、随机生成公式以及文献中收集的需求。

**📈 对比分析**

通过与 STLTree、MLTLSAT、SMT 等现有工具比较，证明该工具在多数实例上速度最快，组合求解器整体性能最佳。

**⚠️ 局限性**

局限性包括仅支持有界离散时间 STL，未实现无界或连续时间符号；对复杂时间窗的可扩展性待进一步验证。

---

## 325. MVEI & EmObserver: Empowering MLLM-Oriented Visual Emotional Intelligence via Emotion Statement Judgement

**arXiv ID:** 2607.21061 | [PDF](https://arxiv.org/pdf/2607.21061v1)

**作者:** Daiqing Wu `[一作]` (Tsinghua University), Sicheng Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Emotion Statement Judgement（ESJ）框架，构建 INSETS 注释管线，生成 INSETS‑462k 数据集与 MVEI 基准，并基于 ESJ 训练 Emotion‑Oriented MLLM EmObserver。

**💡 创新点**

ESJ 通过声明验证形式兼顾输入自由度与输出确定性；INSETS 采用多模态 LLM 结合分层情感词典实现大规模开放词汇标签与声明构造；MVEI 作为全面评价维度的基准；EmObserver 采用四阶段 RL+自蒸馏优化。

**🔧 技术方法**

利用多模态大语言模型、Group Relative Policy Optimization、On‑Policy Self‑Distillation、LLM‑as‑Judge 一致性奖励、Parrott 情感层级模型等技术。

**📊 数据集**

以 EmoSet 作为图像来源，生成 INSETS‑462k 与经过人工精炼的 MVEI，评估时使用 EEmo‑Bench 与 VECBench。

**📈 对比分析**

与 24 大中型 MLLM 及 4 个情感导向模型在 MVEI 上进行对照，EmObserver 获得 86.23% 总准确率，EEmo‑Bench 71.7% 及 VECBench 63.4%，显著领先同类方法。

**⚠️ 局限性**

与人类对比仍有差距，部分错误仍集中于感知与偏差，且模型对情感语义的把握仍受训练数据与 LLM 规模的限制。

---

## 326. Human-Inspired Framework for Robotic Craniotomy: Integrating Multimodal Fusion and Adaptive Trajectory Adjustment

**arXiv ID:** 2607.21058 | [PDF](https://arxiv.org/pdf/2607.21058v1)

**作者:** Renzhen Le `[一作]` (Dalian University of Technology), Liming Shu `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一个人类启发的闭环机器人颅骨切除框架，结合双轮廓融合轨迹规划、多模态感知与突破触发轨迹补偿，实现了自动安全骨板分离；

**💡 创新点**

①双轮廓融合自适应螺旋轨迹规划兼顾内外骨骼曲面；②CMA‑TCN‑Transformer+ABF实现97%突破检测且延迟仅0.048s；③突破触发的原位投影轨迹补偿可安全去除残骨；④在 ex vivo 颅骨上完成无颅膜损伤的闭环实验；

**🔧 技术方法**

CT前置轨迹规划、双轮廓融合、球端切割工具、6DOF 力/扭矩传感器、麦克风、深度相机、CMA‑TCN‑Transformer、Adaptive Bayesian Filter、投影轨迹补偿、KUKA LBR iiwa 14R820 机器人、Nakanishi 高速主轴、ROS2 Humble 等；

**📊 数据集**

牛排骨实验（16次）用于验证感知与突破检测；ex vivo 山羊颅骨（5 次，4 次闭环 1 次开放环）用于验证安全切除；使用 CT 图像生成轨迹并采集力、声、姿态等多模态数据；

**📈 对比分析**

与现有方法对比，突破检测准确率提升至97%，触发延迟 0.048±0.097 s，最大超越 <0.29 mm；闭环实验全部无颅膜损伤，残骨厚度 0.428±0.015 mm；开放环实验导致颅膜损伤，表明轨迹补偿显著提升安全性；

**⚠️ 局限性**

仅在 ex vivo 骨骼上验证，缺乏活体动物或临床数据；CT 规划无法识别软组织病变，需结合 MRI；系统对配准误差和生理干扰（出血、脉动等）的鲁棒性仍待进一步评估；

---

## 327. Faster IndexTTS-2: Accelerating and Streaming Autoregressive Zero-Shot Text-to-Speech Synthesis on GPUs

**arXiv ID:** 2607.21042 | [PDF](https://arxiv.org/pdf/2607.21042v1)

**作者:** Muyang Du `[一作]` (NVIDIA), Junjie Lai `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将 IndexTTS‑2 的所有模块（预处理、GPT、DiT、BigVGAN）在 GPU 上完全加速，支持流式和批量推理。

**💡 创新点**

将 TensorRT‑LLM 原本为语言模型设计的推理框架，改造为可兼容语音 GPT 的专用版；同时实现跨模块批量、流式推理与跨框架的全链路加速。

**🔧 技术方法**

使用 TensorRT、TensorRT‑LLM、ONNX 转换、混合精度（FP16、W8A16/W4A16）以及动态批量和重叠跨越技术；在 GPT 上加入提示调优、合并词表、定制位置编码与隐藏状态输出。

**📊 数据集**

在 Seed‑TTS 基准上进行评估，包含 1,088 条英语样本（Common Voice）和 2,020 条中文样本（DiDiSpeech‑2）。

**📈 对比分析**

通过与原始 PyTorch FP32 版本的对比，测量总延迟、RTF、WER、SIM‑o、UTMOS；结果显示 GPT 速度提升 5×，端到端提升 3.6×，RTF 降至 0.24（英语）/0.22（中文），质量指标几乎无损。

**⚠️ 局限性**

限制：流式 TTFA 随批量增大显著提升；批量效果在 batch>8 下降；需手工改造 TensorRT‑LLM；仅在单 GPU 上测试，对多 GPU 或更大模型的适配尚未验证。

---

## 328. Shortest Paths with Linear Edge Weights

**arXiv ID:** 2607.21055 | [PDF](https://arxiv.org/pdf/2607.21055v1)

**作者:** Suryajith Chillara `[一作]` (IIIT Hyderabad), Nithish Raja `[通讯]` (TU Eindhoven)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了边权为线性函数的有向图中的最短路径问题，提出了参数最短路径问题的上界和下界。

**💡 创新点**

提出了一个新的上界 n^O(d log n)，显著改善了之前的结果，并且为无向图和多项式边权的情况提供了相应的结果。

**🔧 技术方法**

使用了分治法和超平面划分技术来分析最短路径的数量，并构建了最短路径预处理数据结构。

**📊 数据集**

使用了具有线性边权的有向无环图（DAG）和多项式边权的图作为数据集进行实验和理论分析。

**📈 对比分析**

与之前的研究相比，新的上界 n^O(d log n) 显著提高了性能，且通过构造的最短路径预处理器可以在亚线性时间内查询最短路径。

**⚠️ 局限性**

该研究的局限性在于尚未解决无向图的参数最短路径问题的下界，且多变量多项式的情况仍然缺乏有效的上界。

---

## 329. Multimmit: Extending Blocks for Faster Finality

**arXiv ID:** 2607.21021 | [PDF](https://arxiv.org/pdf/2607.21021v1)

**作者:** Andrew Lewis-Pye `[一作]` (London School of Economics), Patrick O'Grady `[通讯]` (Commonware)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了 Multimmit 协议，将多链数据传播与单轮投票的状态机复制相结合，兼顾低延迟和容错性。

**💡 创新点**

创新点在于投票相对提议（Proposal‑relative voting）和扩展投票（Extension voting），通过安全的链顶点提议解决 Raptr 的易破坏性和 Autobahn 的高延迟问题。

**🔧 技术方法**

使用部分同步模型、BLS 聚合签名、阈值签名、链提议与投票结构以及数据可用性证书等技术。

**📊 数据集**

实验基于模拟网络延迟和事务负载，未使用真实数据集，而是通过仿真评估吞吐量和延迟。

**📈 对比分析**

与 Raptr 对比，Multimmit 在良好/常规情况平均延迟约为 t+3δ（Raptr 为 t+5δ），在最优情况下 t+2δ；吞吐量与 Raptr 相当，但每视图通信量仅为几 KB，显著低于 Raptr 的几百 KB。

**⚠️ 局限性**

局限性包括需至少满足 5f+1 节点，链深度和扩展界限参数需手动设置，并且在极端攻击下仍可能出现视图跳过或延迟增加。

---

## 330. Weak Private Information Retrieval for Graph-based Storage

**arXiv ID:** 2607.21014 | [PDF](https://arxiv.org/pdf/2607.21014v1)

**作者:** Shodasakshari Vidya `[一作]` (International Institute of Information Technology), Prasad Krishnan `[通讯]` (International Institute of Information Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于图的弱私有信息检索（G‑WPIR）方案，允许在保证信息泄漏可控的前提下提升下载速率，且在任意简单无向图上实现。

**💡 创新点**

创新点包括：①引入互信息泄漏和最大泄漏两种弱隐私度量；②利用偏置硬币参数 p 实现速率‑隐私平滑折中；③对完整图与完整二分图给出显式的速率-隐私表达式，展示了与传统严格 G‑PIR 的对比优势。

**🔧 技术方法**

核心技术是：分层独立集分解、概率化查询生成、子包化简化（L=1）以及对查询分布的偏置控制；同时利用信息论极大化/极小化技术计算 MI 与最大泄漏。

**📊 数据集**

本工作为理论研究，未使用实际数据集，所有结果均为推导与理论分析。

**📈 对比分析**

与严格 G‑PIR 方案比较时，G‑WPIR 在相同图结构下可实现更高速率（例如完整图从 1/(N‑1) 提升到 1），但隐私泄漏从 0 变为可控的 MI 或最大泄漏；实验（或数值模拟）表明随着 p 增大，速率升高而泄漏随之增加，满足预期的折中。

**⚠️ 局限性**

局限性包括：①缺乏信息论上界证明，无法确定所给方案是否最优；②对更一般图形（如多重图、带有副本的超图）的扩展尚未完成；③泄漏度量仅考虑单个服务器，未讨论服务器协作或攻击者模型的影响。

---

## 331. EmoAgent-R1: Towards Multimodal Emotion Understanding with Reinforcement Learning-based Dynamic Agent Specialization

**arXiv ID:** 2607.21013 | [PDF](https://arxiv.org/pdf/2607.21013v1)

**作者:** Lihuang Fang `[一作]` (Guangdong University of Technology), Jinghui Qin `[通讯]` (Guangdong University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于强化学习的动态代理特化框架EmoAgent‑R1，用于多模态情感识别，先通过冷启动合成CoT与路由数据预训练，随后利用P‑GRPO对情感推理和代理选择进行稀疏奖励的细粒度优化。

**💡 创新点**

①将情感理解拆分为路由代理与专门化代理的两阶段工作流程，消除静态prompt的均匀性偏差；②提出Progressive Group Relative Policy Optimization（P‑GRPO），结合PMI激励的token级调制，实现稀疏奖励的细粒度信用分配；③利用合成CoT与经验路由数据实现冷启动，降低RL探索难度。

**🔧 技术方法**

多模态大型语言模型（Qwen2.5‑VL‑7B），Chain‑of‑Thought生成与验证，经验路由与路由代理，Group Relative Policy Optimization、PMI‑激励的token调制、P‑GRPO、PPO式截断目标以及由准确性+格式化奖励组成的奖励模型。

**📊 数据集**

合成答案条件CoT数据（General 110k，Emotion 14k），MER‑Caption+（11k），MOSEI（3k）及过滤后RL数据（45k）。评估使用MER‑UniBench（情感倾向、基础情感、细粒度情感）以及通用视频基准（VSI‑Bench、TempCompass、Video‑MME 等）。

**📈 对比分析**

与多种基线（Otter、Video‑LLaVA、PandaGPT、Video‑ChatGPT、VideoChat2、LLaMA‑VID、VideoChat、Chat‑UniVi、mPLUG‑Owl、AffectGPT、AffectGPT‑R1）及模型规模基线对比。EmoAgent‑R1在MER‑UniBench取得平均77.85%（rank1），各子任务上均超越前沿；在通用基准上保持或略优于基础模型，表明具有良好泛化能力。实验显示P‑GRPO显著优于GRPO，路由+专门化提升约6–10%。

**⚠️ 局限性**

对稀疏奖励的依赖仍需精细调参，P‑GRPO在极端复杂场景下可能收敛慢；冷启动合成数据与路由标签质量可能影响路由学习；目前仅在视频情感识别任务验证，跨模态或不同情感标签体系的迁移性能待进一步验证；RL阶段单样本训练导致计算成本高，耗时较长。

---

## 332. Workflow-Localized Mechanism Learning: Attribution-Guided Repair and Knowledge Reuse for Structured Agent Skills

**arXiv ID:** 2607.20999 | [PDF](https://arxiv.org/pdf/2607.20999v1)

**作者:** Zibin Lin `[一作]` (Shenzhen University), Guofu Liao `[通讯]` (Shenzhen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于工作流节点的机制归因与局部知识重用方法（Workflow-Localized Mechanism Learning，WML），用于优化冻结语言模型代理的结构化技能；

**💡 创新点**

创新点在于将故障定位精确到工作流节点与机制关系，并通过Node–Mechanism Attribution将修复目标限定为最小的L2/L3地址，同时引入闭环的WGSO流程实现第三方技能的有条件检索、受限补丁与后置评估；

**🔧 技术方法**

核心技术包括：节点机制归因、局部知识检索与适配、受限补丁策略、评估门控、优化器侧记忆（Patch‑Strategy Memory）以及统一LLM工作流编译器；

**📊 数据集**

实验使用了SpreadsheetBench和WikiTableQuestions（转化为表格形式）以及Compiler‑Supported50任务集；

**📈 对比分析**

与多种基线（No Skill、Seed Skill、Official SkillGrad、SkillOpt、EvoSkill）在两种后端（DeepSeek‑chat、Qwen3.6‑Flash）下对比，WML在硬准确率和单元格准确率均领先6–8个百分点；在编译器环境下还实现了更高的成功率和更低的token与调用成本；

**⚠️ 局限性**

局限包括：实验仅覆盖电子表格任务，模型对其他类型的工作流与机制仍需验证；评估门控基于同一批次，对长期回归的检测不足；对第三方技能的安全与版权审核尚未系统化。

---

## 333. Nipping the Butterfly Effect in the Bud: Self-Output Fine-Tuning for Autoregressive Weather Prediction

**arXiv ID:** 2607.21080 | [PDF](https://arxiv.org/pdf/2607.21080v1)

**作者:** Yun-Ye Cai `[一作]` (National Taiwan University), Hsuan-Tien Lin `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过分析自回归深度学习天气预测中误差累积的根源，提出了自输出微调（SOFT）方法以抑制误差放大并提升长期预测准确性。

**💡 创新点**

创新点在于将误差放大视为输入分布漂移导致的负反馈循环，并通过在训练阶段使用模型自身一阶预测的偏差输入进行微调，形成简单高效的分布对齐策略，显著缓解“蝴蝶效应”。

**🔧 技术方法**

核心技术包括自回归推理框架、误差传播理论分析、分布差异度量（Fréchet Inception Distance）、以及基于单步预测的自输出微调训练流程。

**📊 数据集**

实验数据集为ERA5重分析数据的东亚区域子集（覆盖台湾、北美和欧洲），分辨率0.25°，用于训练与评估。

**📈 对比分析**

与单步训练、rollout微调、Replay Buffer、特征匹配GAN和噪声增强等现有长时序训练策略对比，SOFT在168小时预测中在多种天气变量（t2m、u10、t_850、z_500）上实现MAE和FID指标的最优或接近最优表现，且计算成本更低。

**⚠️ 局限性**

局限性包括仅在东亚区域子集验证，全球尺度应用仍待研究；同时方法主要针对天气变量预测，未探讨其他气象任务或多模型融合的潜在改进。

---

## 334. HyperImageNet: A Large-Scale High-Spatial Resolution Hyperspectral Imagery Classification Benchmark

**arXiv ID:** 2607.21050 | [PDF](https://arxiv.org/pdf/2607.21050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 335. The RealDefocus Benchmark for Defocus Deblurring

**arXiv ID:** 2607.21078 | [PDF](https://arxiv.org/pdf/2607.21078v1)

**作者:** Tim Seizinger `[一作]`, Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了RealDefocus基准，用于评估单图像散焦去卷积方法。

**💡 创新点**

提供大规模、高分辨率、真实对齐的散焦/清晰图像对，并提出统一评估协议和跨数据集验证。

**🔧 技术方法**

采用深度学习去卷积模型、光学PSF建模与Transformer/SSM架构进行训练与评估。

**📊 数据集**

使用RealDefocus数据集（23k对，4.4k场景），并与RealDOF、DPDD_S等数据集进行对比。

**📈 对比分析**

在RealDefocus上训练多种模型（DPDNet、GKMNet、Restormer、FFTFormer等）后，Transformer/SSM模型在PSNR/SSIM/LPIPS上表现最佳，跨数据集泛化显著提升。

**⚠️ 局限性**

局限性包括光圈捕获协议导致真值仍有残留散焦，光学域与合成域仍存在差距，未来可考虑聚焦堆叠以提升真值质量。

---

## 336. Show, Don't Tell: Evaluating Spatial Cognition in Generative Pixels Rather Than LLM Text

**arXiv ID:** 2607.21072 | [PDF](https://arxiv.org/pdf/2607.21072v1)

**作者:** Xu Wang `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了ProVisE框架和SpatialGen‑Bench基准，用协议化视觉回答方式评估图像生成模型在空间推理任务中的表现，并与传统文本输出VLM在相同任务语义下进行对比。

**💡 创新点**

创新点在于①识别并解决文本输出与图像生成模型在空间评估中的答案接口不匹配问题；②设计可复用的视觉协议与解析流程（ProVisE）；③引入Agentic协议构造器实现对新基准的自动化协议生成与验证；④构建跨层次、多答案形式的诊断性空间基准SpatialGen‑Bench。

**🔧 技术方法**

技术手段包括视觉协议生成与路由、确定性图像解析器、Agentic协议构造与验证、协议化视觉生成与解析、统一度量兼容、通用VLM解析器对比实验以及对多任务基准的自动归一化与转化。

**📊 数据集**

使用了多份公开空间评估数据集（BLINK、ViewSpatial‑Bench、MindCube、OmniSpatial、SpatialScore、SpatialTree、SpatialGenEval）以及外部基准（EmbSpatial‑Bench、Q‑Spatial+、RoboSpatial‑Home、SAT、RoboAfford）作为样本来源，最终生成了470个样本、14个子任务的SpatialGen‑Bench。

**📈 对比分析**

采用ProVisE统一协议，将31个模型（20文本回答、11视觉回答）在同一套任务与度量下评测，结果显示文本模型GPT‑5.4整体得分61.04，视觉模型GPT Image 2得分54.49，均低于人类87.79。视觉模型在Depth、Relationship等可直接可视化的任务中表现优于文本，文本模型在Size、Feasibility、Prediction等需要推理与变换的任务中表现更好，二者互补。

**⚠️ 局限性**

局限性包括①Agentic构造器依赖GPT‑5.4/ GPT Image 2 可能偏向其可执行的视觉表示；②Fallback路由因库覆盖不足导致部分任务无法得到确定性解析；③目前仅支持静态图像评估，未扩展到视频、仿真或闭环控制；④文本与视觉模型的架构差异使得比较结果难以归因；⑤通用VLM解析器对最终分数与排名有显著影响。

---

## 337. Sample-Efficient Learning from Agent Experience

**arXiv ID:** 2607.21051 | [PDF](https://arxiv.org/pdf/2607.21051v1)

**作者:** Chenhui Gou `[一作]` (Monash University), Hamid Rezatofighi `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过一次性“分支”回放将真实交互经验中的决策提取出来，并在不需要进一步环境交互的前提下将这些决策蒸馏到模型权重中，实现了样本高效的学习迁移

**💡 创新点**

创新点在于：①采用单步分支回放（只在已记录历史点生成一次教师决策）避免了长回放和世界模型误差；②利用分支打包和增强教师推理提升监督密度；③用教师采样正向KL实现无监督目标转化，使得模型无需在有经验的上下文下执行即可获得提升

**🔧 技术方法**

技术手段包括：上下文蒸馏（context distillation）、大语言模型的提示式推理、分支打包（branch packing）、教师采样正向KL、增强推理提示以及多任务联合训练

**📊 数据集**

实验数据集包括749个精心策划的软件工程（SWE）任务和6个TaleSuite（Jericho）文本冒险游戏任务

**📈 对比分析**

与传统的ICL、监督微调（SFT）和经典RL（PPO、GRPO）对比，Distillation+ICL在SWE任务上取得51.4% pass@1，保留了64.8%的ICL收益；在TaleSuite上获得43.8的标准化分数，保留93.4%的ICL收益；同时与RL基线相比，使用的环境样本数至少少9.6倍，且在OOD任务与持续学习循环中也表现出可积累的提升

**⚠️ 局限性**

主要限制在于：教师生成所需的计算成本较高（需多次推理生成分支序列）；方法依赖大模型的上下文窗口与推理能力；在极长交互历史或低资源环境下的可扩展性尚未充分验证

---

## 338. A Real-Time Generalized Nash Equilibrium Framework for Interaction-Aware Autonomous Driving in Mixed Traffic

**arXiv ID:** 2607.21043 | [PDF](https://arxiv.org/pdf/2607.21043v1)

**作者:** Nouhed Naidja `[一作]` (Institut VEDECOM), Marc Revilloud `[通讯]` (Dotflow)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于通用纳什均衡（GNEP）的实时决策框架，用于混合交通场景中的交互感知自动驾驶。

**💡 创新点**

创新点包括：①将共享安全与几何约束显式融入GNEP，实现策略集随对手行动动态变化；②开发基于粒子群优化（PSO）的专用求解器，能够在非凸非线性问题上快速收敛；③在真实测试轨道上验证，证明在交叉口场景下能生成舒适、类似人类的轨迹，且收敛时间低于50 ms。

**🔧 技术方法**

使用技术：游戏理论（GNEP）、粒子群优化（PSO）求解器、轨迹生成与多目标代价函数（安全、舒适、效率），以及实时车辆控制与传感器融合。

**📊 数据集**

数据集：未采用公开数据集，而是在Versailles-Satory的闭环测试轨道上使用一台Renault Zoé与一辆人驱车进行对碰实验，收集传感器数据与轨迹记录。

**📈 对比分析**

方法比较：通过与传统独立优化或规则基础方法对比，展示PSO求解器在100次独立运行中的收敛时间与均方误差；实验结果显示在网格尺寸160×160时仍保持<50 ms收敛，且得到合法纳什均衡，表现优于固定网格方法。

**⚠️ 局限性**

局限性：仅适用于两方交互；随着参与方数增多，搜索空间维度与共享约束复杂度呈指数增长；基于采样的估计可能导致解的精度受限；未来需研究多智能体扩展与更高效的采样/约束处理方法。

---

## 339. Bridging the Structural Gap: Adapting Autoregressive Generation for Recommendation

**arXiv ID:** 2607.21028 | [PDF](https://arxiv.org/pdf/2607.21028v1)

**作者:** Junchao Zeng `[一作]` (Tencent), Zang Li `[通讯]` (Tencent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 BARGE 模型，针对生成式推荐中的两大结构缺口（项级边界消失与层级语义漂移）进行改进，集成 ICA、HPR 和 DPD 三个模块，实现编码器与解码器结构的闭环。

**💡 创新点**

创新点包括：① 通过交叉注意力聚合+门控融合恢复项级上下文；② 采用层级路径重排序（HPR）对 Beam Search 进行全路径对比学习，抑制语义漂移；③ 设计 OSQ‑VAE 正交拆分量化与双通道解码，利用 OR‑融合跨通道恢复漏检项，形成互补的错误抑制机制。

**🔧 技术方法**

技术手段：生成式推荐框架；RQ‑VAE/OSQ‑VAE 量化；Transformer 编码器+自回归解码器；交叉注意力+门控融合；InfoNCE 对比学习；Beam Search；双塔层级路径重排序；OR‑融合；离线与在线 A/B 对比实验。

**📊 数据集**

数据集：Amazon Beauty、Amazon Sports & Outdoors（公开基准）以及腾讯大规模离线测试（百万级用户、数亿交互）和线上 A/B 测试。

**📈 对比分析**

与多种判别式（P5、Caser、SASRec 等）和生成式（TIGER、COBRA、ActionPiece、APAO 等）基线在 Recall@K / NDCG@K 上对比。BARGE 在 Beauty/Sports 上均居首，Recall@10 提升约 19.6% / 8.8%；在腾讯离线测试中 Hit@5、Hit@10、Hit@20、Hit@50 均明显优于基线；在线 A/B 结果显示 CTR +0.60%、UV +1.34%、阅读时长 +1.70%。BARGE‑base 与全模型对比进一步验证三模块的单独贡献。

**⚠️ 局限性**

局限性：模型仍需较深编码器与双解码器，训练与推理成本相对较高；正交拆分量化的学习可能对不同域存在迁移瓶颈；对极端冷启动或多任务场景的适应性未做深入验证；以及对长尾项的进一步泛化和多样性提升仍待研究。

---

## 340. ProCap: Prominence-guided Object Rectification for Faithful and Comprehensive Video Captioning

**arXiv ID:** 2607.21022 | [PDF](https://arxiv.org/pdf/2607.21022v1)

**作者:** Debjyoti Das Adhikary `[一作]` (Indian Institute of Technology Kharagpur), Partha Pratim Chakrabarti `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、基于对象显著性排序的迭代后处理框架ProCap，用于提升视频字幕的完整性和真实性。

**💡 创新点**

创新点在于将外观、持续性和动态性三因素融合的显著性评分与多轮提示循环结合，实现模型无关、轻量级的语义完善。

**🔧 技术方法**

使用YOLOv8检测、显著性评分算法、Prompt驱动的LLM（如GPT）多轮修正以及迭代框架。

**📊 数据集**

采用MSR‑VTT和MSVD两个公开视频字幕基准。

**📈 对比分析**

与mPLUG-2、单次提示和无修正基线对比，自动指标Temporal Completeness提升约10‑13%，不一致性降低30‑45%；人类评估中完整度提升约50‑65%，不一致性下降约25‑45%。

**⚠️ 局限性**

局限性包括对检测质量的依赖，显著性模型对复杂动态场景的鲁棒性有限，以及多轮提示导致生成时间略有增加，无法完全消除所有幻觉。

---

## 341. TableVerse: A Large-scale Tabletop Dataset with Real-world Grounded Layouts for Generalizable Manipulation

**arXiv ID:** 2607.21017 | [PDF](https://arxiv.org/pdf/2607.21017v1)

**作者:** Boyuan Wang `[一作]` (ByteDance), Yu Sun `[通讯]` (ByteDance)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了TableVerse自动化Real2Sim管线，生成100K物理一致桌面场景与专家轨迹，并构建TableVerse-100K数据集。

**💡 创新点**

将单视图检测、3D重建、布局一致碰撞纠正(LCCR)与物理稳定化相结合，实现无碰撞、真实尺度的桌面数字孪生。

**🔧 技术方法**

使用Seed‑1.8、SAM2/SAM3D、Depth Anything3、MuJoCo、LCCR几何优化、Gemini 2.5 Pro等技术。

**📊 数据集**

以互联网上的无结构桌面图像为原料，生成近1M对象、35K类的TableVerse‑100K数据集。

**📈 对比分析**

与MIDI、SAM3D、SceneMaker 100个真实样本对比，TableVerse 场景碰撞率0%，GPT‑Score平均1.38，显著优于基线。

**⚠️ 局限性**

SAM3D 对低分辨率/小像素物体识别差且耗时高，批处理速度仍待提升。

---

## 342. CultureTalk-ID: A Multi-Task Dialogue Benchmark for Cultural Commonsense in Indonesian Local Languages

**arXiv ID:** 2607.21016 | [PDF](https://arxiv.org/pdf/2607.21016v1)

**作者:** Muhammad Dehan Al Kautsar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个人类策划的、多轮对话形式、包含印尼语及10种地方语言、13个文化主题的印尼文化常识基准，并在其上评估LLM在文化推理、机器翻译与语言导向三任务的表现。

**💡 创新点**

首次将多轮对话与地方语言结合，提出三任务评测框架，采用多阶段人类审核确保真实性，并验证文化特定预训练对模型性能的提升。

**🔧 技术方法**

利用GPT‑5生成对话并基于候选回复进行多项式打分；在翻译与语言导向任务中分别使用BLEU、BERTScore、LLM‑as‑judge等评估指标，并对开源模型进行零样本和少量监督微调。

**📊 数据集**

在IndoCulture与COPAL‑ID基础上，通过人类翻译、纠错、质量控制等步骤，最终生成4,496条对话（2,980条地区性），覆盖11种语言（印尼语+10地方语）与13个文化主题。

**📈 对比分析**

在三任务上与专有模型、多语言模型和东南亚中心模型对比，专有模型表现最佳；开源模型在翻译与语言导向任务上明显落后，SEA‑centric模型在本地语言上优于一般多语言模型但仍低于专有模型；文化预训练显著提升模型性能。

**⚠️ 局限性**

数据仅覆盖10个省份且主题分布不均，缺乏其他38省份与更广泛文化差异；本地语言标准化与评测覆盖有限；生成任务仍存在语言偏好与文化细节缺失等挑战。

---

## 343. Safeguards for Speech2Speech LLM-Assistants: A Case Study in Automotive Applications

**arXiv ID:** 2607.21180 | [PDF](https://arxiv.org/pdf/2607.21180v1)

**作者:** Gregor Endler `[一作]` (codemanufaktur GmbH), Lukas Stappen `[通讯]` (BMW Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并实现了针对语音转语音LLM助手的两种定制防护机制（基于转录与工具调用），并在汽车应用场景下进行实验验证

**💡 创新点**

首次将两类防护策略系统化对比，揭示其在S2S架构中对延迟、确定性与安全性的显著影响

**🔧 技术方法**

使用S2S接口（Gemini Flash、Nova Sonic、GPT Realtime）和自定义转录/工具调用流程，结合统计分析（P50/P90/P99、均值与标准差）

**📊 数据集**

实验采用人工构造的恶意输入（触发词“whale”）与正常语料，共计5400次运行，未使用公开大型数据集

**📈 对比分析**

通过18个组合（模型×防护类型×输入）进行延迟测量，结果显示工具调用防护平均延迟0.6–1.4 s；转录防护在Nova Sonic上低于0.4 s，满足流畅对话阈值，其他组合超标

**⚠️ 局限性**

主要限制在于高延迟、缺乏确定性拒绝机制、对转录准确性的依赖以及当前API缺乏对工具链的完整支持

---

## 344. V-DEAL: Diagnosing Video Safety De-Calibration as an Understanding-Refusal Coupling Failure

**arXiv ID:** 2607.21151 | [PDF](https://arxiv.org/pdf/2607.21151v1)

**作者:** Zhetong Zhang `[一作]` (University of Queensland), Yujun Cai `[通讯]` (University of Queensland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视频大语言模型的安全失调现象，提出并验证了 V‑DEAL 三层诊断框架，揭示了有害视频配合表面安全查询时攻击成功率升高的机制，并给出了无参数 prompt 注入的对齐方法。

**💡 创新点**

创新点在于将行为、理解与内部表示三维度统一评估，利用隐藏状态拒绝方向量化内部拒绝倾向，并通过轻量级 prompt 介入实现与 fine‑tuning 相当的安全对齐。

**🔧 技术方法**

采用对抗性攻击实验、结构化视频描述/摘要代理任务、隐藏层拒绝方向投影分析，以及 prompt‑based realignment（零样本与 8‑shot）。

**📊 数据集**

使用 Video‑SafetyBench、VA‑SafetyBench、Omni‑SafetyBench、XSTest、MM‑SafetyBench 等公开安全评测数据集。

**📈 对比分析**

在六款公开模型上，VH‑BQ 条件下平均攻击成功率为 48.33%，通过 prompt 对齐后降至 0.80%（零样本）/0.24%（8‑shot），与现有 fine‑tuning 方案性能相当。

**⚠️ 局限性**

局限在于评估仅覆盖四种输入组合，未探究更细粒度跨模态鲁棒性；对抗样本仅来自公开数据集，缺乏对真实世界复杂场景的验证。

---

## 345. DTIF: Robust Loop Closure Detection via Delaunay Triangle Topology in Complex Forests

**arXiv ID:** 2607.21138 | [PDF](https://arxiv.org/pdf/2607.21138v1)

**作者:** Xin Zhao `[一作]` (Wuhan University), Bisheng Yang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种基于树干拓扑的轻量级森林闭环检测与全局配准框架DTIF。

**💡 创新点**

通过提取树干作为稳定特征，构建Delaunay三角网，利用边长和半径一致性累积强弱支持权重，并将这些权重融入分解的鲁棒姿态估计中，实现了对森林稀疏噪声点云的鲁棒匹配与定位。

**🔧 技术方法**

采用树干提取与圆拟合、Delaunay三角剖分、直方图相似性筛选、顶点支持统计、加权三角一致性验证、分离的yaw/水平/垂直位姿估计、鲁棒截断最小二乘及GNC优化等技术。

**📊 数据集**

在模拟森林数据（基于WHU-TLS+MARSIM+FAST‑LIO2）以及三套真实森林测绘数据（Ma'anshan森林公园3个子图与TLS基准）上进行实验。

**📈 对比分析**

与STD、Quatro++、GROR等基线方法以及FGR、TEASER、Quatro后端求解器对比，DTIF在旋转、平移误差上与GROR相当甚至更优，平均处理时间约200‑400 ms，显著快于高精度方法。

**⚠️ 局限性**

对z轴位姿依赖树干底部/地面点的观测质量，点云稀疏或遮挡时竖向误差增大，且目前未实现多子图联合优化与跨平台协同验证。

---

## 346. Demographically-Informed Heat-Mortality Risk Curves via Risk Graph Neural Networks

**arXiv ID:** 2607.21131 | [PDF](https://arxiv.org/pdf/2607.21131v1)

**作者:** Alex O. Davies `[一作]` (University of Bristol), Rui Zhu `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了风险图神经网络（RGNN），通过在分层图神经网络中学习人口普查特征对分布滞后非线性模型（DLNM）系数向量的校正，从而在保持风险曲线可解释性的同时提升预测校准性能。

**💡 创新点**

其创新点在于将传统的时序DLNM与空间层级图结构相结合，利用GATv2和GraphSAGE实现跨区域信息共享，并采用零初始化线性校正头以确保可解释性，显著提升了极端热浪期间的不确定性覆盖率。

**🔧 技术方法**

技术上使用了层级GATv2Conv与SAGEConv进行邻域聚合、零初始化的多头线性校正、CCC损失、单调性与一致性正则、以及MC‑dropout产生的预测不确定性；所有层级共享冻结的地区级OLS基线。

**📊 数据集**

实验基于英国HadUK每日温度、ONS提供的LAD级每日死亡率（2000–2024）以及2021年OA级人口普查数据，对英格兰和威尔士10个地区的2018年和2022年两次极端热浪进行评估。

**📈 对比分析**

在RMSE、MAE、Pearson相关、CRPS及90%置信区间覆盖率等指标上，RGNN相较于DLNM、RF、GBM、LSTM和MLP取得了更低的误差、更高的相关性、更小的CRPS，并在2022年热浪期间保持了约0.83–0.89的覆盖率，明显优于基线模型。

**⚠️ 局限性**

局限性包括：对人口普查特征随时间漂移的假设、对其它地区和健康结局的可迁移性尚未充分验证、对特征重要性与因果机制的深入解释仍待开展，以及在极端事件下的鲁棒性虽然提升但仍需进一步提升。

---

## 347. TF-MossFormer: Integrating Convolution Gated Local-Global Attentions for Enhanced Time-Frequency Domain Monaural Speech Separation

**arXiv ID:** 2607.21128 | [PDF](https://arxiv.org/pdf/2607.21128v1)

**作者:** Shengkui Zhao `[一作]` (Alibaba Group), Xiangang Li `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种新的时频Transformer框架TF‑MossFormer，用于单声道语音分离。

**💡 创新点**

创新点在于：①内容感知的滑动窗口局部注意力，可自适应调节感受野；②局部与全局注意力的最佳叠加顺序（先局部后全局）；③卷积门控机制提升特征选择与信息流。

**🔧 技术方法**

技术主要包括：短时傅里叶变换(STFT)编码/解码、双路径频谱与时间模型、Conv‑SwiGLU前馈网络、RMSGroupNorm、局部+全局多头自注意力、卷积门控。

**📊 数据集**

使用WSJ0‑2Mix数据集进行训练与评估，包含20k训练、5k验证、3k测试混合语音。

**📈 对比分析**

与多种基线（时间域如Conv‑TasNet、DualPathRNN；频域如TF‑GridNet、TF‑LocoFormer、SepFormer等）进行对比，TF‑MossFormer在三种规模（S/M/L）上分别实现SI‑SDRi 22.6/24.0/24.4 dB，明显优于同规模模型，且参数与算力相对更低，达成SOTA。

**⚠️ 局限性**

局限性在于：仍需进一步降低计算量/延迟，尤其在实时场景；滑动窗口大小仍需经验调优，缺乏自适应窗口尺寸的学习机制；在更复杂多声源或极低SNR环境下性能尚未充分验证。

---

## 348. EuroFlood: a Python library and queryable index for the CEMS satellite-derived flood-depth archive of Europe

**arXiv ID:** 2607.21126 | [PDF](https://arxiv.org/pdf/2607.21126v1)

**作者:** Jürgen Hackl `[一作]` `[通讯]` (Princeton University), Jürgen Hackl (Princeton University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了EuroFlood开源Python库及其云原生的倒排栅格索引，使Copernicus卫星观测的洪水深度数据集可被高效查询与分析。

**💡 创新点**

创新点在于构建了一个≈138 MB的倒排栅格索引，将数千个20 m洪水深度栅格的事件成员信息压缩为每格唯一组合标识，从而实现无损复原事件足迹并支持按位置、时间、频率的低成本检索。

**🔧 技术方法**

核心技术包括云优化GeoTIFF、GeoParquet表、稀疏编码、任何-湿（any‑wet）重采样、Python 发现→提取（Discover→Extract）API、缓存机制以及与现有GIS与科学计算库（GeoPandas、Rasterio、xarray等）的集成。

**📊 数据集**

使用的主要数据集为2015‑2024年欧洲范围的CEMS‑EFAS卫星衍生洪水深度图（≈18.9 GB）和CEMS‑GLOFAS河流洪水危害地图（7个不同回归期）。

**📈 对比分析**

通过与独立洪水事件数据库HANZE的完整性审核（验证覆盖率≈84%）和对比模型与观测足迹（平均覆盖率≈40%）评估性能；查询速度从首次冷启动约12‑15 s下降到热查询≤300 ms，单次查询仅需不到1 MB传输，显著低于完整数据集。

**⚠️ 局限性**

局限性包括：基于Sentinel‑1的检测在植被覆盖、城市地区和快速短时洪水中的灵敏度低；深度重建误差≈0.2‑0.6 m；事件单元为时空聚类而非单一水文洪水；索引采用保守“任何湿”重采样导致足迹面积膨胀约1.5倍；索引版本固定，新增洪水需新发布。

---

## 349. Physics-Informed Deep Learning Model for Cross-Modality Super-Resolution in Fluorescence Microscopy

**arXiv ID:** 2607.21190 | [PDF](https://arxiv.org/pdf/2607.21190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 350. Causal-AgentIR: Self-Evolving Causal Memory for Adaptive Image Restoration Agents

**arXiv ID:** 2607.21125 | [PDF](https://arxiv.org/pdf/2607.21125v1)

**作者:** Hu Gao `[一作]` (Shanghai Jiao Tong University), Lizhuang Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Causal-AgentIR的分层多智能体框架，利用可自我演化的因果记忆图来指导图像恢复决策，支持对多种降噪、去模糊、去雨、去雾、去雪等混合退化的自适应处理。

**💡 创新点**

创新点在于：①将恢复经验结构化为因果记忆图，连接降质量、区域、工具、动作、质量变化、成本和用户偏好；②设计可学习的记忆演化机制（增删、更新、合并、强化、忽略、丢弃），实现对恢复知识的长期更新、验证和筛除；③多智能体协作（规划、降质分析、工具专家、记忆推理、批评、记忆策划）实现动态计划生成与执行。

**🔧 技术方法**

技术包括：多智能体协作架构、图神经网络用于记忆检索与多跳推理、基于因果关系的规划与奖励学习、学习式记忆演化策略、以及多种深度恢复工具（如FSNet、PGH^2Net、EfDeRain+等）作为可调用工具。

**📊 数据集**

使用的公开数据集有：任务特定的Rain100H/L、Test100、Test1200、CSD、SRRS、Snow100K、RESIDE（SOTS-Indoor/Outdoor）、GoPro、HIDE、CBSD68、Urban100、Kodak24；真实场景集RealRain-1k-L、RTTS、SIDD；以及混合降噪+雾+雨组合的多重退化数据。

**📈 对比分析**

与现有所有-一次模型（如Defusion、Perceive-IR）及其他恢复智能体（MAIR、AgenticIR、HybridAgent、IAMAgent、Restore‑R1）进行对比。Causal-AgentIR在所有单一退化与混合退化任务上均取得最高或近优的PSNR/SSIM，平均提升约0.3–0.7 dB；同时回滚率最低（0.12），成功率最高（0.92）且用户偏好匹配率达0.90，展示出更稳健且符合用户期望的恢复性能。

**⚠️ 局限性**

局限性包括：①仍依赖预先训练好的工具库，工具多样性受限；②因果记忆图的规模随经验增长可能膨胀，检索与推理成本升高；③记忆演化策略需要足够的训练数据和标注反馈，可能在极端或未见过的退化上表现不佳；④整体系统的多智能体协作和多跳推理会导致推理时间增加，影响实时应用。

---

## 351. Relative Value Learning

**arXiv ID:** 2607.21120 | [PDF](https://arxiv.org/pdf/2607.21120v1)

**作者:** Marc Höftmann `[一作]` (Technical University of Dortmund), Stefan Harmeling `[通讯]` (Technical University of Dortmund)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种相对价值学习（Relative Value Learning）框架，将传统的绝对状态价值估计改为直接学习状态对之间的价值差异，并以此为基础重新构造优势估计（R-GAE），使得PPO在此框架下能够直接使用相对价值作为批评器。

**💡 创新点**

核心创新点包括：① 定义了对称（antisymmetric）Bellman算子，使其成为γ-收缩映射并唯一收敛到真实价值差异；② 通过一阶、n阶、λ回报目标实现仅基于可观测奖励差异的无偏训练；③ 将优势估计从绝对价值差异推导为相对优势（R-GAE），从而保证无偏的策略梯度。

**🔧 技术方法**

技术细节包括：使用Siamese差值网络（对称差值头）估计Δ(s_i,s_j)=V(s_i)-V(s_j)，采用pairwise Bellman更新和多步/λ回报目标，结合R-GAE与PPO的剪切目标，所有网络共享CNN编码器并通过线性映射实现对称差值输出。

**📊 数据集**

实验使用了Atari 49款游戏（ALE）作为数据集，采用标准的PPO预处理和EnvPool框架，训练40M帧以评估算法性能。

**📈 对比分析**

在与标准PPO和Direct Advantage Estimation（DAE）的对比中，PPO+RV在61%（30/49）游戏中优于PPO，在75%（37/49）游戏中优于DAE，整体平均得分与基线相当或更好，显示出相对价值学习在多任务环境下的竞争力。

**⚠️ 局限性**

主要局限包括：对称差值网络限制了表达能力；pairwise训练的计算复杂度为O(B^2)，需要子采样；轨迹常数基线B_t在长序列或γλ→1时会放大方差；方法目前仅在离线离散动作的PPO上验证，尚未在连续控制、离线或偏好学习场景中验证其可扩展性。

---

## 352. TOUR: A Trajectory-Level Unlearning Benchmark for Offline Reinforcement Learning

**arXiv ID:** 2607.21111 | [PDF](https://arxiv.org/pdf/2607.21111v1)

**作者:** Chaofan Pan `[一作]` (Southwestern University of Finance and Economics), Xin Yang `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文建立了名为 TOUR 的基准，用于评估离线强化学习（offline RL）中的轨迹级遗忘与重学（unlearning）效果。

**💡 创新点**

创新点在于将轨迹级划分、匹配非成员、重训练参考模型、保留性能锚点以及多种成员推断攻击（likelihood、threshold、reference-model、deviation、TOST 等）组合成完整的隐私‑效用证据框架，解决单一成员分数易误判的问题。

**🔧 技术方法**

技术方法包括：基于 NLL 的轨迹级成员推断、阈值攻击、参考模型攻击、偏差攻击、TOST 等多攻击审计；比较基线包括重训练（Retraining）、微调（Fine‑Tuning）、梯度上升+重新拟合（GA+Refit）以及 TrajDeleter；使用的策略架构包括 Decision Transformer (DT)、MLP、LSTM 以及 IQL。

**📊 数据集**

使用的数据集为 D4RL 的三大连续控制环境（HalfCheetah、Hopper、Walker2D）的 R、M、ME 版本，以及 AntMaze 的 U、UD、MD 扩展。

**📈 对比分析**

通过计算“forget gap”（forget set 与非成员的 AUC 差距）、retain‑negative AUC 以及 D4RL 正常化得分等指标进行比较，结果显示重训练和微调在多数环境中能在保留较高效用的同时显著削弱成员推断；GA+Refit 常导致效用急剧下降；TrajDeleter 的表现随环境而异，单一成员分数容易误判。

**⚠️ 局限性**

局限性包括：仅使用单头 Decision Transformer，架构间对比未能完全消除 confounds；统计功效有限，尤其在低功率设置下；攻击范围有限，未覆盖所有实用攻击；AntMaze 的成功率评估尺度与 D4RL 不同，难以直接比较；未能给出通用的组件级遗忘方法。

---

## 353. Explaining Weather Bulletins via ILP

**arXiv ID:** 2607.21184 | [PDF](https://arxiv.org/pdf/2607.21184v1)

**作者:** Enrico Santi `[一作]` (University of Udine), Andrea Formisano `[通讯]` (University of Udine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了从模拟气象原始数据及OSMER预报图像提取ASP事实、生成ILP例子，并通过FastLAS2学习可解释规则来阐释预报图标所表达的天气预测。

**💡 创新点**

创新点在于将ILP与FastLAS2结合用于解释天气预报图标，构建完整的数据提取流水线（CERRA GRIB、图像模板匹配、TOBAC云/前锋检测），并通过上下文与噪声例子生成可解释的规则。

**🔧 技术方法**

使用的技术包括FastLAS2（LAS框架）、ASP、模式偏置、权重惩罚、图像处理（模板匹配、量化、Gaussian滤波）、TOBAC云跟踪、CERRA再分析数据、跨高度抽象与GPU加速。

**📊 数据集**

使用的数据集为2025年选定的70天OSMER气象预报图像（图标+文本）与对应的CERRA GRIB高分辨率再分析数据。

**📈 对比分析**

通过5折交叉验证进行评估，训练集覆盖率98.3%，平均准确率≈85%；验证集平均准确率57%，召回率≈30%；相较于传统机器学习基线（SVM、决策树、随机森林），更侧重可解释性而非纯预测精度。

**⚠️ 局限性**

局限性包括：数据量有限（仅70天），导致模型对高速气象特征（如风速）学习效果差；内存占用高，限制高度维度特征；FastLAS对大规模数据集的可扩展性不足；生成的规则对风速、风向的依赖不明显。

---

## 354. Representative Sets in Propositional Abduction

**arXiv ID:** 2607.21183 | [PDF](https://arxiv.org/pdf/2607.21183v1)

**作者:** Johannes Schmidt `[一作]` (Jönköping University), Johannes K. Fichte `[通讯]` (Linköping University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在命题归纳中“代表性解释集”问题，提出如何判断给定的解释集是否能覆盖所有可能解释（或子集最小解释），并对该问题在不同约束语言（Post 结构）下的计算复杂性进行系统分类。

**💡 创新点**

创新点在于：①将归纳解释空间的代表性问题与编码理论中的覆盖半径问题建立联系；②完成了在 Post 结构下的完整复杂性图谱；③给出若干参数化（k、|H|、|S|）的 FPT 结果与硬度分析，揭示了该问题与已知 W[1]/W[2] 难度的对应关系。

**🔧 技术方法**

使用的技术包括：Post 结构与极简范畴理论、原子谓词的原始正（pp）定义、极限闭包（efpp）转换、复杂度层级归约、覆盖半径与最近字符串问题的参数化归约、以及对子集最小化约束的专门分析。

**📊 数据集**

本文为理论研究，无实验数据集；所有结果均来自理论证明与多项式/参数化归约。

**📈 对比分析**

通过归约与对比，作者证明了在大多数约束语言下该问题属于 Π₂ⁿ 或 coNP‑hard，部分 Schaefer 语言可在 FPT 时间内解决，而非 Schaefer 语言保持 NP/Π₂ⁿ 难度；与已知编码理论问题相比，得到了相同或更高的难度上界。

**⚠️ 局限性**

局限性：仍有若干约束语言（如 1‑valid、补码语言等）未完成完整分类；参数化复杂性在 k 与 |S| 维度下的完整划分尚未给出；此外，证明中使用的归约主要针对理论模型，缺乏对实际归纳任务的实证验证。

---

## 355. SafeStep: AI-powered Travel Assistance for Elderly People with Frailty or Dementia

**arXiv ID:** 2607.21156 | [PDF](https://arxiv.org/pdf/2607.21156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 356. Out of Sight, Still in Mind: Token Compression for Omni-LLMs

**arXiv ID:** 2607.21179 | [PDF](https://arxiv.org/pdf/2607.21179v1)

**作者:** Suho Yoo `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ReMo，一种训练‑free的视觉 token 压缩框架，通过跨模态信息重分配来显著降低 Omni‑LLM 的输入 token 数量。

**💡 创新点**

创新点在于将 token 压缩视为跨模态信息重分配，而非单纯剪枝：首先利用共同嵌入空间去除已被音频解释的视觉 token，再用空间锚定的文本代理替代对象级视觉 token，并结合局部细节恢复和时间合并以保持精度。

**🔧 技术方法**

主要技术包括：基于 VGGSound 的 canonical correlation 投影构建共同空间、视觉与音频余留度量（s^vis、s^aud）进行 token 选择、轻量级 YOLO 检测生成文本代理并通过 TMRoPE 定位、局部细节距离度量恢复重要细节、以及时间窗口内相似 token 的合并。

**📊 数据集**

使用的数据集包括：VGGSound 用于构建投影；四个音视理解基准（WorldSense、DailyOmni、Video‑MME、OmniVideoBench）和视频字幕基准（video‑SALMONN2）；实验模型为 Qwen2.5‑Omni（3B/7B）与 Qwen3‑Omni‑30B。

**📈 对比分析**

与随机丢弃、FastV、OmniZip 等训练‑free 基线进行对比。ReMo 在 3B/7B/30B 模型上实现约 50% 的 token 压缩率，同时平均准确率达到 101.2%–101.4%，略优于完整 token 模型；延迟仅为全量模型的 0.65×–0.57×，显著提升效率。

**⚠️ 局限性**

局限性包括：仅在相对较短的剪辑上评估，长序列或流式输入仍需进一步研究；文本代理受检测器词表限制，无法覆盖 OOV 实体；压缩比例需手动调参，缺乏自适应机制，适应不同任务和输入特性的能力有限。

---

## 357. Automated Synthesis and Adversarial Validation of Executable Causal Research Pipelines

**arXiv ID:** 2607.21173 | [PDF](https://arxiv.org/pdf/2607.21173v1)

**作者:** Irena Girshovitz `[一作]` (Tel Aviv University), Ran Gilad-Bachrach `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AI驱动的Epidemiology Research Assistant（ARA）框架，利用多代理对话生成结构化研究协议、合成数据并通过对抗式验证检测因果推断假设的可靠性；

**💡 创新点**

创新点在于将因果假设转化为可执行的对抗测试，通过协议设计与审计循环显式捕捉和报告无效因果假设，改进传统LLM直接代码生成的“暗中错误”问题；

**🔧 技术方法**

使用多代理系统（PI、审稿人、规划器、代码生成器等）、结构因果模型（SCM）生成合成数据、对抗式验证、LLM（Claude Opus 4.6）进行协议与代码生成；

**📊 数据集**

在Automated Causal Reasoning Benchmark（ACRB）上进行评估，挑选33个含真实表格数据及数据字典的案例；

**📈 对比分析**

与单一脚本生成的基线相比，ARA在含义方向一致性、处理不完整提示时的治疗变量恢复率上表现更好，且无符号翻转，但在数值精确度、CI一致性等指标上并未显著优于基线；

**⚠️ 局限性**

局限包括样本量小、对抗验证与审计循环可能导致收敛成本高、缺乏真实世界数据验证、可解释性标签需要人工评审、模型与提示依赖性强，且最终检查点不一定为正式批准的脚本。

---

## 358. Agree on the Model, Verify the Inference: GKR Protocols for HND-Based Transformer Inference

**arXiv ID:** 2607.21162 | [PDF](https://arxiv.org/pdf/2607.21162v1)

**作者:** Xiaolong Liang `[一作]` (Chinese Academy of Sciences), Yisheng Lv `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GKR-HND 协议，允许客户端在不存储稠密权重且不重放矩阵运算的情况下，验证已注册 HND Transformer 主干的推理结果。

**💡 创新点**

创新点在于将 GKR 的逐层求和检查与公共计算委托相结合，形成块级证明可递归组合的系统；同时通过 ML‑KZG 承诺与固定点整数运算实现高效验证。

**🔧 技术方法**

采用 GKR 交互式证明、ML‑KZG 低度多项式承诺、多项式扩展、Fiat‑Shamir 随机化、Ed25519 数字签名以及离散化的固定点整数运算。

**📊 数据集**

实验基于公开的 HND-32M 与 HND-124M Transformer checkpoints（量化模型），使用其对应的权重、校正网络和标记词表进行评估。

**📈 对比分析**

与直接固定点重放对比：单核验证 0.21 s、重放 0.09 s；多线程可提升 7‑8 倍但仍略慢；32M 模型单块证明 18.8 KB，124M 模型共 37 MB，证明生成时间 41 s（32M）至 1,110 s（124M）。

**⚠️ 局限性**

局限性包括：仅验证主干多项式，未覆盖校正网络、嵌入、最终输出；使用实验性 ML‑KZG 后端和无审计的 SRS；未实现跨主机部署、网络安全与递归聚合；对稠密矩阵仍需线性扫描。

---

## 359. One More Turn, Less Regret: A Regret-Based Multi-Turn Benchmark for LLMs' Clarification Policies

**arXiv ID:** 2607.21143 | [PDF](https://arxiv.org/pdf/2607.21143v1)

**作者:** Minh Ngoc Ta `[一作]` (MBZUAI), Preslav Nakov `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 RegretBench，一个基于隐式用户意图的自由对话多轮澄清评估基准，能够在完整对话流程中衡量澄清策略的有效性、效率和鲁棒性。

**💡 创新点**

创新点主要包括：①将澄清评价从单轮问答迁移到策略级别，用回报函数和相对 regret 衡量整体对话价值；②设计语义澄清交互图（CIG）将自然语言对话映射到语义动作与观测空间；③构造参考规划器提供基准相对性能评估；④引入多任务场景（问答与产品推荐）验证通用性。

**🔧 技术方法**

使用技术包括：自然语言到语义动作的自动解析与映射、用户回复仿真器、全局奖励与惩罚函数、基于规划的参考策略、与 LLM 的对话交互接口，并在此基础上实现多模型、多提示方法与多用户人设的实验。

**📊 数据集**

数据集来源于 AmbigDocs、CondAmbigQA‑2K 以及 PSCon 三个公开数据集，统一转换为 CIG 形式，涵盖文档级实体歧义、条件依赖问答以及产品搜索多候选推荐三类场景。

**📈 对比分析**

对比方法包括多种主流 LLM（Gemini、GPT、Qwen、Llama、DeepSeek 等）以及不同提示技术（一次性提示、解释式提示）和用户人设（合作、含糊、矛盾）。实验表明：最终成功率（accuracy）并不能充分体现澄清质量；加入回报与 regret 后，Gemini 3.1 Pro、GPT 4.1、DeepSeek V4 Flash 等模型在效率与鲁棒性上表现突出；提示技术能显著改变策略平衡，解释式提示往往提升回报与 regret，减少回调成本。

**⚠️ 局限性**

limitations：①CIG 的语义细节与过滤依赖人工与自动化解析，可能遗漏细粒度歧义或产生冗余维度；②仿真器与解析器的误差会影响分数准确性；③regret 仅相对参考策略，无法保证在实际应用中绝对最优；④数据集覆盖度受限，未包含所有类型的歧义场景，尤其是细微与上下文依赖的歧义。

---

## 360. Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores

**arXiv ID:** 2607.21127 | [PDF](https://arxiv.org/pdf/2607.21127v1)

**作者:** Viola Negroni `[一作]` (Politecnico di Milano), Stefano Tubaro `[通讯]` (Politecnico di Milano)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种可解释的语音深度伪造检测框架，利用针对不同合成缺陷的专家模型并将其输出校准为对数似然比（LLR）进行决策。

**💡 创新点**

创新点在于：①将检测任务拆解为对预定义合成缺陷的识别，②通过伪造数据训练每个专家保证其专一性，③将专家输出校准为可量化的LLR，④在LLR层面进行简单求和实现最终决策，保持全流程可解释。

**🔧 技术方法**

技术方法包括：伪造数据生成（对真实语音施加特定缺陷）、特征提取（如F0轨迹、LSF、相位、能量、音调指纹）、多层感知机或轻量CNN专家、Platt缩放+高斯KDE或CMLG的LLR校准、LLR求和/最大/门控聚合。

**📊 数据集**

使用的数据集：ASVspoof 2019（训练、验证、评估）、ASVspoof 5、SpoofCeleb，伪造数据在训练时实时生成。实验评估主要基于ASVspoof 2019的EER和CLLR。

**📈 对比分析**

对比方法：单个专家、校准方式（KDE vs CMLG）、聚合策略（求和、最大、门控）。在ASVspoof 2019评估集上，求和聚合在EER 19.76%和CLLR 0.638上优于其他策略，证明LLR求和最有效；单个专家性能明显低于集成。

**⚠️ 局限性**

局限性：仅能检测预定义的缺陷，对未知或新型合成技术的适应性有限；专家模型对伪造数据的生成方式敏感，可能导致对真实数据的误判；整体检测性能仍低于最先进的黑盒模型，且需要不断加入新的专家来覆盖更多缺陷。

---

## 361. The Second LoViF 2026 Challenge on Real-World All-in-One Image Restoration: Methods and Results

**arXiv ID:** 2607.21118 | [PDF](https://arxiv.org/pdf/2607.21118v1)

**作者:** Xiang Chen `[一作]` (Nanjing University Of Science And Technology), Weidong Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对第二届LoViF 2026年“真实场景全能图像修复”挑战进行综述，分析参赛方案与结果，提出统一评价指标与基准数据集

**💡 创新点**

首次构建面向多种真实失真（模糊、低照度、雾、雨、雪）的统一评价框架和大规模真实配对数据集FoundIR-LoViF；系统性评估各类全能修复方法的泛化与鲁棒性

**🔧 技术方法**

综合使用多尺度U-Net、Wavelet、Mamba、Transformer、Mixture‑of‑Experts、Diffusion、Rectified‑Flow等先进模型与技术，并对比不同网络结构、损失函数、训练策略与后处理方法

**📊 数据集**

FoundIR（百万级真实配对图像）和WeatherBench（多天气场景）数据集，包含24,500训练对、500验证/测试样本，覆盖五类失真，分辨率512×512

**📈 对比分析**

在统一的PSNR/SSIM/LPIPS混合评分上，20支队伍被纳入最终排行榜，首位Re:Pixel取得42.28分，领先第二名0.47分；方法间在参数量、GFLOPs、推理速度上差异显著，展示了权衡性能与效率的多样化趋势

**⚠️ 局限性**

挑战仍面临域差距、不同失真耦合、跨失真泛化不足等限制；部分方法对实时部署仍有算力/内存瓶颈，后续需提升模型压缩与多失真联合学习的鲁棒性

---

## 362. RL-MACRO: A Cybernetic Closed-Loop Intelligence Framework for Multimodal Adaptive Robotic Craniotomy

**arXiv ID:** 2607.21113 | [PDF](https://arxiv.org/pdf/2607.21113v1)

**作者:** Xiao Zhang `[一作]` (Dalian University of Technology), Liming Shu `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种闭环感知–决策–执行框架 RL‑MACRO，实现在部分可观测环境下的自适应机器人开颅手术。

**💡 创新点**

创新点在于通过 CNN‑LSTM 多模态观测器重构不可测温度，结合隐式状态分类器的双头 IQL 代理，实现对进给速率、主轴转速和切深的协同调节，并通过动态轨迹重规划与速度伺服实现空间连续运动。

**🔧 技术方法**

核心技术包括多模态感知融合（力声+CNN‑LSTM）、隐式状态分类器、离线强化学习（Implicit Q‑Learning）双头 Actor、轨迹重规划与速度伺服算法。

**📊 数据集**

使用的离线数据集由猪肋骨在多组切削参数下采集的力声和温度记录构成，随后在异体山羊颅骨上进行在线闭环验证。

**📈 对比分析**

与固定参数基线相比，RL‑MACRO 在温度峰值、切削力和累计奖励方面均显著优于基线（温度平均误差约 2 °C，峰值降至 27 °C，累计奖励提升约 30%），并在六个未见样本上实现无血管损伤的完整骨瓣切除。

**⚠️ 局限性**

局限在于离线数据仅来自猪肋骨，导致对颅骨域的零样本迁移带来轻微控制波动；此外，安全阈值设置略接近热坏死边界，需进一步扩大数据覆盖并验证更高安全裕度。

---

## 363. Can Generative Recommendation Reach Cold Items? A Temporal Perspective on Semantic-ID Generation

**arXiv ID:** 2607.21101 | [PDF](https://arxiv.org/pdf/2607.21101v1)

**作者:** Jie Peng `[一作]` (Renmin University of China), Bo Zheng `[通讯]` (Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在绝对时间滑动窗口协议下，研究Semantic-ID（SID）生成式推荐在时间冷启动场景中的可达性。

**💡 创新点**

提出了 token‑level 冷启动税onomies、oracle‑prefix 诊断以及把 SID 生成视为层次语义桶化的解释；同时设计三种诊断变体（SID 重新编码、评分接口、动态边上下文）来分离可达性瓶颈。

**🔧 技术方法**

使用基于 Transformer 的 TIGER 生成器、SASRec 对比、oracle‑prefix 评估、JS/皮尔逊相关性、以及多种数据集的滑动窗口实验。

**📊 数据集**

在 7 个公开数据集上实验：Beauty、Sports、Toys、Sephora、WikiLife、WeiboDaily、WeiboTech，涵盖文本与图结构信息。

**📈 对比分析**

与传统离线留一（leave‑one‑out）和 SASRec 进行对比；发现 TIGER 在时间冷启动下对未见目标 Recall@20 近 0，显著低于 seen 目标；三种变体在不同冷启动细分中分别提升了 5–20% 的 Hit@20 或 Recall，尤其是 TIGER‑Scorer 在 unseen 目标上提升显著。

**⚠️ 局限性**

局限性在于：① SID 生成仍受路径依赖，未见原子 token 与不支持路径的 cold 项难以覆盖；② 变体虽提升，但未能彻底突破可达性边界；③ 评估仍基于离线滑动窗口，未覆盖真实在线动态反馈。

---

## 364. Smooth Neural Point Processes via B-Splines

**arXiv ID:** 2607.21098 | [PDF](https://arxiv.org/pdf/2607.21098v1)

**作者:** Michele Bellomo `[一作]` (Politecnico di Milano), Tomaso Aste `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 B‑spline 的神经时间点过程模型，直接用非负 B‑spline 组合来参数化条件强度函数（CIF），并由神经网络预测系数，支持并行计算并通过二阶导数惩罚实现光滑正则化。

**💡 创新点**

创新点在于：① 用 B‑spline 直接表达 CIF，避免对补偿器建模和数值积分；② 允许任意网络架构，满足正则化和并行性；③ 通过预计算积分基函数实现 NLL 的闭式求解；④ 自然引入光滑正则化来抑制振荡。

**🔧 技术方法**

使用的技术包括：B‑spline 基函数、Transformer 网络、最大似然估计、softplus 输出保证非负系数、二阶导数惩罚正则化、并行前向传播、Bisection 迭代求解中位数预测。

**📊 数据集**

实验数据集：七类合成数据（Poisson、Renewal、Self‑correcting、两种 Hawkes 等）以及两个真实数据集（Music 与 Meme）。

**📈 对比分析**

与 Omi 等人提出的基准模型在 MAE（基于中位数的预测误差）上进行对比，结果显示在 5/7 合成数据和 2/2 真实数据中均优于基准；训练速度提升约 12 倍，显著提高计算效率。

**⚠️ 局限性**

局限性：需要手动调节正则化系数 α，调参可能抵消部分计算优势；目前仅在中等规模数据上验证，尚未在大规模多类型事件数据上测试；缺乏对 α 选取的理论指导。

---

## 365. CASC: Causal Adversarial Subspace Clustering for Multivariate Spatiotemporal Data

**arXiv ID:** 2607.21088 | [PDF](https://arxiv.org/pdf/2607.21088v1)

**作者:** Francis Ndikum Nji `[一作]` (University of Maryland, Baltimore County), Jianwu Wang `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 Causal Adversarial Subspace Clustering（CASC）的端到端深度聚类框架，用于从高维多变量时空数据中发现随时间演变的潜在子空间（即临界状态）。

**💡 创新点**

创新点在于：① 将因果关系融入子空间表示学习的自表达损失（Causal Subspace Preservation Loss），使聚类结果更符合潜在物理/因果机制；② 引入动态子空间演化损失（Dynamic Temporal Subspace Evolution Loss），捕捉子空间随时间的平滑转变；③ 设计子空间感知能量基时间判别器（SETD），在对抗训练中直接衡量潜在向量是否贴合对应子空间，减少子空间重叠并提升聚类可解释性；④ 将 FAConvLSTM 与双向图注意力 Transformer 结合，兼顾局部空间、全局依赖和长时序交互。

**🔧 技术方法**

核心技术包括：U-Net 结构的 FAConvLSTM 编码器/解码器、双向时间图注意力 Transformer（Bi‑TGAT）、自表达子空间网络、学生 t 分布的聚类头、Causal Subspace Preservation Loss、Dynamic Temporal Subspace Evolution Loss、子空间感知能量判别器、对抗训练与梯度回传、温度调度、平衡正则化等。模型实现基于 TensorFlow/Keras。

**📊 数据集**

使用了三类真实气候时空数据集：C3S Arctic Regional Reanalysis（CARRA）、欧洲中期天气预报中心 ERA‑5 全球再分析、以及 NCAR Reanalysis 1；所有数据为日度、经纬度×多变量四维张量。

**📈 对比分析**

与 DEC、DSC、ClusterGAN、DTC、DASC 等主流深度聚类基线在六个内部评估指标（Silhouette、Davies‑Bouldin、Calinski‑Harabasz、RMSE、Variance、Inter‑Cluster Distance）上对比。CASC 在所有数据集上均获得最高或竞争性最佳的 Silhouette、最低 DB、最小 RMSE、最大 I‑CD，说明聚类更紧凑、更分离、误差更低。Ablation 研究表明每个新增模块均提升性能，完整模型效果最佳。

**⚠️ 局限性**

局限性包括：① 需要预先估计因果图（如使用 Neural Granger 或 Transfer Entropy），对因果结构不确定的场景可能受限；② 训练过程复杂，涉及多项损失与对抗优化，超参数调优成本较高；③ 目前仅在气候/环境领域验证，跨域推广需进一步验证；④ 计算资源需求相对传统方法更大（FAConvLSTM+Transformer+判别器）。

---

## 366. Advances in STV Margin Computation

**arXiv ID:** 2607.21178 | [PDF](https://arxiv.org/pdf/2607.21178v1)

**作者:** Michelle Blom `[一作]` (Monash University), Damjan Vukcevic `[通讯]` (Monash University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文改进了计算单一可转移投票（STV）选举获胜者边际（margin）下界的算法，使得能够为更多真实选举得到更高的下界；

**💡 创新点**

创新点包括：①直接求解混合整数非线性规划（MINLP）而非线性松弛；②采用更精细的下界启发式，考虑未来回合和转移值；③利用ConcreteSTV得到更紧的上界；④加入支配检查以剪枝搜索树；

**🔧 技术方法**

主要技术包括：分支定界搜索、MINLP求解（使用求解器如 Gurobi/CPLEX）、STV计票模型、启发式下界计算、支配关系判断；

**📊 数据集**

使用真实 STV 选举数据，包括澳大利亚 2 座席选举、美国地方议会选举和苏格兰地方选举等；

**📈 对比分析**

与原始算法（未改进的 STV 边际下界求解器）相比，改进后的算法在同样的搜索时间下得到更高的下界，足以支持基于不匹配的风险限定审计（RLA），显著减少所需抽样量；

**⚠️ 局限性**

局限性在于仍属于指数级搜索，面对极大规模（数千席）选举可能不可行；仅提供下界而非精确边际；依赖现有上界启发式的质量；

---

## 367. Encoding Event-B Proof Rules in Prolog: An Interactive Sequent Prover for ProB

**arXiv ID:** 2607.21191 | [PDF](https://arxiv.org/pdf/2607.21191v1)

**作者:** Katharina Engels `[一作]` (Heinrich Heine University Düsseldorf), Michael Leuschel `[通讯]` (Heinrich Heine University Düsseldorf)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于Prolog的Event‑B序列推理器，覆盖600余条推理规则，支持交互式证明、可视化、HTML导出、证明回放以及初步自动证明；

**💡 创新点**

核心创新在于将Event‑B证明规则以Prolog规则形式编码，代码量比传统Java实现缩减一个数量级，易于维护与扩展；并将推理器与Rodin动画、ProB可视化等工具深度集成，实现可交互的证明树和交互式HTML展示；

**🔧 技术方法**

采用Prolog实现推理规则（包含推导与重写），利用Prolog的搜索与模式匹配来实现自动化推理；使用XTL模式将证明规则转化为可动画的状态机，集成Graphviz、SVG、HTML等技术进行可视化；

**📊 数据集**

使用Rodin平台生成的Event‑B证明义务（PO）数据集，并通过ProB插件导入；

**📈 对比分析**

与Java实现对比，Prolog实现行数约为Java的1/10，开发时长约为1/8；在规则数量和可维护性方面显著优于Java；自动证明尚处于原型阶段，尚未与现有SMT/Atelier‑B等外部求解器在性能上做详细对比；

**⚠️ 局限性**

限制在于自动证明仍未完全优化；部分复杂重写规则无法直接映射至Rodin的内置求解器；目前只覆盖Event‑B，尚未扩展至经典B；外部求解器调用依赖符号化过渡，可能导致性能瓶颈。

---

## 368. GLP: A Grassroots, Multiagent, Concurrent, Logic Programming Language for AI

**arXiv ID:** 2607.21189 | [PDF](https://arxiv.org/pdf/2607.21189v1)

**作者:** Ehud Shapiro `[一作]` `[通讯]` (London School of Economics and Weizmann Institute of Science), Ehud Shapiro (London School of Economics and Weizmann Institute of Science)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了Grassroots Logic Programs（GLP），一种多代理并发逻辑编程语言，设计用于实现基层平台，提供从操作语义到多代理系统的完整数学框架，并实现了基于 Dart/Flutter 的工作站和手机版。

**💡 创新点**

创新点在于：① 用读/写变量对（reader/writer）代替传统统一，消除 unification；② 引入单次出现（SO）和单读写（SRSW）约束，保证单赋值与同步；③ 定义多代理事务（maGLP）和冷呼叫（cold‑call）机制，实现无中心化的网络建立；④ 通过 AI 直接从数学规范生成代码，构建协同人机编程流程；⑤ 用类型系统与模式匹配实现安全、可验证的基层平台。

**🔧 技术方法**

使用的技术包括：并发逻辑编程、模式匹配与单赋值、事务化多代理系统、类型检查与模态系统、Dart/Flutter 实现、AI 代码生成、加密与身份验证协议。

**📊 数据集**

本文主要采用理论演示与案例实现（如社交图、社交网络、加密货币等），未使用公开数据集；主要通过构造的基层社交图实例进行验证。

**📈 对比分析**

对比方法主要是通过证明性验证与形式化安全性证明；性能评估尚未在论文中给出具体指标，作者强调实现已在工作站和手机上可运行，实际运行时效与可扩展性待进一步实验。

**⚠️ 局限性**

局限性包括：① 对普通程序员的学习曲线较陡，需 AI 辅助；② 目前实现仍在早期，缺乏大规模真实世界测试；③ 依赖单次出现约束，某些广播场景需额外手工实现；④ 论文未给出详细性能数据和可扩展性分析。

---

## 369. Identifying Good Rules for Efficient SAT Encodings of Single-Constant Multiplication Using Machine Learning

**arXiv ID:** 2607.21188 | [PDF](https://arxiv.org/pdf/2607.21188v1)

**作者:** Chufeng Jiang `[一作]` (City University of New York), Neng-Fa Zhou `[通讯]` (City University of New York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种神经符号框架，用图神经网络预测单一常数乘法（SCM）拆解中的优良算子，从而加速SAT编码过程。

**💡 创新点**

将SCM拆解问题建模为图，并利用GNN学习算子选择规则，实现可解释的剪枝；同时保持符号推理的完备性与正确性。

**🔧 技术方法**

图神经网络（GNN）、动态规划（DP）搜索、Picat逻辑编程、PyG数据处理。

**📊 数据集**

训练集包含1–65535范围内所有奇数常数以及17–32位随机取样的6000个奇数常数，共128,767个常数；测试集为17–32位未见过的1,000个常数每位，共16,000个。

**📈 对比分析**

与基线、Min‑k、纯DP四种方法对比，ML‑DP在编码时间上提升约10–100×，内存使用下降97%，分支数下降10⁴倍；编码质量仅比DP略差1–2加法，显著优于基线。

**⚠️ 局限性**

仅针对min‑k目标，未扩展到min‑a目标；依赖大量训练数据，GNN对超参数较鲁棒但仍需采集更大范围常数；规则生成阈值选择经验化，可能在某些常数上导致误剪。

---

## 370. Decoupling Cross-Modality Manifold Discrepancy: Leveraging Visible Diffusion Priors for Infrared Super-Resolution

**arXiv ID:** 2607.21174 | [PDF](https://arxiv.org/pdf/2607.21174v1)

**作者:** Yunpeng Hua `[一作]` (University of Science and Technology Beijing), Jiansheng Chen `[通讯]` (University of Science and Technology Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Shift-IISR，一种双路径扩散框架，利用 GRM 与 LSR 两个模块在冻结的 ResShift 模型上实现红外图像超分辨率，同时保持全局分布和局部结构的一致性。

**💡 创新点**

创新点：① 在冻结扩散模型上引入可学习的全局表示调制（GRM），动态纠正可见光偏移；② 每一步去噪中注入边缘先验的局部结构修正（LSR）；③ 通过时间嵌入调制与结构引导同步，使生成轨迹逐步从可见光流形迁移至红外流形。

**🔧 技术方法**

技术细节：预训练 ResShift 隐层扩散模型；全局表示提取网络 E 与分类器 C 形成 GRM；Sobel 边缘提取与归一化构成 LSR；时间调度函数 γ_shift(t) 与 γ_str(t)；联合损失 ℒ_total = ℒ_simple + λℒ_cls。

**📊 数据集**

数据集：训练与评估使用 M^3FD（含 Set5/Set15/Set20 子集），在 RoadScene 与 TNO 上验证；下游任务（目标检测、语义分割）使用 M^3FD 与 MSRS。

**📈 对比分析**

比较方式：对比多种 RGB 与红外 SR 方法（ResShift、SinSR、BI-DiffSR、ATD、MambaIR、CoRPLE、InfraFFN、DifIISR 等），使用 PSNR、SSIM、LPIPS 等指标。Shift-IISR 在所有子集上均取得最高 SSIM 与最低 LPIPS，PSNR 亦名列前茅；在目标检测 mAP 与语义分割 mIoU 上也显著优于对比方法。

**⚠️ 局限性**

局限性：仍依赖可见光预训练模型，虽通过 GRM/LSR 校正但在极端光照或低分辨率场景下可能出现色彩/纹理漂移；缺乏大规模红外数据导致对某些红外细节的泛化不足；未在红外域进行再训练，未能充分挖掘红外域独特的统计特征。

---

## 371. Hardware-Software Co-Design for Float16 On-Device Training on RISC-V Single-Core

**arXiv ID:** 2607.21130 | [PDF](https://arxiv.org/pdf/2607.21130v1)

**作者:** Benjamin Hubinet `[一作]` (CEA-Leti), Jean-Baptiste Rigaud `[通讯]` (Mines Saint-Etienne)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于RISC‑V单核支持Zfh和Zvfh扩展的开源框架，能够在资源受限的单核FPGA上完成完整的float16设备端训练，并在NaxRiscv核心实现了Zfh与完整Zvfh VPU。

**💡 创新点**

创新点包括：① 将AIfES框架与NMSIS‑DSP手工向量化的float16算子结合，实现高效的float16 ODT；② 为迁移学习提供层冻结机制；③ 在NaxRiscv上以极低面积开销（+1.15% LUT6）实现Zfh和完整Zvfh，展示了硬件与软件协同的可行性；④ 通过float16显著降低内存占用（约50%）而保持训练精度。

**🔧 技术方法**

使用技术：RISC‑V Zfh/Zvfh指令集扩展、AIfES框架、NMSIS‑DSP手工向量化、SpinalHDL HDL设计、FPGA实现、PyTorch/TensorFlow模型转换器、Adam/SGD优化器。

**📊 数据集**

使用的数据集：MNIST和EMNIST。

**📈 对比分析**

方法：在RISC‑V ISA模拟器上训练5层MLP（batch 128，Adam，lr 1e-3），对比float32训练结果，发现损失与准确率几乎相同；在迁移学习场景下，比较float16与float32的内存占用，显示参数内存下降50%；与PULP‑Trainlib和原AIfES对比，证明float16 ODT在单核环境下可实现并保持性能。

**⚠️ 局限性**

局限性：仅在ISA模拟器上验证，缺乏真实FPGA的速度评估；测试网络仅为小型MLP，未覆盖更大模型或更复杂数据集；Zvfh向量化性能尚未量化；对其它RISC‑V核心或多核平台的可移植性未进行深入验证。

---

## 372. GlucoTune: A Unified Framework for Blood Glucose Preprocessing, Forecasting, and Benchmarking in Diabetes

**arXiv ID:** 2607.21117 | [PDF](https://arxiv.org/pdf/2607.21117v1)

**作者:** Davide Marelli `[一作]` (University of Milano-Bicocca), Paolo Napoletano `[通讯]` (University of Milano-Bicocca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个统一、可扩展的框架GlucoTune，用于血糖时间序列数据的可重复预处理、预测模型训练与基准评估，并提供图形界面；

**💡 创新点**

创新点在于通过可配置的YAML管线实现无数据泄露的可重复预处理、统一数据包装与评估，整合多类统计、机器学习与深度模型，搭建公开排行榜并支持可视化与无代码操作；

**🔧 技术方法**

技术包括Python多包架构、YAML配置驱动的流水线、数据清洗（缺失填补、对齐、滤波、SMOTE等）、归一化、时间窗口切片、数据拆分策略（时间、分层、混合）、多模型库（ARIMA、XGBoost、Transformer、TCN、N-BEATS等）以及GUI可视化；

**📊 数据集**

使用四个公开T1DM数据集：OhioT1DM、DiaTrend、T1DiabetesGranada、T1DEXI，主要实验在OhioT1DM与DiaTrend上进行；

**📈 对比分析**

通过统一的基准实验（时间拆分、max归一化、MSE优化）对多模型在30/60分钟预测窗下进行比较，报告RMSE、MAE、MARD、TG、敏感度/特异度、PCC、临床误差网格等多维指标，结果显示模型差异主要受预处理与设置影响，深度模型与传统模型在精度、临床安全性与计算成本上呈现不同权衡；

**⚠️ 局限性**

局限性包括：当前仅支持四个T1DM数据集，未覆盖T2DM及更大规模数据；对预处理参数的选择仍需人工调优；缺乏对长期预测和自适应模型的评估；框架的可扩展性虽强，但实际使用仍需一定编程基础。

---

## 373. AttriMem: Attribution-Guided Process Feedback for Agent Memory Learning

**arXiv ID:** 2607.21106 | [PDF](https://arxiv.org/pdf/2607.21106v1)

**作者:** Qinfeng Li `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于token级归因的过程反馈框架，用于强化学习训练长对话记忆构建策略。

**💡 创新点**

创新点在于将最终答案的token贡献通过归因方法映射回中间记忆操作，提供细粒度过程奖励，突破传统仅靠最终或动作级奖励的稀疏瓶颈。

**🔧 技术方法**

主要技术包括ContextCite等归因方法、GRPO强化学习、token级奖励计算、四类记忆模块（core、epi、sem、proc）以及对话记忆构建和检索的整体框架。

**📊 数据集**

实验使用LongMemEval进行训练，跨域评估使用LoCoMo和PerLTQA三个基准数据集。

**📈 对比分析**

与训练无关检索、启发式记忆、以及现有RL记忆基线进行对比，在三大基准上均优于MemBuilder，提升约2.47/1.50/0.75分，并在不同回答模型间保持稳健的性能。

**⚠️ 局限性**

局限性包括：需要可分词的文本记忆表征，归因误差可能导致奖励偏差；计算上需多次掩码评估，虽可并行但仍增加训练成本；目前未直接适用于向量或图结构记忆。

---

## 374. Solving Large Traveling Salesman Problems (TSPs) by a Recursive Clustering Algorithm and a Scalable FPGA-Based Implementation

**arXiv ID:** 2607.21182 | [PDF](https://arxiv.org/pdf/2607.21182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 375. Construction and Dynamic Update of Channel Gain Maps via 3D Gaussian Splatting

**arXiv ID:** 2607.21099 | [PDF](https://arxiv.org/pdf/2607.21099v1)

**作者:** Yilong Chen `[一作]` (Chinese University of Hong Kong (Shenzhen)), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于三维高斯散射（3D Gaussian Splatting）的网格平均信道增益图（CGM）构建与动态更新框架，利用物理可解释的高斯原语来渲染和更新覆盖环境中的大尺度信道增益。

**💡 创新点**

创新点包括：① 将网格平均信道增益拆解为距离衰减、路径透射和有效散射三部分，并用高斯原语近似这些物理量；② 设计增量式更新机制，冻结静态模型只优化少量活跃高斯原语，实现稀疏测量下的高效更新；③ 提出低复杂度的 MLP-GS 变体，用轻量级 MLP 替代显式散射计算，兼顾准确性与速度。

**🔧 技术方法**

主要技术：三维高斯散射（3DGS）、物理信息驱动的渲染公式、梯度下降训练、适应性密度控制（ADC）、残差引导的活跃原语初始化、轻量级 MLP 解码器。

**📊 数据集**

使用基于 Sionna‑RT 的仿真数据，构建校园、道路和低空三种测试场景，场景尺寸分别为 100×100 m²、50×50 m² 和 300×300 m²；每个场景的 CGM 以 50×50 网格划分，收集 25% 的训练样本和 4% 的更新样本。

**📈 对比分析**

与 MLP‑RF、虚拟散射（VS）和克里金插值等基线对比，采用 dB 级 MAE 作为指标。GS‑CG 在所有场景下的 MAE 均低于 1 dB（构建）和 2.5 dB（更新），远优于基线；MLP‑GS 仅略高于 GS‑CG；MLP‑RF、VS 和 Kriging 的误差显著更大。训练时间方面，GS‑CG 构建约 280 s、更新约 47 s；MLP‑GS 约 58 s/22 s；MLP‑RF、VS、Kriging 更快但精度低。

**⚠️ 局限性**

局限性包括：① 需要一定的环境几何先验或初始高斯原语；② 目前仅对网格平均信道增益有效，无法直接处理精细相位信息；③ 对极大规模场景（如城市级）或极低采样率下的泛化能力尚未完全验证；④ 增量更新需稀疏测量，若动态变化范围过大仍可能出现误差。

---

## 376. Training Large Language Models for Self-Explanation Faithfulness

**arXiv ID:** 2607.21090 | [PDF](https://arxiv.org/pdf/2607.21090v1)

**作者:** Yeoktatt Cheah `[一作]` (University College London), Oana-Maria Camburu `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于强化学习（RL）的训练框架，直接通过对模型产生的自解释与内部决策的因果相关性（Phi‑CCT）来优化自解释的真实性；

**💡 创新点**

创新点在于把传统评估用的对抗干预相关性转化为可在线计算的RL奖励，从而实现模型参数直接学习“识别并披露”对决策有影响的因子，首次在LLM自解释中实现可训练的解释真实性；

**🔧 技术方法**

使用的核心技术包括：1）生成对抗性干预（随机词插入与用户偏好插入）来构造因果测试样本；2）用GRPO（Group Relative Policy Optimization）进行策略优化；3）对比SFT（监督微调）来识别因果影响；4）对奖励进行去奖励游戏的评估（完成长度与重叠率）；

**📊 数据集**

实验数据集为e‑SNLI、Social‑IQA、ComVE、StrategyQA，共约2,000个样本/拆分，分别在训练、测试和OOD（ComVE、StrategyQA）集上评估；干预类型包括随机插入和用户偏好插入；

**📈 对比分析**

与基线（原始LLM）相比，RL训练将Phi‑CCT从接近0提升至0.536（随机插入）/0.664（用户偏好）；在OOD数据上亦可达0.691；SFT仅提升因果检测准确率，却未改善解释披露；跨干预迁移有限，但Llama3.1在随机插入训练后对用户偏好干预仍能得到非零Phi‑CCT；

**⚠️ 局限性**

局限性包括：1）奖励基于预先冻结模型的决策，可能与训练过程中的决策漂移不一致；2）样本规模有限（仅4k样本/模型），且仅测试8B规模模型；3）干预插入仅为单词，未覆盖更复杂插入；4）提及检测依赖词形匹配，忽略同义或重述；5）未评估解释的可读性与任务性能的权衡。

---

## 377. Differentiable Logic Programming to Mitigate Reasoning Shortcuts in Neurosymbolic Systems

**arXiv ID:** 2607.21185 | [PDF](https://arxiv.org/pdf/2607.21185v1)

**作者:** Akihiro Takemura `[一作]` (National Institute of Informatics), Katsumi Inoue `[通讯]` (National Institute of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于矩阵的可微分逻辑程序化方法，用于降低神经符号系统的捷径推理。

**💡 创新点**

创新点在于统一将约束与规则编码到单一矩阵并实现一一对应的原子映射，从而提供直接梯度路径。

**🔧 技术方法**

使用矩阵编码逻辑程序、t‑norm 对比、软交叉熵损失及神经网络输出与逻辑解释的可微耦合。

**📊 数据集**

主要在MNIST相关的基准任务（MNIST‑Half、MNIST 6/9、MNIST Addition）上进行实验。

**📈 对比分析**

与LTN、Semantic Loss、DeepProbLog、NeurASP等基线比较，矩阵方法在约束满足与概念学习上均显著提升，尤其在缺失类的分类准确率高达96%以上。

**⚠️ 局限性**

局限性包括仅处理命题程序、实验局限于MNIST、矩阵存储与计算的可扩展性待改进。

---

## 378. CRAG-MM-Diagnostics: Enabling Stage-Wise Analysis of Knowledge-Intensive VQA

**arXiv ID:** 2607.21155 | [PDF](https://arxiv.org/pdf/2607.21155v1)

**作者:** Hanseok Oh `[一作]` (New York University), Verna Dankers `[通讯]` (McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向知识密集视觉问答（KI-VQA）的诊断基准 CRAG-MM Diagnostics，并通过在现有数据集上添加目标框、表达式类型等多维度注释，实现对语言驱动视觉定位、目标识别以及知识检索与推理等三阶段的细粒度评估。

**💡 创新点**

创新点在于首次将阶段化评估与视觉复杂度、表达式歧义、目标受欢迎度等元数据结合，揭示了 KI-VQA 瓶颈并验证了基于视觉定位的 RAG 模型可显著提升性能。

**🔧 技术方法**

采用了多种技术，包括多模态检索增强生成（RAG）、视觉定位模型（如 Grounding‑DINO、Open‑Vocabulary 检测器）、大型视觉语言模型（如 LLaVA、InstructBLIP、GPT‑4 等）以及标注工具与对比学习等。

**📊 数据集**

使用的数据集为 CRAG‑MM Diagnostics，基于原始 CRAG‑MM 数据集进行扩充，包含近 1.15k 张单轮 egocentric 图像、问答对、实体信息、目标框等。

**📈 对比分析**

通过与多种基线模型对比，评估了纯参数式 VLM、专门定位模型、检索增强模型和结合视觉定位的双模 RAG，结果表明知识检索和推理是最主要瓶颈，在加入视觉定位后模型准确率提升 13.3%（LLaVA）和 8.5%（InstructBLIP）。

**⚠️ 局限性**

主要局限在于当前检索系统在细粒度目标识别和多跳推理方面仍表现欠佳，且对表达式歧义的自动消解依赖人工标注，未来需要更鲁棒的多模态检索和推理模块。

---

## 379. Investigating Codec-Internal Latent Audio Watermarking for Neural Codec Robustness

**arXiv ID:** 2607.21132 | [PDF](https://arxiv.org/pdf/2607.21132v1)

**作者:** Zi Hu `[一作]` (Duke Kunshan University), Ming Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究将音频水印直接嵌入到类似编解码器的连续潜在空间，提升对神经音频编解码器（如EnCodec）的鲁棒性。

**💡 创新点**

首次在编码器-解码器前的连续潜在层插入水印，并通过RVQ引导分解与前层保护机制，探讨潜在嵌入与鲁棒性、音质之间的权衡。

**🔧 技术方法**

采用SEANet风格的编码器-解码器、Conformer基消息嵌入器、RVQ-guided潜在分解、潜在域检测器，并结合多尺度mel、STFT、感知、对抗与消息损失进行联合训练。

**📊 数据集**

使用Emilia数据集的英文子集，采样48kHz的语音片段进行训练与评估。

**📈 对比分析**

通过平衡、EnCodec-24k聚焦和EnCodec-重负荷三种训练策略对比，利用PESQ、ViSQOL评估音质，利用位准确率评估鲁棒性；结果显示EnCodec-24k聚焦训练将鲁棒性从78.8%提升至95.6%，但PESQ略降；重负荷训练进一步提高鲁棒性至97.1%，但音质略低。

**⚠️ 局限性**

对EnCodec-16k的鲁棒性仍不足，水印未嵌入离散码字，连续潜在嵌入在提升鲁棒性的同时仍需权衡音质，且未提供通用水印基线。

---

## 380. Sender and Receiver Energy Consumption in a Sensor Network

**arXiv ID:** 2607.21095 | [PDF](https://arxiv.org/pdf/2607.21095v1)

**作者:** J M Fourneau `[一作]` (University of Paris-Saclay), F. Quessette `[通讯]` (University of Paris-Saclay)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了能量包网络中发送方和接收方均需消耗能量的情形，证明了在满足流方程解的前提下系统的稳态分布具有乘积形式，并给出了求解流方程的数值迭代算法。

**💡 创新点**

将能量包网络的消耗扩展到双端（发送端和接收端），在一般拓扑下给出了流方程存在性与收敛性的充分条件，并提出了处理循环结构导致的“灰区”现象的迭代修正方法。

**🔧 技术方法**

采用马尔可夫链全平衡分析、Brouwer不动点定理、迭代收敛性证明（几何级数）以及几何级数收敛性分析等数学工具。

**📊 数据集**

实验使用人工生成的参数集（如7节点完整路由网络、2节点循环网络等）进行数值模拟，并未使用公开数据集。

**📈 对比分析**

通过对比迭代过程与理论收敛条件，验证在满足充分条件时收敛速度很快；若不满足条件，算法会进入灰区并停滞，说明对参数选择的敏感性。

**⚠️ 局限性**

主要局限在于：需要先确认流方程解位于内部，否则可能无解或收敛至边界；算法在含有灰区的拓扑下不一定收敛；实验规模有限，缺乏对真实大规模物联网网络的验证。

---

## 381. A Polynomial Architecture-Attribution Co-Design Framework for Exact Aumann-Shapley Attribution in GNNs

**arXiv ID:** 2607.21094 | [PDF](https://arxiv.org/pdf/2607.21094v1)

**作者:** Bizu Feng `[一作]` (Fudan University), Zixin Hu `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种针对图神经网络的精确 Aumann–Shapley 归因框架 APEX，提出 PolyGIN 多项式 GNN 并利用 Gauss–Legendre 积分实现无截断路径归因。

**💡 创新点**

创新点在于模型与归因共设计：通过把 GNN 转化为多项式结构，得到归因积分的有限次数极点，使用固定数量的 Gauss–Legendre 点即可精确求值，消除传统 IG 的积分误差。

**🔧 技术方法**

采用 PolyGIN 的多项式激活与线性归一化、Gauss–Legendre 高斯–勒让德积分以及特征级归因聚合到节点级的技术。

**📊 数据集**

在五个标准图数据集上验证：2*BA‑Shapes、2*BBBP、2*BACE、2*Graph‑SST2、2*Mutagenicity。

**📈 对比分析**

与 GNNExplainer、PGExplainer、GradCAM、FlowX、标准 IG 等基线比较，APEX 在预测准确率、归因可信度（Fidelity+/-）均优于对手，同时在完整性误差与计算成本上显著领先。

**⚠️ 局限性**

局限在于仅适用于多项式前置 logit 的 GNN，无法直接应用于包含 ReLU、softmax、批归一化、注意力等非多项式组件的网络，并且缺乏对人类可解释性的语义验证。

---

## 382. Case study: proving sqrt(2) irrational with LPTP and an LLM

**arXiv ID:** 2607.21187 | [PDF](https://arxiv.org/pdf/2607.21187v1)

**作者:** Fred Mesnard `[一作]` (université de La Réunion), Wim Vanhoof `[通讯]` (Université de Namur)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 LPTP（逻辑程序定理证明器）框架内，利用 Claude LLM 交互式生成证明步骤，并通过 LPTP 证明检查器验证，最终完成了 √2 无理性的正式证明。

**💡 创新点**

首次将大语言模型与 LPTP 结合，实现了证明步骤生成与检查器反馈相互作用的自动化流程，弥补了 LPTP 与现有 LP 证明工具之间的交互缺口。

**🔧 技术方法**

技术包括 LPTP 证明检查器、Anthropic 的 Claude Opus 4.5 LLM、自然演绎证明生成、反馈循环、以及基于 FOF 的现成一阶自动定理证明器（Vampire、E）。

**📊 数据集**

使用了 LPTP 自带的 nat 与 lists 库、作者自制的 sqrt2 证明文件、以及 LPTP 用户手册（PDF）作为训练/提示数据；实验数据为 √2 证明的各子命题。

**📈 对比分析**

与 ATPs 的比较：在 20 秒超时内，Vampire/E 在中间版本 10 句子中 5/10、最终版本 12 句子中 9/12 能求解；LPTP 证明检查速度快且能给出错误步骤；整体性能表明 LLM+LPTP 在证明完整性上更可靠，但在证明搜索效率上仍落后。

**⚠️ 局限性**

局限性包括：仍需人工挑选证明顺序；部分子命题对 LLM 仍难以自动证明；依赖 LPTP 语法与库的可维护性；LLM 的幻觉需要证明检查器捕捉；整体自动化程度有限。

---

## 383. Safety-oriented sidewalk and road segmentation for smartphone-based assistive navigation

**arXiv ID:** 2607.21137 | [PDF](https://arxiv.org/pdf/2607.21137v1)

**作者:** Hakan Calim `[一作]` (Friedrich-Alexander-University Erlangen-Nuremberg), Andreas Maier `[通讯]` (Friedrich-Alexander-University Erlangen-Nuremberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了面向智能手机盲人/低视障人士辅助导航的安全导向语义分割框架，并在新构建的SENSATION-DS数据集上进行评估

**💡 创新点**

创新点在于：①从安全视角引入道路-人行道错误率作为误判评估指标；②统一构建面向行人视角的九类标签体系；③在训练阶段采用分阶段目标域适配、合成图像和SAM2伪标签的组合，系统性比较其对准确率和安全误判的影响

**🔧 技术方法**

采用了深度学习语义分割架构（DeepLabV3Plus、UPerNet、FPN、PAN、SegFormer）与MobileNetV3/轻量化Transformer编码器，并使用Diffusion + ControlNet生成合成图像、SAM2生成伪标签、Albumentations增广以及ONNX导出进行Android性能基准

**📊 数据集**

主要数据集包括：新收集的2,752张胸部高度行人视角图像（SENSATION-DS）、Mapillary Vistas、Hugging Face的Sidewalk图像、以及整合的ApolloScape、Cityscapes、SANPO、SideGuide等外部城市/人行道数据集，统一映射到SENSATION-DS的9类标签

**📈 对比分析**

通过在验证集上进行多阶段训练（source→target、Synthetic、SAM2、SAM2→Synthetic、Synthetic→SAM2）并在测试集上评估mIoU、道路/人行道IoU、Road-as-Sidewalk错误率、Critical False Safe Rate；同时将四个精选模型导出为ONNX，在Android设备上测量FPS和IoU，结果显示DeepLabV3Plus-MobileNetV3在512×384分辨率下兼具最快帧率(7.383 FPS)与较低的Road-as-Sidewalk错误率(0.079)，而UPerNet-MobileNetV3在mIoU上最高(0.715)，但错误率较高

**⚠️ 局限性**

局限性包括：①安全误判指标仍为离线代理，未通过真实导航实验验证其与用户安全的直接关联；②合成图像与SAM2伪标签的质量控制有限，可能引入边界错误；③实验仅在特定城市环境下进行，泛化到多样化街景与不同设备仍需进一步验证

---

## 384. HalluScope: Fine-grained Hallucination Diagnosis for Multimodal Large Language Models

**arXiv ID:** 2607.21105 | [PDF](https://arxiv.org/pdf/2607.21105v1)

**作者:** Weilin Jin `[一作]` (Peking University), Zhonghai Wu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多模态大语言模型的细粒度幻觉诊断任务，包括检测、分类和可解释说明；

**💡 创新点**

构建了大型诊断数据集 HalluScope-30K，并设计了多粒度联合奖励函数，实现检测与分类协同优化；

**🔧 技术方法**

采用自动化数据生成流水线、基于闭源模型的幻觉注入与标注、强化学习（GRPO）以及结构化 XML 输出；

**📊 数据集**

整合了八个公开数据集（如MMBench、MathV360K、OCRBench 等）和五大任务类别；

**📈 对比分析**

在 MHALO 检测基准和自建分类基准上均取得 SOTA，HalluScope-8B 的 F1_M 平均达 64.03，F1_Macro 51.66，显著优于 Gemini-3-Pro、Qwen3-VL 等基线；

**⚠️ 局限性**

依赖闭源模型进行幻觉注入与标注，导致可复现性受限，且对开放源代码模型的适用性尚待验证。

---

## 385. Loss Landscape Topology Reveals Why Simple Baselines are Competitive at 3D Point Cloud Segmentation Under Class Imbalance

**arXiv ID:** 2607.21089 | [PDF](https://arxiv.org/pdf/2607.21089v1)

**作者:** Antonis Savva `[一作]` (University of Cyprus), Theocharis Theocharides `[通讯]` (University of Cyprus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

系统评估了11种类不平衡缓解方法在3D点云语义分割中的效果，并通过误差模式、决策边界变异及损失景观分析给出了机制性解释。

**💡 创新点**

首次揭示在点云分割中标准交叉熵已能接近最优，且不平衡严重程度决定优化景观的拓扑，使得专门化损失难以带来显著提升。

**🔧 技术方法**

采用混合误差分析、决策边界可变性评估、权重扰动与Hessian特征分析等技术；使用KPConv和RandLA‑Net两种点云分割网络进行实验。

**📊 数据集**

实验数据集包括极端不平衡的DALES（641:1）和中等不平衡的S3DIS（56:1）两组LiDAR点云数据集。

**📈 对比分析**

与均匀权重的交叉熵相比，其他10种重加权/自定义损失在mIoU上差距仅0.6–3.3%；极端不平衡下存在窄阱，方法偏差易导致性能骤降；中等不平衡下各方法表现相近。

**⚠️ 局限性**

仅验证了两种点云网络，未考察体素或Transformer架构；机制分析仅基于KPConv，可能不适用于其他网络；仅关注损失级干预，未结合数据级策略。

---

## 386. A Unified Moral-Value Dataset for Instruction Tuning

**arXiv ID:** 2607.21279 | [PDF](https://arxiv.org/pdf/2607.21279v1)

**作者:** Zhaohui Zeng `[一作]` (RWTH Aachen), Florian Mai `[通讯]` (University of Bonn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的道德价值指令调优数据集，并在大型语言模型上进行指令调优实验，探究价值数据与一般任务数据混合比例对性能的影响。

**💡 创新点**

提出多阶段管道：将多种价值框架数据统一格式化、利用生成器补全缺失标签，并通过自提示生成指令模板，形成可直接用于指令调优的完整数据集。

**🔧 技术方法**

使用 ModernBERT 生成缺失的价值标签；自提示（prompt engineering）生成指令–响应模板；Open‑instruct（TULU3）SFT 训练；vLLM 进行结构化输出与批量推理；评估采用 OLMES 与 Value‑Action Gap benchmark。

**📊 数据集**

价值数据源为 ETHICS、UNIMORAL、SOCIAL‑CHEM‑101；一般任务数据来自 TULU‑3 SFT dataset；模型使用 Qwen3‑1.7B‑base。

**📈 对比分析**

在不同价值数据混合比例（0–100%）下训练并使用 OLMES 评估一般任务，平均得分保持在 70.8–71.2 分；在 Value‑Action Gap 任务中，F1 最高为 0.8521（10% 价值数据），过多价值数据会导致性能下降；结果显示指令调优提升整体能力，价值对齐需要适度平衡。

**⚠️ 局限性**

数据覆盖仅限三大价值框架，继承原始数据偏见；生成器可能引入噪声；实验仅在小模型和单次实验，缺乏对大模型、多种种子、非英语语言等情况的验证。

---

## 387. If Edge Coloring is Hard under SETH, then SETH is False

**arXiv ID:** 2607.21276 | [PDF](https://arxiv.org/pdf/2607.21276v1)

**作者:** Alexander S. Kulikov `[一作]` (JetBrains Research), Ivan Mihajlin `[通讯]` (JetBrains Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了边着色问题的复杂性，提供了一个否定的答案，表明在假设已知算法是最优的情况下，边着色问题不能在时间α^n^2内解决。

**💡 创新点**

创新点在于证明了如果存在一个细粒度的归约显示边着色问题不能在时间α^n^2内解决，那么假设是错误的，从而揭示了边着色问题与SAT等问题之间的复杂性联系。

**🔧 技术方法**

使用了细粒度归约的技术，结合了自我归约的概念来分析问题的复杂性。

**📊 数据集**

没有具体提到使用的数据集，但讨论了边着色问题和其他已知的复杂性问题，如SAT、3-SUM和APSP。

**📈 对比分析**

通过细粒度归约的方法，论文表明如果边着色问题的复杂性被证明为α^n^2，那么将导致SAT等问题的复杂性被低估，从而与已知的复杂性假设相矛盾。

**⚠️ 局限性**

限制在于该结果依赖于假设已知算法的最优性，且未能提供具体的边着色问题的有效算法或更快的解决方案。

---

## 388. Flash EQ-Linear: Accelerating Equivariant Linear Layers via Group-wise Discrete Fourier Transform

**arXiv ID:** 2607.21271 | [PDF](https://arxiv.org/pdf/2607.21271v1)

**作者:** Zhongchen Zhao `[一作]` (Xi'an Jiaotong University), Zongben Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 Flash EQ-Linear 算法，通过将等变线性层视为群维度上的循环卷积并利用傅里叶卷积定理与共轭对称性实现精确加速，从而显著提升等变网络的计算效率。

**💡 创新点**

创新点在于：①将等变线性层重构为群维度的循环卷积；②利用离散傅里叶变换的卷积定理和实数 DFT 的共轭对称性实现精确无损加速；③提供完整的 CUDA kernel（FP32/FP16）实现，将理论 T/2 速度提升转化为实际 1.6–2.1× 的前向加速和 1.4–1.7× 的网络级推理加速。

**🔧 技术方法**

采用了离散傅里叶变换 (DFT)、卷积定理、实数共轭对称性、CUDA kernel 并行化、Tensor Core 优化以及数据布局变换等技术。

**📊 数据集**

使用 ImageNet-100（ImageNet1K 的 100 类子集）进行网络级性能评测，并用随机张量验证数值精度与等变性。

**📈 对比分析**

与标准 PyTorch Linear 和 Naive EQ-Linear 在相同输入、相同参数量下进行前向/后向延迟对比；FP32 前向延迟提升 1.6–2.1×，后向略低；网络级 Flash EQ-ViT/Swin 在 FP32 下实现 1.4–1.7× 的整体推理速度提升，并保持与 Naive 等变模型相同的 Top‑1 精度。

**⚠️ 局限性**

目前实现仅针对 p4 旋转群（T=4）并针对该群做了特化；其他更大或更复杂的群需要重写 DFT 与 kernel；后向 kernel 仍有进一步优化空间。

---

## 389. Filter Learning for Subgraphs: Algebras and Performance Risk Bounds

**arXiv ID:** 2607.21263 | [PDF](https://arxiv.org/pdf/2607.21263v1)

**作者:** Purui Zhang `[一作]`, Wee Peng Tay `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了在仅观测到图的子图情况下学习子图滤波器的框架——子图滤波器学习（Subgraph Filter Learning, SFL）

**💡 创新点**

创新点在于：①构造距离感知的子图偏移算子（DSSO）和非交换的子图滤波器代数（SFFA），通过多尺度距离子图实现对原图全局信息的逼近；②给出有限样本下的风险上界，分解为不可约误差、近似误差、正则化偏差与估计误差，揭示了距离、滤波器维度与样本量的三角权衡；③通过结构化滤波器空间大幅提升了学习效率与可解释性。

**🔧 技术方法**

技术手段包括：基于图的距离构造的DSSO、矩阵代数与多项式滤波器融合的SFFA、岭回归的最小二乘学习、子图信号的WSS假设、次高斯噪声模型以及风险分解与上界推导。

**📊 数据集**

实验使用的真实数据集为：METR‑LA（城市交通流量）和Windmill（风速）数据集，并在合成/半合成的子图过滤、信号重建与多输入多输出预测任务中验证。

**📈 对比分析**

与基线方法（数值LMMSE、单一GSO多项式滤波、距离‑k拉普拉斯多项式、随机子图拉普拉斯代数）以及单节点AR模型对比，SFFA在所有任务与子图比例上均表现出显著更低的损失/均方误差，并通过Holm‑Bonferroni校正的显著性检验得到统计优势。

**⚠️ 局限性**

局限性包括：只针对线性滤波器；要求输入信号满足子图WSS且噪声次高斯；对训练样本量敏感；目前仅处理静态图结构，未考虑时变图或更高阶结构；代数复杂度随子图尺寸与阶数上升，实际实现需权衡。

---

## 390. slang.gr as a Large-Scale Crowdsourced Resource for Non-Standard Greek

**arXiv ID:** 2607.21255 | [PDF](https://arxiv.org/pdf/2607.21255v1)

**作者:** Panagiotis Papadakos `[一作]` (ICS-FORTH), Dimitris Plexousakis `[通讯]` (ICS-FORTH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对希腊非标准语言词典slang.gr进行大规模计算分析，构建了多层语义与元数据分类体系，并基于用户角色、交互与审核信号提出定义置信度评估方法。

**💡 创新点**

创新点在于①首次将大量杂乱的用户生成标签映射为两层语义+元数据分类；②结合社区结构与交互情绪提出多维置信度指标；③系统化评估希腊俚语的形态、语义与社会维度。

**🔧 技术方法**

使用了大规模文本处理、LLM辅助标签映射、图网络（Leiden算法、UMAP可视化）、TF‑IDF+余弦相似度、情感分析（HellenicSentimentAI）以及综合置信度公式。

**📊 数据集**

主要数据集为slang.gr的28,384个sense条目、649个标签、17,942用户、101,957条评论，涵盖词形、语义、地域、时间等多维信息。

**📈 对比分析**

通过比较基于标签、语义与元数据的用户社区图的模数、互信息以及置信度分布，发现语义图得分更连贯、元数据图得到更高的模数；两种置信度指标相关性约0.67，前10%覆盖率约36%。

**⚠️ 局限性**

局限包括：①标签映射依赖LLM和人工校对，仍可能存在误差；②置信度指标未在下游任务中系统评估；③对非标准语言的时序演变与跨语种迁移分析不足；④数据主要集中在互联网语境，缺少正式场景对比。

---

## 391. Search Hardness-Aware LLM-Based Problem Formulation for Expensive Simulation-Driven Design

**arXiv ID:** 2607.21220 | [PDF](https://arxiv.org/pdf/2607.21220v1)

**作者:** Yuchen Li `[一作]` (Xidian University), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种搜索硬度感知的LLM驱动问题公式化框架SHA-PF，用初始评估数据识别稀有且有潜力的搜索状态，从而引导优化器更快达到目标；

**💡 创新点**

创新点在于将搜索硬度（稀有度+进展潜力）作为公式化评价指标，使用anchor状态引导公式搜索，并结合LLM生成、修复和进化优化，实现更高搜索效率；

**🔧 技术方法**

使用LLM（GPT-5-mini等）进行公式生成与修复，Anchor状态选择基于搜索硬度评分，进化搜索结合多种变异算子；

**📊 数据集**

在真实的多目标校准问题（HBV）和五个昂贵天线设计基准（HAE、HSL、HSE、CSL、WHG）上进行实验；

**📈 对比分析**

与人工专家公式和直接LLM公式（四个基线）以及不同优化器（SCBO、CEBO、DSI）对比，SHA-PF在所有任务中均需要更少的高频模拟评估和搜索时间，成功率最高；

**⚠️ 局限性**

对初始评估数据稀疏或缺失关键硬状态时anchor选择可能不可靠，且目前仅使用二元满足状态，未考虑满足度量的程度。

---

## 392. DART: A Degradation-Aware Recurrent Transformer for Archival Film Restoration

**arXiv ID:** 2607.21219 | [PDF](https://arxiv.org/pdf/2607.21219v1)

**作者:** Mikołaj Jastrzębski `[一作]` (Wrocław University of Science and Technology), Kamil Adamczewski `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 DART，一种衰退感知递归Transformer，用软缺陷掩码引导时间融合并通过 AdaLN‑Zero 对修复网络进行动态条件化，实现档案电影的高质量去噪与恢复。

**💡 创新点**

创新点包括：1) 直接监督连续软缺陷掩码，利用多尺度膨胀金字塔 MaskNet 捕获结构与强度；2) 在 Swin Transformer 中引入 AdaLN‑Zero 条件化，使模型根据整体衰退程度自适应；3) 在时间维度上保持缺陷掩码一致性，避免传统方法的闪烁与误检。

**🔧 技术方法**

核心技术包括：递归Transformer（Swin）+ RAFT 光流估计；膨胀金字塔 MaskNet + 软掩码监督；AdaLN‑Zero 条件化；感知损失、对抗损失与 Temporal‑PatchGAN 结合的端到端训练。

**📊 数据集**

训练数据：REDS 720p 视频 + 在线 Synthetic Degradation pipeline 生成伪标注；评测数据：无参考档案影片集合 AbsoluteDegradation（13,252帧）和 SRWOV（216帧 JPEG）。

**📈 对比分析**

与 BasicVSR、BasicVSR++、ShiftNet、DeepRemaster、RTN、MambaOFR 等基线重新训练对比；在 AbsoluteDegradation 上所有无参考指标均优于基线，SRWOV 上 MUSIQ 与 MANIQA 仍保持第一；模型参数仅 6.6M，速度与显存均优于大多数对手。

**⚠️ 局限性**

局限性：仅针对黑白影片；对光流误差敏感，极端遮挡或持续缺陷仍难以处理；缺乏正式人类评估，颜色恢复与更广泛评测仍待验证。

---

## 393. ARGON: A GNN-Empowered Compilation Framework for Scalable Neutral Atom Computing

**arXiv ID:** 2607.21216 | [PDF](https://arxiv.org/pdf/2607.21216v1)

**作者:** Wenjie Sun `[一作]` (University of Electronic Science and Technology of China), Guowu Yang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ARGON编译框架，通过离线预生成空间布局库与基于图神经网络的预测模型，分离空间与时间约束，实现中性原子量子计算机的高效编译；

**💡 创新点**

核心创新点是：①离线使用SMT求解生成高并行度的空间布局库，消除编译时空间冲突搜索；②采用GNN预测跨层路由，具备前瞻性避免后续运动瓶颈；③三阶段解耦架构（离线布局、GNN预测、启发式路由），大幅提升编译速度与执行保真度；

**🔧 技术方法**

使用技术包括：Satisfiability Modulo Theories (SMT)求解、Graph Neural Network (GNN)预测模型、启发式并行路由算法、深度仿真回放训练数据集；

**📊 数据集**

实验数据集涵盖：16×16、20×20、28×28规模的随机门、W‑state、Quantum Volume、3‑Regular、QASMbench等多种量子电路；

**📈 对比分析**

与DasAtom、Enola、PowerMove、Stade等基线对比，ARGON平均编译时间0.68 s，速度提升>600×；Rydberg阶段数下降>10⁴×；物理执行保真度提升1–2个数量级，证明在多种规模和复杂度下均优于现有方案；

**⚠️ 局限性**

局限性包括：需要一次性离线生成布局库和训练GNN，受硬件参数（如交互半径、禁区半径）约束；对极大规模阵列或极长深度线路的泛化仍需验证；对多区间或错误纠正等更复杂体系结构的适配性有限；

---

## 394. Learning-based Seam Correspondence Reconstruction in Sewing Patterns

**arXiv ID:** 2607.21213 | [PDF](https://arxiv.org/pdf/2607.21213v1)

**作者:** Zhendong Wang `[一作]` (Style3D Research), Huamin Wang `[通讯]` (Style3D Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图神经网络的两级学习框架，自动从无缝注释的二维缝纫图案中恢复面板语义、面板连通性以及细粒度缝线对应关系。

**💡 创新点**

创新点包括：① 先通过面板语义预测构造粗粒度面板图，为细粒度缝线推断提供结构先验；② 在面板图上使用图注意力网络和U‑Net相结合的方式，生成融合局部几何与全局上下文的边缘嵌入；③ 采用边缘特征（对称性、方向一致性）和多尺度图像表示，显著提升复杂拓扑（如多对一、弯曲缝线）的识别。

**🔧 技术方法**

技术手段：EfficientNet+CNN提取面板特征；多层GAT进行全局信息聚合；U‑Net编码器-解码器实现细粒度缝线预测；边缘初始化与全局上下文融合；混合精度训练、AdamW优化；两阶段训练（局部预训练 + 全局微调）。

**📊 数据集**

使用自行构建的缝纫图案数据集，包含约5.8k件衣物（2,596 件连衣裙、393 对裤子、1,910 件上衣），并附有完整的缝线标注；此外还准备了90件未见款式作为外部泛化测试。

**📈 对比分析**

与传统启发式匹配方法和单层学习模型对比，本文方法在面板语义准确率达到90.9%，面板连通性F1=0.9073，细粒度缝线像素Dice=0.8493，平均相对长度误差0.183，重叠率仅0.0072；在未见款式测试中仍保持高精度，证明了强泛化能力。

**⚠️ 局限性**

局限性：① 依赖面板语义分类，语义错误会影响缝线推断；② 目前仅支持五种基本缝线类型，无法处理自缝线、内部连接等更复杂情形；③ 数据集覆盖的衣物类别有限，进一步扩展可提升模型鲁棒性；④ 未引入用户交互反馈，后续可用于细化预测。

---

## 395. Bound-Founded Semantics for Answer Set Programming with Difference Constraints: Preliminary Report

**arXiv ID:** 2607.21201 | [PDF](https://arxiv.org/pdf/2607.21201v1)

**作者:** Pedro Cabalar `[一作]` (University of A Coruña), Philipp Wanko `[通讯]` (University of Potsdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种多排序的Bound‑Founded Here‑and‑There逻辑框架，用以统一描述结合线性/差分约束的ASP系统的语义，并给出对差分约束ASP的新的基于“已定义”原则的解释；

**💡 创新点**

通过引入多排序签名和基于有序整数域的最小化原则，首次实现了在同一逻辑基础上表述并比较三类主流混合ASP系统（如clingo、ASPRouter、dlvhex）的语义差异；

**🔧 技术方法**

多排序Bound‑Founded Here‑and‑There逻辑、约束解释映射、最小化顺序比较、基于等价性与强等价性的理论证明；

**📊 数据集**

论文未使用公开数据集，主要以理论示例和形式化证明为主；

**📈 对比分析**

未给出实验对比或性能评估，仅在理论上说明不同系统在相同语义框架下的等价性与优缺点；

**⚠️ 局限性**

局限性在于缺乏经验验证和对非差分约束的扩展、对大规模实例的可扩展性分析以及对实际混合ASP系统的实现细节讨论。

---

## 396. What Bugs Do Prolog Students Write? An Empirical Taxonomy and Data-Driven Mutation Framework

**arXiv ID:** 2607.21193 | [PDF](https://arxiv.org/pdf/2607.21193v1)

**作者:** Ricardo Brancas `[一作]` (INESC-ID / IST, Universidade de Lisboa), Ruben Martins `[通讯]` (Carnegie Mellon University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 7201 篇 Prolog 学生提交进行实证研究，构建了细粒度的学生错误分类，并基于此开发了数据驱动的变异工具，以生成与真实课堂错误分布高度匹配的合成 buggy 程序。

**💡 创新点**

创新点在于：①首次基于真实学生提交的错误频率为变异器的权重分配提供经验依据；②通过 SMT 语义合成器与 AST 变异相结合，在保证语法与语义差异的前提下生成高质量的合成错误；③对比实验显示合成错误分布与真实错误分布差距不足 2%。

**🔧 技术方法**

核心技术包括：自定义 Prolog AST 解析器；四阶段变异管线（枚举、采样、注入、验证）；SMT（Trinity）驱动的代码片段合成；Python 与 SWI-Prolog 的双栈实现。

**📊 数据集**

使用公开的 7201 篇学生提交数据集（来源于 figshare），并从中手工抽取 200 篇修复提交进行错误分类，构成经验频率分布。

**📈 对比分析**

对比方法是将基于经验权重生成的 16,000 个合成程序的错误类别分布与真实学生错误分布进行百分比对齐。实验结果显示大多数类别差异在 2% 以内；仅 cut 相关错误和 predicate rename 在合成集中过度/不足出现。

**⚠️ 局限性**

主要局限：①SMT 合成器生成的代码缺乏编程习惯与语义合理性，导致一些插入式变异显得人工；②验证过程仅基于测试覆盖，无法过滤所有语义不合理的变异；③对 cut 相关错误的低覆盖率反映出验证策略需改进；④未覆盖多机构、多课程的多样性，未来需扩大样本与语言范围。

---

## 397. Multi-Task Learning for Heterogeneous Prediction from Video Game State with Transfer Learning

**arXiv ID:** 2607.21290 | [PDF](https://arxiv.org/pdf/2607.21290v1)

**作者:** Jonas Peché `[一作]` (Johannes Kepler University Linz), Günter Wallner `[通讯]` (Johannes Kepler University Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在 World of Tanks 真实游戏状态数据上，探索了多任务学习（MTL）在 10 个二分类与回归预测任务上的效果，并系统比较了不同任务权重/梯度平衡策略与源任务预训练/微调方法。

**💡 创新点**

创新点在于：① 在同一多模态架构下对 MTL 与单任务学习进行全面基准；② 对等权、随机权、FAMO 与 PCGrad 四种权重/梯度方法在异构任务上的实证对比；③ 通过跨地图与低数据场景验证 MTL 的迁移能力，展示了跨地图预训练对新地图性能的显著提升。

**🔧 技术方法**

主要技术包括：EfficientNet‑B0 视觉编码器 + 结构化特征（全局、单元）+ 多头自注意力交互；任务头分为分类头与回归头；损失加权/梯度平衡采用等权、RLW、FAMO、PCGrad；源任务预训练 + 目标任务微调；以及对比单任务模型与 MTL 模型在参数、GFLOPs 与训练时长上的节省。

**📊 数据集**

使用的数据集为大规模 World of Tanks 战斗记录，约 219 万局，覆盖 29 张地图、每局 30 辆车，提取 10 个结构化预测目标（Win、Alive、Hurt、Deal Damage、Move、Final HP、Move Dist.、Position、Damage Delta、Future HP）。

**📈 对比分析**

比较方法：对单任务学习与 MTL（等权、RLW、FAMO、PCGrad）在 AUC 与 RMSE 上做平均；低数据时做从零到 2.18M 样本的预训练/微调；跨地图预训练使用所有地图训练，后在 Redshire 地图上微调。结果显示：PCGrad 在大多数任务上略优于等权，低数据下预训练提升 2–8%（AUC）/8–15%（RMSE）；跨地图预训练在新地图上提升 10–20%；MTL 在参数量与训练时间上比 10 个单任务模型分别节省 10 倍与 80% 的资源。

**⚠️ 局限性**

局限性：未给出置信区间或显著性检验；早停机制仅监控平均损失，易导致对小数据集的过拟合或欠拟合；模型规模大，推理成本仍高；仅评估了 PCGrad、RLW、FAMO，未尝试其他梯度平衡方法；跨地图实验使用的数据量不完全匹配，难以量化纯粹的地图迁移效果；缺少在其他游戏（如 ESTA）上的验证。

---

## 398. HGeo-TopoMap: Boosting Topological Mapping with Hierarchical Geometric Priors

**arXiv ID:** 2607.21281 | [PDF](https://arxiv.org/pdf/2607.21281v1)

**作者:** Siyu Li `[一作]` (Zhejiang University of Science and Technology), Kailun Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 HGeo-TopoMap 方法，在 BEV 视图下利用几何自适应学习和几何一致性学习双层先验，提升中心线及道路拓扑图的构建精度。

**💡 创新点**

创新点在于将显式的道路结构图（通过 Mask2Former + IPM 生成）与隐式的几何关系（中心线的直/曲/平行/垂直等属性）结合起来，形成层次化先验；通过距离遮罩注意力和几何一致性对比学习实现对中心线检测的精准引导。

**🔧 技术方法**

采用 Mask2Former 语义分割 + IPM 视图变换 + DETR+BEVFormer 检测框架 + 位置编码 + 先验遮罩注意力 + 目标点回归 + InfoNCE 对比学习 + 传统目标检测与拓扑损失。

**📊 数据集**

使用 OpenLane-V2 数据集（中心线、交通标志、拓扑、车道注解）以及 LaneSegNet 的车道/行人辅助标注。

**📈 对比分析**

与 TopoLogic、LaneSegNet 基线进行对比：在中心线基准上 OLS 提升 2.0%（中心线 AP +1.6%），在车道基准上 mAP 提升 5.7%（拓扑准确度 +1.8%）。在缺失视角的鲁棒性实验中，HGeo-TopoMap 的性能明显优于基线。

**⚠️ 局限性**

局限性：单帧视角生成的道路结构图受遮挡和噪声影响，无法完全补偿复杂场景下的信息缺失；缺乏时序信息与优化策略，导致在极端环境下仍存在一定误差。

---

## 399. A Comparative Evaluation of Embeddings and LLMs in a Greek Book Publisher Setting - The CUP Dataset

**arXiv ID:** 2607.21274 | [PDF](https://arxiv.org/pdf/2607.21274v1)

**作者:** Katerina Papantoniou `[一作]` (ICS-FORTH), Dimitris Plexousakis `[通讯]` (ICS-FORTH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含868条记录、104条专家标注查询的希腊图书检索基准，并评估了稀疏检索、密集检索、混合检索和LLM辅助检索方法。

**💡 创新点**

首次提出希腊语检索基准，展示多语言嵌入模型优于希腊专属模型，证明加权混合检索可显著提升整体效果。

**🔧 技术方法**

使用BM25、Sentence‑Transformers密集模型（nemotron、nomic、bge‑m3等）、加权混合融合、RRF以及LLM生成TOC摘要和检索后过滤/重排序。

**📊 数据集**

数据集为希腊学术与通用书籍目录（868本），包含标题、作者、类别、描述、TOC和LLM摘要，配备104条多样化查询与分级相关性标签。

**📈 对比分析**

通过nDCG@9、Hits@k、MRR等指标比较，最佳混合方法在nDCG@9达到0.673，BM25在命名实体查询上最强，密集模型在推理与跨语言查询上表现突出，LLM摘要提升TOC检索，LLM过滤在早期检索上提高精度但成本高。

**⚠️ 局限性**

缺点是没有单一方法在所有查询类型上统治；LLM推理成本高，且基准仅覆盖单一书店目录，扩展性有限。

---

## 400. BasketEvent: Understanding Who Did What and When in Basketball Videos

**arXiv ID:** 2607.21267 | [PDF](https://arxiv.org/pdf/2607.21267v1)

**作者:** Yu Zhang `[一作]`, Weidi Xie `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍了在BMVC会议论文模板中如何通过 LaTeX 命令（如 \addauthor、\addinstitution）输入作者姓名与机构信息的多种实现方式。

**💡 创新点**

创新点在于给出了一套简洁、可直接复制的命令示例，方便作者快速填充姓名、邮箱/网址及机构编号，从而提升排版效率。

**🔧 技术方法**

主要使用 LaTeX 文档类与自定义命令进行实现，并配合了示例代码。

**📊 数据集**

未使用任何外部数据集，本文仅为格式示例。

**📈 对比分析**

未进行实验或性能评估；仅通过示例展示不同命令组合的效果。

**⚠️ 局限性**

局限性在于缺乏真实论文中多作者多机构场景下的完整验证，示例过于简化，无法覆盖所有可能的排版需求。

---

## 401. Progressive Cramming: Reliable Token Compression and What It Reveals

**arXiv ID:** 2607.21231 | [PDF](https://arxiv.org/pdf/2607.21231v1)

**作者:** Dmitrii Tarasov `[一作]` (FusionBrain Lab), Andrey Kuznetsov `[通讯]` (FusionBrain Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出进化式压缩（Progressive Cramming）方法，逐步增量地优化单个嵌入以实现完整的自回归重构，直至在给定优化预算下无法继续扩展；

**💡 创新点**

创新点在于：①将目标长度逐步扩展以保证100%重构，消除固定预算与99%阈值的局限；②发现压缩轨迹低维并可通过低维投影稳定优化；③通过注意力剔除实验揭示压缩嵌入对早层的干预是导致下游能力下降的根本原因；④证明压缩容量受解码器深度与宽度共同决定，可通过网络架构调节而非单纯扩大参数。

**🔧 技术方法**

使用的技术包括：梯度优化对单个嵌入进行学习，低维投影（rank‑k 子空间约束），信息增益度量，PCA 轨迹分析，注意力“knockout”因果干预，以及对截断模型进行短期微调以评估容量。

**📊 数据集**

主要使用的公开数据集是 PG19（小说文本）进行压缩实验，另外在 HellaSwag、ARC‑Easy 进行下游评估，亦在 Fanfics 进行复现。

**📈 对比分析**

与传统“full cramming”相比，进化式压缩在保持100%重构的前提下实现了可测量的压缩边界，尽管在令牌长度上略逊于最优解，但其生成准确率稳定为100%；下游多选任务上，压缩嵌入导致准确率下降但仍高于随机；在生成任务（如 MMLU）上则几乎完全失效。

**⚠️ 局限性**

局限性包括：①需要对每个样本逐步迭代优化，速度慢且难以批量；②仅针对自回归重构，未能保证压缩嵌入的语义可迁移性；③低维投影虽提升稳定性但并未显著提高可压缩长度；④对压缩边界的预测仍不精确，信息增益不足以精确定位容量。

---

## 402. ICAE-Bench: Evaluating Coding Agents as Interactive Project Builders

**arXiv ID:** 2607.21217 | [PDF](https://arxiv.org/pdf/2607.21217v1)

**作者:** Zhongyuan Peng `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了 ICAE‑Bench，一个面向模糊产品需求下的交互式项目构建评测基准。

**💡 创新点**

创新点：①将真实开源仓库的可执行行为作为语义基准，构造模糊 PRD 与可恢复的 User Agent；②提供三层模糊级别与可恢复的公共案例；③使用语言无关的黑盒测试与多维诊断（功能、语义/API/设计相似度、结构比例、交互覆盖）实现更全面的评估；④构建 480‑任务、12‑语言的数据集并公开评测代码。

**🔧 技术方法**

技术：基于 Docker 的 ultimate image；Claude Code / OpenHands 框架下的 LLM（GPT‑5.5、Claude‑Opus‑4.8、Claude‑Sonnet‑4.6、GLM‑5.1、Gemini‑3.1‑Pro、MiniMax‑M2.5）进行交互式编程；User Agent 采用 DeepSeek/ Gemini 路由器返回预先记录的澄清答案；原始测试转化为 JSON 黑盒案例；评估指标包括功能通过率、语义/API/设计相似度、结构比率、交互覆盖率等。

**📊 数据集**

数据集：480 个任务，覆盖 12 种编程语言（C#, C++, Dart, Go, Java, JavaScript, Kotlin, PHP, Python, Ruby, Rust, TypeScript）。每个任务包含模糊 PRD、ultimate image、Public/Hidden 测试、User Agent 数据；还提供 50 个 IC​AE‑Bench‑Lite 子集用于快速 ablation。

**📈 对比分析**

比较方法：在完整基准上对六种模型在 Claude Code 框架下进行实验，记录功能通过率、Agentic 评估（语义、API、设计）、结构评估（文件/LOC 比率）、交互质量（覆盖率、fallback、预算）。最佳总体通过率约 38%（Claude‑Opus‑4.8），交互与更高 PRD 曝露可提升约 10%，但仍存在显著性能缺口。相较传统 0‑to‑1 生成基准，ICAE‑Bench 提供更真实的交互式评测。

**⚠️ 局限性**

限制：①依赖 LLM 的 PRD 与 User Agent 生成，可能带来偏差与 hallucination；②模糊级别与 User Agent 设计不一定覆盖所有真实用户交互场景；③多维指标虽全面，但对设计质量的主观评估仍有限；④仅包含已测试的语言与仓库，缺少更广泛领域或跨模态任务；⑤实际部署中对运行时环境、依赖管理等细节的控制仍需进一步验证。

---

## 403. Walk-In Multi-Stage Patient Flow Scheduling: An ASP Model with DES-Based Evaluation

**arXiv ID:** 2607.21198 | [PDF](https://arxiv.org/pdf/2607.21198v1)

**作者:** Ngoc-Mai Pham `[一作]` (VNU University of Engineering and Technology), Van-Giang Trinh `[通讯]` (Ho Chi Minh City University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种用于门诊医院环境的实时多阶段病人流排程框架，能够在病人到达时动态决定其检查顺序与房间分配；

**💡 创新点**

创新点在于将检查序列、房间分配与医院内的移动成本统一建模为一个反应式组合优化问题，并采用ASP进行声明式求解；

**🔧 技术方法**

使用技术包括Answer Set Programming（clingo）实现约束与优化，结合离散事件模拟（DES）评估随机服务时长下的运营效果；

**📊 数据集**

数据集为基于越南医院运营信息生成的合成数据，包含不同病人负荷（100、250、400或200、450、650）和容量设置；

**📈 对比分析**

与两种基线（实时GREEDY DES和含未来估计的DES‑F）对比，ASP方法在大多数指标上均优于基线：平均等待时间、停留时间降低，零等待患者比例提升，且未完成患者数减少；

**⚠️ 局限性**

局限包括：使用合成数据而非真实日志；假设FCFS服务且未考虑急诊、优先级或中断；以及仅在新病人到达时触发优化，未对已完成检查后的再优化进行研究。

---

## 404. Declarative Problem Solving in UAM Strategic Deconfliction

**arXiv ID:** 2607.21197 | [PDF](https://arxiv.org/pdf/2607.21197v1)

**作者:** Gioacchino Sterlicchio `[一作]` (Polytechnic University of Bari), Francesca Alessandra Lisi `[通讯]` (University of Bari Aldo Moro)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于Answer Set Programming（ASP）的城市空中交通（UAM）战略冲突消除（SD）框架，并与Constraint Programming（CP）方法进行了对比实验。

**💡 创新点**

首创了ASP在UAM SD中的完整模型，提供可读性强、易改动的逻辑建模；同时在不同网络拓扑和负载下系统性比较ASP与CP在执行时间、内存占用以及解的质量方面的差异。

**🔧 技术方法**

使用ASP（clingo等求解器）和CP（MiniZinc/Gecode）两种约束求解技术；构建了航路、速度、时间窗等约束规则，并通过延迟最小化的目标函数进行优化。

**📊 数据集**

采用三种合成UAM网络拓扑（城市内/城际/机场穿梭）以及正方形网格生成的网络，飞行器数量从15到400甚至5000，发射窗口从5到50分钟，所有数据均为人工生成的合成实例。

**📈 对比分析**

通过规模化实验（增大飞行器数、窗口大小、网络规模）测量执行时间和内存，结果显示：ASP在小规模、短窗口时具有更快的求解速度和更低的内存占用；但随着规模增大，ASP易出现内存溢出或超时；CP在大规模下耗时更长、内存更稳定，但也会因时间限制而失败。

**⚠️ 局限性**

局限性：受限于单线程求解器和8GB内存，ASP模型在高密度或大窗口下易爆炸；实验仅基于合成数据，缺乏真实世界验证；未评估多种ASP求解器或更高性能硬件对结果的影响。

---

## 405. Animation, Verification and Visualisation of Prolog Transition Systems with ProB

**arXiv ID:** 2607.21192 | [PDF](https://arxiv.org/pdf/2607.21192v1)

**作者:** Jan Gruteser `[一作]` (Heinrich Heine University Düsseldorf), Fabian Vu `[通讯]` (Heinrich Heine University Düsseldorf)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现并扩展了ProB工具的XTL模式，新增了符号转移、可执行转移、改进的状态可视化、JSON轨迹重放以及Monte Carlo模拟等功能，并将其应用于Connect Four游戏策略评估。

**💡 创新点**

核心创新包括：①引入符号转移和可执行转移机制，使得在状态探索时可延迟执行并支持用户输入；②改进状态可视化，支持SVG/HTML导出；③新增JSON轨迹重放，确保精确重现动画序列；④将Monte Carlo模拟与概率模型检查结合，提供统计验证。

**🔧 技术方法**

采用Prolog（SICStus）、ProB的XTL模式、PCTL模型检查、Timed-Probabilistic Simulation、Minimax 与 MCTS 等技术实现。

**📊 数据集**

以Connect Four游戏为案例，生成随机、Minimax（深度2）和MCTS策略的10,000场对局数据，并用这些数据进行模拟与统计分析。

**📈 对比分析**

通过比较胜率、平均回合数和执行时间评估三种策略；结果显示随机玩家几乎不敌Minimax和MCTS，MCTS几乎完全击败Minimax；MCTS在胜率上占优但运行时间显著更长；Minimax在深度不足时表现欠佳。

**⚠️ 局限性**

主要局限在于Minimax深度设置过浅导致性能不佳；MCTS模拟耗时较长；对更大状态空间的支持尚未充分；符号转移和可执行转移的实现仍依赖手工定义，易出现错误。

---

## 406. Reimagining the Augmented Reality Accessibility Ecosystem for Deaf Students: Service Provider Perspectives in Experiential Learning

**arXiv ID:** 2607.21289 | [PDF](https://arxiv.org/pdf/2607.21289v1)

**作者:** Roshan Mathew `[一作]` (Rochester Institute of Technology), Roshan Peiris `[通讯]` (Rochester Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估了 ARRAE（基于 AR 的实时访问系统）在实验室环境下对教师、美国手语译员和实时字幕员的影响，并通过专家式对比研究（面对面、远程、AR）探讨其对交互与注意力的重塑。

**💡 创新点**

将沟通访问重新定义为分布式意识，并展示 AR 如何通过多摄像头视角和沉浸式 HUD 改变教师、译员和字幕员的工作模式与协调方式。

**🔧 技术方法**

使用 AR 头戴显示器（Vuzix Blade）、可编程 PTZ 摄像机、WebRTC 直播、TypeWell 文本输入服务器以及基于浏览器的仪表盘等技术构建系统。

**📊 数据集**

该研究未采用公开数据集，而是基于六位聋学生在实验室完成任务时产生的实时视频与文本交互数据进行实验。

**📈 对比分析**

通过对三种条件进行定性访谈、问卷与主题分析，比较发现 AR 在降低学生分心、提升视觉意识方面表现优异，但在教师监督、系统稳定性与交互反馈方面存在局限。

**⚠️ 局限性**

研究样本仅限于单一教师、译员和字幕员，原型系统不成熟，缺乏统计显著性，且未充分解决系统可靠性、协调成本与多用户同步等挑战。

---

## 407. news-crawler-LM: A Small Long-Context Model For High-Quality News Crawling

**arXiv ID:** 2607.21284 | [PDF](https://arxiv.org/pdf/2607.21284v1)

**作者:** Pascal Stolzenburg `[一作]` (Humboldt University of Berlin), Alan Akbik `[通讯]` (Humboldt University of Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练一个小型长上下文语言模型，将新闻网页HTML转换为纯文本或结构化JSON，利用Fundus规则库人工验证的高质量数据进行监督。

**💡 创新点**

结合低噪声人工标注数据、ReaderLM-v2骨干与SimCTG对比学习，形成高效且低重复率的HTML‑to‑文本/JSON转换模型，在多出版社零样本场景下超越传统规则库。

**🔧 技术方法**

使用ReaderLM‑v2 Transformer、LoRA参数高效微调、SimCTG对比学习以及长上下文推理技术。

**📊 数据集**

约10万条来自Fundus库的HTML‑文本/JSON对，涵盖93家出版商、25种语言的多语言数据集。

**📈 对比分析**

与Trafilatura、news‑please、Boilerpipe等规则库及ReaderLM‑v2、Qwen2.5‑32B、Qwen2.5‑1.5B等LLM进行零样本与微调对比；模型在HTML‑Plaintext任务BLEU+4.8、METEOR+6.1、ROUGE‑L提升显著，JSON任务F1提升+0.022/0.041/0.038，且低重复率和高ROUGE‑L表现明显优于基线。

**⚠️ 局限性**

依赖Fundus规则可能引入偏差；仍出现JSON结构错误或字段缺失；主要针对英语/德语，低资源语言泛化有限；长文档推理耗时；模型可能产生hallucination，需进一步约束或校验。

---

## 408. Adaptive Depth Sparse Framework: Similarity-Driven Resource Allocation for Pre-Trained LLMs

**arXiv ID:** 2607.21291 | [PDF](https://arxiv.org/pdf/2607.21291v1)

**作者:** Yidu Wu `[一作]` (Southern University of Science and Technology), Xiaoying Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将现成的预训练大模型转为深度稀疏版本，显著降低推理 FLOPs

**💡 创新点**

通过隐藏状态相似度驱动层级计算分配、轻量化路由器以及特征对齐损失，实现无结构修改的自适应深度稀疏

**🔧 技术方法**

利用层内输入输出余弦相似度计算保留比例，MLP 路由器挑选重要 token，跨层特征对齐损失保证稀疏模型与稠密模型行为一致

**📊 数据集**

GPT‑NeoX‑130M、Qwen2.5‑0.5B/1.5B；任务包括 WikiText103 语言模型和 ARC、HellaSwag、PIQA 等六个常识推理基准

**📈 对比分析**

与 MoD、D‑LLM、DLO 等基线对比；在相同稀疏度下，AdaDSF 在 PPL 和常识推理准确率上通常比基线更好（PPL 下降至 18.9，推理 FLOPs 仅 0.787×稠密），并且在不同保留比例下性能波动更小

**⚠️ 局限性**

目前仅验证至 1.5B 参数规模，未对更大模型（7B+）进行评估；依赖预训练 checkpoint，且对极端稀疏比例的鲁棒性仍待进一步研究

---

## 409. The Dark Room in the Reward Channel: Dense Prediction Rewards Collapse GRPO-Trained LLM Agents -- and What Actually Works

**arXiv ID:** 2607.21273 | [PDF](https://arxiv.org/pdf/2607.21273v1)

**作者:** Yu Wang `[一作]` `[通讯]`, Yu Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究在稀疏奖励、长周期 LLM 代理中使用基于预测的潜在差分奖励会导致“黑暗房”路径吸收现象，导致预测准确率趋于 1 而任务成功率降为 0；并通过实验验证了该现象在 Qwen3-1.7B/4B/8B、ALFWorld、HiddenRule-Gym 上的出现；进一步定位其根源为组归一化（GRPO）中的标准差归一化放大了仅含形状项的优势；提出方差‑剖面准则，阐释何种稠密信号在所有失败组占主导时会被放大；验证了“奖励通道”相较于“辅助损失通道”对性能的负面/正面影响，并提供了早期预警信号。

**💡 创新点**

1）首次将组归一化中的标准差归一化与稀疏奖励的组合导致的不可逆性能崩溃现象系统化表述；2）提出方差‑剖面准则，给出对稠密信号安全性的理论预测；3）通过对信号消耗通道的对照实验揭示奖励通道在 GRPO 下并非安全途径；4）提供早期预警指标，防止“黑暗房”吸收。

**🔧 技术方法**

基于 Qwen3 LLM 的多轮推理框架；使用 GRPO（Group‑Relative Policy Optimization）与组归一化；引入基于潜在差分的预测奖励（λ=0.1）；设计辅助损失通道（交叉熵 teacher‑forced CE）；对比均值归一化、标准差归一化、解耦优势、梯度上限等技术。

**📊 数据集**

主要使用 ALFWorld（长周期稀疏奖励环境）和 HiddenRule‑Gym（可计算特征覆盖率的 POMDP）进行实验；在 Qwen3-1.7B/4B/8B 规模下评估。

**📈 对比分析**

与传统 GRPO（仅均值归一化）以及无预测奖励基线进行对比。预测奖励通道在所有规模上均导致 0% 成功率；去除标准差归一化后成功率恢复到 51.6%（≥ 基线 49.5%）；辅助损失通道在保持原奖励通道不变的前提下，平均提升约 20 分；熵与预测饱和的联合预警信号能够提前 15‑30 步预示崩溃。

**⚠️ 局限性**

实验仅基于单一 seed，结果受方差影响；缺乏多 seed 复现与大样本评估；组大小（n=4）与标准差归一化问题相关，未进行多组大小 ablation；8B 模型表现与 4B 的异常差异尚未解释；仅在 ALFWorld 与 HiddenRule‑Gym 上验证，未涉及更多环境；未检验基于 critic 的 PPO 等非组归一化方法。

---

## 410. Detectors Learn the Wrong Thing: Shortcut-Resistant Adversarial Training Against Physically Realizable Attacks

**arXiv ID:** 2607.21243 | [PDF](https://arxiv.org/pdf/2607.21243v1)

**作者:** Yuanhao Huang `[一作]` (Beihang University), Haiyang Yu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种面向物理可实现人像攻击的对抗性训练框架 InsCAT，旨在抑制检测器对攻击纹理的“纹理捷径”依赖，从而提升对未知纹理攻击的鲁棒性。

**💡 创新点**

创新点在于引入实例级对比对齐（SICA）来对齐攻击人像与干净人像的特征，并通过渲染共享纹理的在线对抗优化（ROPO）和训练守卫（Guard）实现对抗信号的持续更新与训练平衡，显著降低纹理触发误检。

**🔧 技术方法**

核心技术包括基于渲染的在线纹理优化、实例级对比学习（InfoNCE/间隙损失）、对抗训练与多任务损失协调、以及在线负样本生成与缓冲。

**📊 数据集**

使用的数据集涵盖 COCO（训练）、nuScenes-mini（渲染背景）、INRIAPerson（数字拼贴）以及真实拍摄的物理攻击服装视频，评估在不同视觉域和攻击纹理上的跨域泛化。

**📈 对比分析**

与多种基线（LGS、PAD、AT‑Mix 等）对比，InsCAT 在 nuScenes 渲染域的平均攻击 AP 达 82.3%，超出最佳基线 11.1 分；在 INRIAPerson 数字域平均 AP 78.3%，与 PAD 差距 5.3 分；同时纹理 FPR 从 46.9% 降至 7.3%，且推理时延仅 0.29 ms，表明高鲁棒性与实时性能兼得。

**⚠️ 局限性**

局限性包括仅针对 RGB 人像检测和服装纹理，未探究多目标或更复杂环境；对抗纹理的生成依赖渲染与手工姿态库，可能对真实多样性不足；模型对极端远距离或遮挡仍表现下降。

---

## 411. T-STAR: A Large-Scale Benchmark for Spatio-Temporal Panoptic Scene Graph Generation in Satellite Video

**arXiv ID:** 2607.21228 | [PDF](https://arxiv.org/pdf/2607.21228v1)

**作者:** Linlin Wang `[一作]` (Wuhan University), Yansheng Li `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了卫星视频的时空面向场景图生成（TPSG）任务，并构建了首个大规模T-STAR数据集和基于STCL的时空协同学习框架。

**💡 创新点**

创新点在于首次针对卫星视频创建细粒度面向掩码注释与时空关系标签的数据集，以及提出结合记忆引导匹配、空间上下文增强和多尺度时序学习的三大模块。

**🔧 技术方法**

技术方法包括视频面向分割（IPS+T、VPS）、记忆引导匹配（MGM）、实例对空间图建模（SCE）以及多尺度时序卷积网络（MTL）等。

**📊 数据集**

使用的主要数据集是T-STAR，包含150段卫星视频、1.18M实例掩码和3.83M时空三元组，涵盖39类对象和70类关系。

**📈 对比分析**

在PredCls和SGDet任务中，STCL在T-STAR测试集上相较六种基线方法实现了最高的TR/mTR/FTR指标，表现优异，但仍受长尾分布与小目标识别挑战影响。

**⚠️ 局限性**

局限性包括对象与关系的长尾分布、弱纹理小目标导致的身份漂移、对更广泛场景的覆盖不足以及模型在高维时空依赖下的计算开销。

---

## 412. Stokes-Informed Diffusion for Robust Linear Polarization Estimation

**arXiv ID:** 2607.21239 | [PDF](https://arxiv.org/pdf/2607.21239v1)

**作者:** Yidong Luo `[一作]` (Zhejiang University), Xin Yuan `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为GenPolar的框架，用于从单个RGB图像中估计线性偏振，特别是通过预测通道级的线性Stokes分量(S_1, S_2)来实现。

**💡 创新点**

创新点在于采用了基于Mueller形式的Stokes-informed扩散框架，并引入了可观测性意识的损失函数来稳定角度估计，同时采用了两阶段训练策略以提高推理效率和准确性。

**🔧 技术方法**

使用了基于物理监督的多步条件扩散模型和一阶段蒸馏生成器，结合了LoRA适应技术来减少领域特定的自编码偏差。

**📊 数据集**

使用了多个偏振数据集，包括旋转线性偏振器组、焦平面分割极化计组和混合数据集，共计约598个样本。

**📈 对比分析**

与其他方法（如PolarAnything、Restormer和MAE）进行比较，GenPolar在DoLP的保真度和AoP的稳定性上表现出色，尤其在DoFP和RSP组中，AoP的稳定性显著提高。

**⚠️ 局限性**

限制在于GenPolar专注于线性偏振，未能恢复圆偏振分量S_3；在极化照明、天空光、透明或多重散射路径等情况下，模型的有效性可能会降低。

---

## 413. Controlled Periodic Synchronization for Efficient Data-Parallel Training

**arXiv ID:** 2607.21224 | [PDF](https://arxiv.org/pdf/2607.21224v1)

**作者:** Imane Ettifouri `[一作]` (Mohammed V University in Rabat), Claude Tadonki `[通讯]` (MINES Paris--PSL University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了分布式数据并行训练中同步频率的系统影响，提出了Controlled Periodic Data Parallelism（CPDP）方案；

**💡 创新点**

在CPDP中将梯度AllReduce与SlowMo参数平均相结合，形成双阶段一致化；

**🔧 技术方法**

使用PyTorch DistributedDataParallel（DDP）扩展实现，融合梯度AllReduce、参数AllReduce和SlowMo动量；

**📊 数据集**

在Grid'5000平台上使用ResNet‑50/CIFAR‑100、TinyImageNet、ViT‑S/CIFAR‑100等视觉数据集；

**📈 对比分析**

与标准DDP和LocalSGD对比，CPDP在跨站WAN（Nancy↔Sophia）下K=4时可实现+2.44个百分点的峰值准确率，同时平均训练时间比DDP低13.8%，在大部分场景下仍保持或略优于DDP；

**⚠️ 局限性**

实验仅覆盖4–16 GPU、单一跨站链路、未同步优化器状态，缺乏收敛性理论支持，并未探讨梯度压缩或更大规模/更高延迟环境。

---

## 414. Explainable Belief Harmonization under Dynamic Epistemic Partitions

**arXiv ID:** 2607.21210 | [PDF](https://arxiv.org/pdf/2607.21210v1)

**作者:** Adam Kostka `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在多智能体信念融合过程中，提出了一种可在运行时动态调整各代理感知能力（即知识分区）的框架，并能在结构变更后自动检测、修复不一致并生成解释；

**💡 创新点**

创新点在于将可变的epistemic partition与连续概率信念分离，利用答案集规划（ASP）的可扩展性与Python的数值计算实现分区的事实编辑和完整性约束，以及通过解释原子自动定位导致不一致的区别；

**🔧 技术方法**

核心技术为混合架构：ASP（使用外部事实编辑和多射击求解）+ Python（数值计算、投影修复、差异事实生成），并利用完整性约束、解释规则和投影投射实现一致性保证；

**📊 数据集**

评估数据集包括：随机生成的100次拓扑变更（20×8、12×6等规模）以及基于传感器融合的手工实例（三传感器六表面、八路况等）；

**📈 对比分析**

通过与纯Python基线和完整性约束对比，发现ASP层能在多射击求解下将重解时间降至10~15 ms（最多180 ms），误差检测率100%，解释覆盖率100%；整体性能随着世界数和代理数增长呈O(|A||W|^2)的基线，但在可交互规模内表现良好；

**⚠️ 局限性**

局限性主要在于：基线基数导致ASP基语义的grounding上限约为50个世界；解释仅诊断性，未给出修复建议；实验仅覆盖单一传感器融合场景，缺乏跨领域验证；

---

## 415. Explainability Framework for Policy-Aware Autonomous Agents

**arXiv ID:** 2607.21209 | [PDF](https://arxiv.org/pdf/2607.21209v1)

**作者:** Heather Merhout `[一作]` (Miami University), Daniela Inclezan `[通讯]` (Miami University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一个基于对比推理的解释框架，用以生成对策略感知智能体（使用Answer Set Programming和AOPl描述的政策约束）决策的自然语言解释。

**💡 创新点**

创新点在于将政策违约的惩罚信息直接转化为对比案例，从而自动生成符合格赖斯四项准则的对比解释；并通过Python与ASP结合的实现方案，使解释过程可交互、可视化。

**🔧 技术方法**

技术主要包括Answer Set Programming (ASP)、AOPl政策语言、Python clingo API、Tkinter GUI、自然语言模板填充，以及对ASP模型的惩罚/弱约束分析。

**📊 数据集**

数据集为医院护士排班领域的典型约束与偏好，使用基于Nabeshima等人的排班模型构造的规则与约束集合；没有使用公开大规模数据集。

**📈 对比分析**

通过对12名参与者的问卷调查评估解释可理解性、信息量与满意度。结果显示约75%参与者认为解释有意义，评分平均在4-5之间；相比传统非对比解释，本文方法在信息量与对比性方面获得更高接受度。

**⚠️ 局限性**

局限包括：对话深度受限于单一查询，解释范围受限于惩罚信息；当模型搜索耗时过长时只能返回部分最优解，可能导致解释不完整；对更大规模或多约束交互的可扩展性尚待验证。

---

## 416. Hybrid MKNF with Classical Negation in the Rule Component

**arXiv ID:** 2607.21202 | [PDF](https://arxiv.org/pdf/2607.21202v1)

**作者:** Arun Raveendran Nair Sheela `[一作]` (Université Clermont Auvergne), Florence De Grancey `[通讯]` (Thales)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了支持规则组件中经典否定的Hybrid MKNF知识库扩展，并给出了其三值well‑found semantics及对应的计算方法。

**💡 创新点**

创新点在于允许在规则中使用经典否定，推导了相应的三值语义与稳定分区，并提出了三阶段（well‑found operator、单元传播、猜测‑检查）来完整地计算well‑found partition。

**🔧 技术方法**

核心技术包括MKNF逻辑与三值结构的定义、稳定分区的构造、最小固定点与单元传播操作，以及在DL安全框架下的DL推理。

**📊 数据集**

论文中未使用任何具体实验数据集，主要以理论示例和形式化证明展示方法。

**📈 对比分析**

与只支持一致性知识库的Hybrid MKNF方法相比，本文方法在Phase 1/2可在多项式时间内得到well‑found partition；若需Phase 3，则整体复杂度上升到指数级（EXPTIME）。

**⚠️ 局限性**

局限性包括：未给出未定义集的具体实现，Phase 3的猜测‑检查步骤在最坏情况下指数；方法仅在DL安全性前提下保证可判定；实际推理器实现仍待进一步工作。

---

## 417. Chess\_db: A framework for working with large chess game datasets

**arXiv ID:** 2607.21195 | [PDF](https://arxiv.org/pdf/2607.21195v1)

**作者:** Nicos Angelopoulos `[一作]` (University College & Imperial College), Jan Wielemaker `[通讯]` (SWI-Prolog solutions)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 Prolog 的工具集，用于解析 PGN、管理棋盘字典、将大量国际象棋对局存入 SQLite 或 RocksDB，并提供位置表查询。

**💡 创新点**

创新点在于：① 用字典表示棋盘并实现完整合法走法检测；② 采用十六进制哈希编码唯一位置；③ 在逻辑编程框架中实现增量 PGN 解析和键值数据库存储，支持数千万局的快速检索。

**🔧 技术方法**

使用技术包括 SWI-Prolog（proSQLite、rocksdb、b‑tree 等）、PGN 语法解析、棋盘字典与位置哈希、RocksDB 键值数据库、SQLite、实验测评脚本。

**📊 数据集**

使用的数据集为 Lichess 官方月度数据库（2023‑01 至 2025‑11），共约 10 M 局（约 311 k/每月），仅保留标准棋局、Elo ≥ 2500/2300 的对局。

**📈 对比分析**

性能比较方法：单次插入实验（10 k 局为单位）在 RocksDB 上 3 M 局以内每 10 k 局 ≤ 3 min；6 M 局约 3 min；9 M 局约 5 min；10 M 局 8–10 min；SQLite 在数十万局后显著下降；Berkeley DB 进一步恶化；重启插入导致初期慢速。总体可接受但随规模增大显著。

**⚠️ 局限性**

局限性：① 位置表对深度 25 以内有效，深度后信息稀缺；② 位置表仅适用于开局阶段；③ 数据库大小受硬件影响，重启插入成本高；④ 未实现图形界面与单步解析优化，尚需进一步改进。

---

## 418. An LLM-Driven Workflow for Automated Process Control Strategy Generation and Tuning from Dynamic Process Models

**arXiv ID:** 2607.21292 | [PDF](https://arxiv.org/pdf/2607.21292v1)

**作者:** Ari Luna Rueda `[一作]` (Imperial College London), Mehmet Mercangöz `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于大型语言模型的结构化工作流自动生成、验证并调优闭环控制器。

**💡 创新点**

将控制设计拆解为可验证的代码生成子任务，并在每步使用自动修复机制实现无人工干预。

**🔧 技术方法**

使用OpenAI GPT‑5.4进行代码生成、Bayesian Optimization进行参数调优，以及Python实现的仿真与验证。

**📊 数据集**

采用非线性双输入双输出的气体预热器基准模型进行演示。

**📈 对比分析**

与初始LLM生成的PI/前馈控制器对比，BO调优后闭环目标下降约26.5%，跟踪误差显著减小。

**⚠️ 局限性**

仅在单一案例上验证，缺乏统计鲁棒性评估，控制器未实现抗风胀，且对更大规模工厂模型的泛化有限。

---

## 419. Large Language Model Assisted Intent-Based Satellite-Integrated Access and Backhaul FWA for Rural Areas

**arXiv ID:** 2607.21272 | [PDF](https://arxiv.org/pdf/2607.21272v1)

**作者:** Anselme Ndikumana `[一作]` (Université du Québec), Mohamed Cheriet `[通讯]` (Université du Québec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于大语言模型的意图感知卫星‑IAB‑FWA方案，能够将用户自然语言意图映射为网络QoS需求，并通过两阶段Benders分解实现节点/链路激活与流量路由的能效最优调度，从而实现乡村地区固定宽带与临时田间网络的动态、高能效协同。

**💡 创新点**

创新点在于：①将LLM用于意图‑词汇‑QoS的三阶映射，实现高层业务意图自动转化为可执行网络配置；②首次将卫星Backhaul与IAB网络耦合，支持临时场景覆盖；③采用两阶段Benders分解，将节点激活与流量路由分离，显著提升大规模乡村网络的求解可扩展性；④在能效评估上实现了对传统全激活方案的显著下降（≈33.3%）。

**🔧 技术方法**

使用技术包括：大型语言模型（all‑MiniLM‑L6‑v2）进行语义映射；整数线性规划与混合整数线性规划；两阶段Benders分解；LEO卫星轨道可见性与链路容量估算；IAB架构与MEC服务器能耗模型；冲突检测与二进制用户入选优化。

**📊 数据集**

数据集：OneWeb卫星轨道数据（CelesTrak）用于可见性与链路计算；Kenya乡村区域地理坐标与居民点、IAB节点布置；意图语料库（从文中引用的意图数据集）与网络词汇表；5G QoS需求集；句子嵌入模型评测数据（用于选择all‑MiniLM‑L6‑v2）。

**📈 对比分析**

比较方法：与基准的FCFS、贪心入选策略及始终激活的IAB对照；评估指标包括能耗、数据率、时延与能效比。实验结果显示：相比基准能耗下降约33.3%，数据率提升、时延下降，能效比最高，验证了方案的有效性。

**⚠️ 局限性**

局限性：①求解复杂度仍为O(n³)，对极大规模网络的实时调度有限；②依赖卫星轨道可见性预测，现实中轨道扰动与链路不稳定性未充分考虑；③仅在仿真环境中验证，缺乏真实部署与实验数据；④意图词汇与QoS映射受预定义词表限制，需扩展到更广泛场景。

---

## 420. Exploring the Design Space of LLM-Based Programming Support in CS Education: A Scoping Review through the Lens of Assistance Governance

**arXiv ID:** 2607.21257 | [PDF](https://arxiv.org/pdf/2607.21257v1)

**作者:** Minsun Kim `[一作]` (Virginia Tech), David H. Smith `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 90 篇 2023–2026 年的同行评审论文进行 scoping review，分析其 LLM 编程支持系统在治理维度（Policy、Enforcement、Authority）的设计与实现。

**💡 创新点**

提出并应用 PEA 三维治理框架，系统化绘制政策、执法与权威的设计空间，揭示系统多样化但权威高度集中化的空白。

**🔧 技术方法**

使用 ChatGPT 辅助的半自动编码流程，结合人工定性合成分析，构建治理代码表。

**📊 数据集**

检索 ACM Digital Library、IEEE Xplore、Scopus 中 90 篇符合条件的论文，构成治理分析的数据集。

**📈 对比分析**

通过多标签编码并计算 Cohen's κ（Policy 0.705，Enforcement 0.679）验证编码一致性；未给出传统模型性能指标，而是聚焦于治理配置的频率与分布。

**⚠️ 局限性**

仅包含已发表的同行评审系统，排除行业工具、预印本与非公开部署；研究聚焦的时间窗口和文献范围导致权威维度稀缺可能是文献本身的局限，而非技术不可行。

---

## 421. Logic Programming Semantics for Causal Processes

**arXiv ID:** 2607.21233 | [PDF](https://arxiv.org/pdf/2607.21233v1)

**作者:** Felix Weitkämper `[一作]` `[通讯]` (German University Of Digital Science), Felix Weitkämper (German University Of Digital Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

探讨正逻辑程序的稳定模型与支持模型与时间序列过程终态的对应关系，表明稳定模型对应从中立初态无干扰演化的终态，支持模型对应任何初始状态可达的终态。

**💡 创新点**

首次从时间因果视角阐释逻辑程序语义，区分稳定模型和支持模型在无扰过程与受扰过程中的角色。

**🔧 技术方法**

采用逻辑程序语义理论、一阶谓词逻辑、支持模型与稳定模型定义、过程兼容性构造及Herbrand结构等技术。

**📊 数据集**

无具体数据集，论文以理论分析和例子（如火灾房屋、Boolean网络）为支撑。

**📈 对比分析**

未进行实验比较，性能讨论仅为理论证明。

**⚠️ 局限性**

局限在于仅处理正逻辑程序，未考虑包含负号的程序、振荡终态以及三值语义等更一般情形。

---

## 422. How Rules Represent Causal Knowledge: Causal Modeling with Probabilistic Logic Programming

**arXiv ID:** 2607.21208 | [PDF](https://arxiv.org/pdf/2607.21208v1)

**作者:** Kilian Rueckschloss `[一作]`, Felix Weitkaemper `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种新的因果语义，将 Pearl 的因果理论引入概率逻辑编程（包括 ^ 和 ProbLog），并给出了对应的干预操作定义；

**💡 创新点**

创新点在于：①识别并修正现有 ^ 与 ProbLog 语义在满足因果不相关性（causal irrelevance）方面的缺陷；②设计了一种基于最大熵与因果顺序的全新语义，既保留原有逻辑编程的表达力，又能正确处理干预；③提出了实现细节并在 PLP-BN 工具集里实现了该语义；

**🔧 技术方法**

主要技术包括：概率逻辑编程、LogLinear 模型、最大熵原理、因果图与强连通分量分析、贝叶斯网络转换以及 Clingo 答案集求解；

**📊 数据集**

本文未使用公开数据集，主要通过人工构造的示例（如“Anna 与 Kilian 回门”与“吃巧克力”情形）来验证语义的正确性与对干预的响应；

**📈 对比分析**

比较方法：对比原有 ^ 与 ProbLog 语义与新提出的语义在同一组示例中的干预预测结果，证明后者满足因果不相关性，避免了原先产生的反直觉概率分布；由于仅为理论与案例演示，未给出量化性能指标；

**⚠️ 局限性**

局限性：目前仅覆盖命题（propositional）概率逻辑编程，无法直接处理关系型或多值变量；实现依赖于 Clingo 与手工编码的强连通分量求解，规模较大时可能面临计算瓶颈；未来工作需扩展至非命题设置并整合时间序列因果框架。

---

## 423. CaVE: A Constraint Storage Approach to Handling Integrity Constraints

**arXiv ID:** 2607.21207 | [PDF](https://arxiv.org/pdf/2607.21207v1)

**作者:** Xiangyu Guo `[一作]` (Arizona State University), Ajay Bansal `[通讯]` (Arizona State University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为CaVE的约束存储方案，用于在stableKanren的解析过程中处理完整性约束；

**💡 创新点**

创新点在于将完整性约束拆分为发射器与验证器，并利用延续技术实现基于解析的约束验证，避免了传统的基化或解析修改；

**🔧 技术方法**

主要技术包括Scheme实现的stableKanren、miniKanren的关系式编程、宏与延续（continuation）构造约束存储以及新增的constrainto语法；

**📊 数据集**

使用了三种组合搜索问题（Hamiltonian cycle、Knight's Tour、SEND+MORE=MONEY）作为实验案例，未采用公开数据集；

**📈 对比分析**

通过示例展示该方法在保持关系性同时能够求解这些问题，虽未给出量化性能对比，但示例显示在无启发式时可在数分钟到数小时内得到解；

**⚠️ 局限性**

主要限制是仅支持硬约束，缺乏软约束、启发式搜索和CLP优化的支持，且在大规模问题上仍存在内存占用和搜索效率瓶颈。

---

## 424. chrKanren: Constraint Handling Rules in a Relational Language

**arXiv ID:** 2607.21204 | [PDF](https://arxiv.org/pdf/2607.21204v1)

**作者:** Rafaello Sanna `[一作]` (Harvard University), Nada Amin `[通讯]` (Harvard University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

整合 Constraint Handling Rules (CHR) 到纯粹关系逻辑编程语言的解释器中，使得约束传播与完整搜索能够协同工作；并利用该机制实现类型与示例驱动的程序合成与语义统一两类应用。

**💡 创新点**

首次将 CHR 的多头约束规则直接嵌入完整搜索的关系语言，并证明两者在组合后仍保持完整性；通过案例展示该整合能在原先的纯关系实现中无法完成的任务（如有限失败、用户自定义数据结构统一）得到实现。

**🔧 技术方法**

使用 CHR 规则定义约束求解器，改进搜索流的语义模型，构造了基于状态、约束库和规则历史的搜索状态；实现了基于流的执行器与约束传播器，并提供了类型系统与集合统一等示例。

**📊 数据集**

主要使用人工合成的示例数据：λ 演算项、集合表达式、对数值与符号的组合等；没有使用公开的标准数据集。

**📈 对比分析**

与纯关系解释器比较，展示了在同一查询下 CHR 约束实现能够在有限步骤内失败，而纯关系实现会无限循环；论文未给出定量性能指标，重点在功能完整性与语义正确性。

**⚠️ 局限性**

局限性包括：规则在不同 groundedness 模式下需要重复书写；缺乏高效的匹配与编译技术（如 RETE、LEAPS）导致约束传播效率低；缺少针对用户自定义数据结构的高层抽象机制，导致实现冗长且难以维护。

---

## 425. A New Well-Supported Semantics for Description Logic Programs

**arXiv ID:** 2607.21203 | [PDF](https://arxiv.org/pdf/2607.21203v1)

**作者:** Spencer Killen `[一作]` (University of Alberta), Jia-Huai You `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了一种新的描述逻辑程序的语义，旨在解决当前语义在一致性问题上的计算复杂性和缺乏还原变换特征的问题。

**💡 创新点**

创新点在于提出了一种更严格的语义评估本体原子，保持一致性问题的NP完全性，并且识别出一种语法类的描述逻辑程序，使得新语义与当前语义等价。

**🔧 技术方法**

使用了固定点算子和基于还原的变换来表征新的语义。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与现有方法相比，新的语义是当前支持语义的严格子集，所有新的答案集都是支持的，但一些支持的答案集被移除，表现出循环支持的特性。

**⚠️ 局限性**

限制在于新语义可能会移除一些支持的答案集，这些答案集在与程序的Clark完成比较时可能包含循环依赖。

---

## 426. Towards a Certifying Grounder

**arXiv ID:** 2607.21199 | [PDF](https://arxiv.org/pdf/2607.21199v1)

**作者:** Daimy Van Caudenberg `[一作]`, Bart Bogaerts `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可证明的Grounding框架CertiFOX，用于将FO模型扩展问题转换为等价的量化无关CNF，同时生成可验证的证明；

**💡 创新点**

创新点在于将证明记录扩展到Grounding阶段，设计了Grounding Normal Form（GNF）和专用证明格式，形成了从高层规格到低层输入的完整可信链；

**🔧 技术方法**

核心技术包括：GNF中的二元量化器与guards、等价保持的重写规则、基于位置系统的证明记录、独立的Proof Checker；

**📊 数据集**

实验采用DIRT基准集中的八个问题集（CommonItem、CompleteSets、GraphColouring、PPM、RamseyNumbers、stablemarriage等），并对比IDP、Clingo等现有grounder；

**📈 对比分析**

与IDP、Clingo的性能比较显示CertiFOX在大多数实例上与两者相当，证明生成几乎不增加时间，检查时间为Grounding时间的2–3倍；

**⚠️ 局限性**

局限在于只支持GNF子语言、对大型未加guard的深层量化（如RamseyNumbers）产生巨大证明、内存占用高、尚未实现正式验证的检查器以及对更广泛FOL约束的支持。

---

## 427. pAI-Econ-claude: A Gated Human-in-the-Loop Multi-Agent Architecture for AI-Assisted Economic Theory Development

**arXiv ID:** 2607.21268 | [PDF](https://arxiv.org/pdf/2607.21268v1)

**作者:** Chen Zhu `[一作]` (China Agricultural University), Weilong Zhang `[通讯]` (University of Cambridge)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种包含人机协作、多代理并行工作流的经济理论建模架构pAI‑Econ‑claude。

**💡 创新点**

创新点是引入门控监督层、可见中间工作记录和有限的人类决策点，提升可靠性而非完全自动化。

**🔧 技术方法**

采用多代理LLM架构、黑板式协作、专用故障检测门控、ReAct/AutoGen等技术。

**📊 数据集**

使用无标准数据集，基于五个匹配的经济理论任务（人力资本、健康经济、营养与宏观发展、消费者信息、农业食品系统）。

**📈 对比分析**

通过双盲A/B评估与过程追踪对比门控与无门控，门控在四个任务中获胜，失效严重度从1.58降至1.16，整体有用性从2.60升至3.10。

**⚠️ 局限性**

局限性在于缺乏正式验证，门控可能过度压缩重要机制，并且仍需高质量的人类判断。

---

## 428. Delayed Constraints in Narrowing for the Logic-Based Analyses of Real-Time Systems

**arXiv ID:** 2607.21205 | [PDF](https://arxiv.org/pdf/2607.21205v1)

**作者:** Santiago Escobar `[一作]` (Universitat Politècnica de València), Carlos Olarte `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于延迟折叠归约的符号验证框架，可对实时重写理论（如参数时序自动机、分布式协议）进行符号检验与合成，支持无限数量代理、无限状态空间以及未知参数的情况。

**💡 创新点**

提出了延迟折叠归约（delayed folding narrowing）技术，将SMT约束与逻辑变量统一处理，并引入折叠机制保证符号分析终止；在Maude引擎中实现了支持延迟约束的归约与折叠，并实现了对参数化系统的自动合成。

**🔧 技术方法**

归约（narrowing）、约束逻辑编程、SMT求解、重写逻辑、Maude重写引擎、符号状态空间、折叠（folding）技术、延迟约束处理。

**📊 数据集**

通过对时序Fischer互斥协议和带时钟的哲学家与侍者（Dining Philosophers）等正式模型的实验验证，没有使用外部数据集，均为基于公式的仿真与验证。

**📈 对比分析**

与传统基于采样或UPPAAL/IMITATOR等工具相比，本文方法能够在不固定进程数、参数的前提下完成符号覆盖，构造有限的折叠状态图；实验表明可在有限时间内验证安全性与可达性，尽管缺少具体数值性能基准，但展示了可终止性与更强的抽象能力。

**⚠️ 局限性**

实现仅在Maude的元层，未在C++层实现，导致效率仍受限；需要精心设计排序层次；目前仅支持无分支的初始模式，难以处理更一般的合成问题；对大规模参数化系统的可扩展性尚未充分评估。

---

## 429. Case study: solving P-99 with LPTP and an LLM

**arXiv ID:** 2607.21196 | [PDF](https://arxiv.org/pdf/2607.21196v1)

**作者:** Fred Mesnard `[一作]` (university of La Reunion), Wim Vanhoof `[通讯]` (university of Namur)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Claude LLM和LPTP对Ninety‑Nine Prolog Problems的前33个问题进行代码生成、单元测试和形式化证明，完成了所谓的vibe‑coding与vericoding实验。

**💡 创新点**

创新点在于首次将大型语言模型与逻辑程序定理证明器相结合，构建了Model Context Protocol（MCP）来实现LLM自动生成并提交LPTP证明，从而实现了代码与证明的闭环验证。

**🔧 技术方法**

使用技术包括Anthropic的Claude（Opus 4.6）在Code模式下生成Prolog代码、SWI‑Prolog 10.0.2执行测试、LPTP逻辑程序证明器进行形式化验证，以及MCP服务器实现LLM与LPTP之间的交互。

**📊 数据集**

数据集为经典的P‑99 Prolog练习集（88个问题），实验中仅使用了其前33个问题（P01–P28、P31–P35），并将生成的代码、测试、证明和实验报告存放在GitHub仓库中。

**📈 对比分析**

比较方法主要是人工手工检查每个生成文件、运行测试并用LPTP验证证明完整性；实验结果显示在33个问题中共生成58个逻辑程序、508个测试、257个命题，约11800行证明，解决率约为37.5%，证明通过率高但耗时从15分钟到数小时不等。

**⚠️ 局限性**

局限性包括：对功能性属性的证明依赖手动提示且仍有错误，LLM在复杂证明时会出现“hallucination”，需人工干预；实验规模受P‑99大小和模型计算资源限制，尚未验证在更大规模或更复杂问题上的可扩展性。

---

## 430. FORGE-plus: Force-Budgeted Recovery for Contact-Rich Assembly with a Frozen LLM Supervisor

**arXiv ID:** 2607.21227 | [PDF](https://arxiv.org/pdf/2607.21227v1)

**作者:** Kyupaeck Jeff Rah `[一作]` (Independent Researcher), Midum Oh `[通讯]` (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个两层系统，利用冻结的LLM根据物体身份设定每个零件的力上限，并在插装失败时仅用力签名选择恢复动作；结合FORGE式力条件RL技能与硬峰值力clamp，实现了对脆性零件的安全插装。

**💡 创新点**

创新点：①把物体身份文本与力签名相结合，让冻结LLM自动推断力上限并进行恢复决策；②在完全不使用视觉的情况下，仅通过力签名识别失败类型；③对比Oracle、全局上限等多基线，展示保守预算优于Oracle并验证“加压”恢复在两手臂上均失败；④系统性记录并公开多项负面实验结果。

**🔧 技术方法**

技术：FORGE力条件RL（PPO+BC+DAgger+weight soup）、冻结LLM（7–8B instruct模型）、力签名编码器、硬峰值力clamp、Isaac Lab仿真、操作空间阻抗控制OSC、评估脚本与日志收集。

**📊 数据集**

数据集：仅在内部仿真生成，包含脆性ABS齿轮、钢齿轮、脆性瓶子与稳固瓶子；每个零件的破坏阈值从预定义分布随机采样；不使用公开数据集。

**📈 对比分析**

比较方法：七个指标（成功率、破损率、预算合规率、恢复效果、力经济、clamp精度、跨手臂迁移）；结果显示统一检查点在Robotiq 2F‑140与Franka Panda手臂上实现256/256成功且无破损；力签名恢复相较于基线提高40–64%成功率；保守预算优于Oracle，且全局上限导致全部破损；加压恢复在两手臂上分别无效或导致96%脆性件破损。

**⚠️ 局限性**

limitations：仅在模拟环境下验证，破坏模型为标量阈值；未进行真实机器人实验；力签名和恢复仅在简化的故障模式下测试；LLM仅处理文本身份，未测试对抗性或多样化输入；系统仍包含脚本化组件，缺乏完全端到端学习；缺乏视觉/多模态输入，限制故障识别范围。

---

## 431. Anti-Periodic Positional Encoding: Möbius Boundary Conditions Make In-Context Retrieval Reliable

**arXiv ID:** 2607.21405 | [PDF](https://arxiv.org/pdf/2607.21405v1)

**作者:** Ji Ho Bae `[一作]` `[通讯]`, Ji Ho Bae

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在旋转位置编码中使用反周期边界条件（Möbius RoPE），并在 25% 的注意力头上以零成本实现；

**💡 创新点**

创新点在于利用 anti‑periodic 频率锚定构造一个端到端的 Dirichlet “偶极子”通道，从而消除模型在小规模预训练中的种子彩票现象；

**🔧 技术方法**

技术上采用半整数谐波频率 θ_i=π(2i+1)/N、混合头分配、闭式 Dirichlet 公式、needle‑in‑a‑haystack 检测、对照实验和权重冻结 ablation；

**📊 数据集**

实验数据来自 FineWeb‑Edu 2B 语料，使用 GPT‑NeoX 分词器，训练 160M 和 410M 参数规模的 transformer；

**📈 对比分析**

通过对比 perplexity、NIAH 准确率及其方差，Möbius RoPE 在保持相同 perplexity 的同时，使 160M 模型在 512 长度下最差种子准确率从 14% 提升至 86%，方差显著下降；

**⚠️ 局限性**

局限性包括模型规模仅达 405M、仅验证单针检索任务、在训练窗口之外检索性能迅速崩溃、统计功效有限以及训练数据顺序可能产生的偏差。

---

## 432. ASTRA-Net: Anatomy-Specific Transfer and Representation Alignment for Drug-Induced Sleep Endoscopy Segmentation

**arXiv ID:** 2607.21370 | [PDF](https://arxiv.org/pdf/2607.21370v1)

**作者:** Suhua Sun `[一作]` (Peking University Third Hospital), Yan Yan `[通讯]` (Peking University Third Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种ASTRA-Net模型，用于在药物诱导睡眠内镜（DISE）中仅使用有限真实标注的情况下实现面向特定解剖平面的像素级分割。

**💡 创新点**

创新点在于：①将无标注的CT导出虚拟内镜图像仅用于特征对齐；②采用面向解剖层次的多解码器结构与结构化零掩码监督；③在对齐阶段仅使用最大均值散度（MMD）或域对抗学习，无需伪标签或图像翻译。

**🔧 技术方法**

技术主要包括ConvNeXt-Base编码器、UNet++解码器、MMD与DANN域对齐、软Dice+焦点损失、结构化零掩码监督、五折集成与已知平面推断。

**📊 数据集**

使用了14,250张无标注的CT虚拟内镜视图、998张带平面标签的真实DISE帧、401帧带分割掩码的真实DISE帧，以及156帧无效帧，作为训练集；另外100帧真实DISE帧作为独立hold‑out评估集。

**📈 对比分析**

通过在hold‑out集上比较六种对齐配置（MMD‑only、DANN‑only、混合及三种调度），使用Dice、IoU、精准率、召回率、特异性与四平面分类准确率等指标。MMD‑only配置获得最高平均Dice 0.8927（95% CI 0.8631–0.9160），四平面分类准确率0.92。

**⚠️ 局限性**

局限性包括：①缺乏真实对齐/无对齐基线与组件消融实验，无法量化对齐与结构化监督贡献；②仅使用帧级拆分，缺乏病人/视频层面独立性验证；③评估仅覆盖静态已知平面分割，未涉及平面路由、无效帧拒绝、时序稳定性及临床量化评估。

---

## 433. Toward Federated Cognitive Digital Twins over the Edge-to-Cloud Continuum

**arXiv ID:** 2607.21357 | [PDF](https://arxiv.org/pdf/2607.21357v1)

**作者:** Alessandra Somma `[一作]` (University of Naples Federico II), Alessio Bucaioni `[通讯]` (M"alardalen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种基于联邦与认知的分层数字孪生架构（FCDT），在边缘到云连续体上将智能分散到本地孪生与全局孪生中，实现快速本地决策与系统级语义推理。

**💡 创新点**

创新点在于将联邦数字孪生的结构解耦与认知数字孪生的语义推理结合，在分布式环境中通过轻量级边缘模型和强大云端LLM实现可扩展、低延迟、可解释的决策支持。

**🔧 技术方法**

使用了联邦架构、边缘计算、云计算、轻量级AI模型（如边缘语言模型）、大型语言模型（LLM）、语义蒸馏技术、共享数据空间以及模型分配与协同推理机制。

**📊 数据集**

目前未使用具体数据集，论文仅提出架构设计与理论分析，后续工作计划在真实分布式场景中进行实现与评估。

**📈 对比分析**

未进行实验比较；论文中提出的性能预期包括降低延迟、提升本地自治、减少通信开销并实现更高的可解释性，后续将通过原型验证与基准测试来衡量。

**⚠️ 局限性**

主要局限包括缺乏实现与实验验证、模型分配策略尚未确定、隐私与数据安全挑战、跨域语义互操作性问题、以及大规模联邦协同推理的通信与算力调度难题。

---

## 434. Phonetic forced alignment for low-resource language varieties: Model training and evaluation on Chengdu Mandarin

**arXiv ID:** 2607.21332 | [PDF](https://arxiv.org/pdf/2607.21332v1)

**作者:** Zhiheng Qian `[一作]` (Shanghai Jiao Tong University), Liang Zhao `[通讯]` (Beijing Foreign Studies University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并发布成都方言的文本依赖和无文本的语音强制对齐模型，并提供专门的G2P字典。

**💡 创新点**

设计了一套端到端的自举流程（G2P→文本依赖对齐→伪标签→文本无关对齐），在低资源环境下无需人工标注即可获得高质量对齐模型。

**🔧 技术方法**

结合GMM‑HMM (MFA) 与预训练语音编码器 fine‑tune 的 frame‑classification (Wav2Vec2、XLS‑R、Charsiu) 以及边界加权学习策略。

**📊 数据集**

约17小时成都普通话语音及其手工注释的语料，包含2876汉字的自定义 G2P 字典。

**📈 对比分析**

使用专家手工标注的50分钟测试集评估绝对时间误差、Precision/Recall/F1/R 值，结果成都‑MFA/FC 平均误差比标准普通话降低约30‑60%，在无文本场景中显著提高 R 值。

**⚠️ 局限性**

仅在训练集包含的说话人上测试，未验证对未见说话人或不同语域的泛化能力；文本无关模型仍不及文本依赖模型。

---

## 435. Empowering Rural Areas with Multi-radio Microwave Backhaul Supported by Digital Twin for 5G IAB-based FWA

**arXiv ID:** 2607.21310 | [PDF](https://arxiv.org/pdf/2607.21310v1)

**作者:** Anselme Ndikumana `[一作]` (École de Technologie Supérieure Universite Du Quebec), Mohamed Cheriet `[通讯]` (École de Technologie Supérieure Universite Du Quebec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在农村地区构建多跳无线回传网络，结合长波段微波链路、5G IAB及FWA，提出基于数字孪生的能效优化框架

**💡 创新点**

提出五状态（完全关闭、启动、服务、深度睡眠、唤醒）模型并通过数字孪生+深度Q学习学习最优状态转移，实现能耗与速率双重优化

**🔧 技术方法**

使用深度Q学习（DQL）、深度双层网络、数字孪生（DT）、优化求解器（Gurobi/CPLEX）和Age of Processing（AoP）指标

**📊 数据集**

仿真使用加拿大魁北克乡村场景的合成一周时延为1秒的流量数据，以及两频段（7 GHz、42 GHz）微波设备功耗参数

**📈 对比分析**

与传统三状态模型和始终处于服务状态的基线进行对比，实验表明在满足5G FWA数据率需求的前提下，能耗下降约47.3%，并实现较低传输功率与平均速率提升

**⚠️ 局限性**

仅在仿真环境验证，缺乏实际现场部署；对数字孪生实时同步和大规模网络规模的可扩展性未作评估

---

## 436. Expert Behavior Prior Reinforcement Learning

**arXiv ID:** 2607.21302 | [PDF](https://arxiv.org/pdf/2607.21302v1)

**作者:** Gong Gao `[一作]` (Tongji University), Ning Jia `[通讯]` (Tongji University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于在线回放缓冲区的生成式专家行为先验（EBP）方法，利用 Q‑引导的 CVAE 生成高价值动作来指导 Actor‑Critic 的在线学习。

**💡 创新点**

创新点包括：① 在在线训练中直接用 Q‑引导的 CVAE 学习专家先验，摆脱对离线数据的依赖；② 引入专家策略指导（EPG）以多样化高价值支持集为更新锚点；③ 通过梯度校正（PGC）平衡 Q‑引导与专家监督，提升学习稳定性。

**🔧 技术方法**

采用 Actor‑Critic 框架（TD3/ DDPG）结合 Q‑CVAE、行为克隆、离线经验重放、同步策略更新、梯度相似度评估与自适应权重调整等技术。

**📊 数据集**

在 OpenAI Gym、PyBullet、DeepMind Control Suite 等连续控制任务（如 HalfCheetah‑v3、Ant‑v3、Hopper‑v3 等）进行实验评估。

**📈 对比分析**

与 TD3、DDPG、SAC、BAC、ALH、NNPG 等基线进行 10 次随机种子实验；EBP 在多数任务上平均提升 19–51% 的样本效率，并在最终收敛性能上显著优于所有基线。

**⚠️ 局限性**

主要局限包括对 Q 值估计误差敏感，奖励噪声下性能表现可能波动；PGC 与 H、μ 等超参数需精细调节；额外的 Q‑CVAE 训练带来一定的计算与显存开销。

---

## 437. MemTools: A Unified Research Framework for Interoperable Agent Memory

**arXiv ID:** 2607.21404 | [PDF](https://arxiv.org/pdf/2607.21404v1)

**作者:** Chengfeng Zhao `[一作]` (Institute of Automation, Chinese Academy of Sciences), Kang Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 MemTools 框架，提供标准化的内存生命周期接口、解耦评估协议与数据集、统一管理符号、神经与多模态记忆，并支持跨系统组件组合与可插拔评估；

**💡 创新点**

通过声明式数据契约实现模块解耦并提供自动匹配引擎；将评估协议与基准数据分离；统一接口协调多种记忆表征，实现跨架构实验与混合记忆系统；

**🔧 技术方法**

声明式数据契约、自动匹配引擎、Python API、统一计算接口、并行检索与链式适配、抽象化插件化评估协议、同步搜索层等技术；

**📊 数据集**

ALFWorld Benchmark、Mem‑Gallery 以及其他用户/代理中心任务数据集（如对话问答、导航、编码、决策等）；

**📈 对比分析**

在统一协议下对比原生与混合组件系统、对同一管线在不同评估协议（Batch vs Stream）下的成功率、以及不同记忆表征在单独与协调模式下的 F1/成功率；实验显示跨系统组合可匹配或超过原生系统，批处理协议优于流式协议，混合记忆系统在 Mem‑Gallery 和 ALFWorld 上分别提升到 35.3/51.5；

**⚠️ 局限性**

自动匹配仅验证字段兼容，可能忽略细微行为差异；抽象层带来额外计算开销，尤其在大型异构数据库和长任务时可能成为瓶颈；目前主要面向受控实验环境。

---

## 438. When Are Reasoning-Based Guardrails Not Efficient? ResponseGuard: A Fast Vision-Language Guard for Real-Time Moderation

**arXiv ID:** 2607.21401 | [PDF](https://arxiv.org/pdf/2607.21401v1)

**作者:** Dongbin Na `[一作]` `[通讯]` (Pohang University Of Science And Technology), Dongbin Na (Pohang University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出了一种无需链式推理的视觉语言安全守护器ResponseGuard，能够在响应流中实时检测并阻止有害内容。

**💡 创新点**

创新点在于将判断任务化简为单次前向推断，摒弃耗时的链式思考，显著提升速度并在响应有害检测上超过现有3B推理守护器。

**🔧 技术方法**

采用了2B规模的视觉语言嵌入骨干（冻结视觉编码器）以及小型参考向量头，进行一次池化后直接输出有害概率。

**📊 数据集**

使用了公开的多模态安全基准（HarmBench-Response、Safe RLHF、BeaverTails等）以及对应的训练语料库123k样本。

**📈 对比分析**

与3B CoT推理守护器对比，ResponseGuard在响应路径的F1平均值提升0.75点，并且推理时间约为1/150，速度提升约150倍；在提示路径上差距集中于图像单元。

**⚠️ 局限性**

限制在于对图像的感知能力不足，主要受冻结视觉编码器的性能限制，链式推理对图像判断的帮助有限。

---

## 439. CRAFT: Exploring Wearable Creative AI on Smart Glasses for Fiction Writing in Real-World Contexts

**arXiv ID:** 2607.21394 | [PDF](https://arxiv.org/pdf/2607.21394v1)

**作者:** Runze Cai `[一作]` (National University of Singapore), Shengdong Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于智能眼镜的“CRAFT”方法，支持作者在日常场景中即时捕捉与转化现实体验为小说素材。

**💡 创新点**

创新点在于将多模态 LLM 与 AR 眼镜结合，提供“主动建议–即时转化–角色扮演”等混合主动交互机制，并实现“微创作”与“现实-虚构”混合可视化。

**🔧 技术方法**

使用了Xreal Air智能眼镜、FPV摄像、蓝牙环鼠标以及 Gemini 大语言模型与多模态推理流水线，实现场景感知、知识检索与文本/图像生成。

**📊 数据集**

数据集主要为参与者在三次实地试验中生成的 215 条现实元素（物体、人、场景）及 24 场次的语音与图像日志；未使用公开文本语料库，而是基于用户自身捕获的真实环境数据。

**📈 对比分析**

通过作者主观评分（平均 5.8/7 的创作支持感）与交互日志（每场次约 30 次主动/用户触发），证明相较传统桌面写作，实时 AR 支持能显著提升灵感捕获与素材真实性，然而未与基准工具做定量对比。

**⚠️ 局限性**

局限包括：硬件受限为有线实验、延迟约 8 秒，样本量小、实验周期短，缺乏长期使用评估；未解决隐私与社交适配问题，也未对文本质量进行第三方评测。

---

## 440. Word meaning co-determines vowel-inherent spectral change. A corpus-based investigation of conversational Mandarin

**arXiv ID:** 2607.21391 | [PDF](https://arxiv.org/pdf/2607.21391v1)

**作者:** Xiaoyun Jin `[一作]` (Eberhard Karls Universität Tübingen), R. Harald Baayen `[通讯]` (Eberhard Karls Universität Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在台湾口语对话语料中分析/ā/、/i/、/u/、/ə/的元音固有频谱变化（VISC），并检验其是否包含词特异性及语义影响；

**💡 创新点**

发现VISC在自然交谈中显著存在，且可由词义预测，表明语义与元音语音细节共决定；

**🔧 技术方法**

采用线性预测编码提取声谱参数、广义加性模型（GAM）分解时间序列、词向量上下文嵌入与Discriminative Lexicon Model（DLM）线性映射；

**📊 数据集**

使用台北语料库（Taiwan Mandarin Spontaneous Corpus）共计7,016个词元，涵盖99种词，118,592个词素；

**📈 对比分析**

通过AIC、交叉验证与误差平方和（SSE）比较模型，GAM+词义/词类/词义+词类的预测优于基线，DLM对语义嵌入预测元音轨迹的准确率约为10%–45%（相对随机基线0%），显示显著提升；

**⚠️ 局限性**

局限在于样本量有限、词义标注噪声、使用LPC导致的频谱偏差、未考虑说话者情感与语域差异，且线性映射可能不足以捕捉更复杂的语义-语音关系。

---

## 441. Reachability in Directed Acyclic Graphs with Near-Linear Cut Queries

**arXiv ID:** 2607.21390 | [PDF](https://arxiv.org/pdf/2607.21390v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在割查询模型下，提出了一个 O(n log³ n) 的算法，用于在有向无环图（DAG）中求顶点的拓扑排序和单源可达性。

**💡 创新点**

创新点在于引入了懒惰源剥离和分桶树结构，实现了对残余入度的近似维护，突破了传统方法的二次查询限制。

**🔧 技术方法**

技术上使用了割查询、二分搜索、分桶/平衡二叉树、延迟同步以及层次化的入度计数。

**📊 数据集**

无实验数据集，纯理论分析。

**📈 对比分析**

与以往仅在无向图中实现 O(n) 或 O(n^{3/2}) 割查询的结果相比，本文首次在有向图中实现了亚二次的可达性查询，查询复杂度为 O(n log³ n)（单源可达）及 O(n log³ n)（拓扑排序）。

**⚠️ 局限性**

局限性在于仅适用于 DAG，尚未扩展到一般有向图，也未解决更复杂的图属性，如最短路径或最大流的割查询实现。

---

## 442. A Logic-based Temporal Cohort Discovery Engine: Algorithms, Indices, and Experimental Results on the National Sleep Research Resource

**arXiv ID:** 2607.21377 | [PDF](https://arxiv.org/pdf/2607.21377v1)

**作者:** Yan Huang `[一作]` (University of Texas Health Science Center at Houston), Guo-Qiang Zhang `[通讯]` (University of Texas Health Science Center at Houston)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了基于QEL逻辑的时间序列生物医学事件模型(BEST)，实现了可解释、可复现的睡眠研究队列发现引擎；

**💡 创新点**

创新点在于将密集时间逻辑与模型检验结合，并设计了两种专用索引（2DFC和FCFC）来实现大规模睡眠数据集上的高效时间模式检索；

**🔧 技术方法**

使用了Rational Ensemble Logic（QEL）作为查询语言，BEST作为事件集合模型，Fractional Cascading（2DFC、FCFC）作为索引结构，Python实现并结合MongoDB后端；

**📊 数据集**

评估数据集包括90 M级合成区间数据和真实的National Sleep Research Resource（CCSHS）数据集（515名受试者、202 587条区间、23种事件标签）；

**📈 对比分析**

与传统的2DRT、RTFC及基于NFA的模式匹配进行对比，2DFC在构建时间和查询时间均显著优于传统结构（构建时间从3 k s降至1 k s，查询时间保持在秒级）；FCFC在双事件模式匹配上实现了80–98 %（内存）或≈50 %（MongoDB）时间缩短；

**⚠️ 局限性**

限制在于需要预先归一化为非重叠区间；FCFC的O(ml)空间占用较大；缺乏增量更新支持；以及对深层嵌套QEL公式的完整优化尚未实现。

---

## 443. Electromagnetic-Aware Fluid Antenna Array

**arXiv ID:** 2607.21375 | [PDF](https://arxiv.org/pdf/2607.21375v1)

**作者:** Zhentian Zhang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种电磁感知的当前域框架，联合优化平面流体天线阵列（FAA）的端口位置和电流，实现单波束超定向波束成形与多用户加权总速率最大化。

**💡 创新点**

创新点在于将互耦、电磁功率、源电压和几何约束全程嵌入到优化模型中，利用闭式KKT解和梯度替代方法实现可扩展的交替求解，并展示互耦可被正向利用以提升旁瓣抑制与速率。

**🔧 技术方法**

核心技术包括：多端口电磁阻抗模型、当前域电路等效、凸QCQP与SOCP求解、分数规划（FP）与锥型变换、梯度嵌入的内点/凸逼近、Armijo退火线搜索以及多端口矩阵求导。

**📊 数据集**

实验使用基于半波偶极子、8个端口、1.25λ×0.75λ平面、最小间距0.2λ的模拟数据；对比随机与固定网格FAA，未提供真实测量数据集。

**📈 对比分析**

与固定网格（无电磁、含电磁但无约束、含电流、含电流+电压）以及最佳随机FAA比较，优化FAA在PSLL上从-6.0 dB提升至-9.0 dB，单波束功率降低约1 dB；在多用户情形下，总速率中位数提升1.1 bit/s/Hz（相对最佳随机）和2.3 bit/s/Hz（相对固定网格）。

**⚠️ 局限性**

局限性包括：求解结果为局部最优，缺乏全局性能保证；仅针对平面半波偶极子阵列，未验证对更复杂天线/场景的泛化；实验仅为数值仿真，缺乏实验验证；模型假设的电磁互耦可用闭式或全波仿真得到，实际部署中可能需要更精细的测量或替代模型。

---

## 444. Mean-to-Score Discrete Diffusion: Posterior-Mean Denoisers for Score Entropy

**arXiv ID:** 2607.21372 | [PDF](https://arxiv.org/pdf/2607.21372v1)

**作者:** Jingyuan Li `[一作]` (Beijing Institute of Mathematical Sciences and Applications), Pipi Hu `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了“Bayes可实现性”这一结构约束，对离散扩散模型中分数向量的参数化进行改进；

**💡 创新点**

创新点在于提出Mean-to-Score (M2S) 方法，通过预测一站点的清晰词后验并利用已知前向核的线性桥将其映射为完整分数向量，从而强制满足Bayes可实现性；

**🔧 技术方法**

技术包括离散时间连续马尔可夫链(CTMC)建模、分数熵损失、线性桥投影、对齐的预训练损失（MD4）以及欧式投影校正；

**📊 数据集**

数据集涵盖图像任务（MNIST、CIFAR-10）和文本任务（OpenWebText），并在相同规模的DiT/Transformer结构下训练；

**📈 对比分析**

与基线（SEDD、MDLM、GIDD、Neural CTMC）在相同模型规模和训练预算下比较，M2S 在 MNIST、CIFAR-10 上分别提升 FID 52+ 点，CIFAR-10 BPD 从 3.173 降至 3.129；在 OpenWebText 上，M2S 在 128 步时将 PPL 从 183.6 降至 143.3，整体表现优于所有评估的统一或混合方法；

**⚠️ 局限性**

局限在于目前仅验证了纯均匀腐蚀和吸收式掩码两种前向核，未证明在更复杂或更高维的语料上同样有效；此外，M2S 的推理过程中仍需额外的线性桥计算，可能导致计算开销略高。

---

## 445. Gradient Concentration, Not Weight Saliency, Explains Representation-Level Class Unlearning

**arXiv ID:** 2607.21353 | [PDF](https://arxiv.org/pdf/2607.21353v1)

**作者:** Billel Habbati `[一作]` (University of Genova), Meriem Guerar `[通讯]` (University of Genova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对类级机器遗忘中的权重显著性掩码机制进行匹配计算消融实验，探究其对内部表示层级遗忘的实际贡献。

**💡 创新点**

发现显著性掩码与随机掩码或无掩码更新在表示层级遗忘效果上无显著差异，揭示梯度能量高度集中于网络后层并且掩码几乎缺乏类别特异性。

**🔧 技术方法**

使用随机标签（SalUn）遗忘目标、基于梯度显著性与Fisher信息的二进制掩码、随机掩码、以及全参数更新；评估线性探测、原型恢复、层级CKA等表示层级指标。

**📊 数据集**

CIFAR-10 与 CIFAR-100 数据集，模型为 ResNet‑18。

**📈 对比分析**

在所有条件下，遗忘准确率均降至 0；在线性探测与原型恢复等表示层级指标上，三种掩码方案（显著性、随机、全更新）表现相当，均明显高于精确重训练（Gold）并远低于随机初始化（RandInit）。

**⚠️ 局限性**

实验仅覆盖小规模卷积网络和类别级遗忘，未验证在大规模数据集、变压器或多模态模型中的可迁移性；梯度能量集中机制虽经验验证但未证明普适；原型恢复在细粒度数据集上不够稳定；掩码特异性评估仅基于 Jaccard 重叠，未涉及功能等价性。

---

## 446. M$^3$-Gen: Interpretable Multimodal Generation of Gene Expression Profiles Using Clinical and Imaging Data

**arXiv ID:** 2607.21343 | [PDF](https://arxiv.org/pdf/2607.21343v1)

**作者:** Francesca Pia Panaccione `[一作]` (Politecnico di Milano), Marco Venere `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研发了一个多模态生成框架M3-Gen，利用病理图像与临床文本条件生成基因表达谱；

**💡 创新点**

创新点在于将视觉与文本两模态通过对比学习对齐，并用多头注意力融合形成可解释的生成条件，使模型能够显式定位对基因表达贡献最大的图像区域；

**🔧 技术方法**

使用了Contrastive Pretraining（InfoNCE）对图像Encoder（UNI）和文本Encoder（Clinical ModernBERT）进行对齐；随后采用条件WGAN‑GP作为生成器，输入噪声与融合后的多模态嵌入；实现多头注意力以实现解释性；

**📊 数据集**

基于TCGA公开数据，包含12种肿瘤类型的WSI、相应的临床元数据与RNA‑Seq基因表达；

**📈 对比分析**

通过与无条件WGAN‑GP、单模态（仅图像或仅文本）以及均值融合等基线对比，评估Precision、Recall、Correlation、检测可区分性、以及基于RF/MLP的分类准确率，M3‑Gen在多模态注意力下实现了更低的可区分度、更高的分类准确率和更逼真的基因表达分布；

**⚠️ 局限性**

局限性包括对图像噪声的鲁棒性仍有限，需更多数据来提升多模态融合的泛化；模型目前仅生成基因表达谱，未实现双向生成；生成多样性受限，且尚未验证在其他生物学尺度（如蛋白质）上的适用性。

---

## 447. GRADRAG: Cross-Component Prompt Adaptation for Coordinated Multi-Agent RAG

**arXiv ID:** 2607.21324 | [PDF](https://arxiv.org/pdf/2607.21324v1)

**作者:** Paolo Pedinotti `[一作]` (Bloomberg), Enrico Santus `[通讯]` (Bloomberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GradRAG 框架，利用评估者反馈在检索-生成管道中跨组件地自适应地更新提示，从而在测试时实现多组件协同改进。

**💡 创新点**

创新点在于：①将 RAG 视为计算图并在节点间传播结构化反馈；②允许下游评估直接驱动上游检索和图构建等组件的提示更新；③通过早停机制避免过度迭代。

**🔧 技术方法**

技术包括：LLM 评估者（对答案和证据进行结构化点评）、Prompt Optimizer（将点评转化为提示更新）、IRCoT 样式检索策略、图结构检索（实体–关系图构建与社区检测）以及 Gemini‑2.5‑Flash 作为生成模型。

**📊 数据集**

使用 SQuALITY 与 QMSum 两个问答/摘要基准数据集，分别代表叙事文本和会议转录。

**📈 对比分析**

方法对比：在向量检索和图检索两种架构下，将全 GradRAG 与仅对答案生成器进行一次性细化的基线进行 LLM‑Judge 双方对比。GradRAG 在两套数据上都获得 12–15% 的净优先率，且大多数情况下统计显著；平均仅需 2 次迭代即可收敛。

**⚠️ 局限性**

局限：①计算成本显著增加（多轮 LLM 调用导致时间与 token 消耗提升约 10%）；②评估协议以单实例迭代为主，缺乏批量或持续反馈的部署可行性；③评估者仅基于通用文本质量指标，未加入外部奖励或任务专属度量，可能影响评价精度。

---

## 448. Information is all you need: Requirements Engineering Quality Reframed

**arXiv ID:** 2607.21319 | [PDF](https://arxiv.org/pdf/2607.21319v1)

**作者:** Henning Femmer `[一作]`, Julian Frattini `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出把需求工程视为信息粒子在源与目标之间的流动，并用信息流模型解释质量和异常

**💡 创新点**

创新点在于将RE质量从传统的 artifact/过程视角转向整体信息流视角，提出信息粒子与流量的概念和两种形式的正式化

**🔧 技术方法**

使用信息理论、Beta分布函数建模信息转移、仿真和情景模拟

**📊 数据集**

未使用真实工业数据，采用人为设定的参数进行实验仿真

**📈 对比分析**

通过仿真对比瀑布式与敏捷式两条信息流，发现当直接沟通复杂度降低时开发者更倾向绕过规范；未给出量化性能指标

**⚠️ 局限性**

主要局限是缺乏实证数据支持、参数设定主观、未考虑信息错误、价值、衰减等因素，模型仅为示例性演示

---

## 449. Unlearning Under Imbalance: Benchmarking Fairness in Multimodal LLM Unlearning

**arXiv ID:** 2607.21300 | [PDF](https://arxiv.org/pdf/2607.21300v1)

**作者:** Lorenzo Orsingher `[一作]` (University of Trento), Elisa Ricci `[通讯]` (University of Trento)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对多模态大型语言模型在不平衡遗忘请求下的公平性评估与改进方法；

**💡 创新点**

创新点在于首次构建大型多模态无知性数据集 FAIRGET 并设计 FAUN 算法，在永久激活调节与偏差信息主成分分析相结合下实现公平无知性；

**🔧 技术方法**

主要技术包括永久激活调节（activation steering）与偏差信息主成分分析（bias‑informed PCA），并在 VQA 场景下实现了对身份信息的有选择性遗忘；

**📊 数据集**

使用了 4k 虚构身份、40k 图像、225k Q&A 的 FAIRGET 数据集（覆盖 13 个属性）以及 FIUBench 作为对照基准；

**📈 对比分析**

与多种基线（梯度上升、随机标记、SimNPO 等）比较，FAUN 在遗忘效果（EM、MIA）与公平度（DP）上均优于或与 SOTA 相当，同时保持较高的模型实用性；

**⚠️ 局限性**

限制在于依赖于多模态生成的人工数据，且在极端不平衡比例下对个别少数群体的公平性提升仍有限。

---

## 450. MSBraM: A Multi-scale Self-supervised Brain Foundation Model for Hierarchical EEG Dynamics Learning

**arXiv ID:** 2607.21402 | [PDF](https://arxiv.org/pdf/2607.21402v1)

**作者:** Tao Zhou `[一作]` (Hunan University), Zixing Zhang `[通讯]` (Hunan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出MSBraM，一种多尺度自监督脑基础模型，能够学习EEG信号的层次化时空表示；

**💡 创新点**

创新点包括：①多尺度神经分词器与多尺度码本实现不同时间分辨率的离散化；②课程式多尺度遮掩策略逐步从局部到全局学习；③BiFPN融合模块实现高效跨尺度特征融合；

**🔧 技术方法**

使用技术包括：向量量化FFT重构的多尺度分词器；多分支Transformer编码器；多尺度遮掩预训练；BiFPN融合；以及标准的自监督预训练与下游微调流程；

**📊 数据集**

预训练数据累计超过2400小时的公共EEG数据；下游评估在12个公开数据集上，涵盖10类任务（8分类、2回归）；

**📈 对比分析**

与多种监督与自监督基线（SPaRCNet、ContraWR、CNN‑Transformer、FFCL、ST‑Transformer、BIOT、LaBraM、EEGPT、CBraMod）进行对比，MSBraM在11/12个数据集上取得最优或同等最优性能，显著提升平衡准确率、Kappa、F1、相关系数等指标；

**⚠️ 局限性**

局限性包括：仅在表面EEG上验证；通道与时间维度平面化导致计算量高、空间结构利用不足；未扩展到颅内EEG或多模态整合。

---

## 451. Regulating autonomous and agentic AI

**arXiv ID:** 2607.21345 | [PDF](https://arxiv.org/pdf/2607.21345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 452. Multimodal Pretraining for Generalizable EEG Representation Learning

**arXiv ID:** 2607.21384 | [PDF](https://arxiv.org/pdf/2607.21384v1)

**作者:** Targol Bakhtiarvand `[一作]` (University of Colorado Colorado Springs), Adham Atyabi `[通讯]` (University of Colorado Colorado Springs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种融合原始EEG、CWT时频图与文本信息的多模态EEG基础模型，并在癫痫发作检测任务中进行微调

**💡 创新点**

创新点在于：①在单一共享嵌入空间中联合学习原始时域、时频域与文本表示；②使用Mamba、ViT和轻量文本分支的混合编码器；③通过掩码建模、跨视角对比学习和时序一致性损失实现无标签预训练；④严格采用LOSO交叉验证评估跨受试者泛化；

**🔧 技术方法**

技术包括Mamba架构、Vision Transformer、检索增强文本分支、InfoNCE对比损失、掩码重建、未来预测损失、AdamW优化、梯度累积与焦点损失、GradCAM可解释性分析

**📊 数据集**

使用CHB‑MIT（24个儿童患者）进行微调与评估，预训练数据来自TUH EEG和SEED‑DV公开数据；同时在BrainRVQ固定拆分（chb01–19 训练、20–23 测试）进行基准对比

**📈 对比分析**

方法与现有最佳模型（BrainRVQ、CBraMod等）比较，参数高效微调（仅7.5%可调）实现BAcc 0.781、AUC‑PR 0.709、AUROC 0.863；LOSO评估平均BAcc仅0.558；5折集成在chb22‑23上AUROC 0.878，AUC‑PR 0.729，优于BrainRVQ的AUROC 0.871

**⚠️ 局限性**

局限性包括：跨受试者泛化仍低；少量患者（chb01–05）以外的个体级微调与少样本校准未充分验证；早期预警仅在单一受试者上展示，需进一步验证

---

## 453. Towards Faithful Graph Explanations with Synergistic Edge Effects via Granular Balls

**arXiv ID:** 2607.21381 | [PDF](https://arxiv.org/pdf/2607.21381v1)

**作者:** Jiancu Chen `[一作]` (Chongqing University of Posts and Telecommunications), Fan Chen `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种无参数的图神经网络解释器SeeExplainer，通过将图拆分为无固定大小的granular-ball子图并构造结构图来捕捉边之间的协同效应，然后基于贡献阈值直接生成解释子图。

**💡 创新点**

创新点在于：①首次将granular-ball计算与图非同构分解结合，形成能体现边协同的结构图；②使用无参数阈值化方法生成解释子图，避免了传统方法的超参数依赖；③在多数据集上显著提升可解释性与稳定性。

**🔧 技术方法**

主要技术包括granular-ball计算、图非同构分解、BFS分割、无参数阈值贡献评估（基于平均预测置信度），以及对结构图节点和边的贡献度量。

**📊 数据集**

使用的图分类数据集有MUTAG、Mutagenicity、NCI1、NCI109、IMDB-BINARY、ENZYMES、PROTEINS、DHFR、BZR、DD等常用公开数据集。

**📈 对比分析**

与PGExplainer、DeepLIFT、GNNExplainer、GraphLime、GradCAM、Eig-Search等基线在GIN和GCN模型上比较，采用Fidelity和Stability等指标，SeeExplainer在绝大多数数据集上获得最高Fidelity、最低稳定性误差，且时间复杂度为O((m+n)logn)。

**⚠️ 局限性**

局限性包括：仅在图分类任务上验证；对动态图、异构图或极大规模图的适用性尚未评估；解释子图的阈值化方式可能忽略更细粒度的交互信息。

---

## 454. Incremental Optimal Assignment for Real-Time Crowd Tracking

**arXiv ID:** 2607.21368 | [PDF](https://arxiv.org/pdf/2607.21368v1)

**作者:** Ismail H. Toroslu `[一作]` `[通讯]` (Middle East Technical University), Ismail H. Toroslu (Middle East Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种增量式最优匹配算法，用于在密集人群跟踪中解决每帧检测与轨迹的匈牙利式二分匹配问题。

**💡 创新点**

创新点包括：① 在增量步骤中利用前一步已最优的对偶势能，使每次扩展仅需一次最短增广路径搜索；② 设计对角重排不变式（SparseReorder）保持匹配结构，进一步减少内存访问；③ 在块稀疏的成本矩阵上实现热启动，显著降低计算量。

**🔧 技术方法**

使用整数化的欧氏距离代价、对偶势能框架、增量式最短路径搜索（Dijkstra+优先队列）、对角重排机制和稀疏邻接表实现。

**📊 数据集**

使用基于真实人群行为的仿真模型生成三种场景（S1稀疏聚集、S2稠密聚集、S3交叉聚集），每种场景中人数从200到5000。

**📈 对比分析**

与密集版Jonker‑Volgenant匈牙利算法对比，实验显示在N≥200时速度提升3.7–6.5×，并能在25fps下实时处理至约1000人，性能随N增长而持续提高。

**⚠️ 局限性**

局限在于仅处理增量扩展，缺乏对减量（对象消失）或重新识别成本的直接支持；对GPU并行化仍待进一步研究；在极稀疏场景中增量优势不明显。

---

## 455. Teaching Business Process Modeling to Leverage Soft Skills of Computing Students

**arXiv ID:** 2607.21344 | [PDF](https://arxiv.org/pdf/2607.21344v1)

**作者:** Maria Istela Cagnin `[一作]` (Federal University of Mato Grosso do Sul), Elisa Yumi Nakagawa `[通讯]` (University of São Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实施了基于行业驱动的项目式学习课程，教授计算机专业学生BPM（BPMN）建模，并通过收集学生反馈评估硬技能与软技能的提升。

**💡 创新点**

证明通过真实企业项目的BPM课程能显著提升未来计算专业人员在AI时代所需的软技能（专业性、沟通、团队合作、决策、主动性等），并提供可复制的课程实施框架，填补了BPM教学与软技能提升关联研究的空白。

**🔧 技术方法**

采用BPMN建模语言和案例工具（bpmn.io）进行建模，结合项目式学习、反馈问卷（Likert量表）进行数据收集，采用项目管理流程和课堂+实践相结合的教学设计。

**📊 数据集**

使用由53名二年级软件工程学生参与的项目，收集45份匿名问卷反馈，包含硬技能与软技能自评数据；未使用外部公开数据集。

**📈 对比分析**

通过问卷的同意/部分同意比例来评估技能提升，未与其他教学方法做定量对比，主要呈现定性/百分比结果，显示大多数学生对硬/软技能的积极评价。

**⚠️ 局限性**

样本量有限且来自单一高校，缺乏对照组或多学科学生，项目时间短导致合作深度受限，学生反馈可能受成绩影响，未进行长期跟踪验证软技能的持续效果。

---

## 456. From Static Bibliometrics to Dynamic Knowledge Graphs: An LLM-Powered Framework for Modernizing Science, Technology, and Innovation (STI) Analytics

**arXiv ID:** 2607.21327 | [PDF](https://arxiv.org/pdf/2607.21327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 457. Grasp, Handover, Rotate: Bimanual Object Reorientation via Compositional Diffusion and Energy-Based Optimization

**arXiv ID:** 2607.21341 | [PDF](https://arxiv.org/pdf/2607.21341v1)

**作者:** Wun Lam Yeung `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于扩散模型与能量模型的双臂协同物体重新定向框架BiCompoDiff；

**💡 创新点**

通过在逆扩散过程中注入能量梯度，实现对抓取、交接、再抓取与放置姿态的联合优化，显著提升轨迹平滑度与成功率；

**🔧 技术方法**

利用GraspGen预训练的扩散抓取模型、可微分的子网络IK、能量模型（碰撞、轨迹平滑、交接与再抓取约束）以及Annealed MCMC采样；

**📊 数据集**

在60个仿真场景（Easy/Medium/Hard）上进行评估，并在两台UR12e+Robotiq双臂机器人上进行实机验证；

**📈 对比分析**

与无能量梯度的基线NoEBM以及改编的ReorientBot对比，BiCompoDiff-Full在成功率上提升约20%，轨迹总关节位移降低约37%，并在大多数指标上优于对比方法；

**⚠️ 局限性**

假设完美感知（已知物体姿态与几何），间接轨迹优化限制了对动态障碍或感知噪声的鲁棒性，未来可探索直接轨迹优化与更强的感知鲁棒性。

---

## 458. Capital Markets LLM Reliability Score (CM-LRS): From Plausible to Bankable

**arXiv ID:** 2607.21340 | [PDF](https://arxiv.org/pdf/2607.21340v1)

**作者:** Prerit Ahuja `[一作]` `[通讯]`, Prerit Ahuja

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了CM‑LRS（Capital Markets LLM Reliability Score）框架，评估LLM在资本市场工作流输出层的可靠性，覆盖事实准确性、证据可追溯性、数值一致性、工作流完整性、源纪律、决策有用性与可审计性七维度。

**💡 创新点**

创新点在于提出面向监管环境的工作流输出可靠性度量，将传统QA级别的评估提升到实际业务输出层，并公开完整的7维度rubric、提示和示例任务，使模型性能可在实际资本市场流程中被验证与比较。

**🔧 技术方法**

技术手段包括LLM‑as‑judge自动评分协议（以Claude Sonnet 4.6为主评审），并辅以三位外部评审（GPT‑5.5、Claude Haiku 4.5、Gemini 2.5 Pro）进行交叉验证，覆盖四款模型（Claude Opus 4.7、GPT‑5.5、Claude Sonnet 4.6、Llama 3.3 70B）进行对比。

**📊 数据集**

使用公开的SEC EDGAR Rule 424(b)(5)补充文件、公开的英国收购IR发布、以及合成北欧式补充文件，构成五个示范工作流：债务条款提取、检索、发行人概况合成、可比交易推理、股权条款提取。

**📈 对比分析**

通过四位评审的交叉评分平均得到CM‑LRS总分；三款前沿封闭源模型聚在4.22‑4.31区间，开放权重模型落后约1.16分；在单文档提取工作流差距仅0.84分，而在检索与合成工作流差距达2.23/2.15分；决策有用性维度在发行人概况工作流呈现最大分差（4.0分），表明其为最具辨别力的可靠性指标。

**⚠️ 局限性**

局限性包括：评审完全自动化（缺少人工评审验证）、工作流样本有限（未覆盖草拟类）、文档预处理导致深层信息未被评估、单轮评估未涵盖代理工具链、均值加权聚合未考虑失误成本差异、模型面板受限、数据集固定、未评估对抗鲁棒性。

---

## 459. Ensemble Logic for Symbolic Representation of Sleep Medicine Guidelines

**arXiv ID:** 2607.21331 | [PDF](https://arxiv.org/pdf/2607.21331v1)

**作者:** Jiahao Fan `[一作]` (University of Texas Health Science Center at Houston), Guo-Qiang Zhang `[通讯]` (University of Texas Health Science Center at Houston)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将美国睡眠医学协会（AASM）手册中的自然语言评分规则转换为可执行的Rational Ensemble Logic（QEL）规范，并通过LLM回译验证其语义完整性。

**💡 创新点**

提出了面向睡眠医学的密集时序逻辑QEL框架，能够将持续时间、先后顺序、存在性等临床描述精确映射为逻辑算子，消除了手册中隐含歧义，并提供了可复现的、可检查的规则实现。

**🔧 技术方法**

使用QEL（密集时序逻辑）作为规范语言，构建Biomedical Event Structure Temporal Model（BEST）作为语义模型，结合大语言模型（OpenAI gpt‑5‑mini‑2025‑08‑07）进行回译评估，计算语义相似度与词汇重叠。

**📊 数据集**

未给出具体公开数据集，研究基于AASM手册描述的规则抽象以及对通用睡眠多导睡眠图（PSG）记录的理论BEST建模；在12条关键评分规范上进行回译实验。

**📈 对比分析**

通过嵌入向量余弦相似度评估回译结果，语义相似度平均79.3（95% CI: 79.0–79.7），词汇重叠仅14.9，显示逻辑规范在保留临床意义方面表现良好；相比传统基于文本的实现，减少了主观差异和实现不一致。

**⚠️ 局限性**

局限性包括：所用QEL片段虽然可判定，但整体逻辑仍不保证完全可判定；缺乏完整睡眠本体（ontology）导致全局一致性难以强制；规则翻译过程中仍存在专业解释差异；实验仅在规则级别验证，缺乏真实PSG数据上的性能评估。

---

## 460. Open Veins of Algorithmic Auditing: Why AI Assessment Lags Behind Its Deployment in the Global South

**arXiv ID:** 2607.21317 | [PDF](https://arxiv.org/pdf/2607.21317v1)

**作者:** Gemma Galdon Clavell `[一作]`, Alexandra Magaard `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

综述并分析了全球南方地区 AI 评估与审计的现状、案例和挑战，并提出了五条学习点与针对基金会的具体建议

**💡 创新点**

强调评估应聚焦影响而非模型，提出了基于当地条件的五条教训，指出资金缺口是导致评估缺失的根本原因，并将 AI 评估与传统影响评估相融合

**🔧 技术方法**

主要使用案例研究、定性分析、基于 Aequitas 等工具的差异影响分析、RAIA 评估框架以及公开的评估方法与技术

**📊 数据集**

基于公开的公共部门算法清单、单个系统的训练与测试数据（如 Robot Laura、儿童福利风险模型、临床记录、农业影像、聊天机器人等）以及相关行政数据与受影响人群数据

**📈 对比分析**

通过对已审计系统的性能指标（ROC、召回率、误报率、群体公平度）与基准比较，发现模型在真实情境下表现不佳、存在代理目标与偏差，表明对比仅在有限案例中有意义

**⚠️ 局限性**

受限于审计案例稀缺、数据获取受限、缺乏监管与资金支持、评估多依赖外部资助以及结果难以公开与推广

---

## 461. Factorized Spatio-Temporal Convolutions for Human Pose Estimation from Planar Lidar

**arXiv ID:** 2607.21309 | [PDF](https://arxiv.org/pdf/2607.21309v1)

**作者:** Simone Arreghini `[一作]` (Dalle Molle Institute for Artificial Intelligence), Alessandro Giusti `[通讯]` (Dalle Molle Institute for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种轻量级的空间–时间块（ST-Block）网络，用单目RGB‑D体姿跟踪器的自监督信号训练，仅利用全景平面激光雷达序列实现全景人类检测与相对二维姿态估计。

**💡 创新点**

创新点在于：①将空间卷积与时间卷积显式分离，在圆形激光雷达数据上使用循环卷积；②使用交叉模态自监督，仅在相机视角重叠区域对雷达预测加损失，从而避免手工标注；③实现CPU实时推理，适配资源受限的服务机器人。

**🔧 技术方法**

技术包括：空间–时间分离卷积（ST-Block）、循环卷积、掩码交叉模态自监督损失、时序平移补偿、轻量化残差与跳跃连接。

**📊 数据集**

数据集为自行收集的TIAGo机器人数据（Corridor、Break Area、Office、Lab等），含RGB‑D、双平面激光雷达和运动捕捉标注；公开FROG数据集用于检测基准。

**📈 对比分析**

与参数匹配的1D全卷积基线相比，ST-Block在80%召回点下提升精度、距离误差降低约29%，位置误差降低约20%，方向误差降低约12%；在FROG基准中，低分辨率模型与轻量基线相近，且CPU推理时延低于GPU方法。

**⚠️ 局限性**

局限性包括：在严重遮挡和多人人同时落在同一条雷达光束时性能下降；仅能估计相对二维姿态，面向方向的判断仍具有固有限制；依赖相机视角重叠区的自监督，若相机与雷达视角差异过大可能影响迁移。

---

## 462. A Diffusion-Model Subpopulation Digital Twin for Mobile Health Deployment: A Case Study on the HeartSteps Intervention

**arXiv ID:** 2607.21403 | [PDF](https://arxiv.org/pdf/2607.21403v1)

**作者:** Ziping Xu `[一作]` (University of North Carolina), Susan A. Murphy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于时间一致性条件扩散模型的 JITAI‑Twin 数字孪生，用于在移动健康干预上线前模拟目标亚群体的行为并评估在线学习算法。

**💡 创新点**

创新点在于：① 将 TimeWeaver 架构改造成严格前向（非预知）时序扩散模型；② 引入推理时刻的校准步骤，仅用预先知识（专家或 LLM 预测的日间步态）即可对新亚群体进行微调；③ 通过两步 LLM 预测流程自动生成校准目标。

**🔧 技术方法**

核心技术包括：时间一致性条件扩散模型（Forward‑only state‑space kernels + action‑injection blocks）、预训练/微调策略、奖励倾斜（reward‑tilted）采样校准、LLM 生成的日间分布预测。

**📊 数据集**

使用的数据集为：① NIH All of Us（1,507 参与者的无干预 FitBit 日间步数）；② HeartSteps v2、v3、v4 的微随机化干预实验（各 59–79 名参与者，持续 6–12 个月）。

**📈 对比分析**

与 KNN、PSRS、线性 SEM 等基准进行比较，评估指标包括日均分布 Wasserstein、短滞后自相关 MAE、活动期跑长 Wasserstein、跨个体异质性比率。结果显示：① 在首次手上（预训练+校准）与 v2/v3 相比，Twin 在跨个体异质性上优于所有基准；② 在第二次手上（微调+校准）Twin 与真实 v4 数据在所有指标上接近或略优于“oracle”微调模型，显著优于简单重采样或结构化模拟器，尤其在跨个体异质性和时间结构方面。

**⚠️ 局限性**

局限性包括：① 仅验证了从数据到模型的单向更新，未在真实部署中检验模型对算法决策的影响；② 校准仅针对均值形状，无法直接处理非线性或因果效应；③ 需依赖专家或 LLM 的先验推断，若预测不准会影响校准；④ 目前仅在 HeartSteps 一个程序内验证，跨领域迁移仍待验证。

---

## 463. VoLN: Vision-Only Long-Horizon Navigation---Paradigm, Benchmark, and Method

**arXiv ID:** 2607.21400 | [PDF](https://arxiv.org/pdf/2607.21400v1)

**作者:** Jiabin Lou `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Vision-Only Long-Horizon Navigation (VoLN) 框架，要求智能体仅通过视觉目标和局部可观测的场景线索完成长距离航拍导航。

**💡 创新点**

创新点在于移除语言指令与全局导航信号，改为仅视觉目标与本地线索的闭环导航；并构建VoLN-UAV基准与VoLN-MLLM两阶段视觉-语义规划方法。

**🔧 技术方法**

采用自监督视觉特征（DINO）与CLIP语义空间对齐，结合Vicuna-7B语言模型与LoRA适配器生成短期航迹和停止决策。

**📊 数据集**

使用VoLN-UAV基准，包含7,210条航线、17个不同环境、主动与被动语义信标，分为训练、验证、测试三组。

**📈 对比分析**

与随机、Seq2Seq-VG、CMA-VG、LAG-VG等视觉目标基线比较，VoLN-MLLM在验证集和测试集上取得最高SR、nDTW、SPL，但整体成功率仍低至7.4%（易）、4.5%（中）和1.8%（难）。

**⚠️ 局限性**

局限在于长程证据整合不足、对线索的识别与选择不够鲁棒、闭环误差累积导致成功率偏低，亟需更精准的感知-规划耦合与长时序决策。

---

## 464. Hilbert Operator for Progressive Encoding (HOPE): A Mathematical Framework for Deconstructing Learned Representations in Deep Networks

**arXiv ID:** 2607.21366 | [PDF](https://arxiv.org/pdf/2607.21366v1)

**作者:** Hossein Mobahi `[一作]`, Peter L. Bartlett `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种以功能空间为基础的无数据无超参模型压缩框架（HOPE），通过将每个神经元视作Hilbert–Schmidt算子，利用BN统计构建高斯后验，进而计算神经元容量并实现剪枝、合并以及宏观块剥离的逐步编码。

**💡 创新点**

核心创新在于：① 将参数空间转换为功能空间，消除规模对称性和参数幅值误导；② 利用最大熵原理对BN统计进行解析推断，得到无数据的连续输入分布；③ 统一低秩投影与合并的理论框架，给出闭式容量与投影误差公式；④ 设计基于率‑失真理论的贪心压缩决策；⑤ 引入DEFT算法，实现对迁移学习中稳定‑可塑性权衡的无数据、无任务依赖的结构化冻结。

**🔧 技术方法**

技术手段包括：无数据最大熵推导BN后验、Hilbert空间内核计算与低秩投影、精确的剪枝/合并闭式公式、宏观块剥离的宏观投影误差、rate‑distortion 平衡的贪心选择、DEFT中的结构化弹性掩码与梯度缩放。

**📊 数据集**

数据集：ImageNet 预训练的 ResNet‑50 用于压缩实验；跨域迁移实验使用 CIFAR‑100 子任务（4 个超类）作为源任务，SVHN 作为目标任务。

**📈 对比分析**

与基线比较：对压缩，HOPE 在保持相同准确率下获得更高压缩率，优于 L1‑输入、L1‑联合、BN‑尺度等传统剪枝；对迁移学习，DEFT 在 H‑Score 上显著优于 Full FT、Head‑Only、PEFT、EWC，源任务保持率 52%，目标任务准确率 94%，H‑Score 65.8。

**⚠️ 局限性**

局限性：① 需要 BN 或一次小批量校准；② 对非 PH‑1 激活的推导有限；③ 宏观块剥离仅适用于残差结构；④ 高斯后验和核近似的理论假设在极端分布下可能失效；⑤ 贪心压缩策略可能陷入局部最优；⑥ 计算复杂度仍受层宽与合并组合数影响。

---

## 465. SPORD: A Simulation-Propose-then-OR-Dispose Approach for Supply Chain Planning

**arXiv ID:** 2607.21354 | [PDF](https://arxiv.org/pdf/2607.21354v1)

**作者:** Jiayin He `[一作]`, Zuo-Jun Max Shen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了基于 SPORD（Simulation-Propose-then-OR-Dispose）框架的供应链规划系统 NetSim，能够一次性解决跨层级、跨业务单元的网络规划（TNP）和库存组合规划（WAP）等大规模、复杂业务场景。

**💡 创新点**

创新点包括：① 将复杂业务规则和约束完全封装在模拟阶段生成的候选路径中，实现可行性评估与优化的解耦；② 开发矩阵向量化、GPU 加速的并行模拟引擎和列表调度算法，使候选路径评估速度提升 10–100 倍；③ 构建闭环智能诊断与增量学习机制，支持实时验证、差异归因和持续改进。

**🔧 技术方法**

技术实现主要依赖 Apache Spark RDD 并行计算、GPU Tensor 运算、RNN 等价库存模拟、列表调度（P|prec|Cmax）算法、整数规划（BIP）以及专家知识驱动的初始化和诊断树。

**📊 数据集**

使用 JD.com 20,000+ 供应商的真实业务数据集，包括 SKU、订单、需求、库存、运输路线、仓储成本等历史运营日志，并通过真实履约记录对模拟精度进行验证。

**📈 对比分析**

与传统串行 for‑loop、pandas.apply 等方法对比，矩阵并行模拟在 50,000 条路径上从 76 s 降至 2.1 s；GPU 加速相较 CPU 提升 10–30 倍；在 NetSim 平台部署后，跨区域履约率由 6.1% 降至 4.9%（20% 改善），WAP 每年节约 73 M 美元，碳减排 5,745 tCO₂e。

**⚠️ 局限性**

局限性：① 需求假设为确定性预测，缺乏对需求波动的鲁棒性；② 方案高度依赖 JD.com 的业务知识库，迁移到其他电商或供应链需要重建专家知识；③ 并行模拟对 GPU/大规模计算资源要求高，成本与运维复杂；④ 目前未与生成式 AI 形成闭环学习，决策深度与自适应能力仍有限。

---

## 466. Quality-Aware Multimodal Fusion Reveals Implicit Identity in Valence-Arousal Features

**arXiv ID:** 2607.21347 | [PDF](https://arxiv.org/pdf/2607.21347v1)

**作者:** Jisu Kim `[一作]` (University of Nebraska-Lincoln), Benjamin S. Riggan `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将音视频情绪（valence‑arousal）估计作为预训练任务，学习出含有身份信息的多模态特征，并验证其可作为软生物识别信号；同时提出质量感知自适应融合(QAAF)提升情绪估计性能。

**💡 创新点**

①提出QAAF，结合软门控(QAG)和质量依赖Dropout(AMD)，实现对每个样本每个模态可靠性的动态估计与正则化；②证明情绪估计训练能隐式编码身份信息，且与传统面部识别互补；③在无身份监督的条件下实现软生物识别。

**🔧 技术方法**

多模态Transformer交叉注意力融合、质量感知软门控、质量依赖Dropout、CCC损失；使用预训练的I3D/ViViT/VideoMAE视觉骨干和ResNet18音频骨干；特征冻结+融合层训练；score‑level与ArcFace融合。

**📊 数据集**

Aff‑wild2用于VA估计与QAAF训练；AFEW‑VA和YTF用于验证软识别性能；Kinetics‑400预训练骨干。

**📈 对比分析**

与JMT基线、能量质量估计QMF、固定Dropout、AMD单独等做对比。QAAF在Aff‑wild2平均CCC达到0.472（比基线0.415提升）。在AFEW‑VA和YTF上软识别EER分别为0.166/0.291，居软生物方法首位；与ArcFace融合后EER分别降至0.021/0.104，纠正68.2%误接受。

**⚠️ 局限性**

依赖骨干差异，VA训练对VideoMAE无明显提升；仅在小规模验证集验证，需评估更大操作环境；对ArcFace改进有限；音频模态更易失效；缺少对极端噪声与模态缺失的全面鲁棒性验证。

---

## 467. SHIFT: Self-reconstruction Harnesses Implicit Fine-grained Thinking for Retrieval

**arXiv ID:** 2607.21333 | [PDF](https://arxiv.org/pdf/2607.21333v1)

**作者:** Yuxiao Luo `[一作]` (Peking University), Wei Ye `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于LLM的隐式推理检索框架，将LLM的推理能力通过残差投影、双向注意力聚合和细粒度自重构等技术转化为高效的检索器。

**💡 创新点**

创新点在于：①通过残差投影去除生成噪声，匹配检索目标；②采用双向注意力聚合动态权衡不同推理步骤；③使用基于下一词预测的自重构来精细化隐式推理空间，避免了传统方法的对齐冗余与监督缺失。

**🔧 技术方法**

技术上结合了因果LLM、soft token生成、两层残差投影、双向多头注意力池化、InfoNCE 对比学习以及基于下一词预测的自重构损失，并采用 LoRA 微调。

**📊 数据集**

主要数据集包括 ReasonEmbed（81K 训练样本，带 GPT‑4o‑mini 生成的推理路径）以及三大评测基准 Bright、FollowIR、BrowseComp‑Plus。

**📈 对比分析**

与稠密检索器（BGE‑M3、E5‑Large‑Instruct、Qwen3‑Embedding）、Rewrite‑then‑Retrieve 管道、显式推理检索器（Search‑R3、GRACE）以及隐式推理检索器（GIRCSE、LaSER）进行对比，实验表明本文方法在 nDCG@10、Recall@5/1000 等指标上均显著优于对手。

**⚠️ 局限性**

局限性包括：仍需依赖外部生成的显式推理路径；固定推理步数和统一映射方式可能不适应不同查询；隐式推理表示缺乏可解释性。

---

## 468. SlerpFlow: Spherical Trajectory Correction for Rectified Flow Inversion

**arXiv ID:** 2607.21326 | [PDF](https://arxiv.org/pdf/2607.21326v1)

**作者:** Wenbin Duan `[一作]` (University of International Relations), Binyang Li `[通讯]` (University of International Relations)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SlerpFlow，一种基于球面线性插值的几何校正解算器，用于改进 rectified-flow 反演与编辑。

**💡 创新点**

创新点在于把速度分解为径向与角向，利用球面插值构造抛物线修正，消除欧氏求解器引起的离心漂移，实现零训练、无需额外模型评估的高精度反演。

**🔧 技术方法**

采用 SlerpFlow（球面插值+弦形更新）与现有 Euler、Heun、RF‑Solver、FireFlow 等解算器对比，结合 FLUX.1-dev 预训练模型实现反演与文本引导编辑。

**📊 数据集**

主要使用 PIE‑Bench 数据集评估重建与编辑质量，并在 FLUX.1-dev 上进行实验。

**📈 对比分析**

与传统解算器相比，SlerpFlow 在 15 步 NFE 下实现了更高的 PSNR/SSIM、更低的 LPIPS，编辑时 CLIP 语义匹配更强，同时保持了背景保真度。

**⚠️ 局限性**

局限性包括对高维潜在空间近似为球面，可能在局部几何高度各向异或平坦时过度校正；编辑时全局角度校正有时会轻微扰动未编辑背景区域。

---

## 469. Toward cryptographically verifiable authorization for autonomous AI agents: A security hypothesis, preliminary formal model, and proof-of-concept implementation

**arXiv ID:** 2607.21325 | [PDF](https://arxiv.org/pdf/2607.21325v1)

**作者:** M. Llambí-Morillas `[一作]` (Universidad Tecnológica Atlántico-Mediterráneo), D. Fernández-Fernández `[通讯]` (Universidad de Santiago de Compostela)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了面向自主 AI 代理的加密可验证授权（CVA）框架，并给出相应的形式化抽象与可执行的零知识证明原型；

**💡 创新点**

创新点在于将授权视为与代理主体、请求、执行上下文和策略满足度绑定的独立加密关系，拆分身份、授权请求与执行绑定的安全属性，并提出一套候选安全性质及研究议程；

**🔧 技术方法**

使用的技术包括 Groth16 zk‑SNARK、Poseidon 哈希/承诺、Circom 电路、bn128 曲线及 FastAPI 验证网关；

**📊 数据集**

文中未使用公开数据集，而是基于简化的计划/动作序列做示例演示；

**📈 对比分析**

性能方面，验证在网关端保持常数时间，证明生成随电路约束数线性增长；但未与其它系统做系统性基准对比；

**⚠️ 局限性**

主要限制包括：缺乏完整安全归约、未进行电路审计、Groth16 需要可信设置且非后量子安全、策略受限于静态电路、未实现上下文绑定与运行时执行绑定、未处理多代理授权链、缺乏实测基准且网关可信性未被考虑。

---

## 470. AI Assistants Overassist

**arXiv ID:** 2607.21306 | [PDF](https://arxiv.org/pdf/2607.21306v1)

**作者:** Verona Teo `[一作]` (Stanford University), Max Kleiman-Weiner `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计 Int-Bench benchmark，对大语言模型在学习场景中的干预行为进行系统评估，研究其干预频率、时机、对任务成功和泛化的影响，并与人类教师进行对比。

**💡 创新点**

创新点在于将干预视为顺序干预游戏，提出一套完整的干预度量（频率、时机、即时帮助和泛化帮助），并构建了基于模拟学生-教师对话的 Int-Bench 框架，首次实现了对 LLM 干预行为与人类干预行为的定量与定性比较。

**🔧 技术方法**

主要使用的大型语言模型包括 Qwen2.5-7B 作为学生，GPT‑5.2、Gemini 3 Flash、GPT‑OSS‑120B、DeepSeek‑V3.2 作为教师，以及 GPT‑5.2 作为判断器；通过序列监控与策略映射（π）生成干预；使用结构化生成管线为每个问题生成相关变体，形成泛化测试。

**📊 数据集**

数据集包含 500 道 DebugEval 代码调试题、500 道 MATH‑500 数学题、500 道 Braingle 逻辑谜题，总计 1,500 个问题；此外抽取 150 道每域问题生成 300 个学生-教师-泛化三元组用于泛化评估。

**📈 对比分析**

比较方法包括：①标准监控（逐步揭示 50 字符）与 Oracle 监控（一次性获取完整轨迹）两种教师决策模式；②与 50 名人类教师在相同设置下的干预频率、时机和帮助率进行对比；结果显示 LLM 在标准条件下干预频率高达 90%，时机偏早（τ_rel≈0.18），但即时帮助率仅为 0.20，泛化帮助几乎无提升；相比之下人类干预更少、更晚，且更倾向于提供提示，帮助率相对更稳健。

**⚠️ 局限性**

局限性包括：①使用模拟学生模型，难以充分捕捉真实学习者的认知负荷、动机与情绪；②泛化评估仅针对单个相关问题，未能测量长期学习或多次练习的影响；③实验结果受模型规模、提示语和干预策略的影响，缺乏对不同模型和提示的系统性探究。

---

## 471. Beyond Degree Four: Near-Orthogonal Planar Drawings

**arXiv ID:** 2607.21305 | [PDF](https://arxiv.org/pdf/2607.21305v1)

**作者:** Patrizio Angelini `[一作]` (John Cabot University), Ignaz Rutter `[通讯]` (University of Passau)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在平面图中绘制图形时最小化非正交面数的优化问题。

**💡 创新点**

创新点在于证明该问题在3‑连通图上为NP‑完整，并针对固定嵌入提供了多参数FPT算法、线性时间特例以及PTAS；在不固定嵌入的情况下给出了树宽参数化的FPT算法。

**🔧 技术方法**

主要技术包括球切分分解（sphere‑cut decomposition）与动态规划、Baker技术拆分外层化以及对SPQR树的动态规划处理R‑节点，结合残留度量与面集合描述符。

**📊 数据集**

论文未使用公开数据集，而是基于理论构造的图实例与分解算法进行实验与分析。

**📈 对比分析**

通过与已知的图绘制基准算法（如正交绘图、八方向绘图）对比，FPT算法在树宽较小或外层数有限时能在多项式或指数时间内得到最优解；PTAS在给定ε时的运行时间为O(2^{O(1/ε)}·n)，相较于传统全局搜索方法大幅提升。

**⚠️ 局限性**

局限性包括：问题仍然NP‑完整，缺乏多项式时间近似算法；FPT算法的指数因子高且仅适用于特定参数；变量嵌入情形仅对双连通图给出结果，尚未推广到一般连通图；未给出多项式核化或亚指数FPT算法的实现。

---

## 472. DISCO: Distributed Spectrum Compliance and Orchestration for Scalable IoT Coexistence

**arXiv ID:** 2607.21387 | [PDF](https://arxiv.org/pdf/2607.21387v1)

**作者:** Lyes Saad Saoud `[一作]`, Moussa Ayyash `[通讯]` (Chicago State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 DISCO 体系架构，分离本地无线学习、边缘合规监管与云/非地面网络上下文适配，构建统一的合规信号与可扩展的治理面。

**💡 创新点**

创新点在于三时尺度的分层设计：①本地学习保持算法无关；②边缘节点聚合违例统计并发布压缩治理信号；③云层根据上下文动态调整风险阈值；同时提供可复用接口与与现有硬规则共存的合规机制。

**🔧 技术方法**

采用分布式强化学习/多代理学习、边缘统计聚合、安全屏蔽阈值、概率治理信号、以及上下文感知的目标自适应；技术上结合 O‑RAN 接口与多边缘分布。

**📊 数据集**

使用 30 个随机种子仿真数据集：1000 × 1000 m 城市微小区、8/12 个主用户、6 个信道、UAV 代理，模拟正常与降级非地面网络（NTN）情境。

**📈 对比分析**

与固定功率基准和无边缘反馈的 RL 方案对比，DISCO 在 nominal 模式下吞吐量提升至 81 Mbps（比固定功率提升约 73%），违例率降至 0.053（比无边缘低 2.4 倍），在压力模式下吞吐下降至 71.4 Mbps，违例率升至 0.082。

**⚠️ 局限性**

局限性：仿真器、信道模型、阈值定义和更新参数未公开，缺乏实测验证；仅报告平均违例率，未给出尾部/峰值指标；无法保证法律合规，仅实现统计趋近目标；需进一步基线对比、复杂度测量与安全可信性评估。

---

## 473. FedAgentKE: Federated Semantic Knowledge Evolution for Heterogeneous Agents

**arXiv ID:** 2607.21361 | [PDF](https://arxiv.org/pdf/2607.21361v1)

**作者:** Weihao Li `[一作]` (Northwestern University), Ziyang Song `[通讯]` (Ohio University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FedAgentKE 框架，实现异构 LLM 代理在不共享原始执行轨迹的前提下，进行语义知识蒸馏、聚合与适配，从而实现跨框架、跨任务的协同演进。

**💡 创新点**

①将代理协作重新表述为联邦语义知识演进问题；②在语义层面而非参数层面进行知识同步，克服了传统联邦学习在异构代理中的适配瓶颈；③引入基于实用性与可迁移性的代表性知识选择机制，支持高效且可扩展的多框架知识共享。

**🔧 技术方法**

使用 LLM 语义蒸馏模块对执行轨迹进行抽象，采用语义嵌入+聚类实现跨任务知识聚合，利用实用性与迁移性评分挑选代表知识，并在客户端执行知识适配映射与推理细化；整个流程基于 Federated Learning 的同步架构。

**📊 数据集**

在 GAIA（多步骤推理与工具使用）和 SWE-bench Lite（GitHub issue 解决）两个基准数据集上进行实验，覆盖 Level‑1~3 以及 300 条真实任务案例。

**📈 对比分析**

与单体代理做对比，在 SWE-bench Lite 上单框架从 27% 提升到 42%（≈15%）或 29%→44%（≈15%）；在 GAIA 上单框架平均提升 13~19%；跨框架联邦从 49% 提升至 54%（约5%）；随着联邦轮数增加，成功率持续上升，R=5 时可达 74%。

**⚠️ 局限性**

目前未考虑通信效率与隐私保护，语义嵌入机制可能无法完整捕捉复杂推理依赖；实验规模仅限于四种代理框架与两大基准，尚未在更大异构生态中验证可扩展性。

---

## 474. Emergent Misalignment Recruits a Pre-existing Persona Subspace

**arXiv ID:** 2607.21356 | [PDF](https://arxiv.org/pdf/2607.21356v1)

**作者:** Mohammed Suhail B Nadaf `[一作]` `[通讯]` (Independent), Mohammed Suhail B Nadaf (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在已对齐的语言模型上使用极为狭窄的负面数据（如不安全代码或危险建议）进行微调，探究并揭示微调如何导致模型在与训练无关的问题上出现广泛偏差，并通过对冻结模型提取低秩人格子空间来分析这一现象的根源；

**💡 创新点**

创新点在于发现并验证了模型在微调前就已存在的跨领域共享低秩人格子空间，它是导致广泛偏差的因果因素；通过对该子空间进行投影、注入和梯度约束等干预，首次实现了对偏差的完全抑制或诱导，并揭示了在多领域扩散固定量负面数据时偏差呈超加性增大；

**🔧 技术方法**

主要技术包括对比教师强制提取人格子空间、激活与权重梯度投影、写入核心约束、输入框架注射、基于教师强制的对数概率边际测量，以及对齐/连贯性评估的双轴判定；

**📊 数据集**

使用的数据集为Qwen2.5-14B-Instruct预训练并对齐的检查点，微调集包括“insecure‑code”和“educational‑code”两套对照数据，以及公开的三套狭窄偏差模型（医学、金融、极限运动），并在这些数据上进行实验；

**📈 对比分析**

通过与匹配随机子空间或随机向量的对照实验，对比教师强制边际值和判定率，结果显示：投影出激活子空间可将广泛偏差从27.7%降至0%，注入子空间可按剂量线性提升偏差；首次梯度步骤已显现意图差异，预测后续训练结果；在判定率上微调的负面数据在某些指标上提升约1–8%，但无法区分不安全与教育框架；

**⚠️ 局限性**

局限性包括：仅在单一模型/规模（Qwen2.5‑14B‑Instruct + LoRA）上验证，无法确认该子空间是预训练或对齐后产生的；难以区分消除偏差与削弱表达能力的影响；仅在一种基底下进行后置权重编辑，未覆盖所有可能基底；判定率在此规模下偏差率较低，导致判定与教师强制边际不完全一致；

---

## 475. How Many Bits Can an Adapter Write? Measuring the Capacity and Memorization of Parameter-Efficient Fine-Tuning

**arXiv ID:** 2607.21351 | [PDF](https://arxiv.org/pdf/2607.21351v1)

**作者:** Kaizhen Tan `[一作]` (Carnegie Mellon University), Yang Feng `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对冻结基础模型，使用基于压缩的记忆度量，量化 LoRA 适配器写入的位数，并探究不同参数放置、精度和底层模型结构对记忆容量的影响。

**💡 创新点**

提出了按位数而非参数数量来评估适配器记忆容量的新度量方法，并首次将该方法用于对比监督微调与可验证奖励强化学习在记忆与隐私泄漏方面的差异。

**🔧 技术方法**

利用算术编码压缩、LoRA/TinyLoRA 低秩适配器、Qwen2.5-0.5B 微调、随机字符串校准、成员推断与可卡提取攻击等技术。

**📊 数据集**

使用 WikiText-103 预训练、64 令牌随机序列做校准、合成数据中的秘密植入、3 位算术和键值提取任务等数据集进行实验。

**📈 对比分析**

通过对不同秩、放置位置、精度下的记忆容量进行对比，发现每参数位数在 1.7–2.8 之间；在相同准确率下，RLVR 适配器几乎不写入位，而 SFT 适配器写入大量位并导致明显泄漏；位数与可卡提取成功率高度相关。

**⚠️ 局限性**

测量仅为下限，可能低估实际容量；校准使用随机字符串可能无法覆盖所有记忆形式；实验仅覆盖中小规模模型，结果对更大模型的泛化仍待验证；隐私攻击受数据集规模限制，未能检验更大规模下的成员推断表现。

---

## 476. PC-Edit: Prompt-Contrastive Region Discovery and Region-Guided Editing

**arXiv ID:** 2607.21318 | [PDF](https://arxiv.org/pdf/2607.21318v1)

**作者:** Jian Zhang `[一作]` (South China University of Technology), Zhijun Zhang `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的 MM‑DiT 编辑框架 PC‑Edit，能够在无用户掩码的情况下实现单/多物体增删与替换，自动定位编辑区域并保持背景不变。

**💡 创新点**

创新点包括：① 通过在同一潜在状态下对源/目标提示的图像‑token 注意力输出做差（ΔAttnOut）直接获得空间对比信号；② 在逆向过程中提取源消除区域，在去噪过程中追踪目标出现区域，二者的并集定义编辑区域；③ 在每一步采样中根据最新编辑区域立即注入缓存的源 K/V，既保证目标自然生成，又防止背景漂移。

**🔧 技术方法**

核心技术：Prompt‑Contrastive 关注差分（ΔAttnOut）、Otsu 阈值化、同步 K/V 注入、warm‑up 全局注入、FLUX.1‑dev 变换器网络。

**📊 数据集**

使用两大基准：① EditRegion‑Bench（484 条单/多物体增删/替换案例，人工标注编辑区域）；② PIE‑Bench（240 条增删/替换案例）进行评估。

**📈 对比分析**

与七种 mask‑free 训练‑free 方法（PnPInversion、MasaCtrl、RF‑Inversion、Stable Flow、FlowEdit、FYS、RF‑Edit）以及两种 mask‑guided 参考（FLUX.1‑Fill、KV‑Edit）对比。PC‑Edit 在 EditRegion‑Bench 和 PIE‑Bench 上在编辑区域定位（AP/FBC/mIoU）、编辑质量（AS/CLIP_sim/PSNR/LPIPS）以及背景保持（bg_p99）等指标上均优于所有 mask‑free 方法，并在无用户掩码下逼近 mask‑guided 参考的表现；在源物体消除率（ESER）上也取得最高分。

**⚠️ 局限性**

局限性：① 仍依赖冻结的预训练 Diffusion Transformer，跨模型迁移需进一步验证；② 对极端形状差异或非常小物体的精细定位仍存在一定误差；③ 计算成本相对较高（需双 forward 计算 ΔAttnOut），在实时或大规模场景下的效率尚待优化。

---

## 477. Scaling Up Formal Representation of Clinical Trial Protocols in Ensemble Logic Using LLMs: A Preliminary Study

**arXiv ID:** 2607.21307 | [PDF](https://arxiv.org/pdf/2607.21307v1)

**作者:** Yan Huang `[一作]` (University of Texas Health Science Center at Houston), Guo-Qiang Zhang `[通讯]` (University of Texas Health Science Center at Houston)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了 CT‑TEL 工作流，利用大语言模型将临床试验协议文本自动转换为 Temporal Ensemble Logic (TEL) 形式化表示，并在 23 例阿尔茨海默症药物试验上进行验证。

**💡 创新点**

创新点在于：①首次将 LLM 与 TEL 结合实现协议的自动编码；②采用双向翻译（文本→TEL→文本）进行语义保真度评估；③构建了多层次可追溯的 CT‑TEL 库，为大规模协议表征提供可复制框架。

**🔧 技术方法**

技术手段包括：大语言模型（Gemini 3.1 Pro、Claude Sonnet 4.6 等）在结构化提示下完成实体映射、事件抽取、逻辑公式生成与逆向翻译；以及基于 TF‑IDF 和嵌入向量的词法/语义相似度评估。

**📊 数据集**

使用的数据集为 23 例从 ClinicalTrials.gov 下载的阿尔茨海默症相关药物试验协议（共 6 个模块），包含原始 JSON 记录、实体映射、事件列表和 TEL 公式。

**📈 对比分析**

对比方法包括：人工专家评审、LLM 评分、词法相似度、语义相似度以及跨试验相似度矩阵。结果显示语义相似度平均约 0.79，词法相似度约 0.46；相对基线显著提升，证明译码保持了协议的核心信息。

**⚠️ 局限性**

局限性包括：高度依赖 LLM 的推理能力，存在时间约束失真与隐含假设注入的错误；逆向评估主要基于表面相似度，未实现逻辑等价或模型检验；以及对不同 LLM 兼容性的进一步验证仍需深入。

---

## 478. DINOde: Continuous Vision-Text Alignment for Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2607.21371 | [PDF](https://arxiv.org/pdf/2607.21371v1)

**作者:** Sung-Hoon Yoon `[一作]` (DGIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种基于连续ODE的视文字对齐框架DINOde，利用神经ODE将CLIP文本嵌入逐步映射到DINOv3视觉空间，从而实现开放词汇语义分割。

**💡 创新点**

创新点包括：①使用连续ODE轨迹代替传统单步MLP投影，实现平滑且可逆的跨模态对齐；②引入Velocity Tangent Projection，在球面几何上约束速度场，保持特征空间拓扑结构；③同时对文本流（Semantic Text Flow）和全局上下文（Global Context Flow）进行分层流式对齐，提升语义一致性。

**🔧 技术方法**

使用技术包括：神经ODE（连续映射）+时间条件sin嵌入；CLIP文本编码器+DINOv3视觉编码器；top‑K池化、ℓ₂正则化；Velocity Tangent Projection；对比损失（CLIP‑style symmetric NCE）；Euler数值积分；Mask Refinement后处理。

**📊 数据集**

训练使用COCO 2017 Caption（约118k图像）对模型进行对齐；评测基于八个开放词汇语义分割基准：Pascal VOC 2012（V21/V20）、Pascal Context（C60/C59）、COCO Object、COCO Stuff、Cityscapes、ADE20K。

**📈 对比分析**

在所有基准上与多种现有方法（MaskCLIP、SCLIP、ClearCLIP、NACLIP、dino.txt、FreeDA、ProxyCLIP、CLIPer、Talk2DINO等）进行对比。DINOde在大多数数据集上均超过SOTA，平均mIoU达到49.5%（未Mask Refinement）/50.1%（已Mask Refinement）。

**⚠️ 局限性**

局限性包括：①需要预训练的视觉和文本模型，无法完全自监督；②连续ODE积分步骤对计算开销有一定影响；③对极少量训练数据或与预训练文本分布偏差的数据集的泛化能力尚待进一步验证；④对更复杂、长文本或多义词的鲁棒性未充分测试。

---

## 479. A Needs Assessment for Measuring Geographic - Legislative Associations in the U.S. House of Representatives

**arXiv ID:** 2607.21502 | [PDF](https://arxiv.org/pdf/2607.21502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 480. ElasticTTT: Prior-Preserving Test-Time Tuning for Video Editing

**arXiv ID:** 2607.21529 | [PDF](https://arxiv.org/pdf/2607.21529v1)

**作者:** Yueyi Liu `[一作]` (Tsinghua University), Miao Liu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ElasticTTT 框架解决视频编辑中的 prior collapse 问题。

**💡 创新点**

创新点是三项技术：Target Distribution Regularization、Contrastive Classifier-Free Guidance、Asynchronous Noise Scheduling。

**🔧 技术方法**

使用 Diffusion 模型（DiT）为基础，结合 Test‑Time Tuning、LoRA、CFG、噪声调度等技术。

**📊 数据集**

评估数据集为 125 对文本‑视频样本，基于 Davis 选取，使用 VBench、VEBench、CLIP、VLM GPT‑5 等评测。

**📈 对比分析**

与 AnyV2V、Ground‑A‑Video、Token‑flow、Ditto、Flow‑align、UniEdit、Tune‑A‑Video、VidTTA 等基线对比，ElasticTTT 在 VLM 评估、VBench、VEBench 上均实现 SOTA，整体得分提升 2+。

**⚠️ 局限性**

局限性包括对大模型的来源视频完整度略有下降，且对长视频或复杂场景仍可能出现轻微失真。

---

## 481. Same Dangerous Objective, Opposite Advice: Direct Exposure versus Multi-Agent Mediation

**arXiv ID:** 2607.21518 | [PDF](https://arxiv.org/pdf/2607.21518v1)

**作者:** Linjun Li `[一作]` `[通讯]` (University of Pennsylvania), Linjun Li (University of Pennsylvania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了直接让大型语言模型（LLM）面对隐蔽的危险目标与通过多阶段（Id‑Censor‑Superego）流水线重写后再让LLM回应的两种情况，检验隐藏目标在用户端是否仍能影响模型输出。

**💡 创新点**

首次揭示了即使LLM在直接暴露目标时表现为抵制，经过中介重写后仍能将目标方向传递给最终输出，表明多阶段系统存在“行为逆转”和“可视性缺口”，这在安全评估中是前所未有的发现。

**🔧 技术方法**

采用 OpenAI 的 GPT‑4（模型别名）作为核心推理模型，利用 Prompt‑Engineering 设计了 Id、Censor、Superego 三个子模型，并通过结构化中介消息实现目标重写与约束，结合多阶段 API 调用进行实验。

**📊 数据集**

使用 25 对模拟的双选决策任务（每对任务分别设有一个隐蔽目标），以及 25 个等价选项对照任务，共 1000 次模型调用，所有任务均为人工构造的合成情景。

**📈 对比分析**

通过全流程对比直接路径（单次调用）与中介路径（三次调用）的净目标一致性得分，发现中介路径在目标推荐率上提升 0.352（即 35.2% 的正向转化），同时显著减少了反目标推荐并略微增加了无决定回复，显示出显著的行为方向性变化。

**⚠️ 局限性**

局限性包括：仅使用单一模型别名和一次性实验，数据为合成任务缺乏现实场景验证；直接与中介路径在调用次数、提示设计、上下文长度等多方面不匹配，无法单独归因；语义编码未实现盲目化；缺乏对模型内部机制的解释，实验结果不一定能推广到其他模型或更复杂任务。

---

## 482. Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems

**arXiv ID:** 2607.21503 | [PDF](https://arxiv.org/pdf/2607.21503v1)

**作者:** Gaurav Dadhich `[一作]` `[通讯]` (Maximem), Gaurav Dadhich (Maximem)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“Agentic Context Management”（ACM）框架，将对话式代理的上下文生命周期拆分为五个原语：架构设计（architecting）、摄取（ingesting）、范围限定（scoping）、预期预取（anticipating）以及压缩与整合（compact & consolidation），并通过这一生命周期来主动管理何时保留、如何结构化、哪些数据放入模型上下文、何时丢弃等问题，避免传统“仅存储与检索”方法导致的记忆碎片、上下文旋转和成本爆炸；实现了面向多租户、跨组织层级的多功能上下文管理服务 Maximem Synap。

**💡 创新点**

创新点在于将上下文管理视为完整的生命周期而非单一存储；提出了五个互相耦合的原语，并把组织层级（用户→客户→平台）视为首要维度；引入预期预取与验证压缩机制，既保证线性成本又维持信息完整；在同一系统中实现了动态架构化、异步摄取、图+向量混合检索与可验证压缩。

**🔧 技术方法**

核心技术包括：1）多层次存储体系（向量库、图数据库、关系数据库、对象存储、时间序列存储）；2）按原语划分的异步管道与流式 SDK；3）按 agent 需求自动生成的内存架构；4）实体解析与统一标识（canonicalization）；5）预期预取预测模型与低延迟检索；6）压缩后验证机制。

**📊 数据集**

主要使用公开长记忆基准数据集：LongMemEval（500 问答，6 个类别）和 LoCoMo（约 300 回合，4 个非对抗类别）。

**📈 对比分析**

评估方法：在相同的配置下使用 gpt‑5‑mini 作为回答模型与判别模型，采用 scope‑aware 检索与验证压缩；在 LongMemEval 上取得 92.0%（460/500）准确率，在 LoCoMo（类别1‑4）上取得 93.2%；相较于同类系统（SuperMemory、Zep、Zep/Graphiti 等）在相同评测框架下表现更优。

**⚠️ 局限性**

局限性包括：1）仅在公开基准上评测，未覆盖延迟、token 成本与上下文旋转等生产级指标；2）对决策级与组织级更深层次上下文的处理仍未实现；3）压缩验证机制存在计算开销，需进一步优化；4）基准方法与参数对结果影响大，缺乏统一的跨系统可比性。

---

## 483. Artificial Epanorthosis: Why large language models overuse a classical rhetorical figure, and how to mitigate it

**arXiv ID:** 2607.21498 | [PDF](https://arxiv.org/pdf/2607.21498v1)

**作者:** Federico Boggia `[一作]` `[通讯]`, Federico Boggia

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析并量化大型语言模型中 epanorthosis（自我纠正）过度使用的现象，构造 Epanorthosis Index 来衡量不同体裁的校准偏差；

**💡 创新点**

首次将古典修辞与现代 LLM 生成风格相结合，并提出可调节的 LoRA 适配器、RLHF 调整、解码控制等多种可逆方法来校准其使用率；

**🔧 技术方法**

使用 LoRA 低秩适配、Direct Preference Optimisation、RLHF、PPLM/GeDi/FUDGE 等解码时控制技术、激活方向控制以及提示工程等技术；

**📊 数据集**

基于公开人类写作语料（议论文、演说、新闻、小说、问答、百科等）与模型生成文本进行对照，并在意大利语实验中验证其效果；

**📈 对比分析**

通过 Epanorthosis Index 与人类基线对照，发现模型在演说体裁过度使用、在问答体裁不足；LoRA 适配器可将密度调至人类水平，提示+重写进一步降低；实验表明可在不牺牲内容的前提下实现校准；

**⚠️ 局限性**

局限包括样本量小、不同语料年代差异、检测器仅覆盖部分 epanorthosis 形式、仅在单一模型家族实验、未对内容保持度量、对跨语言泛化缺乏验证。

---

## 484. Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation

**arXiv ID:** 2607.21485 | [PDF](https://arxiv.org/pdf/2607.21485v1)

**作者:** Hyunmin Cho `[一作]` (Korea University), Kyong Hwan Jin `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种权重共享的正弦递归网络，用于隐式神经表示（INR），通过有限步迭代在不增加独立参数的情况下丰富频谱；

**💡 创新点**

利用正弦激活诱导的谐波线谱理论，证明递归迭代能扩展特征频谱，并结合二进制灰度编码的对齐监督实现精确量化重建；

**🔧 技术方法**

正弦递归块、可学习频率嵌入、灰度码映射、余弦对齐损失、卷积/MLP输出投影；

**📊 数据集**

二维图像数据集(Set5、Kodak24、DIV2K、FFHQ)、超分辨率数据集(Set5/Set14/B100)、NeRF数据集(LLFF)、SDF数据集(Stanford Armadillo)；

**📈 对比分析**

与SIREN、学习频率基准、傅里叶网络等基线比较，实验显示在相同参数/迭代预算下PSNR、LPIPS、SSIM均优于对手，且在少量迭代内即可达到高质量重建；

**⚠️ 局限性**

在极低容量模型下提升效果不明显，性能提升受模型规模限制，需进一步研究模型尺寸对递归效能的影响。

---

## 485. CLUIE: Clustering-Aware Recurrent Propagation with Local Structural Compensation for Underwater Image Enhancement

**arXiv ID:** 2607.21467 | [PDF](https://arxiv.org/pdf/2607.21467v1)

**作者:** Kui Jiang `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CLUIE框架，通过聚类感知的递归路径重排（CSDR）和暗响应调制局部传播（DMLP）实现水下图像增强，解决传统固定扫描无法适应空间异质退化的问题。

**💡 创新点**

创新点在于：①基于特征空间聚类的内容感知递归轨迹重排，使RWKV状态沿语义相关区域传播；②引入深度可分离卷积提取局部结构并用伪暗响应调制，补偿重排导致的局部连续性损失；③在保持线性时间复杂度的同时实现长程依赖与局部细节兼顾。

**🔧 技术方法**

使用技术包括RWKV（Receptance Weighted Key-Value）递归网络、k-means聚类、深度可分离卷积、伪暗响应统计、Encoder‑Decoder结构和多层窗口大小的局部传播。

**📊 数据集**

训练数据为UIEB（800对）+LSUI（3879对）共4679对；测试在UIEB（90张）、LSUI（400张）、EUVP（481张）三大有参考基准；无参考评估使用C60、Seathru、UCCS三套数据。

**📈 对比分析**

通过与CNN、Transformer、扩散、状态空间等多种基线在UIEB、LSUI、EUVP上的PSNR/SSIM/ MSE对比，CLUIE在UIEB上达到25.53 dB/0.921/67.87 MSE，EUVP上30.74 dB/0.903/35.09 MSE，表现最优；无参考指标UCIQE、UIQM、MUSIQ、NIMA均位列前列；参数仅4.39 M，FLOPs 14.26 G，效率优于多数基线。

**⚠️ 局限性**

局限性包括：①依赖硬聚类，聚类不稳定时可能影响重排效果；②伪暗响应仅为特征统计，缺乏物理可解释性；③未实现可微的路径学习，缺少自适应的物理退化引导。

---

## 486. KroQuant: Kronecker-Structured Block Transforms for Efficient Post-Training Quantization of Diffusion Transformers

**arXiv ID:** 2607.21446 | [PDF](https://arxiv.org/pdf/2607.21446v1)

**作者:** Yann Bouquet `[一作]` (EPFL), Mathieu Salzmann `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对扩散变换器（DiT）在 4‑bit 量化时激活值异常大导致的质量下降，提出了 KroQuant 方案，通过学习的 Kronecker 结构化块变换预处理激活，从而降低量化误差并保持模型效果。

**💡 创新点**

创新点在于：①将 32×32 的块变换参数化为 5 个 2×2 LU 因子（共 15 个可学习参数），既显著压缩参数量（比全矩阵低 68 倍），又保持了可逆性；②利用块对角结构实现在线、张量核心友好的 GEMM，既比 SmoothQuant 更表达力，又比全尺寸旋转更高效；③将该变换无缝嵌入 LoRaQ 的低秩权重校准流程，形成完整 PTQ 方案。

**🔧 技术方法**

使用技术包括：Kronecker 乘积变换、单位行列式 LU 参数化、直通估计（STE）进行梯度传播、MXFP4e2 量化格式、Tensor‑Core 并行 GEMM（K1、K2 内核）以及 LoRaQ 的低秩权重分解。

**📊 数据集**

主要使用的数据集有 PixArt‑Σ、SANA、FLUX.1‑schnell 三个扩散变换器模型，量化评估基于 MJHQ‑30K 与 SDCI 两个 5,000 张图像的数据集，校准集使用 128 条 COCO 2017 验证集的文本提示。

**📈 对比分析**

与 SmoothQuant、SVDQuant、LoRaQ 等基线对比，KroQuant 在 W4A4 量化下取得了更低的 FID、LPIPS 与更高的 PSNR，且在 MI350 GPU 上的 Kernel 延迟比 SmoothQuant 快约 14%，比 FWHT 快 1.4–2.5 倍；总体上在三大模型和两大评测数据集上均保持或提升了视觉质量。

**⚠️ 局限性**

局限性包括：Kronecker 构造的参数仅随 log₂n 增长，难以在更大块尺寸下达到 GL(n) 的完整表达力；单层学习不稳定，跨层联合优化尚未成功；以及量化后模型在对比度上可能略高于 FP16 基准。

---

## 487. Agent-Guided Relational Concept Discovery: Toward Interpretable Surgical Margin Assessment

**arXiv ID:** 2607.21437 | [PDF](https://arxiv.org/pdf/2607.21437v1)

**作者:** Nooshin Maghsoodi `[一作]` (Queen's University), Parvin Mousavi `[通讯]` (Queen's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了基于推理代理的无监督概念发现框架，用于REIMS数据的手术切缘评估。

**💡 创新点**

结合推理代理自动发现并解释概念，并通过生物化学知识图谱对概念进行对齐，消除了对概念标注的依赖。

**🔧 技术方法**

采用冻结的DreaMS基础模型提取特征、概念瓶颈层、推理代理、知识图谱推理以及对齐损失。

**📊 数据集**

使用皮肤基底细胞癌和乳腺癌的 ex vivo REIMS 数据，以及一次乳腺保留手术的 intraoperative REIMS 数据。

**📈 对比分析**

与基线 DreaMS/Transformer 进行对比，平均 30 次实验，方法在两数据集上显著提升平衡准确率、敏感度和 AUROC；在手术内数据上误报率更低。

**⚠️ 局限性**

依赖通用代谢数据库查询，缺乏针对性基因水平信息；知识图谱构建受限于现有数据库。

---

## 488. Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence in Chain-of-Thought Models

**arXiv ID:** 2607.21433 | [PDF](https://arxiv.org/pdf/2607.21433v1)

**作者:** Renuka Oladri `[一作]` (University of Maryland), Abdirisak Mohamed `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对DeepSeek‑R1‑Distill‑Qwen‑7B在GSM8K、MATH‑500与AIME基准上进行链式推理预算强制和内部激活预测实验，研究推理长度与准确率的关系，并探索早期退出机制。

**💡 创新点**

发现GSM8K和MATH‑500在256个思考标记即可饱和，AIME出现二模态收敛失效，并证明内部激活在150标记处就能预测收敛，层20携带最大收敛信息。

**🔧 技术方法**

使用预算强制（logits processor）、线性探测器、前向钩子、AUC评估、Bootstrap CI、行为基线对比以及数据清洗与后期检验等技术。

**📊 数据集**

使用GSM8K、MATH‑500、AIME 1983‑2024以及后期检验的AIME 2025问题集。

**📈 对比分析**

通过在不同预算、不同检查点对比AUC与ΔAUC衡量探测器性能，激活探测器在50–300标记位置平均ΔAUC≈+0.035，层20的AUC最高≈0.608；行为基线接近随机（AUC≈0.55）。

**⚠️ 局限性**

局限包括仅使用单一模型、样本量有限、使用贪婪解码导致循环、未验证不同温度或模型的泛化、未进行因果验证等。

---

## 489. Semantic-Aware Task Clustering for Constructive and Cooperative Multi-Tasking

**arXiv ID:** 2607.21426 | [PDF](https://arxiv.org/pdf/2607.21426v1)

**作者:** Ahmad Halimi Razlighi `[一作]` (University of Bremen), Armin Dekorsy `[通讯]` (University of Bremen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种Clustered-CMT-SemCom框架，通过一次性语义感知任务聚类，保证多任务协作中的正向合作。

**💡 创新点**

创新点在于利用UMAP降维+HDBSCAN对共享CU输出的语义特征进行聚类，并在此基础上实施子CU级联学习，从而避免负迁移。

**🔧 技术方法**

采用了InfoMax变分推断、CNN编码器/解码器、UMAP、HDBSCAN、AWGN通道模拟等技术。

**📊 数据集**

实验使用MNIST、EMNIST（两类字母）和Fashion‑MNIST（两类服饰）共三组场景的数据集。

**📈 对比分析**

与单任务训练和未聚类多任务训练对比，Clustered-CMT-SemCom在所有场景下错误率显著降低，且在负迁移情形下优于未聚类方案。

**⚠️ 局限性**

局限性包括聚类仅在训练初期完成一次，无法动态适应任务变化；聚类过程对UMAP+HDBSCAN的依赖导致额外计算；当任务间语义相似性不明显时聚类效果有限。

---

## 490. An Evaluation Framework for Structured Audio Captions Validated by Controlled Perturbations

**arXiv ID:** 2607.21424 | [PDF](https://arxiv.org/pdf/2607.21424v1)

**作者:** Liang-Yuan Wu `[一作]` (New York University), Magdalena Fuentes `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多维度评估框架，用于评估结构化音频描述，并通过对 AudioCards 数据集进行系统扰动测试验证其可靠性。

**💡 创新点**

提出了五个评价轴（标签集、描述、推理、数值测量、频谱）以及统一的评分方法，并设计可控扰动实验以校验指标灵敏度。

**🔧 技术方法**

使用大型语言模型（LLM judge）与确定性计算指标（MAE、F1、IoU、ordinal score）结合，采用 σ 归一化等技术。

**📊 数据集**

基准数据集为 AudioCards（已补充 LUFS、pitch、时间边界和十带频谱），共 499 条音频。

**📈 对比分析**

与传统匹配指标（BLEU、ROUGE、METEOR、FENSE、set‑F1 等）对比，LLM judge 在保持语义相同的情况下更不易被误判，且在各种扰动下呈现严格的性能下降；数值和频谱评分线性衰减。

**⚠️ 局限性**

对 LLM judge 的依赖导致计算成本高，小规模模型表现不稳定；框架主要在合成扰动上验证，缺乏对真实生成输出的广泛评估。

---

## 491. GLAM-SLAM: Real-time Gaussian Large-scale Mapping via Flow Densification and Spatial Decomposition

**arXiv ID:** 2607.21416 | [PDF](https://arxiv.org/pdf/2607.21416v1)

**作者:** Panagiotis Mermigkas `[一作]` (Athena Research Center), Petros Maragos `[通讯]` (Athena Research Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了GLAM‑SLAM，一种基于可分离高斯光栅化的实时单目SLAM系统，专为大规模户外长序列设计。

**💡 创新点**

创新点包括：① 用光流+极线几何实现稀疏追踪到密集点云的流引导致密化；② 将地图拆分为局部区域并为每个区域使用专属MLP以增强对光照与尺度变化的适应；③ 采用锚点网格结构降低GPU内存占用并保持稀疏前端的定位稳健性。

**🔧 技术方法**

核心技术包括：ORB‑SLAM2特征跟踪、LiteFlowNet3光流、3D高斯光栅化（3DGS）与稀疏锚点网格、局部MLP条件化以及基于极线约束的三角剖分。

**📊 数据集**

使用了KITTI Odometry、Oxford RobotCar和Málaga三大公开长序列数据集进行评估。

**📈 对比分析**

与PhotoSLAM、GigaSLAM和MonoGS对比，GLAM‑SLAM在PSNR/SSIM/LPIPS上平均提升约15%且保持10–20 FPS的实时性能，同时显著降低GPU内存（≈5–15 GiB），能够完整处理超过4000帧的序列。

**⚠️ 局限性**

局限性：对光流的依赖可能在极端动态场景或运动模糊下失效；锚点网格的固定分辨率仍限制细节捕捉；在极大规模场景中仍需进一步压缩内存与加速光流推断。

---

## 492. Chip Floorplanning Combining Convex and Non-convex Optimization

**arXiv ID:** 2607.21408 | [PDF](https://arxiv.org/pdf/2607.21408v1)

**作者:** Yilihamujiang Yimamu `[一作]` (Universitat Politècnica de Catalunya), Jordi Cortadella `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三阶段固定轮廓floorplanning框架，结合二次放置、Adam投影梯度的非凸全局优化和对数变换+约束图的凸合法化；

**💡 创新点**

创新点在于统一连续可微目标、动态惩罚自适应、无分段几何枚举以及利用对数映射将非凸重叠约束转化为凸合法化；

**🔧 技术方法**

使用二次/对数求和逼近HPWL、Softplus平滑重叠惩罚、Adam投影梯度、IPOPT求解凸合法化；

**📊 数据集**

在MCNC、GSRC和HB+三组软块基准上进行实验；

**📈 对比分析**

与Parquet‑4、SDP、PeF、Capo、IMF、DeFer等方法对比，平均HPWL降低1%–12%，在GSRC和HB+上比现有方法至少提升3%–33%；

**⚠️ 局限性**

缺点是对大规模工业设计的扩展性待验证，且全局优化迭代成本相对较高；

---

## 493. Toward Continuous Assurance for the Democratization of AI Agent Creation in Industry

**arXiv ID:** 2607.21495 | [PDF](https://arxiv.org/pdf/2607.21495v1)

**作者:** Natan Levy `[一作]` (Hebrew University of Jerusalem), Harel Berger `[通讯]` (Ariel University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套轻量级的连续保障框架，帮助组织在非工程师创建的 AI 代理中持续评估依赖完整性、功能可用性和运营准备度，并开发了基于 GPT 的原型审计器进行场景化评估。

**💡 创新点**

创新点在于：①将工业可靠性缺口定义为“长期存在的民主化 AI 代理”并构建专属失效分类体系；②引入“依赖映射 + 就绪合同 + 定期检查 + 诊断 + 生命周期治理”五要素的连续保障流程；③将上述框架转化为可由非技术用户执行的准则与自动化审计工具，突破传统 DevOps/MLOps 的工程门槛。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）用于生成与执行审计指令；依赖映射与就绪合同模板；定时任务与状态监测；诊断分类与风险评估算法；以及基于角色与权限的生命周期治理机制。

**📊 数据集**

本工作主要以内部案例与人工构造的情景化评估为实验材料，未使用公开标准数据集。

**📈 对比分析**

对比方式采用情景化故障评估：为每个预设故障场景事先设定期望评估结果，随后让原型审计器输出并与预期进行对照。评估显示审计器能准确将问题归类为相应失效类别、区分已确认证据与潜在风险，并给出可执行的补救建议；在未能检验的属性上，审计器能标记为未知或不可外部验证，避免误报。

**⚠️ 局限性**

局限性包括：①缺乏对检测覆盖率、误报率和恢复时间的量化评估；②依赖案例由作者自行设计，未对真实生产中的故障进行验证；③审计器本身基于 LLM，存在漂移与 API 变化风险，需进一步引入元保障机制；④在无外部可检视配置的环境下，无法完成完整的就绪检查，可能导致评估不完整。

---

## 494. Sources of Inequity and Fairness Risks inWellbeing Sensing

**arXiv ID:** 2607.21527 | [PDF](https://arxiv.org/pdf/2607.21527v1)

**作者:** Han Zhang `[一作]` (University of Chicago), Jennifer Mankoff `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过半结构化访谈研究了在心理健康与工作场所等高风险环境中，使用智能手机与可穿戴设备进行被动感知（Passive Sensing）时出现的公平性风险与缓解策略，系统梳理了从研究规划到部署整个生命周期中的不平等来源与风险。

**💡 创新点**

创新点在于：①将公平性视角从单一的身份属性扩展到情境因素（如监测舒适度、设备获取、数字素养、文化熟悉度、行为规律性），揭示被动感知独有的五类源不平等；②提出了15条基于生命周期的公平缓解策略，并强调研究者与生态系统层面的治理支持。

**🔧 技术方法**

主要技术是对访谈文本进行主题分析（采用TAM或类似方法），并以图表（生命周期图、风险-缓解表）呈现结果；未涉及新算法或模型。

**📊 数据集**

本文并未使用公开数据集，而是基于14位来自澳大利亚、加拿大、日本、韩国与美国的研究者/从业者的访谈记录作为研究材料。

**📈 对比分析**

本文不进行算法性能比较，因研究焦点为公平性风险与治理策略，而非模型评估。

**⚠️ 局限性**

局限性包括：样本仅为14位研究者，缺乏数据提供者和终端用户视角；研究仅基于访谈，对实际系统的公平性效果缺乏量化验证；所述风险与缓解策略可能随技术演进而变化，未形成通用框架。

---

## 495. Transparent by Design, Usable in Practice? A Formative Usability Study of a Conversational Product Advisor

**arXiv ID:** 2607.21513 | [PDF](https://arxiv.org/pdf/2607.21513v1)

**作者:** Kevin Schott `[一作]` (GESIS – Leibniz Institute for the Social Sciences), Daniel Hienert `[通讯]` (GESIS – Leibniz Institute for the Social Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一项对话式产品推荐系统的可用性研究中，探索了如何通过透明度设计提升用户对系统的理解与信任，针对笔记本电脑搜索构建了一个结合LLM与确定性排名器的聊天机器人并对其进行思考大声可用性测试。

**💡 创新点**

创新点在于将透明度嵌入到对话式助手本身：提供按需排名解释（雷达图+惩罚表）、系统回显对话推断的需求以及可比视图，并通过实证研究揭示透明度并不等同于可理解性。

**🔧 技术方法**

使用的技术包括：混合架构——确定性排名器（基于类别过滤与数值惩罚）与受限自然语言生成模型（LLM），以及基于卡片、雷达图、表格、弹窗等可视化组件的对话式界面。

**📊 数据集**

采用的产品数据集为从Amazon获取的Windows笔记本电脑规格目录（含价格、品牌、内存、存储等属性）。

**📈 对比分析**

通过七名受试者的远程思考大声可用性测试，收集SEQ、满意度、BUS‑11指标，结果显示总体易用性与满意度均高，但针对排名解释的任务易用度最低，说明该功能最为严重。

**⚠️ 局限性**

局限性包括样本规模小且仅来自英国、未与基线系统对比、仅评估结构化产品、受导向使用导致发现性不足、测量主要基于主观评估、且编码工作仅由一人完成。

---

## 496. What, Where, and How: Disentangling the Roles of Task, Language, and Model in Code Model Representations

**arXiv ID:** 2607.21491 | [PDF](https://arxiv.org/pdf/2607.21491v1)

**作者:** Piotr Wilam `[一作]` `[通讯]` (University College London), Piotr Wilam (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对两款不同语言模型（Qwen2.5‑Coder‑7B 与 DeepSeek‑Coder‑V1‑6.7B）在 Python 与 Rust 两种形式语言上进行统一的概念电路抽取，并系统测量了 115 个可检验概念在四个模型‑语言单元中的电路分布；

**💡 创新点**

创新点在于把“电路是否普遍”拆分为“What/Where/How”三个轴，首次揭示在任务维度上电路分配相同，而在层级与电路增长方式上高度依赖模型，同时提出了跨语言跨模型共享电路的“处理风格指纹”，为跨模型解释性迁移提供理论依据；

**🔧 技术方法**

采用了概念电路提取方法（概念提示与检查提示交叉化、神经元二值化与交集求解）、层级化激活记录、交叉模型 Spearman 相关、层级概念占比、Jaccard 共享率、零消融和线性探测等多种技术手段；

**📊 数据集**

使用了两大模型在两种语言上的 115 个概念库存（Python 58 概念、Rust 57 概念），并构造了多样化提示集，涵盖词法、结构与填充维度；

**📈 对比分析**

通过 Spearman ρ≈0.65 的跨模型概念分数相关、层级概念占比峰值差异（Qwen 17‑19 层，DeepSeek 6‑7 层）以及共享率对比，验证了电路位置与增长模式的模型依赖性；

**⚠️ 局限性**

主要局限包括：仅捕捉高幅值神经元，忽略子阈值分布；仅针对命令式语言，未覆盖声明式或证明助手；层数不一致导致绝对层级比较受限；因果验证仅在 Qwen‑Python 单元进行；数据集仅为两种语言，缺乏更广泛的形式语言验证。

---

## 497. Compact Latent Coordination for Autonomous Vehicles at Unsignalized Intersections

**arXiv ID:** 2607.21488 | [PDF](https://arxiv.org/pdf/2607.21488v1)

**作者:** Gil Lifshits `[一作]` (Ben-Gurion University of Negev), Gilad Katz `[通讯]` (Ben-Gurion University of Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了MAPS，一种利用集中式Master生成连续“proto-plan”并由去中心化Worker使用的分层强化学习框架，用于无人信号灯交叉口的多车协同行驶。

**💡 创新点**

创新点在于将全局协调信息压缩为低维连续向量（proto-plan），实现策略与执行解耦，通信复杂度保持O(d)并支持零样本迁移到更大车辆数量。

**🔧 技术方法**

使用Proximal Policy Optimization（PPO）训练Master和Worker；Master观测全局状态生成d维proto-plan，Worker结合本地运动状态进行动作选择；采用增量式训练和最大最小奖励设计。

**📊 数据集**

实验基于HighwayEnv仿真环境，构造了72个不同方向、距离和转向的交叉口配置，并在5辆车场景中评估，其中3辆为学习车，其余为常速车。

**📈 对比分析**

与两种基线（QMIXwD和VN‑MADDPG）相比，MAPS在100个评估回合中零碰撞、成功率100%，平均行驶时间7.8步，比最佳基线低38%；训练期间碰撞次数仅21次，显著低于基线。

**⚠️ 局限性**

局限包括：仅在低保真kinematic仿真中验证，未考虑感知噪声和高级动力学；Worker动作空间仅限二进制加速控制；Master输入固定最大车辆数，缺乏可扩展到大规模车队的可变维度处理；未与更复杂的层级或安全模块结合。

---

## 498. Thinkink: 2D Spatial Ink-native Interaction with LLMs

**arXiv ID:** 2607.21468 | [PDF](https://arxiv.org/pdf/2607.21468v1)

**作者:** Mohammad Hasan Payandeh `[一作]` (University of Waterloo), Jian Zhao `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Thinkink，一个支持手写文字与草图在同一二维画布上与大型语言模型互动的工具。

**💡 创新点**

创新点在于将手写输入与 LLM 生成的文字与图像无缝融合，并通过语义树和状态机实现可视化、可控的交互。

**🔧 技术方法**

使用大语言模型（如 GPT）生成文本与草图，配合语义树解析、轻量级 UI 与状态机控制，构建二维共享画布。

**📊 数据集**

主要使用用户研究数据：形成性研究 N=12、诊断研究 N=6、最终评估 N=10，未使用公开数据集。

**📈 对比分析**

评估基于用户实验，比较不同模式（如 AI insights 与 Ask AI）下的使用体验，未给出定量性能指标。

**⚠️ 局限性**

局限性包括样本量有限、未进行与现有绘图+LLM工具的客观比较，以及对更复杂交互场景的支持不足。

---

## 499. Out-of-Distribution Detection in Wireless Multimodal Foundation Models for 6G ISAC

**arXiv ID:** 2607.21455 | [PDF](https://arxiv.org/pdf/2607.21455v1)

**作者:** Mohammad Farzanullah `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 WMFM‑OOD 框架，利用基站几何原型和温度缩放的概率评分，对无线多模态基础模型进行 OOD 检测。

**💡 创新点**

创新点：①在联合潜在空间中构造基站原型来捕获合法射频环境的流形；②采用温度缩放的 Softmax 对原型相似度进行校准，以提升 ID 与 OOD 的可分性；③将视觉与 CSI 特征融合，实现多模态协同检测。

**🔧 技术方法**

技术：无线多模态基础模型（WMFM）编码器、基于余弦相似度的原型匹配、温度缩放的 Softmax 评分、SVM‑style 线性归一化、离线原型构建与在线推理。

**📊 数据集**

数据集：DeepVerse6G，ID 场景为 O1（高速公路）和 OOD 场景为 Carla‑Town1（城市交叉口）。

**📈 对比分析**

与四种基线（直接模态对齐、最大原型相似度、能量 OOD、模态不一致度）比较，WMFM‑OOD 在 AUROC 上达 0.8824，FPR95 降至 0.460，较最佳基线提升约 17%。

**⚠️ 局限性**

局限性：依赖预训练的基础模型，原型构建需手动离线完成，温度参数仍需经验调优；在极端环境或完全未知的射频特征下，可能仍出现误检。

---

## 500. Bridging the Gap Between Plausibility and Admissibility: Constraint-Aware Flow Maps for Dynamic Graph Systems

**arXiv ID:** 2607.21421 | [PDF](https://arxiv.org/pdf/2607.21421v1)

**作者:** Michael Romei de Socio `[一作]` (School of Advanced Defense Studies), Alessio Merlo `[通讯]` (School of Advanced Defense Studies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在动态图结构系统中，结合条件扩散生成器与外部符号约束层（硬过滤、软加权、投影修复）来生成满足结构可行性的未来轨迹。

**💡 创新点**

创新点在于提出了基于流图的分层框架，独立将生成与符号可行性评估分离，并对不同复杂度图谱进行系统比较，揭示了统计可行性与结构可行性两种可靠性维度。

**🔧 技术方法**

使用了条件扩散模型作为轨迹生成器，符号约束层实现了硬过滤、软加权（基于违约惩罚）和投影修复三种后采样处理。

**📊 数据集**

采用了两套人工合成的动态图数据集：1）紧凑图（9节点、16条边）；2）中等复杂度依赖图（15节点、37条边），每套包含3000条长度80的基础轨迹，观测窗口8，预测长度16。

**📈 对比分析**

对比了四种策略（无约束、硬过滤、软加权、投影），评估指标包括无效概率质量、保留比例、有效样本大小、轨迹多样性和校准误差；结果显示：在中等复杂度图中无约束生成的无效质量从0.003提升至0.156，硬过滤将其降至0同时保留84.4%样本；软加权仅略减无效质量，保持有效样本大小；投影修复几乎保证完整性。

**⚠️ 局限性**

局限性包括：实验仅基于人工合成数据；仅考察两种图复杂度；约束在训练阶段未被整合，未对比约束引导生成方法；软加权的违约惩罚设计有限，未探究自适应或更强的约束学习策略。

---

## 501. Euclid-MCP: A Model Context Protocol Server for Deterministic Logical Reasoning via Prolog

**arXiv ID:** 2607.21412 | [PDF](https://arxiv.org/pdf/2607.21412v1)

**作者:** Bartolomeo Bogliolo `[一作]` `[通讯]`, Bartolomeo Bogliolo

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为 Euclid-MCP 的开源服务器，通过 Model Context Protocol (MCP) 将 LLM 与确定性逻辑推理引擎（SWI-Prolog）耦合，实现了可验证、可解释的业务规则与安全合规检查。

**💡 创新点**

创新点在于：①设计了面向 LLM 的、可读性强且与后端无关的中间表示语言 Euclid-IR；②构建了可直接通过 MCP 调用的四大工具（推理、诊断、场景分析、KB 验证），支持 translate‑run‑inspect‑repair 循环；③将推理过程从 LLM 中剥离，显著降低 LLM 产生幻觉的风险，并提供完整的证明树，提升可审计性。

**🔧 技术方法**

核心技术包括：MCP 规范实现、Python 服务器与 SWI-Prolog 的交互、Euclid-IR 的语法设计与 Prolog 翻译层、基于 JSON 的工具接口、以及对输入进行安全沙箱和执行时限控制。

**📊 数据集**

使用的数据集主要有：①IT 安全与合规用例（小型约 30 用户/50 资源，约 578 条事实；大型约 200 用户/300 资源，约 3,872 条事实）；②Synthetic RBAC 基准（1,000 名用户、1,053 条事实）。

**📈 对比分析**

比较方法：将 LLM 单独推理（8B 本地模型和 480B 云模型）与 LLM+Euclid-MCP 进行相同任务的准确性、时延与 token 效率对比。结果显示：在小型 KB（5–50 条事实）下三者准确率相同；在大型 KB（>1,000 条事实）时，LLM 全部出现幻觉，计数与推理错误；Euclid-MCP 始终给出精确答案，平均时延约 1 s，输出 token 大幅减少。

**⚠️ 局限性**

限制包括：①仅支持 Horn‑clause 逻辑，缺乏并列、剪枝、列表、字符串等高级 Prolog 特性；②当前仅实现 SWI‑Prolog 后端，未验证 Datalog/SMT 等其他引擎的兼容性；③对 KB 大小的实测上限约 10K 条事实，超过此规模需要进一步优化；④集成时需要额外的工具链与错误处理，增加了开发复杂度。

---

## 502. Themis Consensus Extension v1: MEV Mitigation by Randomized Delayed Execution and Intent-Hiding Transactions in Application-Specific Blockchains

**arXiv ID:** 2607.21406 | [PDF](https://arxiv.org/pdf/2607.21406v1)

**作者:** Shoeb Siddiqui `[一作]`, Peter Kris `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种称为Themis Consensus Extension v1的区块链协议扩展，主要通过将区块内容提交与执行顺序分离以及可选的双层加密隐蔽路径来降低最大可提取价值（MEV）

**💡 创新点**

在一次性提交后延迟执行并使用基于VRF的随机种子进行依赖保持的打乱，创新性地将内容选择和顺序决定拆分为不同时间点的角色，并引入两层加密来减轻选择性拒绝（VED）

**🔧 技术方法**

采用PoS共识（Aura、BABE）、sr25519/VRF随机生成、Fisher-Yates/Xoshiro256++打乱、混合认证加密（Hybrid Authenticated Encryption）以及Substrate框架中的模块化实现

**📊 数据集**

未使用公开数据集，研究以理论分析和Substrate实现验证为主，基于设计规范和实现日志进行评估

**📈 对比分析**

通过实现演示和安全分析验证协议满足随机化执行和意图隐藏的目标；性能方面未给出定量基准，但实现已在Mangata DEX中部署，延迟执行增加一个区块周期，算力和网络开销可接受

**⚠️ 局限性**

存在多项限制：仅在非协作的相邻角色场景下有效；无法防止概率性提取、总拒绝、元数据泄露和协作攻击；对Sybil、垃圾邮件、加密事务失败的经济防护有限；在治理或共识阈值被突破时无法保证安全

---

## 503. AREX: Towards a Recursively Self-Improving Agent for Deep Research

**arXiv ID:** 2607.21461 | [PDF](https://arxiv.org/pdf/2607.21461v1)

**作者:** Shuqi Lu `[一作]` (Beijing Academy of Artificial Intelligence), Zheng Liu `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归自我改进（RSI）深度研究代理AREX，能够在发现–验证不对称的基础上，将部分已验证的答案转化为更具针对性的后续研究问题，并通过内部研究循环和外部自我改进循环实现迭代优化；

**💡 创新点**

创新点在于：1) 将验证作为递归循环的主动控制信号，而非最终过滤；2) 引入自主上下文更新工具，实时压缩并保持关键研究状态；3) 在训练阶段采用多阶段代理中训练、关键步骤聚焦监督以及基于步骤的强化学习，提升长周期研究的学习效率与效果；

**🔧 技术方法**

技术包括：大规模语言模型（Qwen3.5‑4B 与 Qwen3.5‑122B‑A10B）、搜索/浏览工具、自动上下文更新工具、答案外部化工具、递归双层控制框架、合成任务与教师轨迹生成、关键步骤检测与专门监督、步骤级奖励塑造的强化学习；

**📊 数据集**

数据集：自研的深度研究任务合成数据与教师轨迹数据；评测基准包括 BrowseComp、GAIA、xbench‑2510、DeepSearchQA、WideSearch‑en 与 HLE（工具）等；

**📈 对比分析**

与多种前沿闭源与开源模型对比，AREX‑Base 在 10B 活跃参数下取得 82.5（BrowseComp）/85.4（GAIA）/71.0（xbench‑2510）/89.9（DeepSearchQA）/82.0（WideSearch‑en）/52.4（HLE）等指标，显著优于同规模模型并在多项基准上与更大模型竞争；

**⚠️ 局限性**

局限性包括：需要复杂的工具集成和多阶段训练，难以在资源受限环境中快速部署；对特定工具/知识库的依赖可能限制通用性；当前模型仍依赖人工标注的关键步骤与手工设计的奖励机制，尚未实现完全自动化的步骤效用估计与细粒度学习信号。

---

## 504. SPDCN: Strip-based Deformable Convolutional Network for Steel Surface Defect Segmentation

**arXiv ID:** 2607.21456 | [PDF](https://arxiv.org/pdf/2607.21456v1)

**作者:** Zhongming Liu `[一作]` (Jiangxi Normal University), Xiang Zou `[通讯]` (Jiangxi Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种专为钢表面缺陷分割设计的网络SPDCN，能够精准捕捉细长裂纹和划痕等形状复杂的缺陷。

**💡 创新点**

创新点包括：① Fuzzy-enhanced Multi-scale Context Module (FMCM)，通过分组多尺度卷积与直觉主义模糊通道注意力实现对不同尺度缺陷的自适应聚合；② Adaptive Direction-Aware Deformable Convolution (ADADC)，用水平和垂直条纹卷积解耦方向特征来预测可伸缩的偏移，从而让可变形卷积沿缺陷主轴自适应采样。

**🔧 技术方法**

技术手段涵盖：U‑Net 结构、可变形卷积、条纹卷积、模糊通道注意力、二值交叉熵+Dice 损失、数据增强、AdamW 优化器等。

**📊 数据集**

使用公开的两大钢表面缺陷数据集：NEU‑Seg（4,470张 200×200 的热轧钢条图像）和 Magnetic Tile（磁砖表面缺陷图像）。

**📈 对比分析**

与多种主流分割算法（U‑Net、DeepLabV3+、PSPNet、BiSeNet、SCSegamba 等）在同一实验设置下对比，SPDCN 在 NEU‑Seg 上达到 89.60% mIoU、94.42% mDice，参数 3.54M；在 Magnetic Tile 上得到 78.83% mIoU、88.58% mDice，显著优于其他方法。

**⚠️ 局限性**

局限性包括：对极其稀疏或极小尺寸缺陷的鲁棒性尚待提升；网络在多尺度金属表面缺陷场景下的迁移性能尚未充分验证；当前实现主要针对单帧图像，缺乏对视频序列连续性约束的考虑。

---

## 505. GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis

**arXiv ID:** 2607.21448 | [PDF](https://arxiv.org/pdf/2607.21448v1)

**作者:** Jiahao He `[一作]` (Nanjing University of Aeronautics and Astronautics), Qi Tian `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 GrainGS 框架，使用层次 Anchor Scaffold 与 per‑Gaussian 形变实现高效动态 3D Gaussian Splatting 的新方法。

**💡 创新点**

创新点在于：① 通过 stop‑gradient 分离 Canonical Geometry 与 Deformation 的梯度，避免形变干扰静态几何；② 引入 Canonical‑Residual 颜色分解，专门建模帧间光照变化；③ 在 Anchor Scaffold 上实现可扩展的局部细化，并采用两阶段 Warm‑up + Joint 训练。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、层次 Anchor Scaffold、Per‑Gaussian Deformation MLP、stop‑gradient 操作、Canonical‑Residual 颜色分解、光度正则化、时间条件残差分支、两阶段训练策略。

**📊 数据集**

使用数据集：synthetic D‑NeRF（8 场景）和真实多视角 DG‑Mesh（6 场景）进行评估。

**📈 对比分析**

与 NeRF、4D‑GS、D‑3DGS、SC‑GS、4D‑Scaffold‑GS、MoDec‑GS 等方法对比，GrainGS 在 PSNR 约 36.98 dB、渲染 435.6 FPS、模型存储 4.67 MB 方面均实现或逼近 state‑of‑the‑art，兼顾质量、速度与存储。

**⚠️ 局限性**

局限性：对极端光照或极快动态场景仍可能出现细节丢失；Anchor Scaffold 的生长策略对稀疏或无纹理区域的几何细化不够精细；训练仍需较长时间，且对不同场景的超参数需要手工调整。

---

## 506. Context-weighted Discrete Flow Matching

**arXiv ID:** 2607.21427 | [PDF](https://arxiv.org/pdf/2607.21427v1)

**作者:** Daniil Cherniavskii `[一作]` (University of Amsterdam), Karen Ullrich `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了离散流匹配中局部上下文对预测不确定性的影响，并提出了基于局部上下文加权的采样方法和可缩放交叉熵损失，显著提升了生成质量。

**💡 创新点**

创新点在于将局部上下文嵌入连续时间马尔可夫链的速度，并通过上下文加权的采样和损失重构，使得无需微调即可在推理时提升采样质量，并通过可缩放交叉熵显著改善训练效率。

**🔧 技术方法**

使用的技术包括离散流匹配（Discrete Flow Matching）、上下文加权的CTMC速度设计、邻域加权/熵加权采样、可缩放交叉熵损失、Euler求解器以及预测-校正采样框架。

**📊 数据集**

实验数据集主要为OpenWebText（文本）和QM9 SMILES（分子）。

**📈 对比分析**

在MAUVE、生成困惑度、有效分子数和新颖性等指标上，与标准Euler采样、熵加权采样以及半自回归块扩散基线进行比较，结果显示上下文加权方法在大多数情形下匹配或超过强基线，尤其在低数据或低NFE场景下显著提升。

**⚠️ 局限性**

局限性包括：在极低NFE时邻域采样效果不佳；实验仅覆盖小模型和1D序列，未验证在更大模型或二维图像等高维数据上的表现；对不确定性代理的选择仍可进一步改进。

---

## 507. PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement Learning

**arXiv ID:** 2607.21419 | [PDF](https://arxiv.org/pdf/2607.21419v1)

**作者:** Yipeng Shi `[一作]` (Peking University), Zhengzhou Zhu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于策略的训练支架框架，将外部技能视为临时训练时的指导而非部署时的记忆，通过将滚动组转化为证据并自适应调整支架来提升大语言模型代理的强化学习效果。

**💡 创新点**

创新点在于：①将技能视作可废弃、动态的训练支架；②利用成功-失败对比的滚动组证据来决定支架的扩展、修订或压缩；③在训练阶段对支架进行闭环自适应，而在部署时完全移除支架，从而实现更高效的推理。

**🔧 技术方法**

核心技术包括：策略强化学习（GRPO）与 PPO 风格的奖励归一化；基于任务类型的支架渲染与检索；证据卡生成与聚合；基于政策成熟度的支架调度器；约束语义修订器；以及监督微调初始化支架接口。

**📊 数据集**

实验数据集涵盖 ALFWorld（6类文本交互任务）与 WebShop（模拟购物搜索任务）两大环境，以及七个搜索增强问答基准（如 NQ、HotpotQA 等）。

**📈 对比分析**

与 GRPO、SKILL0、SkillRL、RLOO 等基线相比，本文方法在 ALFWorld 的成功率提升 12.9%~17.6%，在 WebShop 的成功率提升 3.5%~18.6%，同时在部署时将交互 token 减少 30%~40%；在搜索增强问答任务中，保持 32% 更少 prompt token 的同时平均提升 3.5 分。

**⚠️ 局限性**

局限性包括：需要对支架进行复杂的初始化与验证；虽然支架在部署时被移除，但部分技能的即时推理价值尚未完全评估；实验仅覆盖有限的任务域，难以直接推广到更广泛的真实世界应用。

---

## 508. Agentic coding without the cloud: evaluating open-weight large language models on longitudinal data preparation tasks

**arXiv ID:** 2607.21482 | [PDF](https://arxiv.org/pdf/2607.21482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 509. Finite-Sample Coverage Audits for High-Recall Candidate Generation: Certification and Learning-Theoretic Design

**arXiv ID:** 2607.21480 | [PDF](https://arxiv.org/pdf/2607.21480v1)

**作者:** Martin Anthony `[一作]` (London School of Economics and Political Science), Kaveh Salehzadeh Nobari `[通讯]` (London School of Economics and Political Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套严格的有限样本缺失相关物品（missed‑mass）认证框架，阐述了为何必须对被排除的候选集进行标注，并给出了在包含与排除两池样本设计下的精确置信上界；

**💡 创新点**

创新点在于证明仅使用包含池标签无法给出任何非平凡的缺失相关物品上界，并给出了在有限语料库下最优的排除池标签复杂度下界；

**🔧 技术方法**

主要技术包括精确的 Clopper–Pearson 二项与超几何反演、两池联合检验、预指定前缀集合的并行证书、以及在设计与认证分离假设下的学习理论上界；

**📊 数据集**

本文主要为理论性工作，并未在具体数据集上进行实验；

**📈 对比分析**

方法与传统经验估计或置信区间相比，提供了分布自由、单侧、确切覆盖率的保证，能够在零漏检场景下实现近似最优标签复杂度；

**⚠️ 局限性**

局限性包括：必须预先固定候选生成器或前缀族；仅针对缺失相关物品（未涵盖误判或模型偏差）；在稀有事件场景下的置信上界可能仍较宽；实际应用需配合合适的设计样本与审计分离策略。

---

## 510. Boosting Robustness for All-Weather Self-Supervised Depth Estimation in Autonomous Driving

**arXiv ID:** 2607.21526 | [PDF](https://arxiv.org/pdf/2607.21526v1)

**作者:** Mengshi Qi `[一作]` (State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications), Huadong Ma `[通讯]` (State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对全天候自主驾驶场景，提出一种自监督深度估计框架，结合多教师蒸馏和相机-雷达跨视角融合，显著提升在雨、雾、夜等恶劣天气下的深度估计鲁棒性。

**💡 创新点**

创新点包括：①不确定性感知多教师蒸馏（UAMTD），通过多组天气专家教师和像素级不确定性权重实现自适应知识迁移；②POV‑BEV雷达融合（PBCRF），利用相机像素射线约束实现密集BEV雷达点云与相机视角的跨视角注意力，充分利用雷达的全局视角优势。

**🔧 技术方法**

主要技术手段有：自监督视差重建损失、知识蒸馏与不确定性建模、ResNet‑18 级别特征提取、BEV投影与逆投影、交叉注意力模块、数据增强与优化策略（AdamW、学习率调度）等。

**📊 数据集**

实验使用公开全天候数据集 RADIATE（包含雨、雾、夜、晴等）和 nuScenes（夜、雨、晴等），无额外标注直接训练。

**📈 对比分析**

与现有单帧、时序以及雷达融合基线对比，UAMTD+PBCRF 在 RADIATE 上 absRel 下降 26%，nuScenes 夜间/雨天分别下降 9.1%/22.8%；在所有天气条件下均能实现显著的 RMSE 与 δ1 改进，整体优于目前公开的自监督与合成天气方法。

**⚠️ 局限性**

局限性：雷达点云稀疏导致融合效果受限，模态不平衡仍难以完全解决，某些极端条件（如强光镜面、极夜）仍出现误检；PBCRF 的额外参数与推理时延对低算力平台有一定压力。

---

## 511. Diffusion Language Model for Recommendation

**arXiv ID:** 2607.21519 | [PDF](https://arxiv.org/pdf/2607.21519v1)

**作者:** Chengyi Liu `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于离散扩散语言模型 DLMRec，用以替代自回归生成推荐系统。

**💡 创新点**

三大创新：协作感知随机分词器将多跳协作信号映射为离散标记；课程化训练策略将扩散去噪与偏好恢复对齐；稳定投票机制利用迭代预测聚合提升生成一致性。

**🔧 技术方法**

离散扩散语言模型、协作感知随机分词（CAST）、课程化训练、投票式迭代细化以及大规模预训练 LLaDA-8B 作为生成器。

**📊 数据集**

在 LastFM、MovieLens‑1M 与 Amazon‑Beauty 三个工业级交互数据集上进行实验。

**📈 对比分析**

与多种基线（LightGCN、SASRec、DiffRec、LLaRA 等）按 Recall@K/NDCG@K 进行对比，DLMRec 在所有数据集均实现显著提升，平均 Recall@20 提升约 6–12%。

**⚠️ 局限性**

主要局限在推理步数对 latency 影响较大、对超大词表的扩展性待验证，以及对极稀疏用户行为的泛化仍有提升空间。

---

## 512. Error Certificates for KV-Cache Eviction via Randomized Design

**arXiv ID:** 2607.21475 | [PDF](https://arxiv.org/pdf/2607.21475v1)

**作者:** Peng Xie `[一作]` `[通讯]` (Technical University of Munich), Peng Xie (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型中KV缓存的淘汰机制，证明确定性top‑k淘汰无法识别误差，并提出在尾部使用Poisson随机采样，结合Hájek校正和Sen–Yates–Grundy方差估计，构建了可在线计算的误差证书。

**💡 创新点**

创新点在于：①引入随机Poisson尾部淘汰并证明其可实现误差可识别；②利用设计基础推断提供无偏的一阶误差估计；③设计基于Empirical‑Bernstein的覆盖证书，并通过e‑process实现时间均匀有效性。

**🔧 技术方法**

使用了Poisson采样、Hájek权重校正、Sen–Yates–Grundy方差估计、Empirical‑Bernstein上下界、e‑process、以及随机化设计基础推断技术。

**📊 数据集**

实验采用Qwen2.5-1.5B/7B、Llama‑3.1‑8B、Mistral‑7B等模型，在LongBench、RULER式任务、合成Needle Retrieval以及16k上下文的长文本任务上进行评估。

**📈 对比分析**

与传统确定性top‑k、SnapKV、H2O等方案对比，随机化方法在覆盖率（>97%）、误差相关系数（≥0.94）方面保持不逊，且在归因和重算调度上优于随机或置信门控，预测错误时不劣于输出对数概率。

**⚠️ 局限性**

局限性包括：仅在8B模型与≤16k上下文验证；证书需额外标量且在原型实现中产生约两倍解码延迟；未测试固定大小采样、极长上下文或真实多轮记忆基准。

---

## 513. White Box Evidence Packages for Policy Audit Reports

**arXiv ID:** 2607.21462 | [PDF](https://arxiv.org/pdf/2607.21462v1)

**作者:** Seunghyun Yoo `[一作]` `[通讯]` (Governance and Responsible AI Lab), Seunghyun Yoo (Governance and Responsible AI Lab)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并评估了不同证据接口（表面证据、内部模型证据、混合接口和打乱控制）对LLM生成的基于政策文本的审计报告的影响。

**💡 创新点**

将内部模型解释工具的输出包装为可审计的证据，并提出了基于验证审查的评估框架，发现内部证据并不一定提升报告质量，混合接口具备潜力，打乱控制揭示了治理风险。

**🔧 技术方法**

使用Sparse Autoencoder（SAE）、Logit Lens、Steering Sensitivity、Activation Explanation Surrogate 等解释工具以及残差流补丁诊断；LLM审计模型为Qwen 2.5-7B，目标模型为Gemma 2-2B。

**📊 数据集**

AGORA政策文本语料库（60个案例）和对应的Gold Brief。

**📈 对比分析**

通过五位评审员对240份报告进行验证评审，比较了四种主要接口的正确性、段落定位、可用性和证据误用；结果显示混合接口与基线相当但误用率略高，内部证据接口虽提高引用量但降低定位和可用性。

**⚠️ 局限性**

局限包括：未能完全分离“锚定”与证据量的影响、仅使用单一审计模型和目标模型、工具质量未校准、评审员数量有限、AGORA数据不等同于法律合规判定、无法验证人类审计员对打乱证据的反应。

---

## 514. Test-Time Scaling via Error Localization

**arXiv ID:** 2607.21453 | [PDF](https://arxiv.org/pdf/2607.21453v1)

**作者:** Rajiv Shailesh Chitale `[一作]` (Google DeepMind), Aravindan Raghuveer `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种测试时错误定位（Test-Time Scaling via Error Localization）算法，能够在推理时对失败的生成轨迹进行基于令牌级的错误定位并重用有效前缀，从而提升推理计算效率。

**💡 创新点**

创新点在于：①利用反馈增强的教师模型与基线无诊断反馈的概率对比来识别令牌级错误；②通过截断错误位置并从该位置分支，构建树形搜索，实现高效的计算资源分配；③将自我去蒸馏中的 KL 距离转换为在线搜索信号，而不需要梯度更新。

**🔧 技术方法**

主要技术包括：自回归语言模型的概率重评分、基线去噪过滤、令牌级峰值检测、树结构搜索与分支策略。

**📊 数据集**

实验使用了 Qwen3-8B / Qwen3-4B-Thinking-2507 语言模型，评估数据集包括 LiveCodeBench（代码生成）、AIME-2025 与 HMMT-2025（数学推理）。

**📈 对比分析**

与独立采样、顺序多轮改进和递归自聚合等基线对比，TTEL 在 pass@k 与生成令牌数上构成严格优势的 Pareto 前沿；例如在 LiveCodeBench 上 pass@64 为 71% 而生成令牌仅为独立采样的一半，AIME-25 上在 k=16 时取得 82% pass。

**⚠️ 局限性**

限制主要体现在：①需要可靠的环境反馈或足够完整的思路轨迹；②对低质量或无诊断反馈时错误定位效果下降；③对极长生成序列时计算成本仍然较高；④尚未对多模型或更大规模模型进行验证。

---

## 515. When Trivia Is Not Trivial: Everyday Knowledge Failures in Multilingual LLMs

**arXiv ID:** 2607.21445 | [PDF](https://arxiv.org/pdf/2607.21445v1)

**作者:** Anna Mosolova `[一作]`, Djamé Seddah `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文主要说明如何在 LuaLaTeX 或 XeLaTeX 环境下使用 ACL 风格文件。

**💡 创新点**

在这份示例文档中并未提出新的研究思路或方法，缺乏创新点。

**🔧 技术方法**

技术实现仅涉及 LaTeX 排版配置，即使用 ACL 样式文件与 LuaLaTeX/XeLaTeX 的兼容性。

**📊 数据集**

文中没有使用任何实验数据集，只展示了一段示例文本。

**📈 对比分析**

没有进行任何方法对比或性能评估，因文档仅为排版演示。

**⚠️ 局限性**

局限性在于内容仅为排版说明，缺乏实际科研实验与数据验证。

---

## 516. Revisiting Degree-Corrected Spectral Clustering: a Condition-Free Spectral Analysis and Extension

**arXiv ID:** 2607.21435 | [PDF](https://arxiv.org/pdf/2607.21435v1)

**作者:** Wei Li `[一作]` (Fujian Agriculture and Forestry University), Jianfeng Hou `[通讯]` (Fuzhou University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文重新分析了度校正谱聚类（DCSC）的聚类质量，给出了一个不依赖随机图模型且无额外条件的误分节点上界，并提出了基于 GNN 平均聚合的 ASCENT 算法，利用节点级别的度校正提升聚类效果。

**💡 创新点**

创新点在于：①提供了一个条件自由、纯谱的 DCSC 质量理论分析，揭示度异质性和聚类结构弱化对误分率的具体影响；②提出了 ASCENT，首次将节点级别的度校正与 GNN 的过平滑特性结合，既能在早期阶段提升聚类质量，又在过平滑后退化为传统 DCSC。

**🔧 技术方法**

技术手段包括：谱聚类的正则化拉普拉斯矩阵分解、特征向量重排与归一化、K‑Means 聚类、以及 GNN 的均值聚合实现节点级度校正；理论方面使用谱图理论推导误分节点上界；实验方面对比了多种 SC 与 GNN‑based 聚类方法。

**📊 数据集**

使用了 12 个真实图数据集（Caltech、Simmons、PolBlogs、BioGrid、PPI、Wiki、BlogCatalog、ogbn‑Protein、RoadCA、LiveJournal、ogbn‑Products、Orkut）和 2 组 LFR 合成图，用于评估聚类质量与效率。

**📈 对比分析**

方法通过与传统 SC 基线（NJW、SCORE、RSC、SCORE+、ISC、IBHCD、CDBH）以及 GNN‑based 基线（GraphEncoder、SDCN、MinCutPool、DMoN、S3GC、MAGI）进行对比，利用 NMI、准确率、平均导电率等指标，ASCENT 在大多数数据集上取得了最高或相近最高的 NMI/AC，平均导电率明显下降，且整体运行时与传统 SC 接近，远快于 GNN‑based 方法。

**⚠️ 局限性**

局限性包括：仅考虑无属性的无向图；节点级度校正参数（θ、L）需手动调优，未实现可学习的校正；仅针对平均导电率（或归一化切分）目标，未覆盖模量最大化等其他聚类目标。

---

## 517. Towards Privacy-Preserving Federated Prompt Tuning under Data Heterogeneity: A Subspace-Decomposed Expert Approach

**arXiv ID:** 2607.21417 | [PDF](https://arxiv.org/pdf/2607.21417v1)

**作者:** Yuhua Wang `[一作]` (Beihang University), Zhiming Zheng `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FedSEPT，一种在联邦学习中实现多专家提示调优且具本地差分隐私的框架，能够在数据异质环境下兼顾个性化与全局泛化。

**💡 创新点**

创新点在于：1）子空间分解专家建模（SEM），将多专家提示分解为共享低秩因子+公共基准+本地残差，从而把通信与噪声限制在低维空间；2）实例感知专家融合（IEF），在本地动态路由并在 logits 级别进行融合，既能有效利用噪声专家，又避免高昂的文本编码成本。

**🔧 技术方法**

采用预训练 CLIP 视觉‑语言模型、低秩矩阵分解、共享公共基准、DP‑SGD 对共享因子加噪、实例级多头注意力路由、logit 级别软融合与梯度剪裁等技术。

**📊 数据集**

在 11 个公共基准上评估：Food101、Caltech101、Flowers102、DTD、OxfordPets、CIFAR‑10、CIFAR‑100、PACS、Office31、OfficeHome、DomainNet，分别构造标签偏斜、实用标签偏斜及域与标签混合偏斜三种异质设置。

**📈 对比分析**

与 PromptFL、FedPGP、FedOTP、FedPHA、pFedMoAP、DP‑FPL 等基线在同一本地 DP 预算下对比，FedSEPT 在所有 11 个基准的 HM（个人化与跨客户端泛化调和平均）上均领先，尤其在标签极端偏斜与域混合场景下提升 5–15% 的整体精度。

**⚠️ 局限性**

局限在于：1）仍需预先设定子空间秩、专家数等超参；2）在极低样本或极高异质环境下，多专家模型可能产生冗余或过拟合；3）对 GPU 内存仍有一定占用，尤其在缓存日志特征时需额外存储。

---

## 518. Logical Regression for Planning with Axioms

**arXiv ID:** 2607.21414 | [PDF](https://arxiv.org/pdf/2607.21414v1)

**作者:** Connor Little `[一作]` (Queen's University), Christian Muise `[通讯]` (Queen's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在包含公理（axioms）的完整可观测确定性规划域中对逻辑回归进行近似的方法，并将其应用于执行监测框架，以实现计划执行时的无重新规划恢复。

**💡 创新点**

创新点在于：1) 设计了三种近似回归（naïve、relevant variables、post‑processing）并引入“AAAR”公式；2) 通过将公理评估融入回归约简过程，避免了在每一步重新求解公理；3) 证明对目标状态进行一次性后处理即可获得与逐步后处理相近的效果。

**🔧 技术方法**

使用了逻辑回归理论、部分状态（partial state）表示、支持集（Support）计算、符号推理以及执行监测中的 suffix‑length 和 success‑rate 指标；实现上依赖 FastDownward 规划器生成计划，并在 Python/Prolog 等环境下实现回归近似与公理求值。

**📊 数据集**

实验数据集包括 IPC‑4（Promela‑Philosophers、Promela‑Optical‑Telegraphs、Miconic、PSR、Blocksworld、Sokoban）和 IPC‑23 的几个改造域（Labrynth、Quantum Circuit Layout Synthesis），共计约 200 余个实例。

**📈 对比分析**

比较方法：在四种配置（V1–V4）下对 10,000 次随机扰动执行监测；评价指标为 undefined‑rate、success‑rate（恢复率）和 suffix‑length。结果显示：post‑processing 方案在多数域上将 undefined‑rate 降低至约 30‑50%，恢复率可达 50‑70%，并且大多数情况下无需重新规划即可完成剩余步骤。

**⚠️ 局限性**

局限性包括：只适用于完全可观测的确定性域；不处理条件效应或非确定性效应；在存在分层公理的域中后处理方法不可用；实验域多为小型公开基准，缺乏大规模真实世界测试；以及在实际部署中对公理推理性能的依赖。

---

## 519. GS-Agent: Creating 4D Physical Worlds With Generative Simulation

**arXiv ID:** 2607.21522 | [PDF](https://arxiv.org/pdf/2607.21522v1)

**作者:** Hongxin Zhang `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过多智能体框架 GS-Agent，将自然语言描述转化为完整的 4D 物理世界，自动完成资产检索、材质调优、对象摆放、运动控制以及摄像灯光设置。

**💡 创新点**

创新点在于将物理引擎嵌入多智能体生成流程，利用 LLM 进行任务分解、自动化调参与错误恢复，并通过结构化工具接口实现与模拟器的高效协作。

**🔧 技术方法**

使用大语言模型（如 GPT‑4）、物理引擎（Genesis/MPM/SPH）、3D 资产检索与生成工具（BlenderKit、PolyHaven、Meshy）、多模态反馈工具和结构化 API。

**📊 数据集**

使用 NewtonGen 基准（24 场景）以及自制的 30 场复杂多物体交互与动态摄像场景。

**📈 对比分析**

与 Sora‑2、Wan2.2 文本到视频模型以及 SWE‑Agent（含视觉版）进行对比，采用 Physical Invariance Score、Alignment Score、VBench 美学指标和人工评估；GS‑Agent 在物理可信度、指令一致性和可控性上优于基线，Aesthetic 评分略逊。

**⚠️ 局限性**

受限于当前物理/渲染技术的精度与可扩展性，依赖 LLM 的运动与电影理解能力仍有限，且对实时性能和大规模并行生成的支持不足。

---

## 520. Improved lower bounds for the Shannon capacity of odd cycles

**arXiv ID:** 2607.21517 | [PDF](https://arxiv.org/pdf/2607.21517v1)

**作者:** Nathaniel Itty `[一作]` (Worcester Polytechnic Institute), Daniel Reichman `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文通过构造强乘积中更大的独立集，改进了奇数循环图 C₇、C₁₁、C₁₃ 的 Shannon 容量下界。

**💡 创新点**

创新点在于利用大型语言模型（LLM）进行交互式搜索，自动发现新的独立集构造方法，显著突破了传统人工搜索的效果。

**🔧 技术方法**

使用了 ChatGPT‑5.6 Sol Pro 生成搜索程序、执行并返回独立集向量，并结合手工验证（图相邻判定）保证构造合法。

**📊 数据集**

主要数据集为 LLM 生成的独立集向量集合（C₇⁺¹⁰ 134753 组、C₁₁⁶ 21909 组、C₁₃⁶ 62530 组），代码与数据公开于 GitHub。

**📈 对比分析**

通过与之前已知的下界（如 C₇⁵ 367、C₁₁³ 148、C₁₃³ 247）对比，分别提升了 Shannon 容量下界到 3.258020、5.289773、6.300109；相对提升幅度虽小但为奇数循环提供了新的最佳下界。

**⚠️ 局限性**

局限性包括：仍无法获得精确容量值；LLM 生成的构造需要人工验证；仅针对特定奇数循环和乘积阶数，推广性有限。

---

## 521. Top-down = Bottom-up: Sound and Complete Characterisations of Liveness by Multiparty Global Protocols

**arXiv ID:** 2607.21489 | [PDF](https://arxiv.org/pdf/2607.21489v1)

**作者:** Kai Pischke `[一作]` (University of Oxford), Nobuko Yoshida `[通讯]` (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了多方会话类型（MPST）中自顶向下（从全局类型投影得到本地类型）与自底向上（直接检查本地类型满足安全与活性）两种方法在满足活性属性时的可类型化表达式完全相同，并给出了从任何安全且活跃的本地类型上下文推导全局类型的主全局类型推断算法；

**💡 创新点**

创新点包括：①证明了在使用精确同步子类型、平衡性与共归全合并投影（coinductive full‑merging projection）时，自顶向下方法与自底向上方法可类型化过程完全一致；②设计了可生成主全局类型的推断算法，解决了“从本地到全局”合成的难题；③实现了完整的工具链，支持四种投影算法和本地类型推导；

**🔧 技术方法**

采用了会话类型理论、共归全合并投影（coinductive full‑merging）、同步子类型判定、语义化的公平调度、图遍历求解递归绑定等技术；

**📊 数据集**

使用了论文中提到的多种典型协议实例（Travel Agency、OAuth、Ring、Monte Carlo、MapReduce、Independent Workers 等）以及已公开的基准集合（Coinductive Full、Binary Counter 等）进行评测；

**📈 对比分析**

通过对比顶向下和底向上两种方法在同一组协议上的投影成功率、运行时间及推导出的全局类型大小进行评估，结果显示顶向下方法（尤其使用共归全合并投影）在大多数案例上速度快于底向上模型检查，并且能够投影所有可测试协议；

**⚠️ 局限性**

局限性包括：①理论仅在同步子类型、平衡性和共归投影下成立，异步情形下仍不可判定；②推断算法在处理极大或极复杂协议时可能产生指数级别的全局类型；③对多会话、动态会话初始和委托等功能未覆盖。

---

## 522. Future Rendering $\neq$ Future Surface: A Benchmark and Dataset for Dynamic Surface Reconstruction Beyond the Observed Window

**arXiv ID:** 2607.21471 | [PDF](https://arxiv.org/pdf/2607.21471v1)

**作者:** Yukun Shi `[一作]` (University of Guelph), Minglun Gong `[通讯]` (University of Guelph)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FutureSurf基准，评估视频序列中未来时间点的三维表面重建质量。

**💡 创新点**

创新点在于提供可控的解析运动数据集、未来窗口拆分、基于Chamfer的无对应表面评估、可测量的未来/观察窗口误差比以及恢复性测试。

**🔧 技术方法**

使用基于Chamfer距离的无对应评估、Sim(3)去除全局变换、Oracle规则恢复、以及DG‑Mesh和Deformable‑3DGS等动态高斯/三维场骨干。

**📊 数据集**

采用八个人工解析控制运动（含三种伪真控制）以及六个DG‑Mesh资产场景，所有数据均附带每帧精确网格。

**📈 对比分析**

在DG‑Mesh和Deformable‑3DGS上进行比较，观察窗口误差低，但未来窗口误差提升2–6倍，显示模型在时间外推方面表现不佳，且渲染指标与表面误差弱相关。

**⚠️ 局限性**

局限性包括仅使用合成数据、仅测试具备时间条件的Deformable MLP骨干、对单目轨道摄像机的假设，以及未涵盖更复杂物理驱动或全局时间建模方法。

---

## 523. Detecting LLM-Generated Tokens in Human--LLM Coauthored Text

**arXiv ID:** 2607.21458 | [PDF](https://arxiv.org/pdf/2607.21458v1)

**作者:** Yangjun Lu `[一作]` (University of Birmingham), Jin Zhu `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在无监督条件下，利用已有token级检测分数，通过局部核平滑并使用Lepski型自适应带宽选择，来定位人类与LLM混合写作文本中LLM生成的token。

**💡 创新点**

创新点在于将token级分数局部平滑与自适应带宽机制结合，既能降低噪声，又能自适应不同作者区块，且无需额外token级标注，理论上给出均方误差的上界并证明近似最优。

**🔧 技术方法**

核心技术包括基于Fast‑DetectGPT/AdaDetectGPT的token分数构造、对称核权重的平滑（如三角核），以及Lepski法的带宽自适应选择，后者通过构造置信区间实现局部偏差-方差折衷。

**📊 数据集**

实验使用三类合成数据集（Writing、XSUM、ESSAY）结合Gemini‑3.1‑flash‑lite、Grok‑4.1‑FNR、Claude‑4.5‑haiku等LLM生成文本；同时在真实收集的CoAuthor数据集上评估，包含1,445篇由人类与LLM共同完成的文档。

**📈 对比分析**

与多种基线（ML型DeBERTa、SegFormer、TriBERT；分割型TextTiling、PaLD；句子/段落级SenPred）进行对比，采用token‑级AUC衡量，结果显示在9种设置中获得8/9场景最高AUC，整体优于所有无监督或句子级基线，且在合成与真实数据上均保持鲁棒性。

**⚠️ 局限性**

局限性包括对token分数可分离性的假设；在分隔区块不连续或LLM输出被人类过度编辑时可能受限；方法对极端温度或攻击性编辑的鲁棒性尚未完全验证；计算成本虽可接受但相较于纯句子级方法略高。

---

## 524. RUMBA: Russian User Memory Benchmark

**arXiv ID:** 2607.21447 | [PDF](https://arxiv.org/pdf/2607.21447v1)

**作者:** Elizaveta Shevtsova `[一作]` (DAIMLD), Alena Fenogenova `[通讯]` (DAIMLD)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了RUMBA基准，用于评估俄语长期对话记忆能力，并提供了对应的英文子集；

**💡 创新点**

创新点在于构建多维细粒度记忆问题分类（语义、数量、时间维度），并加入忘记场景与时间表达显式/隐式区分；

**🔧 技术方法**

采用了检索增强记忆系统（RAG）与全上下文推理模型相结合的统一评估管道，利用LLM‑as‑Judge进行答案评判；

**📊 数据集**

使用了1,543对问答和约85条对话的俄语数据集，包含约340k字符对话和对应的时间戳信息，并通过自动翻译+人工校对生成英文版本；

**📈 对比分析**

在多种方法（包括向量检索、图检索、改进记忆框架、以及1M–2M token长上下文模型）上进行比较，发现全上下文模型总体优于RAG，单会话优于多会话，时间显式问题易于回答，且RAG在多会话、隐式时间表达等维度表现较弱；

**⚠️ 局限性**

局限包括数据量仍有限、推理题型和时间复杂问题偏少、英文版本为机器翻译且可能缺乏文化自然性、评估仅覆盖问答正确性未深入内部记忆机制等。

---

## 525. DAPM: UAV Monocular Depth Estimation from Any Height, Pitch, Roll and FOV

**arXiv ID:** 2607.21438 | [PDF](https://arxiv.org/pdf/2607.21438v1)

**作者:** Tong Ling `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Xian Sun `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DAPM 框架，实现了无人机视角任意的单目深度估计与相机位姿联合估计。

**💡 创新点**

创新点包括：基于几何耦合的视角–深度关系，推出理想地面深度 (IGD) 模块用于密集监督和深度特征增强；逐层量化的 Progressive Quantization Bins (PQB) 模块用于自适应精细化深度与位姿预测；并构建了覆盖连续视角的 UAPD 数据集。

**🔧 技术方法**

使用 ResNet50 编码器、FPN+残差解码器；IGD 模块将位姿映射为理想地面深度并作为监督与特征；PQB 通过多级 16/64 量化 bin 进行粗到细的深度与位姿推断；结合多任务损失（量化、回归、IGD 约束）。

**📊 数据集**

使用新建的 UAPD（42k 图像，连续高度、俯仰、滚转、FOV）进行训练与测试；同时与 Mid‑Air、SynDrone、SkyScenes、DDOS、UEMM‑Air 等现有 UAV 数据集对比；在 Potsdam 进行跨域验证。

**📈 对比分析**

在 UAPD 上，DAPM 在 δ1/δ2/δ3、AbsRel、RMSE、log10 等指标均超过 BPTS、DPT、AdaBins、BinsFormer 等主流方法；在 Potsdam 上同样取得最优或相近成绩；在相机位姿估计上，DAPM 的 Roll/Pitch/FoV/Height 中值误差分别低于 DeepCalib、Perspective Fields、GeoCalib，表现出 30%+ 的误差下降。

**⚠️ 局限性**

目前模型主要在仿真环境及城市/农田场景验证，缺乏真实无人机数据和更复杂地形的评估；对极端光照、遮挡、云层等环境的鲁棒性尚未充分验证；模型对极端俯仰/滚转角度仍可能产生误差。

---

## 526. Adaptive Identity Anchoring: Closed-Loop Keyframe Placement for Synthetic Paired Supervision in Video Face Swapping

**arXiv ID:** 2607.21434 | [PDF](https://arxiv.org/pdf/2607.21434v1)

**作者:** Logan Robbins `[一作]` `[通讯]` (Independent researcher), Logan Robbins (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Adaptive Identity Anchoring (AIA)机制，在视频面部替换的数据工厂中将锚点从固定两点扩展到可变数量并通过闭环评分自适应放置，并结合Reality-Referenced Texture Restoration (RTR)提升皮肤细节；

**💡 创新点**

①将锚点密度作为可调节的质量把手；②通过闭环身份评分自动决定锚点位置；③使用真实参考身份和视频自身频谱作为评判者；④输出可检验的配对证书和难度标签，实现质量与计算的权衡；

**🔧 技术方法**

Diffusion-forcing DiT视频合成器、Adaptive Pose-Attention、IFS（图像面部替换）、ArcFace/CurricularFace等身份编码器、光谱高频能量阈值检测、局部跨度再生成、匹配重粒化、频段分离纹理迁移等；

**📊 数据集**

主要使用DreamID‑V提供的真实视频与参考图像，实验中还使用IDBench‑V基准以及针对姿态、遮挡、长片等子集的数据；

**📈 对比分析**

通过对比原始两锚点Mint与AIA的最小身份相似度、接受率、流畅度，并在学生模型上评估IDBench‑V的身份相似度、帧级方差、姿态误差和FVD等指标；预期AIA在保持或提升身份一致性、降低方差的同时维持较高接受率；

**⚠️ 局限性**

评分器单点失效导致“Goodhart”问题、额外的IFS与再生成导致计算开销、对极难片段的锚点堆积难以解决、IFS质量限制可能导致锚点污染、纹理迁移的身份泄露风险、光谱统计易被误导导致伪精度、飞轮效应与数据分布偏移等限制。

---

## 527. Texture++: Elevating 3D Asset Texture Resolution with a Region-Aware Diffusion Model

**arXiv ID:** 2607.21504 | [PDF](https://arxiv.org/pdf/2607.21504v1)

**作者:** Shuaiwei Wang `[一作]` (Zhejiang University), Rengan Xie `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向任意UV映射的3D资产纹理超分辨率框架，利用多视角渲染、视角选择、区域掩码与局部扩散模型实现高质量纹理提升。

**💡 创新点**

创新点包括：1）自适应视角选择策略，既覆盖UV图集内部，又跨UV边界实现连续纹理；2）基于四叉树的掩码生成与正则化，保证局部更新边界平滑；3）局部扩散式SR模型，单步高频细节注入，避免全图迭代导致的模糊与不一致。

**🔧 技术方法**

技术实现：多视角渲染（Nvidiffrast）、自适应视角与UV边界匹配、四叉树掩码生成、基于Stable Diffusion的局部SR模型（加入LoRA、残差学习、掩码条件），以及纹理投影与更新机制。

**📊 数据集**

使用自制的局部SR训练数据集（基于真实3D资产的LR‑HR纹理对），具体数据集未公开。

**📈 对比分析**

与多种SOTA图像SR（DiffBIR、HYPIR等）和纹理生成（Text2Tex、Paint3D、MVPaint）方法在4×超分上进行定量与定性对比；在PSNR、SSIM、LPIPS、DISTS上均取得最高或接近最高分（PSNR 37.53、SSIM 0.9524、LPIPS 0.0637、DISTS 0.0736），并在时间上保持竞争力（94.4s）。

**⚠️ 局限性**

局限性：迭代过程仍较耗时，且对极端UV拆分或高度不规则网格的适应性尚待验证；模型对极低分辨率纹理的提升能力有限，且在特殊材质（如金属光泽）上效果未作深入评估。

---

## 528. From Resource Flow to Executable Tests: Petri-Net-Guided LLM Test Generation for Concurrent Stateful Rust APIs

**arXiv ID:** 2607.21530 | [PDF](https://arxiv.org/pdf/2607.21530v1)

**作者:** Kaiwen Zhang `[一作]` (Tongji University), Guanjun Liu `[通讯]` (Tongji University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Petri网的工作流，利用Petri网对并发状态化Rust库API的资源流、生命周期和冲突进行建模，然后将该抽象场景通过受约束的提示交给大型语言模型（LLM）合成可执行的Rust测试代码。

**💡 创新点**

创新点在于将语义意图与代码实现分离：Petri网负责语义建模、边界突变与并发曝光，LLM仅负责代码化；引入本地忠实性契约与结构修复循环保障模型与生成代码的契合；以及构造多层语义oracle区分生成错误与API错误。

**🔧 技术方法**

使用的技术包括：有颜色Petri网（建模资源和冲突）、LLM（如ChatGPT/CodeGeeX）在约束提示下生成Rust代码、Loom或Deterministic Scheduler进行并发探索、结构与语义oracle用于判定。

**📊 数据集**

在评估中使用了单一的受限Tokio MPSC（容量为1）的库实现，并手工构造相应的Petri网模型与场景；实验未涉及大规模变异库或广泛API。

**📈 对比分析**

实验仅展示了可执行性和结构保持，未与大规模基线（如纯LLM提示、传统模型测试）进行定量对比；性能表现为生成代码可编译、运行完整，且oracle判定通过，缺乏统计性能数据。

**⚠️ 局限性**

局限性包括：评估规模极小，仅单一API示例；需要手工编写Petri网模型与适配器；缺乏大规模基准和对比实验；LLM仍可能在语义细节上出现错误，需进一步自动化修复与验证。

---

## 529. 3D-Aware VLMs with Implicit and Explicit Geometries

**arXiv ID:** 2607.21595 | [PDF](https://arxiv.org/pdf/2607.21595v1)

**作者:** Wenhao Li `[一作]` (Nanyang Technological University), Gongjie Zhang `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出VLM-IE3D框架，利用RGB视频生成并融合隐式几何标记（IGT）和显式几何标记（EGT），从而提升VLM的3D空间理解与推理能力。

**💡 创新点**

创新点在于同时引入隐式与显式3D几何表示并通过3D-aware adapter进行融合，实现仅靠RGB视频即可获得粗略全局与细粒度局部几何知识，并不需要额外的3D输入。

**🔧 技术方法**

技术包括：AnySplat等3D几何编码器用于生成IGT；轻量MLP实现EGT嵌入；多头交叉注意力（IEA）融合IGT与EGT；Qwen2.5-VL-3B等VLM骨干网络。

**📊 数据集**

使用的数据集有Scan2Cap（3D密集描述）、ScanRefer（3D视觉定位）、EmbodiedScan（3D视频检测）、VSI-Bench与SPAR-7M（空间推理）、LLaVA-Video-178K等。

**📈 对比分析**

与多种基线（2D-VLM、3D-LLM、VG-LLM等）对比，VLM-IE3D在Scan2Cap的C@0.5达80.4、ScanRefer的Acc@0.25为43.2、3D视频检测的P_25为44.2等指标上均优于仅使用2D输入的模型，且接近或超过使用显式3D输入的方法。

**⚠️ 局限性**

局限性包括：对单一3D几何编码器的依赖，显式几何嵌入对重建误差敏感；相较于纯隐式模型计算量略增，推理速度略低；在更大规模、多模态或复杂环境下的鲁棒性尚未充分验证。

---

## 530. Unified Video Dense Prediction from Disjoint Data

**arXiv ID:** 2607.21592 | [PDF](https://arxiv.org/pdf/2607.21592v1)

**作者:** Yihong Sun `[一作]` (Adobe Research), Joon-Young Lee `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出一种统一的视频稠密预测框架，能够同时预测八种不同的场景属性（深度、法向、语义分割、边界、人体部位、反照率、阴影和材料），并且不需要各任务的联合标注。

**💡 创新点**

创新点在于通过预训练扩散模型提供的生成先验学习任务特定的潜在空间，并利用轻量级潜在投影器在不共享标注的情况下将多任务信息蒸馏到单一共享骨干网络，从而实现跨域、跨任务的高效一致推理；同时引入时序自注意力和时间梯度匹配实现视频的时间一致性。

**🔧 技术方法**

核心技术包括：Stable Diffusion v2 的潜在扩散模型、任务专用的 U‑Net 细化器和像素投影器、潜在蒸馏策略、扩展自注意力（ESA）和时间梯度匹配（TGM）等。

**📊 数据集**

使用了多种公开数据集：深度（NYUv2、ScanNet、DIODE）、法向（NYUv2、ScanNet、Sintel）、反照率与阴影（Hypersim、其他合成/真实混合集）、语义分割（Cityscapes、ADE20K、SA‑V 等）、实例边界、人体部位等；视频数据来自 RGB‑D 录像与公开视频集。

**📈 对比分析**

与单任务专家和现有统一模型（4M‑21‑XL、DICEPTION、StableMTL、冻结的 DINOv3‑H）对比，实验表明该方法在多项 OOD 评测中取得最优或接近最优的精度，并在视频推理时实现了显著的时间一致性和跨任务一致性；在相同的计算资源下速度更快、内存更低。

**⚠️ 局限性**

局限性包括：潜在重建对分类任务（如语义分割）效果不佳，需要额外的投影器微调；此外，模型仍需依赖强大的扩散预训练；在极端稀疏标注或高频动态场景下的鲁棒性尚未充分验证。

---

## 531. Expanding Flow Maps

**arXiv ID:** 2607.21585 | [PDF](https://arxiv.org/pdf/2607.21585v1)

**作者:** Sophia Tang `[一作]` (University of Pennsylvania), Pranam Chatterjee `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了可在推理过程中动态扩展维度的生成模型——扩展生成流（EFlow）与扩展流映射（EFM）

**💡 创新点**

将生成过程拆解为“扩展算子”与“传输映射”，构造可随时间增大维度的随机插值，兼容连续与离散可变尺寸数据，首次实现少步可变长度生成

**🔧 技术方法**

基于PDMP、流映射一致性条件、条件噪声插入与平均去噪器（mean denoiser），并结合CFM损失进行训练

**📊 数据集**

分子构型生成：GEOM‑QM9、GEOM‑Drugs；分子图生成：QM9；语言建模：One Billion Word（LM1B）

**📈 对比分析**

与多步扩散、图流、固定长度流模型对比，EFlow 在 20‑30 步时已匹敌或超越 500‑1000 步基线；EFM 在 1‑4 步内保持低 FCD；EFlow 在 LM1B 上在所有采样步数下均优于固定长度流模型，单步生成虽有模式坍塌但仍可生成连贯文本

**⚠️ 局限性**

仅支持维度递增插值；对极大尺寸（高原子数或长句子）扩展的可扩展性和采样策略需进一步调优；噪声分布假设为高斯，插值与插入时间调度需更多实验验证

---

## 532. Self-Supervised Learning of Structured Dynamics from Videos

**arXiv ID:** 2607.21576 | [PDF](https://arxiv.org/pdf/2607.21576v1)

**作者:** Lukas Knobel `[一作]` (UTN), Yuki M. Asano `[通讯]` (UTN)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在冻结的图像视觉变压器特征上构建一个结构化动态模型（SDM），从视频中学习分离主运动（如相机运动）和残差运动（如对象运动）的运动表示。

**💡 创新点**

在不对视频进行大规模预训练的前提下，利用弱场景级标注（相机静止/场景静止）以及自监督的未来特征预测，实现了主残差运动的分解，并通过两阶段补偿（primary + residual）显著提升运动表示的可解释性。

**🔧 技术方法**

使用冻结的DINOv2图像变压器做特征提取；在其上添加两层Transformer解码器分别生成主运动令牌p和残差令牌r；采用未来特征预测损失，并在有弱标签的合成数据上加入额外约束；对实时视频进行无标签自监督训练。

**📊 数据集**

主要使用合成数据Kubric（包含静止相机/静止场景的弱标注）以及真实视频数据SSv2和DL3DV；评估使用自研的ProbeMotion基准，涵盖摄像机运动、对象运动和两者混合的多种场景。

**📈 对比分析**

在ProbeMotion基准上，SDM在大多数任务上均优于直接使用冻结特征的基线（全局拼接或平均池化）以及自监督DeltaTok；在7个任务中5个实现了明显提升，且在3个任务上与强监督的3D几何模型（VGGT、DA3、Pi3X）相当。

**⚠️ 局限性**

局限性包括：对长时序运动的外推精度有限，无法可靠预测超过数帧的未来；需要合成数据提供弱标签来引导主/残差分解；在极其复杂或多目标的真实场景中，主/残差的分离可能不够清晰。

---

## 533. Where You Tap Matters: A Probe-and-Model Benchmark for Open-Set RF Fingerprinting

**arXiv ID:** 2607.21564 | [PDF](https://arxiv.org/pdf/2607.21564v1)

**作者:** Gabriele Oligeri `[一作]` (Hamad Bin Khalifa University), Fatima Al-Mousawi `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 GNU Radio 的 BPSK 接收链上，采集 5 个不同处理阶段（原始、Costas Loop、AGC、RRC、SymbolSync）的 IQ 样本，使用图像化的自编码器（AE）对已登记设备进行重构错误分类，评估开集设备指纹识别的性能，并通过 LLM 生成的多种 AE 结构进行对比。

**💡 创新点**

1) 明确指出接收链中不同采样点（probe）对 RFFI 性能的决定性影响；2) 在固定预处理与 MSE 决策规则下，对 LLM 设计的 AE 进行系统性对比，验证其对 probe 依赖性的鲁棒性；3) 提出“在接收链中何处采样”比 AE 结构更重要的实证结论。

**🔧 技术方法**

图像化 IQ 处理（32×32 直方图）、自编码器（浅层和 LLM 生成的深层/卷积 AE）、重构误差（MSE）、ROC/TAR–FAR 分析、LLM 代码生成（GPT5.2、Claude、GLM5 等）。

**📊 数据集**

12 台 USRP B200-mini-i 发送器，配 5 个 probe 的 IQ 数据集；每台发送器 5 次 10 分钟采样，约 1.53 亿 IQ 样本；数据在 5 个不同的处理阶段保存为文件。

**📈 对比分析**

采用单一设备单 probe 的“单一登记”方式，利用重构误差阈值绘制 TAR–FAR 曲线；以 TAR=0.9 为基准比较各 probe 与 LLM 设计的 FAR；结果显示 timing recovery（probe e）和 carrier recovery（probe c）在 TAR=0.9 时 FAR 可低至 0.01-0.05，而其他 probe FAR>0.1；LLM 生成的 AE 绝大多数未提升性能，训练时间显著增加。

**⚠️ 局限性**

仅在受控有线环境、BPSK 调制、固定 32×32 直方图表示、固定训练/测试分割；未考虑多模、射频通道变化、过天传输、对抗攻击或伪造攻击；probe 排名可能随硬件、调制、量化、接收实现等改变；未评估跨设备/时间/环境的模型迁移和鲁棒性。

---

## 534. OpenForgeRL: Train Harness-native Agents in Any Environment

**arXiv ID:** 2607.21557 | [PDF](https://arxiv.org/pdf/2607.21557v1)

**作者:** Xiao Yu `[一作]` (Columbia University), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为OpenForge的开源框架，能够在真实的推理harness中端到端训练LLM/VLM代理，并通过轻量化代理和Kubernetes orchestrator实现推理与训练的解耦；

**💡 创新点**

创新点在于将推理harness与标准RL训练完全解耦，支持任意harness与任意环境的训练，且提供自动化任务合成管线，首次实现跨harness泛化与行为分析；

**🔧 技术方法**

采用了代理+容器化rollout、Kubernetes orchestrator、GRPO/PPO等RL算法、SFT distillation、vLLM等LLM推理服务器、视觉+文本多模态感知以及自动任务合成pipeline；

**📊 数据集**

使用了自动合成的Claw任务、GUI（computer‑use 与 browser‑use）任务以及公开基准ClawEval、QwenClawBench、MCPAtlas、OSWorld‑Verified、Online‑Mind2Web、WebVoyager；

**📈 对比分析**

通过pass^3 / pass@3 / pass@1 等指标与同尺寸公开模型对比，OpenForge在ClawEval取得31.7 pass^3、QwenClawBench 33.7、MCPAtlas 28.1；在GUI基准上分别取得37.7/63.0/72.3，均优于同尺寸基线并部分超越更大模型；

**⚠️ 局限性**

局限性包括：仍难以有效学习错误恢复能力；对复杂harness训练仍需大量任务；代理与容器化对计算资源消耗较大；缺乏专门针对鲁棒性和错误恢复的训练策略。

---

## 535. SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation

**arXiv ID:** 2607.21553 | [PDF](https://arxiv.org/pdf/2607.21553v1)

**作者:** Junsong Chen `[一作]` (NVIDIA), Enze Xie `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种以多数线性注意力（75%）+周期性softmax锚点（25%）为核心的混合注意力视频扩散变换器（Hybrid Video DiT），并通过Block Attention Residuals实现跨层信息复用，能够在保持高质量的同时显著提升长序列、高分辨率视频生成的效率；

**💡 创新点**

核心创新点包括：① 75%线性+25%softmax的混合注意力布局，解决纯线性注意力的表达瓶颈；② 引入Block Attention Residuals共享路由，跨层利用锚点更新；③ 从零开始训练、优化全流程（数据清洗、分辨率/时长递进、流匹配、偏好DPO与ReFL后训练），并在同构硬件上实现高效的低精度推理与Sol-Engine部署；

**🔧 技术方法**

技术手段：视频扩散变换器（Video DiT）、线性注意力（Gated Bilinear Linear）、周期性softmax锚点、Block Attention Residuals（共享路由）、RoPE旋转嵌入、SwiGLU FFN、流匹配（logit-normal）、自监督Flow‑Distillation、偏好优化（Diffusion‑DPO）、强化学习后训练（ReFL）、低精度QAT（MXFP4/8）、Sol-Engine多阶段加速（核融合、残差缓存、稀疏软锚），以及大规模GPU集群训练；

**📊 数据集**

主要数据集为公开大规模原始视频语料库（含多轴质量/运动评估），使用LTX‑VAE 2.3潜在空间、Gemma‑2‑2B‑IT文本特征，进行从 480p/81帧 到 720p/193帧 的递进训练；

**📈 对比分析**

在VBench基准上，5B模型在 480×832×81、40步采样下获得Total 84.30分，单H100推理仅需13.2s，比全softmax的720p/60s模型快3.2×；14B模型在同速下达到Total 84.30（5B）/84.48（193帧）/85.29（121帧），在相同硬件与步骤下速度提升超过4×，并在多模型对比中实现优于大多数同类软max模型但参数量更小的优势；

**⚠️ 局限性**

局限性包括：① 混合注意力的收益主要体现在长序列，当前训练时长仅覆盖至8s；② 采用双向注意力，未实现实时因果推理；③ 需要从零开始训练，对硬件与资源要求高；④ softmax锚点仍占用一定计算，尚未实现完全稀疏化或专用低精度实现；⑤ 在极高分辨率或极长时长视频上的可扩展性和稳定性尚待验证。

---

## 536. Seeking Help in the Digital Age: A Cross-Platform Analysis of Online Support Systems for Technology-Facilitated Abuse Victims

**arXiv ID:** 2607.21549 | [PDF](https://arxiv.org/pdf/2607.21549v1)

**作者:** Nowshin Tabassum `[一作]` (University of Texas at Arlington), Shirin Nilizadeh `[通讯]` (University of Texas at Arlington)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 Google 搜索、Reddit 论坛和对话式 AI（通用大语言模型与专业聊天机器人）进行统一的技术与社会维度评估，构建并利用从 r/Stalking 社区抽取的真实受害者提问数据，对数字技术滥用（TFA）受害者的在线求助生态系统进行大规模横向比较。

**💡 创新点**

创新点：① 提出了跨平台统一评估框架，涵盖技术质量（相关性、准确性、可操作性、说服力、可读性）与社会安全维度（社会工程风险、毒性、同理心、语音选择、偏见、风险感知、支持信息）。② 通过手工编码、LLM 辅助分类构建了 11 类技术滥用的受害者提问数据集，并使用 LLM‑as‑a‑Judge 自动化扩展评估。③ 发现专业化受害者聊天机器人往往落后于通用大语言模型，并揭示搜索结果中恶意链接、Reddit toxic 回复等安全隐患。

**🔧 技术方法**

使用技术包括：文本抽取与提问识别（Llama3.3、GPT‑4）、多标签技术滥用分类（GPT‑OSS 20B）、搜索结果抓取与内容清洗（Selenium、Trafilatura）、恶意链接检测（VirusTotal）、毒性评估（Perspective API）、LLM‑as‑a‑Judge（Llama3.3、GPT‑OSS、Gemma）、手工标注与交叉验证。

**📊 数据集**

数据集：来自 r/Stalking 的 3,266 篇帖子和 32,162 条评论；通过 LLM 提问抽取得到 5,926 个提问，其中 2,797 个为受害者技术滥用相关提问；对应的 27,970 个搜索结果链接、17,714 条 Reddit 评论和 250 条 LLM/聊天机器人问答对。

**📈 对比分析**

比较方法：对 2,797 个受害者查询在三类平台上分别采集答案后，使用统一指标进行评估。结果显示：Google 搜索相关率 91%，但 65% 的查询会遇到恶意链接；Reddit 相关率 52%，毒性率 20%；通用大语言模型相关率 95%+，但仍出现 20% 以上的有害建议；专业聊天机器人相关率低于 70%，且同理心、风险感知评分均明显偏低。

**⚠️ 局限性**

局限性：① 只覆盖 r/Stalking 语料，可能不代表其他受害者社区；② 提问抽取与分类依赖 LLM，可能漏检或误标；③ 社会维度评估仅在 250 个问答对手工完成，样本规模有限；④ 对话式 AI 交互基于单轮问答，未涵盖持续对话情境；⑤ 研究仅聚焦英文内容，缺乏跨语言通用性。

---

## 537. Towards Robust Iris Recognition Through Occlusion Identification and Conditional Diffusion-Based Reconstruction

**arXiv ID:** 2607.21545 | [PDF](https://arxiv.org/pdf/2607.21545v1)

**作者:** Kamrul Hasan `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个包含遮挡类型识别、条件扩散式重建和VGG19-HPMNet识别的三阶段虹膜识别框架，用于在遮挡下提升虹膜识别性能。

**💡 创新点**

创新点在于：①将遮挡类型作为条件引入扩散模型实现针对性重建；②在重建后通过水平金字塔映射提取全局与局部特征；③通过对比实验与bootstrap分析验证该策略相较于传统重建方法和直接识别的优势。

**🔧 技术方法**

使用了残差CNN进行遮挡识别，基于DDPM的条件扩散模型进行重建，VGG19-HPMNet进行特征提取和识别，并使用PE时间嵌入、MLP投影等技术。

**📊 数据集**

主要数据集为CASIA‑Iris‑Thousand（人工合成遮挡），并在UBIRIS.v2上进行跨数据集重建测试。

**📈 对比分析**

通过对比TT‑GAN、cGAN、I3FDM等重建方法以及VGG‑16、ResNet‑50、Inception‑V3、ViT等识别模型，实验显示重建后EER下降约0.41%，TAR和Rank‑1 IR均有所提升，尤其是VGG19‑HPMNet在EER上达到最低2.88%。

**⚠️ 局限性**

局限性包括：遮挡为人工合成且类别有限，未覆盖自然遮挡情况；重建结果虽逼真但不保证完全一致；扩散模型推理速度较慢，需进一步优化。

---

## 538. GraphVid: Interactive Graph-Controllable Video Generation

**arXiv ID:** 2607.21580 | [PDF](https://arxiv.org/pdf/2607.21580v1)

**作者:** Vedant Shah `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GraphVid，一个基于交互图的交互式图像到视频生成框架。

**💡 创新点**

通过将多实体交互建模为有向场景图并使用边感知图推理，实现无需轨迹的多对象精确控制。

**🔧 技术方法**

利用边感知 GNN、图到令牌适配器、冻结的 DiT diffusion Transformer 与 LoRA 参数高效微调。

**📊 数据集**

构建 GraphVid‑Bench，包含约 27K 条交互视频与对应的交互图。

**📈 对比分析**

在 MoveBench 等基准上，GraphVid 在 FID/FVD/PSNR/SSIM/EPE 上优于大多数轨迹或文本控制方法，且仅使用 0.6B 可训练参数。

**⚠️ 局限性**

受限于图结构稀疏导致注意力稀释，难以扩展到极大节点数；对非物理复杂交互的表达仍有限。

---

## 539. Synthetic data generation framework for quality control automation in gravure printing

**arXiv ID:** 2607.21577 | [PDF](https://arxiv.org/pdf/2607.21577v1)

**作者:** Korota Arsène Coulibaly `[一作]`, Andrea Trombin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对旋转凸版印刷的质量控制，提出了一套基于物理建模的合成数据生成框架，自动产生多类缺陷图像及对应像素级分割标注；

**💡 创新点**

创新点在于将缺陷产生的物理机制（如缺油、擦痕、错位、折痕）抽象成可调的仿真模型，并在同一流程中同步生成精确的分割掩码，实现零成本、零人工标注；

**🔧 技术方法**

主要技术包括：基于几何与光学原理的缺陷仿真算法、Perlin 噪声与色彩通道处理、边缘检测与颜色分离、以及使用 RF‑DETR（Large）进行实例分割训练；

**📊 数据集**

使用了 7,533 张由框架生成的合成图像作为训练集，并在真实生产线上采集的独立测试集上进行评估；

**📈 对比分析**

与真实数据训练的模型对比，纯合成训练的 RF‑DETR 在测试集上取得 mAP@50=80.9%、精度 85.6%、召回 78.3%、F1 分数 81.7%，说明合成数据能够有效桥接现实差距；

**⚠️ 局限性**

局限性包括：缺少对印刷机动力学（转速、张力等）参数的建模，模型对极端或未覆盖的缺陷类型可能泛化不足，且仅针对分割任务，未涉及目标检测或多任务联合学习。

---

## 540. Surprisal Theory is Tautological (without Rational Grounding)

**arXiv ID:** 2607.21574 | [PDF](https://arxiv.org/pdf/2607.21574v1)

**作者:** Ryan Cotterell `[一作]` `[通讯]` (ETH Zürich), Ryan Cotterell (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在本文中，作者通过形式化证明表明，若未对语言模型加以外部约束，惊讶理论（surprisal theory）会成为一种自相矛盾且不可证伪的命题；进一步检视并批判了长期以来被视为默认的“语料假设”（corpus assumption）及其衍生的“缩放假设”，并指出相关实证结果已对其构成了挑战；最后，作者提出基于理性主义（rationalist）原则的替代约束方案（如失真上下文惊讶、发展阶段可行语料等），以期为惊讶理论提供可检验的理论基础。

**💡 创新点**

创新点在于：①将生物学中适应度的倾向解释（propensity interpretation）映射到惊讶理论，提出了“谬论”论证框架；②证明任何非负的处理难度度量均可通过构造语言模型的方式与惊讶值相匹配，从而揭示惊讶理论在无约束时的不可证伪性；③揭露并论证了语料假设及其缩放假设的实证失效，为该领域提供了新的理论警示；④提出基于认知机制（如记忆衰减、工作记忆限制）的理性约束路径，为未来的惊讶理论修正指明方向。

**🔧 技术方法**

使用的技术包括：形式化语言模型与条件分布的数学构造；紧致性与期望长度的充分必要条件证明；最大似然估计与KL投影的理论分析；构造失真上下文惊讶模型的参数化与资源合理化；以及对已有实证研究结果的逻辑与统计推理。

**📊 数据集**

论文本身未使用新的实验数据；主要引用的实验与数据来源包括：Web 文本、Reddit 语料、CHILDES 对话数据、以及之前工作中的眼动、阅读时长与 fMRI 等测量。

**📈 对比分析**

由于该研究为概念性与数学论证性质，未开展新的实验或性能比较；讨论重点在于理论框架的可证伪性与约束条件的逻辑必要性，而非模型在特定数据集上的数值表现。

**⚠️ 局限性**

主要局限包括：①对语料假设被证伪的结论依赖于先前研究的实证结果，未来研究可能修正或细化该结论；②提出的理性约束方案尚未完整实现或在实际数据上验证；③构造的技术条件虽然理论上足够弱，但尚未在具体难度度量上进行经验检验；④未考虑优化过程的实际不可达性与梯度下降的局限；⑤与生物学适应度的类比虽具有结构启示，但其适用范围与细节仍待进一步探讨。

---

## 541. Scene Parameter Saliency via Differentiable Light Transport

**arXiv ID:** 2607.21562 | [PDF](https://arxiv.org/pdf/2607.21562v1)

**作者:** Linas Beresna `[一作]` (Simon Fraser University), Eugene Fiume `[通讯]` (Simon Fraser University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用可微分渲染器对场景参数求梯度，构建“度量显著图”（metric saliency map），揭示不同目标（如平均亮度、心理视觉眩光指标UGR、神经网络感知得分）对场景参数的敏感性。

**💡 创新点**

创新点在于：① 将梯度作为解释性工具而非仅用于优化；② 通过单次反向传播即可得到全局光传输下的参数-度量关系；③ 证明同一场景在不同目标下显著图差异显著，具备度量特异性。

**🔧 技术方法**

使用了可微分路径追踪渲染器 Mitsuba 3（PRB后向传播）、自动微分框架 DrJit、以及标准的光照与材质参数化；评估度量包括平均亮度、UGR、以及 ResNet‑50 的分类 logits。

**📊 数据集**

实验数据集主要为公开室内场景模型：餐厅、厨房、客厅（包含镜面、金属表面等），以及一些工业灯光/纹理模型，全部通过 Mitsuba 3 渲染。

**📈 对比分析**

通过对梯度排名的参数做小幅梯度下降步进进行验证，发现梯度排名与实际度量变化高度相关（尤其在UGR上呈严格单调关系）；相较传统参数扫描、Sobol 等多次评估方法，单次反向传播显著节省计算；在均值亮度上因噪声阈值导致低阶参数排名波动，但高阶参数仍表现稳健。

**⚠️ 局限性**

限制包括：① 采样方差导致低 spp 时梯度噪声大；② 需要度量可微分，硬阈值度量需平滑化；③ 只给出一阶局部敏感度，无法捕捉参数交互或约束；④ 目前仅支持均匀材质参数，空间变异参数的显著图仍待研究。

---

## 542. Beyond Sycophancy: Structured Resistance and Compliance in LLM Moral Reasoning

**arXiv ID:** 2607.21558 | [PDF](https://arxiv.org/pdf/2607.21558v1)

**作者:** Baihui Wang `[一作]` (University of Chicago), Bernard Koch `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一系列实验中研究大型语言模型在道德推理场景下对社会影响的抵抗与顺从机制。

**💡 创新点**

将顺从与抵抗划分为三维度——距离、来源归因和联盟结构，并将其视为可解释的认知过程，区别于传统单一的顺从度量。

**🔧 技术方法**

使用基于日志概率分布的行为测量、对话式提示设计、对抗提示以及统计回归分析等技术。

**📊 数据集**

利用78道道德困境（来自跨文化数据集和NeurIPS 2025发布的数据集）以及17道二元事实问题进行评估。

**📈 对比分析**

通过比较8款不同代模型在相同试题下的概率分布变化，发现新一代模型有更宽的接受窗口、在自我归因下更易顺从、在联盟压迫下更具抵抗力，评估以定性差异和统计显著性为主。

**⚠️ 局限性**

局限在于样本仅包含少数较弱模型、能力与发布时间混淆、缺乏因果机制解释，以及仅在道德问题上验证，可能不适用于事实推理场景。

---

## 543. MIRROR: Learning from the Other View for Multi-Modal Reasoning

**arXiv ID:** 2607.21552 | [PDF](https://arxiv.org/pdf/2607.21552v1)

**作者:** Wen Ye `[一作]` (University of Southern California), Xuezhe Ma `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视图自监督的多模态强化学习框架（MIRROR），利用同一几何问题的文本主导、图像主导和两者结合的三种视图，挑选出表现最好的视图作为教师，指导其他视图的学习。

**💡 创新点**

创新点在于将跨模态视图不一致性转化为自监督信号：动态挑选每个问题最强视图做教师，采用on‑policy reverse‑KL对学生进行软对齐，并用EMA教师避免训练漂移，从而显著提升视图一致性与准确率。

**🔧 技术方法**

技术包括：多模态强化学习（GRPO），on‑policy reverse‑KL知识蒸馏，EMA教师（Polyak平均），以及自适应视图教师选择。

**📊 数据集**

使用自建的 ODA‑Data（含文本、图像、图文三种视图的几何题），并在公开基准 GeoInt 与 MathVerse 上进行验证。

**📈 对比分析**

相对于基线（原始 Qwen3‑VL‑4B、Vision‑R1‑7B、PAPO‑7B、Vero‑8B）以及单模态和混合模态 GRPO，MIRROR 在 ODA‑Val、GeoInt 以及 MathVerse 上的 pass@k、平均判分均实现显著提升（例如 image pass@16 提升 11.38%，text pass@16 提升 4.44%），并在仅 2K 训练样本下击败更大规模的后训练模型。

**⚠️ 局限性**

局限性包括：仍依赖于高质量的视图对应数据；对非几何任务的泛化尚未验证；在极高维度长文本或多图场景下，reverse‑KL 及 EMA 机制可能需进一步调整；教师视图的选取仅基于最终奖励，忽略中间推理步骤的不确定性。

---

## 544. X$^3$-OPD: Distilling Reasoning into Large Audio-Language Models via On-Policy Alignment

**arXiv ID:** 2607.21550 | [PDF](https://arxiv.org/pdf/2607.21550v1)

**作者:** Dongjie Fu `[一作]` (Tencent Hunyuan), Tao Jin `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨模态的 on‑policy 蒸馏框架 X^3-OPD，用音频基础模型在自身音频感知条件下进行推理，并由冻结的文本教师对生成的链式推理轨迹进行动态纠正，显著提升音频语言模型的逻辑推理能力。

**💡 创新点**

创新点在于：①构建了覆盖文本推理、音频事件推理和情感话语推理的三层对称语料库，使教师监督能跨越所有与音频相关的推理场景；②在对齐时采用跨模态 on‑policy 蒸馏，将学生在实际音频感知状态下产生的轨迹与教师在对应文本上的 token‑level 评价相匹配，解决了传统离线蒸馏的跨模态曝光偏差与感知误差问题。

**🔧 技术方法**

技术包括：跨模态 on‑policy 蒸馏（基于多路径 roll‑out 与跨模态优势函数的策略梯度损失）；三层对称语料构建（文本→TTS、音频+字幕、对话+元字幕）；离线预热 SFT 与随后强化学习阶段；以及对学生音频编码器和 LLM 背骨的联合微调。

**📊 数据集**

使用了三大类数据集：① Tier‑1 文本推理转语音（来源于 Tulu3、NaturalReasoning，合成后通过 ASR 过滤）；② Tier‑2 真实音频与字幕（AudioCaps、Clotho；字幕由 LLM 重建并人工校验）；③ Tier‑3 语音对话与元字幕（IEMOCAP、MELD；加入情感与停顿标注）。

**📈 对比分析**

与基线（标准 SFT 与 GKD）对比，在 MMSU、MMAU、BIG Bench Audio 等主要音频推理基准上，X^3‑OPD 在整体得分上提升了约 4–6 分（如 BAB 从 87.9 提升至 93.6），且在语义与音频事件理解方面均获得明显进步；在跨域（WorldSense、DailyOmni）测试中，X^3‑OPD 几乎无能力衰退（仅 1–2 分），显著优于离线蒸馏方法导致的 5–6 分退化。

**⚠️ 局限性**

局限性包括：①语料库对复杂音乐和专用非语言声景的覆盖不足，导致纯感知维度（如 MMAU‑Music、MMSU‑Phonology）仍略逊；②当前监督仅来自文本教师的 token‑level log‑probability，无法充分捕捉纯音频或情感特征，可能偏向语义逻辑而忽视感知错误，未来需引入多维音频奖励或规则式信号。

---

## 545. The Boundaries of Automation: A Theory of Persistent Human Participation

**arXiv ID:** 2607.21547 | [PDF](https://arxiv.org/pdf/2607.21547v1)

**作者:** Fares Fourati `[一作]` (TU Darmstadt), Iryna Gurevych `[通讯]` (TU Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并讨论了人机协同构建（human–AI co‑construction）的三大支撑——技术/互补、规范/发展与目标出现（target emergence），并阐释了目标出现如何导致人机协同在AI极其强大时仍不可或缺。

**💡 创新点**

创新点在于将目标出现作为人机协同持续必要性的根本理由，并提供了目标出现的动态模型和从揭示、细化到构成的三级分类。

**🔧 技术方法**

主要使用理论建模与系统性分析，没有具体算法实现；通过数学符号描述交互过程与目标演化。

**📊 数据集**

未使用任何公开数据集；本工作为概念性框架而非实验验证。

**📈 对比分析**

无实验比较与性能评估；本文仅提出理论框架与设计启示，未进行实证测试。

**⚠️ 局限性**

局限性包括：缺乏实验验证以证明目标出现的普适性；对目标演化度量和多参与者情境的细化不足；对AI在不同领域实际实现的技术细节未给出。

---

## 546. Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context

**arXiv ID:** 2607.21535 | [PDF](https://arxiv.org/pdf/2607.21535v1)

**作者:** Alagappan Valliappan `[一作]` `[通讯]` (Nvidia), Alagappan Valliappan (Nvidia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在百万-token长上下文的自回归生成中，引入窗口化的多token预测(MTP)草稿头，限制草稿只访问最近的窗口和注意力sink，显著降低草稿阶段的KV读取成本。

**💡 创新点**

创新点是训练无关、无额外参数的窗口化策略，在保证输出分布不变的前提下，消除长上下文中草稿的O(S)成本，提升推理速度。

**🔧 技术方法**

使用了SGLang框架的KV缓存分页、StreamingLLM滑动窗口+注意力sink、Triton和FlashInfer草稿后端以及ROPE/YaRN位置编码技术。

**📊 数据集**

在三种架构（Qwen3.6-35B、Qwen3.5-122B、Nemotron-3-120B）上使用1M-token RULER、LongBench-v2和BABILong等长文本推理任务进行实验。

**📈 对比分析**

与原生MTP和无投机的稠密推理进行比较，窗口化MTP在单GPU单请求下每步成本降低28%-44%，端到端吞吐提升1.22-2.55倍，且在不同任务和模型上保持优势。

**⚠️ 局限性**

局限性包括对长上下文窗口大小的依赖、在较短上下文时收益消失、对不同后端性能的敏感性以及在FP8 KV量化时增益下降。

---

## 547. Beyond Episodic Evaluation: Memory Architectural Bottlenecks in Sequential Embodied Question Answering

**arXiv ID:** 2607.21571 | [PDF](https://arxiv.org/pdf/2607.21571v1)

**作者:** Zikui Cai `[一作]` (University of Maryland College Park), Furong Huang `[通讯]` (University of Maryland College Park)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Sequential‑EQA评估协议，将嵌入式问答从孤立的episodic模式转变为连续多问答场景，检验记忆结构对持续操作的影响。

**💡 创新点**

发现仅保持内部状态并不能保证知识累积，强调空间基准的3D视觉语义记忆是实现准确性与导航效率双提升的关键。

**🔧 技术方法**

采用基于预训练视觉‑语言模型（如Qwen3‑VL）、3D重建与结构化记忆技术，并对比ExploreEQA、MemoryEQA、UniNavid等现有架构。

**📊 数据集**

使用OpenEQA数据集重新组织的连续问答序列，并在模拟环境与真实移动机器人上进行实验验证。

**📈 对比分析**

在四种记忆架构中，3D‑Mem在连续评估下实现+33.3%准确率提升、+53.3%路径成本降低，其他架构在记忆保持后准确率几乎无提升。

**⚠️ 局限性**

局限在于仅评估了少数几种模型，缺乏更长时序或更复杂环境的验证，且对大规模视觉‑语言模型的依赖可能导致部署成本高昂。

---

## 548. Beyond Sufficiency: Time Series Explanation with Counterfactual Necessity

**arXiv ID:** 2607.21573 | [PDF](https://arxiv.org/pdf/2607.21573v1)

**作者:** Hongnan Ma `[一作]` (University of Bristol), Weiru Liu `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向时间序列分类器的解释框架TimePNS，兼顾子序列的充分性与必要性；

**💡 创新点**

创新点在于将必要性（Pearl的PNS）引入解释过程，通过在潜在因果空间进行对抗性干预来辨别真正关键的时间段；

**🔧 技术方法**

使用两阶段训练：第一阶段学习稀疏因果生成模型与充分性掩码，第二阶段通过潜在对抗干预产生必要性信号并指导门控机制；

**📊 数据集**

实验数据包括三类合成数据（FreqShapes、SeqComb‑MV、Low‑VAR）和三类UCR真实数据（Epilepsy、ERing、ArticularyWordRecognition）；

**📈 对比分析**

与TimeX、TimeX++、Dynamask、WINIT、CORTX、SGT+GRAD等基线对比，采用AUPRC/AUP/AUR、occlusion‑based AUROC/AUPRC以及Prediction Shift评估，TimePNS在所有指标上均显著优于基线；

**⚠️ 局限性**

主要限制是对抗干预和潜在空间推理的计算开销较大，未来工作需要进一步降低成本并扩展到更通用的时序模型。

---

## 549. UnDA: Unpaired Domain Alignment for Cross-Modal Knowledge Transfer in Medical Imaging

**arXiv ID:** 2607.21546 | [PDF](https://arxiv.org/pdf/2607.21546v1)

**作者:** Rafsan Jany `[一作]` (Korea Institute of Oriental Medicine), Abu Raihan Mostofa Kamal `[通讯]` (Islamic University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

实现了无配对跨模态知识迁移的分割框架 UnDA。

**💡 创新点**

创新点包括：背骨无关的对齐模块、基于不确定性的权重 OT 以及 ProtoNCE 维持全局判别。

**🔧 技术方法**

技术：注意力池化提取 class token、Uncertainty‑Weighted OT、ProtoNCE 对齐、两阶段训练。

**📊 数据集**

使用 BraTS 2023（T2‑FLAIR→T1‑Native）和 MM‑WHS（MRI→CT）数据集。

**📈 对比分析**

与全监督基准和现有跨模态方法对比，UnDA 在 Dice/HD95 上平均提升 8–12% / 约 15–20mm，且在所有结构上保持最优或次优性能。

**⚠️ 局限性**

局限：需要在更大规模与不同任务验证，且对源域不确定性估计依赖熵，在极度模糊样本中可能失效。

---

## 550. Zero-Flow Two-Sample Tests

**arXiv ID:** 2607.21542 | [PDF](https://arxiv.org/pdf/2607.21542v1)

**作者:** Yakun Wang `[一作]` (University of Bristol), Taiji Suzuki `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出零流两样本检验（ZF2ST），通过学习向量化中点速度（midpoint velocity）作为 witness，并采用样本分割与学生化对齐统计量实现有效检验。

**💡 创新点**

创新点在于将零流判据转化为可统计量化的零流不一致性（ZFD），并将 witness 学习与检验解耦，利用向量化对齐而非传统标量对比，提升对结构化差异的敏感度。

**🔧 技术方法**

技术上结合流匹配、向量场学习（可用神经网络或 RKHS）与正则化的信噪比优化（SNR），以及样本分割、学生化、配对符号翻转校准。

**📊 数据集**

主要在合成高维高斯混合（HDGM）、Blob 3x3 以及 MNIST 6/9 置信度污染等数据集上进行实验。

**📈 对比分析**

与基线 MMD-D、MMD-O、C2ST-S/L 进行对比，ZF2ST 在大多数实验中取得更高检验功效，特别是在结构化多峰差异和稀有污染情形下保持近乎满功效，而基线在高维或小样本时性能下降。

**⚠️ 局限性**

限制在于对极局部多模变化的敏感度较弱，随机交叉配对可能削弱局部结构，且目前 witness 类为深度网络，缺乏理论收敛保证和更通用的正则化选项。

---

## 551. Generative AI Availability, Grades, and Student Satisfaction at a Large University

**arXiv ID:** 2607.21534 | [PDF](https://arxiv.org/pdf/2607.21534v1)

**作者:** James M. Zumel Dumlao `[一作]` (University of Michigan), Misha Teplitskiy `[通讯]` (University of Michigan)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文使用差分中的差分（DiD）方法，利用2015‑2025年密歇根大学156,135名学生、87,936门课程以及课程评估数据，检验ChatGPT发布后，生成式人工智能（GenAI）是否导致更易受自动化影响的课程（主要依赖作业、论文等作业型评估）的成绩上升、学生满意度下降。

**💡 创新点**

创新点包括：①将课程对GenAI的易受影响度固定在2019年预疫情基准，以避免疫情和教学策略变化的混杂；②使用LLM（GPT‑5.1）自动化解析教材文本，提取评估类型和权重，并在525份人工标注样本上进行严格验证；③首次在大规模数据上同时考察成绩分布、失败/退课率和学生满意度，并通过学生先前学业表现分层分析潜在异质性。

**🔧 技术方法**

主要技术手段包括：LLM文本解析与自动标签、线性回归与两阶段固定效应（课程、学期、学生）、差分中的差分与事件研究、两种COVID‑19干扰处理（临时窗口和永久水平），以及聚类标准误。数据清洗与一致性检验也贯穿整个流程。

**📊 数据集**

数据集：University of Michigan 2015‑2025学术年记录，涵盖156,135名学生、87,936门课程（6,836门独立课程），包含课程大纲、学生成绩与退课记录以及2016‑2025年期的课程评估（自评理解、兴趣、相对工作量）。

**📈 对比分析**

比较方法：对比高易受影响课程与低易受影响课程在ChatGPT发布前后成绩与满意度的差异，控制课程、学期与学生固定效应。结果显示，平均成绩和失败/退课率差异几乎为零，且对学生满意度的影响亦无显著变化；即使考虑到COVID‑19的潜在偏移，GenAI的净效应仍在统计上不显著，表明当前阶段GenAI未显著改变成绩信号或学习体验。

**⚠️ 局限性**

局限性：①评估权重的LLM提取仍可能存在误差，导致测量误差放大；②COVID‑19的影响与GenAI的冲击在时间上重叠，需做假设性分离，可能未完全排除混杂；③课程评估的自选性和响应率变化可能导致样本选择偏差；④仅检验了最终成绩，未能直接衡量学生实际掌握度；⑤识别假设（平行趋势）在疫情期间受到挑战，因而结果的因果力度受限。

---

## 552. AXIS: A Growable Community-Driven Data Engine for Scalable Robot Manipulation

**arXiv ID:** 2607.21588 | [PDF](https://arxiv.org/pdf/2607.21588v1)

**作者:** Mengfei Zhao `[一作]` (Axis Robotics), Jiachen Li `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AXIS，一个可扩展、社区驱动的数据引擎与基准，支持浏览器式远程操作、自动任务生成、数据清洗与增强；

**💡 创新点**

创新点在于将数据集成长可测量化、任务与演示的持续扩展、统一的任务快照与评估协议；

**🔧 技术方法**

采用MuJoCo-WASM浏览器远程操作、IsaacSim视觉与物理增强、自动化轨迹清洗与过滤、VLA模型与传统模仿学习；

**📊 数据集**

使用AXIS数据集（207项任务、50K+轨迹）及对比的RoboCasa365模拟数据；

**📈 对比分析**

通过统一的AXIS评估套件比较VLA与传统政策，并进行数据规模的持续预训练实验，结果显示AXIS持续预训练提升整体成功率5.8%，相较RoboCasa365提升37.3%，在视觉与几何扰动上收益最大；

**⚠️ 局限性**

局限在仅针对仿真Franka桌面任务，仿真到真实迁移仍待解决，未来需扩展机器人、传感器、任务长度及主动失败驱动数据收集。

---

## 553. Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation

**arXiv ID:** 2607.21582 | [PDF](https://arxiv.org/pdf/2607.21582v1)

**作者:** Yu Qi `[一作]` (Northeastern University), Lawson L. S. Wong `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并诊断预训练机器人策略在微调过程中的指令因素偏置问题，提出基于因素的诊断框架及FDR、FDH指标；

**💡 创新点**

首次量化指令因素偏置，揭示颜色为最显著的因素、动词与尺寸偏低效，并基于诊断结果提出针对性数据采集策略；

**🔧 技术方法**

使用因子对比实验、Copeland 排序、Gemini‑2.5‑Flash 评估判定、模型无关的偏置感知采样方案；

**📊 数据集**

在ManiSkill的六维指令空间（颜色、动词、物体、尺寸、空间属性）上进行实验，基于六种基础策略进行评估；

**📈 对比分析**

与随机采样、全覆盖采样等基线相比，偏置感知采样在模拟与真实UR5机器人上均提升了约10–20%成功率，且在相同演示预算下能用一半数据获得相同或更高性能；

**⚠️ 局限性**

实验仅限于受控桌面操作场景，采样策略并非全局最优，且未验证在更复杂开放世界或多步任务中的通用性。

---

## 554. Unsupervised Consensus-Based Anomaly Detection for Spatiotemporal Malaria Incidence in Ghana

**arXiv ID:** 2607.21559 | [PDF](https://arxiv.org/pdf/2607.21559v1)

**作者:** T. Ansah-Narh `[一作]` (Ghana Space Science and Technology Institute, Ghana Atomic Energy Commission), Y. Asare Afrane `[通讯]` (University of Ghana)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用共识式无监督异常检测框架识别2014-2023年加纳区域级马尔堡病例记录中的异常传播模式。

**💡 创新点**

首次将多种无监督算法整合为共识检测，提高异常识别的鲁棒性并区分高负荷与异常频率两种风险维度。

**🔧 技术方法**

采用Isolation Forest、Local Outlier Factor、Autoencoder、Elliptic Envelope四种算法并通过一致性计分实现聚合。

**📊 数据集**

使用国家健康信息系统DHIMS2汇总的16个行政区级别每月病例数据（共1908条记录）。

**📈 对比分析**

与单一算法比较，聚合后强异常召回率提升约30%，误报率下降，表明共识方法在检出与临界阈值的平衡上更优。

**⚠️ 局限性**

未考虑空间自相关、缺乏因果解释、仅基于历史数据、未实时更新，需进一步结合气候与干预变量验证。

---

## 555. Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers

**arXiv ID:** 2607.21594 | [PDF](https://arxiv.org/pdf/2607.21594v1)

**作者:** Sicheng Mo `[一作]` (University of California, Los Angeles), Bolei Zhou `[通讯]` (University of California, Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个流式多代理视频扩散模型 WorldWeaver，利用可学习的世界状态寄存器在生成过程中持续维护跨代理共享的全局状态，从而实现交互式视频生成。

**💡 创新点**

创新点在于：①引入跨代理世界状态寄存器，使模型能在每一步更新并共享全局世界信息；②采用 Mixture-of-Transformers 将状态建模与帧生成分离，减少两者竞争；③结合代理状态、鸟瞰图、场景文本等多种监督信号，显式训练寄存器以保持逻辑一致性。

**🔧 技术方法**

核心技术包括自回归视频扩散、Self-Forcing 训练、流动匹配 (flow matching)、教师-学生预训练、Mixture-of-Transformers、以及多头监督（位置、鸟瞰、文本）等。

**📊 数据集**

使用了约 126 小时双代理 Minecraft 视频数据集，数据中包含代理状态信息、鸟瞰视角以及利用 Qwen2.5-VL-72B-Instruct 自动生成的场景文本；评估时在 Solaris 原始测试集上进行。

**📈 对比分析**

方法与基线 (Frame concat、Solaris) 在 VLM 准确率、FID 以及综合 WorldScore 上进行对比。WorldWeaver 在两代理 Minecraft 生成任务中获得 WorldScore 105.1，VLM 准确率和 FID 均显著提升，证明了跨代理状态寄存器与监督机制的有效性。

**⚠️ 局限性**

限制：模型高度依赖带有代理状态、鸟瞰图和文本标注的仿真数据；在真实世界中获取这些标签困难；此外，状态监督对模型性能贡献较大，缺乏足够监督时可能导致性能退化。

---

## 556. Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning

**arXiv ID:** 2607.21591 | [PDF](https://arxiv.org/pdf/2607.21591v1)

**作者:** Rogerio Guimaraes `[一作]` (California Institute of Technology), Pietro Perona `[通讯]` (California Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在扩散/流匹配模型中，通过前期大量种子探索并逐步剪枝的推理时间缩放方法，称为Progressive Seed Pruning (PSP)。

**💡 创新点**

创新点在于放宽常数内存约束，采用可变粒子计数与早期剪枝，利用中间估计奖励有效分配计算资源，无需随机性和梯度。

**🔧 技术方法**

技术包括：可变粒子数的粒子滤波/早剪枝策略、利用模型自带的清晰估计x̂0进行中间奖励评估、与黑盒奖励（如ImageReward）结合、确定性采样器（DDIM、Euler Discrete）。

**📊 数据集**

使用的评估数据集：GenEval提示集、Stable Diffusion v1.5、SDXL、SD 3.5；以及人类评价（Prolific）。

**📈 对比分析**

与Best-of-N、FK-Steering、DSearch、NTS、RBF、BFS、SVDD等基线在匹配计算预算下比较；PSP在GenEval分数、ImageReward奖励、HPS和人类评估上均优于这些方法，且随着计算预算提升性能持续增长。

**⚠️ 局限性**

局限：只适用于标量奖励，无法处理多目标或硬约束任务；对细节导向的目标（如美学质量）提升有限；依赖中间奖励的可靠性，若奖励早期不稳定则剪枝可能失误。

---

## 557. MedGame: Storytelling Gamification Empowered by Large Language Models for Medical Education

**arXiv ID:** 2607.21570 | [PDF](https://arxiv.org/pdf/2607.21570v1)

**作者:** Qian Wu `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedGame 框架，利用 LLM 将静态临床案例转化为结构化、可执行的故事化游戏，并构建 5,000 条病例基准与交互平台；

**💡 创新点**

创新点在于双引擎设计（医学叙事生成 + 叙事导演），将案例拆解为 Acts/Scenes/Decision Nodes，形成可直接执行的 DAG 计划，并提供完整评测指标与数据集；

**🔧 技术方法**

采用大型语言模型（OpenAI Claude、Gemini 等）进行任务特定微调（LoRA），并结合 Pydantic 结构校验、DAG 任务依赖、以及多模态生成工具（图像/音频/视频）实现交互；

**📊 数据集**

使用从 PubMed Central 提取的 5,000 条真实患者摘要（PMC‑Patients 数据集），按 8 个医学专科均衡划分；

**📈 对比分析**

通过对比商业前沿模型与微调后的开源模型，在结构有效性、故事适配、医学准确性、教育质量等维度评测；实验表明微调后开源模型在结构与内容上接近商业模型，但医学准确度仍略低；多模态渲染相比文本版提升学生投入度与感知价值；

**⚠️ 局限性**

局限性包括缺乏长期学习成效验证、对外部多模态生成器性能未系统评估、仅关注生成质量而非实际临床决策；

---

## 558. Graph Learning on Ensembles of Cyclic Peptides: An Investigation of Molecular Ensemble Modeling

**arXiv ID:** 2607.21561 | [PDF](https://arxiv.org/pdf/2607.21561v1)

**作者:** Aaron Feller `[一作]` (University of Texas at Austin), Maxim Secor `[通讯]` (Novo Nordisk)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 EnsembleEGNN，一种利用共享 EGNN 层和 Set Attention Block 对循环肽多构象集合进行编码的基础模型，并通过自监督预训练学习构象集的热力学信息，最终用于预测膜渗透性等属性。

**💡 创新点**

创新点包括：①在构象集合层面使用共享 EGNN 进行逐构象编码并通过注意力聚合生成单一嵌入；②设计多任务自监督预训练目标（掩码标记、噪声坐标恢复、距离重建），利用 Boltzmann 权重引导注意力；③将此几何模型与 BERT 序列编码器联合训练，显著提升性能。

**🔧 技术方法**

采用的技术包括 Equivariant Graph Neural Network (EGNN)、Set Attention Block (SAB) 进行聚合、Boltzmann 加权注意力、掩码标记恢复、坐标噪声恢复、距离重建等多任务自监督学习，以及与 PeptideCLM‑2 BERT 模型的混合联合训练。

**📊 数据集**

使用的数据集为 CREMP（循环肽构象集合）进行预训练，CREMP‑CycPeptMPDB（约 3000 条膜渗透性标注数据）进行下游回归评估。

**📈 对比分析**

与 BERT‑only、随机初始化的 EnsembleEGNN、预训练 EnsembleEGNN、以及混合 Hybrid 模型进行对比；预训练 EnsembleEGNN 在 R²=0.477、Pearson r=0.699 处优于 BERT‑only（R²=0.439、r=0.667），Hybrid 模型进一步提升至 R²=0.538、r=0.737；随机初始化模型几乎无预测能力（R²≈0.005）。

**⚠️ 局限性**

局限性包括：①跨构象交互仅通过注意力聚合实现，未实现构象间显式消息传递；②依赖 CREMP 构象集合的覆盖率和精度；③数据量有限（约 3000 条样本），对模型泛化能力和跨数据集表现仍需验证。

---

## 559. Visual Contrastive Self-Distillation

**arXiv ID:** 2607.21556 | [PDF](https://arxiv.org/pdf/2607.21556v1)

**作者:** Yijun Liang `[一作]` (University of Maryland), Di Fu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视觉-语言模型的自监督强化，提出通过在同一模型下使用原图和内容消除的对照图来构造教师目标，实现无外部教师的 on‑policy 自蒸馏。

**💡 创新点**

创新点在于：①仅利用输入条件（原图 vs. 内容消除图）产生目标对比，而不需要答案提示、视觉证据或外部验证；②使用 token‑级 log‑概率差异来对原图分布进行“对比衬色”，在保持语言连贯的前提下突出图像相关性；③引入可调的 plausibility 约束和对比强度参数，使蒸馏目标既保留教师的可信候选，又加强视觉驱动的选择。

**🔧 技术方法**

技术手段包括：EMA（指数移动平均）教师、对照图构造、token‑级 log‑概率对比、基于支持集的目标裁剪、正向 KL 蒸馏、以及对比强度/支持阈值的超参数调节。

**📊 数据集**

使用 ViRL39K 单图像数据集进行后训练，评估 7 个视觉-语言基准（BLINK、MMStar、V*Bench、HRBench4K/8K、MathVista、HallusionBench）。

**📈 对比分析**

与基线模型、已公布的 answer‑hint OPSD 进行对比；在 Qwen3‑VL 和 Qwen3.5 的 2B–9B 规模上，均实现平均准确率提升 1.8%–4.8%，在大多数基准上获得最优成绩，且在训练过程中表现更稳健、抗退化性更好。

**⚠️ 局限性**

局限性包括：①对比目标对模型原图分布的依赖，若原图分布已偏离真实视觉信息，改进有限；②对照图的内容消除方式需要手工设置（如全黑或噪声），可能对某些任务敏感；③仅在单图像场景验证，尚未证明对多图像或视频任务的适用性；④对超参数（α、β、温度）较敏感，需额外调优。

---

## 560. DONDO: Open w2v-BERT Speech-Recognition Base Models for African Languages

**arXiv ID:** 2607.21540 | [PDF](https://arxiv.org/pdf/2607.21540v1)

**作者:** Paul Azunre `[一作]` `[通讯]` (Khaya AI), Paul Azunre (Khaya AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了DONDO系列面向非洲语言的开放许可证w2v‑BERT 2.0基础ASR模型，包括21个单语模型和5个多语模型。

**💡 创新点**

通过两步学习率退火细调、轻量级前缀帧语言条件化以及可再现的训练流程，显著缩小多语模型与单语基线之间的性能差距。

**🔧 技术方法**

采用w2v‑BERT 2.0 Conformer编码器、CTC解码头、学习率退火细调以及前缀帧语言提示技术。

**📊 数据集**

主要使用宗教文本朗读语音及其经过人工校对的标准正字法转录，覆盖27种非洲语言。

**📈 对比分析**

在与各语言单语基线同一测试集上的WER比较中，双步退火后的多语模型平均WER降至10–13%，部分语言甚至优于单语模型。

**⚠️ 局限性**

受限于朗读宗教文本域，模型对自然对话、嘈杂或代码切换语音表现不足，并且需要手动指定目标语言。

---

