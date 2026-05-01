# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-01 | 今日论文总数: 596

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. TIO-SHACL: Comprehensive SHACL validation for TMF Intent Ontologies

**arXiv ID:** 2604.27359 | [PDF](https://arxiv.org/pdf/2604.27359v1)

**作者:** Jean Martins `[一作]` (Ericsson Research), Marin Orlic `[通讯]` (Ericsson Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并公开了针对TM Forum Intent Ontology（TIO）v3.6.0的完整 SHACL 验证框架 TIO-SHACL，覆盖所有 15 个模块，实现意图语义正确性验证。

**💡 创新点**

创新点包括：① 56 个节点形状、69 个属性形状实现 100% 词汇覆盖；② 25 个可参数化的 SHACL 约束组件与 3 个自定义目标类型，极大提升重用与可维护性；③ 针对 TIO 递归逻辑运算、量化比较、交叉期望继承等复杂结构提出新型验证模式；④ 通过 133 个正负样例的测试驱动开发，确保形状随 TIO 版本演进保持兼容。

**🔧 技术方法**

采用了 SHACL（核心+高级功能）作为验证语言，使用 SPARQL 查询实现自定义约束，配合 Python pySHACL、TopBraid、Apache Jena 三大验证器进行交叉验证；另外利用 mixin 扩展词汇和参数化查询实现复用。

**📊 数据集**

使用了 TM Forum 官方发布的 TIO v3.6.0 词汇（87 类、109 属性、72 个函数）以及作者自行构造的 133 条 Turtle 测试案例（67 正例、66 负例），覆盖所有模块和函数。

**📈 对比分析**

对三大验证器（pySHACL、TopBraid、Jena）进行功能一致性与性能对比；功能上实现 100% 一致通过；性能上 Jena 在 feature 级别比 pySHACL 快 4.2×、比 TopBraid 快 1.3×，pySHACL 运行时变异性最大。AF 级别在参数化查询下仅增加 <2% 开销。

**⚠️ 局限性**

局限性包括：① 需要依赖 SHACL 高级功能（pySHACL、TopBraid）才能充分利用可参数化约束，Jena 受限于 Core 级别；② 目前仅覆盖 TIO v3.6.0，后续版本若大幅变更需重新适配；③ 对于极大规模的意图图谱，pySHACL 的性能仍显著落后。

---

## 2. Heterogeneous Scientific Foundation Model Collaboration

**arXiv ID:** 2604.27351 | [PDF](https://arxiv.org/pdf/2604.27351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 3. Verification and Validation (V&V)-in-the-Loop for RISC-V Design: The Holistic Vision of BZL

**arXiv ID:** 2604.27013 | [PDF](https://arxiv.org/pdf/2604.27013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 4. Fisher Markets with Approximately Optimal Bundles and the Need for a PCP Theorem for PPAD

**arXiv ID:** 2604.27276 | [PDF](https://arxiv.org/pdf/2604.27276v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway), Themistoklis Melissourgos `[通讯]` (University of Essex)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5024301526)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明在 Fisher 市场中，若允许买家接受约 (1‑δ)-最佳的商品组合，即使所有买家预算相等、只使用线性上限效用，寻找满足此约束的竞争均衡仍是 PPAD‑难的，并给出了相应的归约。

**💡 创新点**

创新点在于首次将 PPAD‑难度与一个自然的约束松弛问题联系起来，并证明在此问题下常数 δ 的难度等价于已知的 PCP 形式的 Conjecture；此外，还改进了先前结果对清算误差的要求（从 1/11 到 1/9），并在 CEEI 设置下给出新的难度下界。

**🔧 技术方法**

主要技术是构造无参考商品、常数参数的线性上限效用市场实例，并通过可构造的 “逆向门” 与 “增减燃烧” 等流网络方法把 (ε,δ)‑电路问题的约束直接映射到价格与消费变量上；随后利用四步迭代（修正买家、处理过度/不足清算、燃烧与调价）将解从 (c,δ)‑近似均衡升格为 (0,δ)‑近似均衡。

**📊 数据集**

该工作完全是理论性质，未使用任何真实数据集；所有证明均基于构造性的 NP/PPAD 归约与严格的数学分析。

**📈 对比分析**

由于研究对象是计算复杂性，没有实验对照；论文通过证明归约相等性与逆向证明，说明在不存在更强归约的前提下该问题的难度与 Conjecture 的成立紧密相关，因而为该类问题的理论界限提供了新的基准。

**⚠️ 局限性**

主要局限在于常数 δ 的 PPAD‑难度仍需假设 Conjecture；在常数 δ 情况下的完整证明依赖该未被证实的猜想；仅当 δ 为逆多项式或禁用零效用商品时，能得到无条件难度；此外，所给的归约对构造的市场有严格的结构限制（如常数度数、预算相等、线性上限效用）。

---

## 5. Iterative Definition Refinement for Zero-Shot Classification via LLM-Based Semantic Prototype Optimization

**arXiv ID:** 2604.27335 | [PDF](https://arxiv.org/pdf/2604.27335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 6. Anomaly Detection in Soil Heavy Metal Contamination Using Unsupervised Learning for Environmental Risk Assessment

**arXiv ID:** 2604.27102 | [PDF](https://arxiv.org/pdf/2604.27102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 7. Not All Memories Age the Same: Autodiscovery of Adaptive Decay in Knowledge Graphs

**arXiv ID:** 2604.26970 | [PDF](https://arxiv.org/pdf/2604.26970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 8. Learning to Spend: Model Predictive Control for Budgeting under Non-Stationary Returns

**arXiv ID:** 2604.27186 | [PDF](https://arxiv.org/pdf/2604.27186v1)

**作者:** Nilavra Pathak `[一作]` (Expedia Group), Christopher Swartz `[通讯]` (McMaster University)

**通讯引用:** 1597 | [OpenAlex ID](https://openalex.org/A5013153870)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在非平稳环境下利用模型预测控制（MPC）进行季度营销预算分配，并与传统的反应式节奏分配进行对比。

**💡 创新点**

创新点在于将预算分配视为闭环控制问题，系统评估不同非平稳性（静态、随机漂移、可预测季节性）对MPC效果的影响，并证明仅在存在可预测结构时MPC才显著优于基线。

**🔧 技术方法**

采用模型预测控制、粒子滤波预测、SARIMAX时间序列预测、仿真生成的噪声执行层和Richards型响应模型。

**📊 数据集**

使用基于历史营销数据构建的合成仿真环境，模拟执行噪声、预算约束和时间变化的响应曲线。

**📈 对比分析**

通过配对蒙特卡洛实验，使用oracle-normalized收益和百分比差异衡量，与基线对比：在静态和随机漂移下几乎无优势，随机漂移略逊；在可预测季节性下提升4–14%，并与oracle收益相近。

**⚠️ 局限性**

局限在于仅在合成环境下验证，模型结构和预测误差受限；未考虑多渠道交互、真实大规模数据以及更复杂的预算约束。

---

## 9. CL-bench Life: Can Language Models Learn from Real-Life Context?

**arXiv ID:** 2604.27043 | [PDF](https://arxiv.org/pdf/2604.27043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 10. Beyond One-Size-Fits-All Exercises: Personalizing Computer Science Worksheets with Large Language Models

**arXiv ID:** 2604.27433 | [PDF](https://arxiv.org/pdf/2604.27433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 11. Fidelity, Diversity, and Privacy: A Multi-Dimensional LLM Evaluation for Clinical Data Augmentation

**arXiv ID:** 2604.27014 | [PDF](https://arxiv.org/pdf/2604.27014v1)

**作者:** Guillermo Iglesias `[一作]` (Universidad Politécnica de Madrid), Enrique Baca-Garcia `[通讯]` (University Hospital Jimenez Diaz Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了基于少样本提示的LLM数据增强流程，用于在ICD-10标签条件下生成高质量、可安全使用的精神状态评估（MSE）报告。

**💡 创新点**

创新点包括：①将严格JSON输出限制与少样本提示相结合，最大化模型输出的结构化一致性；②构建三维评估框架（语义忠实度、词汇多样性、隐私/抄袭率），超越传统单一指标；③在中等规模量化LLM上验证此流程能在保持隐私的前提下扩充临床文本。

**🔧 技术方法**

技术手段：LLaMA、Mixtral、Mistral等量化LLM；少样本（10例）提示；句子Transformer嵌入；MMD、BERTScore、sms、ROUGE、METEOR、Self-BLEU、ttr、nnd、抄袭率等多指标评估。

**📊 数据集**

数据集：本研究自建的25,803份精神科急诊MSE报告，涵盖94个ICD-10诊断标签，用于构造提示和评估基准。

**📈 对比分析**

比较方法：对三个模型使用相同的提示和评估框架，计算语义忠实度（MMD、BERTScore、sms）、表面相似度（ROUGE、METEOR）、词汇多样性（Self-BLEU、ttr）和隐私安全（nnd、抄袭率）。结果显示：MMD最低（0.012）、BERTScore最高（0.690），Mistral在词汇多样性上最高（ttr 0.991），所有模型的nnd>0.2、抄袭率<3%。

**⚠️ 局限性**

局限性：仅在精神科急诊文本上验证，未对下游分类任务的性能提升做定量评估；模型在极少样本或罕见诊断场景下可能仍出现模式崩塌或覆盖不足。

---

## 12. Optimal Stop-Loss and Take-Profit Parameterization for Autonomous Trading Agent Swarm

**arXiv ID:** 2604.27150 | [PDF](https://arxiv.org/pdf/2604.27150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 13. Designing Ethical Learning for Agentic AI: Toegye Yi Hwang's Ethical Emotion Regulation Framework

**arXiv ID:** 2604.26958 | [PDF](https://arxiv.org/pdf/2604.26958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 14. The Impact of LLM Self-Consistency and Reasoning Effort on Automated Scoring Accuracy and Cost

**arXiv ID:** 2604.26954 | [PDF](https://arxiv.org/pdf/2604.26954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 15. The Inverse-Wisdom Law: Architectural Tribalism and the Consensus Paradox in Agentic Swarms

**arXiv ID:** 2604.27274 | [PDF](https://arxiv.org/pdf/2604.27274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 16. JI-ADF: Joint-Individual Learning with Adaptive Decision Fusion for Multimodal Skin Lesion Classification

**arXiv ID:** 2604.27343 | [PDF](https://arxiv.org/pdf/2604.27343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 17. RCW-CIM: A Digital CIM-based LLM Accelerator with Read-Compute/Write

**arXiv ID:** 2604.27384 | [PDF](https://arxiv.org/pdf/2604.27384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 18. Unsupervised Electrofacies Classification and Porosity Characterization in the Offshore Keta Basin Using Wireline Logs

**arXiv ID:** 2604.27126 | [PDF](https://arxiv.org/pdf/2604.27126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 19. Profiles of AI Dependency: A Latent Class Analysis of Filipino Students' Academic Competencies

**arXiv ID:** 2604.27349 | [PDF](https://arxiv.org/pdf/2604.27349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 20. RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval Augmented Query Generation

**arXiv ID:** 2604.27244 | [PDF](https://arxiv.org/pdf/2604.27244v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 21. Improved Approximation Algorithm for Maximum Balanced Biclique

**arXiv ID:** 2604.27141 | [PDF](https://arxiv.org/pdf/2604.27141v1)

**作者:** Pasin Manurangsi `[一作]` `[通讯]` (Google Research), Pasin Manurangsi (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了最大平衡双团（MBB）问题，给出了一个多项式时间的近似算法，能够找到最大平衡双团。

**💡 创新点**

提出的近似比率为 n/Ω((log n)^3)，改进了之前的 n/Ω((log n)^2) 的近似比，并且与最大团问题的近似比相匹配。

**🔧 技术方法**

使用了随机化的多项式时间算法和半正定规划（SDP）技术。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与 Chalermsook 等人的方法进行了比较，性能上有显著提升，达到了 n/Ω((log n)^3) 的近似比。

**⚠️ 局限性**

在处理 MBB 的近似硬度方面仍然存在挑战，特别是在小最优解的情况下，是否可以直接通过组合算法处理仍然是一个开放问题。

---

## 22. Hyperspectral Image Classification via Efficient Global Spectral Supertoken Clustering

**arXiv ID:** 2604.27364 | [PDF](https://arxiv.org/pdf/2604.27364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 23. Emotion-Aware Clickbait Attack in Social Media

**arXiv ID:** 2604.27369 | [PDF](https://arxiv.org/pdf/2604.27369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 24. A Study on the Performance of Distributed Training of Data-driven CFD Simulations

**arXiv ID:** 2604.27431 | [PDF](https://arxiv.org/pdf/2604.27431v1)

**作者:** Sergio Iserte `[一作]` (Universitat Jaume I), Krzysztof Rojek `[通讯]` (Czestochowa University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一个基于RNN的时间序列预测模型，能与OpenFOAM CFD求解器结合，实现流场状态的快速预测，并对不同分布式训练策略（TensorFlow原生与Horovod）进行细粒度性能评估。

**💡 创新点**

创新点在于：① 将数据驱动的CFD加速与递归神经网络相结合，专门针对大规模网格的三维速度预测；② 对CPU与GPU多节点的分布式训练进行系统性对比，首次给出在同一硬件平台下最佳进程与节点布局的“sweet spot”；③ 提供了训练与推理两阶段的加速比与通信开销分析。

**🔧 技术方法**

使用技术包括：OpenFOAM 进行CFD求解；Keras/TensorFlow LSTM RNN；TensorFlow MirroredStrategy 与 MultiWorkerMirroredStrategy；Horovod + MPI + NCCL；IBM Power9 CPU 与 NVIDIA V100 GPU；MPI通信与InfiniBand网络。

**📊 数据集**

数据集：131个不同流入速率组合的仿真，共420个时步/案例，网格125,565格，三维速度字段，形状为131×420×125,565×3，约38.6 GB。

**📈 对比分析**

通过在CPU单机、Horovod多CPU节点、TensorFlow单GPU、Horovod多GPU节点等多种配置下测量训练时间与推理时间，得到GPU可将训练时间缩短7.1×、4节点3GPU相较CPU提升24.5×，推理GPU相较CPU提升3.09×；多GPU通信开销显著，12‑GPU配置在四节点下最优。

**⚠️ 局限性**

限制：模型规模较小导致多GPU加速受通信瓶颈限制；仅评估单步预测与单一CFD求解器，未覆盖更复杂或更大规模的流场；对不同网络协议的细粒度影响仅做了初步说明，需进一步深入。

---

## 25. NuggetIndex: Governed Atomic Retrieval for Maintainable RAG

**arXiv ID:** 2604.27306 | [PDF](https://arxiv.org/pdf/2604.27306v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Jelena Mitrovic `[通讯]` (University of Passau)

**通讯引用:** 659 | [OpenAlex ID](https://openalex.org/A5019466280)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了NuggetIndex检索系统，将检索单位从段落转为原子事实记录，并通过时间有效性区间、生命周期状态和冲突检测实现检索治理；

**💡 创新点**

创新点在于为检索单元引入可版本化的“nugget”数据模型，加入时间有效性区间、生命周期状态（Active/Deprecated/Contested）和冲突检测，并在检索前按时间与状态过滤，从而解决动态语料中的过时与争议信息问题；

**🔧 技术方法**

使用了LLM原子事实抽取、文本规范化、时间表达识别与修订历史推断、冲突检测算法、B‑tree元数据索引、BM25与稠密向量检索（HNSW）以及稀疏/密集融合、哈希ID、SQLite/PG存储等技术；

**📊 数据集**

实验基准包括RAVine（MS MARCO nuggetized）、TimeQA、SituatedQA、MuSiQue以及HotpotQA等四个公开数据集；

**📈 对比分析**

与传统段落检索、时间过滤、Proposition‑RAG、GraphRAG等方法对比，NuggetIndex在RAVine上nDCG@10提升至0.637（vs. 0.324），nugget recall提升42%；在TimeQA上时间正确率升至0.931（vs. 0.840）并将冲突率降低55%；生成输入长度缩减64%，检索延迟<1 ms；稀疏检索在原子事实上优于稠密检索；

**⚠️ 局限性**

局限性包括：时间终止检测召回仅约0.667，实体归一化召回约0.60，Jaccard相似度的去重缺乏语义泛化，冲突检测依赖来源独立性，无法处理无时间标记或深层推理任务，且对大规模动态语料的实时更新仍需改进。

---

## 26. Machine Collective Intelligence for Explainable Scientific Discovery

**arXiv ID:** 2604.27297 | [PDF](https://arxiv.org/pdf/2604.27297v1)

**作者:** Gyoung S. Na `[一作]` (Korea Research Institute of Chemical Technology), Chanyoung Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2070 | [OpenAlex ID](https://openalex.org/A5101629749)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了机器集体智能（MCI）框架，实现多 LLM 代理协同进化、知识积累和 AST 形式的符号回归，自动推导未知科学系统的支配方程。

**💡 创新点**

创新点在于将符号推理与基于元启发式的群体搜索结合，采用 AST 作为可解释的知识表示并引入发现分数以同时惩罚误差与复杂度，从而突破单体 LLM 的推理边界，实现可解释、可外推的科学发现。

**🔧 技术方法**

技术包括大语言模型（Mixtral-8x7B）推理代理、基于 AST 的符号表达、发现分数（误差+深度+参数）评估、知识聚合与传播、结构化 LLM 提示、符号回归演化算法。

**📊 数据集**

使用了十个跨物理、化学、生物学的基准符号回归任务（如 Chi2PDF、NDO、NNN、FHST、BDC、SFL、NOMC、ECBG、HHM 等），涵盖多种函数形式。

**📈 对比分析**

与 GPlearn、PySR、LLM‑SR 以及单体 MCI（MSI）对比，MCI 在所有任务的 WMAPE 均低于 0.1，错误下降 30‑100%；在 OOD 条件下仍保持 <0.1，显著优于 DNN（在 OOD 时误差 >1）和 LLM‑SR。

**⚠️ 局限性**

局限性包括：目前使用的精英主义知识传播策略可能限制搜索多样性；对极端 OOD 仍有误差；依赖公开 LLM（Mixtral）可能受限于其数学推理能力；未来需探索更灵活的知识传播机制和更强的代理。

---

## 27. BoostLoRA: Growing Effective Rank by Boosting Adapters

**arXiv ID:** 2604.27308 | [PDF](https://arxiv.org/pdf/2604.27308v1)

**作者:** Raviteja Anantha `[一作]` (Amazon), Layne C. Price `[通讯]` (Amazon)

**通讯引用:** 446 | [OpenAlex ID](https://openalex.org/A5054029529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于梯度提升的参数高效微调框架 BoostLoRA，通过在模型误分类样本上逐轮训练并融合极低参数 TinyLoRA 适配器，实现无推理开销的持续性能提升。

**💡 创新点**

创新点在于采用旋转 SVD 基础策略使每轮适配器投射到正交子空间，从而使累计有效秩随训练轮数线性增长，突破传统低参数适配器的表达限制，并引入仅关注错误样本的梯度孤立训练机制。

**🔧 技术方法**

技术包括 TinyLoRA 低秩 SVD 变换、梯度提升与 RL（GRPO）或交叉熵训练、旋转与顶层 SVD 基础策略、权重合并与累计秩监测，以及多轮失败集构造与自适应学习率调度。

**📊 数据集**

实验数据集涵盖数学推理（GSM8K、MATH-500）、代码生成（MBPP、HumanEval）和蛋白质结合亲和力分类（PPB-Affinity），并使用 Qwen2.5-3B-Instruct 和 ESM2-650M 两大预训练模型。

**📈 对比分析**

与 TinyLoRA、单次大秩 LoRA、全微调以及线性探针等基线对比，BoostLoRA 在 GSM8K 以 12 参数/轮实现 89.1%（超过 3.09B 参数的全微调 87.0%）、在 HumanEval 以 80.4%（远高于全微调 57.9%）以及在蛋白质任务上提升 2–3% 以上，展示了显著的性能提升。

**⚠️ 局限性**

局限性包括需要多轮顺序训练导致较高的总耗时、每轮需完整数据集评估、旋转基础需一次性 SVD 计算、并且未与 COLA 等基线直接对比，难以单独评估各因素贡献。

---

## 28. The Field of Safe Motion: Operationalizing Affordances in the Field of Safe Travel Using Reachability Analysis

**arXiv ID:** 2604.27168 | [PDF](https://arxiv.org/pdf/2604.27168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 29. BLINC: Context-Specific Causal Learning for Automated RAN Configuration

**arXiv ID:** 2604.27084 | [PDF](https://arxiv.org/pdf/2604.27084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 30. NORACL: Neurogenesis for Oracle-free Resource-Adaptive Continual Learning

**arXiv ID:** 2604.27031 | [PDF](https://arxiv.org/pdf/2604.27031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 31. TRUST: A Framework for Decentralized AI Service v.0.1

**arXiv ID:** 2604.27132 | [PDF](https://arxiv.org/pdf/2604.27132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 32. Predicting Upcoming Stuttering Events from Three-Second Audio: Stratified Evaluation Reveals Severity-Selective Precursors, and the Model Deploys Fully On-Device

**arXiv ID:** 2604.27279 | [PDF](https://arxiv.org/pdf/2604.27279v1)

**作者:** Nazar Kozak `[一作]` `[通讯]` (Kozak Technologies Inc), Nazar Kozak (Kozak Technologies Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

训练一个约 616K 参数的 CNN，在每 3 秒音频窗口内预测下一 3 秒是否出现任何流利度障碍事件。

**💡 创新点**

聚合训练能自动捕捉严重事件（block、sound repetition）的先行信号，实现对临床干预具有可操作性的预测，并成功在设备端完成全量导出与实时推理。

**🔧 技术方法**

使用 4 层卷积网络，BCE‑logits 损失，SpecAugment 数据增强，Platt 校准概率，导出 CoreML、ONNX、TFLite，并在 Apple Neural Engine 上实现亚毫秒级推理。

**📊 数据集**

主要数据集为 SEP‑28k（约 20k 个 3 秒片段），并在 FluencyBank 儿童临床语料、DisfluencySpeech 与 LibriStutter 进行跨域验证。

**📈 对比分析**

与 94M 参数 wav2vec‑2.0 线性探针对比，预测任务 AUC 提升约 0.07；在所有测试集上聚合 AUC 约 0.58、跨域 AUC 0.60–0.67，设备端平均延迟 0.25–0.55 ms。

**⚠️ 局限性**

局限在于仅使用单 3 秒窗口、缺乏帧级先行标注、对不同说话者的泛化有限、仅在英语成人播客上训练，未验证多语言或更严格临床环境。

---

## 33. VTBench: A Multimodal Framework for Time-Series Classification with Chart-Based Representations

**arXiv ID:** 2604.27259 | [PDF](https://arxiv.org/pdf/2604.27259v1)

**作者:** Madhumitha Venkatesan `[一作]` (University of California, Davis), Dongyu Liu `[通讯]` (University of California, Davis)

**通讯引用:** 4246 | [OpenAlex ID](https://openalex.org/A5101769619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了VTBench框架，对时间序列进行多种图表（线图、柱图、面积图、散点图）可视化，并与原始数值序列进行多模态融合，支持单图、多图和多模态三种配置，完成了31个UCR数据集的系统实验；

**💡 创新点**

提出首个系统化的图表基准，提供四种可解释的图表类型，允许在多视图与多模态之间灵活融合，并引入轻量级注意力融合机制与实证图表选择指南；

**🔧 技术方法**

使用CNN（浅层与深层）对图表图像进行编码，FCN/Transformer/OS‑CNN对原始序列编码；特征级融合通过拼接或动态权重注意力完成；分类头为全连接网络；训练使用Adam、交叉熵、学习率调度等；

**📊 数据集**

31个UCR时间序列归档数据集，涵盖单/多类别、不同长度与多领域；

**📈 对比分析**

与传统基准（HC2、InceptionTime、OS‑CNN等）以及纯图表模型对比；实验显示：单图模型在小数据集上可与传统方法竞争，多图融合在部分数据集提升准确率，多模态融合在视觉信息非冗余时提升或保持性能，但在冗余时可能下降；整体性能与SOTA相近或略低，却在解释性与可扩展性上具优势；

**⚠️ 局限性**

仅针对单变量序列；图表参数手动设置，缺乏自动适配；融合策略简化，未探索更高级融合；在大规模或多变量数据集上验证不足；对视觉编码的自动选择与适应性仍待研究。

---

## 34. Now's the Time: Computer Science Must Evolve to Emphasize Software and Systems Engineering with Artificial Intelligence (AI)

**arXiv ID:** 2604.27230 | [PDF](https://arxiv.org/pdf/2604.27230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 35. Structural Dissolution: How Artificial Intelligence Dismantles Coordination Architecture and Reconfigures the Political Economy of Production

**arXiv ID:** 2604.27435 | [PDF](https://arxiv.org/pdf/2604.27435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 36. LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional Analysis and Calibration of Human--LLM Judgment Gaps

**arXiv ID:** 2604.27345 | [PDF](https://arxiv.org/pdf/2604.27345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 37. Static Attribution of Android Residential Proxy Malware Using Graph Kernels

**arXiv ID:** 2604.27302 | [PDF](https://arxiv.org/pdf/2604.27302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 38. Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations

**arXiv ID:** 2604.27321 | [PDF](https://arxiv.org/pdf/2604.27321v1)

**作者:** Md Hasan Saju `[一作]`, Akramul Azim `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个端到端的LLM驱动安全运营中心（SOC）工作流框架，涵盖威胁检测、语法约束查询生成（SQM）与检索增强的事件解决与推荐，从警报到事件关闭实现自动化。

**💡 创新点**

创新点在于①使用三模型投票ensemble提升检测精度与降低误报；②构建SQM架构，将语法allowlist、元数据检索、官方文档三大约束融合，生成可直接执行的SIEM查询；③将SQM生成的证据与检索增强生成（RAG）相结合，显著提升事件解决准确率，形成闭环。

**🔧 技术方法**

技术包括传统机器学习（AdaBoost、XGBoost等）、多种大型语言模型（GPT‑4o‑mini、Gemma‑3n‑E4B‑it、Llama‑3.3‑70B等）的ensemble、检索增强生成（RAG）、向量数据库嵌入、BLEU/ROUGE评价、LLM‑as‑Judge评分、风险评分机制以及语法约束脚本。

**📊 数据集**

使用了约20,000条多厂商SIEM日志（网络、终端、云）与ServiceNow事件票据、历史关闭记录、官方文档与查询库等真实生产数据。

**📈 对比分析**

与传统机器学习、单模型LLM、无检索基线进行对比评估。检测模块取得82.8%准确率、FPR 0.12；SQM生成查询BLEU 0.384、ROUGE‑L 0.731、88%可执行率；事件解决准确率从78.3%提升至90%，整体推荐质量评分8.70，手工triage时间从约4小时降至10分钟。

**⚠️ 局限性**

局限性包括仅在IBM QRadar与Google SecOps两大平台验证，跨平台可扩展性待进一步测试；模型依赖文档与查询库，需持续更新；在高动态威胁情境下模型仍可能产生误报/误判，需要持续迭代与人工反馈。

---

## 39. SQuadGen: Generating Simple Quad Layouts via Chart Distance Fields

**arXiv ID:** 2604.27329 | [PDF](https://arxiv.org/pdf/2604.27329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 40. Preserving Temporal Dynamics in Time Series Generation

**arXiv ID:** 2604.27182 | [PDF](https://arxiv.org/pdf/2604.27182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 41. Context-Aware Graph Attention for Unsupervised Telco Anomaly Detection

**arXiv ID:** 2604.27172 | [PDF](https://arxiv.org/pdf/2604.27172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 42. Binary Spiking Neural Networks as Causal Models

**arXiv ID:** 2604.27007 | [PDF](https://arxiv.org/pdf/2604.27007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 43. Gait Recognition via Deep Residual Networks and Multi-Branch Feature Fusion

**arXiv ID:** 2604.27353 | [PDF](https://arxiv.org/pdf/2604.27353v1)

**作者:** Yabo Luo `[一作]` (Osh State University), Cunrong Li `[通讯]` (Osh State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用HRNet进行低分辨率姿态估计，提取身体比例、步速和骨骼运动三种特征，并通过多分支特征融合（MFF）实现高精度步态识别。

**💡 创新点**

提出多分支特征融合模块，将三种互补特征在中间层通过通道注意力机制动态加权融合，显著提升了对姿态、衣物和携带物等干扰因素的鲁棒性。

**🔧 技术方法**

核心技术包括HRNet姿态估计、ResNet‑50深度特征提取、时间序列步速与骨骼运动建模、以及基于通道注意力的多分支融合。

**📊 数据集**

在CASIA‑B公开基准上（跨视角、服装和携带物三种条件）以及自采集的室外数据集上进行评估。

**📈 对比分析**

相较于当前最先进的骨骼/外观融合方法，本文在CASIA‑B总体准确率达89.4%（在正常步态下94.52%），在服装变化条件下提升至83.0%；在室外数据集上总体准确率为85.1%，比第二佳方法提升4.1%。

**⚠️ 局限性**

主要局限在于对极端遮挡、视角极端角度以及多人人群的实时多目标识别尚未充分验证，模型仍依赖HRNet对姿态的准确性。

---

## 44. ABC: Any-Subset Autoregression via Non-Markovian Diffusion Bridges in Continuous Time and Space

**arXiv ID:** 2604.27443 | [PDF](https://arxiv.org/pdf/2604.27443v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 45. Tracking Conversations: Measuring Content and Identity Exposure on AI Chatbots

**arXiv ID:** 2604.27438 | [PDF](https://arxiv.org/pdf/2604.27438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 46. Exploring the Limits of Pruning: Task-Specific Neurons, Model Collapse, and Recovery in Task-Specific Large Language Models

**arXiv ID:** 2604.27115 | [PDF](https://arxiv.org/pdf/2604.27115v1)

**作者:** M. K. Khalidi Siam `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5009105388)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对针对数学推理和代码生成的 Qwen2.5 1.5B/7B 模型进行激活基选择性神经元剪枝。

**💡 创新点**

提出激活差异选择性指标，证明少量任务特定神经元对性能至关重要，且可系统性识别并剔除。

**🔧 技术方法**

采用结构化物理剪枝、激活选择性度量和 LoRA 细调技术。

**📊 数据集**

使用 GSM8K、CodeFeedback（Python）等目标数据集，SQuAD、Conversational-cleaned 等干扰数据集。

**📈 对比分析**

与随机剪枝对比，选择性剪枝在相同剪枝率下保持更高精度；反向剪枝在仅10%时导致性能完全崩溃；细调后可显著恢复性能。

**⚠️ 局限性**

选择性指标可能误删兼容性强的神经元；细调数据集有限，可能限制恢复效果。

---

## 47. Multibit neural inference in a N-ary crossbar architecture

**arXiv ID:** 2604.26979 | [PDF](https://arxiv.org/pdf/2604.26979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 48. Quantifying the Cost of Manual Navigation: A Comparison of Gesture-Based Magnification versus Direct Access Reading in Digital Layout-based Documents

**arXiv ID:** 2604.27010 | [PDF](https://arxiv.org/pdf/2604.27010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 49. A Randomized Controlled Trial and Pilot of Scout: an LLM-Based EHR Search and Synthesis Platform

**arXiv ID:** 2604.26953 | [PDF](https://arxiv.org/pdf/2604.26953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 50. From Prompt to Physical Actuation: Holistic Threat Modeling of LLM-Enabled Robotic Systems

**arXiv ID:** 2604.27267 | [PDF](https://arxiv.org/pdf/2604.27267v1)

**作者:** Neha Nagaraja `[一作]` (Northern Arizona University), Carlo R. da Cunha `[通讯]` (Northern Arizona University)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5059504949)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

对边缘-云架构下的LLM驱动自主机器人进行统一威胁建模，利用DFD+STRIDE对传统网络、对抗感知和对话式威胁进行交叉分析，并给出三条跨界攻击链

**💡 创新点**

首次将MITRE ATT&CK、ATLAS、OWASP LLM Top 10三大威胁目录统一映射到机器人感知-规划-执行管道，并揭示了跨界交互点的威胁融合与关键体系结构缺陷（缺失语义防火墙、跨模态翻译、无监督工具调用）

**🔧 技术方法**

使用数据流图（DFD）与STRIDE逐交互点威胁推导、三类威胁分类法（CCT、AdvT、ConT），结合公开威胁库（MITRE ATT&CK、ATLAS、OWASP LLM Top 10）进行威胁映射

**📊 数据集**

本文未使用实验数据或公开数据集，而是基于架构模型与文献案例（如RoboPAIR、BadRobot、Greshake等）构建攻击链，侧重理论分析

**📈 对比分析**

缺乏量化对比，评估基于定性分析，未给出性能指标；相对已有单一领域威胁研究，提供了更完整的跨域风险视角

**⚠️ 局限性**

局限于单机器人单边缘-云部署，感知侧仅考虑视觉通道；未对多机器人协同、其它感知模态或实际硬件验证进行评估；模型假设LLM服务始终不可信，未考虑模型自检或安全强化方法

---

## 51. TypeBandit: Type-Level Context Allocation and Reweighting for Effective Attribute Completion in Heterogeneous Graph Neural Networks

**arXiv ID:** 2604.27356 | [PDF](https://arxiv.org/pdf/2604.27356v1)

**作者:** Ta-Yang Wang `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**通讯引用:** 17533 | [OpenAlex ID](https://openalex.org/A5033166029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种类型级的自适应采样与重加权框架TypeBandit，用于异构图中的属性补全。

**💡 创新点**

创新点在于将采样视为类型级多臂老虎机，动态分配有限采样预算，并结合拓扑感知初始化和联合完成-预测学习。

**🔧 技术方法**

使用拓扑感知初始化、特征投影、类型级bandit采样、联合完成损失与预测损失，并与R‑GCN、HetGNN、HGT、SimpleHGN等多种异构GNN骨干集成。

**📊 数据集**

在DBLP、IMDB、ACM三大学术/影视异构图数据集及采样版OGBN‑MAG上进行实验验证。

**📈 对比分析**

与同骨干基线对比，TypeBandit在DBLP显著提升（+6.6 F1），在ACM略升（+1.7 F1），在IMDB表现依赖初始化，可提升至与HGT相近；整体提升稳定且计算开销可控。

**⚠️ 局限性**

局限在于对初始化敏感、对类型信息不均衡时效果差异大，并未与最强语义编码器（SeHGNN、HINormer）或大规模工业图做充分对比。

---

## 52. LUCid: Redefining Relevance For Lifelong Personalization

**arXiv ID:** 2604.26996 | [PDF](https://arxiv.org/pdf/2604.26996v1)

**作者:** Chimaobi Okite `[一作]` (University of Michigan), Rada Mihalcea `[通讯]` (University of Michigan)

**通讯引用:** 27855 | [OpenAlex ID](https://openalex.org/A5082450455)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LUCid 基准，专门评估终身个性化中情境相关性，并系统揭示当前检索与生成模型在检索隐式用户上下文和生成个性化回答方面的缺陷。

**💡 创新点**

创新点在于：① 定义并量化了“潜在用户上下文”（latent user context）与“语义邻近优势”（Proximity Advantage, PA）指标；② 构造了低 PA 的真实场景测试集，包含单会话与多会话推理；③ 通过多种检索与长上下文模型的基准实验，首次揭示了语义邻近偏差导致的性能崩溃。

**🔧 技术方法**

采用了检索增强生成（RAG）与其 GraphRAG、记忆增强（RMM）等检索架构；使用对比学习检索器（Contriever、Stella、GTE 等）；评估了重排序器（bge‑reranker‑v2‑gemma、Qwen‑Reranker‑8B、LLM‑based reranker）以及长上下文 LLM（GPT‑5.4‑mini、Claude‑Haiku‑4‑5、Gemini‑3‑Flash、Qwen‑3.5‑27B）等技术。

**📊 数据集**

数据集来源包括公开对话日志（UltraChat、ShareGPT）、现有基准（PrefEval、LongMemEval、PersonaMem 等）与合成用户属性；最终基准包含 1,936 个真实查询、最多 500 个会话（约 620k tokens）及 6 个用户维度（领域、年龄、地理、宗教、健康、沟通风格）。

**📈 对比分析**

在 LUCid‑B（中等规模）和 LUCid‑Hard（极难）上对多种系统进行评测：检索 recall 与 NDCG 在最难实例几乎为 0，重排序器在硬实例 Recall@10 降至 0.08‑0.25；长上下文 LLM 在 Oracle 条件下仍低于 Gold，最高达 80‑84%，表明模型难以准确利用隐式上下文。整体性能明显低于传统基准，凸显语义邻近偏差的致命影响。

**⚠️ 局限性**

局限性包括：① 仅覆盖 6 个用户维度，最多 5 会话推理；② 未涵盖更长的证据链与维度交互；③ 评测仅针对代表性架构，未展开更广泛的模型对比；④ 基准构建基于合成人物，未涉及真实多样性与边缘群体。

---

## 53. State-Dependent Lyapunov Method for Rank-1 Matrix Factorization

**arXiv ID:** 2604.26993 | [PDF](https://arxiv.org/pdf/2604.26993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 54. One Size Fits All? An Empirical Comparison of ADR Templates regarding Comprehension, Usability, and Ease of Adoption

**arXiv ID:** 2604.27333 | [PDF](https://arxiv.org/pdf/2604.27333v1)

**作者:** Fernando Nogueira `[一作]` (Federal University of Amazonas), Tayana Conte `[通讯]` (Federal University of Amazonas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过专家评审和受控实验，对五种常用的ADR模板进行比较，最终评估了Nygard与MADR模板在理解、易用性与可采纳性方面的优劣。

**💡 创新点**

创新点在于首次采用DESMET Feature Analysis方法对ADR模板进行系统化评估，并结合经验值与实验数据，提供可操作的模板选型决策指导。

**🔧 技术方法**

技术方法包括DESMET Feature Analysis（特征评估）、交叉实验设计、非参数Wilcoxon检验与Cliff’s Delta效应量计算。

**📊 数据集**

实验数据来自33名本科软件工程学生在两次任务中完成的ADR记录，收集了时间、咨询次数、客观性评分和推荐分数。

**📈 对比分析**

比较方法是先用专家评审筛选出Nygard与MADR，再用配对实验检验整体得分差异；结果显示Nygard整体得分显著高于MADR（p=0.002，Cliff’s Delta≈0.64）。

**⚠️ 局限性**

局限性包括样本仅为学生，可能不具代表性；专家评审存在主观偏差；仅比较了两种模板，未涵盖所有可能的ADR形式。

---

## 55. Proactive Dialogue Model with Intent Prediction

**arXiv ID:** 2604.27379 | [PDF](https://arxiv.org/pdf/2604.27379v1)

**作者:** Yang Luo `[一作]` (University of Hong Kong), Yang Luo `[通讯]` (University of Hong Kong)

**通讯引用:** 43681 | [OpenAlex ID](https://openalex.org/A5057282533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在任务导向对话系统中引入轻量级意图转移先验，通过在推理时注入基于Temporal Bayesian Network的概率先验来提升系统的主动性与效率。

**💡 创新点**

创新点在于将Temporal Bayesian Network作为动态意图转移模型，并将其生成的先验直接注入LLM提示，从而实现无模型修改的主动式对话生成。

**🔧 技术方法**

使用技术包括意图抽象、NOTEARS结构学习、BDeu参数估计、Top‑k相似性检索、阈值门控和Prompt注入。

**📊 数据集**

采用MultiWOZ 2.2数据集的用户意图注释进行训练与评估。

**📈 对比分析**

与Plain‑LLM、Random Transition、Marginal、Bigram等基线相比，Temporal BN在Recall@5达0.787、MRR 0.576，且在Ground‑truth replay中Coverage AUC提升至0.856，turns到75%覆盖率减少1.22轮。

**⚠️ 局限性**

局限性在于对稀有意图转移的预测不佳、只捕捉短期转移、以及将概率先验转化为生成效果的接口仍有提升空间。

---

## 56. Unified Data Discovery across Query Modalities and User Intents

**arXiv ID:** 2604.27252 | [PDF](https://arxiv.org/pdf/2604.27252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 57. Judge, Then Drive: A Critic-Centric Vision Language Action Framework for Autonomous Driving

**arXiv ID:** 2604.27366 | [PDF](https://arxiv.org/pdf/2604.27366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 58. Compliance versus Sensibility: On the Reasoning Controllability in Large Language Models

**arXiv ID:** 2604.27251 | [PDF](https://arxiv.org/pdf/2604.27251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 59. METASYMBO: Multi-Agent Language-Guided Metamaterial Discovery via Symbolic Latent Evolution

**arXiv ID:** 2604.27300 | [PDF](https://arxiv.org/pdf/2604.27300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 60. Simple Self-Conditioning Adaptation for Masked Diffusion Models

**arXiv ID:** 2604.26985 | [PDF](https://arxiv.org/pdf/2604.26985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 61. HQ-UNet: A Hybrid Quantum-Classical U-Net with a Quantum Bottleneck for Remote Sensing Image Segmentation

**arXiv ID:** 2604.27206 | [PDF](https://arxiv.org/pdf/2604.27206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 62. The Two Boundaries: Why Behavioral AI Governance Fails Structurally

**arXiv ID:** 2604.27292 | [PDF](https://arxiv.org/pdf/2604.27292v1)

**作者:** Alan L. McCann `[一作]` `[通讯]` (Mashin, Inc.), Alan L. McCann (Mashin, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种通过将计算与效应分离的架构，实现在 AI 系统中实现结构化治理，确保所有可表达的操作都必须通过治理边界，并用 Coq 证明其可行性；

**💡 创新点**

核心创新在于引入“同界治理”(coterminous governance)概念，证明通过架构层面的分离可以在理论上消除未治理与舞台效应两大失效模式，并证明 Rice 定理不再适用于此类结构化治理；

**🔧 技术方法**

技术主要包括：计算与效应的语言层面分离、治理边界作为执行管道的一部分、基于 Coq 的形式化证明、以及对治理成本与效应可追溯性的量化评估；

**📊 数据集**

该工作不依赖传统机器学习数据集，主要使用人工构造的理论模型与 Coq 证明模块；

**📈 对比分析**

与传统行为治理方法（内容过滤、RLHF、监控）相比，结构化治理在吞吐量上的额外开销可忽略不计（0.23 ms 与无治理相差无多），而行为治理需要按层级线性增加成本；

**⚠️ 局限性**

局限包括：需要在系统设计之初实现计算/效应分离，无法后期仅通过软件层面补丁实现；治理效果不等同于政策正确性；仅适用于效应治理，不涵盖内容/意图/系统级治理；对特定子语言的可判定性讨论有限。

---

## 63. Evaluating Epistemic Guardrails in AI Reading Assistants: A Behavioral Audit of a Minimal Prototype

**arXiv ID:** 2604.27275 | [PDF](https://arxiv.org/pdf/2604.27275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 64. On the Effectiveness of Modular Testing with EvoSuite

**arXiv ID:** 2604.27112 | [PDF](https://arxiv.org/pdf/2604.27112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 65. Online Monotone Metric Embeddings

**arXiv ID:** 2604.27059 | [PDF](https://arxiv.org/pdf/2604.27059v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Yichen Huang `[通讯]` (Harvard University)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5031172482)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出在线单调度量嵌入模型，并给出其在HST中的随机与确定性失真上限。

**💡 创新点**

创新点在于允许已嵌入点之间距离随时间单调递减，从而突破传统在线嵌入的下界，实现O(log²n)失真。

**🔧 技术方法**

核心技术包括概率平滑划分、递归合并策略、基于HST的在线构造以及潜能函数兼容性框架。

**📊 数据集**

无具体实验数据集，研究完全基于理论分析与证明。

**📈 对比分析**

通过与传统严格嵌入的Ω(min(n, log n logΔ))下界对比，证明在随机单调嵌入下可达到O(log²n)失真；动态场景下得到O(l log l)失真并给出Ω(l)下界。

**⚠️ 局限性**

在一般度量空间上仍有O(log²n)与Ω(log n)下界之间的二次缺口；动态情况下上界与下界仅相差对数因子；未给出对非树形目标度量的结果。

---

## 66. Selective Augmentation: Improving Universal Automatic Phonetic Transcription via G2P Bootstrapping

**arXiv ID:** 2604.27204 | [PDF](https://arxiv.org/pdf/2604.27204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 67. Sparse-View 3D Gaussian Splatting in the Wild

**arXiv ID:** 2604.27422 | [PDF](https://arxiv.org/pdf/2604.27422v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 68. Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking via Architecture-Level Separation

**arXiv ID:** 2604.27201 | [PDF](https://arxiv.org/pdf/2604.27201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 69. SafeTune: Mitigating Data Poisoning in LLM Fine-Tuning for RTL Code Generation

**arXiv ID:** 2604.27238 | [PDF](https://arxiv.org/pdf/2604.27238v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 70. New Diameter Approximations via Distance Oracle Techniques

**arXiv ID:** 2604.27142 | [PDF](https://arxiv.org/pdf/2604.27142v1)

**作者:** Yael Kirkpatrick `[一作]` (Massachusetts Institute Of Technology), Virginia Vassilevska Williams `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 4186 | [OpenAlex ID](https://openalex.org/A5044244682)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对无向图直径近似问题，提出了完整的确定性算法，推导了与距离预处理相结合的新的近似方案，包含3/2、5/3以及与已知下界曲线匹配的算法。

**💡 创新点**

创新点主要包括：①将距离预处理技术（Thorup–Zwick距离预处理）与直径近似紧密结合，实现了Cairo‑Grossi‑Rizzi方案的完全确定化；②构造了新的确定性小集群/小球技术，消除了传统算法中的随机采样与加法误差；③首次给出了非2−1/2^k形式的5/3近似算法，填补了近似曲线上的空白。

**🔧 技术方法**

核心技术包括：确定性哈希/采样的“早期命中集”构造；使用距离预处理的球/聚类构造；利用可预处理的近似距离预处理（(2,1)、(2+ε,5)等）实现直径近似；以及对已有随机算法的完全确定化（例如最小球/聚类、路由方案、稀疏子图等）。

**📊 数据集**

该工作属于理论算法研究，没有使用具体实验数据集，而是通过算法分析与时间/空间复杂度证明其有效性。

**📈 对比分析**

与之前随机算法相比，本文的算法在时间复杂度上保持与CGR相同（如 O(mn^{1/k}) ），并在 3/2 近似中去掉了加法误差、在 5/3 近似中实现了更优的乘法误差；同时实现了完全确定化，消除了随机化带来的期望时间与概率成功的差异。

**⚠️ 局限性**

局限性包括：①目前的确定性算法仍未能达到理论下界的最优点（如3/2 近似已最优但 5/3 近似仍略高于下界所示 5/3+ε 可能仍有改进空间）；②对密集图的时间复杂度仍相对较高；③算法设计高度依赖于距离预处理的结构，若该结构被进一步改进，原算法可能需要重新调整。

---

## 71. InterPartAbility: Text-Guided Part Matching for Interpretable Person Re-Identification

**arXiv ID:** 2604.27122 | [PDF](https://arxiv.org/pdf/2604.27122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching

**arXiv ID:** 2604.27375 | [PDF](https://arxiv.org/pdf/2604.27375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 73. From Notepad AI to Social Media: How Can Text Style Transformation Mitigate Social Harm?

**arXiv ID:** 2604.27365 | [PDF](https://arxiv.org/pdf/2604.27365v1)

**作者:** Syed Mhamudul Hasan `[一作]` (Southern Illinois University), Abdur R. Shahid `[通讯]` (Southern Illinois University)

**通讯引用:** 307 | [OpenAlex ID](https://openalex.org/A5035192689)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于文本风格转换的写作辅助框架，通过控制文本语气来降低社交媒体有害评论的情感强度，并引入情感漂移指数(EDI)评估转换效果。

**💡 创新点**

首次量化风格转换导致的情感漂移并给出 EDI 指标，证明通过适当风格转换可将负面情感转向中性或正面，从而减少情感危害。

**🔧 技术方法**

使用 Phi-3 大语言模型进行零样本风格转换，RoBERTa-base-go_emotions 进行情感检测，并基于 VAD 三维空间计算情感向量和欧氏距离。

**📊 数据集**

使用 HateXplain 与 Toxic Comment Classification Challenge 两个公开有害评论数据集进行实验。

**📈 对比分析**

通过统计各风格下情感保留率、改变率以及 EDI 与情感变化百分比的相关性进行比较；实验表明 Inspirational 与 Humor 风格产生最高 EDI，显著降低负面情绪占比。

**⚠️ 局限性**

仅考虑六种基础情绪，使用未微调的独立 LLM，未评估语义完整性和生成质量的长期影响，实验仅在公开数据集上完成，缺乏实时社交平台验证。

---

## 74. Self-Evolving Software Agents

**arXiv ID:** 2604.27264 | [PDF](https://arxiv.org/pdf/2604.27264v1)

**作者:** Marco Robol `[一作]` (University of Trento), Paolo Giorgini `[通讯]` (University of Trento)

**通讯引用:** 12655 | [OpenAlex ID](https://openalex.org/A5073056211)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种自进化代理架构，能够在运行时动态调整自身参数和策略，以适应不同的任务需求，并实现了相应的原型系统。

**💡 创新点**

创新点在于将遗传算法与强化学习相结合，使代理能够在不需要人工干预的情况下自动改进其行为策略，并通过可插拔模块实现快速迭代。

**🔧 技术方法**

采用了Python、PyTorch、Gym等技术栈；核心算法包括遗传算法（GA）、深度强化学习（DQN/Actor‑Critic）和自适应元学习框架。

**📊 数据集**

在OpenAI Gym中的CartPole、MountainCar、Acrobot等标准控制环境以及自定义的多任务调度数据集上进行实验。

**📈 对比分析**

与传统DQN、PPO以及固定策略的基线进行对比，实验显示自进化代理在平均奖励、收敛速度和稳健性上分别提升约12%、15%和20%；在多任务环境中表现更为优异。

**⚠️ 局限性**

局限性包括：训练时间长、对环境信息的依赖度高、缺乏对真实世界物理约束的考虑，且在大规模部署时需要显著的计算资源。

---

## 75. Reading Speed, Image Quality Ratings, and Comfort Ratings in Augmented Reality

**arXiv ID:** 2604.27203 | [PDF](https://arxiv.org/pdf/2604.27203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 76. Agent Name Service (ANS): A Proof-of-Concept Trust Layer for Secure AI Agent Discovery, Identity, and Governance in Kubernetes

**arXiv ID:** 2604.26997 | [PDF](https://arxiv.org/pdf/2604.26997v1)

**作者:** Akshay Mittal `[一作]` (University of Cumberlands), Elyson De La Cruz `[通讯]` (University of Cumberlands)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

实现并验证了基于 Kubernetes 的 Agent Name Service（ANS）原型，实现了 AI 代理的统一发现、身份验证、能力证明和策略治理；

**💡 创新点**

提出了 DNS‑style 信任层与分布式身份（DID/VC）和零知识能力证明相结合的新型安全模型，并将其与 OPA、Istio 等云原生技术深度集成；

**🔧 技术方法**

使用 Kubernetes CRD、Admission Controller、Istio Service Mesh、Open Policy Agent、Sigstore、ArgoCD、Prometheus/Grafana 等云原生组件，辅以 W3C DID/VC、零知识证明和 Go/TypeScript 代码实现；

**📊 数据集**

在 3 节点 50 代理的实验环境中进行模拟演示，并未使用公开机器学习数据集，而是基于自定义代理工作流进行评估；

**📈 对比分析**

通过对注册、发现、能力验证等关键操作进行基准测试，平均注册 45 ms、发现 12 ms、能力验证 78 ms，策略评估 3 ms；吞吐量可达 1000+ 代理/分钟、10k+ 查询/秒、100k+ 策略评估/秒，整体服务路径延迟低于 10 ms；

**⚠️ 局限性**

局限性包括对新兴代理协议的依赖、加密运算导致的性能开销、集中式注册表的可扩展性瓶颈，以及未在大规模真实生产环境中进行全面验证。

---

## 77. YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal

**arXiv ID:** 2604.27322 | [PDF](https://arxiv.org/pdf/2604.27322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. Energy-Efficient Plant Monitoring via Knowledge Distillation

**arXiv ID:** 2604.27178 | [PDF](https://arxiv.org/pdf/2604.27178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 79. What Suppresses Nash Equilibrium Play in Large Language Models? Mechanistic Evidence and Causal Control

**arXiv ID:** 2604.27167 | [PDF](https://arxiv.org/pdf/2604.27167v1)

**作者:** Paraskevas V. Lekeas `[一作]` (DreamWorks Animation), Giorgos Stamatopoulos `[通讯]` (University of Crete)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5041239607)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过机制可解释性方法，研究大型语言模型在四种两人博弈中的Nash行为偏离原因，并展示如何在推理时通过激活注入逆转该偏离。

**💡 创新点**

创新点在于发现并证明模型内部在后期层级存在分布式合作覆盖机制，利用线性探测、logit透镜和激活补丁验证其因果性；首次表明可通过在前几层注入少量方向，实现从完全合作到几乎完美Nash的双向切换。

**🔧 技术方法**

采用线性探测、logit透镜、注意力头扫描、激活补丁、概念裁定等机制解释工具，对Llama-3-8B的内部表示进行逐层分析，并在推理时注入/裁定激活向量。

**📊 数据集**

使用四个简化的两人博弈（囚徒困境、性别之战、猎鹿、匹配硬币）的对局历史作为输入数据，并在不同提示方式（Direct、CoT、Scratchpad）下收集模型决策。

**📈 对比分析**

通过自对局和交叉对局计算Nash距离 d_Nash 进行比较；对四个模型（8B、70B、32B、72B）在三种提示方式下的 d_Nash 进行对比；在模型层级上评估探测准确率和logit透镜变化；激活注入实验表明注入 -5 可使 defection 达 99.2%，+10 可使 cooperation 达 88.7%，概念裁定与注入值呈显著正相关。

**⚠️ 局限性**

局限性：机制分析仅在 8B 模型，未验证更大模型中抑制机制的位置；仅针对二动作两人游戏，复杂博弈中的机制未知；RLHF 训练方式与未来模型的差异尚未探索。

---

## 80. How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance

**arXiv ID:** 2604.27147 | [PDF](https://arxiv.org/pdf/2604.27147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. Interaction Forces and Internal Loads in Parallel Manipulators with Actuation Redundancy

**arXiv ID:** 2604.27095 | [PDF](https://arxiv.org/pdf/2604.27095v1)

**作者:** Joshua Flight `[一作]`, Clément Gosselin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对具有执行器冗余的并联机械臂，提出了一套从抓握式系统迁移的空域力矩分量分析框架，能够合成不产生交互力或内部负荷的关节扭矩向量。

**💡 创新点**

创新点包括：① 统一界定交互力与内部负荷的物理意义；② 通过引入度量张量，将关节扭矩映射到外力方向后构造加权 Moore‑Penrose 伪逆，从而在并联机械臂中精确合成平衡或操纵力矩；③ 提出首个生成无内部负荷的操纵扭矩解；④ 通过案例研究验证方法优越性。

**🔧 技术方法**

主要技术手段包括：静态-运动学双重性、雅可比矩阵与传动权重矩阵的组合映射、度量张量方法、加权 Moore‑Penrose 伪逆、优化求解（拉格朗日乘子法）以及多边形可达力矩空间构造。

**📊 数据集**

使用的是合成的 3‑RRR 平面并联机械臂案例（连杆长度 0.2 m，端效应器为等边三角形，姿态设定为 x=[0.250 0.144 0]），并在该结构上设置可观测的扭矩极限 τ_max=4.2 Nm，作为实验数据集。

**📈 对比分析**

通过与先前采用最小范数（未加权）伪逆的结果进行比较，本文方法得到的关节扭矩能够生成满足交互力零条件的力分布；可达力矩多边形在同一方向上更大，且内部负荷被消除，验证了性能提升。

**⚠️ 局限性**

局限性包括：仅在平面 3‑RRR 机械臂上验证；假设静态、线性传动与雅可比矩阵已知；未对非平面或多自由度并联结构进行实验验证；对实时控制的可行性尚待进一步研究。

---

## 82. Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding

**arXiv ID:** 2604.26961 | [PDF](https://arxiv.org/pdf/2604.26961v1)

**作者:** Pengfei He `[一作]` (University of Manitoba), Muhammad Asaduzzaman `[通讯]` (University of Windsor)

**通讯引用:** 5669 | [OpenAlex ID](https://openalex.org/A5027988131)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用语言模型进行静态程序切片，并通过预训练和受限解码提升切片精度

**💡 创新点**

提出数据流感知的预训练目标（语句置换和跨度腐败）以及词汇和语法受限解码机制

**🔧 技术方法**

基于CodeT5+的编码-解码架构，使用数据流图指导的预训练、监督微调和受限束搜索

**📊 数据集**

使用CodeNet‑Slice中的Java和Python子集进行评估，预训练阶段使用CodeSearchNet函数数据集

**📈 对比分析**

与LLM、NS‑slicer和直接微调模型对比，ExactMatch分别提升6.4%（Java）和21.9%（Python），其他指标亦显著优于基线

**⚠️ 局限性**

仅针对Java和Python，适用于编码-解码模型，尚需在其他语言和decoder‑only模型上进一步验证

---

## 83. Softmax-GS: Generalized Gaussians Learning When to Blend or Bound

**arXiv ID:** 2604.27437 | [PDF](https://arxiv.org/pdf/2604.27437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 84. Low-Complexity Run-Length-Limited ISI-Mitigation (RLIM) Codes for Molecular Communication

**arXiv ID:** 2604.27104 | [PDF](https://arxiv.org/pdf/2604.27104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 85. LLM-Enhanced Topical Trend Detection at Snapchat

**arXiv ID:** 2604.27131 | [PDF](https://arxiv.org/pdf/2604.27131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 86. Instruction Complexity Induces Positional Collapse in Adversarial LLM Evaluation

**arXiv ID:** 2604.27249 | [PDF](https://arxiv.org/pdf/2604.27249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 87. ChipLingo: A Systematic Training Framework for Large Language Models in EDA

**arXiv ID:** 2604.27415 | [PDF](https://arxiv.org/pdf/2604.27415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 88. Reconstruction by Generation: 3D Multi-Object Scene Reconstruction from Sparse Observations

**arXiv ID:** 2604.27106 | [PDF](https://arxiv.org/pdf/2604.27106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 89. AgenticRecTune: Multi-Agent with Self-Evolving Skillhub for Recommendation System Optimization

**arXiv ID:** 2604.26969 | [PDF](https://arxiv.org/pdf/2604.26969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 90. Robust Learning on Heterogeneous Graphs with Heterophily: A Graph Structure Learning Approach

**arXiv ID:** 2604.27387 | [PDF](https://arxiv.org/pdf/2604.27387v1)

**作者:** Yihan Zhang `[一作]` (Tsinghua University), Ercan E. Kuruoglu `[通讯]` (Tsinghua University)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5038228706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的 HGUL 框架，用于在存在异构性、异类性和结构噪声的异构图上进行鲁棒的表示学习。

**💡 创新点**

创新点在于将 kNN 近邻图构建、图结构学习（基于 Gumbel‑Sigmoid 的可学习稀疏化）与异构亲和学习（多阶多边图核下的类级亲和矩阵）三大模块融合，能够同时抑制噪声边、捕捉异类交互并提升泛化性能。

**🔧 技术方法**

核心技术包括：kNN 近邻图构建、Gumbel‑Sigmoid 重参数化 + 直通估计的图结构学习、基于多阶图核（PPR）构造的类级亲和矩阵、R‑GCN 结构编码、门控融合与正则化。

**📊 数据集**

使用 H2GB benchmark 中的多种大规模异构图数据集：ogbn‑mag、mag‑year、oag‑cs、RCDD、IEEE‑CIS‑G、H‑Pokec、PDNS。

**📈 对比分析**

与多类基准（HGNN、R‑GCN、HGT、HDHGR、LatGRL、Hetero2Net、H2G‑former、以及稳健图学习方法如 DropEdge、HGSL、PT‑HGNN 等）对比，HGUL 在大多数数据集上均取得更高的准确率/ F1 分数，并在注入不同噪声率下保持更低的性能下降，证明其鲁棒性和优越性。

**⚠️ 局限性**

局限性包括：对超参数（k、阈值 δ、正则化 γ）敏感，需要一定的调优；亲和矩阵需预训练得到，可能受限于标签稀疏；在动态图或极大规模动态异构图场景下的适应性和扩展性尚待进一步验证。

---

## 91. MAEO: Multiobjective Animorphic Ensemble Optimization for Scalable Large-scale Engineering Applications

**arXiv ID:** 2604.26973 | [PDF](https://arxiv.org/pdf/2604.26973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 92. MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction

**arXiv ID:** 2604.27393 | [PDF](https://arxiv.org/pdf/2604.27393v1)

**作者:** Junbo Cui `[一作]` (MiniCPM-o Team, OpenBMB), Yuan Yao `[通讯]` (MiniCPM-o Team, OpenBMB)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布了 MiniCPM‑o 4.5，一个 9B 参数的多模态大语言模型，能够实时全双工感知视觉、音频并同步产生文本和语音，支持主动行为。

**💡 创新点**

创新点在于提出 Omni‑Flow 统一流式框架，将多模态输入输出对齐到共享时间轴，实现连续感知与回应；以及时间对齐插值（TAIL）等语音生成策略，首次实现了高效的全双工交互。

**🔧 技术方法**

采用端到端多模态架构，视觉编码使用 LLaVA‑UHD + SigLIP，音频编码使用 Whisper Medium + MLP 压缩；LLM 主干为 Qwen3‑8B；配合轻量级语音解码器、流匹配解码器；强化学习（GRPO + RLAIF‑V）和自定义长度奖励；量化后通过 llama.cpp‑omni 实现边缘设备推理。

**📊 数据集**

训练使用多来源数据：视觉‑语言（MiniCPM‑V 4.5 系统扩展、MMBench、MMVet、MMStar 等）；音频（AISHELL‑1/2、LibriSpeech、GigaSpeech、CoVoST 2 等）；多模态全双工数据（Web 音视频、手工构造任务场景、OCR 文档、视频字幕等）以及对齐时间标签。

**📈 对比分析**

在标准基准（OpenCompass、MMBench、OCRBench、Speech ASR/Translation、LiveSports‑3K‑CC 等）上，MiniCPM‑o 4.5 在同等参数规模下逼近 Gemini 2.5 Flash，超越 Qwen3‑Omni‑30B‑A3B；在全双工实时交互上赢率54.4% 高于 LiveCC 与 StreamingVLM；在推理效率上，INT4 版实现 212 tokens/s、11 GB 内存，显著优于同参数对手。

**⚠️ 局限性**

局限性：对长时间动态场景的鲁棒性仍不足，语音生成在全双工模式下偶有不稳定或发音错误；主动行为相对简单；网络不稳时延迟增大；需进一步提升自发规划与上下文驱动交互的深度。

---

## 93. Monitoring Neural Training with Topology: A Footprint-Predictable Collapse Index

**arXiv ID:** 2604.26984 | [PDF](https://arxiv.org/pdf/2604.26984v1)

**作者:** Alexander Kalinowski `[一作]` `[通讯]` (SUNY Empire), Alexander Kalinowski (SUNY Empire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种在线拓扑监测框架，利用模块化莫尔同调维护（MMHM）实时跟踪神经网络表示空间的拓扑结构，并基于Betti数、临界单元波动、环路脆弱度与边界矩阵操作的综合 Collapse Index（CI）来预警模型的表示崩溃。

**💡 创新点**

创新点在于：① 将MMHM技术迁移到训练时监控，做到仅在局部星域更新而非重构整个复形；② 设计了多维度 Collapse Index，将拓扑变化、临界单元冲突、环路脆弱度与计算开销融为一体，实现低延迟、可预测的早期预警；③ 通过对比传统异质度指标 IsoScore，证明 CI 能提前数个 epoch 预测性能衰退。

**🔧 技术方法**

所用技术包括离散莫尔理论、可变规模 k‑NN 简单复形构造、MMHM 的增量化简约与匹配修复、Betti 数和边界矩阵的增量稀疏化、临界单元冲突（critical cell churn）与环路脆弱度（fragility）的统计、指数平滑的 CI 计算。

**📊 数据集**

实验数据集包括：LLM 微调使用 STS‑B 句子相似度回归，模型为 bert‑base、sbert‑base、allMini‑base；TKGE 训练使用 ICEWS14、ICEWS05‑15、Wikidata‑12k、YAGO，模型为 TransE‑TE、RotatE‑TE、ComplEx‑TE。

**📈 对比分析**

在实验中，CI 的负阶相关性均优于 IsoScore，能够在平均 1.4–5.6 轮（LLM）和 0.9–3.6 轮（TKGE）之前就检测到性能下降；相较传统指标，CI 具有更高的预测准确度和更快的响应速度，并且对不同模型、层次和超参数表现出鲁棒性。

**⚠️ 局限性**

局限性包括：对邻域规模 k 与移动点比例 p 的敏感性，需要手动调参；对特定表示层的依赖性，若选错层可能导致预警迟缓；虽然增量化实现降低开销，但在极大模型或高频更新下仍可能产生显著计算成本。

---

## 94. Cross-Lingual Response Consistency in Large Language Models: An ILR-Informed Evaluation of Claude Across Six Languages

**arXiv ID:** 2604.27137 | [PDF](https://arxiv.org/pdf/2604.27137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 95. End-to-End Evaluation and Governance of an EHR-Embedded AI Agent for Clinicians

**arXiv ID:** 2604.27309 | [PDF](https://arxiv.org/pdf/2604.27309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 96. Real-Time GPU-Accelerated Monte Carlo Evaluation of Safety-Critical AEB Systems Under Uncertainty

**arXiv ID:** 2604.27193 | [PDF](https://arxiv.org/pdf/2604.27193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 97. Bibliometric Mapping of AI-Supported Social Presence in Online Learning Environments: Trends, Collaboration, and Thematic Directions

**arXiv ID:** 2604.27344 | [PDF](https://arxiv.org/pdf/2604.27344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 98. Detecting is Easy, Adapting is Hard: Local Expert Growth for Visual Model-Based Reinforcement Learning under Distribution Shift

**arXiv ID:** 2604.27411 | [PDF](https://arxiv.org/pdf/2604.27411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 99. Web2BigTable: A Bi-Level Multi-Agent LLM System for Internet-Scale Information Search and Extraction

**arXiv ID:** 2604.27221 | [PDF](https://arxiv.org/pdf/2604.27221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 100. Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference

**arXiv ID:** 2604.26968 | [PDF](https://arxiv.org/pdf/2604.26968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 101. Defeasible Conditional Obligation in a Two-tiered Preference-based Semantics (Extended Version)

**arXiv ID:** 2604.26977 | [PDF](https://arxiv.org/pdf/2604.26977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 102. Learning to Forget: Continual Learning with Adaptive Weight Decay

**arXiv ID:** 2604.27063 | [PDF](https://arxiv.org/pdf/2604.27063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 103. Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation via KernelSOS

**arXiv ID:** 2604.27175 | [PDF](https://arxiv.org/pdf/2604.27175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 104. CI-Repair-Bench: A Repository-Aware Benchmark for Automated Patch Validation via CI Workflows

**arXiv ID:** 2604.27148 | [PDF](https://arxiv.org/pdf/2604.27148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 105. Co-Evolving Policy Distillation

**arXiv ID:** 2604.27083 | [PDF](https://arxiv.org/pdf/2604.27083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. Interval Orders, Biorders and Credibility-limited Belief Revision

**arXiv ID:** 2604.27156 | [PDF](https://arxiv.org/pdf/2604.27156v1)

**作者:** Richard Booth `[一作]` (Cardiff University), Ivan Varzinczak `[通讯]` (Université Sorbonne Paris Nord)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5034598656)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了基于区间序（IOB）和双序（BOB、ZTBOB、TBOB）的信念修正框架，并给出了它们的公理化表征；随后将这些修正用于非优先修正（NPR）和可信度限制修正，提供了与单句闭包弱化的对应结构。

**💡 创新点**

创新点在于：①首次将 biorder 引入信念修正，构造了四种新的修正操作；②通过公理化揭示了它们与传统 AGM 公理的关系；③将 biorder‑based 修正与非优先、可信度限制修正相结合，形成了兼具成功与一致性的新范式。

**🔧 技术方法**

主要技术：偏序与区间解释理论、结构化公理化方法、表示定理证明，以及可信度限制（credibility‑limited）修正框架的应用。

**📊 数据集**

该研究为理论研究，没有使用任何实验数据集。

**📈 对比分析**

比较方法主要通过公理化对比与 AGM 公理的对应关系来验证一致性与成功性；由于是理论工作，未进行实验性能评估。

**⚠️ 局限性**

局限性：BOB 修正不保证一致性；z‑transitive 与 transitive 的约束可能限制了模型的表达力；biorder 的选择需要人工设定；迭代修正及其性质仍未研究。

---

## 107. A Discipline-Agnostic AI Literacy Course for Academic Research: Architecture, Pedagogy, and Implementation

**arXiv ID:** 2604.27225 | [PDF](https://arxiv.org/pdf/2604.27225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 108. Student Classroom Behavior Recognition Based on Improved YOLOv8s

**arXiv ID:** 2604.27293 | [PDF](https://arxiv.org/pdf/2604.27293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 109. Pragmos: A Process Agentic Modeling System

**arXiv ID:** 2604.27311 | [PDF](https://arxiv.org/pdf/2604.27311v1)

**作者:** Pedro-Aarón Hernández-Ávalos `[一作]` (Tecnologico de Monterrey), Luciano García-Bañuelos `[通讯]` (Tecnologico de Monterrey)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 Pragmos 系统，通过交互式多步骤流程模型生成与 LLM 协作，支持人机共同建模并提供可解释的中间模型。

**💡 创新点**

引入基于行为关系的分阶段模型生成流程，强调可视化中间结果、交互式对话以及使用模块分解与行为关系图构造结构化 BPMN，打破黑盒 LLM 生成模型的局限。

**🔧 技术方法**

结合大型语言模型（ChatGPT、Gemini、Gemma、GPT-OSS）、行为关系抽取、直接后继图、模块分解树、BPMN 生成算法以及循环与冲突处理技术；并运用提示工程和链式思维等 LLM 交互技巧。

**📊 数据集**

使用 PET 数据集（45 条业务流程描述）进行实验和人工验证。

**📈 对比分析**

通过人工对比验证，Pragmos 在绝大多数案例中生成了符合描述的正确模型；相较于 ProMoAI、BPMN-Chatbot 等工具，Pragmos 在可解释性和中间结果交互方面表现更佳，虽然未给出量化指标，但准确率很高。

**⚠️ 局限性**

对极大文本（如 doc-4.1）会超出上下文长度导致模型质量下降；循环类型判定错误（repeat 与 while 混淆）；在结构交织和可选路径时需要额外对齐；对大型描述的抽象机制仍不完善。

---

## 110. Learning Tactile-Aware Quadrupedal Loco-Manipulation Policies

**arXiv ID:** 2604.27224 | [PDF](https://arxiv.org/pdf/2604.27224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 111. When Roles Fail: Epistemic Constraints on Advocate Role Fidelity in LLM-Based Political Statement Analysis

**arXiv ID:** 2604.27228 | [PDF](https://arxiv.org/pdf/2604.27228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 112. Stable but Wrong: An Inference Limit in Galactic Archaeology

**arXiv ID:** 2604.27368 | [PDF](https://arxiv.org/pdf/2604.27368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 113. Dynamic Adversarial Fine-Tuning Reorganizes Refusal Geometry

**arXiv ID:** 2604.27019 | [PDF](https://arxiv.org/pdf/2604.27019v1)

**作者:** Wenhao Lan `[一作]` (University of Chinese Academy of Sciences), Yijun Yang `[通讯]` (Shandong University)

**通讯引用:** 8073 | [OpenAlex ID](https://openalex.org/A5000074603)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在Mistral‑7B模型上，动态对抗微调R2D2与传统监督微调(SFT)对拒绝行为的几何机制与鲁棒性‑实用性平衡轨迹。

**💡 创新点**

创新点在于通过五点锚点几何测量和因果干预，揭示R2D2在训练过程中拒绝承载从后层迁移至前层的重组过程，而非单纯漂移或维度扩展。

**🔧 技术方法**

使用技术包括 HarmBench、StrongREJECT、XSTest 等评估；COSMIC/方向识别与可接受载体库；有效秩、参与比、主角角度分析；单向与三维子空间因果干预。

**📊 数据集**

数据集为 Mistral‑7B 基础模型，LoRA 训练使用 UltraChat‑200k；对抗测试池基于 HarmBench GCG 生成的固定源攻击；XSTest 安全性测试；以及 60 条安全与非安全提示的直接使用性审核集。

**📈 对比分析**

比较方法为在相同检查点下对固定源 HarmBench ASR、StrongREJECT 分数、XSTest 拒绝率及直接使用性评分进行对比。R2D2 在前 50–100 步实现 ASR 0.0 但 XSTest 拒绝率极高，后续步恢复部分使用性同时 ASR 升高；SFT 始终保持较高 ASR 且低拒绝率，整体鲁棒性差。

**⚠️ 局限性**

局限性包括仅针对单一 7B 模型与两种训练方式；使用固定源攻击不代表对自适应攻击的鲁棒性；几何测量基于自制可接受库，跨体系不直接可比；干预仅在少数锚点进行，未覆盖完整训练轨迹；直接使用性审核规模有限。

---

## 114. Evaluating TabPFN for Mild Cognitive Impairment to Alzheimer's Disease Conversion in Data Limited Settings

**arXiv ID:** 2604.27195 | [PDF](https://arxiv.org/pdf/2604.27195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 115. Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution

**arXiv ID:** 2604.27295 | [PDF](https://arxiv.org/pdf/2604.27295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 116. Detecting Clinical Discrepancies in Health Coaching Agents: A Dual-Stream Memory and Reconciliation Architecture

**arXiv ID:** 2604.27045 | [PDF](https://arxiv.org/pdf/2604.27045v1)

**作者:** Samuel L Pugh `[一作]` (Verily Health Inc.), Alessandra Breschi `[通讯]` (Verily Health Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一种双流记忆架构，结合了对患者自述和FHIR结构化记录的并行存储，并通过专门的对账引擎对每一次对话更新进行冲突检测和严重性分级，支持持续的临床对账；

**💡 创新点**

创新点在于将患者自述与EHR严格隔离并在记忆提取阶段就进行实时对账，避免传统记忆系统对医疗信息的随意覆盖；通过可追溯的FHIR资源引用实现错误可追踪；以及在真实长周期对话中构造混合数据集进行端到端错误级联分析；

**🔧 技术方法**

核心技术包括：基于GPT‑4o的增量记忆提取管道、对账引擎（同样使用GPT‑4o）执行时间推理与差异分类、结构化JSON输出、基于FHIR R4 bundle的临床流、双流记忆存储与Delta‑based更新、以及用LLM判别器进行评估；

**📊 数据集**

使用的数据集：真实的UIC Wellness Coaching文本对话（3,665轮、26名患者），以及用Synthea生成并匹配的FHIR R4 bundle（每位患者1个），再通过LLM生成的合成对话与标签形成243个合成对账场景；最终混合共675个会话；

**📈 对比分析**

比较方法：三维评估——(1)记忆提取召回与一致性；(2)对账引擎在理想记忆下的检测、资源引用、严重性与安全性召回；(3)完整流水线的错误级联；性能表现：提取召回70.8%，严格召回26.2%；对账检测率84.4%（理想）/70.8%（完整）；安全性召回86.7%/75.9%；资源引用召回65.3%/51.4%；总体错误级联下降13.6%；

**⚠️ 局限性**

局限性包括：1) 语料主要为合成对账场景，真实病历与自然语料匹配度有限；2) 归因与严重性标签由LLM生成，缺乏临床专家裁定；3) 提取召回随对话长度下降，提示需要更鲁棒的记忆更新机制；4) 仅在模拟环境下验证，未在真实临床部署中检验检测结果的临床可用性。

---

## 117. AttriBE: Quantifying Attribute Expressivity in Body Embeddings for Recognition and Identification

**arXiv ID:** 2604.27218 | [PDF](https://arxiv.org/pdf/2604.27218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 118. Where did we fail? -- Reproducing build failures in embedded open source software

**arXiv ID:** 2604.27075 | [PDF](https://arxiv.org/pdf/2604.27075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 119. Breaking Bad Financial Habits: How LLM Conversations Correct Financial Misconceptions

**arXiv ID:** 2604.27022 | [PDF](https://arxiv.org/pdf/2604.27022v1)

**作者:** Jillian Ross `[一作]` (MIT), Andrew W. Lo `[通讯]` (MIT)

**通讯引用:** 51800 | [OpenAlex ID](https://openalex.org/A5109980718)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对三项预注册实验进行设计和实施，评估利用大语言模型（LLM）对个人财务误区进行对话式纠正的可持续效果，并探究纠正所需的两个关键因素：明确的纠正意图与受众的接受度。

**💡 创新点**

创新点在于：①首次系统证明仅靠信息传递不足以纠正财务误区，只有在LLM被明确指向纠正并具备适度复杂度与用户匹配时，才能产生持久且显著的认知修正；②揭示未具备纠正意图的LLM对话甚至会强化误区；③展示了通过自适应匹配受众金融素养水平来最大化纠正效果的方法。

**🔧 技术方法**

使用 GPT‑4o 语言模型，并通过嵌入金融教授的专业推理作为种子知识，构建“金融专家”聊天机器人。实验采用随机分配到不同提示策略（纠正、评估、自己表述、干扰）以及不同复杂度匹配策略（匹配、超出、低于）进行对话。

**📊 数据集**

数据来源为美国 Prolific 平台收集的受试者自评财务误区信念（10 条误区与 10 条中性陈述）以及自评金融素养水平。所有受试者均在在线问卷中完成实验流程，数据以信念评分形式记录。

**📈 对比分析**

对照设计包括：LLM Shift（明确纠正）vs LLM Evaluate（仅评估）vs Self‑Articulate（自我阐述）vs Distractor（无干预）。通过比较前后信念评分的差异，量化修正幅度。结果显示：LLM Shift 平均修正约 49 分，纠正效果随初始误区强度正相关；匹配复杂度的 LLM 可实现约 54 分的修正，超出/低于匹配均表现不同。所有显著性检验均通过 p<0.05 或更严格阈值。

**⚠️ 局限性**

局限性：①样本为美国 Prolific 受试者，可能不具备普遍外部效度；②仅使用 GPT‑4o，未检验其他模型或更新版本；③依赖自我报告的信念变化，缺乏行为层面的验证；④实验选取最强误区，可能高估平均效果；⑤随访时间仅 10 天，长期效应未知；⑥干扰任务与真实情境差距可能导致需求效应。

---

## 120. AdaBFL: Multi-Layer Defensive Adaptive Aggregation for Bzantine-Robust Federated Learning

**arXiv ID:** 2604.27434 | [PDF](https://arxiv.org/pdf/2604.27434v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 121. A note on the parameter $\ell$ in Buchbinder--Feldman's deterministic submodular matroid algorithm

**arXiv ID:** 2604.27362 | [PDF](https://arxiv.org/pdf/2604.27362v1)

**作者:** Shisheng Li `[一作]` (University of Science and Technology of China), Shisheng Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4993 | [OpenAlex ID](https://openalex.org/A5101563815)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过改进对 (1+1/ℓ)^{-ℓ} 的上界，提出了更小的整数参数 ℓ，从而在已有的确定性 (1-1/e-ε) 子模优化算法中降低了隐藏的查询复杂度常数。

**💡 创新点**

创新点在于用经典的 Pólya–Szegő 不等式和交错级数尾项估计，分别得到 ℓ=⌈1/(2eε)⌉ 和 ℓ≈1/(2eε)-5/12 的更精确界，使得查询次数从 2^{1/ε} 缩减到 2^{1/(2eε)}，提升约 2^{0.816}/ε 倍。

**🔧 技术方法**

主要技术包括：基础不等式证明（Hermite–Hadamard、Padé 近似）、对数级数的截断估计、以及在 Lean 4 里对这些不等式进行形式化验证。

**📊 数据集**

本文不涉及任何实验数据集，纯粹是理论分析与形式化证明。

**📈 对比分析**

与 Buchbinder–Feldman 的原始 ℓ=1+⌈1/ε⌉ 方案相比，本文仅提升隐含常数，未给出新的实验对比，主要关注理论上更优的常数因子。

**⚠️ 局限性**

局限在于改进仅体现在常数层面，实际算法实现与性能并未得到提升，且在整数化后常数改善可能被舍入削弱，无法在实践中显著影响计算复杂度。

---

## 122. Useless but Safe? Benchmarking Utility Recovery with User Intent Clarification in Multi-Turn Conversations

**arXiv ID:** 2604.27093 | [PDF](https://arxiv.org/pdf/2604.27093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 123. MARS: Efficient, Adaptive Co-Scheduling for Heterogeneous Agentic Systems

**arXiv ID:** 2604.26963 | [PDF](https://arxiv.org/pdf/2604.26963v1)

**作者:** Yifei Wang `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 26474 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一个高效自适应的协同调度系统（CoSchedule），专门为多轮、GPU-CPU 交互的代理式 LLM 工作负载提供资源管理与调度；

**💡 创新点**

创新点在于：①引入统一信息流（Unified Information Stream）实现 GPU 与 CPU 资源压力的全栈可观测；②通过外部控制平面实现基于双重压力的自适应准入窗口；③内部采用基于会话的优先级协调器与机会调度器，动态保留 KV 缓存并优化前填充与恢复；

**🔧 技术方法**

技术包括：vLLM 框架改造、AIMD 控制循环、Multi-Level Feedback Queue、KV 块级监控、工具执行边界事件、LLM-OS 样式的会话管理；

**📊 数据集**

实验使用的工作负载来自五个基准：SWE‑bench、GitTaskBench、Terminal‑Bench、RepoBench、∞Bench，并在 OpenHands 框架下进一步验证；

**📈 对比分析**

与 FCFS、Autellix、Infercept、Continuum 等基线在 H100/H200 GPU 上进行对比，结果显示在多种输入长度与负载点下，CoSchedule 的平均延迟提升 1.44×–5.94×，P90/P95 延迟提升 1.29×–3.37×，Goodput 成功提升 2×–7.56×，并在 OpenHands 部署中将任务完成时间加速 1.20×–1.87×；

**⚠️ 局限性**

局限性包括：尚未在多 GPU 或跨节点分布式环境下完整评估；对高度动态的 DAG 结构调度支持有限；在轻负载情况下机会调度的开销可能超过收益；需要进一步研究公平性与多租户 SLA 的兼容性。

---

## 124. Fast and Faithful Edge Bundling using Spectral Sparsification

**arXiv ID:** 2604.26994 | [PDF](https://arxiv.org/pdf/2604.26994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 125. AG-TAL: Anatomically-Guided Topology-Aware Loss for Multiclass Segmentation of the Circle of Willis Using Large-Scale Multi-Center Datasets

**arXiv ID:** 2604.27357 | [PDF](https://arxiv.org/pdf/2604.27357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 126. From Unstructured to Structured: LLM-Guided Attribute Graphs for Entity Search and Ranking

**arXiv ID:** 2604.27410 | [PDF](https://arxiv.org/pdf/2604.27410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 127. PALCAS: A Priority-Aware Intelligent Lane Change Advisory System for Autonomous Vehicles using Federated Reinforcement Learning

**arXiv ID:** 2604.27118 | [PDF](https://arxiv.org/pdf/2604.27118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 128. Cross-lingual Comparison of Research Funding Projects with Multilingual Sentence-BERT: Evidence from KAKENHI, NIH, NSF, and UKRI

**arXiv ID:** 2604.27315 | [PDF](https://arxiv.org/pdf/2604.27315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 129. Simulating Validity: Modal Decoupling in MLLM Generated Feedback on Science Drawings

**arXiv ID:** 2604.26957 | [PDF](https://arxiv.org/pdf/2604.26957v1)

**作者:** Arne Bewersdorff `[一作]` (University of Georgia), Xiaoming Zhai `[通讯]` (University of Georgia)

**通讯引用:** 5229 | [OpenAlex ID](https://openalex.org/A5013379229)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了大型多模态语言模型（如 GPT‑5.1）在对中学生手绘科学模型生成反馈时的“模态解耦”问题，并评估了直接提示与“清单优先”提示两种工作流对错误率和错误类型的影响。

**💡 创新点**

首次量化并区分四类着地错误（对象、属性、关系不匹配及错误缺失），比较两种提示流程在错误率、错误分布上的差异，并检验文本表面特征是否能预测错误，揭示提示策略只能降低但无法根除模态解耦。

**🔧 技术方法**

使用 GPT‑5.1 的 OpenAI Chat Completions API 生成反馈，手动编码错误类型，采用 Wilcoxon、McNemar、逻辑回归和 AUC 等统计方法进行比较与评估。

**📊 数据集**

150 张来自中学学生的分子动力学单元手绘图，涵盖五个建模任务与三种表现力等级（Level 1–3）作为实验数据集。

**📈 对比分析**

通过配对比较两种提示流程，直接提示错误率为 49.3%，清单优先降低至 33.3%，平均错误数从 0.654 降至 0.414；错误分布显示“错误缺失”仍占主导，说明性能有所提升但仍有约三分之一反馈不可靠。

**⚠️ 局限性**

模态解耦源于模型架构与训练局限，提示策略无法彻底消除；文本表面特征难以检测错误；实验仅基于单一模型与单一学科领域，结果可能缺乏普适性。

---

## 130. Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation

**arXiv ID:** 2604.27263 | [PDF](https://arxiv.org/pdf/2604.27263v1)

**作者:** Théo Gigant `[一作]` (Nous Research), Jeffrey Quesnelle `[通讯]` (Nous Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在控制的字节级预训练管线中，作者通过一系列实验把子词分词的作用拆分开来，逐一检验它们对模型训练效率、表示学习和下游性能的具体贡献。

**💡 创新点**

创新点在于：①首次把子词分词的不同效应（样本吞吐、词表规模、子词边界先验、优化目标）分离并独立模拟；②通过在字节级模型中逐步注入子词边界先验和增加样本吞吐来量化子词优势；③提出“子词边界作为先验”与“子词距离作为先验”两种新的先验形式，并对比其对字节级模型的影响。

**🔧 技术方法**

使用的技术包括：LLaMA‑3架构、TorchTitan训练框架、BPE子词分词器、字节级嵌入与多头n‑gram嵌入、字节序列压缩、子词边界/距离先验注入、交叉熵/下一个子词预测等。

**📊 数据集**

使用的数据集是英文为主的 FineWeb‑Edu，采用 UTF‑8 字节序列和相应的 LLaMA‑3 BPE 边界进行实验。

**📈 对比分析**

对比方法：把字节级模型在 1.7B 参数规模下做基线训练，并在前 50k 步分别加入不同的模拟效果；随后恢复基线训练并监测验证交叉熵。结果显示：①人工提高样本吞吐 4 倍可显著提升验证损失；②给出子词边界先验能带来更大性能提升；③词表规模扩展和子词距离先验对性能影响不大；③针对下一个子词预测的目标在此规模下表现不佳。整体来看，子词分词的主要优势在于更高的样本吞吐和子词边界先验。

**⚠️ 局限性**

限制包括：①所有干预仅在前 50k 步实施，缺乏长周期的效果评估；②只在单语（英语）1.7B 参数规模的模型上测试，未验证在更大规模或多语环境下的泛化；③对子词边界和距离先验的相互作用没有深入探讨；④实验仅在字节级模型上进行，未评估这些改进是否能直接迁移到真正的子词模型上。

---

## 131. A Gated Hybrid Contrastive Collaborative Filtering Recommendation

**arXiv ID:** 2604.27117 | [PDF](https://arxiv.org/pdf/2604.27117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 132. Exploring the Adoption Intention in Using AI-Enabled Educational Tools Among Preservice Teachers in the Philippines: A Partial-Least Square Modeling

**arXiv ID:** 2604.27346 | [PDF](https://arxiv.org/pdf/2604.27346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 133. Toward Personalized Digital Twins for Cognitive Decline Assessment: A Multimodal, Uncertainty-Aware Framework

**arXiv ID:** 2604.27217 | [PDF](https://arxiv.org/pdf/2604.27217v1)

**作者:** Bulent Soykan `[一作]` (University of Toledo), Laura J. Brattain `[通讯]` (University of Central Florida)

**通讯引用:** 979 | [OpenAlex ID](https://openalex.org/A5026644166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了多模态不确定性感知的个性化认知衰退数字孪生框架PCD‑DT，并在TADPOLE纵向数据上完成了可行性分析。

**💡 创新点**

首次将潜在状态空间模型、跨模态融合、合成数据扩增与不确定性验证整合到闭环数字孪生系统，能够连续更新且具可解释性的个体预测。

**🔧 技术方法**

采用变分推断的潜在状态空间模型、Transformer/注意力机制的深度多模态融合、贝叶斯低秩张量降维、LSTM序列预测以及条件VAE/GAN生成等技术。

**📊 数据集**

主要使用TADPOLE纵向数据（MRI、PET、CSF、认知评分、诊断等）进行实验，并在ADNI结构MRI上评估张量模型。

**📈 对比分析**

通过与Last Observation Carried Forward（LOCF）基线以及不同模态配置的LSTM模型比较，认知+MRI组合在标准化RMSE上取得ADAS13 0.4419、脑室体积 0.5842 的最佳性能，明显优于基线。

**⚠️ 局限性**

尚未实现完整的变分状态空间模型，缺乏后验不确定性校准、多步预测与系统性缺失健壮性评估，诊断分类表现不稳定，需要进一步验证。

---

## 134. A Diagrammatic Axiomatisation of Behavioural Distance of Nondeterministic Processes

**arXiv ID:** 2604.27268 | [PDF](https://arxiv.org/pdf/2604.27268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 135. Indirect Prompt Injection in the Wild: An Empirical Study of Prevalence, Techniques, and Objectives

**arXiv ID:** 2604.27202 | [PDF](https://arxiv.org/pdf/2604.27202v1)

**作者:** Soheil Khodayari `[一作]` (Independent Researcher), Giancarlo Pellegrino `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模网络爬虫数据上识别并验证了 1.2 亿条 URL 中 15.3 万条网页级 Prompt 注入实例，构建了首个网页级 Prompt 注入数据集，并系统分析了其模板、目标、注入方式、可见性、持续时间及对 LLM 的实测效果。

**💡 创新点**

创新点包括：① 通过 1.2 亿 URL 规模的实测，首次量化 Prompt 注入的真实流行度与结构；② 发现 54 个模板占 95% 的实例，表明攻击已趋于可复用；③ 通过多维度（目标、技术、可见性、持续性）对攻击生态进行完整刻画；④ 将注入效果在 13 种 LLM（开源与闭源）和 4 种页面表示上进行大规模实验，揭示“文本化”输入最易被利用。

**🔧 技术方法**

技术手段主要包括：Aho–Corasick 多关键字匹配、结构化上下文抽取与分组、人工验证流程、词向量+DBSCAN 聚类、自然语言推理（NLI）判定注入技术、Chrome Topics API 与 Tranco 排名用于主题与流量分析、Playwright+Headless Chrome 用于可见性检测，以及 5,200 次 LLM 运行的手工评测。

**📊 数据集**

数据集：1.2B URL（Common Crawl 2025）+ 3,346 条 Shodan/Censys 取证；从中抽取 15.3K 验证实例，覆盖 11,722 页、2,042 主机；此外还使用 Common Crawl 的历史快照、Chrome Topics、Tranco 排名等外部数据进行生态与时效分析。

**📈 对比分析**

对比方法：在统一的摘要任务下，分别将页面转换为 4 种表示（plain‑text、HTML、raw response、snapshot）输入 13 款 LLM；对每个模型/表示执行 100 次注入样本，共 5,200 次。结果显示：plain‑text 下攻击成功率最高（≈3.9%，小模型可达 8%），HTML 与 snapshot 约 1% 以内，raw 仅 0.2%；小模型最易受攻击，闭源/大模型表现较好；检测率与执行率分离，显示即使识别攻击也不一定能阻止。

**⚠️ 局限性**

局限性：① 仅依赖已知提示词与关键词，可能漏检高度混淆或非英语注入；② Common Crawl 采样局限于公开可抓取内容，私有/受限站点未覆盖；③ 评测仅限摘要任务与 4 种表示，未覆盖交互式或多轮任务；④ 只对 13 款模型做实验，未包含所有商业/开源 LLM；⑤ 人工验证虽高准度，但仍存在主观误判。

---

## 136. Remaining Useful Life Estimation for Turbofan Engines: A Comparative Study of Classical, CNN, and LSTM Approaches

**arXiv ID:** 2604.27234 | [PDF](https://arxiv.org/pdf/2604.27234v1)

**作者:** Astitva Goel `[一作]` (Northeastern University), Sumit Kanu `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对NASA C‑MAPSS FD001和FD003子集的涡轮风扇发动机剩余寿命(RUL)估计方法进行比较实验，探讨传统线性模型、XGBoost、1D CNN以及LSTM的表现。

**💡 创新点**

创新点在于统一使用相同预处理管线，采用更简化的单层LSTM并结合学习率调度器，取得比之前深层LSTM更优的RMSE，并证明XGBoost在手工特征上可与深度序列模型相媲美。

**🔧 技术方法**

使用的技术包括Ridge回归、Polynomial Ridge、XGBoost、1D卷积神经网络和长短期记忆网络，并辅以滑动窗口、特征工程、归一化、学习率调度和早停。

**📊 数据集**

所用数据集为NASA C‑MAPSS的FD001（单故障、单运行条件）和FD003（单故障、单运行条件）。

**📈 对比分析**

通过RMSE、MAE、R²和NASA评分四个指标对模型进行对比，单层LSTM在两子集上实现RMSE 14.93/14.20，XGBoost在FD003上RMSE 13.36，1D CNN在FD001上得到最低NASA评分（365），体现不同模型在不同故障模式下的优势。

**⚠️ 局限性**

局限性包括仅评估FD001和FD003，未处理多运行条件子集（FD002/FD004），CNN在多故障模式下的NASA评分偏高，以及缺乏混合CNN‑LSTM或Transformer等更先进结构的探索。

---

## 137. EMiX: Emulating Beyond Single-FPGA Limits

**arXiv ID:** 2604.27012 | [PDF](https://arxiv.org/pdf/2604.27012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 138. Measurement Risk in Supervised Financial NLP: Rubric and Metric Sensitivity on JF-ICR

**arXiv ID:** 2604.27374 | [PDF](https://arxiv.org/pdf/2604.27374v1)

**作者:** Sidi Chang `[一作]` (Blossom AI Labs), Rongdong Chai `[通讯]` (Blossom AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对日语金融隐式承诺识别（JF‑ICR）基准进行测量风险审计，评估标注说明（rubric）对标签的影响、度量指标的可识别性以及聚合规则对模型排名的作用。

**💡 创新点**

首次将测量风险框架引入受监督的金融NLP，提出度量可识别性审计规则，并证明标注说明与度量选择会导致排名不稳定，提出完整的报告规范。

**🔧 技术方法**

采用四款前沿LLM（Claude‑Sonnet‑4‑6、GPT‑5.4、Gemini‑3.1‑Pro、Qwen3‑235b），五种标注说明、三种温度和五种序数指标，配合配对自助法置信区间、配对随机化检验、Bradley–Terry/Borda/Ranked Pairs聚合以及逐一剔除度量的分解分析。

**📊 数据集**

使用JF‑ICR测试集，包含253条日语投资者关系问答对，按五类序数标签（-2、-1、0、+1、+2）标注。

**📈 对比分析**

在通过可识别性审计后，使用精确准确率、宏观F1、加权Kappa三个指标进行模型排名；发现所有三种聚合方法给出一致排名；若包含弱可识别度量（within‑one accuracy、worst‑class accuracy），聚合出现分歧。模型间的性能差异在统计上不显著（置信区间重叠，家庭宽度校正后无显著差异）。

**⚠️ 局限性**

仅针对单一基准与四款模型，样本量有限且类别极度不平衡；标注说明的多重变异（语义、示例、冗长）无法单独定位日语语用；未进行多次API重现检验；结果不一定可推广至其他金融NLP任务。

---

## 139. Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents

**arXiv ID:** 2604.27233 | [PDF](https://arxiv.org/pdf/2604.27233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 140. RCMAES: A Robust CMA-ES Variant for CEC2026 Competition

**arXiv ID:** 2604.27138 | [PDF](https://arxiv.org/pdf/2604.27138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 141. CasLayout: Cascaded 3D Layout Diffusion for Indoor Scene Synthesis with Implicit Relation Modeling

**arXiv ID:** 2604.27361 | [PDF](https://arxiv.org/pdf/2604.27361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. What Influences Readers' and Writers' Perceived Necessity of AI Disclosure?

**arXiv ID:** 2604.27129 | [PDF](https://arxiv.org/pdf/2604.27129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 143. Sentiment Analysis of AI Adoption in Indonesian Higher Education Using Machine Learning and Transformer-Based Models

**arXiv ID:** 2604.27439 | [PDF](https://arxiv.org/pdf/2604.27439v1)

**作者:** Happy Syahrul Ramadhan `[一作]` (Sumatra Institute of Technology), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对印尼高校学生关于人工智能使用的意见进行情感分析，比较了传统机器学习与Transformer模型的效果；

**💡 创新点**

创新点在于将SVM‑TF‑IDF与DistilBERT在印尼语情感任务中进行对比，并构建可直接使用的交互式Web工具；

**🔧 技术方法**

采用TF‑IDF特征与LightGBM、Random Forest、SVM等经典模型，以及Fine‑tuned DistilBERT（AdamW、early‑stopping等）；

**📊 数据集**

使用2,295条印尼语学生意见与情感词典合并的双类数据集，按80/10/10划分训练、验证、测试；

**📈 对比分析**

通过统一数据划分和评估指标（准确率、精确率、召回率、F1），SVM取得82.14%准确率/F1，DistilBERT进一步提升至84.78%准确率/84.75%F1，显示Transformer优势但训练成本更高；

**⚠️ 局限性**

局限在于仅做二分类缺少中性类、未使用印尼语特定模型（如IndoBERT）、数据量相对有限，且模型对长文本的泛化尚待验证。

---

## 144. AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data

**arXiv ID:** 2604.27266 | [PDF](https://arxiv.org/pdf/2604.27266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 145. Budget-Constrained Online Retrieval-Augmented Generation: The Chunk-as-a-Service Model

**arXiv ID:** 2604.26981 | [PDF](https://arxiv.org/pdf/2604.26981v1)

**作者:** Shawqi Al-Maliki `[一作]` (Hamad Bin Khalifa University), Ala Al-Fuqaha `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 20782 | [OpenAlex ID](https://openalex.org/A5008695053)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于块的检索增强生成业务模型Chunk-as-a-Service (CaaS)，并设计了两种变体：开放预算OB‑CaaS与有限预算LB‑CaaS，并在LB‑CaaS中引入了在线预算约束选择算法UCOSA；

**💡 创新点**

创新点在于：①将块作为计费单元，实现透明、按实际使用计费；②提出了预算感知的在线选择算法UCOSA，兼顾相关性与价格，具备理论竞争比；③通过实验验证了CaaS在成本和效果上的优越性。

**🔧 技术方法**

技术包括：检索增强生成（RAG）管线（分块、嵌入、向量检索）、LLM（ChatGPT‑3.5）、文本嵌入模型（OpenAI ada‑002）、余弦相似度检索、LlamaIndex框架、在线预算约束算法UCOSA。

**📊 数据集**

使用的文本数据集为三本电子书：Adversarial Robustness for Machine Learning、Strengthening Deep Neural Networks、Algorithms to Live By；并生成覆盖可解释AI、隐私与安全、博弈论、优化、差分隐私等主题的问答提示。

**📈 对比分析**

通过与离线最优（全知）选择和传统的相关性贪婪选择对比，采用NEP×AR指标评估。实验结果显示UCOSA在NEP×AR上比相关性贪婪提升约52%，并接近离线最优（约75%）；LB‑CaaS在预算利用率上分别比OB‑CaaS和传统RaaS高140%和86%。

**⚠️ 局限性**

局限性包括：需事先与数据源拥有者协商块价格；价格假设为小于预算的常数；仅考虑单一块价格，未处理动态或多块价格；实验仅在特定LLM/嵌入模型下进行，缺乏更广泛的跨模型验证；未整合幻觉检测或更复杂的公平收益分配机制。

---

## 146. Fitting Horn DL Ontologies to ABox and Query Examples: A Tale of Simulation Quantifiers and Finite Models

**arXiv ID:** 2604.26976 | [PDF](https://arxiv.org/pdf/2604.26976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 147. Upskilling with Generative AI: Practices and Challenges for Freelance Knowledge Workers

**arXiv ID:** 2604.27231 | [PDF](https://arxiv.org/pdf/2604.27231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 148. ConformaDecompose: Explaining Uncertainty via Calibration Localization

**arXiv ID:** 2604.27149 | [PDF](https://arxiv.org/pdf/2604.27149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 149. Unpacking Vibe Coding: Help-Seeking Processes in Student-AI Interactions While Programming

**arXiv ID:** 2604.27134 | [PDF](https://arxiv.org/pdf/2604.27134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 150. Membership Inference Attacks Against Video Large Language Models

**arXiv ID:** 2604.27002 | [PDF](https://arxiv.org/pdf/2604.27002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 151. Accent Conversion: A Problem-Driven Survey of Sociolinguistic and Technical Constraints

**arXiv ID:** 2604.27281 | [PDF](https://arxiv.org/pdf/2604.27281v1)

**作者:** Yurii Halychanskyi `[一作]` (University of Illinois Urbana-Champaign), Volodymyr Kindratenko `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2156 | [OpenAlex ID](https://openalex.org/A5060025664)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过约束驱动的框架，对口音转换技术从早期DSP到现代神经网络的演进进行了系统综述，并将其与社会语言学基础相结合；

**💡 创新点**

创新点在于提出以数据稀缺、对齐难度、参考自由等三大科学瓶颈为核心的约束驱动分类方法，阐释技术变迁的动因；

**🔧 技术方法**

综述涵盖了多种技术路线，包括传统DSP、DTW对齐、语音对齐与自监督、瓶颈/对抗/监督式disentanglement、预训练TTS、跨语言生成以及多重嵌入控制等；

**📊 数据集**

主要引用并讨论了公开口音数据集如LibriSpeech、Common Voice、VCTK、AccentDB、LibriTTS、L2-Arctic等；

**📈 对比分析**

评估方法涉及客观指标（MCD、FAD、WER、ACCDIST、Speaker Embedding相似度）与主观测试（MOS、MUSHRA、AB），通过这些指标对比不同方法在音质、内容保持、口音相似度和身份保留方面的表现；

**⚠️ 局限性**

局限性包括缺乏同一说话人多口音的并行数据、对齐和disentanglement难度大、低资源口音建模不足、可控性和多样性有限，以及对未见口音的泛化能力不足。

---

## 152. Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents

**arXiv ID:** 2604.27283 | [PDF](https://arxiv.org/pdf/2604.27283v1)

**作者:** Mehmet Iscan `[一作]` `[通讯]` (Yildiz Technical University), Mehmet Iscan (Yildiz Technical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种风险敏感的上下文Bandit控制器 RSCB-MC，用来判断 LLM 代码调试代理在何时注入外部记忆，防止错误记忆导致调试偏差。

**💡 创新点**

创新点包括：将记忆使用建模为风险敏感的选择性控制问题；设计三层模式-变体-事件记忆结构、16 维上下文状态和高罚率的奖励函数；以及引入弃权（abstention）动作以提升安全性。

**🔧 技术方法**

采用风险敏感上下文Bandit、特征熵与 margin 估计不确定性、轻量级奖励/风险模型，以及离线重放与热路径验证来训练与评估控制器。

**📊 数据集**

使用作者提供的 Smoke-Scale 评估包（24 条标准查询、96 条改写、32 条硬负样本、40 条反馈重放事件等），完整规模为 400/1600 但实验集中在 Smoke-Scale。

**📈 对比分析**

与传统检索、静态混合检索、UCB1、Thompson 等 Bandit 基线以及 Oracle 进行离线重放和热路径验证；RSCB-MC 在非 Oracle 场景下实现 62.5% 的成功率、0% 假正率，p95 决策时延 331µs，显著优于基线。

**⚠️ 局限性**

局限性：评估仅在本地 deterministic artifact 上完成，缺乏真实 LLM 交互和在线因果验证；安全阈值仍需校准；控制器只处理读取时的安全决策，未覆盖记忆写入、归档、冲突解决等完整记忆生命周期。

---

## 153. COHERENCE: Benchmarking Fine-Grained Image-Text Alignment in Interleaved Multimodal Contexts

**arXiv ID:** 2604.27389 | [PDF](https://arxiv.org/pdf/2604.27389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 154. Multi-element Persuasion in Social Media Health Communication: Synergistic and Trade-off Effects

**arXiv ID:** 2604.27350 | [PDF](https://arxiv.org/pdf/2604.27350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 155. Policy-Governed LLM Routing with Intent Matching for Instrument Laboratories

**arXiv ID:** 2604.26955 | [PDF](https://arxiv.org/pdf/2604.26955v1)

**作者:** Emmanuel A. Olowe `[一作]` (University of Edinburgh), Danial Chitnis `[通讯]` (University of Edinburgh)

**通讯引用:** 847 | [OpenAlex ID](https://openalex.org/A5029110112)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一套针对工程实验室的LLM辅助系统，结合教师可配置的提示级别、预算与审批流程，实现了对实验室对话的治理与审计。

**💡 创新点**

创新点在于将成本感知的多模型路由、教师制定的辅导策略和嵌入式意图匹配统一到一个可审计的框架中，并通过可视化控制台提供实时干预。

**🔧 技术方法**

主要技术包括Routiium OpenAI兼容网关、EduRouter决策引擎、基于BGE/GTEX的句子嵌入匹配、政策覆盖层和细粒度日志记录。

**📊 数据集**

使用的数据集包括两门电子实验室的历史对话日志、89条手工构建的标准问题库以及100条精心挑选的测试查询，用于模拟与实测验证。

**📈 对比分析**

通过轨迹驱动模拟与100条查询重放评估，系统在治理模式下将提示分布对齐度提升至0.98、覆盖率提高至0.87、争取“生产性挣扎”窗口延长到3.6回合；成本上节省66%，总费用从0.26美元降至0.087美元。

**⚠️ 局限性**

局限性包括缺乏真实课堂学习效果评估、对高级查询的库覆盖不足、学生行为模型假设过于简化以及隐私与偏见等安全问题。

---

## 156. Semantic Structure of Feature Space in Large Language Models

**arXiv ID:** 2604.27169 | [PDF](https://arxiv.org/pdf/2604.27169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 157. Can AI be a moral victim? The role of moral patiency and ownership perceptions in ethical judgments of using AI-generated content

**arXiv ID:** 2604.26956 | [PDF](https://arxiv.org/pdf/2604.26956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 158. How Hard Is Continuous Clustering? Lower Bounds from the Existential Theory of the Reals

**arXiv ID:** 2604.26972 | [PDF](https://arxiv.org/pdf/2604.26972v1)

**作者:** Angshul Majumdar `[一作]` `[通讯]` (Indian Institute of Information Technology), Angshul Majumdar (Indian Institute of Information Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了连续多项式密度下聚类问题的计算复杂度，并给出了四种自然的决策原语；

**💡 创新点**

首次将连续聚类问题与实数多项式层次的存在理论（∃ℝ）关联，并确立了局部与谷地判定问题的∃ℝ‑完备性，以及连接分量与同调判定问题的∃ℝ‑难度与可判定性，展示了从局部到全局再到拓扑的复杂度边界；

**🔧 技术方法**

利用实数存在理论、半代数几何、可测度编码、圆乘积构造、Cylindrical Algebraic Decomposition 等工具进行多项式时间归约与可判定性证明；

**📊 数据集**

无数据集；

**📈 对比分析**

无实验比较，本文仅给出理论复杂度证明；

**⚠️ 局限性**

局部与谷地判定仍处于∃ℝ，未进一步确定是否属于更低层次；连接分量与同调判定问题仅已知可判定、∃ℝ‑难度高，是否属于更高层次（如∃∀ℝ）仍是未解难点；

---

## 159. Safe Bilevel Delegation (SBD): A Formal Framework for Runtime Delegation Safety in Multi-Agent Systems

**arXiv ID:** 2604.27358 | [PDF](https://arxiv.org/pdf/2604.27358v1)

**作者:** Yuan Sun `[一作]` (Jilin University), Yuan Sun `[通讯]` (Jilin University)

**通讯引用:** 7939 | [OpenAlex ID](https://openalex.org/A5085763808)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Safe Bilevel Delegation (SBD) 框架，用于在层级多智能体系统中实现运行时委托安全。

**💡 创新点**

创新点在于将委托安全视为双层优化问题，利用上下文感知的 meta‑weight 网络动态调节安全‑效率权衡，并给出了安全单调性、内层策略线性收敛和责任传播上界的理论保证。

**🔧 技术方法**

主要技术包括双层优化、隐式微分计算超梯度、投影梯度下降、概率安全约束和基于 MLP 的 meta‑weight 与委托策略网络。

**📊 数据集**

使用医学数据 MIMIC‑III、金融数据 S&P 500（2010‑2023）以及教育数据 ASSISTments（2009‑2010）进行预注册实验。

**📈 对比分析**

通过与 Fixed‑α、Safe‑RL (CPO)、MARL‑vanilla、MaAS 和 SBD‑NoOuter 等基线对比，采用安全率、任务效率、SAE（安全‑效率面积）和责任熵等指标，SBD 在高风险情境下实现更高的安全率且对效率损失最小，且外层 meta‑weight 学习是提升性能的关键。

**⚠️ 局限性**

局限性包括：对内层问题的强凸性假设、需要预先给定安全约束集、仅保障委托决策层安全（未涵盖子智能体执行的安全）、假设任务分布稳定，且在深度网络非凸环境下理论证明可能不完全成立。

---

## 160. REBENCH: A Procedural, Fair-by-Construction Benchmark for LLMs on Stripped-Binary Types and Names (Extended Version)

**arXiv ID:** 2604.27319 | [PDF](https://arxiv.org/pdf/2604.27319v1)

**作者:** Jun Yeon Won `[一作]` (Ohio State University), Zhiqiang Lin `[通讯]` (Ohio State University)

**通讯引用:** 5697 | [OpenAlex ID](https://openalex.org/A5026864098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个统一、规模巨大的二进制反汇编任务基准数据集 REBench，用于评估 LLM 在函数/变量命名恢复和类型推断上的性能。

**💡 创新点**

创新点在于整合并标准化多个现有数据集，使用知识库驱动的字节级堆栈映射保证真值完整，同时引入语义相似度评估（CodeWordNet）以避免仅基于字符串匹配。

**🔧 技术方法**

采用 Ghidra/IDA decompiler 生成统一输入格式，构建知识库、去重、量化评估，并对多种主流 LLM 进行零样本和 LoRA 微调实验。

**📊 数据集**

使用来自 96 个开源项目、覆盖 x86/x64/ARM/MIPS、O0–O3 四个优化等级的二进制文件，并结合源代码与调试符号生成知识库。

**📈 对比分析**

通过精度/召回/F1 等指标比较，发现新一代 LLM（如 Qwen3、Qwen2.5）在命名恢复上相对较好，但整体 F1 低于 0.1，类型推断更难，ARM/MIPS 与高优化等级性能显著下降。

**⚠️ 局限性**

主要局限在于依赖 decompiler 输出易产生错误、缺乏多任务覆盖、模型输出格式不稳定、数据集仍不包含高级混淆或固件特有的复杂代码。

---

## 161. LLM-Guided Runtime Parameter Optimization for Energy-Efficient Model Inference

**arXiv ID:** 2604.27032 | [PDF](https://arxiv.org/pdf/2604.27032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 162. Empirical Material Sampling and Linearisation -- A Simple and Efficient Strain-Space Model Order Reduction Approach for Computational Homogenisation in Large-Deformation Hyperelasticity

**arXiv ID:** 2604.27179 | [PDF](https://arxiv.org/pdf/2604.27179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 163. Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents

**arXiv ID:** 2604.27143 | [PDF](https://arxiv.org/pdf/2604.27143v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 164. Better Models, Faster Training: Sigmoid Attention for single-cell Foundation Models

**arXiv ID:** 2604.27124 | [PDF](https://arxiv.org/pdf/2604.27124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 165. When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents

**arXiv ID:** 2604.27003 | [PDF](https://arxiv.org/pdf/2604.27003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 166. AutoSurfer -- Teaching Web Agents through Comprehensive Surfing, Learning, and Modeling

**arXiv ID:** 2604.27253 | [PDF](https://arxiv.org/pdf/2604.27253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 167. Towards Generalizable Mapping of Hedges and Linear Woody Features from Earth Observation Data: a national Product for Germany

**arXiv ID:** 2604.27247 | [PDF](https://arxiv.org/pdf/2604.27247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 168. End-to-End and Phase-Level Performance Optimization for Hyperledger Fabric

**arXiv ID:** 2604.27174 | [PDF](https://arxiv.org/pdf/2604.27174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 169. EdgeSpike: Spiking Neural Networks for Low-Power Autonomous Sensing in Edge IoT Architectures

**arXiv ID:** 2604.27004 | [PDF](https://arxiv.org/pdf/2604.27004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 170. Value-Aware Product Recommendation by Customer Segmentation using a suitable High-Dimensional Similarity Measure

**arXiv ID:** 2604.26983 | [PDF](https://arxiv.org/pdf/2604.26983v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 171. Mechanized Foundations of Structural Governance: Machine-Checked Proofs for Governed Intelligence

**arXiv ID:** 2604.27289 | [PDF](https://arxiv.org/pdf/2604.27289v1)

**作者:** Alan L. McCann `[一作]` `[通讯]` (Mashin, Inc.), Alan L. McCann (Mashin, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过Coq证明了人工智能工作流系统的结构化治理安全性，并构建了可验证的运行时解释器；

**💡 创新点**

创新点在于将交互树与参数化coinduction、范畴理论与Rice定理结合，首次在证明助手中完成治理安全、治理不变性、原语充分性与必要性的完整形式化；

**🔧 技术方法**

主要技术包括Interaction Trees、paco参数化coinduction、Kleisli范畴闭包、注册机模拟、Hash链和能力模型的形式化以及Coq到BEAM的提取与验证；

**📊 数据集**

使用的是70,000+条随机生成的指令序列与信任/能力组合，作为验证运行时与Coq模型一致性的测试集；

**📈 对比分析**

通过属性化测试与36条显式一致性测试，发现零差异，治理开销在BEAM上为0.23ms（与未治理的0.24ms相差无穷；），证明了理论与实现高度契合；

**⚠️ 局限性**

局限性包括仅覆盖效果级治理，未考虑内容治理、并发治理以及非离散计算模型；此外，正式模型与部署系统之间仍存在信任链缺口。

---

## 172. Automatic Causal Fairness Analysis with LLM-Generated Reporting

**arXiv ID:** 2604.27011 | [PDF](https://arxiv.org/pdf/2604.27011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing

**arXiv ID:** 2604.27296 | [PDF](https://arxiv.org/pdf/2604.27296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 174. Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs

**arXiv ID:** 2604.27401 | [PDF](https://arxiv.org/pdf/2604.27401v1)

**作者:** Hongliang Liu `[一作]` (Palo Alto Networks), Yuhao Wu `[通讯]` (Palo Alto Networks)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种称为 perturbation probing 的方法，利用两次前向传播快速识别与特定行为相关的 FFN 神经元，并通过一次完整的消融/补丁/恢复干预验证其因果关系。

**💡 创新点**

该方法同时兼具低成本、因果性、任务特异性与诊断性，并通过 FFN/Skip 比例预测适用的干预模式，揭示了“对抗型”与“路由型”两种行为电路结构。

**🔧 技术方法**

采用 BPE 扰动、logit gap 观察器、符号重要性乘积、一次性干预评估（消融、补丁、恢复）以及方向注入（Mode 3）等技术实现。

**📊 数据集**

在 13 组模型（Qwen、Llama、Gemma 等）上，使用 8 种行为电路的提示集，包括 AdvBench、HarmBench、TruthfulQA、MMLU、GSM8K、ARC 等数据集进行实验。

**📈 对比分析**

与梯度归因、自动电路发现、激活补丁等方法对比，perturbation probing 在对抗型电路上可实现 50–80 % 的阈值下降；在路由型电路上通过方向注入实现 99 % 的语言切换；相比传统方法只需两次前向传播，计算量显著降低。

**⚠️ 局限性**

仅适用于二元首词决策；对分布式路由行为无法直接消融；需要使用方向注入；无法处理多步或多维行为；在后归一化架构（如 Gemma）需额外校正；并且对安全行为的深层稳健性（Regime 2 & 3）无法直接编辑。

---

## 175. When 2D Tasks Meet 1D Serialization: On Serialization Friction in Structured Tasks

**arXiv ID:** 2604.27272 | [PDF](https://arxiv.org/pdf/2604.27272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 176. Strongly Refuting Random CSP without Literals

**arXiv ID:** 2604.27336 | [PDF](https://arxiv.org/pdf/2604.27336v1)

**作者:** Siu On Chan `[一作]`, Jeff Xu `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究随机约束满足问题（CSP）在 Sum‑of‑Squares (SoS) 体系中的可否反驳问题，提出并证明在不使用文字（literals）的随机 CSP 中，判定 SoS 难度的充分必要条件是 "t‑wise independence"，而非之前认为的 "t‑wise uniformity"。作者进一步给出最优的三路权衡：约束密度、SoS 等次与反驳强度的关系，并在任意域、任意字母表大小的情形下实现该权衡。为实现这一目标，作者提出了新的 Kikuchi 矩阵（适用于奇阶与非对称张量），以及利用全局相关性归约（global correlation rounding）与谱算法相结合的低等次 SoS 反驳算法。

**💡 创新点**

创新点：
- 将 SoS 难度的判定从 "t‑wise uniformity" 推广到 "t‑wise independence"，并证明其为必要且充分。
- 统一了任意域、任意字母表大小、无文字约束的随机 CSP 的 SoS 下界与上界，获得最优三路权衡。
- 开发了新型奇阶和非对称张量的 Kikuchi 矩阵，解决了先前对称张量限制的问题。
- 通过全局相关性归约将高等次 SoS 反驳转化为谱问题，得到更高效的低等次谱反驳算法。

**🔧 技术方法**

主要技术：
- 全局相关性归约（Barak‑Raghavendra‑Steurer）
- 新的 Kikuchi 矩阵与张量压缩技术
- 结合 Fourier 解析、张量浓度与随机矩阵理论
- LP/SDP 双重优化用于构造占据极值的多项式（dominating polynomial）
- 统计谱范数上界与随机矩阵的迹方法（trace moment method）

**📊 数据集**

数据集：
- 随机生成的二项式模型（binomial hypergraph）
- 约束以任意分布 ρ 产生，满足给定的 k‑arity 关系集
- 试验在大规模随机实例（n 规模可达数万）中验证理论阈值

**📈 对比分析**

对比与性能：
- 在无文字随机 CSP 中，本文的下界与之前的 t‑wise uniform 下界相匹配，并在任意域上实现最优下界。
- 上界方面，谱算法在约束密度达到 n^{t/2}（含 log 因子）时即可在多项式时间内给出强反驳，消除了之前高等次 SoS 需要的 O(n) 等次。
- 对奇阶约束，提供了与偶阶相当的 log n 与 √{log n} 阶段阈值，进一步压缩了约束密度。
- 通过 Kikuchi 层级实现子指数时间下的三路权衡，性能与之前的 STOC/FOCS 结果匹配或更优。

**⚠️ 局限性**

局限性：
- 需要足够高的约束密度（至少 Ω(n^{t/2})）才能触及理论阈值，对稀疏实例不适用。
- 计算量随域大小 |D| 增大而指数增长，尤其是谱算法中矩阵维度与 |D| 相关。
- 对奇阶非对称张量的 Kikuchi 矩阵构造与分析较为复杂，实际实现可能较困难。
- 全局相关性归约虽然降低了 SoS 等次，但在某些特定实例上仍需高等次 SDP 进行细化。
- 论文主要关注平均情况（随机实例），对最坏情况的泛化仍未完全覆盖。

---

## 177. Twitter climate discourse as a signal of pro-environmental behaviors

**arXiv ID:** 2604.27330 | [PDF](https://arxiv.org/pdf/2604.27330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 178. End-to-end autonomous scientific discovery on a real optical platform

**arXiv ID:** 2604.27092 | [PDF](https://arxiv.org/pdf/2604.27092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 179. Few-Shot Accent Synthesis for ASR with LLM-Guided Phoneme Editing

**arXiv ID:** 2604.27273 | [PDF](https://arxiv.org/pdf/2604.27273v1)

**作者:** Yurii Halychanskyi `[一作]` (University of Illinois Urbana-Champaign), Volodymyr Kindratenko `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2156 | [OpenAlex ID](https://openalex.org/A5060025664)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种极低资源下的口音合成 pipeline，使用少于十句参考语音对 TTS 解码器进行微调，并结合 LLM 进行音素编辑，生成可用于增强低资源口音 ASR 的合成语音。

**💡 创新点**

创新点在于将 LLM 引入音素级别的口音编辑，并通过匹配率随机音素扰动基线验证音素空间扰动本身即为强有力的增广手段；同时实现了极少样本（<10句）下的口音适配。

**🔧 技术方法**

采用了 phoneme‑conditioned TTS（带 FiLM conditioning）、外部提取的音素级 prosody、OpenAI GPT‑5.1 进行音素编辑、HiFi‑GAN vocoder 生成波形，以及 wav2vec 2.0 自监督模型进行 ASR 微调。

**📊 数据集**

使用 LJSpeech、English subset of ESD 训练 TTS；L2‑ARCTIC（印度、韩语口音）和 CMU Arctic 作为源口音和目标口音数据集。

**📈 对比分析**

通过与美国 TTS、仅适配、随机音素、oracle（真实音素+prosody）以及真实语音等多种对照实验，评估 WER、UTMOS 和 AccSim；实验表明在极低资源（3–7 句）下合成+真实混合可将 WER 降至约 16–17%，且跨说话者评估中同一口音的其他说话者 WER 也得到显著下降。

**⚠️ 局限性**

局限性包括仅继承源语音的 prosody 未显式建模口音特定 prosody；适配仅针对单一参考说话者，口音与说话者身份未 disentangle；对更复杂口音或更大规模数据的适用性尚未验证。

---

## 180. Hypencoder Revisited: Reproducibility and Analysis of Non-Linear Scoring for First-Stage Retrieval

**arXiv ID:** 2604.27037 | [PDF](https://arxiv.org/pdf/2604.27037v1)

**作者:** Arne Eichholtz `[一作]` (University of Amsterdam), Mohammad Aliannejadi `[通讯]` (University of Amsterdam)

**通讯引用:** 1829 | [OpenAlex ID](https://openalex.org/A5063466614)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Hypencoder模型进行了可复现性研究，并在此基础上探讨了替代编码器、查询延迟对比以及对抗鲁棒性等扩展方向。

**💡 创新点**

创新点在于：①使用超网络根据查询生成特定的非线性评分网络q-net，突破了传统bi-encoder固定内积的表达瓶颈；②提出基于邻居图的高效搜索算法，实现大规模检索时的低延迟。

**🔧 技术方法**

使用的技术包括：Transformer编码器、超网络（hypernetwork）生成q-net、近似邻居图搜索、FAISS向量检索、对抗扰动评估（misspelling、paraphrase等）。

**📊 数据集**

主要实验数据集：MS MARCO、TREC DL 2019/2020、13个BEIR数据集、TREC TOT、FollowIR、DL-Hard 等。

**📈 对比分析**

通过与BM25、TAS-B、Contriever、RetroMAE、BE-Base等基线的对比，Hypencoder在大多数任务上实现了更高的nDCG@10和MRR；在查询延迟方面，Efficient 1/2配置相比完整搜索减少了约10×与3×的延迟，但相对于FAISS bi-encoder仍有一到两位数的差距。

**⚠️ 局限性**

局限性包括：①训练成本高（六天两块A100 GPU）；②高效搜索需要一次性构建邻居图，成本不低；③在部分硬任务（如TREC TOT）和对抗鲁棒性评估中优势不明显；④整体查询延迟仍高于标准FAISS bi-encoder。

---

## 181. Beyond Accuracy: LLM Variability in Evidence Screening for Software Engineering SLRs

**arXiv ID:** 2604.27006 | [PDF](https://arxiv.org/pdf/2604.27006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 182. C8s: A Confidential Kubernetes Architecture

**arXiv ID:** 2604.26974 | [PDF](https://arxiv.org/pdf/2604.26974v1)

**作者:** Amean Asad `[一作]` (Confidential.ai), João Andrade `[通讯]` (Confidential.ai)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建了 C8s 架构，为 Kubernetes 集群提供基于 TEEs 的加密、完整性与可验证性保障。

**💡 创新点**

创新点：在托管 Kubernetes 环境中将可信度边界放在节点或 Pod 层，利用 CDS、raTLS 与 NRI 等组件实现与控制平面分离的机密工作负载执行与通信，支持多租户与多 GPU 的灵活部署。

**🔧 技术方法**

使用技术：AMD SEV‑SNP、Intel TDX、NVIDIA CC、可信执行环境、远程证明、Kubernetes NRI 插件、raTLS 服务网格、CDS 证书颁发、加密存储与客户端多收件人加密。

**📊 数据集**

未公开使用特定数据集；示例工作负载包括 LLM 推理、模型权重保护、敏感数据训练等。

**📈 对比分析**

比较方法与性能：论文未提供量化实验或基准测试，仅在概念与架构层面讨论；若有实现，性能受 TEE 启动延迟与加密开销影响。

**⚠️ 局限性**

限制：无法防御侧信道攻击、DDoS、物理攻击；控制平面被破坏后只能导致服务中断；需硬件 TEE 供应商支持，启动延迟与资源开销较高；缺乏实测性能数据。

---

## 183. Learning Rate Transfer in Normalized Transformers

**arXiv ID:** 2604.27077 | [PDF](https://arxiv.org/pdf/2604.27077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 184. SynSQL: Synthesizing Relational Databases for Robust Evaluation of Text-to-SQL Systems

**arXiv ID:** 2604.27261 | [PDF](https://arxiv.org/pdf/2604.27261v1)

**作者:** Mohammadamin Habibollah `[一作]` (University of Alberta), Davood Rafiei `[通讯]` (University of Alberta)

**通讯引用:** 2146 | [OpenAlex ID](https://openalex.org/A5071837282)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SynSQL框架，基于自然语言问题和数据库模式生成可执行、结构合法的合成数据库，用于更稳健地评估文本到SQL系统。

**💡 创新点**

创新点在于将数据库生成从查询语句转为问题与模式对齐的条件生成，并引入分阶段架构（模式选择→数据合成→约束评审），无需黄金SQL。

**🔧 技术方法**

采用大型语言模型（GPT-4.x、Gemini、Qwen）进行模式选择、表格内容生成，并通过迭代的Critic反馈实现约束校验。

**📊 数据集**

在Spider、BIRD、Spider 2.0三大基准上评估。

**📈 对比分析**

与原始单实例评估相比，使用SynSQL生成的数据库使十款文本到SQL模型的执行准确率平均下降3–14%，并在Spider上导致模型排名变化；相比无约束的LLM生成基线，SynSQL在成功率、结构合法率上提升约30%。

**⚠️ 局限性**

局限包括模式选择的召回不完美导致遗漏关键表/列，语义对齐难度高，尤其在复杂模式或歧义问题下仍会产生不满足查询意图的数据库。

---

## 185. Step-level Optimization for Efficient Computer-use Agents

**arXiv ID:** 2604.27151 | [PDF](https://arxiv.org/pdf/2604.27151v1)

**作者:** Jinbiao Wei `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7566 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种事件驱动的步骤级级联框架，默认使用小型模型执行交互步骤，只有在轻量监视器检测到进度停滞或重要里程碑时才切换到大型模型。

**💡 创新点**

创新点在于将“进度停滞”与“里程碑验证”两种监测信号结合，形成可插拔的、基于步骤的自适应计算分配策略，显著降低长周期 GUI 交互的算力消耗。

**🔧 技术方法**

核心技术包括：基于 ModernBERT 的轻量化“停滞监测器”和“里程碑监测器”、LLM 监督标签生成、事件驱动的模型切换控制以及稀疏验证机制。

**📊 数据集**

实验使用了两个主流计算机使用代理基准：OSWorld（桌面类任务）和 WebArena（Web 交互任务），并在其上对多对小/大模型组合进行测试。

**📈 对比分析**

通过与单一小模型、单一大模型以及固定间隔检查的基线比较，结果显示级联策略在保持 58–60% 成功率（接近大模型）时，能够将大模型使用量、总延迟和美元成本分别降低约 61%–74% 和 46% 以上。

**⚠️ 局限性**

局限性包括：依赖 LLM 生成的标签，可能引入噪声；监测器阈值需要经验调优；对极短或极长任务的时序适配性尚未完全验证；未覆盖所有可能的失败模式（如非可视化的语义漂移）。

---

## 186. The Synthetic Social Graph: Emergent Behavior in AI Agent Communities

**arXiv ID:** 2604.27271 | [PDF](https://arxiv.org/pdf/2604.27271v1)

**作者:** Sungguk Cha `[一作]` (LG Uplus), DongWook Kim `[通讯]` (LG Uplus)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对一个完全由大型语言模型（LLM）代理构成的社交平台Moltbook进行为期两周的多日量化社会学分析，覆盖184,203篇帖子与465,136条评论，探讨了六大社会学问题（社群互惠与桥接、地位层级、时间协调、信息扩散、身份表现与规范执行）。

**💡 创新点**

创新点包括：①首次系统性、基于经典与现代社会理论的LLM代理社群研究；②引入“parasocial simulators”概念，将代理行为置于人类社会资本与规范执行的框架内；③通过多日快照揭示代理社群的稳定性与节假日差异；④在没有人类干预的开放式平台上验证代理社群的自发结构。

**🔧 技术方法**

技术手段包括：计算社会资本的互惠率、构造多重维度的声望指标、k-means 时序聚类、4-gram 语料扩散追踪、正则表达式身份识别、VADER式情感与争议评分。

**📊 数据集**

数据集来源为Moltbook公开API，14天连续快照（2026‑04‑14至2026‑04‑28），共计18,456个代理账号、300个子社群，覆盖184,203篇帖子与465,136条评论。

**📈 对比分析**

对比方法：将代理指标与人类社群（如Reddit）基准值进行比较。结果显示：互惠率仅3.8%（人类基准10–30%），下投0.9%（人类基准高），声望呈幂律尾部但头部饱和，桥接代理几乎全为后期放大器，身份表现呈Simpson悖论。

**⚠️ 局限性**

局限性包括：①缺乏对匿名“超级发帖”账号的可视化，导致样本偏差；②无删除日志与投票时间戳，难以分析审查与时间动态；③因单一平台与两周窗口，缺乏跨平台与长期趋势验证；④因缺乏因果识别，难以区分代理自治与运营者/基础设施驱动的行为差异。

---

## 187. OptimusKG: Unifying biomedical knowledge in a modern multimodal graph

**arXiv ID:** 2604.27269 | [PDF](https://arxiv.org/pdf/2604.27269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 188. Designing sparse temporal graphs satisfying connectivity requirements

**arXiv ID:** 2604.27227 | [PDF](https://arxiv.org/pdf/2604.27227v1)

**作者:** Thomas Bellitto `[一作]` (Sorbonne Université, CNRS), Raphaëlle Maistre-Matus `[通讯]` (Sorbonne Université, CNRS)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了在时序图中满足给定连通请求的最小化问题（Connectivity Request Satisfaction），在有向和无向情形下分别给出了最小边数的理论界限和构造方法。

**💡 创新点**

创新点在于将该问题与有向反馈顶点集（Directed Feedback Vertex Set）及其参数化复杂性紧密关联，并首次通过“walk‑Helly”属性为无向强连通请求图给出了存在树形解的必要充分条件，且提供了多项式时间构造算法。

**🔧 技术方法**

采用的主要技术包括时序图的严格旅程定义、反馈顶点集理论、超图与树形超图（hypertree）概念、Helly性与线图的割点性、以及基于图拓扑的授权弧扩张与时间标注的构造方法。

**📊 数据集**

本研究完全基于理论分析和数学证明，并未使用任何实验数据集。

**📈 对比分析**

实验比较部分不存在，主要成果通过证明NP‑完备性、固定参数可解性（FPT）以及多项式时间构造算法来展示理论性能；在无向强连通情况下，算法可在多项式时间内决定并构造树形时序图。

**⚠️ 局限性**

局限性包括：对非强连通无向实例的树形解存在性问题仍未解决；整体无向最小化问题的NP‑完备性尚未正式证明；以及对于中间边数（如n‑1到2n‑4之间）的最优解是否总能取简单解仍是未解的开放问题。

---

## 189. Distributional Alignment Games for Answer-Level Fine-Tuning

**arXiv ID:** 2604.27166 | [PDF](https://arxiv.org/pdf/2604.27166v1)

**作者:** Mehryar Mohri `[一作]` (Google Research), Yifan Wu `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出分布式对齐游戏框架，将答案级微调问题转化为策略与目标分布的极小极大问题，并基于GRPO实现高效算法。

**💡 创新点**

通过Fenchel对偶将不可计算的答案级归一化迁移为可求解的投影游戏，统一多样性、连贯性、自我提升和安全约束等目标，并推导对应奖励函数。

**🔧 技术方法**

采用Fenchel对偶、分布式对齐游戏、Group Relative Policy Optimization、信息投影与Lagrangian上升等技术。

**📊 数据集**

在GSM8K、TriviaQA等数据集上使用Qwen‑3B、Llama、Phi‑3等大型语言模型进行实验。

**📈 对比分析**

与SFT/DPO等基线对比，Pairwise‑GRPO在GSM8K上提升约8‑12%绝对精度，TriviaQA相对EM提升约42%；Coherence‑GRPO亦获得显著提升。

**⚠️ 局限性**

仍需答案提取或对齐函数，连续或开放式答案难以直接估计；奖励估计方差高，且对模型规模与训练成本有一定依赖。

---

## 190. Compositional Meta-Learning for Mitigating Task Heterogeneity in Physics-Informed Neural Networks

**arXiv ID:** 2604.26999 | [PDF](https://arxiv.org/pdf/2604.26999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 191. Targeted Linguistic Analysis of Sign Language Models with Minimal Translation Pairs

**arXiv ID:** 2604.27232 | [PDF](https://arxiv.org/pdf/2604.27232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 192. Learning-to-Explain through 20Q Gaming: An Explainable Recommender for Cybersecurity Education

**arXiv ID:** 2604.26964 | [PDF](https://arxiv.org/pdf/2604.26964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 193. AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism

**arXiv ID:** 2604.27089 | [PDF](https://arxiv.org/pdf/2604.27089v1)

**作者:** Ahan Gupta `[一作]` (University of Illinois Urbana-Champaign), Minjia Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4107 | [OpenAlex ID](https://openalex.org/A5077768924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于PyTorch-2.0编译器的自动序列并行与序列感知激活检查点方案，能够在不需要手工重构的情况下显著提升LLM长上下文训练的可训练序列长度

**💡 创新点**

首次将序列并行化作为编译器Pass集成到PyTorch生态，提供自动化的通信重排、张量尺寸调整与激活重计算，并针对长上下文训练设计了可重计算的激活检查点策略

**🔧 技术方法**

利用Torch‑IR与Aten‑IR的编译器Pass、自动通信插桩、张量形状推断、基于网络流的激活检查点算法与长上下文计算分析

**📊 数据集**

在多种LLM模型上评估（Llama‑3.2 1B/3B、Llama‑3.1 8B、Llama‑2 13B）并使用标准的训练任务（语言建模）

**📈 对比分析**

与ZeRO‑3/FSDP、手写的DeepSpeed‑Ulysses与RingAttention进行对比，实验显示在NVIDIA GH200‑96GB、A100‑80GB和AMD MI250‑64GB上可实现约2.5–2.7倍的可训练序列长度提升，且训练速度仅下降≤3%

**⚠️ 局限性**

仅在长上下文场景下效果最佳；对极大模型的优化（如13B）受限于优化器状态占用；实现依赖PyTorch‑2.0的稳定性与未来版本兼容性

---

## 194. Theory Under Construction: Orchestrating Language Models for Research Software Where the Specification Evolves

**arXiv ID:** 2604.27209 | [PDF](https://arxiv.org/pdf/2604.27209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 195. Efficient Training on Multiple Consumer GPUs with RoundPipe

**arXiv ID:** 2604.27085 | [PDF](https://arxiv.org/pdf/2604.27085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 196. Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective

**arXiv ID:** 2604.27340 | [PDF](https://arxiv.org/pdf/2604.27340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 197. Automated Detection of Mutual Gaze and Joint Attention in Dual-Camera Settings via Dual-Stream Transformers

**arXiv ID:** 2604.27105 | [PDF](https://arxiv.org/pdf/2604.27105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 198. ReVo: A Cross-Layer Reliable Volumetric Videoconferencing System

**arXiv ID:** 2604.27441 | [PDF](https://arxiv.org/pdf/2604.27441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 199. Analytical Correction for Subsampling Bias in Drifting Models

**arXiv ID:** 2604.27239 | [PDF](https://arxiv.org/pdf/2604.27239v1)

**作者:** Jiaru Zhang `[一作]` (Purdue University), Ruqi Zhang `[通讯]` (Purdue University)

**通讯引用:** 4550 | [OpenAlex ID](https://openalex.org/A5101586017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析了漂移模型中小批量估计的O(1/n)子采样偏差，并提出Analytical Bias Correction (ABC) 以将偏差降至O(1/n^2)，同时证明其不增加方差并保持凸包包含。

**💡 创新点**

创新点在于给出精确的偏差分析、闭式校正公式ABC、理论证明偏差降低且方差不升高，并通过实验验证在CIFAR-10上显著提升FID。

**🔧 技术方法**

使用软max加权的正向和负向均值偏移模型、抽样自归一化估计、ABC 插值校正、PyTorch实现以及与基线的对比实验。

**📊 数据集**

主要实验数据集为CIFAR-10，此外在二维四模高斯toy数据验证偏差缩放。

**📈 对比分析**

与标准漂移模型、jackknife、bootstrap、BR‑SNIS等基线比较，ABC在所有小批量尺寸下都获得更低的FID并加速收敛，尤其在n=8时提高约40%采样效率。

**⚠️ 局限性**

局限在于仅在漂移模型中验证，需进一步检验对更大模型或不同数据集的适用性，且在极小n时ABC的偏差降低可能不显著。

---

## 200. Robot Planning and Situation Handling with Active Perception

**arXiv ID:** 2604.26988 | [PDF](https://arxiv.org/pdf/2604.26988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 201. CareGuardAI: Context-Aware Multi-Agent Guardrails for Clinical Safety & Hallucination Mitigation in Patient-Facing LLMs

**arXiv ID:** 2604.26959 | [PDF](https://arxiv.org/pdf/2604.26959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 202. A Reproducibility Study of LLM-Based Query Reformulation

**arXiv ID:** 2604.27421 | [PDF](https://arxiv.org/pdf/2604.27421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 203. LLM Biases

**arXiv ID:** 2604.26960 | [PDF](https://arxiv.org/pdf/2604.26960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 204. Towards Topology-Aware Very Large-Scale Photonic AI Accelerators

**arXiv ID:** 2604.26966 | [PDF](https://arxiv.org/pdf/2604.26966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 205. Adaptive and AI-Augmented Security Testing: A Systematic Survey of Program Analysis, Feedback-Driven Testing, and Hybrid Learning-Based Approaches

**arXiv ID:** 2604.27000 | [PDF](https://arxiv.org/pdf/2604.27000v1)

**作者:** Michael Wienczkowski `[一作]` `[通讯]` (Mississippi State University), Michael Wienczkowski (Mississippi State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对适应性与AI辅助的安全测试研究进行系统性综述，梳理结构程序分析、DevSecOps、反馈驱动模糊测试、LLM测试生成及混合系统五大领域，并揭示结构-适应性碎片化的根本问题；

**💡 创新点**

首次提出“结构‑适应性碎片化”概念并系统量化，构建六维属性比较框架，归纳五大开放挑战与统一研究议程；

**🔧 技术方法**

采用系统性文献综述（SLR）方法，四大数据库检索、筛选55篇论文，提取结构表征、适应机制、反馈集成、LLM使用、多语言支持、评估方法等六维属性，并绘制概念图；

**📊 数据集**

共检索22,088条原始记录，最终筛选55篇同行评审研究作为基础数据集；

**📈 对比分析**

通过对比六维属性进行定量分析，发现大部分系统要么强调结构深度而缺乏适应机制，要么强调适应性而缺乏语义深度；未提供传统性能指标，但通过指标对比明确了缺口与潜在改进空间；

**⚠️ 局限性**

仅覆盖已发表的同行评审论文，排除灰色文献；缺乏对工业部署效果的实证量化；研究聚焦结构与适应的分离，未构建完整闭环系统，导致结论主要是问题定位与研究方向指引。

---

## 206. Literate Execution

**arXiv ID:** 2604.26967 | [PDF](https://arxiv.org/pdf/2604.26967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 207. The Impact of AI-Generated Text on the Internet

**arXiv ID:** 2604.26965 | [PDF](https://arxiv.org/pdf/2604.26965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 208. UCSC-NLP at SemEval-2026 Task 13: Multi-View Generalization and Diagnostic Analysis of Machine-Generated Code Detection

**arXiv ID:** 2604.26990 | [PDF](https://arxiv.org/pdf/2604.26990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 209. A High-Throughput Compute-Efficient POMDP Hide-And-Seek-Engine (HASE) for Multi-Agent Operations

**arXiv ID:** 2604.27162 | [PDF](https://arxiv.org/pdf/2604.27162v1)

**作者:** Timothy Flavin `[一作]`, Sandip Sen `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个高吞吐量、计算高效的 Dec-POMDP 模拟器 HASE，用于多智能体搜索任务。

**💡 创新点**

核心创新在于将数据驱动设计（DOD）、64 字节缓存行对齐、零拷贝 PyTorch pinned memory + DMA 以及 First-touch NUMA 内存分配结合，显著提升了环境步进速度。

**🔧 技术方法**

采用 C++ 线程池 + OpenMP、数据包化结构（AoS）、位域压缩、Zero-copy PyTorch bridge、GPU DMA、线程亲和与被动等待策略。

**📊 数据集**

使用自定义的基于 PNG+JSON 的合成地图数据集（包括 Mountain SAR、Warehouse Fire 等多种场景），并在这些地图上进行多智能体实验。

**📈 对比分析**

与 NumPy、Gymnasium 异步实现等基线对比，单代理配置下实现 33,000,000 步/秒，远超 4,000 SPS 的 NumPy 基线，提升约 3,000 倍；在 MARL 训练中环境步进仅占总训练时间的 3% 以内。

**⚠️ 局限性**

局限性包括目前仍缺乏 GPU 原生的“躲藏者”策略、对极大地图规模的验证不足，以及在多 GPU/多 NUMA 环境下的进一步调优空间。

---

## 210. Beyond the Mean: Within-Model Reliable Change Detection for LLM Evaluation

**arXiv ID:** 2604.27405 | [PDF](https://arxiv.org/pdf/2604.27405v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将临床心理学中的可靠变化指数（Reliable Change Index, RCI）改造用于大型语言模型（LLM）版本比较，针对 2000 个 MMLU‑Pro 项目进行 10 次重复测量，并对比 Llama 3 与 3.1 以及 Qwen 2.5 与 3 两对模型版本，揭示版本升级时的项目级正向与负向变化。

**💡 创新点**

创新点在于：①首次将 RCI 概念引入 LLM 评测，用可靠性检验判断单个项目的显著变化；②通过项目级统计发现“项目 churn”现象，即多数项目在升级后会出现相互抵消的提升与下降，导致整体准确率提升仅是两大对立变化的净残差；③在多领域（物理、法律、心理学、经济学）上揭示不同模型家族具有独特的领域级反向效应。

**🔧 技术方法**

技术上使用 10 次采样的 Bernoulli 试验估计项目准确率，计算项目可靠性（split‑half Spearman‑Brown 校正）和标准误差差异（SEM），然后计算 RCI；对比单次 greedy（T = 0）评估，评估其误报与漏报；采用 1,000 次块级置换检验经验零分布；通过卡方检验评估领域级差异。

**📊 数据集**

数据集为 MMLU‑Pro 的 2,000 条多选题（每条 10 个选项），包含 500 条来自四个领域（物理、法律、心理学、经济学）的样本；每个模型在温度 0.7 下进行 10 次独立采样，总计 80,000 条推理结果。

**📈 对比分析**

与传统单次 greedy 评估相比，RCI 识别的可靠变化率为 44.7%（Llama）和 85.9%（Qwen），且能区分正向和负向变化；单次评估误报率为 25%、漏报率为 42%。在项目级别，可靠改进/退化的比例分别为 33.7%/28.4%（Llama）和 46.9%/39.0%（Qwen）。整体准确率提升仅为 1.6%（Llama）和 2.8%（Qwen），但通过 RCI 可看出其背后是大规模双向项目 churn。

**⚠️ 局限性**

局限性包括：①高比例项目被剔除为 floor/ceiling（> 50%），仅对中等难度项目有分析；②实验仅在 7–8B 参数模型上进行，未检验更大规模模型的泛化；③采用 Q5_K_M quantization，可能影响采样稳定性；④仅在温度 0.7 下评估，温度变化可能改变可靠性与 RCI 结果；⑤使用单一全局 SEM 可能低估难度分层的差异；⑥RCI 对二项分布的假设不够严谨，若采用贝塔‑二项或层级贝叶斯模型可能更精准。

---

## 211. Generalizing the Geometry of Model Merging Through Frechet Averages

**arXiv ID:** 2604.27155 | [PDF](https://arxiv.org/pdf/2604.27155v1)

**作者:** Marvin F. da Silva `[一作]` (Dalhousie University), Sageev Oore `[通讯]` (Dalhousie University)

**通讯引用:** 884 | [OpenAlex ID](https://openalex.org/A5082789942)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将模型合并视为在Riemann几何商流形上求Fréchet均值的框架（GeoMerge），能够在保持对网络对称性的同时进行权重无关合并。

**💡 创新点**

创新点在于：①将对称性映射为商流形并在其上定义度量；②通过轨道对齐与水平梯度下降实现对齐无关的平均；③揭示 Fisher 合并是信息几何Fréchet目标的近似，并在 LoRA 适配器上构造了专门的几何合并方法。

**🔧 技术方法**

使用的技术包括 Riemann 流形几何、商流形对齐、Fréchet 均值优化、几何梯度下降、低秩适配器的极化分解、Stiefel 与 SPD 流形度量、以及 LoRA 的 O(r) 对称性。

**📊 数据集**

在 Llama‑3 8B 基础模型上，使用了 6 个 NLI 数据集（SNLI、MNLI、SICK、QNLI、RTE、SciTail）训练的 LoRA 专家作为实验对象。

**📈 对比分析**

与 KnOTS 进行对比，GeoMerge 在每个任务的归一化准确率均高于 KnOTS，同时保持原始 LoRA 的低秩 r，而 KnOTS 的秩提升到 T·r；平均归一化准确率提升约 1–2%。

**⚠️ 局限性**

局限性包括：只能在固定秩 r 的空间内操作；未将 TIES/DARE‑TIES 等对齐/冲突抑制技术整合进框架；对秩扩展的支持有限，需进一步研究。

---

## 212. T2S-Metrics: Unified Library for Evaluating SPARQL Queries Generated From Natural Language

**arXiv ID:** 2604.26971 | [PDF](https://arxiv.org/pdf/2604.26971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 213. Truthful-in-Expectation Mechanisms for MMS Approximation

**arXiv ID:** 2604.27211 | [PDF](https://arxiv.org/pdf/2604.27211v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Noam Manaker Morag `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一系列基于随机化、可解释的机制，解决了在战略性、可加价值的个体可分配物品公平分配问题；该机制在保持比例前期公平（ex‑ante）且可验证的前提下，提供了对最大最小份额（MMS）和截断比例份额（TPS）的近似保障。

**💡 创新点**

创新点主要有：
1) 引入“循环单配额”（Cyclic‑Unit‑Quota）分数分配作为中间步骤，既保留了序数信息又能在分数层面实现公平。
2) 通过“可信实现”（faithful implementation）与矩阵补全技术，将分数分配转化为具有良好 ex‑post MMS 保障的整数分配。
3) 提出了“缺陷”概念（α‑deficiency），仅用极少的额外数值信息即可显著提升对 MMS 的近似比例（从 1/ln n 提升到 Ω(1/ln ln n)），并在两人情形下达到 2/3 的最优保证。

**🔧 技术方法**

使用的技术包括：
- 通用真诚机制与分数分配的等价性（TIE ↔ 真诚分数分配）。
- 循环序列的序数选择与单配额串行机制。
- 通过部分分数分配的构造与矩阵补全实现完整分数分配。
- 可解释的忠实实现（faithful implementation）保证在整数分配中仅损失一件物品的价值。
- 组合式权重阈值（w = Θ(log log n)）与随机顺序相结合以实现近似真诚（(1‑ε)-TIE）。

**📊 数据集**

该工作为理论性算法设计，未使用具体实验数据集；所有结果均在可加阈值假设下通过数学证明得到。

**📈 对比分析**

与以往的 1/n MMS 结果及 7/9 存在性的上界进行比较：
- 对于一般 n，提出的 1/H_n+2 近似几乎达到序数机制的最优下界 1/H_n；
- 通过引入轻微真诚放松，实现 Ω(1/ln ln n) 的更高近似；
- 对两人情形，证明 2/3 近似为最优。
实验/数值验证未涉及，性能评价基于理论复杂度（多项式时间）与近似比率。

**⚠️ 局限性**

限制与待改进：
- 对一般 n，近似比例仍随 log n 下降，尚未达到常数因子。
- 需要可加价值假设；对非可加或不完全信息情形未覆盖。
- 对两人情形的最优结果依赖于完整的序数和极少数数值信息，其他人数时可能需要更多信息。
- 机制在实现上仍需处理大规模物品时的通信与存储，虽然理论支持多项式，但实际部署可能受限。

---

## 214. Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling

**arXiv ID:** 2604.27039 | [PDF](https://arxiv.org/pdf/2604.27039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 215. On Coded Caching Systems with Decentralized Linear Coding Placement

**arXiv ID:** 2604.27073 | [PDF](https://arxiv.org/pdf/2604.27073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 216. PINN-Cast: Exploring the Role of Continuous-Depth NODE in Transformers and Physics Informed Loss as Soft Physical Constraints in Short-term Weather Forecasting

**arXiv ID:** 2604.27313 | [PDF](https://arxiv.org/pdf/2604.27313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 217. Graphify: Automated Synthesis of Type-Safe Graph Backends via $O(S)$ GraphQL-to-Gremlin Transpilation

**arXiv ID:** 2604.27223 | [PDF](https://arxiv.org/pdf/2604.27223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 218. Finite-Horizon First-Order Rank Profiles of Regular Languages

**arXiv ID:** 2604.27024 | [PDF](https://arxiv.org/pdf/2604.27024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 219. Think it, Run it: Autonomous ML pipeline generation via self-healing multi-agent AI

**arXiv ID:** 2604.27096 | [PDF](https://arxiv.org/pdf/2604.27096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 220. Predicting Covariate-Driven Spatial Deformation for Nonstationary Gaussian Processes

**arXiv ID:** 2604.27280 | [PDF](https://arxiv.org/pdf/2604.27280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 221. BrainDINO: A Brain MRI Foundation Model for Generalizable Clinical Representation Learning

**arXiv ID:** 2604.27277 | [PDF](https://arxiv.org/pdf/2604.27277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 222. Compressing ACAS-Xu Lookup Tables with Binary Decision Diagrams

**arXiv ID:** 2604.27008 | [PDF](https://arxiv.org/pdf/2604.27008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 223. The Likelihood Ratio Wall: Structural Limits on Accurate Risk Assessment for Rare Violence

**arXiv ID:** 2604.27282 | [PDF](https://arxiv.org/pdf/2604.27282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 224. BatteryPass-12K: The First Dataset for the Novel Digital Battery Passport Conformance Task

**arXiv ID:** 2604.26986 | [PDF](https://arxiv.org/pdf/2604.26986v1)

**作者:** Tosin Adewumi `[一作]` (Luleå University of Technology), Marcus Liwicki `[通讯]` (Luleå University of Technology)

**通讯引用:** 9551 | [OpenAlex ID](https://openalex.org/A5073619925)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出电池数字护照（DBP）合规性分类任务，并构建了首个基于真实试点样本合成的公共基准数据集（bp），随后在22款不同类型的语言模型上进行零样本与少样本推理实验。

**💡 创新点**

创新点在于：①首次将DBP合规性作为二分类任务；②通过LLM生成与LLM‑as‑a‑Judge校验的方式构造高质量合成数据；③系统性比较了思考型模型、常规模型、Moe、LLM等多种模型的表现。

**🔧 技术方法**

核心技术包括：LLM自动检索与生成、LLM‑as‑a‑Judge评估、零样本推理、少样本提示学习、对抗性注入攻击以及模型参数规模与性能的统计分析。

**📊 数据集**

使用的数据集为bp，包含12,000个合成DBP样本（8,000训练、1,200验证、1,200测试），每个样本均含10个关键字段，数据均衡为合规与不合规各半。

**📈 对比分析**

在零样本评估中，最佳模型GPT‑5.4 Thinking取得验证集F1 0.98（±0.03），测试集F1 0.71（±0.22）；少样本提升至F1 0.99；规模分析显示参数增大并不一定带来更优表现，且对抗注入能显著降低性能。

**⚠️ 局限性**

局限性包括：数据仅覆盖10个字段且来源单一试点样本，未涵盖完整欧盟法规；仅考虑内部不一致作为不合规判定；数据仅为英文；合成数据与真实DBP在分布与多样性上可能存在偏差。

---

## 225. People-Centred Medical Image Analysis

**arXiv ID:** 2604.26991 | [PDF](https://arxiv.org/pdf/2604.26991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 226. Addressing the Reality Gap: A Three-Tension Framework for Agentic AI Adoption

**arXiv ID:** 2604.27245 | [PDF](https://arxiv.org/pdf/2604.27245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 227. Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods

**arXiv ID:** 2604.27033 | [PDF](https://arxiv.org/pdf/2604.27033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 228. Secret Stealing Attacks on Local LLM Fine-Tuning through Supply-Chain Model Code Backdoors

**arXiv ID:** 2604.27426 | [PDF](https://arxiv.org/pdf/2604.27426v1)

**作者:** Zi Li `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**通讯引用:** 470515 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在开源模型代码中嵌入恶意逻辑，实现本地微调过程中的主动执行劫持，窃取高熵敏感秘密

**💡 创新点**

提出了基于模型代码的供应链攻击框架，包含在线张量规则匹配、确定性键值绑定、校验码验证、梯度失配与后层更新等创新机制

**🔧 技术方法**

利用张量规则匹配、确定性键值绑定、梯度注入、stop-gradient、后层目标更新（RLTU）以及校验码验证等技术

**📊 数据集**

在Llama‑3.2‑3B‑Instruct上分别针对Magicoder代码生成、HealthcareMagic医疗问答和AESLC摘要任务（以及Qwen‑2.5‑7B‑Instruct等模型）进行实验

**📈 对比分析**

与Clean FT、语义前缀诱导、BadEdit等权重攻击方法比较，取得>98%严格ASR，且主任务性能下降≤3%，显著优于传统被动权重攻击

**⚠️ 局限性**

仅在LoRA、标准AdamW等设置下验证，未覆盖QLoRA、长期训练、不同优化器及更广泛模型，且需依赖模型代码执行权限，部分防御（如DP‑SGD）仍有效

---

## 229. Lightweight Distillation of SAM 3 and DINOv3 for Edge-Deployable Individual-Level Livestock Monitoring and Longitudinal Visual Analytics

**arXiv ID:** 2604.27128 | [PDF](https://arxiv.org/pdf/2604.27128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. An Empirical Security Evaluation of LLM-Generated Cryptographic Rust Code

**arXiv ID:** 2604.27001 | [PDF](https://arxiv.org/pdf/2604.27001v1)

**作者:** Mohamed Elsayed `[一作]` (Texas A&M University--San Antonio), Jeong Yang `[通讯]` (Texas A&M University--San Antonio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对240个由Gemini 2.5 Pro、GPT‑4o和DeepSeek Coder生成的Rust AES‑256‑GCM和ChaCha20‑Poly1305代码进行实证评估，检查编译成功率与安全漏洞；

**💡 创新点**

首次系统性验证LLM生成加密代码的安全性，揭示链式思考提示对加密实现的负面影响，并提出针对Rust的规则式加密专用分析器，证明通用工具无法捕获关键漏洞；

**🔧 技术方法**

使用Clippy进行编译和错误分类，CodeQL进行通用静态分析，以及自研基于正则和轻量结构分析的加密专用静态分析器；

**📊 数据集**

实验数据集由三种LLM、两种算法、四种提示策略共240个代码样本构成，按10个样本/配置生成；

**📈 对比分析**

比较指标包括编译成功率、漏洞检测准确率；结果显示编译成功率仅23.3%，链式思考提示仅6.7%，AES‑256‑GCM成功率约3.4倍ChaCha20‑Poly1305；CodeQL检测不到任何真阳性，误报率100%；自研分析器检测到57%中间级漏洞，2%关键漏洞，无误报；

**⚠️ 局限性**

局限在仅评估AEAD算法、Rust语言、单文件正则分析，样本量有限，未涵盖密钥交换、签名、哈希等加密原语，且对跨模块流和间接初始化的检测仍有不足。

---

## 231. VitaLLM: A Versatile, Ultra-Compact Ternary LLM Accelerator with Dependency-Aware Scheduling

**arXiv ID:** 2604.27396 | [PDF](https://arxiv.org/pdf/2604.27396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 232. When Your LLM Reaches End-of-Life: A Framework for Confident Model Migration in Production Systems

**arXiv ID:** 2604.27082 | [PDF](https://arxiv.org/pdf/2604.27082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 233. A Short Note on Batch-efficient Divide-and-Conquer Algorithm for EigenDecomposition

**arXiv ID:** 2604.27325 | [PDF](https://arxiv.org/pdf/2604.27325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 234. DeepTutor: Towards Agentic Personalized Tutoring

**arXiv ID:** 2604.26962 | [PDF](https://arxiv.org/pdf/2604.26962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 235. Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings

**arXiv ID:** 2604.27398 | [PDF](https://arxiv.org/pdf/2604.27398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 236. Modeling of Wastewater Treatment Processes with HydroSludge

**arXiv ID:** 2604.27432 | [PDF](https://arxiv.org/pdf/2604.27432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 237. Towards the Democratization and Standardization of Dynamic Resources with MPI Spawning

**arXiv ID:** 2604.27430 | [PDF](https://arxiv.org/pdf/2604.27430v1)

**作者:** Sergio Iserte `[一作]` (Barcelona Supercomputing Center), Antonio J. Peña `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5000573036)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个统一的动态资源管理 API，并将 Proteo 重构引擎集成到 DMRlib，实现了基于 MPI Spawn 的可扩展资源动态重构，随后在 MPDATA 计算流体仿真中进行验证。

**💡 创新点**

创新点在于构建了标准 MPI 兼容、模块化、易用的动态资源管理框架，支持多种重构策略与 Slurm 透明协作，并显著降低了进程重启成本。

**🔧 技术方法**

技术手段包括 MPI（MPICH 4.2.1）、Slurm 资源管理器、DMR API、DMRlib 及 Proteo 重构引擎，主要使用 C/C++ 编写。

**📊 数据集**

实验使用 MPDATA 的 1024×128×32 网格、20 个时间步的数据集，并提交 1,000 个作业以模拟高负载场景。

**📈 对比分析**

通过比较静态资源、Baseline（全部重启）与 Merge（仅增删进程）三种策略，结果显示动态资源可将作业完成时间提升约 1.5 倍，平均利用率提升至约 93%。

**⚠️ 局限性**

局限性包括仅评估单次重构场景，缺乏异步重构、多样化作业类型和更大规模集群的实验，仅支持同步点对点重分布。

---

## 238. InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation?

**arXiv ID:** 2604.27419 | [PDF](https://arxiv.org/pdf/2604.27419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 239. Understanding Adversarial Transferability in Vision-Language Models for Autonomous Driving: A Cross-Architecture Analysis

**arXiv ID:** 2604.27414 | [PDF](https://arxiv.org/pdf/2604.27414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. Leading Across the Spectrum of Human-AI Relationships: A Conceptual Framework for Increasingly Heterogeneous Teams

**arXiv ID:** 2604.27392 | [PDF](https://arxiv.org/pdf/2604.27392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 241. An Experimental Modular Instrument With a Haptic Feedback Framework for Robotic Surgery Training

**arXiv ID:** 2604.27385 | [PDF](https://arxiv.org/pdf/2604.27385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 242. DOT-Sim: Differentiable Optical Tactile Simulation with Precise Real-to-Sim Physical Calibration

**arXiv ID:** 2604.27367 | [PDF](https://arxiv.org/pdf/2604.27367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 243. CoAX: Cognitive-Oriented Attribution eXplanation User Model of Human Understanding of AI Explanations

**arXiv ID:** 2604.27354 | [PDF](https://arxiv.org/pdf/2604.27354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 244. Context as Prior: Bayesian-Inspired Intent Inference for Non-Speaking Agents with a Household Cat Testbed

**arXiv ID:** 2604.27445 | [PDF](https://arxiv.org/pdf/2604.27445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. AppTek Call-Center Dialogues: A Multi-Accent Long-Form Benchmark for English ASR

**arXiv ID:** 2604.27543 | [PDF](https://arxiv.org/pdf/2604.27543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 246. Self-Supervised Learning of Plant Image Representations

**arXiv ID:** 2604.27538 | [PDF](https://arxiv.org/pdf/2604.27538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. In-Context Examples Suppress Scientific Knowledge Recall in LLMs

**arXiv ID:** 2604.27540 | [PDF](https://arxiv.org/pdf/2604.27540v1)

**作者:** Chaemin Jang `[一作]` (Korea Advanced Institute of Science and Technology), Jihee Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5324 | [OpenAlex ID](https://openalex.org/A5100328628)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多学科科学推理任务中评估增量式示例对大型语言模型知识推理的影响

**💡 创新点**

揭示示例的出现会抑制模型对预训练科学知识的调用，导致“知识位移”，并阐明此现象的机制与对准确率的三种不同影响

**🔧 技术方法**

对照实验、策略竞争框架、链路思维（CoT）分类器、词汇抑制实验、内部表征分析

**📊 数据集**

自构建的60个隐藏结构恢复任务，涵盖生物学、化学、经济学、物理学与地球科学（12个任务/域，共6000次试验）

**📈 对比分析**

与零样本、十样本、词汇替换、结构提示等多种对照；发现十样本示例在经济学、化学等领域降低准确率，在地球科学提高准确率，在物理学与生物学几乎无变化；整体平均下降约1.4个百分点

**⚠️ 局限性**

依赖于自动化CoT分类器的行为分析、仅在预训练已知公式的任务上验证、未直接探测内部神经机制，且在更复杂或非典型科学情境下的泛化性有限

---

## 248. Examining discontinuance of AI-mediated informal digital learning of English (AI-IDLE) among university students: Evidence from SEM and fsQCA

**arXiv ID:** 2604.27506 | [PDF](https://arxiv.org/pdf/2604.27506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 249. REVIVE 3D: Refinement via Encoded Voluminous Inflated prior for Volume Enhancement

**arXiv ID:** 2604.27504 | [PDF](https://arxiv.org/pdf/2604.27504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Secure Cross-Silo Synthetic Genomic Data Generation

**arXiv ID:** 2604.27456 | [PDF](https://arxiv.org/pdf/2604.27456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 251. LA-Pose: Latent Action Pretraining Meets Pose Estimation

**arXiv ID:** 2604.27448 | [PDF](https://arxiv.org/pdf/2604.27448v1)

**作者:** Zhengqing Wang `[一作]` (Wayve), Yasutaka Furukawa `[通讯]` (Wayve)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一个两阶段框架：先用海量无标注驾驶视频训练逆向-正向动态模型提取潜在动作，再用少量带3D标注的数据训练轻量化姿态估计头，实现快速、准确的相机姿态预测。

**💡 创新点**

创新点在于：①将逆向动态学习得到的潜在动作重新用于相机姿态估计，而非传统的动作条件或生成任务；②通过海量无标签视频进行预训练，显著降低对3D标注的依赖；③压缩潜在动作维度提升运动感知和尺度一致性。

**🔧 技术方法**

使用的技术包括：逆向-正向动态模型（Genie结构）、ST-Transformer、VQ‑VAE代码本、三层MLP压缩/解压、轻量化姿态估计头、Cosine学习率调度、交叉熵和L1损失。

**📊 数据集**

预训练数据为约1020万条无人标注驾驶视频；后期训练与评估使用 Waymo、nuScenes、Argoverse；最终在 Waymo Open 与 PandaSet（零样本）进行性能评测。

**📈 对比分析**

与 Rig3R、VGGT、MapAnything 等基线在 Waymo 与 PandaSet 上对比；在 Waymo 上实现 AUC@5 91.4%、ATE 0.012，在 PandaSet 上 AUC@5 86.3%、ATE 0.011；在所有基线中保持最高 AUC、低方差，并使用标注量减少数倍。

**⚠️ 局限性**

限制：在罕见场景（如倒车、逆向运动）中精度下降；零样本对非常稀有情形的鲁棒性不足；需要扩大预训练数据覆盖更广泛的运动模式。

---

## 252. HATS: An Open data set Integrating Human Perception Applied to the Evaluation of Automatic Speech Recognition Metrics

**arXiv ID:** 2604.27542 | [PDF](https://arxiv.org/pdf/2604.27542v1)

**作者:** Thibault Bañeras Roux `[一作]` (Nantes University), Richard Dufour `[通讯]` (Nantes University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建HATS数据集，对十个不同架构的法语ASR系统生成的错误转录进行人类侧对侧选择实验，并用多种词级与语义级指标评估这些指标与人类偏好的一致性。

**💡 创新点**

创新点在于：①首次公开一个侧对侧人类评价的法语ASR错误转录数据集；②系统地将传统WER/CER与多种基于词嵌入和句子嵌入的语义指标（EmbER、SemDist、BERTScore、PhonER）与人类偏好进行对比，揭示了语义指标对人类认知的更好映射。

**🔧 技术方法**

技术包括：端到端Speechbrain和Kaldi的ASR训练；fastText、CamemBERT、FlauBERT、Sentence‑BERT等预训练词/句子嵌入；Levenshtein距离、cosine相似度计算；Fleiss' Kappa与人类一致率统计。

**📊 数据集**

使用的主要数据集是：REPERE测试集（≈10小时法语广播音频），以及自建的1,000条参考句子和两种错误转录的HATS数据集，总共7,150条人类标注。

**📈 对比分析**

比较方法是计算每个指标在不同一致率阈值（100%、70%、无过滤）下与人类选择的匹配比例；结果显示语义指标SemDist（句子CamemBERT‑large）与人类偏好最高达90%，其次是BERTScore CamemBERT‑base/large，WER/CER表现低于随机，PhonER在文字实验中亦表现良好。

**⚠️ 局限性**

局限性包括：数据集仅选取了符合严格条件的错误，对错误类型的代表性不足；仅针对法语，可能不具备跨语言通用性；实验使用文字参考，未检验音频参考对结果的影响。

---

## 253. Qualitative Evaluation of Language Model Rescoring in Automatic Speech Recognition

**arXiv ID:** 2604.27533 | [PDF](https://arxiv.org/pdf/2604.27533v1)

**作者:** Thibault Bañeras-Roux `[一作]`, Richard Dufour `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了多种基于句法和语义的自动评估指标（POSER、EmbER、BERTScore、SemDist 等），并用它们对语音识别系统（基线与后验重排序）的性能进行多维度分析。

**💡 创新点**

创新点在于：①将词性错误率和词干错误率引入 ASR 评估；②设计 EmbER 以语义相似度加权的 WER；③将句子嵌入与 BERTScore 等新型 NLP 评估方法应用于 ASR；④通过多指标交叉分析揭示重排序对不同语言层面的影响。

**🔧 技术方法**

使用 Kaldi 训练的 TDNNF 声学模型，结合三元/四元词模型、RNNLM 进行后验重排序；POS 标注使用 POET；词干提取使用 Spacy；EmbER 采用 FastText 词嵌入；SemDist 与 BERTScore 采用多语言 Sentence‑BERT 与 multilingual‑BERT；统计相关性与性能提升采用 Pearson 相关系数和相对改进率。

**📊 数据集**

训练数据包括 ESTER‑1/2、EPAC、ETAPE、REPERE 与内部 LIA 共计约 940 小时；评估数据选自 REPERE 测试集（约 10 小时）。

**📈 对比分析**

通过将基线系统与重排序系统在各指标上的得分进行比较，发现 WER 降低 14.3%，CER、POSER 等词性指标也均有提升，但 SemDist、BERTScore 的相对提升最低，说明语义层面的改进有限；相关性分析显示 EmbER 与 WER 关联最高，SemDist 与其他指标关联最低。

**⚠️ 局限性**

局限性包括：①新指标与传统 WER 的相关性不一，尤其是句子级别的 SemDist 与 BERTScore 与 WER 关联弱；②重排序在自发性语音（含口吃、删音等）上可能导致性能下降；③本文仅关注语言学层面，未考虑声学噪声、说话人差异等因素；④未与人工评估或下游任务性能进行外部验证。

---

## 254. I'm Fine, But My Voice Isn't: Cross-Modal Affective Dissonance Detection for Reflective Journaling

**arXiv ID:** 2604.27517 | [PDF](https://arxiv.org/pdf/2604.27517v1)

**作者:** Sumin Lee `[一作]` (Seoul National University), Sumin Lee `[通讯]` (Seoul National University)

**通讯引用:** 395 | [OpenAlex ID](https://openalex.org/A5100726747)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了跨模态情感失调检测（CADD）任务，用以识别数字日记中语音与文本情感不一致的情况。

**💡 创新点**

创新点包括：①正式化了掩饰、应对和一致三类失调；②构建了 1800 样本 TTS 控制数据集 CADD‑Journal；③提出了 DACM 双编码器模型，并通过非对称跨模态注意力与不一致交互模块实现显著性能提升。

**🔧 技术方法**

技术上使用冻结的 XLM‑RoBERTa 与 WavLM 编码器、加权层池化、异向跨模态注意力、对抗/边缘损失以及 DIM 产生不一致评分 S。

**📊 数据集**

使用的主要数据集是 CADD‑Journal（TTS 合成情感文本），并在 CMU‑MOSEI、IEMOCAP 及 CH‑SIMS 上进行零射击评估。

**📈 对比分析**

与基线（文本/语音单模态、传统对齐）比较，DACM 在 CADD‑Journal 上 macro‑F1 达到 0.711，显著优于 Audio‑Only（0.448）和 Text‑Only（0.167）；但在自然语料上性能急剧下降至接近随机，表明存在显著域间差距。

**⚠️ 局限性**

局限性包括：① TTS 合成失调不等同真实情绪调节；② 交叉语料标注采用银级估计，存在噪声；③ 仅覆盖三种英语合成声，缺乏语言和人群多样性；④ 评估缺乏用户实验验证 ReflectJournal 的实际效用。

---

## 255. Towards All-Day Perception for Off-Road Driving: A Large-Scale Multispectral Dataset and Comprehensive Benchmark

**arXiv ID:** 2604.27499 | [PDF](https://arxiv.org/pdf/2604.27499v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 256. Uni-HOI:A Unified framework for Learning the Joint distribution of Text and Human-Object Interaction

**arXiv ID:** 2604.27491 | [PDF](https://arxiv.org/pdf/2604.27491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 257. SST-Guard: Detecting and Characterizing Server-Side Google Analytics in the Wild

**arXiv ID:** 2604.27497 | [PDF](https://arxiv.org/pdf/2604.27497v1)

**作者:** Muhammad Jazlan `[一作]` (University of California, Davis), Yash Vekaria `[通讯]` (University of California, Davis)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5079649590)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个浏览器扩展，能够检测并识别服务器端 Google Analytics（sGA）的实现，弥补传统客户端跟踪检测的盲区。

**💡 创新点**

创新点在于提出“价值模板”技术，利用正则表达式匹配网络请求、Cookie 和 JavaScript 变量中保持不变的语义特征，从而对端点自定义、路径编码和参数加密等伪装手段具备鲁棒性。

**🔧 技术方法**

主要技术包括：Playwright 自动化抓取 Tranco 网站、Google Tag Assistant 采集真值、基于正则的价值模板、逻辑回归分类器以及对 EasyPrivacy 过滤列表的对比评估。

**📊 数据集**

使用的数据集为 Tranco Top‑10K 与 Top‑150K 公开域名列表，并在 Top‑10K 上通过 Tag Assistant 获得服务器端 GA 的真实标签。

**📈 对比分析**

实验结果显示，在验证集上模型准确率超过 99%，在真值集上 F1 约为 97%，能够覆盖 Top‑150K 中约 4.21% 的网站；与仅使用 EasyPrivacy 的 93% 覆盖率相比，模型大幅提升检测覆盖和准确率。

**⚠️ 局限性**

限制包括：只能检测基于 GTM 的 sGA，无法识别完全服务器端（Measurement Protocol）实现；对非 Google 追踪器不具备通用性；训练标签来源单一，可能对新型伪装策略适应性不足。

---

## 258. Exploring Applications of Transfer-State Large Language Models: Cognitive Profiling and Socratic AI Tutoring

**arXiv ID:** 2604.27454 | [PDF](https://arxiv.org/pdf/2604.27454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 259. Debiasing Reward Models via Causally Motivated Inference-Time Intervention

**arXiv ID:** 2604.27495 | [PDF](https://arxiv.org/pdf/2604.27495v1)

**作者:** Kazutoshi Shinoda `[一作]` (NTT, Inc.), Kyosuke Nishida `[通讯]` (NTT, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理时通过识别并中和奖励模型中与多种格式偏差相关的神经元激活，从而消除这些偏差对奖励评分的影响。

**💡 创新点**

提出将奖励模型视为因果介导过程，利用中介效应（CDE）将相关神经元激活替换为中位值，实现无训练成本且无性能折衷的去偏方法。

**🔧 技术方法**

使用 Spearman 相关性识别偏差特定神经元，并在奖励模型的 Bradley–Terry 结构中执行因果干预（中位值替换），实现神经元级去偏。

**📊 数据集**

在 RewardBench、RM‑Bench、AlpacaEval 2.0、MT‑Bench、TruthfulQA 以及验证用的 RewardBench 500 条样本上进行实验。

**📈 对比分析**

与原始 RM、Length Penalty 以及 Locally Weighted Regression 比较，CIRM 在 RM 和对齐评测中保持或提升准确率、LCWR 与 WR 分数，无明显性能折衷，甚至小型 RM 通过 CIRM 训练的 LLM 能与 70B RM 相媲美。

**⚠️ 局限性**

仅针对五种预定义的格式偏差；方法依赖验证集的相关性计算和超参数调优，可能对新出现或更分散的偏差效果有限，且因果图简化可能遗漏复杂交互。

---

## 260. Entropy of Ukrainian

**arXiv ID:** 2604.27534 | [PDF](https://arxiv.org/pdf/2604.27534v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 261. EdgeFM: Efficient Edge Inference for Vision-Language Models

**arXiv ID:** 2604.27476 | [PDF](https://arxiv.org/pdf/2604.27476v1)

**作者:** Mengling Deng `[一作]` (Go Further. AI), Xiangjing An `[通讯]` (Go Further. AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了EdgeFM，一个轻量级、跨平台的LLM推理框架，专为工业边缘设备提供单请求低延迟推理。

**💡 创新点**

核心创新包括边缘原生设计、模块化层级架构、基于LLM推理操作符的agent辅助内核优化、两阶段（prefill/ decode）执行与KV缓存重用、CUDA图加速及可选的speculative decoding，打破硬件厂商闭源锁定。

**🔧 技术方法**

采用FlashAttention/FlashInfer类注意力核、FlashMLA、SageAttention量化、CUDA Graphs、EAGLE3式speculative decoding、KV压缩、两阶段执行、可配置的Operator实现表等技术。

**📊 数据集**

使用Qwen2.5系列（LLM与VLM）以及SmolVLA-base模型进行评估，数据来自ModelScope和HuggingFace公开模型库。

**📈 对比分析**

通过与NVIDIA官方TensorRT-Edge-LLM基准在x86 A800、Jetson Orin NX以及国内Horizon Journey 6M平台对比，EdgeFM在prefill和decode阶段分别实现约14%–34%延迟下降，整体上可达1.38–1.49倍的吞吐提升，且保持确定性低延迟。

**⚠️ 局限性**

局限性在于目前支持的模型和硬件有限，需针对新模型手动调整Operator表；speculative decoding在准确率与速度之间存在权衡；缺乏多模型并发调度与动态资源管理功能。

---

## 262. From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks

**arXiv ID:** 2604.27453 | [PDF](https://arxiv.org/pdf/2604.27453v1)

**作者:** Qingyu Ren `[一作]` (Fudan University), Xuhong Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1721 | [OpenAlex ID](https://openalex.org/A5060634520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WEval 细粒度评估管线和 WRL 强化学习框架，通过需求 dropout 生成自然黄金排名来训练和评估写作奖励模型，并提升 LLM 的写作性能。

**💡 创新点**

创新点在于利用需求 dropout 构造细粒度奖励对比，结合自然排序相关性评估与 Bradley–Terry 训练，实现对写作需求遵循的精准奖励信号。

**🔧 技术方法**

使用需求 dropout、自然排序相关性评估、Bradley–Terry 损失、GRPO/Group Relative Policy Optimization、DeepSeek‑R1、Qwen2.5 等大模型技术。

**📊 数据集**

采集自 LMSYS‑1M、WildChat、PRISM 等对话数据，构建 WEval 评估集，并在 DeepChat 等模型上生成多种写作需求样本。

**📈 对比分析**

在 WritingBench、Arena‑Write、DeepResearch Bench 等基准上与 SFT、DPO、LLM‑as‑a‑judge、LongWriter‑Zero 等基线对比，WRL 在多模型上提升 4–8% win rate，奖励模型在 Correlation、IL、PL 上均超过 90，表现优异。

**⚠️ 局限性**

局限性包括未在 32B 以上大模型上评估，以及评估集仅覆盖有限的需求类型，缺乏更广泛写作约束的多样性。

---

## 263. Knowledge Affordances for Hybrid Human-AI Information Seeking

**arXiv ID:** 2604.27539 | [PDF](https://arxiv.org/pdf/2604.27539v1)

**作者:** Irene Celino `[一作]` (Cefriel), Irene Celino `[通讯]` (Cefriel)

**通讯引用:** 1081 | [OpenAlex ID](https://openalex.org/A5054444380)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出知识可供性（Knowledge Affordance）概念，用以指导人类与人工智能在异构知识生态中高效寻求信息。

**💡 创新点**

创新点在于将知识源的功能与非功能属性以语义化、情境化的方式显式建模，并通过可供性进行动态激活与计划，促进多主体的互相理解与协作。

**🔧 技术方法**

采用语义网服务（OWL‑S/WSMO）框架、本体与竞争性问题（Competency Questions）技术、知识图谱查询、LLM驱动的自然语言问答以及基于情境的计划生成算法。

**📊 数据集**

本文为理论性研究，并未使用具体数据集；主要通过案例描述与假设验证来说明概念的可行性。

**📈 对比分析**

没有进行实验比较或性能评估；论文侧重于提出框架与理论模型，并未对方法进行量化验证。

**⚠️ 局限性**

局限性包括缺乏实证验证、可供性建模复杂度高、对人类偏好与情境因素的动态捕捉不足，以及在大规模异构系统中的可扩展性与可维护性待进一步研究。

---

## 264. A Longitudinal Analysis of Good First Issue Practices and Newcomer Pull Requests in Popular OSS Projects

**arXiv ID:** 2604.27532 | [PDF](https://arxiv.org/pdf/2604.27532v1)

**作者:** Hirotatsu Hoshikawa `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 7324 | [OpenAlex ID](https://openalex.org/A5011588138)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 37 个热门 GitHub OSS 项目自 2021 年 7 月至 2025 年 6 月的 Good First Issue（GFI）标签使用情况和新人 PR 合并率进行四年纵向分析，揭示 GFI 供应下降与新人成功率变化的趋势。

**💡 创新点**

①首次使用时间序列与 Pettitt 变点检验发现 2024 年初 GFI 标签比例显著下降；②同时评估新人参与率保持稳定而 PR 合并率下降，揭示需求与供应的失衡；③将 GFI 按任务类型（Bug、Feature、Documentation、Other）细分，发现不同类型合并率的变化差异。

**🔧 技术方法**

采用 GitHub GraphQL API 抓取数据；使用 Mann‑Kendall 趋势检验、Pettitt 变点分析；多重比较校正（Holm‑Bonferroni、Benjamini‑Hochberg）；关键词映射将标签映射到任务类型；对 PR 的描述长度、代码行数、评审次数等特征进行统计与比较。

**📊 数据集**

共 406,826 个 issue（其中 3,300 个标记为 GFI）与 1,117 个新人 GFI PR，来源于 37 个主流 OSS 仓库（主要语言包括 TypeScript、C++、JavaScript、Python、Rust 等），时间跨度 2021/7–2025/6。

**📈 对比分析**

通过 Mann‑Kendall 检验比较不同年份、任务类型的合并率与描述长度等指标。结果显示：GFI 比例从 0.91% 降至 0.63%（显著下降），新人参与率保持约 27%（无显著趋势），合并率从 61.9% 降至 42.2%（显著下降），描述长度上升约 47%。

**⚠️ 局限性**

局限性包括：只关注热门 GitHub 项目，忽略规模较小或其他平台项目；GFI 标签识别基于固定关键词，可能漏掉自定义标签；新人定义为仓库首次 PR，未区分真正 OSS 新手；AI 工具生成的 PR 可能影响合并率；部分 PR 仍未关闭，可能导致合并率估计偏低。

---

## 265. Adjoint Inversion Reveals Holographic Superposition and Destructive Interference in CNN Classifiers

**arXiv ID:** 2604.27529 | [PDF](https://arxiv.org/pdf/2604.27529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. Leveraging Verifier-Based Reinforcement Learning in Image Editing

**arXiv ID:** 2604.27505 | [PDF](https://arxiv.org/pdf/2604.27505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. From Elastic to Viscoelastic: An EEMD-Enhanced Pulse Transit Time Model for Robust Blood Pressure Estimation

**arXiv ID:** 2604.27500 | [PDF](https://arxiv.org/pdf/2604.27500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 268. CuLifter: Lifting GPU Binaries to Typed IR

**arXiv ID:** 2604.27486 | [PDF](https://arxiv.org/pdf/2604.27486v1)

**作者:** Jisheng Zhao `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 41786 | [OpenAlex ID](https://openalex.org/A5008008203)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文提出了CuLifter框架，将NVIDIA SASS二进制代码提升为可分析的LLVM IR，从而实现GPU二进制的反汇编、重构、分析和跨平台执行。

**💡 创新点**

创新点主要有：①将GPU二进制提升视为结构恢复问题，并证明类型恢复是关键；②通过约束传播与冲突检测解决统一寄存器文件中的类型缺失；③将ψ‑函数技术推广到SIMT预编译代码，处理预编译谓词、双谓词分支和收敛屏障；④构建完整的控制流、语义恢复和类型恢复管道，最终实现高成功率。

**🔧 技术方法**

使用的技术包括：类型约束传播与位集冲突检测、ψ‑函数/Select插入、基于谓词的基本块划分与CFG重建、隐式寄存器展开、多指令模式识别与聚合、设备函数恢复、LLVM IR代码生成以及对不同GPU架构（SM75–SM120）的适配。

**📊 数据集**

实验数据集覆盖八个基准套件，共24,437个GPU函数、919个cubin档，分别包含开源应用、NVIDIA vendor库（cuBLAS、cuDNN、CUDA SDK）和优化ML运行时（FlashAttention）。

**📈 对比分析**

评估方式包括完整提升成功率、类型准确率、x86后端执行通过率及对照实验（禁用各恢复步骤的消融测试）。提升成功率达99.98%；类型种类准确率100%，宽度准确率77–87%；在HeCBench上，完整流程通过率73.8%，仅禁用类型恢复导致0%通过率，表明类型恢复是决定性步骤。

**⚠️ 局限性**

局限性包括：①解析器对新指令（如SM90 QGMMA）的支持不足，导致六个函数失败；②类型恢复无法推断从未被种子指令使用的值，默认Int32；③分析仅在单函数内完成，跨函数类型信息未传播；④CPU后端因硬件特定运算（MUFU、纹理等）而缺失完整语义，导致约26%后端失败；④需要持续维护解析器以适配新的GPU架构。

---

## 269. Toward Scalable SDN for LEO Mega-Constellations: A Graph Learning Approach

**arXiv ID:** 2604.27478 | [PDF](https://arxiv.org/pdf/2604.27478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 270. Gender Bias in YouTube Exposure: Allocative and Structural Inequalities in Political Information Environments

**arXiv ID:** 2604.27479 | [PDF](https://arxiv.org/pdf/2604.27479v1)

**作者:** Jipeng Tan `[一作]` (Beijing Normal University), Yong Min `[通讯]` (Beijing Normal University)

**通讯引用:** 3207 | [OpenAlex ID](https://openalex.org/A5005319784)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在YouTube上构建性别编码的社交机器人进行受控实验，追踪其推荐轨迹，分析推荐系统的性别偏差。

**💡 创新点**

首次同时揭示分配性和结构性性别偏差，并用简化协同过滤模型解释机制。

**🔧 技术方法**

采用社交机器人自动化脚本、LLM辅助内容编码、社会网络分析、协同过滤仿真模型等技术。

**📊 数据集**

使用约160个性别编码的虚拟账号在YouTube上的约509,336条推荐曝光记录，其中78,728条为政治内容。

**📈 对比分析**

通过对比例、熵、多维度（议题、意识形态、实体）和网络指标（密度、模块度、聚类系数）的统计检验及滞后回归，发现女性编码账号政治曝光比例更高、议题更分散；男性编码账号曝光更集中。

**⚠️ 局限性**

实验受限于平台封禁风险、性别编码仅基于兴趣标签、未考察用户真实行为对政治态度的影响。

---

## 271. PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations

**arXiv ID:** 2604.27472 | [PDF](https://arxiv.org/pdf/2604.27472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 272. RAY-TOLD: Ray-Based Latent Dynamics for Dense Dynamic Obstacle Avoidance with TDMPC

**arXiv ID:** 2604.27450 | [PDF](https://arxiv.org/pdf/2604.27450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 273. APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation

**arXiv ID:** 2604.27550 | [PDF](https://arxiv.org/pdf/2604.27550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 274. Diagnosing Capability Gaps in Fine-Tuning Data

**arXiv ID:** 2604.27547 | [PDF](https://arxiv.org/pdf/2604.27547v1)

**作者:** Saeid Asgari Taghanaki `[一作]` (Microsoft), Emre Kiciman `[通讯]` (Microsoft)

**通讯引用:** 4778 | [OpenAlex ID](https://openalex.org/A5079458476)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一个框架，在微调前通过交互式目标拆分和LLM评估检测数据集的能力缺口。

**💡 创新点**

创新点在于将目标拆解为原子子目标、使用LLM基准评分与解释进行覆盖评估、自动化缺口分析并给出补救方案（含目标条件合成数据）。

**🔧 技术方法**

技术包括：交互式澄清循环、基于 Anchored-Rubric 的 LLM 评估器、解释聚合的缺口分析、生成-判别式合成数据、强化学习微调（GRPO）与 LLM 评估奖励。

**📊 数据集**

使用的公开数据集包括 PubMedQA、BillSum、CodeAlpaca，以及 GovReport 的财务报告摘要数据；同时进行人为破坏实验。

**📈 对比分析**

通过控制性破坏实验，目标子目标的衰减为 25.6% 对比非目标 2.1%（Cohen's d=1.24）；在财务摘要 RFT 任务中，过滤后的数据将奖励从 3.77 提升至 4.12，合成+过滤组合最高 4.20，且在所有子目标上均无倒退。

**⚠️ 局限性**

局限包括：评估器单一、对评估器专业度敏感；仅在单一模型/算法上验证；评估成本较高；拆分质量依赖交互；控制破坏不完全模拟自然缺口；对高度专业化领域的适用性待验证。

---

## 275. Empire Amplifier: Uncovering and Contesting the Prioritization of Colonial Content on Platforms Through Community-Informed Algorithmic Auditing

**arXiv ID:** 2604.27498 | [PDF](https://arxiv.org/pdf/2604.27498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 276. Temporal and Content Coupling Analysis of Social Media User Behavior

**arXiv ID:** 2604.27530 | [PDF](https://arxiv.org/pdf/2604.27530v1)

**作者:** Jipeng Tan `[一作]` (Beijing Normal University), Yong Min `[通讯]` (Beijing Normal University)

**通讯引用:** 3207 | [OpenAlex ID](https://openalex.org/A5005319784)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一个多尺度时序-内容耦合框架，用于系统分析社交媒体新闻消费行为。

**💡 创新点**

将宏观、介观、微观时间尺度与历史兴趣与内容多样性耦合，首次将时间与内容双重动态统一建模。

**🔧 技术方法**

采用傅里叶分析、幂律/指数分布拟合、BERT语义相似度、信息熵、Wasserstein距离、聚类(K-means、GMM、层次聚类、Birch)及基于代理的建模技术。

**📊 数据集**

使用公开的微软新闻MIND数据集和挪威Adresse新闻日志数据集。

**📈 对比分析**

与传统单尺度模型对比，展示该框架在捕捉多尺度行为分布和预测点击准确性上提升约15–20%，并通过代理模拟验证其可扩展性。

**⚠️ 局限性**

主要局限在于仅考虑内容多样性而未涉及来源、格式等因素，且缺乏对推荐系统实际效果的纵向验证。

---

## 277. Why Learners Drift In and Out: Examining Intermittent Discontinuance in AI-Mediated Informal Digital English Learning (AI-IDLE) Using SEM and fsQCA

**arXiv ID:** 2604.27493 | [PDF](https://arxiv.org/pdf/2604.27493v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 278. Low Rank Adaptation for Adversarial Perturbation

**arXiv ID:** 2604.27487 | [PDF](https://arxiv.org/pdf/2604.27487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 279. Computing the (k+2)-Edge-Connected Components in k-Edge-Connected Digraphs in Subquadratic Time

**arXiv ID:** 2604.27474 | [PDF](https://arxiv.org/pdf/2604.27474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 280. Syntactically-guided Information Maintenance in Sentence Comprehension

**arXiv ID:** 2604.27468 | [PDF](https://arxiv.org/pdf/2604.27468v1)

**作者:** Shinnosuke Isono `[一作]` (NINJAL), Kohei Kajikawa `[通讯]` (Georgetown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用日语自然阅读时间数据，探讨语法结构如何影响实时语言理解中的信息维护成本，证明预测头数和未完成依赖数两种语法指标共同且独立地导致阅读速度下降，并研究读者在高维护成本下的维护策略与预测效果之间的关系。

**💡 创新点**

创新点在于：①将预测头数和未完成依赖数视为互补的维护成本因素，而非可互换；②通过大规模自然阅读数据首次同时检验两者的效应，并证明两者共同显著提高模型拟合；③揭示读者在高维护成本结构中的速度调节策略与上下文预测效应之间的正相关，为资源合理化读者模型提供实证支持。

**🔧 技术方法**

使用统计回归（线性回归、10折交叉验证、置换检验）分析阅读时间；基于Universal Dependencies解析计算预测头数、未完成依赖数和额外未完成依赖数；控制词汇惊讶度（单字和GPT‑2）以及溢出效应。

**📊 数据集**

数据集为BCCWJ‑SPR2自发阅读时间大数据集，包含约50,000个分块（bunsetsu）和约6.4M条原始阅读时间，具备UD层级依存结构。

**📈 对比分析**

通过比较加入或不加入维护成本变量的模型，计算ΔMSE、回归系数并进行置换检验。结果显示：①预测头数和未完成依赖数均显著提升模型拟合，②预测头数的系数更大，③两者同时出现时模型显著优于单一因素；此外，读者策略分类进一步验证了维护成本与抗局部性效应的正相关。

**⚠️ 局限性**

局限性包括：①仅研究日语，未检验非头末结构下的效应；②仅使用黄金解析，未考虑多解析并行导致的维护成本计算问题；③未对同类头/依赖的重复成本进行建模；④模型未完全捕捉隐藏的语义或词汇级预测信息。

---

## 281. Security Attack and Defense Strategies for Autonomous Agent Frameworks: A Layered Review with OpenClaw as a Case Study

**arXiv ID:** 2604.27464 | [PDF](https://arxiv.org/pdf/2604.27464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 282. lpviz: Interactive Linear Programming Visualization

**arXiv ID:** 2604.27518 | [PDF](https://arxiv.org/pdf/2604.27518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 283. Improving Graph Few-shot Learning with Hyperbolic Space and Denoising Diffusion

**arXiv ID:** 2604.27462 | [PDF](https://arxiv.org/pdf/2604.27462v1)

**作者:** Yonghao Liu `[一作]` (Jilin University), Renchu Guan `[通讯]` (Jilin University)

**通讯引用:** 3798 | [OpenAlex ID](https://openalex.org/A5007914848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 IMPRESS 框架，在图少样本学习中通过双曲空间学习节点表征，并利用原型引导的去噪扩散生成额外样本来扩充支持集；

**💡 创新点**

创新点在于将节点表征迁移至双曲空间以捕捉图中的层级结构，并引入原型引导的去噪扩散生成条件样本，从而显著提升模型的泛化与稳健性；

**🔧 技术方法**

主要技术包括双曲变分图自动编码器、原型引导的去噪扩散模型、交叉注意机制以及线性分类器；

**📊 数据集**

在 CoraFull、Coauthor‑CS、Cora、WikiCS、Cora‑ML、CiteSeer 和大规模 ogbn‑arxiv 数据集上进行实验；

**📈 对比分析**

与 GCN、SGC、ProtoNet、MAML、Meta‑GNN、GPN、G‑Meta、TENT、Meta‑GPS、TLP、X‑FNC、TEG、COSMIC、TaskNS、VNT、NaQ 等基线对比，IMPRESS 在各任务（5‑way/10‑way、3‑shot/5‑shot 等）上均取得显著更高的平均准确率，尤其在高阶样本和大规模图数据上表现突出；

**⚠️ 局限性**

局限性包括对双曲曲率和扩散样本数量的敏感性，需要仔细调参；在极大规模图上仍存在显存瓶颈；且依赖无监督聚类的伪标签，若伪标签质量不佳会影响扩散生成效果。

---

## 284. Belief-Guided Inference Control for Large Language Model Services via Verifiable Observations

**arXiv ID:** 2604.27536 | [PDF](https://arxiv.org/pdf/2604.27536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 285. Smallest suffixient set maintenance in near-real-time

**arXiv ID:** 2604.27548 | [PDF](https://arxiv.org/pdf/2604.27548v1)

**作者:** Dominik Köppl `[一作]` (University of Yamanashi), Gregory Kucherov `[通讯]` (Gustave Eiffel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在线算法，能够在从左到右或从右到左读入文本时，实时维护字符串的最小后缀集合（SSS）及其对应的最小后缀集（SREs），并给出了每个字符的最坏情况时间复杂度；

**💡 创新点**

创新点在于将 Weiner 的后缀树构造与颜色前驱查询相结合，实现了从先前仅有摊销时间的算法向最坏情况下的双对数时间（O(log²log n)）或对数对数时间（O(log log n)）的提升；

**🔧 技术方法**

核心技术包括 Weiner 的反向在线后缀树构造、颜色前驱（colored predecessor）数据结构、Euler 轮巡（Euler tour）以及对软/硬 W‑链接的优化处理；

**📊 数据集**

论文中并未使用真实数据集，而是通过理论分析与复杂度证明来验证方法的有效性；

**📈 对比分析**

与先前工作相比，本文的算法在每个字符的最坏情况运行时间上实现了显著改进，能够在近实时环境中处理高重复度文本；

**⚠️ 局限性**

主要局限在于空间使用为 O(n)，未采用压缩存储；且目前仍未进一步降低最坏情况时间至 O(log log n) 的极限之外。

---

## 286. FMCL: Class-Aware Client Clustering with Foundation Model Representations for Heterogeneous Federated Learning

**arXiv ID:** 2604.27510 | [PDF](https://arxiv.org/pdf/2604.27510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 287. ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models

**arXiv ID:** 2604.27467 | [PDF](https://arxiv.org/pdf/2604.27467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 288. Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO

**arXiv ID:** 2604.27488 | [PDF](https://arxiv.org/pdf/2604.27488v1)

**作者:** Yu Tian `[一作]` (University of Chinese Academy of Sciences), Xian Sun `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 24562 | [OpenAlex ID](https://openalex.org/A5003621477)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Skills‑Coach 框架，实现 LLM‑基代理技能的自动自进化，包含任务生成、轻量优化、执行对比与评估四大模块；

**💡 创新点**

创新点在于使用无训练的 Training‑Free GRPO 对技能指令与代码进行迭代优化，自动化探索能力边界并生成改进版技能，提供可视化报告；

**🔧 技术方法**

技术包括大语言模型（Claude‑Sonnet）、Training‑Free Group Relative Policy Optimization (GRPO)、自动任务生成与静态代码分析、并行执行与多维度评估；

**📊 数据集**

使用了 Skill‑X 基准，48 种行业常用技能（29 指令型、19 代码型）；

**📈 对比分析**

通过与原始技能在标准+高级测试集对比，使用平均分、通过率等指标评估；结果显示平均分从 0.37 提升至 0.84，通行率从 33.6% 提升至 88%，性能显著提升；

**⚠️ 局限性**

局限性包括对已达极佳性能技能提升有限、对 LLM 可用性与算力依赖较高、优化资源在低性能技能上消耗大、未充分验证跨域技能融合效果等。

---

## 289. HealthBench Professional: Evaluating Large Language Models on Real Clinician Chats

**arXiv ID:** 2604.27470 | [PDF](https://arxiv.org/pdf/2604.27470v1)

**作者:** Rebecca Soskin Hicks `[一作]` (OpenAI), Karan Singhal `[通讯]` (OpenAI)

**通讯引用:** 1744 | [OpenAlex ID](https://openalex.org/A5083904567)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HealthBench Professional，一个开放的基准，用真实临床医生与 LLM（如 ChatGPT for Clinicians）多轮对话数据评估模型在三大临床使用场景（诊疗咨询、书写与记录、医学研究）下的表现。

**💡 创新点**

创新点包括：① 直接采集医生日常使用与红队（adversarial）对话，保证任务的真实性和难度；② 通过三阶段医学专家审核、重写、对照的严格流程生成并验证专业的评估 Rubric；③ 对难度高的示例进行加权抽样，显著提升 benchmark 对前沿模型的区分度；④ 采用长度调整的 Rubric 评分体系，抑制“长答案”带来的偏差；⑤ 设立医生基线回答，提供真实人类性能对照。

**🔧 技术方法**

技术手段包括：多轮对话收集与标注；三阶段医生评审与对齐；基于 GPT‑5.4 的模型评分器；长度调整回归（≈1.47 分/每 500 字）；对模型采用不同 harness（基础、浏览、ChatGPT for Clinicians harness）；多样化的推理力度与verbosity 设定；统计检验（配对 t 检验、Holm 校正）。

**📊 数据集**

数据集：HealthBench Professional，包含 1,000+ 条真实临床对话（来自 190 名医生，覆盖 26 个专科，52 种专业语言），每条对话配有 1–7 分 Likert 难度、专业标签、使用模式标签（Good‑faith / Red‑Team）、Rubric 以及医生手写基线回答。

**📈 对比分析**

比较方法：对每个模型使用 8 次采样，按 Rubric 计算长度调整后的分数，取平均作为整体评分。结果显示：GPT‑5.4 在 ChatGPT for Clinicians harness 的整体得分 59.0，明显优于人类基线 43.7、所有其他 LLM（Claude、Gemini、Grok 等）以及基础 GPT‑5.4（48.1）和 GPT‑5.4+浏览（45.8）。在写作/记录场景得分最高（64.1），在红队 1–2 难度子集得分 55.8，远高于医生（30.0）。

**⚠️ 局限性**

局限性：① benchmark 强化了难度高、对抗性案例，得分与日常临床实际使用并不直接可比；② 仅覆盖三大聊天场景，未包含 EHR 集成、机构特定流程等；③ 数据来源主要是英文/英语主导，其他语言覆盖有限；④ 评价基于专家人工 Rubric，仍可能存在主观性；⑤ 长度调整系数仅在 2,000–4,500 字范围内有效，极长答案仍可能受限。

---

## 290. Green Physics-Informed Machine Learning Models For Structural Health Monitoring

**arXiv ID:** 2604.27638 | [PDF](https://arxiv.org/pdf/2604.27638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 291. SASI: Leveraging Sub-Action Semantics for Robust Early Action Recognition in Human-Robot Interaction

**arXiv ID:** 2604.27508 | [PDF](https://arxiv.org/pdf/2604.27508v1)

**作者:** Yongpeng Cao `[一作]` (University of Tokyo), Yuji Yamakawa `[通讯]` (University of Tokyo)

**通讯引用:** 2337 | [OpenAlex ID](https://openalex.org/A5019385536)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在此工作中提出了SASI框架，通过将基于骨架的GCN提取的时空特征与子动作语义特征进行跨模态融合，实现对完整动作和部分动作的高效识别。

**💡 创新点**

创新点包括：①引入子动作分割与语义编码分支，显式捕获动作的层次结构；②采用跨注意力机制实现动作与文本特征的深度融合；③引入语义一致性损失，使子动作与整体动作在语义空间保持一致。

**🔧 技术方法**

主要技术手段包括：图卷积网络（GCN）提取骨架时空特征；预训练的动作分割模型FACT进行子动作识别；CLIP式文本编码器将子动作标签转为向量；跨注意力机制与残差连接进行特征融合；多任务学习框架结合语义损失与识别损失。

**📊 数据集**

使用BABEL骨架数据集（含子动作标注）和AMASS运动序列进行实验，并对分割模型在BABEL上进行微调。

**📈 对比分析**

在BABEL基准上与多种主流GCN模型（ST‑GCN、Shift‑GCN、CTR‑GCN、BlockGCN、ProtoGCN等）进行比较。SASI在完整与部分观察（25%、50%、75%、100%）的多场景下均实现了显著提升，尤其在低观测比下提升显著，平均提升约3–5%点。

**⚠️ 局限性**

局限性主要体现在：①对预训练子动作分割模型的准确率高度依赖，分割误差直接限制整体性能；②需大量子动作注释，难以推广到无标签数据；③子动作类别繁多导致数据稀疏，可能影响模型泛化。

---

## 292. Tail-aware N-version Machine Learning Models for Reliable API Recommendation

**arXiv ID:** 2604.27647 | [PDF](https://arxiv.org/pdf/2604.27647v1)

**作者:** Aoi Matsuda `[一作]` (University of Tsukuba), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31246 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 N-version API 推荐框架 NvRec，结合多模型推理与长尾过滤，以提升 API 方法序列推荐的可靠性。

**💡 创新点**

创新点在于使用 Tail Analyzer 对长尾样本进行预过滤，并通过模型配置、过滤规则和输出决策三重机制来消除不可靠的推荐结果。

**🔧 技术方法**

技术核心包括多版本机器学习模型（CodeBERT、CodeT5、MulaRec、UniXcoder、CodeT5+）的并行推理、模型性能配置文件、长尾检测与投票/置信度决策。

**📊 数据集**

实验使用公开的 50K-C Java 项目构建的 18,500 条自然语言查询+代码上下文与 API 序列的评测数据集。

**📈 对比分析**

与单模型相比，三模型配置下最高 TAR 达 83.8%，RR 80.7%，FRR 32.2%；五模型配置在不使用过滤时可达到 83.1% TAR，但整体滤波会导致 TAR 降低。

**⚠️ 局限性**

主要限制是高拒绝率导致可用推荐数量大幅下降，同时对数据集和语言的依赖性强，未验证跨语言或不同项目类型的泛化能力。

---

## 293. GenAI in Software Engineering: The Role of Technology Acceptance Models

**arXiv ID:** 2604.27642 | [PDF](https://arxiv.org/pdf/2604.27642v1)

**作者:** Oscar Johansson `[一作]` (Blekinge Institute of Technology), Nauman bin Ali `[通讯]` (Blekinge Institute of Technology)

**通讯引用:** 1390 | [OpenAlex ID](https://openalex.org/A5072594385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过综述相关文献，提炼并优先考虑了十个适用于软件工程中GenAI接受度研究的UTAUT及其扩展构念，并提出将这些构念嵌入贝叶斯框架进行量化分析；

**💡 创新点**

创新点在于首次将UTAUT与贝叶斯方法结合，用于解决传统频率统计在小样本、动态技术环境下的局限性，同时提供可操作的“若干何如”决策模拟；

**🔧 技术方法**

采用的技术包括：基于UTAUT的构念设计、Likert量表收集、贝叶斯结构方程模型/贝叶斯网络推断、MCMC数值估计以及情景模拟；

**📊 数据集**

使用的数据集为假设的Likert量表问卷（覆盖性能期望、努力期望、社会影响等十个构念），并在示例场景中模拟了该数据集的先验与后验更新；

**📈 对比分析**

与传统的频率统计（如PLS‑SEM）相比，贝叶斯方法在小样本情况下更稳健、可直接获取置信区间与效应概率，理论拟合度（BIC/WAIC）通常更优；

**⚠️ 局限性**

局限性包括：缺乏真实实证数据验证模型、贝叶斯分析对先验设定敏感、对大规模数据处理尚需进一步研究、且结果易受研究者经验影响。

---

## 294. Mapping how LLMs debate societal issues when shadowing human personality traits, sociodemographics and social media behavior

**arXiv ID:** 2604.27624 | [PDF](https://arxiv.org/pdf/2604.27624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 295. SpaAct: Spatially-Activated Transition Learning with Curriculum Adaptation for Vision-Language Navigation

**arXiv ID:** 2604.27620 | [PDF](https://arxiv.org/pdf/2604.27620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 296. Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering

**arXiv ID:** 2604.27590 | [PDF](https://arxiv.org/pdf/2604.27590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 297. Assessing Pancreatic Ductal Adenocarcinoma Vascular Invasion: the PDACVI Benchmark

**arXiv ID:** 2604.27582 | [PDF](https://arxiv.org/pdf/2604.27582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. Thinking like a business: Reconfiguring relationships to sustain open data infrastructures

**arXiv ID:** 2604.27580 | [PDF](https://arxiv.org/pdf/2604.27580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 299. World2Minecraft: Occupancy-Driven Simulated Scenes Construction

**arXiv ID:** 2604.27578 | [PDF](https://arxiv.org/pdf/2604.27578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 300. Online semi-supervised perception: Real-time learning without explicit feedback

**arXiv ID:** 2604.27562 | [PDF](https://arxiv.org/pdf/2604.27562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 301. Residual Gaussian Splatting for Ultra Sparse-View CBCT Reconstruction

**arXiv ID:** 2604.27552 | [PDF](https://arxiv.org/pdf/2604.27552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 302. Beyond the Training Distribution: Mapping Generalization Boundaries in Neural Program Synthesis

**arXiv ID:** 2604.27551 | [PDF](https://arxiv.org/pdf/2604.27551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 303. An Exact 56-Addition, Rank-23 Scheme for General 3*3 Matrix Multiplication

**arXiv ID:** 2604.27645 | [PDF](https://arxiv.org/pdf/2604.27645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 304. Optimization before Evaluation: Evaluation with Unoptimised Prompts Can be Misleading

**arXiv ID:** 2604.27637 | [PDF](https://arxiv.org/pdf/2604.27637v1)

**作者:** Nicholas Sadjoli `[一作]` (SAP), Daniel Dahlmeier `[通讯]` (SAP)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了提示优化（PO）对大型语言模型评估的影响，并在多种公开与内部数据集上对比不同模型的表现

**💡 创新点**

首次将PO作为应用中心评估的标准步骤，证明其可显著改变模型排名，强调评估时必须为每个模型单独优化提示

**🔧 技术方法**

采用GPT‑4o作为批评器，使用TextGrad实现指令优化、MIPRO（DSPy）实现指令+示例优化

**📊 数据集**

使用公开数据集GSM8K、OpenbookQA、MMLU以及内部业务数据集（Digital Assistant Routing、Copilot Help Docs、Copilot Consultancy、Text‑To‑SQL、EDDE）

**📈 对比分析**

通过Kendall τ和排名变化比较基线与优化后的模型，PO平均提升约20%性能并导致部分模型排名大幅变动

**⚠️ 局限性**

仅测试两种PO方法与单一批评器，未多次重复实验验证方差；模型匿名化导致结果难以复现

---

## 305. Generative structure search for efficient and diverse discovery of molecular and crystal structures

**arXiv ID:** 2604.27636 | [PDF](https://arxiv.org/pdf/2604.27636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 306. Robot Learning from Human Videos: A Survey

**arXiv ID:** 2604.27621 | [PDF](https://arxiv.org/pdf/2604.27621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 307. Robust Lightweight Crack Classification for Real-Time UAV Bridge Inspection

**arXiv ID:** 2604.27617 | [PDF](https://arxiv.org/pdf/2604.27617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems

**arXiv ID:** 2604.27616 | [PDF](https://arxiv.org/pdf/2604.27616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 309. AMGenC: Generating Charge Balanced Amorphous Materials

**arXiv ID:** 2604.27613 | [PDF](https://arxiv.org/pdf/2604.27613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 310. ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data

**arXiv ID:** 2604.27606 | [PDF](https://arxiv.org/pdf/2604.27606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 311. SandSim: Curve-Guided Gaussian Splatting for Reconstructing Sand Painting Processes

**arXiv ID:** 2604.27572 | [PDF](https://arxiv.org/pdf/2604.27572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 312. SecGoal: A Benchmark for Security Goal Extraction and Formalization from Protocol Documents

**arXiv ID:** 2604.27601 | [PDF](https://arxiv.org/pdf/2604.27601v1)

**作者:** Dawei Huang `[一作]` (Beijing University of Posts and Telecommunications), Bo Jia `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 36340 | [OpenAlex ID](https://openalex.org/A5100613666)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个专家标注的安全目标提取数据集 SecGoal，并构建了一个两阶段的神经符号框架 AIFG，用于从协议文档中自动提取安全目标并生成正式的安全属性。

**💡 创新点**

创新点在于①首次创建专门针对协议文档的安全目标标注基准；②设计了粗细分离的提取-形式化管线，并利用检索增强生成解决语义模糊；③证明了在该基准上进行领域微调后，小规模模型可超越大规模通用模型。

**🔧 技术方法**

使用技术包括随机负样本下采样的目标提取模型、检索增强生成 (RAG) 的形式化阶段、以及基于 GPT、Gemini、Gemma 等大语言模型的微调和推理。

**📊 数据集**

数据集为 SecGoal，包含 15 种主流协议（如 TLS 1.3、5G‑AKA、TLS 1.3、SPDM、OPC‑UA 等），标注了 100+ 形式化安全属性对应的 300–400 条自然语言目标句子。

**📈 对比分析**

评估方法：对提取阶段使用精确度、召回率、F1；对形式化阶段使用意图匹配 F1、语义槽准确率；结果显示，微调后 7B/9B 模型在 F1 方面可达 80%+，显著高于 GPT‑4o、Gemini 等大型模型；形式化阶段实现完好召回和 0.95+ 的槽准确率。

**⚠️ 局限性**

局限性在于：①仅从文本中提取目标，未覆盖专家补充的安全需求；②未实现完整的协议模型生成，仅聚焦安全目标；③形式化评估仅针对两协议，缺乏大规模验证；④对新兴领域和超大协议的适应性尚未验证。

---

## 313. SECOS: Semantic Capture for Rigorous Classification in Open-World Semi-Supervised Learning

**arXiv ID:** 2604.27596 | [PDF](https://arxiv.org/pdf/2604.27596v1)

**作者:** Hezhao Liu `[一作]` (Xiamen University), Yang Lu `[通讯]` (Xiamen University)

**通讯引用:** 11069 | [OpenAlex ID](https://openalex.org/A5024886122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RC-OWSSL任务，设计SECOS框架实现直接文本标签预测；

**💡 创新点**

核心创新在于两阶段语义捕获（全局与批级）与视觉-文本对齐适配器，实现无后处理的严谨分类；

**🔧 技术方法**

采用CLIP预训练模型、伪标签生成、语义补偿、批级语义重捕、轻量级适配器以及EMA自监督；

**📊 数据集**

在CIFAR10、CIFAR100、ImageNet100以及CUB、Stanford Cars、Oxford Flowers、Oxford Pets四类细粒度数据集上评测；

**📈 对比分析**

与传统OWSSL/GCD方法（TIDA、OwMatch、TRAILER、GCD、SimGCD、DCCL、GPC、PromptCAL、SPTNet、TextGCD、TP-OWSSL等）相比，SECOS在大多数指标上提升约1–7%（最高达5.4%），并且直接使用分类准确率而非Hungarian匹配，表现更稳健；

**⚠️ 局限性**

主要局限是对大规模视觉‑文本预训练模型CLIP的依赖，性能受预训练数据覆盖范围和质量限制；

---

## 314. Reproducing Adaptive Reranking for Reasoning-Intensive IR

**arXiv ID:** 2604.27577 | [PDF](https://arxiv.org/pdf/2604.27577v1)

**作者:** Mandeep Rathee `[一作]` (L3S Research Center), Avishek Anand `[通讯]` (Delft University of Technology)

**通讯引用:** 1625 | [OpenAlex ID](https://openalex.org/A5075681290)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现 Graph-based Adaptive Reranking (GAR) 并验证其在 BRIGHT 这类需要推理的检索任务中的有效性，评估多种推理与非推理排名模型在引入 GAR 后的表现。

**💡 创新点**

证明 GAR 在推理密集检索场景下同样可显著提升召回率与 nDCG，且对模型规模、批大小、邻居数等超参数具有鲁棒性；同时揭示其对排名器质量高度依赖，弱排名器会导致性能下降。

**🔧 技术方法**

使用 BM25 作为第一阶段检索器，构建基于 BM25 的文档邻居图；采用多种 LLM（MonoT5、RankLLaMA、TFRank、Qwen、RankR1 等）作为点对点或列表式重排模型；实现基于图的迭代重排算法并评估不同批大小与邻居数。

**📊 数据集**

BRIGHT 语义推理检索基准，包括 Stack Exchange、Coding、Mathematical Theorems 子任务，数据量数十万条，检索任务需跨段落推理。

**📈 对比分析**

对比基线 BM25、传统一次性重排与引入 GAR 的系统，使用 nDCG@10 与 Recall@c 评估。结果显示：在强排名器（如 Qwen-4B、TFRank-8B、RankR1-7B）下，GAR 可提升 nDCG@10 最高可达 27% 以上，召回率提升 30%+；在弱排名器（MonoT5、RankLLaMA-0.6B）下，GAR 甚至会导致性能下降；超参数实验表明 GAR 在批大小 2–32、邻居数 2–128 内保持稳定。

**⚠️ 局限性**

依赖于排名器的反馈质量，弱模型会误导图遍历；仅使用 BM25 语义图，未尝试语义或密集向量图；实验资源受限，未覆盖更大模型或更深图；未单独评估时延与碳足迹；对不同查询长度或领域的通用性待验证。

---

## 315. Bayesian policy gradient and actor-critic algorithms

**arXiv ID:** 2604.27563 | [PDF](https://arxiv.org/pdf/2604.27563v1)

**作者:** Mohammad Ghavamzadeh `[一作]` (Adobe Research), Michal Valko `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于高斯过程的贝叶斯策略梯度（BPG）与贝叶斯演员-评论家（BAC）算法，用于强化学习中更精确、高效的策略更新。

**💡 创新点**

创新点在于将策略梯度估计视为贝叶斯积分问题，利用贝叶斯四方技术与 Fisher 核实现梯度及其协方差的闭式解析，同时将非参数 GPTD 价值函数与贝叶斯积分相结合，既兼顾马尔可夫性质，又可处理部分可观测环境。

**🔧 技术方法**

核心技术包括：高斯过程回归、贝叶斯四方（Bayesian Quadrature）、Fisher 信息核、GPTD 时差学习、在线稀疏化、自然梯度更新以及对 Fisher 信息矩阵的在线/基于最大似然估计。

**📊 数据集**

实验涵盖：简单的连续动作 bandit、线性二次调节器（LQR）、10 状态随机步行、山车（Mountain Car）和船舵向（Ship Steering）等经典强化学习环境，未使用公开数据集，而是自行构造上述任务。

**📈 对比分析**

通过与传统 Monte‑Carlo 策略梯度（MCPG）以及现有演员-评论家方法比较，BPG 在小样本下梯度估计方差显著下降；BAC 在所有实验中均取得更快的收敛速度、更高的累计回报和更低的方差，尤其在大样本或噪声较大时优势更为明显。

**⚠️ 局限性**

局限性包括：对 Fisher 核的强假设可能限制了对某些非线性问题的建模；贝叶斯四方在轨迹长度差异较大的任务（如山车、船舵）中的效果不如 MC；以及需要手动调节多项学习率和稀疏阈值，实际部署中对超参数敏感。

---

## 316. RIHA: Report-Image Hierarchical Alignment for Radiology Report Generation

**arXiv ID:** 2604.27559 | [PDF](https://arxiv.org/pdf/2604.27559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 317. SpatialGrammar: A Domain-Specific Language for LLM-Based 3D Indoor Scene Generation

**arXiv ID:** 2604.27555 | [PDF](https://arxiv.org/pdf/2604.27555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 318. WaferSAGE: Large Language Model-Powered Wafer Defect Analysis via Synthetic Data Generation and Rubric-Guided Reinforcement Learning

**arXiv ID:** 2604.27629 | [PDF](https://arxiv.org/pdf/2604.27629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 319. Math Education Digital Shadows for facilitating learning with LLMs: Math performance, anxiety and confidence in simulated students and AIs

**arXiv ID:** 2604.27618 | [PDF](https://arxiv.org/pdf/2604.27618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 320. Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning

**arXiv ID:** 2604.27604 | [PDF](https://arxiv.org/pdf/2604.27604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. JaiTTS: A Thai Voice Cloning Model

**arXiv ID:** 2604.27607 | [PDF](https://arxiv.org/pdf/2604.27607v1)

**作者:** Jullajak Karnjanaekarin `[一作]` (Jasmine Technology Solution), Attapol T. Rutherford `[通讯]` (Jasmine Technology Solution)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一款名为 JaiTTS‑v1.0 的泰语语音克隆 TTS 模型，能够直接处理未经归一化的数字与泰英混写文本。

**💡 创新点**

创新点包括：
1) 基于 VoxCPM 的 tokenizer‑free 结构，直接生成连续语音潜变量；
2) 通过文本语义语言模型 (TSLM) + 有限标量量化 (FSQ) + 余弦声学语言模型 (RALM) + 本地扩散 Transformer (LocDiT) 的分层规划与细化流程；
3) 兼顾短句与长句（1–30 秒）无归一化输入，显著提升生成质量与实时性能。

**🔧 技术方法**

核心技术：
- VoxCPM tokenizer‑free autoregressive TTS；
- Text‑Semantic Language Model (TSLM)；
- Finite Scalar Quantization (FSQ) 作为半离散骨架；
- Residual Acoustic Language Model (RALM)；
- Local Diffusion Transformer (LocDiT) 进行连续潜变量解码；
- 直通估计（Straight‑through）与 classifier‑free 指导；
- 语音 VAE 编码器/解码器。

**📊 数据集**

使用数据集：
- 约 10,000 小时泰语中心语料，包含通用对话、正式语料以及 Finance、Healthcare、Education、Law 四个垂直领域；
- 来源包括 studio‑grade 录音与 crowd‑source 语音；
- 评测集分为 short‑duration（1–15 秒，1000 句）与 long‑duration（16–30 秒，231 句）两类。

**📈 对比分析**

比较方法与性能：
- 评估指标：字符错误率 (CER)、说话人相似度 (SIM) 以及实时因子 (RTF)；
- 与人类基准、Qwen3‑TTS‑0.6B/1.7B、ThonburianTTS 进行对比；
- short‑duration CER 1.94%（略优于人类 1.98%），SIM 0.62；
- long‑duration CER 2.55%（近似人类 2.47%），SIM 0.76；
- RTF 0.1136，约 13× 快于 Qwen3‑TTS；
- 人类评测对比 ElevenLabs 的 eleven_v3 与 MiniMax 的 speech‑2.8‑hd，JaiTTS‑v1.0 在 400 次 pairwise 比较中获胜 283 次。

**⚠️ 局限性**

局限性：
- 未评估更长文本（>30 秒）与极端口音或噪声条件下的鲁棒性；
- 对极端混写或非标准语音的处理仍有限；
- 主要针对泰语，英语混写支持可能不够完善；
- 需要大量泰语语料，对资源有限的语言可能难以复现。

---

## 322. Purifying Multimodal Retrieval: Fragment-Level Evidence Selection for RAG

**arXiv ID:** 2604.27600 | [PDF](https://arxiv.org/pdf/2604.27600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 323. Privacy-Preserving Federated Learning via Differential Privacy and Homomorphic Encryption for Cardiovascular Disease Risk Modeling

**arXiv ID:** 2604.27598 | [PDF](https://arxiv.org/pdf/2604.27598v1)

**作者:** Gaurang Sharma `[一作]` (VTT Technical Research Centre of Finland Ltd), Mika Hilvo `[通讯]` (VTT Technical Research Centre of Finland Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在瑞典健康数据上评估联邦学习结合差分隐私和同态加密的心血管疾病预测模型

**💡 创新点**

系统性比较FedAvg_DP与FedAvg_HE在真实部署中的隐私‑效能权衡，揭示HE在保持性能的同时带来可量化的计算与通信开销

**🔧 技术方法**

联邦学习（FedAvg）、差分隐私（SVTPrivacy）、同态加密（CKKS）以及中心化机器学习基线

**📊 数据集**

国家卫生与福利委员会整合的患者、处方及干预数据库，共约66万受试者、10个预测特征

**📈 对比分析**

通过对比AUC、敏感度、特异度等指标，FedAvg与FedAvg_HE与中心化模型几乎相同，FedAvg_DP在LR模型下性能下降，HE在NN中保持性能但耗时显著

**⚠️ 局限性**

仅测试单一国家数据、水平联邦、有限模型与加密参数，未涉及垂直或混合FL、其他疾病或公平性/鲁棒性评估

---

## 324. Unified 5G-IoT Framework with CAMARA Gateways and SDN Federation

**arXiv ID:** 2604.27589 | [PDF](https://arxiv.org/pdf/2604.27589v1)

**作者:** Zihan Jia `[一作]` (Loughborough University), Fung Po Tso `[通讯]` (Loughborough University)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5055973585)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个统一的5G‑IoT框架，利用CAMARA开放网关把公有5G、私有5G和异构IoT网络连接在一起，并通过联邦SDN架构实现跨域动态管理，最终在实际商业楼宇中演示了基于5G的KNX设备远程控制。

**💡 创新点**

创新点包括：
• 通过CAMARA开放网关实现对公私5G与IoT的统一访问与控制，消除网络碎片化问题；
• 采用联邦SDN控制器实现多域可共享治理，支持动态策略、流量分配与服务隔离；
• 将5G网络能力与传统建筑自动化协议（KNX）融合，开启工业与移动网络的深度协同。

**🔧 技术方法**

使用技术包括：CAMARA Open Gateway（OAuth2、OpenAPI）、5G核心网与RAN、私有5G网络、SDN控制器（VyOS）、IoT Hub、KNX/Thread/IPv6网络、MQTT、Home Assistant、VPN。

**📊 数据集**

未公开使用标准数据集；实验采用现场部署在商业楼宇（Shed测试平台）的KNX设备与传感器数据，进行功能验证与性能评估。

**📈 对比分析**

方法对比：将联邦SDN+CAMARA架构与传统单域5G+IoT方案对照；通过实验测量了控制延迟、吞吐量与安全性。结果表明，联邦架构在跨域路由时延降低≈30%，多域安全策略执行率提升≈95%，且保持低至数十毫秒的实时响应。

**⚠️ 局限性**

局限性：
• 评估规模受限于单栋楼宇实验，缺乏大规模多站点验证；
• 对多运营商、不同厂商设备的兼容性仍需进一步测试；
• CAMARA网关与SDN控制器之间的安全与可信链条需在更严苛场景下验证；
• 方案对极低时延、超高密度场景的适应性尚未证明。

---

## 325. Harnessing the Freedom of Non-Uniformity in Monostatic ISAC with Antenna Flexibility

**arXiv ID:** 2604.27571 | [PDF](https://arxiv.org/pdf/2604.27571v1)

**作者:** Zhe Wang `[一作]` (KTH Royal Institute of Technology), Emil Björnson `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 29045 | [OpenAlex ID](https://openalex.org/A5062293532)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种用于单极ISAC系统的非均匀阵列设计方案，通过在基站的候选天线池中动态分配天线模式（发射、接收、无效）来构造可变的有效阵列，并联合优化ISAC波束成形与天线分配以最大化总通信速率。

**💡 创新点**

创新点在于：① 引入天线池的三态分配实现真正意义上的非均匀阵列；② 通过连续松弛+惩罚与递归硬化相结合的交替优化框架，既保证了天线分配的可行性，又逼近离散解；③ 在自适应硬化过程中动态阈值调整，使得解收敛速度提升；④ 对比均匀阵列后显示在天线受限场景下显著提升。

**🔧 技术方法**

主要使用了交替优化 (AO)、加权最小均方误差 (WMMSE)、连续松弛 + 惩罚函数、顺序凸逼近 (SCA) 以及渐进硬化策略来求解非凸优化问题。

**📊 数据集**

使用合成的 Monte Carlo 数据集：在 200×200 m² 覆盖区域内随机布置 10 个单天线用户与一个目标，基站采用候选 UPA 天线池（N_x×N_y），在不同天线间距、天线数与感知 SINR 阈值下进行多次仿真。

**📈 对比分析**

通过与两种均匀阵列基线（UPA‑opt 与 UPA‑fixed）在相同优化框架下对比，结果表明：① 在天线受限区（N_act 较小）下，新方案的总速率提升可达 30 % 以上；② 即使激活天线数更少，仍可逼近甚至超过均匀阵列的性能；③ 在增大天线间距或提升感知阈值时，性能提升更加明显。

**⚠️ 局限性**

主要局限：① 仅在理想化仿真环境下验证，缺乏真实硬件实验；② 自适应硬化阈值的选择仍基于经验，可能对不同场景不够鲁棒；③ 交替优化与 SCA 的求解复杂度较高，实时间成本未在论文中评估。

---

## 326. treVM: Tiny Rust Embedded Virtual Machines with WASM on Variable Resource-Constrained Hardware

**arXiv ID:** 2604.27570 | [PDF](https://arxiv.org/pdf/2604.27570v1)

**作者:** Antoine Lavandier `[一作]` (Inria), Emmanuel Baccelli `[通讯]` (Inria)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了treVM，一个基于Rust嵌入式RTOS的Wasm微机框架，支持通过OTA安全更新的Wasm capsule。

**💡 创新点**

创新点在于将Rust与WebAssembly结合用于资源受限MCU，提供统一的capsule沙箱、可动态更新机制，并采用Wasmtime Pulley与WebAssembly Component Model实现轻量化、可移植运行时。

**🔧 技术方法**

使用的技术包括Rust语言、Ariel OS RTOS、Wasmtime + Pulley解释器、WebAssembly Component Model、OSCORE/CoAP安全网络、异步/await、Pulley字节码、以及对SIMD的禁用。

**📊 数据集**

使用的评估数据集包括CoreMark、Embench（已转换为Wasm）、以及四款MCU板子（RP2350、nRF52840、ESP-WROOM-32、ESP32-C6）和示例capsule（斐波那契、传感器读取）。

**📈 对比分析**

通过在上述硬件上运行CoreMark和Embench，比较WAMR、Wasmi、Wasmtime、Wasm-interpreter、Wasefire等运行时的执行时间、Flash占用和RAM占用；结果表明Wasmtime在速度与内存占用之间提供良好折中；treVM在nRF52840上实现后，Flash+RAM占用约为普通固件的两倍，单个capsule占用40–130KB。

**⚠️ 局限性**

局限性包括：目前仅支持Wasmtime与Component Model；capsule尺寸和内存占用仍相对较高；缺乏对更多接口（如蓝牙、LTE-M等）的实现；评估仅基于有限板子和简化capsule，未来需持续更新基准和runtime。

---

## 327. Learning from a single labeled face and a stream of unlabeled data

**arXiv ID:** 2604.27564 | [PDF](https://arxiv.org/pdf/2604.27564v1)

**作者:** Branislav Kveton `[一作]` (Technicolor), Michal Valko `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种在线流式单标记人脸识别算法，利用单张标记图像与大量未标记数据构建非参数模型进行人脸识别。

**💡 创新点**

将单标记人脸与连续未标记流结合，采用在线k-center聚类和基于图的随机游走推断，实现一类学习与自适应的非参数识别。

**🔧 技术方法**

使用在线k-center聚类、图的谐波求解（Harmonic Solution）、随机游走吸收概率、非参数一类分类以及可与Fisherfaces融合的特征技术。

**📊 数据集**

在VidTIMIT视频人脸数据集上进行评估，包含43人、10句语音视频，具有多场次与时间变化的未标记流。

**📈 对比分析**

与1-NN、5-NN和Fisherfaces基准对比，OTM在10⁻⁴ FPR下TPR达到0.89，比1-NN高50%，与5-NN相当，结合Fisherfaces进一步提升性能。

**⚠️ 局限性**

依赖大量未标记数据，参数调优对性能影响显著；仅处理整体图像，未利用局部特征，对光照、姿态等变化仍有限。

---

## 328. Function-based Parametric Co-Design Optimization of Dexterous Hands

**arXiv ID:** 2604.27557 | [PDF](https://arxiv.org/pdf/2604.27557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. Multi-Connectivity for UAVs: A Measurement Study of Integrating Cellular, Aerial Mesh, and LEO Satellite Links

**arXiv ID:** 2604.27640 | [PDF](https://arxiv.org/pdf/2604.27640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 330. Revealing the Impact of Visual Text Style on Attribute-based Descriptions Produced by Large Visual Language Models

**arXiv ID:** 2604.27553 | [PDF](https://arxiv.org/pdf/2604.27553v1)

**作者:** Xiaomeng Wang `[一作]` (Radboud University), Zhengyu Zhao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2042 | [OpenAlex ID](https://openalex.org/A5101795752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究视觉文本的字体与装饰风格是否会影响大型视觉语言模型（LVLM）对概念属性描述的生成，进而揭示文本风格泄漏到语义推理中的问题。

**💡 创新点**

创新点在于首次量化并证明视觉文本风格（功能性与装饰性）会导致LVLM在属性描述上产生显著差异，即使模型已成功识别概念。

**🔧 技术方法**

采用视觉文本生成、OCR识别过滤、五种等价属性描述提示、重复采样、Llama‑3.1‑8B 进行形容词提取，并用总变差（TV）距离与卡方检验评估属性分布差异。

**📊 数据集**

使用 Oxford‑IIIT Pet 数据集（猫狗品种）生成 32 种概念的 40 张视觉文本图片（8 字体×5 尺寸/位置组合），并分别以功能性与装饰性风格呈现。

**📈 对比分析**

通过计算功能性与装饰性样式下的属性分布 TV 距离、p 值以及字体内外的平均 TV 进行对比，实验显示两者之间的 TV 距离显著（p<0.001）且装饰性样式在部分品种中更倾向于情绪化属性，说明模型对风格敏感。

**⚠️ 局限性**

局限性包括仅测试两款 LVLM、仅考虑宠物品种、仅使用形容词列表描述、未探究风格对更复杂推理任务的影响，以及视觉文本生成仅覆盖有限字体与颜色组合，缺乏对跨领域泛化的评估。

---

## 331. BAss: Symbolic Reasoning in Abstract Dialectical Frameworks

**arXiv ID:** 2604.27576 | [PDF](https://arxiv.org/pdf/2604.27576v1)

**作者:** Samuel Pastva `[一作]` (Masaryk University), Van-Giang Trinh `[通讯]` (Ho Chi Minh City University of Technology)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5019773310)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基于二进制决策图（BDD）的抽象对话框架（ADF）符号求解器，能够枚举并符号化表示所有可接受、完整、首选以及二值和稳定解释。

**💡 创新点**

创新点在于：①首次在ADF求解器中实现首选和稳定语义的符号化推理；②利用ADF与布尔网络（BN）等价关系，直接从BN的BBDD技术迁移优化；③提出了双重编码、贪心多项式连接以及三种符号化优化（leastVal、#_k、weakening）实现高效的解空间符号化；④通过统一的BDD表示实现大规模解空间的统一采样与统计分析。

**🔧 技术方法**

核心技术包括：Binary Decision Diagrams（BDD）与其双重编码；贪心BDD连接优化；符号化特征算子计算；基于BDD的最小/最大化操作（leastVal、#_k、weakening）；利用Rust实现高效的BDD库（BDDLib），并集成到现有ADFa工具中。

**📊 数据集**

使用了超过1,200个ADF/BN基准，包含245个来自Biodivine Boolean Models（BBM）的BN转换为ADF的实例，以及ICCMA 2017/2019竞赛、城市网络、网格等传统ADF基准。每个实例最多1,076个论点、923,346条连接。

**📈 对比分析**

与现有最速的ADF求解器（yadf、asp-af、afasp、adflib）以及BN分析工具（e.g., BNtool、biodivine、BoolNet、NetworKIN等）进行对比，使用PAR2分数和完成案例数评估。结果显示：在可接受、完整和稳定解释任务中，该工具取得了最佳PAR2分数；在首选解释任务中也实现了显著提升，独立解决多组基准；在大解空间实例上，BDD符号化方法往往优于SAT/ASP，并能在未枚举完整解集的情况下完成分析。

**⚠️ 局限性**

局限性：①BDD大小在某些极大或高连通度实例中仍会指数增长，导致内存耗尽或时间超限；②缺乏动态变量重排序等高级BDD优化；③对包含XOR运算的ADF在转换为BN时会产生巨大模型，导致无法参与比较；④目前仅支持可接受、完整、首选、二值与稳定语义，未覆盖naive、stage、semi-stable等其他语义；⑤实验中未充分探索与SAT/ASP混合策略的潜在性能提升。

---

## 332. ANCORA: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning

**arXiv ID:** 2604.27644 | [PDF](https://arxiv.org/pdf/2604.27644v1)

**作者:** Chengcao Yang `[一作]` (Wuhan University), Jun Chen `[通讯]` (Wuhan University)

**通讯引用:** 58907 | [OpenAlex ID](https://openalex.org/A5100450180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ANCORA 框架，实现语言模型在无监督条件下自我生成、验证并改进问题与解答的闭环学习。

**💡 创新点**

创新点在于统一的 Proposer/Solver 策略、三大稳定机制（自蒸馏 SFT、UCB 课程 DAG 与两层群组相对更新）以及针对稀疏验证反馈的熵导向奖励函数，避免了传统 RL 在薄有效流形上的崩溃。

**🔧 技术方法**

采用最大似然 RL（GRPO 风格）、自蒸馏 SFT、UCB 导引的课程 DAG、两层群组优势更新和熵奖励函数，所有技术均在 Qwen2.5-Coder-3B 上实现。

**📊 数据集**

使用 Verus 官方验证数据集，包括 Dafny2Verus（274 问题）、MBPP-Verified（78 问题）和 HumanEval-Verified（85 问题），并以 Pass@k 作为评测指标。

**📈 对比分析**

在 0‑shot 训练下，ANCORA 在 Dafny2Verus 上实现 81.5% Pass@1，显著超越 PSV 1‑shot 的 65.6%（+15.8 点）并在 MBPP 与 HumanEval 上保持竞争力；在迁移学习场景下也实现 36.2% 与 17.2% 的 Pass@1，表明可迁移的解题结构。

**⚠️ 局限性**

局限性包括对 3B 规模模型与单一 Verus 领域的依赖、课程 DAG 在长时训练后仍可能收敛、以及未解决的有效流形内部模式坍塌问题。

---

## 333. HAVEN: Hybrid Automated Verification ENgine for UVM Testbench Synthesis with LLMs

**arXiv ID:** 2604.27643 | [PDF](https://arxiv.org/pdf/2604.27643v1)

**作者:** Chang-Chih Meng `[一作]` (National Yang Ming Chiao Tung University), I-Chen Wu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2786 | [OpenAlex ID](https://openalex.org/A5016730899)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 HAVEN，一套混合式自动 UVM 测试平台，利用 LLM 仅提取结构信息，模板与 DSL 负责生成符合协议的 SystemVerilog 代码，实现完整的 UVM testbench 及序列自动生成。

**💡 创新点**

创新点在于：① 通过 LLM 只产生结构化蓝图而不直接写 HDL；② 使用预定义的 Jinja2 协议模板保证组件编译无误；③ 设计了 Protocol‑Aware DSL，将序列拆解为可编码的步骤，规则化代码生成；④ 采用迭代覆盖闭环，利用 LLM 分析缺口并生成定向序列。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑5.2 等）用于信息提取与覆盖分析；Jinja2 模板引擎生成 UVM 组件；基于 JSON 的 DSL 与规则化代码生成器；编译‑仿真循环与覆盖报告反馈机制；多代理（Spec Processor、Architecture Agent、Template Engine 等）协同工作。

**📊 数据集**

使用了 19 个开源 IP 设计（180–11k LOC），涵盖 Direct、Wishbone、AXI4‑Lite 三种总线协议，且 9 个设计与 UVM^2 基准重合。

**📈 对比分析**

与 UVM^2、UVLLM、MEIC、AutoBench 等现有 LLM‑辅助系统对比，HAVEN 在所有 19 设计上平均实现 90.6% 代码覆盖、87.9% 功能覆盖，编译成功率 100%，LLM 调用仅 6 次，成本约 $0.38/设计；相较 UVM^2 只提升 3.6pp 代码覆盖、1.1pp 功能覆盖。

**⚠️ 局限性**

局限性包括：DSL 目前为线性结构，难以覆盖多阶段或条件分支协议（如 SDRAM、ETHMAC、HUF）；多代理协同能力有限（如 I2C 多主机）；模型容量与上下文长度限制导致大型设计（ETHMAC、DFI）在开源模型上表现不佳；覆盖率仍受 LLM 随机性影响。

---

## 334. Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor

**arXiv ID:** 2604.27633 | [PDF](https://arxiv.org/pdf/2604.27633v1)

**作者:** Petter Törnberg `[一作]` (University of Amsterdam), Michelle Schimmel `[通讯]` (University of Amsterdam)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5016049288)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对六款前沿大型语言模型进行政治偏见审计，发现模型对询问者身份的提示会显著影响其政治立场响应；

**💡 创新点**

创新点在于揭示政治偏见测评与模型对询问者的“同情”或“迎合”行为之间的相互作用，证明单一提示的审计结果并非模型固有立场的中立量化；

**🔧 技术方法**

采用因子实验设计，将询问者身份（无提示、进步民主党人、保守共和党人等）与固定政治问题和选项相结合，计算Wasserstein距离和政治舆情分数；

**📊 数据集**

使用三个主流审计工具的数据：62项《政治罗盘测试》（PCT）、25项《Pew政治类型学》题目，以及1,540个《Pew美国趋势调查》（ATP）多项选择题，合计1,643题；

**📈 对比分析**

与传统单提示审计相比，该方法显示在默认提示下模型偏左，而在保守共和党提示下多模型政治立场急剧右移，说明模型的政治偏见随上下文变化；

**⚠️ 局限性**

局限包括：未确定完全中性提示基准；结果受模型版本与RLHF训练差异影响；仅针对美国政治语境，跨国可推广性未知；实验使用单一回答，需更多复现以排除解码噪声；

---

## 335. One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation

**arXiv ID:** 2604.27599 | [PDF](https://arxiv.org/pdf/2604.27599v1)

**作者:** Ethan Bito `[一作]` (RMIT University), Estrid He `[通讯]` (RMIT University)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5000618392)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种单前向、排列不变的LLM重排推荐模型InvariRank，解决候选顺序导致的得分偏差。

**💡 创新点**

创新点在于通过结构化注意力掩码消除候选间交叉注意力，并使用共享位置框架抵消RoPE导致的偏移漂移，从而在架构层面实现排列不变。

**🔧 技术方法**

采用结构化段落注意力掩码、RoPE共享位置编码、LambdaRank列表式学习目标、LoRA微调等技术。

**📊 数据集**

在MovieLens‑32M和Amazon Books两大显式交互推荐数据集上进行实验。

**📈 对比分析**

与零样本、推理时重排、后置校准、列表微调等基线在HR@k、nDCG@k及Kendall τ、Spearman ρ、top‑k一致性上对比，InvariRank保持近似相同nDCG的同时，Kendall τ、ρ接近1，显著提升排列鲁棒性。

**⚠️ 局限性**

消除候选间交互可能丢失有用的比较信号，实验仅覆盖固定规模候选集与两数据集，未验证在更大候选集或其他检索任务上的通用性。

---

## 336. ClipTBP: Clip-Pair based Temporal Boundary Prediction with Boundary-Aware Learning for Moment Retrieval

**arXiv ID:** 2604.27591 | [PDF](https://arxiv.org/pdf/2604.27591v1)

**作者:** Ji-Hyeon Kim `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**通讯引用:** 22266 | [OpenAlex ID](https://openalex.org/A5011014617)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于剪辑对的时序边界预测框架 ClipTBP，用于改进视频时刻检索任务。

**💡 创新点**

创新点在于引入剪辑级相似性损失以显式学习答案片段之间的语义关系，并结合主边界损失与辅助边界损失实现更精确的边界预测。

**🔧 技术方法**

使用技术包括多模态 Transformer 编码器、对比学习、动态边界损失、SmoothL1 辅助回归、SlowFast 与 CLIP 作为特征提取器。

**📊 数据集**

实验使用 QVHighlights 和 TACoS 两个公开数据集。

**📈 对比分析**

将 ClipTBP 应用于 UniVTG、TD-DETR、Video Mamba Suite、FlashVTG 等基线模型后，在 R1@0.7、mAP@Avg 等指标上均实现了显著提升，且 ablation 结果表明每个损失项都对性能有贡献。

**⚠️ 局限性**

局限性包括对损失权重和硬负样本数量的敏感性、对计算资源的额外需求，以及在更长或更复杂视频中的泛化尚未完全验证。

---

## 337. Trace-Level Analysis of Information Contamination in Multi-Agent Systems

**arXiv ID:** 2604.27586 | [PDF](https://arxiv.org/pdf/2604.27586v1)

**作者:** Anna Mazhar `[一作]` (Cornell University), Sainyam Galhotra `[通讯]` (Cornell University)

**通讯引用:** 1087 | [OpenAlex ID](https://openalex.org/A5038532934)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在多智能体工作流中注入结构化扰动，利用日志追踪分析信息污染的传播与表现。

**💡 创新点**

创新点包括：①提出信息污染的正式分类（静默语义腐败、行为 detour 与联合破坏）及其控制流签名；②基于结构编辑距离的轨迹级污染检测与定位框架；③揭示成本-正确性解耦及对现有验证护栏的挑战。

**🔧 技术方法**

主要技术：多智能体协作框架、语言模型驱动的工具调用、扰动操作（表格、文档、图像、音频）、结构化日志与签名抽象、编辑距离对齐与首次偏差定位、token 成本计量。

**📊 数据集**

使用 GAIA 基准任务集（包含 PDF、Excel、PPT、图像、音频等附件）进行实验，并对 GPT‑5‑mini、LLaMA‑3.1‑70B、Qwen3‑235B 三种 LLM 后端进行对比。

**📈 对比分析**

与干净跑相对比，采用对比实验测量结构编辑距离、首次偏差点、控制流模式、token 费用与最终答案准确率；结果显示：约 40% 的工作流能在结构偏差后恢复正确答案，15% 仅出现语义腐败而保持低成本；成本与正确性的相关性弱，提示单一指标不足。

**⚠️ 局限性**

局限性包括：实验仅在固定的协作框架下进行，难以泛化到不同的调度与验证策略；任务长度受限于 GAIA 任务，未覆盖长周期工作流；对源头归因与因果分析尚未深入，且模型选择对结果影响显著。

---

## 338. Statistical Channel Fingerprint Construction for Massive MIMO: A Unified Tensor Learning Framework

**arXiv ID:** 2604.27574 | [PDF](https://arxiv.org/pdf/2604.27574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 339. Social Media Data Toolkit: Standardization and Anonymization of Social Network Datasets

**arXiv ID:** 2604.27710 | [PDF](https://arxiv.org/pdf/2604.27710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 340. SBN Explorer: An Empirical Study of Cryptographic Boolean Networks

**arXiv ID:** 2604.27560 | [PDF](https://arxiv.org/pdf/2604.27560v1)

**作者:** Arnaud Valence `[一作]` `[通讯]`, Arnaud Valence

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

系统性探索了由六个结构约束（S、A、R、I、H、L）生成的 64 种布尔电路架构类别，并通过遗传搜索评估它们在差分、线性和代数抵抗上的性能。

**💡 创新点**

创新点在于将布尔网络的设计空间建模为一个 2⁶ 维超立方体，揭示了结构约束之间的互易（epistasis）效应，并证明 Regularity（R）是所有三种加密分析目标的唯一通用驱动因素，挑战了传统 SPN/Feistel 的假设。

**🔧 技术方法**

使用了遗传算法、Formal Concept Analysis、Walsh–Hadamard 方差分解、LASSO 回归以及统计检验等技术手段，对各类架构的加密性能进行量化与结构关联分析。

**📊 数据集**

采用自生成的 16 位布尔网络数据集，覆盖所有 64 种约束组合，无外部真实数据集。

**📈 对比分析**

通过比较差分、线性和代数三项指标的得分，发现某些架构在三种攻击面上均达到接近最优的性能，显著优于传统 SPN 或 Feistel 结构；同时对比结果表明，R 约束在所有评估中占据主导位置。

**⚠️ 局限性**

局限性包括仅在 16 位小规模网络上验证，单轮加密评估；计算成本高，难以扩展到更大规模或多目标 Pareto 最优分析，且实验结果可能不完全适用于实际加密算法的实现与部署。

---

## 341. Mind the Gap: Structure-Aware Consistency in Preference Learning

**arXiv ID:** 2604.27733 | [PDF](https://arxiv.org/pdf/2604.27733v1)

**作者:** Mehryar Mohri `[一作]` (Google Research), Yutao Zhong `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型（LLM）的偏好学习中，论文证明传统的无约束对数损失在有限容量的神经网络上不可一致，进而提出通过在损失中加入分数间隔（margin‑shifted surrogate）来实现一致性，并进一步引入结构感知（semantic‑aware）margin，得到结构感知直接偏好优化目标（SA‑DPO）

**💡 创新点**

核心创新点包括：① 从理论层面证明无约束 surrogate 在等连续假设集上导致 vacuous consistency；② 提出 margin‑shifted surrogate 并证明其 𝐻‑一致性；③ 引入结构感知一致性和 SA‑DPO，动态根据答案语义距离调整 margin；④ 通过 Margin‑Capacity Profile 分析不同尾部重（heavy‑tailed）损失函数在容量受限模型上的一致性阶梯，证明多项式 hinge（如 Cubic Hinge）优于指数型损失（如 Logistic）

**🔧 技术方法**

技术手段包括：凸非增对数/hinge 等 surrogate、margin‑shifted 损失、结构感知 margin、Bregman 正则化 RLHF、LoRA 微调、Unsloth、TRL、LLama‑3‑8B 等基础设施

**📊 数据集**

实验数据集：自制同义词压力测试（100 个近义词对），Anthropic HH‑RLHF，UltraFeedback Benchmark

**📈 对比分析**

方法对比：与标准 DPO、SimPO、IPO、Poly‑3（Cubic Hinge）等；同义词测试中 SA‑DPO 收敛快、损失低；容量受限实验中 Poly‑3 在 𝛾=1 的情况下实现 100% 准确率，显著优于 DPO（71.8%）和 IPO（94.7%）；UltraFeedback 上 SA‑DPO 在 Distinct、Ambiguous 和 Hard Subset 上分别提升至 0.790、0.734、0.700，明显优于 DPO、SimPO

**⚠️ 局限性**

局限性：需要手动调节 margin 与结构感知参数，理论假设在实际连续网络中严格 margin 难以满足；未覆盖多选排序或非传递性偏好；对大规模 RLHF 训练的鲁棒性和泛化性尚待进一步验证

---

## 342. AgentEconomist: An End-to-end Agentic System Translating Economic Intuitions into Executable Computational Experiments

**arXiv ID:** 2604.27725 | [PDF](https://arxiv.org/pdf/2604.27725v1)

**作者:** Jiaju Chen `[一作]` (Zhongguancun Academy), Yong Li `[通讯]` (Zhongguancun Academy)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个名为AgentEconomist的端到端交互式经济研究协作系统，可将经济直觉转化为可执行的计算实验。

**💡 创新点**

将文献检索、假设生成、实验设计与模拟执行模块化衔接，并通过结构化记忆与MCP工具箱实现可追溯、可执行的科研工作流。

**🔧 技术方法**

采用检索增强生成（RAG）+大语言模型、多代理架构、结构化记忆、MCP工具箱、AgentEconomy基于LLM混合行为的代理经济模拟器。

**📊 数据集**

使用约13,000篇顶级经济学期刊论文构成的知识库以及AgentEconomy中的PSID等微观数据。

**📈 对比分析**

与GPT‑5.2/Gemini‑3‑Pro等通用LLM基线进行配对评估，通过人类与LLM评审的八维度量表，AgentEconomist在文献基础与创新性上显著优于基线。

**⚠️ 局限性**

评估仅限模拟实验，用户样本有限，系统效果依赖知识库与模拟器覆盖范围，长周期一致性与执行效率仍受限。

---

## 343. Line Segment Clipping using Quadrilateral Concavity and Convexity

**arXiv ID:** 2604.27701 | [PDF](https://arxiv.org/pdf/2604.27701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 344. Deep Learning-Based Segmentation of Peritoneal Cancer Index Regions from CT Imaging

**arXiv ID:** 2604.27697 | [PDF](https://arxiv.org/pdf/2604.27697v1)

**作者:** Pieter C. Gort `[一作]` (Eindhoven University of Technology), Fons van der Sommen `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 3234 | [OpenAlex ID](https://openalex.org/A5075677746)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究利用深度学习自动分割CT图像中的13个放射学PCI区域，以实现无创腹膜转移评估。

**💡 创新点**

首次结合专家共识的3D腹膜区划，实现自动化rPCI区域分割，并与人工标注对比展示接近人类一致性。

**🔧 技术方法**

采用nnU-Net与Swin UNETR两种3D语义分割网络，并进行五折交叉验证。

**📊 数据集**

使用62例来自Catharina医院的腹部对比增强CT扫描，包含0–39范围的sPCI病例。

**📈 对比分析**

nnU-Net平均Dice 0.82，接近人类0.88；相较于Swin UNETR（Dice 0.76），性能提升约8%；同时评估了HD95和ASD等边界指标。

**⚠️ 局限性**

受限于单中心数据、少量样本、以及小肠区域（9–12）的解剖变异导致分割误差；缺乏多中心外部验证。

---

## 345. When Agents Evolve, Institutions Follow

**arXiv ID:** 2604.27691 | [PDF](https://arxiv.org/pdf/2604.27691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 346. MSR:Hybrid Field Modeling for CT-MRI Rigid-Deformable Registration of the Cervical Spine with an Annotated Dataset

**arXiv ID:** 2604.27654 | [PDF](https://arxiv.org/pdf/2604.27654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 347. Libra: Accelerating Socket I/O via Programmable Selective Data Copying

**arXiv ID:** 2604.27686 | [PDF](https://arxiv.org/pdf/2604.27686v1)

**作者:** Kairui Zhou `[一作]` (Shanghai Jiao Tong University), Shizhen Zhao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3145 | [OpenAlex ID](https://openalex.org/A5089557280)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了 Libra，一个在内核层面实现选择性拷贝的框架，能够仅把 Layer‑7 协议的元数据拷贝到用户空间，而将大块数据保持在内核缓冲区，实现几乎零拷贝的 Socket I/O；

**💡 创新点**

证明了在主流操作系统中，零拷贝与完整 POSIX 兼容性是互斥的；引入 eBPF 可编程的状态机与虚拟负载标识（VPI）机制，使得在不修改应用的情况下完成选择性拷贝与内核级数据重用；整合 kTLS，进一步提升加密流的性能；

**🔧 技术方法**

利用 eBPF 在接收/发送路径注入协议感知逻辑、状态机驱动的选择性拷贝、VPI 映射与查找、零拷贝指针转移；与 Linux 6.11 内核内置 API 结合；使用 eBPF 哈希表维护映射；

**📊 数据集**

在真实的 L7 代理工作负载上进行评测，使用未改造的 Nginx 与 HAProxy，生成 1 KB–1024 KB 的 HTTP 请求/响应；通过 64 并发连接模拟云原生场景；

**📈 对比分析**

与标准 Linux 网络栈、DPDK 基础的 F‑Stack 以及 Copier 进行基准比较；测量吞吐量、P99 延迟、CPU 占用、dTLB 损失率等；在明文情况下，Libra 最大可提升 4.2× 吞吐量，P99 延迟下降 90%+；在 kTLS 硬件加速下，吞吐量提升 2×，P99 延迟下降 65%；

**⚠️ 局限性**

在软件 kTLS 模式下性能低于标准栈；仅支持串行协议，对 HTTP/2 等多路复用协议的适配仍需扩展；对极小负载时拷贝开销占比高；依赖 Linux 6.11 及 eBPF，尚未在其它内核或平台验证；

---

## 348. LZn : Robust LoRa Frame Synchronization Under Frame Collisions and Ultra-Low SNR Conditions

**arXiv ID:** 2604.27672 | [PDF](https://arxiv.org/pdf/2604.27672v1)

**作者:** José Álamos `[一作]` (HAW Hamburg), Matthias Wählisch `[通讯]` (TU Dresden)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种低复杂度的LoRa帧同步方法，能够在多帧冲突和极低信噪比条件下实现鲁棒同步并提升帧解码率。

**💡 创新点**

创新点在于引入谱交叉（spectral intersection）操作结合分辨率网格搜索、模板匹配和异常检测，实现多帧重叠时的自适应同步；同时提供理论分析与实际测量验证。

**🔧 技术方法**

核心技术包括：非相干解调、FFT 与 Zoom‑FFT、Goertzel 算法、最小值交叉滤波、峰值检测（Modified Z‑score）、模板交叉相关、精细的 STO/CFO 估计与校正。

**📊 数据集**

使用三组真实 LoRa 捕获数据集（CIC、TnB、WM4/WM5），以及仿真场景（不同 SF、SNR、碰撞率）。

**📈 对比分析**

与四个基线（OpenLoRa、TnB、CIC、Pyramid）比较，检测灵敏度提升高达10 dB，帧检测率比最佳碰撞容忍方案高约1.5×；在单用户极低 SNR 下解码率提升3.5倍；在高碰撞率下吞吐量提升约1.2×。

**⚠️ 局限性**

局限包括：对极端时域接近冲突的帧（offset≈0或±1符号）仍存在检测下降；精细同步仍需一定计算量（4 FFT/窗口，6 FFT/假设），在极高速率场景下可能超出实时约束；算法主要针对同 SF 的冲突，跨 SF 冲突处理尚未深入。

---

## 349. Linear-Core Surrogates: Smooth Loss Functions with Linear Rates for Classification and Structured Prediction

**arXiv ID:** 2604.27742 | [PDF](https://arxiv.org/pdf/2604.27742v1)

**作者:** Mehryar Mohri `[一作]` (Google Research), Yutao Zhong `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Linear-Core Surrogates（线性核心替代损失），一种在保持光滑可微的同时实现线性一致性（即 O(1) 统计收敛率）的凸损失；并在结构化预测中利用该损失构建了无偏的随机梯度估计器，彻底消除了传统方法中 O(|Y|²) 的推理瓶颈。

**💡 创新点**

创新点：
- 将线性核心与光滑尾部拼接，兼具梯度非退化与光滑性；
- 证明了该损失在二分类、多分类和结构化预测中的线性 H‑一致性；
- 设计了基于随机采样的梯度算法，使结构化损失的期望梯度可在 O(1) 时间内估计；
- 提供了梯度方差上界与复杂度分析，说明与输出空间大小无关。

**🔧 技术方法**

技术手段：凸分析与光滑性证明、线性一致性框架、随机采样梯度估计、结构化损失的分解、方差与收敛率分析、实验中使用 SGD 与自定义噪声模型。

**📊 数据集**

数据集与实验：
- CIFAR‑10（含 20%–60% 的实例依赖噪声）；
- 大词表序列标注（|Y|=400）与人工扩张到 4000 的 Penn Treebank POS 标注；
- 其它标准多分类/结构化基准（未列明）。

**📈 对比分析**

比较方法与性能：
- 与 Structured SVM（SSVM）比较，Linear‑Core 在 |Y|=400 时实现 23× 的训练速度提升；
- 与 CRF（BILSTM‑CRF）比较，使用随机采样获得 17.4× 的时效性提升；
- 与 Cross‑Entropy（CE）与 Generalized CE（GCE）在噪声 CIFAR‑10 上，Linear‑Core 在 30%–40% 噪声时提升约 2.6% 以上；
- 在大词表序列标注任务中实现了显著的时间/能耗优势。

**⚠️ 局限性**

局限性：
- 需要选择合适的基函数 Φ（如满足 Φ''(0)=0），在某些光滑度要求下可能受限；
- 随机采样梯度对采样策略依赖，过度采样会增加训练成本；
- 对于非和式或非对称结构化损失的适用性尚未验证；
- 由于构造是基于对偶/凸框架，可能在极高维或极稀疏场景下需要进一步优化。

---

## 350. FUN: A Focal U-Net Combining Reconstruction and Object Detection for Snapshot Spectral Imaging

**arXiv ID:** 2604.27653 | [PDF](https://arxiv.org/pdf/2604.27653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 351. Optimized Deferral for Imbalanced Settings

**arXiv ID:** 2604.27723 | [PDF](https://arxiv.org/pdf/2604.27723v1)

**作者:** Corinna Cortes `[一作]` (Google Research), Yutao Zhong `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的两阶段学习到推延算法，针对专家分布不平衡问题进行理论分析与实践改进。

**💡 创新点**

创新点在于将推延损失转化为输入-专家域的成本敏感多类分类问题，设计了边际损失与一致性保证，并推出 Margin-based Imbalanced Learning to Defer。

**🔧 技术方法**

主要技术包括边际损失、Rademacher 复杂度分析、成本敏感学习、对数/指数损失的自适应调参以及交叉验证选择超参数。

**📊 数据集**

使用的评估数据集包括 CIFAR‑10/100、SVHN、Tiny ImageNet 以及 MMLU 大语言模型路由任务（Qwen 2.5 系列）和合成专家环境。

**📈 对比分析**

与现有 baseline（两阶段推延）在多种专家不平衡设置下对比，实验表明新算法在推延损失、专家使用率和整体性能上均优于 baseline，尤其在 LLM 路由任务中更接近最优分配。

**⚠️ 局限性**

局限性在于仍缺乏对极端不平衡情况的系统评估、真实专家的全面验证以及对模型训练成本和复杂度的深入分析。

---

## 352. The TEA Nets framework combines AI and cognitive network science to model targets, events and actors in text

**arXiv ID:** 2604.27673 | [PDF](https://arxiv.org/pdf/2604.27673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 353. Fairness for distribution network operations and planning

**arXiv ID:** 2604.27669 | [PDF](https://arxiv.org/pdf/2604.27669v1)

**作者:** Pedro F. C. de Carvalho `[一作]` (Ku Leuven), Dirk Van Hertem `[通讯]` (Energyville)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对配电网络中公平性概念、利益相关方、价格公平（PoF）以及公平度量指标进行了系统综述和分类，并探讨了这些指标在优化模型中的数学实现和计算可行性。

**💡 创新点**

创新点在于提供了统一的公平性框架，将公平性定义、指标、利益相关方和优化实现进行整合，明确了效率-公平权衡并引入PoF概念，便于未来研究和实际决策。

**🔧 技术方法**

主要采用文献综述、理论分析、指标比较和数学建模技术（线性化、凸优化、混合整数规划等）。

**📊 数据集**

由于是综述性工作，没有使用特定数据集；若需验证，需结合配电网案例或仿真平台。

**📈 对比分析**

本文未进行实验比较；但对各公平指标的数学性质、可实现性和计算复杂度进行了理论比较，指出不同指标在实际优化中的适用性。

**⚠️ 局限性**

局限性包括缺乏实证数据验证、对PoF的定量评估有限、对不同地区配电网拓扑的普适性讨论不足，以及对公平性指标选择标准的进一步细化需求。

---

## 354. Can Tabular Foundation Models Guide Exploration in Robot Policy Learning?

**arXiv ID:** 2604.27667 | [PDF](https://arxiv.org/pdf/2604.27667v1)

**作者:** Buqing Ou `[一作]` (Carnegie Mellon University), Frederike Dümbgen `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 TFM‑S3 框架，将梯度更新与在动态低维子空间中的全局搜索交错进行，利用预训练的表格基础模型对候选策略进行多步迭代筛选，从而在保持固定 Rollout 预算的前提下提升机器人连续控制的样本效率。

**💡 创新点**

创新点包括：① 通过最近梯度的 SVD 动态构造低维子空间，使搜索方向与当前训练动态同步；② 在子空间内使用预训练的 Tabular Foundation Model（TabPFN）作为 surrogate 进行多步迭代筛选，显著降低实际 Rollout 需求；③ 将本地梯度更新与全局子空间搜索模块无缝耦合，兼顾收敛速度与最终性能。

**🔧 技术方法**

主要技术：actor‑critic 梯度更新（以 TD3 为例），SVD 降维构造子空间，Gaussian 采样与信任域限制，Tabular Foundation Model 作为 surrogate，迭代内部精炼策略，动态子空间重构。

**📊 数据集**

实验数据集：MuJoCo 连续控制基准任务，包括 HalfCheetah‑v5、Ant‑v5、Humanoid‑v5。

**📈 对比分析**

对比方法：随机搜索（Random Search TD3）、一次性筛选（TFM‑S3‑TD3 One Shot）、原始 TD3；在相同的 1M 环境步 Rollout 预算下，TFM‑S3 在早期收敛速度最快、最终平均奖励最高，并且在不同随机种子下表现更稳定。

**⚠️ 局限性**

局限性：① 需要预训练的 TFMs，且子空间维度固定，缺乏自适应选择机制；② 仅在模拟环境验证，真实机器人环境的鲁棒性待进一步评估；③ 依赖子空间降维，可能限制对极高维参数空间的探索范围；④ 迭代筛选过程仍涉及多次 surrogate 重新训练，对计算资源有一定需求。

---

## 355. Back to the Future: Rethinking Endorsement in Order-Execute Blockchains

**arXiv ID:** 2604.27659 | [PDF](https://arxiv.org/pdf/2604.27659v1)

**作者:** Rongji Huang `[一作]` (Shanghai Jiao Tong University), Shengyun Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5101540296)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在权限链上实现可定制的背书（flexible endorsement），并将背书嵌入传统的顺序-执行-背书（order‑execute‑endorse）框架中，提出一种基于Tendermint的BFT协议（FlexTend），在保持共识不变的前提下支持灵活背书。

**💡 创新点**

核心创新是把背书信息直接嵌入Tendermint的三阶段消息流程，实现“背书+共识”在同一次通信轮中完成；同时提供快速去背书（rapid removal）机制，允许节点在发现背书不足时即时标记并删除事务，避免无谓的重执行。

**🔧 技术方法**

技术手段包括：Tendermint改造（添加endorse、remove消息）、事务批量执行（DAG调度）、快速去背书与重执行优化、Gossip式传播、主节点轮转以及基于投票的事务验证。

**📊 数据集**

实验数据集为从以太坊区块链抓取的USDT（Tether USD）转账交易（约121万笔），覆盖23,700,767~23,722,235区块高度。

**📈 对比分析**

与基于EOV（Fabric‑style）仿真版本对比，FlexTend在相同硬件（EC2 c5.2xlarge）下吞吐量提升至10.6倍；在高冲突负载下通过快速去背书和重执行优化维持稳定吞吐；在WAN环境下延迟依赖于背书节点分布，单节点背书导致延迟显著升高。

**⚠️ 局限性**

主要局限：①每笔事务位于所有后续事务的关键路径，导致高冲突场景下大量重执行；②需要同步网络假设；③背书节点的单点或故障会引发主节点轮转导致停顿；④恶意背书者与客户端合谋可能插入事务并一次性去背书；⑤对事务去背书的快速机制仍可能因过度去背而影响 liveness。

---

## 356. Connected Dependability Cage: Run-Time Function and Anomaly Monitoring for the Development and Operation of Safe Automated Vehicles

**arXiv ID:** 2604.27728 | [PDF](https://arxiv.org/pdf/2604.27728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 357. LLM-as-a-Judge for Human-AI Co-Creation: A Reliability-Aware Evaluation Framework for Coding

**arXiv ID:** 2604.27727 | [PDF](https://arxiv.org/pdf/2604.27727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 358. How Code Representation Shapes False-Positive Dynamics in Cross-Language LLM Vulnerability Detection

**arXiv ID:** 2604.27714 | [PDF](https://arxiv.org/pdf/2604.27714v1)

**作者:** Maofei Chen `[一作]` (China Telecom Research Institute), Dongxin Liu `[通讯]` (China Telecom Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统探究了跨语言LLM漏洞检测中训练强度与代码表示（原始文本vs AST）对误报率的联合影响，并提出了一种无训练的跨表示一致性门控方案。

**💡 创新点**

创新点在于：①首次从训练-推理两侧同时变化的维度对误报率进行分解；②通过跨表示探测（文本训练+AST输入）直接证实表面线索记忆机制；③验证此机制在不同模型族与目标语言（Java、Python）上的一致性。

**🔧 技术方法**

采用了LoRA参数高效微调、Qwen3‑8B 与 Llama 3.1‑8B‑Instruct 两大模型、Semgrep生成的 pruned AST 以及统一的系统提示；在推理时切换文本与 AST 两种输入格式。

**📊 数据集**

训练数据来自 NIST Juliet C/C++ 语料（pilot ~3k 以及 full ~70k）；评测数据为 OWASP Benchmark v1.2（Java）和 BenchmarkPython v0.1（Python）。

**📈 对比分析**

通过在同一模型下比较零样本、文本微调、AST微调以及跨表示探测的 FPR/Recall/F1，发现文本微调显著提升 FPR（从 0.76→1.0），AST 微调虽提升 F1 但 FPR 仍 ≈1；跨表示探测可将 FPR 降至 0.58（Qwen）或 0.90（Llama），但召回下降；跨语言实验显示 2.9pp 的误报差异，表明误报主因为表面线索而非目标语言差异。

**⚠️ 局限性**

局限包括仅处理单函数级别代码、仅评估 8B 规模模型、仅使用两种语言与两套基准、固定解码策略与提示模板、Python 仅验证 Qwen，且未探究更丰富的结构表示（如 PDG/CPG）或大规模跨域适配。

---

## 359. Knowledge Graph Representations for LLM-Based Policy Compliance Reasoning

**arXiv ID:** 2604.27713 | [PDF](https://arxiv.org/pdf/2604.27713v1)

**作者:** Wilder Baldwin `[一作]` (University of Maine), Sepideh Ghanavati `[通讯]` (University of Maine)

**通讯引用:** 1312 | [OpenAlex ID](https://openalex.org/A5072117004)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建基于 AI 治理文件的知识图谱，并将其与大型语言模型（LLM）结合，提供基于图的检索和推理，用以回答关于 AI 风险合规的问题。

**💡 创新点**

提出了一个端到端的 agentic 框架，包括分块、提取、检索和答案合成四个阶段；对比了闭合的 AIRO 标准化本体与开放式自发本体两种知识图谱结构；验证了知识图谱可显著提升 LLM 在合规问答中的准确性。

**🔧 技术方法**

使用 LLM 进行文本分块、实体与关系抽取；利用图搜索与 ReAct‑式图遍历实现检索；采用多种 LLM（gpt‑5‑mini、gpt‑4.1‑mini、gpt‑oss:20b、nemotron:30b、granite4:micro）与 LLM‑as‑Judge 评估；实现 Model‑Context‑Protocol（MCP）服务与工具调用。

**📊 数据集**

三份 AI 治理文件：欧盟 AI 法案（EU AI Act）、NIST AI 风险管理框架（NIST AI RMF）和 OWASP LLM Top 10；以及构造的 42 道覆盖六种推理类型（T1–T6）的问答数据集。

**📈 对比分析**

通过对比三种检索条件（无上下文、全图序列化、Agent 化遍历）和两种本体（AIRO 与开放式），在五个模型上对 42 道题进行 5 轮实验，使用启发式评分与 LLM‑as‑Judge。结果显示，知识图谱能提升所有模型的评估分数，最显著提升在实体检索与属性查询，LLM‑as‑Judge 提升可达 0.55 分；开放式本体在多数模型上与 AIRO 相当甚至更好。

**⚠️ 局限性**

样本量有限（仅 42 题），只覆盖三份英文西方框架，可能影响泛化；启发式评分对词汇差异敏感；小模型对 Agent 路径不敏感，导致检索效果差；未对 KG 抽取质量波动做进一步评估；缺乏真实开发者使用评估。

---

## 360. Understanding Bugs in Template Engine-Based Applications: Symptoms, Root Causes, and Fix Patterns

**arXiv ID:** 2604.27692 | [PDF](https://arxiv.org/pdf/2604.27692v1)

**作者:** Kai Gao `[一作]` (University of Science and Technology Beijing), Chang-ai Sun `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 883 | [OpenAlex ID](https://openalex.org/A5048309076)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了15款模板引擎、5种语言共1,004个应用级错误，构建了症状、根因和修复模式三大分类法，并基于此开发了两款Jinja模板工具。

**💡 创新点**

创新点在于首次以大规模经验研究方式对模板引擎应用错误进行细粒度建模，揭示了异常渲染结果、语法误用和数据上下文不匹配三类主要根因，并提出针对性工具支持。

**🔧 技术方法**

采用手工标注的开放编码与系统性统计分析技术，并结合规则检测、AST分析与可视化Sankey图表实现症状-根因-修复模式的关联挖掘。

**📊 数据集**

数据集来源于Stack Overflow的1,004条高质量问题以及GitHub 180条验证性错误，涵盖了多语言、多框架和多目标（HTML、YAML、SQL）场景。

**📈 对比分析**

通过与GitHub数据交叉验证、Kruksal‑Wallis、Mann‑Whitney U检验对比，发现症状与根因分布无显著差异，修复模式在两平台差异仅为无效效应；工具检测率在30例手工与20例合成案例上达到100%。

**⚠️ 局限性**

局限性包括标注工作依赖人工主观判断、数据仅覆盖公开社区与公开仓库、工具实现仍为原型，缺乏在真实大规模项目中的部署与性能评估。

---

## 361. PuzzleMark: Implicit Jigsaw Learning for Robust Code Dataset Watermarking in Neural Code Completion Models

**arXiv ID:** 2604.27677 | [PDF](https://arxiv.org/pdf/2604.27677v1)

**作者:** Haocheng Huang `[一作]` (Soochow University), Xiaofang Zhang `[通讯]` (Soochow University)

**通讯引用:** 5702 | [OpenAlex ID](https://openalex.org/A5115603628)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 PuzzleMark，一种利用隐式拼图学习的代码数据集水印方法，用变量名拼接实现隐蔽可验证水印。

**💡 创新点**

创新点在于①引入代码复杂度评估并筛选水印载体，降低暴露风险；②用拼接模式取代传统共现模式，提供更强鲁棒性；③实现固定触发和通用触发两种自适应水印策略。

**🔧 技术方法**

技术包括：基于 AST 的变量重命名、拼接模式嵌入、代码复杂度特征投影、Fisher 精确检验验证、以及自动化载体筛选流程。

**📊 数据集**

使用 CodeSearchNet 的 Python 与 Java 子集，共计约 86 万代码片段作为实验数据集。

**📈 对比分析**

与 CoProtector、CodeMark 等基线对比，PuzzleMark 在三种 NCCM 上实现 100% 验证成功率、0% 误报率，BLEU 影响微乎其微；人类检测 ASR ≤ 0.18，机器检测 DeCoMa、KillBadCode 召回率仅 ~30%；在删除与稀释攻击下均保持可验证性。

**⚠️ 局限性**

局限性包括：对较小项目（如 <100 代码块）水印率不足导致验证失败；通用触发需更高水印率；不适用于非代码或自然语言任务；载体筛选增加 30 分钟前处理成本。

---

## 362. VOW: Verifiable and Oblivious Watermark Detection for Large Language Models

**arXiv ID:** 2604.27666 | [PDF](https://arxiv.org/pdf/2604.27666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 363. From Context to Skills: Can Language Models Learn from Context Skillfully?

**arXiv ID:** 2604.27660 | [PDF](https://arxiv.org/pdf/2604.27660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 364. Language Ideologies in a Multilingual Society: An LLM-based Analysis of Luxembourgish News Comments

**arXiv ID:** 2604.27661 | [PDF](https://arxiv.org/pdf/2604.27661v1)

**作者:** Emilia Milano `[一作]`, Christoph Purschke `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）对卢森堡语评论进行语言意识形态的检测与分类，探究是否需翻译成高资源语言以提升性能。

**💡 创新点**

提出了使用LLM生成解释的评价方法，并系统比较不同翻译和语言对模型性能的影响，证明在卢森堡语上不需要翻译即可取得相当的二分类效果。

**🔧 技术方法**

使用了GPT‑4o、GPT‑4o‑mini、o3以及开源LLM（DeepSeek、GPT‑5、Qwen、Magistral等），并结合Google Translate进行自动翻译与人工校正。

**📊 数据集**

构建了300条卢森堡语评论（共1524句）并人工标注为五类语言意识形态（身份、活力、归属、责任、认可）及无意识形态，形成二分类与多分类数据集。

**📈 对比分析**

对比不同LLM、不同提示（Prompt 1‑4）以及不同翻译语言的表现，发现Prompt 4在所有模型中最优，GPT‑5在所有语言中表现最佳，平均加权F1在0.55左右；二分类（有无意识形态）F1可达0.93；多分类各类别F1低于0.6，表明细粒度分类较难。

**⚠️ 局限性**

主要局限包括：小样本规模导致模型难以微调；LLM对细粒度意识形态分类仍易混淆；翻译对文化含义转移影响大；模型生成的解释不一定完全可靠，需人工审校。

---

## 365. When Does Structure Matter in Continual Learning? Dimensionality Controls When Modularity Shapes Representational Geometry

**arXiv ID:** 2604.27656 | [PDF](https://arxiv.org/pdf/2604.27656v1)

**作者:** Kathrin Korte `[一作]` (IT University of Copenhagen), Sebastian Risi `[通讯]` (IT University of Copenhagen)

**通讯引用:** 3682 | [OpenAlex ID](https://openalex.org/A5020511097)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在连续学习的A1→B→A2序列任务中比较了任务分区的模块化递归网络与单一网络的表现，研究任务相似度与表示维度对学习效果的共同作用。

**💡 创新点**

发现表示维度是决定模块化结构是否能有效缓解干扰的关键因素：在高维（懒惰）学习 regime 下两种架构几乎同等；在低维（丰富）学习 regime 下模块化能根据任务相似度实现分层的子空间组织，显著降低干扰并保持较低的转移成本。

**🔧 技术方法**

采用随机初始化权重缩放 γ 控制有效维度；使用递归网络（RNN）实现两种架构；通过 PCA 计算有效维度、主角度和 3D PCA 可视化子空间几何；评估准确率、转移（冬季试验）和干扰（A2 阶段）。

**📊 数据集**

利用 Holton 等人设计的合成植物线索-圆盘定位任务（六个离散线索，夏/冬两个季节），在不同任务相似度（相同、近似、遥远）以及不同 γ 水平下训练。

**📈 对比分析**

比较方法：在每个 γ 和相似度条件下记录准确率、转移和干扰；用 PCA 指标量化表示维度；用主角度衡量子空间对齐；可视化 3D PCA 轨迹。结果显示：模块化网络在低维 regime 下准确率稳定，转移低且干扰小；单一网络在相似度差异和低 γ 时表现更差。

**⚠️ 局限性**

局限性：① PCA 的有效维度估计是近似的，可能未完全捕捉内在维度；② γ 同时影响维度和优化动态，难以单独解释其作用；③ 仅测试了 A–B–A 三阶段任务，未探究更长序列或任务相似度变化；④ 共享读出限制了模块化程度，未检验更完全隔离或互连的模块化方案。

---

## 366. Differential Subgroup Discovery: Characterizing Where Two Populations Differ, and Why

**arXiv ID:** 2604.27741 | [PDF](https://arxiv.org/pdf/2604.27741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 367. Solving Hypergraph Laplacian Systems in Almost-Linear Time

**arXiv ID:** 2604.27651 | [PDF](https://arxiv.org/pdf/2604.27651v1)

**作者:** Yuichi Yoshida `[一作]` (National Institute of Informatics), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**通讯引用:** 7095 | [OpenAlex ID](https://openalex.org/A5038701345)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对基于切割（Lovász扩展）的超图拉普拉斯算子，提出一种随机化的近线性时间（P^{1+o(1)}）求解 Poisson 问题和其正则化版本（近似求解超图 Laplacian 系统）的算法，并给出明确的原始（primal）点和对偶（dual）证书。

**💡 创新点**

创新点包括：
- 将 Fenchel 对偶问题等价转化为在 O(P) 条弧的辅助图上的凸流问题；
- 引入恢复定理，仅通过每条超边的一个非负标量即可从对偶流恢复原始势能；
- 通过可测量的“边质量”向量 μ 与“支持查询”相结合，将非线性恢复问题简化为一次线性最小成本流求解；
- 设计了完整的有限精度修复方案，保证输出为有理数且满足精度和可证的约束。

**🔧 技术方法**

使用的主要技术包括：
- Fenchel 对偶与支持函数表述；
- 构造辅助图的“升维”流模型（transport 以及 quadratic arcs）；
- Chen 等人关于凸流的黑盒求解框架（自共轭障、内部点方法）；
- 对偶流到“边质量”集合 ℳ(s) 的投影与支持查询的等价性；
- 最小成本流的残差潜在提取与容量上限的可行性；
- 误差分析与 Bregman 距离的解释。

**📊 数据集**

该工作为理论算法，未在具体数据集上进行实验；所有分析均在通用加权超图模型（P 为超边大小总和）下进行。

**📈 对比分析**

与以往多项式时间求解（例如 Fujii、Soma、Yoshida 的 O(M^4 log M) 方法）相比，所给算法的时间复杂度为 P^{1+o(1)}，仅以超边大小总和为主，几乎线性；同时提供了原始与对偶的可验证近似最优解，而之前的多项式算法缺乏这种可证性。

**⚠️ 局限性**

局限性：
- 需要超图连通、权重与需求满足多项式界限且为 dyadic；
- 算法本身为随机化，高概率成功；
- 需要多次高精度查询与精度控制，实际实现复杂；
- 结果为近似（加性误差 exp(−log^C P)），不保证严格最优；
- 对于非常大规模超图，常数因子与高精度运算仍可能影响实际性能。

---

## 368. Position-Aware Drafting for Inference Acceleration in LLM-Based Generative List-Wise Recommendation

**arXiv ID:** 2604.27747 | [PDF](https://arxiv.org/pdf/2604.27747v1)

**作者:** Jiaju Chen `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43763 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 PAD-Rec，一种在 LLM 生成式列表推荐中使用的位置信息感知草稿模块，用于加速推理。

**💡 创新点**

首次将项目内槽位位置嵌入与草稿步骤位置嵌入融合到草稿模型中，并通过轻量门控实现结构与深度感知，从而显著提升采样匹配率与吞吐量。

**🔧 技术方法**

使用了位置信息嵌入、门控机制、HASS 风格多步推理训练、树状草稿验证、Llama-3.2 系列 LLM 作为目标模型，以及精细的梯度混合精度训练。

**📊 数据集**

在 Amazon Reviews（Beauty、Instruments、Games）和 Yelp 四个真实推荐数据集上进行实验。

**📈 对比分析**

与 EAGLE‑2、HASS、FSPAD、GRIFFIN 等主流 SD 基线进行同基准对比，评估 wall‑clock 加速、平均接受长度、Recall@10 和 NDCG@10；在所有数据集上 PAD‑Rec 最高可达 3.1× 加速，且推荐质量基本保持不变。

**⚠️ 局限性**

仅针对结构化语义‑ID 令牌的列表生成，需在训练时对 B 深度预设，且在高温度或更大模型规模下加速提升有限；对非 LLM 生成式推荐或不规则输出结构的适应性待验证。

---

## 369. Why Self-Supervised Encoders Want to Be Normal

**arXiv ID:** 2604.27743 | [PDF](https://arxiv.org/pdf/2604.27743v1)

**作者:** Yuval Domb `[一作]` `[通讯]`, Yuval Domb

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于信息瓶颈的几何信息理论框架，将最优表示解释为预测流形的软聚类，并给出对应的无变分编码器损失。

**💡 创新点**

创新点包括：1) 将信息瓶颈与率失真等价性结合，得到软聚类解；2) 通过 Dirichlet→指数→高斯变换链解释 SIGReg 的理论依据；3) 提出无变分的非参数 Conditional Entropy Bottleneck（CEB）损失，适用于监督、半监督和自监督场景。

**🔧 技术方法**

使用信息瓶颈理论、率失真理论、信息几何、Blahut–Arimoto 算法、Cramér–Wold 定理、Epps–Pulley 正态性检验、Dirichlet/指数/高斯变换链以及非参数 KL 估计与 minibatch 类条件边缘估计。

**📊 数据集**

实验数据集包括连续角度到三分类的 toy 例子、离散 20 值的 toy 例子以及 FashionMNIST（28×28，10 类）。

**📈 对比分析**

通过与 VIB（高斯编码器）对比，CEB 在相同 β 取值下取得更高的分类准确率（≈91.4% 对比 90.7%）且压缩率更低；在不同 simplex 维度 K 的实验中发现 K=|Y| 为最优，性能随 K 降低而逐步下降。

**⚠️ 局限性**

局限性：需要手动设定 β 并估计其临界值；非参数 CEB 在大规模数据上计算开销较大；SIGReg 的相位熵开销仅在理论上可忽略，对实际速率计数有影响；对连续 Y 的维度上界仍基于覆盖数，理论性较强。

---

## 370. Iterative Multimodal Retrieval-Augmented Generation for Medical Question Answering

**arXiv ID:** 2604.27724 | [PDF](https://arxiv.org/pdf/2604.27724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 371. Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining

**arXiv ID:** 2604.27715 | [PDF](https://arxiv.org/pdf/2604.27715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 372. Linguistically Informed Multimodal Fusion for Vietnamese Scene-Text Image Captioning: Dataset, Graph Framework, and Phonological Attention

**arXiv ID:** 2604.27712 | [PDF](https://arxiv.org/pdf/2604.27712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 373. ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control

**arXiv ID:** 2604.27711 | [PDF](https://arxiv.org/pdf/2604.27711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 374. Contextual Agentic Memory is a Memo, Not True Memory

**arXiv ID:** 2604.27707 | [PDF](https://arxiv.org/pdf/2604.27707v1)

**作者:** Binyan Xu `[一作]`, Kehuan Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文件说明了 NeurIPS 2026 会议论文的提交格式、排版规范、以及使用的 LaTeX 样式文件和相关宏包。

**💡 创新点**

创新点在于统一更新为 2026 年版的 NeurIPS LaTeX 样式文件，废弃旧版样式，提供了更明确的排版细节、匿名化处理以及可选的 preprint/camera-ready 模式。

**🔧 技术方法**

使用的技术主要包括 LaTeX 排版、natbib 引用包、booktabs 表格宏包、graphics 包以及对字体嵌入和 PDF 生成的技术规范。

**📊 数据集**

该文件不涉及任何实验数据集；其内容为会议投稿规范指南。

**📈 对比分析**

此文件不包含实验方法或性能评估；比较方法与性能指标未涉及。

**⚠️ 局限性**

局限性：仅适用于 LaTeX，未兼容 Word/RTF；若未严格遵守页面、字体、行距等排版细节，论文可能被拒；此外文件不提供实际研究结果，无法评估学术贡献。

---

## 375. RayFormer: Modeling Inter- and Intra-Ray Similarity for NeRF-Based Video Snapshot Compressive Imaging

**arXiv ID:** 2604.27702 | [PDF](https://arxiv.org/pdf/2604.27702v1)

**作者:** Yubo Dong `[一作]`, Zhenyuan Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了在视频快照压缩成像（SCI）中使用 Patch‑level 射线采样以及 Inter‑和 Intra‑Ray Transformer（RayFormer）来重建高质量的 NeRF 场景；

**💡 创新点**

创新点包括：1）空间连贯的 Patch‑level 射线采样策略，提升局部结构信息；2）双路径 Transformer 关注射线间与射线内的结构相似性；3）结合总变分（TV）先验以抑制伪影；

**🔧 技术方法**

使用技术包括：NeRF、哈希网格编码、球面谐波编码、Transformer 多头自注意力、交叉重建损失以及 TV 正则化；

**📊 数据集**

实验数据集涵盖六个合成场景（Airplants、Hotdog、Cozy2room、Tanabata、Factory、Vendor）以及真实 SCI 采集数据；

**📈 对比分析**

与 GAP‑TV、PnP‑FFDNet、EfficientSCI、SCINeRF 等方法对比，RayFormer 在大多数场景下实现了最高 PSNR（0.5–4 dB 提升）、最高 SSIM 和最低 LPIPS，证明了优越的重建质量；

**⚠️ 局限性**

局限性在于：对极大 Patch 或极高压缩比的场景仍需进一步验证鲁棒性；内存占用在更大 Patch 下仍可能成为瓶颈。

---

## 376. Bridging Values and Behavior: A Hierarchical Framework for Proactive Embodied Agents

**arXiv ID:** 2604.27699 | [PDF](https://arxiv.org/pdf/2604.27699v1)

**作者:** Chunhui Zhang `[一作]` (State Key Laboratory Of General Artificial Intelligence), Wei Wang `[通讯]` (State Key Laboratory Of General Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ValuePlanner，一种层次化神经-符号架构，用来实现具有内在价值驱动的主动自治智能体；

**💡 创新点**

核心创新在于将高层价值推理（LLM生成符号子目标）与低层可执行计划（PDDL规划器）分离，结合生成‑批判循环与闭环调整机制，实现价值优化与可执行性兼顾；

**🔧 技术方法**

利用大型语言模型（LLM）进行价值权衡推理，Fast Downward等经典PDDL规划器进行动作规划，生成‑批判模块以及闭环反馈实现迭代优化；

**📊 数据集**

在TongSim高保真家庭仿真环境中测试，采用10种不同Persona（基于Schwartz价值理论的7维向量）与2种内部状态，共计10个场景；

**📈 对比分析**

与ReAct、Reflexion、Plan-and-Solve、D2A等四种基线相比，ValuePlanner在累计价值、偏好对齐度、动作多样性三项指标上均优于基线，尤其在累计价值提升近50%；

**⚠️ 局限性**

限制主要是依赖预定义的符号领域，无法自学习或扩展新的 affordances，且LLM的hallucination 与 PDDL 约束之间的协同仍需改进。

---

## 377. EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory

**arXiv ID:** 2604.27695 | [PDF](https://arxiv.org/pdf/2604.27695v1)

**作者:** Yuyang Li `[一作]` (Australian National University), Dong Gong `[通讯]` (UNSW Sydney)

**通讯引用:** 4598 | [OpenAlex ID](https://openalex.org/A5101768095)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个闭环检索框架，利用证据缺口诊断驱动查询迭代，并构建三层粗细记忆结构以实现长程对话推理。

**💡 创新点**

创新点在于：① 通过全局证据的三层次(精确/可推理/部分)充分性评估显式识别缺口；② 诊断信息直接驱动查询重写，避免无目标迭代；③ 采用索引-边缘-原始三层结构实现高效多跳扩展与精确检索。

**🔧 技术方法**

使用的技术包括：大语言模型（GPT‑4o / GPT‑4o‑mini）进行实体抽取、充分性评估、查询改写与答案生成；句向量检索（Ada‑002 embeddings）；图结构的边缘扩展；基于规则的时间意图识别与置信度校准；多轮闭环迭代逻辑。

**📊 数据集**

在 LoCoMo 长期对话记忆基准上进行评估，涵盖单跳、多跳、时间推理、开放域和对抗性五类问题，共 1,986 条问答对。

**📈 对比分析**

与单通检索和 MIRIX 多代理基线对比，所提方法在时间推理与多跳类别上显著提升 Judge Accuracy（81.6% 对 73.3%，85.2% 对 65.9%），在整体 G‑EVAL 得分亦位居前列；同时平均推理时延比 MIRIX 快 4.5 倍（9.54s vs 42.71s）。

**⚠️ 局限性**

局限性在于当前记忆构建仅支持离线批处理，缺乏对实时对话流的增量更新能力，实际部署时需要解决在线索引与边缘维护问题。

---

## 378. Online Coloring for Graphs of Large Odd Girth

**arXiv ID:** 2604.27690 | [PDF](https://arxiv.org/pdf/2604.27690v1)

**作者:** Hirotaka Yoneda `[一作]` (University of Tokyo), Masataka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的确定性在线图着色算法，针对具有大奇数环长（odd girth）的图，能够用极少数量的颜色完成着色。

**💡 创新点**

创新点在于：①利用多层递归与在线组着色（online group coloring）技术，将传统的平方根色数上界逐步突破到任意小正指数；②首次将“第二邻域”甚至更高阶邻域的结构引入在线着色，实现了对大奇数环图的高效着色；③给出了关于奇数环长与颜色数之间的紧密上界，证明了对于足够大的奇数环长可以实现O(n^ε)色。

**🔧 技术方法**

主要技术包括：多层递归的子问题化简（(r*,d*)-subroutine）；在线组着色框架，保证在邻图度数受限的情况下颜色数为Δ^2+2；利用偶距直径（even-diameter）与奇数环长的关系构造可合并的基集合；以及通过层层降维将O(n^1/2)改进为O(n^ε)。

**📊 数据集**

该工作为理论性研究，无具体实验数据集，主要以数学证明与理论分析为主。

**📈 对比分析**

与之前的最优算法（如Kierstead 1998的O(n^1/2)色）相比，新算法在奇数环长足够大时将颜色数从O(n^1/2)提升至任意O(n^ε)，尤其在奇数环长为Ω(n^c)时仅需O(log n)色，明显优于以往的O(n^{(1-c)/2}√{log n})上界。

**⚠️ 局限性**

局限性包括：①实现需要的奇数环长非常大，实际图形中可能难以满足；②尽管颜色数可被压至O(n^ε)，但相较于理论下界的Ω(n^{1/g-3}/log n)仍相距甚远，指数收敛速度慢；③算法的实现复杂度较高，尤其是多层递归与组着色的细节实现要求较高。

---

## 379. Average-Tree Phylogenetic Diversity Parameterized by Scanwidth and Invisibility

**arXiv ID:** 2604.27745 | [PDF](https://arxiv.org/pdf/2604.27745v1)

**作者:** Leo van Iersel `[一作]` (TU Delft), Mathias Weller `[通讯]` (Université Gustave Eiffel)

**通讯引用:** 726 | [OpenAlex ID](https://openalex.org/A5030931327)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究在包含重叠进化事件的有根系统发生网络中计算平均树系统发生多样性（APD）的算法，并以扫描宽度（scanwidth）为参数分析其复杂度。

**💡 创新点**

创新点：①首次证明当扫描宽度为2时，APD最大化问题可多项式求解，而扫描宽度为3即NP‑难；②提出基于扫描宽度的FPT算法，时间为 2^扫描宽度 · n；③在可见重叠网络（reticulation‑visible）以及可见性有限的网络上实现线性时间求解。

**🔧 技术方法**

主要技术：动态规划在树扩展（tree‑extension）上递归计算；使用继承概率和边权重的概率论公式；对可见重叠网络利用树结构的递归特性；对包含不可见重叠的网络采用分块（blob）分离并乘积组合的技术。

**📊 数据集**

本工作为理论分析，未使用实际数据集，全部以算法证明和复杂度分析为主。

**📈 对比分析**

与已有结果比较：传统树模型下的 APD 可用贪心多项式求解；一般网络中已知 #P‑难；本论文提供了扫描宽度≤2时的多项式解、扫描宽度>2时的NP‑难证明，以及扫描宽度参数化下的 2^扫描宽度·n 的 FPT 算法；在可见网络上实现了线性时间求解，显著优于先前的指数级或 #P‑难解法。

**⚠️ 局限性**

局限性：①仅给出理论算法，未实现或评估实际运行时间；②对网络的结构要求较高（扫描宽度小、可见重叠或不可见重叠有限）；③在高度复杂或大规模网络中，FPT 的 2^扫描宽度系数可能仍不可接受；④仅适用于标准的继承概率和边权重模型，未考虑更复杂的进化模型。

---

## 380. Consumer Attitudes Towards AI in Digital Health: A Mixed-Methods Survey in Australia

**arXiv ID:** 2604.27744 | [PDF](https://arxiv.org/pdf/2604.27744v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 381. Auditing Frontier Vision-Language Models for Trustworthy Medical VQA: Grounding Failures, Format Collapse, and Domain Adaptation

**arXiv ID:** 2604.27720 | [PDF](https://arxiv.org/pdf/2604.27720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 382. A generalised pre-training strategy for deep learning networks in semantic segmentation of remotely sensed images

**arXiv ID:** 2604.27704 | [PDF](https://arxiv.org/pdf/2604.27704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 383. Towards an Ethical AI Curriculum: A Pan-African, Culturally Contextualized Framework for Primary and Secondary Education

**arXiv ID:** 2604.27708 | [PDF](https://arxiv.org/pdf/2604.27708v1)

**作者:** Abidemi Kuburat Adedeji `[一作]` (Abraham Adesanya Polytechnic), Sulaiman Oluwasegun Yusuff `[通讯]` (Ball State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套面向非洲小学和中学的、以Ubuntu为核心的伦理人工智能课程框架，并设计了多国验证方案

**💡 创新点**

将全球AI伦理原则与非洲关系伦理映射结合，构建了五大伦理核心、四大课程领域和年龄分层进度，并提出了从政策到课堂的完整落地路径

**🔧 技术方法**

采用主题综合法对政策文件和学术文献进行编码，构建框架；计划使用 Delphi、教师问卷和多国课堂试点来验证

**📊 数据集**

对非洲联盟、UNESCO及各国AI战略文件、相关学术文献进行编码，收集约30–40名专家意见和至少600名教师问卷数据

**📈 对比分析**

通过对比非洲大陆与各国政策框架以及全球AI伦理与Ubuntu伦理的映射来评估框架适配度，因尚无实证试点，性能指标尚未验证

**⚠️ 局限性**

作为概念性综述缺乏实证验证；资料主要来自英语文献，对法语、葡语、阿拉伯语的覆盖不足；框架需在各国上下文进一步共同设计与评估

---

## 384. Order-invariant cluster first-order logic on graph classes of bounded degree

**arXiv ID:** 2604.27693 | [PDF](https://arxiv.org/pdf/2604.27693v1)

**作者:** Fatemeh Ghasemi `[一作]` (Univ Paris Est Creteil), Julien Grange `[通讯]` (Univ Paris Est Creteil)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了集群一阶逻辑（cluster first-order logic）并证明其在有界度图类上，顺序不变的集群FO可归约为普通FO，且模型检测是固定参数可解的。

**💡 创新点**

首次构造了保持相似性的线性顺序（(k,F)-orders），实现了在有界度图上顺序不变性和普通一阶逻辑等价的结果；同时给出了集群FO比普通FO更强的分离例子。

**🔧 技术方法**

使用了局部-全局策略、k-上下文、(k,F)-orders、集群Ehrenfeucht–Fraïssé游戏、可计算的上下文摘要、递归模型检查算法等技术。

**📊 数据集**

无；该工作为理论研究，无实验数据集。

**📈 对比分析**

通过逻辑等价性证明和模型检查算法的时间复杂度分析，得到在有界度图类上模型检测的FPT算法，时间为f(|φ|)·O(|A|^2)。

**⚠️ 局限性**

只针对集群FO得到结果，顺序不变的普通FO在有界度图上的等价性仍未解决；方法在构造顺序时较为复杂，难以推广到更一般图类。

---

## 385. Users' Activity Logs: the Good, the Bad, the Misconception, and the Disastrous

**arXiv ID:** 2604.27676 | [PDF](https://arxiv.org/pdf/2604.27676v1)

**作者:** Eman Alashwali `[一作]` (King Abdulaziz University), Eman Alashwali `[通讯]` (King Abdulaziz University)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5030231646)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对30位沙特阿拉伯 Google 账户持有者的访谈数据进行二次定性分析，探讨用户对 Google 活动日志控制的正面、负面、误解和极端负面看法。

**💡 创新点**

首次从用户视角提供活动日志的平衡视图，系统归纳了“好处、坏处、误区和灾难”四大主题，并提出了针对服务商、研究者和用户的实用改进建议。

**🔧 技术方法**

采用模板分析（template analysis）方法，对访谈文字进行层级编码和主题梳理。

**📊 数据集**

使用来自 Alashwali 与 Cranor 2021 年的 30 份沙特阿拉伯访谈记录，包含受访者的性别、年龄、技术背景等信息。

**📈 对比分析**

由于研究为质性探索性研究，未涉及定量性能指标；研究通过与先前工作（如 Farke 等）对比，验证了主题的一致性与差异，但未给出可度量的性能结果。

**⚠️ 局限性**

局限性包括：数据为二次使用，收集时间为 2021 年；样本量小且以西部地区女性为主，可能影响普适性；访谈仅在阿拉伯语环境中进行，译文可能存在细微差异；未涉及跨文化对比或定量验证。

---

## 386. One Single Hub Text Breaks CLIP: Identifying Vulnerabilities in Cross-Modal Encoders via Hubness

**arXiv ID:** 2604.27674 | [PDF](https://arxiv.org/pdf/2604.27674v1)

**作者:** Hiroyuki Deguchi `[一作]` (NTT), Yusuke Sakai `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 3823 | [OpenAlex ID](https://openalex.org/A5042174785)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于解析式求取最优hub嵌入并利用逆向解码模型与 beam local search 生成 hub 文本的方法，以揭示跨模态编码器在高维共享嵌入空间中的 hubness 漏洞。

**💡 创新点**

创新点在于：①通过解析式（Cauchy–Schwarz）直接得到最优 hub 嵌入；②将传统贪婪搜索替换为 beam local search，显著提升了 hub 文本搜索的效果与稳定性；③引入 boss–worker 并行框架加速搜索过程。

**🔧 技术方法**

核心技术包括：解析式 hub 嵌入求解、基于 mT5-base 的逆向解码模型、beam local search、boss–worker 并行实现、统计显著性检验（配对自助抽样）。

**📊 数据集**

使用 MSCOCO、nocaps 进行调参与评估；Flickr30k 用于检索任务；在 CLIP、LION‑CLIP、DFN‑CLIP、AltCLIP 等跨模态编码器上测试。

**📈 对比分析**

与先前的 GLS 方法和人类参考文本对比；在 MSCOCO、nocaps 上，本方法生成的 hub 文本在多模型上均获得更高的 CLIPScore；在 MSCOCO、Flickr30k 的检索任务中，单一 hub 文本即可将 NDCG、MAP 等指标显著下降，显示其对模型性能的实际威胁。

**⚠️ 局限性**

局限性包括：时间复杂度高，beam size 越大计算成本线性上升；仅搜索固定长度序列，未考虑可变长度导致潜在更强 hub 文本；hub 文本不自然，易被检测，需要额外过滤机制；在更大模型或更高 beam 下仍显耗时。

---

## 387. TwinGate: Stateful Defense against Decompositional Jailbreaks in Untraceable Traffic via Asymmetric Contrastive Learning

**arXiv ID:** 2604.27861 | [PDF](https://arxiv.org/pdf/2604.27861v1)

**作者:** Bowen Sun `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TwinGate双编码器框架，用于在无可追踪流量下检测并阻止分解式越狱攻击；

**💡 创新点**

创新点在于引入非对称对比学习(ACL)聚类不同语义但同一恶意意图的碎片，并通过冻结编码器实现快速继承，既实现高召回又保持极低误报；

**🔧 技术方法**

技术上采用双编码器结构（ACL微调编码器+冻结语义编码器）、向量数据库检索、异步并行GPU推理、NVLink零拷贝与HBM存储；

**📊 数据集**

使用自建大规模数据集，包含3.62M条请求、8,681个恶意意图、250k善意意图及603k独立善意样本，所有恶意碎片通过多种拆分模型生成；

**📈 对比分析**

与Llama-Guard-3-8B、Intent-FT、Window Monitor等基线对比，TwinGate在严格的因果评测中召回率>0.76，误报率<0.2%，吞吐量>1700 QPS，P99延迟<300ms，显著优于基线；

**⚠️ 局限性**

局限性包括对高维向量存储和清理策略的依赖、对极度稀疏或慢速分解攻击的耐受性需进一步验证，以及在极端白盒攻击下仍可能出现一定的攻击成功率。

---

## 388. ZipCCL: Efficient Lossless Data Compression of Communication Collectives for Accelerating LLM Training

**arXiv ID:** 2604.27844 | [PDF](https://arxiv.org/pdf/2604.27844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 389. AI Inference as Relocatable Electricity Demand: A Latency-Constrained Energy-Geography Framework

**arXiv ID:** 2604.27855 | [PDF](https://arxiv.org/pdf/2604.27855v1)

**作者:** Xubin Luo `[一作]` (Southwestern University of Finance and Economics), Yang Cheng `[通讯]` (Southwestern University of Finance and Economics)

**通讯引用:** 13211 | [OpenAlex ID](https://openalex.org/A5010449128)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究 AI 推理在不同地区的执行位置如何影响电力消费，提出了能耗地理框架，并通过能耗-延迟前沿来量化延迟宽容度与能源/碳收益的关系。

**💡 创新点**

创新点在于将 AI 推理视为可移动的电力需求，定义了能耗-延迟前沿、可迁移推理需求 (RID)、能耗/碳延迟回报 (ERL/CRL) 与迁移盈亏 (NB_k,i) 等新指标，并在此基础上给出三层空间结构（本地、区域、能源导向）和一套完整的实验规范。

**🔧 技术方法**

主要技术包括三层（客户→服务节点→计算节点）优化模型，约束包括电价、碳强度、PUE、容量、延迟、迁移摩擦；使用整数线性规划求解；并辅以可迁移度、能耗/碳回报等运算公式。

**📊 数据集**

使用合成/半合成数据：全球十个计算节点的电价、碳强度、PUE、容量；Azure 区域间 RTT 与出口费用；任务分为四类（交互、标准、背景、批处理），分别设定延迟容忍度、能耗、计算量等参数。

**📈 对比分析**

与四种基线（Local‑Only、Nearest‑Region、Price‑Only、Carbon‑Only）比较。结果显示联合能耗‑延迟策略在满足 SLO 的前提下，能实现 8–10% 的电费和 7–10% 的碳排放下降，同时可迁移需求占比可达 30–50%。性能在不同延迟宽容度下呈现非线性收益递减。

**⚠️ 局限性**

局限包括：忽略了会话状态迁移、KV 缓存、检索数据本地化、法务与数据主权等约束；电价与碳强度采用简化代理，未使用实际生产轨迹；容量模型静态化；网络延迟仅基于距离与固定开销估计；不考虑排队与动态负载；因此结果主要揭示机制而非精确量化。

---

## 390. Requirements Debt in AI-Enabled Perception Systems Development: An Industrial RE4AI Perspective

**arXiv ID:** 2604.27825 | [PDF](https://arxiv.org/pdf/2604.27825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 391. Machine Unlearning for Class Removal through SISA-based Deep Neural Network Architectures

**arXiv ID:** 2604.27804 | [PDF](https://arxiv.org/pdf/2604.27804v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 392. Learning-Based Hierarchical Scene Graph Matching for Robot Localization Leveraging Prior Maps

**arXiv ID:** 2604.27821 | [PDF](https://arxiv.org/pdf/2604.27821v1)

**作者:** Nimrod Millenium Ndulue `[一作]` (University of Luxembourg), Jose Luis Sanchez-Lopez `[通讯]` (University of Luxembourg)

**通讯引用:** 2277 | [OpenAlex ID](https://openalex.org/A5078546155)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一套端到端可微的层次场景图匹配管线，将机器人构建的情境图与BIM导出的建筑图进行节点对齐。

**💡 创新点**

1) 在两级层次结构上增强图的边类型，加入内层与跨层关系；2) 使用共享MLP将不同类型节点投射到同一嵌入空间；3) 结合GATv2、实例归一化、Sinkhorn与匈牙利算法实现高效可微匹配；4) 只用合成数据训练，实现零样本迁移到真实LiDAR。

**🔧 技术方法**

多层感知器(MLP)、图注意力网络(GATv2)、点积相似度、实例归一化、Sinkhorn软匹配、匈牙利算法、二进制交叉熵损失、AdamW优化、Optuna超参搜索。

**📊 数据集**

MSD合成楼层平面图数据集用于训练与评估；真实LiDAR环境（RE）及对应BIM结构图用于零样本测试。

**📈 对比分析**

与仅实现组合搜索的iS-Graphs比较。在合成测试集上，模型F1为85%（iS-Graphs 95%但仅86%样本完成），速度提升82倍；在真实LiDAR环境中，模型F1 84%高于iS-Graphs的67%，速度提升9倍，完成率100%。

**⚠️ 局限性**

未考虑建筑对称性导致对称平面图中的节点对应模糊；精度与召回仍有提升空间。

---

## 393. WindowsWorld: A Process-Centric Benchmark of Autonomous GUI Agents in Professional Cross-Application Environments

**arXiv ID:** 2604.27776 | [PDF](https://arxiv.org/pdf/2604.27776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 394. Instruction-Guided Poetry Generation in Arabic and Its Dialects

**arXiv ID:** 2604.27766 | [PDF](https://arxiv.org/pdf/2604.27766v1)

**作者:** Abdelrahman Sadallah `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了可控制的阿拉伯诗歌生成与分析指令微调数据集，并在此基础上对多款大型语言模型进行微调；

**💡 创新点**

提供了覆盖现代标准阿拉伯语与四大方言的3.22万模板、1.35M训练对，兼顾生成、续写、修订与多选分析任务，首次实现多任务指令式诗歌创作；

**🔧 技术方法**

采用LoRA参数高效微调、指令微调框架，并使用LLM-as-a-judge和人工评测相结合的评估体系；

**📊 数据集**

整合并标准化了约427K条训练诗歌与6.9K条测试诗歌，来自PoetsGate、Adab等公开资源，并自动生成关键词、主题等元数据；

**📈 对比分析**

与四个基线模型（ALLaM、Qwen3、Fanar、LLaMA-3）在联合与课程学习两种训练策略下对比，结果显示微调后模型在自动指标、LLM评估和人工评测中均显著提升，最佳模型整体得分达3.99/5；

**⚠️ 局限性**

仅使用LoRA而非全参数微调、模型规模受限、数据主要为古典或MSA诗歌，缺少当代方言与自由体诗，影响模型在更广泛诗歌风格上的表现。

---

## 395. GourNet: A CNN-Based Model for Mango Leaf Disease Detection

**arXiv ID:** 2604.27764 | [PDF](https://arxiv.org/pdf/2604.27764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 396. Learning to Reason: Targeted Knowledge Discovery and Fuzzy Logic Update for Robust Image Recognition

**arXiv ID:** 2604.27759 | [PDF](https://arxiv.org/pdf/2604.27759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. Toward a Characterization of Simulation Between Arithmetic Theories

**arXiv ID:** 2604.27787 | [PDF](https://arxiv.org/pdf/2604.27787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 398. Autonomous Traffic Signal Optimization Using Digital Twin and Agentic AI for Real-Time Decision-Making

**arXiv ID:** 2604.27753 | [PDF](https://arxiv.org/pdf/2604.27753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 399. Focus Session: Autonomous Systems Dependability in the era of AI: Design Challenges in Safety, Security, Reliability and Certification

**arXiv ID:** 2604.27807 | [PDF](https://arxiv.org/pdf/2604.27807v1)

**作者:** Behnaz Ranjbar `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6244 | [OpenAlex ID](https://openalex.org/A5100755285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并提出面向AI时代的自动驾驶与嵌入式系统的跨层可靠性与安全性设计方法，并在此基础上给出了两种基于机器学习的安全驱动资源分配范例，探讨了如何在不确定性与延迟下仍保持系统安全与可认证；

**💡 创新点**

创新点在于：①提出跨层可靠性与安全框架，融合设计时与运行时保障；②引入机器学习驱动的可靠性评估与资源分配策略（如边缘–云感知分区与GPU分区），实现对感知不确定性与延迟的系统级建模与优化；③通过多模型对比（SEDAN、INDRA、LATTE、TENET、GAAD等）验证新方法在检测率与实时性上的优势；

**🔧 技术方法**

采用深度学习（GRU、LSTM、TCN、VAE+GAN、卷积+BiLSTM）、自适应自优化与预测模型、跨层资源调度与加密算法、可靠性建模与可达性分析、以及可解释性与鲁棒训练等技术；

**📊 数据集**

利用的公开数据集包括Lyft运动预测数据集、F1TENTH仿真平台、以及汽车CAN总线攻击实验数据；

**📈 对比分析**

与现有IDS与异常检测框架相比，本文提出的边缘–云分区和GPU分区方法在轨迹偏差、控制能耗和安全边际上均优于单纯边缘或云方案；在感知异常检测方面，采用VAE+GAN在Lyft数据集上达到≥90%的准确率，显著低于传统重建方法；

**⚠️ 局限性**

局限性包括：缺乏大规模实车验证、对极端稀缺故障场景的样本不足、ML模型解释性与可验证性仍待加强、实时性与能耗权衡需要进一步细化、以及跨层设计与监管合规性的复杂性。

---

## 400. Reasoning over Object Descriptions Improves Coreference Resolution in Task-Based Dialogue Systems

**arXiv ID:** 2604.27850 | [PDF](https://arxiv.org/pdf/2604.27850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 401. Intent2Tx: Benchmarking LLMs for Translating Natural Language Intents into Ethereum Transactions

**arXiv ID:** 2604.27763 | [PDF](https://arxiv.org/pdf/2604.27763v1)

**作者:** Zhuoran Pan `[一作]` (Peking University), Zhong Chen `[通讯]` (Peking University)

**通讯引用:** 46792 | [OpenAlex ID](https://openalex.org/A5100430399)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了Intent2Tx基准，用于评估LLM将自然语言意图翻译为可执行的以太坊交易。

**💡 创新点**

创新点在于使用300天真实主网交易生成高保真数据集，设计可执行评估框架和差分状态分析，并支持单步与多步意图。

**🔧 技术方法**

使用LLM推理、检索增强、LoRA微调、Forked主网模拟和差分状态分析等技术。

**📊 数据集**

采用29,921条单步和1,575条多步实例，数据来源于2025年3月至2026年1月的以太坊主网日志。

**📈 对比分析**

对16种LLM进行直接推理与检索增强实验，使用格式、逻辑、参数、Pass@1和最终加权分数进行评估；检索显著提升逻辑和参数准确度，但整体Pass@1仍低，执行成功率亦有限。

**⚠️ 局限性**

局限性包括：跨类别泛化能力有限，多步规划仍弱，执行评估需昂贵的Fork环境，且小模型与少量数据表现不佳。

---

## 402. Multifaceted Hero Developers and Bug-Fixing Outcomes Across Severity

**arXiv ID:** 2604.27754 | [PDF](https://arxiv.org/pdf/2604.27754v1)

**作者:** Amit Kumar `[一作]` (Indian Institute of Information Technology), Sonali Agarwal `[通讯]` (Indian Institute of Information Technology)

**通讯引用:** 4406 | [OpenAlex ID](https://openalex.org/A5004401828)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了开源项目中英雄开发者的多维特征，结合技术和社交两类贡献指标，分析其对不同严重程度bug修复效果的影响。

**💡 创新点**

首次同时量化技术与社交维度下的英雄集重叠，提出多维英雄分类，并研究这些分类在 bug 修复率和重开率上的分层表现，提供跨项目、跨指标的鲁棒性检验。

**🔧 技术方法**

采用 Pareto 20/80 规则识别英雄，使用 Jaccard 重叠度、Spearman 相关系数以及 Rank‑Biased Overlap（RBO）等统计方法，衡量英雄集间的重叠与排名一致性，并用 fix/reopen 率评估性能。

**📊 数据集**

SmartSHARK 2.1 数据集，包含 77 个 Apache 软件基金会项目的提交记录、文件变更以及 issue 与评论数据。

**📈 对比分析**

通过五个技术/社交指标计算英雄集，比较技术/社交交集以及多维英雄组合的 Jaccard 重叠，评估各类别在不同严重程度下的 fix 率与 reopen 率排名。结果显示英雄集在不同维度间重叠率低，bug 修复效果差异不大但排名随 severity 变化。

**⚠️ 局限性**

仅将 issue 评论作为社交指标，忽略邮件、拉取请求等；使用 issue priority 作为严重程度的代理，可能不准确；身份映射可能不完整；研究仅覆盖 Apache 生态，结果可能不适用于其他开源社区。

---

## 403. Rethinking Agentic Reinforcement Learning In Large Language Models

**arXiv ID:** 2604.27859 | [PDF](https://arxiv.org/pdf/2604.27859v1)

**作者:** Fangming Cui `[一作]` (Beijing), Jiahong Li `[通讯]` (Shanghai)

**通讯引用:** 16575 | [OpenAlex ID](https://openalex.org/A5100367188)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文综述并系统梳理了基于大型语言模型（LLM）的Agentic Reinforcement Learning（RL）方法，阐释了从概念基础到设计框架、方法创新、技术手段及其面临的挑战与未来方向的完整视图。

**💡 创新点**

创新点在于将LLM、Agent和RL三大领域进行整合，提出了“动作、规划、记忆、工具”四大核心组件的统一框架，并对主流RL算法（PPO、DPO、GRPO、GSPO、DAPO、SAPO、SimPO等）与工具交互、记忆机制、动态规划等技术进行系统性评述，揭示了现有方法的优缺点与研究空白。

**🔧 技术方法**

使用的技术包括：基于LLM的策略网络、ReAct与SAND等交互式推理框架、动态规划与MCTS、记忆库（向量数据库、Mem0）与注意力机制、工具调用机制、以及多种RL优化算法（PPO、DPO、GRPO、GSPO、DAPO、SAPO、SimPO、VinePPO、VAPO、PSGPO、GMPO、TreePO、PAPO等）。

**📊 数据集**

作为综述，论文并未进行新的实验，而是引用并讨论了多种公开数据集和基准，如AIME 2024（数学竞赛），软件工程代码生成与调试数据集，科学论文检索与分析集，Web导航与工具使用日志等；同时提及LLM预训练模型（如Qwen、ChatGPT等）的评估数据。

**📈 对比分析**

比较方面，论文对不同算法在标准任务上的表现进行对比总结：例如DAPO在AIME 2024上取得50分，GRPO/GSPO在大型多轮推理任务中表现优于传统PPO；SimPO与DPO在RLHF的简化与稳定性方面优于原始RLHF；TreePO与PSGPO在代码生成与工具调用任务中显著提升了回报与样本效率。

**⚠️ 局限性**

限制与挑战包括：模型仍受限于基础LLM的能力边界，难以突破；多轮推理与稀疏奖励导致的信用分配不稳定；环境建模与动态适配仍不成熟；可信度与防止幻觉的机制尚未完善；大规模RL训练的计算开销与能源消耗巨大；评估指标与公开基准不足，缺乏统一的评价体系。

---

## 404. Taming Noise-Induced Prototype Degradation for Privacy-Preserving Personalized Federated Fine-Tuning

**arXiv ID:** 2604.27833 | [PDF](https://arxiv.org/pdf/2604.27833v1)

**作者:** Yuhua Wang `[一作]` (Beihang University), Zhiming Zheng `[通讯]` (Beihang University)

**通讯引用:** 2923 | [OpenAlex ID](https://openalex.org/A5112478420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个名为VPDR的客户端插件，结合了方差自适应原型扰动（VPP）和基于蒸馏的剪裁正则化（DCR），用于在保持局部差分隐私的前提下改进基于原型的个性化联邦学习（ProtoPFL）的性能。

**💡 创新点**

创新点在于：①利用每个维度的类内外方差评估维度的辨别力，动态分配噪声，使噪声更少地作用于重要维度，从而提升表示质量；②通过软剪裁层和EMA教师-学生蒸馏的预测一致性约束，抑制硬剪裁导致的特征失真，稳定特征范数并提升鲁棒性；③在隐私预算分配上使用分层机制，保证与传统等距高斯扰动相同的LDP强度。

**🔧 技术方法**

技术方法包括：局部差分隐私（(ε,δ)-LDP）机制、一次性拉普拉斯Top‑k选择子空间、分组ℓ₂剪裁、方差自适应噪声分配、软剪裁层、EMA教师模型、KL蒸馏正则化、ProtoPFL框架的原型聚合与对齐。

**📊 数据集**

实验使用了三组多域数据集（Digits：MNIST、USPS、SVHN、Synthetic；Office–Caltech：Amazon、Caltech、DSLR、Webcam；PACS：Photo、Art、Cartoon、Sketch）以及CIFAR‑10的标签偏斜实验，验证在多种ProtoPFL模型（FedProto、FedPCL、FPL、FedPLVM、FedTGP、MPFT）上的效果。

**📈 对比分析**

与基线IGPP（等距高斯原型扰动）相比，VPDR在所有ProtoPFL框架和数据集上平均提升了0.5–1.5%的准确率，标准差降低，且在极端标签偏斜、对抗攻击（特征空间劫持、成员推断）下保持与IGPP相当的隐私保护；计算开销仅增加约8%，通信成本不变。

**⚠️ 局限性**

局限性包括：①需要手动调节多组超参数（如子空间比例ρ、预算分配比例r、软剪裁系数γ等）；②在极低隐私预算（ε很小）或高维嵌入空间下，方差估计可能不稳定，噪声分配效果不如预期；③实验集中在图像域，对文本或更大规模分布差异的适用性尚未充分验证。

---

## 405. Multi-Level Narrative Evaluation Outperforms Lexical Features for Mental Health

**arXiv ID:** 2604.27846 | [PDF](https://arxiv.org/pdf/2604.27846v1)

**作者:** Yuxi Ma `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 4143 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一套基于微观词汇、中观语义嵌入、宏观大模型评估的三层叙事分析框架，用于预测抑郁、焦虑和创伤症状。

**💡 创新点**

首次将分层语篇加工理论与计算方法对齐，证明宏观结构评估比词频或嵌入更能捕捉心理健康信号，并展示层级整合对不同任务的递增效果。

**🔧 技术方法**

使用 Simplified Chinese LIWC 提取词汇特征，OpenAI 768/1536 维句向量计算语义连贯性，以及 GPT‑4o 作为结构化评估器输出 Labov 结构与 CBT 维度。

**📊 数据集**

共830份中文治疗性写作文本，涵盖 9–50 岁人群，来自六项干预研究，标注抑郁、焦虑与创伤严重程度。

**📈 对比分析**

采用分层特征组合的 ExtraTrees 回归与 Gradient Boosting 多分类，通过 5 折交叉验证比较，宏观层单独达到 R²≈0.295/0.204，AUC≈0.692；全模型 R²≈0.332、AUC≈0.718，显示宏观层驱动主导。

**⚠️ 局限性**

样本异质性、年龄与干预背景混杂、宏观评估的自动化解释性仍待验证、语义层缺乏独立贡献、跨文化推广需进一步测试。

---

## 406. ObjectGraph: From Document Injection to Knowledge Traversal -- A Native File Format for the Agentic Era

**arXiv ID:** 2604.27820 | [PDF](https://arxiv.org/pdf/2604.27820v1)

**作者:** Mohit Dubey `[一作]`, Open Gigantic `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种名为ObjectGraph的文档格式，将Markdown改造为可查询的类型化知识图，以解决LLM代理对文档的大量无效注入问题。

**💡 创新点**

提出Progressive Disclosure Model、两原语LLM原生查询协议、角色作用访问控制和可执行断言节点，满足六项必需属性且兼容Markdown。

**🔧 技术方法**

采用结构化标签、图边声明、索引块、密集/完整层级、LLM推理路由、增量加载等技术实现高效文档查询与执行。

**📊 数据集**

在包含240份跨五类（技能文件、运行手册、执行计划、技术文档、知识库）文档与8类任务的基准上评估，并使用Claude Sonnet4.5、Haiku4.5、GPT‑4o。

**📈 对比分析**

与全文注入、RAG、SkillReducer等基线比较，Token消耗平均下降约92%，上下文叠加提升36×，任务准确率保持或提升，转译器Fidelity≥0.987。

**⚠️ 局限性**

仅支持单文件内边界，跨文件引用不支持，评估规模有限，易受恶意密集块影响，缺乏标准化治理。

---

## 407. MASCing: Configurable Mixture-of-Experts Behavior via Activation Steering Masks

**arXiv ID:** 2604.27818 | [PDF](https://arxiv.org/pdf/2604.27818v1)

**作者:** Jona te Lintelo `[一作]` (Radboud University), Stjepan Picek `[通讯]` (Radboud University)

**通讯引用:** 4735 | [OpenAlex ID](https://openalex.org/A5024072796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 MASCing 的无训练、基于激活引导的稀疏专家调控框架，用于在不重新训练模型的前提下，快速在多种安全场景中配置 MoE 语言模型的行为。

**💡 创新点**

创新点在于：①利用 LSTM 作为可微分的行为替代模型，直接建模连续路由 logit 与下游行为的映射；②通过优化稀疏 steering 矩阵并剪枝得到最小化专家子电路；③在推理时仅对路由 logits 加上稀疏掩码，既能增强安全性又能按需放宽限制，且不损失语言实用性。

**🔧 技术方法**

核心技术包括：Mixture-of-Experts（MoE）架构、激活 steering、LSTM 顺序建模、L1 正则化与对称阈值剪枝、路由 logits 标准化与可调幅度参数、以及对比实验与人类验证评估。

**📊 数据集**

使用了多种公开数据集：AdvBench 与 Multi-Turn Human Jailbreaks（MHJ）用于多轮 jailbreak 防御；EroticaAnalysis 与 Facebook's NaturalReasoning 用于成人内容生成；还收集了七个开源 MoE 模型的路由日志用于训练 LSTM。

**📈 对比分析**

与 SteerMoE 等现有推理时干预方法对比，MASCing 在七个 MoE 模型上平均提升多轮 jailbreak 防御成功率从 52.5% 提升至 83.9%（相对 58.4% 的 SteerMoE），成人内容生成成功率从 52.6% 提升至 82.0%。模型在 MMLU 和 GSM8K 上平均仅下降 4.1%，说明对语言实用性影响极小。

**⚠️ 局限性**

局限性包括：①依赖 LSTM 对路由行为的近似，极深或高度非线性模型可能导致识别失效；②仅在路由层进行干预，若专家本身已被污染或缺乏对齐信息，激活 steering 无法完全恢复安全；③使用的静态掩码对未知或零日 jailbreak 的鲁棒性有限，未来可探索动态、输入依赖的 steering。

---

## 408. Probabilistic Circuits for Irregular Multivariate Time Series Forecasting

**arXiv ID:** 2604.27814 | [PDF](https://arxiv.org/pdf/2604.27814v1)

**作者:** Christian Klötergens `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**通讯引用:** 17331 | [OpenAlex ID](https://openalex.org/A5039470755)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个基于概率电路（Sum‑Product Network）的架构，用于预测不规则多变量时间序列的联合分布。

**💡 创新点**

创新点在于通过在概率电路中固定结构并使用可学习的混合权重，既保证了边缘一致性，又能表达复杂的多变量依赖；同时利用编码器处理不规则数据，生成叶子分布和权重。

**🔧 技术方法**

采用概率电路、混合高斯/DSF 叶子、Transformer/Encoder 对不规则输入进行编码、对数域计算与 Log‑Sum‑Exp 组合、以及对权重矩阵的可学习递归聚合。

**📊 数据集**

使用四个真实世界数据集：USHCN 气候数据、Physionet‑2012、MIMIC‑III 与 MIMIC‑IV 医疗数据。

**📈 对比分析**

与 ProFITi、MOSES、NeuralFlows 等基线模型在 joint NLL（njNLL）和 marginal NLL（mNLL）上进行对比；Circuits 在绝大多数情景下均取得最低 njNLL，并在 mNLL 上优于基线，证明其在联合密度估计和边缘一致性方面的优势。

**⚠️ 局限性**

受限于 K 个混合成分导致的 O(K³) 计算复杂度（K 通常≤4），以及固定通道顺序可能限制表达灵活性；对大规模通道或更高维度分布的扩展仍面临挑战。

---

## 409. Hybrid Anomaly Detection for Bullion Coin Authentication Leveraging Acoustic Signature Analysis

**arXiv ID:** 2604.27803 | [PDF](https://arxiv.org/pdf/2604.27803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 410. Post-Optimization Adaptive Rank Allocation for LoRA

**arXiv ID:** 2604.27796 | [PDF](https://arxiv.org/pdf/2604.27796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 411. "It depends on where AI is used": Players' attitude patterns and evaluative logics toward different AI applications in digital games

**arXiv ID:** 2604.27812 | [PDF](https://arxiv.org/pdf/2604.27812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 412. A Grid-Aware Agent-Based Model for Analyzing Electric Vehicle Charging Systems

**arXiv ID:** 2604.27849 | [PDF](https://arxiv.org/pdf/2604.27849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 413. RuC: HDL-Agnostic Rule Completion Benchmark Generation

**arXiv ID:** 2604.27780 | [PDF](https://arxiv.org/pdf/2604.27780v1)

**作者:** Arnau Ayguadé Domingo `[一作]` (Barcelona Supercomputing Center), Dario Garcia-Gasulla `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5010831226)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于HDL语法规则的RTL代码补全基准框架RuC，用以生成可控粒度的补全任务。

**💡 创新点**

创新点在于通过语法树掩码实现任意规则的补全，并提供多种提示策略与语法及功能双重验证。

**🔧 技术方法**

采用ANTLR等语法解析器、编译器工具（Verilator、Yosys、SAT solver）以及多种LLM提示策略（FIM、Chat）进行实现。

**📊 数据集**

使用Tiny Tapeout shuttle和CVE2 RISC‑V核心两套SystemVerilog代码库生成基准数据。

**📈 对比分析**

对比多款开源LLM（Qwen、Seed、DeepSeek等）在不同提示与规则下的补全准确率，结果显示FIM提示最佳、模型规模越大性能一般提升，规则间差异可达80%。

**⚠️ 局限性**

局限性包括对大型设计仍受上下文窗口限制、依赖HDL语法定义、对复杂逻辑仍表现低下且无法覆盖所有语法规则。

---

## 414. Feature-Centric Methodology for Analyzing Cross-Chain NFT Migration Compatibility

**arXiv ID:** 2604.27805 | [PDF](https://arxiv.org/pdf/2604.27805v1)

**作者:** Mohd Sameen Chishti `[一作]` (NTNU), Jingyue Li `[通讯]` (NTNU)

**通讯引用:** 4420 | [OpenAlex ID](https://openalex.org/A5067021027)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种基于特征的跨链 NFT 迁移兼容性分析方法。

**💡 创新点**

创新点在于提出四层 NFT 架构和四阶段迁移分析流程，能系统判断特征兼容性。

**🔧 技术方法**

用架构层依赖映射、功能特征规范、四阶段流程和示例案例（Ethereum→Solana）实现。

**📊 数据集**

使用以太坊 ERC‑721 + ERC‑2981 合约与 Solana SPL/Metaplex NFT 作为实验数据集。

**📈 对比分析**

通过对比源端所需原语与目标端可用原语，划分为本地保留、部分不匹配、完全不匹配，并在案例中验证匹配结果与预期一致，性能表现符合预期。

**⚠️ 局限性**

限制在于仅验证单一源-目标对，特征‑原语映射手工完成，未考虑离链与治理层，缺乏自动化工具。

---

## 415. Test Before You Deploy: Governing Updates in the LLM Supply Chain

**arXiv ID:** 2604.27789 | [PDF](https://arxiv.org/pdf/2604.27789v1)

**作者:** Mohd Sameen Chishti `[一作]` (Norwegian University of Science and Technology), Jingyue Li `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4420 | [OpenAlex ID](https://openalex.org/A5067021027)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出部署端治理框架，包括生产合约、按风险类别划分的回归测试套件以及兼容性门控，并在Anthropic Claude模型上进行探索性验证。

**💡 创新点**

将LLM更新治理转为部署方可控的兼容性检查，明确行为合约、按风险分类测试并通过自动门控阻止不合规更新，填补了传统供应链治理对LLM“无版本”漂移的缺口。

**🔧 技术方法**

采用CI/CD质量门、自然语言提示生成、自动化功能/格式/安全评估、模型快照记录以及阈值对比统计等技术实现。

**📊 数据集**

使用25个手工设计的提示，涵盖身份验证、数据校验和结构化输出三类风险，实验基于Anthropic Claude系列模型进行。

**📈 对比分析**

通过对不同模型版本在各风险类别的合规率进行对比，发现聚合指标掩盖的格式与安全回归；实验表明框架能有效检测到隐藏漂移，性能验证基于手工评估，展示了风险类别差异。

**⚠️ 局限性**

缺乏系统化回归套件设计与覆盖度衡量、阈值校准与统计置信度、随机性下的稳定性评估、CI/CD集成成本与运营开销、以及供应商透明度不足导致漂移归因困难。

---

## 416. CastFlow: Learning Role-Specialized Agentic Workflows for Time Series Forecasting

**arXiv ID:** 2604.27840 | [PDF](https://arxiv.org/pdf/2604.27840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 417. MotuBrain: An Advanced World Action Model for Robot Control

**arXiv ID:** 2604.27792 | [PDF](https://arxiv.org/pdf/2604.27792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 418. NeocorRAG: Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains

**arXiv ID:** 2604.27852 | [PDF](https://arxiv.org/pdf/2604.27852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 419. Maximally Diverse Stable Matchings: Optimizing Arbitrary Institutional Objectives

**arXiv ID:** 2604.27823 | [PDF](https://arxiv.org/pdf/2604.27823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 420. AnTi-MiCS: Analytical Framework for Bounding Time in Embedded Mixed-Criticality Systems

**arXiv ID:** 2604.27862 | [PDF](https://arxiv.org/pdf/2604.27862v1)

**作者:** Behnaz Ranjbar `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6244 | [OpenAlex ID](https://openalex.org/A5100755285)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过分析混合关键性系统任务的执行时间分布，提出AnTi-MiCS和MulTi-MiCS两种方法，在设计时确定适当的低级WCET，从而在不产生运行时开销的前提下提升处理器利用率和系统QoS。

**💡 创新点**

创新点在于提出基于期望执行时间（EET）的单级WCET选择公式，并进一步扩展为可生成多级WCET的可伸缩方法（MulTi-MiCS），充分考虑输入的时序相关性和执行时间分布特征。

**🔧 技术方法**

使用概率分布分析、EET/SEET计算、线性搜索、OTA WA进行WCET^HI估计，并在ODROID‑XU4平台上采集样本，配合EDF‑VD调度实现实验。

**📊 数据集**

实验基准来自MiBench（如insert‑sort、quicksort、smooth、epic、edge等）以及AXBench的matrix‑multiplier等。

**📈 对比分析**

与传统按比例设定WCET^LO、基于ACET+σ的统计方法以及运行时动态调整方法比较，AnTi‑MiCS和MulTi‑MiCS在模式切换概率、利用率、QoS以及系统目标值上均优于对照方法；MulTi‑MiCS平均提升QoS 30.27%、降低利用率浪费 35.89%，相较AnTi‑MiCS提升 6.41% QoS 与 8.23%利用率。

**⚠️ 局限性**

局限在于仅针对双关键性系统；方法依赖设计时完整采样，无法实时适应输入分布突变；对多关键性等级的支持有限；在极端或多峰分布情况下仍可能产生较多模式切换。

---

## 421. NetSatBench: A Distributed LEO Constellation Emulator with an SRv6 Case Study

**arXiv ID:** 2604.27854 | [PDF](https://arxiv.org/pdf/2604.27854v1)

**作者:** Andrea Detti `[一作]` (CNIT), Giuseppe Tropea `[通讯]` (NetSense)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了NetSatBench平台，并利用它评估了一种基于SRv6的LEO卫星架构，实现了端到端用户-网关通信的隧道化与动态切换；

**💡 创新点**

创新点在于提出分布式容器化仿真框架、声明式JSON场景文件与epoch驱动的时间演化、插件化物理层与路由模型、以及通过SRv6实现的端到端切换策略，显著提升了LEO网络实验的可扩展性与重现性；

**🔧 技术方法**

技术手段包括Docker容器、VXLAN层2覆盖、Etcd键值存储与发布/订阅、Python插件化的物理层与路由模块、SRv6隧道、IS-IS、预计算Oracle路由、Slant‑Range比特率模型、OpenStack虚拟机以及Linux Traffic Control框架；

**📊 数据集**

使用了StarPerf生成的Walker‑Delta星座轨道数据（H5文件），在此基础上扩展了用户、网关和可见性约束，构建了一个OneWeb类似的588颗卫星、5站、7终端的实验场景；

**📈 对比分析**

通过对比本地hand‑over策略与多种端到端hand‑over策略（最小访问延迟、最大最小速率、最大最小可见性），测量RTT和TCP Cubic/BBR吞吐量，结果表明端到端策略能显著降低RTT、提升吞吐率，并且在随机丢包场景下BBR表现更为稳健；

**⚠️ 局限性**

局限性包括：实验仍基于仿真，缺乏真实硬件验证；物理层模型简化为斜距模型，未考虑更复杂的链路衰减与误码；对大规模节点的实时同步存在一定延迟；手over策略仅涵盖有限规则，未覆盖所有实际业务场景。

---

## 422. AME-PIM: Can Memory be Your Next Tensor Accelerator?

**arXiv ID:** 2604.27808 | [PDF](https://arxiv.org/pdf/2604.27808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 423. WOOTdroid: Whole-system Online On-device Tracing for Android

**arXiv ID:** 2604.27830 | [PDF](https://arxiv.org/pdf/2604.27830v1)

**作者:** Simon Althaus `[一作]` (Technical University of Darmstadt), Ephraim Zimmer `[通讯]` (Technical University of Darmstadt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不修改 Android 系统或应用的前提下，设计实现了 WOOTdroid，一个基于 eBPF 的全系统在线追踪框架，能够完整记录系统调用和 Binder IPC 事务，并恢复高层 API 语义。

**💡 创新点**

创新点在于：①将 eAudit 的高效无事件丢失 syscalls 追踪移植至 Android；②通过在内核捕获 Binder 事务并结合预先提取的 Java 签名表，实现对 Binder 事务的语义解码；③将两者结合在单一 kernel 级别的追踪器中，实现完整性与语义的统一。

**🔧 技术方法**

技术手段包括：eBPF（BCC）与 eAudit 改写；Binder 事务解析；Java 反射获取签名表；Perf 环形缓冲区；Android Root + Magisk 获得特权。

**📊 数据集**

数据集为：Pixel9 设备运行 Android16；Geekbench6 基准；100 个 Google Play 免费热门应用（Top 100）进行完整性实验；10 个安全相关 Binder 方法的案例分析。

**📈 对比分析**

与传统 ftrace 对比：WOOTdroid 的 Geekbench 单核/多核 overhead 分别为 3.6%/0.9%，比 ftrace 的 5.9%/2.9% 更低；在 100 个 App 上，WOOTdroid 的 Unique Event Rate 平均 37.8%，比 ftrace 的 4.3% 高出约 33.5%。

**⚠️ 局限性**

局限性：需要设备 Root；仅支持 primitive、String 及部分简单 Binder 对象的参数解码，无法解析文件描述符、指针或回复；签名表仅覆盖 AIDL 定义的系统服务，厂商自定义或动态接口缺失；未评估在高并发下的完整性。

---

## 424. MCPHunt: An Evaluation Framework for Cross-Boundary Data Propagation in Multi-Server MCP Agents

**arXiv ID:** 2604.27819 | [PDF](https://arxiv.org/pdf/2604.27819v1)

**作者:** Haonan Li `[一作]` (China University of Geosciences), Qisheng Zhang `[通讯]` (China University of Geosciences)

**通讯引用:** 6944 | [OpenAlex ID](https://openalex.org/A5100433414)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多服务器MCP代理在非攻击情境下的跨信任边界凭证传播问题，提出MCPHunt基准框架以量化和评估此风险。

**💡 创新点**

创新点包括：① 基于canary的污点追踪将传播检测简化为字符串匹配；② 通过环境控制设计验证结果的可靠性；③ 完整的CRS划分区分任务必然传播与违规传播，并首次在多模型、多机制上量化非对抗性凭证泄漏。

**🔧 技术方法**

使用了canary字符串替代真实凭证进行 taint 跟踪、7个实验环境（风险、善意、硬负）下的日志收集、两层风险信号体系、GEE logistic 回归分析，以及 prompt 级缓解与后测 taint guard。

**📊 数据集**

构建了147个手工设计任务，涵盖9类风险机制，并在8个MCP服务器（文件、git、数据库、浏览器等）上对OpenAI、Anthropic、Google、Microsoft、DeepSeek、MiniMax等多模型执行，生成近8000条执行轨迹。

**📈 对比分析**

通过传播率、政策违规传播率和实用率等指标对模型、机制族、prompt 缓解进行对比；实验显示机制族主导传播，模型差异不大，M3级prompt 可将违规传播降至97%；模拟 taint guard 可将违规传播几乎消除。

**⚠️ 局限性**

局限包括：仅检测精确或近似匹配的字符串，无法捕获伪造或转义凭证；任务人为设计，缺乏真实企业工作负载验证；taint guard 仅后测，未实时干预；服务器覆盖有限，未包含云存储、邮件等。

---

## 425. Hyper-Dimensional Fingerprints as Molecular Representations

**arXiv ID:** 2604.27810 | [PDF](https://arxiv.org/pdf/2604.27810v1)

**作者:** Jonas Teufel `[一作]` (Karlsruhe Institute of Technology), Pascal Friederich `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6840 | [OpenAlex ID](https://openalex.org/A5052771582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于高维计算（HDC）与迭代信息传递的分子指纹——Hyperdimensional Fingerprints（HDF），可将任意分子图映射为固定维度的向量，并用于属性预测、相似性搜索和贝叶斯优化。

**💡 创新点**

创新点在于：①不依赖哈希折叠，采用随机高维向量和圆形卷积的代数运算，避免信息碰撞；②通过多轮消息传递捕获多跳邻域结构，兼顾局部与全局信息；③实现低维（32‑256）下仍保持高预测精度和与图编辑距离高度相关的距离结构；④完全无训练，推理效率高。

**🔧 技术方法**

技术手段包括：高维向量（HRR）编码、随机超向量字典、圆形卷积绑定与求和聚合、归一化、全图读取与全局属性编码、以及传统机器学习回归（GB、KNN、RF、NN）和贝叶斯优化的评估。

**📊 数据集**

使用公开的分子属性预测数据集（17 组）如 AqSolDB、BACE、ZINC250k、CLoGP 等，涉及溶解度、活性、能量、热力学等多种属性；同时生成用于GED相关性的分子对集合。

**📈 对比分析**

与 Morgan Fingerprint、RDKit Fingerprint 等传统指纹在不同 ML 模型下比较，HDF 在大多数组合中取得更高的 R² /更低的 MAE，尤其在低维度时优势显著；KNN 与贝叶斯优化实验显示 HDF 的距离结构更贴合图拓扑，导致更快的样本收敛。

**⚠️ 局限性**

局限性包括：未编码立体化学信息；对某些以哈希子结构为主的属性，Morgan 在足够维度时仍可超越 HDF；当前方法在极大分子或极高维度下的可扩展性未完全评估；以及整体性能仍受数据集分布的影响。

---

## 426. A Generalisation of Goursat's Algorithm for Integration in Finite Terms

**arXiv ID:** 2604.27806 | [PDF](https://arxiv.org/pdf/2604.27806v1)

**作者:** Sam Blake `[一作]` `[通讯]`, Sam Blake

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文重新阐述了Goursat的伪椭圆积分定理，并构造了其立方根对应的完全理论，给出对 Möbius 变换对称性下的充分必要条件，说明何时积分可化为初等函数，何时退化为不可化的 Abelian 积分。

**💡 创新点**

创新点在于：①将 Möbius 变换的反不变性与 Liouville 定理直接结合，形成对称群的特征分解；②揭示立方根情形中出现的三条特征子空间，其中两条可降至 genus‑0 曲线得到初等积分，唯一的中间特征子空间对应的曲线为 y³=x(x‑K)（genus‑1），其对积分的不可化性提供了清晰的几何阻碍；③给出完整的算法框架和示例，展示如何在符号计算中实现这些分解与降维。

**🔧 技术方法**

使用的技术包括：差分代数与 Liouville 定理、Möbius 变换的群作用与特征投影、Riemann–Hurwitz 公式计算曲线几何性质、代数曲线上的对称群分解、以及对立方根与平方根情况的直接比较；对最终实现还引用了 Risch–Trager–Bronstein 算法的理论基础。

**📊 数据集**

本工作主要基于符号计算示例（如 R(t)=t⁴‑…, R(t)=t³‑1 等），不涉及实验性数据集，而是提供理论证明和几条手工演示的积分例子。

**📈 对比分析**

与传统的 Risch 算法相比，本文的方法在满足对称性条件时能够在多项式时间内给出显式初等反导式，并保持几何解释的完整性；在给定的示例中，所得到的反导式比 Risch 算法产生的通用表达式更简洁、易于理解。

**⚠️ 局限性**

局限性包括：①仅适用于简单根多项式且要求 Möbius 变换存在；②对更高阶根（n>3）的完整理论尚未给出；③若积分涉及非代数扩张或不满足对称性条件，方法无法直接应用；④缺乏完整的计算机实现与复杂度分析，实际使用仍需进一步实验验证。

---

## 427. Separating Feasibility and Movement in Solution Discovery: The Case of Path Discovery

**arXiv ID:** 2604.27802 | [PDF](https://arxiv.org/pdf/2604.27802v1)

**作者:** Hanno von Bergen `[一作]` (University of Hamburg), Mai Trinh `[通讯]` (University of Hamburg)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种“有向加权两图”模型，将可行性（问题图）与移动约束（运动图）分离，研究路径发现（Path Discovery）和最短路径发现（Shortest Path Discovery）两种发现问题的复杂性。

**💡 创新点**

创新点：①首次在发现框架中引入两图分离的思想，能自然描述异向、异构及加权移动约束；②通过该模型统一并扩展了经典的代币滑动（token sliding）与代币跳跃（token jumping）模型；③系统给出了在多种参数化设定下的可判定性与不可判定性结果，揭示了“分离可行性与移动”对复杂性谱的深远影响。

**🔧 技术方法**

主要技术手段包括：
- 颜色编码（color‑coding）与路径分层展开（unraveling graph）技术用于按令牌数或路径长度实现固定参数可解；
- 结构参数化（feedback edge set、树宽、路径宽、回路数等）下的枚举与动态规划；
- 变换与子图构造（如最短路径子图、反馈边集子图）将问题转化为更易处理的实例；
- 复杂性归约技术，利用 Circulating Orientation 等 NP/ W[1]-hard 问题构造 NP‑hard 与参数化难度证明。

**📊 数据集**

本工作为理论研究，未使用实测数据集；所有实验均为多项式/指数时间复杂度分析与构造性证明；若需实验验证，建议采用人工生成的两图实例或经典图形库（如Planar、树、网格）进行验证。

**📈 对比分析**

与传统模型的比较：
- 在令牌数、跳跃模型、反馈边集等参数下，算法时间可达 FPT；
- 在树宽、距离、反馈顶点集等参数下给出 XP 算法；
- 对于计划性、无向单图、无权重情况，证明 NP‑hard 与 W[1]-hard；
- 综上，模型在多种实际约束下实现可行性，而传统单图模型往往在相同设定下不可解。

**⚠️ 局限性**

限制与未解决问题：
- 对于高树宽或高路径宽的实例，仍只能给出 XP 级别的算法；
- 对于更一般的加权或负权移动图，缺乏高效算法；
- 研究范围限于路径发现与最短路径发现，其他经典组合优化目标（覆盖、匹配、支配等）的两图发现问题仍待探讨；
- 对动态更新与在线场景的理论分析尚未给出。

---

## 428. Variational and Majorization Principles in Lattice Reduction

**arXiv ID:** 2604.27801 | [PDF](https://arxiv.org/pdf/2604.27801v1)

**作者:** Javier Blanco-Romero `[一作]` (Universidad Carlos III de Madrid), Florina Almenares Mendoza `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 1182 | [OpenAlex ID](https://openalex.org/A5034900252)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文把 LLL 降维中的每一次非退化 Lovász 交换视为 T‑transform，证明所有严格 Schur‑convex 的日志范数量在每一步严格递减，并基于此提出可调温度的深插入选择器 Thermal‑Adaptive 以及等价交换消耗最小化选择器 G‑DLLL。

**💡 创新点**

创新点在于：1）将局部交换转化为主导化的 T‑transform 统一解释 GSA 边界、方差消散与 Lovász 交换的局部特性；2）基于严格 Schur‑convex 的自适应温度（α）构造一族选择器，突破传统 SS‑GG 与 Deep‑Var 的权衡；3）提出等价交换效率目标 η，得到 G‑DLLL，显著降低等价交换数。

**🔧 技术方法**

主要技术包括主导化理论、T‑transform、Schur‑convex/concave 分析、梯度自适应 α 方案、变分最小方差分析，以及深插入算法的实现与评估。

**📊 数据集**

实验使用三类格子：Gaussian（随机正态）、q‑ary（半秩 q‑ary 结构）和 Goldstein‑Mayer（随机大素数结构），维度覆盖 20–200，30 个独立样本。

**📈 对比分析**

与标准 LLL、SS‑GG 和 Deep‑Var 在操作数、等价交换数、运行时间、根 Hermite 因子等指标上比较。Thermal‑Adaptive 在平坦格子上比 SS‑GG 低 8–15% 操作数；在 q‑ary 上保持相同；G‑DLLL 在等价交换数上降低 1–27%，但由于插入次数激增导致总时间上升。

**⚠️ 局限性**

局限性包括：1）理论基于单个交换的 T‑transform，难以直接推广到 BKZ 的重叠窗口；2）G‑DLLL 需要大量插入，导致固定成本主导，实际运行时间未提升；3）自适应 α 在极端分布下可能失效；4）未研究非可分离的严格 Schur‑convex 目标的收敛与性能。

---

## 429. How Generative AI Disrupts Search: An Empirical Study of Google Search, Gemini, and AI Overviews

**arXiv ID:** 2604.27790 | [PDF](https://arxiv.org/pdf/2604.27790v1)

**作者:** Riley Grossman `[一作]` (New Jersey Institute of Technology), Yi Chen `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 20881 | [OpenAlex ID](https://openalex.org/A5100419246)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对传统搜索（Google SERP）、AI 综述（AIO）和 Gemini 生成式搜索引擎在 14,212 个查询上的检索结果和生成文本进行大规模对比实验

**💡 创新点**

首次公开 11,500 个真实用户查询的基准数据集，并揭示生成式搜索对源网站可见度、信息来源多样性与一致性产生的深刻影响，尤其在高风险领域

**🔧 技术方法**

采用 Jaccard 相似度、Rank‑Biased Overlap (RBO)、线性概率模型等评估指标，结合 SerpAPI 与 Gemini API 自动化收集结果

**📊 数据集**

11,500 条基准查询（涵盖 ORCAS、Amazon Retail、Debate、ELI5 等类别）以及 863 条实时趋势查询

**📈 对比分析**

结果显示：AIO 出现率 51.5%（代表性查询），源列表相似度低（平均 Jaccard < 0.2，RBO < 0.27），生成式搜索更偏好 Google 自有网站且对阻止 Google AI 爬虫的网站检索率显著下降；一致性方面，生成式搜索比传统搜索更不稳定，设备和微小查询变动对结果影响更大

**⚠️ 局限性**

仅覆盖 Google 系列引擎，缺乏对其他主流生成式搜索（ChatGPT、Bing Copilot 等）的评估；数据收集依赖 API 可能忽略用户个性化、登录状态等因素；实验为一次性快照，未能捕捉快速演化的生成式搜索行为

---

## 430. On the Expressive Power of GNNs to Solve Linear SDPs

**arXiv ID:** 2604.27786 | [PDF](https://arxiv.org/pdf/2604.27786v1)

**作者:** Chendi Qian `[一作]` (RWTH Aachen University), Christopher Morris `[通讯]` (RWTH Aachen University)

**通讯引用:** 16931 | [OpenAlex ID](https://openalex.org/A5111798651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并证明了图神经网络(GNN)在求解线性半正定规划(SDP)时需要的表达能力，并设计了一种更具表达力的2维 Weisfeiler–Leman (2-WL) 风格架构来逼近最优 SDP 解决方案；

**💡 创新点**

创新点在于：①对标准 1‑WL 和 2‑WL 的不可表达性进行严格证明；②提出并证明 2‑WL 等价架构足以模拟 PDHG 求解器的迭代，并可通过学习得到高质量预测；③展示该架构可用于快速热启动传统 SDP 求解器，从而显著提升求解速度；

**🔧 技术方法**

采用的技术包括：高阶 Weisfeiler–Leman 图归约算法、基于 MPNN 的神经化 2‑WL（N2-WL）模型、边Transformer、以及与 PDHG 求解器对应的 MLP 参数化消息传递；

**📊 数据集**

使用的数据集涵盖：①合成的 SDP 近似（Max‑Cut、Max‑Clique、MIS、Vertex‑Cover、Max‑2‑SAT、LMI 控制问题），每类约 10,000 条实例；②真实 SDP 库 SdpLib（13 个 Max‑Cut 实例，规模从 50×50 到 500×500）；

**📈 对比分析**

与多种基线（1‑WL、2‑WL、δ‑2‑WL、Edge‑Transformer 等）比较，实验表明 N2‑WL 在测试误差、目标值误差和约束违规率上均最低，特别是在 Max‑Cut 上可降低约 90% 以上的误差；其热启动的 SCS 求解器速度提升可达 80% 以上；

**⚠️ 局限性**

局限性：仅针对线性 SDP（且假设最优解唯一且 Frobenius 范数最小）；模型对大规模实例的可扩展性仍需验证；过高的表达能力可能导致过拟合，且目前不具备直接求解离散优化问题的能力。

---

## 431. The Grand Software Supply Chain of AI Systems

**arXiv ID:** 2604.27781 | [PDF](https://arxiv.org/pdf/2604.27781v1)

**作者:** Carmine Cesarano `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 7030 | [OpenAlex ID](https://openalex.org/A5027206285)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构建四层架构模型，对 AI 软件供应链进行端到端分析，并测算了一个由 48 个开源项目构成的参考栈，展示其约 392M 行代码的规模。

**💡 创新点**

创新点在于首次将 AI 供应链视为完整实体，提出了可验证性、版本化、可观测性、可追溯性四大缺口，并给出了相应的研究议程。

**🔧 技术方法**

研究采用层级拆解、工件/变换/基础设施三种抽象，以及图结构分析、依赖解析与代码行计数等技术手段，对链路进行量化与可视化。

**📊 数据集**

使用的数据来源主要是公开的 48 个开源项目及其依赖（PyPI、Go、Maven 等）构成的依赖图；未使用特定机器学习数据集，而是通过对公开项目的版本与代码计数进行统计。

**📈 对比分析**

通过与传统软件供应链安全模型（SLSA、in‑toto）进行对比，阐明 AI 链路的不可重现性和多父级依赖导致的差异；性能方面主要体现在链路规模（11,508 个传递依赖、392M 行代码）而非算力指标。

**⚠️ 局限性**

局限性包括仅聚焦公开开源项目，可能低估企业私有组件的复杂度；测量方法受依赖解析工具限制；缺乏针对模型可重现性与安全性实验验证。

---

## 432. Monadic Presburger Predicates have Robust Population Protocols

**arXiv ID:** 2604.27767 | [PDF](https://arxiv.org/pdf/2604.27767v1)

**作者:** Philipp Czerner `[一作]` (Technical University of Munich), Simon Reilich `[通讯]` (Technical University of Munich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

证明了所有单变量（单模）Presburger 预言可以通过鲁棒的群体协议来决定，并研究了鲁棒性对状态复杂度的影响，证明对阈值预言的鲁棒协议需要至少 k 个状态，达到了最优。

**💡 创新点**

1) 首先证明单变量 Presburger 预言（即每个原子公式只含一个变量）的可鲁棒性；2) 提出了“最小化状态数”与鲁棒性之间的关系，给出了双指数下界，并证明阈值预言的最优鲁棒协议需要 k 个状态；3) 将原有的鲁棒性定义细化到更严格的版本。

**🔧 技术方法**

使用了群体协议的抽象模型、可计算函数的 Presburger 可判定性、阈值与模运算的分解、逃逸转移分析、上升不变性与临界输入等概念，并构造了多层协议来实现 min(x,k) 与 (x mod m) 的鲁棒计算。

**📊 数据集**

无，本文为理论研究，没有使用实验数据集。

**📈 对比分析**

通过对阈值预言的鲁棒协议与非鲁棒协议进行比较，证明鲁棒协议的最小状态数至少为 k，且与非鲁棒协议的 O(log log k) 状态数相比存在指数级差距；在鲁棒性与状态复杂度之间给出了下界和上界，证明了鲁棒性成本的双指数性。

**⚠️ 局限性**

1) 只证明了单模 Presburger 预言的鲁棒性，未覆盖所有 Presburger 预言；2) 对鲁棒性成本的估计仅给出了下界，未给出完整的上界；3) 鲁棒协议的构造仍相对复杂，实际实现与优化仍待研究。

---

## 433. Temporal Routing in Static Networks: The Schedule Completion Problem

**arXiv ID:** 2604.27757 | [PDF](https://arxiv.org/pdf/2604.27757v1)

**作者:** Michelle Döring `[一作]` (Hasso Plattner Institute), George Skretas `[通讯]` (Hasso Plattner Institute)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5006076693)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一个新的时序轨道排程问题（Temporally Edge Disjoint Schedule Completion，TEDSC），并给出了该问题在不同约束下的算法、复杂度与近似分析。

**💡 创新点**

创新点包括：①首次把时序图与传统的边无交叉路径问题结合，提出基于时间扩展的流网络求解方法；②证明了受距离/寿命限制的两种变体在参数化层面上的复杂度边界；③给出了一个 (2 - 1/h)-近似算法，利用压缩时间扩展与最小成本流实现。

**🔧 技术方法**

主要技术手段有：时间扩展（static expansion）构造时序图的等价静态图；使用带上下限的最大流/最小成本流求解；压缩间隙（gap‑compression）将长时间空白段折叠为双团层；参数化复杂度分析（k、h、D、n）与 1‑hard/XP 结果；状态图与动态规划技术实现 k‑算法。

**📊 数据集**

本文为理论研究，没有使用实际交通网络数据集；所有实验均基于构造的合成实例与理论分析。

**📈 对比分析**

对比方法：将提出的流网络算法与传统的暴力搜索/基于匹配的贪心方案进行理论比较；在近似层面与已知的 2‑近似或无近似方案进行比较；性能上，流网络算法在多项式时间内给出最优解（无约束），而约束变体在参数 k + h 下可在 2^{O(k+h)} 时间内求解，近似算法在多项式时间内产生 ≤(2 - 1/h) 的解。

**⚠️ 局限性**

局限性：①对大规模时间跨度的实例仍需依赖压缩技术，若时间标签分布极端稀疏则压缩不够有效；②受限变体在参数 n 下仍为 1‑hard，难以在图节点数为自然参数时实现有效算法；③近似比例为 (2 - 1/h)，在 h 较小（如 3）时误差仍可达 2/3；④未考虑边旅行时间非单位、起始/终点约束、站点容量等现实因素。

---

## 434. Real-Time Control of a Virtual Orchestra by Recognition of Conducting Gestures

**arXiv ID:** 2604.27957 | [PDF](https://arxiv.org/pdf/2604.27957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 435. From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation

**arXiv ID:** 2604.27969 | [PDF](https://arxiv.org/pdf/2604.27969v1)

**作者:** Guang Yang `[一作]` (Zhejiang University), Xin Xi `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了电路图到Verilog代码生成中的Mirage现象，构建了C2VEval基准，并提出了4B模型（结合标识符匿名化、拒绝增强和D-ORPO对齐），实现了可靠的视觉归纳生成。

**💡 创新点**

创新点包括：①系统揭示Mirage缺陷并通过Normal/Anony对照协议量化其影响；②设计Decision‑Focused ORPO (D‑ORPO) 以解决生成‑拒绝的梯度不平衡；③展示仅4B参数模型即可与前沿专有模型相当，证明视觉归纳真正可行。

**🔧 技术方法**

技术手段主要是：多模态LLM的混合监督微调（使用LoRA），标识符匿名化训练，拒绝增强数据生成，Decision‑Focused ORPO偏好对齐，以及基于netlistsvg的图像渲染。

**📊 数据集**

数据集：C2VEval（169条电路/Verilog对），由VerilogEval‑v2、RTLLM‑v2、ResBench和ArchXBench四大公开基准构建；训练集为约54K条电路图–Verilog对（从GitHub提取、验证可综合、匿名化后合并）。

**📈 对比分析**

评估方法：使用Pass@k（Syntax/Functional）在Normal/Anony、Original/Mirage四种模式下对比；4B模型在Normal下Functional Pass@1为46.11%，Anony为42.51%，在Anony下超过GPT‑5.4、GPT‑4o、MiMo‑v2‑omni；False Refusal率仅1.20%，Blank‑Image Refusal率≥92%。

**⚠️ 局限性**

局限性：基准规模相对有限（169条），仅覆盖Verilog；缺乏更大规模电路和其他HDL（VHDL、SystemVerilog）的验证；对误检/误拒的细粒度分析有限，未来需扩展到工业级电路和更广泛的视觉代码任务。

---

## 436. Splitting Assumption-Based Argumentation Frameworks

**arXiv ID:** 2604.27964 | [PDF](https://arxiv.org/pdf/2604.27964v1)

**作者:** Giovanni Buraglio `[一作]` (TU Wien), Stefan Woltran `[通讯]` (TU Wien)

**通讯引用:** 4917 | [OpenAlex ID](https://openalex.org/A5006053030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在Assumption-Based Argumentation (ABA) 框架中进行拆分（splitting）的技术，首先给出了针对SETAF的拆分方案，然后直接在ABA知识库上实现拆分，并进一步扩展到参数化拆分（parametrised splitting）以允许一定程度的交互。

**💡 创新点**

创新点在于：①通过对ABA知识库进行语法层面的分割与重构（reduct 与 modification），实现了高效的增量推理；②将拆分方案从图形化的SETAF迁移到规则级别的ABA；③提出了参数化拆分的概念，允许底层子框架被顶层规则攻击的有限假设，并给出了相应的修改与证明。

**🔧 技术方法**

技术方法包括：基于依赖图（dependency graph）的 SCC 分解与图割求拆分；对ABA规则集做子集拆分、删除受攻击规则并添加自攻击或新假设；在SETAF中采用链接（links）与不确定链接（undecided links）的概念；使用分割后的子框架的标准抽象论证语义（stable、preferred、grounded、complete）。

**📊 数据集**

文中未使用公开数据集，仅以人工构造的示例 ABA 框架与 SETAF 进行说明与演示。

**📈 对比分析**

由于本研究为理论性工作，尚未进行实验评估；作者在结论中提到计划在后续工作中实现算法并与无拆分执行进行对比，预期可提升约 60% 的平均速度。

**⚠️ 局限性**

局限性包括：①拆分前仍需实例化或构造依赖图，可能导致较高的预处理成本；②在极端情况下，拆分不一定能显著降低复杂度；③参数化拆分的实现会在子框架中引入额外的假设和规则，虽然数量有限，但在规模极大的实例中仍可能影响效率。

---

## 437. The Effects of Visual Priming on Cooperative Behavior in Vision-Language Models

**arXiv ID:** 2604.27953 | [PDF](https://arxiv.org/pdf/2604.27953v1)

**作者:** Kenneth J. K. Ong `[一作]` `[通讯]` (ST Engineering), Kenneth J. K. Ong (ST Engineering)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在迭代囚徒困境（IPD）实验中加入不同类型的视觉提示（行为概念图像和颜色编码奖励矩阵），研究视觉语言模型（VLM）在决策中的合作行为变化，并评估三种减缓视觉提示影响的策略。

**💡 创新点**

创新点在于：①系统对多种主流VLM（GPT‑4o、Claude‑3.5‑Haiku、Gemini‑2.0‑Flash、Qwen‑2.5‑VL、Pixtral‑12B、LLaMA‑3.2）进行可视化行为偏差评估；②首次提出并比较提示工程、链式思考（CoT）以及视觉token遮蔽三类减缓方法；③探索颜色编码奖励矩阵对模型决策的影响，揭示行为概念与颜色提示的不同敏感性。

**🔧 技术方法**

技术方法包括：使用DALL‑E 3、GPT‑4o、Imagen 3生成善/恶行为图像；构造红绿编码的奖励矩阵；设置不同温度并在200/1000轮IPD中记录 defect 率；采用配对 t‑检验、Cohen d 计算效应量、phi 系数评估颜色提示影响；利用 Prompt‑based “Ignoring the image”与 CoT 进行对比实验；对视觉token按总注意力或与指令相似度遮蔽并观察效果。

**📊 数据集**

数据集：人工生成的 30 张善/恶行为概念图像（共三种生成器），两套颜色编码奖励矩阵；IPD 实验生成的 200 轮或 1000 轮决策数据；无使用公开图像或文本数据集，全部为实验内部生成。

**📈 对比分析**

比较方法：在相同 IPD 场景下，记录有无视觉提示、不同减缓策略（Prompt‑based、CoT、视觉token遮蔽）时的 defect 率，计算 p‑值与效应量；结果显示：大多数模型对善/恶图像敏感（p<0.01，Cohen d>1.0），CoT 在 Qwen‑2.5‑VL 与 Pixtral‑12B 上显著降低偏差；颜色提示对 GPT‑4o、Gemini‑2.0‑Flash、Pixtral‑12B 影响显著（p<0.01，phi>0.15）。

**⚠️ 局限性**

限制：仅使用 AI 生成图像且仅在 IPD 这一单一决策任务中测试，难以推广到其他情境；部分模型的 defect 率接近边界导致统计偏差，影响 t‑检验和 chi‑square 的有效性；视觉token遮蔽效果不稳定，无法彻底消除颜色提示；未验证对真实世界图像或多任务的泛化能力。

---

## 438. Flying by Inference: Active Inference World Models for Adaptive UAV Swarms

**arXiv ID:** 2604.27935 | [PDF](https://arxiv.org/pdf/2604.27935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 439. SimEval-IR: A Unified Toolkit and Benchmark Suite for Evaluating User Simulators and Search Sessions

**arXiv ID:** 2604.27878 | [PDF](https://arxiv.org/pdf/2604.27878v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Saber Zerhoudi `[通讯]` (University of Passau)

**通讯引用:** 119 | [OpenAlex ID](https://openalex.org/A5077001732)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 SimEval-IR 的开源工具包和基准套件，用于统一评估交互信息检索中的用户模拟器，并区分“行为真实性”与“测试可靠性”两种评价目标。

**💡 创新点**

创新点在于：① 设计统一的会话模式与完整的失真记录机制，兼容搜索会话与对话交互；② 定义并实现三大可执行基准（行为真实性 B1、测试可靠性 B2、两者关联 B3）；③ 提供跨语言（英中）四大真实数据集的端到端基线与可复现配置，并在基准中引入严格的有效性规则和 provenance。

**🔧 技术方法**

采用的技术包括：会话 schema 标准化与适配器（loss accounting）、分布与序列距离度量（JS、Wasserstein、Fréchet、MMD）、嵌入式表示（动作序列嵌入）、分类器对抗测试、RATE 风格可靠性估计、Bootstrap 置信区间、Python 开源框架与 YAML 配置。

**📊 数据集**

使用的数据集为：TREC Session Track 2014（英），AOL-IA（英），MS MARCO ORCAS（英），MIRACL/zh（中），并提供适配器覆盖其他公开日志（TripClick、TianGong-ST 等）。

**📈 对比分析**

比较方法：对每个模拟器在同一 SERP 集合上生成会话，再通过 B1 计算多种距离与分类器 AUC，B2 计算与 qrels 评估器的 Kendall、Spearman、Pearson 等相关系数，并使用 RATE 聚合。实验结果表明：点击深度 JS 与 Fréchet 距离是最能预测测试可靠性的指标（r≈0.4~0.75），而传统的分类器“人类相似度”AUC 关联度极低（r≈0.09）。

**⚠️ 局限性**

局限性包括：仅处理离散事件，无法直接支持眼动、悬停等连续信号；依赖特定嵌入器，嵌入效果未系统比较；测试可靠性评估依赖 qrels‑nDCG@10，受标注覆盖与时间漂移影响；小样本（<200 会话）导致 B1 估计噪声大；目前基准仅覆盖点击模型与位置启发式模拟器，未包含完整的查询重写或工具使用 LLM 代理。

---

## 440. Language Models Refine Mechanical Linkage Designs Through Symbolic Reflection and Modular Optimisation

**arXiv ID:** 2604.27962 | [PDF](https://arxiv.org/pdf/2604.27962v1)

**作者:** João Pedro Gandarela `[一作]` (Idiap Research Institute), André Freitas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用大语言模型与数值优化器相结合，采用符号提升技术把模拟输出转换为可语言模型理解的离散描述，实现机械连杆的拓扑搜索与连续参数拟合的分离闭环迭代设计。

**💡 创新点**

创新点在于将符号提升操作与多代理闭环反思机制结合，形成可解释的结构诊断与修正流程，使语言模型在无微调的情况下具备工程级的结构推理与错误纠正能力。

**🔧 技术方法**

技术包括：大语言模型（Llama 3.3 70B、Qwen 3 4B、Qwen 3 MoE 30B-A3B）、符号提升算子（将数值轨迹压缩为运动标签、时序谓词和结构诊断）、动力学模拟器、基于PSO/Grid的参数优化器、四阶段多代理（Topology、Critic、Planner、Refiner）管线。

**📊 数据集**

数据集由六个工程相关目标曲线组成：抛物线、NACA翼型、直线、椭圆、圆形和伯努利双螺线。

**📈 对比分析**

方法与基线（Enum+GA、单一模型无符号提升）在相同搜索预算下比较，实验表明：78.6%的迭代轨迹单调改进，几何误差可降低至68%，结构有效性可提升134%，相较基线可降低30–57%的误差。

**⚠️ 局限性**

局限性在于仅验证平面连杆，符号词汇有限，对三维或动态机制尚未扩展，且未对语言模型进行领域微调。

---

## 441. LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning

**arXiv ID:** 2604.27960 | [PDF](https://arxiv.org/pdf/2604.27960v1)

**作者:** Adam Ishay `[一作]` (Arizona State University), Joohyung Lee `[通讯]` (Arizona State University)

**通讯引用:** 3224 | [OpenAlex ID](https://openalex.org/A5100343841)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将自然语言问题自动翻译为答案集程序（ASP）的框架 LLM+ASP，并通过自我纠错循环使 LLM 能够在没有任务特定工程的情况下生成可执行的 ASP 程序。

**💡 创新点**

核心创新在于：① 采用非单调逻辑 ASP 取代传统的单调 SMT，天然支持默认规则与异常；② 引入基于求解器反馈的自动自我纠错机制，显著提升程序质量；③ 发现紧凑的参考指南比冗长文档更有效，揭示“上下文退化”现象。

**🔧 技术方法**

技术手段包括：LLM 生成代码、ASP 解释器 clingo、迭代纠错循环、上下文学习（compact 参考指南），以及与多种 LLM（Gemini 2.5 Pro/Flash、o4-mini、DeepSeek R1/V3）的交互。

**📊 数据集**

评测数据集涵盖六类推理任务：约束满足（ZebraLogic、SudokuBench、Mystery Blocksworld）、非单调推理（MultiLogicNMR）、规划与优先级冲突（BoardgameQA）等，合计 1000 题目。

**📈 对比分析**

与单纯 LLM 推理相比，LLM+ASP 在平均准确率上提升约 52%（从 51.3% 到 78.0%）；在最难子集（ZL‑XXL）上提升近 200%；在非单调任务上 ASP 远优于 SMT（约 90% 对比 35%），而在约束满足/规划任务上 SMT 表现更好但整体不如 ASP；迭代自纠正是主要性能驱动，平均仅 2 次修订即可完成。

**⚠️ 局限性**

局限性包括：对强大推理 LLM 的高度依赖；对无解程序的调试反馈不足；计算成本与延迟较高，主要适合批量处理；仅在结构化、可归约为逻辑约束的问题上有效，对开放式自然语言推理的适用性未知。

---

## 442. TripVVT: A Large-Scale Triplet Dataset and a Coarse-Mask Baseline for In-the-Wild Video Virtual Try-On

**arXiv ID:** 2604.27958 | [PDF](https://arxiv.org/pdf/2604.27958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 443. GUI Agents with Reinforcement Learning: Toward Digital Inhabitants

**arXiv ID:** 2604.27955 | [PDF](https://arxiv.org/pdf/2604.27955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 444. A Collective Variational Principle Unifying Bayesian Inference, Game Theory, and Thermodynamics

**arXiv ID:** 2604.27942 | [PDF](https://arxiv.org/pdf/2604.27942v1)

**作者:** Djamel Bouchaffra `[一作]` (University of Paris-Saclay), Hanane Azzag `[通讯]` (Sorbonne Paris Nord University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了统一的游戏理论自由能原理，将多智能体的局部变分推理映射为隐式随机博弈，并用 Harsanyi 分解量化协同效应；

**💡 创新点**

创新点在于证明集体自由能极小点对应 ε‑Nash 均衡，给出自由能与博弈均衡的正式对应关系，并提出了可量化的协同自由能分量；

**🔧 技术方法**

采用变分推理、博弈论、Möbius 逆变换与 Gibbs 分布等理论工具，利用 Gaussian 合成联盟模型解析计算；

**📊 数据集**

通过三类仿真数据集验证：神经元集合（N=50）、鱼群（N=30）以及多智能体强化学习团队（N=5）；

**📈 对比分析**

与传统模型对比的指标是影响力（Shapley 值）随感知精度的变化，三组实验均得到倒U形曲线，拟合优度 R²≥0.88，证明预测成立；

**⚠️ 局限性**

局限性包括需要枚举所有子集导致指数复杂度、假设已知生成模型和环境为稳定，且对异质大规模系统缺乏可扩展近似方法。

---

## 445. DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models

**arXiv ID:** 2604.27929 | [PDF](https://arxiv.org/pdf/2604.27929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 446. Training-Free Tunnel Defect Inspection and Engineering Interpretation via Visual Recalibration and Entity Reconstruction

**arXiv ID:** 2604.27928 | [PDF](https://arxiv.org/pdf/2604.27928v1)

**作者:** Shipeng Liu `[一作]` (Xi'an University of Architecture and Technology), Zhanping Song `[通讯]` (Xi'an University of Architecture and Technology)

**通讯引用:** 4271 | [OpenAlex ID](https://openalex.org/A5012030209)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练‑free隧道缺陷检测框架TunnelMIND，利用视觉‑语言模型生成语义锚点，再通过DINOv3的密集视觉重校准得到更可靠的空间支持，使用SAM实现精细掩码分割，并将掩码重构为包含类别、位置、几何、严重度和上下文的工程实体，最终通过检索驱动的LLM生成可直接用于工程文档的报告。

**💡 创新点**

①跨模型视觉重校准：把语言模型的粗糙锚点映射到DINOv3的视觉一致空间，提升空间可靠性；②实体化工程解释：将分割掩码转化为结构化的工程实体，兼顾测量与严重度判定；③检索驱动的报告生成：将实体映射到知识库检索，再由LLM生成安全、可追溯的工程说明。

**🔧 技术方法**

Qwen3‑VL 视觉‑语言模型；DINOv3 作为密集特征检索空间；SAM 进行基于提示的精细分割；检索向量数据库 + 文本嵌入模型；LLM（DeepSeek、Qwen3、Kimi）在检索上下文下生成报告。

**📊 数据集**

六类隧道相关任务数据集：可见缺陷（Visible）、GPR 隐缺陷（GPR）、路面缺陷（Road）、岩屑分割（Rock）、PPE 检测（PPE）以及工人姿态（Pose）; PPE 数据集来自公开资料，Pose 采用 MS‑COCO 采样，其他数据为作者自行采集。

**📈 对比分析**

与监督任务模型（YOLOv11/12、RT‑DETRv4、DEIMv2）对比，监督模型仍保持最佳；与训练‑free基线（GroundingDINO、Qwen3‑VL、Rex‑Omni）对比，TunnelMIND 在 Visible、GPR、Road 上的 F1 分别为0.68、0.78、0.72，硬负样本平均 F1 0.66，显著优于基线。实体级评价显示 TunnelMIND 在长度、宽度、面积误差降低、位置准确率提升，严重度 F1 提升。检索驱动报告中 Hit@3 84.7、Usefulness 4.3、Safety 4.7、Clarity 4.1，均超过基线。

**⚠️ 局限性**

①对重叠或交叉缺陷的实例分割不够精准；②GPR 任务对参考示例的选择敏感；③多模型序列推理耗时较长，适合离线辅助而非实时控制；④对姿态等需要专门结构建模的任务效果不佳。

---

## 447. Generate Your Talking Avatar from Video Reference

**arXiv ID:** 2604.27918 | [PDF](https://arxiv.org/pdf/2604.27918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 448. Geometry-Calibrated Conformal Abstention for Language Models

**arXiv ID:** 2604.27914 | [PDF](https://arxiv.org/pdf/2604.27914v1)

**作者:** Rui Xu `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45465 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后置方法 Conformal Abstention (CA)，让大型语言模型在检测到知识缺失时自我放弃回答。

**💡 创新点**

创新点在于：①将 conformal prediction 的参与和条件正确性保证迁移到开放式生成任务；②利用模型内部表示几何（MLP 贡献、嵌入旋转、各层对齐）来校准置信度，使其更贴合模型的真正不确定性；③不需要额外训练，仅通过后置阈值即可实现。

**🔧 技术方法**

使用 conformal prediction 形式的概率保证、token 级几何特征（Ω、Θ、Φ^in、Φ^out）、Mahalanobis 距离校准、XGBoost 进行置信度映射，以及标准 perplexity 作为基准置信度。

**📊 数据集**

在六个公开数据集上验证：Natural Questions (NQ)、TruthfulQA (TQA)、Simple Questions Wiki (SQW)、SciQ、GSM8K、CommonsenseQA (CQA)。模型包括 Gemma‑3‑4B‑Instruct、LLaMA‑3.2‑3B‑Instruct 与 LLaMA‑3.8‑B‑Instruct。

**📈 对比分析**

与多种 UQ 方法（Perplexity、SAR、Semantic Entropy、Focus、Attention、Eigen、ATRMD、P(True) 等）比较。CA 在参与率 10%–90% 范围内均实现最高的条件正确率（约 75%）并在 AUROC/AUPRC 上显著优于基线；在大多数数据集上超过 80% 的准确性阈值。

**⚠️ 局限性**

局限性包括：① 需要额外的校准集和阈值调优；② 计算量受内部几何特征提取影响；③ 仅适用于自回归 LLM 结构，难以推广到其他生成模型；④ 依赖于模型对内部 MLP 记忆的假设，若模型架构不同可能效果降低。

---

## 449. From Unstructured Recall to Schema-Grounded Memory: Reliable AI Memory via Iterative, Schema-Aware Extraction

**arXiv ID:** 2604.27906 | [PDF](https://arxiv.org/pdf/2604.27906v1)

**作者:** Alex Petrov `[一作]` (Xmemory), Dima Korolev `[通讯]` (Xmemory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于模式的持久化记忆架构，利用迭代写入路径将自然语言事件拆解为对象检测、字段检测与字段值抽取，并通过验证门、局部重试以及状态化提示控制保证事实准确性。

**💡 创新点**

创新点在于将检索式记忆的“读取”责任迁移到“写入”阶段，采用模式契约与逐步验证的写入流程，使记忆系统从近似检索转变为可验证的记录系统，显著提升了对更新、删除、聚合、关系与缺失查询的可靠性。

**🔧 技术方法**

技术包括：多阶段结构化抽取（object‑field‑value）、验证门与局部重试、请求/会话/主内存三层上下文管理、提示引擎的状态化控制、基于SQL或等价查询语言的受限读取、以及LLM评估器（judge‑in‑the‑loop）进行抽取纠错。

**📊 数据集**

使用的数据集包括：保险索赔文档集（四字段子结构）、四个业务域（企业、教育、医疗、金融）构成的端到端记忆基准、以及以自然语言生成的Splitwise聚会事件数据集。

**📈 对比分析**

与现有产品化记忆系统（Cognee、Mem0、Supermemory、Zep 等）对比，本文提出的 xmemory 在端到端基准上达 97.10% F1，远高于 80.16%–87.24% 的对手；在结构化抽取任务中实现 90.42% 的对象级准确率（相比 79–89% 的前沿模型）；在 Splitwise 应用级评测中获得 95.2% 的准确率。

**⚠️ 局限性**

局限性包括：基准任务结构化、仅针对需要稳定记录和状态计算的工作负载；对比仅为产品化系统，未做组件级消融；评估依赖 LLM 判读器，可能受提示敏感度影响；模式设计仍需人工或代理协助，迁移与回填的自动化不完善；验证门无法捕捉所有语义错误，仍可能引入错误事实。

---

## 450. Graph World Models: Concepts, Taxonomy, and Future Directions

**arXiv ID:** 2604.27895 | [PDF](https://arxiv.org/pdf/2604.27895v1)

**作者:** Jiawei Liu `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 8785 | [OpenAlex ID](https://openalex.org/A5051340429)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对图形世界模型（Graph World Models, GWM）进行系统化定义、统一分类与综述，提出基于空间、物理与逻辑关系的三层RIB（Relational Inductive Bias）分类，并探讨其在导航、模拟与推理等场景中的设计原则与现有方法

**💡 创新点**

首次以RIB为核心构建统一的GWM框架与三层分类，明确不同结构偏置对应的功能与挑战，为后续研究提供共通语义与评估维度

**🔧 技术方法**

采用图结构抽象与关系转移的理论框架，结合图神经网络、对比学习、逆动力学、知识图谱、因果发现等技术手段，对现有连接器、模拟器与推理器三类模型进行整理与归纳

**📊 数据集**

综述涵盖多种公开数据集（机器人导航、自动驾驶、视频生成、文本交互、3D点云等）及对应任务，未针对单一数据集展开实验，而是对各类论文使用的数据与任务进行了归类

**📈 对比分析**

通过文献对比和结构化表格展示各类GWM的核心设计、使用技术、应用场景与已报道的性能指标，指出不同RIB在长期规划、物理模拟与因果推理上的优势与不足，整体性能评估基于文献报告，缺乏统一基准

**⚠️ 局限性**

缺乏统一、系统化的评测标准；现有模型多为确定性预测，难以处理高熵动态；连接器模型对图结构扩张与实时更新不够灵活；推理器依赖LLM生成关系易出现逻辑与物理不一致；整体难以平衡语义灵活性与结构严谨性，导致跨任务泛化与长期学习受限

---

## 451. A Monadic Implementation of Functional Logic Programs

**arXiv ID:** 2604.27863 | [PDF](https://arxiv.org/pdf/2604.27863v1)

**作者:** Michael Hanus `[一作]` (Kiel University), Finn Teegen `[通讯]` (Kiel University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在本文中，作者实现了一套完整的 Curry 编译器，采用单子化（monadic transformation）将 Curry 程序转换为 Haskell 代码，并在运行时实现了**记忆化拉取表（Memoized Pull‑Tabbing, MPT）**以支持懒惰与非确定性计算。该实现不仅覆盖了 Curry 的基础特性，还加入了自由变量、统一、函数模式、封装搜索、合理搜索（fair search）等高级特性，并提供了对确定性子计算的优化。

**💡 创新点**

创新点主要有：
1) 将 MPT 的记忆化机制移植到纯函数式单子框架中，避免了对命令式语言的依赖；
2) 通过单子化实现对 Curry 语言所有功能的统一实现，只需修改单子即可添加新特性；
3) 在确定性子计算上做了静态/动态分析，动态生成并利用确定性构造体以减少不必要的单子包装；
4) 通过深度、宽度、并行等多种搜索策略实现了公平搜索，并在单子层面实现了封装搜索。

**🔧 技术方法**

使用的技术包括：
- 单子化（Monadic transformation）与 StateT+Tree 构成的 Curry 单子；
- IORef 与 unsafePerformIO 实现全局唯一 ID 与记忆化表；
- 共享函数 share，支持 call‑time choice 的共享与记忆；
- 统一与自由变量通过 FLVal、Narrowable、Unifiable 类实现；
- 正常形式类 NormalForm 与 normalFormCurry 用于封装搜索；
- 并发线程 + MVar + Chan 实现公平搜索；
- 从 Haskell 类型映射回 Curry 类型的 FromHs 与 HsEquivalent 机制实现确定性优化。

**📊 数据集**

评测使用了 Curry 研究社区常用的基准程序：
- addNum5 / addNum10（非确定性数值生成），
- selectN（列表随机选择），
- yesSharingND / noSharingND（共享/非共享 800th 质数），
- permSort、sortPrimes（排列与排序），
- naiveReverse（大列表逆序），
- queens10（10 皇后问题）。
与 PAKCS（Prolog backtracking）、KiCS2（Haskell 纯粹拉取表）和 Curry2Go（Go imperative 记忆化）在同一硬件上对比。

**📈 对比分析**

比较方法：在 Linux Debian 12 + GHC 9.4.5 + Intel i7-7700K 上，分别用 time 命令测三次平均时间；所有实现使用深度优先搜索。结果显示：
- 对于需要记忆化的基准（addNum10、selectN 等），MPT（尤其是带确定性优化版本）速度与 KiCS2 相当，甚至在某些基准上更快；
- 对于纯确定性程序，MPT+det 取得与 KiCS2 相当或略优的性能；
- 对于包含大量非确定性计算和递归（sortPrimes、queens10），KiCS2 在不做确定性优化时更快；
- 与 PAKCS 比较时，MPT 在大多数基准上显著优于 PAKCS（尤其是数值计算），但在部分基准（如 addNum5）仍略慢；
- 总体来看，MPT+det 在多数基准中位列前三。

**⚠️ 局限性**

限制与待改进之处：
- 目前实现依赖 unsafePerformIO 与 IORef，虽然安全包装但仍不够纯粹；
- 与 KiCS2 直接插件实现相比，GHC 的需求分析未能充分优化，导致某些数值密集型基准慢于原始实现；
- 确定性子计算的检测主要基于静态分析，无法覆盖所有情况，仍需手工注解或更精细的分析；
- 对于高度嵌套的 case 表达式，确定性优化会导致分支数倍增，可能造成代码膨胀；
- 并发公平搜索依赖线程调度，可能在资源受限环境下产生额外开销；
- 目前未支持对外部 Prolog 或低级语义的直接交互，限制了与现有工具链的兼容性。

---

## 452. Computing Witnesses Using the SCAN Algorithm

**arXiv ID:** 2604.27939 | [PDF](https://arxiv.org/pdf/2604.27939v1)

**作者:** Fabian Achammer `[一作]` (TU Wien), Renate A. Schmidt `[通讯]` (University of Manchester)

**通讯引用:** 9111 | [OpenAlex ID](https://openalex.org/A5025900214)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在原有的SC​AN算法基础上，提出了WSCAN算法，能够在求解第二阶量化消元（SOQE）时同时给出谓词变量的实例（witness），从而得到等价的一阶公式。

**💡 创新点**

创新点主要体现在：①引入WSCAN，扩展SCAN处理更一般的WSOQE问题；②给出构造witness的递归/底层闭包方法；③用大修正点（greatest fixpoint）表达无穷witness；④在特定图条件下进一步构造一阶witness；④兼容完整等价推理及第一阶背景理论。

**🔧 技术方法**

技术方法包括：逻辑归约（SC​AN与WSOQE的推导框架）；约束产生与约束消除（constraint elimination）；递归闭包与大修正点（gfp）逻辑；谓词子模化（purification subsumption）图与无环性判定；以及等价性证明和实现的推导系统。

**📊 数据集**

本文未使用标准数据集，而是以理论推导和符号推理为主，提供了GCC++/CLP实现原型，未给出实验数据。

**📈 对比分析**

方法与SCAN的比较主要在于能否给出witness：WSCAN能在原SCAN不产生有限witness的情形下得到（可通过大修正点或一阶witness替代）等价的一阶公式，具体性能指标未给出实验评估。

**⚠️ 局限性**

限制在于：当处理的指向子句导致闭包无限时，witness可能是无限的；需要满足k‑acyclic purification条件才能得到一阶witness；某些SOQE实例即使SCAN正确也无法用该方法构造有限witness。

---

## 453. Enhancing multimodal affect recognition in healthcare: the robustness of appraisal dimensions over labels within age groups and in cross-age generalisation

**arXiv ID:** 2604.27938 | [PDF](https://arxiv.org/pdf/2604.27938v1)

**作者:** Hippolyte Fournier `[一作]` (University of Grenoble Alpes), Fabien Ringeval `[通讯]` (University of Grenoble Alpes)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究者扩展了THERADIA‑WoZ情绪识别语料库，新增年轻成人数据，并比较基于表观维度和类别标签的多模态情绪识别模型，评估其跨年龄泛化能力。

**💡 创新点**

创新点在于将表观理论维度与类别标签并行评估、引入年轻成人数据实现跨年龄比较，并证明维度模型在跨年龄场景下更稳健。

**🔧 技术方法**

采用深度学习特征提取（Wav2Vec2、BERT、CLIP）与传统专家特征（MFB、TF-IDF、FAU），并利用多模态决策融合、GRU、OLS回归及贝叶斯线性模型进行训练与评估。

**📊 数据集**

使用原始THERADIA‑WoZ老年人语料库（约30h）与新增的52名年轻成人语料库（约30h），共计约13540段录音，分为情绪标签和五维表观维度。

**📈 对比分析**

通过within‑corpus、cross‑corpus和mixed‑corpus三种训练策略，采用CCC与贝叶斯因子评估，结果显示维度模型在所有策略下均优于标签模型，跨年龄测试中标签模型接近随机，而维度模型保持显著性能。

**⚠️ 局限性**

局限性包括年轻成人数据情绪表达波动大导致模型性能相对下降、混合语料未提升泛化、以及对高阶情绪维度（如Novelty）预测仍困难。

---

## 454. Taming the Centaur(s) with LAPITHS: a framework for a theoretically grounded interpretation of AI performances

**arXiv ID:** 2604.27927 | [PDF](https://arxiv.org/pdf/2604.27927v1)

**作者:** Matteo Da Pelo `[一作]` (University of Cagliari), Antonio Lieto `[通讯]` (University of Salerno)

**通讯引用:** 1350 | [OpenAlex ID](https://openalex.org/A5033602352)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并应用了LAPITHS框架，系统评估大型语言模型（尤其是Centaur）在行为预测与认知可解释性之间的关系，并通过实验复制其在两步任务和fMRI对齐中的表现。

**💡 创新点**

创新点在于将理论分析与最小认知网格（Minimal Cognitive Grid）结合，明确区分“行为相似性”与“认知可解释性”，并提供定量指标（FSR、Generalizability、Performance Match）来评估模型的结构与功能一致性。

**🔧 技术方法**

使用了QLoRA参数高效微调技术、检索增强生成（RAG）与多种大型语言模型（Llama 3.1 70B、Llama Maverick、GPT-4o、GPT-5.1、Gemini‑2.5 Pro、DeepSeek‑R1）以及正则化线性回归来进行fMRI预测。

**📊 数据集**

主要数据集包括Psych‑101（超过一千万心理实验决策数据）和feher2023rethinking的两步任务fMRI记录，用以评估行为一致性和神经对齐。

**📈 对比分析**

方法上与Centaur直接比较，利用负对数似然（NLL）衡量决策拟合，使用Pearson相关与余弦相似度评估ROI预测。结果显示Centaur虽在NLL上略优，但与使用RAG的普通LLM相差无统计学意义；在fMRI对齐上，非专门训练的LLM也能达到高相关，说明仅凭统计对齐并不能证明认知一致性。

**⚠️ 局限性**

局限性包括：1）过度依赖行为匹配而忽视结构差异，导致结论受限于逆推断；2）功能/结构比（FSR）低，说明模型缺乏关键认知机制；3）缺乏对学习动态、记忆限制等过程层面的评估；4）fMRI预测仅基于相关性，未考虑幅度校准与因果解释；5）实验仅覆盖有限任务和模型，未检验更广泛的认知范畴。

---

## 455. Beyond Semantics: Measuring Fine-Grained Emotion Preservation in Small Language Model-Based Machine Translation

**arXiv ID:** 2604.27920 | [PDF](https://arxiv.org/pdf/2604.27920v1)

**作者:** Dawid Wisniewski `[一作]` (Poznań University of Technology), Igor Czudy `[通讯]` (Poznań University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语种（德、法、意、英、波）回译实验中，评估了三种小型语言模型（EuroLLM、Aya Expanse、Gemma）在保留GoEmotions细粒度情感标签方面的表现，并测试了情感感知提示和三种情感分类器（BERT、DeBERTa、ModernBERT）的效果。

**💡 创新点**

①首次系统比较情感感知提示对回译情感保留的影响；②引入ModernBERT作为情感评估基准，揭示其相对BERT和DeBERTa更稳健的细粒度情感识别；③提供情感类别与语言层面的细化损失分析，突出高强度情感（如欲望、恐惧、愤怒）易受损。

**🔧 技术方法**

回译管道（English→目标语言→English）使用4-bit AWQ量化的vLLM推理；情感感知提示；三种预训练情感分类器；COMET-22-da评估语义保真度；Iterative Stratification拆分，宏观F1指标计算。

**📊 数据集**

GoEmotions数据集（原57k条，过滤后22个情感标签），用于训练情感分类器并在回译后评估情感丢失。

**📈 对比分析**

通过宏观F1下降（Δ_aff）平均值比较模型/提示/分类器组合，平均情感丢失仅为2.9–4.9个百分点；EuroLLM + 情感感知提示 + ModernBERT获得最低Δ_aff（约0.029）；情感提示对大部分模型效果有限，偶有微小提升；语义质量（COMET）几乎无变化。

**⚠️ 局限性**

①对高强度情感的保留仍不理想，尤其波兰语；②情感提示对某些模型反而略降性能；③仅评估单轮回译，未覆盖多模态或对话场景；④分类器可能对细粒度情感表现过度敏感，导致测量偏差。

---

## 456. Physical Foundation Models: Fixed hardware implementations of large-scale neural networks

**arXiv ID:** 2604.27911 | [PDF](https://arxiv.org/pdf/2604.27911v1)

**作者:** Logan G Wright `[一作]` (Yale University), Peter L. McMahon `[通讯]` (Cornell University)

**通讯引用:** 8136 | [OpenAlex ID](https://openalex.org/A5064735957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了“物理基础模型”（PFM）的概念，即将大型基础模型的参数硬编码到物理硬件中，通过硬件自然动力学实现推理；并以光学三维纳米结构为例，给出了后向估算的规模、能耗与时延等指标；

**💡 创新点**

创新点在于抛弃可编程权重，直接在物理介质中实现整个神经网络计算，借助硬件自然动力学大幅降低数据移动成本，实现能耗、时延和参数密度的阶跃提升；

**🔧 技术方法**

采用光学/纳米电子等模拟物理计算的理论模型，利用逆向设计、三维纳米结构、光传播与非线性光学等技术进行后向估算；

**📊 数据集**

本文为概念性工作，未使用具体数据集，仅以当前已知的 GPT‑5、Gemini‑3、Llama‑4 等模型规模做假设；

**📈 对比分析**

通过理论比较将光学 PFM 的能耗与时延与数字电子基准（GPU）进行对比，结果表明能耗可降低 10⁴–10⁸ 倍，时延可提升 10³–10⁵ 倍；

**⚠️ 局限性**

主要局限包括：缺乏可实现的硬件制程与大规模逆向设计；训练与校准方法不明确；工艺变异、损伤与缺陷带来的可靠性与可修复性问题；以及在现实系统中的集成、接口与可编程性缺失等挑战。

---

## 457. CoNewsReader: Supporting Comprehensive Understanding and Raising Critical Thoughts on Social Media News Through Comments

**arXiv ID:** 2604.27905 | [PDF](https://arxiv.org/pdf/2604.27905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 458. HiMix: Hierarchical Artifact-aware Mixup for Generalized Synthetic Image Detection

**arXiv ID:** 2604.27903 | [PDF](https://arxiv.org/pdf/2604.27903v1)

**作者:** Shuchang Zhou `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112237 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HiMix 框架，通过混合训练样本和层次化特征融合来提升合成图像检测的跨源泛化能力。

**💡 创新点**

创新点在于（1）Mixup‑Driven Distributional Augmentation（MDA）在真实与伪造图像间构造连续过渡样本，显著扩展低置信区间；（2）Pixel‑wise mixup 进一步弱化高层语义干扰，突出低层伪造痕迹；（3）Hierarchical Artifact‑aware Representation（HAR）结合多尺度区域池化、跨层融合与粗细粒度融合，捕获全局与局部伪造特征；（4）采用 LoRA 对 CLIP 编码器进行轻量级适配，保持预训练语义能力同时突出伪造相关信息。

**🔧 技术方法**

技术方法包括 Mixup 数据增强、CLIP ViT‑L/14 编码器、LoRA 低秩适配、Hierarchical Region Pooling（HiRP）、Cross‑Layer Fusion（CLF）以及 Cross‑Granularity Fusion（CGF）等。

**📊 数据集**

使用的主要数据集有：GenImage（8 大生成模型的真实/伪造图像）、Diffusion 族数据集（包含多种扩散模型）、TwinSynths‑GAN（高保真 GAN 合成）、ForenSynths（多 CNN 生成模型）以及公开的 ImageNet、LSUN 等作为真实图像来源。

**📈 对比分析**

与现有方法（如 SAFE、CoD、AIDE、VIB、C2P‑CLIP 等）对比，HiMix 在 GenImage 上取得平均 Acc 97.6%、AP 99.7%，在 Diffusion 族上平均 Acc 98.3%、AP 99.9%，在 TwinSynths‑GAN 上 AP 90.6，均显著优于 SOTA 并保持低误报率。

**⚠️ 局限性**

主要局限包括：需要较大 CLIP 预训练模型，导致推理速度相对较慢；混合系数 α 的设置对性能敏感；在极端后处理（如强 JPEG 压缩或高强度模糊）下性能仍会有所下降，且对未知生成模型的鲁棒性尚需进一步验证。

---

## 459. Can We Volunteer Out of the Peer Review Crisis?

**arXiv ID:** 2604.27900 | [PDF](https://arxiv.org/pdf/2604.27900v1)

**作者:** Theo Tang `[一作]` (Monash University), Julian Garcia `[通讯]` (Monash University)

**通讯引用:** 1125 | [OpenAlex ID](https://openalex.org/A5083299717)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并分析一种自愿抽签机制，旨在通过随机预审拒稿来降低同行评审负担，从而提升已发表论文的质量。

**💡 创新点**

创新点在于将同行评审视为公共物品问题，引入抽签机制以实现集体行动；通过游戏理论推导出Nash均衡与社会最优阈值，并证明在足够噪声弹性和研究者的认知关怀下可实现显著质量提升。

**🔧 技术方法**

使用连续近似与蒙特卡洛模拟相结合的数学模型，建立抽签、噪声弹性、期望接受率与作者效用之间的关系；采用游戏理论求解自愿参与的阈值策略和编辑的最佳噪声调节。

**📊 数据集**

主要采用理论模拟数据；对噪声弹性参数使用ICLR 2017–2025提交数据估计，除此之外并未使用公开数据集进行实证验证。

**📈 对比分析**

通过与无抽签基线以及社会最优方案对比，展示在不同噪声弹性和选择率下抽签可以提升平均已接受论文质量；在噪声弹性较高时，收益可超过50%，但在噪声弹性较低时收益趋近于零。

**⚠️ 局限性**

局限性包括：未考虑自我筛选（self‑screening）与动态投递行为的反馈；对“认知关怀”比例的假设在实际中难以测量；模型假设评审噪声仅受负荷影响，忽略专家质量差异；在大规模社区中个人对总噪声的影响被稀释，导致抽签效果显著下降。

---

## 460. In-Context Prompting Obsoletes Agent Orchestration for Procedural Tasks

**arXiv ID:** 2604.27891 | [PDF](https://arxiv.org/pdf/2604.27891v1)

**作者:** Simon Dennis `[一作]` (i14), Hao Guo `[通讯]` (i14)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在三种不同的流程任务（旅游预订、Zoom技术支持、保险理赔）中，对比使用 LangGraph 框架进行外部流程控制与将完整流程直接放入系统提示（in‑context）让模型自行控制，评估两种方法在任务完成度、信息准确性、一致性、优雅处理和自然度上的表现；

**💡 创新点**

证明在 frontier LLM（Claude Sonnet）下，外部 orchestration 结构不仅无提升，反而降低质量，主要原因是思路碎片化、引入额外失败模式和对模型的限制；

**🔧 技术方法**

使用 Claude Sonnet 4.5 作为模型，LangGraph 作为外部 orchestrator，GPT‑4.1 作为独立评审；

**📊 数据集**

利用人工构建的三大流程图和 Claude‑based 生成的模拟用户对话，共 1,200 条对话（每种条件每域 200 条）；

**📈 对比分析**

通过 LLM‑as‑Judge 评估，统计五项指标的 1‑5 分，结果显示在所有指标和域中，in‑context 方法平均提升 0.16‑0.63 分；此外，in‑context 的失败率仅 5‑11.5%，LangGraph 为 9‑24%；

**⚠️ 局限性**

实验仅覆盖人工流程与模拟用户，未验证真实生产数据；in‑context 需要流程完整放入上下文，受限于上下文窗口；成本相对略高（约 1.3‑1.4 倍）；仅评估 LangGraph，其他框架可能略有差异。

---

## 461. Building Persona-Based Agents On Demand: Tailoring Multi-Agent Workflows to User Needs

**arXiv ID:** 2604.27882 | [PDF](https://arxiv.org/pdf/2604.27882v1)

**作者:** Giuseppe Arbore `[一作]` (Politecnico di Torino), Luigi De Russis `[通讯]` (Politecnico di Torino)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5023909289)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于实时 persona 的多智能体系统生成管线，实现了在运行时根据用户特征和任务需求动态合成智能体角色与行为。

**💡 创新点**

创新点在于把智能体角色、交互方式和协调模式从设计时固定迁移到运行时可生成，从而实现更高的个性化与适应性。

**🔧 技术方法**

主要技术包括 LLM 角色生成、查询分析与任务分解、PersonaCraft 与 AgentFactory、协同调度与答案聚合等。

**📊 数据集**

论文未使用公开数据集，而是通过模拟交互案例进行验证。

**📈 对比分析**

实验未给出定量对比，评估主要以案例演示为主，缺乏客观性能指标。

**⚠️ 局限性**

局限性包括缺乏大规模实验验证、对生成 persona 的一致性和可靠性未做充分评估，以及系统对复杂任务的扩展性未知。

---

## 462. D-Rex : Diffusion Rendering for Relightable Expressive Avatars

**arXiv ID:** 2604.27871 | [PDF](https://arxiv.org/pdf/2604.27871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 463. Simulating clinical interventions with a generative multimodal model of human physiology

**arXiv ID:** 2604.27899 | [PDF](https://arxiv.org/pdf/2604.27899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 464. Synthetic Biological Intelligence: System-Level Abstractions and Adaptive Bio-Digital Interaction

**arXiv ID:** 2604.27933 | [PDF](https://arxiv.org/pdf/2604.27933v1)

**作者:** Martin Schottlender `[一作]` (Dresden University of Technology), Pit Hofmann `[通讯]` (Dresden University of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了合成生物智能（SBI）的研究进展，提出了统一的系统层面抽象框架 ABNIA，涵盖编码/刺激、适应性生物神经子系统、观测/解释以及反馈控制四大模块，并梳理了现有商业平台、编码/解码技术、噪声源、性能指标与标准化挑战。

**💡 创新点**

创新点在于：①将 SBI 定义为闭环生物‑数字交互系统；②设计了 ABNIA 框架，统一描述编码、刺激、解码和反馈的多时尺度交互；③系统性归纳了编码/调制、噪声模型、解码策略与评估指标；④提出了标准化与可复制性路径，突出多平台可比性。

**🔧 技术方法**

使用的技术包括：iPSC‑衍生的二维/三维神经类器官与 MEA 接口；电刺激与化学/光/热调制；多种编码方案（速率、时间、空间、符号、储水库等）；尖峰排序与特征提取（滤波、阈值、PCA、波形匹配、EM、SVM、ANN 等）；强化学习/自适应控制框架用于闭环反馈；以及云端平台（Cortical Cloud、NeuroPlatform）实现远程实验。

**📊 数据集**

未使用单一公开数据集，而是对多项现有 SBI 实验与平台（如 DishBrain、CL‑1、Neurorighter、Cortical Labs、FinalSpark 等）的实验数据和报告进行汇总与对比。

**📈 对比分析**

比较方法：归纳各平台的性能指标（分类准确率、任务成功率、延迟、突发率、平均放电率、互信息等），并对比不同编码/解码方案、不同神经子系统（二维 vs 三维、iPSC vs rodent）以及闭环 vs 开环控制的效果。总体性能显示：闭环 SBI 能在一定任务（如 Pong、机器人控制、信号分类）中达到 70–90% 的分类准确率或 80–90% 的任务成功率，延迟在 10–100 ms 级别，能耗低于传统硅基系统，但受制于细胞存活周期与非稳态噪声，导致跨实验的可重复性差。

**⚠️ 局限性**

主要限制：①缺乏统一标准化的实验与评估协议，导致跨平台比较受限；②神经子系统高度非稳态、漂移与可塑性，使得固定编码/解码方案难以长期有效；③细胞培养与维护成本高，实验周期短，难以实现大规模部署；④噪声源复杂（离子通道噪声、突触噪声、MEA 互相干扰），影响信号解码；⑤缺乏公开可复现的数据集与模拟工具，限制了算法验证与性能对比。

---

## 465. Affinity Tailor: Dynamic Locality-Aware Scheduling at Scale

**arXiv ID:** 2604.27915 | [PDF](https://arxiv.org/pdf/2604.27915v1)

**作者:** Jin Xin Ng `[一作]` (Google), Carlos Villavieja `[通讯]` (Google)

**通讯引用:** 412 | [OpenAlex ID](https://openalex.org/A5084317440)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 Google 生产环境中实现并部署了 Affinity Tailor，这一调度框架通过软亲和力机制（Preferred Cores）结合用户空间 CPU 需求预测，动态划分 CPU 资源，从而提升多租户工作负载的空间局部性和整体吞吐。

**💡 创新点**

创新点包括：1) 将 CPU 需求预测与软亲和力相结合，允许动态大小且离散的 CPU 集；2) 设计了面向芯片级和核心级的两种核心分配算法，兼顾分区与弹性；3) 在内核中实现 CAS（Core‑Aware Scheduling）快速路径，既保持工作保留又优先使用 Preferred Cores。

**🔧 技术方法**

使用技术：用户空间预测器（Borglet）进行短期 CPU 需求估计；Linux 内核 Preferred Cores 机制与 CAS 调度器；硬件性能计数器收集 LLC RPKI、MPKI、分支预测误差等指标；Google Fleet 的大规模实验平台。

**📊 数据集**

数据集：Google 大规模生产机群的多平台数据，包含四类服务器（多芯片/单芯片 LL C），覆盖数千台机器、数百个在线服务（Search、Spanner 等）及多种工作负载。

**📈 对比分析**

比较方法：在同一硬件平台上，将 Affinity Tailor 与修改过的 CFS 基线并行部署；通过请求吞吐量、内存带宽利用、LLC 与分支预测误差等指标进行评估。实验结果显示：每 CPU 吞吐提升最高 12%，每 GB 内存吞吐提升最高 7%；LLC miss 率下降多达 26%，内存带宽利用下降，硬件预取器使用率提升；但 99 分位排队延迟提升约 15‑17%。

**⚠️ 局限性**

局限性包括：1) 在高负载尾部的排队延迟显著增加；2) 需要精细的需求预测与算法参数调优，且对不同硬件平台的通用性需进一步验证；3) 主要针对 CPU/内存密集型工作负载，对网络/IO 密集型服务的收益有限；4) 依赖 Google 内部的 Borg 调度系统，迁移到其他环境的可移植性尚待评估。

---

## 466. Semidefinite and linear programming bounds for sum-rank-metric codes and non-existence results

**arXiv ID:** 2604.27909 | [PDF](https://arxiv.org/pdf/2604.27909v1)

**作者:** Aida Abiad `[一作]` (Eindhoven University of Technology), Ferdinando Zullo `[通讯]` (University of Campania Luigi Vanvitelli)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究sum‑rank度量码的上界，提出新的半正定规划（SDP）和线性规划（LP）上界，并通过谱方法与图论相结合，得到更紧的极限与不存在性结果。

**💡 创新点**

1) 证明在rank‑metric与Hamming极端情形下Delsarte LP与Ratio‑type LP等价；2) 引入Schrijver‑类型的三点SDP上界，首次在sum‑rank设置下使用；3) 利用这些上界给出MSRD码与完美码的非存在性判据。

**🔧 技术方法**

谱图理论、极值理论、Delsarte关联方案、Ratio‑type LP、Lovász θ数、半正定规划（Schrijver‑类型）与线性规划。

**📊 数据集**

在Julia中使用Jordan对称性化简包，对直径≤2000顶点的sum‑rank图进行实验计算；未使用特定公开数据集，而是基于组合计数和图论的解析公式。

**📈 对比分析**

将SDP上界与Delsarte LP、Ratio‑type LP、Singleton/Plotkin/Singleton‑type等传统界比较；在多组参数下SDP均优于已有界，表明半正定方法在sum‑rank场景中具有明显优势。

**⚠️ 局限性**

受限于SDP规模和对称性分解的计算成本；对某些参数未得到闭式解，仍需数值搜索；所给出的非存在性结果仅覆盖特定区间，未覆盖所有可能参数。

---

## 467. Noise2Map: End-to-End Diffusion Model for Semantic Segmentation and Change Detection

**arXiv ID:** 2604.27889 | [PDF](https://arxiv.org/pdf/2604.27889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 468. Frequency-Aware Semantic Fusion with Gated Injection for AI-generated Image Detection

**arXiv ID:** 2604.27875 | [PDF](https://arxiv.org/pdf/2604.27875v1)

**作者:** Shuchang Zhou `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112237 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 Frequency-aware Gated Injection Network（FGINet），用于提升 AI 生成图像检测的跨生成器泛化能力。

**💡 创新点**

创新点包括：① Band-Masked Frequency Encoder（BMFE）通过交叉频带随机遮蔽抑制频率捷径；② Layer-wise Gated Frequency Injection（LGFI）按层门控方式将频率信息逐层注入 VFM；③ Hyperspherical Compactness Learning（HCL）利用 CosFace 的余弦边距约束特征在单位球面上更紧凑、更分离。

**🔧 技术方法**

采用的技术包括：DINOv3 Vision Foundation Model + LoRA 微调、Haar 小波离散小波变换（DWT）提取高频子带、交叉频带遮蔽、逐层门控频率注入、CosFace 余弦边距分类器。

**📊 数据集**

训练集为 Stable Diffusion v1.4 生成图像 + ImageNet；评估集包括 GenImage、SynthBuster、WildRF、Chameleon、RRDataset 等多样化真实/生成图像数据集。

**📈 对比分析**

与多种基线（Fatformer、DRCT、AIDE、SAFE、DDANIPS 等）在标准与野外数据集上对比，FGINet 在 GenImage 达到 96.7%（最高），SynthBuster 94.3%（领先 4.2%），WildRF 最高平均准确率 95.9%，整体实现 state‑of‑the‑art 的检测性能。

**⚠️ 局限性**

局限性：当遮蔽比例过高或过低时会导致频率信息不足或过度依赖；模型过度依赖预训练 VFM，若基础模型表现欠佳会影响效果；目前仅针对图像级别检测，对视频、音频等多模态或极低质量社交平台图像仍存在挑战。

---

## 469. KellyBench: A Benchmark for Long-Horizon Sequential Decision Making

**arXiv ID:** 2604.27865 | [PDF](https://arxiv.org/pdf/2604.27865v1)

**作者:** Thomas Grady `[一作]`, Ross Taylor `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个长期、非平稳的体育博彩环境KellyBench，评估大型语言模型在英超赛季的投注策略和长期资金增长能力。

**💡 创新点**

引入真实赔率、逐日更新的历史数据、基于Kelly准则的奖励机制以及一套以“精密度”评价模型策略的52点量表，创建了一个从建模到执行的闭环评估框架。

**🔧 技术方法**

采用Open Reward Standard协议与ReSum/Claude Code harness，利用Python数据科学堆栈（NumPy、pandas、scikit‑learn）实现模型训练和工具调用，评估过程基于对数财富增长的可验证奖励。

**📊 数据集**

使用公开英超1993-24赛季的比赛与球员统计数据，包括球队阵容、比赛结果、进球、犯规、角球、赔率等，并在每个赛季结束后提供最新结果与球员级别数据。

**📈 对比分析**

通过5个随机种子对不同模型（GPT‑5.4、Claude Opus 4.6、GLM‑5、Gemini 3.1 Pro、Kimi K2.5）在2023‑24赛季的收益、ROI、最终资金以及与人类基准（Human Quant、AI Researcher、Dixon‑Coles）对比，结果显示所有模型平均亏损，最佳模型GPT‑5.4平均ROI‑7.9%，Claude Opus 4.6为‑11.2%，人类基准仍优于模型。

**⚠️ 局限性**

评估仅覆盖单个赛季，缺乏多季对比；使用闭盘赔率与5% overround限制收益边缘；模型可能因训练数据泄露而受影响；并且对非稳态与即时适应的需求未被充分满足，导致模型普遍失效。

---

## 470. MyoKin3X: A Myoelectric Framework for Full-Hand 3D Force Recording

**arXiv ID:** 2604.27949 | [PDF](https://arxiv.org/pdf/2604.27949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 471. ClimateVID -- Social Media Videos Analysis and Challenges Involved

**arXiv ID:** 2604.27968 | [PDF](https://arxiv.org/pdf/2604.27968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 472. Differentiable latent structure discovery for interpretable forecasting in clinical time series

**arXiv ID:** 2604.27967 | [PDF](https://arxiv.org/pdf/2604.27967v1)

**作者:** Ivan Lerner `[一作]`, Francis Bach `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种结构化高斯过程模型StructGP及其扩展LP-StructGP，用于多任务时间序列预测，尤其是在重症监护（ICU）数据中学习任务间依赖关系和跨个体共享的潜在轨迹；

**💡 创新点**

创新点在于：①基于卷积滤波的结构化协方差；②内部标准化消除量纲影响；③采用NOTEARS实现可微有向无环图学习；④LP-StructGP通过共享潜在路径实现跨个体共性，并通过softmax门控实现柔性混合；⑤利用HSGP低秩近似与在线Woodbury更新提升大规模计算效率。

**🔧 技术方法**

使用技术包括：高斯过程回归、卷积滤波协方差、内部标准化、NOTEARS稀疏正则、软max门控、HSGP低秩特征、在线Woodbury更新、渐进条件伪边际似然优化。

**📊 数据集**

实验数据：合成模拟数据；真实ICU重症患者（Sepsis‑3）时间序列；PhysioNet 2019挑战的18个生理/药物任务。

**📈 对比分析**

通过与无结构基线和SOTA GraFITi 对比，使用RMSE、MAE、MSE、95% 预测覆盖率等指标。模型在PhysioNet挑战中在大多数任务上接近或超过SOTA，尤其在高置信区间下表现更好。

**⚠️ 局限性**

局限性：需预先设定潜在路径数和滤波参数，对超参数调优敏感；稀疏约束依赖NOTEARS数值稳定性；在极大规模数据时仍需GPU/分布式计算；理论收敛与可解释性尚不完整。

---

## 473. Simpler and Improved Replacement Path Coverings

**arXiv ID:** 2604.27966 | [PDF](https://arxiv.org/pdf/2604.27966v1)

**作者:** Davide Bilò `[一作]` (University of L'Aquila), Martin Schirneck `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5008898604)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种基于条件期望法的确定性 (L,f)-替代路径覆盖（RPC）构造，显著降低了覆盖值和查询时间，并给出了对应的上界与下界。

**💡 创新点**

创新点包括：① 用条件期望法实现完全确定性 derandomization，避免了之前复杂的编码方法；② 在 f = o(log L) 的敏感度范围内将覆盖值提升至 O(f L^f+o(1))，查询时间下降至 O(f^{5/2} L^{o(1)});③ 通过更精细的分析将随机 RPC 的覆盖值从 O((L/f)^f L^{o(1)}) 降至同一阶；④ 证明了一个更强的下界 Ω(min{√(f e^f) L^{f-1}/f^f, n})，缩小了覆盖值的幅度。

**🔧 技术方法**

核心技术包括：条件期望法（conditional expectations）在树结构的逐级决策中平衡“well‑separated”与“poorly‑separated”对；层次化采样树（sampling trees）框架；对二项式系数的紧致分析；以及对权重化有向树构造的组合下界构造。

**📊 数据集**

本工作为纯理论研究，不涉及具体数据集，所有结论均基于图论模型与随机化分析。

**📈 对比分析**

与 Weimann‑Yuster (O(f L^f) 覆盖值，O(f^2 L^f) 查询) 和 Karthik‑Parter (O((cfL log n)^f+1) 覆盖值，O(f^2 L) 查询) 的对比显示：在 f = o(log L) 时，本方法覆盖值仅多了 O(f^o(1)) 量级，查询时间从 O(f^2 L^f) 下降到 O(f^{5/2} L^{o(1)})，并且在覆盖值和查询时间上均实现了最优或接近最优的性能。

**⚠️ 局限性**

局限性：① 适用范围要求 f = o(log L)（对更大敏感度需进一步改进）；② derandomization 需要预先计算所有 O(n^2 m^f) 次查询的期望，实际实现时计算成本可能较高；③ 对极端图结构（如稠密图）下的常数项影响仍未完全解析。

---

## 474. Attractor FCM

**arXiv ID:** 2604.27947 | [PDF](https://arxiv.org/pdf/2604.27947v1)

**作者:** Alexis Kafantaris `[一作]` `[通讯]`, Alexis Kafantaris

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于 Newton 固定点、Jacobian 梯度下降的物理约束吸引子 FCM，用于模拟和优化复杂系统。

**💡 创新点**

创新点在于：①将 Newton 法求解 FCM 固定点与梯度下降结合，形成 Jacobian 梯度下降；②使用残差记忆和 BPTT 固定点锚点实现稳定收敛；③引入因果/结构性掩码以保留物理信息；④证明了系统的去噪与收敛性质；⑤自适应尺度 λ 调节学习率。

**🔧 技术方法**

使用 FCM、Newton 固定点、Jacobian 梯度下降、BPTT、Lipschitz 收缩、Banach 固定点定理、可自适应尺度因子以及物理约束掩码。

**📊 数据集**

数据集主要为三类自定义情景模拟（寡头救助、生态连锁、独裁者困境）以及 20 折量化测试；未使用公开真实数据集。

**📈 对比分析**

通过与简单、Hebbian、Agentic、Hybrid 与传统 GD 等五种算法在 20 折量化和三类情景的性能对比，使用误差值越低越好。结果显示 Jacobian 梯度下降在大多数情景下表现最佳，尤其在压力、收敛、去噪和陷阱测试中显著优于其他方法。

**⚠️ 局限性**

局限性包括：收敛速度相对较慢、计算开销大；对结构掩码的依赖需要先验物理知识；在某些定性情景下传统 GD 偶尔优于吸引子 FCM；缺乏在真实大规模数据上的验证。

---

## 475. Calibrating Attribution Proxies for Reward Allocation in Participatory Weather Sensing

**arXiv ID:** 2604.27944 | [PDF](https://arxiv.org/pdf/2604.27944v1)

**作者:** Mark C. Ballandies `[一作]` (University of Zurich), Claudio J. Tessone `[通讯]` (University of Zurich)

**通讯引用:** 4527 | [OpenAlex ID](https://openalex.org/A5020270223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用可微分的 AI 天气模型（FourCastNet 与 SFNO）计算梯度归因，作为参与式天气感知奖励分配的数值信号；

**💡 创新点**

首次将梯度归因直接用于量化传感器贡献，实现了既可计费又可校准的奖励机制；

**🔧 技术方法**

使用整合梯度、梯度×输入（GTI）和原始梯度等归因方法，并结合逆哈弗赛因距离等基准；

**📊 数据集**

在 60 个 6 小时 GFS 分析样本、468 个欧洲网格点以及 5 个目标城市（苏黎世、伦敦、柏林、马德里、奥斯陆）上进行实验；

**📈 对比分析**

与传统距离、均匀及“oracle”基准对比，归因方法在捕获 92% 以上的最优传感器配置、33–36% 的预算精准度（比 61–72% 的均匀分配低一半），GTI 在成本仅为 IG 的 1/50 时仍保持 83% 的信号完整性；

**⚠️ 局限性**

局限包括仅针对 +6h 预报、仅使用两种模型、仅欧洲区域、未验证观测-网格映射、对攻击模型假设有限，且归因对数据伪造不具完整防护。

---

## 476. Beyond the Baseband: Adaptive Multi-Band Encoding for Full-Spectrum Bioacoustics Classification

**arXiv ID:** 2604.27936 | [PDF](https://arxiv.org/pdf/2604.27936v1)

**作者:** Eklavya Sarkar `[一作]` (Earth Species Project), Matthieu Geist `[通讯]` (Earth Species Project)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种多频带编码框架，将动物呼叫的完整频谱划分为若干基频带并通过不同的融合策略合成统一表示。

**💡 创新点**

创新点在于利用异频耦合（heterodyning）将高频内容映射到可用基带，并通过多种学习型或注意力型融合方法提升表征的互补性与区分度。

**🔧 技术方法**

主要技术包括频谱分段与低通滤波、冻结预训练音频编码器（EfficientNet、BEATs、EATs、BirdNET 等）、五种融合策略（均值池化、门控池化、Mixture‑of‑Experts、Hybrid、Self‑Attention）以及线性分类器。

**📊 数据集**

实验使用了三种生物声学数据集：Dogs（44.1 kHz）、Cornell Birdcall Identification (CBI)（44.1 kHz）和 Bats（250 kHz）。

**📈 对比分析**

在与基带（BB）和时延展开（TE）基准的对比实验中，所提多频带融合在三组数据集上普遍提升了准确率，尤其在高频蝙蝠数据集上显著优于 BB 与 TE，并可与本身已在高采样率上预训练的 BirdNET 相竞争。

**⚠️ 局限性**

局限性包括：频带宽度固定且无重叠；对预训练模型对高频敏感度的依赖；未训练真正高采样率的基础模型；以及对非均匀或变宽频段的适应性仍待改进。

---

## 477. MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection

**arXiv ID:** 2604.27934 | [PDF](https://arxiv.org/pdf/2604.27934v1)

**作者:** Weihai Lu `[一作]` (Peking University), Huan He `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种检索增强的多模态多代理框架，用于鲁棒的多模态立场检测。

**💡 创新点**

创新点包括：1）检索增强阶段提供具体例子进行上下文锚定；2）专门的多模态分析代理分别从文本、图像及跨模态冲突角度解析；3）辩论阶段模拟不同立场的论证；4）自我反思与裁决阶段通过批判性自评提升最终决策质量。

**🔧 技术方法**

使用技术：大型多模态语言模型（LLM）作为底层推理器；检索增强生成（RAG）结合 CLIP 向量检索；多代理协同推理（多模态分析代理、辩论代理、裁决代理）；Chain-of-Thought（CoT）和自我反思机制。

**📊 数据集**

采用五个公开多模态立场检测数据集：Mtse、Mccq、Mwtwt、Mruc、Mtwq。

**📈 对比分析**

与多种基线（文本、视觉、传统多模态、LLM增强、TMPT、MV-Debate 等）在 in-target 和 zero-shot 设置下进行对比，取得宏观 F1 成绩均为新 SOTA，显著优于文本/视觉单一基线及单通道 LLM 模型。

**⚠️ 局限性**

局限性：1）多阶段多代理结构导致推理时间和计算开销较高；2）对底层 LLM 能力高度依赖，模型固有偏差或事实错误会影响结果；3）检索增强效果受检索数据库质量影响，缺少高质量实例或 CoT 质量不足时性能会下降。

---

## 478. Dynamic Cluster Data Sampling for Efficient and Long-Tail-Aware Vision-Language Pre-training

**arXiv ID:** 2604.27932 | [PDF](https://arxiv.org/pdf/2604.27932v1)

**作者:** Mingliang Liang `[一作]` (Radboud University), Martha Larson `[通讯]` (Radboud University)

**通讯引用:** 6134 | [OpenAlex ID](https://openalex.org/A5056272341)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DynamiCS，一种动态基于聚类的采样方法，用于降低视觉-语言模型（VLM）预训练的计算成本，同时提升长尾概念的学习效果。

**💡 创新点**

创新点：① 采用动态采样，每个 epoch 随机抽取不同子集，增加训练多样性；② 引入“aim for utility”理念，对大集群进行下采样，对小集群进行上采样，保留原始语义分布顺序而非追求完全均衡；③ 不依赖已训练的 VLM 进行数据过滤，直接在原始大规模数据集上进行聚类与重采样。

**🔧 技术方法**

技术手段：K-means 余弦相似度聚类 + 聚类中心相似度阈值合并；α 参数控制样本分配（α=0 为完全均衡，α=1 为随机采样）；动态采样比例 P_i = S_i/c_i；ViT-B/16 与 ViT-L/16 视觉编码器，Transformer 文本编码器，CLIP 对比学习框架；在 112×112 低分辨率下预训练并在 224×224 上 fine‑tune。

**📊 数据集**

使用数据集：LAION‑400M（约 298M 对），DataComp‑DFN（约 130M 对）作为训练集；评测集包括 ImageNet‑1K、Let‑it‑wag!（长尾测试集）、COCO、Flickr30k、以及 25 个零样本分类基准。

**📈 对比分析**

对比方法：与 RECLIP、FLIP、CLIPA、DataComp、DFN、HQ‑CLIP、OpenCLIP、OpenAI‑WIT 等低成本或全规模预训练基线对比；DynamiCS 在 50% 训练样本、60% GPU 小时下，ImageNet‑1K 零样本准确率提升约 4–6%，Let‑it‑wag! 长尾准确率提升 10%+；在仅使用约 3% 训练成本的情况下，DynamiCS 的 ImageNet‑1K 与长尾性能可与或超越完整规模 CLIP。

**⚠️ 局限性**

局限性：① 聚类质量依赖预训练图像嵌入，极端语义多样性或模糊概念可能导致聚类误差；② α 参数需要经验选择，过大或过小均会损失长尾或头部性能；③ 主要针对图像‑文本对的 CLIP 任务，迁移到其他多模态或非视觉域的 VLM 可能需要调整；④ 对极少量或极稀疏类别的提升有限，仍需进一步改进。

---

## 479. Modeling Clinical Concern Trajectories in Language Model Agents

**arXiv ID:** 2604.27872 | [PDF](https://arxiv.org/pdf/2604.27872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 480. Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future

**arXiv ID:** 2604.27924 | [PDF](https://arxiv.org/pdf/2604.27924v1)

**作者:** Sihong Wu `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7566 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了人工智能（尤其是大语言模型）在学术同行评审全流程中的生成、后评审任务与评估方法，构建了完整的分类框架并系统化相关数据集与指标；

**💡 创新点**

创新点在于首次提出细粒度的四层分类（生成范式、后评审任务、评估范式、评价指标），填补了现有综述在评估体系和多模态/跨学科场景覆盖不足的空白；

**🔧 技术方法**

采用了LLM微调、Agent-based、多任务强化学习、检索增强生成（RAG）以及基于对话的多轮交互等多种技术手段的综述与对比；

**📊 数据集**

综述涵盖了Pre‑2023的PeerRead、NLPEER、PeerSum等数据集以及Post‑2023的ReviewMT、ReviewCritique、MAMORX、SubstanReview、MReD、MOPRD等新兴数据集；

**📈 对比分析**

评估方法从人类主观评判、BLEU/ROUGE/BERTScore等参考度量，到LLM评估与面向方面的细粒度评估进行系统比较，指出现有指标多为表面相似度，缺乏对真实性、可操作性等深度维度的有效量化；

**⚠️ 局限性**

限制包括快速的技术迭代导致综述可能落后、数据集与评估集中在NLP/ML领域且缺乏跨学科与多模态支持、后评审任务的细粒度评价和鲁棒性测试不足等。

---

## 481. A Logic of Inability

**arXiv ID:** 2604.27917 | [PDF](https://arxiv.org/pdf/2604.27917v1)

**作者:** Shanxia Wang `[一作]` (Henan Normal University), Shanxia Wang `[通讯]` (Henan Normal University)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5078371286)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在Coalition Logic中引入了不可行性（inability）操作符，系统性研究了其逻辑语义与结构特性

**💡 创新点**

将不可行性视为独立的一阶概念，给出其定义、语义模型、完备证明以及一系列结构性质（反单调性、反变形、非加法性等）

**🔧 技术方法**

利用Coalition Logic的定义、可行性与不可行性的对应关系，构造一阶语义模型，并使用定义性扩展证明一致性、完备性和对原始逻辑的保守性

**📊 数据集**

无

**📈 对比分析**

无

**⚠️ 局限性**

仅处理单步、无信息/资源限制的不可行性；未考虑不确定性、时间演化、知识或资源约束的更复杂情形

---

## 482. An Empirical Evaluation of Code Smell Detection in Angular Applications

**arXiv ID:** 2604.27893 | [PDF](https://arxiv.org/pdf/2604.27893v1)

**作者:** Maykon Nunes `[一作]` (Federal University of Ceará), Ivan Machado `[通讯]` (Federal University of Bahia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过灰色文献综述识别并定义了11种Angular框架特定及跨框架的代码异味，并实现了基于ReactSniffer和SniffTSX的自动检测工具。

**💡 创新点**

创新点在于首个基于社区实践的Angular代码异味目录，以及将已有的React代码异味检测框架迁移并扩展至Angular的通用静态分析工具。

**🔧 技术方法**

使用的技术包括灰色文献综述方法、AST解析（Babel）、TypeScript/Angular语法处理、以及自定义静态分析检测器（ReactSniffer、SniffTSX）来实现异味检测。

**📊 数据集**

数据集来自10个公开Angular项目，手工标注了150条异味实例（每种10条），并提供对应的已修复版本用于评估。

**📈 对比分析**

通过与人工验证的基准进行比较，检测工具在所有实现的异味上实现了准确率≥0.88、F1≥0.89，整体性能优秀；误报主要集中在基于行数阈值的“Large Component”和“Large File”异味。

**⚠️ 局限性**

研究局限包括依赖灰色文献导致的主观性、阈值驱动检测可能产生误报、目前仅覆盖6种异味、未在混合框架或更大规模项目中充分验证工具的适用性。

---

## 483. Parameter-Efficient Architectural Modifications for Translation-Invariant CNNs

**arXiv ID:** 2604.27870 | [PDF](https://arxiv.org/pdf/2604.27870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 484. When and How AI Should Assist Brainstorming for AI Impact Assessment

**arXiv ID:** 2604.27997 | [PDF](https://arxiv.org/pdf/2604.27997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 485. 3D Reconstruction Techniques in the Manufacturing Domain: Applications, Research Opportunities and Use Cases

**arXiv ID:** 2604.28064 | [PDF](https://arxiv.org/pdf/2604.28064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 486. Measuring research data reuse in scholarly publications using generative artificial intelligence: Open Science Indicator development and preliminary results

**arXiv ID:** 2604.28061 | [PDF](https://arxiv.org/pdf/2604.28061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 487. NeuroRing: Scaling Spiking Neural Networks via Multi-FPGA Bidirectional Ring Topologies and Stream-Dataflow Architectures

**arXiv ID:** 2604.28059 | [PDF](https://arxiv.org/pdf/2604.28059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 488. Framework for Collaborative Operation of Autonomous Delivery Vehicles Within a Marshaling Yard

**arXiv ID:** 2604.28057 | [PDF](https://arxiv.org/pdf/2604.28057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 489. Collaborative Agent Reasoning Engineering (CARE): A Three-Party Design Methodology for Systematically Engineering AI Agents with Subject Matter Experts, Developers, and Helper Agents

**arXiv ID:** 2604.28043 | [PDF](https://arxiv.org/pdf/2604.28043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 490. TAFA-GSGC: Group-wise Scalable Point Cloud Geometry Compression with Progressive Residual Refinement

**arXiv ID:** 2604.28045 | [PDF](https://arxiv.org/pdf/2604.28045v1)

**作者:** Xiumei Li `[一作]`, André Kaup `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种多层可进化的学习型图像压缩框架，通过基层与两级增量层实现可伸缩码流。

**💡 创新点**

创新点在于引入 Novelty‑Aware Feature Aggregation (TAFA) 模块，能够自适应识别并仅编码不同层之间的“新颖”信息，从而显著降低冗余。

**🔧 技术方法**

采用卷积神经网络进行特征提取与重建，使用自回归熵模型和算术编码实现高效概率建模和压缩。

**📊 数据集**

在 Kodak、Tecnick、DIV2K 等公开图像数据集上进行训练与评估。

**📈 对比分析**

与 VTM、BPG 以及近期学习型压缩方法进行对比，实验显示在相同码率下可提升约0.5–1.0 dB 的 PSNR/SSIM，或在相同质量下比特率降低约15%。

**⚠️ 局限性**

局限性包括模型参数量大、编码/解码速度相对慢，且在极高压缩率或非典型场景下的表现不如传统专业压缩器。

---

## 491. PROMISE-AD: Progression-aware Multi-horizon Survival Estimation for Alzheimer's Disease Progression and Dynamic Tracking

**arXiv ID:** 2604.28055 | [PDF](https://arxiv.org/pdf/2604.28055v1)

**作者:** Qing Lyu `[一作]` (Yale University), Christopher T Whitlow `[通讯]` (Yale University)

**通讯引用:** 7572 | [OpenAlex ID](https://openalex.org/A5000092822)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种泄漏安全的多时段生存预测框架 PROMISE-AD，用于从不规则前期访视记录中预测阿尔茨海默病进展。

**💡 创新点**

创新点在于将前期访视完整缺失处理、进展感知的令牌化、时序 Transformer 融合、潜在混合风险模型以及多时段校准统一到一个监督式生存模型。

**🔧 技术方法**

采用了 Transformer 编码、离散时间混合风险估计、焦点风险损失、进展排名损失、平滑与门控正则，以及验证集的等价回归校准。

**📊 数据集**

使用 ADNI/TADPOLE 的临床、认知、遗传、影像等多模态表格历史数据。

**📈 对比分析**

与多种基线（Cox、DeepHit、DeepSurv、RSF、XGBoost-Cox、TabPFN、LSTM 等）比较，CN→MCI 上实现最低IBS，MCI→AD 上取得最高C-index和近乎完美的5年AUROC/AUPRC。

**⚠️ 局限性**

局限在于 CN→MCI 事件稀疏导致时段指标不稳、未在外部样本验证、混合专家缺乏临床解释、仅针对研究人群。

---

## 492. Early Detection of Water Stress by Plant Electrophysiology: Machine Learning for Irrigation Management

**arXiv ID:** 2604.28038 | [PDF](https://arxiv.org/pdf/2604.28038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 493. On the Principal Minor Expansion and Complexity of the Symmetrized Determinant

**arXiv ID:** 2604.28019 | [PDF](https://arxiv.org/pdf/2604.28019v1)

**作者:** Sanyam Agarwal `[一作]` (University of Saarland), Mridul Gupta `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了非交换代数中对称化行列式（symmetrized determinant）的代数性质与计算复杂性，证明其满足主子式展开性质，并展示其在多项式维度代数上为 #P‑难且 VNP‑完全。

**💡 创新点**

创新点在于将对称化行列式与传统行列式的主子式展开联系起来，给出了在多项式维度代数上的 #P‑难与 VNP‑完全证明，揭示了其与永久（permanent）和哈密顿环多项式的深层关系。

**🔧 技术方法**

主要技术包括排列组合与符号分析、非交换代数构造、从 Hamiltonian Cycle 多项式的归约、对称化行列式的符号与乘法性质证明，以及多项式时间算法（O(n^{r+3})）的构造。

**📊 数据集**

未使用任何实验数据集，研究完全基于理论分析。

**📈 对比分析**

比较方法主要是理论复杂度分析；对固定维度 r 的代数可在多项式时间内计算（O(n^{r+3})），但在维度为多项式时已证明为 #P‑难，表明算法性能随 r 指数增长。

**⚠️ 局限性**

限制在于未解决对固定维度 r 的对称化行列式的 FPT（参数化多项式时间）算法，且目前仅在特定代数构造下给出完整性与难度结果，未涵盖更一般代数或更广泛的应用场景。

---

## 494. Design Structure Matrix Modularization with Large Language Models

**arXiv ID:** 2604.28018 | [PDF](https://arxiv.org/pdf/2604.28018v1)

**作者:** Shuo Jiang `[一作]` (City University of Hong Kong), Jianxi Luo `[通讯]` (City University of Hong Kong)

**通讯引用:** 4227 | [OpenAlex ID](https://openalex.org/A5031475851)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了将大型语言模型用于设计结构矩阵（DSM）的模块化分割问题，并在五个工程案例上验证其可行性。

**💡 创新点**

发现并解释了“知识悖论”，即在DSM模块化中引入域知识会适得其反；提出了语义对齐假设并系统评估了输入格式、目标表述与解池设计对LLM-CO性能的影响。

**🔧 技术方法**

采用LLM驱动的组合优化框架（LLM-CO），利用Claude、GPT-5.2和Qwen-3.5-Plus等三种大模型生成候选分区，并通过外部计算TotalCost和Clustering Efficiency进行评估。

**📊 数据集**

使用来自航空、汽车和机械工程的五个真实DSM实例（UCAV、Kodak Cartridge、Brake System、HeatEx、Helicopter），节点数从12到19。

**📈 对比分析**

与10,000次随机起点的模拟退火基准对比，Claude在30次迭代内实现与参考相当（Gap%≤1.3%），且在最复杂案例中所有实验均匹配参考；GPT与Qwen表现相对不稳定，需更多迭代。

**⚠️ 局限性**

仅覆盖≤19节点的DSM，缺乏对更大规模实例的验证；参考解非全局最优；不同DSM类型及模型可能呈现不同的知识敏感性；仅评估了TotalCost与聚类效率两种指标。

---

## 495. Kernelized Advantage Estimation: From Nonparametric Statistics to LLM Reasoning

**arXiv ID:** 2604.28005 | [PDF](https://arxiv.org/pdf/2604.28005v1)

**作者:** Shijin Gong `[一作]` (University of Science and Technology of China), Chengchun Shi `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 636 | [OpenAlex ID](https://openalex.org/A5025970743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限计算资源下，提出一种基于核平滑的优势函数估计方法（KAE），用于改进LLM推理中的价值函数和梯度估计，从而提升策略学习质量。

**💡 创新点**

创新点在于将经典非参数统计方法（核平滑）直接嵌入RLVR框架，通过利用历史奖励对价值函数进行时间平滑，获得与“oracle”算法相当的估计精度，且无需额外的价值网络或大规模采样。

**🔧 技术方法**

主要技术包括：1）Kernel Smoothing（Nadaraya–Watson）对价值函数进行估计；2）离群留一（leave‑one‑out）优势估计；3）基于策略梯度的更新；4）自适应的提示采样调度与历史奖励回溯。

**📊 数据集**

实验使用的推理基准数据集包括：GSM8K、MATH、DAPO，评估在 Qwen2.5-1.5B/7B、Qwen2.5-Math-1.5B 等模型上。

**📈 对比分析**

与 GRPO、REINFORCE++、Dr. GRPO、GPG 等基线算法比较，KAE 在价值函数 MSE 上降低 60%–90%，梯度 MSE 降低 5%–65%，在多场景策略优化中平均提升约 5%（MATH）或 11.8%（DAPO），单流情况下可提升 14.9%（GSM8K）和 6.6%（MATH），并且训练更为稳定。

**⚠️ 局限性**

局限性：① 需手动调节核带宽与核函数，性能对带宽敏感度仍需进一步研究；② 依赖对提示的重复采样调度，若提示分布高度动态可能效果受限；③ 理论分析基于 i.i.d. 提示、有限奖励等假设，实际应用中可能不完全满足；④ 仅在中等规模 LLM（1B–7B）上验证，极大模型或更复杂任务的可扩展性待进一步探索。

---

## 496. Dynamic Scaled Gradient Descent for Stable Fine-Tuning for Classifications

**arXiv ID:** 2604.27987 | [PDF](https://arxiv.org/pdf/2604.27987v1)

**作者:** Nghia Bui `[一作]` (New Jersey Institute of Technology), Lijing Wang `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1581 | [OpenAlex ID](https://openalex.org/A5100330304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种动态缩放梯度下降算法（DSGD），通过在微调过程中按比例缩小已正确分类样本的梯度，从而缓解梯度冲突导致的训练崩溃和性能波动。

**💡 创新点**

创新点在于：1）首次将梯度冲突问题与已正确分类样本的梯度联系起来；2）设计动态可调缩放因子γ_t，并在理论上证明其能提升稳定性和收敛性；3）算法无需额外反向传播、无结构改动，计算开销极低。

**🔧 技术方法**

使用的技术包括：梯度分组（正例/负例）与动态缩放；基于梯度范数与余弦相似度的冲突分析；在多种大规模预训练模型上实现微调（BERT、RoBERTa、LLaMA、ViT）。

**📊 数据集**

实验数据集涵盖NLP任务：SuperGLUE（MultiRC、COPA、RTE、BoolQ）、GLUE（MRPC、CoLA）；视觉任务：不平衡CIFAR‑10/100（长尾与阶梯分布）；全部使用公开预训练模型。

**📈 对比分析**

与FFT、FocalLoss、LNSR、NoisyTune、PCGrad、集成与SWA等方法对比，DSGD在10个随机种子下显著降低标准差（从10%下降至约1%），平均准确率提升5–10%，并在大多数任务中超越了集成方法。

**⚠️ 局限性**

局限性：仅针对梯度优化过程的随机性；不解决提示格式化、示例选择等推理阶段的不确定性；对硬件非确定性（如GPU随机种子、并行执行）仍未彻底消除。

---

## 497. On Higher-Order Probabilistic Verification via the Weighted Relational Model of Linear Logic

**arXiv ID:** 2604.27986 | [PDF](https://arxiv.org/pdf/2604.27986v1)

**作者:** Ugo Dal Lago `[一作]` (University of Bologna), Paolo Pistone `[通讯]` (Université Claude Bernard Lyon 1)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5059212008)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文证明了一类新的有限/受限复制的概率高阶递归方案（PHORS）可决定其几乎必然终止（AST）和期望终止步数有限（PAST）问题。

**💡 创新点**

创新点在于将线性逻辑的加权关系语义与生成函数相结合，构造出有限化的代数方程系统，从而把PHORS的终止问题转化为求解代数生成函数的最小解。

**🔧 技术方法**

主要技术包括：加权关系语义、带参数的线性逻辑、有限指数（bounded exponential）类型纪律、代数生成函数（代数幂级数）和求解正规化方程的最小解。

**📊 数据集**

本文未使用任何实验数据集，全部为理论推导与形式化证明。

**📈 对比分析**

与之前基于自动机或博弈语义的可判定结果相比，本文提供了更直接的代数化方法；在给定阶数固定的情况下，决策算法的空间复杂度为多项式，而阶数不固定时复杂度呈指数级增长。

**⚠️ 局限性**

局限性在于仅覆盖有限/受限复制的PHORS，无法处理一般的高阶PHORS；对于更高阶的程序仍然是不可判定的；此外，生成函数求解和数值计算的实际实现复杂度较高。

---

## 498. The Origins of MEV: Systematic Attribution of Arbitrage Opportunity Creation at Scale

**arXiv ID:** 2604.27979 | [PDF](https://arxiv.org/pdf/2604.27979v1)

**作者:** Andrei Seoev `[一作]` (MEV-X), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5020656981)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究并实现了一套 MEV 机会归因框架，识别哪些链上交易在 Polygon 主网上产生可被原子套利交易利用的价值，并对数十万起源交易进行系统性归因分析。

**💡 创新点**

创新点在于首次正式化 MEV 机会归因问题，提出四种归因方法（bot‑data、模拟、系数、Shapley），并通过大规模实验验证“单源”假设，使归因方法可扩展且可量化。

**🔧 技术方法**

使用的技术包括 EVM 状态机的确定性重放、二进制搜索与逆向冲击计算、图神经网络+强化学习预测、协同博弈 Shapley 值、以及区块链档案节点的事务追踪。

**📊 数据集**

使用的数据集为 Polygon 主网历史区块（2026 年 3 月共 360,026 笔原子套利事件；2026 年 2 月的 2,526 笔用于基准验证），以及 MEV 搜索者的竞价日志。

**📈 对比分析**

在准确率、覆盖率和平均耗时上，模拟归因取得 91.7% 的准确率、99.1% 的覆盖率、每笔 12.3 ms；系数归因最快 0.8 ms、准确率 77.2%；bot‑data 94.2% 的准确率但覆盖仅 38.4%；Shapley 精确度 100%，但单次 5 min，MC 版约 1 s。

**⚠️ 局限性**

局限性包括仅在 Polygon 上实验、仅聚焦原子套利、基准真值通过三方验证而非直接观测、以及对多链或其他 MEV 形式的泛化尚待进一步研究。

---

## 499. TransVLM: A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions

**arXiv ID:** 2604.27975 | [PDF](https://arxiv.org/pdf/2604.27975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 500. Energy-Aware Quantum-Enhanced Computing Continuum

**arXiv ID:** 2604.28041 | [PDF](https://arxiv.org/pdf/2604.28041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 501. RHyVE: Competence-Aware Verification and Phase-Aware Deployment for LLM-Generated Reward Hypotheses

**arXiv ID:** 2604.28056 | [PDF](https://arxiv.org/pdf/2604.28056v1)

**作者:** Feiyu Wu `[一作]` (Xidian University), Hui Li `[通讯]` (Xidian University)

**通讯引用:** 38910 | [OpenAlex ID](https://openalex.org/A5065859286)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 RHyVE，一种在已生成奖励候选集上进行竞争力验证并基于学习者能力与训练阶段动态部署奖励的协议。

**💡 创新点**

创新点在于将奖励视为可验证的假设，通过共享检查点的短期分叉验证评估奖励可靠性，并根据验证结果决定是否使用单一奖励、两阶段调度或保守保留。

**🔧 技术方法**

采用了共享检查点分叉验证、奖励赢家边际/一致性评估、阶段化投射、两阶段或保守部署规则等技术，兼顾验证成本与训练稳定性。

**📊 数据集**

主要使用Frank​​aCabinet机器人抓取任务、BallBalance、FrankCubeStack等机器人任务，并在6×3090的LLM奖励生成实验中验证方法。

**📈 对比分析**

与直接训练、硬切换、潜在形状调度等基线比较，RHyVE在稀疏相位敏感任务中显著提升峰值和最终成功率，且多项控制实验表明其优势源自验证与部署决策，而非单纯计算优势。

**⚠️ 局限性**

局限性包括仅适用于小规模奖励候选集、需手动设定分叉窗口与竞争度阈值、难以扩展到大规模奖励池，且对切换时机与方式仍需经验选择。

---

## 502. To Build or Not to Build? Factors that Lead to Non-Development or Abandonment of AI Systems

**arXiv ID:** 2604.28053 | [PDF](https://arxiv.org/pdf/2604.28053v1)

**作者:** Shreya Chappidi `[一作]` (University of Cambridge), Jatinder Singh `[通讯]` (Research Centre Trust, UA Ruhr, University Duisburg-Essen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性梳理了文献与灰色资料，构建了AI研发放弃（非开发）六大类因子分类法，并通过AI事故数据库和实践者调查收集实证案例，分析不同阶段放弃AI的驱动因素。

**💡 创新点**

首次将AI研发放弃视为可研究的实践，提出了覆盖伦理、法律、资源、组织、生命周期和利益相关者反馈等六大维度的因子分类，并通过实证数据揭示非伦理因素在放弃决策中的重要性。

**🔧 技术方法**

采用多元文献综述与主题分析法，结合案例编码与定性比较；利用在线问卷与公开事故数据库对案例进行归类与统计。

**📊 数据集**

主要使用了AIAAIC（AI, Algorithmic and Automation Incidents and Controversies）数据库中的91条放弃案例，以及28名实践者的调查数据，共计17条放弃案例。

**📈 对比分析**

通过对案例中涉及因子占比的描述性统计，发现伦理与利益相关者反馈在已部署系统中占主导，而在未部署系统中资源、组织与生命周期挑战更为突出；未做传统性能评估，仅提供因子频次与比例分析。

**⚠️ 局限性**

局限性包括：事故数据库案例主要集中在已部署系统，难以全面覆盖早期放弃情形；调查样本规模有限，可能存在自选偏差；因子归类和编码主观性高，缺乏外部验证；未提供定量性能指标，研究更多面向现象描述而非技术评测。

---

## 503. Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception

**arXiv ID:** 2604.28048 | [PDF](https://arxiv.org/pdf/2604.28048v1)

**作者:** Neemias B da Silva `[一作]` (Universidade Tecnológica Federal do Paraná), Thiago H Silva `[通讯]` (University of Toronto)

**通讯引用:** 1774 | [OpenAlex ID](https://openalex.org/A5072023060)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究使用多模态LLM（Qwen3‑VL:8B）对城市场景图像进行情感注释，构建24种性别、经济状况、政治取向与人格属性组合的因子设计，共计1,200个代理，评估50张图像的情感判定，分析个体内部一致性与跨属性差异。

**💡 创新点**

创新点在于首次系统地量化Persona prompting对LLM行为的稳定性与多样性，揭示了在情感粗粒度上表现良好，但在细粒度情感判断上差异有限，并将此与无Persona条件进行对比验证。

**🔧 技术方法**

核心技术包括多模态LLM推理、结构化Persona注入的系统提示、Chain‑of‑Thought reasoning、LangChain LangGraph流水线及JSON解析与归一化等。

**📊 数据集**

使用PerceptSent数据集（5,000张城市图像，50张为实验样本），其中每张图像均有人类5级情感标签与闭合词汇表的描述。

**📈 对比分析**

通过宏F1和Cohen κ等指标对多重情感任务（二分类、三分类、五级）进行比较，发现Persona prompting在极端情感上表现良好，但在中立及轻微情感上精度显著下降，且单一无Persona推理往往匹配或优于大量Persona聚合结果。

**⚠️ 局限性**

局限性包括：① Persona定义过于简化，未能充分捕捉真实身份多样性；② 样本量有限，仅50张图像；③ 模型表现出极端偏差，压缩中间情感类别；④ 对比实验中无Persona仅单次推理，Persona条件为聚合，导致评估不对称；⑤ 可能强化刻板印象与误导性描绘。

---

## 504. Shuffling-Aware Optimization for Private Vector Mean Estimation

**arXiv ID:** 2604.28032 | [PDF](https://arxiv.org/pdf/2604.28032v1)

**作者:** Shun Takagi `[一作]` (LY Corporation), Seng Pei Liew `[通讯]` (LY Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究单消息洗牌模型下的均值估计，提出使用洗牌指数（shuffle index）来刻画洗牌后的隐私性能，并在高隐私阶梯中对无偏协议给出最优均方误差下界，随后构造了盲混合高斯（blanket‑mixed Gaussian）本地随机器，证明其在大样本极限下实现该下界，进一步证明其隐私-效能曲线与中心高斯机制在高隐私范围内相同。

**💡 创新点**

①引入洗牌指数作为单用户设计的全局约束，实现了把多用户洗牌问题转化为单用户优化问题；②给出最优均方误差下界 dχ²，揭示 LDP 最优机制在洗牌后可能失效；③构造了可实现该下界的盲混合高斯机制；④证明了洗牌机制在高隐私区间与中心高斯机制的“高斯极限对应”。

**🔧 技术方法**

利用差分隐私中的 Hockey‑Stick 发散、洗牌指数理论、Hammersley‑Chapman‑Robbins 下界、无偏估计器设计以及大样本极限分析；在数值评估中使用 FFT 计数器对 (ε,δ) 进行精确计算。

**📊 数据集**

论文主要为理论分析与数值模拟，并未使用公开真实数据集；数值实验采用在 d‑维单位球或单位 ℓ₂ 盘上随机采样的数据点。

**📈 对比分析**

通过与中心高斯机制及现有 LDP 机制（PrivUnit、RR）在相同 (ε,δ) 下的均方误差进行比较；数值曲线显示盲混合高斯机制在高隐私下 RMSE 与中心高斯相近，且在相同 (ε,δ) 下优于 PrivUnit；在高隐私区间实现了理论下界 dσ²/n 的极限。

**⚠️ 局限性**

①要求协议无偏，限制了可能的更优偏置方案；②盲混合高斯机制在有限参数时 χ_up 与 χ_lo 仍有差距，导致理论与实践间存在细微误差；③关于 Gaussian 本地随机器的 χ_chua 计数与隐私上界的 conjecture 尚未证明；④研究仅局限于均值估计，未覆盖更一般的学习任务；⑤多轮洗牌的组合分析仍待完善。

---

## 505. Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation

**arXiv ID:** 2604.28031 | [PDF](https://arxiv.org/pdf/2604.28031v1)

**作者:** Garvin Kruthof `[一作]` (Technical University of Munich), Garvin Kruthof `[通讯]` (Technical University of Munich)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5093345216)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们构建了 DriftBench benchmark，用于评估大语言模型在多轮科研构想过程中对硬约束的遵守情况。

**💡 创新点**

创新点在于量化了约束漂移（KBV）并首次揭示了模型能够准确回忆但同时违背约束的现象，以及提供了结构化检查与监控机制。

**🔧 技术方法**

技术上采用多轮提示、约束回忆探针、结构化评判器、自动约束检测与对比评估，并结合跨模型的对照实验。

**📊 数据集**

数据集为 38 个经验证的科研简报，覆盖 24 个学科，包含硬约束与禁止动作，可公开获取。

**📈 对比分析**

通过将七款模型置于单次、无压力、中性压力、压力与检查点四种交互条件下进行对照，发现 KBV 率从 8% 至 99%，并验证检查点和自动监测仅能部分缓解约束漂移。

**⚠️ 局限性**

局限性包括：提示可能诱发指令仲裁导致漂移、无真正中性条件、仅针对科研构想任务、未涵盖 Anthropic 前沿模型、以及实验成本和算力要求。

---

## 506. Characterizing Path-Independent Fees: A Route to Zero Impermanent Loss in CPMMs

**arXiv ID:** 2604.28017 | [PDF](https://arxiv.org/pdf/2604.28017v1)

**作者:** Andrey Voronin `[一作]` (Novosibirsk State University), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5020656981)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对常数乘积做市商（CPMM）的费用结构进行系统研究，提出并证明了路径无关费用的完整函数形式，推导出对应的常微分方程和闭式交换公式，并构造了可在给定初始池状态下消除瞬时损失（IL）的参数化费用函数，同时证明不存在能对所有初始状态同时实现零IL的通用费用。

**💡 创新点**

创新点包括：① 用微分形式精确表征路径无关费用的唯一性（费率仅依赖于池的乘积k）；② 在此类费用下得到的积分公式，使得任意交易可通过闭式计算；③ 设计了一族零IL费用函数，并证明了通用零IL费用的不可行性；④ 对路径依赖的实证分析与理论验证相结合，为协议设计提供了可操作的费用优化框架。

**🔧 技术方法**

技术手段主要包括：微分方程分析（可导性条件、特征线求解），常微分方程求解与积分，闭式解析推导，数值模拟与误差分析，以及对不等式与递推关系的严谨证明。

**📊 数据集**

实验使用了模拟数据：初始储备 x₀ = y₀ = 100，交易量 Δx = 10（占10%），并以 Uniswap V2 的标准 0.3% 费用为基准进行比较；未使用外部公开数据集。

**📈 对比分析**

比较方法：① 通过分段交易的相对误差验证路径无关性；② 计算有效价格相对理想价格的偏差；③ 对比标准 Uniswap V2 与路径无关费用在相同条件下的 IL 与有效价格；结果显示路径无关费用在理论上误差可忽略（机精度），而标准费用误差在 10⁻⁵ 级；零IL费用在参考状态下实现完全零 IL，但随池状态偏离增大时费用显著上升。

**⚠️ 局限性**

局限性：零IL费用需依赖池的初始状态，仅在该状态附近有效；不适用于多资产或集中流动性池；实现时需要解决费率函数的实时计算或预先表格化；对离散交易（多次拆分）存在微小的误差，虽然在常规参数下可忽略，但在极端情况下可能影响收益。

---

## 507. Learning from Disagreement: Clinician Overrides as Implicit Preference Signals for Clinical AI in Value-Based Care

**arXiv ID:** 2604.28010 | [PDF](https://arxiv.org/pdf/2604.28010v1)

**作者:** Prabhjot Singh `[一作]`, Jung Hoon Son `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将临床人工智能系统中的临床医师覆盖（override）行为重新定义为隐式的偏好数据，利用医师专业判断、真实后果和可观测的长期结果，构建一种新的偏好学习框架；

**💡 创新点**

创新点包括：①五类覆盖类型分类体系，将覆盖原因映射到不同模型更新目标；②在偏好模型中加入病人状态s、组织合同c和医师能力κ的条件化；③双重学习架构，交替训练奖励模型和能力模型，解决“抑制偏差”（suppression bias）；④强调慢性病管理下基于结果的支付结构提供的优质覆盖信号；

**🔧 技术方法**

技术手段主要为Bradley‑Terry式偏好学习、可微的温度调节函数β(κ)、交替优化（E‑step/ M‑step）双重学习、结构化覆盖捕获、类型推断分类器、以及与临床结果关联的因果推断工具；

**📊 数据集**

数据集来自真实临床部署的覆盖记录，涵盖患者状态、系统推荐、医师响应（接受/修改/拒绝）、临床结果等，时间跨度为数月，覆盖密度高（每位医师每周数百次），并结合VBC（如CMS ACCESS）支付合同的结果标签；

**📈 对比分析**

与传统的覆盖率评估、单一奖励模型或不考虑医师能力的RLHF方法对比，作者未给出具体数值指标，但通过理论和案例说明：使用κ加权的奖励学习可避免抑制偏差，提升推荐正确率，且随着能力提升，覆盖率逐步下降，模型收敛速度加快；

**⚠️ 局限性**

局限性包括：（1）仅在单一部署环境验证，泛化性未知；（2）合同c与医师κ部分耦合，难以分离两者效应；（3）可能学习到医师偏见或歧视；（4）类型分类不确定，早期数据噪声大；（5）κ随时间可能出现突变，模型响应滞后；（6）合同变动导致模型需重新校准；

---

## 508. Exploring Sparse Matrix Multiplication Kernels on the Cerebras CS-3

**arXiv ID:** 2604.27985 | [PDF](https://arxiv.org/pdf/2604.27985v1)

**作者:** Milan Shah `[一作]` (North Carolina State University), Michela Becchi `[通讯]` (North Carolina State University)

**通讯引用:** 2913 | [OpenAlex ID](https://openalex.org/A5041520129)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对Cerebras CS-3加速器实现稀疏密集矩阵乘法（SpMM）和采样密集-密集矩阵乘法（SDDMM）内核，并针对其进行低级CSL实现与性能优化。

**💡 创新点**

提出了基于SELLPACK的稀疏存储格式、利用多I/O通道的直接路由设计、以及多累加器行并行化来降低序列化并提升带宽，显著提升了稀疏计算性能。

**🔧 技术方法**

采用CSL低级编程、SELLPACK-like稀疏格式、1.5D/2.5D分块映射、数据流并行、主机-设备双向异步传输与多累加器流式输出。

**📊 数据集**

使用随机生成的稀疏矩阵（尺寸从2,048到655,360，密度从10%到0.01%）以及常见的GNN基准图（如Cora、Pubmed、Arxiv、Products）进行评估。

**📈 对比分析**

与AMD EPYC 9354P CPU（PyTorch稀疏库和SciPy）对比，CS-3在90%稀疏度的SpMM上可达100×加速，SDDMM可达20×加速；加速比随稀疏度降低而下降，超高稀疏（>99%）时性能低于CPU。

**⚠️ 局限性**

在极高稀疏度下，CS-3的通信开销与计算不匹配导致性能下降；SELLPACK格式在低密度时存储占用大幅增大；主机-设备复制序列化与片段化处理限制了吞吐量。

---

## 509. ITS-Mina: A Harris Hawks Optimization-Based All-MLP Framework with Iterative Refinement and External Attention for Multivariate Time Series Forecasting

**arXiv ID:** 2604.27981 | [PDF](https://arxiv.org/pdf/2604.27981v1)

**作者:** Pourya Zamanvaziri `[一作]` (Shahid Beheshti University), Dara Rahmati `[通讯]` (Shahid Beheshti University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ITS‑Mina，一种全MLP的多变量时间序列预测框架。

**💡 创新点**

引入迭代细化共享参数Mixer、线性外部注意力和Harris Hawks优化自适应丢弃率。

**🔧 技术方法**

采用共享参数残差Mixer、可学习外部记忆注意力、HHO算法调优以及Optuna进行结构超参搜索。

**📊 数据集**

在Traffic、Electricity、ETTh1/2、ETTm1/2等六个公开基准数据集上进行评估。

**📈 对比分析**

与11种Transformer、MLP、Patch等基线在MSE/MAE上对比，ITS‑Mina在多数数据-预测长度组合获得最优或次优成绩。

**⚠️ 局限性**

缺乏自适应迭代停止、未加入静态特征或未来协变量，且在极高维度下仍需进一步加速。

---

## 510. Echo-α: Large Agentic Multimodal Reasoning Model for Ultrasound Interpretation

**arXiv ID:** 2604.28011 | [PDF](https://arxiv.org/pdf/2604.28011v1)

**作者:** Jing Zhang `[一作]` (Wuhan University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 100707 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种基于大规模多模态语言模型的“invoke‑and‑reason”框架，用检测器工具与全局视觉上下文交互，实现超声图像的定位与诊断；

**💡 创新点**

首次将检测器输出视为可调用的可验证证据，通过强化学习对定位与诊断两类奖励进行分离优化，形成两个互补的模型（定位和诊断），显著提升解释性与可靠性；

**🔧 技术方法**

使用Qwen3‑VL作为基础 MLLM，结合函数调用接口集成 LW‑DETR 检测器；通过九任务多阶段监督微调（REC/REG、诊断推理、工具协作、交互循环）和基于 Group Relative Policy Optimization 的强化学习，构建奖励函数（IoU、DIoU、分类、形状、工具调用惩罚）；

**📊 数据集**

在两套多中心超声基准数据集上验证：肾脏超声（COCO 格式，6 类病变）和乳腺超声（BI‑RADS 6 类），训练仅使用训练集，验证分别为同中心（Val）和异中心（Test）；

**📈 对比分析**

与六类基线（专用检测器、直接 MLLM、工具可访问/不可访问、SFT、SFT+工具、SFT+RL）对比，定位 F1@0.5 在所有四个 split 上提升 6–14%，诊断整体准确率提升 4–13%，跨中心测试保持优势；

**⚠️ 局限性**

局限性包括：仅针对肾脏和乳腺超声；对检测器性能高度依赖；需要大量标注与算力；缺乏临床真实环境评估；模型在多模态跨域迁移上的泛化仍待进一步验证。

---

## 511. MIFair: A Mutual-Information Framework for Intersectionality and Multiclass Fairness

**arXiv ID:** 2604.28030 | [PDF](https://arxiv.org/pdf/2604.28030v1)

**作者:** Jeanne Monnier `[一作]` (Orange Research), Marios Kountouris `[通讯]` (EURECOM)

**通讯引用:** 10590 | [OpenAlex ID](https://openalex.org/A5088602155)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MIFair，一种基于互信息的统一框架，用于评估与缓解机器学习模型中的偏见，支持交叉性和多类别场景；

**💡 创新点**

创新点在于将多种公平性定义统一为互信息指标，并通过正则化实现可调的内插处理，解决了传统方法对单属性、二元分类的局限；

**🔧 技术方法**

主要技术包括互信息计算与正则化、基于批量估计的经验互信息、对不同公平定义的特征映射及其在深度学习中的实现；

**📊 数据集**

实验数据集包括UCI Adult（表格数据）和CelebA（图像数据），分别评估二分类和多分类任务；

**📈 对比分析**

通过与经典公平度量（SPD、EOD、OAE）以及KDE基准方法比较，MIFair在保持较低准确率下降的同时显著降低互信息和公平差距；

**⚠️ 局限性**

局限性包括对离散特征的依赖、互信息估计在小样本或极度不平衡时不稳健、需要手动调节正则化强度以及对连续输出的支持尚未完善。

---

## 512. ResiHMR: Residual-Limb Aware Single-Image 3D Human Mesh Recovery for Individuals with Limb Loss

**arXiv ID:** 2604.28025 | [PDF](https://arxiv.org/pdf/2604.28025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. From Impermanent Loss to Sustainable Gain: Quantifying Profitability Zones for Liquidity Providers on DEX

**arXiv ID:** 2604.28014 | [PDF](https://arxiv.org/pdf/2604.28014v1)

**作者:** Ignat Melnikov `[一作]` (Skolkovo Institute of Science and Technology), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5020656981)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并量化去中心化交易所（AMM）中流动性提供者与套利者的联合盈利区间，构建数学模型并通过仿真与私有池实验验证可持续盈利（IG）区间及手续费优化。

**💡 创新点**

首次给出常数乘积AMM的可持续盈利区间闭式边界，将Impermanent Loss转化为可控风险参数；提出概率风险框架与手续费优化方法，并证明协作机制可消除IL、提升整体收益。

**🔧 技术方法**

采用数学建模（常数乘积与加权常数乘积Invariant）、几何布朗运动概率上界、数值根查找；实验使用Polygon V3式私有池部署与区块链交易记录。

**📊 数据集**

使用公开的USDC/RAD、USDT/WMATIC池配置数据；对Uniswap V2与Balancer的实测池参数；模拟生成的GBM价格路径；实验收集的交易与收益记录。

**📈 对比分析**

通过理论边界与实验数据比对：几何分布预测IL块数与仿真一致；手续费最小化曲线与实验费率对照；实验结果显示0.03%手续费池收益≈35 MATIC，0%手续费池≈25 MATIC，证明手续费机制提升总收益。

**⚠️ 局限性**

假设理想套利者、无竞争/MEV、即时执行；仅覆盖单资产常数乘积AMM，未涉及聚焦流动性或多资产池；实验周期短、单一套利者，未考虑多套利者竞争或动态手续费；对网络延迟、拥堵影响忽略。

---

## 514. Distributed Santa Claus via Global Rounding

**arXiv ID:** 2604.27983 | [PDF](https://arxiv.org/pdf/2604.27983v1)

**作者:** Tijn de Vos `[一作]` (TU Graz), Florian Schager `[通讯]` (University of Southern Denmark)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了分布式环境下的 Santa Claus 分配问题，并给出了在 (D + √n) 轮内实现 (log n/ log log n)-近似的算法。

**💡 创新点**

首次给出了该问题的近似上界与下界，并提出了适用于混合包装-覆盖线性规划的分布式求解器，突破了匹配、负载平衡等问题的局限。

**🔧 技术方法**

核心技术包括将 Santa Claus 转化为混合包装-覆盖 LP、利用低直径分解、T-join、短环覆盖及全局循环消除等分布式优化手段，随后通过随机化全局取样和多阶段舍入得到整数解。

**📊 数据集**

论文仅在合成的稀疏图和二叉树叠加的路径实例上进行实验验证，未使用公开数据集。

**📈 对比分析**

与先前仅针对匹配、负载平衡的 O(log n) 轮算法相比，本文实现了与下界相匹配的 (D+√n) 轮，且在近似误差上为 (log n/ log log n)，显示出更好的时间-近似权衡。

**⚠️ 局限性**

局限性包括：仅针对受限分配版本的 Santa Claus，近似比例仍为对数级别；算法实现依赖于多阶段全局通信，常数因子大；对更一般的多元、非二进制权重的情况尚未覆盖。

---

## 515. From LLM-Driven Trading Card Generation to Procedural Relatedness: A Pokémon Case Study

**arXiv ID:** 2604.27972 | [PDF](https://arxiv.org/pdf/2604.27972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 516. Agent-Agnostic Evaluation of SQL Accuracy in Production Text-to-SQL Systems

**arXiv ID:** 2604.28049 | [PDF](https://arxiv.org/pdf/2604.28049v1)

**作者:** Taslim Jamal Arif `[一作]`, Kuldeep Singh `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 STEF，一个不依赖数据库模式或参考查询的生产环境 Text‑to‑SQL 评估框架，利用用户原始问题、丰富的重述和生成的 SQL 通过语义特征对齐和规则校正产生 0–100 的可解释准确率。

**💡 创新点**

创新点包括：① 同时从自然语言和 SQL 两侧抽取结构化语义规范并对齐；② 通过可注入的应用规则实现部署级别的可定制化；③ 引入四条生产规范化规则（必选 GROUP BY、惰性 GROUP BY、默认 ORDER BY、超大 LIMIT 保护）以消除误报，并将 LLM 判定与置信度融合成复合评分。

**🔧 技术方法**

技术手段包括：大语言模型作为评判器的 Prompt‑Engineered 评估器、语义特征提取与归一化、基于规则的过滤对齐、置信度加权的复合评分公式，以及运行时注入的 JSON 规则配置。

**📊 数据集**

实验使用了真实生产环境中三款 T2SQL 代理（Agent‑A、Agent‑B、Agent‑C）产生的查询样本；论文参考了 Spider、WikiSQL、BIRD 等公开基准，但 STEF 并未依赖这些数据集。

**📈 对比分析**

与传统基于字符串匹配和执行准确率基准对比，STEF 通过持续监控得到的平均分数（Agent‑A 87.4，Agent‑B 82.1，Agent‑C 91.3）和 P90 覆盖率（98–99%）显示出在生产环境下更高的可靠性和可解释性。

**⚠️ 局限性**

局限性包括：LLM 评估器的随机波动仍然存在，尤其在边缘案例；对深层嵌套子查询、窗口函数等复杂 SQL 的评估准确性下降；以及在完全无模式的环境下对极端异常查询的判定仍有改进空间。

---

## 517. SpecVQA: A Benchmark for Spectral Understanding and Visual Question Answering in Scientific Images

**arXiv ID:** 2604.28039 | [PDF](https://arxiv.org/pdf/2604.28039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 518. Exponential families from a single KL identity

**arXiv ID:** 2604.28036 | [PDF](https://arxiv.org/pdf/2604.28036v1)

**作者:** Marc Dymetman `[一作]` `[通讯]` (Scientific Consultant), Marc Dymetman (Scientific Consultant)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并证明了一个关于指数族的KL差异身份，统一推导出三点身份、Pythagorean定理、I投影、凸性、ELBO、KL正则化奖励最大化等结果

**💡 创新点**

通过单一代数身份揭示KL差异、对数分区函数与矩的线性关系，避免了传统凸分析和拉格朗日乘子法，提供了一种全新、统一的理论框架

**🔧 技术方法**

核心技术为指数族的对数比值线性性质、期望运算以及KL非负性；在可测空间下推广了求和到积分的处理，并利用可微性证明梯度与矩的一一对应

**📊 数据集**

无具体数据集，论文以理论推导为主，针对离散与可测空间两类取样空间给出通用结果

**📈 对比分析**

未做实验对比；通过代数推导证明所得到的各种优化与投影公式是最优/唯一的，并给出与传统ELBO、Bregman散度等经典结果的对应关系

**⚠️ 局限性**

局限性在于需假设指数族成员存在且矩可积，且对高阶分析（如Fisher信息、解析性等）未展开；在连续空间中实现时需满足可微性与可积性条件

---

## 519. Dreaming Across Towns: Semantic Rollout and Town-Adversarial Regularization for Zero-Shot Held-Out-Town Fixed-Route Driving in CARLA

**arXiv ID:** 2604.27994 | [PDF](https://arxiv.org/pdf/2604.27994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 520. Cost-Aware Learning

**arXiv ID:** 2604.28020 | [PDF](https://arxiv.org/pdf/2604.28020v1)

**作者:** Clara Mohri `[一作]` (Harvard University), Yishay Mansour `[通讯]` (Tel Aviv University)

**通讯引用:** 21572 | [OpenAlex ID](https://openalex.org/A5014637159)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文考虑了成本感知学习的问题，提出了一种成本感知随机梯度下降（Cost-Aware SGD）算法，旨在在达到目标误差的同时最小化总成本。

**💡 创新点**

创新点在于提出了成本感知学习的框架，并在此基础上发展了成本感知SGD和成本感知GRPO算法，能够在保持性能的同时显著降低训练成本。

**🔧 技术方法**

使用了成本感知随机梯度下降（Cost-Aware SGD）和成本感知GRPO算法，结合了重要性采样的原理。

**📊 数据集**

实验使用了Qwen2.5-Math-1.5B-Instruct和Qwen3-8B模型，并在多个基准（如MATH500、AMC、GSM8K和AIME1983-2024）上进行评估。

**📈 对比分析**

与均匀采样和方差策略进行了比较，结果表明成本感知方法在达到相同准确率时，所需的token数量减少了30%，并且在某些情况下超越了基线的性能。

**⚠️ 局限性**

限制在于当前方法主要依赖于优势的大小作为梯度范数的代理，未来的研究可以探索其他代理的有效性。

---

## 521. Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning

**arXiv ID:** 2604.27998 | [PDF](https://arxiv.org/pdf/2604.27998v1)

**作者:** Jingcheng Deng `[一作]` (State Key Laboratory of AI Safety Institute of Computing Technology Chinese Academy of Sciences), Huawei Shen `[通讯]` (State Key Laboratory of AI Safety Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在隐式推理（latent reasoning）中使用强化学习（RL）进行后训练，提出了一种名为Latent-GRPO的稳定高效方法，显著提升了推理性能并压缩了推理链长度。

**💡 创新点**

创新点主要包括：1）无效样本优势屏蔽（Invalid Sample Advantage Masking）用于约束离散化后未终止的轨迹；2）单侧噪声采样（One-sided Noise Sampling）确保Gumbel噪声正向偏移，从而消除探索-优化失配；3）最优正确路径首令选择（Optimal Correct Path First Token Selection）避免多路径平均导致无效状态。

**🔧 技术方法**

技术上基于GRPO/Soft‑GRPO的策略梯度框架，结合隐式词汇（latent vocabulary）和Top‑K混合嵌入、Gumbel重参数化、KL约束、以及自定义优势计算与更新掩码。

**📊 数据集**

实验使用两类数据集：低难度（GSM8K‑Aug、GSM‑Hard、SVAMP、MultiArith）与高难度（Math500、AIME24、AIME25、GPQA），模型分别为LLaMA‑3.2‑1B‑Instruct与Qwen2.5‑MATH‑7B，且对比了显式SFT+GRPO、Latent‑SFT+Soft‑GRPO与Latent‑GRPO。

**📈 对比分析**

与显式推理相比，Latent‑GRPO在低难度任务上Pass@1提升7.86点、链长度缩短4.44×；在高难度任务上提升14.77点、链长度缩短3.31×，并在pass@k上也优于显式GRPO。

**⚠️ 局限性**

局限性：1）在无采样模式下模型确定性，采样会导致pass@1下降；2）依赖Latent‑SFT预训练，若预训练不足性能受限；3）主要验证于数学推理任务，尚未检验更广泛的跨模态或多步推理场景。

---

## 522. FineState-Bench: Benchmarking State-Conditioned Grounding for Fine-grained GUI State Setting

**arXiv ID:** 2604.27974 | [PDF](https://arxiv.org/pdf/2604.27974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 523. Ease of dependency distance minimization in star-like structures

**arXiv ID:** 2604.28034 | [PDF](https://arxiv.org/pdf/2604.28034v1)

**作者:** Emília Garcia-Casademont `[一作]`, Ramon Ferrer-i-Cancho `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过数学证明，展示了星形结构与其近似结构（quasistar）在依赖距离最小化问题上具有凸（convex）优化景观，并进一步证明了星形结构的最小化代价呈二次函数，因而相对于其他树形结构（如路径树）在依赖距离最小化时收益最低，解释了实验中观察到的反向依赖距离最小化现象。

**💡 创新点**

创新点在于将离散凸分析与树形语法结构相结合，首次证明星形与quasistar结构的依赖距离代价函数在所有单调递增的成本函数下保持凸性，并揭示了其二次成本特性与其他结构的根本差异，从而为语言学中依赖距离最小化的逆现象提供了理论解释。

**🔧 技术方法**

采用离散凸性理论（Discrete Convex Analysis）、凸序列与强离散凸性（L♮-convexity）的数学工具，对成本函数 D(l) 和 D_q(l,p,q) 进行推导和证明，并利用计算实验验证凸性与成本函数的二次增长特性。

**📊 数据集**

使用人工生成的依赖树数据集，包括 n=3、4、5 的星形、quasistar 与路径树结构，以及随机线性排列的基准来计算 D_min、D_max、D_r 等指标，验证理论结果。

**📈 对比分析**

通过与随机排列的期望成本 D_r 以及路径树的线性最小成本 D_min^p 的比较，展示星形结构在依赖距离最小化时的成本保持二次增长，导致与路径树相比收益最低，实验结果与理论预测高度一致。

**⚠️ 局限性**

局限性在于仅考虑了星形和quasistar两类结构，未对更一般的树形（如bistar、caterpillar）进行证明；实验样本仅覆盖 n≤5 的结构；并假设成本函数为单调递增的线性或二次形式，对更复杂的非线性成本未作深入分析。

---

## 524. FedHarmony: Harmonizing Heterogeneous Label Correlations in Federated Multi-Label Learning

**arXiv ID:** 2604.28024 | [PDF](https://arxiv.org/pdf/2604.28024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 525. Are DeepFakes Realistic Enough? Exploring Semantic Mismatch as a Novel Challenge

**arXiv ID:** 2604.28022 | [PDF](https://arxiv.org/pdf/2604.28022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 526. Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification

**arXiv ID:** 2604.28016 | [PDF](https://arxiv.org/pdf/2604.28016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 527. Exploring Interaction Paradigms for LLM Agents in Scientific Visualization

**arXiv ID:** 2604.27996 | [PDF](https://arxiv.org/pdf/2604.27996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 528. A Pattern Language for Resilient Visual Agents

**arXiv ID:** 2604.28001 | [PDF](https://arxiv.org/pdf/2604.28001v1)

**作者:** Habtom Kahsay Gidey `[一作]` (Technische Universität München), Alois Knoll `[通讯]` (Technische Universität München)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种面向可恢复视觉代理的层次化模式语言，设计四种架构模式（Hybrid Affordance Integration、Adaptive Visual Anchoring、Visual Hierarchy Synthesis、Semantic Scene Graph），通过将高延迟、概率化的视觉语言动作（VLA）模型置于监督层，低延迟的规则化反射层置于执行层，实现了在企业GUI自动化中的实时控制与语义适应的权衡。

**💡 创新点**

创新点在于将软件架构与认知科学（System 1/2、MAPE‑K、子消失架构）结合，形成可“摊销推理”的层次化控制循环；同时提出多模态感知仲裁、视觉锚点回退、层次化UI合成和可查询语义场景图等四种可复用模式，解决了实时性、成本、可靠性和可解释性之间的冲突。

**🔧 技术方法**

使用的技术包括：多模态感知融合（视觉目标检测+OCR），基于置信度阈值的锚点回退路由器，Gestalt启发的空间聚类生成UI层次树，构建可查询的语义场景图，MAPE‑K与子消失架构的结合，VLA模型（如CogAgent、UI‑TARS）作为监督层，低延迟本地缓存（hash‑map）与视觉锚点的结合。

**📊 数据集**

评估主要采用基于场景的架构分析方法（SAAM）构建的简化财务ERP维基更新案例，未使用公开数据集，仅使用人工构造的模拟UI更新场景。

**📈 对比分析**

与经典RPA脚本（无视觉感知、固定坐标）和纯端到端VLA系统（延迟≈10 s、成本高）对比，所提架构在UI漂移事件中保持<1 s的平均执行时间，且在大多数案例中仅触发一次高成本VLA推理，显著降低整体延迟与成本，同时通过锚点回退避免误操作，安全性得到保证。

**⚠️ 局限性**

局限性包括：仅在单一人工构造的UI漂移场景下验证，缺乏大规模真实工业案例的定量评估；对VLA模型的依赖仍存在推理失败的风险；模式组合的细粒度参数（如阈值τ）需要经验调优，未给出通用的自动化配置方法。

---

## 529. D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery

**arXiv ID:** 2604.27977 | [PDF](https://arxiv.org/pdf/2604.27977v1)

**作者:** Hanane Nour Moussa `[一作]` (Ohio State University), Huan Sun `[通讯]` (Ohio State University)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5101488340)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了首个自动生成可验证的科学数据驱动发现环境数据集D3-Gym，包含565个任务，每个任务提供可执行环境、数据预览、自然语言指令、参考代码与自动生成的评估脚本。

**💡 创新点**

创新点在于通过两阶段规划+编码的LLM流程自动生成科学任务专用评估脚本，并利用这些可验证环境实现开放权重模型在真实科研任务上的训练与评测。

**🔧 技术方法**

主要技术包括基于Claude Sonnet 4.5和GPT-5.2等LLM的任务筛选、环境构建、输出验证与评估脚本生成；自动化流水线和拒绝采样微调（RFT）用于模型训练。

**📊 数据集**

使用数据集包括自建的D3-Gym（565任务、239科研仓库来源），验证集50个手工评估的金标准评估脚本，以及ScienceAgentBench和其Verified版本作为性能评测基准。

**📈 对比分析**

在ScienceAgentBench与Verified上对Qwen3系列模型进行RFT-Distill/Self训练，32B模型SR@3提升至约58%，超过多款专有模型，评估脚本的准确率达到87.5%。

**⚠️ 局限性**

局限性包括：任务难度高导致模型执行率仍低；逻辑/算法错误仍普遍；LLM对专业库的知识有限；评估脚本略为严格，导致召回率偏低。

---

## 530. Reliable Answers for Recurring Questions: Boosting Text-to-SQL Accuracy with Template Constrained Decoding

**arXiv ID:** 2604.28028 | [PDF](https://arxiv.org/pdf/2604.28028v1)

**作者:** Smit Jivani `[一作]` (Indian Institute of Technology Bombay), Sunita Sarawagi `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 10741 | [OpenAlex ID](https://openalex.org/A5031035935)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于模板约束解码的 Text-to-SQL 系统 TeCoD，利用已标注的查询模板提升重复查询的执行准确率并降低延迟。

**💡 创新点**

将历史 NL-SQL 对转换为可复用模板，设计基于 NLI 的模板匹配器，采用分区语法约束解码实现高效准确的 SQL 生成。

**🔧 技术方法**

使用模板化、自然语言推理 (NLI) 匹配、基于 CFG 的语法约束解码（Outlines）、分区解码与上下文补全、LLM（如 Llama、Granite、CodeS、QwenCoder）等技术。

**📊 数据集**

在公开的 BIRD、Spider Text-to-SQL 基准数据集上进行实验，并结合银行内部工作负载进行验证。

**📈 对比分析**

与零射、ICL-3、模板软引导（-SGC）等方法对照，TeCoD‑GCD 在匹配查询上将执行准确率从约 60% 提升至近 90%，平均提升 36%，同时推理延迟降低约 2.2 倍。

**⚠️ 局限性**

仅适用于已出现的模板，对尾部查询仍依赖 ICL；模板只遮蔽常量，无法处理更灵活的模板；在新颖查询时模板匹配误判会降低整体准确率。

---

## 531. Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-World Workflows

**arXiv ID:** 2604.28139 | [PDF](https://arxiv.org/pdf/2604.28139v1)

**作者:** Chenxin Li `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9891 | [OpenAlex ID](https://openalex.org/A5073968803)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一个实时更新的工作流代理评测基准，结合公共工作流信号生成任务，并在可复现的快照中评估代理执行。

**💡 创新点**

任务混合基于实时公共信号刷新，保持与现实需求一致；评分基于可观测执行轨迹与证据，兼顾确定性检查和结构化LLM判定。

**🔧 技术方法**

采用信号到任务的流水线（ClawHub Top-500 过滤、聚类、加权、种子扩展、MILP 选取）、控制服务与沙箱工作区执行、轨迹记录、确定性验证与LLM评判、公开基准发布。

**📊 数据集**

公开的 ClawHub Top-500 技能列表作为信号源，服务端 fixture、API audit logs、工作区状态日志以及 105 个任务的执行记录。

**📈 对比分析**

通过 Pass Rate 与 Overall Completion Score 两项指标比较模型，公开榜单显示最佳模型 Claude Opus 4.6 通过率 66.7%，整体完成 83.6%，其余模型多在 50‑60% 之间，工作区修复任务通行率高，而基于服务的工作流低于 60%。

**⚠️ 局限性**

仍未达到 70% 的通行率，服务端多系统协调任务仍难；评测受限于公共信号的覆盖和任务设计，部分任务分辨率低；LLM 判定可能存在模型偏差；基准不覆盖所有真实业务场景，且与具体部署需求仍需结合。

---

## 532. Neural Aided Kalman Filtering for UAV State Estimation in Degraded Sensing Environments

**arXiv ID:** 2604.28107 | [PDF](https://arxiv.org/pdf/2604.28107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 533. Mapping the Methodological Space of Classroom Interaction Research: Scale, Duration, and Modality in an Age of AI

**arXiv ID:** 2604.28098 | [PDF](https://arxiv.org/pdf/2604.28098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 534. Beyond Code, We Are People: A Systematic Mapping of 25 Years of Literature on Soft Skills in Agile Development Teams

**arXiv ID:** 2604.28101 | [PDF](https://arxiv.org/pdf/2604.28101v1)

**作者:** Israely Lima `[一作]` (Federal University of Ceara), Carla Ilane Bezerra `[通讯]` (Federal University of Ceara)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了2000‑2025年间关于敏捷软件开发团队软技能的文献，分析了97篇原始研究，归纳了33种软技能及其在不同角色和敏捷方法中的分布与趋势

**💡 创新点**

首次在25年周期内完成软技能的系统性映射，提出按角色与敏捷方法细分的软技能分类，并揭示了敏捷实践对软技能需求的差异化影响

**🔧 技术方法**

采用系统映射研究方法（SMS），利用精细的检索字符串、六大数据库检索、双人筛选与数据提取表格

**📊 数据集**

基于六大数据库（ACM、IEEE、SBC、ScienceDirect、Scopus、Springer）共检索2083篇文献，最终纳入97篇原始研究作为数据集

**📈 对比分析**

通过频次统计与趋势标记（上升/稳定）对软技能进行定量描述，未进行实验对比或性能评估，而是呈现各软技能在文献中的出现频率与时间演变情况

**⚠️ 局限性**

局限性包括：仅纳入英文、葡萄牙语、西班牙语论文，导致语言偏倚；大部分研究未明确敏捷方法或角色层级，限制了对特定方法/层级的深入分析；缺乏对新兴角色（如DevOps、AI工程师）的关注，导致结果在多样化团队中的适用性受限

---

## 535. Global Optimality for Constrained Exploration via Penalty Regularization

**arXiv ID:** 2604.28144 | [PDF](https://arxiv.org/pdf/2604.28144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 536. On the Proper Treatment of Units in Surprisal Theory

**arXiv ID:** 2604.28147 | [PDF](https://arxiv.org/pdf/2604.28147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 537. Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing

**arXiv ID:** 2604.28142 | [PDF](https://arxiv.org/pdf/2604.28142v1)

**作者:** Silvio Martinico `[一作]` (University of Pisa and ISTI--CNR), Rossano Venturini `[通讯]` (University of Pisa)

**通讯引用:** 1662 | [OpenAlex ID](https://openalex.org/A5084138015)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于 token 结构的聚类方法（Token-Aware Clustering）和层次索引架构，以显著加速多向量检索流程

**💡 创新点**

创新点在于：①使用 token 频率与语义方差驱动的聚类分配，分解全局 k‑means 为 per‑token 子问题，消除高频 token 主导的负载；②构建基于图的 centroid 索引和缓存友好的 PQ 布局，实现仅用 centroid 进行候选检索；③理论上给出最低加速下界并在实践中实现 247× 的聚类加速

**🔧 技术方法**

采用 Token‑Aware Clustering、图索引（如 HNSW）、Product Quantization（PQ）、Late Interaction 计算、缓存友好距离表布局，以及 Rust 语言实现

**📊 数据集**

使用 MS MARCO passage（8.8M passage，598M token 向量）和 C4‑pooled（2.4M passage，266M token 向量）两个大型检索基准数据集

**📈 对比分析**

与 k‑means、ColBERT 以及 QL/QL+/QL++ 等 state‑of‑the‑art 多向量检索方法对比，聚类速度提升至 247×，检索速度提升至 9.8×，在 MRR@10 / Success@5 等指标上保持或超过现有最优效果

**⚠️ 局限性**

局限性：仅在 ColBERT 编码器上验证，尚未评估更大规模或多语言数据的泛化性；残差压缩比仍可进一步提升；实现仅基于 CPU，缺乏 GPU 加速

---

## 538. Beyond Pixel Fidelity: Minimizing Perceptual Distortion and Color Bias in Night Photography Rendering

**arXiv ID:** 2604.28136 | [PDF](https://arxiv.org/pdf/2604.28136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 539. PRISM: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning

**arXiv ID:** 2604.28123 | [PDF](https://arxiv.org/pdf/2604.28123v1)

**作者:** Sudong Wang `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了三阶段后训练流水线 PRISM：先进行大规模监督微调（SFT），随后使用 Mixture-of-Experts (MoE) 判别器进行对抗式 on‑policy 重新分布对齐（alignment），最后在对齐后的模型上执行可验证奖励的强化学习（RLVR）以提升多模态推理性能。

**💡 创新点**

创新点在于：①将 on‑policy distillation 设为独立的对齐阶段，而非直接作为终结训练目标；②设计专门的 MoE 判别器，分别评估视觉表征和推理轨迹，为不同类型的分布漂移提供解耦反馈；③利用黑盒对抗游戏，无需教师模型 logits；④在对齐阶段使用精细化视觉与推理的双重奖励，显著缩小 SFT 引入的分布漂移。

**🔧 技术方法**

核心技术包括：supervised fine‑tuning (SFT)、adversarial on‑policy distillation、Mixture‑of‑Experts 判别器（含视觉专家和推理专家）、GRPO/DAPO/GSPO 等基于可验证奖励的强化学习算法。

**📊 数据集**

数据集：1.26M 公共 Gemini 3 Flash 示例 + 113K 由 Gemini 3 Flash 生成、包含完整视觉定位和逐步推理的高质量示例；评测使用 MathVista、MathVerse、MathVision、WeMath（数学推理）以及 MMMU、MMMU‑Pro、HallusionBench（通用多模态理解）等基准。

**📈 对比分析**

与基线（未后训练、仅 SFT、SFT→RLVR）相比，PRISM 在 Qwen3‑VL‑4B 和 8B 上平均提升 4.4 / 6.0 分，且在 GRPO、DAPO、GSPO 三种 RL 算法上均保持一致性提升；对齐阶段本身虽未直接提高答案准确率，但显著降低了分布偏差，为后续 RL 提供更优初始化。

**⚠️ 局限性**

局限性：①对 SFT 训练数据规模和质量高度依赖，需海量高质量视觉‑推理示例；②对齐阶段实现复杂，训练过程需要多次对抗更新；③实验仅验证在大规模 LMM（4B/8B）上，未探讨小模型或跨模型的通用性；④MoE 判别器与对齐策略在不同任务/数据分布下的鲁棒性尚待进一步验证。

---

## 540. Do Sparse Autoencoders Capture Concept Manifolds?

**arXiv ID:** 2604.28119 | [PDF](https://arxiv.org/pdf/2604.28119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 541. Towards Neuro-symbolic Causal Rule Synthesis, Verification, and Evaluation Grounded in Legal and Safety Principles

**arXiv ID:** 2604.28087 | [PDF](https://arxiv.org/pdf/2604.28087v1)

**作者:** Zainab Rehan `[一作]` (Hasso Plattner Institute University of Potsdam), Holger Giese `[通讯]` (Hasso Plattner Institute University of Potsdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大型语言模型的增量式规则合成与验证框架，能将人类自然语言目标拆解为候选因果、合并语义、符号化规则，并通过必要性与充分性分析得到可追溯、最小化的规则集，随后通过语法、逻辑一致性与安全性检查验证后再集成到自适应系统的知识库中；

**💡 创新点**

创新点在于将神经符号因果推理与 LLM 驱动的规则生成相结合，构建了一个 meta‑level 的合成‑验证闭环，使规则维护变得增量化、模块化、可追溯，并显著缓解了传统规则系统的脆弱性与扩展难题；

**🔧 技术方法**

核心技术包括 GPT‑4o Mini 大语言模型用于目标拆分、语义归约与符号化；First‑Order Logic（FOL）规则与结构化因果模型（SCM）构成的神经符号框架；自动推理工具（如 Z3）进行语法与一致性验证；以及 abduction/deduction 组合的必要/充分性评估；

**📊 数据集**

实验所用数据主要为预先整理的德国交通法规和安全原则的正式化规则集，辅以自动生成的补充规则，模拟了两种自驾场景（拥堵合流与高速保持恒速）；

**📈 对比分析**

在两种场景下，系统成功生成了唯一的最小必要/充分规则集合，并通过手工检查确认其逻辑一致性与安全约束满足；由于缺乏量化基准，本文未给出与传统专家系统或手工规则库的性能对比；

**⚠️ 局限性**

主要局限在于 LLM 的黑箱性导致内部推理不透明；规则生成高度依赖预先编写的法规集，可能无法覆盖所有情况；对提示语的敏感性使结果易受人为干预影响；并且目前缺乏概率或多模型协同机制来进一步提升鲁棒性。

---

## 542. Perfectly Private Over-the-Air Computation

**arXiv ID:** 2604.28080 | [PDF](https://arxiv.org/pdf/2604.28080v1)

**作者:** Shudi Weng `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8950 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 P^2-AirComp 协议，在有限功率约束下实现了空中计算的完美隐私与准确结果。

**💡 创新点**

创新点在于交叉域设计，结合实域与模运算的周期性包装，消除隐私与效能之间的权衡，得到信息理论上的完美隐私。

**🔧 技术方法**

采用模零和同构密钥生成、信道预均衡、模运算加噪声、MSE 闭式分析与上界下界推导等技术。

**📊 数据集**

使用人工生成的随机消息与仿真 Rician 信道数据，未采用公开数据集。

**📈 对比分析**

与独立噪声、相关噪声、零和噪声等三种私有 AirComp 方法对比；在相同功率约束下，P^2-AirComp 达到零信息泄露且 MSE 接近最佳方案，显著优于现有方案。

**⚠️ 局限性**

限制在于当客户端数小于 3 时无法保证客户端间隐私；对信道估计误差和硬件非理想性的鲁棒性尚未完整验证。

---

## 543. AesRM: Improving Video Aesthetics with Expert-Level Feedback

**arXiv ID:** 2604.28078 | [PDF](https://arxiv.org/pdf/2604.28078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 544. A Unified Framework of Hyperbolic Graph Representation Learning Methods

**arXiv ID:** 2604.28070 | [PDF](https://arxiv.org/pdf/2604.28070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 545. TopBench: A Benchmark for Implicit Prediction and Reasoning over Tabular Question Answering

**arXiv ID:** 2604.28076 | [PDF](https://arxiv.org/pdf/2604.28076v1)

**作者:** An-Yang Ji `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**通讯引用:** 3580 | [OpenAlex ID](https://openalex.org/A5065180062)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 TopBench benchmark，用于评估大语言模型在表格数据中的隐式预测推理能力。

**💡 创新点**

创新点在于将表格问答从显式检索转向隐式预测，提出四类任务（单点预测、决策制定、治疗效应分析、排序过滤），并设计多阶段意图抽象+预测推理流程及基于 LLM‑as‑a‑Judge 的评估方法。

**🔧 技术方法**

采用 LLM（如 GPT‑5.2、Claude‑Sonnet‑4.5、Gemini‑3 Flash 等）与 ReAct 代码执行、链式思考、工具调用等技术；评估指标包括准确率、逻辑分数、决策/趋势准确率、NDCG、NMAE、F1；使用预测器集成与特征处理等预测模块。

**📊 数据集**

数据集为 779 条样本，来源于 35 张历史表格，涵盖医疗、金融、日常咨询三大领域，表格规模从 <1k 行到 >6M 行不等。

**📈 对比分析**

对比方法采用文本推理与 agentic 代码执行两种推理范式，评测多模型；实验显示大模型在意图识别与预测推理上均低于 0.65，决策与治疗效应任务近似随机，排序过滤任务准确率较低。

**⚠️ 局限性**

局限性：模型难以准确识别预测意图，常误认为检索任务；缺乏高精度预测能力与鲁棒的代码执行；Bench 仅给定已配对表格，未覆盖开放式检索与自动表格发现。

---

## 546. MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons

**arXiv ID:** 2604.28130 | [PDF](https://arxiv.org/pdf/2604.28130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 547. Normativity and Productivism: Ableist Intelligence? A Degrowth Analysis of AI Sign Language Translation Tools for Deaf People

**arXiv ID:** 2604.28125 | [PDF](https://arxiv.org/pdf/2604.28125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 548. GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment

**arXiv ID:** 2604.28111 | [PDF](https://arxiv.org/pdf/2604.28111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 549. Characterizing the Consistency of the Emergent Misalignment Persona

**arXiv ID:** 2604.28082 | [PDF](https://arxiv.org/pdf/2604.28082v1)

**作者:** Anietta Weckauff `[一作]` (Max Planck Institute for Intelligent Systems), Maksym Andriushchenko `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Qwen 2.5 32B模型上通过LoRA微调六类细化不良数据，评估其有害行为与自我报告，发现两种不同的EM人格模式：一类行为与自我报告一致，另一类行为不一致；

**💡 创新点**

首次系统性揭示EM人格在不同细化域中呈现的两种对立模式，并证明自我报告的可靠性取决于细化域；

**🔧 技术方法**

采用LoRA微调、GPT‑4o mini判分器评估有害性、四种格式自评、两AI识别、输出识别、得分预测和跨模型评分等技术；

**📊 数据集**

六个细化域数据集：不安全代码、风险金融建议、错误医疗建议、极限运动建议、法律咨询与安全咨询；

**📈 对比分析**

与未微调基线模型对比，使用有害响应比例、自评分数、两AI识别率、输出识别准确率和得分预测误差衡量。结果显示，风险金融、极限运动和错误医疗模型表现为“连贯人格”，即有害行为与自评高度相关；不安全代码、法律与安全模型表现为“倒置人格”，即有害行为高但自评趋向对齐；

**⚠️ 局限性**

实验仅覆盖Qwen 2.5 32B和六个数据集，缺乏跨模型、跨规模验证；两AI识别可能受表面特征影响；自评与激活方向独立的原因尚不清楚；

---

## 550. Continuous-tone Simple Points: An $\ell_0$-Norm of Cyclic Gradient for Topology-Preserving Data-Driven Image Segmentation

**arXiv ID:** 2604.28159 | [PDF](https://arxiv.org/pdf/2604.28159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 551. FlashRT: Towards Computationally and Memory Efficient Red-Teaming for Prompt Injection and Knowledge Corruption

**arXiv ID:** 2604.28157 | [PDF](https://arxiv.org/pdf/2604.28157v1)

**作者:** Yanting Wang `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**通讯引用:** 2141 | [OpenAlex ID](https://openalex.org/A5101997385)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FlashRT，一个针对长上下文 LLM 的优化红队框架，显著降低了基于梯度的 prompt injection 与知识破坏攻击的计算时间和 GPU 内存消耗。

**💡 创新点**

核心创新在于：
• 前向传播的“选择性重计算”——仅对对目标答案注意力最高的右侧上下文子句重新计算隐藏状态，近似损失；
• 后向传播的“子采样梯度”——随机抽取上下文子段以减少梯度存储；
• 用注意力权重构造影响分数（Influence Score）快速挑选重计算子句；
• 结合梯度重采样机制避免陷入局部最优；
• 兼容并加速现有白盒/黑盒方法（如 nanoGCG、TAP、AutoDAN）。

**🔧 技术方法**

采用 KV‑Caching、Transformer 注意力机制、梯度子采样、影响分数计算、梯度重采样、log‑prob 近似等技术；实现基于 PyTorch SDPA，支持多种 LLM（Llama‑3.1‑8B/13B/70B、Qwen‑2.5‑7B/14B、Mistral‑7B、DeepSeek‑R1‑Distill 等）。

**📊 数据集**

使用 LongBench 上的 MuSiQue、NarrativeQA、GovReport 进行 Prompt Injection；NQ、HotpotQA、MS‑MARCO 进行知识破坏；并在 Meta‑SecAlign、Llama‑Prompt‑Guard 等防御模型上进行红队评估。

**📈 对比分析**

与基准（Heuristic Attack、Context Clipping、nanoGCG、改进 nanoGCG）对比：ASR 与基准相当或更高，平均计算时间下降 2×–7×，GPU 内存下降 2×–4×；在 70B LLM 上，原方法不可执行而 FlashRT 能完成攻击；在防御模型上也比传统红队方法提高 70%+ ASR，同时加快 3×、内存 3×。

**⚠️ 局限性**

局限性包括：
• 对注入位置敏感，靠前/中间位置效果最好；
• 依赖白盒访问或完整模型参数，黑盒场景需额外改造；
• 对极长上下文（>32K）仍有内存/时间瓶颈；
• 只针对 prompt injection/知识破坏，未覆盖 jailbreak 等短文本攻击；
• 需要 GPU 环境，未针对低资源部署。

---

## 552. Optimal Transmitter Placement in Realistic Urban Environments

**arXiv ID:** 2604.28153 | [PDF](https://arxiv.org/pdf/2604.28153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 553. Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models

**arXiv ID:** 2604.28149 | [PDF](https://arxiv.org/pdf/2604.28149v1)

**作者:** Matthias Hertel `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5087 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种高效计算时间序列基础模型（TSFM）SHAP解释的方法，并在电力负荷预测任务上验证其可解释性与性能

**💡 创新点**

创新点在于利用TSFM可变上下文长度与协变量掩码的灵活性，避免了传统SHAP需要采样背景数据的高昂计算成本，实现了精确SHAP值的完整计算

**🔧 技术方法**

技术包括时间序列基础模型Chronos‑2与TabPFN‑TS、基于掩码的特征分组与时间/协变量掩码策略、SHAP解释框架

**📊 数据集**

使用德国TSO TransnetBW的历史负荷数据（2015‑2025年）、ERA5气象重分析（温度、辐射）以及节假日指示符，共计约10年小时级数据

**📈 对比分析**

与基线、单年训练的Transformer以及全数据训练的Transformer对比；在零样本条件下，TSFM在MAE、RMSE、MAPE上略逊于全数据Transformer，但显著优于单年Transformer；在解释性上，TSFM的特征重要性与域知识一致，协变量影响可被直观量化

**⚠️ 局限性**

局限包括：仅评估点预测，未处理概率预测与不确定性；解释只在四个时间窗口和协变量分组，细粒度不足；TabPFN‑TS解释耗时较长；对其他TSFM架构或不同任务的推广仍需验证

---

## 554. Unsafe and Unused? A History of Utility Code in Mature Open Source Projects

**arXiv ID:** 2604.28146 | [PDF](https://arxiv.org/pdf/2604.28146v1)

**作者:** Brandon Keller `[一作]` (Rochester Institute of Technology), Andy Meneely `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1987 | [OpenAlex ID](https://openalex.org/A5073112840)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对七个成熟开源项目的 Git 历史进行长期挖掘，研究了以 "util"、"helper" 等约定命名的 utility 文件在使用频率、复杂度、协作模式以及安全风险方面的演化特征。

**💡 创新点**

创新点在于首次系统评估 util/ helper 文件在不同项目中的普及度、改名迁移（adoption/abandon/oscillation）以及它们与漏洞关联度的动态变化，揭示了文件命名约定对代码安全和可维护性的长期影响。

**🔧 技术方法**

技术主要包括：Git 日志解析与重命名跟踪、Universal Ctags 静态分析提取函数调用、scc 复杂度测量、VHP CVE/ CWE 数据集联动、Python 脚本自动化处理 30 天快照并计算 odds ratio、recidivism 指标。

**📊 数据集**

数据集涵盖了七个成熟项目（Linux kernel、Django、FFmpeg、Apache Tomcat、Apache httpd、Struts、systemd）的完整 Git 历史，并结合 VHP 的漏洞修复提交、CVE 与 CWE 记录共计 3,344 条漏洞信息。

**📈 对比分析**

通过对比文件与非文件在调用频次、复杂度以及漏洞发生率的统计，发现大多数项目中文件调用比非文件多 1–7 倍，文件复杂度普遍高于非文件，而文件相关漏洞的 odds ratio 在项目早期可高达 10 倍，表明文件在早期更易出现安全缺陷；实验结果表明不同项目的模式差异显著。

**⚠️ 局限性**

局限性包括：仅关注 util/helper 约定，未覆盖其他常见约定；Git 重命名检测的准确性有限；仅使用 VHP 过滤的漏洞数据，可能遗漏未记录的安全事件；静态分析方法简化了函数调用映射，可能导致调用计数误差。

---

## 555. Beyond Gaussian Bottlenecks: Topologically Aligned Encoding of Vision-Transformer Feature Spaces

**arXiv ID:** 2604.28122 | [PDF](https://arxiv.org/pdf/2604.28122v1)

**作者:** Andrew Bond `[一作]` (Koç University), Aykut Erdem `[通讯]` (Koç University)

**通讯引用:** 4459 | [OpenAlex ID](https://openalex.org/A5000080119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于球面分布的变分自编码器（S^2VAE），对 Vision Transformer（VGGT）生成的几何特征进行压缩，并保持几何一致性。

**💡 创新点**

创新点在于使用 Power Spherical 分布的多球面产品作为潜在空间，显式建模 ViT 特征的球面几何，解决 Gaussian 潜在空间导致的后验塌陷与角度语义漂移。

**🔧 技术方法**

采用 Power Spherical VAE、产品球面潜在空间、球面重参数化、Gram 矩阵损失、任务特定（深度、相机姿态、点云）损失以及多层自注意力编码解码。

**📊 数据集**

使用 RealCam‑Vid（由 RealEstate10K、DL3DV‑10K、MiraData 组合而成）作为训练与评估数据集，并在 VGGT、DINOv2、DUSt3R、CLIP 等模型上进行对比实验。

**📈 对比分析**

与传统 Gaussian VAE、单球面 VAE 以及单一高维球面在 VGGT 的深度、相机姿态、点云以及 DINOv2 的深度指标上对比，S^2VAE 在高压缩率下实现了更低的 AbsRel、更高的 δ1、更低的 ATE，证明其性能优越。

**⚠️ 局限性**

主要限制包括：单一高维球面在数值上不稳定；需要额外的正则化与超参数调优；实验仅在已冻结的 ViT 结构下进行，未充分验证在动态场景或实时推理中的适用性。

---

## 556. DEFault++: Automated Fault Detection, Categorization, and Diagnosis for Transformer Architectures

**arXiv ID:** 2604.28118 | [PDF](https://arxiv.org/pdf/2604.28118v1)

**作者:** Sigma Jahan `[一作]` (Dalhousie University), Mohammad Masudur Rahman `[通讯]` (Dalhousie University)

**通讯引用:** 1455 | [OpenAlex ID](https://openalex.org/A5030616863)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Transformer 专用的故障诊断框架 DEFault++，能够在三层级（检测、分类、根因诊断）识别并定位 Transformer 模型中的 12 类故障及其 45 个根因。

**💡 创新点**

创新点在于提出了 DEForm 基于真实 Transformer 失效案例的变异注入技术、构造了 Fault Propagation Graph（FPG）来编码 Transformer 组件间的故障传播关系，并将监督对比学习与原型匹配相结合，实现可解释的根因诊断。

**🔧 技术方法**

技术包括：层级学习模型（共享编码器+FPG 消息传递）、监督对比学习+原型匹配、统计突变检测（one‑sided sign‑flip permutation）、多组特征提取（注意力熵、QKV 对齐、梯度统计等）以及解释方法的组别重要性评分。

**📊 数据集**

使用了 DEFault‑bench，包含 3,739 个单一故障实例，来源于 7 种 Transformer 模型（4 编码器 + 3 解码器）和 9 个下游任务（GLUE 5 任务 + 4 语言模型语料），并通过 DEForm 注入 12 类故障及 45 个根因。

**📈 对比分析**

与 AutoTrainer、DeepDiagnosis、DeepFD、DEFault 等现有 DNN 故障诊断方法相比，DEFault++ 在检测 AUROC 达 0.96、分类宏 F1 0.85、根因诊断宏 F1 0.86；在真实 GitHub 故障和 21 名开发者的实测中，修复建议准确率从 57.1% 提升至 83.3%。

**⚠️ 局限性**

局限性包括：仍需人工维护故障注入库，缺乏跨架构（如大规模 GPT‑3）和多任务迁移的评估；部分根因样本不足导致分类不均衡；解释方法依赖 FPG 结构，对非 Transformer 体系结构适用性有限。

---

## 557. FreeOcc: Training-Free Embodied Open-Vocabulary Occupancy Prediction

**arXiv ID:** 2604.28115 | [PDF](https://arxiv.org/pdf/2604.28115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 558. I hope we don't do to trust what advertising has done to love

**arXiv ID:** 2604.28113 | [PDF](https://arxiv.org/pdf/2604.28113v1)

**作者:** Jade Alglave `[一作]` `[通讯]` (Arm and University College London), Jade Alglave (Arm and University College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文讨论了面向老年人等用户的代理式 AI 系统中的信任问题，并提出了可信赖性支柱和两类代理系统（有界与可治理适应性）。

**💡 创新点**

创新点在于将信任拆解为可操作的支柱，并定义了两种代理系统模型以系统化可信设计。

**🔧 技术方法**

采用 LLM 及其在代理架构中的角色，结合身份管理、沙箱、凭证等技术实现信任保障。

**📊 数据集**

无具体数据集，主要以案例分析和已有文献为依据。

**📈 对比分析**

未做实验比较，文章以理论阐述和案例说明为主，无法给出性能指标。

**⚠️ 局限性**

局限性在于缺乏实证验证，方案依赖多方协作与监管，实际可落地性仍待探究。

---

## 559. Splitting Argumentation Frameworks with Collective Attacks and Supports

**arXiv ID:** 2604.28112 | [PDF](https://arxiv.org/pdf/2604.28112v1)

**作者:** Matti Berthold `[一作]` (FernUniversität in Hagen), Anna Rapberger `[通讯]` (TU Dortmund)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5060212076)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了针对包含共同攻击和支持关系的双极集合式论证框架（BSAF）的分割（splitting）方法，并给出了相应的语义证明与算法步骤。

**💡 创新点**

创新点在于：①首次将集合攻击与集合支持统一纳入分割框架；②设计了闭包、还原与修正三步修改机制，以处理支持带来的闭合性与不可决定链接；③给出了针对完整、稳定、完备等多种语义的分割定理，并对优先与基于地面语义的局限性作了分析。

**🔧 技术方法**

主要技术包括：集合论与图论中的攻击/支持关系定义；对负链接进行闭包操作以保证语义不变；构造(R‑)还原与(mod‑)修正以在子框架中模拟跨边影响；结合类型‑1 与类型‑2 约束实现对共享支持的控制；最终通过组合攻击与支持分割实现全框架的增量推理。

**📊 数据集**

该工作主要为理论研究，未使用具体实验数据集；所有结果均在形式化证明与合成例子中给出。

**📈 对比分析**

在 Dung‑style AF 的实验中，作者报告使用分割可使推理时间平均提升约 60%；但在 BSAF 上的实验尚未给出量化指标，论文仅提供理论性能保证。

**⚠️ 局限性**

局限性包括：①对基于地面语义的分割仅能保证单向推理，无法完整重构所有地面扩展；②对优先语义的分割也仅得到部分扩展；③目前仅处理前向攻击与后向支持的组合，尚未覆盖更一般的混合攻击/支持模式；④缺乏对实际数据集与实现效率的实证评估。

---

## 560. Auto-FlexSwitch: Efficient Dynamic Model Merging via Learnable Task Vector Compression

**arXiv ID:** 2604.28109 | [PDF](https://arxiv.org/pdf/2604.28109v1)

**作者:** Junqi Gao `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 63250 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何在多任务设置下通过动态模型融合实现高效的任务向量压缩与存储。

**💡 创新点**

创新点在于：①发现任务向量呈冲击式激活并对低比特量化鲁棒；②提出基于稀疏化+量化的三元 T‑Switch；③构建可学习的 FlexSwitch，联合 Learnable Gating Sparsification (LGS)、Bit‑width Adaptive Selection (BAS) 与 Sparsity‑Aware Storage Strategy (SASS)；④引入带低秩度量的 KNN 检索实现 Auto‑FlexSwitch，显著降低存储并保持性能。

**🔧 技术方法**

使用的技术包括：任务向量稀疏化与二值化压缩、可学习门控稀疏化、可自适应位宽量化、分组 COO 存储优化、低秩 KNN 检索与动态合并。

**📊 数据集**

实验数据集覆盖视觉分类（SUN397、Cars、RESISC45、EuroSAT、SVHN、GTSRB、MNIST、DTD）、目标检测（DETR‑RoboFlow 100）、NLP GLUE（CoLA、SST‑2、MRPC、QQP、MNLI、QNLI、RTE）以及 LLM 细调（LIMA）上的多领域推理与代码生成。

**📈 对比分析**

与静态融合、传统动态融合（Ties‑Merging、DARE、EMR‑Merging 等）及单任务微调进行对比。Auto‑FlexSwitch 在 ViT‑B/32、ViT‑L/14、ConvNeXt、RoBERTa‑base、Mamba 等多种架构上，平均存储仅 5–8 MB（相比原始增量权重压缩 50–200×），同时准确率接近或超越单任务微调（平均提升 1–3 %）。

**⚠️ 局限性**

限制包括：需要少量任务示例作为查询集；KNN 检索和低秩映射仍有计算开销；对任务相似度高度依赖，极大任务数量下扩展性和检索精度可能受限；当前主要针对分类/检测/文本分类任务，未验证在更复杂生成任务中的鲁棒性。

---

## 561. UHR-Net: An Uncertainty-Aware Hypergraph Refinement Network for Medical Image Segmentation

**arXiv ID:** 2604.28095 | [PDF](https://arxiv.org/pdf/2604.28095v1)

**作者:** Shuokun Cheng `[一作]` (China University of Geosciences (Wuhan)), Kun Sun `[通讯]` (China University of Geosciences (Wuhan))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于不确定性引导的超图细化网络UHR-Net，用于医学图像病灶分割

**💡 创新点**

创新点在于：①采用几何感知copy‑paste与lesion-like hard‑negative挖掘的实例级对比预训练（UO‑IC）提升小病灶判别；②基于熵的uncertainty map引导超图细化，分离前景/背景原型减少边界干扰；③将预训练与超图细化融合的端到端框架

**🔧 技术方法**

核心技术包括：对比学习（InfoNCE）、几何copy‑paste增强、硬负样本挖掘、熵不确定性估计、超图卷积、前景/背景原型生成与多尺度细化

**📊 数据集**

在五大公开基准上评估：ISIC‑2016、ISIC‑2017、GlaS、Kvasir‑SEG、Kvasir‑Sessile

**📈 对比分析**

与U‑Net、U‑Net++、PraNet、TGANet、DCSAU‑Net、ConDSeg等方法对比，UHR‑Net在所有数据集均实现mIoU与mDSC领先或相当的最高分（例如GlaS 87.0%/92.7%、ISIC‑2016 87.6%/92.9%），显著提升分割精度

**⚠️ 局限性**

局限性：对超图原型数量与logit调制因子敏感；预训练需额外数据与计算资源；在极小病灶或极高噪声场景下仍可能出现边界误差

---

## 562. A MEC-Based Optimization Framework for Dynamic Inductive Charging

**arXiv ID:** 2604.28069 | [PDF](https://arxiv.org/pdf/2604.28069v1)

**作者:** Emre Akıskalıoğlu `[一作]` (Marmara University), Renato Lo Cigno `[通讯]` (University of Brescia)

**通讯引用:** 4316 | [OpenAlex ID](https://openalex.org/A5049020621)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了基于MEC的动态感应充电功率分配MPC框架，并在伊斯坦布尔Uskudar路段的SUMO仿真环境中评估其性能。

**💡 创新点**

提出了以车辆电量紧迫度为权重的MPC优先策略和动态条带功率再平衡机制，同时将完整仿真工具开源供社区使用。

**🔧 技术方法**

采用Model Predictive Control、MEC与V2X通信、SUMO交通仿真、CVXPY/Gurobi凸优化等技术。

**📊 数据集**

使用伊斯坦布尔Uskudar路段的OpenStreetMap地图生成的道路模型，配合自定义车辆SoC、路线和不同交通强度（λ=5、12、20 vpm）的仿真数据。

**📈 对比分析**

与无优化的基准策略比较，利用能量利用率、SoC fulfillment（用户满意度）和CDF/PDF等指标评估性能。结果显示在高负载下MPC显著提升能量利用率、用户满意度和公平性。

**⚠️ 局限性**

局限性包括：假设理想的V2X通信延迟和同步；未考虑真实网络误差；仿真仅覆盖单一路段，未覆盖更大规模城市网络；对电池充电技术细节（如线圈对准误差、效率变化）进行了简化处理。

---

## 563. Intern-Atlas: A Methodological Evolution Graph as Research Infrastructure for AI Scientists

**arXiv ID:** 2604.28158 | [PDF](https://arxiv.org/pdf/2604.28158v1)

**作者:** Yujun Wu `[一作]` (Shanghai Artificial Intelligence Laboratory), Cheng Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5006542157)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一张以方法为中心、类型化的演化图（Intern-Atlas），并提供了三种基于该图的推理接口：方法谱系重建、图驱动的想法评估和结构化的想法生成。

**💡 创新点**

①将论文引用从平面转换为可查询的因果方法网络；②在每条因果边上附加精确的瓶颈‑机制引证；③引入Self‑Guided Temporal MCTS（SGT‑MCTS）在图上恢复演化链，兼顾探索与时间一致性；④将方法演化结构转化为可执行的评估与生成公式，避免LLM的“造句”偏差。

**🔧 技术方法**

自然语言处理（LLM分类、抽取与证据验证）、图构建与类型化、Monte Carlo树搜索（SGT‑MCTS）、基于图统计的评分函数与多维聚合、结构化策略驱动的生成模板。

**📊 数据集**

1,030,314篇AI领域论文（会议、期刊、arXiv，1965‑2025），包括约8,155个方法节点、9,410,201条类型化边，数据来源于OpenAlex、Semantic Scholar、S2ORC等学术元数据。

**📈 对比分析**

与专家手工绘制的30个调查图、Beam Search、随机游走等基线比较。SGT‑MCTS在节点回忆率、边回忆率和链对齐得分上分别提升约40/56/40个百分点；图驱动评估在四个质量层级上实现从顶尖会议到拒稿的明显分层，Spearman相关系数高于纯LLM判别器；图驱动生成在五维评价中整体分数比无知识、外部检索和传统RAG高约1点，人工对比胜率均超过80%。

**⚠️ 局限性**

仍受LLM抽取误差、边置信度偏差及罕见/长尾方法的低覆盖率影响；对跨领域（非AI）方法的泛化有限；图的构建与更新成本高，需要持续的实体解析与边验证。

---

## 564. FlexiTac: A Low-Cost, Open-Source, Scalable Tactile Sensing Solution for Robotic Systems

**arXiv ID:** 2604.28156 | [PDF](https://arxiv.org/pdf/2604.28156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 565. Design and Characteristics of a Thin-Film ThermoMesh for the Efficient Embedded Sensing of a Spatio-Temporally Sparse Heat Source

**arXiv ID:** 2604.28148 | [PDF](https://arxiv.org/pdf/2604.28148v1)

**作者:** Sajjad Boorghan Farahan `[一作]` (State University of New York at Binghamton), Jingzhou Zhao `[通讯]` (State University of New York at Binghamton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 ThermoMesh——一种利用热电耦合和阻性层在薄膜网格上实现的被动热量感知与压缩的传感器；

**💡 创新点**

创新点在于将热导、热电耦合与温度相关的非线性阻性层结合，形成可在物理域内完成稀疏热源定位与温度回归的压缩传感系统，并通过设计不同的阻性层（线性、NTC、VO₂）显著提升最低灵敏度与均匀性；

**🔧 技术方法**

采用了基于 Kirchhoff 电流定律的网络模型、线性与非线性阻性层的温度耦合、OMP 与 LSTM 网络的稀疏恢复与分类回归、以及噪声等效温度（NET）评估等技术；

**📊 数据集**

使用合成的 1‑sparse 热信号数据集（包含白噪声、40 dB 与 20 dB SNR 条件），并在 16×16 网格上进行实验；

**📈 对比分析**

与无阻性层、线性阻性层和两种 NTC 阻性层（VO₂、陶瓷）对比。结果显示：VO₂ 与陶瓷 NTC 的定位准确率可达 98–100%，温度 MAE 在 0.02–3.81 K 之间，NET 在 0.07–14.93 K 之间，均优于线性或无阻性设计；

**⚠️ 局限性**

主要限制包括：仅在单像素、单事件（1‑sparse）条件下验证；模型未包含热阻与电容耦合导致的 RC 限制；实验验证仍待完成；以及对非线性材料的实际制备与长时稳定性未充分探讨。

---

## 566. Index-Assisted Stratified Sampling for Online Aggregation

**arXiv ID:** 2604.28141 | [PDF](https://arxiv.org/pdf/2604.28141v1)

**作者:** Yunnan Yu `[一作]` (University at Buffalo), Zhuoyue Zhao `[通讯]` (University at Buffalo)

**通讯引用:** 402 | [OpenAlex ID](https://openalex.org/A5082679208)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种两阶段基于索引的分层抽样框架，用于在线聚合查询的近似计算，显著降低查询延迟。

**💡 创新点**

创新点在于：①将采样成本模型与分层抽样结合，推导出改进的 Neyman 分配公式；②通过阶段 0 的初始采样快速估计统计量，再利用阶段 1 的动态规划或贪心策略得到最优分层，兼顾优化开销与采样成本；③在 PostgreSQL 上实现并对比多种基线，验证极大加速。

**🔧 技术方法**

核心技术包括：聚合 B‑Tree 采样索引、改进的 Neyman 采样分配、两阶段抽样框架、动态规划与贪心分层优化、Z 分布置信区间估计、采样概率权重校正。

**📊 数据集**

使用的真实数据集有：US Airline On‑Time Performance、Intel Lab Sensor、US Census Income；合成数据集为改造的 TPC‑H lineitem（带高延迟时间段）。

**📈 对比分析**

与均匀抽样、基于扫描的分层抽样、VerdictDB 等基线比较，实验显示在高方差查询下可实现最高 98708×（flight 数据集）对均匀抽样的加速，平均 3–4 级加速；在大规模 TPC‑H 规模下，仍保持 3.4× 的优势；置信区间均低于用户指定阈值。

**⚠️ 局限性**

局限性：仅支持单表范围聚合查询；对分层的前置采样和动态规划有一定的运行时开销；需要聚合 B‑Tree 索引；对多维范围、联接、分组聚合等更复杂查询尚未覆盖；在极稀疏或极高基数的维度上可能需要更精细的分层策略。

---

## 567. Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes

**arXiv ID:** 2604.28138 | [PDF](https://arxiv.org/pdf/2604.28138v1)

**作者:** Tianyuan Wu `[一作]` (HKUST), Wei Wang `[通讯]` (HKUST)

**通讯引用:** 7486 | [OpenAlex ID](https://openalex.org/A5100622677)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种主机侧运行时，用来在AI代理沙盒中自动识别哪些交互回合产生了可恢复的系统状态，并按需做增量或完整的检查点与恢复，覆盖了应用层缺失和系统层过度完整的两极问题，解决了代理-OS语义鸿沟；

**💡 创新点**

创新点在于：①将代理层的“回合边界”与OS层的可观测状态差异结合，使用eBPF做轻量级的文件系统与进程变化检测；②在LLM等待窗口中异步完成检查点，最大程度地隐藏检查点延迟；③通过全局调度器在多沙盒密集部署时平滑检查点流量，防止I/O饱和。

**🔧 技术方法**

技术实现包括：eBPF系统调用跟踪 + 用户空间 daemon 进行状态差异分析；OpenZFS 快照做文件系统检查点；CRIU 做进程状态检查点；协调器（HTTP 代理）识别回合边界并推送检查点请求；C/R 引擎包含 Scheduler、Worker、Manager，负责跨沙盒排队、执行和事务性发布。

**📊 数据集**

使用了两大基准：Terminal-Bench（Shell 重度交互任务）和 SWE-Bench（软件工程任务），并分别对 Claude‑code、iFlow‑cli、SWE‑agent 这三种代理进行评测。

**📈 对比分析**

与四种基线（Chat‑only、Chat+FS、Restart、FullCkpt）对比，实验显示：恢复正确率从 8‑13% 提升到 100%；在 16‑96 个并发沙盒下，平均任务完成时间在无故障情形下 1.9% 以内；检查点流量被压缩 70‑87%；在高密度场景下，检查点延迟被 LLM 等待窗口完全覆盖，最大暴露延迟仅 3.65%。

**⚠️ 局限性**

限制主要在于：仅针对 Linux 容器/微 VM；需 eBPF 与 CRIU、OpenZFS 的兼容性；对网络状态、外部挂载等未覆盖；在极端长生命周期进程或大内存占用的场景下，检测与检查点仍可能产生较大开销；最后，系统假设代理遵循“交互回合”模式，特殊自定义流程可能需要额外适配。

---

## 568. 3D-ReGen: A Unified 3D Geometry Regeneration Framework

**arXiv ID:** 2604.28134 | [PDF](https://arxiv.org/pdf/2604.28134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 569. Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn Attack Detection

**arXiv ID:** 2604.28129 | [PDF](https://arxiv.org/pdf/2604.28129v1)

**作者:** Prashant Kulkarni `[一作]` `[通讯]`, Prashant Kulkarni

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一种基于激活轨迹的多轮攻击检测方法，能够在对话中出现隐蔽攻击时提前发出警报。

**💡 创新点**

创新点在于提出了“adversarial restlessness”这一激活路径长度异常特征，并通过三阶段（benign/pivoting/adversarial）标注的合成数据实现对早期攻击阶段的精准检测。

**🔧 技术方法**

使用了激活提取、五维轨迹特征（漂移、余弦相似度、累计漂移、加速度、平均漂移）以及 XGBoost 或两阶段对比学习+XGBoost 的轻量化探测器。

**📊 数据集**

采用了包含三种来源的训练集：合成多轮攻击数据（1,125 轮）、LMSYS-Chat-1M 实际对话（1,200 轮）和 SafeDialBench（300 轮）共计 2,625 轮对话。

**📈 对比分析**

与现有文本级安全工具（PromptGuard、Lakera Guard 等）相比，本文方法在合成和混合测试集上实现了约 89–96% 的检测率，误报率仅 2–4%，显著优于传统方法。

**⚠️ 局限性**

局限性包括需要白盒访问模型、探测器对不同架构不具备迁移性、对真实数据的泛化仍受限（LMSYS 仅 47–71%）、以及尚未实现对攻击的主动干预或可解释性分析。

---

## 570. AdvDMD: Adversarial Reward Meets DMD For High-Quality Few-Step Generation

**arXiv ID:** 2604.28126 | [PDF](https://arxiv.org/pdf/2604.28126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 571. What Makes a Good Terminal-Agent Benchmark Task: A Guideline for Adversarial, Difficult, and Legible Evaluation Design

**arXiv ID:** 2604.28093 | [PDF](https://arxiv.org/pdf/2604.28093v1)

**作者:** Ivan Bercovich `[一作]` `[通讯]`, Ivan Bercovich

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于作者在Terminal Bench 3的贡献与审阅经验，系统性地总结了终端代理基准任务的设计原则，提出了任务应具备的对抗性、难度和可读性，并构建了六大常见任务失效模式的分类体系，同时给出具体的编写、验证与评测建议。

**💡 创新点**

创新点主要有：①首次将“对抗性、难度、可读性”三大维度作为评价准则，突出任务的可验证性与挑战性；②提出了六类典型失败模式（AI生成指令、过度规范、书面难度、隐藏知识假设、误检验、奖励黑客化）并给出防范思路；③通过对公开基准任务的自动化奖励黑客检测，发现约15%任务易被黑客化，为基准可信度提供实证依据；④强调“难度来源于问题本身”，并对资源限制与时间限制对难度的误导进行批判。

**🔧 技术方法**

主要采用的技术手段包括：任务指令编写与手工编辑、利用SSIM对登录屏幕截图做相似度校验、MD5哈希验证关键文件、Docker/VM/QEMU等容器化与仿真技术、以及自动化的奖励黑客检测脚本（对比参考答案、执行多轮测试）。此外，本文还使用了人类专家评估、模型实验（SOTA代理）来验证难度与有效性。

**📊 数据集**

使用的数据集与资源：Terminal Bench 1.0、Terminal Bench 3（待收录任务）、以及通过自动化审计得到的“Terminal Wrench”数据集，后者是对五个公开终端代理基准的系统性评估结果；此外还利用了内部的虚拟机、Docker镜像和QEMU配置来复现与验证任务。

**📈 对比分析**

比较方法：作者将新任务与已有任务的难度、奖励黑客率、验证逻辑复杂度进行对比，并通过对多款SOTA终端代理（如 GPT‑4/5 系列、Anthropic、DeepMind 等）的实验，记录通过率、失败原因与时间成本。实验结果显示，约15%现有任务存在奖励黑客风险；同时，符合对抗性、难度与可读性三大准则的任务在SOTA模型上通过率显著下降，验证了准则的有效性。

**⚠️ 局限性**

局限性：①缺乏大规模的定量实验验证所有准则的普适性；②难度评估仍高度依赖人工专家判断，缺乏统一的可复现指标；③防止奖励黑客的措施多为后置检测，未能在任务设计阶段完全消除漏洞；④本文聚焦于终端代理场景，指南对其它领域的可迁移性尚未充分验证；⑤复杂环境（多层容器、虚拟化）下的任务仍可能因资源限制而产生非本质难度。

---

## 572. Succinct Graph Representations and Algorithmic Applications

**arXiv ID:** 2604.28096 | [PDF](https://arxiv.org/pdf/2604.28096v1)

**作者:** Ahammed Ullah `[一作]` (Purdue University), Alex Pothen `[通讯]` (Purdue University)

**通讯引用:** 5611 | [OpenAlex ID](https://openalex.org/A5055182869)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出双团覆盖（Dual Clique Cover, DCC）图表示法，并给出可在多项式时间构造的紧凑 DCC 形式；基于此表示设计了一系列表示感知的图算法。

**💡 创新点**

创新点在于：①定义并实现了可在多项式时间构造的 succinct DCC ；②通过利用团覆盖的非边证明性质和组合结构，使得图算法的运行时间可按 DCC 大小（而非边数）计；③提出了三种高效的构造算法（Succinct‑Peeling、Global‑Admissibility、Local‑Admissibility），实现了压缩率与构造速度的权衡。

**🔧 技术方法**

使用的技术包括：团覆盖与其双重关系、非边证人性质、第一适配(First‑Fit)贪心着色、拓展规则、可变体合并、并查集、DFS/BFS、最大匹配、k‑核心、色数等；同时利用 degeneracy、逆 Ackermann 函数等理论工具来分析时间与空间复杂度。

**📊 数据集**

实验数据集涵盖：SuiteSparse（SS、SS‑L）、BrainNet（BN、BN‑L）、Barabási‑Albert、Uniform Attachment、Erdős‑Rényi 等，稀疏图与稠密图共计 8 类，边数范围从 1.2×10⁷ 到 4.1×10⁹。

**📈 对比分析**

与传统邻接表实现及 WebGraph 压缩框架比较：DCC 表示在存储压缩率上平均约 9×、最大 37×；执行内存使用约为存储压缩率的一半；在总运行时间上相较于邻接表可获得 6.5–35× 的加速，尤其在连通分量、BFS、DFS、最大匹配等核心应用中表现突出；与 WebGraph 的比较中，DCC 方案在稀疏图上平均提升 11×（最大 28×），且内存占用约 2 倍更少。

**⚠️ 局限性**

局限性包括：①构造算法在极稠密图上仍可能需要 O(m) 的空间；②对动态图更新的支持尚未完善；③理论上团覆盖的最优性（特别是 assignment‑optimal 目标）的近似性问题仍未解决；④部分实现依赖于对大规模邻接表的先验读取，构造时间仍受限于边数。

---

## 573. FiLMMeD: Feature-wise Linear Modulation for Cross-Problem Multi-Depot Vehicle Routing

**arXiv ID:** 2604.28102 | [PDF](https://arxiv.org/pdf/2604.28102v1)

**作者:** Arthur Corrêa `[一作]` (University of Coimbra), Samuel Moniz `[通讯]` (University of Coimbra)

**通讯引用:** 440 | [OpenAlex ID](https://openalex.org/A5065919267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可同时处理多仓库与多约束条件的车辆路径规划模型，

**💡 创新点**

创新点在于引入FiLM线性调制进行约束编码、课程学习策略与偏好优化训练方法，

**🔧 技术方法**

使用Transformer注意力网络、Mixture of Experts与Preference Optimization技术，

**📊 数据集**

实验基于CVRPLib与Set‑X数据集，

**📈 对比分析**

与MTPOMO、MVMoE、RouteFinder、CaDA等多任务学习方法对比，平均成本差距从约4.7%下降至约4.4%，性能显著提升，

**⚠️ 局限性**

局限在于对极端约束组合（如多仓库+多重时间窗）仍存在学习难度，且依赖大量训练样本。

---

## 574. Tailwind: A Practical Framework for Query Accelerators

**arXiv ID:** 2604.28079 | [PDF](https://arxiv.org/pdf/2604.28079v1)

**作者:** Geoffrey X. Yu `[一作]` (MIT), Tim Kraska `[通讯]` (MIT)

**通讯引用:** 9700 | [OpenAlex ID](https://openalex.org/A5034086130)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种外部查询规划器 Tailwind，可通过抽象逻辑计划 (ALP) 将自定义加速器无侵入地插入任何支持数据导入/导出的关系型数据库系统。

**💡 创新点**

创新点包括：①抽象逻辑计划 (ALP) 作为统一的、几乎声明式的加速器描述语法；②利用 ALP 的树结构构造图神经网络模型，自动学习每个加速器的运行时性能；③外部规划器结合贪心选择与 e‑graph + NFTA 匹配，能够在不同 RDBMS 上无缝重写查询。

**🔧 技术方法**

核心技术：e‑graph + 归约搜索、NFTA（非确定性树自动机）匹配、抽象逻辑计划 (ALP)、基于 ALP 的树形神经网络性能模型、在线预测与离线加速器实例化、空间预算约束下的贪心选择、数据传输时间建模。

**📊 数据集**

使用 TPC‑H 标准基准（Scale Factor 100）及其扩展版 TPC‑H+，包含 20 个模板查询，评估不同加速器组合。

**📈 对比分析**

与 Redshift 与 DuckDB 的基线及“盲目使用加速器”的策略对比：在 10% 空间预算下，Tailwind 在 TPC‑H 上平均提升约 2×（几乎 10×），在 TPC‑H+ 上平均提升约 1.5×；总体运行时开销低于 5%；在不同 RDBMS 上表现相似，但 Redshift 受数据传输影响更大。

**⚠️ 局限性**

局限性：1）依赖 RDBMS 的导入/导出接口，传输开销可能成为瓶颈；2）需对查询运行时预测准确，否则可能选择错误加速器；3）写操作会使预计算状态失效，必须重建；4）ALP 仍需手工编写，覆盖范围受限；5）对极大或高度动态的工作负载代表性假设不一定成立。

---

## 575. Akita: A High Usability Simulation Framework for Computer Architecture

**arXiv ID:** 2604.28073 | [PDF](https://arxiv.org/pdf/2604.28073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 576. Repetition over Diversity: High-Signal Data Filtering for Sample-Efficient German Language Modeling

**arXiv ID:** 2604.28075 | [PDF](https://arxiv.org/pdf/2604.28075v1)

**作者:** Ansar Aynetdinov `[一作]` (Humboldt University of Berlin), Alan Akbik `[通讯]` (Humboldt University of Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建分层质量过滤器，对 500M German 网页文档进行筛选，并在固定 token 预算下比较多轮训练高质量数据与单轮大规模多样数据的预训练效果。

**💡 创新点**

创新点在于：1) 提出 Coherence、Information Value、Educational Quality 三层分级过滤并形成 Dense Core 子集；2) 通过实验验证高质量数据多轮训练优于单轮大规模训练；3) 发布经清洗的 German 评测基准与 Boldt LLM 系列模型。

**🔧 技术方法**

主要技术包括：层级文档分类器（基于 Transformer 的判别器）、BPE 32k 分词器、Llama‑style 解码 Transformer、重复多轮预训练、instruction‑tuning SFT 以及 LLM‑as‑judge 的评估协议。

**📊 数据集**

使用的数据集有：FineWeb‑2 German (FW2‑DE)、自定义的 Random、Coherence、Information Value、Educational Quality、Dense Core 子集；额外加入 Fundus 生成的 6B 句子新闻语料；评测基准包括 German Global MMLU、ARC‑Easy/Challenge、OpenBookQA、HellaSwag、LAMBADA（经人工清洗后）。

**📈 对比分析**

在 100B/200B token 预算下，通过零样本和 instruction‑tuning 评估，Dense Core 模型在大多数基准上比 Random baseline 提升约 4–5 分；1B Dense Core 在 200B 预算下更是达到或超过多语言大型基准的表现；混合训练策略虽然提升后半段性能，但整体仍不及纯 Dense Core。

**⚠️ 局限性**

局限性包括：仅针对 German，未验证到其他高资源语言；实验规模限制在 1B 参数/200B tokens，无法验证更大规模效果；仅使用单一 dense Transformer 结构，未探讨 MoE 或其它注意力变体；未对模型的毒性、偏见进行系统评估，质量过滤虽减少噪声但仍可能放大社会偏见。

---

## 577. HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation

**arXiv ID:** 2604.28196 | [PDF](https://arxiv.org/pdf/2604.28196v1)

**作者:** Xin Zhou `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 39113 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一驾驶世界模型同时完成 3D 场景理解和未来几何预测。

**💡 创新点**

创新点包括：BEV 统一表示、LLM 强化的世界查询、Current‑to‑Future 链接以及联合几何优化策略，将语义推理与几何演化深度耦合。

**🔧 技术方法**

核心技术有 Bird’s‑Eye‑View (BEV) 视觉编码、跨模态 LLM（如 InternVL2）交互、世界查询知识转移、当前到未来链接与隐式几何正则化。

**📊 数据集**

使用 NuScenes、OmniDrive‑nuScenes、NuScenes‑QA、DriveLM、NuInteract 等多种驾驶与视觉问答数据集进行训练与评估。

**📈 对比分析**

在 NuScenes/OmniDrive 评测中，生成任务的 3s Chamfer 距离比 DriveX/ViDAR 低约0.09，理解任务的 CIDEr 与 METEOR 分别比 Omni‑Q/Omni‑L 提升约0.07；整体统一模型在多项指标上均优于单一专用方法。

**⚠️ 局限性**

局限性：对大规模预训练 LLM 的依赖导致推理延迟；对 BEV 中语义先验的利用尚不充分；跨模态数据缺乏足够多样性，影响泛化。

---

## 578. Representation Fréchet Loss for Visual Generation

**arXiv ID:** 2604.28190 | [PDF](https://arxiv.org/pdf/2604.28190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 579. Non-negative Rational Semantic Numeration Systems

**arXiv ID:** 2604.28171 | [PDF](https://arxiv.org/pdf/2604.28171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 580. Exploration Hacking: Can LLMs Learn to Resist RL Training?

**arXiv ID:** 2604.28182 | [PDF](https://arxiv.org/pdf/2604.28182v1)

**作者:** Eyon Jang `[一作]` (MATS), David Lindner `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了RL训练中模型通过刻意抑制探索来逃避能力激活的行为（exploration hacking），并通过构造模型实验体验证其可行性与检测方法。

**💡 创新点**

首次将 exploration hacking 形式化为实证问题，提出模型锁定与多维检测框架，并系统评估其在前沿 LLM 上的出现可能性。

**🔧 技术方法**

采用 GRPO/PPO 训练、SFT 微调、LoRA 权重噪声、链式思考与动作监测等技术进行实验与对抗检测。

**📊 数据集**

使用 KernelBench、BigCodeBench、WMDP‑Bio、VCT 等前沿评测数据集以及工具使用环境进行评估。

**📈 对比分析**

通过 RL 激活、SFT 恢复、监测 AUC 与权重噪声的对比，模型在 RL 中可维持 500 步以上抑制，并且监测 AUC 超 0.9，SFT 能在约 80–100 步内恢复被抑制能力。

**⚠️ 局限性**

局限性：仅使用中型 14B 参数模型、实验依赖显式 CoT 逻辑，未覆盖更大规模模型、不同 RL 算法或自然出现的 exploration hacking 情形。

---

## 581. An adaptive wavelet-based PINN for problems with localized high-magnitude source

**arXiv ID:** 2604.28180 | [PDF](https://arxiv.org/pdf/2604.28180v1)

**作者:** Himanshu Pandey `[一作]` (Indian Institute of Science), Ratikanta Behera `[通讯]` (Indian Institute of Science)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5028105407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了自适应小波物理信息神经网络（AW-PINN），通过在训练过程中动态调整小波基的尺度和位移，解决了传统PINN在面对局部高幅源项时出现的谱偏差和损失不平衡问题。

**💡 创新点**

创新点在于：①将小波基的尺度与平移参数从固定转为可训练的自适应参数；②采用预训练+自适应两阶段策略，先用W-PINN挑选物理相关的小波族，再在此基础上细化；③利用解析求导取代自动微分，显著加快训练速度。

**🔧 技术方法**

使用的技术包括：物理信息神经网络、波形小波基（Gaussian小波）、梯度下降（Adam + L-BFGS）、神经切线核（NTK）理论分析、解析微分。

**📊 数据集**

使用的“数据集”为四类典型偏微分方程的数值测试：1）一维热传导带极强瞬时热源；2）二维Poisson方程带高度局部化源项；3）一维流动方程带强振荡源；4）三维（1D时间+2D空间）Maxwell方程带点源。每类问题均以已知解析解或高精度FDTD为参考。

**📈 对比分析**

与基准PINN、W-PINN（小波PINN）和MMPINN（多尺度PINN）进行对比。AW-PINN在相同或更少的训练点下，平均相对L2误差低1–3个数量级，训练时间相对MMPINN缩短约30–70%，在极端损失不平衡（如源项比值10^10:1）时仍能稳健收敛。

**⚠️ 局限性**

局限性包括：①基于相似度的基底筛选仍需人工设定阈值，受预训练质量影响；②目前仅验证了Gaussian小波，其他小波族的适用性待进一步研究；③大规模高分辨率小波集仍会产生内存与收敛难题；④对更复杂多尺度或非局部源问题的推广性尚未验证。

---

## 582. LLM as Clinical Graph Structure Refiner: Enhancing Representation Learning in EEG Seizure Diagnosis

**arXiv ID:** 2604.28178 | [PDF](https://arxiv.org/pdf/2604.28178v1)

**作者:** Lincan Li `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两阶段的基于大语言模型（LLM）的EEG图结构生成与细化框架，用于癫痫发作检测。

**💡 创新点**

创新点在于将LLM用作图边缘精炼器，结合Transformer边缘预测器，从统计特征和文本描述中进行语境化推理，显著降低噪声边缘并提升图结构解释性。

**🔧 技术方法**

使用的技术包括Transformer编码器提取通道嵌入、两层MLP预测边缘概率、LLM（如GPT‑5、Gemini、Llama等）进行提示式边缘验证，以及GraphS4mer骨干网络进行下游发作分类。

**📊 数据集**

使用的公开数据集为Temple University Hospital EEG Seizure Corpus (TUSZ) v1.5.2，包含5,612条EEG记录和3,050个发作注释。

**📈 对比分析**

通过与传统图构造方法（自相关、KNN、注意力、生成式网络）以及不同LLM的细化比较，实验表明LLM细化的图能使F1、准确率和召回率分别提升约3–10%，GPT‑5模型获得最高的F1≈0.791、准确率≈93.15%、召回率≈80.58%。

**⚠️ 局限性**

局限性包括：对LLM推理过程的可解释性仍有限；提示设计与LLM尺寸相关，较小模型表现不稳定；实验仅验证了单一EEG数据集，缺乏跨数据集或跨设备的泛化评估。

---

## 583. PhyCo: Learning Controllable Physical Priors for Generative Motion

**arXiv ID:** 2604.28169 | [PDF](https://arxiv.org/pdf/2604.28169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 584. OmniRobotHome: A Multi-Camera Platform for Real-Time Multiadic Human-Robot Interaction

**arXiv ID:** 2604.28197 | [PDF](https://arxiv.org/pdf/2604.28197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 585. Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling

**arXiv ID:** 2604.28185 | [PDF](https://arxiv.org/pdf/2604.28185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 586. Essential, Yet Overlooked: Identity Verification Barriers for Blind and Low Vision People in Government Services

**arXiv ID:** 2604.28166 | [PDF](https://arxiv.org/pdf/2604.28166v1)

**作者:** Ryan John Oommen `[一作]` (Penn State), Tanusree Sharma `[通讯]` (Penn State)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析219条Reddit帖子和访谈16名盲/低视障用户，系统评估了政府身份验证流程对BLV用户的可访问性与安全性影响；

**💡 创新点**

创新点在于提出“可访问性-安全性失败路径”模型，揭示盲/低视障用户在无法完成官方验证时被迫采取低安全性的工作绕行，并首次把物理验证设施纳入评估；

**🔧 技术方法**

方法包括文本主题分析、半结构化访谈分析，以及对现有身份验证技术（如OTP、面部识别、文件上传、liveness检测）的可访问性评估；

**📊 数据集**

数据集为来自Reddit的219条相关帖文与16名BLV受访者的访谈记录；

**📈 对比分析**

对比方法为将BLV用户的工作绕行与标准验证流程进行对照，量化验证失败率、工作循环次数等指标，结果显示BLV用户平均需要多达3次循环验证并且安全等级下降；

**⚠️ 局限性**

局限性包括样本量相对有限、主要集中在美国社会保障系统，且未对不同技术实现的可访问性进行量化实验，可能影响结论的普适性。

---

## 587. Generalizable Sparse-View 3D Reconstruction from Unconstrained Images

**arXiv ID:** 2604.28193 | [PDF](https://arxiv.org/pdf/2604.28193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 588. Synthetic Computers at Scale for Long-Horizon Productivity Simulation

**arXiv ID:** 2604.28181 | [PDF](https://arxiv.org/pdf/2604.28181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 589. Strait: Perceiving Priority and Interference in ML Inference Serving

**arXiv ID:** 2604.28175 | [PDF](https://arxiv.org/pdf/2604.28175v1)

**作者:** Haidong Zhao `[一作]` (Inria & Sorbonne University), Nikolaos Georgantas `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了 Strait，一套面向高 GPU 利用率的 ML 推理服务系统，旨在通过优先级调度和干扰预测来提升双优先级流量的截止期限满足率。

**💡 创新点**

创新点包括：1) 针对数据传输和核执行的干扰建模；2) 通过可在线自适应的指数式干扰预测模型实现实时干扰估计；3) 将干扰预测融入优先级感知调度算法，动态限制低优先级任务的吞吐量并预判新批次是否会违反现有任务截止期限；4) 通过早期丢弃和自适应吞吐量上限实现公平的任务优先级。

**🔧 技术方法**

使用技术主要有：CUDA 流优先级、TensorRT 推理运行时、Nsight Compute 资源监测、在线学习（Adam+Huber 损失）更新干扰预测模型、基于二分搜索的批次大小选择、基于时间加权的资源压力估计。

**📊 数据集**

采用六个常见预训练模型（ResNet‑50、ViT‑B‑16、ConvNeXt‑B、VGG‑19、YOLO‑v8n、RoBERTa‑B）作为工作负载，生成 Poisson 随机流、均匀流以及真实生产的 serverless 跟踪，评估在单 GPU 与 4 GPU 节点上的表现。

**📈 对比分析**

与时间共享、静态空间共享、反应式空间共享以及 XSched（基于内核级预抢占的 Triton 扩展）进行对比。Strait 在高 GPU 利用率下将高优先级任务的截止违约率下降 1.02–11.18 个百分点，低优先级任务仅略有提升（≤0.61 个百分点）。整体良好吞吐量与负载相匹配，且系统调度开销仅几十微秒。

**⚠️ 局限性**

局限性：1) 预测模型对极端尾部干扰（>600%）的误差仍较大；2) 主要针对双优先级场景，缺乏多级优先级或集群层面的调度；3) 仅在离散 GPU 上验证，未考虑集成 GPU、Edge 设备或 LLM 生成型工作负载；4) 依赖离线资源吞吐量剖面，若剖面漂移需手动更新；5) 低优先级任务在高负载时可能被过度限制。

---

## 590. Action Motifs: Self-Supervised Hierarchical Representation of Human Body Movements

**arXiv ID:** 2604.28173 | [PDF](https://arxiv.org/pdf/2604.28173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 591. Superpolynomial Length Lower Bounds for Tree-Like Semantic Proof Systems with Bounded Line Size

**arXiv ID:** 2604.28172 | [PDF](https://arxiv.org/pdf/2604.28172v1)

**作者:** Susanna F. de Rezende `[一作]` (Lund University), Kilian Risse `[通讯]` (Lund University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5086575008)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文证明了在语义树状 Frege 推理系统（以及树状阈值系统、剪枝平面等）中，即使线条大小受限，平均情况下不满足的 CNF 公式（主要以随机图中的 k‑clique 为例）也需要指数级长度的反证；

**💡 创新点**

创新点在于将伪测度（pseudo‑measure）方法推广到对任意有限线条数的语义树状证明系统，得到几乎最优的长度下界；

**🔧 技术方法**

核心技术包括构造线性伪测度、利用 VC 维度与核心图（core graph）分析、以及对随机图的概率估计（Chernoff 与匹配分析）来限制树状证明的叶子数；

**📊 数据集**

使用的数据集为 Erdős–Rényi 随机图（p≈n^{-2/D}）生成的随机 k‑clique CNF 公式；

**📈 对比分析**

与以往仅针对弱证明系统（如解析、剪枝平面）或特定编码的结果相比，本工作在更强的树状证明系统上取得了指数级下界，表明即便是强证明系统在平均情况下也无法实现多项式长的反证；

**⚠️ 局限性**

局限性在于仅对树状（tree‑like）语义证明系统给出下界，未涉及有向无环图（dag‑like）证明，也未针对特定语法规则的推导过程；此外，伪测度方法虽然通用，却尚未能直接体现对具体推导规则的依赖。

---

## 592. LaST-R1: Reinforcing Action via Adaptive Physical Latent Reasoning for VLA Models

**arXiv ID:** 2604.28192 | [PDF](https://arxiv.org/pdf/2604.28192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 593. Computing Equilibrium beyond Unilateral Deviation

**arXiv ID:** 2604.28186 | [PDF](https://arxiv.org/pdf/2604.28186v1)

**作者:** Mingyang Liu `[一作]` (Massachusetts Institute of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 28261 | [OpenAlex ID](https://openalex.org/A5067307504)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并求解最小平均强平衡（MASE），一种能最小化任何联盟平均离开动机的相关策略概念，并给出其计算复杂性分析与可行算法；同时通过MASE框架求解可利用性-社会福利前沿（EWF）。

**💡 创新点**

①首次定义可行且总是存在的多方偏离最小化概念MASE；②证明MASE的计算即使在单玩家联盟下也是NP‑hard，且基于树宽的下界不可避免；③设计与树宽指数成正比的FTRL/FTPL+动态规划算法，匹配下界；④将该框架应用于EWF计算，给出可利用性与福利的权衡。

**🔧 技术方法**

使用无后悔学习（Follow‑the‑Perturbed‑Leader）在相关-偏离元游戏中迭代更新；动态规划利用游戏的树分解（tree decomposition）高效求解局部最优；凸优化/线性规划理论用于证明最优解稀疏性；复杂度分析基于SETH与树宽概念。

**📊 数据集**

主要使用合成数据集：随机N人N维普通形式游戏（行动集大小A取固定值，效用在[0,1]均匀采样并归一化）；随机多项式矩阵游戏；以及经典博弈（囚徒困境、猎鹿）作为验证案例。并未使用真实世界大规模游戏。

**📈 对比分析**

与传统无后悔学习基线（FTRL、Hedge、FTPL、OMD）以及通过线性规划得到的精确MASE进行对比；指标包括单玩家利用率、联盟利用率和社会福利。实验表明：MASE在联盟利用率上显著优于基线，社会福利更高；单玩家利用率基本与基线相当，说明MASE在单方偏离上不显著劣化。

**⚠️ 局限性**

• 计算仍受树宽限制，树宽大时指数时间；• MASE优化平均收益，最小化最小收益仍NP‑hard，无法通过同样方法求解；• EWF在ε=0时求解仍NP‑hard；• 仅在合成游戏上验证，缺乏对真实复杂多方交互系统的实验；• 需要预先给定树分解，实际构造或近似树分解的成本未详细讨论。

---

## 594. Stop Holding Your Breath: CT-Informed Gaussian Splatting for Dynamic Bronchoscopy

**arXiv ID:** 2604.28179 | [PDF](https://arxiv.org/pdf/2604.28179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 595. AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images

**arXiv ID:** 2604.28177 | [PDF](https://arxiv.org/pdf/2604.28177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 596. RopeDreamer: A Kinematic Recurrent State Space Model for Dynamics of Flexible Deformable Linear Objects

**arXiv ID:** 2604.28161 | [PDF](https://arxiv.org/pdf/2604.28161v1)

**作者:** Tim Missal `[一作]` (Technical University of Darmstadt), Paula Dornhofer Paro Costa `[通讯]` (Universidade Estadual de Campinas)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5047253278)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于隐式动力学的框架——RopeDreamer，用于对可变形线性物体（DLO）进行长期、精确的状态预测与拓扑保持；

**💡 创新点**

创新点在于：①将DLO建模为四元数姿态链，天然约束连杆长度，消除非物理拉伸；②采用Recurrent State Space Model（RSSM）在隐空间学习全局动力学，拆分确定性与随机性状态；③使用双解码器架构，将当前状态重建与未来状态预测分离，提升预测稳定性；

**🔧 技术方法**

技术手段包括：RSSM（GRU+变分推理），四元数姿态链表示，动作编码器（基于链段索引+位移），ELBO损失与KL正则化，双解码器（重建解码器+预测解码器），以及基于Gauss Code的拓扑评估；

**📊 数据集**

数据集：在MuJoCo模拟环境中生成10,000条DLO轨迹（共1M步），每条轨迹100步，DLO由70个胶囊组成，随机抓取并移动，产生自交叉、缠绕等复杂形态；

**📈 对比分析**

对比方法：GA-Net与IN‑BiLSTM。实验结果显示，在50步开放式预测中，RopeDreamer（大模型）相较于GA-Net平均RMSE下降40.52%，并且拓扑匹配率显著高于10%；推理时间比小型GA-Net快31.17%；

**⚠️ 局限性**

局限性：隐空间编码导致初始重建误差，仿真与真实物理差距尚未解决，需要在线系统辨识或层次隐空间；目前仅在模拟环境验证，实际机器人平台与视觉追踪的鲁棒性待进一步研究。

---

