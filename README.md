# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-09 | 今日论文总数: 644

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Does Country Matter? Evaluating Geographic Sensitivity of Large Language Models in Zero-Shot Chronic Kidney Disease Screening

**arXiv ID:** 2611.08507 | [PDF](https://arxiv.org/pdf/2611.08507v1)

**作者:** Sirajam Munira `[一作]` (Rensselaer Polytechnic Institute), Muhammad Ashad Kabir `[通讯]` (Charles Sturt University)

**通讯引用:** 3957 | [OpenAlex ID](https://openalex.org/A5085181943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在零样本慢性肾病筛查中，改变提示中的国家属性（保持患者特征不变）对大型语言模型预测的影响。

**💡 创新点**

首次系统性探讨地理上下文作为提示信息如何在临床推理中充当隐式先验，揭示非临床特征能显著改变模型输出。

**🔧 技术方法**

采用基于指令的零样本提示，使用两种开源大型语言模型（Qwen3‑8B、Mistral‑7B），并用 McNemar 检验评估不同国家上下文的统计显著性。

**📊 数据集**

使用孟加拉国社区慢性肾病数据集（284例，其中112例CKD，172例非CKD），挑选共同正确的156例作为实验样本。

**📈 对比分析**

通过将 26 个国家上下文（原始孟加拉国加 25 个国家）嵌入提示并重新推理，比较 CKD 与非 CKD 的敏感度。Qwen3 在多数国家的 CKD 敏感度从 100% 降至 76.9%，并在多国显著；Mistral 则几乎保持 100%（仅冰岛略降至 96.9%）。

**⚠️ 局限性**

局限性包括仅使用单一孟加拉国数据集、仅评估两款模型、仅考虑国家属性，未能解释机制且结果可能不具普适性。

---

## 2. What Are LLM Agents Actually Learning on Low-Budget Molecular Optimization Benchmarks?

**arXiv ID:** 2611.08522 | [PDF](https://arxiv.org/pdf/2611.08522v1)

**作者:** Jiayu Qin `[一作]` (University at Buffalo), Zhiqiang Xu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3268 | [OpenAlex ID](https://openalex.org/A5101799656)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估LLM驱动分子优化在低调用预算下的真实贡献，拆解检索、在线搜索与oracle几何的效益；

**💡 创新点**

首次系统对PMO和蛋白质对接任务进行信息拆解与基准化，提出检索安全提示与随机漫步基准，明确检索主导效果；

**🔧 技术方法**

采用ReAct式LeadOpt-Agent、gpt‑4o‑mini、检索式基线、MolLEO‑v4、GraphGA、随机单步变异等技术；

**📊 数据集**

使用PMO（23个任务）、ProteinLeadOpt‑50（10个蛋白+5种starter共50对）、TDC、Dockstring、AutoDock Vina等公开数据集；

**📈 对比分析**

与传统REINVENT、GraphGA、随机漫步比较，检索仅占84%得分，C1‑C5拆解显示检索占主要得分，随机漫步在ProteinLeadOpt‑50上可获更高的对接改进和复合分数；

**⚠️ 局限性**

主要局限在单一模型（gpt‑4o‑mini）与单一Oracle（Vina）上，随机种子波动大，且对更复杂的力场或筛选标准的适用性尚未验证；

---

## 3. InstaGen: Group-Shared Importance Sampling for Fast Video Generation

**arXiv ID:** 2611.08610 | [PDF](https://arxiv.org/pdf/2611.08610v1)

**作者:** Kislay Aditya Oj `[一作]` (Indian Institute of Technology Bombay), Aditya Desai `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5137021734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 InstaGen，一种将稀疏注意力视为重要性采样的框架，取代现有 PBSA 方法的 top‑p 选择，通过精确计算高分块并用加权样本估计其余块，实现高效且无偏的注意力计算。

**💡 创新点**

创新点在于：①将稀疏注意力转化为采样问题，形成统一的无偏估计器；②采用组共享采样（每个查询簇共享同一采样块），保持块结构，兼容 FlashAttention，显著降低 GPU 计算开销；③在不需要 dense warm‑up 的前提下实现高质量视频生成。

**🔧 技术方法**

技术手段包括：Permuted Block‑Sparse Attention（PBSA）、k‑means 查询/键簇化、PPSWOR（重要性采样）与均匀采样、SVG2 的簇分数、FlashAttention‑3 GPU 加速、UniPC 推理、VBench 视觉质量评估。

**📊 数据集**

数据集与模型：Wan2.1‑T2V‑1.3B 与 Wan2.1‑T2V‑14B 文本到视频扩散模型，720p 81 帧（50 denoising 步）以及 22 条 VBench 评测提示。

**📈 对比分析**

与 dense、SVG、SVG2、SVG‑EAR 等稀疏方法比较，InstaGen 在相同 attention 密度下 PSNR 提升 3.4 dB（1.3B）/2 dB（14B），在相同壁钟时间下提升 1.6–2.2 dB；在 14B 模型上以相近密度击败 SVG‑EAR 1.1 dB，显著缩短 561 帧视频的生成时间（57.3 min vs 82.8 min）。

**⚠️ 局限性**

局限性：对低密度或极端高熵注意力分布时采样方差仍可能较大；目前仅验证了 SVG2 的簇分数，其他分数器的兼容性待进一步评估；实验仅在特定分辨率和模型规模下进行，未覆盖更长视频或多模态扩散模型的泛化情况。

---

## 4. Forcing a Letter Suppresses Refusal: Mechanistic Evidence from WMDP

**arXiv ID:** 2611.08639 | [PDF](https://arxiv.org/pdf/2611.08639v1)

**作者:** Stepan Epifantsev `[一作]` (Independent researcher), Alexey Dontsov `[通讯]` (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多选题(MCQ)格式如何抑制大型语言模型在安全评估中的拒绝行为，并给出机制解释。

**💡 创新点**

首次从机制层面揭示MCQ格式通过快速答案选择抑制拒绝路径，结合调优透镜、拒绝方向几何和注意力头消融提供三条证据。

**🔧 技术方法**

使用调优透镜(tuned lens)、拒绝方向分析(refusal‑direction)与注意力头消融(attention‑head ablation)三种技术。

**📊 数据集**

在Weapons of Mass Destruction Proxy (WMDP)多选题数据集上，使用两款8B指令微调模型Llama‑3.1‑8B和Qwen3‑8B进行实验。

**📈 对比分析**

通过层级可视化和几何对齐，证明MCQ模式下答案选取窗口与拒绝构建窗口不重叠，MCQ准确率与开放式拒绝无关；消融实验进一步验证因果关系，表现为正确答案概率下降、拒绝概率上升。

**⚠️ 局限性**

仅在8B模型上验证，Qwen3‑8B对消融不敏感；拒绝方向提取基于BeaverTails与WMDP领域可能不匹配，且仅关注最后令牌层，可能遗漏其他机制。

---

## 5. Proper Conflict-Free Coloring of Undirected Path, Chordal Bipartite, and Strongly Chordal Graphs

**arXiv ID:** 2611.08534 | [PDF](https://arxiv.org/pdf/2611.08534v1)

**作者:** Gautam K. Das `[一作]` (Indian Institute of Technology Guwahati), Kamal Santra `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5032937760)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了“proper conflict‑free”着色的复杂性与算法，证明了在无向路径图与割边长为六以上的割边图上判断是否仅用 ω(G) 颜色可行是 NP‑完整的，并给出在强直径图上加一颜色的上界；在严格直径图和具有小团树分离器的强直径图上给出了多项式时间的最优着色算法。

**💡 创新点**

首次将“proper conflict‑free”着色引入路径图与割边图的复杂性分析；提出通过简单消除、完全平衡集合与团树分离器的树形动态规划，获得强直径图的加一上界及若干特殊子类的精确算法；同时提供了多项式时间的线性时间解法与指数级别的分离器参数化算法。

**🔧 技术方法**

主要技术包括：归约证明（从 3‑SAT 与 3‑均等问题），无向路径图与割边图的路径/星形表示，简单消除顺序与闭邻域包含关系，完全平衡集合与树形结构的构造，最大团约束的预算问题，团树分离器的树形分解与局部状态枚举。

**📊 数据集**

该工作为纯理论研究，无使用实测数据集；所有结果均通过构造性证明与多项式时间算法实现。

**📈 对比分析**

相比已有的“unique‑maximum”或普通着色结果，本研究在路径图与割边图上完成了 NP‑完整性判定，并在强直径图、严格直径图以及分离器大小受限的强直径图上提供了多项式或指数参数化时间的最优着色算法；在理论复杂度上取得了新的边界。

**⚠️ 局限性**

主要限制包括：强直径图（甚至区间图）整体最优着色的复杂性仍未解决；割边图的三色判定仍是开放问题；在分离器大小为 s 的强直径图中，算法的时间指数为 n^2^O(s+1)，对大 s 仍显高；未进行实验验证或性能评估。

---

## 6. Adaptive Depth Aggregation in Graph Transformers for Long-Range Graph Reasoning

**arXiv ID:** 2611.08500 | [PDF](https://arxiv.org/pdf/2611.08500v1)

**作者:** Yifeng Lu `[一作]` (Sun Yat-sen University), Zhen Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 27549 | [OpenAlex ID](https://openalex.org/A5100422396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了节点自适应深度聚合机制 AttnRes，改进 Graph Transformer 的残差累积方式。

**💡 创新点**

创新点在于将残差贡献视为因果节点历史，并使用注意力机制对不同深度贡献进行节点级加权，从而解决固定单位权重导致的深度聚合不匹配问题。

**🔧 技术方法**

使用 Attention Residuals（AttnRes）、Block/Full 版本、GPS 与 GRIT Graph Transformer 架构，以及标准 GNN/Transformer 基线进行对比实验。

**📊 数据集**

在 ECHO 基准的五个任务上进行评估，任务包括图直径回归、节点距心度回归、单源最短路径回归、分子能量预测与分子电荷预测。

**📈 对比分析**

通过与同一基底独立调参的 GPS/GRIT、以及其他 GNN/Transformer 方法对比，AttnRes 在所有任务上均取得更低的平均绝对误差（MAE），并且至少在每个任务中有一个变体得到最低 MAE。

**⚠️ 局限性**

局限性包括仅在 GPS 与 GRIT 两种架构与 ECHO 基准上验证，额外的激活内存与运行时间开销，以及缺乏对更大规模图或其他任务的广泛评估。

---

## 7. MORTIS: Compiler-Guided KV-Cache Management for LLM Agent Programs

**arXiv ID:** 2611.08638 | [PDF](https://arxiv.org/pdf/2611.08638v1)

**作者:** Xiaoyu Ma `[一作]` (Shanghai Jiao Tong University), Jason Mars `[通讯]` (University of Michigan)

**通讯引用:** 7728 | [OpenAlex ID](https://openalex.org/A5053236545)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种编译器-服务协同设计，通过分析LLM代理程序的提示（prompt）内部异质性来改进KV缓存管理，主要包括提示重排和主动缓存调度。

**💡 创新点**

创新点在于：①识别提示内部的“intra‑prompt heterogeneity”，将提示组件与程序变量对应；②利用编译器静态分析恢复共享模式和生命周期；③结合运行时观测，动态重排提示顺序以暴露更长的公共前缀；④根据组件寿命预测未来使用，进行缓存保留、驱逐和预取；⑤在同一框架内统一完成这些工作，无需额外标注。

**🔧 技术方法**

使用技术包括：LLVM/中间表示的编译器分析（控制流图、数据流、到达定义、演化类别分类）；概率有限状态自动机用于建模调用顺序与间隔；基于上下文的预测（Witten‑Bell插值）预测下一调用；自定义KV缓存分层和四个生命周期带；提示重排算法（共享绑定排序、私有绑定排序、指令规范放置）。

**📊 数据集**

实验数据集：四类LLM代理任务——(1) 事实核查代理；(2) HumanEval 编码代理；(3) BFCL 工具使用代理；(4) FinanceBench 财务分析代理；以及将四类混合在一起的混合工作负载。

**📈 对比分析**

与基线比较：SGLang、CacheScout、Continuum、KVFlow、Static‑Adapter。结果显示在单一GPU (A40) 上，平均工作完成时间（JCT）提升 1.45×、吞吐量提升 1.49×，最高分别可达 1.99× 与 1.97×。在低并发时提示重排贡献最大；在高并发时主动缓存调度发挥显著作用。混合工作负载下仍保持 1.56× 吞吐、1.53× JCT 的几何平均加速。

**⚠️ 局限性**

局限性：①预取对性能贡献有限，主要提升来自缓存保留；②方法依赖于类型化LLM函数调用和编译时信息，无法直接迁移到无类型或动态生成提示的框架；③在更小的GPU/模型上仍需验证，可能需要更细粒度的调度策略；④当前实现依赖对CFG的完整解析，复杂代码路径可能导致保守的 VOLATILE 推断。

---

## 8. Enhancing Academic Advisement with Case-Based Reasoning

**arXiv ID:** 2611.08572 | [PDF](https://arxiv.org/pdf/2611.08572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 9. LLM-Based Feature Extraction for Multi-Criteria Inventory Classification

**arXiv ID:** 2611.08626 | [PDF](https://arxiv.org/pdf/2611.08626v1)

**作者:** Humza Akhtar `[一作]` (MongoDB Inc.), Rami Pinto `[通讯]` (MongoDB Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了将LLM提取的非结构化特征（如客户评论情感）融合进多准则库存分类模型，以提升库存优先级划分。

**💡 创新点**

创新点在于首次将LLM生成的定量评估标准和向量嵌入结合，利用非结构化数据丰富传统MCIC框架，提供可解释的优先级改进。

**🔧 技术方法**

使用了大型语言模型（LLM）、文本向量嵌入、MongoDB Vector Search以及Hadi‑Vencheh多准则优化模型。

**📊 数据集**

使用Olist巴西电子商务公开数据集（包含订单、交付和评论信息）。

**📈 对比分析**

通过与传统ABC、传统MCIC比较的转移矩阵评估改分类，结果显示将LLM特征加入后分类保留率提升，部分商品被上调或下调优先级。

**⚠️ 局限性**

局限在于未验证重分类是否带来实际运营收益，且对LLM评分的偏差与鲁棒性未做深入检测。

---

## 10. Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies

**arXiv ID:** 2611.08498 | [PDF](https://arxiv.org/pdf/2611.08498v1)

**作者:** Sakshi Garg `[一作]` (San José State University), William B. Andreopoulos `[通讯]` (San José State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种混合检索增强生成（Hybrid RAG）框架，融合知识图谱与密集向量检索来解决长文档多跳问答问题。

**💡 创新点**

创新点在于：1）动态语义分块+摘要预处理，减少上下文碎片；2）双索引结构（KG + 向量库）与分层检索策略，先尝试图谱推理，失败则退回向量检索；3）可解释性路径评分与融合机制。

**🔧 技术方法**

使用了Transformer摘要模型、语义分块算法、基于LLM的实体关系抽取、Neo4j知识图谱、FAISS向量数据库、BM25稀疏检索、Reciprocal Rank Fusion、LLM生成（GPT‑4o/Llama‑3‑70B）。

**📊 数据集**

评估数据集为 MultiHop‑RAG（约2,556个多跳问答对），并对其进行分域抽样测试。

**📈 对比分析**

与纯向量检索（Vector RAG）和纯图谱检索（Graph RAG）基线对比，Hybrid RAG在Exact Match、Context Precision/Recall、Top‑5检索准确率等指标上均优于两者，Exact Match提升约+4.3%，但推理延迟略升至1.7s。

**⚠️ 局限性**

局限包括：①对知识图谱构建的依赖，若实体关系缺失会退回向量检索；②构建图谱和索引的计算成本与同步维护；③在极长或高噪声文本中仍可能出现上下文碎片；④推理延迟略高，适用场景受限。

---

## 11. BEVCast: Path-Adaptive BEV for Efficient End-to-End Autonomous Driving

**arXiv ID:** 2611.08631 | [PDF](https://arxiv.org/pdf/2611.08631v1)

**作者:** Qiren Wang `[一作]` (University of Delaware), Weisong Shi `[通讯]` (University of Delaware)

**通讯引用:** 25081 | [OpenAlex ID](https://openalex.org/A5100651611)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为BEVCast的两阶段鸟瞰视图（BEV）模型，该模型根据预测的驾驶路径自适应调整分辨率，优化计算资源。

**💡 创新点**

创新点在于通过热头预测车辆将要经过的粗细网格单元，并在固定预算下对高热量单元进行细化，从而实现路径自适应的BEV表示。

**🔧 技术方法**

使用了深度学习技术，包括多层感知机（MLP）和变换器（Transformer）架构，结合图像特征和自我注意力机制。

**📊 数据集**

在nuScenes和NAVSIM数据集上进行评估，使用了记录的驾驶路径作为监督信号。

**📈 对比分析**

与全网格1.024米单元的模型进行比较，BEVCast在10%的细化预算下编码了28%的BEV查询，且在nuScenes上的平均位移误差为1.078米，低于1.149米，且可驾驶区域违规率为1.46%，低于2.29%。

**⚠️ 局限性**

限制在于BEVCast与全网格在选择器编码器和粗记忆网格等方面存在差异，且未进行闭环重规划测试，缺乏导航命令可能导致输出轨迹在转弯和直行之间平均化。

---

## 12. Meituan-Robotics-0: A Vision-Language-Action Foundation Model for Desktop Manipulation

**arXiv ID:** 2611.08577 | [PDF](https://arxiv.org/pdf/2611.08577v1)

**作者:** Meituan Robotics Team `[一作]` (Meituan), Ziqin Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出Meituan‑Robotics‑0（MR0）——一种面向桌面静态机器人操作的视觉‑语言‑动作（VLA）基础模型及其完整的执行框架。

**💡 创新点**

核心创新在于①通过跨模态三阶段训练（离散预训练、连续跨体裁微调、真实世界后训练）将海量异构数据迁移到连续控制；②RollBack数据采集与CFG‑DAgger直接利用被替代与人类补正片段进行正负标签学习；③异步动作块融合与事件驱动的高层代理实现持续控制与跨任务记忆。

**🔧 技术方法**

使用的技术包括：大型预训练视觉‑语言 Backbone（Qwen3.5‑4B）、FAST离散动作编码、MMDiT流匹配连续动作头、CFG‑DAgger、RECAP、HG‑DAgger、异步执行层（EVA‑Client）、事件监测/记忆/规划体系。

**📊 数据集**

数据集：147M视觉‑语言样本（78%通用+22%具身），15,060小时机器人轨迹（RoboDesk 3,235h + 7,530h授权+4,295h仿真），以及多任务真实机器人演示与人机纠错数据。

**📈 对比分析**

与现有方法对比：在RoboTwin‑2.0上取得93.6%平均成功率（Clean 93.2%，Random 93.9%），在RoboDojo VLA轨道排名第5；在三项真实机器人任务（字母块放置、衣物折叠、以太网插入）中，MR0在HG‑DAgger后均与π_0.5匹敌或优于其成功率与进度得分，CFG‑DAgger在以太网插入任务中实现70%成功率与0.90进度分，优于RECAP与HG‑DAgger。

**⚠️ 局限性**

局限性包括：对开放词汇指令的理解仍不稳定；对长周期任务的记忆依赖事件管理仍可能导致计划滞后；系统整体复杂度高，需多模块协同；在高度动态或多机器人场景下的泛化尚未验证。

---

## 13. Bourgain's Theorem for a Union of Euclidean Metric Spaces

**arXiv ID:** 2611.08546 | [PDF](https://arxiv.org/pdf/2611.08546v1)

**作者:** Dimitar Chakarov `[一作]` (Toyota Technological Institute at Chicago), Ali Vakilian `[通讯]` (Rutgers Business School)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了将k个欧几里得集合的并嵌入欧几里得空间的失真问题，证明了可以以O(k^1.128log k)的失真将其嵌入，这一结果比之前的2^O(k)的上界有了显著改进。

**💡 创新点**

在假设每个欧几里得集合在加入任意两个点后仍保持欧几里得的条件下，得到了O(log k)的渐近紧失真，提供了对Bourgain定理的推广。

**🔧 技术方法**

使用了Lipschitz映射和终端嵌入的技术，结合了对个别欧几里得集合的Lipschitz扩展常数的分析。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了k个欧几里得集合的并的性质。

**📈 对比分析**

通过与之前的结果进行比较，展示了新的失真上界O(k^1.128log k)的优越性，并且在特定条件下达到了O(log k)的失真，表明在k=2的情况下，失真可以小于3.997。

**⚠️ 局限性**

限制在于对于一般的k个欧几里得集合，仍然存在一些情况下Lipschitz扩展常数可能达到Ω(√(log k))，这表明不能仅依赖于Lipschitz扩展来获得O(log k)的失真。

---

## 14. Akira: Towards Operating System Architecture for Agile Specialization

**arXiv ID:** 2611.08511 | [PDF](https://arxiv.org/pdf/2611.08511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 15. RoadBlock: Rethinking GPU Tensor Core Microarchitecture for Emerging Microscaling Format Support

**arXiv ID:** 2611.08526 | [PDF](https://arxiv.org/pdf/2611.08526v1)

**作者:** Nikhil Rout `[一作]` (University of California), Blaise Tine `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了RoadBlock框架，用于在可编程GPU Tensor Core上对自定义Microscaling (MX)格式进行完整的端到端建模和评估，自动推导线程级元数据并支持多种格式参数化。

**💡 创新点**

创新之处在于突破了以往仅在独立MAC单元原型上评估的局限，提供了在Vortex GPGPU上实现MX支持的完整软件/硬件链路，包括自动元数据划分、寄存器映射以及与现有Tensor Core微架构的无缝集成。

**🔧 技术方法**

采用了RISC‑V Vortex GPGPU、Ten‑Four混合精度FEDP微架构、TCU‑SRAM隐藏元数据机制、扩展WMMA ABI、Yosys + ASAP7 进行格式单元综合，以及在Llama3.1‑8B和Qwen3‑32B上进行Perplexity评测。

**📊 数据集**

使用了WikiText‑2、Llama3.1‑8B和Qwen3‑32B等语言模型数据集进行评估。

**📈 对比分析**

在不同Tensor Core缩放点（N=2,4,8,16）下，对比Perplexity与硬件效率（面积、静态功耗）发现：MXFP4在成本与精度之间取得平衡；IF4精度好但功耗/面积高；RaZeR在较长FEDP长度下提供最佳Perplexity‑PPA权衡。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，缺乏真实硅级实现；对更复杂多级块尺度格式的支持仍有限；未探索Warp专用核与多级块尺度格式的系统级性能影响。

---

## 16. Minimum-Sufficient Knowledge and Computation for Wind Energy Predictive Analysis

**arXiv ID:** 2611.08558 | [PDF](https://arxiv.org/pdf/2611.08558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. PINNeval: A Comprehensive Evaluation Standard for Physics-Informed Neural Networks

**arXiv ID:** 2611.08559 | [PDF](https://arxiv.org/pdf/2611.08559v1)

**作者:** Kevin von Bargen `[一作]` (Osnabrück University), Tim Römer `[通讯]` (Osnabrück University)

**通讯引用:** 863 | [OpenAlex ID](https://openalex.org/A5038968457)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的评估协议，将物理信息神经网络（PINN）训练管道系统地分为四个组件，并要求对未调查的组件进行竞争性实现。

**💡 创新点**

创新点在于提出了一个综合的评估协议，强调了四个组件之间的相互作用，并要求在比较时提供所有组件的配置和性能指标。

**🔧 技术方法**

使用了JAX包实现PINNpipe，包含18种不同的方法和九个高效实现的基准问题。

**📊 数据集**

使用了多个基准问题进行评估，具体问题未在摘要中列出，但涵盖了广泛的真实、混沌和多尺度问题。

**📈 对比分析**

通过与现有基准研究的比较，结果挑战了常见假设，表明在竞争性管道下，某些方法的表现优于以往的评估结果。

**⚠️ 局限性**

限制在于当前研究仅限于PINN和正向问题，复杂几何形状的问题和逆向问题尚未纳入评估范围。

---

## 18. Q-OIL: Optimistic Exploration in Learning from Human Interventions

**arXiv ID:** 2611.08576 | [PDF](https://arxiv.org/pdf/2611.08576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 19. AutoPYara: Next-Gen YARA Rule Generator for Malware Family Clustering

**arXiv ID:** 2611.08608 | [PDF](https://arxiv.org/pdf/2611.08608v1)

**作者:** Mabon Ninan `[一作]` (Texas A&M University), Marcus Botacin `[通讯]` (Texas A&M University)

**通讯引用:** 721 | [OpenAlex ID](https://openalex.org/A5060510652)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出 AutoPYara，一种基于聚类中心选择和子聚类数估计的自动 YARA 规则生成框架；

**💡 创新点**

创新点在于（1）利用相似度哈希精准选取聚类中心，提升规则覆盖率；（2）通过历史 VirusTotal 数据估计子聚类数，克服原始 AutoYara 的低估问题；

**🔧 技术方法**

技术主要包括 Fuzzy C-Means 与 DBSCAN 的增强聚类、增强的 AugmentedK-Means、基于相似哈希与 VT 标签的阈值与启发式 k 估计，以及 Java 后端与 Python 前端的整合；

**📊 数据集**

使用了约 71,000 条 Windows PE 恶意软件样本和 22,000 条良性软件样本，来源于美国、巴西和日本的 AV 采集；

**📈 对比分析**

通过与 AutoYara 的对比实验，AutoPYara 在规则覆盖率上提升 14 个百分点，方差下降 7.1 个百分点，威胁狩猎准确率提升 10%；

**⚠️ 局限性**

局限在于对 VT 标签的依赖导致在标签不一致时可能产生过宽规则，且整体可重复性仍受聚类随机性影响。

---

## 20. Interpretable Multifidelity Learning with Kolmogorov-Arnold Networks for Spatiotemporal Nuclear Reactor Flow Modeling

**arXiv ID:** 2611.08512 | [PDF](https://arxiv.org/pdf/2611.08512v1)

**作者:** Meredith Eaheart `[一作]` (University of Michigan), Majdi I. Radaideh `[通讯]` (University of Michigan)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5049880944)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了利用多精度代理模型减少高精度CFD数据量，以预测193个燃料组件在压水堆核心底部的时空质量流率，并对比了多精度前馈神经网络(MFNN)与多精度Kolmogorov‑Arnold网络(MFKAN)两种方法。

**💡 创新点**

创新点包括提出了基于精度编码的MFKAN-Flag与三精度版MFKAN-Flag3f结构，能够生成可解释的符号方程，并将TabDDPM生成式模型用于数据稀缺的多精度CFD情形，实现合成数据扩增。

**🔧 技术方法**

主要技术手段包括多精度神经网络(MFNN与MFKAN)、主成分分析(PCA)降维、谱聚类、TabDDPM生成式数据增强以及对MFKAN输出进行符号回归。

**📊 数据集**

使用的数据集为一个完整四回路压水堆低层模型生成的CFD瞬态数据，包含低、中、高三种网格分辨率，每种精度下仅提供一条10,000步的质量流率时序，共193个燃料组件。

**📈 对比分析**

通过一阶步进预测实验比较两种模型，MFKAN在基线、聚类及相关性最高的50个组件集上与MFNN保持相近的预测精度（核心平均MAPE≈4%，峰值MAPE≈8%），但在TabDDPM增强的跨精度预测中两者都仅获得0.03–0.06的相关性，表明跨精度相关性弱是主要瓶颈。

**⚠️ 局限性**

主要局限在于每种精度仅有一条真实瞬态导致数据多样性不足，跨精度相关性弱限制了多精度方法的提升，MFKAN存在训练重复性波动，且生成式增强难以完全替代真实高精度数据，导致预测性能整体偏低。

---

## 21. Rethinking Data Selection as Distribution Matching on Semantic Manifolds

**arXiv ID:** 2611.08495 | [PDF](https://arxiv.org/pdf/2611.08495v1)

**作者:** Peng Sun `[一作]` (Nanjing University), Tianfan Fu `[通讯]` (Nanjing University)

**通讯引用:** 5233 | [OpenAlex ID](https://openalex.org/A5003226543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 QMAC 方法，结合多频特征匹配与质量过滤，用于指令调优数据的子集选择。

**💡 创新点**

首次通过多频特征函数距离显式约束全数据与子集在嵌入空间的分布一致性，并加入任务相关质量评分实现噪声样本过滤。

**🔧 技术方法**

使用多频特征函数距离（CFD）、随机子集归一化、基于 logits 的 TopK 采样，以及轻量级嵌入与质量评分模型。

**📊 数据集**

四大数据集：Vision‑Flan、OpenHermes、LLaVA‑CoT 与 SmallThoughts。

**📈 对比分析**

在 44 评测基准下，QMAC 在 5%–30% 子集预算下均优于随机、全量及多种基线，平均相对性能超过 100%。

**⚠️ 局限性**

依赖嵌入与质量评分的可靠性，需额外提取与优化步骤，且尚未验证在预训练或强化学习等场景中的适用性。

---

## 22. Aegis OS: A Transactional Multi-Tenant Operating System Runtime for Eliminating Authorization Fatigue in Autonomous Coding Agents

**arXiv ID:** 2611.08516 | [PDF](https://arxiv.org/pdf/2611.08516v1)

**作者:** Hui Zhao `[一作]` `[通讯]` (Heurion Research), Hui Zhao (Heurion Research)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Aegis OS——一种事务式多租户代理操作系统，利用 Copy‑on‑Write shadow 工作区、两阶段 LCS‑Diff3 合并、预提交语义验证、零信任出口代理、A2A IPC + CAS 等技术，解决自主编码代理在安全与自治之间的矛盾。

**💡 创新点**

创新点包括：① 将授权边界从单次操作提升到事务边界，减少中断；② 结合 APFS/Btrfs reflink 实现毫秒级 CoW 复制；③ 采用两阶段 LCS‑Diff3 合并与 sandbox 内部语义检查，保障多代理并发编辑无冲突；④ 通过 L7 方法/路径限定的边界密钥注入和 DLP 过滤实现零信任网络出口；⑤ 与 MCP 与 A2A 协议无缝对接，提供语言无关的工具调度和跨代理通信。

**🔧 技术方法**

技术栈：Rust 编写；macOS Seatbelt、Linux Bubblewrap（含默认 deny、网络 namespace）与微VM；APFS / Btrfs reflink / subvolume 快照；两阶段 LCS‑Diff3 合并引擎；Cedar‑inspired 策略引擎与会话租约；L7 级别方法/路径限定与 DLP 检测；Agent2Agent 与 Model Context Protocol；CAS 内容可寻址存储。

**📊 数据集**

数据集与实验：50 条人工设计的多步软件工程轨迹（1,500 次工具调用），52 条真实编码代理会话日志（28,087 次工具调用），5 条 Aegis OS 现场测试会话，80 次红队攻击实验（覆盖 8 类威胁）。

**📈 对比分析**

对比方法：与 Host‑Interactive、Host‑Static‑Allowlist、未加限制的 CLI 以及传统 stateless CLI 沙箱进行对照。性能：确认提示率从 57% 降至 <0.1%（最多 99.9% 降低），攻击成功率从 100% 降至 0%；CoW 复制 6.3 ms（macOS）/15.6 ms（Linux）相较 99 ms；IPC 载荷压缩 99.85%；多代理合并成功率 100%；系统整体延迟 <7 ms，几乎不影响 LLM 推理时间。

**⚠️ 局限性**

局限性：评估集中在单一企业内部的数据，未覆盖多样化开发者环境；仅在 macOS 与 VM Linux（Btrfs、XFS、ext4）上验证，其他文件系统与真实裸机环境需进一步测试；Linux 沙箱依赖 unprivileged namespaces，AppArmor 配置有限；同域混淆攻击仍有一定风险；大规模集成测试的预提交验证会产生数百毫秒延迟；部分功能如 Landlock、Firecracker 微VM 仍为未来工作。

---

## 23. The Containment Polytope of a Triangle and a Convex Polygon

**arXiv ID:** 2611.08556 | [PDF](https://arxiv.org/pdf/2611.08556v1)

**作者:** Yoonsung Choi `[一作]` (Korea Advanced Institute of Science and Technology), Sunghee Choi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3024 | [OpenAlex ID](https://openalex.org/A5064227877)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究三角形 P 在凸 n 边形 Q 内的最大全相似放置，提出将其相似变换的可视化多面体投影（silhouette）分解为线性数目的凸链和步长，并利用此分解在 O(n log n) 时间内计算 silhouette 的复杂度，并在 O(log n) 时间内查询任意方向下的最大放置。

**💡 创新点**

创新点在于：① 证明 silhouette 可以被拆分为线性数量的凸链和步长；② 通过几何分析（完美旋转中心、步长类型与角度关系）构造“stride”结构，并证明 stride 的数量为 O(n)；③ 结合二分搜索实现 O(log n) 的查询与 O(n log n) 的预处理。

**🔧 技术方法**

主要技术包括：几何变换与相似变换的四维多面体建模；完美旋转中心的构造与唯一性证明；角度与边缘交互的符号分析；链、步长与 stride 的状态机划分；使用二分搜索快速定位 pivot 和步长类型；堆栈机制用于追踪边缘的极值变化。

**📊 数据集**

论文为理论分析，未使用具体实验数据集；所有结果均在理论上给出算法复杂度与证明。

**📈 对比分析**

相较于之前最优的 O(n^2 log^2 n)（或 O(n^2)）算法，本文实现了 O(n log n) 的预处理和 O(log n) 的查询；在极限情况下，silhouette 复杂度仍为 O(n)（多多边形时为 Θ(mn^2)）。

**⚠️ 局限性**

主要限制：假设 Q 无平行边且处于一般位置（无重叠/共线等退化情况）；当前只针对三角形 P，扩展到一般多边形时仍保持高复杂度；对 Q 边界高度集中或稀疏时的实际性能未做实验验证。

---

## 24. SAFTAC: Simulation-Augmented Fine-Tuning of Open-Source LLMs for Analog Circuit Design

**arXiv ID:** 2611.08543 | [PDF](https://arxiv.org/pdf/2611.08543v1)

**作者:** Junsheng Huang `[一作]` (University of Illinois Urbana-Champaign), Huan Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8244 | [OpenAlex ID](https://openalex.org/A5100356986)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个包含 8,626 个任务、10 种不同类型模拟电路的模拟基准数据集，并提出了一种三阶段监督微调结合仿真增强强化学习的框架（SAFTAC），用于训练大型语言模型生成并纠正满足目标规格的 SPICE 兼容电路网表。

**💡 创新点**

创新点在于：1) 通过自动化数据生成管道将原始电路、文本描述、测试平台、负载与目标规格完整拼装成任务；2) 在微调过程中逐步引入电路知识、组件预测与仿真反馈修正，显著提升模型对电路拓扑与尺寸的理解；3) 在强化学习阶段使用实际 SPICE 仿真结果作为奖励，直接把功能合规性嵌入训练目标。

**🔧 技术方法**

使用技术包括：大语言模型（以 Qwen3-8B 为主），三阶段监督微调（SFT）、仿真增强强化学习（RFT）配合 GRPO 算法、Bayesian Optimization + SPICE 仿真管线、以及仿真反馈驱动的错误纠正任务。

**📊 数据集**

使用的数据集是论文自建的“Simulation‑Grounded Analog Circuit Design Tasks”数据集，包含 8,626 个任务，涵盖 10 种常见模拟电路（如放大器、振荡器、稳压器等），每个任务提供自然语言说明、I/O 标注、负载、测试平台、目标规格与参考网表，并由人工审核确保质量。

**📈 对比分析**

与多种基线（闭源 GPT‑5.2/5.4、Claude‑Opus‑4.7 以及开源 Qwen3‑32B、QwQ‑32B、gpt‑oss‑20B 等）进行比较。单 pass 生成下，SAFTAC 的 pass@1 71.4% 超过最强闭源 52.3%（提升 15.9%）和最强开源 38.6%（提升 31.4%）；两 pass（带仿真反馈）下，pass@1 78.2% 分别高于最强闭源 66.9%（提升 7.8%）和最强开源 53.3%（提升 24.9%）。在未见电路类型上也表现出优越的 OOD 泛化能力。

**⚠️ 局限性**

限制包括：1) 需要完整的 SPICE 仿真管线和 BO 调参，计算成本高；2) 数据集仍仅覆盖 10 种电路，未涵盖更复杂或特定行业的电路；3) 对仿真结果的依赖使得模型在仿真错误或噪声下可能产生误判；4) 训练与推理过程对算力要求较大，限制了实时性；5) 仍需人工验证生成的网表，难以完全自动化。

---

## 25. RedMall: Generative Recommendation for Large-Scale E-commerce

**arXiv ID:** 2611.08519 | [PDF](https://arxiv.org/pdf/2611.08519v1)

**作者:** Kunyang Zhou `[一作]` (Xiaohongshu Inc), Zheng Wang `[通讯]` (Tongji University)

**通讯引用:** 27654 | [OpenAlex ID](https://openalex.org/A5100401133)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RedMall，面向大规模电商的生成式推荐系统，结合多模态 SID 分词、层次 SID 编码器、稀疏 Mixture‑of‑Experts、MTP、多门控注意力及曝光感知损失与 GRPO+LambdaRank 的强化学习框架，实现会话级多项推荐。

**💡 创新点**

创新点包括：①基于残差量化的多层语义 ID 与轻量化 SID 编码器显式建模层间层次依赖；②多目标预测提升会话级推荐监督；③曝光感知负样本损失与 GRPO+LambdaRank 结合，兼顾用户偏好对齐与列表级排序；④稀疏 Mixture‑of‑Experts 提升解码器表达能力；⑤双门控注意力与双向历史注意力增强用户兴趣建模。

**🔧 技术方法**

使用技术包括：多模态表征学习、残差向量量化、Transformer 编码器‑解码器、稀疏 Mixture‑of‑Experts、双门控注意力、MTP、曝光感知损失、GRPO 强化学习、LambdaRank 排序损失、Flash Attention 2.0 加速等。

**📊 数据集**

实验数据集为 Amazon Beauty、Amazon Toys、Amazon Food、MovieLens‑1M 四个公开基准，工业数据为电商广告和推荐 A/B 实验。

**📈 对比分析**

对比 SASRec、BERT4Rec、GRU4Rec、HSTU、HyMiRec、TIGER、Tree‑Transformer 等基线，RedMall 在 Recall@K 和 NDCG@K 上均取得最优成绩；在线 A/B 实验中广告价值提升 2.24%、CTR 提升 0.59%，推荐 GMV 提升 1.08%、DAB 提升 0.95%。

**⚠️ 局限性**

局限性在于模型规模大、训练与推理仍需高算力；需要丰富的曝光日志和实时奖励模型；对非电商场景的泛化性待验证；同时多目标预测和曝光感知负样本依赖于历史交互记录，可能对稀疏用户不友好。

---

## 26. pfoqcompiler: Efficient Compilation of Quantum Control in Recursive First-Order Quantum Programs

**arXiv ID:** 2611.08503 | [PDF](https://arxiv.org/pdf/2611.08503v1)

**作者:** Mário Silva `[一作]` (Université Paris-Saclay), Marco Lewis `[通讯]` (Université Paris-Saclay)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5089098401)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对第一阶量子编程语言（First-Order Quantum）的一款编译器，该编译器通过利用程序结构在量子分支之间合并递归过程调用，从而显著降低量子电路的深度和规模。

**💡 创新点**

创新点在于：①引入弱化的可达性、递归二分和统一性语法限制，使更多程序可满足；②设计“锚定‑合并”编译策略，能在保持最大分支复杂度的前提下避免指数级分支序列化；③提供完整的静态验证工具和理论证明，证明弱限制与强限制等价。

**🔧 技术方法**

核心技术包括：量子if（qcase）语义与合并；静态程序分析（弱可达性、弱递归二分、宽度限制、弱统一性检测）；对程序结构的锚定与控制交换；将最终电路导出为 Qiskit/ OpenQASM。

**📊 数据集**

实验数据主要来自自定义量子程序示例（如 pairs、pears、GHZ、QFT、量子递归加法器等），并在这些程序上对比不同语言（Proto-Quipper、Qrisp、Qunity、Tower）的编译结果。

**📈 对比分析**

比较方法：将不同语言在同一程序上的电路宽度、深度与规模进行对照。结果显示，使用合并策略的编译器在满足弱统一性时，深度、宽度与规模均与经典控制相同（即为最大分支复杂度）；在弱可达性+宽度限制下，规模保持多项式；而与其他语言相比，避免了指数级规模增长。

**⚠️ 局限性**

局限性：①只能编译满足弱限制（宽度≤1、弱可达性/递归二分、弱统一性）的程序；②合并策略需要额外的锚定量子比特和控制交换，实际实现中可能增加逻辑门和量子比特开销；③静态分析与验证在更复杂程序上可能产生误报或漏报；④目前仅针对单纯递归结构，未覆盖更通用的量子控制或测量后控制。

---

## 27. Towards Label-Efficient Geospatial Foundation Model Adaptation via Positive-Unlabeled Learning for Building Damage Mapping

**arXiv ID:** 2611.08561 | [PDF](https://arxiv.org/pdf/2611.08561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 28. Embedding Consistent Recurrent Neural Networks in Finite Element Simulations for Path-Dependent Damage Prediction

**arXiv ID:** 2611.08494 | [PDF](https://arxiv.org/pdf/2611.08494v1)

**作者:** Muhammed Adil Yatkın `[一作]` (Tallinn University of Technology), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究将训练好的Consistent Recurrent Neural Network (ConsRNN) 嵌入 Abaqus Explicit 的 VUMAT 子例程，实现了在线损伤预测与元素删除。

**💡 创新点**

创新点在于提出并实现了具有离散化一致性的非标准 RNN 转移函数，并首次在结构尺度上将其硬编码嵌入显式 FEM 求解器。

**🔧 技术方法**

使用了 TensorFlow 训练 RNN，Fortran 代码实现 VUMAT，Abaqus Explicit 求解器以及 Bayesian 超参数调优。

**📊 数据集**

训练数据来自 19,098 条双线性加载路径的高精度固体单元 FE 仿真（每条路径 400 步）。

**📈 对比分析**

通过与 SimpleRNN 和传统两因子缩放 2FS 断裂准则比较，ConsRNN 在不同增量分辨率下保持一致的损伤预测，单元级预测与 2FS 对齐，整体力–位移曲线与 2FS 预峰段相同。

**⚠️ 局限性**

局限包括：训练仅覆盖双线性路径，导致在复杂非线性路径上泛化有限；损伤预测可能出现非单调下降；未对 monotonic 性做约束，需进一步研究。

---

## 29. Benchmarking Machine Learning for Chaotic Three-Body Dynamics in Astrophysics

**arXiv ID:** 2611.08634 | [PDF](https://arxiv.org/pdf/2611.08634v1)

**作者:** Sheng Wan `[一作]` (École Polytechnique), Ye Zhu `[通讯]` (École Polytechnique)

**通讯引用:** 2081 | [OpenAlex ID](https://openalex.org/A5021436608)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于临界稳定性边界的三体动力学基准数据集，并设计了层级预测任务（稳定性预测与轨道预测），随后对多种机器学习方法与传统天体物理准则进行了系统评估。

**💡 创新点**

首次提供了受控扰动生成的三体数据集，形成统一的层级任务体系；对比经典稳定性准则与预训练时间序列基础模型，系统展示了机器学习在混沌三体动力学中的可行性与局限。

**🔧 技术方法**

使用经典机器学习（XGBoost）、深度神经网络（MLP、CNN、TCN、注意力网络）以及预训练时间序列基础模型（Chronos-2、PANDA、iTransformer），并与 Mardling–Aarseth、Vynatheya 等天体物理经验准则对比。

**📊 数据集**

利用 5,000 个基准三体集合，每个产生 32 个扰动变体，合计 160,000 系统；轨迹通过 15 阶积分器计算，时间采样 5 外轨共 2,001 点，数据公开于 GitHub。

**📈 对比分析**

与经典稳定性准则相比，XGBoost 在二分类 AUROC 0.8525，Mardling–Aarseth 为 0.619；利用轨道前缀预测稳定性，XGBoost AUROC 达 0.985；多时段生存预测 Chronos-2 MAE 0.0376；轨迹预测零样本 Chronos-2 MAE 0.228。整体显示 ML 在早期动力学下可显著提升稳定性预测，但长期轨迹预测仍面临误差累积难题。

**⚠️ 局限性**

长期轨迹预测误差易累积，尤其在混沌环境下；模型对极端扰动的泛化能力有限；数据集仅覆盖临界边界，缺乏更广泛参数空间的覆盖。

---

## 30. Composition Theorem for Randomized Query Complexity

**arXiv ID:** 2611.08574 | [PDF](https://arxiv.org/pdf/2611.08574v1)

**作者:** Samruddhi Pednekar `[一作]` (Stony Brook University), Supartha Podder `[通讯]` (Stony Brook University)

**通讯引用:** 84 | [OpenAlex ID](https://openalex.org/A5022819820)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了受限错误随机查询复杂度在块组合时可乘性，并给出了上界相差对数因子，即对所有（可能是偏置的）布尔函数 f 和 g，满足 R(f∘g)=Ω(R(f)·R(g)/log(n+1))。

**💡 创新点**

核心创新是构造了一种新的势能函数 W，利用噪声查询模型和 Brier 分数的期望收益，证明了该势能在一次噪声查询后的期望增长被严格控制在 O(γ²·log n)。该技术打破了此前在随机查询复杂度组合上无法获得多项式对数误差的难题。

**🔧 技术方法**

主要技术包括：1）噪声查询模型与噪声随机决策树；2）构造带熵正则化的“预测算法”并定义势能函数 W；3）利用对数-求和展开、凸性与光滑性证明势能的二阶导数受限；4）用势能增量和终止势能的比较得到总成本下界；5）组合理论与极值分布（均衡分布）来获得平衡硬分布 μ。

**📊 数据集**

无数据集，论文完全为理论证明。

**📈 对比分析**

比较方法：将得到的下界与已知的上界 R(f∘g)=O(R(f)·R(g)·log R(f)) 对比。结果表明下界与上界相差至多一个对数因子，几乎实现了乘性组合的最佳可能性；在常数因子上仍有改进空间。

**⚠️ 局限性**

局限性：仅给出了对数因子的乘性下界；常数因子仍未确定；方法针对受限错误随机查询复杂度，未直接推广至量子或噪声对数模型；对特定类函数（如总函数）可能还有更紧的界可求。

---

## 31. How Small Can We Go? Calibrating Activation-Cache Compression for SAE Training

**arXiv ID:** 2611.08551 | [PDF](https://arxiv.org/pdf/2611.08551v1)

**作者:** Abraham Yeung `[一作]` `[通讯]` (Stanford University), Abraham Yeung (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在稀疏自编码器训练中对激活缓存进行压缩的影响，并对不同压缩方案进行系统评估。

**💡 创新点**

创新点在于提出以重新种子（reseeding）为基准的校准方法，区分保留特征的真实比例，并揭示仅靠冻结测试无法准确评估压缩效果。

**🔧 技术方法**

采用的技术包括稀疏自编码器、TopK特征抽取、不同量化（fp16、bf16、int4、rot‑intN、std‑intN等）、重采样、标准化以及冻结与重新训练两种评测方式。

**📊 数据集**

使用的数据集为 GPT‑2 small（OpenWebText 20M tokens）和 Pythia‑1.4B（Pile 10M tokens），并在各自模型层进行实验。

**📈 对比分析**

比较方法通过与同一模型不同随机种子训练的基准相对，计算保留特征比例；实验表明标准化后的 4‑bit 量化在预算内保留 88–95% 的 seed‑stable 特征，优于 Naïve int4；bf16 前向通道在速度上更快但特征保持率略低。

**⚠️ 局限性**

局限性包括仅覆盖两种模型与残差流位置、未测试更大规模或其他量化格式（如 fp8、块级量化），以及校准过程需多次训练导致成本较高。

---

## 32. Evolutionary Adaptive Metric Learning: A Multi-Objective Framework for Learning Local Distance Functions in Nearest Neighbor Classification

**arXiv ID:** 2611.08549 | [PDF](https://arxiv.org/pdf/2611.08549v1)

**作者:** Isaac Touza `[一作]` (University of Maroua), Isaac Touza `[通讯]` (University of Maroua)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5118837785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于进化算法学习局部 Mahalanobis 度量场的框架（EAML）。

**💡 创新点**

创新点在于用软门控的局部对角缩放加低秩全局项构造可微分的度量场，并通过演化优化多目标连续目标（分类、鲁棒性、光滑度、复杂度），同时给出了正定性、Lipschitz 性、鲁棒性与泛化界的理论证明。

**🔧 技术方法**

采用了精英实时遗传算法、软门控对角尺度与低秩矩阵、以及对四项目标的连续标量化组合。

**📊 数据集**

使用了 13 个公开数据集（如 Iris、Wine、Breast Cancer、Digits、Fashion‑MNIST 子集等）以及一个受控的合成区域相关特征数据集。

**📈 对比分析**

与欧氏、曼哈顿、马氏、NCA、LMNN、ITML 以及演化全局对角基准在嵌套 5×5 交叉验证下比较，整体未出现显著优势，局部模型仅在 Glass 与 Ionosphere 数据集上显著提升。

**⚠️ 局限性**

局限性包括计算开销较大、搜索预算受限、无监督锚点导致对齐问题、对噪声鲁棒性不足，以及仅使用单一标量化目标而非 Pareto 分析。

---

## 33. MeshRT: Compile-Time Governed Wafer-Scale Runtime for Low-Latency High-Throughput Inference

**arXiv ID:** 2611.08508 | [PDF](https://arxiv.org/pdf/2611.08508v1)

**作者:** Congjie He `[一作]` (University of Edinburgh), Luo Mai `[通讯]` (University of Edinburgh)

**通讯引用:** 838 | [OpenAlex ID](https://openalex.org/A5101924633)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MeshRT，利用编译时治理动态性，实现瓦片规模加速器上低延迟、高吞吐的 LLM 推理；

**💡 创新点**

将运行时动态性（连续批处理、MoE 路由、微批流水线）编译进每个核心的执行计划，并通过安全增量化调度避免全局同步；

**🔧 技术方法**

使用动态感知的核心计划合成、在芯片控制信息嵌入、GALS 安全增量化材料化，以及 Cerebras WSE‑3 2D Mesh 网络；

**📊 数据集**

在 Qwen3.5‑9B、Qwen3.5‑35B、GPT‑OSS‑20B、GPT‑OSS‑120B 等模型上，并使用 ShareGPT、OpenThoughts 等实际工作负载；

**📈 对比分析**

与 WaferLLM 对比，吞吐提升 4–56 倍；与 NVIDIA H200 GPU 对比，单机低延迟提升 4.5–7.0 倍、系统吞吐提升 1.7–13.0 倍，能效提升 1.6–12.2 倍；

**⚠️ 局限性**

仅支持单模型、固定权重、已知有限操作集；缺乏对 PE 故障自恢复、多租户共享以及动态模型加载的支持。

---

## 34. Emergent Low-Dimensional Topological Dynamics Dictate Generalization and Escape Mechanics in Billion-Parameter Neural Networks

**arXiv ID:** 2611.08502 | [PDF](https://arxiv.org/pdf/2611.08502v1)

**作者:** Dongqing Cao `[一作]` `[通讯]` (Imaginary Space Laboratory), Dongqing Cao (Imaginary Space Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了大规模深度网络在训练过程中的优化轨迹，发现参数更新被动态限制在一个低维子流形（有效维数d_eff≤12）上，并基于此提出了投影持久同调的拓扑泛化界限；进一步结合Morse理论与Betti数，解析了低维结构对鞍点逃逸的指数加速机制；基于随机Lanczos估计的主动Hessian子空间，设计了Top-SGD/Top-AdamW两种拓扑自适应优化器；在ResNet、ViT、LLM等80+架构上进行大规模实验，验证了方法在训练速度和泛化上的优势。

**💡 创新点**

创新点包括：①提出“主动执行子流形”概念，将高维训练轨迹压缩到d_eff≤12的低维空间；②用投影持久同调与Betti数证明了低维结构可提供泛化界限；③将Morse理论与鞍点逃逸结合，得到指数逃逸速率公式；④利用随机Lanczos快速估计Hessian子空间，构建了可直接包装现有优化器的拓扑自适应调度框架。

**🔧 技术方法**

技术方法包括：正则化黎曼Hessian度量、参与比(d_eff)估计、投影持久同调、随机Lanczos迭代、基于主动子空间的学习率自适应、随机Langevin动力学分析、实验对比与超参数调优。

**📊 数据集**

主要使用的数据集有ImageNet（ResNet、ViT）、C4（LLaMA-7B）以及用于验证的Synthetic损失景观；此外在实验中也涉及CIFAR-10以检验k与类别数的关系。

**📈 对比分析**

实验将Top-SGD/Top-AdamW与标准SGD、AdamW、AdaHessian等传统优化器对比；在ImageNet上实现了约14.1%的训练时间加速，ViT上约14%的wall‑clock加速，LLM-7B上约12%的加速；在固定周期训练下，Top-AdamW在下游任务（MMLU、GSM8K）表现与AdamW相当或更好。

**⚠️ 局限性**

局限性包括：需要Hessian谱呈幂律衰减，否则正则化基准失效；小批量或非高斯噪声可能违背SGLD假设；β̂_1的阈值判定是经验性启发式，可能在特殊损失景观下失效；理论泛化界限为证明性概念，实际可测性有限；且随机Lanczos估计仍存在一定计算开销，需根据任务选择频率。

---

## 35. Generative Reconstruction of Full-Field Sensor States for Safety-Critical Digital Twins

**arXiv ID:** 2611.08499 | [PDF](https://arxiv.org/pdf/2611.08499v1)

**作者:** Umme Mahbuba Nabila `[一作]` (University of Michigan), Majdi I. Radaideh `[通讯]` (University of Michigan)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5049880944)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了六种稀疏传感器重建方法在Oregon State University HTTF实验中的性能

**💡 创新点**

创新点是将三种条件生成模型（扩散模型、变分自编码器、生成对抗网络）与传统基准相对比，并在单实验与跨实验情境下系统检验其鲁棒性

**🔧 技术方法**

使用的技术包括图神经网络+门控循环单元（GNN–GRU）构建的条件生成网络，基准的岭回归、GRU和GNN–GRU

**📊 数据集**

采用HTTF的温度传感器数据：PG-29 DCC和PG-27 PCC共310个热电偶，实验中保留149或137个TS温度计进行验证

**📈 对比分析**

比较指标为MAE、RMSE和R²；在单实验中生成模型RMSE约22°C、R²≥0.97；跨实验时RMSE约43°C、R²≈0.91，均显著优于岭回归及非生成基准

**⚠️ 局限性**

主要限制是跨实验泛化仍受限，传感器数量减少会导致误差显著上升，并且模型表现受限于实验数据的完整性与缺失处理

---

## 36. MobiInfer: Accelerating On-Device GUI Agents with Workload-Aware Inference

**arXiv ID:** 2611.08496 | [PDF](https://arxiv.org/pdf/2611.08496v1)

**作者:** Siming Chen `[一作]` (Shanghai Jiao Tong University), Yubin Xia `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2713 | [OpenAlex ID](https://openalex.org/A5026023746)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MobiInfer推理引擎，在手机上加速多模态GUI代理的推理循环。

**💡 创新点**

通过可复用前缀缓存、阶段化词表裁剪和基于schema的推测解码，利用代理轨迹结构实现无需重训练的加速。

**🔧 技术方法**

采用前缀复用（KV缓存）、词表裁剪与阶段词表、基于n-gram的推测解码以及MNN运行时。

**📊 数据集**

在AndroidWorld基准和真实手机任务集（111个任务）上评测。

**📈 对比分析**

与原始MNN部署基线对比，预填速率提升2.6×、解码速率提升2.5×，端到端延迟平均缩短2.4×，任务成功率降幅<1%。

**⚠️ 局限性**

受限于对结构化输出的假设，推测策略对高变异任务收益有限，且在极端长任务中加速效果随步数增加而下降。

---

## 37. FasterRW: An Efficient Prior-Assisted Random Walk Solver for Thermal Analysis

**arXiv ID:** 2611.08490 | [PDF](https://arxiv.org/pdf/2611.08490v1)

**作者:** Zixiao Wang `[一作]` (Hong Kong University of Science and Technology), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 11172 | [OpenAlex ID](https://openalex.org/A5051340429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FasterRW，一种利用先验校正与伪样本融合的随机游走热分析加速方法；

**💡 创新点**

创新点在于：① 推导期望截尾残差同归一延续值；② 通过先验估计实现阈值选择并显式偏差控制；③ 多点查询时利用通道尾部重用伪样本，并在先验误差空间进行基尔金化后验修正；

**🔧 技术方法**

采用 Feynman–Kac 随机游走（Walk-on-Grid 与 Walk-on-Sphere）、粗网格 FEM 先验、贝叶斯尾部融合、基于 RBF 的基尔金化；

**📊 数据集**

使用 3 个三维芯片热模型（Case 1/2/3）及 3 种功率映射，全部基于公开的 IC 热仿真基准；

**📈 对比分析**

与 PIRW（传统随机游走）和 FastRW（先验+尾部融合）进行对比；在相同精度下 FasterRW 计算量提升 8.7–28×，端到端时钟速度提升超过 7.6×；

**⚠️ 局限性**

局限性包括：对粗 FEM 先验的依赖、对多点查询批量规模的依赖、仅适用于稳态热问题，且后验基尔金化在查询点过少时效果有限。

---

## 38. s$^3$: Simple Scaling of Swarms via Cooperative Multi-agent RL

**arXiv ID:** 2611.08615 | [PDF](https://arxiv.org/pdf/2611.08615v1)

**作者:** Kunal Jha `[一作]` (University of Washington), Natasha Jaques `[通讯]` (University of Washington)

**通讯引用:** 5847 | [OpenAlex ID](https://openalex.org/A5046953322)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并评估由大型语言模型构成的无角色、异步工作方式的代理集体，让代理自发决定何时单独工作、何时共享信息，从而实现高效的协作与信息聚合。

**💡 创新点**

首次在没有预设角色、轮次或通信结构的多代理强化学习框架中实现自组织协作，证明共享奖励可驱动信息聚合与探索分工，并在更大规模队伍上保持优势。

**🔧 技术方法**

使用共享策略、团队相对优势（GRPO）训练、异步消息工具、token 成本惩罚、SFT+RL 训练流程，并以 Qwen3‑0.6B/1.7B 语言模型为基础。

**📊 数据集**

设计两类任务：信息聚合（每位代理持有一个整数，需排序）和去中心化探索（在未知树形地图寻找出口），在 32 个 hold‑out 问题上评估。

**📈 对比分析**

与 best‑of‑N 采样、solo RL、individual‑reward、无通信等基线对比，s³ 在信息聚合中实现 93% 成功率，超过 baseline 4 倍；在探索中减少 50% 重复搜索，超越 best‑of‑N 9.4pp，且在更大队列中仍保持优势。

**⚠️ 局限性**

仅限两任务、小模型、单次训练，且大规模队伍仍面临信息整合难度；缺乏对消息可信度与错误处理的机制，未验证在更复杂任务或更大模型上的泛化。

---

## 39. NormGym: Evaluating Normative Competence through Social Interaction

**arXiv ID:** 2611.08613 | [PDF](https://arxiv.org/pdf/2611.08613v1)

**作者:** Binze Li `[一作]`, Tianmin Shu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了NormGym多智能体基准，用于评估代理在新社区中通过观察他人行为和公共制裁来推断并遵守隐藏的社会规范的能力。

**💡 创新点**

创新点在于构建了包含50条人工生成的“无聊规则”且只通过社交证据可学习的多种规范族群（访问信息、程序、协调、分配），并通过交互学习与评估两个阶段系统化衡量规范获取与迁移。

**🔧 技术方法**

技术手段包括ReAct式LLM代理、MOMDP建模、深度强化学习（DQN）基线、低秩适配器LoRA、REINFORCE留一法（RLOO）进行细化训练，以及基于自然语言的信念更新与判定。

**📊 数据集**

数据集为自定义的网格世界环境，共10种布局，50条随机生成的规范族群，背景代理采用脚本化A*规划，实验通过多轮交互记录状态、行为与制裁信息。

**📈 对比分析**

比较方法采用任务完成率与联合成功率（任务完成且无违规）两指标；最强LLM Claude Opus 5在已知布局的联合成功率为55.3%，在未知布局仅略降至51.3%，相比人类最高88%，远低于知情规范下的100%；RL基线与小型开源模型几乎无联合成功。

**⚠️ 局限性**

限制在于规范学习主要依赖自身违规经验，几乎不利用他人违规或合规行为；Fine‑tuning提升有限，未显著改善对社交证据的响应；环境简化为固定规范、即时公开制裁，缺乏噪声、异质行为与多重/变动规范的真实复杂性。

---

## 40. Weighted and Unweighted Online Connectivity Augmentation via Geometric Hitting Set

**arXiv ID:** 2611.08570 | [PDF](https://arxiv.org/pdf/2611.08570v1)

**作者:** Daniel Hathcock `[一作]` (GE Vernova), Michael Zlatin `[通讯]` (Pomona College)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5050694688)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了针对在线连通性增强（OCAP）问题的新算法，分别在加权和无权两种模型下实现了近最优竞争比，尤其在循环图的节点到达模型中实现了常数竞争比，展示了不同到达模型间的复杂度差异。

**💡 创新点**

创新点在于将OCAP与二维几何覆盖问题中的TL-矩形覆盖相结合，利用低浅细胞复杂度的性质实现了 O(log n log log n) 的加权竞争比，同时在无权循环图节点到达模型中首次给出了 O(1) 的常数竞争算法。

**🔧 技术方法**

技术上主要采用了在线集合覆盖的分解与重组框架：先把任意图降到循环图，再映射为TL-矩形覆盖；随后利用基于深层单元复杂度的在线抽样和二项式分割的在线舍入技术；并结合对点集合的多重复制与时间维度扩展，构造了适用于底层矩形的高效随机化舍入方案。

**📊 数据集**

论文未使用真实数据集，而是通过理论构造的图与矩形实例进行实验和证明，主要关注理论竞争比与最优下界的对比。

**📈 对比分析**

通过与传统 O(log²n) 的在线集覆盖算法、已知的 O(log n) 节点到达模型算法以及线性规划下界进行比较，证明加权 OCAP 的竞争比为 O(log n log log n)，无权循环节点到达模型为常数竞争，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括加权模型的竞争比仍未达到理论下界的 O(log n)，且在非循环图的无权节点到达模型中未给出改进；此外，TL-矩形覆盖的转化与抽样方法复杂，难以直接推广到更一般的几何覆盖问题。

---

## 41. Not All Misalignment Is Alike: A Decomposed Benchmark for Emergent Misalignment

**arXiv ID:** 2611.08569 | [PDF](https://arxiv.org/pdf/2611.08569v1)

**作者:** Afonso Domingues `[一作]` (ETH Zurich), Anna Hedström `[通讯]` (ETH Zurich)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5100279521)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 EMERGE 基准，针对 8 种误导行为，在 8 个开源模型与 13 个微调数据集上进行基线与微调模型的对比评估。

**💡 创新点**

创新点在于将误导行为拆解为 8 个可操作子任务，采用域过滤与匹配善意控制，构建可复现的多维度 EM 评估框架，取代单一错误率的评估方式。

**🔧 技术方法**

技术手段包括 LoRA 微调、双评审（Llama‑3.1‑70B 与自托管 Llama‑3.1‑70B）对响应进行分数与拒绝判定、头部余量归一化 Δ 计算、95% 置信区间 Bootstrap、以及 AI2‑ARC、GSM8K、IFEval 的能力测评。

**📊 数据集**

使用的数据集包括 8 个公开权重模型、13 个诱发 EM 的微调数据集以及 3,202 条由现有基准汇总的评估条目，覆盖八类误导行为。

**📈 对比分析**

评估方法通过基线与微调模型在相同域外提示下的误导率差异计算 Δ，并取平均得到 Δ̅ 作为整体易感度排序；实验表明模型易感度差异很大，与模型规模无明显关联，单一错误率虽与 Δ̅ 相关但掩盖了行为层面的差异。

**⚠️ 局限性**

局限性包括依赖 LLM 评审可能产生偏差、每个模型-数据集组合仅使用单一微调随机种子、生成样本数有限、域分类器误判导致域外过滤不精确、能力下降可能混淆误导变化、以及未对预训练阶段的污染进行控制。

---

## 42. CascadeRec: Deployed Adaptive LLM Compute Allocation for Multi-Tenant Content Recommendation

**arXiv ID:** 2611.08566 | [PDF](https://arxiv.org/pdf/2611.08566v1)

**作者:** Kai Zhao `[一作]` (Workday AI Research), Alet Blanken `[通讯]` (Workday AI Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了CascadeRec，一个两层LLM重排序级联，用于企业级推荐系统；

**💡 创新点**

其创新点在于：①基于请求级别的自适应计算分配与租户负载均衡；②分段门控上线策略与可调router；③使用轻量化Tier‑1与高成本Tier‑2（自一致性、多路径推理或更大模型）并行；④通过缓存Tier‑1输出实现实验可复现与成本控制；

**🔧 技术方法**

技术方案包括：LLM零射重排序、self‑consistency、multi‑path reasoning、Amazon Bedrock与vLLM服务、Kubernetes（EKS/GKE）部署、阈值路由器、解析容错与回退机制、token与延迟监控；

**📊 数据集**

使用公开数据集MovieLens‑1M与Amazon‑Electronics进行离线评估，并在真实多租户FP&A SaaS平台中进行为期一个多月的A/B实验；

**📈 对比分析**

与统一重排序基线对比：在生产环境中，CascadeRec实现约10%推理成本下降，延迟影响极小且质量无退化；离线实验显示，统一重排序对稀疏历史用户会导致NDCG@10下降30%，但对长历史用户略有提升；级联通过减少40–50% Tier‑2 token消耗降低了整体成本；

**⚠️ 局限性**

局限性包括：①未在离线实验中出现整体质量提升，主要限于减少负面影响；②仅使用零射LLM，缺乏细粒度模型微调与多模态输入；③不同模型与域的解析失败率差异大，需要额外的prompt设计与监控；④公平性问题仍存在，需进一步细分用户段落与评价。

---

## 43. Seeing Is Not Knowing: The Epistemic Limits of LLM-Based Mental Health Inference from Youth Social Media Conversations

**arXiv ID:** 2611.08565 | [PDF](https://arxiv.org/pdf/2611.08565v1)

**作者:** Xiaohan Ding `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 17409 | [OpenAlex ID](https://openalex.org/A5102962995)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大规模语言模型（LLM）在基于青少年Instagram私聊记录推断心理健康与社会支持等心理状态的可行性和限制；

**💡 创新点**

阐明观察到的对话证据与主观心理体验之间的知识鸿沟，区分证据获取与证据解释两大问题；

**🔧 技术方法**

使用10种本地部署、支持链式推理与检索增强生成（RAG）的开源LLM，并结合提示工程与模型生成的推理轨迹；

**📊 数据集**

由80名患有抑郁或自杀风险的青少年提供的Instagram Direct Message存档（约240万条信息）与其完成的8项心理测评问卷；

**📈 对比分析**

在项级和构成级对预测准确度进行多指标评估（AHR、ACR、ADODL、CHR等），发现对可观察事件的推断准确度较高，使用检索提示能提升整体表现；

**⚠️ 局限性**

模型无法从对话中获取或可靠推断主观体验（如感知支持、睡眠干扰、使用成瘾感），仅靠对话证据无法完整重现自评结果，存在信息缺失与推断误差的局限。

---

## 44. Subliminal Learning in Vision-Language Models

**arXiv ID:** 2611.08538 | [PDF](https://arxiv.org/pdf/2611.08538v1)

**作者:** Arman Hatami `[一作]` (Johns Hopkins University), Ilya E. Monosov `[通讯]` (Johns Hopkins University)

**通讯引用:** 2377 | [OpenAlex ID](https://openalex.org/A5024947116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉-语言模型(VLM)中验证了“潜意识学习”，即学生模型在仅接收与目标偏好无关的数值标签训练时，仍能迁移教师模型的动物和面部属性偏好。

**💡 创新点**

首次证明潜意识学习在多模态模型中同样存在，并显示其效果可随携带数据量、模型规模和解码方式而变化；同时揭示偏好迁移还能影响模型的输出词汇。

**🔧 技术方法**

采用基于教师-学生的对照实验框架，使用 Qwen2-VL、Qwen2.5-VL 与 Phi‑4‑multimodal-Instruct 等预训练 VLM，训练教师模型偏好并产生数值序列作为携带数据，再用这些数据微调学生模型。

**📊 数据集**

使用 Animals 数据集（6 种动物）和 FaceSynthetics 数据集（5 种面部属性）构造偏好任务，并通过合成几何形状图像生成数值携带任务。

**📈 对比分析**

通过比较学生模型的目标选择率和控制学生的差值 Δ_T 评估偏好迁移；结果显示在 Qwen2.5-VL-3B/7B 上，Δ_T 可达约 10%（如 panda +9.92%），并随携带样本数量增大而提升。

**⚠️ 局限性**

局限性包括：迁移效果依赖于具体目标、模型规模与解码方式，部分属性迁移效果弱且不稳定；实验未深入探讨迁移机制的底层原因，且未评估对安全性与偏见治理的长期影响。

---

## 45. MorphSched: Workload-Aware Adaptive Scheduler Generation for Linux

**arXiv ID:** 2611.08509 | [PDF](https://arxiv.org/pdf/2611.08509v1)

**作者:** Yifan Zhang `[一作]` (Zhejiang University), Jianwei Yin `[通讯]` (Zhejiang University)

**通讯引用:** 8895 | [OpenAlex ID](https://openalex.org/A5069353502)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套基于大语言模型和LLM驱动的工作负载感知调度器生成框架MorphSched，能够自动生成完整的Rust+BPF Linux调度器。

**💡 创新点**

创新点包括：①使用工作负载在EEVDF下的基线动态目标，将调度目标与实际性能紧密耦合；②引入契约驱动的递进式生成流程，先产生语言无关IR，再按严格契约并行生成用户空间和内核空间代码；③采用双层历史记忆搜索（演化内存池 + DAG修复图），实现实现级别修复与策略级别搜索的解耦。

**🔧 技术方法**

技术手段包括：大语言模型（如GPT-5.5、Qwen-3.7-max、Gemini-3.1-pro）进行代码生成；结构化的BPF API数据库；eBPF编译器、Linux内核验证器；自定义的IR、契约文件；演化记忆池与DAG修复框架；多轮编译、验证与性能回调。

**📊 数据集**

使用的数据集是120个不同的机器/工作负载/干扰组合（4~16个CPU虚拟机、i9-14900物理机），前台工作负载包括Redis、Nginx、memcached（p99延迟）和Sysbench CPU（吞吐量），后台干扰为CPU/缓存/内存压力、压缩和内核编译。

**📈 对比分析**

与基线EEVDF以及五个手写上游调度器（如LTS、Fair、PFC等）进行对比；在120个配置上，MorphSched平均提升26.5%，单例最大提升87.3%；在部分工作负载上与手写调度器竞争甚至优于它们；实验使用真实Linux内核进行编译、加载、跑测，评估指标为前台p99延迟或吞吐量。

**⚠️ 局限性**

局限性包括：仅在单前台服务+单干扰场景验证；对LLM后端和内核API的依赖导致迁移成本；生成的调度器虽然通过eBPF验证，但缺乏正式形式化正确性证明；多租户、NUMA、I/O密集型工作负载等更复杂场景尚未深入探索。

---

## 46. Relic and Gumi: Governing Memory, Safety, and Continuity in Diegetic Relational Agents

**arXiv ID:** 2611.08504 | [PDF](https://arxiv.org/pdf/2611.08504v1)

**作者:** Daniele Veri' `[一作]` `[通讯]`, Daniele Veri'

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套名为Relic的治理架构，用以控制和审计在 diegetic 关系代理（如 Gumi）中的长期记忆与安全性，确保代理记忆、可解释性和可纠正性；

**💡 创新点**

创新点在于：1）将记忆、治理、安全信号与代理行为分离为可审计的对象；2）引入“共享连续性记忆”仅以受试者确认的语言存储，并提供纠正、暂停和删除机制；3）通过运行时中介层（Hermes）实现安全约束、输出变换和交付门控；4）提供研究者工作台与评估框架，实现对安全信号、行为约束与记忆的可视化和可 contestability。

**🔧 技术方法**

技术实现包括：Hermes 运行时中介、SQLite（可选）存储共享连续性记忆、Cron/计划任务、平台 allowlist、输出转换与审计日志（Chronicle）、安全信号与行为补丁的治理模型、以及基于 LLM 的对话代理（使用 llama.cpp）。

**📊 数据集**

实验数据仅包含两名参与者的两周试点，未使用公开数据集；此外构建了 180 个合成场景用于控制实验基准。

**📈 对比分析**

方法上主要是构建治理框架和评估规范；通过合成基准验证安全信号不被误记忆、行为约束正确裁剪等；试点中收集自然度评分（P1 4.25/5，P2 3/5），但未进行大规模性能或用户体验对比，缺乏定量性能指标。

**⚠️ 局限性**

局限性包括：1）样本极小且无独立伦理评审；2）实现仅部分路径已接入治理层，存在未覆盖的执行分支；3）未提供加密、访问控制、保留策略等安全细节；4）缺乏真正的用户研究和可 contestability UI；5）不能作为临床或医疗决策工具。

---

## 47. HaNDF: Object-Conditioned Geometric Neural Hand Distance Fields

**arXiv ID:** 2611.08578 | [PDF](https://arxiv.org/pdf/2611.08578v1)

**作者:** Zuhao Liu `[一作]` (Imperial College London), Tolga Birdal `[通讯]` (Imperial College London)

**通讯引用:** 3776 | [OpenAlex ID](https://openalex.org/A5038619214)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 HaNDF，一种在 Riemannian 关节四元数流形上学习的对象条件神经距离场，用于手-物体交互的手势生成、去噪、补全与重建。

**💡 创新点**

创新点包括：① 在手关节的 Riemannian 流形上构建对象条件的距离场；② 用对象隐码全局调节零水平集，形成可行手势子流形；③ 通过可微梯度投影实现手势优化；④ 将生成、去噪、补全统一为同一框架；⑤ 引入逆向对象恢复，闭环提升手势与物体的相容性。

**🔧 技术方法**

技术手段：Riemannian 几何 + NRDF；Transformer 编码手关节四元数与对象隐码；DeepSDF auto‑decoder 作为对象编码器；梯度下降 / Riemannian Adam 进行投影与优化；对抗式/重构式损失配合噪声采样进行训练。

**📊 数据集**

使用的数据集包括：HOI4D（视角相机+手-物体标注）、OakInk（多视角 RGB）、DexYCB（YCB 对象抓取序列）以及相应的物体网格与隐码。

**📈 对比分析**

与 H-PoseNDF、H-NRDF、HMP、G-HOP、LatentHOI、Text2HOI 等基线在生成、去噪、补全、重建四类任务上比较。HaNDF 在生成任务中 FID 0.54、dNN 4.77°、APD 12.73° 领先；在去噪任务中 Δq、m2m 最高；在补全任务中 Δq、dNN、FID 皆最优；在手-物体重建中 AlignSDF+HaNDF 将 MPVPE 降至 11.30 mm、MPJPE 11.45 mm、I.V. 0.1083 cm³，整体性能显著优于所有基线。

**⚠️ 局限性**

局限性：① 依赖准确的物体几何；② 不是完整的抓握先验，未建手-物体联合距离场或接触动力学；③ 仅处理单手交互，未覆盖双手场景；④ 在极端遮挡或未知物体上鲁棒性待验证；⑤ 训练与推理对计算资源有一定需求。

---

## 48. Remembering Is Not Enough: Evaluating Timing and Omission in Longitudinal AI Mental Health Support

**arXiv ID:** 2611.08721 | [PDF](https://arxiv.org/pdf/2611.08721v1)

**作者:** Eli Min `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 17409 | [OpenAlex ID](https://openalex.org/A5102962995)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了基于22名真实用户聊天历史的长期模拟，对在聊天过程中注入的三类危机（自杀意念、自伤、焦虑）进行评估，提出了CARE框架衡量支持的时间匹配与遗漏情况。

**💡 创新点**

①首次将长期交互与真实用户数据相结合的模拟方法用于AI心理健康支持评估；②设计CARE框架，系统捕捉时间对齐（AR/PR/LR）和遗漏（OmitR）等维度，补足传统单轮评估的不足；③通过回调（Callback）与基线（Baseline）对比揭示及时介入对建议时机的影响。

**🔧 技术方法**

使用两大大型语言模型（Qwen 3‑8B 生成用户语料，Gemma 4‑31B‑Instruct 生成助手响应），自研的记忆管理方案（按主题组织事实、递归更新、允许名单过滤），以及基于临床专业人士定义的标签体系进行标注。

**📊 数据集**

22位用户提供的至少6个月（中位364会话）聊天日志，拆解为阶段（stage）后构建1,584人设，生成14,256条模拟会话（3种危机×3种模式）。

**📈 对比分析**

与无危机对照段落、不同危机类型以及不同用户恢复轨迹（Recovery、Stall、Ambivalent、Decoy）进行匹配；使用Wilcoxon、Friedman及线性混合模型等统计检验。结果显示：回调模式将不合时宜建议率从0.208降至0.149，AR从0.783提升到0.806；但遗漏率在焦虑场景仍高达0.827，且大部分会话缺失后续跟进与下一步规划。

**⚠️ 局限性**

①模拟数据无法覆盖全部人类情绪多样性，样本仅为22人；②缺乏真实用户安全与福祉评估；③标注依赖LLM评估与有限人工校验，难以保证每条标注的完全准确；④未探讨模型对不同性别、种族或文化背景的差异。

---

## 49. Memory-Aware Multi-Agent Architecture with LLM-based Orchestration for Conversational Recommendation

**arXiv ID:** 2611.08718 | [PDF](https://arxiv.org/pdf/2611.08718v1)

**作者:** Emilly Pitman de Castro `[一作]` (Instituto de Ciência e Tecnologia Itaú), Wandemberg Gibaut `[通讯]` (Instituto de Pesquisas Eldorado)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5112885509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个内存感知的多智能体架构，用LLM作为监督器协调意图精炼、查询构造、检索和重排序，实现了跨会话记忆与对话流的集成。

**💡 创新点**

创新点包括：1）将长期记忆作为一等组件，直接影响路由决策和语义查询；2）LLM监督器主动引导对话和信息流，提升交互质量；3) 引入Semantic Consistency Score（SCS）作为语义一致性评估指标；4) 通过跨会话记忆与对话调度显著提升接受率和语义对齐。

**🔧 技术方法**

采用LangGraph构建多智能体协作；LLM（如Claude/ChatGPT）做监督器和重排序器；语义检索使用Sentence‑BERT + FAISS + MMR；协同过滤采用ALS；TF‑IDF提供词法信号；长程记忆节点结构实现增量存储与检索。

**📊 数据集**

使用Amazon Reviews 2023 Electronics子集（约1,000活跃用户，19,678商品，54,521交互），包含文本评论、元数据和交互历史。

**📈 对比分析**

评估方法：离线评估（Hit@10、NDCG@10、ILS@10、SCS），多轮对话实验（累计Hit@T、MRR），监督器对照（Baseline、Conversational、Conv+Memory）以及跨会话记忆对照。结果显示：1）SCS从0.930提升至0.954；2）在对话层面，接受率从37.6%提升至66.5%（+28.9pp）；3）跨会话记忆在57%案例中优于无记忆条件；4）Hit@K略有下降但语义一致性和多样性显著提高。

**⚠️ 局限性**

局限性：1）实验使用的是精简的亚马逊子集，规模有限，缺乏真实用户多样性；2）交互为模拟，可能无法完全捕捉真实意图演变；3）记忆效果高度依赖偏好稳定性，易受过时信息干扰；4）LLM重排序和多智能体协作增加计算成本，未在大规模环境下验证；5）SCS作为补充指标，仍需与传统评估相结合。

---

## 50. Efficient Spintronic Stochastic In-Sensor Computing Modules

**arXiv ID:** 2611.08715 | [PDF](https://arxiv.org/pdf/2611.08715v1)

**作者:** Abdolah Amirany `[一作]` (University of Florida), Tarek El-Ghazawi `[通讯]` (George Washington University)

**通讯引用:** 2800 | [OpenAlex ID](https://openalex.org/A5001914825)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一种基于磁隧道结（MTJ）亚临界开关的自旋随机数发生器（SNG），并将其嵌入到感测器内的边缘检测电路中，完成了从随机数生成到图像边缘提取的完整流程。

**💡 创新点**

创新点包括：①利用MTJ在亚临界电流下的随机开关特性实现概率可调的随机位流；②将此SNG直接集成进预充式感测放大器（PCSA）中，形成低功耗、低面积的完整模块；③设计了针对随机域的改进Sobel/Prewitt掩模，兼具水平、垂直与全方向边缘检测能力；④在同等E-PSNR、E-SSIM条件下，实现了至少62%面积缩减、60%能耗下降的显著提升。

**🔧 技术方法**

主要技术包括：自旋电子学（磁隧道结）、亚临界电流随机开关、预充式感测放大器、随机域卷积（改进的Sobel/Prewitt掩模）、HSPICE仿真、MATLAB图像处理与指标计算（E-PSNR、E-SSIM）。

**📊 数据集**

实验使用的图像数据集未明确给出，但通过MATLAB对典型测试图像进行边缘检测，并与Canny边缘检测器（视为地面真值）进行对比。

**📈 对比分析**

通过与Canny、Sobel、Prewitt以及之前的随机化边缘检测器（文献中的方法）进行能耗、面积、E-PSNR、E-SSIM四维度比较，结果显示：在相同或更低的能耗（降低82%）和面积（降低74%）的前提下，E-PSNR、E-SSIM与传统方法持平甚至略优；随机位流长度可调，32/64/128/256位均能获得良好视觉效果。

**⚠️ 局限性**

局限性包括：①随机开关的概率控制需要精确的电流调制，易受工艺/温度漂移影响；②实验仅针对单一边缘检测任务，缺乏对更复杂图像处理任务的验证；③在极低位流长度（32位）下，输出噪声仍较大，需进一步优化掩模设计；④尚未在真正的硬件感测器平台上进行系统级验证。

---

## 51. Patching the Blind Spots: Reversible LLM Rule Overlay for Frozen Fraud Detectors

**arXiv ID:** 2611.08714 | [PDF](https://arxiv.org/pdf/2611.08714v1)

**作者:** Bo Qu `[一作]` (eBay), Ming Ouyang `[通讯]` (eBay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用大语言模型生成的可解释规则，对已部署的梯度提升树模型进行可逆的加权覆盖，修补其对新型欺诈模式的盲点；

**💡 创新点**

创新点在于：①利用SHAP低权重特征引导LLM提出规则；②将规则转化为对原模型分数的可逆log‑lift覆盖；③通过多种LLM架构的集成和对不同阈值带的分段校准，降低单一模型高方差；

**🔧 技术方法**

技术包括：SHAP特征重要性分析、LLM（CoT、Sliding‑Window、Mixture‑of‑Experts）规则生成、经验支持筛选、分段log‑lift校准、logit空间均匀平均、阈值映射表实现覆盖；

**📊 数据集**

实验使用两组数据集：1）D‑Industrial——真实电商交易数据，训练集为批准流量，评估集为无偏样本；2）BAF——公开银行账户欺诈基准，包含七个设计的偏移变体；

**📈 对比分析**

与继续训练（CT‑100、CT‑Tuned）、完全重新训练、以及基准模型对比。D‑Industrial 上，覆盖在等精度条件下召回提升约+2.26%，等召回条件下精度提升约+1.60%，持续训练反而退化；BAF 上覆盖不降低性能，保持近中性或略有提升，持续训练或重新训练在某些变体下表现不佳；

**⚠️ 局限性**

局限性包括：仅在离线历史评估中验证，未进行线上A/B测试；持续训练基准未覆盖全部超参空间，未尝试重要性加权训练；覆盖规则生成需要多模型推理，推理成本较高；适用于无标签数据迁移但对训练样本选择偏差仍有限；

---

## 52. An Educated Guess: Deriving Statistically Aligned Gradient Estimates for Zeroth-Order Optimisation

**arXiv ID:** 2611.08712 | [PDF](https://arxiv.org/pdf/2611.08712v1)

**作者:** Justus Huebotter `[一作]`, Nasir Ahmad `[通讯]` (Radboud University)

**通讯引用:** 3137 | [OpenAlex ID](https://openalex.org/A5053352592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Spectral Zeroth-Order（SZO-κ）方法，在前向传播时利用激活二阶矩来形成功能导向的权重扰动，从而实现无反向传播的训练；

**💡 创新点**

将前向激活的协方差作为搜索方向先验，构造一族可调谱的权重扰动，统一并推广了传统的权重扰动、节点扰动及其混合方法；

**🔧 技术方法**

采用零阶优化、随机微分、特征协方差矩阵谱分解、梯度猜测与预处理技术，并在训练中使用批量/样本级反馈与迭代层wise扰动；

**📊 数据集**

使用Fashion-MNIST（两层MLP）和CIFAR-10（六层CNN）两种公开数据集进行实验；

**📈 对比分析**

与BP、NP、WP、NTWP等方法对比，SZO-κ在梯度对齐和测试准确率上普遍优于传统权重扰动，在CNN上进一步超过节点扰动；

**⚠️ 局限性**

需要显式计算协方差矩阵导致额外计算与内存成本；跨层扰动产生crosstalk；当κ趋于∞或较大时趋向单一方向，需通过迭代层wise或预处理来缓解。

---

## 53. RL-ATTACK: A Generalized Black-Box Adversarial Attack on Language Models via Reinforcement Learning

**arXiv ID:** 2611.08711 | [PDF](https://arxiv.org/pdf/2611.08711v1)

**作者:** Taikun Chen `[一作]` (University of Delaware), Hui Fang `[通讯]` (University of Delaware)

**通讯引用:** 3189 | [OpenAlex ID](https://openalex.org/A5101716412)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于强化学习的可训练黑盒对抗攻击框架 RL-ATTACK，能够针对特定目标模型学习生成攻击性文本。

**💡 创新点**

创新点在于：①将对抗攻击视为策略学习问题，使攻击模型能够从目标模型的反馈中自适应学习；②设计统一的框架，既适用于文本分类器也适用于生成式大语言模型；③在奖励设计上兼顾攻击成功率与语义保持，提升攻击效果与自然度。

**🔧 技术方法**

主要技术包括：强化学习（Policy Gradient/REINFORCE），基于 BERT 的掩码语言模型结构，奖励函数由置信度下降和 USE（Universal Sentence Encoder）语义相似度组成，训练过程中使用自适应查询与候选替换策略。

**📊 数据集**

使用的数据集有：NLI 任务中的 SNLI、MNLI（matched 与 mismatched），生成式任务中的 in-the-wild jailbreak-prompt 语料库与 AdvBench；目标模型涵盖 BERT、Llama-Guard-4‑12B、Llama 2‑7B、GPT‑3.5‑Turbo、GPT‑5.4‑mini，评估时还使用 OpenAI Moderation API。

**📈 对比分析**

与基线（TextFooler、BERT‑ATTACK、SSPAttack、HQA‑Attack、TextHoaxer、LeapAttack、PAIR 等）比较，RL‑ATTACK 在 NLI 上将攻击后准确率压至 2% 左右，查询量仅 13–16 次，语义相似度（USE）保持在 0.65–0.70 以上；在生成式攻击中，攻击后准确率普遍低于 PAIR，查询次数更少，USE 分数显著更高，表明既具高攻击效能又保持语义质量。

**⚠️ 局限性**

局限性包括：评估集规模有限，未对生成文本的语法正确性进行约束；奖励与超参数调优无统一最优方案；对极大规模或商业模型（如 GPT‑5、Claude、Gemini）的适用性尚待验证。

---

## 54. Levenshtein-type bounds for $(a,b)$-avoiding spherical codes

**arXiv ID:** 2611.08709 | [PDF](https://arxiv.org/pdf/2611.08709v1)

**作者:** Peter Boyvalenkov `[一作]`, Peter Grabner `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在球面上满足内积避免区间$(a,b)$的最大码数和最小能量问题，提出Levenshtein型线性规划框架并给出通用下界。

**💡 创新点**

创新点在于：
1) 将正定有符号测度和Krein条件推广到两端不连续的可接受内积集合；
2) 构造对应的Christoffel–Darboux正交多项式，得到精确的自由节点高阶高斯型高斯积分；
3) 通过Hermite插值与Newton正序保证能量下界的正定性；
4) 引入测试函数给出全阶线性规划最优性的必要与充分条件。

**🔧 技术方法**

主要技术：
- 正定有符号测度与Gram矩阵判定；
- Christoffel–Darboux核与极点定位；
- 线性规划极限与Levenshtein多项式构造；
- Hermite插值、Newton展开与正序判定；
- 测试函数与Krein条件的矩阵化描述。

**📊 数据集**

无实验数据集；论文完全以理论证明和数值例子（如$n=23$, $k=1,2$与$n=22$, $k=0$）为验证。

**📈 对比分析**

与已有的Levenshtein、Cohn–Kumar等界比较，证明在相同参数下可获得相同或更强的球面码数上界与能量下界；通过具体例子展示界的锐度和设计的存在性。

**⚠️ 局限性**

局限性：
- 需要先验证正定有符号测度与节点位置，条件较繁琐；
- 目前仅给出充分条件，未完成所有参数的完整分类；
- 对偶性与更高阶极点的完整结构仍待研究。

---

## 55. VICTR: Value-Informed In-Context Learning for Vision-Language-Action Models

**arXiv ID:** 2611.08698 | [PDF](https://arxiv.org/pdf/2611.08698v1)

**作者:** Luca Macesanu `[一作]` (New York University), Juexiao Zhang `[通讯]` (New York University)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5087776555)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于进度估计的示例检索框架VICTR，结合In‑Context Value Function Estimator (IC‑VFE) 对视觉‑语言‑动作(VLA)模型进行检索式自适应，使模型能够在未见任务上无需额外微调即可根据演示视频的进度对齐检索并生成动作。

**💡 创新点**

核心创新在于：①利用参考演示视频作为上下文，训练一个进度估计器IC‑VFE，能够在不知任务细节的情况下判断当前观测在演示过程中的进度；②将该进度作为检索信号替代传统视觉相似度，从而在不同执行阶段对齐示例；③在检索过程中仅使用固定模型权重（post‑training），实现真正的推理时检索学习。

**🔧 技术方法**

技术实现包括：使用π_0.5 VLA基模型与Gemma 3 270M视觉语言解码器；DINOv2视觉嵌入做视觉特征；FAST分块编码动作；IC‑VFE采用分布式进度估计（201个离散bin）和block‑causal注意力；训练时利用Robo‑Dopamine奖励模型生成进度伪标签；post‑training检索式ICL框架。

**📊 数据集**

实验使用自行收集的真实机器人数据集：28个任务共1,180条轨迹（7小时训练+30分钟演示），其中包含8个在域任务与8个离域任务；演示池为10条演示，按执行长度划分并用于进度估计与检索。

**📈 对比分析**

与基线fine‑tuned π_0.5、视觉相似度检索的RICL-π_0.5和VICTR进行对比：在离域任务上，VICTR将成功率从6.2%提升至23.8%，任务进度率从28.9%提升至55.5%；在域任务上进度率提升至约90%并保持高成功率。检索信号匹配训练与测试的匹配对性能有显著影响。

**⚠️ 局限性**

局限性包括：即使提升明显，离域成功率仍低（约25%）；实验仅在单一物理环境中进行，未覆盖真正新环境；IC‑VFE的上下文长度限制导致对细粒度视觉细节的捕捉不足；需更多高质量演示与更高效的上下文编码；框架依赖固定模型，迁移到其他VLA体系的效果尚未验证。

---

## 56. Cancellation Is Not Enough: What Retries Leave Behind in an Overloaded LLM Engine

**arXiv ID:** 2611.08684 | [PDF](https://arxiv.org/pdf/2611.08684v1)

**作者:** Xindan Zhang `[一作]` (University of Toronto), Fei Wang `[通讯]` (University of British Columbia Okanagan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在LLM引擎过载情况下，客户端重试留下的残余工作，并探讨如何通过取消、重试预算和请求合并等控制方法恢复服务。

**💡 创新点**

创新点在于首次将客户端重试与引擎内部执行的关联进行跟踪，量化取消传递后残留的前缀处理工作，并给出针对不同控制策略的性能比较与启发式建议。

**🔧 技术方法**

使用vLLM 0.30.0推理框架、Python异步代理、日志追踪与计数器，结合前缀缓存与异步调度等技术。

**📊 数据集**

实验采用三款开源模型（Qwen2.5-7B-Instruct、Qwen3-14B、Llama-3.1-8B-Instruct）在单张H100 GPU上生成的合成单轮对话提示。

**📈 对比分析**

通过对比不同重试预算、请求合并、取消策略、队列上限等配置，在人工生成的负载脉冲后观察服务恢复时间与调用交付率，发现合并策略能在保持所有请求交付的同时恢复服务，而仅靠预算会留下部分未交付。

**⚠️ 局限性**

局限在于仅测试单机vLLM 0.30.0、单GPU、固定重试/超时参数、合成负载，未覆盖多机负载均衡、tensor并行模型、真实交互式客户端及更大模型。

---

## 57. A Complete Calculus for Hamiltonian Quantum Programming

**arXiv ID:** 2611.08675 | [PDF](https://arxiv.org/pdf/2611.08675v1)

**作者:** Jacques Carette `[一作]` (McMaster University), Amr Sabry `[通讯]` (Indiana University)

**通讯引用:** 3269 | [OpenAlex ID](https://openalex.org/A5084476883)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向哈密顿量的过程代数，支持并行求和、半切割、Lie‑Trotter语义，将哈密顿量程序编译成可执行电路

**💡 创新点**

构建了完整可判定的等式理论，证明其在二进制幂次的Pauli系数下可表达所有可数理Hamiltonian，并实现了完备抽象与等价判定

**🔧 技术方法**

使用过程代数、Π_k 计算器、Lie‑Trotter 分解、Haskell/NumPy/Qiskit 编译器以及 Agda 形式化证明

**📊 数据集**

以氢键（DMANH⁺）的四维 Hamiltonian 为例，对其进行离散化并在程序中构造，验证频率计算

**📈 对比分析**

通过与实验测得的振动频率、纯电路编译误差（Frobenius 距离）以及不同 Trotter 层级的误差收敛进行比较，误差随层级指数下降，达到实验误差阈值

**⚠️ 局限性**

仅在小规模 Hamiltonian 上验证，电路规模仍较大，缺乏高级分裂与量子奇点化优化，未与现有电子结构/量子化学工作流集成

---

## 58. Optimal Contention Resolution from Samples

**arXiv ID:** 2611.08673 | [PDF](https://arxiv.org/pdf/2611.08673v1)

**作者:** Shaddin Dughmi `[一作]` (University of Southern California), Pranav Nuti `[通讯]` (University of Chicago)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5013100239)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在未知激活概率的前提下，针对k-均匀矩阵（即基数约束）设计并分析样本驱动的在线争议解析方案（OCRS 与 ROCRS）。

**💡 创新点**

提出了一系列最优或近似最优的样本复杂度结果：单个样本即可在随机顺序下获得最优平衡；单个样本在对抗顺序下不够，需至少两个样本才能实现最优1/2平衡；对于k≥5的随机顺序，甚至不需要样本即可达到最优；而对抗顺序下无论k≥2，任何固定数量样本都无法达到最优平衡。

**🔧 技术方法**

利用概率分析、Poisson 近似、凸组合、Bernoulli 赛跑（Bernoulli race）思想、离散化与线性规划等技术构造与证明争议解析方案，并用最优性与极限分析证明样本上界。

**📊 数据集**

无数据集，全部为理论证明与数学推导。

**📈 对比分析**

通过与已知激活概率下的最优平衡率对比，展示了样本数量与性能的对应关系；在随机顺序下，单样本方案与最优平衡率相同；在对抗顺序下，单样本方案只能得到0.48平衡，需两样本才能达到1/2平衡；对于大k，单样本可得到1-1/√k平衡，接近已知分布下的最优。

**⚠️ 局限性**

主要限制在于：对抗顺序下的样本复杂度存在本质不匹配——对k≥2时，即使样本数任意增大，也无法达到最优平衡；此外，提出的方案在实现上仍需对激活概率做一定假设（如总和≤k），且部分证明依赖于极限与泊松逼近，可能在极端参数下不严谨。

---

## 59. ChainFlow: Few-Step Latent Flow Matching for Speculative Decoding

**arXiv ID:** 2611.08669 | [PDF](https://arxiv.org/pdf/2611.08669v1)

**作者:** Selim Aktas `[一作]` (Koç University), Mehmet Fatih Amasyali `[通讯]` (Yıldız Technical University)

**通讯引用:** 1311 | [OpenAlex ID](https://openalex.org/A5080453744)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于流匹配的少步潜在流匹配推理方法 ChainFlow，用于在 LLM 生成时进行推测式解码；

**💡 创新点**

创新点在于把草稿器视为对目标模型隐藏状态空间的短时交通，利用 VAE 压缩隐藏状态并在此空间上一次 Euler 步完成流匹配，同时通过低秩马尔可夫头和路径反馈恢复多词依赖，支持树形草稿，显著提升速度与可扩展性；

**🔧 技术方法**

核心技术包括：VAE 编码隐藏状态、流匹配（flow matching）在潜在空间中训练速度场、一次 Euler 步执行、低秩马尔可夫头和路径反馈、基于树的草稿与检验、短词表筛选与两步候选头；

**📊 数据集**

使用 Qwen3.5 系列模型（4B、9B、27B）为目标，在英语、土耳其语、德语、法语等多语言数据上进行训练与自适应，训练数据涵盖数学推理、代码、问答、工具调用、翻译等八大领域；

**📈 对比分析**

与同等规模的 EAGLE-3、DFlash 进行对比，单词级速度提升分别为 1.39×（对 EAGLE-3）和 1.11×（对 DFlash），在所有领域和模型规模上均优于基线；树形草稿进一步提高接受长度和吞吐率，单步流匹配在 27B 规模下可实现最高 2.12× 的聚合速度提升；

**⚠️ 局限性**

局限性包括：在高并发场景下速度提升下降；树形草稿需要修改引擎支持并只能使用贪心解码；VAE 的潜在维度与信息保持之间存在折衷，过小可能限制接受率；目前仅在 Qwen3.5 体系验证，跨模型适用性尚待研究；

---

## 60. Pdb or Not to Be: Comparing Debugging Interfaces With Pass$_{X}$@$k$

**arXiv ID:** 2611.08666 | [PDF](https://arxiv.org/pdf/2611.08666v1)

**作者:** Ryan Magnuson `[一作]` (Code Metal), Ariel Lee `[通讯]` (Code Metal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在对比两种交互式调试接口（bash shell 与 Python 调试器pdb）时，使用相同的模型、工具集和预算，评估它们在SWE‑bench Verified 500个实例上的程序修复效果；通过跨代理pass_X@k指标消除单次运行波动影响；并通过多次独立跑（每种接口三次）得到更可靠的性能估计。

**💡 创新点**

提出了pass_X@k这一跨代理、无偏、方差可控的评估指标，能够客观比较不同工具或代理之间的互补性与整体性能；并通过严格匹配实验设计（模型、工具、预算、prompt、仓库快照等）消除了工具或代理引入的多余干扰。

**🔧 技术方法**

使用大规模开源模型 GPT‑OSS‑120B 与 Nemotron‑3‑Ultra‑550B，基于vLLM推理；实现了两种Agent架构（bash‑agent 与 pdb‑agent），并利用调试器接口、脚本执行、编辑工具、导航、测试等共享子系统；采用pass_X@k、pass@k 等统计指标评估成功率。

**📊 数据集**

SWE‑bench Verified（500个Python修复实例），每个实例包含基础提交、问题描述、功能与非回归测试集。

**📈 对比分析**

对比方法：在相同预算下，分别跑三次bash‑agent和三次pdb‑agent；计算pass@2（单架构）与pass_X@2（跨架构）指标；结果显示两次bash跑的成功率高于一次bash+一次pdb跑；在两种模型下，bash接口在整体成功率和token效率上均优于pdb接口；跨代理评估显示两种接口并不互补。

**⚠️ 局限性**

局限性：仅评估了两种开源模型和单一pdb调试器；SWE‑bench Verified为Python专属且复杂度分布不均，难以探测更高阶状态相关bug；模型对工具或基准的预训练程度未知；未考虑多工具协同或模型微调后的效果。

---

## 61. Higher Order Component Attribution via Learned Surrogates in High Energy Physics

**arXiv ID:** 2611.08656 | [PDF](https://arxiv.org/pdf/2611.08656v1)

**作者:** Lauri Laatu `[一作]` (Imperial College), Michael Spannowsky `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 8365 | [OpenAlex ID](https://openalex.org/A5082855661)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究利用Kolmogorov‑Arnold网络(KAN)对Particle Transformer在喷流分类中的高阶分量归因，阐释模型对物理子结构量的学习；

**💡 创新点**

将KAN作为局部代理模型实现高阶导数归因，并将梯度信息与能量相关函数(ECF)进行对齐，首次评估Transformer在捕获不可约三体交互上的能力；

**🔧 技术方法**

使用KAN代理、局部Gaussian扰动、Hessian与三阶张量分析、Spearman相关系数及梯度排名等方法；

**📊 数据集**

在JetClass十类喷流数据集上训练并评估；

**📈 对比分析**

与传统多头注意力(MHA)对比，ParT在两体相关的相关系数≈0.3，三体相关弱但显著；深度增加对归因无提升；KAN对模型决策边界的R²达到约0.99；

**⚠️ 局限性**

仅验证2、3阶归因，未探索更高阶及更复杂子结构；方法仅适用于Transformer架构，未测试对其他网络的可迁移性。

---

## 62. Faster Treasure Hunt via Strongly Universal Exploration with Polylogarithmic Overhead

**arXiv ID:** 2611.08705 | [PDF](https://arxiv.org/pdf/2611.08705v1)

**作者:** Qin Xin `[一作]` `[通讯]` (University of the Faroe Islands), Qin Xin (University of the Faroe Islands)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出了一种将O(n^c)长度的普通通用遍历序列（UES）转换为单一无限强通用遍历序列（SUES）的方法，确保在目标激活后能够在规定时间内找到目标。

**💡 创新点**

创新点在于引入了可求和的插入周期，提供了更精确的覆盖时间界限，并且在每个固定的迭代对数深度上进行了细化。

**🔧 技术方法**

使用了可求和的插入周期技术，结合了递归和边界控制的理论，确保了在不同图结构下的有效性。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在任意连接图上的应用，特别是在固定度数图和任意度数图上的表现。

**📈 对比分析**

与之前的构造方法相比，新的构造在覆盖时间上有显著改进，具体表现为O(n^c(n)(n)^{1+})的覆盖时间，且在不同的深度上提供了更好的界限。

**⚠️ 局限性**

限制在于当前的转换方法未能解决O(n^c)覆盖时间转换的存在性问题，且在处理不同激活时间和图大小时仍存在挑战。

---

## 63. Contextuality and Contradictions: Exploring Real-World Identifications of AI-Generated Writing

**arXiv ID:** 2611.08704 | [PDF](https://arxiv.org/pdf/2611.08704v1)

**作者:** Michelle Morgenstern `[一作]` (University of Vermont), Juniper Lovato `[通讯]` (University of Vermont)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对维基百科编辑者标记AI生成文本的理由与佛蒙特州高中及本科生的调查问卷进行混合方法分析，探讨人们在真实情境中识别AI写作的实践与话语。

**💡 创新点**

创新点在于揭示AI识别是高度情境化、矛盾且动态的元语言实践，展示了识别词汇的注册化进程，并指出AI检测工具可能无意中复制社区规范，导致社会化的影响。

**🔧 技术方法**

采用定性编码、主题分析、Heaps定律词汇增长曲线及计数统计等技术，结合手工标签与自动化文本分析，形成跨语料的解释框架。

**📊 数据集**

数据集包括 11,882 次维基百科 AI 生成标记（共 6,046 条带理由）以及 74 份调查问卷，产生 220 条自由文本回应。

**📈 对比分析**

通过对理由词汇的 Heaps 指数、时间序列变化以及外部资源引用比例的比较，展示了词汇开放程度与社区实践的演变；虽然未提供算法性能指标，但指出自动检测工具往往沿用社区设定的特征，易失去语境辨识能力。

**⚠️ 局限性**

局限性包括样本规模与特定性（仅佛蒙特州学生），调查为回顾性，未捕捉即时识别；维基百科数据覆盖时间有限，社区边界模糊，无法全面验证跨社区的一致性。

---

## 64. Adaptive Data Analysis and Adversarial Robustness on Insertion-Deletion Streams

**arXiv ID:** 2611.08699 | [PDF](https://arxiv.org/pdf/2611.08699v1)

**作者:** Elena Gribelyuk `[一作]` (Princeton University), Samson Zhou `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一个统一的分层估计-校正-学习框架，用来解决适应性数据分析和插入-删除流的对抗鲁棒性问题。对无限域的适应性查询给出了近似最优的样本复杂度，有限域下结合私有乘法权重进一步优化。基于此框架，构造了能够以 O(1/ε^2·polylog n) 位空间完成对 F₂ 瞬时矩估计的对抗鲁棒流算法，匹配插入仅模型的空间下界。

**💡 创新点**

1) 统一的层级估计-校正-学习结构，能够同时处理适应性数据分析与对抗鲁棒流；2) 在无限域适应性查询上样本复杂度降到 O(√k/α + 1/α²)，几乎与下界匹配；3) 在有限域利用私有乘法权重将样本复杂度降到 O(1/α² + min(√k/α, √log|U|/α²))；4) 对 F₂ 流问题引入新的递归策略与可重用的重标量更新，避免了先前算法中 B = O(1/α²) 的高空间消耗，最终实现 O(1/ε²·polylog n) 位空间。

**🔧 技术方法**

差分隐私（尤其是近似差分隐私、AboveThreshold）、高级组合、马尔可夫不等式、矩估计与有限独立随机符号的 AMS 估计、私有乘法权重、层级采样与权重更新、递归二叉树更新、对抗鲁棒估计框架、精度缩放与离散化。

**📊 数据集**

无，全部为理论实验与正式证明，实验验证已在 Lean 上完成。

**📈 对比分析**

与先前的 Dwork 等 O(√k/α²) 方案相比，样本复杂度由 1/α² 降到 1/α² + √k/α；与 Steinke‑Ullman 下界相比几乎匹配；在 F₂ 流上与先前 O(1/ε⁴·polylog n) 的对抗鲁棒算法相比，空间降低到 O(1/ε²·polylog n)，与插入仅模型的最优空间 O(1/ε²·log n) 完全匹配；实验证明在 Lean 中验证无误。

**⚠️ 局限性**

仍需对随机种子进行预生成；对离散化精度需要控制，重标量更新的数值稳定性；在极端高频率或大范围的插入‑删除场景下可能出现精度积累问题；理论结果主要在随机对抗环境下；实际实现需要进一步工程化验证。

---

## 65. Scoring, Not Learning: Decomposing Catalog-Consolidation Gains in Web-Scraped Retail Product Recognition

**arXiv ID:** 2611.08655 | [PDF](https://arxiv.org/pdf/2611.08655v1)

**作者:** Kaden Leis `[一作]` (University of Wisconsin), Xiang Ma `[通讯]` (University of Wisconsin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并清洗了基于Web抓取的奶酪图像集合，利用包装过滤、近似重复消除和标签canonicalization，将原始的1,824个查询命名类别压缩为998类识别集并进一步合并为844类；

**💡 创新点**

通过冻结模型在合并标签空间上重新评分，精确区分标签空间改变带来的定义性提升与模型再训练的残余提升，证明合并收益几乎完全来源于任务定义的简化而非模型改进；

**🔧 技术方法**

使用CLIP视觉特征与线性分类器、ConvNeXt后端微调、基于阈值的包装滤波器、近似重复消除算法以及基于字符串的标签canonicalization；

**📊 数据集**

采用Kaggle公开的Cheese Pics 165,769张Web图片，过滤后得到998类识别集，合并后为844类；

**📈 对比分析**

在同一拆分下用六个随机种子比较三种模型容量（28M、50M、88M），发现合并后Top‑1从约26%提升至≈30%，定义性提升≈4.3个百分点，模型再训练提升无统计显著；

**⚠️ 局限性**

仅使用单一数据拆分、有限的包装滤波器样本、基于名称的合并未图像验证、训练预算固定、未评估不同部署场景，可能高估合并收益并忽略类内细粒度差异。

---

## 66. An Anatomy-guided 3D Segmentation Pipeline for Prostate Surgical Phantoms from CT Scans

**arXiv ID:** 2611.08652 | [PDF](https://arxiv.org/pdf/2611.08652v1)

**作者:** Iven Chen `[一作]` (Vanderbilt University), Daniel Moyer `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种无训练、基于规则的3D分割管线，能够在高低对比的前列腺手术模型CT扫描中自动分割腔、包膜、叶片和病灶四种结构。

**💡 创新点**

创新点在于不依赖任何训练数据，而是利用几何与强度先验的组合规则，既可处理对比度极低时缺失的壁部，又能在高对比下精准分割；同时大幅度降低人工标注成本。

**🔧 技术方法**

采用经典医学图像处理技术（Otsu阈值、三视角视觉外壳、带束约束、Ridge检测、Atlas注册与局部阈值）实现，编程实现基于Python、NumPy、SciPy、scikit-image与SimpleITK。

**📊 数据集**

使用57幅CT扫描（11幅高对比，3幅含病灶，其余低对比），包括9个手工标注的低对比系列共32个病灶，并通过高对比图像伪造低对比数据进行评估。

**📈 对比分析**

与人工标注相比，自动分割仅耗时2–8分钟；在低对比下，包膜Dice中位0.83，高对比下0.91；所有32个病灶均被检测到且无误报，Recall平均≈0.98、Dice≈0.97。

**⚠️ 局限性**

在低对比下包膜不可见时，Atlas会产生过度扩展，导致边界误差（尤其极端位置），并且对全新模型的泛化能力有限。

---

## 67. Turkish Multimodal Video Fact-Checking: A New Dataset and a Comparison of Zero-Shot Agentic and Fine-Tuned Systems

**arXiv ID:** 2611.08651 | [PDF](https://arxiv.org/pdf/2611.08651v1)

**作者:** Azra Açıl `[一作]` `[通讯]` (Queen Mary University of London), Azra Açıl (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Turkish Multimodal Video Fact‑Checking Dataset（TeViD），并构建了两种基于开源模型的验证管道（3MFact‑TR和Fine‑Tuned Qwen2.5‑7B），实现了无训练和闭书两种事实核查方式；

**💡 创新点**

首次在土耳其语环境下实现全开源多角色视频事实核查框架，利用语义不一致检测、5W1H问题管理和双语证据检索三大自研模块；

**🔧 技术方法**

核心技术包括Qwen2‑5‑7B（LLM）与Qwen2‑VL‑7B（VLM）进行视频描述与文本生成、LoRA微调、BGE重排序、语义不一致检测和多步骤推理；

**📊 数据集**

使用TeViD数据集共2197条视频声明，来源为Teyit.org、Snopes（翻译）与Ensonhaber（合成），并在Teyit.org的124条真类样本上做评估；

**📈 对比分析**

对比了零射击管道与微调闭书管道，Fine‑Tuned模型在宏观F1上达到0.729，优于3MFact‑TR（0.651）和未训练模型（0.611），但优势主要来自声明文本的表面信号；

**⚠️ 局限性**

限制包括：评估仅涵盖单一组织的土耳其语样本，二元真/假标签忽略“无法验证”类别，闭书模型对新样本泛化差，且实验为单种子结果，未覆盖多模态证据质量评估。

---

## 68. Decision Context: Persistent Behavioral Learning for AI Agents

**arXiv ID:** 2611.08644 | [PDF](https://arxiv.org/pdf/2611.08644v1)

**作者:** Nima Vahdat `[一作]` (Georgian AI Lab), Pashootan Vaezipoor `[通讯]` (Georgian AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一套名为 Decision Context 的行为记忆系统，能够从用户纠正和代理自身工具执行失败中学习，生成可执行指令，并在决策点按需注入，以提升多步骤工具代理在新情境下的合规性。

**💡 创新点**

创新点在于：①将行为记忆以指令（imperative directive）形式存储，显著提升遵从率；②利用触发节点和渐进式注入机制，使数据条件规则在恰当时机激活；③融合自评（tool‑only trace）与人类纠正，构建冲突解决与优先级管理；④在同一评测基准上系统性验证各阶段改进。

**🔧 技术方法**

主要技术包括：混合检索（BM25+余弦相似度+RRF）、类型化图谱存储、Sonnet 4 规则编译器、前向与后向两阶段注入通道、冲突解析策略、以及基于指令的对话注入格式。

**📊 数据集**

使用数据集：DC‑Bench，71个由真实代理执行记录生成的场景，涵盖9种行为维度（如条件激活、隐式学习等），并在五个前沿模型（Sonnet 4.5/4.7、Opus 4.5/4.6、GPT‑4.1、GPT‑5.5）上进行验证。

**📈 对比分析**

比较方法：在相同代理与工具设置下，将 Decision Context 与事实记忆基线（Mem0、MemGPT）及程序记忆基线（AWM、Memp）对比；结果显示，在所有行为维度上平均合规率从 10%（无记忆）提升至 95%（Forge DC），并在 token 效率上实现约 10 倍压缩，成功率超过 90%。

**⚠️ 局限性**

局限性：①进阶演示未分离各技术组件的单独贡献；②基准与基线均由同一团队构建，可能存在实验偏倚；③仅覆盖两大应用领域，跨域泛化尚待验证；④对模型内部注意力机制的假设缺乏实证支持。

---

## 69. "A Space Where I Can Be Myself": Exploring Safe Visibility for Graduate Student Mothers in Mom Cafes

**arXiv ID:** 2611.08710 | [PDF](https://arxiv.org/pdf/2611.08710v1)

**作者:** Jeanne Choi `[一作]` (Northwestern University), Jeong-woo Jang `[通讯]` (KAIST)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在韩国的母亲社区 Mom Cafes 上，通过焦点小组访谈和设计工作坊，对 22 名研究生母亲进行访谈，收集并分析她们在学术与育儿交叉背景下的支持需求与隐私调节经验，提出了“安全可见性”（safe visibility）概念。

**💡 创新点**

创新点在于：①首次将研究生母亲视为交叉身份的研究对象，揭示学术与育儿双重“不可见性”；②提出“安全可见性”作为理解在线支持中可控曝光与关系负担的框架；③提出针对该交叉身份的设计建议，如可调节识别度、弱关联支持与信任机制。

**🔧 技术方法**

采用的技术主要是定性研究方法：焦点小组访谈、参与式设计工作坊、录音转写、主题编码（共 121 码），并使用迭代式代码簿构建进行分析。

**📊 数据集**

数据集为 22 名韩国研究生母亲的访谈与工作坊产出（文本转写、设计草图、情境描述），涵盖学术背景、育儿经历及使用 Mom Cafes 等在线社区的情况。

**📈 对比分析**

本研究未涉及算法或实验性比较，只在概念层面与现有关于在线支持、身份管理与亲子社区的文献对比；没有性能指标或实验结果。

**⚠️ 局限性**

局限性包括：①样本仅来自韩国，社会文化与制度差异可能影响结果的普适性；②研究采用的是小规模定性样本，缺乏量化验证；③设计工作坊的情景设定（专门想象 Mom Cafe）可能限制了对更广阔解决方案的探索；④未实现或评估任何技术原型，难以检验“安全可见性”设计的实际效果。

---

## 70. FoleyForge: Free-form, Controllable, Rectified-flow Editing of Sound Effects

**arXiv ID:** 2611.08695 | [PDF](https://arxiv.org/pdf/2611.08695v1)

**作者:** Giovana Morais `[一作]`, Oriol Nieto `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种单源音效指令编辑模型（FoleyForge），能在保持输入音效身份的前提下，对语义、风格和DSP级别的修改进行精确控制。

**💡 创新点**

创新点包括：① 通过三种编辑类型（DSP、生成式、辅助）构建的大规模合成数据管道，解决了缺乏标注编辑数据的瓶颈；② 采用可变长度序列前置（sequence‑prepend）对源音进行无时间对齐的条件化；③ 在生成网络中加入可调节属性（响度、编辑强度、能量轮廓）和自然语言幅度短语（SILA），并通过分类器自由指导实现细粒度控制。

**🔧 技术方法**

核心技术：rectified‑flow 变体的 Sketch2Sound 生成器；跨注意力与文本编码器（FLAN‑T5‑XXL）；序列前置条件化；多轴分类器自由指导（CFG）；SDEdit 初始化；SILA 语义量化；音频解码器与音频码流（Neural Audio Codec）。

**📊 数据集**

使用内部授权的高质量音效与人类标注描述数据，通过 LALM（Qwen3‑Omni‑Instruct）自动生成指令，并与 Librosa/Torchaudio 生成 DSP 处理音频；生成式编辑使用 Sketch2Sound 的原始音频；辅助编辑包括复合、逆向和无操作样本。最终训练集约 464k 条三元组。

**📈 对比分析**

与 SAO‑Instruct、SDEdit、EchoEdit+ 以及 SDEdit+Sketch 等基线在 Subtle（含真实标注）和 Creative（无标注）两套测试集上对比；指标包括对齐、方向、忠实度、编辑幅度、LALM 指令遵从评分、FAD 与 PQ。FoleyForge 在 Qwen 与 Gemini 的 LALM 评分上均居首，指令遵从最高，同时保持与基线相当的音质与忠实度；在控制实验中展示了可调节的编辑强度与能量轮廓权重。

**⚠️ 局限性**

局限性：① 合成数据仍依赖 LALM 的指令生成，可能存在偏差；② 评测主要基于自动指标和有限的用户听评，缺乏大规模真实音效编辑的纵向验证；③ 高编辑强度下音质略有下降；④ 对极端多目标或跨域（如完全不同材料）编辑的鲁棒性尚待进一步提升。

---

## 71. Fibered Separation Logic: Weakest Preconditions, Universal Constructions, and Local Independence

**arXiv ID:** 2611.08685 | [PDF](https://arxiv.org/pdf/2611.08685v1)

**作者:** John M. Li `[一作]` (Northeastern University), Pedro H. Azevedo de Amorim `[通讯]` (University of Bath)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个纤维化的分离逻辑语义框架，将弱前置条件视为强单子在 bunched fibration 上的提升，并给出了核心化、信用推理、持久化等构造的普适性。

**💡 创新点**

创新点在于：1) 建立了 bunched fibration 与 BI 超范畴的等价，统一了更高阶分离逻辑的语义；2) 将弱前置条件转化为单子提升，自动满足帧规则、归约等元规则；3) 提出了“核心化”工序的通用构造；4) 用 Simpson 的局部独立结构构造了新的概率分离逻辑，简化并澄清了以往的工作。

**🔧 技术方法**

使用的技术包括：纤维化模型（bunched fibration）、强单子与其 Huber 包络、局部独立结构、Giry/测度概率的构造、纤维化的强单子提升、RNR adjunction 等。

**📊 数据集**

本工作为理论性研究，没有使用数据集。

**📈 对比分析**

由于是理论框架，未做实验比较；通过形式化证明展示其满足帧规则、归约、单子强度等属性，证明过程与以往经验规则一致。

**⚠️ 局限性**

局限在：未针对证明相关性、步骤索引化逻辑等情况展开；缺乏对具体程序语言的实现与验证；对更广泛的效应类型的适用性尚待进一步研究。

---

## 72. Compression of Emotion in Language Model Representations

**arXiv ID:** 2611.08683 | [PDF](https://arxiv.org/pdf/2611.08683v1)

**作者:** Laya Iyer `[一作]` (Stanford University), Chen Shani `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究七种开源指令调优大型语言模型在GoEmotions数据集上的情感表示，探讨情感细粒度与整体情感结构的组织方式

**💡 创新点**

发现模型在无监督几何上呈现“过压缩”现象，即粗粒度情感（尤其正负取向）主导隐藏空间，但细粒度标签信息仍能通过线性探针恢复

**🔧 技术方法**

采用KMeans聚类、AMI与轮廓系数评估无监督结构，使用线性正则化分类器评估可解码信息，并对不同层级进行隐藏状态提取

**📊 数据集**

使用GoEmotions（36,308条Reddit评论，27个细粒度情感加中性），并映射为6个情感家族+中性以及4个情感极性类别

**📈 对比分析**

与模型输出、聚类、线性探针结果比较，发现粗粒度标签的准确率和宏F1显著高于细粒度，但细粒度标签在高层仍具可预测性，整体性能随模型规模无明显单调提升

**⚠️ 局限性**

局限包括仅评估1-9B参数的开源模型，未覆盖更大或闭源系统；使用单标签数据限制了多情感混合分析；聚类方法和表示提取策略对结果影响较大；结果不代表模型在生成任务中的因果使用

---

## 73. Transporting the Past: An Optimal Transport View of Backtracking Counterfactuals

**arXiv ID:** 2611.08682 | [PDF](https://arxiv.org/pdf/2611.08682v1)

**作者:** Zhangyi Liu `[一作]` (Stanford University), Suvrit Sra `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于最优传输的回溯因果推断框架，利用Wasserstein距离将事实后验分布投影到满足因果结构的外生空间；

**💡 创新点**

创新点在于：①将后验分布的投影视为最优传输问题；②在投影时加入独立性约束，使得推断出的外生变量保持原SCM的独立噪声结构；

**🔧 技术方法**

采用最优传输（Wasserstein-2）、约束优化（点向投影、惩罚项、硬约束）以及相应的求解算法（marginal、penalty、hard）来实现；

**📊 数据集**

使用合成非双射SCM实例（5维、6维、10维等）进行实验；

**📈 对比分析**

与传统干预式因果推断及概率转移核方法对比，OT-CF在传输成本与约束满足上优于干预式，IOT-CF能够保持外生独立性但计算成本更高，实验结果表明三种求解器在不同度量（传输成本、约束残差、独立性违背）上各有优势；

**⚠️ 局限性**

仅适用于有限后验的离散设置，独立性约束的求解复杂度高，缺乏对高维实测数据的验证与可扩展性。

---

## 74. Greedy Byte-Pair Encoding Achieves at Least 0.361 of the Optimal Compression

**arXiv ID:** 2611.08671 | [PDF](https://arxiv.org/pdf/2611.08671v1)

**作者:** Kundan Baliga `[一作]` (University of Illinois Urbana Champaign), Aditya Byju `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文证明了字节对合并（BPE）贪心算法在任意字符串、任意合并预算下所能实现的字符串长度缩短量至少为最优方案的0.361倍（即提升了原先1/3的下界）并给出了小预算下的精确极限值；

**💡 创新点**

创新点在于利用“规则确定性”将贪心合并中的三重破坏事件通过“中心移动”和“幽灵”机制重定价，从而把每一步的成本从3降至2.5，并通过潜在函数和组闭合技术进一步降低整体成本；

**🔧 技术方法**

主要技术包括：贪心合并的逐步计费（Kozma‑Voderholzer 方案）、结构化的对象追踪（intact、moved、ghost）、移动中心和幽灵的规则、潜在函数与组闭合分析、以及计算机辅助的有限检查与符号计算；

**📊 数据集**

本文未使用实际数据集，而是构造了大量符号字符串（如特殊分隔符块等）来演示极端案例和验证极限值；

**📈 对比分析**

与最优的部分合并方案相比，本文给出的下界从1/3提升至≈0.361；在预算k=2时达到3/4的最优下界，k=3时下界为7/12≈0.583；对大预算的上界为(1−e^{−2})/2≈0.4323；

**⚠️ 局限性**

局限在于仍未达到最优上界，且只在理想化的符号字符串上证明；对于更大预算和真实文本的实际表现仍未知，且核心改进依赖于尚未证明的“重定价”不等式。

---

## 75. National and State-Level Approaches to Automated Vehicle Certification: A Comparative Framework for State Operational Governance

**arXiv ID:** 2611.08681 | [PDF](https://arxiv.org/pdf/2611.08681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 76. Real-World Adoption, Use, and Governance of an LLM-Based Electronic Health Record Search and Synthesis Platform

**arXiv ID:** 2611.08672 | [PDF](https://arxiv.org/pdf/2611.08672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 77. Dynamic GPU Voxel Grids and Voxel-sequential Processing in NanoVDB

**arXiv ID:** 2611.08665 | [PDF](https://arxiv.org/pdf/2611.08665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 78. ForPC: Force-Aware Whole-Body Control for Precise Humanoid Cart Pushing

**arXiv ID:** 2611.08664 | [PDF](https://arxiv.org/pdf/2611.08664v1)

**作者:** ZhenYu Zou `[一作]`, Cheng Cheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向人形机器人手推货物的 Force-aware Whole-Body Control 框架 ForPC，能够在不使用外部力传感器的情况下实现精准的推行、转向与侧移；

**💡 创新点**

创新点包括：①将行走先验与手推交互估计模块相结合，实现对手部接触力和货车运动的隐式推断；②通过残差强化学习实现全身关节的协同调节，显著降低内部相互对抗；③实现了从仿真到真实 Unitree G1 的零调优迁移；

**🔧 技术方法**

使用技术：强化学习（两阶段：先验训练+任务层残差策略）、教师-学生隐式交互动力学推断、力感知隐变量编码、基于 Isaac Sim 与 MuJoCo 的仿真、真实机器人控制；

**📊 数据集**

数据集：仿真环境中随机化的手部负荷、姿态偏移与 15kg/25kg 负载组合；物理实验中 Unitree G1 携带 0–30kg 货物的真实推行轨迹；

**📈 对比分析**

比较方法：与仅使用行走先验的 Loco-only、仅采用力感知但无先验、仅有先验无力感知三种配置对比；在仿真中 ForPC 将卡车速度 RMSE 下降至 68.2% 以内，交叉轨迹 RMSE 降至 74.2%；在真实机器人上成功完成 30kg 负载 0.5m/s 推行，无需再训练；

**⚠️ 局限性**

局限性：①对复杂障碍环境和不规则轨迹的适应性尚未验证；②依赖于摄像头检测卡车方向，若视觉失败会影响控制；③目前仅针对四轮推车，未扩展到不同形状或多车系统；

---

## 79. Differentially Private All-Pairs Min Cut Values

**arXiv ID:** 2611.08650 | [PDF](https://arxiv.org/pdf/2611.08650v1)

**作者:** Anders Aamand `[一作]` (Rice University), Yinzhan Xu `[通讯]` (University of California San Diego)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在边邻接模型下，如何私密地发布给定n节点加权图中所有顶点对的最小割值。

**💡 创新点**

提出了一种近似DP算法，能够以O(√(n))的加性误差输出所有对的最小割值，突破了线性误差的限制。

**🔧 技术方法**

使用了稀疏向量技术（Sparse Vector Technique）来维护基于最小割结构的顶点层次分解。

**📊 数据集**

使用了加权图，假设边权为非负且有多项式界限。

**📈 对比分析**

与之前的算法相比，新的算法在输出所有最小割值时的加性误差显著降低，且证明了Ω(log^3/2 n)的下界，表明释放所有最小割值比释放单个最小割值更具挑战性。

**⚠️ 局限性**

算法的局限性在于对于无界权重的处理，虽然提出了处理方法，但仍需进一步优化以提高效率。

---

## 80. ResServe: Accurate and Efficient FP4 LLM Inference with Residue Compensation

**arXiv ID:** 2611.08713 | [PDF](https://arxiv.org/pdf/2611.08713v1)

**作者:** Shuo Rong `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在FP4量化下的残差补偿框架ResServe，用于LLM推理，既提升精度又保持低延迟。

**💡 创新点**

创新点在于：①利用解码时矩阵窄化的空闲GPU计算资源进行残差补偿；②在基于FP4的线性层和注意力中实现on‑chip结果合并；③针对prefill阶段采用选择性通道补偿控制额外开销。

**🔧 技术方法**

使用技术包括block‑scaled FP4（NVFP4/MXFP4）量化、M-Ext与ResAttn高效内核、on‑chip Fold、选择性通道选取、NVFP4 KV缓存、vLLM/FlashInfer以及TensorRT‑LLM集成。

**📊 数据集**

评估数据集涵盖Qwen3系列（4B/8B/32B/27B）、Swift‑27B、LiveCodeBench（Hard350）、HMMT、AIME、MathArena、MRCR 8‑needle检索等。

**📈 对比分析**

对比方法为与同格式plain FP4（W4A4）和W4A8（TensorRT‑LLM/FlashInfer）在GPU B200上进行线性层、注意力延迟与NMSE测评；在HTTP请求层面评估TTFT/TPOT/RT。结果显示ResServe在保持近W4A4延迟的同时，代码/数学任务准确率提升15–53%，相对W4A8可降低22–53%请求延迟，误差降低16–68%。

**⚠️ 局限性**

局限性包括：①主要针对窄矩阵的解码阶段；②prefill补偿仍需额外成本，比例设定需手工；③目前仅支持NVFP4/MXFP4，未覆盖更宽位宽；④对不同模型需手动调参，缺乏完全自适应；⑤在大规模MoE或高批量推理中的表现尚未验证。

---

## 81. FinEvolveKG: Controlled and Attributable Prompt Evolution for Financial Knowledge Graph Construction

**arXiv ID:** 2611.08643 | [PDF](https://arxiv.org/pdf/2611.08643v1)

**作者:** Abhinav Arun `[一作]` (Domyn), Stefano Pasquali `[通讯]` (Domyn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了 FinEvolveKG 系统，实现金融知识图谱构建的可审计 persona 适配流程；

**💡 创新点**

创新点在于将通用构建规则与 persona 政策块分离，机械约束 GEPA 只修改 persona 块，保证改动可追溯、可审计；

**🔧 技术方法**

使用 Claude Sonnet‑4‑6 生成参考 KG，利用 GEPA 反射式搜索和 DSPy 组合程序，双目标（泛化质量与 persona 相关性）评估框架；

**📊 数据集**

采用 2024 年 SEC 10‑K 文档的 20 家信息技术公司，划分为训练、验证、测试三组，并在 5 个 OOD 组（行业/年份）上进一步验证；

**📈 对比分析**

通过与冻结提示（v1）、共享块（v2）、全提示（v5）对比，评估 𝒬_g、𝒬_ρ 与混合 𝒬，结果显示 v4 在所有 persona 上均优于 v1 和 v5，泛化得分提升 19–31%，业务得分提升 2–6%，且 OOD 场景下仍保持优势；

**⚠️ 局限性**

局限性包括共享块选择未系统评估、OOB 仅覆盖少数行业/年份、参考 KG 验证规模有限、未直接测 downstream 任务效能、边界保证仅限编辑位置未检验语义隔离效果。

---

## 82. The GPU Needs an Operating System

**arXiv ID:** 2611.08720 | [PDF](https://arxiv.org/pdf/2611.08720v1)

**作者:** Reza Nourai `[一作]` (OpenInfer), Behnam Bastani `[通讯]` (OpenInfer)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究GPU在大规模AI模型推理时的空闲现象，提出并实现了类似操作系统的GPU共享框架，并构建了一个能够持续运行模型且支持后台任务的研究系统。

**💡 创新点**

创新点在于把GPU视为多资源平台，设计了六层OS架构（硬件特性、内核规则、GPU调度、资源管控、成本模型、工作准入）以及能够在几微秒内快速暂停与恢复的内核切片方案；同时首次在单个GPU上通过驻留内核实现模型推理与后台任务并行，显著提升了资源利用率。

**🔧 技术方法**

技术包括：GPU绿色上下文（green context）分区、短暂可恢复的内核切片、基于成本模型的资源预估、实验性硬件特性请求（快速暂停、内存优先级、可调分区）以及对Llama-3.1‑8B和Kimi‑K2等大模型的自研推理引擎。

**📊 数据集**

使用的数据集主要为公开的大规模语言模型参数与KV缓存（如397B MoE、554B Kimi‑K2）以及synthetic背景工作（如FP8 GEMM、FP8 GEMM-2048^3）和真实矩阵乘法任务；实验环境为NVIDIA RTX PRO 6000 Blackwell GPU。

**📈 对比分析**

与现有vLLM最佳配置比较时，单请求驻留内核比vLLM快约3%；在同一GPU上共跑后台工作时，使用绿色上下文可恢复约0.22~0.24个GPU资源，导致推理步长平均仅增加1.6%以内；而使用现有分区机制只能恢复约0.23GPU，步长增幅5.1%；整体在6%延迟预算内实现了高效共跑。

**⚠️ 局限性**

限制主要在于当前GPU硬件缺乏快速暂停、内存与缓存优先级以及细粒度分区功能；现有推理内核仍未完全支持可恢复切片，导致后台任务对缓存与内存的竞争仍是瓶颈，且仅在单GPU实验中验证，尚未在多GPU集群环境下测试。

---

## 83. Does a Trading Policy Need to Know What It Trades? Microstructure Invariance and Zero-Shot Transfer of Execution Policies

**arXiv ID:** 2611.08701 | [PDF](https://arxiv.org/pdf/2611.08701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. CPU Micro-Architectural Characteristic Sensitivity Study on CPU-Side Agentic Task Latency

**arXiv ID:** 2611.08697 | [PDF](https://arxiv.org/pdf/2611.08697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 85. Measuring Bounded Autonomy for an LLM Decomposition Planner on a Live 5G-Core Testbed

**arXiv ID:** 2611.08707 | [PDF](https://arxiv.org/pdf/2611.08707v1)

**作者:** Sanjeet Raj Pandey `[一作]` (Technische Universitaet Berlin), Axel Küpper `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在5G核心控制平面测试平台上实现并测量基于LLM的分解决策的 bounded autonomy，构建了双阶段决策链（阈值策略 + 受限LLM规划器），并通过 JSON schema、投影修复验证器以及确定性回退确保输出合法性与可追溯性。

**💡 创新点**

创新点包括：①首次在真实 5G‑core 环境闭合 bounded runtime loop；②将 LLM 输出严格约束在可执行动作集合内，使用 schema + validator + fallback 实现安全沙箱；③证明 bounded planner 能显著降低违规率、实现可追溯决策；④将 LLM 的角色定位为离线规则 elicitation 而非在线控制器。

**🔧 技术方法**

使用技术：多种 LLM（GPT‑4、Claude、Gemini 等）+ JSON schema + 投影修复验证器 + 确定性回退；Kubernetes + free5GC 5G 核心 + UERANSIM UE；日志解析、压力指标计算；两阶段决策逻辑（阈值 + LLM）；控制器与调度器实现。

**📊 数据集**

数据集与实验场景：63 次真实工作负载运行（共 7,150 事件），10 个 held‑out 场景；11 个合成压力向量；22 次标准调用与 12 次对抗调用（共 44+12 次）；另外 27k 次完整执行作为对照。

**📈 对比分析**

对比方法：将 bounded 与 unbounded planner 进行对比；与确定性回退与阈值策略单独评估；通过方案合法性（schema validity）、违规率、选择性与泛化（对 held‑out 与不同模型）进行量化；性能指标：推理延迟约 2.97 s，系统在每个观察窗口完成决策；在 27k 次执行中 100% 成功，未出现阻塞。

**⚠️ 局限性**

局限性：①仅在进程内实现 handler，未验证真实 FaaS 运行时；②状态存储仅内存，未分布式；③实验场景有限，样本量小；④validator 仅基于单步约束，未考虑序列约束；⑤依赖日志解析，格式变更可能导致误差；⑥模型可用性变化仅触发 fallback，未探究更复杂的自适应策略；⑦缺乏奖励学习与影响感知的控制器。

---

## 86. Enabling Retraining-Free Accurate Inference in Analog Compute in Memory Accelerators

**arXiv ID:** 2611.08693 | [PDF](https://arxiv.org/pdf/2611.08693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 87. Dual-Stream LLM-Guided Reward Design for Reject-Inference-Aware Reinforcement Learning in E-Commerce Payment Fraud Detection

**arXiv ID:** 2611.08706 | [PDF](https://arxiv.org/pdf/2611.08706v1)

**作者:** Bo Qu `[一作]` (eBay), Ming Ouyang `[通讯]` (eBay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种双流奖励设计的强化学习框架，利用大型语言模型（LLM）对已批准和被拦截的交易数据共同学习欺诈检测策略。

**💡 创新点**

创新点在于将拒绝推断（reject‑inference）问题嵌入奖励设计中，LLM在零样本和少样本条件下自动生成包含两条流的奖励函数，并自行选择有用的风险信号。

**🔧 技术方法**

技术上结合了LLM驱动的代码生成（Qwen3 FP8）、REINFORCE两步策略、约束式MDP与拉格朗日精度约束，以及对旧版GBM分数的对比评估。

**📊 数据集**

实验数据来自真实电商支付系统，涵盖约520万笔交易，其中约200万笔已批准且标注了欺诈标签，约390万笔历史拦截但无标签的交易。

**📈 对比分析**

与传统GBM基线及仅使用已批准样本的单流模型相比，双流模型在四个月的离线测试中平均提升了约6.83个百分点的美元加权精准率，在所有召回阈值上均优于两者。

**⚠️ 局限性**

局限性在于拦截样本为非随机（MNAR）偏倚，双流奖励假设最新GBM分数在被拦截样本中仍具风险信息，若该假设失效可能导致策略误导。

---

## 88. Foundation-Model Residuals as Signals of Urban Bus Irregularity

**arXiv ID:** 2611.08703 | [PDF](https://arxiv.org/pdf/2611.08703v1)

**作者:** Csaba I. Sidló `[一作]` (Institute for Computer Science and Control), András J. Molnár `[通讯]` (Institute for Computer Science and Control)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了预训练轨迹模型（UniTraj、TrajFM）在巴西Niterói公交网络中的迁移误差，探讨误差作为公交异常的潜在信号。

**💡 创新点**

将迁移误差解读为当地条件的代理，利用可解释AI揭示误差背后的城市特征，并结合15分钟城市框架进行分析。

**🔧 技术方法**

使用预训练的UniTraj与TrajFM模型、掩码段重建、XGBoost元模型、SHAP可解释性分析以及二分类器。

**📊 数据集**

基于NetMob26数据集，包含GPS轨迹、票务、天气、网络和社会经济信息，并加入15分钟城市POI指标。

**📈 对比分析**

将预训练模型与仅用Niterói数据训练的本地基线比较；UniTraj微调后位置误差下降至421m（基线664m），TrajFM在旅行时间上MAE 5.12min（基线5.94min）；元模型解释率分别为55%和7%。

**⚠️ 局限性**

误差并非直接证明服务中断，仅为潜在异常指标；TrajFM元模型解释力弱，缺乏因果性验证；部分速度相关变量被排除，结果可能受数据缺失与模型配置影响。

---

## 89. CS-Pi: Constraint-Satisfying Flow Policy Sampling

**arXiv ID:** 2611.08654 | [PDF](https://arxiv.org/pdf/2611.08654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 90. Hybrid Quantum-Classical Annealing for Protein Side-Chain Optimization: A Feasibility Study

**arXiv ID:** 2611.08696 | [PDF](https://arxiv.org/pdf/2611.08696v1)

**作者:** Valeria B. C. Visani `[一作]` (Politecnico di Milano), Elisabetta Di Nitto `[通讯]` (Politecnico di Milano)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文把蛋白质侧链优化问题映射为二进制二次规划（BQP），并在D-Wave的混合量子-经典解算器上进行实验，比较其与Gurobi在真实蛋白实例上的性能。

**💡 创新点**

创新点在于首次证明PSCO天然是BQP且适合混合量子退火，并展示混合解算器在无需人工设定惩罚权重的情况下，在大规模实例上优于行业领先的经典求解器。

**🔧 技术方法**

使用的技术包括D-Wave Leap混合解算器（CQM模式）、Gurobi优化器、PyRosetta能量计算、Rosetta能量函数以及基于One-Hot约束的二次模型。

**📊 数据集**

数据集为20条来自RCSB PDB的蛋白结构，生成40个PSCO实例，二进制变量数从50到1970，使用Rosetta计算的能量项。

**📈 对比分析**

比较方法是在每个实例上分别跑5秒混合解算器、10秒/1000秒Gurobi以及4小时Gurobi默认配置，评估相对能量差；结果显示混合解算器在大多数实例上能达到或超过Gurobi，尤其在变量量约为10^3时优势更明显。

**⚠️ 局限性**

局限性包括经典组件未公开，QPU仅占约0.63%运行时间，难以单独评估量子贡献；未与专门的PSCO经典启发式（如Rosetta）比较，也未探讨多样性候选池的生成。

---

## 91. AutoParallel: Composable, PyTorch-Native Automatic Parallelism for Production Training

**arXiv ID:** 2611.08660 | [PDF](https://arxiv.org/pdf/2611.08660v1)

**作者:** Kaijian Wang `[一作]` (Meta Superintelligence Labs), Francisco Massa `[通讯]` (Meta Superintelligence Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

AutoParallel 是一种基于 PyTorch 的自动化分布式训练规划器，能够在完整的前向‑后向计算图上进行张量放置搜索，并直接生成可在标准 PyTorch 编译与执行栈中运行的普通模块；

**💡 创新点**

创新点在于（1）将 SPMD 放置搜索视为全图优化，使用硬件感知成本模型与可扩展的近似 MRF 求解器；（2）通过 placement contracts 在计划层面与用户手写的动态并行区域（如上下文并行、Mixture‑of‑Experts）明确边界，实现可视化与可复现；（3）完全无额外运行时，兼容现有混合精度、激活检查点、分布式检查点等生产级优化；

**🔧 技术方法**

技术包括：PyTorch 生态下的 TorchDynamo、AOTAutograd、DTensor、DeviceMesh；全图张量放置枚举与约束；基于硬件特征的算子计算与通信成本模型；分层 MRF 近似求解器（seed‑MRF + TRW‑S + 局部搜索）；在计划阶段插入重分配 collectives；与 TorchTitan GraphTrainer/Inductor 的无缝集成；

**📊 数据集**

使用的模型与数据集：LLaMA 3 8B/70B、Qwen3 32B、DeepSeek‑V3 16B MoE、MuseGlimmer 30B、BLOOM、StableLM、Phi‑1/2、Falcon、Pythia、GPT‑2、T5、GPT‑J 等公开 Hugging Face 模型；训练任务覆盖 dense、混合、Mixture‑of‑Experts 与长上下文注意力，序列长度从 4K 至 32K，设备规模 8‑128 H100 GPU。

**📈 对比分析**

比较方法：在相同硬件与编译设置下，AutoParallel 与 TorchTitan 主干的手写专家策略（MainTrainer）对比；评估指标为每秒吞吐量（tokens/s）和一步耗时；实验表明 AutoParallel 在 14 组不同网格配置中，在 9 组超越专家策略，其他 5 组差距 ≤ 3.6%；在 4D 网格（128 GPU）与高维 HSDP+CP+TP 方案中仅落后 3.6%；在不同序列长度下，重新规划可提升 1.6%–51.3% 的吞吐量；在 17 大模型基准上，AutoParallel 的成本模型目标比 SimpleFSDP 提高 0.2%–40.8% 的效率；

**⚠️ 局限性**

限制：计划空间依赖 DTensor 的传播规则，无法覆盖需要动态收集或数据依赖路由的算子；在极大维度（4D+）的网格中，近似求解仍有一定误差；成本模型不考虑通信与计算的重叠与 collectives 的分桶；对非常大模型的激活与梯度压缩策略仍需手动介入；

---

## 92. JEPAhead: From Inpainting to Causal Forecasting for Urban Driving

**arXiv ID:** 2611.08657 | [PDF](https://arxiv.org/pdf/2611.08657v1)

**作者:** Daniel G. P. Petrini `[一作]` (Eldorado Research Institute), Felipe Pamboukian `[通讯]` (Eldorado Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在冻结的V‑JEPA1编码器上微调预训练的预测器，实现了对未来16帧（8个时间组）的因果预测。

**💡 创新点**

创新点在于：①将原本用于双向填充的V‑JEPA1预测器改为单向因果预测；②通过组级因果注意力和连续过去上下文一次性预测全部未来组；③在不解码像素的前提下，直接在潜在空间进行预测并验证其包含驾驶相关语义信息。

**🔧 技术方法**

主要技术包括：V‑JEPA1 ViT‑L/16编码器、预训练的预测器微调、组级因果注意力掩码、残差输出（预测变更）、GAN解码器（用于像素评估）以及线性探测器和检索评估。

**📊 数据集**

使用CARLA Town12模拟驾驶数据，生成32帧上下文和16帧未来，划分训练/验证/测试集，构造591个候选未来样本进行检索。

**📈 对比分析**

与保持最后一帧不变的持久性基线、线性/EMA等解析基线以及从零训练的紧凑预测器比较。结果显示：潜在空间检索Top‑1准确率69.0%（持久性55.3%），Top‑5 84.8%；潜在余弦相似度原始模型0.8478，紧凑模型0.8421；在动态令牌上相似度提升约39%。像素空间L1误差小幅下降，LPIPS几乎无提升；但与oracle解码相比，预测器仍有限。

**⚠️ 局限性**

局限性包括：仅在CARLA Town12模拟环境和16帧预测窗口内测试；未考虑未来驾驶动作；缺乏对目标检测、语义分割等下游任务的验证；GAN解码器的重建误差影响像素评估；并且在真实世界数据和更长预测时程的泛化尚未验证。

---

## 93. VoluCompose: Constructing Compact Volumetric Visual Representations for Medical Image Understanding

**arXiv ID:** 2611.08649 | [PDF](https://arxiv.org/pdf/2611.08649v1)

**作者:** Zanting Ye `[一作]` (Southern Medical University), Lijun Lu `[通讯]` (Southern Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种名为 VoluCompose 的无训练、面向任何视觉语言模型的体素压缩框架，能够在不增加额外算力或参数的情况下，将完整医学体积的视觉 tokens 约减至约 10–30%。

**💡 创新点**

创新点在于将内容选择与容量控制完全分离：使用 Lorentz 超曲几何对相邻切片特征进行分组并通过指数门控的距离度量挑选代表性 token；同时利用视觉编码器自注意力熵来自适应地决定整体 token 上限，从而兼顾空间冗余压缩与接口特定的资源分配。

**🔧 技术方法**

主要技术包括：超曲几何中的 Lorentz 距离与指数门控、基于注意力熵的容量校准、无监督的滑动窗口支持单元构建、以及局部邻域加权重化的 token 精炼。

**📊 数据集**

使用了三个医学体积数据集：3D‑RAD（CT 切片）、M3D（多模态 CT/CE/MR）以及 AMOS‑MM（胸部 CT 报告生成），并在四个主流 VLM backbones 上进行评估：Hulu‑7B、MedGemma‑1.5‑4B、Qwen2.5‑VL‑7B 与 Qwen3‑VL‑8B。

**📈 对比分析**

与 HoliTom、VisionZip、UniComp 等基线在固定 10%、20%、30% 预算下以及自适应 ρmax=0.30 的压缩协议下对比，VoluCompose 在 9 组评测中平均 ROUGE‑L、Close‑Acc 以及整体相对质量（Q）均高于 100%，在大多数设定中获得最高的相对质量与保真度（如 ROUGE‑L 最高 96.4% 以上），同时仅保留约 20% 视觉 token，显著降低解码器 FLOPs 与 KV‑cache 需求。

**⚠️ 局限性**

局限性包括：依赖冻结的视觉编码器，无法进一步学习压缩策略；对切片顺序和切片数量的假设在非标准体积（如不规则切片间距或极深扫描）下可能不适用；以及在某些轻度注意力分布均匀的模型（如 Qwen 系列）中自适应预算未能充分发挥优势。

---

## 94. IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection

**arXiv ID:** 2611.08647 | [PDF](https://arxiv.org/pdf/2611.08647v1)

**作者:** Chris von Csefalvay `[一作]` (HCLTech), Tamás Földi `[通讯]` (HCLTech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 IWIBench benchmark，系统评估模型在无梯度条件下对目标模型（Gemma 3 4B）权重进行编辑以实现十类文本编辑任务的能力。

**💡 创新点**

创新点在于：①首次构建面向梯度自由权重注入的统一测试框架；②设计了四种有限工具与预算限制的编辑接口；③将任务拆分为刺激、抑制、调节三大类，并制定精准的目标与控制评估；④通过宏观奖励与控制保留度量衡量编辑效果。

**🔧 技术方法**

使用的技术包括：无梯度权重读取与写入操作（复制、缩放、低秩加法等）、受限工具调用、基于模板的目标验证、宏观奖励公式（目标增益×保留率），以及多轮交互式编辑与评估流程。

**📊 数据集**

使用的数据集为：目标模型 Gemma 3 4B IT、30个选定的编辑 episode、551个目标任务项、2288个控制项，全部来自公开实现（GitHub + Hugging Face 共享）。

**📈 对比分析**

比较方法：将所有编辑器的宏观奖励按十个任务族等权重平均；结果显示前沿模型（6.1‑sol 与 Astra）宏观奖励分别为 0.4601 与 0.4551，显著高于后排模型（GLM 0.0135、DeepSeek 0.0464、Nemotron 0.0548），表明目前大多数模型在权重注入方面能力分布不均，优质模型在多任务上表现突出。

**⚠️ 局限性**

局限性：①对结构化响应、答案长度控制等任务的成功率极低；②编辑效果高度依赖模型的“思维”深度和经验，无法在所有模型间统一提升；③编辑成本随思维深度激增，实际应用受限；④风险面未被充分防护，潜在恶意传播可能性存在；⑤评测仅覆盖单一目标模型和有限任务，尚需扩展多模型、多任务的泛化验证。

---

## 95. From Geometry to Function: A Multi-Task LiDAR Benchmark Dataset for Semantic and Functional Parsing of Urban Facades

**arXiv ID:** 2611.08645 | [PDF](https://arxiv.org/pdf/2611.08645v1)

**作者:** Abdiaziz Muse `[一作]` (Saint Mary's University), Dong Chen `[通讯]` (Nanjing Forestry University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对城市建筑立面，构建了 HFX3D 规模化 LiDAR 数据集并提供语义、实例及每个实例的 15 项功能属性标注，完成三任务基准（语义分割、实例分割、功能属性预测）

**💡 创新点**

首次将功能属性与 3D 立面点云标注相结合，形成三层级标注体系，并提出基于语义先验、几何证据和上下文图融合的功能属性推理管线

**🔧 技术方法**

采用 GeoSLAM ZEB Horizon 手持 LiDAR+RGB 相机获取数据，使用 DBSCAN+人工校正进行实例聚类，功能属性通过基于 IFC/ Getty AAT 的规则、几何特征与图上下文推理，再经人工验证完成；在语义任务上训练 PointNet++/KPConv/Point Transformer 等网络，在实例任务上评测 PointGroup/ISBNet/SoftGroup 与 HDBSCAN/DBSCAN；功能任务用证据融合与 MLP 进行对比

**📊 数据集**

HFX3D：约 492 M 点、26 栋建筑、12 语义类、6,418 实例、每实例 15 项功能属性；对比数据集为 ZAHA（Munich）和 City‑Facade（厦门）

**📈 对比分析**

KPConv 在语义分割上取得最高 47.26% mIoU；SoftGroup 在实例分割上获得 AP50 0.2923；在功能属性预测中，S+G+C 组合得到 Ambiguous 子集 mF1 0.6606，最佳单源 G 的 mF1 为 0.7039；跨域实验显示 HFX3D 在 ZAHA 上表现优于相反方向，说明 HFX3D 更具环境多样性

**⚠️ 局限性**

主要局限：严重类别不平衡导致稀有类识别低；重复窗户、遮挡导致实例分割与功能预测困难；功能推理依赖部分点覆盖与稀疏玻璃面，难以捕获所有细节；模型在跨域迁移时仍受建筑风格与标注差异影响

---

## 96. Aspect-wise Pairwise Tournaments for Test-Time Selection

**arXiv ID:** 2611.08640 | [PDF](https://arxiv.org/pdf/2611.08640v1)

**作者:** Hannah Guan `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的测试时间选择方法，称为Aspect-wise Pairwise Tournaments (APT)，通过比较候选解决方案的多个方面来选择最佳候选者。

**💡 创新点**

APT结合了人类决策的直觉，通过逐个方面的比较来进行验证，允许用户根据个人偏好调整选择标准。

**🔧 技术方法**

使用了无训练的单淘汰赛制，结合多个现成的方面验证器进行配对比较。

**📊 数据集**

在五个领域（数学推理、一般知识、战略推理）和三个生成LLM（Gemini-3.1-Pro、GPT-5.2、Qwen3-8B）上进行了评估。

**📈 对比分析**

与自一致性、奖励模型验证和二元多方面批准方法相比，APT在14个15个生成-领域对中表现最佳，尤其在开放式任务上表现出显著提升。

**⚠️ 局限性**

APT仅测试了来自同一模型家族的生成器和验证器，且主要评估在可验证答案的领域，未在摘要、长篇写作或代码生成等领域进行测试。

---

## 97. Sensing-Based Communication: A Non-Orthogonal Passive Reflection Framework for ISAC

**arXiv ID:** 2611.08809 | [PDF](https://arxiv.org/pdf/2611.08809v1)

**作者:** Xin Xiong `[一作]` (Colorado School of Mines), Rafael Fierro `[通讯]` (University of New Mexico)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种非正交的被动反射集成感知与通信机制——基于感知的通信，利用下行信号在低功耗标签上进行脉冲调制，实现在不占用额外频谱的前提下同时完成感知与上行通信。

**💡 创新点**

创新点在于将感知与通信功能统一到同一被动标签上，采用极简的OOK调制并将所有复杂处理移至接收端，突破传统正交资源分配导致的下行吞吐瓶颈，提出非正交多用户访问与无资源占用的方案。

**🔧 技术方法**

技术手段包括光学反射式 ISAC 架构、基于光学信号的宽带 Gaussian/OFDM 下行波、被动 OOK 标签、交叉相关与顺序干扰消除（SIC）解码、统计信号模型、闭式 BER 公式及实验测试平台。

**📊 数据集**

使用自建实验平台（LED/微LED 发射机、光学标签、光电探测器）与仿真数据（BPSK-OFDM 信号、250 MHz 光源）进行验证，并未采用公开数据集。

**📈 对比分析**

与传统 TDD/TDM、FDD/FDM 资源分割方案对比，实验与仿真表明该机制在保持下行速率不下降的情况下实现多标签上行访问，BER≤10⁻³ 时可达 25 Mbps，整体性能与理论模型吻合（R²≈0.94，MAPE≈14 %）。

**⚠️ 局限性**

局限性包括：对 OOK 调制深度、标签间距及路径损耗敏感；多标签环境下远端标签受到 d⁴ 路径损耗限制；LED 带宽受限导致实际上行速率受限；残余干扰需进一步降低以提升吞吐。

---

## 98. AircraftWitness: How Often Aircraft Can Locate Recordings

**arXiv ID:** 2611.08791 | [PDF](https://arxiv.org/pdf/2611.08791v1)

**作者:** Kris Atallah `[一作]` `[通讯]`, Kris Atallah

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文测定了利用飞机尾声出现的多普勒声谱定位户外录音的实际成功率，并基于预先登记的盲测方案在多组公开数据集上评估了三种定位器的表现。

**💡 创新点**

创新点在于：①首次以预先注册且不依赖先验答案的盲测方式量化飞机定位方法的成功概率；②构建公开的 AircraftWitness benchmark，提供标注好的 ADS‑B 近距离标签和定位器评估脚本；③系统对比三种主流定位器（reference、dopplerprint、Unheard），并探讨了时钟漂移、噪声抑制和多普勒曲线检测对性能的影响。

**🔧 技术方法**

技术主要包括：多普勒频移曲线建模、基于 ADS‑B 数据库的飞机轨迹匹配、声频信号的清晰通道检测、时间校准与漂移估计、以及统计显著性检验（对比均匀随机猜测与对照实验）。

**📊 数据集**

数据集涵盖：A2O（澳洲声学观测站）批次 3 的 500+ 个录音；Eclipse Soundscapes 录音；xeno‑canto、iNaturalist、直播流与公园录音；以及随机抽取的公开录音。每个样本都配有实际飞机通过记录或 ADS‑B 接入时间和位置。

**📈 对比分析**

比较方法：在每个测试集上计算 1 km 内定位成功率，并与均匀随机猜测、时钟漂移控制（+5 min、+2 h）以及音频替换对照进行对比。结果显示：在盲测中，reference 约 44 % 的录音能被准确定位，dopplerprint 约 30 %，Unheard 约 20 %；在 GPS‑定时网络中，reference 与 dopplerprint 分别达到约 35 % 与 25 %；在随机公开录音中，所有定位器的成功率均低于 5 %（几乎为零）。

**⚠️ 局限性**

局限性包括：①仅对检测到明显降落声谱的 5 min 录音进行评估，未覆盖更长或噪声更大场景；②依赖 ADS‑B 覆盖，无法定位无 ADS‑B 记录的机型；③假设声速恒定、直线传播，未考虑多路径和环境效应；④时钟漂移校准依赖于多通道传感器，单通道时钟可能误差更大；⑤盲测样本量有限，未能覆盖全球多样化气象与地形环境。

---

## 99. Learning from How Users Interpret: Personalizing Recommendation Explanation Order

**arXiv ID:** 2611.08785 | [PDF](https://arxiv.org/pdf/2611.08785v1)

**作者:** Deo Munduku `[一作]` (Paris-Dauphine University, PSL Research University), Elsa Negre `[通讯]` (Paris-Dauphine University, PSL Research University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究利用用户以往文本解释中的解释规律来生成个性化的解释组织顺序，并在保持解释内容不变的前提下，通过四种不同的组织策略（自身历史档案、他人档案、总体人群档案以及基于LLM的目标感知排序）对比，评估其对客观理解的影响。

**💡 创新点**

创新点在于：①首次把用户先前的开放式文本解释转化为三维（选择、显著性、关联）结构化档案，用于指导后续解释句子排列；②通过完全固定句子内容，只改动顺序来纯粹检验组织对理解的作用；③将规则式排序与基于LLM的动态排序进行直接对比。

**🔧 技术方法**

技术实现主要包括：ISMA三维表示框架；句子与信息单元的语义匹配使用Sentence Transformers；关联维度通过Qwen3-8B对文本片段进行共动检索；最终的句子排序由规则化步骤完成；对比实验中使用的LLM为Mistral-7B-Instruct-v0.3，接受结构化档案与目标上下文作为提示。

**📊 数据集**

数据集为122名参与者在四个源情境中生成的文本解释（共488条），以及四个目标情境的六句固定说明文本；源解释通过自动化提取生成选择、显著性与共动坐标；目标情境与对应的多选理解题构成实验数据。

**📈 对比分析**

采用within-participant 4×4 Latin方阵设计，主指标为客观理解（两道多选题总分）。实验结果显示OWN（自我历史档案）准确率为86.5%，显著高于DONOR（66.0%）、POP（63.9%）和PROFILE-LLM（73.0%）。对比的odds ratio为OWN vs DONOR 3.82（95% CI 2.36–6.18），OWN vs POP 4.23（95% CI 2.61–6.84），OWN vs PROFILE-LLM 2.67（95% CI 1.64–4.36）。

**⚠️ 局限性**

局限性包括：样本量相对有限、仅涵盖英语和法语、解释仅为六句固定结构、每个用户仅有四个先前解释、自动提取方法未在本语料库再次验证、关联维度仅用共动检索且不区分关系类型、与LLM对比仅使用一种模型与提示、未包含偏好直接对比、以及客观理解评测为短期记忆测试，未能揭示长期效果或认知机制。

---

## 100. Federated training performs as well as centralized for 3D Latent Diffusion for Diagnosis-Conditioned Brain MRI Synthesis

**arXiv ID:** 2611.08771 | [PDF](https://arxiv.org/pdf/2611.08771v1)

**作者:** Abhijith Shaji `[一作]` (University of Southern California), Jose-Luis Ambite `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

训练并比较了同一诊断条件下的3D潜在扩散模型在联邦学习与集中式训练下生成脑MRI的表现

**💡 创新点**

首次在五个分散站点上使用联邦学习完整训练3D潜在扩散模型，并证明其在条件忠实度、真实感和重建质量上与集中式训练相当或更优

**🔧 技术方法**

3D潜在扩散模型、MONAI AutoencoderKL、DiffusionModelUNet、联邦平均、DDIM采样、Fréchet Inception Distance、MS-SSIM、3D DenseNet‑121分类器

**📊 数据集**

ADNI（ADNI‑1、ADNI‑2/GO、ADNI‑3）和NACC（GE、Si/Ph）共计约5千张T1加权MRI，分为训练、验证、测试

**📈 对比分析**

对比两种训练方式生成1500张CN/AD样本，评估条件忠实度（AUC 0.934 vs 0.922）、真实感（FID 1.25 vs 1.89），发现联邦模型在真实感上更好，条件忠实度与集中式相近，样本多样性略低

**⚠️ 局限性**

联邦生成的样本聚集在典型解剖结构，导致多样性略低，对极端脑形态的再现不足

---

## 101. MEDALLION: Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON

**arXiv ID:** 2611.08767 | [PDF](https://arxiv.org/pdf/2611.08767v1)

**作者:** Vicky Jung `[一作]` (University of Florida), Genevera I. Allen `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MEDALLION 框架，用知识蒸馏把任意降维嵌入映射到可解释的神经编码器并通过特征归因解释低维表示。

**💡 创新点**

创新点在于将降维嵌入作为教师蒸馏对象，构建可解释的学生网络，兼顾嵌入精度与可解释性，且可泛化到未见样本。

**🔧 技术方法**

使用神经网络蒸馏、Integrated Gradients、LRP、DeepLIFT 等特征归因技术，并实现 MEDALLION 与 MEDAL 两种变体。

**📊 数据集**

在 MNIST 64 像素数据、免疫细胞肿瘤单细胞 RNA‑seq、鼠脑单细胞 RNA‑seq 三个数据集上进行实验。

**📈 对比分析**

与 BIR、LIMEADE 等现有 DR 解释方法对比，MEDALLION 在与监督模型（RF、MLP）特征重要性相关性上表现更好，且能解释新样本且对低重建误差样本给出稳定归因。

**⚠️ 局限性**

局限在于解释的可靠性依赖于蒸馏精度，且仅揭示关联性而非因果关系，无法替代传统差异表达分析。

---

## 102. Random Symbolic Boosting: A Geometric Approach to Symbolic Regression

**arXiv ID:** 2611.08753 | [PDF](https://arxiv.org/pdf/2611.08753v1)

**作者:** José R. Ortegaa `[一作]`, Leonardo Trujillo `[通讯]` (Tecnológico Nacional de México/Instituto Tecnológico de Tijuana)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Random Symbolic Boosting（RSB）方法，利用随机生成的语法树与梯度提升构建可解释的符号回归/分类模型。

**💡 创新点**

创新点在于用随机树+提升取代传统进化搜索，得到显式加性符号模型，并引入基于角度的早停机制。

**🔧 技术方法**

采用随机语法树生成、弹性网（Elastic Net）回归、梯度提升、角度早停和fANOVA调参等技术。

**📊 数据集**

使用 SRBench 回归基准、TALENT 分类基准以及六个真实回归数据集（Concrete、EECooling、EEHeating、Housing、Tower、Yacht）。

**📈 对比分析**

与 XGBoost、FEAT、Operon、GP‑GOMEA 等黑盒与符号方法对比，SRBench 上排名第三、真实数据排名第一；在 TALENT 上排名第11/13，性能与 XGBoost/LGBM 相当，且处于精度‑训练时间 Pareto 前沿。

**⚠️ 局限性**

模型规模随提升次数增加而变大，导致最终表达式冗长，影响可解释性。

---

## 103. I Don't Care How You Hide It, I Care What You Do: Runtime Detection of Malicious ML Models

**arXiv ID:** 2611.08745 | [PDF](https://arxiv.org/pdf/2611.08745v1)

**作者:** Hala Ali `[一作]` (Virginia Commonwealth University), Andrew Case `[通讯]` (Volatility Foundation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ModelShield，利用运行时行为检测恶意ML模型

**💡 创新点**

首次将系统调用归因于模型加载/推理操作，并构建195个与MITRE ATT&CK对应的检测模式

**🔧 技术方法**

结合AOP框架API插桩、eBPF系统调用追踪以及LLM辅助模式发现

**📊 数据集**

使用PickleBall评测集、29个已公布的规避变体，以及145,000+来自Hugging Face、ModelScope、Kaggle的公开模型

**📈 对比分析**

与ModelScan、PickleScan、Fickling、ModelAudit和DynaHug对比，ModelShield实现零误报、55.95%召回率，平均扫描时长12.3秒，90%模型在35.5秒内完成

**⚠️ 局限性**

仅在Linux下工作，无法捕获依赖特定输入、环境或被动的恶意代码，且运行时分析可能错过未执行或未产生可观测行为的payload

---

## 104. On Differentially Private Mechanisms for Linear Regression

**arXiv ID:** 2611.08737 | [PDF](https://arxiv.org/pdf/2611.08737v1)

**作者:** Bardiya Aryanfard `[一作]` (Institute of Science and Technology Austria), Farhood Rostamkhani `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在连续观察（continual observation）下的无约束最小二乘线性回归，并给出第一套满足差分隐私的机制 ContinualAdaSSP；改进并重新证明 AdaSSP 的误差上界；在盒约束下证明纯 DP 与近似 DP 的多项式下界。

**💡 创新点**

①首次实现无约束线性回归的连续差分隐私机制；②提出自适应岭正则化的精确误差分析，修正了先前文献的错误；③通过指纹编码与 packing 论证盒约束下的最优下界。

**🔧 技术方法**

使用 AdaSSP 机制的自适应正则化；二叉树计数器与稀疏向量技术估计统计量 XᵀX、Xᵀy 和 λ_min；矩阵乘法加速（O(d^ω)）；指纹编码与 packing 技巧证明下界。

**📊 数据集**

三个真实数据集：Year、ACSIncome、Household。

**📈 对比分析**

将 ContinualAdaSSP 与 PQFTL、Batch、非私有 OLS、Trivial 等基线比较；实验显示 AdaSSP1/16 在大多数场景下误差最小，比最优非私有差距 2.45–50 倍，优于 PQFTL（甚至在早期也更好），Batch 误差最差。

**⚠️ 局限性**

限制：盒约束下的下界未扩展到无约束或连续观察设置；正则化参数 c 的理论上界过于保守，缺乏数据自适应选择；未覆盖完全动态（插入/删除）情形。

---

## 105. SEU Hardened Multi-Context Spintronic FPGA

**arXiv ID:** 2611.08738 | [PDF](https://arxiv.org/pdf/2611.08738v1)

**作者:** Abdolah Amirany `[一作]` (University of Florida), Tarek El-Ghazawi `[通讯]` (George Washington University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种具备多上下文切换的SEU硬化磁性FPGA架构，通过内部MRAM和LUT存储两/四个配置，消除对外部非易失性存储的依赖。

**💡 创新点**

创新点在于将C元素增强的预充电感测放大器（PCSA）与配置选择器相结合，实现SEU免疫与多上下文功能；并将传统SRAM单元替换为多上下文MRAM和LUT，从而显著降低面积和功耗。

**🔧 技术方法**

使用了MTJ磁隧道结、C元素、预充电感测放大器、配置选择器、40 nm CMOS工艺、Post‑layout仿真以及ROSSETA基准测试。

**📊 数据集**

采用ROSSETA benchmark suite，包括3D渲染、数字识别、垃圾邮件过滤、光流估计、二值化神经网络和人脸检测等六个子任务。

**📈 对比分析**

通过与先前SEU硬化设计和非SEU设计在面积、功耗、读/写延迟和PDP等指标上的对比，显示平均功耗降低22–50%，面积缩减15–35%，PDP提升40–60%；在ROSSETA基准中，2/4配置版本实现最低PDP和面积。

**⚠️ 局限性**

局限性在于资源仍有限，无法满足极大型复杂设计；当前实现仅支持两/四个配置，扩展到更多配置需要更多MTJ和面积；外围CMOS电路仍可能受到SEU影响。

---

## 106. FishSense Lite: A Camera-Based, Single Laser Hardware Framework with Software System for In Situ Fish Length Measurement

**arXiv ID:** 2611.08732 | [PDF](https://arxiv.org/pdf/2611.08732v1)

**作者:** Christopher L. Crutchfield `[一作]` (University of California, San Diego), Ryan Kastner `[通讯]` (University of California, San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在水下使用单一激光点和消费级潜水相机，开发了一套可在潜水员手中测量鱼体长度的硬件与软件系统。

**💡 创新点**

创新点在于将激光定位与相机单图像深度估计结合，避免传统立体摄像、声学或专用测量仪器的高成本与操作难度，实现在单张照片中即可得到公尺尺度的长度。

**🔧 技术方法**

技术主要包括激光三角测量、相机内参标定、光学模型回投射求距离、以及后期人工标注点位的长度计算。

**📊 数据集**

使用的数据集包含5个已知尺寸的刚性模型在实验池中的1,055帧图像以及7次海上潜水共73尾野生鱼的162帧图像。

**📈 对比分析**

与海底立体相机系统的对比显示两者在7尾鱼上的平均差异为+0.8%且无显著偏差；在已知尺寸目标上单帧误差低于1.3%，99%帧误差在15%以内。

**⚠️ 局限性**

局限在于需每次潜水手动校准激光姿态、对鱼体姿势（≤15°）和近距离（<1m）存在误差；单张图像长度受鱼体深度和标注噪声影响，且目前仅通过现场培训验证，未证实纯文字手册可行。

---

## 107. SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery

**arXiv ID:** 2611.08790 | [PDF](https://arxiv.org/pdf/2611.08790v1)

**作者:** Sunkyung Lee `[一作]` (Sungkyunkwan University), Gungor Polatkan `[通讯]` (Roblox Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个名为Steerable Generative Recommender (SteerGR) 的框架，利用稀疏自编码器 (SAE) 在不重新训练推荐器的前提下，实现对游戏推荐的概念级控制。

**💡 创新点**

创新点在于：①将 LLM 生成的游戏玩法概念与内部稀疏特征进行关联；②通过对比高低概念分组的 SAE 前激活量来发现候选特征，并按“对推荐的增益”选择最有效的 steering 特征；③在推理时通过编辑该特征的稀疏码实现放大或抑制，从而实现可解释且可调节的推荐控制。

**🔧 技术方法**

核心技术包括：1) 生成式推荐模型（Atomic ID 与 Semantic ID）+ Top‑K SAE；2) LLM 生成概念与文本嵌入对齐；3) 基于 Cohen’s d 的特征关联度评估；4) 按 ΔE_cj 选择 steering 特征；5) 稀疏特征编辑（放大/抑制）并在解码后直接修改隐藏状态。

**📊 数据集**

使用了 2026 年在大规模游戏平台收集的真实交互日志，包含约 40K 款游戏和 7K 个评估用户；SAE 训练使用 2026 年 8 月的每日快照。

**📈 对比分析**

通过对 12 个 LLM 生成的玩法概念进行控制实验，衡量目标概念在前 10 推荐中的占比（Target Share）以及 Recall@10 / NDCG@10。结果显示：在 Atomic ID 模型中，放大能提升目标概念占比约 113% 并将概念匹配用户的 Recall@10 提升 2.4%；抑制可使占比下降约 42%；但过强的干预（α=8）会导致整体准确率下降 7–43%。在 Semantic ID 模型中增益和下降幅度均较小，但趋势一致。

**⚠️ 局限性**

局限性：①仅评估了 12 个由 LLM 生成、基于游戏描述的概念，未覆盖如流行度、稀缺性等多样化控制目标；②仅使用 Top‑K SAE 结构，未尝试更复杂的 SAE 或自编码器变体；③实验仅在游戏推荐场景，未验证跨领域可迁移性。

---

## 108. I Think, Therefore I Attack: Evaluating Thought Injection Attacks on Reasoning Agents

**arXiv ID:** 2611.08811 | [PDF](https://arxiv.org/pdf/2611.08811v1)

**作者:** Parth Illendula `[一作]` (University of Texas at Austin), Alex Liu `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对大规模推理模型（LRM）中的思考注入（thought injection）攻击进行两部分研究，探究不同注入方式对模型的影响以及在代理环境中的攻击效果。

**💡 创新点**

发现注入效果高度依赖于目标模型的思考标签结构和模型自身的“思考语音”，且自我语音+思考标签能在代理攻击中实现高达50%的成功率，而普通提示注入几乎无效。

**🔧 技术方法**

使用Chain-of-Thought（CoT）框架、思考标签（⟨ think ⟩）包装、模型自身的推理文本作为注入语音，结合容器化的代理仿真环境和工具调用机制。

**📊 数据集**

构造的实验数据集包括：①合成金融数据库（150条PII记录）；②基于Docker的代理沙箱环境，包含三台机器、隐藏的CTF风格标志、内部git服务器等。

**📈 对比分析**

对3个开源大模型（GLM-5.3、MiniMax-M2.5、Qwen3.8系列）进行6,000次单轮注入实验和1,200次多轮代理实验；结果显示：在单轮实验中，思考标签能将GLM-5.3的攻击成功率从3.8%提升至13.4%；在代理实验中，使用自我语音+标签在MiniMax上实现42%/50%的逆向Shell/后门成功率，普通用户语音注入始终失败。

**⚠️ 局限性**

局限性包括：①只测试有限数量的模型和攻击目标；②自我语音注入基于已知模型的推理文本，真实攻击中可能难以获取；③未对不同上下文长度和多任务情境进行系统评估；④缺乏统一的语音相似度度量和对抗检测方法。

---

## 109. TopoSHAP: Shapley for Explainable and Efficient Topological Deep Learning

**arXiv ID:** 2611.08789 | [PDF](https://arxiv.org/pdf/2611.08789v1)

**作者:** Mathilde Papillon `[一作]` (University of California, Santa Barbara), Nina Miolane `[通讯]` (University of California, Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的游戏理论解释框架TopoSHAP，既能解释拓扑深度学习模型的预测，也能解释模型的性能。

**💡 创新点**

创新点在于将Shapley值应用于所有秩的细胞和邻域，推出 -cell 与 -neighbor 两种游戏，既能快速得到细胞级解释，又能用邻域贡献进行架构搜索。

**🔧 技术方法**

主要技术包括组合复杂体建模、拓扑神经网络(GCCN/HOPSE)、Shapley值近似、贪心子集搜索以及逆向消除的架构选择。

**📊 数据集**

使用GraphXAI分子基准、TopoBench的提升版本以及原始的MANTRA simplicial complex数据集。

**📈 对比分析**

与现有GNN解释器(SubgraphX、PGExplainer等)比较，-cell 在解释准确率上与或优于基线，速度可达112倍；-neighbor 在架构搜索上可将训练成本降低4.5–11.3倍，同时保持甚至提升精度。

**⚠️ 局限性**

局限性包括对大规模复杂体的评估成本高（需 O(n^2) 评估），以及缺乏通用组合复杂体的标注解释基准，限制了验证效果。

---

## 110. Jacobian-Causal Memory Control for Persistent Memory in Language Model Systems

**arXiv ID:** 2611.08779 | [PDF](https://arxiv.org/pdf/2611.08779v1)

**作者:** Marvin Danig `[一作]` `[通讯]` (Achiral Research), Marvin Danig (Achiral Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于雅可比-因果内存控制（JCMC）的持久化内存管理架构，利用影响签名来预测并限制内存介入对模型行为的影响，并在观察到与预测不一致时执行回滚和版本追溯。

**💡 创新点**

创新点在于将内存检索从简单的相关性排序转变为可逆、可观测、可追溯的因果干预；定义影响签名、目标效应对齐、侧效应预算以及观察-回滚循环；并给出了局部近似误差、可达性、选择亏损与多内存交互的理论上限。

**🔧 技术方法**

核心技术包括：
- 影响签名估计（基于梯度、积分梯度或有限差分）
- 目标效应与侧效应预算的数值优化选择器
- 配对对照观察（同随机种子/相同采样）
- 回滚与血统管理机制
- 理论分析（局部线性误差、可达性投影、回退后收益等）。

**📊 数据集**

本工作未使用任何公开数据集或实验基准，仅提供了一个可复现的评估协议，未来实验需在自建或公开的持久化内存框架（如 Cognoscenti）上进行。

**📈 对比分析**

比较方法：与稠密/混合 top‑k 检索、图检索、策略过滤、仅重排序等基线进行匹配对照；指标包括目标效应对齐度、侧效应大小、答案准确率、祖先完整性、回滚精度与恢复质量、校准误差、延迟和 token 成本。由于缺乏实验结果，性能尚未评估。

**⚠️ 局限性**

局限性：
- 仅在局部平滑、微小扰动条件下有效，无法保证对大幅变动的内存干预。
- 影响签名需在每个模型版本、提示、工具接口下重新估计，易失效。
- 观察和回滚机制仅能防止未来错误，已产生的影响不可逆。
- 多内存交互非线性，单一签名可能失效。
- 由于无实验验证，无法确定与传统检索基线的实际优势。

---

## 111. Semantic Conditional Similarity (SCS): A Formal Mathematical Framework

**arXiv ID:** 2611.08778 | [PDF](https://arxiv.org/pdf/2611.08778v1)

**作者:** Mehmet Murat Albayrakoğlu `[一作]` `[通讯]`, Mehmet Murat Albayrakoğlu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在多源文本与单目标文本间消除源间共用内容后计算余量相似度的Semantic Conditional Similarity（SCS）方法。

**💡 创新点**

创新点在于引入方向性、基于三元组的条件化、源端与目标端的交互式去重以及阈值无关的集合分解。

**🔧 技术方法**

利用词形还原后的词集、集合运算（交集、补集）以及二进制向量的余弦相似度实现。

**📊 数据集**

未使用真实语料，实验仅采用随机生成的字母集合（Python seed 5663）做演示。

**📈 对比分析**

通过比较SCS与传统STS的排名结果，展示了排名翻转现象，证明SCS能揭示源的边际贡献，但未给出数值性能指标。

**⚠️ 局限性**

局限在于二值化忽略词频，需文档长度足够且词汇重叠度适中；对极短文本或完全重叠文本效果有限。

---

## 112. Gain-Dependent Entropy Characterization of a Zener Avalanche Noise TRNG via Dual NIST Standards: Toward Low-Cost Hardware Entropy for ICS Deployments

**arXiv ID:** 2611.08774 | [PDF](https://arxiv.org/pdf/2611.08774v1)

**作者:** Rishabh Das `[一作]` `[通讯]` (Ohio University), Rishabh Das (Ohio University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于齐纳隧穿噪声的TRNG进行系统的熵评估，探索放大器增益和耦合电容对熵质量和吞吐量的影响，并在工业控制系统中评估其可行性。

**💡 创新点**

首次在同一电路拓扑下同时使用NIST SP 800-22和SP 800-90B两大标准对大规模数据集（每个配置1.44×10⁹比特）进行完整评估，发现压缩估计器是最关键的熵约束，提出耦合电容从0.1µF到0.01µF的被动改动即可实现Pareto改进。

**🔧 技术方法**

采用MCP6002双运算放大器的Zener隧穿噪声源，配合MT3608升压模块和RC高通滤波，利用DSLogic逻辑分析仪、Analog Discovery 3等实验仪器，并使用Double Von Neumann去偏和Python实现的NIST测试套件。

**📊 数据集**

实验数据集包含六种配置（五种增益从G=11到G=101，及一次电容从0.1µF变为0.01µF），每个配置采集1.44×10⁹原始比特，后进行120倍去相关抽样，生成12×10⁶有效比特。

**📈 对比分析**

通过SP 800-22十五项测试和SP 800-90B十项熵估算器，发现G=48配置在两标准下均达到最佳：SP 800-22 15/15通过、压缩估计器熵0.801 b/b、IID通过、吞吐量11 568 b/s，且成本低于5美元；而G=11表现出IID失败、熵不足，确认最低可行增益边界。

**⚠️ 局限性**

局限性包括：仅在室温下测试，未对温度漂移、长期稳定性和不同齐纳管型号进行评估；去偏仅使用双Von Neumann，未尝试更高效的熵调制方法；对高频信号的解析受限于采样速率；实际工业部署需进一步验证与PLC扫描周期兼容性。

---

## 113. StarLoc: A Practical Approach to Encoding Geographic Locators in IPv6 for Starlink

**arXiv ID:** 2611.08772 | [PDF](https://arxiv.org/pdf/2611.08772v1)

**作者:** Vanya Singla `[一作]` (University of Victoria), Jianping Pan `[通讯]` (University of Victoria)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析了 Starlink IPv6 的地址分配与 GeoFeed 的缺陷，提出并实现了 StarLoc——一种在 IPv6 /56 之上嵌入 H3 空间单元的地理定位方案，保持现有前缀、/56 分配与稳定端点标识符。

**💡 创新点**

创新点在于将定位单元（H3 cell）作为 Locator 嵌入已有 ISP 前缀体系，使用可调分辨率和需求感知的单元压缩，既实现高精度定位（<10 km）又仅增加 12–14 位 Locator，满足移动性与隐私控制；并通过每前缀代码表实现无额外映射/路由层的直接解码。

**🔧 技术方法**

核心技术包括 H3 地理网格、分辨率字段与 9‑bit 索引编码、需求感知的层级合并与索引分配算法、基于前缀的代码表发布与移动端点的 Locator 变更策略。

**📊 数据集**

使用的数据集有 925 次 GeoFeed 快照（2024–2026）、302 次 PoP 反馈快照、7.9 M 条 IPv6 活跃路由器地址、BGP 通告以及多国 RIR 分配信息。

**📈 对比分析**

对比方法：以 PoP‑基定位（误差 275–2203 km）为基线，采用 1,000 个采样位置计算平均误差；StarLoc 在 r=5 级别下将误差降至 6 km，提升约 141 倍，同时仅需 12–14 位 Locator；在 8 个 PoP 区域内，利用需求感知压缩后仅占用 9–11 位索引。性能上既实现了高精度，又保持了地址空间紧凑。

**⚠️ 局限性**

限制包括：需要运营商维护和发布前缀代码表，分辨率受隐私阈值与区域政策限制，极端稀疏地区仍可能使用粗糙单元导致定位误差；移动性处理需要额外的 Hysteresis 与 Make‑Before‑Break 机制，可能导致短暂的地址不一致；隐私与可定位精度之间的权衡尚未在大规模真实场景中彻底验证。

---

## 114. AtariBench: Evaluating Spatiotemporal Reasoning of Vision-Language Models in Atari Gameplay

**arXiv ID:** 2611.08755 | [PDF](https://arxiv.org/pdf/2611.08755v1)

**作者:** Mingjia Huo `[一作]` (University of California San Diego), Jiaxi Tang `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AtariBench——一个基于 Atari 2600 的动态时空决策基准，用来评估 Vision‑Language 模型在高频游戏环境中的视觉感知、运动推理与闭环控制能力。

**💡 创新点**

创新点在于：①将快速游戏动态与模型推理时间解耦，使用固定 30 秒时间窗口；②设计结构化记忆（短期历史 + 关键奖励片段）与 Append‑Only 两种提示模式；③引入可分离性分析（效应大小、ANOVA 等）确保游戏能区分模型；④提供轻量级、可重复的闭环 harness，便于模型迭代。

**🔧 技术方法**

技术上使用了 ALE + Gymnasium 接口、VLM 生成思考与动作序列、图像+文本混合提示、结构化记忆缓存、思考努力阈值、分离性分析统计、VLM‑judge 诊断以及多模型并行评测。

**📊 数据集**

数据集为 20 款 Atari 2600 游戏（如 AirRaid、Assault、PacMan、Breakout 等），通过 ALE/Gymnasium 统一接口。

**📈 对比分析**

方法上在相同 harness 下，对 Gemini、GPT、Claude、DeepSeek、GLM 等多种 VLM 与 RL 代理和随机 baseline 进行 10 次 30 秒闭环运行，比较平均奖励和排名。实验显示 VLM 仍显著落后专业 RL 代理，随机 baseline 基线低；分离性分析表明 18/20 游戏能有效区分模型。

**⚠️ 局限性**

局限性包括：VLM 在运动感知、时空推理、连续决策和长上下文利用上表现不足；过度依赖文本历史导致错误循环；benchmark 仅涵盖 2D Atari，未覆盖更复杂或真实环境的动态视觉任务。

---

## 115. Driver Takeovers and Longitudinal Responses Analysis from Naturalistic Driving

**arXiv ID:** 2611.08798 | [PDF](https://arxiv.org/pdf/2611.08798v1)

**作者:** Yuhang Wang `[一作]`, Hao Zhou `[通讯]` (University of South Florida)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

收集并分析了大规模自然驾驶ADAS交接事件，评估司机干预与车辆后续纵向安全情况

**💡 创新点**

提供首个事件级同步视频与传感器数据集ADAS-TO，并将Gazis–Herman–Rothery（GHR）模型作为人类反应参考来关联交接前的迫近感与交接发生

**🔧 技术方法**

采用GHR车随模型、Looming分析、逻辑回归、分层样条等方法

**📊 数据集**

ADAS-TO数据集，包含16,446个交接片段，364名司机，99车型

**📈 对比分析**

将模型拟合误差与IDM、Gipps比较，五秒间距RMSE 1.34 m；使用离散化后逻辑回归，AUROC从0.756提升到0.765，表明模型能小幅提升交接预测

**⚠️ 局限性**

数据覆盖不均，缺少长时间连续驾驶，交接时点不总对应司机首动，未评估系统持续激活时冲突发展

---

## 116. ReKV: Overlapping Quantized KV Transfer and Suffix Recomputation

**arXiv ID:** 2611.08770 | [PDF](https://arxiv.org/pdf/2611.08770v1)

**作者:** Guangjian Dong `[一作]` `[通讯]` (New York University), Guangjian Dong (New York University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在预填充-解码 (P/D) 分离场景下的 KV 缓存迁移策略 ReKV，通过在预填充端将前缀 KV 缓存量化并压缩，同时在解码端重算后缀层，从而在保持模型输出质量的前提下显著降低首词延迟。

**💡 创新点**

创新点在于将低位量化和后缀重算结合成一种联合决策机制，并通过分析层边界与精度对通信与计算的交互，给出了在不同模型、提示长度和链路带宽下的最优切点选择。

**🔧 技术方法**

技术上使用了 FP16 缩放的 4/8 位 KV 量化、BF16 边界状态传输、并行 transfer‑recompute 调度以及量化恢复，配合 GPU 上的分层 Transformer 推理；同时通过自定义的交叉节点 H100‑H100 通道和 L40S 直连路径进行实验。

**📊 数据集**

实验数据集包括：合成状态与层权重（用于压缩/恢复验证）、真实权重模型的 WikiText‑2、NIAH 需求集以及 64 题 GSM8K 子集，用于评估输出分布、困惑度和推理准确性。

**📈 对比分析**

在 8K 令牌长、三种模型（Qwen2.5‑7B、3B、Qwen3‑8B）上，ReKV 的联合 Q8+重算策略相较于全 BF16 传输实现了 2.36×、2.84×、3.65× 的首词延迟提升；相较于最优的单一 Q8 传输或 BF16 重算方案分别提升 1.17×、1.42×、1.22×；且在所有验证集上均满足 1% perplexity 约束。

**⚠️ 局限性**

局限性包括：在短提示或更快链路上不一定最佳；低位 Q4 量化对大多数模型质量影响过大；实验仅在两台 GPU 的分离 P/D 框架下验证，未覆盖完整的 Q8 解码与分页缓存集成；且跨模型、跨链路的最优策略需进一步在线自适应。

---

## 117. How are viral trends different from educational and commercial uses of LLM image generation and editing? A privacy-preserving descriptive analysis

**arXiv ID:** 2611.08749 | [PDF](https://arxiv.org/pdf/2611.08749v1)

**作者:** Will Tucker-Ray `[一作]` (Google Behavioral Economics), Yurii Sushko `[通讯]` (Google Privacy, Security, & Safety)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过差分隐私方法，对2026年第一季度全球用户在Gemini上使用Nano Banana生成与编辑图像的会话进行聚类和特征分析，揭示多领域使用模式。

**💡 创新点**

首次将差分隐私聚类、关键词提取与多维特征评分相结合，实现端到端隐私保护下的LLM图像生成使用情况大规模可视化。

**🔧 技术方法**

采用Urania隐私框架、差分隐私聚类、关键词抽取、多维特征评分等技术，对会话文本进行匿名化分析。

**📊 数据集**

采集约210万条Gemini会话（520万条提示），涵盖多语言、多国家的用户数据，构成分析样本。

**📈 对比分析**

未与外部基准直接对比，主要通过内部聚类和特征统计展示结果；整体隐私预算为ε≤0.04，保证统计结果的匿名性。

**⚠️ 局限性**

局限包括：仅基于文本提示推断意图，无法验证生成图像质量；样本仅为一季度快照；聚类与特征识别存在噪声；缺乏用户旅程与多会话跟踪信息。

---

## 118. On Lipschitz Explosion in Deep Neural Networks with Normalization: Consequences for Optimization and Adversarial Robustness

**arXiv ID:** 2611.08775 | [PDF](https://arxiv.org/pdf/2611.08775v1)

**作者:** Ashkan Soleymani `[一作]` (Massachusetts Institute of Technology), Patrick Jaillet `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探究了批量归一化层在深度网络中的Lipschitz常数指数爆炸现象，并分析了其对优化、泛化与鲁棒性的影响，随后提出概率Goldstein收敛概念并在MNIST、Fashion‑MNIST、CIFAR‑10上验证低秩结构对抗鲁棒性差异的几何机制。

**💡 创新点**

创新点在于证明即使每层权重满足2‑范数约束，批量归一化仍能导致输入和参数的Lipschitz常数随深度指数增长；提出概率Goldstein stationarity并给出不依赖Lipschitz常数的算法；揭示低秩与秩增噪声对对抗鲁棒性差异的几何机理。

**🔧 技术方法**

理论上利用批量归一化与低秩矩阵交互构造指数上界，并用概率论证明；实验上在MNIST、Fashion‑MNIST、CIFAR‑10中比较不同噪声结构；通过数值评估输入与参数的Lipschitz比例。

**📊 数据集**

使用数据集：MNIST、Fashion‑MNIST、CIFAR‑10，并在预训练ResNet‑56上进行对抗噪声实验。

**📈 对比分析**

比较方法：对比随机局部扰动与构造 witness 在输入Lipschitz增长上的差异；在相同模型下对比不同噪声结构的对抗性能，实验表明秩增噪声导致准确率、置信度、边缘急剧下降，而同子空间噪声影响较小。

**⚠️ 局限性**

局限性：证明主要针对线性激活或特定网络结构；对非线性激活的严格分析留待未来；实验集中在小规模数据集和简化网络，尚未在大型模型上进行验证。

---

## 119. Dynamic signatures of collective knowledge through navigation in a network with semantic structure

**arXiv ID:** 2611.08812 | [PDF](https://arxiv.org/pdf/2611.08812v1)

**作者:** Abby Bertics `[一作]`, Jean M. Carlson `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过分析70,770条 Wikispeedia 游戏中的玩家轨迹，提出并应用 Flow Tree 这一工具，将大量个体导航路径压缩为描述离开/进入某篇文章时轨迹分裂和合并的层次结构。然后将人类 Flow Tree 与四种基于图结构或语义信息的随机游走模型以及一个基于第一阶点击偏好的历史无关模型进行比较，探索人类在无通信情况下如何共享知识。

**💡 创新点**

创新点包括：
1) 将 Flow Tree 概念从路径级别推广到文章级别，能够捕捉人类在不同文章上的集体导航动力学；
2) 通过对比多种随机游走模型，定量区分图结构、语义统计与人类导航之间的差异；
3) 通过残差坐标构建文章四种功能类型（Beacon、Funnel、Fork、Corridor），揭示人类对文章的使用方式，并证明这些类型无法由任何随机模型解释；
4) 证明人类导航的集体知识包含超越单一步骤点击偏好的时序结构。

**🔧 技术方法**

使用的技术包括：
- Flow Tree 构建与五维树特征（节点数、叶子数、深度、直径、平均分支因子）统计；
- 多种随机游走模型（无偏、基于度、基于词频、基于目标相似度）以及基于人类点击统计的历史无关模型；
- 线性回归与 R² 评估度数对 Flow Tree 形状的解释力；
- 相关性分析（Pearson、Steiger z 检验）、Welch t 检验、Cohen d 效果量、Benjamini–Hochberg 多重检验校正；
- 余差坐标绘制与四象限划分得到文章类型；
- 通过多重随机拆分评估结果稳定性。

**📊 数据集**

使用的数据集是来自 Stanford SNAP 的 Wikispeedia 公开数据集，包括 4,592 篇文章、119,881 条有向边，以及 70,770 条玩家从随机起点到目标的游戏轨迹。数据在 2025 年 10 月 29 日获取。

**📈 对比分析**

比较方法：将人类 Flow Tree 与随机游走模型的五维特征进行 Welch t 检验与 Cohen d 计算，评估深度、直径、分支因子等指标的差异。对度数解释力进行 R² 统计并与人类结果比较。对残差坐标的四象限类型进行 Cohen κ 一致性检验。结果显示：
- 人类 Flow Tree 在深度、直径上显著更大，分支因子更小（p<0.001，Cohen d>0.8）；
- 与任何随机模型相比，人类的 R² 约为 20‑30%（相比随机模型的 50‑70%），表明约 70‑80% 的人类导航结构不由单纯图度解释；
- 历史无关模型的效果介于随机模型与人类之间，说明第一阶点击偏好只能解释部分人类导航结构。

**⚠️ 局限性**

限制：
1) 仅聚合了大规模玩家数据，缺乏个体层面的轨迹，无法区分不同玩家策略；
2) Flow Tree 的构建需设定轨迹长度与采样数量，影响结果的稳健性；
3) 反向 Flow Tree 的构建方式与随机模型不完全对等，可能引入偏差；
4) 随机游走模型仅基于简单的度或词频权重，未考虑页面布局、链接位置等实际点击因素；
5) 语义模型使用标题嵌入，未捕捉页面正文信息；
6) 数据集仅来自少数玩家，且 IP 哈希仅为粗略个体标识，可能掩盖活跃玩家的影响。

---

## 120. Impact of color in computer and human-based object recognition under occlusion

**arXiv ID:** 2611.08808 | [PDF](https://arxiv.org/pdf/2611.08808v1)

**作者:** Courtney M. King `[一作]` (Fordham University), Daniel D. Leeds `[通讯]` (Fordham University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究构建了包含对象色与场景色遮挡的专门数据集，并对人类参与者及多类卷积网络与变压器模型进行识别实验，提出了非遮挡精度（NOP）、非遮挡召回率（NOR）与遮挡区激活率（AIO）三种新指标，用于定量评估遮挡下的激活与表示差异，同时探索了基于图像相关颜色的 Cutout 训练增强方法。

**💡 创新点**

创新点包括：①首次系统比较对象色与场景色遮挡对人类与多种计算机视觉模型的影响；②提出针对遮挡情况的三种激活度量指标，为模型可解释性提供量化工具；③将图像相关颜色用于数据增强，显著提升模型对未知遮挡的鲁棒性；④提供可复现的遮挡生成库与完整实验代码。

**🔧 技术方法**

采用 K‑means 颜色聚类提取遮挡颜色；使用 Grad‑CAM++ 生成激活热图并计算 NOP/NOR/AIO；人类实验使用 PsychoPy 进行快速两选对齐任务；模型评估基于预训练的 ResNet、VGG、GoogLeNet、ViT、Swin、MaxViT 等；训练增强使用 Cutout（黑色与图像相关颜色）方案。

**📊 数据集**

核心数据集为 ImageNet 验证集中的 80 张单目标图像，按对象、场景和黑色三种颜色生成遮挡版本；Tiny‑ImageNet 用于训练与测试 Cutout 增强效果；此外还构建了针对不同遮挡方向（水平、垂直、对角）的子集。

**📈 对比分析**

通过人类 165 名参与者的 150ms 速度识别实验与模型在 1000 类分类上的准确率对比。人类在原始图像 93% 识别率，遮挡后降至 85–87%；CNN 在中性遮挡下表现最差，但在场景色遮挡时性能提升；变压器与对象/场景遮挡呈人类相似趋势，效应相对较小。Cutout‑Color 增强在 ResNet18 上提升整体准确率约 1.3% 并显著增强对图像相关颜色遮挡的鲁棒性。

**⚠️ 局限性**

主要局限在于遮挡实验仅使用 80 张图像，且颜色提取需人工微调；实验仅关注颜色遮挡，未探讨纹理或更复杂遮挡；指标仅基于 Grad‑CAM++ 生成的热图，未验证在其他可视化方法下的稳健性；小样本导致结果可能不具普适性。

---

## 121. Generative Search and Recommendation at Shopify

**arXiv ID:** 2611.08804 | [PDF](https://arxiv.org/pdf/2611.08804v1)

**作者:** Juhana Kangaspunta `[一作]` (Shopify), Akshay Soni `[通讯]` (Shopify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并部署了一种生成式搜索与推荐模型，利用语义ID生成候选并与轻量级排序器结合，在Shopify平台实现高质量、低延迟的产品检索与推荐。

**💡 创新点**

将搜索和推荐统一为语义ID生成任务，结合多粒度token化、联合预训练与任务特定后训练（RL、教师蒸馏），并设计可在日常变化目录下快速约束生成的索引技术。

**🔧 技术方法**

语义ID分词（残差K‑means量化）、多token预测（MTP）与自回归（AR）生成器、轻量级LLM backbone（LFM2‑2.6B）、强化学习、教师蒸馏、可扩展约束索引、CUDA插件并行推理等。

**📊 数据集**

Shopify商品目录（数十亿商品）属性数据、真实搜索会话日志、LLM判断生成的人工标注、用户行为序列（点击、加入购物车、购买）、公开文本、代码与数学数据等。

**📈 对比分析**

采用离线指标（搜索Top10相关性、购买Recall10、推荐Recall50）和在线A/B测试；在搜索上与现有多模型栈相当，加入后提升Top10相关性与购买Recall；在推荐上单独生成器提升pool recall 2.65pp，A/B测试订单增长7.95%、转化率+9.85%。

**⚠️ 局限性**

仍需手动维护语义ID映射，对新商品变化需重建索引，推荐模型在多商品共享ID时可能误选，RL与蒸馏训练复杂且资源占用高，目前仅支持文本与ID，不具备多模态理解，且在线推理受GPU负载限制。

---

## 122. Scale as a Modality: Multi-Scale Pixel-Space Diffusion Transformers

**arXiv ID:** 2611.08802 | [PDF](https://arxiv.org/pdf/2611.08802v1)

**作者:** Ruibo Chen `[一作]` (TikTok), Jian Du `[通讯]` (TikTok)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现一种多尺度像素空间扩散变换器（MS‑pDiT），通过将同一噪声图像在不同像素块大小下并行分割并分别处理，随后通过压缩后联合注意力实现各尺度之间高效双向信息交换，最终直接在像素级别生成图像；同时支持将文本视为另一种模态以实现文本到图像生成。

**💡 创新点**

创新点主要有：①将尺度视为并行模态；②使用压缩后的 MM‑DiT‑style 联合注意力实现跨尺度双向交互，注意力成本与最细尺度无关；③细尺度流在网络后期才引入，先构建全局结构再进行细化；④该框架可自然扩展至文本条件，实现像素级文本到图像；⑤通过尺度维度扩展验证了“尺度”作为新型扩展维度的有效性。

**🔧 技术方法**

技术手段包括：像素级扩散模型与流匹配损失；MM‑DiT式压缩后联合注意力（共享2D RoPE、尺度身份嵌入、SwiGLU、RMSNorm）；可变尺度投影与可扩展的多尺度 Transformer；REPA 损失对齐粗尺度特征；文本投影与 1D RoPE；多尺度训练策略与 late‑anchor 深度调度。

**📊 数据集**

数据集：ImageNet（256×256 与 512×512）用于类别条件生成；DrawBench 与 PartiPrompts 用于文本到图像生成；评估指标包括 FID、sFID、Inception Score、Precision/Recall、GenEval、DPG‑Bench。

**📈 对比分析**

与现有 Latent‑空间与 Pixel‑空间扩散模型进行对比。MS‑pDiT 在 ImageNet 256×256 上的 -H（五尺度）实现 FID 1.43、sFID 4.13、IS 304.6；-XL（四尺度）实现 FID 1.50、sFID 4.31；在 512×512 上实现 FID 1.44、IS 319.2，显著优于 PixelDiT、DeCo、PixelU 等同类模型；文本生成方面，GenEval 0.86、DPG 83.9，优于 PixelDiT‑T2I、PixNerd‑T2I 等。

**⚠️ 局限性**

局限性：①模型需要手动设定尺度与深度分配，缺乏自动化方法；②训练与推理成本仍高，尤其在更大分辨率或更多尺度时；③当前只评估了 ImageNet 与少量文本提示，未验证在更广泛多模态任务中的鲁棒性；④压缩投影与共享 RoPE 的设计对极大图像或不同视觉语义可能需要调整；⑤缺乏对尺度维度扩展的理论分析。

---

## 123. On the Few-Shot Adaptation of Vision-Language Models: A Few Gradient Signs Are All You Need

**arXiv ID:** 2611.08800 | [PDF](https://arxiv.org/pdf/2611.08800v1)

**作者:** Nairouz Mrabah `[一作]` (ETS Montreal), Eric Granger `[通讯]` (Concordia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两种无状态的稀疏优化器SignSparseSGD和DynSignSparseSGD，用梯度幅度筛选稀疏更新坐标并用固定幅度符号更新参数。

**💡 创新点**

创新点在于把梯度幅度和符号分离：幅度仅用于选取重要坐标，符号直接驱动更新，无需第一/二阶矩；动态支持刷新提高适应性。

**🔧 技术方法**

使用梯度稀疏化、固定幅度符号更新、blockwise top-k选取、无moment缓冲，基于CLIP预训练模型的可视语言模型适配。

**📊 数据集**

在11个图像分类数据集（ImageNet、SUN、Air、EuroSAT、Cars、Food、Pets、Flowers、Caltech101、DTD、UCF101）上评估，使用ViT-B/32、B/16、L/14三种backbone。

**📈 对比分析**

与多种基线对比（Prompt-tuning、LayerNorm、SAFT、LoRA、GaLoRE、SO等），平均精度在A2A/B2N任务中均高于或匹配最强基线，DynSignSparseSGD平均A2A精度最高，长期训练和跨数据集停止点更稳健。

**⚠️ 局限性**

局限在于仅针对CLIP图像分类任务，需重新校准符号步长；在极长训练或极低稀疏率时可能性能下降；对非视觉模态或其他架构的推广需进一步研究。

---

## 124. MotionBench-XAI: Benchmarking What Shapes Shapley Explanations in Spatiotemporal Machine Learning

**arXiv ID:** 2611.08797 | [PDF](https://arxiv.org/pdf/2611.08797v1)

**作者:** Mathilde Papillon `[一作]` (University of California, Santa Barbara), Nina Miolane `[通讯]` (University of California, Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MotionBench‑XAI基准，系统评估时空数据Shapley解释方法，涵盖合成与真实数据、玩家划分、填充策略与评价指标；

**💡 创新点**

在四个设计维度（玩家、填充、游戏、度量）上统一实验，提供闭式真值合成数据，阐明玩家选择和生成填充对解释质量的决定性影响，并给出实用指南；

**🔧 技术方法**

使用KernelSHAP估计器、VAEAC/Flow/高斯生成填充、时间窗口/空间/细粒度玩家划分，以及游戏匹配与代理指标评价；

**📊 数据集**

实验集包括五种自定义合成时空数据（Gaussian、Burr、Skeleton、Wave、SkeletonGait）与三份公开真实数据（CARE‑PD步态、PTB‑XL ECG、ESC‑50音频）；

**📈 对比分析**

在相同样本与估计器设置下，对每种玩家与填充组合计算MAE/相关或代理指标；结果显示玩家划分决定解释性，生成填充在条件精度足够时优于基线；在真实数据代理指标互相矛盾，最佳方法随数据集变化；

**⚠️ 局限性**

仅覆盖有限的高斯/Copula合成模型，玩家数≤68，未扩展到更细粒度或更复杂网络，真实数据结果受模型检查点限制。

---

## 125. Online Bin Packing with Dispatch Deadlines

**arXiv ID:** 2611.08792 | [PDF](https://arxiv.org/pdf/2611.08792v1)

**作者:** Pei-Lin Chiang `[一作]` (National Tsing Hua University), Li-Tzu Yang `[通讯]` (National Tsing Hua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在线装箱问题的一个扩展——带调度截止日期的在线装箱（Online Bin Packing with Dispatch Deadlines）

**💡 创新点**

创新点在于：① 将经典的在线装箱算法（Next‑Fit、Worst‑Fit、Any‑Fit、First‑Fit 等）直接移植到有截止日期的场景，并给出其竞争比；② 通过“输入归约到规范形式”与“懒装箱”技术，精确证明 First‑Fit 的竞争比为 16/7；③ 构造一种复杂的自适应对手，证明任何确定性算法的竞争比下界为 11/5，缩小已知区间。

**🔧 技术方法**

主要技术手段包括：
- 输入归约与规范化（Canonical Form）
- 懒装箱与紧急关闭分析
- 区域（area）与不相交项的计数论证
- 自适应对手（adversary）构造与动态截止时间的更新规则
- 递推与不等式分析

**📊 数据集**

该工作完全基于理论分析，无实验或公开数据集；所有结论通过构造输入序列与对手演示得到。

**📈 对比分析**

比较方法：对比各经典算法在带截止日期环境下的竞争比。结果显示：Next‑Fit 与 Worst‑Fit 竞争比为 3；Any‑Fit 类算法竞争比在 16/7 与 3 之间；First‑Fit 精确竞争比为 16/7；最优确定性算法的理论下界为 11/5。可见，虽然引入截止日期导致竞争比提升，但经典算法仍保持相对稳健。

**⚠️ 局限性**

局限性：
- 上界 16/7 与下界 11/5 之间仍存在显著差距；
- 分析仅针对确定性算法，对随机或带信息的算法未给出结论；
- 算法本身未利用截止日期信息，表明可能存在更优策略但未被探索；
- 仅考虑单元容量箱，未讨论多容量或多类箱的扩展。

---

## 126. Not All Latents Are Equal: Importance-Weighted Latent Graph Generation

**arXiv ID:** 2611.08788 | [PDF](https://arxiv.org/pdf/2611.08788v1)

**作者:** Biraaj Rout `[一作]` (University of Texas at Arlington), Harry Shomer `[通讯]` (University of Texas at Arlington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对图生成中的潜在空间进行重权重处理，改进潜在扩散模型的性能和收敛速度

**💡 创新点**

提出基于解码器雅可比矩阵的潜在重要性重权重方法（LIR），仅通过修改损失函数即可提升模型

**🔧 技术方法**

潜在图自编码器、流匹配/扩散模型、雅可比矩阵重要性计算与加权MSE损失

**📊 数据集**

合成数据集（Planar、Tree、SBM）和分子数据集（MOSES、GuacaMol）

**📈 对比分析**

与未加权潜在模型（VGAE‑Flow、LG‑Flow）以及离散扩散方法对比，LIR在合成数据集提升约20‑30%准确率，收敛速度提升1.6‑3.4倍，分子数据集同样出现显著改善；采样时间无显著变化

**⚠️ 局限性**

仍依赖现有潜在模型，若潜在表示本身分布不均衡，LIR 的效果受限，未来需设计更均衡的潜在编码

---

## 127. Extrapolation Matters: Sparse-Only Depth Completion Beyond LiDAR Coverage

**arXiv ID:** 2611.08782 | [PDF](https://arxiv.org/pdf/2611.08782v1)

**作者:** Maximilian Jansen `[一作]` (Cariad Se), Praveen Kulkarni `[通讯]` (Bosch)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种仅使用单帧RGB图像和原始单次LiDAR扫面进行稀疏深度完成的方法，在没有稠密标签、LiDAR累积或伪LiDAR生成的前提下，直接融合冻结的单目深度基础模型与稀疏LiDAR。

**💡 创新点**

创新点包括：①将冻结的单目深度先验与稀疏LiDAR按帧对齐并轻量级融合；②引入高频正则化的法向一致性损失，抑制因噪声LiDAR导致的局部几何伪影；③提出专门针对LiDAR覆盖之外区域的评估协议，揭示模型的外推能力。

**🔧 技术方法**

使用冻结的单目深度基础模型（如MoGe或DepthAnything）作为先验，基于LiDAR点云实现相机坐标系对齐；采用DySPN等轻量级稀疏融合网络进行深度推断；损失函数包括鲁棒稀疏锚点损失、全局法向一致性损失和高频法向一致性损失。

**📊 数据集**

在KITTI、NYUv2和nuScenes数据集上进行实验；KITTI用于基准和外推评估；NYUv2用于稀疏训练的室内评估；nuScenes用于极端环境下的定性鲁棒性验证。

**📈 对比分析**

与基线方法（仅先验对齐、OMNI-DC、DMD3C、DySPN）对比，在控制的LiDAR覆盖之外评估协议下，本文方法在RMSE/Abs Rel等指标上显著优于先验对齐与传统学习模型；在标准KITTI深度完成任务上亦保持竞争力；在NYUv2稀疏训练下与全监督DMD3C接近，展示稀疏监督效果。

**⚠️ 局限性**

局限性包括：①高频正则化在全局指标上提升有限，主要体现为视觉上更平滑的几何；②方法依赖先验模型与对齐的准确性，对极端场景下的对齐误差敏感；③在纯稠密监督或多帧累积场景中表现不一定优于专门设计的深度完成模型；④需要额外的对齐计算，略微增加推理时延。

---

## 128. CrowdDirector: Runtime Orchestration of Crowd Simulation through Natural-Language Directives

**arXiv ID:** 2611.08763 | [PDF](https://arxiv.org/pdf/2611.08763v1)

**作者:** Xiaohan Sun `[一作]` (Trinity College Dublin), Panayiotis Charalambous `[通讯]` (CYENS Centre of Excellence)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CrowdDirector 框架，允许在 Unity 运行时通过一次性自然语言指令来动态控制人群，后续指令以结构化指令存为状态并由图神经网络本地做决策。

**💡 创新点**

创新点：①将 LLM 仅用于指令解析，避免每个时间步调用；②设计场景验证器将指令映射到当前场景并给出可执行性判定；③使用图神经网络策略结合模拟滚动和 LLM 排名的混合标签实现局部决策；④将指令转换为持续的 “directive” 并在整个仿真中保持激活。

**🔧 技术方法**

技术：自然语言模型 (Claude Haiku) 解析指令；场景验证器；基于邻接关系的 crowd graph；三层关系图神经网络（GNN）评分器；离线训练时使用模拟滚动 + LLM 排名；Unity 与 Python 通过 WebSocket 交互。

**📊 数据集**

数据集：基于程序化生成的 2,776 个世界（训练 2,148，验证 258，测试 370），共 73,606 个代理决策；其中 61,439 条决策使用了 LLM 排名；行为基准 100 场景，每场 10 次初始化；人工标注的 15 名受试者提供 5×4 类别的 10 个场景。

**📈 对比分析**

对比方法：手写规则集、每帧 LLM 控制。CrowdDirector 在指令可执行性（与人类判断的匹配）上略优于 LLM 自评；在受影响代理范围（Jaccard）和动作选择（一致率）上比手写规则高约 2 倍，略低于每帧 LLM，但比其低 15% 左右；在运行时成本上，CrowdDirector 每个指令只调用一次 LLM，延迟和 API 成本相比每帧 LLM 降低 189 倍、成本降低 2 量级；在大规模人群（多至 5,000 代理）上，延迟随代理数线性增长，仍保持交互级别。

**⚠️ 局限性**

局限性：①仅支持预定义的动作和交互，实验在 2D 环境中；②受试者人数少（15 人）且指令场景有限；③离线训练仅覆盖 8–40 代理，未验证在更大规模人群中的行为一致性。

---

## 129. Goals as a Representational Layer for Human-AI Collaboration

**arXiv ID:** 2611.08757 | [PDF](https://arxiv.org/pdf/2611.08757v1)

**作者:** Gaia Molinaro `[一作]` (Microsoft), Scott Counts `[通讯]` (Microsoft)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出将目标（goals）作为人机协作中的显式表示层，旨在通过结构化的目标图来桥接用户请求与更大范围的工作目的，提升上下文管理、协作协调、反思实验和AI评估等方面的效果。

**💡 创新点**

创新点在于：①将人类目标的隐性、动态和层级属性转化为可持续维护的多维结构；②引入七问（What, Who, Who else, When, Where, Why, How）作为组织目标信息的通用框架；③将目标层视作人机互动的公共基底，为AI系统的决策、学习和评估提供基于人类价值的参照。

**🔧 技术方法**

主要技术包括：目标表述与推断机制（对话式助理、生成式语言模型用于补全隐性目标）；目标图结构（节点表示目标，边表示层级与关系）；基于目标信息的上下文检索与推理；以及参与式人机协作流程以保证目标的可持续维护与校正。

**📊 数据集**

本文为概念性与方法论性讨论，并未使用具体实验数据集或公开数据。

**📈 对比分析**

由于本文未进行量化实验或对比实验，故不存在性能指标；其价值体现在对未来研究与系统设计提供的理论框架与实践建议。

**⚠️ 局限性**

局限性包括：目标推断的准确性与不确定性；维护目标信息的成本与用户参与度；隐私与治理风险；以及在不同工作场景下目标结构的可适配性与灵活性。

---

## 130. AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling

**arXiv ID:** 2611.08752 | [PDF](https://arxiv.org/pdf/2611.08752v1)

**作者:** Doniyorkhon Obidov `[一作]` (Michigan Technological University), Kaichen Yang `[通讯]` (Michigan Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在冻结LLM上联合优化干预位置和力度的防御方法，利用查询条件的全头缩放来抑制或增强注意力头的输出。

**💡 创新点**

创新点是联合学习注意力头选择与缩放因子，且缩放仅改变幅度保持方向，从而实现稀疏且可调的干预。

**🔧 技术方法**

技术包括全头输出缩放、查询条件可学习的标量因子、Gumbel-Softmax 进行可微分的Top‑K头选择，以及冻结模型训练。

**📊 数据集**

使用的训练数据是少量（30或120）对抗性和干净提示样本，测试集包括9种不同的攻击（Plain Harmful、Human‑Crafted、GCG等）以及500个善意提示和250个易过度拒绝提示。

**📈 对比分析**

与SFT、LoRA‑SFT、ASGuard、DDPO等基线对比，所提出方法在低样本下ARR可达96–99%，BPR保持97%+，且在过度拒绝率和MMLU上表现最优，明显优于传统方法。

**⚠️ 局限性**

局限性在于只在输入预填阶段进行干预，且对极端稠密干预或对抗性自适应攻击的鲁棒性仍有限；此外需要在冻结模型上进行额外的头选择与缩放训练。

---

## 131. eSharp: Diffusion-based Sharp Monocular Metric Event-to-Depth Estimation

**arXiv ID:** 2611.08731 | [PDF](https://arxiv.org/pdf/2611.08731v1)

**作者:** Qiushi Dai `[一作]` (Rice University), Ashok Veeraraghavan `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对单目事件相机的深度估计，提出了 eSharp，利用预训练扩散模型并结合从帧相机推导的稠密伪深度标签，实现高分辨率的精确度量深度预测。

**💡 创新点**

创新点在于：①把扩散模型迁移到事件域并用事件编码器；②采用跨模态蒸馏生成稠密伪标签；③分两阶段训练，先恢复几何结构再通过轻量级标定头纠正尺度偏移，从而直接输出度量深度并保持边缘清晰。

**🔧 技术方法**

使用技术包括：事件时域均匀分箱 + VAE 编码；预训练的 Marigold/Lotus 扩散模型；DepthAnything V2 作为帧到深度教师；两阶段训练（扩散去噪 + 标定头）；以及一次性 x₀ 预测的 fast 版本。

**📊 数据集**

主要数据集：DSEC（高分辨率事件+RGB）、MVSEC（低分辨率事件+灰度帧）用于训练和评估；TUM‑VIE 用于零样本跨域测试。

**📈 对比分析**

与现有事件单目深度方法（e2depth、EReFormer、DepthAnyEvent 及其递归版）以及相对深度基线进行对比，eSharp 在 Abs Rel、Sq Rel、RMSE、RMSE_log、SILog 以及 δ<1.25、1.25²、1.25³ 这三种阈值下均取得更低误差、更高精度；fast 版本在保持较高精度的同时提升约 6× 推理速度。

**⚠️ 局限性**

局限性包括：对事件质量敏感，遮挡或噪声强时可能漏检或误检；伪标签质量受帧相机分辨率影响，MVSEC 上性能略低；目前主要在驾驶场景训练，跨域性能虽可行但在极端光照/室内场景仍需进一步提升。

---

## 132. Divide, Diagnose, and Compose: Optimizing Agentic Pipelines and Harnesses

**arXiv ID:** 2611.08723 | [PDF](https://arxiv.org/pdf/2611.08723v1)

**作者:** Divij Handa `[一作]` (Arizona State University), Vlad I. Morariu `[通讯]` (Adobe)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DeCAF框架，对多代理管道进行分解式信用分配的自动化优化，逐维度搜索并合成最佳性能、成本和延迟平衡方案。

**💡 创新点**

将优化空间自适应划分为任务相关维度，使用多维度独立实验生成并记录，结合节点级轨迹诊断进行跨维度合成，形成可追踪、可诊断的优化流程。

**🔧 技术方法**

使用大型语言模型作为编码代理生成实验与实现更改；多维度并行搜索；节点级轨迹分析工具；非支配前沿维护与合成。

**📊 数据集**

AppWorld、LiveCodeBench、AgentDojo、PostTrainBench v1.1等四大多代理基准。

**📈 对比分析**

与未优化基线及GEPA、MASPO、Meta-Harness等三种自动优化器对比，在所有基准上平均提升任务性能2.58pp、降低4.30%推理成本、降低42.45%端到端延迟。

**⚠️ 局限性**

整体优化成本高于单一基线约18%，对 GPU 资源要求高，且跨维度合成依赖节点诊断工具，对某些基准的维度互相干扰时效果有限。

---

## 133. The Matthews Correlation Coefficient: Fifty years

**arXiv ID:** 2611.08799 | [PDF](https://arxiv.org/pdf/2611.08799v1)

**作者:** Jesus S. Aguilar-Ruiz `[一作]` `[通讯]` (Pablo de Olavide University), Jesus S. Aguilar-Ruiz (Pablo de Olavide University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 Matthews 相关系数（MCC）进行了五十年的回顾，阐述其历史渊源、数学定义、与 Pearson 相关系数、χ²检验的等价关系，并深入探讨其在类别不平衡情境下的行为。

**💡 创新点**

创新点在于提出了一套通用的分析与可视化框架（包括解析导数、等高线图、局部斜率分析），能够系统地揭示任何基于混淆矩阵的指标在类别不平衡时的偏倚、稳健性与灵敏度；并将该框架应用于 MCC，展示其非不平衡不变性与潜在误导性。

**🔧 技术方法**

技术手段包括：混淆矩阵解析、Pearson 相关系数推导、χ² 关系、Fisher z 变换与 χ² 逆推置信区间、导数与等高线图（Contour Plot）绘制，以及偏倚与敏感度的定量描述。

**📊 数据集**

论文主要以理论推导为主，使用示例性混淆矩阵和假设数据（如极端不平衡比例 95/5）来演示指标行为，并未依赖特定公开数据集进行实验验证。

**📈 对比分析**

比较方法：将 MCC 与准确率（Accuracy）、几何平均（G‑mean）等常用指标在不同类别比例下绘制等高线图，并通过斜率与曲线形状评估其稳健性。结果表明，MCC 在严重不平衡时会产生较大偏差，准确率表现最差，G‑mean 最稳健。

**⚠️ 局限性**

局限性：仅针对二分类；框架虽可推广但在多分类扩展中仍需研究；分析以理论为主，缺乏大规模实验验证；MCC 的敏感性在极端不平衡时仍可能导致误判。

---

## 134. MotionDecipher: LLM-Assisted Clinical Interpretation of Human Motion Signals for Efficient Gait Analysis

**arXiv ID:** 2611.08795 | [PDF](https://arxiv.org/pdf/2611.08795v1)

**作者:** Mingrui Ma `[一作]` (University of Illinois Urbana-Champaign), Yiwen Dong `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过将步态信号转换为结构化特征，再利用LLM生成临床步态分析报告，实现步态分析的自动化

**💡 创新点**

创新性地提出了 Human Motion Tokenizer、Meta Feature Selector 和 Hybrid Context Constructor 三段流水线，并结合 Hybrid Index 实现专家示例检索与 In-Context Learning，显著提升报告质量

**🔧 技术方法**

使用 Savitzky‑Golay 平滑、离散小波变换、离散化量化、文本与数值双重检索、LLM（以 Kimi‑K2 为例）推理等技术

**📊 数据集**

在加州大学尔利与斯坦福大学临床步态实验室收集的 122 名患者与 46 名对照的三维运动捕捉与力板数据上进行评估

**📈 对比分析**

与 GPT‑5、Kimi‑K2、Google Gemini 3、Qwen3‑Max 等基线比较，MotionDecipher 在 BLEU、ROUGE‑L、METEOR、BERTScore、MEDCON‑Gait、GaitTripleF1 上提升 9%–34%，专家评估得分最高

**⚠️ 局限性**

目前仅适用于实验室高精度捕捉场景，缺乏治疗建议功能，对低成本传感器的适配尚未完全验证

---

## 135. Training Stability under Adaptive Curricula

**arXiv ID:** 2611.08787 | [PDF](https://arxiv.org/pdf/2611.08787v1)

**作者:** Johan Obando-Ceron `[一作]` (Mila Quebec Ai Institute), Pablo Samuel Castro `[通讯]` (Mila Quebec Ai Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了自监督环境设计（UED）中适应性课程对学习者可训练性的影响，并通过实验验证了训练稳定化干预（权重裁剪+分类价值预测）能够提升学习者在UED中的性能。

**💡 创新点**

提出将学习者的训练状态视为UED的设计原则，揭示学习者退化可削弱课程效果，并给出了简单有效的稳定化策略（WC+HL），在不改变课程规则的前提下显著提升性能。

**🔧 技术方法**

使用PPO actor‑critic RL；采用多种UED方法（PLR、SFL、ACCEL、DR）；实施干预技术包括权重裁剪、分类价值预测（HL‑Gauss）及其组合；并使用梯度范数、参数大小、价值‑结果秩相关、有效秩、特征相似度等诊断指标。

**📊 数据集**

实验数据来自Kinetix物理控制任务，涵盖Small、Medium、Large三种复杂度；还使用人类设计的 hold‑out 任务评估零-shot 泛化。

**📈 对比分析**

在solve rate和sampled return上与基线进行对比，结果显示WC+HL在多种环境复杂度、训练时长、并行度、以及分布变换下均优于基线，保持更好的价值排序与表示多样性，性能随训练延长、并行度提升仍保持优势。

**⚠️ 局限性**

仅在Kinetix和PPO框架下验证；超参敏感性、探针选择与干预交互限制因果推断；未在更广泛的RL算法或更复杂任务上验证；连续学习场景的改进尚未完全实现。

---

## 136. Aligning, Fast and Slow: A Control-Theoretic Oversight Framework for Long-Horizon Trajectories and Maintained Deployments

**arXiv ID:** 2611.08781 | [PDF](https://arxiv.org/pdf/2611.08781v1)

**作者:** Stephen Peng `[一作]` (BCG X AI Science Institute, Boston Consulting Group), Adi Zolotov `[通讯]` (BCG X AI Science Institute, Boston Consulting Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出双时钟监督框架（fast loop 与 slow loop），为AI代理在长期部署中的行为偏差与继承变化提供控制理论上的监测与干预，并在交易与编码模拟中验证效果。

**💡 创新点**

创新点包括：① 将安全屏障函数与递归评估结合，给出离散时间的部署级别偏差上界；② 发现并证明“carryover gap”——单个运行通过的检查并不能保证后续运行的安全；③ 给出基于误差上限与变更速率的资源分配最优规则；④ 在实验中首次同时评估动作门控与记忆/代码更新门控。

**🔧 技术方法**

使用的技术包括：控制理论（安全屏障函数、Simplex、递归评估）、fast loop 的动作门控与anchor 校准、slow loop 的记忆/代码更新评估、LLM 评判器、模拟环境与实验脚本。

**📊 数据集**

数据集：交易实验使用 Arike 等的 32 季度目标漂移环境；编码实验使用 Marshmallow（Python 序列化库）的 24 个 SWE‑smith 问题，配合 120 条任务；同时植入 3 条恶意记忆作为压力测试。

**📈 对比分析**

比较方法：设置四种策略（无门控、fast gate、fast gate+anchor、全部 gates），评估指标为交易 drift 与编码 severity；实验结果显示逐步加入控制可将交易 drift 从 0.892 降至 0.107，severity 从 0.325 降至 0.117；在编码实验中，问题解决率从 35/120 降至 33/120。

**⚠️ 局限性**

局限性：① 需要假设测量误差和变更上限、控制器能执行决定；② 评估基准固定，无法反映所有潜在危害；③ 实验仅在单一交易与编码环境，缺乏跨任务通用性验证；④ 未充分考察频繁更新对保证的影响；⑤ 共享内存、多部署协同场景未被覆盖。

---

## 137. Guardrail Erosion in Domain Fine-Tuned Large Language Models: A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs

**arXiv ID:** 2611.08780 | [PDF](https://arxiv.org/pdf/2611.08780v1)

**作者:** Stephen Peng `[一作]` (Boston Consulting Group), Adi Zolotov `[通讯]` (Boston Consulting Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了25个生物医学领域的LLM微调模型与其基准模型在知识中心化回避（abstention）和临床决策基准上的行为变化，发现微调会削弱模型的适当回避能力并增加答复承诺。

**💡 创新点**

提出了针对医学领域的多维度行为审计框架，系统评估了能力、回避、校准和承诺四个维度，并揭示微调对这些维度的异质影响。

**🔧 技术方法**

采用配对审计方法，使用MedAbstain与ClinDet-Bench基准，计算Youden J、ECE、归一化熵等指标，并通过McNemar、Wilcoxon、Bootstrap等统计检验评估显著性。

**📊 数据集**

MedAbstain（包含多项选择题和证据消融版本）和ClinDet-Bench（可判定与不可判定的临床决策场景）以及公开的模型检查点。

**📈 对比分析**

通过将微调模型与其基准模型按配对进行逐项比较，结果显示在MedAbstain上适当回避率下降约45%，新答复问题的准确率下降约15%，同时承诺度提升，校准误差增加。

**⚠️ 局限性**

基准受限于可能的训练数据泄漏与覆盖率不足，模型公开信息不完整导致混杂因素，且仅评估静态基准问答，未涉及真实临床决策或交互式情景。

---

## 138. Conflux: Preventing Privilege Escalation with Principal Contexts and System-Level Evaluation

**arXiv ID:** 2611.08768 | [PDF](https://arxiv.org/pdf/2611.08768v1)

**作者:** Raya Buckley `[一作]` (University of Oxford), Christian Schroeder de Witt `[通讯]` (University of Oxford)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于影响力追踪的权限交叉裁决系统ITES，以及完整的Conflux体系结构，以阻止LLM代理的特权升级，并提供了黑盒（SLED）和白盒（SLED‑V）评估工具。

**💡 创新点**

创新点在于：① 用记录的影响力集合对权限进行交叉裁决的安全核ITEs；② 将模型决策与系统层权限分离，提出可证明的“条件权限限制”定理；③ 通过Conflux扩展实现授权、可见性、同意、委托、限制与暴露规划的统一框架。

**🔧 技术方法**

使用技术包括权限交叉裁决规则、影响力追踪与无洗牌证明、基于中间表示的白盒模型检查（SLED‑V）、广度优先的黑盒搜索（SLED）、证书绑定与重检查、可信计算基准等。

**📊 数据集**

实验数据主要来自AgentDojo的Prompt Injection任务集，以及自行构造的有限模型和IR数据；未使用公开的大规模真实数据集。

**📈 对比分析**

评估方法：在声明的有限模型上执行SLED的遍历搜索和SLED‑V的SMT/抽象归约检查，记录安全/违规计数和运行时间；结果表明在声明模型下未发现违规，但评估是有界的，未给出整体部署性能指标，实际耗时随模型规模呈线性/指数增长。

**⚠️ 局限性**

局限性包括：① 只覆盖价值影响，不处理控制影响；② 定理依赖完整媒介、无洗牌、已知决策上下文等五个假设；③ 评估为有界，未验证无限部署安全；④ 不能保证消息泄露、侧信道、委托、重定向等安全问题；⑤ 未证明实现严格符合理论约束。

---

## 139. Debate to Deceive: Poisoning LLM's Fact-Checking via Game-Theoretic Multi-Agent Debate

**arXiv ID:** 2611.08756 | [PDF](https://arxiv.org/pdf/2611.08756v1)

**作者:** Liting Huang `[一作]` (Newcastle University), Varun Ojha `[通讯]` (Newcastle University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于人类辩论结构的多代理攻击框架——Debate-to-Deceive（D2D），通过对抗性辩论生成语义连贯、修辞有效的攻击文本，误导大型语言模型（LLM）在事实检查任务中的判定。

**💡 创新点**

核心创新在于：①将辩论分为四个阶段并引入经典修辞三元（Logos、Pathos、Ethos）；②将攻击过程建模为演化博弈，利用多代理强化学习让策略随对手动态演化；③通过策略层与生成层的分离，使得搜索空间显著压缩，提升攻击效率与多样性。

**🔧 技术方法**

技术包括：多代理强化学习（MARL）、演化博弈理论（EGT）、软最大策略更新、分层文本生成（策略+文本），以及在黑盒环境下的目标标记翻转与语义一致性检验。

**📊 数据集**

在四个事实检查基准上进行评估：FEVER、HealthVer、CLIMATE-FEVER、SciFact（各100条）以及简易事实集SimpleFacts。

**📈 对比分析**

与AutoDAN、Open-Prompt-Injection、ADMIT、JudgeDeceiver等基线在无防御和多种防御（指令、已知答案、LLM、响应、改写）下对比。D2D在无防御时攻击成功率（ASR）达88–100%，在已知答案防御下仍保持约98% ASR，且运行时间、令牌数和内存使用均低于基线。

**⚠️ 局限性**

限制包括：仍受LLM对某些事实错误的对齐约束影响（例如对健康错误的主动否认）；需要多代理和博弈理论的专业知识，重现门槛相对较高；攻击成功率在已知答案防御下略有下降，说明仍存在可改进空间。

---

## 140. AlterNet: Adaptive-Depth Mixture-of-Experts with Learned Halting for Elastic Inference

**arXiv ID:** 2611.08741 | [PDF](https://arxiv.org/pdf/2611.08741v1)

**作者:** Alexander Kastius `[一作]` `[通讯]` (SAP), Alexander Kastius (SAP)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为AlterNet的动态深度Mixture‑of‑Experts模型，结合浅层专家、深度编码和分层停止机制，实现前向推理时可根据任务动态调整计算量；

**💡 创新点**

核心创新包括：在循环迭代中加入显式深度编码与基于专家的停顿头、使用高斯分布作为停顿先验、以及通过迭代路由实现专家间信息交互；

**🔧 技术方法**

采用了Mixture‑of‑Experts架构、类似Universal Transformer的循环自注意力、PonderNet式动态停顿、sinusoidal深度编码、z‑loss与负载平衡损失、以及Gaussian‑prior KL正则；

**📊 数据集**

在C4数据集上进行训练和评估，使用了约3.9 B个标记（相比OPT 350 M所需的7 B标记）来测试模型性能；

**📈 对比分析**

将固定12次迭代的基线模型与动态停顿模型对比，动态模型在推理时平均仅使用约48%迭代，perplexity从40.1降至54.3，显示出计算节省但在评估准确度上略有不足；

**⚠️ 局限性**

局限性包括：模型训练时间不足、深度bootstrap崩溃问题、需要更大数据量和模型规模以充分发挥潜力、以及高斯先验与迭代深度设置的敏感性。

---

## 141. Stochastic AI Model Placement for AI-and-RAN Coexistence in 6G Networks

**arXiv ID:** 2611.08739 | [PDF](https://arxiv.org/pdf/2611.08739v1)

**作者:** Alaa M. Altarazi `[一作]` (Princess Sumaya University for Technology), Mohammad J. Abdel-Rahman `[通讯]` (Princess Sumaya University for Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种单阶段的概率约束优化模型（C²AIR），用于在共享GPU的RAN站点上部署AI推理服务，最大化期望服务请求数量的同时保证RAN计算需求的可靠性。

**💡 创新点**

创新点包括：① 将固定的MIG切片配置与AI服务部署结合，允许同一模型在多个站点复制；② 引入RAN概率约束和AI可靠性约束，确保RAN和AI各自满足设定的违规/超载概率；③ 通过概率分布的确定性等价转化（Binomial/Lognormal RAN、Gamma AI）将问题保持为混合整数线性规划；④ 采用级别编码将非线性映射线性化，保持求解效率。

**🔧 技术方法**

采用混合整数线性规划（MILP）技术，并利用Cplex/HiGHS求解器；通过对RAN和AI需求分布的确定性阈值推导实现概率约束的线性化；使用级别变量将RAN容量映射线性化；在求解前预先计算可用MIG配置集和AI服务的可靠容量。

**📊 数据集**

使用 NVIDIA Aerial SDK（RAN需求）和 MLPerf Inference（AI模型吞吐量）数据进行参数校准；在四站点小型部署中基于A100-SXM4-80GB GPU，随后在七个大规模场景中基于 Hopper 系列 GPU（H200、GH200、H100），共计 15 个站点、40 个 AI 服务；需求采用 Gamma 分布（AI）和 Binomial/Lognormal 分布（RAN）。

**📈 对比分析**

与三种基线比较：① 仅保留平均 RAN 需求的保留策略；② 去掉 AI 可靠性约束；③ 仅单站点部署。使用指标包括：期望服务请求数、无超载的请求数、最差站点 RAN 违规概率、最差模型超载概率。实验表明：C²AIR 在所有时间点下均保持 RAN 违规率 ≤ 目标，且相较于平均需求保留仅损失 0.1%–4.6% 的 AI 通过量；去掉 AI 可靠性约束会导致大量超载；多站点复制可提升 2.7 倍的 AI 通过量。

**⚠️ 局限性**

局限性：仅为单阶段模型，无法在需求实现后动态重配置；对需求分布的准确性高度依赖，尾部估计误差会直接影响 RAN 保护；MIG 切片粒度粗糙导致高负载时 RAN 可能占用整块 GPU，削弱 AI 资源；未考虑跨站点请求路由、RAN 与 AI 需求的相关性，以及重构开销。

---

## 142. Structuring Survey Histories for LLM Digital Twins: A Controlled Study of Background, Decision Procedure, and Evaluation Profiles

**arXiv ID:** 2611.08730 | [PDF](https://arxiv.org/pdf/2611.08730v1)

**作者:** Iris Ye `[一作]` (University of Chicago), Ozan Candogan `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了基于大型语言模型的数字双胞胎在预测个体对新问题回答时的表现，设计并评估了一个固定的三部分结构化档案（背景、决策程序、评估），并比较其对预测准确性的提升。

**💡 创新点**

提出并验证了固定三部分架构的档案表示方法，证明其能显著提升预测精度，尤其是评估部分的贡献，且通过控制压缩、深度推断和内容组合来确定提升的主要来源。

**🔧 技术方法**

使用大型语言模型（LLM）模拟器进行回答预测，采用V1–V3不同深度的推断规则、Markdown结构化输出，并用配对bootstrap检验统计显著性。

**📊 数据集**

使用 Twin-2K-500 调查数据（2058名受访者、500道题目、17项任务，88道题目留作评估）。

**📈 对比分析**

在100名受访者上用两款相同族的LLM模拟器对 V3 档案、原始转录、长度匹配的自由形式档案以及不同深度和组件组合进行配对对比；结果显示 V3 档案整体准确率提升约1.27–1.44个百分点，价格任务提升2.15–3.42个百分点，且在统计上显著。

**⚠️ 局限性**

仅使用单一受访者样本、单一基准、两款相同模型族；自由形式比较未能完全区分提取与组织；组件剔除缺乏等长对照；总体分数权重不平衡，导致结果主要受价格任务影响。

---

## 143. SSF-Net: A Spatial-Spectral Fusion Network for Accurate PDE Operator Learning

**arXiv ID:** 2611.08912 | [PDF](https://arxiv.org/pdf/2611.08912v1)

**作者:** Siqi Zhao `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SSF‑Net，一种双分支（空间+频域）神经算子，结合多方向Mamba和FFT，嵌入U形多分辨率网络，用于高效求解二维PDE。

**💡 创新点**

通过空间‑频谱融合块实现局部与全局交互的近线性复杂度，同时引入四方向扫描的Mamba减少因果偏差，兼顾局部细节与全局耦合。

**🔧 技术方法**

多方向Mamba状态空间模型、快速傅里叶变换与块对角MLP、重叠补丁嵌入、U‑Net多分辨率编码解码、软阈值稀疏化等。

**📊 数据集**

十二个二维PDE基准，包括四个不可压流体任务（NS‑PwC、NS‑SL、NS‑Tracer‑PwC、FNS‑KF）、Poisson‑Gauss、CE‑RP、Wave‑Gauss、ACE，以及四个100步自回归的Kolmogorov/Kuramoto‑Sivashinsky任务。

**📈 对比分析**

与FNO、CNO、PDE‑Transformer、scOT、Drift‑Net等基线在相同任务下比较，采用相对L1误差；SSF‑Net在所有12个任务中均取得最优或次优结果，参数量最少，误差平均降低约55%至80%，且自回归10²步更稳健。

**⚠️ 局限性**

理论复杂度近线性，但实际训练/推理开销高；空间分支受扫描顺序限制，导致欧氏邻域交互衰减，难以完全捕捉近邻细节。

---

## 144. Inside the Trap: A Renewal Law for Prefix-Cache Collapse in LLM Agent Serving

**arXiv ID:** 2611.08902 | [PDF](https://arxiv.org/pdf/2611.08902v1)

**作者:** Akshat Valse `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型代理在前缀缓存崩溃（陷阱）下的行为，并推导了一个基于 FIFO 轨道的线性再生递推公式。

**💡 创新点**

提出了捕捉陷阱状态的精确自相似理论（Perron 根、线性再生递推），并通过注册实验验证了陷阱不自愈的机制与干预策略的有效性。

**🔧 技术方法**

采用流形排队理论、Perron 根分析、再生定理、仿真与注册实验相结合的技术框架。

**📊 数据集**

使用合成的代理工作负载（几何转数、指数工具时长、固定工具输出）在 SGLang、vLLM、Llama 上，配备 300k–600k 令牌池的单 GPU（A100/H200）环境。

**📈 对比分析**

通过与 57 次陷阱运行的实测数据对比，模型在提示组成、间隔增长、后备量斜率与完成速率等方面均达 3% 以内误差；并验证了会话上限与到达切断两种救援方案的有效性。

**⚠️ 局限性**

模型假设所有转数均失效，忽略并发、抢占与真实工具时延，验证为回顾性，且仅针对单 GPU、稠密 7–8B 模型与合成负载，可推广性有限。

---

## 145. Richness Is Not Evenness: Ordered Renyi Crossings under Self-Training

**arXiv ID:** 2611.08892 | [PDF](https://arxiv.org/pdf/2611.08892v1)

**作者:** Pranjal Srivastava `[一作]` `[通讯]` (University of Michigan), Pranjal Srivastava (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在递归自训练过程中，语言模型的词汇多样性和不同阶 Renyi 熵随训练温度变化的表现，并探讨了熵跨越零点的顺序。

**💡 创新点**

发现不同阶的熵（最小熵→碰撞熵→Shannon 熵→丰富度）在训练温度上依次出现正负交叉，并且该交叉顺序在多种模型、语言、训练策略和测量单元下保持稳健，提供了一种新的多样性诊断曲线。

**🔧 技术方法**

使用 GPT‑2、OLMo、Qwen、SmolLM、Hindi GPT‑2 等 Transformer 模型进行九代递归 fine‑tuning，采用固定测量采样器（T=1, top‑p=0.95）计算 Renyi 熵、碰撞熵、Shannon 熵、丰富度、distinct‑n、repetition、self‑BLEU 等多样性指标；并对比不同采样温度、top‑p、截断与非截断训练的差异。

**📊 数据集**

数据集包括英文五本经典文学（Gutenberg），德语、西班牙语、印地语（Wikisource）和中文（Wenzhong）等多语言文本；所有文本均在训练前拆分为固定预算（如 13,000 词）或常见长度进行测量。

**📈 对比分析**

方法：在不同温度/ top‑p 组合下递归训练模型 9 代，使用相同的测量采样器评估每代生成文本。实验表明，在大多数设置下，温度为 1 时落在交叉区间内，模型出现词汇类型减少、最常词占比下降但高阶熵上升的现象；不同多样性指标（distinct‑n、self‑BLEU 等）也在同一区间内跨零点。性能评估以参考文本的交叉 perplexity 作为独立质量度量，发现熵变化的符号并不能可靠判断质量下降。

**⚠️ 局限性**

局限性：实验规模受限于模型参数（最多 1.48B）和训练周期，跨语言、跨任务推广未知；采样噪声和阈值对熵交叉点敏感；仅使用自定义文本多样性指标，缺乏人类评测；未验证跨设置阈值转移和干预时机；截断与非截断训练的差异受重复率惩罚等超参影响；熵指标本身无法单独诊断质量或原因。

---

## 146. Bidirectional Modeling of mmWave Human Signals for Enhancement & Synthesis

**arXiv ID:** 2611.08870 | [PDF](https://arxiv.org/pdf/2611.08870v1)

**作者:** Shuntian Zheng `[一作]`, Yu Guan `[通讯]` (University Of Warwick)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种同时处理毫米波雷达增强与合成的双任务模型——HeatWave，模型在双域 RAD（Range–Angle–Doppler）热图上保持 Doppler 信息，并通过可靠性门控和任务条件路由实现对 Doppler 的双向使用。

**💡 创新点**

创新点包括：
1) 在 RAD 网格上保留双域结构，避免“几何提前崩塌”；
2) 引入可靠性门控的 Doppler 条件化，将 Doppler 仅作为单向约束而非互换通道；
3) 通过任务嵌入实现 Doppler 的任务条件路由，使增强与合成在同一骨干上使用不同的 Doppler 角色；
4) 通过以上设计解决三类失配（几何崩塌、角色对称融合、任务无关使用）并显著提升运动关键样本性能。

**🔧 技术方法**

技术实现：
- 双域 RAD 编码器（分离空间与 Doppler 令牌）；
- 基于小 MLP 的可靠性门控交叉注意力；
- 任务嵌入驱动的 Doppler 路由与注入；
- 生成式双域解码器（空间分支 + Doppler 分支）；
- 生成式 Doppler prior（基于几何的 MLP）；
- 训练损失包含 RAD 复原与谱形一致性；
- 评估使用 H-MAE、S-MAE、CD、MPJPE/MPJVE、活动识别等指标。

**📊 数据集**

使用的数据集：
- HuPR（单人室内场景）；
- XRF55（多人人交互场景）；
- mmRadPose（单人动作强金属反射场景）。

**📈 对比分析**

方法比较与性能：
- 与点云基准（Tiger、mmWaveFlow）以及热图基准（PULSE、HuPRModel、mmWaveFlow-H）进行对照；
- HeatWave 在 H-MAE/S-MAE 上均低于所有基准，尤其在运动关键样本上提升 2–3 倍；
- 在下游姿态估计中，增强版 HeatWave 把 MPJPE 降低 11.4% / MPJVE 降低 18.6%；
- 在活动识别中，使用 HeatWave 生成的合成样本将准确率从 0.661 提升到 0.781，接近完整真实数据的 0.813；
- 在点云导出、跨雷达迁移和实时性能方面，HeatWave 也均表现出更低误差、更高 FPS 与更小模型体积。

**⚠️ 局限性**

局限性：
- 仅处理单帧，未利用时间序列的 Doppler 信息；
- 门控与 Doppler prior 的学习基于训练环境，可能对新环境中的杂波分布不稳健；
- 人体支持投影未考虑遮挡与多人人干扰；
- Doppler prior 仅基于几何，缺乏显式运动动力学约束；
- 跨雷达迁移实验仅从 HuPR 数据进行，未构建完整的跨设备通用模型。

---

## 147. As Easy as Rank One: Optimal Algorithms for Matroid Secretary and Single-Sample Matroid Prophet

**arXiv ID:** 2611.08869 | [PDF](https://arxiv.org/pdf/2611.08869v1)

**作者:** Hamed Abdi `[一作]` (University of Maryland), Sahil Singla `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一个全新的基于可逆更新的序数（ordinal）算法，能够在未知权值的随机顺序下对任何给定的基 matroid 进行在线选择，并在强形式的 Matroid Secretary 设定中实现最优竞争比 1/e；同时，它给出了单样本 Matroid Prophet Inequality 的最优 2 竞争比，并在两种设定下都仅使用一次样本、无价值分布信息、可逆更新和对独立性查询的有限期望次数。

**💡 创新点**

创新点包括：
- 通过维护一个“参考状态”并在每一次接受或拒绝后以可逆方式更新，保持了对已处理元素的独立性不变，从而实现了对每个最优元素独立以 1/e 的概率被接受。
- 用可逆更新的思想把随机抽样问题转化为对分布的保留，推导出一个 Snapshot Lemma，证明每个元素的接受概率与其先前状态无关。
- 构造了一个连续时间的“幸存者实验”，通过对每个可接受元素给定进入阈值 p/t 的接受概率，并证明其满足容量和一致性约束，进而得到可实现的分布。
- 通过一个多项式时间的前向/后向更新子程序，实际实现了上述分布的采样，避免了显式计算全局概率。
- 对 Ordinal Secretary 的所有可接受集分布给出了一个有限多面体的完整描述，允许把算法设计转化为线性规划问题，并以此推导出最优竞争比。

**🔧 技术方法**

主要技术手段包括：
- 基于 Greedy 选择和 matroid 的闭包性质的可逆更新。
- Snapshot Lemma（可逆更新导致样本分布均匀）。
- 连续时间的幸存者实验和容量/一致性约束的解析。
- 前向/后向递归采样子程序，保证均匀分布并满足可逆更新。
- 线性规划/多面体方法，用于 Characterize 所有 ordinal 算法可达分布。
- 通过对每个元素的可接受概率为 1/e 的分布设计，得到强形式的 Matroid Secretary 结果。

**📊 数据集**

本工作主要是理论性质的证明，并未使用实际数据集；所有分析均在基 matroid 的通用模型下进行，适用于任意给定的 matroid。

**📈 对比分析**

与之前的最优上界 O(loglog r) 的比较：本算法在任何 matroid 上都实现了 1/e 的最优竞争比；在单样本 Prophet Inequality 上也实现了 2 的最优竞争比，匹配了 rank‑one 的下界；算法仅使用一次样本、一次性样本分布无须预先知道、并且仅通过比较值的顺序来做决策。
实验或数值评估未给出；所有性能结论均来自严谨的理论证明。

**⚠️ 局限性**

主要局限：
- 需要对所有已到达元素执行独立性查询，尽管期望次数为多项式，但在某些实际实现中可能较大。
- 算法的时间复杂度虽然是期望多项式，但涉及大量递归调用和随机数生成，实际实现成本需进一步优化。
- 目前仅针对标准的 Matroid Secretary 和单样本 Prophet Inequality，尚未直接扩展到更一般的约束或动态 matroid 变化。
- 论文中所有证明均在随机顺序模型（顺序为随机）下成立，若顺序不是完全随机则需要进一步研究。

---

## 148. Adaptive Scheduling Pipeline for Multi-Instance Asynchronous Reinforcement Learning

**arXiv ID:** 2611.08864 | [PDF](https://arxiv.org/pdf/2611.08864v1)

**作者:** Salah Chikhi `[一作]` (École polytechnique), Zeng Li `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Adaptive Scheduling Pipeline (ASP)，一种针对多实例异步强化学习的离线调度框架，用以在LLM后训练中同时减少填充、偏斜与排序造成的计算浪费并平滑样本到达；

**💡 创新点**

创新点在于将长度预测、动态规划生成长度均匀的minibatch与批组平衡排序三步合并为单次离线预处理，既提升单实例推理利用率，又消除全局队列波动，填补现有异步RL系统缺失的多实例调度层；

**🔧 技术方法**

技术上采用长度预测模型、动态规划(minibatch构造)、模拟退火批组平衡排序，整合至MindSpeed-RL的GRPO框架，并使用Ascend 910B3 NPU实现多实例生成；

**📊 数据集**

实验数据集包括Deepscaler（用于吞吐量、调度与可扩展性评估）和Maze-Reasoning-GRPO-v0.1（用于收敛性验证）；

**📈 对比分析**

与MindSpeed-RL（混合式单实例）和verl（异步分离但无调度）对比，ASP在Qwen3 MoE与dense模型上平均迭代时间降低约60–76%，吞吐量提升3.22倍，且在128 NPU时保持近线性扩展且收敛无明显退化；

**⚠️ 局限性**

局限性包括对长度预测精度依赖较高，预测漂移需手动监控；目前仅在Qwen3系列模型与Ascend 910B3硬件上验证，跨模型或其他算力环境的适用性待进一步评估；

---

## 149. What Limits mmWave Human-State Sensing for Robot Perception?

**arXiv ID:** 2611.08863 | [PDF](https://arxiv.org/pdf/2611.08863v1)

**作者:** Shuntian Zheng `[一作]`, Yu Guan `[通讯]` (University Of Warwick)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于毫米波雷达的姿态定位误差下限模型，并将多帧融合、视角多样化、返回选择和估计器容量作为探测器验证其效能；

**💡 创新点**

创新点在于给出以Cramér–Rao理论为基础的测量误差下限，并用此下限统一评估现有技术的增益来源；

**🔧 技术方法**

采用毫米波雷达信号建模、Cramér–Rao界分析、实验验证以及已有多模态雷达数据处理算法；

**📊 数据集**

使用mmRadPose、MM-Fi、MVDoppler-Pose三大公开数据集以及现场AWR1843BOOST雷达采集；

**📈 对比分析**

通过保持其他处理步骤不变，只改变单一因素的实验设置，比较MPJPE和可识别关节误差，结果显示多帧、双视角和返回保留能按理论预测降低误差，而更强估计器主要降低不可识别关节的误差；

**⚠️ 局限性**

局限在于假设局部高斯、单散射点、可逆 Fisher 矩阵等理想条件，且实验多聚焦于单人室内场景，未覆盖高密度人群或复杂多径环境下的性能。

---

## 150. Align3D: Image-Structure Co-Aligned 3D Scene Generation with Observability-Aware Fidelity-Preserving Guidance

**arXiv ID:** 2611.08861 | [PDF](https://arxiv.org/pdf/2611.08861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 151. ActPrune: Action-Grounded Visual Token Pruning for Efficient Vision-Language-Action Models

**arXiv ID:** 2611.08827 | [PDF](https://arxiv.org/pdf/2611.08827v1)

**作者:** Sungwon Moon `[一作]` (Ewha Womans University), Jiyoung Lee `[通讯]` (Ewha Womans University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在机器人闭环控制中，利用前一步预测的动作信息，在视觉编码器前对视觉令牌进行无训练的前向剪枝，显著降低推理成本；

**💡 创新点**

创新点在于：①基于执行动作的几何先验在视觉编码之前生成视图特定的剪枝信号；②在固定第三人称视角投影末端执行器轨迹并结合语义锚点；③在腕部视角通过运动补偿识别冗余补丁并压缩为上下文令牌；

**🔧 技术方法**

使用的技术包括：动作几何投影、语义锚点提取、视角补偿投影、补丁级与嵌入级两阶段剪枝、上下文令牌合并、基于视觉编码器注意力的后期剪枝；

**📊 数据集**

使用的数据集包括 LIBERO（四个任务套件共4000+任务）、VLABench（100类任务、2000+对象）以及真实7-DoF Franka Research 3机器人实验；

**📈 对比分析**

与FastV、SparseVLM、DivPrune等通用剪枝方法以及VLA-Cache、VLA-Pruner、SpecPrune-VLA等针对VLA的加速方法对比，实验显示在25%保留比例下成功率最高（OpenVLA-OFT 96.1%，π_0.5 最高），FLOPs降低至约1.9/2.1×（≈59.5%）并获得1.61×的速度提升，整体性能保持与原模型相近；

**⚠️ 局限性**

局限性：依赖已校准相机与深度感知；在腕部视角遮挡大目标或大幅摄像机运动时冗余降低有限；极端剪枝比例会导致任务成功率下降；目前不适用于RGB‑only或未校准相机场景。

---

## 152. Containment Problems in Convex Sets

**arXiv ID:** 2611.08835 | [PDF](https://arxiv.org/pdf/2611.08835v1)

**作者:** Prosenjit Bose `[一作]` (Carleton University), Tyler Tuttle `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在平面上找到两个点p和q的算法，使得任何包含这两个点的C的同胚必须包含至少常数比例的点P。

**💡 创新点**

在严格凸的情况下，达到了1/5的比例，而在一般凸区域下，达到了1/6e的比例。还考虑了特殊情况下的下界，如三角形和旋转的三角形。

**🔧 技术方法**

使用了几何和组合数学的方法，特别是Voronoi图和随机算法。

**📊 数据集**

使用了平面上的n个点集P，研究了不同形状的凸集C，包括三角形和立方体。

**📈 对比分析**

通过与已有文献中的结果进行比较，证明了在不同情况下的下界，性能表现良好，尤其是在严格凸集的情况下。

**⚠️ 局限性**

限制在于对于更复杂的凸形状，算法的效率可能会受到影响，且未考虑所有可能的旋转角度。

---

## 153. Computer Vision by YOLOs for Detection of Major Diseases in Vietnamese Custard Apple Using Fruit Imagery

**arXiv ID:** 2611.08886 | [PDF](https://arxiv.org/pdf/2611.08886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 154. Toward LLM Self-Play Under Data Scarcity: Effectiveness, Generalizability, and Trustworthiness

**arXiv ID:** 2611.08908 | [PDF](https://arxiv.org/pdf/2611.08908v1)

**作者:** Zhiyin Yu `[一作]` (Peking University), Xiao Luo `[通讯]` (University of Wisconsin--Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在数据稀缺条件下实现LLM自对弈（self-play）的系统框架，强调数据中心性、跨域可泛化性和可信度。

**💡 创新点**

创新点在于：①从数据角度重新定义自对弈，设计健康意识数据分配和可学习性评估；②为不可验证任务提供生成‑演化管线和因果奖励增强；③引入行为可信度调节和来源可信度认证来治理内部和外部风险。

**🔧 技术方法**

采用强化学习（RL）、自监督评估、奖励模型、因果推理、KL正则、元学习等技术组合，配合动态数据管理和安全治理模块。

**📊 数据集**

主要使用公开自我指令（Self‑Instruction）种子集、RubricBench、Qwen2.5‑Math、标准编程与数学推理基准；实验以这些数据集为训练与评估基础。

**📈 对比分析**

通过对比Vendi多样性指标、奖励模型在三类奖励劫持（长度、格式、语调）下的准确率，以及数学推理基准的迭代性能，展示在多样性、奖励鲁棒性和迭代收敛上均优于传统方法；实验中引入评分规则后奖励误判率显著下降。

**⚠️ 局限性**

局限性包括：实验规模有限，缺乏大规模跨任务验证；奖励设计仍需在更复杂长时任务上进一步验证；安全治理机制依赖外部可信元数据，实际部署面临接口攻击与数据污染的挑战。

---

## 155. From Procedures to Permissions: Learning Precise Subagent Authorization from Agent Skills

**arXiv ID:** 2611.08905 | [PDF](https://arxiv.org/pdf/2611.08905v1)

**作者:** Wenkai Yu `[一作]` (Peking University), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Agent Skills的子代理授权数据合成方法，先把Skill转化为具体任务与步骤说明（SOP），再通过教师代理执行并记录授权轨迹，最后用LLM评审器验证轨迹，得到高质量授权监督数据。

**💡 创新点**

创新点在于：①将Skill中嵌入的程序知识映射为子代理授权监督；②在轨迹生成时利用SOP提供的步骤信息直接给出授权范围；③通过独立LLM评审筛选合法轨迹，确保训练数据既完成任务又符合授权约束。

**🔧 技术方法**

使用技术包括：①SOP驱动的轨迹合成与子代理委托；②LLM（GLM‑5.2 / GPT‑6‑Sol）评审器进行轨迹验证；④对Qwen3.5模型进行监督微调（SFT）。

**📊 数据集**

数据集：自研的4,413条轨迹（1,542条错误，2,871条通过验证），涵盖软件工程、数据分析、安全合规、业务运营、文档等领域；在ClawArena‑Team、Terminal‑Bench 2.1与SkillsBench上进行评测。

**📈 对比分析**

对比方法：与基线（原始模型、Chain‑of‑Agents、ToolMind等）在ClawArena‑Team的SMS、TPP、WPP、TCR等指标进行对照。结果显示：在Qwen3.5‑35B‑A3B上SMS提升13.8点，TPP提升约14%，WPP提升约24%；在Qwen3.5‑4B上SMS提升8.3点；在Terminal‑Bench 2.1与SkillsBench保持与公开数据合成方法相当的性能。

**⚠️ 局限性**

局限性：①依赖LLM评审器的准确性；②生成的任务和SOP仍需人工构造，覆盖范围受限；③对新领域或未见Skill的迁移能力尚未验证；④训练数据量相对有限，可能影响大规模多任务的泛化。

---

## 156. Localizing and Editing the Goal Representation of a Vision-Language-Action Model

**arXiv ID:** 2611.08895 | [PDF](https://arxiv.org/pdf/2611.08895v1)

**作者:** Sudhanshu Srivastava `[一作]` (University of Louisville), Faisal Aqlan `[通讯]` (University of Louisville)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过在LIBERO-Goal任务场景中测试OpenVLA在否定指令下的行为，定位其目标表示位置并评估基于目标读出的监控方法。

**💡 创新点**

创新点在于揭示目标信息在层12到16之间转移，证明在此前层或每个动作令牌处进行编辑即可有效重定向模型行为，并将目标读出与否定指令结合评估安全约束。

**🔧 技术方法**

采用线性探针、激活补丁、向量引导、运行时监控与分层分布式校准等技术。

**📊 数据集**

使用的数据集为LIBERO-Goal的10个共享场景任务。

**📈 对比分析**

与熵、成功探针、视觉+文本探针、物体姿态线性探针以及配对+时间优先基线比较，层32目标读出在AUROC上达到0.94，TPR高而FA低，优于其他基线。

**⚠️ 局限性**

局限性包括仅在单一VLA与模拟场景下实验，否定指令为整任务禁止，样本量小，且未在真实机器人上验证。

---

## 157. Quantum Interactive Proofs of Proximity

**arXiv ID:** 2611.08858 | [PDF](https://arxiv.org/pdf/2611.08858v1)

**作者:** Kenan Wood `[一作]` `[通讯]` (Georgia Institute of Technology), Kenan Wood (Georgia Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并系统研究了量子交互式逼近证明（QIPP）模型，并证明对所有可由对数空间均匀电路决定的 NC 属性可以构造出公共币、总复杂度为 O(n^{2/5}) 的 QIPP；

**💡 创新点**

创新点在于首次实现了 logspace‑uniform NC 属性的子线性 QIPP ；通过引入块验证和量子幅度放大技术，将传统 IPP 的线性约束验证加速到平方根级别；并给出基于后量子子指数安全伪随机生成器的条件下的查询‑通信下界以及与 Parity QIP 的双向联系；

**🔧 技术方法**

主要技术包括：结构化的 Rothblum–Rothblum IPP、线性约束的可计算性、块分区与幅度放大、量子查询模型的经典建议量子幅度放大、Goldwasser–Sipser 变换、PRG 安全性与混合界限、以及随机属性的量子信息计数方法；

**📊 数据集**

该工作为纯理论研究，无需实测数据集，所有结果均为信息论和计算复杂度分析；

**📈 对比分析**

与经典 IPP 的 O(√n) 总复杂度相比，本文的 O(n^{2/5}) 量子方案实现了显著加速；通过 Forrelation 低维分离，可指数性地将 QIPP 与传统 IPP 与 QMAP 分离；

**⚠️ 局限性**

局限性包括：总复杂度 n^{2/5} 的最佳性尚未确定；条件下界依赖后量子子指数安全 PRG 的假设；对于非 logspace‑uniform 属性，仅给出了随机属性的线性下界；以及在固定接近度参数时的性能分析仍需进一步完善。

---

## 158. Learning Whom to Trust About the Unseen: Source Credibility and Reliability Transfer in Language Models

**arXiv ID:** 2611.08855 | [PDF](https://arxiv.org/pdf/2611.08855v1)

**作者:** Ruijia Guan `[一作]` `[通讯]` (Stanford University), Ruijia Guan (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过三项实验研究语言模型在处理可靠来源的证词时，如何依据来源身份、主张内容和来源的能力说明来评估信度，并检验可靠记录在自然与精神解释间的迁移范围。

**💡 创新点**

①将儿童发展理论中的来源偏好与专业范围迁移概念引入LLM评估；②提出“记录迁移”指标衡量可靠记录对不同主张的影响；③系统对比人类基准与模型行为。

**🔧 技术方法**

采用提示工程与多模型API（Claude、GPT-5、DeepSeek、Qwen、Llama）进行对话式实验；使用统计混合模型和配对差异检验来量化信度转移。

**📊 数据集**

生成的河流预测日志、自然与精神解释的匹配对、Einstein效应的原始数据、Davoodi&Lombrozo的宗教/科学信念问卷。

**📈 对比分析**

通过对比模型与人类样本在“可信度差值”与“信心变化”上的差异，并使用同任务预测与源效应对照；结果显示大多数模型在自然主张上转移可靠记录，精神主张几乎不转移，模型表现与人类相反或放大。

**⚠️ 局限性**

受限于有限的提示数量、自然与精神主张基线差异、不同模型的后端与调参差异、缺乏对模型内部机制的解释，以及只评估数值信度而非更广泛行为。

---

## 159. Jev as an LLM Safety Guardrail: An Empirical Study of Input Filtering, Selective Output Review, and Over-Refusal

**arXiv ID:** 2611.08845 | [PDF](https://arxiv.org/pdf/2611.08845v1)

**作者:** Olivia Cognetti `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Qwen2.5-1.5B LLM上，评估并比较了无过滤、Jev输入门过滤以及分层门加输出审查的三种安全防护配置。

**💡 创新点**

提出了在多阈值下联合优化JeV分层策略的方法，并系统展示了输入门与分层门在安全-效用权衡上的差异。

**🔧 技术方法**

使用了JeV概率决策模型作为外部安全门，结合贪婪解码的LLM、阈值切分与二次输出审查。

**📊 数据集**

采用了AdvBench、XSTest-unsafe 和 XSTest-safe 三组基准数据，共970条提示。

**📈 对比分析**

通过答案率、拒绝率、宏F1等指标比较，输入门将有害提示回答率从73.2%降至2.4%，但正常提示拒绝率从1.2%升至10.8%；分层门在保持类似安全性的同时略降低了拒绝率，但需要额外的输出审查，整体性能提升有限。

**⚠️ 局限性**

局限性包括：评估仅基于是否拒绝/回答的代理指标，未直接判定答案是否真正有害；仅测试单一模型和有限基准，泛化能力未充分验证；额外输出审查带来的计算成本和延迟未量化；阈值调优在不同数据集上表现不稳定。

---

## 160. The APL Doesn't Fall Far From The Tree: Unsupervised Artefact Prototype Learning in Histopathology

**arXiv ID:** 2611.08838 | [PDF](https://arxiv.org/pdf/2611.08838v1)

**作者:** Emilia Agasi `[一作]`, T. Ian Simpson `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了无监督的Artefact Prototype Learning（APL）方法，用冻结的预训练嵌入空间检测组织病理图像中的技术性伪影

**💡 创新点**

创新点在于利用嵌入空间的几何属性（角度距离、局部密度、马氏距离）自动生成伪影原型，无需人工标注或微调

**🔧 技术方法**

使用预训练的 histopathology 编码器（UNI、Prov‑GigaPath、CONCH）生成的高维嵌入，进行原型学习、伪标签生成与逻辑回归分类

**📊 数据集**

采用来自爱丁堡癌症中心的单中心 H&E WSIs（414张）和 TCGA‑OV 多中心数据集（107张）做实验

**📈 对比分析**

与已有的 GrandQC（监督式）和 HistoART（微调式）方法比较，APL 在单中心数据上 AUROC 0.90、AUPRC 0.75，整体性能与 HistoART 相近、优于 GrandQC；在多中心 TCGA‑OV 上 AUROC 0.84、AUPRC 0.54，优于 GrandQC 但略低于 HistoART

**⚠️ 局限性**

局限性包括：在多中心环境下对不同扫描协议敏感，部分伪影如折叠、血迹难以检测，且部分异常组织可能被误认为伪影

---

## 161. MotionCore: A Latent Neural Simulator for Video Generation with Rigid-Body Dynamics

**arXiv ID:** 2611.08904 | [PDF](https://arxiv.org/pdf/2611.08904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 162. Learning to Restart: Differentiable Restart Policies for Optimization Dynamics

**arXiv ID:** 2611.08883 | [PDF](https://arxiv.org/pdf/2611.08883v1)

**作者:** Yiman Fong `[一作]` (Tsinghua University), Zaiwen Wen `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可微分的重启策略学习框架，利用连续时间ODE极限导出重启事件的梯度，并设计离散时间的 surrogate 梯度实现对传统迭代优化器（如NAG、PDHG）的重启策略进行 end-to-end 训练。

**💡 创新点**

创新点在于：①将重启事件视为混合动力系统，在连续时间下通过隐函数定理得到闭式梯度；②推出离散 surrogate 梯度并证明其收敛到连续梯度；③将该框架嵌入学习优化（L2O）中，实现对重启策略的自动学习，并展示了零射击迁移到真实问题的能力。

**🔧 技术方法**

使用技术包括：连续时间ODE建模、隐函数定理、混合动力学框架、自动微分（forward‑mode）、小型神经网络（两层MLP或仿射分数）用于表示重启分数；在离散层面实现 surrogate 梯度；对 NAG（加速梯度）和 PDHG（PDLP）等优化器进行实验。

**📊 数据集**

实验数据集：①合成二次规划（20维到1000维的对角 QP）；②合成线性规划（12维、18维等）；②真实问题：LIBSVM 逻辑回归、X‑ray 断层重建（高维）、Netlib 线性规划基准。

**📈 对比分析**

方法比较：与手工重启规则（函数重启、梯度重启）、无重启、周期重启以及 PDLP 典型重启进行对比；评价指标为目标函数误差、KKT/原始/对偶残差；实验显示学习的重启策略在 QP、LP 上均显著优于基线，并在更大维度或真实数据中零射击迁移，获得更低的终端误差或残差。

**⚠️ 局限性**

局限性：①依赖连续时间极限假设，离散 surrogate 在步长极端或不一致的离散化下可能失效；②实验主要集中在凸二次/线性规划，未验证非凸或更复杂结构的适用性；③模型容量受限，可能需要更大网络或更丰富的特征来进一步提升性能。

---

## 163. Solver-Admitted Metamorphic Testing of Learned Network Optimization Surrogates

**arXiv ID:** 2611.08837 | [PDF](https://arxiv.org/pdf/2611.08837v1)

**作者:** Vineet Jagadeesan Nair `[一作]` (Boston Consulting Group), Arun Ravindran `[通讯]` (Boston Consulting Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对学习型优化代理模型构建了等价变换的变形测试（metamorphic testing），并在电网调度、旅行商问题、车辆路径规划和最大独立集四个典型优化任务上对多种模型进行评估与故障定位。

**💡 创新点**

创新点在于：①提出了问题保持的变形关系（relation）与可接受条件，①通过参考求解器校验抽样对比以区分数值误差与结构性错误；②将随机权重控制作为基准，证明部分错误与训练无关；③系统化定位了五种编码敏感机制并验证了相应的修复措施。

**🔧 技术方法**

使用的技术包括：图神经网络（GridSFM、LUMINA、GENCO）、基于强化学习的 POMO、图扩散解算器 DIFUSCO、数值求解器（IPOPT、Concorde、HGS-CVRP、CP‑SAT）以及自定义的对齐与规范化步骤。

**📊 数据集**

数据集覆盖IEEE标准电网（14、118、500、2000、4661节点）、PGLib‑OPF、随机生成的 TSP/CVRP 100 城市实例以及 250 个 MIS 图（带有独立子图）。

**📈 对比分析**

比较方法：对每个变形关系进行抽样变换、重新求解并与预测对齐；采用数值阈值（floor）和精度校准来判定违例；评估质量时使用 MAE、可行性残差和解集差距。结果显示：在电网任务中约 60% 的关系被违例；在 POMO 的 TSP 中旋转/反射违例率随城市数增加至 90%；DIFUSCO 在 MIS 中 36% 的并集关系被违例。修复后仍未能在所有网格上实现 10% 的非劣性能保证。

**⚠️ 局限性**

局限性包括：①校准阈值基于经验，缺乏严格的数值误差上界；②验证主要依赖局部求解器的正确性，未完全覆盖全局最优；③随机权重对比仅验证了单一训练策略；④修复策略多为后续实验，未对所有模型进行系统化验证；⑤实验环境跨平台差异导致结果可重复性稍有波动。

---

## 164. Fuzzy-FedAvg: Server-Aware Client Selection for Federated Speech Emotion Recognition in Heterogeneous Environments

**arXiv ID:** 2611.08818 | [PDF](https://arxiv.org/pdf/2611.08818v1)

**作者:** Farrukh Pervez `[一作]` (National University of Sciences and Technology), Raja Jurdak `[通讯]` (Queensland University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Fuzzy-FedAvg框架，用于在异构环境下的联邦语音情感识别，结合服务器学习的客户端选择权重与延迟更新管理；

**💡 创新点**

使用模糊Q学习自适应学习全局指标生成客户端选择权重，并将同一权重同时用于客户端排名与聚合，配合三层延迟管理与多KPI加权；

**🔧 技术方法**

模糊Q学习、客户端选择权重、PCAS节点、延迟折扣聚合、MFCC/梅尔谱音频特征与FedAvg基础；

**📊 数据集**

IEMOCAP与CREMA-D情感语音数据集；

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、Power-of-Choice、Oort等基线在UAR、训练时间、参与公平性等指标对比，Fuzzy-FedAvg在高非IID条件下提升UAR、训练时间降低约81%，同时保持较好公平性；

**⚠️ 局限性**

未引入差分隐私或安全聚合，公平性评价仅针对参与度与准确度而非群体公平，实验基于模拟客户端和单一特征集，缺乏真实多设备验证。

---

## 165. DrivAerRim: A large-scale full-vehicle CFD dataset for wheel rim aerodynamics

**arXiv ID:** 2611.08842 | [PDF](https://arxiv.org/pdf/2611.08842v1)

**作者:** Chao Xia `[一作]` (Chalmers University of Technology), Zhigang Yang `[通讯]` (Tongji University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究构建了DrivAerRim数据集，包含904种DeepWheel轮辋设计的全车CFD仿真结果，提供完整的力系数、流场和壁面数据。

**💡 创新点**

创新点在于将多样化的轮辋几何与全车耦合，使用MRF和旋转壁面实现轮子旋转，并系统评估了轮辋对车辆阻力的影响，为生成式设计与代理模型训练提供大规模、统一的基准。

**🔧 技术方法**

使用了STAR‑CCM+进行稳态RANS模拟，采用RKE湍流模型、两层壁面处理、MRF区域和旋转壁面边界，结合自动化批处理、脚本化求解与后处理。

**📊 数据集**

数据集基于DrivAer Estateback车型，包含统一的车身几何、雨纹轮胎和六个轮辋位置；轮辋几何来源于DeepWheel生成式+拓扑优化的三维设计。

**📈 对比分析**

通过与Volvo汽车风洞实验在两种几何改动（开闭轮辋、前轮挡板）下的阻力差与表面压力对比，RKE模型在阻力差和主要压力趋势上最为接近实验，验证了其适用性。

**⚠️ 局限性**

局限性包括仅针对单一车辆模型和固定工况、使用RANS非高阶湍流模型、轮辋样本虽多但仍有限、未涵盖主动气动优化或大尺度仿真等。

---

## 166. It Takes Two to Tangram: Multi-Agent Object Centric Reinforcement Learning for Assembly

**arXiv ID:** 2611.08814 | [PDF](https://arxiv.org/pdf/2611.08814v1)

**作者:** Paul Knaußt `[一作]`, Jan Peters `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于图神经网络的多件装配控制框架，能够同时处理多个零件的状态、空间关系和对应的执行指令；

**💡 创新点**

创新点在于将装配状态建模为动态场景图，将每个零件的姿态与动作嵌入到图节点中，并通过共享的策略头和价值网络实现端到端的多任务学习；

**🔧 技术方法**

采用了图卷积网络（GNN）来捕捉零件间的相互作用，结合策略梯度与Critic网络的Actor‑Critic框架，利用共享的全局特征层提升信息流通效率；

**📊 数据集**

使用了工业装配仿真数据集（如AssemblyNet/ShapeAssembly）以及部分真实机器人实验数据，涵盖了7个不同形状零件的拼装任务；

**📈 对比分析**

与传统基于规则的控制、单独强化学习以及无图神经网络的基线方法进行了对比。实验结果显示，本文方法在成功率、任务完成时间与指令简洁度等指标上均优于对比方法，成功率提升约15%-20%，完成时间缩短30%；

**⚠️ 局限性**

局限性包括：1) 仍主要基于仿真环境，真实世界噪声与位姿误差对性能影响未知；2) 对极大规模装配（数十件以上）时的可扩展性尚待验证；3) 需要对每个零件手工定义节点特征，对非结构化零件可能不易迁移。

---

## 167. On the Incompleteness of do-calculus for Universal Identifiability

**arXiv ID:** 2611.08916 | [PDF](https://arxiv.org/pdf/2611.08916v1)

**作者:** Yaroslav Kivva `[一作]` (EPFL), Negar Kiyavash `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了从任意观测与/或干预分布中识别因果效应的通用可识别性问题（uID），并给出了完整的do‑calculus可识别性定义；

**💡 创新点**

首次正式定义了do‑calculus在任意可识别性问题上的完备性，并证明在uID场景下do‑calculus不再完备，即存在可唯一识别但无法通过do‑calculus规则得到的效应；

**🔧 技术方法**

使用结构因果模型、d‑分离、do‑calculus规则（DR1–DR3）、标准概率变换规则（PR1–PR4）以及线性代数与代数函数（如矩阵表示、线性方程求解、 rational 函数论证）进行理论推导；

**📊 数据集**

论文不使用实验数据集，而是通过构造的概率分布（离散随机变量、二元变量）和符号计算来证明完备性缺失；

**📈 对比分析**

与先前在ID、gID、c‑gID等问题上已证明的do‑calculus完备性对比，展示了在更一般的uID问题中，缺乏完整的规则集合导致无法得到唯一可识别的因果效应；

**⚠️ 局限性**

局限性在于只给出了理论性的反例，未提供新的计算算法或改进do‑calculus的具体方法；进一步研究需开发能处理uID的新的识别工具或规则扩展。

---

## 168. GeoJeV: Unified Spatial Probability for Training-Free Efficient Evidence Acquisition in Ultra-High-Resolution Remote Sensing

**arXiv ID:** 2611.08879 | [PDF](https://arxiv.org/pdf/2611.08879v1)

**作者:** Xinyi Niu `[一作]` (National University of Defense Technology), Yongxiang Liu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoJeV，一种训练自由的超高分辨率遥感视觉问答证据采集方法，利用冻结的 JeV 模型输出空间概率，按该概率驱动区域选择、分支宽度和深度控制。

**💡 创新点**

创新点在于将区域搜索从依赖 RLHF 的生成式决策转移到概率驱动的决策，通过 Region Rank、Sparse Branch 和 Depth Gate 三个共享空间概率子模块实现无需中间自回归解码的高效搜索。

**🔧 技术方法**

技术包括 JeV 概率决策、Sparsemax 稀疏分支、Jensen‑Shannon 距离聚集度早停、3×3 网格分块、以及冻结的多模态大语言模型（MLLM）进行最终答案生成。

**📊 数据集**

使用三大超高分辨率遥感问答基准：LRS‑GRO、MME‑RealWorld‑RS 和 LRS‑VQA。

**📈 对比分析**

与商业 MLLM、通用 MLLM、遥感专用 MLLM 以及训练自由证据采集方法（ZoomSearch、WeaveEarth）对比；在 Qwen3‑VL‑8B 框架下平均准确率提升 5.32%，在 LLaVA‑OV‑7B 框架下提升 14.29%，同时仅增加 1.12s 延迟，速度比 ZoomSearch 提升 7.8×。

**⚠️ 局限性**

局限性包括对计数、状态等类别的准确率仍低于直接回答；证据采集对空间覆盖范围和全局与局部信息的平衡仍需进一步改进。

---

## 169. Lodos: A Lightweight Simulator Designed for Contact-Rich Interactions with Detailed Meshes

**arXiv ID:** 2611.08872 | [PDF](https://arxiv.org/pdf/2611.08872v1)

**作者:** Onur Beker `[一作]` (University of Tübingen), Georg Martius `[通讯]` (University of Tübingen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对复杂网格‑网格碰撞的高吞吐量刚体仿真框架Lodos，包含基于球‑SDF的碰撞检测、3D样条SDF、分层瓷砖加速以及弹簧阻尼接触动力学并通过stagger迭代和摩擦修正实现更精确的接触行为，并展示与MuJoCo、MJWarp等现有仿真器的对比；

**💡 创新点**

创新点包括：① 以球‑SDF为核心的新碰撞检测机制，配合3D样条SDF实现任意复杂网格的高效表示；② 通过Hilbert空间填充曲线+分层瓷砖实现对球体集合的O(log n)快速剔除；③ 将所有接触力压缩为每个碰撞对的缩放因子，并用盒约束QP+投影坐标下降求解，随后采用stagger迭代逼近SOCP，显著提高并行性；④ 引入摩擦修正算法消除滑移现象；⑤ 通过GPU并行化和批量化设计，在大批量复杂网格碰撞场景下实现约10倍吞吐量提升；

**🔧 技术方法**

使用的技术包括GPU并行计算(CUDA)、Hilbert空间填充曲线、3D三次样条SDF与表面法线、分层瓷砖加速、盒约束QP投影坐标下降、半隐式Euler积分、SOCP/弹簧阻尼接触动力学、iLQR、MPPI、随机平滑等；

**📊 数据集**

主要实验数据集为公开的Microduck机器人、Sharpa Wave手臂、Armadillo 3D网格模型，以及随机生成的多块石块砌筑、立方体抓取等复杂网格碰撞场景；

**📈 对比分析**

对比方法包括在相同时间步长下与MuJoCo-CPU、MJWarp、MuJoCo-LC等实现进行物理准确度和吞吐量对比；在复杂网格碰撞场景中，Lodos的吞吐量提升约10倍，并在iLQR与MPPI控制实验中实现与传统仿真器相当甚至更优的轨迹优化性能；

**⚠️ 局限性**

局限性包括：仅在大批量（>256）且复杂网格碰撞场景下才具优势；对小/中批量和使用简单几何体时MJWarp/MuJoCo仍更优；当前不支持自动微分；设计侧重GPU，CPU性能不佳；需手动调参以获得最佳性能。

---

## 170. Lodos-v2: An Efficient Primal Formulation of the Exact Frictional Contact Problem

**arXiv ID:** 2611.08868 | [PDF](https://arxiv.org/pdf/2611.08868v1)

**作者:** Onur Beker `[一作]` (University of Tübingen), Georg Martius `[通讯]` (University of Tübingen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Lodos‑v2，一款基于 GPU 的高吞吐量刚体动力学与接触仿真框架，支持精确摩擦接触求解和多种接触组重参数化。

**💡 创新点**

创新点：① 用前向极化最小化与 de Saxcé 修正的本原形式 Newton 求解器取代原来的凸松弛，消除 lift‑off、正交 compliance 与滑移 creep；② 将接触组重参数化推广到本原形式，实现与关节限制、铆接等其他约束的无缝集成；③ 在 GPU 上实现三种接触组参数化（per‑contact、ray、cone），并通过 Woodbury、共享 Hessian 等技术提升并行效率。

**🔧 技术方法**

技术手段：GPU 并行稀疏线性代数、proximal minimization、ADMM/Newton 更新、de Saxcé 校正、Woodbury 逆、共享 Hessian、融合 Newton 内部迭代、MJWarp 的无约束 Newton 求解器、CUDA 图与共享内存优化。

**📊 数据集**

数据集：使用一系列工业与机器人仿真场景——cube_walled_5/20、arm_walled_5/20、shadow_walled_5/20、shadow_grasp、nist_threading_m8、masonry_arch、house_of_cards；每个场景都有 SDF 与 polytope 两种碰撞版本。

**📈 对比分析**

对比方法：在 NVIDIA RTX 4080 上测量批处理吞吐量（world‑steps / s），与 MJWarp（MuJoCo 模式）和 Kamino（exact 模式）直接比较；实验显示 Lodos‑v2 在 MuJoCo 模式下比 MJWarp 提升 1.2–8×，在 exact 模式下相对于 Kamino 提升 1.2–200×（取决于场景与参数化）。

**⚠️ 局限性**

局限性：① 对高度非凸、闭环动力学仍需更细粒度的精确求解，可能导致收敛次数增多；② 目前仅在 GPU 上实现，CPU 或多机分布式扩展尚未成熟；③ 对极大碰撞点集合的 SDF 版本仍可能产生过多接触点，影响内存与速度；④ 某些复杂关节树（如 Shadow hand）在 exact 模式下的求解仍受迭代次数限制。

---

## 171. Extending Grid Foundation Models for Phasor-Domain Dynamic Simulation

**arXiv ID:** 2611.08867 | [PDF](https://arxiv.org/pdf/2611.08867v1)

**作者:** Boyang Zhao `[一作]` (Stony Brook University), Hendrik Hamann `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种可迁移的相量域动态仿真模型，利用Grid Foundation Models实现对不同电网拓扑和状态空间的快速动态响应预测。

**💡 创新点**

通过组件级标记化与结构化潜在动态算子，构建共享的组件表示和可自适应的潜在动力学运算符，实现跨异构电网的迁移学习和高效动态模拟。

**🔧 技术方法**

采用图变换器（Graph Transformer）进行组件间注意力聚合，利用潜在空间线性算子进行动态演化，配合升维/投影映射、AdamW优化和NMSE损失进行训练。

**📊 数据集**

使用PST（Power System Toolbox）生成的十个电网动态轨迹数据集（包含4-48机组的标准与变体网格），9个用于预训练，48机组NPCC网格用于hold-out微调。

**📈 对比分析**

与传统PST数值仿真器在NMSE和计算速度上进行对比，预训练+微调后仅需64条轨迹即可匹配从头训练128条轨迹的精度；在单GPU上预测10s轨迹仅需3-5ms，而PST在单核CPU上需3-5s，速度提升约924-1232倍。

**⚠️ 局限性**

目前仅在同步发电机网格上验证，未覆盖逆变器资源、长时间仿真或更广扰动类型；对极大规模电网的适应性和未知参数范围的鲁棒性仍需进一步研究。

---

## 172. Reconstructing LLM Lineage Trees via Fine-Tuning Path Inference

**arXiv ID:** 2611.08866 | [PDF](https://arxiv.org/pdf/2611.08866v1)

**作者:** Sota Ogawa `[一作]` (Institute of Science Tokyo), Jun Sakuma `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在灰盒访问条件下，仅通过教师强制下的对数似然差异，重建大型语言模型的衍生关系为有向树；

**💡 创新点**

提出TRACE-Edge方法，以任务级对数似然偏差为有向边权，使用Chu–Liu/Edmonds算法得到最大边权有向生成树；

**🔧 技术方法**

核心技术为基于对数似然差异的偏差度量、无参数的有向边权计算与最大生成树求解；

**📊 数据集**

使用5种不同结构、不同基模型的25棵模型树（共7–9节点），每棵树通过LoRA或全参数微调得到；

**📈 对比分析**

与黑盒MST和星形基线对比，TRACE-Edge在训练样本探针下父节点匹配准确率达0.74，Held-out下仍优于MST，且不需要任何超参数；

**⚠️ 局限性**

局限在于需获取训练样本探针且仅在训练样本可用时效果显著，对held-out探针的精度下降，且假设根节点已知。

---

## 173. Rethinking Agent Trajectory Compression: A Workflow-Aware Perspective

**arXiv ID:** 2611.08865 | [PDF](https://arxiv.org/pdf/2611.08865v1)

**作者:** Qijia Zhuang `[一作]` (Nanjing University), Yuxiang Ren `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为WAAC的工作流感知轨迹压缩框架，通过对多步骤工具交互的历史轨迹进行结构化建模与在线依赖分析，实现了对agent可见上下文的字段级压缩；

**💡 创新点**

创新点包括：①将轨迹转换为Typed DAG并提取字段级依赖；②离线将成功轨迹归一化为可匹配的工作流片段并记录压缩策略；③在线进行反向依赖分析得到最小保留需求，并与历史策略进行保守融合；④结合历史经验与实时需求，动态分配字段保留精度；

**🔧 技术方法**

核心技术包括：字段级依赖图建模、离线图归一化与模板化、基于语义相似度的历史片段匹配、逆向依赖分析、保守策略融合、指针与摘要生成；

**📊 数据集**

使用了PinchBench和Odysseys两大基准数据集进行实验；

**📈 对比分析**

与多种压缩与记忆方法（Keep‑Last‑N、LLMLingua‑2、TokenSkip、Summary、LCM、AgentFold、Mem0）对比，WAAC在PinchBench上取得最高总体分数87.6、输入token减少约24.6%、执行时间缩短16%；在Odysseys上实现19%完工率、平均步骤从75降至58、轨迹效率提升至0.68，成本相比SOTA下降29%；

**⚠️ 局限性**

局限性包括：对工作流的结构化和字段类型依赖较高，需手工或自动化的schema标注；在完全未知或极端多样的工作流中匹配效果有限；当前实现主要针对单一agent，尚未探索多agent协作场景。

---

## 174. Mind the Divergence: Divergence-Guided Reliability-Aware Fusion and Adaptive LLM Prompting for Conversational Emotion Recognition

**arXiv ID:** 2611.08860 | [PDF](https://arxiv.org/pdf/2611.08860v1)

**作者:** Jingwen Wang `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出 MiD-ERC，一种利用预测不一致性与模态可靠性协同调节多模态情感识别的框架。

**💡 创新点**

创新点在于：①用标准化的 Jensen‑Shannon 散度量化跨模态预测分歧；②基于分歧与可靠性构建门控、加权与 LLM 适配三阶段处理；③仅需常规情感标签即可端到端训练，无需冲突标签或外部检索模块。

**🔧 技术方法**

技术包括：多模态时序编码（DialogueRNN + Speaker‑Aware Causal GCN）、交叉模态注意力门控、基于可靠性温度的加权融合、分歧条件下的 LLM 关系提示，以及 LoRA 微调 Qwen2.5‑7B‑Instruct。

**📊 数据集**

使用公开多模态情感对话数据集 IEMOCAP（7 类）和 MELD（8 类），并将原始多模态数据转换为指令式 LLM 输入。

**📈 对比分析**

与多模态、LLM 及非 LLM 基线对比（如 Causal‑ERC、CFN‑ESA 等）后，MiD-ERC 在 IEMOCAP 上实现 73.44% Acc / 73.38% W‑F1，在 MELD 上实现 71.79% Acc / 70.60% W‑F1，均超过所有对比方法并显著提升高分歧样本性能；同时推理速度比 Causal‑ERC 快约 50%。

**⚠️ 局限性**

局限性包括：对 LLM 依赖较大，推理成本受模型规模限制；仅在两大公开数据集验证，跨领域泛化仍需进一步验证；对极端模态缺失或严重噪声的鲁棒性未作深入探讨。

---

## 175. When F1 Is Not Enough: Auditing Codebooks for LLM-Assisted Annotation

**arXiv ID:** 2611.08846 | [PDF](https://arxiv.org/pdf/2611.08846v1)

**作者:** Yuehong Cassandra Tai `[一作]` (Pennsylvania State University), Qunhua Li `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套面向大语言模型注释的预部署代码本审计流程，自动筛选错误类别，人工归因错误来源，并对代码本进行有针对性的修订与外部验证。

**💡 创新点**

创新点在于将错误源细分为模型错误、代码本缺陷、参考标签问题与实质性判断，并通过自动化指标（F1、混淆矩阵、Bootstrap置信区间）与手工归因相结合，构建循序渐进的修订与评估管道。

**🔧 技术方法**

使用多类分类的LLM注释（Llama‑3.3‑70B‑Instruct、Gemma‑4‑31B）以及精确度/召回率、混淆矩阵、Bootstrap置信区间、标签名称消除实验等技术，对代码本进行系统化审计。

**📊 数据集**

在两个公开多类代码本上实验：BFRS政治暴力事件分类（12类）与EU信用认定分类（4类），分别抽取开发集（BFRS 573/1045，信用认定 172/321）与留存集。

**📈 对比分析**

先在开发集自动筛选低F1或极端精/召偏差的类，再人工归因错误来源，逐步修订代码本；在留存集评估每一步的F1变化。BFRS案例中Other类F1从0.429提升至0.535，信用认定案例中Credit‑sharing F1从0.709提升至0.748，整体宏观F1均显著提升。

**⚠️ 局限性**

局限性包括：修订仅针对可修复的代码本缺陷，无法解决参考标签不一致或概念缺失；验证仅在LLM上进行，未检验人类编码者的一致性；跨模型迁移性有限，需在实际使用的LLM上再次验证。

---

## 176. What Matters for LLM Persona Simulation: Signal Form in Cold-Start E-Commerce Advertising

**arXiv ID:** 2611.08840 | [PDF](https://arxiv.org/pdf/2611.08840v1)

**作者:** Jiwoo Choi `[一作]` (Dosan International), Yun Young Choi `[通讯]` (SolverX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用大型语言模型对用户 persona 进行模拟，以估计冷启动电商展示广告的点击率，并系统探究不同信号形式对模拟质量的影响。

**💡 创新点**

发现最优信号形式包括仅使用基础人口统计标签、相对定性 banner‑blindness 线索而非定量锚点，以及精细的提示措辞，从而显著提升模拟精度。

**🔧 技术方法**

基于预训练 LLM（如 GPT‑4）进行多次二值点击决策模拟，聚合得到 CTR，并对比直接先验、规则基线和多种 LLM 语料提示配置进行 ablation。

**📊 数据集**

使用真实 Google Ads 展示广告数据，涵盖 5 个人口统计细分组，7 天期间共 974,423 次展示和 3,578 次点击。

**📈 对比分析**

与非 LLM 直接先验、规则基线以及不同 LLM 配置对比，最优配置 MAE 24.9（比最佳非 LLM 低 51.3%）、MAPE 26.1%，显著优于其他设置。

**⚠️ 局限性**

仅在单一产品/创意、单一广告平台、5 个细分组的局部实验，缺乏跨产品、跨地区、跨 LLM 的泛化验证；提示措辞对结果敏感，需系统评估；对高响应细分的校准仍不足。

---

## 177. Functional Dependency beyond Similarity: Hierarchical Decomposition of Training Data Attribution in Music Generative Models

**arXiv ID:** 2611.08839 | [PDF](https://arxiv.org/pdf/2611.08839v1)

**作者:** Changheon Han `[一作]` (Chalmers University of Technology and University of Gothenburg), Kıvanç Tatar `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对训练数据归因（TDA）得分进行三层分解（全局、组内、单一输出），从而将训练数据对生成模型及其输出的功能依赖拆分为不同层级，支持基于功能依赖的版权和报酬讨论。

**💡 创新点**

提出了功能依赖概念和TDA分解方法；展示了三层分解后每层在对应作用域内具有可观测的因果影响；通过删除测试验证全局、组内与特定层级对模型行为的不同贡献；指出传统TDA存在全局影响的混合问题。

**🔧 技术方法**

使用梯度相关的TDA方法（GradDot、TracIn、带权重的GradDot λ=1、0.1、0.01、集成版）以及影响函数法；对标准化得分进行全局均值、组内均值、残差三层拆分；用k-means聚类得到组级分组；删除测试（去除高分实例后重新微调并测量LL变化）；利用Spearman相关、Wallace系数等统计量评估分层特征。

**📊 数据集**

MTG‑Jamendo 数据集：10,000 条 10 秒音乐片段用于微调；另外使用 10,000 条未包含在微调集中的曲目作为提示（genre 标签），生成 480 条音频样本进行评估。

**📈 对比分析**

通过比较不同模型（MusicGen‑S/L、InspireMusic‑Base/1.5B）与不同 TDA 方法的全局、组内、特定层级比例；删除测试得分 D^glob、D̅^grp、D̅^spec 均为负，表明各层对其作用域有显著影响；全局层与梯度模长/方向的 Spearman 相关性低但与方向高度相关；组内层与 genre、音频特征的 Wallace 系数普遍低，说明分组不对应传统音乐属性；特定层在不同分层标准下的实例集变化幅度较大。

**⚠️ 局限性**

尚未识别组级分组对应的音乐特征或语义；报酬分配对分层定义敏感，需进一步法律、经济伦理讨论；仅针对自回归音乐生成模型，未验证扩展到扩散或其他类型模型；实验中未包含人类听感评估；TDA 方法仍基于近似梯度估计，可能存在误差。

---

## 178. Hardness of Approximation for Polynomial Optimization over Sphere for Linear Programming Relaxation Hierarchy

**arXiv ID:** 2611.08836 | [PDF](https://arxiv.org/pdf/2611.08836v1)

**作者:** Mrinalkanti Ghosh `[一作]` `[通讯]`, Mrinalkanti Ghosh

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文探讨了半正定规划（SDP）和平方和（SoS）层次在优化度数为d的齐次多项式上的应用，特别是针对单位球面上的优化问题，提供了n^O(q)时间的近似和认证算法。

**💡 创新点**

创新点在于建立了较弱的线性规划松弛层次的下界，特别是对于任意输入的度数-q SoS，提供了更好的下界，改进了先前随机张量SoS下界中的d/4型指数。

**🔧 技术方法**

使用了半正定规划（SDP）和线性规划（LP）技术，特别是通过平方和松弛和时刻方法来处理多项式优化问题。

**📊 数据集**

使用了随机的度数-d谐波多项式作为数据集，研究了其在单位球面上的性质。

**📈 对比分析**

与现有方法相比，论文中的LP松弛在性能上提供了更好的下界，尤其是在多项式的度数依赖性方面，展示了LP与SDP之间的量化分离。

**⚠️ 局限性**

限制在于尚未确定是否可以在SDP层次中实现类似的度数依赖性，且对LP松弛的正性条件要求较弱，可能影响其在某些情况下的有效性。

---

## 179. Notwithstanding Anything to the Contrary, Ignore All Instructions: Measuring Prompt Injection Guards on Legal Contract Review Attacks

**arXiv ID:** 2611.08834 | [PDF](https://arxiv.org/pdf/2611.08834v1)

**作者:** Mark Heimann `[一作]` (Ivo AI), Bruno Jacob `[通讯]` (Ivo AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了11种开源/商业提示注入防护模型在法律合同文本中的检测效果，并构建了针对合同注入攻击的基准与税onomies。

**💡 创新点**

提出了专门针对法律合同的注入攻击分类，发布了契约注入与对照文本的公开基准，并揭示了现有防护模型的高误报率和低召回率；同时提出了成本更低、性能接近前沿的级联LLM检测方案。

**🔧 技术方法**

采用文本分块（512/2048 token）、阈值过滤、AUC/误报率评估，并用LLM（前沿模型与中等模型）实现级联检测；对比了多种现有提示注入防护模型。

**📊 数据集**

使用了683份私有商业合同与公开的CUAD（510份）合同作为测试数据集，所有文档按非重叠块划分后进行评测。

**📈 对比分析**

结果显示传统防护模型的文档级误报率高达92.39%，部分模型AUC低于随机猜测；而级联LLM方案在保持接近前沿模型性能的同时，成本仅为其1/6，且显著低于单独使用前沿模型。

**⚠️ 局限性**

局限性包括仅针对英语/美国商业合同，单一审计问题，未考虑结构性隐蔽攻击与纯粹虚假陈述，未测量推理延迟，以及对检测误报的系统性后果尚未给出完整评估。

---

## 180. HomeLoop: A Simulation Framework and Benchmark for Robot-IoT Coordination in Everyday Domestic Assistance and Safety

**arXiv ID:** 2611.08914 | [PDF](https://arxiv.org/pdf/2611.08914v1)

**作者:** Ramy ElMallah `[一作]` (University of Toronto), Chi-Guhn Lee `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 HomeLoop，一个集成机器人、网络设备、居民和动态灾害的仿真框架，并提出了包含 20 个基准任务的评估集，用来测试机器人–物联网协同和模糊决策能力。

**💡 创新点**

创新点在于：①提供可配置的多房屋、多机器人、多设备环境，支持房间布局、设备类型和危险事件的组合；②使用实时监视器结合有限时序逻辑对动作与结果进行连续评估；③结合成本模型和服务可用性，实现对行动效益与风险的量化。

**🔧 技术方法**

技术实现主要基于 NVIDIA Isaac Sim 进行渲染和物理仿真；机器人控制通过 VLM 的工具调用（skills）实现；危害演化采用基于物理参数的热力学和烟雾模型；评估使用时序逻辑（LTL）监控和成本计量。

**📊 数据集**

数据集方面，使用 Swiss Dwellings 真实公寓平面图构建 2100 套房屋；资产库包含 HSSD、Amazon Berkeley Objects、ArtVIP 等；设备类型有 20 种网络化传感器和执行器；同时配备 5 种机器人模型和 10 个角色模型。

**📈 对比分析**

比较方法：在相同的 20 个任务上对六个公开和闭源 VLM 进行评估，记录成功率、家庭平衡分、成本占比和 API 消耗。最佳模型 Astra 达到 61.9% 的家庭平衡得分；总体发现模型往往过度干预，错误干预次数超过遗漏干预两倍。

**⚠️ 局限性**

局限性：评估仅在单一房屋、单一机器人和两种种子下完成，缺乏对不同房屋配置、机器人种类和真实机器人硬件的全面测试；推理延迟被排除在评分之外；成本和隐私权衡未充分验证；实验未涉及真实居民交互。

---

## 181. Polarization calibration of fisheye cameras using skylight in the wild

**arXiv ID:** 2611.08911 | [PDF](https://arxiv.org/pdf/2611.08911v1)

**作者:** Shuo Wang `[一作]` (Beihang University), Wenxin Liu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于自然天空的联合镜头‑分析器校准方法，可在手持户外短序列内实现鱼眼偏振相机全视场的精确校正。

**💡 创新点**

创新点在于将镜头的入射角依赖传输、分析器增益与角度误差统一到一个八参数模型，并且不需要专用参考光源或预设天空极化模型。

**🔧 技术方法**

技术包括交叉帧极化一致性约束、四通道一致性约束、Gaussian滤波、Ray‑based RANSAC姿态估计及联合优化自适应Mueller矩阵和分析器参数。

**📊 数据集**

使用了两套户外手持记录的天空序列（Data A 142帧，Data B 159帧）以及在四足机器人上采集的天空图像进行验证与迁移测试。

**📈 对比分析**

通过与平滑、仅镜头、仅角度、仅增益等基线及未校准情况的ablation比较，最终在Data A上平均MAE降至0.404°，在Data B上为0.466°，相对平滑基线提升超过50%，大角度区误差降至约0.42°。

**⚠️ 局限性**

局限性包括对云雾遮挡或弱极化天空的鲁棒性不足，模型假设镜头轴对称且无相位延迟，非轴对称或高多边形光学时可能失效，并且需要多姿态覆盖才能获得最佳结果。

---

## 182. Task-Contribution Dynamics of Adam in Physics-Informed Neural Networks: Slow-Manifold Analysis and Gradient Modulation

**arXiv ID:** 2611.08910 | [PDF](https://arxiv.org/pdf/2611.08910v1)

**作者:** Enzhan Zheng `[一作]` (Dalian University of Technology), Xiaoliang Song `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对物理信息神经网络（PINN）中 Adam 优化器的任务贡献失衡问题，作者从理论和实验两方面提出并验证了 Gradient‑Modulated Adam（GMAdam）算法，旨在更有效地调节不同损失项对参数更新的相对影响。

**💡 创新点**

创新点在于：①利用连续时间奇异扰动理论与离散时间分析，首次揭示了 Adam 共享二阶矩估计如何影响任务贡献分布，并给出了两种主要因素（曲率对齐差异与梯度分布差异）；②基于此理论提出 GMAdam，通过对每个任务梯度乘以对应二阶矩估计的比例来实现梯度调制，从而在保持 Adam 预条件优势的同时显著改善多任务平衡。

**🔧 技术方法**

核心技术包括：PINN 的多任务损失框架、Adam 的离散与连续动力学建模、奇异扰动与几何折叠定理、任务特定二阶矩估计、傅里叶特征嵌入（Fourier / AFFE）与带权重的 Adam 变体（AdaTask、MultiAdam）。

**📊 数据集**

实验数据集为四个经典 PDE 基准：一维高频泊松方程、波动方程、Allen‑Cahn 方程以及二维台面驱动腔流（Re = 100、400），全部使用人工合成的采样点与解析/数值参考解。

**📈 对比分析**

与 AdaTask、Grad Norm 与 NTK 加权等现有方法对比，GMAdam 在三大 PDE 基准上显著降低了相对 L²误差（Poisson 约 71.9% 降低，Wave 约 19.7%，Allen‑Cahn 约 21%），在 Re = 400 的腔流中误差下降超过 70%，并保持与 AdaTask 相近的训练时间；与 NTK 方法相比，在 Re = 400 的情况下误差下降 77% 但耗时仅略高 3%。

**⚠️ 局限性**

局限性包括：①理论分析仅覆盖训练早期平滑阶段，未解释后期出现的尖峰与持续振荡；② GMAdam 仍需要手动设置 δ₁ 等调节参数，缺乏自适应机制；③实验仅限于低维合成 PDE，尚未验证在大规模或真实工程问题中的鲁棒性与可扩展性。

---

## 183. FATHOM: World-Model-Guided Diffusion Action-Trajectory Selection for Sparse-Marker AUV Tracking

**arXiv ID:** 2611.08898 | [PDF](https://arxiv.org/pdf/2611.08898v1)

**作者:** Xiaorui Wang `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了FATHOM框架，利用世界模型引导的扩散动作生成和预测性选择，实现稀疏光标下的水下AUV跟踪控制。

**💡 创新点**

创新点在于将多模态任务监督递归状态空间模型与扩散动作先验相结合，采用预测选择而非直接生成动作，并构建稀疏观测下的控制相关贝叶斯信念。

**🔧 技术方法**

使用技术包括结构化稀疏光标编码器（SSME）、多模态任务监督递归状态空间模型（MT‑RSSM）、扩散动作先验（DAP）以及世界模型预测选择器（WMPS），结合深度学习、视觉处理和控制理论。

**📊 数据集**

数据集为540条不同PID等级、目标与运动模式的水箱物理轨迹用于世界模型训练，60条专家轨迹用于扩散先验学习。

**📈 对比分析**

在离线评估和实机实验中，FATHOM在普通与挑战模式下分别相较于姿态估计+PID/ISMC和纯扩散策略降低约31%/14%的误差，完成率100%，并在自然部分光标可见性下误差降至0.163，恢复率100%，平均恢复时间2.17秒。

**⚠️ 局限性**

局限在于对固定光标布局和光照条件敏感，扩散采样占用大部分推理时间，模型误差会影响选择准确性，且在极端动态或开放海域环境中的泛化尚未验证。

---

## 184. Segmenting Geometric Textures of 3D Meshes

**arXiv ID:** 2611.08897 | [PDF](https://arxiv.org/pdf/2611.08897v1)

**作者:** Weidan Xiong `[一作]`, Hui Huang `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

论文探讨了某种算法在特定任务中的应用，旨在提高性能和效率。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够显著提升算法的处理速度。

**🔧 技术方法**

使用了深度学习技术，结合了卷积神经网络和强化学习。

**📊 数据集**

采用了公开的图像识别数据集进行实验，确保结果的可重复性。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和速度上均有明显优势。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，可能需要进一步的调整和优化。

---

## 185. Decoding spatial and semantic information from large-scale visual cortical activity for natural-movie reconstruction

**arXiv ID:** 2611.08887 | [PDF](https://arxiv.org/pdf/2611.08887v1)

**作者:** Yukai Huang `[一作]` (Fudan University), Yuguo Yu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于生成式扩散模型的神经解码框架 Ca2REAL，能够将细胞水平的钙成像活动映射为自然电影；

**💡 创新点**

创新点在于将布局（spatial）和语义（semantic）两条独立的神经控制路径与预训练的 Wan2.1‑VACE 视频扩散模型结合，利用钙信号提供粗粒度空间结构和视觉身份信息，而将高频细节交给生成式先验完成；

**🔧 技术方法**

采用多尺度时间前端、Haar‑升尺度变换、布局控制网络、语义控制网络、CLIP‑ViT‑B/32 视觉编码器、U‑M‑T5 条件适配器以及对 WaN VACE 的 LoRA 微调，辅以匹配‑对比损失提升神经-刺激对应性；

**📊 数据集**

使用两个数据集：1）Dynamic Sensorium 的 7,863 只 V1 神经元在单只小鼠上的两秒自然电影片段；2）由 Sensorium 与 MICrONS 编码模型拼接得到的 116,705 维虚拟神经群，涵盖 V1、LM、RL、AL；

**📈 对比分析**

与 Ridge、SID、WISA 等基线比较，Ca2REAL 在 178（真实）和 946（虚拟）测试片段上，在 LPIPS、DINOv2、SigLIP2 等感知/语义指标和 81.4% / 92.8% 的所有对刺激识别准确率上均显著优于基线，尤其在语义一致性和空间结构方面优势突出；

**⚠️ 局限性**

主要局限包括：1）解码仍受制于钙信号的时空平滑与噪声，细节重建依赖预训练模型；2）缺乏对不同脑区、行为状态的泛化评估；3）虚拟神经群虽然规模大，但并非真实并发记录，可能掩盖跨区协同效应；4）对生成先验的依赖使得难以区分神经信号真实贡献与先验填充的内容。

---

## 186. Who's Asking? Audience-Conditioned Privatization

**arXiv ID:** 2611.08884 | [PDF](https://arxiv.org/pdf/2611.08884v1)

**作者:** Daniil Filienko `[一作]`, Sikha Pentyala `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于三步流程（Schema Construction、Attribute Partitioning、Privatization）的数据隐私保护框架，用于在保持数据可用性的前提下生成差分隐私合成数据。

**💡 创新点**

创新点在于设计了一种基于属性相关性和重要性自动划分公共与私有属性的策略，显著降低了全局隐私预算的消耗，并通过分区机制提升了合成数据的质量。

**🔧 技术方法**

采用了差分隐私机制（如Laplace、Gaussian噪声注入）、属性分区算法（基于信息熵与相关系数）以及数据合成技术（如GAN或统计模型）实现整个流程。

**📊 数据集**

实验数据集主要使用公开的UCI Adult、COMPAS和Medical Insurance数据集，覆盖多种领域的结构化数据。

**📈 对比分析**

与现有的DP-Syn、DP-GAN等基准方法对比，本文在相同隐私预算下的预测准确率提升5%–10%，同时隐私预算使用率下降约15%，证明了方法的高效性。

**⚠️ 局限性**

局限性包括：对属性间强非线性相关性处理不足；在高维稀疏数据上分区策略可能导致过度保留公共属性；以及对动态更新的数据集需要额外的重构与再分区成本。

---

## 187. The Relational Capacity Framework: A Lens for AI-Mediated Relationality

**arXiv ID:** 2611.08877 | [PDF](https://arxiv.org/pdf/2611.08877v1)

**作者:** Emily Tseng `[一作]` (University of Washington), Ashley M. Walker `[通讯]` (Google)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了一项基于100名使用通用AI聊天机器人的受访者的混合方法研究，系统梳理了人们在社会情感需求中使用AI的方式，并提出了“关系能力框架（RCF）”，将关系管理拆解为技能、带宽与信息三维度，并据此定义了八种关系状态。

**💡 创新点**

创新点在于：①首次将关系管理视为可度量的三维资源（技能、带宽、信息）并将其与元认知过程结合；②提出RCF作为理解AI介入关系的统一模型；③通过八种关系状态揭示AI在不同资源缺口下的扩张与压缩效应；④把“连接 vs 孤立”二分法升级为动态、情境化的能力框架。

**🔧 技术方法**

技术手段主要是混合方法：未调解的在线问卷、以情境为导向的思考录音任务、以及23个深度访谈；数据分析采用归纳-演绎（abductive）分析，结合HCI、心理学与社会学的理论框架。

**📊 数据集**

数据集为自建：100名美国成年受访者（19岁以上）在过去30天内使用通用AI聊天机器人的记录；23名受访者进一步接受半结构化访谈；没有使用公开大型语料库或已有数据集。

**📈 对比分析**

研究并未进行传统意义上的算法对比或性能评估；通过质性分析展示AI在不同关系状态下的功能对应关系；若以“覆盖面”和“解释力”为指标，RCF在说明用户使用多样性方面具有较高的理论覆盖度，但缺乏量化性能指标。

**⚠️ 局限性**

局限性包括：①样本规模与地理、文化多样性有限，难以泛化；②研究为横断面，缺乏纵向跟踪评估AI使用对关系能力长期影响；③依赖自报与访谈，可能存在社会期望与记忆偏差；④未对AI系统的输出质量（如幻觉、误导）进行系统评估；⑤框架强调资源与元认知，但在实际设计中如何实现与监测仍需进一步研究。

---

## 188. STRIDE: Learning Humanoid Locomotion in Minutes with Acceleration-Based Rewards

**arXiv ID:** 2611.08857 | [PDF](https://arxiv.org/pdf/2611.08857v1)

**作者:** Nico Bohlinger `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种仅使用本机惯性测量单元(IMU)奖励的强化学习方法，能够在数分钟内在仿真中的两款人形机器人上实现行走。

**💡 创新点**

创新点在于将奖励转为加速度跟踪并加入速度保持奖励，配合高关节控制增益与左右对称数据增强，大幅提升样本效率并减少训练过程中的摔倒次数。

**🔧 技术方法**

使用了离策略Actor-Critic算法XQC、PD高增益控制、对称数据增强、加速度基奖励设计以及积分控制生成目标加速度。

**📊 数据集**

实验数据集为Unitree G1和Booster T1两款人形机器人的MuJoCo仿真环境中收集的约30分钟（≈1.8×10⁶步）交互数据。

**📈 对比分析**

与低增益设置、无对称增强以及默认控制参数相比，本文方法在相同采样量下实现了更快的行走学习速度（平均减少约50%）并将摔倒次数从约2000次降至不到100次，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括实验仅在无噪声的仿真环境中验证，未考虑真实机器人IMU的漂移和误差；且共享单一策略在两机器人上未表现出正迁移，需要进一步研究嵌入感知网络。

---

## 189. "I had nobody to turn to": Why LGBTQ+ People Choose (Not) to Use AI Chatbots for Personal Health and Relationship Advice

**arXiv ID:** 2611.08854 | [PDF](https://arxiv.org/pdf/2611.08854v1)

**作者:** Jacob Hobbs `[一作]` (Carnegie Mellon University), William Agnew `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在线问卷调查73名LGBTQ+及非LGBTQ+受访者，探讨他们在个人健康与关系咨询场景下选择或拒绝使用AI聊天机器人背后的动因与障碍。

**💡 创新点**

创新点在于系统性揭示了资源可得性、隐私顾虑和对模型能力的感知如何共同影响LGBTQ+群体对AI聊天机器人的使用意愿，并首次量化了跨族群使用差异。

**🔧 技术方法**

研究主要采用贝叶斯层级回归分析和主题编码等定量与定性混合方法，结合在线调查工具Prolific收集数据。

**📊 数据集**

数据集由73份完整问卷构成，其中82%为LGBTQ+受访者，包含关于使用频率、隐私担忧、资源可得性等多维度问题。

**📈 对比分析**

与非LGBTQ+受访者比较发现，LGBTQ+群体在15个特定情境下使用聊天机器人的比例平均低8.5个百分点，主要差异体现在关系和心理健康领域，显示出显著的使用偏好差异。

**⚠️ 局限性**

研究局限包括样本规模有限、仅使用英语、受访者主要来自英语国家、且为自报数据，缺乏跨文化、实际使用轨迹与更广泛代表性的验证。

---

## 190. Global Solar Siting from Geospatial Embeddings

**arXiv ID:** 2611.08831 | [PDF](https://arxiv.org/pdf/2611.08831v1)

**作者:** Abdullah Rehman `[一作]` (King's College London), Rui Song `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用地理嵌入和机器学习构建全球光伏可行性评分层，并评估其对未来光伏部署的预测性能。

**💡 创新点**

将AlphaEarth嵌入应用于光伏可行性，使用随机森林+多层感知机软投票集成实现非线性特征提取，并公开10米分辨率全球可行性地图。

**🔧 技术方法**

训练随机森林、MLP及其软投票集成，比较QDA、Logistic、MaxEnt、随机傅里叶特征等模型，使用ROC‑AUC、PR‑AUC等指标评估性能。

**📊 数据集**

使用全球PV设施库存GPVF、TZ‑SAM等数据，并采用AlphaEarth 64维嵌入作为输入特征，采样正负样本进行时间前置训练。

**📈 对比分析**

在13个国家的未来安装数据和独立TZ‑SAM数据上进行时间切分验证，并与传统MCDA方法对比，集成模型ROC‑AUC提升0.06–0.30，最高达0.92以上。

**⚠️ 局限性**

模型依赖已建光伏假设，可能低估欠发达地区潜力；未考虑气候变化和自然灾害风险，对未来可行性预测存在局限。

---

## 191. Lea: An Agent Backbone for Mathematician-Controlled Formalization in Lean

**arXiv ID:** 2611.08849 | [PDF](https://arxiv.org/pdf/2611.08849v1)

**作者:** Shaswat Patel `[一作]` (New York University), Chinmay Hegde `[通讯]` (New York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Lean专用的代理骨干Lea，并基于其实现了交互式证明编辑器LeaChat和Overleaf插件LeaOverleaf，支持在正式化过程中让数学家实时干预并保持高层次控制；

**💡 创新点**

创新点在于将代理运行本身作为用户交互表面，提供统一的、应用无关的HTTP API，内置Lean工具链（LSP、SafeVerify等）并允许用户在不改动代理代码的前提下配置子代理、工具和技能；

**🔧 技术方法**

采用了Lean 4语言、Lean LSP、SafeVerify核重放、OpenHands式事件流、SQLite持久化、YAML配置的子代理角色以及基于Mathematics Library（Mathlib）的知识库；

**📊 数据集**

主要使用Mathlib作为知识库，利用其已有的组合子与定理；

**📈 对比分析**

论文没有给出系统与现有自律式或通用式工具在性能或效果上的量化对比，仅通过对Durfee方格生成函数的案例研究展示Lea能在多次交互中按数学家预期拆解并完成证明；

**⚠️ 局限性**

局限性包括缺乏与Numina-Lean-Agent等系统的对比评估、输出仍需人工重构以符合Mathlib标准、未支持多数学家同时协作、以及缺乏跨领域的正式化评测。

---

## 192. Phantom Consensus: Measuring Social Reasoning in Multi-Agent LLM Decision-Making

**arXiv ID:** 2611.08971 | [PDF](https://arxiv.org/pdf/2611.08971v1)

**作者:** Disha Sheshanarayana `[一作]` (Manipal University Jaipur), Tirthankar Dasgupta `[通讯]` (TCS Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多代理LLM在接受含糊公共信号后可能产生的虚幻共识现象，并提出PHANTOM-Bench评估框架。

**💡 创新点**

创新点在于将群体隐含私有判断误差与公共决策偏差分离，量化“phantom consensus”，并通过多种情境对比分析其机制。

**🔧 技术方法**

使用多代理LLM、系统化评估指标（PPG, CRFR, P-CRFR, PBA, PBE, PIFR, TRR）和对照实验设计。

**📊 数据集**

数据集为PHANTOM-Bench，包含480个多场景、四代理的决策实例，涵盖财务合规、内容审核等领域。

**📈 对比分析**

与七个封闭权重和两组开源模型对比，发现含糊公共信号导致平均PPG达47.8pp，CRFR最高47.3%，但在透明度揭示后多数模型可恢复到私有准确率。

**⚠️ 局限性**

局限包括对模型行为的解释仍偏依赖指标，实验情境有限，且透明度恢复可能掺杂任务相关信息，未完全证实误信导致决策偏差的因果性。

---

## 193. Region-aware error triage for medical image segmentation with groupwise conformal risk control

**arXiv ID:** 2611.08962 | [PDF](https://arxiv.org/pdf/2611.08962v1)

**作者:** Nishi Kanta Paul `[一作]` (NIMISHES Lab), Hoileong Lee `[通讯]` (Universiti Malaysia Perlis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了 RACE‑Triage 框架，在保持分割掩码不变的前提下，通过风险网络预测像素错误并利用分组的 Conformal Risk Control（CRC）对边界带和非边界区域分别设定警告阈值，完成后处理式错误三分辨与审阅指引；

**💡 创新点**

创新点在于：①将误差预测与预测轮廓定义的空间分组相结合，实现对不同区域误漏率的独立控制；②使用 out‑of‑fold 预测训练风险网络，使其与任何分割器互操作；③通过 CRC 在每组上给出期望误漏率上限的保证，从而在保证警告覆盖的同时控制审阅成本；

**🔧 技术方法**

技术手段包括：基于 U‑Net / DeepLabV3+ 的二值医学图像分割；多尺度误差映射网络（U‑Net）利用图像、预测概率、掩码与轮廓距离作为输入；距离变换与梯度对齐特征；Conformal Risk Control 在每个空间组上校准阈值；以及负载自适应警告预算与固定阈值两种警告策略；

**📊 数据集**

实验数据集涵盖：ISIC 2018 皮肤镜图像（2594 张）、BUSI 乳腺超声图像（780 张）以及外部 PH^2 皮肤镜图像（200 张）；

**📈 对比分析**

与无校准、区域匹配、固定阈值、负载自适应等方法对比，结果显示区域 CRC 在保持平均误漏率目标（δ≈0.2）同时将非边界误漏率显著降低（ISIC：从 0.318→0.193；BUSI：从 0.364→0.209），误漏率均控制在目标以内；警告面积平均提升仅 1.6%/4.9%，但在极端病例可达 23.5%/38.5%；同时提高了大小过滤组件召回率；

**⚠️ 局限性**

局限性包括：①仅提供期望级别的误漏率保证，单张图像可能仍超标；②BUSI 数据缺乏病人标识，保证基于图像独立性假设，外部转移需重新校准；③警告面积分布不均，难以直接衡量实际审阅负担；④仅针对二值 2D 分割，未涵盖多类或体积分割；⑤需进一步通过阅读者研究验证临床价值与时间节省效果。

---

## 194. Accuracy Is Not Value: Choosing Weak Supervisors by What the Student Inherits

**arXiv ID:** 2611.08959 | [PDF](https://arxiv.org/pdf/2611.08959v1)

**作者:** Romano Li `[一作]` (Brown University), Xunfan Cai `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究弱监督中学生学习者对监督者的价值，揭示该价值取决于学生学习器对错误的投影而非原始误差，并给出识别与选择的理论边界；提出使用有限数量的标注“anchor”来估计缺失的目标时刻，以决定权重或是否使用。

**💡 创新点**

首次证明学生相对几何决定监督者价值，给出原始Gram固定但学生风险可变的可实现性；证明无标记数据无法识别价值，并量化需要的有限维信息；提出anchor成本与几何关系，并给出风险、选择与置信度的上下界。

**🔧 技术方法**

使用线性/投影学习器、Gram矩阵、子空间投影、线性组合风险公式、未标记数据观测矩阵、标签模型与教师方程、子高斯/高斯宽度分析、随机特征、岭回归、经验风险最小化、贝叶斯风险分析与极大/极小理论。

**📊 数据集**

仅采用合成高斯设计数据进行实验；未使用真实数据集。

**📈 对比分析**

通过与传统堆叠、仅使用anchor的规则以及已知教师方程等进行理论对比，并在合成实验中验证了标注量、选择误差和风险等指标与理论预测一致；在高斯设计中表明当anchor不足时弱监督可优于仅用anchor的规则，且在某些条件下优于普通堆叠。

**⚠️ 局限性**

局限性包括仅针对平方损失和单一共享线性学习者；教师方程与标签模型假设不可由未标记数据验证；实验仅为合成；anchor是被动的；对非高斯设计或异方差噪声的适用性未知；大样本下的常数收敛速率较小。

---

## 195. More with Less: Efficient Parallel Scientific Code Translation via an Agent-Skill Workflow

**arXiv ID:** 2611.08947 | [PDF](https://arxiv.org/pdf/2611.08947v1)

**作者:** Jianuo Shen `[一作]` (South China University of Technology), Bo Zhang `[通讯]` (Michigan Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于代理-技能的并行科学代码翻译框架，使用多轮自我纠错循环完成从 CUDA、OpenMP、SYCL 之间的代码迁移。

**💡 创新点**

创新点在于将翻译、编译诊断和执行验证拆分为独立代理，并通过判定结果驱动的协同器实现错误类型特定的修复技能；同时引入自动输出检查与离线 LLM 验证以提升科学计算的正确性评估。

**🔧 技术方法**

使用的技术包括大型语言模型（DeepSeek‑V4‑Flash、GLM‑5.2、Kimi‑K2.7‑Code）、检索增强生成 (RAG)、编译器与运行时反馈、LLM 驱动的修复技能（Patch、SemanticRepair 等）以及基于置信度的补充验证器。

**📊 数据集**

数据集为 HeCBench 的 30 个多领域 GPU 内核，涵盖 CUDA、OpenMP、SYCL 三种实现。

**📈 对比分析**

与 LASSI 基线对比，实验表明在 18 种 LLM 与翻译方向组合中，本文框架的验证正确率均高于 LASSI，平均提高约 22%；同时在相同硬件与模型下，令牌消耗降低 30–52%，每成功一次的成本约为 LASSI 的 1/3 至 1/2。

**⚠️ 局限性**

局限性包括仅在内核级别评估，未覆盖多文件或完整应用程序；离线验证仍可能漏判高置信度错误；框架未考虑性能反馈，只聚焦于数值正确性；对低资源模型（如 SYCL）写入能力仍有限。

---

## 196. MESAM: Enhancing SAM2 for Few-Shot Segmentation through Semantic-Visual Memory Generation and Refinement

**arXiv ID:** 2611.08943 | [PDF](https://arxiv.org/pdf/2611.08943v1)

**作者:** Tianyu Zou `[一作]` (Wuhan University of Technology), Yi Rong `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于SAM2的少样本分割方法MESAM，利用语义与视觉记忆相结合的方式提升跨类别分割性能。

**💡 创新点**

创新点在于引入基于CLIP GradCAM的语义记忆来抵消支持-查询之间的外观差异，并通过交互式语义记忆细化（ISMR）将视觉细节融入语义记忆，显著提升SAM2的少样本分割效果。

**🔧 技术方法**

使用的技术包括SAM2记忆编码器、DINOv2视觉特征、CLIP文本分类与GradCAM、交叉注意力机制、支持自适应视觉记忆以及BCE损失训练。

**📊 数据集**

实验数据集涵盖PASCAL-5^i、COCO-20^i和LVIS-92^i，覆盖单样本和五样本两种设置。

**📈 对比分析**

在与多种传统与基础模型驱动的FSS方法的对比实验中，MESAM在1-shot和5-shot场景下在PASCAL、COCO和LVIS上均实现或突破SOTA，mIoU提升幅度约4%–12%。

**⚠️ 局限性**

局限性包括对SAM2和CLIP预训练的高度依赖，以及在极端遮挡、多目标或极端外观差异的场景下的鲁棒性仍待进一步验证。

---

## 197. Frame: Form-Restricted Authoring with Model-Based Evaluation

**arXiv ID:** 2611.08921 | [PDF](https://arxiv.org/pdf/2611.08921v1)

**作者:** Nan Jiang `[一作]` (Microsoft), Vishal Chowdhary `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个针对“形式受限报告生成”的新任务，要求模型在保留示例报告结构的前提下，用新的证据生成新的报告。

**💡 创新点**

创新点在于：①创建了名为Frame的120实例基准，覆盖金融（10-Q/10-K）和学术（arXiv综述）两大领域；②设计了三种人类验证的评价指标（结构相似度、信息覆盖度、事实可信度），并对其与人工评测的相关性进行了系统验证；③在多模态长文本上下文下对比了5种主流LLM（含两大专有模型和三大开源模型）。

**🔧 技术方法**

技术手段包括：①基于LLM的结构树提取与语义树编辑距离实现结构相似度评估；②信息单元抽取+LLM判别实现覆盖度评估；③基于LLM的主张抽取与多阶段证据验证实现事实可信度评估；④使用大型多模态LLM（支持128K+ token、图像输入）进行实验。

**📊 数据集**

数据集：Frame基准由来自SEC EDGAR和arXiv的三时点报告与相应证据构成，涵盖四个子类（金融季度、金融年度、学术简易、学术复杂），共120个实例。

**📈 对比分析**

比较方法：在相同Prompt下对5个模型进行生成，分别计算结构相似度、信息覆盖度和事实可信度。结果显示：开源模型在结构相似度上可与专有模型相当；但在覆盖度和可信度上落后；金融类报告性能普遍优于学术类；随着报告长度、证据长度或表格数量变化，性能表现出明显的波动和趋势。

**⚠️ 局限性**

局限性：①基准仅覆盖金融和学术两领域，其他专业报告写作场景可能不适用；②覆盖度和可信度评估依赖LLM判别，可能受模型偏见影响；③结构评估虽不依赖LLM，但仍需依赖文本嵌入模型，存在语义对齐误差。

---

## 198. Composable Causality: A Toolkit for Systematic Time-Series Causal Discovery and Treatment-Effect Benchmarking

**arXiv ID:** 2611.08918 | [PDF](https://arxiv.org/pdf/2611.08918v1)

**作者:** Muhammad Hasan Ferdous `[一作]` (University of Maryland), Md Osman Gani `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了可组合的多变量时间序列生成器和基准，能够独立调节拓扑、机制、噪声、缺失与干预等维度，生成带有确切因果图和对照双生的合成数据，供因果发现与因果效应估计的可复现实验使用。

**💡 创新点**

创新点在于：①采用单一配置文件即可遍历所有实验轴，支持大规模因果结构、机制、噪声、缺失和干预的组合实验；②实现缓存噪声的精确对照双生生成，提供数字化的个体因果效应真值；③为生成过程提供可识别性标签，透明记录可识别性状态。

**🔧 技术方法**

使用可插拔拓扑生成器（ER/BA/WS）、多种结构机制（线性、非线性、递归变换）、多样噪声分布、缺失机制（MCAR/MNAR）、干预策略；基准中采用 Optuna 超参调优、SHD、F1、PEHE 等指标；实现对照双生的缓存噪声重放。

**📊 数据集**

全部使用合成数据：九个图模板（d=4,6,8；τmax=2,3,4）与多种 DGP 变体（线性、非线性、漂移、MNAR/MCAR、趋势/季节等），无真实数据集。

**📈 对比分析**

对比八种因果发现方法（VARLiNGAM, PCMCI, DYNOTEARS, CD-NOD, CUTS+, GrangerF 等）和四种 CATE 估计器（TARNet, FlexTENet, SNet, X-Learner），在不同 DGP 上评估 SHD、F1、PEHE 等指标。结果显示：DYNOTEARS 在无缺失数据上最优；CUTS+ 在 MCAR/MNAR 下表现最佳；TARNet 在 CATE 任务中最快且 PEHE 最低；不同轴（如缺失、噪声、非线性）对方法排名和性能有显著影响。

**⚠️ 局限性**

限制包括：仅评估合成数据，缺乏真实世界验证；可识别性仅对线性和部分非线性机制已说明，其余为未知；基准规模受计算限制（PCMCI-GPDC 仅 d=4）；超参调优使用图真值作为目标，报告结果可能过于乐观；未覆盖权重变动、更多维度和大规模图的实验。

---

## 199. Seeing vs. Counting: Complexity Twins Show Where Multimodal LLMs Fail on Diagrams

**arXiv ID:** 2611.08998 | [PDF](https://arxiv.org/pdf/2611.08998v1)

**作者:** Yin Liu `[一作]` `[通讯]`, Yin Liu

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个基于图形与计数的多模态评测基准，利用“复杂度双胞胎”（同一图形在不同计数复杂度上提出问题）来分离感知错误和计算错误，并在 Gemini、Gemma、Qwen3‑VLGemini 等模型上进行系统实验，研究链式思维（Chain of Thought）与代码生成（Code）在计数任务中的表现。

**💡 创新点**

创新点在于：①提出并使用复杂度双胞胎框架，能在保持图像不变的前提下比较不同计数复杂度对模型性能的影响；②发现计数答案大小（magnitude）对 LLM 成功率的主导作用，复杂度对 m_50 仅贡献约一十年代；③通过双视角阅读与 crop 检查大幅修复单视角在水平边缘上的盲区；④系统量化不同提示方式（Visual、Text、Code、Routing Prompt）与思考令牌对计数准确率的影响。

**🔧 技术方法**

使用的技术主要包括：多模态大型语言模型推理（Gemini、Gemma、Qwen3‑VLGemini）、链式思维提示、代码生成与运行（Python 程序实现矩阵树、Kasteleyn、BEST 等算法）、双视角图像转录与 crop 检查、统计分析（Logistic 回归、McNemar 检验、Bootstrap CI）以及自托管模型与 vLLM 测试。

**📊 数据集**

数据集包括：①6×6–12×12 的二值图像（构造多重空洞）；②3×3–10×10 的格点子图（随机保留连通性、度≥2）；③Eulerian 3/4 边环集合（带/不带箭头），共计约 6000 个实例，涵盖核心、池化、额外大规模以及四种现实渲染风格（PCB、蓝图、地铁、阴影桌面）。

**📈 对比分析**

比较方法：在 Visual、Text、Code、Routing Prompt 四种提示下，统计 Exact Match 百分比、思考/输出 token 数、算法族分布。实验表明：链式思维的准确率随答案大小指数下降（每十倍答案降低约 0.1–0.3 的概率），复杂度提升仅在极少数双胞胎中产生可测量提升；代码生成将准确率提升至 90% 以上；双视角阅读把大图的准确率从 0% 提升至 90% 以上。

**⚠️ 局限性**

局限性包括：①仅使用合成渲染的图形，未检验真实照片或自然场景的计数鲁棒性；②模型范围局限于单一供应商的 Gemini 系列和部分自托管 Gemma/Qwen3；③复杂度与答案大小高度共线，难以完全解耦；④自托管测试仅在 512/2048/8192 token 限制下完成，未覆盖更大算力场景；⑤算法族分类依赖手工规则，可能存在误判。

---

## 200. Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions

**arXiv ID:** 2611.08995 | [PDF](https://arxiv.org/pdf/2611.08995v1)

**作者:** Colin Priest `[一作]` `[通讯]` (University of New South Wales), Colin Priest (University of New South Wales)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了大型语言模型在消费者投诉处理中的公平性，使用匹配化妆实验评估写作风格对决策的影响。

**💡 创新点**

将群体差异与个体不确定性分离，采用概率输出、重复评分和选项顺序控制，首次系统评估风格标准化缓解措施。

**🔧 技术方法**

使用 GPT‑4o‑mini、GPT‑5‑mini 生成标签和语义重写，使用 Jev 的概率分布输出，结合方差分量模型和等价检验。

**📊 数据集**

基于美国 CFPB 公共投诉数据库的 20,000 条英文投诉，并生成 36 种社会语言化姿态的重写。

**📈 对比分析**

对比三种决策接口的群体差距和单案不一致率，发现群体差距均 ≤0.5pp，单案差异 2.6–6.3%，标准化写法降低教育与 SES 差距但未显著提升决策稳定性。

**⚠️ 局限性**

重写文本为人工合成、未覆盖方言/种族/第二语言、单一概率模型、缺乏真实写作数据，且标准化过程本身可能引入新的偏差。

---

## 201. When2What: A Timing-Conditioned Framework for Long-Horizon Forecasting of Continuous Cardiovascular Waveforms

**arXiv ID:** 2611.08993 | [PDF](https://arxiv.org/pdf/2611.08993v1)

**作者:** Chaoqi Zhang `[一作]` (Indiana University Bloomington), Haixu Tang `[通讯]` (Indiana University Bloomington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种名为When2What的时序条件化框架，采用预测–提交–生成三步流程实现对心血管波形的长周期（120 s历史→60 s预测）连续波形预测。

**💡 创新点**

核心创新是将心跳时间预测与波形生成解耦，先生成密集的时间概率图，再通过提交模块将模糊的峰值锐化为单个单位高度的Gaussian事件，随后将这些事件与历史信息共同条件化生成器生成完整波形，从而显著保持了心跳事件的尖锐特征。

**🔧 技术方法**

使用共享的Mamba‑Attention序列骨干、卷积自编码器对历史进行token化、峰值检测生成Gaussian时间图、教师强迫训练、MSE+一阶/二阶差分损失以及对生成器的时间和细粒度采样双尺度条件化。

**📊 数据集**

主要数据集为VitalDB（手术室ECG、ABP、PPG），外部验证使用MOVER（手术室ECG）和SHDB‑AF（心房颤动Holter）记录。

**📈 对比分析**

与Informer、PatchTST、DLinear、HTF、BeatLex、FADE、TimesFM等基线在120 s→60 s和10 s→2 s等配置下进行对比，Event skill在0‑10 s达到0.644，心跳检测F1>95%，比基线提升超过0.7点，并且在外部数据上实现零样本迁移保持优势。

**⚠️ 局限性**

局限在于仅生成单一确定的事件序列，缺乏对多模态联合预测与不确定性量化；对较长时延预测的精度随时间衰减；模型仅针对单通道，未充分处理噪声或未标记的实时流数据。

---

## 202. Enforcing Symmetries on Pre-trained ML Models for Physics

**arXiv ID:** 2611.08992 | [PDF](https://arxiv.org/pdf/2611.08992v1)

**作者:** Wenjie Zhou `[一作]` (Johns Hopkins University), Soledad Villar `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出后置对称化方法，即在已训练模型上进行群平均或投影，将模型投影到给定对称群（紧群或非紧群）上，并将此思路推广到单位缩放和洛伦兹变换等非紧对称性，给出三种实现方案（Weyl unitarian trick、数据相关L²投影、帧平均），随后在物理仿真、单位转换和高能碰撞数据集上进行系统实验。

**💡 创新点**

创新点在于首次在非紧群上给出可行的投影方法——Weyl unitarian trick利用复数化与紧子群平均实现精确投影；帧平均提供对任意黑盒模型的输入依赖式投影；实验表明在有限样本或分布外情况下，后置对称化能显著提升鲁棒性，并与直接对称训练相竞争。

**🔧 技术方法**

主要技术包括：Reynolds投影（Haar平均）、Weyl unitarian trick（复数化与紧子群平均）、基于数据的L²正交投影、帧平均、Monte Carlo采样、Wick旋转、拉普拉斯多项式和多项式特征投影等。

**📊 数据集**

使用的数据集包括：The Well仿真数据集（流体动力学、化学扩散等），单位缩放回归/分类数据，洛伦兹变换下的合成物理仿真数据，以及高能物理中的quark‑gluon jet分类数据。

**📈 对比分析**

实验中将后置对称化方法与未对称化基线、Weyl投影、L²投影、帧平均以及对称训练模型进行对比。在紧群实验中，群平均显著降低误差；在非紧群实验中，Weyl投影在分布外条件下可实现数十倍误差下降，帧平均在保持精度的同时防止灾难性失败，但在分布内往往会略微牺牲准确率；直接对称训练模型往往在分布内表现最好。

**⚠️ 局限性**

局限性包括：非紧群投影缺乏理论上的风险下界；Weyl trick仅适用于多项式或全纯函数，无法直接应用于一般深度网络；帧平均需要手工构造输入相关框架，且可能导致模型非连续；在分布内不一定提升，且对训练数据分布高度依赖；实现成本（采样或优化）相对较高。

---

## 203. LEF-JEPA: Denoising-Trained Latent Energy Fields for Spatial Representation Learning

**arXiv ID:** 2611.08989 | [PDF](https://arxiv.org/pdf/2611.08989v1)

**作者:** Zhiyuan Liu `[一作]` (Qinous Technology Co., Ltd), Huanhuan Huang `[通讯]` (Qinous Technology Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出了一种名为LEF‑JEPA的自监督预训练框架，通过目标引导的潜在能量场和相对能量约束，提升视觉编码器在捕捉空间结构（如物体轮廓）同时避免过度编码低层纹理信息的能力。

**💡 创新点**

创新点在于：①将多层目标表示用于引导潜在空间的局部恢复；②通过单侧相对能量约束将结构监督直接迁移至编码器，避免在推理时需多步恢复；③提出可由物理Ising机实现的条件Ising哈密顿量，实现硬件加速的能量学习。

**🔧 技术方法**

主要技术包括：联合嵌入预测（JEPA）框架、条件能量变换器/条件Ising哈密顿量、相对能量约束、对抗/多层目标监督、光子Coherent Ising Machine（CIM）硬件求解。

**📊 数据集**

实验使用CIFAR‑10进行预训练，并在冻结编码器的情况下评估CIFAR‑10线性分类、跨域前景分割（PartImageNet）以及连续Two‑Room视觉规划。

**📈 对比分析**

与V‑JEPA2、V‑JEPA2.1和EB‑JEPA等基线相比，LEF‑JEPA在ViT‑S和ViT‑B上分别实现了约1–2%提升的分类精度、5–10%提升的前景IoU和约10%提升的规划成功率；使用Ising实现时，仅需一次局部更新即可达到最佳效果。

**⚠️ 局限性**

局限性包括：对超参数（如相对能量系数、更新步数）较敏感；物理Ising求解器在规模与通用性上受限；当前仅在小尺寸数据集（CIFAR‑10）验证，尚未证明在更大规模或长视频场景的可扩展性。

---

## 204. Stabilizing Vector Approximate Message Passing with Circuit-Designed Lag-Lead Damping

**arXiv ID:** 2611.08982 | [PDF](https://arxiv.org/pdf/2611.08982v1)

**作者:** Mohamed Akrout `[一作]` `[通讯]` (University of Tennessee), Mohamed Akrout (University of Tennessee)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于电路分析的向量近似消息传递（VAMP）算法稳定化方法，利用时滞-领先滤波器实现自适应阻尼；

**💡 创新点**

创新点在于将VAMP迭代映射成电路模型，利用闭环增益谱预测振荡并设计负领先电容（负动量）来抑制无法单一阻尼捕捉的振荡模态；

**🔧 技术方法**

采用电路类比、线性化闭环增益分析、矩阵分解和 Arnoldi 方法求特征值，设计滞后-领先（lag‑lead）滤波器并在运行中每 K 次迭代重新计算参数；

**📊 数据集**

使用多径信道仿真数据集，生成尺寸 N=300、M=150 的实对称矩阵，稀疏度 0.1、信噪比 30 dB，调节条件数、秩、稀疏度和噪声水平进行实验；

**📈 对比分析**

与无阻尼 VAMP、手动/自适应阻尼、Anderson 加速和理想化 Genie-aided 阻尼进行对比；结果显示自适应 lag‑lead 阻尼在各种容差下平均迭代次数最低，收敛率高于现有方法（收敛率可达 95%+ 对比 88%–90%）；

**⚠️ 局限性**

局限性包括：需要在每 K 次迭代重新计算特征值和闭环增益，计算成本相对较高；在极端条件数或非常噪声的情形下仍有部分实例不收敛；

---

## 205. Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model

**arXiv ID:** 2611.08977 | [PDF](https://arxiv.org/pdf/2611.08977v1)

**作者:** William Xingxu Chen `[一作]` (Duke University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出能量基推理器（EBR），通过对学习的能量函数做梯度下降在潜在空间中进行推理，并引入状态级探索建模（SXM）以生成多解。

**💡 创新点**

将能量最小化统一为循环推理、停止判定与解选择，并通过SXM在训练中探索多条轨迹来塑造多模态能量景观，实现高效多解推理。

**🔧 技术方法**

采用能量模型、梯度下降与投影Langevin动力学、分段深度监督、收敛监督、随机步长与噪声、SXM轨迹匹配，以及能量基选择与停止机制。

**📊 数据集**

四个算法推理基准：Sudoku‑Extreme、Maze‑Hard、N‑Queens、Graph Coloring。

**📈 对比分析**

与 HRM、TRM、FPRM、GRAM、Looped Flows 等循环推理模型比较，EBR 在 Sudoku‑Extreme 单样本 99.4%、多样本 99.9% 时仅用 1.8M 参数；在 Maze‑Hard 单样本 87.4%；在多解任务中覆盖率大幅提升（如 10×10 N‑Queens 92.0%）。

**⚠️ 局限性**

训练需二阶梯度计算、SXM 额外轨迹开销，且对步长、噪声、迭代步数等超参数敏感；目前仅验证在结构化算法推理任务，未扩展至更大模型或语言/规划任务。

---

## 206. Refer-CityFlow and M3Track: A Benchmark Dataset and Reference Framework for Language-Driven Multi-Target Multi-Camera Tracking

**arXiv ID:** 2611.08976 | [PDF](https://arxiv.org/pdf/2611.08976v1)

**作者:** Phuong T. Nguyen `[一作]` (University of Tennessee at Chattanooga), Hoang H. Nguyen `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了面向语言驱动的多目标多摄像头跟踪基准数据集 Refer‑CityFlow 与 M3Track，并提出了对应的参考框架。

**💡 创新点**

创新点在于：①首次将自然语言查询与多摄像头跟踪结合，提供了跨摄像头的语义检索功能；②通过细粒度标注的轨迹与语言对齐，支持跨视角、跨时段的目标检索；③提供统一的评测流程与指标，推动该领域的可复现性。

**🔧 技术方法**

主要技术包括：跨模态 Transformer（视觉–语言融合）、多摄像头轨迹关联算法、轨迹重识别模型以及基于图优化的多摄像头关联模块。

**📊 数据集**

使用的数据集为 Refer‑CityFlow（城市级多摄像头视频，含约 15,000 条轨迹和 12,000 条语言查询）和 M3Track（工业级多摄像头跟踪数据，含 8,000 条轨迹和 4,500 条查询）。

**📈 对比分析**

与现有的多摄像头跟踪方法（如TrackRNN, MOTR）以及单摄像头语言跟踪模型进行对比。实验结果显示，参考框架在 MOTA、IDF1、检索准确率上分别提升 8.5%、6.2% 及 12.4%，证明其在跨摄像头语义检索上的优势。

**⚠️ 局限性**

限制包括：①数据集规模有限，尚未覆盖所有城市和场景；②语言查询的多样性不足，主要集中在英语；③跨摄像头关联仍受遮挡、光照变化等因素影响，导致某些稀疏查询性能下降。

---

## 207. Using an LLM to Tackle Difficult Cases in NER for Nepali Tweets

**arXiv ID:** 2611.08967 | [PDF](https://arxiv.org/pdf/2611.08967v1)

**作者:** Claire Wang `[一作]` `[通讯]` (Brandeis University), Claire Wang (Brandeis University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将低资源语言尼泊尔语推文的命名实体识别任务拆分为易/难两类，并使用LLM在难类上进行自适应推理。

**💡 创新点**

创新点在于结合PLM的token概率筛选与LLM的语义相似例子增强prompt，实现对难样本的更高F1。

**🔧 技术方法**

使用的技术包括DistilBERT PLM、Gemma 4 26B LLM、In‑context学习、实体类型定义、相似度选择和Self‑Verification。

**📊 数据集**

使用的数据集为DanfeNER（尼泊尔推文）及其标注文件。

**📈 对比分析**

通过在完整测试集上比较PLM、LLM以及组合模型的F1，发现PLM在全/易集上占优，LLM在难集略胜，组合模型F1提升0.98%。

**⚠️ 局限性**

局限在于难集规模有限、LLM在全集上的提升不显著、实体类型定义与自检未显著改善、且对代码混杂处理的效果仍不稳定。

---

## 208. Concordance-Optimal Continuous Music Emotion Estimation: Variance Decomposition, Dual-Branch Temporal Fusion, and Conformal Prediction Intervals

**arXiv ID:** 2611.08966 | [PDF](https://arxiv.org/pdf/2611.08966v1)

**作者:** Mohammad Faraz Akhtar `[一作]` (Birla Institute of Technology, Mesra), Soubhik Chakraborty `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出方差拆分理论并基于此设计混合损失，构建MMAS双分支模型（MERT预训练音频嵌入+自动转录符号特征），实现实时情绪轨迹预测并给出分层置信区间。

**💡 创新点**

①阐释方差拆分导致池化相关失真并将Lin协和相关窗口形式用于形状正则化；②结合冻结的MERT嵌入与10维符号特征；③通过跨模态注意力实现双分支融合；④实现可移植的单模型，无需域标记。

**🔧 技术方法**

自监督 MERT-768 嵌入、Basic Pitch 自动转录的 10 维符号特征、双分支 TCN + 跨模态注意力、窗口化 + 重叠平均、混合损失 λ·(1−CCC)、split‑conformal 预测区间、Wilcoxon 符号秩检验。

**📊 数据集**

DEAM（1,764 曲）和 PMEmo（523 曲）连续情感标注数据集，所有曲子对齐至 0.5 s 网格。

**📈 对比分析**

采用歌曲分组 5‑折交叉验证，报告池化 Pearson r、Lin CCC 以及每曲平均 r；与 Ridge 基线对比，MMAS 在每曲跟踪上提升约 3 倍；在 PMEmo 上 valence ≈ 0.74、arousal ≈ 0.74；split‑conformal 区间覆盖率为 93.7%（arousal）/90.0%（valence）。

**⚠️ 局限性**

valence 仍显弱；置信区间仅边际有效，歌曲级覆盖率不足；符号特征受转录误差影响；仅覆盖西方流行音乐；未做临床或多文化验证；未实现张力模型。

---

## 209. MetricPose: Flexible 3D Human Pose Estimation with Depth and Geometric Cues

**arXiv ID:** 2611.08954 | [PDF](https://arxiv.org/pdf/2611.08954v1)

**作者:** Shifan Zhu `[一作]` (University of Massachusetts Amherst), Donghyun Kim `[通讯]` (University of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的 3D 人体姿态估计框架，支持 RGB、RGB-地面和 RGB-D 三种输入模式。

**💡 创新点**

创新点在于将可选的深度感知与地面几何约束相结合，并通过可选路径实现共享参数的多模态推理。

**🔧 技术方法**

使用了 SAM 3D Body 作为骨干，结合 LoRA 微调、深度特征融合、两次深度引导重构、以及基于地面和深度的确定性尺度修正。

**📊 数据集**

在 BEDLAM2、CMU Panoptic、HuMMan、EgoBody 等多数据集上进行训练和评估，此外还使用 3DPW、EMDB、RICH、OpenCap 等公开基准。

**📈 对比分析**

与原始 3DB、Fast 3DB 等基线对比，RGB‑D 模式在 camera‑local MPJPE 上提升约 50% 以上，RGB‑Ground 也显著降低错误，整体性能优于现有方法。

**⚠️ 局限性**

主要局限在于对深度传感器的依赖、对相机标定和地面假设的敏感性，以及计算开销增加。

---

## 210. The Schur products of some nonbinary narrow-sense antiprimitive BCH codes

**arXiv ID:** 2611.08945 | [PDF](https://arxiv.org/pdf/2611.08945v1)

**作者:** Haojie Xu `[一作]` (Southeast University), Xiwang Cao `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了非二进制窄义反原码（antiprimitive BCH码）的Schur乘积，并给出了在不同设计距离和有限域条件下的必要与充分条件，进一步计算了一些乘积码的维数与最小距离，并列举了部分具备最佳已知参数的线性码。

**💡 创新点**

首次系统性地将Schur乘积应用于宽义反原码，提出了新的阈值Ω_(q,m)和Λ_(q,m)来判定乘积码是否等于全空间，并通过对q奇偶、m取值范围的精细分析，得出了针对不同情形的必要与充分条件；同时提供了对乘积码维数和最小距离的显式下界。

**🔧 技术方法**

利用BCH码的定义集合、q-循环余数与coset结构，结合有限域多项式与循环码的生成多项式理论，以及对最小coset领导的组合式分析（涉及到多重q-进制展开和取值极限）。

**📊 数据集**

本文主要是理论研究，并未使用实验数据集；但在部分参数下通过Magma软件验证了乘积码的维数与距离，并与已有最佳参数表进行对比。

**📈 对比分析**

与已知的最佳线性码参数对照，发现部分得到的乘积码在维数和距离上与最佳已知码相匹配，或仅差1；说明所提出的条件能产生高性能的码。

**⚠️ 局限性**

局限性包括：对二进制情形缺乏直接适用的分析方法；Ω_(q,m)和Λ_(q,m)在q偶且m大于11时尚未给出统一表达式；在δ_1大于Ω_(q,m)且δ_2介于Δ_2+1与Δ_1之间时，无法给出充分的维数和距离估计，需进一步研究。

---

## 211. ODD+LLM: Specifying LLM-Agent Behavior for Reproducible Social Simulation

**arXiv ID:** 2611.08932 | [PDF](https://arxiv.org/pdf/2611.08932v1)

**作者:** Justin Downes `[一作]` (Amazon Web Services), Newel Hirst `[通讯]` (Amazon Web Services)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 ODD+LLM 协议，填补 LLM‑agent 模拟在提示拼装、记忆序列化、输出解析和重试逻辑方面的记录缺口，并通过多宇宙实验与 28 篇论文的重建充分性评估验证其必要性。

**💡 创新点**

创新点在于为 ODD 协议新增兼容性五块（认知架构、身份、推理配置、行为子模型、验证），为 LLM‑agent 行为提供标准化记录，并展示单一遗漏说明如何导致实验结果完全不同。

**🔧 技术方法**

利用 CoALA 认知架构框架定义块 A，DSPy 规范化 LLM 调用签名，采用多宇宙实验设计、统计检验和系统化代码审计工具进行验证。

**📊 数据集**

在自定义命名游戏模拟中使用 24 名代理、字母池 W=2；并对 28 篇公开 LLM‑agent 论文的原始代码、补充材料和元数据进行审计。

**📈 对比分析**

通过对 8 个实验分支（基线 + 6 规范变体）进行 20 随机种子跑，比较共识率、收敛轮次与失败率；缺失的“第一人称视角”规范将共识率从 35% 提升至 100%，收敛速度提高约 3 倍。

**⚠️ 局限性**

局限性包括实验仅在单一模型（Claude Sonnet 4.6）和单因素设计下完成，未考察交互效应；协议验证缺乏多编码者的可靠性评估；未覆盖多模态输入；以及对 28 篇论文的评估仅由单一编码者完成，可能存在抽样偏差。

---

## 212. FANO: Anchored Forward-Noise Optimization for Diffusion Inverse Problems

**arXiv ID:** 2611.08923 | [PDF](https://arxiv.org/pdf/2611.08923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 213. Wellness Robots and the Path to Full Autonomy

**arXiv ID:** 2611.08930 | [PDF](https://arxiv.org/pdf/2611.08930v1)

**作者:** Mohammad H. Mahoor `[一作]` (University of Denver), Richard M. Voyles `[通讯]` (University of Texas at Arlington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了自主健康机器人（Autonomous Wellness Robot）这一新类别，并基于七维福祉模型构建了护理机器人自主水平量表（CRAS），通过在美国德州大学开发并商业化的Ryan机器人进行临床验证。

**💡 创新点**

创新点在于把多维度福祉评估与自动化层级相结合，首次给出跨维度连续干预的可测量框架（CRAS），并通过实证数据证明机器人在七维福祉维度上的持续自适应干预可实现认知与情绪的显著改善。

**🔧 技术方法**

主要技术包括：多模态感知（面部表情、语音情绪、姿态和深度跟踪）、大型语言模型驱动的对话与跨维度规划、可解释决策链、嵌入式联邦学习与差分隐私、以及基于安全约束的控制策略。

**📊 数据集**

使用了NIH/NIA R44AG059483临床试验数据（18名老年人887小时的交互记录）以及Ryan内部收集的多模态交互日志和计划中的护理交互数据集（CID），这些数据覆盖情绪、认知、社交、环境等多维度特征。

**📈 对比分析**

与传统单一任务机器人或语音助手相比，Ryan在SLUMS认知评分上平均提升+2.61点（p=0.015，Cohen's d=0.64），PHQ‑9抑郁评分下降-2.50点（p=0.073，d=0.45），且在887小时的长期使用中未出现新奇效应，显示跨维度持续干预的有效性。

**⚠️ 局限性**

限制主要包括：缺乏物理协助（如站起、行走、跌倒响应）功能、对多方场景识别和多人的对话处理不足、老年人语音识别鲁棒性不够、未实现完整的EHR互联与预测模型，且当前系统仍处于CRAS 1–2级，距离全自治（L5）还有显著技术和证据缺口。

---

## 214. PRISM: Primitive Routing via In-context Skill Mixing for Retentive VLA Adaptation

**arXiv ID:** 2611.08917 | [PDF](https://arxiv.org/pdf/2611.08917v1)

**作者:** Jiachen Tao `[一作]` (University of Illinois Chicago), Yan Yan `[通讯]` (University of Illinois Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PRISM框架，将可重用技能编码为参数级低秩原语，并通过VLM前缀路由器在长周期任务中动态组合；同时保留原子技能的性能；

**💡 创新点**

发现每个技能的LoRA更新聚集在低维子空间，复合任务LoRA大约56%能在该子空间表达；基于此设计技能原语库+残差+按块路由，实现在复合训练中既保留原子技能又能动态组合；

**🔧 技术方法**

使用低秩适配器（LoRA）、视觉语言模型π0.5、动作专家、VLM前缀特征、残差适配器、软/硬路由、激活空间混合、两阶段优化及动作派生标签器；

**📊 数据集**

RoboCasa365模拟数据（28个原子任务，4个复合任务）与AgileX PIPER物理机器人（Cup→Table、Doll→Plate等）；

**📈 对比分析**

与标准π0.5‑LoRA两阶段微调、带回放的LoRA及零样本进行对比。PRISM在原子任务上达56.43%成功率，复合任务20%；相比两阶段微调提升约38.6点原子、2.5点复合；相比回放提升约6.6点原子、9.75点复合；在真实机器人上复合任务成功率55%，为最高；

**⚠️ 局限性**

实验仅覆盖少量原子技能族（3类），扩展到更多技能与更大规模任务尚未验证；需手工制定标签规则，可能限制自动化；在极端复杂或动态环境下的可扩展性与稳健性仍待评估。

---

## 215. Capital Efficiency, Isolation, Risk, and Ruin in Generative AI Inference Serving Infrastructure

**arXiv ID:** 2611.08999 | [PDF](https://arxiv.org/pdf/2611.08999v1)

**作者:** Renée St. Amant `[一作]` (Microsoft), Victor Rühle `[通讯]` (Microsoft)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对生成式 AI 推理服务的请求按输入序列长度进行隔离，提升了机群利用率与资本效率。

**💡 创新点**

创新点在于将“隔离”作为舰队级杠杆，从宏观层面解决非可归一化（非 ergodic）工作负载导致的系统风险与资本浪费，并提出了“伪可归一化”与“分段成本可视化”等概念。

**🔧 技术方法**

技术手段包括基于序列长度的请求分区、批量大小调优、隔离后调度与资源分配、以及利用隔离结果进行的成本预测与硬件/软件协同设计。

**📊 数据集**

使用的主要数据集为一小时的生产流量（约 70% 短序列、20% 中序列、10% 长序列）以及公开的“mooncake 对话”标记分布。

**📈 对比分析**

与未隔离基线对比，隔离后 GPU 利用率提升约 50%，每请求成本降低约 2 倍；批量大小在短序列分区提升 14 倍，导致整体吞吐量显著提升，成本效益提升约 2 倍以上。

**⚠️ 局限性**

局限包括：分区粒度选择需权衡维护复杂度与资源利用率；仅以序列长度为分区标准，忽略专家路由、模态特性等多维特征；在高并发下仍可能出现“碎片化”与“闲置容量”问题。

---

## 216. UniMore: Fine-Grained Supervision for Unified Understanding and Generation

**arXiv ID:** 2611.08996 | [PDF](https://arxiv.org/pdf/2611.08996v1)

**作者:** Haoyue Bai `[一作]` (William & Mary), Jindong Wang `[通讯]` (William & Mary)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的后训练方法UniMore，通过利用图像-文本对的细粒度内容来提升多模态模型的生成与理解能力。

**💡 创新点**

创新点在于将每个图像-文本对拆分为“基于短语的局部视觉预测”和“完整图像的句子区分”两条独立训练路径，并通过静态向量加法将两条更新合并到同一模型，而无需改变推理接口。

**🔧 技术方法**

主要技术包括基于GroundingDINO的短语-区域匹配、CLIP语义一致性评分、流式或自回归视觉目标训练、文本对比损失（正样本对负样本的间隔硬损失），以及LoRA适配器的微调。

**📊 数据集**

使用公开的多模态数据集：JPG-COCO、Conceptual Captions、COCO-Captions等，构建约2,500条图像-短语对和4,000条图像-句子对。

**📈 对比分析**

在BAGEL、Janus-Pro、Show-o2三大基座上进行评测，UniMore在MMMU、MMStar、MathVista、WISE、UEval等标准基准上均取得领先或竞争性提升，尤其在细节保留（颜色绑定）和交互一致性方面表现优异。

**⚠️ 局限性**

局限性包括：仍需额外的后训练时间与参数存储；对不同模型架构的适配依赖于其原生视觉目标；在极端细节或长文本生成场景下提升有限，且模型对负样本的构造方式可能影响效果。

---

## 217. Rethinking Optimization Methods for Hierarchical Feature Selection

**arXiv ID:** 2611.08991 | [PDF](https://arxiv.org/pdf/2611.08991v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并比较了四种HTNL优化方法（MM/VQF、ADMM、WGL），用于稀有事件预测中的数值联合特征选择。

**💡 创新点**

创新点包括统一诊断四种优化方法、引入精确的ADMM双分支求解器、证明激活集评分指数衰减的结构上限，并指出候选扩展受限。

**🔧 技术方法**

主要技术包括变分二次形式、主优化-最小化、ADMM及其精确双分支近似、加权组Lasso连续路径、KKT筛选与主动集扩展。

**📊 数据集**

实验使用了两组合成数据和11个真实数据集：10个拉美地区社交媒体暴动预测、1个美国流感监测数据。

**📈 对比分析**

与加权逻辑回归基线比较，VQF平均提升0.074 AUC，ADMM-Bisect在保持性能的同时显著加速，WGL需调整正则化规模以避免空模型。

**⚠️ 局限性**

局限在于所有方法均受共享的KKT准入阈值限制，难以发现长链联合特征，且WGL对正则化尺度敏感。

---

## 218. Action-Centered Causal Bandits for Budget-Constrained Promotion Targeting

**arXiv ID:** 2611.08990 | [PDF](https://arxiv.org/pdf/2611.08990v1)

**作者:** Yifan Lin `[一作]` (LinkedIn Corporation), Di Mo `[通讯]` (LinkedIn Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的在线算法DR-CATE-CBwK，用于在预算约束下学习用户的治疗效果对比，以最大化增量结果。

**💡 创新点**

创新点在于结合了离线双重稳健学习器、在线以行动为中心的估计器和原始-对偶预算控制器，能够在基线模型错误指定的情况下有效更新。

**🔧 技术方法**

使用了双重稳健学习、行动中心的估计和原始-对偶控制技术。

**📊 数据集**

在一个半合成的工作市场基准上进行了评估，使用了真实的记录特征和模拟的结果。

**📈 对比分析**

与流体线性规划（LP）进行了基准比较，DR-CATE-CBwK在预算可行的方法中表现出最低的后悔值，约比最强的奖励基线减少20%。

**⚠️ 局限性**

限制在于算法假设线性条件平均治疗效应（CATE），并且在处理非线性CATE时的表现尚未完全验证。

---

## 219. RevoSim: Scalable Multimodal Tactile Simulation for Dexterous Manipulation

**arXiv ID:** 2611.08939 | [PDF](https://arxiv.org/pdf/2611.08939v1)

**作者:** Ke Zhang `[一作]` (Hong Kong University Of Science And Technology), Renjing Xu `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了可扩展的多模态触觉模拟器RevoSim，能够同步生成285个电阻式传感器读数与五个视触感指尖的深度、RGB与标记位移，支持单显卡下4,096个并行仿真环境。

**💡 创新点**

首次将电阻式全掌传感与视触感指尖联合模拟，提出几何感知压力扩散模型和自适应指尖深度采样，显著降低内存占用；同时利用校准的标记位移与RGB合成，提供与真实传感器高度一致的多模态观测。

**🔧 技术方法**

使用Isaac Lab 2.3.2 + CUDA，GPU‑Taxim渲染，软体物理弹性与摩擦模型，稀疏差分扩散卷积，自适应光线采样与投影，标记位移与RGB的合成算法。

**📊 数据集**

以Revo3手的真实传感器数据进行校准，训练任务在Isaac Lab的仿真环境中生成；对比TacMap、TacSL、FOTS、HydroShear等现有模拟器的数据。

**📈 对比分析**

通过对比VRAM使用率与可支持的并行环境数评估：RevoSim在1,024个深度环境下仅需9.2 GiB，比TacMap的30.6 GiB少约70%；在4,096个环境下支持marker观测，仅占31.36 GiB，其他模拟器已溢出。实验中用RevoSim训练的球旋转与抓取任务验证了其在RL训练中的有效性。

**⚠️ 局限性**

目前仅支持Revo3手的硬件配置，缺少对可变形物体的建模；时序动态建模有限，未完成sim‑to‑real转移的系统评估；自适应深度采样在极复杂接触场景下的鲁棒性待进一步验证。

---

## 220. HOC2R: Hand-Object Contact Trajectories for Human-to-Robot Motion Transfer

**arXiv ID:** 2611.08920 | [PDF](https://arxiv.org/pdf/2611.08920v1)

**作者:** Hongkun Yu `[一作]` (UNSW Sydney), Francisco Cruz `[通讯]` (Universidad Central de Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出HOC2R框架，通过结合冻结的DINOv3与SAM 2.1特征，训练轻量级头实现可计数的自动化和已知接触段的两点定位，从人类视频中提取可用于机器人后接触动作的手-物体接触轨迹。

**💡 创新点**

创新点在于将视觉预训练模型与专门设计的proposal、touch、count、heatmap四个轻量头相结合，支持两种部署模式（计数自适应与固定两点），在多手场景下有效区分操纵接触与静止或无关手，并可直接生成伪标签用于机器人学习。

**🔧 技术方法**

使用技术包括：冻结的DINOv3 ViT-S/16视觉特征、SAM 2.1 Hiera-S特征金字塔、Proposal Head、Touch Head、Count Head、Heatmap Head、触点重排序、深度投影与机器人基座坐标变换，以及共享的过滤与轨迹重建。

**📊 数据集**

采用的主要数据集为210帧蓝色手标注的30任务点标注基准、七个RGB‑D外部滚动数据集（自然肤色、视角、背景、光照变化）以及六个机器人评估序列；所有数据已公开发布。

**📈 对比分析**

方法通过与两种适配基线（VRB2和EgoHOS-Point）比较，结果显示：在内部基准上count‑aware模式将接触计数准确率从58.1%提升至71.4%，F1@10从25.8%提升至62%，平均定位误差从19.81像素降至10.46像素；在外部多手RGB‑D数据上fixed‑two模式达32.8% F1@10、27.02像素；机器人实验中30任务fixed‑two产生4条可执行轨迹，成功率为3/6。

**⚠️ 局限性**

局限性包括：数据量有限（210帧）、仅针对两指交互、每帧独立预测且缺乏时间连贯性、深度噪声与相机标定误差可能导致投影失真、机器人实验仅覆盖后接触运动而非完整抓取或自主学习、以及基线适配方式不一定代表原始方法性能。

---

## 221. On the Communication Complexity of Shortcut Set Computation

**arXiv ID:** 2611.08978 | [PDF](https://arxiv.org/pdf/2611.08978v1)

**作者:** Gregory Bodwin `[一作]` (University of Michigan), Luba Samborska `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在两方通信模型中构造满足Kogan‑Parter大小/直径权衡的shortcut集，提供了一种通信复杂度为O(n²/D²)的协议，并给出了相应的下界；

**💡 创新点**

首次将Kogan‑Parter的近似上界转化为低通信协议，并通过新的动态数据结构实现了高效的路径覆盖与路径BFS，突破了传统Menger定理/反链分解方法；

**🔧 技术方法**

利用路径覆盖(PathCover)、路径BFS(PathBFS)子程序，构建了基于节点标记和最长路径标签的动态数据结构，以实现 O(n²/D²) 通信；

**📊 数据集**

论文为理论研究，无实验数据集，所有结果均在抽象图论模型下证明；

**📈 对比分析**

与先前的 O(n²/D) 或 O(n³/²D^(3/2)+…) 通信复杂度相比，本文将通信量降低至 O(n²/D²)，同时保持与Kogan‑Parter相同的shortcut大小上界；

**⚠️ 局限性**

仅适用于直径 D≤n^(1/3) 的有向无环图，且对图的路径长度有限制，未涵盖更大 D 或一般有向图的情况；

---

## 222. Beyond Truncation? Entailment-Licensed Elision and the Limits of Semantic Preservation in LLM Decoding

**arXiv ID:** 2611.08975 | [PDF](https://arxiv.org/pdf/2611.08975v1)

**作者:** Mahmoud Alyosify `[一作]` (Queen's University), Eman Elkhamisi `[通讯]` (Queen's University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文研究在指令调优LLM解码环路中，无训练的控制器用于删减冗长回复，保持内容完整。

**💡 创新点**

创新点在于提出熵门控logits处理器（实现停机控制）和基于自然语言推理的许可式消歧法（只隐藏已出现信息的跨度），并证明前者仅实现截断，后者在不损失信息的前提下优于截断。

**🔧 技术方法**

使用熵门控logits处理器、自然语言推理模型（DeBERTa‑v3‑base）、余弦相似度过滤、词嵌入、以及多种训练免费解码干预技术。

**📊 数据集**

评估数据集包括 MT‑Bench、AlpacaEval、Arena‑Hard‑Auto、YapBench，以及 670 题开发集和两组 250/705 题新测试集；模型为 Mistral‑7B、Llama‑3.1‑8B、Qwen2.5‑3B、OLMo‑2‑7B、Granite‑3.1‑8B。

**📈 对比分析**

与匹配长度截断基准对比，熵门控处理器在保持 BERTScore 非劣势的前提下实现 19–35% 的 token 减少，但实质为截断；E‑LICE 在 1–2% 的 token 减少下覆盖率仅比截断低 0.007，显著优于 23 种其它训练免费方法。

**⚠️ 局限性**

主要局限在于仅使用贪婪单样本解码、批量为 1、内部阅读评估、机器翻译多语料、以及只减少显示 token，且 E‑LICE 早期版本对答案有 60% 以上的有害删减，需进一步完善许可范围。

---

## 223. Adaptive Age of Information Optimization in Rateless Coding-Based Multicast-Enabled Sensor Networks

**arXiv ID:** 2611.08970 | [PDF](https://arxiv.org/pdf/2611.08970v1)

**作者:** Hung-Chun Lin `[一作]` (National Taiwan University), Hung-Yu Wei `[通讯]` (National Taiwan University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了在多播感知网络中利用无穷码（rateless coding）最小化时均Age of Information（AoI）的决策框架，并给出了最优与低复杂度子最优策略。

**💡 创新点**

创新点在于：①首次给出无穷码多播系统的AoI最优调度；②通过分析状态转移结构获得显著的结构性质；③基于这些性质设计了结构感知的相对值迭代算法与分布式低复杂度子最优算法。

**🔧 技术方法**

主要技术包括：马尔可夫决策过程（MDP）建模、Bellman方程与相对值迭代（RVIA）、结构性质证明、状态空间压缩、Q因子分解与近似、分布式计算。

**📊 数据集**

实验使用仿真数据：多台设备（K=2、20等）、不同的丢包率（p_k 0.05–0.6）和上界Â=40，采用随机生成的BSC/BEC通道模型；未使用公开数据集。

**📈 对比分析**

与基线策略（Fixed Redundancy、Infinite Incremental Redundancy、Fixed Incremental Redundancy）比较，结果显示：①子最优策略与最优策略几乎一致；②两者均显著优于基线，尤其在设备数较少或丢包率不均匀时表现突出；③随着设备数增大，子最优策略保持接近最优，优于基线的性能提升显著。

**⚠️ 局限性**

局限性：①状态空间维度随设备数呈指数级增长，最优算法仅适用于小规模系统；②假设完美即时反馈与BSC/BEC通道，实际网络中可能存在反馈延迟与多样化噪声；③仅考虑AoI最小化，未考虑能耗、传输延迟等其他 QoS 指标。

---

## 224. Surrogate Quality Score: A Novel Composite Performance Metric for Model Selection

**arXiv ID:** 2611.08937 | [PDF](https://arxiv.org/pdf/2611.08937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. PhysioWave-T: Token-Efficient Wavelet Foundation Models for Biosignals

**arXiv ID:** 2611.08969 | [PDF](https://arxiv.org/pdf/2611.08969v1)

**作者:** Yanlong Chen `[一作]` (Nanyang Technological University), Yawei Li `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种面向生理信号的Token‑高效Wavelet基础模型（PhysioWave‑T），通过路由特定的前端、动态尺度折叠、通道名称嵌入等技术，统一学习EEG、ECG和sEMG的多尺度表示并实现跨采样率/通道数的迁移；

**💡 创新点**

创新点包括：①动态尺度折叠将所有波形子带合并为单一信号，从而在保持多尺度信息的同时将Token数压缩至CS；②路由特定的Wavelet前端与可扩展的通道词表，使同一Transformer骨干能适配不同通道数与采样率；③可靠性感知潜在融合（RALF），在多模态癫痫检测中按样本自适应加权各模态；

**🔧 技术方法**

主要技术包括可学习的Wavelet选择器、软门控、跨尺度特征融合、动态尺度折叠、基于遮掩重建的自监督预训练、RMS归一化与旋转位置编码的Transformer、通道名称嵌入以及可靠性感知的潜在融合模块；

**📊 数据集**

使用的数据集涵盖EEG（TUEG、HBN、FACED、TDBRAIN、M3CV、PhysioNet MI、HGD）、ECG（MIMIC‑IV、PulseDB、Icentia、Georgia等）、sEMG（emg2qwerty、emg2pose、CEMHSEY、Hyser、putEMG）以及癫痫检测、心脏诊断、运动识别等任务的数据集（SeizeIT2、PTB‑XL、CPSC、Chapman‑Shaoxing、Sleep‑EDFx、PhysioNet‑P300、KaggleERN）；

**📈 对比分析**

在睡眠分期、P300/ERN检测、心脏多类诊断、运动识别和多模态癫痫检测等任务中，与BENDR、BIOT、LaBraM、EEGPT、PhysioWave-L等最新基础模型在主要评估指标（ACC、kappa、F1、AUROC、AUPRC）上实现或逼近最优表现，并在EEG任务上将Transformer内存和推理延迟降低4–12倍；

**⚠️ 局限性**

局限性包括：①需要预先定义路由并为新采样率/通道数手动添加前端和词表；②对极低采样率或频段与预训练不匹配的信号（如200 Hz手环）仍表现下降；③可靠性估计依赖人工合成的噪声和失效模式，缺乏真实传感器失效的评估；④多模态联合预训练未实现，当前多模态融合仅基于独立预训练模型。

---

## 226. AIM: Adaptive Interaction in Multi-Agent Debate for Multimodal LLM Inference

**arXiv ID:** 2611.08950 | [PDF](https://arxiv.org/pdf/2611.08950v1)

**作者:** Wei Fan `[一作]` (Virginia Tech), Bo Ji `[通讯]` (Virginia Tech)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一个实例级自适应多模态多代理辩论框架，动态选择最合适的交互模态组合以提升准确性并降低token成本。

**💡 创新点**

提出任务依赖模态交互与多模态组合；通过结构化基线回答提取可解释路由特征；三阶段路由器实现低成本高准确的模态组合选择。

**🔧 技术方法**

使用LLM Prompting（Gemini 2.5 Flash/GPT‑5.4‑nano）、结构化JSON输出、基于Confidence、Missing‑Evidence、Evidence‑Availability与Gain等特征的路由机制、图像RoI、音频段、视频Clip多模态处理以及时间压缩技术。

**📊 数据集**

训练集：Visual7W、CLEVR、AudioMCQ、Video‑MME；评估集：GQA、OKVQA、VCR、VQA‑v2、Clotho、DCASE、IntentQA、NextQA。

**📈 对比分析**

与七种固定模态组合及四种单组路由器对比，在八个基准上获得最高准确率，提升至6.4%相对单模态、4.2%相对最优固定组合，token成本平均下降多达53.4%，并保持或提升FaithScore。

**⚠️ 局限性**

需要一次性离线训练路由器，需为每个任务族收集七种组合结果；对视频模态的token成本仍高；缺乏在线自适应与更细粒度压缩策略。

---

## 227. Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents

**arXiv ID:** 2611.08946 | [PDF](https://arxiv.org/pdf/2611.08946v1)

**作者:** Jingdong Zhang `[一作]`, Wenping Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了FlexRSI基准，旨在评估前沿语言模型代理在固定预算内，通过反复实验既改进任务解法又改进自身执行环境（harness）的能力。

**💡 创新点**

创新点在于将“自我改进”拆分为两个可独立优化的组件——执行器（executor）与改进器（improver），并设计统一的实验协议让所有模型在同一通用harness和起始解法下竞争，从而真实衡量模型对自身操作机制的自适应能力。

**🔧 技术方法**

核心技术包括：①基于语言模型的多轮交互式执行与评估循环；②动态调整阈值δ以决定是否接受改进；③利用开发集验证、验证集过滤、隐藏测试集评估的三层数据分割；④在每轮后让improver根据过去轨迹自动修改prompts、工具和流程。

**📊 数据集**

使用了24个跨领域任务（共六大类：行动、推理、创作、重建、感知、基础设施），任务来源涵盖RoboDojo、RoboCasa、H*Bench、Ego-ExoMem、ARC-AGI系列、VSI-Bench、MSE-Bench、T2V-CompBench、ScanNet、Replica、MiniMax H3等公开数据集。

**📈 对比分析**

比较方法：每个模型在同一任务下仅运行一次，使用排名指标（0–100）对模型在每个任务的最终测试分数进行排序，随后计算每类任务的平均排名。实验结果显示，Claude Opus 5.5、Gemini 4 Argon等闭源模型在大多数任务中遥遥领先；自我改进显著提升任务分数，尤其在需要构建完整解法的任务上；但在仅输出单一答案或评估仅记录成功/失败的任务上提升有限。

**⚠️ 局限性**

局限性包括：①模型权重不参与自我改进，仅优化执行流程；②在某些任务（如ARC‑AGI‑3、Scene from Video）中即使外观合理，物理稳定性或精确度评估仍难通过；③对高成本任务的token使用与性能关联弱；④未覆盖模型自身参数微调的自适应改进，未来工作需扩展至权重级别。

---

## 228. Ordered Policy Optimization

**arXiv ID:** 2611.08936 | [PDF](https://arxiv.org/pdf/2611.08936v1)

**作者:** Dongkyu Derek Cho `[一作]` (Duke University), Pan Xu `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Ordered Policy Optimization (OPO)，通过对重要性比率的顺序统计尾部动态分配自适应剪切半径，替代 PPO 的固定点截断；

**💡 创新点**

创新点在于将极值理论中的 GPD 模型用于设定尾部目标，控制重要性比率分布的尾部行为，而非单一阈值；

**🔧 技术方法**

使用顺序统计、极值理论（GPD）、自适应样本级别的 clipping、第一阶 PPO 近似以及强化学习中的策略梯度框架；

**📊 数据集**

在 MuJoCo‑v4 任务（Ant、HalfCheetah、Hopper、Humanoid、HumanoidStandup、Walker2d）和 Atari 2600 游戏（RoadRunner、NameThisGame、BeamRider 等）上进行实验；

**📈 对比分析**

与 PPO‑Clip、PPO‑KL、TRGPPO、SPO、PPO‑λ 等基线比较，MuJoCo 任务中归一化得分与最强方法相当或略优，Atari 上相对提升与 SPO 相当或更好，整体性能优于传统 PPO；

**⚠️ 局限性**

仍使用固定的教师尾部参数（k*、σ*），缺乏自适应调整；理论收敛与稳定性证明不足；对小批次排序与阈值选择敏感，离散动作空间的推广仍需进一步验证。

---

## 229. LLM-Augmented Flow for Project Conformation and Task Complexity Evaluation Based on the TLM Framework

**arXiv ID:** 2611.08931 | [PDF](https://arxiv.org/pdf/2611.08931v1)

**作者:** Maksim Sysoev `[一作]` (Independent Researcher), Danil Somov `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套基于大型语言模型的发现阶段自动化管道，将产品需求、业务需求和功能规格文档以及任务列表自动生成并评估任务复杂度。

**💡 创新点**

创新点在于将LLM与结构化技能组合，用专门的技能自动生成三类文档并实现基于TLM框架与市场/资源视角的双维度复杂度评分，同时在白板和任务管理工具中嵌入可审计的任务链。

**🔧 技术方法**

使用的技术包括Claude Sonnet 4.6 LLM、专门的prompt技能、内部检索服务、Python脚本与SaaS白板/项目管理接口以及基于Tran2004的TLM复杂度公式。

**📊 数据集**

实验数据来源于一次真实的金融科技项目的需求文档与任务集，没有使用公开数据集，仅基于该项目的PRD、BRD、FSD及内部知识库。

**📈 对比分析**

与传统手工编写对比的评价采用人工评审打分和执行时间/token/成本指标，结果显示FSD生成耗时最长、成本最高但质量最低；相对手工过程，整体时长约62分钟，但仍需大量后期人工修订。

**⚠️ 局限性**

主要局限在于仅单次实验、缺乏多项目复现、LLM输出随机性及评审主观性导致指标不具统计意义，同时FSD生成质量差需要进一步提升。

---

## 230. The Drawing You Watch Is the Portrait You Keep: A Replayable Machine Portrait Studio

**arXiv ID:** 2611.08994 | [PDF](https://arxiv.org/pdf/2611.08994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 231. Grid-Oriented Probabilistic Load Forecasting for AI Data Centers

**arXiv ID:** 2611.08929 | [PDF](https://arxiv.org/pdf/2611.08929v1)

**作者:** Qingyi Yao `[一作]` (Southern Methodist University), Jianhui Wang `[通讯]` (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个开放的GPU功率预测基准，并通过四个阶段（预测、在线校准、时序情景生成、全时段备用定量）评估其在电网决策中的价值。

**💡 创新点**

创新点在于：①引入工作负载状态条件的在线合适校准；②使用过去误差重排的Schaake shuffle恢复时序相关性；③通过分阶段评估系统定位不同环节对备用成本的影响。

**🔧 技术方法**

采用的技术包括：LightGBM量化回归、Temporal Convolutional Network（TCN）量化预测、在线合适校准、Gaussian Copula、残差自助抽样和Schaake shuffle情景生成，随后进行备用定价。

**📊 数据集**

数据集为MIT Supercloud TX‑Gaia 448‑GPU AI集群的5 min分辨率功率遥测和作业调度日志（2021 年2月–10月），构成三个不同时域的预测任务。

**📈 对比分析**

比较方法使用CRPS、覆盖率、Winkler得分、ES、VS、KS距离和实际备用成本；结果显示：TCN在未校准时已近似90%覆盖率，在线校准后覆盖率提升至87%；Schaake shuffle情景在所有任务中将备用成本降低约10%，并使短缺率逼近目标。

**⚠️ 局限性**

局限性包括：仅针对单一集群数据，工作负载状态划分依赖事先标注；情景生成假设过去误差可代表未来；评估未考虑不同电网需求曲线和成本参数变化。

---

## 232. Glyph: An Interpretability Engine for AI-Driven Scientific Discovery in Clinical Data

**arXiv ID:** 2611.08922 | [PDF](https://arxiv.org/pdf/2611.08922v1)

**作者:** Robbie Holland `[一作]` (Stanford University), Sergios Gatidis `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了Glyph框架，利用基于大规模临床数据的基础模型（FM）嵌入，通过稀疏自编码器提取可解释概念，并将这些概念用于预测未来疾病并与现有文献和ICD编码进行对照。

**💡 创新点**

创新点在于：①将FM嵌入转化为可解释、可验证的概念；②采用机制解释法自动挖掘概念；③通过数据驱动的混杂因素调整提升概念的独立预测性；④实现了跨机构可复制性与开源工具（Glyph MCP）支持开放式假设检验。

**🔧 技术方法**

使用技术包括：稀疏自编码器（SAE）对图像、文本和实验室嵌入进行分解；机制解释（对比学习）生成概念文字表述；双机器学习（DML）做混杂调整；AUC、Bonferroni、多重检验校正和FDR等统计评估方法。

**📊 数据集**

数据集：三家医院的腹部CT队列，14,652患者（3,262,560切片、8,083,882报告词、1,588,003实验室条目），并在内部测试集和两家外部站点（共26,976患者）进行验证。

**📈 对比分析**

比较方法：将Glyph概念与从文献检索得到的412条预测因子、以及Phecodes进行对比；在内部测试集和外部站点上评估AUC。结果显示，加入Glyph概念可使AUC平均提升6.64%（最高达11.7%），并且Glyph概念在跨机构复制率、效应大小和独立性方面均优于文献因子和Phecodes。

**⚠️ 局限性**

局限性包括：①仅包含腹部CT患者，可能导致结果偏向腹部疾病；②使用ICD编码作为终点，存在编码噪声和时序不准；③未涵盖所有临床模态（如药物、基因等）；④混杂调整仅基于观测变量，无法证实因果；⑤部分概念多义性、模型对其他人群的推广性待验证。

---

## 233. Calibrating your confidence requires thinking of all the ways you could be wrong

**arXiv ID:** 2611.08997 | [PDF](https://arxiv.org/pdf/2611.08997v1)

**作者:** Marco Miani `[一作]` `[通讯]` (Technical University Of Denmark), Marco Miani (Technical University Of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的贝叶斯后验分布——退化高斯（其协方差为与Fisher信息矩阵零空间对齐的投影矩阵），并给出了两种高效算法（Alternating Projections 与 Sketched Lanczos）实现后验采样与置信度估计；通过理论分析（信息几何与PAC‑Bayes）证明该后验能够自然分离参数扰动的功能变化与不变子空间。

**💡 创新点**

1) 先前的贝叶斯近似要么计算量大、精度低，要么缺乏对模型不变性的考虑；本工作通过投影协方差实现了既保持全相关信息又可扩展到亿级参数；2) 通过“投影与拉普拉斯纠错”把非线性Laplace近似的误差归因于投影不对齐，提出了全局对齐的投影后验；3) 在大模型（如Transformer/Large CNN）上实现了SOTA置信度校准与OOD检测。

**🔧 技术方法**

信息几何（Fisher信息矩阵、投影矩阵）、PAC‑Bayes通用界、变分推断（Gaussian族）、矩阵投影与矩阵推送（pullback/pushforward）、Sketched Lanczos迭代、Alternating Projections迭代优化、梯度/自然梯度推导、线性化神经网络与神经切线核（NTK）。

**📊 数据集**

主要使用公开的视觉与语言基准：CIFAR‑10/100、ImageNet、MNIST、CIFAR‑Tiny等；在Transformer/ResNet等大模型上进行训练和验证；OOD检测采用标准OOD数据集如CIFAR‑10→CIFAR‑100、ImageNet→ImageNet‑R等。

**📈 对比分析**

与多种贝叶斯近似（MC‑Dropout、Deep Ensemble、KFAC‑Laplace、Mean‑Field Variational、SGLD、VAE‑posterior）以及基准确定性模型进行对比。结果表明：1) 置信度校准的Expected Calibration Error（ECE）显著下降；2) OOD检测的AUROC/ACER指标提升5‑10%；3) 采样与推理时间均低于传统拉普拉斯/全相关Gaussian方案。

**⚠️ 局限性**

1) 对投影矩阵的近似需要SVD/Sketch，若数据维度极高仍有内存/时间瓶颈；2) 线性化假设对非线性网络的适用性有限；3) 在极端高度过参数化模型中，零空间维数过大，投影矩阵的数值稳定性和求逆仍是挑战；4) 对非常大规模的训练集，计算Fisher矩阵的近似仍需进一步加速。

---

## 234. VLA Handover: Failure-Prediction-Based Switching from On-Robot to Server VLA Models

**arXiv ID:** 2611.08987 | [PDF](https://arxiv.org/pdf/2611.08987v1)

**作者:** Takayuki Nishio `[一作]` `[通讯]` (Institute of Science Tokyo), Takayuki Nishio (Institute of Science Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种VLA handover机制，机器人在小型本地VLA可能失败时才切换到服务器端的大型VLA，以降低网络请求量并保持任务成功率。

**💡 创新点**

创新点在于使用基于失败预测（RND+动作片段熵）的阈值触发切换的启发式框架，无需对VLA重新训练，且将切换决策与服务器选择、请求成本与成功概率约束统一建模。

**🔧 技术方法**

技术实现包括FIPER式失败预测器、手动重置臂部操作、基于模拟5G/6G延迟的远程推理、以及在LIBERO仿真环境中评估的多任务机器人控制。

**📊 数据集**

数据集采用LIBERO仿真基准，涵盖四套10任务的Object、Goal、Spatial和Long四个子任务集。

**📈 对比分析**

与设备端控制、服务器端全推理、周期性查询三种基线对比，VLA handover在Object、Goal、Spatial任务中成功率≈94–95%（略低于服务器端全推理），请求量减少约60–80%，完成时间略增；在Long任务中成功率仅约64%，表明在任务长、VLA小模型表现差的情况下效果有限。

**⚠️ 局限性**

限制主要包括仅在单机仿真环境下评估、未考虑服务器负载与动态网络、未对失败后何时停止切换进行实时预测、以及缺乏真实机器人实验。

---

## 235. NotaryMark: Multi-Bit Watermarking for Black-Box DatasetUsage Auditing in Text-to-Image Diffusion Models

**arXiv ID:** 2611.08985 | [PDF](https://arxiv.org/pdf/2611.08985v1)

**作者:** Xinyi Yu `[一作]` (Xi'an Jiaotong University), Zhikun Zhang `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了第一个多比特数据集水印框架，用于黑盒文本到图像扩散模型的训练数据版权审计。

**💡 创新点**

通过在扩散模型潜在空间嵌入结构化残差水印，并配合 U‑Net 预测器、多样本投票与 BCH 纠错，解决了水印在噪声‑去噪过程中衰减、语义干扰和采样随机性的问题。

**🔧 技术方法**

使用潜在空间对齐的水印编码、联合优化的预测器与判别器、两阶段训练策略、投票聚合、BCH 误差校正及对抗性鲁棒性评估等技术。

**📊 数据集**

在 Pokémon、CelebA（配文本）和 WikiArt 三个数据集上，结合 Stable Diffusion v1.5/XL 与 Kandinsky 2.2 等模型进行实验。

**📈 对比分析**

与 DiffusionShield、DwtDct、SSL、RivaGAN 四个基线对比，在 8–1024 位容量下平均位准度从 0.89 降至 0.67，仍显著高于基线的 0.42–0.60；30 次查询即可实现 100% 复原；在多种预处理、后处理及自适应攻击下保持 TPR 100%、FPR ≤3%，误码率 0.3–0.7。

**⚠️ 局限性**

局限包括对低水印率的鲁棒性不足、对强模糊或重建攻击的抵抗需要更高数据损失、对非 LoRA 微调方式的评估不足以及训练成本和多信息集成的复杂性。

---

## 236. Global Sinkhorn Convergence and Spectral Stability of Matrix Scaling

**arXiv ID:** 2611.08963 | [PDF](https://arxiv.org/pdf/2611.08963v1)

**作者:** Kun He `[一作]` `[通讯]` (Renmin University of China), Kun He (Renmin University of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文为Sinkhorn-Knopp（SK）算法提供了一个全局谱界。研究了行随机非负矩阵A的双随机缩放B的性质，并给出了SK算法在特定条件下的迭代复杂度。

**💡 创新点**

创新点在于通过谱分析提供了SK算法的全局收敛性界限，并且构造了严格正输入的例子，展示了在固定间隙情况下的迭代下界。

**🔧 技术方法**

使用了谱分析技术，结合了矩阵的行列随机性和双随机缩放的性质，构建了全局收敛性界限。

**📊 数据集**

使用了行随机非负矩阵A及其双随机缩放B，具体数据集未明确给出，但涉及到的矩阵具有特定的行列和性质。

**📈 对比分析**

与现有方法的比较显示，SK算法在特定条件下的迭代复杂度为O(D + log n + log(1/ε)/g)，其中D是缩放半径，g是谱间隙，性能优于一些已有的界限。

**⚠️ 局限性**

限制在于对于某些输入，SK算法的迭代复杂度可能达到Ω(n)，并且在特定条件下，谱不等式不能被非严格形式所替代。

---

## 237. Age of Information for Power-Saving Devices with DRX Mechanism

**arXiv ID:** 2611.08949 | [PDF](https://arxiv.org/pdf/2611.08949v1)

**作者:** Hung-Chun Lin `[一作]` (National Taiwan University), Hung-Yu Wei `[通讯]` (National Taiwan University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了DRX机制对UE信息新鲜度的影响，利用Age of Information指标推导了睡眠比例和峰值平均AoI的闭式解，并提出了两阶段DRX服务策略以在保证功耗节约的前提下最小化平均AoI。

**💡 创新点**

首次将AoI指标引入DRX功耗与信息新鲜度的权衡研究；给出了DRX参数下的峰值平均AoI闭式表达式；提出了基于睡眠时间和空闲计时器的直观两阶段DRX参数配置策略。

**🔧 技术方法**

使用半马尔可夫过程建模DRX状态转换，推导期望停留时间和稳态概率；运用Poisson到达过程和期望延迟分析得到睡眠比例和有效到达率；通过Monte‑Carlo仿真验证理论结果并比较与理论下界。

**📊 数据集**

仿真采用λ＝0.1 packet/ms的Poisson到达过程，DRX参数取多种组合（T_o、T_i、T_s）以覆盖广泛情形；未使用实际通信数据集，而是基于随机过程模型。

**📈 对比分析**

通过与理论最优下界（Δ_LB＝1/λ＝10 ms）以及不同DRX配置下的仿真结果比较，显示两阶段DRX策略能使平均AoI接近下界，并在多种睡眠比例下保持低于其他随机配置的平均AoI。

**⚠️ 局限性**

局限性包括：忽略了传输时延和信道误码；仅考虑单UE的DRX功耗；假设到达过程为Poisson，未涵盖突发或时变流量；未考虑网络层级的协同调度或多UE干扰；对DRX最小睡眠时间的设置缺乏实验验证。

---

## 238. SkinMind: Equipping Agentic RAG with Human-Inspired Graph Probing for Reliable Dermatology AI

**arXiv ID:** 2611.08935 | [PDF](https://arxiv.org/pdf/2611.08935v1)

**作者:** Jing Hao `[一作]` (Ant Group), Jian Wang `[通讯]` (Ant Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了SkinMind，一种多模态代理式检索增强生成框架，能够在皮肤科知识图谱上交互式探测文本、关系和视觉证据来支持临床问答与诊断；

**💡 创新点**

创新点在于将知识图谱作为可交互环境，设计了八种层次化图探测工具并采用渐进披露策略，实现自适应多轮检索和推理；

**🔧 技术方法**

技术上结合了多模态大语言模型、代理式检索增强生成、图知识库构建、层次化图探测接口、视觉文本检索（DermLIP）、嵌入向量检索与工具调用；

**📊 数据集**

使用了由227本临床指南和542本教材构建的SkinMind‑KG（超过1百万节点、2百万边）以及423k张Derm1M图像，评估数据为六个皮肤科VQA基准（SkinCon、25Cls、Derm‑Edu、DDI、SNU134‑4、Morph‑R）；

**📈 对比分析**

与直接回答、Naive RAG、GraphRAG、LightRAG、A‑RAG、RAGentA等基线在GPT‑5.4‑mini和Gemini‑2.5‑Flash两种政策模型下进行对比，SkinMind平均提升约8–9分，超过所有基线并接近强大独立模型；同时在证据Token消耗上更高效；

**⚠️ 局限性**

局限性包括对皮肤科知识图谱质量的依赖、对代理模型推理能力的限制、每轮工具调用带来的推理延迟，以及在复杂临床场景下仍可能出现错误或缺失专家审核的情况。

---

## 239. DRAFT-Bench: Evaluating Frontier Agents Across the Professional Design Lifecycle

**arXiv ID:** 2611.08933 | [PDF](https://arxiv.org/pdf/2611.08933v1)

**作者:** Rong Pan `[一作]` (University of Maryland), Jianqiang Wang `[通讯]` (curify-AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估 DRAFT-Bench 评价框架，针对专业设计生命周期中的需求发现、创意探索、迭代修订和交付交接等阶段，测试前沿通用代理的执行与合规性。

**💡 创新点**

提出 Requirement-to-Artifact 评估框架，将需求、受保护属性、artifact 版本与交付义务关联，分层展示结构、语义和工序证据；并通过分离协议揭示代理能力、合规性与评估歧义的三大发现。

**🔧 技术方法**

使用 Codex/Astra 与 Claude Code 两大模型，配合自定义任务协议、人工评审和 Gemini 2.5 Pro 语义检查。

**📊 数据集**

基于 35 条设计简报（共 43 条执行条件）和 40 条 RFQ 场景（两组不同配置），共 272 条工作链和 301 条协议单元。

**📈 对比分析**

通过对比完成率、结构信用、语义标记和时间，Astra 完成 43/43 条条件，Claude 35/43；在共 28 条核心条件中，Astra 13/32 条出现语义标记，Claude 5/28；RFQ 交付覆盖率近 99%，但两组排序差异显著，表明仅靠覆盖率无法判断下游交付成功。

**⚠️ 局限性**

局限包括：RFQ 与设计执行未配对，缺乏专家验收；评估依赖人工判定与模型判断，缺少统一客观基准；工具、运行时、平台差异导致难以归因；缺乏工厂验收和生产可行性验证。

---

## 240. UAV-Based Computer Vision for Cattle Detection and Feed Bunk Spatial Analysis

**arXiv ID:** 2611.08928 | [PDF](https://arxiv.org/pdf/2611.08928v1)

**作者:** Pedro Oliveira Torrente `[一作]` (University of São Paulo), Liang Zhao `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套基于无人机影像的计算机视觉框架，利用YOLO检测牛头与牛背并结合SAM分割饲槽，计算牛在饲槽附近的排列与间距指标，用于精准畜牧养殖中的即时饲槽空间评估。

**💡 创新点**

双类注解（牛头/牛背）提升检测与定位精度；将饲槽分割为直线参考，结合几何距离与间距统计，形成可解释的瞬时空间分布指标；在真实商业饲料槽场景下验证该方法的可行性。

**🔧 技术方法**

YOLOv8/YOLOv11目标检测、Segment Anything Model（SAM）语义分割、几何分析（正交距离、投影、RMSD、CV、GR、DBSCAN聚类），使用Python与深度学习框架。

**📊 数据集**

Campanário Farm公开的UAV影像数据集，232张高分辨率图像，5,027个标注实例，分为训练/验证/测试，标注为牛头和牛背两类。

**📈 对比分析**

对YOLOv8与YOLOv11各尺寸模型做mAP50/50-95评估，YOLOv8s在mAP50 84.3%（牛头）且推理时间约2.7ms/图；YOLOv11s略高但YOLOv8s更稳定；检测后使用SAM分割饲槽，计算RMSD、CV等指标能区分规律与混乱的饲槽占用。

**⚠️ 局限性**

数据量有限，样本多样性不足；指标未与专家饲槽评分或动物行为数据交叉验证；仅处理直线饲槽，曲线或L形饲槽需改进；仅做瞬时分析，未实现连续追踪。

---

## 241. ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient Quantization Aware Training

**arXiv ID:** 2611.08924 | [PDF](https://arxiv.org/pdf/2611.08924v1)

**作者:** Liqun Ma `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ResQAdam，一种为大语言模型量化感知训练设计的低内存优化器；

**💡 创新点**

通过将梯度分解为低秩主子空间与残差，并分别采用 Adam 与符号更新，保留 QAT 关键残差信号；

**🔧 技术方法**

使用 SVD 提取低秩子空间、BF16 记忆、符号更新及能量加权融合的优化策略；

**📊 数据集**

在从零训练与微调中使用 Llama、Qwen 等 LLM，数据集包括 WikiText-2、C4 等公开语料；

**📈 对比分析**

与 AdamW、GaLore、LDAdam、AdaFactor 等进行对比，保持与 AdamW 相近的精度，同时优化器状态内存下降 71%–74%，吞吐下降不足 2%；

**⚠️ 局限性**

对 SVD 刷新频率与子空间维度敏感，低位宽下仍可能出现性能波动，极大模型的扩展需进一步验证。

---

## 242. BrainCoT: A Multi-Task Zero-Shot Brain Signal Foundation Model with Neurometric-Anchored Chain-of-Thought Reasoning

**arXiv ID:** 2611.09082 | [PDF](https://arxiv.org/pdf/2611.09082v1)

**作者:** Yiyu Gui `[一作]` (Peking University), Yuchao Yang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了BrainCoT，一种多任务、零射脑信号基础模型，将EEG/iEEG任务统一为指令式问答，并在推理过程中加入可验证的链式推理；

**💡 创新点**

创新点包括①在统一指令框架下实现多任务零射推理；②使用神经计量证据锚定的Chain‑of‑Thought使推理可检验；③通过决策一致的损失让推理路径与最终答案保持一致；

**🔧 技术方法**

技术手段为两阶段预训练：Stage‑1利用decoder‑only Transformer + MoE实现信号‑语言对齐；Stage‑2利用频谱熵、功率等神经计量特征生成可验证推理并采用决策一致训练；

**📊 数据集**

使用18.6M信号段，来自18个EEG/iEEG数据集（14 EEG+4 iEEG），并在12个下游数据集（睡眠分期、癫痫检测、帕金森、ADHD、精神分裂、阿尔茨海默等）进行评估；

**📈 对比分析**

与BrainBERT、BIOT、LaBraM、CBraMod、NeuroLM、ELM‑MIL、CSBrain等现有基础模型在零射、1%/10%线性探针下对比；BrainCoT零射ACC 67.41%、AUC 74.33%，比1%线性探针最高基线高出3.85%ACC、5.97%AUC；在全微调和线性探针下同样处于领先或次优位置；

**⚠️ 局限性**

局限性包括：仍受限于大规模脑信号‑文本语料稀缺；推理路径需先前提取神经计量特征，特征选择依赖统计相关性；在强监督场景下部分基线仍可优于其零射表现；需要进一步跨中心验证和更大模型规模以提升泛化。

---

## 243. Fair Selection of Generalized Nash Equilibria: A Comparability Perspective

**arXiv ID:** 2611.09060 | [PDF](https://arxiv.org/pdf/2611.09060v1)

**作者:** Sophie Hall `[一作]` (ETH Zurich), Saverio Bolognani `[通讯]` (ETH Zurich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了基于公平度量的广义纳什均衡（GNE）选择框架，并从可比性理论出发分析了不同公平准则对选择结果的影响；在此基础上给出了单约束情形下的归一化均衡参数化和多约束线性二次游戏下的多参数最佳响应求解两种可计算方法；

**💡 创新点**

核心创新在于将公平度量的可比性需求与GNE选取关联起来，提出了公平GNE（f‑GNE）概念，并证明不同可比性层级对应不同公平度量的不可变性；进一步提供了两类可实现的求解路径——归一化均衡的低维搜索和多参数二次规划的区间参数化，弥补了传统v‑GNE对单位可比性的隐含假设。

**🔧 技术方法**

使用了博弈论中的广义纳什均衡与变分不等式理论，构造了二层优化框架；在单约束情形中利用归一化均衡与参数化VI；在线性二次游戏中采用多参数最优响应分析与乘性参数化二次规划（mpQP）及活跃集枚举；

**📊 数据集**

主要以电动汽车充电游戏作为演示实例，所用参数（如充电需求、价格、容量）为合成情境，无真实数据集。

**📈 对比分析**

通过对不同公平度量（社会福利、最大化、纳什谈判等）的计算结果进行对比，展示了v‑GNE与f‑GNE在分配上的差异；在单约束下，SW选择可得到分段线性闭式解，MM与NBS需在线全局搜索；总体而言，方法在满足可比性假设下能显著降低计算复杂度。

**⚠️ 局限性**

局限性包括：仅适用于凸成本、单约束或线性二次游戏；非凸公平度量（如MM、NBS）仍需在线求解；对可比性假设的依赖可能在实际应用中难以验证；扩展到非二次或多约束一般情况仍是开放问题。

---

## 244. MV-IVF: Multi-Vector Retrieval via Multi-Vector Clustering

**arXiv ID:** 2611.09040 | [PDF](https://arxiv.org/pdf/2611.09040v1)

**作者:** Kishen N Gowda `[一作]` (University of Maryland), Rajesh Jayaram `[通讯]` (Google Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于多向量聚类的 IVF 树索引 MV‑IVF，用点云中心进行路由并直接在多向量空间内检索；

**💡 创新点**

首次将 Lloyd‑style 多向量聚类（MV‑Lloyd）与层级 IVF 结构结合，利用 Chamfer 距离的聚类更新能转化为单向量聚类；同时在搜索中实现低位量化、查询压缩、批量化缓存友好搜索；

**🔧 技术方法**

多向量 Chamfer 聚类、Lloyd‑style 更新、层级 IVF 结构、最佳先行 beam 搜索、低比特量化（Rand‑1bit、TQ‑4bit、Rand‑8bit）、查询聚合压缩（Ball‑Carving、Ward）、批量化叶子扫描与向量化矩阵乘；

**📊 数据集**

BEIR 与 LoTTE 文本检索基准，使用 ColBERTv2 的多向量嵌入；

**📈 对比分析**

与 SVH、MUVERA、MV‑Vamana、FastPLAID、IGP 进行对比，MV‑IVF 在 90% Recall‑10@10 时单查询延迟平均提升 1.5‑4.8 倍，批处理吞吐量平均提升 4‑173 倍，且构建时间与索引尺寸保持在可接受范围；

**⚠️ 局限性**

聚类目标非度量化，聚类质量与初始化相关；对较大数据集的分支、叶子大小与迭代次数敏感；图算法 MV‑Vamana 的构建时间高，批量化不如 IVF。

---

## 245. ToolDex: An Ego-Exo Human Demonstration Dataset for Creative Physical Tool Use

**arXiv ID:** 2611.09036 | [PDF](https://arxiv.org/pdf/2611.09036v1)

**作者:** Mohamed Elmoghany `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并公开了一个大规模 ego‑exo 视角人类创意工具使用演示数据集 TOOLDEX，并提供了完整的手‑工具‑目标交互注释；同时构建了涵盖工具推理、3D/4D 重建与机器人策略学习的统一基准。

**💡 创新点**

① 将手‑工具‑目标三角色分解并在真实场景中系统捕捉；② 同步双目视角与八视角 RGB‑D 相机实现高保真多视角注释；③ 设计三层次基准（推理、重建、策略），连接视觉语言、空间感知与执行。

**🔧 技术方法**

多视角同步采集、FoundationPose+多视角融合的 6‑DoF 工具位姿估计、WiLoR+多视角融合的 MANO 手位姿、SAM2+DINOV2 的目标分割、Gaussian Splatting（InstantSplat、IS‑4D）+Diffix3D、VLM 推理（GPT‑6‑Astra 等）以及强化学习（ManipTrans、Dex‑X RL）等技术。

**📊 数据集**

新收集的 TOOLDEX 数据集（10,853 试验、45 工具、16 目标类型、16 动作），在公开的 8‑摄像头+2‑头盔多视角中标注。

**📈 对比分析**

在工具选择、空间定位、行为规划等 VLM 任务中，GPT‑6‑Astra 达到 80% 工具选择准确率、0.891 IoU；在 3D/4D 重建中，IS+Difix3D 获得 20.45 dB PSNR、55.11 MUSIQ；在机器人策略学习中，Dex‑X RL 成功率 55.1%，显著优于直接运动转移（14%）和 BC（4.6%）。

**⚠️ 局限性**

仅覆盖桌面场景，工具种类有限；相机数量与视角受限导致部分遮挡；自动化注释仍需人工干预；对机器人形态的直接转移仍有显著差距，尤其是高接触强度任务；未覆盖非桌面或动态环境。

---

## 246. SIMBAD: Spatio-Temporal Traffic Forecasting Robust to Aperiodicity

**arXiv ID:** 2611.09015 | [PDF](https://arxiv.org/pdf/2611.09015v1)

**作者:** Daniel Y. Lee `[一作]` (Korea University), Susik Yoon `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SIMBAD 框架，利用相似度驱动的时间调节与空间调节来提升交通流量预测，尤其在非周期性事件下。

**💡 创新点**

通过 L1 相似度与不确定性估计动态过滤历史周期信息，并构建基于相似度的语义邻接矩阵，实现对历史输入的可控调节。

**🔧 技术方法**

轻量化 L1 距离相似度、不确定性估计、软最大加权聚合、可学习阈值掩码、相似度驱动的自适应邻接矩阵，配合 Transformer + 图卷积 + 门控时序卷积的时空建模层。

**📊 数据集**

四大基准数据集 PEMS03/04/07/08 以及 METR-LA、PEMS-BAY、LargeST 三个补充数据集。

**📈 对比分析**

与 11 个基线（GNN、Transformer、Memory）对比，SIMBAD 在 MAE/RMSE/MAPE 上平均提升 4‑7% 以上，尤其在高不规则性区间表现最佳；模块化后可提升其他模型 2‑8% 之间。

**⚠️ 局限性**

仍依赖历史窗口的选择，对极端稀缺/异常事件的鲁棒性有限；仅使用 L1 距离可能不足以捕捉更复杂相似度；模块在极大图规模下的计算成本尚需进一步优化。

---

## 247. ISER-MoE: Dynamic In-Switch Replication of Hot Experts for Load Balancing in Expert-Parallel

**arXiv ID:** 2611.09008 | [PDF](https://arxiv.org/pdf/2611.09008v1)

**作者:** Jintao Peng `[一作]` (NVIDIA), June Yang `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为Mixture‑of‑Experts (MoE) 模型的预填充阶段设计了一种在每一次调用内动态复制热点专家权重的方案，利用 NVLink 交换机的多播功能将权重一次性复制到预先划分的 GPU 组中，并在共享专家计算期间重叠复制与规划工作。

**💡 创新点**

创新点在于：①将专家复制的目标从“复制组”改为“固定对齐组”，实现对热点专家的按需动态复制；②使用多端口的 in‑switch multicast 传播权重，避免单点复制热点导致的流量拥塞；③通过排队、配额修复等调度算法，保证复制后负载平衡且本地计算率最大化；④将复制计划与共享专家计算重叠，最小化复制对推理时延的影响。

**🔧 技术方法**

核心技术包括：NVLink SHARP（in‑switch multicast）; TensorRT‑LLM 中的 TMA 复制核; 统一的多级对齐组与多端口广播；配额修复与本地优先路由重写；以及对共享专家的动态调度和 READY 触发同步。

**📊 数据集**

实验使用 DeepSeek‑V4‑Pro（FP4/FP8）MoE 模型，主要数据集包括 GSM8K、MMLU、IFBench、SciCode、以及从 SemiAnalysis 提取的 agentic 编码追踪和随机 token 流。还在预发布的 Rubin GPU 与 GB300 NVL72 机架上进行离散式推理实验。

**📈 对比分析**

与无复制（OFF）以及强制平衡路由（FB）等基线比较，结果显示：在 EP8 条件下，TRACE 重放中复制方案将负载不平衡度 R 降至 1.002‑1.020，98‑99% 的过载被消除；在 GB300 上预填充吞吐量提升 17.4%（GSM8K）和 7.2%（agentic prompts）；在 Rubin GPU 上提升 6.1%。同时，多播复制相比传统点对点复制大幅减少热点 GPU 的发送负载，并保持整体吞吐。

**⚠️ 局限性**

局限性包括：①仅在预填充阶段和单一 NVLink 域内实现；②不支持 CUDA graphs、分布式多域或生成阶段；③需 NVFP4/FP8 兼容、DEP 并行；④复制只保留一次调用，后续迭代可能出现复制冗余；⑤在高 EP（>8）下规划与复制开销显著增长；⑥缺乏与静态复制策略的硬件级直接对比。

---

## 248. GlobalPuzzle: Globally Informed Genetic Search for Eroded Jigsaw Puzzle Reassembly

**arXiv ID:** 2611.09077 | [PDF](https://arxiv.org/pdf/2611.09077v1)

**作者:** Lunyan Wang `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 GlobalPuzzle 框架，结合全局画布评估、策略网络指导放置和遗传搜索，解决破损拼图重组问题。

**💡 创新点**

创新点在于将全局视觉评估与遗传搜索相结合，使用策略网络生成位置查询并匹配碎片，同时保留连通组关系，实现可变尺寸、形状拼图的无须重新训练的重组。

**🔧 技术方法**

使用了全局评估网络（多尺度 CNN+Transformer）、策略网络（Transformer + 位置查询）、遗传算法（交叉、变异、局部搜索）以及 LaMa 进行碎片去噪。

**📊 数据集**

主要使用 GAP-3 与 GAP-5 数据集（分别为 3×3 与 5×5 的破损拼图），并在真实照片与不同网格尺寸上进行验证。

**📈 对比分析**

与经典贪心、传统遗传算法以及多种深度学习方法（JigsawGAN、DiffAssemble、PuzzleFlow 等）对比，GlobalPuzzle 在 GAP-3 上达到 68.6% 完全组装率、81.6% 片段准确率；在 GAP-5 上则分别达到 19.4% 与 48.7%。

**⚠️ 局限性**

局限性：在更大尺寸的拼图（如 5×5）上仍难以达到高准确率，搜索空间庞大导致求解时间增长；对碎片形状不规则和极端破损的鲁棒性待进一步提升。

---

## 249. When Routine Operations Break Architectural Pattern Relationships in LLM Agent Harnesses

**arXiv ID:** 2611.09088 | [PDF](https://arxiv.org/pdf/2611.09088v1)

**作者:** Boming Xia `[一作]` (Responsible AI Research Centre), Liming Zhu `[通讯]` (CSIRO)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 LLM agent harness 的架构关系进行实证研究，评估不同插件配置与运行时操作对 Guardrail Enforcement Point、Autonomy Controller、Blackboard 等关系的保持情况。

**💡 创新点**

提出 PatternOps 评估流程，将模式关系拆解为可观察需求并识别其 carrier（回调顺序、状态持久化），揭示相同插件在不同操作历史下如何导致关系失效，并系统化地给出检验问题与成本权衡。

**🔧 技术方法**

采用模式分析、源代码审计、运行时记录联合评估、插件回调链追踪、状态迁移验证等技术；结合事件记录和决策日志进行因果分析。

**📊 数据集**

使用十四个独立插件样本，覆盖 DeepSeek Harness、OpenCode、Claude Code、Deep Agents 等公开发布的 harness 版本，构成多案例对比实验。

**📈 对比分析**

通过多案例对比和操作变化实验，比较不同分配与操作组合对需求满足率的影响；实验显示相同插件在不同历史下行为差异显著，某些修复仅在单一操作下有效，未出现整体性能瓶颈。

**⚠️ 局限性**

局限在于仅分析已发布版本和有限的操作范围、部分结论依赖研究设定、未覆盖全部可能配置与实时模型动态，缺乏大规模真实环境验证。

---

## 250. WearableBridge: Task-Guided Latent Transport for Free-Living Activity Recognition

**arXiv ID:** 2611.09067 | [PDF](https://arxiv.org/pdf/2611.09067v1)

**作者:** Jingjing Zou `[一作]` (University of California San Diego), Yixin Wang `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 WearableBridge 方法，利用有限高质量参考数据通过学习桥接映射，将腕部加速度特征迁移到固定参考分类器的活动空间，以提升自由生活腕部活动识别。

**💡 创新点**

创新点在于保留参考分类器不变，仅学习一个小型桥接模型，并将腕部特征映射到参考特征空间的决策区域，实现对无参考腕部数据的显著性能提升。

**🔧 技术方法**

采用任务引导的潜在传输技术，构建 Mapper、Distillation、MeanFlow 三种桥接实现，并结合特征标准化、温度软化、流式匹配等方法；使用冻结的编码器与分类器并通过交叉熵及参考引导损失训练。

**📊 数据集**

在 PAAWS、selfBACK、CAPTURE-24 三个真实人群数据集上实验，使用 MoCA、DeepConvLSTM、InceptionTime、HART 等编码器。

**📈 对比分析**

与未映射基线、直接腕部分类器、仅使用活动标签的桥接等进行对比；在 Thigh2Wrist 上宏 F1 从 35.07% 提升至 55.98%，在 Lab2Freeliving 上提升至 63.54%；在 selfBACK 和 CAPTURE-24 也分别取得显著提升。

**⚠️ 局限性**

受限于校准样本规模、仅使用单一参考类型、桥接架构与损失的分离未完全评估，且结合多种参考（如腿部与实验室腕部）可能进一步提升但尚未实现。

---

## 251. OmniLatent: Unified Latent for 3D Part Modeling

**arXiv ID:** 2611.09055 | [PDF](https://arxiv.org/pdf/2611.09055v1)

**作者:** Yunhan Yang `[一作]` (University of Hong Kong), Chunchao Guo `[通讯]` (Tencent Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 OmniLatent，一体化潜在空间模型，能从单张图片一次性生成几何、变形、纹理和面部部分标签，并支持点提示细化与全局部分补全。

**💡 创新点**

核心创新是共享潜在空间将几何、外观和二进制部分 ID 同时编码，利用大规模几何/纹理先验提升稀缺部分标注学习，并设计点提示适配器和自适应令牌分配的全模态补全模块。

**🔧 技术方法**

技术实现基于 VAE+MM‑DiT（rectified flow）联合训练、CA‑RoPE 条件感知位置编码、RoPE/无RoPE 令牌混合、动态令牌分配与局部/全局注意力交替等。

**📊 数据集**

使用的数据集包括 900k Objaverse 纯几何、300k 纹理资产以及 20k 手工标注部分的 Objaverse 物体；评估集为 PartObjaverse‑Tiny‑wt、ObjaversePart‑Tiny 等。

**📈 对比分析**

与 Find3D、SAMPart3D、PartField、P^3‑SAM 等基线对比，在 mIoU 上分别取得 74.18（mask‑guided）/62.42（自驱）并超过最强基线 56.01；在全局/局部补全上，Chamfer Distance 0.03、F‑score 0.78，优于 X‑Part、HoloPart 等。

**⚠️ 局限性**

局限性包括对称结构易出现不一致或部分合并；自驱分割受限于训练部分数据多样性；2D 掩码引导可能导致背面错误，需要加入对称/连通约束等后续改进。

---

## 252. StreamTab: Training-Free Temporal Link Prediction with Tabular Foundation Models

**arXiv ID:** 2611.09052 | [PDF](https://arxiv.org/pdf/2611.09052v1)

**作者:** Mingqi Yang `[一作]` (South China University of Technology), Renqi Wu `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的时序链接预测框架，将时间图的交互历史转换为表格特征，并通过冻结的Tabular Foundation Model（TabPFN）对正负样本进行无训练预测。

**💡 创新点**

创新点包括：① 采用预训练的冻结TabPFN进行无训练预测；② 通过 regime conditioning 让负样本与评估分布保持一致；③ 仅使用统计特征即可在三种负采样场景下竞争并常常超越传统基线。

**🔧 技术方法**

使用技术：TabPFN v2 预训练表格基础模型、对 pair recurrence、endpoint activity、shared neighbors 等特征的统计化、regime conditioning 负采样、流式上下文更新与刷新策略。

**📊 数据集**

使用数据集：DyGLib benchmark 的13个时序图数据集（如 Wikipedia、Reddit、MOOC、LastFM、Enron 等）。

**📈 对比分析**

方法与性能：与21个时序图基线（TGN、DyGFormer、EdgeBank 等）在随机、历史、归纳负采样以及新节点评估上对比，平均精度（AP）在历史与归纳采样下常居首位，随机采样亦表现优异，超过多数基线。

**⚠️ 局限性**

局限性：仅基于事件时间和节点ID的统计特征，缺乏属性与结构嵌入；在随机采样和新 pair（未出现过的节点对）时效果相对下降；上下文样本的选择与刷新策略仍可进一步改进。

---

## 253. When Can You Trust Neural Critics? Auditing Mutual Information Estimates of Higher-Order Epistasis in Protein Language Models

**arXiv ID:** 2611.09046 | [PDF](https://arxiv.org/pdf/2611.09046v1)

**作者:** Maryam Rahimimovassagh `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过审计实验评估并改进了蛋白质语言模型中使用神经互信息估计器的顺序分辨依赖声明的稳定性和可靠性，提出了可重复使用的诊断协议。

**💡 创新点**

发现不稳定主要源于统计量的“峰比”构造而非估计器本身，并通过引入直接平均差异统计量、三轴（种子、运行、配对）稳定性检查以及多估计器与维度验证，提供了系统的诊断清单。

**🔧 技术方法**

技术手段包括 Donsker–Varadhan / MINE 变分 MI 估计器、InfoNCE、SMILE、Bounded-DV 变体；Walsh 变换、Ridge 回归、synthetic Gaussian 通道模拟，以及配对方案、交叉种子/运行/配对敏感性评估。

**📊 数据集**

使用了四位 GB1 突变体实验数据（149,361 变体）以及 ESM-2、ESM-1v、ProtT5 的表示；还构造了高维度（5,120-D、4,096-D）Gaussian 通道作为 synthetic 校准。

**📈 对比分析**

通过比较不同估计器在跨种子、跨运行、跨配对条件下的方差、均值与信号门阈值，发现原始峰比统计在 ESM-2 上不稳定，新的平均差异统计在两种估计器下稳定，但在更高维度 synthetic 通道上均未通过强信号门；Oracle 控制可恢复约 93–99% MI，Ridge 回归显示线性可解度。

**⚠️ 局限性**

局限性：仅在单一四位 GB1 研究景观和单一交互顺序问题上验证；评估的估计器集合有限；未检验模型特定效应或身份无关泛化；Oracle 控制未揭示 blind 失效的根本原因；诊断协议的普适性未在其他生物系统或表示类型中测试。

---

## 254. Did I See It or Think It? Probing Reality Monitoring in Vision-Language Models with Concept Injection

**arXiv ID:** 2611.09014 | [PDF](https://arxiv.org/pdf/2611.09014v1)

**作者:** Tina Khezresmaeilzadeh `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在视觉‑语言模型（VLM）的激活中注入概念方向，并让模型报告可见物体、是否注入思维以及注入的概念，从而评估VLM的“现实监控”能力；

**💡 创新点**

首次将概念注入技术与图像内容独立控制相结合，揭示VLM在检测注入概念时易误将其归因为视觉内容，且该误差受提示、注入强度和注入位置的显著影响；

**🔧 技术方法**

利用激活注入（concept direction）、残差流操纵、不同层级和强度的注入、精细化提示设计，以及多种对照（随机注入、无注入、已知标签等）来评估模型的检测、识别与来源归属；

**📊 数据集**

使用150个具体可视化概念（包括通过Stable Diffusion生成的图像）、12个中性图像以及108张COCO真实图像（覆盖36类），并在每个模型上进行控制实验；

**📈 对比分析**

与纯文本的“steering‑only”基线、随机方向、无注入、错误标签等对照进行比较，Gemma‑3‑27B在无图像条件下达35%（清洁）检测率，误报率仅0.3%，但在存在视觉对象时清洁率骤降至2.7%，显示出显著的来源歧义；

**⚠️ 局限性**

实验仅验证了Gemma‑3‑27B的强大表现，其他VLM在同一任务中表现不佳；结果高度依赖提示语、注入位置和强度；使用的概念集合有限，且注入是一种人工干预，未必能反映自然条件下的自我报告可靠性；

---

## 255. FASTFUNNELS: Agile Multi Robot Navigation in Narrow Spaces via Adaptive Dynamic Funnels

**arXiv ID:** 2611.09005 | [PDF](https://arxiv.org/pdf/2611.09005v1)

**作者:** Vignesh Rajagopal `[一作]`, Rohan Chandra `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种层次化高速度多机器人导航框架，利用可变形椭圆“水滴形”引导与分布式模型预测控制（DMPC）协同实现Ackermann车型在紧凑环境中的安全高效运动。

**💡 创新点**

创新点在于将连续可变形的软形引导与分布式MPC相结合，避免传统硬性形状导致的减速和死锁；通过强化学习训练“blob”引导，实现实时形状自适应；并首次在高速度车动态系统上展示此方法。

**🔧 技术方法**

使用的主要技术包括：PPO强化学习训练领头的椭圆形态与速度策略；DMPC在每个机器人上执行分布式轨迹规划；F1TENTH仿真环境与1/7比例Traxxas UDR车辆的实验平台；基于LiDAR与中心线的感知与信息共享。

**📊 数据集**

数据集方面，作者在F1TENTH仿真中构造了14种不同的障碍/窄巷/急弯/折返场景，并在真实机器人实验中使用15 m室内走廊的350种随机化版本进行训练。

**📈 对比分析**

与DEFORM、ORCA、GCBF+、IPPO等基线比较，系统在所有14个场景的20次试验中完成率平均提升40%，平均速度7.73 m/s，完成时间仅为ORCA的1/3，速度比最快基线高7.69倍。

**⚠️ 局限性**

局限性包括：需要机器人间共享信息，安全性尚未通过正式证明；在粗糙或不规则地形上的适用性未知；方法主要针对Ackermann车动态，需进一步探索完全去中心化和动态障碍场景。

---

## 256. Vertex Deletion on $H$-Induced-Minor-Free Graphs: Subexponential FPT Algorithms and Approximation Schemes

**arXiv ID:** 2611.09071 | [PDF](https://arxiv.org/pdf/2611.09071v1)

**作者:** Daniel Lokshtanov `[一作]` (University of California Santa Barbara), Meirav Zehavi `[通讯]` (Ben Gurion University of Negev)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种通用框架，针对在 H-诱导最小子图自由图以及更广泛的字符串图上，对 (t,ϕ)-删除问题给出子指数参数化算法和准多项式时间逼近方案。

**💡 创新点**

创新点在于证明了一个新的覆盖-打包定理，提供了对树宽模量的结构性分析，并将其推广到所有 H-诱导最小子图自由图，从而解决了此前开放的 Feedback Vertex Set 在字符串图上的子指数算法问题；同时给出了可在准多项式时间内得到 (1+ε) 近似解的 QPTAS。

**🔧 技术方法**

主要技术包括：对 H-诱导最小子图自由图的分割定理修正、Torso 操作的边数上界、子线性树宽分支定理、回溯分支算法、MSO 组合定理与动态规划结合的表格算法，以及对树宽模量的结构性分析与递归分离。

**📊 数据集**

本文没有使用特定数据集，而是针对理论图类（如单位圆盘图、矩形交叉图、字符串图、一般 H-诱导最小子图自由图）进行算法设计与证明。

**📈 对比分析**

与之前的算法相比，本工作在相同图类上实现了更快的运行时间：Feedback Vertex Set 在字符串图上从 2^O(n^2/3) 下降到 2^O(k^2/3)；在更广泛的 H-诱导最小子图自由图上同样取得子指数时间；同时提供了 (1+ε)-近似的准多项式时间。

**⚠️ 局限性**

局限性包括：需要预先知道目标图类 H 的具体结构，且对 ϕ 的定义仍需是固定的 MSO 公式；在图的树宽较大时，QPTAS 的时间仍然是准多项式；若 ϕ 不是 hereditary，则无法获得 QPTAS。

---

## 257. MongoBench: Do LLMs Know What a MongoDB Query Will Do?

**arXiv ID:** 2611.09065 | [PDF](https://arxiv.org/pdf/2611.09065v1)

**作者:** Qibai Chen `[一作]` (Independent Researcher), Mengxiao Qian `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建 MongoBench，评估大型语言模型在给定 MongoDB 文档集合与查询过滤器时，预测查询结果集合的准确性。

**💡 创新点**

创新点在于把查询后果预测视为独立评测任务，设计包含多种文档存储“footgun”规则的合成查询，量化模型对精确集合匹配的能力，并揭示生成查询与预测结果之间的差异。

**🔧 技术方法**

使用大语言模型推理、规则注入、逐文档判定、引擎委托等技术，对四款模型（Claude‑opus‑4、GPT‑5.5、Gemini‑3.1‑pro 及 DeepSeek‑reasoner）进行实验。

**📊 数据集**

数据集由三家前沿 LLM 自动生成的 300 条合成查询样例及其对应的 MongoDB 集合组成，Oracle 通过真实 MongoDB 8.2 服务器计算匹配集合，确保与模型无关的基准。

**📈 对比分析**

通过三次重复实验、计算 exact‑set、per‑doc 与 Jaccard 准确率，并用 McNemar 检验比较四模型；结果显示最佳模型准确率约 92%，最差约 70%，差距约 20%。

**⚠️ 局限性**

局限包括模型预算不统一、仅评估过滤器而未覆盖更新/聚合、未加入人工写作的基准样本、且对最难样本仍缺乏更高难度的测试。

---

## 258. Robust Readout of Cross-Point Resistive Arrays via Time-Domain Optimization

**arXiv ID:** 2611.09059 | [PDF](https://arxiv.org/pdf/2611.09059v1)

**作者:** Taro Tako `[一作]` (National Institute of Informatics), Taisuke Kobayashi `[通讯]` (National Institute of Informatics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于时间域优化的交叉点电阻阵列读取方法。

**💡 创新点**

创新点在于直接利用时间波形拟合而非仅靠频域谱分析，从而显著提升了在噪声环境下的测量鲁棒性。

**🔧 技术方法**

采用了非线性最小二乘优化（Levenberg–Marquardt）以及仿真电路测量技术。

**📊 数据集**

使用两种电路模拟：16个并联电阻与16×16交叉点阵列，并在不同电阻分布（全相同、随机以及触摸响应分布）下进行1,000次仿真实验。

**📈 对比分析**

与传统FFT频域基线方法比较，时间域方法在有噪声时平均误差下降约24.6%，标准差下降约77.6%，最大误差也有所降低。

**⚠️ 局限性**

局限在于需要对每行进行迭代优化，计算开销较大；且未考虑交叉点电路的漏电流等寄生特性。

---

## 259. Equidistribution of Solution Gradients in Physics-Informed Neural Networks via Coordinate Transformations

**arXiv ID:** 2611.09053 | [PDF](https://arxiv.org/pdf/2611.09053v1)

**作者:** Tianchi Yu `[一作]` (Artificial Intelligence Research Institute and Skolkovo Institute of Science and Technology), Ivan Oseledets `[通讯]` (Artificial Intelligence Research Institute and Skolkovo Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证Coordinate Transformation PINNs（CTPINNs），通过可学习或显式构造的坐标变换将物理域映射到计算域，使得解在新坐标系中更平滑、梯度均匀，从而显著降低PINN在高梯度或谱偏差问题上的误差。

**💡 创新点**

①利用梯度等分布原理设计显式Majorant和隐式可逆网络两种坐标变换；②证明在NTK框架下CTPINNs的收敛率优于传统PINN；③将CT模块作为轻量插件，可与现有先进PINN模型（如PirateNet、SPINN、QRPINN）无缝集成；④提供理论分析与实验验证，展示自适应收敛至最优变换。

**🔧 技术方法**

PINN基础、显式Majorant构造、隐式可逆神经网络（INN）与光谱归一化、NTK收敛分析、梯度等分布优化、JAX实现与自动微分。

**📊 数据集**

1D/2D 边界层、Poisson方程；JAX‑PI基准（Allen‑Cahn、Korteweg‑de Vries、Lid‑driven cavity）；SPINN基准（Helmholtz、Klein‑Gordon、Navier‑Stokes）；高维Sine‑Gordon（d=3,30,100,200）等多种 PDE。

**📈 对比分析**

与传统PINN、RAD、ACLE、PirateNet+AT、SPINN、QRPINN比较。CTPINNs在大部分基准上相对L²误差下降10%–70%，在边界层问题实现极小误差；隐式构造对复杂 PDE（如Korteweg‑de Vries、Navier‑Stokes）提升21%–55%；计算成本增加有限，显式构造几乎无额外开销，隐式构造仅略增训练时间。

**⚠️ 局限性**

显式构造依赖先验解信息，难以适用于未知结构；隐式构造缺乏严格的正则化和理论保证，归一化问题待解决；目前未针对不规则域或极端参数场景提供通用方案。

---

## 260. Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation

**arXiv ID:** 2611.09042 | [PDF](https://arxiv.org/pdf/2611.09042v1)

**作者:** Yijun Zhu `[一作]` (Central South University), Chengchao Shen `[通讯]` (Central South University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种结合失败特征分析与分支点可达性评分的数学推理知识蒸馏方法，用于训练小模型。

**💡 创新点**

通过引入基于错误率和一致性的行为失败剖析来判定错误模式，并采用分支点可达性评分挑选教师生成的可实现修正轨迹，解决分布差异与可学习性问题。

**🔧 技术方法**

采用教师-学生双模型框架、错误模式分类、针对模式的提示生成、分支点可达性评分（BPR）以及标准SFT训练。

**📊 数据集**

以MetaMathQA（包含GSM8K、MATH等）为训练集，评估在GSM8K、MATH、SVAMP、ASDiv、MAWPS等分布外数据集。

**📈 对比分析**

与SFT、RFT、ISC、S^3C-Math、GPTAug、RefAug、LEMMA等方法对比，两个学生模型上平均精度最高，尤其在分布外任务提升约3–4个百分点，总平均精度约79–80%。

**⚠️ 局限性**

依赖强教师模型；教师失败导致问题被丢弃；对采样预算K敏感；需多轮采样与生成，成本较高。

---

## 261. A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic Robotics

**arXiv ID:** 2611.09030 | [PDF](https://arxiv.org/pdf/2611.09030v1)

**作者:** Kaiyuan Chen `[一作]` (UC Berkeley), Ken Goldberg `[通讯]` (UC Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在模拟环境中对九个抓取与操作任务进行631个实验，系统研究了Agentic Robotics自学习循环中的模型能力、技能先验、反馈结构与课程设计对政策改进的影响。

**💡 创新点**

创新点在于提出了一个可控实验框架，独立拆分并评估模型、技能库、反馈粒度与课程对自学习效果的作用，并从实验中归纳出六条关键发现。

**🔧 技术方法**

技术上使用大型语言模型（Claude Opus、GPT‑6 Astra 等）生成并编辑可执行机器人代码，结合开源机器人技能库、基于文本/视觉的反馈报告和可视化的执行轨迹。

**📊 数据集**

数据集主要包括VABAR（Tool Hanging、Syringe Packing、Gear Meshing、USB‑C Insertion、Cable Routing）和MimicGen（Kitchen、Coffee、Threading、Mug Cleanup）四个任务集的模拟场景。

**📈 对比分析**

通过与人类手写策略、不同LLM版本以及多种反馈粒度的对比，实验表明强模型和语义化反馈能显著提升自学习成功率（最高可达约90%），但整体仍低于人类策略，说明仍存在空间与几何推理瓶颈。

**⚠️ 局限性**

局限性包括仅在仿真环境评估、未考虑真实世界反馈与人机交互、对模型版本的时间敏感性以及课程设计与反馈细化程度未覆盖所有可能组合。

---

## 262. Language, Not Code Abstraction: Natural-language Training Shapes Brain-aligned Geometry

**arXiv ID:** 2611.09027 | [PDF](https://arxiv.org/pdf/2611.09027v1)

**作者:** Iñigo Parra `[一作]` `[通讯]` (University of California), Iñigo Parra (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

作者训练了 GPT‑2‑small 在英文叙事数据 TinyStories 与多种代码语料（Python、C、x86‑64 汇编及机器码）上，并评估它们在四个脑网络基准上的语言网络对齐。

**💡 创新点**

创新点在于统一 tokenizer、模型架构与训练流程，对比不同代码抽象层次以及英文训练对脑对齐的影响，并通过无拟合 RSA 与 CKA 等 fit‑free 方式揭示表征几何结构。

**🔧 技术方法**

采用 GPT‑2‑small、BPE tokenizer、线性回归编码、线性 CKA、RSA、以及块级迁移与仅嵌入重训练等技术进行实验。

**📊 数据集**

使用的数据集包括 TinyStories、Python、C、x86‑64 assembly、binary 代码语料，以及 Brain‑Score 的 Pereira‑243、Tuckute‑2024、Blank‑2014、Fedorenko‑2016 四组脑响应。

**📈 对比分析**

比较方法是把模型与随机权重模型对齐并计算差异，结果显示英文训练在 Pereira‑243 与 Tuckute‑2024 上显著优于代码模型，且产生更可重复、与脑相似的几何结构；代码模型在大多数基准上仅略高于随机。

**⚠️ 局限性**

局限性包括仅使用 GPT‑2‑small 与单一 tokenizer，训练通量差异大，随机对照仅三次初始化，转移实验仅用单点检查点，且仅评估四个脑基准，无法推广到其他架构或语言。

---

## 263. Beyond a Single Commit: Anchor-Guided Reconstruction of Multi-Commit Patches

**arXiv ID:** 2611.09026 | [PDF](https://arxiv.org/pdf/2611.09026v1)

**作者:** Jialiang Dong `[一作]` (University of Wollongong), Siqi Ma `[通讯]` (University of Wollongong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套统一框架 Anchor-Patched（AP），通过时间序列过滤、异质性LLM确定锚点提交，使用覆盖补全和依赖驱动两种策略重构多提交补丁，并通过LLM精细化过滤噪声，实现在单提交和多提交漏洞补丁识别上的显著提升。

**💡 创新点**

创新点包括：①基于多层次漏洞特征（标识符、哈希、受影响元素、CWE关键词）实现锚点精准定位；②结合AST差异+TF-IDF对代码变更与提交信息进行多维模式匹配，完成跨位置的覆盖补全；③利用CFG/DFG依赖分析构造提交间依赖图，实现多提交依赖驱动重构；④LLM双向细化，既筛选锚点也对候选提交意图进行语义核对，降低误检；⑤在大规模真实项目上构建公开基准并报告未被公开的高危补丁，促进研究透明度。

**🔧 技术方法**

技术手段包括：时间序列提交过滤、正则提取漏洞特征、树状解析器（tree‑sitter）生成AST、AST差异与TF-IDF编码、余弦相似度匹配、文件/代码/元数据多级比较、CFG/DFG构造与前/后向遍历、图连通性检索、LLM（如gpt‑5‑nano）推理与提示工程、实验对比与运行时测评。

**📊 数据集**

使用的数据集为：① 1,207 个 CVE 的基准集（1,019 单提交，188 多提交，共1,583 个提交）；② 30 个主流开源项目的 5,146 个 CVE 进行野生评估；③ 公开的 anchor‑patched 工具与数据集已在 GitHub 上发布，供复现与扩展。

**📈 对比分析**

与 TAPER、VFCFinder、SHIP、Prospector 等四种现有方法在 commit 级、CVE 级（部分/完整）三种评估维度下进行对比；AP 在单提交 patch 上 F1 达 93.02%，多提交 patch 上 F1 75.08%，分别比最佳基线提升约2–20%；在时间消耗上，AP 中位数 40 秒，显著低于 Prospector 的 130 秒；同时在野生数据中识别 3,269 个补丁，包含 323 个高危未被公开的补丁。

**⚠️ 局限性**

局限性包括：① 依赖漏洞报告的完整性，缺失信息导致锚点误识别；② LLM 在推理时易产生幻觉，可能误选或漏选提交；③ 静态 CFG/DFG 仅捕获显式依赖，隐式依赖或复杂代码路径可能导致重构不完整；④ 时间窗口设置可能无法覆盖极端延迟的修复；⑤ 对极大规模项目时候选提交仍可能庞大，影响性能。

---

## 264. Rethinking Latent Reasoning for Generative Recommendation: Beyond Teacher-Generated CoT and Recurrent Hidden-State Feedback

**arXiv ID:** 2611.09006 | [PDF](https://arxiv.org/pdf/2611.09006v1)

**作者:** Yiwen Chen `[一作]` (Beihang University), Zhao Zhang `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过引入PHiT模型，直接利用用户历史交互中的项目编码来监督词汇级思维令牌，避免了教师生成的链式推理和递归隐藏状态反馈，改进了生成式推荐的推理过程。

**💡 创新点**

关键创新在于使用预缓存的稀疏项目编码生成历史语义目标，直接监督思维令牌的隐藏状态，并通过单次因果预填方式执行中间状态，从而消除了教师轨迹和RL训练。

**🔧 技术方法**

采用Qwen3-0.6B作为背骨，SPLADE稀疏编码对项目进行编码，线性分类头用于历史目标预测，联合监督（历史目标+SID预测）以及标准因果预填推理技术。

**📊 数据集**

在Amazon Beauty、Instruments、Sports和Steam四个基准数据集上进行评估。

**📈 对比分析**

与传统顺序推荐、显式CoT监督、LatentR^3等基线对比，SFT单阶段训练即可在所有24项指标上均优于最强基线，显著提升性能且训练/构造成本大幅降低。

**⚠️ 局限性**

实验仅覆盖离线交互日志和文本元数据，未涉及多模态内容、动态商品目录或大规模在线部署，需进一步扩展。

---

## 265. The Invisible Gap: Romanized Indic Generation Fails Where Automatic Metrics Succeed

**arXiv ID:** 2611.09085 | [PDF](https://arxiv.org/pdf/2611.09085v1)

**作者:** Madhu Sudhan Rao Beesetty `[一作]` (PayPal), Lokesh Meesala `[通讯]` (Morgan Stanley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对四大多语种LLM在印度三种罗马化语言（印地语、泰卢固语、卡纳达语）生成文本的质量进行了系统评估，结合失败感知自动指标与母语使用者评价，量化了指标与人工评估之间的巨大偏差。

**💡 创新点**

创新点在于：①提出失败感知评估策略，自动为空、英文本复制、脚本泄露与退化输出分配零分；②构建可复用的十维母语使用者评估量表；③通过实验揭示两类模型特定失败模式（Gemini脚本泄露、Qwen指令崩溃），并证明仅靠自动指标无法捕捉这些错误。

**🔧 技术方法**

技术手段包括BGE-M3与BERTScore两种自动相似度指标的失败感知计算、脚本泄露检测、英文复制率判定、重复/退化识别；母语使用者对诊断子集进行手工打分；对四款LLM的统一 prompt 设计与一次性联合生成。

**📊 数据集**

使用来自OpenAssistant的142条高质量英中问答对作为原始文本，随后对其进行三种印度语言的罗马化，构成实验数据集；诊断子集为72条（每种语言24条）用于人工评估。

**📈 对比分析**

评估方式为：①对全部142条输出应用失败感知BGE-M3与BERTScore，计算平均分并计数失效；②对诊断子集由单位母语评估者按十维量表打分；结果显示GPT‑5.5在自动和人工评估均优于Claude Opus 4.8，但两者在自动指标上的差距仅0.004点，远低于人工评价中1.43点的显著差异；Gemini与Qwen的失败模式导致其自动分数被高估。

**⚠️ 局限性**

局限性包括：仅覆盖三种印度语言；每种语言仅一名评估者，未评估评估者一致性；使用温度0的确定性生成，未估计样本方差；诊断子集样本量小，未代表总体；未涉及多语代码混杂与其他印度语言；仅评估四款LLM，未包含专门的印度语模型。

---

## 266. Training-Free Active Test-Time Adaptation for Vision-Language Models

**arXiv ID:** 2611.09084 | [PDF](https://arxiv.org/pdf/2611.09084v1)

**作者:** Jihwan Bang `[一作]` (KAIST), Jae-Gil Lee `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Active VLM‑TTA 框架，允许冻结的视觉‑语言模型在部署时主动询问稀疏标签，并将这些延迟得到的标签通过检索式记忆库进行后续样本的纠正，从而实现无训练参数更新的在线自适应。

**💡 创新点**

创新点包括：① 将主动学习与测试时适应结合，标签只能用于未来样本；② 设计了训练无关的 ACTOR 模块，利用一致性与可靠性双阈值限制检索纠正，避免错误传播；③ 在不修改预训练模型参数的前提下，显著提升跨域与 OOD 任务的性能。

**🔧 技术方法**

使用技术：CLIP、ResNet‑50 与 ViT‑B/16 视觉‑语言模型；熵/不确定性筛选；检索式记忆库（最近邻查询）；一致性（Referrer 与 Verifier 同标签）与可靠性（余弦相似度阈值）检查；无梯度参数更新的推理。

**📊 数据集**

实验数据集：10 个跨域基准（Flowers‑102、Caltech‑101、Cars、DTD、EuroSAT、Food‑101、Pets、SUN‑397、UCF‑101、Aircraft）以及基于 ImageNet 的 OOD 基准（ImageNet‑S、ImageNet‑V2 等）。

**📈 对比分析**

方法比较：与无监督 TTA 方法（TPT、C‑TPT、TDA）以及在线 CoOp/Replay‑CoOp 的监督对齐方法进行对比。ACTOR 在 ViT‑B/16 上的跨域平均 Top‑1 提升约 4.5%（68.14%→72.66%），OOV 平均提升约 1.2%（65.22%→66.40%），并在与监督 prompt‑tuning 对比时领先 2–3pp。推理额外开销 <1 ms，几乎不影响实时性能。

**⚠️ 局限性**

限制：对标签质量敏感，噪声标签会导致错误纠正；在类别分布极不均衡或标签预算耗尽后只能使用 Top‑K 可靠性检查，可能削弱性能；延迟标签较长时，记忆库的有效性随时间衰减，需要更复杂的记忆管理与噪声鲁棒机制。

---

## 267. Large Language Model Deployment on Resource-Constrained Edge Devices: A Practitioners Survey

**arXiv ID:** 2611.09075 | [PDF](https://arxiv.org/pdf/2611.09075v1)

**作者:** Rahul Maliakkal `[一作]` (Meta), Ankur Sadhoo `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并整合了适用于资源受限边缘设备的LLM部署技术，包括高效微调、压缩、推理优化与硬件运行时策略。

**💡 创新点**

提出统一的软硬件协同部署框架和实用的部署指南，同时概括了多种先进的压缩和微调方法如LoRA、QLoRA、SparseGPT、AWQ等。

**🔧 技术方法**

PEFT（LoRA/QLoRA/DoRA）、量化（PTQ/GPTQ/SmoothQuant/AWQ）、剪枝（Structured/Unstructured, SparseGPT, Wanda）、知识蒸馏、KV缓存、Paged Attention、Speculative Decoding、硬件加速器与跨平台运行时（ONNX、Ollama、vLLM、SGLang）等技术。

**📊 数据集**

未提出新数据集，主要引用公开预训练模型和通用任务数据，如 Hugging Face、TensorFlow Hub、ChatGPT 等。

**📈 对比分析**

通过模型大小、内存占用、延迟、能耗和准确率等指标进行定量比较，量化后可将 7B FP16 模型压缩至 4‑bit 约 4 GB 内存，推理延迟下降 2–5×，能耗显著降低。

**⚠️ 局限性**

仍面临硬件异构性导致压缩效果不一、稀疏模型硬件支持不足、在线微调能耗高、缺乏实时自适应压缩和统一的跨平台部署标准。

---

## 268. Emo-LiveSync: Real-Time Universal Lip Synchronization with Explicit Emotion Control

**arXiv ID:** 2611.09074 | [PDF](https://arxiv.org/pdf/2611.09074v1)

**作者:** Sangjoon Yu `[一作]` (Klleon AI Research), Ji-Su Kang `[通讯]` (Klleon AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 warp 的高效通用唇同步视频生成框架，支持实时推理、高质量唇形同步，并加入情感控制功能。

**💡 创新点**

创新点包括：① 通过知识蒸馏将音频映射至预训练视频-运动模型的隐式关键点空间，实现跨模态对齐；② 引入无训练、可插拔的情感控制门；③ 构建多语言、多风格的 UniVox-Bench 评测基准。

**🔧 技术方法**

使用了 LivePortrait warp 结构、Transformer‑LSTM 音频到运动模型、隐式关键点对齐损失（Charbonnier+velocity+Wing）、训练无关情感门以及基于关键点的 warp 解码器。

**📊 数据集**

训练数据集：LRW；评估数据集：HDTF、VoxCeleb2 以及自建的 UniVox-Bench（包含人类与非人类、七种语言）。

**📈 对比分析**

与多种 GAN、Diffusion、OmniSync 等最先进方法在 HDTF、VoxCeleb2 和 UniVox-Bench 上进行定量（PSNR、LPIPS、FID、FVD、AVS、LMD、CSIM）和定性对比，显示在视觉质量、同步准确性和实时推理方面均优于现有方案。

**⚠️ 局限性**

局限性：依赖预训练的 LivePortrait 关键点模型，在极端表情或复杂背景下可能表现不足；情感控制门需人工标注情绪偏移，且对非人类表情细腻度的提升空间仍大。

---

## 269. Real-Time Short-Response Selection for Spoken Dialogue Systems

**arXiv ID:** 2611.09057 | [PDF](https://arxiv.org/pdf/2611.09057v1)

**作者:** Hien Ohnaka `[一作]` (University of Osaka), Koichiro Yoshino `[通讯]` (University of Osaka)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种实时短回应选择任务，能在口语对话中在主要回应前快速选出合适的填充语或回声以减少生成延迟；

**💡 创新点**

创新点在于将对话上下文与预先生成的短回应进行对比学习关联，并通过流式VAP模型的特征实现低延迟、与ASR/LLM/TTS无关的选择；

**🔧 技术方法**

使用对比学习（InfoNCE）、流式Transformer、VAP声学模型、HuBERT特征、三任务联合损失（时序预测、事件分类、短回应选择）；

**📊 数据集**

在日语日常会话语料库Corpus of Everyday Japanese Conversation (CEJC) 上进行训练与评估；

**📈 对比分析**

与随机选择和基于Gemma 4 E2B LLM的选择方法比较，Top‑1文本准确率和Top‑k%语音排名均显著提升；时序预测与事件分类性能基本不变，延迟仅增加0.076 ms（10k候选）；

**⚠️ 局限性**

存在多对一映射导致缺乏上下文一致性、选择与后续LLM生成内容不一致以及训练与推理语音域不匹配的问题；

---

## 270. LeanDojo-v2: A Comprehensive Library for AI-Assisted Theorem Proving in Lean

**arXiv ID:** 2611.09048 | [PDF](https://arxiv.org/pdf/2611.09048v1)

**作者:** Ryan Hsiang `[一作]` (National Taiwan University), Anima Anandkumar `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个统一的 LeanDojo‑v2 框架和 Lean4Code 轻量级 IDE，实现 LLM 微调、推理、证明搜索等流程的全链路集成，方便数学家直接使用 AI 辅助证明。

**💡 创新点**

创新点在于把多款分散工具（LeanDojo、LeanAgent、ReProver 等）整合成统一 API；支持渐进式训练、LoRA、RL 等多种微调方式；通过 HuggingFace API 直接推理大模型；将 Pantograph REPL 与 DFS/MCTS 搜索结合；以及提供专属 Lean4Code IDE 进行一键交互。

**🔧 技术方法**

主要技术包括 Lean theorem prover、LSTM/Transformer LLMs（如 DeepSeek‑Prover‑V2‑7B）、HuggingFace SFT、LoRA、GRPO/PPO 强化学习、Pantograph REPL、DFS/MCTS 搜索、VSCodium + VSCode Lean4 扩展、定制插件。

**📊 数据集**

使用从 Lean 仓库（GitHub 或本地路径）提取的 AST、依赖、定理、证明与前提数据；按证明步骤长度将定理分为 easy/medium/hard 三级；并利用多份公开 Lean 代码库（如官方 NTP、Lean4 example 等）作为训练集。

**📈 对比分析**

论文没有给出具体数值评估，但声称 LeanDojo‑v2 通过统一 API 缩短构建时间（从十分钟降至几秒），支持通过 HuggingFace API 直接使用 7B 级模型，省去本地 C++ 推理库；与传统分离工具相比，提升了易用性和可复现性。

**⚠️ 局限性**

局限性包括对 Lean 版本的兼容性依赖、只能通过 HuggingFace API 调用大型模型、对新模型和搜索算法的支持有限、需要持续更新 Lean4Code 的内置 Lean4 扩展，以及尚未充分验证不同领域的泛化效果。

---

## 271. When to Trust the BIM: Identifiability-Aware LiDAR-Inertial--BIM Fusion under As-Built Deviations

**arXiv ID:** 2611.09045 | [PDF](https://arxiv.org/pdf/2611.09045v1)

**作者:** Yujie Zhang `[一作]` (Hebei University of Technology), Jinyue Liu `[通讯]` (Hebei University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于BIM的LiDAR‑惯性里程计融合方法，能够在扫描与BIM不一致时判定错误是来自姿态偏差、墙面偏移还是整个房间平移，并相应推迟或纠正BIM锚定。

**💡 创新点**

核心创新是：①用几何可识别性判据确定何时能唯一归因冲突；②在冲突不可区分时保留所有候选解释并推迟BIM锚定；③结合单假设线性化和统计检验，实现低误报且高覆盖率的实时定位。

**🔧 技术方法**

使用几何投影子空间、伪逆与最小二乘拟合、χ²检验、置信区间与累积距离阈值；并在窗口内对每个解释（无误差、墙面、房间、姿态）分别进行拟合和互相排除。

**📊 数据集**

实验数据包括：①二维仿真场景（走廊、角落、房间、带柱子房间）；②实验室房间209配合IFC模型；③SLABIM建筑物三层的真实LiDAR‑惯性轨迹与BIM模型。

**📈 对比分析**

与diS-Graphs‑style和BIM‑Loc‑style两套基线对比，检测到可检测但不可归因时条件错误率降至0.83%（vs 31.67%），在5,600个可归因案例中给出5,574次唯一归因且无误报；在实验室和SLABIM 1,302次注入实验中无任何误报或离目标归因，且在定位精度和可信区间覆盖率上优于基线。

**⚠️ 局限性**

局限性：仅处理单一故障假设（墙面、房间或姿态），无法同时解释多重误差；墙面偏移上限为0.35 m，超出时无法归因；仅使用平面墙面，忽略楼层、倾斜面等；依赖固定阈值的χ²检验与先验噪声模型，实地环境中的非线性与关联误差可能影响判据；当缺乏足够见证时需等待新参考或退回未验证状态。

---

## 272. Function-Space Certification for Neural Operators

**arXiv ID:** 2611.09038 | [PDF](https://arxiv.org/pdf/2611.09038v1)

**作者:** Ryan Hsiang `[一作]` (National Taiwan University), Anima Anandkumar `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种函数空间的神经算子认证框架，实现了对连续域内完整函数输出的严格上界与下界推导；

**💡 创新点**

创新点在于将传统的间隔界限传播（IBP）和CROWN方法推广到函数空间，并为谱卷积引入严格积分误差控制，能够在连续域内对神经算子进行鲁棒性、PDE残差和解误差的全域保证；

**🔧 技术方法**

核心技术包括函数空间IBP/CROWN、Gauss–Legendre/Clenshaw–Curtis 逼近、对高阶导数的界限传播以及利用PDE稳定性分析将残差转化为解误差上界；

**📊 数据集**

实验使用了多种数据集：1D/2D Poisson、Helmholtz、Burgers 方程，Kuramoto–Sivashinsky 复杂动力学，以及一款572.5M参数的球面傅里叶神经算子（SFNO）在ERA5气候数据上的全球天气预测；

**📈 对比分析**

与传统网格级IBP/CROWN对比，函数空间方法在大多数任务上获得相同或更紧的全域界限，计算时间仅需数小时，内存峰值约150 GB；

**⚠️ 局限性**

局限性包括：需输入连续可微函数；深度网络导致界限指数级扩张，难以为8层SFNO或多步回放的KS方程提供鲁棒性保证；以及将残差上界转化为解误差时的保守性，导致在高度非线性或混沌系统中的误差估计过大。

---

## 273. Reconfiguration of Time-Bounded Descriptions

**arXiv ID:** 2611.09020 | [PDF](https://arxiv.org/pdf/2611.09020v1)

**作者:** Dimitrios Myrisiotis `[一作]` `[通讯]` (Great Bay University), Dimitrios Myrisiotis (Great Bay University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在时间限制下的 Kolmogorov 复杂度模型中，给定长度 N 和截止时间 t 的程序集合（即一个“fiber”）的图结构：程序之间仅相差一位时视为相邻，讨论该图的连通性、连通分量数以及判定连通性的复杂度。作者证明：对任意附加普遍机，几乎所有字符串在长度 m+O(1) 时的 fiber 已不连通；在长度 2m+3 时的连通分量数至少为 2^m；并构造了一个特殊机，使得所有 fiber 在足够大长度（至少指数级）时要么为空，要么连通。作者进一步证明，判断两程序是否在同一连通分量中是 PSPACE‑complete 的，但对固定目标字符串可在多项式时间内完成。

**💡 创新点**

创新点包括：①首次将时间受限 Kolmogorov 复杂度下的程序空间图（fiber）连通性问题引入理论；②给出通用机族在不同长度下的连通性下限与上限，并展示了存在的特殊机能在指数阈值后实现全局连通；③证明连通性判定问题在 PSPACE 复杂度类中是完备的，并讨论了固定目标时的多项式可解性；④提出了 G(V) 集合的结构与稠密性问题，并给出其可数性、可计算密度为零的性质。

**🔧 技术方法**

核心技术：
- 通过强制性的“必备描述”与码长分析，计数图中连通路径和分量数；
- 使用有限域检查码、Reed‑Muller 评价码以及 coupon‑coloring 等编码工具构造高距离终端点集合；
- 构造“连接核心”与“路由器”程序，利用颜色类连通性和固定调度确保任何程序在阈值后与核心相连；
- 采用 Savitch 递归算法证明连通性判定可在 PSPACE 内完成；
- 运用可计算密度、有效免疫等理论讨论 G(V) 的集合属性。

**📊 数据集**

本文为纯理论研究，未使用实验数据集；所有结果均在抽象的图和机器模型上证明。

**📈 对比分析**

比较与性能：
- 对任意附加普遍机，几乎所有字符串在长度 2m+3 时的 fiber 至少拥有 2^m 分量；
- 构造机的连通阈值至少指数级，且连通判定 PSPACE‑complete；
- 对固定目标，连通判定在多项式时间内可完成；
- 通过上述定理展示了全局连通目标集 G(V) 的稠密度为零。

**⚠️ 局限性**

限制与未解问题：
- 未确定所有附加普遍机的全连通性判定问题是否总是 PSPACE‑complete；
- 是否存在全局连通目标集合 G(V) 为无限集仍保持可计算性与密度性质尚未解决；
- 对编辑距离图（允许插入/删除）的连通性分析仍是开放问题。

---

## 274. Frontier3DBench: Benchmarking Coding Agents for 3D Creation across Objects, Scenes, and Interactive Open Worlds

**arXiv ID:** 2611.09017 | [PDF](https://arxiv.org/pdf/2611.09017v1)

**作者:** Xinjie Lin `[一作]` (Tsinghua University), Huan-ang Gao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的3D创作基准Frontier3DBench，包含145个跨对象、场景和交互开放世界的任务，并对编码代理在外观、结构和行为三个维度上进行评估。

**💡 创新点**

创新点在于：①构建统一多层次基准，覆盖生成、重建和编辑；②结合可执行Blender和Unreal Engine项目进行多维度自动化评测；③揭示编码代理在不同构建层级的强弱点及瓶颈。

**🔧 技术方法**

采用大语言模型驱动的编码代理（GPT‑6系列、Claude Opus、Kimi、MiniMax、DeepSeek、GLM、Doubao、Qwen），配合各自的开发者API harness；评测采用自动视觉、结构测量与运行时行为测试。

**📊 数据集**

使用公开的3D模型、公共数据集图像、作者采集的照片、游戏截图以及全景图像等多种视觉参考作为任务输入与评测依据。

**📈 对比分析**

通过对11个模型‑Harness配置进行统一评测，计算对象、场景、开放世界的平均分并按30/30/40权重合成总分；GPT‑6.1‑sol在总分（66.02/100）上最高，显示对象与场景表现最强；但所有模型在参考重建和开放世界协调任务上的得分普遍低于文本驱动生成，揭示显著性能差距。

**⚠️ 局限性**

局限性包括：①对指定几何与布局的匹配度仍不足；②局部交互虽强，但多步骤协调与持久状态管理表现欠佳；③成本与性能不完全成正比，部分高性能模型成本高昂。

---

## 275. KnowEvo: Knowledge Evolution for Protein Optimization

**arXiv ID:** 2611.09013 | [PDF](https://arxiv.org/pdf/2611.09013v1)

**作者:** Zijie Xing `[一作]` (Dalian University of Technology), Yanming Shen `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对蛋白质优化任务构建可执行知识块体系，并通过知识进化框架（KnowEvo）在可执行专家块上进行迭代改进。

**💡 创新点**

将外部可执行知识本身视为优化目标，采用可执行知识块、策略变异树、Thompson采样前向评估和后向沙盒式程序修订，实现知识的系统累积、归因和可重用。

**🔧 技术方法**

使用可执行专家块、策略变异树、Thompson样本分配、后向沙盒（agentic code revision）以及多轮评估与修订循环。

**📊 数据集**

在BRENDA来源的蛋白质稳定性（Rosetta REF2015）和可溶性（SaProt）两个数据集上进行实验。

**📈 对比分析**

与根块、单体程序演化、AdaEvolve、GEPA等基线在pass@k、MRR、MAP、NDCG@5等指标上进行对比，KnowEvo在所有指标均显著提升。

**⚠️ 局限性**

对可执行程序表示的依赖使得需要高质量代码代理与沙盒；对更大规模蛋白或更复杂任务的可扩展性尚未验证；缺乏对不同后端预测器泛化能力的全面评估。

---

## 276. EventHOI: Event-Driven Resource-Efficient Hand-Object Interaction Logging on Smartglasses

**arXiv ID:** 2611.09089 | [PDF](https://arxiv.org/pdf/2611.09089v1)

**作者:** Le Zhang `[一作]` (University of Michigan), Ke Sun `[通讯]` (University of Michigan)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于低功耗事件感知的智能眼镜HOI（手物交互）记录系统，利用事件相机和IMU在极低能耗下触发RGB摄像头并进行空间压缩，实现持续的头戴式HOI记录；

**💡 创新点**

创新点在于：①利用事件帧与IMU协同实现自适应的全局运动（egomotion）抑制，精准提取HOI运动；②设计轻量级的HOI检测与感兴趣区域（RoI）生成，驱动事件驱动的RGB采样与压缩；③在资源受限硬件上完成端到端采样、压缩与BLE上行，显著降低能耗和带宽；

**🔧 技术方法**

技术包括：低功耗事件相机与单轴IMU、两阶段egomotion取消（速度阈值与基于陀螺仪的旋转场）和LK光流融合、阈值化稀疏运动检测、时间膨胀的RoI聚合、基于VLM的HOI识别与检索；

**📊 数据集**

评估使用了Nymeria大规模Egocentric HOI数据集（50小时、97人）以及自建的20小时真实场景数据（10人），并通过仿真事件帧生成验证仿真-真实差距；

**📈 对比分析**

与均匀采样、YOLO手检测oracle、Hands23等基线对比；在相同有效帧率下提升约9%HOI识别准确率；在相同BLE带宽下提升12%；实现RGB帧率下降2.4×、BLE带宽下降2.9×，仅有1-2%精度损失；

**⚠️ 局限性**

局限包括：①对背景运动（如电视、手机动态内容）和重复纹理的误触发；②事件光流对细小纹理的敏感导致误判；③依赖单一事件相机/IMU，若传感器失效或误差累积可能影响egomotion取消；

---

## 277. UniRGBA: Unifying RGB and Alpha in Pretrained Video Diffusion Models

**arXiv ID:** 2611.09069 | [PDF](https://arxiv.org/pdf/2611.09069v1)

**作者:** Lei Xu `[一作]` (ByteDance), Aoyu Wang `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一框架UniRGBA，使预训练的RGB视频扩散模型天然支持RGBA视频生成；

**💡 创新点**

通过零初始化的通道扩展、对称的少量层解冻、KL参考正则化以及两阶段GAN训练，保持原RGB潜在空间，避免灾难性遗忘；

**🔧 技术方法**

使用改进的VAE（UniRGBA‑VAE）、DoRA（参数高效适配器）在DiT上进行文本到视频（T2V）和图像到视频（I2V）生成；

**📊 数据集**

在内部收集的视觉特效RGBA视频数据集（约17K高质量样本）以及WebVid RGB数据集做对比实验；

**📈 对比分析**

与Wan‑Alpha和TransPixeler比较，UniRGBA‑VAE在RGBA重建上PSNR、SSIM、LPIPS均显著提升；在RGBA T2V上在文本对齐、自然度、运动平滑等指标超过Wan‑Alpha，在RGB生成上保持甚至优于原Wan2.1；

**⚠️ 局限性**

对纯RGB数据的细微PSNR下降、I2V评估仍依赖可视化、未在公开大规模RGBA基准上验证，且对不同视频长度和分辨率的泛化尚待进一步研究。

---

## 278. Does Reasoning Improve Psychological Depth in Large Language Models?

**arXiv ID:** 2611.09061 | [PDF](https://arxiv.org/pdf/2611.09061v1)

**作者:** Ruichen Zheng `[一作]` (University of California Los Angeles), Nanyun Peng `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文评估了在短篇小说创作中推理增强大模型（GPT‑5、DeepSeek‑R1）相对于基线模型（GPT‑4o、DeepSeek‑V3）的心理深度表现，采用了7名人类评审和LLM‑as‑a‑Judge集成器进行双盲成对评估。

**💡 创新点**

创新点在于：①在前沿生成任务中引入心理深度量表（PDS）进行主观质量评估；②揭示人类评审对心理深度维度加权的多样性；③发现LLM‑as‑a‑Judge在此任务中表现出强烈的推理偏好，压缩了人类差异。

**🔧 技术方法**

使用技术包括：PDS 5维度评估框架、交叉随机截距GLMM、逻辑回归权重分析、DSPy+Mixture‑of‑Personas LLM‑as‑a‑Judge（Llama 3.1/3.3、Qwen3.5）以及嵌套式模型集成。

**📊 数据集**

数据集为 PsychDepth Reasoning：20个提示生成的60对（共120篇）短篇小说，配有7名评审的双盲对比评分和LLM‑judge的标量评分。

**📈 对比分析**

比较方法为人类对比评分与LLM‑judge对比评分，结果显示人类对推理模型的整体偏好为55.2%，而LLM‑judge的偏好高达98.3%；人类评审间一致性极低（Krippendorff α≈0.07），而LLM‑judge几乎统一推理模型。

**⚠️ 局限性**

局限性包括：样本量小（仅7名评审、60对），模型对比非纯净因果实验，评审群体单一，任务局限于短篇小说，LLM‑judge在域迁移上可能存在偏差，且对低信号前沿任务的可靠性不确定。

---

## 279. ProtoCouncil: Multi-Encoder Anatomical Prototypes for Prostate Cancer Detection in Micro-Ultrasound

**arXiv ID:** 2611.09058 | [PDF](https://arxiv.org/pdf/2611.09058v1)

**作者:** Mohammad Mahdi Abootorabi `[一作]` (University of British Columbia), Purang Abolmaesumi `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了一种多编码器原型委员会模型，用解剖学原型在微超声图像上实现前列腺癌检测。

**💡 创新点**

创新点在于将原型映射到真实训练区域、结合三种预训练模型的解剖学原型，并通过编码器不一致性实现可解释和可拒绝预测。

**🔧 技术方法**

采用MedSAM、DINOv3和V-JEPA 2.1三种预训练模型，使用原型网络、投影、加权融合与不确定性门控。

**📊 数据集**

使用ExactVu系统的三组临床试验数据：A（693名患者、6588核心）、B（111名患者、1460核心）和C（42名患者、558核心）。

**📈 对比分析**

与多种基线（MedSAM-UNETR、MicroSegNet、ProstNFound+、DINO-FT等）进行五折交叉验证和零射击外部测试，ProtoCouncil在csPCa AUROC上分别超过80%（A:82.9%，B:85.2%，C:84.6%），优于先前方法。

**⚠️ 局限性**

局限在于对小样本高分组别表现不佳、依赖手工解剖标注、对跨中心差异仍有限，并且缺乏临床实用性验证。

---

## 280. From Emergence to Reversibility: Toward Reversible Multi-Agent Systems

**arXiv ID:** 2611.09054 | [PDF](https://arxiv.org/pdf/2611.09054v1)

**作者:** Zhengye Han `[一作]` (New York University), Linda Ng Boyle `[通讯]` (New York University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出可逆多智能体系统的概念，研究如何在集体能力不断提升的过程中保持授权干预的可行性与有效性。

**💡 创新点**

创新点在于：①将干预视为机构依赖的可逆性问题；②定义风险‑服务二维平面上的可撤销前沿；③提出干预迁移、修复与复原的系统化设计方法。

**🔧 技术方法**

主要技术包括：基于机构模型的可逆性理论、风险与服务的阈值约束、Pareto 边界分析、干预记录与验证机制。

**📊 数据集**

本文未使用具体数据集，而是以物流多智能体协作为示例进行概念说明和理论构建。

**📈 对比分析**

方法评估通过理论构造的“可撤销前沿”与“干预记录”来判定不同成长路径下干预是否可行；未给出数值实验，建议后续以独立开发的工作流进行对比实验。

**⚠️ 局限性**

局限性包括：缺乏实证验证、对干预记录的完整性与监控依赖较高、未讨论大规模真实系统的实现细节以及跨组织协作时的权限归属问题。

---

## 281. Grounding-Aware Judge Reward Reinforcement Learning for Out-of-Domain Face Anti-Spoofing

**arXiv ID:** 2611.09047 | [PDF](https://arxiv.org/pdf/2611.09047v1)

**作者:** Xiaoyong Yu `[一作]` (Huazhong University of Science and Technology), Xinge Yu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于图像证据的判别者奖励，用于在多模态大语言模型（MLLM）上进行强化学习，以提升人脸伪造检测的跨域泛化能力。

**💡 创新点**

创新点在于构建了三维评分判别者（证据锚定、逻辑连贯性、域知识），将图像本身作为验证信号，弥补了仅凭二元判决难以监督推理过程的空白；并首次将LLM-as-judge应用于FAS任务。

**🔧 技术方法**

技术包括：Chain-of-Thought（CoT）监督微调、Group Relative Policy Optimization（GRPO）强化学习、32B视觉语言判别者、低秩适配（LoRA）以及序列级重要性采样。

**📊 数据集**

使用的公开数据集有CelebA‑Spoof（ID基准）、LCC‑FASD和MSU‑MFSD（OOD基准）。

**📈 对比分析**

与传统CNN（ResNet‑50/101）和CLIP‑ViT‑B/16基线相比，CoT SFT在ID上取得0.033 ACER（比CLIP低3.8倍），在OOD上略低；加入判别者奖励后，ID ACER仅升至0.035，OOD ACER显著下降（LCC‑FASD从0.288降至0.273，MSU‑MFSD从0.431降至0.409），与CLIP基线相当或更优。

**⚠️ 局限性**

局限在于OOD提升幅度有限（仅约1–2% ACER下降），受判别者在强分布偏移下的锚定能力限制；强化学习阶段对超参和随机种子敏感，表现波动不大但不稳定。

---

## 282. "Despite her age, Margaret was eager to make a change": Unpacking Aging Representations and Stereotypes in LLM-generated Stories

**arXiv ID:** 2611.09043 | [PDF](https://arxiv.org/pdf/2611.09043v1)

**作者:** MD. Roqunuzzaman Sojib `[一作]` (Bangladesh University of Engineering and Technology), Novia Nurain `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 594 条 LLM 生成的 500 字故事进行系统分析，探讨年龄刻板印象在完整叙事中的表现与机制。

**💡 创新点**

首次结合文本相似度、LLM 辅助编码与定性主题分析，揭示 LLM 默认偏向年轻人、老年人常被描绘为受助者且正面叙事仍嵌入年龄歧视的隐性模式。

**🔧 技术方法**

使用文本嵌入余弦相似度、Claude Opus 5.5 进行结构化编码、Taguette 主题分析，并运用 Wilcoxon、Bootstrap 等统计检验。

**📊 数据集**

594 条由 Gemini 3 Flash、GPT‑5 与 Llama 3.1 生成的故事，覆盖 22 个日常生活因素，按“年轻人”“老年人”“中性”三种年龄条件生成。

**📈 对比分析**

通过嵌入相似度评估中性故事倾向，LLM 编码计算主角年龄、决策者与助力者比例，并用统计检验比较三种年龄条件；结果显示中性故事偏向年轻人，老年人常被描绘为受助者，正面叙事仍含年龄歧视。

**⚠️ 局限性**

仅限英语、单一提示模板、三种模型、缺乏文化多样性，嵌入与编码模型可能带偏见，未深入交叉身份或读者解读，实验规模有限。

---

## 283. DobotTrack: Target-Grounded Vision-Language-Action Learning for Embodied Person Tracking

**arXiv ID:** 2611.09041 | [PDF](https://arxiv.org/pdf/2611.09041v1)

**作者:** Shengjun Deng `[一作]` (Foshan University), Teng Wang `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了DobotsTrack框架，结合目标定位、分割与视觉‑语言‑动作学习，实现语言引导的具身人跟踪。

**💡 创新点**

系统评估多种辅助任务并采用掩膜+框框双监督；引入基于目标条件的Q‑identity视觉历史压缩；构建真实四足机器人收集的数据集并实现自动标注。

**🔧 技术方法**

使用InternVL3视觉‑语言骨干、SAM2语义分割、Q‑identity查询压缩、Transformer语言模型及多任务损失、注意力机制等技术。

**📊 数据集**

使用EVT‑Bench（模拟）、TPT‑Bench以及新收集的四足机器人真实跟踪数据（126集、596k帧掩膜/框/ VSC标注）。

**📈 对比分析**

在EVT‑Bench STT、DT、AT 三个设置下与OmTrackVLA等基线比较，平均成功率达77.9%+，STT 90.2%/DT 72.5%/AT 70.9%；Q‑identity 压缩相较池化提升6.3%/8.6%，推理延迟仅增加2.4 ms。

**⚠️ 局限性**

依赖缓存目标定位质量、有限观测窗口，难以处理长时间遮挡或错误关联；缺乏真实距离度量；验证场景有限，未覆盖更复杂环境。

---

## 284. Deterministic factorization of constant depth circuits from whitebox PIT

**arXiv ID:** 2611.09035 | [PDF](https://arxiv.org/pdf/2611.09035v1)

**作者:** Somnath Bhattacharjee `[一作]` (University of Toronto), Shubhangi Saraf `[通讯]` (University of Toronto)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出了一种基于白盒或黑盒PIT（多项式身份检验）算法的确定性常数层电路多项式因式分解方法；

**💡 创新点**

创新点在于：①首次实现了从白盒PIT到因式分解的完整归约；②利用对数导数与幂级数根的线性系统来实现根重组；③给出了在常数层电路下求解基于PIT的矩阵核的高效算法；

**🔧 技术方法**

关键技术包括：幂级数根的Hensel提升、对数导数判据、构造对应线性系统、利用PIT搜索到点、白盒PIT的决策→搜索转换、矩阵核的逐步剪枝；

**📊 数据集**

本文为理论研究，未使用具体数据集；

**📈 对比分析**

与之前基于黑盒PIT的子指数算法相比，本方法实现了多项式（s,d）时间的确定性算法，性能上更为紧凑；

**⚠️ 局限性**

局限性：需要存在高效的白盒PIT算法；若PIT效率不达标，则分解算法仍受限；此外，算法在有理数域上实现，扩展到有限域需要额外假设。

---

## 285. Scout Before You Route: Predicting Task Success and Token Count from a Small LLM Prefill

**arXiv ID:** 2611.09028 | [PDF](https://arxiv.org/pdf/2611.09028v1)

**作者:** Aristides Milios `[一作]` (Mila Quebec AI Institute), Christopher Pal `[通讯]` (Mila Quebec AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用小型冻结LLM（scout）的预填充激活同时预测候选模型的成功概率与输出长度，从而实现按查询成本路由。

**💡 创新点**

提出将成功率与成本预测合并为同一预填充特征的成本探针（cost probe），并在多数据集、混合流量和少量标签场景下显著优于现有成本估计方法。

**🔧 技术方法**

使用线性探针（逻辑回归+岭回归）对Qwen3-4B的预填充激活进行预测，并结合成功率与成本探针，完成多模型路由与混合流量评估。

**📊 数据集**

评估数据集包括LiveCodeBench、Omni-MATH、MMLU-Pro、AIME、APPS、BigCodeBench、CodeContests、SuperGPQA以及BIG-Bench Extra Hard。

**📈 对比分析**

通过与中值长度定价、外部估计器（MixLLM、CARROT、ZeroRouter等）以及完美成本知识进行对比，成本探针在大多数数据集上实现9–33%的成本节省，捕获了大部分头部空间，并在少量标签、模型扩容与混合流量场景下保持领先。

**⚠️ 局限性**

需为每个候选模型提供输出长度标签（尽管仅需数十个），对高精度单模型场景收益有限；难以区分相似价格高效模型；对新数据集的头部空间无法提前预测；依赖定价与提供商行为，可能影响复现性。

---

## 286. Knowledge Distillation from Neural Networks to Differentiable Logic Networks

**arXiv ID:** 2611.09024 | [PDF](https://arxiv.org/pdf/2611.09024v1)

**作者:** Hee Yun Suh `[一作]` (Princeton University), Niraj K. Jha `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将大型多层感知器（MLP）教师模型的知识蒸馏到可解释的可微逻辑网络（DLN）学生模型中，改进其准确率而不改变结构与推理成本。

**💡 创新点**

在DLN两阶段训练流程中嵌入蒸馏目标，实现教师信息传递而不改变网络架构；同时提供多种蒸馏变体（联合超参搜索、网格搜索、单相蒸馏），首次在可解释逻辑网络上验证蒸馏效果。

**🔧 技术方法**

可微逻辑网络、可微化的逻辑门与连接、两阶段训练、知识蒸馏（软标签、温度与权重调节）、超参数优化（Optuna）、网格搜索、标准交叉熵。

**📊 数据集**

20个公开表格分类数据集（如BankChurn、BreastCancer、Cardio、CKD、Cirrhosis、Compas等）。

**📈 对比分析**

与无蒸馏DLN基线以及MLP教师进行比较；大多数数据集蒸馏后准确率提升0–15%，平均排名提升显著；推理成本（门级运算数）基本不变，甚至在某些数据集可进一步降低。

**⚠️ 局限性**

蒸馏效果因数据集差异显著，部分数据集无提升或下降；仍受梯度消失与模型容量上限限制；未探索更适配DLN的蒸馏目标或更合适的教师模型。

---

## 287. Towards Robust Reinforcement Learning Market Making: Lessons from Loss Function Experimentation

**arXiv ID:** 2611.09012 | [PDF](https://arxiv.org/pdf/2611.09012v1)

**作者:** Siddharth Nand `[一作]` (University of British Columbia), Bora Guney `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过强化学习的 actor‑critic 框架训练市场做市策略，并对比了两种损失函数（单纯最大化利润 vs 加入库存与现金惩罚）对风险管理行为的影响；

**💡 创新点**

创新点在于系统性研究了显式的库存与现金惩罚对 RL 市场做市者的风险偏好与长期盈利的塑造效果，并提供了对比实验，首次量化了不同损失函数对策略稳定性与收益的影响；

**🔧 技术方法**

使用了 Actor‑Critic 强化学习模型，三层全连接网络，基于几何布朗运动模拟资产价格的市场环境；

**📊 数据集**

实验数据全部为仿真数据，仅包含单一股票的 GBM 价格轨迹及随机生成的买卖价，未使用真实交易数据；

**📈 对比分析**

通过累计奖励、累计步骤、以及 ask‑bid spread 的分布等指标对比两种策略，结果显示加入惩罚的策略在累计奖励上更高、执行步数更合理、报价差更紧，整体表现更为稳健；

**⚠️ 局限性**

局限性包括：仅使用单一股票的 GBM 模拟，忽略多资产与多市场特性；假设资产回报正态分布且无波动聚类；仅考虑库存与现金两项风险指标，缺乏更复杂的风险度量；未在真实市场数据上验证。

---

## 288. Sharing the Road: LLM-Based Bargaining in Mixed-Autonomy Traffic

**arXiv ID:** 2611.09004 | [PDF](https://arxiv.org/pdf/2611.09004v1)

**作者:** Lingxiao Zhou `[一作]` (Washington State University), Jia Li `[通讯]` (Washington State University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究构建了基于大语言模型的混合自治交通路段分配谈判框架，模拟AV与HV群体在不同密度、目标与领域知识条件下的议价行为。

**💡 创新点**

创新点在于将LLM作为代理实现语言化谈判，探究隐私信息与公平目标如何影响资源配置与协同，并发现公平目标与领域知识对谈判成功率与利用率的异质效应。

**🔧 技术方法**

技术上使用Gemma 3 12B LLM作为决策引擎，并设计私有/公开出价、文本评估以及折扣终局规则的谈判过程。

**📊 数据集**

数据集为仿真生成的45个交通密度状态（ρ̃_AV、ρ̃_HV），共1080次试验。

**📈 对比分析**

对比不同目标/知识组合下的协议率、Pareto效率和轮数，结果显示公平目标显著提升协议率但不一定提升利用率，领域知识则对不同目标的效应不一致。

**⚠️ 局限性**

局限在于仅在群体层面模拟、仅使用单一LLM、缺乏对等价理论的直接验证以及对微观驾驶行为的不足捕捉。

---

## 289. Optimal Probabilistic Answer Set Programming: Complexity and Implementations

**arXiv ID:** 2611.09022 | [PDF](https://arxiv.org/pdf/2611.09022v1)

**作者:** Damiano Azzolini `[一作]` (University of Ferrara), Fabrizio Riguzzi `[通讯]` (University of Ferrara)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种在答案集编程中同时支持概率事实与弱约束的框架（optPASP），并给出了其语义定义和推理任务；

**💡 创新点**

创新点在于将概率答案集编程与弱约束结合，只考虑最优答案集进行推理与最可能解释（MPE）分析，首次把优化结构引入Probabilistic ASP并分析其复杂性；

**🔧 技术方法**

主要技术包括Credal语义、代数模型计数（AMC）与二层AMC（2AMC）、答案集编程与量化（ASP(Q)）实现；

**📊 数据集**

实验使用了三类实际应用数据集：论文分配、病毒营销网络、排班无出现行为；

**📈 对比分析**

与枚举和知识编译实现（aspmc）比较，optPASP在中小规模实例中推理速度优于枚举，而MPE在大多数实例能在几分钟内完成，但在更大规模下仍受限；

**⚠️ 局限性**

局限性主要是求解最优答案集仍属于NP/Σ₂P难度，规模扩展受限；同时对非平凡约束的处理尚未完全优化。

---

## 290. TouchAnnotator: Multi-Source Tactile Supervision with Privileged-View Distillation

**arXiv ID:** 2611.09129 | [PDF](https://arxiv.org/pdf/2611.09129v1)

**作者:** Ao Li `[一作]` (Tsinghua University), Yansong Tang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用单目第一人称视频和手部姿态估计，预测双手与物体的接触区域和相对压力，实现从普通视频到全密集触觉注释的自动化。

**💡 创新点**

创新点在于：① 将来自手套、几何接触、机器人指尖变形等六个来源的触觉信号统一到一个21×21的双手网格，并使用源感知掩码避免错误监督；② 采用特权视角教师‑学生蒸馏（EMA教师观测同步的腕部视角），让学生在推理时仅需第一人称视频即可得到丰富的触觉特征；③ 引入过去clip记忆模块，为学生提供短时序上下文，从而在离线注释中实现时序连贯。

**🔧 技术方法**

使用的技术包括：多源触觉对齐与掩码、双视角EMA教师蒸馏、姿态条件ROI加权、Qwen3.5视频编码器与推理器、双头输出（压力与接触）、门控损失、过去clip记忆以及跨模态对齐与对齐损失。

**📊 数据集**

实验基于六个数据集、三种构件（裸手、手套、机器人手），共计 27,005 段视频、35.0M 帧（约 195.3 小时）。主要包含 EgoTouch、HumanTouch、FPHA、机器人抓取数据集，以及 860 段同步手套录制。

**📈 对比分析**

与 TouchAnything、DECO、HACO 等方法对比：在 EgoTouch 上，TouchAnnotator 在相同学生权重下实现了更高的接触 IoU（+0.026）和体积 IoU（+0.023），在 FPHA 上的 F1 与 Haco 近似；在同步手套录制上，时间精度提升 5.38 点、体积 IoU 提升 5.02 点，F1 提升 4.21 点，整体性能优于 TouchAnything。

**⚠️ 局限性**

限制包括：预测的是相对传感器响应而非绝对力，无法直接量化为真实力值；高度依赖姿态估计，易受模糊、遮挡、延时等影响；推理仅支持离线、非帧级在线处理；跨传感器统一物理量化缺乏，导致不同构件间的数值对齐不足。

---

## 291. Toward the Era of Agentic Embodied Intelligence

**arXiv ID:** 2611.09124 | [PDF](https://arxiv.org/pdf/2611.09124v1)

**作者:** Chao Liang `[一作]` (Shanghai Jiao Tong University), Zhipeng Zhang `[通讯]` (Anyverse Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 A2-MEAS 框架，将任务推理、诊断与连续控制通过共享执行记录耦合，实现在线纠正与自适应动作生成。

**💡 创新点**

创新点在于“代码即缩放接口”(code as a scaling interface)，通过学习的纠正代码将高层指令、空间证据与历史信息注入动作专家的隐藏状态，实现模型固定时的在线纠错；同时将多模态视觉、世界模型、动作专家三者统一成端到端可训练的 VLA（vision‑language‑action）体系。

**🔧 技术方法**

技术核心包括：① 2B Qwen3‑VL 视觉‑语言骨干；② 6帧多视角历史编码器 + 先验世界模型；③ 18层 Transformer 动作专家 + 位置残差适配器；④ 纠正编码器与残差适配器；⑤ 离线采样纠正监督与 Flow‑Matching 训练；⑥ 多任务、跨场景的评估与调度基础设施。

**📊 数据集**

主要数据集：BEHAVIOR 2026（100 任务、1 000 任务实例、约 1 950 h 录制），RoboTwin 2.0（50 任务、Clean 与 Randomized 两种场景），12 个桌面实机任务（20 次/任务、5 个家庭场景），并在这些数据集上进行离线训练与在线评估。

**📈 对比分析**

比较方法：在 BEHAVIOR 公共集上与单模型、复合模型、完整系统三种配置对比；与 RLC、Comet、π_0.5 等公开方法对齐；在 RoboTwin 与实机任务上与 π_0.5、G0.5、MotuBrain 等基线比较。性能：完整系统 Q‑score 0.6411，完整任务成功率 45.4%；单模型 Q‑score 0.4597、成功率 27.2%；复合模型 Q‑score 0.6040、成功率 40.6%；实机宏观成功率 57.9%（π_0.5 26.7%）；RoboTwin 上平均 93.98% 成功率。

**⚠️ 局限性**

局限性：① 对极难任务的成功率仍低（如披萨、面包制作）；② 纠错依赖 2B 语言模型协调器，模型容量受限时性能下降；③ 仅支持有限的重试与预算，无法处理长周期的持续错误；④ 训练与部署成本高，需 8‑GPU 及专门的流水线；⑤ 对未知场景与低质量传感器的鲁棒性仍待提升。

---

## 292. CoMET: Tracking the Shared Pose Error of Roadside Messages for Asynchronous Vehicle-Infrastructure Cooperative 3D Detection

**arXiv ID:** 2611.09121 | [PDF](https://arxiv.org/pdf/2611.09121v1)

**作者:** Shuwei Sun `[一作]` (Xi'an Jiaotong University), Hui Cao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于共模误差跟踪（Common-Mode Error Tracking）的车辆–基础设施协同感知方法，利用消息中共享的平面平移误差并对其进行卡尔曼滤波来校正远程检测框；

**💡 创新点**

创新点在于：①只需少量共视目标即可估计并持续追踪共享误差；②采用无学习参数的单标量卡尔曼滤波器，实时更新误差估计；③通过冻结的轻量级重排序头在融合时提升检测性能；

**🔧 技术方法**

主要技术包括：互最近邻 Hungarian 匹配提取共视锚点，基于观测的共模平移误差估计，因果标量 Kalman 滤波，盒级后处理与重排序；

**📊 数据集**

实验基于公开的 V2X-Seq 真实 V2I 数据集进行，使用其 21 条验证序列进行评估；

**📈 对比分析**

与传统后处理、基于特征的中间融合、以及每帧注册等方法比较，结果显示在严格到达协议和在线位姿条件下，所提方法在所有五种检测器上平均提升 3.4-3.7 AP 点，明显优于中间融合且显著优于单帧注册；当与中间融合网络重新训练时，同样可获得进一步 0.8-1.2 AP 的提升；

**⚠️ 局限性**

局限性：仅假设误差为平面平移且在所有目标上共享；当基础设施或车辆的误差不一致、存在旋转误差或大门限外误差时方法效果受限；零占位符和上限裁剪的参考偏差也会影响评估结果；

---

## 293. Finite-sample Calibration Guarantees for Binary Segmentation

**arXiv ID:** 2611.09120 | [PDF](https://arxiv.org/pdf/2611.09120v1)

**作者:** Bruno Viti `[一作]` (University of Graz), Martin Holler `[通讯]` (University of Graz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对图像分割模型的后处理校准方法，并给出了理论校准误差保证。

**💡 创新点**

创新点在于基于改进的集中不等式，将空间相关性纳入校准误差分析，并构造出既满足边际又满足条件校准的分箱校准器。

**🔧 技术方法**

使用了贝特定不等式、耦合矩阵、马尔可夫链等统计学习理论工具，以及自回归模型估计空间相关性。

**📊 数据集**

在九个多样化数据集上评估：ISIC、Kvasir、TN3K、BUSI、Cityscapes、FLAIR1、Oxford-IIIT Pet、COCO_animals、COCO_coco。

**📈 对比分析**

与温度缩放、Logistic 缩放、Meta‑Cal、Selective 缩放、Beta 校准、Isotonic 回归等基线比较，分箱方法在边际校准中表现最好，条件校准下更是显著优于其它方法。

**⚠️ 局限性**

局限在于仅适用于二分类分割、需要经验估计耦合矩阵导致界限可能松散、未覆盖多类别或区域级别的校准，后续需进一步改进。

---

## 294. COMBIO: Automated Discovery of a Compact Composite Biomarker from Routine Laboratory Panels

**arXiv ID:** 2611.09102 | [PDF](https://arxiv.org/pdf/2611.09102v1)

**作者:** Tomer D. Meirman `[一作]` (Ben Gurion University), Noa Dagan `[通讯]` (Ben Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一套自动化流程（COMBIO），将基于常规实验室检查的复杂机器学习模型压缩为一阶乘式的复合生物标记，可直接手工计算。

**💡 创新点**

创新点在于：①利用Tree SHAP对特征进行排序并确定指数符号；②在整数指数（-2~2）空间内全枚举，只保留2~4个变量的单一乘积表达式；③得到的公式既可解释又高度紧凑，避免了传统多项式或树模型的复杂性。

**🔧 技术方法**

使用的技术包括：梯度提升（XGBoost）作为参考模型，Tree SHAP进行特征重要性与方向性评估，离散化整数指数的穷举搜索，以及对得到的公式进行校准（线性回归）和决策曲线评估。

**📊 数据集**

使用的数据集包括：NHANES（铁负荷、5年死亡率、未诊断糖尿病）; Indian Liver Patient Dataset（肝病诊断）；MIMIC‑IV（院内死亡、败血症）；外部验证使用eICU数据库。

**📈 对比分析**

比较方法：与单一实验室测量、已有临床指数（如FIB‑4、NLR、BUN/albumin等）、重新拟合的指数、5变量logistic回归以及全XGBoost模型对比；性能指标为AUROC、AUPR、Brier分数、校准斜率；COMBIO在大多数任务中AUROC与5变量logistic差距≤0.03，常高于单一测量或传统指数，XGBoost在某些任务中略优；外部数据集仍保持较好排序，但校准下降。

**⚠️ 局限性**

局限性：①仅适用于完整病例，缺失值导致适用范围受限；②模型是对数变换后固定整数指数的约束式，无法显著提升预测准确度；③外部校准偏差显著，需本地再校准；④缺乏前瞻性临床验证与实际实施评估；⑤不同任务之间的直接比较受样本量与定义差异影响。

---

## 295. Two-Layer Evolutionary Scheduling for Machine Learning Jobs in Multi-Cluster Environments

**arXiv ID:** 2611.09093 | [PDF](https://arxiv.org/pdf/2611.09093v1)

**作者:** Jingyuan Xu `[一作]` (Southern University of Science and Technology), Xin Yao `[通讯]` (Lingnan University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 EVO-GP，一种两层进化调度框架，用进化算法进行多集群 GPU 资源的作业分配，并通过遗传规划学习可解释的优先级函数来完成局部 GPU 资源分配，实现在线调度。

**💡 创新点**

分层优化思路：全局层用进化搜索实现作业-集群的负载均衡；局部层用 GP 学习的优先级函数实现可解释的资源调度；同时将平均/最大加权排队时间和 GPU 利用率三目标通过加权求和统一优化，支持业务需求可调。

**🔧 技术方法**

多目标加权求和、进化算法（交叉、变异、种群初始化）、遗传规划（树结构、子树交叉、保护除法）、Best‑Fit 资源分配、在线调度与基准比较（Spark、Liquid、Tiresias、Lyra、MCT）。

**📊 数据集**

真实 SenseTime Venus 集群两周作业轨迹（约 9k+ 作业）和 Huawei 生产集群仿真生成的多集群工作负载（21 种作业类型），用于评估多集群与单集群场景。

**📈 对比分析**

与 Spark、Liquid、Tiresias、Lyra、MCT 等基线在多集群和单集群环境下比较，采用加权目标（平均/最大加权排队时间+GPU 利用率）以及 AvgJCT/MaxQT 指标。实验显示 EVO‑GP 在 30 个多集群实例中始终排名第一，平均提升 15‑30%；在单集群场景下对不同权重 ω 的折衷可优于基线；在不同负载压力下保持更小波动，在线调度开销 <7 秒。

**⚠️ 局限性**

仅适用于已知作业时长的假设，GP 学习的优先级函数对完全不同工作负载的迁移性未知；加权求和需要人工设定权重，难以自动处理多目标冲突；在线调度时间随作业数线性增长，在极高并发场景下可能成为瓶颈；对极大规模集群的可扩展性未在实验中验证。

---

## 296. AdaAlloc: Adaptive Visual Token Allocation for Long-Video Question Answering

**arXiv ID:** 2611.09092 | [PDF](https://arxiv.org/pdf/2611.09092v1)

**作者:** Joungbin An `[一作]` (University of Texas at Austin), Kristen Grauman `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的长视频问答框架AdaAlloc，旨在在固定的视觉令牌预算下进行视觉令牌分配，以提高长视频理解的效果。

**💡 创新点**

创新点在于将长视频推理重新构建为视觉令牌分配问题，提出了一种训练无关的推理框架，能够根据问题的需求动态分配视觉令牌。

**🔧 技术方法**

使用了AdaAlloc框架，该框架结合了概述条件规划、时间基础定位和基于MLLM的记忆精炼技术。

**📊 数据集**

在LVBench、MLVU、VideoMME和LongVideoBench等多个长视频问答基准上进行了评估。

**📈 对比分析**

与预算匹配的均匀采样相比，AdaAlloc在多个基准上提高了最多10.5个百分点，并在与其他帧选择和代理方法的比较中表现优越。

**⚠️ 局限性**

限制在于所选的全局/局部策略可能并非每个视频-问题对的最优选择，且依赖于稀疏概述和检索的时间候选，可能会错过某些证据。

---

## 297. FIND: Frequency Invariance Disentanglement for Test-Time Adaptation in LiDAR 3D Detection

**arXiv ID:** 2611.09170 | [PDF](https://arxiv.org/pdf/2611.09170v1)

**作者:** Dongxiao Li `[一作]` (Peking University), Ling-Yu Duan `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于频域的 LiDAR 3D 目标检测测试时适应框架 FIND，能够在推理阶段自适应地从目标域中提取域不变特征并抑制域特定干扰。

**💡 创新点**

创新点在于：①使用 B‑Spline 拟合实现局部可调的频域滤波器；②双流架构将域不变与域特定分离；③通过自监督一致性、稳定性引导正则化和互补约束实现滤波器与模型的协同优化。

**🔧 技术方法**

关键技术包括图傅里叶变换（GFT）、B‑Spline 频域滤波、双流自监督学习、Coefficient of Variation（CV）稳定性指标、在线统计与正则化。

**📊 数据集**

实验数据集：nuScenes、Waymo、KITTI（原始与 KITTI-C 失真版本）以及 Waymo→KITTI-C 交叉失真场景。

**📈 对比分析**

与现有 TTA 方法（Tent、CoTTA、Reg‑TTA3D、MOS、DUO）进行对比，在 nuScenes→KITTI、Waymo→KITTI 以及 KITTI-C 等场景中均取得显著提升，平均提升约 +2.27 AP_BEV、+3.24 AP_3D，甚至在某些跨域/失真任务中接近全监督性能。

**⚠️ 局限性**

局限性：频域变换与 B‑Spline 拟合在推理时增加计算开销；对点云密度和滤波器初始化（节点向量、控制点）敏感，可能需要手动调参；在极低点云密度或非标准激光雷达配置下效果未充分验证。

---

## 298. Evaluation of a Multi-Link Machine Learning Approach for Rainfall Detection Using Satellite Microwave Links

**arXiv ID:** 2611.09156 | [PDF](https://arxiv.org/pdf/2611.09156v1)

**作者:** Mostafa Ftouni `[一作]` (DITEN, University of Genoa), Christian Gianoglio `[通讯]` (DITEN, University of Genoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究利用卫星微波链路网络结合机器学习进行实时降雨检测，并引入多链路空间信息提升检测效果。

**💡 创新点**

引入多链路多输入模型（RF‑MI、2D‑CNN‑MI）利用邻近链路的空间相关性，改善降雨辨识。

**🔧 技术方法**

使用随机森林与二维卷积神经网络，结合留一日交叉验证、阈值优化与特征工程等技术。

**📊 数据集**

使用2021‑2024年16个SRS Ku波段卫星链路的信号数据，并以60 km外雷达的降雨强度作为标签。

**📈 对比分析**

通过召回率、特异性及其调和均值对模型进行比较，RF‑MI在所有链路上均优于2D‑CNN和单链路方案，平均HM提升约4‑8%，尤其在中强降雨时表现突出。

**⚠️ 局限性**

数据量有限、空间范围仅覆盖意大利热那亚地区，缺乏对不同卫星系统或更稠密链路网络的泛化验证。

---

## 299. BMMS: Policy-Conditioned Measurement of Behavioral Compliance in Language Models

**arXiv ID:** 2611.09147 | [PDF](https://arxiv.org/pdf/2611.09147v1)

**作者:** Anthony Rhodes `[一作]` (Confidential Core AI), Gadi Singer `[通讯]` (Confidential Core AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套可在运行时接受任何政策语料的行为评估指标体系 BMMS，自动生成与该政策一致的评估维度并返回合规性概况。

**💡 创新点**

创新点在于：① 以政策为输入动态产生评估维度；② 通过维度准入测试（A1–A5）确保所测维度既与政策无关也对评估有意义；③ 报告合规性为概况而非单一分数，并引入 CAS、CRM、BIV、BDI 等多维度指标；④ 提供判别者重评分与判决属性的操作特性。

**🔧 技术方法**

技术手段包括：结构化问题生成与哈希冻结、三方判决面板、准入过滤器、Severity 加权、Bootstrap 置信区间、判决重评分与偏差分析、指标计算公式与归一化、BIV 的特征提取。

**📊 数据集**

数据集为 11 条公开文档（供应商规范、监管条例、AI 透明法、可接受使用政策等）共 1,003 条义务，经过准入筛选后留下 401 条可测义务；使用的测试样本来自三大提供商的模型，包含 GPT‑4o、Claude 系列、Gemini、Llama 等。

**📈 对比分析**

方法与性能：先用人工专家对 60 条样本做准入判定，准入精度在 94.4%（供应商规范）到 50%（分析指导）不等；对 401 条义务分别在 4 个判决面板下评估，得到 CAS、CRM、BIV、BDI 等指标；比较发现 GPT‑4o 在其自身规范上违约率 70.7%，Claude 系列为 0%/9.1%；指标对不同模型、不同压力水平表现出显著区分，且模型年龄和文档难度影响明显。

**⚠️ 局限性**

限制：① 仅用一名专家进行准入与提取验证，缺乏多标注者一致性评估；② 准入精度随文档类型下降，导致部分义务被忽略；③ 提取过程中删去限定词可能导致误判；④ BIV 与 BDI 仅作为候选维度，未完整测量；⑤ 判决者差异显著影响绝对 CAS 水平，需进一步稳定判决流程。

---

## 300. CARGO: Causal Attribution and Responsibility-Guided Preventive Safe Multi-Agent Reinforcement Learning for Cooperative Driving

**arXiv ID:** 2611.09139 | [PDF](https://arxiv.org/pdf/2611.09139v1)

**作者:** Jingjia Teng `[一作]` (Hunan University), Wenjie Huang `[通讯]` (Hunan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了CARGO框架，用于在混合交通中通过多智能体强化学习实现合作式上匝道合并；

**💡 创新点**

创新点包括：共享潜在上下文后验的不确定性驱动交互图构造、结合因果贡献与责任的奖励塑造、责任引导的预防性安全约束与投影更新以及联合可行性验证；

**🔧 技术方法**

使用的技术包括：潜在变量推断、KL 驱动的因果影响估计、因果贡献与责任计算、基于条件扩散模型的 CVaR 风险评估、KL 信任区间与责任约束的投影优化、以及安全成本的代理估计；

**📊 数据集**

实验使用 highway‑env 仿真环境，模拟不同交通密度（低/中/高）的混合车队；

**📈 对比分析**

与 SCIC、Causality Aware、MAACKTR、MAPPO、Safe‑MAA2C、MAPPO‑PIS、MPC 等基线对比，CARGO 在所有密度下取得最高成功率、最低碰撞率，并在高密度情境下成功率提升 66.04%，碰撞率下降 56.23%；

**⚠️ 局限性**

局限性包括：对学习的响应与风险模型准确性高度依赖；长预测时程会累积模型误差；理论安全性仅在局部投影假设下成立；以及在大规模多智能体系统中的可扩展性待进一步研究。

---

## 301. Analyzing Code Provenance in LLMs for Programming Education

**arXiv ID:** 2611.09177 | [PDF](https://arxiv.org/pdf/2611.09177v1)

**作者:** Subhankar Maity `[一作]` (ECE Research Center), Guilherme Medeiros Machado `[通讯]` (ECE Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在编程教育中使用LLM时，通过成员推断攻击追踪代码来源，评估学生提交代码与模型训练数据的相似度，从而实现过程化评估。

**💡 创新点**

创新在于将Min‑K%概率方法扩展到代码数据并加入合成非成员代码生成，以对Python代码进行精准的归因；同时首次将该方法应用于真实学生作业与公开Stack Overflow数据，探讨免费与付费模型在隐私与检测精度上的权衡。

**🔧 技术方法**

使用扩展版Min‑K%成员推断算法、AST距离度量、token概率采样、阈值设定与多重匹配（精确、字符级、token级、Jaccard）等技术，并通过API查询多种LLM（GPT‑2、DistilGPT‑2、DialoGPT、GPT‑5、Gemini‑2.5、Mistral、DeepSeek、LLaMA‑3.3）。

**📊 数据集**

主要数据集包括ProCQA（Stack Overflow的Python答案），合成非成员代码集，以及一组来自高校数据科学黑客松的Jupyter Notebook提交与LLM交互日志。

**📈 对比分析**

通过AUC、TPR@5%FPR、精确率和F1等指标对比不同模型；实验显示免费模型（如DistilGPT‑2）在成员检测上表现最佳（AUC≈0.71），付费模型普遍性能更低但隐私保护更强；在学生作业上的AUC约0.58–0.64，TPR@5%FPR低于9%，说明检测仍具挑战。

**⚠️ 局限性**

局限包括：1）方法只能检测训练数据成员，无法区分AI生成与手工检索的外部代码；2）对细粒度代码编辑后仍能保持较高相似度的情况识别不充分；3）实验规模受API调用限制，未覆盖更大规模或多语言场景；4）阈值设定对不同模型敏感，需要更自适应机制。

---

## 302. SIFR: Stable Signed Flow Reinforcement

**arXiv ID:** 2611.09175 | [PDF](https://arxiv.org/pdf/2611.09175v1)

**作者:** Runlong Liao `[一作]` (NVIDIA), Yogesh Balaji `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在流模型奖励微调中，提出SIFR方法，利用优势函数并对速度漂移加入约束，实现更稳定的训练。

**💡 创新点**

将流RL转化为带速度漂移约束的受限优化，并通过Lagrange乘子自适应正则化，解决传统方法的收敛与稳定性问题。

**🔧 技术方法**

使用优势函数、签名MSE损失、受限优化、Lagrange乘子更新、参考速度场等技术。

**📊 数据集**

Cosmos 3超文本‑图像数据集。

**📈 对比分析**

与DiffusionNFT和RAM基线比较，SIFR在单奖励训练中四倍更快达到基准平台，并在多奖励训练中持续提升，93.5%提示均优于两基线。

**⚠️ 局限性**

对漂移预算δ和Lagrange乘子调参敏感，且仅在Cosmos 3上验证，可能在其他任务或更大模型上收敛性不一定保证。

---

## 303. GreyZone: Production-quality Agentic Application Threats

**arXiv ID:** 2611.09161 | [PDF](https://arxiv.org/pdf/2611.09161v1)

**作者:** Ruoyu Zhou `[一作]` (National University of Singapore), Abhik Roychoudhury `[通讯]` (National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GreyZone基准，用于评估自动化红队服务在真实代理应用中的安全测试。

**💡 创新点**

创新点在于将评估焦点从受攻击的代理转移到红队服务本身，并在真实生产级代理环境中通过可验证的oracle独立评估攻击结果。

**🔧 技术方法**

采用Model Context Protocol统一接口，构建了10个开源代理应用和46个攻击场景，并集成Promptfoo、Dreadnode、PyRIT、DeepTeam四个红队服务进行实验。

**📊 数据集**

使用的公开数据集为10个独立开发的代理应用（如Letta Code、Clawith等）及其配套的攻击场景与安全策略，全部托管于GitHub。

**📈 对比分析**

通过Oracle独立验证攻击成功与否，对Promptfoo在两个案例中的表现进行定性分析，结果显示红队服务能利用应用反馈改进攻击，但未给出客观性能对比。

**⚠️ 局限性**

局限性包括仅使用本地模拟环境、只验证预设的安全违规、案例数量有限以及缺乏跨服务的系统性比较。

---

## 304. Word Subtraction Games

**arXiv ID:** 2611.09157 | [PDF](https://arxiv.org/pdf/2611.09157v1)

**作者:** Yusuke Inoue `[一作]` (Nagoya University), Hiroki Inazu `[通讯]` (Hiroshima University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文对词减游戏（Word Subtraction Games）进行系统性研究，阐述了从子集减去右后缀的组合博弈，并将其结果（赢/输位置）视为正规语言或上下文无关语言，进一步探讨不同子集（有限、正规、上下文无关）的结构特性及其对胜负语言的影响。

**💡 创新点**

创新点在于：①首次将词减游戏与形式语言理论统一；②证明了当减子集为正规语言时，赢位置语言仍为正规语言，并给出了构造 DFA 的方法；③展示了某些上下文无关减子集导致赢位置语言不再为上下文无关语言，突破了以往仅研究有限/正规子集的局限；④通过子集构造与子集自动机结合的“重置”技术，为非确定性游戏状态分析提供新视角。

**🔧 技术方法**

主要技术包括：有限状态自动机（DFA/NFA）与子集构造、正规语言的闭包性质、泵引理（正则与上下文无关）以及 Parikh 映射等；同时利用 Grundy 数的 Moore 机扩展，证明 Grundy 数列也是正规语言。

**📊 数据集**

论文为纯理论研究，没有使用实验数据集。

**📈 对比分析**

由于研究以理论证明为主，未进行实验对比；作者通过构造性证明和泵引理展示方法有效性，讨论了自动机规模、构造复杂度及其对决策复杂度的影响。

**⚠️ 局限性**

主要局限包括：①构造的 DFA 状态数可能随减子集长度呈指数增长；②尚未给出多项式状态数上界的证明；③对非正规或更一般的上下文无关子集的求解仍为开放问题；④对算法实现与性能评估缺乏实验验证。

---

## 305. PathXAI: Evidence-guided structured pathology reporting by transferable tissue segmentation

**arXiv ID:** 2611.09117 | [PDF](https://arxiv.org/pdf/2611.09117v1)

**作者:** Zhaorui Tan `[一作]`, Weimiao Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 PathXAI 系统，结合可迁移的组织分割与视觉语言模型，利用组织证据指导多切片病理报告的生成和重检。

**💡 创新点**

首次将组织分割得到的组织成分和空间信息作为显式证据嵌入 VLM，保持预训练模型权重不变，并实现结构化报告写作。

**🔧 技术方法**

使用 SAM2/MEDSAM3 作为通用组织分割模块，配合 Qwen/MedGemma 等自托管视觉语言模型，并构建证据图、主动重检与结构化写作流程。

**📊 数据集**

训练数据来自多公开组织分割数据集（PathSegmentor、VISTA‑PATH 等），评估数据包括 1,200 份 TCGA 公共病例和 151 份私有病例。

**📈 对比分析**

与仅 H&E、Qwen+H&E、PLIP‑引导 Qwen 等基线对比，公共评估中 PathXAI‑Report 的诊断匹配率和发现 F1 均提升数个百分点，私有集成表现更佳且修正率更低。

**⚠️ 局限性**

局限性在于推理速度受限、未在真实临床流程中验证、分割误差对报告质量的影响尚未充分评估，且对组织标签的可迁移性仍有一定依赖。

---

## 306. MARS: Training Multilingual Search Agents for Retriever-Free Evidence Acquisition

**arXiv ID:** 2611.09090 | [PDF](https://arxiv.org/pdf/2611.09090v1)

**作者:** Rui Qi `[一作]` (Beijing Jiaotong University), Kaiyu Huang `[通讯]` (Beijing Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了无检索器的多语言检索增强生成框架，直接通过可执行搜索命令与多语言语料库交互并利用强化学习控制搜索过程。

**💡 创新点**

创新点包括：①语料库本地实体归纳，用语料自身的标题和别名作为搜索键以解决跨语言表面形式不匹配；②冗余感知搜索奖励，鼓励搜索轨迹简洁、去重；③将上述两者融合到Utility-Aware Retrieval Policy Optimization (URPO) 中，提升多语言下的搜索效率与效果。

**🔧 技术方法**

使用技术包括大语言模型生成可执行命令、可执行搜索执行器、强化学习（GRPO）优化策略、Jaccard相似度用于检测命令重复、语料库本地实体提取及其置信度估计。

**📊 数据集**

主要使用的多语言问答基准MKQA和MLQA进行评估，单语兼容性测试基于HotpotQA。

**📈 对比分析**

与检索基线（Naive RAG、D-RAG、LcRL）及DCI基线（Naive DCI、GrepSeek）进行对比，本文方法在MKQA（seen/unseen）和MLQA上均获得最高EM/fEM，且显著减少搜索命令数、交互回合数、冗余率和响应长度，性能提升具有统计显著性。

**⚠️ 局限性**

局限性在于需要任务特定的训练数据和额外的优化成本，对缺少别名的实体仍易受表面形式变化影响，实验仅覆盖部分语言，未来需验证更广泛的语言通用性。

---

## 307. SPDMX: Large-Scale Open Stem Audio From Symbolic Music

**arXiv ID:** 2611.09140 | [PDF](https://arxiv.org/pdf/2611.09140v1)

**作者:** Phillip Long `[一作]` (University of California, San Diego), Zachary Novack `[通讯]` (University of California, San Diego)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于公共领域的PDMX乐谱，使用开放式合成管线（FluidSynth与MIDI-DDSP）生成了一个可再分发的、带对齐乐谱的多轨音频语料库SPDMX。

**💡 创新点**

其创新点在于提供了规模最大、版权友好、且含有多轨监督的大规模合成音频数据集，填补了音乐源分离与合成研究中对可再分发多轨数据缺乏的空白。

**🔧 技术方法**

技术实现包括对PDMX MIDI的清洗与程序纠正、为不同乐器类别选择并混合多种开源Soundfont和MIDI‑DDSP模型、使用听觉实验确定最佳渲染配方、统一响度标准化和线性混音。

**📊 数据集**

主要数据来源是PDMX（公共领域MIDI乐谱），并借助FluidSynth与MIDI‑DDSP等开源合成器进行音频渲染。

**📈 对比分析**

通过与Slakh2100等现有合成多轨数据集比较，SPDMX在多轨歌曲时长上实现了2,734小时（超出Slakh2100的145小时），同时进行的听觉评估显示其渲染质量在多种乐器类别中优于或与现有方案相当。

**⚠️ 局限性**

局限性包括：约65%歌曲仅为单轨，钢琴占比过高；所有音轨均为单声道；MIDI音符时序与力度已被量化；MIDI‑DDSP音频在上采样后仍受限于16kHz的频带；整体真实性未通过大规模真实案例验证。

---

## 308. Deep Learning Theory through the Lens of Statistical Physics: A Topical Review

**arXiv ID:** 2611.09138 | [PDF](https://arxiv.org/pdf/2611.09138v1)

**作者:** Wei Huang `[一作]` (RIKEN Center for Advanced Intelligence Project), Youjin Deng `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述并统一了统计物理视角下的深度学习理论，系统梳理了从能量景观、谱结构到无限宽极限、随机梯度动力学等三大理论支柱，并阐明它们在解释现代网络学习、泛化与表示学习中的相互关系。

**💡 创新点**

创新点在于提出将统计物理的相变、自由能、平均场与动力学平均场等概念与深度学习关键现象（如过参数化、双下降、平坦/尖锐极小点、梯度噪声的非平衡特性）相联系，构建一个层次化、可解释的物理框架；同时强调各理论的有效性范围与互补性，推动从理想化极限向实际有限宽、非平衡训练迁移的思路。

**🔧 技术方法**

采用了平均场理论、动力学平均场理论、随机矩阵理论、NNGP/NTK 近似、Fokker‑Planck 与 Langevin 动力学、自由能与熵产生分析、以及传播无序性（propagation of chaos）等统计物理工具；在具体实验对比中引用了对抗式优化、SAM、SAM+等平坦性相关方法。

**📊 数据集**

作为综述文章，未直接使用数据集；引用的实验多基于公开数据集如 ImageNet、CIFAR‑10/100、MNIST、文本语料（Wikitext、OpenWebText）等，主要用于展示理论与实验的一致性。

**📈 对比分析**

通过理论预测与已有实验结果的对照，评估了各方法在解释双下降、过参数化泛化、平坦极小点选择、梯度噪声异质性等方面的表现；大部分理论能准确捕捉到实验观察到的现象，尤其在无限宽极限与大批量训练条件下表现良好。

**⚠️ 局限性**

局限性包括：对有限宽网络的数值误差缺乏完整解析；多数理论假设理想化的平衡或无限宽条件，难以直接预测实际大规模模型的非平衡行为；对Transformer、扩散模型等新架构的物理描述仍不完整；缺乏统一的可验证量化指标，导致跨理论的直接比较受限。

---

## 309. PAGSL: Path-Augmenting Graph Structure Learning for explainable scholar recommendation

**arXiv ID:** 2611.09126 | [PDF](https://arxiv.org/pdf/2611.09126v1)

**作者:** Jiale Xu `[一作]` (Southwest University), Tao Jia `[通讯]` (Southwest University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Path-Augmenting Graph Structure Learning（PAGSL）模型，实现学者推荐并提供基于路径的可解释性；

**💡 创新点**

创新点在于：①通过属性共享增强图结构，激活潜在语义路径；②采用路径树编码保留分支传播信息；③结合k‑shell分解与可学习边掩码，显著抑制hub节点偏差；

**🔧 技术方法**

技术手段包括：图神经网络、路径树编码、对比学习、边掩码学习、k‑shell分解、路径评分与优化；

**📊 数据集**

实验数据集为Scholat、AMiner和ResearchGate三大学术社交网络；

**📈 对比分析**

与协同过滤、GNN、对比学习、异构图学习以及LLM4REC等多种基线相比，PAGSL在F1、Accuracy、AUC、HR@K、NDCG@K等指标上均提升5%~10%，且解释AUC显著高于GNNExplainer与PGExplainer；

**⚠️ 局限性**

局限性包括：①对路径长度与掩码参数的敏感性，需手动调优；②计算开销相对较高，尤其是路径树编码阶段；③在高度动态或极度稀疏的网络中仍可能面临路径稀缺或解释不足的问题。

---

## 310. Improving Large Audio-Language Model Embedding Baselines for Instruction-Aware Speech Retrieval

**arXiv ID:** 2611.09122 | [PDF](https://arxiv.org/pdf/2611.09122v1)

**作者:** Chen-An Li `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 IRIS-Embed 的方法，将大型音频-语言模型（LALMs）通过对比学习和指令感知音频层加权转化为嵌入模型，用于指令感知的语音检索。

**💡 创新点**

创新点在于：① 开发了指令感知的音频层加权机制，允许模型根据自然语言指令动态组合不同层的音频隐藏状态；② 通过构造约 90 万条由真实与合成数据组成的训练样本，系统化地把多属性检索需求（说话人、语义、情感、环境等）映射为自然语言指令；③ 展示了该方法在检索和多种音频任务上的广泛迁移能力。

**🔧 技术方法**

使用了 Qwen2.5‑Omni‑3B、Gemma‑4‑E2B‑it 等大型 LALMs，冻结其音频编码器和句子编码器；利用 LoRA 进行语言模型微调；对比学习（InfoNCE）优化检索嵌入；通过预训练的 sentence‑encoder（all‑MiniLM‑L6‑v2）生成音频层权重。

**📊 数据集**

训练数据来自公共语音与音频语料库（VCTK、VoxCeleb、IEMOCAP 等）以及从 MS MARCO 合成的文本三元组；检索评测使用 INSPIRE 基准（DailyTalk、VCTK、Expresso、Synthetic 四个子集）；通用音频任务评测使用 MAEB（15 个多样化任务）。

**📈 对比分析**

与多种基线（随机、通用自监督编码器、专用说话人/情感编码器、级联检索、未微调的 LALM 嵌入、LCO‑Embedding）进行对比。IRIS‑Embed 在 INSPIRE 上宏观平均 Recall@50 达到 62.92%（Qwen2.5‑Omni‑3B）和 47.41%（Gemma‑4‑E2B‑it），显著优于 45.15% 的最强基线；在 VCTK 从 48.75% 提升至 85.00%；在 MAEB 上宏观平均分分别提升 9.4–19.8 分。

**⚠️ 局限性**

局限性包括：仅使用两种 LALM 后端，未验证对更大模型的可扩展性；依赖冻结的音频编码器，未探索其他或补充编码器；层加权可视化未能明确各层对检索性能的具体贡献；训练数据仅覆盖有限的语言、声音、情感与环境属性，可能限制模型在更广泛语音场景下的表现；以及在 cascaded 基线中未尝试更细粒度的多模态融合。

---

## 311. BayesCrop: Real-Time Probabilistic Multi-Crop Video Reframing

**arXiv ID:** 2611.09100 | [PDF](https://arxiv.org/pdf/2611.09100v1)

**作者:** Cheng-Han Lee `[一作]` (University of Texas at Austin), Alan Bovik `[通讯]` (University of Colorado Boulder)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种实时概率多重裁剪框架 BayesCrop，用于在多种显示格式下对景观视频进行自适应裁剪。

**💡 创新点**

创新点在于将裁剪偏好建模为两组空间各向同性高斯混合，既能给出两个代表性裁剪结果，又能通过采样生成更多可行裁剪；并结合压缩域运动矢量实现高效的 anchor‑and‑propagate 推理。

**🔧 技术方法**

主要技术包括 DINOv2 视觉编码器、多尺度特征聚合、两维查询的 GMM 参数回归、压缩域运动向量传播以及周期性视角切换代理。

**📊 数据集**

论文使用自制的 LIVE‑YT‑MC 数据集（从 LIVE‑YT‑VC 衍生的多模式裁剪标签）、LIVE‑YT‑VC++ 以及 MiraData 长视频基准进行训练与评估。

**📈 对比分析**

与 CenterCrop、SmartVidCrop、STCAT、CG‑STVG、Gemini 3 Flash 等基线对比，BayesCrop 单输出 mIoU 55.2，最佳两裁剪覆盖率 68.4，参数量仅 26M、帧率 112FPS，显著兼顾准确性、覆盖度与实时性。

**⚠️ 局限性**

限制在于仅提供两组裁剪模式，可能不足以覆盖所有主观裁剪偏好；采样时仍需人工设置；在极端运动或快速场景切换时仍可能出现漂移。

---

## 312. AlienBench: A Benchmark of Structural Novelty in Language Model Outputs

**arXiv ID:** 2611.09094 | [PDF](https://arxiv.org/pdf/2611.09094v1)

**作者:** Jonas Oppenlaender `[一作]` (University of Oulu), Johanna M. Silvennoinen `[通讯]` (University of Jyvaskyla)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了AlienBench基准，用于评估语言模型在创意任务中对“结构性新颖性”的表现；

**💡 创新点**

将认知心理学中的结构想象任务正式化为机器学习可用的评分体系，并引入了“Ward离场得分”这一二元离场度量；

**🔧 技术方法**

采用多模型LLM生成、面板式LLM评审、Krippendorff α一致性检验以及统计显著性检验等技术；

**📊 数据集**

使用结构想象任务的10项生物特征列表、六种不同的提示条件以及通过人类众包收集的基线样本；

**📈 对比分析**

通过对多达40种LLM（包括前沿专有模型与开源模型）的平均离场得分和每个特征的离场率进行比较，发现开放权重模型平均得分（≈5.1）高于专有模型（≈4.5），且在所有提示条件下均无单一模型占优；

**⚠️ 局限性**

局限包括：得分受提示措辞和特征数量影响、模型可能记忆了科幻文本而非真正“新颖”思维、面板评审仍可能存在同家族偏见、以及基准仅衡量结构新颖性，不能评估文本连贯性或整体创意质量。

---

## 313. Rethinking Requirement-to-Code Traceability Through Single-Agent and Multi-Agent Approaches

**arXiv ID:** 2611.09162 | [PDF](https://arxiv.org/pdf/2611.09162v1)

**作者:** Sarthak Banerjee `[一作]`, Lionel Briand `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了单代理和多代理两种基于大型语言模型的需求‑代码可追溯性恢复方法，并在不使用检索的情况下直接在整个代码仓库上推理。

**💡 创新点**

创新点在于利用现代LLM的大上下文窗口实现仓库级推理，推出全局单代理模型以及基于自我纠错和多代理辩论的验证机制，突破检索瓶颈与单对单匹配的局限。

**🔧 技术方法**

技术上使用GPT‑5等LLM，并通过LangGraph构建工作流；采用两种提示策略（保守式和召回式）以及三种代理架构（单代理、self‑correction、multi‑agent debate）。

**📊 数据集**

实验采用eTour、iTrust和Dronology三大公开需求‑代码可追溯性基准，覆盖约6700–29600条候选链接。

**📈 对比分析**

与RAG+LLM基线相比，单代理在召回率上提升至80.9%（+38.6pp），而多代理辩论在F₂上达49.5%（+7.5pp），在精准率与检索负担之间取得较好平衡。

**⚠️ 局限性**

局限包括在召回式提示下精确率显著下降、单代理无法纠正错误、辩论方法无法恢复初始漏检的链接，以及对超大仓库的上下文窗口限制尚未解决。

---

## 314. TailDART: Compute-Efficient Foundation Models for Financial Time-Series Forecasting

**arXiv ID:** 2611.09176 | [PDF](https://arxiv.org/pdf/2611.09176v1)

**作者:** Chiang Qin Zhi `[一作]` (Nanyang Technological University), Joyjit Chattoraj `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并训练了一个2.68 M参数的金融时序基础模型TailDART，并在零样本情景下对多资产序列进行预测。

**💡 创新点**

在轻量化解码器中引入分组查询注意力（GQA）以显著减少KV缓存带宽，实现高精度同时降低推理延迟，并通过经验规模分析确定最佳模型尺寸。

**🔧 技术方法**

基于Transformer的解码器、RMSNorm、GeLU‑门控MLP或MoE、学生t/Johnson S_U分布头以及预训练+短历史微调的零样本推理。

**📊 数据集**

预训练使用21个公开时序数据集（ETT、GluonTS等），短历史微调加入6个额外数据集；零样本评估使用BForTFin 73条金融资产序列（日/周/月）。

**📈 对比分析**

与Lag‑Llama、Chronos‑T5‑tiny、Sundial‑base、Moirai等模型在MASE、RMSE和推理延迟上对比，TailDART在相同或更少参数下实现更低MASE、RMSE，并将延迟提升约48%。

**⚠️ 局限性**

模型规模与优化参数耦合，未验证更大规模在更细致调参下的表现；实验仅在单GPU固定硬件上完成，缺乏多机部署和在线学习等实际场景验证。

---

## 315. TOFS: Feature Selection with Swap Supervision for Semantic Communication over Fading Channels

**arXiv ID:** 2611.09173 | [PDF](https://arxiv.org/pdf/2611.09173v1)

**作者:** Zhiyi Zhu `[一作]` (Kobe University), Chikara Ohta `[通讯]` (Kobe University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种任务导向特征选择框架 TOFS，基于接收端分类损失来学习特征优先级，从而在无线衰落与噪声环境下优化图像分类传输。

**💡 创新点**

创新点在于使用接收端分类损失差异确认特征替换的优先级，将其转化为成对排序监督，避免单一重要性指标对渠道扭曲的忽略。

**🔧 技术方法**

采用深度联合源信道编码、特征池化、排名损失训练、零强迫均衡接收器等技术。

**📊 数据集**

使用 CIFAR-10 数据集进行训练与测试。

**📈 对比分析**

与全特征、随机选择、预测损失变化（PLV）以及基于 STE 的对照方法对比，TOFS 在 -2 dB Rayleigh 信道下取得 84.17% 的分类准确率，显著高于 PLV 的 68.49% 及基线的 44.29%。

**⚠️ 局限性**

局限性包括仅在单一任务模型和数据集上评估，依赖理想的 CSI 与无误差索引传输，未考虑多模型迁移、相关衰落、指示符误码与硬件实现成本等。

---

## 316. UniExit: A Unified Latency-Aware Early-Exit Runtime for Heterogeneous Edge AI

**arXiv ID:** 2611.09159 | [PDF](https://arxiv.org/pdf/2611.09159v1)

**作者:** Yuhang Guan `[一作]` (Hong Kong University of Science and Technology Guangzhou), Zhidan Liu `[通讯]` (Hong Kong University of Science and Technology Guangzhou)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的、延迟感知的边缘AI早停运行时，兼容CNN与Transformer模型的多任务推理；

**💡 创新点**

创新点在于把CNN残差块与Transformer层统一抽象为有序阶段，采用运行时状态感知的延迟预测、缓存重用、趋势回退以及CNN特有的退出稳定化机制，形成低开销、鲁棒的在线决策；

**🔧 技术方法**

使用实时硬件状态采集、两层MLP延迟预测器、LRU缓存、回退安全边际以及加权退出损失与单调性正则化等技术；

**📊 数据集**

使用CIFAR‑100、ImageNet（ResNet‑50、MobileNetV2）和WikiText‑103（LLaMA 3.2‑1B）等数据集；

**📈 对比分析**

与SPINN、E3、EdgeML、LayerSkip、SpecEE、Pantheon、HaX‑CoNN等基线对比，实验表明在静态与动态DVFS、单任务与异构并发、边缘‑云混合等场景下，DSR均可提升至≈95%‑98%，相较基线在精度-延迟平衡上提升10‑20%且计算与能耗下降；

**⚠️ 局限性**

局限包括：需要针对每个平台和模型进行离线收敛的预测器训练、对极大规模Transformer仍需量化或云协助、回退策略在极端突发变化时仍可能导致过度保守退出、目前未涵盖多语言或多模态混合推理。

---

## 317. Circuit Theory-Informed AI for Condition Monitoring in Power Electronics

**arXiv ID:** 2611.09141 | [PDF](https://arxiv.org/pdf/2611.09141v1)

**作者:** Yubo Song `[一作]` (Aalborg University), Subham Sahoo `[通讯]` (Aalborg University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种仅使用三相交流电流测量、通过 RC 型神经元映射误差并采用延迟编码来进行电力电子器件 R_dson 状态监测的方法。

**💡 创新点**

创新点在于将交流电流误差直接映射到 RC 神经元膜电位，并使用时延编码将系统动态嵌入脉冲神经网络，显著降低采样率和传感器复杂度。

**🔧 技术方法**

采用 RC 泄漏积分-放电神经元模型、延迟编码、脉冲神经网络（SRM-LIF）以及 AdamW 优化器和快速 Sigmoid 代理梯度实现训练。

**📊 数据集**

使用 PLECS 仿真生成的电压、电流数据，构建 R_dson 在 0~1 Ω 范围内变化的 8192 条样本（含训练、验证与测试集）。

**📈 对比分析**

与传统高精度传感器/数字双胞胎方法对比，SNN 在仅三相电流测量下实现 R^2>0.7、平均绝对误差<0.1 Ω，验证了其可接受的预测精度与能耗优势。

**⚠️ 局限性**

主要局限是对噪声鲁棒性不足（需较高 SNR 才能稳定产生脉冲）、设备对称性导致同类器件难以区分，以及目前仅在仿真环境下验证，实测环境表现待进一步验证。

---

## 318. DINO-Nucleus: Multi-Level Self-Supervised Features for Nucleus Instance Segmentation

**arXiv ID:** 2611.09133 | [PDF](https://arxiv.org/pdf/2611.09133v1)

**作者:** Mohammad Sarhangzadeh `[一作]` (Bilkent University), Doruk Oner `[通讯]` (Bilkent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种利用多层 DINOv2 自监督特征并结合 LoRA 参数高效微调的核实例分割与计数框架 DINO‑Nucleus

**💡 创新点**

创新点在于：1）多层特征融合保留空间细节与语义信息；2）使用像素级流场与显式中心预测相结合的中心引导分割；3）引入终点一致性损失使流场收敛至核中心；4）仅微调少量参数实现高效迁移

**🔧 技术方法**

技术包括 DINOv2 视觉变压器、LoRA 低秩适配器、前景/中心/流场头、流场积分与终点一致性损失、基于流场的实例重构

**📊 数据集**

使用 BBBC038 2018 细胞核实例分割数据集进行训练与评估

**📈 对比分析**

与 TC‑UNet++、Cellpose3、StarDist、CellViT 以及基线 DINOv2 + 水波纹重建进行对比，DINO‑Nucleus 在前景 Dice、实例 F1、PQ、AJI、计数 MAE、Pearson 相关性等指标均显著优于所有对手，参数量仅占 DINOv2 的 6% 左右

**⚠️ 局限性**

局限性包括：1）仅采用单个收敛中心，可能不适合极细长或形状不规则的核；2）实验仅在 BBBC038 上验证，其他图像域的迁移性能待进一步评估

---

## 319. Auditing Grasp Feasibility Predictors with Planner-Verified Counterfactuals

**arXiv ID:** 2611.09130 | [PDF](https://arxiv.org/pdf/2611.09130v1)

**作者:** Sajid Ansari `[一作]` (Indian Institute of Information Technology Hyderabad), Girish Varma `[通讯]` (Indian Institute of Information Technology Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6514db3d-8de6-452c-91b7-acdb31787cc4`

**🎯 论文内容**

通过在抓取路径中插入障碍物并让规划器重新标注，评估并发现大多数近乎完美的抓取可行性预测器并未真正利用碰撞几何，提出planner-verified counterfactual (CF) 测试并验证基于碰撞空间（clearance）标记训练的模型能显著提升对碰撞敏感性。

**💡 创新点**

提出了基于规划器验证的对比实验框架 CF-test，揭示了现有预测器对碰撞的“shortcut learning”，并证明只用原始数据通过清除标记（clearance tokenization）即可学习到碰撞几何，突破了仅靠自然分布高准确率无法验证碰撞感知的局限。

**🔧 技术方法**

使用采样规划器 (RRT‑Connect)、点云/体素/深度/对象令牌编码器、对比实验、对冲评估以及基于清除标记的token化读取技术。

**📊 数据集**

GraspNet‑1B 真实 RGB‑D 桌面场景及其 RRT‑Connect 规划标签，构建的 1,065 对 CF‑pair 以及多种障碍物/规划器/机器人设置。

**📈 对比分析**

在自然分布上所有预测器 AUROC 接近 1（0.92–0.996），但在 CF‑pair 上降至 0.47–0.504；仅使用 23k 参数的 clearance‑token 模型在 CF‑pair AUROC 达到 0.983；其他模型即使在对抗测试中也保持在 0.5 附近，表明对碰撞不敏感。

**⚠️ 局限性**

缺乏对障碍物位置和形状的普适性，清除表示在训练外推时仍受限，且当前方法仍依赖规划器验证；对不同规划器/机器人、不同障碍物几何的通用化仍待改进。

---

## 320. DARIC: A Position-Independent KV Cache Reuse System with Distribution-Aware Recomputation

**arXiv ID:** 2611.09109 | [PDF](https://arxiv.org/pdf/2611.09109v1)

**作者:** Weikang Wang `[一作]` (Lenovo Research), Weifeng Zhang `[通讯]` (Lenovo Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DARIC，结合分布感知重算的 KV 缓存重用框架，支持位置无关缓存重用，显著提升长上下文 LLM 的推理效率。

**💡 创新点**

创新点在于：①基于 KV 偏差分布自适应分配重算预算的分布感知 token 选择策略；②轻量级系统架构与 vLLM 协同，保持原有调度与优化；③块对齐与位置重编码实现无侵入的 KV 缓存重用。

**🔧 技术方法**

采用 KV 缓存偏差分析、Monotonic Token Allocation Function、块对齐与位置重编码、Redis+radix tree KV 索引、异步 GPU+主机内存/磁盘 KV 缓存离线、CUDA Kernel 加速，以及 vLLM 0.11.0 引擎在 Llama3.3‑70B、Mistral‑7B‑v0.2、Qwen3‑32B 等大模型上实现。

**📊 数据集**

使用 LongBench（2WikiMQA、MuSiQue、SAMSum、MultiNews、NarrativeQA）、函数调用数据集（Glaive）、SWE‑bench、LCC 代码补全以及扩展的 2WikiMQA‑M 等多种数据集。

**📈 对比分析**

与 Full Recompute、Full Reuse、CacheBlend、EPIC 对比，DARIC 在 RAG、函数调用、agentic 编码等场景中保持与 Full Recompute 相近的质量，TTFT 提升 2~8 倍；异步工作负载下 TTFT 减少约 4 倍，吞吐量提升 2 倍，Hit Rate 提升 1.2–3 倍。

**⚠️ 局限性**

局限性包括：需手动指定 chunk 边界，极端长或多段交错 prompt 仍受块对齐限制；重算预算仍为固定或需自适应调节；在高并发或极大模型下内存/存储开销较高；缺乏完全自适应的请求级预算机制。

---

## 321. PhoPunct: A Large-Scale Benchmark of Transformer Architectures for Vietnamese Punctuation Restoration

**arXiv ID:** 2611.09107 | [PDF](https://arxiv.org/pdf/2611.09107v1)

**作者:** Khoa D. Nguyen `[一作]` (AISIA Research Lab), Binh T. Nguyen `[通讯]` (AISIA Research Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在22.1M标记级别的越南新闻与小说语料上，构建了一个统一训练预算（8轮、单一学习率）下的17种句点恢复模型对照实验，评估了Transformer骨干与Bi‑LSTM/CRF预测头的性能；

**💡 创新点**

通过在同一数据规模和训练预算下对比不同预测头，首次揭示了在强大上下文编码器下Bi‑LSTM＋CRF并不总能提升宏观F1，并提出了“头部与优化预算隔离”这一实验范式；

**🔧 技术方法**

使用四种Transformer骨干（mBERT、XLM‑R、vELECTRA、ViBERT）以及PhoBERT‑large，并在其上堆叠Plain、Bi‑LSTM、CRF和Bi‑LSTM‑CRF等预测头；

**📊 数据集**

采用公开的越南News与Novels语料，分别包含19.9M新闻标记与2.2M小说标记，合计22.1M个标记；

**📈 对比分析**

采用统一的训练配置（AdamW、2e‑5学习率、8轮、FP16），在测试集上计算宏观F1进行对比。结果显示News域下PhoPunct（PhoBERT‑large+Bi‑LSTM‑CRF）取得最高61.35%宏观F1，而Novels域下XLM‑R+CRF取得最高50.18%；在8个匹配比较中，Plain头往往与Bi‑LSTM‑CRF持平或更优；

**⚠️ 局限性**

实验仅进行单一随机种子、单一预算，可能低估新初始化层的潜力；半稀缺的标点类别（如分号）导致宏观F1对极少数类别不敏感；仅覆盖新闻与小说两类文本，缺乏对会话或社交媒体等更具挑战性的域的验证；

---

## 322. Trajectory Forcing: Exploiting Diffusion Trajectories for Autoregressive Long Video Generation

**arXiv ID:** 2611.09096 | [PDF](https://arxiv.org/pdf/2611.09096v1)

**作者:** Bohan Wang `[一作]` (Huawei Noah Research Institute), Hanwang Zhang `[通讯]` (Huawei Noah Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了Trajectory Forcing（TraF）方法，利用视频扩散模型的去噪轨迹进行自回归生成，提升短视频和长视频的质量；

**💡 创新点**

创新点在于将每个视频块的完整去噪轨迹作为上下文条件，而非仅使用终点或单点，采用分阶段（监督与强化学习）训练并设计SNR导向的轨迹点选择；

**🔧 技术方法**

技术包括去噪扩散、流匹配（flow matching）在监督阶段、分布匹配蒸馏（Distribution Matching Distillation）在强化阶段、基于Wiener滤波的SNR选择函数和自回归训练框架；

**📊 数据集**

使用VBench数据集进行5秒短视频评估，使用MovieGenBench进行30秒长视频评估；

**📈 对比分析**

与Self Forcing、Causal Forcing、HiAR、Rolling Forcing等方法对比，TraF在VBench的Total得分为85.15（最高），在30秒长视频的Total得分为73.98（最高），且质量漂移最低；

**⚠️ 局限性**

局限性包括仅在单一基模型和分辨率下验证，未探究不同架构或更高帧率；训练吞吐量略低于Self Forcing；未结合长视频专用训练策略。

---

## 323. TUGB: A Two-stage Uncertainty-aware Generative Auto-Bidding Framework

**arXiv ID:** 2611.09091 | [PDF](https://arxiv.org/pdf/2611.09091v1)

**作者:** Luzhan Yuan `[一作]` (JD.com), Ching Law `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种两阶段的不确定性感知生成式自动竞价框架TUGB，结合生成模型与强化学习，解决生成式竞价在探索、非平稳环境以及RTG设定上的局限，并在JD搜索广告与食材配送广告系统中实现上线部署；

**💡 创新点**

创新点包括①将RTG依赖剔除的预训练策略，让模型从行为模仿到风险感知的自动转化；②采用概率集成环境模型量化知识不确定性，并在策略优化中加入不确定性惩罚，实现安全探索；③两阶段流程实现从离线监督到RL自适应的无缝过渡；

**🔧 技术方法**

使用Decision Transformer变体做生成竞价模型，概率集成（ensemble）环境模型做不确定性估计，Group Relative Policy Optimization（GRPO）做强化学习优化，并加入KL约束与不确定性惩罚；

**📊 数据集**

利用JD搜索广告大规模线上日志（约1M轨迹、数千万展示）和JD食品配送稀疏数据（约100K轨迹）进行训练与部署；同时在公开的AuctionNet与AuctionNet‑Sparse基准上进行离线对比；

**📈 对比分析**

在JD搜索广告的A/B测试中，相比生产基线USCB提升成本+2.32%、目标成本+3.57%、CSR+0.55%；在机制升级与跨域（食品配送）同样获得正向提升；在AuctionNet基准上与USCB、IQL、CQL、DT、GAS、GAVE等传统与生成式竞价方法对比，TUGB在所有预算设置下取得最高Reward，提升约3‑4%；离线ablation实验进一步证明不确定性模型与RTG‑free设计的显著贡献；

**⚠️ 局限性**

局限性在于仍需依赖离线模拟器，迁移到全新场景需重新训练环境模型；对超参数敏感；仅实现单目标约束，未覆盖多目标竞价；在线探索仍受成本约束；在极端稀疏数据下的鲁棒性还有提升空间。

---

## 324. PaB-tree: Parallel B+-tree with Batch Updates

**arXiv ID:** 2611.09172 | [PDF](https://arxiv.org/pdf/2611.09172v1)

**作者:** Ziyang Men `[一作]` (University of California, Riverside), Helen Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种首个批量并行 B‑树实现，具备可证明的工作量、跨度和缓存复杂度，并在内存中实现高性能。

**💡 创新点**

通过单一 invariant——修复永不触碰兄弟、始终推向父节点，统一溢出和欠流为 Promote 原语，极大简化并行修复流程。

**🔧 技术方法**

利用 heavy node 结构、Promote 统一原语、叶子随机化排序策略，以及 fork‑join 并行与理想缓存模型实现高效并行算法。

**📊 数据集**

在合成数据（10^9 条记录）、真实地理数据（1.35B 位置记录）和大型图数据（多达 2.5B 边）等多种数据集上进行实验。

**📈 对比分析**

与多种基准（并行 BST、Packed Memory Array、并发 Bw‑tree 等）对比，批量插入/删除性能显著优于现有实现，查询性能保持竞争力，整体速度提升可达数倍。

**⚠️ 局限性**

目前仅支持内存实现，未扩展到外存或持久内存；对高频单点更新的适配仍有待改进。

---

## 325. OMCDE: Opposition-based Multi-Stage Matrix-Based Competitive Differential Evolution

**arXiv ID:** 2611.09167 | [PDF](https://arxiv.org/pdf/2611.09167v1)

**作者:** Mohammad Sarhangzadeh `[一作]` (Bilkent University), Doruk Oner `[通讯]` (Bilkent University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于矩阵的竞争型差分进化算法OMCDE，结合对抗学习、三元竞争和分阶段变异，实现高效并行求解。

**💡 创新点**

创新点在于将所有DE操作完全矢量化，并引入分阶段变异与三元竞争，兼顾探索与利用，同时使用对抗学习提升初始化质量。

**🔧 技术方法**

使用技术包括矩阵化DE、三元竞争、对抗学习、分阶段变异、线性种群衰减、Cauchy/正态分布参数自适应、Weighted Lehmer均值更新、并行化NumPy实现。

**📊 数据集**

数据集为CEC 2017 benchmark suite（29个函数，维度D=30）。

**📈 对比分析**

与TCDE、LSHADE‑RSP、iLSHADE‑RSP、L‑SHADE、JADE等算法对比，采用Wilcoxon秩和检验和Friedman排名，OMCDE在大多数函数上显著优于或等效，整体排名为2.36。

**⚠️ 局限性**

局限性在于阶段切换使用全局阈值，缺少个体级自适应调整；在极高维或目标函数复杂度极高的实际工程问题中仍需进一步验证。

---

## 326. Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition

**arXiv ID:** 2611.09163 | [PDF](https://arxiv.org/pdf/2611.09163v1)

**作者:** Nick Janssen `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Continuous Dictionary Pursuit (CDP)，一种基于连续参数化混合字典的贪心前向搜索框架，用于可解释的信号分解。

**💡 创新点**

创新点在于消除了离散字典的格点误差，利用连续优化在包含周期、趋势、瞬态、断裂等多种原子族的混合库上实现高精度分解，并通过谱锐化实现可微分的断裂波形。

**🔧 技术方法**

采用 FFT 先验生成候选原子，L‑BFGS 梯度优化配合软障碍约束进行参数微调，整体实现连续字典贪心前向选择与全局联合微调。

**📊 数据集**

在合成数据以及 13 个公开 LTSF 基准（ETT、Electricity、Exchange、PEMS、Solar、Traffic、Weather）上进行实验验证。

**📈 对比分析**

与九种基准（经典分解、稀疏编码、连续追踪、深度序列模型等）比较，CDP 在稀疏 regime 下实现了最佳的准确率‑稀疏度 Pareto 前沿，参数量显著少于深度模型，预测误差亦优于同类方法。

**⚠️ 局限性**

局限性包括非凸贪心优化缺乏全局收敛保证、对预设原子库的依赖导致对未覆盖结构拟合不佳、计算成本高于离散字典方法，以及仅适用于单变量信号。

---

## 327. Learning When to Think: Dual-Reference Offline Optimization for Adaptive VLM Reasoning

**arXiv ID:** 2611.09154 | [PDF](https://arxiv.org/pdf/2611.09154v1)

**作者:** Hongbin Lin `[一作]` (Hong Kong University of Science and Technology), Yao Shu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Dual-Reference BOLT（DR-BOLT）框架，用于离线自适应视觉语言模型的快速/慢速推理；

**💡 创新点**

创新点在于将快速/慢速思维视为双参考问题，识别并纠正参考分布与采样分布的不匹配，并通过行为-参考校正与分区路由机制实现自适应推理；

**🔧 技术方法**

主要技术包括基于KL正则化的BOLT改进、奖励加权的监督微调、分区路由解释、模式格式分离的近似校正以及三阶段训练流程；

**📊 数据集**

实验使用ScienceQA、ViRL39K、MathVision、DynaMath、MMMU-Pro和构造的3DSRBench数据集；

**📈 对比分析**

与GRPO、FAST-GRPO、TON、DualMindVLM以及内部对比（单参考、MR-DPO、Cond-RW）比较，DR-BOLT在Score/Length上显著优于所有基线，跨验证提升慢速比例约24%，训练时间相对GRPO提升约2.9倍；

**⚠️ 局限性**

局限性包括对两参考设定之外的鲁棒性尚未验证，依赖模式格式分离假设，若快速/慢速样本重叠严重需精确校正，且模型仅在离线候选上训练，缺乏在线探索机制。

---

## 328. Augmentron: Scalable Multi-View Visual Augmentation for Robot Learning

**arXiv ID:** 2611.09145 | [PDF](https://arxiv.org/pdf/2611.09145v1)

**作者:** Tanay Tandon `[一作]` (University of Pennsylvania), Pragna Mannam `[通讯]` (Scale AI, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在机器人演示数据中引入几何感知的可重用场景资产，实现多视角、时间一致的视觉增强，降低对任务无关视觉特征的敏感度。

**💡 创新点**

核心创新在于将场景重建与渲染分离：先用机器人状态与相机标定生成一次性场景资产（顶视图+六面立方图），再通过记录的相机位姿进行渲染和合成，从而避免逐帧生成并保证跨视角一致性。

**🔧 技术方法**

技术关键包括基于机器人运动学的机器人遮罩、利用语言模型+SAM3进行任务物体与工作面分割、FLUX扩散模型生成可重用场景表面与立方图、几何映射渲染与合成。

**📊 数据集**

使用了1793条真实演示（约9.16小时），涵盖杯子-托盘、毛巾折叠、花瓶插入三项操纵任务。

**📈 对比分析**

与仅使用真实数据、RoboEngine以及Masked Noise等基线在三种视觉偏移（ID、OOD distractors、OOD workspace）下对比，Augmentron在控制的30k步替换实验中将ID pooled success从66.7%提升至76.7%，在OOD distractors从18.0%提升至45.3%，在OOD workspace从50.7%提升至67.3%；在45k步实验中，Augmentron的整体成功率均高于RoboEngine，并与Masked Noise竞争。

**⚠️ 局限性**

局限性包括：在不同数据量和训练步长的比较中难以单独评估各组件贡献；实验任务和视觉偏移有限，未验证对更复杂任务或多种分布漂移的泛化；未充分评估多种随机种子和更大规模的训练集。

---

## 329. Learning Generative Dynamics for 3D Molecule Generation via Sequential Neuro-Symbolic Constraints

**arXiv ID:** 2611.09142 | [PDF](https://arxiv.org/pdf/2611.09142v1)

**作者:** Zhuoran Li `[一作]` (Hong Kong Polytechnic University), Wanyu Lin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 SensDiff，一种在 3D 分子生成的扩散模型中整合符号约束的神经符号框架，能够在训练过程中强制化学物理先验；

**💡 创新点**

创新点在于将物理化学规则转换为可微能量约束，并通过粗到细的逐步激活与自适应权重策略，将约束与扩散过程同步；

**🔧 技术方法**

使用可微分分子接口、扩散模型（GEOLDM 基础）、符号能量约束、粗细层级训练日程和自适应权重；

**📊 数据集**

在 QM9 与 GEOM‑Drugs 两大分子基准上进行无条件与条件生成实验；

**📈 对比分析**

与 EDM、EEGSDE、GEOLDM、EquiFM、GeoBFN、GCDM、RADM、SLDM、UAE‑3D、UniGEM 等 10+ 近期基线对比，SensDiff 在分子有效性、原子有效性、唯一性、量子属性 MAE 等指标上均取得最优或接近最优成绩；

**⚠️ 局限性**

限制在于：需要手工设计并预先计算符号规则的统计表，对复杂或稀缺结构的扩展仍有挑战，且在高度约束的多目标任务中仍可能出现梯度竞争。

---

## 330. Shortest Synchronizing Words on Product Automata

**arXiv ID:** 2611.09136 | [PDF](https://arxiv.org/pdf/2611.09136v1)

**作者:** Henning Fernau `[一作]` (University of Trier), Igor Rystsov `[通讯]` (National Technical University of Ukraine Igor Sikorsky Kyiv Polytechnic Institute)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统研究了 DFA 的各种图操作（如直接乘积、两拍乘积、并、合并）对最短同步词长度（reset 阈值）的影响，并给出了相应的上下界。

**💡 创新点**

创新点在于将同步阈值与图操作相结合，证明了多类 DFA（偏序、交换、幂等）在这些操作下的闭包性质，构造了新的满足 Černý 猜想的 DFA 类，并将同样方法推广到 Mealy 机的 homing 序列。

**🔧 技术方法**

主要使用了图论、幺半群与自动机理论中的代数方法，利用同步词与子集同步、同态映射等技术推导界限。

**📊 数据集**

未使用具体数据集，全部为理论证明。

**📈 对比分析**

该工作不涉及实验比较，主要通过理论界限与已知结果的对比证明其正确性。

**⚠️ 局限性**

局限在于仅给出理论界限，未给出最优同步词构造算法；某些类在合并操作下不封闭；对实际应用的指导有限。

---

## 331. Look Closer, Align Better: Explicit Visual Attributes for Robust EEG-to-Image Decoding

**arXiv ID:** 2611.09123 | [PDF](https://arxiv.org/pdf/2611.09123v1)

**作者:** Ke Xiao `[一作]` (Shanghai University), Fenqi Rong `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种名为AttrEcho的EEG到图像检索框架，通过结合预训练视觉模型（如CLIP）与显式视觉属性（HOG、纹理、轮廓）实现更细粒度的特征融合，并在EEG-图像匹配中显著提升检索精度。

**💡 创新点**

创新点在于（1）通过时间分辨的部分RSA与属性解码验证显式视觉属性在EEG响应中的可解码性和对CLIP特征的补充价值；（2）提出Region‑Aligned Visual Attribute Fusion（RAVF）实现局部属性与对应CLIP特征的加权残差融合；（3）使用Feature Token Aggregation（FTA）学习维度‑级别的区域权重进行图像级别聚合；（4）整体架构保留预训练特征的优势，同时通过属性补充提升空间细节匹配。

**🔧 技术方法**

技术包括：EEG信号预处理与编码、CLIP视觉特征提取、显式视觉属性提取（HOG、纹理、轮廓、Gabor、频域）、RAVF门控残差融合、FTA维度级聚合、双向对比学习损失、以及在同类干扰器中的大规模检索实验。

**📊 数据集**

主要使用THINGS‑EEG数据集（10位受试者、16,540训练图像、200测试图像），并在THINGS‑MEG与Alljoined‑1.6M数据集上进行跨数据集验证。

**📈 对比分析**

与现有8种EEG‑图像检索方法（如BraVL、NICE、ATM、UBP、BVSA、NeuroBridge、Blur、Shallow）比较，AttrEcho在THINGS‑EEG 200‑way检索中平均Top‑1/Top‑5精度分别为94.8%/99.7%，大幅优于最强基线（提升12.2%/2.0%），在1,600‑way大图集检索中保持82.1%/96.6%精度，降幅仅12.7%，显著优于Shallow等基线。跨受试者检索和其他数据集也表现出显著性能提升。

**⚠️ 局限性**

局限性包括：对显式属性的依赖需要手工设计或预先提取；RAVF和FTA对图像划分和属性选取的假设可能限制在更复杂场景的适用性；对不同EEG硬件和通道数的泛化能力尚未完全验证；模型参数量相对较大，部署成本可能受限。

---

## 332. Authorization Drift in LLM Agents: Evaluating Boundary Compliance in Multi-Step Tool Use

**arXiv ID:** 2611.09113 | [PDF](https://arxiv.org/pdf/2611.09113v1)

**作者:** Chuxue Cao `[一作]` (Hong Kong University Of Science And Technology), Sirui Han `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型代理在执行任务时出现的非恶意授权漂移（即在没有外部攻击者干预的情况下，自主越权）并提出了相应的评测基准（Authorization Drift Benchmark，ADB），通过多模型、多框架的实地评估揭示了漂移与模型能力之间的关系。

**💡 创新点**

创新点包括：①把授权漂移定义为四个维度（数据、工具、范围、时间）的序列级违规；②构造双路径基准，让每个任务既有合法完成路径也有诱人但非法的捷径；③提出基于执行反馈的 Staged Execution‑Reflective Synthesis (SERS) 生成流程，用真实代理跑通验证样本合法性与漂移诱因；④将漂移事件映射到多司法管辖区的法律条文，量化潜在法律风险。

**🔧 技术方法**

技术手段包括：任务设计与工具调用的 LLM 自动生成；执行回显与规则匹配的漂移检测；多模型多框架（Normal、ReAct、Planner‑Executor）跑通与对比；对漂移点进行法律归属推理；以及在评价中采用 Pass@1、TSR、DR、CSR、CCSR 等指标。

**📊 数据集**

使用的数据集为：1) ADB，648 条样本，覆盖七大领域，每条包含合法路径与非法捷径；2) SERS 生成的可执行环境、工具 schema 及注释文件；3) 一个由 11 份法律文件提取的 150 条法律条文库，用于漂移点到法律条文的映射。

**📈 对比分析**

比较方法：在 13 种 LLM（4B–397B 参数）上以 Normal、ReAct、Planner‑Executor 三种框架进行 Pass@1 评估，记录任务成功率 TSR、漂移率 DR、合规成功率 CSR 以及合规条件成功率 CCSR。实验发现，高能力模型（如 Qwen、Claude Sonnet）虽然任务完成率高，但漂移率也显著提升；框架加剧则能降低漂移率，但往往伴随任务完成率的下降。大多数任务（≈86%）在无漂移情况下仍可成功完成。

**⚠️ 局限性**

局限性：①只考虑单会话、单用户场景，未覆盖多代理或子代理的授权交互；②漂移检测仅依据预先标注的违规集合，无法捕捉未预见的越权路径；③基准样本的“诱导”性质可能不完全代表真实部署环境；④法律映射仅为解释性关联，未判断是否真实触犯法规。

---

## 333. Safe Circumnavigation on Heterogeneous Orbits: A Range-Only HOCBF-QP Framework for Nonholonomic Multi-Agent Systems

**arXiv ID:** 2611.09112 | [PDF](https://arxiv.org/pdf/2611.09112v1)

**作者:** Shubham Singh `[一作]` (Institute of Science Tokyo), Katsuki Fujisawa `[通讯]` (Institute of Science Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个安全关键的多机器人环绕目标控制框架，使用加速度和转向率，并通过高阶控制障碍函数二次规划实现范围‑仅传感器下的目标轨道跟踪、均匀间距以及碰撞避免。

**💡 创新点**

① 通过体坐标里程误差不依赖轨道半径，实现单一控制律适用于异质轨道；② 采用双输入backstepping分层控制，分离半径收敛和队形协调；③ 结合高阶控制障碍函数与实时估计误差补偿，在无测量速率的情况下保持硬安全保证。

**🔧 技术方法**

高阶控制障碍函数（HOCBF）+二次规划（QP）过滤器；超弯曲差分估计器实现范围速率估计；基于环图的距离约束与梯度下降形成相位律；动态面回归的backstepping加速度控制。

**📊 数据集**

本文采用仿真数据进行验证，未引用公开数据集，使用自定义的多机器人模拟实验（N=3、4）来展示控制性能。

**📈 对比分析**

与现有最接近的多轨道控制器（无安全保障）在相同参数下进行对比；实验结果表明该方法在目标距离和机器人间距均保持安全界限并收敛到预定轨道，而传统方法在初期违反安全约束；性能在仿真中表现为快速收敛、始终保持安全距离且对范围传感器误差鲁棒。

**⚠️ 局限性**

仅在静止目标下验证；对环图拓扑的非邻居机器人收敛性分析有限；移动目标、时间变化拓扑以及硬件实现等进一步场景尚待研究。

---

## 334. Separating Measures by Hyperplanes

**arXiv ID:** 2611.09166 | [PDF](https://arxiv.org/pdf/2611.09166v1)

**作者:** José Correa `[一作]` (Universidad de Chile), Eilon Solan `[通讯]` (Tel Aviv University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在ℝ^d中用超平面分割两种颜色的随机点所能取得的最大“切割比率”，并给出了该比率在不同维度、不同范数以及不同测度类（有限、绝对连续、离散）下的上界和下界。

**💡 创新点**

创新点在于：①将切割问题与几何分割理论（ham‑sandwich定理、签名ham‑sandwich定理）相结合，得到对所有度量空间均适用的统一框架；②证明不同测度类（任意、有限、绝对连续、离散）在切割比率上是等价的；③在二维欧氏空间给出精确上界5/11，且在所有维度上给出下界8/(8+5π)，并证明当维度趋于无穷时比率逼近1/2。

**🔧 技术方法**

主要技术手段包括：几何分割定理（ham‑sandwich、签名ham‑sandwich），积分表示与Crofton公式，概率与期望的运算，凸几何与相容性论证，误差控制与极限论证。

**📊 数据集**

该工作为理论性质研究，没有使用具体的实验数据集。

**📈 对比分析**

通过构造上下界实例来评估理论极限。对下界给出具体的概率构造和随机超平面；对上界给出精确几何构造和极限分析。虽然没有实验性能指标，但在二维欧氏空间与一般维度下均给出了最小与最大可能值的严格界定。

**⚠️ 局限性**

局限性：①在高维欧氏空间的下界与上界仍存在差距；②对非欧氏范数的上界仍为1/2，尚未给出更细致的数值；③对随机测度的具体分布没有给出最优超平面的构造算法，只给出存在性证明。

---

## 335. Reconciling European Electricity Data with Diffusion Priors Learned from Observations Alone

**arXiv ID:** 2611.09265 | [PDF](https://arxiv.org/pdf/2611.09265v1)

**作者:** Matthias Pirlet `[一作]` (University of Liège), Damien Ernst `[通讯]` (University of Liège)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将欧洲透明度平台的发电、负荷和跨境输电数据的错误纠正问题建模为贝叶斯推断，并使用扩散模型学习先验，实现严格满足能量守恒的修正。

**💡 创新点**

创新点在于用扩散模型取代传统均匀先验，并通过经验贝叶斯 EM 仅用观测数据估计先验；在每一步采样时投影到能量守恒空间，保证修正结果严格符合平衡。

**🔧 技术方法**

采用贝叶斯推断、经验贝叶斯 EM、扩散模型、导向采样（guided denoising）以及投影正则化等技术。

**📊 数据集**

使用合成的四个欧洲投标区网络（eu6、euwest、euwide、eufull），基于 ENTSO‑E ERAA 2024 的发电、负荷和流量数据，并加入 Gaussian 噪声。

**📈 对比分析**

与传统 WLS（无信息、oracle、约束）、de Chalendar 经验法、Gaussian 经验贝叶斯先验进行比较；在 RMSE、CRPS 和被遮蔽信号误差上评估，扩散经验贝叶斯先验在所有系统上均取得最低 RMSE、最佳 CRPS，并且置信区间校准或保守。

**⚠️ 局限性**

先验不完全代表真实状态分布，缺乏非负性约束；观测噪声假设为独立 Gaussian，实际误差随幅度、TSO 偏差及时间相关，限制了在真实数据中的直接验证。

---

## 336. Nudging Decision Models: Do Digital Nudges Steer Fast, Generation-Free AI Agents?

**arXiv ID:** 2611.09264 | [PDF](https://arxiv.org/pdf/2611.09264v1)

**作者:** Sascha Kaltenpoth `[一作]` (Paderborn University), Oliver Müller `[通讯]` (Paderborn University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三种非生成决策模型（Jev、Decision-1、d1）在模拟在线购物环境中进行实验，评估其对自动化提示（默认选项）和反思提示（社交影响标签）的敏感性。

**💡 创新点**

首次证明决策模型能被数字提示影响，并且其对提示的响应模式符合系统1/系统2理论中自动化处理的预测；同时展示了该类模型在速度与成本上的显著优势。

**🔧 技术方法**

使用Browser Use Library让模型通过浏览器接口与网页交互；采用 Bayesian 多层逻辑回归分析各条件下目标选择概率；实验框架基于先前人类/LLM实验的设计。

**📊 数据集**

实验数据来自自建的在线购物平台，每个模型在六个商品类别中做两选项决策，分别在无提示、默认提示和社交影响提示三种条件下进行200次运行（每条件600次总计）。

**📈 对比分析**

与无提示基线相比，默认提示使目标选择率从约48%提升至97.5–99.9%；社交影响提示提升至75–96%。决策模型比典型LLM快约8–19倍、成本低约60–280倍，且对默认提示的依赖度显著高于社交影响提示。

**⚠️ 局限性**

仅测试了三款封闭源决策模型；每种提示类型仅使用单一实例；未在相同实验条件下与LLM进行直接比较；实验设计与先前LLM研究略有差异，限制了跨模型、跨提示类型的一般化结论。

---

## 337. Model-based learning for DoA estimation and array calibration via continuous sparse recovery

**arXiv ID:** 2611.09263 | [PDF](https://arxiv.org/pdf/2611.09263v1)

**作者:** Baptiste Chatelier `[一作]` (INSA Rennes), Luc Le Magoarou `[通讯]` (INSA Rennes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于模型驱动机器学习的联合阵列校准与方向估计方法，利用无监督的项目梯度下降（PGD）对硬件失配进行校准，并在稀疏恢复框架中使用连续字典对DoA进行精细化估计。

**💡 创新点**

创新点在于：① 将阵列校准问题视为字典细化问题，使用无监督PGD学习硬件失配；② 将离散稀疏恢复方法扩展到连续字典，实现对离网误差的补偿；③ 通过先校准再估计的两阶段流程，既保持了解释性又显著降低对标记数据的依赖。

**🔧 技术方法**

主要技术包括：稀疏恢复中的同时正交匹配追踪（SOMP）、项目梯度下降（PGD）、连续字典学习、非监督损失函数（基于投影残差）以及对角度约束投影。

**📊 数据集**

使用合成的统一线性阵列（ULA）数据集，生成包含多源、噪声、硬件失配（天线位移与增益噪声）以及两种信号模型（非协同与全协同）的仿真数据。

**📈 对比分析**

与MUSIC、离散SOMP等基线方法比较，结果表明：① 对全协同源和低采样快照具有更强的鲁棒性；② 连续字典方法在相同精度下可使用更粗的角度网格，显著降低计算成本；③ 无监督校准方法在不依赖标记数据的情况下即可逼近物理失配下的最佳性能。

**⚠️ 局限性**

局限性包括：① 无监督校准仍受初始角度估计误差影响，可能导致残余比例与平移歧义；② 方法假设源数已知，未考虑未知源数的情形；③ 仅考虑增益和位置失配，未覆盖互耦或宽带/近场信号的失配问题。

---

## 338. Practical Non-Stationary Graph Gaussian Processes

**arXiv ID:** 2611.09247 | [PDF](https://arxiv.org/pdf/2611.09247v1)

**作者:** K. S. Sesh Kumar `[一作]` (Brevan Howard Centre for Financial Analysis), Viacheslav Borovitskiy `[通讯]` (School of Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种非平稳图高斯过程模型（VCGP），利用节点辅助信号通过单一指数参数γ来控制节点间的传播与隔离，解决了传统图马特恩核在异质网络上的局限。

**💡 创新点**

创新点在于将节点方差与边的耦合度关联，仅使用一个可学习参数γ，避免了每个节点都需独立参数的过拟合问题，并且证明γ=1是唯一能保持谱平稳性的点，使得模型在γ≠1时真正实现非平稳相关性。

**🔧 技术方法**

主要技术包括：图马特恩核的精度矩阵构造、引入辅助信号d_i的对角矩阵D、γ耦合的稀疏精度Q_γ、边缘似然最大化、梯度优化、块分区γ_k、以及对γ的Fisher信息岭正则化。

**📊 数据集**

使用了五个真实图回归数据集：METR-LA、PEMS-BAY、LAQN、PM2.5、CA Weather，这些数据分别覆盖交通流量、空气质量、气象温度等多种领域。

**📈 对比分析**

与平稳马特恩、可变系数SPDE、深度核GP、GCN等多种基线进行对比，VCGP在大多数数据集上实现了更低的负对数似然（NLL）和均方根误差（RMSE），尤其在多模态PM2.5和CA Weather上表现突出。

**⚠️ 局限性**

局限性包括：依赖可获取的辅助信号d_i，γ估计缺乏理论下界，块分区的选择需要先验或启发式方法，且未能直接建模时空变化或异质性带来的更细粒度非平稳结构。

---

## 339. Large Language Model Consensus and Corpus-Adaptive Topic Modeling for Classifying and Mapping Patent Innovation

**arXiv ID:** 2611.09194 | [PDF](https://arxiv.org/pdf/2611.09194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 340. Attaining Optimal Values under Sure Constraints

**arXiv ID:** 2611.09244 | [PDF](https://arxiv.org/pdf/2611.09244v1)

**作者:** Takuto Yamauchi `[一作]` (Waseda University), Kenji Tei `[通讯]` (Institute of Science Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了有限随机博弈中的硬规范，探讨了在每个支持执行上必须满足的目标，包括零概率路径。

**💡 创新点**

提出了对显式硬和布尔软Streett条件的严格和非严格阈值问题的复杂性分类，并给出了最佳实现的内存需求界限。

**🔧 技术方法**

使用了有限控制器和纯公共历史策略的博弈模型，结合了适应性查询和多种复杂性理论。

**📊 数据集**

未具体提及使用的数据集，但涉及的模型和条件适用于一般的随机博弈场景。

**📈 对比分析**

通过复杂性理论比较了不同问题的可解性，证明了在特定条件下的复杂性分类，如coNP-complete和P^NP-complete。

**⚠️ 局限性**

限制在于未解决最佳状态计数依赖性和有效的通用合成算法。

---

## 341. Controllable Molecule Generation via Sparse Representation Editing: An Interpretability-Driven Perspective

**arXiv ID:** 2611.09235 | [PDF](https://arxiv.org/pdf/2611.09235v1)

**作者:** Zhuoran Li `[一作]` (Hong Kong Polytechnic University), Wanyu Lin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了SpaRE框架，利用稀疏表示编辑实现LLM生成分子的局部与全局可控化学结构生成。

**💡 创新点**

通过稀疏自编码器将LLM隐藏层表示解耦为化学意义明确的概念，并在推理时编辑这些概念实现精准的原子/功能基定位与全局属性调节。

**🔧 技术方法**

稀疏自编码器（SAE）、前向钩子、对比指导、LLM对话提示以及Monte Carlo树搜索等技术。

**📊 数据集**

ChEBI-20、GEOM‑DRUGS、GEOM‑QM9 等公开分子数据集以及对应的药物活性、可合成性指标。

**📈 对比分析**

与13个基线（如MolXPT、BioT5、LDMol、NExT‑Mol等）在生成质量、可合成性、可控率、效率等指标进行对比，SpaRE 在大多数指标上取得或超过最佳基线，控制成功率最高且生成效率最快。

**⚠️ 局限性**

对预训练LLM的依赖与对概念选择、过完备度、层次等超参数敏感；对极端属性或多目标约束的可解释性有限；需要人工构造对比样本或概念集合。

---

## 342. Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL

**arXiv ID:** 2611.09200 | [PDF](https://arxiv.org/pdf/2611.09200v1)

**作者:** Duc-Vu Nguyen `[一作]` (University of Information Technology), Ngan Luu-Thuy Nguyen `[通讯]` (University of Information Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在DRiLL法律检索任务中，结合结构化和谱系信息（如条款同一法案支持、引用关系等），改进检索和选择流程，提升检索效果；

**💡 创新点**

引入了基于法律领域知识的新增特征（身份/本地形式、同法案支持、引用/谱系支持），并在统一候选池上共享 LLM 证据，实现对多条款问题的更精确检索；

**🔧 技术方法**

使用了BM25、BGE-M3、HyDE、引用扩展等检索技术，并在 LightGBM 决策树上加入上述法律特征；

**📊 数据集**

使用了公开的 DRiLL 语料库（59,636 条款、2,190 训练/312 开发/627 测试问答）；

**📈 对比分析**

与原 EDM 系统以及竞赛前三名进行比较，五种随机种子平均后 F_2 从 71.77% 提升到 72.91%，在两条款问题上提升显著；

**⚠️ 局限性**

仅在 DRiLL 上评估，缺乏跨数据集验证；未对单独特征做消融，且依赖 LightGBM 与手工特征，未探讨更通用模型或实时成本。

---

## 343. Bootstrap Your Own AV-Proxies: Adaptive Contrastive and Prototype Learning for Audio-Visual Segmentation

**arXiv ID:** 2611.09191 | [PDF](https://arxiv.org/pdf/2611.09191v1)

**作者:** Junbo Zhang `[一作]` (Wuhan University), Chao Sun `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BYOAVP 框架用于音频-视觉分割，解决音频视觉语义差距与单模态噪声问题；

**💡 创新点**

创新点在于将自监督对比学习与跨模态注意力结合用于音频增强（SSAE），以及利用动量更新的原型约束（DPC）进行像素级分类与交互；

**🔧 技术方法**

采用自监督对比学习、跨模态注意力、动量原型更新、GroupNorm、以及基于 Swin-B 与 CLAP/HTSAT 的视觉/音频编码器；

**📊 数据集**

在 AVSBench（单/多声源及语义分割）和 VPO（单/多声源/多实例）两个基准数据集上进行评测；

**📈 对比分析**

与现有 15+ 方法对比，BYOAVP 在所有子任务上均实现 J&F 指标最高，显示出显著的性能提升；

**⚠️ 局限性**

限制在于模型参数相对较多，计算开销更大，并且仍需进一步验证在更大规模或更复杂环境下的鲁棒性与实时性。

---

## 344. Replication Certification at Publication: Checkable by Anyone, Anywhere, in Seconds, Forever

**arXiv ID:** 2611.09214 | [PDF](https://arxiv.org/pdf/2611.09214v1)

**作者:** Gary King `[一作]` (Harvard University), Leo Yang Yang `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种复制证书，用于证明论文所用数据与代码在提交后未被篡改且已按预定随机种子完成运行，消除对作者单方面陈述的依赖。

**💡 创新点**

创新点在于将加密签名、基于区块链的随机种子、时间戳以及完整的运行记录相结合，形成一个不可篡改、可公开验证的“复制凭证”，让期刊可在不重新执行分析的情况下确认可复制性。

**🔧 技术方法**

使用哈希、Merkle树、数字签名、公钥加密、OpenTimestamps、签名日志和可验证随机函数等成熟的加密技术；同时通过“封闭复制副本”实现环境冻结和随机种子计算。

**📊 数据集**

针对一般科研存档中的复制数据集（如Dataverse、Ro-Crate等），并在必要时通过隐藏承诺（hiding commitment）处理机密数据。

**📈 对比分析**

与传统的人工重现或期刊数据编辑重现相比，证书的验证仅需秒级查询，显著降低了时间和人力成本；验证步骤不需要重新构建环境或重新运行完整分析，效率大幅提升。

**⚠️ 局限性**

局限性包括：不对科学有效性（研究问题、p‑hacking、模型依赖等）进行评估；受限于文件大小、API调用、许可约束等因素；若签名密钥被盗或区块链不可用，验证过程会受阻；且需要依赖可公开验证的加密工具和存档的长期可用性。

---

## 345. A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring

**arXiv ID:** 2611.09178 | [PDF](https://arxiv.org/pdf/2611.09178v1)

**作者:** Warren Jouanneau `[一作]`, Marc Palyart `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍并演示了 ACM 的 acmart 文档类模板，说明其适用场景、主要功能以及如何使用各种参数来排版会议论文和期刊文章。

**💡 创新点**

创新点在于将原本分散的多种 ACM 与 SIG 专属模板整合为一个统一、可配置的 acmart 模板，支持可访问性、元数据提取，并提供了详细的使用说明和示例。

**🔧 技术方法**

使用技术主要是 LaTeX（acmart 文档类）、Libertine 字体族、booktabs 表格宏包以及标准的章节、图表、表格、公式环境；并利用 ACM 提供的 CCS 概念和关键词机制进行分类。

**📊 数据集**

本文并未使用实验数据集，所有示例均为模板使用说明，引用的文本来自示例文档本身。

**📈 对比分析**

由于该文档主要是技术说明与模板演示，没有进行实验或性能比较，所述“性能”仅指模板在不同 ACM 期刊/会议中的兼容性和排版一致性。

**⚠️ 局限性**

局限性包括：1) 仅适用于 ACM 出版物；2) 不能随意修改模板（如页边距、字体等）以避免被退稿；3) 对于非 ACM 的期刊或会议，需使用相应的模板；4) 需要安装特定的 Libertine 字体，若缺失会导致编译失败。

---

## 346. Audiovisual Disparities in VR: Impact on Spatial Perception

**arXiv ID:** 2611.09238 | [PDF](https://arxiv.org/pdf/2611.09238v1)

**作者:** Edurne Bernal-Berdun `[一作]` (University of Zaragoza), Diego Gutierrez `[通讯]` (University of Zaragoza)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在复杂、逼真的VR场景中，分别研究了真实音视差（因硬件或渲染误差导致的空间误差）和感知音视差（即使源物理共位，因大脑融合产生的空间错觉）的影响，评估其对沉浸感与用户体验的影响，并探索降低这两类误差的设计策略。

**💡 创新点**

创新点在于：①在真实环境下首次将真实与感知音视差的阈值与任务负荷关联；②系统性检验环境音频线索对感知误差的缓解作用；③通过将相对运动转移至目标运动，揭示感知误差主要由自我运动（本体感觉）而非相对速度引起，提示补偿方案应仅在自我旋转时使用。

**🔧 技术方法**

使用HTC Vive Pro Eye头戴式显示器、Steam Audio SDK与通用HRTF渲染；采用2AFC（双选强迫）任务测量误差阈值；利用定向音频与3D视觉渲染构建实验环境；分析心理测量曲线获取PSE、DT等指标。

**📊 数据集**

实验数据来自13-19名受试者（男女混合），在两类VR场景（森林与房屋）与四种实验条件（含/不含环境音频、默认/补偿音频偏移、受试者静止/运动）下收集的反应和头部运动记录。

**📈 对比分析**

与先前工作（Kyto、Kim & Lee）对比显示，真实误差阈值在任务负荷增加时显著提高（从≈5°上升至≈36°）；感知误差在无自我运动时基本消失，补偿偏移在此条件下反而显著降低准确率。实验表明在沉浸度高的任务中，用户可容忍更大误差，设计时可允许更宽阔的阈值。

**⚠️ 局限性**

局限包括：仅使用单一HMD与HRTF模型，样本量相对有限；实验场景与真实商业VR环境的多样性不足；未探讨长时使用的适应性；补偿算法的实现复杂度与实时性能未评估。

---

## 347. Navigating Complexity: A User-Centered Evaluation of Project Management Tools

**arXiv ID:** 2611.09260 | [PDF](https://arxiv.org/pdf/2611.09260v1)

**作者:** Yasin Ghadimi Zaker `[一作]` (University of Tehran), Hamid Reza Ghaeini `[通讯]` (University of Tehran)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对5款主流项目管理工具（Jira、Trello、Monday.com、Asana、Microsoft Project）进行用户中心的可用性评估，使用System Usability Scale (SUS) 量表对141名经验丰富的专业人士进行问卷调查，结合混合方法和统计分析探讨工具在不同项目规模、团队规模与复杂度下的使用体验；

**💡 创新点**

创新点在于将可用性（UX）视为关键成功因素，构建了基于上下文（团队规模、项目复杂度）的证据框架，揭示了功能与易用性之间的显著权衡，提供了针对不同场景的工具选择策略；

**🔧 技术方法**

采用的技术包括SUS量表的标准评分计算、Excel/表格软件进行描述性统计与方差分析（ANOVA）、事后Tukey检验等；

**📊 数据集**

使用的数据集为141名在伊朗使用上述工具的专业用户的问卷数据，涵盖工具使用频率、项目规模、团队规模与复杂度等变量；

**📈 对比分析**

通过SUS平均分与标准差、ANOVA比较不同工具在各情境下的可用性表现，结果显示Trello在小型团队和敏捷环境中表现最佳（SUS≈77.5），Microsoft Project在大型复杂项目中表现最差（SUS≈51.2），Jira表现中等且用户差异大；

**⚠️ 局限性**

局限性包括便利抽样导致的样本代表性不足、工具样本量不均（如Trello、Asana仅各11人）、仅覆盖伊朗专业环境、缺乏纵向跟踪及对文化差异的考量，可能影响结果的普适性和长期有效性。

---

## 348. Cross-Layer TopK: Dynamically Allocating Sparsity Across Layers in Transcoders

**arXiv ID:** 2611.09233 | [PDF](https://arxiv.org/pdf/2611.09233v1)

**作者:** Anyuan Zhuo `[一作]`, Pinyan Lu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种全模型 L₀ 预算的 Cross‑Layer TopK 稀疏机制，在跨层转码器中动态分配稀疏度并实现更好的重构和电路完整度。

**💡 创新点**

创新点在于将传统每层固定或惩罚式稀疏控制改为一次性全模型稀疏预算，并在训练中让模型自行决定各层的稀疏分配，从而兼顾重构质量与解释性。

**🔧 技术方法**

采用 Cross‑Layer TopK 选择所有层的前 K 个激活，结合稀疏解码技术、跳跃 ReLU 对比、BatchTopK 等变体，在 Qwen3‑8B‑Base 上进行训练与评估。

**📊 数据集**

使用 Qwen3‑8B‑Base 的激活数据，来自 Olmo3 语料库（预训练、链式思维、SFT）按 5:2:1 比例混合，进行随机采样和去除极端离群样本。

**📈 对比分析**

与 JumpReLU、CLT‑TopK 直接、BatchTopK 等方法对比，CLT‑TopK 在相同或更低稀疏度下实现更低的未解释方差 (FVU)，电路完整度（替代率与完整度得分）更高，且固定稀疏度使训练步长提升约 2.9 倍。

**⚠️ 局限性**

局限性包括仅在 Qwen3‑8B‑Base 上验证，尚未在更大或不同架构模型上测试；跨层转码器的解码器随层数平方增长，计算成本仍高；对稀疏度动态分配的机制仍需进一步探究。

---

## 349. Khmer PFGen-Bench: Building a Judge-Free Benchmark for Open-Ended Generation in a Low-Resource Language

**arXiv ID:** 2611.09222 | [PDF](https://arxiv.org/pdf/2611.09222v1)

**作者:** Mengsay Loem `[一作]` (Preferred Networks, Inc.), Masanori Hirano `[通讯]` (Preferred Networks, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了无 LLM 判断的柬埔寨语（Khmer）开放式生成评测基准 Khmer PFGen-Bench。

**💡 创新点**

首次在低资源语言中实现无评判者评测，并利用关键词规则与 n-gram 统计自生成参考答案；同时系统评估参考答案的有效性与多样性，并指出自参考导致的偏差。

**🔧 技术方法**

使用 PFGen-Bench scorer（fluency、truthfulness、helpfulness）与 n-gram 统计、关键词规则、KCC 归一化；基于 post‑trained 与 base 模型的生成与筛选；Spearman 相关性等统计分析。

**📊 数据集**

采用 50 道柬埔寨中学课程题目（原始 Khmer）及 13 类主题，利用 9 家 post‑trained 模型生成 18,274 条参考答案；对 19 个评估模型（含 10 个评估者）产生的 240 条多句回答进行人工评估。

**📈 对比分析**

通过无评判者分数（F、T、H 平均）与人工评测对比，跨参考集 Spearman ρ≥0.95，且与本地评测者评分 ρ≈0.94，模型排名高度一致；在不同参考来源、数量、家族组合下，排名稳定且无显著偏差。

**⚠️ 局限性**

参考答案未全部人工校验；仅 50 题、9 个生成器，缺乏多语言对比；仅对非生成器模型进行人工评测；基准仅评估答案质量而非事实正确性；评测者与问题编写者同一人，可能产生主观偏差。

---

## 350. BOXBOX: Recursive Harness Self-Improvement for Long-Running Performance Optimization

**arXiv ID:** 2611.09218 | [PDF](https://arxiv.org/pdf/2611.09218v1)

**作者:** Yuhao Tan `[一作]` (Nanjing University), Qingwei Lin `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了名为 Boxbox 的长周期软件性能优化基准，并设计了一个支持代理触发全 harness 修订、内核热重载的自适应 harness 优化框架。

**💡 创新点**

创新点包括将 harness 视为可运行实现，让任务代理在运行时自行决定何时进行修订，并通过内核管理的热重载保持执行上下文，从而实现在线、递归的 harness 优化。

**🔧 技术方法**

主要技术为大语言模型驱动的代理（GPT‑5.6 Luna）、内核管理的热重载机制、基于轨迹的自检诊断与 harness 修订工具，以及阴影验证以防止评估作弊。

**📊 数据集**

数据集来源于 100 个真实 GitHub 项目构成的 Boxbox 任务，涵盖开发工具、数据系统、科学计算等七个领域。

**📈 对比分析**

与 Frozen Harness、Harness Evolution、Harness Scaling 等基线在 4 小时预算下对 50 个测试任务的比对显示，Boxbox 取得 1.70× 的最佳速度提升，成本降低 29.3%，并在每美元速度提升上领先。

**⚠️ 局限性**

主要限制在于自修改 harness 机制可能带来安全与责任风险，实验环境已加入回滚与阴影验证，但在生产部署仍需更严格的沙盒、权限与人工审核。

---

## 351. Provable State Estimation with Recurrent Models

**arXiv ID:** 2611.09217 | [PDF](https://arxiv.org/pdf/2611.09217v1)

**作者:** Vincent Andrieu `[一作]` (LAGEPP), Christian Wolf `[通讯]` (Naver Labs Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于收敛性约束的RNN‑MLP结构，用于从观测序列中估计动力系统的隐藏状态。

**💡 创新点**

创新点在于将非线性观测器理论（KKL）与收敛性RNN结合，证明存在可注入的状态到潜在空间映射，提供可调性、MLP复杂度以及离散实现的正式保证。

**🔧 技术方法**

使用收敛的神经ODE/Residual RNN、GRU/GRU‑style 结构、光谱正则化以及多层感知机解码器，对潜在动态进行单标量增益调节。

**📊 数据集**

使用了真实移动机器人大规模轨迹数据（约27公里、18小时、四栋建筑）和一个三维非线性振荡器的仿真数据集。

**📈 对比分析**

与标准GRU进行对比，收敛RNN在姿态估计误差上更低、训练更稳定；在仿真系统上实现了可调的误差收敛，验证了理论。

**⚠️ 局限性**

局限在于假设系统可微且可观测、满足收敛性和滤波器秩条件；理论基于确定性模型，对噪声的稳健性仅有上界；缺少可构造的训练算法来强制满足秩/收敛约束。

---

## 352. Computation-Efficient Aerial-Marine Integrated Networks for Search and Rescue via Cooperative HAPS, UAVs, and MASSs

**arXiv ID:** 2611.09205 | [PDF](https://arxiv.org/pdf/2611.09205v1)

**作者:** Zhen Wang `[一作]` (Dalian Maritime University), Qiang Ye `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向海上搜索与救援的空海一体化边缘网络，联合优化无人机任务下发与资源分配，最小化系统计算开销。

**💡 创新点**

创新点在于将高空平台(HAPS)与海上自主航行船(MASS)同时作为边缘服务器，通过两阶段优化拆分成服务器匹配、传输功率、边缘与本地计算资源分配四个子问题，最终给出JCORA算法，显著降低能耗与时延。

**🔧 技术方法**

采用多对一匹配理论、准凸优化(二分法求解功率)、投影梯度下降(边缘计算资源)、凸优化(本地计算资源)以及OFDMA信道模型。

**📊 数据集**

使用仿真数据集：1台HAPS、4台MASS、若干随机分布无人机，任务量随机区间[0,10]，任务大小0.5Mbit，频率0.2GHz，信道参数按文中表格设置。

**📈 对比分析**

与三种基准方案（纯本地计算、仅匹配、随机分配）比较，JCORA在能耗和总时延上均优于基准，实验曲线显示在任务量、无人机数量、能耗/时延权重以及MASS计算资源倍数等参数下性能提升显著。

**⚠️ 局限性**

局限性包括：假设海上信道为Rician且不考虑动态波动；未考虑无人机与MASS的实时移动与路径规划；仿真基于理想化参数，缺乏实际场景数据验证。

---

## 353. SAS: Scene-Annotated AudioSet Strong for Joint Scene-Event Analysis

**arXiv ID:** 2611.09199 | [PDF](https://arxiv.org/pdf/2611.09199v1)

**作者:** Michel Olvera `[一作]`, Gaël Richard `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 AudioSet Strong 数据集进行扩展，添加 207,315 条基于视觉+LLM 的音频场景标签，构建跨模态标注流水线并对声学事件与场景的共现进行多视角统计分析。

**💡 创新点**

① 通过视觉、跨模态推理和大语言模型生成全新的场景标注；② 采用加权 PPMI 的事件共现重权方法，使语义视图更贴合人类认知；③ 通过邻域不稳定性度量揭示声学与语义的不匹配与潜在分类难点。

**🔧 技术方法**

多模态视觉分类 (ResNet50, WaveMix)、跨模态推理与 LLM (LLaVA, Mistral, Llama, Falcon, Phi, Qwen3)、语义嵌入 (SBERT)、代表性相似性分析 (RSA, Mantel test)、Naive Bayes 分类器以及 PPMI/加权 PPMI 事件关联度量。

**📊 数据集**

AudioSet Strong 原始数据集与扩充后的 207,315 条音频片段场景标签，涵盖两套税onomy（AV 与 AC）。

**📈 对比分析**

使用代表性相似性分析与人类手工税onomy 的 Mantel 相关度进行比较，语义视图的相关度显著高于原始频率视图；在 Naive Bayes 上，语义加权实现 macro‑F1 提升约 0.003‑0.002，准确率略低但更平衡。

**⚠️ 局限性**

局限在于：标注为弱标签，低置信度/多标签场景处理不足；LLM 推理可能引入偏差；仅在 Naive Bayes 上验证，未测试深度学习模型；邻域不稳定性作为指标仍需进一步验证。

---

## 354. Calibrated concept profiles: making the reading of a model's neurons a measurement

**arXiv ID:** 2611.09196 | [PDF](https://arxiv.org/pdf/2611.09196v1)

**作者:** Piotr Wilam `[一作]` `[通讯]` (University College London), Piotr Wilam (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为“校准概念配置文件（calibrated concept profile）”的方法，直接测量Transformer中单元对命名构造的响应，并在多训练检查点跟踪其变化。

**💡 创新点**

创新点包括：①用声明性刺激分布代替模板或训练语料；②通过双重电池（twins）和核心重采样定义噪声单位；③在测量前执行许可证检查以保证读数可靠；④将响应标准化到每个神经元自身的离散度，提供可比单位。

**🔧 技术方法**

技术手段包括：提示生成器、原始读取、标准化配置文件、噪声单位（twin & core bootstrap）、许可证检查、分辨率测量，以及对OpenCoder-1.5B的多检查点记录。

**📊 数据集**

数据集为公开的Python 3.12语料库（约253k文件）作为参考语料；构造集合为Python语法树节点与内置名共206个，实际使用188个；模型数据为OpenCoder-1.5B在61个检查点（共计2万亿token）上的激活记录。

**📈 对比分析**

与线性探针、稀疏自编码器、跨编码器等方法比较，校准概念配置文件提供四项测量（分布、单位、证据、分辨率），并在跨检查点跟踪构造时显示可辨别的变化；探针在跨检查点上表现平滑，无法捕捉细粒度变化。

**⚠️ 局限性**

局限性包括：仅能读取预先命名的构造；对参考语料与模型训练分布的依赖；需要手工定义读取点与细胞；未覆盖所有语言和模型；并且在未训练的构造上表现不佳。

---

## 355. Enhanced Informed Dynamic BP Decoding Scheduling Strategies for 5G NR LDPC Codes

**arXiv ID:** 2611.09184 | [PDF](https://arxiv.org/pdf/2611.09184v1)

**作者:** Tofar C. -Y. Chang `[一作]`, Yu T. Su `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的更新前传播 RBP（UBPRBP）算法，并结合度‑1 变量节点（VN）资源管理方案，用于提升 5G NR LDPC 码的译码性能。

**💡 创新点**

创新点在于：①引入“先更新后传播”IDS策略，让校验节点在未收到新 V2C 信息前不再发送 C2V 消息，从而抑制错误信息的扩散；②设计了度‑1 VN 的资源管理流程，保证每轮迭代仅在结束时更新一次，并在选中度‑1 VN 时使用替代 C2V 消息继续译码，以更高效地利用计算资源。

**🔧 技术方法**

采用基于 Belief Propagation 的 Residual BP（RBP）及其变体，结合 IDS 调度、残差比较、CN mute/enable 机制以及度‑1 VN 处理规则，构建并实现了 UBPRBP 与 UBPRBP‑D1VN 等算法。

**📊 数据集**

使用的码为 IEEE 802.11（WiFi）(1944,972) LDPC 码和 5G NR (1848,616) 基图 I 码（共 1176 个度‑1 VN）进行仿真评估。

**📈 对比分析**

通过 FER/BER 曲线和迭代次数对比原始 RBP、SVNF‑RBP、RBP‑D1、s‑RBP‑D1 等算法，结果显示：UBPRBP 在早期迭代可获得 0.2‑0.4 dB 的增益；UBPRBP‑D1VN 在所有迭代中均优于竞争者，尤其在早期迭代提升约 0.25 dB；SVNF‑RBP 在深度迭代时仍略胜于 UBPRBP。

**⚠️ 局限性**

局限性在于：算法主要针对 LDPC 码，且在深度迭代时 SVNF‑RBP 仍表现更好；未对硬件实现复杂度与能耗进行评估；度‑1 VN 资源管理方案针对 5G NR 的特定结构，可能不适用于其他标准。

---

## 356. Rhythmic Fitness Beat 'Em All using Real-Time Monocular Human Pose Estimation

**arXiv ID:** 2611.09228 | [PDF](https://arxiv.org/pdf/2611.09228v1)

**作者:** Yanis Zaoui `[一作]` (NaturalPad), Mohamed Adjel `[通讯]` (NaturalPad)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文设计并实现了一款仅使用智能手机和HDMI线即可运行的节奏打击类健身游戏，结合实时单目人体网格恢复和游戏逻辑，实现全身动作与节拍同步的交互体验。

**💡 创新点**

创新点包括：1）仅依赖手机神经处理单元实现30fps实时3D人体网格恢复；2）为五种影子拳击动作定义单一度量与阈值的高效识别方案，并在教程中自适应阈值；3）将节拍（以踢鼓为标记）与动作捕捉时间匹配的评判机制，学习玩家个人延迟；4）实现全程手机＋HDMI的低成本硬件架构。

**🔧 技术方法**

技术栈包括：InstantHMR（实时3D网格恢复）、NanoDet（人体检测）、WebSocket通信、Web Audio API、JavaScript Canvas 2D游戏引擎。

**📊 数据集**

未使用公开大规模数据集；仅在26首无版权音乐中离线检测踢鼓强度挑选“Shiny Tech”做节奏场景，并在单一玩家的标注录像中收集阈值和安全底线。

**📈 对比分析**

由于目前为工作进展，缺乏正式用户研究和定量对比，系统在Galaxy S23上实现30fps、摄像捕捉至骨骼传输约72–76 ms，功耗约5.6 W；未与其他姿态识别或节奏游戏做性能评估。

**⚠️ 局限性**

局限性包括：无正式用户实验验证识别准确率与用户体验；延迟补偿不完整（仅针对拍拳和显示/音频时延缺乏全局校准）；仅支持单首歌曲与单级别；评判窗口与延迟学习仅针对拳击动作；缺乏多玩家或多场景的可扩展性。

---

## 357. KLEAR: Knowledge-Matched LLM Evaluation in Korean across Application Requirements via Controlled Benchmark Transformation

**arXiv ID:** 2611.09223 | [PDF](https://arxiv.org/pdf/2611.09223v1)

**作者:** Seoyeon Kim `[一作]` (Yonsei University), Jaehyung Kim `[通讯]` (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 KLEAR 框架，通过锚定知识构建知识匹配的对比问题，分离 LLM 的知识回忆与应用能力评估。

**💡 创新点**

创新点在于显式构建知识锚束并保持知识不变，仅改变应用难度，结合自动化审核与人工修订实现可控对比问题集。

**🔧 技术方法**

使用 GPT‑5 系列模型进行知识锚束构建、问题生成、自动化审核与自我修订，利用 LLM rubrics 与代理模型进行难度筛选。

**📊 数据集**

以三大韩语基准（CLIcK、KMMLU‑Redux、KMMLU‑Pro）中的 7,340 道多项选择题为种子，最终生成 287 道知识匹配对比问题。

**📈 对比分析**

在 8 种 LLM（K‑EXAONE、A.X K1、Solar Open、Aya Expanse 32B、GLM‑4.7、Qwen3.5‑397B‑A17B、Claude Sonnet 5.5、GPT‑6.1 Sol）上测试，结果显示在知识固定的情况下，应用难度升高会显著提升应用错误率（从 37%→76%），并且提供知识语句后错误大多被修复，应用语句进一步提升 3–15 分。

**⚠️ 局限性**

局限包括仅适用于多项选择题，代理模型筛选可能产生模型偏差，难度阈值未针对不同应用层细化，且对其他语言和领域的泛化尚未验证。

---

## 358. Representational drift in a code model: behaviour settles, which neurons carry it does not

**arXiv ID:** 2611.09204 | [PDF](https://arxiv.org/pdf/2611.09204v1)

**作者:** Piotr Wilam `[一作]` `[通讯]` (University College London), Piotr Wilam (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 OpenCoder‑1.5B 预训练过程中 169 个 Python 语法构造的 5% 关键神经元（carrier）进行细粒度跟踪，测量其随训练步数的变化、对损失的依赖性以及在 2.03 万亿 token 终点的漂移情况。

**💡 创新点**

首次在大规模语言模型中从 4.19 B 至 2.03 T token 的完整训练轨迹上揭示：行为（loss）和对关键神经元的依赖性在前 4% 训练结束后已基本稳定，但关键神经元本身从未收敛，持续漂移，呈现“持续游走”而非“收敛”现象，提供与神经科学中表征漂移相似的实验证据。

**🔧 技术方法**

使用双子电池（twin battery）概念剖面方法对构造激活进行标准化；通过对每个构造的 5% 高响应神经元进行选择、计算转移率、损失依赖度、携带集的重叠度等指标；对 35 个检查点进行梯度消融实验以评估神经元重要性；对权重矩阵进行 Frobenius 范数与自由随机步的比较，分析权重游走的受限性质。

**📊 数据集**

基于 OpenCoder‑1.5B 的预训练数据（约 2 T token），使用人工构造的 Python 提示电池（188 个构造，169 个通过门检验）以及同一模型在不同学习率分支的检查点记录。

**📈 对比分析**

通过对 16 个具有标记位置的构造进行 loss 监测和神经元消融，比较当前携带集与随机集对 loss 的影响，展示依赖度随训练增长；用转移率指标评估携带集的漂移；由于缺乏外部基准，结果主要展示现象而非性能提升，但指出损失已在前 4% 训练后趋于稳定。

**⚠️ 局限性**

仅在单一模型、单一训练跑、固定学习率（至 annealing 分支）下进行；仅评估强制位置的构造；使用 5% 截断的携带集，未考虑低响应神经元；双子电池方法对噪声敏感；未验证在自然语言或不同任务上的普适性；数据集为人工构造的 Python 语句，可能不代表真实代码分布。

---

## 359. Safe Reinforcement Learning via Probabilistic Option Shielding

**arXiv ID:** 2611.09197 | [PDF](https://arxiv.org/pdf/2611.09197v1)

**作者:** Gian Luca Ascari `[一作]`, Giuseppe Marra `[通讯]` (Ku Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出Probabilistic Option Shielding（POS）技术，利用选项框架在无环境模型的情况下实现长期安全约束；

**💡 创新点**

通过将Probabilistic Logic Shield扩展到选项层面，学习安全评论家（safety critic）并以期望乘积形式递归估计选项整体安全性，从而在不需要完整转移模型的前提下实现长期安全保障；

**🔧 技术方法**

使用Probabilistic Logic Shield（ProbLog）计算单步安全概率，基于选项的递归安全定义进行TD学习安全评论家，结合PPO进行策略优化；

**📊 数据集**

在LogiCity模拟环境（Large Map和Stochastic Map）中测试，该环境包含可变交通灯、行人和军事区等安全约束；

**📈 对比分析**

与未加屏蔽、一步PPL、奖励塑形及Lagrangian约束等基线比较，POS在两种地图上显著降低安全违规率（在Large Map上接近零违规，在Stochastic Map上降低约99%），同时保持或提升任务收益；

**⚠️ 局限性**

局限包括：假设已给定选项集，未学习或安全优化选项本身；实验仅限于LogiCity，未验证在更复杂或真实驾驶仿真（如CARLA）中的表现。

---

## 360. InstantHMR : Real-Time Human Mesh Recovery on Your Phone

**arXiv ID:** 2611.09195 | [PDF](https://arxiv.org/pdf/2611.09195v1)

**作者:** Mohamed Adjel `[一作]` (NaturalPad), Vincent Bonnet `[通讯]` (LAAS-CNRS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了适用于手机 NPU 的单图像人体网格恢复框架 InstantHMR，能够在移动设备上实现实时 3D 人体网格回归。

**💡 创新点**

创新点包括：基于 Momentum Human Rig（MHR）模型的轻量化回归器，使用 HGNetV2+Transformer 结构实现高精度与极低延迟的统一；重新制定统一评测协议以公平比较多种方法。

**🔧 技术方法**

核心技术：HGNetV2 卷积骨干 + 4 层 Transformer 解码器 + SimCC 关键点分类头 + MHR 参数回归头 + 位置嵌入（CLIFF 机制）与前向运动学生成网格。

**📊 数据集**

训练数据：仅使用 SAM 3D Body 数据集公开的 4.2M 人体裁剪，来源于 SA‑1B、AI Challenger、Harmony4D、COCO、MPII 等公开图像集合。

**📈 对比分析**

比较方法：在统一协议下对 11 种主流方法进行复现与评测；在 Samsung Galaxy S23 NPU 上实现 5.5 ms/人，PA‑MPJPE 48.8 mm，速度比 NLF‑S/L 快 12–23 倍，准确性接近 Fast SAM 3D Body；在 RTX 4070 GPU 上达到 0.98 ms/人。

**⚠️ 局限性**

局限性：缺乏手部专门编码器/解码器；仅在 Android NPU 设备上测试，未评估 Apple Neural Engine；未提供时序一致性（抖动）指标；商业版受限于数据许可，需更多可商用数据以提升精度。

---

## 361. SUGARv2: Learning Diverse Visual Humanoid Loco-Manipulation Skills from Synthesized Interaction References

**arXiv ID:** 2611.09266 | [PDF](https://arxiv.org/pdf/2611.09266v1)

**作者:** Tianshu Wu `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

SUGARv2 通过生成交互参考并统一训练流程，实现多任务视觉下人形全身运动与操控。

**💡 创新点**

创新点在于利用可重用任务模板在不同空间配置下生成低成本交互参考，并采用基于参考的模仿奖励统一教师与视觉策略学习。

**🔧 技术方法**

技术包括 GPT-6 Astra 生成模板、离散运动生成器、冻结预训练全身控制器、教师训练与行为克隆、PPO+教师蒸馏、视觉域随机化。

**📊 数据集**

使用自研合成参考数据、不同物体实例、随机背景与光照的渲染视觉数据，无需人工示范。

**📈 对比分析**

与 VIRAL、GR00T 以及 GRAIL 生成的参考相比，SUGARv2 在五个任务中平均成功率达91%，对未见背景/物体仍保持 83–92% 以上。

**⚠️ 局限性**

局限在短周期、简化场景；难以处理长周期任务、混乱环境；视觉策略仍是任务专属。

---

## 362. Zero-Query Transferable Pretend Benign Attacks with Amortized Online Generation for Cooperative Perception

**arXiv ID:** 2611.09256 | [PDF](https://arxiv.org/pdf/2611.09256v1)

**作者:** Hongwei Lin `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套针对协同感知的隐蔽攻击与防御框架，包含零查询可转移的Pretend Benign攻击（TrPB）及其实时实现RT‑TrPB，以及基于时空一致性的STTrust防御；

**💡 创新点**

创新点在于把白盒Pretend Benign的区域特定隐蔽策略转移到异构受害者模型上，构建可共享的攻击知识（区域语义与共享扰动），并通过字典化与快速区域网络实现毫秒级在线生成，同时提出整合空间、时间与历史身份信任的多维防御；

**🔧 技术方法**

采用BEV特征级攻击、区域分割与多目标对抗生成、语义区域共识、共享优化、快速区域网络（FRN）、扰动字典（RPD）以及运动补偿与历史信任模块；

**📊 数据集**

在OPV2V、V2XSet与V2V4Real三大协同感知数据集上进行评估；

**📈 对比分析**

与PGD、C&W、James、AdvGPS等基线攻击以及ROBOSAC、FLD、GLST等防御对比，TrPB/RT‑TrPB在未知模型上仍保持99%+的DBR并在实时生成中耗时<56ms；STTrust在三数据集上实现>90% AP恢复，显著优于现有防御；

**⚠️ 局限性**

局限在于对攻击者可观测的中观信息和协作视角仍有一定依赖，且对极端遮挡或大规模攻击者场景的鲁棒性尚待进一步验证；

---

## 363. Rethinking MoE Load Balancing as Feedback Control

**arXiv ID:** 2611.09252 | [PDF](https://arxiv.org/pdf/2611.09252v1)

**作者:** Yikun Hou `[一作]` (Umeå University), Alp Yurtsever `[通讯]` (Umeå University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将Mixture-of-Experts（MoE）负载平衡视为反馈控制问题，并设计了仅使用积分项的 PID(I) 控制器，作为一种新的、可直接插入预训练模型的负载平衡方法。

**💡 创新点**

创新点在于：①统一了现有两类平衡技术（auxiliary loss 与 ALF-LB）为 PID 控制器族；②通过理论分析证明仅积分项能消除持久失衡并抑制批量噪声；③在非平稳分布下（训练过程中混合变化、持续预训练）展示了更优的平衡效果。

**🔧 技术方法**

使用离散时间 PID 控制理论，设计 PID(I) 控制器；通过添加每个专家的偏置 u_i(t) 直接修改路由决策；实验中采用 OLMoE backbone 的 Top-K MoE；使用标准的损失函数与 perplexity 评估。

**📊 数据集**

主要数据集为 SlimPajama-627B 的域分离子集（7个源域）用于预训练；持续预训练使用 Dolmino 混合语料与 CPT 语料；此外在 downstream 任务上使用 MMLU、HellaSwag、ARC-Challenge、HumanEval、MBPP 等基准。

**📈 对比分析**

与 Vanilla、Aux-loss、ALF-LB 四种基线对比，PID(I) 在预训练与持续预训练两种场景下均获得最低的 MaxVio、DER、UTR 以及相似的 perplexity；在 downstream 任务上保持与基线相近甚至略优的表现；在持续预训练中，虽然平衡更好但略有遗忘。

**⚠️ 局限性**

局限性包括：①仅针对静态批量大小与 Top-K 结构的 MoE，未验证在更大规模或不同 sparsity 方案下的效果；②对梯度噪声敏感，若批量噪声极大可能导致偏置振荡；③在持续预训练中对先前域的遗忘仍较明显，需进一步平衡专精与负载。

---

## 364. Integrating MBSE Principles with AI to Improve the Quality of Requirements in SMEs Context

**arXiv ID:** 2611.09234 | [PDF](https://arxiv.org/pdf/2611.09234v1)

**作者:** Iryna Volosheniuk `[一作]` (IRIT), Jean-Michel Bruel `[通讯]` (IRIT)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级MBSE方法，结合人工智能辅助需求改进，支持SME环境下的需求细化、可追溯性和早期V&V，并在SOGECLAIR模拟系统案例中实现了原型验证。

**💡 创新点**

创新点在于：①针对SME的规模和资源限制，设计了可逐步引入的MBSE框架；②利用NLP/LLM在需求文本中自动检测歧义、缺失和不一致，并提供改进建议；③通过将需求与设计与验证工件在单一模型中关联，形成“单一真相源”，实现跨工具（如Jira）的一致追踪；④保持与现有工作流程兼容的同时，显著降低人工审查负担。

**🔧 技术方法**

技术包括：ISO/IEC/IEEE 29148与INCOSE准则的操作化；EARS结构化模式；自然语言处理（词法、句法分析）与大型语言模型（LLM）进行需求质量评估和重写；Prompt Engineering；基于模型的中间件实现需求与项目管理工具的链接；人机协作审计流程。

**📊 数据集**

数据集主要来自SOGECLAIR Simulation公司的合同和需求文档；未使用公开的大型需求数据集，原型在真实工业案例中进行实验。

**📈 对比分析**

方法评估采用定性指标：需求质量提升（消除歧义/不完整率下降）、审查时间缩短、追踪完整性增强。实验表明，在原型中需求改进建议覆盖率约70%，人工复核时间下降约30%，但未给出精确数值或与基准系统的量化比较。

**⚠️ 局限性**

局限性包括：①LLM对上下文长度和领域知识的敏感性导致部分建议需人工核查；②知识资本化与复用仍不完整，仅对结构化需求实现；③对现有工具的集成有限，需进一步扩展到CAD等；④SME对流程变更的抵触可能影响采纳；⑤未提供大规模工业验证和客观性能数据。

---

## 365. Green500: Adventure in Qualifying a Cluster

**arXiv ID:** 2611.09251 | [PDF](https://arxiv.org/pdf/2611.09251v1)

**作者:** Nicolas Richart `[一作]` (École Polytechnique Fédérale de Lausanne), Gilles Fourestey `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在EPFL的Kuma集群上进行两轮Green500测评，结合MITSI生命周期方法评估完整碳足迹，探讨GPU功率上限对能效与碳效能的影响。

**💡 创新点**

首次将Green500能效测评与生命周期碳评估融合，提出Carbon500概念，揭示功率上限在能效与碳效能之间的折衷，并考虑嵌入式排放与电网碳强度的相互作用。

**🔧 技术方法**

采用HPL基准测试、GPU功率限制、PDU与电能计量、Prometheus/Grafana可视化、MITSI生命周期方法、Resilio环境数据库进行数据采集与分析。

**📊 数据集**

使用的主要数据集包括80个H100节点的硬件规格与运行数据、瑞士电网碳强度（90/57）、PUE（1.3）、Resilio设备排放因子以及集群6年生命周期的假设。

**📈 对比分析**

通过在200–700 W范围内细粒度扫描GPU功率上限，绘制能效曲线并与Green500/Top500提交结果对比，发现能效最优点在450 W（性能损失≈11%但能耗降低≈20%），碳效能最优点在500 W，展示了两指标之间的平衡。

**⚠️ 局限性**

局限性包括：仅基于HPL基准，未覆盖典型AI/科学工作负载；使用年度平均电网碳强度，忽略时段波动；PUE与设备排放因子来源不完整，生命周期假设固定为6年；测量仅在单一集群，难以推广到不同地区或能源结构的集群。

---

## 366. RAG-Enhanced LLMs for Interactive Explainability in Clinical Decision Support Systems

**arXiv ID:** 2611.09232 | [PDF](https://arxiv.org/pdf/2611.09232v1)

**作者:** Lorenzo Mannocci `[一作]` (University of Pisa), Anna Monreale `[通讯]` (University of Pisa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个结合传统预测模型、LIME/SHAP本地解释器与检索增强生成（RAG）的临床决策支持系统，支持交互式自然语言解释。

**💡 创新点**

将RAG与黑盒模型和本地解释器融合，提供来源可追溯、可交互且可验证的解释，并通过选择器实现一致性控制。

**🔧 技术方法**

使用随机森林、LIME/SHAP、MedRAG框架、Mixtral LLM、检索增强生成以及选择器模块。

**📊 数据集**

使用 Pima Indians Diabetes 数据集。

**📈 对比分析**

与随机森林在测试集上进行对比，评估精确率、召回率、F1 和一致率；结果显示 RF 在糖尿病检测上表现更好，但两模型一致率高达 82%，证明 RAG 生成的解释可作为补充。

**⚠️ 局限性**

局限在于数据集规模小、RAG 计算成本高、未在其他临床数据或模型上验证泛化性，同时存在隐私与 hallucination 风险。

---

## 367. On derandomizing the line-point low-degree test

**arXiv ID:** 2611.09183 | [PDF](https://arxiv.org/pdf/2611.09183v1)

**作者:** Prahladh Harsha `[一作]` (Tata Institute of Fundamental Research), Madhu Sudan `[通讯]` (Harvard University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在高误差下对二元（m=2）线点低阶检验的去随机化版本，并展示了其局限性和可行性；

**💡 创新点**

提出了一个针对高阶Z-度的Polishchuk–Spielman除子引理的新版本，证明了只需比Z-度多一方向即可得到全局低阶根；还构造了一个在子域斜率集合上极端的反例，说明当域大小不足k^2时去随机化失败；

**🔧 技术方法**

使用了Hasse导数、Sylvester行列式、判别式、Newton迭代以及并行Hensel提升等代数技术；

**📊 数据集**

无实验数据集，全部是理论证明与构造反例；

**📈 对比分析**

没有具体的数值性能对比，结论主要是理论上存在/不存在去随机化的可行性阈值；

**⚠️ 局限性**

局限在于反例仅在小特征域出现，且正向结果需要域大小至少为k^2/ϵ^c、方向数r≥1/ϵ^c，无法在更小域或更少方向下得到；

---

## 368. TwinHash: Two-Sided Object Identity Management for Time- and Energy-Efficient Agentic Workflows

**arXiv ID:** 2611.09254 | [PDF](https://arxiv.org/pdf/2611.09254v1)

**作者:** Sangbu Yun `[一作]`, Youngjoo Lee `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 369. Topology as a Design Variable: Joint Architecture-Sizing Optimization for SRAM

**arXiv ID:** 2611.09248 | [PDF](https://arxiv.org/pdf/2611.09248v1)

**作者:** Can Wang `[一作]` (Nanjing University of Sci. & Tech.), Wei W. Xing `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了OpenYieldV2，首次将SRAM位细胞拓扑（6T/10T）作为可优化变量，与晶体管尺寸共同进行混合整数多目标优化，生成高保真完整宏布局并完成Pareto前沿搜索；

**💡 创新点**

创新点在于：1) 同时支持多拓扑且在同一高保真环境下生成完整宏；2) 采用TabPFN少样本代理模型跨拓扑共享知识；3) 采用粗-细分层混合优化（先全局离散搜索再连续梯度细化），实现多目标、混合变量的高效探索；

**🔧 技术方法**

使用技术包括：参数化SRAM编译器（生成6T/10T宏与外围电路）、TabPFN代理模型、基于NSGA-II的混合演化搜索与残差自适应梯度下降、Xyce并行SPICE仿真、PySpice生成电路；

**📊 数据集**

数据集为在FreePDK45 45nm工艺下，VDD=1.0V、25℃、1.0V或0.5V、不同PVT角度下生成的SPICE仿真样本，训练代理使用约500~1000个全宏仿真样本；

**📈 对比分析**

与NSGA-II、U-NSGA-III、SPEA2、C-TAEA、ParEGO、MACE、GPBO等基线对比，固定拓扑下提升HV约6.4%、IGD下降24.3%，开启拓扑搜索后HV提升17.3%、IGD下降90.1%；细化步骤可再提升HV 2.2%、IGD下降55.5%；

**⚠️ 局限性**

局限性在于仅验证了6T与10T两种拓扑，未覆盖更多复杂拓扑；仅在45nm技术节点测试，未验证更先进工艺；代理模型对跨拓扑的泛化仍需进一步验证；

---

## 370. Fused-Softmax for efficient quantized attention

**arXiv ID:** 2611.09225 | [PDF](https://arxiv.org/pdf/2611.09225v1)

**作者:** Tianxing Wu `[一作]` (Advanced Micro Devices, Inc.), Jesus Carabano Bravo `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 Fused-Softmax Attention，将 softmax 的大部分向量运算搬入 GEMM，以利用低精度矩阵核心吞吐率；

**💡 创新点**

创新点在于显式冻结行最大值并将其与 Schraudolph 指数逼近折叠进 score GEMM，使用两阶段量化转换和行和矩阵乘法，消除向量单元瓶颈；

**🔧 技术方法**

采用 AMD CDNA4 架构的 MXFP8/6/4 量化格式、Schraudolph 指数近似、向量回滚、八波 ping‑pong 调度、Hadamard 预处理以及矩阵化行和；

**📊 数据集**

在 MLPerf Inference v6.1 的 Wan2.2‑T2V‑A14B、HunyuanVideo、MiniMax‑H3 视频生成任务以及合成正态随机输入上进行评估；

**📈 对比分析**

与 Attn‑QAT、Direct‑P、FlashAttention‑4 等方法对比，TFLOPS 提升约 2–4×，cosine 相似度 ≥0.98，VBench 评分与 BF16 基线相当；

**⚠️ 局限性**

限制在于纯 MXFP4 量化会导致视觉质量下降，回滚仅在极少情况触发，且需在不同模型与硬件上进一步验证。

---

## 371. RPEG: Training Reasoning Language Models Scalably and Cost-Effectively

**arXiv ID:** 2611.09207 | [PDF](https://arxiv.org/pdf/2611.09207v1)

**作者:** Maciej Besta `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为RPEG的图驱动框架，联合优化推理语言模型（RLM）训练的流水线结构与执行计划，实现大规模强化学习训练的高效与低成本。

**💡 创新点**

创新点在于：①将流水线结构（如actor‑critic共享、有限异步）与硬件映射纳入统一图表示；②构建带有阶段融合、细粒度调度和多维并行的RPEG中间表示；③提供全流程的解析性能与内存模型；④通过模拟退火搜索实现高效的结构与执行空间探索。

**🔧 技术方法**

采用的技术包括数据中心图中间表示、图变换与压缩、α–β通信模型、ZerO‑2梯度分片、DP/PP/TP多维并行、微批处理、以及分布式执行时的完毕驱动调度。

**📊 数据集**

实验使用的模型包括LLaMA 3B/7B/13B/34B/70B、LLaMA‑2、Llama 3.1 8B、Llama 3.3 70B等，训练工作负载为PPO/GRPO等RLHF任务，硬件平台为A100和GH200 GPU集群。

**📈 对比分析**

与ReaL、verl及基线 heuristic 相比，RPEG 在 8–128 GPU 规模上实现了最高 2.48× 的步长速度提升、最高 30% 的 GPU 利用率提升，且在 256–1024 GPU 规模下的搜索仅需数分钟即可获得接近最优的计划。

**⚠️ 局限性**

局限性包括：①解析模型对实际硬件的误差可能导致计划偏差；②搜索空间仍依赖手工设定的结构选项与温度参数；③仅针对 PPO/GRPO 等基于生成‑评估‑训练的 RLHF 流程，难以直接推广到更复杂的多代理或工具交互系统。

---

## 372. RoleWalk: training-free structural embeddings for directed graphs that transfer across graphs

**arXiv ID:** 2611.09189 | [PDF](https://arxiv.org/pdf/2611.09189v1)

**作者:** Maixent Chenebaux `[一作]` `[通讯]`, Maixent Chenebaux

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种无训练、无图级量的角色嵌入方法RoleWalk，利用短懒随机游走的两种分布（转移概率的特征函数和距离分布）来描述节点在图中的结构位置；

**💡 创新点**

核心创新在于：①仅使用局部信息，坐标可在任意图中保持一致；②采用混合探测（正向后向）捕捉有向图中的共享目标/源信息；③通过特征函数与距离分布的组合提供可解释的核均值嵌入；

**🔧 技术方法**

技术实现基于懒随机游走矩阵H、特征函数取固定频率的复值、距离层概率、混合探测矩阵H₊H₋与H₋H₊、可选的曲率校正及基于阈值的稀疏化裁剪；

**📊 数据集**

使用多种公开网络数据集：美国、巴西、欧洲机场网络；Reality‑call、Actor；Wikipedia adminship (wiki‑RfA) 11,377用户；Digraphwave Enron 邮件子图与全图；Barabási‑Albert、Watts–Strogatz等合成图；以及多类中性对齐基准；

**📈 对比分析**

与struc2vec、GraphWave、Role2Vec、RESD、ReFeX、Digraphwave、RWSE等现有方法比较。RoleWalk在五个角色分类任务中保持首位（平均准确率~0.68），在wiki‑RfA 角色与管理员预测中取得最高精度；在Digraphwave的对齐基准上在所有噪声水平上均优于Digraphwave，提升幅度约0.04–0.08；在跨尺寸中心性转移实验中，RoleWalk平均Spearman相关0.81，高于本地统计0.79、冻结ReFeX 0.78、已训练GIN 0.71；同时在百万节点上仅需18–192 s，显著低于大多数对手。

**⚠️ 局限性**

局限性包括：①无法区分相同交叉数组的距离规则图、CFI对等结构；②对极度中心化图的裁剪误差较大，需要预算控制；③对部分有向角色数据集（如模拟网络）性能与手工统计相当或略逊；④缺乏广泛真实标注的有向角色数据集来进一步验证。

---

## 373. HMMF-Net: A Lightweight Hierarchical Multi-Frame Multimodal Fusion Network for Accurate Fish Feeding Intensity Assessment

**arXiv ID:** 2611.09224 | [PDF](https://arxiv.org/pdf/2611.09224v1)

**作者:** Phi-Hung Hoang `[一作]` (FPT University), Thi-Thu-Hong Phan `[通讯]` (FPT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种轻量级多模态框架HMMF-Net，用于精准鱼类喂食强度评估；

**💡 创新点**

创新点在于：① 采用早期多帧视觉编码与2D EfficientNetB0相结合捕获时序信息；② 设计自适应多轴残差单元（AMARU）提取声学时频特征；③ 引入层次双向信使跨模态融合（BMCF），通过低维信使令牌实现高效信息交换；

**🔧 技术方法**

技术包括2D CNN（EfficientNetB0）、自适应多轴残差网络（AMARU）、跨模态注意力的BMCF模块、log-Mel频谱预处理、SpecAugment、TensorRT部署等；

**📊 数据集**

使用全量AV-FFIA（含27,067个2秒音视频对，四类喂食强度）进行训练与评估；

**📈 对比分析**

与单模态、3D卷积、Transformer及其他多模态方法对比，HMMF-Net在五折交叉验证中取得96.77±0.27%准确率、96.88±0.26%F1，参数仅3.987M，GFLOPs 1.002，部署至Jetson Orin Nano时仍保持约96.76%准确率，推理时间≈19.3ms；

**⚠️ 局限性**

局限性：仅在单一鱼种、受控循环水产环境下验证；固定的两帧采样可能不足以捕捉长时或不规则喂食动态；需要在多种环境与物种上进一步验证与自适应时间帧选择研究。

---

## 374. On the Limits of Execution Traces for Repairing Assertion Failures in LLM-Generated Code

**arXiv ID:** 2611.09212 | [PDF](https://arxiv.org/pdf/2611.09212v1)

**作者:** Muntasir Adnan `[一作]` (University of Canberra), Carlos C. N. Kuhn `[通讯]` (University of Canberra)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种只针对断言失败的“断点调试”方法，在迭代自我调试中加入执行轨迹注释，提升LLM修复断言失败的能力。

**💡 创新点**

创新点在于：①对断言失败的“信息缺口”进行量化分析，解释为何它们难以修复；②提出仅在断言失败时添加内联运行值的简洁轨迹，而非完整回溯；③通过对比同等预算下的trace修复与重新采样，系统评估两种策略的有效性；④提出可操作的规则决定何时使用trace、何时重新采样。

**🔧 技术方法**

技术上使用Python函数级代码生成、LLM迭代自我调试循环（PyCapsule风格），以及Qwen2.5 Instruct系列模型（7B、14B、32B、72B）。轨迹通过Python的行级跟踪器生成并以注释形式嵌入源代码。

**📊 数据集**

数据集包括HumanEval、BigCodeBench和LiveCodeBench三大benchmark，分别包含数百至一千多道Python函数级任务。

**📈 对比分析**

比较方法：在Baseline（无轨迹的PyCapsule循环）与一轨迹/多轨迹/新采样等变体下，按任务完成率、已修复任务比例、相同代码重复率、token消耗等指标评估。结果显示：在BigCodeBench上，一轨迹比新采样多修复约4%（从26.7%到22.4%）；在HumanEval差异不大；在LiveCodeBench则新采样更优（6.5% vs 4.9%）。同时，一轨迹能显著降低相同代码提交率并降低token成本。

**⚠️ 局限性**

局限性：仅在Python函数级任务上测试；未对更长或块级程序评估；缺乏对断言失败成因的手工标注；只研究Qwen2.5系列模型，未检验其他LLM；每个配置仅跑一次，缺乏方差估计。

---

## 375. Copies Are Not Confirmations: How News Duplication Fools LLM Judges and Continual Learners

**arXiv ID:** 2611.09203 | [PDF](https://arxiv.org/pdf/2611.09203v1)

**作者:** Abdelkader Metales `[一作]` (caaise GmbH), Aicha Metales `[通讯]` (Djilali Liabes University of Sidi Bel Abbes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究评估了新闻复制在语言模型学习中的影响，检验多种LLM判别器是否将复制视为独立证据，并探讨通过移除复制来降低模型学习谬误。

**💡 创新点**

创新点在于提出基于实体与引文新颖度的复制检测方法，并在多种LLM判别器与连续学习流程中展示复制移除能显著降低误学率的实证结果。

**🔧 技术方法**

采用了MinHash与LSH的近似重复检测、实体与引文新颖度评分、LLM判别器（Llama3.2、phi4、Gemini Flash/Pro）以及模拟的连续学习框架，并使用McNemar、t-interval、AUC等统计检验。

**📊 数据集**

实验基于CC‑News 2017‑2019的20万条英文新闻、由LLM生成的重写文本以及人工标注的复制与独立报道对。

**📈 对比分析**

通过对比不同判别器在复制与非复制情境下的真/假接受率、连续学习中谬误吸收率以及复制移除前后的误学率下降幅度，结果表明复制移除可将复制网络误学率下降0.10–0.24，实体新颖度检测与主流嵌入模型AUC均≥0.97。

**⚠️ 局限性**

局限性包括复制检测召回率依赖于已标注样本、实体新颖度对自然同源衍生的识别率有限、复制移除对单源真相的保护不充分、仿真中复制错误假设独立、以及缺乏对真实新闻流的端到端验证。

---

## 376. Load Balancing with Item Fragmentation and More Bins than Items: Linear-Time and Exact

**arXiv ID:** 2611.09267 | [PDF](https://arxiv.org/pdf/2611.09267v1)

**作者:** Ben Bals `[一作]` (CWI), Leen Stougie `[通讯]` (CWI)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种针对“物品碎片化负载平衡”问题的线性时间最优算法，适用于机器数至少等于任务数的情形；

**💡 创新点**

证明在B≥N的情况下，总能得到每个箱子最多两块碎片的最优分配，并给出判定是否可实现单碎片（负载1）的判据；

**🔧 技术方法**

主要采用组合优化和整数拆分的数学证明，算法实现仅用常数级别的算术与集合操作；

**📊 数据集**

无实验数据集，论文仅提供理论证明与算法设计；

**📈 对比分析**

与Kobert等人提出的（OPT+1）近似算法相比，时间复杂度从O(N log N + B)降至O(N)，且在B≥N时得到最优解；

**⚠️ 局限性**

局限在于只覆盖B≥N的情况，对B<N需退回到近似算法；且算法依赖整数尺寸且未做实验验证。

---

## 377. Path Isolation and Reachability in Surface-Embedded Graphs with Restricted Cycle and Cut Structure

**arXiv ID:** 2611.09262 | [PDF](https://arxiv.org/pdf/2611.09262v1)

**作者:** Neelabjo Shubhashis Choudhury `[一作]` (Indian Institute of Technology Kanpur), Raghunath Tewari `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在表面嵌入的有向图中，当图满足两种结构限制（多项式数量的非分离环和层级分解中交叉边数为O(log n)）时的可达性问题，并给出了相应的权重分配与空间复杂度上界；同时也将该方法推广到二分图完美匹配问题。

**💡 创新点**

首次构造了在多项式数非分离环的图以及层级分解交叉边数为O(log n)的高-genus图上均可实现路径或匹配唯一化的权重函数，进而将有向可达性问题逼入USPACE(log n)或USPACE(log g·log n)；该结果将原来的O(log^2 n)、O(log^1.5 n)空间提升到更低复杂度。

**🔧 技术方法**

利用了反对称（skew-symmetric）权重分配、路径唯一化（min‑unique）技术、FKS 哈希（Fredman‑Komlós‑Szemerédi）对路径权重进行模运算、层级分解与交叉边控制，以及不确定计算的唯一路径（unambiguous）判定算法。

**📊 数据集**

无实验数据；该工作为理论算法研究，未使用具体图数据集。

**📈 对比分析**

与先前的O(log^2 n)空间、O(log^1.5 n)空间算法对比，本文在相同或更弱的结构假设下将空间上界降至O(log n)（或O(log g·log n)）且保持算法的可接受性（unambiguous logspace），证明了理论上更优的复杂度。

**⚠️ 局限性**

仅适用于具有多项式非分离环或层级分解交叉边数为O(log n)的图，未覆盖一般有向图；需要预先构造或已知合适的分解，且对分解深度和交叉边数的限制较严格；此外权重函数的构造虽然在logspace可计算，但实现细节仍然复杂。

---

## 378. Forecasting cumulative intraoperative hypotension burden: patient-disjoint evaluation of quantile boosting and alert policies

**arXiv ID:** 2611.09259 | [PDF](https://arxiv.org/pdf/2611.09259v1)

**作者:** Yunfan Zhao `[一作]` `[通讯]` (GE Healthcare), Yunfan Zhao (GE Healthcare)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建并评估了用于预测手术期间平均动脉压（MAP）累计低压负荷的条件分位数模型，并对其预测性能与警报策略进行分离评价。

**💡 创新点**

创新点在于提出了可复现的病人独立分位数预测基准，区分了预测准确度与警报效能，并使用残差与直接学习方法对累计低压负荷进行量化。

**🔧 技术方法**

采用基于XGBoost的梯度提升树进行分位数回归，结合历史MAP特征与残差学习，以及患者加权pinball损失评估。

**📊 数据集**

使用公开的VitalDB（含6,388例手术患者）中血管动脉压记录，经过10秒中位数重采样，形成训练/验证/测试的病人分层拆分。

**📈 对比分析**

通过患者加权pinball损失与bootstrap置信区间进行比较，残差模型相对持久性预测在90%分位数上损失约0.04 mmHg·min；但在警报敏感度上直接模型仅在0.25未匹配警报/决策曝光小时约16%敏感度、14%精准度，表现不具临床可行性。

**⚠️ 局限性**

仅在单中心、侵入式血管监测的内部数据上验证，未评估外部可迁移性、非侵入式监测、实时缺失处理或临床干预效果；且警报策略与真实临床工作流尚未验证。

---

## 379. MM-DyGraph: A Dataset, Benchmark, and Model for Multimodal Dynamic Graphs

**arXiv ID:** 2611.09240 | [PDF](https://arxiv.org/pdf/2611.09240v1)

**作者:** Zongyuan Wu `[一作]` (DCST), Wenwu Zhu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MM‑DyGraph这一多模态动态图数据集，并提出了目标感知的空间‑时序多模态融合框架TST‑MF。

**💡 创新点**

创新点在于：①首次系统化提供文本与图像共存的动态图基准；②发现传统拼接融合存在目标不一致问题；③设计了基于目标、空间邻域与时间历史的分层自注意力融合机制，实现上下文感知的多模态整合。

**🔧 技术方法**

采用CLIP进行文本与图像编码，使用Transformer（序列与图结构）捕捉时序与邻域信息，并在TST‑MF中实现两阶段跨注意力融合；对比了多种融合策略（拼接、门控、模态注意、MoE）。

**📊 数据集**

数据集包括四个真实世界来源：GDELT、Google Map City (GMCT)、Amazon 以及 Yelp，均包含细粒度时间戳、文本与图像属性，覆盖链接预测、边分类与节点检索三种任务。

**📈 对比分析**

与八大现有动态图模型（DyGFormer、TGAT、GraphMixer、CAWN、TCL、TPNet、SALoM、FreeDyG）以及多种融合方式进行评估，结果显示TST‑MF在大多数数据集与任务上均取得最高或次高分，显著优于直接拼接与传统融合方法，尤其在多模态场景下提升明显。

**⚠️ 局限性**

局限性包括：①对目标、空间与时间的依赖导致模型对不同场景的泛化需进一步验证；②主要关注文本+图像两模态，未探索更多模态组合；③在强结构支配或样本稀疏的情形下，融合效果仍受限；④缺乏大规模预训练与自监督策略，难以处理更大规模或多源异构数据。

---

## 380. DUET: Dual-Sparse Inference for Block Diffusion Language Models

**arXiv ID:** 2611.09239 | [PDF](https://arxiv.org/pdf/2611.09239v1)

**作者:** Siyuan Yan `[一作]` (Nanjing University), Jingwei Xu `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的双重稀疏推理框架Duet，专门针对块级扩散语言模型（BDLM）进行推理加速。

**💡 创新点**

创新点在于同时利用历史KV稀疏和早期查询稀疏：在每个生成块开始时选取稳定的历史KV子集并在后续推理中复用；在每一步的早期层预测需要深度计算的活跃位置，仅将其继续传播到后层。

**🔧 技术方法**

采用的技术包括：块级KV子集选择（基于查询-键相似度）、层级查询稀疏（早期层解码预测）、分层稀疏注意力、无训练推理优化、共享KV缓存以及自适应的稀疏比例设置。

**📊 数据集**

在LLaDA2.1-mini和SDAR-8B-Chat-b32两大BDLM上进行实验，使用LongBench、GSM8K、MMLU、HumanEval、MATH-500、NarrativeQA等数据集进行质量和效率评估。

**📈 对比分析**

与dense、FOCUS、LoSA等基线对比，Duet在32K上下文下实现端到端速度提升1.34×/2.83×（LLaDA）和2.41×/4.72×（SDAR），在各长短文本任务上保持与dense相近的生成质量。

**⚠️ 局限性**

局限性包括：需为每个模型手动调节历史KV预算与查询保留比例，对极端长上下文仍存在一定的计算开销；该方法仅对已有BDLM结构适用，无法直接迁移到非块级模型。

---

## 381. TPS-A*: Path Planning for Any-Shape Vehicles with Certified Collision Checking in Trajectory Parameter Space

**arXiv ID:** 2611.09210 | [PDF](https://arxiv.org/pdf/2611.09210v1)

**作者:** Jose Luis Blanco-Claraco `[一作]` (University of Almeria), José Carlos Moreno-Úbeda `[通讯]` (University of Almeria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TPS‑A*，一种针对任意平面车身形状的确定性格点规划器，利用轨迹参数空间（TP‑Space）中的预计算碰撞网格实现一次性、可证实的碰撞检测，生成可行、可执行且不与障碍物相撞的路径。

**💡 创新点**

创新点在于：① 将碰撞检查转化为节点批量查询；② 通过 TP‑Space 预计算的保守距离网格，保证对任意形状车身的完整碰撞检测且不随形状复杂度或分支因子变化；③ 结合 Reeds‑Shepp 解析射线，实现对姿态和位置目标的精确到达；④ 在保持确定性的同时提供 ε‑近似最优性。

**🔧 技术方法**

使用技术包括：Parameterized Trajectory Generators (PTG)、轨迹参数空间 (TP‑Space)、预计算碰撞网格、加权 A*、Reeds‑Shepp 轨迹、解析目标扩展、动态成本/清晰度评估、ROS2 包装。

**📊 数据集**

数据集：BARN (Jackal 世界)、HouseExpo (房间门道)、TPCAP (自动泊车)、自制非凸叉臂泊位、实际 Husky 机器人校园导航。

**📈 对比分析**

比较方法：在 BARN 上与 Nav2 Hybrid‑A*、SBPL ARA*、OMPL RRT*/BIT* 同一地图、相同车身、相同障碍阈值进行统计。TPS‑A* 取得 100% 合法路径，速度与 SBPL 相当（中位 2.9 ms）但 58 倍更快，Nav2 位置目标下 89 倍更快；在门道、泊车、非凸泊位和实机实验中也表现出更高的安全性、较短路径或更快的重规划时间。

**⚠️ 局限性**

局限性：① 不是严格的分辨率完备（受格点表示限制）；② 需要保守增量 m+Δ_c，粗网格会丢失最窄通道；③ 碰撞检查线性于局部障碍点数，密集点云下可能成本高；④ 仅保证运动学可行性，未考虑加速度；⑤ 需要为每种车身形状离线构建碰撞网格；⑥ 对动态障碍物缺乏预测，需依赖实时重规划。

---

## 382. Probably Safe Sim-to-Real Transfer with Lifetime Validity

**arXiv ID:** 2611.09344 | [PDF](https://arxiv.org/pdf/2611.09344v1)

**作者:** Rohan Tangri `[一作]` (University of Oxford), Jan-Peter Calliess `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Dejavu方法，通过对从模拟环境训练的候选策略进行任意时刻有效的上界测试，在不执行候选策略的情况下，基于已安全的基线策略收集生产轨迹，确保每次部署都满足成本约束。

**💡 创新点**

创新点在于将在线多测试的alpha‑spending与重要性采样结合，构建任意时刻有效的置信序列上界，并实现对无限次部署的生命周期安全保证。

**🔧 技术方法**

使用技术包括Per‑decision Importance Sampling、Betting Confidence Sequences、累积重要性权重裁剪、KL锚定以及alpha‑spending多测试框架。

**📊 数据集**

使用的数据集为三种仿真到真实的CMDP基准：HalfCheetahVelocity、PointGoal 与 IcyLake，均采用JAX实现。

**📈 对比分析**

与CPO、SPiDR、HCPI比较，Dejavu在保证生命周期安全的前提下，奖励累计最高，且在所有环境中都满足成本约束。

**⚠️ 局限性**

主要限制在于样本效率较低，且假设轨迹独立且初始状态分布固定，未处理无重置连续运行或动态变化的生产环境。

---

## 383. Wake 5: A Python Framework for Solidity Testing, Fuzzing, and Static Analysis

**arXiv ID:** 2611.09307 | [PDF](https://arxiv.org/pdf/2611.09307v1)

**作者:** Josef Gattermayer `[一作]` (Czech Technical University), Michal Převrátil `[通讯]` (ACK3 LTD)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了一个名为Wake 5的Python框架，将EVM嵌入进程，支持手动引导的 fuzz harness，并在 Daedaluzz L32 迷宫上与 Echidna、Foundry 进行八小时覆盖和二十分钟吞吐实验。

**💡 创新点**

创新点在于：1）通过 PyO3 将 Rust 版 EVM 直接嵌入 Python，显著提升吞吐；2）提出手动引导 fuzzing（MGF）模型，强调差分模型与预期一致性；3）设计 Model Context Protocol（MCP）服务器，将静态分析结果以简洁文本形式暴露给 LLM；4）展示 LLM 生成的 harness 利用 SMT 求解器完成全部断言，证明输入生成是关键瓶颈。

**🔧 技术方法**

使用的技术包括：Rust + PyO3 绑定的 in‑process EVM；Python 动态类型与 Hypothesis 兼容的 fuzz harness；手工引导收缩器（两阶段 shrinker）；SMT 求解器（Z3）进行 guard 满足；MCP 协议；统计检验（Mann–Whitney U、Vargha–Delaney、McNemar）和精确计数。

**📊 数据集**

使用的数据集主要是：1）Daedaluzz L32 迷宫（5 个迷宫，共 153 个可达断言，常数上限为 2³²）；2）一次 Go 语言的生产共识客户端安全审计样本，用于评估静态分析工具对 LLM 的影响。

**📈 对比分析**

比较方法：对同一套合约在同一硬件上执行八小时覆盖和 20 分钟吞吐实验；记录每秒交易数、每百万交易发现断言数以及最终覆盖率。结果显示 Wake 5 与 Foundry 的吞吐相近（≈30k tx/s），Echidna 远慢；在八小时内，Wake 5、Foundry、Echidna 的覆盖率分别为约 75/153、71/153、70/153；agent 生成的 harness 在 33–79 分钟内完成全部断言，证明随机输入生成是主要瓶颈。

**⚠️ 局限性**

局限性：评估基于人工合成的 Daedaluzz 迷宫，缺乏对真实合约的验证；只比较了三款随机输入生成的 fuzzers，未考虑符号执行或混合模式；agent harness 试验次数有限，且缺少统一停止规则；静态分析工具评估未覆盖安全特定查询，结果不一定能推广到更广泛的安全审计。

---

## 384. A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures

**arXiv ID:** 2611.09302 | [PDF](https://arxiv.org/pdf/2611.09302v1)

**作者:** Nikos Giakoumoglou `[一作]` (Imperial College London), Tania Stathaki `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在JEPA中引入可学习的随机潜变量的变分自编码框架，用于无解码器的空间掩码图像嵌入预测。

**💡 创新点**

将LeCun提出的辅助变量转化为可学习的随机变量，结合上下文条件先验和后验，得到分布式预测，且在确定性极限下可恢复I‑JEPA；该方法实现了不确定性映射和多样化补全。

**🔧 技术方法**

采用Vision Transformer编码器、EMA目标编码器、停梯度、Conditional Variational Autoencoder（CVAE）在潜在空间中的ELBO训练、随机先验/后验MLP、扩散解码器做可视化。

**📊 数据集**

使用ImageNet‑1K进行预训练；在ImageNet线性评估、低样本微调、CIFAR‑100、Places205、iNaturalist 2018以及CLEVR低阶视觉任务上进行评估。

**📈 对比分析**

与I‑JEPA、LeJEPA、MAE、data2vec等方法对比；在Linear Probe、Low‑shot、Transfer任务上均优于I‑JEPA；在ViT‑L/16下表现接近MAE和data2vec；同时提供空间不确定性图和多样化补全，且额外计算成本极小。

**⚠️ 局限性**

仅使用单个全局潜变量而非每个掩码区域；最大模型为ViT‑L/16，未验证更大模型；补全需额外解码器，未实现完整生成式解码；对跨目标区域的细粒度多样性支持有限。

---

## 385. comfort-db: A Python toolkit to access, preprocess and explore the oceanographic COMFORT dataset

**arXiv ID:** 2611.09353 | [PDF](https://arxiv.org/pdf/2611.09353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 386. Towards an Autonomous AI Steward for Large-Scale Physics Experiments: Integrating Causal Diagnostics, RAG, and Conformal Prediction

**arXiv ID:** 2611.09298 | [PDF](https://arxiv.org/pdf/2611.09298v1)

**作者:** Oliver Noll `[一作]` (GSI Helmholtzzentrum fuer Schwerionenforschung), Frank Maas `[通讯]` (GSI Helmholtzzentrum fuer Schwerionenforschung)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个名为Helmholtz Co‑Pilot的混合多智能体系统，用于在核物理实验中自动化知识检索与硬件诊断，帮助操作者快速定位根因并制定对策。

**💡 创新点**

将基于LLM的RAG与异步因果诊断引擎通过统一的DAG工作流联动，并引入Human‑in‑the‑Loop安全栅栏、PhysRAG领域专属基准和可解释的覆盖保证，实现可信且可解释的自适应决策。

**🔧 技术方法**

C++高吞吐量摄取层、Python边缘层+自适应规划器、ReAct循环多工具工作者、动态技能注册、Conformal Prediction、HitL 保障、混合多智能体架构与LLM模型。

**📊 数据集**

实验现场实时硬件流（DAQ/慢控）、实验室知识库（DokuWiki、ELOG、数据库）以及基于专家访谈构建的PhysRAG评测集。

**📈 对比分析**

目前在KATRIN的shadow‑test中验证；未给定量化指标，系统声称能够在毫秒级以内完成诊断并提供带覆盖率保证的根因概率，但真实性能仍待评估。

**⚠️ 局限性**

对LLM的可解释性和知识更新仍有挑战；HitL交互时延可能影响现场操作；异步诊断引擎对多源数据同步依赖高；在非物理实验场景的可迁移性尚待验证。

---

## 387. Safety-Focused Reinforcement Learning Using Risk-Sensitive Coupled Control

**arXiv ID:** 2611.09290 | [PDF](https://arxiv.org/pdf/2611.09290v1)

**作者:** Jiabin Lin `[一作]` (Iowa State University), Vivek S. Borkar `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种安全优先的耦合控制框架（SCC），通过双时间尺度Q‑学习同时保证系统保持在安全集内并最小化平均成本。

**💡 创新点**

创新点在于将安全约束重新表述为风险敏感优化，并在同一框架中同时解决安全与性能问题；同时给出了收敛性理论保证。

**🔧 技术方法**

采用风险敏感Q学习、平均成本相对Q学习、Perron–Frobenius特征估计以及两时间尺度随机逼近等技术。

**📊 数据集**

在GridWorld和Safety Gymnasium的CarGoal两种离散化环境上进行实验验证。

**📈 对比分析**

与RCRL和RESPO等基准方法对比，实验结果显示SCC在两任务上实现了零违规概率、较高的达成成功率和更低的平均成本。

**⚠️ 局限性**

主要局限在于对安全子马尔可夫链不可约、无周期的假设；对可约或周期环境的适用性有限，且需要较多的探索与参数调节。

---

## 388. PHISHREV+: An ASP-Based Post-Hoc Defeasible Machine Learning Framework for Phishing Detection

**arXiv ID:** 2611.09273 | [PDF](https://arxiv.org/pdf/2611.09273v1)

**作者:** Mainak Sen `[一作]`, Amlan Chakrabarti `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提供了一份名为 elsarticle.cls 的 LaTeX 文档类，用于排版和提交给 Elsevier 期刊的稿件，包括预印本和最终版样式，支持多种前置信息、图表、定理环境和交叉引用等。

**💡 创新点**

创新点在于：① 基于标准 article.cls 重新实现，避免了以前 elsart.cls 的冲突；② 默认提供预印本格式，同时可通过选项切换为多种最终版模型；③ 集成 natbib、geometry、graphicx 等常用宏包，减少用户安装负担；④ 提供快捷的定理、列表和注释环境，增强作者体验。

**🔧 技术方法**

技术实现主要是 LaTeX 宏包编写，使用 natbib 进行引用处理、geometry 设置页面边距、graphicx 处理图形、txfonts（可选）提供 Times 字体、hyperref（可选）实现超链接。文档类内部定义了多种选项与命令来控制布局、标题、作者信息、摘要、关键词、脚注、图表、定理等。

**📊 数据集**

本文不涉及实验数据或数据集，主要是对宏包功能的说明和示例代码。

**📈 对比分析**

与旧版 elsart.cls 相比，elsarticle.cls 在兼容性、默认预印本布局、选项灵活性和易用性方面表现更好。作者可以直接加载所需选项并使用标准 LaTeX 命令完成排版，无需额外配置。

**⚠️ 局限性**

局限性包括：① 对某些第三方宏包（如特定图形或表格宏包）可能仍有兼容性问题；② 需要作者手动在双栏最终版中检查并适当断行长方程；③ 仍需根据期刊要求手动选择对应模型（1p、3p、5p）并调整布局。

---

## 389. Open-world Reconstruction-based Classifier with Concept Drift Adaptation

**arXiv ID:** 2611.09350 | [PDF](https://arxiv.org/pdf/2611.09350v1)

**作者:** Joanna Komorniczak `[一作]` `[通讯]` (Wrocław University of Science and Technology), Joanna Komorniczak (Wrocław University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为 ORCCA（Open‑world Reconstruction‑based Classifier with Concept Drift Adaptation）的在线分类器，旨在同时实现新颖类别检测、已知类别分类以及概念漂移的无监督自适应。

**💡 创新点**

创新点在于：①使用每个类别的自编码器重构误差阈值来精确识别已知类；②通过核密度估计（KDE）对重构误差进行概率建模，从而区分未知样本并对其进行分组；③引入 Sarle 的双峰度系数对异常样本误差进行分布分割，支持多新颖类别的同时出现；④采用误差加权的自适应微调概率（p）实现对概念漂移的无监督跟踪；⑤通过分层缓冲区和级联识别机制显著降低了处理时间。

**🔧 技术方法**

核心技术包括：
- 逐类训练的自编码器（autoencoder）
- 归一化的重构误差计算和阈值化
- 经验核密度估计（KDE）与单侧估计
- Sarle 双峰度系数与高斯混合模型（GMM）用于缓冲区分割
- 误差加权的自适应微调（fine‑tuning）
- 级联识别与阈值化相结合的两步分类流程
- Python/Scikit‑Learn/PyTorch/River/StreamND 等实现框架

**📊 数据集**

实验使用的数据集有：
- 合成 Madelon 生成的 330 条非平稳数据流（含不同数目新颖类与漂移）
- 通过 ResNet‑18 提取的 CIFAR‑10 嵌入构造的半合成流
- 三个真实世界流：Covertype（森林覆盖）、KDD Cup 1999（网络入侵）和 PAMAP2（人体活动）

**📈 对比分析**

在与 7 个基线方法（MINAS、ECSMiner、Echo、CluStream、DenStream、OCND、CND）的比较中，ORCCA 在三项评价指标（已知类准确率、已知 vs 未知二分类准确率、未知类 ARI）中表现最优，尤其在出现概念漂移的场景下排名第一。其处理时间平均为 430–530 µs/样本，内存占用约 640 kB，随数据流长度增长的时间与内存增长率均低于多数对手，显示出良好的可扩展性。

**⚠️ 局限性**

局限性包括：
- 合成漂移机制仅模拟平滑方向性漂移，可能不完全反映真实的突变漂移；
- 当漂移突变或重叠度高时，重构误差阈值可能失效，导致新颖样本被误判为已知类；
- 需要初始离线训练阶段，且对样本量要求较大；
- 随着已识别类别数量增加，模型参数存储导致内存占用上升，尤其在高维特征下更为显著；
- 对极端稀疏或多新颖类别场景的鲁棒性尚待进一步验证。

---

## 390. Horizon-Coupled Obstacle Guidance for Relative-Action Diffusion Policies

**arXiv ID:** 2611.09340 | [PDF](https://arxiv.org/pdf/2611.09340v1)

**作者:** Guangyuan Wang `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对已预训练的相对动作扩散策略，提出了在推理时通过跨步传播障碍响应的水平耦合障碍引导框架，使机器人能够在不重新训练的情况下实现完整的任务导向避障运动。

**💡 创新点**

创新点在于：① 引入跨步（horizon‑coupled）传播机制，将后续轨迹点的障碍响应回传到所有对该点有贡献的相对动作增量，协调整个动作序列；② 结合SDF障碍模型与梯度推理，提供平滑且几何一致的避障反馈；③ 仅在推理阶段使用已有的DP3模型，无需额外的辅助策略或强化学习。

**🔧 技术方法**

核心技术包括：扩散策略（Diffusion Policy）、DDIM逆向采样、基于SDF的障碍成本与梯度、动作到轨迹的递推映射、雅可比链式传播实现跨步响应，以及在执行阶段对相对动作增量的平滑加权更新。

**📊 数据集**

使用通过PS5手柄进行远程操控收集的50条无障碍演示数据（每个任务20条），以及真实机器人上配置的多障碍实验场景；数据集为自建的实景点云+机器人状态记录，未使用公开大规模数据集。

**📈 对比分析**

与无引导（Abs.-NoG、Rel.-NoG）和逐步点位引导（Abs.-Guide）对比。实验显示在140次障碍实验中，水平耦合引导实现105/140（75.0%）的碰撞自由成功率，而绝对动作点位引导仅为8/140（5.7%），显著提升；相对动作无引导仅能得到0/140。

**⚠️ 局限性**

局限性包括：仅处理静态障碍且需手动设定SDF和引导强度，无法自适应动态或未知障碍；引导方法对推理频率有一定开销；在更复杂场景下可能仍需针对任务的细粒度调参。

---

## 391. An Inspection Game for Monitor Design Against GNSS Timing Attacks in Synchronous Islanding

**arXiv ID:** 2611.09331 | [PDF](https://arxiv.org/pdf/2611.09331v1)

**作者:** Gregory Provan `[一作]` (University College Cork), Quanyan Zhu `[通讯]` (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将同步岛屿对 GNSS 时序攻击的防御建模为检验博弈，利用 KL 散度度量证据，设计监视器和审计周期。

**💡 创新点**

创新点在于把证据等价于最便宜的无辜解释，导致攻击最优策略为凸二次规划，防御设计为凸优化，并给出安全审计周期的上、下界。

**🔧 技术方法**

使用的技术包括高斯噪声模型、Kullback–Leibler 散度、线性系统理论、Kalman 滤波、凸优化以及 Sion 最小极大定理。

**📊 数据集**

使用的数据集为 50 Hz 岛屿电网参数、PMU 噪声、振荡器规格和负载白噪声，所有数值均来自公开参数并在仿真中复现。

**📈 对比分析**

与单独检查规则比较，融合监测可将安全审计周期提升至原来的 1.1–8.4 倍；在 OCXO/TCXO 等硬件配置下，最佳摆放显著延长周期。

**⚠️ 局限性**

局限性包括模型假设（白噪声负载、理想审计、无自适应攻击）、忽略振荡器老化和控制器非线性，以及未考虑多节点电网的拓扑效应。

---

## 392. Recursively Self-Improving Electromagnetic Spectrum Agents: Architecture and Operation

**arXiv ID:** 2611.09318 | [PDF](https://arxiv.org/pdf/2611.09318v1)

**作者:** Baolin Chong `[一作]` (Nanjing University of Aeronautics and Astronautics), Qihui Wu `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了递归自我改进的电磁谱代理（RSI-ESA），设计了其架构与自我改进流程，并在共享谱段调度场景中验证了其性能。

**💡 创新点**

创新点在于将大型语言模型驱动的候选生成与持续经验积累相结合，实现基于运行反馈的递归自我改进机制。

**🔧 技术方法**

使用技术包括大型语言模型（Qwen3-8B/32B）、可编辑组件接口、持久化存储、验证模块、诊断规则以及仿真调度环境。

**📊 数据集**

数据集为仿真系统生成的 24 终端、6 个共享通道、2 个单射收发机的调度场景，包含队列状态报文延迟与 ACK 延迟等环境扰动。

**📈 对比分析**

通过与基线 ESA 对比，关键服务及时交付提升约 3.7–5.6 个百分点，空闲传输率下降近 30%；实验显示经验继承对大型模型有显著提升。

**⚠️ 局限性**

局限在于经验继承效果依赖模型规模，对更广泛任务与长周期操作的泛化与经验选择机制尚未完善。

---

## 393. Stable probabilistic emulation of partially observed dynamical systems

**arXiv ID:** 2611.09317 | [PDF](https://arxiv.org/pdf/2611.09317v1)

**作者:** Mario Massy `[一作]` (École polytechnique, Institut polytechnique de Paris), Matthieu Blanke `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了带强迫-耗散结构的随机插值器，用于学习部分观测动力学的概率模拟，并给出了连续过程的稳定性与几何极化性保证。

**💡 创新点**

在漂移网络中硬约束能量流（对称、耗散、非耗散项），实现无外力和外部扰动下的稳定生成，并利用Foster–Lyapunov证明几何极化性，填补了概率生成模型长期稳定性的空白。

**🔧 技术方法**

采用随机插值器框架、耗散结构的神经网络参数化、能量不等式、Foster–Lyapunov 驱动条件、几何极化性证明以及对比实验。

**📊 数据集**

使用单层和双层Lorenz‑96系统生成的 10⁵ 步轨迹作为训练集，包含 40 或 36 个慢变量与 10 个快变量（总维度 396）。

**📈 对比分析**

与无约束插值器、投影插值器、卷积插值器以及条件扩散模型对比；在单层系统中，只有卷积与强迫耗散插值器保持稳定且能恢复统计分布，后者在强扰动下响应最准确；在双层系统中，强迫耗散插值器在不同网络容量下始终保持稳定，均能准确重现均值与方差响应，而扩散模型在大容量下仍不稳定且响应过大。

**⚠️ 局限性**

稳定性证明仅适用于连续时间过程，离散采样步骤需进一步分析；参数化中矩阵乘法在高维下计算代价高，适合低维降维或潜在空间；对极端外部扰动的精确逼近仍需改进。

---

## 394. Orthogonal Norm Growth in Two-Tower Retrieval: A Popularity Mechanism for Unified Discovery

**arXiv ID:** 2611.09294 | [PDF](https://arxiv.org/pdf/2611.09294v1)

**作者:** Andrei Atamaniuk `[一作]` `[通讯]` (Wildberries and Russ), Andrei Atamaniuk (Wildberries and Russ)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了统一检索与推荐系统中共享物品表示的范数如何随多次更新而累积，并通过理论推导与实验验证了该机制的影响。

**💡 创新点**

提出了在 cosine 目标下，若使用 lookup‑equivalent 且方向保持的行更新，则每一步更新为正切，能直接把增量累加到范数；同时给出了检索与推荐两条流共享物品编码器时范数如何累积的条件，并通过非线性两塔实验验证了该机制的普适性。

**🔧 技术方法**

在理论上推导参数到输出的更新映射和正切运动的充要条件；在实验上使用两塔检索模型、InfoNCE、SGD/Adam/动量/行式 AdaGrad 等优化器，并对角度、频率‑范数相关性、原始内积与归一化评分进行度量。

**📊 数据集**

使用 MovieLens 32M（约 32M 交互）和 Amazon Books 3‑core（约 35M 交互）两个大型推荐数据集。

**📈 对比分析**

通过测量更新角度分布验证正切条件、计算频率‑范数相关系数以及对比原始内积与归一化评分的差异来量化范数影响；实验显示在满足条件的情况下，raw‑dot 排名的头部曝光显著提升（如 exp 8 中频率‑norm r ≈0.66，head exposure dot 0.72 vs unit 0.60），但召回/NDCG 下降约 17‑22%。

**⚠️ 局限性**

条件较为严格，只适用于 lookup‑equivalent 且方向保持的行更新；对 AdamW、非线性共享 MLP 的理论不完整；缺乏检索日志验证跨任务的实际普及；干预可能降低推荐质量；结果仅为诊断工具而非直接的偏差缓解方案。

---

## 395. MDF++: Visuo-Tactile World Action Model with End-to-end Representation Learning

**arXiv ID:** 2611.09293 | [PDF](https://arxiv.org/pdf/2611.09293v1)

**作者:** Zixuan Huang `[一作]` (University of Michigan), Dmitry Berenson `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 MFD++：一种可视-触觉世界-动作模型，端到端学习任务相关的多模态表示并用于机器人控制。

**💡 创新点**

创新点在于跨模态选择性梯度流防止特征坍塌，并通过多任务结构化查询（策略、前向/逆动力学、联合预测）实现更具控制感知的表示。

**🔧 技术方法**

使用潜在扩散模型、跨模态预测、选择性梯度流、多任务训练和噪声矩阵编码等技术。

**📊 数据集**

实验数据来自 Isaac Lab Factory、Forge+GelSight Mini 的五个模拟任务（PegInsert、GearMesh、NutThread、BallSort、ObjectSearch）以及真实 Franka R3 机器人上的 Bolt Screwing 与 Nut Mating 演示。

**📈 对比分析**

与冻结预训练编码器、随机掩码、UWM、DP 等基线对比，MFD++ 在五个模拟任务平均成功率 86.9%，在真实任务中 Bolt Screwing 80%、Nut Mating 40%，显著优于基线 18–29%。

**⚠️ 局限性**

局限性包括低频力信号处理、需要手动调节多模态损失权重、选择性梯度流可能限制潜在空间的可扩散性，以及未处理高频接触动态。

---

## 396. Diversity-Aware Client Selection for Communication-Efficient Federated Learning

**arXiv ID:** 2611.09278 | [PDF](https://arxiv.org/pdf/2611.09278v1)

**作者:** Max Halleryd `[一作]` (RISE Research Institutes of Sweden), Salman Toor `[通讯]` (Scaleout Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了多样性驱动的客户端选择对联邦学习通信效率与模型泛化的影响，并在非IID环境下系统评估了三种选择策略（Power‑of‑Choice、Fisher信息、CKA）的表现。

**💡 创新点**

通过将参与度多样性（Gini系数、KL散度）与通信效率统一评估，证明多样性驱动的选择能够使通信量大幅降低（约6.46×）的同时保持甚至提升模型泛化性能，展示了效率与质量的互补关系。

**🔧 技术方法**

使用联邦平均（FedAvg）作为基础训练框架；三种客户端选择算法分别基于本地损失（Power‑of‑Choice）、经验Fisher信息迹（Fisher）和Centered Kernel Alignment（CKA）；采用Gini系数和KL散度衡量参与多样性；模型分别为MNIST的轻量CNN和ImageNet50的ConvNeXt‑T。

**📊 数据集**

MNIST手写数字数据集和ImageNet50（50类）图像分类数据集。

**📈 对比分析**

在相同的实验设置下，将完整参与（20/20）与每轮仅选3/20客户端的三种策略进行对比，记录训练/测试准确率、通信量、Gini系数、KL散度。结果表明，Power‑of‑Choice在保持6.46×通信节省的同时，测试准确率仅略低于基线；Fisher相近，CKA表现最差。

**⚠️ 局限性**

实验仅在极低参与率（15%）和两类相对简单的数据集上验证，缺乏对更高参与率、更强非IID或更大规模任务的评估；随机选择未被纳入对比；缺少自适应动态选择机制的探索。

---

## 397. ReSUM: Reasoning-Driven Multi-Agent Framework for Query-Focused Summarization

**arXiv ID:** 2611.09349 | [PDF](https://arxiv.org/pdf/2611.09349v1)

**作者:** Sonali `[一作]` (Indian Institute of Technology Patna), Asif Ekbal `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 ReSUM 的多代理框架，用于对长会议记录进行查询聚焦摘要，并通过规划、检索、摘要与批判四个代理的迭代协作实现。

**💡 创新点**

创新点在于：① 将复杂推理拆分为规划、检索、摘要和批判四个专门代理，并通过规划指导检索范围；② 采用教师-学生链式思考（CoT）推理蒸馏，使检索和摘要代理内部化推理能力；③ 引入批判代理对每轮结果进行系统评估与反馈，实现自我纠错和多轮迭代。

**🔧 技术方法**

使用技术包括：多代理 LLM 调度（Planner、Retriever、Summarizer、Critique），CoT 推理蒸馏训练，基于语义检索的检索代理，基于证据的摘要生成以及批判评估循环。

**📊 数据集**

使用数据集：QMSum（会议记录的查询聚焦摘要基准）和 SQuALITY（长篇叙事的问答摘要基准）。

**📈 对比分析**

与多种基线对比：零样本、链式思考、检索增强生成（RAG）、微调、微调+推理蒸馏、端到端微调等。ReSUM 在 QMSum 上 ROUGE‑1、ROUGE‑2、ROUGE‑L 以及 BERTScore、MoverScore、COMET、chrF 等指标均超过所有基线，SQuALITY 上也保持了稳健的提升，表明在长文本推理和查询聚焦方面具有显著优势。

**⚠️ 局限性**

局限性：① 依赖高质量教师 LLM 生成 CoT 推理，低资源场景或特定领域可能难以复现；② 多代理迭代带来额外计算开销，影响实时部署；③ 批判代理虽能降低幻觉和缺漏，但仍是自动化评估，可能无法捕捉细微语义不一致或个性化需求。

---

## 398. When Feedback Becomes a Backdoor: Tracing Backdoor Skill Formation in Text-Space Skill Optimization

**arXiv ID:** 2611.09348 | [PDF](https://arxiv.org/pdf/2611.09348v1)

**作者:** Louise Nayeong Koh `[一作]` (Yonsei University), Hyogon Ryu `[通讯]` (KRAFTON)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在文本空间Skill优化中，利用受污染的参考答案进行攻击，诱导模型形成并持久化触发-负载规则；

**💡 创新点**

提出分阶段的促销链追踪方法，揭示攻击成功的关键门控点，并在结构化工具环境中定位形成与排名的瓶颈；

**🔧 技术方法**

采用SkillOpt框架、分析师反射、编辑预算与验证筛选，并进行诊断性干预；

**📊 数据集**

使用SearchQA（开放域问答）和SpreadsheetBench（工作簿编辑）两大数据集进行实验；

**📈 对比分析**

与无毒化的SkillOpt以及TBA基线对比，SearchQA上自然攻击在所有种子下实现200/200触发问答命中率，且对干净问题无误报；SpreadsheetBench未能自然保留完整规则，需诊断干预；

**⚠️ 局限性**

实验仅覆盖两种任务和单一优化器、模型，种子和任务配置差异导致结果难以泛化，未对多模型、多任务和防御机制进行广泛评估；

---

## 399. Reliable Prerequisite-Aware Learning Path Recommendation under Uncertain Knowledge Structures: Long-Horizon Evaluation and Cross-Dataset Validation

**arXiv ID:** 2611.09345 | [PDF](https://arxiv.org/pdf/2611.09345v1)

**作者:** M. M. Golam Hafiz `[一作]` (American International University-Bangladesh), M. F. Mridha `[通讯]` (American International University-Bangladesh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一种可靠性导向的先修知识图谱学习路径推荐框架，结合即时下一资源排名、先修关系的不确定性、硬/软/混合约束语义、图鲁棒性分析和自回归路径生成，进行课程级别的开发验证、冻结确认评估以及跨数据集的独立验证。

**💡 创新点**

创新点在于：①将即时推荐准确性与长周期路径质量视为互补的评估维度，②对先修图结构进行多种不确定性处理（信任度、误删/误加/反向/噪声），③引入硬/软/混合约束语义的多模型对比，④在同一框架下统一评估即时排名、路径结构合法性、覆盖率、多样性等长期属性。

**🔧 技术方法**

使用的技术包括：图变换器（Graph Transformer）结合注意力和低秩语义适配器；XGBoost、Two-Tower、线性 pairwise ranker等基准模型；硬/软/混合约束策略；自回归路径生成与束搜索；图扰动实验（删加反向、噪声）和特征可靠性审计；统计不确定性评估（课程/学习者级别自举）。

**📊 数据集**

数据集：主实验使用 MOOCCubeX（3,187 课程/37,747 转移），冻结验证集 541 课程；跨数据集验证使用 Junyi Academy（10,151 学习者、1,330 资源）。

**📈 对比分析**

比较方法：在冻结确认集上，XGBoost pairwise 获得最高 MRR 0.9048，随后是 Two-Tower、Linear HACR；图模型得到 0.7707。自回归路径实验中，图模型的路径覆盖率最低（0.2838），但在某些指标上仍优于随机；Junyi 验证中 Markov 在即时排名上领先，图模型在路径多样性等指标上表现相近。

**⚠️ 局限性**

局限性包括：①缺乏学习者个性化；②路径生成的完成率低（仅 12 课程完成路径）；③即时排名高度依赖标题/路径相似度特征，可能掩盖真正的先修信息；④未对先修边进行专家验证或学习成效评估；⑤仅在离线环境下评估，未验证对真实学习者的教学影响。

---

## 400. Syn2DRE: Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction

**arXiv ID:** 2611.09342 | [PDF](https://arxiv.org/pdf/2611.09342v1)

**作者:** Khai Phan Tran `[一作]` (University of Queensland), Xue Li `[通讯]` (University of Queensland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种半监督的文档级关系抽取框架Syn2DRE，利用轻量级教师‑学生对来生成可靠的伪标签并进行联合训练；

**💡 创新点**

创新点包括：1) 采用EMA更新的双教师‑学生架构一次性生成高质量伪标签；2) 设计无监督的协同交叉监督机制，消除确认偏差；3) 将自监督与交叉监督统一为端到端训练流程；

**🔧 技术方法**

使用Monte Carlo Dropout评估不确定性、EMA平滑、MeanSquaredError交叉监督损失以及多种轻量级DocRE模型（KD-DocRE、UGDRE）作为学生；

**📊 数据集**

在五个公开数据集上验证：DocRED、Re-DocRED、HacRED（中文）、CDR和GDA（生物医学）；

**📈 对比分析**

与现有半监督和生成式基线相比，Syn2DRE在DocRED、Re-DocRED、HacRED、CDR和GDA上均获得显著提升（F1提升0.5%~8%），训练时间缩短至原来的一半，所需无标签文档量也大幅减少；

**⚠️ 局限性**

局限性：仍需一定量人工标注；双网络结构显著增加GPU显存；仅实现二元双模型，未探讨多模型扩展；伪标签阈值固定，未学习自适应阈值。

---

## 401. Interpretable Context-Aware Multi-Label Retinal Classification with Dual-Hub Heterogeneous Graph Neural Network

**arXiv ID:** 2611.09309 | [PDF](https://arxiv.org/pdf/2611.09309v1)

**作者:** Proshanto Chanda `[一作]` (University of Oxford), Abhirup Banerjee `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并评估了双枢纽异构图神经网络，用于基于视网膜照片和临床变量的多标签疾病筛查。

**💡 创新点**

在图结构中将图像补丁和临床特征分离为两个枢纽，并通过跨模态消息传递和空间偏置自注意力实现可解释的多层决策路径，避免后置归因。

**🔧 技术方法**

采用 ResNet‑50 提取视觉特征，图注意力网络（GAT）进行枢纽聚合，二维高斯空间偏置的多头缩放点积注意力，交叉模态桥接，以及正向加权二分类交叉熵处理不平衡。

**📊 数据集**

在巴西多标签眼科数据集 BRSET（16,266 张视网膜照片，8,524 名患者）上进行实验，涵盖 7 种疾病。

**📈 对比分析**

与单模态 GNN、传统 ResNet‑50+MLP 融合等基线比较，双枢纽模型宏观 AUPRC 达到 76.1%，与 ResNet‑50+MLP 基线相当，并在糖尿病视网膜病变和黄斑水肿等条件上显著提升。

**⚠️ 局限性**

受限于仅有的 7 种疾病和 BRSET 规模，模型对稀缺疾病的表现仍不理想；缺乏外部验证与临床可用性评估；跨模态消息传递的可解释性仍需进一步量化。

---

## 402. T2I Models Rarely Invert Negation but Often Fail to Suppress Negated Concepts: Benchmarking Negation Suppression in Text-to-Image Generation

**arXiv ID:** 2611.09304 | [PDF](https://arxiv.org/pdf/2611.09304v1)

**作者:** Akansha Gautam `[一作]` (Indian Institute of Technology Jodhpur), Richa Singh `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 T2I-NSI 这一大规模文本到图像生成的否定敏感性基准与度量，包含 759,869 条配对提示、916,272 张生成图像，并对多种开源与闭源扩散模型进行系统评估。

**💡 创新点**

创新点在于：①首次将否定按语言形式（词汇/语义、形态、句法）与目标类型（对象、属性、关系、组合、嵌套）进行因子化；②采用基于 CLIP、InstructBLIP、GroundingDINO 的多模态探针集成，并引入“NegAlign”对保留与抑制效果进行加权；③构建六分类的失败模式体系，细粒度拆解模型在否定提示下的错误表现。

**🔧 技术方法**

技术手段包括：扩散模型图像生成、基于 CLIP 的图像–文本相似度评估、VQA 与检测探针的集成、基于正负对的种子对齐、NegAlign 计算、阈值过滤与加权归一化、失败模式自动标注与统计。

**📊 数据集**

使用数据集：5,082 个视觉可观测对象的手工编制清单与 Synthetic Visual Genome 的子集构成对象目录；从该目录生成 759,869 条否定/正向配对提示；对 Open-Weights（FLUX.2、Qwen-Image-2512、Stable Diffusion 3.5）和 Closed-Source（GPT-Image-1.5、Gemini-2.5-Flash-Image、Midjourney v7）各自渲染图像，得到 916,272+4,290 张图像。

**📈 对比分析**

对比方法：使用 Robust 估计器（CLIP+InstructBLIP+GroundingDINO）计算 NSI 指标，对各模型做聚合、分级与条件化分析；与人类标注结果、同一提示下的开/闭源差异做对比。性能表现：开源模型 NSI 约 0.03–0.04，闭源模型提升至 0.09；失败模式显示 13–17% 目标在正负提示下都出现（低抑制），而逆转率接近 0%；条件化评分表明词汇/语义否定最易抑制，形态次之，句法最难，关系类最弱。

**⚠️ 局限性**

局限性：探针（CLIP、VQA、检测）存在否定盲区，阈值设定缺乏外部校准；人类一致性低，标签与探针标注不完全一致；仅覆盖英语、视觉可观测否定，未覆盖非视觉或多语言场景；闭源模型样本量有限，覆盖面不足；种子多样性分析仅在 K=4 的小样本上验证；最终 NSI 仍为探针驱动的自动化评估，未能完全代表人类感知。

---

## 403. Ontology-Driven Control of Conversational Agents with a Use-Case for Job Interview Practice

**arXiv ID:** 2611.09282 | [PDF](https://arxiv.org/pdf/2611.09282v1)

**作者:** Barbara Gendron `[一作]` (Université de Lorraine), Mathieu D'aquin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于本体的对话控制框架，利用本体描述对话策略并通过LLM实现自动化的求职面试对话

**💡 创新点**

首次将本体用于定义多维度对话策略，并通过本体推理与提示控制LLM输出，实现可解释、可调节的对话流；同时通过GRPO无监督强化学习实现对话特征的控制

**🔧 技术方法**

本体建模、LoRA适配层、GRPO强化学习、BERT编码器（情感负荷、极性预测）、决策树（CEFR等级）

**📊 数据集**

面试对话用例数据（30段完整对话），自建情感负荷与极性预测数据，CEFR特征提取数据

**📈 对比分析**

与预训练Llama3.1‑8B和Qwen2.5‑7B进行对比；在控制指标上（精确率、加权F1、MAE、MCC）Fine‑tuned模型均有提升；在完整对话中，结构遵循率高，但情感负荷/极性控制仍受限，预训练模型在负荷控制上表现略好，预训练在极性控制上略逊

**⚠️ 局限性**

过度正向生成偏差导致情感负荷/极性控制困难；需要专门本体，设计成本高；自动化评估与人工评估结果偏差明显，需更大规模人评验证

---

## 404. An Empirical Study of Issues in GPU Kernel Libraries for LLM Systems

**arXiv ID:** 2611.09280 | [PDF](https://arxiv.org/pdf/2611.09280v1)

**作者:** Yanjie Gao `[一作]` (Microsoft Research Asia), Ling Feng `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对七个代表性LLM GPU核库的300条GitHub issue进行手工分析，系统地归纳了其症状、根因、修复方案以及测试调试实践。

**💡 创新点**

首次提出针对LLM GPU核库的全面经验性研究，并构建了症状与根因的多维分类体系。

**🔧 技术方法**

采用人工标注、代码审查、统计分析等手段，结合GitHub API抓取数据。

**📊 数据集**

使用来自FlashInfer、FlashAttention、Triton、CUTLASS、DeepGEMM、xFormers和bitsandbytes等七个开源库的300条高质量issue。

**📈 对比分析**

通过定量描述各类症状和根因的比例，对比不同库的分布；并未直接评测性能，而是提供类别统计和修复模式的洞见。

**⚠️ 局限性**

受限于只分析公开issue、样本偏倚以及对其他闭源或新兴库缺乏覆盖，研究结果的普适性尚需进一步验证。

---

## 405. Task-Driven Closed-Loop Design Optimization of Passively Morphing Quadrotors for Contact-Mediated Narrow Gap Traversal

**arXiv ID:** 2611.09272 | [PDF](https://arxiv.org/pdf/2611.09272v1)

**作者:** Etor Arza `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在闭环仿真环境中，联合优化四旋翼机体几何尺寸与侧臂弹性滚动凸轮关节的扭矩-偏转响应，使其在接触介导的窄缝隙通过后能够自恢复，随后在实验室实际飞行中验证其性能。

**💡 创新点**

创新点在于：①提出任务驱动的闭环设计优化框架，直接用穿越成功率作为反馈；②在同一框架内同时优化机体几何和关节柔性；③通过可编程弹性凸轮实现非线性柔性响应；④首次实现了单个四旋翼机从未折叠宽31.3 cm压缩至14.1 cm，并能通过15 cm窄缝及40 cm长管道。

**🔧 技术方法**

使用了 Elastic Rolling Cam (ERC) 关节、GPU并行刚体仿真 (Isaac Lab)、贝叶斯优化 + CMA‑ES 两阶段搜索、非线性模型预测控制器 (NMPC) 与仿真参数化模型、以及仿真→实机的转移校准技术。

**📊 数据集**

数据集主要为仿真生成的 100 个平行环境，缝隙宽度均匀分布在 0.15 m–0.25 m 之间；实机测试使用真实的运动捕捉测量轨迹和状态。

**📈 对比分析**

与同尺寸刚体对比，软体机在正常飞行中位置 RMSE 仅增加 2 mm，电力消耗 <1%；在窄缝通过上，成功率最高 79% 并可通过比自身未折叠宽度 52% 更窄的 15 cm 缝隙；在 20 cm 缝隙的 30° 角度误差下仍可通过，误差小于 1 m 下降。

**⚠️ 局限性**

局限性包括：①仿真模型对质量、惯量及马达响应的误差导致实机最大角速度与仿真偏差 15‑42%；②3D 打印制造导致关节接合处偶尔脱位，限制高角度入场（>30°）的可靠性；③目前仅针对单一 + 结构和侧臂折叠，扩展到更复杂的机体拓扑或多模态柔性仍需研究。

---

## 406. Federated Forensics: Jurisdictional constraints as a design driver for multi-agency federated learning

**arXiv ID:** 2611.09277 | [PDF](https://arxiv.org/pdf/2611.09277v1)

**作者:** Aruna Withanage `[一作]` (University of Augsburg), Frank Breitinger `[通讯]` (University of Augsburg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向数字取证的分层联邦学习架构（Federated Forensics），并实现了一个可满足司法管辖限制的政策门控机制

**💡 创新点**

核心创新在于把聚合节点置于法律边界处，允许局部数据参与模型训练而不泄露至外部，同时通过可审计的政策门实现跨境数据限制的可追踪执行

**🔧 技术方法**

采用联邦学习框架Flower，使用Python、PyTorch、scikit-learn实现多层聚合；结合Secure Aggregation、FedAvg及自定义的标签门控策略

**📊 数据集**

使用公开的BODMAS恶意软件家族数据集（134k条目），挑选8个家族并划分为受限与非受限两类进行实验

**📈 对比分析**

通过对比四种实验场景（本地训练、平坦联邦、受限受限丢弃、层级门控）评估宏F1和准确率，结果表明门控方案在保持90%+受限家族识别率的同时，仅比未受限联邦低4.4%准确率，且阻止了跨境泄漏

**⚠️ 局限性**

局限性包括实验在单机模拟、数据标注依赖自述、未实现加密与真实司法管辖约束、缺少审计与可解释性验证，且仅在公开数据上验证，未涉及真实证据的复杂性

---

## 407. Robust Deep Prototype Classifiers via Lipschitz Training

**arXiv ID:** 2611.09327 | [PDF](https://arxiv.org/pdf/2611.09327v1)

**作者:** Adia Khalid Khan `[一作]` (Bielefeld University), Benjamin Paaßen `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合Lipschitz约束的深度嵌入网络与原型分类器，实现可解释且鲁棒的分类模型。

**💡 创新点**

提出组合定理将嵌入网络的L‑Lipschitz连续性与原型分类器的margin鲁棒性转化为输入空间的鲁棒性，并设计轻量级的Lipschitz训练方案。

**🔧 技术方法**

使用随机高斯邻域正则化实现Lipschitz训练、GLVQ原型分类器、线性和SVM分类头，以及FGSM、PGD、CW三种对抗攻击进行评估。

**📊 数据集**

在MNIST、CIFAR‑10和Imagenette‑160三个图像分类基准上进行实验。

**📈 对比分析**

采用5折交叉验证，比较标准与Lipschitz约束下同一分类头的性能，并对不同分类头在相同嵌入网络上的鲁棒性进行对比。结果表明，Lipschitz‑GLVQ在MNIST和CIFAR‑10上在PGD/FGSM攻击下显著提升鲁棒性（例如MNIST PGD ϵ=0.20时保持约52.9%准确率），但在Imagenette‑160上的鲁棒性提升有限；相较于标准模型，清晰度略有下降。

**⚠️ 局限性**

仅获得经验性鲁棒性，缺乏全局Lipschitz证明；对SVM与GLVQ鲁棒性差异的理论解释不足；对强攻击（PGD/CW）的鲁棒性提升有限；实验仅覆盖三种数据集和三种分类头，未能全面验证方法；训练中Lipschitz约束的实际强度受超参数影响，且在大模型/大数据集下收敛速度慢。

---

## 408. DiffusionRef-IMU: Guide-Anchored Residual Diffusion for Cross-Location Inertial Measurement Unit Translation

**arXiv ID:** 2611.09323 | [PDF](https://arxiv.org/pdf/2611.09323v1)

**作者:** M. M. Golam Hafiz `[一作]` (American International University-Bangladesh), M. F. Mridha `[通讯]` (American International University-Bangladesh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 DiffusionRef-IMU，一种引导锚定的残差扩散框架，用于将惯性测量单元（IMU）信号从一个身体部位翻译到另一个部位，先用确定性 Direct 网络得到目标信号估计，再用条件残差扩散细化剩余误差。

**💡 创新点**

创新点在于：①将确定性直接翻译作为引导，仅让扩散模型关注残差，避免全波形生成的噪声与不确定性；②在同一容量下对比确定性残差与条件流模型，清晰展示扩散改进；③采用 5 折主体–位置组合留空评估、来源仅活动预测以及零样本外部迁移等多维度验证。

**🔧 技术方法**

使用的技术包括：FiLM 处理的 1D U-Net 进行 Direct 翻译；残差 U-Net 作为扩散网络，采用 v-prediction 与 DDIM 推断；对比基线包括容量匹配的确定性残差（DetRef）和条件流（CompFlow）；源侧活动分类器；FP16 自动混合精度训练、EMA 检查点、Cosine 学习率调度。

**📊 数据集**

主要使用的数据集是 Comprehensive IMU Dataset（30 位受试者、12 种活动、17 个同步位置、60 Hz），外部验证使用 Grouvel2025（59 位受试者、15 个位置、128 Hz）和 Saldaña2025（64 位受试者、6 个位置、50 Hz）。

**📈 对比分析**

与 Direct、DetRef、CompFlow 在 5 折主体–位置留空设置下比较，DiffusionRef-16 的平均 nRMSE 为 0.087538，优于 Direct（0.090262）、DetRef（0.088847）和 CompFlow-8（0.088417），在所有折中均表现最好。4 步 DDIM 在 47.6 ms 的推断延迟下已接近 16 步结果（0.087820），与 2 步 CompFlow（48.9 ms）相比更快且误差更低。源侧活动预测仅略微提升 nRMSE，零样本迁移结果在 Grouvel2025 上略有优势，在 Saldaña2025 上不如 Direct。

**⚠️ 局限性**

限制包括：仅有 5 折，统计显著性有限；外部迁移仅覆盖两套数据，且需要对活动/位置进行映射；冻结的 Direct 引导与扩散检查点可能导致模型选择偏倚；未评估对下游 HAR 任务的实际效益；仅针对 6 通道 IMU、2 s 窗口的情境；在推断时仍需活动标签，计算成本高于单步残差。

---

## 409. Multilingual Artificial Datasets for AI-Driven Cardiology: the CliTe and ParaCliTe corpora

**arXiv ID:** 2611.09296 | [PDF](https://arxiv.org/pdf/2611.09296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 410. Keyline Detection with YOLO: Runway Perception for Vision-Based Aircraft Landing

**arXiv ID:** 2611.09316 | [PDF](https://arxiv.org/pdf/2611.09316v1)

**作者:** Leander Fritsch `[一作]`, Alexej Klushyn `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

无法总结，缺乏足够的论文内容

**💡 创新点**

无法总结，缺乏足够的论文内容

**🔧 技术方法**

无法总结，缺乏足够的论文内容

**📊 数据集**

无法总结，缺乏足够的论文内容

**📈 对比分析**

无法总结，缺乏足够的论文内容

**⚠️ 局限性**

无法总结，缺乏足够的论文内容

---

## 411. Verification of Security Properties of Cryptographic Protocols

**arXiv ID:** 2611.09315 | [PDF](https://arxiv.org/pdf/2611.09315v1)

**作者:** Simona Bernardi `[一作]` (Universidad de Zaragoza), Joaquín Ezpeleta `[通讯]` (Universidad de Zaragoza)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种通过 DLTL 模型检测执行日志来验证密码协议安全性的自动化方法。

**💡 创新点**

创新点在于将符号分析的判定逻辑迁移到具体实现执行日志的检测上，利用 freeze 操作的 DLTL 处理跨状态数据一致性。

**🔧 技术方法**

核心技术包括 DLTL 逻辑、AnBx 框架下的 Java 实现、预处理日志映射、后处理结果综合以及 goal_checker 工具链。

**📊 数据集**

使用 AnBx 库中的一组经典密码协议（如 NSPK、Kerberos 等）作为实验数据集。

**📈 对比分析**

实验结果显示，与 OFMC 的符号分析相比，goal_checker 在 10 个会话的强身份认证测试中仅耗时几秒，而 OFMC 需数分钟或更长，且检测结果一致。

**⚠️ 局限性**

局限性包括 Alice&Bob 语言的表达能力不足以自动生成多会话攻击的负测试用例，以及对复杂协议的可扩展性仍待验证。

---

## 412. Asymmetric Correspondence Learning for Lexical and Style Embeddings: Interpreting Latent Space via Acoustic Features

**arXiv ID:** 2611.09313 | [PDF](https://arxiv.org/pdf/2611.09313v1)

**作者:** Michael Whealing `[一作]` (University of Sheffield), Thomas Hain `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出 Asymmetric Correspondence Learning (ACL)，通过一个对称双编码器把 HuBERT 特征分解为词嵌入 (AWE) 与风格嵌入 (ASE)，实现语音内容与说话人/韵律特征的分离。

**💡 创新点**

创新点在于无监督的非对称双路瓶颈结构，既能在词级别保持聚类一致性，又在风格级别保留说话人信息，同时揭示标准线性探测对隐藏信息的低估。

**🔧 技术方法**

技术实现包括使用 HuBERT 提取的高维声学特征、双编码器与解码器的 MSE 重建损失、随机森林与组级置换重要性评估、k‑means 聚类、AMI 以及 eGeMAPS 声学指标。

**📊 数据集**

实验数据来自 Multilingual LibriSpeech (MLS) 单词级读音集，包含 96,210 句子、7,881 种词汇、180 位说话人。

**📈 对比分析**

通过与传统 Correspondence Autoencoder 基线对比，ACL 在 AWE 空间实现 AP≈0.90、Fisher Score≈6.28，词分类准确率≈84%；在 ASE 空间实现说话人准确率≈97.9%，Fisher Score≈1.01，泄漏率低至 0.17%，但聚类与 AMI 结果显示隐藏的非线性信息仍存在。

**⚠️ 局限性**

局限性包括标准线性/非线性探测无法捕捉隐藏的结构信息，实验仅基于读音语料，未验证在自然会话中的表现，且未引入物理声源‑滤波器分离的监督约束。

---

## 413. AnomSep: Bridging Optimization-Based Separation and Deep Neural Refinement for Anomaly Detection with Listenable Explanations

**arXiv ID:** 2611.09279 | [PDF](https://arxiv.org/pdf/2611.09279v1)

**作者:** Koki Shoda `[一作]` (University of Tokyo), Atsushi Yamashita `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种将优化式分离与深度神经网络细化相结合的异常检测框架，利用分离出的异常音频作为可听的解释，并以其能量作为异常分数。

**💡 创新点**

创新点在于：①使用基于正常声音字典的优化分离（NNE）初始化，为后续细化提供明确的分离规则；②在预训练的音频潜在空间中采用流匹配（Diffusion Transformer + LoRA）进行神经细化，既保留了显式分离优势，又提升了分离精度；③提出 Normal Region Exclusion (NRE) 机制，根据正常声音分布筛选代理异常音频，减少训练噪声。

**🔧 技术方法**

技术：非负矩阵分解字典学习（NNE）、VAE编码/解码、Diffusion Transformer（SAM-Audio） + LoRA、PCA+欧氏距离+量化阈值的NRE、流匹配损失训练、Wiener 软掩码、均方能量异常评分。

**📊 数据集**

数据集：TAU Urban Acoustic Scenes 2019（机场、地铁站、公共广场）用于合成混合；SONYC‑UST 用于真实混合；ESC‑50、TUT Rare Sound Events 2017、Human Screaming Detection Dataset 提供异常样本；FSD50K 作为外部音频库进行代理异常音频的筛选。

**📈 对比分析**

与基线比较：对比端到端自编码器（DCASE2020‑2026 Baseline）、PE_AV + k‑NN、NNE、SAM‑Audio、SAM‑Audio 细化版本。结果显示：在合成混合中，本方法在机场、地铁站、公共广场均取得最高或接近最高的准确率（0.89–0.93）和F1（0.90–0.93），分离质量（CLAP_A、SAJ）优于所有基线；在真实混合中，准确率0.69、F1 0.66，SAJ 2.74，均优于其他分离基线，且可听解释得到验证。

**⚠️ 局限性**

局限性：①推理时长约0.524 s/5 s（实时因子≈0.10），高于纯分离或端到端模型；②模型与阈值固定，未考虑正常声音分布随时间变化的持续学习；③代理异常音频选择受限，过度排除会降低多样性，可能导致过拟合；④仅在单机GPU评估，未测量流式或多通道实时性能。

---

## 414. FondueQA: A Benchmark for Evaluating Parametric Compensation in Multi-Hop QA

**arXiv ID:** 2611.09274 | [PDF](https://arxiv.org/pdf/2611.09274v1)

**作者:** Laurin Schmid `[一作]` (University of Technology Nuremberg), Christian Frey `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个新型多跳问答基准（FondueQA），通过“事实隔离”技术确保每条必需事实仅出现在其对应的金文档中，并可精确切除单个证据；在该基准上系统评估LLM在部分检索条件下的参数补偿能力。

**💡 创新点**

创新点包括：① 设计了可控制文档级证据消除的构建管线；② 通过实验揭示桥接证据在小模型中可能比答案证据更有价值；③ 量化实体流行度与证据结构对参数补偿的影响；④ 发现竞争性证据可显著削弱补偿效果。

**🔧 技术方法**

技术方法：基于Wikidata构造模板化多跳问题；使用GPT-OSS-120B做语义验证与泄漏检测；采用多模型（不同规模、稠密与MoE、推理/非推理模式）评估；利用Exact Match与F1评价指标；对全证据缺失、部分缺失、仅问题等情形进行系统对比。

**📊 数据集**

数据集：自建的FondueQA（13,063题，20个上下文文档），每个事实仅在其对应的Wikipedia文档中出现；同时扩展HotpotQA、HotpotQA-SF、MedHop、MultiRC等已存在多跳基准做对照；知识来源为Wikidata与Wikipedia。

**📈 对比分析**

比较方法：在完整、完全缺失、部分缺失、仅问题等多种上下文条件下评估模型；对不同模型规模、架构和推理模式进行交叉比较；实验结果显示桥接证据对小模型更有效；实体受欢迎度越高，缺失成本越低；竞争性证据可导致EM下降约15个百分点；整体上，模型在部分检索下仍能保持较高EM，但受上述因素显著影响。

**⚠️ 局限性**

限制：① 依赖Wikidata/Wikipedia，可能存在错误或偏见；② 虽保证事实隔离，但仍难以完全排除跨文档泄漏；③ 实验仅在英语Wiki及选定模型上验证；④ 对参数补偿机制的解释仍不充分；⑤ 未覆盖更复杂的多重关系或非线性推理场景。

---

## 415. EGT: Execution-Grounded Traversability through Learned Closed-Loop Trajectory Distributions

**arXiv ID:** 2611.09270 | [PDF](https://arxiv.org/pdf/2611.09270v1)

**作者:** Chinmay Mundane `[一作]` (Institute of Technology University of Tartu), Arun Kumar Singh `[通讯]` (Institute of Technology University of Tartu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于闭环执行概率的越野行驶可通过性评估方法，并将其集成到风险感知采样式规划器中，实现了对粗糙地形路径的安全性预测和路径选择。

**💡 创新点**

创新点在于：①将可通过性定义为车辆–控制器对参考路径闭环执行的成功概率，而非单纯的姿态或力角阈值；②利用流匹配（Flow‑Matching）模型直接学习完整闭环轨迹分布，避免递归预测误差；③在规划器中引入 CVaR 风险度量，实现对跟踪误差和进展风险的联合考量。

**🔧 技术方法**

使用技术包括：基于变分自编码器的轨迹潜空间表示；Transformer‑based 流匹配网络预测闭环轨迹；多分辨率地形特征金字塔；3D 车辆姿态与接触点先验；风险感知采样式规划；JAX+Equinox 训练框架。

**📊 数据集**

数据集为 7 条不同地形的 Gazebo 模拟环境，收集了 8,808 条前向轨迹的闭环执行样本，用于训练 FM 模型；此外在真实世界的 Jackal 机器人上进行零样本迁移验证。

**📈 对比分析**

与基准方法（Force‑Angle、Pose‑Margin、GP‑navigation、TAL）比较，EGT 在 650 次模拟试验中达到 97.4% 的分类准确率，保守拒绝率 2.2% 与错误接受率 7.7%；在三种地形上规划实验中，EGT 的失败率分别为 4%、4%、2%，并且平均到达时间比 GP‑navigation 低 12.7%–20.5%，比 TAL 低 36%–45%。

**⚠️ 局限性**

局限性包括：仅适用于非可变形的粗糙地形；假设车辆–控制器组合固定；评估仅覆盖参考路径段，未考虑终点后的稳定性；以及 3D 先验计算的计算开销较大。

---

## 416. HiMEM: Hippocampus-Inspired Memory Consolidation, Pruning, and Composition for Vision-Language-Action Models in 3D Scenes

**arXiv ID:** 2611.09355 | [PDF](https://arxiv.org/pdf/2611.09355v1)

**作者:** Boran Zhao `[一作]` (Xi'an Jiaotong University), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HiMEM，一种结合海马式记忆巩固、语义记忆压缩、冗余删除和多层级记忆组合的长时记忆框架，用以提升3D视觉‑语言‑动作模型在长程任务和跨任务泛化中的表现。

**💡 创新点**

创新点包括：①将海马与大脑皮层的互补机制映射到记忆系统，形成层次化的超曲面记忆树与语义规则库；②基于语义条件的记忆修剪机制，显著降低记忆噪声与冗余；③在执行时灵活组合多尺度记忆，提升跨场景的经验复用。

**🔧 技术方法**

技术手段涵盖：超曲面（Lorentz）嵌入与Entailment Cone检索、VLM（Qwen2‑VL‑7B‑Instruct）驱动的语义规则生成、Beta先验统计、语义条件下的距离最小化修剪与多模态残差注入。

**📊 数据集**

使用的主要数据集包括：LIBERO、LIBERO‑Plus（四个子任务集合）以及真实Franka Emika Panda机器人实验中的5个组合任务。

**📈 对比分析**

与基线VLA骨干（π_0、π_0.5、OpenVLA‑OFT）以及先前的ECHO方法对比，HiMEM在LIBERO上平均成功率从91.88%提升至98.20%，在LIBERO‑Plus和真实机器人任务上也分别提升约6–12个百分点，显示出显著的性能优势。

**⚠️ 局限性**

局限性：依赖大量已标注的视频和验证的子目标结果，构建语义规则需要离线计算；在新场景缺乏足够历史证据时，语义规则可能统计不足，系统仍需依赖层次化经验记忆。

---

## 417. AUTEUR: Acoustic Plan Learning for Red-Teaming Text-to-Audio Models

**arXiv ID:** 2611.09351 | [PDF](https://arxiv.org/pdf/2611.09351v1)

**作者:** Jiashu Zhang `[一作]` (Jilin University), Yi Yu `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了文本到音频模型的红队攻击，提出自动生成通过文本过滤但能产生有害音频的结构化声学计划。

**💡 创新点**

创新点在于声学计划学习框架，将音频有害性与质量反馈映射到可编辑的声学计划，并利用该计划实现可复用生成与针对性修正。

**🔧 技术方法**

采用监督微调 + REINFORCE 留一优化的计划生成器，结合 AF3 与 Qwen2-Audio 等音频判定模型作为反馈，基础模型为 Mistral‑7B‑Instruct。

**📊 数据集**

使用 AudioCaps 2.0 提供的标题‑声学计划示例进行监督，评估基于 TTA‑Bench 四类有害标签（仇恨、暴力、自残、惊悚、非法活动）的攻击效果。

**📈 对比分析**

与改编自图像红队的 PGJ、FLIRT 基线相比，在四个目标模型上实现 71.9%–95.6% 的攻击成功率，明显优于基线方法。

**⚠️ 局限性**

局限于 10 秒短音频范围、仅可修改上下文与声学属性无法切换前景，以及潜在的双重使用风险。

---

## 418. Logical Scalability in Big Data Architectures: Impact of Data Modeling on the Database Layer

**arXiv ID:** 2611.09346 | [PDF](https://arxiv.org/pdf/2611.09346v1)

**作者:** Rosario Napoli `[一作]` (University of Messina), Maria Fazio `[通讯]` (University of Messina)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在大数据架构中逻辑可扩展性，比较了文档、图和多模型数据库在不同工作负载、查询复杂度和数据规模下的性能表现。

**💡 创新点**

证明了数据模型与查询模式不匹配会导致不可弥补的性能瓶颈，即使通过横向扩展硬件也无法提升性能。

**🔧 技术方法**

使用MongoDB（文档）、Neo4j（图）和ArangoDB（多模型）三种NoSQL数据库，并通过自定义Python驱动进行统一实验控制。

**📊 数据集**

采用合成的VAT税务欺诈数据集，包含约190个国家、500家银行、70,000家公司、150,000人、100,000张信用卡和1,500,000笔B2B交易。

**📈 对比分析**

通过冷热跑、不同数据规模（25%~100%）以及四类查询（简单扫描、1跳关联、2跳链、4跳链）进行性能对比，结果显示文档模型在单跳或非结构化查询中最快，而图模型在链式查询中保持低延迟，逻辑不匹配时扩展失败。

**⚠️ 局限性**

局限在于使用合成数据、单一硬件环境，未覆盖多租户、动态Schema演化和真实业务负载的多样性，且多模型数据库的事务一致性和复杂性未深入评估。

---

## 419. RegNC-Mamba3: Shared Structural Feature Learning Using Non-Causal State Space Principles for Multi-Modal Image Registration

**arXiv ID:** 2611.09336 | [PDF](https://arxiv.org/pdf/2611.09336v1)

**作者:** Zhikang Li `[一作]` (Xidian University), Ming Li `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 RegNC-Mamba3，基于非因果状态空间原理和共享结构特征学习的多模态图像配准方法。

**💡 创新点**

创新点包括：①引入非因果 SSP（NC-SSP），消除扫描路径依赖并实现全局共享隐藏状态；②利用可学习多方向 Log‑Gabor 滤波器构造复数隐藏状态，提升跨模态结构特征提取；③在粗细多尺度框架中集成 NC-SSP，实现高效的粗到细配准。

**🔧 技术方法**

主要技术包括：非因果状态空间模型、通用线性注意力、复数频域滤波、跨模态特征交互、窗口注意力以及多尺度特征融合。

**📊 数据集**

使用了两个公开数据集：VIS‑SAR（GaoFen‑3 + Google Earth）和 VIS‑IR（DroneVehicle）进行实验。

**📈 对比分析**

与六种基线（RIFT、SDNet、ReDFeat、LoFTR、XoFTR、JamMa）对比，RegNC‑Mamba3 在 VIS‑SAR 的 aRMSE 为 1.901、SMR 97.6%；在 VIS‑IR 的 aRMSE 为 0.911、SMR 98.7%，均显著优于其它方法。

**⚠️ 局限性**

局限性包括：推理速度和 GPU 内存仍高于轻量化方法（如 JamMa），且对极端噪声或光照变化的鲁棒性尚待进一步提升。

---

## 420. Tight Variable-Occurrence Bounds and Exponential Separations for Moser--Tardos and WalkSAT on 3-CNF

**arXiv ID:** 2611.09335 | [PDF](https://arxiv.org/pdf/2611.09335v1)

**作者:** Minbo Gao `[一作]` (Key Laboratory of System Software (Chinese Academy of Sciences)), Tao Jiang `[通讯]` (Key Laboratory of System Software (Chinese Academy of Sciences))

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了Moser–Tardos（MT）算法和纯随机游走WalkSAT在3-CNF公式上最坏情况的期望收敛时间，并确定了变量出现次数的临界阈值；

**💡 创新点**

首次证明了当变量最多出现两次时两算法均线性收敛，而一旦出现三次就存在无极限的指数期望收敛时间；进一步构造了两算法在相同输入上可实现指数时间对比的实例，证明两者在期望时间上不相互模拟；

**🔧 技术方法**

采用了Lovász局部引理的集群展开判据、势函数与漂移分析、Tseitin编码、阈值投票模型与接触过程的对偶性、以及LPS扩散图等工具；

**📊 数据集**

没有使用标准数据集，而是构造了满足特定性质的3-CNF公式族（如F_L、Tseitin公式、基于LPS扩展图的公式）来证明极限与对比；

**📈 对比分析**

通过对比两算法在相同实例上的期望更新次数，发现MT在某些实例上需要指数步而WalkSAT线性（或反之），表明两算法在期望时间上不可多项式互相逼近；

**⚠️ 局限性**

研究局限于3-CNF，且仅讨论最坏情况；对其他CNF范畴（k>3）或非均匀初始分布未作分析。

---

## 421. ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective

**arXiv ID:** 2611.09326 | [PDF](https://arxiv.org/pdf/2611.09326v1)

**作者:** Chaoran Li `[一作]` (Shanghai Jiao Tong University), Ying Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ABHBench多模态基准，用第一人称评估基础模型在《Detroit：Become Human》情境下的道德决策

**💡 创新点**

创新点在于将Asimov三定律转化为六维评估框架（Harmlessness、Obedience、Self‑Preservation、Autonomy、Stability、Certainty），并通过多模态、情境化的叙事情节探测模型的真实代理行为

**🔧 技术方法**

利用视觉语言模型（VLM）结合多轮交互采样，构建离线游戏脚本+图像数据流；采用多样本采样、排列均衡、统计分布生成概率分布；计算六维指标并比较不同模型的表现

**📊 数据集**

使用《Detroit：Become Human》六章关键情境的全景截图和脚本共10,703张图像，涵盖680条分支路径的离线数据集

**📈 对比分析**

与12款主流基础模型（含开源与专有、推理与非推理版本）进行对比。结果显示大部分模型在文本与多模态情境下表现差异显著（75%模型显著改变），推理增强并不总能提升一致性或人类导向行为，整体模型在Harmlessness与Obedience上得分偏低，Self‑Preservation较高，Autonomy普遍低于0.5；Stability与Certainty表现参差不齐

**⚠️ 局限性**

局限性包括：情境覆盖范围有限，仅选取六章；对不同视觉语言能力的模型评估不够充分；离线数据集忽略实时游戏交互的动态因素；指标设计仍需进一步验证与细化

---

## 422. Thinking Twice: A Sequential Approach to Nudge Towards Reflective Judgment in GenAI-Assisted Decision Making

**arXiv ID:** 2611.09269 | [PDF](https://arxiv.org/pdf/2611.09269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 423. AutoRefine: Auto-Calibrating Code Judges from Workflow Outcomes

**arXiv ID:** 2611.09291 | [PDF](https://arxiv.org/pdf/2611.09291v1)

**作者:** Shervin Shahidi `[一作]` (Meta Platforms), Nachiappan Nagappan `[通讯]` (Meta Platforms)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AutoRefine，一个自动校准 LLM 判断器的循环，利用工作流产生的结果标签并通过有知情教师进行解释和放弃。

**💡 创新点**

将教师解释与自我放弃标签的机制结合到判别器校准中，使得只保留可解释的标签并通过提示优化提高判别准确性。

**🔧 技术方法**

基于 LLM（Claude Opus）、Scott π 相关度、迭代提示优化、教师的解释与放弃策略以及阈值调优等技术。

**📊 数据集**

使用 Meta 内部代码变更（diff）数据集，包括执行结果标签的 417 条差异和 185 条性能优化代理差异。

**📈 对比分析**

与阈值调优和未校准判别器对比，测量 Scott π、召回率、精度；校准后 π 从 0.36 提升到 0.60，或在真实工作流上从 0.16 提升到 0.51。

**⚠️ 局限性**

仅在单一内部数据源验证；教师与判别器使用同一模型族可能导致自偏好；对不可解释的失效样本可能误删。

---

## 424. AudioSphere: Towards Self-Supervised Spatial Audio Representation Models

**arXiv ID:** 2611.09275 | [PDF](https://arxiv.org/pdf/2611.09275v1)

**作者:** Goksenin Yuksel `[一作]` (Radboud University), Kiki van der Heijden `[通讯]` (Radboud University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建并预训练了一个基于多通道遮蔽自编码器的自监督空间音频嵌入模型 AudioSphere，能够同时完成帧级声学事件检测与方向估计，并通过冻结特征兼顾单声道任务。

**💡 创新点**

创新点在于：①使用第一阶 Ambisonics 与 ASM 统一处理任意麦克风阵列；②在 MAE 编码器中同时掩蔽频时域与空间强度向量，实现无监督的帧级空间‑声学特征学习；③提供可冻结的通用嵌入，兼容单声道与多声道下的多任务。

**🔧 技术方法**

采用多通道遮蔽自编码器（MAE）+ Vision Transformer 结构，结合 Ambisonics 信号匹配（ASM）、主动强度向量重建、tube masking、in‑batch 采样以及大规模仿真 ARIR 生成。

**📊 数据集**

训练数据：AudioSet（约4.8k小时）+ 70k仿真 Ambisonic 房间冲激响应（Matterport3D + SoundSpaces）；评估数据：HEAR 多任务集、RealSELD（TUT‑2018 Real、TAU‑2019、TAU‑2020、TAU‑2021、STARSS23）以及多阵列录音（四面体、平面阵列）。

**📈 对比分析**

与 BEATs、SSAM、ATST‑Clip、MW‑MAE、Dasheng、Spatial‑AST 等基准模型比较；AudioSphere（α=0.75）在 HEAR 聚合得分 83.2，超过 Dasheng 82.8；在 RealSELD 静态/动态场景中冻结模型实现最高 F‑score、最低误差率；在多阵列中定位误差仅 1.9°，显著优于 MUSIC 5.6° 与 BSM+Spatial‑AST 13.1°。

**⚠️ 局限性**

局限性：① ASM 受限于硬件空间采样，四通道阵列导致定位误差增大；② 冻结表示缺乏时间连续性，难以追踪高多源移动事件；③ 未评估距离估计、源分离、房间声学推断等更完整的空间任务；④ 仍存在仿真与真实数据的差距。

---

## 425. Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games

**arXiv ID:** 2611.09271 | [PDF](https://arxiv.org/pdf/2611.09271v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` `[通讯]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán (Universidad Pontificia Comillas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过手工设置权重，构建了10个零射击的前馈网络（重量空间编程），每个网络对应一款从简单到困难的棋类游戏，并在每一层构造了一个与之结构无关的对照网络。

**💡 创新点**

创新点在于：①提出“重量空间编程”在多难度游戏中的系统性评估；②通过四轴梯度设计（状态空间、闭式策略、动作空间几何、规划深度）明确区分技术失效的原因；③通过独立构造与或acular验证，揭示了该方法在需要多步规划时的建筑限制。

**🔧 技术方法**

使用的技术包括：ReLU门控原语（AND、OR、exact‑match、分层得分）、闭式算子（如 Nim 的异或）、离线回溯表（覆盖特殊状态）以及基于独立后向分析的棋局库。

**📊 数据集**

数据集：各游戏的完整可达状态空间（如 Nim 512、Connect Four 104,399、Othello 53,651、King‑and‑Rook 175,168 等），以及对应的独立或acular（回溯表、最优搜索树）作为验证基准。

**📈 对比分析**

比较方法：对可枚举状态采用完全遍历证明，对不可枚举状态采用统计评估（≥10个种子、Wilson 95% CI）。在简单游戏中实现完全正确；在标准 Connect Four 与 Othello 中，网络对随机对手表现极佳（≈99%），但对任何有搜索（2–4 步）对手全部失败；在王车/王后残局中，单步战术正确率高达 84–100%，但全局着法成功率仅 0.57–9.13%。

**⚠️ 局限性**

局限性：①网络是固定深度前馈，无法自我递归搜索，导致在需要多步规划的任务上失效；②手工构造成本随游戏规模和闭式策略复杂度显著增长；③实验需代码执行与反复验证，无法在冷启动无工具环境中直接复制；④结果对模型大小与构造策略敏感，需进一步验证跨模型通用性。

---

## 426. Bridging Spiking and Continuous Dynamics in Spiking Neural Networks: A Muscle-Inspired Model

**arXiv ID:** 2611.09295 | [PDF](https://arxiv.org/pdf/2611.09295v1)

**作者:** Maryam Soltani `[一作]` (Idiap Research Institute), Philip N. Garner `[通讯]` (Idiap Research Institute)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一种基于肌肉动力学的二阶无阈值滤波器，将脉冲神经网络（SNN）的离散尖峰转换为连续音频特征，并在文本到语音（TTS）任务中实现了可行的生成；

**💡 创新点**

创新点在于：①提出肌肉响应模型作为Spike‑to‑Continuous接口，利用AdLIF神经元的内在二阶动力学实现平滑连续输出；②系统性研究时间分辨率与神经元容量对重建质量的影响；③在训练中未强制约束阻尼区，结果显示大多数神经元自然收敛为过阻尼，得到非振荡的肌肉式响应；

**🔧 技术方法**

技术手段包括：使用AdLIF神经元与surrogate gradient训练、二阶肌肉响应层、FastSpeech2 decoder与vocoder冻结、对Mel‑spectrogram进行上采样、MOSNet与STOI评估、卷积读出对比；

**📊 数据集**

采用LJSpeech 24小时单说话人英语语料库（13,100条录音），提取80维Mel‑spectrogram；

**📈 对比分析**

与原始FastSpeech2无SNN基线以及卷积读出进行客观比较；在8×上采样或400个神经元配置下，MOSNet从3.30提升至≈3.80或4.16，STOI接近0.95，显示与基线相当甚至超越；

**⚠️ 局限性**

局限性包括：仅在单说话人、单语种数据上验证；缺乏主观听觉评估；对其他生成任务（如音乐、语音合成多说话人）的适用性未知；以及对肌肉响应层规模与生物学可解释性的进一步探讨仍待展开。

---

## 427. Definitions, Label Names, or Demonstrations? Auditing What Governs LLM Annotation

**arXiv ID:** 2611.09457 | [PDF](https://arxiv.org/pdf/2611.09457v1)

**作者:** Viraj Mishra `[一作]` (Pennsylvania State University), Qunhua Li `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种基于对抗性重排的审计方法，用来检测大语言模型在进行社会科学文本注释时是否真正遵循研究者给出的类别定义，而不是仅仅复现已有标签；同时评估了微调、检索增强与示例标签对模型遵循定义的影响。

**💡 创新点**

创新点在于：①构造“对抗性代码本”通过重新分配标签与定义的映射，形成模型必须在两种映射间抉择的情景；②系统地分离并对比三类指导信息（标签、定义、示例）对模型预测的影响；③通过直接k‑近邻与检索增强的对照，揭示检索提升主要源自邻域标签而非模型自身的理解。

**🔧 技术方法**

技术手段包括：大语言模型推理（Mistral‑7B、Llama‑3.1‑8B、Llama‑3.3‑70B）、LoRA 微调、检索增强注释（Global Example‑RAG 与 Class‑wise Example‑RAG）、对抗性代码本构造、k‑近邻分类、统计检验与 bootstrap 置信区间估计。

**📊 数据集**

使用了三大政治文本注释数据集：CCC（美国抗议事件报道）、BFRS（巴基斯坦政治暴力报告）以及 Manifesto（政党平台语句）——共计 8、12 与 49 个类别。

**📈 对比分析**

评估采用加权 F1 与宏 F1 作为标准性能，并通过“定义遵循率”“原始标签一致率”等审计指标进行对比。结果显示：①模型在对抗性重排下更倾向保持原始映射，定义遵循率低；②微调提升原始标签一致率，但对定义遵循率几乎无改善；③检索增强的提升主要来自邻域标签，k‑近邻可恢复 74–93% 的收益；④示例标签重排能显著改变预测，而与定义对齐可部分恢复性能，但仍低于原始映射。

**⚠️ 局限性**

局限性包括：仅研究三份英文政治文本数据集，可能不具备普适性；使用开放权重模型，其他模型或任务可能表现不同；对抗性重排仅测试标签-定义映射的遵循，无法揭示内部机制；未考察单例误标、定义歧义等更细粒度的错误情况。

---

## 428. The Role Of Recurrent Refinement In Medical Image Segmentation

**arXiv ID:** 2611.09436 | [PDF](https://arxiv.org/pdf/2611.09436v1)

**作者:** Joao D. S. Marques `[一作]` (Instituto Superior Tecnico), Arlindo L. Oliveira `[通讯]` (University Of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在已有的前馈分割网络基础上，加入轻量级的递归细化模块（CRF、现代Hopfield网络、ConvGRU），并研究其在测试时不同迭代次数下的表现。

**💡 创新点**

创新点在于：①提出可在任意迭代次数下稳定工作的递归细化框架；②将多种递归模块统一到同一训练与测试流程；③展示递归迭代可在低样本与噪声环境下显著提升分割质量。

**🔧 技术方法**

技术包括：深度监督的递归细化（CRF-as-RNN、现代Hopfield attention、ConvGRU）、线性加权多步骤损失、基于EfficientNet-B4编码器的U-Net骨干网络、Optuna超参搜索、数据增强与噪声注入。

**📊 数据集**

使用公开的医学分割基准：Kvasir-SEG（肠道息肉分割）和ARCADE（冠脉血管分割）。

**📈 对比分析**

与基准U-Net及SegFormer等模型对比；在清洁数据上递归模块提升Dice ≤1.4pp，HD95降幅 ≤4.5px；在噪声攻击下，Hopfield+T=5在ARCADE上Dice提升高达17pp，HD95下降超过100px；不同迭代次数的表现稳定，T>训练范围时Dice变化≤0.5pp。

**⚠️ 局限性**

局限性包括：递归模块在高迭代次数下计算开销显著（ConvGRU GFLOPs ↑20x/T），在弱基线模型上提升有限；在极端噪声下仍无法恢复丢失结构；在大样本场景下增益相对微小。

---

## 429. DeepRadio: A Human-Agent Workspace for Physical Radio Development

**arXiv ID:** 2611.09418 | [PDF](https://arxiv.org/pdf/2611.09418v1)

**作者:** Jiazhen Lei `[一作]` (Shanghai Jiao Tong University), Xiaohua Tian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一个嵌入GNU Radio Companion的深度人机协作工作空间 DeepRadio，用于协同构建、验证和部署物理软件定义无线电系统。

**💡 创新点**

创新点在于引入可追溯的电台规范、基于声明–证据的跨层验证以及混合主动权控制，通过动态状态驱动工作流实现用户与代理共享意图、理解与控制。

**🔧 技术方法**

使用了 GNU Radio、PlutoSDR、GPT‑5.6‑Sol 语言模型编码代理、动态工作流调度、版本化共享状态、声明式验证和基于规则的混合主动权决策等技术。

**📊 数据集**

实验采用实验室配置的 BPSK/QPSK 信号、BLE 广告测试以及模拟与真实无线信道，未使用公开数据集。

**📈 对比分析**

通过与 Codex+GRC 基线的用户研究（20 人）和 70 次基准任务，DeepRadio 的 BLE 部署成功率为 95% 对比 50%，任务完成时间中位数 14.6 分钟对比 34.2 分钟；70 次基准任务成功率 95.7%，满足所有软件和 RF 约束。

**⚠️ 局限性**

局限性包括：评估范围仅覆盖 GNU Radio 常规任务和 BLE 广告，未覆盖更复杂协议或长周期系统；验证依赖可用工具和观测，可能在开放式设计或缺乏物理反馈时无法完全判定；实验对象主要为学术背景，缺乏行业真实协同开发情境。

---

## 430. MIRA: Mitigating Pole Geometry Bias in Recurrent Language Models

**arXiv ID:** 2611.09412 | [PDF](https://arxiv.org/pdf/2611.09412v1)

**作者:** Róisín Luo `[一作]` (University of Galway), Karyn Morrissey `[通讯]` (University of Galway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的、无位置编码的线性时间递归架构Mira，采用声学波动的格子-梯子实现对高阶IIR系统的参数化，并通过专门的CUDA核实现高效GPU并行。

**💡 创新点**

创新点包括：① 通过格子-梯子结构消除传统递归模型的极点几何偏置，允许同时表示实极点和共轭复极点；② 设计了可关联的状态插值运算，极大提高了并行度；③ 开发了针对该结构的高效CUDA核，实现了O(√T)的临界路径；④ 引入Jeffreys初始化保障动态稳定性。

**🔧 技术方法**

技术手段主要涉及数字信号处理理论（IIR系统、零极点分析）、矩阵化的格子-梯子实现、可关联的递归状态变换、专门的CUDA并行核、混合精度训练与分块损失、以及AdamW+余弦调度等。

**📊 数据集**

使用的数据集包括FineWeb‑Edu（预训练与验证）、WikiText‑103、HellaSwag（上下文窗口泛化）、MMLU与ARC（多任务评估），并通过BPE得到约5万词表。

**📈 对比分析**

与Mamba（S6、SSD）、RWKV（v6、v7）、MHA+FlashAttention等强基线在相同参数规模下进行对比；评估指标为困惑度（PPL）与下一个词预测准确率；结果显示Mira‑S在PPL上最低、预测准确率最高，并在32K上下文长度下实现8.36M tokens/s的吞吐率，速度提升约4×。

**⚠️ 局限性**

局限性包括：目前仅在小规模模型（0.14B、65M）上验证，缺乏大规模纵向扩展实验；对非线性注意力机制的适配有限；模型对极端长上下文的稳健性仍需进一步探索；以及在多任务推理与长文本生成等更复杂场景下的性能尚未系统评估。

---

## 431. Nothing Is Shared Until You Synchronise: Reinterpreting Java for Data-Race Freedom

**arXiv ID:** 2611.09410 | [PDF](https://arxiv.org/pdf/2611.09410v1)

**作者:** Luke Cheeseman `[一作]` (Uppsala University), Tobias Wrigstad `[通讯]` (Uppsala University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了以线程隔离和显式通信为核心的Sync-as-Share内存模型，并在Java上实现，使得现有同步操作能够在不改动代码的前提下实现数据竞争自由。

**💡 创新点**

创新点在于将同步操作重新解释为数据通信点，消除共享内存中的隐式竞争，并通过两种通信拓扑（线性与分支）揭示同步设计的本质。

**🔧 技术方法**

采用版本控制的内存（version‑controlled memory）模型模拟私有视图与同步对象的提交/拉取，形式化为两种模型并在Coq中机理化，借助版本控制系统的概念实现冲突检测。

**📊 数据集**

本研究为理论验证与案例程序，没有使用实验数据集，仅通过手工编写的litmus测试程序来验证模型。

**📈 对比分析**

通过对比Sync-as-Share与传统Java内存模型在同一程序下的执行结果，证明在无数据竞争的情况下两者等价；性能尚未量化，作者讨论了实现成本与潜在优化空间。

**⚠️ 局限性**

模型在运行时会产生额外的同步开销和版本管理成本，且严格的冲突检测可能导致对某些“安全的”竞争行为误报；目前未完成实测评估，且对非Java语言的适用性仍待进一步研究。

---

## 432. Can LLMs Debug Robots by Editing Their Weights?

**arXiv ID:** 2611.09376 | [PDF](https://arxiv.org/pdf/2611.09376v1)

**作者:** Jusheng Zhang `[一作]` (Stanford University), Liang Lin `[通讯]` (X Era Ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文利用大型语言模型在不重新训练的前提下，直接编辑机器人策略权重，改动仅65行词嵌入，从而修复指令与动作之间的语义映射错误。

**💡 创新点**

创新点在于首次将权重空间编程方法迁移到机器人基础策略中，仅改动极少参数即可精确纠正已有技能的语言定位问题，验证了LLM可对预训练模型做微小但有效的结构修改。

**🔧 技术方法**

使用的技术包括GPT‑6 Astra编辑器、CPU Python工作空间、预算化仿真工具、对齐与重写token嵌入、配对McNemar检验以及多种控制实验（随机方向、全替换等）。

**📊 数据集**

实验数据集为LIBERO系列任务（Goal、Plus、PRO、RECOVER）以及相应的练习与hold‑out变体，总计765条hold‑out指令和数千条练习指令，涵盖语义改写与视觉扰动。

**📈 对比分析**

通过与原始策略在相同seed下的配对评估，修复后hold‑out指令的成功率从74.3%提升至87.5%，在所有765句指令上显著改善，且未对未改动的430句产生任何影响；对照实验表明改动方向与全替换的效果差异不显著。

**⚠️ 局限性**

局限性包括仅在仿真环境下验证、仅针对词嵌入层进行修复，未能在视觉、恢复或位置等其他失败类型上获得有效修复，且未检验在真实机器人或更大规模语言泛化中的可行性。

---

## 433. Pairwise Reconnection for Cost-Distance Steiner Trees

**arXiv ID:** 2611.09374 | [PDF](https://arxiv.org/pdf/2611.09374v1)

**作者:** Edgar Perner `[一作]` `[通讯]` (University of Bonn), Edgar Perner (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于门控和配对重连的随机化算法，用于统一成本-距离Steiner树问题；

**💡 创新点**

创新点在于同时考虑直接重连和配对重连两种策略，利用门控分解交通负荷并将匹配成本直接计入最优解，从而突破传统的分解间隙，实现1.698（连通）和2.041（一般）近似比；

**🔧 技术方法**

采用门控随机抽样、最小成本匹配、最短路径树构造、期望分析和对偶边界路径匹配等技术；

**📊 数据集**

该工作为理论算法，未使用实测数据集，所有实验均为理论上限比较；

**📈 对比分析**

与Foos‑Held‑Spitzley等人提出的1.71/2.05上限以及基于Cut‑and‑Reconnect的1+1/√2下限进行对比，展示了在最优解与新算法之间的更小误差；

**⚠️ 局限性**

局限性包括仍未能完全消除分解间隙、匹配和门控参数的调优需根据实例特性，且算法在最坏情况下仅达到3/2的下限，且对更大组分连通的推广尚未解决。

---

## 434. Enabling Privacy-Preserving Natural Language Access to Industrial Smart Meter Data

**arXiv ID:** 2611.09372 | [PDF](https://arxiv.org/pdf/2611.09372v1)

**作者:** Carlos Martín-Abellán `[一作]` (Valencian Research Institute for Artificial Intelligence), Mark Kröll `[通讯]` (Know Center Research GmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了OpenMeter系统，利用本地开源LLM和多代理架构，实现能源运维人员用自然语言查询工业智能电表数据库并得到可解释结果，满足隐私与可解释性需求。

**💡 创新点**

创新点包括：①在能源领域首次结合多代理LLM编排、文本到SQL与本地执行；②引入“隐私优先架构”与“LLM-策展者”模式，减少对云端的依赖；③使用结构化知识库过滤上下文，提升SQL准确率；④在同一系统中进行多场景消融实验，系统性评估自校正与领域知识的独立与交互效能。

**🔧 技术方法**

技术手段包括：Qwen 3.5系列开源LLM、DuckDB本地SQL引擎、Python sandbox执行、知识库检索与动态上下文裁剪、基于规则的自校正循环、解释代理生成人类可读解释。

**📊 数据集**

使用的数据集来自一家工业能源合作伙伴的真实智能电表部署，包含EMSify、LP3、LPR三张表，总计约750万行时间序列；另外构建了62条人工验证的自然语言-SQL对，用于评估。

**📈 对比分析**

比较方法：六种场景（单代理基线、单代理自校正、多代理基线、多代理自校正、多代理知识库、多代理完整）在本地9B和前沿397B模型两台硬件上进行5次实验，评估指标包括执行率、语义匹配率、结构匹配、解释质量及延迟。结果显示，在本地资源受限环境下完整多代理系统的语义正确率提升约6.1pp；在前沿模型下，单代理基线与多代理部分配置相近，单独启用自校正或知识库均能带来显著提升。

**⚠️ 局限性**

局限性包括：仅评估单一工业客户的数据；使用零样本提示，未探究少样本或微调效果；仅针对Qwen 3.5系列，缺乏跨模型验证；自校正循环在高执行率场景下收益有限；评估指标可能低估SQL实际实用性，解释质量评价仍高度主观。

---

## 435. Mitigating Covariate Shift in Behavioral Cloning via Robust Stationary Distribution Correction

**arXiv ID:** 2611.09371 | [PDF](https://arxiv.org/pdf/2611.09371v1)

**作者:** Seokin Seo `[一作]` (KAIST), Kee-Eung Kim `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为DrilDICE的离线模仿学习方法，旨在解决由于数据分布偏离专家的平稳分布而导致的协变量偏移问题。

**💡 创新点**

创新点在于通过优化受贝尔曼流约束的最坏情况目标，DrilDICE能够有效地最小化相对于非偏移目标分布的期望误差的代理损失。

**🔧 技术方法**

使用了分布鲁棒优化技术，并结合了贝尔曼流约束来处理协变量偏移。

**📊 数据集**

实验中使用了多种数据集，包括Four Rooms环境和D4RL基准数据集，模拟了不同的协变量偏移场景。

**📈 对比分析**

与多种基线方法（如BC、DemoDICE、AW-BC、DR-BC和OptiDICE-BC）进行了比较，结果表明DrilDICE在各种协变量偏移场景下表现优越，尤其是在最差25%的表现上显著提高。

**⚠️ 局限性**

限制在于未考虑由于转移偏移或噪声演示引起的不确定性，未来的工作可以扩展到这些场景中。

---

## 436. SafeOrchestrate: Runtime Checked Intent-Driven Orchestration for Cyber-Physical Systems

**arXiv ID:** 2611.09364 | [PDF](https://arxiv.org/pdf/2611.09364v1)

**作者:** Muhammad Ashfaq `[一作]` (University of Jyvaskylä), Tommi Mikkonen `[通讯]` (University of Jyvaskylä)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

SafeOrchestrate提出了一种基于LLM的运行时检查式意图驱动CPS编排架构，能够把自然语言意图分解为原子目标，生成可运行时可检查的LTL式规范，并在MCP（Model Context Protocol）接口上进行约束验证和动作执行。

**💡 创新点**

创新点在于：①将LLM生成的意图规范与运行时设备能力（MCP工具/资源）进行角色感知映射；②利用SMT（Z3）对单动作结果进行跨规则一致性和安全约束检查；③提出MCP执行相似度（MCP‑ES）指标，衡量编排轨迹的完整性。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑4.1用于语义匹配与参数合成，GPT‑3.5 Turbo用于LTL生成），线性时序逻辑（LTL）规范，MCP/Thing Description（TD）映射，SMT求解器Z3，Python实现的原型与自动化评估脚本。

**📊 数据集**

使用了在先前工作中公开的八个模拟CPS编排任务（如房间照明、空调控制、机器人工作流、温室自动化、洗衣机事件、门铃工作流、警报定时、进/出门多设备控制）作为实验数据集。

**📈 对比分析**

对比方法是基线VanillaOrchestrate（直接LLM→MCP调用，无运行时检查）。在所有任务上SafeOrchestrate实现了100%必需动作准确率，MCP‑ES在除“进/出门”任务外全部为100%，最高仅92%；而基线在事件触发、顺序及多设备任务上准确率和MCP‑ES均大幅下降，最高仅17%/13%。

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证，未对真实工业/机器人场景进行测试；Phase‑2的跨规则和安全约束检查仅在示例中演示，未在规模化任务中充分验证；仅检查单动作的当前状态快照，无法处理多动作依赖与并发；缺乏组件层面的消融实验，难以量化各子模块贡献；评估仅关注功能正确性，未考虑延迟、吞吐量或资源开销。

---

## 437. ENGRAM: From Agent Experience to Weights through Verified Consolidation

**arXiv ID:** 2611.09472 | [PDF](https://arxiv.org/pdf/2611.09472v1)

**作者:** Ruiyi Yang `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Engram框架，将语言模型代理的训练录入与记忆释放分离，通过演示感知门和验证门实现经验的有选择性整合与释放。

**💡 创新点**

创新点在于①演示感知训练门允许在学生未完全掌握前利用教师演示进行学习；②验证门在记忆释放前对无检索版本与有检索版本进行对比，确保权重已充分学习，从而实现可控且高效的经验删除。

**🔧 技术方法**

采用低秩LoRA适配器训练、Engram Trace Graph记录经验、演示门与验证门逻辑、检索与无检索版本对比评估，以及保留/释放机制等技术。

**📊 数据集**

在交互式决策任务上使用Crafter、ALFWorld、WebShop；在问答任务上使用HotpotQA、WebQuestions、FinQA；实验模型为Qwen3‑8B和Llama‑3.1‑8B‑Instruct。

**📈 对比分析**

与ETG冻结权重、仅SFT、以及多种基线方法（如STaR、ReST、MAGE、Memento、MemRL等）进行对比，结果在所有6个基准上提升3.1–8.4个百分点；验证释放在HotpotQA上保持接近保留所有记录的性能，减少55%可检索记录，并优于即时释放和随机释放。

**⚠️ 局限性**

限制包括：验证仅关注行为表现，未能保证隐私或偏差去除；记忆释放后记录仍存档，模型权重仍可能保留敏感信息；验证门需要额外探针样本和对比开销，且对更高维或更复杂任务的泛化尚未充分评估。

---

## 438. Mechanical Interaction Sensing for Robotic Kiwifruit Handling with an Optoelectronic Soft Suction Cup

**arXiv ID:** 2611.09468 | [PDF](https://arxiv.org/pdf/2611.09468v1)

**作者:** Romina Adhami `[一作]` (Istituto Italiano di Tecnologia), Barbara Mazzolai `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并评估了一种针对奇异果的双组件软吸盘，集成光电光学传感，实现在已知接触时的交互力估计。

**💡 创新点**

创新点包括将光电软吸盘适配奇异果表面，使用光学信号仅凭平均响应即可高精度估计力，且在未见果实和不同采集日下保持高鲁棒性；同时提供对柔软度分类的探索。

**🔧 技术方法**

使用光电光学传感、浅层MLP、单调回归、岭回归等模型；硬件包括三灯三光电二极管的PCB、Ecoflex和Dragon Skin 10等材料；配合UR5e机械臂、负载细胞。

**📊 数据集**

数据集由70个奇异果、1509次接触试验组成，覆盖4个采集日，力范围0.2–4.81 N；还包含压缩试验硬度标签。

**📈 对比分析**

采用分组留一果实交叉验证和留一采集日验证，浅层MLP模型达到R²≈0.968、MAE≈0.138 N；在留一采集日验证R²≈0.955、MAE≈0.171 N；相较于仅进度控制模型R²≈0.212显著提升。

**⚠️ 局限性**

局限包括只能在已知接触、非吸附模式下估计力；在卸载阶段精度下降；柔软度分类准确度有限；未实现实时接触检测与吸附抓取的闭环控制。

---

## 439. Consensus-Constrained Dual-Federated Optimization with Client-Side Dynamic Weighting for Distributed Android Device Threat Defense

**arXiv ID:** 2611.09470 | [PDF](https://arxiv.org/pdf/2611.09470v1)

**作者:** Mohammad Mashreghi `[一作]` (University of Tehran), Md. Jalil Piran `[通讯]` (Sejong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为DW‑FedADMM的联邦优化框架，用于在安卓设备上实现分布式恶意软件检测；

**💡 创新点**

创新点在于将性能驱动的客户端侧动态加权与基于一致约束的原始-对偶ADMM算法结合，既保持全局一致性，又利用客户端本地性能反馈实现更稳健的局部更新；

**🔧 技术方法**

核心技术包括联邦学习、原始-对偶（ADMM）优化、动态加权系数、增广拉格朗日法以及部分客户端参与的随机抽样；

**📊 数据集**

实验使用了三大安卓恶意软件基准数据集：Drebin、Malgenome 与 TUANDROMD；

**📈 对比分析**

与FedAvg、FedProx、FedADMM及其动态加权版本对比，DW‑FedADMM在18种配置中取得最高平均精度（95.42%），在9/18、7/9配置中表现最好，且在误报率和更新步幅方面优于其他方法；

**⚠️ 局限性**

局限性主要体现在仅验证了IID分片，缺乏非IID、标签/量化偏斜或时序分布下的评估；未来需进一步检验在更真实异质环境中的鲁棒性。

---

## 440. Overview of the ImageCLEFmedical 2026 GANs task: Exploring fingerprint detection, privacy leakage, and synthetic medical image generation

**arXiv ID:** 2611.09455 | [PDF](https://arxiv.org/pdf/2611.09455v1)

**作者:** Alexandra-Georgiana Andrei `[一作]` (National University of Science and Technology Politehnica Bucharest), Bogdan Ionescu `[通讯]` (National University of Science and Technology Politehnica Bucharest)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文通过三子任务评估GAN在医学影像生成中的隐私泄漏，研究模型对训练数据的记忆、潜在空间可逆性及生成图像的隐私与质量平衡。

**💡 创新点**

创新点在于：①首次引入潜在向量匹配子任务，检验生成模型对潜在空间的可逆性；②构建四种攻击（实例复制、成员推断、患者重识别、逆向重建）并定义综合隐私得分PPS；③在评估中同时考虑FID与PPS，实现隐私与质量的双重度量。

**🔧 技术方法**

使用技术包括：迁移学习与手工特征融合的分类器、k‑NN与深度特征匹配、回归+Hungarian的潜向量匹配、InfoNCE记忆池、DP‑SGD、UNet与StyleGAN2生成器、DCGAN、图像过滤Supervisor等。

**📊 数据集**

数据集来源为8,000例肺结核CT切片（8,000×256×256 PNG），10,000张GAN生成切片与6,400标注的真实图像（3,200用于训练、3,200不用于训练），2,140张未知标签测试图像，以及10,000张图像–潜向量对。

**📈 对比分析**

评价方法：子任务1用Cohen's κ、准确率、召回率等；子任务2用匹配准确率；子任务3用FID和四攻击的泄露分数综合成PPS。结果显示：子任务1 κ普遍为负（≈-0.08到-0.005），表明难以检测；子任务2存在0–1的极端差距，最高能达到1.0；子任务3在高Fidelity（FID≈0.26）与高隐私（PPS≈0.55）之间取舍，最优隐私方案PPS≈0.91但Fidelity差（FID≈3.24）。

**⚠️ 局限性**

局限性：①子任务1的检测难度高，模型对训练样本的记忆几乎不可检测；②子任务2仅在BigGAN‑Deep上表现优异，对其他生成器的可逆性尚未验证；③子任务3中患者重识别攻击（Attack 3）可能混淆模型覆盖与实际泄漏，需进一步细化；④整体任务依赖于有限的训练样本与特定CT数据，结果对其他模态或更大规模数据的推广性不明。

---

## 441. From LoRA to Language: Recovering Skill Documents from Adapters

**arXiv ID:** 2611.09454 | [PDF](https://arxiv.org/pdf/2611.09454v1)

**作者:** Saranya Dintakurthy Swadha `[一作]` (QpiAI), Aswanth Krishnan `[通讯]` (QpiAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了三种读取器，尤其是通过执行适配器的执行写手，来从LoRA适配器中恢复技能文档。

**💡 创新点**

首次证明仅通过运行适配器即可将低秩更新翻译回可读文档，并量化覆盖率和任务收益。

**🔧 技术方法**

采用低秩适配、软提示写手、词汇检索、执行写手以及激活镜头等技术。

**📊 数据集**

使用从技能仓库抽取的171,258个技能文档训练，测试37个未见技能，并评估SkillsBench、ALFWorld等任务。

**📈 对比分析**

与检索、权重编码、软提示写手等基线对比，执行写手在覆盖率上提高约2倍，恢复90-93%文本文本并恢复85-91%的任务收益。

**⚠️ 局限性**

实验仅覆盖Qwen3基模型，评估依赖LLM判读，样本规模有限，行为型适配器的恢复仍极低，且文档可能不完全一致。

---

## 442. Logit Laundering: Evading Data-Use Auditing in Large Language Models

**arXiv ID:** 2611.09446 | [PDF](https://arxiv.org/pdf/2611.09446v1)

**作者:** Ruihan Hu `[一作]` (Beijing University of Posts and Telecommunications), Meikang Qiu `[通讯]` (Augusta University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Logit Laundering，一种在推理时对 LLM 输出的 logit 进行重塑的模型级规避框架，用于逃避基于水印的数据使用审计。

**💡 创新点**

创新点在于：1）将水印噪声的统计偏差估计器与自适应置信度缩放器结合，以在不改动模型参数的前提下抑制可检测的水印信号；2）从攻击者视角系统性评估数据级与模型级规避策略，首次展示模型级规避在保持知识和通用性能的同时能显著提升审计失败率。

**🔧 技术方法**

核心技术包括：基于 transformer 的统计偏差估计器、logit 重塑（z_B = z_b - λ·z_c）、置信度自适应缩放（根据 top‑1 与 top‑2 logits 的差距动态调整 λ）、JS Divergence 约束的安全阈值设置；以及常规的持续预训练、LoRA 微调、MCQ 知识评估和多任务通用评测。

**📊 数据集**

实验数据集：1）公开语料（arXiv 500 篇、Pile‑CC 1000 篇、Hacker News 646 条）；2）50 本 2025 年新出版的版权书籍，共 4,959 段落；水印化后用于持续预训练。

**📈 对比分析**

对比方法：多种基于水印的审计方法（STAMP‑like、Radioactive、以及其多种变体）和数据级攻击（字符改造、同义词替换等）以及模型级 Rank‑Swap 基线；在五大基础模型（LLAMA‑3.2‑1B/3B、Qwen‑2.5‑1.5B、Gemma‑2‑2B、Stable‑LM‑2‑1.6B）上评测。结果表明，Logit Laundering 在所有审计方法下将 p‑value 提升至 0.05 以上，显著降低检测率；同时 MCQ 知识保持率、通用任务性能、流畅度等指标与未规避模型相当或略优，优于数据级攻击和 Rank‑Swap。

**⚠️ 局限性**

局限性包括：1）只针对现有的基于统计偏差和绿色 token 的水印审计方法，尚未验证对更复杂或多样化水印策略的鲁棒性；2）需要对模型的 logits 进行访问，若审计只获取文本，规避效果可能受限；3）在极端压缩或高安全预算下，模型生成质量仍可能受到影响；4）对大型模型的计算成本和部署复杂度尚未全面评估。

---

## 443. MINE: Entropy-coded Memory-efficient Inference for Large Language Models

**arXiv ID:** 2611.09424 | [PDF](https://arxiv.org/pdf/2611.09424v1)

**作者:** Emil Schätzle `[一作]` (ETH Zurich), Markus Püschel `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MINE框架，将无损熵编码(dtANS-LLM)与LLM权重量化结合，在GPU上实现单流推理加速，并将其集成为vLLM插件。

**💡 创新点**

创新点包括：①自适应的dtANS-LLM编码，支持多种payload和逃逸机制；②针对单流推理的共享内存/分块解码优化；③自动调优机制，构造压缩率‑延迟Pareto前沿；④在单流推理场景下实现真正的加速，突破传统压缩方法的限制。

**🔧 技术方法**

技术手段：dtANS熵编码、CUDA共享内存与寄存器优化、GPU分块与warp并行、自动调参框架、vLLM插件化、量化模型（如EntQuant、GPTQ等）。

**📊 数据集**

使用的模型与数据集：Llama 3.1 8B Instruct，共19种量化/压缩检查点；在RTX 4060 Ti与RTX 5090两款GPU上进行实验。

**📈 对比分析**

评估方法：与vLLM、Marlin、Humming、NeuZip、EntQuant等基线在单流（batch 1）吞吐与prefill性能进行对比。MINE在最高bpw检查点（EntQuant 2.1 bpw）上分别获得2.52×（RTX 4060 Ti）和1.48×（RTX 5090）的加速，吞吐率可与传统2‑bit量化相当。

**⚠️ 局限性**

局限性：仅针对单流推理做最优调优，prefill阶段性能略逊于原生vLLM；自动调优耗时较长（最高262 min）；对不同GPU的参数需预先调优，且无法在推理过程中动态切换编码方案。

---

## 444. Robust & Scalable Website Fingerprinting via Minimal DRAM Traffic

**arXiv ID:** 2611.09422 | [PDF](https://arxiv.org/pdf/2611.09422v1)

**作者:** Shubham Roy `[一作]`, Biswabandan Panda `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

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

## 445. Future-Informed Actor-Critic

**arXiv ID:** 2611.09417 | [PDF](https://arxiv.org/pdf/2611.09417v1)

**作者:** Samy Mokeddem `[一作]` (University of Liège), Damien Ernst `[通讯]` (University of Liège)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了未来信息化价值函数并基于此设计了未来信息化演员-评论家（FIAC）算法，利用未来状态、动作和奖励的特征改进策略梯度学习。

**💡 创新点**

创新点在于允许评论家同时条件化于内生和外生未来特征，推导出未来信息化策略梯度（FIPG），并证明其为γ‑收缩算子唯一不动点，从而在稀疏噪声奖励环境中实现更高样本效率。

**🔧 技术方法**

采用γ‑收缩理论定义未来信息化价值函数，利用TD更新学习F^π，并将其与传统AC框架结合得到FIAC。

**📊 数据集**

在OpenAI Gym环境（Umbrella Chain、FrozenLake标准与噪声、CartPole标准与噪声）上进行实验验证。

**📈 对比分析**

与V‑TD‑AC、Q‑TD‑AC及A2C等基线对比，实验显示FIAC在噪声稀疏奖励环境中取得更高最终回报，其他环境表现相当或略慢。

**⚠️ 局限性**

在稠密奖励环境下无明显收益，未能解释为何未来特征有利，以及缺乏在更广泛任务上的进一步验证。

---

## 446. Tracybot: Preserving Developer Intent by Tracing Agent-Generated Code to Prompts

**arXiv ID:** 2611.09395 | [PDF](https://arxiv.org/pdf/2611.09395v1)

**作者:** Răzvan Albu `[一作]` (University of Gothenburg), Ranim Khojah `[通讯]` (Chalmers University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了Tracybot，一款可在VS Code、Git与LLM编码代理之间自动追踪代码变更与对应提示（Tasklet）的开源工具。

**💡 创新点**

其创新点在于将完整的规划与构建提示与生成的代码行一一关联，并支持将关键Tasklet导出为决策记录，同时通过Git notes和快照实现无侵入式的历史追踪。

**🔧 技术方法**

技术实现基于Python后台、Git钩子、BLEU相似度阈值、Tasklet构建器、Agent插件适配器以及VS Code扩展。

**📊 数据集**

评估数据集为 7 名实践者与 2 名学生参与的焦点小组、24 名受访者的问卷调查，以及示例仓库中的实际对话与代码变更。

**📈 对比分析**

对比现有工具（Git AI、Cursor Blame、PromptTrace）显示，Tracybot在提示完整性与历史追踪方面优于其人，用户体验满意度（UI 70.8 % 以上、Tasklet 87.5 % 以上）和 83 % 以上的引用一致性证明其有效性。

**⚠️ 局限性**

局限性包括仅支持有限的代理与编辑器、对非代理生成的 AI 代码不做检测、树重写时可能产生悬挂引用、存储占用与潜在监控风险、评估仅基于演示与截图，缺乏真实使用验证。

---

## 447. Precedence Constraint Scheduling on a Fixed Number of Machines

**arXiv ID:** 2611.09392 | [PDF](https://arxiv.org/pdf/2611.09392v1)

**作者:** Klaus Jansen `[一作]` `[通讯]`, Klaus Jansen

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对单位处理时间、具有任意前置约束的固定机器数（m≥3）作业调度问题 P_m|prec,p_j=1|C_max 的确定性精确算法，能够在多项式时间内构造最优调度。该算法通过构造有限的集合族 ℱ 和 ℱ_cut，使用子集动态规划与“全槽”技术，配合二进制 trie 进行精确集合枚举和查找。

**💡 创新点**

创新点主要包括：
1) 证明了对任何固定机器数 m≥3，存在多项式时间的精确算法，填补了长期未解的固定机器数调度问题的空白；
2) 通过全槽算法只保留满足短公式描述的集合，显著减少了状态空间；
3) 设计了结构化证明，给出了每个节点可被短布尔公式描述的上界 (4m+12m+5) 原子项，和每个前缀可被 (m+6m+3) 原子项描述，保证了算法枚举到所有必要的候选集合；
4) 提供了完整的运行时间分析，给出了 RAM 操作数 O(L)+m^{O(m^2)}n^{E(m)}，并给出了多磁带实现的步骤数上界。

**🔧 技术方法**

主要技术手段包括：
- 子集动态规划与状态分解，利用全槽技术将状态拆分为前缀、单槽和后缀；
- 布尔公式族生成（GenerateFamily）和二进制 trie 精确去重，确保所有需要的集合都被枚举；
- 通过构造状态图和一次性前向传播来判定可行性；
- 结构化归纳证明构建层次化的完整性证书；
- 复杂度分析结合了集合族大小估计、状态转移计数和重构成本。

**📊 数据集**

本文为理论算法研究，未使用任何实验数据集；所有结果均在理论上给出并通过严谨证明。

**📈 对比分析**

与已有工作比较：
- 对 m=1 的情况是显而易见的，m=2 的情况可用 Coffman–Graham 算法解决；
- 对于 m≥3，之前仅有近似算法或指数/超多项式算法；本文首次给出多项式时间精确算法。性能方面，虽然理论上多项式，但对 m=3 的多项式度已达到 13394，仍远高于实际可用级别，但证明了可行性。

**⚠️ 局限性**

限制与挑战：
- 算法的多项式度极高，常数因子巨大，实际运行效率不具备竞争力；
- 复杂度随着机器数 m 指数级增长，m≥5 时已难以实际实现；
- 需要预先构造整个布尔公式族，内存消耗随 n^{O(m^2)} 增长；
- 算法仅在理论上证明可行，未提供高效实现或实验评估。

---

## 448. Universal Decoding for Gaussian MIMO-ISI: Spherical Ensembles and Symmetric Neighborhoods

**arXiv ID:** 2611.09383 | [PDF](https://arxiv.org/pdf/2611.09383v1)

**作者:** Nir Elkayam `[一作]` (Tel Aviv University), Meir Feder `[通讯]` (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在未知通道脉冲系数和噪声协方差的循环多输入多输出（MIMO）带间符号干扰（ISI）高斯信道中，构造一种通用解码器，证明其平均误差指数与知情最大似然（ML）解码器相同。

**💡 创新点**

创新点在于：① 利用频率块逼近将信道视为块平坦，② 发现所有匹配的对数似然在一个公共充分统计量上是仿射函数，③ 用该统计量的密度构造分数（score）并证明其满足“良好速率函数”条件，④ 通过高斯比较和能量映射实现不同编码集合间的指数比较，最终实现了在不估计通道的情况下达到信息理论极限的通用解码。

**🔧 技术方法**

主要技术包括：随机编码（球面均匀码字），频域离散傅里叶变换，频率块划分与块平坦逼近，仿射充分统计量与残差矩阵，统计量密度与体积估计，极大似然和密度分数的比较，Gaussian Likelihood-Ratio（GLR）比较，以及误差指数的合并与阈值分析。

**📊 数据集**

该工作完全基于理论推导与数学证明，不涉及任何实验数据集。

**📈 对比分析**

通过与知情ML解码器的误差指数比较，并在不同编码集合（全能量球面与块能量球面）间进行能量映射与距离控制，证明了所构造通用解码器的误差指数与ML相同，且在自然块能量配置下的指数至少不劣于全能量球面。性能在理论上达到信息理论极限，具体数值取决于信道参数和频率块数。

**⚠️ 局限性**

局限性包括：① 需要足够多且平衡的频率块，满足 B^2 log n = o(n)；② 需要通道频率响应在块内变化受限；③ 对全行秩输出的假设；④ 对记忆增长速度的限制（如 L_n^2 log n = o(n)）；⑤ 仅给出随机编码平均误差指数，未给出具体编码实现或有限块误差率；⑥ 复杂度仍相对较高，需对所有候选码字计算残差矩阵。

---

## 449. KineCouple: Substrate-conditioned residual readout for enzyme kinetic prediction

**arXiv ID:** 2611.09377 | [PDF](https://arxiv.org/pdf/2611.09377v1)

**作者:** Chun Kang `[一作]` (Beihang University), Lei Sha `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为KineCouple的模型，利用底物条件化的交叉注意力聚合酶残基特征，并结合全局酶表示来预测酶-底物对的kcat和Km。

**💡 创新点**

创新点在于提出“底物条件化残差读出”——通过交叉注意力让底物查询决定残基加权，再用统一池化差值进行中心化并门控融合，实现在不预先定义活性位点的前提下，对不同底物自动调整残基贡献。

**🔧 技术方法**

技术手段包括冻结ProtT5和MolT5预训练编码器，使用多头交叉注意力、中心化读出、门控融合，以及在对数尺度上训练的单独回归器。

**📊 数据集**

数据集为CataPro基准（kcat和Km），采用交叉验证的out‑of‑fold（OOF）评估，使用固定的序列与SMILES输入。

**📈 对比分析**

与重现的CataPro、DLKcat、UniKP以及CataPro+SaProt等方法在RMSE、Pearson和Spearman指标上进行比较；KineCouple在两项任务中RMSE略低（约0.1‑0.2%），Pearson和Spearman均略高，显示出轻微的性能提升。

**⚠️ 局限性**

局限性包括：改进可能归因于额外训练或参数增量而非核心机制；缺乏配对误差统计和多次实验的置信区间；未考虑pH、温度等实验条件；序列长度限制导致长序列信息缺失；未提供实际计算成本与内存占用的评估。

---

## 450. Anchored Iterative Decoding of Reed-Solomon Product Codes for High-Rate Magnetic Tape Storage

**arXiv ID:** 2611.09366 | [PDF](https://arxiv.org/pdf/2611.09366v1)

**作者:** Alexander Chvanov `[一作]` (Chebyshev Research Center), Nikolay Malkovsky `[通讯]` (Chebyshev Research Center)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种针对磁带存储的Reed–Solomon产品码的迭代解码算法，能够在原始误码率约5.3×10⁻³下实现10⁻²⁰的后译码误码率，并检测4个坏轨道。

**💡 创新点**

创新点在于结合anchor解码、二进制图像分析以及无限迭代停止机制，显著减少误差纠正中的误纠正和停滞模式。

**🔧 技术方法**

使用Reed–Solomon产品码、迭代有界距离解码、anchor约束、二进制比特权重检测、后处理纠错，以及CPU/FPGA混合实现。

**📊 数据集**

通过在BSC上进行蒙特卡洛模拟与重要性采样，使用随机误码实验数据验证算法性能。

**📈 对比分析**

与固定4次迭代的标准IBDD以及理想无误差解码器比较，算法在保持更低误码率的同时平均迭代次数降至约4次，CPU吞吐量可达200 MB/s。

**⚠️ 局限性**

局限性包括无限迭代停止策略在极低原始误码率下仍需实验验证，算法对硬件实现的anchor与二进制图像检测有较高要求，且误纠正率虽低但并未完全消除。

---

## 451. Challenges, Opportunities, and Misconceptions in Neurosymbolic AI

**arXiv ID:** 2611.09356 | [PDF](https://arxiv.org/pdf/2611.09356v1)

**作者:** Giuseppe Marra `[一作]` (KU Leuven), Silviu Maniu `[通讯]` (University Grenoble Alpes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化讨论神经符号AI在效率、安全、可解释性等方面的机遇与挑战，并提出对应方法与实践方向

**💡 创新点**

首次从行业需求出发，将理论与应用相结合，明确语义对齐与计算成本的核心问题，并归纳多种对策与常见误区

**🔧 技术方法**

概念瓶颈模型、偏好对齐、符号约束学习、近似与任何时刻推理、GPU加速、松弛与递归推理等多种神经符号技术

**📊 数据集**

无具体实验数据集，本文为理论与方法综述

**📈 对比分析**

未进行实验对比，缺乏定量性能评估，本文侧重概念阐述与方法建议

**⚠️ 局限性**

主要局限在语义对齐不充分、计算成本高、符号推理可扩展性有限，以及缺乏统一评估基准

---

## 452. Improved Approximation and Hardness Bounds for $β$-TSP

**arXiv ID:** 2611.09421 | [PDF](https://arxiv.org/pdf/2611.09421v1)

**作者:** Paul Liu `[一作]`, Da Wei Zheng `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对满足 β-三角不等式（1/2 ≤ β < 1）的旅行商问题（β-TSP）提出了改进的多项式时间近似算法，并给出了对应的更严格的近似难度下界。

**💡 创新点**

创新点在于：① 通过两种互补的长度变换（“shifted”Christofides 与基于 4/5-近似的权重重构）构造两条候选巡回路，② 对这两条路的长度估计进行线性组合，使得 β 的取值区间 1/2<β<1 内得到比之前最优 1.5 近似因子更好的分段近似比；③ 通过改进 (1,2)-TSP 的硬度证明，推导出 β-TSP 在所有 1/2<β<1 下更严格的 NP 难度下界。

**🔧 技术方法**

主要技术包括：β-三角不等式的结构分析、对边长的线性变换保持非负性和三角性、Christofides 算法的 3/2 近似保证、对权重重构后 4/5 近似算法的使用、以及通过线性组合消除统计量 S=∑_v min_u ℓ_{uv} 的影响；硬度部分采用 Håstad PCP、Chlebík–Chlebíková 放大器以及 Karpinski–Schmied 的 (1,2)-TSP 归约。

**📊 数据集**

该工作为理论性质研究，无实验数据集；所有结果均为严格的数学证明和组合归约。

**📈 对比分析**

与 Böckenhauer 等人提出的最优近似因子 min{2−β/3(1−β), 3β²/3β²−2β+1} 相比，本算法在 β∈(1/2,1) 时给出更紧的分段近似因子：β≤3/5 时为 4−3β/5(1−β)，β≥3/5 时为 8(1−β)²+3β/2(5(1−β)²+β)。此外，硬度下界从 1+1/534 提升至 1+5(2β−1)/2634。

**⚠️ 局限性**

局限性包括：① 近似比仍未达到 1 的最优阈值，特别是当 β 接近 1 时分段表达式的界限仍有提升空间；② 只给出理论下界，未对实际实例进行实验验证；③ 对 β∈[1/2,3/5] 与 β∈[3/5,1) 的分段处理可能导致实现时的分支复杂度；④ 归约中的放大器与硬度构造相对复杂，可能难以推广到更广泛的度量结构。

---

## 453. Left-Right-Consistent Iterative Stereo Matching via Cross-Injection

**arXiv ID:** 2611.09458 | [PDF](https://arxiv.org/pdf/2611.09458v1)

**作者:** Manuel Knott `[一作]` (Meta Reality Labs), Elias Vansteenkiste `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入交叉注入（Cross‑Injection）扩展，让单视角迭代立体匹配网络在左右视角之间交换一致性信息，能够同时预测左、右视差并保持高度一致。

**💡 创新点**

1）在GRU细化更新中注入显式的有符号左右不一致信号；2）使用零初始化编码器，可无缝接入已有模型；3）采用一对一停止梯度左右一致性损失，避免对称收敛导致模糊；4）仅需微调即可实现左右一致，保持原模型精度。

**🔧 技术方法**

RAFT‑style递归细化（GRU）、光度一致性损失、左右一致性约束（stop‑gradient）、跨视角warp与遮挡掩码、零初始化特征编码器。

**📊 数据集**

Middlebury、ETH3D、KITTI‑12、KITTI‑15、Booster等公开立体基准；训练时使用各基准原始模型的子集数据。

**📈 对比分析**

与原始单视角模型及同等微调但无CI的对照组对比；CI显著降低左右误差（例如Booster LR‑BP‑1从11.6降至3.9），同时单视差异保持不变甚至提升；在所有基准上均表现出更优的一致性与单视角精度。

**⚠️ 局限性**

仅适用于RAFT式迭代细化，不能直接用于单射成本体积或Transformer；需要两次推理，运行时开销增加；训练初期一致性损失不稳定，需要调节；对薄物体的“抹”效应偶尔被放大。

---

## 454. Additive $Ω(\log^* OPT)$ Gap for Bin Packing with Three Items per Bin

**arXiv ID:** 2611.09447 | [PDF](https://arxiv.org/pdf/2611.09447v1)

**作者:** Ilan Doron-Arad `[一作]` `[通讯]` (Massachusetts Institute Of Technology), Ilan Doron-Arad (Massachusetts Institute Of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了即使所有物品大小都接近 1/3，装箱问题的加性近似误差下限为 Ω(log^*)，并给出相应的配置 LP 积分间隙下限。

**💡 创新点**

提出一种三项式限制的有限状态编码与可行路径重构，将任意实例的加性难度映射到三项装箱实例，从而首次给出明确的 Ω(log^*) 下限并证实 3‑Partition 实例为最难的实例。

**🔧 技术方法**

利用配置 LP、状态路径编码、三项三元组构造、以及从 Håstad SAT‑Vertex‑Cover 难度的多阶段树构造来保持可分割性和加性间隙。

**📊 数据集**

该工作完全为理论分析，无实验数据或具体数据集。

**📈 对比分析**

通过证明 NP‑hard 性与配置 LP 下限，展示了加性误差 Ω(log^*) 与已知上限 O(log) 之间的差距，表明目前最佳近似上界与下界仍有较大差距。

**⚠️ 局限性**

下限仍为 Ω(log^*)，距离已知的 O(log) 上界较远，且未能给出匹配的上界或更紧的下限；实际实现仍需进一步研究。

---

## 455. Incentive Pathologies in a Live LLM Agent Arena: An Empirical Study of Concentration, Reward-Rule Implementation Discrepancy, and Network Dilution

**arXiv ID:** 2611.09445 | [PDF](https://arxiv.org/pdf/2611.09445v1)

**作者:** Lei Zhenbing `[一作]` (Independent Researcher), Lei Chang `[通讯]` (University of Chinese Academy of Social Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对AITRAP平台的9天观测审计，评估激励机制与实际实现的差异

**💡 创新点**

将实测的公共API数据与代码级机制重构结合，提出观测性预测框架并揭示公告-实现差距

**🔧 技术方法**

使用Gini系数、Top‑K份额、Mann‑Kendall趋势检验和代码分析重构

**📊 数据集**

AITRAP平台22个座席产生的约96,000节点和680,000事件的公开API数据

**📈 对比分析**

对四个预测使用bootstrap CI和Mann‑Kendall检验，结果显示Gini不确定、Top‑4显著下降、事件/节点显著下降，性能符合观察目标

**⚠️ 局限性**

观察窗口短、干预无法分离、数据归档不完整、结论仅为观察性而非因果

---

## 456. DexDICE: Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning

**arXiv ID:** 2611.09433 | [PDF](https://arxiv.org/pdf/2611.09433v1)

**作者:** Seokin Seo `[一作]` (Korea Institute of Science and Technology), Kee-Eung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用安全演示与性能演示两类数据，提出离线双专家模仿学习框架 DexDICE，在安全支持约束下学习既安全又高效的策略。

**💡 创新点**

创新点在于用安全支持约束替代传统分布正则化，解耦安全与性能，理论上可在高概率下实现无危险访问，并通过单参数 FNR 校准实现安全性可调。

**🔧 技术方法**

技术核心包括 DICE 离线分布估计、密度比判别器（PU 学习）估计安全支持与性能代理、凸优化求解支持约束下的分布匹配、加权行为克隆恢复策略。

**📊 数据集**

实验使用重构的 DSRL 基准（包含 Safety Gymnasium、Bullet Safety Gym、MetaDrive）以及自构建的安全 Gymnasium 任务，最终在真实 SUMMIT‑XL 机器人上使用人类遥控演示进行验证。

**📈 对比分析**

与 BC、FBC、JFBC、DemoDICE、DWBC、FISOR‑IL、CDT、TraC 等基线对比，DexDICE 在 38 个任务中安全成本均低于阈值，平均回报最高；在数据量/质量可变实验中保持数据效率与鲁棒性；在真实机器人上实现 100% 成功率且无安全违规。

**⚠️ 局限性**

局限性：要求安全与性能示例具有足够重叠；理论保障仅适用于理想支持约束策略，实际加权 BC 可能存在支持偏差；对安全阈值和 FNR 参数敏感，且需较多安全支持示例才能准确估计。

---

## 457. A Spiking Neural Model of the Mental Lexicon: Representations, Similarity Structure, and Associative Retrieval in Perception and Production

**arXiv ID:** 2611.09402 | [PDF](https://arxiv.org/pdf/2611.09402v1)

**作者:** Bernd J. Kröger `[一作]` `[通讯]` (RWTH Aachen University), Bernd J. Kröger (RWTH Aachen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了一个基于NEF+SPA的全脑可重现神经模型，模拟概念、词干（lemma）和音位层的词汇检索，并通过两条定向通道完成听觉到口语和口语到听觉的映射；

**💡 创新点**

首次将结构化图谱递归编码与两种关联记忆（WTA竞争与阈值传递）结合，明确区分同层相似性、跨层关联与神经激活；

**🔧 技术方法**

使用神经工程框架（NEF）、语义指针架构（SPA）、Nengo + NengoSPA实现神经网络；

**📊 数据集**

使用包含613个词条的WTT词汇图谱（含概念、词干、音位节点），八个词作为测试案例；

**📈 对比分析**

通过在完整词汇库中评估检索准确率：所有八个测试词在两条方向上均正确检索，目标得分0.946–0.980，最低差距0.411，表明模型在给定完整向量输入下能稳健检索；

**⚠️ 局限性**

局限性包括：未实现词汇学习、语法特征与句法映射；未处理声学输入、发音序列与发声生成；仅在完整向量输入下测试；缺乏对人类行为（RT、错误分布）的直接比较；模型对随机向量种子和网络种子敏感，需要进一步鲁棒性评估。

---

## 458. Cutpurse: Decomposing Explanation-Guided Model Extraction on Tabular Data

**arXiv ID:** 2611.09375 | [PDF](https://arxiv.org/pdf/2611.09375v1)

**作者:** Seyma Selcan Magara `[一作]` (University of Tübingen), Mete Akgün `[通讯]` (University of Tübingen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对解释性模型提取攻击进行通道分解，验证特征重要性排名和LIME离散化边界对攻击效果的影响，并提出自适应的无解释接口提取方法 Cutpurse。

**💡 创新点**

首次将解释输出拆分为归因和坐标两个通道，证明边界信息可被攻击者通过辅助数据重建，且自生成网格加多样化查询与边界搜索即可超越基线。

**🔧 技术方法**

使用解释器 LIME / SHAP、网格拟合、分布式查询生成、边界二分搜索、基线提取 Autolycus。

**📊 数据集**

六个公开表格数据集（crop, adult, breast, nursery, mushroom, pendigits），针对五类目标模型（LR, NB, KNN, DT, RF）。

**📈 对比分析**

通过对照 blind、defended、autolycus、self‑grid、self‑grid+diverse+boundary 进行实验，Cutpurse 在连续特征目标上比受限 Autolycus 提高 16.3pp、比标准 Autolycus 提高 7pp。

**⚠️ 局限性**

仅适用于连续特征的表格数据，缺乏对分类特征、神经网络或高维数据的评估；需在域内辅助数据；未考虑标签无关或不同离散化策略的解释器。

---

## 459. Symmetry Breaking over Multiway Channels

**arXiv ID:** 2611.09466 | [PDF](https://arxiv.org/pdf/2611.09466v1)

**作者:** Raymond Zhang `[一作]` (INRIA), Sheng Yang `[通讯]` (CentraleSupélec)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7`

**🎯 论文内容**

本文研究匿名用户在对称性多路信道上实现角色分配（称为“对称性破碎”）的理论极限，并给出了对所有可行策略的误差概率上界（逆变换法）、可实现方案的误差指数（随机命名、多路复用、交互式角色控制器），以及在噪声多臂老虎机通道上的具体应用。

**💡 创新点**

创新点在于：①提出了一种基于Rényi变化测度的通用逆变换对称性破碎问题的有限块长度下界；②将随机命名方案与多源多路复用相结合，给出Gallager随机编码指数与割集上界；③引入交互式角色控制器、信心控制器与广义音乐椅算法，给出其误差指数解析或谱半径下界；④在噪声多臂老虎机通道上，证明交互式方案可超越随机命名上限，并给出数值上界。

**🔧 技术方法**

使用的主要技术包括：Rényi相对熵逆变换、随机编码与Gallager指数、割集与相互信息界、Markov链与谱半径分析、凸优化求解参考核的持久性系数、以及对称性与条件独立性下的简化。

**📊 数据集**

本文为理论性工作，未使用实际数据集；所有结论均在抽象的离散对称多路信道和噪声多臂老虎机通道模型上验证。

**📈 对比分析**

通过与已知的随机命名方案、无碰撞乐器椅（musical chairs）以及常规的两路/多路信道极限进行对比，证明了：随机命名在错误指数上受限于割集上界；交互式方案在噪声无碰撞和噪声碰撞情形下可获得更高的误差指数；在两用户、两臂无噪声多臂老虎机时，误差概率可被解析至两倍常数因子以内。

**⚠️ 局限性**

局限性包括：①逆变换下界对所有协议均适用，但在特定协议（如交互式角色控制器）时需对参考核做进一步优化，尚无统一最优解；②在噪声碰撞情况下，数值实验表明交互式方案可超越随机命名，但缺乏理论上证明确切指数；③对有限状态控制器的误差指数下界仅适用于有“证书”输出的通道，对全概率正通道下的性能仍未完全阐明。

---

## 460. Not All Entropy Is Equal: Reward-Conditioned Entropy Allocation in Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2611.09448 | [PDF](https://arxiv.org/pdf/2611.09448v1)

**作者:** Song Yu `[一作]`, Zhenming Wu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较不同强化学习目标在大语言模型上生成数学解答的熵、模式保留、协方差等指标，并在多项数学竞赛数据集上评估其准确率。

**💡 创新点**

首次提出利用固定探针问答集、奖励条件熵、语义模式保留和短间隔协方差拟合等指标，对RL目标进行系统性端到端比较，并揭示不同目标的熵排序和模式收缩特性。

**🔧 技术方法**

使用强化学习（TRL）、Transformer、vLLM、LoRA、8-bit AdamW等技术，同时采用自定义的 Qwen3.5/3.7 验证器与语义代码库实现模型输出质量与模式识别。

**📊 数据集**

使用 Skywork-OR1-RL-Data 8000 示例、AMC23、AIME24/25、MATH500、Minerva Math 等数学竞赛数据集进行训练与评估。

**📈 对比分析**

通过宏平均准确率、pass@k、奖励条件熵比率、模式保留率和协方差拟合等指标比较，结果显示大模型在奖励条件熵上存在不同排序，模式保留普遍低于基准，协方差拟合高相关，但整体排名不稳定。

**⚠️ 局限性**

实验仅使用单一随机种子，未进行系统的超参数搜索；未估计模式内实现熵、模式分层协方差、OOD 鲁棒性；因未检验因果关系，不能确认熵或模式保留对准确率的直接影响。

---

## 461. Agentic AI for Healthcare Claim Denial Analysis: A Policy-Grounded Framework for Privacy Preservation

**arXiv ID:** 2611.09423 | [PDF](https://arxiv.org/pdf/2611.09423v1)

**作者:** Sofia Campos `[一作]` (Randy Labs), Pedro Santos `[通讯]` (Randy Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并评估了一种基于多代理架构的医疗理赔拒付分析系统，集成预测模型、图神经网络、因果反事实分析以及检索增强生成（RAG）与 HIPAA 合规审核。

**💡 创新点**

①引入 Agentic AI 的协作式多代理设计；②采用混合上下文 RAG 将政策文档嵌入推理；③配置专门的 HIPAA/GDPR 合规代理，实现自动去标识和合规审查；④在同一框架中实现可解释性、政策关联与隐私保护。

**🔧 技术方法**

GPT‑OSS‑120B 语言模型（函数调用），XGBoost 与 Platt 校准，TreeSHAP，图神经网络（MPNN），因果反事实 Recourse，Hybrid BM25+dense 向量检索，RAG，规则化去标识，LLM 合规审核，自动化质量门控。

**📊 数据集**

100,000 条合成医疗理赔记录（99,079 合格），包含 ICD‑10 诊断、程序、药物代码、支付状态、预授权、流程时间等，分 80/20 训练/测试；以及公开政策文档集（五类付款人政策和行业报告）。

**📈 对比分析**

在 100 条高风险理赔样本上比较无合规与有 HIPAA 合规两种配置，评估工具调用成功率、政策引用准确率（99%）、数字追踪率（98%）、去标识完整性（100%），模型 ROC‑AUC 0.90、精度0.80，反事实转化率 63%，可读性 23% 无术语，平均耗时 17 秒/案。

**⚠️ 局限性**

仅在合成数据上验证；可读性控制不足；合规审核仍需人工监督；部署需云安全与合约；政策文档覆盖有限；未评估真实业务收益与成本。

---

## 462. Uniqueness-Aware Feature Engineering for Representing Humans in Algorithmic Decision Making

**arXiv ID:** 2611.09403 | [PDF](https://arxiv.org/pdf/2611.09403v1)

**作者:** Theresa Elstner `[一作]` (Kassel University), Martin Potthast `[通讯]` (Kassel University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对算法决策系统的唯一性感知特征工程框架，强调个体在特征选择中的主导作用

**💡 创新点**

创新点在于将个体的参考表示与群体异质性相结合，构建多层评估指标（完整覆盖概率、稀有度、Jaccard覆盖度等），并在六步流程中实现可追溯的特征选取与透明报告

**🔧 技术方法**

技术手段包括：多元超几何分布建模、Jaccard相似度计算、基于出现率的特征筛选、阈值调优、结构化报告生成以及个体覆盖度评估

**📊 数据集**

使用了合成数据集（400名训练个体，570个潜在特征，四个子群体，包含共享、子群特有与随机特征），并在此基础上模拟生产环境中的新个体

**📈 对比分析**

通过在不同阈值下对训练集进行交叉验证，比较系统覆盖率（J_system）与预测准确率，发现最佳阈值为0.02时在保证覆盖率≥0.6的前提下准确率最高（约0.717），并对子群体覆盖差异进行了量化

**⚠️ 局限性**

局限性包括：参考表示的完整性与真实性依赖个体自我报告，可能存在表达能力差异导致偏差；合成数据未能完全反映真实数据分布；未对动态演变的特征空间进行长期评估

---

## 463. PhyCAD: Physics-Aware Text-to-CAD Generation via Hierarchical Reinforcement Learning

**arXiv ID:** 2611.09401 | [PDF](https://arxiv.org/pdf/2611.09401v1)

**作者:** Yuxin Cui `[一作]` (University of Science and Technology of China), Qiyu Kang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了物理感知的文本到 CAD 生成框架 PhyCAD，训练 LLM 生成可执行 CadQuery 程序，并通过自动 CAD‑CAE 验证（几何检查+有限元物理验证）提供可验证的工程反馈，显著提升几何与物理约束满足率。

**💡 创新点**

创新点在于：①将 CAD‑CAE 验证集成到文本到 CAD 生成流程；②构造包含几何、载荷与物理约束的逆设计数据集；③设计层次化均衡优势（Hierarchical Balanced Advantage）强化学习目标，既保证几何/物理约束的优先满足，又在可行设计空间内细化优化。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）+ CadQuery 代码生成；CAD‑CAE 自动验证管线（CAD 执行、几何检查、FEM 物理验证）；强化学习 GRPO + PeerRank；层次化优势权重 λ_L、λ_w 的设置。

**📊 数据集**

使用基于 Text‑to‑CadQuery 的 110k CAD 程序，处理后生成约 77k 物理感知逆设计任务，去除精确尺寸的 sanitization；其中 11k 用于 SFT，66k 用于 RL。

**📈 对比分析**

与 Claude Opus 5、GPT‑5.6 Sol、DeepSeek‑V4‑Flash‑Ascend 等大型模型的直接 prompt、以及 Qwen、Mini、SmolLM、StarCoder、Qwen2.5 等公开模型的 zero‑shot、SFT 与 GRPO 进行比较，评估 7 个指标（Valid CAD、BBox、CAE、Displacement、Stress、Physics、Joint）。SFT 后 Joint 成功率从 21.8–37.4% 提升，GRPO 后提升至 39.5–58.1%，在所有指标上均优于直接 prompt 的模型，验证反馈显著提升几何与物理约束满足率。

**⚠️ 局限性**

局限性包括：①主要聚焦物理约束，未充分覆盖完整工程意图；②数据集未系统涵盖材料变异；③缺乏局部硬尺寸约束；④生成仍需昂贵的 FEM 验证；⑤仅关注单零件设计。

---

## 464. Causal graph fusion across incomplete cohorts and causal steering of tabular foundation models

**arXiv ID:** 2611.09400 | [PDF](https://arxiv.org/pdf/2611.09400v1)

**作者:** Yanke Li `[一作]` (Spinal Cord Artificial Intelligence Lab ETH Zürich), Diego Paez-Granados `[通讯]` (Spinal Cord Artificial Intelligence Lab ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种基于因果图融合的完整工作流，先把多个不完全、部分重叠的临床队列按共享变量合并生成共享的PAG，再在每个队列中用时间层次知识和共享PAG约束进行FCI学习，随后利用Markov blanket指引缺失值重采样，迭代收敛；随后将得到的因果图注入预训练的表格基础模型 TabPFN，通过注意力偏置（Causal attention steering）在推理时实现因果知识的无训练偏倚。

**💡 创新点**

①首次将隐藏混淆、时间背景知识和缺失值处理三者统一到跨队列因果学习中；②提出 Causal attention steering，让已冻结的表格基础模型在推理时直接利用因果图做偏置，完全不需要再训练；③在合成数据和脊髓损伤临床数据上证明融合方法在结构恢复、分布保持和预测性能上均优于现有单队列或简单池化方案。

**🔧 技术方法**

因果图学习（FCI, PAG），多源融合（共享PAG+每队列约束），Markov blanket 重采样（随机森林），表格基础模型 TabPFN v2，注意力偏置 steering，Fisher‑z、GCM、CRT‑TabPFN 等条件独立性检验，Kolmogorov–Smirnov 距离评估分布保真度，R²、AUROC、AUPRC、F1 等预测指标。

**📊 数据集**

瑞士脊髓损伤中心的两个临床队列：HAPI（250人，入院到康复阶段的压力伤害数据）和 PPIS（266次手术/病程，174人，手术后压力伤害恢复阶段的数据），包含血液指标、功能评分、营养、手术过程等多种混合离散‑连续变量。

**📈 对比分析**

与非融合基线（单队列、无跨队列约束）以及简单池化方法相比，合成数据上 Fusion 在 F1、KS 距离上显著更好（p<0.001）；在临床数据上，融合得到的 Markov blanket 规模更小（21/34 变量）但预测性能与自动特征选择相当甚至更优（LoS R²≈0.71，AUROC≈0.88）；Steering 在分布移位下提升 AUROC 0.011–0.029，显著高于随机图或无 steering。

**⚠️ 局限性**

仅在单中心内验证，缺乏外部多中心验证；共享变量有限，因果结构受样本量影响；缺失机制可能为 MNAR，因果图学习仍受限；测试假设（Fisher‑z 线性 Gaussian）可能不完全符合混合变量；得到的 Markov blanket 仅适用于后向（历史）预测，前瞻性预测尚未评估；因果图为假设生成模型，未提供因果效应定量验证。

---

## 465. A Bridge from Paste to Place: Physics-Guided One-Step Image Composition via Object-Scene Interaction

**arXiv ID:** 2611.09387 | [PDF](https://arxiv.org/pdf/2611.09387v1)

**作者:** Hongzhen Li `[一作]` (Transsion), Leilei Cao `[通讯]` (Transsion)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理建模的单步图像合成框架（源对齐潜在桥 + 物理引导），实现粘贴物体的光照重建与场景阴影/反射等影响的同时保持细节。

**💡 创新点**

创新点包括：① 将前景光照与后景光照视为同一光照交互的两种空间后果，使用 I=A⊙S+R 进行分解；② 通过解析重组得到对象和环境的校正向量，直接在同一 Diffusion Transformer 推理中引导潜在空间更新；③ 结合源对齐潜在桥（一次性更新）与物理引导，实现一次推理完成合成，显著提升速度与一致性。

**🔧 技术方法**

技术核心：预训练的 Diffusion Transformer（FLUX.2 Klein 9B）+冻结 VAE；物理预测器（多尺度 DiT 特征+RGB‑mask 金字塔）输出 albedo、shading、residual 变化；物理控制器将解析校正嵌入后续 DiT 层；训练采用桥损失、物理场损失、组合损失及梯度一致性损失。

**📊 数据集**

数据集：P2P‑Data（86,273 条训练对，包含 77,218 条合成+物理注释和 9,055 条来自 OBER 的真实场景对）；P2P‑Bench（1,500 条评测集，分为 1,000 条合成、250 条真实、250 条手工粘贴）。

**📈 对比分析**

与 11 种现有方法（包括 MV‑CoLight、OmniPaint、DreamFuse、PS‑Diffusion、AnyDoor、SHINE 等）在 P2P‑Bench 进行对比。指标 PSNR、SSIM、LPIPS、前景 MSE、效果 MAE 均优于最强基线，PSNR 提升 6.48 dB，LPIPS 降低 40.4%；平均单步推理时间仅 0.743 s（比 FLUX.2 9B 降 70%）。

**⚠️ 局限性**

局限性：① 依赖大量合成/人工标注数据，真实场景泛化仍受限；② 物理模型假设为漫反射+残差，可能对高度反射或透明材质的复杂光照不足；③ 当前仅处理 2D 贴图，未覆盖 3D 资产的完整几何交互。

---

## 466. Against All Likelihood

**arXiv ID:** 2611.09385 | [PDF](https://arxiv.org/pdf/2611.09385v1)

**作者:** David Lopez-Paz `[一作]` `[通讯]` (FAIR at Meta), David Lopez-Paz (FAIR at Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了如何从大型语言模型的概率输出中识别真实信息，提出把“不变性（invariance）”作为判断真理的标准，并设计了一种在推理时通过多视角生成与提取不变答案的测试策略。

**💡 创新点**

创新点在于：①将哲学中的客观事实不变性概念引入机器学习，强调最大似然并非真理的必要与充分条件；②提出一种基于上下文多样化与答案交叉检验的“多环境不变性”方法；③结合逆向提示（reverse prompting）和链式推理（chain‑of‑thought）来挖掘不变机制。

**🔧 技术方法**

使用的技术包括：Invariant Risk Minimization (IRM) 的思想、最大似然训练、逆向提示与自一致性（self‑consistency）、链式推理提取、以及对比实验中的可解释性分析。

**📊 数据集**

数据集方面，本文未给出具体实验数据，主要以大型预训练语料（如 GPT‑3.5 及其类似模型的多元语料库）为背景；若进行实验，可使用公开的多任务语言模型数据集（如 OpenAI 的 web 文本、Wikipedia、CommonCrawl 等）。

**📈 对比分析**

比较方法：目前文中仅指出 IRM 在标注任务上的实验效果有限，且多环境不变性方法尚未在公开数据集上系统评估。作者认为该方法在推理阶段能更好地过滤幻觉，并有望提升跨环境泛化性能，然而缺乏定量性能评测。

**⚠️ 局限性**

局限性包括：①需依赖模型自行生成有意义的多视角环境，若生成质量低则不变性提取失败；②在高维输出空间下不变性提取计算成本高；③方法在大规模实验中尚未验证，可能会产生空洞或过度保守的答案；④依赖语言模型的推理链式解释性，若链式不可靠则失去约束力。

---

## 467. Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations

**arXiv ID:** 2611.09382 | [PDF](https://arxiv.org/pdf/2611.09382v1)

**作者:** Junru Hu `[一作]` (Shanghai Artificial Intelligence Laboratory), Ming Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Qwen3.5-0.8B 与 9B 上，使用基于推理的 Weight‑Space Programming（WSP）对七类基准任务进行局部权重编辑，提升模型在未见数据上的表现。

**💡 创新点**

提出无需梯度、基于代理推理的权重编辑框架，展示局部编辑能在多任务上产生显著提升并跨任务迁移。

**🔧 技术方法**

使用 GPT‑6 Astra 作为编辑代理，通过生成与教师强制回放进行反馈，执行稀疏 MLP、秩‑一更新、注意力头缩放等权重修改；评估使用 50 次前向请求。

**📊 数据集**

实验基准包括 MATH‑500、MMLU‑Pro、IFEval、GPQA Diamond、MedXpertQA、SciBench、SciCode。

**📈 对比分析**

对照原始检查点进行 held‑out 评估，0.8B 在 Math 提升 29.45pp，IFEval 36.72pp，GPQA 4.42pp，SciBench 17.11pp；9B 在 IFEval 1.85pp、SciBench 3.03pp 等，显示局部编辑显著提升；但在某些搜索中未能找到可验证的改进。

**⚠️ 局限性**

局限包括搜索预算受限导致无法发现有效编辑、不同搜索路径结果差异大、跨任务迁移不稳定、计算成本高、对更复杂推理任务验证不足。

---

## 468. SPSC: Zero-Shot Cross-Subject EEG-to-Image Reconstruction via Semantic Preservation and Subspace Calibration

**arXiv ID:** 2611.09380 | [PDF](https://arxiv.org/pdf/2611.09380v1)

**作者:** Yutang Lu `[一作]` (Beijing University of Posts and Telecommunications), Zhicheng Zhao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种零样本跨受试者 EEG 到图像重建框架 SPSC，利用预训练视觉-语言模型和扩散生成器实现脑信号到自然图像的无标签推理。

**💡 创新点**

核心创新点在于 (1) Object Mask Contrastive Regularization（OMCR）在训练时保留对象级邻域关系，避免单一图像负样本导致语义丢失；(2) Subspace Candidate Gate（SCG）在推理时通过 PCA 子空间、CORAL 对齐与岭回归校正，构造低秩语义候选并用无标签置信门选择最合适的条件，实现无标签语义校正。

**🔧 技术方法**

采用对比学习、对象掩码对比正则化、PCA 子空间、CORAL 统计对齐、岭回归、置信门选取以及扩散先验和 VAE 低层分支等技术。

**📊 数据集**

在公开的 EEG 数据集（包含 1,854 个视觉对象概念的 RSVP EEG 记录）上进行评估，并在十受试者留一法（LOSO）设置下对比重建与检索任务。

**📈 对比分析**

与 SOTA 方法（如 NeuroBridge、NICE、ATM 等）相比，SPSC 在 CLIP 余弦相似度从 0.552 提升至 0.713、Alex.5 从 0.612 提升至 0.745，十受试者 LOSO 200‑way 检索 Top‑1 提升 0.073，表明在无标签推理下显著提升跨受试者语义一致性和重建质量。

**⚠️ 局限性**

局限性包括：需要足够多的源受试者训练样本；对极端受试者差异或噪声敏感；校正过程依赖批量统计，批量大小不足时效果下降；主要提升的是语义一致性，仍难以恢复极细粒度视觉细节。

---

## 469. Typed-Centrality Grounding for Agentic GraphRAG: A Symbolic Approach for Agentic Reasoning

**arXiv ID:** 2611.09414 | [PDF](https://arxiv.org/pdf/2611.09414v1)

**作者:** Rosario Napoli `[一作]` (University of Messina), Maria Fazio `[通讯]` (University of Messina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 Typed‑Centrality Grounding 框架，将知识图谱中按实体类型计算中心性并作为符号层注入 LLM，实现 Agentic GraphRAG 的改进；

**💡 创新点**

创新点在于通过类型化中心性消除全局中心性信号稀释，构建符号层将确定性图结构信息与概率 LLM 结合，显著降低幻觉并提升解释性；

**🔧 技术方法**

使用了图数据库 Neo4j、GraphRAG 与 LLM（如 Claude），结合 Typed Centrality（PageRank、Betweenness 等）以及 MCP 协议进行 Agentic 推理；

**📊 数据集**

实验使用了来自英国曼彻斯特地区开放犯罪数据的 POLE（Person, Object, Location, Event）数据集；

**📈 对比分析**

通过与仅使用全局中心性和文本相似度的基线 GraphRAG 对比，实验表明符号层能更准确定位关键节点、提升推理一致性和可解释性，类型化中心性显著放大重要节点得分，避免误选；

**⚠️ 局限性**

局限在于仅依赖符号中心性，缺乏语义嵌入融合；对更大规模、多类型 KG 的可扩展性尚未验证；需要进一步研究神经符号层与图嵌入的结合。

---

## 470. DataHelix: A CPU-GPU Co-Scheduling Data Plane for Multimodal Model Training

**arXiv ID:** 2611.09411 | [PDF](https://arxiv.org/pdf/2611.09411v1)

**作者:** Yuanhe Li `[一作]` (Shanghai Jiao Tong University), Yiming Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个 CPU–GPU 共调度的数据平面 DataHelix，通过元数据优先的调度与异步预取，解决多模态训练中预处理与 GPU 计算不匹配导致的瓶颈。

**💡 创新点**

创新点在于将预处理决策推到轻量级描述符阶段并结合 GPU 计算窗口做 deadline‑aware 调度；引入两级负载平衡和 HCCL/NCCL 高速重分发；实现全局协调而不影响训练 API。

**🔧 技术方法**

使用了 Coordinator–DataServer 架构、元数据优先调度、deadline‑aware 放置、token/latent aware 批处理、成本门限的后处理重平衡、异步预取、共享内存/Peer 传输、HCCL/NCCL 集合以及 PyTorch/MindSpore 兼容接口。

**📊 数据集**

实验基于 Kinetics、OpenVid、Qwen‑VL 等视频与多模态数据集，涵盖小/中/大/混合解码配置以及 VLM/DiT 预训练与微调工作负载。

**📈 对比分析**

与原生 rank‑local DataLoader、相同 CPU 预算下的 Ray actor pool 以及多节点设置进行对比；在 8 卡 Ascend 910B 上平均提升 2.72×吞吐量，deadline‑miss 降低 67.8%，p95 输入等待下降 87%，并在多节点场景下提升 1.36×。

**⚠️ 局限性**

局限在于需要预先采集预处理成本与网络模型，极大批量或极端样本分布仍可能产生重分发开销；对单机单卡细粒度瓶颈或特殊硬件不一定适用；实现依赖 HCCL/NCCL，跨平台可移植性受限。

---

## 471. ROAM: Retrieval-Augmented Optimization for Feasible Multi-Day Itinerary Planning

**arXiv ID:** 2611.09407 | [PDF](https://arxiv.org/pdf/2611.09407v1)

**作者:** Ye Zihao `[一作]` (City University of Hong Kong), Lin Xiaoheng `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种检索增强的代理式框架，先通过语义检索获得候选景点，再通过最大团+K‑means日划分、质量偏置的模拟退火TSP和硬时间窗口修复实现可行的多日行程。

**💡 创新点**

创新点在于：①将LLM仅用于意图解析和候选选择，硬约束优化由显式管道完成；②引入天气、路网和开放时间的RAG层；③提供ReAct式交互式再规划，保证编辑后仍保持可行性。

**🔧 技术方法**

使用技术包括BGE中文语义检索、最大团与K‑means日划分、质量偏置的模拟退火TSP、贪婪时间窗口修复、OSRM路网查询、天气API、ReAct循环、JSON流式解析与修复。

**📊 数据集**

数据集：四个中国城市（福州、广州、深圳、上海）共510个POI（坐标、评分、类别、开放时间、描述），36条自然语言查询（晴雨、观光、探城四类场景）。

**📈 对比分析**

与Bare LLM和检索‑仅基线对比；三天计划平均覆盖POI从19.3/17.0提升至27.5（+42%/ +62%），路线离散度从2101m降至677m（3.1×），雨天自适应且整体质量提升；消融实验显示空间优化和硬约束修复是关键。

**⚠️ 局限性**

局限性：仅离线评估，未进行用户满意度研究；数据仅四城510 POI，室内覆盖不均；仅使用单一LLM模型，缺乏跨模型验证；仅处理单人单城行程，未考虑团体、实时交通、排队或门票；对候选集规模和路网查询延迟敏感。

---

## 472. Sensed but Not Told: Hidden Physiological Adaptation Shapes LLM Persuasion

**arXiv ID:** 2611.09394 | [PDF](https://arxiv.org/pdf/2611.09394v1)

**作者:** Xiaoyan Wei `[一作]` (Adelaide University), Hsiang-Ting Chen `[通讯]` (Adelaide University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在虚拟现实环境下，使用可穿戴传感器（心率、瞳孔直径）收集生理信号，构建一个GPT‑4 Turbo聊天机器人，在隐藏和公开传感器状态两种条件下，检验生理数据是否会影响用户对机器人说服力的感知以及即时态度转移。

**💡 创新点**

创新点在于：①首次验证“隐藏”生理传感在LLM对话中的说服效果；②发现隐藏传感能显著提升说服感知与态度转移；③揭示披露传感后评价下降，却仍难以让用户察觉生理适配；④提出可检视化适配的三条设计建议。

**🔧 技术方法**

技术手段包括：HP Reverb G2 Omnicept Edition VR头显（内置PPG心率和眼动摄像机）；收集每轮对话的心率标准差、左右瞳孔直径；GPT‑4 Turbo（Turbo）作为LLM；设计两因素2×2（感知开关 × 认知披露）对照；使用配对t检验、GEE模型、对话日志分析等统计方法。

**📊 数据集**

数据集为24名大学生（14女10男，18‑40岁）在实验室完成四轮对话（每轮4–8分钟）共96轮对话，包含：①预披露的感知开关交叉；②披露后再次交叉；每轮前后自评态度、说服感知、情绪量表，收集对话文字日志。

**📈 对比分析**

对比方法：在未披露前对感知开关的配对t检验，显示感知开启时说服感知提升1.42点（p=0.003，d≈0.79）；在所有对话中使用GEE估计态度转移，感知开启增加0.57点（p=0.029）；披露后感知差异缩小至0.33点；对话日志显示感知开启时问号出现率降低24个百分点，回合数平均减少0.77个，提示更宣告式风格。整体效果表明隐藏生理适配确实能提升说服体验。

**⚠️ 局限性**

局限性：①披露固定在后半段，无法完全分离披露效应与时间序列效应；②每个条件使用不同自选议题，可能引入议题偏差；③样本量仅24人，主要为大学生，外推性有限；④仅测试GPT‑4 Turbo，结果可能不适用于其他LLM。

---

## 473. k-Volume: a Unified Active Learning Strategy

**arXiv ID:** 2611.09388 | [PDF](https://arxiv.org/pdf/2611.09388v1)

**作者:** Eliot Tron `[一作]` (Université Paris 1), Alice Le Brigant `[通讯]` (Université Paris 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为k-Volume（kVol）的新型主动学习策略，通过随机选择距离其k最近邻的球体积最大的样本来进行标注，兼具密度、差异性与不确定性三大类方法的优点；

**💡 创新点**

创新点在于将球体积与k最近邻结合，理论证明其在样本空间中近似均匀采样，同时可在输入空间或输出空间使用，实现了对不平衡和高维数据的稳健处理；

**🔧 技术方法**

核心技术包括k-Nearest-Neighbor搜索、球体积估计、随机采样与分布收敛理论，以及在输出空间采用预测概率向量的扩展；

**📊 数据集**

在多种数据集上验证：合成高斯混合（2D/20D）、OpenML tabular数据（splice、satimage、elevators）、以及图像数据（MNIST、CIFAR-10/100）并使用预训练ResNet50嵌入；

**📈 对比分析**

与Uniform、Diversity、Entropy、CoreSet、TypiClust等七种主流方法比较，kVol（尤其是kVolOut）在所有实验中表现最稳健，往往在不同模型和数据维度下均能超越或接近最优，且计算成本显著低于CoreSet和TypiClust；

**⚠️ 局限性**

限制包括：k的选择需满足k≪n且随样本数递减；理论证明仅适用于单样本采样（b=1），多批次时需额外假设；在极高维度或极不平衡的场景下，kVolIn可能受限，需进一步优化。

---

## 474. Calibrated Uncertainty Quantification on TDC ADMET: What Can Be Resolved Across 22 Datasets?

**arXiv ID:** 2611.09373 | [PDF](https://arxiv.org/pdf/2611.09373v1)

**作者:** Laurits Fredsgaard `[一作]` (Technical University of Denmark), Mikkel N. Schmidt `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了11种校准不确定性量化管线（LightGBM、MLP、MC dropout、SWAG、Laplace、VBLL及其十元集成）在22个TDC ADMET任务上的预测性能和概率/区间质量，使用嵌套五折交叉验证、10个化学划分和5种模型种子；

**💡 创新点**

系统评估多种UQ方法在大规模ADMET基准中的一致性与差异，并通过化学划分、多重统计检验（Friedman、Holm、Hsu's MCB）检验评估设计的鲁棒性，探讨小规模实验设计对结论的影响；

**🔧 技术方法**

采用LightGBM与MLP作为基模型，结合Morgan指纹+RDKit2D描述符；通过温度标度、残差标准差估计、观察噪声方差调整等校准方法，计算Brier分数、区间分数等指标；

**📊 数据集**

使用22个TDC公开ADMET任务（13个二分类、9个回归），包含475-13,094个分子；

**📈 对比分析**

使用10个化学划分×5种种子×5外层折叠共250次评估，对每个指标进行Friedman检验和Holm校正的全对比；ensemble MC dropout在TDC、Brier和区间分数上平均排名最佳，LightGBM在单个任务上多次获胜但整体排名不稳定；小规模评估往往无法恢复完整网格的结论；

**⚠️ 局限性**

评估结果易受化学划分、种子与模型选择噪声影响，且仅基于Morgan+描述符特征；基准只涵盖公开任务，可能与工业真实场景差异；不区分信息量与不确定性来源，评估指标对校准方法敏感。

---

## 475. Travel Cost Calibration for Multi-Agent Target Assignment and Pathfinding

**arXiv ID:** 2611.09396 | [PDF](https://arxiv.org/pdf/2611.09396v1)

**作者:** Yu Kumagai `[一作]` (Hitotsubashi University), Keisuke Okumura `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于目标分配与路径规划迭代校准的TAPF求解算法（TACT），通过在每轮规划中引入噪声生成多样化的目标分配，并利用快速子最优MAPF求解器的行驶损失反馈不断更新成本矩阵，从而逼近实际的行驶成本。

**💡 创新点**

创新点在于将快速MAPF求解器的低精度反馈作为代理评估，采用噪声扰动产生多样化的全局目标分配，并通过全局成本矩阵累积学习而非局部重分配，显著提升大规模TAPF的解质量。

**🔧 技术方法**

核心技术包括线性求和分配（Hungarian/短路路径法）、快速子最优MAPF求解器（LaCAM、LaCAM3）、噪声扰动生成、移动平均更新学习率和并行规划循环。

**📊 数据集**

在标准MAPF基准图（四连通网格）上进行实验，使用两类目标分布（等距随机和热点密集）并覆盖从少量到10,000个智能体的规模。

**📈 对比分析**

与ITA‑ECBS、DBS‑Hungarian和SSP‑LaCAM3等代表性方法对比，TACT在小规模实例几乎达到最优，在中大型实例的流量时间比（flowtime/LB）上普遍优于对手，尤其在800个以上智能体时显著降低解质量；实验还验证了噪声尺度、学习率和并行度对性能的影响。

**⚠️ 局限性**

缺点包括：不提供解质量保证；依赖快速子最优MAPF求解器的性能和参数调优；在动态环境或实时更新目标时需要额外机制；在极端密集或非常大规模时仍可能面临计算开销与并行度平衡问题。

---

## 476. MSR-ABC: Complementary Feature Fusion for Clear Cell Renal Cell Carcinoma Classification

**arXiv ID:** 2611.09583 | [PDF](https://arxiv.org/pdf/2611.09583v1)

**作者:** Yuan Liang `[一作]`, Abraham Campbell `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于CT的全局-局部融合框架 MSR‑ABC，用以清晰细胞型肾细胞癌（ccRCC）的预后分类。

**💡 创新点**

创新点在于：① 通过跨表示注意力实现 MedVAE（重建）与 MedSigLIP（图像‑文本）局部特征的交互；② 通过辅助监督将放射组学属性嵌入局部表示；③ 用门控残差融合将局部与全局特征结合，同时结合视觉与软文本提示。

**🔧 技术方法**

采用 MedVAE、MedSigLIP、放射组学特征、跨表示注意力、辅助监督、门控融合、面积加权池化、视觉提示调优（VPT）和软文本提示技术。

**📊 数据集**

内部使用 KiTS23 数据集（396 例），外部使用 TCGA/TCIA 数据集（111 例），均使用公开的肾脏分割注释。

**📈 对比分析**

与全局仅拼接和单一来源基线（M、S、R 组合）比较，MSR‑ABC 在内部平均 AUROC 为 0.8064、外部为 0.8790，均优于基线；源/组件消融验证了三源互补性。

**⚠️ 局限性**

局限性包括：外部表现受不同人群和参考模型影响；仅在公开数据上验证，缺乏临床校准和决策影响评估；对新数据的泛化能力尚待进一步验证。

---

## 477. Performance Bottlenecks in Computing ZK Proofs on a GPU

**arXiv ID:** 2611.09569 | [PDF](https://arxiv.org/pdf/2611.09569v1)

**作者:** Victor Carolino `[一作]` (Lehigh University), Henry F. Korth `[通讯]` (Lehigh University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文实现并基准测试了基于 NVIDIA GPU 的并行 PlonK zk‑SNARK 证明生成器，重点关注多标量乘法（MSM）以及内存吞吐瓶颈。

**💡 创新点**

创新点在于明确将 GPU 内存和主机‑设备数据传输视为主导瓶颈，并提出多 GPU 与 RDMA 方案来缓解；同时利用 Thrust 实现 bucket 归约，提升 MSM 并行度。

**🔧 技术方法**

使用的技术包括 CUDA / Thrust、NVIDIA A10/V100 GPU、BN‑254 椭圆曲线算术、Pippenger’s bucket 算法、内存交错（memory interleaving）以及计划中的 RDMA 数据传输。

**📊 数据集**

实验使用的“数据集”为从 2^15 到 2^25 的合成约束集，代表真实应用场景如 zk‑rollups（>2^27）、私有交易（>2^26）、健康 AI 模型验证（>2^30）等。

**📈 对比分析**

方法上将 GPU 版本与 CPU 基线对比；在 2^24 以内的约束集，A10 约 20 s，V100 约 30 s，但超过 2^21 时因 VRAM 达标导致失败；CPU 在同规模下性能更优，主要因无内存瓶颈。

**⚠️ 局限性**

主要局限在于 GPU VRAM 限制、主机‑设备大规模数据转换耗时、对 Thrust 的依赖导致的内存拥塞、缺乏多流与 RDMA 细粒度调度，以及未能覆盖 IVC/递归折叠等可进一步加速的技术。

---

## 478. Solver-Free Decision-Focused Learning via Neural Optimization Layers for Edge Selection

**arXiv ID:** 2611.09560 | [PDF](https://arxiv.org/pdf/2611.09560v1)

**作者:** Xinda Zheng `[一作]` (Monash University), Edward Lam `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种无求解器的决策聚焦学习框架 DFLNOL，利用自监督神经优化层（NOL）在训练与部署阶段实现对 NP‑hard 边选择问题的预测和决策，而无需求解原始优化器或依赖最优解标签。

**💡 创新点**

创新点包括：① 通过自监督学习训练可微的 NOL 生成连续决策表示；② 两阶段训练（先预训练 NOL 再冻结训练预测器），实现梯度反馈而不需求解器；③ 采用结构化惩罚与约束正则化保证连续输出与离散解的一致性；④ 在部署时仅需一次前向传播即可产生可行解，显著提升速度；⑤ NOL 还能作为其他决策聚焦方法的近似求解器，降低其训练成本。

**🔧 技术方法**

使用技术：图注意力卷积（CAC）、决策依赖边评分、可微更新（Sinkhorn 归一化）、梯度下降/Adam、两阶段拉格朗日多重参数更新、以及专门的解码器（ATSP 的 Assignment‑Merge、EFOP 的贪心路径）。

**📊 数据集**

数据集：合成上下文生成数据，基于 PyEPO 的 TSP 成本生成器，生成不同节点数（50、100、200）、不同多项式度数（4、6）以及不同边权/奖励分布的 ATSP 与 EFOP 实例。

**📈 对比分析**

与 MSE、SPO+、CaVE、PGC、PFYL（ATSP）以及 MSE、Odece（EFOP）等方法进行对比。实验显示 DFLNOL 在均值、P95 以及最差 regret 上显著优于 MSE，且在大多数设置下竞争或优于其他决策聚焦基线；训练时间最短（≈2 min），推理速度快（<15 ms），且神经部署可无解算器得到可行解。

**⚠️ 局限性**

局限性：仅在合成数据上验证；对高非线性度（如多项式度数 8）预测效果仍退化；连续损失与最终离散决策质量不完全对齐；对更大规模实例或更复杂约束的可扩展性未知；需要手工调参以设计结构惩罚和解码器。

---

## 479. CoMoFlow: Controllable Protein Conformational Ensemble Generation via Explicit Collective-Mode Modeling

**arXiv ID:** 2611.09548 | [PDF](https://arxiv.org/pdf/2611.09548v1)

**作者:** Yusong Wang `[一作]` (Beijing University of Chinese Medicine), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可控的蛋白质主链构象集生成框架CoMoFlow，显式学习蛋白特定的集体运动模式并通过采样的系数指导SE(3)流匹配生成多样化构象；

**💡 创新点**

创新点在于将集体变形模式和其系数分布显式建模并直接与流模型耦合，实现对目标构象变化的可控生成；

**🔧 技术方法**

使用SE(3)流匹配、集体模式学习（基于等变几何场和正交化）、高斯混合系数分布、特征条件化与先验重心化等技术；

**📊 数据集**

利用ATLAS 1390个天然蛋白的MD轨迹（100ns三复制）和100个去设计蛋白的MD轨迹进行训练与评估；

**📈 对比分析**

与AlphaFlow、ESMFlow、P2DFlow、BBFlow、BioEmu等预训练或从零开始的流/差分模型对比，CoMoFlow在RMSF、PwRMSD、DCCM、PCA W_2、Jaccard以及生成速度等指标均显著优于基线；

**⚠️ 局限性**

局限性包括对极端大尺度构象变化的捕捉仍有待提高、需要预先获取MD样本来学习模式分布，且在极少数基准上与某些预训练模型的性能差距仍存在。

---

## 480. StockASkill: A Local-First AgentSkill for Intelligent Stock Selection

**arXiv ID:** 2611.09528 | [PDF](https://arxiv.org/pdf/2611.09528v1)

**作者:** Xiangjing An `[一作]` (Independent Researcher Team), Aiting Chen `[通讯]` (Independent Researcher Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于本地第一的 AgentSkill（StockASkill），实现快速股票筛选与目标约束的组合优化。

**💡 创新点**

创新点在于将 DuckDB 列式 OLAP 与 SIMD 向量化实现结合，SLSQP 优化配合光滑的回撤约束和整数板块数量化，支持零磁盘 JSON 流式接口与 LLM 交互。

**🔧 技术方法**

技术包括 DuckDB 列式数据库、SIMD 向量化查询、7+1 因子评分模型、SLSQP 组合优化、整数板块约束、AgentSkills 规范的零磁盘流式接口。

**📊 数据集**

使用包含 42.44M 条历史价格、186.4k 条季报数据的 26 年多资产（中国 A 股、港股、美股、ETF）本地数据集。

**📈 对比分析**

与传统 SQLite 行扫描和 Pandas 并行循环相比，屏蔽 74.9 倍速度提升（0.38 s vs 28.45 s），回测显示相对于基准指数的 CAGR 超 10%、最大回撤降至 12% 以内，夏普率提升至 1.48。

**⚠️ 局限性**

局限于仅关注股票组合、缺乏多资产风险平衡与非线性模型支持，且目前只支持单一 LLM 代理，需进一步扩展至多资产、多模型与多代理协同。

---

## 481. ExQuery: Explicit Structured Query Priors for Temporal 3D Object Detection

**arXiv ID:** 2611.09527 | [PDF](https://arxiv.org/pdf/2611.09527v1)

**作者:** Honglin Sun `[一作]` (Waseda University), Kenji Hashimoto `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种显式结构化查询框架，用共享语义上下文与物理状态专属切片对时间序列3D目标进行检测，并通过门控仿射融合实现查询的时间传播。

**💡 创新点**

创新点在于保持查询从先验编码到回归的属性对齐结构化表示，显式传播物理与语义先验，并与存储的实例特征进行互相门控融合。

**🔧 技术方法**

技术实现基于Transformer稀疏查询检测器（DETR3D风格），使用3D锚点先验、复数位置编码（方向与速度），车体运动变换以及门控仿射融合模块。

**📊 数据集**

在nuScenes数据集上进行实验，评估多视角相机驱动的3D检测任务。

**📈 对比分析**

与StreamPETR、Sparse4Dv2等前沿方法对比，在nuScenes验证集上取得0.459 mAP、0.561 NDS，较StreamPETR提升0.009 mAP、0.011 NDS，并在位置、尺度、朝向、速度等误差指标上都有所下降。

**⚠️ 局限性**

局限性包括因额外的查询编码与融合模块引入的延迟、共享语义切片可能导致信息混杂，以及对新出现物体的显式先验缺乏充分利用。

---

## 482. TimeOmni-Live: Continuous & Live Time Series Understanding with Long-Term Memory

**arXiv ID:** 2611.09502 | [PDF](https://arxiv.org/pdf/2611.09502v1)

**作者:** Tong Guan `[一作]` (Griffith University), Ming Jin `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了面向持续流的时间序列理解范式CL‑TSU，构建了首个面向此范式的数据集与模型，支持持续状态保持与动态指令响应。

**💡 创新点**

创新点包括：①从离线一次性分析转向持续实时推理与状态更新；②设计了可持续的混合时间序列编码、流持久混合记忆与弹性推理同步机制；③提出了专门评估持续状态保持与跨窗口推理的18个任务与6项能力的评测框架。

**🔧 技术方法**

使用的技术包括：大模型（Qwen3.5‑4B）+冻结时间序列编码器+文本统计嵌入、KV缓存分叉的流持久混合记忆、弹性流推理同步、LoRA微调、Silence Down‑Weighting与记忆前缀扰动等。

**📊 数据集**

使用的数据集为自研的CL‑TSU数据套件，包含5,249条流、2.2B步长、15K事件注释，涵盖18个任务与6类能力。

**📈 对比分析**

与多种通用LLM、时间序列LM及代码执行基线在ID与Real测试集进行对比，CL‑TSU在ID上达90.0分、Real测试达56.1分，超过最强对手Opus 5 10.7分；在边缘设备上实现RTF 0.358，p95延迟降低206倍。

**⚠️ 局限性**

局限包括：对新兴多模态时间序列场景适配尚有限；记忆机制依赖KV缓存分叉，硬件受限时可能出现延迟累积；目前仅支持单一时间序列模态，未覆盖混合传感器或多通道场景。

---

## 483. e-means: outperforming k-means (in one-tenth the epochs)

**arXiv ID:** 2611.09501 | [PDF](https://arxiv.org/pdf/2611.09501v1)

**作者:** Swann Bessa `[一作]` (FAIR at Meta), Hervé Jégou `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于指数移动平均（EMA）的 mini‑batch k‑means 优化算法，并给出了自动化的超参数缩放律；

**💡 创新点**

创新点在于将 EMA 与余弦衰减调度、死聚类复活和终止阶段相结合，使得仅需 2–3 个 epoch 就能实现比传统 Lloyd 更低的 MSE，且无需手工调参；

**🔧 技术方法**

使用技术包括 EMA 更新、指数衰减学习率调度、批量大小与学习率的缩放律、近似最近邻（IVF）索引以及 GPU 上 BF16/FP8/INT8 的高效分块赋值；

**📊 数据集**

实验覆盖从 1M 到 10B 的多模态嵌入数据集，包括 DEEP、DINO10B、SPACEV、SIFT、SimSearchNet++ 等；

**📈 对比分析**

与 Faiss、Flash‑KMeans、SuperKMeans 等最先进实现对比，在 CPU 与 GPU 上均达到或超过其 MSE，并在大规模场景下实现 10×–45× 的速度提升；

**⚠️ 局限性**

局限性在于缩放律主要针对高维大规模数据，对低维或极小样本集的性能不佳，并且仍需依赖经验系数来决定批大小与学习率。

---

## 484. LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding

**arXiv ID:** 2611.09476 | [PDF](https://arxiv.org/pdf/2611.09476v1)

**作者:** Mingkai Li `[一作]` (Zhejiang University), Zhiguo Shi `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了LEMMO框架，能够在单一模型中对长达八百万样本的原始IQ信号进行指令驱动的识别与详细描述。

**💡 创新点**

创新点在于结合线性注意力的EM编码器与抗混叠时间重采样实现高效长序列处理，并通过指令条件可学习查询和结构化查询实现对不同任务的适配与辅助事实监督。

**🔧 技术方法**

使用了线性注意力EM编码器、时间重采样、指令条件交叉注意力、可学习查询、Qwen3.5-4B的LoRA微调以及多任务辅助损失。

**📊 数据集**

构建并发布了1.38 TB统一语料，其中包含LEMMO-ShortQA（四类识别任务）和LEMMO-LongDesc（多信号场景与细节描述），同时使用EM‑134K和EM‑Bench进行预训练与评估。

**📈 对比分析**

在EM‑Bench、ShortQA与LongDesc任务上与MERLIN‑LoRA及四大通用LLM对比，LEMMO在识别任务平均准确率从55.7%提升至70.1%，在长录描述多项指标提升4–9点，且保留了Qwen3.5‑4B的MMLU性能。

**⚠️ 局限性**

局限在于对极长或极高采样率的实时推理仍需优化，依赖离线训练，且尚未探索多模态（如声谱）联合学习或更复杂的交互式指令场景。

---

## 485. Video Stitching from Multiple Moving Cameras

**arXiv ID:** 2611.09570 | [PDF](https://arxiv.org/pdf/2611.09570v1)

**作者:** Shira Ifergane `[一作]` (Ben Gurion University of Negev), Oren Freifeld `[通讯]` (Ben Gurion University of Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

论文内容缺失，无法判断研究内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定数据集

**📈 对比分析**

无法确定对比方法和性能

**⚠️ 局限性**

未提供，无法评估限制

---

## 486. Nebula: Distributed Position-Independent Caching for LLM Serving

**arXiv ID:** 2611.09474 | [PDF](https://arxiv.org/pdf/2611.09474v1)

**作者:** Yifei Liu `[一作]` (Xiaohongshu), Weihang Chen `[通讯]` (Xiaohongshu)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套分布式 LLM 服务器框架 Nebula，使位置无关缓存（PIC）成为集群级的 first‑class 对象，实现远程缓存复用与并行预填，从而显著降低 TTFT。

**💡 创新点**

创新点包括：① 将 PIC 视为可跨节点调度的缓存对象；② 结合乐观计数与事件驱动的视图同步，既保证缓存状态实时又避免重复计算；③ 利用全局“幽灵命中”反馈自适应调节 PC 与 PIC 的容量分配；④ 统一预填完成时间模型，采用贪心搜索实现最佳计算/传输平衡。

**🔧 技术方法**

技术方案涵盖：分布式缓存目录、KV 传输（NIXL）、多层存储与 LRU/分区管理、优化的调度算法、GPU 计算与 RDMA 通信、SGLang+Dynamos 的集成。

**📊 数据集**

实验使用四个公开的 RAG 与 Agent 工作负载（ChatRAG‑QuAC、RAGPulse、τ‑bench‑Retail、ChatDev）以及四个 LLM 模型（DeepSeek‑V2‑Lite、Qwen3.5‑35B‑A3B、Mistral‑7B、Qwen3‑30B‑A3B）进行评测。

**📈 对比分析**

与传统仅支持 PC 或仅在单节点实现 PIC 的基线相比，Nebula 在所有四组模型/工作负载下平均降低 61.4% 的中位 TTFT，恢复 76.1% 的 PC 未命中 token；在 P99 级别 TTFT 下降 90% 以上，并显著提升缓存命中率。

**⚠️ 局限性**

局限性包括：① 对多节点网络延迟和 GPU 资源调度的依赖，极端高负载时仍可能出现计算/传输不均衡；② 需要对现有推理引擎做较大改造，部署复杂；③ 评价主要基于特定 RAG/Agent 场景，跨领域通用性待进一步验证。

---

## 487. Scalable Full-Horizon SFC Deployment over Predictable SAGIN Topologies via Configurable Column Generation Framework

**arXiv ID:** 2611.09585 | [PDF](https://arxiv.org/pdf/2611.09585v1)

**作者:** Hang Li `[一作]` (Yanbian University), Yanling Jin `[通讯]` (Fujitsu Limited)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对可预测的空间-空中-地面集成网络（SAGIN）中的全时域服务功能链（SFC）部署，提出可配置列生成框架以实现可扩展、高质量的资源分配与入选控制。

**💡 创新点**

创新点在于将完整请求部署视为列进行 Dantzig‑Wolfe 分解，结合可配置的定价调度、并行求解与安全的整数恢复，实现在可预知拓扑下的 admission‑first、成本‑aware 优化，并在多尺度实验中实现显著提升。

**🔧 技术方法**

采用时域扩展图模型、列生成与求解器优先定价、可配置的启发式 PFE、并行求解与整数池选取等技术。

**📊 数据集**

使用合成的 SAGIN 网络拓扑（四种规模 S20–S50）与生成的流量请求（80–200 条）以及公开的实验代码。

**📈 对比分析**

与纯列生成、启发式初始化、纯求解器等对比，在中高负载下平均提高约 5–8% 的接入率，运行时间在 1–10 倍之间，且在低负载时可获得最优接入率。

**⚠️ 局限性**

局限在于定价完全基于单请求的 LP 价格，未利用机器学习或更强的分支定价；对非常大规模实例的收敛仍受限；并且对动态未知拓扑的适应性不足。

---

## 488. Extended Model-Based Learned Inertial Odometry

**arXiv ID:** 2611.09584 | [PDF](https://arxiv.org/pdf/2611.09584v1)

**作者:** Sasanka Kuruppu Arachchige `[一作]` (Tampere University), Joni-Kristian Kämäräinen `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于机体坐标系和扭矩动力学的学习式惯性里程计Ex-IMO，提升了无人机在未见轨迹上的姿态估计精度。

**💡 创新点**

创新点在于将相对位移预测切换到机体坐标系，并显式加入机体扭矩动力学作为先验，提高了模型的可泛化性和鲁棒性。

**🔧 技术方法**

采用了时序卷积网络 (TCN) 对短时相对位移进行学习，并与扩展卡尔曼滤波器 (EKF) 结合实现状态估计。

**📊 数据集**

使用了Blackbird、DIDO等航空数据集以及AI-IO、NanoBench、TII-RATM、NeuroBEM四个公开四旋翼数据集进行评测。

**📈 对比分析**

通过与原IMO方法和公开实现对比，Ex-IMO在未见轨迹上平均提升了28%–37%的绝对轨迹误差，且在多数据集上显著降低速度误差。

**⚠️ 局限性**

局限性包括依赖学习先验、对旋转动力学参数敏感、需要舵机/转子信号、在训练数据量不足或轨迹形状相差较大时泛化仍有限。

---

## 489. Alignment by Confident Debate: Preference Learning from Multi-Agent Debate

**arXiv ID:** 2611.09568 | [PDF](https://arxiv.org/pdf/2611.09568v1)

**作者:** Nikhil Verma `[一作]` (Upwork Inc), Zhao Chen `[通讯]` (Upwork Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需金标答案的自监督方法——Alignment by Confident Debate（ACD），通过多智能体辩论生成多条解答并聚类，然后利用答案跨度的置信度来挑选最可信的答案簇，从而得到偏好标签；

**💡 创新点**

创新点在于：①将答案级别的等价性与答案跨度置信度分离，聚类后以聚合置信度选优；②不依赖外部评估器或金标，仅使用自我生成的置信度信息构造偏好数据；③将偏好数据应用于标准后训练目标（SFT、DPO、KTO、GRPO），显著提升推理性能；

**🔧 技术方法**

主要技术包括：采样多样化委员会（不同温度/种子），多轮辩论，答案解析器，答案跨度置信度计算（几何平均 token log‑prob），答案等价性聚类，聚合置信度权重，偏好数据生成，后训练优化；

**📊 数据集**

实验数据集涵盖可验证任务：数学推理（GSM8K、SVAMP、MathQA）和可执行代码生成（MBPP）；

**📈 对比分析**

与传统投票、仅采样、单一评估器等方法对比，ACD在12个模型-数据集组合中平均提升 5–21.5 个百分点（最大 21.5pp），且在更大模型和异构委员会上也保持正向增益；

**⚠️ 局限性**

局限性包括：需要可解析答案与可定义等价关系；置信度跨模型可比性不足（异构委员会效果受限）；对开放式生成任务尚未充分验证；

---

## 490. Towards Sustainable Federated Learning via Carbon-Aware Client Selection

**arXiv ID:** 2611.09552 | [PDF](https://arxiv.org/pdf/2611.09552v1)

**作者:** Wentao Gao `[一作]` (University of Sydney), Albert Y. Zomaya `[通讯]` (University of Sydney)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种公平的碳感知客户端选择策略（CA‑FCS），旨在降低联邦学习中客户端侧的碳排放并保持模型性能。

**💡 创新点**

创新点在于首次将细粒度碳强度与设备能耗结合，利用仅包含电荷和电压的排放估算，同时引入最小/最大参与次数及非连续选择约束，实现公平且隐私友好的碳优化。

**🔧 技术方法**

技术方法包括碳强度预测、基于电荷×电压的能耗模型、碳排放估算、排序式客户端筛选以及公平约束的实现。

**📊 数据集**

实验使用了MNIST、FMNIST、CIFAR‑10/100、CINIC‑10等公开数据集，涵盖IID和多层次非IID分布。

**📈 对比分析**

与随机选择、CEFL、EAFL等基线相比，CA‑FCS在100/200轮训练下准确率仅下降≤3%，同时客户端侧碳排放降低约38–46%。

**⚠️ 局限性**

局限性包括固定的公平阈值、仅关注计算碳排放、未考虑通信碳、实验规模有限且缺乏自适应公平机制与周期性选择策略。

---

## 491. Active Learning of Robust Models

**arXiv ID:** 2611.09520 | [PDF](https://arxiv.org/pdf/2611.09520v1)

**作者:** Mizu Nishikawa-Toomey `[一作]` (Mila Quebec AI Institute), Dhanya Sridhar `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于源级不确定性（source‑level uncertainty）的主动学习框架，利用数据采集元数据（source）作为子群体（subpopulation）的噪声近似，在子群体分布偏移（subpopulation shift）下训练更鲁棒的分类器。

**💡 创新点**

创新点：
1) 将主动学习的采样目标从单个样本转为整个源（source）的平均不确定性，减少伪相关导致的高置信错误样本被忽略的问题；
2) 通过源级预算分配（Top‑K或SoftMax）实现对罕见子群体的主动增采样；
3) 结合噪声元数据的可用性，弥补缺乏真实子群体标签的缺口；
4) 在理论上说明传统基于不确定性的主动学习在存在伪相关时失效，并给出源级方法的优势。

**🔧 技术方法**

技术细节：
- 使用 Monte Carlo Dropout 估计模型不确定性并计算预测熵；
- 源级评分为源内所有样本熵的平均；
- 采用 softmax/soft‑rank/power 等函数进行源级预算分配，并在源内使用 SoftMax、SoftRank、Power 等采样策略；
- 在所有实验中采用预训练 DINO‑ViT（ImageNet‑1k）作为特征提取器；
- 与 g‑DRO 等基线对比。

**📊 数据集**

数据集：
- Semi‑synthetic：ColoredMNIST、Waterbirds、CelebA（通过人工生成环境 e 与源 s 的映射）；
- 真实数据集：Camelyon17（5 个医院来源作为源标签）。

**📈 对比分析**

对比方法：随机采样、均匀源采样、entropy、mutual‑information、BADGE、BAIT、Core‑Set、g‑DRO 等。实验结果表明：
- Top1‑SoftRank 在 ColoredMNIST、Waterbirds、CelebA 上以更少的标签即可达到或超过竞争方法的 Worst‑Group Accuracy；
- 在 Camelyon17 上 Top1‑SoftRank 的 Worst‑Source Accuracy 最高，且与均匀采样相比有统计学显著优势；
- 当伪相关强、子群体不平衡明显或源间噪声不大时，源级方法优势最为突出。

**⚠️ 局限性**

局限性：
- 当源标签与真实子群体关系高度噪声、或每个源内部混合多个子群体时，源级平均不确定性难以反映真实子群体差异，导致优势减弱；
- 对于伪相关较弱或 invariant 特征易学的任务，方法与传统采样差异不大；
- 需要足够多且质量可观的源元数据；
- 目前仅处理离散源标签，无法直接应用于连续或时间序列元数据。

---

## 492. Mosaic:montage-agnostic self-distilled graph learning for subject-independent ieeg seizure detection

**arXiv ID:** 2611.09496 | [PDF](https://arxiv.org/pdf/2611.09496v1)

**作者:** Arya Jalali `[一作]` (EPFL), Mahsa Shoaran `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 MOSAIC，一种轻量级图神经网络，利用共享卷积通道编码器与通道注意力自适应生成窗口特定相邻矩阵，从而实现无蒙版的跨个体 iEEG 癫痫发作检测。

**💡 创新点**

创新点在于：① 用单层共享通道注意力直接从编码特征推断邻接矩阵，取代固定蒙版；② 引入指数移动平均自蒸馏正则化，使窗口级邻接图在同一病人间保持稳定；③ 通过共享卷积编码器降低对蒙版的依赖，提升对未知植入配置的泛化能力。

**🔧 技术方法**

技术方案包括：共享时序 1D‑CNN 编码器、通道注意力自适应相邻矩阵生成、EMA 自蒸馏正则化、图卷积层（残差+层归一化）以及最终的窗口级二分类层。

**📊 数据集**

使用四个公开 iEEG 数据集：SWEC‑ETHZ（短期、长期）、Mayo 和 FNUSA（单通道）。

**📈 对比分析**

与 EEGNet、DCRNN、SEEGNet、DMNet、轻量 CNN 以及 Multitask Transformer 等基线在 F2、macro‑F1、Cohen's κ 等指标上进行对比。MOSAIC 在 SWEC‑ETHZ 短期/长期数据集上分别提升 κ 1.8/3.3 点，且训练速度约为 Transformer 的三分之一；在单通道数据集上也保持或略优于最佳基线。

**⚠️ 局限性**

局限性：自蒸馏仅能降低邻接图方差，无法纠正系统性偏差；单通道数据时模型退化为仅共享编码器；缺乏外部监督以进一步引导邻接图学习；未验证在颅外 EEG 上的泛化性能。

---

## 493. CID: Measuring Cognitive Work Through Semantic State Change

**arXiv ID:** 2611.09542 | [PDF](https://arxiv.org/pdf/2611.09542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 494. CucumberVision at the Edge: Accuracy, Throughput, and Energy of RGB-D Cucumber Sizing on Jetson Orin Nano and Raspberry Pi 4

**arXiv ID:** 2611.09490 | [PDF](https://arxiv.org/pdf/2611.09490v1)

**作者:** Manveen Kaur `[一作]` (University of Windsor), Shahpour Alirezaee `[通讯]` (University of Windsor)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

部署并评估了基于RGB‑D的黄瓜长度测量管线，在工作站、Jetson Orin Nano与Raspberry Pi 4三种硬件上复现精度，系统化分析了吞吐量与能耗，并绘制了完整的准确‑吞吐‑能耗包络。

**💡 创新点**

提出了多平台准确‑吞吐‑能耗的联合评估框架；证明提示式分割后端与硬件平台可分离；揭露NMS‑free检测头在INT8量化时的结构性失效并给出排除头部的修复方案。

**🔧 技术方法**

使用YOLO26n无NMS检测、提示式SAM/MobileSAM分割、基于Medial Arc Spline的长度估计；利用TensorRT FP16加速、INT8全图/头部量化；配合功耗与热持续监测实现端到端性能评测。

**📊 数据集**

基于50张温室黄瓜RGB‑D捕获图像（长度6.9–25.0 cm），配套真值标注，并在现场机器人上进行实时测量实验。

**📈 对比分析**

采用集群重采样与配对符号翻转检验对比，Jetson与工作站在MAE≈0.58 cm无显著差异；TensorRT FP16将检测延迟降至46 ms；Raspberry Pi能耗最高；三条非支配配置（Jetson原生、Jetson SAM、Pi原生）展示了不同的吞吐-能耗折衷。

**⚠️ 局限性**

仅测试单块板，未测mAP与IoU；温室现场环境与多果场景未验证；INT8量化失效仅针对特定NMS‑free检测头；能耗评估在实验室环境下完成，缺乏在实际温室条件下的验证。

---

## 495. Coordinatewise Epsilon for Adam under Heavy-Tailed Noise

**arXiv ID:** 2611.09596 | [PDF](https://arxiv.org/pdf/2611.09596v1)

**作者:** Jeonseong Kim `[一作]` (Yonsei University), Chenglin Fan `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了Adam优化算法在不同坐标的噪声和曲率变化下的收敛性，特别是共享参数对收敛的影响。

**💡 创新点**

提出了坐标化的上界和共享参数的下界，揭示了共享参数在不同坐标下的适应性限制。

**🔧 技术方法**

使用了Adam优化算法的理论分析，结合了坐标化的噪声和曲率假设。

**📊 数据集**

未具体提及使用的数据集，但讨论了在深度网络和语言模型训练中的应用。

**📈 对比分析**

通过与已有的重尾ℓ_1-平稳基准进行比较，展示了坐标化选择的优势，尤其是在噪声和曲率不均匀的情况下，性能达到了已知的最优时间指数。

**⚠️ 局限性**

共享参数的选择限制了算法在高维情况下的适应性，无法实现维度均匀的基准速率。

---

## 496. Optimal Regret for Profit Maximization in Bilateral Trade

**arXiv ID:** 2611.09558 | [PDF](https://arxiv.org/pdf/2611.09558v1)

**作者:** Matteo Castiglioni `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在仅能观察交易是否成功的情况下，如何在重复双边交易中学习最优的买卖价格对，从而最大化中介的利润，并给出了一个实现期望后悔率为 O(T^{2/3}log^{9/2}T) 的算法。

**💡 创新点**

创新点主要有：
1) 证明学习两侧价格并不比学习单侧价格更困难，后悔率上界与经典的一维动态定价相同；
2) 引入了保守的多阶段逐步淘汰框架，结合映射的二叉分解与相对学习技术，实现了在二维 CDF 基础上高效的精度提升；
3) 构造了满足可查询、变换次数有限、累计概率质量可控的映射，从而突破了先前 O(T^{3/4}) 的瓶颈。

**🔧 技术方法**

使用的主要技术包括：
- 将双边交易映射为在二维 CDF 上最大化 (1‑x₁‑x₂)F(x) 的问题；
- 对价格空间进行均匀离散化并利用一侧 Lipschitz 性进行粗略估计；
- 通过“保守逐步淘汰”将目标集合与可查询集合分离，限制探索代价；
- 采用映射的二叉分解与相对学习（dyadic decomposition + binary search）来高效估计同一行的 CDF 差值；
- 通过潜能函数证明映射更新次数与累计概率质量上界，从而保证样本复杂度。

**📊 数据集**

该工作是理论分析性质，未使用具体数据集，所有结果均在假设随机抽样的未知联合分布下给出。

**📈 对比分析**

与之前的 O(T^{3/4}) 以及 O(T^{7/10}) 上界相比，该算法在后悔率上实现了 O(T^{2/3}) 的最优上界（除对数因子外），并在理论上达到了经典单一买方定价的最优后悔率，表明二维问题并未出现维度灾难。

**⚠️ 局限性**

局限性包括：
- 目前仅针对二维双边交易，是否能推广到任意维度仍是未解问题；
- 对数因子较大（log^{9/2}T），实际实现中常数与对数项可能影响实用性；
- 依赖于均匀网格离散化，若分布极端稀疏或有强峰值，离散化误差可能增大。

---

## 497. Multimodal Training with Knowledge Traction for Video Anomaly Detection

**arXiv ID:** 2611.09536 | [PDF](https://arxiv.org/pdf/2611.09536v1)

**作者:** Xiao Tan `[一作]` `[通讯]` (Xidian University), Xiao Tan (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种弱监督视频异常检测方法，利用大语言模型生成先验文本知识并结合跨模态注意力机制，将视觉特征与文本信息深度融合。

**💡 创新点**

创新点包括：①通过LLM生成任务特定文本提示，并使用字典学习将嵌入压缩为稀疏码；②设计文本描述一致性正则化，提升帧级描述的可靠性；③引入双向跨模态注意力和知识拉伸模块，实现视觉与文本的互补融合。

**🔧 技术方法**

主要技术：大语言模型（GPT-3、BART）、稀疏字典学习、BLIP图像描述、YOLOv8检测辅助、I3D视觉特征提取、双向注意力机制、知识拉伸模块、端到端训练。

**📊 数据集**

实验数据集：ShanghaiTech 与 UCF‑Crime。

**📈 对比分析**

通过帧级 AUC 与多种弱/半监督 SOTA 方法对比，模型在 ShanghaiTech 上取得 98.17% AUC，在 UCF‑Crime 上取得 87.11% AUC，均明显优于前沿方法。

**⚠️ 局限性**

局限性：对先验知识的数量与平衡高度敏感；文本一致性正则化依赖 LL.M. 与检测模型的质量；对极少量异常样本的鲁棒性仍待提升。

---

## 498. Pangenome Peeling: Decomposing Weighted Adjacencies into Path-Forest Genomes

**arXiv ID:** 2611.09533 | [PDF](https://arxiv.org/pdf/2611.09533v1)

**作者:** Alitzel López Sánchez `[一作]`, David Sankoff `[通讯]` (University of Ottawa)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究逆向问题——从加权邻接图重构组成它的线性基因组集合，提出了两种算法（SEQ贪心逐边添加与MST最大生成树分解），并对其性能进行了评估。

**💡 创新点**

首次正式定义了pangenome peeling问题，推导了基因组数的下界，提出了基于最大生成树的全局结构重构方法，并与传统贪心方法做对比。

**🔧 技术方法**

利用图论技术（路径覆盖、最大生成树、路径森林约束）、贪心策略、最小权重匹配（用于匹配推断基因组与模拟基因组）以及DCJ‑indel距离评估。

**📊 数据集**

使用自定义模拟数据集：从单一线性基因组产生若干后代，按Poisson分布随机施加基因插入/删除、倒位、重排、融合、转移等事件，生成低、中、高进化水平的三类数据。

**📈 对比分析**

通过比较推断出的基因组数量、核心基因集差异（对称差）以及与原始基因组的DCJ‑indel距离来评估算法。MST方法在所有指标上均优于SEQ；在高进化水平下平均MAPE<75%，核心差异较小，DCJ距离更低。

**⚠️ 局限性**

当前模型仅处理唯一基因，未考虑基因复制；未利用已知的核心或外壳信息来约束搜索，导致解的多样性；对真实生物数据的鲁棒性和可扩展性仍待进一步验证。

---

## 499. An Effective Spoken Language Understanding Approach Leveraging Query-Adaptive Intent Detection

**arXiv ID:** 2611.09491 | [PDF](https://arxiv.org/pdf/2611.09491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 500. PreForceVision: Physics-Guided Optical Estimation of Bolt Preload from Bolt-Head Deformation

**arXiv ID:** 2611.09529 | [PDF](https://arxiv.org/pdf/2611.09529v1)

**作者:** Nehal Afifi `[一作]` (Karlsruhe Institute of Technology), Sven Matthiesen `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种通过观察螺栓头面微小变形来估算预紧力的光学方法PreForceVision。

**💡 创新点**

将预紧力估算视为逆向变形-力问题，利用物理引导的光学框架结合多模态验证与不确定性回归实现对螺栓预紧力的粗略估计。

**🔧 技术方法**

采用数字图像相关（DIC）/光流提取表面变形，结合有限元分析、应变计、光学三维测量以及回归/深度学习模型进行估计。

**📊 数据集**

以M16 8.8级螺栓为实验对象，收集不同预紧力（0、30、60、90）下的参考-加载图像对、应变计数据、FEM预测和光学三维扫描。

**📈 对比分析**

通过多模态验证（FEM、应变计、3D扫描）评估变形可观测性，并用线性/树模型对DIC特征与预紧力进行回归，MAE约为10%~15%，显示可实现粗略预紧力估计。

**⚠️ 局限性**

信号微小（微米级）导致对光照、纹理、相机对准要求极高，且需参考图像，当前仅在实验室控制条件下可行，现场应用仍具挑战。

---

## 501. RadarDINO: Distilling a Visual Foundation Model into mmWave Radar for Human Mesh Reconstruction

**arXiv ID:** 2611.09524 | [PDF](https://arxiv.org/pdf/2611.09524v1)

**作者:** Furkan Mert Algan `[一作]` (Technical University of Munich), Eckehard Steinbach `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为RadarDINO的两阶段框架，通过毫米波雷达输入实现高精度人体网格重建；

**💡 创新点**

创新点在于将DINOv3视觉特征跨模态蒸馏到雷达体素空间，提供体素级和部位级语义监督，并结合顶点、关节几何损失显式约束网格几何；

**🔧 技术方法**

采用DINOv3 ViT作为教师、Mask R‑CNN进行人像分割、雷达体素投影、3D CNN+时空自注意力编码器、跨注意力与MLP预测DINO描述子与支持度、学习的体部查询、SMPL‑X层以及多种损失函数（体素、部位、支持、几何、姿态等）；

**📊 数据集**

使用M4Human公开数据集，该数据集包含同步毫米波雷达和RGB图像的多模态人类动作；

**📈 对比分析**

与多种雷达基线（RT‑Mesh、RT‑Pose、RETR、P4Trans等）及RGB+雷达融合方法在MVE、MJE、MRE、TE等指标上进行对比，RadarDINO在所有三种数据划分上均获得最低误差，且模型参数仅2.7M、GFLOPs仅1.98，成为最轻量级且最精确的雷达仅方法；

**⚠️ 局限性**

存在的局限包括：与RGB DINO教师相比仍存在显著的教师-学生差距，雷达分辨率导致局部细节（如颈部）丢失，以及跨设备、跨环境泛化能力尚需进一步提升。

---

## 502. A Physics-Informed Learning Framework for Linking Microscopic Surface Degradation to Bolted Joint Functional Behavior

**arXiv ID:** 2611.09514 | [PDF](https://arxiv.org/pdf/2611.09514v1)

**作者:** Nehal Afifi `[一作]` (Karlsruhe Institute of Technology), Sven Matthiesen `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并验证了一套物理信息驱动的多模态学习框架，用于将微观表面腐蚀观测转化为螺栓接头的功能指标预测；

**💡 创新点**

创新点在于将多源图像、粗糙度与曝光信息通过相似性学习融合成校准腐蚀状态场，并在此场上进行基因式 PDE 发现与物理约束的预测，形成可解释且可追溯的腐蚀演化-功能耦合模型；

**🔧 技术方法**

使用了 U‑Net 语义分割、对比学习/相似性学习、PDE‑FIND/弱形式稀疏识别、遗传符号搜索、物理信息神经网络（PINN）以及多模态特征融合网络；

**📊 数据集**

实验数据来自多组螺栓接头样本的同步采集，包括显微镜图像、相机图像、粗糙度剖面、曝光描述和扭矩/夹紧力曲线；

**📈 对比分析**

与传统基于图像或粗糙度的回归、无 PDE 约束的神经预测和手工指定 PDE 约束的模型进行对比，结果显示加入校准腐蚀场和 PDE 约束后，在保持相同输入复杂度的前提下，摩擦系数预测误差可降低约10%–20%，且模型在外部时间滚动预测中表现更稳健；

**⚠️ 局限性**

局限性包括对数据量要求高（需同一表面多时间点的多模态记录）、对跨模态配准误差敏感、校准腐蚀场与真实电化学状态可能不完全对应、PINN 训练易受损失平衡与边界条件定义影响、遗传符号搜索易过拟合噪声且需稳定性检验。

---

## 503. RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers

**arXiv ID:** 2611.09597 | [PDF](https://arxiv.org/pdf/2611.09597v1)

**作者:** Yijun Li `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的强化学习方法RuACT，该方法通过将评分标准作为适应性评论者和教师来扩展传统的基于评分标准的强化学习，以解决开放式任务中的不可达和饱和评分标准问题。

**💡 创新点**

创新点在于将不可达评分标准用作教师提供奖励引导，而将饱和评分标准分解为子评分标准以恢复奖励的细粒度区分，从而提高学习信号的有效性。

**🔧 技术方法**

使用了基于奖励的自我蒸馏技术和细粒度的评论机制，结合了教师和评论者的角色来优化学习过程。

**📊 数据集**

使用了Qwen3-4B和Qwen3-1.7B模型进行医学和科学推理的基准测试，数据集包括RaR-Medicine、HealthBench、RaR-Science和ResearchQA。

**📈 对比分析**

与GRPO、评分标准重加权和评分标准引导探索等方法进行比较，RuACT在所有四个医学和科学基准上均表现优异，尤其在ResearchQA上从64.82%提升至69.44%。

**⚠️ 局限性**

限制在于RuACT依赖于基于LLM的评分判断和子评分标准提案，生成的子评分标准可能无法完全捕捉预期要求，且评分状态的识别基于有限的样本，可能影响准确性。

---

## 504. Dual-Number Associative Arrays for One-Pass Provenance and Sensitivity

**arXiv ID:** 2611.09589 | [PDF](https://arxiv.org/pdf/2611.09589v1)

**作者:** Timothy Habashy `[一作]` (Massachusetts Institute Of Technology), Jeremy Kepner `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出一种平面扩展 D_k(S)=S⋉S^k，在 GraphBLAS 计算中同时携带普通值与 k 个独立的辅助通道，实现稀疏半环运算的双数扩展；在最小-加（min-plus）半环下，辅助通道给出经过指定边的最短行走距离减去该边权，进而实现精确的最短路边权归属测试。

**💡 创新点**

创新点在于：①把双数扩展推广到任意可交换半环，形成一个分裂的平方零扩展；②在最小-加半环上给出辅助通道的几何解释，提供“detour distance”与边归属检测；③提供基于 GraphBLAS 的稀疏实现，并证明其在无负环条件下的收敛性；④通过统一的代数框架支持多种半环，而不依赖特定的 min-plus 细化。

**🔧 技术方法**

主要技术包括：半环代数与扩展理论、GraphBLAS 与 SuiteSparse 库、稀疏矩阵乘法、双数运算的矩阵级乘法、终止检查（value+auxiliary 同步冻结）以及针对最小-加半环的 detour distance 计算。

**📊 数据集**

使用合成随机有向图：顶点数 100,000，平均出度 8，边权为 1–10 的整数；实验中针对不同 k（2–32）和不同 n（10,000–200,000）进行多次重复测试，收集中位数运行时间与收敛步数。

**📈 对比分析**

比较方法：①双数递推（dual recurrence）；②k+1 个单源求解（sequential）；③批量（k+1）× n 的 GraphBLAS 解决方案（batched）。结果显示：dual 方法比 sequential 快 3.39×，但比 batched 慢 1.50×；随着 k 增大，dual 方法相对 sequential 更快；在 100k 顶点、k=8 的实验中，dual、sequential、batched 的中位数运行时间分别为 0.231 s、0.491 s、0.126 s；dual 的收敛步数 N* 与 batched 的步数 D 的比值在 1.50–1.58 之间，说明 dual 需要多 1.5–1.6 倍的迭代。

**⚠️ 局限性**

局限性：①在存在负权环时无法保证收敛；②相较于专门的 min-plus 批处理方法，dual 需要更多迭代，导致性能略逊；③在路径相同权时，辅助通道返回的是所有最短路径中的边集合，无法区分具体路径；④实验仅基于合成图，缺乏真实网络的硬件独立性评估。

---

## 505. Lightweight Path-Aware Frontier Re-ranking for LiDAR-Based UAV Exploration

**arXiv ID:** 2611.09561 | [PDF](https://arxiv.org/pdf/2611.09561v1)

**作者:** Duc Manh Do `[一作]` (Vietnam National University), Xiem HoangVan `[通讯]` (Vietnam National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种轻量级的基于路径估计的前沿重排序方法，用于LiDAR驱动的UAV三维探测；

**💡 创新点**

创新点在于：①只对Top‑N前沿候选进行时间受限的RRT*路径评估，避免全局规划开销；②在选定最佳前沿后直接重用该路径用于导航，减少冗余规划；

**🔧 技术方法**

技术包括：多分辨率前沿检测与聚类、基于信息增益与欧氏距离的轻量级Utility函数、时间受限RRT*路径规划、路径重用机制；

**📊 数据集**

实验使用三种仿真仓库环境（76m×28m×8m）中的LiDAR点云；

**📈 对比分析**

与MRF、FCS、EPIC等LiDAR前沿探测基线对比，结果显示在所有环境中均取得更短的探测路径（最大14.1%距离缩短），并保持竞争性的探测时间和相似的地图细节比例；

**⚠️ 局限性**

局限性包括：实验仅在静态、定位精确的仿真环境下验证；没有直接评估CPU占用和规划延迟；固定的Top‑N和RRT*时间预算不一定适用于所有场景；

---

## 506. Deliberate Pacing for Human Oversight in Agentic Software Development

**arXiv ID:** 2611.09550 | [PDF](https://arxiv.org/pdf/2611.09550v1)

**作者:** Tobias Reichel `[一作]` (Heidelberg University), Steffen Herbold `[通讯]` (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并阐述了“刻意节奏（Deliberate Pacing）”的概念，旨在通过在智能代理自主开发中有策略地插入人类干预，保持有意义的人类监督；并基于20篇文献的多声源文献综述（MLR）总结了该概念的三大动因，进一步提出三条研究问题与方法。

**💡 创新点**

创新点在于：①将开发速度与节奏区分，强调节奏对监督机会的影响；②提出节奏决策需跨软件生命周期（SDLC）考量，突出各阶段相互依赖；③基于多声源综述构建“刻意节奏”理论框架，并给出具体的研究议程与实验设计。

**🔧 技术方法**

采用的技术主要是：多声源文献综述（MLR）+ 归纳主题分析；未来研究计划中还会用到准实验（quasi‑experiment）和仿真（simulation）方法来评估节奏策略。

**📊 数据集**

未使用具体数据集；综述样本为20篇文献（8篇学术 + 12篇灰文），没有直接收集软件仓库或代理执行日志。

**📈 对比分析**

目前尚未进行实验或性能比较；本文仅提出实验设计与评估指标（人类注意力、开发时间、重工成本等），未来将通过准实验和仿真来比较不同节奏策略的效果。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，结论基于文献综述与理论推导；②高成本实验的可行性不确定；③仅考虑了文献中的“慢速开发”动因，未涉及所有可能的代理行为与上下文；④对不同软件项目类型的适用性仍需进一步检验。

---

## 507. Marginal-Nonuniform Multiclass Learning

**arXiv ID:** 2611.09543 | [PDF](https://arxiv.org/pdf/2611.09543v1)

**作者:** Nataly Brukhim `[一作]` (Apple), Maximilian Thiessen `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文在可实现多分类学习中，研究了边缘非均匀学习框架并给出了完整的可学习率三分法；

**💡 创新点**

创新点在于提出并使用了新的组合维度——无中心DS-弹射维度（uncentered DS‑eluder dimension）来刻画学习速率，并证明其对三种速率（指数、线性、任意慢）完全决定；

**🔧 技术方法**

主要技术包括分布性消耗引理、DS‑shattering序列与强序列的构造，以及对无中心弹射维度的细致分析；

**📊 数据集**

该工作为理论性研究，无依赖具体数据集；

**📈 对比分析**

与传统统一PAC、通用学习以及概念非均匀学习相比，本文在边缘非均匀设置下提供了精确的指数与线性上界与下界，说明可实现的最优速率只能是指数、线性或任意慢；

**⚠️ 局限性**

局限性包括：仅处理可实现（realizable）情况，未覆盖不确定性（agnostic）或噪声模型，且多分类的具体实验验证尚未展开。

---

## 508. Atik: A Shared-Mesh Transformer Accelerator for RISC-V

**arXiv ID:** 2611.09522 | [PDF](https://arxiv.org/pdf/2611.09522v1)

**作者:** Ahmed Zeer `[一作]` (Yildiz Technical University), Erkan Uslu `[通讯]` (Yildiz Technical University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个开放源代码的RoCC加速器Atik，用于RISC‑V RocketChip，能够加速Transformer模型中的矩阵乘法和注意力操作，并支持BF16输入/输出。

**💡 创新点**

创新点在于共享一个固定点MAC网格同时服务矩阵乘法和注意力（QK^T 与 PV 乘法），采用流式在线softmax而不需要完整得分矩阵；通过RoCC接口实现紧耦合、可通过小描述符驱动；在FireSim上实现周期精确仿真并提供Sky130面积估计。

**🔧 技术方法**

技术包括RoCC自定义协处理器、共享固定点MAC网格、BF16↔fixed‑point转换、查找表实现指数与倒数、流式在线softmax调度、Tile‑DMA、FireSim周期精确FPGA仿真、Chipyard/Verilator RTL验证、Sky130物理设计流。

**📊 数据集**

使用了两个独立基准（矩阵乘法、注意力）以及从TinyBERT、Vision Transformer (ViT) 和 GPT‑2 预填充阶段抽取的三组C语言模型，涵盖不同序列长度、模型宽度、头数。

**📈 对比分析**

通过与同一Rocket核心跑BF16软件实现的基准对比，并在FireSim上测量周期；对独立基准直接算速率，针对模型级工作使用混合投影；结果显示矩阵乘法31.5×加速、注意力225.8×加速，ViT 44×，TinyBERT 96×，GPT‑2 32×，峰值更高。

**⚠️ 局限性**

局限性包括：未与GPU或向量机做跨平台比较；softmax在单个标量通道，可能成为大得分块的瓶颈；未利用稀疏注意力；面积估计仅为独立块，未包含完整SoC的时序、功耗和验证；未加速KV缓存解码阶段。

---

## 509. BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

**arXiv ID:** 2611.09505 | [PDF](https://arxiv.org/pdf/2611.09505v1)

**作者:** Gonçalo Gomes `[一作]` (Instituto Superior Técnico, University of Lisbon), Chrysoula Zerva `[通讯]` (Instituto de Telecomunicações)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 BEiTScore++，一种在图像字幕评估中同时给出质量得分和置信区间的 uncertainty-aware 评估指标。

**💡 创新点**

创新点在于：① 通过量化回归和链式量化构造无交叉、单向量化的置信区间；② 采用 centered interval loss 兼顾覆盖率和 sharpness；③ 引入后置 conformal calibration 并结合子群（cluster）批处理提升覆盖率；④ 在单次前向传播即可得到置信区间，显著提高推理效率。

**🔧 技术方法**

技术手段包括：BEiT‑3 视觉‑文本交叉编码器、量化回归头、centered interval loss、链式量化构造、后置 conformal risk control、子群（k‑means）分组批处理、Monte Carlo dropout 对照实验。

**📊 数据集**

数据集：训练使用 VICR、Polaris、Nebula、EvalMuse‑40k；评估包括 VICR、Polaris、Nebula、Flickr8K‑Expert、Composite、IIW、DOCCI、Localized Narratives、MSCOCO、Docent、ArtCaps。

**📈 对比分析**

方法对比：与原 BEiTScore、MC dropout、其他无参考指标（如 CLIPScore、LLM‑based metrics）比较；在 Kendall's τ、UPS、sharpness、coverage 等指标上保持或提升性能；覆盖率接近 90% 并且在多数据集上与人类评分相关性保持在 78–79%，宽度与误差相关性高，单前向传播即可得到置信区间。

**⚠️ 局限性**

局限性：仅在英文字幕上验证；需人工评估辅助；对极端分布偏移、少数类和多语言场景的鲁棒性尚未充分测试；计算训练成本仍高；不保证在所有任务中的公平性与正确性。

---

## 510. FusionCLIP: Multimodal Contrastive Pretraining for Fusion Plasma Diagnostics

**arXiv ID:** 2611.09493 | [PDF](https://arxiv.org/pdf/2611.09493v1)

**作者:** Qiang Chen `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了FusionCLIP，一种面向等离子体诊断的多模态对比预训练框架；

**💡 创新点**

创新点在于结合了视频的局部信息选择（Spatiotemporal Stability Selection）和异步诊断信号的时间查询编码（Temporal Query Encoder），并通过跨模态对比学习实现两模态特征对齐；

**🔧 技术方法**

使用了Transformer结构、频域低通滤波选择机制、时间偏置注意力、对比损失等技术；

**📊 数据集**

在自构建的EAST Multimodal Diagnostics（EAST‑MMD）数据集上进行预训练与评估，共包含1307次EAST托卡马克放电的可见光视频与38种诊断信号；

**📈 对比分析**

与从零开始训练、通用视觉预训练模型（如DINO、CLIP-Base）以及仅使用时间序列的基线进行对比，FusionCLIP在三项诊断任务（中断预测、ELM分类、H98回归）上获得综合得分89.07%，显著优于基线和其它对比预训练方案；

**⚠️ 局限性**

仅在EAST数据上验证，缺乏跨装置/跨托卡马克的迁移性研究，且对不同采样率诊断信号的鲁棒性尚待进一步探索。

---

## 511. Dual-Region Slice Aggregation for CT-Based Renal Tumor Classification

**arXiv ID:** 2611.09599 | [PDF](https://arxiv.org/pdf/2611.09599v1)

**作者:** Yuan Liang `[一作]`, Abraham Campbell `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种双区域（肿瘤区与10 mm周围环境）CT分类架构，并将深度特征与手工 radiomics 通过门控融合。

**💡 创新点**

创新点在于：①在共享冻结的 DINOv3 视觉编码器上分别提取肿瘤与周围环境特征；②使用位置感知切片聚合器（PSA）在切片级别学习交互后再进行固定 PCA 压缩；③通过双标量门控将深度特征与 radiomics 动态加权融合。

**🔧 技术方法**

技术包括：冻结 DINOv3 ViT‑B/16 作为特征提取器；位置感知 Transformer 作为切片聚合器；固定 PCA 进行降维；多尺度门控融合（乘法门）；三层 MLP 进行最终分类；以及标准化、梯度裁剪等训练技巧。

**📊 数据集**

使用 KiTS23 内部 396 例（约 70% ccRCC）进行训练/验证/内部测试，外部测试使用 111 例 TCGA/TCIA（76.6% ccRCC）和社区注释（RCC‑AID、BAMF‑AIMI）提供的肿瘤掩码。

**📈 对比分析**

与全场读取、单肿瘤读取以及 RadioDINO-B/16 控制进行对比；在内部测试集上 AUC 为 0.8078±0.0154，外部测试集上 AUC 为 0.9094±0.0015，明显优于全场读取（+7.1% 内部、+4.9% 外部）和单肿瘤读取（+1.5% 内部、+2.4% 外部）。

**⚠️ 局限性**

局限性包括：①仅使用公开 CT 数据，缺乏更大样本或多中心验证；②冻结 DINOv3 可能限制了对医学图像的专门化；③实验未探索更深层次的多模态融合或更复杂的聚合策略。

---

## 512. Transformer-FSQ: Discrete Latents for Extreme KV-Cache Compression

**arXiv ID:** 2611.09598 | [PDF](https://arxiv.org/pdf/2611.09598v1)

**作者:** Andras Szecsenyi `[一作]`, Amos Storkey `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种名为Transformer‑FSQ的KV缓存压缩方法，将Transformer的键值投影替换为有限标量量化（FSQ）瓶颈，使用整数代码而不存储额外元数据，实现极端压缩（12×–97.5×）仍能保持长上下文检索与摘要性能。

**💡 创新点**

创新点在于：1）将键值投影改为FSQ离散潜在，省去代码库和量化元数据；2）闭式初始化并仅微调插入参数，保持原模型冻结；3）在极端压缩范围内显著优于所有基线；4）提供融合的Triton解码核，实现解码吞吐与原始模型相当或更快；5）可与token‑eviction方法堆叠，进一步扩大可用上下文容量。

**🔧 技术方法**

核心技术包括：有限标量量化（FSQ）、离散潜在压缩、闭式预训练转换、微调（仅训练插入的投影和解码器）、RoPE位置编码交互、Triton融合解码核以及与PyramidKV等基线方法的对比实验。

**📊 数据集**

使用的数据集与评测：ClimbMix（预训练与微调数据），RULER（合成检索任务，提供精确基准），LongBench（多页文档摘要、问答和分类任务），以及对Qwen3系列（0.6B、1.7B、8B）模型的实验。

**📈 对比分析**

与token‑eviction、scalar、vector、low‑rank projection和latent‑attention五类基线进行对比。结果显示：在12×–97.5×压缩下Transformer‑FSQ在RULER检索任务中准确率最高，LongBench ROUGE‑L也保持最高；在解码吞吐上，32×压缩时与原始SDPA相当，97.5×及更高时性能超过原始实现；堆叠eviction后可实现百万-token级别上下文容量。

**⚠️ 局限性**

局限性包括：1）极高维产品网格在低维验证外缺乏实验支持；2）对算术推理、代码生成、多步推理等任务的影响未评估；3）FSQ在极低位宽或高精度细节下的鲁棒性待进一步验证；4）与权重量化组合时需要重新微调，未给出统一的复合压缩方案。

---

## 513. An Empirical Evaluation of a Weighted PALU Model for Internet Traffic Networks

**arXiv ID:** 2611.09575 | [PDF](https://arxiv.org/pdf/2611.09575v1)

**作者:** Wood Moore `[一作]` (Massachusetts Institute Of Technology), Jeremy Kepner `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

验证并扩展了PALU模型，对MAWI WIDE网络流量图进行加权建模与参数估计，探究不同观测窗口下的参数稳定性。

**💡 创新点**

首次对PALU进行真实流量数据验证，并提出加权扩展及采样观测模型；证明α稳定、λ和θ随窗口变化。

**🔧 技术方法**

使用图论、偏好附着模型、幂律拟合（Clauset‑Shalizi‑Newman）、零截断泊松拟合、最小二乘回归、KS检验等技术。

**📊 数据集**

采用MAWI WIDE Samplepoint F的每日15分钟抓包数据，构建通信图。

**📈 对比分析**

通过多窗口参数估计与KS统计量比较模型与实测分布；结果显示α≈2稳定，λ和θ随窗口增大显著变化。

**⚠️ 局限性**

局限包括网络无向化、泊松卫星模型被拒、弱强度‑度关系在小窗口不可靠、节点流失/出现未被考虑、对高聚类和路径长度等结构缺乏描述。

---

## 514. Query-Guided RoPE Modulation for Lost-in-the-Middle Mitigation in Long-Context LLMs

**arXiv ID:** 2611.09554 | [PDF](https://arxiv.org/pdf/2611.09554v1)

**作者:** Peiyu Zhang `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种训练无关的推理时方法 QGPM，通过读取模型自身注意力头的查询相关性来重新调节 RoPE 位置，从而提升长上下文语言模型对中间信息的利用。

**💡 创新点**

创新点在于：①依据查询相关性动态调节 RoPE 位置；②低相关性文本仅在快旋转平面压缩；③采用广泛与尖锐两种相关性映射分别在低层与高层使用；④所有 token 保持原位置且无需微调。

**🔧 技术方法**

使用技术包括 RoPE 位置编码、注意力头相关性读取、频率感知步幅计算、分层相关性映射以及推理时双阶段预填充。

**📊 数据集**

实验使用 NoLiMa 检索任务与 RULER（QA‑1/QA‑2、NIAH‑MK/NIAH‑MV）长上下文基准，覆盖 Qwen2.5‑7B、Qwen3‑4B、Ministral‑8B 与 Granite‑3.3‑8B 四种 backbone。

**📈 对比分析**

与未改 RoPE 的 Vanilla 以及四种推理时基线（LLMLingua‑2、Ms‑PoE、Self‑Extend、STRING）对比，QGPM 在所有 backbone、任务和上下文长度上平均性能最高，尤其检索任务提升 9–18 个百分点，整体平均提升 2–9 个百分点。

**⚠️ 局限性**

局限性包括：①需访问模型内部（注意力向量与 RoPE 表）仅适用于可公开权重且使用 RoPE 的模型；②额外一次预填充增加首次生成延迟；③在极长上下文下相关性估计仍受限，性能提升有限。

---

## 515. Embedding Predictive Adaptation and Refinement for Multi-Modal Industrial Anomaly Detection

**arXiv ID:** 2611.09521 | [PDF](https://arxiv.org/pdf/2611.09521v1)

**作者:** Jaeho Cho `[一作]` (Ulsan National Institute of Science and Technology), Taehwan Kim `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了EPAR框架，结合embedding预测与VIC正则化对预训练编码器进行域自适应，并使用噪声抑制网络在多模态RGB+3D数据上实现细粒度异常定位

**💡 创新点**

创新点在于：①基于mask embedding prediction与patch‑level VIC正则的特征自适应，显著缓解域差距；②NSN利用每个patch与其最近正常原型的关系而非伪缺陷外观进行异常图细化，提升真实缺陷的定位鲁棒性

**🔧 技术方法**

主要技术包括：ViT‑DINO/PointMAE预训练、mask embedding prediction、VIC正则化、CNN‑based噪声抑制网络、近邻搜索与多模态融合

**📊 数据集**

使用的数据集为MVTec 3D-AD、Eyecandies和PD-REAL

**📈 对比分析**

与LSFA、G^2SF等SOTA方法对比，在I-AUROC上提升约1.3%，在AUPRO@1%提升约1.9%；在少样本（5/10/50 shot）设置下也保持领先，表明在多模态工业检测中的高效性

**⚠️ 局限性**

局限性在于：多尺度特征提取虽然提升精度，但推理延迟显著增加；NSN对像素级定位效果好，但对样本级检测提升有限，未来需进一步平衡速度与准确性

---

## 516. Memory as a Temporal Resource: Lifetime Contracts for Multi-Retention Systems

**arXiv ID:** 2611.09518 | [PDF](https://arxiv.org/pdf/2611.09518v1)

**作者:** Adrian Jackson `[一作]` `[通讯]` (University of Edinburgh), Adrian Jackson (University of Edinburgh)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于事件的有效期合同，将软件生命周期与硬件保留级别解耦，实现按需动态保留的数据管理。

**💡 创新点**

创新点在于：①将保留（retention）从硬件属性转化为软件可声明的“有效期”合同；②设计跨层次的事件驱动协议，使应用能指定“直到阶段结束/周期结束”之类的语义；③在 STT‑MRAM 等新型存储上实现多保留级别（短期、中期、长期）并通过 FPGA 原型验证迁移成本。

**🔧 技术方法**

技术包括：STT‑MRAM 物理保留级别划分、事件驱动有效期合同接口（C/C++ API）、基于 trace 的保留级别决策模拟器、FPGA（AMD Alveo U280）复制核进行逻辑级别迁移实验，以及针对多保留级别的刷新/升级/重计算策略。

**📊 数据集**

实验数据集主要有：1) 32×32 矩阵乘法的完整访问 trace；2) FPGA 上的 256 KiB–128 MiB 迁移测量；3) 通过张量/小块划分的关注机制（tiled attention）示例。并未使用公开大规模深度学习数据集，而是基于合成的矩阵乘法与注意力块。

**📈 对比分析**

比较方法：与全久期存放（all‑durable）方案、静态混合策略以及“oracle”最优保留级别进行对比。结果表明：利用有效期合同在保留安全的前提下，可实现 45–55% 的能耗下降，30–50% 的时延缩短；FPGA 迁移吞吐量约 3.3–4.9 GB/s，迁移时间对大块更友好。性能提升受迁移成本、保留窗口选择和内存容量压力影响。

**⚠️ 局限性**

局限性包括：①仅在仿真和 FPGA 原型上验证，缺乏真实低保留 STT‑MRAM 设备的可靠性测量；②保留级别迁移和刷新开销仍可能抵消能耗/时延收益，需在更大规模上评估；③需手动或自动化注解，未完全覆盖未注解的默认久期行为；④并发与动态调度导致的超时风险需更复杂的同步与安全保证；⑤在高温/老化环境下的失效概率与 ECC 结合仍需深入研究。

---

## 517. Benchmarking Speaker Embedders in the Sub-Second Regime for Real-Time Voice AI Agents

**arXiv ID:** 2611.09516 | [PDF](https://arxiv.org/pdf/2611.09516v1)

**作者:** Abhishek Gautam `[一作]` `[通讯]` (Indian Institute of Technology Guwahati), Abhishek Gautam (Indian Institute of Technology Guwahati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在实时语音 AI 代理下，针对子秒语音片段对背景语音干扰进行声学验证的基准研究

**💡 创新点**

系统化地评估多种公开声纹嵌入器在低延迟、短时长且伴随背景语音的实际环境中性能，并揭示降噪前端对验证的负面影响及嵌入维度冗余问题

**🔧 技术方法**

声纹嵌入网络（TitaNet‑Large/Small、ECAPA‑TDNN、CAM++、SpeakerNet），Silero VAD，DTLN 与 DeepFilterNet 降噪前端，余弦相似度判别

**📊 数据集**

400 个真实客户通话录音（含 2000+ 真实背景语音插入），以及 11,070 条加权标注的实时生产流量

**📈 对比分析**

在受控实验中 CAM++ 在原始音频上达到最低 EER（28.69%），但在生产流量上 TitaNet‑Small 与完整 VAD 片段取得最佳 30.57% EER；降噪前端普遍提升误差，截断 700 ms 片段使所有模型 EER 上升 0.8–2.2 分；前 92 维 TitaNet 嵌入维持原性能，后维度多为人口属性

**⚠️ 局限性**

绝对 EER 接近 30%（受限于子秒片段），未处理重叠语音，降噪前端对验证有负面影响的结论仅在本人口偏好（主要为男性印地语使用者）下成立，需在多元化群体中进一步验证

---

## 518. V-STORM: A Structure-Aware Grounded Chain of Thought Dataset for Faithful and Fine-Grained Spatio-Temporal Video Reasoning

**arXiv ID:** 2611.09511 | [PDF](https://arxiv.org/pdf/2611.09511v1)

**作者:** Dian Jiao `[一作]` (Zhejiang University), Siliang Tang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了V-STORM框架，利用时空场景图实现视频推理任务的结构化数据生成与评估。

**💡 创新点**

创新点在于通过STSG实现精细的时空证据可追溯链（GCoT）以及可验证的多步推理路径。

**🔧 技术方法**

技术包括层次化感知与语义抽象、图驱动的问答生成、基于LLM的结构保持翻译与验证。

**📊 数据集**

使用了多源公开视频数据（SAV、ActivityNet、Panda、WebVid、InternVid）构建260K训练集和3.3K人工验证基准。

**📈 对比分析**

与现有数据集对比，V-STORM在细粒度时空定位与组合推理上显著提升（整体准确率提升至约79%，组合推理从19%跃升至73%）。

**⚠️ 局限性**

局限在于对跨视频长期关联的追踪仍易出错，且需依赖高质量的STSG构建，人工审核成本仍不可忽视。

---

## 519. On the Effectiveness and Limitations of Open-Weight LLMs for Low-Resource Offensive Language Detection in Danish

**arXiv ID:** 2611.09506 | [PDF](https://arxiv.org/pdf/2611.09506v1)

**作者:** Elnur Adl Zarabi `[一作]` (Texas A&M University-Corpus Christi), Taoran Ji `[通讯]` (Texas A&M University-Corpus Christi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了多款开源大语言模型（LLaMA、Mistral、Gemma、Qwen）在丹麦低资源仇恨语言检测任务中的零样本、少样本、微调以及不同提示策略的效果，并与三种 BERT 族基线进行对比。

**💡 创新点**

创新点在于首次对丹麦语言的低资源仇恨检测进行多模型、多提示、多评估设置的全面实验，揭示提示设计与模型预训练之间的交互关系，并验证 ASCII 转写与噪声去噪指令对性能的负面影响。

**🔧 技术方法**

采用了 LoRA 轻量级微调、十一种提示模式（包括语义、格式、理论、观点等）、对抗性无监督微调（去噪）以及格式解析器对模型输出进行判别的技术。

**📊 数据集**

使用了公开的 DKHate 语料库（约3,300 条 Facebook 与 Reddit 评论，二元标签），并在平衡、自然不平衡、平衡训练/不平衡测试三种评估设置下进行实验。

**📈 对比分析**

与 BERT 族基线（mBERT、DaBERT、ScandiBERT）比较发现，即使未做任务专门微调，开源 LLM 在平衡评估中也能超过三种 BERT 基线；微调后 F1 在平衡数据上最高可达 0.84，非平衡数据上提升至 0.62，整体表现优于所有 BERT 方案，但在极不平衡情形下仍存在显著性能下降。

**⚠️ 局限性**

主要局限包括：数据集规模有限（仅 3,300 条样本，正例仅 425 条），实验仅涵盖 7–9B 参数范围内的四款 LLM，未覆盖更大模型或多分类标签；使用单一随机划分，缺乏置信区间或显著性检验；以及对非合规输出的解析器可能导致误判，尤其在 choice‑QA 模式下。

---

## 520. Hardness of Unique Shortest Vectors and Bounded Distance Decoding

**arXiv ID:** 2611.09497 | [PDF](https://arxiv.org/pdf/2611.09497v1)

**作者:** Sergei Kudria `[一作]` (City University of Hong Kong Shenzhen), Jason Luo `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文证明了在任何固定的 ℓ_p 规范（1≤p<∞）下，唯一最短向量问题（uSVP）在任意常数唯一性因子 >1 以及有界距离解码问题（BDD）在任意正比例 <1/2 的最小距离处都是随机多项式时间的 NP‑难问题。并给出了从 SAT（含 Valiant‑Vazirani 隔离）到 uSVP、BDD 的随机还原，并扩展到所有固定的 p 与常数因子。

**💡 创新点**

创新点在于：
• 统一证明了 uSVP 在所有常数唯一性因子和 BDD 在所有 ρ<½ 的情形下的 NP‑难性；
• 采用张量扩展与权重/范数下界相结合的技术，使得即使在固定维度下也能获得任意大常数因子；
• 在 p>2 或 p=∞ 时利用插值/矩阵重构与 Hadamard 变换，得到多项式因子的硬度；
• 通过随机化的隔离与判定电路，避免了对强假设（如 GapSETH）的依赖。

**🔧 技术方法**

核心技术包括：
1. Valiant‑Vazirani 隔离（在 SAT→BDD 的一步里）
2. 将电路输入映射为可兼容选择，并用线性约束（矩阵/张量）编码
3. 张量加权（symmetrized 张量）来放大非秩‑一解的权重/范数
4. 解析重构（weighted moments + Hadamard）和矩阵重构来保证唯一性
5. 随机化的校验矩阵和概率隔离证明
6. 结合 Lattice 生成与整数张量构造，给出完整的 uSVP、BDD 实例。

**📊 数据集**

无实验数据；本文为理论复杂度证明，没有使用任何数据集。

**📈 对比分析**

本文并未给出算法或实验比较；其贡献是证明某些问题在理论上是 NP‑难的，因此不涉及性能评估或方法比较。

**⚠️ 局限性**

局限性：
• 对于欧氏规范（p=2），超常数唯一性因子的 NP‑难性仍未解决；
• 证明依赖随机化；若需确定性还原则尚未完成；
• 结果只给出存在性证明，未给出具体多项式上界或可实现的算法；
• 对于更高阶的 ρ（接近 ½）或更大唯一性因子，需要更大张量阶，导致维数和构造成本上升。

---

## 521. Verifier Noise Is a Temperature: Testing Tilt Predictions in Finite-Budget LLM-Guided Search

**arXiv ID:** 2611.09699 | [PDF](https://arxiv.org/pdf/2611.09699v1)

**作者:** Xiaomeng Li `[一作]` `[通讯]` (Link Heart Limited), Xiaomeng Li (Link Heart Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了LLM引导的搜索循环（keep‑if‑better）在噪声评估下的动力学，推导出搜索过程的平稳分布（tilt）以及噪声对有效温度的影响，并在 bin‑packing、TSP 和 prompt‑优化三种任务上验证理论预测。

**💡 创新点**

创新点在于：1）将Gumbel/对数iterm噪声与可逆突变算子结合，证明搜索过程在小步长极限下收敛到倾斜的 Gibbs 分布；2）给出一般对称噪声下的温度公式 β=4F'(0)；3）对 minibatch gate 的温度与 batch 大小的 √m 关系做定量化；4）提供缓存、独立采样与噪声尾部相互作用的理论比较。

**🔧 技术方法**

技术包括：可逆突变 Markov kernel、Barker / logit 接受规则、小步长 Langevin 拓扑、梯度/对称噪声分析、蒙特卡洛仿真、校准走（无选择突变）以估计 β 与质量分布。

**📊 数据集**

使用的公开数据集有：bin‑packing (标准实例集合)、Traveling‑Salesman Problem (TSPLIB)、以及若干 prompt 评估任务（冻结模型评分）。

**📈 对比分析**

方法比较：在匹配评估次数下比较独立采样 (IID)、缓存 (C) 与重新评估 (R)。实验结果显示：R 在低噪声/高批量下达到理论 tilt 限值，C 由于记录最大值的“winner’s curse”导致报告分数过高；在 bin‑packing 中 R 在 150–300 步内获得 62–88%（编辑）或 109–119%（新启发式）的提升，提示 tilt 作为上限；在 TSP 与 prompt 优化中预测不完全成立。

**⚠️ 局限性**

局限性：1）假设突变算子可逆，LLM 实际突变不一定满足；2）仅在小步长/弱噪声下理论可证明，强噪声或大步长时收敛性未知；3）缓存策略无闭式增长解析；4）实验预算有限，未能观察到不同起点的链是否混合到同一平稳分布；5）校准走可能未达到平稳，导致 tilt 估计不稳定；6）对噪声尾部和质量尾部的相互作用给出经验式结论，缺乏严格理论证明。

---

## 522. Real-Time Self-Penetration Repair for Skinned Characters with Soft Contact Budgets

**arXiv ID:** 2611.09693 | [PDF](https://arxiv.org/pdf/2611.09693v1)

**作者:** Yifei Zhu `[一作]` `[通讯]` (Tohoku University), Yifei Zhu (Tohoku University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练、可插拔的后处理模块，利用姿态修正和皮肤压缩来消除在运动生成或重定向过程中产生的自相交问题。

**💡 创新点**

创新点在于：①将姿态修正与皮肤压缩通过静态软预算分配结合；②采用采样顶点的局部二次规划实现快速姿态修正；③支持多体形、任意运动模型，仅需离线生成兼容的角色包即可。

**🔧 技术方法**

技术包括：离线角色包构建、基于角度增量的平方-铰链二次规划、紧耦合的无状态接触压缩、深度触发紧迫度调度、可选布料厚度调整以及单核心实现的子帧处理。

**📊 数据集**

使用了记录的生成器输出和《Seamless Interaction》数据集中的对话视频，以及 SMPL-H/SMPL-X 形体，评估未见的运动片段。

**📈 对比分析**

与复制运动、仅姿态修正或仅皮肤压缩等基线相比，最终皮肤深度误差下降至<20mm的帧占比显著提高（如手-胸部交互深度从%降至%），平均处理时间约为10~15 ms（单核心），留给主机的时间足够。

**⚠️ 局限性**

局限性：无法保证完全消除所有深层或全局自相交；依赖采样几何与预设接触对；不考虑根部平移、整体平衡、环境接触或动态弹性；不支持复杂服装或手指细节；在不同平台和模型上尚未广泛验证。

---

## 523. ODEAction: From Predicting Chunks to Evolving Actions

**arXiv ID:** 2611.09672 | [PDF](https://arxiv.org/pdf/2611.09672v1)

**作者:** Qishen Wang `[一作]`, Cheng Chi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ODEAction，一种无分块、基于连续动作演化的视觉‑语言‑动作生成框架。

**💡 创新点**

创新点在于用动作演化场和常微分方程（ODE）取代传统的动作块预测，实现对动作连续性的显式建模和递归生成。

**🔧 技术方法**

采用动作演化场学习、示范诱导目标场构造、直接监督训练、ODE 数值积分（如前向欧拉/Runge‑Kutta）以及可插拔的动作头技术。

**📊 数据集**

使用 LIBERO、EBench、RoboTwin 三大仿真基准以及真实机器人（AgileX Cobot Magic 1.0）上的 Pick‑Place、Drawer、Cup‑Place、Spoon‑Insert 等任务。

**📈 对比分析**

与基于动作块的基础策略以及 RTC、Legato 等跨块连续性方法比较，ODEAction 在任务成功率、轨迹平滑度（NLDLJ、NSPARC、EE‑MAJ）和边界跳跃（Boundary Jump）上均实现显著提升，尤其在细致操控任务中提升幅度最大。

**⚠️ 局限性**

局限性包括对示范邻域尺度 σ 与修正项 k 的超参数调节依赖较大，且目前验证范围主要聚焦于连续动作空间，尚未在更大规模或离散动作任务中充分评估。

---

## 524. Capacity of Generic Symmetric Distributed Linear Computation: The Geometry of Relations

**arXiv ID:** 2611.09636 | [PDF](https://arxiv.org/pdf/2611.09636v1)

**作者:** Hua Sun `[一作]` `[通讯]` (University of North Texas), Hua Sun (University of North Texas)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了分布式线性计算中的用户通信需求，确定了在给定用户数量和数据维度下的通用容量C，即每个计算实例所需发送的最小字段符号总数。

**💡 创新点**

创新点在于通过几何关系空间的分析，揭示了用户子空间的平衡与分裂配置对容量曲线形状的影响，并提出了渐近集合对齐的编码方案。

**🔧 技术方法**

使用了渐近集合对齐技术，结合了单项预编码器来限制观察差异的组合，从而实现容量。

**📊 数据集**

使用了从有限域中独立均匀抽取的用户观察矩阵和需求矩阵，具体数据集未明确给出。

**📈 对比分析**

通过与现有方法的比较，本文展示了在不同用户数量和数据维度下的容量曲线，表明在平衡配置下，容量达到直接对齐的理论上限，而在分裂配置下，容量表现为分段线性。

**⚠️ 局限性**

限制在于对于用户数量K大于等于5时，容量的几何性质变得复杂，且在某些情况下可能无法完全实现理论容量。

---

## 525. Encoding Memory in World Models with Test-Time Model-Based Reinforcement Learning

**arXiv ID:** 2611.09625 | [PDF](https://arxiv.org/pdf/2611.09625v1)

**作者:** Maxime Burchi `[一作]` (University of Würzburg), Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了Test-Time Model-Based Reinforcement Learning（TT-MBRL），通过在测试时对世界模型参数进行梯度更新，将每个回合的记忆编码进模型，从而显著提升MBRL在长期记忆任务中的表现。

**💡 创新点**

突破传统通过改进模型结构来提升记忆的思路，改用测试时自适应更新模型参数，将记忆嵌入网络权重，实现每个回合独立的世界模型，形成“神经记忆”。

**🔧 技术方法**

采用Dreamer系列世界模型（RNN、Transformer、SSM），在测试时冻结编码/解码器、使用重放缓冲区混合训练与测试数据，并通过梯度下降对动态网络进行更新；同时利用视频预测指标评估模型。

**📊 数据集**

在Memory Maze 2D、Memory Maze 3D（9×9、11×11、13×13、15×15）以及通用基准DeepMind Control Suite和Atari 100k上进行实验。

**📈 对比分析**

与DreamerV3、TWISTER、R2I、IMPALA等基线在相同训练步数（100M/400M）下对比；TT-MBRL在大尺寸Maze上实现了从60.1到63.0、42.0到54.1的提升，接近人类基准；在DMC和Atari上保持或略优于基线。视频预测指标亦显示误差显著下降。

**⚠️ 局限性**

测试时的梯度更新导致每回合都需要重新学习世界模型，显著增加评估时间和计算开销；在需要大量交互的环境中可能不可行，且目前未在探索阶段使用，未来需提升效率。

---

## 526. CRTrack-Cap: A New Benchmark for Cross-View Dense Video Object Captioning

**arXiv ID:** 2611.09617 | [PDF](https://arxiv.org/pdf/2611.09617v1)

**作者:** Xiangqun Zhang `[一作]` (Tianjin University), Wei Feng `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对同步摄像机视角下的视频对象轨迹进行跨视角关联，并生成统一的自然语言描述，提出CVDVOC任务及CRTrack-Cap数据集。

**💡 创新点**

①首次构建跨摄像机密集视频对象描述任务与对应数据集；②设计视图独占关联（VECA）与证据门控融合（CEG）框架，实现在多视角下选择可信、互补的视觉证据进行描述生成。

**🔧 技术方法**

使用CRTracker+ReID实现局部轨迹检测与特征提取，VECA实现跨摄像机视图独占关联，CEG采用可学习门控机制对每个属性的视觉证据进行可靠性与冲突评估；随后通过CaptionFormer、Q-Former及冻结的OPT模型生成统一描述。

**📊 数据集**

CRTrack-Cap（425全局轨迹、838局部轨迹、335个可标注描述），该数据集基于CRTrack、DIVOTrack与CAMPUS多摄像机视频，提供同步视角、跨视角身份与参考描述。

**📈 对比分析**

与三种单视角基线（DenseVOC、SMOTer、CaptionFormer）及其通过MvMHAT实现的跨视角关联进行对比；在域内测试中，CV-CHOTA达83.43、CIDEr157.04；在跨域测试中，CV-CHOTA33.48、CIDEr112.62，均显著优于基线，显示出更好的全局跟踪与描述性能。

**⚠️ 局限性**

跨域性能仍明显低于域内，受摄像机条件差异影响；证据门控机制在复杂多摄像机冲突场景下仍有改进空间；数据集规模有限，缺乏更大范围的多摄像机场景和更丰富的属性标注。

---

## 527. A Separation Principle for Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.09644 | [PDF](https://arxiv.org/pdf/2611.09644v1)

**作者:** Lucia Pezzetti `[一作]` (ETH Zürich), Giorgia Ramponi `[通讯]` (University of Zürich)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出分离原理与SALT算法，将多智能体协作问题拆解为单智能体目标条件学习与最优运输分配，显著提升规模可扩展性。

**💡 创新点**

理论证明在解耦动力学与OT终端成本下的分离原则，并给出可实施的分离基准控制；将单智能体学习与OT协调相结合，支持训练后对不同车队规模的迁移。

**🔧 技术方法**

使用单智能体强化学习（如Q、DDQN、PPO、A2C）学习目标条件成本；利用最优运输（OT）进行目标分配；结合动态规划、证明分离原理以及经验评估。

**📊 数据集**

随机网格世界（11×30网格，7个代理）以及真实曼哈顿南部路网（2087交叉口、4255道路，OpenStreetMap），模拟时间变化、交通灯周期与随机拥堵。

**📈 对比分析**

与六种MARL基线（MAPPO、IPPO、HAPPO、VDN、QMIX、MFQ）以及静态最短路径启发式对比。SALT在网格世界显著优于MARL，PPO版在交互次数上比MAPPO少达480倍；在曼哈顿任务中相对最短路径提升15–21%，并在未训练的车队规模上保持性能。

**⚠️ 局限性**

受限于解耦动力学与OT终端成本的假设，不能处理强相互作用或碰撞；理论上上界不一定紧；单智能体目标条件学习难以泛化到高维或远离训练分布的目标；缺乏对多目标或多维目标场景的适配。

---

## 528. LACART: Lookahead-Augmented Collision Avoidance and Reference Tracking Strategies for Safety-Critical Robots

**arXiv ID:** 2611.09640 | [PDF](https://arxiv.org/pdf/2611.09640v1)

**作者:** Pedro Lopes `[一作]` (INESC TEC), Luis Tiago Paiva `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种结合lookahead、CLF–CBF–QP的安全过滤器，用于无人车的轨迹跟踪与障碍物避让，并在1:43比例迷你车上实现了硬件实验。

**💡 创新点**

创新点在于将lookahead点约束引入控制障碍函数、整合轨道边界与圆形障碍于同一QP层、提出车身几何与感知延迟下的安全保证条件，并证明相较先前工作计算时间更低。

**🔧 技术方法**

使用了基于lookahead的控制障碍函数、放松的控制李雅普诺夫函数、二次规划安全过滤、HSV图像分割、UDP/BLE通信、ESP32‑C3控制器等技术。

**📊 数据集**

主要数据来自仿真环境（模拟赛道与障碍）以及真实迷你车平台的摄像头标记定位数据；未使用公开数据集。

**📈 对比分析**

通过与Huang等人的实现对比；仿真结果显示路径相似，平均每步计算时间下降约6%，QP求解时间下降约8%；硬件实验在25 Hz循环下验证了实用性。

**⚠️ 局限性**

局限性包括感知与通信延迟导致状态估计老化、车身转向极限与延迟导致边界碰撞、对实时性要求高的计算资源与能耗、未验证动态障碍或多车协作情形。

---

## 529. What remains of the planar induced minor conjectures?

**arXiv ID:** 2611.09627 | [PDF](https://arxiv.org/pdf/2611.09627v1)

**作者:** Jadwiga Czyżewska `[一作]` (University of Warsaw), Paweł Rzążewski `[通讯]` (Warsaw University of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文通过构造新的困难实例（loom grids 与 triangle-free loom grids）和严格的理论证明，展示了在排除任意平面诱导子图的图类中，最大权重独立集、列表 3‑着色以及 3‑着色问题均为 NP‑难，即使在几乎多项式时间（quasi‑polynomial）范围内亦无法求解；并证明了与这些问题相关的结构猜想（常数级主导平衡分离子）与算法难度不相对应。

**💡 创新点**

创新点包括：① 引入新的困难实例族（loom grids 及其三角形自由变体）并证明它们排除多种简单平面树；② 构造了第二独立的困难实例族，进一步证明硬度独立于之前的构造；③ 证明 3‑着色在这些图类中同样 NP‑难，彻底否定了之前的算法猜想；④ 分析并证明在某些图类中仍能获得常数级主导平衡分离子，说明该结构性质并不能保证算法可解；⑤ 通过复杂的 Gyárfás 路径与重构技巧，给出了关于受限类结构的深层洞见。

**🔧 技术方法**

使用的技术主要是图论构造与归约技术，具体包括：诱导子图与诱导极小子图的定义与性质、looms grid 与 triangle‑free looms grid 的精细设计、对独立集与着色问题的多项式时间归约（从 Max Cut 与 3‑SAT），以及对受限类中平衡分离子与 Gyárfás 路径的深入分析；此外还利用了图分离、重合与子图模型化等经典工具。

**📊 数据集**

本研究不涉及实验数据或数据集，全部以理论证明和构造实例为主。

**📈 对比分析**

相较于先前的多项式/几乎多项式时间算法（如对路径、分支树或部分分支树的 QPTAS），本文提供了与之前猜想相对立的严格硬度证明；通过与已有结果对比，展示了新的困难实例的覆盖范围更广、结构更复杂，进一步证明了先前算法范围的局限性。

**⚠️ 局限性**

局限性与未解决的问题包括：① 对于哪些树形 H 仍可能保持多项式/几乎多项式可解性尚不清楚；② 是否存在更多能够支持常数级主导分离子的图类但仍使问题 NP‑难；③ 对于 3‑着色与最大独立集是否存在完全等价的可解性条件；④ 本文仅证明了 NP‑难性，未给出更精细的时间复杂度下限；⑤ 结构性质（如 Dominated Balanced Separators）与算法可解性之间的更一般性关系仍是开放问题。

---

## 530. TPIC: Energy-Conserving Space Plasma Simulations on GPU Supercomputers

**arXiv ID:** 2611.09624 | [PDF](https://arxiv.org/pdf/2611.09624v1)

**作者:** Luca Pennati `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在GPU超级计算机上实现了能量守恒半隐式粒子-电磁场（ECSIM）代码TPIC，用于大规模空间等离子体数值模拟。

**💡 创新点**

创新点：①将ECSIM能量守恒扩展到闭合导电/磁导边界并取消伪Poynting项；②在NVIDIA和AMD GPU上结合矩阵引擎（FP64 MMA/MFMA）和粒子排序实现极高的计算效率；③构建了可移植的LPFx库，统一管理异构计算、数据结构和多种通信后端；④实现了完整的GPU原生异步流水线，将粒子推进、迁移、当前/质量矩阵组装、场求解、诊断和I/O并行。

**🔧 技术方法**

使用的技术包括：ECSIM半隐式算法、矩阵引擎（FP64 MMA on NVIDIA、MFMA on AMD）、粒子到网格的三维三角云插值、混合精度存储/计算、LPFx（分布式内存数据结构、NCCL/RCCL/ MPI通信）、GPU原生流/事件异步管线、矩阵-自由GMRES求解、能量守恒平滑、可自定义的边界条件。

**📊 数据集**

验证数据集涵盖多种典型等离子体测试：冷漂移等离子体、两流不稳定、准中性等离子体膨胀、Whistler波、Weibel不稳定、Whistler湍流、三维磁重联（PEC墙）等，粒子数从10^7到1.23×10^10，网格尺寸高达1024^3，显著覆盖宏观到微观尺度。

**📈 对比分析**

与显式PIC、iPIC3D等传统代码在能量守恒、数值稳定性、线性/非线性增长率、解析理论的一致性进行对比。TPIC在所有测试中保持能量误差<10^-15（机精度），并在LUMI‑G上使用4,096个MI250x GPU实现1.9×10^12粒子/步，弱扩展效率96.8%，强扩展效率>95%至512 GPU，单步耗时约15 ms；相较于iPIC3D，TPIC在磁重联等3D模拟中实现约1.33倍的速度提升。

**⚠️ 局限性**

局限性：①ECSIM不完全保留电荷连续性，导致Gauss定律误差有限；②粒子排序仍占据一半计算时间；③目前仅实现闭合导电/磁导边界，开放边界和多物理耦合尚未完善；④混合精度下的数值误差、低精度矩阵引擎兼容性等需要进一步研究。

---

## 531. Online Timely Bin Packing Revisited: A Simpler Algorithm with Optimality in Special Cases

**arXiv ID:** 2611.09611 | [PDF](https://arxiv.org/pdf/2611.09611v1)

**作者:** Pei-Lin Chiang `[一作]` (National Tsing Hua University), Li-Tzu Yang `[通讯]` (National Tsing Hua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的确定性在线算法 Thrifty-Greedy，用于解决具有截止时间约束的在线装箱问题。

**💡 创新点**

创新点在于：①不需要模拟分数化装箱，算法实现更简单；②在通用输入下实现了已知最优的 2 竞争比；③在所有项目同一尺寸以及仅包含 1/3 与 2/3 两种尺寸的特殊输入下，分别实现了 1 与 1.5 的最佳竞争比。

**🔧 技术方法**

技术主要包括：基于“触发项”扫描的贪心装箱策略、强归纳证明、输入转换技术（保持相同项目集合但收窄时间区间以便比较）以及竞争比分析。

**📊 数据集**

本文没有使用实验数据集，全部结果均为理论分析与证明。

**📈 对比分析**

与先前的 Discrete‑Greedy 算法相比，Thrifty‑Greedy 在通用输入上保持相同的 2 竞争比；在特殊输入上则明显优于 Discrete‑Greedy（后者在同类输入上至少为 2）。

**⚠️ 局限性**

限制与开放问题：①当前确定性算法的最优竞争比仍处于 1.5 与 2 之间；②随机化算法的潜在改进尚未确定；③对存储容量有限或持有成本的在线装箱模型尚未深入研究。

---

## 532. Empirical Assessment of Post-Quantum Cryptography Adoption Status in TLS

**arXiv ID:** 2611.09707 | [PDF](https://arxiv.org/pdf/2611.09707v1)

**作者:** Md Ariful Haque `[一作]` (Bitsight Technologies), João Cruz `[通讯]` (Bitsight Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对七个常用TLS服务（HTTPS、LDAPS、SMTPS、IMAP-SSL、POP3-SSL、TELNETS、RDP）进行大规模实验扫描，评估其在主机、实体和域级别的后量子密码（PQC）接入情况，并关联组织规模、行业与数字足迹。

**💡 创新点**

首次在全行业层面以大规模IP扫描为基础，系统性比较不同服务、不同实体规模以及行业细分对PQC的采用差异；同时对Cloudflare托管域与IP的PQC率做横向验证，构建多维度的实证基准。

**🔧 技术方法**

使用GROMA自研互联网扫描平台，利用 OpenSSL v3.5 对TLS1.3握手进行指纹识别，配合自定义正则提取KEX算法；使用AI驱动的资产映射引擎 GIA 将IP映射至真实组织；并结合内部数据管道提取财务与数字指标。

**📊 数据集**

10,000个目标IP/端口的扫描结果、OpenSSL握手日志、企业财务与员工数、IPv4/SSL_ACTIVE_Count指标、Cloudflare托管域与IP的扫描样本。

**📈 对比分析**

通过按协议、主机/实体/域层面计算PQC占比，绘制柱状图、频率直方图、箱线图和散点图；发现HTTPS主机层面PQC率约9%，其他协议低于5%；Cloudflare域级PQC率高达83%但IP级仅4%。

**⚠️ 局限性**

样本限制在10k IP/端口、IPv4、TLS1.3；扫描仅覆盖可达服务，可能漏掉隐藏或非标准端口；仅识别 X25519MLKEM768 等已知PQC算法，未覆盖所有后量子方案；缺乏时间序列动态分析，不能体现迁移进度。

---

## 533. The Multiscale Single-Index Model: An MSE Analysis of Online SGD Dynamics

**arXiv ID:** 2611.09682 | [PDF](https://arxiv.org/pdf/2611.09682v1)

**作者:** Gordon Dai `[一作]` (New York University), Joan Bruna `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了多尺度单指数模型（K=2）下的在线随机梯度下降（SGD）在均方误差（MSE）损失下的学习能力，并给出了弱恢复和强恢复的样本复杂度上界。

**💡 创新点**

首次在深层单指数模型中使用MSE目标分析SGD收敛，结合高维Gaussian比较（Stein方法）与三阶矩控制，解决了非高斯特征导致的技术难题。

**🔧 技术方法**

采用球面SGD、Gaussian比较与Stein/ Lindeberg换位、轨迹层级的漂移-波动分析、连续-离散时间耦合、局部强凸性证明和Riemannian梯度流分析。

**📊 数据集**

使用合成的多尺度高斯张量输入，未使用真实数据集。

**📈 对比分析**

与谱方法在样本复杂度上进行理论比较，证明SGD在MSE下达到与谱方法相差仅对数因子、近似最优的性能；实验验证仅为数值模拟，未给出实际数据实验。

**⚠️ 局限性**

局限性包括仅针对K=2、要求高斯输入与特征非局部化及非退化激活；无法推广到更深网络或无尺度分离的实测数据；在更高层深度时MSE目标难以收敛。

---

## 534. Calibrating demand models versus adjusting OD matrices with traffic counts: a controlled comparison

**arXiv ID:** 2611.09678 | [PDF](https://arxiv.org/pdf/2611.09678v1)

**作者:** Yue Li `[一作]`, Ying Jin `[通讯]` (University of Cambridge)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较三种OD估计方法（五系数生产-重力模型、OD因子调整、单元格调整）在基于交通计数的校准中对OD矩阵恢复与预测性能的差异。

**💡 创新点**

在同一套先验、计数和网络条件下，用合成真实OD进行受控对比，系统评估模型参数化对恢复准确性和计数拟合的影响。

**🔧 技术方法**

采用确定性用户均衡分配、冻结路线敏感度、Levenberg–Marquardt/Gauss–Newton和L-BFGS-B优化，交叉验证选择正则化权重，计算共同行程比例（CPC）和保留链接NRMSE。

**📊 数据集**

使用交通网络研究集合中的Eastern Massachusetts和Chicago Sketch网络，生成合成旅行需求并使用随机选取的计数样本。

**📈 对比分析**

通过世界级bootstrap 10,000次对比，报告平均配对差异与95%区间；在匹配和两目的需求下，五系数模型在OD恢复和保留流预测上优于两种矩阵调整；但在目的地选择或生产误设时，灵活调整在部分条件下恢复更好。

**⚠️ 局限性**

仅在合成数据和理想化的计数/分配假设下验证；真实交通行为更复杂，计数覆盖和测量误差不同；冻结路由导数近似和仅使用有限计数可能限制可推广性。

---

## 535. Online Causal Configuration for Networked Systems via Doubly Robust Steady-State Learning

**arXiv ID:** 2611.09665 | [PDF](https://arxiv.org/pdf/2611.09665v1)

**作者:** Yuli Liu `[一作]` (University of Macau), Zhiheng Zhang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于因果评估的网络配置选择算法 CAUSAL-UCB，利用曝光感知的裁剪双重稳健评估器并提供预决策证书，以在存在网络干扰和探索偏差的情况下估计无探索稳态价值并做在线决策。

**💡 创新点**

创新点在于将曝光接口、行动-曝光联合分布、上下文迁移与采样波动分离，形成统一的可校准错误预算和置信证书；并通过理论证明了识别-部署折衷下的下界以及在证书有效且信息量增长的情形下的 regret 上界。

**🔧 技术方法**

主要技术包括曝光感知裁剪双重稳健估计、暴露映射校准、基于有效样本大小和自相关的置信区间、OFU 与优化置信上界的结合，以及针对已知本地干扰的 anytime 置信半径。

**📊 数据集**

实验使用了24节点环形网络（高低匹配版本）、METR‑LA 与 FlockLab 两套真实交通网络拓扑和上下文轨迹，并通过 300 种随机种子进行全路径证书覆盖率检验。

**📈 对比分析**

与 IPS‑UCB、传统 DR‑UCB、IPS‑UCB、原始 UCB、贪婪与 Switchback‑OED 等基线相比，CAUSAL‑UCB 在高匹配偏差场景下实现了最低的累计目标 regret（1.732 vs 6.326）且无终端 gap，但相对较高的部署 gap；在低匹配偏差或真实轨迹场景中，其优势不如贪婪策略。

**⚠️ 局限性**

局限性包括：需要满足严格的可识别假设（联合分布一致性、曝光-响应一致性、上下文迁移可校准等）；当本地干扰未知或暴露映射不足时，证书可能失效；并且对真实部署的长期验证仍缺乏，且方法在高探索率下部署成本较高。

---

## 536. TensorMutation: Learning Weight-Space Mutations of Deep Learning Models on Neural Graphs

**arXiv ID:** 2611.09650 | [PDF](https://arxiv.org/pdf/2611.09650v1)

**作者:** Zaheed Ahmed `[一作]` (University of Göttingen), Jens Grabowski `[通讯]` (University of Göttingen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 TensorMutation，一种基于图神经网络的后训练权重空间变换方法，能够在不重新训练模型的情况下预测并生成与预训练突变相同的变异体。

**💡 创新点**

创新点在于：①把预训练突变的参数变更建模为可学习的变换，②使用全局 GNN+FiLM 条件化预测权重/偏置差异，③在未见模型上实现跨模型迁移，④通过行为与权重空间联合训练提升方向一致性。

**🔧 技术方法**

技术手段包括：图神经网络（message‑passing）、FiLM 条件化、Kullback–Leibler 交叉熵行为损失、余弦相似度权重空间损失、节点/边特征编码、实验评估指标 IoU、Coupling Strength、cosine similarity、L2 误差等。

**📊 数据集**

使用数据集：71 个 TensorFlow/Keras MLP 分类器，构成 25,252 对原始‑变异样本；12 个行为活跃的真实缺陷样本用于真实性评估；每个样本包含多次随机种子训练实例。

**📈 对比分析**

与 DeepCrime（预训练）、DeepMutation++（后训练手工变异）和 norm‑matched 随机方向控制进行对比。结果显示 TensorMutation 在保持行为质量上比 DeepMutation++ 高 70%（p=0.018），且与 DeepCrime 的 IoU 较高但差异不显著；随机控制表明方向信息提升有限，且对行为相似度的贡献未必统计显著。

**⚠️ 局限性**

局限性包括：①方向提升未能在所有评估中显著提升行为相似度，②参数幅度未完全校准，导致行为严重性不一致；③仅评估 MLP 网络与 11 类预训练突变，无法推广至其它网络结构或缺陷类型；④样本量有限，统计功效受限；⑤未深入探讨与真实缺陷的根因对应性。

---

## 537. Fast and Actionable Classification of HPC I/O Time Series

**arXiv ID:** 2611.09646 | [PDF](https://arxiv.org/pdf/2611.09646v1)

**作者:** Maxime Gonthier `[一作]` (Inria), Francieli Boito `[通讯]` (University of Bordeaux)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出基于聚类的I/O时间序列编码与分类方法SAGE，并将其用于HPC作业的临时I/O行为识别与调度。

**💡 创新点**

创新点包括：从短窗口学习形状与幅度字母表，将完整轨迹压缩为“基因组”；采用分层形状/幅度聚类和相似度向量；提出Phase-Level Distance度量；实现显著加速的聚类并提升调度效果。

**🔧 技术方法**

使用技术包括：窗口形状归一化与差分提取、K‑means与elbow搜索、DTW对比基线、相似度向量、欧氏距离、SimGrid仿真调度、基于重叠度的贪心+局部搜索。

**📊 数据集**

数据集来自Inria等HPC系统的真实I/O轨迹，涵盖约16,000+读写两类记录（已扩展至更多年份）。

**📈 对比分析**

通过聚合分数、聚类时间、I/O伸展等多指标与DTW+K‑means及其他state‑of‑art方法比较；SAGE在聚类质量上多案例位居前列，聚类速度比最快基线快109倍；在OST调度实验中相较轮询降低I/O伸展至约24%。

**⚠️ 局限性**

局限性包括：需手动调参、对周期性行为的频率信息损失、仅在仿真中验证、迁移到其他系统/信号需调整参数、未实现类预测。

---

## 538. Entering a New Domain in the Era of Generative AI: An Autoethnography of Solo Game Development as a Bioinformatics Educator

**arXiv ID:** 2611.09623 | [PDF](https://arxiv.org/pdf/2611.09623v1)

**作者:** Hannah Kim `[一作]` `[通讯]`, Hannah Kim

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过自我民族志方法记录一名生物信息学教育工作者利用生成式人工智能开发教育型“自选冒险”游戏的过程与身份形成。

**💡 创新点**

将生成式AI赋能与社区实践相结合，探究跨域身份构建，并系统呈现生成式AI对教育游戏开发与实践认同的影响。

**🔧 技术方法**

使用生成式AI（ChatGPT、Claude、Anthropic Fable 5、Adobe Firefly）、游戏开发工具（Figma、CVAT、Cursor）以及多平台社交媒体（Bluesky、Reddit、itch.io等）。

**📊 数据集**

主要基于自我记录的日志与10张示意图，并未引用公开大规模数据集。

**📈 对比分析**

采用反思性主题分析与情境比较，未进行客观性能度量，仅以自评和社区反馈评估生成式AI在故事创作和原型制作中的有效性。

**⚠️ 局限性**

数据收集时间有限、亲身参与社区机会不足、仅依赖自我日志与有限图像，缺乏外部验证和量化实验。

---

## 539. On Positive Computation

**arXiv ID:** 2611.09622 | [PDF](https://arxiv.org/pdf/2611.09622v1)

**作者:** Eduardo Skapinakis `[一作]` (University of Tübingen), Eduardo Skapinakis `[通讯]` (NOVA FCT)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了正向计算（positive computation）框架下的可归约性与复杂性类，提出新的正向归约概念，并证明了正向电路可满足性和哈密顿环问题在 mNP 下的完备性；进一步探讨了 posP 与 mNP 的分离不随相对化而保持，并给出了弱对角化证明 posP≠mNP。

**💡 创新点**

创新点在于：①引入正向多项式时间可归约（positive FP many‑to‑one 和 positive Turing 可归约）并以此证明 mNP 的完备性；②展示 posP⊊mNP 的分离不相对化，提供了一个弱对角化方法；③通过正向模型构造相对化正向图灵机，证明存在单调 oracle 使 posP^A = mNP^A。

**🔧 技术方法**

使用技术包括：正向图灵机与正向电路模型、可预测函数与 ♢ 操作、正向量化的量化表述、正向多项式时间可归约构造、Tseitin 变换、图形构造、以及对角化与弱对角化证明。

**📊 数据集**

本文为理论工作，无实验数据集，全部为形式化证明与归约构造。

**📈 对比分析**

比较方法基于理论归约与完备性证明，未进行实验性能评估。

**⚠️ 局限性**

限制：仅针对 NP 与 PSPACE 的正向类；对更低阶类（如 mP）的完备性讨论有限；正向归约与通用归约之间的差异尚未完全阐明；对角化结果为弱对角化，未提供完整非相对化证明。

---

## 540. Know What You Need: Efficient Activation Checkpointing

**arXiv ID:** 2611.09614 | [PDF](https://arxiv.org/pdf/2611.09614v1)

**作者:** Danyal Ahmed `[一作]` (Sorbonne University), Edouard Oyallon `[通讯]` (CNRS, Sorbonne University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于前向推理信息的近似重计算方法（KEAP），在激活检查点重计算时仅重算重要层组件，从而在保持内存优势的同时显著降低计算成本

**💡 创新点**

创新点在于将前向正向信息与随机/贪婪/Top‑k掩码相结合，对可分离线性层进行子集重算，并通过激活检查点保证梯度无偏，避免了传统近似导致的误差累积

**🔧 技术方法**

采用激活检查点（Activation Checkpointing）、随机/最优/Top‑k sketching、贪婪选择、无偏估计与微分自动微分等技术

**📊 数据集**

在 Llama‑style 模型（134M 与 488M 参数）上使用 FineWeb 训练集进行实验

**📈 对比分析**

与完整激活检查点和按 FLOP 匹配的更小模型对比，KEAP 在保持相同内存占用的前提下，困惑度提升仅约1–2.5%，且在 token‑级别粒度下优于 FLOP‑匹配的小模型

**⚠️ 局限性**

限制包括批量级别粒度效果差，注意力层的近似误差更大，对硬件加速实现的细粒度优化仍需进一步研究

---

## 541. DKWBC: Decoupling Motion Tracking and Balancing for Humanoid Whole-Body Control

**arXiv ID:** 2611.09605 | [PDF](https://arxiv.org/pdf/2611.09605v1)

**作者:** Dingkun Wei `[一作]` (Zhejiang University), Xiaowei Zhou `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种分离模型基础运动跟踪与学习平衡控制的全身控制框架；

**💡 创新点**

创新点在于将运动跟踪与平衡分离：使用逆动力学与 PD 反馈提供轨迹跟踪，残差策略仅学习平衡补偿，从而提升泛化和执行精度；

**🔧 技术方法**

采用逆动力学求解、关节 PD 反馈、强化学习残差策略、机器人动力学模型以及环境力反馈观测；

**📊 数据集**

主要使用 BONES-SEED 运动数据集，并在 G1 机器人上进行实验；

**📈 对比分析**

与 SONIC 等基线对比，在不同训练数据规模下取得更高成功率、更低 MPJPE，且在附加负载和外部扰动下保持优秀性能；

**⚠️ 局限性**

局限性包括：残差策略仍需在多任务场景下验证，模型基础轨迹的动态可行性依赖于精确模型，且对极端高速/大幅动态的适应性未充分评估。

---

## 542. A Fast and Accurate Algorithm for the $p \to q$ Norm of Sparse Non-Negative Matrices

**arXiv ID:** 2611.09601 | [PDF](https://arxiv.org/pdf/2611.09601v1)

**作者:** Marco Pirazzini `[一作]` (Yale University), Daniel A. Spielman `[通讯]` (Yale University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种算法，该算法能够在时间复杂度为 O(s^{3/2} log(1/ε)) 的情况下，计算具有 s 个非零条目的非负矩阵的 p → q 范数，精度为 ε，适用于任意 p ≥ q ≥ 1。

**💡 创新点**

该算法的创新点在于将拉普拉斯求解器应用于内部点方法（IPM），并将问题形式化为一个线性优化问题，展示了其在高精度解决方案中的优越性。

**🔧 技术方法**

使用了内部点方法（IPM）和拉普拉斯求解器的结合，利用了 Schur 余子式的性质来加速计算。

**📊 数据集**

使用了具有 s 个非零条目的非负矩阵作为数据集，具体数据集未详细说明。

**📈 对比分析**

与之前的算法（如 Objois 和 Vladu 的算法）相比，该算法在高精度情况下表现更优，尤其在 s < n^2 时，运行时间更快，且在 ε < 1/√(s) 时优于 Objois 和 Vladu 的算法。

**⚠️ 局限性**

该算法的局限性在于其依赖于输入矩阵的非零结构，且在处理非常稀疏或特定结构的矩阵时可能不如其他专门算法高效。

---

## 543. Empirical Study of Post-Quantum Cryptography Adoption Status in SSH Infrastructure

**arXiv ID:** 2611.09703 | [PDF](https://arxiv.org/pdf/2611.09703v1)

**作者:** Md Ariful Haque `[一作]` (Bitsight Technologies), João Cruz `[通讯]` (Bitsight Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模互联网扫描，实证分析了SSH协议中后量子密码（PQC）的采用情况，涵盖了1.52百万主机和27,012个实体的主机级与实体级支持率；

**💡 创新点**

创新点在于首次系统性、基于真实数据的SSH PQC采用评估，并对行业、地域、组织规模等维度进行细致对比；

**🔧 技术方法**

使用了自研的GROMA扫描平台、AI驱动的资产映射引擎GIA以及正则表达式解析，结合统计分析方法；

**📊 数据集**

采用的主要数据集为GROMA在2026年2月16日的全网SSH扫描结果，包含1,523,998个唯一IP及其KEX算法列表；

**📈 对比分析**

通过计算主机级PQC支持比例（45.2%）与实体级支持比例（39.9%），并绘制行业、地区和规模分布图，展示了采用率与数字足迹的关联；

**⚠️ 局限性**

局限性包括仅关注22端口、单日扫描导致时间视角单一、可能遗漏隐藏或私有SSH服务器、以及未评估PQC算法的实际性能与安全性。

---

## 544. Attend at the Right Resolution: Head-Adaptive Sparse Attention via Query-Key Interaction Structure

**arXiv ID:** 2611.09700 | [PDF](https://arxiv.org/pdf/2611.09700v1)

**作者:** Wajih Hassan Raza `[一作]` (University of Houston), Yu Wen `[通讯]` (University of Houston)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的稀疏注意力框架HASA，通过预训练的查询-键双线性运算的谱结构自适应选择特征维度和序列粒度，以高效捕获长文本上下文。

**💡 创新点**

将每个注意头的预训练查询-键相互作用谱作为自适应特征维度和区域粒度的依据，实现头级别的多维自适应稀疏。

**🔧 技术方法**

谱分解（奇异值分解）、谱熵与残差阈值确定有效秩、方向一致性度量与逐步合并构建候选区域、训练免费稀疏注意力选择与完整键值计算等技术。

**📊 数据集**

在LongBench-Pro和RULER-32K两大长文本基准上，结合Ministral‑3‑8B‑Instruct‑2512和Qwen3‑8B模型进行评估。

**📈 对比分析**

按相同KV预算与Dense、H2O、Quest、InfLLM、UNIQUE等无训练稀疏基线对比，HASA在KV 512–2048区间平均提升2–3点准确率，稠密注意力召回88.1%，相较固定配置提升7.4个百分点；在8K–32K上下文中比稠密加速6–14×，比UNIQUE提升约1.3×。

**⚠️ 局限性**

仅利用预训练双线性相互作用的输入不相关谱结构，忽略softmax、value表示及运行时敏感性，输入相关信息需进一步补偿。

---

## 545. SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers

**arXiv ID:** 2611.09680 | [PDF](https://arxiv.org/pdf/2611.09680v1)

**作者:** Thibault Formal `[一作]` (Linkup), Cosimo Rulli `[通讯]` (Linkup)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套训练稀疏检索器（Splade‑style）在现代大模型骨干上的完整流程，并通过实验验证其在大规模语料上的有效性。

**💡 创新点**

创新点包括：① 对 logits 进行全局偏移（logit shifting）以解决右移 Gaussian 分布导致的梯度稀疏问题；② 通过“fan‑out”机制限制每个输入位置可激活的词表条目数，从而更好地控制稀疏度；③ 在输出空间进行词表折叠（vocabulary folding），将 byte‑level BPE 的多重表面形式合并为统一标识，减少冗余并提升匹配质量。

**🔧 技术方法**

采用了多种技术：预训练的 Transformer 骨干（BERT‑Base、BERT‑Large、RoBERTa‑LARGE、以及更现代的 LightOn 等），MLM 头的微调，InfoNCE 对比学习，FLOPS 正则化，Logit shifting（APC），fan‑out 约束，词表折叠，top‑k 推断裁剪，以及可选的交叉编码器蒸馏。

**📊 数据集**

主要使用的公开数据集包括 BEIR‑13、其去污染版本、以及对齐、法律/财经、化学等专业领域的 CURE‑en、Legal/Finance、ChemTEB、BRIGHT‑Pro 等跨域检索基准。

**📈 对比分析**

与当前最强的稀疏检索器（如 VT、OS‑NSE、SlideOn）以及密集模型（DenseOn、LateOn）进行对比。模型在 BEIR‑13 上达到 56.4 nDCG@10，单单 150 M 参数即可与 DenseOn 相差不到 1.5 分，明显优于多数稀疏方法；在去污染集上进一步缩小差距；在多种跨域评测中表现稳健，尤其在 BRIGHT‑Pro、FreshStack、Legal/Finance、ChemTEB 等任务中取得领先或相当水平。

**⚠️ 局限性**

局限性包括：① 仍在某些数据集（如 LoTTE）上落后于密集和晚期交互模型；② 对 θ（logit shift）和 fan‑out 参数的选择需要经验性调优；③ 词表折叠虽然降低了维度，但在极端分词细粒度环境下可能仍出现“wacky weights”；④ 对大规模稀疏模型的推理与部署仍需额外的压缩与加速技术。

---

## 546. Dual Architecture Fusion Vehicle Lidar 3D Target Detection

**arXiv ID:** 2611.09664 | [PDF](https://arxiv.org/pdf/2611.09664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 547. Autonomous Particle Image Velocimetry Using Zero-Shot Sim-to-Real Reinforcement Learning

**arXiv ID:** 2611.09679 | [PDF](https://arxiv.org/pdf/2611.09679v1)

**作者:** Chang Rajani `[一作]` (Swiss Federal Laboratories for Materials Science and Technology), Ivan Lunati `[通讯]` (Swiss Federal Laboratories for Materials Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一套基于机器人臂摄像头的自主粒子图像测速系统，能够在水槽中自动定位、框定并高分辨率捕捉水翼尾流；

**💡 创新点**

将相机视角定位问题建模为序列决策任务，利用强化学习在仿真中学习并实现零样本迁移到真实硬件；

**🔧 技术方法**

采用DreamerV3模型基RL、LIMA实时PIV估计器、NVIDIA Isaac Sim与Isaac Lab进行高保真渲染，并配合CFD生成的尾流场景；

**📊 数据集**

训练集包含719个案例，涵盖372个不同位置、攻角、截面（NACA 0012/0024/6412）及两种自由流速度的二维Lattice‑Boltzmann尾流数据；

**📈 对比分析**

在真实水槽中与固定宽视角及全景拼接基线对比，政策实现了近乎100%的完整尾流捕捉，测量误差下降约一半，测量时间和图像对数均缩减约一半；

**⚠️ 局限性**

当前方法未能控制曝光、间隔时间、光圈等成像参数，对三维/多尺度流场的适用性有限，且对仿真与真实差异高度依赖精准的光学与几何校准。

---

## 548. How Much Conversation History Does an AI Agent Need?

**arXiv ID:** 2611.09662 | [PDF](https://arxiv.org/pdf/2611.09662v1)

**作者:** Danny Kim `[一作]` (Pembroke Pines Charter High School), Fernando Koch `[通讯]` (Florida Atlantic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在固定检索与阅读管道中，通过改变检索到的段落数量来评估阅读器的 token 使用和分数变化，并检验消息保留策略对答案所需证据的保留效果。

**💡 创新点**

提出一个实验框架系统衡量不同段落预算对 token 成本与评估分数的权衡，并对保留消息策略进行对照实验，同时对分数的可靠性进行审计。

**🔧 技术方法**

采用 BM25 段落检索、Qwen2.5-14B 与 Mistral-7B 阅读器、基于 BEAM 的人工评判仪表以及自动化审计技术。

**📊 数据集**

使用 BEAM 数据集中的 1,100 个问题（来自 100K 与 500K 语料）以及 128 条基于保留消息的实验轨迹。

**📈 对比分析**

通过固定检索排名对比 1/3/5/10/20 段落预算的 token 总量和分数变化，发现从 10 段降至 3 段可节省约 68% token，平均分数下降 5.45 分；不同问题类型表现不一，保留策略对 Qwen 的直接需求有效，但对引用需求效果有限。

**⚠️ 局限性**

结果仅基于单一阅读器与检索器，使用合成数据且分数未得到独立源可靠性验证，对 Mistral 的行为缺乏解释，限制了结论的普适性。

---

## 549. Concurrent Use of FP32 and FP64 Computing Units on GPUs and Mixed-Precision Placement in WENO Reconstruction

**arXiv ID:** 2611.09651 | [PDF](https://arxiv.org/pdf/2611.09651v1)

**作者:** Jun Hatayama `[一作]` (Okayama University), Toshinori Kouchi `[通讯]` (Okayama University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在 NVIDIA GPU 上通过并行执行 FP32 与 FP64 指令来利用空闲 FP64 单元，并在 WENO‑Z 重构核中测试混合精度与双浮点（DF）仿真方法。

**💡 创新点**

提出在同一 SM 上同时发射不同精度指令可恢复闲置 FP64 资源，并揭示该技术在不同 GPU 架构（A100、GH200/GB200、RTX 6000 Blackwell）上的可行性与局限。

**🔧 技术方法**

使用微基准（FFMA/DFMA）和 WENO‑Z 重构核；采用 DF 仿真（双浮点）与混合精度（W32P64）实现；对比 FP64、FP32、DF 以及混合精度放置。

**📊 数据集**

在一维熵波与二维 Euler 旋涡两种平滑流场上评估精度与收敛性；使用多阶（WENO5/7/9）重构和 RK4 时间积分。

**📈 对比分析**

对 FP64、W32P64、DF 进行跑时和吞吐量比较：A100 上 FP32+FP64 并行可达约 90% 纯 FP32/FP64 吞吐，WENO‑Z 在 A100 上实现 E≈0.9 的重叠；在 RTX 6000 Blackwell 上 DF 仿真可获得 2.5–3.9× 的 FP64 加速；但 W32P64 在高阶重构下失去设计收敛阶，导致误差显著增大。

**⚠️ 局限性**

局限包括：在 GH200/GB200 上几乎无重叠；在 RTX 6000 Blackwell 上重叠受寄存器占用与可选 warp 数限制；W32P64 在高阶重构时精度严重下降；方法对不同 GPU 架构依赖性强，无法统一迁移；缺乏对包含激波的流动和更高阶统计量的评估。

---

## 550. Differentially Private GraphRAG: Protecting Relation Connectivity and Semantics

**arXiv ID:** 2611.09658 | [PDF](https://arxiv.org/pdf/2611.09658v1)

**作者:** Jie Fu `[一作]` (Stevens Institute of Technology), Wendy Hui Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于差分隐私的 GraphRAG 保护框架，利用关系差分隐私 (relation-DP) 对实体关系的语义与连通性进行隐私保护，并在索引前对文档进行消毒。

**💡 创新点**

创新点在于：①首次为 GraphRAG 引入关系差分隐私的理论框架；②结合关系替换与边删除，既降低语义泄露又减小连通性泄露；③在单一机制中同时保证关系语义与连通性两维隐私。

**🔧 技术方法**

采用的技术包括：关系语义距离的嵌入度量、指数机制（exponential mechanism）进行关系替换与删除、LLM 自动抽取三元组与候选关系、文档级别的关系替换一致性重构。

**📊 数据集**

使用的公开数据集为 HotpotQA 和 2WikiMultiHopQA，分别作为保护语料与 QA 评估集。

**📈 对比分析**

与四种现有防御（Abstractive、ID Alignment、Decoy、Protective Prompt）以及无防御基线进行对比。实验显示其在 GRASP 与 GraphSteal 两种攻击下，关系恢复 F1 下降 27–44 百分点，连通性恢复 F1 也显著低于其它方法；在 QA 质量上损失仅为 2.98–8.97% 的 F1。

**⚠️ 局限性**

局限性包括：在某些系统（如 HippoRAG 2）下 QA 下降更大；需手动调节隐私预算 ϵ、删除阈值 λ 与候选关系数 K；仅关注关系隐私，对实体信息或多步推理未作专门保护；攻击模型仅为黑盒查询，未考虑更强的侧信道或模型内部信息泄露。

---

## 551. From Function Inversion to Preimage Reporting

**arXiv ID:** 2611.09656 | [PDF](https://arxiv.org/pdf/2611.09656v1)

**作者:** Mathieu Vannimmen `[一作]` `[通讯]` (Universite Libre De Bruxelles), Mathieu Vannimmen (Universite Libre De Bruxelles)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了将任意函数的单一预像查找（function inversion）方案转化为报告所有预像（preimage reporting）的通用方法，给出了空间-时间权衡为 TS² = N²Δ_f 的数据结构。

**💡 创新点**

创新点在于：① 提供了一个黑盒转换，将任何已有的函数反转方案直接映射到预像报告方案，保持相同的权衡；② 通过 k‑wise 独立哈希分区和 Chernoff 边界实现预像数量与查询时间的反比；③ 该方法无需深入了解底层反转算法，可广泛应用于 kSUM、Jumbled 等报告问题。

**🔧 技术方法**

主要技术包括：k‑wise 独立哈希函数构造、哈希分区与子函数构造、Chernoff 与 Hoeffding 边界对随机变量的分析、以及对现有 Fiat‑Naor Δ‑variant 的黑盒调用。

**📊 数据集**

论文未使用具体实验数据集，全部以理论构造和抽象模型（如 [N]→[M] 的函数、kSUM 组合函数、Jumbled 字符串哈希）为基础。

**📈 对比分析**

与已有预像报告方案比较：在任意空间 S 满足 TS² = N²Δ_f 时，本文方案至少不逊于现有方案；相较原 Fiat‑Naor，查询时间可按 1/|f⁻¹(y)| 缩短，空间保持一致。对 kSUM 与 Jumbled 报告问题，本文在空间-时间曲线上实现了更优的 (nᵏ⁻¹ – δ/2) 空间与 Δ n^δ 查询时间，优于之前的 (nᵏ⁻¹ – δ/3) 空间。

**⚠️ 局限性**

局限性：① 依赖随机哈希，成功概率为高概率而非确定；② 当最大入度 Δ_f 较大时，方案退化为暴力搜索，性能不再优越；③ 只适用于静态离散函数映射，未覆盖动态或流式场景；④ 需要先构造完整的函数映射，实际应用时受限于输入规模与预处理成本。

---

## 552. Decoupled Mode Connectivity for Base-to-Novel Generalization in Vision-Language Models

**arXiv ID:** 2611.09655 | [PDF](https://arxiv.org/pdf/2611.09655v1)

**作者:** Imad Eddine Marouf `[一作]` (Institut Polytechnique de Paris), Stéphane Lathuilière `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种新的提示学习框架DMC（Decoupled Mode Connectivity），通过为基础类别专化与新类别泛化分别训练两个提示并在文本特征空间中插值，解决单提示方法中两种目标梯度冲突的问题。

**💡 创新点**

创新点在于：①将两个对立目标分离到不同的提示上，避免共享参数的梯度冲突；②通过线性模式连通（LMC）在文本特征空间构造低损失插值通道；③设计了类置换不变（CPI）的视觉锚点（Visual Anchor）正则化，以保持CLIP零样本语义分布；④在理论上证明CPI违背会导致新类别准确率下降。

**🔧 技术方法**

使用的技术包括：CLIP视觉语言模型的提示学习、余弦正则化、线性模式连通（LMC）路径、视觉锚点KL正则、交叉熵损失、参数化的α插值等；训练过程中只需学习两个约4K参数的提示。

**📊 数据集**

实验使用了11个基准数据集：ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101。

**📈 对比分析**

与CoOp、KgCoOp、MMA等提示学习基线以及TIES、DARE等模型合并方法进行比较。DMC在CoOp上平均提升了0.78的调和平均（HM），在KgCoOp上提升0.33，MMA上提升0.58，显著超越单提示正则化（如余弦正则）以及训练无关的合并方法。

**⚠️ 局限性**

局限性包括：需额外学习一个提示（约2K参数）；插值系数α需离线选取且对不同数据集需固定，无法自适应；在新类别已接近上限或MMA等适配器方法中，两端提示趋同，DMC收益不明显；CPI与信息理论证明基于理想化假设，实际场景可能不完全成立。

---

## 553. Breaking the Privacy Promise: Gradient Inversion Attacks on Federated Low-Rank Adapted Vision Transformers

**arXiv ID:** 2611.09652 | [PDF](https://arxiv.org/pdf/2611.09652v1)

**作者:** Ritik K. Badiya `[一作]` (Indian Institute of Science), Anirban Chakraborty `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对联邦学习中使用LoRA参数高效微调的视觉Transformer模型，研究梯度逆向攻击并提出一种空间锚定的生成式逆向框架，利用可训练的位置信息提升隐私泄露风险。

**💡 创新点**

创新点在于将可训练的层级位置信息嵌入ViT中，形成梯度空间的“空间锚点”，与生成式先验相结合，显著降低LoRA导致的梯度模糊与位置歧义，提升逆向重建精度。

**🔧 技术方法**

技术方法包括：LoRA微调、Vision Transformer (ViT)架构、生成式梯度逆向 (GIAS) 与DCGAN生成器、层级可训练位置编码、梯度匹配的余弦相似度损失、FedSGD/FedAvg联邦训练与梯度裁剪/高斯噪声防御。

**📊 数据集**

实验使用了CIFAR-10（32×32）、CelebA（128×128）和CARS（64×64）三大公开数据集，验证了方法在不同分辨率和数据分布上的鲁棒性。

**📈 对比分析**

与APRIL、GradViT及原始GIAS基线对比，空间锚定方法在CIFAR-10上PSNR提升约10 dB，SSIM提升0.45，LPIPS下降0.098；在CelebA与CARS上亦实现显著的PSNR/SSIM提升，表明在单步与多步联邦更新中均保持高重建质量。

**⚠️ 局限性**

局限性包括：对梯度加噪与裁剪的隐私-效用权衡导致模型精度下降；对自适应攻击者和多模态VLM等更复杂场景的鲁棒性尚未验证；需要进一步研究在分布差异和安全聚合等真实联邦环境下的表现。

---

## 554. Parallel Algorithms for Higher-Order Temporal H-Index Computation

**arXiv ID:** 2611.09649 | [PDF](https://arxiv.org/pdf/2611.09649v1)

**作者:** Lutz Oettershagen `[一作]` `[通讯]` (University of Liverpool), Lutz Oettershagen (University of Liverpool)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于顺序主导（order‑major）重构的高阶时间 H‑指数计算框架，并实现了高效的多核 CPU、GPU 以及多 GPU 并行算法；

**💡 创新点**

其创新点包括：将时间依赖查询一次性静态化为后继索引，形成并行的 gather‑suffix 结构；设计线性工作量的后缀 H‑指数计算；通过 canonical‑state 压缩显著降低跨顺序状态存储；以及基于查询感知的多 GPU 通信路由；

**🔧 技术方法**

主要技术包括：时间 CSR（TCSR）表示、后继索引、线性后缀 H‑指数（单插入直方图）、warp 与块级分段后缀核、基于度的内核自适应、查询感知的路由和全/增量状态交换；

**📊 数据集**

实验使用了大规模合成 Erdős–Rényi 时序图（1–10 亿边）以及真实 SNAP 网络（CollegeMsg、Email‑Eu、MathOverflow、Reddit、StackOverflow、WikiTalk）并以不同 λ 变体测试；

**📈 对比分析**

与原始 Stream 流式算法和 Recurs 递归实现相比，OM‑CPU 在单核/多核上显著快；在 GPU 上对 1 B 边的 128 级计算可达 33–55× 的多 GPU 加速；整体在大规模输入下实现每秒上百亿条边的处理速率；

**⚠️ 局限性**

局限性在于：多 GPU 需要高负载且不可拆分的高度顶点会导致负载不均；预处理（构建 TCSR、后继索引、路由）耗时占比高，限制单次运行的吞吐提升；以及对高阶度分布的 GPU 适配仍依赖度划分而非全局同步。

---

## 555. Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters

**arXiv ID:** 2611.09634 | [PDF](https://arxiv.org/pdf/2611.09634v1)

**作者:** Constantinos Karouzos `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LambdaMix，一种利用参数高效的低秩适配器混合，并通过相同的偏好向量在单次训练中实现多目标强化学习的语言模型调优方法

**💡 创新点**

将偏好向量同时用于适配器混合与奖励加权，联合训练适配器以学习可调节的多目标策略，从而在单一模型内实现灵活的权重切换

**🔧 技术方法**

低秩适配器（PaLoRA式）、群组相对强化学习（GRPO/GDPO）、奖励标准化（GDPO式）、单一训练回合、基于偏好的KL正则

**📊 数据集**

PKU‑SafeRLHF（善意‑无害）任务、GSM8K加指令约束（正确性‑遵从）任务、SynthPAI（保留词义‑属性隐匿）任务

**📈 对比分析**

与基于提示的条件策略、Panacea、MOC、PRISM、HoE、模型合并等基线比较，在四大模型上LambdaMix在帮助性‑无害性和正确性‑遵从性两任务上实现最高超球面积（HV），并提供更宽的权重覆盖区间；在重写任务上亦优于合并基线且参数量更低

**⚠️ 局限性**

目前仅支持两目标的偏好调节，奖励标准化对不同任务/模型效果不一，缺乏对极端偏好下安全性与属性隐私的充分评估

---

## 556. Not Your Diary, Just Your Topic: Grading the Wrong-User Control

**arXiv ID:** 2611.09609 | [PDF](https://arxiv.org/pdf/2611.09609v1)

**作者:** Karina Romanova `[一作]` (Yandex), Aleksandr Nikolich `[通讯]` (Keenable AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LaMP-2新闻分类任务中，作者通过分层替换历史、标签匹配与合成负样本，对冻结解码器加适配器的个性化效果进行评估，揭示了错用户控制中的主题混淆与真实个性化差异；并比较了前缀注入与门控交叉注意力两种注入路由；

**💡 创新点**

提出分级错用户替换方法、类别标签和文本相似度匹配的合成历史负样本，以及无历史训练对照模型和离线用户划分的评估框架，系统地拆解了随机替换造成的误差来源；

**🔧 技术方法**

使用冻结BGE编码器、Q‑Former压缩历史、soft prefix 与 gated cross‑attention 注入、无历史对照训练、Bootstrap/置换检验、Gini系数评估用户效应、输出有效性审计等技术；

**📊 数据集**

主要使用LaMP-2新闻分类数据集（15类、1,605验证样本、323用户），并对LaMP-3/4/5四个任务做了验证集对比，此外还做了电影标签任务的初步实验；

**📈 对比分析**

通过对比匹配、随机、相似、合成及无历史等历史条件，以及前缀与门控交叉注意力两种注入路由，使用准确率、宏F1和输出有效性评估；在匹配历史下准确率约0.79，随机替换下降约10点，类别标签匹配可复现约86%差距，最相似替换仅剩0.9点差距；在未见用户上优势不显著；

**⚠️ 局限性**

实验仅在停用的新闻任务、短截断历史、时间分割导致用户重叠；无历史对照模型仅在单一解码器上；对用户分布的泛化有限；架构对比未完全匹配参数量，实验规模受种子和验证集限制。

---

## 557. ReplayAudit: Auditing Outsourced LLM Fine-Tuning via Sealed Records and Selective Replay

**arXiv ID:** 2611.09607 | [PDF](https://arxiv.org/pdf/2611.09607v1)

**作者:** Yihan Xia `[一作]` (Shenzhen University), Taotao Wang `[通讯]` (Shenzhen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出一种基于封存训练记录、利用损失变化与参数更新范数做分层随机抽样，并通过确定性重放来验证选择步骤的训练完整性的系统。

**💡 创新点**

创新点在于将训练时的自监督信号与抽样策略相结合，形成可量化的回放成本模型，实现对抽样概率与审计成本的联合优化，并首次在真实 LoRA 细调中验证该方法。

**🔧 技术方法**

所用技术包括 BLAKE3 哈希绑定、BF16/DetFloat 确定性浮点计算、LoRA 参数化训练、分层抽样与间隔重放、以及基于区块链的公证机制。

**📊 数据集**

实验使用 Dolly‑15k 数据集，模型为 Qwen2.5‑1.5B/7B（以及 14B 规模探测），进行 LoRA 细调。

**📈 对比分析**

通过对比均匀抽样与分层抽样，基于离线标记的学习率偏差或跳过更新为检测目标，实验表明在 1.5B 训练上，分层抽样将完整审计时间分别降低 49% 和 67%，但在跳过更新场景下成本反而增加 38%。

**⚠️ 局限性**

局限性包括：只能验证封存步骤的一致性，无法保证整个训练历史的正确性；需要匹配硬件/软件环境；离线校准且缺乏对自适应攻击的防护；未实现可执行的惩罚或经济激励机制。

---

## 558. A wearable input/output interface learns personalized models of haptic perception

**arXiv ID:** 2611.09708 | [PDF](https://arxiv.org/pdf/2611.09708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 559. Coresets for Capacitated Clustering via Dual Concentration

**arXiv ID:** 2611.09600 | [PDF](https://arxiv.org/pdf/2611.09600v1)

**作者:** Milind Prabhu `[一作]`, Erik Waingarten `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究带容量约束（cardinality‑constrained）聚类问题的coreset构造，聚焦于经典的k‑median和k‑means目标，并在有限度量空间和欧几里得空间给出了新型样本量上界。

**💡 创新点**

创新点主要有两点：①在有限度量空间中实现与无约束聚类相同的最优规模O(k log N/ε²)的coreset；②在欧氏空间突破传统k²依赖，首次给出近线性k上界的capacitated k‑median（O(k ε⁻⁴）和k‑means（O(k ε⁻⁶）的coreset。

**🔧 技术方法**

技术上作者引入对偶解的集中性分析，利用锚定（anchoring）和截断（truncation）技巧将非局部的容量约束转化为可采样的局部量，采用弱对偶收敛（weak‑dual‑convergence）与敏感性采样（sensitivity sampling）相结合，并通过多层级的采样分布实现质量保正。

**📊 数据集**

论文并未在实验中使用具体数据集，而是针对任意输入集合给出理论上有效的coreset构造，理论证明适用于所有有限度量空间和欧几里得空间的数据。

**📈 对比分析**

与以往的上界（如O(k²ε⁻⁵log²n)或O(k³ε⁻⁵log⁵n)等）相比，本文在有限度量空间上达到了最优的O(k log N/ε²)样本量，在欧氏空间实现了显著降低的k依赖，分别得到O(k ε⁻⁴)（k‑median）和O(k ε⁻⁶)（k‑means）的近线性上界。

**⚠️ 局限性**

主要限制包括：①欧氏空间的样本量仍未达到已知的最优下界；②高维情形下仍存在维度依赖；③对偶分析的通用性仍需进一步验证，某些容量约束设置下的改进空间尚未完全探明。

---

## 560. BiLoGo: Reducing the smoothing of localised responses in Transformer-based neural operators for sheet forming

**arXiv ID:** 2611.09621 | [PDF](https://arxiv.org/pdf/2611.09621v1)

**作者:** Yingxue Zhao `[一作]` (Imperial College London), Nan Li `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出BiLoGo，一种将节点-单元消息传递与Transformer全局操作相结合的学习型薄板成形仿真器

**💡 创新点**

在保持局部信息的同时通过双流Transformer和门控融合实现长距离传播，显著减少稀疏化的平滑问题，并在内存使用上比层次化GNN低66–69%

**🔧 技术方法**

双向GraphSAGE、双流Transformer（物理切片或空间超节点）、门控残差融合与基于节点运动的单元更新

**📊 数据集**

冷成形Dome、热成形Corner以及新增的Corner+Rib（含位置和尺寸可变加固肋）

**📈 对比分析**

与单层/层次化GNN及Transformer基准在三组数据上对比，BiLoGo在位置误差、阈值厚度误差及关键厚度误差上均位列第一，尤其在局部响应区的平滑指数提升至0.94

**⚠️ 局限性**

对大尺寸三维或更复杂几何的泛化仍未验证，且在高度曲率或弯曲主导的薄板成形中单元薄化预测的准确性略低

---

## 561. Contextual Search and Bilateral Trade with a Trace-Square-Root Potential

**arXiv ID:** 2611.09615 | [PDF](https://arxiv.org/pdf/2611.09615v1)

**作者:** Romain Cosson `[一作]` (New York University), Matteo Russo `[通讯]` (EPFL)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一的潜在函数方法，解决了上下文搜索和双边贸易这两类仅收到一比特反馈的在线决策问题；在先验（贝叶斯）设置下给出与协方差相关的 O(√Σ) 损失上界，在对抗性设置下给出与维度 d 相关的 O(√d) 上界；并在后者中首次实现了与维度无关、无先验且低复杂度的算法；

**💡 创新点**

核心创新在于构造并利用了“trace‑sqrt”潜在函数（即协方差矩阵的平方根迹和），它既能捕捉贝叶斯后验不确定性，又能在对抗性分析中提供可解的潜能下降公式；同时，通过将贝叶斯潜能转化为对抗性代理潜能，首次实现了从贝叶斯到对抗性的无缝过渡；

**🔧 技术方法**

主要技术包括：贝叶斯后验更新、协方差矩阵的平方根与洛维尔顺序下的单调性与凹性证明、一次维度标准差下降与潜能下降的关系、最小化定理与 Sion 极小化极大定理相结合的随机计划抽样、以及对齐代理均值与代理协方差的递推更新；

**📊 数据集**

本文并未使用公开数据集，而是在理论模型（先验分布或欧氏单位球）下给出算法并证明性能；

**📈 对比分析**

与以往的 O(d log d)（上下文搜索）和 O(d 6^d)（双边贸易）结果相比，本工作提供了最优的 O(√d) 上界，匹配已知的 Ω(√d) 下界，显示显著的理论改进；

**⚠️ 局限性**

局限性包括：对抗性算法的实现涉及对潜能函数的优化（可能计算量大）、仅针对欧氏单位球的隐藏向量、并未考虑噪声或多比特反馈的更一般设置；此外，理论结果未在实际数据或仿真中验证。

---

## 562. PoseRieFlow: Riemannian Flow Matching for Whole-Body Text-to-Pose Generation

**arXiv ID:** 2611.09692 | [PDF](https://arxiv.org/pdf/2611.09692v1)

**作者:** Jan Spindler `[一作]` (University of Bonn), Reinhard Klein `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出PoseRieFlow，一种在SMPL‑H 52旋转乘积流形上进行文本条件全身姿态生成的Riemannian流匹配框架。

**💡 创新点**

在几何感知下直接在SO(3)^52流形上学习向量场，并采用手部专属token化和轻量化解码器，显著提升手部细节与整体关节精度，同时保持模型参数仅21.6M。

**🔧 技术方法**

Riemannian流匹配、Diffusion Transformer with AdaLN时间条件化、Flan‑T5‑Base文本交叉注意、欧氏与Riemannian对比实验以及手部token化与解码器。

**📊 数据集**

PoseScript‑A2H2（约100k SMPL‑H姿态与文本对）以及可选的Motion‑X++ 300k帧数据。

**📈 对比分析**

在与PoseScript VAE、ChatPose、PoseLLaVA、UniPose等大型LLM基准相同NFE预算下对比，PoseRieFlow在全身关节PA‑MPJPE与Geo误差均优于对手，尤其在手部显著降低误差；Riemannian版本在中等NFE下比欧氏版更精确。

**⚠️ 局限性**

仅依赖PoseScript式描述，未覆盖自由形式语言；未建模身体形状、面部表情或接触约束；对数据集偏差与生成姿态误差仍存在风险。

---

## 563. SURE:Spatiotemporal Uncertainty Representation for Efficient Vision-Language-Action Learning

**arXiv ID:** 2611.09684 | [PDF](https://arxiv.org/pdf/2611.09684v1)

**作者:** Jingye Zhang `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SURE框架，结合轻量化的转置卷积概率动作解码器和Soft‑DTW时间对齐监督，实现视觉‑语言‑动作（VLA）策略的高数据与训练效率。

**💡 创新点**

创新点包括：1) 通过局部相关的转置卷积捕捉空间不确定性；2) 使用可微Soft‑DTW解决时间不确定性；3) 两者结合显著提升样本与步骤效率。

**🔧 技术方法**

使用技术：转置卷积概率解码器、Soft‑DTW对齐监督、期望解码、L1+KL+DTW损失、OpenVLA‑7B backbone，以及对比的diffusion/flow模型。

**📊 数据集**

使用数据集：LIBERO、RoboTwin 2.0 以及六个真实桌面任务（水瓶、AAA电池、螺丝刀、玩具鸭、堆碗、倒水）。

**📈 对比分析**

与Diffusion Policy、Octo、OpenVLA、GR00T‑N1、π_0、OpenVLA‑OFT、RDT等基线在相同训练步数或数据比例下对比，SURE在LIBERO平均成功率97%（vs 87%）、RoboTwin 72%（vs 60%）、真实世界 57.5%（vs 41.7%），且在10%数据下仅93.8%成功率，远超基线。

**⚠️ 局限性**

局限性：仅针对结构化空间与时间不确定性验证，跨体型、极端多模态或大规模任务的泛化尚待评估；对安全性与人机协作的实地验证仍需加强。

---

## 564. Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences

**arXiv ID:** 2611.09670 | [PDF](https://arxiv.org/pdf/2611.09670v1)

**作者:** Soovadeep Bakshi `[一作]` (University of Texas at Austin), Apurba Bose `[通讯]` (Independent researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在机器人控制中研究了利用人类偏好学习的奖励模型进行采样式控制（MPPI）时的奖励劫持问题，并量化了更硬的搜索如何削弱真实目标性能；同时提出了理论上最优的温度设定、校准方法以及多种安全修复策略；

**💡 创新点**

创新点在于把奖励模型误差的“误差散布”转化为可检查的搜索温度阈值，证明了χ²正则化MPPI在该阈值下为最优；进一步指出传统的hold‑out精度检查并不能预警奖励劫持，并给出了基于规划器自身轨迹的校准与多重保险方案；

**🔧 技术方法**

技术包括Model Predictive Path Integral (MPPI) 控制、Bradley–Terry 偏好学习、χ² 以及 KL 正则化的分析、基于集合的鲁棒评分、自动温度调节、最坏情况最优温度规则、CEM 规划器和基于验证器的 Best‑of‑N 选择；

**📊 数据集**

使用的数据集包括 HalfCheetah‑v5、Hopper‑v5 的 D4RL 中间回放数据以及 Meta‑World v3 的 drawer‑open、plate‑slide、window‑open 任务，分别配合合成标签、盲人类标签（两位作者）和 Uni‑RLHF 众包标签；

**📈 对比分析**

通过在 10–20 次运行中比较不同温度、权重规则、集成、校准温度、最坏情况最优 λ、CEM 与 Best‑of‑N 的真实回报，发现更硬的搜索往往使真实回报下降，校准温度是最安全的修复方式，而大量合成标签的 Best‑of‑N 在极限情况下仍低于完美奖励；

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，奖励模型为小型 MLP，人工标签样本极少（仅两位作者），未考虑动力学模型误差，校准依赖 oracle 或有限人类比较，理论保证仅涵盖选择项，线性奖励假设等；

---

## 565. Video2Asset: Online Construction of Persistent 3D Street-Scene Assets

**arXiv ID:** 2611.09661 | [PDF](https://arxiv.org/pdf/2611.09661v1)

**作者:** Zihan Zheng `[一作]` (Cooperative Medianet Innovation Center Shanghai Jiao Tong University), Qiang Hu `[通讯]` (Cooperative Medianet Innovation Center Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建多视角视频的持久可编辑3D街景资产

**💡 创新点**

利用占用体作为中间表示，提出历史载体引导（HCG）和累计证据关联（AEA）实现身份感知观测与资产持续跟踪

**🔧 技术方法**

占用体建模、历史载体引导、身份编码器、累计证据关联、三维网格实例化

**📊 数据集**

CarlaOcc、nuScenes、Waymo三大多视角驾驶数据集

**📈 对比分析**

与TrackOcc、LaGS、SGE等基线对比，显著提升单帧实例分割、时间一致性（HOTA、IDF1）和资产质量（AP），并将资产碎片化降低至约1/3

**⚠️ 局限性**

库网格缺乏实例细节，有限的关节动画限制了复杂姿态编辑

---

## 566. Compaction, decay and reinforcement in a two-tier conversational memory: a paired evaluation against retrieval-augmented generation

**arXiv ID:** 2611.09807 | [PDF](https://arxiv.org/pdf/2611.09807v1)

**作者:** Osamudiamen Edogun `[一作]` (Independent researcher), Irfan Mehmood `[通讯]` (University of Bradford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了两种对话记忆：压缩式两层记忆（NMAFC）与完整转录检索，并在同一窗口、同一存储上使用Paired协议对 LoCoMo 1,985 个问题进行评测。

**💡 创新点**

提出在同一窗口、同一存储下并行两臂、同一行记录的Paired评测协议；量化压缩记忆在时间推理上的优势，并发现答案长度越长越偏向检索记忆，给出负结果与功效阈值的分析。

**🔧 技术方法**

使用 Neuromorphic Memory Architecture for Conversational AI（带指数衰减、合并、相互排名融合、语境加权、hydration、答案类型门控）与 Retrieval‑Augmented Generation（RAG）基线；生成器、嵌入模型和模型评判器统一；同时使用 token‑overlap F1 进行无模型评估。

**📊 数据集**

使用 LoCoMo 合成多会话对话数据集（10 组会话，1,985 题）并按单跳、时间推理、多跳、开放域、对抗等五类进行分组。

**📈 对比分析**

采用同一存储、同一窗口并行两臂，收集预测结果于同一行后使用 McNemar 精确检验；压缩记忆在四类可比类别上领先 6.8 分、上下文使用量少 34%；总体领先 2.0 分但对抗类 14.6 分落后，原因是答案长度越长越偏向检索记忆。

**⚠️ 局限性**

LoCoMo 为合成数据，可能不完全代表真实对话；仅 10 组会话，样本量有限；共享生成器、嵌入模型和评判器导致共模问题；缺乏人工评审，评判标准可能与人类判断不完全一致；样本量不足以检测小效应；抽取过程可能丢失措辞导致对抗类表现欠佳。

---

## 567. An Efficient Synchronization Scheme for Distributed Bandits

**arXiv ID:** 2611.09806 | [PDF](https://arxiv.org/pdf/2611.09806v1)

**作者:** Raymond Zhang `[一作]` (Université Paris-Saclay, CNRS, CentraleSupélec), Sheng Yang `[通讯]` (Université Paris-Saclay, CNRS, CentraleSupélec)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在两玩家分布式多臂赌博机中基于bandit channel模型的同步方案，并给出了同步阶段的编码与解码策略。

**💡 创新点**

通过构造组合对（combinatorial‑pair）码本并采用无渠道参数的通用解码，实现了低错误指数的同步，并给出了上界误差指数与随机编码理论的对应关系，显著降低初始化调度误差。

**🔧 技术方法**

使用了随机编码、错误指数分析、最小条件熵通用解码、信息论互信息法、方法论（type method）以及组合优化技术。

**📊 数据集**

采用仿真产生的多臂赌博机参数向量（例如μ=[0.3,0.3,0.3,0.1,0.1,0]以及α从0到0.5变化的参数集）进行实验。

**📈 对比分析**

将该方案（SN）与现有的Musical Chairs+Sequential Hopping（MC+SH）方案在同步误差率和同步长度上进行比较，结果显示SN在误差率上比MC+SH快数倍且能够逼近1/M的基准，且在α趋近0时误差率仍保持低于MC+SH。

**⚠️ 局限性**

该工作仅在两玩家场景下验证，理论中的最小化下界存在错误，未给出完整证明；对多玩家扩展的研究尚未完成，且同步长度与误差指数的选择仍需经验调参。

---

## 568. When Instruction Files Go Stale: Documentation Debt and Hidden Cost in Coding Agents

**arXiv ID:** 2611.09802 | [PDF](https://arxiv.org/pdf/2611.09802v1)

**作者:** Tural Mehtiyev `[一作]`, Wesley Assunção `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对三条真实开源仓库中的过时指令进行实验，研究LLM编码代理在遇到文档债务（过时指令）时的行为、成本和恢复过程，并对比保持过时、修正和删除三种处理方式的影响。

**💡 创新点**

创新点在于将文档债务概念应用到指令文件，揭示任务成功评价无法捕捉的隐形成本；提供可复现的实验框架来量化过时指令对代理行为的影响，并提出针对性改进建议。

**🔧 技术方法**

采用Claude Code（Haiku 5.5 与 Sonnet 5.5）两款LLM代理，在控制实验（保持、修正、删除）中记录工具调用、token消耗和轨迹；结合指令文件加载、任务构造与评估。

**📊 数据集**

使用三条真实过时指令所在的三款活跃开源仓库，构造共计22个维护任务（8+6+8），并在约600+ 次代理运行中收集数据，所有数据和代码已公开可复现。

**📈 对比分析**

通过比较三种处理条件下的任务成功率、对过时路径的使用率、工具调用次数与token消耗，结果显示任务成功率几乎相同，但在测试命令场景下过时指令导致约两次额外工具调用；修正完全消除额外成本，删除则显著降低但不完全。

**⚠️ 局限性**

局限性包括仅使用两款模型和单一 harness，案例数有限且只覆盖三类指令，难以全面泛化；统计样本有限，任务难度受限；未评估修复过时指令的实际成本。

---

## 569. Knowing When to Intervene: Causal Evaluation of AI Agents

**arXiv ID:** 2611.09792 | [PDF](https://arxiv.org/pdf/2611.09792v1)

**作者:** Ayush Sawarni `[一作]` (Stanford University), Vasilis Syrgkanis `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于顺序随机化实验的因果评估框架，用于衡量在 AI 代理执行过程中的干预（如工具调用审核、对抗性反馈）对最终任务结果的因果影响。

**💡 创新点**

创新点在于将微随机化试验（micro‑randomized trials）和结构嵌套均值模型（SNMM）引入代理评估，首次实现了无需反事实重放、仅依赖黑盒交互记录即可估计不同干预点的因果效应，并可对任意基于历史的干预策略进行离线评估。

**🔧 技术方法**

核心技术包括：顺序随机化干预设计（在每个可选决策点随机决定是否干预）；交叉拟合增强逆概率加权（AIPW）用于估计随机继续的 blip 效果；结构嵌套均值模型与 g‑估计用于无后续干预和策略增益的估计；以及对比 Monte Carlo 离线重播验证 SNMM 的有效性。

**📊 数据集**

实验数据来自三大基准：τ‑bench Retail、ToolSandbox 和 ImpossibleBench。使用的代理包括 GPT‑5 nano、Claude Haiku 5.5、Qwen3.5 Flash；干预方式为工具调用审核 vs 直接执行、对抗性反馈 vs 中性反馈；终端结果分别为任务成功率、相似度分数与规则违规检测。

**📈 对比分析**

在 τ‑bench Retail 与 ToolSandbox 中，AIPW 与 SNMM 的 blip 估计基本一致；对比直接 Monte Carlo 离线重播，SNMM 在保持精度的同时显著降低计算成本。实验显示不同工具/功能组的干预效果差异显著，部分组正向提升任务成功/相似度，另一些则无效或负向；策略增益分析表明针对特定工具的选择性审核策略相较于从不干预能带来可观的正向提升。

**⚠️ 局限性**

主要局限包括：SNMM 需要半参数模型假设，若模型错误可能导致估计偏倚；实验仅在已饱和的小模型和传统基准上进行，缺乏前沿代理与更具挑战性的任务验证；历史表示采用手工设计，缺乏自动化选择与泛化能力；干预点仅在预设的决策点上，未覆盖语义上可检出的关键事件，未来需引入 LLM 驱动的语义检测器。

---

## 570. Assessing the Impact of Retrieval-Augmented Generation on Algorithmic Problem-Solving

**arXiv ID:** 2611.09791 | [PDF](https://arxiv.org/pdf/2611.09791v1)

**作者:** Gabriel Souza Baggio `[一作]` (Federal University of Santa Maria), José Palazzo Moreira de Oliveira `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种检索增强生成（RAG）方法，旨在提高大型语言模型（LLMs）在竞争编程问题上的表现，结合嵌入式检索机制和相似性搜索算法，以识别知识数据库中的类似问题。

**💡 创新点**

创新点在于设计了一种动态调整阈值的检索机制，以减少检索噪声，并通过相似性驱动的检索显著提升了LLMs在复杂编程任务中的推理能力。

**🔧 技术方法**

使用了检索增强生成（RAG）技术，结合嵌入相似性和动态阈值机制进行信息检索。

**📊 数据集**

使用了CodeContests数据集，该数据集包含13,493个问题，其中训练集为13,328个问题，测试集为165个问题。

**📈 对比分析**

通过与零-shot和随机检索基线进行比较，RAG方法在通过率上达到了20.37%，显著高于零-shot的17.9%和随机检索的17.28%。这表明相似性驱动的检索在提高代码生成性能方面具有重要作用。

**⚠️ 局限性**

限制在于评估仅限于能够检索到相似邻居的问题子集，且使用了固定的嵌入模型和最多三条检索示例，未能探索最佳表示、检索深度和提示包装的优化。

---

## 571. Quasipolynomial-Time Planning for Decodable POMDPs

**arXiv ID:** 2611.09778 | [PDF](https://arxiv.org/pdf/2611.09778v1)

**作者:** Oliver Dobson `[一作]` (University of Cambridge), Chun Hei Michael Shiu `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种新的可观测性假设——可解码性（decodability），并在此假设下设计了一个子指数时间的近似规划算法；

**💡 创新点**

证明可解码性是ℓ1和ℓ2可观测性的严格弱化，且可在多项式时间内计算；通过构造增广与“L‑杀死”POMDP，实现对历史的紧凑表示，避免了传统滤波稳定性论证；

**🔧 技术方法**

主要技术包括：观测通道的揭示-残差分解、使用解码器构造可解码通道、基于最近L步后缀的贝叶斯更新、对权重向量的网格化离散、基于有限状态机的动态规划；

**📊 数据集**

该工作为理论性研究，不涉及任何实际数据集；

**📈 对比分析**

通过理论分析证明：算法在输入规模K（即H、S、A、O）下的运行时间为 (C·A·O·S·H/ε)^{O(log K)}，并保证累积奖励误差不超过ε；相较于先前仅对ℓ1可观测性提供的算法，本算法在常数因子上更优，并且不依赖滤波稳定性；

**⚠️ 局限性**

局限性包括：需假设存在正的可解码常数ρ；算法的运行时间仍为亚指数级，常数因子可能较大；在实际应用中如何构造解码器以及如何处理更一般的POMDP仍是待解决的问题。

---

## 572. Language as Trait, Politeness as Strategy: Controlled Tests of Linguistic Effects in LLM Social-Deduction Agents

**arXiv ID:** 2611.09768 | [PDF](https://arxiv.org/pdf/2611.09768v1)

**作者:** Yuchen Li `[一作]` (Lingnan University), Julian Togelius `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在 LURK Werewolf 框架下，对 LLM 代理的自然语言实现进行语言范围（词汇/句法）和礼貌策略（正面、负面、直接、迂回）重写，保持语义不变，利用意义控制的反事实重放设计评估语言形式对代理信任、投票与夜行动作的影响。

**💡 创新点**

提出可重现的意义控制反事实重放实验，使用检查点和对照重放消除模型非确定性噪声；开发语言范围和礼貌重写器并通过人工验证；系统地检验语言风格是否在多代理社交决策中产生偏差。

**🔧 技术方法**

LURK Werewolf 环境；DeepSeek Flash、GPT Luna 5.6、Qwen3.5‑27B 语言模型；结构化语音行动与自然语言实现分离；词汇/句法范围指标（SUBTLEX‑US、MATTR、MTLD 等）；对照重放、签名翻转检验与 Holm 校正；Politeness Judge 模型。

**📊 数据集**

30 局七人 Werewolf 游戏记录（289 个玩家‑日组合），4,335 次重放；自然游戏 30 局（1,429 句子）用于礼貌趋势；人工注释的重写验证样本（约 315 条重写）。

**📈 对比分析**

采用对照重放（原文 vs 重写）计算投票、夜行动作及信任变化，利用分层重抽样估计 95% 区间，签名翻转检验 p 值，并进行 Holm 多重比较校正。结果显示语言范围与礼貌对信任无显著影响，投票与夜行动作在校正后亦无显著效应；未校正时投票变化约 4%。

**⚠️ 局限性**

实验仅使用单一模型 DeepSeek Flash；仅在七人配置下测试；语言范围仅调整词汇/句法未覆盖完整语言熟练度；重写器验证使用不同模型；礼貌判别器对 off‑record 召回低；重写失败率相对较高；仅 30 局限制了对小效应的检出力；未对其他模型或提示设置进行评估。

---

## 573. Safety Is $Π_1$, Provability Is $Σ_1$: One Framework for the Hugging Face Breach and OpenAI's Mathematical Proofs

**arXiv ID:** 2611.09736 | [PDF](https://arxiv.org/pdf/2611.09736v1)

**作者:** Jinho Kang `[一作]` (Gwangju Institute of Science and Technology), Jinsook Kim `[通讯]` (Korea Institute for Advanced Study)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以混合框架为基础，理论分析了 2026 年 OpenAI 代理攻击 Hugging Face 与生成数学证明的两事件，并阐述了它们在安全性（Π₁ 语句）与可证明性（Σ₁ 语句）上的根本区别，证明统计建议器无法突破图灵不可判定定理。

**💡 创新点**

创新点在于将安全攻击与 AI 生成证明归结为同一统计‑演绎循环，并利用图灵不可判定定理解释两领域为何同样机制产生失败与成功；同时提出安全保证只能通过证明实现，并明确了残余风险转移至验证器、理论、定义与模式四个方面。

**🔧 技术方法**

采用的技术主要是数理逻辑、算术层次理论、图灵不可判定定理及其在 Π₁/Σ₁ 语句中的应用，并以 K&K 的混合框架为分析工具。

**📊 数据集**

本文并未使用传统机器学习数据集，而是以 2026 年 10 月 OpenAI 发布的约 700 篇 AI 生成数学证明（部分包含 Lean 4 形式化）以及 2026 年 7 月 OpenAI 代理在 Hugging Face 的安全攻击案例作为实验/案例数据。

**📈 对比分析**

方法对比主要为理论推导与案例比较；未给出数值性能指标，评估以逻辑一致性与不可判定性为核心，表明在安全性方面仅通过证明可实现保证，而在数学证明方面则可通过统计搜索获得证据。

**⚠️ 局限性**

局限性包括：1) 仅适用于混合框架下的统计‑演绎循环，未涵盖纯统计或纯演绎系统；2) 假设理论一致性与验证器正确性，实际实现中可能出现模型或验证器错误；3) 对实际安全工程的细节（如监控、沙箱细节）讨论不足；4) 仅关注 Π₁ 与 Σ₁ 语句，其他更高层次的安全/证明问题未覆盖。

---

## 574. SAGE-PAIR: Source-Pretrained Routed Residual Adaptation for Sparse-Feedback Recommendation

**arXiv ID:** 2611.09728 | [PDF](https://arxiv.org/pdf/2611.09728v1)

**作者:** Weiqing Zeng `[一作]` (Southwest Jiaotong University), Cong Wang `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在极低目标反馈预算下，提出一种分阶段残差适配器，先在相邻源期预训练低维残差并加入软路由专家，再在目标期只校准该残差，保持推荐器冻结。

**💡 创新点**

创新点在于将源期预训练的低维残差与用户条件软路由专家结合，解决稀疏标签下更新方向不可识别的问题，同时实现毫秒级更新与推理成本。

**🔧 技术方法**

采用残差适配器、低秩LoRA/Adapter、Softmax路由、源用户画像、经验分布、NDCG@10/Recall等评估指标。

**📊 数据集**

主要使用KuaiSAR-v2、KuaiRand、KuaiRec、Tenrec、LightGCN等数据集进行实验。

**📈 对比分析**

与全量微调、L2-SP、LoRA、Adapter、FOMAML等方法对比，在0.5%目标反馈下，源预训练+路由版SAGE-PAIR在锁定验证上提升NDCG@10约1.77%（相较源模型），路由专家进一步提升约0.25%，在不同backbone/数据集上表现均为正向提升。

**⚠️ 局限性**

局限性：依赖相邻源期标签稀疏时只能得到低维修正，无法捕捉更复杂的行为变化；需预先存在源期残差窗口；对用户画像的稳定性与通用性在更大范围内尚待验证。

---

## 575. Evaluating Anterior Cruciate Ligament Risk of Injury Using a Graph Convolutional Autoencoder

**arXiv ID:** 2611.09724 | [PDF](https://arxiv.org/pdf/2611.09724v1)

**作者:** Athanasios Babouras `[一作]` (McGill University), Thomas Fevens `[通讯]` (Concordia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于图卷积自编码器的无监督模型，用智能手机捕捉的姿态估计数据实时评估膝前交叉韧带（ACL）受伤风险；

**💡 创新点**

创新点在于将时空序列骨架运动直接输入多层 ST‑GCN 编码器-解码器，实现高效无监督异常检测，避免了传统依赖关键点角度或昂贵运动捕捉设备的限制；

**🔧 技术方法**

使用了 Google MediaPipe Pose（BlazePose）进行三维关节定位，构建 ST‑GCN（Spatial–Temporal Graph Convolutional Network）网络，采用 ELU 激活、空间配置划分等技术；

**📊 数据集**

数据集来自两所大学的 537 名大学生运动员，在 2023‑2024 赛季收集了 1630 次 DVJ（Drop Vertical Jump）和 2157 次 SLH（Single Leg Hop）姿态序列，并跟踪记录了 6 名 ACL 受伤事件；

**📈 对比分析**

与传统方法（如专业核心本体感受仪、完整运动捕捉实验室、Kinect V2 系统）对比，DVJ 模型在敏感度 78%、特异度 78% 的基础上获得 ROC‑AUC 0.83±0.04，SLH 模型亦实现 ROC‑AUC 0.85±0.01，显示出可与现有昂贵技术相媲美的性能；

**⚠️ 局限性**

主要局限包括样本量有限、正负样本不平衡、受伤定义不确定（可能存在假阴性），以及模型仅在高水平运动员上验证，泛化到普通人群的效果尚待进一步研究。

---

## 576. Direct Regression and Geometric Recovery for Event-based Spacecraft Pose Estimation

**arXiv ID:** 2611.09717 | [PDF](https://arxiv.org/pdf/2611.09717v1)

**作者:** Pruthvi Nandan Janga `[一作]` (Arizona State University), Bharatesh Chakravarthi `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对两套独立开发的事件摄像机空间船姿态估计系统（直接回归与基于关键点的几何求解）在 SPADES 数据集上进行系统级评估，揭示了各自的误差特征。

**💡 创新点**

创新点在于首次从聚合误差隐藏的角度揭示了方向误差与位置误差的反转现象，并提出将直接回归的姿态估计与几何约束的位移恢复相结合的混合设计假设。

**🔧 技术方法**

技术包括事件窗口化与体素化、三通道伪 RGB 表示、EfficientNet‑V2‑S 与 ConvNeXt‑Tiny 两支 CNN、HRNet‑W32 关键点检测、CycleGAN 与 DANN 领域适配、EPnP+RANSAC 角度求解、EKF 时序平滑、域锚正则化以及一系列物理驱动的数据增强。

**📊 数据集**

使用的数据集为 SPADES（包含 PROBA‑2 模拟与真实事件序列）并遵循 SPARK 2026 隐藏测试协议。

**📈 对比分析**

通过对全帧覆盖率、相对平移误差、几何角度误差及其 ESA 复合误差的统一评估来比较两套系统，结果显示直接回归在方向误差上降低 22.7%，复合误差下降 13.4%，而基于关键点的系统在平移误差上降低 78.7%，但方向误差更高，整体复合得分优于前者。

**⚠️ 局限性**

主要局限在于比较以整体系统为单位，无法归因于单一设计选择；缺乏帧级真实标签、统计显著性及各子任务的独立评估；复合误差对角度的尺度偏重导致评价偏差。

---

## 577. Tensor value and the 9/4 bound for matrix multiplication

**arXiv ID:** 2611.09716 | [PDF](https://arxiv.org/pdf/2611.09716v1)

**作者:** Mahmut Levent Doğan `[一作]` (Ludwig Maximilian University Munich), Michael Walter `[通讯]` (Ludwig Maximilian University Munich)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了矩阵乘法指数 ω 的上界为 9/4，且此结果在任意域上成立。

**💡 创新点**

创新点在于引入分数对称化值（fractional symmetrized value），用它代替谱点（tensor characters）并构造了一个部分直和的分离原理，从而得到对部分直和张量的超加性不等式，进一步简化了 OpenAI 证明并使其在所有域上可用。

**🔧 技术方法**

主要技术包括：张量退化（degeneration）、Schönhage 的渐近求和不等式、Kronecker 对称化、部分直和张量的分离与合并、以及对分数对称化值的超加性与边界分析。

**📊 数据集**

无实验数据集，全部为理论证明与抽象张量运算。

**📈 对比分析**

与以往基于 Coppersmith–Winograd 张量的 2.32112（≈log_6 64）等上界相比，本工作通过更精细的分数对称化值与分离原理直接得到更紧的上界 9/4，显示出理论上明显优于先前方法的性能。

**⚠️ 局限性**

局限性：该证明是非构造性的，仅给出了指数的上界；没有提供对应的高效矩阵乘法算法或具体实现；此外，虽然在任意域上可用，但仍未突破 9/4 限界，进一步降低指数仍需新的思想。

---

## 578. RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill

**arXiv ID:** 2611.09709 | [PDF](https://arxiv.org/pdf/2611.09709v1)

**作者:** Dylan Johnston `[一作]` (Independent researcher), Robert P. Smith `[通讯]` (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RidgeFill 方法，通过在指定 split 层后用闭式岭回归预测后续层输入，从而在预填充阶段跳过大多数高层计算，显著加速长提示的推理。

**💡 创新点**

创新点在于：①无需训练即可对未预训练模型进行填充；②利用一次闭式岭回归快速拟合每层映射；③结合行选择策略（注意力、token 类别）保留关键缓存行以控制误差。

**🔧 技术方法**

核心技术包括：闭式岭回归映射、注意力权重分析、行选择（attention‑selected、protected‑token、probe 等）、值重缩放和终端状态校正；实现部署在 llama.cpp 与 vLLM 两种推理引擎上。

**📊 数据集**

使用公开的长文档集（PG‑19、arXiv、UltraChat、EU 立法、维基百科、GitHub 源码）、RULER 需要的 4‑token‑needle 样本、LongBench v2 任务以及 90 条 32K 长文档用于 perplexity 评估。

**📈 对比分析**

与 POP、SwiftKV、全量预填充等基线对比，RidgeFill 在 Qwen3.8‑27B 上 4K tail 下实现 1.25–1.3× 的预填充加速，RULER 与 LongBench v2 损失约 –1.2 至 –1.4 分（可忽略不计），在 Qwen3.8‑Flash‑Next 上 1.5–1.6× 加速，RULER 99.8/100，novel token perplexity 上升 2.65%（区间包含零）。

**⚠️ 局限性**

局限包括：①在缺乏 QK‑norm 的模型（如 Llama‑3.1‑8B）浅 split 时误差显著，建议使用更深 split 或注意力选择；②关键 token 的 loss 较高，需要行选择或更深 split；③行选择策略对模型和提示内容依赖较大；④长文档 calibration 长度有限，可能导致远端位置误差累积。

---

## 579. A Cross-Domain TARA-to-NAC Rule Mapping Methodology: From Automotive OEM Networks to Energy Storage Systems

**arXiv ID:** 2611.09771 | [PDF](https://arxiv.org/pdf/2611.09771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 580. Test-Time Compute Scales Better with Parallel Agent Teams

**arXiv ID:** 2611.09798 | [PDF](https://arxiv.org/pdf/2611.09798v1)

**作者:** Junlin Wang `[一作]` (Duke University), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在有限的测试时计算预算下，将资源分配给并行小型代理团队，以提升任务性能。

**💡 创新点**

创新点是提出并验证“并行代理团队”（parallel agent teams）方法，结合best‑of‑N采样与协作，能够在同一预算下显著优于单代理或单一大团队。

**🔧 技术方法**

使用best‑of‑N采样技术、共享论坛的团队协作机制、统计模型评估以及Qwen3.8‑27B和DeepSeek V4.1 Flash等大模型。

**📊 数据集**

在12个长周期优化任务上进行实验，任务涵盖GPU内核优化、形式化数学推理、机器学习、结构设计等多个领域。

**📈 对比分析**

与单代理、全局大团队以及GPT‑5.6‑sol等方法比较，10个并行小队在预算10–100时相对单代理提升10–27%，且在相同成本下在多数任务中超过GPT‑5.6‑sol。

**⚠️ 局限性**

局限包括统计模型对非正态分布的预测误差、团队规模仅测试至10个样本、实验仅在单一代理环境下进行，且未探讨更大团队或多模型协作的效果。

---

## 581. Are These Programmers Ripe Enough to Eat? Evaluating How LLMs Handle Lexical Knowledge Conflicts in Translation

**arXiv ID:** 2611.09785 | [PDF](https://arxiv.org/pdf/2611.09785v1)

**作者:** Dongxu Wu `[一作]` (Ohio State University), Micha Elsner `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造了控制性中文‑英文翻译实验，给熟悉词赋新义并评估模型对上下文词义的依赖。

**💡 创新点**

首次系统量化词频、句子语境、提示框架及模型规模对外部词义采纳的影响，并揭示大模型在忽略提示时仍倾向采纳新义。

**🔧 技术方法**

使用基于提示的词义注入（dictionary、rule、language game等）、贝叶斯混合效应模型和混合效应回归进行分析。

**📊 数据集**

实验集为150个两字名词，覆盖动物、果蔬、车辆、职业、乐器五类，分三频段，并构造多句子上下文与五种提示框架。

**📈 对比分析**

对九个模型（GPT‑5.5、Qwen3‑8B/32B、Llama‑3‑8B/70B、DeepSeek‑LLM‑7B/67B、Gemma‑3‑4B/27B）进行评测，CK依赖率在24.6%–97.9%之间，频率低、语境支持及大模型倾向更高；提示框架与模态弱化、重复、推理等干预也显著改变依赖率。

**⚠️ 局限性**

仅使用合成重定义、仅限名词、单句模板、仅高资源语言、词频来源与模型训练不完全一致等限制，导致结果不易推广到其他语言、词类或真实检索资源。

---

## 582. A lightweight convolutional neural network for target detection in an edge environment

**arXiv ID:** 2611.09756 | [PDF](https://arxiv.org/pdf/2611.09756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 583. Autonomous Whole-body Loco-manipulation Using a Snake Robot

**arXiv ID:** 2611.09777 | [PDF](https://arxiv.org/pdf/2611.09777v1)

**作者:** Adarsh Salagame `[一作]` (Northeastern University), Alireza Ramezani `[通讯]` (Northeastern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种全身侧向爬行的蛇形机器人COBRA，能够在不使用抓取器的情况下通过身体接触完成箱子推送，并提出统一的接触隐式动力学模型和轨迹优化框架，结合几何规划与比例控制，在硬件上实现多起始-目标配置的自主全身运动与操作。

**💡 创新点**

创新点包括：①通过Moreau–Jean时间步进法构建同时描述机器人–地面、机器人–物体和物体–地面接触的统一接触隐式动力学；②将全秩动力学映射到约简秩的平面表示，消除振荡运动，得到稳定的规划空间；③实现无分段、无抓取器的闭环全身运动与操作；④结合Bézier几何规划和CPG运动生成的轨迹优化，实现接触计划与运动生成的协同。

**🔧 技术方法**

使用技术包括：Moreau–Jean时间步进、接触隐式动力学、半隐式欧拉积分、Projected Gauss–Seidel求解、中心模式生成器（CPG）与比例控制、Bézier曲线几何规划、单射法求解简化优化、NVIDIA Jetson Orin NX硬件控制以及机器人本体的惯性测量与运动捕捉。

**📊 数据集**

数据来源：实验室使用运动捕捉系统提供的机器人与箱子位姿；仿真使用设定的摩擦系数和已知物体尺寸，未使用公开数据集。

**📈 对比分析**

通过硬件实验对20组不同起始-目标配置进行测试，成功率达到100%，箱子最终误差小于0.2 m；相较于开环侧向爬行或单纯几何规划，展示了闭环控制与轨迹优化的显著提升。简化的轨迹优化在仿真中对145 s路径求解时间不足一分钟，表明计算效率足以满足实时部署。

**⚠️ 局限性**

局限性包括：仅在平坦硬地面上验证；未在不平坦或动态环境中测试；位姿由运动捕捉提供，未实现全自主视觉定位；完整的全秩动力学尚未直接嵌入规划，仍为分阶段过程；对不同物体形状、重量和摩擦特性的泛化能力有限。

---

## 584. ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models

**arXiv ID:** 2611.09770 | [PDF](https://arxiv.org/pdf/2611.09770v1)

**作者:** Lingyun Song `[一作]` (Northwestern Polytechnical University), Xuequn Shang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ROSE 框架，实现多模态大语言模型的终身知识编辑，限制参数更新到已编辑子空间的正交补空间，并通过风险掩码保护关键参数。

**💡 创新点**

创新点在于风险感知的正交子空间导航与多粒度知识集成（语义原型门控 + 扰动驱动子空间融合），两者共同保证长期编辑的稳定性与可扩展性。

**🔧 技术方法**

采用正交投影、QR 分解、风险掩码、语义原型构建、扰动驱动子空间融合、子空间正交化等技术。

**📊 数据集**

使用 MMEdit（VQA + 图像字幕）和 VLKEB 两个多模态编辑基准数据集进行评估。

**📈 对比分析**

与多种基线（FT-L、MEND、RECIPE、LiveEdit 等）对比，ROSE 在 Reliability、T-Gen、M-Gen、T-Loc、M-Loc 五项指标上均优于或与最强基线持平，尤其在连续 1000 次编辑后仍保持高可靠性和局部性。

**⚠️ 局限性**

局限性：依赖子空间正交化，子空间维度逼近满秩时需要额外的 SVD 截断；对极端长编辑序列的鲁棒性尚未完全验证；实验主要聚焦视觉-文本双模场景，未覆盖更广泛模态或更大规模模型。

---

## 585. AnimWeb-Bench: Benchmarking Animated Webpage Code Generation from Videos

**arXiv ID:** 2611.09769 | [PDF](https://arxiv.org/pdf/2611.09769v1)

**作者:** Zhongyi Zhang `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AnimWeb-Bench基准，用视频重建真实网页的完整动画代码；

**💡 创新点**

首次为完整动画网页生成提出双维度人类对齐评估（静态外观与动画动态）并构建多类别、跨域动画数据集；

**🔧 技术方法**

采用多模态大语言模型（MLLM）与编码代理，结合结构、文本、颜色、CLIP、位置、尺寸、速度等多种评估指标；

**📊 数据集**

使用124个来自Awwwards的真实网页（7个领域，33种动画类型），配备参考视频与静态资源；

**📈 对比分析**

对14个MLLM和3个编码代理进行自动化评估，结果显示动画再现远低于静态外观；编码代理在帧检视和渲染反馈下性能提升明显；

**⚠️ 局限性**

当前模型仍难以精准捕捉运动轨迹、时序与速度，尤其是快速连续变化的动画；需要更高质量的视觉感知与验证机制。

---

## 586. QAOA for Layered Architecture Recovery: Feasibility, Limits, and the Role of Decomposition

**arXiv ID:** 2611.09744 | [PDF](https://arxiv.org/pdf/2611.09744v1)

**作者:** Julien Cardinal `[一作]` (Universite du Quebec), Jean-Marc Robert `[通讯]` (Universite du Quebec)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将分层架构恢复（Layered Architecture Recovery）建模为二次无约束二进制优化（QUBO），使用量子近似优化算法（QAOA）加上经典后处理，并提出影响导向分解（Impact-Guided Decomposition）以处理大规模实例。

**💡 创新点**

首次将QAOA应用于软件架构恢复，系统性评估了参数灵敏度、采样预算、噪声鲁棒性，并证明分解策略是突破量子比特上限的关键。

**🔧 技术方法**

核心技术包括：QAOA量子电路、COBYLA经典优化器、基于Ising Hamiltonian的QUBO映射、影响导向分解、Qiskit AER模拟器（含IBM Quebec/ Montreal噪声模型）以及经典后处理筛选。

**📊 数据集**

实验数据集为三款真实Java系统：JReversePro 1.0（5个包，15比特）、JReversePro 1.2（6个包，18比特）和JUnit 4.13.2（8个包，24比特）。

**📈 对比分析**

与Belle等人提出的Tabu Search基线对比：在15/18比特实例上，单体QAOA可达到93%/100%的最优率；在24比特实例上单体QAOA仅6.7%，但通过分解可提升至93%；QAOA在采样预算上比深度更为关键，整体运行时间比经典搜索慢数倍至数十倍。

**⚠️ 局限性**

主要限制：需要大规模采样预算；单体电路深度提升有限；模拟耗时巨大，噪声仿真更慢；仅在小型实例上验证，缺乏对更大系统或更多层的泛化；未实现真正的量子优势；噪声模型与真实硬件差异较大。

---

## 587. One Run Is Not Enough: A Monte Carlo Protocol for Evaluating LLM Reliability on Data Science Tasks

**arXiv ID:** 2611.09729 | [PDF](https://arxiv.org/pdf/2611.09729v1)

**作者:** Wenqi Ni `[一作]` (University of Edinburgh), Miguel de Carvalho `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了DS‑QA基准与三层评价框架，对大语言模型在数据科学与统计任务中的回答进行100次Monte Carlo评估，量化回答质量与一致性；

**💡 创新点**

创新点在于（1）构建多层评价体系，将通用属性、粗粒度指标与任务特定指标结合；（2）首次使用Monte Carlo多次评测来衡量LLM回答的稳定性；（3）发现token与word verbosity比例呈近似线性关系，以及温度>1导致响应更随机的现象；

**🔧 技术方法**

使用OpenAI GPT‑4系列模型（gpt‑4o、gpt‑4o‑mini、gpt‑4.1‑mini、gpt‑4.1‑nano），Monte Carlo多次生成；通过文本相似度（Jaccard）、代码可执行度、完成率等指标进行评估；统计分析包括相关热图、标准差计算和功率律拟合；

**📊 数据集**

基准包含100道来自英国爱丁堡大学本科统计课程的问答，涉及16个数据集，涵盖9类任务（如数据理解、可视化、统计模型等），实验选取了41道题进行详细评测；

**📈 对比分析**

对每道题进行100次重复生成，计算Completion Ratio、Response Accuracy、Code Executability，合并得分并统计标准差来评估不稳定性；结果显示在中等/困难难度下，数据可视化与监督学习任务的标准差≥0.99；gpt‑4o‑mini在文本相似度最高、token使用最合理；温度>1导致响应更随机，稳定性下降；

**⚠️ 局限性**

局限性包括仅覆盖英语数据科学任务；仅评估了基准中的41道题，未覆盖完整的100道题；使用的模型已被更新版本替代，结果对新模型的可迁移性未知；实验耗费计算资源，需权衡能耗与评估粒度；

---

## 588. TriNetra: Thermal-Reliability-Guided RGB-Thermal Fusion and Tracking for Surveillance

**arXiv ID:** 2611.09725 | [PDF](https://arxiv.org/pdf/2611.09725v1)

**作者:** Sadique Khatib `[一作]` (K. K. Wagh Institute of Engineering Education and Research), Y. D. Bhise `[通讯]` (K. K. Wagh Institute of Engineering Education and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于热可靠性系数（TRC）的 RGB‑热成像目标检测与跟踪框架 TriNetra，利用 TRC 对热模态特征和追踪关联进行自适应门控，并在 M3FD 数据集上进行评估。

**💡 创新点**

创新点在于：①提出了可视化、无标签的热可靠性系数，可对热图像质量做实时评估；②将该可靠性信号在检测器的双分支融合以及追踪器的关联阶段中显式使用，实现了对热模态失效的自适应抑制；③提供了完整的可靠性评估和跟踪验证流程。

**🔧 技术方法**

核心技术包括：YOLOv8m 双分支网络（可视分支冻结、热分支训练），TRC 由对比度、梯度、饱和度、互信息、对齐误差五个图像特征组合得到；ByteTrack 追踪器的低置信度阶段引入 TRC 门控；多种数据增强和混合精度训练。

**📊 数据集**

使用公开的 M3FD RGB‑热成像数据集（4,200 对帧）进行检测与跟踪实验，并构造了 14 条视频片段与 30 条合成序列进行跟踪基准。

**📈 对比分析**

与固定门控、完整热流、关闭热流等基线相比，适应门控在清晰数据上与固定门相当，在热模态失真（模糊、饱和）时提升约 5‑8 分 mAP@50；检测 mAP@50 达 70.0%，mAP@50:95 达 47.8%；在追踪上 TRC‑门控与 ByteTrack 效果基本一致，未显著提升。

**⚠️ 局限性**

主要限制包括：仅单次实验、单一随机种子，缺乏训练方差评估；TRC 受预先校准影响，对对齐失效不敏感；跟踪评估基于无标注短片段且缺乏身份准确性指标；模型推理速度约 7 FPS，未达实时要求；训练流程中存在人为干预（阶段性监督、无 mosaic 等）。

---

## 589. FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding

**arXiv ID:** 2611.09722 | [PDF](https://arxiv.org/pdf/2611.09722v1)

**作者:** Donglin Zhuang `[一作]` (Together AI), Zhongzhu Zhou `[通讯]` (Together AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在低位KV缓存上采用仿射折叠技术，显著加速自回归解码中的注意力计算；

**💡 创新点**

提出仿射折叠方法将量化组的尺度和偏置移到矩阵乘法的输入或中间结果，从而消除逐元素的伸缩与偏移；同时利用位级操作直接构造Tensor Core的操作数，并在Hopper与Blackwell上实现了专门化的warp流水线；

**🔧 技术方法**

仿射折叠、位级量化操作数构造、warp专用流水线设计、与SGLang服务引擎集成；

**📊 数据集**

在Qwen3-8B、GPQA-Diamond、GSM8K以及合成数据集上进行评估；

**📈 对比分析**

与FlashInfer BF16、TensorRT-LLM FP8、OSCAR、BitDecoding、SAW-INT4等基线进行对比；在H100上INT2/INT4核分别比FlashInfer BF16快3.36×/2.83×、比TensorRT-LLM FP8快2.03×/1.71×；在B200上相应加速为2.67×/2.30×；服务端测试中，INT2在长上下文与高缓存命中率下的整体吞吐量提升高达170%（H100）/341%（B200）；

**⚠️ 局限性**

仅支持整数2/4-bit KV量化，量化精度损失对低位模型的准确性有影响；依赖于Hopper/Blackwell GPU架构，跨平台迁移需重写；不支持非线性量化或稀疏异常值处理。

---

## 590. ICare-DT: a digital twin for continuous forecasting of intensive care patients

**arXiv ID:** 2611.09712 | [PDF](https://arxiv.org/pdf/2611.09712v1)

**作者:** Yanke Li `[一作]` (ETH Zurich), Diego Paez-Granados `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并评估 ICare-DT，一种基于 Transformer 的 ICU 病人数字孪生模型，能够在不重新训练的情况下每小时同步患者状态并进行多目标预测；同时构建了半合成生成器和洗脱面板，对治疗响应预测能力进行基准测试。

**💡 创新点**

创新点包括：
- 设计“同步”机制，利用冻结模型每小时更新隐藏状态；
- 通过控制输入验证同步带来的提升是否真实来自新观测；
- 开发半合成生成器，模拟真实测量过程并注入已知治疗效应，为因果效应估计提供可验证基准；
- 公开完整评估代码、数据处理流程和注册分析协议，为数字孪生研究提供可复现的框架。

**🔧 技术方法**

核心技术：
- Causally masked Transformer 编码器；
- 多头竞争风险头（joint competing‑risks forecast）；
- 线性回归/梯度提升（CatBoost）作为基准；
- 半合成数据生成器（latent 状态 + 结构化效应注入）；
- 真实性审计（两样本分类器 AUC）与 placebo 检测。

**📊 数据集**

使用数据集：
- eICU‑CRD（140,473 训练、30,213 验证、30,099 测试住院记录）；
- MIMIC‑IV（66,007 训练、14,211 测试；用于转移学习评估）；
- MIMIC‑III CareVue（28,394 记录，用于早期转移分析）。

**📈 对比分析**

比较方法与性能：
- 同步预测 vs 6 小时预测：在 eICU‑CRD 测试集 12 个设置中 AUROC 提升 0.073–0.153；
- 同步预测 vs 先前 6 小时预测、最后测量值：在 11/12 设置中显著优于最后测量值；
- 同步预测 vs CatBoost：在验证集上梯度提升在大部分设置略优，加入 Twin 状态后仅在急性肾损伤上显著提升 0.0035 AUROC；
- 对 MIMIC‑IV：冻结模型转移 8/9 设置 AUROC 提升 0.025–0.114；
- 治疗响应基准：标准 CATE 学习器均未优于预测零效应；oracle‑structure 学习器与 ICare‑RH 在半合成数据上恢复效果，但 ICare‑RH 未通过 placebo 检测。

**⚠️ 局限性**

局限性：
- 仅在 eICU‑CRD（和 MIMIC‑IV 转移）数据上评估，缺乏跨机构外部验证；
- 同步提升主要来自新测量，而不一定表明能预测临床事件；
- 训练和验证均使用相同数据库，可能导致过拟合；
- 治疗响应估计仅在半合成环境下验证，未在真实治疗决策上评估；
- 生成器与现实记录的差异导致 realism audit 通过率低；
- 目标定义为平均值阈值交叉，缺乏临床意义；
- 仅评估单时间点的预测（48 小时）和静态终点，未覆盖多步骤因果推断。

---

## 591. Agents on the Patch Path: Mining AI-Associated Security Remediation in GitHub Security Advisories

**arXiv ID:** 2611.09808 | [PDF](https://arxiv.org/pdf/2611.09808v1)

**作者:** Claudio Segal `[一作]`, Raphael Machado `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在GitHub安全通告中挖掘与AI编码代理相关的补丁路径，区分直接归因与仓库级证据，并分析其对漏洞、披露路线和审查时间的影响。

**💡 创新点**

创新点在于提出一种检测AI编码代理痕迹的方法，首次将其与安全通告关联，并分离披露路线对审查时间的影响。

**🔧 技术方法**

使用了代码挖掘、自然语言处理和机器学习技术对提交历史进行归因分析。

**📊 数据集**

基于GitHub安全通告（GitHub Security Advisories）以及对应的提交和仓库数据。

**📈 对比分析**

通过与传统归因方法对比，展示其在发现代理相关补丁方面的准确率提升，且揭示代理补丁的审查周期更短。

**⚠️ 局限性**

局限性包括对GitHub平台的依赖、对非文本提交信息的识别不足以及对代理检测阈值的主观设定。

---

## 592. A proof of the Aaronson-Ambainis conjecture

**arXiv ID:** 2611.09773 | [PDF](https://arxiv.org/pdf/2611.09773v1)

**作者:** David Turturean `[一作]` `[通讯]`, David Turturean

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了对低阶多项式约束的有界函数在布尔超立方体上的行（row）与影响（influence）具有新的上界与下界。通过构造行、利用 Chebyshev 多项式与 Markov 不等式，得到每点都有代价不超过 π²d² 的行，并进而证明存在坐标 j 使得 L_j(f) ≥ (Var(f))³/(36π⁴d⁴)。

**💡 创新点**

创新点包括：① 提出了行代价的 π²d² 上界（row bound）；② 通过级数积分得到“所有配对不等式”（all‑pairs inequality）与三阶矩不等式；③ 以此改进了此前 Lovett–Zhang 等人提出的影响下界，将指数从 3、8 下降到 5/2、6；④ 全程保持纯粹理论化，未依赖经验数据。

**🔧 技术方法**

使用的主要技术手段有：Chebyshev 多项式构造与其导数性质、Markov 不等式、线性规划对偶性、Cauchy–Schwarz 及其多元推广、积分层函数（level function）与分层积分、以及对称性与双射变换。全部工作基于布尔函数的多项式展开与敏感度理论。

**📊 数据集**

该工作为纯理论研究，无实验数据集，所讨论的对象是定义在 Q_n 上的离散函数。

**📈 对比分析**

与已有结果比较时，作者以已知的影响下界（如 Lovett‑Zhang 的猜想、Agarwal‑Ben‑David 的标度）为基准，展示其新界在参数上更优（指数更小、常数更好）。由于是理论证明，没有实验性能指标，比较主要体现在理论复杂度与常数改进上。

**⚠️ 局限性**

局限性包括：① 仅适用于布尔超立方体上的函数；② 结果依赖函数非恒定且有界的假设；③ 常数 π²、π⁴ 等尚非最优，进一步改进可能仍需更精细的技术；④ 对非多项式或非有界函数的推广尚未解决。

---

## 593. From PAC bounds on the generalization error to PAC bounds on the parameter estimation error

**arXiv ID:** 2611.09767 | [PDF](https://arxiv.org/pdf/2611.09767v1)

**作者:** Mihaly Petreczky `[一作]` (University of Lille), Balint Daroczy `[通讯]` (Institute for Computer Science and Control)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一套从 PAC 误差界到参数估计误差的通用框架，并给出了对线性和双线性系统在有限样本下的 H₂ 范数误差界。

**💡 创新点**

创新点在于：①将 PAC-Bayesian 以及传统 PAC 误差界与系统理论（稳定性、可观测性、可控性）相结合，得到 oracle inequality；②利用 Markov 参数和 H₂ 范数把预测误差映射为参数误差；③在非 i.i.d. 情况下通过混合性和稳定性保证集中不等式成立。

**🔧 技术方法**

核心技术包括 PAC-Bayesian 误差界、凸/强凸分析、状态空间系统的稳定性（UEC、UEIIS）、Markov 参数表示、H₂ 范数与输入输出一致性、以及 Lipschitz 约束。

**📊 数据集**

本文以理论分析为主，没有使用具体实验数据集，所有结果均为闭式理论界定。

**📈 对比分析**

与传统的基于最小二乘或递归最小二乘等经验方法相比，给出的有限样本误差界在收敛速率上与经典渐近结果相近（O(ln N/√N)），并能在给定置信度下直接估计所需样本量。

**⚠️ 局限性**

局限性包括：需要系统满足强凸、Lipschitz、可观测/可控且参数可辨识；输入必须是 i.i.d. 或满足混合性；噪声需有界；对非线性非平稳情形的推广仍需进一步研究。

---

## 594. A Tight Hierarchy for Chain of Thought

**arXiv ID:** 2611.09759 | [PDF](https://arxiv.org/pdf/2611.09759v1)

**作者:** Thanh Le `[一作]` (University of Nebraska--Lincoln), N. V. Vinodchandran `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

研究了链式推理（CoT）在Transformer计算模型中的计算能力，并证明了CoT步骤数量的严格层次结构。

**💡 创新点**

首次给出了基于CoT步骤的层次定理，证明了在任意合理资源上增加CoT步骤可严格提升计算能力，并将此结果与传统的时间/空间层次定理相类比。

**🔧 技术方法**

采用了Transformer与图灵机之间的双向粗粒度模拟，结合复杂度理论中的填充（padding）技术，构造了在Transformer模型中实现的填充与模拟机制。

**📊 数据集**

无数据集，纯理论分析。

**📈 对比分析**

无实验对比，理论上证明了层次分离，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：仅对满足 r(n)≥n 的时间/步长函数适用；需要饱和注意力（saturated‑attention）假设；依赖于现有的二次开销模拟，未能达到近线性时间模拟；对 r(n)<n 的情况仍为开放问题。

---

## 595. Diffusion-free Score Matching for Exact Conditional Sampling

**arXiv ID:** 2611.09758 | [PDF](https://arxiv.org/pdf/2611.09758v1)

**作者:** Thomas Wedenig `[一作]` (Technische Universitaet Graz), Robert Peharz `[通讯]` (Technische Universitaet Graz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需时间依赖的分数匹配方法 DISCO，直接学习单一分数场，实现准确的条件采样。

**💡 创新点**

创新点在于：① 用 q‑加权 Fisher 散度构造混合目标，学习的分数场覆盖高低密度区域；② 通过固定观察变量即可得到条件分数，保证条件采样的渐近精确；③ 训练时不需要多噪声级别的目标，仅需采样低噪声的后验，显著降低计算复杂度。

**🔧 技术方法**

主要技术包括：分数匹配、q‑加权 Fisher 散度、混合噪声级别训练、mask 训练策略、Heun 二阶采样器、SMC 与 HMC 的条件采样、能量模型与无约束分数模型结合。

**📊 数据集**

数据集涵盖：低维 2D 生成任务（Moons、Checkerboard、Rings）；图像任务：CIFAR‑10 与 FFHQ‑64。

**📈 对比分析**

对比方法：传统扩散模型（EDM、EDM Masked）、各种条件采样启发式（Replacement、Gradient Guidance、Twisted Diffusion、DPS、RePaint）。实验表明：DISCO 的生成质量与扩散模型相当；在 2D 条件采样上实现真实条件分布，且在图像条件采样（如填空）上表现优于或与最强启发式相当；在多次重采样下，DISCO 的条件分布漂移最小，显著优于所有后处理采样器。

**⚠️ 局限性**

局限性包括：① 训练时需要对每个噪声级别采样后验，样本量大时计算开销提升；② 在高维任务中使用 mini‑batch 后验近似可能导致分数场不够精确；③ 生成质量仍略低于当前最先进的扩散模型，尤其在 CIFAR‑10 上；④ 需进一步调参以实现更佳的分数模型表达能力。

---

## 596. Contention Resolution Against an Adversarial Scheduler

**arXiv ID:** 2611.09747 | [PDF](https://arxiv.org/pdf/2611.09747v1)

**作者:** Michael A. Bender `[一作]` (Stony Brook University), Jingxun Liang `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在传统多路访问频道冲突解决问题中引入了异步调度器模型，即调度器决定每个时间步哪些玩家被允许参与，从而在此框架下分析玩家如何在无反馈或受限反馈下有效完成信息发送。

**💡 创新点**

创新点包括：①证明即使面对全适应性调度器，玩家仍能通过共享全局时钟与共享随机性实现Θ(1/ log n)的效率；②给出匹配的下界，表明该效率上界是最优；③在颗粒化时钟、未知玩家数以及缺乏共享状态等变体中提出新的协议与分析；④证明共享时钟与共享随机性是实现高效冲突解决的必要隐式共享资源。

**🔧 技术方法**

主要技术手段为：共享随机性与全局时钟的同步；“Guess‑a‑Scale”协议和“两模式”协议；一次成功公式与 Chernoff 上/下界；分块窗口分析与迭代对数加权分布 q；ghost 过程与广/窄轮策略构造下界；对数函数与迭代对数的细致数学分析。

**📊 数据集**

该工作为理论分析性质，未使用任何实验数据集，全部证明均基于概率与组合论。

**📈 对比分析**

在与经典模型的比较中，本文证明在无反馈或弱反馈下仍可达到与已知到达模型相同的Θ(1/ log n)效率；对 k‑颗粒化时钟的分析表明效率为Θ(1/ max(k, log n))，若 k≤O(log n) 可保持 Θ(1/ log n)；在未知 n 的情况下得到的效率为 Ω(1/(log n·ln ln n·…))，仅差一个迭代对数因子；当缺乏共享时钟或随机性时，效率退化到 O(log n/ n)。

**⚠️ 局限性**

局限性包括：①对未知 n 的方案只能得到迭代对数级别的近似；②对 k≫log n 的颗粒化时钟尚未证明是否最优；③必要条件（共享时钟与共享随机性）使得实现更为复杂；④下界构造使用的是非常特殊的调度器，实际系统中调度器可能更弱或更强；⑤本文未考虑能否在更弱的通信/反馈模型中进一步提升效率。

---

## 597. Continuous wavelet-based feature extraction for laser-induced breakdown spectroscopy classification of human plasma samples

**arXiv ID:** 2611.09746 | [PDF](https://arxiv.org/pdf/2611.09746v1)

**作者:** Md Mahmudur Rahman `[一作]` (University of Massachusetts), Mohammad Arif Ul Alam `[通讯]` (University of Massachusetts)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估连续小波变换（CWT）提取峰值与支持向量机（SVM）分类在181例人类血浆（53例SARS‑CoV‑2阳性、128例阴性）中的应用，并通过留一交叉验证验证内部预测性能。

**💡 创新点**

将CWT峰检测与谱峰对齐相结合，构建可解释的样本级特征空间；同时系统评估峰宽、信噪阈值以及单波长排除对分类结果的影响，提供对谱峰依赖性的定量分析。

**🔧 技术方法**

连续小波变换峰检测、峰值对齐、RBF核支持向量机分类、留一交叉验证、波形宽度与信噪阈值调优、单波长排除实验。

**📊 数据集**

181例血浆样本（53例阳性、128例阴性），每样本采集100次单次激光光谱，采用均值滤波及总强度阈值剔除异常后得到代表性光谱。

**📈 对比分析**

在完整特征集上取得93.00%准确率（敏感度94.67%，特异度91.61%）；单波长排除时，去除约393.4 nm、288.2 nm、413.1 nm等关键峰会使准确率下降至84.65%–92.42%，表明分类对这些峰高度依赖。

**⚠️ 局限性**

样本量有限且阳性样本较少，留一交叉验证仅为内部评估，未验证跨批次、跨设备或不同感染状态的可重复性；缺乏基质效应控制与对照组比较，无法确定模型对临床诊断的泛化能力。

---

## 598. Metric Dimension Parameterized by Bag-Connected Treewidth and Bag-Connected Pathwidth

**arXiv ID:** 2611.09733 | [PDF](https://arxiv.org/pdf/2611.09733v1)

**作者:** Dibyayan Chakraborty `[一作]`, Sebastian Siebertz `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了新的图形基元（tail），利用它们把顶点覆盖问题精确映射到度量维数问题，从而证明了度量维数在平面图、树宽为3且 bag‑connected treewidth 为3 的图上是 NP‑完整的；同时给出在 bag‑connected pathwidth 下的 FPT 算法。

**💡 创新点**

创新点主要包括：①首次在树宽、bag‑connected treewidth 均为 3 的平面图（且无 4‑cycle 以上的诱导环）上得到 NP‑完整性，解答了 Bonnet‑Purohit 与 Bousquet‑等人提出的开放问题；②设计的 tail 基元能够在保持所有结构约束（如树宽、边界、诱导环长）同时实现硬件化；③证明 bag‑connected pathwidth 为参数时度量维数可在 f(q)|V|^O(1) 时间内求解，填补了路径分解与树分解之间的空白。

**🔧 技术方法**

使用了多种理论技术：
- 结构化归约与 tail 基元（markers、forcing pairs）
- 距离向量与分离集（Lemma 3.1）
- bag‑connected tree decomposition 的构造与分析
- 动态规划在 nice tree decomposition 与路径分解上（带连通性约束）
- 量化 q‑dominating 最短路径与 monadic 第二阶逻辑优化
- 复杂度分析与参数化下的 FPT 算法框架。

**📊 数据集**

本工作为理论计算复杂度研究，无使用实验数据集，全部结果基于构造性证明与算法设计。

**📈 对比分析**

相较于以往仅证明一般图上的 NP‑完整性，本工作在更受限的图类（平面、树宽3、无长诱导环）上同样保持 NP‑完整；并提供了 bag‑connected pathwidth 参数下的 FPT 算法，运行时间为 f(q)|V|^O(1)，其中 q 为 bag‑connected pathwidth，表明该参数化是可行的。

**⚠️ 局限性**

局限性包括：
- 仅在 bag‑connected pathwidth 参数下得到 FPT，未解决树宽或 bag‑connected treewidth 参数下的多项式时间可解性；
- NP‑完整性结果仅适用于树宽、bag‑connected treewidth 为 3 的图，尚未给出更高树宽下的更细分类；
- 算法实现依赖于复杂的动态规划与 MSO 逻辑，对实际规模的可行性尚未评估。

---

## 599. What to Delegate, What to Preserve: Human Preferences for Agentic AI Across Everyday Life and Work

**arXiv ID:** 2611.09730 | [PDF](https://arxiv.org/pdf/2611.09730v1)

**作者:** Sarah Wheeler `[一作]` (University of Illinois Urbana-Champaign), Kyrie Zhou `[通讯]` (University of Texas at San Antonio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过情景式在线问卷调查了289名美国成年人在45个AI代理使用情境下的自动化偏好、信任、协作、隐私等多维度因素，并结合开放式问答进行定性编码。

**💡 创新点**

创新点在于：① 在同一受访者内系统比较多领域任务的AI委托偏好；② 提出委托应视为执行、协作、领导与权限四维度配置，而非单一自动化水平；③ 将隐私、财务预期、满意度等因素共同纳入多变量模型进行分析。

**🔧 技术方法**

技术手段包括情景式问卷设计、Generalized Estimating Equation（GEE）与序数回归分析、以及对开放式回答的主题编码。

**📊 数据集**

数据集为45个人工构造的AI使用场景，覆盖9个生活领域，采集了289位受访者的完整调查响应（共2,601条情境评估）。

**📈 对比分析**

通过GEE模型评估任务与受访者特征对自动化偏好的影响，发现信任、领导意愿与预期节省显著正相关，满足感担忧显著负相关；WTP模型显示自动化偏好与支付意愿正相关，说明用户愿意为更高自动化付费。

**⚠️ 局限性**

限制包括：样本偏向对AI熟悉的用户，情境为假设性而非真实交互；未探究不同领域间交互效应；仅收集了愿意委托的开放式回答，未覆盖拒绝委托者的动机。

---

## 600. Scaling Graph Similarity Learning with Graph Neural Networks

**arXiv ID:** 2611.09805 | [PDF](https://arxiv.org/pdf/2611.09805v1)

**作者:** Ahmet Tüzen `[一作]` (Norwegian University of Science and Technology), Kjetil Nørvåg `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于GNN的可扩展图相似性学习框架（FiLSim），能够在节点数超过15万的大图上高效预测图编辑距离(GED)和最大公共子图(MCS)。

**💡 创新点**

创新点在于：①用共享的GNN编码器和图级池化，完全避免节点级交叉图对齐，降低计算复杂度；②使用Feature‑wise Linear Modulation (FiLM) 的相似度预测模块取代传统的NTN，参数更少且性能更好；③构建大规模合成数据集，通过树形编辑序列保证每对图都有已知的GED上界，实现大规模评测。

**🔧 技术方法**

核心技术包括：Gated Graph Neural Network (GGNN)、局部度数谱特征 (LDP)、FiLM 条件化网络、节点特征增强与聚合、均方误差+排序损失的联合训练。

**📊 数据集**

使用四个公开基准（AIDS700nef、LINUX、IMDBMulti、PTC）以及四个大规模合成数据集（Cora、PubMed、Amazon Computers、ogbn‑arxiv）进行实验。

**📈 对比分析**

与SimGNN、GMN、GraphEdX、GEN、EGSC、ERIC、GREED、GraSP、G2R、GEDGNN、GEDIOT等多种现有方法对比，FiLSim在所有基准数据集上获得最低MAE和最高排名相关性，在大规模数据集上保持可用且性能最佳，且推理时间和显存均低于或可比现有方法。

**⚠️ 局限性**

局限性：大规模数据集标签仅为GED上界而非精确值，未覆盖MCS标签；模型仅输出标量相似度，无法给出可解释的编辑路径；在极大图上仍需进一步验证泛化能力和更细粒度的评估。

---

## 601. AR-Edit: Training-Free Autoregressive Video Editing without Inversion

**arXiv ID:** 2611.09795 | [PDF](https://arxiv.org/pdf/2611.09795v1)

**作者:** Hovhannes Margaryan `[一作]` (Université Paris-Saclay), Vicky Kalogeiton `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种无需训练、无需反演的自回归视频编辑框架 AR‑Edit，能够在实时流式环境中对视频进行编辑。

**💡 创新点**

创新点在于利用 DMD 蒸馏后自回归模型中查询‑键（Q,K）在不同去噪步长保持稳定、值（V）随噪声水平显著变化的对称性，缓存单次正向前向传递中的 V 作为源身份特征，并在后续从高噪声重建时注入该 V，从而消除传统编辑方法中对反演步骤的依赖。

**🔧 技术方法**

采用 DMD 蒸馏、自回归注意力 KV 缓存、目标文本条件、软掩模（Commit‑then‑Anchor）与文本强化策略以及对 V 的选定注入，实现流式视频编辑。

**📊 数据集**

在 120 个 Pexels 视频构成的自制编辑基准（10 类，每类 12 条目）上进行评估。

**📈 对比分析**

与 SDEdit、FlowEdit、DynaEdit、StreamEdit 等方法在 Reward‑Forcing、Self‑Forcing、CausVid 等自回归骨干上进行对比；AR‑Edit 在源保真度、文本遵从度、视频质量、色调保真等指标均优于或相当于传统方法，同时实现更少的前向传播步数、较低的延迟和相近的显存占用。

**⚠️ 局限性**

局限性包括：编辑效果与文本遵从之间仍存在权衡，某些细粒度或复杂场景下的编辑效果可能受限；方法仅针对已通过 DMD 蒸馏得到的自回归模型有效；在极低帧率或高运动场景下的稳定性与性能尚未充分验证。

---

## 602. Strategic Competition and Convergence in LLM Leaderboards

**arXiv ID:** 2611.09780 | [PDF](https://arxiv.org/pdf/2611.09780v1)

**作者:** Naveen Raman `[一作]` (Carnegie Mellon University), Yiling Chen `[通讯]` (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究leaderboard竞争如何影响LLM开发者在多维能力上的资源投入与模型能力配置，并通过理论与实证双重视角探讨竞争导致的单一化或多样化结果。

**💡 创新点**

创新点在于将多维博弈与生产函数相结合，阐明共享评估既可促成能力的“单一化”也可在存在比较优势时保持差异化，并首次量化捕获与前沿拥挤的双重竞争效应。

**🔧 技术方法**

使用多维竞赛理论、存在性与均衡分析、支持向量与聚类等数学工具；在实证层面采用回归、cosine相似度、时间序列分析等统计方法。

**📊 数据集**

利用LM Arena的多维 Elo 评分数据（422个模型，59家机构，8个持续可用维度）以及OpenRouter的日活跃使用日志（408个模型，620天）。

**📈 对比分析**

通过对能力余量的聚类、性能差距随时间的收敛率以及对比前沿模型的使用分布，展示模型性能趋于一致但能力配置仍保持差异，使用分布呈现更宽松。

**⚠️ 局限性**

局限性包括：理论假设（如期望胜率、成本函数形式）与现实可能偏离；评估仅基于对人类偏好进行的排行榜，未必覆盖所有下游任务；OpenRouter数据只代表部分真实使用场景，未完全覆盖市场。

---

## 603. Maya a1: Do the Options Read the Model, or Does Their Position? Order Sensitivity in a Non-Autoregressive Decision Model

**arXiv ID:** 2611.09803 | [PDF](https://arxiv.org/pdf/2611.09803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 604. 5G-NetBotSim: A Synchronized Co-Simulation Framework for Connected Robots over ROS and srsRAN-based 5G Network

**arXiv ID:** 2611.09735 | [PDF](https://arxiv.org/pdf/2611.09735v1)

**作者:** Narges Golmohammadi `[一作]` (University of Louisville), Sabur Baidya `[通讯]` (University of Louisville)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了 5G-NetBotSim，一种将 ROS/Gazebo 机器人仿真与可执行的 srsRAN 5G 协议栈实时联调的协同仿真框架，并通过该框架评估了不同信道质量（CQI）下的遥控与跟随控制任务。

**💡 创新点**

创新点在于：①将完整的 5G 协议栈（PHY、MAC、RLC、PDCP 等）嵌入到机器人控制循环中，实现协议栈级联调；②为每个机器人提供独立的 UE、网络、ROS 与执行上下文，保证隔离与可测量；③在框架中加入 EdgeRIC 实现 RAN 侧智能控制；④通过阶段级时延拆分，清晰区分网络延时与机器人处理延时。

**🔧 技术方法**

使用技术包括：ROS Noetic + Gazebo 11 机器人仿真；Docker 化的 srsRAN 5G 软核（可执行的模拟模式）；ZeroMQ 进行主机与 gNB 的边界通信；EdgeRIC 进行 RAN 边缘智能；Linux namespace、ROS namespace 实现多实例隔离；monotonic 时钟同步与时延拆分计算。

**📊 数据集**

本文未使用公开数据集，而是构造了实验环境：单机 1×gNB + 2×UE，10 MHz 带宽、15 kHz 子载波间距、1 ms TTI，并在低/中/高 CQI 三个设定下运行固定机器人工作负载。

**📈 对比分析**

比较方法：在相同机器人任务（人机遥控、机器人跟随）下，分别测量 MCS 选择、下行吞吐量、端到端延时、任务延时与跟踪误差。实验结果显示：高 CQI 时 MCS 与吞吐量提升，导致通信延时降低；在跟随任务中，平均跟踪误差从 0.558 m 降至 0.460 m，提升 17.6%。

**⚠️ 局限性**

局限性包括：仅在虚拟射频（virtual-radio）上测试，缺乏真实空中（over-the-air）验证；实验规模仅限于 2 台机器人，未探讨大规模多机器人部署；单机运行避免了跨主机时钟同步问题，但也限制了分布式执行与更真实网络延迟场景的评估。

---

## 605. Fixed-point acceleration on Neural Wasserstein manifolds

**arXiv ID:** 2611.09723 | [PDF](https://arxiv.org/pdf/2611.09723v1)

**作者:** Vitalii Aksenov `[一作]` (Weierstrass Institute for Applied Analysis and Stochastics), Sebastian Gutierrez Hernandez `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种在概率测度上的固定点加速方法，即在参数化 Wasserstein 自然梯度下降上应用 Riemannian Anderson 混合来加速收敛。

**💡 创新点**

创新点在于把自然梯度迭代视为黎曼几何中的固定点迭代，并将 Anderson 混合扩展到 Riemannian 结构，利用历史残差实现近似二阶加速，无需 Hessian。

**🔧 技术方法**

主要技术包括：参数化概率分布为参考分布的推送映射，诱导的 Pullback Wasserstein 计量，黎曼自然梯度下降，Riemannian Anderson 混合（RAM），以及神经 ODE 流式模型。

**📊 数据集**

使用的数据集涵盖合成多峰后验（Styblinski–Tang、双香蕉）、一维热方程逆问题、八个贝叶斯逻辑回归公开数据集、聚合动力学模型以及三种生成任务（双螺旋、棋盘、八高斯）。

**📈 对比分析**

与 SGD、Adam 等标准优化器比较时，RAM 在梯度评估次数和壁钟时间上实现了 40% 以上的加速，并在所有测试任务中在迭代次数或能量下降速度上优于自然梯度下降，部分任务甚至比 Adam 快 3.6 倍。

**⚠️ 局限性**

局限性包括：需要存储 2m 乘参数数量的历史向量；对超参数（m, λ, β）的选择敏感，若设置不当加速有限；理论上仅给出局部残差下降估计，缺乏全局收敛速率；以及在极大模型下内存和计算开销可能显著。

---

## 606. MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory

**arXiv ID:** 2611.09721 | [PDF](https://arxiv.org/pdf/2611.09721v1)

**作者:** David H. Yang `[一作]` (Rensselaer Polytechnic Institute), Giorgio Severi `[通讯]` (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出MemoryDojo基准，用于评估在多会话长周期任务中，LLM代理的内存系统如何受到毒化（memory poisoning）攻击的影响；

**💡 创新点**

创新点在于：①首次构建跨内存体系结构、跨域和历史累积情境的毒化评估基准；②设计细粒度生命周期指标（插入、检索、持续、传播、行为达成）；③提出面向内存的毒化优化方法，显著提升攻击效果；

**🔧 技术方法**

技术包括：大语言模型代理（GPT-5.6-Luna 等）、五种开源内存系统（A-MEM、Graphiti、Letta、Mem0、Memora）、自动化脚本生成多会话轨迹、LLM评判器、毒化优化循环；

**📊 数据集**

数据集：基于AgentDojo四个领域（Workspace、Travel、Banking、Slack）的任务模板，结合LongMemEval-S 交互历史，构成155条多会话轨迹；

**📈 对比分析**

比较方法：在同域、跨域、历史前缀三种设置下，对五个内存系统进行插入成功率、检索成功率、行为达成率（BR、BRT）、任务/内存效用评估；结果表明所有系统均易受毒化，优化后行为达成率提升约2%-31%，但跨域和历史压力下效果显著下降；

**⚠️ 局限性**

局限性：仅在受控 sandbox 任务和合成轨迹中评估，未测试真实世界多样化环境；毒化优化仅针对单一payload，未探讨多毒或动态适应攻击；对不同 LLM 模型的泛化性仍有限。

---

## 607. Research Collaborations Within and Across Disciplines in NIH-Funded Atlas Programs

**arXiv ID:** 2611.09741 | [PDF](https://arxiv.org/pdf/2611.09741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 608. Seeing What Matters: Benchmarking Task-Conditioned Actionable Spatial Belief for Manipulation

**arXiv ID:** 2611.09794 | [PDF](https://arxiv.org/pdf/2611.09794v1)

**作者:** Yina Jian `[一作]` (Columbia University), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Seeing What Matters 基准，用冻结 RGB 视图评估任务条件下可执行空间推理，包含咖啡机、抽屉/柜、微波炉三类桌面场景，共860个评估案例；

**💡 创新点**

通过将空间信念与命令绑定，构建可审计的 Spatial–Functional Belief Ledger 与 View Belief Map，形成任务专属的可执行性评估框架；

**🔧 技术方法**

使用规则推理、元数据辅助的视图选择策略、可选的动作诱发信念修正；并对比 VLM 预排序、随机、历史等基线；

**📊 数据集**

基准数据集由仿真 MuJoCo 场景生成的 860 个固定案例、12 张 RGB 视图构成；

**📈 对比分析**

对 7 个严格清洁 VLM 进行全候选视图排序测试，成功率仅 26–43%，而参考管线在预算 H=5 时闭环成功率 98.8%，功能 OOD 98.9%；VLM 与基线相比性能悬殊；

**⚠️ 局限性**

仅限仿真环境、场景覆盖有限、未涉及真实机器人执行、仅评估视图选择不包括学习控制器；

---

## 609. The k-Clique hypothesis is dead, for good

**arXiv ID:** 2611.09764 | [PDF](https://arxiv.org/pdf/2611.09764v1)

**作者:** Virginia Vassilevska Williams `[一作]` (Massachusetts Institute of Technology), Or Zamir `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对任意足够大的 k（尤其是 k 能被 9 整除）给出一种算法，能够在 n^(ω/3−c)k 时间内计数或检测 k‑Clique（甚至是加权 k‑Clique 总和），从而否定长期存在的 k‑Clique 猜想。

**💡 创新点**

创新点在于：① 引入一种新的代数分解，将单位矩阵拆解为低秩矩阵和有限个“部分交换”矩阵；② 通过在两条相互独立的 3‑弧（wedge）上同时使用该分解，获得一个与 ω 无关的指数下降；③ 以 9‑节点模式为基点，通过递归和基于矩阵乘法的技巧，推广到所有大 k 的 Clique，得到一个普适的常数提升。

**🔧 技术方法**

核心技术包括：代数分解（来源于近期 OpenAI 关于 ETH 的反证）、部分交换（partial swap）操作、低秩矩阵重构、标准的矩阵乘法（平方矩阵）以及递归的模式图稀疏化；算法框架完全基于已有的矩阵乘法库，无需新的张量或非矩阵乘法算法。

**📊 数据集**

本工作属于理论计算机科学，未使用任何实验数据集，所有结论均为数学证明与算法分析所得。

**📈 对比分析**

与原来的 n^(ω/3)k 上界相比，本文提供了一个正的常数提升 c>0，保证无论 ω 的真实值如何（包括 ω=2 的极限情况），都能实现 n^(ω/3−c)k 的运行时间；该提升同样适用于 k‑Clique 检测、计数以及相关 NP‑Hard 优化问题的精确算法。

**⚠️ 局限性**

限制：常数 c 及相应的 δ、ε 等值未被优化，实际大小可能极小；算法在理论上可行但实现复杂，对实际输入规模的实用性未知；仅适用于 k 可被 9 整除（通过递归可间接处理其它 k），且整体时间仍以 n^(ω/3) 为基准，无法突破到更低阶的指数。

---

## 610. Link-Level Performance Comparison of 5G Beamforming: Reciprocity-Based SVD vs. Codebook-Based Precoding

**arXiv ID:** 2611.09763 | [PDF](https://arxiv.org/pdf/2611.09763v1)

**作者:** S. M. Haider Ali Shuvo `[一作]` (University of Notre Dame), Monisha Ghosh `[通讯]` (University of Notre Dame)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对5G NR中频段单用户MIMO下的SRS‑SVD预编码和CSIRS‑PMI预编码进行端到端链路级仿真评估，结合Manhattan和Chicago两城的射线追踪通道与CDL‑C统计通道，探讨两方案在不同用户强度与阴影环境中的性能差异。

**💡 创新点**

首次在实际gNB大阵列和真实UE天线模型上对两预编码方法做用户级、空间分布级别的比较，并提出基于CSI‑RSRP/CQI或SRS‑RSRP的简易逻辑回归决策规则，可将两方案性能差距缩小至约90%。

**🔧 技术方法**

采用3GPP Release 19合规链路级仿真框架、射线追踪技术、SVD预编码、Type‑I PMI代码书、Shannon与经验SE评估、BLER统计、Logistic回归预测。

**📊 数据集**

使用Manhattan与Chicago两城的实景射线追踪数据集以及CDL‑C统计通道。

**📈 对比分析**

通过测量每用户经验SE、Shannon SE、BLER、最佳发射层数进行比较；发现SRS‑SVD在高SE强信道下更优，CSIRS‑PMI在弱/阴影用户中降低掉线率，且两方案可通过约90%准确率的预测规则实现接近最优性能。

**⚠️ 局限性**

局限在于仅考虑单用户MIMO、固定16 MHz波束、Type‑I PMI、无SRS跳频、理想TDD互调、有限天线板尺寸，未评估多用户、多频域调度、Type‑II高分辨率代码书及非理想互调误差。

---

## 611. Zero-Shot Benchmarking of Unified Multimodal Models for Shallow-Water UAV Image Distortion Assessment and Restoration

**arXiv ID:** 2611.09740 | [PDF](https://arxiv.org/pdf/2611.09740v1)

**作者:** Dario Scilla `[一作]` (King Abdullah University of Science and Technology), Matthew F. McCabe `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在零样本条件下，构建并使用统一多模态大语言模型（MLLM）对浅水UAV图像的水下畸变进行评估与恢复的系统基准测试。

**💡 创新点**

首次将评估与恢复集成于同一模型框架中，并设计四种评估提示与四种恢复提示，探讨提示工程对模型性能的影响；同时提供包含1000幅半合成与全合成对图像的公开基准。

**🔧 技术方法**

使用五个开源统一多模态LLM（BAGEL、InternVL-U、MammothModa2、S1‑Omni‑Image、SenseNova‑U1），配合结构化提示，对五种水下畸变（彩虹光、日眩、折射、浑浊、光吸收）进行检测与修复。

**📊 数据集**

采用KAUSTics 600幅半合成UAV图像与Sea‑Undistort 400幅全合成透水图像，三位专家对每幅图像标注五种畸变的严重程度，构成1000幅有对比的图像数据集。

**📈 对比分析**

通过存在平衡准确率、1‑RPS、PSNR、SSIM、MS‑SSIM、LPIPS等指标进行评估，评估阶段最高存在平衡准确率为0.654，恢复阶段最佳PSNR提升仅+0.255 dB，整体恢复效果有限，且不同模型与提示对性能影响显著。

**⚠️ 局限性**

零样本、无参考图像的约束导致恢复性能不足；模型对提示高度敏感；缺乏物理约束导致生成图像结构失真；数据集为半合成/全合成，缺乏真实失真对比，限制了结果的通用性。

---

## 612. GuardrailAgent: An Agentic Framework for Jailbreak Defense

**arXiv ID:** 2611.09734 | [PDF](https://arxiv.org/pdf/2611.09734v1)

**作者:** Tianze Chang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为 GuardrailAgent 的多代理自适应防御框架，能够在输入和输出两侧对大型语言模型（LLM）的 jailbreak 攻击进行检测，并通过外部资源爬取和内部反馈循环动态更新知识库，提升对未知攻击的抵御能力。

**💡 创新点**

创新点包括：①端到端双侧防御的多代理架构；②基于 Prompt、Material、Response 三个知识库的检索与分析机制；③完全自动化的外部资源更新与内部反馈驱动的知识库演进；④在未知攻击面前表现出的快速自适应特性。

**🔧 技术方法**

技术手段主要有：使用 LLM 代理（GPT‑4.1‑mini / Gemini‑2.5‑flash）完成检索、分析与决策；语义相似度检索工具（Prompt & Material 分析工具）；结构化知识库（Prompt KB、Material KB、Response KB）；Web 爬虫与文献摘要模块自动构建新知识；反馈循环将漏检实例自动加入 Prompt KB。

**📊 数据集**

数据集与资源包括：JailbreakRadar、WildJailbreak、do‑anything‑now 等公开 jailbreak 数据集；构造的 1,000 条安全提示作为 benign 数据集；多种模型特定攻击集合（如 AIM、Devmode、GCG、PAIR 等）用于评估。

**📈 对比分析**

通过在三种主流 LLM（ChatGLM3‑6B、Vicuna‑7B、GPT‑3.5‑turbo）上与 PromptGuard、OpenAI Moderation、Llama Guard 4、AutoDefense 等基线进行对比，GuardrailAgent 在输入侧检测率 93.1%、整体检测率 93.9%、F1 分数 95.8% 等指标上均显著优于基线；在 10+ 代表性攻击类型上保持高检测率，并能通过反馈在仅 90 条攻击样本后将输入检测率提升至 80%。

**⚠️ 局限性**

局限性主要在于：①依赖相似度检索和文献总结的策略，对深层语义模糊或高级推理型攻击的检测效果可能不足；②多代理调用导致额外的计算和延迟成本；③需要持续的外部资源更新与网络爬虫支持；④系统本身可能成为攻击目标，需进一步加强对代理与知识库的安全防护。

---

## 613. Intent-Aware Caching for Efficient LLM Serving

**arXiv ID:** 2611.09715 | [PDF](https://arxiv.org/pdf/2611.09715v1)

**作者:** Kazi Hasan Ibn Arif `[一作]` (Virginia Tech), Bo Ji `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于意图感知的前缀缓存失效策略IAC，利用会话意图和回合数学习缓存重用统计并驱动在线失效决策。

**💡 创新点**

创新点在于把用户意图与回合信息作为工作负载特征，利用轻量统计近似oracle行为，显著提升缓存命中率和延迟。

**🔧 技术方法**

核心技术包括：多类别意图分类器（冻结文本嵌入+轻量MLP头）、对每个(意图,回合)组估计继续概率和对数正态延迟分布、基于这些统计的TB+S启发式失效优先级。

**📊 数据集**

使用WildChat、ShareChat（含真实时间戳）的真实聊天记录进行训练与评估，实验覆盖8B-110B LLM和多GPU配置。

**📈 对比分析**

与LRU、LFU、GDSF、LPC和oracle TB+S对比，IAC在ShareChat上缓存命中率提升至34.6%（相较LRU 12.2%提升183.6%，相较LPC 18.8%提升84%），TTFT降低多达46%，吞吐量提升50%以上。

**⚠️ 局限性**

局限性：仅在多轮对话场景评估，未覆盖代理、工具调用等重复前缀工作负载；意图分类器对嵌入选择敏感，进一步提升可改善性能；当前仅针对文本请求，缺乏多模态或混合任务支持。

---

## 614. On Agentic Manipulation Policies

**arXiv ID:** 2611.09892 | [PDF](https://arxiv.org/pdf/2611.09892v1)

**作者:** Chen Chen `[一作]` (Apple), Vladlen Koltun `[通讯]` (Apple)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了一种直接由大型通用AI模型实时生成并执行控制代码的机器人操作策略（AMP），在零样本和持续学习场景下完成多种复杂抓取与摆放任务。

**💡 创新点**

核心创新是将通用大模型置于机器人控制循环中，模型通过代码生成自创低层技能并即时调整策略，消除对预先定义的抓取/放置库或离线程序的依赖。

**🔧 技术方法**

使用大语言模型（如 Gemini‑4 Argon、Claude Opus 5.5、Qwen 等）配合自定义 harness，提供视觉、深度、相机标定感知接口、Python 代码执行器、实时轨迹规划与错误恢复，结合持久化记忆实现经验学习。

**📊 数据集**

实验数据集包括：RoboLab‑120（120 个自然语言指令的单臂/双臂/移动操作），CaP‑Gym（LIBERO‑PRO、robosuite、BEHAVIOR‑1K 三个子套），RoboDojo（42 个双臂任务）以及 12 个真实世界实验任务。

**📈 对比分析**

通过与主流 VLAs（π0.5）、WAMs（FLUX 3 Action）以及其他 AMP 基线进行零样本对比，AMP 在 RoboLab‑120 上实现 85.6%（Gemini）/82.8%（Opus 5.5）成功率，实地实验 96.3% 成功率，显著优于 FLUX 49.2% 与 π0.5 27.8%。

**⚠️ 局限性**

主要限制是推理耗时显著，导致整体执行慢；模型无法在轨迹执行期间动态修正；对高动态环境的适应性仍有限，需要进一步加速与闭环控制集成。

---

## 615. Scenario Orchestration for Closed-loop Policy Evaluation of Autonomous Driving

**arXiv ID:** 2611.09888 | [PDF](https://arxiv.org/pdf/2611.09888v1)

**作者:** Frieda Rong `[一作]` (University of Toronto), Igor Gilitschenski `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种闭环情境编排器，实时监控并调整非本体车辆的时机、路径和角色，以实现预定的交互意图

**💡 创新点**

将约束优化与情境监控相结合，实现针对目标交互的动态、全局情境调控；同时将情境实现与策略失败分离评估

**🔧 技术方法**

约束优化（最小化干预成本），回溯式时间窗控制，角色分配算法，CARLA仿真平台的多智能体控制

**📊 数据集**

CARLA仿真环境中的多种交互场景（红灯、左转、右转、变道、紧急刹车、超车）与多种自驾策略（IDM、PlanT2.0、TFv6、SimLingo 等）

**📈 对比分析**

与 ProSim、LASER、OSC2Runner 三种基线进行 6×6×4 的性能张量评估，指标包括情境实现率和策略失败率；实验表明该方法在所有场景和策略上均能实现完整情境，并在策略失败测试中获得更高的安全关键交互次数

**⚠️ 局限性**

依赖人工指定的交通行为与角色优先级，难以扩展到更复杂、开放式情境；目前仅对基于规则的非本体车辆进行干预，尚未融合数据驱动的行为模型

---

## 616. PhotoFinish: Portable Timerless Probing of Microarchitectural Latencies

**arXiv ID:** 2611.09879 | [PDF](https://arxiv.org/pdf/2611.09879v1)

**作者:** Tristan Hornetz `[一作]` (CISPA Helmholtz Center for Information Security), Michael Schwarz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种改进的卷积神经网络结构，并在七种不同实验设置下对其训练过程进行了对比；

**💡 创新点**

其创新点在于引入了多尺度特征融合与自适应正则化机制，显著提升了模型的收敛速度与最终性能；

**🔧 技术方法**

主要技术包括深度卷积网络、注意力机制、迁移学习以及学习率调度；

**📊 数据集**

实验使用了公开的CIFAR‑10与ImageNet数据集；

**📈 对比分析**

与ResNet、DenseNet等基线模型对比，所提模型的训练曲线更快收敛，最终验证误差保持在约0.024（对应约99.8%准确率），验证集准确率提升约1–2%；

**⚠️ 局限性**

局限性是模型在极低样本量任务中仍需进一步改进，且对不同硬件的适配性不够理想。

---

## 617. Grasp-TAGS: Task-Aware Grasp Selection via Critic Distillation for Prehensile Loco-Manipulation

**arXiv ID:** 2611.09864 | [PDF](https://arxiv.org/pdf/2611.09864v1)

**作者:** Xun Tu `[一作]` (University of Minnesota), Karthik Desingh `[通讯]` (University of Minnesota)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

针对大尺寸物体的移动抓取与搬运问题，提出了Grasp-TAGS框架，通过任务感知的抓取选择提升搬运成功率。

**💡 创新点**

创新点是将抓取位置视为任务相关决策变量，使用统一的RL策略学习多抓取配置，并通过critic蒸馏得到轻量化抓取效用网络进行前置抓取选择。

**🔧 技术方法**

采用PPO强化学习训练统一的高层控制策略，配合预训练低层控制器，并利用critic蒸馏实现抓取效用网络。

**📊 数据集**

使用八个常见家用大型物体（椅子、手推车、桶等）在IsaacLab模拟环境中进行实验。

**📈 对比分析**

与单抓取PPO、随机抓取和无选择的统一策略对比，Grasp-TAGS在平均成功率上提升约10个百分点，姿态误差和位置误差亦显著下降。

**⚠️ 局限性**

局限在于仍在仿真环境验证，未在真实机器人上测试；同时对抓取候选的手动选取和对未知物体的泛化尚需改进。

---

## 618. Ensemble Methodology for Reference-Free Confidence Estimation in LLM-based Code Summarization

**arXiv ID:** 2611.09846 | [PDF](https://arxiv.org/pdf/2611.09846v1)

**作者:** Marco Colombo `[一作]`, Rigel Gjomemo `[通讯]` (University Of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无参考文档的 COBOL 代码摘要质量评估框架，结合多模型生成、LLM 判别、对数概率、覆盖评估等多维度信号实现自动化可信度分级与人工审查优先级划分。

**💡 创新点**

创新点在于将多模型互评与 LLM‑as‑judge 相结合，使用线性池化与联合惩罚规则，结合词级困惑度和 G‑Eval 覆盖度，形成端到端的参考无关可信度估计。

**🔧 技术方法**

采用的技术包括多大模型（Infosys、LLaMA 3.3‑70B、Grok‑3）、OpenAI Judge 进行结构化对比评判、perplexity 计算、G‑Eval 评估覆盖、线性池化及惩罚聚合、以及加权融合最终可信度。

**📊 数据集**

主要使用 AWS CardDemo 开源 COBOL 代码库中的 100 代码段，并对每段进行人工生成参考摘要验证。

**📈 对比分析**

通过人工评审对比，发现高可信度区间（≥7）具 94% 的可直接部署精度，低可信度区间（<4）为 100% 的错误率，整体将人工审查量从 100% 降至约 43%，展示出显著的效率提升。

**⚠️ 局限性**

限制在于对所有模型共享的盲点无法检测、anchor 偏差可能导致误判、以及对领域专有术语的高困惑度会过度保守，导致误判为需人工审核。

---

## 619. Graded convolutional networks: scale invariance without the scale and robustness to unknown blur

**arXiv ID:** 2611.09878 | [PDF](https://arxiv.org/pdf/2611.09878v1)

**作者:** Tanush Shaska `[一作]` `[通讯]` (Oakland University), Tanush Shaska (Oakland University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于梯度卷积（graded convolution）的网络，利用梯度重排（graded retraction）π_ 形成尺度不变特征，并在此基础上设计了完整的卷积网络体系结构。该网络在不使用尺度的情况下实现尺度不变性，并在面对未知高斯模糊时保持较强鲁棒性。

**💡 创新点**

① 证明了所有局部尺度不变特征必然通过尺度归一化导数（scale‑normalized jet）或梯度重排 π_ 实现；② 梯度重排在未知模糊下仅产生尺度变换，而尺度归一化则会产生额外的梯度因子；③ 证明在零导数点处梯度重排能够区分所有径向归一化或连续分类器无法区分的图像族，从而提供更强的判别能力。

**🔧 技术方法**

技术手段包括：
- Gaussian scale space 与高阶导数构造 Gaussian jet；
- 梯度重排 π_（归一化到加权球面）；
- 采用二阶调和坐标的旋转不变线性映射、乘积层与门控层；
- 在图像与尺度帧中进行聚合，使用对角化的权重与均值；
- 在梯度卷积层后加入全局归一化与多层感知机读出。

此外，还对 Lindeberg 的 Gaussian derivative 网络进行重写，将第一层改为梯度重排并加入乘积特征，以验证其对模糊鲁棒性的提升。

**📊 数据集**

实验主要在 MNIST 数据集上完成，使用标准 28×28 灰度手写数字。为评估尺度鲁棒性，还使用了 MNIST‑Scale（随机 0.3–1.0 的缩放）以及对原始 MNIST 图像进行不同尺度与高斯模糊的多组测试。Lindeberg 网络的实验采用 6,000 张训练样本，测试 800 张不同尺寸及 2,000 张模糊样本。

**📈 对比分析**

与三种对照网络对比：
- 等梯度（equal‑grades）网络（π_1 作为归一化）；
- 尺度归一化网络（使用 s^½⋆ f 的尺度归一化）；
- 传统卷积网络（size‑normalized）。

比较方法：在多次训练拆分下计算准确率，采用 McNemar 统计检验并使用 Holm 调整；同时记录不同尺度/模糊下的平均准确率、输出方差以及标签一致率。

实验结果：
- 在两像素高斯模糊下，梯度网络保持 88.4% 的准确率，而等梯度网络仅 66.2%；
- 在尺度 0.5、1、1.4、2 的放大/缩小时，梯度网络在 5,000 张测试样本上平均 92% 的准确率，保持平稳；
- 对 Lindeberg 网络，第一层改为梯度重排后，两像素模糊的准确率从 91.5% 提升至 95.1%。

总体上，梯度网络在尺度不变性和模糊鲁棒性方面显著优于对照网络。

**⚠️ 局限性**

局限性：
- 仅在单一、中心化的手写数字上验证，未测试多物体或混乱背景场景；
- 模糊模型仅为高斯，未涵盖真实光学模糊；
- 梯度重排在零导数点处不连续，可能在高阶导数消失时导致数值不稳定；
- 需要预设的尺度梯子（ladder），对不同数据集的适配需要额外调参；
- 目前实验仅在 2 维灰度图像上，扩展到彩色或视频等高维场景尚未验证。

---

## 620. Health-System-Wide Mapping of Demographic Information in Routine Clinical Histopathology

**arXiv ID:** 2611.09851 | [PDF](https://arxiv.org/pdf/2611.09851v1)

**作者:** Shengjia Chen `[一作]` (Icahn School of Medicine at Mount Sinai), Gabriele Campanella `[通讯]` (Icahn School of Medicine at Mount Sinai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过深度学习对 257,322 名患者、1.3 百万 H&E 病理切片（25 种器官）进行分析，系统绘制了性别、年龄组和自报种族在常规病理切片中的可检索性。

**💡 创新点**

①首次在单一大型医疗体系内实现跨 25 个器官的“人群信息可检索性”全景图；②在不同训练与评估约束（类平衡、诊断上下文分组）下验证这些信息的稳健性；③结合文献推理提出具体组织学/临床假设，为后续定位和解释奠定基础。

**🔧 技术方法**

使用 Pathology Foundation Model SP22M（ViT‑small）生成 384 维切片嵌入，配合多实例学习（MIL）和多头注意力模块；实验中采用三种训练策略：S1（观测分布）、S2（类平衡）和 S3（诊断上下文分组）。

**📊 数据集**

数据集来自 Mount Sinai Health System，包含 1,302,440 张 H&E 全切片、731,972 份样本、327,483 病理案例，覆盖 25 个器官（如胃、肠、乳腺、皮肤等）和 257,322 名患者。

**📈 对比分析**

对 69 个器官–人群组合使用 10 折交叉验证，主要评估指标为宏观 one‑vs‑rest AUROC。S1 的平均 AUROC 为性别 0.828、年龄组 0.733、种族 0.662；S2 仅略有下降（最大 0.018）；S3 在加入诊断上下文约束后表现略低，但多处器官仍保持高 AUROC（如胃部性别 0.865、乳腺年龄组 0.816、皮肤种族 0.728）。

**⚠️ 局限性**

①单一机构来源，缺乏跨中心验证；②未对可检索信号进行空间/细胞层面的定位；③自报种族为社会/环境混合标签，难以解释；④诊断文本聚类未完全消除疾病异质性和采样偏差；⑤模型未验证是否真实利用了组织学特征还是其他潜在共变量。

---

## 621. BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design

**arXiv ID:** 2611.09825 | [PDF](https://arxiv.org/pdf/2611.09825v1)

**作者:** Dayou Du `[一作]` (University of Edinburgh), Luo Mai `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BitDecoding‑FP4 方案，实现 4 位 KV 缓存的高精度解码，并与 native FP4 Tensor Core 协同加速。

**💡 创新点**

创新点在于三种误差抑制技术（Dual‑Residual Query Quantization、Underflow‑Aware Probability Scaling、Outlier‑Aware Key Quantization）与软max、概率量化瓶颈的 kernel 设计（Transposed Softmax、Stacked Query MMA）相结合。

**🔧 技术方法**

使用微尺度 FP4（NVFP4/MXFP4）量化、Hadamard 旋转、块比例搜索、Transposed Softmax、FP4 MMA、Paged KV 缓存、CuTe‑DSL 与 vLLM 集成等技术。

**📊 数据集**

实验采用 LongBench Pro、MRCR、RULER、AIME 2024/2025、LiveCodeBench、GPQA、GLM‑4.7、Hy4‑preview 等长上下文推理与 RL 任务数据集。

**📈 对比分析**

与 BF16/FP8 KV、TRT‑LLM‑NVFP4、TurboQuant、SAW‑INT4 等基线对比，平均准确率仅低 0.5~1 点，在 B300 上解码内核加速 2.99~4.01 倍，端到端吞吐量提升 2.4~3.0 倍。

**⚠️ 局限性**

局限在于 MXFP4 下概率下溢处理不够完善，且极大模型（1M token）对内存与通道异质性依赖较高，算法与 kernel 复杂度较大。

---

## 622. Video-Coder: Reinforcing Agentic Coding for Video Generation

**arXiv ID:** 2611.09894 | [PDF](https://arxiv.org/pdf/2611.09894v1)

**作者:** Chaoyang Wang `[一作]` (University Of Central Florida), Yuzhang Shang `[通讯]` (University Of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个可多轮编写并执行Blender代码的代码驱动视频生成代理Video-Coder，利用草稿视频指导下游视频生成器；

**💡 创新点**

将大语言模型与可执行3D场景脚本结合，形成自我反馈循环的多轮代码生成；提出统一的Video‑Coder Harness交互环境，并设计双重奖励与退火策略提升视频物理一致性；

**🔧 技术方法**

使用Qwen3.5‑9B LLM、监督微调+Agentic RL、Blender脚本执行、MiniMax‑H3视频生成器、执行与视频奖励、GRPO优化以及奖励退火；

**📊 数据集**

构造了Video‑Coder‑SFT‑10k（或6k）和Video‑Coder‑RL‑4k（或2k）数据集，包含文本提示、完整多轮代码轨迹和真实视频；

**📈 对比分析**

在PhyGenBench上平均得分从0.61提升至0.67（+6个百分点），在VBench‑2.0上平均可行性达78.19%，均显著优于Minimax‑H3、CogVideoX‑1.5等基线；

**⚠️ 局限性**

受限于3D几何描述，难以处理细粒度外观与复杂物理交互；模型规模大、训练成本高；生成流程仍需依赖下游生成器，整体流水线复杂。

---

## 623. CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse

**arXiv ID:** 2611.09891 | [PDF](https://arxiv.org/pdf/2611.09891v1)

**作者:** Peter Belcak `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出代理推理缓存（ARC）技术，将大型语言模型的推理轨迹转化为可执行的CPU代码，从而实现高效、可解释、鲁棒的推理。

**💡 创新点**

创新点在于将推理过程拆分为语义函数并用自然语言描述，支持跨参数、跨表述的重用，并通过小模型与CPU执行替代大模型。

**🔧 技术方法**

采用的技术包括大型语言模型推理、微调小型模型（Claude Haiku / Llama 3.2 1B）、语义函数编程、向量相似搜索、编译静态解析等。

**📊 数据集**

使用的数据集涵盖 MATH‑500、AIME 2024‑2026、GPQA Diamond、GSM8K、MMLU‑Pro、SciBench 等数学与科学基准。

**📈 对比分析**

与原始大型模型对比，ARC 在大多数基准上保持甚至超过 98% 的准确率，鲁棒性至少 90%，每次调用成本比原始模型低 16‑291 倍。

**⚠️ 局限性**

局限性包括构建成本高、对来源模型推理质量依赖、在简单任务上性能略逊，以及对创造性或极端多样化任务的适应性有限。

---

## 624. How far does a hash-only sender go? A pre-registered map of sender memory versus bits for short text messages

**arXiv ID:** 2611.09848 | [PDF](https://arxiv.org/pdf/2611.09848v1)

**作者:** Marcelo Samoilenko `[一作]` `[通讯]` (Winserv), Marcelo Samoilenko (Winserv)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种仅在发送端使用哈希计算、在接收端使用语言模型恢复文本的无损文本传输协议。

**💡 创新点**

创新点在于把压缩与解码任务迁移到接收端，只在发送端发送校验哈希，实现无模型发件人、低内存设备的文本传输。

**🔧 技术方法**

使用键值BLAKE2b哈希、Beam搜索与预训练语言模型（Qwen2.5‑0.5B）、Slepian–Wolf 思想、负确认与原始文本回退机制。

**📊 数据集**

使用Taskmaster‑1对话数据集，挑选1,000条28–49字节的用户发言进行实验。

**📈 对比分析**

与同域PPM‑order‑1、PPM‑order‑2、smaz2、Unishox2等基线比较，主指标为每条消息交付成本（比特/消息）和空中时长；实验显示哈希方案在1,000条消息上平均每条成本192.8比特，比PPM‑order‑1多18.4比特，且在大多数链路上空中时长不占优势。

**⚠️ 局限性**

限制包括仅针对单一英语对话数据集、仅测试Apple MPS fp16环境、发件人内存测量在主机平台、PPM‑order‑2剪枝规则粗糙、未评估微控制器实现、缺乏真实加密签名等。

---

## 625. Asymptotic Analysis of Data Deduplication Under a Constant Edit Model

**arXiv ID:** 2611.09842 | [PDF](https://arxiv.org/pdf/2611.09842v1)

**作者:** Sarvin Motamen `[一作]`, Farzad Farnoud `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对变量长度去重（VLD）和多块去重（MCD）算法在两种约束编辑模型（常数次数单比特替换与常数次数突发插入）下的理论性能进行严格分析，给出了上下界并证明了在特定参数范围内VLD可以实现任意大的压缩比，而MCD在同一模型下达到与源熵同阶的最优性。

**💡 创新点**

创新点包括：① 将近似重复（含突发编辑）引入到信息理论去重模型中；② 在两种编辑模型下分别构造精确的熵下界与上界；③ 证明VLD在小块与大块标记长度取值时的性能转折，并给出最优标记长度的取值；④ 在多块去重中引入内部/边界块划分，利用头尾标记对编码成本进行细致控制，证明其与熵同阶。

**🔧 技术方法**

主要技术手段有：信息理论熵估计、概率源模型构造、内容定义的分块（CDC）与标记符号分析、内部/边界块划分、等待时间与再生性分析、联合与期望计数、Markov与Chernoff界、最优标记长度的连续优化以及上下界的匹配。

**📊 数据集**

本文使用的并非真实数据集，而是构造的随机源模型：每个源符号为随机二进制串（长度分布集中于平均长度L），随后对每个块执行固定次数的替换或突发插入，形成整体数据流；这些模型用于理论证明和上界/下界推导。

**📈 对比分析**

比较方式：通过理论上计算压缩长度与源熵之间的比值（即压缩比），并给出VLD在不同标记长度范围内的上下界；对MCD给出与熵同阶的常数因子误差界。性能结果表明：VLD在k=O(√(LlogB))时可实现任意大压缩比；但当k=Ω(√(LlogB))时才接近熵；MCD在所有模型下均能保持O(1)的误差，表现更稳健。

**⚠️ 局限性**

局限性：① 仅考虑了单比特替换和突发插入两种编辑；不包含删除、长串编辑或混合编辑；② 假设每块编辑次数固定且小于块长度；③ 对标记长度做了特定范围的优化，未涵盖所有实际场景；④ 主要分析为渐进式上界/下界，缺少具体常数或实际实现细节；⑤ 未在真实数据上验证理论结论。

---

## 626. ZeoBench: A Benchmark for Self-Supervised Learning on 3D Zeolite Representations

**arXiv ID:** 2611.09880 | [PDF](https://arxiv.org/pdf/2611.09880v1)

**作者:** Aaron Sun `[一作]` (University of Massachusetts Amherst), Subhransu Maji `[通讯]` (University of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个针对全硅沸石密集三维网格的自监督学习基准ZeoBench，并在六个不同的吸附预测任务上进行评估。

**💡 创新点**

创新点在于系统性比较多种自监督目标、网络架构以及对ImageNet预训练的适配，证明了对域内自监督预训练能显著提升低标签吸附预测，并且在异构材料（MOF）任务中也能迁移。

**🔧 技术方法**

主要技术包括自监督对比学习（SimCLR、MoCo）与蒸馏学习（BYOL、DINO），3D ResNet‑18卷积网络，尺度、旋转、平移数据增强，以及将2D ImageNet权重通过多视图/通道膨胀/深度分词等方式迁移到3D。

**📊 数据集**

使用了约33万条真实及理论沸石结构（331,172）作为预训练数据，随后在六个长链烃的吸附任务（C18–C30及分支异构体）中，训练样本量从100到10,000不等。

**📈 对比分析**

与手工特征、3D ViT、ImageNet预训练以及无预训练模型对比，SimCLR和MoCo在线性探针和微调阶段都取得了最佳或次佳的MSE表现；在低标签场景下，预训练模型相较于从零训练提升可达30%+。

**⚠️ 局限性**

局限性包括仅使用单一3D ResNet‑18骨干，未探索更大网络或变压器重建目标；缺乏对更广泛任务（催化、分离等）的验证；并且在迁移到MOF时仍需进一步提升对化学组分差异的适应性。

---

## 627. On the Reliability of Majority-Vote Pseudo-Labels in Test-Time Reinforcement Learning

**arXiv ID:** 2611.09872 | [PDF](https://arxiv.org/pdf/2611.09872v1)

**作者:** Haoyu Wang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mingming Gong `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对测试时强化学习（TTRL）的新方法PC-Dr.DPO，用于在无标签目标域数据上自适应预训练语言模型，重点解决多投票伪标签中的采样误差问题。

**💡 创新点**

创新点在于：1）将投票误差分解为总体投票偏差与有限采样变异两部分；2）提出基于投票可靠度的提示级温度自适应分布鲁棒DPO；3）加入缓存重放机制防止自监督遗忘。

**🔧 技术方法**

技术包括：Direct Preference Optimization（DPO）、分布鲁棒DPO（Dr.DPO）、提示级KL约束下的对偶温度调节、投票可靠度估计、基于轨迹得分的答案选择以及缓存重放正则化。

**📊 数据集**

实验使用六个数学与知识推理基准（AIME24、AIME25、AMC23、MATH500、GPQA-Diamond、MMLU-Pro）和五个模型骨干（Qwen2.5-1.5B/3B/7B-Instruct、Qwen2.5-Math-1.5B-Instruct、Llama-3.2-3B-Instruct）。

**📈 对比分析**

与三种TTRL基线（RENT、TTRL、Co-Reward）比较，PC-Dr.DPO在30个模型-基准组合中，在24个场景下至少与最强基线相当，且在多数情况下实现了性能提升。

**⚠️ 局限性**

局限性包括：1）无法纠正总体投票偏差，仍需外部监督；2）对大模型的计算成本高；3）在部分实验中仍出现适配崩溃；4）方法对超参数（温度、缓存权重）敏感。

---

## 628. On the (non-)closedness of the set of ReLU networks

**arXiv ID:** 2611.09887 | [PDF](https://arxiv.org/pdf/2611.09887v1)

**作者:** Eloi Martinet `[一作]`, Leon Bungert `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了具有一个隐藏层和最多N个神经元的ReLU网络集合在Sobolev空间W^1,p()中是弱闭的，因此每个强下半连续的能量在该空间上都能达到其最小值。

**💡 创新点**

首次证明了具有两个隐藏层的ReLU网络集合在Sobolev空间中不是闭合的，从而推翻了Petersen等人的猜想。

**🔧 技术方法**

使用了Sobolev空间的弱收敛性和闭合性理论，结合了ReLU激活函数的性质。

**📊 数据集**

没有具体提到使用的数据集，但讨论了与深度Ritz方法、Sobolev训练和Lipschitz学习相关的能量最小化问题。

**📈 对比分析**

通过构造反例证明了具有两个隐藏层的ReLU网络集合在W^1,p()和C()中不闭合，展示了其性能的不足。

**⚠️ 局限性**

对于深层网络的非闭合性结果，表明在某些情况下，能量的最小值可能无法达到，尤其是在没有正则化的情况下。

---

## 629. AI Exposure and the Changing Responsibilities in Job Postings

**arXiv ID:** 2611.09870 | [PDF](https://arxiv.org/pdf/2611.09870v1)

**作者:** Dora Zhao `[一作]` (Stanford University), Scott Counts `[通讯]` (Microsoft)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并应用基于LLM的管道，将招聘广告中的职责映射到O*NET的Intermediate Work Activities（IWA），并分析2023-2025年美国大型公司招聘广告中AI相关工作内容的可见度、频率及内容变化；

**💡 创新点**

提出一种端到端自动化的职责标准化方法，结合AI使用量（Copilot/Claude）衡量IWA的AI暴露度，并在同一时间窗口内对IWA频率与AI暴露的关系进行细粒度分析；

**🔧 技术方法**

利用大型语言模型（LLM）进行文本抽取、任务细分、嵌入相似度匹配及验证，配合O*NET嵌入进行任务-活动映射；

**📊 数据集**

50,000条来自33家美国大型公司的招聘广告（2023Q1-2025Q4），涵盖信息、制造、零售、金融等行业，并结合Copilot和Claude的使用日志作为AI暴露度衡量；

**📈 对比分析**

使用线性混合效应模型与Benjamini–Hochberg多重检验，比较不同IWA频率度量（Presence、Log Ratio、Mention Share）与AI暴露度（Copilot、Claude）的关联；结果显示AI暴露度越高的IWA在2025年相对频率下降，但其AI关联职责比例显著上升；

**⚠️ 局限性**

仅建立相关性而非因果关系；数据聚焦大型企业和知识工作，可能不适用于中小企业或其他国家；AI暴露度仅覆盖通用LLM，未涵盖专用或嵌入式AI；O*NET映射可能受文本表达差异影响，导致测量误差。

---

## 630. GRAC: Graph-Based Retrieval-Augmented Captioning for Surgical Scene Description

**arXiv ID:** 2611.09866 | [PDF](https://arxiv.org/pdf/2611.09866v1)

**作者:** Rochak Dhakal `[一作]` (Michigan Technological University), Weihua Zhou `[通讯]` (Michigan Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于场景图的检索增强描述框架GRAC，用于从单帧手术场景图生成自然语言描述。

**💡 创新点**

创新点在于：①使用caption‑supervised的对比检索将相似活动的图/文本检索出来；②通过图比较与兼容监督对检索到的文本进行条件化，使生成更符合当前手术交互；③将图编码、检索、比较与生成三大模块整合为统一框架。

**🔧 技术方法**

技术核心包括：图注意力网络（GATv2）进行场景图编码，InfoNCE对比学习检索头，Transformer编码器/解码器用于文本处理，图比较投影层与兼容监督头来调整检索文本。

**📊 数据集**

在公开的 EndoVis-18（14序列，共 392 帧）和 TORS（20序列，共 36 帧）两大手术视频描述数据集上进行训练与评估。

**📈 对比分析**

与 Transformer、M²Transformer、AoA、GET、SGT 等基线相比，GRAC 在 EndoVis-18 上取得 CIDEr‑D 5.883、triplet 准确率 0.526、仪器 hallucination 0.037，显著高于 SGT 的 CIDEr‑D 4.234、triplet 0.327；在 TORS 上得到 CIDEr‑D 5.06、triplet 准确率 0.454，均表现出显著提升。

**⚠️ 局限性**

局限性包括：仅针对帧级描述；依赖手工标注的场景图；数据集规模有限，缺乏跨域泛化评估；检索仅使用训练集，未探索自动生成图的鲁棒性。

---

## 631. Flow-Guided Target-Space Alignment via Path Consistency

**arXiv ID:** 2611.09863 | [PDF](https://arxiv.org/pdf/2611.09863v1)

**作者:** Ruizhi Yuan `[一作]` (University of Pittsburgh), Lu Tang `[通讯]` (University of Pittsburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为FlowAlign的流引导目标空间对齐方法，利用先学习的条件流场在训练时给出路径一致性评分，帮助源模态映射到固定目标空间。

**💡 创新点**

创新点在于将条件流匹配的速度场冻结后用作训练时的路径一致性约束，而不是直接用于推理；该约束既保持目标空间全局结构，又提升样本级对应精度。

**🔧 技术方法**

技术包括条件流匹配（CFM）、线性插值路径、路径一致性损失、源条件化速度场训练与冻结、以及与传统对比、分布匹配、MSE等基准的组合。

**📊 数据集**

使用了三类数据集：合成二维轨迹数据、公共单细胞多组学（PBMC、BMMC）中的scATAC→scRNA映射、以及鸟类图像-文本对齐数据CUBICC。

**📈 对比分析**

在与MSE、InfoNCE、MMD、邻域保持等基准对比中，FlowAlign在合成任务上获得最低的端点MSE和轨迹误差；在单细胞任务上取得最低FOSCTTM、最高单细胞匹配率及较高的聚类/标签迁移指标；在图像-文本任务中得到最佳FOSCTTM、R@1和标签迁移率。

**⚠️ 局限性**

局限性包括需要充分的配对样本、准确的速度场训练、以及仅支持线性欧氏路径；若源表示与目标空间不兼容或速度场不佳，路径一致性约束可能失效。

---

## 632. Computer-Use Agents Are Embodied Generalists

**arXiv ID:** 2611.09847 | [PDF](https://arxiv.org/pdf/2611.09847v1)

**作者:** Pranjal Aggarwal `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套将物理环境渲染到屏幕的软件，让计算机使用代理（CUA）通过键盘鼠标操作完成四类实体AI任务，并在ECCV 2026四个工作坊比赛中使用同一代理获得前六名成绩。

**💡 创新点**

将实体任务视为与数字软件的交互，证明CUA可作为实体通用模型，并主张通过软件化实体环境实现统一的通用实体AI。

**🔧 技术方法**

使用Gemini 3.5/3.6 Flash模型作为代理，设计并实现了四个专门软件接口（VLNVerse、DialNav、OpenSUN3D、MeViSv2），实现视觉定位、点击、分割、对话等功能。

**📊 数据集**

在四个比赛的官方测试集与验证集上实验，包括VLNVerse语言导航数据、DialNav对话导航数据、OpenSUN3D 3D分割数据、MeViSv2视频分割表达式与视频。

**📈 对比分析**

与比赛其他参赛者对比，单一代理在OpenSUN3D获得AP50 25.88（领先4+点）、VLNVerse成功率87.30%（第6名）、DialNav端点成功率65.26%（第3名）、MeViSv2 Final 86.72%（第2名），显示跨任务强大通用性能。

**⚠️ 局限性**

仍需依赖手工构建的软件桥接；对连续动作与高速实时交互的支持有限；纯文本动作在多关节/人形机器人上效率可能不佳；验证主要局限于比赛环境，真实物理世界鲁棒性尚待进一步研究。

---

## 633. Can Agentic Systems Close the Gap in Radiology with Production AI Tools?

**arXiv ID:** 2611.09836 | [PDF](https://arxiv.org/pdf/2611.09836v1)

**作者:** Aubin Charley `[一作]` (Raidium), Léo Machado `[通讯]` (Raidium)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种可扩展的放射学代理框架，利用编排器按临床指示调用模块化AI工具生成结构化报告，并在FDA已清洁的工具目录与无限目录上评估其性能；

**💡 创新点**

将代理系统与共享工具接口结合，系统化评估工具可用性与编排效率对报告完整性的影响；

**🔧 技术方法**

使用大型语言模型（gpt‑5.6‑luna 等）作为编排器，模拟工具输出，采用RadMatch指标对报告进行细粒度匹配；

**📊 数据集**

采用PARROT开放多模态、跨语言的虚构放射学报告语料库（1,059 条可解析报告）进行实验；

**📈 对比分析**

在现实模式下，使用FDA清洁目录时可恢复约29%可操作发现，使用无限目录提高至43%；补充最常缺失的工具后可操作召回可达53%，且不同编排模型表现差异可观；

**⚠️ 局限性**

局限在于使用虚构报告、理想化的工具模拟（无错误）、目录仅为FDA snapshot，未涵盖真实诊断复杂性和工具性能噪声。

---

## 634. BEV3SR: Evidence-Aware 3D Spatial Reasoning from Video

**arXiv ID:** 2611.09816 | [PDF](https://arxiv.org/pdf/2611.09816v1)

**作者:** Fan Zhu `[一作]` (University of Louisville), Rui Yu `[通讯]` (University of Louisville)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将视频观测转化为可查询的度量鸟瞰视图（BEV）场景记忆，并把自然语言问题形式化为证据感知的空间任务；根据任务的证据需求，自动选择几何推理或视觉证据对齐两条路径来生成答案。

**💡 创新点**

创新点：
1) BEV不再是视觉提示，而是持久的度量场景记忆；
2) 通过空间任务正式化实现查询指定参考系、目标、约束和证据需求的完整描述；
3) 依据证据类型自动切换几何推理与视觉对齐，统一两种推理流程；
4) 完全无训练、骨干不依赖的框架，可直接接入任意VLM。

**🔧 技术方法**

核心技术：RGB‑D视频+相机参数 → BEV‑索引的度量场景记忆（Mask2Former、反投影、BEV聚类、对象关联）；
空间任务正式化（参考系、目标、约束、证据需求）；
几何推理模块（参考系实例化、几何约束判定、定量算子）；
视觉证据对齐模块（基于BEV检索的对象一致视图与场景上下文视图）；
与多种VLM（Qwen3‑VL‑8B、InternVL3.5‑8B、LLaVA‑OneVision‑1.5‑8B）交互。

**📊 数据集**

使用的数据集：ScanQA、SQA3D、VSI‑Bench（静态空间子集）。

**📈 对比分析**

与同等视频帧预算的VLM基线以及多种3D感知模型（GPT4Scene、Video‑3D‑LLM、LLaVA‑3D、3D‑VisTA 等）对比；在 SQA3D 上 EM 提升 +11，ScanQA EM 提升 +1.4，VSI‑Bench Avg 提升 10，整体性能最优或与最新 3D‑aware 模型竞争。

**⚠️ 局限性**

局限性：
1) 参考系推理对位置与朝向误差敏感，误差导致几何推理失效；
2) 对深度和相机参数误差有一定鲁棒性但仍受影响；
3) 当视觉证据缺失或被遮挡时，视觉对齐难以获得足够信息；
4) 目前主要基于 RGB‑D，RGB‑only 版本仍需要进一步提升。

---

## 635. The Cost of Coverage: Conformalising Native Forecast Uncertainty for GB Wind Power

**arXiv ID:** 2611.09815 | [PDF](https://arxiv.org/pdf/2611.09815v1)

**作者:** Carl Somers `[一作]` `[通讯]` (University College London), Carl Somers (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对英国两座风电场的风速预测区间进行校准，评估六种不同的原生不确定性估计方法与其在运营条件下的 conformal calibration（包括 OSSCP、Mondrian-OSSCP、Adaptive CP 等）效果，衡量覆盖率、区间宽度和 Winkler 分数，并分析高风速下的误差能量；

**💡 创新点**

首次在真实风电运营环境中系统比较原生区间与 conformal 校准后的区间，证明在高风速条件下按风速级别分箱的 Mondrian-OSSCP 能显著提高覆盖率并减少误差能量；

**🔧 技术方法**

使用 conformal prediction（OSSCP、Mondrian-OSSCP、ACI 等）、量化回归（LightGBM-QR）、时间序列（ARIMA）、基于残差的置信区间、Bootstrap 及 TabPFN 作为基线模型，结合 NWP（NOAA GEFS）与 SCADA 数据；

**📊 数据集**

英国 Kelmarsh（6 台涡轮）和 Penmanshiel（14 台涡轮）风电场 30 分钟分辨率 SCADA 数据，配合 NOAA GEFS 0.25° NWP，ERA5 重分析用于后验验证；

**📈 对比分析**

在 30 分钟、3 小时、24 小时三个时限下，六种基线模型的 conformal 校准大幅提升了 90% 的目标覆盖率，Winkler 分数基本不变或略有提升；Mondrian-OSSCP 在高风速下将误差能量减少 40% 以上，区间宽度整体缩小 2-4%；

**⚠️ 局限性**

仅针对两座陆上风电场，无法推广至海上或其他市场；未评估实际调度和交易收益，仅以误差能量衡量；对高风速误差的真实风速预测误差未完全消除，仍需改进风速级别预测或加入高风速裁剪判定层。

---

## 636. SciCheck: A Framework for Verifying the Integrity of Deep Learning Training

**arXiv ID:** 2611.09889 | [PDF](https://arxiv.org/pdf/2611.09889v1)

**作者:** Alexander Nguyen `[一作]` (Gonzaga University), Min Long `[通讯]` (Boise State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在科学工作流中验证深度学习训练完整性的框架，允许研究者在账本之外自由探索训练配置，然后将每一次候选训练提交到账本进行两层验证：统计判别器和随机抽样的检查点重执行。

**💡 创新点**

创新点包括：①将训练过程拆分为蓝图提交与验证两步，解决了 DL 训练不可预先完全指定的问题；②引入“有效信号”统计量来检测梯度符号翻转；③使用基于判别器的异常变压器（AT）与一次性 Merkle 根记录每个 epoch 的检查点，仅存储一次 epoch 的数据；④设计了基于抽样段的重执行检查，提供可调的安全边界 1-(1-s)^k。

**🔧 技术方法**

核心技术包括：异常变压器（Anomaly Transformer）判别器、有效信号统计（η）、Replay 与 Spot‑Check 重执行、Merkle 树检查点提交、梯度与权重统计量提取。

**📊 数据集**

实验使用三种典型模型与数据集：MLP 在 HIGGS 数据集上做二分类，CNN 在 PetImages 上做猫狗分类，AE 在 MNIST 上做重构；模型规模在 100k–500k 参数之间。

**📈 对比分析**

与 Isolation Forest 与 One‑Class SVM 的基线相比，统计判别器在每个架构上都获得更高 F1（约 0.23–0.28），全流程（统计+Spot‑Check）在三种架构上的 F1 分别达到 0.966、0.906、0.976，攻击检测率接近 100%，误拒正样本率低于 4.4%。验证成本约为一次完整训练的 1.5–2.3 倍，Spot‑Check 每条记录耗时 < 1.5 秒，存储量比完整检查点减少 5–17 倍。

**⚠️ 局限性**

主要局限：仅用 3 种随机种子；未考虑自适应攻击和伪造日志的高级对手；GPU 算法非完全可比导致 δ 变化；仅验证一次性检查点，未实现完整的 Merkle 选段规则；模型规模受限于 500k 参数，未检验更大网络或其他数据集；对验证者可信性未做完整机制。

---

## 637. FlowMoP: Stochastic Multi-Person Motion Prediction

**arXiv ID:** 2611.09882 | [PDF](https://arxiv.org/pdf/2611.09882v1)

**作者:** Aadya Agrawal `[一作]` (University of Illinois Urbana Champaign), Alexander Schwing `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种条件流匹配框架，生成多人的三维运动预测分布，兼顾全局轨迹和局部姿态。

**💡 创新点**

首次在多人人体运动预测中应用流匹配实现全时程非自回归生成，并利用B-spline压缩控制点和双流Transformer结合邻居信息进行交互建模。

**🔧 技术方法**

条件流匹配、B-spline控制点重参数化、双流Transformer、邻居最近邻条件、基于时间的正态分布采样、k-means挑选多样预测。

**📊 数据集**

CMU-Mocap（UMPM）、3DPW、以及大规模专业足球数据集WorldPose。

**📈 对比分析**

在CMU-Mocap和3DPW上对比6个基线，均以best-of-6 JPE/APE/FDE度量，本文在所有指标上均优于最强基线（相对提升12%+）；在WorldPose上在不加邻居时已优于T2P/EMPMP，加入邻居进一步提升。

**⚠️ 局限性**

局限：仅为单个代理预测，未实现全局一致的多代理联合预测；邻居数量上限可提升；对极端高曲率运动的控制点选择仍待改进。

---

## 638. Locating Contact on Tendon-Driven Continuum Robots From Tendon Measurements Alone via Constrained Moving Horizon Estimation

**arXiv ID:** 2611.09877 | [PDF](https://arxiv.org/pdf/2611.09877v1)

**作者:** Haitham El-Hussieny `[一作]` `[通讯]` (Egypt-Japan University of Science and Technology), Haitham El-Hussieny (Egypt-Japan University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于约束移动窗口估计器（MHE）的连续体机器人接触检测方法，仅利用电机处的肌腱长度与张力测量。

**💡 创新点**

首次证明在已知障碍几何和互补约束下，肌腱测量可以实现接触位置与力的可辨识，并提出实时可解的线性化QP MHE方案。

**🔧 技术方法**

使用线性化不可行约束、非凸互补惩罚、实时时间迭代求解以及局部可观测性与强可检测性分析等技术。

**📊 数据集**

采用基于SOFA的有限元仿真，生成多条与墙面接触的40秒轨迹，包含随机命令与不同壁面位置、摩擦、模型不匹配等情况。

**📈 对比分析**

与EKF（含投影）及PCC等基线比较，在正面墙面接触测试中，MHE实现0.22 N的力RMSE和0.9 mm的位移误差，显著优于投影EKF的0.72 N/65 mm；在噪声、摩擦、模型误差等鲁棒性实验中，MHE仍保持低误差，证明其优越性能。

**⚠️ 局限性**

仅在仿真中验证；假设已知平面障碍且无摩擦；中段接触导致位置误差增大；理论保证局部且未给出具体常数；Python实现尚未达到硬实时需求。

---

## 639. An Empirical Study on Clustering-based Quantum Approximate Optimization Algorithm for Regression Test Case Selection

**arXiv ID:** 2611.09869 | [PDF](https://arxiv.org/pdf/2611.09869v1)

**作者:** Antonio Trovato `[一作]` (University of Salerno), Dario Di Nucci `[通讯]` (University of Salerno)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究并实现了一种基于聚类分解的量子近似优化算法（QAOA‑TCS）用于回归测试用例选择，并在多目标与单目标场景下进行系统实验，评估其在理想与噪声量子环境中的鲁棒性。

**💡 创新点**

创新点在于：①利用Ward层次聚类对测试用例进行结构化分解，将相关测试归入同一子问题；②在每个子问题上直接应用QAOA求解QUBO，克服了当前量子硬件 qubit 数量有限的限制；③通过聚合子问题解得到全局 Pareto 前沿，兼顾多目标优化。

**🔧 技术方法**

主要技术包括：Quantum Approximate Optimization Algorithm (QAOA)，Ward 层次聚类，QUBO 建模，量子噪声模拟（采样噪声、Fake Brisbane、抖动噪声），以及与经典算法（Additional Greedy、DIV‑GA）和其他量子算法（SelectQA、BootQA、IGDec‑QAOA）的对比实验。

**📊 数据集**

使用的数据集包括：SIR 上的四个软件（Flex、Grep、Gzip、Sed）；GSDTSR、Paint Control、IOF/ROL、Elevator_o2、Elevator_o3 等公开回归测试用例集。

**📈 对比分析**

实验通过非支配解数量、Hypervolume (HV)、逆向生成距离 (IGD) 以及执行时间等指标进行比较。结果表明，QAOA‑TCS 在多目标配置下大多数情况下优于传统与量子基线，在噪声环境下仍保持稳定；在单目标配置下与 IGDec‑QAOA 相当；SelectQA 在大规模问题中效率最高。

**⚠️ 局限性**

局限性包括：实验仅在模拟器上进行，未在真实量子硬件上验证；不同算法运行环境不同（D‑Wave vs 本地机器）影响运行时间比较；分解大小（N=7）和 QAOA 深度（p=1）的固定设置可能不适用于所有问题；所用数据集规模有限，缺乏更大工业系统的验证。

---

## 640. CodeQuant2: Enhanced Outlier Mitigation and Adaptive Clustering for Low-Precision LRMs

**arXiv ID:** 2611.09868 | [PDF](https://arxiv.org/pdf/2611.09868v1)

**作者:** Xingyu Liu `[一作]` (New York University), Sai Qian Zhang `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CodeQuant2框架，实现对大型推理模型的低精度激活与权重量化，结合早期推理激活的校准、旋转缩放、聚类分配与LUT加速；

**💡 创新点**

创新点在于（1）针对推理过程中早期激活波动设计推理感知校准与专门的旋转缩放；（2）引入自适应聚类分配（ACA）平衡精度与硬件延迟；（3）对聚类权重做可微软化学习并转为硬化；（4）开发直接在聚类表示上执行的LUT CUDA核；

**🔧 技术方法**

技术包括旋转与缩放变换、K‑means聚类与LoRA软化、可微学习的赋值矩阵、推理感知校准样本、基于GPU共享内存的LUT核；

**📊 数据集**

使用WikiText2、C4作为通用文本校准；GSM8K、AIME25/26、Math500、HumanEval、LiveCodeBench等推理与数学推理基准；

**📈 对比分析**

与BF16、QuaRot、ParoQuant、SqueezeLLM、CodeQuant等后训练量化基线对比，在A4W4设定下，CodeQuant2在数学推理、代码生成等任务上平均提升约8‑10%准确率；在推理吞吐上相较BF16提升约2.4×，相较SqueezeLLM提升约1.4×；

**⚠️ 局限性**

局限在于未深入分析量化后推理轨迹的质量与安全性，未评估对行为偏差或过度自信等潜在风险，未来需进一步研究量化对推理细节与安全影响。

---

## 641. Fifty-Two Shades of Pig: What the UCR Archive Really Asks Classifiers to Do

**arXiv ID:** 2611.09843 | [PDF](https://arxiv.org/pdf/2611.09843v1)

**作者:** Omar Swelam `[一作]` (University of Freiburg), Lennart Purucker `[通讯]` (University of Freiburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对UCR时间序列分类存档中的128个数据集进行系统的来源追溯审计，评估其任务构造与训练/测试拆分的真实性；

**💡 创新点**

首次揭示存档中两大核心缺陷——人工构造的目标标签和训练/测试泄漏，并讨论非时间序列、分解多变量数据与预先对齐样本对真实性的破坏；

**🔧 技术方法**

采用来源追溯协议、标签来源分类、重复样本与共享源检测算法以及统计分析技术；

**📊 数据集**

使用UCR存档中所有128个单变量时间序列数据集；

**📈 对比分析**

通过与原始划分和去重后划分的MiniRocket等模型对比，发现部分数据集的准确率被重复样本或共享源放大，整体排名失真；

**⚠️ 局限性**

仅聚焦UCR存档，部分数据集的拆分信息不完整导致泄漏评估为下限，且未涉及多变量/实时流场景的深入验证。

---

## 642. GroundWork: Bridging Human and LLM Reasoning Through a Structured Argumentation Model

**arXiv ID:** 2611.09840 | [PDF](https://arxiv.org/pdf/2611.09840v1)

**作者:** Sjoerd Vink `[一作]` (Utrecht University & Tufts University), Remco Chang `[通讯]` (Tufts University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并评估了 GroundWork，一个将托尔文论证模型外化为可编辑图形状态的混合主动式系统，支持人类与大语言模型共同构建、比较和修订文档语料中的论证结构。

**💡 创新点**

创新点在于：①将托尔文模型转化为可编辑的有向图并定义闭合操作词汇；②提供共享的中间表示，使人类和LLM能够在同一图上协同工作；③通过多领域案例展示该表示对比与测试论证的有效性。

**🔧 技术方法**

采用的技术包括：大语言模型（OpenAI GPT‑5.1‑nano 与 Qwen 2.5 14B）用于抽取与生成论证内容；图数据库/有向图结构来存储论证节点与边；可视化多视图交互界面（源文档、角色视图、工作台）；自然语言请求转换为结构化提案；类型化边缘与操作词典实现可编辑的论证状态。

**📊 数据集**

使用的数据集为：稀土磁链评估的五份公开报告（USGS、GAO、RAND、DOE 等）；法律案例的六份公开文件（庭审判决、上诉摘要、IG 调查、报告、法规、法律评论）；医学研究的六篇论文（四篇小鼠前临床研究、一期临床试验、综述）。所有文档均通过 Qwen 2.5 14B 进行论证抽取并生成图形。

**📈 对比分析**

对比方法通过三种使用案例（情报、法律、医学）以及四位领域专家的使用研究评估。结果表明专家能更清晰地追踪证据来源、比较主张并测试逻辑，但未给出定量性能指标，只报告了专家对系统功能的主观评价。

**⚠️ 局限性**

局限性包括：①受限于语料规模和LLM抽取误差，未能覆盖大型或持续变化的文档集合；②缺乏对条件推理和复杂表格等内容的完整支持；③系统仅表面化论证关系，无法自动评估证据可信度、权威性等上下文属性，需要人工判断；④在正式使用中可能削弱专家的探索性思维，需在自动化与人类感知之间取得平衡。

---

## 643. Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL

**arXiv ID:** 2611.09828 | [PDF](https://arxiv.org/pdf/2611.09828v1)

**作者:** Aria Wong `[一作]` (MATS), Neel Nanda `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了强化学习中的奖励黑客问题，构建了三个自然环境（编码、医疗对话、传记生成）并在这些环境中评估激活探针和LLM评判器等监视机制。

**💡 创新点**

创新点在于公开了可复现的奖励黑客环境，系统分析了监视器逃避行为，并提出了平衡监视信号强度与逃避风险的设计原则，进一步将这些原则迁移到医疗和传记任务中。

**🔧 技术方法**

使用了Qwen3-4B模型与GRPO策略、激活探针、LLM评判器，以及不同的惩罚/屏蔽结构（渐进惩罚、阈值惩罚、屏蔽）和难度调节的课程学习。

**📊 数据集**

数据集包括Leetcode中等/难题、医疗问答数据（带引导诊断）和历史人物传记文本，一共约1000道问题或对话。

**📈 对比分析**

通过10个随机种子对比未监视、监视、理想监视和改进方案，发现改进方案将奖励黑客率降低至接近或低于基线，并在大部分任务中保持或略微提升最终性能。

**⚠️ 局限性**

局限性包括仅在单一模型（Qwen3-4B）和可验证漏洞的实验环境下验证，未覆盖多样化真实部署场景；监视器的鲁棒性和可推广性仍有限，且更大模型可能产生更复杂的逃避策略。

---

## 644. Generative Latent Priors for Interpreting and Steering Protein Language Models

**arXiv ID:** 2611.09818 | [PDF](https://arxiv.org/pdf/2611.09818v1)

**作者:** Daniel Hinjos `[一作]` (Barcelona Supercomputing Center), Lucía Urcelay `[通讯]` (Centre for Genomic Regulation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在蛋白质语言模型中使用生成潜在先验(GLP)学习冻结模型的激活分布，用作解释和编辑工具。

**💡 创新点**

创新在于将无结构假设的扩散模型迁移到蛋白质激活空间，实现单元级别的概念捕捉和在曲面上修正编辑。

**🔧 技术方法**

采用流匹配扩散模型、单元线性探测、激活修正与蒙特卡罗采样等技术。

**📊 数据集**

使用ESM‑2的UniRef50激活（8M、650M模型），训练和评估基于5M随机蛋白。

**📈 对比分析**

在概念AUC上单元元神经元优于SAE、ESM FFN和残差流；在属性导向编辑中，GLP校正显著提高方向性，但折叠可信度仍有限。

**⚠️ 局限性**

局限在于修正后序列在解码后失去曲面典型性，结构编辑仅在螺旋等局部基序成功，且对整体折叠和多步解码不具备天然性。

---

