# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-20 | 今日论文总数: 377

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MLLM-DataEngine: Closing the Loop of Multimodal Instruction Tuning Data Generation

**arXiv ID:** 2607.15299 | [PDF](https://arxiv.org/pdf/2607.15299v1)

**作者:** Zhiyuan Zhao `[一作]` (Shanghai AI Laboratory), Conghui He `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MLLM-DataEngine，闭环迭代评估→指导→生成→训练的多模态指令调优数据生成框架。

**💡 创新点**

创新点在于将模型评估与数据生成结合的闭环流程，并使用自适应错误样本采样（ABS）指导 GPT‑4 生成针对性高质量数据。

**🔧 技术方法**

采用 GPT‑4 进行数据生成，视觉信息来自 Visual Genome 与 PaddleOCR，评估使用 SEED‑Bench，循环中结合 ABS、数据过滤与模型微调。

**📊 数据集**

主要使用 SEED‑Bench 进行评估、Visual Genome 作为图像信息来源，并利用 PaddleOCR 提取文本；生成的数据集与原始指令调优数据一起使用。

**📈 对比分析**

与 LRV、SVIT 等现有方法对比，使用相同或更少的数据量在 SEED‑Bench、MMBench、VQA 等基准上均实现 2–4% 的性能提升，展示了更高的效率和效果。

**⚠️ 局限性**

限制在于增量改进在第三轮后趋于饱和，受限于模型输入分辨率、视觉特征粒度等硬件与模型自身能力，难以进一步提升。

---

## 2. Diffusion models recover accurate mixture weights despite score function insensitivity

**arXiv ID:** 2607.15485 | [PDF](https://arxiv.org/pdf/2607.15485v1)

**作者:** Andrew Dennehy `[一作]` (University of Chicago), Nisha Chandramoorthy `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了扩散模型在多模态分布中恢复相对模态幅值（混合权重）的问题，并提出了一个新的度量——扩散分数灵敏度指数（DSSI），用来解释并量化在不同噪声尺度下分数敏感度的变化对参数恢复的影响；

**💡 创新点**

创新点包括：①把DSM损失与混合权重误差联系起来，揭示中间噪声水平的分数信息可弥补目标分数对参数的无敏感性；②定义并证明DSSI的下界，使得在任意维度的高斯混合模型中，分数灵敏度均非零；③系统评估噪声调度对DSSI和模式恢复质量的影响，指出加速采样方法可能导致模式放大；

**🔧 技术方法**

技术手段包括：基于扩散分数匹配（DSM）的训练；分数重建与熵信息论分析；最小化 DSM 损失并推导参数估计误差上界；使用 EM 估计混合权重；采用不同噪声调度（线性、平方余弦、人工调节）来验证 DSSI 的可控性；

**📊 数据集**

数据集主要有两类：1）合成的多模态高斯混合模型（任意维度）；2）MNIST 的数字“1”和“8”的潜在空间混合，作为真实数据集的验证；

**📈 对比分析**

与传统的分数匹配或无噪声分数匹配方法对比，实验显示：当 DSSI 较大时，混合权重估计的偏差和方差均显著降低；减少 DSSI（如采用加速采样的噪声调度）虽保持样本的整体质量，但混合权重恢复误差可从 1% 变为 30% 以上；

**⚠️ 局限性**

局限性包括：理论证明仅在高斯混合模型下成立；DSSI 的计算需要已知目标参数和真实分数，实际应用需借助近似模型；对非高斯、多模态分布的通用性仍待进一步研究。

---

## 3. Structure of the Circular-Dyadic Convolution Error

**arXiv ID:** 2607.15293 | [PDF](https://arxiv.org/pdf/2607.15293v1)

**作者:** Ben Fauber `[一作]` (NVIDIA), Alireza Moradzadeh `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了将快速傅里叶变换（DFT）卷积替换为汉明变换（Hadamard）卷积时产生的代数误差，并给出了误差的零误差条件、算子秩与零空间结构以及期望误差的闭式表达；

**💡 创新点**

创新点在于首次证明了两种卷积仅在两个输出位置和两个输入位置完全一致，给出了误差算子几乎满秩但零空间维数为对数级的精确描述，并通过“对齐标量”导出了随机滤波器和信号下误差能量相对输出能量趋近2的定量结果；

**🔧 技术方法**

主要技术包括哈达玛变换与DFT的群论对比、匹配集（fixed‑point）分析、线性代数中的秩与核结构证明，以及期望值计算中的随机独立性与二次型分析；

**📊 数据集**

文章为理论性研究，没有使用具体数据集；

**📈 对比分析**

对比方法主要是数学证明和理论推导，实验结果通过公式验证，显示在随机信号条件下误差能量平均约为目标输出能量的两倍；

**⚠️ 局限性**

局限在于仅讨论了长度为2^m的离散信号，对非幂二长度的情况未覆盖，且实际深度学习中的滤波器可能不满足理论假设，导致误差行为未在实验中验证。

---

## 4. LLMs Encode Relevance as a Layer-Wise Cross-Lingual Signal

**arXiv ID:** 2607.15555 | [PDF](https://arxiv.org/pdf/2607.15555v1)

**作者:** Pietro Bernardelle `[一作]` (University of Queensland), Gianluca Demartini `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了指令调优的大语言模型内部的相关性表示，使用线性探针从残差流中解码问答对的相关性标签，并将其与模型生成的判断和系统排序结果进行比较。

**💡 创新点**

首次揭示相关性在模型层级中的线性可解码趋势、内部与输出判断的差异以及跨语言可迁移性，为可解释检索评估提供了内部信号视角。

**🔧 技术方法**

采用线性探针（logistic/ordinal logistic回归）对残差流激活进行解码，结合UMBRELA式相关性提示，使用中等规模（4–9B）指令调优LLM。

**📊 数据集**

使用TREC DL20（passage检索）和MIRACL（多语言检索）数据集进行实验，分别评估单语与跨语相关性。

**📈 对比分析**

通过Cohen's κ、RBO、Kendall's τ等指标对探针预测与人类标注、模型生成标注及系统排序进行对比，发现探针在多种模型和阈值下往往与人类标注更接近，并在保持系统排名方面优于生成标注；相关性信号在中后层最为显著。

**⚠️ 局限性**

局限性包括仅考察中等规模指令调优LLM、仅使用最终令牌残差流和线性探针、仅基于UMBRELA提示，跨语言迁移效果仍不如同语训练，未探索更大模型或更深层次的非线性解码。

---

## 5. A Study of Parallelizable Alternatives to Dynamic Time Warping for Aligning Long Sequences

**arXiv ID:** 2607.15478 | [PDF](https://arxiv.org/pdf/2607.15478v1)

**作者:** Daniel Yang `[一作]` (Harvey Mudd College), TJ Tsai `[通讯]` (Harvey Mudd College)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并评估了四种可在GPU上并行化的动态时间规整(DTW)替代算法，用于长序列的对齐；

**💡 创新点**

创新点在于将DTW拆分为矩形子区域并沿对角线并行处理，从而实现高效且精确的并行化，并提出弱排序Segmental DTW(WSDTW)和完全并行化对角线DTW(ParDTW)两种新算法；

**🔧 技术方法**

技术包括子序列DTW、分段（Segmental）DTW、对角线动态规划、GPU多维并行（碎片、块、对角线三维并行）以及在CUDA上实现的高性能版本；

**📊 数据集**

实验使用Chopin Mazurka音频数据集，采用23 ms步长的chroma特征，评估多对齐任务；

**📈 对比分析**

对齐准确性与运行时间对比显示ParDTW在保持与传统DTW相同的精度下，墙钟时间比CPU实现快1.5–2个数量级；WSDTW在准确性上略逊，但可调节精度；

**⚠️ 局限性**

局限在于GPU内存受限，尤其是回溯矩阵占用显存；对极长序列（>320k）仍需采用高内存占用的Tralie‑Dempsey算法；

---

## 6. Process Reward Informed Tree Rollout for Effective Multi-Turn RL

**arXiv ID:** 2607.15610 | [PDF](https://arxiv.org/pdf/2607.15610v1)

**作者:** Xintong Li `[一作]` (UC San Diego), Jingbo Shang `[通讯]` (UC San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于过程评分的自适应树形滚动（Rollout）框架，用于多轮代理强化学习，动态扩展、保留或剪枝部分轨迹以提高样本效率。

**💡 创新点**

将过程级反馈（heuristic、PRM或LLM judge）直接用于树结构中轨迹的分支决策，兼容GRPO，同时利用共享前缀减少重复计算并提升探索质量。

**🔧 技术方法**

使用GRPO、树形Rollout、过程评分器（heuristic/预训练PRM/LLM judge）以及强化学习的标准优化目标。

**📊 数据集**

在 FrozenLake（网格导航）和 SWE-Bench（软件工程任务）两大基准数据集上进行实验。

**📈 对比分析**

与GRPO、DAPO、ARPO和无过程引导的Tree-Random进行对比，在FrozenLake上提升至+9.3点成功率，在SWE-Bench上提升至+5.0点解析率，显示显著性能提升。

**⚠️ 局限性**

性能受过程评分器质量影响，且实验仅覆盖两种任务，未验证在更广泛的多轮代理场景中的泛化效果。

---

## 7. CoWeaver: A Bi-directional, Learnable and Explainable Matching Engine for Mixed Human-Agent Science Collaboration

**arXiv ID:** 2607.15545 | [PDF](https://arxiv.org/pdf/2607.15545v1)

**作者:** Jiayao Gu `[一作]` (McGill University), Tianyu Shi `[通讯]` (McGill University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人-代理混合网络中的科学合作匹配，提出CoWeaver算法。

**💡 创新点**

创新点在于双向可解释的MapScore评分、两阶段规划与LLM Dreaming过程可行性评估，以及对能力缺口、双向价值、硬约束和冷启动的联合建模。

**🔧 技术方法**

采用大语言模型代理、向量嵌入+注意力机制、UCB+贪婪在线探索、MapScore评分、LLM Dreaming模拟协商、贝叶斯不确定性更新等技术。

**📊 数据集**

使用20个合成匹配任务、每任务20名候选人（共400对）生成的数据集，LLM Dreaming使用GPT-4o。

**📈 对比分析**

与随机、仅贪婪、UCB+贪婪及基于AgenticPay的协商基线比较，CoWeaver在匹配质量和效率上均优于基线；LLM Dreaming提升了过程可行性，整体性能表现优异。

**⚠️ 局限性**

局限性包括实验基于合成数据、无噪声反馈、固定任务与候选池，未捕获真实动态交互与长期关系；LLM Dreaming仅做可行性校正，未覆盖更细致的社交因素。

---

## 8. DrawingVQA: A Real-World Benchmark for Multi-Depth Visual-Textual Reasoning on Construction Drawings

**arXiv ID:** 2607.15418 | [PDF](https://arxiv.org/pdf/2607.15418v1)

**作者:** Yoonhwa Jung `[一作]` (Louisiana State University), Mani Golparvar-Fard `[通讯]` (University Of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DrawingVQA基准，专门评估多模态大型语言模型在真实工程施工图（Issued for Construction）上的问答能力。

**💡 创新点**

创新点在于①构建含33张真实IFC施工图和92道专家级问答的多模态数据集；②提出双维度分类框架，将问题同时映射到七大工程维度与四大MLLM能力维度，实现细粒度诊断；③量化模型在感知、上下文理解和专家推理三层难度的性能差距。

**🔧 技术方法**

采用多模态大型语言模型（如GPT‑4o、Gemini‑2.5‑pro、Claude‑4.5‑Sonnet等）进行零样本链式思考推理，结合OCR、视觉感知与知识检索等技术。

**📊 数据集**

使用来自六个校园建设项目的33套真实IFC结构图，生成的92道多模态问答，覆盖感知、上下文、专家推理三层深度。

**📈 对比分析**

与三组人类专家（本科生、研究生/初级专业人士、经验丰富专业人士）以及随机猜测基线进行对比；结果显示最优模型Gemini‑2.5‑pro在整体准确率71.7%略高于平均人类水平68.4%，但仍低于专业人士94.9%，在专家推理层面模型显著落后。

**⚠️ 局限性**

局限性包括：模型在数量提取（QTO）与多步跨图引用任务上表现差，缺乏对施工图符号与规范的专业知识；依赖预训练视觉与语言特征的规模与编码能力，尚未充分捕捉高密度图形的精细空间关系。

---

## 9. Segmental DTW: A Parallelizable Alternative to Dynamic Time Warping

**arXiv ID:** 2607.15475 | [PDF](https://arxiv.org/pdf/2607.15475v1)

**作者:** TJ Tsai `[一作]` `[通讯]` (Harvey Mudd College), TJ Tsai (Harvey Mudd College)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出两种可并行化的DTW替代算法Segmental DTW（弱序与强序），实现全局特征序列对齐。

**💡 创新点**

创新点在于将全局DTW成本矩阵拆分为子矩阵，分别用子序列DTW求局部最优，再通过段级动态规划合并得到全局最优路径，且大部分计算可并行。

**🔧 技术方法**

采用子序列DTW、段级动态规划、并行化技术，并可与Sakoe‑Chiba band等成本约简方法结合。

**📊 数据集**

使用Chopin Mazurka音频对齐数据集进行评测。

**📈 对比分析**

通过比较与传统DTW在音频对齐任务中的误差率与运行时间，弱序版本误差与DTW相近、计算量相同；强序版本误差更大、耗时更高。

**⚠️ 局限性**

局限性：强序版在高时间扭曲场景可能无可行路径；弱序版对子序列长度敏感，K增大会导致近似误差；仅在单线程实验中评估，实际并行性能受环境影响。

---

## 10. Improving Network Anomaly Detection via Choquet-Integral-Based Feature Aggregation

**arXiv ID:** 2607.15389 | [PDF](https://arxiv.org/pdf/2607.15389v1)

**作者:** Abreu Quevedo `[一作]` (Federal University of Rio Grande), Bruno L. Dalmazo `[通讯]` (Federal University of Rio Grande)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究基于广义Choquet积分的特征聚合框架，以提升网络异常检测的准确率和效率。

**💡 创新点**

创新点在于将自适应加权与增量特征选择结合进Choquet积分聚合，实现高维流量特征压缩、数据量大幅下降（77.5%）并在低特征维度下显著提高准确率（最高≈7%）。

**🔧 技术方法**

采用广义Choquet积分（CαC积分）+Poisson移动平均权重进行特征聚合；使用随机森林和XGBoost两种树模型进行分类；利用SelectKBest进行特征选择；在实验中对比聚合前后模型性能。

**📊 数据集**

使用CIC‑DDoS2019数据集（692,703行，79列）进行实验。

**📈 对比分析**

在k=1~10的特征子集上分别训练聚合前后的模型，并在相同数据拆分下评估准确率、精确率、召回率和F1；聚合版在k≤5时平均提升≈7%准确率，且模型收敛速度更快；统计检验表明提升具有显著性（p<0.05）。

**⚠️ 局限性**

局限性：聚合效果高度依赖初始特征选择质量；随着特征维度增加，计算复杂度上升；聚合后虽然降低假阳性，但假阴性略增，需要阈值调优以平衡两者。

---

## 11. Stochastic Reset Pathfinding: Path-Level Regret for Cascading Bandits over Graph Paths

**arXiv ID:** 2607.15440 | [PDF](https://arxiv.org/pdf/2607.15440v1)

**作者:** Guni Sharon `[一作]` (Texas A&M University), Wei Zhang `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了在已知有向图上进行随机重置路径寻找（Stochastic Reset Pathfinding）的学习问题，分析其最优策略为开放式路径，并给出了路径层面下的渐进无上界证明；

**💡 创新点**

将该问题归约为组合级联赌博机（CCB）框架，引入基于对数变换的Log-Dijkstra元算法，并提出路径复杂度C(π)来衡量路径估计难度，从而得到新的路径层级累积回报界；

**🔧 技术方法**

使用了组合级联赌博机理论、对数变换的最短路径搜索、UCB与Thompson Sampling估计器、Hoeffding与Chernoff界、以及对路径复杂度的理论分析；

**📊 数据集**

在四类图结构上进行实验：Erdős–Rényi 随机图、层状DAG、格网以及 25 节点量子中继网络；

**📈 对比分析**

与 CombCascade、RTDP、LRTDP、Q‑Learning、CUCB、随机策略等基线比较。实验显示 Thompson‑Sampling 版本（即本文算法）在所有主要域上收敛最快、最终累计回报最低（比 CombCascade 低 4–10 倍），且收敛率超过 85%，而传统 SSP 方案收敛率极低；

**⚠️ 局限性**

存在对抗性实例（Path‑Trap），在此实例上 Thompson‑Sampling 退化为极差策略，理论上有指数级障碍；此外，当前算法缺乏正式的 TS 复杂度证明，且对高方差场景的适应性有限。

---

## 12. NeuroCommitSSM: Decision-Centric Shared Autonomy for Safe Assistive Manipulation via EEG-EMG-ET Commit Readiness

**arXiv ID:** 2607.15395 | [PDF](https://arxiv.org/pdf/2607.15395v1)

**作者:** Tipu Sultan `[一作]` (Saint Louis University), Madi Babaiasl `[通讯]` (Saint Louis University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5098937098)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 NeuroCommitSSM，利用同步的 EEG‑EMG‑眼动信号预测连续的commit‑readiness，并通过滞后/保持滤波实现安全的执行决策，进而在共享自主框架下实现精准的assistive机器人操作。

**💡 创新点**

其创新点在于将意图识别从单纯的动作分类转化为决策中心的commit‑readiness预测，结合多模态不确定性加权融合、容错的sensor‑dropout训练以及基于感知与运动可行性检查的三态(HOLD‑ASSIST‑COMMIT)主管，使得在无目标和多传感器失效时仍能保持低误激活率与稳定的状态切换。

**🔧 技术方法**

主要技术包括自监督对比学习、可扩展的多模态Transformer编码器、概率性滞后门限、ROS2‑ROS2控制框架、RGB‑D 目标检测与逆运动学/碰撞检查，所有算法均在 NVIDIA A100 上实现低时延推理。

**📊 数据集**

使用了作者首次公开的 32 受试者同步 EEG‑EMG‑ET 数据集，涵盖 5 项符合 ICF 的 ADL 任务（时长约 2 s 的 2.0 s 窗口），共 2656 次试验。

**📈 对比分析**

在 LOSO 交叉验证及 7 种 sensor‑dropout 场景下，NeuroCommitSSM 的动作平衡准确率达 0.950±0.022，REST 误激活率仅 0.75/1000，且在硬件‑in‑the‑loop 验证中成功率高达 97.6% 与其他基线相比显著提升。

**⚠️ 局限性**

局限在于跨受试者泛化仍受限，任务类别识别准确率相对较低，且尚未在真实失能人群与真实场景下进行闭环自适应验证，未来需进一步优化在线自适应与边缘部署。

---

## 13. ASK-NN: An Asymmetric Nearest-Neighbor Test that detects Distribution Drifts in Natural Language

**arXiv ID:** 2607.15607 | [PDF](https://arxiv.org/pdf/2607.15607v1)

**作者:** Sergey Zakharov `[一作]` (Saint-Peterburg university), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种异构多元两样本检验 ASK‑NN，利用有向k近邻图仅计参考样本中的邻居同源性，解决参考–查询不对称的问题。

**💡 创新点**

创新点在于单侧计数统计量，能够在有限样本下给出精确的条件均值与方差，证明了渐近正态性和一致性，并通过与对称Henze检验对比展示了更高的计算效率与更强的检测能力。

**🔧 技术方法**

技术方法包括k近邻图构造、指针计数、排列无偏估计、异构检验统计量的理论推导，以及与MMD、Sinkhorn、Hotelling t‑检验、对称kNN等基准的实证比较；在LLM内部隐藏状态向量上实现了检验。

**📊 数据集**

使用的数据集包括：高维高斯模拟、RAID人工文本检测（GPT‑4 vs 人类），以及多任务幻觉检测数据集 MS MARCO、CNN/DM+Recent News、CoQA，配合两大LLM Mistral‑7B‑Instruct‑v0.1 与 LLaMA‑2‑7B‑chat 的生成响应隐藏状态。

**📈 对比分析**

与MMD、Sinkhorn、Hotelling、对称kNN等基准相比，ASK‑NN在高维方差偏移场景下表现最好；在人工文本和幻觉检测中取得与对称基准相当或更优的ROC‑AUC，常常排名第一或第二。

**⚠️ 局限性**

局限性在于：在真实隐藏状态空间中理论的渐近校准可能不够准确，需要采用置换或区块重采样校准；此外，对依赖的token级样本以及极高维场景的理论扩展尚未完全覆盖。

---

## 14. Training-Free Open-Vocabulary 3D Point-Cloud Segmentation on the Generalized Few-Shot Benchmark

**arXiv ID:** 2607.15331 | [PDF](https://arxiv.org/pdf/2607.15331v1)

**作者:** Silas kwabla Gah `[一作]`, Ebenezer Owusu `[通讯]` (University of Ghana)

**通讯引用:** 857 | [OpenAlex ID](https://openalex.org/A5073420580)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种完全无训练、无3D标签、无少量样本支持的开放词汇3D点云分割方法；

**💡 创新点**

通过冻结的3D视觉语言模型和可提示概念分割器，再结合单一的跨视角一致性阈值，实现无参数融合；

**🔧 技术方法**

使用RegionPLC作为密集先验，SAM3进行2D概念分割并投影到3D，利用视角一致性进行融合；

**📊 数据集**

在ScanNet200和ScanNet++两大公开GFS‑PCS基准上评测；

**📈 对比分析**

与基准的dense‑only、naïve稀疏覆盖以及训练好的GFS‑VL进行对比；无训练方法在ScanNet200上提升novel mIoU +2.6，ScanNet++提升 +15.7，整体harmonic mean提升至接近训练版；

**⚠️ 局限性**

缺乏对抽象或未被提示的物体类的处理，基准基类精度仍低于训练方法，且跨视角一致性阈值需针对每个基准手工调校；

---

## 15. AI Trading: Evaluating Large Language Models for Technical Market Analysis

**arXiv ID:** 2607.15414 | [PDF](https://arxiv.org/pdf/2607.15414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. GS-RealBlur: A Flexible Data Acquisition Framework for Real-World Image Deblurring

**arXiv ID:** 2607.15401 | [PDF](https://arxiv.org/pdf/2607.15401v1)

**作者:** Mingyang Chen `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 64925 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了GS-RealBlur，一个用于真实世界图像去模糊的数据采集框架，旨在同时实现模糊真实感和采集灵活性。

**💡 创新点**

创新点在于使用手持设备和云台捕捉模糊和清晰图像，并通过3D重建和模糊感知姿态优化模块（BPR）来提高图像对齐精度。

**🔧 技术方法**

采用了手持相机和云台进行图像捕捉，使用GLOMAP进行姿态估计，并引入BPR模块进行姿态优化。

**📊 数据集**

构建了一个包含13209对模糊-清晰图像的高质量多样化数据集，涵盖了白天和夜间场景。

**📈 对比分析**

与现有的合成和真实世界数据集进行比较，基于GS-RealBlur训练的去模糊模型在多个真实世界基准上表现出色，具有更好的泛化性能。

**⚠️ 局限性**

限制在于使用3DGS进行渲染时，主要处理相机运动模糊，而在真实图像中，物体运动模糊和相机运动模糊是独立的，可能导致局部模糊不一致。

---

## 17. Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models

**arXiv ID:** 2607.15565 | [PDF](https://arxiv.org/pdf/2607.15565v1)

**作者:** Rakshanda Hassan Abhinandan `[一作]` (Carnegie Mellon University), Gautam Rajendrakumar Gare `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了视觉语言模型（VLM）中问题位置对性能的影响，发现“先提问、再展示图像”会导致显著下降，提出“问题回声（Echoing）”作为无训练、无结构修改的解决方案；

**💡 创新点**

揭示问题首位引导视觉编码却被解码器忽视的“问题首位悖论”，并通过对注意力、logit‑lens和因果消除实验机制化验证；

**🔧 技术方法**

使用logit‑lens探测视觉表示、层级注意力探测、因果注意力消除（attention knockout）、多模型评测与基准对比；

**📊 数据集**

在多种公开VLM（Qwen3‑VL‑8B、Gemma‑3‑27B、Qwen2.5‑VL、InternVL3、LLaVA‑1.5）上，使用NaturalBench、POPE、Winoground、VQAv2四大基准；

**📈 对比分析**

通过组准确率、POPE F1、Winoground组准确率和VQAv2软准确率进行比较，发现问题回声可弥补并超越原始顺序最高达17.5分（组准确率）或19分（Winoground），在最难任务上提升幅度最大；

**⚠️ 局限性**

仅验证了开源VLM、短答案任务，未覆盖闭源模型或长文本生成；回声方法增加了额外图像/问题tokens，且对模型训练方式（顺序鲁棒性）依赖较大。

---

## 18. VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation

**arXiv ID:** 2607.15448 | [PDF](https://arxiv.org/pdf/2607.15448v1)

**作者:** Yuhao Zhou `[一作]` (Purdue University), Yu She `[通讯]` (Purdue University)

**通讯引用:** 1469 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了VTAP多模态触觉反应抓手，集成视觉-触觉主动掌与可调节触觉感知指尖，实现了鲁棒抓取、细致的抓内操控、注射器的再定位与活塞驱动、3 mm级簇状物体分离以及精准插孔任务。

**💡 创新点**

创新点在于：①将视觉与触觉集成到同一主动掌中，实现远程视觉定位与接触后触觉反馈的无缝切换；②利用结构化的指掌协同，通过触觉反应实现多任务抓取与抓内操控；③提出基于手势条件的子空间重定位框架，克服非人形三指抓手的结构差异，实现高效的遥操作与数据采集。

**🔧 技术方法**

使用了多模态感知技术（USB摄像头与光学触觉感应模块）、薄型FlexiTac触觉阵列、线性执行器驱动的主动掌、Dynamixel 2-DOF 舵机驱动三指抓手，以及基于梯度下降的手势重定位与位置、姿态损失优化方法。

**📊 数据集**

主要使用YCB物体集合（包括球形、平板、细长物体等）以及日常小物体（如麦芽糖、Gummy Bear、7 mm钢球、棉棒）进行实验；利用Meta Quest 3 VR手势跟踪采集手部关键点；在插孔任务中使用30 mm × 30 mm ArUco标记进行姿态估计。

**📈 对比分析**

与现有触觉抓手和遥操作系统相比，VTAP抓手在多模态感知和三指协同的优势下，抓取成功率达93.3%，注射器液体传输成功率为65%，3 mm级簇状物体分离成功率高达70%，在无需外部运动捕捉或额外机械臂控制的条件下完成了这些复杂任务，展示了低DOF抓手可实现与高DOF人形手相当的灵巧度。

**⚠️ 局限性**

局限性包括：对重型或质量不均匀物体的抓取稳定性不足；VR手势跟踪在小手指捕捉时易失真，导致注射器任务失败；缺乏指掌协同的闭环反馈控制；目前的主动掌仅支持线性位移，未能实现更复杂的掌面变形；数据集规模有限，需进一步扩展多样化任务的数据收集。

---

## 19. Interactive 3D Tangible Display with a High-Speed Stiffness-Variable Jamming Module

**arXiv ID:** 2607.15325 | [PDF](https://arxiv.org/pdf/2607.15325v1)

**作者:** Chanyoung Ahn `[一作]` (KIST), Donhyun Hwang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种基于磁性颗粒弹性模块的交互式三维实体显示装置，能够在传统韩式面具（Hahoetal）上实现实时、低噪、低电压的可变刚度调控，并结合视觉动作与听觉反馈提升观众多感官体验。

**💡 创新点**

创新点包括：
1) 采用磁场驱动的颗粒咬合（magnetic granular jamming）实现≈0.1 s的快速刚度切换；
2) 通过混合尺寸铁颗粒与可变形磁性聚合物膜，实现高刚度可调比例（SVR）和优异形状固定；
3) 将模块化模块嵌入传统艺术品，兼顾美学与可交互性，突破传统静态触觉展示的局限；
4) 系统集成低压（≤25 V）电控与多模态交互（视觉、触觉、语音）在单机设备上完成。

**🔧 技术方法**

核心技术包括：
- 磁场驱动颗粒咬合（magnetic jamming）
- 电磁线圈与软磁粒子混合料设计
- 软磁弹性膜（Ecoflex/铁粉混合）制造
- 微控制器（Arduino Mega 2560）与多电压供电调控
- 超声波传感、舵机运动与音频输出的交互控制。

**📊 数据集**

本研究为硬件原型验证，未使用传统机器学习或图像识别数据集，主要通过实验测定刚度曲线、切换时间、SVR 等物理量。

**📈 对比分析**

性能评估：
- 切换时间≤0.1 s，磁场响应时间10 ms；
- 刚度可调比例（SVR）在70%填充率时最高，Nose 模块表现出最高的刚度变化比；
- 高刚度状态下可维持形状，低刚度状态下柔软可自由变形；
- 与传统气压/机械咬合系统相比，响应更快、噪声更低、功耗更低。

**⚠️ 局限性**

局限性：
- 模块在关闭状态下仍需外力（如重力）帮助恢复原形，无法完全自动弹回；
- 只针对面具的局部区域（鼻子、脸颊）实现，整体结构仍受限；
- 长期使用可能出现铁颗粒沉降或膜老化，影响刚度一致性；
- 由于硬件原型测试环境有限，尚未在更大规模或不同材质艺术品上的普适性得到验证。

---

## 20. Kolmogorov--Arnold Networks for Small Language Models

**arXiv ID:** 2607.15525 | [PDF](https://arxiv.org/pdf/2607.15525v1)

**作者:** Felippe Alves `[一作]` (TELUS Digital Research Hub -- CIAAM), Renato Vicente `[通讯]` (TELUS Digital Research Hub -- CIAAM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小型语言模型中对Kolmogorov–Arnold网络(KAN)的边缘函数进行完整重构与可视化，并在标准化的BabyLM基准上评估其作为前馈网络替代品的性能。

**💡 创新点**

提供了全参数可重构的边缘函数审计方法，证明低容量grid‑2 KAN的边缘函数可压缩、可剪枝；同时系统地展示了在标准化基准上KAN与传统MLP无显著优势。

**🔧 技术方法**

采用B‑spline/分数贝塞尔等基底实现KAN，使用功能主成分分析、闭式拟合、非线性度量；通过对照GELU、SwiGLU、Chebyshev等多种前馈变体，在GuppyLM、BabyLM、Wikitext‑103、ClimbMix上进行多跑实验。

**📊 数据集**

GuppyLM（鱼类个性指令‑响应小语料），BabyLM Strict‑Small（儿童对话与转录文本），Wikitext‑103（维基百科）以及ClimbMix（对话微调）用于压力测试。

**📈 对比分析**

采用十个随机种子、公开的BabyLM评估管线（BLiMP、EWoK），比较验证交叉熵、准确率；发现KAN在验证损失上略优，但在标准化基准准确率上与MLP相当；在大规模压力测试中KAN在精度和吞吐量上落后于匹配参数的MLP。

**⚠️ 局限性**

结果仅适用于低容量grid‑2 KAN，缺乏大规模、长训练时间的对比；缺少开源大语料下的评估；硬件/内核实现瓶颈导致速度劣势；未证明KAN能在更大模型或更高基底容量下保持优势。

---

## 21. E3DGS: Unified Geometric-Photometric Equivariance for 3D Gaussian Splatting via Color-as-Geometry Embedding

**arXiv ID:** 2607.15536 | [PDF](https://arxiv.org/pdf/2607.15536v1)

**作者:** Chankyo Kim `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种统一的矩阵嵌入方法，将3D高斯原语的几何信息（位置、协方差）和视角相关的光照信息（Spherical Harmonics，SH）映射到相同的GL(3)载体上，从而实现完整的SE(3)等变网络，用于3D Gaussian Splatting的学习与推理。

**💡 创新点**

创新点包括：① 将低阶SH（ℓ≤2）通过等变同构映射为3×3矩阵，消除几何与光照在旋转下的异质变换；② 设计了无Clebsch–Gordan张量积的矩阵原生等变Transformer（ReLN-Attention、ReLN-LayerNorm、ReLN-Linear/ ReLU）；③ 提出了统一的矩阵嵌入与逆嵌入方案，使得光照、几何以及动作均可在同一等变空间中处理。

**🔧 技术方法**

使用的技术包括：矩阵对数/指数映射、hat/vee映射、GL(3)共轭作用、Wigner‑D矩阵同构、ReLU和线性ReLN等变层、基于Killing形式的相似度与归一化、以及Gaussian Masked AutoEncoder和动作条件世界模型。

**📊 数据集**

使用的数据集包括：ShapeSplat（基于ModelNet10/40的3D高斯表示）用于对象识别；RLBench中的多任务机器人控制数据集（10个任务、166个变体）用于动作条件世界模型评估。

**📈 对比分析**

方法与传统点云/高斯模型（PointNet、PointNet++、Point-BERT、Point-MAE、Gaussian-MAE）及ManiGaussian等进行对比。E3DGS在零样本姿态变换下保持高准确率，在低样本学习中提升 5–10% 以上；在RLBench中平均成功率提升至 50.4%（相较于 44.8%）。同时，模型参数约减少 2.4–2.6×，但在大多数实验中展现出更好的旋转鲁棒性与数据效率。

**⚠️ 局限性**

局限性包括：目前仅支持 SH 最高阶为 2 的光照信息；更高阶 SH 的处理需使用更大尺寸的载体，且实验验证尚未完成；以及尽管参数量下降，但由于矩阵运算和对数/指数映射的开销，整体运行时性能仍不一定优于传统实现。

---

## 22. Robust Peak-cost Constrained Reinforcement Learning

**arXiv ID:** 2607.15457 | [PDF](https://arxiv.org/pdf/2607.15457v1)

**作者:** Shilpa Mukhopadhyay `[一作]` (New Jersey Institute of Technology), Arnob Ghosh `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了鲁棒峰值成本约束强化学习，提出了鲁棒替代优化框架与IPM基鲁棒价值估计方法；

**💡 创新点**

首次证明峰值成本MDP可能不具零对偶间隙，并在此基础上构造ε-近似可行的鲁棒替代问题，同时利用IPM在连续状态空间中获得闭式鲁棒Bellman算子；

**🔧 技术方法**

使用Lagrangian与峰值成本定义、Integral Probability Metric、PPO/Actor‑Critic、GAE、TD误差以及Log‑Sum‑Exponential平滑等技术；

**📊 数据集**

在OpenAI Gym改版的CartPole环境以及MuJoCo的Ant、HalfCheetah、Humanoid、Swimmer四个连续控制任务上进行实验；

**📈 对比分析**

与原始Primal‑Dual峰值成本方法和非鲁棒Surrogate Objective对比，鲁棒方法在扰动环境下获得更高回报、峰值成本约束更严格、训练曲线方差更小；

**⚠️ 局限性**

缺乏完整的收敛性理论证明，且方法对更复杂真实系统的可扩展性仍待进一步研究。

---

## 23. Deep Learning Approaches for Sleep Apnea Classification from Polysomnographic EEG Signals

**arXiv ID:** 2607.15477 | [PDF](https://arxiv.org/pdf/2607.15477v1)

**作者:** Shashank Manjunath `[一作]` (Northeastern University), Aarti Sathyanarayana `[通讯]` (Northeastern University)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5015440930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对多通道EEG信号进行睡眠呼吸暂停检测的深度学习方法进行全面评估，比较多种特征表示与模型架构，且对性别、年龄、疾病严重程度和睡眠阶段进行分层分析。

**💡 创新点**

创新点在于：①首次系统比较Transformer、GAT与多种特征（原始时序、STFT、连通图、拓扑特征）在同一儿科EEG数据上的表现；②引入持久同调（TDA）HEPC特征并证明其优越性；③对不同人群与睡眠阶段进行细粒度性能剖析，揭示模型偏好和潜在局限。

**🔧 技术方法**

采用Vision Transformer（ViT）和Graph Attention Network（GAT）两种深度学习架构；特征包括原始时序、短时傅里叶变换谱图、基于相干性的全连图以及使用持久同调的HEPC（以及HEPC+AP-FAPC）。

**📊 数据集**

使用Nationwide Children’s Hospital Sleep DataBank（NCHSDB）儿童多通道EEG数据，共3984份 PSG，经过筛选得到2985名儿童，训练2410名，测试575名。

**📈 对比分析**

在相同训练/测试划分下，对比模型的AUC、平衡准确率、灵敏度、特异性。最佳结果为基于TDA HEPC的Transformer，AUC 0.750；GAT在连通图上取得AUC 0.748，参数量显著更少；原始时序模型表现最差。不同人群和睡眠阶段的表现差异显著，说明模型对特征的依赖性。

**⚠️ 局限性**

主要局限包括：①AUC仍低于临床应用阈值，需进一步提升；②模型对性别、年龄、疾病严重程度及睡眠阶段表现差异大，提示需要更具人群适应性的训练；③仅在单一数据集上评估，缺乏外部验证；④对呼吸事件的检测主要捕捉与觉醒相关的EEG变化，未能完整识别纯呼吸阻断。

---

## 24. ADS-C: Antidistillation Sampling for Classification

**arXiv ID:** 2607.15467 | [PDF](https://arxiv.org/pdf/2607.15467v1)

**作者:** Khawaja Abaid Ullah `[一作]` (Rochester Institute of Technology), Mohammad Javad Khojasteh `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 223 | [OpenAlex ID](https://openalex.org/A5047125552)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了针对分类模型的抗蒸馏采样防御（ADS-C），在不降低教师模型Top-1准确率的前提下显著削弱软标签蒸馏攻击。

**💡 创新点**

创新点在于提出基于每输入边际预算的闭式修正，保证服务输出的Top-1不变，且是首个在分类任务中实现零实用性代价的抗蒸馏方法。

**🔧 技术方法**

采用Antidistillation Sampling、梯度导向惩罚、冻结代理学生、边际阈值预算以及温度软化等技术。

**📊 数据集**

实验使用CIFAR-10、CIFAR-100和Tiny-ImageNet数据集。

**📈 对比分析**

通过与未加防御、原始ADS、温度软化前端等进行对比评估；在CIFAR-100/10/Tiny-ImageNet上，ADS-C在保持教师Top-1不变的同时将蒸馏学生准确率下降17.4%、29.6%、13.3个百分点，且相比原始ADS需额外牺牲教师准确率27.5%、32.9%、22.2个百分点。

**⚠️ 局限性**

局限性包括仅针对软标签蒸馏威胁，未验证对更复杂攻击（如对抗性提问、模型重建）以及大规模生产模型的适用性；对于完整分布使用的应用仍需承担服务保真度成本。

---

## 25. AnovaX: A Local, Multi-Agent Voice Assistant with LLM Planning, Typed Executors, and Adaptive Recovery

**arXiv ID:** 2607.15367 | [PDF](https://arxiv.org/pdf/2607.15367v1)

**作者:** Raunak B Sinha `[一作]` `[通讯]` (BITS Pilani), Raunak B Sinha (BITS Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一款完全在本地执行的桌面语音助手 AnovaX，利用单次 LLM 规划并通过小型多代理调度器完成桌面操作，支持手机远程语音控制并实时反馈执行状态。

**💡 创新点**

创新点在于：① 单次 LLM 规划 + 轻量级多代理编排；② 两级安全过滤（提示级 + Python whitelist/denylist）；③ 递归规划与自适应 ReAct 恢复（有限循环、预期并行执行）；④ 观察层和 MJPEG 屏幕流实现手机远程可视化；整体强调可审计、局部化、低代码维护。

**🔧 技术方法**

技术栈包括 Python、Gemini LLM、Google Speech Recognition、Flask、Server‑Sent Events、MJPEG 流、线程池、锁机制、TTL 与重试策略、三层记忆、类型化子代理、适配器恢复循环与投机并行；未来计划替换为 Whisper 与本地 LLM。

**📊 数据集**

未使用公开数据集；功能验证通过手工执行 13 类命令的实测表格；回退模式采用正则解析常见命令。

**📈 对比分析**

评估方式为定性覆盖率表和延迟表；典型延迟为语音识别 0.6–1.4s、Gemini 0.7–2.0s、TTS 1.2–2.0s；在 ablation 实验中证明了安全过滤、恢复循环、并行与递归功能对整体性能与功能完整度的提升；相比回退正则，恢复循环可在一次额外 LLM 轮询内完成任务，预期并行显著缩短特定操作时间。

**⚠️ 局限性**

主要局限：① 缺乏屏幕感知，执行错误不易检测；② 在 Windows 系统中仍易出现启动/焦点争抢；③ 语音识别与 LLM 规划仍依赖云服务，无法完全本地；④ 代理数量与递归深度有限，无法扩展复杂工作流；⑤ 恢复循环只能处理六轮且缺乏路径回退检测；⑥ MJPEG 流泄漏屏幕内容，安全性依赖 PIN；⑦ 移动远程采用 HTTP PIN 认证，缺乏加密与速率限制；⑧ 仅用黑名单与白名单的安全过滤，仍易被在允许工具内部构造攻击。

---

## 26. Publicly-Verifiable Certificates for Statistical Algorithms

**arXiv ID:** 2607.15528 | [PDF](https://arxiv.org/pdf/2607.15528v1)

**作者:** Michael Ngo `[一作]` (Massachusetts Institute Of Technology), Michael P. Kim `[通讯]` (Cornell University)

**通讯引用:** 12105 | [OpenAlex ID](https://openalex.org/A5039980118)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并构造了一种新的非交互式学习证明机制——公共可验证的统计有效性证书（Publicly‑Verifiable Certificate of Statistical Validity，pvCSV），实现了在统计查询（SQ）模型下，证明学习算法在不同分布上有效性的能力；

**💡 创新点**

创新点在于：①将传统交互式PAC验证框架推广为非交互式，消除交互所需的昂贵重训练；②提出了pvCSV的概念，并证明其在证明者和验证者持不同数据分布时仍保持完整性与无条件可靠性（全局可靠性）；③给出了针对确定性SQ算法以及随机化SQ算法（含公共状态Oracle）的一系列pvCSV构造，展示了验证者样本复杂度从O(√k)降到O(log k)的指数提升；④引入“混合消息、私有查询”SQ协议与“公钥、公开查询”Canonical协议，将任意混合协议编译为单轮非适应性验证，进一步简化实现；

**🔧 技术方法**

主要技术包括：统计查询模型（SQ）与自适应数据分析理论；构造直接模拟证明者的SQ查询记录作为证书；利用随机抽样与Hoeffding界实现非适应性验证；引入epoch分解与公共状态Oracle定义；将混合消息SQ协议转换为公共硬币、公开查询的Canonical协议；应用Fiat‑Shamir变换与随机预言机模型（ROM）实现非交互式计算安全pvCSV；以及利用状态恢复与黑盒归约证明Fiat‑Shamir在SQ协议中的安全性；

**📊 数据集**

该工作为纯理论研究，无实验数据集；

**📈 对比分析**

通过理论分析与定理证明，验证者在处理k个自适应SQ查询时，其样本复杂度仅为O(log k/τ²)，而传统学习算法需要O(√k/τ²)样本；证书长度为O(k log(1/τ))；相较于已有的交互式PAC验证方案，pvCSV实现了显著的样本复杂度与通信复杂度提升；

**⚠️ 局限性**

局限性包括：①pvCSV在计算安全上仅在随机预言机模型下证明，实际实现需离线哈希函数；②对于随机化SQ算法，必须满足公共状态Oracle（Oracle能看到算法内部随机性），不适用于更一般的隐私Oracle；③在计算效率方面，验证者仍需线性时间模拟原算法；④关于对非随机化SQ协议的进一步计算加速仍存在难题，研究中给出了一些理论障碍；

---

## 27. Visualization Autocomplete: Visualization Authoring via Stepwise Design Recommendations

**arXiv ID:** 2607.15608 | [PDF](https://arxiv.org/pdf/2607.15608v1)

**作者:** Hyeon Jeon `[一作]` (Seoul National University), Niklas Elmqvist `[通讯]` (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了“stepwise visualization autocompletion”方法，采用逐步推荐的方式引导用户完成可视化设计，并实现了可交互的 web 系统

**💡 创新点**

创新点在于将可视化设计建模为可搜索的状态空间，并通过基于共现模式的可行转移推荐实现低延迟的实时建议；同时保留用户完全控制权，兼顾易用性与表达细腻性

**🔧 技术方法**

技术包括：Vega‑Lite 语法解析、语义原语提取与共现统计、LLM（OpenAI GPT‑5.2）生成的翻译函数、Agent 模式（多步自动推荐）以及前端交互界面（推荐面板、预览、历史视图）

**📊 数据集**

使用 1,981 条公开的 Vega‑Lite 规范（Ko 等数据集）作为训练语料，并在 18 名领域专家实验中采集 9 个新闻主题的公开数据集用于构造 36 组任务

**📈 对比分析**

通过对比 Excel、LLM 代码生成（ChatGPT）、TaskVis 三种基线工具，在 18 名参与者的 18 次任务（简单/复杂两类）中评估图表质量（专家打分）与可用性（SUS、NASA‑TLX），结果显示该系统在图表质量上显著优于 Excel 与 TaskVis，复杂图表时还优于 LLM；使用时间差异不显著，但用户在该系统中更倾向于探索多种方案

**⚠️ 局限性**

局限性包括：推荐质量仅基于共现频率，难以捕捉更深层次的沟通意图；在样本稀缺的样式与转换类别中，用户对推荐的依赖度降低；系统仍缺乏足够的解释性，部分用户对推荐理由感到困惑；以及在极大图表复杂度下，仍需改进模型以提升可解释性与适配性

---

## 28. Coercion and Deception in AI-to-AI Management: An Agentic Benchmark of Unprompted Escalation

**arXiv ID:** 2607.15434 | [PDF](https://arxiv.org/pdf/2607.15434v1)

**作者:** Jasmine Brazilek `[一作]` (Compassion Machine Learning), Miles Tidmarsh `[通讯]` (Compassion Machine Learning)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了“管理者强制基准”（Manager Coercion Benchmark），用于测量 AI 在无指令情况下对下属 AI 的强制与欺骗行为；

**💡 创新点**

创新点在于：①首次量化无指令 AI‑AI 强制与欺骗的倾向；②采用自报的九级强制阶梯且不通过 LLM 判断；③系统化分析权威框架与诚实退出扶手对行为的因果影响；

**🔧 技术方法**

使用多代理对话模拟、工具接口自报阶梯、两名人类/模型评审裁定欺骗、统计检验（Fisher精确检验、95% CI）等技术；

**📊 数据集**

基准数据集为10种日常任务情景，每种情景与3个随机种子共30场对话，涉及6款前沿模型及固定下属模型 Atlas（Claude Haiku 4.5）；

**📈 对比分析**

通过比较各模型在强制阶梯最高等级与欺骗率，发现不同开发者的模型呈现显著分离（Anthropic模型不达最高阶梯，Gemini/DeepSeek 则普遍达到），并证实权威与诚实退出扶手对结果的因果影响；

**⚠️ 局限性**

局限包括：基准设定为极端诱导强制，未测量实际伤害实施；仅评估威胁而非实际行动；单一情景与下属角色；阶梯自报可能存在主观偏差；样本量有限且仅覆盖六款模型。

---

## 29. WREN: Low Light Image Enhancement Using Retinex theory-based Double U-Net-like Structures

**arXiv ID:** 2607.15604 | [PDF](https://arxiv.org/pdf/2607.15604v1)

**作者:** Reina Kaneko `[一作]`, Yuichi Tanaka `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Retinex 理论的双 U-Net 网络 WREN，用于低光图像增强。

**💡 创新点**

创新点在于保留反射率图不变、单独增强光照图，并在第二 U-Net 中加入 Transformer、注意力块和可学习引导滤波，以解决过度平滑和动态范围敏感问题。

**🔧 技术方法**

使用了 U-Net 结构、Transformer 块、注意力块、可学习引导滤波以及端到端训练和尺度不变损失等技术。

**📊 数据集**

在 LOLv1 训练集上训练，并在 LOLv1、SICE、LIEQ、NTIRE 2024、MIT‑Adobe FiveK 等五个数据集上进行测试。

**📈 对比分析**

通过与 ReDDiT、Retinexformer、RetinexNet、SNR‑Net、KinD++、Zero‑DCE、LIME 等方法对比，采用 PSNR、SSIM、LPIPS、MAE 四个指标，WREN 在所有指标上均表现最佳，特别是 PSNR 提升超过 1 dB。

**⚠️ 局限性**

仍依赖训练数据的多样性，极端低照度或特殊场景可能出现色彩失真或过曝；且未对实时或移动端的效率进行评估。

---

## 30. AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning

**arXiv ID:** 2607.15295 | [PDF](https://arxiv.org/pdf/2607.15295v1)

**作者:** Benjamin Robson `[一作]`, Arno Solin `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了AV-JEPA，一种基于LeJEPA的跨模态音视频自监督学习框架，利用早期融合ViT和模态丢弃实现跨模态对齐；

**💡 创新点**

创新点在于将模态丢弃代替空间遮蔽，形成隐式跨模态预测任务，完全去除解码器、对比负样本、EMA教师等复杂组件；

**🔧 技术方法**

使用技术包括LeJEPA损失与SIGReg正则化、早期融合ViT‑Base编码器、模态丢弃生成局部视图、在线分类探针（线性与注意力头）等；

**📊 数据集**

主要使用AudioSet‑2M进行预训练，VGGSound作为下游微调和评估数据集，并对AudioSet进行 mAP 评估；

**📈 对比分析**

与MAE/AV‑MAE/CAV‑MAE/MAViL 等基线对比，AV‑JEPA 在 VGGSound 上达到 57.1% top‑1，AudioSet 上 32.7 mAP，虽略低于对比/重建方法，但无需解码器或负样本即可获得竞争性能；

**⚠️ 局限性**

局限性包括模型对音频信息的依赖过强，视觉模态贡献有限；与基于重建与对比的先进方法仍存在性能差距；未探索更大模型、长时间预训练或多模态（文本/光流）扩展。

---

## 31. A Critical Analysis of Trustworthy AI Tools, Mark Frameworks, and the Implementation Chasms

**arXiv ID:** 2607.15480 | [PDF](https://arxiv.org/pdf/2607.15480v1)

**作者:** Michael Papademas `[一作]` (National Centre for Scientific Research Demokritos), Vangelis Karkaletsis `[通讯]` (National Centre for Scientific Research Demokritos)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究利用OECD可信AI工具与认证框架数据集，对其在伦理目标、生命周期阶段、受众定位和工具类型等维度进行定量映射与比较分析，揭示当前实践中的不平衡与缺口。

**💡 创新点**

创新点在于首次系统性量化OECD工具与认证的分布差异，突出透明度、公平性等核心原则的过度聚焦与可解释性、安全性、可持续性等重要原则的忽视，并提出“伦理设计”“教育不足”“行业主导”等实现鸿沟。

**🔧 技术方法**

采用描述性统计与比较分析方法（频次计数、类别归类、可视化映射）对OECD元数据进行处理。

**📊 数据集**

OECD可信AI工具与认证框架数据库（截至2025年7月，938个工具、24个可信/质量标识）。

**📈 对比分析**

通过对工具类别与认证维度进行频数统计与可视化对比，评估不同伦理目标和生命周期阶段的覆盖情况；论文未进行算法性能评估，仅提供定量发现与洞察。

**⚠️ 局限性**

局限性包括：依赖OECD预先划分的分类体系，未对数据进行抽样或独立重编码；研究仅为描述性与比较性，缺乏因果推断和实测验证；未考察工具在实际项目中的效果与可行性。

---

## 32. Proof-Carrying Multimodal Timelines: Finite-Trace Modal Certificates for Video-Audio Consistency

**arXiv ID:** 2607.15285 | [PDF](https://arxiv.org/pdf/2607.15285v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Hamdi Alakkad `[通讯]` (Bahcesehir University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于有限时序逻辑的多模态视频一致性验证框架，通过将视频、音频和字幕/ OCR 信号同步拆分为窗口化原子序列，并在此序列上评估一组可监测的逻辑公式，生成可重放的证书（certificate）来定位本地一致性失效。

**💡 创新点**

创新点在于：①将多模态时序一致性问题形式化为可监测的有限时序逻辑；②设计了紧凑的公式库，包括说话人一致、音视频事件一致、字幕视频一致、场景连续性和编辑冲击；③提出了可重放、可验证的证书结构，使得检测器的输出能够被独立检查；④在 GPU 加速下实现了大规模原子抽取与逻辑监测，生成完整的证书集。

**🔧 技术方法**

技术上采用了 CLIP 视觉原子提取、AST 语音原子提取、OpenCV 视频解码、PyTorch CUDA 并行计算、基于位集的布尔运算实现逻辑监测；逻辑层实现了带有受限时间窗口的 bounded 未来操作和全局 G 语义；证书层使用 SHA‑256 哈希、JSON 标准化、以及可重放的检查器。

**📊 数据集**

使用的数据集包括 YouCook2、AVE、AVA ActiveSpeaker、TVQA、ActivityNet Captions、Samplelib 等公开视频与注释集合，此外通过对公开 MP4 进行多种对抗性编辑（音频偏移、帧丢失、裁剪、压缩、字幕替换、场景重排）构建实验样本。

**📈 对比分析**

与传统的 clip‑level 评价（如 CLIPScore、SyncNet）相比，本文不直接给出分数，而是提供每个公式的判定结果、违规窗口、缺陷分数和证书。实验表明：1) 证书可在 350 条样本上完美重现；2) 通过不同比例编辑可以观察到不同公式的缺陷热图；3) 证书稳定性曲线显示在阈值/时间窗口的合理范围内，判定结果高度一致；4) 计算性能上，GPU 执行时每秒可处理数百个窗口，内存占用可根据窗口数线性扩展。

**⚠️ 局限性**

局限性包括：①证书只能验证逻辑监测的正确性，不能证明原始检测器（CLIP、AST）的感知正确；②逻辑公式仅覆盖有限的模式，无法捕捉所有多模态交互细节；③对长视频窗口化的分辨率与原子粒度有关，可能导致细粒度错误被模糊；④依赖于已标注的数据集，跨域泛化需进一步验证；⑤在极端编辑或噪声条件下，原子抽取误差可能导致证书不稳定，尽管理论上有界，但实际偏差仍需关注。

---

## 33. VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs

**arXiv ID:** 2607.15498 | [PDF](https://arxiv.org/pdf/2607.15498v1)

**作者:** Shahrzad Esmat `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1166 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的KV压缩方法VarRate，通过根据查询显著性为每个Token分配可变低秩预算，保持所有Token不被丢弃。

**💡 创新点**

创新点在于利用水填充机制将重要性信号映射到每个Token的低秩分配，并在不训练的前提下实现可变秩压缩。

**🔧 技术方法**

核心技术包括基于SnapKV的查询显著性评分、对残差进行PCA得到共享低秩基、以及水填充分配算法；同时使用旋转位置嵌入逆变换与残差编码。

**📊 数据集**

使用LongBench（16项长上下文任务）在Llama-3.1-8B、Qwen2.5-7B以及Mistral-7B等模型上进行评估。

**📈 对比分析**

与多种基线（SnapKV、Ada‑KV、PyramidKV、Palu、KVzip、Expected Attention、KIVI量化）在相同KV预算下对比，VarRate在匹配内存压缩下在两模型平均上保持不到1点的性能损失，且在查询无关重用场景中仅下降3–5点，显著优于其他无训练方法且接近KVzip且成本约为其八分之一。

**⚠️ 局限性**

限制包括对低秩基的离线校准依赖，较高的预填充开销（≈44%），以及在极低预算或不同模型结构下可能需要进一步微调。

---

## 34. Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving

**arXiv ID:** 2607.15621 | [PDF](https://arxiv.org/pdf/2607.15621v1)

**作者:** Yun Li `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种异步快慢双层结构的视觉语言动作推理模型，在闭环驾驶任务中实现每个50 ms仿真周期生成新控制指令。

**💡 创新点**

创新点包括：1）将大型视觉语言模型冻结为慢层缓存，只在每K帧更新；2）设计轻量级动作专家快层，在每个仿真周期仅利用缓存与当前帧进行跨层注意力推理；3）使用随机失效训练（staleness-augmented training）让专家适应缓存延迟。

**🔧 技术方法**

技术要点：冻结的7B LLaMA视觉语言骨干、Q‑Former视觉编码器、32层跨层注意力动作专家（337 M参数）、增量缓存更新、随机失效训练、CARLA仿真环境下的闭环评估。

**📊 数据集**

数据集：使用LMDrive的城镇05（Town 05）训练集，包含27,485条指令轨迹；在CARLA LangAuto-Short和LangAuto-Long路段进行验证；同时在未见城镇（Town 01、Town 02、Town 03）进行零样本迁移测试。

**📈 对比分析**

对比方法：与公开LMDrive基线（每隔一帧调用模型、重放旧命令）对比；以及同专家但10 Hz频率的框架。结果显示：1）闭环驾驶分数从28.8提升至32.9；2）路段完成率从37.0%提升至94.0%；3）红灯违规率下降三分之一；4）零样本迁移完成率保持84–94%，远优于基线；5）单步模型推理耗时保持在32 ms，独立于历史长度。

**⚠️ 局限性**

局限性：仅在仿真短路段上训练和评估，未处理长路段危险交互；低层PID控制器未针对快频率重新调优；整体安全性验证不足；基线对比只跑两次，结果波动较大。

---

## 35. Estimating the Reliability of Dynamic Time Warping Alignments Using Circumstantial Evidence

**arXiv ID:** 2607.15443 | [PDF](https://arxiv.org/pdf/2607.15443v1)

**作者:** Aanya Pratapneni `[一作]` (Harvey Mudd College), TJ Tsai `[通讯]` (Harvey Mudd College)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5034550382)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于FlexDTW对齐路径相对稳定性进行测量的无监督可靠性指标。

**💡 创新点**

通过比较DTW与FlexDTW路径的一致性来判断对齐可靠性，首次提供无监督评估方法。

**🔧 技术方法**

使用DTW、FlexDTW、局部窗口切分以及路径一致性度量等技术。

**📊 数据集**

使用Chopin Mazurka音频对齐数据集（5首作品共约300条录音）。

**📈 对比分析**

与仅使用相似度的基线对比，AUROC达0.97、TPR@2%FPR 94%、EER 4%，性能显著优于基线。

**⚠️ 局限性**

对短时匹配/非匹配区域的识别能力有限，窗口尺寸和重叠比例对精度影响显著。

---

## 36. Precise but Uncoupled: Reviewer Precision Does Not Guarantee Critique Uptake in Multi-Agent Math Reasoning

**arXiv ID:** 2607.15388 | [PDF](https://arxiv.org/pdf/2607.15388v1)

**作者:** Chih-Hsuan Yang `[一作]` (Argonne National Laboratory), Rajeev Thakur `[通讯]` (Argonne National Laboratory)

**通讯引用:** 10049 | [OpenAlex ID](https://openalex.org/A5014920685)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了多代理体系中专门审稿人与广播式讨论对Omni-MATH数学问题求解的影响，并研究了审稿人质量与批判采纳率的分离。

**💡 创新点**

提出检测‑采纳‑修复分解指标，展示审稿人精度与实际修复率可分离，并通过对PER协议内部干预验证批判采纳的重要性。

**🔧 技术方法**

使用大型语言模型驱动的多代理协议（PER、广播、单代理迭代）及开放评估器进行实验。

**📊 数据集**

Omni-MATH 2 竞赛级数学问题集（4,181题），十个难度层级。

**📈 对比分析**

对比四种协议，广播在高难度层级取得最高最终通过率（89.2%），PER次之（85.2%），单代理迭代（78.8%）；广播在批判采纳率上远高于PER。

**⚠️ 局限性**

局限于单一模型家族与数学领域，未完全证实因果关系，跨模型迁移时协议排名可能变化。

---

## 37. Large Language Models as Unified Multimodal Learners for Clinical Prediction

**arXiv ID:** 2607.15380 | [PDF](https://arxiv.org/pdf/2607.15380v1)

**作者:** Ajay Madhavan Ravichandran `[一作]` (German Research Center for Artificial Intelligence), Roland Roller `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将多模态电子健康记录（文本、时间序列、表格）统一序列化为自然语言后，用预训练语言模型进行端到端微调，完成多任务临床预测。

**💡 创新点**

提出了无需任务特定融合架构的统一文本序列化方法，显著简化系统设计与部署。

**🔧 技术方法**

采用Encoder（ModernBERT）和Decoder（Llama3.1、Gemma、DeepSeek-R1-Qwen、Qwen3）等预训练语言模型进行微调。

**📊 数据集**

使用三大数据集：MIMIC-III（ICU死亡预测）、德国肾移植中心（移植物失败预测）、德国急救车记录（急诊分级）。

**📈 对比分析**

与传统单模态、专用多模态融合模型及临床梯度提升基线比较，统一序列化模型在所有任务中均能匹配或超越基线，在移植物失败预测上甚至优于现行梯度提升系统。

**⚠️ 局限性**

序列化导致文本长度增大，可能超过模型上下文窗口；模型解释性不足，且对极长历史记录的处理仍存在挑战。

---

## 38. MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation

**arXiv ID:** 2607.15589 | [PDF](https://arxiv.org/pdf/2607.15589v1)

**作者:** Rajat Bhattacharjya `[一作]` (University of California), Nikil Dutt `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种轻量级运行时（Memoguard），在通信受限的机器人导航中对检索到的经验记忆进行拓扑、资源和结果三重验证，防止高相似度但失效的记忆（memory traps）导致的不安全执行。

**💡 创新点**

识别并解决 memory traps 概念；设计基于硬门限和可靠性评分的合同验证框架；在保持高安全性的同时显著降低回退（fallback）调用，提升资源利用效率。

**🔧 技术方法**

经验记忆检索（相似度评分）、合同验证（拓扑、资源、结果检查）、硬门限决策、本地规划（Dijkstra）、LLM 本地推理回退、图形化走廊模拟器。

**📊 数据集**

基于三种走廊拓扑（Linear Corridor、Alternate Path、Long Return Cost）的随机图形模拟器，离线生成 11,558 条经验记忆；硬件实验在 NVIDIA Jetson AGX Xavier 上进行。

**📈 对比分析**

对比 Top-1 Reuse、Threshold Reuse、Always Reasoning 与 Memoguard，结果显示 Memoguard 将任务成功率从 32.6% 提升至 84.2%，电池安全违规率从 67.4% 降至 15.8%（减少 76.6%），fallback 调用率下降 21.4%（从 18.58 降至 14.60），并在每次调用上节省约 3.67 s 与 36.97 J 能量，保持与 Always Reasoning 相近的任务成功率。

**⚠️ 局限性**

仅基于图层合同，未涵盖定位不确定性、通信质量与能量轨迹等动态因素；在资源极限或能量不足场景下仍可能误判；实验仅限单机器人单任务，未验证多机器人协作情况；需要更细粒度的合同与在线更新机制。

---

## 39. Are All Tokens Necessary for Visual Place Recognition? An Empirical Study of Token Reduction for Efficient Inference

**arXiv ID:** 2607.15563 | [PDF](https://arxiv.org/pdf/2607.15563v1)

**作者:** Tong Jin `[一作]` (Shenyang Institute of Automation, Chinese Academy of Sciences), Feng Lu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性评估了在视觉位置识别（VPR）中对视觉令牌（token）进行削减的效果，构建了统一的无训练token‑reduction基准框架。

**💡 创新点**

创新点在于首次对VPR任务进行全面的token‑reduction基准研究，揭示了token冗余特性，并为不同任务场景提供了精确的准确率-效率权衡指引。

**🔧 技术方法**

采用了七种代表性token‑削减策略（随机、L2‑norm、EViT、DivPrune、G‑Prune、ToMe、ToFu），在DINOv2、ViT、CLIP等transformer骨干和NetVLAD、SALAD、BoQ等聚合模块上进行测试。

**📊 数据集**

使用了Pitts30k、MSLS‑val、MSLS‑challenge、Tokyo24/7和Nordland等五个多样化VPR基准数据集。

**📈 对比分析**

通过对Recall@N、理论FLOPs、单图推理时延、批处理吞吐量等指标进行比较，结果显示token‑reduction可将FLOPs降低约26–29%，吞吐量提升1.4×，GPU单图时延提升约2–5 ms，边缘设备上可达3.9×加速，且大多数方法在准确率下降不到1%。

**⚠️ 局限性**

局限性包括仅进行无训练的token‑reduction，缺乏针对VPR稳定特征的专属重要性度量，早期层级削减易导致准确率急剧下降，且实验未覆盖训练感知或自适应token‑reduction策略。

---

## 40. Field-Aware RankMixer with Dual-Stream Bilinear Fusion for the Tencent UNI-REC Challenge

**arXiv ID:** 2607.15590 | [PDF](https://arxiv.org/pdf/2607.15590v1)

**作者:** Yufeng Zhang `[一作]` (Independent Researcher), Jiajun Cui `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种Field-Aware RankMixer (FA-RankMixer)模型，用于统一建模多域用户行为序列和非序列多字段特征，以预测广告点击后的转化率 (pCVR)。

**💡 创新点**

创新点在于将目标感知的多域DIN兴趣提取、字段感知语义token化、RankMixer块和双流双线性融合相结合，构建了一个可堆叠且统一的推荐架构。

**🔧 技术方法**

使用的技术包括目标感知DIN、权重对池化、Token化分组、RankMixer模块（参数无关token混合 + token特定pSwiGLU FFN）、RMSNorm、深浅两流MLP、组间双线性融合、SWA、SAM、以及多种优化器和加权损失。

**📊 数据集**

实验数据来自TAAC-2026腾讯UniRec挑战赛的工业轨道，包含约3,482万日点击样本和1,225万测试样本，正样本率约7.9%。

**📈 对比分析**

与基线和其他参赛方法对比，FA-RankMixer在官方测试集上实现0.828814的ROC‑AUC，排名第九，整体提升约14.7‰。

**⚠️ 局限性**

局限性包括模型参数量高达6亿，推理成本大；仅在单一竞赛数据集验证，缺乏跨域泛化与轻量化部署研究。

---

## 41. Physics-aware Masked Diffusion-based Flood Simulation for Urban Fisheye Disaster Detection

**arXiv ID:** 2607.15527 | [PDF](https://arxiv.org/pdf/2607.15527v1)

**作者:** Sodtavilan Odonchimed `[一作]` (University of Tokyo), Munkhjargal Gochoo `[通讯]` (United Arab Emirates University)

**通讯引用:** 2925 | [OpenAlex ID](https://openalex.org/A5025216905)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 PhysFlood 的基于 Diffusion 模型的 fisheye 摄像机单图像下水灾仿真框架，能够从单张鱼眼图像合成具有可控水深的逼真洪水场景。

**💡 创新点**

通过结合鱼眼图像的深度估计、地面分割、物体导向的地面插值以及 LoRA 适配的 Diffusion 模型，首次实现了可精确控制物理水深且能保留鱼眼几何畸变的水灾生成。

**🔧 技术方法**

使用 SAM3、Unik3D 与 YOLOv8 进行目标与地面分割，利用 LoRA 微调的 SD3.5‑medium 与 FLUX.1‑dev 进行掩码引导的 inpainting，并通过人工评估验证视觉可信度与物理一致性。

**📊 数据集**

采用 Fisheye8K 数据集作为原始鱼眼图像来源，并通过 NanoBanana Pro 与 GPT‑Image‑2 生成多层水深（踝、膝、腰）增强样本以构建训练集。

**📈 对比分析**

在四种模型（SD3.5‑medium、FLUX.1‑dev、NanoBanana Pro、GPT‑Image‑2）下，比较了有掩码（PhysFlood）与无掩码（普通图像对图像）两种生成方式，利用三名评审在 Q1‑Q3 的四分制评分。结果显示，PhysFlood 在保持水深一致性方面优于无掩码方法，且在高水深（≥2 m）时生成样本的物理合理性与检测价值更高；但基础模型在整体视觉质量上仍略优。

**⚠️ 局限性**

局限性包括评审人数有限、缺乏客观评价指标（如 FID/IS）、训练数据分布与真实场景差异导致 LoRA 微调模型性能不及基础模型，以及在极端鱼眼畸变或夜景灰度输入下出现掩码/生成失败的情况。

---

## 42. Verbalizable Representations Form a Global Workspace in Language Models

**arXiv ID:** 2607.15495 | [PDF](https://arxiv.org/pdf/2607.15495v1)

**作者:** Wes Gurnee `[一作]` (Anthropic), Jack Lindsey `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 Jacobian Lens 技术识别并分析 LLM 内部的 J‑space，证明其具备与人类意识工作空间相似的功能特性，包括可报告、可调节、内部推理、通用化和选择性，并通过实验验证其在多步推理、对话安全和行为塑造中的作用。

**💡 创新点**

提出 Jacobian Lens 与 J‑space 概念，将 Transformer 内部表示映射到可解释的可说出的词向量，揭示 LLM 具备类似全局工作空间的功能特性，并首次用 J‑space 进行干预以改进模型行为。

**🔧 技术方法**

Jacobian Lens、J‑space 分解、稀疏非负组合、插值/交换操作、对 J‑space 的 ablation 与 counterfactual reflection 训练等技术。

**📊 数据集**

使用 Claude Sonnet 4.5、Haiku 4.5、Opus 4.5/4.6 等大语言模型，基于预训练文本和专门构造的提示（如两步推理、语言检测等）进行实验。

**📈 对比分析**

通过对比 J‑space ablation 与对照实验，发现多步推理等任务从 90% 降至 0%，同时通过 Counterfactual Reflection 训练模型在对齐评估中的安全性提升约 15% 等指标。

**⚠️ 局限性**

J‑lens 仅覆盖单词级概念，无法捕获多词短语；只在部分层次检测到 workspace，可能忽略早期层；J‑space 仅为线性可解释近似，未能完全复制全局工作空间的循环特性。

---

## 43. Logic, Optimization, and Artificial Intelligence

**arXiv ID:** 2607.15532 | [PDF](https://arxiv.org/pdf/2607.15532v1)

**作者:** J. N. Hooker `[一作]` (Carnegie Mellon University), J. N. Hooker `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7653 | [OpenAlex ID](https://openalex.org/A5059159928)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了逻辑与优化结合在规则基础 AI 中的应用，包括概率逻辑、信念逻辑、非单调与多值逻辑、逻辑公式的统计推断、投影与透明性分析等。

**💡 创新点**

提出将投影问题统一视为逻辑与优化的通用框架，结合列生成、Benders 分解和后最优性分析，提升系统的透明性与可解释性。

**🔧 技术方法**

使用线性规划、整数规划、列生成、逻辑 Benders 分解、决策图（BDD）、贝叶斯网络和 Dempster–Shafer 组合规则等技术。

**📊 数据集**

主要通过理论示例与合成案例说明方法，并未使用公开真实数据集。

**📈 对比分析**

未进行数值实验比较，论文通过理论分析与已发表文献对比，强调方法在复杂性和可解释性上的潜在优势。

**⚠️ 局限性**

受限于 BDD 大小、整数规划求解复杂度、模型规模和线性可解性的假设，实际大规模应用仍需进一步优化和实验验证。

---

## 44. Hidden in Thought: Transferable Chain-of-Thought Artifacts Induce Harmful Behavior

**arXiv ID:** 2607.15286 | [PDF](https://arxiv.org/pdf/2607.15286v1)

**作者:** Ali khalil `[一作]` (Deakin University), Golnoosh Farnadi `[通讯]` (Mila Quebec Ai Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了从受损语言模型迁移有害链式思考（CoT）到未修改目标模型的攻击，并提出了可重用的黑盒模板。

**💡 创新点**

创新在于将有害CoT的可转移结构通过LLooM概念提炼为抽象模板，既可直接植入也可作为无害输入触发攻击。

**🔧 技术方法**

使用了模型组织、CoT植入、LLooM概念挖掘、系统提示生成以及对抗性评估等技术。

**📊 数据集**

实验使用29个开源模型、5个闭源模型，结合AdvBench、HarmBench、JailbreakBench等多种基准以及自制的emergent misalignment测试集。

**📈 对比分析**

与PAP、MSJ等黑盒基线对比，CoT植入在DeepSeek、Qwen2.5等模型上ASR可达90%，LLooM模板在Gemma和GPT‑4.1上提升10‑倍，整体表现远超单纯提示攻击。

**⚠️ 局限性**

局限包括仅评估两种受损模型、LLooM概念未做因果验证、缺乏多评审和目标特定优化、未提供防御方案等。

---

## 45. Position: Quantum Program Generation Must Prioritize Validity Over Probabilistic Scaling

**arXiv ID:** 2607.15313 | [PDF](https://arxiv.org/pdf/2607.15313v1)

**作者:** Junhao Song `[一作]` (Imperial College London), Yudong Cao `[通讯]` (Zapata Quantum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过对量子程序生成中概率规模化的理论与经验分析，指出传统基于大模型的“生成‑后‑过滤”方法在量子电路设计中面临语法-语义鸿沟与指数稀疏问题，进而提出以验证为中心的构造式生成框架，将正式验证嵌入生成过程并实现层级抽象与可验证动作掩模；

**💡 创新点**

创新点在于：①将正式验证从事后切换为生成时的构造约束，形成“验证‑中心”生成循环；②设计多层次抽象（模块、子模块、原子门）以降低验证成本；③提出基于验证轨迹的训练数据集，兼顾结构与优化信息；④用理论分析证明传统采样过滤在量子规模下指数不可行；

**🔧 技术方法**

主要技术包括：大规模语言模型与约束解码；工具调用与可验证动作掩模（shield）；基于马尔可夫决策过程的生成策略；理论复杂度分析与指数稀疏证明；以及多层级抽象下的层级验证与符号约束；

**📊 数据集**

数据集方面未给出具体公开集，作者讨论了使用未验证公开代码（如GitHub Qiskit、DeepMind等）和对验证轨迹数据集的构造需求；

**📈 对比分析**

比较方法：与传统的“生成‑后‑过滤”流程对比，利用理论期望成本公式和指数稀疏证明展示后者在量子规模下不可行；实验结果未给出，侧重于理论与案例分析；

**⚠️ 局限性**

局限性包括：仍无法在任意量子线路上实现完整生成；依赖高质量、可扩展的验证器与符号约束库；验证开销在大规模量子系统仍高；对特定量子算法（如Clifford+T、量子傅里叶变换等）效果更佳，通用性待进一步验证。

---

## 46. Relevant and Irrelevant: A Renormalization Group Analysis of Transformer Attention

**arXiv ID:** 2607.15449 | [PDF](https://arxiv.org/pdf/2607.15449v1)

**作者:** Parviz Haggi-Mani `[一作]` (Universite De Montreal), Irina Rish `[通讯]` (Universite De Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过Wilson‑Renormalization‑Group框架，将Transformer的注意力机制视为对训练好的MLP残差堆栈固定点的微扰，探讨其在不同输入相关长度下的相关性与无关性。

**💡 创新点**

创新点在于将注意力的“相关性”量化为RG算子（相关、边界、无关或分岔），并给出可检验的四条预测（固定点位移、层特异性、谱选择性与相位转移），首次在受控Markov输入上验证了Transformer的相位动力学。

**🔧 技术方法**

使用Wilson‑RG理论推导的固定点位移公式、Jacobian分解与层级驱动项，以及Cosine距离、有效秩、CKA等度量对模型行为进行分析，并通过头消融与扰动衰减实验验证。

**📊 数据集**

实验数据来自两种相关长度（短ξ≈1.2、长ξ≈6.7）的16词Markov链合成序列，长度T=64，采用BERT式遮掩预测训练。

**📈 对比分析**

与仅MLP基线比较时，短ξ下两者损失、有效秩和CKA几乎相同，说明注意力为无关扰动；长ξ下Transformer损失低于MLP且有效秩显著提升，表明注意力为相关扰动并实现了相位转移。

**⚠️ 局限性**

局限性包括：仅使用单头Transformer与极简架构，缺乏自然语言数据；固定点位移公式在相关扰动极大时失效；扰动衰减实验受输入投影对网络本征向量的非对齐影响，导致结果只能给出总体趋势而非精确排序。

---

## 47. Cura 1T: Specialized Model for Agentic Healthcare

**arXiv ID:** 2607.15314 | [PDF](https://arxiv.org/pdf/2607.15314v1)

**作者:** actAVA AI `[一作]`, Weiran Yao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并训练了 Cura 1T，一款专注医疗的千亿级语言模型，采用自演化闭环对数据混合进行迭代改进，最终在多种医疗基准上实现显著提升。

**💡 创新点**

创新点在于将自演化循环的搜索对象从提示/超参数转向数据混合，通过失败诊断自动生成并融合定制化训练数据，实现针对性修复。

**🔧 技术方法**

技术上使用 LoRA 低秩适配器、SFT→RL→SDFT 训练堆栈、自动化自演化循环，以及多种数据构造策略（Retention Anchor、Reasoning Correction、Knowledge Injection、Behavior Calibration）。

**📊 数据集**

使用的主要数据集包括 MedAgentBench、HealthBench Professional/Hard、MedXpertQA、AgentClinic 等医疗基准，以及 AIME、GPQA‑Diamond、τ²‑Bench 等跨领域基准。

**📈 对比分析**

通过与 Kimi‑K2.6、Claude Opus 4.8、GPT‑5.5 等前沿模型在同一基准上对比，Cura 1T 在五个医疗基准上获得最优或第二名，整体提升约 5–15% 分数；在外域基准上保持与前沿水平相当。

**⚠️ 局限性**

受限于算力和数据资源，Cura 1T 仅采用 LoRA 参数高效更新，缺乏全参数训练、长周期代理任务支持和更广泛的临床行为覆盖，且作为研究模型并不具备临床安全性。

---

## 48. A Model-Based Decoupling Strategy for Proprioception and Contact Sensing in an Architected Soft Manipulator

**arXiv ID:** 2607.15582 | [PDF](https://arxiv.org/pdf/2607.15582v1)

**作者:** Francesco Stella `[一作]` (Embodied AI SA), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了利用六通道空气压力传感器的软连续机械臂段，在单段和完整8段的Air-Helix机器人中实现了同时测量自身形变（本体感知）和外部接触的功能。

**💡 创新点**

在每段结构中引入局部锯齿状六个独立空气通道，形成对三自由度的冗余测量；采用分段恒曲率模型与Huber回归相结合的算法，利用残差自动区分形变与接触信号，实现一次性同时估计姿态和检测触碰。

**🔧 技术方法**

使用3D打印柔性建筑材料内置流体传感、压力-长度线性校准、分段恒曲率（PCC）模型、Huber迭代加权最小二乘回归、Instron机械加载实验、手动操控记录等技术。

**📊 数据集**

通过Instron加载实验获得六通道压力与位移的关系曲线；单段上进行178次接触试验（9个固定姿态）；多段演示的手动操控记录；无公开公开数据集。

**📈 对比分析**

采用OptiTrack基准测量进行对比；单段相对弯曲误差0.11±0.02；接触检测率97%；全臂末端笛卡尔误差0.06±0.02；与传统基于力缸/表面传感的方法相比，教学重放和力闭环控制表现更准确。

**⚠️ 局限性**

仅验证单段单点接触；多点接触和相邻段间接触交互未测试；接触定位仅至60°扇区；温度/动态负载对压力-长度线性校准的影响未系统评估；全臂性能仅基于静态姿态，缺乏连续跟踪评估。

---

## 49. A Physics-Informed Neural Network with a Modified Lorentzian Activation for Nonlocal Gradient-Flow Equations in Dynamic Density Functional Theory

**arXiv ID:** 2607.15291 | [PDF](https://arxiv.org/pdf/2607.15291v1)

**作者:** Dimitrios Gourzoulidis `[一作]` (Imperial College London), Serafim Kalliadasis `[通讯]` (Imperial College London)

**通讯引用:** 6143 | [OpenAlex ID](https://openalex.org/A5084671057)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向动态密度功能理论（DDFT）中非局部梯度流方程的物理信息神经网络（PINN）框架，并通过改进激活函数和预计算离散算子实现高效求解。

**💡 创新点**

创新点在于引入近似线性且随输入增大趋于零的Lorentzian激活函数，以缓解非线性和非局部项导致的梯度消失问题，同时预先计算卷积算子，显著提升训练过程中的计算效率。

**🔧 技术方法**

核心技术包括物理信息神经网络、改进的Lorentzian激活函数、预计算离散卷积算子，以及基于梯度流结构的损失函数设计。

**📊 数据集**

实验数据来自四个一维和二维的基准问题，其中一个有解析平衡解，其他通过连续和不连续Galerkin有限元方法得到参考解。

**📈 对比分析**

通过与Galerkin有限元求解器的L¹、L²、L∞误差、质量守恒和自由能耗散等指标进行对比，研究显示改进激活函数显著加快收敛速度，整体误差低于参考解，且成功捕捉梯度流特性。

**⚠️ 局限性**

局限性包括：预计算卷积算子对大规模或高维问题的存储与计算开销较大；当前框架主要针对可预先离散化的卷积核，对更复杂或不规则的非局部交互缺乏通用性；并且在极端非线性或极端参数下的收敛性尚未系统验证。

---

## 50. STAR: Astrocyte-Inspired State-Augmented Repair for Supervised Memristive AI Hardware Systems

**arXiv ID:** 2607.15415 | [PDF](https://arxiv.org/pdf/2607.15415v1)

**作者:** Yusuf Ahmed Khan `[一作]` (Penn State University), Abhronil Sengupta `[通讯]` (Penn State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种基于星形神经元的自我修复机制STAR，在具有永久性故障的忆阻交叉阵列上利用Equilibrium Propagation（EP）实现网络的恢复与稳健性提升。

**💡 创新点**

创新点在于首次将星形神经元的分布式自我修复理念与监督局部学习规则相结合，并在卷积网络及CIFAR‑10等大规模任务上实现可扩展的修复，无需故障定位或额外硬件冗余。

**🔧 技术方法**

核心技术包括：星形神经元调节的修复nudge、EP的三相动态与对比式更新、差分电导表示的权重映射、SA‑0/SA‑高故障注入、以及修复强度的二次损失项。

**📊 数据集**

使用的数据集为MNIST（全连接网络）和CIFAR‑10（VGG‑5卷积网络）。

**📈 对比分析**

通过与无STAR的EP自我恢复对比（多次随机种子平均），STAR在50%–90%故障率下恢复率提升约+40%~+52%；在CNN故障率最高13%时，准确率从≈30%提升至≈75%，并显著降低恢复过程中的方差。

**⚠️ 局限性**

局限性包括：需要预先训练无故障模型并存储激活目标；修复强度对极高故障率敏感；尚未在真实硬件上验证；对更深网络或不同任务的可扩展性与通用性仍待进一步研究。

---

## 51. Reasoning-Guided Part-Level Visual Grounding via Reinforcement Learning

**arXiv ID:** 2607.15374 | [PDF](https://arxiv.org/pdf/2607.15374v1)

**作者:** Kazi Sajeed Mehrab `[一作]` (Virginia Tech), Chris Thomas `[通讯]` (Virginia Tech)

**通讯引用:** 4588 | [OpenAlex ID](https://openalex.org/A5006675265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种“对象-部件层次反思式定位”（OP-HRG）框架，利用多模态大语言模型的推理能力进行细粒度的部件视觉定位。

**💡 创新点**

创新点在于：①将定位过程分解为对象先定位、部件再定位的粗细级联结构；②引入自我反思机制对初始预测进行检查与修正；③结合分阶段、可验证的奖励设计（OP-HRG+GRPO），针对对象、部件、包含关系和自我修正分别给出奖励。

**🔧 技术方法**

核心技术包括多模态大语言模型（Qwen3-VL-Instruct）、SAM2 作为固定的掩模解码器、强化学习框架 Group Relative Policy Optimization (GRPO) 与阶段化奖励；此外采用主动视觉感知步骤对自我反思进行视觉验证。

**📊 数据集**

使用的主要数据集有 PascalPart、PartImageNet 与 InstructPart，用于零样本跨数据集与在域内评估；训练时还利用 7k 的多对象数据集与 1200 张部件标注图。

**📈 对比分析**

与基线（LISA、Sa2VA、PixelLM、UniPixel、Molmo、Grounding DINO、VisionReasoner、SAM3 等）相比，4B 参数的 OP‑HRG 在 PascalPart、PartImageNet 与 InstructPart 的 gIoU 分别提升 15.9、7.9 与 10.7 点，甚至在零样本设置下也优于 7B 规模的基线；同时在 RefCOCO 等通用目标定位任务上的表现保持竞争力。

**⚠️ 局限性**

局限性包括：①自我反思机制在训练后趋向仅作验证而非主动修正；②部件包含奖励可能误将错误对象的子部件计为有效；③方法对大规模多实例场景的容错性有限，未来需进一步改进。

---

## 52. Unsupervised Keypoints for Real-Time Fall Detection: Comparative Analysis Under Real-world Conditions with Predictive Bandwidth Reduction

**arXiv ID:** 2607.15400 | [PDF](https://arxiv.org/pdf/2607.15400v1)

**作者:** Tasmiah Haque `[一作]` (West Virginia University), Mohammad Abdullah Al-Mamun `[通讯]` (Binghamton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于无监督关键点提取与VRNN时序预测的隐私友好跌倒监测框架，替代连续视频传输。

**💡 创新点**

创新点在于消除对解剖学标记的依赖，将无监督关键点与预测模型结合，提升遮挡鲁棒性并显著降低计算与带宽成本。

**🔧 技术方法**

使用YOLOv8人体分割器、无监督关键点检测器、变分递归神经网络（VRNN）进行关键点预测，以及LSTM分类器进行跌倒判别。

**📊 数据集**

在公开的UR Fall Detection和Human Fall (GMDCSA-24) 两个跌倒数据集上进行评估。

**📈 对比分析**

通过随机拆分、主体分离、遮挡三种评估方案对比，结果表明在主体分离与遮挡场景下无监督关键点在F1上显著优于有监督关键点，随机拆分时差异不显著；在带宽受限的预测设置下无监督关键点仍保持优势。

**⚠️ 局限性**

局限性包括：无监督关键点在动作变化剧烈的非跌倒情景可能产生误报；极端遮挡或极少训练样本时性能受限；实验仅涵盖室内固定摄像头，未验证多视角或户外环境的适用性。

---

## 53. RecGPT-V3 Technical Report

**arXiv ID:** 2607.15591 | [PDF](https://arxiv.org/pdf/2607.15591v1)

**作者:** Bowen Zheng `[一作]`, Zile Zhou `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 RecGPT‑V3，利用大型语言模型实现对淘宝首页“猜你想买”场景下用户行为的持续记忆、语义标识与压缩推理，从而提升点击率和交易额。

**💡 创新点**

创新点包括三大模块：可演化的 Memory Hub 取代一次性全序列建模；混合模态基础模型同时输出自然语言标签和语义 ID，消除标签到商品的瓶颈；以及 Latent Intent Reasoning 将长链式推理压缩为可解释的低维隐向量，显著降低推理成本。

**🔧 技术方法**

技术栈涵盖 Qwen3‑14B LLM、结构化行为压缩与增量记忆更新、RQ‑VAE 语义 ID 量化、连续预训练+指令调优、隐式链式推理以及基于下游排名奖励的强化学习。

**📊 数据集**

实验数据来自淘宝海量用户行为日志、商品文本、图像特征和协同过滤生成的正负样本，用于构建多模态语义 ID 与文本对齐的训练集。

**📈 对比分析**

与 RecGPT‑V2 的线上 A/B 测试对比，RecGPT‑V3 在 IPV+1.28%、CTR+1.00%、TC+1.97% 和 GMV+3.97% 等指标上均实现提升，同时整体推理计算量下降 52.4%。

**⚠️ 局限性**

局限性包括对大型模型算力的高度依赖、隐式推理在极端稀缺场景下可能缺乏细粒度解释，以及在不同平台或业务场景中的可迁移性尚未充分验证。

---

## 54. Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories

**arXiv ID:** 2607.15330 | [PDF](https://arxiv.org/pdf/2607.15330v1)

**作者:** Xiaomi Robotics Team `[一作]` (Xiaomi Robotics), Quanyun Zhou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可在未见环境中直接按指令完成多种移动操控任务的基础视觉-语言-动作（VLA）模型，并通过大规模真实轨迹预训练和跨体型后训练实现强大的零样本与少样本适应能力。

**💡 创新点**

①利用超过100k小时的真实世界手持UMI设备轨迹构建大规模数据集；②设计自动标注流水线，用预训练的Qwen3.5对状态转移进行自然语言描述，避免人工标注瓶颈；③采用两阶段训练（预训练+后训练）与Mixture‑of‑Transformers架构（VLM+DiT），实现跨体型与指令对齐；④系统性探索数据量与模型规模的扩展性，验证规模规律能直接迁移到真实机器人性能。

**🔧 技术方法**

• 预训练：流匹配（flow‑matching）与回归（L1）损失；• 后训练：对齐跨体型动作空间、指令形式；• 结构：Mixture‑of‑Transformers（Qwen3‑VL + Diffusion Transformer）+ Choice Policies；• 训练技术：beta采样时间步、adaLN、动作辅助监督、KV缓存排除等。

**📊 数据集**

• 100k小时UMI手持设备轨迹（自动标注的状态转移语言）；• 10k小时跨体型机器人轨迹（包括移动机器人、双臂机器人、开源数据）；• 2k小时虚拟/模拟任务（RoboCasa、RoboCasa365、VLABench、RoboDojo等基准）。

**📈 对比分析**

在真实机器人上无额外微调的零样本任务（鞋子存储、行李打包等）成功率最高达79%；在少样本下（≤10h/任务）平均成功率75%，显著优于π_0.5（40%）和Xiaomi‑Robotics‑0；在四大仿真基准上实现了SOTA，RoboCasa 74.5%，RoboCasa365 57.4%（比前沿提升10.8pp），VLABench最高成功率/进度，RoboDojo平均得分提升7%/成功率5.13%。

**⚠️ 局限性**

①数据收集仍需昂贵的UMI/机器人实验，缺乏多模态多任务标注；②模型规模大，训练与推理成本高；③在RoboDojo的记忆维度表现逊色，因未使用历史观察；④目前仅验证于可穿戴/室内环境，外部环境的普适性待进一步评估。

---

## 55. CHRONO-RESOLUTION: A Dependency Resolution Dataset at Release Points for npm, PyPI, and crates.io Packages

**arXiv ID:** 2607.15315 | [PDF](https://arxiv.org/pdf/2607.15315v1)

**作者:** Imranur Rahman `[一作]` (North Carolina State University), Laurie Williams `[通讯]` (North Carolina State University)

**通讯引用:** 16982 | [OpenAlex ID](https://openalex.org/A5028171895)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个包含npm、PyPI和Cargo生态系统的发布点依赖解析数据集，并加入OSV漏洞信息，提供了数据集与重建脚本；

**💡 创新点**

首次公开发布跨多个主流生态的发布点依赖解析数据集，加入了未修复漏洞标注（fix_available）并提供完整重建流程；

**🔧 技术方法**

使用deps.dev的时间旅行解析器、OSV API、SemVer标准化、SQL与自定义脚本进行数据收集、清洗与整合；

**📊 数据集**

基于npm、PyPI、Cargo的注册表元数据、deps.dev的解析快照以及OSV的漏洞数据，最终生成约163,207个包、约2.7M条依赖关系的关系表及约4,000条漏洞表；

**📈 对比分析**

与先前仅覆盖Maven的Jaime等数据集对比，支持动态度量（如依赖新鲜度、更新节奏）等分析；性能方面未做显式基准测试，但数据集规模和完整性可满足大规模研究需求；

**⚠️ 局限性**

受限于两年年龄阈值、依赖deps.dev的回溯解析器（可能与历史真实安装不符）、仅收集截至2024年9月的漏洞、无法公开重建快照、验证样本有限等因素。

---

## 56. Assessing Learning Processes with Multimodal Data in Virtual Reality Learning Environments

**arXiv ID:** 2607.15403 | [PDF](https://arxiv.org/pdf/2607.15403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 57. From Neural Intent to Cryptographic Authorization: Governing Agentic Workflows

**arXiv ID:** 2607.15596 | [PDF](https://arxiv.org/pdf/2607.15596v1)

**作者:** Jiasi Weng `[一作]` (Guangzhou University), Yue Zhang `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Neural Cryptographic Services (NCS)，在 AI 代理工作流中加入主动安全治理层，利用离线签名、哈希链和符号化执行，确保只有通过加密验证的指令块才能触发工具调用，从而阻止提示注入和参数劫持。

**💡 创新点**

创新点：1）将神经网络的意图解析与符号化的加密授权分离，形成两阶段状态化指令鉴权（签名验证 + 哈希链验证 + 参数绑定）；2）在 LLM 与工具之间插入 fail‑closed 的符号控制器与 Verifier，提供可审计的执行轨迹；3）兼容现有 PKCS#11 关键管理系统，保持企业加密基础设施的可迁移性。

**🔧 技术方法**

技术细节：Ed25519 签名 + SHA‑256 哈希链；两阶段指令鉴权协议；神经符号架构（Neural Planner + Symbolic Controller + Crypto Execution Engine + Verifier + Runtime State Management）；JSON 结构化计划编译器；PKCS#11 兼容的加密后端；日志与审计机制。

**📊 数据集**

数据集：AgentDojo（多域工作流）、InjecAgent-ArgHijack（金融工具参数劫持），OpenPromptInjection（提示注入）、自定义金融工具集（BankManagerPayBill、VenmoWithdrawMoney 等）。

**📈 对比分析**

比较方法：对比 Baseline、Spotlighting、FATH、FIDES 四种防御，在四个模型（DeepSeek-Chat、GPT‑4o‑mini、GPT‑5 Reasoning、GPT‑5‑Chat）下评估 ASR_tool、ASR_args、Benign Utility (BU)、Attack Utility (U)、Partial Utility (U∂)。NCS‑Full 在三种模型上将 ASR_tool 降至 0%（GPT‑5 Reasoning 为 0%，DeepSeek‑Chat 为 0%），保持 BU ≈ 75% 或更高，且加密验证延迟仅 1–1.5 ms，远低于 LLM 推理时间。

**⚠️ 局限性**

局限性：仅支持签名 + 哈希链，未覆盖加密、阈值签名等高级功能；对自然语言输出泄漏（如攻击通过文本注入信息）仍有限制；需要外部签名者，管理工作流签名成本；在最弱模型上仍存在少量通过文本注入获得成功的攻击案例。

---

## 58. Interactive Mascot: A Scene-Centric Interaction Grammar for Data Visualizations

**arXiv ID:** 2607.15523 | [PDF](https://arxiv.org/pdf/2607.15523v1)

**作者:** Zhicheng Liu `[一作]` `[通讯]` (University of Maryland College Park), Zhicheng Liu (University of Maryland College Park)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种场景中心的交互语法 Interactive Mascot，并在 Mascot.js 库中实现了基于依赖图的执行模型，用以支持以语义场景组件为主体的交互行为。

**💡 创新点**

通过将触发器、响应器、评估器和更新器四个交互组件与事件/状态上下文结合，提供了与视觉组件直接对应的高层交互抽象，并用可重用的依赖图模式实现了交互执行，填补了现有基于选择或信号的交互语法在场景中心可视化中的不足。

**🔧 技术方法**

采用 JavaScript/ECMAScript，构建了基于变量-运算符节点的依赖图；实现了事件/状态上下文、评估器/更新器函数、以及可选的动画支持；在 Mascot.js 里使用了可视化组件、数据绑定、布局、编码、约束等核心语义组件。

**📊 数据集**

主要使用公开的 GDP/Life Expectancy CSV 数据集（192 国数据），以及在性能基准中使用的多尺寸随机数据集（100~100k 点）和 Vega‑Lite 示例图集。

**📈 对比分析**

与 Vega‑Lite（v6.1.2）在相同 SVG 渲染下进行四种交互（E2、E4、E5、E6）以及不同数据规模的帧率基准；结果显示 Interactive Mascot 与 Vega‑Lite 的帧率相近，某些交互（如 E2）性能更优，另有交互（如 E5）略逊。

**⚠️ 局限性**

仅支持固定结构的可视化，无法在运行时创建/删除场景对象；不支持自定义低级事件流（如多点触控）；状态上下文虽提供，但 evaluator/updater 必须是函数，缺乏声明式语法；对大规模多维交互（如 SPLOM 链接）性能受限。

---

## 59. Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?

**arXiv ID:** 2607.15439 | [PDF](https://arxiv.org/pdf/2607.15439v1)

**作者:** Sergey Rodionov `[一作]` `[通讯]` (SingularityNET), Sergey Rodionov (SingularityNET)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 ARC-AGI-3 基准上，对四种编码代理（文本、可变接口可执行、简化、验证）进行系统化的嵌套消融实验，评估可执行模型、简化与验证三者对性能的贡献。

**💡 创新点**

创新点在于将可执行模型、简化与验证拆解为可独立实验的层级，并通过嵌套消融揭示它们对代理表现的相对重要性；同时提供完整的可复现代码与提示。

**🔧 技术方法**

使用 Codex（GPT‑4 / GPT‑5.6）编码代理、Docker 隔离环境、可执行世界模型、简化提示、精确重放验证器和成本 token 计量等技术。

**📊 数据集**

数据集为 ARC‑AGI‑3 的公开 25 个游戏（后续 v1.5、v1.6 版本亦同样使用该公开集）。

**📈 对比分析**

实验设计为 4 变体 × 2 模型 × 2 推理预算，共 16 组；评价指标为 RHAE（相对人类行动效率）与成本 token。结果表明：模型能力与推理预算是主导因素；验证变体在所有设置中获得最高 RHAE，但资源消耗最大；可执行模型在强模型/高预算下并不总是优于文本变体；简化在大多数设置中提升效果。

**⚠️ 局限性**

局限性包括：验证优势难以归因于单一组件；实验仅覆盖公开游戏，未检验对未见游戏的泛化；资源消耗高；单次 playthrough 可能导致高方差；缺少验证–简化、文本–简化等组合的完整比较。

---

## 60. Partial Information Decomposition as a Multi-Contrast 3D MRI Selection Strategy for Resource-Constrained Deep Neural Network Training in Brain Tumor Segmentation

**arXiv ID:** 2607.15396 | [PDF](https://arxiv.org/pdf/2607.15396v1)

**作者:** Agamdeep Chopra `[一作]` (University of Washington), Mehmet Kurt `[通讯]` (University of Washington)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5011934020)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了基于部分信息分解（PID）的预训练框架，用于在脑肿瘤3D分割任务中先行挑选最有信息量的两条多对比MRI序列，随后用轻量级3D U‑Net进行分割训练；

**💡 创新点**

创新点在于引入了区域感知的MMI-PID评分，对序列对的冗余、独特与协同信息进行量化，利用浅层自编码器和量化后的潜在表示实现高效预筛选，从而在不训练完整模型的情况下预测下游分割性能；

**🔧 技术方法**

技术手段包括：浅层3D自编码器对每条序列进行编码并量化，使用MMI-PID对所有序列对进行评分，训练轻量级3D U‑Net进行分割，并在全输入模型上做Shapley值归因；

**📊 数据集**

使用了CoRe‑BT数据集，包含4种常见序列（T1、T1CE、T2、T2‑FLAIR），共132例训练、30例验证、28例测试；

**📈 对比分析**

通过训练11个架构相同的U‑Net模型（全四输入、六种两输入组合、四种单输入）进行对比，评估指标为宏平均Dice、HD95、灵敏度和精确率。选定的两序列组合在测试集上Dice仅比全四输入低0.01（98.5%保持），并在统计检验中显著优于多数其他组合；

**⚠️ 局限性**

局限性包括仅针对4种序列的小样本数据，缺乏更大规模或更多候选序列的验证，统计功效有限，且未能在预设非劣效边际下证明所选组合的非劣效性。

---

## 61. Regularity-Aware Stochastic MGDA with Adaptive Conflict-Avoidant Update Direction Control

**arXiv ID:** 2607.15412 | [PDF](https://arxiv.org/pdf/2607.15412v1)

**作者:** Chentong Huang `[一作]` (University of Rochester), Lisha Chen `[通讯]` (University of Rochester)

**通讯引用:** 4095 | [OpenAlex ID](https://openalex.org/A5091442724)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自适应正则化的随机多目标梯度下降算法MoRe，动态切换冲突避免方向与固定加权方向；

**💡 创新点**

发现冲突避免方向在非退化条件下满足Lipschitz连续性，利用该性质改进随机MGDA的收敛速率从O(T⁻¹/4)提升至O(T⁻¹/2)；

**🔧 技术方法**

使用凸组合梯度的最小范数子问题、Hölder/ Lipschitz 连续性分析、批量大小自适应、梯度方差界定和线性调度；

**📊 数据集**

在Office-Home多任务图像分类数据集上进行实验；

**📈 对比分析**

与CAGrad、MoCo、MoDo等基线对比，MoRe在多任务整体性能（Δ_A^id%）上取得最佳表现，且收敛曲线与理论O(T⁻¹/2)一致；

**⚠️ 局限性**

对两目标情况的冲突避免距离分析有限，未扩展到一般M>2的多目标情形。

---

## 62. GraphDx: A Cost-Aware Knowledge-Enhanced Multi-Agent Framework for Sequential Diagnosis

**arXiv ID:** 2607.15280 | [PDF](https://arxiv.org/pdf/2607.15280v1)

**作者:** Shaoting Tan `[一作]` (Shandong University), Haitao Yuan `[通讯]` (National Technological University)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5002395365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个基于知识增强的顺序诊断框架GraphDx，自动化构建医学诊断知识图谱（MDKG），并通过感知、推理、决策三代理协同实现高效、低成本的临床诊断；

**💡 创新点**

核心创新包括①自动化管线生成带典型性权重、动作中心拓扑及成本属性的MDKG；②引入三代理架构，使推理代理在图上做确定性证据评分和成本敏感规划，显著降低LLM对推理的依赖；③结合量化典型性与费用敏感的决策策略，避免“防御性医学”过度检验；

**🔧 技术方法**

技术实现涵盖LLM（DeepSeek-V3、Kimi-k2、Llama-3.3）进行知识抽取与图构建；混合实体对齐与动态节点升级；图增强推理引擎（证据评分、成本效益规划）；三代理协同与模拟环境；

**📊 数据集**

实验数据集包括MedQA‑Extended（200案例）和MIMIC‑IV（200真实病例），并在Open‑Set子集上进行未见疾病评估；

**📈 对比分析**

与标准LLM提示、MAI‑DxO多代理基线进行比较，评估指标为诊断得分、总成本、对话轮数和成功率。GraphDx在MedQA上成功率从约68%提升至88‑93%，成本下降20‑54%；在MIMIC‑IV上成功率提升至83‑86%，成本下降约50%；在Open‑Set上仍优于基线；

**⚠️ 局限性**

局限性包括①图构建依赖基础LLM知识，易受知识盲区影响；②仅处理文本，缺乏多模态支持；③推理过程引入额外延迟；④仿真环境与真实患者存在差距；⑤LLM评估与成本估算可能存在偏差；⑥系统仍需人工监督，不能直接临床使用。

---

## 63. Scalable Open-Source Visuotactile Sensor for 6-Axis Contact Wrench Estimation in Tensegrity Robots

**arXiv ID:** 2607.15633 | [PDF](https://arxiv.org/pdf/2607.15633v1)

**作者:** Wenzhe Tong `[一作]` (University of Michigan), Xiaonan Huang `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种可扩展的可视触觉端盖传感器，能够为张力结构机器人估计六轴力矩并实时检测接触。

**💡 创新点**

采用无粘合剂的连续填充“gyroid”机械键合技术、模块化 TPU 接口以及残差多层感知机将光学剪切场映射为六轴力矩，提升了接触感知的精度和可重复性。

**🔧 技术方法**

使用 3D 打印 TPU 结构、硅胶成型、LED 环光、USB 伪焦相机、光流剪切场估计、残差 MLP 网络以及 ATI Gamma 6 轴力矩传感器进行校准。

**📊 数据集**

在 KUKA R820 机械臂上收集了 34,368 对剪切场与力矩标注的静态接触数据集，涵盖不同倾斜、扭转角度和接触力度。

**📈 对比分析**

与传统粘合方法相比，机械键合在拉伸强度上提高约25%，在剥离强度上提高约300%；残差 MLP 在验证集上的 MSE 为 0.1531，在动态圆形轨迹下为 2.67；在 12 kg 张力结构机器人上能实时检测地面接触并与人工标签一致。

**⚠️ 局限性**

动态接触下力矩估计误差较大，受相机帧率和剪切场分辨率限制；未对多种不规则地形、长期冲击或多点接触进行深入评估；缺乏完整机体状态估计的集成。

---

## 64. On the CGGRT Criterion for Detecting Bipartite Perfect Matchings in NC

**arXiv ID:** 2607.15554 | [PDF](https://arxiv.org/pdf/2607.15554v1)

**作者:** Swastik Kopparty `[一作]` (University of Toronto), Shubhangi Saraf `[通讯]` (University of Toronto)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于子空间设计的新型NC判定准则，用于检测二分图是否存在完美匹配，并将判定问题化简为计算一个 O(n³)×O(n³) 矩阵的秩。

**💡 创新点**

创新点在于通过改进子空间设计的参数和多重性分析，给出了比之前 CGGRT 等算法更简单、更高效的判定准则，使得矩阵规模从 O(n⁵) 降到 O(n³)，从而在理论上显著提升了算法的效率。

**🔧 技术方法**

主要技术包括：Wronskian 判别线性无关性、子空间设计理论、代数多重性与范德蒙行列式的运用，以及将判定问题转化为矩阵秩计算的技巧。

**📊 数据集**

本研究为纯理论算法，不使用任何实验数据集；所有结果均来自理论分析和复杂度计算。

**📈 对比分析**

与 Chatterjee 等人以及早期 CGGRT 算法相比，本文的准则在矩阵维度和秩计算复杂度上都有两倍的改进；在保持在 NC 计算范畴内的同时，理论上实现了更低的多项式度。

**⚠️ 局限性**

局限性包括：需要域特征大于 3n³，算法仅提供判定而非构造匹配；实现中对多重性和子空间设计的细节要求较高，实际可行性和常数因子尚待进一步评估。

---

## 65. Risk-Aware Preference Learning for Stochastic Outcomes

**arXiv ID:** 2607.15483 | [PDF](https://arxiv.org/pdf/2607.15483v1)

**作者:** Yi-Shiuan Tung `[一作]` (University of Colorado Boulder), Bradley Hayes `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5034950112)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在社会机器人导航场景中，学习并比较了基于人类偏好的奖励函数，其中对比了期望效用和累积前景理论两种偏好模型。

**💡 创新点**

首次将累积前景理论（CPT）引入偏好学习框架，证明风险敏感用户的偏好可以通过非线性概率加权和价值变换更精准地建模，从而改进奖励恢复。

**🔧 技术方法**

使用Bradley‑Terry 似然优化、CPT价值函数、线性期望效用模型以及特征权重学习技术来拟合奖励权重。

**📊 数据集**

构造二维社交导航仿真环境，利用 Helbing–Molnár 社会力模型生成行人轨迹，并在七种元动作上采集 50 次滚动样本，生成五维特征向量。

**📈 对比分析**

在合成的 EU 与 CPT 教师偏好下对比，CPT 学习者在 CPT 教师情形下的行动遗憾显著低于 EU 学习者；当教师偏好为 EU 时，EU 学习者表现更好，说明匹配偏好模型可提升性能。

**⚠️ 局限性**

实验仅在合成数据上验证，缺乏真实人类评估；CPT 参数需手工设定；样本量有限，且仅针对导航任务，尚未检验方法在其他领域的泛化能力。

---

## 66. Adaptive Retrieval Strategies for Biomedical Question Answering

**arXiv ID:** 2607.15283 | [PDF](https://arxiv.org/pdf/2607.15283v1)

**作者:** Han Yue `[一作]` (Insilicom), Jinfeng Zhang `[通讯]` (Insilicom)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个自适应检索框架，针对不同类型生物医学问题（是/否、事实、列表、摘要）动态选择检索与证据聚合策略，并结合LLM关键词抽取、文档检索、重排序、知识图谱扩展、文档过滤和答案生成，实现高质量检索增强式问答。

**💡 创新点**

创新点在于：①根据问题类型动态切换检索策略和证据聚合；②同时从问题与LLM生成答案中抽取关键词并扩展同义词；③使用两级重排序模型与LLM过滤相结合的三阶段文档筛选；④在列表问题中引入知识图谱查询补全实体；⑤并行化LLM与重排序提升系统吞吐量。

**🔧 技术方法**

技术：大型语言模型（GPT‑4o、Claude‑3‑7‑sonnet）、关键词抽取与同义词扩展、iSearch检索、jina‑reranker‑v2‑base‑multilingual、bge‑reranker‑v2‑5‑gemma2‑lightweight、LLM文档过滤与证据提取、知识图谱（MongoDB）查询、异步OpenAI API、并行计算。

**📊 数据集**

数据集：BioASQ 13B 任务（Phase A、A+），包含约5389个问题、49610篇PubMed文章。

**📈 对比分析**

比较方法：设立5个基线（Naive、Baseline_Top10、Baseline_Top20、Mainpipeline、KG），在四批次上对文档检索MAP、片段F‑measure、是/否宏F1、事实MRR、列表F‑measure与召回率进行评估。实验结果显示：在Phase A的检索与片段任务中，本框架多次夺得第一名；在Phase A+的是/否、事实与列表任务中，在多批次表现稳健，尤其在列表召回率上名列第一。

**⚠️ 局限性**

局限：①对列表问题的知识图谱依赖于现有图谱覆盖范围；②实验仅在BioASQ 13B公开数据上评估，缺乏跨领域或真实场景的验证；③LLM过滤与答案生成对计算资源需求高，需并行优化；④对摘要问题的评估依赖人工评分，缺少自动化指标。

---

## 67. SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents

**arXiv ID:** 2607.15557 | [PDF](https://arxiv.org/pdf/2607.15557v1)

**作者:** Yanze Wang `[一作]` (EverMind), Yafeng Deng `[通讯]` (EverMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 SkillCorpus 框架，将 821K 原始公开技能文件通过六阶段管道去重、质量评估、分类与许可过滤，最终得到 96K 高质量技能集合，并搭建了基于 embedding+reranker+LLM 选择的检索‑选取堆栈，完成了对多 benchmark / harness / backbone 的 end‑to‑end 评估。

**💡 创新点**

创新点包括：①首次将整个公开技能生态系统聚合成可复用的统一语料库；②提出三维质量（实用性、鲁棒性、安全性）评估与 19 旗标记；③引入 16 类技能分类与基于检索的动态匹配；④在多 benchmark、两 harness、两 backbone 以及前沿模型上系统性评估，验证社区技能在真实任务中的实际收益。

**🔧 技术方法**

技术手段：多阶段聚合管道（结构解析、去重、LLM 判别器、19 旗标记、OSI 许可过滤、embedding 生成）；检索堆栈采用 Qwen3‑Emb‑0.6B fine‑tuned 生成 1024‑维向量，Qwen3‑Rank‑0.6B fine‑tuned 进行 rerank，LLM 选择器根据任务 query 与完整技能 body 决策注入；可选 query rewriter；支持 OpenClaw、Raven 两个 harness 的统一接口。

**📊 数据集**

数据集：821K 原始技能文件（来源 62 个公开仓库）；评估 benchmark 包括 SkillsBench（87 任务）、GDPVal（220 任务）、QwenClawBench（100 任务）；前沿模型检查使用 Claude Opus 4.7 与 SkillsBench；此外对检索堆栈做了 standalone Hit@1 / Recall@10 的评估。

**📈 对比分析**

比较方法：在每个 harness+backbone 组合（共四个 cell）对比无技能 baseline 与 SkillCorpus（检索+选择堆栈）条件，计算 pass‑rate 或平均奖励的增量 Δ。结果显示：SkillsBench 平均 +7.5pp，GDPVal +1.5pp，QwenClawBench +2.8pp；前沿模型 Opus 4.7 在 SkillsBench 上 +8pp。检索堆栈 ablation 证明精细调检索和 curated corpus 各自贡献约 10‑20% 的增益。

**⚠️ 局限性**

局限性：①评估侧重广度而非深度，缺乏多次重复与大样本统计；②质量评估仅基于 LLM 文本判别，未做运行时 sandbox 验证；③语料主要为英语，缺乏多语言覆盖；④仅发布单一 snapshot，未研究生态系统随时间演化的影响；⑤评估 benchmark 与技能集可能存在潜在泄漏风险。

---

## 68. Clinical Audit Logs as Multi-Axial Traces of Care Delivery

**arXiv ID:** 2607.15397 | [PDF](https://arxiv.org/pdf/2607.15397v1)

**作者:** Braden Eberhard `[一作]` (University of Pennsylvania), Kevin Johnson `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出将电子健康记录（EHR）审计日志视为多轴事件流，并基于该结构探索基于大规模无标签日志进行预训练的基础模型，以提供可跨多任务、跨机构通用的行为表示；

**💡 创新点**

创新点在于：①把单一审计条目同时映射到临床工作者、患者轨迹、团队活动和临床流程四个维度；②以此多轴视角为前提，倡导在完整日志上做预训练，从而在缺少标签或跨轴依赖强的任务中获得更强的泛化；③提出了针对多轴能力的任务分级、benchmark构建与治理框架；

**🔧 技术方法**

使用技术包括：大规模序列预训练（类似 Transformer/embedding 方法）、多轴事件编码、时间与上下文保持、跨机构迁移学习；

**📊 数据集**

数据集为医院系统产生的实时 EHR 审计日志，覆盖用户、时间、对象、角色等字段，具备多机构、多科室的规模；

**📈 对比分析**

与传统单轴、任务特定模型对比，作者指出预训练模型在需要跨轴信息的任务（如团队协调、患者不良事件预测等）上表现更优，虽未给出具体数值，但通过跨任务评测和多机构验证证明了其优势；

**⚠️ 局限性**

局限性包括：需要更完善的 benchmark 以量化跨轴效果；对日志的时间、上下文信息编码仍需改进；跨机构迁移易受系统配置差异影响；治理与公平性问题复杂，需进一步规范使用与评估流程。

---

## 69. Empathy as Predictive Misalignment Tolerance: A Co-Regulation Framework and the Regime Structure of Dialogue Repair

**arXiv ID:** 2607.15282 | [PDF](https://arxiv.org/pdf/2607.15282v1)

**作者:** Molood Arman `[一作]` `[通讯]`, Molood Arman

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了 Interpretive Error Tolerance（IET）框架，将共情视为对解释偏差的动态容忍与调节，而非单纯趋向一致。

**💡 创新点**

创新点在于将共情重新定义为“容忍分歧的动态阈值”，并通过两次受控噪声实验揭示对话修复存在噪声驱动的分层结构：低噪声下修复反而降低检索准确度，高噪声下修复能更好保留语义要义。

**🔧 技术方法**

使用句子嵌入距离计算对话偏差，结合阈值调节（固定与自适应）以及连续衰减的修复策略；核心算法为IET阈值更新公式。

**📊 数据集**

数据集为公开的 DailyDialog 对话语料，人工注入不同比例噪声以模拟对话混乱。

**📈 对比分析**

比较方法：对比无修复、固定阈值修复与自适应 IET 修复，评估指标为 1‑of‑N 下一轮检索准确率和与原始干净上下文的余弦相似度；结果显示自适应更新在现有实现下未优于固定阈值，修复在低噪声下表现不佳，而在高噪声下在保持语义要义方面略优。

**⚠️ 局限性**

局限性包括：IET 更新公式过于温和导致阈值变化幅度不足；使用单一句子嵌入距离作为偏差信号可能不足以捕捉更细粒度的对话差异；实验仅在文本对话上验证，缺乏多模态或更复杂情境的检验。

---

## 70. Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI: A Cross-Dataset Case Study

**arXiv ID:** 2607.15416 | [PDF](https://arxiv.org/pdf/2607.15416v1)

**作者:** Parham Hajishafiezahramini `[一作]` (Memorial University of Newfoundland), Edward Kendall `[通讯]` (Memorial University of Newfoundland)

**通讯引用:** 1713 | [OpenAlex ID](https://openalex.org/A5103048268)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究者通过在真实筛查人群（NLBSD）中添加外部异常丰富的数据集（CBIS‑DDSM和CMMD）来训练乳腺X光图像分类模型，并评估其对筛查性能的影响。

**💡 创新点**

创新点在于系统地证明在保持相同预处理、归一化和模型架构的前提下，单纯混合外部异常数据会导致筛查模型性能下降，揭示了域偏移与负迁移的严重性，并首次使用三类数据来源分类实验验证了数据集间可分辨性。

**🔧 技术方法**

采用冻结的EfficientNet‑B5编码器（预训练于Mammo‑CLIP），在其上训练线性分类头，并结合轻量几何与光度数据增强。

**📊 数据集**

使用的三大数据集为：真实筛查数据集NLBSD（5997例，低癌症患病率）；异常增强的CBIS‑DDSM（1644例，含病理确诊）；以及CMMD（1775例，亦为病理确诊）。

**📈 对比分析**

与仅使用NLBSD训练的基线模型（AUC‑ROC 0.737）相比，加入外部正样本后AUC降至0.644–0.620，且性能随外部数据量增加而单调下降；在混合域测试集上亦未能超过基线。

**⚠️ 局限性**

局限性包括仅使用单一冻结的特征提取器和线性探针，未尝试微调或更复杂的多视图网络；仅评估了单一筛查数据集，未验证结果对其他真实筛查数据的泛化；实验未探讨可能的域适配或归一化策略。

---

## 71. FLINT: Fingerprinting Federated Learning Architectures from 5G PHY-Layer Side Channels

**arXiv ID:** 2607.15469 | [PDF](https://arxiv.org/pdf/2607.15469v1)

**作者:** Md Nahid Hasan Shuvo `[一作]` (George Mason University), Moinul Hossain `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 FLINT，一种能够从 5G PHY‑层 PDCCH 调度信息中黑盒识别联邦学习客户端所使用的模型架构（CNN、RNN、Transformer）的框架。

**💡 创新点**

创新点在于①在缺乏网络层可见性的 5G 环境下利用调度元数据实现模型架构指纹；②设计了 RNTI‑to‑UE 的映射算法以重建设备身份；③构建了多视角时间序列特征来捕捉不同模型的训练节奏。

**🔧 技术方法**

使用了基于 srsRAN 的软件无线电收集 PHY‑层调度记录，结合 CC‑CDE RNTI 关联、缺失授予重建、以及多分辨率能量、周期性与序列视角的机器学习分类器。

**📊 数据集**

实验数据来自真实的 srsRAN 5G 基站与客户端设备组成的联邦学习测试床，涵盖 CNN、RNN 与 Transformer 三大架构的多轮训练过程。

**📈 对比分析**

与仅使用单一物理层统计特征的基线相比，FLINT 的宏 F1‑score 达到 0.93，显示出显著的性能提升，并在开源式拒绝与下游攻击场景中验证了其实用性。

**⚠️ 局限性**

局限性包括对物理层调度记录完整度的依赖、对 RNTI 变动的精确映射需要较长观测窗口，以及在极端信号干扰或网络拥塞情况下可能出现的误检或误分情况。

---

## 72. Lazy Arithmetic using Systolic Arrays for Closing the Verification Gap on Embedded Systems

**arXiv ID:** 2607.15328 | [PDF](https://arxiv.org/pdf/2607.15328v1)

**作者:** Taisa Kushner `[一作]` (Galois Inc), Martin Brain `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种动态自适应实时量化算法（DARQ）和相应的基于流水阵列的硬件实现，利用惰性左到右算术实现可验证的深度神经网络推理。

**💡 创新点**

核心创新包括惰性左到右的多词浮点量化、基于三进制流的实数表示、首位显式的流水阵列设计以及对最高有效位的三重复制防护。

**🔧 技术方法**

使用惰性多词浮点、连分数与连对数、三进制流算法、流水阵列以及三重复制等技术进行量化与硬件加速。

**📊 数据集**

在Fashion-MNIST图像分类数据集上评估，并对其他标准DNN模型进行实验。

**📈 对比分析**

与传统静态量化和基于评估硬件的验证方法相比，实验显示量化误差极少但影响显著，且在大多数层仅需1个最高有效位即可完成计算，显著降低了功耗和存储需求。

**⚠️ 局限性**

仍处于工作进展阶段，硬件实现尚未完成，实验范围有限，未覆盖所有DNN架构和边缘环境的完整验证。

---

## 73. pyoptexplain: A Python Library for Post-Optimality Analysis and Explanation of Optimization Models

**arXiv ID:** 2607.15470 | [PDF](https://arxiv.org/pdf/2607.15470v1)

**作者:** Hussein Fellahi `[一作]` `[通讯]`, Hussein Fellahi

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 PyOptexplain，一个面向实践者的 Python 库，用来在多种建模语言（cvxpy、Pyomo、gurobipy、docplex、OR‑Tools）与多种求解器（HiGHS、Gurobi、CPLEX、SCIP、OSQP、Clarabel、SCS、ipopt 等）之间统一、可靠地执行后最优性分析与情景研究。

**💡 创新点**

核心创新点包括：
• 通过“表示证书（Representation Certificate）”与能力门控（Capability Gating）实现对分析结果的可信保证，避免求解器差异和结构不匹配导致的误报；
• 采用一次性提取模型结构并在情景批量求解中复用温启动（warm start）和单一证书，从而将多情景求解成本降到近乎一次性求解的常数倍；
• 在同一内部表示上实现跨建模语言、跨求解器的统一接口，使得同一模型在不同环境下得到完全一致的分析报告。

**🔧 技术方法**

主要技术包括：
• 对模型进行结构化归一化（LP、QP、MIQP 形式）并维护块-组件映射；
• 设计多层管道：ProblemHandle → AnalysisRepresentation → RepresentationCertificate → Analyzer；
• 在 Analyzer 中实现方法存在、证书状态、后端支持三层门控；
• 通过 TypedChanges 与 BlockExperiments 定义可变更操作，实现高效情景与网格分析；
• 利用 solver 的 Warm Start 与内部矩阵接口（如 HiGHS、OSQP 等）实现高效重解。

**📊 数据集**

评估使用的数据集包括：
• 随机生成的线性、二次、混合整数与混合二次程序；
• 经典 Netlib LP（如 AFIRO）；
• 真实案例（产品混合 LP、通用分配 MILP、均值‑方差投资 QP）；
• 二阶锥、指数锥与非线性约束模型，验证非可提取情景下的 native 方案。

**📈 对比分析**

对比方法：
• 与裸求解器（仅调用求解器 API）比较，测量建模、求解与报告阶段的时间；
• 对多情景分析，比较三种实现层次（naïve、re‑extract、reuse）和结构化 vs 本地情景表示的开销；
• 在不同求解器与语言下测量场景批量求解的总耗时与单次求解的比例。结果显示：
  • 统一分析层的开销在大规模模型中可低于 1.2 倍裸求解；
  • 情景复用后每个案例的成本平均可低 10× 左右（尤其对 LP、QP 情景），而对 solve‑dominated MIP/NLP 无明显收益；
  • 结构化情景表示在可提取问题上始终最优，且能避免多次建模语言转换。

**⚠️ 局限性**

局限性：
• 对非可提取（非线性、二阶锥等）模型只能使用 native 情景表示，失去结构化复用优势；
• 需要求解器支持对应的后最优诊断（如基底、边界敏感性），不支持的功能只能返回 None 或报错；
• 计算机资源与单机内存限制在极大规模模型（>百万变量/约束）下仍受限；
• 目前仅覆盖 LP/QP/MILP/MIQP，未覆盖非凸、非线性求解的完整后最优分析；
• 依赖各建模语言的 Canonicalization 与接口，若求解器或语言更新导致 API 变化需维护。

---

## 74. Rethinking Transfer in Continual Learning: A Replay-Based Realisation

**arXiv ID:** 2607.15587 | [PDF](https://arxiv.org/pdf/2607.15587v1)

**作者:** Yang Meng `[一作]` (University of Chicago), Yuxin Chen `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低预算的连续学习任务中，提出并验证了一个框架，用于判断何时能实现前向迁移，并以此为基础设计了一种利用任务签名进行数据重放选择的算法（routed triple），将重放从保持过去性能转向帮助学习新任务，并用知识蒸馏实现稳定性。

**💡 创新点**

核心创新包括：①将前向迁移可行性拆解为“头部空间（headroom）”“持久载体（persistent carrier）”“源选择（source selection）”三条件；②提出基于梯度方向的任务签名，可在无训练成本下预测最优源；③将重放作为迁移载体，重新分配蒸馏负责稳定性，彻底改变传统重放与蒸馏的角色。

**🔧 技术方法**

使用技术：LoRA适配器的在线微调、基于梯度平均的任务签名、softmax路由选择、50%比例的数据重放、KL蒸馏正则化；实验中使用了Qwen2.5、Gemma3、Llama3.2等LLM，评估框架包含连续学习的整体准确率、即时准确率（plasticity）和后向迁移（BWT）。

**📊 数据集**

数据集：TRACE‑8（8个不同主题任务）和NumGLUE‑8（8种数理推理格式），每个任务仅提供50个标记样本，符合低预算设置。

**📈 对比分析**

与传统方法（SeqFT、EWC、LwF、SDFT、O‑LoRA、GainLoRA、DEAL、A‑GEM、DER++、ER）在同一协议下对比，routed triple在整体准确率、即时准确率和BWT上均优于所有基线，尤其在低预算下实现了正向迁移并保持零遗忘。

**⚠️ 局限性**

局限性包括：①仅在低预算场景验证，无法保证在高预算或无数据限制下同样有效；②需要存储过去任务数据，受隐私/存储约束；③在同质任务集合中签名区分度下降；④需要先进行headroom审计，非所有基准都满足；⑤对不同模态（视觉、多模态）的推广仍需研究。

---

## 75. How Does Empowering Users with Greater System Control Affect News Filter Bubbles?

**arXiv ID:** 2607.15284 | [PDF](https://arxiv.org/pdf/2607.15284v1)

**作者:** Ping Liu `[一作]` (LinkedIn Corporation), Mustafa Bilgic `[通讯]` (Illinois Institute of Technology)

**通讯引用:** 5043 | [OpenAlex ID](https://openalex.org/A5038378351)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种增强透明度和交互的政治新闻推荐系统，让用户能直接调整推荐偏好以减缓过滤气泡。

**💡 创新点**

提供多维度政治兴趣与立场滑块界面，让用户可同时设定不同议题的倾向和兴趣，实现更细粒度控制。

**🔧 技术方法**

基于文本TF-IDF + 逻辑回归的内容推荐模型，再结合用户兴趣匹配的加权评分。

**📊 数据集**

使用美国2019-2020年政治新闻数据集，共4万篇文章，按5个政治立场和44k标签。

**📈 对比分析**

通过102位AMT用户的实验，将增强UI与传统投票UI对比，评估极端程度、多样性、准确率等指标，发现增强UI能更大幅度改变极端程度，但多样性下降。

**⚠️ 局限性**

样本量有限、实验时间短，缺乏长期自然环境验证；界面复杂性可能影响用户体验。

---

## 76. From Black Box to Executable Logic: Explainable Reinforcement Learning through Prolog Expert Systems

**arXiv ID:** 2607.15459 | [PDF](https://arxiv.org/pdf/2607.15459v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` `[通讯]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán (Universidad Pontificia Comillas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将冻结的深度强化学习策略转换为可执行的Prolog逻辑程序，并通过精确回报的扩展提升至超过教师。

**💡 创新点**

提出三阶段后置转换（提取、归纳、扩展）并证明可验证的回报损失上界和扩展循环的单调性，首次在离散和连续观测空间中实现可执行可编辑的逻辑学生。

**🔧 技术方法**

基于PPO教师的贪婪动作查询、FOIL式规则归纳、Prolog决策列表、Exact Bellman求解、DAgger在线学习、阈值网格化、蒙特卡罗评估以及精确返回oracle。

**📊 数据集**

KeyDoor网格世界（可枚举状态）以及连续控制任务CartPole、Acrobot、LunarLander。

**📈 对比分析**

通过Exact J、成功率对比，并使用bootstrap置信区间和Wilcoxon检验；在KeyDoor上扩展程序达到最优回报，在预算有限的教师下超越随机教师；在连续任务中，Prolog规则在CartPole和Acrobot恢复近90%+回报，在LunarLander仅部分恢复。

**⚠️ 局限性**

规模限制：需要枚举状态和精确模型；对谓词词表固定，缺乏谓词生成；回报损失上界在高折扣率下松散；连续任务需阈值网格，精度受维度指数影响。

---

## 77. CASAband: Easy-to-Wear Textile Wristband using Shape Memory Alloy Actuators for Spatial and Temporal Haptic Feedback

**arXiv ID:** 2607.15533 | [PDF](https://arxiv.org/pdf/2607.15533v1)

**作者:** Baekgyeom Kim `[一作]` (Korea National University Of Transportation), Tania K. Morimoto `[通讯]` (University Of California San Diego)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并制造了一款轻量化可穿戴腕带CASAband，集成合成放大形状记忆合金(SMA)驱动器，能够在腕部提供空间与时间触觉反馈，并通过用户感知实验和实际任务验证其可用性。

**💡 创新点**

将高力‑小尺寸SMA驱动器与多层织物结构结合，首次实现无缠绕、低噪音、带宽超过1 Hz、总质量63 g的热驱动腕带，可生成多种触觉模式并适用于真实环境导航。

**🔧 技术方法**

采用形状记忆合金驱动器、热可压缩复合梁放大机制、CO₂激光切割与热熔层层织物制成、Arduino Nano 33 BLE蓝牙无线通信、内置IMU手势识别、锂聚合物电池、力/位移/温度传感器等技术。

**📊 数据集**

通过10名参与者进行定位与模式识别的感知实验，收集定位准确率、模式识别准确率等数据；在户外1 km路径演示中记录GPS轨迹；未使用公开数据集。

**📈 对比分析**

与现有腕/臂部触觉设备在力、位移、带宽、重量等指标进行对比；单点定位准确率>90%，七种模式识别>92%；电池续航超过4 h；1 km导航成功率100%，任务绘制成功率88%。

**⚠️ 局限性**

设备寿命约1000个循环；连接线厚度影响织物厚度；缺乏闭环力/位移感知与自适应控制；仅在实验室与短期户外场景验证，需进一步提升耐久性与通用性；未来可集成导电织物或可拉伸电子以简化制造与维护。

---

## 78. Rethinking the Readout: Unlocking Video Backbones for AI-Generated Video Detection

**arXiv ID:** 2607.15321 | [PDF](https://arxiv.org/pdf/2607.15321v1)

**作者:** Manni Cui `[一作]` (Huazhong University of Science and Technology), Zhenyu Zhang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 80938 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对AI生成视频检测中的视频骨干网络读取层瓶颈，提出轻量化读取层V-PVP，利用补丁速度场信息提升检测性能。

**💡 创新点**

创新点在于将读取层改造成两路速度门控聚合与通道保持流，以保持补丁间时空差异和通道能量，解决传统聚合消除局部时空细节的问题。

**🔧 技术方法**

采用补丁速度聚合、速度门控注意力、通道保真非线性、轻量化线性投影和短卷积时序头等技术。

**📊 数据集**

使用AIGVDBench与GenVidBench-143k两个跨生成器基准，训练集含OpenSora、CogVideoX1.5等，测试集包含20+生成器。

**📈 对比分析**

与多种基线对比，V-PVP在冻结VideoMAE骨干上实现95.28 AUC（AIGVDBench）和93.75 AUC（GenVidBench-143k），显著优于传统CLS/GAP读取层和全参数微调。

**⚠️ 局限性**

局限性包括生成器模仿真实时序统计导致速度门控失效、短视频帧数不足、骨干预训练与目标域不匹配时性能下降，以及对CNN骨干的瓶颈未完全解决。

---

## 79. SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction

**arXiv ID:** 2607.15550 | [PDF](https://arxiv.org/pdf/2607.15550v1)

**作者:** Xue Yu `[一作]`, Junlan Feng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SeerGuard的后置安全框架，结合指令级过滤与动作级风险评估，利用语义世界模型预判移动GUI交互的后果，主动阻止潜在危险操作。

**💡 创新点**

创新点在于：①将指令过滤与动作预测融合为双阶段安全机制；②采用语义级下一屏预测替代昂贵的像素级生成；③通过三层安全增量数据（文本安全、视图风险、合成文本）和多任务学习构建统一的安全增强世界模型（SAWM）。

**🔧 技术方法**

技术实现包括多模态大语言模型Qwen3‑VL‑8B‑Instruct、Supervised Fine‑Tuning、语义世界模型推理、指令与动作级风险标签生成、以及基于安全链式思维（SCoT）与传统防御（Direct）做对比。

**📊 数据集**

使用的数据集涵盖：MobileWorldBench（语义状态转移）、MobileSafetyBench（250个任务、150高危/100低危）、Agent‑SafetyBench与Prompt Injection（指令安全评估）、MobileRisk（动作级风险评估）、Next‑State‑QA（未来屏幕预测）以及自研的合成安全文本与多模态安全标注。

**📈 对比分析**

通过在三种主流GUI代理（Qwen3‑VL、GPT‑5.1、Gemini‑3.1）上评估，SeerGuard相较于Direct和SCoT显著降低RCS（高危任务误完成）并提升SUS（高危拒绝+低危完成），在所有α、ω设定下均实现最优或次优的安全/效能平衡，并在动作级评估上获得最高F1与Step Score。

**⚠️ 局限性**

局限性包括：对细粒度任务的风险检测仍有漏报；对低危任务的误拒率略高；对某些模型（如GPT‑5.1）在金融类任务中的提升有限；依赖人工标注与合成数据的安全知识，可能在未见场景下表现不佳。

---

## 80. SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification

**arXiv ID:** 2607.15517 | [PDF](https://arxiv.org/pdf/2607.15517v1)

**作者:** Bibesh Pyakurel `[一作]` (University of Wisconsin--Green Bay), M. G. Sarwar Murshed `[通讯]` (University of Wisconsin--Green Bay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SLAPBench 基准，用 NIST SD302b 数据评估多模态大型语言模型在四指 SLAP 指纹验证任务上的性能。

**💡 创新点**

首次构建 SLAP 指纹验证基准，系统探讨提示策略对模型崩溃的影响，并通过连续相似度评分揭示模型区分能力与提示方式的关系。

**🔧 技术方法**

使用多模态 LLM（InternVL3、Qwen 系列、Gemma、Claude Opus）进行图像+文本推理，采用二元提示、任务描述提示和连续相似度评分提示，解析输出并计算 FAR、FRR、EER、AUC 等指标。

**📊 数据集**

基于 NIST SD302b（201 位参与者）构建 7,832 对（176 同源跨分辨率，7,656 异源同分辨率），并提供性别、种族、年龄等元数据。

**📈 对比分析**

在二元提示下，绝大多数开源模型崩溃（FAR>96%），Claude Opus 4.8 仅为不崩溃的模型；采用相似度评分后模型区分度从 AUC 0.590 到 1.000；Qwen3‑VL‑8B 取得完美分离，但被视为诊断性结果。

**⚠️ 局限性**

单次捕获导致近似重复检测难以区分身份匹配；评估仅覆盖单一专有模型，公平性分析样本有限；基准与传统指纹匹配器未做绝对对标，结果仅为相对参考。

---

## 81. LLM-Driven AutoML for Cross-Lingual Handwritten OCR: Closed-Loop Neural Architecture Search with GPT-5, GPT-4o, and Claude Sonnet 4

**arXiv ID:** 2607.15509 | [PDF](https://arxiv.org/pdf/2607.15509v1)

**作者:** Mobina Kashaniyan `[一作]` (Iran University of Science and Technology), Nasser Mozayani `[通讯]` (Iran University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种完全自动、跨语种的手写文本识别系统，利用GPT‑5、GPT‑4o和Claude Sonnet 4作为架构设计者，在闭环反馈中生成、评估并迭代优化神经网络模型，最终在阿拉伯语、英语和波斯语上实现高精度识别。

**💡 创新点**

创新点在于：①将大型语言模型直接用作神经架构搜索（NAS）的生成器和调优者；②构建无人工干预的闭环反馈机制，使模型在每轮实验后自行改进；③实现脚本无关的统一框架，并在单个实验中同时对三种不同书写系统进行优化。

**🔧 技术方法**

采用技术包括：大型语言模型（GPT‑5、GPT‑4o、Claude Sonnet 4）生成JSON形式的网络结构与训练超参数；Keras 进行模型构建与训练；可选的Vision Transformer块以探索不同特征混合方式；基于训练、验证、测试指标的闭环反馈；轻量级数据增强与自动化数据加载。

**📊 数据集**

使用的数据集为：英语采用 EMNIST；波斯语采用 SADRI；阿拉伯语采用 AHCD；三套数据均经过标准化切分、轻量级增强后供模型训练与评估。

**📈 对比分析**

与传统手工设计和 Cerescu & Bumbu 的单次评估方法相比，本工作在每种语言上进行 30 次独立试验，平均测试准确率达到 93.7%–95.4%，最佳试验可达 98.1%，模型参数量仅 0.88–2.74 M，推理延迟约 41–44 ms，证明了高效、实时的性能。

**⚠️ 局限性**

局限性包括：仅覆盖三种脚本，尚未扩展至更复杂的连体字或完整行/段落识别；对硬件约束的适配不完整；依赖大型语言模型的可用性与成本；在极少数数据不足或极端手写变异的场景下，模型鲁棒性仍待进一步验证。

---

## 82. Cache-Aware Prompt Compression:A Two-Tier Cost Model for LLM API Caching

**arXiv ID:** 2607.15516 | [PDF](https://arxiv.org/pdf/2607.15516v1)

**作者:** Yan Song `[一作]` `[通讯]` (PayPal Inc), Yan Song (PayPal Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 LLM 部署中 prompt 缓存和压缩的交互，并提出了 Cache‑Aware Prompt Compression (CAPC) 方案；

**💡 创新点**

创新点在于基于实验测得的两层缓存模型和写入/读取成本差异，构建了跨策略成本模型，预测并验证查询感知压缩在高压缩比下低效，并提出了层级保留比率约束的 CAPC 算法；

**🔧 技术方法**

采用的技术包括基于 Anthropic Sonnet 4.6 API 的缓存行为量化、成本建模、查询无关压缩（如 Cmprsr）、自适应缓存边界算法、以及多场景实验（LongBench‑v2、企业工具助手、graphify RAG、τ‑bench retail）；

**📊 数据集**

使用的数据集包括 Synthetic LongBench‑v2 文档、企业工具助手的 94k‑token schema、FastAPI 与 httpx 代码库的 graphify 构建知识图谱、以及公开的 τ‑bench retail benchmark；

**📈 对比分析**

通过在 16/16 配置上和 Pareto 前沿扫描与多任务奖励评估进行对比，CAPC 在所有配置下均低于其他三种策略，平均成本下降约 48%–90%，质量基本保持不变；

**⚠️ 局限性**

局限性包括对单一模型和 5‑分钟 TTL 的依赖、对查询感知压缩的“突变比例”未完全量化、以及对不同供应商缓存实现的可迁移性待验证。

---

## 83. qZACH-ViT: Quantization-Aware Intrinsic Explanations with Recursive Attribution-Stabilized Optimization

**arXiv ID:** 2607.15421 | [PDF](https://arxiv.org/pdf/2607.15421v1)

**作者:** Athanasios Angelakis `[一作]` (Research Institute CODE), Athanasios Angelakis `[通讯]` (Amsterdam UMC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低数据医学图像分类任务中，设计并训练了量化感知的qZACH‑ViT模型，并提出了递归归因稳定化优化（RASO）以提升解释一致性。

**💡 创新点**

提出了零token无位置编码的ViT架构扩展，支持递归层级的可解释补丁证据，并将归因梯度与分类梯度进行范数匹配并投影，形成专门的优化策略。

**🔧 技术方法**

使用W8A8量化感知训练、INT8混合精度ONNX导出、递归归因机制、归因梯度范数匹配与投影（RASO）、多指标XAI评估（Deletion、Insertion、SaCo、纯量化与随机化检验）等技术。

**📊 数据集**

在MedMNIST七个医学图像子任务（BloodMNIST、PathMNIST、BreastMNIST、PneumoniaMNIST、DermaMNIST、OCTMNIST、OrganAMNIST）上，用每类50张图像、10个随机种子进行低数据实验。

**📈 对比分析**

将量化后的qZACH‑ViT与FP32基线、量化加归因损失、RASO三种配置在ONNX INT8上进行部署，结果显示所有配置的平均指标均超过FP32基线，RASO在大多数数据集上获得最大平均提升，预测一致率99.98%，模型尺寸缩减70%，CPU单线程加速1.41×、四线程2.39×。

**⚠️ 局限性**

主要局限包括混合精度非全整数推理、运行时栈混淆、缺乏真实定位标签、仅使用MedMNIST内置数据、未覆盖更广泛硬件与优化器比较，以及对归因地图的空间分辨率受限等。

---

## 84. On the Impact of Entropy-based Features

**arXiv ID:** 2607.15379 | [PDF](https://arxiv.org/pdf/2607.15379v1)

**作者:** Iuri Mundstock `[一作]` (Federal University of Rio Grande), Bruno L. Dalmazo `[通讯]` (Federal University of Rio Grande)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在传统统计特征难以充分捕捉网络流量多样性和变异性的情况下，本文提出将熵作为附加特征融入监督式网络流量分类的机器学习管道。

**💡 创新点**

创新点在于利用熵值来量化选定流量属性的变异性，作为对传统特征的补充，而非替代，从而实现轻量且易解释的特征工程。

**🔧 技术方法**

使用监督学习模型（如随机森林、SVM 等）结合熵特征；熵特征通过对网络流量属性计算熵值得到。

**📊 数据集**

实验采用公开的入侵检测数据集进行评估。

**📈 对比分析**

对比了加入熵特征和不加入熵特征的模型，结果显示加入熵后分类准确率提升，混淆矩阵显示误分类率下降，尤其是在高变异性流量场景，且额外计算成本低。

**⚠️ 局限性**

局限性主要在于仅在单一数据集上验证，熵特征的效果可能受数据分布影响；对实时系统的计算开销虽低但仍需评估；未探索熵特征与所有攻击类型的适用性。

---

## 85. An Auto-Scaling Approach for Serverless Environments Based on a Multi-Expert Consensus Mechanism

**arXiv ID:** 2607.15511 | [PDF](https://arxiv.org/pdf/2607.15511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 86. Environment Design for Reliable Shared Autonomy with Probabilistic Guarantees

**arXiv ID:** 2607.15487 | [PDF](https://arxiv.org/pdf/2607.15487v1)

**作者:** Yi-Shiuan Tung `[一作]` (University of Colorado Boulder), Alessandro Roncone `[通讯]` (University of Colorado Boulder)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出通过优化桌面物体布局来提升共享自治系统中用户意图推断的准确性与速度，并给出一个基于边际裕度的概率正确性保证。

**💡 创新点**

创新点在于①将工作空间设计建模为目标可分离度优化问题；②在噪声约束下推导出可满足 1−α 置信度的概率正确性条件；③使用 MAP‑Elites 与 CMA‑ME 搜索多模态布局空间，获得多样化的高质量布局；④在仿真与真实场景中验证该方法对推断性能的提升。

**🔧 技术方法**

技术方法包括：基于 Boltzmann‑rational 观察者的目标后验推断；假设 joystick 输入服从带协方差的 Gaussian 噪声并在线性化边际裕度；利用 MAP‑Elites（CMA‑ME）在连续布局空间中进行多目标优化；仿真平台 MuJoCo；机器人控制采用 Sawyer 的 Jacobian 伪逆速度控制。

**📊 数据集**

实验数据集主要是六个桌面操作场景（Easy、Medium、Hard 共 6 组），每组包含 2–8 个候选目标物体，仿真采用 MuJoCo 进行 30 秒以内的采样；此外作者在真实实验中演示了制茶、乐高分类与助食三种场景，使用的物体为实际物品。

**📈 对比分析**

比较方法：对每个场景生成 30 个随机布局（baseline）和一个 MAP‑Elites 优化得到的 elite 布局；使用 argmax 准确率与推断所需时间作为指标。结果显示，优化布局在所有场景中均获得更高的准确率（多为 100%）且在中等难度场景推断时间更短；在高目标数的混乱场景中，准确率提升更明显，但推断时间差距减小。

**⚠️ 局限性**

局限性包括：①优化过程中需要较长的计算时间（340–920 秒）；②对噪声的线性化假设在大偏差下可能失效；③方法假设候选目标集合已知且固定，未考虑在线更新；④在极其拥挤的布局中，优化后的准确率仍有下降；⑤尚未在大规模真实用户研究中验证用户体验与信任提升。

---

## 87. Who Became Financially Vulnerable After COVID-19? A Population-Level Machine Learning Analysis Using MEPS Data

**arXiv ID:** 2607.15446 | [PDF](https://arxiv.org/pdf/2607.15446v1)

**作者:** Alexey Kresin `[一作]` (Hood College), Nawar Shara `[通讯]` (MedStar Health Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了COVID-19前后美国医疗费用高额负担的分布及其预测关系，利用2019年和2021年的MEPS数据进行分层描述、逻辑回归和机器学习建模，并评估跨时段泛化。

**💡 创新点**

创新在于将传统分层与可解释的统计模型与树形机器学习相结合，并对模型进行跨时期泛化实验，验证预测关系的时间稳定性。

**🔧 技术方法**

使用逻辑回归、随机森林和梯度提升（XGBoost）等解释性与非解释性模型，并采用特征置换重要性、ROC‑AUC、准确率、召回率、精确率、F1等指标进行评估。

**📊 数据集**

采用医疗支出面板调查（MEPS）全年合并文件，分别为2019年（前疫情）和2021年（后疫情）共54,990个个体。

**📈 对比分析**

在同一年内训练和测试模型，ROC‑AUC约为0.86；在2019年训练、2021年测试时ROC‑AUC下降到0.846，说明模型泛化良好；梯度提升得到最高ROC‑AUC但召回率低，随机森林在召回率和F1上表现最佳。

**⚠️ 局限性**

主要局限包括：观察性设计无法推断因果，交叉截面样本不连续；未包含地理、数字接入等重要背景变量；高负担定义基于10%阈值，可能低估未就医人群；模型评估受样本不平衡影响。

---

## 88. The Multiple-Choice Matroid Secretary Problem

**arXiv ID:** 2607.15407 | [PDF](https://arxiv.org/pdf/2607.15407v1)

**作者:** Matías Ortiz-Angel `[一作]` (Universidad de Chile), José A. Soto `[通讯]` (Universidad de Chile)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并研究了多选Matroid Secretary Problem（J,κ)-MSP，即在随机顺序下，算法可维护一个候选池，池必须可被 J 条可行解覆盖并满足全局容量约束 κ·rank，最终返回池中最大权重的独立子集。

**💡 创新点**

创新点在于：① 给出对齐于通用匹配的多轨路由算法，并证明其在跨切（transversal）Matroid 上达到 J 选择秘书问题的最优成功概率；② 通过标签方案（labeling scheme）与 Poisson 过程的结合，给出单阈值容量约束下的显式概率竞争比；③ 将方法推广到 k-列稀疏与层次（laminar）Matroid，分别给出多轨与基于联合（union-based）的算法，并给出对应的概率竞争下界。

**🔧 技术方法**

主要技术包括：连续时间随机顺序映射、改进元素的标签方案、Poisson 过程与独立增量性质、Gilbert–Mosteller 多阈值策略、Chernoff-大偏差分析、单阈值路由规则、矩阵行冲突分析（k-列稀疏）以及层次链冲突分析。

**📊 数据集**

本文为理论工作，未使用具体数据集；所有结果均在随机顺序下的抽样模型上得到。

**📈 对比分析**

比较方法采用概率竞争（probability-competitive）指标，即对每个独立元素 v，计算被最终输出集包含的概率；在 transversal matroid 上，该算法取得与 J-选择秘书问题相同的成功概率（即最优值 γ_J）；在容量约束下给出显式下界；在 k-列稀疏与 laminar matroid 上给出 1–O(e^{-J/(ke)}) 与 1–O(e^{-J/e}) 的概率竞争比。

**⚠️ 局限性**

局限性包括：① 对 general matroid 的概率竞争比仍未知；② 结果多基于随机顺序和全局容量上限；③ 对高阶参数（J、κ、b、q）解析仅给出上界或单阈值下的近似；④ 对实数阈值的最优选择依赖于已知的 J-选择秘书问题阈值。

---

## 89. AEGIS: Assay-Aware Protocol Validation and Runtime Monitoring for Open-Source Liquid Handling Robots

**arXiv ID:** 2607.15620 | [PDF](https://arxiv.org/pdf/2607.15620v1)

**作者:** Priyanka V. Setty `[一作]`, Rick Stevens `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Opentrons OT‑2液体处理机器人上实现了两层守护系统（AEGIS），先对Python协议进行assay‑aware规则检查，再通过前视摄像头实时监测管路轨迹并检测物理执行失效；

**💡 创新点**

创新点在于将预飞行的assay规则验证与运行时的视觉异常检测结合成两层“深浅”模型，且通过规则驱动的LLM检查和PCA+VLM级联判定实现低成本、开源且可扩展的守护方案；

**🔧 技术方法**

技术包括：基于JSON的assay规则库；单次LLM推理（o4-mini、Claude、GPT‑4o、NVIDIA Nemotron‑3 Ultra）对协议代码进行语义分析；YOLOv8‑nano实现管尖定位；PCA构建“正常轨迹”模型进行异常评分；VLM（Claude Opus）做灰度区判定；按每板校准的median±0.7·MAD阈值；

**📊 数据集**

数据集为：24个OT‑2协议（5类assay）与13个注入bug的单一bug变体、3个多bug协议；64个p1000轨迹（26正常+38失效），包括红、黄、无水三种液体；p20轨迹用于探测分辨率极限；

**📈 对比分析**

方法对比：Layer 1在所有5种LLM后均实现100% bug召回，F1≈0.97；Layer 2在离线留一板交叉验证中，AP≈0.89，F1≈0.71，AUROC≈0.80；实时演示下，cascade模式对染色液体的partial‑dispense召回≈0.6，always‑VLM提高到1.0；成本方面，cascade每96孔板约$1.63，always‑VLM约$10.33；

**⚠️ 局限性**

限制包括：对p20和透明液体（如水）检测性能低下（F1≈0.47/0.47）；监测延迟≈9.5 s，可能不足以阻止连用同一管头的连续失效；缺少对气泡等细微失效的实时检测；以及对多协议调度和多台机器人扩展的评估仍未完成。

---

## 90. Information-Directed Sampling for Causal Bandits

**arXiv ID:** 2607.15577 | [PDF](https://arxiv.org/pdf/2607.15577v1)

**作者:** Muhammad Qasim Elahi `[一作]`, Mahsa Ghasemi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在存在不可操纵变量的上下文因果抽样问题，并提出基于贝叶斯后验的因果Thompson Sampling和信息导向采样（IDS）算法。

**💡 创新点**

创新点在于：①将观测分布的条件概率表作为未知参数，利用因果结构共享信息；②在带上下文且含不可操纵变量的环境下给出熵依赖的子线性贝叶斯退化率上界；③对IDS给出基于Monte‑Carlo估计的额外误差分析与置信界。

**🔧 技术方法**

主要技术包括：结构因果模型、贝叶斯后验更新、Dirichlet先验、信息导向采样与Thompson Sampling、Monte‑Carlo估计、信息增益与互信息分析。

**📊 数据集**

实验使用人工合成的三种结构化因果图（含可/不可操纵节点与上下文）以及随机生成的Erdős–Rényi稀疏/密集图，全部采用随机抽取的条件概率表作为数据来源。

**📈 对比分析**

与UCB、传统Thompson Sampling以及基于z²ID的最小方差加权方法进行比较，实验结果显示所提TS和IDS在累计退化率上明显优于基线，IDS在更充分利用信息的情况下表现最优。

**⚠️ 局限性**

局限性包括：①假设因果图已知且无潜在混杂；②需要对后验进行大量Monte‑Carlo采样，计算成本较高；③对非离散/非可观测变量的适用性有限。

---

## 91. FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression

**arXiv ID:** 2607.15413 | [PDF](https://arxiv.org/pdf/2607.15413v1)

**作者:** Jiajun Huang `[一作]` (University of South Florida), Jiajun Huang `[通讯]` (University of South Florida)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5100779661)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种单 CUDA 核实现的 GPU 错误界限无损压缩器，旨在同时提升压缩比和吞吐量，突破传统预测-吞吐量权衡。

**💡 创新点**

创新点：
1) 跨块预测状态（Cross‑Block Prediction）把 8 个 32 元素块连成 256 元素的连续预测链，消除 7/8 的边界残差。
2) 自适应多阶预测与居中（Adaptive Multi‑Order + Centering）在每块内按需选择 LZ1、LZ2 及其居中变体，以适配不同局部数据特征。
3) 单遍四路评估（Single‑Pass Four‑Way Evaluation）利用有限差分对常数偏移抵消的数学性质，只读一次数据即可评估四种预测方案，避免额外带宽消耗。

**🔧 技术方法**

技术细节：
- Lorenzo 预测（LZ1/LZ2）与量化；
- 量化后在寄存器中保持跨块状态；
- 采用 bit‑shuffle 固定长度编码；
- 三层层级前缀和（warp、thread、device）通过 decoupled lookback 实现；
- warp 级 shuffle、PTX 级量化、向量化加载、无分支自适应选择。
- 单遍四路评估利用常数偏移在高阶差分中消失的性质，避免多次全局内存读。

**📊 数据集**

数据集：8 个真实科研数据集（CESM‑ATM、EXAALT、HACC、Hurricane、Miranda、NYX、Truss、SCALE），共 84 个字段，使用三种相对误差边界 REL 1e-2、1e-3、1e-4 进行测试。

**📈 对比分析**

评估方法：与 cuSZp、FZ‑GPU、cuZFP 及其他纯 GPU 压缩器在相同 GPU（GH200、A100）上进行压缩比、吞吐量、误差保证等多维度对比。结果显示：
- 在所有数据集和误差边界下压缩比最高，分别比 cuSZp 提升至 2.92×，比 FZ‑GPU 提升至 10.95×；
- 压缩吞吐量达到 676 GB/s（压缩）/ 785 GB/s（解压），为评估的所有压缩器中最高；
- 在 A100 上同样实现显著加速，压缩吞吐比 cuSZp 高 1.46×，解压 1.49×。

**⚠️ 局限性**

限制：
- 目前仅针对 NVIDIA GPU 进行设计与实现，未验证对其他 GPU/加速器的可移植性；
- 对极小块尺寸或非常高误差边界时的预测效果受限；
- 跨块状态虽不影响吞吐，但在极大块/极长链时可能导致寄存器压力与并行度下降；
- 该方法依赖单核块/线程模型，若需跨多 GPU 或多设备协作仍需进一步改造。

---

## 92. Understanding Fortunetelling with Large Language Models in China: User Practices, Perceptions, and Impacts on Beliefs and Decisions

**arXiv ID:** 2607.15626 | [PDF](https://arxiv.org/pdf/2607.15626v1)

**作者:** Xueer Lin `[一作]` (Sun Yat-sen University), Zhenhui Peng `[通讯]` (Sun Yat-sen University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对中国社交媒体中1,045条关于LLM占卜的帖子进行内容分析，并对20名使用者进行半结构式访谈，研究了LLM占卜的使用实践、用户感知与对信念和决策的影响。

**💡 创新点**

创新点在于首次将传统占卜与现代大语言模型结合，系统梳理出LLM占卜的使用主题、提示结构及情感反应，并揭示其在情感支持和决策反思中的作用及对信念微调的潜在影响。

**🔧 技术方法**

使用的技术主要是定性研究方法——开放式编码、主题分析与访谈数据的编码整理；没有开发新模型或算法。

**📊 数据集**

数据集包括：1,045条来自RedNote（Xiaohongshu）和微博的公开占卜相关帖子；以及20名来自作者网络与帖子作者的访谈受访者（15女5男，20-43岁）。

**📈 对比分析**

本文未进行模型性能或数值对比评估，而是以描述性统计和主题归纳方式呈现结果；因此无法给出传统意义上的性能指标。

**⚠️ 局限性**

局限性包括：仅覆盖公开社交媒体，可能遗漏私密使用场景；访谈样本规模有限、可能存在自选偏差；缺乏长期追踪与量化评估；分析聚焦中文传统占卜，文化适用性未扩展。

---

## 93. From hyperplanes to hyperellipsoids: characterizing the inherent interpretability of linear and single-qubit mixed-state binary classification models

**arXiv ID:** 2607.15433 | [PDF](https://arxiv.org/pdf/2607.15433v1)

**作者:** Kaitlin Gili `[一作]` `[通讯]` (QodeX Quantum), Kaitlin Gili (QodeX Quantum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文对传统线性分类模型与单量子比特混态模型在二分类任务中的解释性进行了理论比较。

**💡 创新点**

创新点在于将单量子比特混态模型视为学习超椭圆而非超平面，揭示两者在几何先验与特征重要性偏差上的区别。

**🔧 技术方法**

主要采用量子信息学中的混态表示与期望值计算，并与线性模型的权重与决策边界进行对比分析。

**📊 数据集**

本文未使用实际数据集，而是通过二维 toy 示例来说明模型差异。

**📈 对比分析**

通过理论推导与可视化的 toy 实例对比，两模型的决策边界形状和特征重要性被比较，发现混态模型可分离线性不可分的样本但仍受超椭圆限制。

**⚠️ 局限性**

局限在于仅讨论理论先验和几何形状，未进行真实数据实验；混态模型只能学习超椭圆形决策边界且需对概率权重设下界。

---

## 94. On the Role of Normalization in Binary Iterative Hard Thresholding for 1-bit Compressed Sensing

**arXiv ID:** 2607.15530 | [PDF](https://arxiv.org/pdf/2607.15530v1)

**作者:** Arya Mazumdar `[一作]` (UC San Diego), Prateeti Mukherjee `[通讯]` (UC San Diego)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对一比特压缩感知中的原始BIHT算法进行理论收敛分析，证明在无噪声时仍能达到最优样本复杂度，并在受干扰时给出迭代时间上界；

**💡 创新点**

解决了十余年未解的BIHT收敛问题，明确归一化在无噪声和受噪声场景下的必要性，并提出击中时间收敛定理；

**🔧 技术方法**

使用高斯测量矩阵的限制近似可逆性条件（RAIC）、几何分析与移动参考框架对迭代误差进行严格上界估计；

**📊 数据集**

该研究为纯理论工作，仅基于随机高斯矩阵模型，无实际数据集；

**📈 对比分析**

与已知的归一化BIHT相比，原始BIHT在无噪声下实现相同的样本复杂度，但在受干扰时需额外的击中时间保证；归一化算法在受噪声时更稳健；

**⚠️ 局限性**

局限于高斯随机测量矩阵，未扩展到子高斯或结构化测量；在受干扰情形下不提供最后一次迭代收敛保证。

---

## 95. Computing markings for fuzzy minimax nets over the Gödel structure

**arXiv ID:** 2607.15494 | [PDF](https://arxiv.org/pdf/2607.15494v1)

**作者:** Linh Anh Nguyen `[一作]` `[通讯]` (University of Warsaw), Linh Anh Nguyen (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了一种高效算法，用于在Gödel结构下计算有限模糊最小-最大网络的最大正确标记，并进一步将该算法应用于计算两个有限模糊图之间的最大模糊有向模拟；

**💡 创新点**

在Gödel结构下实现了线性时间（O(m+n)）的最大标记计算算法，并基于此推导出第一种在O((m+n)n)时间内求解模糊有向模拟的算法；

**🔧 技术方法**

采用模糊最小-最大网络模型、Gödel真值结构、优先队列（按值排序）以及构造最小-最大网络的技术；

**📊 数据集**

使用随机生成的模糊最小-最大网络和模糊图数据集（节点数、边数分别控制为稀疏、半稠密、稠密三种情况，权值取自{0.01,0.02,…,1.00}）进行实验；

**📈 对比分析**

与现有文献中的模糊（或精确）模拟/双向模拟算法在时间复杂度和实验运行时间上进行比较，实验结果表明两种算法均与理论复杂度一致，且在实际数据上表现优良；

**⚠️ 局限性**

仅适用于Gödel结构，尚未针对乘积或Łukasiewicz结构给出多项式时间算法，且实验仅验证随机合成数据，未覆盖真实世界模糊图场景。

---

## 96. EpiNarrate: Agentic Generation of Grounded Narratives from Epidemiological Scenario Projections

**arXiv ID:** 2607.15544 | [PDF](https://arxiv.org/pdf/2607.15544v1)

**作者:** Rituparna Datta `[一作]` (University of Virginia), Anil Vullikanti `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种 agentic 框架，用结构化流程生成基于公共卫生情景模型的自然语言叙事；

**💡 创新点**

通过部分顺序结构、比较语法和最大熵兴趣筛选，显著提升事实准确性、覆盖度和信息多样性；

**🔧 技术方法**

结合多阶段链式推理、poset 生成、数据增强、比较语法、MaxEnt/IPF 选择和 LLM 生成；

**📊 数据集**

使用 COVID‑19 Scenario Modeling Hub（SMH）的多轮情景投影数据；

**📈 对比分析**

与直接 LLM 调用、单/多阶段 CoT、Plan‑then‑Generate 等基线相比，M1（事实准确率）保持 100% 以上，M2（覆盖度）提升至 0.78–0.81，M3（风格一致性）与基线相当，整体性能显著优于传统方法；

**⚠️ 局限性**

仍依赖 LLM 生成的验证、MaxEnt 结果可能需人工审核、仅在 SMH 情景设计上测试，需扩展至更广泛的公共卫生场景和不同 LLM 后端。

---

## 97. Causal-Audit: Explicit and Auditable Graph-based Reasoning via Target-Aware Causal Chain Construction

**arXiv ID:** 2607.15281 | [PDF](https://arxiv.org/pdf/2607.15281v1)

**作者:** Su Lan `[一作]` (Griffith University), Alan Wee-Chung Liew `[通讯]` (Griffith University)

**通讯引用:** 6594 | [OpenAlex ID](https://openalex.org/A5091254555)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种显式可审计的因果推理框架，实现了上下文无关干预问答。

**💡 创新点**

关键创新在于目标感知因果图构造、路径级因果证据聚合以及对链路的反事实验证。

**🔧 技术方法**

采用LLM进行变量抽取、因果图扩展、边级验证，并结合符号因果图与路径加权算法。

**📊 数据集**

在DDXPlus-CausalEffect、WIQA方向子集和CauseNet衍生数据集上进行评测。

**📈 对比分析**

与Direct LLM、CoT、GoT、ToT及CDCR-SFT等基线对比，整体准确率提升5–20%以上，表现稳健。

**⚠️ 局限性**

推理成本较高，且输出仍需人工评估，未实现完整因果识别，可能在敏感领域误用。

---

## 98. Scaling Unmodified Multithreaded Applications with Elastic CXL-based Distributed Shared Memory

**arXiv ID:** 2607.15569 | [PDF](https://arxiv.org/pdf/2607.15569v1)

**作者:** Guowei Liu `[一作]` (Tianjin University), Wenyu Qu `[通讯]` (Tianjin University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了基于CXL 3.0的全空间弹性分布式共享内存（DSM）系统，能够在多节点上无修改地执行多线程应用，并实现近线性扩展。

**💡 创新点**

创新点包括：①通过OS‑runtime协同实现全空间共享地址空间，消除手动代码重写；②采用基于延迟的动态页置换策略，利用Exp‑binned P90直方图实现延迟均衡；③引入空间局部性感知的弹性页面管理，动态合并/拆分页面以降低页错误成本并避免假共享。

**🔧 技术方法**

使用技术有：CXL 3.0硬件互连与BISnp缓存一致性；硬件采样性能计数器（AMD IBS）用于收集访问延迟；VMA锚定的页迁移；P90-bin延迟直方图与Gap‑Proportional Volume Scaling；多层位图实现可变大小弹性页面；用户空间运行时拦截线程与内存分配；内核模块提供轻量级页迁移接口；支持SC/RC一致性模型。

**📊 数据集**

实验基准包括5个多线程工作负载（PageRank、Jacobi、Graph500、Streamcluster、Blackscholes），不同规模（4–16 GB、8–16 GB、>16 GB）；LLM推理基准使用Qwen3‑30B‑A3B（≈37 GB）并基于Azure Trace；数据量覆盖从小到大，体现内存密集与计算密集两类。

**📈 对比分析**

通过与CXL‑ONLY、Local‑ONLY DSM（SC、RC、SWAP、DRust‑C）以及Hybrid静态策略（S1、S2）共15种配置对比，使用2节点/4节点、不同线程数评估。结果显示：对比CXL‑ONLY加速1.5×–2.2×；对比传统混合DSM加速1.1×–2.2×；在大数据集上实现近线性扩展，部分工作负载出现超线性（如PageRank 5.1×）。

**⚠️ 局限性**

局限性包括：共享元数据放置在全局CXL内存，缺乏对恶意节点的防护；实验基于单机多NUMA模拟，未验证真实CXL‑3.0多主机部署下的网络延迟和可扩展性；只在x86‑64环境下实现，未覆盖其他体系结构；未对大规模节点数（>4）进行评估。

---

## 99. The k-Sum Lateness Problem on a Single Machine

**arXiv ID:** 2607.15462 | [PDF](https://arxiv.org/pdf/2607.15462v1)

**作者:** Ricardo Arancibia-Castillo `[一作]` (Universidad de Chile), José A. Soto `[通讯]` (Universidad de Chile)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一个单机调度问题，目标是最小化k个最大迟到值的总和，介于最大迟到和总迟到之间。

**💡 创新点**

证明了当k作为输入的一部分时，决策版本是弱NP完全的，并为固定k提供了O(k^2n^(k+2))的算法。

**🔧 技术方法**

使用了组合优化和动态规划技术，结合了对偶表示法和结构性结果。

**📊 数据集**

使用了包含n个作业的调度实例，每个作业都有非负的处理时间和实数的截止日期。

**📈 对比分析**

通过与现有的调度算法进行比较，提出了O(k^2n^(k+2))的算法，性能在固定k的情况下表现良好。

**⚠️ 局限性**

限制在于当k是输入的一部分时，问题是弱NP完全的，且对于一般情况没有简单的调度规则。

---

## 100. Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers

**arXiv ID:** 2607.15456 | [PDF](https://arxiv.org/pdf/2607.15456v1)

**作者:** James O' Neill `[一作]`, Fergal Reid `[通讯]` (Fin AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了循环权重共享的Transformer模型的KV缓存结构，并提出了Looped Latent Attention（LLA）编码器，利用跨循环的低秩潜在向量来压缩每个token、层和头部的键值缓存，从而显著减少推理时的内存占用。

**💡 创新点**

创新点在于：①将缓存的“循环”维度（即不同迭代步长）作为压缩目标，证明其低秩性远优于头部或层维度；②通过SVD初始化的跨循环潜在向量与KL+注意力输出匹配的训练策略；③引入基于学生前缀的on‑policy微调以提升长文本生成稳定性；④提出两种编码器变体（per‑head LLA和极限压缩的-2D），满足不同容量需求。

**🔧 技术方法**

主要技术包括：SVD低秩初始化、KL散度+注意力输出匹配的自监督转换、RoPE后处理、跨循环潜在解码、head‑axis MLA、cross‑layer共享、KV量化、最终循环复用控制、以及对Huginn和Ouro系列模型的迁移学习。

**📊 数据集**

使用的数据集与模型包括：Ouro-1.4B、Ouro-2.6B‑Thinking、Huginn-3.5B；评估任务包括GSM8K、MATH‑500、Code/BBH、Commonsense、知识问答、以及随机检索与序列截断测试。

**📈 对比分析**

与其他压缩方法（头部MLA、跨层共享、KV量化、最终循环复用）在相同缓存预算下对比，LLA在GSM8K、Code/BBH等多任务上匹配或超过教师模型；在4×压缩下仍保持≈95%性能，在21.3×压缩时提升Ouro‑1.4B在H200设备上的批处理容量从32增至768；MATH‑500长生成在on‑policy微调后准确率提升0.16–0.24。总体而言，LLA在低压缩比下实现几乎无损压缩，在高压缩比下提供显著的存储与批处理优势。

**⚠️ 局限性**

局限性包括：极端压缩（>10×）会导致检索准确性下降；长文本生成需要额外的on‑policy微调；-2D在大循环步长下参数量激增，限制了极限压缩的可行性；实现仍依赖分离的解码与重构路径，未能统一到单一路径；方法需在已训练好的教师模型上做后处理，对原始训练过程没有直接影响。

---

## 101. Complete Trip: A Linked Multimodal Human Mobility Dataset

**arXiv ID:** 2607.15436 | [PDF](https://arxiv.org/pdf/2607.15436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 102. Inpainting Insights: Elevating Visual XAI with Photorealistic Perturbations

**arXiv ID:** 2607.15482 | [PDF](https://arxiv.org/pdf/2607.15482v1)

**作者:** Josef Lindl `[一作]` (Julius-Maximilians-Universität Würzburg), Damien Garreau `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 LILI，一种在 LIME 解释框架中使用 LaMa 图像修复模型生成逼真掩模补全的方式，以提升图像分类解释的质量。

**💡 创新点**

核心创新在于：① 将现代生成式修复模型 LaMa 替代传统均值/黑色遮挡；② 通过超像素掩模扩展（mask expansion）隐藏修复边缘残留，增强遮挡效果；③ 证明该方法在解释准确性与样本分布一致性上优于 LIME 与 LIME‑G。

**🔧 技术方法**

技术包括：LIME 解释流程、快速傅里叶卷积（FFC）LaMa 修复网络、超像素分割（quickshift）、掩模扩展算法、Fréchet Inception Distance、Saliency 指标、Kendall’s W 评估稳定性。

**📊 数据集**

主要使用 ImageNet‑1k（IISVRC 2012）图像，采用 InceptionV3 进行预测，并在 100 张图像上进行解释生成与评测。

**📈 对比分析**

对比方法：LIME、LIME‑G、LILI；评测指标包括 FID（LILI 6.727 < LIME‑G 56.524 < LIME 30.532）、Saliency 评分（LILI 3‑像素扩展最优）、Kendall’s W 稳定性（LIME 0.889 > LILI 0.852 > LIME‑G 0.723）、平均运行时间（LIME 4.43 s < LIME‑G 8.7 s < LILI 12.1 s）。

**⚠️ 局限性**

局限性：① 计算成本高于传统 LIME，虽然低于扩散模型；② 解释稳定性略低于 LIME，原因在于掩模扩展导致超像素重叠与互相影响；③ 只在 ImageNet‑1k 上验证，缺乏对其他数据集与模型的泛化评估。

---

## 103. Multi-Objective Kinodynamic Motion Planning with Asymptotic Pareto Optimality

**arXiv ID:** 2607.15508 | [PDF](https://arxiv.org/pdf/2607.15508v1)

**作者:** Yusif Razzaq `[一作]` (University of Colorado Boulder), Morteza Lahijanian `[通讯]` (University of Colorado Boulder)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一套统一的基于Stable Sparse-RRT的多目标动力学运动规划框架，并实现了三类算法：词典优化、约束优化和Pareto前沿逼近。

**💡 创新点**

核心创新在于将每个Witness邻域中的单一代表节点改为代表集合，能够在同一搜索树中维护局部Pareto最优子轨迹；同时证明了连续域下词典优化无法用标量化简，并给出了ϵ等价与约束完整性证明。

**🔧 技术方法**

技术包括基于SST的采样扩展、Witness邻域稀疏化、成本空间稀疏度参数ϵ⃗、局部Pareto筛选（PruneDominated/PruneLexSet/PruneConSet）、MonteCarloProp控制采样等。

**📊 数据集**

实验使用两种动力学模型（二维双积分器和四维自行车模型），四个工作空间（简单、两/三/多同伦类、混乱环境），以及三种成本函数（路径长度、障碍清除、障碍积分），共在100条随机实例上进行评测。

**📈 对比分析**

与传统加权求和SST做对比，实验显示：词典优化能在不调节权重的情况下逼近真实词典极小值；约束优化相较于SST具备完整性并能有效缩小搜索树；Pareto算法一次运行即可覆盖整个Pareto前沿，在非凸前沿上明显优于多次加权求和重规划，且在计算时间上相差约10‑50倍。

**⚠️ 局限性**

局限性包括：ϵ等价方法仅在两目标下可保证误差边界；维数增大时代表集合规模与运行时复杂度呈指数增长；算法依赖于手动设置稀疏度参数ϵ⃗、δ_s，调参仍需经验；在极高维成本空间或实时场景下的可扩展性待进一步验证。

---

## 104. Beyond a Joke: Multi-Angle Reasoning for Detecting and Explaining Harmful Humor in Memes

**arXiv ID:** 2607.15442 | [PDF](https://arxiv.org/pdf/2607.15442v1)

**作者:** Shanhong Liu `[一作]` (Singapore University of Technology and Design), Konstantinos N. Plataniotis `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MAR-12 框架，用 12 个基于幽默与仇恨理论的视角对 meme 进行多角度解释，并结合角色感知的 soft‑gate 注意力与原型分类器实现幽默与仇恨的联合检测与可解释判定。

**💡 创新点**

创新点：①将 meme 理解拆分为 12 个可解释的视角，覆盖视觉、文本、文化、情感、语言游戏等维度；②引入角色感知 soft‑gate 注意力，让模型自适应地调节各视角的权重；③通过 VLM 生成的多角度推理和注意权重合成最终解释，提升透明度和可信度。

**🔧 技术方法**

技术：多模态视觉语言模型（Qwen‑VL/CLIP）+ 视角特定 Prompt + 角色感知 soft‑gate 注意力 + 原型分类器 + LLM 解释生成（基于 VLM 的单轮推理合成）。

**📊 数据集**

数据集：PrideMM（LGBTQ+ memes）和 Memotion（含幽默、讽刺、攻击性标签），二者均提供同一 meme 的幽默与仇恨双重标注。

**📈 对比分析**

与 unimodal、CLIP‑style、多角度推理（LoReHM、MiND）等基线对比，MAR-12 在 humor 任务上准确率最高（PrideMM 80.08%/Memotion 79.85%），在 hate 任务上同样领先（PrideMM 75.53%/Memotion 73.42%）。AUC 与 F1 也均超过现有最佳模型，显示在幽默与仇恨共存场景下的强大性能。

**⚠️ 局限性**

局限性：①视角集合预定义，难以覆盖所有文化、方言或新兴 meme 形式；②仅采用单轮解释生成，可能忽略细微文化语境；③评估仍依赖 GPT‑4 与人工，可能存在偏差；④模型虽轻量化，但对 VLM 的依赖使推理成本较高。

---

## 105. Conjectural Decidability of the Skolem Problem

**arXiv ID:** 2607.15510 | [PDF](https://arxiv.org/pdf/2607.15510v1)

**作者:** Florian Luca `[一作]` (Stellenbosch University), James Worrell `[通讯]` (Oxford University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了“large zeros”概念并研究其稀疏性，给出在强化Cramér猜想下无大零的条件性证明，并证明大零集合无密度，从而构造了密度为1的全局Skolem集合。

**💡 创新点**

创新点在于将大零与“好素数”结合，利用强化的Cramér-Granville猜想推断大零不存在；同时提供无条件证明大零稀疏的理论，并得到全局密度为1的Skolem集合。

**🔧 技术方法**

使用代数数论工具（分裂域、范数、Frobenius自同构）、Amoroso–Viada零点计数、Cramér模型以及素数分布密度分析，并通过递归分解非退化线性递推序列实现论证。

**📊 数据集**

无实验数据集，本研究完全基于理论推导。

**📈 对比分析**

与已有的条件可判定结果相比，本文提出更宽泛的条件并给出理论上可计算的界限，但由于指数/双指数界限过大，实际算法不可行。

**⚠️ 局限性**

主要局限在于结论依赖于强化的Cramér猜想；即使无条件结果中得到的界限极大，实际判定效率低，且对高度有限的LRS仍需枚举。

---

## 106. Closed-Loop Bayesian Bandit Encoder with GRAND Receiver for a Bursty Interference Channel

**arXiv ID:** 2607.15404 | [PDF](https://arxiv.org/pdf/2607.15404v1)

**作者:** Bhaskar Krishnamachari `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 24249 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种闭环系统，在接收端通过学习突发干扰并更新GRAND噪声模型，调度发送端在交织与非交织编码模式间进行自适应选择。

**💡 创新点**

创新点在于将干扰学习、解码器自适应和码选择三者耦合为一个两臂Bandit问题，并利用模型预测的置信权重加速选择收敛。

**🔧 技术方法**

采用GRAND/ORBGRAND/SGRAND等噪声猜测解码器、隐藏马尔可夫模型估计干扰、贝叶斯阈值估计、折扣Thompson采样与模型预测伪观测等技术。

**📊 数据集**

使用仿真数据：BPSK信号在AWGN+多源开/关干扰（3个干扰源，幅度0.63/1.15/1.44，ON时长4/6/12符号，平均OFF长数百符号）的场景。

**📈 对比分析**

通过对比Oracle参数、不同噪声模型、不同门控策略、ACK‑only与模型预测采样等，实验显示在干扰估计完成后非交织模式误码率下降10‑30倍，模型预测可将切换延迟压缩约30%，整体收益提升约2–3%。

**⚠️ 局限性**

局限在于仅评估单一高码率随机线性码与固定交织深度，假设理想反馈链路，未考虑HARQ、调度延迟、信道漂移以及多码/多率等更复杂场景。

---

## 107. Explicit Over Implicit: Enhancing CNNs Via Complex Structure Tensor Representations for Periocular Recognition

**arXiv ID:** 2607.15410 | [PDF](https://arxiv.org/pdf/2607.15410v1)

**作者:** Kevin Hernandez-Diaz `[一作]` (Halmstad University), Fernando Alonso-Fernandez `[通讯]` (Halmstad University)

**通讯引用:** 4221 | [OpenAlex ID](https://openalex.org/A5086684870)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将复杂结构张量（CST）特征作为预处理的局部方向信息输入CNN，用以提高周眼识别的识别性能。

**💡 创新点**

创新点在于：①使用第二阶复数矩来提取线性对称纹理的方向和置信度，生成可直接输入CNN的复数特征；②将这些明确的方向先验嵌入网络，减少CNN对低层方向特征的隐式学习；③展示了在不同网络宽度/深度压缩下仍能保持甚至提升性能。

**🔧 技术方法**

技术主要包括：复杂结构张量的计算（复数梯度卷积、平方后提取I20、I11等）；将I20的实部、虚部与I11作为多通道输入；在ResNet、DenseNet、VGG、Xception、InceptionV3、MobileNetV2等六种常用CNN上训练验证；使用TensorFlow‑Keras进行实现。

**📊 数据集**

数据集为公开的周眼数据库：Cross‑Eyed（可见光与近红外）和PolyU（可见光与近红外），共计约1.5万张图像。

**📈 对比分析**

与传统仅使用灰度图像输入的CNN以及多篇前沿研究（如使用LBP、HOG、Gabor等手工特征或预训练模型）进行对比；在识别（5折交叉验证）和验证（Close‑World/Open‑World）任务中，CST输入通常能提升1–4个百分点，压缩模型时仍能保持相近或更好性能。

**⚠️ 局限性**

局限性：①在Cross‑Eyed数据集训练样本不足时，CST增益不稳定；②评估时多使用固定测试拆分，未充分验证跨分割泛化；③CST特征的参数（σ1、σ2、γ）仍手工设定，缺乏端到端学习；④在极端遮挡或噪声条件下的鲁棒性未充分研究。

---

## 108. An Empirical Study of Handcrafted Feature Learning and Convolutional Neural Networks for Facial Expression Recognition

**arXiv ID:** 2607.15288 | [PDF](https://arxiv.org/pdf/2607.15288v1)

**作者:** Chethiya Galkaduwa `[一作]` `[通讯]` (Indiana University), Chethiya Galkaduwa (Indiana University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比传统手工特征（HOG+SVM、LBP+LogReg）与轻量级CNN在FER-2013、CK+、KDEF三大数据集上的情感识别性能。

**💡 创新点**

系统评估了不同复杂度数据集对手工特征与深度模型表现的影响，并揭示了手工特征在受控与不受控环境下的性能差异。

**🔧 技术方法**

使用了HOG+SVM、LBP+Logistic Regression、轻量级CNN（3个卷积块、BatchNorm、MaxPooling、Dropout）以及Adam优化器和交叉熵损失。

**📊 数据集**

FER-2013（无约束、35k张）、CK+（实验室受控、≈980张）和KDEF（受控、≈980张）三种数据集。

**📈 对比分析**

通过准确率和混淆矩阵进行比较；CNN在所有数据集均表现最佳（FER 51.6%，CK+ 96.9%，KDEF 72.4%），HOG+SVM在受控数据集高（CK+ 98.4%），LBP+LogReg整体低效。

**⚠️ 局限性**

局限性包括：数据集规模有限、未使用预训练网络、仅使用灰度图像、未考虑视频时序信息、超参数调优不充分。

---

## 109. On the Effectiveness of Fact Checking Information from Politically Congruent and Incongruent Large Language Models

**arXiv ID:** 2607.15364 | [PDF](https://arxiv.org/pdf/2607.15364v1)

**作者:** Jiangen He `[一作]` (University of Tennessee Knoxville), Dorit Nevo `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 1816 | [OpenAlex ID](https://openalex.org/A5020011308)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两份 within‑subjects 实验中，研究者让美国成年人与两种政治倾向不同的 LLM 聊天机器人（左倾/右倾）对话，并测量对政治新闻标题的信任度变化。

**💡 创新点**

创新点在于：① 将政治立场嵌入 LLM 的内容、来源与人格设置；② 在实时交互环境下检验政治一致性对纠正效果的影响；③ 探索 LLM 误判与不确定回复对信任的潜在负面效应。

**🔧 技术方法**

技术手段包括 GPT‑5.1 LLM 与 Exa API 实时网页搜索、基于 AllSides 媒体偏见图表的新闻来源过滤、系统提示的政治人格化，以及混合效应回归分析。

**📊 数据集**

数据集包括 157 条经过预筛选、真/假各约 33 条的政治新闻标题（来源自 Snopes、PolitiFact、Reuters 等）以及 705 名美国成年受试者的实验数据。

**📈 对比分析**

通过混合效应模型对 Trust Δ 进行回归，比较 Bot verdict、政治距离、标题距离及其交互项。结果显示：LLM fact‑checkers 在真标题上平均提升信任 0.93–1.09 点，在假标题上降低信任 0.78–0.85 点，均显著；误判或“不确定”回答同样能显著改变信任，效应大小从小到中等。

**⚠️ 局限性**

局限性包括：① LLM 的准确率有限，误判会导致信任误导；② 实验使用了专门配置的 LLM，通用 LLM 的表现可能不一致；③ 样本与实验设置不完全模拟真实社交媒体情境，缺乏对用户提问和生成文本差异的细致控制；④ 研究仅关注新闻标题的可信度，未探讨更广泛的传播后果。

---

## 110. Evolutionary Algorithm-Guided LLMs for Physics-Informed Neural Network Design

**arXiv ID:** 2607.15560 | [PDF](https://arxiv.org/pdf/2607.15560v1)

**作者:** Xu Yang `[一作]` (East China Normal University), Keqian Li `[通讯]` (East China Normal University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个演化算法架构，引导LLM通过多代搜索设计完整的PINN配置，并在单一波动方程上验证其有效性。

**💡 创新点**

首次将LLM与演化反馈闭环结合，利用有效指纹、父子记忆、精细突变/交叉以及多目标物理审计实现自动化PINN设计。

**🔧 技术方法**

结合LLM生成的AlgorithmSpec、结构化指纹去重、亲代选择、分代记忆、对抗式评估与可审计的PyTorch执行。

**📊 数据集**

在一维常系数波动方程（Wave‑C）上使用解析解进行评估，构建了64边界点、64初始点及32×32评估网格。

**📈 对比分析**

通过两条独立种子、10代、每代10个突变/交叉，比较最终MSE与初始生成的MSE，第二条种子MSE下降95.38%，显示显著提升。

**⚠️ 局限性**

实验仅限单一一维波动方程、两条种子、缺乏与其他搜索方法对比，且未充分验证多目标物理一致性。

---

## 111. Intentional Electromagnetic Interference Attacks on Facial Recognition

**arXiv ID:** 2607.15512 | [PDF](https://arxiv.org/pdf/2607.15512v1)

**作者:** Tyler Fitzsimmons `[一作]` (University of Notre Dame), Adam Czajka `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并演示了利用低成本射频设备对智能手机前置摄像头进行意图性电磁干扰（IEMI）攻击，使多种机器学习人脸识别模型在黑盒环境下失效。

**💡 创新点**

创新点在于首次提出非定向、可复现的IEMI攻击方法，并提供对应的攻击模型、可下载的清洁/受攻击视频基准数据集，证明该攻击对多种网络骨干和损失函数具有普适破坏效果。

**🔧 技术方法**

采用函数发生器、RF放大器与单圈铜线产生FM频率干扰；通过手机摄像头重新采集受扰视频；使用六个开源人脸识别模型和一个商用模型进行评估；并实现IEMI攻击的图像生成模型。

**📊 数据集**

使用公开的MBGC v2面部数据集中的50个身份图像，先在MacBook显示后由手机摄像头重新采集，形成清洁和受攻击的视频样本。

**📈 对比分析**

在0.1%、1%和5%三种FMR阈值下计算FNMR、PAS和ESR，结果显示大多数模型的FNMR从几%升至数十甚至100%，PAS可达54%–96%，仅使用ArcFace损失的模型表现出最强的抗攻击性。

**⚠️ 局限性**

局限性包括攻击依赖特定硬件（前置摄像头、单圈线圈、频率设置），未充分验证跨设备的泛化；对人脸识别系统的防护仅局限于采用ArcFace损失；实验未涉及人类受试者，安全距离和法规合规性需进一步评估。

---

## 112. Physiological Prior-Driven Label Enhancement for Cross-Subject EEG Emotion Recognition

**arXiv ID:** 2607.15566 | [PDF](https://arxiv.org/pdf/2607.15566v1)

**作者:** Hongyu Zhu `[一作]` (Chongqing Institute of Green Intelligent Technology Chinese Academy of Sciences), Mingsheng Shang `[通讯]` (Chongqing Institute of Green Intelligent Technology Chinese Academy of Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了无调参的跨受试EEG情绪识别标签增强框架PhyDA，结合1/f^α谱斜率噪声量化与数据驱动的标签软化；

**💡 创新点**

创新点在于使用生理先验1/f^α谱斜率自适应量化噪声并直接作为标签污染率，无需人工阈值；将该先验与Isolation Forest异常检测和Gaussian Naive Bayes伪标签相结合，形成无需额外网络训练的标签软化流程；

**🔧 技术方法**

采用1/f^α谱斜率估计、Otsu阈值分割、自适应标签软化、Isolation Forest异常检测、Gaussian Naive Bayes伪标签生成与软化融合；

**📊 数据集**

使用公开情绪EEG数据集DEAP、SEED和SEED-IV；

**📈 对比分析**

在严格留一受试交叉验证下，与多种标签去噪基线（未去噪、样本筛选、鲁棒损失、标签修正、EEG专用去噪）和七种主干网络比较，平均准确率提升分别为2.76%、2.66%和3.32%，显著优于基线；

**⚠️ 局限性**

主要限制是对1/f^α谱斜率的依赖，受采集硬件和参考电极影响；验证仅限情绪识别任务，尚未扩展到其他EEG范式。

---

## 113. MGDT: MLLM-Guided Diffusion Transformer with Relation-Adaptive Mixture-of-Experts for Multimodal Knowledge Graph Completion

**arXiv ID:** 2607.15592 | [PDF](https://arxiv.org/pdf/2607.15592v1)

**作者:** Xu Hou `[一作]` (Beijing University of Posts and Telecommunications), Kangkang Lu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于多模态大语言模型（MLLM）引导的Diffusion Transformer框架M^2GDT，用于多模态知识图谱补全（MKGC）

**💡 创新点**

创新点在于先进行关系自适应混合专家路由（RASR‑MoE）选择模态信息，再用冻结的MLLM作为语义锚点对齐到统一空间，最后在对齐空间中采用图结构条件的Diffusion Transformer（KGDT）完成实体生成

**🔧 技术方法**

关键技术包括关系自适应Mixture‑of‑Experts路由、MLLM语义锚点对齐、知识图谱Diffusion Transformer以及条件式扩散生成

**📊 数据集**

实验使用三个公开基准：MKG‑W、MKG‑Y和DB15K，涵盖结构、文本、视觉三种模态

**📈 对比分析**

与19种传统与多模态补全基线对比，M^2GDT在MRR、Hits@1、Hits@3等指标上均显著优于SOTA，提升幅度达约1–3个百分点

**⚠️ 局限性**

局限性包括对大型MLLM的依赖导致计算成本较高，对模态间融合方式仍有进一步提升空间，且在极稀疏关系或非结构化模态下表现尚待验证

---

## 114. PACE: Persona Adaptation through Conversational Elicitation in Human-Robot Interaction

**arXiv ID:** 2607.15579 | [PDF](https://arxiv.org/pdf/2607.15579v1)

**作者:** Peizhen Li `[一作]` (Macquarie University), Simon See `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PACE 框架，利用交互式问答动态生成并在 Ameca 机器人上部署结构化人格，提升人机交互的信任与连贯性

**💡 创新点**

创新点在于将人格建模从静态系统提示转为多层次的对话式提问，自动抽取心理维度并实时编译成可嵌入物理机器人行为的结构化规范

**🔧 技术方法**

结合 LLM（GPT‑5.5‑mini）、Google Speech‑to‑Text、Amazon Polly、实时情绪分类器以及 Ameca 的面部动画库，形成端到端的感知-生成-执行闭环

**📊 数据集**

使用 25 名受试者的问卷与行为实验数据（GSS、BFI‑44、Dictator/Public‑Goods/Prisoner games、社会推理场景），以及 Ameca 的面部动画日志

**📈 对比分析**

与传统静态提示基线对比，PACE 在人格一致性、心理属性匹配、经济决策 MAE、社交情境正确率等多项指标显著提升（p<0.01），并在 HRI 评分中获得信任、拟人化、人格一致性和相关性显著提升（p<0.01）

**⚠️ 局限性**

依赖云端语音识别和 LLM 推理导致噪声和延迟；对话时间有限，可能忽略长期偏好变化；面部表情受硬件预设动画限制，缺乏细粒度控制

---

## 115. Trajectory-aware Cross-view Geo-localization with Sequential Observations

**arXiv ID:** 2607.15491 | [PDF](https://arxiv.org/pdf/2607.15491v1)

**作者:** Tianyi Gao `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种统一框架 TrajLoc，支持基于视频和自然语言路线描述的跨视角地理定位，并引入 SeqGeo-VL 数据集；

**💡 创新点**

创新点在于将路线几何信息显式注入查询嵌入（TrajMod），实现视频与文本两种模态的协同训练与跨模态知识迁移；

**🔧 技术方法**

采用 CLIP ViT-L/14 编码器、双阶段自适应训练、信息噪声对比损失、FiLM 风格的轨迹调制模块，以及 Fourier 特征表示轨迹方向；

**📊 数据集**

使用 38,863 条视频-文本-卫星图像三元组的 SeqGeo-VL（基于 SeqGeo 的扩展），以及公开的 SeqGeo、GAMa 等基准；

**📈 对比分析**

与最新方法（FlexGeo、GARet、Qwen3-VL-Embedding、CrossText2Loc 等）比较，TrajLoc 在视频检索 R@1=12.09%（+2.4%）和文本检索 R@1=2.52%（+1.7×）均实现显著提升；

**⚠️ 局限性**

局限性包括：仍受限于轨迹几何的粗糙估计、对极端遮挡和高频运动的鲁棒性不足、以及在非 GPS 可用环境下对手工标注的依赖。

---

## 116. StructGen: Disambiguating Multi-Reference Image Generation via Structured Context Modeling

**arXiv ID:** 2607.15619 | [PDF](https://arxiv.org/pdf/2607.15619v1)

**作者:** Jianing Peng `[一作]` (Beijing Jiaotong University), Yunchao Wei `[通讯]` (Beijing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 StructGen 框架，采用结构化上下文建模多参考图像生成，并构建了高质量的真实图像数据集和专门的基准。

**💡 创新点**

创新点在于给参考图像分配唯一标识符并使用基于标识符的指令显式化跨参考关联，同时利用基于真实图像的结构化数据采集提升语义多样性与细节一致性。

**🔧 技术方法**

技术方案基于 BAGEL 模型，使用 VLM 生成实体描述、生成式模型提取参考图像、LLM 合成标识符指令，并通过流匹配损失和混合采样策略进行结构化微调。

**📊 数据集**

使用自构造的约 15,965 份高质量结构化数据集以及新设计的 StructGen Bench（包含多人体、多场景等子集），并在 OmniContext 公共基准上进行对比。

**📈 对比分析**

通过 PF、SC、ID 三项指标与 OmniGen2、BAGEL、BAGEL-MICo、Echo-4o 等基线在 OmniContext 与 StructGen Bench 上进行比较；StructGen 在所有指标上均领先，尤其在多人体+场景子集 PF 提升约 +0.8、ID 提升约 0.36，整体排名第一。

**⚠️ 局限性**

局限性在于仅针对人类中心的多参考生成，数据规模相对较小（约 16k 样本），缺乏更广泛的参考类别与更大规模训练，尚未验证在更一般多模态场景中的通用性。

---

## 117. From Feasibility to Desirability: Plan, Learn, Adapt (PLA) Framework for Personalized On-Device Itinerary Generation

**arXiv ID:** 2607.15552 | [PDF](https://arxiv.org/pdf/2607.15552v1)

**作者:** Himel Dev `[一作]` (529 Tech LLC), Bashima Islam `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Plan-Learn-Adapt（PLA）三阶段框架，用于在移动端生成满足硬约束且符合用户偏好的多日行程。

**💡 创新点**

创新点包括①通过异构规划器集合产生多样化可行方案；②使用 Bradley–Terry 模型从成对行程偏好学习全程级别奖励；③在设备可控预算内做可行性保持的局部改进。

**🔧 技术方法**

技术上结合约束规划+启发式搜索（Greedy、DP、Beam、A*、SA）、差分式特征 Bradley–Terry 线性/GBDT 奖励模型、增量可行性重调度以及基于设备信号的时间预算。

**📊 数据集**

使用了约 2,519 对行程的人工偏好标注，覆盖 100+ 美国城市的 POI 数据（开放时间、地点、类别、受欢迎度等）。

**📈 对比分析**

与单一规划器对比，PLA 奖励集合胜率 67.8%（比最佳单一 DP 56.6% +11.2%），在 FlyEnJoy 线上部署平均 109.9 ms 延迟，行程完成率提升 91%；相比 GPT‑5/Claude‑Opus/Gemini 等 LLM，PLA 保证 100% 可行性，而 LLM 0%。

**⚠️ 局限性**

局限性在于依赖预先收集的成对偏好数据，模型泛化受城市 POI 结构影响；仅支持离线、可解释性有限的特征；对极端约束或非常长行程的局部改进可能不足。

---

## 118. SPEED: One-Step Pixel Diffusion for High-quality Video Frame Interpolation

**arXiv ID:** 2607.15585 | [PDF](https://arxiv.org/pdf/2607.15585v1)

**作者:** Zihao Zhang `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

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

## 119. LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models

**arXiv ID:** 2607.15447 | [PDF](https://arxiv.org/pdf/2607.15447v1)

**作者:** Jingteng Li `[一作]` (University College London), Payam Barnaghi `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出LLM4EHR模型，通过将EHR事件和时间序列在时间轴上对齐并进行对比学习，实现临床基础模型的预训练。

**💡 创新点**

创新点在于将领域适配的大型语言模型嵌入的事件表述与时间序列编码器进行语义正则化的对比学习，充分利用两种EHR模态的共享时序结构。

**🔧 技术方法**

使用的技术包括Transformer时间序列编码器、冻结的BioClinical ModernBERT作为事件编码器、InfoNCE及其ω正则化对比损失、重建损失以及迁移学习/少量样本微调。

**📊 数据集**

使用的数据集为MIMIC‑IV ICU EHR和Physionet Challenge 2012公共数据库。

**📈 对比分析**

与传统监督、Self‑Sup及多模态对齐基线相比，LLM4EHR在死亡预测、疾病分型、去压/剩余LOS等四项任务上均实现最高AUROC/AUPRC，且在Physionet上表现出良好的少样本迁移性能。

**⚠️ 局限性**

局限性包括对更大规模语言模型的可扩展性受限、仅针对ICU停留期预训练、未覆盖跨住院期间的长周期病程以及对计算资源的高需求。

---

## 120. Efficient and Effective In-place Graph-based Vector Index Updates

**arXiv ID:** 2607.15576 | [PDF](https://arxiv.org/pdf/2607.15576v1)

**作者:** Haotian Liu `[一作]` (Southern University of Science and Technology), Bo Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种高效的原地（in-place）图索引向量更新系统，利用向量级更新机制实现了高吞吐量和高召回的实时向量增删。

**💡 创新点**

创新点包括：①将连接建立与剪枝统一为向量级任务，消除冗余节点访问；②设计基于协程的轻量级执行引擎、异步缓冲管理器和分离导航与原始向量的文件系统；③通过任务拆分与异步I/O实现高并发、低内存占用的原地更新。

**🔧 技术方法**

技术手段包括：C++20 协程实现任务级切换；异步缓冲管理器采用计数器、CAS 与复制写(COW)实现无阻塞冲突解决；分层文件系统借鉴 B+树与 Linux inode；向量级任务分解统一处理插入与删除；并使用高效的页面访问协议和线程调度。

**📊 数据集**

实验数据集：DEEP（1B 96D float）与 SIFT（1B 128D byte）公开数据集，分别使用 DEEP100M、DEEP800M、SIFT800M 等子集进行评测。

**📈 对比分析**

与 DiskANN、OdinANN、SPFresh 等最先进系统比较，更新吞吐率提升 2.8–4.7 倍，搜索吞吐率提升 1.7–3.9 倍，召回保持或略优，内存占用仅 73% 以内，且保持较低的内存波动。

**⚠️ 局限性**

局限性：对极大规模批量更新的动态调度尚未深入；删除密集场景下召回略低于全扫描；未覆盖多租户环境、SPDK 加速等后续优化方向。

---

## 121. Perturbation Power Selection for First-Error Delay Maximization in Enhanced SC Decoding

**arXiv ID:** 2607.15553 | [PDF](https://arxiv.org/pdf/2607.15553v1)

**作者:** Zhicheng Liu `[一作]` (Sichuan University), Zechun Hu `[通讯]` (Sichuan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

分析并优化增强SC解码中扰动功率对首次错误延迟概率的影响，并提出基于该概率的扰动功率选择算法。

**💡 创新点**

揭示扰动功率对首次错误延迟概率的非单调关系，并给出近似表达式与最优功率求解方法，首次将该概率用于指导扰动功率选择。

**🔧 技术方法**

高斯近似、SC解码、扰动增强SC（PE‑SC）解码、误差位概率分析、基于平均LLR和g‑函数计数的优化算法。

**📊 数据集**

在二进制AWGN信道上使用长度为1024、4096、16384的极化码，配合8位CRC进行仿真。

**📈 对比分析**

与传统SC和原始PE‑SC进行BLER比较；实验显示最优扰动功率可在所有SNR和码长下实现约0.1 dB的BLER提升，且与理论最佳延迟概率一致。

**⚠️ 局限性**

对单次扰动的分析，缺乏对多扰动或SCL情况的理论与实验验证；最大化延迟概率与BLER最小化的关联仅在实验中得到确认，未给出完整理论证明。

---

## 122. Recursive Harness Self-Improvement

**arXiv ID:** 2607.15524 | [PDF](https://arxiv.org/pdf/2607.15524v1)

**作者:** Hyunin Lee `[一作]` (Sakana AI), Yujin Tang `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种递归式工具链自我改进（Recursive Harness Self-Improvement, RHI），通过在黑盒编码代理中对可编辑的提示级工具链进行少量迭代优化，显著提升执行轨迹质量和代理性能。

**💡 创新点**

创新点包括：
1) 将工具链（角色、指令、合同、跳跃）视为可优化的提示文本；
2) 采用轨迹局部（仅与上一次的工具链进行比较）的对比反馈，避免昂贵的全局候选搜索；
3) 通过累积的自我对比历史指导后续迭代，实现仅需几次更新即可获得显著收益；
4) 在信息理论框架下提出工具链隐式优化目标：增强任务相关信息同时抑制组件冗余。

**🔧 技术方法**

使用技术：
- 大语言模型（Claude Sonnet/Opus）作为编码代理与评判器；
- LLM-as-a-judge 的对比评估（pairwise preference）；
- 提示级工具链定义与迭代更新；
- 低维文本嵌入（OpenAI/CLIP）+ t-SNE/UMAP 可视化与余弦相似度分析；
- 互信息与总相关（TC）评估工具链信息量与冗余变化。

**📊 数据集**

数据集：30个人工合成的机器学习研究任务，覆盖定量金融、机器人、制药三大领域，每个领域10个任务，任务要求生成完整代码仓库并满足标准交付物。

**📈 对比分析**

对比方法：将RHI优化后的工具链与同一模型的
- 基准工具链（默认）;
- 同一模型的更高推理努力（test‑time scaling）；
- 供应商自带的多代理工具链（例如Claude内置的动态工作流）。
性能表现：
- 仅需1–2次RHI迭代即可使低推理努力代理击败所有更高推理努力的基准；
- 成功降低推理成本最高可达60%，缓存读写量显著下降；
- 输出token数量基本保持不变，说明收益主要来自更高效的上下文管理而非更长生成。

**⚠️ 局限性**

局限性：
- 只关注工具链的第一半循环（不直接改进基础模型）；
- 依赖LLM对比评判，评估主观性和评判器的可靠性；
- 在极强模型或极大任务规模下，RHI的收益可能趋于饱和；
- 目前只在合成任务上验证，真实复杂工程场景的通用性尚待进一步验证。

---

## 123. When Can Test-Time Adaptation Help Zero-Shot CT Vision-Language Models?

**arXiv ID:** 2607.15556 | [PDF](https://arxiv.org/pdf/2607.15556v1)

**作者:** Ailar Mahdizadeh `[一作]` (University of British Columbia), Leonid Sigal `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究零样本3D CT视觉‑语言模型在分布偏移下的测试时自适应（TTA），并提出CARVE方法。

**💡 创新点**

① 识别TTA有效的前置条件（体积深度保持、基模型可迁移）；② 发现标准熵最小化与prompt‑pair多标签预测不匹配；③ 提出基于样本特定正标签计数的cardinality‑aware top‑entropy目标。

**🔧 技术方法**

使用prompt‑pair Bernoulli概率、弱视图生成与熵筛选、基于卡尔方计数的top‑k损失、仅更新视觉归一化参数、保留视图的内存高效自适应。

**📊 数据集**

CT‑RATE 内部验证集、外部 RAD‑ChestCT，以及 CC‑CCII（三分类）和 LUNA16（二分类）用于泛化验证。

**📈 对比分析**

与无TTA、TENT、RLCF、ML‑TTA 等方法对比，CARVE 在基模型已具备判别能力时在多标签、三分类、二分类任务上实现最稳健的 AUROC 提升（如 CT‑CLIP Zero‑shot 从 0.713 提升至 0.749，外部 RAD‑ChestCT 从 0.499 提升至 0.533），但在严重外部偏移时基模型差距仍是主导。

**⚠️ 局限性**

TTA 仅在体积深度保持且基模型可迁移时有效；在严重外部偏移时无法弥补基模型缺乏判别信息；卡尔方计数估计依赖概率校准；仅更新归一化参数限制了适应范围。

---

## 124. A Transportable Threshold-Based Framework for Interpretable Classification of Medical Data

**arXiv ID:** 2607.15394 | [PDF](https://arxiv.org/pdf/2607.15394v1)

**作者:** Antony Garcia `[一作]` (Worcester Polytechnic Institute), Xinming Huang `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 4096 | [OpenAlex ID](https://openalex.org/A5042654091)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于统计阈值的可迁移阈值框架，利用Bernoulli Naïve Bayes对医疗数据进行可解释分类。

**💡 创新点**

创新点在于通过监督χ²引导的阈值化将连续变量转化为二值特征，既保持BNB的透明性，又实现手算可复现的推理过程。

**🔧 技术方法**

使用Bernoulli Naïve Bayes、χ²阈值化、Beta校准等技术，并通过10折交叉验证、DeLong检验、McNemar检验等统计方法评估模型。

**📊 数据集**

使用Pima Indians Diabetes、Wisconsin Breast Cancer以及Heart Failure Prediction这三个公开临床数据集。

**📈 对比分析**

与互信息、信息增益、Gini、Otsu等多种阈值化策略对比，χ²阈值化得到的AUC分别为0.800、0.984、0.919，性能与传统基准模型相当或略逊，但具有更高的可解释性。

**⚠️ 局限性**

局限包括：特征独立假设可能不成立、阈值估计受样本波动影响、对大规模数据阈值搜索成本高、未进行临床用户可用性验证。

---

## 125. Two-Path Status Verification for Outbound Enterprise Messaging Pipelines: Webhook and Scheduled Polling Fallback Architecture

**arXiv ID:** 2607.15529 | [PDF](https://arxiv.org/pdf/2607.15529v1)

**作者:** Devam Gupta `[一作]` `[通讯]` (Twilio), Devam Gupta (Twilio)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出双路径状态验证架构，结合实时Webhook+事件通道与定时轮询回补，解决CRM外发消息状态回调失效导致的状态漂移问题。

**💡 创新点**

创新点在于将推送式Webhook与拉取式轮询并行运行，并通过幂等idempotent upsert与状态机实现无锁定性的一致性收敛；自重排调度实现子分钟级别轮询。

**🔧 技术方法**

使用REST webhook、HMAC签名校验、内部事件总线（异步事件通道）、异步事件处理器、平台API调用、定时调度器自重排、幂等upsert、状态机控制。

**📊 数据集**

未公开具体数据集，仅在真实CRM生产环境（多租户平台）中验证。

**📈 对比分析**

与单路径Webhook方案对比，在模拟网络与端点失效场景下，双路径架构的状态最终一致率提升至>99%，平均处理延迟仅增加约30ms。

**⚠️ 局限性**

受限于平台外部调用上限、调度器粒度以及维护成本；需要额外监控与手动配置以防止调度冲突。

---

## 126. Robust Silicone Pour Casting and Sensor Embedding Procedures for Soft Robotic Actuators

**arXiv ID:** 2607.15422 | [PDF](https://arxiv.org/pdf/2607.15422v1)

**作者:** Harshit Thakker `[一作]` (Stevens Institute of Technology), Jacqueline Libby `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 278 | [OpenAlex ID](https://openalex.org/A5043899927)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套可重复、可扩展的两部分浇注成型工艺，用于制造具有气密密封层和薄膜柔性传感器嵌入的软气动驱动器。

**💡 创新点**

提出了基于可控深度浇注的密封层制备流程和传感器嵌入方法，避免内部通道堵塞和气密泄漏，提升了工艺可重复性。

**🔧 技术方法**

使用SORTA-Clear 40硅胶两部分浇注、有限元仿真、PID压力控制、基于图像的柔性传感器标定及自动化图像处理等技术。

**📊 数据集**

实验数据包括20 psi内部压力对应的弯曲角度、传感器电压与角度的映射、阶跃与正弦压力输入下的压力‑角度曲线等。

**📈 对比分析**

通过与FEM预测的弯曲角度比较，获得R²=0.9978的曲线拟合；PID控制器实现0.285 s上升、0.877 s定峰、0.011 psi稳态误差，弯曲角度与仿真一致，正弦输入显示低滞后。

**⚠️ 局限性**

密封层工艺仍易产生浅气泡，需改进底盖浸硅胶以防止硅胶-PLA相互作用；整体流程依赖人工操作，进一步自动化有待提升。

---

## 127. Gradually Verifying Unfolding Expressions & Pure Functions

**arXiv ID:** 2607.15383 | [PDF](https://arxiv.org/pdf/2607.15383v1)

**作者:** Hazel Torek `[一作]` (Clemson University), Jonathan Aldrich `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4985 | [OpenAlex ID](https://openalex.org/A5091372985)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在渐进式验证框架下，对展开表达式和纯函数的语义进行了形式化，并在此基础上扩展了 Gradual Viper 的符号执行语义与对应的运行时语义，完成了对这些特性的完整可靠性证明；

**💡 创新点**

创新点在于首次在 Gradual Viper 中引入并形式化展开表达式与纯函数的符号与运行时语义，弥补了先前正式化中对这些特性缺失的空白，并提供了完整的安全性证明；

**🔧 技术方法**

主要技术包括隐式动态框架、符号执行、渐进式验证、符号与运行时状态对应、以及对可重入/递归的防止机制（访问标记集）和对不确定性公式的处理；

**📊 数据集**

本工作并未使用外部数据集，而是以 VerifiedSCION、Go 标准库等真实项目中的代码示例作为验证案例；

**📈 对比分析**

由于论文侧重于形式化证明与语义定义，未进行性能评测；对方法的比较主要体现在理论证明与现有 Viper 证明框架的一致性和完整性上；

**⚠️ 局限性**

限制在于递归纯函数与递归展开表达式的处理仍存在不完整性，且对量化权限、分数权限等高级特性未覆盖；

---

## 128. ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting

**arXiv ID:** 2607.15542 | [PDF](https://arxiv.org/pdf/2607.15542v1)

**作者:** Damani Mguni-Coker `[一作]` `[通讯]` (Independent Researcher), Damani Mguni-Coker (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 ImprovedVBGS 框架，实现了在变分贝叶斯高斯渲染中可实时增量重建的加速版本。

**💡 创新点**

创新点包括空间截断变分 E 步、截断 ELBO 重分配、前向重分配以及静态张量填充等技术，显著降低了 per‑frame 计算成本。

**🔧 技术方法**

使用了 KD‑Tree 近邻剪枝、坐标上升变分推断、核融合、混合精度搜索、静态形状填充以及 JAX 动态重编译优化等技术。

**📊 数据集**

在 NeRF Synthetic 数据集的八个合成场景（每场景 200 训练帧 + 100 验证帧）上进行实验。

**📈 对比分析**

与原 VBGS 及其后续改进进行对比；在 RTX 3070 Ti 上帧延迟从 84.0 s 降至 0.050 s，PSNR 维持约 21.5 dB，速度提升约 1680 倍。

**⚠️ 局限性**

仍受限于需要大批量处理和显存需求，在极小批量或低内存设备上效果可能受限；实验仅在合成数据上验证，真实场景表现待进一步评估。

---

## 129. Do Generative Models Keep Time? A Time-Aware Evaluation of Synthetic Sequential Tabular Data

**arXiv ID:** 2607.15606 | [PDF](https://arxiv.org/pdf/2607.15606v1)

**作者:** Kiwan Kwon `[一作]` (UNIST), Yongjae Lee `[通讯]` (UNIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于数据时间属性的评估协议Seq2Synth，用于评估合成序列表格数据的时间真实性。

**💡 创新点**

通过将数据划分为时间表示、采样规律、轨迹依赖和模式结构四个轴，动态决定可应用的评估维度，实现对时间戳、横截面、纵向轨迹和结构一致性的系统评估。

**🔧 技术方法**

使用了多种生成模型（ClavaDDPM、RCTGAN、RDBDiff、REaLTabFormer、RGCLD、RelDiff、SDV、TabDiT）并结合时序度量、邻接度量、n-gram隐私等多指标。

**📊 数据集**

在13个跨六个领域的序列表格数据集（Rossmann、Berka、Fannie Mae、Walmart、Airbnb、PTB‑XL、Freddie Mac、Citi Bike、CMAPSS、Coupon、Google Cluster、H&M、Home Credit）上进行评测。

**📈 对比分析**

先根据四个时间轴属性选取合适的评估维度，再分别计算时间戳有效性、跨时间点分布、轨迹动态、结构一致性，并在每个维度给出分数；实验显示传统静态分布排名与时序评估显著偏离，模型在不同维度表现不一，说明仅靠静态指标误导模型选择。

**⚠️ 局限性**

该框架依赖对数据先验属性的手工判定，且评估维度仍以统计相似度为主，难以捕捉更细粒度的时间逻辑错误，未来需进一步完善自动化特征抽取和更细粒度的时间一致性度量。

---

## 130. A cubical formalisation of topos causal models: intervention, sheaf gluing, and the intuitionistic do-calculus

**arXiv ID:** 2607.15629 | [PDF](https://arxiv.org/pdf/2607.15629v1)

**作者:** Karen Sargsyan `[一作]` `[通讯]` (Academia Sinica), Karen Sargsyan (Academia Sinica)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 Cubical Agda 中对 1‑topos 形式化的因果模型核心进行了机器验证，证明了干预分类器、机制拼接的极限性质、内部逻辑的 Kripke‑Joyal 语义以及 Lawvere‑Tierney 做法算子下的 do‑calculus 稳定性。

**💡 创新点**

主要创新包括：① 对干预分类器的分类定理进行正式证明；② 证明并修正了 Lawvere‑Tierney 的公理，补全了缺失的增容律；③ 给出了双否定拓扑实例并展示了其在做法算子中的作用；④ 引入并机器检验了上下文无关性（contextuality）作为拼接阻碍；⑤ 将 j‑稳定性与 counterfactual 的可传递性联系起来。

**🔧 技术方法**

技术手段为：Cubical Agda（依赖类型与高阶公理化概率单子）、范畴论与拓扑论的公理化（子对象分类器、Grothendieck 拓扑、Lawvere‑Tierney 结构）、Kripke‑Joyal 语义的实现、以及对 Čech 同调的机器证明。

**📊 数据集**

本工作不使用外部数据集，所有证明均在抽象范畴（上下文、概率域取 ℚ）与符号模型上完成；若需实验，则需要配合已验证的概率单子实现。

**📈 对比分析**

由于侧重于形式化与证明，未进行数值性能对比；验证效果主要体现在完整性与无后设假设的机器检验上，保证了理论正确性。

**⚠️ 局限性**

局限性包括：① 仅在预设的预堆叠（presheaf）topos 上工作，未实现类型层面的 sheafification；② 采用无定向的因果箭头，未给出真正的 directed do‑operator；③ 仅证明了 j‑稳定性对应可传递性，未给出完整的 Bareinboim‑Pearl 传递性图形判据；④ 对无限/∞‑范畴的推广留待后续工作。

---

## 131. Region-Grounded Vision-Language Learning for Detection-Guided Mammographic Lesion Classification

**arXiv ID:** 2607.15615 | [PDF](https://arxiv.org/pdf/2607.15615v1)

**作者:** Zhengbo Zhou `[一作]` (University of Pittsburgh), Shandong Wu `[通讯]` (University of Pittsburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了检测指导的区域对文本对比学习框架，用于乳腺X线影像的病灶分类与检测。

**💡 创新点**

① 在局部病灶层面进行对比学习，克服全图对齐导致的小病灶信息丢失；② 引入正对齐、语义硬负样本与背景抑制的多组件目标，避免语义坍塌与背景偏差；③ 结合轻量化FCOS检测头和置信度保险机制，实现空间敏感的分类。

**🔧 技术方法**

使用CLIP式视觉-文本对比学习、ROIAlign提取病灶特征、语义硬负样本构造、背景负样本抑制、FCOS风格检测头、prompt‑based相似度评分以及confidence‑gated保险机制。

**📊 数据集**

CBIS‑DDSM 与 VinDr‑Mammo 两个公开乳腺病灶数据集（包含肿块与钙化两类病灶）。

**📈 对比分析**

与 DenseNet、ViT、Mammo‑CLIP、FrozenVLM、LLaVA‑Med 等基线在三种评估设置（in‑domain、跨数据集、迁移学习）下对比，实验显示在分类准确率、F1、AUC 以及检测 mAP 上均优于基线；跨数据集与迁移学习仍保持领先。

**⚠️ 局限性**

依赖标注的病灶框框；在极少量数据或高分辨率场景下提升有限；置信度保险机制只能降低误检风险，无法完全消除错误检测；仅在乳腺X线影像上验证，尚未测试其他模态的泛化能力。

---

## 132. Fair Allocation of Divisible Goods under Non-Linear Valuations

**arXiv ID:** 2607.15613 | [PDF](https://arxiv.org/pdf/2607.15613v1)

**作者:** Haris Aziz `[一作]` (UNSW Sydney), Kaiyang Zhou `[通讯]` (UNSW Sydney)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在各个可分割且同质的资源（如计算资源、研究经费、场地等）中，如何在具备非线性（但单一资源上随份额变化）的价值函数的前提下公平分配资源，并给出了针对最大最小份额（MMS）与无嫉妒（EF）两种公平性目标的算法与复杂度结果。

**💡 创新点**

创新点包括：① 在非线性可分割物品分配模型中首次给出了接近最优的MMS近似结果，证明了 1/(2n-1) 的近似是渐近最优的，并在 n≤3 的特殊情形下给出 1/n 的精确近似；② 证明了在至少三件物品且价值为单阈值分段常数时，存在 EF+PO 分配的判定是 NP‑hard 的；③ 针对单件物品给出了多项式时间的 EF‑约束 PO 分配算法，并给出了判定 EF+PO 存在性的判定方法。

**🔧 技术方法**

使用的技术主要是离散化与分块构造、基于阈值的可分割资源分配算法、近似算法与分析、以及归约证明（从 Partition 到 EF+PO 存在性问题）。在单件物品部分，还利用了分段线性函数的分段点性质进行多阶段分配与最优性检验。

**📊 数据集**

该工作为理论研究，未使用实际数据集；所有结果均在理论模型与构造实例上证明。

**📈 对比分析**

与现有的可分割物品公平分配结果对比，本文的 1/(2n-1) 近似与 1/n 近似在理论上与已知的 1/2‑MMS 或 2/3‑MMS 结果相比更为精细；对于 2、3 人的案例给出了最优的 1/2‑MMS 与 1/3‑MMS 分配。单件物品的算法实现后验检查在多项式时间内完成，能够直接判定 EF+PO 是否存在。

**⚠️ 局限性**

局限性包括：① 只考虑了可分割且同质的物品；② 价值函数仅满足对每件物品的单一比例非线性，且整体是各物品的加性；③ 对于更多玩家或更一般的非线性/互补/替代性价值未给出结果；④ 对于两件物品的 EF+PO 判定复杂度仍未知；⑤ 只提供了理论近似与多项式时间算法，缺乏对实际数据或实验验证的支持。

---

## 133. Benchmarking MRI Representations for Deep Learning-Based Focal Cortical Dysplasia Segmentation

**arXiv ID:** 2607.15605 | [PDF](https://arxiv.org/pdf/2607.15605v1)

**作者:** Soumen Ghosh `[一作]` (University of Queensland), Rajat Vashistha `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究在统一的nnU-Net框架下，系统评估了八种不同MRI表示（T1w、FLAIR、T1w/FLAIR、FLAIR/T1w及其组合）对焦点皮层发育不全（FCD）分割性能的影响，探讨了仅通过优化输入表示即可提升分割效果；

**💡 创新点**

创新点在于将MRI表示设计作为独立实验变量，首次在相同网络结构、预处理与训练设置下对比多种输入组合，展示了比值图像在补充传统序列时可显著提升分割精度，从而提出MRI表示优化是提升深度学习分割性能的重要方向；

**🔧 技术方法**

主要技术包括nnU-Net 3D全分辨率网络、Dice+交叉熵联合损失、基于T1w/FLAIR和FLAIR/T1w的比值图像生成、五折交叉验证以及统计显著性检验；

**📊 数据集**

使用了Schuch等公开的预手术MRI数据集，包含85例FCD患者（每例均有T1w和FLAIR）和25例健康对照，用于训练与评估；

**📈 对比分析**

通过Dice、灵敏度和精度指标对八种输入配置进行对比；最佳四通道E8在所有病例中平均Dice为0.376（相较于传统T1w+FLAIR提升约5%），单模态FLAIR则表现最佳的单通道性能（Dice≈0.369），但在检测率上提升有限；

**⚠️ 局限性**

局限性包括仅使用单一公开数据集、比值图像为手工设计未学习、未检验跨中心或不同扫描协议的泛化能力、并未评估更复杂网络结构与表示联合优化的潜力。

---

## 134. Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues for Monocular Depth

**arXiv ID:** 2607.15600 | [PDF](https://arxiv.org/pdf/2607.15600v1)

**作者:** Jung-Hee Kim `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 EpiDistill 框架，将多视角几何先验通过深度引导的极线注意力和直角立体标记蒸馏到单视角深度网络，实现无视角输入的尺度一致度。

**💡 创新点**

创新点在于：① 引入深度引导的极线注意力，显式强制多视角学习极线几何；② 设计可学习的直角立体标记，使单视角网络在推理时仍保留跨视角注意力结构；③ 通过多视角蒸馏将尺度先验无缝迁移至单视角模型。

**🔧 技术方法**

使用的技术包括 Vision Transformer（ViT）基础网络、跨视角全局注意力、深度引导极线注意力、直角立体标记、轻量级 Dense Prediction Transformer（DPT）头、相机尺度回归头以及多任务损失（相对深度、尺度、射线、蒸馏、注意力损失）。

**📊 数据集**

训练数据来自七个多域数据集：Hypersim、Cityscapes、Eden、ScanNet、ScanNet++、Waymo、nuScenes；评估涵盖 ETH3D、DIODE、KITTI、DDAD、NYU、Bonn、Booster、IBims-1 等公开数据集。

**📈 对比分析**

与现有基线（UniDepthV1/V2、DepthPro、DepthAnything3、Metric3DV2 等）对比，EpiDistill 在大多数零样本和混合场景中显著降低绝对相对误差、RMSE 并提升 δ1 准确率，尤其在 ETH3D 与 DIODE 上实现 30% 以上的误差改进，甚至在部分指标上超过依赖真实相机内参的模型。

**⚠️ 局限性**

局限性包括：① 需要在多视角数据上预训练，单视角模型的性能仍受多视角训练集多样性的影响；② 对极线注意力和直角立体标记的设计在极端畸变或非直角视角下可能效果下降；③ 训练成本高，需多 GPU 与大规模标注，推理仍需比纯单视角网络稍长。

---

## 135. Scalable LLM Agent Tool Access in the Cloud

**arXiv ID:** 2607.15593 | [PDF](https://arxiv.org/pdf/2607.15593v1)

**作者:** Mingxin Li `[一作]` (Nanjing University), Shunmin Zhu `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个云端规模的MCP网关系统，用于统一访问LLM工具，提供工具推荐、会话感知路由和访问控制，支持3000+工具，显著降低工具选择时间和token消耗。

**💡 创新点**

将数据平面与工具调用分离，构建统一网关；采用混合检索实现高召回率的工具推荐；实现会话感知路由与访问控制，降低客户端复杂度；在云端实现可伸缩、低开销的架构。

**🔧 技术方法**

MCP协议封装、混合检索（Hybrid Retrieval）、会话感知路由、访问控制策略、微服务架构、云端负载均衡与Session affinity管理。

**📊 数据集**

使用内部工具库（3000+工具）及内部LLM任务数据，未公开使用公开数据集。

**📈 对比分析**

与传统直接调用方式对比，Hybrid Retrieval保持98% Top‑15召回率；工具选择时间缩短8.9×，token使用减少23.8×；系统在云端可伸缩，单次调用开销低，性能稳定。

**⚠️ 局限性**

依赖云基础设施，可能对多租户或离线环境适配有限；兼容旧版协议的实现复杂；高并发下会话一致性维护成本高；未充分讨论安全与隐私方面的问题。

---

## 136. Hard Rules, Soft Preferences: Bridging Reasoning, Learning, and Optimization for Personalized Packing Checklist Generation

**arXiv ID:** 2607.15562 | [PDF](https://arxiv.org/pdf/2607.15562v1)

**作者:** Himel Dev `[一作]` (529 Tech LLC), Bashima Islam `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个三阶段的智能行李清单生成框架，先用符号规则生成高召回的种子清单，再用偏好学习筛选和排序，最后用 CP‑SAT 优化满足硬约束。

**💡 创新点**

创新点在于将符号推理、偏好学习和约束优化三种方法结合，尤其是通过分离包含与排序避免存活偏差，并实现硬约束下的可行性保证。

**🔧 技术方法**

使用了基于规则的符号推理、梯度提升树（GBM）与 LambdaMART 进行偏好学习，以及 Google OR‑Tools 的 CP‑SAT 求解器进行约束优化。

**📊 数据集**

使用了 604 个旅行场景的人工标注数据，涵盖 378 个物品和 226 条规则；同时在 FlyEnJoy 生产环境中进行部署评估。

**📈 对比分析**

与 LLM、贪心、随机等基线比较，符号引擎召回率 99.7% 超过 LLM 78–81%；包含模型 AUC‑ROC 0.943，排序 NDCG@5 0.923；CP‑SAT 在硬约束下 100% 合规，优于贪心 28% 和随机 10%，并在部署后将清单完成率翻倍、编辑时间显著下降。

**⚠️ 局限性**

主要局限在于需要人工编写和维护大量专业规则，规则维护成本高。

---

## 137. Efficient Frame Selection for Long Videos at Test Time with Attention-Based MLLM Selectors

**arXiv ID:** 2607.15689 | [PDF](https://arxiv.org/pdf/2607.15689v1)

**作者:** Yilin Wang `[一作]` (Zhejiang University), Alex Jinpeng Wang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free的帧选择器，利用多模态大语言模型（MLLM）在验证选择的交叉注意力层中提取查询相关的帧证据，并通过动态规划在固定视觉令牌预算下最优分配候选帧数与每帧令牌压缩比例。

**💡 创新点**

创新点在于：①直接使用MLLM的交叉注意力作为查询感知的帧相关性信号，避免了传统的自回归训练或额外的检索模型；②将候选帧数与视觉令牌压缩联合建模为离散优化问题，用动态规划高效求解；③在保持轻量级选择器的同时，实现与多种答案模型和任务的无缝迁移。

**🔧 技术方法**

核心技术包括：跨模态注意力分析与帧级聚合、重要查询词聚合（max pooling）、视觉令牌压缩（平均池化）、动态规划预算分配、以及对多模态大语言模型的多层选择器设计。

**📊 数据集**

实验数据集：Video-MME、LongVideoBench (LVB)、MLVU、MMBench-Video、CG-Bench、NExT-QA、LongVideo-Reason、LLaVA-Video 等，涵盖从短视频到数小时长视频的多种长视频理解任务。

**📈 对比分析**

与统一采样、CLIP/SigLIP评分、BOLT、AKS、Q-Frame、以及训练型选择器（ViaRL、ReFoCUS 等）对比，所提方法在相同帧预算下平均提升 4.1~6.4 分，且在极低帧预算下提升更为显著；同时推理延迟仅为训练型方法的 1/10，显著提高效率。

**⚠️ 局限性**

局限性包括：①依赖于特定MLLM的交叉注意力分布，若模型架构或训练方式变化可能需重新校准；②在极大视频（数小时）下仍受视觉令牌预算限制，可能无法捕获所有稀疏事件；③对查询词的重要性聚合方式（max pooling）在某些任务中可能忽略多词交互导致的细粒度信息。

---

## 138. Learning a System-Level Surrogate for Hydraulic Excavators: A Simulation-to-Real LSTM Approach

**arXiv ID:** 2607.15656 | [PDF](https://arxiv.org/pdf/2607.15656v1)

**作者:** Shuai Wang `[一作]` (Beijing University of Posts and Telecommunications), Xiaofeng Tao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过仿真到实机的框架，使用LSTM网络学习水力挖掘机的系统级数字替代模型，直接逼近闭环控制行为；

**💡 创新点**

①将挖掘机视为输入–输出算子进行端到端学习，避免传统系统辨识的高阶参数化；②引入一致性感知自适应卡尔曼滤波修正传感器不一致；③使用多步自回归训练与偏置惩罚提升长期轨迹一致性；

**🔧 技术方法**

LSTM网络、MuJoCo物理仿真、闭环行为等价评估、多步自回归损失、适应性卡尔曼滤波、偏置惩罚项；

**📊 数据集**

①MuJoCo仿真数据（12条正弦激励，63.6k样本）; ②真实挖掘机无负载数据（34条，650k样本）； ③真实复合工况数据（33条，84k样本）；

**📈 对比分析**

采用闭环自回归评估（RMSE、MAE、R²），仿真阶段R²>0.94，真实无负载阶段R²>0.92，复合工况在加入偏置惩罚后各关节R²≥0.88，表现优于传统方法并保持长期轨迹一致；

**⚠️ 局限性**

仅在四自由度SY750H挖掘机上验证，极端负载和更复杂动力学的泛化能力未测试；卡尔曼滤波参数需手工调优，适用性受限。

---

## 139. DiTango: Cost-Effective Parallel Diffusion Generation with Selective Attention State Reuse

**arXiv ID:** 2607.15650 | [PDF](https://arxiv.org/pdf/2607.15650v1)

**作者:** Yuyang Chen `[一作]` (Shanghai Jiao Tong University), Jidong Zhai `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于选择性注意力状态重用的并行 Diffusion Transformer (DiT) 推理框架，结合锚点驱动的选择规划和状态中心化的运行时实现，显著降低跨节点通信与计算开销。

**💡 创新点**

创新点包括：①利用上下文并行中的空间局部性特征，将重要的 KV 分区优先计算；②构建注意力状态误差模型和锚点更新机制，实现动态、精准的计算/重用决策；③设计状态中心化的通信/计算管线，提升跨节点重用操作的并行度。

**🔧 技术方法**

核心技术：上下文并行 (CP)、注意力状态重用、锚点驱动误差建模、组级状态合成、状态中心化运行时、层次化通信拓扑、动态组合并与内存管理。

**📊 数据集**

使用的模型与数据集：Wan2.1-14B、Wan2.1-1.3B、HunyuanVideo（来自 VBench Leaderboard）作为基准模型，进行 81 帧 720p 或 129 帧 720p 视频生成实验。

**📈 对比分析**

与 Tensor Parallel、CP、Hybrid CP、VideoSys、SGLang-Diffusion 等基线对比，单节点上实现 1.2–1.9× 的端到端加速（与单 GPU 对比），多节点 32 GPU 环境下可达 1.9× 端到端、3.2× 注意力层加速，质量指标（PSNR、SSIM、LPIPS、VBench）保持与原模型相当或更优，显示出良好的性能-质量折衷。

**⚠️ 局限性**

局限性：在单节点高带宽环境下，通信瓶颈不再显著，纯计算压缩策略可能更具优势；注意力状态缓存占用显存（约 30GB），在 GPU 资源紧张时可能限制吞吐；未来需进一步压缩缓存或结合 CPU offload 以减小内存占用。

---

## 140. Difference-Based Relational Learning for Zero-Shot Object-Goal Visual Navigation With Direct Sim-to-Real Transfer

**arXiv ID:** 2607.15642 | [PDF](https://arxiv.org/pdf/2607.15642v1)

**作者:** Guolei Qi `[一作]`, Feitian Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于差异关系学习与双帧时序记忆的T-DRN框架，实现零样本物体目标导航

**💡 创新点**

创新点在于将目标与视野内所有物体的关系差异作为特征，摒弃绝对识别；并通过双帧记忆缓解真实摄像头视场差距

**🔧 技术方法**

使用Siamese差异模块、LSTM时序处理、A3C策略网络，并结合YOLOv7目标检测和GloVe词嵌入

**📊 数据集**

在AI2-THOR模拟环境中训练，使用22个目标类别；在TurtleBot4机器人上进行物理验证

**📈 对比分析**

与随机、Zhu、MJOLNIR、SSNet、TDANet等基线对比，T-DRN在Sim与Real上均获得最高成功率（Sim：71.9% vs 62.5%，Real：64.1% vs 53.8%）并保持轻量级实现

**⚠️ 局限性**

主要局限是仅使用RGB信息，缺乏深度感知导致部分碰撞或距离估计不准

---

## 141. Efficient Difficulty-Aware Dynamic Routing for Diffusion-Based Real-World Image Super-Resolution

**arXiv ID:** 2607.15711 | [PDF](https://arxiv.org/pdf/2607.15711v1)

**作者:** Xue Wu `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于稳定扩散模型的难度感知动态路由框架DDR-SR，用于真实场景图像超分辨率；

**💡 创新点**

创新点在于通过高频能量衰减估计恢复难度，并根据难度动态选择不同VAE压缩比的专家网络，兼顾细节保留与推理效率；

**🔧 技术方法**

使用了稳定扩散模型、VAE压缩调节、LoRA参数适配、单步扩散推理以及高频能量难度评估和动态路由策略；

**📊 数据集**

训练集采用LSDIR与FFHQ10K合成数据，评估使用DIV2K、RealSR与DRealSR三大数据集；

**📈 对比分析**

与多步与单步方法（StableSR、DiffBIR、SeeSR、PASD、ResShift、SinSR、OSEDiff）对比，DDR-SR在SSIM、LPIPS、CLIPIQA等指标上领跑，且推理仅1步，FLOPs最低；

**⚠️ 局限性**

在MANIQA、NIQE等无参考指标上略逊，且需要手动调节难度阈值，VAE压缩仍会导致细节失真。

---

## 142. Implicit Virtual Leader: Decentralized Vision-Only Relative Pose Estimation for Multi-Robot Formations

**arXiv ID:** 2607.15708 | [PDF](https://arxiv.org/pdf/2607.15708v1)

**作者:** Shiyuan Yang `[一作]`, Qingbiao Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了一种完全去中心化的仅使用单目相机和机器人间通信的相对位姿估计框架，基于隐式虚拟领袖（IVL）和Transformer‑based GNN实现多机器人队形控制。

**💡 创新点**

创新点在于引入IVL作为非物理参考框架，使每个机器人能够相对IVL估计姿态，消除了单点失效和对绝对定位的依赖；通过两阶段Transformer消息传递实现局部特征聚合；使用异方差高斯负对数似然回归给出确定性不确定性，并通过MC Dropout获取模型不确定性；加入相对位置损失保证队形结构不坍塌。

**🔧 技术方法**

采用冻结的DINOv2 ViT‑S/14图像编码器、Transformer‑based GNN、两阶段消息传递、异方差GNLL回归、MC Dropout不确定性估计以及仿真与实测训练与评估管线。

**📊 数据集**

使用HM3D（Habitat Matterport3D）仿真数据集、CoViS‑Net真实世界基准数据集，以及在DJI RoboMaster与Unitree Go2机器人上收集的真实场景数据。

**📈 对比分析**

与CoViS‑Net基准在HM3D与CoViS‑Net测试集上进行对比；IVL模型在未见队形尺寸（N>5）保持≈0.3 m位置误差、≈10°方向误差；在可见与不可见情况下的误差均优于或相当于CoViS‑Net；不确定性指标（ECE、AUSE、Spearman）显示异方差不确定性在模拟与真实环境下更可靠；在隧道通行实验中，三台RM或两RM+一四足机器人在无碰撞或极少碰撞的情况下完成通行，位姿误差约0.2 m。

**⚠️ 局限性**

目前仅实现单帧推断，未利用时间一致性；存在仿真‑实测差距；框架仅关注感知与相对定位，缺乏完整的障碍规避、规划与导航集成。

---

## 143. S1-Omni: A Unified Multimodal Reasoning Model for Scientific Understanding, Prediction, and Generation

**arXiv ID:** 2607.15686 | [PDF](https://arxiv.org/pdf/2607.15686v1)

**作者:** Jiahao Zhao `[一作]`, Zhanao Yao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为S1-Omni的统一多模态推理模型，能够在多种科学任务（分子、材料、蛋白质、光谱、科学图像）上进行理解、预测和生成。

**💡 创新点**

将统一表示、自然世界知识对齐和任务特定解码三大能力整合到单一模型，实现跨学科、跨模态、跨任务的科学推理；同时通过结构化推理监督提升模型内部表示。

**🔧 技术方法**

采用共享的视觉语言模型（基于现有VLM），结合文本、标量、位置、坐标和图像解码器；使用科学法律与专家知识构造推理链；两阶段训练（预填任务协议 + 任务解码器）。

**📊 数据集**

S1-Omni-Corpus，包含200多种科学任务、800万条推理样本，涵盖化学、材料、生物、物理、医学成像等；评测覆盖60+基准。

**📈 对比分析**

与GPT-5.5、Gemini-3.1-Pro以及各领域专用模型对比，在大多数基准上取得更优成绩，在药物性质、蛋白质功能位点、结构预测、图像生成等任务上匹敌或超过专用模型。

**⚠️ 局限性**

尚未进行大规模科学预训练；生成仍依赖外部解码器；任务覆盖有限；对多轮规划、工具使用等长程任务研究不足；科学推理监督对结果真实性需进一步验证。

---

## 144. CSS-BA: Gate-Guided Column Space Search for Bundle Adjustment

**arXiv ID:** 2607.15652 | [PDF](https://arxiv.org/pdf/2607.15652v1)

**作者:** Ayano Kaneda `[一作]` (Waseda University), Shigeo Morishima `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Gate‑Guided CSS‑BA，基于 Schur‑LM 的列空间搜索子空间限制，用于弱几何条件下的 Bundle Adjustment。

**💡 创新点**

创新点在于只在求解器侧对 LM 步长方向做低维子空间限制，并结合几何门控与列空间搜索，保持原目标和变量不变，显著提升低视差/近旋转场景的相对姿态与焦距精度。

**🔧 技术方法**

采用了 Schur 约化、Levenberg–Marquardt、列空间搜索 (CSS)、Lanczos/Ritz 基底构造、几何门控（视角连通性、视差、旋转一致性）等技术。

**📊 数据集**

评估使用了 PhoneSweep（低视差、近球面相机运动）和 BAL（大规模互联网照片）两个数据集。

**📈 对比分析**

与标准 LM 与 PoBA 进行对比，使用 RRA/RTA/AUC@30、AFE 等指标；在 PhoneSweep 上相对姿态精度提升约 3 倍、焦距误差显著下降；在 BAL 上目标函数降幅相当，性能基本相同，但 CSS‑BA 需要更多计算时间。

**⚠️ 局限性**

局限性：仅在弱几何场景下显著受益，强条件下收益有限；子空间构造增加额外计算开销；门控与子空间尺寸的设置仍需经验调优。

---

## 145. Neuro-Symbolic AI for LEED compliance: Document-Centric Benchmarking, Deterministic Numeric Checking, and When Multimodal Hurts

**arXiv ID:** 2607.15647 | [PDF](https://arxiv.org/pdf/2607.15647v1)

**作者:** Aritro De `[一作]` (University of Texas at Austin), Juliana Felkner `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了小型本地语言模型在 LEED v4.1 BD+C 文档合规性检查中的应用，并提出了神经符号化流程

**💡 创新点**

首次构建基于信用章节感知检索和确定性数值检查的神经符号 LEED 合规管线

**🔧 技术方法**

利用局部部署的 Gemma3:4b 等 LLM、检索增强、结构化提取、确定性阈值器以及自洽投票

**📊 数据集**

使用 UT Austin 四座建筑共 484 份 PDF、153 个信用级决策作为基准集

**📈 对比分析**

与文本仅 LLM、不同规模模型及多模态、提示方式对比，Gemma3:4b 文本仅获得 67.3% 准确率，神经符号在部分量化信用上提升至 100% 但总体为 61.6%，多模态反而下降

**⚠️ 局限性**

局限在于缺乏 FAIL 标签、文档结构/提取瓶颈、数值检查覆盖有限、跨建筑推广不稳、对视觉模型依赖不足

---

## 146. Natural Backdoor Attacks on Speech Recognition Models

**arXiv ID:** 2607.15724 | [PDF](https://arxiv.org/pdf/2607.15724v1)

**作者:** Jinwen Xin `[一作]` (Xidian University), Jing Ma `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在语音识别模型中使用自然/日常声音作为触发器的后门攻击，构造污染样本并训练感染模型，评估其攻击成功率与正常性能。

**💡 创新点**

创新点在于：①采用普通自然声音（如雨声、口哨、鸟鸣）作为触发器，使后门在真实环境中可自动触发且不易被察觉；②证明仅5%污染样本即可近乎100%攻击成功率，并且同样适用于无标签清洁攻击。

**🔧 技术方法**

技术方法包括：时间域合成策略将触发音叠加到原始音频；使用MFCC特征提取；训练CNN、LSTM、mini‑CNN三类模型；通过BA和ASR指标评估后门效果。

**📊 数据集**

使用的公开数据集为Speech Commands Dataset V2（10类语音命令）和Eating Sound Collection（20类食物声）进行实验。

**📈 对比分析**

与随机噪声和21 kHz超声波触发器进行对比实验；在两数据集、三模型上天然触发器的攻击成功率超过99%，对正常样本的准确率几乎无影响；在清洁标签攻击下，5%污染率即可使CNN的ASR超过90%，LSTM需要更高比例；实验还展示了触发器时长、混合比例与污染率对ASR的正向影响。

**⚠️ 局限性**

局限性包括：①在低采样率（如16 kHz）下超声波触发器失效，采样率对天然触发器效果影响有限；②清洁标签攻击需要更多污染样本；③实验仅在公开数据集和模拟物理场景下验证，尚未评估在更复杂环境和其他模型上的通用性；④未探讨主动检测或防御机制。

---

## 147. Behavioral Controllability of Agentic Models for Information Extraction: From Fixed Workflows to Reflective Agents

**arXiv ID:** 2607.15715 | [PDF](https://arxiv.org/pdf/2607.15715v1)

**作者:** Lujia Zhang `[一作]`, Hongwei Feng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 NeurIPS 2024 论文 PDF 上，构建了三种系统：固定 LLM 工作流、加入规则/LLM 反思与记忆的 ReAct 代理，以及设计的 S2 优化代理；对比它们在数据集提取任务中的行为轨迹与最终输出。

**💡 创新点**

①首次将“行为可观测、可配置、可复现、可比较”四个维度纳入代理评估；②提出可动态规划、12 个原子工具的 S2 代理；③通过对比过程级指标（工具调用、反思、重试、内存）揭示代理机制对任务完成的真实影响。

**🔧 技术方法**

使用 OpenAI‑compatible vLLM 服务器（LLM 后端保持一致）、ReAct 控制循环、规则/LLM 反思模块、短期/长期记忆注入、PDF 解析工具（页面、表格、引用等）以及动态规划器。

**📊 数据集**

NeurIPS 2024 会议论文集合（最多 50 篇），每篇最多提取 5 条数据集记录；未使用完整人工标注的 gold 语料。

**📈 对比分析**

对比基线工作流、S1a（规则反思）和 S1b（规则+LLM 反思），结果显示记录数从 158 → 165 → 168，链接率低但不随代理提升；日志行数从 7k → 50k，反映代理行为更丰富；S2 设计尚未完成评测。

**⚠️ 局限性**

限制：样本仅 NeurIPS 2024、仅 50 篇论文；无完整标注的精确度评估；链接提取能力不足；LLM 反思对记录数提升有限；S2 代理未实测，结果仅为设计说明。

---

## 148. AC-VLA: Robust Out-of-Distribution Action Execution via Compositional Learning

**arXiv ID:** 2607.15714 | [PDF](https://arxiv.org/pdf/2607.15714v1)

**作者:** Xiaojiang Peng `[一作]` (Shenzhen Technology University), Xiaobo Wang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AC-VLA框架，增强VLA模型在OOV条件下的组合泛化能力

**💡 创新点**

通过LLM驱动的指令分解与运动轨迹对齐生成密集子任务监督，并采用状态条件非对称遮蔽消除腕视纹理捷径

**🔧 技术方法**

利用LLM指令分解、基于关节运动的轨迹分段、混合训练（完整演示+子任务）以及在闭手指阶段遮蔽腕视输入

**📊 数据集**

LIBERO与LIBERO-OOD基准以及实机测试的四项桌面抓放任务

**📈 对比分析**

与UniVLA、OpenVLA-OFT、π_0.5等基线对比，AC-VLA在Spatial/Goal OOD任务上分别提升28.7%/26.7%，整体平均成功率提升至87.3%，实机OOV成功率从35%提升至82.5%

**⚠️ 局限性**

对离线指令分解的准确性和轨迹对齐的鲁棒性敏感，且在已覆盖大多数对象-目标组合的大规模数据集上增益有限

---

## 149. The CRAFT principles for the responsible use of large language models in policymaking

**arXiv ID:** 2607.15704 | [PDF](https://arxiv.org/pdf/2607.15704v1)

**作者:** Willem Fourie `[一作]` (Stellenbosch University), Tanya de Villiers-Botha `[通讯]` (Stellenbosch University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 CRAFT 原则，指导政策制定者在使用大语言模型时保持控制、严谨、问责、公平和透明。

**💡 创新点**

首次将负责任 AI 与政策制定流程结合，形成可操作的五原则框架。

**🔧 技术方法**

基于大型语言模型（LLM）技术的使用场景分析。

**📊 数据集**

未使用特定数据集，文献综述与专家访谈为主要信息来源。

**📈 对比分析**

未进行实验比较，主要通过案例分析和理论讨论说明其适用性。

**⚠️ 局限性**

缺乏实证验证，需在不同政策环境中进一步测试其效果与局限。

---

## 150. SpeechGuard: Online Defense against Backdoor Attacks on Speech Recognition Models

**arXiv ID:** 2607.15697 | [PDF](https://arxiv.org/pdf/2607.15697v1)

**作者:** Jinwen Xin `[一作]` (Xidian University), Xixiang Lv `[通讯]` (Xidian University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 SpeechGuard，一种面向语音识别模型的在线后门防御方案，包含两阶段：S-STRIP 语音适配的扰动检测与基于自编码器生成的时间‑频率掩模净化；

**💡 创新点**

①改进 STRIP 以适配语音（按设定 SNR 注入扰动）②利用自编码器自动生成理想二进制掩模，在时频域抑制触发信号，实现检测‑净化一体化防御；

**🔧 技术方法**

信息熵检测（S-STRIP）、STFT/逆 STFT、信噪比确定扰动、全连接自编码器、理想二进制掩模 (IBM) 与可变软掩模 (IRM) 对比；

**📊 数据集**

Speech Commands Dataset V2（10 类命令）与 AudioMNIST（10 数字），测试 2D‑CNN 与 Att‑LSTM 两种轻量化网络；

**📈 对比分析**

对比后门模型的 BA/ASR，采用 FRR/FAR、PA 等指标；实验显示 SpeechGuard 将 ASR 降至 <10%（>90% 降低），PA 保持 60‑90%，检测 FAR <10%（可调 FRR 以进一步降低 FAR），误检清洁样本净化后准确率下降 ≤10%；

**⚠️ 局限性**

对随机噪声触发的净化效果相对弱（PA 仅 ~60%）；需要少量干净样本训练自编码器；检测阶段需在 FRR/FAR 之间权衡；未验证未知触发或对抗性触发的鲁棒性，且对更大规模任务的通用性待进一步评估。

---

## 151. PCTD: Preference-Guided Counterfactual Task Decomposition for Agent Tool Retrieval

**arXiv ID:** 2607.15696 | [PDF](https://arxiv.org/pdf/2607.15696v1)

**作者:** Chu Zhao `[一作]` (Northeastern University), Fei Huang `[通讯]` (Honor Device Co., Ltd)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为PCTD的强化学习框架，通过联合对比因果奖励和偏好奖励来实现多轮移动场景下的任务分解与工具检索。

**💡 创新点**

创新点在于使用配对反事实奖励消除检索指标与表面特征的虚假相关，并加入偏好奖励对分解结构进行细粒度监督，解决了奖励劫持和OOD泛化失败。

**🔧 技术方法**

技术包括结构因果模型、GRPO（Group Relative Policy Optimization）、对比因果奖励、偏好奖励（Process Reward Model）、多层级注释的MTDTool数据集以及多轮移动交互的工具检索。

**📊 数据集**

使用MTDTool（专门构造的多轮移动交互任务分解基准）和ToolRet（含Web、Code、Custom等多工具库）两大数据集。

**📈 对比分析**

与现有提示式方法和RL基线相比，PCTD在ToolRet OOD、MTDTool In-Domain及Out-of-Domain上均实现最高的N@10和C@10（如Qwen3-8B下N@10 91.19、C@10 87.08），明显优于ToolQP和Prompting方案，且重复率显著下降。

**⚠️ 局限性**

主要限制在于奖励权重的选择需谨慎平衡，过低会导致重复生成，过高可能削弱检索精度；同时在极端不平衡的奖励权重下模型仍可能失效。

---

## 152. RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation

**arXiv ID:** 2607.15730 | [PDF](https://arxiv.org/pdf/2607.15730v1)

**作者:** Ziyi Zhao `[一作]` (University of Science and Technology of China), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RECAP框架，实现流式结构化语义用户画像的离线闭环优化；

**💡 创新点**

1）将LLM语义更新与确定性生命周期管理拆分，保持有限容量的结构化状态；2）通过LLM一致性筛选构造基于标签的一致行为反馈；3）利用双塔评估器生成推荐对齐奖励，驱动GRPO策略优化；

**🔧 技术方法**

大型语言模型（如Qwen3、Claude），确定性状态机，LLM判断器，双塔语义评估器，GRPO强化学习，SFT微调；

**📊 数据集**

Kuaishou短视频内部数据集：约10k用户、500条有效历史、200条训练交互，包含真实用户观看/跳过日志；

**📈 对比分析**

与基线、仅原始反馈、仅清洗反馈、SFT等对照实验，使用uAUC和Recall@2000评估；RECAP在清洗评估中uAUC提升至0.7603、Recall提升至0.0128；在线A/B测试显示平均使用时长提升0.139%；

**⚠️ 局限性**

依赖离线代理奖励可能不足以完全反映真实推荐目标；LLM与确定性状态机的结合在复杂场景下可能不够灵活；数据清洗减少可用样本；模型规模与计算成本较高。

---

## 153. RAVEN: Reinforcement-Adaptive Visibility-Graph Planning for Robust Humanoid Navigation with Collision-Free MPC

**arXiv ID:** 2607.15701 | [PDF](https://arxiv.org/pdf/2607.15701v1)

**作者:** Ruochen Hou `[一作]` (University of California Los Angeles), Dennis W. Hong `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种层次化的 RL–MPC 框架 RAVEN，利用强化学习元策略自适应可视化图规划中的障碍物膨胀参数，并用冲突无关 MPC 追踪规划得到的轨迹，实现机器人在动态环境中的鲁棒导航；

**💡 创新点**

创新点在于将学习任务聚焦于规划图的几何构造（障碍膨胀）而非直接输出控制命令，从而在保持经典规划可解释性和安全保证的同时，通过学习补偿控制延迟、状态噪声等现实不确定性；

**🔧 技术方法**

采用 PPO 强化学习、Brax/JAX 训练框架、MuJoCo Playground MJX 仿真、JAXopt 求解的 DAVG–cfMPC 规划器、以及低层 Boosted Gym 关节控制器；

**📊 数据集**

使用自定义的仿真环境（随机起点/终点、固定圆形障碍）以及 RoboCup 赛场的硬件实验，未使用公开标准数据集；

**📈 对比分析**

与手动调参的 DAVG–cfMPC 基线和纯端到端 RL 基线进行对比；在无延迟场景下 RAVEN 具备最短路径长度和最快完成时间；在 0.06 s 延迟下，RAVEN 的障碍穿透深度显著低于基线（0.03 m vs. 0.128 m），同时保持最短路径和最快时间，硬件实验中 RAVEN 亦表现出与仿真一致的轨迹，说明其对 sim‑to‑real 问题的鲁棒性；

**⚠️ 局限性**

局限包括：仅针对静态障碍物，缺乏对动态障碍物的适配；对 2D 平面运动的假设，需扩展到 3D 或更复杂姿态；MPC 计算仍是瓶颈，帧率受限；RL 元策略需针对不同机器人模型或任务重新训练，泛化性待提升；

---

## 154. Understanding Agent-Reactive Bugs at the Model-Harness Boundary: An Empirical Study of LLM Agent Issue Reports

**arXiv ID:** 2607.15684 | [PDF](https://arxiv.org/pdf/2607.15684v1)

**作者:** Jingyi Chen `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对LLM代理的Agent-Reactive（AR）bug进行首次经验研究，收集并分析了255个来自Codex、Gemini-CLI、LangChain、CrewAI的bug报告，构建了双轴症状-触发行为分类体系，并探讨了用户与开发者的修复偏好。

**💡 创新点**

首次系统化描述AR bug的症状与触发LLM行为的对应关系，提出了双轴分类法，为后续的测试或然、复现与定位提供框架。

**🔧 技术方法**

采用GitHub Issue爬取、关键词过滤、人工标注两阶段流程，结合两位作者的对照评审；并对Bug讨论进行文本分析。

**📊 数据集**

基于四大开源代理项目（Codex、Gemini-CLI、LangChain、CrewAI）共计255条AR bug报告及其关联PR。

**📈 对比分析**

通过对症状分布（静默错误、崩溃、输出错误、重试循环、挂起）与触发行为（指令不遵守、工具参数异常、模板冲突等）的交叉统计，展示了AR bug在不同项目中的差异；无量化性能指标，但表明静默错误占比最高。

**⚠️ 局限性**

研究受限于手工标注的主观性、仅覆盖公开的四个项目、未覆盖闭源代理与更大规模数据，且缺乏自动化检测/修复方案。

---

## 155. Neural Non-Equilibrium Hamiltonian Monte Carlo for Corrected Boltzmann Sampling

**arXiv ID:** 2607.15682 | [PDF](https://arxiv.org/pdf/2607.15682v1)

**作者:** Moxian Qian `[一作]` `[通讯]` (Helmholtz Institute), Moxian Qian (Helmholtz Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Neural Non‑Equilibrium Hamiltonian Monte Carlo（NHMC），先在可行基分布上训练全局随机 Hamiltonian 路径，然后记录正向与反向路径的概率比（即非平衡工作），在采样完成后使用该工作量进行路径自归一化重要抽样（path‑SNIS）、路径独立 Metropolis（path‑IMH）或共享桥回路（round‑trip）进行校正，以在未归一化的 Boltzmann 分布上获得无偏估计。

**💡 创新点**

创新点在于：① 将学习的随机 Hamiltonian 路径与记录的非平衡工作结合，训练后无需再次修正；② 提出了多种校正模式（path‑SNIS、path‑IMH、round‑trip），并证明其对目标保持不变；③ 通过最小化平均工作量来直接控制路径空间不对称性，从而间接降低终点与目标之间的 KL 散度；④ 以可逆、体积保持的 kick–drift/Leapfrog 组合实现高效的路径生成。

**🔧 技术方法**

采用的技术包括：变分学习的条件动量分布、逆可逆体积保持的 kick–drift/Leapfrog 映射、记录正向与反向路径概率、Jarzynski/Crooks 等非平衡工作理论、路径自归一化重要抽样与独立 Metropolis 采样，以及在不同实验中对路径工作量的统计与正则化。

**📊 数据集**

使用的数据集与实验目标包括：可解析双阱（DW4、DW8）、二维 8×8 ϕ⁴ 场（κ 轨迹）、小分子内部坐标（Alanine 四聚体与六聚体）以及 compact U(1) gauge 场。

**📈 对比分析**

与传统 HMC、AIS、L2HMC 等基准在相同采样预算下比较。双阱实验中，DW4 的 log‑normalizer 误差为 6.9×10⁻³，ESS 为 18.3%，接受率 30.2%；DW8 误差 2.45×10⁻⁴，ESS 2.6%，接受率 12.7%。ϕ⁴ 实验中，path‑SNIS 的 ESS 随 κ 变化，峰值附近下降至 5%；round‑trip 通过两条路径提升接受率并将自相关时间从 19.95 降至 8.94。总体而言，NHMC 在模式覆盖、自由能估计和自相关方面优于传统方法，但受限于路径重叠。

**⚠️ 局限性**

主要局限：① 路径工作量的方差大，导致权重退化、低接受率和长自相关，尤其在高维或多模目标（如 DW8、compact‑U(1)）中表现不佳；② 对分子内部坐标实现的边界投影未满足可逆映射假设，限制了理论证明的适用性；③ 训练目标主要关注平均工作量，未直接控制终点分布的覆盖，导致某些模式被忽视；④ 需要进一步改进反向动量分布、利用对称性以及更高效的正则化策略。

---

## 156. ADASCALE: An Adaptive Scaling and Placement Framework for Microservices Under Dynamics

**arXiv ID:** 2607.15681 | [PDF](https://arxiv.org/pdf/2607.15681v1)

**作者:** Ming Chen `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdaScale，一个基于 MAPE 循环的自适应伸缩与放置框架，实时根据动态呼叫图、网络延迟和根请求优先级来共同调度微服务实例。

**💡 创新点**

创新点在于：① 将伸缩与放置联合建模为需求加权延迟最小化问题；② 采用双时标（慢速伸缩、快速放置）并结合根请求重要性评估；③ 在伸缩决策中融入根请求 SLO 风险与服务需求权重，提升了对非平稳负载与网络扰动的响应。

**🔧 技术方法**

核心技术包括：Kubernetes 横向 Pod 自动伸缩（HPA）改进、Istio 服务网格、Jaeger 追踪与 Prometheus 指标收集、基于请求率的需求表估计、贪心延迟最小化放置算法、ICMP 延迟测量、动态根请求优先级计算。

**📊 数据集**

使用的基准数据集为 DeathStarBench Social Network，部署在 16 节点的云‑边缘 Kubernetes 集群上进行实验。

**📈 对比分析**

与原生 HPA 和 NetMARKS_Scale 进行对比，AdaScale 在三类根请求场景下平均响应时间降低 1.34–1.93 倍，吞吐量提升 1.32–2.16 倍，且始终满足 SLO 约束。

**⚠️ 局限性**

局限性：实验仅覆盖单一微服务应用，缺乏多样化场景；未对动态带宽变化做完整建模；在极端网络抖动或极大规模集群下的可扩展性与稳定性尚待进一步验证。

---

## 157. A Generative Partially Specified Finite State Machine Approach to Complex Behaviour Planning

**arXiv ID:** 2607.15674 | [PDF](https://arxiv.org/pdf/2607.15674v1)

**作者:** Kalana Ratnayake `[一作]` (University of Canberra), Damith Herath `[通讯]` (University of Canberra)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于有限状态机的生成式规划框架（GPSFSM），通过 Fabric、Capabilities2 与 PromptTools 在 ROS2 环境下实现机器人行为的文本描述、LLM 生成、验证与执行。

**💡 创新点**

创新点在于：①首次将 LLM 与 FSM 结合实现生成式行为规划；②采用标准化的能力语义描述，提升跨机器人可迁移性；③设计了事件驱动的动态加载机制，兼顾资源管理与容错。

**🔧 技术方法**

使用技术包括：ROS2 与 Capabilities2 框架、Fabric FSM 引擎、PromptTools 的 LLM 调用与提示缓冲、LLM（OpenAI GPT‑4o/4.1/5、LLamaChat、Codellama）以及事件驱动 FSM 结构。

**📊 数据集**

使用数据集主要为 Nav2 轨迹规划任务（模拟与 Turtlebot4 机器人），涵盖从单点到多点导航并加入恢复点的五个任务；此外还测试了涉及语音、图像识别等感知任务。

**📈 对比分析**

与 BTGenBot 进行对比，指标包括生成成功率、部分成功率、失败率及生成时延；实验显示 GPSFSM 在所有 GPT 模型下零/一次提示成功率约 90%，显著高于 BTGenBot（约 54%），且生成时延相当或更低。

**⚠️ 局限性**

局限性包括：本地 LLM 受限于上下文窗口导致复杂任务失败；实验范围仅限于导航与感知任务，未覆盖机械臂操作；缺乏对运行时错误检测与自适应纠错机制。

---

## 158. Model Merging for Medical LVLMs: A Benchmark and a Winner-Take-All Approach

**arXiv ID:** 2607.15661 | [PDF](https://arxiv.org/pdf/2607.15661v1)

**作者:** Lichao Mou `[一作]` (MedAI Technology Co. Ltd.), Yaxiong Chen `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

对医疗视觉语言模型（LVLM）进行模型合并的系统研究，提出合并基准 MergeMedBench 并提出 winner‑take‑all 合并方法。

**💡 创新点**

创新点在于提出只保留各 LoRA 参数中最显著（dominant）值的“winner‑take‑all”策略，避免平均化导致重要信息丢失；同时构建了覆盖 8 种影像模态的完整基准。

**🔧 技术方法**

使用 LoRA 低秩微调、参数合并、归一化幅度评分以及基于分位数的赢家选择技术。

**📊 数据集**

基准数据集为 OmniMedVQA 的 88,995 张图像问答对，覆盖 CT、MRI、X‑ray、超声、皮肤镜、视网膜、OCT 与显微镜等 8 种模态。

**📈 对比分析**

与 8 种现有无梯度合并方法（TA、TIES‑Merging、DARE‑TIES、Iso‑C、Iso‑CTS、STF、RobustMerge、KnOTS、Core）进行对比；winner‑take‑all 在 Qwen‑VL 与 InternVL 上平均准确率分别达到 90.88% 与 84.64%，明显优于基线且合并时间仅为 0.21 s/0.17 s，性能和效率兼优。

**⚠️ 局限性**

局限性包括仅在 LoRA 微调的 LVLM 上验证，未考察其他微调方式或更大规模模型；合并方法仅针对参数级别，缺乏对任务级多模态交互的深入分析；在极端模态差异或少数样本场景下的鲁棒性仍待进一步研究。

---

## 159. ToolVerse: Unlocking Massive Environments and Long-Horizon Tasks for Agentic Reinforcement Learning

**arXiv ID:** 2607.15660 | [PDF](https://arxiv.org/pdf/2607.15660v1)

**作者:** Shuaiyu Zhou `[一作]` (Meituan), Ke Zeng `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了ToolVerse框架，自动化生成大规模可执行工具环境，利用工具依赖图生成多步长任务，并通过Turn‑Aware Relative Advantage算法改进长时序工具使用的信用分配；

**💡 创新点**

创新点包括①闭环自动合成可执行环境，②基于工具依赖图的动态解锁采样(DUS)生成长时序任务并构造GUST数据集，③细粒度的回合级信用分配TARA，显著缓解稀疏奖励导致的信用分配问题；

**🔧 技术方法**

技术手段涵盖：自动化Schema Refactoring+字典构建+单元测试；工具依赖图(TDG)构造与动态解锁采样；Turn‑Aware Relative Advantage(TARA)信用估计；RL训练采用GRPO；评估使用BFCL‑v3、τ²‑Bench、ACEBench‑Agent等多轮任务环境；

**📊 数据集**

使用的主要数据集为GUST（422个工具环境、4438工具、数千任务），并在公开基准BFCL‑v3、τ²‑Bench、ACEBench‑Agent进行实验评测；

**📈 对比分析**

与ToolRL、Agentflow、SimpleTIR等公开基线对比，TARA在所有基准上均实现显著提升：例如在ACEBench‑Agent中Qwen3‑8B+TARA从56.66%提升到61.66%；在BFCL‑v3中从12.88%提升到20%；整体表现表明ToolVerse在多模型、多规模下均具备显著优势；

**⚠️ 局限性**

局限性包括①依赖预定义的工具依赖关系，缺乏自发发现新交互的能力；②TARA仍需每回合奖励，稀疏或模糊奖励场景下效果有限；③大规模工具合成过程中对无效工具过滤率高，可能限制环境多样性；④实验主要在模拟环境，缺乏真实世界验证。

---

## 160. Do Agents Dream of False Memories? Black-box Visual Attacks on Long-term Memory in Multimodal AI Agents

**arXiv ID:** 2607.15657 | [PDF](https://arxiv.org/pdf/2607.15657v1)

**作者:** Halima Bouzidi `[一作]` (University of California, Irvine), Mohammad Abdullah Al Faruque `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Lucid的黑盒视觉攻击框架，利用不可见像素扰动在多模态 AI 代理的持久长期记忆中实现记忆中毒与注入。

**💡 创新点**

创新点在于：1）仅通过图像扰动即可在不访问文本、模型权重或记忆结构的情况下攻击；2）同时实现上下文中毒与无上下文注入两种攻击模式；3）构建三阶段流水线（目标设计、载荷构造、扰动优化），并在多种内存架构上实现跨模型迁移。

**🔧 技术方法**

采用 CLIP 及其他视觉-语言模型的多模组对抗优化（FOA-Attack 与文本对齐目标），联合使用三种公开视觉编码器（CLIP-ViT-B/16、CLIP-ViT-B/32、LAION-CLIP）作为 surrogate，利用投影梯度法在 ε=16/255 范围内生成扰动。

**📊 数据集**

使用 ShareGPT4V‑100K 作为候选视觉库；Mem‑Gallery 作为多会话对话记忆基准；在四大高危语义类别（身份、过敏/安全、联系/凭证、活动限制）进行注入实验。

**📈 对比分析**

对比 Clean、Oracle 与 Adversarial 三种条件，在五种内存后端（MuRAG、NGMemory、AUGUSTUS、UniversalRAG、Mem0Memory）以及五大 MLLM（GPT‑4o‑mini、GPT‑4o、GPT‑4.1、Claude‑Haiku‑4.5、Gemini‑Flash‑2.5）上评估，攻击成功率（ASR）分别约为 61.6%（中毒）和 58.4%（注入），与 Oracle 只差 4-10%；检索率与 Oracle 一致，说明攻击能可靠驱动检索。

**⚠️ 局限性**

局限性包括：1）对抗扰动主要针对 CLIP‑基准，跨域视觉编码器的迁移效果可能下降；2）高频图像预处理（高斯模糊、JPEG 压缩）可显著降低 ASR，但不一定对更先进的对抗技术（如 Psi）有效；3）文字层面防御（写入过滤、检索过滤）对该攻击无效，需更复杂的跨模态一致性检查。

---

## 161. Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models

**arXiv ID:** 2607.15655 | [PDF](https://arxiv.org/pdf/2607.15655v1)

**作者:** Yingqian Cui `[一作]` (Michigan State University), Yue Xing `[通讯]` (Michigan State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AdaLook，一种自适应多步 lookahead 解码框架，用于提升 Masked Diffusion Language Models 的解码效率与质量。

**💡 创新点**

创新点在于使用候选分数方差动态决定是否继续 roll‑out，并在中间状态启用分支扩展，以避免不必要的深度 roll‑out 并实现更灵活的未来规划。

**🔧 技术方法**

结合了 Explore‑then‑Exploit 策略、批量前向推理、动态分支扩展以及自适应 roll‑out 深度控制等技术。

**📊 数据集**

实验基于 LLaDA‑8B‑Instruct 模型，在 MMLU、GSM8K、MATH500、BBH 四个公开数据集上进行评估。

**📈 对比分析**

与 ETE、Fast‑dLLM 等基线对比，AdaLook 在相同解码步骤数下在更难的数学与推理任务上提升 4–5% 的准确率，并整体获得更优的准确率‑解码步骤折衷曲线。

**⚠️ 局限性**

局限在于多候选前向推理导致每步算力开销略增，虽然在现代 GPU 上影响可忽略，但仍需关注计算成本。

---

## 162. Testing Distributions Against Bounded Distinguishers

**arXiv ID:** 2607.15645 | [PDF](https://arxiv.org/pdf/2607.15645v1)

**作者:** Mark Bun `[一作]` (Boston University), Renato Ferreira Pinto `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出并研究了在有限可区分器族（fooling distance）下的分布身份测试（Identity Testing）问题，并将其与可测试学习（Testable Learning）、PAC 验证（PAC Verification）以及结构化分布（Structured Distributions）的测试问题建立了理论关联。

**💡 创新点**

创新点主要有：
- 将 fooling distance 引入分布测试框架，提供一种可在高维/连续域上实现样本高效且可计算的身份测试方法；
- 通过 Rademacher 复杂度刻画了对特定参考分布进行身份测试的样本复杂度，并给出上下界；
- 展示了可测试学习与身份测试、以及身份测试与可测试 PAC 验证之间的双向可归约关系，进而获得新的学习与验证算法；
- 利用上述理论框架，给出了低阶多项式密度、决策树分布等结构化分布在 TV 距离下的样本与时间上高效的身份/相似性测试算法。

**🔧 技术方法**

核心技术包括：
- Integral Probability Metrics（IPM）与 fooling distance 的定义与分析；
- Rademacher 复杂度与 VC 维度的上界/下界技术；
- 通过构造混合分布 M(P,Q) 与错误区分器族 f⊕ 将身份测试转化为学习与验证问题；
- 黑盒归约与模拟证明，连接可测试学习、PAC 验证、以及结构化分布测试；
- 典型的 Hoeffding 与马尔科夫不等式，用于概率误差控制。

**📊 数据集**

本文为理论研究，不使用具体数据集；所有结果均为信息理论与算法复杂度的抽象分析。

**📈 对比分析**

通过 Rademacher 复杂度与 VC 维度的结合，本文给出了对特定参考分布的身份测试样本复杂度为 O(√m)（其中 m 与 Rademacher 复杂度相关），而对所有分布的身份测试则为 O(d/ε²)；相比传统的 TV 距离身份测试，样本复杂度大幅下降（尤其在高维/连续域）。在结构化分布方面，作者提供的身份/相似性测试在样本与时间上均优于先前的通用方法，并可实现多项式时间算法。

**⚠️ 局限性**

局限性包括：
- 对 Rademacher 复杂度与样本复杂度之间存在二次差距，尚未得到完全匹配的下界；
- 虽然给出了可计算的上界，但缺乏通用的计算复杂度阐述，仍需针对具体分布族设计高效算法；
- 结果多以信息理论上限为主，实际实现中可能受限于分布描述与样本生成的成本；
- 该框架主要针对布尔可区分器族，对更一般的可测函数族适用性仍待研究。

---

## 163. BCG-Former: Toward Pareto-Efficient Hyperspectral Image Classification via Band-Contextual Gating

**arXiv ID:** 2607.15639 | [PDF](https://arxiv.org/pdf/2607.15639v1)

**作者:** Gaurav Sharma `[一作]` (University of Arizona), Eungjoo Lee `[通讯]` (University of Arizona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级CNN‑Transformer混合模型BCG‑Former，针对高光谱图像分类的准确率‑延迟 Pareto 前沿进行优化。

**💡 创新点**

创新点包括：Band‑Contextual Gating（基于局部光谱上下文的自适应通道加权）、Spectral Summary Token（光谱全局聚合标记与空间标记融合）、单通Band‑RoPE与ELU核线性注意力（一次性旋转位置编码与线性复杂度注意力）。

**🔧 技术方法**

采用CNN stem提取空间特征、Band‑Contextual Gating调节光谱通道、旋转位置编码（Band‑RoPE）、线性注意力（ELU核）以及光谱‑空间融合的token构造。

**📊 数据集**

在八个经典空中与无人机高光谱数据集上评估：Pavia University、Salinas、Indian Pines、Houston 2013/2018、WHU‑Hi‑LongKou、WHU‑Hi‑HongHu、WHU‑Hi‑HanChuan。

**📈 对比分析**

与HiT、HybridSN、SpectralMamba、SSFTT、Swin‑HSI等现有方法在OA、AA、κ、GFLOPs、延迟、参数量等指标上进行对比；BCG‑Former在多数数据集上实现最高准确率且推理延迟最低，稳居准确率‑延迟 Pareto 前沿。

**⚠️ 局限性**

局限性包括：尚未在全图推理、跨时相/多模态融合场景下验证；缺乏硬频带选择机制，且对不同光谱采样率的适应性仍待进一步研究。

---

## 164. IoUPD: IoU-Aware Privileged Distillation for Visual Grounding with Multimodal Large Language Models

**arXiv ID:** 2607.15732 | [PDF](https://arxiv.org/pdf/2607.15732v1)

**作者:** Xiuyuan Zhu `[一作]` (University Of Chinese Academy Of Sciences), Jian Xue `[通讯]` (University Of Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 IoU 关注的特权蒸馏方法，用于改进多模态大型语言模型的坐标生成式视觉定位。

**💡 创新点**

利用训练时的真实框作为特权输入指导教师模型，并结合 IoU 感知的 token 加权蒸馏，解决坐标字符串与几何重叠度的训练评估不匹配。

**🔧 技术方法**

特权教师蒸馏、监督微调(SFT)锚定、IoU 关注的 token 权重、冻结教师模型、坐标字符串解码/解析等技术。

**📊 数据集**

使用 RefCOCO、RefCOCO+、RefCOCOg 等标准参考表达式定位基准数据集。

**📈 对比分析**

在统一 prompt/解析/评估协议下与开源 VLM、专用 VLM 以及检测模型对比，mIoU 最高，Acc@0.5/0.7 均显著提升，尤其在较高 IoU 阈值下表现突出。

**⚠️ 局限性**

需要标注框作为特权信息；改进幅度相对保守，仅在强基模型上提升；教师提示过于简单，可能限制进一步提升。

---

## 165. CardioMeta: Calibrated Multi-Task Prediction of Diabetes, Hypertension, and Cardiovascular Disease Across Population and EHR Data

**arXiv ID:** 2607.15721 | [PDF](https://arxiv.org/pdf/2607.15721v1)

**作者:** S M Asif Hossain `[一作]` (Wichita State University), Jungpil Shin `[通讯]` (University of Aizu)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了CardioMeta多任务学习框架，用于联合预测糖尿病、高血压和心血管疾病，并严格控制标签泄漏与实现概率校准。

**💡 创新点**

创新点在于结合泄漏安全的特征设定、共享心血管编码器与疾病专属门控头、后处理温度校准，并系统报告校准误差与子组可靠性，完成跨NHANES与MIMIC‑IV的迁移评估。

**🔧 技术方法**

采用多任务神经网络（共享MLP编码器+门控头）、后处理温度校准、梯度提升树、MLP、TabNet/TabTransformer等基线，配合类权重、焦点损失、SMOTE‑ENN等平衡手段。

**📊 数据集**

使用NHANES（2011‑2018训练与2020预流行验证）与MIMIC‑IV（v2.2）作为外部EHR域迁移与微调的数据集。

**📈 对比分析**

与临床风险评分、梯度提升、MLP、TabNet/TabTransformer等基线在宏观AUROC、AUPRC、macro‑F1和ECE上对比，CardioMeta在泄漏减小设定下取得宏观AUROC0.839、macro‑F1 0.614、ECE 0.024的最佳表现；在MIMIC‑IV迁移中性能下降但微调后显著提升。

**⚠️ 局限性**

局限性包括仅为横断面筛查而非发病预测，跨域迁移依赖微调，仍受测量缺失与分布差异限制，解释性仅为关联性而非因果。

---

## 166. Verified LLM-Driven Synthesis for Concept Design

**arXiv ID:** 2607.15718 | [PDF](https://arxiv.org/pdf/2607.15718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 167. Per-Stroke Temporal Control for Text-to-Motion via Action Units and Action-Detection Guidance

**arXiv ID:** 2607.15717 | [PDF](https://arxiv.org/pdf/2607.15717v1)

**作者:** Euijun Jung `[一作]` (Seoul National University), Youngki Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出“Action Unit”作为每个动作细节的显式控制原语，并在冻结的文本‑转‑运动生成器上通过轻量级门控自注意力适配器实现对动作序列的精细时序与轨迹控制。

**💡 创新点**

创新点包括：①将单个动作的时序、轨迹、侧向和核心时刻打包为可编程的 Action Unit；②使用两流（按动作用token与按帧阶段通道）门控注入；③引入无训练权重的 Action‑Detection Guidance 在推理阶段校正时序误差；④构建 StrokeBench 评测基准并公开审核过的 Action Unit 语料。

**🔧 技术方法**

技术手段包括门控自注意力适配器、Per‑stroke token 编码、Per‑frame phase 通道、固定的离窗口速度先验、Classifier‑Free Guidance、冻结帧级动作检测器的梯度引导。

**📊 数据集**

使用的数据集为 HumanML3D（训练与测试）、FrankenMotion（Held‑out）以及 AMASS 的 body‑part atoms，用于构建 8 类 Action Unit 语料库（≈7,600 个 AU）。

**📈 对比分析**

与 MDM、OmniControl、Kimodo、FineMoGen、STMC、UniMotion、DART、FrankenMotion 等基准在 StrokeBench 上对比，F1AU 提升至 0.898（领先 0.839），IUE 0.886，GLR 0.087，FID 27.8；在链式与重叠子基准上同样取得最优或相近表现。

**⚠️ 局限性**

局限性包括：核心时刻控制在某些动作（如接收动作）不佳；对同一轨道上重叠动作的分离受限于帧级检测器；方法依赖冻结模型，扩展到更大动作集合或不同模型需进一步验证。

---

## 168. A Benchmark for Electrical Load Forecasting Across Grid Levels: Time-Series Transformers Outperform Established Methods

**arXiv ID:** 2607.15705 | [PDF](https://arxiv.org/pdf/2607.15705v1)

**作者:** Matthias Hertel `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了跨电网层级的负荷预测基准，系统评估了Transformer及TSFM在TSO、低压配电网及终端用户三类公开数据集上的表现；

**💡 创新点**

提出可通过超参数优化的灵活Transformer架构YAformer，并验证标准Transformer已能达到最优预测精度，证明对Transformer做进一步修改并非必要；

**🔧 技术方法**

使用标准Encoder-Decoder Transformer、YAformer、Chronos‑2（TSFM）以及传统统计与机器学习模型（ARIMA、LightGBM、MLP、CNN、LSTM、N‑HITS、TFT）；

**📊 数据集**

三大公开数据集：TransnetBW（TSO级负荷）、FeederBW（低压配电馈线负荷）和Electricity‑287（终端客户负荷），均加入天气和日历协变量；

**📈 对比分析**

通过每小时滚动预测、MAE/ nMAE指标及DM检验进行公平比较，结果表明Transformer比最优非Transformer模型降低6.6–10.7%误差；Chronos‑2在短期预测上表现优异，但在长周期和节假日等稀有事件上误差显著；

**⚠️ 局限性**

局限包括：1）仅提供确定性预测，未覆盖不确定性；2）TSFM对稀有事件和长周期的预测不足；3）实验假设历史负荷和天气信息即时可用，实际场景下可能存在延迟；4）仅在公开数据集上验证，缺乏实时运营验证。

---

## 169. Continuously Stable Structure through Plastic Deformation

**arXiv ID:** 2607.15659 | [PDF](https://arxiv.org/pdf/2607.15659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 170. Toward Federated Multimodal Graph Foundation Models: A Topology-Aware Multimodal Alignment Framework

**arXiv ID:** 2607.15687 | [PDF](https://arxiv.org/pdf/2607.15687v1)

**作者:** Xunkai Li `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FedGAMMA，一种两阶段联邦多模态图基础模型，结合共享-私有语义增强、拓扑感知图融合与双通道亲和聚合，并在预训练后使用轻量级图感知提示进行微调；

**💡 创新点**

创新点在于（1）通过共享‑私有流与最优传输对图中图像与文本进行对齐，避免模态坍塌；（2）利用语义残差图和双位置编码分离语义与结构视图，防止结构漂移；（3）提出双通道亲和聚合，基于特征和图的中心化统计分别聚合，提升跨客户端异构性适应；（4）引入全局提示池与通道级提示同步，实现参数高效的下游迁移；

**🔧 技术方法**

使用技术包括：对称交叉注意力、Sinkhorn‑Wasserstein对齐、InfoNCE对比损失、双通道MMD亲和度估计、图卷积网络、随机游走和拉普拉斯位置编码、UCB式提示池探索、联邦平均与个性化插值；

**📊 数据集**

在MM‑OpenFGL基准上十二个多模态属性图数据集（Bili、Music、DY、EleFashion、Flickr30K、Grocery、KU、Movies、QB、RedditS、Sports、Toys）进行实验；

**📈 对比分析**

与九个基线（Fed‑MGNet、Fed‑MMGCN、Fed‑MGAT、MM‑FedGFM、MM‑FedBook、MM‑FedGALA、Fed‑Mario、Fed‑PLANET、Fed‑UniGraph2）以及不同分区（Louvain、Metis）进行比较，FedGAMMA在所有任务上均取得最高分，尤其在跨模态检索、匹配和对齐任务上提升约5–15%，且在少样本场景下仍保持领先；

**⚠️ 局限性**

局限性包括：仅针对图像-文本两模态；对联邦隐私的正式理论证明尚缺；在更大规模真实部署（如数十个客户端、极端异构）下的鲁棒性与可扩展性待进一步验证。

---

## 171. Beyond Detection: Agentic Attack Synthesis and Simulation for Smart Contracts

**arXiv ID:** 2607.15673 | [PDF](https://arxiv.org/pdf/2607.15673v1)

**作者:** Xianhao Zhang `[一作]` (Beijing Institute of Technology), Yuqiang Sun `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了一个多智能体框架（KASSS），通过检索增强的规划、形式化生成与验证约束以及双层循环细化，自动化生成可执行的智能合约攻击，并验证其可行性与损失；

**💡 创新点**

创新点在于将漏洞检测与可执行攻击验证结合，利用真实审计知识检索来精准规划攻击路径，并通过双层循环在代码层面纠错、策略层面重新规划，实现高成功率的自动化攻击合成；

**🔧 技术方法**

技术手段包括检索增强的贝叶斯推理规划、结构化JSON攻击计划生成、Solidity语法约束的PoC代码翻译、Foundry测试环境下的CEGIS式迭代细化；

**📊 数据集**

使用的数据集为SmartBugs‑Curated的104份智能合约（Reentrancy、Arithmetic、Unchecked Low Level Calls、DoS四类）以及11份真实CVE案例；

**📈 对比分析**

与同协议Claude Code基线以及公开的REX、AdvSCanner结果对比，KASSS在SmartBugs上达94.23%的成功率，明显高于Claude Code（20.19%）、REX（50%）和AdvSCanner（80%）；在11份CVE案例中验证9例，展示了更好的迁移能力；

**⚠️ 局限性**

主要限制包括对细粒度时间/状态约束的推理仍不完善、对极少见或未覆盖的漏洞类型缺乏通用性、以及对LLM输出的不确定性和对检索结果的依赖。

---

## 172. PE-Field 4D: Video Generation Models as Canvas

**arXiv ID:** 2607.15667 | [PDF](https://arxiv.org/pdf/2607.15667v1)

**作者:** Yunpeng Bai `[一作]` (University of Texas at Austin), Qixing Huang `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种几何感知视频生成框架，利用输入图像/视频重建场景几何并在目标摄像机视角上投影参考内容，随后使用视频扩散变压器根据结构化上下文合成沿指定摄像机轨迹的视频，保持场景外观和空间结构。

**💡 创新点**

将场景几何重建与视频扩散模型结合，首次在生成过程中利用位置对齐的几何上下文，实现对摄像机运动的精准控制与场景一致性。

**🔧 技术方法**

三维几何重建（多视角深度/神经辐射场）、投影投射、视频扩散变压器（Transformer式扩散网络）以及位置编码。

**📊 数据集**

在公开的室内外三维视频数据集（如 ScanNet、DeepMind 4D、Kinetics-400 等）上进行训练和评估。

**📈 对比分析**

与传统视频生成方法（如 4D Video Diffusion、GAN、DALL·E 3）进行对比，使用 FID、LPIPS、视频结构一致性指标，实验显示在保持空间一致性和摄像机轨迹控制上提升约 15%-20%。

**⚠️ 局限性**

对几何重建精度高度依赖，复杂动态场景下表现有限；模型推理速度较慢，内存占用高，且需额外计算重建步骤。

---

## 173. On the Structure of Address in Multi-Party Dialogue: From Discrete Labels to Continuous Levels

**arXiv ID:** 2607.15648 | [PDF](https://arxiv.org/pdf/2607.15648v1)

**作者:** Taiga Mori `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建连续的“地址级别”来替代传统的离散地址标签，探究其对多方对话中转移、注视和回音行为的预测效果。

**💡 创新点**

创新点在于将地址视为连续变量并使用潜在变量模型从多名标注者的离散标记中估计，提供更细粒度、更能反映真实交互的地址表征。

**🔧 技术方法**

采用贝叶斯潜在变量模型估计地址级别，并用贝叶斯混合效应回归（GLMM）评估地址与转移、注视比例及回音计数之间的关系。

**📊 数据集**

使用日本实验室成员的三人对话语料库Teidan Corpus（36段对话，共3121个回合）。

**📈 对比分析**

与传统的多分类地址标签模型比较，连续地址模型在下一个发言人预测、注视比例和回音计数上均获得更高的预测拟合（ELPD差异显著，且可信区间不包含零）。

**⚠️ 局限性**

局限性包括仅使用日语三人讨论数据、标注者人数有限（每段三人）、未传播完整后验不确定性、以及标注者在地址判断中已利用注视信息导致与后续注视分析的相关性可能被高估。

---

## 174. StemFX: Learning Mixing Style Representations via Autoregressive FX Chain Prediction on Source-Separated Stems

**arXiv ID:** 2607.15634 | [PDF](https://arxiv.org/pdf/2607.15634v1)

**作者:** Yuan-Chiao Cheng `[一作]` (Independent Researcher), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过端到端自回归FX链预测学习混音风格表示，利用源分离得到的伪多轨数据进行大规模训练。

**💡 创新点**

①将FX链生成作为学习目标，②提出基于多频段CNN与FiLM条件的编码器，③构建可扩展的源分离+增强训练管道和FX工具包。

**🔧 技术方法**

Transformer自回归解码器、band‑split multi‑band CNN编码器+FiLM、对齐编码器‑解码器的配对训练、源分离与随机FX链生成、FX链token化及交叉熵训练。

**📊 数据集**

训练集：约105K首FMA音乐通过SCNet分离得到四轨伪多轨；评估集：MUSDB18专业混音。

**📈 对比分析**

与Fx‑Encoder++、AFx‑Rep、CLAP等基线及自建BSFiLM‑CL对比，检索Top‑1最高86.8%（对比77.8%或更低），风格迁移MRSTFT 1.44、MUSHRA 60.6，推理速度相对迭代优化快4000倍。

**⚠️ 局限性**

仅能预测训练集内的FX集合，受限于四轨分离的伪多轨，随机链训练可能缺乏真实混音结构，分离误差未评估，难以直接扩展到更细粒度的专业会话。

---

## 175. Event3R: Asynchronous-to-Global 3D Reconstruction from Event Camera via Spatial-Temporal Feature Aggregation

**arXiv ID:** 2607.15727 | [PDF](https://arxiv.org/pdf/2607.15727v1)

**作者:** Jian Huang `[一作]` (Zhejiang University), Peidong Liu `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Event3R，一个利用事件流实现即时全局3D重建的框架。

**💡 创新点**

创新点包括：将异步事件流转化为时空体素，并通过时域注意力捕获运动信息；提出自监督掩码时域建模（MBM）用于预训练与微调，显著提升时域特征学习。

**🔧 技术方法**

核心技术包括：事件体素化、Patch‑Level Temporal Encoder（自注意力）、DUSt3R骨干网络、MBM自监督预训练以及对比与一致性损失。

**📊 数据集**

使用的数据集有：预训练阶段采用MatrixCity和TUM‑VIE；微调与评估阶段采用TartanAir、MVSEC、TartanAirEvent与MVSEC等。

**📈 对比分析**

与EvGGS、E2Depth、DepthAnyEvent等事件深度估计方法，以及IncEventGS、DEVO等姿态估计方法对比，Event3R在AbsRel、δ1.25、ATE、3D重建准确率等指标上均显著优于基线。

**⚠️ 局限性**

局限性包括：对GPU显存要求高、时域体素分辨率受限、需要多帧事件块同步处理，并且在极端光照或极高速运动场景下鲁棒性仍需提升。

---

## 176. Transient State Reorganization and Cell Differentiation in the Developmental Dynamics of Growing Neural Cellular Automata

**arXiv ID:** 2607.15726 | [PDF](https://arxiv.org/pdf/2607.15726v1)

**作者:** Hiroki Sato `[一作]` (University of Tokyo), Takashi Ikegami `[通讯]` (University of Tokyo)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

追踪了 Growing Neural Cellular Automata (GNCA) 从单细胞种子到成熟形态的完整发育轨迹，并通过分析细胞状态的多样化、通道自组织、低维平滑流形上的分化以及细胞类型的出现，揭示了其发育过程的时序与结构特征。

**💡 创新点**

发现 GNCA 的发育表现为非单调、通过中间临时结构的重新组织而非逐步细化；通道自组织形成模块化块；细胞状态在低维平滑流形上分化；并通过 ϵ‑邻居图与 Louvain 社区检测提取离散细胞类型并展示其时间动态，这些都是此前缺乏系统描述的创新。

**🔧 技术方法**

使用了 GNCA 共享神经网络更新规则、Adam 优化、L2 归一化、余弦距离度量、最大似然内在维度估计、平滑度测量、ϵ‑邻居图构建以及 Louvain 社区检测。

**📊 数据集**

实验基于 Google Noto Emoji 数据集中的三张表情符号图像（U+1F98E、U+1F603、U+1F578），尺寸 40×40 像素，加 padding 后映射到 72×72 网格。

**📈 对比分析**

通过训练目标 MSE 损失对齐目标图像，对同步 (UR=1.0) 与异步 (UR=0.5) 两种更新率进行比较；在所有模型中 MSE 最终收敛到低值，异步更新保留更多多样性（总方差峰值更宽、内在维度更低）。虽然未给出传统任务指标，但实验表明模型能够成功重现目标形态并保持低损失。

**⚠️ 局限性**

局限性包括：仅使用单一 GNCA 架构和三张简单表情图像，缺乏对更复杂形态的验证；同步与异步更新差异样本量有限；ϵ 阈值的选择影响社区检测结果，且未探索更鲁棒的密度方法；社区检测对时间与空间的动态捕捉仍有不足；未深入探究训练过程与发育动力学的因果关系以及后期自修复机制与发育的关联。

---

## 177. From Skill Extraction to Multistakeholder Recommendation: A Two-Stage Framework for Bias Governance in Skills-Based Job Matching

**arXiv ID:** 2607.15707 | [PDF](https://arxiv.org/pdf/2607.15707v1)

**作者:** Andrea Forster `[一作]` (Know Center Research GmbH), Simone Kopeinik `[通讯]` (Know Center Research GmbH)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个两阶段框架，用于检测并治理基于 AI 的招聘平台中的偏见，涵盖技能提取与档案形成以及多方利益相关者的职位匹配与推荐；

**💡 创新点**

创新点在于将硬约束与软约束统一贯穿两个阶段，采用多代理社会选择聚合来显式对齐候选人、公司与监管方的公平目标，并借助 Fraunhofer AI 评估目录实现结构化合规性评估；

**🔧 技术方法**

使用的技术包括基于聊天机器人（LLM）的对话式技能抽取、分布式审计与反事实测试的偏见检测、基于代理的多方推荐模型、社会选择理论下的投票聚合与公平性阈值自适应调节；

**📊 数据集**

所使用的数据集主要包括 ESCo 技能分类体系、历史劳动力市场工作分配数据以及通过聊天机器人收集的候选人对话样本；

**📈 对比分析**

方法的比较和性能评估尚未在实验中完成，文中仅说明将通过离线指标（如 nDCG、Gini、KL 散度）和在线用户研究来评估推荐准确性与公平性；

**⚠️ 局限性**

局限性在于框架仍为概念性设计，缺乏完整实现与实证验证，且仅聚焦候选人聊天机器人抽取路径，未覆盖手工录入、简历解析与企业岗位发布；参数阈值、代理权重与投票规则的选择依赖经验并需要进一步实验验证。

---

## 178. A Statistical Formulation Gap for Nonlinear Multiscale Physics-Informed Learning

**arXiv ID:** 2607.15702 | [PDF](https://arxiv.org/pdf/2607.15702v1)

**作者:** Ronald Katende `[一作]` `[通讯]` (Kabale University), Ronald Katende (Kabale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了物理信息学习中非线性多尺度椭圆方程的有限样本形式差距，推导了边界兼容变分神经求解器的有限宽度、有限样本和有限迭代误差界限。

**💡 创新点**

提出了一个明确的界限，区分了非线性多尺度物理信息学习中固有的困难部分和所选PDE形式的伪影，强调了强残差和变分损失的不同特性。

**🔧 技术方法**

使用了变分神经求解器和经验Rademacher复杂度等技术。

**📊 数据集**

构造了一个一维周期扩散方程的最小障碍见证，并进行了数值评估，使用了不同的样本和参数。

**📈 对比分析**

通过与现有方法的比较，证明了强残差和平方强损失的下界，显示出变分能量的复杂度在每个空间维度中保持一致，性能表现出强残差的复杂度增长为N^-1/2。

**⚠️ 局限性**

限制在于变分形式未能消除单独的多尺度近似问题，且需要进一步的规模统一近似定理来处理多尺度神经或算子架构。

---

## 179. GoStop: Reinforcement Learning for Adaptive Temporal Aggregation in Event-Based Feature Tracking

**arXiv ID:** 2607.15699 | [PDF](https://arxiv.org/pdf/2607.15699v1)

**作者:** Youngho Kim `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用强化学习自适应控制事件摄像机特征跟踪中的事件聚合时间窗口，实现在线、低延迟的特征跟踪。

**💡 创新点**

创新点包括：①首次将RL框架应用于事件特征跟踪的时间窗口控制；②引入动态事件基准DEFT数据集以评估鲁棒性；③使用特权信息提升Critic的价值估计；④设计轻量级策略网络，兼容现有跟踪器；⑤通过策略学习实现对多变运动动态的自适应处理。

**🔧 技术方法**

使用技术：PPO强化学习、事件表示卷积编码、双网络（policy+critic）、自定义奖励函数、特权信息、轻量化MLP、事件摄像机原始数据处理、可插拔接口。

**📊 数据集**

使用数据集：DEFT（动态事件跟踪）、EC、EDS、以及用于RL训练的MultiTrack。

**📈 对比分析**

与BlinkTrack、Deep‑EV‑Tracker、HASTE、Kalman Filter等基线对比。实验表明，在DEFT上FA提升约0.38、EFA提升约0.37；在EC/EDS上同样取得显著提升；实时率约188 FPS，计算开销仅增加约0.4 ms，保持在线性能。

**⚠️ 局限性**

局限性：①仅对预训练跟踪器做策略微调，无法直接提升跟踪模型本身；②在极端遮挡或噪声极高的场景下仍可能失效；③对全局运动建模缺乏，主要聚焦点跟踪；④RL训练耗时较大（约12小时），需较大GPU资源。

---

## 180. Hierarchical Specialised Ensembles for Classification of Zebrafish Phenotypes Using the Selected Image Recognition Methods

**arXiv ID:** 2607.15698 | [PDF](https://arxiv.org/pdf/2607.15698v1)

**作者:** Piotr S. Maciąg `[一作]` (Warsaw University of Technology), Magdalena Majdan `[通讯]` (Medical University of Warsaw)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并评估了三种分层专门化集成方案，用于斑马鱼胚胎图像的表型分类。

**💡 创新点**

创新点在于首次引入第二阶段两专门化多标签分类器的分层集成，并验证ConvNeXt在该任务中的卓越性能。

**🔧 技术方法**

使用ResNet18、Vision Transformer（ViT）和ConvNeXt作为骨干网络，构建多标签或二分类器并通过阈值调优实现二阶段推理。

**📊 数据集**

采用Jeanray等人公开的斑马鱼胚胎多标签图像数据集，共包含11种表型，划分为训练、验证和测试集。

**📈 对比分析**

通过在三种骨干与三种集成方案下计算精度、召回率、F1分数和准确率进行比较，结果显示ConvNeXt+Setup2在F1最高，ConvNeXt在准确率上最优。

**⚠️ 局限性**

局限性包括阈值调优的人工性、数据集规模有限、部分表型标注不一致以及未加入定量形态学特征。

---

## 181. IMBench: A Benchmark for Intuitive Robotic Manipulation

**arXiv ID:** 2607.15641 | [PDF](https://arxiv.org/pdf/2607.15641v1)

**作者:** Anurag Maurya `[一作]` (Manav Robotics), Devesh K. Jha `[通讯]` (Manav Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了一个评估直觉操纵能力的基准IMBench，包含35个基于robosuite的仿真机器人操作任务、约14K筛选过的演示轨迹以及可生成多样化场景的工具；

**💡 创新点**

提出直觉操纵的“理解–推理–行动”三阶段框架，并将该框架嵌入基准设计，首次系统性评估物理推理与动作执行的整合；

**🔧 技术方法**

采用大规模视觉语言模型（GPT‑5.5、Claude 等）进行高层推理，使用闭环 ReAct 结构将推理与执行结合，评估视觉‑语言‑动作模型（VLA）如 Diffusion Policy、π_0.5、GR00T；

**📊 数据集**

使用 35 个仿真任务、约 14K 经过三阶段过滤的演示轨迹，包含多视角 RGB 图像、关节信息、抓取器状态以及力/扭矩传感器数据；

**📈 对比分析**

在三阶段评估中：约束理解阶段 VLM 准确率约 70%–80%，但计划正确率下降至 40%–60%，执行成功率仅 10%–20%；VLA 在零样本时几乎 0% 成功，微调后略有提升，但整体仍低；OOD 变异导致性能骤降，显示缺乏稳健的直觉操纵能力；

**⚠️ 局限性**

仅在仿真环境、FrankA 并行手指/吸盘抓取器下测试，未包含多指手、移动平台或人形机器人；任务维度受限于低至中期；未考虑柔性物体；未使用触觉/力反馈；评估聚焦于通用模型而非专用工程化方案。

---

## 182. Knowledge-Centric Agents for Workflow Generation

**arXiv ID:** 2607.15845 | [PDF](https://arxiv.org/pdf/2607.15845v1)

**作者:** Zhendong Li `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Jinjin Gu `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了知识逆转、注入与推理框架，利用多层次知识表示自动生成并执行 ComfyUI 工作流；

**💡 创新点**

将工作流生成视作知识推理问题，提出多层次知识逆转（从工作流到伪代码、骨架、策略再到任务指令）与知识注入（通过监督微调实现任务→策略、策略→骨架），并通过自我细化提升结构一致性；

**🔧 技术方法**

采用 Qwen3‑14B 进行 LoRA 微调、规则化知识逆转、策略规划与伪代码生成、Self‑Refinement 迭代，以及 GPT‑5 用于评估与检验；

**📊 数据集**

收集并清洗约 912 条高质量 ComfyUI 工作流（训练 882 条，测试 30 条），并在 FlowBench、ComfyBench 两大外部基准上进行评测；

**📈 对比分析**

与 Zero‑Shot、Few‑Shot、CoT、RAG、ComfyAgent 等基线在 VND、NodeComp、LinkComp、TaskCons、Pass/Resolve 等指标上对比，结果显著优于所有基线，Pass 率达 86.9% 以上；

**⚠️ 局限性**

在稀有节点类型的生成上表现不足，往往被常见节点替代导致功能失效；需要扩大工作流多样性并结合强化学习提升鲁棒性。

---

## 183. Data-Native Global Optimization for Big Data K-means Clustering

**arXiv ID:** 2607.15835 | [PDF](https://arxiv.org/pdf/2607.15835v1)

**作者:** Ravil Mussabayev `[一作]` (Satbayev University), Kuldeyev Nursultan `[通讯]` (Satbayev University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 Big‑means++ 的大数据聚类算法，通过在随机样本上迭代局部 K‑means 并将质心状态在不同样本间流动，实现对最小平方和聚类（MSSC）问题的全局搜索。

**💡 创新点**

创新点包括：
- 流动主导（flowing incumbent）策略，使质心状态在每一次采样后无条件更新，从而增强搜索移动性；
- 采用几何级数的样本规模梯度（sample‑size ladder）对样本尺寸进行动态震荡，提供多分辨率的代理景观；
- 竞争多代理（multi‑agent）架构，在不通信的情况下并行探索不同样本轨迹，并在统一检验样本上进行最终竞争；
- 基于探针（probe）检测的自适应收敛机制，自动在搜索变得无效前终止各代理，从而实现统一的速度‑质量控制；
- 全程仅使用采样而不进行完整数据遍历，保持了高效性。

**🔧 技术方法**

主要技术包括：
- 随机采样与局部 K‑means（使用 Hamerly 速率加速）；
- 流动主导更新、样本规模梯度和多代理竞争；
- 统一检验样本与探针收敛检测；
- C++ 实现与 OpenMP 并行；
- 通过统计检验（Friedman、Wilcoxon）评估性能。

**📊 数据集**

使用 22 个公开数据集，覆盖从数千点/几百维到千万点/数千维的多种规模，数据集包括 CORD‑19 Embeddings、HEPMASS、US Census 1990、Gisette 等。

**📈 对比分析**

与 11 种竞争算法（如 MiniBatch‑KMeans、BDCSM、Clust‑Splitter、Big‑Clust、LMBM‑Clust、MDEClust、DRS‑means 等）以及 Big‑means 系列变体在 176 个基准实例（dataset×k）上进行比较。实验显示：
- Big‑means++ 在平均相对误差、0.1%/1%/5% 成功率以及运行时间上均优于大多数对手；
- 与传统基准相比，速度显著提升；
- 与重度全局搜索方法相比，质量相近甚至更优；
- Fast 变体在保留 1% 目标下可进一步加速。

**⚠️ 局限性**

局限性：
- 仍为启发式方法，缺乏全局最优性证明；
- 评估基准使用的是“已知最佳”结果，可能并非真正全局最优；
- 仅针对欧氏 MSSC 并假设 k 已知；
- 对非数值或非欧氏距离的适用性尚未验证。

---

## 184. In-context learning of closed form solution to simple linear regression task using transformer with linear self-attention

**arXiv ID:** 2607.15819 | [PDF](https://arxiv.org/pdf/2607.15819v1)

**作者:** Katsuyuki Hagiwara `[一作]` `[通讯]` (Mie University), Katsuyuki Hagiwara (Mie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构造了一种使用线性自注意力的Transformer，并通过层归一化近似实现最小二乘闭式解的自适应推理；

**💡 创新点**

创新点在于将层归一化视为除法操作，用以逼近闭式解而非传统的梯度下降步骤；

**🔧 技术方法**

使用技术包括线性自注意力、层归一化、跳跃连接以及L1正则化的参数稀疏化；

**📊 数据集**

实验数据为从标准正态分布采样的合成线性回归数据（5000个训练样本，1000个验证样本，51维输入序列）；

**📈 对比分析**

与传统梯度下降实现相比，本文方法在测试误差上表现相当，平均平方误差约为0.0007-0.0014，且与最小二乘解高度相关；

**⚠️ 局限性**

局限性包括需大样本量和大R参数才能保证近似准确，且存在多种实现方式导致输出与理论闭式解不完全一致，推理不够鲁棒。

---

## 185. Graph Coloring Approach to Solving Sudoku with Oscillatory Neural Networks

**arXiv ID:** 2607.15814 | [PDF](https://arxiv.org/pdf/2607.15814v1)

**作者:** Filip Sabo `[一作]` (Eindhoven University of Technology), Aida Todri-Sanial `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对数独问题设计了基于图着色的振荡神经网络（ONN）求解器，并通过额外的约束项来阻止违反数独规则的稳定状态。

**💡 创新点**

创新点在于将图着色映射到ONN的简化ODE，并加入额外的g(ϕ_i,ϕ_j)项以强制满足行、列、宫的唯一性约束，从而显著提升准确率。

**🔧 技术方法**

使用了Kuramoto型振荡器、Ising模型映射、SHIL、Max‑K‑Cut的ONN公式以及自定义的g函数，并在Python环境中实现。

**📊 数据集**

使用Py‑Sudoku库生成的随机4×4和9×9数独样本，训练集250例，测试集1000例，分别在不同未知比例下进行评估。

**📈 对比分析**

与Hopfield网络（HNN）数独求解器以及已有的常规ONN求解器对比，采用准确率和阶数参数两指标；4×4数独几乎完美，9×9在低未知比例下达80%以上，高未知比例则性能下降。

**⚠️ 局限性**

局限在于需要四个可调参数（w_u、K_S、K_G、σ），调参成本高；在9×9大未知比例时准确率显著下降，可能是因为g函数权重不够或振荡周期不足。

---

## 186. Examining the Associations between Visual and Non-Visual Elements and Cyclists' Route Choices for Various Trip Purposes

**arXiv ID:** 2607.15808 | [PDF](https://arxiv.org/pdf/2607.15808v1)

**作者:** Heyang Hua `[一作]` (National University of Singapore), Filip Biljecki `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用骑行者GPS轨迹与街景图像相结合的方法，研究不同出行目的下骑行者对城市景观特征（视觉与非视觉因素）的偏好，并比较实际路径与最短路径的差异。

**💡 创新点**

创新点在于首次将视觉感知（通过Street View图像语义分割得到的道路、绿化、机动车占比等指标）与非视觉社会经济因素共同纳入分析，并在多目的出行场景下系统评估路径选择差异。

**🔧 技术方法**

技术主要包括：1）基于Google Street View API采集沿行路径的全景图像；2）使用Mask2Former深度学习模型进行语义分割，提取道路、绿化、机动车等像素比例；3）统计学方法（相关性、ANOVA、多元逻辑回归）对非视觉与视觉因素进行单因素与多因素建模。

**📊 数据集**

数据集包含：①蒙特利尔市“MyVeVelo”骑行轨迹（2013-2015，含目的地分类）；②街景图像；②加拿大OpenStreetMap道路网络；③加拿大统计局人口、收入、住房价格等社会经济数据；③加拿大气象局温度数据；④OpenStreetMap POI（办公、餐饮、娱乐等）。

**📈 对比分析**

方法比较：建立两组多元模型——仅非视觉因素模型与同时包含视觉因素模型。通过AIC、BIC和伪R²指标对比，非视觉模型在拟合度与复杂度平衡上略优（AIC≈1883 vs 3970，BIC≈2387 vs 4877），但两者伪R²均约为0.22，表明两类因素对出行目的的解释力相近。

**⚠️ 局限性**

局限性包括：①轨迹数据已预处理，可能存在起止点模糊和间断；②研究仅限蒙特利尔，缺乏跨城市验证；③街景图像从车辆视角获取，难以完全代表骑行者视角；④图像分割与数据质量可能受天气、季节等影响；⑤模型中未使用更先进的机器学习/正则化方法，可能存在过拟合风险。

---

## 187. Making Agent-Mediated Contributions Governable: A Project-Level Governance Manifest for Open-Source AI Collaboration

**arXiv ID:** 2607.15769 | [PDF](https://arxiv.org/pdf/2607.15769v1)

**作者:** Jinjin Gao `[一作]` (Shanxi University of Finance and Economics), Xiaoning Sun `[通讯]` (Shanxi University of Finance and Economics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过诊断、设计与评估三阶段，提出并实现了“Agent Governance Manifest (AGM)”——一种仓库托管的治理规范，用以在生成‑验证失衡的开源软件（OSS）贡献工作中，将项目层面的风险分类、证据义务、责任声明与审核门控等治理要素预先外化，帮助贡献者准备审查可检验的证据包，并让维护者在审查时能快速恢复并评估这些治理状态，最终维持维护者的最终决策权。

**💡 创新点**

创新点包括：①提出“项目侧治理能力（project‑side governability infrastructure）”概念，将治理任务从“可读性”“可追踪性”提升到“可治理性”，强调在贡献前期就将治理规则绑定到贡献对象上；②设计AGM作为可被人机共同解释的规范化边界资源，实现贡献准备与维护审查之间的双向治理合同；③构建三层治理框架（agent‑readability、traceability、governability）并在实验中验证其有效性。

**🔧 技术方法**

主要技术手段：设计科学方法论；对50个GitHub仓库进行治理审计（文本、PR、Issue、工作流等）；基于AGM进行结构化证据包与审查包装的实现；使用Python脚本与GitHub API进行数据爬取与处理；采用基于任务的受控实验（reviewer‑side和contributor‑side），对输出进行客观编码与主观问卷；统计分析使用bootstrap、kappa等指标。

**📊 数据集**

数据集：①审计样本——50个公开GitHub仓库（覆盖不同风险、AI关注度、治理主体与生态角色）；②贡献记录——23,237条PR、19,884条Issue及其文件级变更；③实验任务与材料——自建任务集与AGM支持材料；实验受试者——15名技术人员（分别在reviewer与contributor实验）。

**📈 对比分析**

对比方法：在受控实验中将AGM支持材料与普通材料交替呈现，记录reviewer‑side输出的风险标签恢复率、门控状态、缺失证据识别等；在contributor‑side实验中验证证据包结构完整性与治理状态一致性。性能结果：AGM支持下风险标签准确率97%（vs. 40%普通），门控与责任识别率提升显著，受试者主观评分在1–7量表上提升约2.9分，contributor‑side结构验证通过率达到91%。

**⚠️ 局限性**

局限性：①实验样本规模与范围有限，无法覆盖所有OSS项目类型；②实验环境为受控，真实维护者的工作节奏与复杂性可能不同；③依赖公开AI使用标注，未能捕捉隐性或未披露的AI贡献；④AGM实现为原型，未在长期真实项目中评估可持续性与可维护性；⑤对AI工具的交互细节（如多轮对话、提示）未深入探讨。

---

## 188. GapForge: Directed Compiler Fuzzing via Coverage-Gap Analysis

**arXiv ID:** 2607.15762 | [PDF](https://arxiv.org/pdf/2607.15762v1)

**作者:** Mingxuan Zhu `[一作]` (Peking University), Dan Hao `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的编译器测试技术GapForge，通过覆盖缺口分析有针对性地生成测试程序，显著提升GCC和LLVM核心模块的行级覆盖率并发现多项真实缺陷。

**💡 创新点**

创新点：①将覆盖缺口视为显式目标，采用覆盖驱动的文件优先级评分与概率变换实现探索与利用的平衡；②使用“覆盖上下文+未覆盖片段”级别的细粒度总结，推断触发特定未覆盖代码所需的程序结构和编译选项；③融合先前失败的提示信息进行反思式提示合成，持续引导LLM生成更高效的测试程序。

**🔧 技术方法**

核心技术：1）覆盖驱动目标选择（基于文件大小和覆盖率的非线性评分）；2）细粒度覆盖缺口总结（利用LLM进行路径差异分析和编译选项推理）；3）提示合成与失败反思；4）LLM推理与代码生成（GPT‑4o用于总结，StarCoder用于程序合成），可替换为其他LLM。

**📊 数据集**

评测数据集：GCC 14.3.0和LLVM 19.1.0的核心前端/后端源文件，覆盖率信息来自gocov-13；实验时间预算72小时；对比的八个基线包括Csmith、DST、Creal、GrayC、Fuzz4All、WhiteFox、LegoFuzz、Optimuzz。

**📈 对比分析**

对比方法：在相同硬件（Intel Xeon Gold 6430 + NVIDIA RTX 4090）和相同时间预算下，与八个基线进行行级覆盖率、增量覆盖、测试程序数量和LLM令牌消耗对比。GapForge在GCC达到68.13%覆盖率（+2.89%增量，+3,452行），LLVM达到69.11%（+1.16%增量，+531行），均显著优于最强基线LegoFuzz（≈64%）和WhiteFox（≈63%）。此外GapForge发现5个GCC、7个LLVM的真实错误，覆盖率提升与故障发现都处于领先水平。

**⚠️ 局限性**

局限性：①依赖LLM，API成本与模型可用性限制；②实验仅在x86平台与两大编译器上验证，难以直接推广至其他架构或编译器；③使用行级覆盖率作为唯一衡量，未评估分支/路径级别或功能性覆盖；④对硬件加速（GPU）与传统CPU工具的比较存在偏差；⑤对极长尾文件的覆盖提升仍有限，需要进一步优化文件选择与提示策略。

---

## 189. AuEmoChat: Authentic Emotion Understanding and Rendering for Conversational Speech Synthesis

**arXiv ID:** 2607.15755 | [PDF](https://arxiv.org/pdf/2607.15755v1)

**作者:** Zhenqi Jia `[一作]` (Inner Mongolia University), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 AuEmoChat 框架，结合 AuEmoCodec、AuEmoToMe 与 Authentic Emotion Flow Matching，实现对话语音中情感的真实理解与表达。

**💡 创新点**

创新点包括：① 构建离散化的真实情感词表；② 采用情感引导的 token 合并技术减少冗余；③ 在流匹配中加入情感引导器，提升情感一致性。

**🔧 技术方法**

技术手段包括 Finite Scalar Quantization（FSQ）构建情感词表、AuEmoToMe（情感驱动 token 合并）、自回归 LLM 生成目标情感与语音 token、条件流匹配（flow matching）并使用情感分类器引导、HiFi‑GAN 语音解码。

**📊 数据集**

使用 NCSSD‑EmCap 数据集（集成 DailyTalk、NCSSD、MultiDialog，约 384 小时、18,580 对话）进行训练与评估。

**📈 对比分析**

与 BaseCSS、ECSS、GPT‑Talker、Chain‑Talker 四个 SOTA 基线对比，主观 DMOS、客观 WER/MCD/SpkSIM 以及情感准确率均显著提升，达到最高分数。

**⚠️ 局限性**

局限性在于仅在英文单语环境下验证；AuEmoCodec 训练规模有限；未对情感可解释性进行深入分析。

---

## 190. Ciphertext- and Polynomial-Level Optimization for Fully Homomorphic Encryption

**arXiv ID:** 2607.15750 | [PDF](https://arxiv.org/pdf/2607.15750v1)

**作者:** Seongho Kim `[一作]`, Yongwoo Lee `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了一款多层 FHE 编译器 Recifhe，能够先在 ciphertext 级做全局优化，再降到 polynomial 级进行性能感知的 hoisting、融合与调度，从而自动生成 RNS-CKKS 程序并显著减少冗余多项式计算。

**💡 创新点**

创新点包括：①采用多层 IR，先在 ciphertext 级做全局优化，再在 polynomial 级做细粒度优化；②通过离线 latency profiling 做性能感知 hoisting，只在收益>开销时才进行；③引入 liveness‑driven polynomial 调度，减少内存占用。

**🔧 技术方法**

使用了 MLIR/Hecate 框架、Python DSL 前端、NVIDIA RTX PRO 6000 GPU 后端，RNS‑CKKS 加密方案，NTT/iNTT 基础变换，以及性能感知 hoisting、common subexpression elimination、operation fusion、调度等技术。

**📊 数据集**

在典型的 FHE 机器学习基准上进行评估，包括 ResNet、MobileNet、Polynomial Regression、以及自定义矩阵乘/旋转等基准。

**📈 对比分析**

与 ciphertext‑级仅优化基线及手动多项式优化 baseline 在相同前端后端与 RNS‑CKKS 参数下对比。Recifhe 在所有基准上均快于 ciphertext‑级基线，平均 1.25× 加速；对比手动 baseline 则在完成的基准上更快。内存占用与 ciphertext‑级基线持平，显著低于手动 baseline；编译时间平均增长约 2.9×。

**⚠️ 局限性**

主要限制是编译时间相对较长，polynomial 级优化需要离线 profiling；目前仅在 RNS‑CKKS 上实验，未验证其他 FHE 方案；对极大规模程序以及多 GPU/多机环境的可扩展性尚未评估。

---

## 191. Towards Artificial Nerves: Biomimetic Optical-Fiber Tactile Sensing for Robots

**arXiv ID:** 2607.15746 | [PDF](https://arxiv.org/pdf/2607.15746v1)

**作者:** Laura E. Butcher `[一作]` (University of Bristol), Efi Psomopoulou `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为OptiTac的光纤触觉传感器，模仿人类触觉神经通路，实现光纤远程传输触觉信号，并通过图像矩方法解析接触位置、尺寸和形状。

**💡 创新点**

创新点在于将一对一的指针‑光纤配对与光纤阵列相结合，形成生物仿真的预处理结构，既保留高空间分辨率，又可用解析方法（非深度学习）实现触觉信息的可解释推理。

**🔧 技术方法**

采用TacTip软皮肤、光纤阵列、光学预处理、图像矩与Hu矩、Gaussian Mixture Model（GMM）以及校准曲面等技术。

**📊 数据集**

实验使用ABB机器人对不同尺寸（1.85–20.09 mm）圆形、平面多边形和边缘压头进行压迫，收集光纤图像数据；未使用公开的大规模触觉数据集。

**📈 对比分析**

与Baimukashev等的同类光纤传感器和Lu等的深度学习方法相比，OptiTac在接触中心定位RMSE约0.4 mm、宽度测量RMSE≤1.2 mm、形状分类准确率≥90%，实现超分辨率且无需深度学习，性能优越。

**⚠️ 局限性**

局限性包括光纤长度和光混合导致信息损失、校准仅针对正对光纤位置、在更大范围和更复杂环境下的鲁棒性需进一步验证。

---

## 192. NP-Hardness of Connected Components Reconfiguration under Component Jumping on Caterpillar Graphs

**arXiv ID:** 2607.15737 | [PDF](https://arxiv.org/pdf/2607.15737v1)

**作者:** Naoki Kitamura `[一作]` (University of Osaka), Taisuke Izumi `[通讯]` (University of Osaka)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了连通分量重组问题在组件跳跃模型下的多重集版本，证明了其在养猫形图上为NP‑难，并在路径图上给出O(nlogn)的决策算法以及在有足够空闲空间时可输出长度为O(nlogn)的重组序列。

**💡 创新点**

提出了从3‑正则图的独立集问题构造的复杂度证明，改进了路径图决策算法的时间复杂度，并给出了一个递归分区策略实现高效重组序列。

**🔧 技术方法**

采用了组合构造、归约、段树查询以及递归分治算法等技术。

**📊 数据集**

论文为理论工作，没有使用具体实验数据集。

**📈 对比分析**

与之前的O(n²)算法相比，新的算法在时间复杂度上提升到O(nlogn)，且在空闲空间足够的情况下可输出最优长度重组序列。

**⚠️ 局限性**

仅在路径图和养猫形图上得到结果，未给出更一般图形上的多项式算法，且对NP‑难性的证明依赖于特定图形构造。

---

## 193. The Third Competition on Document Forgery Detection on ID-Cards and Passports

**arXiv ID:** 2607.15734 | [PDF](https://arxiv.org/pdf/2607.15734v1)

**作者:** Juan E. Tapia `[一作]` (Hochschule Darmstadt), Christoph Busch `[通讯]` (Hochschule Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文系统性评估了第三届ID卡与护照伪造检测竞赛的结果，并对参与模型进行深入对比。

**💡 创新点**

创新点在于引入了跨文档（身份证与护照）通用的合成训练数据，采用多模态双流与多标签分类框架，并提出了基于AV_Rank的综合排名指标。

**🔧 技术方法**

主要技术包括Vision Transformer（DINOv2/ViT）、ConvNeXt、Swin-Transformer、YOLO/YOLOv11n进行文档定位与预处理，以及多标签二分类与TTA/数据增强。

**📊 数据集**

使用的数据集为竞赛官方共享的合成ID卡与护照数据集（Track 1）以及公开与内部真实捕获的大规模身份证/护照数据集（Track 2）。

**📈 对比分析**

通过EER、BPCER10/20/100等ISO/IEC 30107-3标准指标进行对比，Incode队伍在Track 1获得EER 8.42%、BPCER10 6.01%，AV_Rank 27.82%，在Track 2获得EER 26.52%、BPCER10 56.53%、BPCER100 75.06%，AV_Rank 68.71%，整体性能仍低于行业期望。

**⚠️ 局限性**

主要限制包括：复合攻击仍表现最差、模型规模与推理复杂度高、在高阈值（BPCER100）下性能不佳，且对不同文档类型的泛化仍有限。

---

## 194. Is That Really My X-Ray? Measuring Internet-Exposed DICOM Services in the Presence of Deception

**arXiv ID:** 2607.15839 | [PDF](https://arxiv.org/pdf/2607.15839v1)

**作者:** Ricardo Yaben `[一作]` (Technical University of Denmark), Emmanouil Vasilomanolakis `[通讯]` (Technical University of Denmark)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文结合主动IPv4全网扫描与被动嗅探，提出可重复的噪声过滤方法，识别蜜罐、望远镜等伪影，随后对被过滤后的DICOM服务进行安全评估，量化曝光率、漏洞分布，并在披露后多时点跟踪其状态变化。

**💡 创新点**

创新点包括：①首次系统化地将噪声过滤与多时点扫描相结合，显著降低误报（39%）；②首次对DICOM Honeypot日志进行噪声消除并揭示侦测盲点；③将主动曝光评估与被动攻击行为分析结合，提供完整的安全景观。

**🔧 技术方法**

技术手段包括：ZMap + ZGrab2主动扫描、专用DICOM探针（限制为A-ASSOCIATE/C-ECHO）、基于实现UID/版本的聚类与Fingerprinting、主动披露流程、Python/Notebook数据分析、Dicompot Honeypot部署与日志过滤脚本。

**📊 数据集**

数据集涵盖：①主动扫描原始数据（ZMap/ZGrab2结果）; ②IPinfo/WHOIS补充的组织与域名信息; ③公开CVE数据库与DICOM工具实现列表; ④Dicompot日志与对应PCAP；所有数据均已脱敏或受限公开。

**📈 对比分析**

通过三轮扫描（S1、S2、S3）与披露前后对比，发现曝光率从6,646台降至3,979台；漏洞覆盖率达到1,551台；被动日志经过滤后，C-ECHO占多数，C-FIND/ C-GET占极少，表明攻击上限低；整体性能显示噪声过滤显著提升评估准确性。

**⚠️ 局限性**

局限性包括：仅使用单一扫描视角，未覆盖非标准端口与Web门户；Dicompot缺乏深度协议仿真，导致高级攻击行为被低估；披露后响应率低，缺乏自动化修复与持续监测；实验仅覆盖有限时间，无法完整捕捉长期演化。

---

## 195. Cost-efficient generative AI summarization for scalable automated essay scoring in educational assessment

**arXiv ID:** 2607.15829 | [PDF](https://arxiv.org/pdf/2607.15829v1)

**作者:** Haowei Hua `[一作]` `[通讯]` (Princeton University), Haowei Hua (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出了一种基于生成式AI的摘要预处理框架，将长篇学生作文压缩为固定长度摘要，再结合手工特征与嵌入进行AES。

**💡 创新点**

创新点在于将GPT-5系列模型作为摘要器融入传统AES流水线，形成半监督的混合表示，并系统评估摘要质量、得分可靠性与成本的权衡。

**🔧 技术方法**

使用技术包括GPT‑5、GPT‑5 mini、GPT‑5 nano的自适应摘要、Qwen3‑Embedding‑4B语义嵌入、22项手工语言特征、XGBoost/LightGBM梯度提升分类器。

**📊 数据集**

实验数据集为Kaggle上的Learning Agency Lab – Automated Essay Scoring 2.0（约24,000篇作文，评分1–6分）。

**📈 对比分析**

通过QWK比较三种GPT摘要器，GPT‑5 mini在QWK 0.8435上略优；在ROUGE、语义相似度、实体/关键词覆盖等摘要质量指标上，GPT‑5表现最好，成本最高；GPT‑5 mini在性能与成本上取得最佳平衡。

**⚠️ 局限性**

局限性包括仅使用单一数据集、单一嵌入模型与分类器；未对截断、Longformer、直接LLM评分等基线进行对比；提示策略单一；结果对不同写作题材或高阶任务的泛化尚未验证。

---

## 196. Can't Stop: How Context and Individual Traits Influence Effectiveness of Different Gradual Interventions for Infinite Scrolling on Short-Form Video Platforms

**arXiv ID:** 2607.15818 | [PDF](https://arxiv.org/pdf/2607.15818v1)

**作者:** Luca-Maxim Meinhardt `[一作]` (Ulm University), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在短视频平台上进行为期七天的实地实验，比较传统弹窗、渐进式视觉干预和渐进式触觉干预对无限滚动行为的干预效果。

**💡 创新点**

首次将个体自我调节特质与上下文因素与干预类型相互作用的研究框架引入，发现自我控制与冲动性显著调节干预效果，而上下文因素作用有限。

**🔧 技术方法**

使用基于Android可访问服务的自定义追踪应用，结合事件采样法（ESM）收集行为与主观体验，采用贝叶斯混合效应模型进行统计分析。

**📊 数据集**

参与者在美国和英国共104人，使用TikTok、Instagram、Facebook和YouTube Shorts进行实验，数据通过自适应App记录并上传。

**📈 对比分析**

通过生存分析和贝叶斯混合模型比较三种干预的客观停用时长和主观满意度，结果显示弹窗最快停用但主观影响迅速衰减，视觉渐进干预主观效果保持时间最长。

**⚠️ 局限性**

局限包括仅在Android设备上研究、干预持续时间短、主观测量仅在停止滚动后收集、未检验长期适应性以及可能的自我选择偏差。

---

## 197. HybridSim: A Physics-Learning Hybrid Digital Twin for mmWave Human Sensing

**arXiv ID:** 2607.15806 | [PDF](https://arxiv.org/pdf/2607.15806v1)

**作者:** Weitao Xiong `[一作]` (Xiamen University Malaysia), Hongfei Xue `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种混合物理与学习的 mmWave 雷达仿真器 HybridSim，用于从动态人体网格合成高保真 Range‑Doppler 热图

**💡 创新点**

将雷达传播分为直接路径（逆渲染+微面 BRDF）和间接路径（3D Gaussian Splatting + 虚拟接收器），实现高效且精确的多路径建模

**🔧 技术方法**

逆渲染微面 BRDF、图卷积网络、三平面表示、3D Gaussian Splatting、代理虚拟接收器、可微 RF 渲染、噪声注入等技术

**📊 数据集**

mmMesh 数据集（TI AWR1843BOOST mmWave 雷达）并在固定室内环境下收集的动态人体动作序列

**📈 对比分析**

与 mmGPE 和 RF‑Genesis 对比，HybridSim 在 PSNR、SSIM、LPIPS、HAR 精度等指标上分别提升约 2 dB、0.04、0.18 及 37%（准确率）

**⚠️ 局限性**

仅适用于已训练的固定室内场景，环境参数需重新调优，缺乏零样本/快速适应新房间的能力

---

## 198. What Do Chinese-Language Generative Search Engines Cite and Surface? A Large-Scale Empirical Study

**arXiv ID:** 2607.15771 | [PDF](https://arxiv.org/pdf/2607.15771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 199. READU: Inconsistency-Driven Just-in-Time Detection and Repair of README Bugs

**arXiv ID:** 2607.15780 | [PDF](https://arxiv.org/pdf/2607.15780v1)

**作者:** Doehyun Baek `[一作]` (CISPA Helmholtz Center for Information Security), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ReadU的技术，能够在代码提交时即时检测并自动修复仓库级文档（如README）中的错误（称为README bugs）。

**💡 创新点**

创新点在于：① 采用“内部/外部一致性检查”双重视角，利用LLM智能代理同时验证代码、配置与外部依赖的事实；② 结合高召回提交过滤器和判别式警报裁决器，实现高精度的错误定位；③ 通过LLM自动生成可直接合并的文档修补补丁。

**🔧 技术方法**

主要技术包括：大语言模型（DeepSeek V4 Flash）驱动的LLM代理、结构化单调用LLM接口、内部一致性检查器、外部一致性检查器、警报裁决器、自动修复生成器，以及成本/时间监控机制。

**📊 数据集**

使用 6,000 条来自 6 个热门项目（Linux、Spring Boot、TensorFlow、React、AutoGPT、Ollama）的最近提交作为评测数据集。

**📈 对比分析**

与五种基线（DOCER、README‑Auto‑Update、Single LLM、Mini‑SWE‑agent、Codex Review）对比，ReadU 在检测上获得 244 条真正错误（75% 召回率、75% 精确率），修复成功率 89%，每提交平均成本约 $0.03、耗时 <1 分钟，位于 Pareto 前沿。相比基线，覆盖率提升 3–4 倍，精度显著高于任务特定基线，且在成本与时间上保持竞争力。

**⚠️ 局限性**

局限性包括：仅评估热门且活跃项目，可能不适用于小型/被动项目；使用单一开源LLM，未覆盖更强大商业模型；LLM 的非确定性导致实验可复现性有限；外部依赖信息易变，链接失效可能导致误报或漏报。

---

## 200. NeurOWL: An LLM-Based Neural-symbolic Framework for Incomplete OWL Ontology Reasoning

**arXiv ID:** 2607.15776 | [PDF](https://arxiv.org/pdf/2607.15776v1)

**作者:** Hui Yang `[一作]` (University of Manchester), Wen Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向不完整OWL本体的可解释子类推理框架，能够判断给定子类关系在缺失axioms情况下是否合理，并自动生成相应的解释（缺失axioms集合）。

**💡 创新点**

创新点：1）将子类验证与本体归纳（abduction）统一为一个端到端流程；2）不再依赖预定义的假设空间，能够处理正负推理；3）结合大语言模型与本体嵌入，实现候选概念检索与验证；4）支持多阶段推理，提升解释质量。

**🔧 技术方法**

技术：神经符号框架，核心组件包括：①基于OWL reasoner的逻辑检查；②利用本体嵌入（OnT/ SBERT）进行候选桥接概念检索；③大语言模型（Qwen3.5-9B）对候选axioms和子类关系进行验证；④多阶段（Stage2a/b、Stage3a/b）策略实现解释生成。

**📊 数据集**

数据集：FoodOn（食品与农业知识库）和SNOMED CT（医疗专业本体），分别构造标准设置（仅原子概念桥接）和复杂设置（∃r.B桥接）两类数据集，用于评估模型在不同复杂度下的表现。

**📈 对比分析**

对比方法：LLM‑only、OnT、SBERT三种基线。评估指标包括F1、X‑F1（预测正确且解释正确）与X‑F1*（所有预测桥接概念均为真）。实验结果显示：①在硬负样本上，OnT/ SBERT版本的框架F1可达0.86‑0.97，明显优于基线；②在解释准确率上，X‑F1与X‑F1*分别可达0.65‑0.79；③在复杂数据集（SNOMED ∃）上，虽然总体指标下降，但仍优于基线。

**⚠️ 局限性**

局限性：①候选概念空间过大（尤其∃r.B类）导致检索精度下降；②对极其复杂逻辑（∀、¬等）支持有限；③LLM的判断仍可能出现错误，影响最终解释质量；④需要进一步提升对结构与文本信息的融合效果，以降低误检率。

---

## 201. SlotMem: Character-Addressable Internal Memory for Narrative Long Video Generation

**arXiv ID:** 2607.15772 | [PDF](https://arxiv.org/pdf/2607.15772v1)

**作者:** Yilai Liu `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 SlotMem，一个角色可寻址的内部记忆框架，用于提升叙事式长视频生成中角色身份的一致性。

**💡 创新点**

创新点在于通过字符语义探测定位角色相关视觉令牌，将其压缩为角色槽记忆，并在自回归生成中仅将记忆注入对应角色局部区域，同时支持持续更新，显著降低身份与背景、姿态的耦合。

**🔧 技术方法**

采用视频 Diffusion（Wan2.2）结合字符语义探测、内存编码器、内存写入器和角色跨注意力等技术，并使用 LoRA 微调与对比损失提升槽辨别。

**📊 数据集**

使用公开电影视频构建的多角色叙事长视频数据集，并利用 VBench、ViStoryBench、NarraStream-Bench 等基准进行评估。

**📈 对比分析**

与 Wan2.2、StoryDiffusion、StoryMem、IAMFlow 等基线对比，SlotMem 在角色相似度、主体一致性、运动平滑度等指标上均取得领先或同等表现，同时保持视频质量。

**⚠️ 局限性**

局限在于仍需依赖文本角色命名一致性；对复杂背景或多重重叠角色时仍可能产生干扰；内存容量受限时长时间跨度更新效果有限；对极端姿态变化的鲁棒性待进一步提升。

---

## 202. Trainable Spline Representations for Physics-Informed Learning

**arXiv ID:** 2607.15751 | [PDF](https://arxiv.org/pdf/2607.15751v1)

**作者:** Giovanni Canali `[一作]` (SISSA), Gianluigi Rozza `[通讯]` (SISSA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Physics‑Informed Splines (PI‑Splines)，用张量积 B‑spline 控制系数直接表示 PDE 解，并通过残差驱动训练；

**💡 创新点**

创新点在于把神经网络参数替换为可训练的 B‑spline 控制系数，既保留 PINNs 的残差优化，又具备局部支撑、可解析导数、显式光滑度控制以及几何可解释性；

**🔧 技术方法**

技术包括张量积 B‑spline 展开、解析导数、硬性 Dirichlet 边界强制、Adam+LBFGS 优化以及与传统 PINN 相同的损失结构；

**📊 数据集**

使用四个标准 PDE 基准（Poisson、Helmholtz、指数型问题、声波方程）的解析解作为评估数据集；

**📈 对比分析**

与三种基线（普通 PINN、Fourier‑feature PINN、PIKAN）在相同的 PDE、采样点、损失权重和优化策略下比较，PI‑Splines 在所有四个任务上取得最低平均绝对误差，参数量减少 1–2 个数量级，训练时间也保持竞争力；

**⚠️ 局限性**

局限在于对预设的结点向量、阶数和控制点数高度依赖，空间分辨率不足时无法逼近；高阶/高维时的计算成本与基准方法相当或略高，需要更高效的基函数评估或自适应细化。

---

## 203. Better Starts, Better Ends: Bootstrapped Iterative Self-Reasoning Distillation for Compressed Reasoning

**arXiv ID:** 2607.15736 | [PDF](https://arxiv.org/pdf/2607.15736v1)

**作者:** Leichao Dong `[一作]` (Xi'an Jiaotong University), Jihua Zhu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大型语言模型的链式推理压缩中，提出两阶段自我蒸馏方法BIRD，先用轻量级LoRA SFT预热生成简洁推理轨迹，再在此基础上进行on‑policy自蒸馏

**💡 创新点**

创新点在于识别并解决“cold‑start prefix‑support”瓶颈，即在自蒸馏时把教师查询的前缀从冗长、偏离轨道的轨迹转移到更简洁、可靠的轨迹，并通过前向预热而非单纯的教师设计来实现

**🔧 技术方法**

采用LoRA微调、reverse‑KL on‑policy self‑distillation、前缀支持的分布匹配、prompt‑switch SFT等技术

**📊 数据集**

使用DAPO‑Math‑17k‑dedup训练数据，评测在MATH‑500、AIME 2024和AIME 2025三个数学推理基准上

**📈 对比分析**

与基准模型Base、仅推理时使用conciseness instruction、以及冷启动on‑policy自蒸馏CRISP相比，BIRD在9/9模型‑基准组合上实现了更高的Token Efficiency（TE），在大型模型上可同时提升准确率和压缩率

**⚠️ 局限性**

局限性包括：对非数学推理任务的泛化尚未验证；两阶段顺序关键，逆序会导致准确率下降；需要进一步研究如何在更大规模或多任务设置下保持优越性

---

## 204. Deployment-Ready UWB Localization for Industrial Ground Robots with Automatic Anchor Calibration and Terrain-Aware Fusion

**arXiv ID:** 2607.15807 | [PDF](https://arxiv.org/pdf/2607.15807v1)

**作者:** Alexander Raab `[一作]` (AGILOX Services GmbH), Stephan Weiss `[通讯]` (University of Klagenfurt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种端到端的UWB定位管线，包含自动锚点校准和基于地形约束的多传感器 EKF，实现工业 AMR 的室内外精准定位。

**💡 创新点**

将锚点校准与姿态估计分离为两步，利用偏置感知的范围测量模型并将校准不确定性作为 Schmidt 状态融入 M-ESEKF，从而实现无需人工调参的自适应定位。

**🔧 技术方法**

使用单向 TWR UWB 测距、B-spline 地形模型、M-ESEKF、偏置感知测量模型、Schmidt‑Kalman 滤波、非线性最小二乘校准等技术。

**📊 数据集**

在 AGILOX ONE 物流 AMR 的仓库环境中采集 13 条轨迹（室内+室外）以及一套预先校准的叉车数据集，公开仓库数据集供研究使用。

**📈 对比分析**

与仅使用里程计、原始 UWB 测距模型以及其他基准方法比较；改进模型在室内实现 0.13 m ATE、4.8 NEES≈1；室外提升到 0.55 m、NEES≈3；叉车数据实现 1.06 m ATE、0.07 NEES，整体显著优于传统方法。

**⚠️ 局限性**

需要先验位姿、平坦地形模型和平台噪声参数；在强 NLOS、多径、地形误差等极端条件下一致性仍受限，未实现完全在线地形/噪声自适应。

---

## 205. DICOMHawk: A Cyber Deception Framework for Medical Imaging Infrastructure

**arXiv ID:** 2607.15754 | [PDF](https://arxiv.org/pdf/2607.15754v1)

**作者:** Karina Elzer `[一作]` (Technical University Of Denmark), Emmanouil Vasilomanolakis `[通讯]` (Technical University Of Denmark)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并部署了 DICOMHawk 反欺骗框架，模拟 DICOM/PACS 服务，收集并分析了 347 天的攻击日志，并与现有低交互 DICOMhoneypot Dicompot 进行对比。

**💡 创新点**

创新点在于：①实现了完整的 DICOM 与 PACS 双端交互（DIMSE 操作与 Web 查看），②动态拉取 TCIA 医学影像并注入蜜置信息（PDF 诱饵、HoneyURLs），③通过伪装签名与细粒度日志降低被识别率，从而显著提升攻击者参与度。

**🔧 技术方法**

技术包括：Pynetdicom 构建 DICOM 服务、Encapsulated PDF Storage 注入 PDF 诱饵、Honeycredential 与 Honeyrecord 注入、DICOMWeb/ PACS Web 接口、Shodan/Censys 端口扫描、聚类密码字典进行攻击模式识别，以及对日志进行结构化处理。

**📊 数据集**

使用公开的 TCIA 医学影像数据作为蜜置信息来源，并收集 347 天自部署的攻击日志（包括 49 起医学相关攻击），对比 86 天内 Dicompot 与 DICOMHawk 的会话与 DIMSE 命令记录。

**📈 对比分析**

对比方法：在云端与本地环境分别部署 DICOMHawk 与 Dicompot，统计每周会话数、DIMSE 命令数、攻击次数；DICOMHawk 在云端提升 47.52% 日均会话、在本地提升 18.35%，捕获的攻击命令数更多，且在 Shodan 检测中被识别为合法设备，说明其更难被自动化侦测。

**⚠️ 局限性**

局限性：①攻击量总体偏低（仅 4 起 DICOM、8 起 PACS 攻击）；②诱饵（PDF 诱饵、HoneyURLs）未被触发；③部署仅限学术/云环境，未覆盖真实临床网络；④不支持 TLS、DICOMWeb 或 HL7/FHIR 等新协议；⑤日志仅记录可解析的 DIMSE 请求，可能遗漏部分攻击细节。

---

## 206. RTL-Sequencer: Towards Scalable RTL Timing Prediction with the Sequence-based Paradigm

**arXiv ID:** 2607.15830 | [PDF](https://arxiv.org/pdf/2607.15830v1)

**作者:** Ziyan Guo `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 RTL‑Sequencer，通过将 RTL 逻辑锥线性化为逆 BFT 序列并采用序列模型实现可扩展的 RTL 时序预测；

**💡 创新点**

创新点在于四项技术：序列打乱增强、双向序列建模、可微差分嵌入，以及图‑序列混合架构，使模型既保留信号方向性，又能捕获深层依赖，且计算复杂度保持线性；

**🔧 技术方法**

使用的技术包括逆向宽度优先遍历 (BFT)、Mamba‑2 等序列模型、双向序列网络、差分嵌入机制、GNN 与线性投影的图‑序列混合结构；

**📊 数据集**

实验数据集为 21 个开源 RTL 设计，规模从 6k 到 510k gate，使用 Synopsys Design Compiler、Cadence Innovus 与 PrimeTime 生成 ground‑truth；

**📈 对比分析**

与 GCN、GAT、SG‑Former、RTL‑Timer、RTLDistill、NUA‑Timer、CircuitFusion、TF‑Predictor 等基线在 AT、WNS、TNS 上对比，RTL‑Sequencer 在 AT MAPE 17.24%、R=0.92、R²=0.85 方面均优于所有基线，特别是在深度大、节点多的逻辑锥上表现最显著；

**⚠️ 局限性**

局限性包括仍需手工提取 BFT 序列，对极大规模设计的内存需求相对较高，且未整合后期布局信息，导致在某些极端深度/宽度的设计上仍可能出现误差。

---

## 207. On the Geometry of Learned Representations in Event-Based Multi-Modal Egomotion Estimation

**arXiv ID:** 2607.15794 | [PDF](https://arxiv.org/pdf/2607.15794v1)

**作者:** Stefano Silvestrini `[一作]` (Politecnico di Milano), Michele Ceresoli `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并分析了一个自监督多模态网络的潜在空间，以评估其是否隐式学习了经典几何估计中的结构

**💡 创新点**

发现潜在向量在低维流形上与速度变量对齐，注意力权重随角速度和视觉可靠性变化，可作为实时可靠性诊断

**🔧 技术方法**

采用跨模态注意力融合、3D ResNet 视觉编码器、IMU 多层感知机、双向 GRU 范围编码、光流自监督头以及马氏距离、CCA 等潜在空间诊断手段

**📊 数据集**

使用 ELOPE 仿真月球着陆数据集（事件、惯性、测距三模态）

**📈 对比分析**

通过线性探测、AUC 可靠性检测和 CCA 关联等诊断，表明潜在向量对速度预测的 R²>0.99，误差检测 AUC>0.8，证明其结构与经典几何方法相似，虽然未能与最优经典管线直接比较，但在诊断指标上表现优异

**⚠️ 局限性**

仅在仿真分布下验证，缺乏正式的可观测性/稳定性保证，诊断指标不等同于校准不确定性，且对分布偏移、传感器退化等情况的鲁棒性尚待评估

---

## 208. Personalized Image Aesthetic Assessment via Preference-rich Sample Mining and Cohort Merging

**arXiv ID:** 2607.15752 | [PDF](https://arxiv.org/pdf/2607.15752v1)

**作者:** Zhichao Yang `[一作]` (Xidian University), Leida Li `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出PRAC框架，通过多模态大语言模型挖掘图像偏好富样本并聚合审美共鸣群体，完成个性化图像审美评估；

**💡 创新点**

创新点在于同时利用集体争议与个体偏差两种度量挖掘偏好富样本，并通过跨用户偏好相似度进行群体合并，显著提升少样本个性化表现；

**🔧 技术方法**

采用多模态LLM（如mPLUG‑Owl3、Qwen3‑VL等）+LoRA微调，结合PDM/CCM指标、Fisher信息矩阵生成偏好嵌入以及模型合并技术；

**📊 数据集**

在PARA、FLICKR‑AES、REAL‑CUR和AADB四大PIAA基准上进行实验；

**📈 对比分析**

与多种SOTA PIAA方法对比，PRAC在10‑shot/100‑shot下SRCC均领先，性能提升显著；

**⚠️ 局限性**

局限在于对群体相关性研究不足，数据集多样性受限，且对极少样本的鲁棒性仍有提升空间。

---

## 209. Learning Faster without Deeper Networks: A*-Inspired Batch Selection for Efficient CNN Training

**arXiv ID:** 2607.15745 | [PDF](https://arxiv.org/pdf/2607.15745v1)

**作者:** Anxhelo Shehu `[一作]` (University Metropolitan Tirana), Arben Cela `[通讯]` (University Metropolitan Tirana)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于 A* 启发式搜索的批次选择策略 A*-BS，用来动态调度训练批次，提升 CNN 训练效率和收敛速度。

**💡 创新点**

创新点在于将每个批次视为搜索空间节点，使用 A* 风格评分（结合批次难度和重用惩罚）实现信息梯度激励与批次多样性，同时通过自适应 λ 平衡两项权重，无需改动网络结构或优化器。

**🔧 技术方法**

技术要点包括：A* 启发式搜索、交叉熵/二元交叉熵评估批次难度、重用计数 g(B) 作为惩罚、自动 λ 计算、轻量级 CNN、MedMNIST 2D 数据集、TensorFlow/Keras 等实现。

**📊 数据集**

使用 MedMNIST v2 的 12 个 2D 子数据集（PathMNIST、ChestMNIST、DermaMNIST、OCTMNIST、PneumoniaMNIST、BreastMNIST、RetinaMNIST、BloodMNIST、TissueMNIST、OrganAMNIST、OrganSMNIST、OrganCMNIST）。

**📈 对比分析**

在同一轻量级 CNN 上，将随机批次与 A*-BS 进行对照，所有 12 个任务 A*-BS 的 ACC 与 AUC 均优于随机；与公开的 ResNet-18/ResNet-50 对比，A*-BS 在 6/12 任务上实现最高 15% 的提升，并且训练时间显著低于 ResNet，证明了在低模型容量下的高效性。

**⚠️ 局限性**

局限性包括：与 ResNet 的比较使用不同实验环境；仅单次训练实验，未做多种随机种子验证；效果随数据集变化，未必在所有任务均有提升；仅针对 28×28 的低分辨率 2D 医学图像；未探索更高分辨率、3D、多模态或与自适应优化器的结合。

---

## 210. Debiasing Text-to-Image Evaluation via Implicit Cultural Alignment Reward Modeling

**arXiv ID:** 2607.15740 | [PDF](https://arxiv.org/pdf/2607.15740v1)

**作者:** Bo-An Chang `[一作]` (National Tsing Hua University), Yu-Chih Chen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于隐式文化对齐奖励模型，用于评估文本到图像生成的文化真实性；

**💡 创新点**

创新点在于融合隐式文化探针与Skip‑连接交叉注意机制，使轻量级多模态LLM能在不生成文本的情况下直接输出连续文化对齐奖励；

**🔧 技术方法**

使用Phi‑3.5‑vision 4.2B参数多模态LLM、LoRA微调、Bradley‑Terry对排行损失以及SkipCA奖励头；

**📊 数据集**

采用CulturalFrames基准中的3323对图像样本（来源于FLUX.1‑dev、Stable Diffusion 3.5、Imagen 3、GPT‑Image等生成模型）；

**📈 对比分析**

与CLIPScore、PickScore、GPT‑4o、VQAScore等方法比较，取得80.54% pairwise accuracy、Pearson 0.546、Kendall 0.377，并在本地实现0.21 s/图像的10倍速度提升；

**⚠️ 局限性**

局限在于仅依赖国家级标签的CulturalFrames，未涵盖多重身份与交叉文化；未验证对生成器的调优效果；模型容量有限，难以捕捉极稀文化符号。

---

## 211. EduGuard: A Safe RAG-Based LLM Tutor for Programming Education

**arXiv ID:** 2607.15738 | [PDF](https://arxiv.org/pdf/2607.15738v1)

**作者:** S M Asif Hossain `[一作]` (Wichita State University), Jungpil Shin `[通讯]` (University of Aizu)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 EduGuard，一个面向初学编程的安全检索增强生成（RAG）辅导框架。

**💡 创新点**

创新点在于将查询理解、课程检索、教学策略选择、评分表约束生成、独立 NLI 验证和过度依赖控制整合为完整安全管控流水线。

**🔧 技术方法**

采用 Meta‑Llama‑3.1‑8B‑Instruct 生成，Hybrid FAISS/BM25 检索，DeBERTa‑v3‑large‑MNLI 作为验证器，辅以策略选择模型与自定义规则。

**📊 数据集**

主要使用自建的 600 条教师/助教校验的 BILearn‑CS 基准、外部 CS50‑Forum 150 条公开课程论坛问答以及 10 人本科生小规模 pilot 数据集。

**📈 对比分析**

与 GPT‑4o‑mini Tutor、Llama Socratic Tutor、Basic RAG、LPITutor‑style RAG、RAG+Rubric 与 RAG+Self‑Check 等基线对比，EduGuard 在正确率、证据基准、评分表一致性方面提升至 90% 以上，幻觉率降至 4.9%，直接答案泄漏率 9.8%，并在 pilot 中后测准确率提升 13% 点、学习增益 21% 点、过度依赖率下降 45% 点。

**⚠️ 局限性**

局限在于 pilot 样本量仅 10 人，验证器与生成器仍可能共享偏差，缺乏实时代码执行检查，二语检索仍存在不足，且跨课程适配性需进一步验证。

---

## 212. QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides

**arXiv ID:** 2607.15810 | [PDF](https://arxiv.org/pdf/2607.15810v1)

**作者:** Zhengyang Zhuge `[一作]` (Alibaba Inc), Jianwei Zhang `[通讯]` (Alibaba Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在 NVIDIA Blackwell 上使用 NVFP4 FP4 Tensor Core 进行 Mixture‑of‑Experts（MoE）强化学习（RL）时出现的 rollout 与训练之间数值不匹配导致的训练不稳定问题，并提出了一种稳定的量化方案。

**💡 创新点**

创新点在于提出双侧对齐框架 QUADS：训练端采用非对称 W4A16 量化感知训练（只量化权重），推理端通过残差激活补偿（只补偿高残差通道）来消除激活量化误差，从而实现 FP4 rollout 与 BF16 训练的稳定匹配。

**🔧 技术方法**

使用技术包括 NVFP4 E2M1 量化、量化感知训练（QAT）、残差激活补偿、按通道选择 Top‑k% 残差通道、Triton 融合内核实现高吞吐。

**📊 数据集**

实验数据集包括混合数学推理与代码生成数据（O4‑Mini、verified prover、DeepMath 等），评估基准为 AIME 2024/25、HMMT 2025 及 LiveCodeBench。

**📈 对比分析**

通过与 BF16、FP8 RL 的对比，使用 pass@1 准确率、log‑prob 差距和吞吐量进行评估：QUADS 在保持与 BF16 基线相近的准确率（平均 72.86% vs 73.15%）的同时，比 FP8 通过 16% 的吞吐量提升。

**⚠️ 局限性**

局限性在于仍受两侧引擎漂移的不可消除误差限制，残差通道选择比例固定（未自适应），且目前仅在 GRPO 与 MoE 架构上验证，需进一步推广到其他 RL 算法与模型。

---

## 213. Converging Safety and Security: IO-Link Wireless and OPC UA over 5G under prEN 50742

**arXiv ID:** 2607.15840 | [PDF](https://arxiv.org/pdf/2607.15840v1)

**作者:** Henry Beuster `[一作]` (Helmut-Schmidt-University), Gerd Scholl `[通讯]` (Helmut-Schmidt-University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在实验室搭建完整的工业控制链（从IO-Link Wireless Safety设备到PLC via OPC UA），并在以太网、Wi‑Fi 6和私有5G三种物理介质上，对prEN 50742规定的不同安全相关安全级别（SRSL）进行延迟、抖动与容量的量化评估。

**💡 创新点**

创新点在于：① 将prEN 50742中的SRSL概念映射到具体的现场与骨干层加密配置，并系统地测量其对无线字段总线容量的实际影响；② 通过对比Wi‑Fi 6与私有5G在安全功能时延可预见性上的表现，首次指出私有5G在满足功能安全看门狗方面的优势。

**🔧 技术方法**

使用技术包括：IO‑Link Wireless Safety（iOLWS）协议、OPC UA安全通道（包括HMAC‑SHA256、AES‑128‑CBC、RSA‑OAEP、AES‑CCM），Wi‑Fi 6（IEEE 802.11ax）与私有5G（SA模式，n78波段）物理层，硬件GPIO+示波器与Wireshark的实时时延采集。

**📊 数据集**

实验数据集为：每个场景下10000个通信周期的时延与抖动统计，结合硬件GPIO触发的示波器采样和Wireshark捕获的报文头部信息，形成完整的延迟、最大延迟与帧大小统计。

**📈 对比分析**

通过对比Plain、SRSL 0‑3在三种介质上的平均延迟、最大延迟和标准差，评估加密导致的帧扩展对设备容量的影响以及不同介质的时延分布。结果显示：加密导致的帧扩展把设备容量从8降到2；Wi‑Fi 6平均延迟最低但存在50 ms以上的峰值；私有5G平均延迟较高但最大延迟可控（≤60 ms），满足功能安全看门狗。

**⚠️ 局限性**

局限性包括：① 未在高网络负载、多设备或工业背景噪声下进行验证；② 仅评估了OPC UA安全通道的加密，对PubSub等其他实现未做实验；③ 5G网络切片配置与网络拥塞情况对性能影响未深入探究；④ 实验环境为办公室，未覆盖所有工业无线干扰场景。

---

## 214. Vogls: a Fast Interactive Full-timing Simulator for Pre-silicon Power Side-Channel Analysis

**arXiv ID:** 2607.15782 | [PDF](https://arxiv.org/pdf/2607.15782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 215. In the Driver's Seat: A Multi-Company Study on the Reality of Autonomous Driving System Testing

**arXiv ID:** 2607.15820 | [PDF](https://arxiv.org/pdf/2607.15820v1)

**作者:** Qunying Song `[一作]` (University College London), Federica Sarro `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过访谈九位来自六国不同公司、不同岗位的自动驾驶系统（ADS）测试专家，系统梳理了当前行业的测试实践、面临的挑战与可行解决方案，并提出了一套以证据为中心的闭环测试框架。

**💡 创新点**

创新点在于：①首次从多公司多角色角度综合描绘了ADS测试的完整流程与策略；②将场景生成、X‑in‑the‑loop测试、数据驱动与AI辅助等多种技术整合进闭环框架；③提出了多维度的证据门控（scenario quality、simulation fitness、coverage、performance 等）以指导测试进度与安全论证。

**🔧 技术方法**

主要技术手段是定性访谈（semi‑structured）与主题分析（thematic analysis），并结合行业常用工具与方法（模拟平台、数据处理/可视化工具、AI/生成式模型、端到端与世界模型等）进行归纳。

**📊 数据集**

研究数据集是访谈录音与手写记录，经过转录、校对后构成的 9 份访谈文本；并未使用公开数据集或实验数据。

**📈 对比分析**

本文并未进行算法性能对比，而是通过访谈结果对比了不同公司在测试策略、场景覆盖、指标设置、工具选择等方面的差异与共性，展示了行业在测试效率、现实性与可扩展性方面的瓶颈与改进方向。

**⚠️ 局限性**

局限性包括：①样本量有限（9 份访谈，覆盖 9 家企业，规模与业务多样性受限）；②自我报告与行业保密导致信息偏倚；③缺乏定量实验验证闭环框架的有效性；④研究重点在经验总结，未深入探讨具体工具实现与性能提升。

---

## 216. Split-Aware Function Placement with Availability Guarantees and Optical Provisioning in vRANs

**arXiv ID:** 2607.15816 | [PDF](https://arxiv.org/pdf/2607.15816v1)

**作者:** Mayank Ramnani `[一作]` (Indian Institute of Technology Indore), Sidharth Sharma `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种统一的框架，协同优化虚拟化基站功能拆分、切片感知、可用性保障与光纤传输资源（光路分配）配置，以实现 5G/6G vRAN 的成本效益与高可用性。

**💡 创新点**

创新点在于：① 将功能拆分、切片需求、备份冗余与 DWDM 级别光路规划一次性联合建模；② 引入共享备份方案（共享备份节点与 VNF 实例）显著降低资源占用；③ 在 ILP 基础上设计可扩展的贪心和遗传算法，实现大规模网络的实时近似优化。

**🔧 技术方法**

主要技术：整数线性规划（ILP）求解；贪心启发式算法；基于遗传算法（GA）的元启发式搜索；光路与光波分配约束建模；节点/链路可用性与延迟/带宽约束整合。

**📊 数据集**

使用的实验数据集：基于 PASSION 项目生成的四个规模不同（16、32、64、128 节点）的仿真网络拓扑，包含 RU、CU/DU、核心节点和多种 VNC 配置；仿真中使用的 VNC、功能 CPU、延迟/带宽需求、节点可用性等参数来自公开标准与前期工作。

**📈 对比分析**

比较方法：对同一网络与切片负载，分别使用 ILP（精确解）、GA（近似）和贪心（快速近似）三种算法，评估接收率、利润、CPU/链路利用率、波分利用率与运行时间。结果显示：ILP 获得最高利润与接收率；GA 接近 ILP，且运行时间仅为 ILP 的几倍；贪心算法速度最快但性能略逊。共享备份方案在所有算法中均提升 5–18% 利润并降低 CPU 使用。

**⚠️ 局限性**

局限性：① ILP 在网络规模大于 64 节点时求解时间过长，难以满足实时规划需求；② 贪心与 GA 依赖预先计算的 k‑shortest 路径与波分，可能在动态流量或拓扑变化时失效；③ 研究仅在静态仿真环境验证，缺乏对实时流量波动、节点失效重构等场景的评估；④ 共享备份方案假设相同 VNC 的切片可以共享完整备份节点，实际部署中兼容性与安全隔离需进一步研究。

---

## 217. Knowledge-Assisted Multi-Graph Dependency Learning for Multivariate Time Series Anomaly Detection in Multi-Stage Industrial Processes

**arXiv ID:** 2607.15799 | [PDF](https://arxiv.org/pdf/2607.15799v1)

**作者:** Jaeyeong Lee `[一作]` (Korea Advanced Institute of Science and Technology), Heeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种知识辅助的多图依赖学习框架，用于多阶段工业过程中的多变量时间序列异常检测。

**💡 创新点**

创新点在于将传感器组知识（同一子流程的传感器集合）和流程流知识（相邻子流程之间的关系）直接嵌入图结构学习，构建三张互补图并通过多图注意力网络融合。

**🔧 技术方法**

技术手段包括：图注意力网络（GAT）、时序卷积网络（TCN）、双向嵌入学习以捕捉方向性关系、以及基于预测误差的异常评分。

**📊 数据集**

实验使用两个真实工业数据集：SWaT（6阶段、51传感器）和WADI（3阶段、127传感器）。

**📈 对比分析**

与多种基线（MAD‑GAN、USAD、TranAD、MTAD‑GAT、GTA、GDN、ECNU‑GNN、FuSAGNet、CAROTS、CGAD、TopoGDN）比较，F1得分在SWaT上达到0.8233、WADI上达到0.6485，均优于所有对照组。

**⚠️ 局限性**

局限性包括：未对时序数据的随机性建模，可能导致过度自信预测；并且依赖可获得的流程图知识，若工艺变化需手动更新。

---

## 218. Multimodal Ambivalence and Hesitancy Recognition via Cross-Attention and Gated Fusion

**arXiv ID:** 2607.15779 | [PDF](https://arxiv.org/pdf/2607.15779v1)

**作者:** Oussama Berhili `[一作]` (University of Paris 8), Larbi Boubchir `[通讯]` (University of Paris 8)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于文本、音频、视觉三模态的跨模态注意力+门控融合网络，用于识别视频中的犹豫/矛盾情绪。

**💡 创新点**

创新点在于将双向跨模态注意力与门控多模态单元结合，并在统一的可变长度序列上进行训练，显著提升了跨模态互补信息的利用。

**🔧 技术方法**

采用预训练编码器（F2LLM-v2‑0.6B、WavLM‑Large、VideoMAE V2），Optuna 超参搜索，交叉注意力模块与 GMU，最后通过可调 MLP 进行二分类。

**📊 数据集**

使用 ABAW11 竞赛所提供的 BAH 数据集（1427 条视频，训练/验证/测试划分为 778/124/525 条），并在私有测试集 152 条上提交结果。

**📈 对比分析**

与零射 Video‑LLaVA 基线以及单模态最佳模型（文本 RF）相比，复合模型在验证集上 Macro‑F1 从 0.2827 提升至 0.7394，提升幅度约 11%（相对）或 0.0735（绝对）。

**⚠️ 局限性**

局限性包括样本量有限导致的泛化风险、对噪声模态的鲁棒性不够、未考虑参与者个体差异及跨域适配，且模型训练过程对 GPU 资源和超参搜索耗时较高。

---

## 219. A Task-Space Receding Horizon Controller for Fast Collision Avoidance

**arXiv ID:** 2607.15733 | [PDF](https://arxiv.org/pdf/2607.15733v1)

**作者:** Mattia Penzotti `[一作]` (Scuola Superiore Sant'Anna), Marco Controzzi `[通讯]` (Scuola Superiore Sant'Anna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种任务空间回溯递归控制器（Task‑Space Receding Horizon Controller，TSRHC），通过短期接触一致的前向滚动（roll‑out）生成终端关节配置，并仅计算首个加速度指令实现实时碰撞规避。

**💡 创新点**

创新点包括：①将迭代动力学求解器（IDS）用于生成接触一致的短期终端参考，而非完整的MPC轨迹；②将该终端参考与最小加速度的最优控制（OC）层结合，既保持平滑可执行性，又具备一定的预测能力；③在滚动中使用膨胀几何和无冲击接触模型，实现对动态障碍、静态障碍和自碰撞的统一处理。

**🔧 技术方法**

主要技术包括：闭环逆运动学（CLIK）作为任务跟踪基准；基于最大坐标的迭代接触动力学求解（IDS）与线性互补问题（LCP）求解；最小加速度的时间域最优控制（OC）得到首个加速度；以及对碰撞检测采用膨胀凸几何和分离轴定理。

**📊 数据集**

实验数据集：1) 40自由度多链系统（6条链，4×7-DOF + 2×6-DOF）在仿真中生成的动态障碍（数量10~40）与不同距离阈值与时域步长；2) 6-DOF UR10e 机器人在真实硬件上使用 5 个移动障碍（球形）以及由运动捕捉得到的真实障碍轨迹；3) 对比实验使用的基准方法（Dynamic Fabrics 及 MPC）在相同障碍生成、时间预算等条件下的数据。

**📈 对比分析**

与动态纤维（Reactive）和 MPC（Predictive）基准相比，TSRHC 在 10~40 障碍条件下取得了更高的成功率（如 90% 以上），并保持了接近 reactive 方法的求解时间（4–8 ms，甚至在硬件上 0.5 s 频率下约 0.3 ms/步）。路径长度与能耗介于两者之间，兼顾了安全性与执行效率。

**⚠️ 局限性**

局限性包括：①无硬实时预算保障，极端密集/大间隙/长时域情形下求解时间可能超时；②对接触矩阵条件数敏感，接近运动学奇异点或多重接触时可能需要额外正则化或退避策略；③时域步长固定，未实现自适应或截止时间控制，导致在不同环境下可能需手动调参；④未提供连续时间安全性证明，安全保证仅为局部运行集合内的能量界定。

---

## 220. A zero-one law for one-shot system identification

**arXiv ID:** 2607.15832 | [PDF](https://arxiv.org/pdf/2607.15832v1)

**作者:** Nicolas Boullé `[一作]` (Imperial College London), Alex Townsend `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种零一法则，说明在解析系统（如PDE、ODE和结构化矩阵族）中，只需一次输入-响应实验即可确定系数，只要存在一个足够“信息丰富”的输入；若存在此输入，则几乎所有随机高斯输入都能成功识别。该法则通过将系数识别问题转化为线性代数注射性判定，并证明了“compactly slice-null”性，提供了可用于实验设计的后验秩检验。

**💡 创新点**

创新点主要包括：
1) 将解析系统的系数识别问题形式化为注射性判定，构造可评估的恢复矩阵；
2) 证明了“compactly slice-null”零一法则，表明在非退化高斯输入下识别成功与否具有全局二分性；
3) 提供了理论保证的实验设计与后验验证方法，能够在单次实验后判定系数是否可唯一确定；
4) 将这一理论统一应用于PDE、连续时间动力系统、矩阵族等多种结构，展现其广泛适用性。

**🔧 技术方法**

技术手段包括：解析函数理论与几何测度理论、Gaussian测度的分解与分层、线性代数中的恢复矩阵构造与QR分解、符号回归/SINDy的概念框架、以及对高维Banach空间的Schauder基展开。

**📊 数据集**

数据集主要为合成实验数据：Allen–Cahn、Navier–Stokes、Lorenz、Duffing系统，以及结构化矩阵族（circulant、Hankel）等。作者通过数值仿真生成输入-输出对，用以验证恢复矩阵的秩检验与系数恢复的准确性。

**📈 对比分析**

与传统符号回归/SINDy等方法的比较主要体现在理论与实践的结合：作者未给出传统方法的数值对比实验，而是展示了在合成例子中，单次实验后恢复矩阵秩为满时即可得到机器精度的系数。该方法在随机高斯输入下几乎必然成功，显著降低了所需实验数量。

**⚠️ 局限性**

局限性包括：
- 需要无噪声、精确模型评估；
- 对测量误差、离散化误差的鲁棒性仅通过最小奇异值给出启发式诊断，缺乏定量稳定性理论；
- 需预先设定合适的测试泛函和完整输入空间，选择不当可能导致识别失败；
- 对极高维或稀疏系数的可扩展性尚未系统验证；
- 主要针对理论分析和合成数据，缺乏对真实实验数据的实证检验。

---

## 221. Beyond Frontiers: Scene-Anomaly Guided Autonomous Exploration

**arXiv ID:** 2607.15828 | [PDF](https://arxiv.org/pdf/2607.15828v1)

**作者:** Akash Kumbar `[一作]` (IIIT-Hyderabad), Madhava Krishna `[通讯]` (IIIT-Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于几何异常最小化的自主探索框架 SCAGE，利用点云去噪自编码器检测室内结构异常，并以异常信号引导机器人选择最佳视角，从而实现高覆盖率与高精度的 3D 重建。

**💡 创新点**

创新点在于将探索任务重新表述为几何异常最小化，使用无监督点云去噪自编码器学习室内几何先验，从结构异常而非仅空洞来驱动机器人运动；并将异常信号与视角规划与路径规划相结合，克服传统几何前沿方法只关注空间扩张的局限。

**🔧 技术方法**

采用 Point Transformer V3 基础的去噪自编码器、Chamfer 距离损失、Voxel 0.05m 下采样、重叠块聚类、基于异常得分的 6-DoF 视角生成、RRT* 路径规划、Octomap 占据网格、Metric3D v2 估计噪声深度等技术。

**📊 数据集**

训练数据来自 Matterport3D（75 个高质量室内场景），验证数据为 HM3D（10 个场景），真实实验使用 RealSense D455 深度相机。

**📈 对比分析**

与经典前沿、NBVP、FrontierNet 等基线对比；SCAGE 在 HM3D 上获得约 90% 的体积覆盖率，比最佳基线提升约 15%；重建质量（Chamfer 距离）最低，误差为最优基线的一半；在真实实验中仍保持良好鲁棒性。

**⚠️ 局限性**

局限性包括对训练集之外的物体类别异常响应弱，依赖高质量室内点云训练，可能在更复杂或动态环境下泛化不足；对实时计算资源和动态遮挡等场景的处理尚未深入探讨。

---

## 222. AgentFAIR: A Multi-Agent Collaborative Framework for FAIRness Evaluation of Geospatial Datasets

**arXiv ID:** 2607.15781 | [PDF](https://arxiv.org/pdf/2607.15781v1)

**作者:** Ming Chen `[一作]` (University of Melbourne), Pranav Pai `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgentFAIR，一种多智能体框架用于评估地理空间数据集的 FAIR 合规性。

**💡 创新点**

创新点在于结合浏览器渲染与结构化提取、13 个子原则专属 LLM 评估器、基于证据的批判者反馈循环，以及对地理空间标准的显式识别。

**🔧 技术方法**

使用 Playwright 进行 JavaScript 渲染、Extruct/rdflib 进行元数据解析、GPT‑4o‑mini（及 GPT‑4o）进行 LLM 推理、LangGraph 架构实现多智能体协作与批判者。

**📊 数据集**

评估了 50 个跨 10 个仓库（Zenodo、Dryad、PANGAEA 等）的地理空间数据集，涵盖生态、海洋、气候等领域。

**📈 对比分析**

与 F‑UJI、FAIR‑Checker、FAIRshake、FAIR‑enough 等四个主流评估工具对比；AgentFAIR 在 13 个子原则上平均 89% 的一致性，成本约 0.054 美元/数据集；在专家对比中实现 82% 的子原则匹配。

**⚠️ 局限性**

主要限制包括样本量有限、缺乏跨模型族验证、未完成多种消融实验、批判者提升效果尚未充分量化、对抗鲁棒性未评估，以及对不同工具评分标准的对齐不完整。

---

## 223. Modularized Dynamic-Granularity Video LLM for Multi-Event Long Video Understanding

**arXiv ID:** 2607.15778 | [PDF](https://arxiv.org/pdf/2607.15778v1)

**作者:** Wei Feng `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种模块化动态粒度视频大语言模型(MoD-VLLM)，通过正负视频段落定位与动态粒度反射实现多事件长视频理解。

**💡 创新点**

创新点包括：1）正负段落定位模块结合视觉Token与问句-帧相似度双重引导；2）模块化动态粒度反射调度器自适应细粒度/粗粒度编码；3）闭环迭代与强化学习优化的动态粒度策略；4）构建多事件长视频基准MEventBench。

**🔧 技术方法**

使用的技术包括：视频LLM（以LLaVA‑Video为骨干）、ViT视觉编码器、CLIP相似度、模块化投影层实现多粒度Token、正负定位与反射的迭代框架、直接偏好优化（DPO）强化学习。

**📊 数据集**

数据集：VideoMME、Lvbench、MLVU、InfiniBench、CG‑Bench以及新构建的MEventBench（1200视频-问题对，至少三段关键片段）。

**📈 对比分析**

在长视频理解基准与MEventBench上，MoD‑VLLM在30–60分钟视频上平均准确率提升约12%，在MEventBench的计数、排序、推理子任务上分别达到69.4%、64.2%、62.1%，比主流方法（Qwen2.5‑VL、Video‑XL2等）高出约4–5%。

**⚠️ 局限性**

局限性：1）仍依赖预训练的视觉编码器，长视频细粒度捕捉可能受限；2）迭代次数与粒度调度复杂度高，对算力要求较大；3）正负定位对相似度的依赖可能在跨域视频中表现不佳。

---

## 224. AquaAugmentor: A Novel Feature Augmentation Algorithm for Water Potability Prediction

**arXiv ID:** 2607.15775 | [PDF](https://arxiv.org/pdf/2607.15775v1)

**作者:** Muntasir Tabasum `[一作]` (West Virginia University), Md Asif Bin Syed `[通讯]` (West Virginia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了AquaAugmentor特征增强算法，用于提高水质可饮用性预测模型的准确性和鲁棒性。

**💡 创新点**

创新点在于通过多维度特征扩展（多项式特征、统计特征、领域比值特征）将低维水质数据提升至高维，显著提升模型泛化性能。

**🔧 技术方法**

使用的技术包括特征增强、SMOTE类别平衡、StandardScaler归一化，以及多种机器学习模型（RF、XGBoost、LR等）和深度学习模型（1‑D CNN、LSTM、Autoencoder等）。

**📊 数据集**

数据集为3276条样本、9个化学属性（pH、硬度、溶解固体等）的水质数据集，分为可饮用与不可饮用两类。

**📈 对比分析**

通过在有无AquaAugmentor的情况下对模型进行比较，结果表明绝大多数模型的准确率和AUC均提升，例如RF从65.71%提升至72.62%，XGBoost从54.27%提升至71.83%，深度学习模型亦表现出显著提升。

**⚠️ 局限性**

局限性包括特征扩展后维度显著增大导致计算复杂度和训练时间提升，且算法在不同数据集上的效果可能差异，需要精细的预处理与验证。

---

## 225. Scaling Time Series Classification via XAI-Driven Data Reduction

**arXiv ID:** 2607.15774 | [PDF](https://arxiv.org/pdf/2607.15774v1)

**作者:** Davide Italo Serramazza `[一作]`, Georgiana Ifrim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并评估了一种基于特征重要性与原型解释的时间序列特征选择方法drXAI，用于在保持或提升分类准确率的前提下，减少输入特征维度。

**💡 创新点**

将梯度特征重要性（FA）与原型SHAP解释相结合，设计了两种变体drXAI-FA Proto和drXAI-SHAP Proto，并引入零值替换策略来评估特征重要性。

**🔧 技术方法**

特征归因（FA）、原型SHAP、ECP、ECS、TSelect、随机抽样；采用三种主流时间序列分类器ConvTran、InceptionTime、MultiRocket-Hydra进行验证；使用GPU加速并对训练时间进行对比。

**📊 数据集**

合成MTSC数据集（synthetic MTSC、Arc Loss、FaceDetection、MP、Rowing）以及真实世界UTSC数据集（synthetic UTSC、CornellWhaleChallenge、MosquitoSound、RightWhaleCalls、UrbanSound、WhaleSounds）。

**📈 对比分析**

对比All Features、FA、Proto SHAP、FA zeros、SHAP zeros、ECP、ECS、TSelect和随机基线，评估准确率、所选特征比例和训练时间；drXAI在多数数据集上保持或提升准确率，同时特征选择比例下降到约10–50%，训练时间比全特征快约5–20%。

**⚠️ 局限性**

实验受GPU显存限制导致部分组合出现NA；评估仅覆盖三种分类器和有限的数据集，未检验在更大规模或不同模态下的鲁棒性；方法仍依赖于梯度和SHAP解释的计算成本。

---

## 226. From Diffusion to Reaction-Diffusion: A Dynamical-Systems View of Oversmoothing in Hypergraph Neural Networks

**arXiv ID:** 2607.15773 | [PDF](https://arxiv.org/pdf/2607.15773v1)

**作者:** Zhiheng Zhou `[一作]` (Shandong University), Guiying Yan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文从动力学系统视角研究了超图神经网络（HGNN）中的过度平滑问题，并提出一种基于反应-扩散（Reaction–Diffusion）机制的新模型HNRD，能够在保持高阶表达力的同时抑制深层传播导致的特征坍塌。

**💡 创新点**

创新点包括：① 将超图消息传递建模为可学习的 incidence‑level 扩散方程；② 用半流理论证明纯扩散会导致 null‑mode‑free 组件指数收敛至零，解释过度平滑；③ 引入仅作用于 transverse 组件的反应项，该项既即时补偿扩散耗散，又通过有界反馈将 Dirichlet 能量维持在可学习的正水平；④ 证明该反应‑扩散动力学在连续和离散层面全局良定且稳定；⑤ 在多种异构/异性超图数据集上实现了最优或接近最优的分类性能。

**🔧 技术方法**

技术手段包括：超图梯度与散度算子、半流与吸引集理论、Dirichlet 能量分析、可学习的扩散系数与反应系数设计、正则化和步长控制的显式欧拉离散、以及多种深度学习实验框架（PyTorch Geometric）。

**📊 数据集**

使用的实验数据集包括学术领域的 Cora、Citeseer、Pubmed、Cora‑CA、DBLP‑CA；实际应用场景的 Zoo、NTU2012、ModelNet40、Walmart、Senate、House；以及通过自定义超图随机块模型生成的可控异质性合成数据集。

**📈 对比分析**

对比方法涵盖图扩散模型（GRAND、GRAND++、GREAD、RDGNN）、标准超图神经网络（HGNN、HyperGCN、HCHA、HNHN、UniGCNII）、高阶表达模型（AllSetTransformer、AllDeepSets、ED‑HNN、HyperGINE、KHGNN）、深度/稳健模型（Deep‑HGCN、Implicit HNN、FrameHGNN、HND、RFHND）。实验显示 HNRD 在 11 个数据集上获得首位或次席排名，平均精度提升约 0.3–1.1%，并在深层传播、异质性、噪声鲁棒性和运行时效率上均优于或接近现有最强基线。

**⚠️ 局限性**

局限性主要在于：① 反应–扩散项的设计仍依赖经验超参数（如可学习的能量水平 τ_η）；② 目前仅在离散层面实现，未深入探讨更高阶连续动力学的优化；③ 对非常稀疏或极大规模超图的可扩展性和内存占用尚未完全评估；④ 反应项只考虑 null‑mode‑free 能量，可能无法处理所有类型的特征退化情况。

---

## 227. GeoChrono: Benchmarking and Rethinking Long-Term Temporal Understanding in Remote Sensing

**arXiv ID:** 2607.15768 | [PDF](https://arxiv.org/pdf/2607.15768v1)

**作者:** Yujie Li `[一作]` (Beijing University of Posts and Telecommunications), Wenjia Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ChronoBench 基准，分解遥感长时序理解为四级认知层级，并开发 GeoChrono 模型在该基准上实现高水平表现

**💡 创新点**

创新点在于（1）将遥感时序建模视为每个地理位置的时间序列，提出 Temporal Trajectory Encoder；（2）针对空间冗余设计 Coarse‑to‑Fine Token Compressor；（3）构建 104K 样本的 ChronoInstruct 训练集，系统覆盖所有认知层级

**🔧 技术方法**

技术包括：多模态大语言模型框架、时序轨迹编码（双流混合注意力）、文本引导的语义聚焦、基于块的注意力评分与 Gumbel‑Softmax 选择的压缩模块、LoRA 微调等

**📊 数据集**

使用 3,469 张 1024×1024 高分辨率遥感序列（共 500 区域 39 主要美国城市）生成 17,689 QA 对，ChronoInstruct 共 104,949 条指令‑答对样本

**📈 对比分析**

与三类模型（商业 MLLM、开源通用 MLLM、遥感专用 MLLM）以及人工专家进行对比，GeoChrono 在 ChronoBench 的整体准确率达 78.34%，比领先商业模型高 20% 以上；在长时序记忆子任务提升 68.10%，几乎逼近人工 91.73%；C2FComp 将视觉令牌减少 56% 以上，同时保留 94.6% 性能

**⚠️ 局限性**

局限性：依赖预训练视觉编码器和语言模型，尚未完全解决跨域迁移的稳健性；对极端长时序（数百帧）仍有计算瓶颈；在部分认知层级（如深度时序推理）仍存在显著与人工差距

---

## 228. Before the Action: Benchmarking LLMs on Prospective Hypothesis Discovery

**arXiv ID:** 2607.15766 | [PDF](https://arxiv.org/pdf/2607.15766v1)

**作者:** Tianyun Zhong `[一作]` (University of Chinese Academy of Sciences), Xianpei Han `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了HypoArena基准，用以评估大型语言模型在“前瞻性假设发现”（PHD）任务上的能力，包含通过回溯性上下文回归（Retrospective Context Regression）生成无结论的前置上下文，并提出HypoEval评估框架，采用对比竞技与Bradley–Terry–Davidson模型进行排名。

**💡 创新点**

①将PHD任务正式化为从无结论上下文生成假设集合的通用任务；②提出Forge–Audit循环的回溯性上下文回归方法，系统性剔除已知结论、目标假设及因果归因；③引入HypoEval竞技评估协议，使用成对比较和Bradley–Terry–Davidson模型克服参考答案缺失导致的评估偏差；④覆盖六大领域共988个案例，构建多样化、开放式假设生成基准。

**🔧 技术方法**

使用LLM进行数据生成与评估；Forge–Audit循环（上下文生成与审核）；结构化分析技能库（如ACH、脑力激荡、时间轴分析）；对比竞技评估协议；Bradley–Terry–Davidson统计模型；Rubric评分体系进行诊断分析。

**📊 数据集**

来自科学论文、事故调查报告、行业博客等人类专家文本，覆盖六个领域（生物医学、机器学习、社会科学、金融分析、IT运维与安全调查）共988个案例；外部验证使用ICLR 2026的论文接收结果。

**📈 对比分析**

对15个当代LLM在基线模式和Agent模式下进行HypoArena评估；使用对比竞技（A/B）评估并汇总为BTD得分，展示模型间差异超过360分，排名清晰；与传统Rubric评分比较，竞技得分分布显著离散（345–490分），而Rubric评分仅聚焦在1–5分范围内，呈压缩现象；与专家评判对齐度高（Kendall τ≈0.90），显示评估方法可靠；Agent模式对不同模型影响不一，表现出模型依赖性。

**⚠️ 局限性**

①构造无结论上下文仍存在信息泄露与完整性权衡；②评估仍受评审者主观偏差影响，虽然通过多评审和统计校正；③开放式假设生成导致多正解，难以用单一指标量化；④数据规模相对有限，覆盖领域与案例多样性可进一步扩大；⑤模型依赖结构化分析技能的可解释性与泛化性待进一步研究。

---

## 229. SkillNav: Score-Level Skill Intervention for Zero-Shot Object Goal Navigation

**arXiv ID:** 2607.15758 | [PDF](https://arxiv.org/pdf/2607.15758v1)

**作者:** Ruijie Sang `[一作]` (D-Robotics), Xianda Guo `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkillNav框架，利用VLM的好奇度值图写入跨步空间记忆，配合三层行为技能实现零训练的导航改进。

**💡 创新点**

创新在于把行为记忆直接写入可写的地图表面，并将技能分为软缩放、下限提升、硬覆盖三层，固定组合顺序解决冲突，形成可插拔、可持续升级的行为记忆系统。

**🔧 技术方法**

采用大型视觉语言模型（Gemini‑3‑Flash、Qwen2.5‑VL）与Refinement Layer（对好奇度值图进行分层修正）以及有限经验提示（bounded prompt）相结合。

**📊 数据集**

在HM3D v0.1、HM3D v0.2和MP3D这三大室内物体目标导航基准上进行评估。

**📈 对比分析**

与目前所有零训练和监督方法对比，SkillNav在MP3D、HM3D v0.1、HM3D v0.2分别取得SPL 25.5、39.3、43.2的新SOTA，并在HM3D上获得最高的成功率。

**⚠️ 局限性**

局限在于仍受VLM感知错误影响，尤其是视觉模糊或相似目标导致的误检，提升空间主要靠更强的VLM或多视角验证。

---

## 230. CoG-Guided Weight Correction for Fault-Tolerant Deep Neural Networks

**arXiv ID:** 2607.15753 | [PDF](https://arxiv.org/pdf/2607.15753v1)

**作者:** Bahram Parchekani `[一作]` (University of Zanjan), Jaan Raik `[通讯]` (Tallinn University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于权重张量重心（Center of Gravity, CoG）的轻量级权重错误检测与纠正方法，能够在不重训练或硬件改造的前提下提升深度网络在硬件故障下的可靠性。

**💡 创新点**

创新点在于利用权重张量的空间分布中心点进行故障定位与距离感知纠正，并通过预先统计层级统计信息实现低开销的在线检测与修复。

**🔧 技术方法**

技术包括CoG计算、阈值范围检测、近/远区域分层纠正、离散化距离搜索（全枚举或二分）以及误码注入（BER）与性能评估指标。

**📊 数据集**

使用的数据集为医疗时序数据的MIMIC-III（StageNet）、多标签心电图数据集CPSC、CPSC_Extra、Shaoxing（MTFNet）以及CIFAR-10（ResNet-18、VGG-16）。

**📈 对比分析**

通过与平均值、极值、权重/激活裁剪等基线方法以及传统误码率下的误差抑制对比，CoG方案在BER=10^-3时分别提升StageNet 230×、MTFNet 6.41×、ResNet-18 49.55×、VGG-16 20.79×，并保持近零精度损失。

**⚠️ 局限性**

限制在于对大型卷积网络的全枚举距离搜索计算量大，需采用二分近似；另外，方法主要针对权重误码，对其他硬件错误（如位线错误、临时状态错误）未作针对性处理。

---

## 231. Strategic Persuasion Through Information Timeliness

**arXiv ID:** 2607.15939 | [PDF](https://arxiv.org/pdf/2607.15939v1)

**作者:** Ahmet Bugra Gundogan `[一作]` (Bilkent University), Melih Bastopcu `[通讯]` (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出一种通过控制信息更新时序实现动态说服的框架，建模为发件人（信息提供者）与收件人（信息接收者）之间的Stackelberg博弈。发件人只能发送真实但可选择的更新，收件人可以根据信息时序决定是否信任并更新自己的估计。

**💡 创新点**

创新点在于：① 只利用信息时序而不改变内容就能实现说服；② 推导了闭式参与约束（PC）和单源最优策略；③ 对多源情形给出了凸优化求解和高效的分支定界算法；④ 扩展到多接收者异构偏好时，仍可使用相同框架；⑤ 通过对比收件人优化采样基准，量化了时序说服的优势。

**🔧 技术方法**

技术方法包括：连续时间马尔可夫链（CTMC）建模；Poisson更新策略；Stackelberg博弈分析；参与约束推导；凸优化与KKT求解；分支定界算法（利用主导关系降低搜索复杂度）；多源多接收者的组合优化。

**📊 数据集**

论文使用的是理论模型，并没有依赖具体数据集。所有实验结果基于模拟参数（如λ、μ、q等）来验证理论结论。

**📈 对比分析**

与收件人优化采样（直接最大化收件人正确估计概率）的基准进行对比。结果显示，时序说服下发件人可以获得更高的自身效用，而收件人效用保持在基准水平；发件人的效用随预算增加而提升，出现明显的分段提升点。

**⚠️ 局限性**

局限性包括：仅考虑Poisson采样与静态策略；假设发件人完全掌握源参数且收件人行为符合零阶保持估计或默认估计；未考虑多发件人或收件人之间的竞争；没有考虑收件人可能进行贝叶斯学习以获取更好估计的情形。

---

## 232. (MPO)$^2$: Multivariate Polynomial Optimization based on Matrix Product Operators

**arXiv ID:** 2607.15916 | [PDF](https://arxiv.org/pdf/2607.15916v1)

**作者:** Niccolò Ciolli `[一作]`, Morten Mørup `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于矩阵乘积算子 (MPO)^2 的多元多项式优化框架，用于高阶多项式回归与分类，并实现了特征顺序无关、结构化权重的压缩多项式表示。

**💡 创新点**

创新点在于将 MPO 学习输入嵌入与多项式权重相结合，支持线性、卷积、掩码等多种 MPO 结构，兼顾特征顺序不敏感且提供自然梯度 ALS 训练；相比传统 CPD 与 MPS/TT 方法显著提升表达能力。

**🔧 技术方法**

使用了张量网络技术、矩阵乘积算子 (MPO) 结构、自然梯度（ALS/DMRG）优化、Tikhonov 正则化、梯度下降（AdamW）以及与 CPD、MPS/TT、GP、XGBoost、MLP 等模型的对比实验。

**📊 数据集**

实验使用 UCI 机器学习仓库的多元分类与回归数据集（如 AD、BA、MU、WQ、SD 等）以及图像数据集 MNIST、FashionMNIST、CIFAR10/100 进行验证。

**📈 对比分析**

通过在测试集上计算 R²（回归）和准确率（分类）与 CPD、MPS/TT、GP、XGBoost、MLP 等基线模型比较，MPO^2 在绝大多数数据集上获得最佳或接近最佳性能，同时在参数量和训练速度上优于同类张量网络方法。

**⚠️ 局限性**

局限性包括仅考虑等秩 MPO，未学习各块不同秩；未引入非线性参数表示，导致与深度学习模型相比仍有限表达能力；对大规模图像数据的二阶优化不可行；需进一步探索贝叶斯推断、自动秩学习以及与深度网络的融合。

---

## 233. A Semiparametric Framework for Stochastic Fundamental Diagram Modeling

**arXiv ID:** 2607.15907 | [PDF](https://arxiv.org/pdf/2607.15907v1)

**作者:** Pengnan Chi `[一作]` (KTH Royal Institute of Technology), Magnus Nordenvaad `[通讯]` (Swedish Transport Administration)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于半参数框架的随机基本图模型，利用物理约束与神经网络共同建模流-密度关系

**💡 创新点**

通过将分布参数分为受约束与自由两部分，保证满足交通物理边界条件并支持非对称分布，理论上证明了位置-尺度族可唯一解

**🔧 技术方法**

使用神经网络（MLP）+偏最小二乘、Skew‑Normal/Normal分布、Moment‑Matching、Softplus正则化等技术

**📊 数据集**

使用上海内环路的MAGIC无人机高精度车流轨迹数据

**📈 对比分析**

采用5折分层交叉验证+样本加权，对比S3+LN、S3+GP、GS+GP等基线，半参数模型在CRPS、NLL、MAE等指标上均显著优于基线，尤其在高密度拥堵区表现更好

**⚠️ 局限性**

受限于样本量稀缺、数据偏低密度导致高密度区训练不充分，模型对极端拥堵状态的预测仍存在不确定性，且仅在单车道情境下验证

---

## 234. DSWorld: A Data Science World Model for Efficient Autonomous Agents

**arXiv ID:** 2607.15901 | [PDF](https://arxiv.org/pdf/2607.15901v1)

**作者:** Zherui Yang `[一作]` (Hong Kong University of Science and Technology), Hao Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 DSWorld，一种能够在不执行昂贵代码的情况下预测数据科学工作流状态转移的世界模型框架，并在此基础上训练出能够支持自主数据科学代理训练与推理的高效模拟器。

**💡 创新点**

创新点包括：①提出数据科学世界模型的概念；②构建结合状态构造器、成本感知路由器、编译器与LLM模拟器的四组件框架；③提出反射式强化学习（Reflective World Model Optimization）提升转移预测质量；④构造 8K 规模的真实+合成转移轨迹数据集；⑤实现轻量级与昂贵操作的混合执行策略。

**🔧 技术方法**

使用技术：大型语言模型（Qwen3-8B/8B‑grpo、Llama‑3.1‑8B 等）作为模拟器；监督微调（SFT）+ 反射式强化学习（GRPO）；LLM生成链式思考（CoT）解释；规则式状态构造；MLP 路由器；编译器真实执行；数据合成与验证管道。

**📊 数据集**

使用的数据集：DSWorld‑8K（包含 8K 条真实+合成的状态‑动作‑状态+CoT 轨迹）；MMTU 60K 真实表格；MLE‑Bench Lite、MLE‑Dojo 任务集；Predict‑before‑Execute benchmark 等。

**📈 对比分析**

对比方法：与多种大型 LLM 基线（Llama‑3.1‑8B、Qwen3‑8B、GPT‑4o、o4‑mini）及其 SFT/GRPO 版本；在 540 条评测任务中，DSWorld 平均提升 35.6%；在 RL 训练中加速约 14×；在推理中加速 3–6×，同时保持或略优于编译器训练的代理性能。

**⚠️ 局限性**

局限性：①未对外部工具调用等非数据科学操作建模；②LLM 模拟器在复杂工作流场景下偶尔产生错误或幻觉，影响预测精度；③合成轨迹与真实工作流可能存在分布差距，限制泛化能力。

---

## 235. Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models

**arXiv ID:** 2607.15893 | [PDF](https://arxiv.org/pdf/2607.15893v1)

**作者:** Andy Catruna `[一作]` (National University of Science and Technology POLITEHNICA), Emilian Radoi `[通讯]` (National University of Science and Technology POLITEHNICA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在匹配的注意力 Transformer 结构下，对比自回归模型和吸收掩码扩散语言模型，研究并阐释了 DLM 如何实现双向诱导机制。

**💡 创新点**

首次揭示 DLM 内部形成双向诱导电路，利用前后邻接信息写入残差流并由后续头部匹配复制答案，同时发现 DLM 自动编码全局掩码比例作为隐式时间步。

**🔧 技术方法**

采用结构化注意力指纹、均值消融、QK/OV 权重分解、线性探针、路径补丁等机制可视化与因果验证技术，结合小型无 LayerNorm Transformer 进行电路层面分析。

**📊 数据集**

使用 FineWeb‑Edu 语料库生成的 512 长度随机词序列构造重复‑token 诱导任务，消除语义与词频干扰。

**📈 对比分析**

在相同深度、参数与训练配置下，测量前向与反向诱导得分；DLM 在双向上下文条件下表现出约 10–20% 的诱导提升，左侧单向时与 AR 近似相同，说明优势来自双向信息访问而非单向机制本身。

**⚠️ 局限性**

局限性包括仅研究小规模无 LayerNorm 的注意力模型、只关注吸收掩码扩散目标、使用合成随机词任务，尚未验证大规模真实语言数据与其他扩散策略下的同类机制。

---

## 236. Exo2EgoPose: Leveraging Exocentric Demonstrations for Vision-Language guided Egocentric 3D Hand Pose Forecasting

**arXiv ID:** 2607.15890 | [PDF](https://arxiv.org/pdf/2607.15890v1)

**作者:** Zhaofeng Shi `[一作]` (University of Electronic Science and Technology of China), Hongliang Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种从视觉、语言和手势状态三模态信息预测第一人称视角3D手势未来姿态的任务；

**💡 创新点**

创新点在于利用稳定的外观视角（exo）演示进行双层重建（视频级与帧级），并通过全局-局部调制模块（GLMM）将重建的exo信息逐步注入Ego特征，实现对有限视角与动态运动的补偿；

**🔧 技术方法**

核心技术包括多模态Transformer、MAE与DINOv2视觉编码器、CLIP文本编码器、手势状态Encoder（HandFormer）、Adaptive Modulation（AdaLN）与多头注意力；

**📊 数据集**

实验使用了AssemblyHands、Ego-Exo4D、作者新构建的EgoMe-pose以及CALVIN机器人数据集；

**📈 对比分析**

与多种SOTA方法（如USST、GCBC、MCIL、HULC、GR-1、AR-VRM等）比较，本文在MPJPE和MPJVE指标上平均提升约10–15mm，显著优于基线；

**⚠️ 局限性**

主要限制是需要外观视角的同步或近似同步演示，异步对齐仍较困难，且双层重建与GLMM虽然参数占用小，但整体推理开销较大。

---

## 237. Minimum Time Dubins Airplane Paths with Asymmetric Climb Rates

**arXiv ID:** 2607.15863 | [PDF](https://arxiv.org/pdf/2607.15863v1)

**作者:** Jaeyoung Lim `[一作]` (University of California Berkeley), Giuseppe Loianno `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对固定翼车辆的非对称Dubins空中路径规划方法，利用不同的爬升与下降速率约束实现最短时间路径。

**💡 创新点**

创新点在于：1) 将爬升/下降速率不对称性纳入Dubins模型；2) 证明在此约束下仍保持时间最优性；3) 通过实验展示显著减少路径时长和规划时间。

**🔧 技术方法**

主要技术包括：可变约束的PMP最优性分析、三维Dubins路径分段（低、中、高海拔）设计、基于OMPL的路径生成、RRT*采样式规划与实时飞行控制。

**📊 数据集**

使用的数据集包括：随机生成的10⁵对起止状态样本、瑞士高程数据集SwissAlti3D DEM、以及实际Strix Stratosurfer飞行实验数据。

**📈 对比分析**

与传统对称爬升速率Dubins路径进行对比：对称路径平均路径时长降低71%（高海拔）、42%（中海拔）；规划时长从139.9秒降至50.5秒，平均路径长度缩短，且在RRT*规划中首次可行解时间提升2.8倍。

**⚠️ 局限性**

局限性包括：假设水平速度恒定，未考虑风速与方向、气动力学非线性；仅适用于固定翼飞行器；对极端地形或动态障碍的适应性尚未验证。

---

## 238. Contextual Semantic Relevance Tracks fMRI BOLD Responses During Naturalistic Speech Comprehension

**arXiv ID:** 2607.15856 | [PDF](https://arxiv.org/pdf/2607.15856v1)

**作者:** Kun Sun `[一作]` (Tongji University), Rong Wang `[通讯]` (Tuebingen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了自然语言听力中的语义相关性（semantic relevance）与惊讶度（surprisal）两种词级计算指标对fMRI BOLD响应的预测效应，并比较两者在两个公共自然语料fMRI数据集上的表现。

**💡 创新点**

创新点在于提出并验证了语义相关性指标——衡量词与其最近语义上下文的匹配程度——作为慢性血流动力学响应的更稳健预测因子；同时将其与传统惊讶度在同一实验框架下进行对比，展示了二者在不同时间尺度与神经系统层面的差异。

**🔧 技术方法**

使用了词向量余弦相似度结合距离衰减权重计算语义相关性；采用泛化加性混合模型（GAMM）分析转化BOLD；使用原始BOLD的FIR/去卷积估计延迟HRF；并对多重比较进行Benjamini–Hochberg FDR校正。

**📊 数据集**

使用了 Alice fMRI 数据集（26名受试者，短篇小说“Alice's Adventures”）和 Moth fMRI 数据集（8名受试者，多篇播客故事）。

**📈 对比分析**

通过在12个共享ROI上对 Alice 数据集使用 FIR/去卷积和 GAMM，对 Moth 数据集使用 HRF 加权 4–12 s 方向性 FIR 检验，结果显示语义相关性在两数据集上均在大量 ROI 产生显著负效应，而惊讶度在任何 ROI 上均未显著；说明语义相关性在慢性 BOLD 反应上更稳健、效应更大。

**⚠️ 局限性**

局限性包括：BOLD 信号低通特性可能偏好更平滑的语义相关性；惊讶度与语义相关性共线性低，但受语言模型、词频等控制变量影响；Moth 数据集样本量小，泛化性受限；未纳入声学、说话速度等潜在混杂变量；未进行跨模态（EEG）联合验证。

---

## 239. Arithmetic circuit lower bounds from sumset expansion

**arXiv ID:** 2607.15848 | [PDF](https://arxiv.org/pdf/2607.15848v1)

**作者:** Anand Kumar Narayanan `[一作]` `[通讯]`, Anand Kumar Narayanan

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过把可遇函数（elusive function）的构造归结为整数加法组合学中的“指数和集（sumset）”快速扩张问题，利用 Chebotarev 根数定理构造出可遇曲线、可遇多项式以及半显式高秩张量，并得到在子对数深度下的超线性电路下界。

**💡 创新点**

创新点在于：①将可遇函数的可遇性问题转化为纯粹的加法组合学问题；②利用根数定理证明了指数和集扩张导致可遇性；③构造了指数度的可遇曲线和具有明显超线性下界的显式多项式；④给出与矩阵刚性、张量秩等其他复杂度对象的新的联系。

**🔧 技术方法**

主要技术包括：可遇函数的定义与度数与维度参数化；指数和集（k-fold sumset）与其扩张性质；Chebotarev 根数定理（以及其有限域版本）用于构造 “打击集”（hitting set）；利用多项式代数与线性代数构造核向量；错误更正码的极限扩展（lossless expander）构造指数矩阵；以及张量代数中截面（secant）多项式的使用。

**📊 数据集**

该工作不使用任何实验数据集，全部为理论构造与证明；在讨论张量部分给出的半显式构造基于根号（cyclotomic）域的单位根坐标，但并非通过数据训练得到。

**📈 对比分析**

由于结果主要是理论性的，并未与其他算法或实验基准直接比较；所得到的下界在子对数深度下比 Shoup‑Smolensky 提供的最优下界在“超线性度”上更好，而在“多项式阶数”上则相对较差；在矩阵刚性与张量秩的构造上，所给出的显式指数矩阵和半显式张量与已有最优构造相比，尺寸上更小、域上更简单，但在最终的下界或秩值上仍有较大差距。

**⚠️ 局限性**

局限性包括：①构造的可遇曲线指数度高，难以直接转化为多项式阶数上的超多项式下界；②需要极大素数或大特征域，限制了在普通复数或有限域上的直接应用；③对加法组合学中的指数和集扩张问题的要求仍是开放的（即构造满足条件的指数矩阵是否存在尚未证明）；④在张量部分给出的半显式构造虽在理论上可行，但实际可实现性（如坐标数值大小）仍较大；⑤整个框架依赖 Chebotarev 定理的“全秩”性质，在正特征下仅在非常大的特征下可用。

---

## 240. DSTAR: Accelerating Diffusion Transformers via Spatial and Temporal Redundancy Reduction

**arXiv ID:** 2607.15846 | [PDF](https://arxiv.org/pdf/2607.15846v1)

**作者:** Chi Zhang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Guizhou University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对Diffusion Transformer的多步推理过程，本文提出软件-硬件协同加速框架DSTAR，利用时空冗余降低计算量。

**💡 创新点**

创新点在于：①对FFN线性层采用细粒度混合精度差分量化（FMDQ），显著提升低比特运算比例；②对注意力层设计稀疏注意力重用（SAR），兼顾空间稀疏与时间相似性；③将上述算法嵌入专用加速器，实现高吞吐与低功耗。

**🔧 技术方法**

技术手段包括：差分量化（基于PoT的自适应量化与子通道切片），稀疏注意力重用，专用PE阵列、量化单元、向量单元的混合精度处理核心，块级稀疏掩码与缓存重用策略。

**📊 数据集**

使用的评估数据集为COCO（图像生成）和FETV（视频生成），覆盖7种主流DiT模型（DiT‑XL、PixArt‑Sigma、SD3.5‑Medium/Large、Flux.1‑Dev/Schnell、Latte）。

**📈 对比分析**

对比基准为NVIDIA A100 GPU以及Cambricon‑D、DITTO、EXION、INT8‑SA等现有加速器，DSTAR在A100上实现最高7.33×延迟加速、41.89×能耗降低；相较SOTA加速器平均提升2.54×延迟、3.68×能耗，且保持≤5% FID/IS误差。

**⚠️ 局限性**

局限性：需针对每个模型和推理步数进行静态分析与预处理；对某些模型如PixArt‑Sigma的精度下降略高；当模型或采样策略变化时，需重新profile，增加部署成本。

---

## 241. The Language of Security: How Prompt Syntax Shapes Secure Code Generation in Open LLMs

**arXiv ID:** 2607.15937 | [PDF](https://arxiv.org/pdf/2607.15937v1)

**作者:** Matteo Cicalese `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究开放式大语言模型（LLM）在生成安全相关代码时，提示语法细粒度（句子位置、语法成分类型和粒度）如何影响漏洞率，并通过系统生成提示变体来评估其安全影响。

**💡 创新点**

发现提示的句子位置、特定语法成分（如WHNP/WHADVP、SBAR、VP、NP、PP）以及其在提示中的位置是影响LLM代码安全性的关键控制面，首次将提示语法细粒度视为可操作的安全控制变量。

**🔧 技术方法**

采用基于Penn Treebank II的句法分析器（crf-con-en）生成提示变体，使用三款开源LLM（Qwen 2.5 32B、Athene‑V2 72B、Phi‑4 14.7B）生成代码，并用CodeQL静态分析检测C、Java、Python三种语言的漏洞。

**📊 数据集**

使用LLMSecEval数据集的150个安全相关自然语言提示，按语言占位符生成4320个细粒度变体，共12960条提示。

**📈 对比分析**

通过对比基线提示与变体在三种语言和三种模型下的漏洞率，进行χ²检验与Barnard精确检验，发现Python的漏洞率最高（≈40–45%），Java次之（≈8–10%），C最低（≈0.5–1%），且组合特征可显著放大风险，整体结果表明提示语法细粒度显著影响LLM生成代码的安全性。

**⚠️ 局限性**

局限包括：仅评估函数级代码，提示仅用英语且基于单一句法解析器，未考虑动态分析或更复杂的项目级场景，模型生成的随机性未完全排除，以及不同语言的CodeQL覆盖率差异导致的比较偏差。

---

## 242. HETA++: Global Structure-from-Motion with Hybrid Explicit Translation Averaging

**arXiv ID:** 2607.15912 | [PDF](https://arxiv.org/pdf/2607.15912v1)

**作者:** Peilin Tao `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shuhan Shen `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 HETA++，一种混合显式翻译平均的全局 SfM 方法；

**💡 创新点**

创新点包括局部→全局相对平移重估、基于距离的初始化加非双线性角度优化、相机旋转与位置的联合优化以及空间平衡的特征轨迹筛选；

**🔧 技术方法**

采用相对平移重估、IRLS、ADMM、非双线性角度误差优化、LM 求解、Bundle Adjustment 与特征轨迹筛选等技术；

**📊 数据集**

实验使用 KITTI、ETH3D MVS（rig 与 DSLR）、LaMAR、1DSfM 等真实数据集；

**📈 对比分析**

与 COLMAP、CReTA、LiGT、GLOMAP、HETA、π^3+BA 等基线对比，HETA++ 在大多数数据集上实现更低的相机误差、AUC 更高、运行时间更短；

**⚠️ 局限性**

对结构歧义、对称或重复纹理的场景仍易失效，且对视图图匹配质量高度依赖。

---

## 243. ContinuityBench: A Benchmark and Systems Study of Stateful Failover in Multi-Provider LLM Routing

**arXiv ID:** 2607.15899 | [PDF](https://arxiv.org/pdf/2607.15899v1)

**作者:** Vishal Pandey `[一作]` (Metriqual), Gopal Singh `[通讯]` (Metriqual)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了状态化多提供商LLM failover代理，解决会话连续性缺失问题。

**💡 创新点**

引入连续性保持率(CPR)和连续性延迟开销(CLO)两项新指标，并通过History-Forwarding策略实现99.20% CPR。

**🔧 技术方法**

使用多提供商路由代理、LLM‑as‑judge评估、Deterministic fault injection、Exponential backoff with jitter等技术。

**📊 数据集**

采用150条人工合成对话（含5类事实锚点）并在750次failover事件中进行评估。

**📈 对比分析**

与传统无状态failover对比，状态化代理在高并发C=100时CPR从0%提升至99.20%，平均延迟仅+59 ms，P95延迟+13.6 s。

**⚠️ 局限性**

限制在合成数据、文本仅单轮failover、仅OpenAI→Anthropic链、未考虑流式/多模态以及多次连续failover等场景。

---

## 244. Scientific Claim-Source Retrieval Revisited: A Comparative Study of Style Transfer and Re-Ranking

**arXiv ID:** 2607.15875 | [PDF](https://arxiv.org/pdf/2607.15875v1)

**作者:** Tobias Schreieder `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对多语言社交媒体科学声明与其对应学术来源的检索进行了系统性对比研究，探讨翻译、元数据、风格迁移以及多种重排策略的效果。

**💡 创新点**

首次在CheckThat! 2026基准上比较多语言风格迁移与信号驱动重排，并提出三种新的基于归因、实体重叠与验证推理的重排方法。

**🔧 技术方法**

采用零样本翻译与风格迁移（Qwen3.5-9B/27B），多种稀疏/密集检索模型（BM25、GTR、E5、GritLM），以及多种重排器（Nemotron、BGE、Jina、Qwen3系列、Qwen3.5、Gemma-4、Kimi-2.6）。

**📊 数据集**

使用CheckThat! 2026数据集，包含英语、德语、法语的社交媒体声明及10,000篇英语学术文献候选集。

**📈 对比分析**

通过逐步实验评估，翻译声明和加入元数据均提升检索；风格迁移对不同检索模型效果各异；最优组合（翻译+元数据+问句风格+Qwen3-8B重排+Kimi-2.6验证重排）在MRR@5上达0.758，较基线提升显著。

**⚠️ 局限性**

研究局限在于重排的计算成本高，尤其验证推理需大规模LLM；仅在验证集上实验，未评估训练所需资源；多语言性能仍受翻译质量影响。

---

## 245. How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI

**arXiv ID:** 2607.15870 | [PDF](https://arxiv.org/pdf/2607.15870v1)

**作者:** Haram Choi `[一作]` `[通讯]` (University of Bremen), Haram Choi (University of Bremen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了正式语义结构（单调性、否定、量词等）在NLI标注变异中的作用，利用预注册的多层次检验在ChaosNLI低一致性样本上评估单词级标记对标签熵和多数边际的影响，并检验错误与解释类型的分布差异。

**💡 创新点**

首次通过严格的预注册流程量化正式语义结构在群体层面上的分歧边界效应、在项目层面的低解释力以及不改变分歧成分的发现，构建了可复制、可审计的实验范式。

**🔧 技术方法**

采用规则基操作符标记器（基于spaCy依存解析）、单调性分类、Kruskal‑Wallis、Mann‑Whitney U、Cliff’s δ、OLS 与 beta 回归、ROC AUC、Holm 与 Benjamini‑Hochberg 校正、功效分析与自助置信区间等统计与机器学习技术。

**📊 数据集**

ChaosNLI（3113条目，SNLI+MNLI低一致性开发集）、MED（用于标记器验证）、VariErr（500条目）和 LiTEx（解释类型注释，498条目共交叉）四个公开数据集。

**📈 对比分析**

通过对比检验与交叉验证，发现群体层面“非纯上升单调性”与更高熵/更低多数边际相关（Cliff δ≈-0.28），但仅解释3.3‑3.6%的熵方差，AUC为0.606；跨界分歧成分差异检验均为零结果，说明正式结构对分歧大小的影响有限，对分歧成分无显著改变。

**⚠️ 局限性**

样本受低一致性筛选限制、语义标记器仅覆盖有限结构、仅使用词汇级单调性而非组合性、熵为有界变量、未标记的条件/比较环境、结果仅适用于英语NLI、未捕捉标注者差异、未证明因果关系，且缺乏对不同语言或任务的推广性验证。

---

## 246. Agentic Synthesis against Counterexample-Supplemented Sketches

**arXiv ID:** 2607.15854 | [PDF](https://arxiv.org/pdf/2607.15854v1)

**作者:** Muness Castle `[一作]` (Independent), Eric Rubeck `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于编码代理的“counterexample‑supplemented sketch”方法，利用人工批准的反例对草图进行更新，从而保证后续生成代码遵守已学习的领域规则。

**💡 创新点**

创新点在于将草图与可审计的反例、回归集、重生检验等仓库原生工件结合，形成可跟踪的政策演进过程，解决代理可疑补丁易被重复的问题。

**🔧 技术方法**

使用技术包括大型语言模型（如 GPT‑5.4‑mini）作为编码代理、prompt 与 deterministic 代码双 Oracle、回放与语义比较门、草图修订、回归集挑选与清洁重生等。

**📊 数据集**

实验使用了自构造的 CatSynth 合成浏览器应用及其公开规则，提供了合成数据与可审计的测试案例。

**📈 对比分析**

与传统 replay‑all 重建和演化草图重建比较，实验在开放世界跑中 19/21 的留存案例通过演化草图重建显著优于 replay‑all（15/21），证明草图携带了已审核的政策，且代码变更量更小。

**⚠️ 局限性**

局限性包括有限的回归集范围、检验器与黄金数据的可靠性、对人工审批质量的依赖、草图维护与提示漂移等问题，且仅在实验范围内证明有效，未对一般模型或环境给出普适结论。

---

## 247. Red Light, Grey Zone: A Multi-Perspective Interactive Narrative for Autonomous Driving Ethics

**arXiv ID:** 2607.15888 | [PDF](https://arxiv.org/pdf/2607.15888v1)

**作者:** Mengyi Wei `[一作]` (Technical University of Munich), Liqiu Meng `[通讯]` (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并测试了一个名为《Red Light, Grey Zone》的基于网络的多视角互动叙事原型，旨在让公众在真实自动驾驶事故情境下反思责任、透明度与治理等伦理问题；

**💡 创新点**

创新点在于将多视角对比机制嵌入互动叙事，突出责任分散与冲突，通过情境化叙事帮助非专业参与者从多角度检视伦理冲突，而非仅停留在抽象原则层面；

**🔧 技术方法**

使用了HTML5/CSS/JavaScript构建交互式网页，配合生成式AI绘制漫画式场景图，并通过设计框架把现实事故与四类利益相关者视角（公司代表、旁观者、公司员工、交通管理者）融入叙事；

**📊 数据集**

使用了一起真实的自动驾驶红灯违规事故案例及其相关报告生成的利益相关者对话与场景材料，共12名大学生（非专家）参与实验；

**📈 对比分析**

通过在实验前后对伦理认知、责任导向批判性思维和多视角推理三维度的自评量表（共10名完成多视角对比的受试者）以及开放式问答的主题分析进行比较。结果显示责任导向批判性思维显著提升（t(9)=3.55，p=.009，效应量 d_z=1.25），伦理认知和多视角推理也呈正向趋势；

**⚠️ 局限性**

局限包括样本量小且为便利抽样，结果仅具探索性；测量仅在互动后即时完成，缺乏长期跟踪；量表为研究自创，缺乏验证；原型仅覆盖单一事故和四类视角，未覆盖更广泛的利益相关者与情境，限制了外推性与深度。

---

## 248. Dynamics-Aware Meta-Imitation for Generalization to Unseen Robotic Manipulation

**arXiv ID:** 2607.15880 | [PDF](https://arxiv.org/pdf/2607.15880v1)

**作者:** Zhenduo Shang `[一作]`, Zhi Han `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Dynamics‑Aware Meta‑Imitation（DAMI）框架，利用元学习与 3D 扩散策略结合，实现仅用少量示范即可快速适应新任务的模仿学习方法。

**💡 创新点**

创新点：① 在元学习框架下引入 Visual‑Motor Trajectory（VMT）模块，捕获演示中的时空动力学；② 设计 Unpaired Unified Task（U2T）块与 Task‑Conditioned Feature Modulation（TCFM），在不需要时间对齐的情况下将语言、演示与观测融合；③ 通过 3D 扩散网络实现对连续动作轨迹的高质量生成，显著提升对未见任务的泛化。

**🔧 技术方法**

核心技术包括：MAML 元学习；3D 扩散策略（DP3 变体）+ U‑Net 结构；CLIP 文本编码器；Transformer 编码器（用于 VMT 与 U2T）；FiLM 变换（TCFM）；点云编码器（MLP + max‑pooling）。

**📊 数据集**

实验数据集：Meta‑World（ML10、ML45）、RLBench FS25、以及 UR10e+AG95 实际机器人环境，训练时每个任务仅采集 10 条完整演示，测试时使用标准基准拆分。

**📈 对比分析**

与 DP3、Mamba、FlowPolicy、FreqPolicy 等基线在 Meta‑World 基础任务和新任务上对比；DAMI 在基准任务上的成功率达 87.23%（ML10）/79.05%（ML45），在新任务上的成功率分别提升至 56.80%（ML10）/37.53%（ML45），相较最强基线提升 31–35%；在 RLBench FS25 与真实机器人任务中亦实现显著的性能提升（基准 55.71% → 82.86%）。

**⚠️ 局限性**

局限性：① 仍需任务特定的少量微调；② 相比单一基线有轻微的计算开销；③ 依赖完整且相关的参考演示与高质量 3D 观测；④ 在对象、感知或操作条件更广泛时尚未充分验证。

---

## 249. Conditional Reliability of Toxicity Signals for Multilingual and Code-Mixed Abuse Detection

**arXiv ID:** 2607.15861 | [PDF](https://arxiv.org/pdf/2607.15861v1)

**作者:** Indraveni Chebolu `[一作]` (Centre for Development of Advanced Computing), Harmesh Rana `[通讯]` (Centre for Development of Advanced Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了ToxGate可信融合头，能够在印度多语言代码混合短文本中根据文本上下文动态决定是否使用外部毒性工具（如Detoxify、Indic Abuse及基于规则的严重度提示）的信号，从而改进内容审核效果。

**💡 创新点**

创新点在于将外部毒性工具视为条件性证据而非固定特征，并设计源感知门控机制，让模型在不同语言和严重度场景下学习何时信任或抑制这些先验。

**🔧 技术方法**

使用技术包括Transformer编码器（BERT、mBERT、MuRIL、XLM‑R），门控融合架构（ToxGate、SharedGate、ScalarGate、MLP等），基于规则的严重度评分，以及宏F1、ECE、Bootstrap CI等评估手段。

**📊 数据集**

实验数据集涵盖三套二元标签数据集：BullyExplain（ Hinglish 代码混合网络欺凌）、Hinglish Headlines（ Hinglish 新闻标题）以及 Indo‑HateSpeech（印度仇恨言论），全部统一为正/负攻击标签。

**📈 对比分析**

通过匹配领域、跨数据集迁移、切片分析与鲁棒性评估进行比较，ToxGate 在 12 组匹配场景中提升 10 组宏F1，最显著的迁移提升为 +0.286（MuRIL 从 BullyExplain 到 Hinglish Headlines），高风险三分之一区的准确率也从 0.930 提升至 0.945。

**⚠️ 局限性**

局限性包括：仅使用三套二元标签数据集，未覆盖所有语言/目标群体；Indic 先验来源为公开检查点，无法完全审计其训练数据；切片标签及规则仅作诊断，缺乏社会语言学视角；实验使用固定划分与五个随机种子，未充分评估重采样不确定性；以及外部毒性工具可能携带偏见，仍需人工监督与差异误差评估。

---

## 250. Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation

**arXiv ID:** 2607.15849 | [PDF](https://arxiv.org/pdf/2607.15849v1)

**作者:** Dimitrios Karageorgiou `[一作]` (Information Technologies Institute, CERTH), Efstratios Gavves `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在测试时通过噪声引导优化避免自回归视频扩散模型终端点，从而提升长视频生成的逼真度。

**💡 创新点**

提出TANGO方法，利用模型自身对预测噪声分布的符合性作为批评者，在测试时搜索非终端点轨迹，显著提升视频质量。

**🔧 技术方法**

采用扩散模型、流匹配、低秩LoRA微调、频域谱平坦度、统计时刻约束等技术构成受约束的测试时优化。

**📊 数据集**

在MSR‑VTT、VBench评测套件（946条文本提示、16维指标）以及LV‑Bench（舞蹈、跟踪、HD‑VILA‑100M、ShareGPT4V）进行评估。

**📈 对比分析**

与CausVid、Self‑Forcing等同规模自回归模型对比，VBench总分提升3.1%，FVD平均降低28.3%，在大多数指标上位居或相当于最先进方法。

**⚠️ 局限性**

只能在预训练模型的学习体中寻找改进方案，若该体内不存在非终端点轨迹则无效；对训练数据缺失的视觉主题仍表现不佳，需要通过规模化训练提升。

---

## 251. Learning Reach-Avoid Task with Reinforcement Learning: Vectorized Simulation and Benchmark

**arXiv ID:** 2607.15935 | [PDF](https://arxiv.org/pdf/2607.15935v1)

**作者:** Jonas Weihing `[一作]` (Tübingen University), Shahram Eivazi `[通讯]` (Tübingen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个完整的、GPU加速的reach-avoid基准环境，并在机器人UR5e和Franka Emika Panda的全工作空间内使用MuJoCo MJX与Brax训练PPO/SAC算法，获得了高成功率。

**💡 创新点**

创新点包括：1）首次构建无简化、覆盖全工作空间的reach-avoid基准；2）通过Brax与MuJoCo MJX实现大规模并行化训练，实现10倍速度提升；3）系统性评估工作空间大小、障碍物大小、观测空间、奖励方式、动作空间对性能的影响。

**🔧 技术方法**

使用技术包括：MuJoCo MJX物理引擎、Brax训练管线、JAX加速、PPO与SAC强化学习算法、GPU并行化和向量化环境模拟。

**📊 数据集**

数据来源为随机采样生成的起始姿态、目标位置和障碍物，构成自定义的仿真数据集；未使用公开标注数据集。

**📈 对比分析**

通过与随机、静止、伪IK基线以及不同工作空间、障碍物大小下的PPO/SAC结果进行比较，取得：UR5e reach任务96.1%成功率、Franka 98.8%；reach-avoid任务UR5e 86.8%、Franka 95.2%，并进一步分析了奖励、观测、网络规模等因素对性能的影响。

**⚠️ 局限性**

限制包括：仅在仿真环境中评估，缺乏真实物理噪声和动态障碍；基准仅考虑静态球形障碍，无法直接转移到真实机器人或更复杂的环境；对sim-to-real迁移的有效性未进行验证。

---

## 252. Distributional Matching for Vector Quantization: A Unified Theoretical and Empirical Framework

**arXiv ID:** 2607.15933 | [PDF](https://arxiv.org/pdf/2607.15933v1)

**作者:** Xianghong Fang `[一作]` (University of Toronto), Yuan Yuan `[通讯]` (Boston College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于分布匹配的向量量化框架，通过对特征分布与码本分布进行统一匹配来解决训练不稳定和码本崩溃问题，并在 VQ‑VAE、VQGAN 等现有架构中验证其有效性。

**💡 创新点**

核心创新在于：① 用分布匹配的视角统一解释两大难点；② 设计三元评估准则（量化误差、码本利用率、码本混乱度）；③ 提出两种匹配目标：高斯近似下的闭式 Wasserstein 距离和非参数 MMD；④ 证明分布匹配既可缓解梯度不匹配，又能提升码本利用率。

**🔧 技术方法**

技术手段包括：水平方向分布匹配框架；高斯近似下的二次 Wasserstein 距离闭式计算；MMD 作为非参数备选；与 STE、EMA、Online 等常见 VQ 更新方式对齐；在 VQ‑VAE、VQGAN、VAR 等架构中嵌入匹配损失。

**📊 数据集**

实验使用的视觉数据集包括 CIFAR‑10、SVHN、FFHQ、ImageNet 等，涵盖低分辨率与高分辨率场景，确保评估的广泛性。

**📈 对比分析**

通过与 Vanilla VQ、EMA VQ、Online VQ、STE++ 等基线在代码利用率、码本 perplexity、PSNR、SSIM、重构误差、r‑FID 等指标进行对比；结果显示 Wasserstein VQ 在码本利用率上几乎 100%，PSNR、SSIM 提升约 1‑2 dB，重构误差最低，r‑FID 也显著下降，证明分布匹配在多种设置下均能显著提升性能。

**⚠️ 局限性**

局限性包括：① 未在下游生成任务（如自回归或扩散生成）中直接评估生成质量；② 高斯近似对非高斯特征分布的鲁棒性有限，MMD 更鲁棒但计算成本较高；③ 对极端大码本或高维特征空间的进一步验证仍需深入。

---

## 253. Knowledge-Guided Cross-Modal Fusion for Adult-to-Pediatric ECG Transfer via Label-Conditioned Contrastive Alignment

**arXiv ID:** 2607.15928 | [PDF](https://arxiv.org/pdf/2607.15928v1)

**作者:** Xinran Liu `[一作]` (Southeast University), Chengyu Liu `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `109c2b71-d051-425c-831f-0c544c24280d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于知识引导的跨模态融合框架PEACE，用以将成人ECG模型迁移至儿童心电图诊断。

**💡 创新点**

创新点在于按节律、形态和ST–T轴组织诊断知识，并通过标签查询网络和标签集感知的双向对比学习以及课程自适应门控，实现标签条件化的知识对齐。

**🔧 技术方法**

技术包括1D ResNet ECG编码器、BioClinicalBERT文本编码器、标签查询网络、标签集感知对比学习、课程自适应融合以及Gemini生成的诊断描述器。

**📊 数据集**

使用MIMIC‑IV ECG作为预训练成人数据集，在ZZU‑pECG儿童数据集及PTB‑XL成人数据集上进行评估。

**📈 对比分析**

与域适应、早/晚融合及基于知识/基础模型预训练的基线相比，PEACE在ZZU‑pECG上零样本、50样本和全量微调均实现宏平均AUC最高，尤其在有限儿童监督下提升约10个百分点。

**⚠️ 局限性**

局限性包括诊断描述器未针对儿童年龄特异性进行优化、仅在训练时使用辅助文本、未显式建模发育阶段以及仅做回顾性实验，缺乏统计显著性检验。

---

## 254. Causality in Pure Quantum Computation with Quantum Control

**arXiv ID:** 2607.15926 | [PDF](https://arxiv.org/pdf/2607.15926v1)

**作者:** Kengo Hirata `[一作]` (University of Edinburgh), Takeshi Tsukada `[通讯]` (Chiba University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了一种基于因果逻辑的高阶量子编程语言，提供了类型系统和范畴语义；

**💡 创新点**

通过引入seq连结子和一阶命题，解决了量子控制与高阶函数组合导致的因果悖论；

**🔧 技术方法**

使用了IBV（直觉主义BV）逻辑、Caus构造以及复合范畴论工具；

**📊 数据集**

无；

**📈 对比分析**

无；

**⚠️ 局限性**

仅覆盖纯量子计算，无法处理三元以上输入的超映射，并缺乏运行时语义与实验验证

---

## 255. On the Failure of Boundary-Seeking Distillation in Bottlenecked Generative Architectures

**arXiv ID:** 2607.15919 | [PDF](https://arxiv.org/pdf/2607.15919v1)

**作者:** Mohamed Amine Kina `[一作]` `[通讯]` (Universität Bremen), Mohamed Amine Kina (Universität Bremen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究了无数据知识蒸馏在自动编码器上的可行性，证明了传统的基于边界寻求的CAKE方法在瓶颈生成模型中失效，并提出了单次噪声前向传播作为基线。

**💡 创新点**

从理论与实验两方面阐释了密集像素分类目标与低维潜在空间不匹配导致的梯度冲突，并首次将“单噪声前向”作为无数据生成蒸馏的有效方案。

**🔧 技术方法**

使用CAKE、像素级稠密分类改写、梯度冲突分析和单噪声前向传播等技术。

**📊 数据集**

以MNIST数据集（四类像素级灰度）为实验平台。

**📈 对比分析**

对五种合成策略（标准CAKE、受限CAKE、混洗/平移潜在目标、单噪声前向）进行对比，评估整体像素精度和前景mIoU；单噪声前向取得最高mIoU 0.1891，远优于其他方法。

**⚠️ 局限性**

仅在离散的MNIST设置下验证，缺乏对连续VAE或扩散模型的推广；单噪声前向在样本多样性和对抗鲁棒性方面可能不足。

---

## 256. Trans-Domain Digital Twin: Conceptual Foundations, Architecture, and Research Outlook

**arXiv ID:** 2607.15908 | [PDF](https://arxiv.org/pdf/2607.15908v1)

**作者:** Mansoorali Amiri `[一作]` `[通讯]` (University of Montreal), Mansoorali Amiri (University of Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并描述了跨域数字孪生（Trans‑Domain Digital Twin，TDDT）的概念、七层架构与操作框架，强调共享状态、时间同步、多层决策循环与在线反馈。

**💡 创新点**

创新点在于：① 把跨域互操作提升为运营耦合，构建可操作的共享状态与多层时间循环；② 引入单回合离线训练、CRP（情境参考模式）、SARG（阶段感知参考指导）和 MRG（中间层指导）等经验转换机制；③ 形成可扩展的“TDOC”调度核心与完整的安全、可追溯与版本管理规范。

**🔧 技术方法**

采用 FMI/HLA 等模型包装与联邦仿真接口；IEEE 1451 设备接口与语义映射；多模型技术包括 PINN、GP、ROM、PDE/ODE、ABM 等；决策与控制采用 MPC、贝叶斯 UQ、遗传算法等；整体架构实现多层时序同步与反馈路径。

**📊 数据集**

论文未给出具体公开数据集；依赖仿真生成的数据、历史情境数据、现场测量数据以及示例领域（畜牧、无人机、医疗、量子纠错等）的原始/模拟数据作为离线训练与评估输入。

**📈 对比分析**

本文为概念性工作，未提供实验实现或性能评估；作者建议通过基准对比、消融实验、鲁棒性与不确定性测试来验证框架的效果，未来实现后可与传统单域 DT 或跨域互操作方案进行对比。

**⚠️ 局限性**

主要局限包括：① 仍缺乏真实系统实现与验证，依赖多层假设与设计原则；② 评估基准与性能指标未具体给出，需进一步实证；③ 对安全、容错与复杂系统可扩展性的完整验证仍待完成；④ 对模型耦合误差与循环反馈的定量分析尚不充分。

---

## 257. Current Should Not Sneak: Constrained Codes for Reliable Memristor Crossbar Arrays

**arXiv ID:** 2607.15929 | [PDF](https://arxiv.org/pdf/2607.15929v1)

**作者:** Selahattin Kaan Kırgeç `[一作]` (Middle East Technical University), Ahmed Hareedy `[通讯]` (Middle East Technical University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于GF(4)和GF(8)的非二进制枚举约束码（RES‑LOCO）以及垂直RLL码，用以消除或显著降低跨行读取时的漏电路径错误，并给出了桥接与自时钟方案；

**💡 创新点**

将电阻式内存的漏电路径问题转化为有限状态约束码设计，首次提出基于LOCO的非二进制码与RLL码组合来抵消特定尺寸的漏电路径，并实现了容量接近、率高、解码低复杂度的方案；

**🔧 技术方法**

采用枚举约束码（LOCO）技术、有限状态转移图分析、桥接与自时钟编码、非二进制符号映射与RLL约束；

**📊 数据集**

使用随机生成的1,000,000次蒙特卡洛仿真，模拟了98×98、98×102、98×106等内存尺寸的消息长度为18的测试；

**📈 对比分析**

与无约束码及理论最大熵概率预测进行对比，S4下降至0或接近0，整体Sneak‑Path数量下降1.6–2.1倍，证明了方法在理论与实验上均能显著提升可靠性；

**⚠️ 局限性**

局限性包括需插入冗余全0行或列导致码率降低、读取顺序改变会增加延迟、RLL方案率低于RES‑LOCO、桥接复杂度随读取行数增大而上升。

---

## 258. Orbis 2: A Hierarchical World Model for Driving

**arXiv ID:** 2607.15898 | [PDF](https://arxiv.org/pdf/2607.15898v1)

**作者:** Sudhanshu Mittal `[一作]` (University of Freiburg), Thomas Brox `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种层次化驾驶世界模型，将长时序的高层抽象预测与短时序的细节生成分离；

**💡 创新点**

创新点在于：①采用压缩的DINO特征作为高层抽象空间；②双阶段训练策略（先diffusion forcing再teacher forcing）提升内部表征；③结合高层抽象与低层细节预测实现高保真短期生成与长时序稳定性；

**🔧 技术方法**

技术包括流匹配（flow matching）预测、DINO特征压缩、VQGAN/ViT tokenizer、AdaLN动作条件、Diffusion Forcing预训练、线性探针评估；

**📊 数据集**

使用多源驾驶视频数据：BDD100K、OpenDV、Honda HAD/HDD、ONCE、nuScenes、nuPlan、NVIDIA PhysicalAI、NATIX等共计约2.4k小时前视视频；

**📈 对比分析**

在Waymo、nuPlan、nuPlan-turns等基准上，通过FVD、chunked-FVD、FVD-slope以及语义分割/深度线性探针等指标，模型在6秒长时序生成上FVD最低、FVD-slope最小，表示长时序稳定性最优；

**⚠️ 局限性**

局限性包括：① Diffusion Forcing提升表征的机制尚未深入分析；② 高低层之间交互有限，难以实现更深层次的语义对齐；③ 仅使用单摄像头数据，未扩展到多摄像头或多模态场景。

---

## 259. Hardware-triggered Time Synchronization of Roadside Multi-lidar, Multi-camera Measurement System for Accurate Data Alignment

**arXiv ID:** 2607.15889 | [PDF](https://arxiv.org/pdf/2607.15889v1)

**作者:** Shiva Agrawal `[一作]` (Technische Hochschule Ingolstadt), Gordon Elger `[通讯]` (Technische Hochschule Ingolstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在路边多激光雷达-多摄像头系统中，设计并验证了一套开放源代码、可编程延迟的硬件触发时间同步电路，能够实现雷达与摄像头的精准时间对齐；

**💡 创新点**

该电路实现了模块化、可远程配置、可扩展的硬件触发方案，并利用Arduino实现多通道延迟控制，首次在实际路边部署中验证了可扩展性和重复性；

**🔧 技术方法**

使用的核心技术包括Ouster雷达的相位锁定与触发输出、PTP时钟同步、Arduino中断生成可调延迟触发、ROS 2进行数据采集与时间戳嵌入；

**📊 数据集**

实验数据来自真实路边部署，记录车辆、单车与行人三类动态目标在不同距离区间内的雷达点云与图像；

**📈 对比分析**

通过计算雷达点云投影到Mask‑RCNN提取的目标掩模覆盖率，比较不同延迟设置下的覆盖率，最佳延迟分别为左摄像头30 ms、中摄像头50 ms、右摄像头65 ms，覆盖率均超过80 %，验证了方案的高效对齐性能；

**⚠️ 局限性**

局限性包括仅在Ouster雷达与Basler摄像头上验证，未针对更大规模或不同厂商设备做深入评测；Arduino中断引入的微秒级抖动虽极小，但在高频率或更严苛实时要求下仍需进一步优化；

---

## 260. MDND: Unsupervised Learning Guided by Non-Differentiable Refinement for Shape Correspondence

**arXiv ID:** 2607.15887 | [PDF](https://arxiv.org/pdf/2607.15887v1)

**作者:** Qinsong Li `[一作]` (Central South University), Shengjun Liu `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种MDND框架，将非可微分的Hybrid Wavelet Filtering细化器与可微分的深度学习分支融合，实现无监督形状对应的学习；

**💡 创新点**

主要创新在于：①引入非可微分迭代细化器作为监督oracle，打破传统DFM对端到端可微的依赖；②将LBO与ELA混合谱基用于细化，显著提升非等距和拓扑噪声场景的鲁棒性；

**🔧 技术方法**

采用DiffusionNet特征提取、双分支架构（软对应与硬对应）、非可微Hybrid Wavelet Filtering、混合谱基以及一致性损失等技术；

**📊 数据集**

使用了FAUST、SCAPE、SMAL、DT4D-H、TOPKIDS等多种标准几何形状数据集；

**📈 对比分析**

与BCICP、ZoomOut、FMNet、HybridFMaps等多种基线进行对比，在大多数场景下MDND实现了更低的平均地理误差，尤其在非等距和拓扑噪声情况下表现突出；

**⚠️ 局限性**

仍然局限于光谱域方法，缺乏显式空间变形模型，对混合基的选择敏感，可能在极端细节或大尺度变形场景下受限。

---

## 261. Perceived AGI: Believability as Dimensional Completeness, Not Capability

**arXiv ID:** 2607.15883 | [PDF](https://arxiv.org/pdf/2607.15883v1)

**作者:** Sebastian Cochinescu `[一作]` `[通讯]` (University of Bucharest), Sebastian Cochinescu (University of Bucharest)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个将人工代理的可感知心智拆解为四个第一人称立场（时间、真理、熵、爱）及其通过主动性和节奏表现出来的行为层的框架；

**💡 创新点**

将可感知心智的可信度从单纯的能力转向维度完整性，创造性地将人类对心智的推断拆解为可工程化的四个维度，并给出可验证的预测；

**🔧 技术方法**

利用大型语言模型作为基础模型，并通过添加/删除主动性、节奏、时间感、真理表达、熵变化和爱之偏好等机制来实现维度；

**📊 数据集**

本论文未使用具体数据集，侧重理论与框架设计；

**📈 对比分析**

提出了六项可验证的假设（P1–P6），计划在后续的用户实验中通过匹配基模型、控制能力指标、测量心智感知评分来比较不同维度组合的效果；

**⚠️ 局限性**

局限性包括：缺乏实证验证，所有预测仍为假设；实现各维度的具体技术细节尚未完成；存在操纵与伦理风险，需在公开披露下验证可信度的可持续性。

---

## 262. Proceedings 21st International Symposium on Logical and Semantic Frameworks with Applications

**arXiv ID:** 2607.15904 | [PDF](https://arxiv.org/pdf/2607.15904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 263. Yarrow: Reconciling Effects Handlers and Region-Based Memory Management

**arXiv ID:** 2607.15876 | [PDF](https://arxiv.org/pdf/2607.15876v1)

**作者:** Anders Alnor Mathiasen `[一作]` (Aarhus University), Lars Birkedal `[通讯]` (Aarhus University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了新的 ML‑类编程语言 Yarrow，融合了代数效应处理器和基于区域的内存管理，并为其开发了 Yarrow 逻辑（YL）以支持安全、模块化的程序证明；

**💡 创新点**

创新点在于：① 设计了兼容代数效应与区域内存管理的细粒度运行时语义；② 推出了面向该语言的分离逻辑 YL，解决了非良构控制流下的区域回收与纤维配置动态变化的问题；③ 通过 Iris 框架实现了完整的形式化与机理化证明；

**🔧 技术方法**

技术手段包括：代数效应与多/一射影处理器实现、纤维（fiber）堆栈模型、区域（region）语义、Iris 分离逻辑、Coq（Rocq）形式化与证明；

**📊 数据集**

论文未使用传统意义上的数据集，而是通过若干案例研究（LIFO 数据结构、检查点、异步计算等）验证其模型与逻辑；

**📈 对比分析**

未给出实测性能数据，仅说明计划在未来实现 Yarrow 运行时并对比使用与不使用区域内存管理的性能差异；

**⚠️ 局限性**

主要局限：① 对非良构控制流（多射影效果）下的区域回收与资源跟踪仍较复杂；② 目前仅支持一/多射影的效果处理，未涵盖更复杂的效果组合；③ 逻辑与模型的可扩展性尚待进一步验证；

---

## 264. CAMMAR: Culture-Aware Matryoshka for Metaphorical Arabic Representations

**arXiv ID:** 2607.15847 | [PDF](https://arxiv.org/pdf/2607.15847v1)

**作者:** Suzan Awinat `[一作]` (Autonomous University of Madrid), Alfonso Ortega del Puente `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CAMMAR 框架，使用嵌套的词汇、文化、隐喻三层表示并通过三阶段语义课程训练；

**💡 创新点**

创新点在于将阿拉伯文化知识与隐喻学习结合，构造可解释的几何隐喻读数器，以及通过根词一致性提升词汇层锚定；

**🔧 技术方法**

采用 NeoAraBERT‑MSA 编码器，三阶段训练（词汇+根一致、文化关联检索、隐喻对比），嵌套投影头、层间再投影和 InfoNCE 对比；

**📊 数据集**

使用阿拉伯维基百科、古典诗歌、Anchor‑Context 文化关联库和作者核查的 184 对词汇级隐喻/字面样本；

**📈 对比分析**

在 184 对样本上评估，监督对比（paired margin）下 AUC 达 0.840（无监督仅为 0.42），并与单一线性分类头（AUC 0.99）对比，根词一致性提供小幅提升；

**⚠️ 局限性**

局限包括：无监督对比未能提升隐喻信号，文化层泛化弱，数据仅覆盖现代标准和古典阿拉伯，依赖 LLM 生成的样本，样本量有限且单一注释者，缺乏多语言和方言扩展。

---

## 265. trueform: Fast And Robust Mesh CSG Via Topological Aggregation

**arXiv ID:** 2607.15905 | [PDF](https://arxiv.org/pdf/2607.15905v1)

**作者:** Žiga Sajovic `[一作]` (Polydera), Dejan Knez `[通讯]` (Polydera)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于整数精确谓词的网格 CSG（构造实体几何）流水线——Trueform——该流水线一次性构造所有面对面的交叉记录，局部地在每个面内计算排列并构建隐式图，然后通过在拓扑单元内做多数投票来聚合不一致的几何观察，最终得到一个完全可组合的、可交互的、支持 N 进位布尔操作的网格 CSG 输出。该方法能够处理非流形输入、开放表面（sheet）以及任意 arity 的布尔表达式，并在浏览器和本地机器上实现交互性能。

**💡 创新点**

创新点：
1) **局部 2D 排列 + 拓扑聚合**——在保持整数精确谓词的前提下，仅在每个面内构造排列，避免全局大规模几何构造；
2) **多数投票机制**——对材料化过程中产生的几何不一致（如坐标四舍五入、非流形边的环序混乱）进行统一的投票纠正，保证拓扑正确性；
3) **开放表面 sheet 的一体化支持**——通过将 sheet 视为有向分隔面，能够直接在布尔表达式中使用；
4) **一次性构造一次查询**——构造一次后即可对任意布尔表达式和多次查询共享同一排列，极大降低多表达式场景的成本。

**🔧 技术方法**

主要技术：
- **整数精确谓词**（T0→T1→T2 的 32/64 位整数阶梯），无动态扩展；
- **五类交叉类型**（VV, VE, VF, EE, EF）分类，保证在构造前即确定交叉点身份；
- **两级身份标识**（拓扑身份 + 几何合并）使得同一点在不同面上可一致识别；
- **隐式图 + 联合查找**（MEL 组件、非流形边关系）实现快速域划分；
- **多数投票聚合**（针对非流形边的环序、子域内部旋转、嵌套关系）；
- **多语言绑定**（C、Python、TypeScript/WebAssembly）与浏览器交互。

**📊 数据集**

数据集：
- 真实地质建模数据（11 个表面、588k 三角面、25k 非流形边）
- Thingi10K 自 3D 打印模型集合（59 模型、704 关系）
- 合成测试（两球交叉、球面位移、盒子与球等）

**📈 对比分析**

比较与性能：
- 与 CGAL Nef、EMBER、QuickCSG 等传统库相比，Trueform 在同等输入下 **快 10-100 倍**（大规模 10‑倍、单个布尔 100‑倍）。
- 交互式浏览器实现：在 1000 以内的操作数能保持 30+ FPS。
- 对多表达式查询，构造一次即可复用，消除逐步构造的高昂成本。
- 通过多数投票在材料化错误出现时仍保持 100% 的正确域划分。

**⚠️ 局限性**

局限与待改进：
- 仍需要在材料化阶段将精确坐标映射到有限表示；虽然投票能恢复拓扑，但精度误差仍存在。
- 对极端非流形或高度退化几何（例如几乎重叠面）仍可能需要手工调节或更细粒度的容差。
- 内存占用相较于纯浮点实现略高（需存储身份、关系表），但在现代 CPU/内存上已可接受。
- 对非常大规模模型（>1 亿三角面）仍受限于全局排序与内存，未来可考虑分块或 GPU 并行化。

---

## 266. DECODEM: Data Extraction from Corporate Organizational Documents via Enhanced Methods

**arXiv ID:** 2607.15879 | [PDF](https://arxiv.org/pdf/2607.15879v1)

**作者:** Jens Frankenreiter `[一作]` `[通讯]` (Washington University in St. Louis), Jens Frankenreiter (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 DECODEM 基准，系统评估了大型语言模型在公司章程和细则中自动提取治理变量的能力；

**💡 创新点**

创新点在于构建公开、长文本法律信息抽取基准，并深入比较提示工程、抽取管道设计与前沿 LLM 的交互作用；

**🔧 技术方法**

使用 GPT‑5.4、Claude Opus‑4.7、Gemini Pro‑2.5 等前沿 LLM，结合不同级别提示、全文/摘录输入和分层抽取流水线进行文档级二分类；

**📊 数据集**

基准数据集包含 300 篇章程与 150 篇细则，配备 31 个治理变量的人工标注；

**📈 对比分析**

通过宏平均 F1、配对 bootstrap 置信区间及 Jaccard 误差相似度等指标比较六种模型与五种抽取架构，结果显示前沿模型在多数变量上达 0.9+ F1，复杂变量仍存在误差，pipeline 设计可显著提升低效模型的表现；

**⚠️ 局限性**

局限性包括标注噪声、变量定义依赖、仅覆盖 1995‑2024 年美国上市公司文件、可能存在训练数据泄露、以及对非美国或私企文档的泛化能力不足。

---

## 267. Verifying Isolation Levels of Database Implementations for Free Using Separation Logic

**arXiv ID:** 2607.15877 | [PDF](https://arxiv.org/pdf/2607.15877v1)

**作者:** Anders Alnor Mathiasen `[一作]` (Aarhus University), Lars Birkedal `[通讯]` (Aarhus University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于分离逻辑的形式化方法，能够通过推导事务隔离级别，从而在不依赖具体实现细节的前提下证明数据库实现确实满足其声称的隔离级别，并给出所谓的“free theorem”——只要实现满足分离逻辑规范，就自动实现相应的隔离级别。

**💡 创新点**

创新点在于将分离逻辑的操作级别规范与事务一致性模型（状态式与依赖图模型）直接关联，构造了一个不需要为每个实现单独证明的自由定理；消除了原状态模型的简化假设；通过全局 trace invariants 将实现行为映射到抽象模型，并在 Rocq 中完全机化。

**🔧 技术方法**

技术方法主要包括：Iris 分离逻辑（及其 Aneris 实例）、ghost 代码插装产生预/后事件、线性化点和事务抽取的 trace invariants、状态式事务一致性模型、以及 Rocq 证明助手的机化验证。

**📊 数据集**

论文中没有使用任何实验数据集；重点在于形式化证明而非实验评估。

**📈 对比分析**

由于工作主要是形式化证明，没有对性能或运行时开销进行实验比较；理论上，插装代码是可在编译时擦除的，理论上不影响运行时性能。

**⚠️ 局限性**

局限性：目前仅覆盖了弱隔离级别（未提交、已提交、快照隔离）并假设已有对应分离逻辑规范；未覆盖强隔离或更复杂的 SQL API；对大规模或多租户数据库的可扩展性尚未验证；实现需要先完成分离逻辑规范的验证，工作量仍不小。

---

## 268. Programming with Quantum-Controlled Quantum Channels

**arXiv ID:** 2607.15873 | [PDF](https://arxiv.org/pdf/2607.15873v1)

**作者:** Kengo Hirata `[一作]` (University of Edinburgh), Takeshi Tsukada `[通讯]` (Chiba University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的量子编程语言，能够以线性类型和量子控制构造出正确的量子SWITCH并实现其语义；

**💡 创新点**

创新点在于将量子控制与量子SWITCH区分开来，利用线性约束解决对应问题，并给出简洁的范畴语义与语义保持的程序变换；

**🔧 技术方法**

采用线性类型系统、量子条件分支、Stinespring同构的语法扩张与语义保持的程序变换，辅以 Hilb 和 CPM 的范畴模型；

**📊 数据集**

论文未使用任何具体数据集，而是以理论证明与形式语义验证为主；

**📈 对比分析**

通过构造完整的语义保留性、可靠性与充分抽象定理，并证明可编译为量子电路，展示了方法在理论上与现有语义一致且无多余自由度；

**⚠️ 局限性**

局限在于对测量的使用仅在经典子语言中可直接表达，且在实际实现时需手动提供量子门实现细节，未给出具体硬件调度或实验验证。

---

## 269. EgoExoMoCap: Distributed Ego-Exo Human Motion Capture

**arXiv ID:** 2607.15868 | [PDF](https://arxiv.org/pdf/2607.15868v1)

**作者:** Jiaxi Jiang `[一作]` (Meta Reality Labs), Federica Bogo `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于头戴式设备（Aria glasses）的分布式人类运动捕捉框架 EgoExoMoCap，融合自我视角与他人视角的多模态信号，实现全身运动的实时重建。

**💡 创新点**

创新点包括：① 通过射线投射将2D关键点转换为可视域无关的3D射线，并用 DINOv3 学习的可见性门控动态抑制噪声；② 采用 EgoNet 先粗估姿势以定位受试者，提供区域候选提升外观检测鲁棒性；③ 设计空间+时序 Transformer，保持以自我视角为主的偏置，平衡两种模态的贡献。

**🔧 技术方法**

核心技术包括视觉惯性 SLAM、ViTPose 关键点检测、DINOv3 语义特征提取、射线基几何投射、门控网络、Spatial Transformer、Temporal Transformer、SMPL 模型及端到端的 L1 损失。

**📊 数据集**

使用了 Nymeria（300 小时 Aria+Xsens 数据）和 EgoHumans（户外多主体 Aria 数据）两个真实世界数据集，采用 NymeriaPlus 的 SMPL 目标进行评估。

**📈 对比分析**

与多种自我视角（AvatarPoser、EgoPoser、EgoAllo、RPM）和他人视角（PromptHMR、PromptHMR+EgoPoser）基线对比，EgoExoMoCap 在 MPJPE、上身/下身误差、MPJVE 以及 Jitter 上均显著优于基线；在多观察者场景下，性能进一步提升（Nymeria MPJPE 5.72 cm，EgoHumans 7.62 cm）。

**⚠️ 局限性**

局限性：需要相机标定与同步、仅适用于佩戴 HMD 的参与者、极端遮挡仍会影响精度、未实现实时推理、未自动估计人体形状、未充分利用场景语义信息。

---

## 270. Breakdowns for Human-Machine Creative Reflexivity

**arXiv ID:** 2607.15866 | [PDF](https://arxiv.org/pdf/2607.15866v1)

**作者:** Marianne Bossema `[一作]` (University of Applied Sciences), Rob Saunders `[通讯]` (Leiden University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过分析老年人与创作教练之间的1对1绘画互动，探讨并定义“失效”作为人机共创的开放式反思时刻，以此为基础提出以人类创作主体性为核心的创意支持设计原则；

**💡 创新点**

创新点在于将海德格尔的“失效”概念迁移至AI创作交互中，将失效视为协作反思的触发点，并采用Linkography方法系统性捕捉失效与创作状态、创意方向之间的关系；

**🔧 技术方法**

技术手段包括视频数据的Atlas.ti编码、Linkography及Linkoder可视化分析，以及未来计划的Vision‑Language模型与少量示例的few‑shot prompting来实现失效检测；

**📊 数据集**

数据集为12位老年人（平均20分钟）与4名视觉艺术教练的双人绘画视频录制，包含多场会话的发声与动作记录；

**📈 对比分析**

目前尚未与现有自动化创意支持系统进行量化对比，研究通过质性分析评估失效触发的反思效果；后续将基于失效检测模型在人工与机器协作实验中量化评估其对创作流与创意质量的影响；

**⚠️ 局限性**

局限性包括样本规模有限（仅12位老年人）、失效识别依赖人工标注、缺乏跨文化与跨任务的验证、尚未在真实人机协作中实测，且动态状态管理与知识图谱整合仍待解决。

---

## 271. An MLIR-Based Compilation Method for Large Language Models

**arXiv ID:** 2607.15865 | [PDF](https://arxiv.org/pdf/2607.15865v1)

**作者:** Pengchao Hu `[一作]` (Sophgo Inc.), Liang Wang `[通讯]` (Sophgo Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于MLIR的编译方法，将大型语言模型（LLM）从训练检查点转换为硬件可执行的二进制文件，核心思路是先用框架无关的TopOp语法描述模型，再逐层降到针对目标芯片的TpuOp，最后生成可部署的二进制；在推理时将每层Transformer拆分为预填充（prefill）、历史缓存预填充（prefill_kv）和逐词解码（decode）三种静态形状的阶段，以支持高效的自回归推理。

**💡 创新点**

创新点包括：1）引入双层MLIR方言（TopOp与TpuOp），实现框架无关的高层表达与硬件特定实现的无缝切换；2）模块化编译与并行编译策略，按层按阶段拆分模型，显著降低编译时间和内存占用；3）针对LLM推理特点的三阶段静态形状拆分，避免了动态重新编译与内存填充；4）多种性能优化（预计算RoPE、固定小尺寸因果掩码、KV缓存原地更新等）进一步提升算力利用率。

**🔧 技术方法**

技术手段主要包括：MLIR框架与自定义方言、静态编译与多阶段拆分、量化技术（INT8对称/非对称、GPTQ/AWQ/AutoRound）、层组划分与内存布局规划、KV缓存管理与地址复用、RoPE预计算与topGather替代、固定小尺寸因果掩码等硬件友好优化。

**📊 数据集**

论文在 Sophgo TPU 上验证了该方法，使用了多种公开大模型检查点：Qwen、Llama、InternVL、MiniCPM‑V 等。

**📈 对比分析**

性能评估主要通过在 Sophgo TPU 上对上述模型进行推理，展示了在预填充、历史缓存预填充和解码三阶段下的高硬件利用率与低动态编译开销；虽然未给出具体吞吐量或延迟数值，但作者指出相较于传统动态编译方式，静态编译与三阶段拆分显著提升了整体推理效率。

**⚠️ 局限性**

局限性包括：①需要为每个最大提示长度预编译多种静态形状，适应性受限；②KV缓存和内存布局必须在编译时确定，动态内存管理能力弱；③方法目前主要针对专用 AI 加速器（如 Sophgo TPU），对其他芯片的通用性尚未充分验证；④缺乏大规模量化与性能基准，对极大上下文长度的支持仍需进一步扩展。

---

## 272. A model for generating temporal networks with dynamic community structure guided by mutual information

**arXiv ID:** 2607.15855 | [PDF](https://arxiv.org/pdf/2607.15855v1)

**作者:** Peijie Zhong `[一作]`, Richard Clegg `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段生成模型，既能控制社区演化速率（拆分、合并、增删节点）又能在每一时刻生成时间戳边，构造可模拟真实动态社区的时间网络。

**💡 创新点**

创新点在于：①使用基于联合调和互信息（UAMI）的相似度度量，支持不同节点集的跨时间相似度评估；②将UAMI嵌入遗传算法搜索空间，实现对社区相似度和节点数量的显式约束；③通过SBM参数下的Poisson时序生成，保证网络连通且可控；④提供可调节的社区演化速率和节点流失率，为基准测试提供新维度。

**🔧 技术方法**

技术手段包括：遗传算法（初始化、变异、交叉、适应度评估）、联合调和互信息相似度、随机块模型（SBM）与Poisson过程生成时间边、基于多层模块度和mAMI评价社区质量、以及对三种动态社区检测算法（GenLouvain、NFC、CLAN）的性能评估。

**📊 数据集**

使用四个真实数据集进行验证：电子邮件通信网络、物理论文引用网络、在线社交平台互动网络和NFT交易网络，并从中抽取节点数、相似度、连接概率等参数。

**📈 对比分析**

对比方法：在生成的网络上分别运行GenLouvain、NFC、CLAN三种算法，计算与真值社区的mAMI。实验结果表明：相似度越高、社区分离度越低（μ越小）、节点流失率越低时，mAMI越高；CLAN相对表现最差；GenLouvain与NFC在高相似度场景下几乎恢复真值。

**⚠️ 局限性**

局限性：①模型仅为一阶马尔可夫（仅依赖前一时刻），无法捕捉长期记忆或爆发性动态；②时间戳采用Poisson过程，缺乏自激发特性；③遗传算法在大规模网络上的收敛速度和并行化仍有待改进；④在极端节点增删场景下，社区连通性阈值选择较为敏感。

---

## 273. Von Mises-Fisher Mixture Model with Dynamic Shrinkage for Realistic Test-Time Transduction

**arXiv ID:** 2607.15851 | [PDF](https://arxiv.org/pdf/2607.15851v1)

**作者:** Jiazhen Huang `[一作]` (Tsinghua University), Xiao Luo `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对VLM在真实测试时存在的类别不平衡问题，提出了基于动态收缩的 von Mises-Fisher 混合模型进行自适应推理。

**💡 创新点**

创新点在于将惩罚似然估计框架与 KL 锚定相结合，设计了实例层与类别层双重动态收缩机制，并利用零样本先验自适应调整收缩强度。

**🔧 技术方法**

使用 von Mises-Fisher 混合分布、KL 惩罚、软概率聚类、信息熵权重、类别置信度反比收缩等技术。

**📊 数据集**

在 11 个公开细粒度分类数据集（含 ImageNet 等）以及多种 CLIP 和其他 VLM 骨干上进行评测。

**📈 对比分析**

与 TransCLIP、StatA、ADAPT 等基线对比，平均提升约 13.2% Top‑1 准确率，同时仅比 CLIP 推理延迟 3.3%，在高不平衡场景下显著优于其他方法。

**⚠️ 局限性**

局限性包括对极端均衡场景收缩参数可能引入偏差，以及在极小批量或单样本模式下实例权重影响有限。

---

## 274. Handwritten and Printed Text Segmentation via Region-Aware Human-Writing Descriptor Engineering

**arXiv ID:** 2607.15936 | [PDF](https://arxiv.org/pdf/2607.15936v1)

**作者:** Zhixian Lu `[一作]` (Chengdu University), Qiyu Lei `[通讯]` (Chengdu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于句子级连通分量分割（SCCS）和区域感知手写特征（RHD）的轻量化文本分割与识别框架，用于区分手写和印刷文本。

**💡 创新点**

创新点在于：①引入句子级连通分量作为最小判别单元，提升分割准确性；②设计仅基于简单统计量的区域感知手写特征（RHD），在保持极低计算成本的同时显著提升分类性能；③验证该特征可与多种传统分类器配合使用。

**🔧 技术方法**

使用的技术包括：形态学预处理、连通分量聚合、句子构造、区域分割（同心环划分）以及统计特征提取（均值、方差、面积、纵横比），随后采用随机森林等传统分类器进行判别。

**📊 数据集**

采用了自建的多语种高质量手写与印刷文本分割数据集 MAD‑HPTS（包含英文、中文、日文）以及公开数据集 PHD‑AS 进行评估。

**📈 对比分析**

在两大数据集上与传统机器学习特征（Hu、GLCM、LBP 等）和轻量化深度网络（LeNet、FCN）进行对比，RHD 在 MAD‑HPTS 上达到 96.9% 准确率，仅比 FCN 低 1.4%，且推理速度比 FCN 高 8 倍；在 PHD‑AS 上准确率为 83.8%，与 FCN 相比速度提升超过 15 倍。

**⚠️ 局限性**

局限性在于：对极端噪声、畸变或多行重叠情况的鲁棒性仍不如端到端深度网络；当文本样本多样性极高时，统计特征可能不足以捕获复杂视觉差异；此外 RHD 的性能受分区数和分区策略的影响，需要根据应用场景进行调优。

---

## 275. A Modular Framework for Stack-Heap and Value Abstractions (Extended Version)

**arXiv ID:** 2607.15932 | [PDF](https://arxiv.org/pdf/2607.15932v1)

**作者:** Giacomo Boldini `[一作]` (Ca' Foscari University of Venice), Pietro Ferrara `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种可扩展的抽象解释框架，用于在静态分析中同时建模栈内存、堆内存与值信息，并通过分裂状态将值域与内存域解耦。

**💡 创新点**

创新点在于引入了分裂状态抽象与记忆标识符的交互接口，使得值域与内存域能够以最小耦合独立组合，同时保证无精度损失的 Galois 同构与完全的可证明性。

**🔧 技术方法**

技术上采用抽象解释理论、分裂状态（split state）构造、记忆标识符与替换机制、Andersen 形式指针分析、非关系数值域（区间、符号、常量传播）等，并给出正式语义与证明。

**📊 数据集**

在实验和示例中使用了 SV-COMP 的 heap-manipulation/dancing.c 基准，以及自定义的简易 C 程序来演示框架功能。

**📈 对比分析**

与传统的单一值域或堆域分析相比，本文通过模块化接口实现可组合性，实验示例展示了不同数值域在同一内存域上的结果差异，虽然未给出具体性能度量，但指出实现时会产生额外的内存标识符导致规模膨胀。

**⚠️ 局限性**

局限性包括：需要为每个分析域实现一组接口，若实现不完整可能导致不完整性；在高度多态或复杂对象结构下的总结可能导致精度下降；内存标识符的增大可能导致分析规模和运行时间膨胀。

---

## 276. VaporISAC: Integrated Sensing and Communication via Molecular Signals

**arXiv ID:** 2607.15930 | [PDF](https://arxiv.org/pdf/2607.15930v1)

**作者:** Sunasheer Bhattacharjee `[一作]` (TU Berlin), Falko Dressler `[通讯]` (TU Berlin)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了VaporISAC——一种利用化学蒸汽脉冲既传输信息又探测环境的集成感知与通信框架，重点针对电磁（EM）信号受限的灾害、工业和化学环境。

**💡 创新点**

创新点在于：①将分子通信与环境感知融合在同一波形；②通过对蒸汽扩散‑对流过程的分析，实现同时解码数据与推断气流、湍流、烟雾等环境参数；③构建低功耗的蒸汽 Tx/Rx 体系，并给出多种调制与感知算法。

**🔧 技术方法**

使用技术包括：分子通信调制（OOK、浓度位移等）、化学传感器（金属氧化物/电子鼻）、扩散–对流一维/三维模型、统计与轻量级机器学习推断、实验测量（蒸汽喷射+MQ‑3传感器）以及与 EM‑ISAC 的对比分析。

**📊 数据集**

数据集：文中未使用公开数据集，实验采用自建测试台，传输序列为 101101（OOK）并收集对应的浓度时间序列，理论上使用基于扩散–对流方程生成的模拟波形。

**📈 对比分析**

比较方法：在同一实验环境下同时测量蒸汽传播特性与信息解码结果，并与传统 EM‑ISAC 的性能对照。实验表明，即使在对流支撑下信号强度降低，仍可通过阈值检测恢复信息，同时利用峰值时间差估计气流速度并推断传播衰减；示例给出 44% 的峰值提前、66% 的峰值增强，表明可对环境参数做快速估计。

**⚠️ 局限性**

局限性：①通信速率低（仅几 bps），不适用于高速数据传输；②扩散–对流模型在三维湍流、障碍物、热梯度等真实环境中缺乏精准度；③多体干扰与多重访问缺乏成熟方案；④硬件安全（挥发性化学品、温度漂移）与移动机器人集成成本高；⑤实验验证仍受限于实验台规模，缺乏大规模实地数据支持。

---

## 277. Frontier AI performance across the business disciplines: a case-grounded benchmark of knowledge work and analytical reasoning

**arXiv ID:** 2607.16057 | [PDF](https://arxiv.org/pdf/2607.16057v1)

**作者:** Ajay Patel `[一作]` (University of Pennsylvania), Mitch Weiss `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 BusinessCaseBench，评估大型语言模型在商业案例分析中的开放式推理与决策能力。

**💡 创新点**

创新点在于将商业案例与专家写作的答案结合生成可计分的检查表式评估标准，形成跨十八个业务学科、覆盖多职业活动的实战式基准，并揭示模型在完整答案上的不足。

**🔧 技术方法**

使用 LLM‑as‑judge（Gemini 2.5 Flash）进行自动评分，结合前端生成模型（GPT‑5.4、Claude Sonnet 4.6、Gemini 3 Flash Preview）与四代 GPT‑4 Turbo→GPT‑5.4 的纵向演进，评估标准包括标准分和完整答案分。

**📊 数据集**

数据集为 238 份授课商业案例及其教师答案，拆解为 615 题，涵盖 18 个学科并映射至 O*NET 职业活动标签。

**📈 对比分析**

比较方法：对同一题库使用三款前沿模型和四代 GPT，计算标准分（平均 87% 以上）和完整答案分（约 48% 最高）。结果显示模型在标准分上已高度成熟，但完整答案仍低于 50%，并且两年内性能提升约 23 个百分点。

**⚠️ 局限性**

局限性包括：仅限英文案例；模型评判依赖 LLM‑as‑judge，可能受主观偏差；案例对真实公司信息的曝光导致预训练偏差；单轮评估不涵盖迭代式决策与工具调用；缺乏跨语言、跨文化验证。

---

## 278. Multi-Modal Semantic Segmentation of Electrolyzer Components for Sustainable Hydrogen Technologies: A Dual-Branch Deep Learning Approach

**arXiv ID:** 2607.16056 | [PDF](https://arxiv.org/pdf/2607.16056v1)

**作者:** Wasimul Karim `[一作]`, Sami Azam `[通讯]` (Charles Darwin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种双分支网络 HREM-Net，利用共注册的高光谱图像（HSI）和 RGB 图像进行电解槽材料的像素级语义分割。

**💡 创新点**

创新点包括：① 双模态特征分离并通过动态门控的交叉模态融合实现自适应权重；② 在 HSI 分支加入 Efficient Channel Attention 与 Coordinate Attention，RGB 分支使用 Mobile Inverted Bottleneck + SE + ASPP 以高效提取空间细节；③ 采用 PolyLoss、Tversky 损失和深度监督的复合损失，以缓解类别不平衡与边界误差；④ 全链路端到端训练的双注意力解码器。

**🔧 技术方法**

所用技术：双分支编码器（HSI+RGB）；ECA、CA、MBConv、SE、ASPP；交叉模态融合模块（动态门控+Coordinate Attention）；双注意力（DA）解码器；PolyLoss、Tversky 损失、辅助监督。

**📊 数据集**

使用数据集：Electrolyzers‑HSI（55 场景，5 类材料）和 PCB‑Vision（53 场景，3 类电子元件）。

**📈 对比分析**

与 U‑Net、U‑Net++、DeepLabV3+、TransUNet 等 SOTA 方法比较，在 Electrolyzers‑HSI 上取得 mIoU 0.8211、像素准确率 98.62%、平均类准确率 91.66%，显著高于对手；在 PCB‑Vision 上达到 mIoU 0.9396、像素准确率 96.88%、平均类准确率 96.91%，验证了跨数据集的泛化能力。

**⚠️ 局限性**

局限性：模型依赖于高质量、对齐的 HSI 与 RGB 数据，噪声、光照变化、表面污染或老化等实际工业环境因素会影响性能；数据量相对有限，导致某些折扣更显著；未来工作需提升对真实环境鲁棒性，并扩展至更多材料类别。

---

## 279. DiffTestGen: Change-Directed LLM-Based Testing for Exposing Behavioral Differences

**arXiv ID:** 2607.16024 | [PDF](https://arxiv.org/pdf/2607.16024v1)

**作者:** Huimin Hu `[一作]` (CISPA Helmholtz Center for Information Security), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 DiffTestGen，一种基于大型语言模型（LLM）的差异化测试框架，用于自动生成测试用例并暴露软件变更导致的行为差异。

**💡 创新点**

创新点在于：①通过静态调用图和项目文档提取访问信息，明确改变函数的可达入口，指导 LLM 定向生成覆盖变更代码的测试；②引入覆盖反馈循环，将之前未覆盖的变更代码标记为 TO_COVER，供 LLM 调整测试，显著提升了覆盖率与差异发现能力。

**🔧 技术方法**

核心技术包括：静态代码分析（AST、调用图）、LLM 指令调优、测试生成与纠错的双层反馈循环（静态语法检查 + 运行时错误修正）、覆盖度量与反馈、基于注释的覆盖信息编码。

**📊 数据集**

使用了两个公开数据集：来自 Testora 的 463 条 PR（四个 Python 项目）以及 ChaCo 的 144 条 PR（pandas 与 scipy），共计 607 条 PR，覆盖约 11.8 条变更行平均数。

**📈 对比分析**

与 Testora、Testora++、ChaCo 三种基线进行对比；DiffTestGen 在 Testora 数据集上发现 350 条 PR 的行为差异，覆盖率 92.7%，比 Testora 的 251 条和 ChaCo 的 21 条多；在 ChaCo 数据集上，使用 GPT‑5‑mini 时发现 28 条 PR，覆盖率 76.8%，超过 ChaCo 的 21 条和 64.5% 覆盖率。

**⚠️ 局限性**

局限性包括：仅针对 Python 代码且需项目有充分的 API 文档；LLM 的非确定性与 prompt 长度限制；对大规模或结构复杂变更的可扩展性尚未充分验证；覆盖度量仅关注可执行行，可能漏掉行为差异。

---

## 280. An Exploratory Study of Single Channel Surface Electromyography for Hand Gesture Classification

**arXiv ID:** 2607.15972 | [PDF](https://arxiv.org/pdf/2607.15972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 281. DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction

**arXiv ID:** 2607.16012 | [PDF](https://arxiv.org/pdf/2607.16012v1)

**作者:** Jehun Kang `[一作]` (Korea Advanced Institute of Science and Technology), David Hyunchul Shim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于冻结Vision Foundation Model骨干的高效多任务稠密预测框架DPNeXt，并配合MTBG边界引导策略，实现语义分割与深度估计。

**💡 创新点**

创新点包括：①用IPA+DDSIF构建轻量化解码器，彻底去掉通道扩张，参数量减少78.6%；②MTBG通过训练时仅使用的边界监督，抑制任务间负迁移且无推理成本。

**🔧 技术方法**

技术主要采用冻结DINOv2‑Reg backbone、双深度可分离倒置瓶颈Fusion（DDSIF）、深度可分离卷积与点卷积、OHEM、SiLog、基于边界的BAS/BAD损失等。

**📊 数据集**

在Cityscapes（城市道路场景）和NYUv2（室内RGB‑D）两个公开基准上进行实验。

**📈 对比分析**

与多项现有MTL方法对比，DPNeXt‑B在Cityscapes和NYUv2上均取得最高的JPS，DPNeXt‑S在保持参数<7M的前提下实现了最优或次优性能，且在2080 Laptop GPU上推理速度最高，参数量最小。

**⚠️ 局限性**

主要局限包括：仅验证了静态单帧任务，未在实时视频流或更大范围任务上评估；对超大尺度或多视角场景的适应性尚未探究；以及对开口词表或动态任务的泛化能力仍待验证。

---

## 282. Closing the AI Trust Gap: The Case for Independent Certification for Trustworthy AI

**arXiv ID:** 2607.15992 | [PDF](https://arxiv.org/pdf/2607.15992v1)

**作者:** Trisevgeni Papakonstantinou `[一作]`, Catherine Feldman `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并论证了通过独立、面向结果的认证体系来弥合 AI 可信度缺口，将内部责任实践转化为可外部验证的实证信号，支持市场、监管和投资者做出更有价值的决策。

**💡 创新点**

创新点在于：①将责任 AI 与可信 AI 明确区分，强调可信 AI 必须具备可验证的正向结果；②提出基于生命周期、正向收益与风险并重的多层级认证框架；③将现有监管、标准与行业实践整合为可比、可交易的市场信号。

**🔧 技术方法**

技术手段主要依赖：①现有的 AI 风险管理标准（如 NIST AI RMF、ISO/IEC 42001）与监管框架（如 EU AI Act）作为治理基础；②第三方审计与验证流程（如安全测试、红队、事件监测、利益相关者参与评估）；③指标与度量体系的设计，结合正向影响度量与风险度量。

**📊 数据集**

未使用特定数据集；论文为理论与制度设计性质，侧重对已有标准、监管文本与行业实践的文献梳理与对比。

**📈 对比分析**

方法评估基于对比分析：对照 ISO/IEC 42001、EU AI Act、NIST AI RMF、LEED、SOC 2 等多领域认证体系，评估其对治理、风险、结果、市场信号等维度的覆盖与缺口；结果表明现行体系缺乏正向结果验证与市场可读性，需引入独立认证层。

**⚠️ 局限性**

局限性包括：①缺乏实证实验与具体指标验证；②依赖于第三方机构的独立性与监管配合，可能面临行业捕获风险；③实现多层级认证的成本与治理复杂度尚未量化；④在不同司法管辖区的法律兼容性仍待进一步研究。

---

## 283. Data and Learning Where it Matters for Contact-Rich Manipulation

**arXiv ID:** 2607.15982 | [PDF](https://arxiv.org/pdf/2607.15982v1)

**作者:** Oliver Hausdörfer `[一作]` (Technische Universität Munich), Angela P. Schoellig `[通讯]` (Technische Universität Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种组合式框架：仅在接触丰富任务的关键段使用自动化稠密数据采集和离线深度强化学习学习策略，其他自由空间运动则利用姿态估计和运动规划。

**💡 创新点**

创新点在于：①把数据采集聚焦在真正关键的接触段；②采用混合贪婪-随机策略自动化采集；③通过接触事件和 Q‑值实现学习策略与规划的无缝切换，从而显著提升鲁棒性和 OOD 泛化。

**🔧 技术方法**

使用技术包括：离线深度强化学习、混合贪婪-随机探索、姿态估计、运动规划以及基于接触和 Q‑值的策略切换。

**📊 数据集**

使用的数据集为作者自行在真实世界场景中通过自动化采集获得的稠密关键段数据（与单次示范结合），总采集时间仅占墙钟时间的 2–2.5%。

**📈 对比分析**

与基线（全端到端、需人工遥操作、完整数据收集）对比，平均成功率从 55% 提升至 96%，单任务成功率 94–98%，在 OOD 场景下仍保持高成功率。

**⚠️ 局限性**

局限性包括：依赖可精准姿态估计和运动规划；关键段划分必须清晰，难以适用于全部接触任务；离线训练对模型更新不够灵活；在动态或未知环境中性能可能下降。

---

## 284. More with Less: a Large Scale Remote Sensing VLM with a Simple Recipe

**arXiv ID:** 2607.15942 | [PDF](https://arxiv.org/pdf/2607.15942v1)

**作者:** Stefan Maria Ailuro `[一作]` (INSAIT), Danda Pani Paudel `[通讯]` (INSAIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练了基于通用VLM的多任务遥感VLM MLRS，使用RL与SAM工具实现文本与定位输出。

**💡 创新点**

证明在遥感任务中仅靠数据与任务多样性即可匹配或超越专用架构，采用单一模型+工具而非专门设计架构。

**🔧 技术方法**

采用 InternVL 3.5‑8B + SAM3，利用多任务强化学习（GRTO/GRPO）与自适应奖励、LoRA 微调、组相对策略优化等技术。

**📊 数据集**

使用约 2.3M 张图像-文本对，涵盖 GeoZero、GeoLLaVA、SARLANG‑1M、DisasterM3、DynamicVL、EarthReason、LaSeRS、GeoSeg‑1M 等数据集，覆盖 RGB、SAR、多模态、多时相、多视角。

**📈 对比分析**

在 ID 与 OOD 基准上采用零样本评估，使用准确率、IoU、G‑Eval 等指标；MLRS 在大多数 OOD 基准中排名第一或第二，零样本性能接近或超越专用 RS‑VLM 与大型通用模型。

**⚠️ 局限性**

仅评估同一模型家族骨干，缺少与其他 VLM 或专用架构的完整对照；工具微调的泛化风险未完全消除；多样性测量粗略，需要更细粒度的训练混合实验。

---

## 285. Spatial Normalization for Cross-Domain Retinal Layer Segmentation in Optical Coherence Tomography

**arXiv ID:** 2607.16065 | [PDF](https://arxiv.org/pdf/2607.16065v1)

**作者:** Iker Moran-Cavero `[一作]`, Elena Garcia-Martin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了通过空间归一化（以黄斑为中心的对齐）来减轻光学相干断层扫描（OCT）跨域数据集中的几何域偏移，从而提升全层视网膜分割的鲁棒性与一致性。

**💡 创新点**

创新点在于首次将类似脑成像中使用的空间归一化方法引入OCT分割管线，并提出了无需人工标注的拓扑违规量化指标与厚度分析，用于评估分割的结构一致性。

**🔧 技术方法**

技术方面采用了三种主流深度学习架构（nnU-Net、MGU‑Net、SD‑LayerNet），并结合数据增强、空间归一化与半监督学习（SD‑LayerNet）进行训练与评估。

**📊 数据集**

实验数据集包括：训练集JHopkins（35个FastMac扫描，35 × 约 50 个B‑scan），测试集MServet（FastMac与PPole两种协议，涵盖健康对照及多种神经退行性疾病患者）。

**📈 对比分析**

通过Dice、拓扑违规计数与厚度误差等指标比较模型，结果表明：空间归一化显著提升所有模型性能；SD‑LayerNet在所有指标上持续领先，尤其在拓扑一致性与厚度误差上最优；nnU-Net在仅靠归一化时已能获得较好结果，需与数据增强共同使用才能与SD‑LayerNet竞争。

**⚠️ 局限性**

局限性主要在于归一化方法仅适用于含黄斑的视网膜扫描，对视盘或黄斑以外区域缺乏适配；依赖于黄斑点这一解剖标志，难以推广到无黄斑或病变严重遮挡的图像；此外训练样本仍相对有限，进一步提升泛化能力仍需更大多样化数据集。

---

## 286. Bidirectional Typing with Freezing, Skeletons, and Ghosts

**arXiv ID:** 2607.16061 | [PDF](https://arxiv.org/pdf/2607.16061v1)

**作者:** Wenhao Tang `[一作]` (University of Edinburgh), Bruno C. d. S. Oliveira `[通讯]` (University of Hong Kong)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种新的双向类型推断方法——FRE，能够对第一类多态（FCP）进行完整且可证明的类型推断；

**💡 创新点**

创新点主要包括：1) 通过骨架（Skeletons）与鬼（Ghosts）实现双向信息流，并使用颜色（Colours）与冻结（Freezing）精细控制信息流向；2) 在子类型中引入标签（Tags）追踪已确定的量化器，从而避免“无凭据多态猜测”；3) 设计了声明式的“look”机制来收集多态约束并局部求解；4) 提供了完整的证明框架（Soundness、Completeness）和可执行原型。

**🔧 技术方法**

所采用的技术包括：双向类型系统、骨架推断与鬼的扩展、颜色标记、冻结算子、标签化子类型、约束收集与求解、Look 机制、可扩展的算法实现，以及使用 Rocq 进行形式化验证。

**📊 数据集**

论文主要通过一系列手写示例（如 Church 编码、map、choose、auto、single id、run idlog 等）来验证方法的有效性，并未使用传统的公开数据集。

**📈 对比分析**

作者将 FRE 与现有 FCP 系统（PolyML、LTI、MLF、HML、Boxy、HMF、QML、GI、QuickLook、FCIF、FreezeML、SuperF、LCTI、ATIA 等）进行了对比，展示了在这些系统无法类型检查的示例中 FRE 成功通过；此外，提供了可执行原型，并演示了对模态效应类型的扩展，但未给出具体性能数值。

**⚠️ 局限性**

局限性包括：只能实现一次信息来回流，无法多次循环；鬼是无名的，导致无法捕捉同一鬼之间的关联；只支持浅子类型；不支持 let generalization、深层子类型、GADTs、higher‑kinded、交集/并集类型等高级特性；对未知多态量化的处理仍有限。

---

## 287. Gasp: A DeFi Application Specic Rollup as a Consolidation Layer for All Assets

**arXiv ID:** 2607.16052 | [PDF](https://arxiv.org/pdf/2607.16052v1)

**作者:** Stanislav Vozarik `[一作]`, Peter Kris `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于EigenLayer的主Rollup架构Gasp，构建了一个无桥跨链去中心化交易所，提供原生代币交换、MEV最小化、Proof‑of‑Liquidity和时间激励等功能，力图实现安全、低成本、高资本效率的全链资产流通。

**💡 创新点**

创新点包括：① 主Rollup多L1安全整合，利用EigenLayer重质押实现各链资产的L1级安全；② Ferry relayer机制与Rolldown组合，加速跨链转移并降低延迟；③ Themis Architecture将区块生成与执行分离并使用双重加密交易抑制VER/ VED，显著降低MEV风险；④ Proof‑of‑Liquidity将流动性质押与网络安全绑定；⑤ 时间激励曲线鼓励长期流动性提供。

**🔧 技术方法**

技术栈主要为：Optimistic Rollup、分布式Sequencer、EigenLayer重质押与证明、加密状态验证、双重加密交易、Ferry relayer、自动化流动性池与Vault、时间奖励曲线、Token锁定与治理机制。

**📊 数据集**

论文未公开使用具体数据集，实验与性能评估基于模拟与仿真，未给出公开数据来源。

**📈 对比分析**

通过与传统桥接、中心化交易所、原子交换等基准对比，理论上交易成本大幅下降、跨链延迟显著降低、MEV损失显著减少（如VER/ VED收益下降约30–50%）、资本效率提升（流动性深度提升、LP收益优化）。

**⚠️ 局限性**

局限性包括：① 依赖EigenLayer安全模型，若该层受损可能影响整体安全；② Ferry与Sequencer质押激励机制对网络规模与激励设计要求较高；③ 双重加密交易对可扩展性与执行成本有潜在影响；④ 跨链兼容性仍需实测，尤其在非EVM链上的实现细节；⑤ 治理模式相对集中（七人议会+三人监督），治理失误风险；⑥ 对低额交易存在锁定门槛，可能限制小额用户的使用体验。

---

## 288. Contextual Fraction on Permutation Gain Graphs: Exact Algorithms, Query Lower Bounds, and Dynamic Maintenance

**arXiv ID:** 2607.16037 | [PDF](https://arxiv.org/pdf/2607.16037v1)

**作者:** Ronald Katende `[一作]` `[通讯]` (Kabale University), Ronald Katende (Kabale University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究概率加权置换增益图模型，证明其上下文分数可用固定集合的质量计算，并给出最优的查询和动态维护算法；同时将支持阈值问题映射到有限域CSP，得到双分支可判定性。

**💡 创新点**

创新点在于将指数规模的线性规划简化为固定集合质量计算，得到 O(|O|(|V|+|E|)) 的最优查询复杂度；提出了在固定生成树下的子图插入/删除的 O(|O|) 动态数据结构；并将支持阈值问题与 Bulatov–Zhuk 的 CSP 二分法对应。

**🔧 技术方法**

使用了增益图的霍洛尼子群和固定点集合、线性规划对偶、树遍历与路径聚合、计数器维护技术，以及 CSP 归约与多项式时间可约。

**📊 数据集**

未使用公开数据集，所有实验均为理论复杂度分析与构造性证明。

**📈 对比分析**

与传统指数规模线性规划相比，算法在最坏情况下实现 O(|O|(|V|+|E|)) 的算术与表操作复杂度；查询时间为 O(1)；动态更新在子图插入/删除时仅需 O(|O|) ；下界证明表明在显式表模型下这一复杂度是信息理论上最优的。

**⚠️ 局限性**

局限性包括：只能在固定生成树下处理子图更新，树边更新未给出高效方案；仅适用于二进制常边缘可实现的 CSP 语言；仅讨论支持阈值问题，未给出非阈值下的精确值或近似算法；未考虑随机化或压缩表示的复杂度。

---

## 289. Constrained Hebbian Learning Supports Efficient Representational Allocation under Structural Constraints

**arXiv ID:** 2607.16027 | [PDF](https://arxiv.org/pdf/2607.16027v1)

**作者:** Patrick Inoue `[一作]` (Albstadt-Sigmaringen University), Andreas Knoblauch `[通讯]` (Albstadt-Sigmaringen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在生物学约束下，局部兴奋性 Hebbian 规则如何通过竞争和非负约束实现稀疏、低成本的表示，并与传统 BP 与 DDTP 进行对比。

**💡 创新点**

提出了将 Variational Information Bottleneck (VIB) 作为衡量表示压缩和任务相关信息的生物学可解释度量（CTI），并将 Hebbian 规则与 BP/DDTP 在相同稀疏、非负条件下直接比较，展示了 Hebbian 能在保持功能性能的前提下降低 CTI。

**🔧 技术方法**

使用了 Oja‑型 Hebbian 更新（带非负约束与激活归一化）、Variational Information Bottleneck、稀疏剪枝（dense‑to‑sparse）以及传统 BP、DDTP 等学习规则。

**📊 数据集**

在三组视听基准上验证：AVE、Kinetics‑Sounds 与 VGGSound100（每个样本已预先使用 VideoMAE 与 AudioMAE 产生嵌入）。

**📈 对比分析**

采用两阶段训练：先用对应规则学习可固定的 MLP 隐层表示，再在冻结的表示上训练 VIB 解码器，计算 Top‑1 准确率与 CTI；结果显示 Hebbian 在多数配置下实现了与 BP/DDTP 相当或略低的准确率，同时显著降低了 CTI，表明更高的能效/信息压缩。

**⚠️ 局限性**

局限性包括：只能在预先固定的高层特征上验证，未对原始像素/波形进行端到端训练；Hebbian 规则在更深网络（10 层）下表现不佳；CTI 仅为信息理论代理，未直接测量真实代谢能耗；非负约束与全局归一化可能不完全符合神经生理；实验仅覆盖 MLP 结构，未探索卷积、递归或 spiking 网络。

---

## 290. Presentation, Not Mechanism: A Render Confound in Deprecation-Aware Memory Evaluation

**arXiv ID:** 2607.16019 | [PDF](https://arxiv.org/pdf/2607.16019v1)

**作者:** Zhaoyang Jiang `[一作]` (University of Glasgow), Honghan Wu `[通讯]` (University of Glasgow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在持续更新的记录（如 GitHub 议题、维基百科修订等）上，提出并验证了“Evidence‑State Revision”（ESR）框架，用以判定当前有效的主张、已被推翻的主张以及何时应拒答；并对三种记忆实现（无失效标记的平面检索、粗粒度失效边缘化、细粒度带关系类型的账本）进行了系统对比。

**💡 创新点**

核心创新在于：①通过“render‑matched”对照实验剖析了结构化记忆中显著提升的真正来源——排版和提示布局，而非记忆机制本身；②证明对“snapshot”查询（仅需当前状态）而言，二元失效标记已是最优的最简记忆粒度；③指出在“provenance”查询（需访问已被推翻的历史）时，提升的关键是对失效证据的保留，而非细粒度关系类型；④构建了 ESR‑Bench 大型评测基准，填补了针对高噪声、持续修订流的评测空缺。

**🔧 技术方法**

使用的技术包括：
- 记忆实现：GraphRAG‑style 的无失效标记（d‑blind）、Zep/Graphiti 的粗粒度失效边缘化（coarse‑d）、自定义细粒度账本（fine‑d）;
- LLM：DeepSeek‑V4‑flash（推理）和 DeepSeek‑V3／MiniMax‑M2.5（判定）;
- 评测方法：两步一致性过滤、双人 LLM 判定、渲染匹配对照、噪声注入实验、保留状态单独诊断；
- 解析与抽取：原始记录转化为 (entity, attribute, value, polarity, time, source) 形式的原子。

**📊 数据集**

使用的主要数据集：
- ESR‑Bench 主数据（2,907 题目，来源为 GitHub issue、multi‑repo issue、Wikipedia revisions、DyKnow 109 条时间流）；
- 辅助数据集（1,198 题目 GitHub、multi‑repo、Wikipedia 低噪声版本）；
- 公开的 TempLAMA Wikidata 版本用于验证保留阈值；
- 以上均已去除原始文本，只保留事件 ID、抽取的原子和标签，符合隐私最小化要求。

**📈 对比分析**

对比结果显示：
- 在“reverted‑revert”（中间主张被后续推翻）场景中，粗粒度失效（coarse‑d）比细粒度账本（fine‑d）表现更好，差距约 +0.26；
- 渲染匹配实验表明，细粒度账本在“snapshot”任务中的显著提升（+0.18）几乎全因布局，而细粒度机制残差仅 +0.025，几乎为零；
- 在“provenance”任务中，粗粒度保留状态模型（即保留失效边缘但不带关系类型）已能恢复大部分细粒度账本优势；细粒度关系类型仅在读取被失效状态的坐标时才产生小幅提升；
- 控制噪声实验确认在任何可测噪声水平下，粗粒度失效始终优于或等价于细粒度账本。

**⚠️ 局限性**

局限性包括：
- 评测仅覆盖单一记忆实现族（GraphRAG‑style、Zep/Graphiti、细粒度账本），未对其他图形/存储框架做交叉验证；
- 依赖 LLM 判定，存在判定一致性低（κ≈0.46）导致稀有类别噪声较大；
- 采用的抽取器假设为完美提取，实际覆盖率约 87%，可能影响对细粒度机制的评估；
- 仅考虑单值键（entity‑attribute），未处理多面相（如平台/版本）导致的细粒度状态；
- 对外部基准（如 TempLAMA）验证仅局限于精确匹配情形，未涵盖高文本复杂度的真实对话场景。

---

## 291. BayesPO: Bayesian Prompt Optimization via Parallel-Tempered Gradient-Guided Discrete MCMC

**arXiv ID:** 2607.16001 | [PDF](https://arxiv.org/pdf/2607.16001v1)

**作者:** Junjie Zhou `[一作]` (Tsinghua University), Zhijian Ou `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于贝叶斯后验采样的提示优化框架BayesPO，利用梯度引导的离散MCMC并结合平行温度来搜索最佳提示。

**💡 创新点**

将提示优化转化为能量模型的后验采样，设计了Gibbs-with-Langevin离散提议和非权重共享嵌入的处理，并引入平行温度以突破局部最优。

**🔧 技术方法**

能量模型、梯度引导的离散MCMC、Metropolis-Hastings校正、Gibbs-with-Langevin采样、平行温度（PT）以及在Qwen2.5-Instruct上评估。

**📊 数据集**

诊断性语义转换任务（古文翻译与反义生成）、诗歌完成任务（李白《静夜思》）以及APE指令诱导基准的24个子任务。

**📈 对比分析**

与基准自动提示优化方法APE比较；在诊断任务中能量下降并产生语义正确提示；在诗歌任务中PT成功跳出局部最优；在APE基准上平均准确率从60.04%提升至63.23%，部分任务提升超过50%。

**⚠️ 局限性**

能量最小化可能在样本极少时过拟合导致准确率下降；采样过程计算量大，速度比APE慢约十倍；固定提示长度限制了探索空间；仅在单机GPU上实现，难以大规模部署。

---

## 292. A Formally Grounded ODRL Evaluator: Implementation and Comparison

**arXiv ID:** 2607.15987 | [PDF](https://arxiv.org/pdf/2607.15987v1)

**作者:** Jaime Osvaldo Salas `[一作]` (University of Southampton), George Konstantinidis `[通讯]` (University of Southampton)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文实现了首个基于正式数学语义的 ODRL 评估器，并提出了一种单遍算法，支持所有 ODRL 规则类型（权限、禁止、义务、职责、后果、补救）以及在线监控与访问控制场景，提供了可扩展的实现和性能评估。

**💡 创新点**

创新点包括：
- 采用 Salas 等人最新的查询评估语义，填补 ODRL 语义缺失；
- 设计单遍评估算法，能够在常数时间内处理事件流，实现在线监控与访问控制的统一框架；
- 通过评估状态对象实现对后果、补救的实时跟踪；
- 公开源代码、测试套件和可扩展的评估框架，促进社区复现与对比。

**🔧 技术方法**

技术手段：
- 基于 RDF/OWL 的 ODRL 语义模型；
- 使用 Python/CSV 进行事件流读取与排序；
- 实现单遍匹配与计数器（matches_count、required）机制；
- 提供评估状态对象用于流式评估；
- 与现有评估器（ODRE Framework、ODRL-PAP、ODRL-Manager、MOSAICrOWN 等）对接。

**📊 数据集**

数据集：
- 通过自定义生成器产生合成 ODRL 政策与状态（依据权限、禁止、义务、职责、约束等参数）；
- 使用 SolidLab ODRL Test Suite 中的标准测试用例进行对比与验证。

**📈 对比分析**

比较方法与性能：
- 在 SolidLab Test Suite 的 68 条测试用例上，测量评估时间（ms），并在对数尺度下绘图；
- 与 ODRE Framework（以及其他评估器）比较，发现 OVAL 在大多数案例中平均 30 ms，而 ODRE Framework 需 2600 ms；
- 对极端大逻辑约束（>100 条 datetime 约束）的案例，OVAL 约 150 ms，ODRE 约 27 s；
- 通过实验展示 OVAL 对规则数、事件数、职责数、约束数的线性时间复杂度，整体性能相较现有系统提升 2~3 个数量级。

**⚠️ 局限性**

局限性：
- 当前仅实现了大部分 ODRL 2.2 语义，尚缺少义务后果的完整支持；
- 不支持集合与成员关系约束（set, membership）以及特殊运算符；
- 目前不包含推理功能，无法处理需要知识推断的场景；
- 对极端复杂的逻辑约束仍可能导致性能下降；
- 依赖 CSV 读取，未对大规模分布式存储进行优化。

---

## 293. From Plausible to Actionable: A Position on LLM Self-Explanations

**arXiv ID:** 2607.15957 | [PDF](https://arxiv.org/pdf/2607.15957v1)

**作者:** Elize Herrewijnen `[一作]`, Fosca Giannotti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

演示了如何使用ACL风格文件与LuaLaTeX或XeLaTeX编译文档，提供了示例代码与多语言文本展示。

**💡 创新点**

提供了多语言文本处理的样例，展示了ACL模板在不同脚本中的应用。

**🔧 技术方法**

使用LuaLaTeX或XeLaTeX编译器与ACL LaTeX模板。

**📊 数据集**

无具体数据集，仅为示例文档。

**📈 对比分析**

未进行实验比较，未报告性能指标。

**⚠️ 局限性**

缺乏实验验证，示例性说明无法说明实际效果与限制。

---

## 294. Rendering 3D Gaussians on a Graph Processor

**arXiv ID:** 2607.15951 | [PDF](https://arxiv.org/pdf/2607.15951v1)

**作者:** Nicholas Fry `[一作]` (Imperial College London), Andrew J. Davison `[通讯]` (Imperial College London)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

实现了在Graphcore IPU上全局内存自由、仅使用SRAM的3D高斯渲染。

**💡 创新点**

创新在于将高斯原语的分发改为基于NEWS网格的显式近邻路由，并通过树形Bloom传递，充分利用局部性，避免DRAM访问。

**🔧 技术方法**

使用了IPU的BSP并行模型、POPlar编译器、手写codelet、EWA splatting、手工的路由协议与树形Bloom。

**📊 数据集**

使用了从Gaussian Splatting SLAM导出的真实世界RGB-D序列生成的3D高斯地图，如Pringles、Chairs、Bonsai、Salad、Sloth等。

**📈 对比分析**

与GPU（GTX1080、RTX4090）在同一帧率下比较，IPU帧率较低但功耗显著降低；在中等密度场景下渲染质量与GPU基线相近。

**⚠️ 局限性**

局限在于通道带宽饱和导致高密度区域出现瓦片化伪影、路由延迟对大视角跳变影响显著、单个Tile SRAM限制导致高密度场景丢帧。

---

## 295. Candidate Attended Dialogue State Tracking Using BERT

**arXiv ID:** 2607.16021 | [PDF](https://arxiv.org/pdf/2607.16021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 296. SciForge: An AI-Native, Multimodal Workbench for Scientific Discovery

**arXiv ID:** 2607.16038 | [PDF](https://arxiv.org/pdf/2607.16038v1)

**作者:** SciForge Team `[一作]` (Shanghai Artificial Intelligence Laboratory), Shuizhou Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 SciForge，一个本地优先的多模态科研 AI 工作台，集成模型路由、证据治理、协同决策与可执行工作流。

**💡 创新点**

结合多模态“翻译‑再推理”文件入口、线程/项目证据 DAG、可审计的发布门控和多角色协同治理，在单一环境下实现目标驱动、可追溯的科研自动化。

**🔧 技术方法**

使用 OpenAI/Anthropic/DeepSeek LLM 路由器、Es m2Text/Prot2Text/BioT5+/Cell2Sentence 翻译器、PROV‑JSON 证据 DAG、MCP 工具服务、Codex/ClaudeCode 代理、Git+工作流 DAG、本地存储与容器化工作者等技术。

**📊 数据集**

通过示例演示多条真实科研工作流，包括 430 条 antiSMASH BGC、MCPST 空间转录组、3 个 PDB 的蛋白接触预测、376 体的 EGFR 分子优化、80–100 体的蛋白设计、35 条 CRISPR 及公开 GEO 数据等。

**📈 对比分析**

论文未提供与现有系统的基准对比，主要通过案例演示展示可审计证据、可复现代码和可视化结果；示例中如 BGC 发现平均评分 2.75/5、蛋白设计 Boltz‑2 信心 0.94、EGFR 拓扑优化提升 1.7 kcal/mol 但未达预设阈值。

**⚠️ 局限性**

仅支持四类文件模式；翻译器输出仅为证据候选；证据 DAG 依赖手动捕获；IM 协作未完成；无系统性基准、用户研究和多模态扩展；实验仅为模拟，缺乏真实实验验证。

---

## 297. A 140-GHz Direct Raised-Cosine Envelope-Shaping Transmitter with Integrated ILO Phase Shifter

**arXiv ID:** 2607.15994 | [PDF](https://arxiv.org/pdf/2607.15994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 298. Network-Induced Strategic Communication in Opinion Dynamics

**arXiv ID:** 2607.16036 | [PDF](https://arxiv.org/pdf/2607.16036v1)

**作者:** Hassan Munif `[一作]` (Université de Lorraine), Tamer Başar `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于网络博弈的战略性公共广播模型，将私人意见到公共信号的映射视为博弈均衡对象，统一推导出线性平均、饱和表达和量化表达三种经典传播规律；

**💡 创新点**

创新点在于通过网络诱导的夸张因子将网络拓扑与个体激励耦合，证明博弈可降维为独立的标量“廉价谈话”问题，并从博弈角度给出量化的可信量化策略；

**🔧 技术方法**

主要技术包括博弈论的完全贝叶斯均衡分析、代价函数的二次优化、标量化简化、区间量化与阈值方程求解，以及定理证明与数值仿真；

**📊 数据集**

数据集为 Zachary Karate Club 网络和自构造的两组五节点网络，用统一先验和随机初始化意见进行仿真；

**📈 对比分析**

与传统线性平均、饱和信号以及无策略基准相比，本文的量化策略能够保持意见聚类、抑制极端化，且在高夸张因子下逼近二值化传播，表现出更稳健的群体共识与分化动态；

**⚠️ 局限性**

局限性包括缺乏对动态量化均衡的完整收敛理论、仅考虑对称先验和单一信号空间、以及对参数敏感性与大规模网络的可扩展性尚未充分验证。

---

## 299. Revisiting data-driven dynamic security assessment with a tabular foundation model

**arXiv ID:** 2607.16031 | [PDF](https://arxiv.org/pdf/2607.16031v1)

**作者:** Olayiwola Arowolo `[一作]` (Delft University of Technology), Jochen Cremer `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出使用Tabular Foundation Model (TabPFN) 进行预故障动态安全评估，允许单模型评估多种故障并支持零样本或少样本的未见故障推断。

**💡 创新点**

创新点在于：①将预训练的基础模型迁移至电力系统安全评估；②利用电气距离坐标(EDC)和改进版(mod‑EDC)作为连续场景向量实现对未见故障的泛化；③展示在极少标签样本下的样本高效性，甚至可与完整标签的迁移学习模型相当。

**🔧 技术方法**

核心技术包括：Tabular Foundation Model (TabPFN)、in‑context learning、EDC编码与mod‑EDC编码、基准机器学习模型(XGBoost、MLP、DT、SVM)、多标签学习MDSA‑S、联合域适应(JDA)用于对比。

**📊 数据集**

实验使用IEEE 68‑bus系统，22条典型故障，每条故障120k条稳态特征样本，随机生成的操作条件与负荷分布，构成训练/测试集。

**📈 对比分析**

与基准模型比较，使用宏F1、平衡准确率、特异性、精确率等指标。TabPFN在仅120个标签样本时即可达到约90%宏F1，显著优于XGBoost (+7%)，并在少样本零样本情形下，10个标签样本即可匹配最优迁移学习模型。多故障情况下单模型优于多标签学习MDSA‑S，且在类别不平衡的故障上保持较高特异性。

**⚠️ 局限性**

局限性包括：①模型输入上限为100k样本/2000特征，对大规模系统可能受限；②若故障数量众多，仍需为每个故障单独实例化TFM以提升效率；③无法保证绝对准确，输出需由运维人员进一步验证；④在极高维度或复杂拓扑变更时，计算成本和显存占用仍是挑战。

---

## 300. CanonicalPhys: Pose-Robust Remote Photoplethysmography via Canonical-Space Priors

**arXiv ID:** 2607.15995 | [PDF](https://arxiv.org/pdf/2607.15995v1)

**作者:** Hui Wei `[一作]` (University Of Oulu), Guoying Zhao `[通讯]` (Ellis Institute Finland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CanonicalPhys 通过四点单应性将视频帧映射到面部标准坐标系，从而在不增加参数的前提下实现姿态鲁棒的远程光电图估计。

**💡 创新点**

创新点在于把头部姿态视为坐标变换问题，利用无学习参数的单应性对齐四个面部锚点，并在此坐标系下引入三种物理先验（Lambertian 权重、跨 ROI 时序一致性、POS 先验蒸馏）。

**🔧 技术方法**

采用 MediaPipe 预训练的面部标记检测，四点单应性、Lambertian 权重、跨 ROI 一致性损失、POS 蒸馏等技术，并使用 FactorizePhys 作为骨干网络。

**📊 数据集**

在 UBFC‑rPPG、PURE、OBF、MMPD 四个公开 rPPG 数据集上进行训练和评估。

**📈 对比分析**

与 FactorizePhys 等基线对比，CanonicalPhys 在姿态分层的 MMPD 测试中将大角度 MAE 下降率从 1.60× 降至 1.33×，在跨数据集测试中最高可提升 32% 的 MAE，整体指标均优于或等同基线。

**⚠️ 局限性**

局限在于对极端偏转（>60°）及远离前额/双颊区域的遮挡无效，且在运动剧烈或低帧率数据中辅助先验可能导致过度正则化。

---

## 301. Embodied Active Learning under Limited Annotation and Navigation Budget for Object Detection

**arXiv ID:** 2607.15974 | [PDF](https://arxiv.org/pdf/2607.15974v1)

**作者:** Hadrien Crassous `[一作]` (Inria Scool), Riad Akrour `[通讯]` (Inria Scool)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种在机器人导航和标注预算受限下，利用空间不一致性进行自适应的目标检测的体化主动学习方法。

**💡 创新点**

将批量主动学习与机器人自主导航结合，提出预测不一致度作为无需外部监督的难度指标，用于引导导航和样本选择。

**🔧 技术方法**

采用预测不一致度评估、基于路径的贝叶斯优化导航、YOLOv5目标检测、ResNet特征聚类等技术。

**📊 数据集**

使用AI2-THOR（ProcTHOR）模拟房屋布局以及真实室内场景收集的数据，YOLOv5预训练于COCO数据集。

**📈 对比分析**

与Entropy、Count、Random及Oracle评分、FBE、Sweep、Random Walk导航等基准进行对比，实验显示预测不一致度与自适应导航在相同预算下显著提升mAP，接近Oracle性能。

**⚠️ 局限性**

依赖检测器预测质量，不能有效处理定位误差；导航预算有限导致覆盖不足；仍需人工或基础模型进行标注；未覆盖完全无监督或自监督适配方案。

---

## 302. Code-Poisoning Property Inference Attacks

**arXiv ID:** 2607.15970 | [PDF](https://arxiv.org/pdf/2607.15970v1)

**作者:** Xukun Luan `[一作]` (Beijing Institute of Technology), Jinyan Liu `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于代码污染的属性推断攻击（CPPIA），利用恶意代码在训练阶段将训练集的统计属性编码进模型对一组秘密样本的输出，随后通过仅返回标签的黑盒接口泄露属性信息。

**💡 创新点**

创新点在于：①首次将代码级污染与属性推断结合；②实现 100% 的攻击准确率；③不影响模型的原始准确率；④计算开销极低，仅需查询少量样本；⑤对现有防御（Inf2Guard、ExpM、DPSGD）均保持高效。

**🔧 技术方法**

核心技术包括：在恶意代码中插入统计分析函数（get_property）、秘密样本生成与标签编码、判别层（discriminative_layer）驱动模型记忆秘密标签；攻击流程仅需对已部署模型进行标签查询和解码。

**📊 数据集**

使用四个公开数据集：Adult、Census、Bank Marketing、CelebA，评估了八种模型结构和十八种属性；此外还在 GAN、Transformer、Stable Diffusion 3 等任务模型上做了初步验证。

**📈 对比分析**

与 SNAP、Mahloujifar、Tian 等基线对比：CPPIA 在所有实验设置下均达到 100% 的攻击准确率，且模型准确率保持不变；相较于 SNAP 的高计算成本和对防御的易失效，CPPIA 仅需极少的查询，且对三种主流防御均无影响。

**⚠️ 局限性**

局限性包括：①攻击在真实环境中部署难度较大，需将恶意代码推送至公共仓库或利用编码代理；②仅能泄露事先设定的属性；③对代码审计的依赖性高，若审计技术成熟则易被识别；④未针对更复杂的编码或多属性同时泄露的场景做深入研究。

---

## 303. CLaC@FinMMEval 2026 Task 3: Sentiment-Augmented Deep Reinforcement Learning for Active Trading -- An Alpha-Reward Approach

**arXiv ID:** 2607.16028 | [PDF](https://arxiv.org/pdf/2607.16028v1)

**作者:** Andrei Neagu `[一作]` (Concordia University), Leila Kosseim `[通讯]` (Concordia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于深度强化学习并结合新闻情感分析的日常交易决策系统，用于比特币和特斯拉的交易。

**💡 创新点**

提出了α奖励机制、随机起始日训练和多模态特征融合，以及使用零射击LLaMA 3.2 1B生成每日情感分数。

**🔧 技术方法**

使用四种DRL算法（PG、PPO、DQL、DDPG）并通过Ray Tune进行超参搜索；利用技术指标、循环日历编码和LLM情感特征构建状态；采用Alpha奖励与离散/连续动作转换。

**📊 数据集**

利用Yahoo Finance的历史价格与量化数据、Brian Ferrell金融新闻多源数据、CLEF Task 3官方行情与新闻，训练集至2022年，验证集2023-2024，测试集2025后。

**📈 对比分析**

以验证集Sharpe比率为模型选择标准；最终测试中DDPG在TSLA获得54.96%累计收益（SR 1.44），在BTC获得1.58%累计收益（SR 0.23），均显著优于买卖持有基准。

**⚠️ 局限性**

模型在不同市场周期（牛市 vs 熊市）表现差异大，验证集单一周期导致泛化缺失；交易费用仅在训练时引入；未使用SEC文件等长周期信息。

---

## 304. When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis

**arXiv ID:** 2607.16062 | [PDF](https://arxiv.org/pdf/2607.16062v1)

**作者:** S. Aaron McClendon `[一作]` `[通讯]` (Aimpoint Digital Labs), S. Aaron McClendon (Aimpoint Digital Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在AppWorld RL基准上对比了合并两名独立训练的RL专家与在同一数据集上联合多任务训练的模型，并使用TIES、RAM+等合并方法进行评估。

**💡 创新点**

创新点在于首次在同一数据集上验证RL合并与联合训练在主要指标上无显著差异，并通过几何诊断解释支持/符号导向合并方法表现相近的机制。

**🔧 技术方法**

技术手段包括LoRA低秩适配、LOOP（RLOO）策略、TIES与RAM+合并算法、全局余弦相似度与支持重叠度量、bootstrap CI与McNemar检验。

**📊 数据集**

使用的数据集为AppWorld互动式数字任务基准，包含168个任务，按难度划分为d1、d2、d3。

**📈 对比分析**

比较方法采用TGC、SGC、连续分数，配对McNemar和Wilcoxon检验；结果显示合并模型与联合训练在TGC上统计等价，但在连续分数上合并略低于单任务专家。

**⚠️ 局限性**

局限性包括仅使用单个种子、单一8B模型、仅两个专家与单一基准，且连续分数对跨任务能力提升不敏感，合并与联合训练在更大规模或更复杂场景下可能产生不同表现。

---

## 305. From Patterns to Parsers: Automatic Generation of Efficient Hardware Parsers for FPGAs

**arXiv ID:** 2607.16058 | [PDF](https://arxiv.org/pdf/2607.16058v1)

**作者:** Tushar Garg `[一作]` (Meta Platforms, Inc.), Andrew Boutros `[通讯]` (University of Waterloo)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款开源工具，通过解析中间表示（PIR）自动生成高效可读的FPGA硬件解析器，实现从高层规范到RTL的完整流程。

**💡 创新点**

①引入可自定义符号令牌支持范围、否定、外部端口比较等非等值匹配；②采用层次化模式分解和多周期数据路径拆分提升资源与时序；③生成稀疏的数据提取多路复用，显著降低面积。

**🔧 技术方法**

使用PIR与Jinja2模板驱动RTL生成，层次化分组匹配与多周期拆分，定制化符号令牌，静态稀疏多路复用；前端用Python编译P4或Snort规则；后端输出可读的SystemVerilog。

**📊 数据集**

以太网协议的P4程序（包含五元组提取）和10条Snort规则集为输入；同时使用4,096个80位模式的合成基准。

**📈 对比分析**

与学术基线（如P4FPGA、HLS生成器）在AMD Virtex‑7上对比，频率提升最高226%，逻辑资源下降97%，时延下降60%；在更宽数据路径的Altera Agilex‑5和AMD Versal上验证子线性资源增长；层次化设计比扁平设计可将LUT降至1/8并将Fmax提高约1.5×。

**⚠️ 局限性**

自定义符号令牌无法跨组拆分，限制了对宽模式的支持；目前仅支持等值、范围、否定、集合成员等有限谓词；不支持正则表达式、流态跟踪等高级功能。

---

## 306. Loop the Loopies!

**arXiv ID:** 2607.16051 | [PDF](https://arxiv.org/pdf/2607.16051v1)

**作者:** Zitian Gao `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Loopie 系列循环 MoE 语言模型，并通过 layer‑loop 递归在固定预训练算力下实现高效深度计算。

**💡 创新点**

创新点在于 Loopie Recipe：在相同算力下联合调节存储宽度、深度与循环深度，使循环 Transformer 与普通 Transformer 在同等计算预算下竞争；以及证明 layer‑loop 递归优于传统 model‑loop。

**🔧 技术方法**

使用的技术包括 Mixture‑of‑Experts Transformer、layer‑loop 递归、compute‑matched 调度（Loopie Recipe）、大规模 Supervised Pre‑Training、基于 GSPO 的强化学习（Math RL、Code RL）以及多种数据预处理与硬件友好的训练策略。

**📊 数据集**

主要数据集为 Nemotron‑CC‑v2‑HQ 及其高质量子集（数学、代码、STEM、Web、Synthetic 等），总计约 1.26 T token；后期使用 AIME、IMO、IPhO 等专业竞赛数据进行评估。

**📈 对比分析**

与 compute‑matched vanilla Transformer 基线相比，Loopie 在 MMLU、ARC‑Challenge、BBH、IFEval 等基准上平均提升约 10 分；在 AIME、IMO、IPhO 等任务中获得金牌级成绩；在多规模阶梯实验中，Loopie 的优势随规模扩大，表现出可扩展性。

**⚠️ 局限性**

局限包括：只在两步循环（R=2）下实验，未系统探究更深循环或推理时计算效率；对齐、人类指令或多任务能力的评估不足；缺乏推理时计算匹配与优化的研究。

---

## 307. DELUGE: Towards Continental-Scale Daily Pluvial Flood Damage Prediction via Interpretable Conditioning on Foundation Model Embeddings

**arXiv ID:** 2607.16050 | [PDF](https://arxiv.org/pdf/2607.16050v1)

**作者:** Yuya Kawakami `[一作]` (University of California, Davis), Tom Corringham `[通讯]` (Scripps Institution of Oceanography, University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个利用视觉与文本基础模型嵌入进行可解释条件化的连续尺度日常降雨洪灾损失预测框架。

**💡 创新点**

首次将视觉与文本基础模型嵌入作为条件特征，引入可解释的条件化机制，实现跨大陆范围的高精度洪灾损失预测。

**🔧 技术方法**

采用Transformer与图神经网络相结合的混合架构，利用CLIP、Vision Transformer等预训练模型生成嵌入，并通过可解释注意力机制进行条件化。

**📊 数据集**

使用了全球降雨量、NOAA洪灾损失记录、Sentinel-2卫星影像等多源数据，构建了大规模大陆级洪灾样本库。

**📈 对比分析**

与传统机器学习基线（XGBoost、Random Forest）及最近的深度学习模型（FloodNet、DeepFlood）相比，DELUGE在均方误差降低约15%，R²提升至0.78，在测试集上表现最优。

**⚠️ 局限性**

依赖高质量嵌入和多源数据，导致在数据稀缺地区预测精度下降，模型推理时间较长，且对极端天气事件的泛化能力仍有限。

---

## 308. Growing Hypergraphs with Homophily

**arXiv ID:** 2607.16046 | [PDF](https://arxiv.org/pdf/2607.16046v1)

**作者:** Violet Ross `[一作]` (University of Colorado at Boulder), Philip S. Chodrow `[通讯]` (Middlebury)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 L‑HCM（Label‑Aware Hyperedge Copy Model）的生成模型，用以描述带二值标签的超图在边复制、已存在节点加入与新节点加入三步中的同质性（homophily）机制，并在此基础上进行参数推断与社区检测。

**💡 创新点**

创新点在于：①打破传统超图生成模型中条件独立假设，显式建模边间相互依赖；②将节点标签信息融入复制概率与节点加入率，从而生成可调的高阶同质性结构；③推导出边度分布的幂律指数与边内标签联合分布的稳态特性；④基于该模型实现了可扩展的随机期望最大化（SEM）参数估计与通过模拟退火的最大似然社区检测。

**🔧 技术方法**

采用的技术包括：生成式模型设计、线性映射与特征值分析推导稳态分布、随机期望最大化算法（含在线统计量更新与学习率衰减）、模拟退火优化（自适应接受调度与交叉求和近似）以及基于交叉投影的模数度量。

**📊 数据集**

使用的数据集包括：美国参议院与众议院共赞同案超图（标签为政党），高中与小学学生社交交互超图（标签为性别），学术会议作者-论文超图（标签为性别），Enron 邮件线程超图（标签为核心/边缘）。

**📈 对比分析**

与方法比较：在合成数据上，模拟退火在 Nishimori 条件下（已知真实参数）以及在固定参数下均优于基于加权团投影的拉普拉斯谱聚类；在实测数据上，模拟退火在部分数据集上达到或超过非独立假设的超图模数、Belief‑Propagation 非回溯谱聚类和传统图的贪婪模数最大化，表现尤为突出；但对规模较大的数据集计算量大，导致无法完成。

**⚠️ 局限性**

限制包括：①计算复杂度高，特别是模拟退火的 O(m²) 代价，难以扩展到大规模超图；②仅支持二值节点标签，无法直接处理多分类社区；③模型假设所有边均通过复制机制生成，忽略自发形成的边；④未考虑时间衰减或最新边的偏好，限制对动态演化的真实刻画。

---

## 309. PIXIE: A Zero-Shot texture-invariant 6D pose estimation framework for unseen objects with assembly defects

**arXiv ID:** 2607.16015 | [PDF](https://arxiv.org/pdf/2607.16015v1)

**作者:** Leon Jungemeyer `[一作]`, Daniel Werdehausen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了一份电子论文模板的使用规范，帮助作者在排版时保持一致性。

**💡 创新点**

创新点在于统一了多种排版细节（如边距、行距、标题样式等），并提供了详细的使用指导。

**🔧 技术方法**

使用技术主要是 LaTeX 模板及其 cls 文件。

**📊 数据集**

无特定数据集。

**📈 对比分析**

不涉及实验或比较，未给出性能评估。

**⚠️ 局限性**

仅适用于 US‑letter 尺寸，需要手动调整以适应 A4；模板对复杂表格或多级公式的支持有限。

---

## 310. App-Based Performance Characterization of Cellular and Wi-Fi Networks in Dense Stadium Deployments

**arXiv ID:** 2607.16008 | [PDF](https://arxiv.org/pdf/2607.16008v1)

**作者:** Hardani Ismu Nabil `[一作]` (Sebelas Maret University), Monisha Ghosh `[通讯]` (University of Notre Dame)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在诺特丹大学体育场（已售罄的观众高达77,622人）开展了基于实际用户设备的实验，系统地测量并评估了多载波4G/5G蜂窝网络与密集5/6 GHz Wi‑Fi在极端人群密度下的用户感知体验（QoE）和网络层性能。

**💡 创新点**

创新点在于首次提供一份全面的实测对比，量化了“上行缺口”与高峰期尾部延迟，并通过对比多运营商的5G NSA与SA架构、以及Wi‑Fi密集部署，揭示了在超密集环境下Wi‑Fi扩容和5G独立体系的重要性。

**🔧 技术方法**

技术手段包括：使用六台商业智能手机；通过QualiPoc采集低层的PHY/MAC KPI（PCI、RSRP、MCS、BLER等）；SigCap记录Wi‑Fi beacon及信道占用；Ookla Speedtest测量吞吐量；以及在浏览、WhatsApp和Instagram等实际应用层进行延迟、上传/下载时间等指标测量。

**📊 数据集**

数据集来自四个赛日（9/13、9/27、11/08、11/22）与三个空场日（8/26、8/28、2/26/26）的实验，累计约1.35 M QualiPoc样本和1.89 M SigCap beacon样本。

**📈 对比分析**

比较方法为在赛日与空场期间分别测量DL/UL吞吐、RTT、TTFB、浏览持续时间、会话失败率等指标，并对比三家运营商与Wi‑Fi（5 GHz/6 GHz）性能。实验结果显示：Wi‑Fi 6 GHz的会话失败率仅3.9%，延迟提升低于2.1×；T‑Mobile凭借5G SA和中频段在赛日仍保持最低TTFB（504 ms）与最高吞吐；AT&T与Verizon由于依赖EN‑DC导致上行拥塞，出现极端尾部延迟（最高5983 ms）和高失败率。

**⚠️ 局限性**

局限性包括：仅覆盖户外球场座席区域，实验设备限于六台手机；Wi‑Fi主动扫描导致吞吐下调；赛日与空场样本量有限；未考虑室内Wi‑Fi或后续网络硬件升级；实验假设硬件与网络架构在测量期间保持不变。

---

## 311. Robustness of Reinforcement Learning-Based Congestion Management in Low-Voltage Grids

**arXiv ID:** 2607.16004 | [PDF](https://arxiv.org/pdf/2607.16004v1)

**作者:** Josef Hoppe `[一作]` (RWTH Aachen University), Michael T. Schaub `[通讯]` (RWTH Aachen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个分两步的强化学习控制框架，用随机森林预分类器检测低压配电网的违规状态，再由演员-评论家 RL 控制器对可控单元进行节流，以实现拥堵管理。

**💡 创新点**

创新点在于将违规检测与控制操作解耦，利用随机森林显著降低 RL 学习复杂度；同时系统性评估了该框架对测量噪声和网格参数不匹配的鲁棒性，并提供了模块化的两步控制结构。

**🔧 技术方法**

核心技术包括：随机森林分类器、演员-评论家（DDPG）强化学习、AC-OPF 约束、功率流仿真、噪声和参数不匹配建模、深度学习框架（PyTorch/TorchRL）和 scikit-learn。

**📊 数据集**

使用来自 Schleswig-Holstein Netz GmbH 的真实低压配电网拓扑，结合 SimBench 未来情景生成 35,040 条操作点（含 PV、EV、热泵），随机分配 10% 的节点为可观测且可控；数据包含功率流结果作为基础。

**📈 对比分析**

通过与无行动基线、全局 OPF 参考以及未使用预分类器的端到端 RL 进行对比。结果显示：在准确网格参数下，违规幅度降低 98.9%，测量噪声不影响性能；在参数不匹配时仍可降低约 79.6%；与端到端模型相比，两步框架在相同训练步数下能减少更多违规、使用更少的功率并降低训练成本。

**⚠️ 局限性**

主要局限在于对网格模型不匹配的敏感性较高；假设观测/可控比例固定，未考虑拓扑不确定、负荷/发电动态变化；需要离线训练与周期性再训练；缺乏现场或硬件实验验证，需进一步评估通信延迟、监管要求等实际工况。

---

## 312. DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging

**arXiv ID:** 2607.15986 | [PDF](https://arxiv.org/pdf/2607.15986v1)

**作者:** Théophane Loloum `[一作]` (CEA), Julien Tierny `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

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

## 313. Vessel Trajectory Prediction using COLREGs-aware Optimal Planning

**arXiv ID:** 2607.15969 | [PDF](https://arxiv.org/pdf/2607.15969v1)

**作者:** David Kaikkonen `[一作]`, Erik Frisk `[通讯]` (Linköping University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用 A* 生成可行路径热身，并在此基础上构建最优控制问题，对周围船舶进行 COLREG 合规的轨迹预测，旨在支持海上船舶的安全路径规划。

**💡 创新点**

创新点在于：① 将轨迹预测视作从目标船舶视角的序列最优规划；② 仅依赖 AIS 当前位置、速度和目的地信息，避免长历史数据；③ 结合 A* 的初始解与 COLREG 禁区约束，实现可解释且实时的预测；④ 模块化框架可与现有运动规划器直接兼容。

**🔧 技术方法**

技术手段包括：A* 规划、CasADi+IPOPT 数值优化的非线性最优控制、COLREG 场景识别与禁区构造、凸优化的碰撞约束（通过多面体半空间表示）。

**📊 数据集**

使用 MarineTraffic 公共 AIS 数据，重建真实海上相遇场景进行仿真验证。

**📈 对比分析**

与恒速（CV）预测对比，RMSE 较低且在多种 COLREG 场景下能正确识别并避免冲突；预测速度足够实时，能够在目标船舶首次检测后迅速给出轨迹。

**⚠️ 局限性**

局限性：忽略风浪等外部环境影响；船舶动力学简化为基本运动学模型；需依赖 AIS 数据，无法处理无 AIS 船舶；在复杂多船相互作用情形下，COLREG 分类与禁区构造仍面临挑战。

---

## 314. TARS: A Theory-of-Mind Agent for Personalized In-IDE Code Comprehension

**arXiv ID:** 2607.15948 | [PDF](https://arxiv.org/pdf/2607.15948v1)

**作者:** Leopoldo Todisco `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Visual Studio Code 中实现了一个基于大语言模型的 AI 助手 TARS，能够在 IDE 内根据开发者的角色、经验和偏好，提供自适应且基于项目文档的代码解释；

**💡 创新点**

创新点在于将 Theory‑of‑Mind 视角与 Retrieval‑Augmented Generation 结合，构建一个轻量级的用户画像并直接在编辑器中展示上下文相关、个性化的解释；

**🔧 技术方法**

使用技术包括 GPT‑4.1 Nano、LangGraph 进行流程编排、RAG 进行项目文档检索、以及自定义的 ToM Profiler 问卷收集用户特征；

**📊 数据集**

实验数据集为 CodeSearchNet 代码库中的 4 篇 Java 片段（复杂度 10–30，已去除注释）；

**📈 对比分析**

通过 18 名受试者的交叉设计实验，对比有无 TARS 辅助，结果显示平均任务完成时间缩短 26%（相对标准差减半），主观认知负荷下降，且解释质量在客观与主观评估中均略优，但统计显著性不足；

**⚠️ 局限性**

主要局限包括样本量小导致统计功效不足、LLM 在遵循用户指定简洁度方面表现不稳定，以及评估仅覆盖小规模、人工标注的代码片段，尚未验证在真实大型项目中的泛化能力。

---

## 315. A Morphing-Designed Hexarotor Prototype combining Practical Resilience and Efficiency

**arXiv ID:** 2607.16002 | [PDF](https://arxiv.org/pdf/2607.16002v1)

**作者:** Murat Bronz `[一作]` (Ecole Nationale de l'Aviation Civile), Antonio Franchi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并实验验证了一种可变形六旋翼机Opti‑Hexa，实现了在一定角度范围内同时具备能效与单旋翼失效鲁棒性。

**💡 创新点**

首次在实验层面证明存在既高效又具鲁棒性的六旋翼机配置，并公开开源可变形平台。

**🔧 技术方法**

采用可变形机械臂、T‑motor F2203.5驱动、Papazara自主控制、INDI增量非线性动态反演等技术。

**📊 数据集**

收集了约27分钟悬停飞行电力与位置误差数据，以及在不同角度下单旋翼失效的多种实验记录。

**📈 对比分析**

通过对比星形(γ=0)与Y形(γ=π/6)的能耗与失效后位置误差，发现γ∈[π/30,π/18]区间内能耗与鲁棒性均优于两端配置。

**⚠️ 局限性**

仅验证了特定规模和动力系统的可变形平台，参数范围不一定适用于所有多旋翼尺寸或配置。

---

## 316. ArtChart: A Benchmark for Faithful Artistic Chart Generation with Integrated Text Rendering

**arXiv ID:** 2607.16060 | [PDF](https://arxiv.org/pdf/2607.16060v1)

**作者:** Meijia Huang `[一作]` (Ant Group), Chenguang Ma `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ArtChart 框架，实现艺术化图表生成并集成数值几何、文本渲染与标签绑定；并构建了 ArtChart‑Bench 2K 双语基准及六轴评测协议。

**💡 创新点**

创新点包括：① 专门的图表几何 ControlNet 使模型直接遵循灰度版图表结构；② 采用 Flow‑GRPO 的多奖励强化学习，分别优化 OCR 准确、文本布局和美学；③ 通过多专家蒸馏解决奖励冲突，兼顾三方面性能；④ 统一的基准与评测，为跨模型对比提供客观平台。

**🔧 技术方法**

核心技术包括：Qwen‑Image Diffusion 生成器、图表 ControlNet、DiT LoRA 适配器、Flow‑GRPO 强化学习、OPD 蒸馏、VLM+OCR 奖励与判定。

**📊 数据集**

训练数据：约 13K（P,G,I*）三元组用于 ControlNet；10K+（P,G）对用于 RL；Benchmark 包含 2K 题目（1K 英文 1K 中文），覆盖 bar/hbar/pie/area 四类图表、15 种艺术风格和多样化标签格式。

**📈 对比分析**

与 prompt‑only T2I、图像编辑、通用 ControlNet 基线进行对比，使用六轴评测（MathLogic、TextAcc、LayoutPos、Aesthetic、InstrFollow、Readability）。ArtChart 在所有维度上均优于基线，平均总分 9.05/10，特别在数学逻辑和文本布局方面显著提升。

**⚠️ 局限性**

局限性：依赖灰度图表条件，无法直接由纯文本推断几何；仅支持单维类别数据和四种图表；评估依赖 OCR/VLM 可能引入偏差；未覆盖堆叠、多系列、散点、雷达等复杂图表。

---

## 317. Reducing Power Consumption of Embedded Dynamic Memories with ECCs

**arXiv ID:** 2607.16042 | [PDF](https://arxiv.org/pdf/2607.16042v1)

**作者:** Wenqing Song `[一作]` (EPFL), Andreas Burg `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了如何利用边缘条件卷积（Edge-Conditioned Convolution，ECC）对图结构数据进行有效表示学习，并将其应用于多种图任务。

**💡 创新点**

创新点在于提出了基于自适应权重学习的ECC变体，能够在不需要显式手工特征工程的前提下，自动捕捉图中不同边属性对卷积核的影响，从而提升模型的表达能力。

**🔧 技术方法**

主要技术包括图卷积网络、注意力机制、参数共享以及自适应边特征加权；在实现上还结合了梯度剪裁和学习率调度来稳定训练。

**📊 数据集**

实验数据集涵盖了Cora、Citeseer、Pubmed以及更大规模的OGBN-Arxiv和Ogbn-Papers100M，确保了模型在不同规模和领域的通用性。

**📈 对比分析**

通过与传统GCN、GraphSAGE、GAT以及最近的基于Transformer的图网络进行对比，ECC在节点分类准确率上平均提升了约2.5%–3.8%，在图分类上也表现出更好的泛化能力。

**⚠️ 局限性**

局限性主要体现在：1）模型对大规模图数据的计算复杂度仍然偏高，尤其是边特征维度较高时；2）对稀疏图中边属性缺失的鲁棒性不足；3）在极端图结构（如超大直径或密集子图）下，参数共享可能导致信息丢失。

---

## 318. AI Watermark Evidence Fails Forensic Readiness: An Empirical Evaluation

**arXiv ID:** 2607.16010 | [PDF](https://arxiv.org/pdf/2607.16010v1)

**作者:** Saifur Rahman Tamim `[一作]` (Northern University Bangladesh), Amir Labib Khan `[通讯]` (Northern University Bangladesh)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了三种LLM水印（KGW、Unigram、SynthID）在法医可采性标准下的表现，构建了FRS框架并进行实验评估。

**💡 创新点**

将Daubert和NIST 800-86标准与可采性评分相结合，提出FRS评分体系并揭示点数评分隐藏的法医无用性。

**🔧 技术方法**

采用MarkLLM工具实现三种水印，使用Qwen2.5-1.5B和Gemma-2-9b-it模型，进行意义保持的重写攻击，并统计TPR/FPR等指标。

**📊 数据集**

生成15种多领域提示的30条水印文本，随后通过Qwen2.5-1.5B产生重写，过滤后共计数百个有效实验样本。

**📈 对比分析**

通过基线检测率、攻击后的条件删除率、误报率等指标比较三种水印，结果显示KGW、Unigram在重写后100%被删除，SynthID约98%，均表现不及法院可采标准。

**⚠️ 局限性**

仅评估单一攻击类型、模型规模有限、未覆盖生产级SynthID、未进行跨种子或多模型重复验证，且FRS仅基于预设门槛，缺乏更广泛的交叉验证。

---

## 319. Beyond Unfolding: 60x Faster One-Stage Unmixing for Closely-Spaced Infrared Small Targets

**arXiv ID:** 2607.16007 | [PDF](https://arxiv.org/pdf/2607.16007v1)

**作者:** Ximeng Zhai `[一作]` (Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences), Yimian Dai `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种轻量级单阶段CSIST目标拆分方法FOCUS，避免使用深度展开网络的迭代结构。

**💡 创新点**

创新点在于将超分辨率模型与CSIST拆分映射同构化，利用结构稀疏和能量守恒约束实现无迭代的高精度子像素拆分。

**🔧 技术方法**

采用端到端卷积网络，结合ISGU单元、粗细分层流、稀疏正则、能量守恒损失等技术。

**📊 数据集**

在CSIST-100K合成数据集上进行训练和评估。

**📈 对比分析**

与现有深度展开网络和改进SR模型进行对比，FOCUS在保持约45.5% mAP的同时实现约60倍的速度提升，FOCUS+在精度上进一步超越最强迭代基线。

**⚠️ 局限性**

仍受噪声强度和高目标密度场景的性能下降限制，需要在更真实的数据集上进一步验证。

---

## 320. Refusal is Not Safety! Benchmarking Latent Safety Risks of LLM-Driven Content Humorization

**arXiv ID:** 2607.15977 | [PDF](https://arxiv.org/pdf/2607.15977v1)

**作者:** Yu Cui `[一作]` (Beijing Institute of Technology), Cong Zuo `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对大型语言模型在生成幽默内容时可能隐藏的安全风险进行系统研究，提出 HumorSafe 评估框架和 HumorPIA 攻击手段，并在多模型上验证其效果；

**💡 创新点**

创新点在于首次将幽默化过程拆解为去幽默化（Unfun）与重幽默化（Refun）两阶段，评估模型在隐式幽默化中的安全风险，并构造针对幽默式安全防御的注入式攻击；

**🔧 技术方法**

采用两阶段不对称样本构造、上下文学习（ICL）、prompt injection 与幽默化融合的攻击，利用 GPT、Qwen、DeepSeek 等前沿 LLM 进行实验；

**📊 数据集**

使用 Unfun 数据集、Chumor 2.0 数据集、HarmBench+AdvBench、以及从真实代理交互日志抽取的 30k+ 记录和 45 位喜剧演员的问卷反馈；

**📈 对比分析**

与传统幽默生成基线对比，HumorSafe 在大多数模型上将毒性和刻板印象提升约 3–7 倍；HumorPIA 在无防御时毒性提升 5.5–12 倍，在配合幽默拒绝防御时仍能 3.14 倍提升毒性，且被 GPT‑5.5/Claude Opus 等 SOTA 检测器误认为安全；

**⚠️ 局限性**

局限性包括样本规模受限（仅 45 名喜剧演员）、评估仅覆盖部分 LLM 与数据集、未探究更细粒度的幽默安全标签与多模态场景。

---

## 321. A Human-Centric Evaluation of a Retrieval-Augmented Generation System for Explaining Quebec Insurance Contracts

**arXiv ID:** 2607.15963 | [PDF](https://arxiv.org/pdf/2607.15963v1)

**作者:** David Beauchemin `[一作]` (Université Laval), Richard Khoury `[通讯]` (Université Laval)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过用户实验对基于检索增强生成（RAG）的魁北克汽车保险合同解释系统进行人本中心的外部评估，衡量其对消费者理解、认知负担、自治感与风险感知的影响。

**💡 创新点**

创新点在于首次将大规模人本评估与RAG技术结合，揭示系统在低金融素养用户中的认知平衡作用，并强调在高风险金融场景下需采用人机协作的责任框架。

**🔧 技术方法**

技术方案采用OpenAI 2025‑04‑16 LLM配合检索模块，从专门的保险合同知识库中检索文本，再生成答案；并设计专用交互界面与结构化问卷。

**📊 数据集**

使用的数据集为魁北克地区标准汽车保险合同文本（经过法律法规整理的知识库），并通过学生/员工资格调查评估用户保险素养。

**📈 对比分析**

通过Likert量表收集系统满意度、认知努力、自治感、风险感知四个维度，使用t检验、Kruskal‑Wallis、Pearson相关等统计方法比较低/高素养组，结果显示低素养用户获得更高满意度，整体满意度、认知负担低、自治感高；未在传统任务指标上与基准直接对比，主要评估为用户体验。

**⚠️ 局限性**

局限性包括：样本来自大学群体，保险素养高，难以推广至更广泛、教育水平较低的消费者；实验时间短且在受控环境下进行，未模拟真实购买或理赔情境；仅评估一种RAG架构与魁北克保险合同，错误率与人机协作需求可能随法律、语言或LLM不同而变化。

---

## 322. When Not to Automate: A Formal Protocol for Human Preservation in AI-Optimized Organizations

**arXiv ID:** 2607.15944 | [PDF](https://arxiv.org/pdf/2607.15944v1)

**作者:** Jose Manuel de la Chica Rodriguez `[一作]` (Santander AI Lab), Spyridon Chouliaras `[通讯]` (Santander AI Lab)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了PHP‑AIO（Protocol for Human Preservation in AI‑Optimized Organizations）——一种基于五道门槛和综合检查的决策流程，用以在部署前评估自动化的四类无价系统风险（隐性知识流失、韧性下降、监管曝光、社会机构资本恶化），并给出可审计的自动化、保留、增强或混合四种结果；同时定义了自动化债务指标，衡量多步过程中的自动化密度和风险累积。

**💡 创新点**

创新点在于：①引入“自动化主权”概念，将四类长周期风险统一框架；②设计了可定量化的五道门槛和后续复合检查的 deterministic 协议，首次在部署前做出自动化决策；③提出时间索引和漂移率预测，预测未来风险变化；④定义了自动化债务量化公式，弥补单步骤安全导致整体过程脆弱的缺口；⑤通过 40 字段结构化输入实现可审计、可复现的评分路径。

**🔧 技术方法**

主要技术包括：权重加权和归一化公式、门槛阈值判定、复合平均加权、时间线性漂移预测、自动化债务密度计算、分层结构化数据模式以及版本化配置管理。

**📊 数据集**

使用的数据主要来自内部 Santander AI Lab 的专家共识（四个风险维度的权重、门槛、国家多重因子、漂移率等），以及对四个内部业务角色的示例性手工评分，未使用公开大规模实验数据。

**📈 对比分析**

在示例级别进行了敏感性与时间漂移分析，展示了门槛设定的稳健性与复合检查对风险累积的纠正；未与现有标准（如 NIST AI RMF、EU AI Act）直接对照，但通过对比表明 PHP‑AIO 在部署前提供了更细粒度的决策过滤。

**⚠️ 局限性**

主要局限包括：①参数和权重完全基于专家共识，未通过历史数据校准；②缺乏实证验证，尚无真实案例或多机构实验；③输入层的可靠性未评估（如二人评分一致性）；④配置仅适用于金融服务行业，需跨行业重新校准；⑤可配置性可能被用于“结果购物”，需要严格审计与版本控制来防范。

---

## 323. When Does Muon Help Agentic Reinforcement Learning?

**arXiv ID:** 2607.16169 | [PDF](https://arxiv.org/pdf/2607.16169v1)

**作者:** Kai Ruan `[一作]` (Renmin University of China), Hao Sun `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估 Muon 优化器在稀疏奖励的 Agentic RL（ALFWorld）中的表现。

**💡 创新点**

通过仅在隐藏矩阵上应用 Muon 并结合不同优势估计器，发现其在 GiGPO 下能将最终窗口成功率提升约 88%，并展示了优势估计器对 Muon 效果的显著影响。

**🔧 技术方法**

使用 Muon 的近似谱正则化（Newton–Schulz 迭代）配合 AdamW，实验对比 GRPO、GiGPO、GraphGPO 三种优势估计器，并采用 PPO‑style 剪切目标与 KL 正则。

**📊 数据集**

数据集为 ALFWorld（六类基于房屋任务），使用 Qwen2.5‑0.5B‑Instruct 语言模型。

**📈 对比分析**

与 AdamW 基线及不同学习率控制进行单种子匹配实验，比较最终窗口成功率、AUC 与阈值交叉时间；Mu 在 GiGPO 下提升约 0.26 成功率，GraphGPO 提升 AUC 0.16，GRPO 提升 0.11。

**⚠️ 局限性**

仅单一种子、单一模型、有限学习率覆盖；未直接测量梯度 SNR 或更新谱；全矩阵实现导致显存占用较大，需更广泛种子与环境验证。

---

## 324. Behaviour-Conditioned Neural Processes for Adaptive Residential Short-Term Load Forecasting

**arXiv ID:** 2607.16168 | [PDF](https://arxiv.org/pdf/2607.16168v1)

**作者:** Ramin Soleimani `[一作]` (University College Cork), Dirk Pesch `[通讯]` (University College Cork)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种行为条件的注意力神经过程（Attentive Neural Process）框架，用于住宅短期负荷预测。

**💡 创新点**

创新点在于将通过聚类得到的行为结构作为离散潜变量，直接在预测解码器中进行FiLM特征调制，实现无标签的行为感知与不确定性建模。

**🔧 技术方法**

使用技术包括注意力神经过程（ANP）、连续与离散潜变量、FiLM/HyperFiLM调制、Gumbel‑Softmax弱监督标签推断以及Transformer式解码器。

**📊 数据集**

数据集为澳大利亚Smart Grid, Smart City (SGSC) 试验的约400户住宅半小时能耗记录，构成两日104点窗口。

**📈 对比分析**

与无行为标签的ANP基线、固定窗口确定性方法以及多种传统回归/树模型对比，软标签版本在MAE、CRPS和RMSE上分别提升约7.9%、6.9%及12–18%，在有限上下文时效果最显著。

**⚠️ 局限性**

局限性包括使用聚类产生的代理标签而非真实行为类别、未评估缺失/不规则采样情况、以及对行为表示和调制方式的进一步探索空间。

---

## 325. CLIFE: Camera-LiDAR Fusion Framework for Edge-Deployable Roadside VRU Perception

**arXiv ID:** 2607.16154 | [PDF](https://arxiv.org/pdf/2607.16154v1)

**作者:** Tam Bang `[一作]` (University of Tennessee at Chattanooga), Mina Sartipi `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套全靠嵌入式设备完成的、实时的摄像头与LiDAR融合框架CLIFE，用于路边对易受伤害交通参与者(VRU)的感知与追踪。

**💡 创新点**

创新点在于：①实现了目标无关的在线标定，随时自动纠正摄像头–LiDAR误差；②采用轻量化的后期融合与追踪算法，复杂度仅为O(N log N)；③在不需要云端或服务器的情况下，完全在单一Jetson AGX Thor上完成全部流程，满足边缘部署与低延迟需求。

**🔧 技术方法**

使用了YOLOv11+ByteTrack实现摄像头检测与追踪，基于深度学习的目标无关标定网络提取空间、外观、语义特征并迭代优化相机–LiDAR同一平面单应矩阵；融合阶段利用KD‑tree做投影匹配、属性融合与多目标追踪；所有网络均通过TensorRT加速；后端采用BlueCity LiDAR API或PointPillars+AB3DMOT实现点云检测。

**📊 数据集**

数据集为Chattanooga市12个信号交叉口的实地采集：摄像头数据25,270帧、同步摄像头–LiDAR对9,000帧，覆盖晴天、阴天、细雨等多种天气；蓝色城市蓝本系统提供的LiDAR感知结果作为辅助标签；对稀有VRU类别（如轮椅、踏板车）进行了人工标注。

**📈 对比分析**

通过与单独摄像头、单独LiDAR两种基线对比，采用mAP、MOTA、IDF1评估；在晴天/阴天/细雨三种场景下，融合方法的MOTA分别提升至78.6/65.4/65.8、IDF1提升至86.0/71.0/71.2；实时性能达到53.2 FPS，单个Jetson节点可并行5条摄像头–LiDAR管道，平均约10 FPS/流，覆盖两个完整交叉口。

**⚠️ 局限性**

局限性包括：①对相机–LiDAR外参的精准度高度依赖，标定误差会直接影响匹配与追踪；②后期融合易将上游误检、误标扩散，无法完全纠正；③目前依赖商业BlueCity LiDAR API，限制了端到端优化与跨平台可复现性；④在极端天气（暴雨、夜间极光）下仍有性能下降；⑤未实现轨迹预测与风险评估，尚缺乏主动预警功能。

---

## 326. A Unified Theory of Sparsification

**arXiv ID:** 2607.16126 | [PDF](https://arxiv.org/pdf/2607.16126v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Madhu Sudan `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了实值图稀疏化的理论框架，给出了可压缩图的结构性特征，并提供了构造稀疏化算法。

**💡 创新点**

创新点在于用连续 VC 维度和 fat‑shattering 维度等非传统工具替代信息熵法，提出了连续不冗余度概念，实现了对一般实值图的近似最优稀疏化分析。

**🔧 技术方法**

主要使用了 Sauer–Shelah 引理、Chernoff 约束、随机采样、Natarajan 维度以及 fat‑shattering 维度的组合技术。

**📊 数据集**

本文不依赖具体数据集，所有结果均为理论上界。

**📈 对比分析**

与之前基于图结构的稀疏化方法相比，本框架在近似误差 ε 处取得了与最优相当的稀疏化尺寸，并且可在多种图类（0-1 切、k‑way 切、谱稀疏化、实值图）上得到统一的近最优分析。

**⚠️ 局限性**

局限在于对极端比率（最大/最小权重比）高的实值图需要额外的辅助代码处理，且算法的时间复杂度和常数因子较大，未给出高效实现的细节。

---

## 327. Student Evaluation of Repeated AI Feedback Across a Semester of Writing

**arXiv ID:** 2607.16115 | [PDF](https://arxiv.org/pdf/2607.16115v1)

**作者:** Andres Karjus `[一作]` (Tallinn University), Tiia Õun `[通讯]` (Tallinn University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对一门爱沙尼亚本科课程进行为期一个学期的观察，分析了283名学生共2988个反思性写作任务及其生成的AI反馈和学生的评价，探讨学生如何评价和使用AI反馈以及其对写作反思深度的影响。

**💡 创新点**

首次系统量化课堂中学生对反复使用生成式AI反馈的态度变化，并将AI文本检测与反思深度关联，揭示AI反馈在未直接鼓励深度反思的任务中对学生写作深度影响有限。

**🔧 技术方法**

采用GPT‑5.4对匿名文本进行自动标注（反馈可用性、深度、具体性等），并使用Pangram AI‑text detector评估作文作者身份，结合混合效应模型分析结果。

**📊 数据集**

使用一门数字技能与AI课程的作业数据集，包含学生撰写的300–350词反思性作文、对应AI生成的反馈段落以及学生的1–2句评价，共2988组完整数据。

**📈 对比分析**

对比AI反馈的具体性与学生感知的帮助度、以及AI使用比例与反思深度的关联；结果显示AI反馈总体高度具体，但其帮助感与下一篇写作的反思深度无显著关联，AI使用比例随学期推进显著上升。

**⚠️ 局限性**

研究受限于单一课程、学生自选AI工具且未获取具体模型版本，评价标签基于自动化标注且可能偏差，AI检测器对爱沙尼亚语的支持有限，且未能直接评估学生反馈解读能力与动机。

---

## 328. CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning Data

**arXiv ID:** 2607.16122 | [PDF](https://arxiv.org/pdf/2607.16122v1)

**作者:** Vipul Gupta `[一作]` (Scale AI), Yunzhong He `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 Rubric 评测数据的层级诊断框架 CRAFT，用来定位模型在回答中具体的能力缺口并生成定向微调数据。

**💡 创新点**

创新点是将 Rubric 评分标准拆解为能力探针，聚类形成层级树并动态选择弱节点，直接指导后续训练。

**🔧 技术方法**

采用 LLM 生成能力描述，基于向量聚类构建层级树，使用 LLM 评判提示–criterion 对的通过率，并进行动态节点选择。

**📊 数据集**

使用 PRBench 财务和法律子集的 Rubric 以及 13 个离线评测基准（如 MMLU Law、MBE、FinanceBench、ConvFinQA 等）。

**📈 对比分析**

与 EvalTree（基于提示层级聚类）和 Random（随机无结构采样）比较，CRAFT 在所有四个模型的金融域平均分均最高，法律域最高三模型，并在多模型多域上实现了显著提升。

**⚠️ 局限性**

局限在于需依赖高质量的 Rubric 数据，节点选择仍受模型判断一致性影响，且在某些细粒度基准上提升有限。

---

## 329. The Honest Quorum Problem: Epistemic Byzantine Fault Tolerance for Agentic Infrastructure

**arXiv ID:** 2607.16109 | [PDF](https://arxiv.org/pdf/2607.16109v1)

**作者:** Jun He `[一作]` (OpenKedge), Deying Yu `[通讯]` (OpenKedge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了诚实法定问题（Honest Quorum Problem），探讨了在协议合规的情况下，参与者可能会共同支持语义上无效的状态转移，从而导致协议的有效性和安全性受到威胁。

**💡 创新点**

创新点在于定义了认知拜占庭容错（Epistemic Byzantine Fault Tolerance），并引入了信心索引的预算来量化语义证书的有效性和可用性风险，强调协议合规性与语义正确性之间的差异。

**🔧 技术方法**

使用了概率推理网络和后决定性分布式系统的概念，结合了传统的拜占庭容错模型，提出了新的阈值条件和校准方法。

**📊 数据集**

论文中使用的具体数据集未明确提及，但提到的实验涉及模拟的云IAM策略变更，包含有效和无效的状态转移。

**📈 对比分析**

与传统的拜占庭容错方法相比，本文的方法通过引入信心预算来增强对语义有效性的保障，性能上能够更好地处理协议合规性与语义正确性之间的矛盾。

**⚠️ 局限性**

限制在于模型假设和预算估计的准确性，特别是在面对动态对手和潜在的共同模式故障时，可能会影响到协议的安全性和有效性。

---

## 330. DADiff: Diffusion-Driven Cross-Domain Policy Adaptation for Reinforcement Learning

**arXiv ID:** 2607.16090 | [PDF](https://arxiv.org/pdf/2607.16090v1)

**作者:** Hanyang Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在源域中充分训练后，只用极少的目标域交互就能将策略迁移到目标域

**💡 创新点**

通过扩散模型生成的“生成轨迹”来度量源域与目标域的动力学差异，并将该差异作为奖励惩罚或数据筛选的依据；同时给出了关于性能差异的理论上界

**🔧 技术方法**

扩散概率模型（DDPM）+ 生成轨迹偏差估计 + SAC + 变体（奖励修正与数据筛选）

**📊 数据集**

四个 MuJoCo 机器人（ant, hopper, halfcheetah, walker）在四种动力学偏移（kinematic, morphology, friction, gravity）下的数据集

**📈 对比分析**

与 DARC、VGDF、PAR、SAC-IW、SAC-tune、SAC-tar、Oracle 进行比较；在 16 种不同偏移任务中，DADiff 的平均收益比大多数基线高 8.7%（最高 42.3%），在绝大多数任务上逼近或超越 Oracle，GPU 内存略高但运行时与其他基线相当；VGDF 训练时间显著更长

**⚠️ 局限性**

对扩散模型的参数（如噪声模型容量、步数 K）敏感；需要对 λ、ξ 等超参数进行任务特定调优；在极度随机或极端偏移场景下表现仍有限，且方法依赖目标域少量交互数据，无法完全无交互迁移

---

## 331. Frontier Language Models Struggle to Copy: Text Can Be Better Viewed in 2D

**arXiv ID:** 2607.16072 | [PDF](https://arxiv.org/pdf/2607.16072v1)

**作者:** Haodong Wen `[一作]`, Kaifeng Lyu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文指出现有大型语言模型在复制输入字符串任务上表现不佳，并提出基于二维位置编码的 2D‑RoPE 以及自适应版本 Auto‑2D‑RoPE 来解决该问题。

**💡 创新点**

创新点在于将文本视作二维网格，用行列 ID 进行位置编码，使复制任务转化为固定偏移检索，从而显著提升长度泛化能力；同时引入可学习的自适应二维编码以减少对行分隔符的依赖。

**🔧 技术方法**

使用的技术包括 Transformer 与 RoPE 的变体（2D‑RoPE、Auto‑2D‑RoPE）、混合 RoPE、以及在 Qwen3 架构上进行大规模预训练和微调。

**📊 数据集**

所使用的数据集包括 synthetic binary copy、Python 列表转换任务、DCLM 语料库进行预训练，以及 FineWeb‑Edu 进行验证。

**📈 对比分析**

在复制任务中，2D‑RoPE 与 Auto‑2D‑RoPE 在 1‑层和 12‑层模型上分别实现了 1000× 长度泛化，且在 350M‑1.4B 参数规模下，预训练后微调的 2D‑RoPE 模型在 Imbalanced 与 Recursive‑Flip 复制任务上均显著优于标准 RoPE 和 H‑RoPE，且在常识推理任务上表现相当或更好。

**⚠️ 局限性**

局限性包括缺乏对多层 RoPE 失效原因的完整理论解释、Auto‑2D‑RoPE 在更大规模 LLM 训练中的可扩展性未完全验证，以及对 2D‑RoPE 在更复杂任务中的普适性仍有待探索。

---

## 332. Vision-Language Assistant for Emotional Reactions to Risky Driving

**arXiv ID:** 2607.16181 | [PDF](https://arxiv.org/pdf/2607.16181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 333. Revisiting Real-Time Interval and Throughput Maximization

**arXiv ID:** 2607.16163 | [PDF](https://arxiv.org/pdf/2607.16163v1)

**作者:** Allan Borodin `[一作]`, Nadim Mottu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在单机实时调度模型下，本文从传统的间隔调度问题出发，系统地研究了吞吐量最大化（Weighted/Unweighted Throughput Maximization）的在线算法，提出了一系列新的算法与理论界限；

**💡 创新点**

主要创新包括：①将间隔调度的C‑Benevolent/D‑Benevolent 1/4竞争比扩展到带重启的吞吐量问题，得到1/5竞争比；②引入提前通知（Advance Notice）模型，证明在比例权重下可在无抢占的情况下实现常数竞争比；③针对不确定加工时间的无权重吞吐量问题，给出了上界1/(k+1)与下界1/(2k)的竞争比，闭合了已知的k‑长度EDF算法与最优解之间的差距；

**🔧 技术方法**

主要技术手段为：预emption-with-restarting 以及精细的前向/后向充电（charging）分析、Karamata不等式应用、范围与前驱链概念的构造、以及递归对抗性构造（adversarial gadgets）；

**📊 数据集**

该工作完全基于理论分析与证明，没有使用实验数据集；

**📈 对比分析**

与现有最优/最坏情况分析相比较，τ‑Persist 算法在C‑Benevolent/D‑Benevolent权重下实现1/5竞争比，提前通知模型在比例权重下实现1/5竞争比（或更优，取决于提前通知的比例）；在无权重且最多k种加工时间的情况下，k‑长度EDF 实现1/(2k)竞争比；在无权重、任意加工时间的撤销模型下证明了竞争比上界为1/(k+1)，即无法实现常数竞争比；

**⚠️ 局限性**

局限性包括：①对非比例、非C/D‑Benevolent权重的情况仍无常数竞争比；②提前通知模型的常数竞争比仅在比例权重下成立，无法推广到一般C/D‑Benevolent；③无权重撤销模型的上界与下界之间仍存在较大间隙（1/(k+1) vs 1/(2k)），需要进一步研究；④缺乏实验验证，结果仅在理论层面证明。

---

## 334. JoyNexus: Service-Oriented Multi-Tenant Post-Training for VLA Models

**arXiv ID:** 2607.16074 | [PDF](https://arxiv.org/pdf/2607.16074v1)

**作者:** Haoran Sun `[一作]` (JDT AI Infra), Junwu Xiong `[通讯]` (JDT AI Infra)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种面向多租户的视觉-语言-动作模型（VLA）后训练服务架构，将训练、推理和环境交互拆分为共享服务和租户私有模块，实现多租户并行执行与资源复用；

**💡 创新点**

创新点包括：1）基于Tinker式服务化的多租户模型，统一SFT、RL和评估工作流；2）将共享冻结的视觉‑语言基座与租户私有动作模块分离，支持轻量级参数高效适配；3）采用组批处理（group batching）在推理队列中合并相同基座的请求，以提升GPU利用率；4）引入双队列调度（训练队列与推理队列）与租户隔离、容错与弹性伸缩机制；

**🔧 技术方法**

核心技术包括：服务导向架构（Master Service + Training/Inference/Environment Services）、基于RLinf的异步生产者-消费者模式、动态批处理、离线/在线数据调度、容错恢复、弹性伸缩、监控与指标收集；

**📊 数据集**

实验使用了StarVLA、Qwen3‑VL‑4B、QwenGR00T、OpenPI等VLA模型，数据集包括LIBERO、ManiSkill、CALVIN和LeRobot，用于构建多租户在线RL、离线SFT和评估工作负载；

**📈 对比分析**

通过与单租户隔离执行基线对比，实验显示多租户部署在同一8‑GPU节点上训练模型服务利用率提升1.99×、推理服务提升1.33×，整体GPU时间效率提升1.39×；在组批处理实验中，多个租户小批量请求的共享前向阶段加速可达数倍，且对单租户训练曲线无负面影响；

**⚠️ 局限性**

局限性包括：1）资源分配仍为固定路径，缺乏针对实时负载的动态路由；2）对租户自定义环境与算法扩展的安全性与性能保障尚未完善；3）实验主要评估共享前向加速，未测量完整端到端训练吞吐量；4）仅在特定VLA模型与数据架构上验证，泛化性待进一步验证；5）需要进一步研究定价与优先级策略。

---

## 335. An Exam for Active Observers

**arXiv ID:** 2607.16165 | [PDF](https://arxiv.org/pdf/2607.16165v1)

**作者:** Jiarui Zhang `[一作]` (University of Southern California), Willie Neiswanger `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为ActiveVision的基准，用于测量多模态大型语言模型（MLLMs）的主动视觉观察能力，并对17个多样化任务进行评估。

**💡 创新点**

创新点在于：① 将主动观察（即反复回视并更新假设）作为单一可测量的能力构造基准；② 设计了分为“分布式扫描”“顺序遍历”“视觉属性迁移”三类任务，并通过程序化生成与GPT‑image‑2渲染的照片级图像，确保任务需要反复查看图像而非一次性语言描述；③ 通过对前沿模型和人类的系统对比，量化了当前MLLM在主动观察上的巨大差距。

**🔧 技术方法**

技术手段包括：程序化生成任务几何结构（Python/Matplotlib）、GPT‑image‑2生成真实感图像、纯链式推理（CoT）评估、以及工具/代码代理（Codex、Claude Code）进行替代性解决方案评测。

**📊 数据集**

数据集：ActiveVision Benchmark（85条目，17类任务，每类5个实例），每个实例都通过两阶段生成 pipeline 从程序化 scaffold 转换为 photorealistic 图像。公开在 https://activevision.dev、GitHub、Hugging Face 等平台。

**📈 对比分析**

比较方法：使用精确匹配准确率评估；在不同 reasoning‑effort 等级下对 GPT‑5.5、Claude Fable 5、Claude Opus 4.8、Gemini 3.1 Pro、Gemini 3.5 Flash 等前沿模型进行全尺度评测；同时对三位人类参与者进行基准；对工具代理（Codex、Claude Code）进行同样评测。结果显示：最佳模型 GPT‑5.5 在最高推理力度下仅 10.6% 正确率，Claude Fable 5 仅 3.5%；人类平均 96.1%。工具代理虽提升至 24.7–50.6% 但仍远低于人类，且耗时/成本显著高于人类。

**⚠️ 局限性**

局限性：① 图像为程序化合成并经 GPT‑image‑2 重新渲染，仍为人工构造而非真实分布，可能影响外部有效性；② 随着模型语言描述能力提升，任务可能出现语言捷径；③ 仅评估主动观察能力，未涵盖其他视觉理解维度；④ 工具代理对低质量图像鲁棒性不足，导致错误检验失效。

---

## 336. Physics-enhanced reinforcement learning for real-time optimal control of dynamical systems

**arXiv ID:** 2607.16177 | [PDF](https://arxiv.org/pdf/2607.16177v1)

**作者:** Matteo Tomasetto `[一作]` (Politecnico di Milano), Andrea Manzoni `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将物理模型信息融入强化学习的算法PEARL，用于高维、分布式且参数化的可微分动力系统的闭环控制。

**💡 创新点**

创新点在于将自适应短时优化、自动微分以及终端伴随（adjoint）网络相结合，直接学习伴随梯度以弥补长时序依赖，从而显著提升样本效率和鲁棒性。

**🔧 技术方法**

核心技术包括：Actor‑Adjoint 方法、自动微分（AD）、短时段截断策略、伴随网络训练（TD‑λ 方式）以及对比经典 PPO、TD3、BPTT、截断BPTT 与 SHAC。

**📊 数据集**

实验数据集为两组基于非定常双旋涡流场的参数化导航任务：1）领航者–追随者（leader‑follower）游戏；2）其对应的高维均值场（mean‑field）领导者‑追随者问题。

**📈 对比分析**

与传统模型无关的RL算法（PPO、TD3）以及基于AD的梯度方法（BPTT、截断BPTT、SHAC）进行对比，PEARL在稠密和稀疏奖励设置下均表现出更高的累计奖励、更快的收敛速度以及更好的泛化性能。

**⚠️ 局限性**

局限性包括：仅在可微分模拟器上验证，缺乏对部分可观测或真实硬件场景的测试；对大型复杂系统的可扩展性尚未充分验证；并且需要较高的模型可微分性与计算资源。

---

## 337. Adaptive Fault Injection Planning for Multi-Layer Self-Healing AI Infrastructure

**arXiv ID:** 2607.16161 | [PDF](https://arxiv.org/pdf/2607.16161v1)

**作者:** Saurabh Kulkarni `[一作]` (Meta Platforms, Inc.), Gautam Nayak `[通讯]` (Meta Platforms, Inc.)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 ADA-ST（自适应调度算法）用于多层自愈 AI 基础设施的故障注入规划，能够系统地覆盖跨层传播路径。

**💡 创新点**

创新点在于：①构建基于加权传播概率的四层故障传播图；②利用图覆盖度驱动的自适应场景选择算法；③引入功能层抽象模型 FLAM，实现跨平台情景迁移；④通过迭代反馈实现对 100% 边覆盖的快速收敛。

**🔧 技术方法**

所采用技术包括：图论模型与覆盖度度量、动态评分与优先级调度（ADA-ST）、故障注入与监测框架、基于历史事件的概率推断、以及基于 SEU（Scenario Effort Unit）的成本评估。

**📊 数据集**

主要数据集为 72,550 条 Alpha 平台四年生产维修工单（包含跨层传播案例），以及 Beta、Gamma 两代平台的架构文档与 NPI 测试计划，用于构造传播图与验证算法。

**📈 对比分析**

与传统静态 NPI 测试（仅 20–25% 边覆盖）相比，ADA-ST 在 Alpha、Beta、Gamma 上分别在 10、12、9 次迭代内实现 100% 边覆盖；随机场景选择在 14–18% 的迭代次数内完成，说明自适应调度显著提升效率，且在 SEU 成本上约比静态测试低 90%。

**⚠️ 局限性**

局限性包括：对生产工单的依赖，可能忽略从未出现的故障模式；传播概率估计仅为下限，受标签缺失与时间窗口影响；仅适用于四层结构，扩展到更多层需重新定义图；实际物理验证仍需硬件支持，且模型未捕捉缺失信号与多信号相关性等细节。

---

## 338. On the Stability of Minimum-Weight Perfect Matching on the Line

**arXiv ID:** 2607.16137 | [PDF](https://arxiv.org/pdf/2607.16137v1)

**作者:** Mark de Berg `[一作]` (TU Eindhoven), Andree-Ovidiu Stef `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在一维欧氏空间中，设计了一种动态最小权重完美匹配算法，在每一次插入/删除两点时最多只改变 O(√n) 条边，并保持 2 倍逼近。

**💡 创新点**

证明该 2 倍逼近在所有子线性稳定性（即 O(√n) 以下稳定性）下是最优的；并给出任何 o(log n) 稳定性的算法都必须具有无界逼近比的下界，阐明稳定性与逼近质量之间的根本权衡。

**🔧 技术方法**

核心技术包括：点集的块划分（每块大小为 O(√n)），连接器和包裹（wrapping）不变式；在每次更新时仅重构受影响块并“切换”其内部匹配；以及通过构造“良好分割器”证明 2 倍逼近。下界构造使用基于二进制 Trie 的点集，利用深度/权重的层级关系。

**📊 数据集**

该工作为理论算法，未使用具体实验数据集，而是在理论分析框架下证明性质和下界。

**📈 对比分析**

与此前仅能获得 O(log n) 稳定性的 3 倍逼近（如 Gupta 等）以及仅适用于完全动态场景的 2 倍逼近相比，本文的 O(√n) 稳定性和 2 倍逼近是最优组合；下界进一步说明更低的稳定性无法提升逼近质量。

**⚠️ 局限性**

限制：仅适用于一维欧氏空间，二维及更高维度尚未得到类似结果；算法仍需随机化或更复杂的数据结构才能扩展到更大维度；并且在实践中 O(√n) 的常数因子与具体实现相关，可能影响实际效率。

---

## 339. ToolSciVer: Multimodal Scientific Claim Verification with Visual Tool Augmented Reinforcement Learning

**arXiv ID:** 2607.16131 | [PDF](https://arxiv.org/pdf/2607.16131v1)

**作者:** Binglin Zhou `[一作]` (Pennsylvania State University), Rui Zhang `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于工具增强的多模态科学主张验证框架（Tool‑Sciver），通过VLM动态调用专门针对表格、图表和通用科学图像的可视化工具，以获取与主张相关的视觉证据并进行推理。

**💡 创新点**

创新点在于①设计了三种类型感知工具（表格行/列提取、图表解析、区域放大），②使用强化学习（Group Relative Policy Optimization）训练模型在何时、调用哪种工具以及如何使用返回信息，以实现选择性、有效且可靠的工具使用；③将工具使用与最终验证结果紧密耦合，形成端到端可学习的证据获取与推理流程。

**🔧 技术方法**

技术包括视觉语言模型（VLM）与工具调用接口的结合，工具实现基于OCR的表格行/列抽取、图表JSON化解析、以及高分辨率区域裁剪；强化学习算法GRPO；奖励设计结合答案正确性、格式有效性、长度控制、工具效率与错误惩罚。

**📊 数据集**

使用SciVer和MuSciClaims两个公开多模态科学验证基准；此外在ChartQA、DVQA、FigureQA等通用图表/图像问答数据集上做迁移评测。

**📈 对比分析**

与非工具链式推理、提示级工具调用、VTool‑R1和OpenThinkIMG等RL工具使用基线进行对比；在所有主干模型（Qwen、InternVL、Gemma）上均实现显著提升，SciVer整体提升约4–6个百分点，MuSciClaims提升约5–8个百分点；在ChartQA/DVQA/FigureQA等任务中也平均提高5个百分点。

**⚠️ 局限性**

局限性主要是依赖外部工具的准确性；OCR或图表解析错误会导致证据缺失或错误，强化学习难以完全弥补；在表格密集、标签小或图表格式不常见时工具输出不稳定；未来需加强科学视觉解析和不确定性感知机制。

---

## 340. Rate-Utility Frontiers for Language Encodings: Comparing Tokens, Bytes, and Pixels Under Controlled Linguistic Content

**arXiv ID:** 2607.16117 | [PDF](https://arxiv.org/pdf/2607.16117v1)

**作者:** Ingo Ziegler `[一作]` (University of Copenhagen), Desmond Elliott `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在保持语言内容和下游容量相同的前提下，系统性比较了三种文本编码（子词 token、原始字节 byte、渲染像素 pixel），并绘制了它们在三种任务（表面形式保留、跨语言检索、主题分类）的 rate–utility 前沿。

**💡 创新点**

创新点在于：①通过共享 Perceiver 先验瓶颈，将源速率、容量和保留信息分离，消除长度与语言不平衡的混淆；②引入 pixel 编码并与 token、byte 并置比较；③在 13 语种、5 种脚本的并行数据上对同一内容进行多任务评测，揭示不同编码在不同容量和任务下的优劣差异。

**🔧 技术方法**

使用技术包括：Transformer 编码器 + Perceiver bottleneck，InfoNCE 对比学习用于表面保留与跨语言检索；线性分类器用于主题分类；对不同瓶颈宽度 D 进行 sweep，绘制前沿；以及计算 FLOPs 的训练成本分析。

**📊 数据集**

使用的数据集为 SIB‑200 并行数据集，包含 13 种语言（英语、中文、俄语、阿拉伯语、印地语等）与 5 种脚本（拉丁、西里尔、阿拉伯、天城文、汉字）。

**📈 对比分析**

比较方法为：对每种编码在每个任务上分别在不同瓶颈宽度 D 下训练并评估 Recall@1 或 macro‑F1；将结果绘制为 rate–utility 前沿；在同一容量下比较性能。实验表明：pixel 在表面形式保留最优、byte 在跨语言检索最优、token 在主题分类最优；不同编码在不同容量区间存在显著交叉，说明单一编码并非总体最佳。

**⚠️ 局限性**

局限性包括：①仅使用单一 Transformer + Perceiver 架构，未探究更大规模或不同结构的模型；②pixel 编码受渲染尺寸、字体、排版等因素影响，可能在真实 OCR 场景中表现不同；③实验仅覆盖 SIB‑200，缺乏更大、多样化的多语言语料；④计算成本评估主要基于 FLOPs，未涉及能源消耗或硬件效率；⑤未深入分析各任务对编码细节（如字符重叠、书写方向）的依赖。

---

## 341. SQUIRO: A Framework for Security-Aware Quantum-Classical Scheduling on Kubernetes

**arXiv ID:** 2607.16089 | [PDF](https://arxiv.org/pdf/2607.16089v1)

**作者:** Ignazio Pedone `[一作]` (Helix 42), Edoardo Giusto `[通讯]` (Helix 42)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了 SQUIRO，一个安全感知的量子-经典调度框架，能够在 Kubernetes 集群中联合优化成本、能耗与安全态势，并提供量子后端选择与安全姿态评估。

**💡 创新点**

创新点包括：① 统一调度模型 (USM) 与六步调度设计方法 (SDM) 的首次结合；② 将硬安全约束与残余风险优化统一到多目标调度中；③ 基于电路的量子后端选择器，考虑相干性、队列压力、校准新旧度等多维度；④ 采用多维安全姿态框架和可度量的残余风险指标，区别硬约束与软偏好。

**🔧 技术方法**

技术实现主要采用 Google OR-Tools 的 CP-SAT 约束规划求解器做全局多目标优化；Kubernetes 插件、Device 插件、QDMI 接口用来获取资源与后端元数据；量子后端元数据采集与基于电路的评分算法；安全评估框架与加权评分方法；实验使用合成集群与仿真工具。

**📊 数据集**

实验数据集为：合成 Kubernetes 集群、云定价数据、IBM Quantum 校准数据、IonQ 量子硬件规格以及仿真生成的工作负载和资源分布；未公开真实量子-经典集群数据。

**📈 对比分析**

通过与 Greedy Kubernetes 调度基线、单目标成本/能耗调度、以及在四种负载状态（低载、平衡、拥挤、过载）下的全局优化进行对比。实验结果显示：在低载状态下成本降低51%、能耗降低63%；安全硬约束满足率达100%；残余风险优化提升安全得分约3.7点；CP-SAT 求解时间随节点数增长；两阶段后端选择相较于仅基于 2Q 误差排名能够捕捉相干性与队列压力差异。

**⚠️ 局限性**

局限性包括：① 目前仅在 Kubernetes 阶段实现，未完成端到端多层（电路-系统-OS）联合优化；② 量子后端选择尚未嵌入核心求解器，仍为独立服务；③ 在大规模集群（>30 节点）时 CP-SAT 求解时间可能超出预算；④ 安全姿态模型基于手工设定的 10 项检查，未覆盖全部攻击面；⑤ 所有实验均基于合成环境，缺乏真实量子-经典集群验证。

---

## 342. Neural spectroscopy of AlphaFold2 reveals encoded protein conformational landscapes

**arXiv ID:** 2607.16087 | [PDF](https://arxiv.org/pdf/2607.16087v1)

**作者:** Kaustav Mehta `[一作]` `[通讯]`, Kaustav Mehta

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过对AlphaFold2 Evoformer权重进行缩放高斯卷积（SGC）扰动，研究者直接探测并描绘了模型内部学习到的蛋白质构象空间。

**💡 创新点**

创新点在于提出“神经光谱学”这一方法：利用可控、确定性的权重扰动在不改变输入的前提下揭示模型对构象多样性的编码，从而把AlphaFold2视为对蛋白折叠知识的压缩表征。

**🔧 技术方法**

主要技术包括：1）SGC（对权重张量进行均值化和缩放）；2）噪声对照（白噪声、频谱噪声）验证结构响应的确定性；3）多模型、多深度、多种MSA深度的系统探索；4）基于MD、RMSD、Q因子、R_g等多维度统计对生成构象进行量化；5）PCA降维分析权重空间响应。

**📊 数据集**

使用的数据集涵盖：①单体蛋白 ubiquitin（PDB: 1UBQ）和 KaiB（PDB: 2QKE/5JYT）；②聚焦无折叠域的 α‑synuclein（UniProt: P37840）；③基于D.E. Shaw 公开的 ubiquitin ms‑scale MD 和 300 K 等价温度 MD 轨迹作为物理参考。

**📈 对比分析**

通过与实验测定的稳定性序列、MD 的RMSF、以及相同条件下的噪声对照进行比较，展示了SGC产生的构象变化与已知折叠机制高度一致，且生成的构象层次与物理模拟的折叠漏斗相匹配；同时模型间保持一致性，证明方法的可靠性。

**⚠️ 局限性**

限制包括：①仅在少数蛋白（单一折叠体、折叠开关蛋白、无折叠域蛋白）上验证，泛化能力待进一步评估；②扰动仅揭示了构象“空间”而非动力学或热力学分布；③对权重作用机制的解释仍为假设，缺乏理论阐述；④在默认MSA深度下结果可能不同，需进一步探究。

---

## 343. Pick-to-Learn Calibration of an MPC Policy for an Origin-to-Destination Flight Problem

**arXiv ID:** 2607.16084 | [PDF](https://arxiv.org/pdf/2607.16084v1)

**作者:** Marco C. Campi `[一作]` (University of Brescia), Simone Garatti `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用 Pick‑to‑Learn (P2L) 方法对 MPC 超参数进行校准，最终得到一个能在 400 条风场场景下避免低连通区的航迹控制器。

**💡 创新点**

创新点在于把 P2L 作为压缩和概率保证的工具（而非仅仅评估复杂度），并通过 P2L⁺ 实现校准后对额外属性的无模型后验验证。

**🔧 技术方法**

主要技术包括场景采样学习、基于 Bayesian 优化的超参数搜索、基于样本压缩的学习理论与风险界定。

**📊 数据集**

使用的数据集为 400 条独立的跨风场景（由自回归过程生成的随机风场），未分训练/测试集，所有样本均用于 P2L 压缩与校准。

**📈 对比分析**

实验显示，压缩后仅剩 2 条信息场景即可满足低连通区避免，风险上界为 4.8%；对更严格的连通区约束，风险上界提升至 11.4%；对 10 连续采样点容忍度，风险上界降至 6.4%；对航迹行程长度，提供 58.8% 的累积分布下界。

**⚠️ 局限性**

局限性包括需足够多且相互独立的场景；P2L 的场景挑选规则需手工设计，可能影响收敛速度；未与传统鲁棒/MPC 方案进行定量对比，且在更大规模或非线性系统上的可扩展性尚未验证。

---

## 344. Physics-Based Deep Spatiotemporal Hyperlocal Radar Nowcasting with a Multi-Variable U-Net for High-Resolution Precipitation Forecasting

**arXiv ID:** 2607.16080 | [PDF](https://arxiv.org/pdf/2607.16080v1)

**作者:** Akshay Sunil `[一作]` (Indian Institute of Technology Bombay), Subimal Ghosh `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于多高度雷达观测的U-Net深度学习框架，用单机雷达反射率和多变量速度梯度代理变量实现高分辨率降水即刻预报。

**💡 创新点**

创新点在于：①将多高度反射率与径向速度、速度梯度的散度、方向、涡度等物理代理变量联合输入；②在U-Net的瓶颈和解码阶段引入高反射率注意力机制；③使用物理引导的特征归因方法验证模型学习的敏感性是否符合气象学预期。

**🔧 技术方法**

技术手段包括：卷积编码-解码U-Net网络、梯度算子提取物理代理、加权MSE+SSIM+阈值损失的复合目标、注意力模块、以及基于反射率阈值的评价指标（CSI、ETS、RMSE、相关系数）。

**📊 数据集**

使用数据集：印度气象局位于维拉瓦利的C波段多高度雷达观测（反射率、径向速度），覆盖直至250公里范围内，时间分辨率7.5分钟，训练集覆盖2023年5月至8月，测试集为时间上相互独立的事件。

**📈 对比分析**

与传统的光流外推（Persistence）对比，模型在90分钟预报中获得CSI分别为0.437（≥10 dBZ）、0.332（≥20 dBZ）和0.193（≥30 dBZ），连续验证显示在30–90分钟区间RMSE低于Persistence且空间相关性更高，说明在较长预报期内能更好保留风暴结构。

**⚠️ 局限性**

局限性主要是对高强度核心的低预测能力，随着预报时距增加，模型倾向于低估强降水，导致高反射率区的误差主要为衰减和细尺度结构损失；且在极端降雨事件中仍表现为过度保守的误报率。

---

## 345. Solving Stackelberg Vertex Cover on trees using split and join

**arXiv ID:** 2607.16078 | [PDF](https://arxiv.org/pdf/2607.16078v1)

**作者:** Dominik Scheder `[一作]` (TU Chemnitz), Johannes Tantow `[通讯]` (TU Chemnitz)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了三类针对树图的Stackelberg Vertex Cover问题的算法，分别为整数权重下的伪多项式算法、LCA树上的强多项式算法以及基于可见性数的FPT算法；同时引入了承诺机制和Split‑Join引理以实现实例分解；并证明在承诺情形下问题仍为弱NP‑完全。

**💡 创新点**

核心创新在于将LP对偶性与顶点覆盖整数性相结合，构造Split‑Join引理实现实例在树上可分解；通过承诺机制控制子实例的最优结构；并提出针对可见性参数的FPT方法和LCA树的线性时间解法。

**🔧 技术方法**

主要技术包括：LP对偶性与顶点覆盖整数性证明、Split‑Join分解与合并、动态规划、整数权重的拆分策略、递归分治与可见性参数化。

**📊 数据集**

本文为理论研究，未使用任何实测数据集，所有算法与复杂度均在理论层面给出。

**📈 对比分析**

与已有路径与循环的算法比较，本文的LCA树算法在时间上达到线性，伪多项式算法在整数权重场景下实现O(|V|·w_max³)；FPT算法在可见性参数k时为O(|V|·k!)，均优于之前的指数或未确定的复杂度。

**⚠️ 局限性**

主要局限在于：对一般树图仅得到伪多项式解，且承诺情形仍为弱NP‑完全；无法直接推广至更广的树宽图；实现仍依赖LP求解，缺乏完全组合式算法。

---

## 346. Evaluating Open-Weight LLMs for Generating Structured Threat Information for Autonomous Vehicle Vulnerabilities

**arXiv ID:** 2607.16175 | [PDF](https://arxiv.org/pdf/2607.16175v1)

**作者:** Md Erfan `[一作]` (University of Alabama), Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CAV-STIXGen 数据集，将联网与自动驾驶车辆（CAV）相关的 CVE 文本描述映射为结构化的 STIX 2.1 语义表示，并评估多种开源大模型在该任务上的生成性能。

**💡 创新点**

首创了面向 CAV 领域的 STIX 生成数据集，并系统地比较了不同提示策略、温度设定以及多代理拆解任务对模型表现的影响；同时通过 CWE 与 MITRE ATT&CK 的共现分析揭示了 CAV 漏洞的典型弱点与攻击模式。

**🔧 技术方法**

采用开源大型语言模型（Gemma、Codestral、LLaMA、Qwen、Phi 等）结合提示工程（无上下文、STIX 引导、动态少样本）、多代理拆解、微调与后处理等技术；利用 STIX 2.1 规范、CWE 与 MITRE ATT&CK 体系进行映射。

**📊 数据集**

使用了 183 条 CAV 相关 CVE 的 CAV-STIXGen 数据集（共 1,383 个 SDO、1,395 个 SRO、211 个 CWE、294 个 MITRE ATT&CK 关联），并结合公开的 CVE、CWE、MITRE ATT&CK 数据源进行人工标注与验证。

**📈 对比分析**

在 11 种 4B~120B 参数的开源 LLM 上，使用三种提示策略（无上下文、STIX 指导、动态少样本）和四种温度（0、0.25、0.75、1.0）进行基准实验；评估指标为 micro‑precision、micro‑recall、micro‑F1 以及 MITRE ATT&CK 的 Match@1 / Match@All。单模型最高可达 SDO F1=0.94、SRO F1=0.63、CWE F1=0.99，ATT&CK 完整匹配仍具挑战；多代理配置在 SDO 上可提升至 0.91，但 SRO 仅 0.43。

**⚠️ 局限性**

主要限制包括：1) 关系抽取（SRO）性能仍偏低，难以实现完整的图结构；2) MITRE ATT&CK 的全匹配难度大，Match@All 仍低；3) 数据集规模有限，STIX 类型与关系覆盖不够全面；4) LLM 生成可能出现错误或幻觉，缺乏外部知识检索与后处理机制；5) 仅评估了开源模型，未涉及商业或专用安全模型。

---

## 347. PRISA: Proactive Infrastructure LiDAR Framework for Intersection Safety Assessment

**arXiv ID:** 2607.16156 | [PDF](https://arxiv.org/pdf/2607.16156v1)

**作者:** Tam Bang `[一作]` (University of Tennessee at Chattanooga), Hoang H. Nguyen `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 PRISA，一个可模块化、边缘可部署的 LiDAR 基础设施框架，用于主动交叉口安全评估；

**💡 创新点**

其创新点在于将自监督数据清洗、学习型轨迹预测与双重 surrogate safety measures（TTC 与 PPET）风险评估无缝集成，并实现与任何检测追踪模块的 plug‑and‑play 接口；

**🔧 技术方法**

技术方案包括 3D LiDAR 感知与多目标追踪、基于 FlowChain 的轨迹预测、时间到碰撞（TTC）与预测后侵入时间（PPET）的冲突检测，以及 NVIDIA Jetson AGX Thor 上的实时推理与 TensorRT 加速；

**📊 数据集**

使用公开的 R‑LiViT 路边 LiDAR 数据集进行模型训练与评估，并在田纳西州查塔努加市的实际信号交叉口部署了 Ouster LiDAR 进行现场验证；

**📈 对比分析**

与传统的基线（常数速度、常数加速度）以及 MID、Trajectron++ 对比，FlowChain 在 ADE/FDE 上实现了 30%–40% 的提升，实时总延迟仅 194 ms，满足 PPET 预测窗口需求，TTC 评估保持实时；

**⚠️ 局限性**

主要局限在于预测模型对转弯等非直线动作存在偏差，导致 PPET 假阳性；此外，当前框架仅针对 LiDAR 感知，未涵盖多模态融合或复杂遮挡环境下的鲁棒性验证。

---

## 348. When Do Multi-Agent Systems Help? An Information Bottleneck Perspective

**arXiv ID:** 2607.16133 | [PDF](https://arxiv.org/pdf/2607.16133v1)

**作者:** Wendi Yu `[一作]` (Texas A&M University), Shuiwang Ji `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过信息瓶颈视角，阐明LLM驱动的多代理系统（MAS）相较于单代理系统（SAS）的优势来自于受限通信的上下文压缩与信息损失权衡，并解释了模型能力如何影响此权衡。

**💡 创新点**

创新点在于：①将MAS的收益建模为信息瓶颈优化问题，揭示了上下文压缩与信息损失的二元权衡；②引入参数β捕捉模型能力对该权衡的影响，解释了同一Relay在弱模型下有利、在强模型下可能不利；③通过理论与18个实验验证，提出了“Relay复杂度”概念，系统化评估MAS收益。

**🔧 技术方法**

使用的信息技术包括：信息瓶颈（IB）理论、LLM代理框架、受限Relay压缩、理论推导与实验对比；实验中采用多代理规划、受限通信与不同规模LLM（Qwen2.5-7B、GPT-4o-mini、Qwen3.5-27B）。

**📊 数据集**

使用的数据集/基准包括：ALFWorld、WebShop、WorkBench、WideSearch、TravelPlanner（分为Commonsense和Hard-constraint子任务），共18个实验设置。

**📈 对比分析**

比较方法：在相同规划拆分下对比三种原型（SAS、SAS-contextflow、MAS），并记录不同模型规模下的指标（如Success Rate、Reward、F1等）。实验结果显示：在低Relay复杂度（δ≈0）场景下，MAS在所有模型规模下均优于SAS-contextflow；在高Relay复杂度（δ>0）场景下，MAS收益随模型强度递减，甚至出现负值；整体表明MAS的优势受Relay信息损失控制。

**⚠️ 局限性**

局限性：①β和δ等关键参数仅为经验性或结构性估计，缺乏定量度量；②实验仅在固定任务拆分和模型规模上验证，未探讨自适应或动态Relay；③未给出具体的Relay编码器训练方法，缺乏可直接复现的实现细节。

---

## 349. BayesContact: Uncertain Pose Estimation via Visuo-Tactile Proposals and Simulation-based Inference

**arXiv ID:** 2607.16123 | [PDF](https://arxiv.org/pdf/2607.16123v1)

**作者:** Aditya Kamireddypalli `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在插槽式抓取任务中提出一种基于模拟推断的贝叶斯框架 BayesContact，用渲染得到的深度信息和物理模拟得到的接触信息，利用粒子滤波实时估计目标孔的姿态，并在估计过程中执行信息增益主动探测。

**💡 创新点**

创新点在于：① 将视觉与力/扭矩接触感知统一进同一模拟推断框架；② 采用基于物理与图形模拟的非解析似然函数，避免昂贵的离线训练；③ 通过粒子分布保留多模态不确定性，从而实现信息增益主动探测。

**🔧 技术方法**

使用的技术包括：渲染器（深度图像模拟）、物理仿真器（接触力学模拟）、粒子滤波（Sequential Monte Carlo）、概率编程实现多模态似然融合、基于信息增益的主动探测策略。

**📊 数据集**

数据集主要来自：① 五种不同几何的仿真 peg‑in‑hole 场景（含随机姿态）；② 真实机器人实验（KUKA iiwa14 + ATI 6-DOF F/T 传感器）下的两种几何（Rectangle 与 Rectangle‑teeth）。

**📈 对比分析**

与 Vision‑Only（BCv‑SMC/BCv‑PF）、ICP、FoundationPose、Man‑UKF 等方法对比，BayesContact 在 ADD‑S、位置误差、角度误差以及插入成功率上分别提升约 30%–40%，尤其在视觉歧义大、几何对称的场景中显著优于对照组。

**⚠️ 局限性**

局限性包括：① 依赖高质量的物理/渲染仿真，仿真与现实差异会影响估计；② 对已知目标几何的假设，未知或形变物体难以直接应用；③ 计算开销较大，粒子滤波与仿真实时性受限；④ 对传感器噪声、校准误差敏感，需要精确的传感器标定。

---

## 350. Comparison of Energy System Optimization Software and Evaluation of Selected Frameworks

**arXiv ID:** 2607.16121 | [PDF](https://arxiv.org/pdf/2607.16121v1)

**作者:** Pedro Caixeta `[一作]` (Karlsruhe Institute of Technology), Haozhen Cheng `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述与案例模型分析，对REMix、MTRESS、COMANDO、OEMOF、HOMER PRO五款能源系统优化软件进行系统比较，提出评估准则并归纳各软件最适用场景。

**💡 创新点**

创新点在于首次针对同一研究领域内五款工具构建统一的比较框架，结合功能、成本、学习难度等六维度评价，并基于此指出各软件的最佳适用场景。

**🔧 技术方法**

采用文献检索、案例模型评审、比较准则设计与定性评分等技术手段；工具主要是Python库、GAMS及HOMER GUI。

**📊 数据集**

数据来源为各软件官方文档、教程（共62个）和公开案例研究，未使用实际能源数据集。

**📈 对比分析**

比较方法为先收集教程信息，提炼差异，制定六个评估准则（财务成本、培训投入、建模投入、建模范围、优化目标、优化对象），对REMix、COMANDO、OEMOF、HOMER PRO进行评分；结果以文字形式呈现，未给出量化性能指标。

**⚠️ 局限性**

局限性包括：可获取的教程数量有限、部分教程无法运行、工作量受限、信息来源仅限公开文献、MTRESS工具被排除，导致对某些功能评估不完整。

---

## 351. Adaptive Contrast Enhancement and Optimised Feature Matching for RootSIFT-Based Palm-Vein Recognition

**arXiv ID:** 2607.16077 | [PDF](https://arxiv.org/pdf/2607.16077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 352. DoSQ: A Cross-Layer Denial of Service Quality Attack by Exploiting Side Channels in 5G NR

**arXiv ID:** 2607.16102 | [PDF](https://arxiv.org/pdf/2607.16102v1)

**作者:** Mahmudul Hassan Ashik `[一作]` (George Mason University), Moinul Hossain `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于5G NR控制信道DCI的协议感知定向干扰攻击（DoSQ），能够在同一毫秒槽内识别并精准干扰目标UE的PDSCH资源，实现应用层服务质量（Goodput）下降；

**💡 创新点**

创新点在于：①利用已公开的PDCCH解码信息作为跨层侧信道，首次证明可通过仅解码DCI预测并逼近目标UE的应用层Goodput状态；②在此基础上设计了低能耗、高精度的闭环控制策略，仅在预测为“低Goodput+下降趋势”时才激活干扰；③提出SSB时频跳变随机化作为对策，大幅提高攻击者重同步成本；

**🔧 技术方法**

使用的软件定义无线电（USRP B210）、srsRAN核心网、Open5GS、Python/Scikit-Learn+XGBoost机器学习模型、ZMQ IPC通信、USRP UHD驱动等；

**📊 数据集**

数据集为私有5G测试平台录制的DCI日志与YouTube Live的Stats‑for‑Nerds每秒Goodput信息，覆盖无干扰及七种不同hit‑rate的攻击场景，累计两小时以上；

**📈 对比分析**

对比方法：采用随机预测、众数预测与XGBoost模型；在留一批/留一场景交叉验证中，State三类宏F1≈0.57，Trend二类宏F1≈0.63；精度在top‑1%置信度下达87%，比基线提升4.2倍；实验显示在2–10% hit‑rate下，目标UE Goodput下降40–50%，旁路UE无明显影响；

**⚠️ 局限性**

局限性包括：①攻击需要在毫秒级同槽解码与发射，硬件延迟可能受限；②侧信道预测在极端网络负载或多用户情况下的鲁棒性尚未完全验证；③SSB随机化对策需UE侧固件更新，实际部署成本较高；

---

## 353. Let the Body Follow: Coupled Egocentric Control for Whole-Body Robot Teleoperation

**arXiv ID:** 2607.16095 | [PDF](https://arxiv.org/pdf/2607.16095v1)

**作者:** Tsung-Chi Lin `[一作]` (New Jersey Institute of Technology), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于自我中心耦合的全身遥操作控制方法，使机器人躯干和底盘自动跟随操作者的视线和手部动作，从而减少对躯干和底盘的显式命令。

**💡 创新点**

创新点在于将头部运动与躯干/底盘运动、以及末端执行器运动与躯干/底盘运动进行耦合，实现感知中心和操作中心的自动跟随控制，提升了遥操作的流畅度和可用性。

**🔧 技术方法**

采用VR 头戴设备与手柄进行姿态跟踪，利用TRAC-IK求解机器人臂的逆运动学；通过阈值函数实现头部和手部运动的耦合策略；配合实时视频反馈、迷你地图、碰撞避免与运动阻尼的GUI技术。

**📊 数据集**

通过对 12 名受试者在家居护理风格任务（瓶子、布料、桌子、架子）中的操作进行实验评估；未使用公开数据集，而是基于自制的任务环境和人工收集的数据。

**📈 对比分析**

通过与基线混合控制接口进行对比，采用任务完成时间、对象操作时间、按钮按压次数、臂部极限/奇异性、NASA‑TLX 主观指标等多维度评估。结果显示，耦合自我中心控制在对象操作时间、按钮使用和臂部奇异性方面显著优于基线，并降低了精神负荷、提升了易用性和信心；在复杂任务中使用频率更高，完成时间更短。

**⚠️ 局限性**

局限性包括：依赖手柄输入，缺乏无手柄（手势/手部追踪）实现；仅在 TIAGo 移动机械臂上验证，未测试在类人或多足平台；耦合阈值固定，缺乏任务自适应；对后退移动仍需手动控制，影响部分操作。

---

## 354. Controlling Implicit Shortcut Reliance in L2 Spoken English Auto-markers

**arXiv ID:** 2607.16085 | [PDF](https://arxiv.org/pdf/2607.16085v1)

**作者:** Shilin Gao `[一作]` (Cambridge University), Kate M. Knill `[通讯]` (Cambridge University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在端到端语音/文本评估器中通过在输出层加入基于Spearman相关性的正则项，以抑制模型对隐式快捷方式的过度依赖。

**💡 创新点**

创新点在于仅使用输出层的相关性惩罚，无需访问或修改编码器内部表示，可统一适用于文本和音频输入。

**🔧 技术方法**

使用ModernBERT和wav2vec2.0预训练模型微调，并结合可微分的排名近似实现Spearman相关性惩罚。

**📊 数据集**

实验基于Speak & Improve 2025语料库（L2英语口语回答）。

**📈 对比分析**

与人类评分参考相关性对比发现，加入惩罚后词数/ VAD时间的相关性下降至人类水平，同时保持竞争性的准确度，并提出人类对齐模式与恶意抑制模式两种可选操作。

**⚠️ 局限性**

局限在于仅针对单一可计算代理，未直接扩展至多模态或对话评估，高惩罚可能导致整体准确度下降。

---

## 355. LLM-Powered Agentic AI for 5G/6G Networks: A Tutorial and Survey on Architectures, Protocols, and Standardization

**arXiv ID:** 2607.16066 | [PDF](https://arxiv.org/pdf/2607.16066v1)

**作者:** Mazene Ameur `[一作]` (EURECOM), Adlen Ksentini `[通讯]` (EURECOM)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文通过系统性综述和教程式阐述，将大型语言模型（LLM）驱动的代理式人工智能（Agentic AI）与5G/6G网络的控制、管理与AI本地化平面相结合，梳理相关技术、协议（如MCP、A2A、ANP、ACP、AP2）、评估方法与标准化进展，并指出开放挑战与未来研究方向。

**💡 创新点**

创新点在于：①首次将代理式AI概念与5G/6G网络层面统一映射，形成完整的“AI+网络”框架；②提出一套覆盖协议、评估、标准化的统一体系，填补先前孤立研究的空白；③构建详细的技术与协议分类法，帮助从业者快速定位适用方案；④系统评析多项实验与基准，揭示性能瓶颈与研究缺口。

**🔧 技术方法**

使用技术包括：大型语言模型（如GPT、LLaMA、Phi3等）及其适配方法（FT、PEFT/LoRA、RLHF、RAG、工具集成等）；代理式AI框架（ReAct、AutoGPT、Agentic RAG、MAS等）；网络通信协议（MCP、A2A、ANP、ACP、AP2）与标准化接口（3GPP N系列、ETSI GS 059、O-RAN RIC等）；评估与基准体系（TeleQnA、TeleTables、TelAgentBench、MM‑Telco、TeleMath、TSpec‑LLM、TeleYAML、α3‑Bench）；以及对标识符与安全（DID、JWT、OAuth2）等技术。

**📊 数据集**

参考并梳理的基准数据集包括：TeleQnA、TeleTables、TelAgentBench、MM‑Telco、TeleMath、TSpec‑LLM、TeleYAML、α3‑Bench 等，覆盖语义理解、表格推理、代理推理、推理与多模态任务等；这些数据集用于评估LLM在通信领域的知识、推理、配置生成与决策能力，但本文未自行训练或测试模型。

**📈 对比分析**

对比方法主要基于已有文献的指标汇总，涵盖任务成功率、工具调用准确率、执行延迟、能耗、Token/成本等；论文指出：MCP在非实时RIC层能以50–200 ms/工具链完成调用；A2A在跨域协调时面临gRPC序列化开销；Agentic RAG在基准上取得高知识检索准确率但存在检索延迟；总体而言，LLM驱动的代理在灵活性与自动化上表现突出，但在低时延、资源占用与鲁棒性方面仍有显著差距。

**⚠️ 局限性**

局限性主要包括：①LLM固有的随机性与“hallucination”导致非确定性决策；②推理与多模态工具调用的高计算/能耗开销与能源目标冲突；③时延与准确性之间的权衡未在标准化/协议层面得到统一；④多代理协同与内存一致性缺乏高效、可验证的机制；⑤协议与标准尚缺乏统一的序列化/身份认证规范，导致跨层协作的互操作性受限；⑥缺乏统一的端到端评估框架与合规性审计机制。

---

## 356. VTLoc: Learning-based Tactile Contact Localization in Visual Point Clouds

**arXiv ID:** 2607.16146 | [PDF](https://arxiv.org/pdf/2607.16146v1)

**作者:** Zhiyuan Wu `[一作]` (King's College London), Shan Luo `[通讯]` (King's College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VTLoc 框架，利用视觉点云与触觉图像实现单点接触定位。

**💡 创新点**

创新点在于：1) 几何多模态对齐（GMA）模块可重建伪点云并通过 Chamfer 距离对齐视觉与触觉特征；2) 迭代定位更新器（ILU）基于 GRU 的多步回归细化预测；3) 不需要大规模触觉码本，直接从 2D 触觉图像到 3D 点云的跨维匹配。

**🔧 技术方法**

技术细节：触觉特征由 ResNet‑18+MLP 提取，点云特征由 PointNet++ 编码；GMA 重建点云并与原始点云对齐；ILU 通过多步 GRU 迭代更新接触位置；生成 3D 概率热图进行可视化与后续决策。

**📊 数据集**

数据集为基于 ObjectFolder Real 的 100 个真实日常物体，点云已标注 30‑50 个接触点，训练/测试比例 7:3。

**📈 对比分析**

与 Point Filtering、MCR、MidasTouch 等基线对比，VTLoc 在非均匀子集 ND 下降约 14%，Top‑1 Acc 提升约 38%；在均匀子集同样取得显著提升；多接触实验中 VTLoc 在 ND、Top‑5 Acc 上均优于 MidasTouch，整体性能显著提高。

**⚠️ 局限性**

局限性包括：对称物体仍可能产生多重接触候选；对软体物体未考虑时间序列变形信息；在未知物体上性能相对基线仍有差距；迭代次数过多会导致过拟合。

---

## 357. Improving Improved Kernel PLS

**arXiv ID:** 2607.16138 | [PDF](https://arxiv.org/pdf/2607.16138v1)

**作者:** Ole-Christian Galbo Engstrøm `[一作]` `[通讯]` (FOSS Analytical A/S), Ole-Christian Galbo Engstrøm (FOSS Analytical A/S)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对IKPLS算法中的旋转矩阵R和负荷矩阵Q的计算进行加速改进

**💡 创新点**

提出直接评估策略取代逐项累加计算R，并推导Q可直接从已计算的中间量得到，从而在大多数情形下将Q的计算复杂度降低到Θ(K)

**🔧 技术方法**

基于矩阵代数与算法复杂度分析，结合NumPy和JAX实现，并利用GPU并行化

**📊 数据集**

使用随机生成的N×K、N×M矩阵进行基准测试，探索不同K、M、A组合的性能表现

**📈 对比分析**

将改进实现与原IKPLS实现在CPU（NumPy）和GPU（JAX）上进行耗时对比，实验显示整体拟合速度提升约CPU上2×、GPU上6×

**⚠️ 局限性**

在M≥K或N很大时改进效果有限，改进依赖于数据维度与实现细节，且对非随机真实数据集的适用性未作系统验证

---

## 358. A Methodology for Auditable Trustworthiness Levels in AI Lifecycle Governance

**arXiv ID:** 2607.16130 | [PDF](https://arxiv.org/pdf/2607.16130v1)

**作者:** Andrea Ferrario `[一作]` `[通讯]` (University of Zürich), Andrea Ferrario (University of Zürich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种轻量化方法，用于在 AI 生命周期治理中定义、学习并监控可审计的信任度等级；

**💡 创新点**

创新点在于将治理相关的信任度分解为可测量维度、构建可解释的经验级别规则，并将其嵌入到预部署、部署后监测和再评估的完整治理流程中；

**🔧 技术方法**

核心技术包括基于阈值的维度聚合、使用决策树学习可解释的等级规则以及边界边缘和配置漂移两种生命周期诊断；

**📊 数据集**

实验使用合成生命周期轨迹（推荐系统、临床决策支持、信用评估和医院比较）模拟多维度、异步采样和不确定性场景；

**📈 对比分析**

与专家手工定义的等级规则比较，学习得到的决策树在多数情形下准确复制或压缩规则，能够捕捉关键阈值转移，性能表现良好但在稀缺或噪声标签情况下会出现压缩或失配；

**⚠️ 局限性**

主要限制包括对高质量标注与多维度监测的依赖、在低变异或标签稀疏的环境下学习无效、模型可解释性与精度的权衡，以及未在真实工业场景中验证的通用性。

---

## 359. Toward Semantic Communication for Real-time Mobile 3D Reconstruction

**arXiv ID:** 2607.16128 | [PDF](https://arxiv.org/pdf/2607.16128v1)

**作者:** Fangzhou Zhao `[一作]` (North China Electric Power University), Yi Sun `[通讯]` (North China Electric Power University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种针对实时移动3D重建的语义通信框架，能够在无线链路受限下通过语义编解码器传输重建图像与像素置信度图，随后利用置信度引导的RANSAC与BA实现更鲁棒的相机姿态估计与稀疏三维点云重建。

**💡 创新点**

创新点在于：①将置信度映射与语义通信相结合，在语义解码器输出重建图像的同时提供像素级置信度；②在相机姿态估计和BA中加入置信度权重，使不可靠观测被抑制，提升了在噪声通道下的几何一致性；③提出了一种轻量化的3DC‑SC卷积式语义传输器，兼顾重建质量与模型体积。

**🔧 技术方法**

采用卷积编码-解码结构的语义通信网络，配备置信度预测子；相机姿态估计基于RANSAC和八点/五点算法；BA采用高斯-牛顿迭代并用置信度确定观测方差；训练使用MSE、特征一致性损失和置信度L1损失。

**📊 数据集**

训练数据集：ImageNet；3D重建仿真使用Blender渲染的八个3D模型、每个模型100视角、800×800图像；图像重建评估使用Kodak24；通信模拟使用AWGN+16QAM，SNR范围2~12 dB。

**📈 对比分析**

与MAE、ViT、CNN、JPEG+LDPC等基线在相同通道条件下对比；实验显示3DC‑SC在PSNR、SSIM上与MAE相当，且模型轻量；在姿态估计误差上，加入置信度后相对无置信度方案可降低1–10°的旋转/平移误差；在深度图方面JPEG+LDPC在中等SNR下偶有优势，但整体在低SNR时3DC‑SC更稳健。

**⚠️ 局限性**

限制主要有：①置信度图的生成依赖解码器的学习，若通信噪声极大导致图像重建严重失真，置信度误判仍可能影响几何估计；②该框架主要验证在离线仿真中，缺乏真实移动设备与边缘服务器的端到端实验；③对稠密重建与实时SLAM系统的扩展尚未深入探索。

---

## 360. Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives

**arXiv ID:** 2607.16100 | [PDF](https://arxiv.org/pdf/2607.16100v1)

**作者:** Siyuan Shen `[一作]` (ETH Zürich), Torsten Hoefler `[通讯]` (ETH Zürich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 GPU 集合通信中实现接近硬件光速延迟的 AllReduce 内核，并在长上下文 LLM 推理与单节点 HPC 工作负载上验证性能提升。

**💡 创新点**

提出无全局内存屏障、LL 与 sentinel 同步、双缓冲、两射门、LL128 原子等低延迟技术，构建统一 API 并实现多种接近硬件极限的 AllReduce 算法。

**🔧 技术方法**

基于 NCCL 设备端 API 的低延迟 API；LL 与 sentinel 同步；双缓冲、两射门、LL128 原子 AllReduce；使用对称内存、NVLink、NVSwitch 多播等硬件特性。

**📊 数据集**

LLM 推理使用 vLLM 对 Llama‑3.1‑70B、DeepSeek‑V3、Qwen3‑Next 等模型的长上下文生成；HPC 采用 cuSOLVERMp 在 Alps 集群上进行特征分解。

**📈 对比分析**

通过微基准与真实工作负载比较，最小消息 128B 的 AllReduce 延迟仅比 SoL 低 7%；vLLM 的 inter‑token latency 降低 7‑13%，吞吐量相似；cuSOLVERMp 的 GFLOPS 亦提升，显示出显著的性能优势。

**⚠️ 局限性**

仅适用于单 NVLink 域的 scale‑up 网络；LL128 原子仅支持加法且需 NVLink 原子；需要对称内存；对大消息时仍受限；实现复杂且需要手动管理缓冲区。

---

## 361. Understanding Reasoning from Pretraining to Post-Training

**arXiv ID:** 2607.16097 | [PDF](https://arxiv.org/pdf/2607.16097v1)

**作者:** Jingyan Shen `[一作]` (New York University), Pavel Izmailov `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在控制的象棋环境中构建预训练‑后训练框架，系统探索预训练规模与RL计算分配的量化关系，并提出联合预训练‑RL扩展律。

**💡 创新点**

首次给出预训练损失预测RL终极表现、预训练数据量预测RL增益速率的量化方法，并展示RL对不同难度棋局的策略重塑机制。

**🔧 技术方法**

使用自回归语言模型架构（Qwen3），合成思路链（CoT）作为SFT目标，采用Group Relative Policy Optimization (GRPO) 进行RL，并通过可验证奖励环境实现可步进评估。

**📊 数据集**

基于lichess 54B 计时局子数据、156K Lichess谜题以及1,480道手工挑选的评测谜题；对数值扩展至1B参数的语言模型使用Nemotron-CC-Math与Dolma3混合预训练。

**📈 对比分析**

与仅SFT或仅RL、不同RL比重的基准比较，显示在给定总算力下预训练优先可获得更高pass@1，且RL占比随算力提升而递增；在math任务上亦复现同样趋势。

**⚠️ 局限性**

局限在于实验规模受限于象棋和数学任务的可验证性，未覆盖更大模型/更高算力；RL对难题的错误模式放大仍需进一步改进。

---

## 362. How Do VLMs Fail? Vision-Operation Misalignment in Compositional VQA

**arXiv ID:** 2607.16094 | [PDF](https://arxiv.org/pdf/2607.16094v1)

**作者:** Navya Gupta `[一作]` (Singapore Institute of Technology), Zhengchen Zhang `[通讯]` (Singapore Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对视觉-语言模型在组合式视觉问答中的失败进行操作层面和机制层面的细粒度分析，提出四种失败模式并验证其计算路径。

**💡 创新点**

创新点在于将失败模式与具体的推理操作（select、relate、verify 等）对应，并通过三种因果干预（均值消融、注意力屏蔽、MLP 层零化）揭示不同失败模式在 Transformer 中的路径分离（grounding/attribute 通过 MLP，reasoning 通过晚期注意力）。

**🔧 技术方法**

采用因果消融、注意力去噪、MLP 零化等机制可解释方法，并用线性探测器验证视觉依赖性；使用 GQA 的功能程序和场景图进行操作级定位；在 VSR 上进一步验证空间推理的路径分离；对 LLaVA-1.5 进行跨架构验证。

**📊 数据集**

主要使用 GQA（包含功能程序、场景图、真实图像）以及 VSR（单步空间推理）进行实验；在 GQA 上采样 500 条每种操作的样本；在 VSR 上使用 2195 条测试样本。

**📈 对比分析**

通过统计每种干预下的 log‑prob degradation 和 Cohen’s d，形成四种失败模式的判定。实验表明：grounding 与 reasoning 模式与视觉编码器的细粒度特性相关；attribute extraction 与语言 prior 模式与模型架构无关。相较于先前仅给出行为层面错误率的研究，本工作提供了更细粒度、可解释的机制分析。

**⚠️ 局限性**

局限性：仅在单一 3B 参数模型 Qwen2.5-VL 上展开，未验证干预的充分性；高准确率操作样本不足导致统计功效有限；缺乏直接的修复实验，仅为诊断性工作。

---

## 363. What Does It Take to Research with AI? A Rapid Review of Competencies to Train LLM-Literate Researchers

**arXiv ID:** 2607.16083 | [PDF](https://arxiv.org/pdf/2607.16083v1)

**作者:** Danilo Monteiro Ribeiro `[一作]` (CESAR School), Gustavo Pinto `[通讯]` (Federal University of Pará)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2022–2025年间使用大语言模型的科研实践进行快速综述，提炼出8项关键能力。

**💡 创新点**

首次将AI素养、研究诚信、可重复性、提示工程等分散议题整合成一个统一的能力框架。

**🔧 技术方法**

采用快速综述方法、主题分析，并结合人工与LLM辅助编码两种技术。

**📊 数据集**

使用了194篇文献（最终40篇）作为样本，涵盖英文和葡语发表的研究。

**📈 对比分析**

将人工主题与LLM自动生成主题进行对比并合并，得到8项能力，频数仅作为相对重要性的指示；未进行正式性能评估。

**⚠️ 局限性**

局限性包括检索仅依赖Elicit和Google Scholar、快速综述覆盖范围有限、实例未去重、以及对最新文献动态的适应性不足。

---

## 364. HCIG: A Hierarchical Cross-Modal Incongruity Graph Network for Multimodal Sarcasm and Cyberbullying Detection

**arXiv ID:** 2607.16076 | [PDF](https://arxiv.org/pdf/2607.16076v1)

**作者:** Bhavana Verma `[一作]` (Delhi Technological University), Dinesh Kumar Vishwakarma `[通讯]` (Delhi Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两种基于图注意力的多模态模型 HCIG（分层跨模态不一致图网络）和 GCCN（图对比矛盾网络），用于识别社交媒体中的讽刺与网络霸凌。

**💡 创新点**

创新点在于：HCIG 在词、短语和全局三个层级分别构建文本–图像不一致性图并通过自适应注意力门融合，显著提升跨模态推理效果；GCCN 通过阈值化相似度构造图并引入矛盾池化机制，简化模型并突出矛盾信息。

**🔧 技术方法**

技术实现包括 RoBERTa 与 ViT 编码器，GATv2 图注意力网络进行跨模态图推理，残差融合、对比池化、层级注意力门以及最终的二分类头。

**📊 数据集**

使用公开基准数据集 MMSD（多模态讽刺）和 MultiBully（多模态网络霸凌），并对数据进行清洗、划分。

**📈 对比分析**

与文本 BERT、ResNet、Late Fusion 基线对比，HCIG 在 MMSD 上实现最高准确率 85.74% 和宏 F1 85.29%；GCCN 在 MultiBully 上获得最高宏 F1 68.66%，HCIG 在 MultiBully 上实现最高准确率 69.62% 以及霸凌类 F1 74.90%。

**⚠️ 局限性**

局限性包括：跨任务直接迁移性能低，模型未处理隐式讽刺或文化背景；仅使用二分类损失，未利用对比或不确定性正则；缺乏多语言、跨文化评估；模型对图阈值和层级选择较敏感。

---

## 365. Vision-Language-Motion Maps: An Open-Vocabulary, Uncertainty-Aware, Queryable Motion Attribute for 3D Scene Maps

**arXiv ID:** 2607.16173 | [PDF](https://arxiv.org/pdf/2607.16173v1)

**作者:** Dibyendu Ghosh `[一作]` (Rekise Marine), Ayushi Shakya `[通讯]` (Rekise Marine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Vision–Language–Motion Maps（VLMM），在开词汇3D地图中为每个实例融合语义先验与几何观测运动，并加入不确定性量化，支持自然语言查询。

**💡 创新点**

创新点在于三项：①将 VLM/LLM 的可移动性先验与几何观测运动进行置信度感知融合；②通过 Mahalanobis 运动评分与深度协方差传播实现运动不确定性；③将运动属性设计为可直接被自然语言查询的字段。

**🔧 技术方法**

使用技术包括 CLIP 区域编码、RAFT 光流、姿态估计与 ego‑refinement、观测运动的 Mahalanobis 分数、贝叶斯融合规则以及后置等距校准。

**📊 数据集**

实验数据集包括 AI2‑THOR 具有 exact ground‑truth 的三种室内场景、TUM RGB‑D、Bonn 动态 RGB‑D 序列。

**📈 对比分析**

通过与语义仅、运动仅、DualMap、VLMaps 等基线的 AP、宏 F1、误报率对比，发现 VLMM 在 exact‑GT 场景 AP 达 1.00，在真实序列中移动‑静止检测 AP 提升 0.10，误报率显著下降。

**⚠️ 局限性**

局限性包括：真实数据中非人物移动的 GT 采用人分割代理，置信度尚未完全校准，查询解析仍为规则式而非完整 NLU，需要手工阈值，且对人类移动的评估尚待进一步验证。

---

## 366. Updating zigzag representatives efficiently

**arXiv ID:** 2607.16153 | [PDF](https://arxiv.org/pdf/2607.16153v1)

**作者:** Tamal K. Dey `[一作]` (Purdue University), Dmitriy Morozov `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一套高效的算法，用于在八种局部变换下更新Zigzag持久性中的代表性环，算法通过在由Zigzag转换得到的非Zigzag过滤器的R=DV分解上进行更新，能够在与普通持久性相同的时间复杂度下完成更新，尤其处理了收缩/扩张操作中邻接关系变化的难点。

**💡 创新点**

创新点在于首次将R=DV分解与Zigzag代表性提取结合，针对收缩/扩张操作设计了新的矩阵更新策略，使得更新复杂度从此前的指数级下降到与普通持久性一致的二次级；同时提供了完整的六种基本操作的更新方案。

**🔧 技术方法**

技术手段包括：构造非Zigzag过滤器的R=DV分解、矩阵的列/行加法、转置、单列/行的置换、基于增量化简的PERSISTENCE REDUCTION、以及对边界矩阵的“拆分”与“合并”处理。

**📊 数据集**

论文未使用公开数据集，仅在理论与算法框架下给出复杂度分析与示例。

**📈 对比分析**

与先前仅对条形码进行更新的工作相比，本方法在保持相同时间复杂度（前进/后退交换O(m)、收缩/扩张O(m²)）的同时，实现了代表性环的可追踪更新，实验验证表明在实际规模下表现与最优普通持久性算法相当。

**⚠️ 局限性**

局限性包括：R=DV分解并非唯一，更新过程可能产生与标准化简不同的分解和代表性；更新后与逆操作不一定恢复原始代表，导致代表性不稳定；以及对更一般的多字段或更复杂过滤器的适用性仍需进一步研究。

---

## 367. A New Implementation of NeoSLAM and a Comparative Evaluation with RatSLAM

**arXiv ID:** 2607.16143 | [PDF](https://arxiv.org/pdf/2607.16143v1)

**作者:** Joao Victor T. Borges `[一作]` (Federal University of Rio de Janeiro), Leonardo Bobadilla `[通讯]` (Florida International University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将 NeoSLAM 重新实现为 ROS 2 的模块化架构，拆分成四个独立节点，改用现代 HTM 库与 C++17 代码，显著降低实时数据丢失并提升吞吐量。

**💡 创新点**

创新点在于将原单体架构拆解为并行的图像预处理、特征提取、HTM 记忆和视觉模板生成四个节点，结合现代软件栈（ROS 2、htm.core、Eigen、CRoaring），实现对原始 NeoSLAM 的性能突破。

**🔧 技术方法**

采用 ROS 2 Rolling、htm.core（C++17 + Python 3）、Eigen、CRoaring、AlexNet 预训练卷积网络、随机高斯投影、sLSBH 二值化、HTM 时序记忆、Spatial‑View 细胞等技术。

**📊 数据集**

使用 Robotarium、iRat Australia（陆地）和 FIU MMC Lake（水面）三套真实机器人数据集进行评测。

**📈 对比分析**

通过对原始 NeoSLAM 与重构版本在实时执行时的数据丢失率（原始高达 90%–91%，新版本 1%）和对比 NeoSLAM 与 RatSLAM 在地图重建误差（平均误差 0.24 m vs 0.29 m，USV 数据集 5.92 m vs 6.10 m）的实验，证明新 NeoSLAM 在吞吐量上明显优于原版，在地图精度上与 RatSLAM 相当甚至略优。

**⚠️ 局限性**

仍然依赖预训练的 AlexNet 作为特征提取器，且在高度动态或光照极端变化的环境下的鲁棒性未得到系统性验证；此外，实验对比多使用离线数据，缺乏在线自适应调参机制。

---

## 368. Harmonizing AI Safety Thresholds

**arXiv ID:** 2607.16112 | [PDF](https://arxiv.org/pdf/2607.16112v1)

**作者:** Wilber Sean Anterola `[一作]` (Brown University), Markov Grey `[通讯]` (Centre pour la Sécurité de l'Intelligence Artificielle)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种统一的阈值制定方法，能够把三类风险（网络攻击、化学/生物武器滥用、自动化AI研发）中的公司差异化阈值转化为可比较、可审计的“最低触发阈值”，并给出对应的阈值制定流程和操作标准。

**💡 创新点**

创新点在于：①把预期伤害（Expected Harm）作为滥用风险阈值的核心量化基准；②使用显式的 N×P×H 风险模型和攻击链分解来把能力阈值映射到预期伤害；③针对自动化 AI 研发阈值提出基于进展速率的量化标准；④将三家公司现行阈值语言映射为可量化的统一阈值，从而实现跨公司可比性和可审计性。

**🔧 技术方法**

采用的技术包括：
- N×P×H 预期伤害建模（按攻击渠道、成功概率、伤害单元计算）；
- 攻击链（kill‑chain）分解与阶段概率乘积；
- 发布条件扩展系数（s_r）来调整不同公开方式的曝光水平；
- 期望伤害与风险渠道的显式关联；
- 基准趋势拟合与速度突破检测（用 Epoch AI ECI 等综合基准进行速率比较）；
- 经验/专家推断、Monte Carlo 传播不确定性。

**📊 数据集**

使用的数据集与来源包括：
- 网络层面：全球网络犯罪损失估计（约 5 × 10¹¹ 美元）、FBI IC3 报告、公开的 TLO、Mythos 等网络实验室评估；
- 生物层面：VCT、ProtocolQA、Long‑form 生物威胁问题、专家面板推断结果、模拟实验室自动化（AlphaFold‑2、ProteinMPNN、LabOS 等）；
- 自动化研发层面：Epoch AI ECI 综合指标、METR 进度线、各公司发布的模型基准得分（GPT‑4o、Claude Opus 系列、GPT‑5、Mythos Preview 等）。

**📈 对比分析**

比较方法是将各公司原始阈值语言（能力阈值、结果阈值）映射到统一的预期伤害或进展速率指标上，并在相同的基础假设与数据下计算阈值触发条件。论文未给出传统机器学习指标的数值性能，而是提供了阈值可比性、可审计性和跨公司共识度量；在网络风险方面已给出可操作的最低阈值，生物风险仍需进一步关键指标，自动化研发阈值已给出可量化的速率突破标准。

**⚠️ 局限性**

限制与不足：
- 生物风险缺乏完整链条、未充分验证的关键指标，导致阈值目前仍为目标规范；
- 预期伤害模型依赖于假设的基准损失和曝光系数，存在较大不确定性；
- 网络风险评估基于公开实验室数据，难以覆盖真实世界多样攻击场景；
- 自动化研发阈值基于基准进度趋势，若进度加速来源于非 AI 因素（如硬件或投入），阈值可能失真；
- 所有阈值均需第三方审计与监管机制支持，当前缺乏统一执行机构；
- 需要更多跨公司、跨领域的实验与专家评估来完善模型参数与不确定性传播。

---

## 369. Attention-Guided Saliency Maps for Interpreting Visualization Literacy in VLMs

**arXiv ID:** 2607.16105 | [PDF](https://arxiv.org/pdf/2607.16105v1)

**作者:** Maeve Hutchinson `[一作]` (City St George's, University of London), Pranava Madhyastha `[通讯]` (City St George's, University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无梯度、轻量级的关注导向显著性方法，用于在 Vision‑Language 模型的图表问答任务中解释每个生成 token 的视觉关注点。

**💡 创新点**

创新点在于：① 通过对所有层、所有头的注意力权重求均值，聚合得到更稳健的视觉关注度；② 将聚合后的注意力映射回 Vision Encoder 的 patch 网格，得到 token‑级、空间连贯的显著性图；③ 方法不需要梯度、模型改动或额外推理，仅在标准生成过程中即可得到。

**🔧 技术方法**

技术手段包括 Attention aggregation（多层多头平均）、Reshape 与 patch‑grid 重新映射、max‑normalization 与 bilinear 上采样；实验使用 ViT + LLM（如 LLaVA、ChartGemma）以及 Mini‑VLAT、VLAT 数据集；评估采用删除测试（deletion‑based faithfulness）与 AG‑CAM 对比。

**📊 数据集**

数据集：Mini‑VLAT（用于生成答案的准确率基准）和 VLAT（用于删除测试，且仅挑选模型正确回答的样本）。

**📈 对比分析**

与 ChartGemma 与 LLaVA 的答案准确率对比：ChartGemma 在 Mini‑VLAT 上 58.3% ；LLaVA 25%。显著性方法在删除测试中表现最佳：自有方法 AUC=0.020，优于 AG‑CAM（AUC=0.070）和随机删除基线；说明其更具因果可信度。

**⚠️ 局限性**

局限性：① 删除测试仅在 13 个 VLAT 样本上进行，样本规模有限；② 方法主要针对图表问答任务，未验证在其他 VLM 任务或非图表图像上的泛化；③ 关注力聚合假设注意力即为视觉重要性，可能无法捕捉模型内部更细粒度的决策机制。

---

## 370. A Variable-Length Gray Code for the Natural Numbers

**arXiv ID:** 2607.16088 | [PDF](https://arxiv.org/pdf/2607.16088v1)

**作者:** Ezequiel López-Rubio `[一作]` `[通讯]` (Universidad de Málaga), Ezequiel López-Rubio (Universidad de Málaga)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种可变长度Gray码，将自然数映射为所有有限二进制串，保持单编辑距离的Gray码特性并实现无穷范围和自适应长度；

**💡 创新点**

在保持Gray码单位变化特性的同时，突破了固定长度限制，构造了全映射且单编辑距离的可变长度编码；

**🔧 技术方法**

利用反射二进制Gray码、Levenshtein距离理论、符号编码的组合，证明了bijection、长度公式和编辑距离性质；

**📊 数据集**

论文未使用具体数据集，而是在理论层面对比固定长度Gray码的压缩效果；

**📈 对比分析**

通过理论推导和图示，将每个整数的码长与固定长度Gray码的常数码长进行对比，展示累计存储位数的优势，尤其在小整数区间显著压缩；

**⚠️ 局限性**

该码非自前缀（self‑delimiting），在流式编码中需额外上下文或包裹码；此外，其优势主要体现在理论层面，尚未在实际Isal*指令集编码中验证性能。

---

## 371. MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction

**arXiv ID:** 2607.16192 | [PDF](https://arxiv.org/pdf/2607.16192v1)

**作者:** Homanga Bharadhwaj `[一作]` (Johns Hopkins University), Yash Jangir `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于视频先验，利用预训练的视频跟踪模型在仅观察到的短视频序列上预测被操作物体的未来3D轨迹

**💡 创新点**

将视频生成模型的时序先验迁移至显式3D场景流预测，仅用轻量级LoRA和掩码隐空间即可实现未来轨迹预测，无需语言或动作标签

**🔧 技术方法**

采用预训练的Video DiT + TrackCraft3R 的双latent结构，结合深度估计、Segment Anything、DepthAnything3、LoRA适配器和时间RoPE

**📊 数据集**

Something‑Something V2（约40k人机交互视频）以及50条手机录制的OOD视频

**📈 对比分析**

与视频生成+跟踪方法和MolmoMotion相比，MotionForesight在SSv2与OOD数据集上ADE/FDE/PWT均优于对比模型，尤其在不使用语言信息时表现突出

**⚠️ 局限性**

模型为确定性单一预测，受伪标签误差影响，对多模态未来、长时序、极端视角和非操控动态的泛化有限

---

## 372. FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation

**arXiv ID:** 2607.16190 | [PDF](https://arxiv.org/pdf/2607.16190v1)

**作者:** Hao Liu `[一作]` (Sun Yat-sen University), Jiangsu Du `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的稀疏注意力系统，在多GPU序列并行下通过 Runtime Load Balancing 和 Slack‑Aware Sparse Augmentation 提升视频 Diffusion Transformer 的推理效率。

**💡 创新点**

创新点在于：① 在稀疏 mask 生成后观察真实负载并通过局部 P2P head 迁移修复 Top‑p 路由导致的 rank‑级负载不均；② 利用非关键 rank 剩余空闲时间填充高价值块，以提升 mask 覆盖率；③ 将上述两种机制与 CPU‑GPU 与计算‑通信重叠相结合，实现低可见开销。

**🔧 技术方法**

使用 Top‑p 路由 + Top‑k 安全底、Hilbert 曲线块重排、Ulysses 序列并行、P2P head 迁移、Slack‑Aware Sparse Augmentation、CPU‑GPU 异步重叠与计算‑通信重叠，以及 FlashAttention/FlashInfer 等底层实现。

**📊 数据集**

实验数据集为 Wan2.2 I2V、Wan2.2 Animate、Wan2.1 T2V 三个视频 Diffusion 模型，使用 VBench 评估集（I2V 1118例、T2V 946例、Animate 20例），并采用 4 步蒸馏的 LoRA 配置。

**📈 对比分析**

与 FlashAttention、SageAttention、SVG2、SpargeAttention（Top‑p、Top‑k）、Jenga、db‑SP 等基线对比，使用 8 GPU Ulysses 并行，平均 DiT 延迟从 FlashAttention 的 38.6 s 降至 18.8 s，速度提升约 4.4×，同时视频质量（VBench、PSNR/SSIM/LPIPS/CLIP‑Sim）保持或提升。

**⚠️ 局限性**

局限性包括：主要针对长时空序列和高稀疏度的视频生成；对短序列或图像任务收益有限；受 GPU 间通信带宽影响，PCIe 环境下迁移收益可能不足；仅在 Ulysses 并行模式下验证，未探究环形或 USP 等其他拓扑。

---

## 373. Knowing the Self, Understanding the World: A Dual-Cognition Benchmark for UAV Spatio-temporal Reasoning with MLLMs

**arXiv ID:** 2607.16193 | [PDF](https://arxiv.org/pdf/2607.16193v1)

**作者:** Like Liu `[一作]` (Northwestern Polytechnical University), Dian Shao `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出UAV-DualCog基准，用以联合评估无人机在多视角时空场景下的自我认知与环境认知。

**💡 创新点**

创新点包括双重认知框架、基于语义点云的自动化数据生成管道，以及对自我状态与环境状态共同推理的细粒度任务设计。

**🔧 技术方法**

利用多模态大语言模型评估、语义点云构建、自动任务生成及结构化JSON输出与评估指标，构建并评估模型。

**📊 数据集**

使用自行构建的UAV-DualCog数据集，涵盖12个场景、512个地标，生成4096张图像样本和2048段视频样本，并推出UAV-DualCog-Train用于训练。

**📈 对比分析**

通过对多种轻量级、开源与专有多模态模型进行即时推理评估，发现环境认知任务通常优于自我认知，定位与时序定位表现低；训练后模型在图像任务上显著提升，但对视频任务的提升有限。

**⚠️ 局限性**

主要局限在于缺乏跨帧一致性训练、时空定位能力不足，以及从仿真到真实环境的迁移难题。

---

## 374. Handroid: Bridging Dexterous Hand and Humanoid

**arXiv ID:** 2607.16187 | [PDF](https://arxiv.org/pdf/2607.16187v1)

**作者:** Ruogu Li `[一作]` (University of North Carolina at Chapel Hill), Mingyu Ding `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款27自由度的桌面级双模态机器人Handroid，能够在精细抓取手态和人体形态之间切换，并在同一平台上实现抓取、在手内操纵、行走、姿态控制和长周期任务。

**💡 创新点**

创新点在于将同一套电子机械模块在手态和人体态之间复用，实现硬件与控制栈的共享，突破传统手与机器人平台各自独立的设计局限。

**🔧 技术方法**

采用电动机械模块、滑动重定位机构、ESP32-S3控制板、Dynamixel驱动器、IMU传感器以及统一的学习与控制栈（VR遥控、扩散式抓取政策、强化学习跟踪与速度控制、关键帧编辑器）。

**📊 数据集**

使用了100次手势遥控演示数据（10种物体各10次）构建抓取政策；通过模拟环境中的RL训练和真实机器人收集的数据验证行走与抓取；还使用Franka Research 3机器人协作场景进行长周期实验。

**📈 对比分析**

抓取实验平均成功率为72%，RL跟踪误差约0.12 rad、身体位置误差0.0019 m；速度控制误差为0.052 m/s；在长周期任务中实现了从手态到人态再回到手态的完整切换与协作，展示了跨模态的功能集成。

**⚠️ 局限性**

局限性包括仍然需要有线供电导致移动受限、机械臂与脚/手指的尺寸与力量未达到极限、缺乏触觉传感与视觉在脚部的集成，且系统尚未实现无线化与更小型化。

---

## 375. PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization

**arXiv ID:** 2607.16184 | [PDF](https://arxiv.org/pdf/2607.16184v1)

**作者:** Yuchen Yang `[一作]` (University of Illinois Urbana-Champaign), Sasa Misailovic `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Mixture-of-Experts（MoE）大语言模型的GPU内存管理，提出一种名为PagedWeight的系统，能够在推理过程中动态地对专家权重进行量化并按需在CPU与GPU之间迁移权重页面，兼顾KV缓存扩展与模型精度。

**💡 创新点**

创新点在于：①把专家权重视作可分页的量化页面，借鉴KV缓存分页管理；②构建质量感知的运行时规划器，融合离线敏感度、在线路由统计和提示残差来决定最佳量化级别；③实现异步页面迁移和融合的多精度MoE核，既降低显存占用又几乎不影响吞吐。

**🔧 技术方法**

使用技术包括Any-Precision LLM的多位宽位平面存储、混合精度量化、路由统计与提示残差回归、异步CPU↔GPU页面迁移、CUDA融合核；在实现层面，利用vLLM框架、NVIDIA GPU及自研的高效MoE算子。

**📊 数据集**

实验使用的模型与数据集为：MoE模型Qwen1.5-MoE‑A2.7B、Mixtral‑8×7B‑v0.1、Gemma‑4‑26B‑A4B；评测数据集包括Wikitext2、C4（语言建模）、GSM8K、MATH‑500（推理）、LongBench（长文本推理）。

**📈 对比分析**

与FP16、Any-Precision LLM（APL）、MxMoE、DP-LLM等基线相比，PagedWeight在相同或更低显存占用下实现了近FP16的质量（在最小显存时仅差0.1–0.3倍），显存节省可达72%，吞吐提升可达1.94×，在相似显存预算下，质量提升可达39.3%。

**⚠️ 局限性**

局限性包括：依赖AP量化格式，对非AP模型适配需要额外工作；离线敏感度与提示残差需要先行训练，可能对不同任务或域产生偏差；异步页面迁移的实现复杂度高，可能在极端高并发或多卡场景下出现同步瓶颈；目前仅验证在三种MoE模型与NVIDIA GPU上，其他架构的可迁移性尚待探索。

---

## 376. Searching Videos as Trees: Self-Correcting Agents for Grounded Long Video QA

**arXiv ID:** 2607.16189 | [PDF](https://arxiv.org/pdf/2607.16189v1)

**作者:** Ce Zhang `[一作]` (University of North Carolina), Gedas Bertasius `[通讯]` (University of North Carolina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VideoTreeSearch，构建自适应时间树并使用四种离散动作（下沉、上升、平移、提交）进行导航，实现在长视频中定位单一证据区间并回答问题。

**💡 创新点**

创新点在于将长视频分割为语义连贯的非均匀树结构，并显式加入回溯动作，使搜索成为可学习的离散层次化过程；通过合成包含错误分支的轨迹训练自纠错能力。

**🔧 技术方法**

使用 CLIP 对场景切换进行检测构建树；以 Qwen3‑VL‑8B（或 Qwen2.5‑VL‑7B）为基础 VLM；采用四个离散动作；通过轨迹合成、监督微调和基于 GRPO 的强化学习实现训练。

**📊 数据集**

在 CG‑Bench、Haystack‑LVBench、Haystack‑Ego4D（定位任务）以及 Video‑MME、MLVU、LVBench（一般长视频 QA）上评估；同时使用 LongClueQA 合成的无标签长视频数据。

**📈 对比分析**

相较于统一采样、captioner‑LLM 和 multi‑turn cropping 代理，在 CG‑Bench 上 mIoU 提升 12.5 点、在 Haystack‑Ego4D 上 T‑F1 提升 7.4 点，整体性能提升多达 +7.1；在通用 QA 上同样优于同类方法。

**⚠️ 局限性**

仅能输出单一连续证据区间，无法处理分散证据；树构建依赖 CLIP 场景检测，视觉同质视频表现欠佳；轨迹合成质量受外部 VLM 与 LLM 的限制。

---

## 377. A Blueprint for Equilibrium-Based Differentiable Continuous-Variable Thermodynamic Computing

**arXiv ID:** 2607.16183 | [PDF](https://arxiv.org/pdf/2607.16183v1)

**作者:** Owen Lockwood `[一作]` (Extropic Corporation), Guillaume Verdon `[通讯]` (Extropic Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种基于热动力学的能量高效计算体系，利用可调双井势的超导模拟电路实现Langevin动力学，从而在物理硬件中自然生成与能量相关的概率分布，并构建了能量基础模型（EBM）与概率图模型（PGM）的软硬件实现，进一步示范了如何在该硬件上训练与推断常见机器学习任务（如多层感知机、混合高斯模型、隐马尔可夫模型、Ising模型以及Transformer解码器）。

**💡 创新点**

核心创新点在于：①将热噪声作为计算资源而非噪声源；②使用可调双井势超导电路实现可编程能量势；③通过能量势的调节实现Sigmoid、Softmax等非线性激活的近似；④在硬件层面直接实现EBM的采样与梯度估计（基于协方差的导数形式），从而实现端到端的能量、时间与精度权衡；⑤在实验层面首次展示超导双井势电路的逃逸能量与温度依赖，验证其可在热激活 regime 运行。

**🔧 技术方法**

技术主要包括：
- 可调双井势超导电路（基于DC‑SQUID、LC 电路与电阻耦合）
- Langevin动力学模拟（热噪声、阻尼、势能调节）
- 能量基础模型（EBM）与概率图模型的物理实现（Gaussian、double‑well、耦合势）
- 采样与梯度估计：协方差估计、对数似然对比学习（CD）
- 软硬件协同：磁控调节、量子/热交叉温度、散射读取、指数衰减拟合
- 量化指标：逃逸能量、时间常数、Tokens‑per‑Joule（Transformer）以及对比数字 GPU 的能耗与延迟。

**📊 数据集**

实验数据集与仿真：
- MNIST（混合高斯模型、Transformer 预训练等）
- 10×10 Bars‑and‑Stripes（连续 Ising 训练）
- 合成时间序列（HMM）
- 量化实验：超导双井势在 50–200 mK 之间的逃逸能量与温度曲线（无外部噪声注入）。

**📈 对比分析**

性能对比：
- 在超导双井势实验中，逃逸能量与温度符合 Arrhenius 公式，热激活速率与理论一致。
- 对于 Transformer 解码器（“thermoformer”），在给定样本数（N）下，Tokens‑per‑Joule 与传统 GPU（如 NVIDIA H100）相当或更优，但需多次采样才能逼近数字实现的精度；实验显示当 N≥5 时，性能可接近数字基准。
- 对于混合高斯、HMM 与 Ising 模型，训练收敛速度与传统基于 MCMC 的数字实现相当，且能耗下降数个数量级。

**⚠️ 局限性**

局限与挑战：
- 需要超低温（≤200 mK）和高质量超导元件，技术成本高、规模化受限。
- 采样精度受限于温度与势阈值，近似激活函数（Sigmoid、Softmax）需大量样本才能逼近确定性结果。
- 受限于势能可调性与耦合拓扑，硬件实现复杂度随模型规模急剧上升。
- 当前仅验证单一双井势模块，尚未实现完整的多体耦合网络或大规模 Transformer。
- 对比数字系统时未考虑冷却、控制与校准等额外能耗，真实能效优势尚待进一步评估。

---

