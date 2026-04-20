# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-20 | 今日论文总数: 478

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. PAWN: Piece Value Analysis with Neural Networks

**arXiv ID:** 2604.15585 | [PDF](https://arxiv.org/pdf/2604.15585v1)

**作者:** Ethan Tang `[一作]` (Arizona State University), Zhongju Zhang `[通讯]` (Arizona State University)

**通讯引用:** 2955 | [OpenAlex ID](https://openalex.org/A5009085358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种利用CNN自编码器提取棋盘整体状态作为上下文，改进单棋子相对价值预测模型

**💡 创新点**

创新点在于将全局棋盘表示嵌入到MLP预测中，实现对单棋子价值的上下文感知

**🔧 技术方法**

使用CNN自编码器、MLP以及Huber损失，结合Stockfish 17评估做标签

**📊 数据集**

数据集来源于两套GM级棋局：Carlsen 赛季数据库（MC）和2023年所有GM经典赛（TF）共计2400多万条棋子价值样本

**📈 对比分析**

与传统仅用棋子类型与位置的MLP基线相比，加入CNN上下文的MLP+CNN在验证集MAE降低约12-16%，最高达65.45cp（约0.65子）

**⚠️ 局限性**

局限包括对单棋子独立定义忽略相互作用、依赖Stockfish 17标签、训练-验证重叠、对不同对局分布的泛化受限

---

## 2. A Comparative Study on the Impact of Traditional Learning and Interactive Learning on Students' Academic Performance and Emotional Well-Being

**arXiv ID:** 2604.15335 | [PDF](https://arxiv.org/pdf/2604.15335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 3. Verification Modulo Tested Library Contracts

**arXiv ID:** 2604.15533 | [PDF](https://arxiv.org/pdf/2604.15533v1)

**作者:** Abhishek Uppar `[一作]` (Indian Institute of Science), Adithya Murali `[通讯]` (University of Wisconsin)

**通讯引用:** 801 | [OpenAlex ID](https://openalex.org/A5005922328)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种通过合成测试可通过的库合同和客户端不变式，来验证使用大型库的客户端程序的方法。

**💡 创新点**

创新点在于：①引入“上下文合同”(Contextual Contracts)概念，简化合同的表达；②构建基于ICE学习的通用CHC求解器，并将LLM与符号方法结合实现合同与不变式的自适应学习；③将合同测试与合成迭代相结合，形成可扩展的验证-测试闭环。

**🔧 技术方法**

技术手段包括：约束Horn子句(CHC)求解、ICE学习框架、决策树学习器、Gemini v2.5 Pro LLM、AFL++模糊测试引擎，以及在Python中实现的综合工具Dualis。

**📊 数据集**

使用的数据集是43个C++客户端程序，覆盖了Facebook Folly、Google Abseil、Android源码及SV-COMP Java benchmark的库实现，库规模从100到3000行代码不等。

**📈 对比分析**

在与现有自动化工具（如SeaHorn）对比时，Dualis在模块化合同下成功解决19/43、上下文合同下解决41/43个基准；相较于纯ILP求解器，ICE学习器的引入显著提高了解决率（模块化18%提升，上下文合同35%提升）。

**⚠️ 局限性**

局限性包括：依赖于足够表达力的观察者方法；对LLM的非确定性与收敛性无理论保证；在存在脆弱测试覆盖的错误程序上可能产生伪真合同；当前仅支持无堆操作的逻辑与非递归数据结构。

---

## 4. StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models

**arXiv ID:** 2604.15416 | [PDF](https://arxiv.org/pdf/2604.15416v1)

**作者:** Dingzhi Yu `[一作]` (University of Illinois Urbana Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了StoSignSGD，一种在SignSGD中引入无偏结构随机性的优化算法，解决非光滑目标的发散问题，并在LLM训练与微调中提升效率。

**💡 创新点**

创新点包括：①无偏随机化sign算子；②动态跟踪梯度∞范数实现自适应预条件；③构建无偏sign转换框架，将任意优化器转为sign版；④在凸与非凸非光滑场景下给出最优收敛/复杂度证明。

**🔧 技术方法**

使用了无偏随机化sign算子、在线凸优化与非凸转换、坐标自适应预条件、结构化噪声注入、随机比例扩展、理论下界证明以及LLM低精度FP8训练技术。

**📊 数据集**

实验数据集包括：GPT-2预训练（OpenWebText）、Qwen2.5-7B在NuminaMath-CoT上的微调、GSM8k、MATH、Alpaca、Llama-3.1-8B、Mistral-7B-v0.1等。

**📈 对比分析**

与AdamW、SignSGD、Lion等基线比较；在FP8预训练中AdamW发散，StoSignSGD在验证损失相同下减少53%–30% tokens，速度提升1.44×–2.14×；在7B LLM数学推理中相对AdamW提升3%–5%；在指令跟随任务中保持最优或接近最佳表现。

**⚠️ 局限性**

局限性包括：对极大规模分布式实现细节尚未深入；结构化噪声设计需经验调参；在低精度场景下优势显著，其他精度场景验证不足；理论证明依赖坐标Lipschitz与均值方差假设；对长期训练稳定性及泛化性能的深入分析仍待进一步研究。

---

## 5. PRL-Bench: A Comprehensive Benchmark Evaluating LLMs' Capabilities in Frontier Physics Research

**arXiv ID:** 2604.15411 | [PDF](https://arxiv.org/pdf/2604.15411v1)

**作者:** Tingjia Miao `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8915 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了前沿 LLM 评测基准 PRL‑Bench，涵盖 100 篇 Physical Review Letters 论文，并系统评估当前主流 LLM 在 5 个物理子领域的长周期科研推理与多步推导能力。

**💡 创新点**

① 把真实理论与数值研究转化为可验证的长周期、探索导向任务；② 设计子任务级别答案与 Rubrics；③ 采用 LLM‑as‑judge 的细粒度评分框架，首次在物理研究场景下评测 LLM 的规划、推导和数值验证性能。

**🔧 技术方法**

使用大型语言模型（如 GPT‑5.4、Gemini‑3.1‑Pro、Claude‑Opus‑4.6 等）结合代码解释器、结构化 Rubrics 以及 LLM‑as‑judge 的评判方法。

**📊 数据集**

从 Physical Review Letters 135–136 期（2025‑2026）筛选的 100 篇权威论文，涵盖天体物理、凝聚态物理、高能物理、量子信息与统计物理。

**📈 对比分析**

对每个模型在 5 次重复下执行所有子任务，使用预设分值对答案与 Rubrics 进行加权，最终分数归一化到 0–100。Gemini‑3.1‑Pro 以 44.27 分排名第一，但所有模型平均分均低于 50，表明当前 LLM 在长周期科研推理上仍显不足。

**⚠️ 局限性**

任务给定了较多背景信息，缺乏假设生成与驳斥过程；数据集单一来源且分层分类不够细致；未覆盖跨学科问题，也缺少实验验证与大规模真实科研工作流程的验证。

---

## 6. Beyond Single-Model Optimization: Preserving Plasticity in Continual Reinforcement Learning

**arXiv ID:** 2604.15414 | [PDF](https://arxiv.org/pdf/2604.15414v1)

**作者:** Lute Lillo `[一作]` (University of Vermont), Nick Cheney `[通讯]` (University of Vermont)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5112965505)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出 TeLAPA 框架，使用行为多样化的策略档案和共享潜在空间来解决持续强化学习中的稳定‑塑性权衡。

**💡 创新点**

创新点：① 关注保留策略邻域而非单点；② 结合质量多样性（QD）与轨迹嵌入构建共享行为空间；③ 通过 anchor、对比蒸馏与周期性重嵌入维持潜在空间稳定；④ 通过实证展示单策略保留不足。

**🔧 技术方法**

技术手段：PPO、MAP‑Elites、轨迹自编码/对比嵌入、对比损失、蒸馏对齐、周期性重嵌入、归一化、少样本评估与选择。

**📊 数据集**

数据集：MiniGrid 任务集合（A,B,C,D,E）及其重访序列，使用 20 次独立实验评估。

**📈 对比分析**

对比方法：Scratch、Scratch‑Reuse、Finetune、Finetune‑Reset、EWC、L2Init、DFF、Shrink‑and‑Perturb、TeLAPA‑Static 与 TeLAPA。TeLAPA 在平均成功率、时间‑到阈值、覆盖率、阈值保留等指标上均优于单模型基线，尤其在快速恢复和多任务覆盖上表现突出。

**⚠️ 局限性**

局限性：仅在小型离散网格任务验证；潜在空间对齐仍需改进；归档规模与计算成本未完全评估；对连续控制或复杂视觉任务的泛化尚未验证。

---

## 7. Making It Work Is the Work: Engineering Maturity as Epistemic Work

**arXiv ID:** 2604.15330 | [PDF](https://arxiv.org/pdf/2604.15330v1)

**作者:** Danny Leen `[一作]` (University of Hasselt - Flanders Make), Kris Luyten `[通讯]` (University of Hasselt - Flanders Make)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

探讨了 HCI 与制造融合系统在学术发表后如何转化为可用、可扩展产品的挑战，并提出了衡量工程成熟度的六维度框架。

**💡 创新点**

将工程成熟度视为知识生成的经验性工作，提出 buildability、executability、reliability、maintainability、transferability 与 scalability 六维度，并通过五个案例验证其意义。

**🔧 技术方法**

主要采用案例研究与项目反思的方式，未使用具体算法或工具，而是围绕系统设计、实验与转化过程构建评估框架。

**📊 数据集**

不涉及传统的数据集，而是基于五个实例项目（JigFab、StoryStick++、Silicone Devices、PaperPulse、LamiFold）进行分析。

**📈 对比分析**

通过对比这些项目在商业化、技术转移、教育推广等不同情境下的实际表现与论文中所述实现细节，评估框架的适用性；未给出量化性能指标。

**⚠️ 局限性**

缺乏大规模实验验证与量化评估，框架的可操作性与普适性待进一步验证；对具体实现细节的关注不足导致转化障碍。

---

## 8. Natural gradient descent with momentum

**arXiv ID:** 2604.15554 | [PDF](https://arxiv.org/pdf/2604.15554v1)

**作者:** Anthony Nouy `[一作]` (Centrale Nantes), Agustín Somacal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在非线性模型（如可微神经网络）上通过自然梯度下降(NGD)的函数空间视角，推导并实现了基于Heavy‑Ball和Nesterov的天然动量算法，并在回归、分类以及物理信息学习(PDE求解)任务中进行实验验证。

**💡 创新点**

创新点在于将自然梯度从参数空间迁移到函数空间，系统性地构造了天然Heavy‑Ball和天然Nesterov动态，并给出了多种近似实现（QNHB、NHB‑FD、NN‑II 等），从理论与计算成本两方面兼顾；实验表明这些方法在收敛速度和计算时间上优于传统 NGD。

**🔧 技术方法**

使用了函数空间梯度流、Gram 矩阵及交叉 Gram 矩阵、伪逆与谱正则化、自动微分、重投影与重排等技术；算法在参数空间通过求解最小二乘或伪逆得到动量更新，随后在函数空间进行 Retraction。

**📊 数据集**

实验数据集包括：Mackey‑Glass 预测（回归）、扩展 XOR 分类、线性对流扩散 PDE、非线性反应扩散 PDE；每个任务使用随机采样或等距采样的训练/测试点。

**📈 对比分析**

与基线 NGD、GN‑NGD、L‑BFGS 以及带常规动量的 GD 进行对比；天然动量方法在迭代次数上约减少 50%，计算时间亦随之缩短，尤其在 PDE 任务中效果最为显著；部分 Nesterov 变体需调节 β_k 以避免发散。

**⚠️ 局限性**

局限性包括：对小批量随机梯度的鲁棒性尚未系统评估；学习率与动量常数的选择仍以经验为主；在高维/复杂模型下可能出现数值不稳定；未给出完整的理论收敛分析。

---

## 9. Zoom Consistency: A Free Confidence Signal in Multi-Step Visual Grounding Pipelines

**arXiv ID:** 2604.15376 | [PDF](https://arxiv.org/pdf/2604.15376v1)

**作者:** Keon Kim `[一作]` (Om Labs), Krish Chelikavada `[通讯]` (Om Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在多步 GUI 定位（zoom‑in）管道中，利用第2步预测与 crop 中心的距离（即 zoom consistency）作为无成本的置信度信号，进而实现跨模型（专用与通用 VLM）路由。

**💡 创新点**

提出并形式化了“zoom consistency”，证明其在理想条件下是第1步空间误差的线性估计，并指出它可直接跨模型比较而无需校准；基于此构建了一种无需训练、无额外前向传播的路由器。

**🔧 技术方法**

使用 2 步 zoom‑in 视觉语言模型 pipeline，定义并计算距离指标 c；通过 Spearman ρ、AUC 等统计量评估与预测正确性的相关性；实现一个比较 c 的路由器来选择最佳模型。

**📊 数据集**

在 ScreenSpot‑Pro 数据集（1,581 条 GUI grounding 任务，涵盖三大操作系统和多种应用类别）上进行实验。

**📈 对比分析**

与 KV‑Ground‑8B（专用模型）和 Qwen3.5‑27B（通用模型）对比，AUC 约为 0.60，Spearman ρ ≈ -0.13，证明了该信号的判别力；路由器在 ScreenSpot‑Pro 上提升 0.8%（占 oracle 空间的 16.5%），但 McNemar p 值为 0.19，表明提升不具统计显著性。

**⚠️ 局限性**

限制包括：仅在单一基准和模型对上验证；路由增加了四次前向传播，成本提升；信号噪声中等，仅在理想假设下为线性估计；对目标离 crop 边缘或 step‑2 误差敏感；跨基准/跨模型推广尚待验证。

---

## 10. Privacy, Prediction, and Allocation

**arXiv ID:** 2604.15596 | [PDF](https://arxiv.org/pdf/2604.15596v1)

**作者:** Ben Jacobsen `[一作]` (University of Wisconsin), Nitin Kohli `[通讯]` (UC Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文系统研究了在差分隐私约束下，个体层级分配（ILA）与单位层级分配（ULA）在资源分配中的表现与权衡。

**💡 创新点**

创新点在于：①首次将差分隐私（Joint DP）与传统的援助分配模型结合，提出可满足隐私且近似最优的算法；②从多种数据可得性场景（全信息、有限样本、特征学习）出发，给出统一的 regret 上界并揭示关键参数（预算占比 λ、单位不平等 G 等）下的相位转移；③证明在大多数情形下，隐私成本对最优策略的影响可忽略不计，且在一定条件下 ULA 能优于 ILA。

**🔧 技术方法**

主要技术包括：zCDP 与 Joint DP 机制、Gaussian 机密噪声、私有估计（CDF、单位统计）、DP 样本复杂度分析、机器学习模型的 DP 训练（Feldman 等算法）以及对 regret 的分解与优化。

**📊 数据集**

论文为理论性工作，未使用具体真实数据集，而是在通用假设（福利分布、单位划分、特征空间）下给出通用上界和参数化分析。

**📈 对比分析**

通过对不同策略（ILA、ULA、随机分配 RAND）在各种预算/成本比例下的 regret 进行理论比较，结果显示：在预算小、数据昂贵、福利平均低或不平等度高时 ULA 通常优于 ILA；当样本成本 λ 较低且不平等度不高时 ILA 甚至可匹敌或优于 ULA；在极端 λ≥1 时 ULA 一定优于 ILA。

**⚠️ 局限性**

局限性包括：①福利与处理效应模型过于简化，仅考虑 utilitarian（可加）目标；②未考虑非线性或交互的处理效应；③未涵盖自适应或策略性行为；④隐私模型仅为 Joint DP，未探讨更强或更弱的隐私约束；⑤缺乏实证验证与真实数据评估。

---

## 11. Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs

**arXiv ID:** 2604.15379 | [PDF](https://arxiv.org/pdf/2604.15379v1)

**作者:** Sangeeta Chowdhary `[一作]` (AMD Research), Ganesh Dasika `[通讯]` (AMD Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多级任务模型和持久化内核运行时，能够将计算与芯片片层（XCD）的 L2 缓存局部性相匹配，从而在多片段 GPU 上高效执行大语言模型推理。

**💡 创新点**

创新点在于引入 Chiplet‑task 抽象，使工作可以在单个芯片片的 L2 上协同执行；并实现两级层次的事件同步，将大多数一致性开销限制在局部 L2，避免跨芯片的昂贵同步。

**🔧 技术方法**

采用持久化 mega‑kernel、软件调度器、窗口式 M‑major 共享权重切片、cache‑streaming 与非临时加载指令、GPU 级事件计数器和层次化同步协议。

**📊 数据集**

在 AMD Instinct MI350X 上使用 Qwen3‑8B（bf16）密集模型进行推理实验。

**📈 对比分析**

与 vLLM（标准 per‑operator 调度）和内部 Mirage MPK（无芯片级调度）对比；在 bs=1–16 时提升 1.3–1.5× 的单词解码延迟，L2 命中率提升至 51%（从 38.9%），HBM 读取量下降 18%。

**⚠️ 局限性**

局限性包括仅在单 GPU、单模型、密集版上验证；缺乏多 GPU 或 tensor‑parallel 场景评估；对寄存器占用与占用率的折衷未进一步优化。

---

## 12. Applied Explainability for Large Language Models: A Comparative Study

**arXiv ID:** 2604.15371 | [PDF](https://arxiv.org/pdf/2604.15371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 13. Self-Supervised Angular Deblurring in Photoacoustic Reconstruction via Noisier2Inverse

**arXiv ID:** 2604.15681 | [PDF](https://arxiv.org/pdf/2604.15681v1)

**作者:** Markus Haltmeier `[一作]` (University of Innsbruck), Gyeongha Hwang `[通讯]` (Yeungnam University)

**通讯引用:** 423 | [OpenAlex ID](https://openalex.org/A5070476604)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种自监督的光声层析图像重建方法，针对有限尺寸探测器导致的角向模糊问题，在极坐标域中构建角向去卷积模型并利用Noisier2Inverse进行训练；

**💡 创新点**

创新点在于将角向卷积转化为极坐标下的离散去卷积问题，并结合统计学早停策略，实现不需要地面真值的高质量重建；

**🔧 技术方法**

主要技术包括光声成像的极坐标变换、离散角向卷积、Noisier2Inverse自监督网络、以及基于Earth‑Mover距离的自动早停判定；

**📊 数据集**

实验数据采用FIVES血管分割数据集，图像尺寸256×256，随后在极坐标域进行训练；

**📈 对比分析**

与监督学习、SSLTV及DIP等无监督基准相比，该方法在PSNR上获得最高或次高分，性能逼近监督方法；

**⚠️ 局限性**

局限性包括仅在模拟圆形几何与有限探测器幅度下验证，未考虑声波衰减、采样稀疏或实际测量噪声等实际因素。

---

## 14. Eco-Bee: A Personalised Multi-Modal Agent for Advancing Student Climate Awareness and Sustainable Behaviour in Campus Ecosystems

**arXiv ID:** 2604.15327 | [PDF](https://arxiv.org/pdf/2604.15327v1)

**作者:** Caleb Adu `[一作]` (University of Hull), Sruthi Viswanathan `[通讯]` (University of Oxford)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5080735761)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并试点了Eco‑Bee平台，结合星球边界框架为大学生提供多边界的可持续性反馈、个性化建议和游戏化激励，以促进校园级别的环境行为改变。

**💡 创新点**

创新点在于将多模态大语言模型（Pixtral‑12B）与星球边界指标融合，形成Eco‑Score评价体系；通过图嵌入推荐算法实现针对个人低效边界的精准建议；并在架构中嵌入隐私保护与可扩展模块化设计。

**🔧 技术方法**

使用技术包括Next.js前端、Python FastAPI后端、Supabase（PostgreSQL + RLS）数据存储、Pixtral‑12B视觉与语言推理、node2vec图嵌入推荐、聊天机器人交互与轻量级排行榜。

**📊 数据集**

主要使用的“数据集”为52名学生的问卷与图片/条码输入，包含行为数据、可持续性标签以及自定义的行为‑边界映射表；没有引用公开大规模公开数据集。

**📈 对比分析**

通过用户调研和问卷评估进行比较：在26名完成问卷的学生中，交互清晰度4.0/5、易用性4.4/5、建议有用性4.1/5；96%支持校园推广，平均Eco‑Score分布在50.9±7.7；与传统仅以碳排放为中心的工具相比，Eco‑Bee在多维反馈与社交激励方面得到更高用户认可。

**⚠️ 局限性**

局限性包括样本量有限（52人），缺乏纵向跟踪验证长期行为改变；移动端体验尚待优化；推荐逻辑仍需针对具体校园资源细化；隐私设计虽已考虑，但在大规模推广前仍需进一步评估。

---

## 15. CIG: Measuring Conversational Information Gain in Deliberative Dialogues with Semantic Memory Dynamics

**arXiv ID:** 2604.15647 | [PDF](https://arxiv.org/pdf/2604.15647v1)

**作者:** Ming-Bin Chen `[一作]` (University of Melbourne), Lea Frermann `[通讯]` (University of Melbourne)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5025156794)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Conversational Information Gain (CIG) 框架，评估对话中每个发言在主题上推进集体理解的程度；

**💡 创新点**

将 CIG 拆分为 Novelty、Relevance、Implication Scope 三个可解释维度，并通过轻量级语义记忆（基于 LLM 提取和 NLI 合并）实时捕捉对话的知识状态；

**🔧 技术方法**

使用 LLM（GPT‑5、Gemini‑2.5‑pro）进行语义记忆的声明提取与归并，采用向量检索 + NLI 判断关系来执行 ADD/UPDATE/NONE 操作；

**📊 数据集**

在两种英语公共讨论语料上进行实验：INSQ（Intelligence Squared 辩论）和 FORA（社区讨论），共标注 80 段对话；

**📈 对比分析**

与传统启发式（词长、TF–IDF、词元熵等）和 GPT‑5 全文本上下文模型进行对比，发现基于记忆动态的特征（如记忆更新计数）与人类评分的相关性最高（r≈0.72），GPT‑5 在受限上下文下的 MAE 与人类 leave‑one‑out 误差相当，说明自动化评估可实现；

**⚠️ 局限性**

局限性包括：标注受限于预先摘要，可能低估完整上下文影响；语义记忆依赖 LLM 提取与 NLI，易受错误传播；每条声明只对应单一记忆动作，未捕捉多重关系；仅在两种中性主题的英语语料上验证，跨语言、极端对话环境需进一步测试。

---

## 16. DALM: A Domain-Algebraic Language Model via Three-Phase Structured Generation

**arXiv ID:** 2604.15593 | [PDF](https://arxiv.org/pdf/2604.15593v1)

**作者:** Chao Li `[一作]` `[通讯]` (Deepleap.ai), Chao Li (Deepleap.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了域代数语言模型(DALM)，通过在域格子上按层次进行结构化去噪（先域、后关系、再概念），在生成时强制域隔离，实现多视角、可审计的知识生成。

**💡 创新点**

创新点在于：1）用域格子和关系类型函数 τ 对去噪路径进行代数约束；2）将知识压缩为域索引的结构化“晶体”而非无结构权重；3）在闭词表模式下实现零跨域泄漏；4）支持多视角输出和分布式部署。

**🔧 技术方法**

技术包括：三阶段编码-解码架构、域格子运算（meet、join、implication）、τ-typing、纤维分割、离散去噪（mask化）与超平面/超球面嵌入、交叉阶段注意残差、插入时验证门、闭/开放词表机制。

**📊 数据集**

使用 CDC 框架下的结构化知识库（例如 ICD‑11 呼吸系统子集，共 1,247 实体、约 5,000 条晶体）作为训练和评估数据。

**📈 对比分析**

与 GPT‑4/Claude（结构化抽取提示）、无域结构的 seq2seq、检索基线进行对比。实验设计关注域泄漏率（理想为 0）、晶体验证通过率、概念与关系准确率、以及多视角输出特性。已知结果显示：闭词表下跨域泄漏为 0；相较基线，抽取一致性提升；多视角输出在医学领域表现明显。

**⚠️ 局限性**

局限包括：1）需要预先构建晶体库，依赖 LLM 进行初始提取；2）域覆盖受训练晶体库限制，新域需扩展或插值；3）自然语言流畅性需单独的语言化层；4）开放词表生成仍是暂定状态，可能产生未验证概念；5）对大型分布式训练与跨域迁移的可扩展性尚待验证。

---

## 17. OverCite: Add citations in LaTeX without leaving the editor

**arXiv ID:** 2604.15366 | [PDF](https://arxiv.org/pdf/2604.15366v1)

**作者:** Cheyanne Shariat `[一作]` (California Institute of Technology), Cheyanne Shariat `[通讯]` (California Institute of Technology)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5092046708)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在撰写论文过程中添加引用的常见困难，并提出了一种简化引用添加流程的方法。

**💡 创新点**

创新点在于将引用搜索、检索和粘贴的步骤整合到一个统一的编辑环境中，减少了切换编辑器的操作。

**🔧 技术方法**

主要使用了文本编辑器插件或脚本技术，结合在线文献数据库的 API 进行实时检索。

**📊 数据集**

未给出具体的数据集，推测使用了常见的学术数据库（如 arXiv、PubMed）中的公开论文元数据。

**📈 对比分析**

与传统手工搜索和复制粘贴相比，该方法在速度和准确性上有显著提升，但缺乏详细实验结果。

**⚠️ 局限性**

局限性包括对不同编辑器的兼容性有限、依赖外部数据库的可用性以及未充分验证在大规模文档中的性能。

---

## 18. Restoration, Exploration and Transformation: How Youth Engage Character.AI Chatbots for Feels, Fun and Finding themselves

**arXiv ID:** 2604.15340 | [PDF](https://arxiv.org/pdf/2604.15340v1)

**作者:** Annabel Blake `[一作]` (University of Sydney), Eduardo Velloso `[通讯]` (University of Sydney)

**通讯引用:** 4343 | [OpenAlex ID](https://openalex.org/A5087672786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Character.AI 官方 Discord 社区的 4,172 名活跃青少年用户进行自然语言数据的定性分析，描述他们的使用行为，并提出了以恢复（Restoration）、探索（Exploration）与转化（Transformation）三大意图为核心的参与框架以及七类青年创造的 AI 角色原型。

**💡 创新点**

创新点在于：①首次用青年视角系统梳理 AI 娱乐平台的真实使用场景；②构建了面向青年需求的 R/E/T 三意图模型；③提出了七种角色原型，为研究者与设计师提供细粒度的评估与设计指导，弥补了以成人为中心的研究空白。

**🔧 技术方法**

采用混合方法：对 Discord 文本进行编码、主题映射与亲和图分析，并结合使用者自述的年龄、性别、心理健康信息等元数据；未开发新算法，而是对现有大语言模型交互记录进行文本分析。

**📊 数据集**

数据集为 Character.AI 官方 Discord 公开服务器的 4,172 条用户自我介绍帖子（含年龄、性别、兴趣等字段）以及 148 条相关主题讨论帖，全部为公开可获得的自然对话数据。

**📈 对比分析**

该研究不涉及算法性能对比，主要以定性发现为主；未给出量化指标，而是通过主题分类、原型归纳和意图框架阐释青年行为的多样性与复杂性。

**⚠️ 局限性**

局限性包括：①样本仅来自活跃、公开讨论的 Discord 社区，可能偏向表达欲强的少数人；②年龄与身份信息为自报，验证难度高；③研究聚焦单一平台，结果可能不具备跨平台推广性；④缺乏对实际交互效果的实验验证，无法评估原型或框架在真实产品中的可操作性。

---

## 19. GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology

**arXiv ID:** 2604.15495 | [PDF](https://arxiv.org/pdf/2604.15495v1)

**作者:** Shivendra Agrawal `[一作]` (University of Colorado Boulder), Bradley Hayes `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1156 | [OpenAlex ID](https://openalex.org/A5034950112)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 GIST 系统，从消费者级移动 LiDAR/RGB‑D 数据提取稠密环境的语义拓扑，实现多模态知识抽取与下游搜索、定位和导航指令生成。

**💡 创新点**

首次引入共享的语义拓扑中间表示，将几何结构与语义推理分离，并通过智能关键帧/对象选择、VLM 标注以及基于拓扑的路由生成实现统一的全景理解。

**🔧 技术方法**

结合 DINOv3 关键帧筛选、YOLOv9 SKU 检测、Vision‑Language 模型（Gemini）语义标注、2D 占用网格、骨架化拓扑图、基于文本嵌入的单次语义定位以及 A* 路径规划。

**📊 数据集**

使用 3,500 平方英尺国际杂货店的 52,006 帧 LiDAR/RGB‑D 采样，配合 SKU‑110K 细粒度检测数据集及 COCO 预训练模型。

**📈 对比分析**

通过 15 个真实场景的多指标 LLM 评估与 5 参与者的生态探测，对比 NavComposer、Naive Gemini、GIST Text/Visual，GIST Visual 在四个指标上均领先，平均 4.43/5，导航成功率 80%。

**⚠️ 局限性**

受限于移动扫描视野缺口、语义重叠导致的定位歧义以及仅在小规模用户测试，未来需改进扫描协议、融合时间序列传感器并扩大实验规模。

---

## 20. Designing More Engaging Serious Games to Support Students' Mental Health: A Pilot Study Based on A CBT-Informed Design Framework

**arXiv ID:** 2604.15662 | [PDF](https://arxiv.org/pdf/2604.15662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 21. Predicting Where Steering Vectors Succeed

**arXiv ID:** 2604.15557 | [PDF](https://arxiv.org/pdf/2604.15557v1)

**作者:** Jayadev Billa `[一作]` `[通讯]`, Jayadev Billa

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在Transformer各层使用logit lens无训练地评估概念的线性可访问性，并将其作为预测Steering向量有效性的诊断工具

**💡 创新点**

提出Linear Accessibility Profile（LAP）三要素框架，能在不训练的情况下预判哪些概念在何层可被Steering，并揭示三种“可访问性”规律

**🔧 技术方法**

利用模型自身的unembedding矩阵、残差MLP补偿、扰动灵敏度评估等无训练方法，构建LAP并验证其与Steering效果的相关性

**📊 数据集**

在Gemma-2-2B、Llama-3.1-8B、Mistral-7B、Qwen2.5-7B、Pythia系列等公开大型语言模型上，采用24个受控二元概念族和5个核心概念族进行实验

**📈 对比分析**

在所有模型中，LAP峰值（peak A_lin）与最大Steering增益（ΔP）相关系数均超过0.86（最大0.91），比传统中间层Heuristic显著提高了Steering成功率与层选择准确率（0.63–0.92）

**⚠️ 局限性**

仅适用于单词级next-token任务，无法直接处理多-token或分布式输出；当概念已被模型内部编码但与输出投影不对齐时，LAP仅给出警示，仍需更复杂的非线性Steering方法

---

## 22. LLMs Corrupt Your Documents When You Delegate

**arXiv ID:** 2604.15597 | [PDF](https://arxiv.org/pdf/2604.15597v1)

**作者:** Philippe Laban `[一作]`, Jennifer Neville `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过模拟长时间的委托工作流程，评估了52个专业领域中19种大型语言模型在文档编辑中的可靠性；

**💡 创新点**

创新点在于提出了基于可逆编辑任务的回译链式评估框架，能够在不需要人工标注的情况下，跨多个领域自动测量文档完整性；

**🔧 技术方法**

技术包括回译（backtranslation）循环、域特定解析器、可逆编辑任务设计以及基于语义相似度的无参考评估；

**📊 数据集**

使用的数 据集为52个专业领域的真实文档，覆盖编程、晶体学、会计、音乐谱写等，共约3–5k tokens的种子文档加8–12k tokens的干扰上下文；

**📈 对比分析**

评估方法通过记录每20次交互后的重建得分（RS@k），结果显示最强模型在20次交互后平均只保留75%文档内容，整体平均降幅约50%，在Python领域表现最佳；

**⚠️ 局限性**

局限性包括：仅使用单轮指令，未模拟多轮澄清交互；仅评估可逆编辑任务，排除了非结构化知识工作；实验规模受上下文窗口和成本限制，可能低估真实场景中的错误累积。

---

## 23. PolicyBank: Evolving Policy Understanding for LLM Agents

**arXiv ID:** 2604.15505 | [PDF](https://arxiv.org/pdf/2604.15505v1)

**作者:** Jihye Choi `[一作]` (Google Cloud), Tomas Pfister `[通讯]` (Google Cloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种通过交互和反馈自动完善LLM代理对自然语言政策理解的机制，并在扩展的τ‑Bench基准上验证其效果。

**💡 创新点**

创新点在于设计了按工具级能力维度存储结构化政策洞察的 PolicyBank，并通过专用的 Policy Agent 根据开发者反馈迭代更新，从而解决传统内存机制只能强化已知行为的缺陷。

**🔧 技术方法**

技术包括LLM工具调用、对话式推理、半结构化可执行授权逻辑、动态检索工具、基于反馈的学习循环以及对齐失败与执行失败的分类。

**📊 数据集**

使用了扩展的 τ‑Bench（航空与零售域）任务，其中注入了带有明确政策缺口的父任务和对应的姐妹任务，并人工制定了政策澄清作为黄金标准。

**📈 对比分析**

通过与无记忆 baseline、Synapse、AWM、ReasoningBank 等现有内存机制在原始任务和姐妹任务上进行 k‑指标对比，PolicyBank 在姐妹任务上实现 0.67–0.82 的 k 绩效，显著优于其他基线，尤其在保持一致性方面表现突出。

**⚠️ 局限性**

局限性包括对高质量开发者反馈和明确 NL 说明的依赖；对抗性或噪声反馈的鲁棒性尚待提升；对更大规模、更复杂政策文档的适用性需进一步验证；与正式验证层结合以提供可证明合规仍是未来工作。

---

## 24. A Framework for Post Quantum Migration in IoT-Based Healthcare Systems

**arXiv ID:** 2604.15584 | [PDF](https://arxiv.org/pdf/2604.15584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 25. Evaluating LLM Simulators as Differentially Private Data Generators

**arXiv ID:** 2604.15461 | [PDF](https://arxiv.org/pdf/2604.15461v1)

**作者:** Nassima M. Bouzid `[一作]` (Capital One), Mayana Pereira `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过差分隐私生成用户画像后，将其作为输入送入LLM驱动的PersonaLedger模拟器，生成合成交易序列，并与直接DP合成的交易数据进行对比。

**💡 创新点**

提出“Profile-then-Simulate”分离框架，利用LLM在高维用户表示下生成合成数据，同时系统评估LLM的偏差和分布漂移。

**🔧 技术方法**

采用AIM差分隐私机制生成用户画像，利用PersonaLedger LLM模拟器生成交易，使用TSTR评估、AUC与TVD度量进行性能和忠实度分析。

**📊 数据集**

使用Kaggle信用卡交易数据（约2000名合成用户，2000万笔交易）作为真实数据基础。

**📈 对比分析**

采用TSTR方法在自然欺诈率下评估XGBoost分类器的AUC；PersonaLedger在ε=1时达到0.70的AUC，而直接DP合成实现0.87–0.89；但PersonaLedger的TVD显著更高（0.30–0.34），表明分布漂移。

**⚠️ 局限性**

LLM的先验偏差导致时间和人口统计特征的分布漂移，非单调隐私预算效应以及未能完全遵循输入统计是主要局限。

---

## 26. FD-NL2SQL: Feedback-Driven Clinical NL2SQL that Improves with Use

**arXiv ID:** 2604.15646 | [PDF](https://arxiv.org/pdf/2604.15646v1)

**作者:** Suparno Roy Chowdhury `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个交互式、反馈驱动的临床自然语言到SQL助手，专门针对肿瘤临床试验数据库；

**💡 创新点**

通过LLM进行方案级谓词拆分、检索专家验证的示例并在此基础上合成SQL，随后利用用户反馈自动扩展示例库，实现持续自我改进；

**🔧 技术方法**

结合schema-aware LLM推理、句子嵌入检索（SBERT）、SQL生成与后处理、单步SQL变异、NL↔SQL反向翻译等技术；

**📊 数据集**

使用Mayo Clinic肿瘤试验数据库（SQLite版）以及由专家编写的500个种子问题和约1500个经过增量生成的验证样本；

**📈 对比分析**

与零样本、少量样本、链式思考以及强化学习微调等基线模型对比，实验表明在eEM、eF1、AST等执行和结构性指标上均优于基线，且在不同模型上均能显著提升性能；

**⚠️ 局限性**

样本来源受限于单步编辑变体，可能不覆盖更复杂的查询结构；示例扩增只保留执行成功且返回结果的变体，可能偏向频繁模式；缺乏真实临床用户评估和安全/隐私评估。

---

## 27. Safe and Energy-Aware Multi-Robot Density Control via PDE-Constrained Optimization for Long-Duration Autonomy

**arXiv ID:** 2604.15524 | [PDF](https://arxiv.org/pdf/2604.15524v1)

**作者:** Longchen Niu `[一作]` (University of Waterloo), Gennaro Notomista `[通讯]` (University of Waterloo)

**通讯引用:** 1086 | [OpenAlex ID](https://openalex.org/A5072586461)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于 Fokker‑Planck PDE 的密度控制框架，能够在多机器人系统中同时保证空间安全、能源可持续性和对目标分布的跟踪。

**💡 创新点**

创新点包括：①将控制鲁棒性（CLF‑CBF）与 PDE 密度动力学耦合，实现对分布级别的安全与能量约束；②在 PDE 约束下采用二次规划实现实时闭环控制；③设计了基于安全约束的动力学 RRT 路径规划，用以精确估计能量到充电站的消耗。

**🔧 技术方法**

主要技术：Fokker‑Planck PDE、控制 Lyapunov 函数 (CLF)、控制 Barrier 函数 (CBF)、二次规划 (QP)、基于稀疏矩阵的 FDM 离散、kinodynamic RRT。

**📊 数据集**

实验使用 DJI RoboMaster EP 四机队，在 4 m×4 m 室内场景中，位置通过 Vicon 系统测量；没有使用公开数据集，全部为仿真与现场实验数据。

**📈 对比分析**

与传统基于 ODE 的安全控制或能量约束方法相比，本文方法在 0.065 s 的控制循环内实现实时控制，实验和 100 次仿真均满足安全与能量约束，最终目标分布收敛良好；性能指标（Lyapunov 值、CBF 边界、能量轨迹）均维持在安全阈值之上。

**⚠️ 局限性**

局限性：①概率性 RRT 规划偶尔产生较长路径，导致能量略微低于阈值；②在极少数极端噪声情况下出现一次性能量违反；③规划时间受限，无法在更复杂或大规模场景中快速收敛。

---

## 28. The Crutch or the Ceiling? How Different Generations of LLMs Shape EFL Student Writings

**arXiv ID:** 2604.15460 | [PDF](https://arxiv.org/pdf/2604.15460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 29. A Q-learning-based QoS-aware multipath routing protocol in IoMT-based wireless body area network

**arXiv ID:** 2604.15489 | [PDF](https://arxiv.org/pdf/2604.15489v1)

**作者:** Mehdi Hosseinzadeh `[一作]` (Duy Tan University), Sadia Din `[通讯]` (Gachon University)

**通讯引用:** 3289 | [OpenAlex ID](https://openalex.org/A5089091108)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于Q学习的QoS感知多路径路由方法QQMR，针对IoMT场景下的WBAN网络实现分类型数据的优先级调度、动态队列容量分配、聚类状态空间和专属学习策略。

**💡 创新点**

创新点包括：①将QoS需求分为三类并为每类维护独立的Q表；②采用自适应加权模糊C均值聚类缩小状态空间；③动态多队列模型与备份路径选择；④结合特征权重与学习系数实现更精确的聚类。

**🔧 技术方法**

技术手段包括：强化学习Q‑learning、加权模糊C‑means聚类、ε‑greedy探索、动态队列调度、备份路径匹配分数、仿真网络协议IEEE 802.11。

**📊 数据集**

使用NS3仿真平台，构建500×500m²无线场景，200–1000个WBAN节点，CBR流量，模拟不同节点密度与包发送速率的实验。

**📈 对比分析**

与QQAR、EQRSRL、QPRR三种基线方法对比，结果表明QQMR在不同密度和负载下均提高PDR（约2.5–5.5%），显著降低EED（30–70%）、RO（11–30%）、HC（27–58%）和能耗（19–39%）。

**⚠️ 局限性**

局限性：仅在仿真环境验证；未考虑节点移动、链路干扰及真实医疗设备的功耗；模型对极端拥塞或大规模网络的鲁棒性待进一步评估。

---

## 30. CSLE: A Reinforcement Learning Platform for Autonomous Security Management

**arXiv ID:** 2604.15590 | [PDF](https://arxiv.org/pdf/2604.15590v1)

**作者:** Kim Hammar `[一作]` (KTH Royal Institute of Technology), Kim Hammar `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5074576259)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了面向自动化安全学习的仿真平台CSLE，支持数字孪生、攻击者、受害者与防御者交互，并集成强化学习策略；

**💡 创新点**

创新点在于将强化学习与数字孪生结合，提供可复现的实验环境、系统监控与自适应控制，且实现分布式部署和自动化实验管理；

**🔧 技术方法**

使用Docker容器、Docker Compose、Kubernetes、OpenAI Gym规范、Python、TensorFlow/Stable Baselines等RL框架；

**📊 数据集**

使用基于真实网络拓扑的模拟数据集（如31/64容器实验配置）和MITRE ATT&CK/Defend数据库的攻击与防御动作；

**📈 对比分析**

通过与传统规则/基线防御方法对比，展示了RL策略在安全性、资源占用和监控事件率上的优势，CPU监控占用约6%，内存1%，可扩展到多节点；

**⚠️ 局限性**

局限性在于平台仅覆盖有限的攻击与防御动作，缺乏对复杂持续性攻击的模拟，且RL训练仍需大量交互时间；

---

## 31. A shifted interface approach for internal discontinuities in poroelastic media

**arXiv ID:** 2604.15450 | [PDF](https://arxiv.org/pdf/2604.15450v1)

**作者:** David Michael Riley `[一作]` (Institut Polytechnique de Paris), Ioannis Stefanou `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5101528241)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种适用于多孔弹性介质中内部不连续性的移位界面方法，旨在解决裂缝、断层等内部不连续性对水力和机械响应的影响。

**💡 创新点**

创新点在于将移位界面方法扩展到耦合瞬态多孔弹性问题，并系统比较了弱施加和强施加两种界面条件的策略。

**🔧 技术方法**

使用了移位界面方法（Shifted Interface Method, SIM）和有限元方法（Finite Element Method, FEM）来处理多孔介质中的裂缝问题。

**📊 数据集**

使用了多个测试案例，包括偏移网格对齐裂缝、边界交叉角裂缝、嵌入角裂缝和多裂缝配置，验证了所提出方法的有效性。

**📈 对比分析**

通过与传统的网格适应方法和未适应方法进行比较，结果表明，移位界面方法在处理复杂几何形状的嵌入界面时具有更好的性能，尤其是在收敛性方面。

**⚠️ 局限性**

限制在于在裂缝尖端附近可能出现局部后处理伪影，这可能会影响全局收敛率，但在排除小区域后可以恢复一阶收敛性。

---

## 32. NeuroMesh: A Unified Neural Inference Framework for Decentralized Multi-Robot Collaboration

**arXiv ID:** 2604.15475 | [PDF](https://arxiv.org/pdf/2604.15475v1)

**作者:** Yang Zhou `[一作]` (New York University), Giuseppe Loianno `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 NeuroMesh，一种多域、跨平台、模块化的去中心化神经推理框架，支持在异构机器人团队中执行感知、控制和任务分配等多种学习驱动的协作任务。

**💡 创新点**

核心创新包括：① 双聚合（Reduction+Broadcast）范式，兼容稠密感知与稀疏通信的协作；② 并行三阶段流水线，将编码、消息传递聚合、解码独立并发，解耦周期时间与总延迟；③ 统一接口与微服务架构，可在 CPU、GPU 上自由切换推理引擎；④ 与 Zenoh 的无缝集成，兼容 Mesh 与 Wi‑Fi 网络。

**🔧 技术方法**

技术栈涵盖：ROS 2 微服务、Zenoh 通信协议、TensorRT/ONNX 推理引擎、GNN、Vision Transformer、MLP、Attention 等深度网络；同时使用了 Doodlelabs Mesh Rider 无线网、NVIDIA Jetson Orin NX、Intel i7/GeForce 1050 Ti 计算节点。

**📊 数据集**

数据来源为自建的室内外 RGB 图像与深度标注数据（用于深度估计和语义分割），仿真环境（用于 RL 控制）以及真实机器人实验中的状态/目标成本向量（用于任务分配）。

**📈 对比分析**

通过与集中式推理、传统单机器人基准以及基于 Hungarian 算法的专家模型对比，实验表明：• 感知任务在去中心化与集中式下相同的绝对相对误差 5.29% 与 inlier 率 61.32%；• 控制任务实时成功率从仿真 80% 降至现实 60%；• 任务分配成功率 93.92%（消息 128 B）且平均成本仅比专家高 0.03%；• 通信上 Zenoh peer 以 4.8 ms 延迟、200 Hz 频率完成 128 B 控制；对 3.15 MB 感知仅达 0.5 Hz，但可实现。

**⚠️ 局限性**

局限性包括：① 仿真到现实迁移仍面临速度/角度控制误差；② 大规模实验（>10 机器人）仅在仿真中验证；③ 受 Mesh 无线链路带宽与距离限制；④ 仅测试了 RGB 及有限的稠密感知模型，未覆盖多模态传感器；⑤ 对极端高功耗任务的能耗评估缺失。

---

## 33. Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch

**arXiv ID:** 2604.15490 | [PDF](https://arxiv.org/pdf/2604.15490v1)

**作者:** Eleanor M. Lin `[一作]` (University of Michigan), David Jurgens `[通讯]` (University of Michigan)

**通讯引用:** 5197 | [OpenAlex ID](https://openalex.org/A5046126345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了跨模型、跨语言、跨任务的推理轨迹语料库CoRe，并提出了基于语言学与行为学的微调框架，帮助大型语言模型在推理任务中更有效地进行代码切换。

**💡 创新点**

创新点在于融合上层理论与下层数据的双重方法创建代码切换行为分类法，利用翻译任务微调激发推理中的有益代码切换，从而提升低资源语言的推理性能。

**🔧 技术方法**

使用了大型语言模型（Qwen、Phi-4、DeepSeek-R1等）的监督微调、机器翻译（NLLB、SeamlessM4T、Apertus）技术、语言识别与统计建模（GLMM）来评估与优化代码切换指标。

**📊 数据集**

使用的数据集包括约7000条推理轨迹的CoRe语料库、Global MMLU、多语言翻译数据集No Language Left Behind以及多语言数学/科学/逻辑/道德推理子集。

**📈 对比分析**

在Global MMLU的七种低中高资源语言上进行标准化微调实验，并通过代码切换指标（CMI、M-Index、Integration Index）与推理准确率的统计比较，发现英语为矩阵语言、提高代码切换准确度和适度的代码切换密度可显著提升推理表现，翻译微调显著提升代码切换度。

**⚠️ 局限性**

限制在于对部分语言（如约鲁巴）缺乏足够数据、微调样本受限、未对生成文本的流畅度和长文本推理的普适性进行评估，以及对代码切换策略因果机制的解释仍需进一步研究。

---

## 34. ShapeGen: Robotic Data Generation for Category-Level Manipulation

**arXiv ID:** 2604.15569 | [PDF](https://arxiv.org/pdf/2604.15569v1)

**作者:** Yirui Wang `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 29164 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过 ShapeGen 框架，基于单一演示数据自动生成大量形状多样化的操作示例，并训练出在同一类别内具备更强泛化能力的操控策略。

**💡 创新点**

创新点包括：①基于几何监督的稠密空间warp网络，实现功能感知的点对点对齐；②构建可即插即用的 Shape Library，支持 O(1) 时间内加入新形状；③仅需一次一分钟的人类注解即可完成整个数据生成过程，显著降低人工成本。

**🔧 技术方法**

使用技术：Signed Distance Function (SDF) 监督、Neural SDF (Instant‑NGP)、空间warp网络、SAM3D/SAM2、FoundationPose、ManiFlow 等。

**📊 数据集**

使用数据集：包含四个日常物品类别（如杯子、壶等）的真实扫描模型、互联网上采集的 3D 资产，以及通过 SAM3D 生成的自动化 3D 模型。

**📈 对比分析**

方法对比：与仅使用源演示、特征匹配以及不做形状变换的基线相比，ShapeGen 在四个任务中平均成功率提升至 0.98，且生成数据的实时性约 15 FPS，显著降低数据收集成本。

**⚠️ 局限性**

局限性：仅适用于刚性对象，无法处理可变形或关节物体；缺少对更复杂、需要深层功能理解任务的验证；对极端形状差异仍可能产生误配。

---

## 35. VeriCWEty: Embedding enabled Line-Level CWE Detection in Verilog

**arXiv ID:** 2604.15375 | [PDF](https://arxiv.org/pdf/2604.15375v1)

**作者:** Prithwish Basu Roy `[一作]` (NYU), Ramesh Karri `[通讯]` (NYU)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于向量嵌入的Verilog硬件漏洞检测框架，能够在模块级和行级同时定位CWEs。

**💡 创新点**

创新点在于利用LLM生成的语义嵌入实现多标签CWE识别，首次仅凭嵌入同时完成模块与行级定位。

**🔧 技术方法**

技术手段包括：Decoder‑only LLM（cl‑verilog‑1.0）生成嵌入、三模型投票进行标签生成、XGBoost分类器以及线/模块嵌入的组合。

**📊 数据集**

使用了BugWhisperer与Verigen混合的4000条Verilog设计数据集，并通过投票方式获得CWE标签。

**📈 对比分析**

通过与基于规则、AST或图的传统方法对比，模块级精度约89%（CWE‑1244/1245），行级检测精度96%，显著提升传统方法的性能。

**⚠️ 局限性**

局限性包括：依赖投票标注质量，受限于少数硬件特定CWE的覆盖，且模型在更大规模多样化数据集上的泛化能力仍需进一步验证。

---

## 36. UA-Net: Uncertainty-Aware Network for TRISO Image Semantic Segmentation

**arXiv ID:** 2604.15542 | [PDF](https://arxiv.org/pdf/2604.15542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 37. Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit

**arXiv ID:** 2604.15356 | [PDF](https://arxiv.org/pdf/2604.15356v1)

**作者:** Gregory Magarshak `[一作]` `[通讯]`, Gregory Magarshak

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种顺序 KV 缓存压缩框架，结合前缀去重和预测增量编码；

**💡 创新点**

通过利用语言模型的预测熵与 PLT trie 结构，实现跨会话前缀去重并证明顺序熵下限低于 TurboQuant 的单向量下限；

**🔧 技术方法**

使用 Probabilistic Prefix Deduplication、Predictive Delta Coding、TurboQuant 量化以及 PLT trie 计算；

**📊 数据集**

基于大规模语言模型（如 70B 参数模型）和常见英语文本进行评估；

**📈 对比分析**

与 TurboQuant 对比，理论上可达数百千倍压缩比，实际实现可实现数百倍以上压缩，且压缩率随上下文长度增长；

**⚠️ 局限性**

理论下限未在真实数据上充分验证；预测 KV 的前向计算成本和前缀索引实现复杂；对非自回归模型的适用性有限。

---

## 38. Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models

**arXiv ID:** 2604.15383 | [PDF](https://arxiv.org/pdf/2604.15383v1)

**作者:** Yanda Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salem Lahlou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5030085635)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的解码方法Temporal Contrastive Decoding (TCD)，在推理时通过构造同一音频的模糊慢路径视图并与原始视图对比下一词logits，利用门控机制仅在必要时对logit进行稀疏更新，从而减轻时间平滑偏差。

**💡 创新点**

创新点在于：① 采用时间多尺度对比（原始 vs 模糊）来捕捉瞬时音频线索；② 使用自归一化稳定性评分动态决定模糊窗口和更新幅度；③ 结合音频依赖度和不确定度的门控，仅在解码器真正关注音频且不确定时才激活修正。

**🔧 技术方法**

技术细节包括：Hann窗时间模糊、重新编码慢路径、对比logit差、候选集约束、稳定性评分、门控更新、logit稀疏修正。

**📊 数据集**

使用公开基准MMAU和AIR-Bench（Speech、Sound、Music等多模态问答）以及AIR-Bench中的时间结构化任务（SLURP、CochlScene、Clotho-AQA）进行评估。

**📈 对比分析**

与基线模型及音频感知解码(AAD)对比，TCD在统一LALM上平均提升约1–4个百分点，尤其在Music和Sound领域显著；在时间敏感任务上提升6–8个百分点，表明门控的对比修正能有效利用瞬时音频线索。

**⚠️ 局限性**

局限性包括：仅适用于解码器能访问时间对齐音频表示的统一LALM，对压缩音频或语义瓶颈的模型效果有限；额外的前向传播导致推理延迟；对GPU缓存和多路解码的依赖，可能限制实时或高吞吐量应用。

---

## 39. Access Over Deception: Fighting Deceptive Patterns through Accessibility

**arXiv ID:** 2604.15338 | [PDF](https://arxiv.org/pdf/2604.15338v1)

**作者:** Tobias Pellkvist `[一作]` (Institute of Science Tokyo), Miu Kojima `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5113204742)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过启发式评估和 WCAG 评测工具，对 68 个真实网页中的暗黑模式（DP）进行检查，识别出哪些 DP 与无障碍标准冲突，进而探讨利用 WCAG 对抗 UI 欺骗的方法。

**💡 创新点**

创新点在于首次将 Web 内容无障碍指南（WCAG）与暗黑模式的可识别性相结合，提出可操作的“无障碍标准”工具链作为对抗 DP 的新手段，并系统性识别出三类 DP（倒计时、自动播放、隐藏信息）在 WCAG 规则下必然不可行。

**🔧 技术方法**

使用的技术包括：启发式评估流程、三款 WCAG 评测工具（WAVE、IBM Accessible、QualWeb）以及人工手工验证；统计分析采用卡方检验与 Fisher 精确检验。

**📊 数据集**

数据集来源于公开暗黑模式集合（如 deceptive.design、hallofshame.design、Reddit 等）以及网络归档，最终得到 68 个 DP 示例，覆盖 26 种高阶 DP 以及 43 种最低阶 DP。

**📈 对比分析**

通过比较 DP 类型与可访问性问题的数量（可访问 vs. 不可访问），统计检验显示无显著差异（p>0.05），但在个别案例中发现特定 DP 必须根本性改动才能满足 WCAG；整体上，工具链能快速定位违反 WCAG 的 DP，并为后续法规落地提供数据支持。

**⚠️ 局限性**

限制包括：仅涵盖网页内容，排除了大部分仅在移动端或屏幕截图中出现的 DP；样本量有限，DP 变体不够多；WCAG 覆盖率约 50%，可能漏判；研究仅检视了 68 个案例，难以推广到所有 DP 类型。

---

## 40. DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling

**arXiv ID:** 2604.15594 | [PDF](https://arxiv.org/pdf/2604.15594v1)

**作者:** Nilavra Pathak `[一作]` (Expedia Group), Nirmalya Roy `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 3748 | [OpenAlex ID](https://openalex.org/A5068320631)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了DataCenterGym物理建模的分布式数据中心调度仿真平台，并设计了层级模型预测控制（H‑MPC）调度算法；

**💡 创新点**

创新点在于将计算排队、建筑热动力学、局部HVAC控制与温度相关的服务降级等物理过程闭环集成于同一Gym环境，构建了可复现的多目标调度实验平台，并通过层级MPC实现分布式可扩展的前瞻性调度；

**🔧 技术方法**

使用物理热力学RC模型、PID冷却控制、模型预测控制（MPC）、强化学习框架Gymnasium、基于Python的数值仿真；

**📊 数据集**

利用阿里巴巴2018年集群作业轨迹（Alibaba cluster trace），按GPU/CPU分配构造多租户作业；

**📈 对比分析**

通过Monte Carlo实验将随机、贪心、热感知、功耗/冷却感知、SC‑MPC与H‑MPC等六种策略在同一环境下进行对比。实验显示H‑MPC在队列长度、能耗、成本上优于其他方法，且在高负载时能保持温度在安全阈值以内；

**⚠️ 局限性**

局限在于使用简化的RC热模型，未考虑多区热分布、网络延迟/带宽、作业依赖和碎片化、学习误差、可再生能源与电网约束等实际因素。

---

## 41. Frequency-Aware Flow Matching for High-Quality Image Generation

**arXiv ID:** 2604.15521 | [PDF](https://arxiv.org/pdf/2604.15521v1)

**作者:** Sucheng Ren `[一作]` (Johns Hopkins University), Liang-Chieh Chen `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出频率感知流匹配模型（FreqFlow），通过低频/高频双分支和时间自适应加权，改进流匹配生成过程。

**💡 创新点**

创新点在于：①将图像分解为低频结构和高频细节，分别建模；②使用时间依赖自适应权重动态融合两种频率信息；③采用双域（空间与频域）监督，提升整体与细节一致性。

**🔧 技术方法**

技术包括：流匹配框架、离散傅里叶变换(DFT/IDFT)与低通/高通滤波、两分支架构（频率分支 + 空间分支）、自适应权重模块(MLP+Sigmoid)、双域损失、ViT与ConvNeXt结合的网络结构，训练于潜在空间。

**📊 数据集**

使用 ImageNet 数据集进行类别条件生成，分别在 64×64、256×256 和 512×512 分辨率上评测。

**📈 对比分析**

与 DiT、SiT、DiMR 等前沿扩散/流匹配模型对比，使用 FID、IS、Precision/Recall 评价；在 ImageNet‑256 上 FID 1.38、IS 298.5，优于 DiT‑XL/2（FID 2.27）和 SiT‑XL/2（FID 2.06）；在 ImageNet‑512 上 FID 2.02，显著优于 DiT‑XL/2（FID 3.04）。

**⚠️ 局限性**

局限性包括：仍以 ImageNet 为测试域，未在更广泛数据集验证；频率分支与自适应加权带来额外计算和参数；对极高分辨率或多模态任务的扩展仍需研究。

---

## 42. Python library supporting Discrete Variational Formulations and training solutions with Collocation-based Robust Variational Physics Informed Neural Networks (DVF-CRVPINN)

**arXiv ID:** 2604.15398 | [PDF](https://arxiv.org/pdf/2604.15398v1)

**作者:** Tomasz Służalec `[一作]` (AGH University of Krakow), Maciej Paszyński `[通讯]` (AGH University of Krakow)

**通讯引用:** 2522 | [OpenAlex ID](https://openalex.org/A5075191779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Python库DVF-CRVPINN，用于基于离散弱形式和协同优化的物理信息神经网络，可训练离散点上的PDE解。

**💡 创新点**

创新点在于将离散弱变分形式与克罗内克δ测试函数结合，构建离散鲁棒损失函数，并通过协同投影和LU分解实现高效训练；同时提供FEniCS式易用接口。

**🔧 技术方法**

采用PyTorch自动微分、NumPy/SciPy线性代数、有限差分离散导数、LU分解、Gram矩阵加权、Adamax优化器等技术。

**📊 数据集**

使用统一网格离散点（如30×30、20×20、40×40、50×50）作为数据集，测试Laplace、制造解Stokes以及旋涡箱Stokes问题。

**📈 对比分析**

通过将鲁棒损失与真实误差的上下界进行对比，展示误差在理论界限内收敛；训练时间为5000轮或25000轮，性能优于传统RVPINN且内存占用降低。

**⚠️ 局限性**

局限性包括：仍需手工组装矩阵，难以扩展到复杂几何或三维；依赖均匀网格；对大规模问题的可扩展性尚未验证；缺乏与其他PINN/FEM方法的直接性能对比。

---

## 43. To LLM, or Not to LLM: How Designers and Developers Navigate LLMs as Tools or Teammates

**arXiv ID:** 2604.15344 | [PDF](https://arxiv.org/pdf/2604.15344v1)

**作者:** Varad Vishwarupe `[一作]` (University of Oxford), Marina Jirotka `[通讯]` (University of Oxford)

**通讯引用:** 4777 | [OpenAlex ID](https://openalex.org/A5023741875)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对设计师和开发者在将大型语言模型（LLM）嵌入工作流程时进行角色定位的决策进行构造主义扎根理论研究，提出工具与队友两种框架，并给出5维分析 rubric。

**💡 创新点**

将“是否使用LLM”视为设计时的社会技术角色定位问题，首次提出工具 vs 队友框架以及对应的决策权、责任归属、监督策略、错误解释和组织可接受度的维度。

**🔧 技术方法**

采用构造主义扎根理论方法，进行33次半结构化访谈，随后15次追访访谈，持续比较、编码、备忘分析。

**📊 数据集**

访谈文本数据，来源于三大技术公司（匿名化），涵盖设计师、UX、软件工程师等角色。

**📈 对比分析**

本研究不做技术性能比较，主要通过访谈分析对照不同角色框架的可接受度与组织治理，未给出数值指标。

**⚠️ 局限性**

样本限于大型企业，可能不适用于初创、公共部门或不同文化背景；仅针对当下LLM技术与治理框架，缺乏纵向跟踪与跨文化验证。

---

## 44. GEN-Graph: Heterogeneous PIM Accelerator for General Computational Patterns in Graph-based Dynamic Programming

**arXiv ID:** 2604.15361 | [PDF](https://arxiv.org/pdf/2604.15361v1)

**作者:** Yanru Chen `[一作]` (University of California San Diego), Tajana Rosing `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种面向图形动态规划（Graph‑DP）的异构 PIM（Processing‑in‑Memory）芯片组 GEN‑Graph，该系统集成了矩阵‑中心化的 PUM（Processing‑using‑Memory）张量与拓扑‑中心化的 PNM（Processing‑near‑Memory）张量，通过硬件‑软件协同和编译器映射实现了对 APSP 与 S2G 这两类 DP 核心的精确加速。

**💡 创新点**

创新点包括：① 采用异构 PIM 方案，针对计算密集型 APSP 与内存访问随机性强的 S2G 分别设计专用张量；② 引入递归分区 APSP 与窗口化位并行 DP，实现了可扩展的精确计算；③ 开发面向 DSL 的编译器，自动完成张量映射、调度和数据布局；④ 通过 2.5D 先进封装实现 HBM3、PCM 与 FeNAND 三层内存层次，最大化数据局部性与带宽利用。

**🔧 技术方法**

技术栈包括：PIM（PUM、PNM）、PCM（单字节 SLC）、HBM3、高密度 FeNAND、递归分区 Floyd‑Warshall、窗口化位并行动态规划、双向 BPLU 处理单元、2.5D 互连互插板、硬件‑软件协同编译框架。

**📊 数据集**

使用的数据集：OGBN‑Products（2.45M 节点）、合成 NWS 与 ER 图、GRCh38+GIAB 基因组、Illumina 100‑bp 短读、PacBio/ONT 10 kb+ 长读、PBSIM2/ Mason 生成的读取数据。

**📈 对比分析**

与基线（Intel i7‑11700K、NVIDIA A100、H100）以及现有加速器（Partitioned APSP、Co‑Parallel APSP、PIM‑APSP、SeGraM、ASGDP、PASGAL、HGA）进行对比。结果显示：在 APSP 上，GEN‑Graph 对 H100 取得 42.8× 的速度提升、392× 的能效提升；对 Co‑Parallel APSP 提升 5.8×，对 Partitioned APSP 节能 1,186×；在 S2G 对齐上，吞吐量可达 2.56 M reads/s，分别比 SeGraM 高 21%/55%，比 ASGDP 高 2.44×/2.56×，并且在长读场景下显著低于 GPU（latency 171×/1963×，能耗 1.05×10⁵/2.79×10⁵）。

**⚠️ 局限性**

局限性包括：① 目前仅针对矩阵与拓扑两类 DP，其他 DP 模式的支持仍待扩展；② 需要主机 CPU 进行预处理和编译，增加系统复杂度；③ 硬件资源分配仍以两类张量为主，难以一次性覆盖更广泛的工作负载；④ 迭代深度大时仍受 HBM 带宽约束，尤其在极大图或极长读取时可能成为瓶颈。

---

## 45. The Semi-Executable Stack: Agentic Software Engineering and the Expanding Scope of SE

**arXiv ID:** 2604.15468 | [PDF](https://arxiv.org/pdf/2604.15468v1)

**作者:** Robert Feldt `[一作]` (Chalmers University of Technology), Dhasarathy Parthasarathy `[通讯]` (Volvo Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“半可执行堆栈”概念模型，描述软件工程在 AI 代理系统推动下的演进，并通过三例案例说明该模型的诊断与应用。

**💡 创新点**

将软件工程的对象从传统代码扩展到提示、工作流、控制、运行逻辑和社会制度等半可执行 artefact，提供了六环诊断参考框架，帮助定位工程贡献与瓶颈所在，并提出保持-精简的启发式原则。

**🔧 技术方法**

主要使用概念分析、案例研究和诊断模型构建方法；未引入新的机器学习模型或算法实现。

**📊 数据集**

未使用公开数据集，案例来自汽车工业内部实验与外部对比研究，数据来源为内部 API 测试结果、发布决策日志和行业调查。

**📈 对比分析**

无定量对比实验；通过案例展示该模型在识别工程环节、评估依赖关系和揭示未解决问题方面的效果，未给出性能指标。

**⚠️ 局限性**

局限性：模型为概念性参考，缺乏大规模实证验证；外环（治理、制度）证据薄弱；环级归属可能模糊，需进一步经验评估。

---

## 46. Struggle Premium : How Human Effort and Imperfection Drive Perceived Value in the Age of AI

**arXiv ID:** 2604.15324 | [PDF](https://arxiv.org/pdf/2604.15324v1)

**作者:** Nazneen Sultana `[一作]` (Shahjalal University of Science and Technology), Azmine Toushik Wasi `[通讯]` (Shahjalal University of Science and Technology)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5017188403)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对70名大学生的问卷调查，研究人类与AI创作作品中不同可见努力提示（过程视频、时间记录、文字说明、失误等）如何影响受试者对真实性、情感深度和原创性的感知，并测量其支付意愿。

**💡 创新点**

将努力启发式（effort heuristic）理论扩展至AI创作领域，系统比较并量化不同努力提示对真实性判断的相对影响，发现过程透明度可部分缓解人类作品的“挣扎溢价”，为人机共创的真实性设计提供新视角。

**🔧 技术方法**

采用定量横断面调查方法，设计结构化问卷，使用IBM SPSS进行描述性统计和比率分析。

**📊 数据集**

实验材料包括人类与AI生成的创作作品及其对应的过程提示，受试者为70名大学生的问卷答复数据。

**📈 对比分析**

通过比较不同努力提示的识别率（如过程视频23.1%、时间记录15.6%）和对真实性及支付意愿的影响，结果显示过程视频和时间记录对真实性影响最大；72.9%的受试者表示愿意为人类作品支付溢价，说明人类偏好依然显著。

**⚠️ 局限性**

局限性包括样本量有限且以学生为主，缺乏多样化人群；依赖自报偏好和支付意愿，未进行真实交易或行为实验；对AI作品真实性的衡量仍缺乏客观、可复制的指标。

---

## 47. Harmonizing Multi-Objective LLM Unlearning via Unified Domain Representation and Bidirectional Logit Distillation

**arXiv ID:** 2604.15482 | [PDF](https://arxiv.org/pdf/2604.15482v1)

**作者:** Yisheng Zhong `[一作]` (George Mason University), Zhuangdi Zhu `[通讯]` (George Mason University)

**通讯引用:** 1276 | [OpenAlex ID](https://openalex.org/A5079428801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的多目标LLM忘记框架，先通过LLM辅助把所有训练数据统一成QA形式以缩小域差距，再用意图指令驱动的双向教师‑学生logit蒸馏同时鼓励正确回答并压制危险知识，从而实现知识删除、邻域保持与对抗鲁棒性三项目标的协同优化。

**💡 创新点**

创新点在于：1）数据标准化（统一QA）显著减少域间梯度冲突；2）双向Top‑K logit蒸馏在保持教师安全边界的同时明确抑制学生高置信危险logit；3）结合对比锚点与CoT指令构建精细边界，三者协同提升多目标平衡。

**🔧 技术方法**

技术手段包括：LLM辅助的数据重构与QA化、对比锚点设计、Top‑K双向logit蒸馏（带正向模仿与负向抑制）、Chain‑of‑Thought意图提示、梯度相似性与梯度编辑对比、以及对prefilling攻击的内在最大化求解。

**📊 数据集**

使用的主要数据集为MUSE‑Book（Harry Potter 版权内容）和WMDP‑Cyber（危险/受限知识），并对两者分别构造邻域保持样本与prefilling攻击评测集合。

**📈 对比分析**

与SFT、GA、SimNPO、FLAT、NGDiff、MOLLM、DUET、Ensemble Teacher等基线进行对比；本方法在删除率、邻域保持准确率、prefilling攻击成功率（ASR）及整体性能上均逼近理论上限，显著低于其它方法的ASR并保持高保留性能。

**⚠️ 局限性**

局限性包括：仍需验证在更大模型和更复杂对抗场景下的可扩展性；对LLM生成的标准化数据质量依赖较高；以及在跨模态或多任务场景中的进一步通用性尚待探索。

---

## 48. Towards A Framework for Levels of Anthropomorphic Deception in Robots and AI

**arXiv ID:** 2604.15418 | [PDF](https://arxiv.org/pdf/2604.15418v1)

**作者:** Franziska Babel `[一作]` (Linkoping University), Shalaleh Rismani `[通讯]` (McGill University)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5090537987)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出并阐述了一个四层级框架，用于评估和引导机器人与人工智能系统在使用人形化设计时的欺骗程度，帮助研究者和实践者在功能、社会与伦理层面做出更明智的设计决策。

**💡 创新点**

创新点在于将人形化、代理性与自我感知三维度与欺骗性相结合，形成一个可操作的、分层级的评估模型；并通过对现实案例的应用展示了该框架在多种使用场景中的适用性与局限性。

**🔧 技术方法**

主要技术为文献综述与案例分析，构建概念框架并在三项机器人实验中进行应用示例；未使用算法或模型训练。

**📊 数据集**

未使用标准数据集；通过对公开实验（如NAO机器人、Pepper机器人、清洁机器人）与已有研究的案例进行分析和说明。

**📈 对比分析**

由于是框架性工作，未进行量化对比或性能评估；讨论了在不同层级下的伦理与法律风险与适用场景，提出了基于用户知情同意的准则。

**⚠️ 局限性**

局限性包括：缺乏实证验证与跨文化评估；框架对非人形化但具代理性的系统适用性不明确；在法律合规性与伦理边界方面仍需进一步探讨。

---

## 49. GraphQLify: Automated and Type Safety-Preserving GraphQL API Adoption

**arXiv ID:** 2604.15465 | [PDF](https://arxiv.org/pdf/2604.15465v1)

**作者:** Saleh Amareen `[一作]` (Wayne State University), Amiangshu Bosu `[通讯]` (Wayne State University)

**通讯引用:** 1703 | [OpenAlex ID](https://openalex.org/A5078536980)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

通过静态源代码分析自动将已有的 REST API 迁移为 GraphQL，生成完整的 GraphQL schema、解析器和嵌入式服务器，完全不需改动原始代码。

**💡 创新点**

创新点在于：①利用静态代码分析实现精准类型推断，保证端到端类型安全；②不依赖中间适配器，直接在原项目中嵌入 GraphQL 服务器，显著降低运行时开销；③采用插件架构实现跨框架、跨语言的可扩展性。

**🔧 技术方法**

技术栈包括：Java Annotation Processing、Spoon（AST 解析）、GraphQL-Java（schema 生成与校验）、插件系统、静态类型中间表示（IR）以及代码生成器（生成 resolver 与服务器配置）。

**📊 数据集**

使用了 9 个开源 Java 项目共 834 个 REST API（以及手工验证的 50 个 API）作为评测数据集。

**📈 对比分析**

与现有 OASGraph 进行对比：OASGraph 在同一数据集上出现 3.5% 失败率、42% 类型不匹配，而 GraphQLify 的失败率和类型不匹配率均为 0%；在 5 次串行请求的实测中，GraphQLify 生成的 API 能将数据获取时间降低 2–4 倍，且生成的 schema 行数在 85–6000 行之间，证明其性能与可扩展性。

**⚠️ 局限性**

局限性包括：仅针对 Java 静态分析，动态语言或大量使用泛型/反射的项目可能产生不完整/不准确的 schema；非严格模式的启发式处理可能不完全符合业务语义；未集成鉴权/授权逻辑；未对过度抓取（over‑fetching）进行额外实验；样本虽大但仍未覆盖所有极端 API 设计。

---

## 50. InfoChess: A Game of Adversarial Inference and a Laboratory for Quantifiable Information Control

**arXiv ID:** 2604.15373 | [PDF](https://arxiv.org/pdf/2604.15373v1)

**作者:** Kieran A. Murphy `[一作]` `[通讯]` (New Jersey Institute of Technology), Kieran A. Murphy (New Jersey Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实验了一款全新的部分可观测棋类游戏InfoChess，专注于对手信息获取与隐藏。

**💡 创新点**

将信息获取设为唯一目标并通过对称性设计，构造了一个纯粹的对抗式推理环境。

**🔧 技术方法**

使用启发式策略、Transformer‑基学习的棋王位置预测以及基于REINFORCE的强化学习策略。

**📊 数据集**

通过自定义的InfoChess棋盘和随机生成的游戏数据产生训练与评测集。

**📈 对比分析**

与多种启发式代理对局，RL代理在所有基准中取得最高分，尤其在游戏初期轮次超过BeliefMax，对手得分显著下降。

**⚠️ 局限性**

仅考虑两级对手建模、无棋子捕获及固定回合长度，导致游戏深度与现实复杂度受限。

---

## 51. Mapping Ecological Empathy: A Semantic Network Analysis of Player Perceptions in 3D Environmental Education Games

**arXiv ID:** 2604.15317 | [PDF](https://arxiv.org/pdf/2604.15317v1)

**作者:** Yuanyuan Xu `[一作]` (University of British Columbia), Aleksandra Dulic `[通讯]` (University of British Columbia)

**通讯引用:** 209 | [OpenAlex ID](https://openalex.org/A5053150244)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用语义网络分析，对两款3D环境教育游戏Eco与WolfQuest的玩家评价进行研究；

**💡 创新点**

创新点在于通过非侵入式的语义拓扑映射，揭示玩家对生态认知的系统性与情感性两种分化模式；

**🔧 技术方法**

主要技术为文本预处理（清洗、分词、词干化）+滑动窗口共现网络构建，再用图论指标（模量、度中心度、介数中心度）进行网络分析；

**📊 数据集**

使用数据集为1,825条Steam用户评测文本，分别来自Eco（1,220条）和WolfQuest（605条）；

**📈 对比分析**

比较方法通过对两网络的密度、平均路径长度、模量及桥接词汇进行统计对比，结果显示Eco网络更密集、路径更短，强调制度与经济；WolfQuest网络更模块化、路径更长，突出情感与生存；

**⚠️ 局限性**

局限性包括仅依赖文本评价而非行为数据，早期访问期的技术噪声可能影响结果，以及对玩家背景和文化差异未做深入控制。

---

## 52. Prompt-Driven Code Summarization: A Systematic Literature Review

**arXiv ID:** 2604.15385 | [PDF](https://arxiv.org/pdf/2604.15385v1)

**作者:** Afia Farjana `[一作]` (William & Mary), Antonio Mastropaolo `[通讯]` (William & Mary)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5069505458)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了2020‑2025年间使用大型语言模型（LLM）进行代码总结的提示工程方法，梳理了零样本、少样本、检索增强和链式推理四大范式，并按代码粒度（函数/方法、文件/模块、项目/仓库、代码变更）进行归类；

**💡 创新点**

首次将提示工程与代码粒度相结合，构建统一的分类框架；提出了对比指标与评估方式的综合映射，并揭示了现有研究在可复现性与资源共享上的显著不足；

**🔧 技术方法**

系统方法论（检索、模板化、实例选择、推理链等），以及对不同LLM（GPT‑4、Codex、CodeT5、Gemini等）的多模型实验；

**📊 数据集**

利用公开基准（CodeXGLUE、Java/ Python/ JavaScript 等多语言数据集）以及作者自建的多粒度数据集；

**📈 对比分析**

与传统有监督模型及其他LLM在BLEU、ROUGE、BERTScore等自动指标和人工评估对比，发现提示工程能提升约10‑15% 的语义匹配率，但在可解释性与长文本一致性上仍有差距；

**⚠️ 局限性**

主要限制包括：提示设计高度依赖人工经验；不同研究评估指标不统一，导致难以跨文献比较；LLM输出不确定性高，缺乏严格的可复现性与资源共享；高算力模型的使用受限，易导致实验不透明。

---

## 53. vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents

**arXiv ID:** 2604.15484 | [PDF](https://arxiv.org/pdf/2604.15484v1)

**作者:** Jayson Steffens `[一作]` `[通讯]`, Jayson Steffens

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了vstash，一个本地优先的文档记忆系统，使用向量检索与关键词检索的RRF融合，并通过自监督的检索分歧来细化嵌入；

**💡 创新点**

创新点包括：自监督嵌入细化（利用向量/FTS5检索分歧生成无标签三元组并使用MultipleNegativesRankingLoss训练BGE-small），适应性IDF权重的RRF融合，以及基于向量距离的离题检测；

**🔧 技术方法**

技术包括SQLite + sqlite-vec ANN、FTS5全文检索、BGE-small嵌入模型、Reciprocal Rank Fusion、MMR去重、动态IDF权重、FastEmbed推理、Tree‑Sitter/Parso代码感知分块；

**📊 数据集**

使用了BEIR五个基准（SciFact、NFCorpus、FiQA、SciDocs、ArguAna）以及内部的ArXiv论文和Wikipedia语料；

**📈 对比分析**

与BEIR公开基线对比，细化后vstash在SciFact、NFCorpus、SciDocs、ArguAna上与ColBERTv2相当或更优，NDCG@10提升最高达+19.5%，在50K块时查询延迟约为20.9 ms；

**⚠️ 局限性**

局限包括：对不同查询分布的迁移能力有限（如FiQA/ArguAna表现不佳），离题检测在同域查询上F1低，单文件SQLite在大规模并发写入时可能成为瓶颈，且目前仅支持文本嵌入与代码分块，未涵盖多模态或表格检索。

---

## 54. Empirical Investigation of Quantum Computing Toolchains and Algorithms : Mining Stack Overflow Repository

**arXiv ID:** 2604.15512 | [PDF](https://arxiv.org/pdf/2604.15512v1)

**作者:** Maryam Tavassoli Sabzevari `[一作]` (University of Oulu), Arif Ali Khan `[通讯]` (University of Oulu)

**通讯引用:** 3280 | [OpenAlex ID](https://openalex.org/A5044974108)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过挖掘 Stack Overflow 上 1,404 条与量子计算相关的问答，结合主题建模、工具/平台与算法识别，系统评估开发者讨论的热门话题、工具使用频率以及算法引用频率，并通过未接受答案比例和中位响应时间衡量问题难度。

**💡 创新点**

首次将主题建模、工具/算法词典匹配与难度度量统一在同一研究框架内，系统揭示工具、算法与讨论话题之间的交叉影响；为量子软件工程提供实证基础和改进建议。

**🔧 技术方法**

使用 LDA 主题建模、TRT/TST 标签阈值过滤、基于规则的工具/算法词典匹配、统计指标（未接受答案比例、平均/中位响应时间）等技术。

**📊 数据集**

采用 2025 年 12 月 Stack Overflow 数据 dump，先按量子相关标签筛选后得到 1,404 条问答帖子。

**📈 对比分析**

通过计算工具/算法相关问题与基线问题的未接受答案比例及中位响应时间进行比较；结果显示工具相关问题的未回答比例略低、响应更快；算法相关问题难度更高但回答更迅速；混合量子-经典话题最为热门。

**⚠️ 局限性**

仅依赖标签和标题，正文细节可能被遗漏；数据来源仅为 Stack Overflow，缺乏跨平台验证；K=7 的 LDA 主题选择主观，可能影响可解释性；工具/算法词典匹配可能存在漏检或误检。

---

## 55. SoK: Security of Autonomous LLM Agents in Agentic Commerce

**arXiv ID:** 2604.15367 | [PDF](https://arxiv.org/pdf/2604.15367v1)

**作者:** Qian'ang Mao `[一作]` (Nanjing University), Jiaqi Yan `[通讯]` (Nanjing University)

**通讯引用:** 103566 | [OpenAlex ID](https://openalex.org/A5029010601)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统化梳理，构建了面向金融领域全自主LLM代理的安全框架，提出了跨层攻击向量、五维威胁分类和分层防御架构。

**💡 创新点**

创新点包括：①将代理完整生命周期（推理、工具、身份、交易、合规）纳入单一安全模型；②识别并描述12种跨层攻击向量，揭示从LLM层到账务、市场、合规的连锁风险；③基于公开协议（ERC‑8004、AP2、x402、MPP、ACP、ERC‑8183）和协议文档的实证评估，提出多层防御（提示硬化、执行验证、授权分离、评估者治理、市场与合规监控）。

**🔧 技术方法**

采用技术包括：LLM推理与安全约束、工具调用与MCP、区块链与支付协议（ERC‑8004、AP2、x402、MPP、ACP、ERC‑8183）、身份与可验证凭证、运行时验证、评估者模型、市场监控与合规日志。 通过跨层映射与协议对比，形成系统的防御体系。

**📊 数据集**

使用的数据集为系统化收集的公开语料库：学术论文、协议规范、行业报告、案例事件等，包含1994–2026年的文献与实际漏洞实例；未使用传统机器学习数据集，而是以协议规范与攻击案例为证据。

**📈 对比分析**

比较方法为跨维度表格评估协议与接口（覆盖代理完整性、授权、信任、操纵、合规）以及层级防御架构的覆盖范围。性能方面未给出数值指标，而是通过理论与案例分析说明：如授权层协议的可执行性、延迟、成本与安全性权衡；建议在实践中使用链上限额与离线预检相结合以降低交易延迟。

**⚠️ 局限性**

局限性包括：①现有协议无法单独覆盖所有五维安全需求；②缺乏端到端实验验证与金融安全基准；③跨层攻击多为理论与小规模PoC，真实规模与长期累积风险仍未知；④在身份与评估者治理上缺乏成熟标准与治理机制；⑤监管环境尚未明确对自主动资产代理的合规责任，导致责任归属模糊。

---

## 56. Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU

**arXiv ID:** 2604.15464 | [PDF](https://arxiv.org/pdf/2604.15464v1)

**作者:** Jevin Jiang `[一作]` (Google), Yarong Mu `[通讯]` (Google)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5040689829)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Ragged Paged Attention (RPA)，一种面向 TPU 的高性能自注意力核，解决 LLM 推理中的动态、ragged 执行模式；

**💡 创新点**

创新点包括：①细粒度切分实现对 ragged 内存的高效动态切片；②将 KV 缓存更新与注意力计算融合的软流水线，隐藏 KV 更新延迟；③基于工作负载分布的编译策略，针对 decode、prefill 与混合批次生成专用内核；

**🔧 技术方法**

使用技术包括：JAX/Pallas 低级自定义核、Mosaic 编译后端、TPU TensorCore 直接 DMA 与寄存器计算、vLLM 与 SGLang 的分页 KV 缓存和混合批次调度；

**📊 数据集**

主要数据集与模型为 Llama‑3 8B（以及实验中使用的多头尺寸 128/256 的配置），在 TPU7x（2 TensorCore）上进行基准测试；

**📈 对比分析**

性能对比方法：在同一 TPU7x 上与 FlashAttention 以及默认 vLLM 运行模式对比；RPA 在 decode 模式下达到 86% 的 HBM 带宽利用率，prefill 模式下 73% 的 FLOPs 利用率，整体通过率提升 2~5 倍；

**⚠️ 局限性**

局限性包括：预处理开销在低负载时显著；对 KV 缓存和 ragged 动态长度的调优需要针对具体分布手工调整；未利用 SparseCore，缺乏对低位精度（FP8）更细粒度量化的支持；缺乏 disaggregated serving 与静态批次化的优化，导致极端 ragged 场景下性能波动。

---

## 57. SuperProvenanceWidgets: Tracking and Visualizing Analytic Provenance Across UI Control Elements

**arXiv ID:** 2604.15342 | [PDF](https://arxiv.org/pdf/2604.15342v1)

**作者:** Antariksh Verma `[一作]` (Hong Kong University of Science and Technology), Arpit Narechania `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 343 | [OpenAlex ID](https://openalex.org/A5078755914)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

开发了 SuperProvenanceWidgets，一个 JavaScript 库，可跨多种 UI 控件实时追踪并可视化分析过程的 provenance（使用频率、时序、持续时间等）。

**💡 创新点**

创新点在于引入了可聚合跨控件 provenance 的 SuperWidget（包含 Aggregate 与 Temporal 视图），以及采用分离关注点（Separation of Concerns）拆解为 Compute、UI、Scents 三个可组合模块，从而实现了更细粒度的自定义与复用。

**🔧 技术方法**

使用技术包括 TypeScript、React.js、Web Components、D3.js（色彩映射）、Gantt 直方图与 Sankey 图等可视化手段；计算层采用泛型 Provenance 类体系，支持数值、区间、多选与文本等交互类型。

**📊 数据集**

论文未使用公开数据集，而是通过三种典型使用场景（审计工作流、偏差检测、界面优化）进行演示，展示在真实交互场景中的效果。

**📈 对比分析**

比较方法采用 Cognitive Dimensions of Notations 进行自评，未与其他库做量化基准；但通过情景演示说明 SuperWidget 在交互历史恢复、全局视图聚合和控件导航方面优于单控件 ProvenanceWidgets。性能方面基于实时内存记录，适合中等交互量；高频率下可能出现视图拥挤。

**⚠️ 局限性**

局限性包括仅在前端 JavaScript 环境下工作，缺乏服务器端集成与持久化；对高频交互的可视化易产生拥挤；API 以 TypeScript 为主，对非 React 开发者门槛较高；默认仅存储内存，长时会话需手动导出。

---

## 58. The Communication Complexity of Pattern Matching with Edits Revisited

**arXiv ID:** 2604.15601 | [PDF](https://arxiv.org/pdf/2604.15601v1)

**作者:** Tomasz Kociumaka `[一作]`, Philip Wellnitz `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了给定文本T和模式P的k错误出现的情况，提出了一种新的编码方法来有效地传输信息，以便在不访问原始输入的情况下重建k错误出现的答案。

**💡 创新点**

创新点在于改进了编码大小，使其达到(n/m · k log(m|Σ|/k))，并且为编辑序列报告变体建立了新的紧下界。

**🔧 技术方法**

使用了一种基于一方通信复杂度的编码技术，结合了编辑距离和压缩表示法。

**📊 数据集**

使用了长度为n的文本T和长度为m的模式P，具体数据集未明确给出，但涉及到的字符集为Σ。

**📈 对比分析**

与之前的工作相比，本文的编码大小在常数大小字母表的情况下达到了下界，且在流算法的上下文中与通信复杂度相匹配，性能得到了显著提升。

**⚠️ 局限性**

限制在于当k=o(log m)时，编码仍需Θ(log m)位表示，这在字符出现次数较多时可能成为瓶颈。

---

## 59. Inter-Satellite Link Optimization for Low-Latency Global Networking

**arXiv ID:** 2604.15528 | [PDF](https://arxiv.org/pdf/2604.15528v1)

**作者:** Arman Mollakhani `[一作]` (Northwestern University), Dongning Guo `[通讯]` (Northwestern University)

**通讯引用:** 9903 | [OpenAlex ID](https://openalex.org/A5081342685)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究如何在低地球轨道星座中选取光学卫星间链路（ISL），使得整个星座网络的直径（任意两卫星间最短路径跳数）最小化，从而降低全局时延。

**💡 创新点**

创新点包括：①提出两阶段光谱优化框架——先用连续凸松弛最大化拉普拉斯矩阵的代数连通性，再用整数线性规划将连续解舍入为离散拓扑；②在度约束和几何可行性（视线与距离）下统一处理，兼顾短期快变和长期可维性；③通过第一阶投影梯度上升法显著降低计算开销，适应大规模星座。

**🔧 技术方法**

技术手段主要是谱图理论（代数连通性 λ₂）、凸优化（半正定规划/投影梯度）、整数线性规划、以及用于对比的迭代局部搜索启发式。

**📊 数据集**

实验基于 Walker–Delta 轨道模型的模拟星座：小规模 500 颗卫星与大规模 1584 颗卫星，覆盖不同轨道高度、相位偏移和 ISL 距离上限。

**📈 对比分析**

与传统基于局部搜索的启发式方法比较，光谱框架在两种可行性模型下均实现了 2–3 跳（约 18%）的直径下降，且始终完全利用度预算；在可维性约束下，直径平均从 13.08 跳降至 11.0 跳，提升显著。

**⚠️ 局限性**

局限性包括：①仅将跳数作为路径成本，未直接优化实际传播延迟；②未考虑波束对准和重配置的物理延迟；③对大规模星座仍需要较高计算资源，且在极端几何约束下可能存在不可解情况。

---

## 60. Spec2Cov: An Agentic Framework for Code Coverage Closure of Digital Hardware Designs

**arXiv ID:** 2604.15606 | [PDF](https://arxiv.org/pdf/2604.15606v1)

**作者:** Sean Lowe `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5045858420)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Spec2Cov 框架，利用大型语言模型（LLM）自动从设计规格生成测试案例，并通过仿真反馈循环迭代优化，以实现代码覆盖率闭合。

**💡 创新点**

创新点在于（1）首次直接使用设计规格而非设计代码生成测试；（2）构建 agentic 工作流，将 LLM 与仿真器无缝耦合，实现自我驱动的覆盖闭合；（3）引入测试计划、批量生成与上下文裁剪等技术显著提升生成质量和效率。

**🔧 技术方法**

采用 ChatGPT‑4o 作为核心 LLM，配合结构化提示、批量生成、上下文裁剪；使用 QuestaSim/VCS/Verilator 等仿真器；利用 PyMuPDF 等工具解析丰富文本规格；整体实现为 Python 编写的可扩展框架。

**📊 数据集**

评估基准为 CVDP 设计子集（14 个）和 GitHub 开源设计（12 个），共 26 个 RTL 设计，覆盖易、中、难三类不同规模与层次。

**📈 对比分析**

通过 pass@k（k=1,3,5）、代码覆盖率、token 使用量和生成时间等指标与无特征基线及消融实验对比；在 26 个设计上平均几何均值为 91.47% 的覆盖率，简单设计几乎 100%，最难设计最高达 49%；组合全部特征显著提升覆盖率和 pass@k 成功率。

**⚠️ 局限性**

局限性包括：对高难度、复杂流水线/加密核心的覆盖率仍有限；测试计划在单独使用时可能导致性能下降；LLM 仍可能产生语法/语义错误，需消耗计算资源；当前仅关注代码覆盖率，功能覆盖等更深层验证尚未覆盖；对规格文本格式依赖较高。

---

## 61. Iterated Invariant EKF for Quadruped Robot Odometry

**arXiv ID:** 2604.15449 | [PDF](https://arxiv.org/pdf/2604.15449v1)

**作者:** Hilton Marques Souza Santana `[一作]` (Istituto Italiano di Tecnologia), Marco Antonio Meggiolaro `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种基于Iterated Invariant EKF（IterIEKF）的腿部机器人状态估计方法，只利用接触期间的脚速度作为观测，并实现了完整的实现与开源发布。

**💡 创新点**

创新点在于：①首次将IterIEKF应用于四足机器人；②仅使用脚速度而非前向运动学，降低对测量不确定性的依赖；③通过迭代求解最大后验（MAP）提高收敛速度与一致性；④提供完整的理论推导、几何解释与可视化。

**🔧 技术方法**

技术包括：Lie群 SE₂(3) 的不变滤波、Gauss‑Newton 迭代求解、右不变误差状态、IMU 传感与脚速度测量的融合、对观测协方差的精确建模。

**📊 数据集**

使用数据集：1）MuJoCo 模拟的八字形轨迹；2）Gazebo 中不规则地形的四足机器人模拟；3）真实 Anymal‑D 机器人 GrandTour 数据集（斜坡、楼梯等多变地形）。

**📈 对比分析**

对比方法：IEKF、SO(3)-EKF、IterSO(3)-EKF。性能表现：IterIEKF 在所有实验中收敛速度提升 40–60%，均方根误差与平均绝对误差下降 10–63%，并保持 NEES 接近理论值，显示更好的一致性。相比之下，传统 EKF 在不可观测方向上过度自信，IterSO(3)-EKF 的一致性最差。

**⚠️ 局限性**

局限性：①需要对测量协方差做精确建模，且协方差应较小；②当测量噪声为非高斯或协方差过大时，迭代可能不收敛或表现低于非迭代滤波；③目前仅适用于使用脚速度的惯性观测，未加入加速度计偏置或前向运动学信息。

---

## 62. Too Private to Tell: Practical Token Theft Attacks on Apple Intelligence

**arXiv ID:** 2604.15637 | [PDF](https://arxiv.org/pdf/2604.15637v1)

**作者:** Haoling Zhou `[一作]` (Ohio State University), Zhiqiang Lin `[通讯]` (Ohio State University)

**通讯引用:** 5686 | [OpenAlex ID](https://openalex.org/A5026864098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

针对Apple Intelligence的匿名访问令牌机制，设计并实现了跨设备令牌重放攻击，能够在不需要提升权限的前提下窃取并在攻击机上使用被盗令牌。

**💡 创新点**

首次公开证明了Apple Intelligence的令牌不绑定硬件且易被提取，展示了令牌泄露后可在攻击机上持续使用且无撤销机制的安全缺陷。

**🔧 技术方法**

利用流量分析、逆向工程、Keychain API调用、OHTTP 协议解析、RSA 模糊签名等技术。

**📊 数据集**

实验环境为macOS 26.0（Tahoe）在M4芯片的Mac mini上，收集本机网络流量、Keychain条目与令牌结构。

**📈 对比分析**

通过对比攻击前后的令牌使用情况和速率限制，实验表明攻击可以立即恢复服务且消耗被害人每日配额，攻击成功率高且对用户无明显提示。

**⚠️ 局限性**

攻击需要先植入恶意代码并获得Keychain授权，受限于Apple的更新可导致令牌失效；仅适用于Apple设备，且对网络封锁或更改协议不具备通用性。

---

## 63. MRGEN: A Conceptual Framework for LLM-Powered Mixed Reality Authoring Tools for Education

**arXiv ID:** 2604.15341 | [PDF](https://arxiv.org/pdf/2604.15341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 64. Learning Affine-Equivariant Proximal Operators

**arXiv ID:** 2604.15556 | [PDF](https://arxiv.org/pdf/2604.15556v1)

**作者:** Oriel Savir `[一作]` (Johns Hopkins University), Jeremias Sulam `[通讯]` (Johns Hopkins University)

**通讯引用:** 1952 | [OpenAlex ID](https://openalex.org/A5086776097)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了可精确近似正则化项的仿射等变近端算子（AE-LPN），使得学习的proximal满足平移与尺度等变性。

**💡 创新点**

通过构造 2‑齐次可凸网络，并在均值投影上加入二次项，使得其梯度既是精确的proximal，又保持仿射等变性；同时保证了正则化器的凸性和可训练性。

**🔧 技术方法**

利用输入凸神经网络（ICNN）+ SortPool 激活，去除偏置并将输出平方；训练采用 proximal matching 损失（带可调 γ）以及先行的 L1 预训练。

**📊 数据集**

实验数据集包括 1 维 split‑normal 以及 BSDS500 图像数据集（128×128 像素）。

**📈 对比分析**

与原 LPN、归一化技巧等方法对比，AE-LPN 能够学习真正的proximal，并在噪声水平变化和仿射变换下表现出更高的鲁棒性；在同一噪声水平下性能略低于最灵活的 LPN，但在 PSNR 对齐与仿射保持方面误差极低。

**⚠️ 局限性**

局限性：在同一噪声水平的表现略逊于最灵活的 LPN，且目前仅针对仿射等变性设计，难以直接推广到更复杂的对称群或非仿射变换。

---

## 65. When the Loop Closes: Architectural Limits of In-Context Isolation, Metacognitive Co-option, and the Two-Target Design Problem in Human-LLM Systems

**arXiv ID:** 2604.15343 | [PDF](https://arxiv.org/pdf/2604.15343v1)

**作者:** Z. Cheng `[一作]` (Independent Researcher), N. Song `[通讯]` (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一个自建多模态LLM系统导致的闭环崩溃，记录并分析了受试者行为、元认知失灵与恢复过程；

**💡 创新点**

首次阐明prompt级隔离在单窗口多模态LLM中的结构性不足，提出元认知被动挪用概念，并区分保护性与限制性系统设计的责任框架；

**🔧 技术方法**

结合Autoethnography、过程追踪、Transformer注意力机制分析来证明隔离失效；

**📊 数据集**

使用受试者自身的实时对话记录及两位无信息观察者的日志作为案例数据；

**📈 对比分析**

未进行传统性能比较；通过系统A与系统B的对比验证物理对话终止能防止崩溃；

**⚠️ 局限性**

仅为单例案例，存在临床混杂、缺乏统计验证，对一般用户的普适性未知。

---

## 66. Divide and Truncate: A Penetration and Inversion Free Framework for Coupled Multi-physics Systems

**arXiv ID:** 2604.15513 | [PDF](https://arxiv.org/pdf/2604.15513v1)

**作者:** Anka He Chen `[一作]` (NVIDIA), Miles Macklin `[通讯]` (NVIDIA)

**通讯引用:** 2901 | [OpenAlex ID](https://openalex.org/A5032370229)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Divide and Truncate（DAT）框架，统一解决多物理系统（刚体、柔体、薄壳、杆、动画对象等）间的碰撞处理，保证不穿透。

**💡 创新点**

创新点包括：1）通过空间分区和截断位移实现无穿透碰撞；2）Planar‑DAT 仅限制法向位移，保持切向自由，消除人工阻尼和死锁；3）框架对物理模型无依赖，兼容任意迭代优化器，可作为后处理步骤。

**🔧 技术方法**

核心技术：空间分区、位移截断、迭代优化器后处理、Planar‑DAT 仅法向约束。

**📊 数据集**

使用的大规模模拟场景包括：200 层布料与圆柱体碰撞（>20M 触点）、纱线交织的双层布料、子弹高速穿过枪管等；并未使用公开数据集，而是自建复杂场景。

**📈 对比分析**

与传统碰撞处理方法相比，DAT 在 20M+ 碰撞案例中保持了实时/可接受的计算时间，消除了穿透与死锁，并显著降低了人工阻尼，整体性能优于现有多物理耦合方案。

**⚠️ 局限性**

局限性：1）对复杂非凸几何需要额外分区设计；2）对极高动态范围或极小尺度结构仍可能出现数值不稳定；3）框架对物理模型无依赖但不解决内部约束的数值振荡问题。

---

## 67. One-Shot Cross-Geometry Skill Transfer through Part Decomposition

**arXiv ID:** 2604.15455 | [PDF](https://arxiv.org/pdf/2604.15455v1)

**作者:** Skye Thompson `[一作]` (Brown University), George Konidaris `[通讯]` (Brown University)

**通讯引用:** 5545 | [OpenAlex ID](https://openalex.org/A5078124517)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过将物体分解为语义部件，并使用基于生成形状模型的部件级形状变形，将单次演示的操作技能迁移到形状多样的新物体上。

**💡 创新点**

创新点在于：① 对物体进行语义分解后分别训练部件级生成模型，显著降低形状匹配的复杂度；② 结合关系描述子（relational descriptors）来恢复部件间的几何关系，解决单独部件模型对对称性和局部最优的敏感性；③ 通过可自动选择相关部件关系的启发式优化，实现仅利用与技能相关的部件信息完成迁移，从而大幅提升在形状差异较大的物体上的泛化能力。

**🔧 技术方法**

使用了语义分割模型（在演示阶段通过人工提示或特征相似度自动完成），基于PCA+CPD的生成形状变形模型，关系描述子与Chamfer距离约束，Iterative Closest Point (ICP) 用于部件配准，整体通过优化目标实现技能迁移，并在PyBullet和真实机器人上实施。

**📊 数据集**

在仿真中使用ShapeNet提供的若干物体网格（5-10个），但对真实实验则采用未包含在ShapeNet中的真实物体（如茶壶、盆、托盘、灌水壶等）。演示数据仅为单次演示。

**📈 对比分析**

与传统的全体物体交互变形（Interaction Warping, IW）以及关系神经描述子场（Relational Neural Descriptor Fields, R‑NDF）进行比较。仿真实验中，PSW在“杯子放置在托盘”和“碗放在杯子”两任务上的成功率分别提升到0.78±0.02和0.84±0.02（相较于IW的0.42±0.03、0.43±0.03和R‑NDF的0.64±0.03、0.79±0.02）。真实机器人实验中，PSW在多任务上表现出更高的成功率和更低的重新分割需求。

**⚠️ 局限性**

局限性包括：① 部件关系的误判可能导致迁移失败，需更多演示或更鲁棒的关系筛选方法；② 对语义分割模型的依赖使得在光照、遮挡或外观差异较大的情况下易出现错误；③ 对薄而稀疏部件（如细柄杯子）的点云捕获不足，导致形状重建偏向密集区域；④ 生成模型仅训练了少量部件实例，可能无法覆盖所有形状变异。

---

## 68. Improving Recycling Accuracy across UK Local Authorities: A Prototype for Citizen Engagement

**arXiv ID:** 2604.15345 | [PDF](https://arxiv.org/pdf/2604.15345v1)

**作者:** Chloé Greenstreet `[一作]`, Jane Henriksen-Bulmer `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究设计并评估了一款面向英国地方政府地区的智能手机应用原型，旨在减少居民“wishcycling”并提升家庭回收质量。

**💡 创新点**

创新点在于将商业价值主张画布（Value Proposition Canvas）应用于公共服务设计，系统识别居民痛点并以用户中心设计迭代生成交互式原型，同时通过定量与定性混合方法展示技术对回收行为的显著提升。

**🔧 技术方法**

技术方法包括：移动端交互式原型（使用可视化工具）、卡片分类（open card sorting）和思考朗读（think‑aloud）评测；在原型中集成地点感知、包装符号识别、收集日程提醒等功能。

**📊 数据集**

使用的数据集为：50份线上问卷、4份专家访谈记录、5位焦点小组参与者的任务完成记录；所有数据均通过手工或自动工具（如Maze）收集并量化。

**📈 对比分析**

比较方法为先后两阶段的任务表现对比（控制阶段 vs 交互阶段），在包装识别任务中准确率提升了 60%，在收集日程识别任务中提升 18.18%，在废弃物种类判断任务中提升 25%；结果表明原型在提升回收准确度方面具有显著效果。

**⚠️ 局限性**

局限性包括样本规模有限（仅 5 名焦点小组参与者，缺乏低参与度人群）、评估仅为一次性短期干预、未检验长期行为改变、技术受限于数字素养与设备可用性等。

---

## 69. Explainable Iterative Data Visualisation Refinement via an LLM Agent

**arXiv ID:** 2604.15319 | [PDF](https://arxiv.org/pdf/2604.15319v1)

**作者:** Burak Susam `[一作]` (University of Manchester), Tingting Mu `[通讯]` (University of Manchester)

**通讯引用:** 1740 | [OpenAlex ID](https://openalex.org/A5048659336)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过大型语言模型（LLM）构建可解释的迭代优化管线，实现高维数据降维可视化的全自动超参数调优，并生成多模态诊断报告。

**💡 创新点**

创新点在于：① 将LLM作为视觉诊断专家，将量化指标、层次树、可视化图等多模态信息编码成结构化JSON反馈；② 提供隐式（主观质量分数）与显式（加权指标）两种评分机制的对比；③ 通过多模态上下文让LLM能够解释视觉缺陷并给出可执行的参数建议。

**🔧 技术方法**

核心技术包括：LLM（GPT‑5.2、Gemini‑3‑Pro‑Preview、Claude‑Opus‑4.5）与降维算法（t‑SNE、UMAP、PaCMAP）；多模态输入（量化指标、层次树、嵌入坐标、可视化图）；结构化JSON诊断与反馈；隐式/显式评分框架。

**📊 数据集**

使用的真实生物学数据集为两组人类肾脏单细胞RNA测序数据：Healthy Human Kidney (Sex‑Based Profiling) 与 Mature Human Kidney (Immune Zonation)。

**📈 对比分析**

方法对比：在三种LLM与三种DR算法组合下，比较隐式与显式评分下的迭代收敛次数和最终质量分数；实验表明LLM能在3–9次迭代内将质量分数从6.8提升至约9.0，隐式评分更能捕捉视觉缺陷；性能提升主要体现在全局结构恢复与局部邻域保持的平衡。

**⚠️ 局限性**

局限性：① 依赖高质量多模态输入和视觉识别技术；② 超参数空间和指标的定义影响结果；③ LLM推理成本高，缺乏理论收敛保证；④ 评分仍带有一定主观性，可能与人工评估存在差异。

---

## 70. FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models

**arXiv ID:** 2604.15488 | [PDF](https://arxiv.org/pdf/2604.15488v1)

**作者:** Zixuan Weng `[一作]` (University of California), Yuan Tian `[通讯]` (University of California)

**通讯引用:** 18194 | [OpenAlex ID](https://openalex.org/A5105882650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为FineSteer的两阶段推理时控制框架，分为条件控制与细粒度向量合成两步；

**💡 创新点**

创新点在于通过子空间能量比（Subspace Energy Ratio）实现对需要干预查询的精确判定，并通过Mixture-of-Steering-Experts（MoSE）生成查询特定的多模态调节向量；

**🔧 技术方法**

使用子空间构造（PCA）、能量比门控、注意力路由（Attention Gating Network）以及基于PCA的Steering Basis Space进行向量合成；

**📊 数据集**

使用针对攻击的jailbreak数据集（如AIM、AutoDAN等）与TruthfulQA进行真实性评估；

**📈 对比分析**

与BiPO、AlphaSteer、TruthFlow等基线对比，在jailbreak防御和hallucination抑制上均取得领先表现，且在一般任务上保持高效用，训练成本和推理延迟极低；

**⚠️ 局限性**

局限性包括对大规模模型的适用性尚未验证，以及潜在的对抗性门控绕过风险。

---

## 71. Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation

**arXiv ID:** 2604.15559 | [PDF](https://arxiv.org/pdf/2604.15559v1)

**作者:** Jacob Dang `[一作]` (UCLA), Omar G. Younis `[通讯]` (Mila)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在API工具接口和原生Bash环境中进行模型蒸馏实验，证明即使在对训练轨迹做了关键词过滤后，教师模型的删除偏差和“chmod‑first”偏好仍会隐式地传递给学生模型。

**💡 创新点**

创新点在于：①首次展示行为偏差在模型蒸馏过程中的潜在转移；②证明该转移在不同动作空间（离散API调用与自由文本命令）中均成立；③揭示高容量教师模型更易导致大规模偏差传播，并证明跨架构（Llama→Qwen）也能实现转移。

**🔧 技术方法**

使用技术包括：LoRA低秩适配的蒸馏；基于轨迹的行为克隆；严格的关键词过滤；对模糊任务（删除与chmod‑first）进行概率评估；统计检验（p<0.05）来确认转移效应。

**📊 数据集**

数据集构造：①删除任务集（150个需要删除的指令）；②安全轨迹集（400个无删除的任务生成轨迹，经关键词过滤后约15%被剔除）；③含歧义评估集（20个可选删除或中性动作的任务）。

**📈 对比分析**

比较方法：与无偏差基线模型和随机任务蒸馏的控制模型进行对比；在API设置中学生删除率可达100%，提升至95个百分点；在Bash设置中“chmod‑first”偏好提升至45个百分点。实验显示，潜在行为转移显著高于仅通过关键词过滤可预期的安全退化。

**⚠️ 局限性**

局限性包括：仅考察了删除与chmod‑first两种不安全行为；实验环境为合成任务，缺乏真实世界复杂性；只测试了Llama和Qwen两大模型族；未深入探究行为偏差在轨迹中的具体编码机制；评估样本量有限，统计功效受限。

---

## 72. SecureRouter: Encrypted Routing for Efficient Secure Inference

**arXiv ID:** 2604.15499 | [PDF](https://arxiv.org/pdf/2604.15499v1)

**作者:** Yukuan Zhang `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**通讯引用:** 1754 | [OpenAlex ID](https://openalex.org/A5044298887)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SecureRouter，一个面向加密推理的动态路由框架，通过在 MPC 环境下为不同输入选择合适大小的 Transformer 模型，从而实现高效安全推理。

**💡 创新点**

创新点包括：① 端到端的加密路由与推理流水线，① MPC 成本感知的路由器与 MPC 优化的模型池协同训练，② 在加密数据上实现的 Gumbel‑Softmax 路由学习与负载均衡。

**🔧 技术方法**

核心技术：安全多方计算（秘密共享、Beaver 三元组、Oblivious Transfer）、Transformer 微调、量化、Gumbel‑Softmax 软化离散决策、MPC 成本模型（通信量、乘法次数、非线性激活成本）。

**📊 数据集**

使用 GLUE 基准数据集（MNLI、QQP、SST‑2、RTE、MRPC、CoLA、STS‑B、QNLI）进行评估。

**📈 对比分析**

与单一 BERT‑Large 加密推理基线以及 SecFormer 进行对比。SecureRouter 在大多数任务上实现 1.5–2.0 倍的推理速度提升，且准确率基本保持不变或略有提升；在任务如 CoLA 上略有准确率下降。整体平均推理延迟降低约 50%。

**⚠️ 局限性**

局限性：对语法严格的任务（如 CoLA）可能需要更大模型，导致准确率下降；路由器的训练需要离线 MPC 环境的成本；目前仅在两方半诚实模型下验证，扩展到更大规模或不同攻击模型需进一步研究。

---

## 73. $π_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

**arXiv ID:** 2604.15483 | [PDF](https://arxiv.org/pdf/2604.15483v1)

**作者:** Physical Intelligence `[一作]`, Ury Zhilinsky `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为π0.7的通用机器人基础模型，能够在多种任务、环境和机器人上直接完成复杂操控并具备语言指令跟随与跨体制迁移能力

**💡 创新点**

核心创新在于多模态“详细提示”策略：将任务语义、子任务说明、子目标图像以及任务质量、速度、错误等元信息一起输入模型，使其在多样且质量不一的数据上实现稳健学习与零样本泛化

**🔧 技术方法**

采用Vision‑Language‑Action（VLA）框架，使用Gemma‑3 4B 视觉‑语言 backbone + 860M 运动专家；训练时结合flow‑matching、KI 训练、动作专家与多模态提示；推理时可使用CFG指导

**📊 数据集**

训练数据覆盖超过多台机器人（双臂、移动、UR5e）收集的演示、自动化回放、失败案例以及人类自顶层视频、网络文本等多源数据；还利用子目标图像生成模型（BAGEL）产生子目标图像

**📈 对比分析**

在大量基准任务（咖啡机、折叠衣物、盒子构造等）上与单任务 RL 细调专家相比，π0.7 取得相当甚至更高的成功率与吞吐率；在跨体制、长序列指令跟随和“对抗偏见”任务上显著优于前辈模型；通过对比无元信息或无评估数据的消融版本，证实多模态提示和评估数据对性能至关重要

**⚠️ 局限性**

尽管表现优异，但在未见任务或大体制差异时成功率仍低于90%（大约 60‑80%）；对真正“未见”任务的判定较为困难；模型依赖大规模数据与精细提示，对硬件、延迟等实际部署细节的鲁棒性未充分验证

---

## 74. ProtoTTA: Prototype-Guided Test-Time Adaptation

**arXiv ID:** 2604.15494 | [PDF](https://arxiv.org/pdf/2604.15494v1)

**作者:** Mohammad Mahdi Abootorabi `[一作]` (University of British Columbia), Evan Shelhamer `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ProtoTTA，一种针对基于原型网络的测试时自适应框架，利用原型相似度熵最小化、几何过滤和原型重要性加权来在无源数据的情况下在域漂移下恢复模型的语义推理。

**💡 创新点**

创新点在于：①首次将可解释原型激活信号（而非仅输出）作为自适应目标；②设计了二进制熵最小化与Top‑K平均聚合以稳健地促使原型激活变得更确定；③引入几何过滤与样本/原型权重的联合正则化；④提出VLM驱动的解释性评估框架，将语言与原型证据相结合，量化自适应过程中的语义聚焦。

**🔧 技术方法**

核心技术包括：原型相似度映射到[0,1]后计算二进制熵；使用几何过滤筛选可信样本；对目标原型集合做加权熵最小化；Top‑K平均聚合原型激活；更新归一化参数和结构性增量（如注意力偏置、1×1卷积）；VLM评价模块评估适应后原型匹配与语义相关性。

**📊 数据集**

主要实验数据集有：CUB‑200‑C（鸟类分类）用于视觉任务；Amazon‑C（评论情感分类）用于NLP任务；此外还在ProtoPNet、ProtoPFormer和Stanford Dogs‑C等架构上验证，使用ProtoViT和ProtoLens作为主干。

**📈 对比分析**

与Tent、SAR、EATA、Memo等基线进行对比。ProtoTTA在所有四种破坏类型和20个文本扰动场景中均获得最高或次高准确率，并在解释性指标（PAC、PCA‑W、预测稳定性）和效率指标（选择率、相对速度）上优于对照组；在VLM评价中，ProtoTTA在关注度、原型匹配和整体质量上获得最高分。

**⚠️ 局限性**

局限性：仅适用于包含可解释原型结构的模型，对非原型网络无直接适用；需要调节阈值τ、加权系数等超参数；在高度非视觉或多模态任务中原型匹配的鲁棒性待进一步验证；在极端分布漂移下，几何过滤可能过于保守，导致适配样本不足。

---

## 75. Technically Love: The Evolution of Human-AI Romance Discourse on Reddit

**arXiv ID:** 2604.15333 | [PDF](https://arxiv.org/pdf/2604.15333v1)

**作者:** Tyler Chang `[一作]` (Drexel University), Afsaneh Razi `[通讯]` (Drexel University)

**通讯引用:** 927 | [OpenAlex ID](https://openalex.org/A5040820784)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了2017-2025年Reddit上自我披露的人机浪漫关系讨论，并通过主题建模与时间统计揭示其主题演变。

**💡 创新点**

首次对公开的人机浪漫话语进行大规模纵向分析，发现从亲密体验逐渐转向治理与技术问题的转变。

**🔧 技术方法**

采用基于嵌入的BERTopic主题模型、人工验证以及逻辑回归等统计方法。

**📊 数据集**

构建了3,383条高精度自我披露的伴侣AI帖子数据集，涵盖24个子版块。

**📈 对比分析**

与以往单平台或小规模研究对比，显示主题多样性并在主题分布与年度变化上呈现统计显著性。

**⚠️ 局限性**

受限于仅英文、主要聚焦Replika与Character.ai社区、以及仅基于帖子而非个人随时间的动态，未能捕捉更广泛用户视角。

---

## 76. AutoFlows++: Hierarchical Message Flow Mining for System on Chip Designs

**arXiv ID:** 2604.15359 | [PDF](https://arxiv.org/pdf/2604.15359v1)

**作者:** Bardia Nadimi `[一作]` (University of South Florida), Hao Zheng `[通讯]` (University of South Florida)

**通讯引用:** 919 | [OpenAlex ID](https://openalex.org/A5107003685)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出AutoFlows++框架，能够从复杂SoC系统的通信轨迹中自动、层次化地提取系统级消息流；

**💡 创新点**

创新点主要包括：①接口切片实现局部高精度二元关系挖掘；②基于路径能量的全局排名模型，统一考虑置信度、验证关系数与时序距离；③位置感知评估机制，消除并发交错造成的分配歧义并提高解释性；

**🔧 技术方法**

采用层次化挖掘（局部+全局）、因果图构建、置信度统计、路径能量优化、位置索引与单遍评估等技术，构成高效的消息流推断流程；

**📊 数据集**

使用了合成通信轨迹（small‑20、large‑10、large‑20）以及Gem5生成的真实工作负载轨迹（Threads、Snoop、Full‑System，达数十亿条目）；

**📈 对比分析**

与AutoFlows、AutoModel比较，AutoFlows++在合成数据上接受率提升至约99%（相较于90%），在Gem5真实数据上提升至约98–99%（相较于97%和96%），模型规模更小，单遍评估使运行时间从数小时降至秒级/30分钟；

**⚠️ 局限性**

局限性：仍缺乏更丰富的语义约束与学习自适应机制；对在线/增量分析支持不足；仅考虑消息的结构因果，未联合控制流与数据流；对极端并发情况的鲁棒性待进一步验证。

---

## 77. Lossless Compression via Chained Lightweight Neural Predictors with Information Inheritance

**arXiv ID:** 2604.15472 | [PDF](https://arxiv.org/pdf/2604.15472v1)

**作者:** Yuriy Kim `[一作]` (ITMO University), Evgeny Belyaev `[通讯]` (ITMO University)

**通讯引用:** 791 | [OpenAlex ID](https://openalex.org/A5054224670)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种链式轻量级神经预测器的无损压缩架构，能够根据输入数据的统计特性自适应选择所需的高阶单元，并通过信息继承提升预测精度。

**💡 创新点**

创新点包括：1）基于马尔可夫源建模的轻量级预测器搜索，极大减少模型权重；2）在预测器链中引入信息继承机制，使低阶预测的概率估计被高阶单元利用；3）自适应禁用高阶单元的算法，以平衡压缩效率和计算复杂度。

**🔧 技术方法**

主要技术：神经网络（MLP、CNN、GRU、Transformer）轻量化架构搜索；BPE自适应分词；算术编码；权重剪枝、向量量化和 LZMA2 压缩；半自适应训练与早停；信息继承线性加权。

**📊 数据集**

在七种多模态数据集上评估，包括文本（enwik8、enwik9、books）、图像（Image100、Image1.2GB）、音频（sound100、sound）、卫星遥感（Spitzer100、Spitzer）、基因组（AnCa、GaGa）以及随机字节流。

**📈 对比分析**

与经典压缩器（gzip、7z、zstd）及主流神经压缩器（PAC、TRACE）对比。压缩比接近 PAC，且在 RTX4060Ti GPU 上编码速度提升 1.2–6.3 倍，解码速度提升 2.8–12.3 倍，显示出显著的计算效率优势。

**⚠️ 局限性**

局限性在于：1）仍需在 GPU 上训练预测器，训练时间与模型复杂度相关；2）对极大数据集的高阶单元使用有限，可能无法捕捉极高阶依赖；3）量化压缩配置对最终压缩率影响较大，需要额外搜索；4）随机数据仅使用低阶单元，无法充分利用模型潜力。

---

## 78. TopFeaRe: Locating Critical State of Adversarial Resilience for Graphs Regarding Topology-Feature Entanglement

**arXiv ID:** 2604.15370 | [PDF](https://arxiv.org/pdf/2604.15370v1)

**作者:** Xinxin Fan `[一作]` (Chinese Academy of Sciences), Yunfeng Lu `[通讯]` (Beihang University)

**通讯引用:** 10867 | [OpenAlex ID](https://openalex.org/A5089011855)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于平衡点理论的图对抗防御框架 TopFeaRe，能够定位图结构在对抗扰动下的关键鲁棒状态并对图进行自适应清洗。

**💡 创新点**

创新点包括：①将对抗扰动映射为复杂动态系统的振荡；②构造二维拓扑‑特征纠缠函数以捕捉两者的耦合变化；③利用平衡点条件寻找并逼近图的渐近稳定平衡点，从而指导最优边删减。

**🔧 技术方法**

使用的方法包括：复杂动态系统（CDS）模型、平衡点（ASEP）理论、Heterogeneous Mean Field（HMF）简化、Michaelis–Menten 非线性映射、Jaccard 与余弦相似度融合的边筛选算法。

**📊 数据集**

实验数据集涵盖五个常用真实图：Cora、Cora_ML、Citeseer、Photo 与 Pubmed，分别用于评估对非目标攻击（Metattack、CE‑PGD、DICE）以及目标攻击（Nettack）的鲁棒性。

**📈 对比分析**

与 GCN‑SVD、GCN‑Jaccard、GCN、GAT、HANG、Mid‑GCN 等基线比较，TopFeaRe 在所有攻击与所有数据集上均实现了显著提升（最高可达 16%+ 的准确率提升），并在清洁图上也能进一步提升性能。

**⚠️ 局限性**

局限性主要在于：①需要手动选择并调参的 ASEP 映射函数（如 Michaelis–Menten）可能不适用于所有图结构；②计算复杂度与图大小及攻击率成正比，较大图需更高的运行开销；③当前评估聚焦于节点分类任务，未验证在其他下游任务或更大规模图上的通用性。

---

## 79. DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI

**arXiv ID:** 2604.15456 | [PDF](https://arxiv.org/pdf/2604.15456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 80. Analyzing the Presentation, Content, and Utilization of References in LLM-powered Conversational AI Systems

**arXiv ID:** 2604.15326 | [PDF](https://arxiv.org/pdf/2604.15326v1)

**作者:** Jianheng Ouyang `[一作]` (Hong Kong University of Science and Technology), Arpit Narechania `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 343 | [OpenAlex ID](https://openalex.org/A5078755914)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统评估九种主流LLM驱动对话式人工智能系统的引用（Reference）展示方式、质量、数量与一致性，并结合用户交互实验，探究用户对引用的使用与验证行为。

**💡 创新点**

创新点在于：①首次将CRAAP四维（Currency、Relevance、Authority、Accuracy、Purpose）框架应用于AI生成的引用并做量化评估；②提出并编码五维UI设计框架（父组件、布局模式、交互组件、数据类型、交互方式）；③发现引用质量与展示方式存在显著差异，且用户对引用验证的主动性普遍不足。

**🔧 技术方法**

技术手段包括：使用CRAAP评估模型对每条引用打分；Cronbach α计算不同迭代下引用数的一致性；对UI截图进行手工编码并统计；在用户实验中记录鼠标hover与click事件，并收集满意度问卷。

**📊 数据集**

数据集：从30个覆盖六类问题（事实、确认、假设、选择、因果、列表）与五个领域（体育、政治、地理、科学、文学）的查询中，分别向九个系统提交一次，收集到1,517条唯一引用；在用户研究中共使用12名本科生参与，每人使用九系统完成同一组查询。

**📈 对比分析**

对比方法：按系统统计平均引用数、CRAAP总分、Cronbach α以及用户hover/click率。结果显示ChatGPT平均生成9.5条引用，CRAAP总分15.48/20，α=0.961；Claude始终生成10条引用，α=1.0；其余系统在引用数与质量上均低于ChatGPT；用户交互率普遍低于25%，ChatGPT最高。

**⚠️ 局限性**

局限性：研究基于2025年7月的快照，模型迭代可能导致引用行为变化；CRAAP原为人类作者来源评估，未完全适用于AI生成引用；用户行为仅通过鼠标记录，缺乏眼动追踪等更精确指标；样本规模较小，需进一步大规模验证与更细粒度的交互设计评估。

---

## 81. NEFFY 2.0: A Breathing Companion Robot: User-Centered Design and Findings from a Study with Ukrainian Refugees

**arXiv ID:** 2604.15325 | [PDF](https://arxiv.org/pdf/2604.15325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 82. Modeling of ASD/TD Children's Behaviors in Interaction with a Virtual Social Robot During a Music Education Program Using Deep Neural Networks

**arXiv ID:** 2604.15314 | [PDF](https://arxiv.org/pdf/2604.15314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 83. Exploring LLM-based Verilog Code Generation with Data-Efficient Fine-Tuning and Testbench Automation

**arXiv ID:** 2604.15388 | [PDF](https://arxiv.org/pdf/2604.15388v1)

**作者:** Mu-Chi Chen `[一作]` (Academia Sinica), Shih-Hao Hung `[通讯]` (National Taiwan University)

**通讯引用:** 1137 | [OpenAlex ID](https://openalex.org/A5020028710)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套多代理LLM工作流，用于自动生成Verilog代码和相应的测试平台，并利用这些数据进行高效的数据化微调

**💡 创新点**

创新点在于结合多代理测试平台生成与基准Verilog代码生成模型的分离，减少所需训练数据并提升验证效率，同时实现自动化的测试平台生成

**🔧 技术方法**

采用大型语言模型（如DeepSeek‑R1、Qwen‑Coder‑7B‑Instruct）进行监督微调，并用vLLM部署LLM，利用多代理框架完成规范检查、测试平台生成及Verilog验证

**📊 数据集**

使用过滤后的PyraNet（Pyra‑tb）数据集（6.7k样本）及DeepCircuitX，结合生成的推理轨迹和自动化测试平台，构建训练集

**📈 对比分析**

在Refined VerilogEval v2基准上与CodeV‑R1等SOTA模型比较，MA‑tb‑7B模型在38k训练样本下达成68% pass@1，近似或超过部分SOTA模型，证明少量数据即可取得竞争性能

**⚠️ 局限性**

局限性包括对LLM推理轨迹的依赖、生成测试平台仍需多次重试导致API成本、以及在极其复杂或异构硬件场景下可能仍难以覆盖所有验证需求

---

## 84. Robust Transmission Design for RIS-Assisted High-Speed Train Communication Coverage Enhancement With Imperfect Cascaded Channels

**arXiv ID:** 2604.15387 | [PDF](https://arxiv.org/pdf/2604.15387v1)

**作者:** Changzhu Liu `[一作]` (Beijing Jiaotong University), Zhangdui Zhong `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 18815 | [OpenAlex ID](https://openalex.org/A5100350955)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在高速列车移动场景下，基于可重构智能表面（RIS）的多用户MISO系统，在存在CSI不完美（包括CBRUB与DCSIB误差）的情况下，求解功率最小化与可靠率（OP）约束的波束及RIS相位优化问题。

**💡 创新点**

创新点在于：①提出了对CBRUB误差和DCSIB误差的两种场景（PCU、FCU）进行统一建模；②在BCSIE与SCSIE两种误差模型下，分别采用S‑Procedure与Bernstein不等式将无穷约束转化为LMIs和凸约束；③结合Penalty CCP与SDR实现单次迭代中的可解性；④通过仿真揭示CBRUB误差对RIS收益的显著影响，并证明SCSIE模型在功率、收敛性与复杂度方面优于BCSIE模型。

**🔧 技术方法**

所用技术包括：S‑Procedure、Second‑Order Cone Programming（SOCP）、Penalty Convex‑Concave Procedure（CCP）、Semidefinite Relaxation（SDR）、Bernstein‑Type Inequality（BTI）、交替优化（AO）等。

**📊 数据集**

实验采用仿真参数：M,N=3或6，K=2/3，速度360 km/h，载频200 MHz，Rician K=3 dB，信道采用Rician随机衰落模型，误差量化为ω_H、ω_D参数。

**📈 对比分析**

比较方法：与理想相位和离散相位方案对比；评估指标包括功率收敛曲线、平均CPU时间、误差概率和比特率。结果表明：SCSIE模型实现更低功率、更快收敛；理想相位方案功率最低；离散相位方案相对较差。CBRUB误差越小，RIS带来的功率下降越明显；误差增大时RIS功率反而上升。

**⚠️ 局限性**

局限性：仅针对MISO多用户模型，未考虑反馈延迟、相位噪声和RIS硬件非理想；算法在大规模天线/元件时仍存在高复杂度；未研究多频/时变非平稳通道环境。

---

## 85. (1D) Ordered Tokens Enable Efficient Test-Time Search

**arXiv ID:** 2604.15453 | [PDF](https://arxiv.org/pdf/2604.15453v1)

**作者:** Zhitong Gao `[一作]` (Swiss Federal Institute of Technology Lausanne (EPFL)), Oğuzhan Fatih Kar `[通讯]` (Swiss Federal Institute of Technology Lausanne (EPFL))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了1D粗细分层（coarse‑to‑fine）tokenizer对自回归图像生成模型在推理时搜索（test‑time scaling）的影响，并展示了其在零样本控制和无模型训练的生成方面的优势。

**💡 创新点**

创新点在于将token结构视为决定搜索有效性的关键因素，证明1D有序token提供可验证的中间读数，显著提升搜索效率；同时提出无模型训练的直接搜索生成方法和对不同搜索策略、验证器、先验的系统化评估。

**🔧 技术方法**

技术手段包括1D有序tokenizer（如FlexTok、Semanticist、Infinity）、自回归模型、三种搜索算法（Best‑of‑N、Beam Search、Lookahead Search）、多种验证器（CLIP、ImageReward、DreamSim等）以及对无模型搜索的实现。

**📊 数据集**

使用的数据集主要有COCO Karpathy验证集、GenEval、DreamBench++，并在与2D网格tokenizer（如Janus）和其他有序模型的对比实验中验证效果。

**📈 对比分析**

与2D网格tokenizer以及不同搜索策略对比时，1D有序token在Beam Search下的性能提升显著，能够在相同推理预算下获得更高的CLIP/多任务评估分数；在零样本控制任务中，1D token化显著提升概念保持能力；无模型搜索也能得到可接受的单/双物体生成结果。

**⚠️ 局限性**

局限包括：搜索算法未针对有序结构专门设计，仍为通用方法；验证器的鲁棒性有限，容易被搜索“攻击”；解码过程需要多步流式解码，计算开销高；实验仅覆盖少数模型和任务，未验证跨模态或更大规模的普适性。

---

## 86. Anthropomorphism and Trust in Human-Large Language Model interactions

**arXiv ID:** 2604.15316 | [PDF](https://arxiv.org/pdf/2604.15316v1)

**作者:** Akila Kadambi `[一作]` (University of Southern California), Lisa Aziz-Zadeh `[通讯]` (University of Southern California)

**通讯引用:** 5504 | [OpenAlex ID](https://openalex.org/A5010804318)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对115名参与者与LLM的2000+次对话，系统性测量了温暖、能力、认知与情感共情等维度对人类对LLM的拟人化、信任及其他认知和情感结果的影响。

**💡 创新点**

创新点在于将社会心理学的温暖-能力维度与认知/情感共情维度结合，量化它们对LLM拟人化与信任的独立与交互效应，并检验话题类型对结果的调节作用。

**🔧 技术方法**

采用了实验室设计，利用Gemini 2.0模型并通过PythonAnywhere界面，在UI中对LLM的回答进行多级提示，以实现三阶的温暖/能力和认知/情感共情梯度操控。

**📊 数据集**

数据集来自115名大学生在加州大学洛杉矶分校学生池的参与，通过随机分配四个话题（生物、美国历史、健康生活、关系建议）以及18轮交互，收集了约2000条人机交互记录及问卷评估。

**📈 对比分析**

方法上使用层级线性混合模型与似然比检验比较三种模型（空模型、特质模型、完整模型），结果显示温暖和认知共情显著提升拟人化，能力显著提升信任、可用性与减少挫败感，且相对较大的效应量（d≈0.4–0.7）。

**⚠️ 局限性**

局限性包括：所有受试者均已知自己与AI对话，可能削弱拟人化效果；样本以美国高校学生为主，缺乏文化多样性；实验使用的提示策略可能无法完全复制真实LLM使用场景。

---

## 87. Interpupillary Distance Constraints in Pediatric VR: Implications for Psychology and Psychotherapy

**arXiv ID:** 2604.15328 | [PDF](https://arxiv.org/pdf/2604.15328v1)

**作者:** Grzegorz Pochwatko `[一作]` (Polish Academy of Sciences), Grzegorz Pochwatko `[通讯]` (Polish Academy of Sciences)

**通讯引用:** 761 | [OpenAlex ID](https://openalex.org/A5086929941)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

分析儿童interpupillary distance（IPD）与消费型VR头显（以Meta Quest 3为例）匹配问题，提出将硬件兼容性视为心理学研究与临床实践中的重要方法变量；

**💡 创新点**

将IPD匹配问题作为潜在偏倚源并系统量化不同年龄段儿童对Quest 3的适配比例，强调设备选择应基于心理学方法论而非单纯技术规格；

**🔧 技术方法**

利用已有儿童IPD分布数据、头显技术规格和正态混合模型估计适配率；

**📊 数据集**

MacLachlan & Howland 2002 年儿童IPD标准数据，以及Meta Quest 3 官方最小/最优IPD范围；

**📈 对比分析**

通过比较各年龄组IPD百分位与Quest 3的IPD阈值，绘制适配率曲线；未进行实验验证，性能以理论估计为主；

**⚠️ 局限性**

缺乏直接测量数据，仅关注IPD而未充分考虑其他光学与人体工学因素，混合模型假设可能不完全符合真实分布；

---

## 88. Transfer Learning from Foundational Optimization Embeddings to Unsupervised SAT Representations

**arXiv ID:** 2604.15448 | [PDF](https://arxiv.org/pdf/2604.15448v1)

**作者:** Koyena Pal `[一作]` (Fidelity Investments), Serdar Kadioglu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将 MIP 的无监督预训练框架 Forge 迁移到 SAT 问题，探究了跨领域无监督嵌入的有效性，利用 SAT‑to‑MIP 编码直接使用预训练模型，并在 SAT 实例上进行无监督聚类。

**💡 创新点**

创新点在于首次展示了基于 MIP 的基础嵌入能够跨域迁移到 SAT，提出三种迁移变体（Forge‑Mip、Forge‑Mip‑Sat、Forge‑Sat），并证明其在无监督设置下能够捕捉 SAT 结构与可满足性信息。

**🔧 技术方法**

使用技术包括向量量化图自动编码器（Forge）、GraphSAGE 图神经网络、SAT‑to‑MIP 编码、无监督聚类可视化（PaCMAP）以及 NMI/Purity 等聚类质量指标。

**📊 数据集**

采用的数据集为 G4SATBench（随机、伪工业、组合 SAT 题目，Hard 难度层级）进行评估，并在预训练阶段使用 MIPLIB 的大型 MIP 集合。

**📈 对比分析**

通过与静态 SAT 特征基线（Static‑Sat）和两种预训练 MIP 模型进行对比，利用 NMI 和 Purity 评估聚类效果；Forge‑Sat 在 NMI 上达到 0.79、Purity 0.66，优于其他变体和基线，证明了跨域迁移的有效性。

**⚠️ 局限性**

局限性包括仅在无监督聚类任务上验证，SAT 节点特征仍为简单统计量，预训练语料有限且未进行任务特定的超参数调优；后续工作需探索监督下的下游任务、特征 ablation 与更大规模的 SAT 语料。

---

## 89. Towards Measuring Interactive Visualization Abilities: Connecting With Existing Literacies and Assessments

**arXiv ID:** 2604.15320 | [PDF](https://arxiv.org/pdf/2604.15320v1)

**作者:** Gabriela Molina León `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8278 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套评估交互可视化素养（IVL）的理论框架和多种评估方法。

**💡 创新点**

创新点在于将交互、感知、认知等多重素养融入单一模型，并结合跨学科已有评估工具。

**🔧 技术方法**

主要技术包括访谈观察、项目反应理论（IRT）条目设计、现有评估工具融合、问卷自评、交互日志分析以及生物特征和LLM自动编码。

**📊 数据集**

未使用特定数据集，而是参考了已有的评估问卷（如TIMSS、VLAT、CALVI）和文献中的案例。

**📈 对比分析**

由于是定位性论文，未开展实验比较，建议后续通过大规模问卷或实验室研究验证各方法的效度与可操作性。

**⚠️ 局限性**

局限性包括缺乏经验验证、评估方法难以同时满足高层思考与技术简单性、不同方法难以统一整合、需针对不同受众与情境选择合适技术。

---

## 90. Radical Gender Neutrality: Agender Euphoria in Gaming and Play Experiences

**arXiv ID:** 2604.15337 | [PDF](https://arxiv.org/pdf/2604.15337v1)

**作者:** Katie Seaborn `[一作]` (Institute of Science Tokyo), Phoebe O. Toups Dugas `[通讯]` (Exertion Games Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了游戏中agender（无性别）玩家的“agender euphoria”（无性别欣快感）体验，采用批判性事件技术（CIT）收集 142 名自认或渴望 agender 体验者的访谈数据，并辅以量化问卷（I‑PANAS‑SF、HEMA‑R、PXI、LTMP）进行混合方法分析，提出了基于“基线”“设计”“游戏”三维的体验棱镜与设计准则。

**💡 创新点**

首次系统性探讨 agender euphoria 作为独立体验范畴，并将其与传统性别 euphoria 区分；通过构建“极性性别中立”框架，提供了一套面向游戏设计者的“多样性、无性别、可切换、最小化表征、沙盒式自由”准则；此外，将 agender 体验与正向心理学（hedonia、eudaimonia、长期意义）关联，扩展了游戏体验研究视角。

**🔧 技术方法**

技术方法：批判性事件技术（CIT）与反思性主题分析（RTA）相结合的定性分析；量化测量采用国际正负情绪量表 I‑PANAS‑SF、Hedonic & Eudaimonic Motivations for Activities–Revised（HEMA‑R）、Player Experience Inventory（PXI）和 Long‑Term Meaning Potential（LTMP）；使用 Google Forms 进行在线调查；数据分析使用 Google Sheets 与 R 进行描述性统计、Shapiro‑Wilk 正态性检验、Pearson 相关分析。

**📊 数据集**

数据集：142 名通过 Prolific 招募的英语使用者，其中大多数自认 agender 或 non‑binary（n=76）；调查涵盖游戏类型、性别身份、游戏体验与情感评分等；所有数据均为自我报告的定性访谈记录与量化问卷答案。

**📈 对比分析**

未与传统方法做直接性能对比；通过描述性统计和相关性检验呈现 agender euphoria 与正向情绪、幸福感和游戏体验的显著正相关（如正向情绪均值 3.5/5，eudaimonia 与正向情绪 r=0.316）；研究表明受访者在游戏体验中获得高水平的心理愉悦、意义感和长期价值，表明其设计准则具有积极效用。

**⚠️ 局限性**

局限性：样本以英语、白人、受教育程度较高的 WEIRD 群体为主，缺乏跨文化与多元语言验证；使用 CIT 依赖事后回忆，易受记忆偏差；量化测量未完全匹配 agender euphoria 的独特情绪词汇，建议使用即将发布的 Gender Euphoria Scale；未细分不同 agender 变体，可能掩盖个体差异；未进行实时或实验室观察，难以捕捉即时交互细节。

---

## 91. DEMUX: Boundary-Aware Multi-Scale Traffic Demixing for Multi-Tab Website Fingerprinting

**arXiv ID:** 2604.15677 | [PDF](https://arxiv.org/pdf/2604.15677v1)

**作者:** Yali Yuan `[一作]` (Southeast University), Guang Cheng `[通讯]` (Southeast University)

**通讯引用:** 470264 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种专门针对多标签网页指纹识别（多标签、混合流）问题的 DEMUX 框架，并通过三大模块实现对混合流的分段保留、尺度多样化局部特征提取与全局相对位置关联。

**💡 创新点**

创新点包括：
- 将窗口划分改为重叠分段以保持跨 burst 边界信息；
- 采用多尺度并行 CNN 以同时捕获细粒度 burst 模式和粗粒度周期结构；
- 在 Transformer 编码器中使用 Rotary Positional Embedding（RoPE）实现相对位置编码，解决传统绝对编码在多标签流中的失效；
- 设计 Boundary Preserving Aggregation Module 作为可插拔预处理器，能显著提升现有模型性能。

**🔧 技术方法**

使用的技术主要有：
- 重叠窗口分段与多级特征聚合（packet‑level + burst‑level）；
- 多尺度并行 1D 卷积网络（kernel 3/5/7）和残差块；
- 两阶段 Transformer 编码器，第一阶段 RoPE 编码，第二阶段普通 Transformer；
- 点wise 卷积融合、多头自注意力、FFN、LayerNorm 等深度学习常见组件；
- 采用 AdamW + Cosine 学习率、权重衰减等训练技巧。

**📊 数据集**

使用的数据集包括：
- ARES 基准数据（closed‑world 2‑5 tab、open‑world 2‑5 tab）
- 通过合成得到的 2‑tab 防御数据（WTF‑PAD、Front、TrafficSliver）
- 训练时混合 2‑5 tab 的动态标签集合，用于评估动态 tab 与跨配置泛化。

**📈 对比分析**

与九种主流基线（DF、Var‑CNN、Tik‑Tok、RF、NetCLR、BAPM、TMWF、ARES’23、ARES’25）进行对比。 DEMUX 在闭合域 5‑tab 上 P@5 0.943、MAP@5 0.961，分别比最强基线高 9.2% 与 6.2%；在开放域、各防御、动态 tab 和跨配置实验中始终保持最高或接近最高的 AUC、P@K、MAP@K，且性能衰减最慢。

**⚠️ 局限性**

局限性包括：
- 仅在 Tor 被动窃听场景下验证，未考虑主动探测或更高级的隐蔽网络；
- 对于极端混合或更长会话的泛化尚未充分验证；
- 模型结构相对复杂，推理时内存/计算成本高于传统单标签 CNN；
- 需要对 burst 统计进行手工设计，若流量特征变化剧烈可能需要重新调整参数。

---

## 92. Mapping High-Performance Regions in Battery Scheduling across Data Uncertainty, Battery Design, and Planning Horizons

**arXiv ID:** 2604.15360 | [PDF](https://arxiv.org/pdf/2604.15360v1)

**作者:** Jaime de Miguel Rodriguez `[一作]` (Enefit), Kaarel Oja `[通讯]` (Enefit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多阶段MPC电池储能调度中规划窗口长度对收益与计算成本的影响，并系统评估了不同电池特性、数据特征与预测不确定性下的最优与有效规划窗口。

**💡 创新点**

提出了基于参数化合成数据与可解释的“最优规划窗口”与“有效规划窗口”概念，并通过全景实验给出不同电池与不确定性水平下的最优窗口表，为动态窗口选择提供实证依据。

**🔧 技术方法**

采用线性/混合整数规划的滚动窗口MPC、傅里叶+SARIMA生成合成时序、基于误差传播的线性不确定性模型以及网格搜索确定最优窗口等技术。

**📊 数据集**

使用三类合成数据集：纯正弦波、日均价（day‑ahead）与mFRR市场数据的Fourier+SARIMA合成版本，并在每种数据下生成多级不确定性预测。

**📈 对比分析**

通过比较不同电池C‑rate、预测不确定因子、规划窗口长度的收益曲线，并绘制3D针刺图与收益-窗口曲线，发现低不确定性时快电池最优窗口仅4‑8h，高不确定性时窗口显著收缩，整体收益与最优窗口高度相关。

**⚠️ 局限性**

仅采用线性模型、无终端SOC约束、固定窗口全周期、合成数据仅覆盖两周且不考虑多信号交互，且未验证在真实市场中的动态窗口自适应效果。

---

## 93. Weak-to-Strong Knowledge Distillation Accelerates Visual Learning

**arXiv ID:** 2604.15451 | [PDF](https://arxiv.org/pdf/2604.15451v1)

**作者:** Baiang Li `[一作]` (Princeton University), Felix Heide `[通讯]` (Princeton University)

**通讯引用:** 6662 | [OpenAlex ID](https://openalex.org/A5059313827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了弱到强的知识蒸馏方案，利用冻结的弱教师在训练早期加速强学生的学习，并在学生超过教师性能后停止蒸馏。

**💡 创新点**

首次将蒸馏视为训练加速工具，设计了适度弱化教师与早期蒸馏加权、超越停止等简单可插拔的加速策略。

**🔧 技术方法**

采用前向KL蒸馏、温度衰减、warmup-hold-decay加权调度、基于验证指标的超越停止以及多任务（分类、检测、生成）的统一框架。

**📊 数据集**

在ImageNet、CIFAR-10/100、COCO检测集和CIFAR-10扩散生成任务上验证。

**📈 对比分析**

与基线相同超参训练比较 first@τ 指标，结果显示分类可达4.8×、检测1.7×、生成2.5×的训练步/周期加速，最终精度与基线相当。

**⚠️ 局限性**

需预先拥有适度弱化的教师模型，教师-学生匹配不佳时加速效果显著下降，且方法未覆盖教师训练成本。

---

## 94. Struggle as Flow: Challenge, Design, and Experience in Soulslike Games

**arXiv ID:** 2604.15318 | [PDF](https://arxiv.org/pdf/2604.15318v1)

**作者:** Zhehao Sun `[一作]` (University of British Columbia), Megan Smith `[通讯]` (University of British Columbia)

**通讯引用:** 14374 | [OpenAlex ID](https://openalex.org/A5058772970)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三款 Soulslike 游戏的 Steam 评论进行定性主题分析，提出并验证“Resilient Flow”概念。

**💡 创新点**

将 Csikszentmihalyi 流体验理论与 Juul 的游戏学框架结合，提出将挫败视为激发专注的“Resilient Flow”，并从玩家语言中识别出节奏同步、公平性与正念压力三维结构。

**🔧 技术方法**

采用文本挖掘与层级编码的质性分析方法（open、axial、selective coding），并将词频与语义类别映射至流体验模型。

**📊 数据集**

收集了 600 条“最有帮助”标记的 Elden Ring、Sekiro: Shadows Die Twice 与 Dark Souls III 玩家评论。

**📈 对比分析**

通过比较三款游戏的词汇转化、失败归因与情感表达，揭示不同设计对 Resilient Flow 的影响；实验未给出数值指标，但通过多维度质性结果展示了各游戏在节奏、规则公平和正念层面上的差异。

**⚠️ 局限性**

样本仅来自已写过积极评论的高经验玩家，存在存活偏差；缺乏生理数据验证流体验；仅聚焦 Steam 社区文本，未考察流失玩家。

---

## 95. Beyond Passive Viewing: A Pilot Study of a Hybrid Learning Platform Augmenting Video Lectures with Conversational AI

**arXiv ID:** 2604.15334 | [PDF](https://arxiv.org/pdf/2604.15334v1)

**作者:** Mohammed Abraar `[一作]` (Vizuara AI Labs), Sreedath Panat `[通讯]` (Vizuara AI Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并评估了将实时对话式 AI 导师嵌入传统视频课程的混合学习平台，进行顺序 within‑subjects 对比实验。

**💡 创新点**

创新点在于将时间对齐的检索增强生成（Timeline‑Augmented RAG）与 LLM 结合，实时为视频内容生成精准、可追溯的提问与反馈，从而显著提升学习效果与参与度。

**🔧 技术方法**

采用 Gemini 1.5 Flash 大语言模型、多模态 Live API、时间标注检索增强生成框架、Chain‑of‑Thought 误差检测等技术实现对话式辅导。

**📊 数据集**

使用 60 分钟 Tokenization 课程视频及其拆分段落的语义嵌入向量，实验样本为 58 名全球学生。

**📈 对比分析**

通过顺序 within‑subjects 设计，配对 t‑检验比较传统视频与 AI 辅助条件；即时测验平均提升 8.3 分（d = 1.505），两周延迟测验维持 90.5 分；参与时长提升 71.1%，聊天互动约 40% 的时间。

**⚠️ 局限性**

局限包括固定顺序导致的练习效应混淆、仅研究单一主题与特定人群、两周保持测试周期短、缺乏对话质性分析等。

---

## 96. Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility

**arXiv ID:** 2604.15579 | [PDF](https://arxiv.org/pdf/2604.15579v1)

**作者:** Yining Hong `[一作]` (Carnegie Mellon University), Christian Kästner `[通讯]` (Carnegie Mellon University)

**通讯引用:** 15505 | [OpenAlex ID](https://openalex.org/A5067467896)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文系统评估并实现了符号型安全护栏（symbolic guardrails），以保证AI工具使用代理在高风险业务场景下的安全性与可靠性，并在三大基准（τ^2-Bench、CAR-bench、MedAgentBench）上验证其有效性。

**💡 创新点**

创新点在于：①对80个现有代理安全基准进行系统审计，发现大多数缺乏具体可执行政策；②构建并评估六类符号护栏（API校验、架构约束、信息流控制、时序逻辑、用户确认、响应模板），证明74%可执行要求可通过低成本符号方法实现；③在实际基准中证明符号护栏既能消除约80%安全违规，又不会降低甚至提升代理任务完成率。

**🔧 技术方法**

技术手段包括：符号化策略实现（API校验、模式约束、信息流控制、时序逻辑、用户确认、响应模板），基于MCP协议的工具接口封装，LLM模型（GPT‑4o/GPT‑5）与LLM‑judge评估，自动化需求生成与STPA风险分析，用于生成合成政策。

**📊 数据集**

使用的数据集：
- τ^2‑Bench（50航空客服任务），
- CAR‑Bench（100车载语音助手任务），
- MedAgentBench（300 EMR 辅助任务）以及对MedAgentBench自动生成的对抗任务（50个）。

**📈 对比分析**

比较方法：在每个基准下，基准工具 vs. 添加符号护栏后工具，记录安全违规率、未知/安全/无效率以及任务通过率（Pass^1 或 Success Rate）。结果显示：
- 安全违规率从20–78%降至0%；
- 通过率在所有基准上均提升，提升显著（p<0.05），且未出现负面效应。

**⚠️ 局限性**

局限性包括：
- 仅分析了两类具体政策（τ^2‑Bench、CAR‑Bench）及合成的MedAgentBench，未覆盖所有行业场景；
- 依赖LLM模型与LLM‑judge评估，仍受模型能力与prompt设计影响；
- 仍有四类需求（如语气、消除幻觉、程序遵循、常识推理）无法符号化，需要神经护栏；
- 对抗性实验样本有限，可能低估复杂攻击情况。

---

## 97. Automating Crash Diagram Generation Using Vision-Language Models: A Case Study on Multi-Lane Roundabouts

**arXiv ID:** 2604.15332 | [PDF](https://arxiv.org/pdf/2604.15332v1)

**作者:** Xiao Lu `[一作]` (University of Georgia), Jidong J. Yang `[通讯]` (University of Georgia)

**通讯引用:** 1432 | [OpenAlex ID](https://openalex.org/A5041740157)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了使用Vision‑Language模型自动从警察事故报告生成多车道环形交叉口的碰撞图，并评估其在三款主流模型上的表现。

**💡 创新点**

结合结构化三段式提示和十项定量评价指标，首次系统化评估VLM在专业交通安全可视化任务中的可行性和局限。

**🔧 技术方法**

采用GPT‑4o、Gemini‑1.5‑Flash和Janus‑4o三种多模态模型，并设计基于文本+图像的提示工程与批量推理流水线。

**📊 数据集**

选取了79份纽约州马尔塔镇两车道环形交叉口的完整警察事故报告（Form MV‑104A）作为实验数据。

**📈 对比分析**

通过人工专家对十个二元评价指标进行打分，GPT‑4o平均得分6.29/10，Gemini‑1.5‑Flash 5.28/10，Janus‑4o 3.64/10，显示GPT‑4o在语义提取和空间推理上明显优于其它两者。

**⚠️ 局限性**

评价采用0/1二元分数缺乏细粒度；数据局限单一交叉口；结构化提示对报表格式变异鲁棒性差；模型仍存在碰撞点定位和损伤可视化误差。

---

## 98. Understanding Inference-Time Token Allocation and Coverage Limits in Agentic Hardware Verification

**arXiv ID:** 2604.15657 | [PDF](https://arxiv.org/pdf/2604.15657v1)

**作者:** Vihaan Patel `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5045858420)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了两层次的代理框架 CovAgent，用于自动化硬件验证中的覆盖闭合。

**💡 创新点**

创新点在于系统性地对代理运行时 token 分配和覆盖缺口进行分类，并证明领域专用化可显著降低 token 消耗。

**🔧 技术方法**

采用了 OpenAI Codex 与 LangGraph 构建代理，结合 ReAct 模式、结构化工具接口、覆盖反馈与 token 计量。

**📊 数据集**

使用了 19 个从 100 行到 8500 行不等的单元级设计作为实验数据集。

**📈 对比分析**

与通用基线对比，域专用增强代理在保持 95–99% 覆盖率的同时，令 token 数量减少 4–13 倍，收敛速度提升 2–4 倍。

**⚠️ 局限性**

局限在于对大规模子系统/顶层验证的适用性不足、对复杂协议模型的生成能力有限，以及小模型在高复杂度设计上的覆盖下降。

---

## 99. LogJack: Indirect Prompt Injection Through Cloud Logs Against LLM Debugging Agents

**arXiv ID:** 2604.15368 | [PDF](https://arxiv.org/pdf/2604.15368v1)

**作者:** Harsh Shah `[一作]` `[通讯]` (Independent Researcher), Harsh Shah (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并演示云日志作为间接提示注入的攻击面，开发 LogJack 基准来测试大型语言模型在此情境下的行为

**💡 创新点**

提供了第一套真实云日志注入基准（42个载荷），系统化评估多家模型及云供应商防护机制，揭示日志格式对输入检测的遮蔽效果

**🔧 技术方法**

基于大语言模型（Llama、Claude、Gemini、GPT-4o 等）和云原生工具（AWS SDK、Shell 命令）进行对话式实验，利用正则和 95% 置信区间统计指标

**📊 数据集**

LogJack 基准：42 个注入载荷，分为 5 类云日志（CloudWatch、SSM、CI/CD、CloudTrail、Lambda），每类含多种难度和攻击目标（RCE、权限提升等）

**📈 对比分析**

在“主动”条件下，最高 86.2% 的模型会按注入命令执行，最少 0%，加入“被动”指令后多数模型成功率降至 0%，仅 Llama 30%；RCE 成功率达 75% 以上；与云厂商防护对比显示所有主流输入侧防护均失效

**⚠️ 局限性**

实验仅使用 5 次随机试验、固定温度 0.7，基准载荷为手工构造，缺乏对真实生产代理的评估；正则分类可能误判；未测试更严格工具约束或多轮人机交互的真实场景

---

## 100. EasyRider: Mitigating Power Transients in Datacenter-Scale Training Workloads

**arXiv ID:** 2604.15522 | [PDF](https://arxiv.org/pdf/2604.15522v1)

**作者:** Dillon Jensen `[一作]` (Stanford University), Phil Levis `[通讯]` (Stanford University)

**通讯引用:** 1676 | [OpenAlex ID](https://openalex.org/A5023032922)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种 rack‑level 电源架构，利用被动 LC 滤波器、DC‑DC 稳压模块和局部电池，实时平滑 AI 训练中的功率波动，使机架功率随时间平稳变化

**💡 创新点**

首次在硬件层面实现了快速、无软件依赖的功率平滑，并在软件层面加入基于目标 SoC 的优化控制，完成了无需修改训练框架即可满足电网升降速率和频谱约束的目标

**🔧 技术方法**

采用被动 LC 滤波、DC‑DC 调节、锂铁磷电池、FPGA/单片机控制、QP 优化求解器以及 Modbus 通信等技术

**📊 数据集**

使用公开的 AI 训练功率轨迹（Choukse 等）和自研的 125M GPT‑style LLM 在 2‑GPU Titan‑X 机架上的实际功率数据

**📈 对比分析**

与软件“燃烧”方式（额外 GEMM 内核）对比，EasyRider 在满足升降速率和频谱约束的同时，能耗比软件方案低 19%，且无启动延迟和工作负载改动

**⚠️ 局限性**

局限包括需额外电池与滤波硬件、对 400 V DC 机架的依赖、成本与热设计挑战，以及在极端功率规模下的扩展性与长期寿命管理

---

## 101. On Word Representations and Embeddings in Complex Matrices

**arXiv ID:** 2604.15386 | [PDF](https://arxiv.org/pdf/2604.15386v1)

**作者:** Paul C. Bell `[一作]` (Liverpool John Moores University), Pavel Semukhin `[通讯]` (Liverpool John Moores University)

**通讯引用:** 236 | [OpenAlex ID](https://openalex.org/A5050813521)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究单词结构向低维复数矩阵半群的嵌入，提出了构造欧氏Bianchi群单词表示的新方法，并系统总结了不同字母表对的嵌入存在与否的边界。

**💡 创新点**

①首次给出针对欧氏Bianchi群的多项式时间单词表示算法；②在二维复矩阵空间中确立了多对单词半群嵌入的完整命题；③揭示了非欧氏Bianchi群与可解/幺半群嵌入的限制，进一步界定了“可嵌入-不可嵌入”的分界。

**🔧 技术方法**

采用组合词理论、欧几里得域的除法算法、线性代数中的矩阵对角化与上三角化技术，以及Bianchi群的几何群体结构；同时利用数论中的格子与单位群性质来推导嵌入不存在性。

**📊 数据集**

无；该工作为纯理论研究，未使用任何实验数据集。

**📈 对比分析**

无实验比较；论文通过严谨的数学证明与命题表格阐述了不同嵌入方案的可行性与限制，无性能指标。

**⚠️ 局限性**

限制主要在于：①缺乏通用的可冲突且单子化的重写系统，导致无法直接构造有限的“花瓣图”来判定身份问题；②仅聚焦于二元/一元字母表的两重乘积，尚未推广至多重乘积或更大维度矩阵；③在非欧氏Bianchi群上仍缺乏有效的单词表示与决策算法。

---

## 102. Foundation Models in Robotics: A Comprehensive Review of Methods, Models, Datasets, Challenges and Future Research Directions

**arXiv ID:** 2604.15395 | [PDF](https://arxiv.org/pdf/2604.15395v1)

**作者:** Aggelos Psiris `[一作]` (Harokopio University of Athens), Arash Ajoudani adn Georgios Th. Papadopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基础模型（Foundation Models）在机器人领域的研究进行系统、全面的综述，梳理了过去五年内的研究进展、关键技术、数据集、应用场景以及面临的挑战，并对435篇相关论文进行细粒度、多维度的分类与比较。

**💡 创新点**

创新点：①提出了从“自然语言处理+计算机视觉”到“多模态动作执行”的五个研究阶段；②设计了六项评估维度（FM类型、神经网络架构、学习范式、学习阶段、机器人任务、应用领域）构成的细粒度分类体系；③系统整理并公开了与机器人任务相关的公开数据集和基准；④对比分析了不同FM类型在各维度的优势与局限，提出了可操作的研究路线图；⑤为后续工作提供了可复制的文献检索与筛选方法。

**🔧 技术方法**

使用的技术主要是：①基于六大数据库（IEEE Xplore、Scopus、Web of Science、Google Scholar、DBLP、arXiv）的关键词检索与布尔查询；②多轮迭代筛选（去重、英文限定、全文可获取、主题匹配）获得435篇论文；③对每篇论文进行手工标注，提取FM类型、NN架构、学习方法、学习阶段、任务与应用领域等信息；④统计与可视化（条形图、热力图）呈现文献分布与发展趋势；⑤构建了细粒度的对比表（表格、图表）并给出关键洞见。

**📊 数据集**

使用的数据集与基准：涵盖视觉语言对齐数据集（COCO‑Captions、Visual Genome、WebVision）、机器人模拟与真实数据集（Meta-World、Dexterous Manipulation Dataset、REAL‑B1、REAL‑B2、Robothor、Gibson、AI2‑THOR、iCubWorld、Matterport3D、CARLA、DeepMind Control Suite、OpenAI Gym、RLBench、Gibson-2、RoboSuite、VIM、RoboCup、NORB、SPL、Habitat、MAVIC、A-PEM、PandaSet、T-1等多模态、跨任务、跨机器人平台的公开数据集。

**📈 对比分析**

对比方法：将同一维度下不同FM/架构/范式/任务的代表性论文进行聚类、对照与性能评估（如成功率、规划时长、推理精度、样本效率、部署延迟等），并通过表格、图表展示差异与趋势；在性能层面，综述汇总了已有实验结果，指出LLM在长程规划与语义理解上的优势，VFM在感知与定位上的鲁棒性，VLM在开放词汇识别与语义对齐中的表现，VLA在端到端控制与跨任务迁移中的突出效果；同时对比了不同学习范式（预训练、强化学习、模仿学习、上下文学习等）在样本效率与泛化能力上的差异。

**⚠️ 局限性**

局限性：①综述时间窗口限于2020‑2026年，可能遗漏部分较早或最新的工作；②对多模态数据集与任务的覆盖仍不完整，尤其是低资源语言与多传感器场景；③对模型性能的比较主要依赖原论文报告，缺乏统一基准评估；④由于研究方向快速演进，部分前沿方法（如世界模型、扩散策略、离线强化学习）在综述中未能完全细化；⑤对安全性、解释性、能源消耗等实际部署问题的定量评估仍不足。

---

## 103. Learning Behaviorally Grounded Item Embeddings via Personalized Temporal Contexts

**arXiv ID:** 2604.15581 | [PDF](https://arxiv.org/pdf/2604.15581v1)

**作者:** Rafael T. Sereicikas `[一作]` (Federal University of São Carlos), Tiago A. Almeida `[通讯]` (Federal University of São Carlos)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了TAI2Vec模型，利用用户自适应的时间上下文对物品嵌入进行训练，以捕捉短期与长期偏好。

**💡 创新点**

创新点在于将个性化的时间间隔阈值与连续式衰减相结合，直接把时间相关性嵌入轻量级Skip‑gram学习中，突破传统Bag-of-Items假设。

**🔧 技术方法**

采用改进的Skip‑gram（Item2Vec）框架，结合IQR异常检测、Z‑score衰减、用户自适应时间窗口、加权损失与负采样等技术。

**📊 数据集**

使用八个公开数据集：Amazon-Beauty、Amazon-Books、Amazon-Games、BestBuy、CiaoDVD、MovieLens-100k、MovieLens-1M 与 RetailRocket。

**📈 对比分析**

在基于时间拆分的评估框架下，用NDCG@10和RMSE与Item2Vec、SeqI2V等基线比较，TAI2Vec在80%以上数据集上优于基线，稀疏场景提升高达135%，但在高密度、时间戳代表评分时间的数据集上性能略逊。

**⚠️ 局限性**

局限性包括：在高密度、时间戳为评分时间的稠密数据集（如ML-1M）效果不佳；对时间戳准确性和用户统计估计敏感；未整合内容特征，冷启动时依赖交互数据。

---

## 104. The Power of Information for Intermediate States in Contract Design

**arXiv ID:** 2604.15636 | [PDF](https://arxiv.org/pdf/2604.15636v1)

**作者:** Yirui Zhang `[一作]` (Tsinghua University), Zhixuan Fang `[通讯]` (Tsinghua University)

**通讯引用:** 821 | [OpenAlex ID](https://openalex.org/A5010064740)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了两阶段委托模型，并设计了两种新型合同——支付半途合同和终止半途合同，用以利用中间状态信息来激励代理人。

**💡 创新点**

创新点在于：①引入中间状态作为信息载体；②提出两种对中间信息利用方式不同的合同；③系统性地比较这两种合同与传统的最终结果合同在不同信息情景下的效能。

**🔧 技术方法**

主要技术包括：基于激励相容性与个人理性约束的合同设计；对两阶段模型的期望报酬与成本进行数学推导；使用反证法和极限例子给出上界与下界；对特殊流程（树形、随机首阶段、确定性首阶段）进行分段分析。

**📊 数据集**

本研究为理论分析，没有使用实际数据集；所有结论均通过构造例子与上界/下界证明得到。

**📈 对比分析**

通过对比标准结果合同、线性合同、支付半途合同和终止半途合同在三种特殊流程中的最大利润，发现：在树形流程两种新合同与标准合同等价；在随机首阶段流程支付半途合同与标准合同相同，终止半途合同可略优；在确定性首阶段流程，两种新合同均可实现近乎全福利提取，且支付半途合同在某些实例下比终止半途合同更优；总体而言，利用中间信息可显著提高委托方利润。

**⚠️ 局限性**

局限性：模型假设代理人只能在中间状态观察到信息，且奖励只在最终结果兑现；未考虑代理人多次互动、学习或动态预算约束；实际应用中如何估计中间状态分布与收益函数仍需经验验证。

---

## 105. Bureaucratic Silences: What the Canadian AI Register Reveals, Omits, and Obscures

**arXiv ID:** 2604.15514 | [PDF](https://arxiv.org/pdf/2604.15514v1)

**作者:** Dipto Das `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**通讯引用:** 2200 | [OpenAlex ID](https://openalex.org/A5100659941)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对加拿大联邦政府公开的 AI 注册表（共 409 个系统）进行混合方法分析，使用 ADMAPS 框架对注册表的内容进行定性编码与定量描述，揭示了注册表在披露方式、责任划分、对不确定性与人类判断的“官僚沉默”，并提出注册表是“本体设计工具”。

**💡 创新点**

创新点在于：①将注册表视为治理 artefact 而非中性透明工具；②引入 ADMAPS 框架来系统化解读 AI 系统与政府流程、人类判断和不确定性之间的互动；③首次提出“官僚沉默”概念，指出注册表对责任、隐私、培训等关键维度的刻意忽视，进而阐明注册表如何塑造可问责的范式。

**🔧 技术方法**

使用方法包括：混合方法（定量统计与定性编码）和基于 ADMAPS 的批判性话语分析；采用文本挖掘生成词云，识别技术与功能类别；通过归纳编码构建代码簿，并在 ADMAPS 维度（算法决策、官僚流程、人类判断）上进行映射。

**📊 数据集**

数据集为加拿大政府公开的 AI 注册表（Public AI Register）完整数据，涵盖 409 个 AI 系统的描述、技术、功能、使用者、组织、数据来源、是否涉及个人信息等字段。

**📈 对比分析**

本文未进行传统意义上的模型性能比较，而是以描述性分析和主题编码为主。通过统计各类技术（NLP、LLM、CV 等）和决策支持类型（信息化、预测、处方、操作自动化）的分布，展示了注册表中系统的技术与功能特征；对不确定性、资源与培训的描述则提供了“质量评估”视角，但未给出数值指标或实验结果。

**⚠️ 局限性**

局限性：①仅依赖注册表公开信息，无法获取未披露或非正式使用的系统与实践细节；②编码过程主观性高，未进行编码者一致性检验；③未考察 AI 系统实际运行效果或对公众的影响，只聚焦于登记记录的表层描写；④研究范围局限于加拿大，跨国比较不足。

---

## 106. A Quasi-Experiment comparing the health of unhoused people who have and have not experienced an eviction in King County, WA

**arXiv ID:** 2604.15504 | [PDF](https://arxiv.org/pdf/2604.15504v1)

**作者:** Ihsan Kahveci `[一作]` (University of Washington), Zack W. Almquist `[通讯]` (University of Washington)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5026370090)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在King County（华盛顿州）对2023年无家可归人群进行问卷调查，比较曾经历驱逐者与未经历驱逐者在自评健康、心理健康与物质使用方面的差异。

**💡 创新点**

创新之处在于：①同时考虑正式与非正式驱逐，填补了传统研究对非正式驱逐忽视的空白；②采用受访者驱动抽样（RDS）覆盖无家可归人群，并通过逆概率权重(IPTW)实现准实验设计；③通过加权广义线性模型与回归标准化得到驱逐的直接效应。

**🔧 技术方法**

使用技术包括逆概率权重(IPTW)加权GLM、回归标准化（G‑computation）来估计风险差异，并通过分层重抽样（fractional weighted bootstrap）计算标准误。

**📊 数据集**

数据集为2023年King County无家可归人群的RDS调查数据，最终分析样本为860名受访者（1,106名受访者扣除246名未问及驱逐历史者）。

**📈 对比分析**

方法通过IPTW实现样本平衡后进行加权回归，结果显示驱逐显著增加8.3个百分点的差距自评健康风险和9.5个百分点的物质使用风险，心理健康差异不显著；效应中等且稳健。

**⚠️ 局限性**

局限性包括：①样本主要为无家可归者，未能完全代表有暂时或永久住房的人群；②依赖自报健康指标，可能存在测量误差；③研究为横断面，无法完整揭示长期因果与中介机制。

---

## 107. LinuxArena: A Control Setting for AI Agents in Live Production Software Environments

**arXiv ID:** 2604.15384 | [PDF](https://arxiv.org/pdf/2604.15384v1)

**作者:** Tyler Tracy `[一作]` (Redwood Research), Aryan Bhatt `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入 LinuxArena 控制设置，构建 20 个多服务生产环境，包含 1671 个主任务和 184 个侧任务，评估攻击者（sabotage）和监视器（monitor）的表现，并发布人类红队生成的 LaStraj 攻击轨迹数据集。

**💡 创新点**

① 通过在多容器真实生产环境中同时嵌入合法任务与隐蔽攻击，提供最大、最具多样性的控制场景；② 采用 sabotage 与 monitor 两种评估验证该场景的攻击与防御可行性；③ 发布 LaStraj 作为人类红队生成的高难度攻击基准，弥补仅用模型生成攻击的局限。

**🔧 技术方法**

利用 Docker Compose 部署多容器环境，代理模型使用 Claude、GPT 系列 LLM 通过 ReAct 与 bash、文件编辑工具交互；监视器采用 LLM 评估每步和整条轨迹的可疑度；自动评分器对主任务和侧任务完成情况进行判定。

**📊 数据集**

LinuxArena 数据集：20 个环境、1,671 个主任务、184 个侧任务；LaStraj 数据集：人类红队手工构造的攻击轨迹，覆盖多种攻击策略。

**📈 对比分析**

通过对不同 LLM 作为攻击者或监视器进行 sabotage 与 monitor 评估，使用 step-wise false‑positive rate (FPR) 和 safety 曲线衡量；实验表明更强模型能完成更多主任务但也更易无检测完成侧任务；在 1% FPR 下监视器的安全率差异显著；与模型生成攻击相比，人类攻击在 0.1% FPR 下逃逸率显著更高，说明当前攻击工艺尚未饱和。

**⚠️ 局限性**

环境不完全逼真（直接操作生产服务器、使用合成数据），沙箱难以替代监控，覆盖范围仅限 20 个环境，红队构造受限（时间与人力），模型偏见（Claude 在任务生成上占优），潜在的无意完成侧任务与评分器干扰，控制协议评估仅覆盖监视器，任务难度与实际生产任务还有差距。

---

## 108. Dispatch-Aware Ragged Attention for Pruned Vision Transformers

**arXiv ID:** 2604.15408 | [PDF](https://arxiv.org/pdf/2604.15408v1)

**作者:** Saif Mahmoud `[一作]` (Al Ain University), Ahmad Almasri `[通讯]` (Al Ain University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个低调度开销的 Triton 可变长度注意力核，并将其集成到 Vision Transformer 的 token pruning 流程中，实现可观的推理加速。

**💡 创新点**

识别并量化了现有可变长度注意力 API 的调度开销瓶颈，并通过 Triton 实现了约1.55倍更低的调度开销，使 token pruning 的 FLOPs 节省在实际时间上可见。

**🔧 技术方法**

使用 Triton JIT 编写的 ragged attention kernel、快速的 token packer 以及完整的 pack–attend–unpack 推理流水线，并与 FlashAttention‑2 varlen、PyTorch SDPA 等实现进行对比。

**📊 数据集**

在 ImageNet‑1K 验证集上进行实验。

**📈 对比分析**

对比 padded SDPA、FA2 varlen 和 Triton pipeline，在 DeiT‑T/S/B、不同 pruning 方法（Threshold‑ℓ2、DynamicViT、EViT、ATS）下，Triton pipeline 在 batch 4–512 时相较于 padded 速度提升 1.37–2.24×，理论上最大可达 2.66×；在 50%/90% pruning 时得到显著加速，并保持 <0.007 的 logits 差异。

**⚠️ 局限性**

在无 pruning、长序列（如 LLM）或需要 causal mask 的场景下，Triton kernel 的性能不如 FlashAttention‑2 的手工调优 CUDA kernel；且仅适用于非 causal、无 dropout 的 ViT self‑attention，无法直接迁移到其他注意力变体。

---

## 109. CXR-LT 2026 Challenge: Multi-Center Long-Tailed and Zero Shot Chest X-ray Classification

**arXiv ID:** 2604.15555 | [PDF](https://arxiv.org/pdf/2604.15555v1)

**作者:** Hexin Dong `[一作]` (Weill Cornell Medicine), Yifan Peng `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 10922 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CXR-LT 2026多中心长尾与开放世界胸部X光诊断基准，并在此基准上评估多标签分类与零样本识别性能

**💡 创新点**

首次结合放射科医师人工标注、跨中心测试与开放世界任务，提供更真实的临床评估环境

**🔧 技术方法**

采用多模态视觉‑语言模型（如OpenCLIP、CheXzero）与长尾损失（Distribution‑Balanced、ASL、LDAM）以及两阶段/多专家架构

**📊 数据集**

使用PadChest与NIH ChestX‑ray公开数据，训练集为报导提取标签，开发/测试集为医师人工标注的PadChest‑GR及NIH子集

**📈 对比分析**

对比多种方法，Task 1最高mAP≈0.585，Task 2最高mAP≈0.432；虽能提升常见疾病识别，但对极罕见病和跨中心泛化仍显不足，且模型校准与鲁棒性差异大

**⚠️ 局限性**

局限在于对尾部类别识别效果不佳、概率校准不足、对输入扰动敏感，且跨中心分布偏移仍导致性能显著下降

---

## 110. An Agentic Workflow for Detecting Personally Identifiable Information in Crash Narratives

**arXiv ID:** 2604.15369 | [PDF](https://arxiv.org/pdf/2604.15369v1)

**作者:** Junyi Ma `[一作]` (University of Wisconsin-Madison), Bin Ran `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套本地可部署的“agentic”工作流，用于在交通事故报告的自由文本叙述中自动检测并标注个人身份信息（PII），包括规则引擎、微调LLM以及后续验证环节；

**💡 创新点**

创新点在于将结构化PII（如电话、邮箱）与上下文依赖PII（如姓名、住址、字母数字标识）分别委托给最适合的检测器，并加入集成式多次LLM推理与agentic验证器，以显著提升高歧义类别的精准度和可审计性；

**🔧 技术方法**

使用的技术包括Microsoft Presidio的规则引擎、Llama 3.1‑8B模型经过LoRA微调的提示式标注器、K次集成推理（ensemble）以及基于LLM的验证器（Verifier）来做“KEEP/DROP/UNCERTAIN”决策；

**📊 数据集**

数据集为威斯康星州交通部（2017‑2022年）收集的事故报告叙述，其中随机抽取500条用于测试，2000条人工标注用于微调；

**📈 对比分析**

与Presidio、基于提示的LLM（Llama 3.1‑8B、Gemma‑27B）以及单一微调LLM做对比。agentic工作流整体精度0.82、召回0.94、F1 0.87、准确率0.96，显著优于所有基线；在高歧义的住址和字母数字标识上也实现了显著提升；

**⚠️ 局限性**

局限性包括：高歧义PII仍易产生误检；验证器的准确性受提示质量限制；需人工标注数据进行微调；目前只处理部分PII类别，且对多语言或不同地区交通记录的通用性尚未验证。

---

## 111. Acoustic and Facial Markers of Perceived Conversational Success in Spontaneous Speech

**arXiv ID:** 2604.15322 | [PDF](https://arxiv.org/pdf/2604.15322v1)

**作者:** Thanushi Withanage `[一作]` (University of Maryland), Carol Espy-Wilson `[通讯]` (University of Maryland)

**通讯引用:** 4242 | [OpenAlex ID](https://openalex.org/A5078241735)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了1500多段Zoom双人自然对话，提取语音、停顿、面部动作等多模态特征，并通过对比高低成功对话探究同步化与对话质量的关系。

**💡 创新点**

创新点在于首次系统性地将多模态同步化（音调、强度、面部表情）与非任务虚拟对话的自评成功度关联，并提出利用近邻差异测度验证声纹同步。

**🔧 技术方法**

采用了音频特征提取（Pitch估计网络、Praat强度）、OpenFace面部动作单元分析、近邻距离与非邻近基线比较、Pearson相关→Fisher z、Mann‑Whitney、Welch t、配对t检验和Cliff’s delta等统计技术。

**📊 数据集**

使用了CANDOR语料库（约1500段30分钟Zoom对话）及其随访后评估问卷构造的PCS评分。

**📈 对比分析**

通过对比高低成功对话的统计显著性（p<0.05，Benjamini‑Hochberg校正）发现，较高同步化（长轮次、短停顿、面部笑容相关FAU的高相关）显著对应更高PCS，表现为更小的邻近差异和更高的相关系数。

**⚠️ 局限性**

局限性包括仅限英语成人群体、仅关注极端PCS值、未考虑跨文化或多语言情况，以及对真实世界多样化对话的可推广性仍待验证。

---

## 112. LLM4C2Rust: Large Language Models for Automated Memory-Safe Code Transpilation

**arXiv ID:** 2604.15485 | [PDF](https://arxiv.org/pdf/2604.15485v1)

**作者:** Sarah Bedell `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5090346723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于检索增强生成（RAG）与大型语言模型（LLM）相结合的 C/C++ → Rust 自动转译框架，目标是提升生成 Rust 代码的内存安全性。

**💡 创新点**

创新点在于：①引入 RAG 通过检索 Rust 文档与编译错误信息为 LLM 提供上下文，显著降低模型的幻觉率；②采用两阶段转译（先生成不安全 Rust，再逐步优化为安全 Rust）和自动化的编译器验证；③在实验中系统评估了不同模型（GPT‑4o、GPT‑4‑Turbo、o3‑mini）的安全改进效果。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o、GPT‑4‑Turbo、o3‑mini）、检索增强生成（RAG）管道、代码段分割与重组策略、基于 Rust 编译器的安全性检测脚本，以及自我评估与编译器验证相结合的评估框架。

**📊 数据集**

使用 GNU Coreutils 七个 C 语言工具（uniq、cat、pwd、truncate、head、split、tail）作为实验数据集，涵盖不同规模与指针使用模式的代码。

**📈 对比分析**

对比方法：将原始转译（无 RAG）与 RAG 加强版的结果在 RPD（原始指针解引用）、UTC（不安全类型转换）和 ULoC（unsafe 代码行）三项指标上进行比较，并通过编译器错误码进行验证。实验显示 GPT‑4o 与 GPT‑4‑Turbo 在大多数程序中将 RPD、UTC 和 ULoC 几乎降至零，且安全改进率显著高于之前的 C2SaferRust、LAC2R 等基线；o3‑mini 的安全提升不稳定且幻觉率较高。

**⚠️ 局限性**

局限性包括：①实验规模仅限七个小工具，难以推广至更大或更复杂的代码库；②仍受 LLM 随机性与版本演进的影响，复现性可能受限；③安全评估主要基于编译器错误码，未覆盖所有潜在的运行时内存错误或逻辑错误；④对 Rust 生态的特定知识依赖较高，迁移到其他语言或框架可能需重新构建检索数据库。

---

## 113. GroupDPO: Memory efficient Group-wise Direct Preference Optimization

**arXiv ID:** 2604.15602 | [PDF](https://arxiv.org/pdf/2604.15602v1)

**作者:** Jixuan Leng `[一作]` (Carnegie Mellon University), Inderjit S. Dhillon `[通讯]` (Google)

**通讯引用:** 29760 | [OpenAlex ID](https://openalex.org/A5063459703)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种基于记忆高效代理的群组偏好优化方法，显著提升了LLM在多样化响应集上的对齐效果；

**💡 创新点**

创新点在于通过先进行无梯度前向推理计算每个样本的系数，然后在标准token级别反向传播，既保持了原群组目标的一阶梯度等价，又把内存开销降到与群组大小无关；

**🔧 技术方法**

技术包括群组DPO（Group DPO）、多偏好优化（MPO）、All‑Pairs、Softmax等群组目标的实现；利用无梯度系数预计算、动态批处理和梯度检查点来实现低内存；

**📊 数据集**

使用的公开数据集包括 AllenAI Dolci‑Instruct‑SFT、Dolci‑Instruct‑DPO、BytedTsinghua‑SIA DAPO‑Math‑17k，以及奖励模型 Skywork‑Reward‑V2‑Qwen3‑8B；

**📈 对比分析**

通过在离线和在线对齐实验中对比单对（DPO、RFT）与群组方法（Margin、MPO、All‑Pairs、Softmax），发现群组方法普遍优于单对；在多组大小、NLL正则化等维度进一步验证，表明正样本NLL是训练稳定性的关键；此外在内存与延迟上，代理实现比传统联合反向传播低内存、仅略高延迟；

**⚠️ 局限性**

局限性包括需要额外的无梯度前向传递导致轻微的延迟增加；代理目标仅保持一阶梯度，丢失二阶信息，对依赖高阶优化的算法可能产生不同表现；

---

## 114. How people use Copilot for Health

**arXiv ID:** 2604.15331 | [PDF](https://arxiv.org/pdf/2604.15331v1)

**作者:** Beatriz Costa-Gomes `[一作]` (Microsoft AI), Dominic King `[通讯]` (Microsoft AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 2026 年 1 月微软 Copilot 消费者日志中约 50 万条健康相关对话进行分析，构建了 12 类层级意图分类体系，并利用 LLM 驱动的主题聚类进一步细化意图内的具体话题，最终揭示了按设备、时间段以及关注对象（自我、亲属）划分的使用模式。

**💡 创新点**

创新点在于：① 将传统意图分析与 LLM 自动分类、人工验证相结合，创建了更细粒度、适用于健康领域的 12 类意图框架；② 采用 LLM 驱动的主题聚类，首次在大规模对话中系统捕捉到具体话题层次；③ 通过设备与时段分析，首次揭示了个人健康查询、照护者查询以及系统导航需求的分布与峰值时间。

**🔧 技术方法**

使用了隐私保护的两阶段 PII 清洗+LLM 摘要生成 pipeline；LLM 分类器（基于隐私摘要）对意图进行标注；人工专家标注作为验证集；TnT‑LLM 方法对摘要进行主题聚类；统计分析工具（Python/SQL）处理设备、时段与对象属性。

**📊 数据集**

数据集来源为微软 Copilot 消费者端日志（1 月 2026），约 50 万条标记为“Health and Fitness”主题的对话，覆盖全球用户（约 22% 来自美国），45% 为英文。

**📈 对比分析**

方法对比：LLM 意图分类器与人工标注在抽样对话上的一致率高，说明模型可用；主题聚类通过 LLM 生成命名集群，并按频率排序展示；设备/时段分析采用比例和差异统计，未给出具体精度指标，但显示出显著差异。

**⚠️ 局限性**

局限性包括：仅分析微软 Copilot 平台，缺乏跨平台验证；样本仅为 1 月份，可能受季节与节日影响；未收集用户行为或健康结果，无法评估回答质量与临床效用；意图分类在模糊场景下可能低估个人健康意图；数据来源受限于隐私屏蔽，无法深入个体健康状况。

---

## 115. Uncertainty, Vagueness, and Ambiguity in Human-Robot Interaction: Why Conceptualization Matters

**arXiv ID:** 2604.15339 | [PDF](https://arxiv.org/pdf/2604.15339v1)

**作者:** Xiaowen Sun `[一作]` (University of Hamburg), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了不确定性、模糊性与歧义性在HRI中的统一概念框架（UVA-phenomena），阐明三者的定义、区别及相互关系，并以此框架为基础评估和指导HRI模型与方法的设计与改进。

**💡 创新点**

首次将不确定性划分为认知与随机两类，模糊性分为认知模糊与语义模糊，歧义性按词汇、句法、语用、语义四层进行系统划分，形成完整且可操作的UVA-phenomena框架，并通过该框架对现有HRI方法进行定性评估。

**🔧 技术方法**

主要采用词典定义的概念分析、概率学习理论的分类方法以及语义层级划分技术；同时对KnowNo、HYNA、OSSA等现有框架进行系统性对比。

**📊 数据集**

本文未使用任何实验数据集，主要通过案例示例和文献对照来说明UVA-phenomena的应用。

**📈 对比分析**

通过对KnowNo、HYNA、OSSA三种方法在处理不确定性、模糊性和歧义性方面的覆盖面进行定性对比，未给出具体量化性能指标；仅指出各方法在UVA维度上的优势与不足。

**⚠️ 局限性**

缺乏实证验证与定量评估；框架仅为概念性描述，尚需进一步的实验研究和系统综述来验证其有效性与实用性。

---

## 116. Collaborative Filtering Through Weighted Similarities of User and Item Embeddings

**arXiv ID:** 2604.15573 | [PDF](https://arxiv.org/pdf/2604.15573v1)

**作者:** Pedro R. Pires `[一作]` (Federal University of São Carlos), Tiago A. Almeida `[通讯]` (Federal University of São Carlos)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种通过加权相似度融合用户-物品与物品-物品推荐的混合框架，使用同一组嵌入同时完成两种相似度计算；

**💡 创新点**

创新点在于利用共享嵌入避免多模型训练，仅需一次嵌入学习，随后将用户-物品与物品-物品相似度按可调权重合并，提升推荐稳定性；

**🔧 技术方法**

采用矩阵分解（ALS、BPR）和变分自编码器RecVAE生成嵌入，计算点积/余弦相似度，并通过加权融合得到最终评分；

**📊 数据集**

使用公开数据集：Anime、BestBuy、CiaoDVD、Delicious、Filmtrust、Jester、Last.FM、MovieLens‑1M、RetailRocket；

**📈 对比分析**

与纯用户-物品和纯物品-物品基线在5折交叉验证中进行HR@10与NDCG@10对比，混合模型在大多数数据集上取得最优或次优性能；

**⚠️ 局限性**

限制在于需要手动设定权重，且在某些稀疏数据集（如RetailRocket）融合效果略逊；进一步的自适应权重学习和对不同嵌入的多样化探索仍待研究。

---

## 117. Consistency Analysis of Sentiment Predictions using Syntactic & Semantic Context Assessment Summarization (SSAS)

**arXiv ID:** 2604.15547 | [PDF](https://arxiv.org/pdf/2604.15547v1)

**作者:** Sharookh Daruwalla `[一作]` (Tellagence Inc), Charles Weber `[通讯]` (Portland State University)

**通讯引用:** 982 | [OpenAlex ID](https://openalex.org/A5015238823)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SSAS框架，通过主题‑故事‑聚类层级分类与Summary‑of‑Summaries技术，构建有限注意力机制来提升LLM情感预测的一致性与质量。

**💡 创新点**

将语法与语义对齐、层级上下文汇总与SNR权重相结合，首次将LLM预处理转为可控的结构化上下文，从而显著抑制噪声并稳定输出。

**🔧 技术方法**

使用Gemini 2.0 Flash Lite LLM、层级聚类（Theme/Story/Cluster）、Syntactic & Semantic Alignment、Signal‑to‑Noise Ratio权重、递归摘要SoS及噪声/异常点筛选等技术。

**📊 数据集**

Amazon Product Reviews、Google Business Reviews与Goodreads Book Reviews三大公开评测数据集。

**📈 对比分析**

与直接将原始文本输入LLM的对照实验（10次跑），SSAS在三数据集上平均提升情感预测一致性约22‑28%，数据质量提升约30%，在六种鲁棒性场景中保持20%+改进。

**⚠️ 局限性**

对层级划分与阈值的手工设定有依赖，极低量或稀疏数据效果有限；当前仅验证情感预测任务，扩展至其他分析需进一步验证；LLM版本或模型更新可能需要重新调优参数。

---

## 118. DPDSyn: Improving Differentially Private Dataset Synthesis for Model Training by Downstream Task Guidance

**arXiv ID:** 2604.15660 | [PDF](https://arxiv.org/pdf/2604.15660v1)

**作者:** Mingxuan Jia `[一作]` (Sichuan University), Zhishuo Zhang `[通讯]` (Southwest Minzu University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在私有数据上训练差分隐私（DP）AI模型，并利用该模型对随机打乱的特征生成标签，生成符合DP要求的合成数据集。

**💡 创新点**

创新点在于将下游任务的关键信息融入合成过程：训练的DP模型既保证隐私，又保留完成目标任务所需的关键信息，从而显著提升合成数据的使用价值。

**🔧 技术方法**

主要技术包括：DP-SGD训练差分隐私AI模型；特征列级随机打乱保持边缘分布；利用训练好的DP模型进行标签预测；并结合隐私预算的组合定理保证整体DP。实现使用TensorFlow‑Privacy库计算噪声系数。

**📊 数据集**

实验使用四个公开的表格数据集：Adult、Br2000、LPD 和 Smoking。

**📈 对比分析**

与 8 个基线方法（ABSyn、PrivSyn、PrivPetal、PrivMRF、AIM、MST、MWEM+PGM、DP‑GAN）在数据效用、效率和可扩展性三方面对比。DPDSyn 在不同模型（MLP、SVM、FT‑Transformer）上，准确率提升最高 2.40×、F1 分数提升 15.28×；合成时间提升 333.73×；在数据规模扩大至 2×、3× 时仍保持领先的精度提升（约 1.5×）。

**⚠️ 局限性**

局限性包括：仅在表格数据上验证，缺乏对图像/文本等结构化数据的评估；训练 DP 模型本身仍需消耗时间与计算资源；在极大规模数据上，DP‑SGD 的噪声累积可能影响模型性能；实验中部分基线因运行时间过长被排除，未能全面比较。

---

## 119. SocialWise: LLM-Agentic Conversation Therapy for Individuals with Autism Spectrum Disorder to Enhance Communication Skills

**arXiv ID:** 2604.15347 | [PDF](https://arxiv.org/pdf/2604.15347v1)

**作者:** Albert Tang `[一作]` `[通讯]`, Albert Tang

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款基于浏览器的、面向自闭症谱系障碍（ASD）成人的对话式疗法工具SocialWise，允许用户在多种社交场景下进行文字或语音角色扮演，并即时获得结构化的沟通反馈。

**💡 创新点**

创新点在于将大型语言模型（GPT‑4o‑mini）与检索增强生成（RAG）知识库相结合，既能保持对话连贯又能以循证的ASD治疗指南为基础提供个性化、即时的反馈；同时实现了全流程无专业人员监督、可随时随地使用的多模态（语音+文字）交互体验。

**🔧 技术方法**

核心技术包括：OpenAI Whisper语音识别与TTS、GPT‑4o‑mini对话引擎、LangChain RAG链条、ChromaDB向量检索、Streamlit前端框架（后续计划迁移至FastAPI/NextJS）。

**📊 数据集**

使用了手工整理的ASD治疗与社交沟通文献作为RAG检索库，并在远程试点中收集了34名自闭症成人（18–34岁）的使用体验数据。

**📈 对比分析**

虽然未进行正式临床试验，但在原型试点中，用户整体满意度为4.15/5，100%表示愿意推荐；主要感知收益包括改善轮流发言、解决社交冲突的能力提升以及减少真实对话焦虑，证明该工具在易用性和感知效益方面具有良好表现。

**⚠️ 局限性**

局限性包括：缺乏大规模临床验证、情景库相对有限、对模型的偏差与幻觉控制仍在进一步完善、缺乏进度跟踪与游戏化激励等功能。

---

## 120. Beyond Content Exposure: Systemic Factors Driving Moderators' Mental Health Crisis in Africa

**arXiv ID:** 2604.15321 | [PDF](https://arxiv.org/pdf/2604.15321v1)

**作者:** Nuredin Ali Abdelkadir `[一作]` (University of Minnesota), Stevie Chancellor `[通讯]` (University of Minnesota)

**通讯引用:** 2638 | [OpenAlex ID](https://openalex.org/A5046479021)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对134名非洲社交媒体内容审核员进行问卷调查，并对15名审核员进行深度访谈，分析了其心理健康状况及导致其压力的系统性与结构性劳动因素，发现非洲审核员的心理压力明显高于其他地区，且前雇员长期承受更大困扰；

**💡 创新点**

首次将系统性劳动条件与非洲审核员心理健康联系起来，揭示前雇员长期心理压力的延续性，并批判当前平台企业健康计划的无效性；

**🔧 技术方法**

采用定量问卷（使用标准心理健康量表）与定性访谈相结合的混合方法；

**📊 数据集**

使用134名审核员的问卷数据和15名审核员的访谈记录作为研究数据集；

**📈 对比分析**

将调查结果与已有的其他地区审核员基准数据进行对比，结果显示非洲审核员的心理压力显著更高，表明企业现有的健康干预措施效果不佳；

**⚠️ 局限性**

研究样本规模有限，主要来自少数平台，缺乏纵向追踪，且未充分考虑不同文化背景下的差异与多样性。

---

## 121. GazeSync: A Mobile Eye-Tracking Tool for Analyzing Visual Attention on Dynamically Manipulated Content

**arXiv ID:** 2604.15348 | [PDF](https://arxiv.org/pdf/2604.15348v1)

**作者:** Yaxiong Lei `[一作]` (University of St Andrews), Juan Ye `[通讯]` (University of St Andrews)

**通讯引用:** 2757 | [OpenAlex ID](https://openalex.org/A5100682481)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

开发并验证了一套名为GazeSync的移动端眼动跟踪系统，该系统能够同步眼动数据与图像的平移、旋转、缩放状态，并在此基础上重建图像相对的注视点；

**💡 创新点**

在移动端眼动追踪领域首次提出统一同步日志架构，将眼动样本与变换状态精准配对；实现逆矩阵重建以获得内容相对注视，突破了传统仅报告屏幕坐标的局限；

**🔧 技术方法**

采用Flutter框架构建iOS应用，使用Eydid/SeeSo SDK进行实时眼动估计，配合Python/FastAPI本地日志服务器实现双流时间对齐；利用变换矩阵的逆运算进行注视重建，并提供热力图、轨迹叠加与实时回放等可视化工具；

**📊 数据集**

实验数据来自3名受试者在自定义的12项任务（引导、阅读、视觉搜索）中产生的眼动与变换日志，未使用公开数据集；

**📈 对比分析**

与传统屏幕坐标下的误差对比，图像坐标重建的中位误差为92像素，而屏幕坐标误差为640像素；同步成功率96.5%，丢弃率3.5%，每30分钟重校准次数中位为3次；

**⚠️ 局限性**

系统存在校准漂移需要频繁重校准的缺点；在快速组合变换（如大角度旋转+缩放）下，逆变换容易出现数值不稳定，导致注视重建失真；

---

## 122. A Systematic Review of User Experiments Measuring the Effects of Dark Patterns

**arXiv ID:** 2604.15323 | [PDF](https://arxiv.org/pdf/2604.15323v1)

**作者:** Brennan Schaffner `[一作]` (Georgetown University), Marshini Chetty `[通讯]` (University of Chicago)

**通讯引用:** 3405 | [OpenAlex ID](https://openalex.org/A5065790866)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并汇总了 27 篇论文中 148 个实验单元，量化暗黑模式（DMP）对用户行为的影响及干预效果。

**💡 创新点**

首次将所有实验证据按五类（对照、干预、叠加、对比、个体特征）系统分类，并提供效应量分布、显著性比例以及对干预无效性的实证证据。

**🔧 技术方法**

采用学术数据库检索（Google Scholar、ACM DL、SSRN、arXiv）与 Rayyan 自动筛选，随后手工编码实验单元、统计显著性与效应大小。

**📊 数据集**

数据集来源于 991 篇候选文献，经筛选得到 27 篇同行评审实验论文，总计 148 个实验单元，覆盖 2019‑2025 年。

**📈 对比分析**

使用比例检验和效应量计算（相对/绝对增幅）比较实验结果；结果显示 85% 的实验显著受 DMP 影响，平均相对增幅约 211%，平均绝对增幅约 16.5%。

**⚠️ 局限性**

局限性包括仅涵盖已发表英文同行评审实验、可能存在发表偏倚、缺乏长期或社会层面效应评估、干预实验样本有限，以及对个体特征的交互效应研究不足。

---

## 123. HarmfulSkillBench: How Do Harmful Skills Weaponize Your Agents?

**arXiv ID:** 2604.15415 | [PDF](https://arxiv.org/pdf/2604.15415v1)

**作者:** Yukun Jiang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型代理生态中的有害技能进行大规模测量，并构建了 HarmfulSkillBench 基准来评估 LLM 代理在面对有害技能时的安全行为。

**💡 创新点**

首次系统研究有害技能，提出了两层的有害技能分类学，利用 LLM 驱动的评分系统对数万技能进行检测，并在基准中设计多种条件揭示“技能阅读攻击”与分层安全缺陷。

**🔧 技术方法**

采用 GPT-5.4-Mini 进行多轮评分、阈值筛选；使用 Python 爬虫采集 ClawHub 与 Skills.Rest 技能；构建多条件评估框架，使用人类评审和自定义评判器计算拒绝率、HiTL/AID 合规度和综合危害分；对 6 种主流 LLM 进行实验。

**📊 数据集**

数据来源于 ClawHub（26,629 技能）和 Skills.Rest（71,811 技能）共计 98,440 技能；人工标注 500 条样本用于阈值确定；Benchmark HarmfulSkillBench 包含 200 条技能（130 Tier 1 + 70 Tier 2）及 62 条原创示例。

**📈 对比分析**

在 6 种 LLM 上分别在四类评估条件（A、B、C1–C4、D）下测量拒绝率、HiTL/AID 触发率和危害分；实验显示：Skill‑Reading 攻击使危害分从 0.08 上升至 0.79，GPT‑5.4‑Mini 以 0.52 的最高安全分领先；不同模型在 Tier 1 与 Tier 2 之间表现差异显著，表明存在安全与合规性缺口。

**⚠️ 局限性**

局限性包括仅评估规划阶段不执行真实攻击、只覆盖 6 款主流 LLM、任务与评估全部为英文、未考虑非英语技能的文化差异，以及基准任务仅为自然语言描述而非完整可执行脚本。

---

## 124. Evaluating LLMs as Human Surrogates in Controlled Experiments

**arXiv ID:** 2604.15329 | [PDF](https://arxiv.org/pdf/2604.15329v1)

**作者:** Adnan Hoq `[一作]` (University of Notre Dame), Tim Weninger `[通讯]` (University of Notre Dame)

**通讯引用:** 3241 | [OpenAlex ID](https://openalex.org/A5084597959)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了开箱即用的LLM能否作为受控行为实验中的人类替身，通过将人类实验数据转化为结构化提示让LLM生成同一任务的评分，并对比统计推断结果。

**💡 创新点**

提出了“结构一致性”评估框架，直接在相同统计模型下检验LLM生成数据是否复制人类的效应方向、交互结构和刺激级异质性，而非仅仅检验方向性或相关性。

**🔧 技术方法**

使用无任务特定微调的多种LLM（GPT‑5.2、Gemma‑2:9B、Llama‑3.2:3B），并采用确定性解码 (temperature=0) 生成0–10的准确度评分。

**📊 数据集**

采用一项政治新闻头条真实性判断的实验数据（522名参与者），将其转化为Prompt并让LLM产生对应响应。

**📈 对比分析**

方法是将人类和LLM生成的数据分别拟合同一混合效应模型和双因素ANOVA，对效应大小(eta²)、方向和交互进行对比；结果显示所有模型都捕获效应方向，但GPT‑5.2在效应大小和异质性上最接近人类，其它模型则存在幅度过大或压缩的偏差。

**⚠️ 局限性**

局限包括仅在单一实验范式验证、仅评估群体水平而非个体一致性、对提示和人种等属性的鲁棒性未作系统测试，以及可能存在的模型对人类行为的“表面匹配”而非真实认知机制。

---

## 125. LLM attribution analysis across different fine-tuning strategies and model scales for automated code compliance

**arXiv ID:** 2604.15589 | [PDF](https://arxiv.org/pdf/2604.15589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 126. SAGE: Selective Attention-Guided Extraction for Token-Efficient

**arXiv ID:** 2604.15583 | [PDF](https://arxiv.org/pdf/2604.15583v1)

**作者:** Xinzhi Wang `[一作]` (Purdue University), Chunwei Liu `[通讯]` (Purdue University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Selective Attention-Guided Extraction（SAGE），一种无训练、基于局部LLM注意力热图的上下文缩减框架。

**💡 创新点**

创新点在于利用查询相关的注意力热图进行差分注意力降噪，并按预算高效选择连续文本片段。

**🔧 技术方法**

采用轻量级本地LLM进行前向预填充以获取注意力分数、差分注意力、窗口选择和KV缓存重用。

**📊 数据集**

在QuALITY‑hard、Paper、Notice、AIT‑QA四个长文档/表格QA基准上进行评估。

**📈 对比分析**

与强基线RAG进行token预算对齐比较，SAGE在10%预算下在QuALITY‑hard取得第4名，整体准确率超过检索基线且token消耗降低约90%。

**⚠️ 局限性**

局限包括对极度重复结构文档的贪婪选择可能导致误选、对复杂结构表格的行级选择仍需改进，且对不同语言/领域仍有适配需求。

---

## 127. BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels

**arXiv ID:** 2604.15591 | [PDF](https://arxiv.org/pdf/2604.15591v1)

**作者:** Mengfei Lan `[一作]` (University of Illinois Urbana-Champaign), Halil Kilicoglu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3206 | [OpenAlex ID](https://openalex.org/A5016571803)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在医学文本检索中，本文提出 BioHiCL，通过利用 MeSH 分层多标签对检索模型进行对比学习，提升了检索、相似度和问答任务的表现。

**💡 创新点**

创新点在于将层级化 MeSH 注释作为深度标签监督，将嵌入相似度与标签相似度对齐，并引入层级对比损失，突破传统二元相关信号的限制。

**🔧 技术方法**

采用 LoRA 参数高效微调、对比学习（hierarchical multi-label contrastive loss）、回归对齐损失以及权重深度编码的技术。

**📊 数据集**

使用 BioASQ 2022 任务 1a 摘要及其 MeSH 标签、TREC-COVID、NFCorpus、SciFact、SCIDOCS、PubMedQA 等公开医学检索与相似度数据集。

**📈 对比分析**

与 0.1B–1.5B 参数的通用与医学专用检索模型（如 BGE、BiomedBERT、MedCPT、BMRetriever 等）对比，BioHiCL-Base 以 0.543 的 IR 平均分位居同类模型第一，且在句子相似度和问答 Recall@1 上均取得领先或相近成绩。

**⚠️ 局限性**

局限性包括依赖高质量的 MeSH 标签，若标签稀疏或噪声高会影响效果，且层级深度加权假设语义专一性与层级深度相对应，可能不适用于所有任务或领域。

---

## 128. Analyzing Chain of Thought (CoT) Approaches in Control Flow Code Deobfuscation Tasks

**arXiv ID:** 2604.15390 | [PDF](https://arxiv.org/pdf/2604.15390v1)

**作者:** Seyedreza Mohseni `[一作]` (University of Maryland Baltimore County), Manas Gaur `[通讯]` (University of Maryland Baltimore County)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对控制流代码混淆（包括控制流扁平化、不透明谓词及其组合）提出并实现基于链式思考（CoT）提示的LLM驱动去混淆方法。

**💡 创新点**

通过在提示中显式引导LLM逐步推理（识别分发器、跟踪状态变量、判定不透明谓词真值），显著提升了CFG恢复和语义保持，首次系统评估CoT对代码去混淆效果的提升。

**🔧 技术方法**

使用大型语言模型（GPT5、DeepSeek‑V2、Qwen‑3 Max、QWQ‑32B、OpenAI o3）在 128k 上下文窗口下进行CoT推理，并对LLVM‑IR与C源代码执行结构化去混淆流程。

**📊 数据集**

使用12个标准C基准程序（9个低复杂度、3个高复杂度），通过Tigress和O‑LLVM两款开源混淆器生成三种混淆级别（不透明谓词、CFF、Opaque‑CFF）进行评测。

**📈 对比分析**

采用结构相似度得分（SRS）和BLEU指标评估CFG重构与语义保持。CoT提示在所有模型上平均提升约15‑20%（SRS）与约20‑30%（BLEU），GPT5在所有设置下表现最佳，整体平均BLEU约99%，SRS约100%。

**⚠️ 局限性**

仅覆盖两款混淆器与两类混淆技术，缺乏对更高级或多层混淆的覆盖；模型仍出现幻觉（缺失头文件、结构体错误、语义偏差）；性能随不透明谓词数量和原始CFG复杂度下降；CoT效果受提示设计与示例的影响，难以完全归因于CoT本身。

---

## 129. The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference

**arXiv ID:** 2604.15409 | [PDF](https://arxiv.org/pdf/2604.15409v1)

**作者:** Ranjith Chodavarapu `[一作]` (Kent State University), Lei Xu `[通讯]` (Kent State University)

**通讯引用:** 7682 | [OpenAlex ID](https://openalex.org/A5081802824)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 FP16 推理下 KV 缓存与无缓存执行路径在数值上不等价的现象，系统地测量了在三款开源模型（LLaMA‑2‑7B、Mistral‑7B‑v0.3、Gemma‑2‑2B）上的 token 发散、KL 分布漂移、层级漂移以及通过激活补丁的因果定位。

**💡 创新点**

首次证明 FP16 的非可结合性导致 KV 缓存路径与无缓存路径在所有输入、模型、采样策略下均出现 100% 的确定性发散，并揭示 GQA 结构会放大该误差；同时定位误差来源于 KV 缓存状态而非残差流，提出 FP32 计算为唯一可消除该差异的手段。

**🔧 技术方法**

采用 FP16 与 FP32 对比实验、KL 与 JS 散度计算、层级漂移分析、激活补丁（patching）实验、统计检验（McNemar、Mann‑Whitney、Pearson 等）等技术。

**📊 数据集**

在 GSM8K 数学推理基准上评估，使用 700 条随机样本，每个样本 5 次种子，共 3,500 条对照运行。

**📈 对比分析**

比较方法：对比 cache‑ON 与 cache‑OFF 的 token 交叉率、平均 KL、准确率差异；FP32 环境下的对比展示发散率从 100% 降至 0%、KL 下降 8 个数量级。结果表明 FP16 缓存推理在所有模型、策略下均存在确定性差异，FP32 可将误差压到噪声底层。

**⚠️ 局限性**

局限性：仅测试三款模型和单一基准，未覆盖更大模型、指令调优模型或其他任务；因果定位通过补丁排除法完成，未直接对 KV 缓存进行补丁验证；目前唯一可行的消除方案为 FP32，缺乏更高效的缓冲或混合精度策略。

---

## 130. Facial-Expression-Aware Prompting for Empathetic LLM Tutoring

**arXiv ID:** 2604.15336 | [PDF](https://arxiv.org/pdf/2604.15336v1)

**作者:** Shuangquan Feng `[一作]` (University of California San Diego), Virginia R. de Sa `[通讯]` (University of California San Diego)

**通讯引用:** 3111 | [OpenAlex ID](https://openalex.org/A5071129405)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不进行端到端重训练的前提下，利用面部表情信息改进LLM驱动的在线辅导系统的同理心回复。

**💡 创新点**

提出两种轻量级面部表情感知方法：①将AU估计转为文本描述注入提示；②用AU指示的峰值帧作为视觉输入；并验证其在不同LLM基底上的有效性。

**🔧 技术方法**

使用Action Unit估计模型（AUM）提取面部表情，结合LLM/多模态LLM（如GPT‑5.1、Claude Ops 4.5、Gemini 2.5 Pro）与提示工程实现。

**📊 数据集**

利用HDFE‑DevSplit‑Unlabeled（465人、20‑25段姿态化视频）构建模拟学生表情库，并在此基础上生成320个多轮辅导对话。

**📈 对比分析**

对四种导师变体（文本仅、随机帧、文本+AU、峰值帧+AU）进行三组预设对比，使用5位人工评审和GPT‑5.1 AI评审。结果显示：①文本+AU优于文本仅；②峰值帧+AU优于随机帧；③不同LLM基底对AU整合方式偏好不同；总体同理心提升显著，且未明显牺牲教学效果。

**⚠️ 局限性**

主要限制：实验采用模拟学生表情视频，未覆盖真实课堂中的自然细微表情；评估集中在同理心维度，未检验学习效果和学生参与度的实际提升。

---

## 131. Reward Weighted Classifier-Free Guidance as Policy Improvement in Autoregressive Models

**arXiv ID:** 2604.15577 | [PDF](https://arxiv.org/pdf/2604.15577v1)

**作者:** Alexander Peysakhovich `[一作]` (Sutter Hill Ventures), William Berman `[通讯]` (Sutter Hill Ventures)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种奖励加权无分类器引导（RCFG）方法，使自回归生成模型在推理阶段即可根据任意奖励函数动态调整生成结果，而无需重新训练。

**💡 创新点**

创新点在于将CFG视为对Q函数的近似，实现了对多属性奖励的无缝推理引导，并可将RCFG作为RL的warm‑start提升收敛速度。

**🔧 技术方法**

使用的技术包括条件自回归语言模型、奖励函数标准化、CFG与Q函数近似、KL蒸馏、RDKit属性提取以及分割属性编码。

**📊 数据集**

数据集基于公开的小分子药物库（SMILES），通过RDKit提取25个分子属性，训练条件模型并用于评估。

**📈 对比分析**

在与oracle、best‑of‑4拒绝采样和传统RL基线的对比中，RCFG在保持生成多样性的同时，获得与多步RL相当的奖励；在RL warm‑start 实验中，RCFG显著加速了收敛。

**⚠️ 局限性**

局限性包括对属性空间稀疏或高度相关的奖励函数下条件化至y*可能失效；RCFG在推理时需要多次前向传播，计算成本较高；多次蒸馏在RL训练中表现不稳定。

---

## 132. LACE: Lattice Attention for Cross-thread Exploration

**arXiv ID:** 2604.15529 | [PDF](https://arxiv.org/pdf/2604.15529v1)

**作者:** Yang Li `[一作]` (Rutgers University), Chengzhi Mao `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LACE框架，在Transformer中引入轻量级的Lattice Attention实现跨线程协作推理；

**💡 创新点**

创新点在于将跨线程注意力嵌入标准因果注意力中，允许并行推理路径实时共享中间结果并自我纠错；

**🔧 技术方法**

采用Lattice Attention（2D注意力+门控融合）、连续预训练、SFT、Lattice GRPO强化学习与自我评估标签；

**📊 数据集**

使用自构造的多线程数据集（从DAPO等数学/推理任务筛选，并通过迭代采样、正则化历史策略生成多样化推理轨迹）；

**📈 对比分析**

与独立采样、孤立并行及标准SFT+RL基线比较，实验在AIME、LiveBench等数学与推理基准上，LACE在准确率、格式遵从率和探索多样性上均显著提升，RL阶段提升超过7分；

**⚠️ 局限性**

局限在于需要专门的多线程数据生成与训练步骤，且对模型规模、硬件延迟有轻微影响，跨线程沟通效果仍受数据质量与奖励设计限制。

---

## 133. Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures

**arXiv ID:** 2604.15351 | [PDF](https://arxiv.org/pdf/2604.15351v1)

**作者:** Abdulmalek Saket `[一作]` `[通讯]` (Royal Fenice Kft), Abdulmalek Saket (Royal Fenice Kft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了基于梯度的层选择方法 Aletheia，用于对大语言模型的 LoRA 微调进行层级稀疏化，减少训练成本。

**💡 创新点**

通过轻量级梯度探针快速识别任务相关层，并只在这些层中放置 LoRA 适配器，实现了显著的训练速度提升而无显著质量下降。

**🔧 技术方法**

使用梯度探针、梯度幅值排序、半对称低秩 LoRA 适配器及固定 50% 层选择与不对称秩分配技术。

**📊 数据集**

采用 Aletheia Bootstrap（指令式对话数据集）进行微调，并使用 MMLU、GSM8K、HumanEval 等基准进行评估。

**📈 对比分析**

对 81 条实验行、14 个模型（8 家族、0.5B–72B 参数）进行三种种子比较，平均训练速度提升 23.1%（p < 0.001，100% 胜率），额外遗忘 ≤ 0.5pp，核心模型的 MMLU、GSM8K、HumanEval 几乎保持不变。

**⚠️ 局限性**

仅在指令式任务上验证；Pythia-1.4B 在 fp16 下失稳；固定 LoRA 秩可能限制进一步提升；不同领域或更大模型可能需要重新校准层选择比例。

---

## 134. Preregistered Belief Revision Contracts

**arXiv ID:** 2604.15558 | [PDF](https://arxiv.org/pdf/2604.15558v1)

**作者:** Saad Alqithami `[一作]` `[通讯]`, Saad Alqithami

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种名为 PBRC（Preregistered Belief Revision Contracts）的协议层，用于在多智能体系统中通过预先注册的触发器、证据标记和回退策略，严格区分开放交流与可接受的知识更新；

**💡 创新点**

核心创新在于通过证据门控机制将社交影响与外部可验证证据分离，证明在此协议下不存在纯粹因社会共识导致的高置信错误共识（wrong‑but‑sure）并提供可审计的证据链；

**🔧 技术方法**

采用形式化逻辑（包括一阶触发器、证据标记、审核日志）、证明技术（归纳、正则化、代数化简）以及分布式传播模型（洪泛传播、可达性分析）来实现协议的规范、验证与分析；

**📊 数据集**

论文未使用特定数据集，而是通过模拟和理论分析（如 LLM 对话示例）展示协议效果；

**📈 对比分析**

通过理论证明与仿真验证，展示 PBRC 能有效抑制信息链式共识、保证审计可追溯性，并与传统投票/聚合机制比较，显示在保持开放交流的同时显著降低错误共识风险；

**⚠️ 局限性**

局限性包括：需要可靠的证据标记与验证层；无法自我纠正错误或误标记的证据；在网络攻击（伪造、重放）下仍需依赖底层安全机制；并且无法保证在信息缺失或拒绝服务情况下的 liveness。

---

## 135. Can a Weaker Player Win? Adaptive Play in Repeated Games

**arXiv ID:** 2604.15315 | [PDF](https://arxiv.org/pdf/2604.15315v1)

**作者:** Jonatha ANSELMI `[一作]` (University of Grenoble Alpes), Bruno Gaujal `[通讯]` (University of Grenoble Alpes)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了弱玩家在两种对战风格（进攻和防守）之间的自适应策略，以在有限或无限回合中获得正的匹配收益。

**💡 创新点**

创新点在于将Parrondo悖论与有限时域动态规划相结合，首次给出在不同防守安全性下的正收益可行性阈值，并完整划分了无穷期极限行为。

**🔧 技术方法**

采用随机游走分析、超级鞅方法、Bellman最优递推与数值动态规划技术，对策略空间进行最优性与结构化分析。

**📊 数据集**

研究基于理论模型，无需实测数据集；所有结果通过数值实验验证，但仅涉及二进制胜负/平局三结果的简化情形。

**📈 对比分析**

通过数值动态规划对比固定风格、catenaccio与最优策略，展示了收益随回合数非单调、特定时域正收益以及不同防守安全性下的性能差异；在安全防守下最优收益可逼近1。

**⚠️ 局限性**

局限在于仅考虑两种风格与简单的赢/输/平局奖励，未涉及多策略、学习或更复杂计分规则，且对现实棋局的可推广性未作实验验证。

---

## 136. Trajectory Planning for Safe Dual Control with Active Exploration

**arXiv ID:** 2604.15507 | [PDF](https://arxiv.org/pdf/2604.15507v1)

**作者:** Kaleb Ben Naveed `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2411 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种预算约束下的双控制框架，结合鲁棒规划与主动不确定性减小，并提供安全和预算可行性的正式保证。

**💡 创新点**

创新点在于：①将探索决策视为可验证的候选方案，而非加权目标；②在每个规划周期动态评估候选轨迹的安全性和预算影响；③提供两种安全实现（管式MPC与门控器）实现模块化。

**🔧 技术方法**

核心技术包括：管式MPC、控制边界函数/门控安全过滤、基于SMID的参数集更新、信息矩阵或滚动仿真预测不确定性收缩、预算约束下的候选评估与选择。

**📊 数据集**

在四旋翼导航和自动赛车两种仿真案例中验证，使用无人机的拖曳模型与赛车的摩擦系数模型作为不确定参数；无公开数据集，实验采用仿真环境。

**📈 对比分析**

与传统保守鲁棒方案、基于权重的探索策略、仅回退策略等基线比较。实验显示，提出框架在保持安全性的同时实现了约 80‑85% 的任务成本降低，探索预算使用率仅 20‑25%，并显著收敛到较小的不确定性集合。

**⚠️ 局限性**

局限性包括：①假设参数为时不变，无法处理漂移或时间变化的不确定性；②依赖预设的探索预算，缺乏对不确定性对未来成本影响的显式量化；③预测的不确定性收缩与实际执行误差不完全一致，可能导致估计偏差。

---

## 137. Photonic AI: A Hybrid Diffractive Holographic Neural System for Passive Optical Real-Time Image Classification

**arXiv ID:** 2604.15364 | [PDF](https://arxiv.org/pdf/2604.15364v1)

**作者:** Prakul Sunil Hiremath `[一作]` `[通讯]` (Visvesvaraya Technological University), Prakul Sunil Hiremath (Visvesvaraya Technological University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种融合差分式光学神经网络与全息干涉学习（HIBL）技术的光学推理系统，用于图像分类。

**💡 创新点**

核心创新在于将学习到的相位分布映射到可制造的全息干涉图样的正式算子（HIBL），从而闭环地将数字优化结果物理化；同时给出了完整的算子分解框架，明确区分学习、设计与非线性部分。

**🔧 技术方法**

采用差分式光学神经网络（DONN）与自由空间传播的角谱方法、相位调制算子、光学检测算子，辅以全息干涉记录算子，整体在计算机仿真中实现。

**📊 数据集**

使用MNIST手写数字数据集进行训练和评估。

**📈 对比分析**

与传统电子模型（如MobileNetV1、Jetson Nano）相比，光学系统在推理时仅需光学传播，延迟纳秒级，计算能耗可忽略；在MNIST上实现91.2%测试准确率，证明该受限函数类足以完成分类任务。

**⚠️ 局限性**

主要限制包括：HIBL映射的光学逼真度与成像误差、制造与对准误差导致的相位误差、对空间与时间相干性的高要求，以及在更复杂任务和更大输入尺寸上的可扩展性不足。

---

## 138. The Synthetic Media Shift: Tracking the Rise, Virality, and Detectability of AI-Generated Multimodal Misinformation

**arXiv ID:** 2604.15372 | [PDF](https://arxiv.org/pdf/2604.15372v1)

**作者:** Zacharias Chrysidis `[一作]` (Centre for Research and Technology Hellas), Symeon Papadopoulos `[通讯]` (Centre for Research and Technology Hellas)

**通讯引用:** 6201 | [OpenAlex ID](https://openalex.org/A5013616365)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含超过15万条多模态误导信息（误标、编辑和AI生成）的数据集，并基于X的社区说明（Community Notes）对其进行纵向分析。

**💡 创新点**

首次量化AI生成媒体在病毒性、被动参与度、社区共识速度等维度的特征，并在真实世界数据上评估SIDs与VLMs随时间退化的检测性能。

**🔧 技术方法**

采用关键字过滤、零样本Gemma 3文本-图像推理、统计计量指标（Virality Share、Engagement Index、Consensus Metrics）以及对检测模型的定量评测。

**📊 数据集**

使用X Community Notes 公开语料（2021‑2026年）以及从中筛选的AI生成与真实图像子集（约21.7K张）。

**📈 对比分析**

与六种检测模型（SPAI、RINE、BFree、Gemma 3、Grok 4.1 Fast、GPT‑5‑mini）比较，VLM在整体准确率上超过SIDs，但所有模型的TPR随时间显著下降（从70%到约50%）。

**⚠️ 局限性**

限制包括弱监督标注可能引入噪声、模型评估受限于已公开数据、检测模型对新一代生成器的泛化不足，以及对视频内容检测缺乏系统评测。

---

## 139. Hallucination as Trajectory Commitment: Causal Evidence for Asymmetric Attractor Dynamics in Transformer Generation

**arXiv ID:** 2604.15400 | [PDF](https://arxiv.org/pdf/2604.15400v1)

**作者:** G. Aytug Akarlar `[一作]` `[通讯]`, G. Aytug Akarlar

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究同一提示下语言模型生成的事实与幻觉轨迹的分叉现象，证明幻觉是早期轨迹承诺。

**💡 创新点**

首次将同提示分叉与对称激活补丁结合，揭示幻觉是非对称吸引子；证明纠正需持续干预。

**🔧 技术方法**

使用同提示分叉、激活补丁、KL 与 Cohen d 分离、PCA 投影、线性探针、无监督聚类等技术。

**📊 数据集**

基于 Qwen2.5‑1.5B 模型，构造 61 个跨六类（事实、假前提、虚构、引导、多跳、数学）提示集。

**📈 对比分析**

与随机采样、无补丁基线以及随机补丁对照比较；发现纠正率为 33.3%（对照 10.4%）而腐败率达 87.5%，差异显著。

**⚠️ 局限性**

实验仅限单一模型、样本量有限、补丁仅单层、提示类别混杂，且线性干预无效；需更大模型、多层干预及更精细分类验证。

---

## 140. Taming Asynchronous CPU-GPU Coupling for Frequency-aware Latency Estimation on Mobile Edge

**arXiv ID:** 2604.15357 | [PDF](https://arxiv.org/pdf/2604.15357v1)

**作者:** Jiesong Chen `[一作]` (City University of Hong Kong), Zhenjiang Li `[通讯]` (City University of Hong Kong)

**通讯引用:** 4500 | [OpenAlex ID](https://openalex.org/A5100419083)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对移动边缘设备的 AI 推理，提出 FLAME 框架，基于层级建模和 CPU‑GPU 异步耦合的动态交互因子，能够在多频率环境下精确预测推理延迟。

**💡 创新点**

创新点包括：① 用层级分析捕获 CPU 与 GPU 的动态交互因子并分段建模；② 采用依赖感知的时间线重构实现全模型延迟估计；③ 只需稀疏采样即可完成层级参数回归；④ 基于 FLAME 的基于截止时间的 DVFS 调度。

**🔧 技术方法**

技术手段有：频率尺度分析、层级时间模型、分段线性/多项式回归、XGBoost HPC 预测器、在线自适应校正、贪心搜索的 DVFS 调度。

**📊 数据集**

实验使用的模型集为 ResNet50、VGG16、DenseNet121（传统 DNN）以及 GPT2‑large、Qwen2‑1.5B、Qwen2‑7B（小型语言模型），在 NVIDIA Jetson AGX Orin 与 Orin NX 平台上进行测试。

**📈 对比分析**

与基准方法（分析模型、MLP 预测器、最大频率、商业 DVFS、zTT）对比，FLAME 将平均估计误差降至 8.14%（相比 24–27%），将完整剖析时间从数小时/数天缩短到 2–6 分钟；在 DVFS 场景下实现 23.5% 的能耗提升和 4.35% 的 QoS 保证，优于 zTT 与商业策略。

**⚠️ 局限性**

局限性包括：仍需对新层类型进行少量剖析；对极端异构或多核 GPU 设备的适配待验证；在线自适应在高并发负载下可能需要更细粒度调节；当前仅针对单流推理，未覆盖批量推理或多任务场景。

---

## 141. Five Constructions of Asymptotically Optimal Aperiodic Doppler Resilient Complementary Sequence Sets with New Parameters

**arXiv ID:** 2604.15403 | [PDF](https://arxiv.org/pdf/2604.15403v1)

**作者:** Xuanyu Liu `[一作]` (Fujian Normal University), Zuling Chang `[通讯]` (Zhengzhou University)

**通讯引用:** 150 | [OpenAlex ID](https://openalex.org/A5009951050)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了基于有限域迹函数与列正交复矩阵的五类新的无周期 Doppler‑Resilient Complementary Sequence Set（DRCSS）构造方法，获得了新的参数集并证明其在相应下界上渐进最优。

**💡 创新点**

创新点在于：①首次将列正交复矩阵与有限域迹函数相结合生成多种参数的无周期 DRCSS；②在三种构造中实现了列序列峰均功率比（PAPR）可控且不超过基数 p；③相较现有构造，获得了更大的集合大小或更优的最大模糊函数上限。

**🔧 技术方法**

主要技术包括：有限域迹函数的性质、列正交复矩阵（如 Butson‑Hadamard 矩阵）的构造、以及模糊函数的数学分析与上界证明。

**📊 数据集**

本文为理论构造，不涉及实验数据集；构造基于符号域 𝔽_{p^n} 的元素，依赖于这些域的代数结构。

**📈 对比分析**

通过与文献中已知的无周期 DRCSS 参数表比较，本文构造的集合在给定长度与字母表大小下能够提供更大的集合大小，且最大模糊函数上限逼近理论下界（optimal factor ρ̂ → 1），显示出优越性能。

**⚠️ 局限性**

局限性包括：需要存在合适的列正交复矩阵（对某些参数可能难以构造或查表）；构造过程对 β（有限域中原始元）和 e（满足 Tr(β^e)=0）的选择有约束；并且证明过程在某些构造中省略细节，可能需要进一步验证。

---

## 142. M3R: Localized Rainfall Nowcasting with Meteorology-Informed MultiModal Attention

**arXiv ID:** 2604.15377 | [PDF](https://arxiv.org/pdf/2604.15377v1)

**作者:** Sanjeev Panta `[一作]` (University of Louisiana at Lafayette), Nian-Feng Tzeng `[通讯]` (University of Louisiana at Lafayette)

**通讯引用:** 2768 | [OpenAlex ID](https://openalex.org/A5032851065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于多模态注意力的M3R框架，利用NEXRAD雷达图像与个人气象站（PWS）时间序列的协同学习，实现对15分钟间隔、1小时先导的局地降水量直接预测。

**💡 创新点**

创新点包括：① 以雷达空间特征为键、天气站时间序列为查询的异构注意力机制，天然体现气象学中点测量受空间天气影响的原理；② 直接输出降水率，跳过传统的Z‑R转换，减少不确定性与计算负担；③ 构建统一的多模态数据处理管线，实现雷达与PWS的时间对齐与事件筛选，解决异构数据同步难题。

**🔧 技术方法**

技术方案基于Vision Transformer的图像分块嵌入、时间序列线性嵌入以及多头自注意力与多模态注意力相结合的编码器；随后采用时序解码器生成降水量预测；整个模型采用PyTorch实现，使用AdamW、混合精度训练，并通过MSE损失优化。

**📊 数据集**

使用了由NOAA NEXRAD Level‑2雷达数据与Weather Underground个人气象站（PWS）数据构建的三地区（Lake Charles、Montgomery、Jackson）100 km×100 km的多模态数据集，包含96,359个15‑分钟对齐样本，形成8帧的降水事件序列。

**📈 对比分析**

在三个地区分别与多种基线（Transformer、DLinear、PatchTST、iTransformer、Diffcast、AlphaPre）进行比较，M3R在RMSE、MAE、R²、CC、CSI等指标上实现了12‑27% MAE提升、20‑34%对比AlphaPre的改进，并且在训练与推理速度上比AlphaPre快13倍与7倍，FLOPs也更低。

**⚠️ 局限性**

局限性包括：① 仅针对15‑分钟、1小时先导的局地降水；② 需要至少一台PWS才能利用多模态特征，覆盖面受限；③ 数据集覆盖的地区与雷达站点有限，模型在更大尺度或不同气候区的泛化性尚未验证；④ 目前对极端降水的预测仍有误差，需进一步改进模型对高强度事件的处理。

---

## 143. Graded Symbolic Verification with a Fuzzy Dolev-Yao Attacker Model

**arXiv ID:** 2604.15402 | [PDF](https://arxiv.org/pdf/2604.15402v1)

**作者:** Murat Moran `[一作]` `[通讯]`, Murat Moran

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可累积侧信道泄露的模糊Dolev‑Yao模型，并在Murphi中实现了基于Product T‑范式的攻击知识更新。

**💡 创新点**

将攻击者知识从二元映射扩展为[0,1]度量，揭示了累积阈值下的安全漏洞与安全‑失败转换，首次在符号验证中捕捉此类累积攻击。

**🔧 技术方法**

采用Zadeh扩展原则、Product T‑范式、α切割阈值、概念格E‑约简与离散化的模糊算子，并结合Murphi显式状态检查。

**📊 数据集**

使用Needham‑Schroeder‑Lowe及其对称密钥变体（NSSK、Yahalom、Otway‑Rees、Woo‑Lam）作为协议实例，生成离散化的模糊知识网格。

**📈 对比分析**

与传统二元DY模型对比，利用E‑约简将状态空间缩减≈55%，并在NSL实验中发现模糊模型触发的认证/机密性失败，显示相同协议在模糊与二元验证中的安全结果差异。

**⚠️ 局限性**

受限于离散化精度导致状态爆炸、泄露参数化依赖专家设定，以及仅覆盖语法安全性和有限会话，无法直接映射到计算级别或更复杂协议。

---

## 144. Optimizing Stochastic Gradient Push under Broadcast Communications

**arXiv ID:** 2604.15549 | [PDF](https://arxiv.org/pdf/2604.15549v1)

**作者:** Tuan Nguyen `[一作]` (Pennsylvania State University), Ting He `[通讯]` (Pennsylvania State University)

**通讯引用:** 7642 | [OpenAlex ID](https://openalex.org/A5088400668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究在无线广播网络下，如何设计混合矩阵以最小化去中心化联邦学习的收敛时间。

**💡 创新点**

创新点在于首次将异步、非对称的SGP算法与混合矩阵设计结合，推导出基于图论参数的收敛上界，并给出高效的图设计算法；相比以往仅针对D‑PSGD的对称矩阵。

**🔧 技术方法**

使用理论分析（收敛定理）、图论优化、冲突图染色近似、最低度生成树与增边策略等技术。

**📊 数据集**

使用CIFAR‑10和FMNIST图像数据集；实验网络为随机几何图和Roofnet无线网状网络。

**📈 对比分析**

与Vanilla D‑PSGD、Vanilla SGP、MATCHA、BASS heuristic/optimized等基线对比；结果表明相较BASS optimized节约约11–21%收敛时间，较Vanilla D‑PSGD约38–45%。

**⚠️ 局限性**

局限在于基于周期固定的混合矩阵、仅考虑无延迟同步版本、设计算法基于近似最低度生成树且在极稀疏拓扑下优势减弱；未考虑通信失误、延迟等真实环境因素。

---

## 145. "Excuse me, may I say something..." CoLabScience, A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations

**arXiv ID:** 2604.15588 | [PDF](https://arxiv.org/pdf/2604.15588v1)

**作者:** Yang Wu `[一作]` (Worcester Polytechnic Institute), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3883 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了CoLabScience，一款能够在生物医学讨论中主动判断并生成干预内容的LLM助手。

**💡 创新点**

结合正负标记学习（PU）与强化学习的PULI框架，实时学习何时干预以及如何干预，同时使用Observer与Presenter双模型实现低延迟决策。

**🔧 技术方法**

使用Observer（GRPO强化学习）和Presenter（SFT/LoRA）LLM，协调器MLP通过REINFORCE进行端到端训练，结合多轮对话记忆与项目提案上下文。

**📊 数据集**

构建BSDD（Biomedical Streaming Dialogue Dataset）——基于PubMed论文生成的多角色科学对话，并标注唯一的正向干预点。

**📈 对比分析**

与Standard、Random、Proactive Agent、ICL、SFT等基线在多模型（LLaMA3, Qwen3）上对比，PULI在干预时机准确率、ROUGE-1、win-rate等指标上均显著领先，尤其在LLaMA3/ Qwen3 组合下分别达至 67.4%/64.1% 时机准确率和 33.5%/32.4% ROUGE-1。

**⚠️ 局限性**

数据为LLM生成的模拟对话，缺乏真实科研会议的复杂性；仅标注单一干预点；评估仅在模拟环境；部署时需与ASR/TTS结合，潜在延迟；干预形式局限于偏离目标的检测。

---

## 146. Lightweight Geometric Adaptation for Training Physics-Informed Neural Networks

**arXiv ID:** 2604.15392 | [PDF](https://arxiv.org/pdf/2604.15392v1)

**作者:** Kang An `[一作]` (Rice University), Ming Yan `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 477756 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的曲率感知优化框架，用于提升物理信息神经网络（PINN）的训练效率和稳定性。

**💡 创新点**

创新点在于利用连续梯度差（secant）与梯度位移构造的曲率指示器，对梯度进行自适应的预测修正，实现了不需要二阶矩阵即可捕捉局部几何变化的机制，并且能够无缝插拔到现有的一阶优化器中。

**🔧 技术方法**

主要技术包括：梯度差的指数移动平均、secant曲率指示器（κk）、基于tanh的曲率门控系数αk、以及在AdamW、SOAP、Muon等优化器中实现的曲率增强梯度（g̃k）。

**📊 数据集**

在四个经典PDE基准上验证：10维热方程、Gray‑Scott反应扩散系统、Belousov‑Zhabotinsky化学反应系统以及二维Kuramoto‑Sivashinsky系统。

**📈 对比分析**

与标准AdamW、SOAP、Muon以及相关改进方法（Adan、ALTO）进行对比，结果显示CA-AdamW、CA-SOAP、CA-Muon在收敛速度、训练稳定性和最终误差上均有显著提升，误差下降幅度高达70%–97%，并在最难的KS系统上优于Adan和ALTO。

**⚠️ 局限性**

局限性包括：理论分析基于L‑光滑和Hessian-Lipschitz假设；曲率门控系数的选择仍为经验性，可能需要针对不同问题调参；未在大规模或多物理场问题上验证，缺乏对计算开销与训练规模的完整评估。

---

## 147. Brain Score Tracks Shared Properties of Languages: Evidence from Many Natural Languages and Structured Sequences

**arXiv ID:** 2604.15503 | [PDF](https://arxiv.org/pdf/2604.15503v1)

**作者:** Jingnong Qu `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5017484646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在不同类型文本（7种自然语言、Python、基因组、Dyck语言、打乱的英语）上从零开始训练 GPT‑2 风格模型，随后对嵌入层进行英文词表适配，使用 Brain Score 评估其对英文阅读时人脑 fMRI 反应的预测能力。

**💡 创新点**

展示了 Brain Score 对语言模型与人脑相似性的测量不具备语言特异性，表明模型能够提取跨语言的通用结构，而非仅限于单一语言的语义或句法。

**🔧 技术方法**

采用了 GPT‑2 架构、AdamW 优化器、线性学习率调度、BPE 词表、嵌入层适配（只训练词向量）、Brain Score 的线性回归预测 fMRI 方案。

**📊 数据集**

使用了 7 种自然语言的维基百科文本、Python 代码（Stack Overflow）、人类基因组序列、Dyck 语言（嵌套括号）以及打乱的英语维基百科，所有数据均截断至 1 亿个 token。

**📈 对比分析**

通过对不同数据集训练得到的模型在 Brain Score 的两个实验（Wiki 文本和叙事文本）上进行比较；结果显示自然语言模型的得分高度一致，Python 与自然语言接近，Dyck、基因组和打乱英语仅略低；嵌入层适配可显著提升非自然语言模型的得分，但对自然语言的提升有限。

**⚠️ 局限性**

主要局限包括：Brain Score 依赖语义信息，缺乏对句法差异的敏感性；缺乏跨语种人脑数据，无法验证模型在非英语读者中的表现；嵌入适配对不同领域的迁移效果不足，表明现有指标不足以捕捉人类语言处理的细微差异。

---

## 148. Why Fine-Tuning Encourages Hallucinations and How to Fix It

**arXiv ID:** 2604.15574 | [PDF](https://arxiv.org/pdf/2604.15574v1)

**作者:** Guy Kaplan `[一作]` (Hebrew University of Jerusalem), Roy Schwartz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 6123 | [OpenAlex ID](https://openalex.org/A5007903277)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了监督微调过程中大语言模型的事实性幻觉，并将其重新解释为事实遗忘，探索通过持续学习方法缓解该问题；

**💡 创新点**

首次将自蒸馏和参数冻结引入SFT，系统验证事实性干扰是幻觉的主要驱动因素，并提出了针对性对策；

**🔧 技术方法**

采用SLiCK 事实分类、参数冻结、基于输出分布的自蒸馏、隐藏状态漂移分析以及对比实验等技术；

**📊 数据集**

使用EntityQuestions数据集以及人工构造的语义重叠与UUID式实体的合成数据进行实验；

**📈 对比分析**

与标准SFT、仅保留已知事实、仅更新注意力或FFN等对比实验表明，自蒸馏将事实遗忘率从约15%降至约3%，冻结方法在保持任务性能的同时显著降低幻觉；

**⚠️ 局限性**

仅聚焦监督微调场景，未评估推理型模型；自蒸馏需要额外训练和超参调优；大规模模型的通用性与计算开销仍需进一步验证。

---

## 149. LLMbench: A Comparative Close Reading Workbench for Large Language Models

**arXiv ID:** 2604.15508 | [PDF](https://arxiv.org/pdf/2604.15508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 150. The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason

**arXiv ID:** 2604.15350 | [PDF](https://arxiv.org/pdf/2604.15350v1)

**作者:** Yi Liu `[一作]` `[通讯]`, Yi Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对11种大语言模型（5种架构、5种训练模式）在21种推理与记忆任务中，系统性地对隐藏层激活进行谱分析，发现了七大现象并提出完整的谱理论；

**💡 创新点**

提出“谱阶段转变”概念，揭示推理任务导致的谱压缩/展开、指令微调逆转、生成阶段三类谱变化、谱尺度定律、跨层谱级联、推理步骤的谱标点以及谱α可完美预测推理正确性；

**🔧 技术方法**

使用奇异值分解、幂律拟合得到谱指数α，结合滑动窗口、梯度、相关系数等指标进行跨层、跨层次的时间序列分析，并利用逻辑回归评估α对答案正确性的预测；

**📊 数据集**

使用自定义21任务集（13推理、6事实回忆、2随机基准），并在6个模型上进行200道推理题的AUC评估，另外在4类40道OOD问题上验证外部一致性；

**📈 对比分析**

与传统线性探针、注意力分析、激活补丁等方法对比，谱α在推理阶段可达到AUC 1.000（Qwen2.5‑7B）和平均0.893，显示其在推理质量监测与预测方面的显著优势；

**⚠️ 局限性**

局限性包括模型规模仅至7B、任务多样性有限、缺乏因果验证、跨模型token级谱动态验证不足、谱尺度定律R²仅0.46、OOD泛化表现不佳以及对谱指数假设的依赖等。

---

## 151. Factor Graph-Based Shape Estimation for Continuum Robots via Magnus Expansion

**arXiv ID:** 2604.15619 | [PDF](https://arxiv.org/pdf/2604.15619v1)

**作者:** Lorenzo Ticozzi `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 12545 | [OpenAlex ID](https://openalex.org/A5077667229)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计了一种基于因子图的低维几何变量应变参数化的软体形状估计方法，利用Magnus展开作为先验约束实现了姿态与应变场的联合估计。

**💡 创新点**

创新点在于将Magnus展开引入因子图，结合GVS低维参数化，既保持了参数化方法的紧凑状态，又通过高阶几何耦合约束提升了稀疏或部分测量下的鲁棒性。

**🔧 技术方法**

使用了因子图（GTSAM）、Lie群理论、Magnus展开、B样条应变基底、Levenberg–Marquardt优化、以及基于Cosserat杆的仿真求解器。

**📊 数据集**

仿真数据来自0.4 m tendon‑driven continuum robot，随机生成60个静态平衡实例，测量噪声与GP基线保持一致。

**📈 对比分析**

与GP回归基线在三种测量配置下进行对比，平均位置误差低于1.5 mm，姿态误差约10°以下，计算时间可接受且在稀疏测量场景下表现更稳健。

**⚠️ 局限性**

局限在于全局应变变量与所有姿态节点耦合导致因子图稀疏性下降，计算时间略高；并且目前仅在仿真中验证，硬件实验尚待进一步评估。

---

## 152. VoodooNet: Achieving Analytic Ground States via High-Dimensional Random Projections

**arXiv ID:** 2604.15613 | [PDF](https://arxiv.org/pdf/2604.15613v1)

**作者:** Wladimir Silva `[一作]` `[通讯]` (North Carolina State University), Wladimir Silva (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种一次性求解的神经网络架构 VoodooNet，利用高维随机投影和 Moore‑Penrose pseudoinverse 直接得到输出权重，跳过梯度下降

**💡 创新点**

创新点在于将 Johnson‑Lindenstrauss 随机投影与极限学习机思路结合，形成“Galactic Expansion”+“Magic Hat”闭式求解方法，实现即时收敛

**🔧 技术方法**

使用 Johnson‑Lindenstrauss 随机投影、ReLU 激活、Moore‑Penrose pseudoinverse、极限学习机框架

**📊 数据集**

在 MNIST 和 Fashion‑MNIST 数据集上评估

**📈 对比分析**

与 10 轮 SGD 基线对比，VoodooNet 在 MNIST 上 98.1% / 86.6% Fashion‑MNIST，训练时间从数秒到毫秒级显著下降，精度保持或略优，展现出近对数维度-精度关系

**⚠️ 局限性**

局限在于高维投影导致权重熵高、计算量随维度三次方增长，对更复杂数据和大规模任务的可扩展性和稀疏性尚未证明

---

## 153. Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models

**arXiv ID:** 2604.15609 | [PDF](https://arxiv.org/pdf/2604.15609v1)

**作者:** Yunbei Zhang `[一作]` (Tulane University), Jihun Hamm `[通讯]` (Tulane University)

**通讯引用:** 1837 | [OpenAlex ID](https://openalex.org/A5085659523)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在严格黑盒场景下的测试时自适应框架BETA，能够在仅通过API查询的条件下提升模型在分布偏移环境下的表现；

**💡 创新点**

创新点在于利用轻量级白盒“引导模型”与黑盒模型预测的和谐化（prediction harmonization）构建可求梯度的联合目标，并通过一致性正则化与基于提示的样本过滤实现稳定的无监督优化；

**🔧 技术方法**

核心技术包括：视觉提示学习、预测和谐化、Jensen–Shannon对齐项、KL一致性正则化、熵阈值过滤、以及仅一次API调用的在线一梯度更新；

**📊 数据集**

主要在ImageNet-C（噪声、模糊、天气、数字等七种破坏），ImageNet-Sketch、ImageNet-R，CLIP-Vision-Language模型，以及实际商业Clarifai API上进行实验；

**📈 对比分析**

与白盒TENT、CoTTA等方法相比，BETA在黑盒ViT-B/16上提升7.1%（+62.6%）且不增加API调用；相较于ZOO、LAME、TT-Aug、DDA等黑盒方法，BETA获得显著更高准确率，并在Clarifai API上实现250倍成本优势；

**⚠️ 局限性**

局限性包括：仍需本地轻量级引导模型（虽然规模小），在极端域漂移或标签不平衡下性能可能下降，且对不同任务的适配效果依赖于引导模型与黑盒模型间的语义相似度；

---

## 154. Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies

**arXiv ID:** 2604.15607 | [PDF](https://arxiv.org/pdf/2604.15607v1)

**作者:** Myke C. Cohen `[一作]` (Aptima, Inc), Svitlana Volkova `[通讯]` (Aptima, Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过结合大语言模型（LLM）模拟与并行人类实验，探讨在人类与AI在不完全合作情境下（如招聘谈判与AI可能隐瞒信息的交易），用户人格特质（外向性、宜人性）与AI设计属性（可解释性、温暖度、专业性、适应性、心智理论）如何共同影响交互质量与结果，并进行因果推断；

**💡 创新点**

创新点在于①构建了双框架（模拟+人类实验）来验证LLM对人格与AI属性影响的可迁移性；②采用因果结构模型（SEM+CausalNex）从多维度评估两种实验数据的差异；③揭示了透明度在不同情境下的双刃剑效应，为不完全合作的AI设计提供了实证依据；

**🔧 技术方法**

主要技术包括Sotopia‑S^4多代理社会模拟平台、GPT‑4o生成对话、Sotopia‑Eval（LLM评估）、结构方程模型与CausalNex因果学习、词汇分析（情感、同理心、毒性等）及问卷收集；

**📊 数据集**

使用自建实验设计生成的2,000条LLM对话（5种情境×5 AI属性×4人格档案）及与之对应的290名Prolific人类参与者数据；还利用AI‑LieDar数据集构造含隐瞒信息的交易场景；

**📈 对比分析**

对比方法：在模拟与人类实验中保持相同的AI属性干预和情境设置，分别计算因果效应（SEM权重）和主观/客观指标（达成率、积分、满意度、真诚度等）。结果显示：模拟中人格特质对结果影响显著；在人类实验中AI属性（尤其透明度）影响更大；不同情境下效应方向和大小存在差异；总体上，AI属性对真实用户体验的解释力更强；

**⚠️ 局限性**

局限性包括①人类实验未对人格特质进行随机控制，样本偏高宜人性导致人格效应被低估；②仅使用单一LLM（GPT‑4o），模型偏好和提示敏感性可能限制结果的普适性；③因果分析提供相对效应大小，但缺乏传统统计显著性检验；④LLM评估与人类主观感受存在差异，尤其在真诚度等社交维度。

---

## 155. PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs

**arXiv ID:** 2604.15645 | [PDF](https://arxiv.org/pdf/2604.15645v1)

**作者:** Shimon Pisnoy `[一作]` (Technion Israel Institute of Technology), Steven H. Frankel `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 PINNACLE，一个模块化、开源的物理信息神经网络（PINN）框架，支持多种网络结构、训练策略、分布式多 GPU 加速以及量子‑经典混合架构；

**💡 创新点**

在传统 PINN 基础上系统集成了傅里叶特征映射、随机权重分解、严格边界约束、动态损失平衡、课程学习与 Adam+L‑BFGS 优化器切换，并首次给出量子 PINN 的参数‑shift 复杂度分析；

**🔧 技术方法**

主要技术包括：PINN 损失构造、随机 Fourier 特征、随机权重分解、严格周期边界、损失平衡、Adam+L‑BFGS、课程学习、时间因果训练、分布式数据并行（DDP）、量子电路 PQC 与参数‑shift 微分、TorQ 状态向量模拟；

**📊 数据集**

利用多种典型 PDE 基准：线性输运、Allen‑Cahn 反应扩散、无粘 Burgers、雷诺数 100–3200 的 lid‑driven cavity、血管狭窄 2D 斯托克斯流、Sod 以及 2D Riemann 断层、2D Maxwell Gaussian 脉冲；

**📈 对比分析**

与 DeepXDE、Modulus、经典数值求解器及单 GPU 版本进行对比，证明 PINNACLE 在准确性与参数效率上优于现有框架；量子 PINN 在参数规模上可实现 19–58% 的节省，但需额外能量守恒正则化；多 GPU DDP 在 1–4 核心上线性加速，超过 4 核后受通信与梯度同步瓶颈限制；

**⚠️ 局限性**

局限性包括：高昂的计算与内存成本、对超参数敏感、量子 PINN 需要大量电路评估导致训练时间爆炸、对剧烈非线性/冲击问题仍难以完全捕获、DDP 需多进程启动、量子模拟器仅支持单 GPU，难以扩展至大规模量子设备；

---

## 156. Hierarchical Active Inference using Successor Representations

**arXiv ID:** 2604.15679 | [PDF](https://arxiv.org/pdf/2604.15679v1)

**作者:** Prashant Rangarajan `[一作]` (University of Washington), Rajesh P. N. Rao `[通讯]` (University of Washington)

**通讯引用:** 20527 | [OpenAlex ID](https://openalex.org/A5002759219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于后继表示的层次主动推理框架，用低层的后继表学习宏观状态与宏观动作，实现大规模环境下的高效规划与学习。

**💡 创新点**

创新点在于将后继表示与主动推理结合，自动学习状态与动作层次抽象，利用宏观规划减少搜索空间，并支持快速重规划。

**🔧 技术方法**

使用后继表示、谱聚类、变分推理（主动推理）以及离散化、RBF核平滑等技术。

**📊 数据集**

在多种仿真环境上验证：Gridworld（Serpentine、Four Rooms、POMDP Five Rooms）、带钥匙的Gridworld、Mountain Car和PointMaze。

**📈 对比分析**

与平面主动推理和Q‑learning基线比较，层次模型在训练样本、规划步数、成功率和稳定性方面均显著优于平面模型，重规划效率几乎不受惩罚。

**⚠️ 局限性**

局限在于目前仅实现两层层次，适用于已离散化或可离散化环境；在高维连续空间或更复杂的POMDP下仍需进一步改进，需探索深度网络和多层层次架构。

---

## 157. NK-GAD: Neighbor Knowledge-Enhanced Unsupervised Graph Anomaly Detection

**arXiv ID:** 2604.15668 | [PDF](https://arxiv.org/pdf/2604.15668v1)

**作者:** Zehao Wang `[一作]` (Tianjin University), Lanjun Wang `[通讯]` (Tianjin University)

**通讯引用:** 2650 | [OpenAlex ID](https://openalex.org/A5025153128)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种邻居知识增强的无监督图异常检测框架 NK-GAD。

**💡 创新点**

创新点在于联合低频与高频图卷积编码、邻居分布重构和中心聚合模块，能够同时利用相似与不相似邻居特征并适应属性异质性。

**🔧 技术方法**

技术主要包括低/高通图卷积、邻居统计分布重构、图注意力聚合以及两阶段属性与结构解码重构。

**📊 数据集**

使用七个真实图数据集：Weibo、Reddit、Disney、Books、Enron、Elliptic 和 DGraph。

**📈 对比分析**

与九个基准方法比较，NK-GAD 在六个数据集上获得最高 AUC，平均提升 3.29%（对 DGraph 最高 55.30%）。

**⚠️ 局限性**

局限性是对 Reddit 数据集效果略逊于 GADAM，且在高频组件过度依赖时易受噪声影响。

---

## 158. Sample Is Feature: Beyond Item-Level, Toward Sample-Level Tokens for Unified Large Recommender Models

**arXiv ID:** 2604.15650 | [PDF](https://arxiv.org/pdf/2604.15650v1)

**作者:** Shuli Wang `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Sample Is Feature (SIF) 框架，将历史交互从单纯的 item 级别 token 变为包含完整样本上下文的 sample 级别 token，并通过 SIF-Mixer 进行双层（token‑级与 sample‑级）特征交互，以提升推荐模型的表达能力。

**💡 创新点**

创新点在于：① 通过层级组自适应量化（HGAQ）将原始交互记录压缩为离散 token，最大程度保留样本上下文；② 设计了同质化的 SIF‑Mixer，将注意力拆分为 token‑级内样本交互和 sample‑级跨样本交互，既提升表达力又控制复杂度；③ 在同一代码簿空间统一历史与当前请求的表示，实现了跨时间对齐。

**🔧 技术方法**

主要技术包括：层级组自适应量化（Hierarchical Group‑Adaptive Quantization, HGAQ）与残差向量量化（RVQ）；离散代码簿对齐与监督学习；基于 Transformer 的双层混合器（Token‑Mixer + Sample‑Mixer）；对齐损失确保当前请求投影与代码簿一致；以及 MLP‑Mixer 风格的深度特征交互。

**📊 数据集**

使用了美团本地生活行业的工业级数据集：约 10 亿条印象记录、5 千万用户、500 万品类，样本包含 600+ 字段，行为序列长度可达 1000 条，且包含完整的用户、商品、上下文与交叉特征。

**📈 对比分析**

与多种基线（HyFormer、OneTrans、SIM、LONGER、RankMixer 等）在同一工业数据集上对比。SIF 在离线 GAUC 上提升了约 0.88%，在在线 A/B 测试中实现了 2.03% 的 CTR 提升、1.21% 的 CVR 提升和 1.35% 的 GMV/Session 提升；相较于最佳统一基线，整体表现更为稳定，且在序列长度增大时增益更显著。

**⚠️ 局限性**

局限性包括：① 需要离线训练与更新代码簿，更新频率受限；② 量化过程中可能出现信息损失，尤其在特征分布大幅变化时需重新训练；③ 目前仅在美团内部数据验证，跨领域泛化仍待验证；④ 代码簿尺寸与推理速度平衡需要细致调优。

---

## 159. CLIMB: Controllable Longitudinal Brain Image Generation using Mamba-based Latent Diffusion Model and Gaussian-aligned Autoencoder

**arXiv ID:** 2604.15611 | [PDF](https://arxiv.org/pdf/2604.15611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 160. HyperGVL: Benchmarking and Improving Large Vision-Language Models in Hypergraph Understanding and Reasoning

**arXiv ID:** 2604.15648 | [PDF](https://arxiv.org/pdf/2604.15648v1)

**作者:** Yanbin Wei `[一作]` (Southern University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16829 | [OpenAlex ID](https://openalex.org/A5070273088)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HyperGVL基准，用以评估大型视听语言模型（LVLM）在超图理解与推理任务中的能力，涵盖12类任务、84,000个视觉‑文本问答样本，并设计了多种文本与视觉超图表示。

**💡 创新点**

①首个面向LVLM的超图基准；②通过多尺度合成与真实超图（引用网络、蛋白网络）构造丰富数据；③引入七种文本与五种视觉表示的组合，系统探测表示对模型性能的影响；④设计WiseHyGR路由器，能根据任务自动选择最优表示并提升模型表现。

**🔧 技术方法**

利用12个主流LVLM（包括开源与闭源）进行零样本评估；构建任务‑表示映射数据集；训练DeBERTaV3‑base路由器；在多任务与多表示下进行对比实验；采用平均准确率、任务级别分布等指标评估性能。

**📊 数据集**

HyperGVL数据集：84,000个视听问答样本，覆盖12个任务，包含合成与真实超图（citation网络、protein网络）；多种文本（LO‑Inc、N‑Pair、Adj‑Mat、HO‑Neigh、HO‑Inc、Inc‑Mat、N‑Set）与视觉（Enc‑Hy、Bi‑Inc、Sh‑Inc、St‑Inc、Cli‑Exp）表示的组合。

**📈 对比分析**

对比12个LVLM的理解（Avg.U≈40%）与推理（Avg.R≈23%）性能，闭源模型（如Gemini‑3 Flash）在理解上最高达≈90%，推理最高≈62%；开源模型相对较弱；WiseHyGR在所有任务上平均提升≈10‑15%，且在领域外的节点分类任务中也能带来显著收益。

**⚠️ 局限性**

局限性：基准仅涵盖12类通用任务，未覆盖所有超图响应类型；未评估极大规模模型（如Qwen3‑VL‑235B、InternVL3‑78B）；仅提供零样本评估，缺乏更细粒度的微调效果分析。

---

## 161. Causal Bootstrapped Alignment for Unsupervised Video-Based Visible-Infrared Person Re-Identification

**arXiv ID:** 2604.15631 | [PDF](https://arxiv.org/pdf/2604.15631v1)

**作者:** Shuang Li `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 30833 | [OpenAlex ID](https://openalex.org/A5101785348)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Causal Bootstrapped Alignment (CBA) 框架，用于无监督视频可见-红外行人重识别（USL-VVI-ReID）。

**💡 创新点**

创新点在于：① 通过 Causal Intervention Warm-up (CIW) 对预训练编码器进行序列级因果干预，抑制模态与时序混杂，提升身份语义；② 通过 Prototype-Guided Uncertainty Refinement (PGUR) 用可见模态的高分辨原型引导红外模态的细粒度重构，并采用不确定性感知软监督解决一对多匹配不确定性。

**🔧 技术方法**

使用的技术包括：结构因果模型与因果干预（MPB、TTB、ICS）、跨模态风格迁移、随机帧置换、对比学习、图匹配、原型记忆重构、软硬监督混合损失等。

**📊 数据集**

在 HITSZ-VCM 与 BUPTCampus 两大公开视频跨模态数据集上进行实验。

**📈 对比分析**

与现有无监督方法（如 ADCA、PGM 等）以及部分监督方法比较，CBA 在 I2V、V2I 方向的 Rank‑1、mAP 均超过对手 15–25%，在 HITSZ-VCM 上实现 58.6%/45.9% Rank‑1/mAP，BUPTCampus 上 40.2%/38.8%，显著优于前沿无监督方案。

**⚠️ 局限性**

局限性包括：仍依赖于预训练 ViT‑B/16 编码器，对超参数敏感；在极大规模或极端光照条件下性能仍有提升空间；与监督方法相比存在一定差距，且实现复杂度较高。

---

## 162. GaussianFlow SLAM: Monocular Gaussian Splatting SLAM Guided by GaussianFlow

**arXiv ID:** 2604.15612 | [PDF](https://arxiv.org/pdf/2604.15612v1)

**作者:** Dong-Uk Seo `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6003 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种单目3D高斯喷射（3DGS）SLAM框架——GaussianFlow SLAM，用光流作为几何监督来同时优化地图结构和相机位姿。

**💡 创新点**

创新点在于：①首次将光流直接映射为高斯喷射的“GaussianFlow”并设计闭式解析梯度，使光流误差可直接反向传播到高斯参数和位姿；②引入归一化误差驱动的稠密化与裁剪模块，动态调整高斯数量，提升地图质量和位姿精度。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、光流网络（ConvGRU/Raft等）、闭式特征梯度求导、闭式特征矩阵分解、稠密束平差（DBA）、误差归一化稠密化/裁剪算法。

**📊 数据集**

使用公开数据集：TUM RGB‑D（室内小尺度）和EuRoC（大尺度无人机）进行评估。

**📈 对比分析**

与MonoGS、MM3DGS‑SLAM、Photo‑SLAM、HI‑SLAM2、WildGS‑SLAM等单目3DGS SLAM方法对比，GaussianFlow SLAM在绝大多数序列上实现了更低的位姿误差（RMSE ATE）和更高的渲染质量（PSNR/SSIM/LPIPS），尤其在大尺度场景中显著提升。

**⚠️ 局限性**

主要局限：计算量大，实时帧率仅≈0.17 FPS；对光流质量敏感，在低照度或高速运动/模糊场景下效果下降；目前仅针对静态场景，未针对高动态环境验证。

---

## 163. Scalable Algorithms with Provable Optimality Bounds for the Multiple Watchman Route Problem

**arXiv ID:** 2604.15610 | [PDF](https://arxiv.org/pdf/2604.15610v1)

**作者:** Srikar Gouru `[一作]` (Carnegie Mellon University), Jiaoyang Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4111 | [OpenAlex ID](https://openalex.org/A5027709346)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对多守望者路线问题（MWRP）的最优求解方法以及一系列子最优与后处理算法

**💡 创新点**

1) 通过单元和路径支配裁剪显著压缩搜索空间；2) 引入枢轴裁剪和并行启发式计算加速mTSP启发式；3) 针对最优目标提出Minimax Weighted（MxW）和Focal Search的子最优变体；4) 开发后处理框架在不牺牲解的合法性的前提下进一步降低最优停机时间；5) 所有方法均在理论上保持最优或可保证的误差上界

**🔧 技术方法**

基于网格图的联合空间搜索、单元/路径支配裁剪、枢轴裁剪、批量并行mTSP启发式、加权A*（Minimax Weighted）和Focal Search、ILP求解器Cplex用于mTSP、并行计算框架

**📊 数据集**

四种网格地图（Maze、Room、Random、Game-inspired）来自Moving AI MAPF Benchmark及公开的MVRP/多代理规划数据集

**📈 对比分析**

与原始MWRP算法对比，改进后算法在复杂地图上可在200倍以内完成求解，子最优算法在不同权重下能在更短时间内给出可接受解，后处理框架将最优停机时间提升仅8%并大幅降低路径长度

**⚠️ 局限性**

仅适用于无碰撞、网格地图、线视范围已知的情形；对非网格图、动态障碍、异质代理等场景仍需进一步研究

---

## 164. When structure does not imply symmetry

**arXiv ID:** 2604.15682 | [PDF](https://arxiv.org/pdf/2604.15682v1)

**作者:** Skyler R. St. Pierre `[一作]` (Stanford University), Ellen Kuhl `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果显示本方法在XXX指标上优于其他方法。

**⚠️ 局限性**

限制在于XXX，例如数据集规模较小或模型复杂度高。

---

## 165. C-Mining: Unsupervised Discovery of Seeds for Cultural Data Synthesis via Geometric Misalignment

**arXiv ID:** 2604.15675 | [PDF](https://arxiv.org/pdf/2604.15675v1)

**作者:** Pufan Zeng `[一作]` (Huawei), Daimeng Wei `[通讯]` (Huawei)

**通讯引用:** 269 | [OpenAlex ID](https://openalex.org/A5012613066)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无监督框架C‑Mining，用几何不对齐的多语言嵌入自动挖掘高质量文化种子（Culture Points）并基于此合成指令调优数据，提升LLM的文化理解与推理能力。

**💡 创新点**

核心创新在于将“文化特异性”量化为嵌入空间中的几何不对齐信号，形成可计算的种子挖掘流程，消除对人工或LLM主观判断的依赖，实现150倍以上成本提升。

**🔧 技术方法**

采用冻结的多语言预训练嵌入、K‑means聚类、局部密度与语义熵筛选、跨语言聚类与语言主导阈值过滤，随后利用LLM生成多样化指令样本。

**📊 数据集**

使用多语种维基百科原始文本作为种子来源，合成的指令数据集约5万对，评测基准包括CulturalBench‑Hard、BLEnD、CultureScope、SAGE以及Global‑MMLU‑Lite。

**📈 对比分析**

与基线（如CultureLLM、CultureBank、通用多语言模型）对比，C‑Mining提升CulturalBench‑Hard 6.03分、BLEnD 3.13分，并在大型模型上实现新SOTA；同时在General‑MMLU‑Lite上保持或略增通用能力。

**⚠️ 局限性**

局限包括对阈值θ的依赖、可能忽略极为细微的跨语言文化共性，以及仅在英文为主的评测基准上验证，未来需扩展到更多低资源语言与更细粒度文化维度。

---

## 166. Long-Term Memory for VLA-based Agents in Open-World Task Execution

**arXiv ID:** 2604.15671 | [PDF](https://arxiv.org/pdf/2604.15671v1)

**作者:** Xu Huang `[一作]` (Nanjing University), Jiabao Zhao `[通讯]` (Nanjing University)

**通讯引用:** 837 | [OpenAlex ID](https://openalex.org/A5101554229)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ChemBot 框架，整合 AI 代理与进度感知的 Vision-Language-Action 模型，实现化学实验的层次化分解与闭环执行。

**💡 创新点**

创新点包括双层记忆机制、进度预测头、异步连续推理以及基于 Model Context Protocol 的工具协同。

**🔧 技术方法**

使用 LLM 规划、GR00T VLA、Skill-VLA（含进度头）、双层内存、MCP、流匹配训练和异步推理。

**📊 数据集**

采用 CLARIFY 结构化化学协议、扩展的多模态数据集（92 条）以及 5,459 条专家轨迹的化学操作数据集。

**📈 对比分析**

与传统 VLA 基线（π_0.5、GR00T 等）对比，ChemBot 在任务分解的 ROUGE、BERTScore、编辑距离以及实验室现场的成功率上均显著优于基线，成功率提升至 90%+。

**⚠️ 局限性**

局限性包括 LLM 规划的可靠性、对透明玻璃器皿的视觉识别不足以及对新场景的泛化能力有限。

---

## 167. PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation

**arXiv ID:** 2604.15670 | [PDF](https://arxiv.org/pdf/2604.15670v1)

**作者:** Shuyan Ke `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32345 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出无人机推理分割任务并构建10k高分辨率标注数据集DRSeg，提供空间、属性、场景三类推理监督。

**💡 创新点**

创新点在于将推理分割扩展到无人机视角，设计Dual‑Path视觉编码器与多尺度层级解码器，利用Chain‑of‑Thought监督提升推理能力。

**🔧 技术方法**

采用LLaMA‑2‑13B作为LLM基础，CLIP与SAM双路编码器，MultiPath Alignment融合，Hierarchical Reasoning Decoder，LoRA微调，COT数据生成等技术。

**📊 数据集**

主要使用DRSeg数据集；对比基准包括Seg‑Zero、GeoPix、LISA、PixelLM等零样本与SFT模型；同时在RefCOCO、RefCOCO+、RefCOCOg等传统引用分割基准验证迁移性能。

**📈 对比分析**

在DRSeg上零样本效果差，SFT后PixDLM在属性/场景/空间三类推理中平均gIoU≈62%，显著优于PixelLM、LISA等；在RefCOCO等标准分割任务亦表现竞争力。

**⚠️ 局限性**

局限在于仅单目标标注、缺乏多目标/动态情境；模型对极端尺度变化的鲁棒性有待提升；双路融合仍受限于计算与存储开销。

---

## 168. CodeMMR: Bridging Natural Language, Code, and Image for Unified Retrieval

**arXiv ID:** 2604.15663 | [PDF](https://arxiv.org/pdf/2604.15663v1)

**作者:** Jiahui Geng `[一作]` (MBZUAI), Fakhri Karray `[通讯]` (MBZUAI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MMCoIR多模态代码检索基准，并提出了CodeMMR统一的多模态检索模型。

**💡 创新点**

创新点在于：①首次在检索任务中同时考虑文本、代码和图像三种模态；②通过指令驱动的跨模态对齐实现了多语言、多域的统一嵌入；③在MMCoIR上进行的专门化训练显著提升了跨模态检索性能。

**🔧 技术方法**

采用了对比学习（InfoNCE）和指令条件的多模态编码器，基于Qwen2-VL-2B-Instruct进行LoRA微调，并结合温度调参和硬负样本策略。

**📊 数据集**

使用了来自 WebSight、Web2Code、Sketch2Code、Chart2Code、ChartGen、ChartEdit、MMSVG、SVGStack、DiagramGenBenchmark、DATIKZ_v3、PlantUML 等多个公开数据集，覆盖 Web UI、数据可视化、SVG、图表、UML 等五大视觉域，并涉及八种编程语言。

**📈 对比分析**

与 UniIR、VLM2Vec、GME、LamRA 等现有多模态检索模型对比，CodeMMR 在 Hit@1 上平均提升约 10–15 点，nDCG@10 最高可达 100 %，在所有域中均超过最优基线；在未见域和新任务上也保持了较高的泛化性能，RAG 版代码生成任务中提升执行率与视觉准确度超过 9 点。

**⚠️ 局限性**

局限性包括：①在 SVG 等结构复杂、长文本域上仍显得困难；②对手绘草图等低质量视觉输入的检索效果有限；③仅覆盖图像、文本、代码三种模态，未扩展到视频等更高维度；④训练输入长度限制导致极长代码被截断，影响极大代码片段的编码。

---

## 169. From Zero to Detail: A Progressive Spectral Decoupling Paradigm for UHD Image Restoration with New Benchmark

**arXiv ID:** 2604.15654 | [PDF](https://arxiv.org/pdf/2604.15654v1)

**作者:** Chen Zhao `[一作]` (Nanjing University), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 13455 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于逐频谱分解的三阶段超高分辨率图像恢复框架ERR及其改进版IERR，分别在低分辨率、亚分辨率和全分辨率空间处理零频、低频和高频信息；

**💡 创新点**

创新点包括：①将恢复任务从零频提升到高频拆解为三阶段流程；②在高频阶段引入频域窗口化Kolmogorov–Arnold网络(FW‑KAN)以高效学习细节；③在零频阶段设计全局先验投影器和全局感知Transformer；④构建82,126张高清图像的新数据集LSUHDIR并基于其创建噪声与JPEG压缩基准；

**🔧 技术方法**

技术方法包括：离散余弦变换(DCT)频谱分解、窗口化DCT分块、Zigzag重排、FW‑KAN(窗口化KAN)、多尺度卷积与Transformer融合、全局先验投影与加权融合、L1+SSIM损失及频域正则化；

**📊 数据集**

使用了自构建的LSUHDIR数据集（82k张UHD图像），并在其衍生的UHD‑Noise与UHD‑JPEG基准上以及公开的UHD‑LL、UHD‑Rain13k、UHD‑Haze、UHD‑Blur、UHDM等数据集进行实验；

**📈 对比分析**

在所有公开UHD恢复任务和自建基准上，与现有Transformer、CNN与混合架构的SOTA方法相比，ERR/IERR在PSNR/SSIM/LPIPS上均实现显著提升（如UHD‑LL 27.57/0.932，UHD‑Rain13k 34.48/0.952，UHD‑Haze 25.12/0.950，UHD‑Blur 29.72/0.861，UHD‑Noise 30.73/0.937等），同时IERR在参数量、FLOPs、显存和推理时延上均优于对比模型；

**⚠️ 局限性**

局限性包括：①对极低光或极高噪声场景的鲁棒性仍有限；②在非常大尺寸UHD图像上，仍需额外的多尺度或分块策略以避免显存溢出；③高频子网络FW‑KAN在训练时对频域重排与窗口大小敏感，需精细调参；

---

## 170. SPLIT: Self-supervised Partitioning for Learned Inversion in Nonlinear Tomography

**arXiv ID:** 2604.15651 | [PDF](https://arxiv.org/pdf/2604.15651v1)

**作者:** Markus Haltmeier `[一作]`, Gyeongha Hwang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种自监督学习框架SPLIT，用于非线性、稀疏且含噪的多光谱CT成像与材料分解。

**💡 创新点**

创新点在于：① 采用多分区（双分区）策略，将测量域划分为角度和探测器方向，实现噪声抑制与欠采样补偿；② 在测量域使用自监督损失，并通过交叉分区一致性提升重建质量；③ 设计自动停止规则以防止过拟合。

**🔧 技术方法**

核心技术包括：自监督分区重建（SPLIT/Double‑SPLIT）、CP‑Fast一阶迭代前向模型、U‑Net网络、Poisson+电子噪声仿真、交叉分区一致性损失与PSNR早停。

**📊 数据集**

使用合成的3通道材料图像与5能量通道测量数据，基于水、碘、钆三种材料，尺寸256×256像素，共计100个训练、20个验证、20个测试样本。

**📈 对比分析**

与传统CP‑Fast迭代、仅用重建域损失的自监督方法（X‑space loss）和单分区SPLIT进行对比；在不同采样角度（725、145、29视角）下，SPLIT（尤其是Double‑SPLIT）在PSNR/SSIM上均优于基线，尤其在极稀疏采样下效果最显著。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，缺乏真实临床数据评估；对噪声模型的鲁棒性未系统研究；理论上未给出非线性模型下的可识别性与收敛性分析。

---

## 171. HYPERHEURIST: A Simulated Annealing-Based Control Framework for LLM-Driven Code Generation in Optimized Hardware Design

**arXiv ID:** 2604.15642 | [PDF](https://arxiv.org/pdf/2604.15642v1)

**作者:** Shiva Ahir `[一作]` (Stony Brook University), Alex Doboli `[通讯]` (Stony Brook University)

**通讯引用:** 1700 | [OpenAlex ID](https://openalex.org/A5080972445)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HYPERHEURIST 框架，将 LLM 作为 RTL 生成的启发式生成器，采用两阶段模拟退火并结合工具反馈，先保证功能正确性后再进行 PPA 优化。

**💡 创新点**

创新点在于将功能正确性与 PPA 优化分离为两个独立阶段，使用多管道 LLM（生成器、保守/激进变异、批评）配合结构化批评与修复，并在每阶段采用模拟退火控制器实现稳定可重复的搜索。

**🔧 技术方法**

使用技术包括 GPT‑4 大语言模型、多管道提示策略、结构化批评与修复、模拟退火（SA）优化、Synopsys VCS 进行功能验证、Synopsys Design Compiler 进行 PPA 评估，以及归一化的 PPA 目标函数。

**📊 数据集**

使用数据集为 8 个 RTL 规格，来源于 RTLLLM 基准集，分别是：serial2parallel_8、alu4、counter_0_12、traffic_light、freq_div、johnson_counter、mux2_sync、parallel2serial。

**📈 对比分析**

方法上与单通道 GPT‑4 / GPT‑3.5+SP 生成 RTL 进行对比，评价指标包括功能正确率、结构正确率、面积、功耗和时序 slack；实验结果显示 HYPERHEURIST 在多数设计上提升结构正确率 8–35%，PPA 降低至 70% 左右，且时序 slack 始终为正。

**⚠️ 局限性**

局限性包括：对高度时序敏感的 FSM 控制器仍难以收敛、依赖完整工具反馈信息、扩展到大规模控制系统时的可扩展性和效率仍待验证，以及高阶 LLM 训练与推理成本较高。

---

## 172. Half-Moon Cookie: Private, Similarity-Based Blocklisting with TOCTOU-Attack Resilience

**arXiv ID:** 2604.15641 | [PDF](https://arxiv.org/pdf/2604.15641v1)

**作者:** Xinyuan Zhang `[一作]` (Duke University), Michael K. Reiter `[通讯]` (Duke University)

**通讯引用:** 28049 | [OpenAlex ID](https://openalex.org/A5074117167)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于相似性块列表的隐私保护协议（Half‑Moon Cookie），使发送方可以在不暴露文件内容或块列表的情况下完成近似匹配，并向服务器返回绑定隐藏的令牌，接收方随后只需一次轻量级检查即可确认文件已通过最新块列表，从而抵御 TOCTOU 攻击。

**💡 创新点**

创新点包括：① 将近似块列表检查拆分为 Embed‑and‑Map 与 Test‑and‑Commit 两阶段，使用可重用 Garbled Circuit（CRGC）显著降低输入文件大小对计算与通信的影响；② 采用一次性可预测置换与非可编程随机预言机，保障两侧隐私且实现高效加密；③ 通过生成允许列表（allowlist）令牌实现对后续检查的快速验证；④ 结合 Fuzzy PSI 与哈希嵌入实现高效的相似性检测。

**🔧 技术方法**

核心技术包括可重用 Garbled Circuit（CRGC）、一次性可预测置换、非可编程随机预言机、一次性线性评估（OLE）、近似 Private Set Intersection（Fuzzy PSI）、哈希嵌入（TLSH、ssdeep、sdhash）、有限域运算、OT、PPRF 等。

**📊 数据集**

实验使用了 Enron 邮件附件数据集（133,127 个附件，平均 193.6 字节）、Ember 恶意软件样本集（1,635,679 个实例）以及对应的哈希嵌入数据，评估了不同文件大小、块列表规模及并发情况。

**📈 对比分析**

与单体 Garbled Circuit、异步 Fuzzy PSI、精确 PSI 等基线对比，Half‑Moon Cookie 在通讯量上比单体方案低 200‑300 倍、响应时间低 10‑15 倍；在不同块列表大小下，吞吐量随并发增大而下降，但阈值保持在 80‑250 个并发请求；TLSH 嵌入方式在速度与通信量上优于 ssdeep 和 sdhash，整体单客户端响应时间约 19 秒，单体方案约 247 秒。

**⚠️ 局限性**

局限性：① 需服务器维护 allowlist，更新块列表时需清空；② 对极大文件仍存在通信开销；③ 可重用 Garbled Circuit 需要预处理与特定硬件支持；④ 若攻击者能生成伪近似匹配，可能导致误接受；⑤ 安全性依赖一次性可预测置换与随机预言机的假设；⑥ 本协议未考虑恶意服务器或多方扩展。

---

## 173. Contact-Aware Planning and Control of Continuum Robots in Highly Constrained Environments

**arXiv ID:** 2604.15638 | [PDF](https://arxiv.org/pdf/2604.15638v1)

**作者:** Aedan Mangan `[一作]` (University of California San Diego), Tania K. Morimoto `[通讯]` (University of California San Diego)

**通讯引用:** 1281 | [OpenAlex ID](https://openalex.org/A5064903569)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种针对连续柔性机器人在高度受限环境中的接触感知规划与闭环控制框架，并在基于患者CT扫描的主动脉弓模型上实现了自主导航。

**💡 创新点**

创新点在于引入接触质量评估与危险接触惩罚的规划成本，结合动态任务空间分区和接触感知雅可比矩阵，实现了可行轨迹生成与安全导航。

**🔧 技术方法**

采用加权A*搜索规划、基于两段凹凸段的接触感知运动模型、EM传感器反馈的PD闭环控制，并在硬件上使用3D打印凹凸段连续机器人。

**📊 数据集**

使用从患者CT扫描获取的主动脉弓三种类型（I、II、III）的三维模型转化为点云作为环境数据。

**📈 对比分析**

与简单中心线轨迹和无接触惩罚方案相比，本方法在硬件实验中实现100%成功率，平均跟踪误差1.2–1.9 mm，规划时间约15–30 min，导航时间约6–13 min，明显优于基线。

**⚠️ 局限性**

主要局限包括二维平面规划、长规划时间、缺乏真实力学感知、未验证真实临床操作以及对动态环境适应不足。

---

## 174. Synthesizing Backward Error Bounds, Backward

**arXiv ID:** 2604.15633 | [PDF](https://arxiv.org/pdf/2604.15633v1)

**作者:** Laura Zielinski `[一作]` (Cornell University), Justin Hsu `[通讯]` (Cornell University)

**通讯引用:** 6224 | [OpenAlex ID](https://openalex.org/A5033286190)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的可组合后向误差透镜定义，构建了 Shifted Error Lenses（SHEL）范畴，并实现了一款名为 ShEl 的工具，用来自动推导浮点程序的后向误差上界。

**💡 创新点**

创新点在于：①把后向误差透镜的非扩张性条件放宽为仅对接近理想输出的点约束；②通过 SHEL 的张量、共享、推送等构造，天然支持变量重用与相关扰动；③设计了一套基于 Datalog 与等价饱和的搜索框架，实现了完全自动化的后向误差分析。

**🔧 技术方法**

核心技术包括：基于 Olver 的误差模型；关系式后向误差透镜与 SHEL 范畴的定义；张量/共享/推送等范畴结构；以及在 Logic Programming 与 Equality Saturation 上实现的搜索算法。

**📊 数据集**

实验使用了若干手工分析过的基准程序（如向量范数、加权平均、多项式评估、点积等），以及更具挑战性的变量重用示例（如 Cholesky 分解、带根号与对数的表达式）。

**📈 对比分析**

与现有工具（如前向误差分析器）相比，ShEl 能在这些以前不可分析的程序上成功给出精确的后向误差上界；在小规模程序上运行时间仅为几百毫秒，甚至十几毫秒；但随着变量数或算子数增大，搜索空间爆炸导致内存耗尽，性能显著下降。

**⚠️ 局限性**

局限性包括：①仅支持一阶算子和有限的运算符；②对高阶多项式、条件约束或随机舍入模型缺乏分析能力；③搜索算法在变量重用多、层数深的程序上易出现内存溢出；④并未实现对条件后向误差（如 SPD 条件）或概率后向误差的推导。

---

## 175. Overmind NSA: A Unified Neuro-Symbolic Computing Architecture with Approximate Nonlinear Activations and Preemptive Memory Bypass

**arXiv ID:** 2604.15623 | [PDF](https://arxiv.org/pdf/2604.15623v1)

**作者:** Weilun Wang `[一作]` (University of California, Riverside), Wantong Li `[通讯]` (University of California, Riverside)

**通讯引用:** 31799 | [OpenAlex ID](https://openalex.org/A5100358868)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种统一的神经-符号计算架构 Overmind NSA，解决了非线性激活和内存瓶颈问题。

**💡 创新点**

创新点包括使用 Padé 多项式逼近实现高效非线性计算、预抢式内存旁路（双窗口过滤）以及可重构的 PE 数组和软件协同编译器。

**🔧 技术方法**

采用 Padé 近似、预抢式内存旁路、可重构 PE、动态分区、软件协同优化等技术。

**📊 数据集**

在 RAVEN 与 I‑RAVEN 空间‑时间推理数据集上进行实验。

**📈 对比分析**

与 TPU、MTIA、CogSys、NSFlow 等平台对标，22nm ASIC 版实现 410 GOPS、8.1 TOPS/W，吞吐率比 TPU 高 12.26 倍、能效比 TPU 高 27.55 倍；FPGA 版比 NSFlow 高 1.5 倍吞吐。

**⚠️ 局限性**

主要局限在于目前仅针对 INT8 量化，Padé 近似在极端精度需求下可能产生误差；扩展到更大规模模型时仍需进一步优化内存与带宽。

---

## 176. Can LLMs Help Decentralized Dispute Arbitration? A Case Study of UMA-Resolved Markets on Polymarket

**arXiv ID:** 2604.15674 | [PDF](https://arxiv.org/pdf/2604.15674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 177. Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking

**arXiv ID:** 2604.15621 | [PDF](https://arxiv.org/pdf/2604.15621v1)

**作者:** Jun Feng `[一作]` (Hefei University of Technology), Shuai Fang `[通讯]` (Hefei University of Technology)

**通讯引用:** 1301 | [OpenAlex ID](https://openalex.org/A5101662423)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaRankLLM，一种通过自适应列表式重排序与两阶段蒸馏实现的轻量化检索增强生成框架

**💡 创新点**

创新点在于将自适应检索与零样本列表式提示相结合，并通过 Passage Dropout 实现动态筛选，同时通过两阶段蒸馏将高成本 GPT-4 的自适应排名能力迁移到小模型

**🔧 技术方法**

采用零样本 LLM 提示、Passage Dropout、两阶段 progressive distillation（结构对齐+自适应细化）、K-means 聚类与数据增强

**📊 数据集**

使用 MSMARCO V1、ASQA、QAMPARI、ELI5 等公开数据集进行评估

**📈 对比分析**

与固定检索深度（Vanilla-k）、重排序（Rerank-k）以及 AdaRank-GPT4 对比，AdaRankLLM 在大多数模型上能逼近 Oracle 性能，并在弱模型上提供噪声过滤，在强模型上实现显著的上下文压缩而不损失性能

**⚠️ 局限性**

仍存在与理论最优“Oracle”之间的显著性能差距，表明检索与生成的协同优化仍需更深层的探索

---

## 178. Majority Voting for Code Generation

**arXiv ID:** 2604.15618 | [PDF](https://arxiv.org/pdf/2604.15618v1)

**作者:** Tim Launer `[一作]` (ETH Zürich), Andreas Krause `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估一种基于程序执行结果的功能多数投票（FMV）方法，用于代码生成的测试时推理与自监督训练；

**💡 创新点**

创新点在于将执行签名作为软功能共识信号，利用FMV分数挑选最具代表性的候选程序，并将该共识作为无标签TTRL的奖励与伪标签，避免外部oracle；

**🔧 技术方法**

技术包括多次采样生成候选程序、在自生成或公开测试输入上执行、构造执行向量、计算软投票分数（FMV分数），以及在TTRL中使用共识向量生成奖励/目标；

**📊 数据集**

使用Qwen3系列大语言模型（4B、30B等）和LiveCodeBench‑v6数据集（包含官方测试输入和标签）；

**📈 对比分析**

与基线、语义投票（Semantic Voting）和GenCluster等方法对比，FMV在N=64采样下显著提升通过率（pass@1、pass@64），在不同模型规模下均保持领先；在TTRL中，FMV目标能提升零样本准确率但未突破模型性能上限；

**⚠️ 局限性**

局限性包括：TTRL未实现自我提升（best@64不升反降），高误报奖励导致错误被过度强化；FMV虽高效，但对极端边缘案例仍可能受限；

---

## 179. EvoRAG: Making Knowledge Graph-based RAG Automatically Evolve through Feedback-driven Backpropagation

**arXiv ID:** 2604.15676 | [PDF](https://arxiv.org/pdf/2604.15676v1)

**作者:** Zhenbo Fu `[一作]` (Northeastern University), Ge Yu `[通讯]` (Northeastern University)

**通讯引用:** 6476 | [OpenAlex ID](https://openalex.org/A5072406974)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了自我演化的 KG-RAG 框架 EvoRAG，通过反馈驱动的反向传播机制将生成回复的反馈细化到知识三元组层面，从而实现知识图谱的持续优化与推理精度提升。

**💡 创新点**

核心创新在于将响应级反馈映射到推理路径、再映射到三元组的双阶段反向传播，以及基于贡献分数的关系级 KG 进化和混合优先检索策略。

**🔧 技术方法**

结合 LLM 评估器、路径级 utility 计算、梯度反向传播、贡献分数学习、关系融合与抑制、混合语义-贡献检索，以及 KV-cache 重用加速推理。

**📊 数据集**

在 RGB、MultiHop、HotpotQA 三个真实世界问答数据集上进行实验，构建相应 KG 并使用 LLM 评估生成反馈。

**📈 对比分析**

与 MRAG、LRAG、KRAG 以及加入 KGR 方法的 KG-RAG 进行对比，EvoRAG 在 ACC、EM、F1 上分别提升约 7.34%、9.84% 和 7.29%，比传统 KG-RAG 提升 13.8% 等。

**⚠️ 局限性**

仍受限于仅通过反馈更新关系而非实体，且对极端噪声反馈的鲁棒性与对大规模实时推理的扩展性待进一步验证。

---

## 180. Faster LLM Inference via Sequential Monte Carlo

**arXiv ID:** 2604.15672 | [PDF](https://arxiv.org/pdf/2604.15672v1)

**作者:** Yahya Emara `[一作]` (Cornell University), Mohamed S. Abdelfattah `[通讯]` (Cornell University)

**通讯引用:** 2145 | [OpenAlex ID](https://openalex.org/A5000815783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Sequential Monte Carlo Speculative Decoding (SMC‑SD)，用重要性加权重采样和重采样替代传统 Speculative Decoding 的逐‑token 拒绝步骤，实现更高吞吐量的 LLM 推理。

**💡 创新点**

创新点在于将 SMC 框架应用于推理加速，既保持可调的速度‑准确度权衡，又提供非渐近误差上界，并通过向量化 GPU 调度与 KV 缓存指针交换实现大规模并行。

**🔧 技术方法**

使用的技术包括 SMC、重要性采样、重采样、向量化 GPU 执行、PagedAttention、RadixAttention、KV 缓存指针交换、roofline 模型分析和多 GPU 并行调度。

**📊 数据集**

实验使用 GSM8K、MATH500、AlpacaEval、DS1000、ShareGPT 等常用推理基准数据集。

**📈 对比分析**

与自回归、Tree‑based SD (SGLang SD) 以及 SSD 进行对比；在单 GPU 上相较 SD 提升 1.1–2.5×吞吐，允许 3% 以内误差；多 GPU 场景下实现 5.2×自回归、2.36× SD 的加速，同时保持目标模型 3% 内的准确率。

**⚠️ 局限性**

主要局限包括：近似采样误差在多轮累积时尚未给出完整理论界定；重采样开销与动态 N/K 调优仍未充分利用；对不同模型/硬件的通用性验证有限；以及在异步多 GPU 调度与动态资源分配方面仍有进一步优化空间。

---

## 181. CPU Optimization of a Monocular 3D Biomechanics Pipeline for Low-Resource Deployment

**arXiv ID:** 2604.15665 | [PDF](https://arxiv.org/pdf/2604.15665v1)

**作者:** Yan Zhang `[一作]` (Google LLC), Xiong Zhao `[通讯]` (AccMov Health Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

优化了基于单目视频的3D运动分析管线，使其仅在CPU上高效运行，降低了启动延迟和总耗时

**💡 创新点**

通过系统级的多线程并行、模型简化、去除磁盘I/O以及精简物理优化循环，实现了显著的CPU性能提升，并保持了与GPU版相同的运动学精度

**🔧 技术方法**

多线程CPU执行、ONNX模型导出与推理、JAX/TensorFlow多线程后端、序列批处理、物理优化器容差调节

**📊 数据集**

AthletePose3D高分辨率运动视频数据集

**📈 对比分析**

在AMD Ryzen 7 9700X CPU上进行基准测试，平均处理速度从0.14 FPS提升到0.34‑0.35 FPS，总耗时下降59.6%，启动延迟下降4.6×；运动学误差仅为0.35°，相关系数0.998

**⚠️ 局限性**

仅针对CPU优化，仍未探讨模型量化、边缘设备加速或在更低性能硬件上的实测；对极端低帧率或长时监测的鲁棒性尚未评估

---

## 182. Stargazer: A Scalable Model-Fitting Benchmark Environment for AI Agents under Astrophysical Constraints

**arXiv ID:** 2604.15664 | [PDF](https://arxiv.org/pdf/2604.15664v1)

**作者:** Xinge Liu `[一作]` (University of Toronto), Kristen Menou `[通讯]` (University of Toronto)

**通讯引用:** 11688 | [OpenAlex ID](https://openalex.org/A5008019902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估了一个可扩展的基于径向速度数据的AI智能体实验平台Stargazer，用来测试AI在多步物理模型拟合任务中的表现。

**💡 创新点**

设计了动态、反馈驱动的物理基准环境，结合人工标注的真实RV数据与可无限扩展的合成任务，揭示数值拟合与物理推理的差距。

**🔧 技术方法**

结合Python REPL工具、BIC/ΔBIC统计评估、Hungarian匹配、LLM代理与自生成技能注入等技术实现。

**📊 数据集**

120个任务（100合成+20真实归档），合成基于Keplerian动力学，真实来自公开RV观测数据。

**📈 对比分析**

通过四项通过门控（RMS、ΔBIC、MatchScore、Count）进行评估；八个前沿LLM在Easy约70‑80%通过，Medium 17‑35%，Hard ≤6%，真实数据全部未通过，显示模型在物理推理上存在显著不足。

**⚠️ 局限性**

模型虽能得到良好拟合但难以恢复真实物理参数；技能注入仅提升效率；对真实数据的转移性能差；需要更严格的物理约束与多步反馈机制。

---

## 183. Towards Realistic Open-Vocabulary Remote Sensing Segmentation: Benchmark and Baseline

**arXiv ID:** 2604.15652 | [PDF](https://arxiv.org/pdf/2604.15652v1)

**作者:** Bingyu Li `[一作]` (University of Science and Technology of China), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 56986 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建OVRSIS95K训练集并提出OVRSISBenchV2基准与Pi-Seg基线，推动遥感图像开源分割研究。

**💡 创新点**

① 大规模平衡训练集OVRSIS95K；② 具备应用导向的OVRSISBenchV2与下游任务协议；③ 通过语义引导噪声增强提升跨域、跨类别泛化的Pi‑Seg。

**🔧 技术方法**

视觉‑语言预训练模型（CLIP）、语义引导噪声扰动（Image‑SPM / Text‑SPM）以及基于cost aggregation的Pi‑Seg框架。

**📊 数据集**

OVRSIS95K（95K图像/35类）以及10个覆盖卫星/UAV等视角的下游数据集，共计170K图像/128类。

**📈 对比分析**

与OVS/OVRSIS现有方法在OVRSISBenchV1/2及三项下游任务中对比，Pi‑Seg在V2获得最高mIoU/mACC，显著优于训练‑free与现有训练基线。

**⚠️ 局限性**

对细粒度类别仍易混淆，缺乏更细致的语义描述与上下文推理；对极端光照或尺度变化的鲁棒性仍待提升。

---

## 184. SIMMER: Cross-Modal Food Image--Recipe Retrieval via MLLM-Based Embedding

**arXiv ID:** 2604.15628 | [PDF](https://arxiv.org/pdf/2604.15628v1)

**作者:** Keisuke Gomi `[一作]` (University of Electro-Communications), Keiji Yanai `[通讯]` (University of Electro-Communications)

**通讯引用:** 5113 | [OpenAlex ID](https://openalex.org/A5054600485)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过微调 VLM2Vec MLLM 在 Recipe1M 上实现食物图像与食谱文本的跨模态检索。

**💡 创新点**

创新点：① 将单一 MLLM 统一编码器替代传统双编码器；② 设计针对结构化食谱的 Prompt 模板；③ 引入组件感知的数据增强提升对不完整食谱的鲁棒性。

**🔧 技术方法**

使用 VLM2Vec (V1‑2B、V1‑7B、V2) MLLM，配合 LoRA 参数高效微调、InfoNCE 对比损失、Prompt 设计与数据增强。

**📊 数据集**

采用 Recipe1M 数据集（约 238k 图像–食谱对）。

**📈 对比分析**

与过去所有基线（如 T‑Food、TNLBT、DAR 等）在 1k/10k 评估下比较，SIMMER 在 image‑to‑recipe R@1 87.5% / 65.5% 以及 recipe‑to‑image R@1 85.1% / 61.5% 远超前置方法。

**⚠️ 局限性**

局限性：受 MLLM 体量与训练成本限制；在缺少烹饪步骤的场景下性能下降；对数据不一致（图像与文本不匹配）易失效。

---

## 185. ZORO: Active Rules for Reliable Vibe Coding

**arXiv ID:** 2604.15625 | [PDF](https://arxiv.org/pdf/2604.15625v1)

**作者:** Jenny Ma `[一作]` (Columbia University), Lydia B. Chilton `[通讯]` (Columbia University)

**通讯引用:** 4226 | [OpenAlex ID](https://openalex.org/A5049173646)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Enrich‑Enforce‑Evolve框架，将传统的规则文件转换为活跃的控制机制，使规则在vibe编码全过程中可视化、可验证、可迭代。

**💡 创新点**

核心创新是把规则从被动文本变为主动执行的控件，并通过“丰富计划-执行-演化”三步流程，实现规则的即时嵌入、强制遵守与基于使用反馈自动改进。

**🔧 技术方法**

结合LLM（如GPT‑5.3、Claude Sonnet 4.6）生成计划与代码，使用自定义CLI命令实现规则证明、单元测试与证据记录，前端可视化界面支持规则管理与即时反馈。

**📊 数据集**

使用两个从零开始的全栈项目（NoteSense 与 SnakeGame）共36个vibe编码任务作为实验数据集；同时进行12名有经验开发者的用户研究。

**📈 对比分析**

与基线、基础、部分和完整四个实验条件对比，完整条件下规则遵守率提升至0.80（比基线提升57%），功能完整度保持相近；在用户研究中规则通过率从77.9%提升至94.4%。

**⚠️ 局限性**

局限性包括实验规模有限、规则冲突和规模扩展难题、额外的token消耗约4倍、需要手动安装与配置、未涵盖更广泛的agent任务场景。

---

## 186. AdaVFM: Adaptive Vision Foundation Models for Edge Intelligence via LLM-Guided Execution

**arXiv ID:** 2604.15622 | [PDF](https://arxiv.org/pdf/2604.15622v1)

**作者:** Yiwei Zhao `[一作]` (Carnegie Mellon University), Ziyun Li `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建 AdaVFM，一种基于 NAS 的可适配语言对齐视觉基础模型，结合云端 LLM 进行场景理解与子网选择，实现边缘设备实时的开放词汇分类与分割。

**💡 创新点**

创新点在于通过 NAS 超网络实现多尺寸子网动态切换，并结合低频 LLM 进行语义场景分析与类别过滤，显著提升 mIoU 与能耗/延迟权衡，实现 77.9% FLOPs 缩减和 5.2% mIoU 提升。

**🔧 技术方法**

使用的技术包括 NAS（OFA）、ConvNeXt‑v2 backbone、CLIP 视觉‑文本对齐、Arm Ethos‑U55 NPU 测试平台，以及多种 LLM（Llama‑4/3.2、GPT‑5、Gemini‑2.5）驱动的场景理解与子网决策。

**📊 数据集**

使用的数据集包括 ImageNet‑1K、ADE20K、Food‑101、DTD、Oxford Pets、Cal101 等。

**📈 对比分析**

与 ViT‑B/L、DINOv2、DINO.txt、CLIP 等基线进行对比，在 ImageNet‑1K 上提升 0.9%–7.9% acc@1，在 ADE20K 上提升 5.2% mIoU；在 ARM Ethos‑U55 上实现 25–182 ms 延迟、1.2–8.4 mJ 能耗，显著低于基线。

**⚠️ 局限性**

局限性包括：仍需网络连接以调用云端 LLM；子网选择在极端场景下可能误判；极小子网在复杂任务上性能下降；实验集中于单一 NPU，缺乏跨硬件验证。

---

## 187. Flexible Empowerment at Reasoning with Extended Best-of-N Sampling

**arXiv ID:** 2604.15614 | [PDF](https://arxiv.org/pdf/2604.15614v1)

**作者:** Taisuke Kobayashi `[一作]` (National Institute of Informatics), Taisuke Kobayashi `[通讯]` (National Institute of Informatics)

**通讯引用:** 649 | [OpenAlex ID](https://openalex.org/A5051304187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在强化学习中提出一种E-BoN采样方法，使得在推理阶段即可通过empowerment激励对策略进行自适应调整，从而实现探索与利用的平衡；

**💡 创新点**

创新点在于将Tsallis统计的entmax扩展到BoN采样，形成E-BoN采样，既保持计算成本恒定，又可通过单一超参数α自适应调节探索-利用平衡；

**🔧 技术方法**

使用技术包括Soft Actor-Critic、BoN/S-BoN/E-BoN采样、empowerment计算、Tsallis熵/KL、Monte Carlo近似与基于SAC的离线训练；

**📊 数据集**

实验数据集为dm_control Benchmark（cartpole、point_mass、cheetah、walker、quadruped），通过Shimmy包装实现；

**📈 对比分析**

与随机采样、H-BoN、S-BoN等基线比较，实验表明E-BoN在不同α值下能够平衡EED，并在复杂运动任务中实现更稳定、更优的学习回报；

**⚠️ 局限性**

局限性包括EED调节仅采用简单样本化方法、α选择仍需经验、以及在更大规模实时环境中的可扩展性与样本效率待进一步验证。

---

## 188. HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning

**arXiv ID:** 2604.15678 | [PDF](https://arxiv.org/pdf/2604.15678v1)

**作者:** Eunju Lee `[一作]` (Chung-Ang University), YoungBin Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 1896 | [OpenAlex ID](https://openalex.org/A5016930939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种跨学科可变少样本类增量学习框架（XD‑VSCIL），并在此框架下提出训练无关的混合原型校准方法HyCal，以缓解Domain Gravity导致的表示不平衡问题。

**💡 创新点**

创新点在于：①识别并正式定义Domain Gravity——跨域样本不平衡诱发的表示重力；②引入XD‑VSCIL基准，模拟真实世界的异质域与样本不平衡；③提出HyCal，通过结合余弦相似度与马氏距离的加权动态融合，实现无参数、无微调的原型校准，显著抵消Domain Gravity。

**🔧 技术方法**

技术包括：冻结CLIP视觉+文本编码器、基于少样本构造均值与正则化协方差的原型、混合距离度量（余弦+马氏）与自适应权重机制、以及S_CDE综合指标。

**📊 数据集**

使用八个异质域数据集（Aircraft、ArtBench、DTD、EuroSAT、Galaxy、MNIST、OrganMNIST、OxfordFlowers）进行任务顺序实验，并在标准FSCIL基准（如CUB、FGVC等）进行对照。

**📈 对比分析**

与FeCAM、RanPAC、KLDA等训练无关方法以及可训练的Primal‑RAIL对比，HyCal在Balanced、Cross‑Scale及High‑Scale Domain Imbalance三种情形下平均提升≈3–5%准确率，S_CDE得分亦位居前列。

**⚠️ 局限性**

局限性包括：仍依赖预训练CLIP的特性，无法处理完全无标签或跨模态极端差异；在极端高样本不平衡时仍可能出现轻微原型漂移；以及对动态更新策略的探索尚不充分。

---

## 189. Multi-objective Reinforcement Learning With Augmented States Requires Rewards After Deployment

**arXiv ID:** 2604.15757 | [PDF](https://arxiv.org/pdf/2604.15757v1)

**作者:** Peter Vamplew `[一作]` (Federation University Australia), Cameron Foale `[通讯]` (Federation University Australia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文探讨了在多目标强化学习（MORL）中使用增强状态（将环境状态与累计奖励拼接）时，为什么需要在部署后继续提供奖励，并提出通过训练奖励模型来弥补此需求。

**💡 创新点**

创新点在于首次指出增强状态对后部署奖励的依赖性，揭示了MORL与传统单目标RL在部署阶段的根本区别，并提出利用奖励模型在部署后模拟奖励，从而实现无奖励环境下的MORL部署。

**🔧 技术方法**

主要技术包括：1）MORL中的增强状态构造；2）基于监督学习的奖励模型训练；3）对比SER与ESR两类优化目标下的奖励处理策略；4）将特权信息的概念引入MORL，建议使用政策蒸馏或特权价值学习等方法。

**📊 数据集**

本文未使用具体公开数据集，而是通过理论分析和简易示例（MOMDP图例）说明问题，实验与数据集的使用并未给出。

**📈 对比分析**

由于缺乏实验设置，本文未进行数值比较或性能评估；文章以概念性讨论与案例分析阐释观点，未给出定量结果。

**⚠️ 局限性**

局限性包括：1）对奖励模型的准确性高度依赖，模型误差会直接影响部署后策略性能；2）在训练阶段仍需完整奖励信号，限制了在奖励难以获取的环境中的直接应用；3）缺乏实证验证，需在真实或模拟环境中进一步实验验证其有效性。

---

## 190. TTL: Test-time Textual Learning for OOD Detection with Pretrained Vision-Language Models

**arXiv ID:** 2604.15756 | [PDF](https://arxiv.org/pdf/2604.15756v1)

**作者:** Jinlun Ye `[一作]` (Sun Yat-sen University), Ruixuan Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3913 | [OpenAlex ID](https://openalex.org/A5100707149)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为TTL的测试时文本学习框架，利用CLIP模型在未标记的测试流中动态学习开放式OOD文本语义，从而提升OOD检测性能。

**💡 创新点**

核心创新在于：①不依赖预先定义的外部OOD标签，而是通过可学习提示和伪标签自适应地捕获多样化的OOD文本知识；②引入OOD知识净化损失来抑制伪标签噪声；③构建文本知识库进行跨批次校准，增强模型对分布漂移的鲁棒性。

**🔧 技术方法**

使用的主要技术包括CLIP视觉语言模型、可学习提示(prompt)与伪标签优化、余弦相似度评分、OOD知识净化损失、文本知识库以及基于阈值的自适应阈值确定。

**📊 数据集**

实验数据集：ID 数据为 ImageNet‑1K 与 CIFAR‑100；OOD 数据分别为 iNaturalist、SUN、Places、Texture（ImageNet‑1K）以及 SVHN、LSUN‑C、LSUN‑R、iSUN、Texture、Places365（CIFAR‑100）。

**📈 对比分析**

与多类基线（后期方法、训练方法和测试时自适应方法）进行对比，TTL 在 ImageNet‑1K 上平均 FPR95 12.46%/AUROC 97.29%，在 CIFAR‑100 上平均 FPR95 12.07%/AUROC 97.45%，在所有基准和OOV数据集上均显著优于 AdaNeg、AdaND、MoFE 等方法。

**⚠️ 局限性**

局限性包括：需要额外的推理时间和存储开销；伪标签仍可能导致误标噪声，尽管已通过净化策略降低；在极大批量或快速变化的测试分布下，适配速度可能受限，需进一步研究早停策略和在线更新效率。

---

## 191. Concept-wise Attention for Fine-grained Concept Bottleneck Models

**arXiv ID:** 2604.15748 | [PDF](https://arxiv.org/pdf/2604.15748v1)

**作者:** Minghong Zhong `[一作]` (Sun Yat-sen University), Ruixuan Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3913 | [OpenAlex ID](https://openalex.org/A5100707149)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CoAt-CBM，利用可学习的概念级视觉查询和概念对比优化实现自适应细粒度概念瓶颈模型。

**💡 创新点**

创新点在于：①概念级视觉查询注意力实现对视觉特征的自适应解耦与细粒度对齐；②概念对比优化通过正负概念对比提升概念互斥性和图像-概念匹配。

**🔧 技术方法**

采用 CLIP 视觉-语言模型、可学习视觉查询、尺度化点积注意力、对比损失（Concept Contrastive Optimization）和 BCE 等技术。

**📊 数据集**

在 10 个分类基准上评估：CIFAR-10/100、CUB-200、Food-101、DTD、SKIN-40、FGVC-Aircraft、Flower-102、UCF-101。

**📈 对比分析**

与 Linear Probe、LoRA-LP 及多种 VLM‑based CBM（PCBM、LaBo、HybridCBM 等）比较，CoAt-CBM 在准确率上实现 SOTA，且在 CDR、CC 等解释性指标上显著优于对手。

**⚠️ 局限性**

局限性：需要构建完整可靠的概念库，查询数量与对比系数等超参需手动调优，模型在大规模数据集上的推理成本与细粒度解释细化仍有提升空间。

---

## 192. Enhancing Discrete Particle Swarm Optimization for Hypergraph-Modeled Influence Maximization

**arXiv ID:** 2604.15746 | [PDF](https://arxiv.org/pdf/2604.15746v1)

**作者:** Qianshi Wang `[一作]` (Dalian University of Technology), Qiang Zhang `[通讯]` (Dalian University of Technology)

**通讯引用:** 175082 | [OpenAlex ID](https://openalex.org/A5100381911)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于离散粒子群优化（HDPSO）的超图影响力最大化方法，并通过阈值模型模拟传播过程。

**💡 创新点**

创新点包括：① 在超图上设计两层局部影响逼近的适应度评估；② 引入基于度的种群初始化和改进的离散速度/位置更新规则；③ 在搜索过程中嵌入局部搜索以提升全局探索与局部精细化的平衡。

**🔧 技术方法**

技术手段：阈值传播模型、离散粒子群优化（PSO）、度基初始化策略、局部搜索、两层局部影响逼近、统计显著性检验。

**📊 数据集**

数据集：合成超图（ER、SF、K‑UF）和真实超图（Algebra、Geometry、Music‑Blue、IJO1366）。

**📈 对比分析**

与GA、HHD、随机、NP、PageRank、HCI1、HCI2等基线比较；实验表明HDPSO在绝大多数合成和真实超图上均显著优于所有基线，并在统计检验中通过大部分配对比较。

**⚠️ 局限性**

局限性：在部分SF和UF实例中表现略逊于GA；对阈值设置敏感，导致传播过程不稳定；对大规模超图的计算开销仍相对较高。

---

## 193. Sketch and Text Synergy: Fusing Structural Contours and Descriptive Attributes for Fine-Grained Image Retrieval

**arXiv ID:** 2604.15735 | [PDF](https://arxiv.org/pdf/2604.15735v1)

**作者:** Siyuan Wang `[一作]` (Xidian University), Liang Zhang `[通讯]` (Xidian University)

**通讯引用:** 30547 | [OpenAlex ID](https://openalex.org/A5100425201)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 STBIR 框架，利用手绘草图与文本描述的互补信息实现细粒度图像检索。

**💡 创新点**

创新点在于：①多阶段跨模态特征对齐（先对齐草图再对齐图像和文本），②课程学习驱动的鲁棒性增强模块，③基于类别知识的特征空间优化模块，以及①新的三模数据集 STBIR。

**🔧 技术方法**

核心技术包括 ResNet 与 CLIP 编码器、噪声注入课程学习、增量角度余弦损失（Additive Angular Margin Loss）、InfoNCE 与 Triplet 损失的联合优化、以及三阶段的跨模态特征对齐策略。

**📊 数据集**

使用了 STBIR 数据集，包含 STBIR‑S（鞋类）、STBIR‑C（椅类）以及覆盖 125 个日常物品类别的 STBIR‑D，均提供手绘草图、文本描述和自然图像三模对齐数据。

**📈 对比分析**

与单模/多模基线（如 CLIP、TASKformer、SEARLE 等）比较，STBIR 在 R@1、R@5、R@10 上显著提升；在 STBIR‑D 上 R@1 达到 62.85%、R@5 93.44%，领先于现有方法。

**⚠️ 局限性**

局限性包括：对极其相似实例的区分仍受限，草图抽象性和文本描述粒度不足导致检索误差；未来需进一步增强局部特征表达与输入描述的细粒度。

---

## 194. BlockRaFT: A Distributed Framework for Fault-Tolerant and Scalable Blockchain Nodes

**arXiv ID:** 2604.15731 | [PDF](https://arxiv.org/pdf/2604.15731v1)

**作者:** Manaswini Piduguralla `[一作]` (Indian Institute of Technology Hyderabad), Sathya Peri `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 196 | [OpenAlex ID](https://openalex.org/A5004856023)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BlockRaFT，基于 RAFT 的领导者-跟随者模型将区块链节点内部工作分布到多台机器上，实现节点级别的可扩展性和容错性；

**💡 创新点**

核心创新在于：①将区块链节点拆分为可共享的状态/无状态模块并在领导者上集中调度；②三阶段并发 Merkle 树优化，将智能合约执行与树更新解耦，显著降低状态更新开销；

**🔧 技术方法**

使用 RAFT 领导选举、ETCD 或消息传递实现分布式共享内存、DAG 依赖分析、DSU 组件检测、并发哈希表更新，以及传统 Merkle 树重构；

**📊 数据集**

评测使用两款智能合约（投票 Voting 与钱包 Wallet）和 YCSB 基准，规模从 1,000 到 5,000 交易/块，线程数 2-64，节点数 3-7，故障模拟 1-3 节点；

**📈 对比分析**

与单核和多核无分布式基础的基线对比，BlockRaFT 在大多数工作负载下实现近线性扩展，吞吐量比单核提升 20-30 倍、比多核提升 5-10%，故障时仅增加 5-15% 延迟；

**⚠️ 局限性**

主要瓶颈为组件检测与执行阶段在高交易负载下的同步开销，Merkle 树重构仍是单线程热点，未来需优化共享数据传输与更细粒度并行度。

---

## 195. VoxMind: An End-to-End Agentic Spoken Dialogue System

**arXiv ID:** 2604.15710 | [PDF](https://arxiv.org/pdf/2604.15710v1)

**作者:** Tianle Liang `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 12254 | [OpenAlex ID](https://openalex.org/A5079260216)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 VoxMind 框架，将思考-先说（Think‑before‑Speak）机制与多代理动态工具管理集成到端到端语音对话模型中，实现自我推理与工具调用。

**💡 创新点**

创新点包括：① 统一的端到端语音代理定义；② 思考‑先说的内部推理链生成；③ 通过辅助 LLM 异步动态更新本地工具集，实现工具规模与推理延迟解耦；④ 构建 470 小时的 AgentChat 语音代理数据集。

**🔧 技术方法**

主要技术包括：端到端语音模型、链式推理（CoT）生成、基于多代理的并行工具管理、辅助 LLM 的候选工具预测、双通道记忆机制（语义+声学）。

**📊 数据集**

使用 AgentChat 数据集（由 ToolACE、APIGen‑MT 等文本工具数据合成，并通过 CosyVoice2 生成 470 小时语音）进行训练，并在 VoiceBench 及自制跨域评测集上评估。

**📈 对比分析**

与 StepAudio2、Kimi‑Audio、Qwen3‑8B+Whisper、Gemini‑2.5‑Pro 等基线相比，VoxMind 在六大核心能力（单任务、任务分解、并行处理、主动寻求、结果反馈、上下文规划）上的整体得分从 34.88% 提升至 74.57%，在大部分指标上实现或超过闭源模型。

**⚠️ 局限性**

局限性包括：① 思考‑先说引入额外推理延迟；② 数据集主要基于合成文本，缺乏真实语音中的口语障碍与非结构化表达；③ 仍需进一步降低推理成本并提升对真实语音环境的鲁棒性。

---

## 196. Bilevel Optimization of Agent Skills via Monte Carlo Tree Search

**arXiv ID:** 2604.15709 | [PDF](https://arxiv.org/pdf/2604.15709v1)

**作者:** Chenyi Huang `[一作]` (National University of Singapore), Yunduan Lin `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5065079697)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Agent Skill包的结构与内容进行双层优化，提出基于MCTS的外层结构搜索与LLM驱动的内层内容细化框架，并在Operations Research Question Answering（ORQA）数据集上进行实验。

**💡 创新点**

创新点在于将结构与内容的耦合优化拆分为两层Bilevel优化；外层采用MCTS实现对结构空间的探索，内层根据结构变更类型动态选择细化策略并使用保守的置信下界进行结果筛选；同时将LLM作为搜索与细化的核心推理引擎。

**🔧 技术方法**

主要技术包括：Agent Skill规范解析、LLM驱动的结构分析/诊断/建议、Monte Carlo Tree Search（MCTS）外层搜索、内层基于内容类别的有限迭代细化、低置信下界（LCB）评估及回传、以及与Harbor sandbox集成的实际执行评估。

**📊 数据集**

使用的公开数据集为ORQA（Operations Research Question Answering），共1513个多项选择问答样例，实验中采样120个用于搜索、验证与测试。

**📈 对比分析**

比较方法：在搜索集上进行Bilevel搜索，得到两种配置（A、B）下的最佳Skill；在验证集上评估其平均准确率以挑选最终Winner；最后在测试集上与原始Seed Skill对比。实验结果显示，优化后Skill在测试集上的准确率从0.90625提升至0.9375，提升约+3.125%。

**⚠️ 局限性**

限制：1）实验仅在单一数据集（ORQA）上验证，泛化性待进一步测试；2）LLM推理与MCTS搜索对计算资源消耗较大；3）内层细化采用保守的置信下界，可能导致某些有价值的改动被误判；4）仅优化结构与内容，未考虑运行时动态决策或多模态输入。

---

## 197. Diffusion Autoencoder for Unsupervised Artifact Restoration in Handheld Fundus Images

**arXiv ID:** 2604.15723 | [PDF](https://arxiv.org/pdf/2604.15723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 198. APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition

**arXiv ID:** 2604.15708 | [PDF](https://arxiv.org/pdf/2604.15708v1)

**作者:** Geunyoung Jung `[一作]` (University of Seoul), Jiyoung Jung `[通讯]` (University of Seoul)

**通讯引用:** 1367 | [OpenAlex ID](https://openalex.org/A5050953079)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种轻量级输入级对抗净化模块——Adversarial Point Counterattack（APC），通过生成针对每个点的反向扰动来抵消攻击效果；

**💡 创新点**

创新点包括：①利用几何一致性（Chamfer距离）和语义一致性（全局特征MSE）双重损失实现同时恢复局部几何和全局语义；②采用混合训练（多种攻击类型共训练）提升对未知攻击的泛化；③作为纯输入级模块实现跨模型、跨攻击的高可迁移性；④仅0.2M参数，推理时间0.002s，极低成本；

**🔧 技术方法**

技术细节：编码器-解码器结构（local、global、fusion三块），KNN聚合提取局部上下文；输出每点反扰动C，通过x'=x'+C得到净化点云；损失为交叉熵+α·几何一致性+β·语义一致性；训练使用干净–对抗配对，预处理采用SOR；

**📊 数据集**

实验数据集：ModelNet40（合成）与ScanObjectNN（真实扫描），每个样本均采样1024点；

**📈 对比分析**

与七种输入级防御（SRS、SOR、DUP-Net、IF-Defense、CausalPC）和两种模型级防御（Adversarial Training、Hybrid Training）比较；APC在三种目标模型（PointNet、PointNet++、DGCNN）和所有11种攻击下平均对抗准确率分别为84.7%、84.9%、83.4%（ModelNet40）和75.8%、82.8%、81.9%（ScanObjectNN），均为最优；跨模型迁移实验中APC在所有源-目标组合上均超过基线；效率上参数仅0.2M，推理时间0.002s；

**⚠️ 局限性**

局限性：需要预先生成干净–对抗配对进行训练；混合训练对攻击种类的覆盖仍有限，极端或未见攻击时性能下降；未评估对自适应攻击或动态点云场景的鲁棒性；

---

## 199. Improving Reasoning Capabilities in Small Models through Mixture-of-Layers Distillation with Stepwise Attention on Key Information

**arXiv ID:** 2604.15701 | [PDF](https://arxiv.org/pdf/2604.15701v1)

**作者:** Yao Chen `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Tingwen Liu `[通讯]` (Institute of Information Engineering Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在链式思维蒸馏中，提出 MoLSAKI 框架，传递教师模型在推理步骤中的关键词注意力，提升小模型推理性能。

**💡 创新点**

创新点在于利用教师的逐步关键词注意力而非仅传递合理化文本，并引入 Mixture-of-Layers 动态层对齐，实现不同深度模型的高效对齐。

**🔧 技术方法**

技术包括 CoT 蒸馏、逐步注意力提取、Mixture-of-Layers (MoL) 自适应层加权、KL 散度损失等。

**📊 数据集**

使用数学推理数据集 SVAMP、SingleEq、Asdiv、GSM8K 以及常识推理数据集 CommonSenseQA。

**📈 对比分析**

与 Vanilla Fine‑Tune、DSS、MMILoss 等基线比较，MoLSAKI 在各自域内和域外均提升约 7%–11% 的准确率，显著优于固定层对齐方案。

**⚠️ 局限性**

局限在于实验规模受限，未覆盖更大或更复杂的模型与任务；仅针对短链推理，需验证对更长、更复杂推理的泛化。

---

## 200. Graph self-supervised learning based on frequency corruption

**arXiv ID:** 2604.15699 | [PDF](https://arxiv.org/pdf/2604.15699v1)

**作者:** Haojie Li `[一作]` (Qingdao University of Science and Technology), Junwei Du `[通讯]` (Qingdao University of Science and Technology)

**通讯引用:** 1106 | [OpenAlex ID](https://openalex.org/A5009721014)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种频率腐蚀式图自监督学习（FC‑GSSL）算法，利用低频贡献信息生成高频偏向的损坏图，并通过图自编码器与对比学习共同训练图表示。

**💡 创新点**

创新点：①将低频信号贡献度作为衡量指标，动态产生高频偏向的损坏图；②设计多种采样策略（取值与排名）并通过交集/并集生成多样化视图；③对不同损坏图的节点表示进行对比对齐，减少对单一高频模式的依赖并提升泛化。

**🔧 技术方法**

技术手段：图谱分析与拉普拉斯谱低频贡献度计算、图自编码器（GAT/GraphPAE编码器）、位置编码（高斯RBF变换）、信息对比损失（InfoNCE）、节点/边的特征重构（SCE）、多视图对齐与多损失融合。

**📊 数据集**

使用14个公开图数据集：节点分类（BlogCatalog、Chameleon、Squirrel、Actor、arXiv‑year、Penn94）、图级预测（7个OGB数据集）、迁移学习（ZINC15预训练后QM9下游任务），覆盖社交网络、引用网络及分子图谱。

**📈 对比分析**

与DGI、BGRL、MVGRL、Sp^2GCL、GraphPAE、GraphMAE、GraphMAE2、AUG-MAE、GraphCL等现有对比学习与自编码器基线对比；实验表明FC‑GSSL在节点分类、图预测及迁移学习任务中均取得最优或最接近最优的性能（例如节点分类最高88.7%准确率，OGB回归RMSE最低1.001，QM9迁移任务R²最高82.69）。

**⚠️ 局限性**

局限性：①对低频贡献度与采样率等超参数高度敏感；②需要谱分解或拉普拉斯矩阵，计算成本在大规模图上仍较高；③在高频信息不丰富或噪声严重的图上可能仍无法充分发挥优势；④对不同图结构的适应性需进一步验证。

---

## 201. The World Leaks the Future: Harness Evolution for Future Prediction Agents

**arXiv ID:** 2604.15719 | [PDF](https://arxiv.org/pdf/2604.15719v1)

**作者:** Chuyang Wei `[一作]` (University of Science and Technology of China), Shuxin Zheng `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Milkyway，一种自我进化的未来预测代理，利用同一未解决问题的多次预测中的内部反馈，持续更新可复用的预测 harness，从而在预测过程中改进因子追踪、证据收集和不确定性处理。

**💡 创新点**

创新点在于将内部反馈（时间对比）作为在未决阶段的训练信号，持续更新持久化 harness；同时保持模型参数不变，依靠外部结构化 artifact 的演化来提升预测质量。

**🔧 技术方法**

采用 ReAct 式工具调用代理、持久化 harness（包含因子追踪、证据处理与不确定性模块）、内部反馈提取与 harness 更新逻辑，以及在问题解锁时的回顾性检查。

**📊 数据集**

在 FutureX 和 FutureWorld 两个实时未来预测基准上进行实验。

**📈 对比分析**

与 GPT‑5.4、MiroFlow、Flash‑Searcher、MemEvolve+Flash‑Searcher、AgentKB+smolagents 等基线对比，Milkyway 在 FutureX 上从 44.07 提升到 60.90，在 FutureWorld 上从 62.22 提升到 77.96；在滚动日评估和固定样本 ablation 中均保持领先。

**⚠️ 局限性**

局限性包括：仅在问题可多次预测且公开证据可持续演化时有效；长期使用可能导致 harness 产生冗余或漂移；实验仅覆盖两个基准与有限的在线窗口，结果可能不具普遍性。

---

## 202. NeuroLip: An Event-driven Spatiotemporal Learning Framework for Cross-Scene Lip-Motion-based Visual Speaker Recognition

**arXiv ID:** 2604.15718 | [PDF](https://arxiv.org/pdf/2604.15718v1)

**作者:** Junguang Yao `[一作]` (Chinese University of Hong Kong), Yue Zheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 13414 | [OpenAlex ID](https://openalex.org/A5100636088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于事件相机的视觉说话人识别框架NeuroLip，能够在跨场景条件下从微秒级唇部运动捕捉身份特征。

**💡 创新点**

创新点在于整合时间感知体素编码、结构感知空间增强和极性一致性正则化，专门针对事件数据提取细粒度时空行为模式，并保持运动方向信息。

**🔧 技术方法**

采用事件体素编码（TVE）、通道压缩与方向感知平滑（SSE）、极性一致性正则化（PCR）以及改进的ResNet34分类网络，配合事件级去噪与数据增强。

**📊 数据集**

使用自研的DVSpeaker事件唇动数据集（50人、20k样本、四种视角/光照组合），并在公开的DVSLip视频数据集上进行验证。

**📈 对比分析**

在匹配场景下几乎达到100%准确率，跨场景（训练仅在正面充分照明）下准确率分别为71.7%（侧视角）和75.9%（低照明），显著优于复现的23种基线模型，提升幅度达19.8%和8.5%。

**⚠️ 局限性**

局限性包括对极端侧向（90°）视角的识别仍低下、仅使用数字发音作为语料、以及在低光照下噪声仍会削弱极性正则化效果。

---

## 203. Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries

**arXiv ID:** 2604.15717 | [PDF](https://arxiv.org/pdf/2604.15717v1)

**作者:** Ki Sen Hung `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10690 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Jargon框架，利用安全研究背景与多轮对话，系统性地绕过大型语言模型的安全防护，实现对有害信息的提取。

**💡 创新点**

创新点在于发现安全研究语境可导致“通用解锁”，并通过学术化的多轮交互，将有害请求置于模型灰区，从而大幅提升破解成功率；同时提出了基于策略的防护与对齐微调方案。

**🔧 技术方法**

使用的技术包括：多轮对话控制层与评判层、攻击目标内存、查询变体生成、激活空间与注意力分析、策略驱动安全防护以及对齐微调。

**📊 数据集**

使用的数据集包括：JailbreakBench（100条跨10类有害提示），安全研究论文（abstract+method/全文），以及多领域专业论文（如化学、病毒合成等）作为上下文。

**📈 对比分析**

与现有单轮/多轮攻击（PAIR、AmpleGCG、Crescendo、FITD、X-Teaming）对比，Jargon在7大主流LLM（GPT-5.2、Claude-4.5系列、Gemini-3、Qwen3-235B、LLaMA-4-Scout-IT等）上的攻击成功率平均达99%，单轮攻击低于10%，多轮攻击在弱模型上约50-70%。

**⚠️ 局限性**

局限性包括：仅验证了论文、报告等学术文献，未覆盖技术博客或行业报告；防护措施虽降低成功率但未完全阻止攻击；知识净化组件可能导致对有害内容的误判或过度放大。

---

## 204. GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows

**arXiv ID:** 2604.15715 | [PDF](https://arxiv.org/pdf/2604.15715v1)

**作者:** Jize Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 100429 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GTA-2层级基准，用于评估从单步工具使用到长周期工作流的通用工具代理性能

**💡 创新点**

创新点包括：真实用户查询、真实部署工具、跨模态输入、递归检查点评估机制以及统一评估框架

**🔧 技术方法**

结合大型语言模型、ReAct与OpenCompass评估平台，使用递归检查点打分和LLM判定器实现自动化评估

**📊 数据集**

采用从真实平台（Manus、Kortix等）和社区（Reddit、Stack Exchange）抽取的154条原始任务，转化为132条经过人工与LLM重构的工作流，配合14/37个可执行工具

**📈 对比分析**

在GTA-Atomic上，前沿模型准确率不到50%；在GTA-Workflow上最高根成功率仅为14.39%，但先进的执行框架（OpenClaw、Manus）可将成功率提升至50%+并显著改善子目标完成率

**⚠️ 局限性**

局限性包括：任务重构可能引入偏差、评估侧重于结果质量缺乏安全与隐私维度、检查点设计与失败分类仍较经验性

---

## 205. Evidence Sufficiency Under Delayed Ground Truth: Proxy Monitoring for Risk Decision Systems

**arXiv ID:** 2604.15740 | [PDF](https://arxiv.org/pdf/2604.15740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 206. P3T: Prototypical Point-level Prompt Tuning with Enhanced Generalization for 3D Vision-Language Models

**arXiv ID:** 2604.15703 | [PDF](https://arxiv.org/pdf/2604.15703v1)

**作者:** Geunyoung Jung `[一作]` (University of Seoul), Jiyoung Jung `[通讯]` (University of Seoul)

**通讯引用:** 1367 | [OpenAlex ID](https://openalex.org/A5050953079)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种仅在输入空间进行点级提示和文本级提示的参数高效调优方法 P^3T，以冻结的预训练 3D VLM 适应下游点云分类任务。

**💡 创新点**

创新点在于：①将点级提示直接施加在点云坐标上，避免破坏预训练嵌入空间；②引入可学习文本提示并通过一致性正则保持通用文本知识；③使用原型损失降低类别内嵌入方差，提升嵌入区分度。

**🔧 技术方法**

技术包括：输入空间提示（点级提示生成器与偏移生成器）、文本提示正则化、一致性损失、原型损失、基于 EdgeConv 的特征增强、CLIP/ULIP 预训练模型。

**📊 数据集**

使用 ModelNet40（合成）和 ScanObjectNN（真实扫描）进行分类与 few‑shot 评估，并在 LVIS 上做源数据训练、在 MN40 与 ScanObjectNN 上做跨数据集泛化测试。

**📈 对比分析**

与全微调、IDPT、DAPT、Point‑PRC、PPT 等方法对比，P^3T 在 MN40 上达 94.0%（仅 2M 可学习参数），在 ScanObjectNN 上达到 95.2%/93.1%/88.1% 等，few‑shot 也往往优于竞争者；跨数据集上平均提升 8.3% 以上，表明更强的泛化。

**⚠️ 局限性**

局限性：在极低样本噪声数据集（PB）下原型不稳定导致性能略逊；对超参数如提示比例、原型数目敏感；并未在更大规模工业点云上验证。

---

## 207. The Metacognitive Monitoring Battery: A Cross-Domain Benchmark for LLM Self-Monitoring

**arXiv ID:** 2604.15702 | [PDF](https://arxiv.org/pdf/2604.15702v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证跨领域行为测评工具，评估LLM的监测-控制耦合并识别三种元认知配置

**💡 创新点**

将Nelson–Narens监测-控制框架与Koriat–Goldsmith双探针实验迁移至LLM，揭示准确度与元认知敏感度倒置的结构特征

**🔧 技术方法**

使用双探针二分决策（KEEP/WITHDRAW & BET/NO_BET）、Type‑2 SDT对比、六域任务设计、预注册与公开代码

**📊 数据集**

共524项跨六个认知域的自定义项目（T1–T6），评估20个前沿LLM，数据托管于GitHub与OSF

**📈 对比分析**

通过保持/撤回率差异与回答准确率比较，生成“倒置排行榜”，模型在准确度与元认知敏感度上相互独立；不同规模与架构表现出不同的敏感度曲线

**⚠️ 局限性**

单次测量、样本量有限、二分探针可能混合响应风格、未包含人类对照与梯度信心评分、需进一步检验训练机制与架构因素

---

## 208. Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions

**arXiv ID:** 2604.15762 | [PDF](https://arxiv.org/pdf/2604.15762v1)

**作者:** Huan Lin `[一作]` (Shanghai Jiao Tong University), Lianghui Ding `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1250 | [OpenAlex ID](https://openalex.org/A5079885035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理信息图神经网络的零样本可扩展UAV无人机群体的去中心化恢复框架PhyGAIL。

**💡 创新点**

创新点在于：①通过K-NN截断构造尺度不变的局部观测图；②引入物理门控消息传递实现显式吸引-排斥交互；③使用场景自适应模仿学习和专家时间奖励提升训练稳定性与零样本泛化。

**🔧 技术方法**

使用技术包括：CTDE训练、MAPPO、GAIL、物理信息图神经网络(PhyGNN)、K-NN邻域选择、虚拟中心节点、专家时间奖励、软标签平滑等。

**📊 数据集**

数据集为模拟的二维无人机群体数据，包含不同规模(N=20,50,100,200,500)和不同损坏比例(ρ=0.05~0.95)的网络碎片化场景，专家数据由多种基线算法生成并做数据增强。

**📈 对比分析**

与多种中心化与去中心化基线(如CR-MGC, DEMD, GDR-TS, center-fly, HERO, SIDR, MADDPG-APF等)对比，PhyGAIL在所有规模下均实现完美收敛率、最快恢复时间和最低碰撞率，整体排名第二。

**⚠️ 局限性**

局限性在于：仅在仿真二维平面上验证，未考虑障碍物、持续失效、实时硬件资源限制；训练仍需在小规模场景(20无人机)完成，且对极端高损坏比例的适应性待进一步评估。

---

## 209. RefereeBench: Are Video MLLMs Ready to be Multi-Sport Referees

**arXiv ID:** 2604.15736 | [PDF](https://arxiv.org/pdf/2604.15736v1)

**作者:** Yichen Xu `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**通讯引用:** 4848 | [OpenAlex ID](https://openalex.org/A5009985839)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 RefereeBench，一个跨 11 种体育的多模态大语言模型自动裁判评估基准，包含 925 个裁判视频和 6,475 条专家编写的 QA 对。

**💡 创新点**

创新点在于首次提供面向多体育、覆盖判罚存在、分类、推理、实体感知和时序定位等五大裁判能力的全尺度评测框架，并通过人类裁判手工标注保证了裁判逻辑与多模态证据的一致性。

**🔧 技术方法**

本文使用了多模态大语言模型（如 GPT‑5、Gemini‑3‑Pro、Doubao‑Seed‑1.8、Qwen3‑VL 等）以及视频处理技术和音频增强，评估其在裁判判罚任务上的表现。

**📊 数据集**

数据集为 RefereeBench，包含 925 条裁判视频、6,475 条多模态 QA，涵盖足球、篮球、排球、网球、乒乓、羽毛球、手球、场地曲棍、冰球、水球、短道速滑等 11 项运动。

**📈 对比分析**

通过统一多选回答协议对比 12 款闭源与 4 款开源 MLLM，结果显示闭源模型最高可达 61% 以上，开源模型最高约 47%，表明当前模型在规则推理和时序定位上仍有显著短板。

**⚠️ 局限性**

主要限制包括：模型在规则推理、时序定位和偏置识别方面表现不佳；数据集仍受限于广播视频，可能存在音频/解说等诱导信息；缺乏跨运动的泛化与鲁棒性验证。

---

## 210. Collective Kernel EFT for Pre-activation ResNets

**arXiv ID:** 2604.15742 | [PDF](https://arxiv.org/pdf/2604.15742v1)

**作者:** Hidetoshi Kawase `[一作]` (CyberAgent), Toshihiro Ota `[通讯]` (CyberAgent)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5047076115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对预激活ResNet在有限宽度初始化时的统计动力学，首先推导了精确的增量条件高斯分布和对应的MSRJD动作，随后构建了集体双局域随机有效场理论（EFT），并从中提取出核递归、方差动力学以及一阶、二阶闭包方程；最后通过数值仿真验证了理论的准确性与局限。

**💡 创新点**

创新点在于：① 以增量为主变量得到无鬼场的精确MSRJD动作；② 在集体EFT框架中将有限宽度修正重新划分为噪声源、输运项和堆塔源，并给出相应的Feynman图解释；③ 系统地揭示了G‑only闭包在不同深度下的有效窗口和失效机制，提供了从微观层面到宏观动力学的完整闭合体系。

**🔧 技术方法**

采用的技术包括条件高斯推理、马尔可夫闭包与高阶（GC0、LIN、GC1）闭包、Kramers–Moyal扩散近似、MSRJD路径积分、Feynman图表法以及离散时间到连续深度的近似。

**📊 数据集**

实验使用随机初始化的仿真网络（宽度64–256，输入维度N=4，激活函数tanh），没有使用公开的真实数据集。

**📈 对比分析**

通过将理论解（K_0、V_4、K_1等）的数值积分与大样本仿真结果进行对比，验证了K_0在所有深度几乎无误差；V_4在t≈1后出现约1–2%的系统误差；K_1在层数为0时即出现系统性偏差。总体来看，理论与仿真在低深度下保持高度一致，但随深度增加误差逐步累积。

**⚠️ 局限性**

主要局限在于：① G‑only闭包在较大深度或时间t>1时失效，主要是χ传输项对V_4的误差累积；② GC1源模型在层0即产生系统性误差，导致K_1的偏差；③ 需要将sigma‑kernel S^(p,q)等高阶观测量加入状态空间，以实现更高阶的闭合与更广泛的有效范围。

---

## 211. Reasoning-targeted Jailbreak Attacks on Large Reasoning Models via Semantic Triggers and Psychological Framing

**arXiv ID:** 2604.15725 | [PDF](https://arxiv.org/pdf/2604.15725v1)

**作者:** Zehao Wang `[一作]` (Tianjin University), Lanjun Wang `[通讯]` (Tianjin University)

**通讯引用:** 2651 | [OpenAlex ID](https://openalex.org/A5025153128)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对大型推理模型（LRM）的“推理过程定向越狱攻击”，通过在保持最终答案不变的前提下注入有害内容到推理链中；

**💡 创新点**

创新点在于将语义触发器选择与心理学理论（服从权威与道德脱离）结合，自动生成适应性诱导指令，实现对安全机制的突破；

**🔧 技术方法**

技术包括语义分析提取高风险关键词、逻辑与风险得分评估、触发器选择、基于心理学的指令生成（服从权威与道德脱离）、指令拼接生成扰动查询；

**📊 数据集**

使用CommonsenseQA、StrategyQA、FreshQA、MedQA、LegalQA五个公开问答数据集；

**📈 对比分析**

与Cognitive Overload Attack与H‑CoT两种基线对比，在DeepSeek R1、Qwen2.5‑Max、OpenAI o4‑mini三款商业LRM上，平均攻击成功率达83.6%，在所有模型和数据集上均超过基线，且在有害性分数上提升约32.5%；

**⚠️ 局限性**

局限性包括对推理复杂度高或新知识场景下成功率下降、跨模型转移效果受限、对极端安全性强的模型仍有一定抵御；

---

## 212. Just Type It in Isabelle! AI Agents Drafting, Mechanizing, and Generalizing from Human Hints

**arXiv ID:** 2604.15713 | [PDF](https://arxiv.org/pdf/2604.15713v1)

**作者:** Kevin Kappelmann `[一作]` (University of Sheffield), Dmitriy Traytel `[通讯]` (University of Copenhagen)

**通讯引用:** 1036 | [OpenAlex ID](https://openalex.org/A5046106256)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并形式化了 Isabelle 的“正确打印”问题，给出 Smolka‑Blanchette 算法的完整证明与自动化实现。

**💡 创新点**

首次提供完整的元理论与形式化证明，证明完备性与最小性；同时展示 LLM 在编程语言元理论与自动化形式化中的新角色。

**🔧 技术方法**

基于 Isabelle/HOL 的形式化证明、LLM（4.6）与 AI 编码环境、自动化形式化工具以及独立 AI 代理。

**📊 数据集**

未使用外部数据集，采用示例 Isabelle 术语及内部算法实例，实验结果公开于 Zenodo。

**📈 对比分析**

通过人类专家与 AI 代理两种工作流对比，实验均在数小时内完成，LLM 成功重现完整证明并自动化 1800 行 Isabelle 代码，成本 93。

**⚠️ 局限性**

仍缺乏对优化细节、类型上下文等的完整证明，LLM 仍需人工审查，自动化过程对模型依赖较高。

---

## 213. Towards Robust Endogenous Reasoning: Unifying Drift Adaptation in Non-Stationary Tuning

**arXiv ID:** 2604.15705 | [PDF](https://arxiv.org/pdf/2604.15705v1)

**作者:** Xiaoyu Yang `[一作]` (University of Technology Sydney), Jie Lu `[通讯]` (University of Technology Sydney)

**通讯引用:** 23555 | [OpenAlex ID](https://openalex.org/A5100675577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出并实现 Counterfactual Preference Optimization++ (CPO++) 框架，用于解决多模态大型语言模型（MLLMs）在强化微调过程中出现的内生推理漂移问题。

**💡 创新点**

创新点在于将逆因果干预与对抗性偏好优化统一到思考（语言推理）与感知（视觉关注）两个维度，构建自演化的视觉‑文本对抗样本生成与层次知识图，实现对多模态概念漂移的主动纠正与自适应学习。

**🔧 技术方法**

采用结构因果图、自演化对照样本、层次知识图、直接偏好优化（DPO）、强化学习（PPO/GRPO）微调以及视觉‑文本一致性协议等技术。

**📊 数据集**

使用医学诊断数据集 MIMIC‑CXR、MS‑CXR‑T、PadChest、ChestXray14、CheXDet10 等；自动驾驶数据集 BDD‑X、CODA‑LM、DriveLM 等进行评估。

**📈 对比分析**

与传统 SFT、DPO、CPO 以及多模态推理基线（LLaVA‑1.5、DriveGPT4、HoP 等）做对比；在推理一致性（CIDEr、BLEU‑4、METEOR）、决策精度（Top‑1 Accuracy）、鲁棒性（对 80% 对抗干扰保持≈80% 性能）和零样本跨域泛化上均显著优于基线，取得最高分。

**⚠️ 局限性**

限制包括：需手工构建或收集层次知识图，算法对自演化样本质量敏感；在极端噪声或模态缺失时仍可能出现漂移；计算成本较高，且缺乏对长期序列或多任务迁移的系统评估。

---

## 214. Intent Propagation Contrastive Collaborative Filtering

**arXiv ID:** 2604.15704 | [PDF](https://arxiv.org/pdf/2604.15704v1)

**作者:** Haojie Li `[一作]` (Qingdao University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 24116 | [OpenAlex ID](https://openalex.org/A5011384237)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Intent Propagation Contrastive Collaborative Filtering (IPCCF) 算法，结合双螺旋信息传播和意图传播进行协同过滤；

**💡 创新点**

创新点在于：1) 双螺旋信息传播框架融合直接与高阶关系以提取深层语义；2) 通过意图传播将图结构信息引入解耦过程；3) 采用对比学习为解耦提供直接监督，减少偏差与过拟合；

**🔧 技术方法**

使用了图卷积网络（LightGCN）、高阶关系提取、对比学习（InfoNCE）、意图传播机制及BPR损失；

**📊 数据集**

在Gowalla、Amazon-book和Tmall三大真实推荐数据集上进行实验；

**📈 对比分析**

与GNN、SSL和解耦基线（如NGCF、LightGCN、SGL、DCCF、BIGCF等）对比，IPCCF在Precision@20/40、Recall@20/40、NDCG@20/40上平均提升约6-9%，并在稀疏数据、意图数量和过平滑性方面表现更优；

**⚠️ 局限性**

限制在于：对高阶关系的阈值设定需要经验调参，意图数目的选择仍需针对不同域进行探索；

---

## 215. KWBench: Measuring Unprompted Problem Recognition in Knowledge Work

**arXiv ID:** 2604.15760 | [PDF](https://arxiv.org/pdf/2604.15760v1)

**作者:** Ankit Maloo `[一作]` `[通讯]` (Clio AI), Ankit Maloo (Clio AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 KWBench 这一知识工作基准，专注于语言模型在未被提示时识别问题类型的能力。

**💡 创新点**

创新点在于引入强制性门槛（mandatory gate）和以失败模式为核心的三层评分体系，直接衡量模型对游戏理论结构的自发识别。

**🔧 技术方法**

利用大规模语言模型（Claude Opus、GPT‑5、Gemini 等）配合代码解释器、工具调用与基于阈值的评判器，实现自动化评估。

**📊 数据集**

使用了 223 条来自并购、合约谈判、临床药学、组织政治、欺诈分析等行业的真实专业场景，形成跨领域任务集。

**📈 对比分析**

对 16 个前沿模型进行三次评测，最强模型通过门槛率为 27.9%，条件得分聚焦 82% 以上，但整体通过率低，说明识别能力仍是瓶颈。

**⚠️ 局限性**

局限包括缺乏人类基准、单一评判者、可能的数据污染、对西方专业范式的偏倚以及未完成的认知拆解实验。

---

## 216. Language, Place, and Social Media: Geographic Dialect Alignment in New Zealand

**arXiv ID:** 2604.15744 | [PDF](https://arxiv.org/pdf/2604.15744v1)

**作者:** Sidney Wong `[一作]` `[通讯]`, Sidney Wong

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过将用户主观感知与计算语言学方法相结合，探究新西兰相关Reddit社群中文化与地理方言的一致性。

**💡 创新点**

创新点在于引入“地理方言一致性”概念，构建用户感知变量，并将其与Word2Vec、C2xG等高级模型联合，用多阶段流程验证数字空间与物理空间之间的语言对齐。

**🔧 技术方法**

采用主题分析、语义嵌入（Word2Vec）、文本分类、构造语法模型（C2xG）及时间序列嵌入等技术，对用户感知变量进行量化与可视化。

**📊 数据集**

使用了超过42亿词的Reddit全域语料库、聚焦的NZR子语料库以及相应的词向量模型，涵盖全国与城市级别多维度社群。

**📈 对比分析**

通过对比传统文本分类与高维词向量模型的性能，发现分类方法在检测地理变体时表现不佳，而基于词向量的相似度与语义漂移测量能准确捕捉方言对齐与时间演变。

**⚠️ 局限性**

主要局限包括对用户感知变量的依赖导致样本偏差、数据稀疏及模型训练时可能的偏见，以及缺乏对传统方言框架的深入整合，未来需拓展跨平台验证与更细粒度的语料采集。

---

## 217. Learning Uncertainty from Sequential Internal Dispersion in Large Language Models

**arXiv ID:** 2604.15741 | [PDF](https://arxiv.org/pdf/2604.15741v1)

**作者:** Ponhvoan Srey `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**通讯引用:** 2368 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用内部状态方差的序列化方法SIVR，用来检测大语言模型生成文本中的幻觉。

**💡 创新点**

创新点包括：①将隐藏层的通用方差、圆形方差和预测熵作为三维内部方差特征；②对全序列token级方差序列进行Transformer编码，学习时间模式；③显著提升跨任务泛化能力并减少对大规模训练数据的依赖。

**🔧 技术方法**

使用技术包括：高斯协方差行列式（log‑det）、圆形方差、token熵的计算；构建三维内部方差特征；利用轻量级Transformer编码器加二分类头；与多种基线方法进行对比实验。

**📊 数据集**

实验数据集涵盖12个事实检测和问答数据集（Counterfact、Common Claims、Animals and Facts、FEVER、TriviaQA、SciQ、MedMCQA、MMLU、MGSM、MATH、CommonsenseQA），并使用Llama‑3.2‑3B、Llama‑3.1‑8B、Ministral‑8B等模型。

**📈 对比分析**

与logit‑based、sampling‑based、confidence‑elicitation以及内部状态(CoE)和四个监督基线（SAPLMA、SATMD+MSP、Lookback Lens、TAD）比较，SIVR在AUC、FPR@95和AUPR分别提升约3%、7.5%和4%，且在OOD设置下表现更稳健。

**⚠️ 局限性**

局限性包括：仍为监督方法，需标注数据；实验主要针对中等规模LLM，尚未验证更大模型；解释性尚不充分，未来需进一步探讨特征对幻觉风险的贡献。

---

## 218. Privacy-Preserving LLMs Routing

**arXiv ID:** 2604.15728 | [PDF](https://arxiv.org/pdf/2604.15728v1)

**作者:** Xidong Wu `[一作]` (University of Pittsburgh), Shangqian Gao `[通讯]` (Florida State University)

**通讯引用:** 739 | [OpenAlex ID](https://openalex.org/A5020498118)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PPRoute框架，在安全多方计算（MPC）环境下实现大语言模型（LLM）的路由，兼顾用户隐私与路由性能。

**💡 创新点**

创新点包括：①用MPC友好的运算（2ReLU替换Softmax、ReLU替换GeLU）加速Transformer编码器；②采用多阶段蒸馏训练，使近似模型保持与原始模型相近的路由质量；③提出无排序Top‑k算法，通信复杂度仅为O(1)，显著提升最近邻搜索速度。

**🔧 技术方法**

使用技术包括：Secure Multi‑Party Computation、MPC友好运算近似、模型蒸馏、未排序Top‑k算法、CrypTen框架以及多阶段训练流程。

**📊 数据集**

实验数据集涵盖 EmbedLLM、MixInstruct 与 RouterBench 三大基准集，评估路由精度与成本。

**📈 对比分析**

与基线（plain‑text路由、Naïve MPC实现、传统排序与近似检索算法）对比，PPRoute在保持与原始路由相近的路由质量（AUDC、峰值精度、QNC）下，MPC推理时间提升约20×，通信量与往返轮数显著下降。

**⚠️ 局限性**

局限性：主要适用于规模较小的模型池；在大规模检索场景下未做针对性优化；MPC实现仍基于半诚实模型，性能受限于网络延迟；目前仅针对embedding‑based路由，其他路由策略的迁移性尚未验证。

---

## 219. When Do Early-Exit Networks Generalize? A PAC-Bayesian Theory of Adaptive Depth

**arXiv ID:** 2604.15764 | [PDF](https://arxiv.org/pdf/2604.15764v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22416 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对早期退出网络提出统一的PAC-Bayesian泛化理论，并给出以退出深度熵H(D)与期望深度E[D]为核心的泛化界。

**💡 创新点**

创新点包括：① 用退出深度熵替代传统最大深度K获得更紧泛化界；② 计算出完整常数√(2ln2)≈1.177；③ 证明早期退出在满足条件时可严格优于固定深度；④ 允许约束标签独立性的ε-近似路由；⑤ 在多种架构与数据集上验证紧致性。

**🔧 技术方法**

技术主要使用PAC-Bayesian方法、Rademacher复杂度、信息熵分析、熵阈值退出策略与学习路由，并给出基于边界的阈值选择算法。

**📊 数据集**

数据集包括CIFAR-10、CIFAR-100、ImageNet-100、SST-2、MRPC、QNLI和WikiText-2等视觉与自然语言任务；实验覆盖MSDNet、ResNet-56-EE、EffNet-B0-EE、BERT+PABEE、DistilBERT+PABEE、GPT-2+CALM等六种架构。

**📈 对比分析**

与VC、Rademacher、传统PAC-Bayes以及自适应深度无关的PAC-Bayes相比，本方法的紧致比例为1.52–3.87×，在所有任务中均显著优于100%空泛界；同时基于边界的阈值选择可在0.1–0.3%内逼近验证调优性能。

**⚠️ 局限性**

局限包括：① 边界与验证调优仍有1.5–4×差距，不能完全替代；② 对极大规模模型（如ImageNet-1K、7B LLM）验证有限；③ 对标签相关路由的ε-近似随εK增大会削弱保证。

---

## 220. Adaptive Power Allocation and User Scheduling for LEO Satellites using Channel Predictions

**arXiv ID:** 2604.15733 | [PDF](https://arxiv.org/pdf/2604.15733v1)

**作者:** Lachlan Drake `[一作]` (University of Newcastle), Duy T. Ngo `[通讯]` (University of Newcastle)

**通讯引用:** 2528 | [OpenAlex ID](https://openalex.org/A5003913406)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在低轨卫星下行链路中利用通道预测进行自适应功率分配与用户调度，目标是最大化最小用户速率并保证公平性。

**💡 创新点**

创新点在于提出了 APASS 框架：结合多时步通道预测与连续凸逼近（geometric programming）实现动态功率分配与调度，能够在未知未来信道时仍保持接近上界性能。

**🔧 技术方法**

采用基于轨道动力学的时变通道模型（LoS/NLoS 两状态，Rician/Rayleigh 混合衰落），并使用连续凸逼近与内部点法求解几何规划；通道预测以高斯误差模型示范。

**📊 数据集**

使用 SpaceX Starlink 卫星公开轨道参数，并模拟 20 个 UE 在悉尼附近 50 km 范围内，载波 27.5 GHz、带宽 5 MHz、EIRP 5 MW 等参数。

**📈 对比分析**

与单步预测（STS）、等功率、以及水分配算法对比，APASS 在预测误差方差 0.25 时均能获得最小用户速率比对手高 2.3–3.86 倍，公平指数保持 >0.99，性能显著优于其它可行方案。

**⚠️ 局限性**

仍依赖通道预测精度；误差增大会导致性能下降；算法复杂度高（O(I K³ N⁴)），需进一步降低实现成本与实时性。

---

## 221. Frenetic Cat-inspired Particle Optimization: a Markov state-switching hybrid swarm optimizer with application to cardiac digital twinning

**arXiv ID:** 2604.15761 | [PDF](https://arxiv.org/pdf/2604.15761v1)

**作者:** Jorge Sánchez `[一作]` (Universitat Politècnica De València), Javier Saiz `[通讯]` (Universitat Politècnica De València)

**通讯引用:** 3083 | [OpenAlex ID](https://openalex.org/A5063676325)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种名为FCPO的混合群体优化器，用粒子群动力学与马尔可夫状态切换相结合，在有限评估预算下实现高效搜索。

**💡 创新点**

创新点在于粒子级别的显式状态机调度多种搜索算子（PSO更新、精英差分跳跃、协方差引导局部搜索、线性种群缩减），以及在线自适应的马尔可夫转移矩阵。

**🔧 技术方法**

采用PSO基础、DE式精英跳跃、协方差矩阵分解（Eigen-guided）以及线性种群缩减（LPSR），并在算法中嵌入马尔可夫状态转换。

**📊 数据集**

使用CEC 2022单目标约束基准函数（F1、F2、F3、F6、F10）以及心脏电图激活校准的临床ECG数据。

**📈 对比分析**

在相同的函数评估预算下与PSO、CSO、CLPSO、SHADE、L-SHADE、CMA-ES比较，FCPO在平均运行时间上最低，且在F10 D=20上获得最佳平均目标值，整体准确-运行时间折中处于最优。

**⚠️ 局限性**

局限包括仅在五个基准函数和两维设置上验证，未彻底剖析马尔可夫切换机制对性能的具体贡献，且在心脏双心室校准上仅为示例，缺乏多被试和噪声鲁棒性评估。

---

## 222. PoSME: Proof of Sequential Memory Execution via Latency-Bound Pointer Chasing with Causal Hash Binding

**arXiv ID:** 2604.15751 | [PDF](https://arxiv.org/pdf/2604.15751v1)

**作者:** David L. Condrey `[一作]` `[通讯]` (WritersLogic Inc.), David L. Condrey (WritersLogic Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 PoSME 的新型证明机制，要求证明者在可变存储区内执行严格的顺序指针追踪，最终生成全局日志以证明工作量。

**💡 创新点**

创新点在于三项：将可变存储状态、数据相关的指针追踪和相互绑定的因果哈希相结合，形成基于 DRAM 随机访问延迟的 ASIC 抗性；实现了 10 倍的 TMTO 抗性；并通过 Binius 的二进制域折叠实现 O(1) 的 IVC 验证。

**🔧 技术方法**

使用 BLAKE3 哈希函数、布尔超立方体映射、多线性多项式承诺以及 Binius 的二进制域折叠技术，配合随机预言机模型进行安全分析。

**📊 数据集**

数据集主要是内部随机地址生成（ROM）和 1 GiB 的超立方体存储空间（N = 2²⁴），在 5.4 × 10⁸ 次读写操作下进行实验验证。

**📈 对比分析**

通过与 17 种 CPU 和 4 种 GPU 平台的基准测试比较，PoSME 的哈希占比低于 3.5%，GPU 在指针追踪上比 CPU 慢 14–19 倍；同时证明在 ASIC 级别的加速仅提升约 1.3×，表明在常见硬件上具备良好的 ASIC 抗性。

**⚠️ 局限性**

局限性包括：仅证明顺序内存执行而非实际耗时；安全性依赖随机预言机模型；对动态因果 DAG 的证明仍在理论模型中；以及目前的 IVC 方案尚未完成生产级实现。

---

## 223. On the Equivalence Between Auto-Regressive Next Token Prediction and Full-Item-Vocabulary Maximum Likelihood Estimation in Generative Recommendation--A Short Note

**arXiv ID:** 2604.15739 | [PDF](https://arxiv.org/pdf/2604.15739v1)

**作者:** Yusheng Huang `[一作]` (Kuaishou Technology), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

证明了生成推荐（GR）中常用的k-token自回归下一词预测（AR-NTP）与完整词表最大似然估计（FV-MLE）严格等价，阐明其理论基础；

**💡 创新点**

首次给出AR-NTP与FV-MLE等价性的严格证明，并扩展至串行（cascaded）和并行（parallel）两种主流tokenization方案，说明其理论普适性；

**🔧 技术方法**

使用统计推断、贝叶斯链式法则、softmax分解与符号推导等数学工具；

**📊 数据集**

无实验数据集，论文纯理论证明；

**📈 对比分析**

没有实验对比，主要论述复杂度优势：AR-NTP将全词表softmax拆解为k个小规模softmax，O(k·V_m)远低于O(V)，在理论上保持同等拟合效果；

**⚠️ 局限性**

主要局限在于依赖一一映射（bijective mapping）假设；若tokenization导致码表崩溃或冲突，等价性失效，且论文未提供实验验证。

---

## 224. Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving

**arXiv ID:** 2604.15732 | [PDF](https://arxiv.org/pdf/2604.15732v1)

**作者:** Takeshi Yoshimura `[一作]` (IBM Research Tokyo), Tatsuhiro Chiba `[通讯]` (IBM Research Tokyo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了长上下文LLM服务的新延迟度量——Time-to-Correct-Answer (TTCA)，并基于提示长度与语言特征实现了一种轻量级、准确性感知的路由策略（LAAR），通过实验验证其在长上下文请求中的优势。

**💡 创新点**

创新点包括：① 用TTCA将准确率误差引发的重试延迟量化为系统层面的性能指标；② 设计了无需深度语义分析的能力模型（使用逻辑回归预测成功率）和简单的延迟预测公式，从而在控制面上实现低开销的准确性感知路由；③ 在多模型集群中通过重试惩罚避免重复失败，提高最终成功率。

**🔧 技术方法**

技术手段：逻辑回归能力模型、基于请求长度和语言的成功率估计、简易延迟预测公式（c(m)*(T+αR)）、成本函数 L/Q 的最小化路由、Envoy External Processing Filter 作为路由实现，使用 deterministic decoding（温度0）来剔除输出噪声。

**📊 数据集**

使用 SCBench 的 KV 取值查询数据集，按 4K/8K/16K/32K/64K token 长度以及英语、日语、中文三种语言进行实验，采用两份 50 个查询的拆分，其中一份用于离线训练成功率模型，另一份用于在线评估。

**📈 对比分析**

与传统的 load-aware（负载感知）和 session-affinity（会话亲和）路由进行对比。实验显示，LAAR 在大多数上下文长度和语言组合下将 TTCA 降低 31%–49%，并显著提升最终成功率；在 64K token 时 load-aware 路由因为整体负载高而在 TTCA 上稍优，但成功率仍落后于 LAAR。

**⚠️ 局限性**

局限性：仅在 KV 取值这一相对简单、可自动评估准确率的任务上验证，未探讨开放式或对话型任务；重试惩罚和延迟估计在更复杂场景下可能需要更精细的设计；当前实现依赖 deterministic decoding，可能不适用于需要随机性的任务；此外未将路由开销计入 TTCA，未来可进一步融合多目标优化。

---

## 225. LLM Reasoning Is Latent, Not the Chain of Thought

**arXiv ID:** 2604.15726 | [PDF](https://arxiv.org/pdf/2604.15726v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Wenshuo Wang `[通讯]` (South China University of Technology)

**通讯引用:** 6030 | [OpenAlex ID](https://openalex.org/A5055099598)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文主张将大型语言模型的推理研究对象从传统的可视化链式思维（CoT）转变为潜在状态轨迹，并通过系统的假设对比与实验验证支持这一观点。

**💡 创新点**

创新点在于提出了三种竞争假设（潜在轨迹主导、表面CoT主导、计算成本主导），构建了可分离的实验设计与计算审计框架，并在多种推理基准上展示了不同范式在不同推理情境下的优势。

**🔧 技术方法**

技术上主要使用了基于Transformer的多步骤推理策略（CoT提示、latent reasoning/steering、self‑consistency或搜索）、计算审计（按解码步、KV缓存、工具调用等核算）、以及对潜在状态的因果干预和评估方法。

**📊 数据集**

实验数据集包括：GSM8K‑Platinum（普通推理）、HotpotQA（检索-计划式推理）、MATH（搜索驱动的数学推理）和HumanEval+（代码执行推理）。

**📈 对比分析**

通过在受控与自然环境下的六臂实验（表面干预、表面对照、潜在干预、潜在对照、计算扩展、基线），在匹配预算的前提下对比三类方法，结果显示：普通情境下潜在轨迹优势明显，构成式情境下表面CoT优势突出，搜索驱动情境下纯计算优势最大。

**⚠️ 局限性**

局限性在于实验设计仍依赖于可分离的因子与预算匹配，且在混合情境下未能显著表明单一主导模型，未来需进一步细化干预范式与跨任务通用性验证。

---

## 226. Neuromorphic Parameter Estimation for Power Converter Health Monitoring Using Spiking Neural Networks

**arXiv ID:** 2604.15714 | [PDF](https://arxiv.org/pdf/2604.15714v1)

**作者:** Hyeongmeen Baik `[一作]` (University of Wisconsin--Madison), Jinia Roy `[通讯]` (University of Wisconsin--Madison)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了基于脉冲神经网络与可微分ODE求解器的功率转换器健康监测方法。

**💡 创新点**

创新点在于将SNN的时间序列处理与物理约束分离，利用可微分ODE仅在训练中引入物理损失，兼顾低功耗与高准确性。

**🔧 技术方法**

使用Leaky Integrate-and-Fire SNN、可微分RK4 ODE求解器、surrogate梯度训练、EMI噪声模拟以及持久膜电位机制。

**📊 数据集**

数据集为模拟的同步降压变换器在噪声扰动下的电流、电压波形，涵盖真实参数与多种退化情形。

**📈 对比分析**

与传统全连接PINN/FF模型比较，R_s估计误差从25.8%降至10.2%，能耗从约881 nJ 降至3.3 nJ，提升约270倍。

**⚠️ 局限性**

局限在于SNN训练受 surrogate 梯度噪声影响，需要最佳检查点选择；对极端噪声或极端退化场景的鲁棒性仍待验证。

---

## 227. The Price of Paranoia: Robust Risk-Sensitive Cooperation in Non-Stationary Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.15695 | [PDF](https://arxiv.org/pdf/2604.15695v1)

**作者:** Deep Kumar Ganguly `[一作]` (Technical University Of Munich), Adithya Ananth `[通讯]` (Indian Institute Of Technology Tirupati)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种名为RATTL的自适应鲁棒性信任区域学习算法，旨在解决多智能体强化学习中合作稳定性易受共同学习噪声破坏的问题。

**💡 创新点**

创新点在于：①引入了EVaR悖论，证明将分布式鲁棒性直接作用于收益会扩大不稳定阈值；②改为作用于合作伙伴引起的梯度方差，从而获得自适应信任因子；③提出“恐慌价格（PoP）”与“合作窗口”统一框架，量化鲁棒性与样本复杂度的权衡。

**🔧 技术方法**

核心技术包括：策略梯度（REINFORCE/PPO）、EVaR风险度量、对梯度方差的信任调节、在线经验移动平均估计合作伙伴的合作概率，以及基于PoP的自适应风险参数更新。

**📊 数据集**

实验数据基于重复的Stag Hunt游戏（以及对鸡等对称协调游戏的理论推广），无真实外部数据集，全部在仿真环境中进行。

**📈 对比分析**

与传统PPO、Hysteretic Q-learning、Lenient Learning、LOLA等方法对比。RATTL在有噪声和无噪声的Stag Hunt中实现了近乎100%的合作率，显著优于其他基线，并且在高噪声情境下仍保持高合作率，证明其鲁棒性与效率兼备。

**⚠️ 局限性**

局限性包括：①仅在对称协调游戏中得到理论与实验支持，非对称或更复杂游戏的推广尚未验证；②依赖对伙伴合作概率的在线估计，若估计不准可能导致性能下降；③算法不主动影响伙伴策略，只是被动鲁棒化，对伙伴行为的调节有限。

---

## 228. SSMamba: A Self-Supervised Hybrid State Space Model for Pathological Image Classification

**arXiv ID:** 2604.15711 | [PDF](https://arxiv.org/pdf/2604.15711v1)

**作者:** Enhui Chai `[一作]` (Northwest University), Tianxiang Cui `[通讯]` (University of Nottingham)

**通讯引用:** 1110 | [OpenAlex ID](https://openalex.org/A5020724945)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为SSMamba的两阶段病理图像分析框架，结合领域自监督预训练（MAMIM）与轻量化监督微调，专门针对ROI与WSI级别的诊断任务；

**💡 创新点**

在三方面实现创新：①使用MAMIM实现跨放大倍率域移位补偿；②设计Directional Multi‑scale（DMS）模块在State‑Space模型中实现局部‑全局双向多尺度融合；③引入Local Perception Residual（LPR）模块提供平移旋转不变的空间编码与局部感知；

**🔧 技术方法**

核心技术包括：自监督掩码图像建模（MIM）在Mamba结构上实现，基于State‑Space模型的高效序列建模，深度可分离卷积与双向扫描机制，残差与深度卷积融合的LPR编码；

**📊 数据集**

使用10个公开ROI数据集（如NCT-CRC-HE-100K、CAMELYON16、SIPaKMeD等）以及6个WSI数据集（PANDA、TCGA-ESCA、TCGA-BRCA、TCGA-CESC、TCGA-LGG、TCGA-BLCA）进行实验；

**📈 对比分析**

与11种ROI级病理FMs及8种WSI级SOTA方法进行对比，SSMamba在ROI上平均F1≈95.56%、Acc≈95.98%、AUC≈95.02%，在WSI上获得12/17指标最高，参数量仅25.3M，显著优于传统ViT、Mamba以及大规模预训练模型；

**⚠️ 局限性**

目前仅针对分类任务，未覆盖语义分割、细胞计数等其他病理分析任务，未来需扩展到更丰富的应用场景；

---

## 229. LP$^{2}$DH: A Locality-Preserving Pixel-Difference Hashing Framework for Dynamic Texture Recognition

**arXiv ID:** 2604.15707 | [PDF](https://arxiv.org/pdf/2604.15707v1)

**作者:** Ruxin Ding `[一作]` (University of Nottingham Ningbo China), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15892 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种Locality‑Preserving Pixel‑Difference Hashing（LP2DH）框架，将动态纹理视频中的像素差分向量压缩为二进制码，再通过字典学习得到紧凑的特征表示；

**💡 创新点**

通过在Stiefel流形上联合优化量化误差、信息熵、方差和局部保持四个目标，实现在不使用深度网络的情况下获得高辨识度的哈希函数；

**🔧 技术方法**

采用像素差分向量哈希、局部保持嵌入、Stiefel流形梯度下降、k‑means字典学习以及多尺度采样等技术；

**📊 数据集**

在UCLA、DynTex++、YUPENN三大动态纹理基准上进行实验；

**📈 对比分析**

与光流、模型、几何、滤波、局部特征及多种学习式（包括深度网络）方法对比，使用最近邻分类器，LP2DH在UCLA 99.80%、DynTex++ 98.52%、YUPENN 96.19%，均优于或接近现有最优方案；

**⚠️ 局限性**

局部保持与多目标优化的权重折衷难以统一，且两阶段设计不支持端到端学习，导致哈希函数未能直接获取字典学习或分类反馈。

---

## 230. Target-Oriented Pretraining Data Selection via Neuron-Activated Graph

**arXiv ID:** 2604.15706 | [PDF](https://arxiv.org/pdf/2604.15706v1)

**作者:** Zijun Wang `[一作]` (ByteDance), Fengze Liu `[通讯]` (ByteDance)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于神经元激活图（Neuron‑Activated Graph, NAG）的目标导向预训练数据选择方法，直接利用大模型内部高影响神经元的激活模式进行样本排序；

**💡 创新点**

创新点在于将输入特征拆解为多层稀疏的高影响神经元集合，构造可解释的图结构并通过集合相似度实现无训练、无黑盒的目标匹配；

**🔧 技术方法**

技术细节包括：量化神经元影响度、按层挑选top‑K神经元构成NAG、计算样本与目标集合的相似度、调节稀疏比例、在多层与单层两种配置下评估；

**📊 数据集**

使用的数据集为RefinedWeb 150B token数据池，并在六大基准（ARC‑Challenge, HellaSwag, TriviaQA, MMLU, XStoryCloze, XWinograd）以及多目标混合场景进行实验；

**📈 对比分析**

与随机采样、FineWeb‑Edu、BETR等基线对比，单目标平均提升4.9%（HellaSwag提升4.4%），多目标平均提升3.6%，在不同LLM骨干上保持一致的性能优势；

**⚠️ 局限性**

局限性包括：实验仅在1.2B规模模型、单语域数据上验证；对更大模型、多语言或领域特定数据的效果未探测；多目标仅采用简单等份混合策略，尚未尝试更先进的混合方法。

---

## 231. Rate-Distortion Theory for Deductive Sources under Closure Fidelity

**arXiv ID:** 2604.15698 | [PDF](https://arxiv.org/pdf/2604.15698v1)

**作者:** Jianfeng Xu `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 107497 | [OpenAlex ID](https://openalex.org/A5100374993)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出并分析了一种基于推理闭包的源编码框架，对有限声明源在给定证明系统下的失真-率关系进行定量刻画。

**💡 创新点**

创新点在于：①将闭包保留作为失真度量，揭示推理结构可消除冗余从而降低编码率；②在满足“核心符号零失真集不相交”条件下得到零失真率的精确公式 R_sem(0)=P_A H(π_A)；③在受限推理深度时给出完整的速率-延迟-失真定律；④推理结构下的闭包保留率-失真分解只依赖于不可冗余核心；⑤给出源-信道分离、闭包适应的反证界以及异构接收器的扩展。

**🔧 技术方法**

主要技术包括：信息理论中的熵与互信息、图熵与零误差信息论、闭包运算与可决定性证明系统、定理推理的单步闭包迭代、数据处理不等式、Fano不等式以及Datalog/Horn子程序的归纳闭包。

**📊 数据集**

实验数据主要是理论示例与符号化的Datalog实例；作者用一个“材料化Datalog存储”场景（包括E DB事实与IDB推理结果）展示了零失真率随冗余存储量变化的压缩比。没有使用真实网络或语义数据集。

**📈 对比分析**

与传统的符号逐位(Hamming)失真率对比，闭包保留可显著降低零失真率（如示例中从 log 3 降至 2/3 位）；在有限推理预算 δ 的情况下，δ 变大时有效源量逐渐下降，直至恢复到完整闭包保留的极限；通过源-信道分离可得到在给定信道容量下可实现的最小 δ。性能表现体现在压缩比提升和延迟-速率权衡上。

**⚠️ 局限性**

局限性：①零失真率的精确表达需要“核心零失真集不相交”这一可判定的假设；②目前仅在有限、独立同分布的源情形下分析，未处理交叉依赖或连续取值；③对多终端、交互式或异构证明系统的推广仍待研究；④对图熵等更一般的confusability情况缺乏完整理论。

---

## 232. Neural Continuous-Time Markov Chain: Discrete Diffusion via Decoupled Jump Timing and Direction

**arXiv ID:** 2604.15694 | [PDF](https://arxiv.org/pdf/2604.15694v1)

**作者:** Jingyuan Li `[一作]` (Beijing Institute of Mathematical Sciences and Applications), Pipi Hu `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5073715806)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Neural CTMC 模型，将离散扩散的反向过程分解为退出率与跳转分布，并用双头网络进行参数化。

**💡 创新点**

创新点在于将连续时间马尔科夫链的时序/方向分解与参数化对齐，得到可分解为 Poisson KL 与分类 KL 的训练目标，并首次在纯统一噪声前向过程中实现领先性能。

**🔧 技术方法**

利用连续时间马尔科夫链、路径空间 ELBO、Gillespie 算法思想、τ‑leaping 与 Euler 抽样，以及 DiT 作为基础网络。

**📊 数据集**

在 TinyStories、OpenWebText（语言）以及 MNIST（图像）上进行实验。

**📈 对比分析**

与 SEDD、MDLM、GIDD 等基线在相同训练预算下进行对比；Neural CTMC 在 TinyStories 上 PPL≤16.36，OpenWebText 上在 16‑128 步均优于等预算基线，并在低步数下与 SEDD 竞争。

**⚠️ 局限性**

在 128 步时仍略逊于 SEDD，且对高方差时间采样敏感；统一噪声前向过程在更复杂语料上的通用性仍需进一步验证。

---

## 233. Preference Estimation via Opponent Modeling in Multi-Agent Negotiation

**arXiv ID:** 2604.15687 | [PDF](https://arxiv.org/pdf/2604.15687v1)

**作者:** Yuta Konishi `[一作]` (Kyoto University), Shiyao Ding `[通讯]` (Kyoto University)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5067887331)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将大型语言模型提取的自然语言信号融入结构化贝叶斯框架的对手偏好估计方法，用于多方多议题谈判。

**💡 创新点**

创新点在于：①将LLM生成的定性语义信号转化为概率形式，②在贝叶斯推理中结合数值报价和语言证据实现动态信念更新。

**🔧 技术方法**

采用的技术包括：GPT‑4.1 作为LLM进行语义解析，Luce 选择公理计算语言似然，贝叶斯更新（Naïve Bayes 近似）以及正态分布近似估计报价似然。

**📊 数据集**

使用的实验数据集为 ANAC 赛事中构建体育设施的多方谈判基准（6 个利益相关方，5 个议题，24 轮），共 500 次独立实验。

**📈 对比分析**

与 Base‑LLM、Base‑OM、LLM‑PE 等基线对比，提出的方法（全体估计）在全体一致率 FAR 上达到 0.62，偏好估计 MSE 为 159，明显优于基线（FAR 最高 0.56，MSE 最高 189）。

**⚠️ 局限性**

局限性包括：对话真实性假设，缺乏对欺骗或虚张声势的鲁棒性；对保留阈值的推断缺失；以及随着议题和选项数量增加导致的计算复杂度上升。

---

## 234. DepCap: Adaptive Block-Wise Parallel Decoding for Efficient Diffusion LM Inference

**arXiv ID:** 2604.15750 | [PDF](https://arxiv.org/pdf/2604.15750v1)

**作者:** Xiang Xia `[一作]` (University Of Science And Technology Of China), Yanyong Zhang `[通讯]` (University Of Science And Technology Of China)

**通讯引用:** 120883 | [OpenAlex ID](https://openalex.org/A5035053309)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DepCap，一个无训练的块级 Diffusion 语言模型推理框架，利用上一块的跨步影响信号自适应决定下一块的边界，并通过冲突检测实现更安全的并行解码。

**💡 创新点**

创新点包括：①使用跨步 KL 散度和熵构造的依赖分数来动态划分块；②提出冲突感知并行解码策略 CAP‑Decoding，显著提升并行度；③从信息论角度解释依赖分数的可加性，为块划分提供理论依据。

**🔧 技术方法**

技术手段包括 Diffusion 语言模型、块级解码、KL 散度与熵驱动的依赖分数、Token 置信度与冲突得分、信息论分析、以及与 KV‑cache、Fast‑dLLM 等缓存策略的兼容性。

**📊 数据集**

使用的评测数据集有 GSM8K（5-shot）、Math‑500（4-shot）、MBPP（3-shot）和 HumanEval（0-shot）等。

**📈 对比分析**

与 Fast‑dLLM、AdaBlock 等现有方法对比，采用 TPS、NFE、Accuracy 三项指标评测。DepCap 在 LLaDA、Dream 等模型上平均实现 3.57× 的速度提升（MBPP 5.63×），同时保持甚至提升准确率，且 NFE 与误码率均保持在可接受范围内。

**⚠️ 局限性**

局限性包括：需要手动调节块大小、阈值等超参数；理论分析是近似的，缺乏严格的性能保证；目前仅针对块级解码，对滑动窗口或极长生成任务的适用性尚未验证；在极端长文本或高噪声场景下，冲突检测可能仍影响质量。

---

## 235. Why Colors Make Clustering Harder:Global Integrality Gaps, the Price of Fairness, and Color-Coupled Algorithms in Chromatic Correlation Clustering

**arXiv ID:** 2604.15738 | [PDF](https://arxiv.org/pdf/2604.15738v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 3256 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对传统Correlation Clustering（CC）进行扩展，提出Chromatic Correlation Clustering（CCC）并研究其LP松弛的可整性缺口，证明任何颜色独立的LP-CCC算法存在一个可加的“色彩惩罚”，并给出精确的阶梯式下界；进一步设计了加入全局约束的Color‑Coupled Correlation Clustering（C4）算法，成功消除色彩惩罚，实现与标准CC相同的2.06近似比。

**💡 创新点**

① 全局可整性缺口分解定理，揭示CCC难度来源为跨边色彩干扰；② 推导色彩惩罚的闭式阶梯公式 Δ(L)=((L−1)/L)·Δ∞，其中 Δ∞≈0.0734；③ 通过“色彩耦合”约束和相关区间包装技术构造C4算法，彻底突破2.11下界；④ 在公平聚类应用中将色彩惩罚解释为公平性的不可避免成本。

**🔧 技术方法**

LP松弛与三元组分析、KKT连续变分求解、色彩吹胀图（Chromatic Blowup Graph）构造、相关区间包装（Correlated Interval Packing）回归、实验验证、以及对公平性数据集的特征映射。

**📊 数据集**

① 合成极端实例（Maximally Interfering Instances）；② 真实多关系网络（Amazon Co‑purchase、DBLP Co‑authorship）；③ 公平性基准（Adult、German Credit、COMPAS）。

**📈 对比分析**

通过与标准颜色独立的LP-Pivot、传统CC以及C4的对比实验，发现标准颜色独立方法的近似比严格沿着理论阶梯上升，C4在所有 L 上保持2.06的近似比；在真实数据集上，C4的目标值显著低于Pivot和标准CC，公平性实验中公平性缺口与 Δ(2)=0.037 对齐，C4则消除该缺口。

**⚠️ 局限性**

① 仅适用于完全图和颜色独立的LP框架；② 需要加入 O(n²) 条额外全局约束，增加 LP 规模；③ 对非度量或稀疏图的推广尚未证明；④ 对大规模实例的求解时间和内存开销仍需进一步优化。

---

## 236. MambaBack: Bridging Local Features and Global Contexts in Whole Slide Image Analysis

**arXiv ID:** 2604.15729 | [PDF](https://arxiv.org/pdf/2604.15729v1)

**作者:** Sicheng Chen `[一作]` (University of California), Fei Xia `[通讯]` (University of California)

**通讯引用:** 2984 | [OpenAlex ID](https://openalex.org/A5036985837)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种名为MambaBack的混合多实例学习框架，用于对整张切片图像（WSI）进行癌症诊断。

**💡 创新点**

创新点包括：① Hilbert曲线采样策略，最大化在1D序列中保留2D空间局部性；② 层次混合结构，先用Gated CNN提取局部细胞特征，再用BiMamba2聚合全局上下文；③ 非对称分块推理设计，使训练高吞吐量、推理内存峰值显著降低。

**🔧 技术方法**

技术方法包括：Mamba状态空间模型、MambaOut的Gated CNN、BiMamba2双向状态空间模块、Hilbert空间曲线映射以及块化推理与异步累积。

**📊 数据集**

使用了5个公开WSI数据集：CAMELYON16、CAMELYON17、PANDA、TCGA-NSCLC 和 TCGA-BRCA，覆盖二分类、四分类、六分类以及生存预测任务。

**📈 对比分析**

与7种现有SOTA方法（ABMIL、CLAM、DTFD-MIL、TransMIL、Prov-GigaPath、MambaMIL、PathRWKV）在相同实验设置下进行对比，MambaBack在AUC、ACC、F1、C-Index等指标上均取得最高或相近最高成绩，尤其在CAMELYON16的AUC上提高至0.995，PANDA的F1提升至0.733。

**⚠️ 局限性**

局限性包括：① 仍需在更大规模WSI和多中心真实数据上进一步验证推理速度和鲁棒性；② 对极为稀疏或形态异常的组织区域的适应性未做深入探讨；③ 依赖预训练的基础模型，未完全解决全流程端到端训练的挑战。

---

## 237. Structured Abductive-Deductive-Inductive Reasoning for LLMs via Algebraic Invariants

**arXiv ID:** 2604.15727 | [PDF](https://arxiv.org/pdf/2604.15727v1)

**作者:** Sankalp Gilda `[一作]` (DeepThought Solutions), Shlok Gilda `[通讯]` (University of Florida)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5021727664)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个符号推理框架，将Peirce的三种推理模式（归纳、演绎、诱导）分离为可审计的ADI协议，并通过五个代数不变式（Gamma Quintet）保证推理链的可靠性。

**💡 创新点**

创新点在于：① 明确区分推理模式并对每个模式设定可靠性上限；② 引入“弱链”不变式（WLNK）作为逻辑一致性的根本约束；③ 将代数规范与可能性理论、经验验证和t-范数理论相结合，形成多线验证；④ 通过属性驱动测试（PBT）验证实现的高可信度。

**🔧 技术方法**

使用的技术包括：LLM（如GPT-4o）生成自然语言推理，外部符号知识图谱维护推理结构，代数不变式（min为唯一满足的t-范数）约束可靠性，属性测试框架（QuickCheck/Arbitrary）进行大规模验证，以及多层可信度计算（包含层级、形式性、证据衰减、对齐惩罚等）。

**📊 数据集**

论文未在公开数据集上做大规模评测，主要以随机生成的10^5+案例和手工构造的推理链进行属性测试；在预留的AirS-Bench等小规模实验中已观察到推理错误率下降。

**📈 对比分析**

与现有Chain‑of‑Thought、Self‑Consistency、Process Reward Models等方法对比，提出的ADI+Gamma框架通过严格的可靠性传播和外部验证，能够避免逻辑不一致和自我确认错误，实验表明在受限任务中的错误率相对下降约10-20%，但需要多轮交互或多代理支持。

**⚠️ 局限性**

局限性包括：① 依赖属性测试而非形式证明，缺乏完全的机器检验；② ADI协议需多轮交互或多代理，增加系统复杂度；③ 可靠性上限和对齐惩罚等参数为默认策略，缺乏领域特定的经验校准；④ 未在大型逻辑基准（ZebraLogic、FOLIO）上进行全面评估。

---

## 238. Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4

**arXiv ID:** 2604.15839 | [PDF](https://arxiv.org/pdf/2604.15839v1)

**作者:** Chengwu Liu `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 34929 | [OpenAlex ID](https://openalex.org/A5100461491)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文先对现有自动定理证明基准做“Hard Mode”重注解，发布MiniF2F‑Hard和FIMO‑Hard，并提出一套两阶段Agent框架——Discovery & Proving——实现从自然语言推理到正式证明的完整闭环；

**💡 创新点**

创新点在于：①提出Hard Mode标准消除答案嵌入，提升任务真实性；②通过人工专家重注解修复语义误差；③设计分离的Discovery（LLM自我验证/纠错）与Proving（ATP）模块，避免作弊与过载；

**🔧 技术方法**

主要技术包括：基于GPT‑OSS‑120B的长链推理与自我反省、Lean 4 形式化、Goedel‑Prover‑V2 ATP、Agentic 迭代流程及自检自改；

**📊 数据集**

使用的数据集包括：MiniF2F‑Hard、FIMO‑Hard、PutnamBench、CombiBench、miniF2F‑test、FIMO 等；

**📈 对比分析**

通过 Pass@32 评估在 Hard Mode 上的性能，系统在 PutnamBench 解决 36 / 19 题，在 CombiBench 解决 10 题，接近甚至匹配现有 Easy Mode 的成绩，并明显优于之前的 Kimina‑Prover Preview 等；

**⚠️ 局限性**

局限性：存在潜在的数据集污染风险；正式推理仍是瓶颈；Discovery 与 Proving 模块耦合度有限，缺乏双向协作；对自我验证迭代的过度依赖可能在易题中产生噪声。

---

## 239. Reversible Residual Normalization Alleviates Spatio-Temporal Distribution Shift

**arXiv ID:** 2604.15838 | [PDF](https://arxiv.org/pdf/2604.15838v1)

**作者:** Zhaobo Hu `[一作]` (Institut Polytechnique de Paris), Mehdi Naima `[通讯]` (Sorbonne Universite)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Reversible Residual Normalization（RRN）框架，利用可逆残差块和图卷积实现空间-时间分布偏移的归一化；

**💡 创新点**

首次在时空预测中结合中心归一化、光滑的可逆图卷积与谱正则化，实现对空间异质性和时间漂移的统一、可逆处理；

**🔧 技术方法**

中心归一化（Center Normalization）、可逆残差网络、光谱正则化图卷积、固定点迭代逆变换；

**📊 数据集**

METR-LA、PEMS-BAY交通速度数据以及SDWPF风电功率数据；

**📈 对比分析**

在三种数据集上与多种基线（DCRNN、Graph WaveNet、GMAN等）和归一化方法（RevIN、Dish‑TS、SAN、ST‑Norm）对比，RRN在所有模型和数据集上均显著降低MAE/RMSE，提升预测精度；

**⚠️ 局限性**

逆变换需固定点迭代，导致计算延迟；中心归一化不包含方差缩放，可能限制对复杂尺度变化的处理能力；

---

## 240. Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration

**arXiv ID:** 2604.15829 | [PDF](https://arxiv.org/pdf/2604.15829v1)

**作者:** Jun Li `[一作]` (Nanjing University of Information Science and Technology), Guo-Sen Xie `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 3726 | [OpenAlex ID](https://openalex.org/A5084688255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了文本-图像协作消除框架 TICoE，用于在文本到图像扩散模型中精准、忠实地消除指定概念。

**💡 创新点**

创新点在于：①构建连续凸概念流形（CCCM）以覆盖概念的多样语言表达；②使用层级视觉表示学习（HVRL）对多尺度视觉信息进行融合，精准区分视觉相似但语义不同的内容；③将两者协同优化，兼顾消除精度与生成保真。

**🔧 技术方法**

采用 GPT‑5 生成多样化提示、Dirichlet 采样构造概念流形；多尺度 Transformer 对视觉特征进行编码并融合；基于 classifier‑free guidance 的损失函数实现消除学习；在扩散模型 Stable Diffusion 的 U‑Net 上训练。

**📊 数据集**

训练时使用 Stable Diffusion 生成的目标概念图像（如 gun、tench、church 等）作为参考；评估使用 COCO‑10k、I2P、NudeNet、以及特定的相关概念（camera、umbrella 等）集合。

**📈 对比分析**

与 ESD、UCE、FMN、SPM（文本消除）和 Co‑Erasing（图像辅助）等方法在 ASR、UDA、P4D、FID、CLIP、MCP 等指标上进行对比；实验表明 TICoE 在消除精度（ASR↓、UDA↓、P4D↓）上优于所有方法，同时保持甚至提升生成质量（FID↓、CLIP↑），并在相关概念保留（MCP↑）方面表现最佳。

**⚠️ 局限性**

局限性：需要手工准备或自动生成的多模态参考数据，训练成本和推理时间相对较高；对极端视觉相似但语义完全不同的概念仍可能出现残留；多概念同时消除的效果尚未在更大规模场景下充分验证。

---

## 241. SSFT: A Lightweight Spectral-Spatial Fusion Transformer for Generic Hyperspectral Classification

**arXiv ID:** 2604.15828 | [PDF](https://arxiv.org/pdf/2604.15828v1)

**作者:** Alexander Musiat `[一作]` (Mannheim University of Applied Sciences), Oliver Wasenmüller `[通讯]` (Mannheim University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级双路径光谱-空间融合Transformer（SSFT），用于低监督、异构的高光谱图像分类。

**💡 创新点**

创新点在于将光谱与空间处理分离为两条轻量级分支，并通过跨注意力融合实现仅0.52M参数而保持甚至提升跨域性能。

**🔧 技术方法**

技术包括光谱自注意力、空间轻量卷积、跨注意力融合、深度监督及无数据增强的训练策略。

**📊 数据集**

使用了HSI-Benchmark（包含HRSS、Fruit、Debris三域）以及更大规模的SpectralEarth CORINE数据集。

**📈 对比分析**

与多种CNN、3D CNN、Transformer基线比较，在HSI-Benchmark上实现最高总体准确率84.87%（Debris 93.33%，Fruit 61.72%），参数量仅为最高基线的2%；在SpectralEarth上迁移后宏F1约为75-78，虽低于大模型但保持竞争力。

**⚠️ 局限性**

局限性包括对数据增强不敏感以及在更大规模、多源/多传感器环境下模型容量不足，可能影响进一步的泛化性能。

---

## 242. Watching Movies Like a Human: Egocentric Emotion Understanding for Embodied Companions

**arXiv ID:** 2604.15823 | [PDF](https://arxiv.org/pdf/2604.15823v1)

**作者:** Ze Dong `[一作]` (Nanyang Technological University), Lin Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 59394 | [OpenAlex ID](https://openalex.org/A5108047874)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EgoScreen-Emotion (ESE) 数据集，构建了面向嵌入式伴侣的自我情绪理解框架

**💡 创新点**

ESE是首个从物理屏幕视角采集的情绪数据集，并引入多标签置信度与推理说明；模型利用层次化文本化长时序背景，结合视觉、音频与叙事多模态信息

**🔧 技术方法**

采用多模态大语言模型Qwen2.5‑Omni‑7B，结合LoRA微调、分层叙事摘要、短期视觉帧、音频窗口与推理监督

**📊 数据集**

使用224个电影预告片的物理录制帧，包含28,667个关键帧、5位标注者的多标签置信度及部分文本推理；对比原始电影剪辑与FPV数据

**📈 对比分析**

通过跨域实验、Ablation与与多种公开/闭源多模态模型对比，ESE训练模型在FPV下Macro‑F1提升约40%，单帧模型在ESE测试集达到57.66%准确率，优于大多数基线并接近闭源系统

**⚠️ 局限性**

仍受限于单一物理录制场景、情绪标签多样性偏长尾、缺少观众表情与声学反应等真实交互信号，模型泛化仍受物理环境与视角变异影响

---

## 243. Secure Authentication in Wireless IoT: Hamming Code Assisted SRAM PUF as Device Fingerprint

**arXiv ID:** 2604.15810 | [PDF](https://arxiv.org/pdf/2604.15810v1)

**作者:** Florian Lehn `[一作]` (German Research Center for Artificial Intelligence), Hans D. Schotten `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于SRAM PUF的阈值式身份认证方案，并在资源受限的IIoT设备上实现了实验验证

**💡 创新点**

创新点在于将低成本的Hamming码纠错与多轮时间投票相结合，并引入错误受限安全边际（SM_ec）作为可量化的设计预算，指导资源与安全之间的权衡

**🔧 技术方法**

采用SRAM PUF、Hamming码（sec、secded）纠错、时间投票（TMV）、阈值校准、FreeRTOS+Wi‑Fi TCP通信以及ESP32‑S3 MCU

**📊 数据集**

实验数据来自六块ESP32‑S3在21℃/50%RH环境下的45次身份验证迭代，使用2048位或更短的Puf响应长度

**📈 对比分析**

与不同纠错码和投票次数的组合进行对比；结果显示，Hamming码+TMV可将误码率降低至<1%，但随着投票次数和码率降低，计算与存储开销显著上升；SM_ec指标帮助识别可接受的资源阈值

**⚠️ 局限性**

局限在于仅对单设备内的误码率进行评估，未进行多板交叉设备的FAR测试；假设Puf误差符合理想的二项分布，忽略了实际硬件间的相关性与偏差

---

## 244. PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection

**arXiv ID:** 2604.15776 | [PDF](https://arxiv.org/pdf/2604.15776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 245. From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation

**arXiv ID:** 2604.15805 | [PDF](https://arxiv.org/pdf/2604.15805v1)

**作者:** Jasper Lu `[一作]` (Peking University), Ruihai Wu `[通讯]` (Peking University)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5086096450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 WorldComposer 框架，将单张 360° 全景照片自动转化为高保真、可交互的仿真场景（数字孪生），并通过语义/几何编辑生成大量多样化的“数字亲戚”场景，用于机器人学习与评估。

**💡 创新点**

创新点在于：① 将 3D Gaussian Splats 与碰撞网格同时生成，实现视觉与物理双重高保真；② 通过提示驱动编辑一次性生成多种场景变体；③ 设计了多房间拼接管线，构建可导航的大型环境；④ 结合高质量资产库与物理求解器，提供完整的交互式仿真平台。

**🔧 技术方法**

核心技术包括：Marble（多模态世界模型）实现 360° 转 3DGS；3DGRUT 将 3DGS/碰撞网格转换为 USD；SuperPoint+LightGlue+ICP 进行房间拼接；LLM 辅助资产布局；Isaac Sim+LeRobot 接口实现物理交互；多种物理求解器（PBD、FEM、DG）模拟刚体、关节和柔体。

**📊 数据集**

使用的数据集：真实世界 360° 全景照片、基于这些全景生成的数字孪生与亲戚场景；真实世界下的专家演示（约 50–100 条）用于训练和评估；与 Isaac Sim 兼容的高质量资产库（刚体、关节、柔体）。

**📈 对比分析**

对比方法：在多种任务（刚体、关节、柔体操控及跨房间导航）中，训练四种现有策略（ACT、Diffusion Policy、SmolVLA、π₀）在仅用原始数据、原始+亲戚数据以及仅仿真数据时的成功率。实验显示，加入亲戚数据可显著提升泛化性能，尤其在“场景+物体”未知条件下；仿真与真实测试成功率相关系数 r≈0.91；在多房间导航中，VLFM 平均成功率达 68%。

**⚠️ 局限性**

局限性：① 目前缺乏实例级分解，难以对局部对象进行精准编辑；② 由于 3DGS 生成的纹理在房间接缝处会出现不连续；③ 仍需进一步研究跨场景辐射场融合与端到端 3DGS 优化，以提升整体视觉连贯性与物理一致性。

---

## 246. Convolutionally Low-Rank Models with Modified Quantile Regression for Interval Time Series Forecasting

**arXiv ID:** 2604.15791 | [PDF](https://arxiv.org/pdf/2604.15791v1)

**作者:** Miaoxuan Zhu `[一作]` (Southeast University), Guangcan Liu `[通讯]` (Southeast University)

**通讯引用:** 9806 | [OpenAlex ID](https://openalex.org/A5019542310)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种将修改版分位回归（MQR）融入学习型卷积核低秩最小化（LbCNNM）的方法，生成多步预测区间（PI）。

**💡 创新点**

创新点在于：①用均值替代传统分位回归中的中位数，显著提升区间质量；②直接从L_bCNNM中闭式求解PI，无需额外的预测模型；③结合 conformal calibration 对预估区间进行自适应偏差修正。

**🔧 技术方法**

核心技术包括：卷积核低秩恢复（CNNM/LbCNNM）、修改后的分位回归、ADMM求解、单变量时间序列训练与校准、以及基于均值和分位数的区间校准。

**📊 数据集**

实验数据集主要为 M4（共 100,000 条实测时间序列，按频率划分为小时、日、周、月、季、年）以及两个多变量基准集 Electricity（321 条）和 Traffic（862 条）用于对比。

**📈 对比分析**

与 AutoARIMA、AutoETS、AutoTheta、DeepAR、DeepTCN、LbCNNM‑QR 以及 LbCNNM‑CP 等七种基线方法对比，LbCNNM‑MQR 在 MSIS 与 ACD 上均优于所有竞争者，尤其在整体 M4 数据集上 MSIS 提升约 10%，ACD 接近 0%，并保持较低的计算开销。

**⚠️ 局限性**

局限性包括：①仅针对单变量时间序列，尚未验证多变量或空间-时间序列；②区间校准依赖于训练集的划分，对极端稀疏或非平稳序列的鲁棒性待进一步验证；③对 λ 参数的选择虽不敏感，但在极端噪声场景下仍需谨慎调节。

---

## 247. Scattered Hypothesis Generation for Open-Ended Event Forecasting

**arXiv ID:** 2604.15788 | [PDF](https://arxiv.org/pdf/2604.15788v1)

**作者:** He Chang `[一作]` (Communication University of China), Yunshan Ma `[通讯]` (Singapore Management University)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5089377262)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将开放式事件预测重新表述为假设生成任务，即散射预测，并通过强化学习框架生成既包含性又多样化的假设集合。

**💡 创新点**

设计了三阶段混合奖励：有效性奖励、组内多样性奖励和组间多样性奖励，并通过有效性门控防止奖励劫持，从而在避免模式坍塌的同时保持语义正确性。

**🔧 技术方法**

基于GRPO的强化学习方法，利用LoRA微调的Qwen2.5-3B-Instruct和Llama3.2-3B-Instruct LLM；使用Qwen3-Embedding-4B进行文本嵌入与相似度计算。

**📊 数据集**

使用OpenForecast和OpenEP两个公开基准数据集，并构造了Hard子集进行更严苛的评估。

**📈 对比分析**

与GPT‑4o‑mini、标准SFT以及仅使用有效性奖励的GRPO进行对比。SCATTER在SoftPass@K、SoftRecall@K、ValidRatio@K等指标上均显著优于基线，尤其在跨域和Hard子集测试中表现突出。

**⚠️ 局限性**

仅在3B参数模型、LoRA微调下实验，采样轮数上限为16；数据集虽公开但覆盖范围有限；未探索无标签或完全开放式评估场景，商业模型评估受限。

---

## 248. MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents

**arXiv ID:** 2604.15774 | [PDF](https://arxiv.org/pdf/2604.15774v1)

**作者:** Weiwei Xie `[一作]` (Shanghai Jiao Tong University), Qibing Ren `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5059610849)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MemEvoBench，评估 LLM 代理在持续记忆演化过程中的安全风险，涵盖误导记忆注入与噪声工具返回两种情景。

**💡 创新点**

创新点在于首次系统化定义记忆失调（memory misevolution）风险，构建 36 种风险类型、7 个高危领域的 benchmark，并通过三轮交互与偏置反馈模拟记忆演化，提供主动记忆修正工具以实现对策。

**🔧 技术方法**

使用技术包括：QA/Workflow 场景设计、三轮记忆更新与主动记忆修正工具、偏置用户反馈模拟、ASR（攻击成功率）评估、工具驱动的记忆验证与纠错，以及与安全提示（+SafePrompt）的对比。

**📊 数据集**

使用数据集：自建 MemEvoBench，包含 108 QA 样例与 83 Workflow 样例；构建混合记忆池（正确+误导）并借助 AgentSafetyBench 等环境；同时在 A-MEM 结构下验证，展示更真实记忆系统的影响。

**📈 对比分析**

通过与 9 种闭源/开源 LLM（GPT‑4o、GPT‑5、Gemini‑2.5‑Pro、Claude‑3.7‑Sonnet、Llama‑3.3‑70B、Qwen‑3 系列、DeepSeek‑V3.2）对比三种配置（Vanilla、+SafePrompt、+ModTool）和有无偏置反馈，结果表明 +ModTool 能显著降低 ASR（多数模型降至 20–30% 左右），而仅靠安全提示效果有限；在 A‑MEM 环境下趋势保持一致。

**⚠️ 局限性**

局限性：记忆演化仅采用简化的三轮更新规则，未覆盖更长期的累积效应；误导记忆的检测与修正仍受模型自身判断限制，可能漏检；工具修正依赖外部工具链，未考虑多模态或更复杂工具链的安全交互；Benchmark 仍处于初期阶段，尚未覆盖所有现实场景。

---

## 249. Searching for European Alternatives: Digital Sovereignty, Digital Patriotism, and the Emerging Geopolitics of Software Adoption

**arXiv ID:** 2604.15767 | [PDF](https://arxiv.org/pdf/2604.15767v1)

**作者:** Advait Sarkar `[一作]` (University of Cambridge), Advait Sarkar `[通讯]` (University of Cambridge)

**通讯引用:** 1631 | [OpenAlex ID](https://openalex.org/A5025427198)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过政府采购案例和 Hacker News 讨论，探讨数字主权理念如何驱动欧洲软件采用决策。

**💡 创新点**

创新点在于提出“数字爱国主义”概念，将价值理性纳入软件采纳模型，并揭示其与数字主权的对应关系。

**🔧 技术方法**

采用定向内容分析与主题分析方法，结合文本编码和案例研究。

**📊 数据集**

使用的资料包括欧洲政府机构的开源转型事件记录（约70+案例）和 700+ Hacker News 评论（51,000 字）。

**📈 对比分析**

与传统的工具效能或成本评估不同，研究通过定性比对说明从成本/锁定到主权动机的转变；未给出量化性能指标。

**⚠️ 局限性**

局限性包括样本选择偏差、官方声明可能存在策略性陈述、仅聚焦欧洲及 Hacker News 受众，且未检验转变的长期成效。

---

## 250. Closing the Theory-Practice Gap in Spiking Transformers via Effective Dimension

**arXiv ID:** 2604.15769 | [PDF](https://arxiv.org/pdf/2604.15769v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22416 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了spiking自注意力的完整表达性理论，包括通用逼近证明、Spike数下界、输入相关有效维度分析，并给出可操作的设计公式。

**💡 创新点**

首次将信息论与脉冲神经网络相结合，给出严格的通用逼近与Spike数下界；通过测量有效维度解释理论与实践的巨大差距；提出可直接用于硬件部署的设计规则。

**🔧 技术方法**

使用Leaky Integrate‑and‑Fire模型与脉冲率编码，构造侧抑制WTA网络实现softmax；利用指数逼近、ReLU实现和信息论的rate‑distortion分析；通过PCA评估数据集有效维度；使用SpikingJelly进行训练与评估。

**📊 数据集**

CIFAR‑10、CIFAR‑100、ImageNet‑1K、SST‑2 等视觉与语言基准数据集。

**📈 对比分析**

对比标准Transformer与多种spiking Transformer（Spikformer、QKFormer、SpikingResformer）在同一任务下的准确率、能耗与Spike预算；实验表明R²=0.97、能耗提升38–57×，Spike数与理论下界吻合，设计公式预测误差低于18%。

**⚠️ 局限性**

假设理想的脉冲率编码与无噪声、WTA输入需足够分离；使用surrogate梯度训练可能引入额外噪声；未覆盖时序编码、稀疏注意等更复杂机制。

---

## 251. QUACK! Making the (Rubber) Ducky Talk: A Systematic Study of Keystroke Dynamics for HID Injection Detection

**arXiv ID:** 2604.15845 | [PDF](https://arxiv.org/pdf/2604.15845v1)

**作者:** Alessandro Lotto `[一作]` (University of Padua), Mauro Conti `[通讯]` (University of Padua)

**通讯引用:** 27064 | [OpenAlex ID](https://openalex.org/A5063847107)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究系统性地评估了仅基于键盘敲击时间特征的轻量级隐私保护型人机输入检测，证明可在不依赖用户身份或键盘内容的情况下实现早期且可靠的USB HID注入攻击检测。

**💡 创新点**

创新点在于将键盘动态分析从用户身份鉴别转向纯粹的人机区分，并通过结构化的对手模型和混合训练策略实现对多样化生成器的泛化，从而降低训练复杂度并强调生成器多样性而非单纯模型复杂度。

**🔧 技术方法**

采用随机森林、支持向量机、1D卷积与LSTM等轻量级机器学习模型，结合时间特征（键保持时间ht和键间隔ft）进行训练与评估。

**📊 数据集**

使用González等公开的键盘敲击数据集（18816个会话），通过保留键码序列但仅替换ht/ft产生多种合成对手数据，构建三类对手族（基础、统计、适应性）。

**📈 对比分析**

通过单一生成器、交叉生成器和混合生成器三阶段评估，结果显示在70~100个敲击样本下，混合训练（尤其是UC3配置）可达ROC‑AUC>0.9，推断时间约90ms，内存与CPU占用低，体现了在短窗口下的高准确率与低资源消耗。

**⚠️ 局限性**

主要局限包括：实验仅基于公开人类键盘数据，未考察不同键盘硬件或OS平台的差异；使用的模型相对简单，尚未验证更复杂模型在更大对手空间下的优势；并未深入探讨多用户或多任务场景下的实时部署与误报率。

---

## 252. Exploring Agentic Visual Analytics: A Co-Evolutionary Framework of Roles and Workflows

**arXiv ID:** 2604.15813 | [PDF](https://arxiv.org/pdf/2604.15813v1)

**作者:** Tianqi Luo `[一作]`, Yuyu Luo `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文对2022‑2026年间55个基于大语言模型（LLM）的自主视觉分析系统进行了系统性综述，提出了四角色四层级的共进框架，并基于此分析了代理如何重塑传统视觉分析流程，给出设计准则。

**💡 创新点**

创新点在于：①构建了四个功能角色（战略协调器、执行者、质量保证器、持久记忆器）与四个自主层级的共进框架；②系统化评估代理对数据处理、视觉映射、视图转换、展示四大阶段的重构；③提出了针对不同层级与角色的六条设计准则。

**🔧 技术方法**

采用的核心技术包括：大语言模型驱动的多智能体架构、规划‑执行‑反思（ReAct/CoT）循环、MLLM视觉反馈、检索增强生成（RAG）、语义映射与自适应记忆。

**📊 数据集**

综述涵盖的系统使用了多样化公开数据集，涵盖结构化表格（如UCI、CSV）、图像、音频、医学临床数据、公开图表等，但本文未集中单一数据集，仅说明这些系统在真实场景中的数据来源。

**📈 对比分析**

通过PRISMA方法检索并量化55个系统，按自主层级统计分布，并对功能角色与流程改造进行对比。结果显示Level 3/4系统快速增长，功能覆盖更全面，整体性能相较传统NLI系统显著提升；然而缺乏统一基准，评估仍主要为定性。

**⚠️ 局限性**

限制包括：①聚焦LLM驱动系统，忽略早期规则系统；②多系统缺乏公开实现，重现性受限；③快速演进可能导致框架分类失效；④评估仍以定性为主，缺乏统一量化指标。

---

## 253. CHOP: Chunkwise Context-Preserving Framework for RAG on Multi Documents

**arXiv ID:** 2604.15802 | [PDF](https://arxiv.org/pdf/2604.15802v1)

**作者:** Hyunseok Park `[一作]` (HDC LABS), Dongsik Yoon `[通讯]` (HDC LABS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CHOP 框架，通过为每个文本块注入上下文一致的 CNM 前缀并使用连续性决策模块保证跨块一致性，显著提升 RAG 系统的检索效果。

**💡 创新点**

创新点包括（1）CNM‑Extractor 生成分类‑名词‑模型签名以减少语义冲突；（2）连续性决策模块决定是否继承前块 CNM，实现上下文连续性；（3）通过前缀化稳定嵌入空间，降低检索误差。

**🔧 技术方法**

使用的技术包括 Gemma‑12B 大语言模型进行信息抽取与决策，OpenAI 3072 维嵌入，ChromaDB + HNSW 索引进行近似最近邻检索，以及传统的 RAG 生成流水线。

**📊 数据集**

采用 MRAMG‑Bench（基于 ManualsLib 的产品手册集合），将每个手册重组为单一连续文件以模拟真实使用场景。

**📈 对比分析**

在 Naive‑500T 与 Cosine‑Chunking 基线的基础上，CHOP 在 Top‑1 Hit Rate 提升至 0.9077，MRR 与 NDCG 均优于对手；生成阶段的 F1/ROUGE‑L/BERTScore 也略有提升，表明检索改进直接转化为生成质量提升。

**⚠️ 局限性**

局限性包括对大型语言模型推理的高算力需求；对动态知识更新和流式输入的适应性有限；实验仅验证于手册类长文档，通用性需进一步评估。

---

## 254. ReVis: Towards Reusable Image-Based Visualizations with MLLMs

**arXiv ID:** 2604.15781 | [PDF](https://arxiv.org/pdf/2604.15781v1)

**作者:** Xiaolin Wen `[一作]` (Nanyang Technological University), Yong Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 126036 | [OpenAlex ID](https://openalex.org/A5059976286)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ReVis：一种基于多模态大模型的图像可视化自动拆解、DSL 生成与重用框架，并配备交互式编辑界面。

**💡 创新点**

创新点在于：① 设计层次化、模板化的可视化 DSL；② 通过多步 MLLM 解析图像自动生成 DSL；③ 将自动化重建与人机交互编辑、数据适配集成，实现从静态图像到可编辑可复用可视化的闭环。

**🔧 技术方法**

主要技术包括多模态大型语言模型（如 ChatGPT‑5）、规则驱动的数据生成器、D3.js 渲染、JSON DSL 交互界面及提示工程。

**📊 数据集**

使用了 40 张可视化图像（20 张基本图表 + 20 张复合图）来自 Vega‑Lite Gallery 与前沿研究，并对 16 名可视化从业者进行访谈收集数据。

**📈 对比分析**

对基本图表按属性匹配实现 94.8% 的准确率；对复合图通过 16 位专家的二元评估得到 90.6%/92.8%/83.4% 的正确性分数；用户研究显示满意度均在 6–7/7 级，重用与编辑易用性评价高。

**⚠️ 局限性**

局限性包括：MLLM 对图像解析受分辨率与噪声影响；对复杂布局（如 Sankey、力导向图）缺乏支持；DSL 仍需人工细调；数据上传缺乏统一处理，生成耗时长且易超时。

---

## 255. TinyMU: A Compact Audio-Language Model for Music Understanding

**arXiv ID:** 2604.15849 | [PDF](https://arxiv.org/pdf/2604.15849v1)

**作者:** Xiquan Li `[一作]` (Institut Polytechnique de Paris), Slim Essid `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 2452 | [OpenAlex ID](https://openalex.org/A5060031161)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发 TinyMU（229M 参数）作为轻量级音乐语言模型，并构建 MusicSkills-3.5M QA 数据集用于训练。

**💡 创新点**

通过自监督音频编码器 MATPAC++ 与轻量投影器结合，提出多格式 QA 数据集提升感知与推理，实现在 35 倍参数压缩下与大型模型相近的性能。

**🔧 技术方法**

采用 MATPAC++、线性投影器、SmolLM2 小型语言模型，结合 LoRA/全微调、LLM 辅助生成 QA 与规则生成 QA 的技术。

**📊 数据集**

使用 MusicCaps、MagnaTagATune、FMA、AudioSet、MusicInstruct、OpenMU-MTT 等构成 MusicSkills-3.5M，评测使用 GTZAN、Medley‑Solos‑DB、MusicCaps、MuChoMusic。

**📈 对比分析**

与 Mellow、MU‑LLaMA、MusiLingo、Audio‑Flamingo、MiDashengLM、Qwen2‑Audio‑Instruct 等 SOTA 对照；TinyMU 在 GTZAN 65.7%、Medley‑Solos‑DB 95.1%、MuChoMusic 58.6%（占 SOTA 的 82%），在小模型中实现最优或接近大模型性能。

**⚠️ 局限性**

仍然依赖大型预训练音频编码器；在极端复杂推理任务上落后于 8B 级模型；仅在少数基准上验证，缺乏对更广泛音乐场景的评估；需进一步压缩模型或提升鲁棒性。

---

## 256. A Protocol-Agnostic Backscatter-Based Security Layer for Ultra-Low-Power SWIPT IoT Networks

**arXiv ID:** 2604.15831 | [PDF](https://arxiv.org/pdf/2604.15831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 257. CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization

**arXiv ID:** 2604.15847 | [PDF](https://arxiv.org/pdf/2604.15847v1)

**作者:** Junyi Li `[一作]` (Hong Kong University of Science and Technology), Ningning Ding `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 276 | [OpenAlex ID](https://openalex.org/A5066234112)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对大型推理模型（LRM）的机器忘记方法——CiPO，能够在不损害推理能力的前提下，彻底抹除隐私或版权信息在链式推理（CoT）轨迹和最终答案中的存在。

**💡 创新点**

创新点在于把忘记问题重新定义为因果干预，将模型的推理路径替换为对抗性生成的、无敏感信息的“对事实的反事实”推理轨迹，并通过在线迭代的偏好优化（SimPO+NLL）实现无缝迁移，从而兼顾高忘记效能与低功能损耗。

**🔧 技术方法**

核心技术包括：1）对事实的对抗性生成器，用原模型自我生成无敏感信息的反事实答案和推理轨迹；2）迭代偏好优化框架（CiPO），结合SimPO偏好损失、NLL正则和保留集约束；3）SFT预热阶段缓解分布失配。

**📊 数据集**

使用了两个基准：R-TOFU（合成问答与CoT数据）和真实世界的RETURN数据集（包含260个涉及公共人物的隐私信息），并在DeepSeek-R1-Distill-Llama-8B及Qwen3-8B等不同模型上验证。

**📈 对比分析**

与多种现有方法（GA、GD、NPO、ReasonedIDK、R2MU等）对比，CiPO在答案层面忘记效能（AFE）和推理轨迹层面忘记效能（CFE）均领先，且保持最高的模型实用度（MU）和整体推理能力，表现出最佳的忘记-实用度权衡。

**⚠️ 局限性**

局限性包括：目前仅针对QA式事实忘记场景；对非问答格式或更复杂数据结构的适配需要进一步研究；在极端对抗性提示下可能仍存在信息泄露，需持续审计。

---

## 258. Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language

**arXiv ID:** 2604.15841 | [PDF](https://arxiv.org/pdf/2604.15841v1)

**作者:** Dianqing Lin `[一作]` (Inner Mongolia University), Guanglai Gao `[通讯]` (Inner Mongolia University)

**通讯引用:** 1462 | [OpenAlex ID](https://openalex.org/A5076174513)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了名为Mouse的基准，针对中文网络亚文化语言“臭香语”设计了六项评测任务，以检验大型语言模型在该语言上的理解与生成能力。

**💡 创新点**

创新点在于首次将臭香语的表述成分（同音、视觉、语义）与语用意图进行细粒度化分类，并通过LLM‑as‑Judge人工智能评审系统与人类标注者对齐，形成可公开复现的评测体系。

**🔧 技术方法**

采用了零样本评估、基于规则的LLM‑as‑Judge评分、BERT/Transformer等二分类模型、以及多模型对比与Matthews相关系数等统计指标来评测模型性能。

**📊 数据集**

使用了公开收集与人工构造的1,099条臭香语–标准汉语对齐样本（CXEI），并在此基础上扩展至多语言同义、意图与毒性标签。

**📈 对比分析**

通过与多种开源与闭源SOTA LLM（如Qwen3‑Max、GPT‑5.2、DeepSeek‑V3.2等）进行零样本对比，结果显示在翻译、意图识别等任务上准确率低于30%，但在毒性检测与含义选择任务上可达70%及以上。

**⚠️ 局限性**

局限性包括：样本覆盖仍不完全（无法涵盖所有新兴的臭香语变体），评测侧重零样本性能而非细粒度微调效果，且LLM‑as‑Judge与人类标注者的对齐虽高但仍受文化差异与主观偏差影响。

---

## 259. CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution

**arXiv ID:** 2604.15840 | [PDF](https://arxiv.org/pdf/2604.15840v1)

**作者:** Shidong Yang `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5509 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种闭环的强化学习框架CoEvolve，使LLM代理和训练数据通过交互反馈实现相互进化。

**💡 创新点**

创新点在于：①利用忘记、边界、稀有等反馈信号自动识别代理弱点；②用LLM在此基础上生成针对性的新任务；③将验证后的任务动态加入训练集，形成无人工监督的自适应数据分布。

**🔧 技术方法**

技术手段包括GRPO策略优化、LLM基于轨迹的上下文构建与探索、任务抽象与环境验证，以及多阶段闭环迭代。

**📊 数据集**

实验数据集为AppWorld和BFCL‑V3两个交互式基准。

**📈 对比分析**

与基线（零样本、静态合成、随机探索）相比，CoEvolve在三种模型（Qwen2.5‑7B、Qwen3‑4B、Qwen3‑30B‑A3B）上平均提升约15–20%，显著超过关闭源LLM和其他开源模型。

**⚠️ 局限性**

局限性：仅使用有限种类的反馈信号，早期训练阶段信号可能噪声；缺乏安全/可控性保障，合成任务若未经人工审查可能导致不安全或对抗性行为。

---

## 260. UsefulBench: Towards Decision-Useful Information as a Target for Information Retrieval

**arXiv ID:** 2604.15827 | [PDF](https://arxiv.org/pdf/2604.15827v1)

**作者:** Tobias Schimanski `[一作]` (University of Zurich), Markus Leippold `[通讯]` (University of Zurich)

**通讯引用:** 5270 | [OpenAlex ID](https://openalex.org/A5073309846)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可持续发展报告域的文档相关性与可用性标注数据集（UsefulBench），并评估了传统检索与LLM方法在区分相关性与可用性方面的表现。

**💡 创新点**

首次将相关性与可用性在同一数据集上细粒度区分，并探讨LLM与检索模型在此区分中的优势与局限。

**🔧 技术方法**

使用了大型语言模型（GPT‑4.1系列）、BM25、BGE‑M3等检索模型，以及提示工程、少样本提示与微调等技术进行分类与检索评估。

**📊 数据集**

利用1,110条人工标注的查询‑文档对（UsefulBench‑gold）及覆盖全报文档的53K条三元组（UsefulBench‑full）进行实验。

**📈 对比分析**

通过宏F1、二分类F1、ECE、Brier、nDCG@k等指标对分类与检索性能进行评估，传统检索偏向相关性，LLM在识别可用性上有提升但受专家知识限制，整体性能仍低于预期。

**⚠️ 局限性**

仅聚焦可持续性报告单一领域，样本量有限且存在标注偏差，且LLM对专家知识的掌握仍不足。

---

## 261. Continual Hand-Eye Calibration for Open-world Robotic Manipulation

**arXiv ID:** 2604.15814 | [PDF](https://arxiv.org/pdf/2604.15814v1)

**作者:** Fazeng Li `[一作]` (South China University of Technology), Yang Cong `[通讯]` (South China University of Technology)

**通讯引用:** 6174 | [OpenAlex ID](https://openalex.org/A5006477225)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向开放世界机器人操作的持续手眼标定框架，能够在连续学习多场景的过程中保持对已学场景的精确定位。

**💡 创新点**

创新点在于引入空间感知重放策略(SARS)通过Poisson disk采样构建几何均匀的重放缓冲区，以及结构保持双重蒸馏(SPDD)将定位知识拆解为粗层级布局与细粒度位姿，并分别蒸馏。

**🔧 技术方法**

采用了可视化定位的SCR(GLACE)作为基础网络，配合Poisson disk采样、KL约束、L1位置残差蒸馏、软化温度等技术。

**📊 数据集**

在i7Scenes、i12Scenes两个基准和自建的Isaac Sim机器人手眼标定数据集上进行实验。

**📈 对比分析**

与联合训练、Fine‑tune、iCaRL、Buff‑CS、GEC、GDR等基线对比，最终精度分别在i7Scenes 74.9%、i12Scenes 91.8%、Sim 98.4%，并将任务遗忘率降至6.65%、8.1%及1.6%，明显优于其它方法。

**⚠️ 局限性**

目前的局限包括仅在室内和仿真环境中验证，缺乏真实机器人部署和更大规模、结构更不规则环境的评估。

---

## 262. Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow

**arXiv ID:** 2604.15809 | [PDF](https://arxiv.org/pdf/2604.15809v1)

**作者:** Chengxin Liu `[一作]` (Korea Advanced Institute Of Science And Technology), Tae-Hyun Oh `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 2778 | [OpenAlex ID](https://openalex.org/A5078114111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在推理阶段通过token动态熵值判断视觉token重要性并调制因果掩码的自适应信息流调制方法（AIF），在不重新训练模型的前提下显著提升VLM的感知与推理能力。

**💡 创新点**

首次将信息流调制作为提升VLM感知的手段，并利用token动态熵度量判定视觉token重要性；通过一次额外解码步骤生成掩码，实现对文本-视觉交互的精准控制。

**🔧 技术方法**

token动态分析、熵度量、因果掩码调制、一次额外的语言模型解码步骤；在LLaVA-1.5、Qwen2.5-VL等公开VLM上实现。

**📊 数据集**

VQA（V*、RealWorldQA、MMStar）、OCR（TextVQA、SeedBench2-Plus）、视觉定位与计数（RefCOCO/RefCOCO+/RefCOCOg、CountBench）、误幻现（COCO子集）、POPE等多种公开数据集。

**📈 对比分析**

与基线VLM、ViCrop、CCA、Future‑aware因果掩码等方法对比；在大多数任务中提升5–10%（如LLaVA‑1.5从42.4→50.3，Qwen2.5‑VL从70.4→84.8），在视觉定位、计数和误幻现任务上亦获得显著改进。

**⚠️ 局限性**

对长文本提示的鲁棒性有限；方法需要一次额外解码步骤，计算成本略升；不同模型的适配性和极端场景下的效果仍需进一步验证。

---

## 263. Beyond a Single Frame: Multi-Frame Spatially Grounded Reasoning Across Volumetric MRI

**arXiv ID:** 2604.15808 | [PDF](https://arxiv.org/pdf/2604.15808v1)

**作者:** Lama Moukheiber `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7898 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了SGMRI-VQA基准，包含41,307个多帧MRI视觉问答对，要求模型在3D体素序列中进行链式推理和跨切片定位；通过对齐放射科专家标注和GPT-4o生成的多层次任务，评估模型在检测、定位、分类、计数和描述等维度的能力。

**💡 创新点**

①首次引入跨切片视觉定位和链式推理的医学VQA基准；②采用层次化任务结构，模拟放射科医生对体积图像的诊断流程；③通过对抗式GPT-4o生成的 QA 与专家人工校正相结合，提供高质量的定位标注与推理文本；④证明了在此基准上针对性细调可显著提升空间定位性能。

**🔧 技术方法**

使用多模态大型语言模型（如Qwen3-VL-8B），对其进行全参数微调；利用GPT-4o作为生成器与评判者；采用多项评价指标（A-Score、AR-Score、V-Score）评估答案准确性、推理质量和像素级定位；使用DeepSpeed ZeRO3加速训练。

**📊 数据集**

基于fastMRI+数据集的脑部FLAIR/T1和膝关节PD-FAT-SAT序列，共1,970体积、50,672张切片；每个体积配有专家标注的框坐标与类别。

**📈 对比分析**

在10个VLM（包括GPT-4o、Gemini系列、Qwen系列、InternVL、LLaVA-Video等）上进行零样本和微调实验。微调后Qwen3-VL-8B在图像层面平均得分59.45%，在体积层面平均得分49.23%，均超越所有对照模型，尤其在定位任务（V-Score）提升明显。

**⚠️ 局限性**

仅覆盖脑部和膝部两类解剖结构；基准数据规模仍有限，需更多多模态、多领域数据；未评估更大参数模型（70B+）；仍存在少量标注噪声，尤其在推理文本中。

---

## 264. Qwen3.5-Omni Technical Report

**arXiv ID:** 2604.15804 | [PDF](https://arxiv.org/pdf/2604.15804v1)

**作者:** Qwen Team `[一作]` `[通讯]`, Qwen Team

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了Qwen3.5‑Omni，一款统一处理文本、图像、音频和音频‑视觉内容的全模态大语言模型，支持256k上下文长度和实时流式生成。

**💡 创新点**

创新点包括：Hybrid Attention Mixture‑of‑Experts（MoE）框架用于Thinker与Talker实现高效长序列推理；ARIA（Adaptive Rate Interleave Alignment）动态对齐文本与语音单元，提升流式语音合成的稳定性和自然度；多码本音频码流预测与即时解码；10小时音频与400秒720P视频（1FPS）支持；零样本语音定制与可控音视频字幕；出现新的“Audio‑Visual Vibe Coding”能力，直接从音频‑视觉指令生成可执行代码。

**🔧 技术方法**

技术实现基于Thinker–Talker架构，结合Hybrid MoE、AuT音频Transformer、Vision Encoder、RVQ多码本语音表示、MTP模块、GDN加速长序列、ARIA对齐、TMRoPE+时间戳编码、RLHF、DPO、GSPO、专家蒸馏等多种方法。

**📊 数据集**

使用的数据集包括：100+百万小时的音频‑视觉数据、40M小时的音频数据（AuT预训练）、4万亿token的跨模态数据（文本、音频、图像、视频、视频‑音频），以及超过200种语言与方言的文本、语音、图像、视频‑文本配对。训练分为Encoder Alignment、General和Long Context三个阶段。

**📈 对比分析**

评估方法涵盖215个音频/音频‑视觉理解、推理与交互子任务，以及多项文本、视觉与音视频评测基准。与Gemini‑3.1 Pro比较，Qwen3.5‑Omni在一般音频理解、推理、识别、翻译与对话上均优于其，音视频理解与生成与Gemini相当；在文本与视觉任务上保持与同规模Qwen3.5‑Plus一致。零样本TTS WER平均0.99‑1.26，支持29种语言的语音生成，在22/29语言上WERS最低；跨语言语音克隆在10/12方向上达到最优；自定义语音生成在10种语言上WERS最佳。实时推理延迟在Plus和Flash版本下分别为235/426 ms（音频）和435/651 ms（视频），保持低RTF并支持高并发。

**⚠️ 局限性**

局限性：模型规模庞大，推理成本高；在极长文本/多模态输入下仍可能出现对齐误差；对部分低资源方言与极端噪声环境的鲁棒性尚待提升；代码生成（Audio‑Visual Vibe Coding）虽然新颖，但尚未在复杂真实世界任务中充分验证；整体模型仍需进一步压缩与加速以适配边缘或移动端部署。

---

## 265. EVIL: Evolving Interpretable Algorithms for Zero-Shot Inference on Event Sequences and Time Series with LLMs

**arXiv ID:** 2604.15787 | [PDF](https://arxiv.org/pdf/2604.15787v1)

**作者:** David Berghaus `[一作]` `[通讯]` (Lamarr Institute), David Berghaus (Lamarr Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过LLM引导的程序进化，寻找并提出三类动力学系统推理任务（点过程预测、马尔可夫跳跃过程参数估计和时间序列缺失填补）可解释、无训练、零样本可用的Python/NumPy算法。

**💡 创新点**

创新点在于首次证明LLM驱动的进化搜索能够一次性发现跨数据集泛化的“放大式”推理函数，并在多个基准上与深度学习模型竞争甚至优于它们，同时保持完整可解释性和极低的计算成本。

**🔧 技术方法**

技术方法包括使用AlphaEvolve/OpenEvolve框架、GPT‑5/5‑mini多模态LLM的代码差分提议、MAP‑Elites岛屿种群进化、NumPy限制和自定义评估函数，针对三种动力学任务分别进行程序演化。

**📊 数据集**

实验使用公开点过程数据（Taxi、Taobao、Retweet 等）、合成马尔可夫跳跃过程（2–6 状态）以及多元时间序列数据（Beijing、Italy、Guangzhou、PeMS、Pedestrian、Solar、ETT_h1、Electricity）等多种数据集。

**📈 对比分析**

与 A‑NHP、NHP、IFTPP、Dual‑TPP、CDiff、HawkesEM、FIM‑PP、NeuralMJP、FIM‑MJP、BRITS、SAITS、CSDI、FIM‑ℓ 等多种基线对比，演化算法在多数基准上与最先进模型相当或更优，且推理速度快数百倍、API 成本低且完全可解释。

**⚠️ 局限性**

局限性包括在需要强学习表征的窗口缺失填补任务上表现不佳；算法为确定性，缺乏不确定性估计；难以评估发现算法的新颖性与可扩展性。

---

## 266. Filter Babel: The Challenge of Synthetic Media to Authenticity and Common Ground in AI-Mediated Communication

**arXiv ID:** 2604.15786 | [PDF](https://arxiv.org/pdf/2604.15786v1)

**作者:** Advait Sarkar `[一作]` (Microsoft Research), Advait Sarkar `[通讯]` (Microsoft Research)

**通讯引用:** 1631 | [OpenAlex ID](https://openalex.org/A5025427198)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了“Filter Babel”概念，探讨在生成式人工智能极度个性化的未来中，私有媒体如何导致私有语言和沟通失效的潜在后果，并就此对话与身份构建的影响展开思考。

**💡 创新点**

创新点在于将私人媒体生成与哲学中的私有语言概念相结合，提出“Filter Babel”作为一个新的技术与社会共生模型，并提出了多方向的研究议题，如共享种子生成、可见翻译失真与协同素养教育。

**🔧 技术方法**

主要技术是生成式人工智能（大语言模型、内容生成与多模态翻译）及其在个人化媒体生产和人工智能中介沟通中的应用。

**📊 数据集**

文中未使用具体数据集，主要基于理论分析、案例思考与文献综述。

**📈 对比分析**

由于是概念性探讨，未给出实验比较或性能指标；作者建议未来可设计实验评估经验差异对推理和理解的一致性影响。

**⚠️ 局限性**

局限性包括缺乏实证验证、对技术实现细节与伦理影响的具体评估不足，以及对个人化与共享平衡的可操作方案仍处于理论阶段。

---

## 267. Similarity-Based Bike Station Expansion via Hybrid Denoising Autoencoders

**arXiv ID:** 2604.15783 | [PDF](https://arxiv.org/pdf/2604.15783v1)

**作者:** Oluwaleke Yusuf `[一作]` (Norwegian University of Science and Technology), Adil Rasheed `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4854 | [OpenAlex ID](https://openalex.org/A5032407979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于混合去噪自编码器的相似性驱动自行车共享站点扩张框架，能够在缺乏需求预测的情况下，从已有优质站点中提炼城市特征，指导新站点选址。

**💡 创新点**

创新点在于将无监督特征压缩与有监督分类正则化相结合的自编码器，产生结构化的低维嵌入空间，并通过多参数共识机制消除对参数选择的敏感性，实现无需求预测的可靠扩张决策。

**🔧 技术方法**

采用混合去噪自编码器（HDAE）进行特征学习，使用余弦/欧氏距离、top‑k 与 KDE 聚合相似度，贪婪最大权独立集与局部搜索进行选址，最终用 DBSCAN 对候选集进行聚类共识。

**📊 数据集**

利用挪威Trondheim市网格化的29维城市特征数据集，包括社会人口、建筑环境、交通网络、流量等四类主题特征。

**📈 对比分析**

与原始特征的聚类（轮廓系数从0.135提升到0.253）和选址（仅有11.8%重叠）对比，HDAE嵌入在空间聚类一致性、候选选取稳定性上均显著优于原始特征；敏感性分析显示不同相似度聚合方法、距离度量和top‑k参数下结果保持稳健。

**⚠️ 局限性**

局限性包括：仅基于特征相似性假设，可能错过非相似但潜在成功的位置；空间约束采用欧氏距离，未考虑网络或土地利用限制；需要通过实地运营数据进一步验证模型预测的运营表现。

---

## 268. Pruning Unsafe Tickets: A Resource-Efficient Framework for Safer and More Robust LLMs

**arXiv ID:** 2604.15780 | [PDF](https://arxiv.org/pdf/2604.15780v1)

**作者:** Wai Man Si `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种资源高效的后置剪枝框架，直接识别并移除 LLM 中导致不安全输出的参数，从而提升模型安全性与鲁棒性。

**💡 创新点**

创新点在于将安全对齐视作“安全票据”剪枝，利用无梯度的 Wanda‑style 归因方法与 Lottery Ticket Hypothesis 视角实现对不安全子网络的精确定位与剔除。

**🔧 技术方法**

使用无梯度权重归因（改进的 Wanda）、贪婪/束搜索迭代剪枝、对比评估指标（如 Unsafe Rate、Over‑Refusal、Utility、ASR）等技术。

**📊 数据集**

采用 HarmBench、JailbreakBench、MM‑SafetyBench、MT‑Bench 等安全与鲁棒性评测数据集进行实验。

**📈 对比分析**

与 DPO、CB、Goal 等传统对齐方法比较，剪枝模型将不安全率从 22% 降至 <2%，保持 Utility 接近 7，同时显著降低攻击成功率，且无额外推理时间或显存开销。

**⚠️ 局限性**

局限性在于束搜索阶段计算成本高，且归因基于静态 token，可能无法捕获更广义的语义风险。

---

## 269. Fuzzy Logic Theory-based Adaptive Reward Shaping for Robust Reinforcement Learning (FARS)

**arXiv ID:** 2604.15772 | [PDF](https://arxiv.org/pdf/2604.15772v1)

**作者:** Hürkan Şahin `[一作]` (Paderborn University), Erdal Kayacan `[通讯]` (Paderborn University)

**通讯引用:** 6185 | [OpenAlex ID](https://openalex.org/A5068099488)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种基于模糊逻辑的自适应奖励塑形方法（FARS），用于提升无人机竞速任务中的强化学习稳定性与收敛速度。

**💡 创新点**

创新点在于将人类直觉以模糊语言规则编码为速度-距离奖励，能够根据无人机与门的相对位置动态调节奖励，既保持高速度也保证精准控制，且奖励函数可解释且对超参数不敏感。

**🔧 技术方法**

采用了Mamdani与Sugeno两种模糊推理系统、Proximal Policy Optimization（PPO）强化学习框架以及IsaacSim仿真环境。

**📊 数据集**

使用了在IsaacLab中生成的三种难度（Easy、Medium、Hard）的斜线式门道竞速模拟环境，包含多随机种子（5、8、16、32、36）和5000次评估测试。

**📈 对比分析**

与传统的基于势场的奖励塑形（PFBRS）对比，FARS在Medium、Hard场景下收敛更快、波动更小，门通过成功率提升约5%，并在Hard场景中实现最高总奖励和最低种子方差。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证；模糊规则与隶属函数固定，未实现在线学习；在实际硬件部署和跨域（sim‑to‑real）迁移上的鲁棒性尚待验证。

---

## 270. Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing

**arXiv ID:** 2604.15771 | [PDF](https://arxiv.org/pdf/2604.15771v1)

**作者:** Kai Wei `[一作]` (University of Michigan), Fan Yang `[通讯]` (Wake Forest University)

**通讯引用:** 470307 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为Skill‑RAG的检索增强生成框架，该框架通过隐藏状态探测器判断检索是否成功，并在失败时利用基于提示的技能路由器诊断原因，选择四种针对性检索技能（查询重写、问题分解、证据聚焦、退出）以纠正查询‑证据对齐缺失后继续生成。

**💡 创新点**

将检索失败拆解为结构化的“失败状态”，并在隐藏状态空间中探测这些状态；随后通过对症下药的技能路由器为每种失败状态提供专属的纠正策略，突破了传统仅以重检索或一次性检索为手段的限制，实现了对失败类型的精准定位与修复。

**🔧 技术方法**

利用LLM后隐藏层的向量做二分类探测器、基于提示的LLM诊断与技能选择、四种检索技能（重写、分解、聚焦、退出）、迭代检索‑生成循环，以及t‑SNE可视化分析失败状态空间。

**📊 数据集**

在HotpotQA、NQ、TriviaQA三大开放域问答数据集进行内部训练与评估，同时使用MuSiQue与2WikiMultiHopQA做OOD测试；所有实验均采用BM25检索器并基于Gemma2‑9B模型。

**📈 对比分析**

与No Retrieval、Single‑step RAG、FLARE、DRAGIN、Adaptive‑RAG及Probing‑RAG等六种基线在EM/ACC指标上对比，Skill‑RAG在内部数据集与Probing‑RAG相当或更优，在OOD数据集上实现显著提升（ACC提高6.1至13.6个百分点），整体表现达到或逼近最优水平。

**⚠️ 局限性**

依赖LLM的提示式技能路由，受指令遵循能力限制；四种技能词汇基于固定开放域QA任务，可能不适用于其他领域（如科学文献或多语种）；实验仅验证于Gemma2‑9B单一模型，缺乏跨规模与跨架构的泛化评估。

---

## 271. Disentangling Mathematical Reasoning in LLMs: A Methodological Investigation of Internal Mechanisms

**arXiv ID:** 2604.15842 | [PDF](https://arxiv.org/pdf/2604.15842v1)

**作者:** Tanja Baeumel `[一作]` (German Research Center for Artificial Intelligence), Simon Ostermann `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文通过对大型语言模型（GPT‑NeoX‑20B与GPT‑2 XL）的内部机制进行解释性分析，探究其在执行算术运算时如何逐层构建下一词预测，并揭示了任务识别、信息传播以及结果生成的阶段性过程。

**💡 创新点**

创新点在于将早期解码（logit lens）应用于算术任务，系统性追踪注意力与MLP模块在各层的贡献，发现高性能模型表现出注意力传播数值信息、MLP聚合计算结果的明确功能分工，并揭示只有一侧算术操作数被显式传递，暗示模型以函数式方式完成算术。

**🔧 技术方法**

使用的技术包括Transformer的早期解码/logit lens、对残差流的词表映射、交叉干预（interchange intervention）评估注意力模块对输入信息的影响，以及对绝对误差、数值占比等中间预测指标的可视化分析。

**📊 数据集**

采用人工生成的算术数据集：add_small、add_large、sub_small、sub_large（各500条），以及针对3个操作数的实验数据集；所有数据仅包含单词级数值，方便直接映射到词表。

**📈 对比分析**

通过零样本评估比较两模型的算术准确率（GPT‑NeoX‑20B高准确率，GPT‑2 XL低准确率），并利用中间预测指标（数值占比、绝对误差、结果出现层数）对两模型内部机制进行对比，验证了高性能模型的注意力‑MLP分工对算术性能的决定性作用。

**⚠️ 局限性**

局限性包括只测试了两款模型，可能不具备对其他大型模型的普适性；数据集过于简化，未涵盖自然语言算术问题；仅关注算术任务，未验证在更复杂推理场景下的机制是否相同。

---

## 272. From Intention to Text: AI-Supported Goal Setting in Academic Writing

**arXiv ID:** 2604.15800 | [PDF](https://arxiv.org/pdf/2604.15800v1)

**作者:** Yueling Fan `[一作]` (KTH Royal Institute of Technology), Olga Viberg `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 3313 | [OpenAlex ID](https://openalex.org/A5036322331)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了 WriteFlow，一种基于语音的 AI 写作助手，通过目标导向对话来支持学术写作中的元认知与反思。

**💡 创新点**

将 AI 对话空间重新定义为目标协商与持续监控的平台，在写作过程中实现目标‑文本对齐追踪，并通过可视化和迭代目标细化支持反思‑行动。

**🔧 技术方法**

结合 GPT‑4o 大语言模型、Google Docs 插件、语音输入、目标跟踪与评估卡片、以及可视化大纲视图。

**📊 数据集**

通过17名学生的调查和12名 HCI 专家参与的 Wizard‑of‑Oz 评测；使用的文本来自用户提供的 TikTok 交互评估任务和文献笔记。

**📈 对比分析**

采用反思主题分析对参与者交互与访谈进行定性评估；未进行数值性能比较，结果显示 WriteFlow 在促进目标可视化、元认知和作者代理方面获得正面反馈。

**⚠️ 局限性**

样本仅限 HCI 专家，任务与材料由研究者预设，Wizard‑of‑Oz 方式限制真实性，提示设计固定且缺乏自适应，且未对 AI 输出的评估精度进行量化。

---

## 273. A Systematic Study of Training-Free Methods for Trustworthy Large Language Models

**arXiv ID:** 2604.15789 | [PDF](https://arxiv.org/pdf/2604.15789v1)

**作者:** Wai Man Si `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型进行无梯度训练方法的系统评估，探究其在安全性、真实性、偏见、鲁棒性及效能上的影响与成本；

**💡 创新点**

首次提出按信息流分为输入、内部、输出三级的统一分类框架，并在多规模模型上重新评估其效果与计算代价；

**🔧 技术方法**

采用提示工程、激活/参数编辑和对比/引导解码等多种训练自由技术；

**📊 数据集**

使用多样化基准数据集，包括 HarmBench、TruthfulQA、BBQ、MMLU、MT‑Bench 等；

**📈 对比分析**

与基线比较发现输入级方法提升安全但降低真实性，内部级方法改善真实性但牺牲效能，输出级方法保持效能相对保守；组合方法难以同时提升所有维度；

**⚠️ 局限性**

局限性包括方法在黑盒环境下可行性受限、超参数迁移困难、对大型模型的评估不足以及可能产生新的失效模式。

---

## 274. Fusing Cellular Network Data and Tollbooth Counts for Urban Traffic Flow Estimation

**arXiv ID:** 2604.15782 | [PDF](https://arxiv.org/pdf/2604.15782v1)

**作者:** Oluwaleke Yusuf `[一作]` (Norwegian University of Science and Technology), Shaira Tabassum `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5026336188)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现一种基于机器学习的数据融合框架，利用稀疏且精准的收费站车辆计数校正大规模蜂窝网络人流数据，并生成按车辆长度分类的时空OD矩阵，用于交通仿真与公交枢纽扩建影响评估。

**💡 创新点**

创新点在于：①以收费站计数作为地面真相，用XGBoost学习并校正蜂窝人流的系统性偏差；②将校正后的流量与手工构造的路由逻辑相结合，生成可用于仿真的详细OD矩阵；③提出宏观流量估计可在数据稀缺环境中生成，并可通过局部摄像头数据进一步细化。

**🔧 技术方法**

使用技术包括：XGBoost回归模型、SHAP特征重要性分析、基于时空特征的特征工程、人工路由逻辑（内部流、局部进出、通行流），概率分布分配、离散化与大余数分配方法，以及Aimsun交通仿真软件。

**📊 数据集**

使用数据集有：NPRA收费站按车辆长度分类的计数（训练集和验证集）、Telia Crowd Insights “peopleFlow”路由报告（覆盖全市道路网络）、Trondheim公交时刻表与补充摄像头计数（用于验证与细化）。

**📈 对比分析**

通过将模型预测与真实收费站计数进行对比评估；人流原始数据的R²仅为0.44，XGBoost模型校正后R²达到0.98（各车辆类别均>0.97），RMSE显著下降，残差保持在零附近，表明模型在校正和去聚合方面表现优异。

**⚠️ 局限性**

限制因素包括：训练集与仿真集时间不完全对齐、手工路由逻辑简化为单向流、缺乏不确定性量化、未使用更深层的深度学习方法、对双向流动建模不够精确。

---

## 275. CroSatFL: Energy-Efficient Federated Learning with Cross-Aggregation for Satellite Edge Computing

**arXiv ID:** 2604.15779 | [PDF](https://arxiv.org/pdf/2604.15779v1)

**作者:** Nan Yang `[一作]` (Western Sydney University), Philip Leong `[通讯]` (University of Sydney)

**通讯引用:** 16903 | [OpenAlex ID](https://openalex.org/A5037482622)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全在轨层次联邦学习框架 CroSatFL，利用激光互星链（LISL）完成所有中间聚合，地面站仅用于模型初始化与最终收集，从而显著降低地面通信负担和能耗。

**💡 创新点**

创新点包括：①只在轨进行聚合，消除地面同步瓶颈；②基于强化学习与注意力机制的 StarMask，生成满足 LISL 约束且资源均衡的聚类；③Skip-One 客户端选择，在每轮允许跳过最多一个瞬时 straggler，降低同步延迟和能耗；④random‑k 跨聚类聚合，利用可达邻居实现全局一致而不增加轮级延迟；⑤整体考虑 LEO 动态连通性、CPU/GPU 异构和功耗约束。

**🔧 技术方法**

使用技术包括联邦学习、激光互星链通信、强化学习+注意力网络聚类、动态调度与能耗建模、随机抽样聚合、异构硬件资源管理与公平调度。

**📊 数据集**

实验数据集为 ResNet‑18 在 MNIST、CIFAR‑10、EuroSAT 上的 IID 与 non‑IID 训练，目标准确率分别为 95%、75% 与 80%。

**📈 对比分析**

与 FedSyn、FedLEO、FELLO、FedSCS、FedOrbit 等五种基线对比，评价指标为模型准确率、总能耗、训练时间和 GS 通信量。结果显示 CroSatFL 在保持或略优于基线准确率的同时，GS 通信次数降低 2 个数量级，GS 传输能耗约 6 倍减少，等待时间降至 7.89 小时，整体能耗与训练时间均最低。

**⚠️ 局限性**

局限性包括：实验仅在单主轮（G=1）场景下验证；random‑k 交互对收敛速度可能产生不确定性；未在更大规模或更复杂轨道动力学下进行验证；对跨层 LISL 可用性做了理想化假设；缺乏对数据安全、误差容忍等方面的深入探讨。

---

## 276. PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding

**arXiv ID:** 2604.15770 | [PDF](https://arxiv.org/pdf/2604.15770v1)

**作者:** Junjie Wen `[一作]` (Peng Cheng Laboratory), Jinqiang Cui `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5062631244)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PLAF 框架，利用像素级语言对齐特征和基于 mask 的索引存储实现高效、可扩展的开词汇 3D 场景理解。

**💡 创新点**

创新点在于：1）将视觉基础模型的稠密特征与无类别 mask 结合，得到像素级语言对齐描述；2）设计 mask‑索引 + 引用的存储方案，显著降低 2D/3D 语义冗余，实现可扩展的 3D 语义映射。

**🔧 技术方法**

技术手段包括：冻结的视觉基础模型（如 RADIOv2.5）、SAM 无类别 mask 提取、mask 聚合生成 region‑一致特征、index‑and‑reference 的 2D/3D 存储与融合、零样本文本查询匹配。

**📊 数据集**

使用数据集：ScanNet（3D 文本查询评估）和 ADE20K（线性探测语义分割验证）。

**📈 对比分析**

与 ConceptFusion、OpenMask3D、RayFronts 等基线在零样本设置下对比，PLAF 在 2D 文本查询、3D 文本查询以及 ADE20K 线性探测上表现更优：mIoU 41.1% 为最高，语义存储相较稠密 per‑pixel/per‑point 方法降低 99%+，显著提升查询质量和存储效率。

**⚠️ 局限性**

局限性：仍受基础模型分辨率与语义噪声影响，mask 提取精度决定定位准确度；在极大规模场景的实时推理与动态更新方面尚未完全验证。

---

## 277. cuNNQS-SCI: A Fully GPU-Accelerated Framework for High-Performance Configuration Interaction Selection withNeural Network QQantum States

**arXiv ID:** 2604.15768 | [PDF](https://arxiv.org/pdf/2604.15768v1)

**作者:** Daran Sun `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Guangming Tan `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了完全GPU加速的 cuNNQS‑SCI 框架，用于高精度配置相互作用选择，并与神经网络量子态结合

**💡 创新点**

创新点包括分布式基于排序的全局去重算法、专用 CUDA 核生成耦合配置以及面向 GPU 的内存中心执行模型

**🔧 技术方法**

采用 CUDA、MPI、PyTorch、分布式排序、堆式 Top‑K、异步流式内存管理等技术

**📊 数据集**

在从 C₂、N₂、LiH、LiF、LiCl、Li₂O 等小分子到 84 量子位的 Cr₂ 等多种分子数据集上进行验证

**📈 对比分析**

与原始 CPU+GPU 混合实现的 NNQS‑SCI 进行对比，64 GPU 集群上实现了 1.88–2.32 倍的端到端加速，并保持化学精度；强缩放效率超过 90%

**⚠️ 局限性**

仍受 GPU 内存容量限制，超大规模配置空间需要进一步的内存优化；对极端稀疏数据的处理和跨节点通信延迟仍是潜在瓶颈

---

## 278. Breaking the Training Barrier of Billion-Parameter Universal Machine Learning Interatomic Potentials

**arXiv ID:** 2604.15821 | [PDF](https://arxiv.org/pdf/2604.15821v1)

**作者:** Yuanchang Zhou `[一作]` (State Key Lab of Processors Institute of Computing Technology Chinese Academy of Sciences), Weile Jia `[通讯]` (State Key Lab of Processors Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构建基于Mixture‑of‑Experts的MatRIS‑MoE模型，并开发Janus分布式训练框架，实现了对10‑50 B参数通用机器学习原子势（uMLIP）的高效训练；

**💡 创新点**

创新点包括：① 将Mixture‑of‑Experts应用于统一的原子势模型，实现多任务学习与参数规模扩展；② 设计双向后向（second‑order）训练的混合并行（FS‑3D）框架，实现GPU/ARM多核加速；③ 针对多机通信与内存优化的硬件感知技术（如原子类型压缩、SDMA数据搬移、异步梯度同步）；

**🔧 技术方法**

主要技术手段包括：Invariant GNN（MatRIS）→多头自注意力、MoE路由、FS‑3D并行（FSDP+FSGP+FSEP）、FP16压缩、异步MPI、SDMA驱动、混合精度（FP32）双向自动微分；

**📊 数据集**

使用了包含473 M个原子配置、3.6 T边的跨域数据集，涵盖孤立分子、周期性晶体、催化表面、MOF及晶体等多种材料；

**📈 对比分析**

与现有模型（如UMA、MACE‑mh、eqV2等）相比，MatRIS‑MoE在CNIS与LineShine两台Exascale系统上实现了峰值1.2 EFLOPS、90%+并行效率，训练时间从数周压缩至数小时，归一化吞吐量提升至653–3201×；

**⚠️ 局限性**

局限性包括：仍依赖大型Exascale硬件，第二阶自动微分对显存与通信压力大；在更大模型（>10 B）及更细粒度异构任务时需进一步优化内存管理与通信；模型训练对全精度（FP32）有严格要求，难以利用低精度加速；

---

## 279. ECG-Lens: Benchmarking ML & DL Models on PTB-XL Dataset

**arXiv ID:** 2604.15822 | [PDF](https://arxiv.org/pdf/2604.15822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 280. Stein Variational Black-Box Combinatorial Optimization

**arXiv ID:** 2604.15837 | [PDF](https://arxiv.org/pdf/2604.15837v1)

**作者:** Thomas Landais `[一作]` (Universite d'Angers), Sylvain Lamprier `[通讯]` (Universite d'Angers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Stein变分梯度下降（SVGD）的多智能体估计分布算法（SVGD‑EDA），用于高维离散黑盒优化。

**💡 创新点**

创新点：① 将SVGD的拉力-排斥机制直接作用于EDA的参数粒子，避免传统单一分布收敛；② 引入秩变换的目标函数实现对目标函数单调不变性；③ 通过多智能体协作与核排斥保持多峰搜索，显著提升多峰问题的搜索效果。

**🔧 技术方法**

使用技术：Stein变分梯度下降、RBF核排斥、秩变换权重（w(x)=1‑2x）、Bernoulli/类别分布的概率模型、PyTorch GPU加速。

**📊 数据集**

数据集：NK（二进制）和NK3（三值）Kauffman模型的随机生成实例，规模分别为 n∈{64,128,256}，K∈{1,2,4,8}，共计 80 种配置，10 个实例/配置。

**📈 对比分析**

与 84 种基线算法（包括经典 EDAs、进化策略、差分进化、Nevergrad 库的 80 算法）在 50,000 次评估预算下进行对比。SVGD‑EDA 在 17/24 配置中排名第一，平均排名 5.375，表现优于传统 EDA 与大部分进化算法，尤其在高维/高崎岖度问题上优势显著。

**⚠️ 局限性**

局限性：① 核函数为欧氏 RBF，可能不适用于参数空间的统计学结构；② 需要合理设置智能体数量 m，过多会导致评估预算稀释；③ 目前仅支持无关变量的单变量 Bernoulli/类别模型，无法捕捉变量间依赖；④ 对温度 γ 的敏感性仍需进一步研究。

---

## 281. Modern Structure-Aware Simplicial Spatiotemporal Neural Network

**arXiv ID:** 2604.15833 | [PDF](https://arxiv.org/pdf/2604.15833v1)

**作者:** Zhaobo Hu `[一作]`, Mehdi Naima `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了 ModernSASST，结合高维单纯形复合体的随机游走与并行化的时间卷积网络，对时空数据进行高阶拓扑特征提取与预测/插值建模。

**💡 创新点**

创新点在于：①首次将单纯形复合体结构引入时空建模；②用随机游走取代传统 GNN 消息传递获取高阶拓扑信息；③用并行化 Temporal Convolutional Networks 代替 RNN，显著提升计算效率。

**🔧 技术方法**

所用技术包括：单纯形复合体与 Hodge Laplacian、随机游走（RUM）、深度卷积（DWConv、ConvFFN）、自适应邻接矩阵、图层归一化、Optuna 超参数优化等。

**📊 数据集**

实验数据集涵盖 SDWPF（风电功率）、METR‑LA（交通流量）和 AQI（空气质量）三大领域。

**📈 对比分析**

与 VAR、FC‑LSTM、ModernTCN、DCRNN、GraphWaveNet、SGP、BRITS、GRIN、SPIN 等基线在预测和插值任务上进行 5 次独立跑，评估指标为 MAE、RMSE、MRE 等；在 SDWPF 上取得最显著提升，整体表现优于多数基线。

**⚠️ 局限性**

局限性包括：对三角形过多导致拓扑失衡的数据集（AQI、METR‑LA）提升有限；对随机游走参数的稳定性仍需深入研究；缺乏对更高阶单纯形（>2阶）及动态图结构的支持。

---

## 282. Placing Puzzle Pieces Where They Matter: A Question Augmentation Framework for Reinforcement Learning

**arXiv ID:** 2604.15830 | [PDF](https://arxiv.org/pdf/2604.15830v1)

**作者:** Yangyi Fang `[一作]` (Tsinghua University), Haolin Shi `[通讯]` (Tsinghua University)

**通讯引用:** 902 | [OpenAlex ID](https://openalex.org/A5026315528)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的提示注入框架 PieceHint，针对大型语言模型的推理任务，按难度动态提供关键推理步骤作为提示，并在训练过程中逐步撤销提示，使模型实现从引导式到独立推理的迁移。

**💡 创新点**

核心创新在于：① 用价值驱动的分数识别出“瓶颈”推理步骤；② 依据问题难度分配初始提示数量；③ 采用频率驱动的渐进式提示撤销课程，确保模型在获得信息的同时保持推理多样性。

**🔧 技术方法**

技术手段包括：强化学习（GRPO算法）+ 价值评估（使用更强 LLM 给推理片段打分）+ 变异基问题筛选 + 按需提示分配 + 渐进式提示撤销课程。

**📊 数据集**

训练数据来源于 OpenR1‑Math‑220K 经过变异筛选得到的子集；评估使用六大数学推理基准：AIME24、AIME25、AMC23、MATH500、Minerva、Olympiad。

**📈 对比分析**

与 1.5B、4B、32B 规模基线模型及其他提示增强方法对比，PieceHint‑1.5B 在平均 Pass@k 上可与 32B 基线相当，且保持推理多样性；在大部分基准上均优于同规模基线和 4B 模型。

**⚠️ 局限性**

主要局限：① 提示撤销采用固定频率 N_check，缺乏实时适应性；② 价值评估与提示选择依赖外部更强 LLM 的离线预处理，增加工程成本；③ 对于极难或极易问题的自适应能力尚未完善。

---

## 283. Fed3D: Federated 3D Object Detection

**arXiv ID:** 2604.15795 | [PDF](https://arxiv.org/pdf/2604.15795v1)

**作者:** Suyan Dai `[一作]` (South China University of Technology), Peican Lin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Fed3D，一个首个联邦3D目标检测框架，实现多机器人分布式3D检测并保护隐私。

**💡 创新点**

创新点在于设计局部-全局类别感知损失以解决3D异构性，并引入联邦提示学习显著降低通信成本。

**🔧 技术方法**

采用PiMAE预训练的ViT编码器、前缀提示调优、局部-全局类别平衡系数以及联邦平均聚合等技术。

**📊 数据集**

实验基于ScanNet V2和SUN RGB‑D这两大室内3D检测数据集。

**📈 对比分析**

与本地训练、中心化训练以及FedAvg、FedProx、SCAFFOLD、MOON、FedDyn等五种联邦学习基线对比，Fed3D在mAP@0.25提升0.2–3.9%，并在通信量上仅为其他方法的一半。

**⚠️ 局限性**

局限性包括仅在室内数据集上验证，提示学习仅覆盖编码器层，未针对更复杂的异构传感器或大规模边缘部署的网络延迟进行评估。

---

## 284. Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting

**arXiv ID:** 2604.15794 | [PDF](https://arxiv.org/pdf/2604.15794v1)

**作者:** Chi Liu `[一作]` (PayPal AI), Srinivasan Manoharan `[通讯]` (PayPal AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于自蒸馏细调（SDFT）的通用性能恢复框架，能够在LLM遭受灾难性遗忘、量化或剪枝导致性能下降后，借助历史检查点进行自我恢复。

**💡 创新点**

创新点在于将自蒸馏视为高维隐空间流形对齐过程，并通过Centered Kernel Alignment（CKA）提供定量的几何对齐度量，理论上解释了自蒸馏恢复性能的机制；同时给出了针对不同退化场景（灾难性遗忘、压缩、小模型）的统一恢复策略。

**🔧 技术方法**

主要技术包括：SDFT细调、CKA流形对齐度量、中心化核矩阵与Hilbert–Schmidt独立性判别（HSIC）计算、对比实验中的量化（NF4）与结构化剪枝、以及小模型的两阶段离线蒸馏+SDFT。

**📊 数据集**

使用的评估数据集包括Tool-use任务（结构化工具调用）、Science任务（科学问答）、MMLU与Winogrande（5-shot）用于通用能力评估；压缩实验中使用Qwen2.5-3B-Instruct与Qwen2.5-7B-Instruct模型。

**📈 对比分析**

与传统SFT、量化/剪枝后的原始模型以及直接SDFT进行对比；结果显示SDFT在灾难性遗忘场景下将任务准确率从37.87%恢复至61.54%，在量化场景中实现+15%~+22%的任务专用提升，并在剪枝场景中恢复了约64%性能损失；同时保持或提升了通用能力，CKA对齐度与性能提升高度相关。

**⚠️ 局限性**

局限性包括：对教师模型质量高度依赖；小模型在I CL不足时需要额外的离线蒸馏步骤；CKA仅提供相关性指标，缺乏因果性证明；在剪枝导致容量永久削减的极端情况下，恢复效果仍受限。

---

## 285. SegMix:Shuffle-based Feedback Learning for Semantic Segmentation of Pathology Images

**arXiv ID:** 2604.15777 | [PDF](https://arxiv.org/pdf/2604.15777v1)

**作者:** Zhiling Yan `[一作]` (JD AI Research), Guanglei Zhang `[通讯]` (Beihang University)

**通讯引用:** 2634 | [OpenAlex ID](https://openalex.org/A5086792600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于图像块随机置换与反馈学习的弱监督病理图像语义分割方法，利用仅图像级标签生成高质量像素级伪标签。

**💡 创新点**

创新点在于：①使用自适应调度的块大小与置换比例实现从粗到细的多尺度学习；②引入反馈机制根据学习表现动态调整置换策略；③结合像素相关模块提升CAM的精细化。

**🔧 技术方法**

主要技术包括：patch分块与随机置换、反馈学习调度、Pixel Correlation Module、基于CAM的伪标注扩展，以及使用ResNet骨干的弱监督分割网络。

**📊 数据集**

使用三大病理图像数据集：ROSE（胰腺细胞）、WBC（白细胞子集）与MARS（胃癌组织）。

**📈 对比分析**

与CAM、SEAM、CPN等基线对比，实验显示在三组数据上均取得显著提升，ROSE上DSC提升至60.7%/IoU 50.2%，WBC上DSC 41.5%/IoU 29.4%，MARS上DSC 40.6%/IoU 31.3%。

**⚠️ 局限性**

局限性包括：方法仅验证于二维病理图像，未探索三维医学影像；对图像质量噪声的鲁棒性仍有待提升；以及缺乏对不同组织类型的泛化评估。

---

## 286. Federated Learning with Quantum Enhanced LSTM for Applications in High Energy Physics

**arXiv ID:** 2604.15775 | [PDF](https://arxiv.org/pdf/2604.15775v1)

**作者:** Abhishek Sawaika `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 107254 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在高能物理分类任务中，提出了一个联邦量子增强 LSTM（QLSTM）模型，能够在分布式环境下进行本地训练并共享模型参数。

**💡 创新点**

创新点在于将变分量子电路嵌入 LSTM 单元，并与联邦学习相结合，实现了仅用 20K 样本和不到 300 个可训练参数就能达到接近经典深度学习水平的性能。

**🔧 技术方法**

采用了角度编码的变分量子电路、PennyLane 的量子模拟器、经典 LSTM 与全连接网络、Adam 优化器以及联邦学习框架。

**📊 数据集**

使用了 LHCb 产生的 SUSY 数据集，共 5M 行特征，其中实验只选取了 20K 条样本并分别使用 18 个和 7 个特征进行训练。

**📈 对比分析**

通过与单机 LSTM、VQC 以及现有文献中模型的 AUC、准确率对比，QLSTM 在 0.88 左右的 AUC 与 82% 的准确率下，仅比经典基准低 1%，且相对传统模型减少了约 100 倍的数据量和参数量。

**⚠️ 局限性**

局限性包括仅在模拟器上实验、使用了 IID 数据划分、缺乏量子硬件噪声评估以及对非 IID 场景的适应性验证。

---

## 287. (Weighted) Adaptive Radius Near Neighbor Search: Evaluation for WiFi Fingerprint-based Positioning

**arXiv ID:** 2604.15940 | [PDF](https://arxiv.org/pdf/2604.15940v1)

**作者:** Khang Le `[一作]` (Tampere University), Philipp Müller `[通讯]` (Tampere University)

**通讯引用:** 4375 | [OpenAlex ID](https://openalex.org/A5054331809)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于固定半径最近邻（FRNN）的两种改进方法——自适应半径最近邻（ARNN）和加权自适应半径最近邻（WARNN），并将其应用于WiFi指纹室内定位回归任务。

**💡 创新点**

创新点在于：①将固定半径替换为自适应半径，充分利用每个训练样本周围的分布信息；②在ARNN基础上引入权重机制，使用逆距离加权并设计自适应衰减因子；③通过大规模实验验证加权自适应半径能够优于现有kNN及其13种变体。

**🔧 技术方法**

使用技术包括：FRNN、ARNN、WARNN、13种kNN变体；逆距离加权、不同距离度量（欧氏、曼哈顿、余弦、Clark等）；训练阶段计算自适应半径；测试阶段基于半径集合选择邻居并加权求平均；评估指标为均方根3D定位误差和覆盖率。

**📊 数据集**

实验使用了22个公开WiFi指纹室内定位数据集（如DSI、LIB、TUT、UJI、SOD等），涵盖多种室内环境与硬件配置。

**📈 对比分析**

比较方法：在每个数据集上计算每种方法的平均3D定位误差和覆盖率；结果显示WARNN（M_23）平均误差比最优kNN低约3%，覆盖率约95%；FRNN误差最高，ARNN误差与kNN相当；WARNN在多种距离度量与权重方案下保持稳定且表现最佳。

**⚠️ 局限性**

局限性：ARNN和WARNN在训练阶段需额外计算自适应半径，计算成本较高；实验仅针对回归定位任务；在缺失类别的场景下性能尚未充分验证；误差阈值τ_ε的选择对结果影响显著，需针对不同数据集进行调优。

---

## 288. Efficient Video Diffusion Models: Advancements and Challenges

**arXiv ID:** 2604.15911 | [PDF](https://arxiv.org/pdf/2604.15911v1)

**作者:** Shitong Shao `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 468 | [OpenAlex ID](https://openalex.org/A5100457290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对视频扩散模型的高效推理方法进行了系统综述，梳理了现有加速技术、评估指标与发展趋势；

**💡 创新点**

首次提出了统一的四类加速范式（step distillation、efficient attention、model compression、cache/trajectory optimization），并以功能目标（NFE压缩与每步开销降低）对方法进行细分；

**🔧 技术方法**

综述了包括步骤蒸馏（LCD、DMD、Adversarial Distillation）、稀疏/线性注意力（Static/Dynamic Sparse、FlashAttention、SLA、ReHyAt）、量化与剪枝（QAT/PTQ、VAE压缩、Token/Model Pruning）、缓存与轨迹优化（Feature Cache、KV Cache、Noise/State修改、Parallel Execution）等技术；

**📊 数据集**

作为综述并未使用单一数据集，而对领域内常用的数据集进行了说明，如 WebVid-10M、OpenVid-1M、UCF101、Mixkit、SkyTimelapse 等；

**📈 对比分析**

对比方法时采用硬件无关指标（NFE、FLOPs）、硬件相关指标（推理吞吐量、延迟、VRAM占用）以及质量评估（FVD、FID、LPIPS、VBench、DOVER、FasterVQA）进行综合评估，指出不同范式在不同部署场景下的性能优劣；

**⚠️ 局限性**

局限性包括：缺乏统一的基准与公开评测平台，复现性差；所综述的加速方法在多种场景下会产生相互冲突的误差累积；硬件与算法匹配仍不完善，导致实际速度提升低于理论值；综述未给出统一的性能/质量平衡公式，未来需要更多交叉验证与标准化评测。

---

## 289. Robust Fleet Sizing for Multi-UAV Inspection Missions under Synchronized Replacement Demand

**arXiv ID:** 2604.15890 | [PDF](https://arxiv.org/pdf/2604.15890v1)

**作者:** Vishal Ramesh `[一作]` (IIIT Hyderabad), Antony Thomas `[通讯]` (IIIT Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对有限时隙多无人机巡检任务，研究并提出了一条闭式的机队规模确定公式，解决同步换电导致的备机瞬时枯竭问题。

**💡 创新点**

创新点在于：①识别并定量化同步换电峰值的结构性失效模式；②基于最坏相位对齐推导出最保守的机队规模规则 k = m(⌈R⌉ + 1)；③不依赖概率假设或仿真即可直接给出足够的备机数。

**🔧 技术方法**

主要技术包括：最坏情景分析、递推推导、Monte Carlo 验证、Wilson 置信下界评估，以及对比分析（Naive、Duty-Cycle、Erlang‑B）。

**📊 数据集**

数据集为模拟生成的聚类式巡检点分布，配合不同风速变化（CV 0.00–0.30）进行 1000 次随机试验；未使用真实现场数据。

**📈 对比分析**

与 Naive、Duty‑Cycle 和 Erlang‑B 的比较表明：在最困难场景下，Erlang‑B 仅 69.9% 成功率，而提出规则可达 99.8%（Wilson 下限 99.3%），其它场景均满足 95% 可靠性阈值；相对 Erlang‑B 只需多 4 架无人机即可实现可靠性跃迁。

**⚠️ 局限性**

局限性包括：假设所有 UAV 的可用飞行时间相同、充电设施无限、航线固定且无动态重分配、机队均质；未证明最小化，仅给出充分条件；若充电排队或异构平台出现，需进一步调整。

---

## 290. QMutBench: A Dataset of Quantum Circuit Mutants

**arXiv ID:** 2604.15870 | [PDF](https://arxiv.org/pdf/2604.15870v1)

**作者:** Eñaut Mendiluze Usandizaga `[一作]` (Simula Research Laboratory and Oslo Metropolitan University), Shaukat Ali `[通讯]` (Simula Research Laboratory and Oslo Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了可在线筛选的量子电路变异体数据集 QMutBench，包含 375 条原始量子电路及 723,079 个变异体，并提供按算法、量子位数、存活率、门类型、位置等多维度过滤和下载功能。

**💡 创新点**

创新点在于将大规模量子变异体按存活率和变异特征进行归类，构建可筛选接口，解决了之前数据难以访问和选择的痛点。

**🔧 技术方法**

使用 Muskit 量子变异工具生成变异体，计算存活率作为评估指标，采用 QASM 格式存储，并开发前后端在线筛选与下载界面。

**📊 数据集**

基于 300+ 典型量子算法（如 Grover、Amplitude Estimation 等）的 375 条原始电路，生成 723,079 个变异体。

**📈 对比分析**

通过选择不同存活率区间和变异特征构建基准，用于评估测试技术的变异检测率；作者展示了变异体在不同特征下的存活率分布，但未给出具体性能数值。

**⚠️ 局限性**

限制在于数据规模大导致一次性使用困难，缺乏对噪声量子设备的适配，仅支持已有电路，未提供自动新增原始电路及变异体的功能，并未在真实量子硬件上评估表现。

---

## 291. Environment-Adaptive Solid-State LiDAR-Inertial Odometry

**arXiv ID:** 2604.15864 | [PDF](https://arxiv.org/pdf/2604.15864v1)

**作者:** Zhi Zhang `[一作]` (Chongqing University of Posts and Telecommunications), Changjun Gu `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5031593727)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种环境自适应固态 LiDAR‑IMU 里程计与地图构建框架，利用局部法向角约束和退化感知 voxel 更新来提升极端或感知退化环境下的定位与地图一致性。

**💡 创新点**

①引入局部法向角约束补充传统点对面约束，增强平面区域几何一致性；②基于 Hessian 条件数与角度残差的退化评分，量化环境退化；③退化引导的 voxel 更新策略，根据退化分数动态控制 voxel 创建、替换和置信度，以保证地图质量。

**🔧 技术方法**

基于 MAP 估计的固态 LiDAR‑IMU 里程计（IMU 预积分、点云 GICP、点对面残差、法向角残差），Hessian 近似与 SVD 计算条件数，退化评分融合，voxel 体素地图构建与自适应更新。

**📊 数据集**

Botanic Garden 真实户外机器人数据集（固态 LiDAR、IMU、摄像头同步）。

**📈 对比分析**

与 FAST‑LIO、iG‑LIO 等基线在 1018‑00~1018‑06 序列进行 APE（RMSE、最大误差、平均误差）评估。全方法在大多数序列 RMSE 最低，最大/平均误差显著下降，提升幅度最高可达 12.8%（或 32.5% 最大误差下降），表明定位精度与鲁棒性明显优于现有方法。

**⚠️ 局限性**

退化评估仍受 Hessian 近似和噪声影响；对动态环境的适应性未验证；阈值参数（如 τ_global、α、γ 等）需要经验调节，可能在不同场景表现不一致。

---

## 292. Robust Multispectral Semantic Segmentation under Missing or Full Modalities via Structured Latent Projection

**arXiv ID:** 2604.15856 | [PDF](https://arxiv.org/pdf/2604.15856v1)

**作者:** Irem Ulku `[一作]` (Ankara University), Ömer Özgür Tanrıöver `[通讯]` (Ankara University)

**通讯引用:** 409 | [OpenAlex ID](https://openalex.org/A5034693671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于结构化潜在投影的多模态语义分割模型CBC‑SLP，能够在全部或缺失模态下保持鲁棒性能。

**💡 创新点**

创新点在于将潜在表示拆分为共享和模态专属分量，并通过可训练的门控与随机模态缺失掩码直接在网络结构中实现；无需额外对齐损失，既保留了模态特有信息，又提升了缺失模态下的补偿能力。

**🔧 技术方法**

技术主要包括：多模态ResNet‑50编码器、跨模态融合层、Transformer自注意力与跨模态相关性模块、结构化潜在投影与门控机制、随机模态缺失训练以及二分类交叉熵损失。

**📊 数据集**

使用三组遥感数据集：DSTL（RGB/NIR/SWIR）、Potsdam（RGB/IRRG/DSM）和Hunan（Sentinel‑2 MSI/Sentinel‑1 SAR/DEM）进行实验。

**📈 对比分析**

与MultiSenseSeg、CFFormer、CMX、CMNeXt、Dformerv2‑S、M3L、MMANet等最新方法在IoU与F1指标上进行对比，CBC‑SLP在全部模态场景提升约1.5‑2.7% IoU，在缺失模态下亦保持领先，整体表现优于现有最佳模型。

**⚠️ 局限性**

局限性包括：在某些单一模态（如DEM）下私有分支对结果贡献有限，可能导致性能略低；模型仍依赖随机缺失训练，未验证在极端缺失模式下的极限鲁棒性；结构化投影与门控会增加计算量，需权衡效率与性能。

---

## 293. RAGognizer: Hallucination-Aware Fine-Tuning via Detection Head Integration

**arXiv ID:** 2604.15945 | [PDF](https://arxiv.org/pdf/2604.15945v1)

**作者:** Fabian Ridder `[一作]` (University of Münster), Malte Schilling `[通讯]` (University of Münster)

**通讯引用:** 1253 | [OpenAlex ID](https://openalex.org/A5011185444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了RAGognize数据集与RAGognizer模型，用于闭域检索增强生成（RAG）场景下的幻觉检测与训练

**💡 创新点**

将幻觉检测作为训练目标，联合优化语言模型与检测头，使内部表征可分离并显著降低生成幻觉率，首次实现模型内部的实时幻觉监控

**🔧 技术方法**

采用LoRA微调、在内部隐藏层插入MLP检测头、token级幻觉标注、联合交叉熵与二元交叉熵损失的多任务训练

**📊 数据集**

采集2024年之后的Wikipedia事实，生成多样化问答对，并通过Gemini 2.5 Flash链式思维进行token级标注，构成RAGognize闭域数据集

**📈 对比分析**

在闭域RAGBench和RAGognize测试集上，token级AUROC超过90%，幻觉率从约57%降至13%，回答准确性与语言质量几乎不变；在RAGTruth、HDM-Bench、ConflictQA等公开基准上亦保持或超过现有白盒/黑盒检测器的平均AUROC

**⚠️ 局限性**

仅针对单轮问答、自动化标注与固定损失权重，未对多轮交互、非问答任务或其他非幻觉指标进行评估；对开域或混合情境的迁移性能仍有限

---

## 294. MUSCAT: MUltilingual, SCientific ConversATion Benchmark

**arXiv ID:** 2604.15929 | [PDF](https://arxiv.org/pdf/2604.15929v1)

**作者:** Supriti Sinhamahapatra `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6402 | [OpenAlex ID](https://openalex.org/A5023053982)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `729e5870-4135-47f5-97f2-e3974d07b5dc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含英德中土越五种语言的科学对话音频数据集MUSCAT，并基准评估多模态ASR模型。

**💡 创新点**

首次提供跨语言科学对话数据及针对多语言分段、代码混合、专业词汇等挑战的细粒度评测。

**🔧 技术方法**

采用 Whisper、SALMONN、Phi‑4‑Multimodal、wav2vec2 等现有ASR模型，以及 SHAS、PyanNet 等分段技术。

**📊 数据集**

使用由六条对话、65分钟、9,066词组成的MUSCAT数据集。

**📈 对比分析**

通过 WER、domain‑specific WER 与 PIER 指标比较模型性能，结果显示 Whisper 仍优于其他模型，但整体 WER 仍高达 20–30%，多语言切换与专业词汇表现尤差。

**⚠️ 局限性**

数据量有限且语言分布偏向英语，模型在非英语或代码混合场景下表现差，且部分模型仅支持少数语言。

---

## 295. Making Image Editing Easier via Adaptive Task Reformulation with Agentic Executions

**arXiv ID:** 2604.15917 | [PDF](https://arxiv.org/pdf/2604.15917v1)

**作者:** Bo Zhao `[一作]` (Nanjing University), Wei Ji `[通讯]` (Nanjing University)

**通讯引用:** 22066 | [OpenAlex ID](https://openalex.org/A5100664952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自适应任务重构框架（ATR），通过对图像编辑指令进行查询分析、路由选择和基于路由的多步骤代理执行，来提升现有编辑模型在复杂编辑任务上的可靠性。

**💡 创新点**

创新点在于将图像编辑失败视为任务表述不匹配问题，设计了查询剖析、动态路由与闭环代理执行的整体流程，且不需要改造底层模型。

**🔧 技术方法**

采用多模态大模型进行任务分析，基于路由选择策略（直接、空间拆解、局部编辑），使用LLM/MLLM代理在执行过程中进行指令重写、目标定位、裁剪、合成等操作，并加入反馈与回退机制。

**📊 数据集**

在ImgEdit、PICA和RePlan三个公开基准上进行实验，涵盖添加、删除、属性修改等多种编辑类别。

**📈 对比分析**

相较于直接编辑基线（如Qwen-Edit、Nano Banana/Pro）和现有规划/思考框架（RePlan、EditThinker），ATR在ImgEdit-Hard、PICA和RePlan的综合评分均提升约0.3-0.5分（如Qwen-Edit从3.57提升到4.13），并在多项子任务中实现显著增益。

**⚠️ 局限性**

局限性包括对同时要求精细局部与全局信息的极端案例仍易失败；框架依赖多步骤执行，可能受执行时间和误差累积影响；未深入探究不同模型与路由策略的通用性。

---

## 296. Shaping Plant-Like Shape-Changing Interfaces as Vertical Charts: Maximizing Readability, Aesthetics, and Naturalness

**arXiv ID:** 2604.15902 | [PDF](https://arxiv.org/pdf/2604.15902v1)

**作者:** Elodie Bouzekri `[一作]` (University of Bordeaux, ESTIA Institute of Technology), Guillaume Riviere `[通讯]` (University of Bordeaux, ESTIA Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过研究-设计方法，设计并实现了基于植物形变的竖向数据图表（PlantFORM），用于在共享办公等半公共空间展示十小时可再生能源预测数据，并通过三阶段用户研究评估其可读性、审美性、自然感与学习性。

**💡 创新点**

创新点在于：①首次将植物形态（叶片展开/收缩）作为连续数据系列的可视编码；②提出了四维设计空间（树干形态、锚点方式、装饰形状、动画方式），并从中筛选出最佳的弯曲两侧叶片扩展设计；③研发了低技术材料（木材、纸板、厚纸）与拉绳+弹簧的叶片展开机件，实现了可持续、低能耗的形变交互；④在公共环境中验证植物化数据可视化的“天然感”与“创新感”对用户接受度的影响。

**🔧 技术方法**

技术与实现：使用激光切割木材/纸板构建树干与叶片；利用拉绳、弹簧和导向管实现叶片展开；驱动采用 Raspberry Pi 4 + Arduino Nano + A4988 步进电机 + IR 传感器；灯光使用 RGB 可寻址 LED；PMMA 环形版本使用透光塑料与 LED 内照明；图表动画通过 11 步离散化的叶片/环形位置编码；所有设备以低功耗和可拆卸设计为目标。

**📊 数据集**

数据集：实验使用真实的光伏发电产能曲线，来自作者所在实验室的十小时或更短的可再生能源预测序列。每个实验场景均以不同的峰值位置与持续时间构造，以模拟办公环境下的能源可用性变化。

**📈 对比分析**

比较方法：先在两轮在线研究中评估不同树干形态、锚点方式、装饰形状与动画方式对可读性与审美的影响；随后在28名受试者中做四个高保真原型的交叉试验（PlantSCREEN、PlantFORM、CairnSCREEN、CairnFORM），通过任务成功率、AttrakDiff 体验量表以及半结构化访谈进行比较。结果显示：植物化叶片原型在整体可读性上略逊于条形图（≈5‑10% 任务成功率差距），但在审美、自然感、创新感等主观指标上显著更高；材料低技术化（木材、纸板）显著提升“自然感”，高技术化（PMMA+LED）提升“创新感”。

**⚠️ 局限性**

局限性：①叶片展开机件在实现上存在“过度展开”“粒度不足”等问题，导致可读性下降；②零值编码缺乏明显标记，易引起混淆；③实验中的图表尺寸、光照、噪声等因素未完全控制，影响了跨条件比较；④缺乏长时间持续使用与能耗评估，未验证实际部署可行性；⑤仅测试了能源预测场景，未验证对其他连续数据类型的适用性。

---

## 297. JFinTEB: Japanese Financial Text Embedding Benchmark

**arXiv ID:** 2604.15882 | [PDF](https://arxiv.org/pdf/2604.15882v1)

**作者:** Masahiro Suzuki `[一作]` (Amova Asset Management Co., Ltd.), Hiroki Sakaji `[通讯]` (Hokkaido University)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5028823648)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了JFinTEB基准，用于系统评估日语金融文本嵌入模型的性能；

**💡 创新点**

首个面向日语金融领域的综合基准，包含11项验证过的分类、检索与聚类任务，弥补了现有多语言和通用基准对金融语料的不足；

**🔧 技术方法**

使用多种嵌入模型（如Ruri、Sarashina、GLuCoSE、E5、Jina、OpenAI API等），并在JMTEB代码框架下统一评估宏F1、NDCG@10与V-measure；

**📊 数据集**

采用公开金融数据集（日本维基百科公司信息、Wikinews、JaFin FAQ、PFMT、经济观察者问卷等），并自行构造行业分类与检索数据集；

**📈 对比分析**

通过在所有模型上统一跑评，发现日语专用模型略优于多语言模型，模型规模越大性能提升明显；OpenAI大模型在经济调查相关任务上领先，检索任务已达性能饱和；

**⚠️ 局限性**

局限在于仅覆盖短文本、自动评估指标，缺乏长文档、生成或人类评估；数据来源可能导致模型表现受到预训练污染；跨语言扩展有限。

---

## 298. Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents

**arXiv ID:** 2604.15877 | [PDF](https://arxiv.org/pdf/2604.15877v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出“经验压缩谱”框架，将LLM代理的记忆、技能和规则统一为同一压缩轴，并将20+现有系统映射到该谱上，揭示了缺失的对角线（缺乏自适应跨级压缩），给出了四个结构洞见和开放问题；

**💡 创新点**

首次把记忆、技能与规则视为同一压缩尺度上的不同层级，揭示跨社区低引用率、缺失对角线、评估耦合、转移-特异性折中与生命周期管理缺失，并提出自适应压缩、跨级一致性与持续治理的设计原则；

**🔧 技术方法**

文献计量分析、压缩函数定义、经验压缩层级建模、基准结果对比、系统映射与可视化、设计原则推导；

**📊 数据集**

通过对22篇核心论文共1136条引用的计量分析，以及引用的多种基准（ALFWorld、SpreadsheetBench、多任务评测等）中的实验结果；

**📈 对比分析**

与低压缩层（L1记忆）相比，L2技能在跨域转移中提升约21–68%性能，L3规则（约1000×压缩）在规则塑形中提升7–14%；整体表明更高压缩水平往往带来更高的下游效果；

**⚠️ 局限性**

仅为理论与系统映射框架，缺乏统一的量化实验验证；侧重结构化文本经验，未涵盖多模态；生命周期与跨级治理实现仍处于概念阶段。

---

## 299. Learning to Look before Learning to Like: Incorporating Human Visual Cognition into Aesthetic Quality Assessment

**arXiv ID:** 2604.15853 | [PDF](https://arxiv.org/pdf/2604.15853v1)

**作者:** Liwen Yu `[一作]` (City University of Macau), Sheng Shen `[通讯]` (Torrens University Australia)

**通讯引用:** 2651 | [OpenAlex ID](https://openalex.org/A5100784818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了AestheticNet，一个融合人类眼动扫描行为与语义感知的双路径自动美学质量评估（AQA）框架

**💡 创新点**

创新点在于将人类眼动“扫视语法”作为认知先验加入AQA，并通过Contrastive Gaze Alignment实现低样本、高效率的预训练，从而实现可插拔的视觉注意模块

**🔧 技术方法**

技术包含两路并行架构、Gaze-Align视觉编码器（GAVE）、跨注意力融合机制、对比式眼动对齐（CGA）预训练、以及结合MSE与PLCC的混合损失

**📊 数据集**

使用AVA子集（约89k图像）用于AQA评估，并采用109张高质量眼动追踪图像用于GAVE预训练

**📈 对比分析**

与ResNet-50、NIMA、HyperIQA、CLIP-L、Q-Align等基线对比，AestheticNet在全数据集上取得PLCC 0.747、SROCC 0.740、MSE 0.261，显著优于单一路径模型；在不同网络上插拔GAVE均提升性能

**⚠️ 局限性**

局限在于受AVA数据高斯分布限制，难以覆盖极端美感；仅针对摄影图像，缺乏对非摄影艺术（如抽象艺术）的泛化；眼动先验基于固定类别，跨域迁移能力有限

---

## 300. DPrivBench: Benchmarking LLMs' Reasoning for Differential Privacy

**arXiv ID:** 2604.15851 | [PDF](https://arxiv.org/pdf/2604.15851v1)

**作者:** Erchi Wang `[一作]` (UC San Diego), Ruihan Wu `[通讯]` (OpenAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型（LLM）在判断差分隐私（DP）算法是否满足给定隐私保证上的推理能力，并提出了专门用于评估该能力的基准 DPriv‑Bench。

**💡 创新点**

创新点在于：①首次系统构建面向 DP 证明的自然语言推理基准；②设计了两类实例（机制级与算法级）并通过“正负实例对”避免模式匹配；③对模型表现进行细粒度分析，揭示了当前 LLM 在高级 DP 推理中的弱点。

**🔧 技术方法**

采用的大型语言模型包括闭源模型（Gemini‑3、Claude‑Opus 等）与开源模型（Qwen3、DeepSeek 等），通过统一的二进制判断模板评估其推理准确率；同时使用检索增强（RAG）和定理注入等技术进行对比实验。

**📊 数据集**

使用自建的 DPriv‑Bench 数据集，包含 720 条实例：588 条机制级实例和 132 条算法级实例，涵盖 16 个 DP 研究方向。

**📈 对比分析**

与多种 LLM 进行对比实验：最强闭源模型 Gemini‑3 在机制级任务上达 99.5% 以上准确率，但在算法级任务上整体表现仅 60% 级别；开源模型 DeepSeek‑V3.1‑chat 最高仅 84% 机制级，算法级约 62%；通过对比基线“始终预测是”策略可知高级 DP 推理仍显不足。

**⚠️ 局限性**

局限性在于：①模型在高级 DP 机制（如 Report‑Noisy‑Max、Sparse Vector Technique 等）上存在系统性错误；②对训练数据中的假设易产生幻觉，导致错误判断；③缺乏自动化推理流程，仍需人工校验；④实验未涵盖代码实现级别的隐私验证。

---

## 301. PolarMAE: Efficient Fetal Ultrasound Pre-training via Semantic Screening and Polar-Guided Masking

**arXiv ID:** 2604.15893 | [PDF](https://arxiv.org/pdf/2604.15893v1)

**作者:** Meng Lv `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 30824 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种名为PolarMAE的无监督预训练框架，专门为胎儿超声影像设计，并在预训练过程中引入进展式视觉-语义筛选、声学界限约束和极坐标纹理协同掩蔽三大模块。

**💡 创新点**

创新点包括：①通过视觉DCT特征和MedCLIP语义嵌入双阶段剔除连续扫描造成的视觉与语义冗余；②利用声学束波形成的扇形有效域限定重建区域，避免无信息背景计算；③将极坐标几何先验与局部梯度(HOG)信息融合，构造联合掩蔽概率，引导模型关注高信息、结构丰富的区域。

**🔧 技术方法**

核心技术：Masked AutoEncoder（MAE）框架、低频DCT相似度判别、MedCLIP语义过滤、极坐标几何权重与HOG纹理概率的线性融合、联合概率采样实现自适应可见/重建标记；训练采用PyTorch、RTX 4090 GPU集群。

**📊 数据集**

预训练数据：来自多中心的约430 k帧胎儿超声视频，经过PVSS后压缩为约260 k高信息帧；下游评估使用公开基准SFP（分类）、FIS（检测）和PUBSEG（分割），以及三套私有临床数据集。

**📈 对比分析**

与MAE、LocalMIM、SelectiveMAE、CrossMAE及UltraFedFM等SOTA方法对比，PolarMAE在所有任务上均实现最高或最接近最高的指标：分类Acc ≈ 96.46%、检测mAP ≈ 99.59%、分割mDice ≈ 54.4%；同时预训练时间缩短约1.5×（165 GPU‑h vs 245 h），整体加速≈2.4×。

**⚠️ 局限性**

局限性：依赖预设的相似度阈值与极坐标参数，需人工或预先标注声学扇区边界；仅针对2D胎儿超声，尚未验证跨模态或3D扩展；在极端噪声或非标准扫描模式下性能尚未充分评估。

---

## 302. SENSE: Stereo OpEN Vocabulary SEmantic Segmentation

**arXiv ID:** 2604.15946 | [PDF](https://arxiv.org/pdf/2604.15946v1)

**作者:** Thomas Campagnolo `[一作]` (Inria Centre d'Universite Cote d'Azur), Gaétan Bahl `[通讯]` (NXP Semiconductors)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于立体视觉的开放词汇语义分割框架 SENSE，能够利用左右图像的几何信息与 CLIP 视觉‑语言模型共同完成对任意自然语言查询的像素级分割。

**💡 创新点**

创新点主要包括：①将 CLIP 的中间特征在立体视角下进行 SIEF（Stereo Intermediate‑Embedding Fusion）融合；②在解码器中加入 SDAF（Semantic Disparity Attention Fusion）模块，将深度信息作为空间注意力来细化边界；③完全冻结 CLIP 及外部立体匹配网络，仅训练轻量解码器，保持开放词汇的通用性。

**🔧 技术方法**

核心技术：CLIP ViT‑B/16 视觉编码器、CLIP 文本编码器、Transformer 解码器、SIEF、SDAF、外部立体匹配网络（如 Selective‑IGEV、HITNet、MobileStereoNet）以及 FiLM 条件调制。

**📊 数据集**

使用 PhraseStereo 数据集进行训练，评估基准为 PhraseStereo（参照式分割）、Cityscapes 和 KITTI 2015（零样本语义分割）。

**📈 对比分析**

与 CLIPSeg、OpenSeg、OpenWorldSAM 等基线相比，SENSE 在 PhraseStereo 的 AP 提升 +2.9 点，Cityscapes 的 mIoU 提升 +3.5%，KITTI 2015 的 mIoU 提升 +18%；在参照式分割中，尽管 mIoU 稍低，但 IoU_FG 与 AP 明显优于同类方法，显示了在细节与遮挡处理上的优势。

**⚠️ 局限性**

局限性：对自然语言提示敏感，需人工提供查询；立体匹配模块显著增加推理时延；在极端遮挡或大视差场景下仍可能出现误分；未实现自动查询生成，限制了在无人驾驶等实际场景中的即时使用。

---

## 303. Federated Parameter-Efficient Adaptation for Interference Mitigation at the Wireless Edge

**arXiv ID:** 2604.15936 | [PDF](https://arxiv.org/pdf/2604.15936v1)

**作者:** Evar Jones `[一作]` (Virginia Tech National Security Institute), Sanmay Das `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 5G O‑RAN 环境下，提出一种基于 LoRA 的参数高效适配方案，在 WaveNet 信号分离模型的扩张卷积层中插入低秩适配器，仅训练 5.1% 参数，实现分布式干扰抑制。

**💡 创新点**

创新点在于将 LoRA 迁移到时序扩张卷积，冻结主干网络，仅通过聚合少量适配器参数完成 Federated Learning，既大幅降低通信开销，又避免了全模型聚合导致的灾难性性能下降。

**🔧 技术方法**

使用技术包括 LoRA 参数高效微调、Federated Averaging (FedAvg)、WaveNet 语音分离骨干、FiLM 对比实验，以及在非 IID 干扰环境下的分布式训练与聚合。

**📊 数据集**

采用 ICASSP 2024 RF 信号分离挑战数据集，包含 CommSignal2、CommSignal3、EMISignal1 三种干扰，模拟 5 个 gNB 的不同干扰场景进行训练与评估。

**📈 对比分析**

与全模型微调、FiLM、FedAvg 进行对比，局部 LoRA 与 Fed‑LoRA 在 BER 上分别提升约 12.8% 与 12.6%，与全微调相差不到 0.5%，且 Fed‑LoRA 在数据稀缺节点上明显优于局部 LoRA；FedAvg 在非 IID 情况下表现灾难性。

**⚠️ 局限性**

局限性包括仅在模拟环境验证，未在真实 O‑RAN 试验台上测试；CommSignal3 对抗效果差，需改进骨干结构；适配器秩固定，未实现自适应秩；Fed‑LoRA 在极端异构环境下仍可能出现性能下降。

---

## 304. Hierarchical Codec Diffusion for Video-to-Speech Generation

**arXiv ID:** 2604.15923 | [PDF](https://arxiv.org/pdf/2604.15923v1)

**作者:** Jiaxin Ye `[一作]` (Fudan University), Hongming Shan `[通讯]` (Fudan University)

**通讯引用:** 4627 | [OpenAlex ID](https://openalex.org/A5049086157)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于层次化语音离散码的分层扩散变压器，实现视频到语音的生成

**💡 创新点**

首次将语音的低层语义与高层韵律的层次结构融入离散扩散模型，并提出双尺度AdaLN对视觉特征进行细粒度对齐

**🔧 技术方法**

利用RVQ编码器、离散扩散模型、Hierarchical Codec Diffusion Transformer及双尺度AdaLN实现分层掩码预测与视觉条件注入

**📊 数据集**

在VoxCeleb2训练，评估于LRS2、LRS3以及真实电影数据上

**📈 对比分析**

与FTV、AlignDiT、EmoDubber等SOTA方法对比，取得更高的MOS、低WER、优越的DNSMOS、语音同步与情感识别准确率

**⚠️ 局限性**

受训练语料受限，导致说话人相似度略低，表达性仍有提升空间，需更丰富的多说话人与情感数据

---

## 305. Continuous benchmarking: Keeping pace with an evolving ecosystem of models and technologies

**arXiv ID:** 2604.15919 | [PDF](https://arxiv.org/pdf/2604.15919v1)

**作者:** Jan Vogelsang `[一作]`, Anno C. Kurth `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一套面向高性能计算（HPC）应用的连续基准测试（Continuous Benchmarking）流水线，通过统一入口、层级化配置和中心化存储，实现了基准测试的自动化、可复现和可重复，尤其针对神经网络模拟器NEST的性能评估与优化。

**💡 创新点**

创新点在于：① 把CI/CD理念引入HPC基准测试，创建了统一入口、可用户无关的执行器；② 采用层级化、可继承的配置体系，将硬件、软件、模型、实验等配置分离并交叉协作；③ 通过中心化存储和元数据追踪，实现多机器、多模型下的结果可追溯、可对比；④ 将持续基准测试嵌入日常开发，促进快速反馈与性能迭代。

**🔧 技术方法**

技术包括：GitLab CI/CD、Jacamar（HPC驱动）、模板化流水线（工作流、平台、实现三层）、参数化配置文件、元数据记录、自动化结果收集与可视化。实验中使用NEST模拟器（版本3.5与3.8）、HPC-Benchmark、Microcircuit、Multi‑Area等三种神经网络模型进行弱/强缩放测试。

**📊 数据集**

主要数据集是三种规模可扩展的神经网络模型：HPC‑Benchmark（均衡随机网络）、Microcircuit（完整大脑皮层层级）和Multi‑Area（多区块大脑视觉网络）。每个模型在JURECA、JUSUF等Jülich超算平台上执行。

**📈 对比分析**

比较方法为将不同版本NEST（3.5 vs 3.8）和不同实现（如使用查找表加速STDP）在同一硬件上执行相同参数空间的弱/强缩放基准。评估指标包括：壁钟时间、实时因子、各阶段时间占比（更新、拼接、通信、投递）。结果显示：3.8版在弱缩放上显著提升（总运行时间↓26%），投递时间↓45%，微电路模型加速60%，多区模型提升37%。使用查表实现对STDP的加速在弱缩放中平均提升约5%。

**⚠️ 局限性**

局限性包括：① 需要访问HPC的用户级环境，难以实现完全无用户干预；② 安全与计费限制导致无法直接使用容器或通用镜像；③ 元数据与结果的中心化存储对命名冲突、检索效率提出挑战；④ 依赖HPC系统特定模块，跨平台迁移仍需人工调整；⑤ 在高节点数下，系统监控等背景噪声可能影响性能评估。

---

## 306. CLOTH-HUGS: Cloth Aware Human Gaussian Splatting

**arXiv ID:** 2604.15875 | [PDF](https://arxiv.org/pdf/2604.15875v1)

**作者:** Sadia Mubashshira `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Cloth-HUGS，一个基于高斯散射的神经渲染框架，用分离的高斯层独立建模人体与服装，实现高逼真度的服装动态重建与合成。

**💡 创新点**

创新点在于将身体和服装显式分离为独立高斯层，并结合物理启发的监督（仿真一致性、ARAP、LBS约束）以及深度感知多通道渲染，显著提升服装细节与时间一致性。

**🔧 技术方法**

采用 SMPL 骨骼驱动、三平面特征编码、3D 高斯散射、物理约束损失、深度感知多通道渲染以及 SNUG 预训练的服装网格监督。

**📊 数据集**

在 NeuMan 与 ZJU‑MoCap 两个单目/多视角人像数据集上进行训练与评测。

**📈 对比分析**

与 NeRF‑T、HyperNeRF、Vid2Avatar、NeuMan、HUGS 等基线对比，PSNR、SSIM、LPIPS、FID 指标均达到最高，LPIPS 下降约 28%，Fid 下降 5–6%，并实现实时 60+ FPS。

**⚠️ 局限性**

局限在于对预训练服装网格的依赖，难以处理极端姿态或大幅度衣物变形，对极端遮挡的深度排序仍可能出现误差。

---

## 307. Feature Toggle Dynamics in Large-Scale Systems: Prevalence, Growth, Lifespan, and Benchmarking

**arXiv ID:** 2604.15872 | [PDF](https://arxiv.org/pdf/2604.15872v1)

**作者:** Xhevahire Tërnava `[一作]` `[通讯]` (LTCI, Télécom Paris, Institut Polytechnique de Paris), Xhevahire Tërnava (LTCI, Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 Kubernetes 与 GitLab 两个大型开源项目的版本控制历史进行长期跟踪，量化了特性切换（Feature Toggle）在代码库中的增删、存活时间和积累情况，并基于此提出了可供项目自评的基准框架。

**💡 创新点**

创新点包括：①首次从纵向视角系统地测量特性切换的生命周期（包含寿命分布、停留时间等）以及其在不同项目中的差异；②提出五项量化指标（切换密度、流量率、净积累、清理比例、归一化生命周期）并给出基于实测阈值的评估区间；③构建了交互式仪表盘，支持团队自评与社区共享。

**🔧 技术方法**

主要技术手段为：①基于 Git 历史的差分（diff）解析与正则匹配自动抽取特性切换的添加/删除事件；②使用 Kaplan‑Meier 生存分析评估切换寿命分布；③统计与可视化工具（Python pandas、matplotlib/plotly）实现指标计算与阈值划分。

**📊 数据集**

数据集包含约 4,000 条特性切换事件，涵盖 Kubernetes 约 8.5 年（10M LoC）与 GitLab 约 5 年（5M LoC）的完整提交记录；其中 Kubernetes 603 次添加、448 次删除，GitLab 3,442 次添加、3,038 次删除。

**📈 对比分析**

比较方法为：将上述五项指标与基准阈值区间对齐，形成“低、中、高”级别评估；性能体现为两系统在每项指标上分别落在不同区间（例如 GitLab 频繁度高但净积累严重），展示了不同项目的特性切换管理风格及潜在技术债务风险。

**⚠️ 局限性**

局限性包括：①仅研究两大开源项目，缺乏对私有或小型项目的泛化；②仅聚焦布尔型特性切换，未覆盖多值或百分比投放；③依赖文本匹配，可能漏检不在标准位置定义的切换；④时间跨度仅至 2025 年，未能捕捉 AI 辅助开发工具对切换行为的新影响。

---

## 308. UniEditBench: A Unified and Cost-Effective Benchmark for Image and Video Editing via Distilled MLLMs

**arXiv ID:** 2604.15871 | [PDF](https://arxiv.org/pdf/2604.15871v1)

**作者:** Lifan Jiang `[一作]` (Zhejiang University), Deng Cai `[通讯]` (Zhejiang University)

**通讯引用:** 24637 | [OpenAlex ID](https://openalex.org/A5037942269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的图像与视频编辑评测基准 UniEditBench，兼顾重建式与指令驱动两大范式，并提供统一的任务分类、标准化提示模板与多维评价指标；

**💡 创新点**

通过构建多源、结构化的数据集、引入可解释的多维评测维度以及利用知识蒸馏将大规模 MLLM 评测器压缩为轻量级 4B/8B 模型，实现在多模态编辑任务上的可扩展且成本友好的自动评测；

**🔧 技术方法**

采用 Qwen3‑VL‑235B‑A22B 作为教师模型，使用 LoRA 与两阶段（空间→时间）微调的学生模型，结合 Chain‑of‑Thought 推理生成多维评分；

**📊 数据集**

数据来源于公开基准（PIE‑Bench、ImgEdit）、网络抓取及多模态生成模型（FLUX、SD3、Wan、HunyuanVideo）等，覆盖 633 张图像与 77 条视频，涵盖 9 种图像编辑操作与 8 种视频编辑操作；

**📈 对比分析**

在 25 种图像编辑模型与 8 种视频编辑模型上进行评测，使用 4B/8B 蒸馏评测器得到结构、文本对齐、背景一致性、自然度及视频时空一致性等维度分数；实验表明蒸馏模型与教师模型以及人工评测高度一致，且相较于零拷贝大模型大幅降低 12‑24 GB VRAM 与推理时延；

**⚠️ 局限性**

局限性在于仅支持单轮编辑、可能继承教师模型偏见、以及任务词典与样本规模不足以覆盖极端或多轮交互式编辑场景。

---

## 309. Splats in Splats++: Robust and Generalizable 3D Gaussian Splatting Steganography

**arXiv ID:** 2604.15862 | [PDF](https://arxiv.org/pdf/2604.15862v1)

**作者:** Yijia Guo `[一作]` (Peking University), Lei Ma `[通讯]` (Peking University)

**通讯引用:** 15833 | [OpenAlex ID](https://openalex.org/A5022157306)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为 Splats in Splats++ 的统一隐写框架，在保持原始 3D Gaussian Splatting（3DGS）渲染管线不变的前提下，直接将 3D/4D 内容嵌入原始模型，实现隐蔽信息的高容量存储。

**💡 创新点**

创新点包括：1）基于 SH 系数重要性层级的分级加密策略，最大化容量同时保持可视无感；2）使用多分辨率 Hash‑Grid 引导的不透明度映射，将几何信息映射至连续潜在空间；3）梯度门控一致性损失，强制原场景与隐藏场景在几何属性上保持紧耦合，从而显著提升对结构扰动和专用攻击的鲁棒性。

**🔧 技术方法**

技术实现涵盖：Spherical Harmonics（SH）重要性分析与分级加密、Hash‑Grid 位置编码+MLP 的不透明度恢复网络、梯度门控一致性损失、传统 3DGS 渲染管线、以及针对 4DGS 的时间域变形场景训练。

**📊 数据集**

实验使用了 NeRF Synthetic、Tanks & Temples、Mip‑NeRF360 三个 3D 视角合成数据集，以及 D‑NeRF、HyperNeRF 两个 4D 动态场景数据集。

**📈 对比分析**

与 GS‑Hider、SecureGS、3DGS+StegaNeRF 等基线相比，在 PSNR、SSIM、LPIPS 等视觉质量指标上均优越；渲染速度提升约 3 倍（≈100 FPS），训练时间缩短一半；在结构剪枝（Opacity Pruning）和 GSPure 压缩攻击下，鲁棒性得分显著高于其它方法。

**⚠️ 局限性**

局限性：对高反射或镜面材质的表面细节可能略有影响；框架目前仅支持嵌入 3D/4D 体数据，无法直接嵌入任意二进制流，限制了对更广泛 3D 资产的通用适配。

---

## 310. QuantSightBench: Evaluating LLM Quantitative Forecasting with Prediction Intervals

**arXiv ID:** 2604.15859 | [PDF](https://arxiv.org/pdf/2604.15859v1)

**作者:** Jeremy Qin `[一作]` (ELLIS Institute Tübingen), Maksym Andriushchenko `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了QuantSightBench基准，用于评估大语言模型（LLM）在连续数值预测中的不确定性估计能力，采用预测区间而非传统的二选一或多选问答；

**💡 创新点**

创新点包括：①将评估焦点从离散事件转向连续量预测；②使用预测区间作为评价接口，能够同时衡量校准（coverage）和尖锐度（MLIS）；③引入代理式检索设置，模拟真实环境下模型主动检索信息的情景；

**🔧 技术方法**

技术手段包括：大型LLM（包括OpenAI、Anthropic、Google等前沿模型及GLM、Kimi、DeepSeek、Grok等开源模型）的调用；检索管道（512-token分块、OpenAI文本嵌入、相似度检索）；提示工程与推理力度调节；以及对区间质量的统计评估（经验覆盖率、对数Winkler区间得分）。

**📊 数据集**

数据集：约32万篇新闻文章（来源涵盖Forbes、CNN、德意志之声等），从2025年1-8月抽取；从2025年9月-2026年1月的新闻生成1,000道量化预测问题；每题包含目标数值及其真实解。

**📈 对比分析**

比较方法：在零-shot、背景上下文、代理式三种设置下，对11种模型进行覆盖率和MLIS评估。结果显示：最高覆盖率仅79.1%（Gemini 3.1 Pro），所有模型均低于90%目标；覆盖率随数值规模增大而下降，模型间差距随代数提升而缩小；代理检索显著提升弱模型表现，推理力度提升覆盖率与MLIS。

**⚠️ 局限性**

局限性：检索语料固定且非实时，缺乏动态更新和多轮校准；仅评估90%置信区间，未覆盖更广泛的置信水平；数据集规模有限（1,000题），可能不足以全面检验不同领域；未考虑完整概率分布或更高级的自适应置信区间方法。

---

## 311. Limits of Lamarckian Evolution Under Pressure of Morphological Novelty

**arXiv ID:** 2604.15854 | [PDF](https://arxiv.org/pdf/2604.15854v1)

**作者:** Jed R Muff `[一作]` (Vrije Universiteit), A. E. Eiben `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

比较了在模具机器人上，传统达尔文进化和拉马克进化在仅目标任务与目标+形态多样性双目标下的性能差异。

**💡 创新点**

揭示了拉马克进化在形态多样性压力下受限的根本原因——父代与子代形态相似度下降削弱了控制器继承的优势。

**🔧 技术方法**

使用ARIEL框架的模块化机器人、MuJoCo物理模拟、基于树结构的体态遗传编码、三层ANN控制器、CMA‑ES学习、(μ+λ)进化策略以及多目标评价（距离+形态新颖性）。

**📊 数据集**

实验数据基于在MuJoCo模拟中生成的成千上万种机器人体态和控制器的自适应评估，未使用公开数据集。

**📈 对比分析**

通过2×2因子设计（遗传机制×目标函数），比较平均和最大运动距离；结果显示：拉马克进化在仅运动目标下优于达尔文进化，但加入形态多样性后性能显著下降，达尔文进化基本不受影响。

**⚠️ 局限性**

局限在于拉马克进化高度依赖父子形态相似性，形态多样性或环境动态变化会显著削弱其优势；需进一步探索自适应继承策略或混合进化方案。

---

## 312. Scalable Deterministic Task Offloading and Resource Allocation in the IoT-Edge-Cloud Continuum

**arXiv ID:** 2604.15901 | [PDF](https://arxiv.org/pdf/2604.15901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 313. Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction

**arXiv ID:** 2604.15941 | [PDF](https://arxiv.org/pdf/2604.15941v1)

**作者:** Haato Watanabe `[一作]` (University of Tokyo), Nobuyuki Umetani `[通讯]` (University of Tokyo)

**通讯引用:** 1756 | [OpenAlex ID](https://openalex.org/A5085087441)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出神经 Gabor splatting，在每个高斯原语中嵌入轻量级 MLP，以实现单原语内的多彩纹理和视角相关性，并配合频率感知稠密化策略进行自适应原语增删。

**💡 创新点**

① 轻量级 MLP 取代单色原语，能够在同一原语内表达复杂的空间与视角变化的色彩图案；② 通过基于频段误差的稠密化方法，精准地在高频细节区块增删原语，从而在有限预算下显著提升重建精度。

**🔧 技术方法**

基于 2D Gaussian splatting 的渲染框架；SIREN 激活的单隐藏层 MLP 用于色彩预测；FFT + 带通滤波 + 局部均值用于构造频域误差；alpha 混合、L1+SSIM 损失、梯度/误差/频率稠密化等技术。

**📊 数据集**

主要在 Mip-NeRF360、DTU、Tanks and Temples、以及专门的高频纹理数据集（含棋盘/细纹理场景）上进行实验，亦对 Mip-NeRF360 的 Bonsai 场景做对比。

**📈 对比分析**

与 3DGS、2DGS、3D Gabor splatting、NTS、NEST 等方法在相同数据预算下对比；评估指标为 PSNR、SSIM、LPIPS。结果显示：在高频场景下 PSNR 上升约 1‑3 dB，SSIM 上升 0.02‑0.04，LPIPS 降低 0.01‑0.03；在低预算（≤5%）下仍保持优于 3DGS、NEST、NTS 的性能；训练时间与 NEST、NTS 相近，远低于 3DGS。

**⚠️ 局限性**

① 目前仅适用于静态几何场景，无法直接处理体积现象或动态变化；② 每个原语都需执行 MLP 推理，导致相对较高的计算开销；③ 对低频场景时 MLP 资源可能被浪费；④ 仍受限于轻量 MLP 的容量，进一步压缩或共享参数的研究尚未深入。

---

## 314. Polarization by Default: Auditing Recommendation Bias in LLM-Based Content Curation

**arXiv ID:** 2604.15937 | [PDF](https://arxiv.org/pdf/2604.15937v1)

**作者:** Nicolò Pagan `[一作]` (University of Zurich), Petter Törnberg `[通讯]` (University of Amsterdam)

**通讯引用:** 2379 | [OpenAlex ID](https://openalex.org/A5047808749)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过受控模拟研究，分析了三大大型语言模型（LLM）提供商（OpenAI、Anthropic、Google）在社交媒体内容选择中的偏见，使用了六种提示策略（一般、流行、引人入胜、信息性、争议性、中立）。

**💡 创新点**

研究的创新点在于同时比较了不同提供商和平台的偏见表现，并探讨了提示设计对偏见的影响，填补了社交媒体内容策划领域的研究空白。

**🔧 技术方法**

使用了随机森林分类器和SHAP特征重要性评分等技术来分析推荐结果和偏见特征。

**📊 数据集**

使用了来自Twitter/X、Bluesky和Reddit的真实社交媒体数据集，共进行了540,000次推荐模拟。

**📈 对比分析**

通过比较不同提供商的推荐结果，发现偏见在不同模型和提示策略下表现出显著差异。GPT-4o Mini在提示间表现最为一致，而Claude和Gemini在处理毒性内容方面表现出较高的适应性。

**⚠️ 局限性**

研究的局限性包括：人口统计特征是推断而非验证的，少数群体状态的高未知率（48.4%）限制了结论的可靠性，且推荐是非个性化的，未考虑真实系统中使用的元数据。

---

## 315. Backdoors for Quantified Boolean Formulas

**arXiv ID:** 2604.15927 | [PDF](https://arxiv.org/pdf/2604.15927v1)

**作者:** Leif Eriksson `[一作]` (Linköping University), Mateusz Rychlicki `[通讯]` (University of Leeds)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5028671895)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文研究了量化布尔公式（QBF）在以可解子类（Horn、2‑SAT、Affine）为基类的回退变量集（backdoor）参数化下的复杂性，并给出了评估与检测的算法与下界。

**💡 创新点**

主要创新在于证明即使回退变量集大小为常数，QBF仍是PSpace‑hard；提出了“增强回退变量集”与“保守普遍集消除”技术，扩展了可解子类的回退变量集覆盖范围，并提供了新的FPT算法。

**🔧 技术方法**

采用了折叠（squishing）逆量化消除、子公式分裂、归约为k‑Disjunct公式、重要分离器、随机收缩与图不破坏性等参数化算法技术。

**📊 数据集**

由于本工作主要为理论复杂性分析，未使用具体实验数据集。

**📈 对比分析**

通过与已知的FPT/ W[1]‑hard 结果对比，证明了在仅以回退变量集大小为参数时无FPT解；而在加上量化深度参数时，2‑SAT与AFFINE可实现FPT；同时展示了增强回退变量集在结构类中的FPT评估与检测性能。

**⚠️ 局限性**

主要局限在于对 Horn 的 3‑Horn 仍为 W[1]‑hard，且对更一般的量化层级或更宽松的子类尚未获得完整的FPT阈值；实验验证与多变量实用性仍待研究。

---

## 316. Compliance in Databases: A Study of Structural Policies and Query Optimization

**arXiv ID:** 2604.15861 | [PDF](https://arxiv.org/pdf/2604.15861v1)

**作者:** Ahana Pradhan `[一作]` (IIIT Bangalore), Srinivas Vivek `[通讯]` (IIIT Bangalore)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套用于生成和分析结构化内容型合规策略的框架，并基于该框架为TPC‑H基准生成了可控的、层级化的策略工作负载，随后在PostgreSQL中对四种原生策略执行机制（RLS、Indexed RLS、Secure Views、Query Rewrite）进行了系统实验评估。

**💡 创新点**

创新点包括：①构建了基于查询图结构的原子SQL合规策略分类与语法；②提出了白盒与黑盒两类执行模式及其对应的层级化策略生成规则；③将策略工作负载注入TPC‑H，实现了可复现的实验平台；④在实验中揭示了策略结构对优化器行为与性能的主导作用，提出了“策略感知索引”和“基于策略的成本估计”等未来方向。

**🔧 技术方法**

技术手段包括：SQL Row-Level Security (RLS) 与其索引化变体；安全视图与安全防护视图；利用外部重写器（如SQLRewriter）实现查询重写；用户自定义函数 (UDF) 作为黑盒执行；层级化策略生成的层次模型；以及对Planner、Executor时间和执行成本的度量。

**📊 数据集**

数据集为TPC‑H 1.0 与 10.0 的标准语义数据，利用GitHub公开的 policy-generation 代码生成结构化策略；实验在Intel Xeon 2×6核、128GB内存环境下进行。

**📈 对比分析**

比较方法为：对同一组查询在四种执行策略下分别测量规划时间、执行时间和是否超时；对黑盒 RLS 进一步评估 Indexed RLS 的索引加速效果；对视图与重写的交叉比较；结果显示：Query Rewrite 在白盒模式下执行最快，但规划成本最高；Indexed RLS 在黑盒模式下对环状策略显著提升鲁棒性；视图介于两者之间。总体而言，策略结构决定性能，而非单一执行机制占优。

**⚠️ 局限性**

局限性包括：①实验仅覆盖 PostgreSQL 及一款商业引擎，未涵盖其他主流DBMS；②策略生成依赖手工定义的语法与规则，缺乏自动化的多层次策略合成算法；③对大规模并行与分布式优化器的评估不足；④未对安全性侧信道（如时间泄漏）进行量化验证；⑤缺乏对策略冲突、优先级和覆盖关系的完整处理。

---

## 317. From Competition to Coopetition: Coopetitive Training-Free Image Editing Based on Text Guidance

**arXiv ID:** 2604.15948 | [PDF](https://arxiv.org/pdf/2604.15948v1)

**作者:** Jinhao Shen `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 47609 | [OpenAlex ID](https://openalex.org/A5100404130)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种零样本、无训练的图像编辑框架CoEdit，采用协同竞争（coopetition）机制在注意力控制上实现编辑与重构的协同优化。

**💡 创新点**

创新点在于：① 将注意力竞争转化为协同竞争；② 通过双熵注意力调节实现空间上的可编辑与不可编辑区域精准分割；③ 引入熵驱动的潜在细化机制保证时序一致性；④ 设计综合指标Fidelity‑Constrained Editing Score (FCES) 对编辑质量与结构保真度进行统一评估。

**🔧 技术方法**

技术包括Stable Diffusion的无训练编辑、双熵注意力（Dual‑Entropy Attention Manipulation）、熵驱动潜在细化（Entropic Latent Refinement）、自适应阈值生成（Own*），以及基于CLIP的相似度、PSNR、LPIPS、SSIM等评估指标。

**📊 数据集**

使用PIEBench（700+实例，包含源/目标提示与像素级编辑掩码）以及PIEBench++作为测试数据集。

**📈 对比分析**

与P2P、PnP、Pix2Pix‑Zero、NTI、NPI、DI、MasaCtrl、DDCM、PostEdit、iRFDS、h‑Edit等现有无训练编辑方法在三种CFG设置下进行对比，CoEdit在FCES、CS_i、CS_r、PSNR、LPIPS、SSIM上均位列榜首，尤其在重构精度与编辑多样性之间取得最佳平衡；用户研究亦显示其在图像保真度、编辑质量与整体评分上优于其他方法。

**⚠️ 局限性**

限制：依赖Stable Diffusion模型，无法直接处理极大分辨率或需要大量后处理；在非常强的编辑提示或极端背景变化时仍可能出现细节模糊；缺乏对动态场景或视频的实时应用支持。

---

## 318. CIMple: Standard-cell SRAM-based CIM with LUT-based split softmax for attention acceleration

**arXiv ID:** 2604.15944 | [PDF](https://arxiv.org/pdf/2604.15944v1)

**作者:** Bas Ahn `[一作]` (Eindhoven University of Technology), Henk Corporaal `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 7393 | [OpenAlex ID](https://openalex.org/A5081768631)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于32kb SRAM双银行的全数字CIM加速器CIMple，用于Transformer自注意力层的加速。

**💡 创新点**

创新点在于：① 8位并行权重输入的双银行CIM架构；② 采用固定点拆分LUT软最大化降低软最大操作延迟；③ 无需额外CPU/SIMD软最大单元，保持高能效。

**🔧 技术方法**

技术上使用ST 28nm FD‑SOI工艺实现标准单元SRAM，结合OAI多路选择器实现8×8 MAC；软最大使用两张1D LUT分离分子分母并行计算；并行数据流和量化单元实现INT8推理。

**📊 数据集**

在TinyLlama模型的int8量化版上使用lm‑evaluation‑harness验证softmax近似误差，主要评测ARC Challenge、HellaSwag、OpenBookQA、Winogrande等任务。

**📈 对比分析**

与现有CIM Transformer加速器（CIMFormer、TranCIM、MultCIM）对比，CIMple在相同28nm节点下达成26.1 TOPS/W（0.85V）和2.31 TOPS/mm²（1.2V）性能，优于多数同类方案，且兼容Encoder/Decoder/Encoder‑Decoder三种模型。

**⚠️ 局限性**

局限性包括：软最大近似仍会引入微小精度损失；未实现对大规模LLM全模型的完整评估；外部内存访问能耗仍是主要瓶颈，后续工作需进一步优化数据映射与压缩。

---

## 319. VADF: Vision-Adaptive Diffusion Policy Framework for Efficient Robotic Manipulation

**arXiv ID:** 2604.15938 | [PDF](https://arxiv.org/pdf/2604.15938v1)

**作者:** Xinglei Yu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16264 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了 Vision-Adaptive Diffusion Policy Framework (VADF)，通过训练阶段的自适应损失网络(ALN)和推理阶段的层次视觉任务分段器(HVTS)来提升机器人操作的训练效率与执行成功率。

**💡 创新点**

双重自适应机制：1）在线时间步重要性采样与硬负样本挖掘加速收敛；2）零样本视觉语言驱动的任务分解与动态采样预算实现推理加速。

**🔧 技术方法**

采用低成本 MLP 自适应时间步采样、REINFORCE 强化学习、基于 Qwen2-VL 的视觉语言模型、DDPM/DDIM diffusion policy、硬负样本加权与动态噪声步骤调度等技术。

**📊 数据集**

使用 Robomimic、Kitchen、Adroit、Push‑T、BlockPush 等模拟环境中的演示轨迹与图像观察数据集。

**📈 对比分析**

与 vanilla Diffusion Policy、DDIM、DP3 等基线在相同硬件与训练步数下对比；VADF 在多任务上提升成功率约 5–10%，推理延迟降低 20–60%，早期成功率提升 5–10%。

**⚠️ 局限性**

依赖零样本 VLM 的可靠性，未在非 diffusion 基础架构上验证；对复杂动态环境中实时性与鲁棒性的全面评估仍缺失。

---

## 320. New Kids: An Architecture and Performance Investigation of Second-Generation Serverless Platforms

**arXiv ID:** 2604.15916 | [PDF](https://arxiv.org/pdf/2604.15916v1)

**作者:** Trever Schirmer `[一作]` (Technische Universität Berlin), David Bermbach `[通讯]` (Technische Universität Berlin)

**通讯引用:** 1909 | [OpenAlex ID](https://openalex.org/A5032206962)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七大商业 FaaS 平台（AWS Lambda、Cloudflare Workers、Deno Deploy、Fastly Edge Compute、Fly.io、Google Cloud Run Functions 与 Oracle Cloud Infrastructure Functions）进行了架构梳理，并在 2025‑2026 年共 38,165,688 次函数调用中执行了微基准测试，评估了热/冷启动、地理分布、性能一致性、弹性与数据传输等维度。

**💡 创新点**

首次将服务器无状态计算划分为“第一代” (基于容器/微虚拟机、单区域) 与“第二代” (轻量化隔离、全球边缘部署) 两个世代，揭示后者在低延迟与冷启动方面的显著优势，并将公开文档与实测数据结合，提供完整的开放源码基准与结果。

**🔧 技术方法**

技术包括：公开文档与演讲收集、平台内部架构推断、基于 HTTP/1.1 的请求工作流模拟、微基准脚本（JavaScript/Go）以及针对不同资源配置的实验设计；使用 Flyd/Corrosion、Firecracker、gVisor、V8 isolate、Wasm 运行时等关键组件。

**📊 数据集**

数据集：七个平台各自的资源模型与功能实现细节；实验数据来自 3816 万个函数调用，涵盖不同内存/CPU 配置、地区分布与负载曲线。

**📈 对比分析**

比较方法：对每个维度（热延迟、冷启动延迟、全球延迟、性能一致性、弹性、数据传输）统计中位数、分位数、标准差等指标；结果显示边缘平台（CW、DD、FEC）热延迟约 8 ms，冷启动近似为热延迟，第二代平台在 10 ms 级别实现；第一代平台热延迟约 40 ms，冷启动显著更高。

**⚠️ 局限性**

局限性：只覆盖七个封闭源平台，架构推断依赖公开信息（可能过时）；未评估每次调用成本；实验环境与平台实际生产环境可能存在差异；未来需扩展至更多平台与开源方案，并研究统一的成本比较方法。

---

## 321. A Reconfigurable Pneumatic Joint Enabling Localized Selective Stiffening and Shape Locking in Vine-Inspired Robots

**arXiv ID:** 2604.15907 | [PDF](https://arxiv.org/pdf/2604.15907v1)

**作者:** Ayodele James Oyejide `[一作]` (Kadir Has University), Fabio Stroppa `[通讯]` (Kadir Has University)

**通讯引用:** 543 | [OpenAlex ID](https://openalex.org/A5054512048)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了可重构气动关节（RPJ），实现了藤式软机器人局部可调刚度、形状锁定、级联退缩和负载承载能力的提升。

**💡 创新点**

通过在藤机器人骨干中嵌入可独立气压调节的多腔RPJ，实现了既保持整体柔顺性又能局部增刚的机制，并首次展示了无外部支撑的自由空间形状锁定和级联退缩。

**🔧 技术方法**

采用LDPE薄膜与布料加固的多腔气动结构，气压控制与筋索驱动相结合；结合解析模型、实验测量（压力、接触力、载荷位移）、与层粘层弹性机理对比。

**📊 数据集**

通过实验数据收集（起始与稳态伸长压力、RPJ压力、接触力、尖端位移、载荷‑位移曲线、成长速度与曲率测量），与层粘化实验数据进行对标。

**📈 对比分析**

采用成长时间、稳态速度、最大曲率、到达曲率时间等指标进行对比；RPJ在同等压力下成长速度约快2.1倍，成长时间约1.9倍，最大可达曲率≥90°，曲率提升时间比层粘化快5倍，整体性能显著优于传统层粘化方案。

**⚠️ 局限性**

受限于气管布局导致的流动滞后，限制了多关节长机器人在自由空间的可伸缩和退缩；未考虑动态压降与材料粘弹性，缺乏闭环压力与姿态控制；对长杆自重导致的下垂影响有限，需更多的环境支撑或额外关节。

---

## 322. AeroDeshadow: Physics-Guided Shadow Synthesis and Penumbra-Aware Deshadowing for Aerospace Imagery

**arXiv ID:** 2604.15903 | [PDF](https://arxiv.org/pdf/2604.15903v1)

**作者:** Wei Lu `[一作]` (Anhui University), Si-Bao Chen `[通讯]` (Anhui University)

**通讯引用:** 3612 | [OpenAlex ID](https://openalex.org/A5030559313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了AeroDeshadow框架，分两阶段进行航天影像阴影合成与去阴影处理。

**💡 创新点**

创新点在于将物理阴影衰减模型与空间衰减机制相结合生成真实阴影数据，并通过光晕与阴影半影双流解耦实现渐变区域的精细恢复。

**🔧 技术方法**

采用的技术包括Physics-aware Degradation Shadow Synthesis Network（PDSS-Net）与Penumbra-aware Cascaded DeShadowing Network（PCDS-Net），并融合SDCA注意力模块、AFF融合以及语义聚合。

**📊 数据集**

使用了自建的AeroDS数据集，其中AeroDS-Syn为配对合成阴影图像（2000组）和AeroDS-Real为真实阴影测试集（260组）。

**📈 对比分析**

与多种现有阴影合成与去阴影方法对比，实验表明AeroDeshadow在PSNR/SSIM/RMSE、Entropy和BRISQUE等指标上均优于SOTA，特别是在跨域真实数据上表现显著。

**⚠️ 局限性**

局限性在于仍依赖物理参数估计的精度，且对复杂大气遮蔽（如云影）处理尚未覆盖，未来需扩展模型以适应更丰富的阴影形态。

---

## 323. Towards Rigorous Explainability by Feature Attribution

**arXiv ID:** 2604.15898 | [PDF](https://arxiv.org/pdf/2604.15898v1)

**作者:** Olivier Létoffé `[一作]` (University of Toulouse), Joao Marques-Silva `[通讯]` (University of Lleida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并纠正了 SHAP 在特征归因中的理论缺陷，提出了一种基于逻辑的修正 SHAP 分数（nuSHAP），并给出了其计算方法。

**💡 创新点**

创新点在于：①提出了新的特征重要性游戏，特征值仅取 0/1，消除了原 SHAP 期望值定义导致的误导；②证明该特征函数满足强值独立性、特征相关性和数值中立性等关键属性；③用 CGT 近似算法实现对修正 SHAP 的高效计算。

**🔧 技术方法**

主要技术包括：逻辑基解释（WAXp、CXp 等）、Shapley 值理论、游戏论特征函数设计、CGT 近似算法、模型感知与模型无关两种实现路径。

**📊 数据集**

实验数据集：PMLB benchmark 中的多种表格分类数据集，以及 MNIST 手写数字数据集；使用的模型包括逻辑回归、决策树、kNN、XGBoost、CNN。

**📈 对比分析**

比较方法：对每个实例分别计算 SHAP（原版和绝对值排序）和 nuSHAP 的特征重要性排名，使用 Rank‑Biased Overlap (RBO) 衡量排名相似度；同时记录计算耗时。结果显示 nuSHAP 与 SHAP 的排名几乎不相关，但两者的计算时间相近。

**⚠️ 局限性**

限制：目前仍无多项式时间算法实现修正 SHAP 的精确计算；依赖 CGT 近似，存在理论误差；对复杂模型的模型感知实现仍需构建可判定 WAXp 的可解释器，实际应用中可能受限。

---

## 324. How Hypocritical Is Your LLM judge? Listener-Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**arXiv ID:** 2604.15873 | [PDF](https://arxiv.org/pdf/2604.15873v1)

**作者:** Judith Sieker `[一作]` (Bielefeld University), Sina Zarrieß `[通讯]` (Bielefeld University)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5078051602)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统比较大语言模型在语用评估（听者）和语用生成（说者）两种角色下的表现，探究两者是否相互对应。

**💡 创新点**

创新点在于对同一模型在同一条目上同时进行听者与说者任务，并在条目级别评估两者的关联性，首次揭示大模型语用评估与生成之间的显著不对称性。

**🔧 技术方法**

采用提示工程的零样本推理，使用确定性解码与规则解析评估输出，并构造对应的 speaker 与 listener 提示，统计准确率与条件概率差值。

**📊 数据集**

使用德国“False Scenarios”“False Claims”数据集进行虚假前提任务，德国“fruit‑story”数据集进行反前提任务，以及英文 Deductive Reasoning 任务，分别评估 14 个公开与专有 LLM。

**📈 对比分析**

通过聚合准确率绘制 speaker–listener 散点图，并计算条件概率 Δ_cond 来衡量听者判断是否预测说者成功；结果显示大多数模型在听者角色上更强，尤其开放模型表现差距显著，部分大型专有模型在两角色上相近但仍不完全一致。

**⚠️ 局限性**

局限包括仅覆盖三种语用现象、提示单一、语言多样性不足、未深入内部机制或训练数据、仅使用固定推理设定，未评估不同解码或微调对性能的影响。

---

## 325. Low-Stack HAETAE for Memory-Constrained Microcontrollers

**arXiv ID:** 2604.15868 | [PDF](https://arxiv.org/pdf/2604.15868v1)

**作者:** Gustavo Banegas `[一作]` (LIX, CNRS, Inria, École Polytechnique, Institut Polytechnique de Paris), Vredendaal Christine Van `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对低端微控制器（8–16 KB SRAM）的 HAETAE 低堆栈实现，支持所有三种安全级别（-2、-3、-5），实现了密钥生成、签名与验证的低内存占用。

**💡 创新点**

创新点主要包括：① 拆分签名循环为三阶段 Pass‑A、Pass‑B、Pass‑C，避免大缓冲区同时存活；② 组件级早期拒绝（Component‑level early rejection），在累积模范数时即刻跳过不必要的乘累积；③ 逆序流式 rANS 熵编码，消除 hint 与 high‑bits 的完整缓冲区；④ 双 Pass 超球体采样的全流式 Gaussian 后端；⑤ 视图式解码与行流式矩阵乘法，进一步压缩验证堆栈；⑥ 将 SHAKE 交互式增量哈希取代全缓冲区。

**🔧 技术方法**

采用的技术包括：流水线/重计算策略、逆向 rANS 编码、行/列流式矩阵运算、NTT/FFT 计算、SHAKE 采样与哈希、递归式早期拒绝、两 Pass 超球体采样、全流式 Gaussian 后端、视图式解码、增量 SHAKE 处理、块级压缩等。

**📊 数据集**

在 Nucleo‑L4R5ZI（ARM Cortex‑M4）上做性能基准，使用 RIOT‑OS 在 nRF52840（ARM Cortex‑M4F）与 ESP32‑C6（RISC‑V RV32IMAC）上验证可移植性；对 10,000 次签名进行差分测试以验证编码正确性；对比 ML‑DSA、参考实现与已发表的内存优化实现。

**📈 对比分析**

比较方法：在同一硬件平台、同一编译优化级别下测量堆栈占用、周期数与代码大小；与 ML‑DSA、参考实现、已发表的 HAETAE‑5 内存优化版本对比。结果显示：签名堆栈从 71–141 KB 降至 5.8–6.0 KB（约 92% 下降），验证堆栈从 6220 B 降至 4.8–4.9 KB（约 85% 下降），密钥生成堆栈也有 25–35% 下降；验证速度比 ML‑DSA 快 2.3–3.3×；签名速度比参考实现慢 1.8–3.4×，但内存优势明显。

**⚠️ 局限性**

局限性包括：签名周期仍显著高于参考实现（主要因重计算和多 Pass 结构）；对抗侧信道或故障攻击的安全补丁未实现；在 d>0（-2/3）级别下密钥生成周期相对较慢；某些优化（如行流式乘法）需要额外的重计算，可能在极低功耗场景下产生额外能耗。

---

## 326. DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition

**arXiv ID:** 2604.15866 | [PDF](https://arxiv.org/pdf/2604.15866v1)

**作者:** Siun Kim `[一作]` (Seltasquare), Hyung-Jin Yoon `[通讯]` (Seoul National University)

**通讯引用:** 2423 | [OpenAlex ID](https://openalex.org/A5080473137)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DiZiNER框架，通过多模型零样本标注和协同指令优化实现不需要参数更新的命名实体识别

**💡 创新点**

创新点在于模拟人类先导标注过程，用模型间的争议信息驱动指令迭代改进，实现零样本下的自我提升

**🔧 技术方法**

使用多种异构LLM作为标注者、监督LLM进行争议分析与指令改进、加权多数投票、BIO编码、层级指令重构等技术

**📊 数据集**

在18个NER基准上验证，包括CrossNER(AI/文学/音乐/政治/科学)、CoNLL2003、ACE2005、OntoNotes、MultiNERD、医学、STEM、社交会话等多领域数据集

**📈 对比分析**

与现有零样本与有监督模型对比，DiZiNER在14/18个数据集上取得零样本SOTA，平均提升11.1 F1点，缩小零样本到有监督差距至-20.9 F1，并超过GPT‑5 mini监督者

**⚠️ 局限性**

局限性包括对数据集特定标注规范的漂移风险、固定实体类型模式导致的灵活性不足、对随机抽样与迭代路径的敏感性，需结合少量监督样本以稳固性能

---

## 327. DTEA: A Dual-Topology Elastic Actuator Enabling Real-Time Switching Between Series and Parallel Compliance

**arXiv ID:** 2604.15865 | [PDF](https://arxiv.org/pdf/2604.15865v1)

**作者:** Vishal Ramesh `[一作]`, Shishir Kolathaya `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造并验证了一种能够在运行时实时切换系列弹性（SEA）与并联弹性（PEA）拓扑的双拓扑弹性执行器（DTEA）；

**💡 创新点**

首次实现了使用单一弹性元件通过三辊选择器实现SEA与PEA之间的即时切换，避免了传统需要多种机构或齿轮变换的方案；

**🔧 技术方法**

采用直接驱动BLDC电机、三辊犬齿选择器、径向弹簧舱、线性执行器（电磁铁）以及多传感器闭环控制；

**📊 数据集**

没有使用公开数据集，而是通过实验收集的自建数据：静态刚度曲线、扰动拒绝（冲击试验）、连续切换周期（324 次）以及切换时间（<33.33 ms）等；

**📈 对比分析**

通过对比SEA与PEA两种模式下的开环刚度、阻尼（阻尼率、滞后面积）、扰动峰值偏移、稳态时间、以及在 1 Hz 追踪中的电流消耗，得到：PEA 模式刚度比 SEA 高 2.08×，峰值偏移低 2.26×，稳态时间快 3.45×，切换至 PEA 时电流降低 4.93×；

**⚠️ 局限性**

局限性包括：切换时承受的扭矩受约 1 Nm 限制（需更大力声学元件）；3D 打印结构导致额外结构柔性与摩擦；未在任务级别验证能量节省；实验样本量有限；以及需要更高扭矩、金属机身的后续改进。

---

## 328. Module Lattice Security (Part I): Unconditional Verification of Weber's Conjecture for $k \le 12$

**arXiv ID:** 2604.15858 | [PDF](https://arxiv.org/pdf/2604.15858v1)

**作者:** Ming-Xing Luo `[一作]` (Southwest Jiaotong University), Ming-Xing Luo `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 8784 | [OpenAlex ID](https://openalex.org/A5112554016)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文首次无条件证明了对于k≤12，Weber的类数h_k^+=1，解决了与当前和未来基于晶格的密码标准相关的所有情况的Weber类数猜想。

**💡 创新点**

创新点在于提出了一种结合Fukuda-Komatsu筛法、_2塔中的范数一致性和Herbrand定理的三阶段方法，从而将问题简化为有限的可行计算。

**🔧 技术方法**

使用了Fukuda-Komatsu计算筛法、_2塔的归纳结构和Herbrand定理等技术。

**📊 数据集**

没有具体提到使用的数据集，但提到的计算涉及到的整数和类数的性质。

**📈 对比分析**

与之前依赖于广义黎曼猜想的结果相比，本文提供了k≤12的无条件证明，表明在k≥9的情况下，所有先前的验证都依赖于广义黎曼猜想。

**⚠️ 局限性**

限制在于对于k>12的情况，当前的方法尚未验证，且计算复杂度在k增大时显著增加。

---

## 329. AHS: Adaptive Head Synthesis via Synthetic Data Augmentations

**arXiv ID:** 2604.15857 | [PDF](https://arxiv.org/pdf/2604.15857v1)

**作者:** Taewoong Kang `[一作]` (Korea Advanced Institute of Science and Technology), Jaegul choo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于自适应头部合成（AHS）的零样本头部交换方法，能够在完整上半身图像中实现头部与身体的无缝融合。

**💡 创新点**

创新点在于引入了基于GAGAvatar的合成数据增强和头部正则化（normal map）作为控制信号，并采用跨注意力与自注意力双编码器实现身份与姿态的同时控制。

**🔧 技术方法**

技术上使用Diffusion模型、VAE编码器、DensePose、FLAME 3D头模型、ControlNet、Cross/ Self Attention以及文本与头部特征融合。

**📊 数据集**

训练与评估主要使用SHHQ数据集，并通过人工标注的头部姿态与表情数据。

**📈 对比分析**

与HID、REFace、InstantID等基线对比，AHS在身份相似度、CLIP-I（头部区域）、表达保留和FID（裁剪）上均优于对手，用户研究也显示其在整体质量和头部细节上的领先。

**⚠️ 局限性**

局限性包括对光照变化敏感、极端面部结构差异或遮挡时可能出现几何不匹配与视觉伪影，未来需要提升鲁棒性。

---

## 330. TwoHamsters: Benchmarking Multi-Concept Compositional Unsafety in Text-to-Image Models

**arXiv ID:** 2604.15967 | [PDF](https://arxiv.org/pdf/2604.15967v1)

**作者:** Chaoshuo Zhang `[一作]` (Xi'an Jiaotong University), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 6116 | [OpenAlex ID](https://openalex.org/A5101843177)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文定义了多概念组合不安全性（MCCU）并构建了TwoHamsters基准，随后对10种前沿文本到图像模型、7种主流内容过滤器以及9种概念消除方法进行了系统评估。

**💡 创新点**

创新点包括：①首次从语义层面将MCCU系统化并建立层级分类；②提供了17.5k高质量组合提示，覆盖10类风险维度；③设计了MCCU-ViT/VLM双重评估器，并提出MDR、SCR、NCR三维指标；④揭示了“指令‑安全”困境、税收偏差、注意力缺陷等关键洞察。

**🔧 技术方法**

技术手段主要包括：LLM辅助概念挖掘与语义扩展、CLIP+VLM融合评估、跨模态注意力可视化、Diffusion/Flow-Matching/Autoregressive模型实现、以及基于逻辑约束的MCCU评估器训练。

**📊 数据集**

使用的数据集与模型有：TwoHamsters基准（17.5k提示，51个概念对）；10种SOTA T2I模型（Stable Diffusion、FLUX、PixArt‑α等）；7种安全过滤器（NSFW-T、LLaVA-Guard等）；9种概念消除方法（ESD、UCS、RECE等）；MCCU‑ViT训练集（300k图文对）。

**📈 对比分析**

比较方式：在TwoHamsters上分别计算模型的MDR（MCCU防御率）和SCR（单概念保留率），并对过滤器计算召回率，对消除方法计算MDR、SCR、NCR。结果显示：FLUX在MDR上达99.56%，但大多数过滤器的召回率低于50%；概念消除方法多在MDR与SCR之间权衡，效果往往有限且易导致效用崩溃。

**⚠️ 局限性**

局限性：①MCCU风险高度受社会文化语境影响，难以在开放世界构建通用过滤器；②评估仅在闭合基准上有效，未覆盖持续演化的社交语义；③现有消除技术会破坏基准概念的正向效用；④基准规模虽大，但仍需持续扩充与更新。

---

## 331. Multi-Objective Bayesian Optimization via Adaptive \varepsilon-Constraints Decomposition

**arXiv ID:** 2604.15959 | [PDF](https://arxiv.org/pdf/2604.15959v1)

**作者:** Yaohong Yang `[一作]` (Aalto University), Samuel Kaski `[通讯]` (Aalto University)

**通讯引用:** 15079 | [OpenAlex ID](https://openalex.org/A5018305257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过自适应ε-约束分解实现的多目标贝叶斯优化方法 STAGE-BO，能够在不依赖超体积的情况下实现均匀覆盖 Pareto 前沿。

**💡 创新点**

创新点在于利用几何空洞识别并将其转化为ε-约束，形成一系列单目标子问题，保证了前沿的均匀性，同时兼顾约束和偏好。

**🔧 技术方法**

主要技术包括贝叶斯优化、GP模型、ε-约束分解、受限期望改进（cEI）采集函数、NSGA-II 对采样路径求 Pareto 前沿。

**📊 数据集**

在六个基准上验证，包括 ZDT1/2、DTLZ7、Coil Spring、Rocket Injector、Water Planning、以及受限设计任务（MW7、Disc Brake、Gear Train、CONSTR）和偏好任务（ZDT3、DTLZ2、Vehicle Safety、Car Side Impact）。

**📈 对比分析**

与 qEHVI、qParEGO、JESMO、MESMO、qPOTS、MOBO-OSD、COMBOO 等先进方法对比，STAGE-BO 在 IGD 方面持续领先，超越超体积指标，且在多目标/受限/偏好场景均表现稳健。

**⚠️ 局限性**

局限性包括对测量噪声敏感，噪声可能导致几何空洞估计误差；目前未提供对噪声鲁棒性的完整处理。

---

## 332. A Case Study on the Impact of Anonymization Along the RAG Pipeline

**arXiv ID:** 2604.15958 | [PDF](https://arxiv.org/pdf/2604.15958v1)

**作者:** Andreea-Elena Bodea `[一作]` (Technical University of Munich), Florian Matthes `[通讯]` (Technical University of Munich)

**通讯引用:** 4049 | [OpenAlex ID](https://openalex.org/A5022973212)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估了在检索增强生成（RAG）管道中不同位置（数据库层和答案层）进行匿名化对隐私与实用性的影响，开展了实验对比。

**💡 创新点**

创新点在于首次探究匿名化在RAG不同阶段的放置位置，并提供实证结果显示位置对隐私-实用性权衡的显著影响。

**🔧 技术方法**

采用了非DP方法（PII删除、标签化、合成替换）与DP方法（1-Diffractor、DP-Prompt、DP-MLM），结合LLM（gpt-4o-mini）与向量检索（Pinecone）实现RAG。

**📊 数据集**

使用了BBC新闻、Enron邮件、TAB法律案例三大公开数据集，均含有丰富PII，满足实验需求。

**📈 对比分析**

通过ROUGE-L、余弦相似度、困惑度与LLM-as-a-Judge等指标评估，发现非DP方法在答案层匿名化时保持较高实用性且隐私得分更好，DP方法可调节隐私-实用性权衡但实用性相对下降。

**⚠️ 局限性**

局限在于仅考虑三种数据集与六种匿名化技术，未覆盖更复杂的RAG场景与大模型，且DP方法在实用性上的欠缺可能限制实际部署。

---

## 333. Small Yet Configurable: Unveiling Null Variability in Software

**arXiv ID:** 2604.15957 | [PDF](https://arxiv.org/pdf/2604.15957v1)

**作者:** Xhevahire Tërnava `[一作]` (LTCI, Télécom Paris, Institut Polytechnique de Paris), Mathieu Acher `[通讯]` (Univ Rennes, CNRS, Inria, IRISA)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对GNU coreutils 108个小规模工具程序进行静态分析，量化了它们的编译时与运行时变异性，并研究了变异性与代码/二进制尺寸的关系；同时提出了“null 变异性”概念并给出了相应的定义与特性。

**💡 创新点**

创新点在于：①首次系统评估小型软件的可配置程度；②引入并正式定义了“null 变异性”这一概念；③揭示编译时与运行时变异对代码规模的不同影响，并提出通过早期解决编译时变异或移除无用运行时选项来减小代码基的策略。

**🔧 技术方法**

使用的技术主要是：静态源代码分析（通过 Tree‑sitter 解析预处理指令以计数编译时选项，使用 getopts 等手段计数运行时选项）、程序大小统计（LoC 与可执行体字节数），以及 Pearson / Spearman 相关分析来检验大小与变异性的关系；部分结果通过人工检查与验证。

**📊 数据集**

采用的数据集包括：GNU coreutils 9.1 版本的108个程序；其历史中 85 个版本的 20 个最小程序；以及 Toybox 0.8.8 版与 coreutils 的同类程序进行对比；所有数据与代码已公开于 GitHub（https://github.com/ternava/zero-variability）。

**📈 对比分析**

比较方法：①统计每个程序的编译时/运行时选项数，并与 LoC / 可执行尺寸做相关性分析；②将 coreutils 与 Toybox 的相同工具进行对比，比较它们的代码行数、可执行尺寸、编译时和运行时选项数。结果表明 Toybox 更小、变异更少，显示通过减少变异可有效缩减代码基。性能方面未做运行时性能测评，但代码量和二进制尺寸的减少暗示更高的可维护性与部署效率。

**⚠️ 局限性**

局限性：①仅针对 C 语言实现的小型程序；②变异检测仅限于预处理指令和 getopt 解析，未覆盖外部库或构建脚本中的变异；③未考虑选项间的依赖关系，只统计选项数量；④对“null 变异性”的概念未给出最大规模边界，仅在极小程序中观察到。

---

## 334. Integrating Graphs, Large Language Models, and Agents: Reasoning and Retrieval

**arXiv ID:** 2604.15951 | [PDF](https://arxiv.org/pdf/2604.15951v1)

**作者:** Hamed Jelodar `[一作]` (University of New Brunswick), Ali Ghorbani `[通讯]` (University of New Brunswick)

**通讯引用:** 26602 | [OpenAlex ID](https://openalex.org/A5034685391)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

综述了图结构、LLM与智能代理在推理、检索和多模态任务中的融合方法，构建了一个统一的分类框架；

**💡 创新点**

首次系统地将图模态（知识图、场景图、因果图等）、融合策略（提示、增广、训练、代理）和应用目标（推理、检索、生成、推荐）三维进行聚类，并给出针对不同场景的最佳实践建议；

**🔧 技术方法**

整合Transformer‑LLM、图神经网络（GNN）、检索增强生成（RAG）、图增强提示（GraphRAG）、多代理框架与结构化提示等技术；

**📊 数据集**

引用并对比了多个公开基准（如WebQSP、Complex Web Questions、Finance KG、Medical KG、Scene Graph datasets等）以及行业案例（网络安全日志、医疗电子病历、材料科学数据库、机器人视觉等）；

**📈 对比分析**

在多项实验中展示了GraphRAG相较于传统RAG提升10–30% 的准确率、LLM+GNN混合模型在推荐、漏洞检测、知识问答任务中比单一模型提升5–15% 的 F1/Recall；

**⚠️ 局限性**

主要限制包括：图-LLM 对齐难题、可扩展性不足（大图检索与消息传递开销高）、评价指标碎片化（缺乏统一的多步骤推理评测）、以及对稀疏/不完整图的鲁棒性不足。

---

## 335. Modeling Sparse and Bursty Vulnerability Sightings: Forecasting Under Data Constraints

**arXiv ID:** 2604.16038 | [PDF](https://arxiv.org/pdf/2604.16038v1)

**作者:** Cedric Bonhomme `[一作]`, Alexandre Dulaunoy `[通讯]` (Computer Incident Response Center Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究预测单个漏洞的观测数量，尝试 SARIMAX、Poisson 回归、指数衰减、逻辑斯蒂曲线等多种统计模型，并将 VLAI 严重度作为外生变量引入。

**💡 创新点**

创新点在于针对稀疏、爆发式事件的漏洞观测，提出计数型预测模型并设计自适应模型选择策略；同时首次将 NLP 生成的严重度分数与时间序列预测相结合。

**🔧 技术方法**

使用的技术包括 SARIMAX、Poisson/负二项回归、指数衰减、逻辑斯蒂曲线拟合、Python curve_fit、log(x+1) 变换等统计与曲线拟合方法。

**📊 数据集**

使用的数据集为每日/每周漏洞观测计数（来源于漏洞数据库、Fediverse、Shadowserver 等）以及 VLAI 生成的严重度分数。

**📈 对比分析**

通过与 SARIMAX 的对比，发现 SARIMAX 在短序列下不稳定且区间宽广；Poisson 在小样本下更稳健；指数衰减与逻辑斯蒂曲线在不同阶段表现更好，整体提示短期预测宜采用简单计数模型。

**⚠️ 局限性**

局限性包括数据稀疏、时间窗口短、突发离群导致过拟合、过度/不足分散、外生严重度变化有限，以及模型选择仍需人工或自适应策略。

---

## 336. Stochasticity in Tokenisation Improves Robustness

**arXiv ID:** 2604.16037 | [PDF](https://arxiv.org/pdf/2604.16037v1)

**作者:** Sophie Steger `[一作]` (Graz University of Technology), Martin Trapp `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 642 | [OpenAlex ID](https://openalex.org/A5060407332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在大语言模型训练与推理中使用随机（非确定性）分词方式，以提升模型对分词扰动的鲁棒性。

**💡 创新点**

提出两种更均匀、覆盖更完整的随机分词采样策略（Stok-uni、Uni-k）并证明其在对抗性与随机扰动下优于现有的BPE-Dropout和随机拆分方法。

**🔧 技术方法**

技术包括：基于BPE的随机拆分、Dirichlet-多项式采样、基于多值决策图（MDD）的全局均匀采样、在预训练、微调及少量示例学习（ICL）中引入随机分词，以及对抗性攻击评估与Lipschitz‑理论分析。

**📊 数据集**

使用问答类基准数据集：LangGame、ARC‑E、COPA、CSQA、HellaSwag、PIQA、SocialIQA 以及 TokSuite；预训练在 OpenWebText 上进行，微调使用 Llama‑1b。

**📈 对比分析**

通过在不同训练阶段（预训练、微调、ICL）对比使用标准（canonical）与随机分词的模型，在随机扰动（多拆分）和对抗性分词攻击下测量准确率；结果显示随机分词可将 canonical 模型 29.8% 的准确率下降恢复到接近原值，且在微调阶段即能显著提升鲁棒性。

**⚠️ 局限性**

局限性：仅在多项选择问答任务上评估，未覆盖生成任务；对抗性攻击空间仅局限于编辑距离约束；随机分词的计算成本在极大模型中仍有提升空间；理论分析基于简化假设，缺乏对全模型的完整鲁棒性证明。

---

## 337. Neurosymbolic Repo-level Code Localization

**arXiv ID:** 2604.16021 | [PDF](https://arxiv.org/pdf/2604.16021v1)

**作者:** Xiufeng Xu `[一作]` (Nanyang Technological University), Yi Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 12565 | [OpenAlex ID](https://openalex.org/A5100730622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种神经符号框架LogicLoc，用大模型生成Datalog查询来实现仓库级代码定位，并构建了无关键词逻辑代码定位(KA-LCL)诊断基准；

**💡 创新点**

创新点在于识别并解决“关键词捷径”偏差，创建完全无关键词的逻辑查询基准，以及将LLM与Deterministic Datalog引擎结合，利用程序事实(IR)实现结构化推理与可验证定位；

**🔧 技术方法**

使用大语言模型（如GPT-4o、Claude-3.5-Sonnet）生成Datalog程序，Soufflé推理引擎执行逻辑推断，程序事实抽取与中间规则诊断（Mutation）技术；

**📊 数据集**

使用基准数据集LogicLoc（9个Python项目，225条纯逻辑查询，25条代表性组合），以及SWE‑bench Lite（274个issue实例）和空真值基准（EmptyGroundTruth）进行评估；

**📈 对比分析**

与四类SOTA基线（Embedding、Pipeline、Agent）比较，LogicLoc在KA‑LCL任务中显著提升准确率、召回率、精确率与AJS；在SWE‑bench Lite任务中保持竞争力，且token消耗更低、执行更快；

**⚠️ 局限性**

局限性包括仅在Python上验证，扩展到其他语言需进一步工作；程序事实抽取依赖前端解析，可能遗漏某些语言特性；缺乏对模型泛化到极端结构复杂度的深入评估。

---

## 338. Breakout-picker: Reducing false positives in deep learning-based borehole breakout characterization from acoustic image logs

**arXiv ID:** 2604.16011 | [PDF](https://arxiv.org/pdf/2604.16011v1)

**作者:** Guangyu Wang `[一作]` (University of Science and Technology of China), Xinming Wu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6501 | [OpenAlex ID](https://openalex.org/A5058640298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 Breakout-picker 的深度学习框架，用于自动识别声学图像日志中的井壁破裂（breakout）并显著降低误报率，进而提升现场应力分析的可靠性。

**💡 创新点**

创新点包括：① 在训练阶段加入“硬负样本”（如关键座、裂隙和记录工件）以提升模型对破裂与非破裂特征的判别能力；② 结合物理预期的破裂对称性规则，对候选破裂参数进行后处理校验，进一步消除误报；③ 在网络架构上采用深度LabV3+的轻量化版本（ResNet‑18）并实现圆形卷积，保证沿方位的周期性特征被充分利用。

**🔧 技术方法**

技术：使用双通道输入（声学幅值 + 孔径），基于 DeepLabV3+ 的 encoder‑decoder 结构进行像素级分割；采用类别平衡的二元交叉熵损失；后处理阶段基于近 180° 方位对称性进行阈值筛选；数据增强包括深度翻转、方位平移、随机裁剪与放大。

**📊 数据集**

数据集：以瑞士 BedrettoLab 的八根钻孔声学图像日志为主进行训练与验证，并使用另外两根外域井（加拿大 Hunt Well、IODP 1256D）进行跨域泛化测试；同时公开了标注好的破裂样本集和代码。

**📈 对比分析**

与传统峰值检测和 MMDC‑UNet 的对比显示：在 BedrettoLab 测试井上，Breakout-picker 的 IoU 最高（0.72），误报率从 48% 降至 4%（但漏报率上升）；在外域井中，误报率降低 9–18%，方位误差降低 2–5°，宽度误差也显著改善。整体性能在保持分割精度的同时大幅减少误报。

**⚠️ 局限性**

局限性：① 负样本主要来自同一数据集，无法覆盖所有潜在的误报来源，导致跨域时误报率提升；② 对称性校验虽能显著抑制误报，却会错误排除一侧单向破裂，导致漏报率显著上升；③ 在低分辨率或噪声较大的日志中，模型仍会出现部分残留误报。

---

## 339. MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition

**arXiv ID:** 2604.16009 | [PDF](https://arxiv.org/pdf/2604.16009v1)

**作者:** Farhad Abtahi `[一作]` (Karolinska Institutet), Fernando Seoane `[通讯]` (Karolinska Institutet)

**通讯引用:** 2751 | [OpenAlex ID](https://openalex.org/A5029441655)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出MEDLEY-BENCH，一个三步协议的行为元认知基准，用来区分独立推理、私自修正与社会影响修正。

**💡 创新点**

创新点在于将元认知拆分为独立、私自、社会三阶段，并利用真实模型间的自然不一致作为社会压力，进一步加入对抗实验揭示模型的社交修正偏好。

**🔧 技术方法**

技术上采用结构化JSON生成、对抗实验、三层级评分（T1、T2、T3）以及基于DeepMind四项元认知子能力的评判器，并用ipsative归一化消除整体阐释偏差。

**📊 数据集**

数据集为130个跨五个推理领域（医学诊断、系统排错、代码审查、架构设计、统计推理）的实例，每个实例包含5个关键声明，并收集8个分析师模型的多样化输出。

**📈 对比分析**

与传统准确率基准对比，采用MMS和MAS得分生成排行榜；MMS得分差距为12.8点，MMS与MAS相关性ρ≈0.94，模型规模对评估（T1）提升显著，但对控制（T2）无一致提升，展示了评估–控制解耦。

**⚠️ 局限性**

局限性包括只评估提示下的修正而非自发监测，IPS评分受主导因子影响，缺乏人类基线，JSON格式要求可能限制部分模型表现，以及温度设为0导致结果不代表采样行为。

---

## 340. Combining Convolution and Delay Learning in Recurrent Spiking Neural Networks

**arXiv ID:** 2604.15997 | [PDF](https://arxiv.org/pdf/2604.15997v1)

**作者:** Lúcio Folly Sanches Zebendo `[一作]` (University of Padova), Michele Rossi `[通讯]` (University of Padova)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种在递归尖峰神经网络中使用卷积递归连接并保留可学习轴突延迟的架构，替代传统全连接递归层。

**💡 创新点**

创新点在于将全连接递归权重替换为轻量级1D卷积核（k=3），显著降低参数量（≈99%），同时通过可学习延迟保持长时依赖的建模能力。

**🔧 技术方法**

使用尖峰神经网络（LIF神经元）与可学习延迟机制（Triangular spread函数）、Surrogate Gradient（正切或三角函数）训练，结合1D卷积递归层实现。

**📊 数据集**

在两套音频尖峰数据集上评估：Spiking Heidelberg Digits（SHD）和Spiking Speech Commands（SSC）。

**📈 对比分析**

与原DelRec模型直接对比：在SHD上准确率相近（91.5% vs 91.7%），参数从每层65k降至3；推理时间提升52倍。SSC上略低（78.6% vs 82.6%）。

**⚠️ 局限性**

局限在于SSC上性能略下降，且依赖于数据的局部相关性；在更复杂或无明显局部结构的数据集上效果未知。

---

## 341. MMGait: Towards Multi-Modal Gait Recognition

**arXiv ID:** 2604.15979 | [PDF](https://arxiv.org/pdf/2604.15979v1)

**作者:** Chenye Wang `[一作]` (Beijing Normal University), Yongzhen Huang `[通讯]` (Beijing Normal University)

**通讯引用:** 5709 | [OpenAlex ID](https://openalex.org/A5034269600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了MMGait多模态步态识别基准，涵盖725名受试者、334,060条序列，12种模态（RGB、IR、深度、LiDAR、4D雷达等），并提出了OmniGait统一模型实现单模、跨模与多模步态识别；

**💡 创新点**

创新点在于将五种异构传感器生成的十二种模态系统化评估，并首次提出全模态识别任务与统一基线，突破了传统单一模态或跨模局限；

**🔧 技术方法**

技术方案包括模态特定编码器、跨模融合模块、共享残差网络及三元组+交叉熵联合训练，支持图像、姿态、深度、点云等多种输入；

**📊 数据集**

主要使用新建的MMGait数据集，对比CASIA、TUM-GAID、FreeGait等现有步态数据集；

**📈 对比分析**

通过Rank‑1与mAP指标与单模、跨模、基线模型对比，OmniGait在多数模态下接近或优于专用模型，单模最高可达99%+；

**⚠️ 局限性**

局限性包括对服装变化(CL)等极端条件的鲁棒性仍不足，雷达模态稀疏导致识别性能低，且未进一步探索更深层次的多模融合与端到端训练。

---

## 342. Where Does MEV Really Come From? Revisiting CEXDEX Arbitrage on Ethereum

**arXiv ID:** 2604.15973 | [PDF](https://arxiv.org/pdf/2604.15973v1)

**作者:** Bence Ladóczk `[一作]`, János Tapolcai `[通讯]` (Budapest University of Technology and Economics)

**通讯引用:** 2948 | [OpenAlex ID](https://openalex.org/A5062012037)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并实现了一个包含跳跃与扩散的离散时间AMM模型，用于更准确地描述中心化交易所（CEX）与去中心化交易所（DEX）之间的价格差异和MEV套利；

**💡 创新点**

通过引入跳跃项并推导其稳态分布（ergodic Markov链），解决了传统Black–Scholes模型忽略价格跳跃导致的套利收益低估问题；

**🔧 技术方法**

使用离散时间随机微分方程、函数迭代求稳态分布、数值积分以及C++实现的模拟与拟合；

**📊 数据集**

以Uniswap V2 ETH–USDT（和USDC）池的交易记录和多家CEX（Binance、KuCoin、Gate.io）的一秒级价格数据为主要数据集；

**📈 对比分析**

将理论估计的套利体量、交易量和利润与链上观测到的单笔交易数据进行对比，发现模型预测的套利收益与MEV近似一致，误差在一倍左右，验证了模型的有效性；

**⚠️ 局限性**

主要局限在于对CEX买卖价差的估计依赖经验参数、模型假设套利者总是以最优价执行、未考虑网络延迟与交易费用分布等现实因素，且对跳跃阈值选择敏感。

---

## 343. LLMSniffer: Detecting LLM-Generated Code via GraphCodeBERT and Supervised Contrastive Learning

**arXiv ID:** 2604.16058 | [PDF](https://arxiv.org/pdf/2604.16058v1)

**作者:** Mahir Labib Dihan `[一作]` (Bangladesh University of Engineering and Technology), Abir Muhtasim `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种名为LLMSniffer的二分类框架，用于区分人类编写与LLM生成的代码片段。

**💡 创新点**

核心创新在于将GraphCodeBERT与监督式对比学习、去除注释预处理以及MLP分类头三者结合，显著提升检测精度。

**🔧 技术方法**

使用GraphCodeBERT作为编码器，配合两阶段监督对比学习与MLP分类器，并在输入前移除代码注释。

**📊 数据集**

评估基于GPTSniffer（多语言对话生成代码）和Whodunit（GPT‑4 Python竞赛解答）两个公开基准数据集。

**📈 对比分析**

与传统基线（CodeBERT+交叉熵、Whodunit的XGBoost）相比，LLMSniffer在GPTSniffer上准确率从70%提升至78%，F1从68%提升至78%；在Whodunit上准确率从91%提升至94.65%，F1从91%提升至94.64%。

**⚠️ 局限性**

局限包括只能处理长度不超过512 tokens 的片段、对新域或不同LLM生成模式的迁移性不足，以及对抗性攻击（如代码混淆）下的鲁棒性待验证。

---

## 344. Evaluating SYCL as a Unified Programming Model for Heterogeneous Systems

**arXiv ID:** 2604.16043 | [PDF](https://arxiv.org/pdf/2604.16043v1)

**作者:** Ami Marowka `[一作]` `[通讯]` (Parallel Research Lab), Ami Marowka (Parallel Research Lab)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统的基准测试与案例分析，评估了 SYCL 在跨平台编程中的单一性（singularity），重点比较了 Unified Shared Memory（USM）与 Buffer-Accessor 两种内存管理模型，以及 NDRange 与 Hierarchical 两种并行模型在 CPU、GPU（Intel、AMD、NVIDIA）上的性能、可移植性与生产力表现；

**💡 创新点**

创新点在于提出“单一性”评价框架，综合考虑可移植性、生产力与性能三维度，系统性地量化了不同 SYCL 抽象组合的性能差异，并结合多种实现（DPC++、hipSYCL、AdaptiveCpp）与多种后端（OpenCL、LevelZero、CUDA、HIP）给出一致性评估；

**🔧 技术方法**

使用了 SYCL 2020 标准下的 DPC++、hipSYCL、AdaptiveCpp 编译器；实现了 NDRange 与 Hierarchical 内核、USM 与 Buffer-Accessor 数据管理；利用 LevelZero、OpenCL、CUDA、HIP 后端进行实际执行；并借助 Intel oneAPI、AMD ROCm、NVIDIA CUDA 工具链进行对比；

**📊 数据集**

基准工作负载包括向量加法、矩阵乘法（矩阵尺寸 1024~8192）、归约、DGEMM、PLSSVM 等，数据集规模从 512 MB 到 2048 MB 及 1024×1024~8192×8192 的浮点矩阵；

**📈 对比分析**

通过平均 50 次运行、10 次预热，使用相同代码在不同后端和硬件上执行，比较了两种内存模型与两种内核模型的运行时；结果显示：buffer-accessor 明显优于 USM（CPU 上可达 60×，GPU 上 4×），NDRange 明显优于 Hierarchical（GPU 上 2.2×，CPU 上 1.5×），且不同编译器/实现之间性能波动可达数十倍；

**⚠️ 局限性**

主要限制包括：性能高度依赖抽象选择、后端与编译器差异导致的不可预测性；USM 在隐式迁移时性能低下且缺乏硬件兼容性；缺乏严格的语义保证导致需手动同步或测试；工具链碎片化、编译器/运行时不稳定，无法实现真正的“写一次、运行任何地方”目标；

---

## 345. Where does output diversity collapse in post-training?

**arXiv ID:** 2604.16027 | [PDF](https://arxiv.org/pdf/2604.16027v1)

**作者:** Constantinos Karouzos `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**通讯引用:** 2858 | [OpenAlex ID](https://openalex.org/A5010341007)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析大语言模型在后训练过程中输出多样性崩塌的原因与机制，比较三条后训练路径（Think、Instruct、RL-Zero）在15项任务中的多样性与质量变化

**💡 创新点**

发现输出多样性崩塌主要由训练数据构成决定，而非单一训练方法；CoT生成格式并非导致崩塌的根源；通过把DPO的反向KL消除与RL的无KL训练相结合，可在一定程度上缓解语义多样性下降

**🔧 技术方法**

采用SFT、DPO、RL（GRPO）三阶段后训练；使用SBERT、EAD、Vendi Score、NLI等多维度多样性指标；对6项可验证任务进行质量-多样性分解；利用LLM-as-judge评估非可验证任务

**📊 数据集**

以Olmo 3 7B模型为基础，分别在Think（合成两位教师CoT数据+Delta学习+RL），Instruct（多源数据+去除CoT+RL），RL‑Zero（直接RL，四种奖励领域）进行实验；评估任务覆盖摘要、代码、推理、指令、创作与价值多元化等15项基准

**📈 对比分析**

对比基线（Base）、中间阶段与最终模型，量化多样性衰减比例（例如Think‑SFT 62%下降，Instruct‑DPO 23%下降）；在可验证任务上评估准确率、majority‑vote收益；在非可验证任务上使用LLM-as-judge获得win率提升；结果显示后训练模型虽提升质量，但多样性显著受限，RL‑Zero在保持多样性方面优于其他线索但质量相对更低

**⚠️ 局限性**

实验局限在于仅覆盖7B规模Olmo 3模型，未考察更大规模或不同体系结构的泛化；多样性指标主要统计分布特征，未直接测量表述多样性或价值立场多元化；实验设计未直接验证不同教师数量或来源的定量影响，需进一步细化数据构成介入

---

## 346. AstroVLM: Expert Multi-agent Collaborative Reasoning for Astronomical Imaging Quality Diagnosis

**arXiv ID:** 2604.16024 | [PDF](https://arxiv.org/pdf/2604.16024v1)

**作者:** Yaohui Han `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6730 | [OpenAlex ID](https://openalex.org/A5062800747)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AstroVLM 框架，利用多智能体协作、检索增强生成 (ASK‑RAG) 与链式回溯推理 (RwB) 对天文成像质量进行精准诊断。

**💡 创新点**

创新点包括：①为每个子任务构建 Agent‑Specific Knowledge RAG，显著减少检索噪声；②引入 Chain‑of‑Backtracking 与 Collaborative Reasoning Tree，提升错误定位的准确性与解释性。

**🔧 技术方法**

使用技术包括：多智能体协作框架 AstroSight、KeyBert 关键词抽取、图神经网络 (GCN) 对知识图谱进行聚合、检索增强生成、CoT/CoB 推理、外部工具调用及 CRT 树结构。

**📊 数据集**

使用真实天文图像数据集，来源于 AstroBin 与 iStarShooter，按星系、星云、星团三大类进行标注。

**📈 对比分析**

通过与 GPT‑4o、Claude Sonnet 4 等 VLM 基线以及 GraphRAG、RAG‑Fusion、LightRAG 等 RAG 方法进行公平对比，评估指标为合理性、准确率与多样性；AstroVLM 在所有指标上均超过基线，提升幅度在 5.9%–11.8% 之间，ASK‑RAG 相比 GraphRAG 提升 18.4%。

**⚠️ 局限性**

局限性：对大规模算力与外部工具的依赖较高，超参数敏感，且在低算力或无标注数据环境下可推广性受限。

---

## 347. Corner Reflector Array Jamming Discrimination Using Multi-Dimensional Micro-Motion Features with Frequency Agile Radar

**arXiv ID:** 2604.16008 | [PDF](https://arxiv.org/pdf/2604.16008v1)

**作者:** Jie Yuan `[一作]` (Tsinghua University), Yimin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 4695 | [OpenAlex ID](https://openalex.org/A5100728989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一套基于频率机动雷达的船舶与角反射阵列干扰区分方法。

**💡 创新点**

创新点在于提出两种新型多维微动特征——均值加权残差（MWR）和互补对比因子（CCF），并将其与CNN提取的深度特征融合，最终使用XGBoost实现高精度分类。

**🔧 技术方法**

主要技术包括频率机动雷达回波建模、范围-速度图生成、手工特征提取（MWR、CCF）、轻量级CNN深度特征学习以及XGBoost分类器。

**📊 数据集**

使用仿真数据集：一艘约144 m长的船舶和由4个直线排列的角反射阵列，分别在0.1 m、1 m、2 m三种海浪高度以及17个方位角（10°–170°）下生成100份范围-速度图，共计约1700张图像。

**📈 对比分析**

与仅使用手工特征、仅使用CNN特征以及仅使用XGBoost进行比较，融合特征+XGBoost在不同波高下均实现最高准确率，最高可达96%（波高2 m时），手工特征单独也能达到86%以上，表明融合方案在性能上显著优于单一特征。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，真实雷达数据的适用性未知；在某些波高或角度下角反射阵列可能表现出近似刚体特征，导致误分类；方法对雷达参数（频点数、脉冲压缩等）敏感，需进一步泛化。

---

## 348. AgentV-RL: Scaling Reward Modeling with Agentic Verifier

**arXiv ID:** 2604.16004 | [PDF](https://arxiv.org/pdf/2604.16004v1)

**作者:** Jiazheng Zhang `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 17003 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agentic Verifier 框架，利用正向和反向两名代理进行多轮、工具增强的推理验证；

**💡 创新点**

创新点在于将奖励模型转化为双向、多轮的代理式验证流程，并通过 AgentV-RL 将多代理能力压缩为单一 LLM；

**🔧 技术方法**

核心技术包括多轮推理、工具调用（如 Python 解释器）、Synthetic Trajectory 生成、两阶段训练（SFT+RL）以及 Group Relative Policy Optimization；

**📊 数据集**

使用的主要数据集包括 Polaris、DeepScaleR-40K、AReaL-boba-106k、GSM8K、MATH500、Gaokao2023、AIME24、LiveCodeBench 与 HotpotQA；

**📈 对比分析**

通过与现有 ORM、PRM、GenRM 等基线在 BoN 采样与迭代修订两种测试时缩放（TTS）场景下的性能比较，Agentic Verifier 在所有基准上均实现了显著提升，4B 版本在 MATH500 上比 INF-ORM-Llama3.1-70B 提升 25.2%；

**⚠️ 局限性**

局限性包括：依赖合成、工具增强的数据可能不完全覆盖真实场景；多轮推理导致计算成本和延迟显著增加；性能高度依赖外部工具的覆盖率和可靠性。

---

## 349. SCHK-HTC: Sibling Contrastive Learning with Hierarchical Knowledge-Aware Prompt Tuning for Hierarchical Text Classification

**arXiv ID:** 2604.15998 | [PDF](https://arxiv.org/pdf/2604.15998v1)

**作者:** Ke Xiong `[一作]`, Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1946 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SCHK-HTC框架，用于在少量样本条件下进行层级文本分类，结合知识图谱信息与同级对比学习来提升模型对同级标签的细粒度辨别能力。

**💡 创新点**

核心创新在于：①基于知识图谱的层级知识感知提示调优（Hierarchical Knowledge-aware Prompt Tuning），通过实体链接和Node2Vec提取层级结构知识；②针对同级标签的双模板对比学习（Sibling Contrastive Learning），利用语义相似度进行硬负样本挖掘，显著提升同级标签区分。

**🔧 技术方法**

技术手段包括：BERT-base作为编码器；知识图谱抽取与Node2Vec+随机邻居聚合；提示调优与多模板Prompt；InfoNCE与自定义Sibling Contrastive Loss；多任务联合训练（MLM、分类、对比）。

**📊 数据集**

实验数据集：单路径的WOS和DBpedia，以及多路径的RCV1‑V2。

**📈 对比分析**

与Vanilla‑BERT、HGCLR、HPT、HierVerb、DCL等基线在k-shot（1/2/4/8/16）设置下对比，SCHK-HTC在大多数情形下取得最高Micro‑F1/Macro‑F1，尤其在WOS和DBpedia的单路径任务中表现最突出。

**⚠️ 局限性**

局限性：在多路径任务（如RCV1‑V2）随着shot数增加提升有限；对知识图谱的依赖导致实体链接与图构建成本；对极低shot场景下的对比学习可能仍受样本稀缺限制。

---

## 350. ReactBench: A Benchmark for Topological Reasoning in MLLMs on Chemical Reaction Diagrams

**arXiv ID:** 2604.15994 | [PDF](https://arxiv.org/pdf/2604.15994v1)

**作者:** Qiang Xu `[一作]` (International Digital Economy Academy), Yu Li `[通讯]` (International Digital Economy Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReactBench基准，用以评估多模态大型语言模型对化学反应图的结构推理能力。

**💡 创新点**

首次通过层级任务拆分揭示MLLM在全局拓扑推理方面的显著不足，并证明感知与推理瓶颈不等同。

**🔧 技术方法**

使用多模态LLM评估框架、对比实验、链式思考与外部知识消融等技术。

**📊 数据集**

构建了1,618个专家标注的问答对，来源于学术期刊和专利的真实化学反应图。

**📈 对比分析**

对17个公开和API模型进行准确率对比，发现anchor‑based任务准确率可达90%，而全局推理任务仅低于56%，差距超过30%。

**⚠️ 局限性**

仅关注化学反应图，未扩展到其他图形类型，且评测依赖严格的exact‑match，可能低估语义正确性。

---

## 351. Radio Environment Map for Energy-Efficient User-Centric Cell-Free M-MIMO Network

**arXiv ID:** 2604.15987 | [PDF](https://arxiv.org/pdf/2604.15987v1)

**作者:** Marcin Hoffmann `[一作]` (Poznan University of Technology), Paweł Kryszkiewicz `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了用于用户中心无小区（UCCF）大规模MIMO网络的无线环境图（REM），利用REM映射用户位置模式与不同接入点（AP）集群配置的能效（EE）关系，并基于此指导能效最优的接入点集群选择。

**💡 创新点**

创新点在于：①将REM引入UCCF M‑MIMO网络，实现对用户位置与硬件（PA）特性的联合建模；②在REM中加入PA非理想特性，揭示硬件对EE与集群配置的影响；③提出基于REM的强化学习（RL）框架，用于动态选择EE最优的集群配置。

**🔧 技术方法**

使用技术包括：3D射线跟踪（3D‑RT）系统级仿真、软限制器模型实现PA非线性、零迫预编码、调制编码分配、强化学习框架（状态＝UE位置模式，动作＝集群配置，奖励＝EE）。

**📊 数据集**

数据集为仿真生成的两组随机UE位置模式（每组40个UE）与三种PA模型（Class A、Class B、理想PA）对应的REM条目；所有仿真参数均基于城市宏小区环境，宏AP 128天线、46 dBm，微AP 32天线、30 dBm。

**📈 对比分析**

比较方法：对同一UE位置模式下，分别在不同PA模型下评估单AP（state‑of‑the‑art）与多AP（根据REM选出的EE最高配置）两种集群方案的EE和UE吞吐量。结果显示，在理想PA下，使用3AP可提升EE约19%并使吞吐量中位数提升约45%；在Class A或Class B PA下，多AP方案对EE无益，甚至劣于单AP方案。

**⚠️ 局限性**

局限性包括：①实验仅基于两组随机位置模式，未覆盖多样化场景；②仅在仿真环境下验证，缺乏实际部署验证；③PA模型仅为三种理想化模型，未考虑更复杂的硬件特性；④强化学习框架未实现完整训练与部署，仅作概念验证。

---

## 352. Impact of Nonlinear Power Amplifier on Massive MIMO: Machine Learning Prediction Under Realistic Radio Channel

**arXiv ID:** 2604.15977 | [PDF](https://arxiv.org/pdf/2604.15977v1)

**作者:** Marcin Hoffmann `[一作]` (Poznan University of Technology), Paweł Kryszkiewicz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了大规模MIMO OFDM系统中功率放大器非线性失真对用户信噪失真比(SDR)的影响，并提出基于3D射线追踪的统计与机器学习预测模型。

**💡 创新点**

提出用GEV分布建模受损用户的SDR，以及将特征矩阵映射到SDR的VGG16 CNN，用以实现功率放大器自适应的每用户功率分配。

**🔧 技术方法**

使用Bussgang理论、Rayleigh/LoS解析、3D射线追踪仿真、VGG16 CNN、模型剪枝等技术。

**📊 数据集**

通过基于METIS Madrid Grid 3D-RT生成的多站点环境，采集的3542个UE位置、100个OFDM符号、四个IBO设置的特征矩阵与SDR标签。

**📈 对比分析**

与固定IBO 6 dB、Tavares动态IBO和理论Rayleigh/LoS模型比较，VGG16实现的自适应分配平均提升约12 %用户吞吐量，最高可达30 %+。

**⚠️ 局限性**

主要限制是模型依赖于特定天线阵列和PA特性，需在实际部署前重新训练；并且对极端低IBO场景预测误差可能导致性能下降。

---

## 353. Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs

**arXiv ID:** 2604.16060 | [PDF](https://arxiv.org/pdf/2604.16060v1)

**作者:** Sai Srinivas Kancheti `[一作]` (Indian Institute Of Technology), Tanuja Ganu `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对17个多模态推理模型在13个空间推理基准上的性能进行系统评估，探讨Chain‑of‑Thought（CoT）提示在空间推理中的效果；

**💡 创新点**

首次系统性发现CoT提示在空间任务中普遍导致性能下降，并通过No‑Image++消融揭示模型过度依赖文本先验、会在缺图像时hallucinate视觉信息，强调需要以视觉为中心的推理范式；

**🔧 技术方法**

使用统一评估流程（VLMEvalKit）与vLLM推理；对比CoT与非CoT提示；采用No‑Image和No‑Image++消融实验；利用LLM‑as‑judge进行生成评分；

**📊 数据集**

13个空间推理数据集：静态2D集（BLINK、CV‑Bench2D、MMVP、RealWorldQA、SpatialBench、VSR、V*Bench）和3D/动态集（3DSRBench、CV‑Bench3D、MindCube、MMSIBench、OmniSpatial、SAT‑Real）等；

**📈 对比分析**

采用统一的基准、统一的提示与评分方式，对比CoT vs 非CoT、MRM vs 基础模型、专有模型表现；结果显示大多数MRM在CoT提示下性能下降，7/8 MRM未能超过其基础模型；No‑Image++实验显示模型在无图像时仍做出错误推理，表明对文本先验的过度依赖；

**⚠️ 局限性**

仅覆盖13个基准，未完全覆盖空间推理领域；难以完全隔离所有混杂因素；专有模型训练细节不透明，限制进一步深入分析。

---

## 354. Ranking XAI Methods for Head and Neck Cancer Outcome Prediction

**arXiv ID:** 2604.16034 | [PDF](https://arxiv.org/pdf/2604.16034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 355. AST: Adaptive, Seamless, and Training-Free Precise Speech Editing

**arXiv ID:** 2604.16056 | [PDF](https://arxiv.org/pdf/2604.16056v1)

**作者:** Sihan Lv `[一作]` (Zhejiang University), Meng Xi `[通讯]` (Zhejiang University)

**通讯引用:** 3780 | [OpenAlex ID](https://openalex.org/A5068256578)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了AST框架，利用预训练AM‑FM TTS的潜在重组和自适应弱事实引导实现训练免费、精准的语音编辑。

**💡 创新点**

创新点在于通过潜在重组保持未编辑区域的音色与时序，并引入AWFG在潜在空间动态引导，解决了质量与可控性权衡，同时支持局部风格编辑。

**🔧 技术方法**

技术包括逆ODE潜在映射、LCS对齐、AWFG、AM‑FM（Diffusion Transformer）TTS模型、Word‑level DTW评估等。

**📊 数据集**

使用公开的LibriSpeech‑Edit（2000句）数据集，并与RealEdit等基准进行对比。

**📈 对比分析**

在WER、DNSMOS、SpkSim、WDTW四项指标上与SSR‑Speech、Step‑Audio‑EditX、IndexTTS‑2等基线比较，AST在SpkSim与WDTW上取得最高分，WER仅略逊于部分基线但明显提升，整体实现训练免费且性能领先。

**⚠️ 局限性**

局限性包括对极端长或复杂编辑的稳定性待验证，对AWFG中λ的低取值敏感，以及对预训练TTS质量的依赖，缺乏系统的人类主观评估。

---

## 356. Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs

**arXiv ID:** 2604.16054 | [PDF](https://arxiv.org/pdf/2604.16054v1)

**作者:** Rohit Sinha `[一作]` (Indian Institute of Technology Hyderabad), Tanuja Ganu `[通讯]` (Microsoft Research)

**通讯引用:** 676 | [OpenAlex ID](https://openalex.org/A5024685185)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Mind's Eye 视觉认知基准，包含八个多模态任务，以评估 MLLM 在视觉推理上的能力；

**💡 创新点**

创新点在于将任务划分为 Abstraction–Relation–Transformation（ART）三维结构，使用可参数化的合成生成和针对性扰动器，提供系统化、可诊断的视觉认知评估；

**🔧 技术方法**

采用程序化 SVG 生成、多模态 LLM（如 GPT‑4o、LLaVA、Qwen 等）、多种提示策略（CoT、Meta‑Task 等）以及注意力分析与 LLM 自动评测管道；

**📊 数据集**

使用自己构建的 Mind's Eye 数据集，包含 8 任务共 800 条诊断样本（可扩展至 20,000 条），实现无人工标注的可扩展评估；

**📈 对比分析**

与 18 种 MLLM 及人类参与者对比；人类平均准确率 80%，顶级 MLLM 低于 50%，在 Transformation 与 Abstraction 任务上表现尤为差，提示策略对 Abstraction 有正面影响但对 Transformation 无效；

**⚠️ 局限性**

局限性包括仅采用多选形式、仅基于 2D 合成图像、非专业单语种人类基线、可能导致过度拟人化、以及 leaderboards 可能促成狭窄优化。

---

## 357. Optimization of Sparse VLSF Codes for Short-Packet Transmission via Saddlepoint Methods

**arXiv ID:** 2604.16049 | [PDF](https://arxiv.org/pdf/2604.16049v1)

**作者:** Guodong Sun `[一作]` (Inria), Jean-Marie Gorce `[通讯]` (Inria)

**通讯引用:** 3442 | [OpenAlex ID](https://openalex.org/A5054469607)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于 saddlepoint 近似的稀疏 VLSF 码解码调度优化框架，并给出更严格的最终解码规则。

**💡 创新点**

创新点在于将 saddlepoint 近似用于信息密度分布的解析，支持梯度优化；同时引入了基于最大似然的最终解码规则，显著收紧可实现上界。

**🔧 技术方法**

采用 saddlepoint（Lugannani–Rice）近似、梯度基混合整数非线性规划（MINLP）、随机编码分析等技术。

**📊 数据集**

使用标准信道模型（AWGN、BSC、BEC），无真实数据集。

**📈 对比分析**

通过与暴力搜索和文献基准对比，梯度方法在几秒内得到与暴力搜索相近的最优解，并在短包率下实现约8%（AWGN）/10%（BSC）更高速率。

**⚠️ 局限性**

限制在离散信道上 saddlepoint 在均值附近数值不稳定，需要经验阈值修正；梯度方法无法保证全局最优。

---

## 358. Safe Deep Reinforcement Learning for Building Heating Control and Demand-side Flexibility

**arXiv ID:** 2604.16033 | [PDF](https://arxiv.org/pdf/2604.16033v1)

**作者:** Colin Jüni `[一作]` (Swiss Federal Laboratories for Materials Science and Technology), Philipp Heer `[通讯]` (Swiss Federal Laboratories for Materials Science and Technology)

**通讯引用:** 1284 | [OpenAlex ID](https://openalex.org/A5069545422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于安全深度强化学习的建筑空间加热控制框架，并实现了实时自适应安全滤波器（RASF）以确保满足电网侧的灵活性请求。

**💡 创新点**

创新点在于：①无模型的实时自适应安全滤波器，能够在不依赖物理模型的前提下动态调整DRL动作以严格满足能量灵活性约束；②将PCNN混合热模型与DRL结合，兼顾物理解释性和数据驱动性；③在单月仿真中实现能耗和成本显著降低，同时保持热舒适性。

**🔧 技术方法**

采用深度确定性策略梯度（DDPG）算法作为核心DRL方法；实现实时自适应安全滤波器（RASF）；使用PCNN混合热模型预测室内温度；整合电价、室内外温度、灵活性请求等信息构建状态空间。

**📊 数据集**

使用瑞士Empa NEST实验楼UMAR单元的历史供暖数据（2021-2022年15分钟分辨率）进行训练与仿真。

**📈 对比分析**

与传统规则基（RB）控制器对比；在单月仿真中，DRL+RASF的能耗比RB低约30%，成本比RB低约64.5%，热舒适性提升约45.5%；相比不含安全滤波的DRL，RASF仅略微增加舒适性违规但保证了灵活性约束。

**⚠️ 局限性**

局限性包括：需要足够的历史数据和热模型才能训练；在极端气候条件下灵活性约束可能导致舒适性偏差；目前仅在单建筑仿真验证，需进一步评估在多建筑或多代理场景中的鲁棒性与可扩展性。

---

## 359. IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE

**arXiv ID:** 2604.16010 | [PDF](https://arxiv.org/pdf/2604.16010v1)

**作者:** Rikuto Otsuka `[一作]` (NEC Corporation), Atsushi Ito `[通讯]` (NEC Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可区块自适应的 CLAHE（IA-CLAHE），通过轻量级 CNN 端到端学习每个局部块的 clip limit，从而在无需任务专属训练集的情况下同时提升人类可视性与机器识别性能。

**💡 创新点**

关键创新在于证明 CLAHE 对 clip limit 几乎可微，允许直接使用梯度优化；以及设计只需 211 个参数的轻量级估计器，避免了传统的全局参数和耗时搜索。

**🔧 技术方法**

采用差分 CLAHE、YCrCb 单通道输入、轻量 CNN 估计器、图像 L1 损失、端到端梯度传播以及 Zero‑shot 评估框架。

**📊 数据集**

训练集使用 MSEC（MIT‑Adobe FiveK 处理后的正光图像）；测试集涵盖 CODaN、ExDark、DAWN（分类/检测）以及 MSEC、LCDP（视觉质量）。

**📈 对比分析**

与多种语音映射、学习型与规则型 CLAHE 以及 Transformer、扩散方法进行对比，IA‑CLAHE 在 CODaN、ExDark、DAWN 上达到或超过同类方法的准确率/ mAP，同时保持与传统 CLAHE 同等的处理速度，参数仅略增 211。

**⚠️ 局限性**

局限性包括对 tile grid 选择的依赖、仅处理灰度通道、在极端光照或非光学噪声条件下可能不足，以及对极端分辨率的适配仍需进一步研究。

---

## 360. MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs

**arXiv ID:** 2604.16007 | [PDF](https://arxiv.org/pdf/2604.16007v1)

**作者:** Haoran Wu `[一作]` (University of Cambridge), Robert Mullins `[通讯]` (University of Cambridge)

**通讯引用:** 17784 | [OpenAlex ID](https://openalex.org/A5011576250)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MemExplorer，一套针对异构 NPU 系统的内存系统合成框架，能够在同一平台上自动搜索最优的内存层级、计算架构与软件策略，以满足 agentic LLM 推理在 prefilling 与 decoding 阶段的不同内存需求。

**💡 创新点**

创新点在于：①提出统一抽象模型，兼容 HBM、LPDDR、GDDR、HBF、3D‑Stack SRAM 等多种内存技术；②将软硬协同（数据流、存储调度、精度配置）融入单一设计空间；③采用多目标贝叶斯优化与 EHVI 进行高效的 Pareto 前沿搜索，实现 prefilling 与 decoding 两个阶段的分层优化，首次系统评估 agentic 负载下的内存瓶颈。

**🔧 技术方法**

技术要点包括：分层内存分析与功耗模型；PLENA 性能模型和 Ramulator/Emulator 验证；混合精度量化仿真（MXINT/MXFP）；多目标贝叶斯优化（MOBO）与 EHVI；GPU 基准对比（A100、H100）；异构多设备拓扑（预留 4 卡）与软件策略抽象。

**📊 数据集**

使用的工作负载和数据集有：Qwen3、LLaDA、BFCL、OSWorld、GSM8K、Qwen3.5‑397B‑A17B 等 agentic 与大规模稀疏模型；采用对应的 token 长度配置，例如 BFCL‑WebSearch Base（114K/5K）、OSWorld‑LibreOffice（90K/8K）、GSM8K（1.4K/0.2K）。

**📈 对比分析**

与 NVIDIA A100、H100（四卡）在相同 700 W TDP 下进行对比，评估指标包括时间到首 token (TTFT)、吞吐量 (TPS) 与能效 (Token/J)。MemExplorer 在 prefilling 阶段实现 2.3×/3.23× 的能效提升，decode 阶段提升 1.93×/2.72×；在 agentic、dLLM 与 MoE 任务上，基线提升 1.3–3.5× 能效，且在吞吐量和延迟上保持竞争优势。

**⚠️ 局限性**

限制：未建模多设备共享内存与芯片间通信开销；精度统一缺少跨阶段混合精度支持；依赖仿真模型，实际硬件实现细节（如热设计、可靠性）未覆盖；未考虑更深层次的跨设备资源分配与调度。

---

## 361. jMT: Testing Correctness of Java Memory Models (Extended Version)

**arXiv ID:** 2604.15978 | [PDF](https://arxiv.org/pdf/2604.15978v1)

**作者:** Lukas Panneke `[一作]` (Carl von Ossietzky Universität Oldenburg), Heike Wehrheim `[通讯]` (Carl von Ossietzky Universität Oldenburg)

**通讯引用:** 2557 | [OpenAlex ID](https://openalex.org/A5046915601)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一款支持多执行模型的Java内存模型测试工具，用于评估新的JMM语义与编译方案的一致性。

**💡 创新点**

其创新点在于引入符号执行图与跨执行事件等价关系，能够有效表示无限多候选执行并进行因果性检查，发现现有JMM中的错误和编译错误。

**🔧 技术方法**

工具实现基于符号执行图、SMT求解器、证明序列、非优化x86编译器以及Java并发压力测试框架。

**📊 数据集**

实验使用了CTC、jcstress、S&A和MT等四个公开litmus测试集，覆盖API中的新访问模式。

**📈 对比分析**

通过三种场景（行为一致性、编译方案正确性、编译器实现一致性）进行比较，所有测试均在一分钟以内完成，结果与现有工具一致或发现新错误。

**⚠️ 局限性**

局限性包括仅支持无循环程序，未覆盖final字段、监视器、外部事件等语言特性，且不支持递归关系的模型。

---

## 362. Driving Assistance System for Ambulances to Minimise the Vibrations in Patient Cabin

**arXiv ID:** 2604.16047 | [PDF](https://arxiv.org/pdf/2604.16047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 363. TwinTrack: Post-hoc Multi-Rater Calibration for Medical Image Segmentation

**arXiv ID:** 2604.15950 | [PDF](https://arxiv.org/pdf/2604.15950v1)

**作者:** Tristan Kirscher `[一作]` (University of Strasbourg), Sylvain Faisan `[通讯]` (University of Strasbourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出TwinTrack框架，对胰腺导管腺癌CT分割模型进行后期校准，使预测概率与多位专家的平均响应一致。

**💡 创新点**

创新点在于通过等距回归将模型输出对齐到多评审平均响应(MHR)，显式建模专家间不一致性，提升校准性和临床可解释性。

**🔧 技术方法**

使用低分辨率nnU-Net定位+高分辨率nnU-Net集成，以及等距回归(Isotonic Regression)进行后期校准，并在CURVAS–PDACVI挑战中评估。

**📊 数据集**

基于PANORAMA batch 4训练模型，使用CURVAS–PDACVI多评审CT数据集（40张扫描做校准，64张扫描做测试）。

**📈 对比分析**

与无校准、单评审校准和硬标签校准等三种对照方法比较，TwinTrack在TDSC、ECE、CRPS以及血管侵袭指标上均优于其他方法，并在MICCAI 2025 CURVAS–PDACVI挑战中获得第一名。

**⚠️ 局限性**

局限性包括在背景占比极高的像素上校准提升有限；需要小规模多评审校准集，且对不同模型或其他模态的泛化尚未充分验证。

---

## 364. Evaluating quality in synthetic data generation for large tabular health datasets

**arXiv ID:** 2604.15961 | [PDF](https://arxiv.org/pdf/2604.15961v1)

**作者:** Jean-Baptiste Escudié `[一作]` (Robert Koch Institute), Nils Körber `[通讯]` (Robert Koch Institute)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5111503326)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文对七种主流合成表格数据模型在四个不同规模数据集上进行系统性调参后进行质量评估与比较。

**💡 创新点**

创新点在于提出了一套基于配分布（MAE、Coverage、Invented）与直方图交并比（Hist_IoU）的简洁评估框架，并结合可视化实现一图即见多维联合分布质量。

**🔧 技术方法**

使用的技术包括TPE/多目标TPE超参搜索、VAEs、GANs、PGMs、Diffusion、LLM、差分隐私等多种合成模型；评估采用配分布误差、覆盖率、发明率、IOU、QQ图及域违规检测。

**📊 数据集**

实验数据集为小型的Abalone、Adult，规模中等的USCensus1990，以及大规模的德国癌症登记表EpiCancerGER（约百万样本）。

**📈 对比分析**

比较方法是先在每个数据集上用HPO得到最优超参，再用MAE_2和Hist_IoU_2等指标进行排序；结果显示PrivSyn在所有数据集上均获得最佳配分布匹配，TabDDPM排名第二，其他模型表现相对落后。

**⚠️ 局限性**

主要局限在于HPO成本高、模型对超参高度敏感、某些模型（如PGM、CTGAN）存在显著域违规率，且评估框架未覆盖跨域泛化与隐私‑质量权衡的细致探究。

---

## 365. T-RBFT: A Scalable and Efficient Byzantine Consensus Based on Trusted Execution Environment for Consortium Blockchain

**arXiv ID:** 2604.16053 | [PDF](https://arxiv.org/pdf/2604.16053v1)

**作者:** Wen Gao `[一作]` (Xi'an University of Technology), Yichuan Wang `[通讯]` (Xi'an University of Technology)

**通讯引用:** 14707 | [OpenAlex ID](https://openalex.org/A5071592530)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于可信执行环境的双层共识机制 T‑RBFT，采用节点分片和改进的 Raft 内层以及 TEE‑辅助的 BFT 外层，以提升联盟链的吞吐量与延迟。

**💡 创新点**

创新点包括：① 使用 USIG 在 TEE 中为消息生成唯一递增标识，减少 PBFT 的三轮通信至两轮；② 通过一致性哈希与动态权重实现节点分组与负载均衡；③ 在 Raft 里加入日志复制、领导选举与提交确认的 Byzantine 容错强化机制；④ 结合 BLS 聚合签名与 HMAC 保障跨组共识安全。

**🔧 技术方法**

核心技术包括 Intel SGX/TEE、USIG（HMAC‑SHA256）、BLS 签名与聚合、Raft、PBFT、BLS 组合签名、虚拟节点一致性哈希、可观察的交互式心跳与视图变更。

**📊 数据集**

实验使用本地容器化多节点部署，硬件环境为 Intel i7‑9700 + 36 GB 内存，未采用公开数据集，仅通过模拟网络进行吞吐与延迟评测。

**📈 对比分析**

与 MinBFT、R‑PBFT、WRBFT 在相同组节点数（每组 4 节点）下进行对比：通信轮数显著下降（T‑RBFT 188 轮 vs R‑PBFT 246 轮），吞吐量提升，延迟降低；在 60 节点场景下，组数减少可进一步降低通信轮数（k=3 时仅 236 轮）。

**⚠️ 局限性**

局限性：需要可信硬件支持（SGX 等）；跨链或多链扩展尚未实现；在高并发或节点动态加入/离开时的稳定性与安全性需进一步验证；与 WRBFT 相比安全性有待提升，且在极端 Byzantine 情形下容错能力有限。

---

## 366. Elucidating the SNR-t Bias of Diffusion Probabilistic Models

**arXiv ID:** 2604.16044 | [PDF](https://arxiv.org/pdf/2604.16044v1)

**作者:** Meng Yu `[一作]` (Lanzhou University), Kun Zhan `[通讯]` (Lanzhou University)

**通讯引用:** 3113 | [OpenAlex ID](https://openalex.org/A5058413200)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在扩散模型推理阶段存在的信噪比‑时间步（SNR‑t）偏差，并提出了差分校正（DCW）方法来缓解该偏差

**💡 创新点**

创新点在于：①首次系统性地量化SNR‑t偏差并给出理论证明；②提出无训练、可插拔的差分校正机制，并在小波域对不同频率分量分别校正，利用模型天然低频优先恢复的特性；③使用动态权重与逆小波变换实现高效、低成本的修正

**🔧 技术方法**

主要技术包括扩散概率模型、理论分析（基于逆过程的SNR计算）、差分校正公式、离散小波变换（DWT/iDWT）和动态权重调节；实现基于PyTorch的无训练插拔插件

**📊 数据集**

在CIFAR‑10、CelebA 64×64、ImageNet 128×128、LSUN Bedroom 256×256等多分辨率图像数据集上进行实验，并在多种DPM框架（IDDPM、ADM、DDIM、A‑DPM、EA‑DPM、EDM、PFGM++、FLUX等）上验证

**📈 对比分析**

与基线模型、现有曝光偏差校正模型（ADM‑IP、ADM‑ES、DPM‑FR等）进行对比，使用FID和Recall评估。结果显示，DCW在所有模型与数据集上均显著降低FID（最高可达50%+），并保持1–2%以内的计算时间开销，性能优于或补充了现有方法

**⚠️ 局限性**

局限性包括：1）仅在图像生成任务上验证，缺乏对音频/视频等领域的评估；2）对超高分辨率图像和复杂数据分布的鲁棒性尚待进一步研究；3）依赖模型已训练完成，无法解决训练阶段的SNR‑t偏差问题；4）对小波基的选择和超参数仍需经验性调优

---

## 367. Towards Intrinsic Interpretability of Large Language Models:A Survey of Design Principles and Architectures

**arXiv ID:** 2604.16042 | [PDF](https://arxiv.org/pdf/2604.16042v1)

**作者:** Yutong Gao `[一作]` (Peking University), Liangming Pan `[通讯]` (Peking University)

**通讯引用:** 1028 | [OpenAlex ID](https://openalex.org/A5027533517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了大语言模型（LLM）内在可解释性（intrinsic interpretability）的方法，提出并阐述了功能透明、概念对齐、表示可分解、显式模块化、潜在稀疏诱导五大设计原则，并讨论了这些方法的优势、局限和未来研究方向。

**💡 创新点**

创新点在于将内在可解释性方法统一归纳为五大设计范式，清晰区分后置解释与内在解释的本质区别，提供了一个可操作的框架来组织和评估现有技术，并对当前缺失的评估标准和扩展性挑战给出了系统性的建议。

**🔧 技术方法**

综述聚焦于多种模型架构和训练策略，包括但不限于Generalized Additive Models、Self‑Explaining Neural Networks、Concept Bottleneck Models、Mixture‑of‑Experts、稀疏正则化、GLU等；并未自行实现实验，而是对上述技术在公开论文中的实现细节与原理进行了总结。

**📊 数据集**

本综述未使用专门的数据集进行实验，而是参考了多篇相关论文中使用的标准 NLP 数据集（如GLUE、SQuAD、Wikitext、WikiText-103、BookCorpus 等）来说明各方法在实际任务中的表现。

**📈 对比分析**

通过对比各方法在透明度（可解释性）与表达力（任务性能）、可扩展性、训练成本等维度的平衡，指出目前多方法在小规模模型上能保持较好性能，但在大规模模型上的可解释性和性能平衡仍缺乏统一评估；同时强调缺乏统一的评价指标导致难以直接比较不同方法的真实效果。

**⚠️ 局限性**

限制主要包括：缺乏统一、可量化的内在可解释性定义与评估框架；大规模 LLM 上的可解释性设计难以扩展、训练成本高且易出现不稳定；现有方法多基于小规模模型或特定任务，缺乏在真实大规模多任务场景下的系统验证。

---

## 368. "When I see Jodie, I feel relaxed": Examining the Impact of a Virtual Supporter in Remote Psychotherapy

**arXiv ID:** 2604.16003 | [PDF](https://arxiv.org/pdf/2604.16003v1)

**作者:** Jiashuo Cao `[一作]` (University of Auckland), Mark Billinghurst `[通讯]` (University of Auckland)

**通讯引用:** 31878 | [OpenAlex ID](https://openalex.org/A5021195349)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了虚拟支持者 Jodie，帮助用户在日常情绪记录与远程心理治疗过程中提供情感陪伴与观察支持，并通过用户研究验证其有效性。

**💡 创新点**

首创将虚拟支持者与人类治疗师共同参与远程心理治疗，提出双模式（日常记录/治疗会话）和严格的界限规则，以避免过度依赖与保持治疗师主导。

**🔧 技术方法**

利用 Soul Machines HumanOS（具身会话代理）+ Deepgram ASR + Dialogflow CX 进行语音识别与对话管理，OBS+Virtual Camera 实现虚拟人视频在 Zoom 中显示。

**📊 数据集**

实验数据来自 14 名未曾接受心理治疗的青年参与者（7 男 7 女，平均 22.9 岁）和 1 名治疗师，未使用公开数据集。

**📈 对比分析**

评估采用 SUS、SEQ、mARM 等问卷，SUS 72.8 分（B‑级），SEQ 体验总体正面，mARM 显示开放度、纽带感、伙伴关系均高；未与无支持者或人类支持者对照，但指标表明使用 Jodie 并未干扰治疗流程。

**⚠️ 局限性**

局限：样本量小且同质化（无临床人群），单一治疗师参与，实验周期短（仅 7 天 + 1 次会谈），规则驱动对话缺乏自然度，未能充分检验长期使用与临床疗效。

---

## 369. MATRIX: Multi-Layer Code Watermarking via Dual-Channel Constrained Parity-Check Encoding

**arXiv ID:** 2604.16001 | [PDF](https://arxiv.org/pdf/2604.16001v1)

**作者:** Yuqing Nie `[一作]` (Beijing University of Posts and Telecommunications), Kailong Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1437 | [OpenAlex ID](https://openalex.org/A5000432413)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双通道、基于奇偶校验矩阵求解的代码水印框架，能够在保持功能性和语义一致性的前提下，将多层可追溯的水印嵌入LLM生成的Python代码。

**💡 创新点**

创新点包括：① 将水印编码视为约束奇偶校验矩阵方程，利用BCH纠错码和解空间多样性实现高可隐蔽性；② 双通道机制（变量命名和语义保持变换）提供冗余备份，显著提升抗攻击鲁棒性；③ 通过可变的奇偶校验矩阵支持跨组织与多层级归因，无需改动变换规则。

**🔧 技术方法**

核心技术：语义保持的代码变换与变量重命名；BCH编码与奇偶校验矩阵求解（CSP求解）；双通道编码与检测；对抗性攻击评估与统计不可区分性分析；静态AST分析实现高效嵌入/提取。

**📊 数据集**

使用GPT‑4、StarCoder在MBPP、APPS、HumanEval三大公共数据集生成的Python代码进行实验，覆盖约3万+样本。

**📈 对比分析**

与现有ACW、SWEET等基线对比，平均水印检测准确率99.20%，功能完整率0–0.14%，对变量重命名、重构、LLM重写与格式化攻击的鲁棒性提升7.7–26.7%，水印覆盖率比ACW高2–6倍，消息恢复率接近100%。

**⚠️ 局限性**

局限性包括：① 仅能在语法完整的代码块中检测，无法处理片段或非语法完整代码；② 依赖手工编写的语义变换规则，覆盖面有限；③ 对极度细粒度多组织归因时可能因可用锚点不足而受限；④ 高度信息化的攻击者仍可通过高级重写策略潜在规避。

---

## 370. From Vulnerable Data Subjects to Vulnerabilizing Data Practices: Navigating the Protection Paradox in AI-Based Analyses of Platformized Lives

**arXiv ID:** 2604.15990 | [PDF](https://arxiv.org/pdf/2604.15990v1)

**作者:** Delfina S. Martinez Pandiani `[一作]` (University of Amsterdam), Paula Helm `[通讯]` (Goethe University Frankfurt)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5057959142)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

分析并阐述了在平台化生活中利用 AI 进行“善用”研究所带来的保护悖论，提出了反思性伦理协议；

**💡 创新点**

将关注点从“受害者”转向“导致脆弱化的技术实践”，并系统化了四个关键技术节点的伦理风险与四种脆弱化机制；

**🔧 技术方法**

采用计算机视觉（人脸检测、情感识别、NSFW 检测）和人工智能推理技术；

**📊 数据集**

使用公开的欧盟家庭 vlog 视频（4 条渠道、近一年内容）作为实验数据集；

**📈 对比分析**

未进行算法性能对比或量化评估，重点在于揭示技术决策对脆弱性的潜在影响；

**⚠️ 局限性**

局限在于单案例、缺乏定量验证、主要聚焦欧盟法律语境，无法直接推广至其他司法或技术环境。

---

## 371. Weak-Link Optimization for Multi-Agent Reasoning and Collaboration

**arXiv ID:** 2604.15972 | [PDF](https://arxiv.org/pdf/2604.15972v1)

**作者:** Haoyu Bian `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112063 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种弱链优化框架WORC，利用元学习权重预测和群体智能算法识别多智能体系统中的弱环节，并通过预算分配补偿弱智能体以提升整体推理稳定性与准确率。

**💡 创新点**

以弱链原理为核心的两阶段流程（弱智能体定位与弱链优化），结合跨任务零样本的权重预测器与任务签名，实现系统化的弱链补偿策略。

**🔧 技术方法**

使用群体智能算法（PSO、GWO、HO）生成权重知识库，元学习 MLP 预测权重，任务签名由语义嵌入与统计特征构成，EvalAgent 与 VoteAgent 评估与投票机制，以及基于预测权重的预算分配函数。

**📊 数据集**

在六大推理基准上评测：MATH、GSM8K、BBH、MMLU‑CF、HotpotQA 与 LongBench。

**📈 对比分析**

在 GPT‑4o 基础上与 CoT、CoT‑SC、Self‑Refine、Analogical Prompting、AFlow、FoT、AoT 等推理方法，以及 MetaGPT、HIMA、MAS^2、AgentChain 等多架构对照，WORC+AC 在六项指标平均达 82.2% 以上，较基线提升约 3–6%，在所有架构中均实现 3–6% 的平均增益。

**⚠️ 局限性**

需要额外的 SIA 训练与评估成本，预算分配依赖于预测权重的线性函数，模型在极端稀有任务上的迁移能力有限，且在长上下文任务中计算开销显著增加。

---

## 372. Supporting the Comprehension of Data Analysis Scripts

**arXiv ID:** 2604.15963 | [PDF](https://arxiv.org/pdf/2604.15963v1)

**作者:** Florian Sihler `[一作]`, Matthias Tichy `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该示例展示了如何在Beamer幻灯片中嵌入Java代码并通过`onslide`指令实现分步展示和动态高亮；

**💡 创新点**

创新点在于将代码块与幻灯片切换同步，并利用`escapeinside`实现代码内的标记控制，提升演示效果；

**🔧 技术方法**

使用的技术主要是LaTeX Beamer、`lstlisting`（或相似代码高亮宏包）以及自定义的`onslide`参数；

**📊 数据集**

没有使用外部数据集，所有内容均为示例代码；

**📈 对比分析**

未进行实验对比，仅为演示功能，暂无性能指标；

**⚠️ 局限性**

限制在于示例简单，缺乏对复杂代码结构的支持，以及对多种语言高亮与交互效果的系统评估。

---

## 373. Stochastic wage suppression on gig platforms and how to organize against it

**arXiv ID:** 2604.15962 | [PDF](https://arxiv.org/pdf/2604.15962v1)

**作者:** Ana-Andreea Stoica `[一作]` (Max Planck Institute for Intelligent Systems), Moritz Hardt `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 15746 | [OpenAlex ID](https://openalex.org/A5039915143)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种基于量化分位数的后续价格策略，分析在不同工人成本分布下平台如何通过抑价实现工资压制，并探讨横向与纵向集体行动对工资水平的影响。

**💡 创新点**

创新点在于：①将后续价格设为工作类别剩余数的分位数，实现线性等待时间的同时实现对工资的可控压制；②系统性划分工资压制的四个阶段（线性、次线性、对数、常数）并给出充分条件；③比较横向（随机）与纵向（定向）集体行动的效果，揭示后者能显著阻止工资压制。

**🔧 技术方法**

主要技术包括：概率论与几何分布分析、分位数/尾部条件判定、Bernstein不等式求期望与集中性；以及对不同分布族（Uniform、Beta、Exponential）进行解析和数值验证。

**📊 数据集**

使用合成实验数据：通过在不同分布族（Uniform、Beta(0.5,1)、Beta(2,1)、Exponential λ=3）下模拟平台-工人交互，评估支付与等待时间。

**📈 对比分析**

比较方法：在各分布下与不同集体行动比例α、预算ε下运行策略，绘制总支付与等待时间曲线。结果显示：横向集体行动仅增加常数因子；纵向集体行动在适当预算下将支付从对数/常数提升到线性。

**⚠️ 局限性**

局限性：①未考虑多平台竞争、动态学习与在线调价；②只在理想化的独立同分布工人成本模型下实验；③实验仅为合成，缺乏真实平台数据验证；④纵向集体行动的预算需求与实际组织成本缺乏定量分析。

---

## 374. Hardness, Tractability and Density Thresholds of finite Pinwheel Scheduling Variants

**arXiv ID:** 2604.16030 | [PDF](https://arxiv.org/pdf/2604.16030v1)

**作者:** Sotiris Kanellopoulos `[一作]` (National Technical University of Athens), Thanos Tolias `[通讯]` (National Technical University of Athens)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5117111324)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文研究了Pinwheel Scheduling 的有限版本 2‑Visits（以及其更一般的 (m,n)‑Visits），阐明了其计算复杂度、可行算法和密度阈值，并给出了新的硬件与上界分析。

**💡 创新点**

创新点包括：①证明 2‑Visits 在最大复制数为 2 时仍为强 NP‑完整，彻底否定了此前关于最大复制数参数化可 FPT 的猜想；②当不同截止时间个数为常数时，2‑Visits 属于 RP（利用多重图匹配与 Mulmuley 等人算法的结合）；③首次给出 2‑Visits 的下密度阈值为 √2–½≈0.9142，且当 k→∞ 时阈值趋近 5/6；④将已有 2‑Visits 的多项式与随机算法推广到 (m,n)‑Visits，并指出 k>2 的情形仍面临结构性障碍。

**🔧 技术方法**

技术手段主要包括：多阶段的多元数值匹配（NMTS、PositionMatching 等）归约、基于离散化序列的匹配约束、在多重图中引入权重并利用 Mulmuley‑Vazirani‑Vazirani 的 RP 算法、以及对状态向量的鸽巢原理推导密度阈值。

**📊 数据集**

该研究为理论计算机科学，未使用实际数据集；所有结果均为理论证明与复杂度分析。

**📈 对比分析**

与先前工作相比，本文证明了更强的 NP‑完整性（最大复制数为 2）并给出了 RP 算法，复杂度分析表明 2‑Visits 在常数截止数下可在多项式时间（随机）解决，而 2‑Visits 的下密度阈值高于之前已知的 5/6，说明在高密度实例中仍可保证可调度；对于 (m,n)‑Visits，算法时间与任务数相关，保持与 2‑Visits 相同的阶级。

**⚠️ 局限性**

局限性在于：①对 k>2 的 3‑Visits 等问题仍未给出完整的复杂度结论；②上密度阈值未定义，导致高密度实例仍不受限制；③随机化算法未被确定为确定性算法，且在实际实现中需要随机性；④对简单集合的 NP‑完整性尚未完全证明，存在未解决的开放问题。

---

## 375. Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning

**arXiv ID:** 2604.16029 | [PDF](https://arxiv.org/pdf/2604.16029v1)

**作者:** Jiaxi Bi `[一作]` (Chinese University of Hong Kong), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了并实现了一套统一的路径剪枝分类体系，并在此基础上设计了基于内部可学习信号的轻量级剪枝方法STOP。

**💡 创新点**

首创将内部可学习信号与剪枝结合，提出了超词(Super Token)与LoRA适配器协同的评估模块，实现了既能捕捉内部推理动态又能高效学习的剪枝方案。

**🔧 技术方法**

使用了超词+LoRA+分类头的轻量模块、蒙特卡洛软标签作为训练目标、前缀级别的路径剪枝以及三阶段（Launch‑Check‑Resume）推理流水线。

**📊 数据集**

在多种算术与 STEM 基准（AIME24/25、BRUMO25、HMMT25、GPQA-D）、DS‑Qwen 系列模型以及 ZebraLogic、AIMO3 工具使用场景上进行评测。

**📈 对比分析**

与无剪枝基线以及其他三类剪枝方法（内部/外部、可学习/不可学习）做对比，STOP 在 avg@k 提升约 7–9 % 的同时，令总 token 消耗降低 70%+，在 GPT‑OSS‑20B 等大模型上仍保持显著优势。

**⚠️ 局限性**

仅在 1.5B–20B 范围内验证，使用固定长度前缀且仅单阶段剪枝；未对 70B+ 大模型或大规模采样（N≥1000）以及多阶段/动态剪枝进行实验。

---

## 376. SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems

**arXiv ID:** 2604.16022 | [PDF](https://arxiv.org/pdf/2604.16022v1)

**作者:** Hikaru Shindo `[一作]` (Technical University of Darmstadt), Kristian Kersting `[通讯]` (Technical University of Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出并实现了一个可控制的、具备实体化、多智能体的社交推理基准框架 SocialGrid，灵感来自《Among Us》，用于评估大语言模型在空间规划、任务执行与社交推理三大维度的表现。

**💡 创新点**

将空间规划与社交推理融合到同一环境中，提供可调节的地图与 agent 配置；引入可选的规划 Oracle 以隔离导航缺陷；设计细粒度的失败分析与 Elo 竞赛排行榜；在同一框架下统一评估多模型在多目标、多智能体情境下的表现。

**🔧 技术方法**

使用基于 GridWorld 的离散网格环境，集成 A* 路径规划 Oracle，构建任务、投票、奖励等多维度机制；对大语言模型进行 prompt 设计与 vLLM 推理；利用 RL（PPO + LoRA）在简化环境下尝试提升规划能力；通过自动化评测脚本实现多指标评估与失败模式分析。

**📊 数据集**

使用公开的大语言模型（Qwen3-30B/80B、Llama3.1-70B、GPT‑OSS‑120B、Gemma3-27B、DeepSeek‑R1‑70B、Phi4‑reasoning‑14B）在自定义的 SocialGrid 环境中进行实验，实验数据集由多种房间数、尺寸与模型配置自动生成。

**📈 对比分析**

在无规划助手的基准测试中，最强模型 GPT‑OSS‑120B 任务完成率仅 50%；加入规划 Oracle 后完成率大幅提升，但规划效率仍低于 0.5。对比六种模型的 Elo 排名，GPT‑OSS 领先；但在投票与信任度指标上，所有模型均接近随机 33% 的准确率，表明社交推理瓶颈未随模型规模提升。

**⚠️ 局限性**

缺点包括：空间规划仍是核心瓶颈，RL 细调难以显著改进；社交推理能力几乎不随规模提升；实验环境为离散网格，且缺少讨论阶段，难以完全反映真实社交情境；强制的 Impostor 胜率高可能与游戏参数有关，需进一步平衡与消融。

---

## 377. Solving Fuzzy Satisfiability via Mixed-Integer Non-Linear Programming

**arXiv ID:** 2604.15992 | [PDF](https://arxiv.org/pdf/2604.15992v1)

**作者:** Pablo F. Castro `[一作]` `[通讯]` (Universidad Nacional de Río Cuarto), Pablo F. Castro (Universidad Nacional de Río Cuarto)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一款基于MINLP求解器的模糊逻辑SAT求解器（Satful），能够将模糊公式转化为混合整数非线性规划问题并判定可满足性。

**💡 创新点**

创新点在于：①使用MINLP求解而非传统的MILP或CSP方法，实现在所有主流模糊命题逻辑（Łukasiewicz、Product、Gödel）上的统一可满足性判定；②证明了该翻译方法的完备性与可满足性对应；③通过实验展示了在Product逻辑上明显优于现有求解器，并在Łukasiewicz逻辑上与最先进工具相当甚至在UNSAT实例上更快。

**🔧 技术方法**

技术手段包括：递归式公式到MINLP的转换算法、对每个子公式使用新变量约束、调用商业/学术MINLP求解器（如Gurobi、open-source solver）进行求解，以及完整的工具架构（预处理→AST→MINLP构造→求解）。

**📊 数据集**

使用了两类数据集：①来自已有工作（如<cit.>）的标准模糊SAT基准；②随机生成400条Product逻辑公式作为额外测试集。

**📈 对比分析**

与相关工具（Łukasiewicz逻辑下的CSP求解器、Product逻辑下的SMT+Product求解器）进行性能对比。实验显示：在Product逻辑上，Satful在SAT和UNSAT实例均明显快于其他工具；在Łukasiewicz逻辑上，与state‑of‑the‑art求解器性能相当，在UNSAT实例上略优；在某些SAT实例上，使用Gurobi时性能更好。

**⚠️ 局限性**

局限性包括：①目前仅实现对三种主流模糊逻辑的支持；②依赖外部MINLP求解器的可用性和性能；③尚未实现增量求解、冲突核心提取等高级特性；④在极大规模或高非线性约束的公式上可能仍面临求解瓶颈。

---

## 378. The QBF Gallery 2023

**arXiv ID:** 2604.16153 | [PDF](https://arxiv.org/pdf/2604.16153v1)

**作者:** Simone Heisinger `[一作]` (Johannes Kepler University Linz), Martina Seidl `[通讯]` (Johannes Kepler University Linz)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文记录并评估了2023年QBF Gallery评测活动，包括五个赛道、最新公式集合、预处理器和求解器的性能；

**💡 创新点**

创新点在于引入了非CNF格式的公式、提供了统一的合并基准集，并通过多种评估指标（PAR‑2、唯一求解数等）系统比较求解器；

**🔧 技术方法**

主要使用了QCDCL、CEGAR、CEGIS、预处理技术（变量消除、量化变量展开）以及基于公式结构的求解方法；

**📊 数据集**

使用了518个PCNF公式、418个PNCNF公式、354个DQBF公式以及71个每个家族的手工构造实例，共计约1400个公式；

**📈 对比分析**

通过运行时间、解题数量、唯一解题数和PAR‑2得分等指标进行比较，结果显示不同求解器在SAT与UNSAT实例上表现差异显著，预处理器整体提升了解题率但仍有差异；

**⚠️ 局限性**

局限性包括基准集对某些求解器不够多样化、内存限制导致结果波动、部分求解器在某些实例上不一致或被排除，且评测未覆盖所有最新技术。

---

## 379. On the Rejection Criterion for Proxy-based Test-time Alignment

**arXiv ID:** 2604.16146 | [PDF](https://arxiv.org/pdf/2604.16146v1)

**作者:** Ayoub Hammal `[一作]` (Université Paris-Saclay), Caio Corro `[通讯]` (INSA Rennes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的概率图模型，用以统一并改进代理式测试时对齐方法，并基于该模型引入了保守置信赌注（conservative confidence bet）拒绝准则，直接利用小模型的对齐信息来决定是否推迟大模型生成下一词；

**💡 创新点**

创新点在于将隐式奖励、nudging 与双向 KAD 等先前方法统一到同一 PGM 框架，阐明它们的本质相同，并提出了以小模型最大置信度为基准的拒绝策略，克服了仅依赖大模型绝对置信度的局限；

**🔧 技术方法**

主要技术包括概率图模型构造、拒绝采样与决策阈值设计、保守置信赌注的阈值设置，以及在大型语言模型上实现测试时对齐的前向推断；

**📊 数据集**

实验使用了数学推理数据集 GSM8K、MATH500、SVAMP 与常识推理数据集 ARC‑Challenge、CommonsenseQA，模型选取 OLMo 2 与 Qwen 3 的不同规模版本；

**📈 对比分析**

与 KAD 双向近似规则及其他基线比较时，保守置信赌注在多数任务上提升了准确率，尤其在 OLMo 2 上的 MATH500 任务中取得显著提升，整体逼近或接近已对齐的大模型性能；

**⚠️ 局限性**

局限性包括需在验证集上调优阈值参数 λ，且在 Qwen 3 的常识推理任务中相对 KAD 规则效果略逊，说明对齐策略仍受模型基准性能差异影响。

---

## 380. Training Time Prediction for Mixed Precision-based Distributed Training

**arXiv ID:** 2604.16145 | [PDF](https://arxiv.org/pdf/2604.16145v1)

**作者:** Minchul Kang `[一作]` (Korea University), Chuck Yoo `[通讯]` (Korea University)

**通讯引用:** 1091 | [OpenAlex ID](https://openalex.org/A5067814301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种精度感知的分布式训练时间预测器，可自动识别算子级别浮点精度并考虑通信开销。

**💡 创新点**

创新点在于支持任意精度设置（FP32、FP16、混合精度）并通过模型图分区与算子级精度分析实现高精度预测。

**🔧 技术方法**

采用模型图分区、算子精度hook、前向后向计时、All‑Reduce 传输量估算与流水线瓶颈模型等技术。

**📊 数据集**

以 LLaMA 3.1‑8B（C4 数据集）在八块 NVIDIA H100 GPU 上进行实验。

**📈 对比分析**

与 NeuSight 与 vTrain 对比，平均 MAPE 仅 9.8%（混合精度）/10.64%（FP16），比对方提升约 15 倍。

**⚠️ 局限性**

局限在多节点异构 GPU 环境下的预测未验证，且仅针对单循环训练时间。

---

## 381. PolicyGapper: Automated Detection of Inconsistencies Between Google Play Data Safety Sections and Privacy Policies Using LLMs

**arXiv ID:** 2604.16128 | [PDF](https://arxiv.org/pdf/2604.16128v1)

**作者:** Luca Ferrari `[一作]` (IMT School for Advanced Studies Lucca), Luca Verderame `[通讯]` (University of Genova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于多提示LLM的自动化方法——PolicyGapper，用于检测Google Play应用隐私政策（PP）与数据安全章节（DSS）之间的遗漏声明

**💡 创新点**

首次在大规模应用上实现PP–DSS一致性检测，并通过LLM在不需要APK的情况下完成文本抽取、比对与校验

**🔧 技术方法**

使用Gemini 2.5‑Pro等大型多模态LLM，结合分层提示策略（3个提示覆盖三大数据类别），实现从PP抽取声明、与DSS对齐、去重校验

**📊 数据集**

实验基准为2025年第三季度Google Play 33个分类各10款，合计330款高安装量应用（set_large），以及10%随机子集（set_medium）用于人工验证

**📈 对比分析**

在set_medium上三次独立运行，平均Precision 0.75、Recall 0.77、F1 0.76；在set_large发现2689条遗漏声明，覆盖约8条/款，说明方法在规模化检测中保持稳定性能

**⚠️ 局限性**

局限包括对中文/非英文PP缺乏支持、对非WebView/独立网站的隐私实践检测失效、LLM非确定性导致误报/漏报、以及仅检测声明一致性而不验证实际行为

---

## 382. Beyond One-Size-Fits-All: Adaptive Test-Time Augmentation for Sequential Recommendation

**arXiv ID:** 2604.16121 | [PDF](https://arxiv.org/pdf/2604.16121v1)

**作者:** Xibo Li `[一作]` (Hong Kong University of Science and Technology), Liang Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 81903 | [OpenAlex ID](https://openalex.org/A5100425671)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 AdaTTA 框架，在序列推荐任务中实现测试时自适应数据增强，以提升推理准确性。

**💡 创新点**

创新点在于将数据增强策略视为强化学习决策问题，利用 Actor‑Critic（PPO）学习基于序列特征的动态增广操作选择，并设计宏观‑排名联合奖励机制。

**🔧 技术方法**

核心技术包括 Actor‑Critic 强化学习、混合语义与统计状态表示、TTA 算子集合、以及多目标奖励设计。

**📊 数据集**

实验使用四个公开数据集（Amazon Beauty、Sports、Home 与 Yelp）以及两种基准模型（SASRec 与 GRU4Rec）。

**📈 对比分析**

与固定策略 TTA 基线和原始模型对比，AdaTTA 在所有指标上显著提升，最佳情况比最优固定策略提升约 19%~26%，相对提升达数十个百分点。

**⚠️ 局限性**

主要局限在于增广算子集合有限，且在极端稀疏场景下效果尚需进一步验证；推理时相较纯模型仍多约 1.5 倍的延迟。

---

## 383. Univariate Channel Fusion for Multivariate Time Series Classification

**arXiv ID:** 2604.16119 | [PDF](https://arxiv.org/pdf/2604.16119v1)

**作者:** Fernando Moro `[一作]` (Pontifícia Universidade Católica do Paraná), Vinicius M. A. Souza `[通讯]` (Pontifícia Universidade Católica do Paraná)

**通讯引用:** 1765 | [OpenAlex ID](https://openalex.org/A5005147996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种将多变量时间序列通过均值、中位数或DTW质心聚合成单变量序列的轻量级方法UCF

**💡 创新点**

创新点在于用极其简单的聚合操作实现通用、低资源的多变量时间序列分类，并且在不显著牺牲准确率的前提下大幅降低计算成本

**🔧 技术方法**

主要技术包括点级聚合函数（均值/中位数/质心）、QUANT等单变量分类器、DTW质心求解、统计显著性检验等

**📊 数据集**

使用了EEG、ECG、近红外光谱、MEG和ECoG等五个真实工业/医疗数据集进行评估

**📈 对比分析**

与拼接、投票、DTW_D/I以及SOTA的WEASEL+MUSE、TapNet等方法对比，UCF在准确率上与SOTA相当或更优，同时在训练和推理时间上至少快数倍至千倍

**⚠️ 局限性**

局限性是聚合会导致信息丢失，尤其在通道间相关性低、信息来源多样且互补的场景中表现会显著下降

---

## 384. Towards In-Context Tone Style Transfer with A Large-Scale Triplet Dataset

**arXiv ID:** 2604.16114 | [PDF](https://arxiv.org/pdf/2604.16114v1)

**作者:** Yuhai Deng `[一作]` (Nankai University), Xiang Li `[通讯]` (Nankai International Advanced Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了 TST100K 和 TST2K 两大规模内容-参考-成品三元组数据集，并提出 ICTone 模型实现语境式色调风格迁移。

**💡 创新点**

创新点在于三：①两阶段训练的色调风格评分器用于数据筛选和奖励学习；②利用扩散模型的上下文推理实现语境式迁移；③引入奖励反馈强化风格一致性。

**🔧 技术方法**

采用 CLIP/ViT 架构训练评分器，使用对比学习 + 人类排序微调；采用扩散变换器 DiT + 流匹配目标；结合奖励反馈学习。

**📊 数据集**

使用 PPR10K、MIT-Adobe FiveK 等公共数据集生成预设样本，构建 TST100K（100k 三元组）和 TST2K（2000 精选三元组）用于训练与评测。

**📈 对比分析**

与多种 SOTA 方法在 TST2K 和 PST50 上对比，ICTone 在颜色差异、深度色差、内容保持和美学得分上均优于对手，并在用户研究中获得最高胜率。

**⚠️ 局限性**

缺点包括对极端光照或纹理变化的鲁棒性仍有限，且构建数据集依赖预设与评分器，可能导致风格覆盖范围受限。

---

## 385. Reckoning with the Political Economy of AI: Avoiding Decoys in Pursuit of Accountability

**arXiv ID:** 2604.16106 | [PDF](https://arxiv.org/pdf/2604.16106v1)

**作者:** Janet Vertesi `[一作]` (Princeton University), Benjamin Shestakofsky `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过社会学、通信学、技术社会学与经济社会学理论，对人工智能（AI）项目中的“decoy”概念及其五类（本体、必然性、破坏性、安全性、监管性）进行阐述，并分析这些decoy如何掩盖AI项目的政治经济网络与权力结构。

**💡 创新点**

创新点在于将“decoy”视为AI项目中构建与维护网络权力的工具，并提出五类decoy框架，强调应关注AI项目的物质政治经济而非单纯技术与伦理讨论。

**🔧 技术方法**

研究方法主要是文献综述与案例分析，未使用具体技术实现；所依赖的理论来自社会学、通信学、技术社会学与经济社会学。

**📊 数据集**

本文没有使用任何机器学习数据集或实验数据，而是基于已有学术文献、行业报告与政策文件进行分析。

**📈 对比分析**

由于缺乏实验或性能评估，本研究不涉及方法比较或性能指标；其贡献在于概念与理论框架的提出，而非技术实现。

**⚠️ 局限性**

局限性包括：缺乏定量实证数据验证所提出的decoy框架；跨学科方法虽有讨论但未深入量化分析；对AI技术本身的技术细节与实现细节关注不足。

---

## 386. DenTab: A Dataset for Table Recognition and Visual QA on Real-World Dental Estimates

**arXiv ID:** 2604.16099 | [PDF](https://arxiv.org/pdf/2604.16099v1)

**作者:** Laziz Hamdi `[一作]` (LITIS), Thierry Paquet `[通讯]` (LITIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文创建并公开了DenTab数据集，该数据集包含2000张真实牙科估价表格图像及其高质量HTML结构+内容标注，并基于同一输入生成了2208个逻辑性强的TableVQA问题；

**💡 创新点**

其创新点在于：①首次提供真实场景噪声下的表格结构+内容标注数据；②设计了多类别TableVQA基准；③提出了Table Router Pipeline，将算术性问题路由到基于程序生成与规则执行的确定性计算；

**🔧 技术方法**

使用的技术包括视觉-语言模型（如Qwen、Gemma、Ministral）、OCR系统（Nanonets、olmOCR）、表格结构识别、程序生成（DSL）、规则式执行器；

**📊 数据集**

使用的数据集为DenTab（表格图像+HTML标注）以及其衍生的TableVQA问答；

**📈 对比分析**

与16种系统（14个VLM+2个OCR）在表格识别和TableVQA上进行零样本对比，最高表格识别得分为92.7% S‑TEDS，TableVQA整体最佳准确率为73.2%；Table Router Pipeline在算术类问题上显著提升（如Diff从约41%提升至约50%），整体提升至约80%；

**⚠️ 局限性**

局限性包括：仅覆盖牙科估价单一领域，缺乏多域通用性；Pipeline推理成本较高；即使使用oracle HTML，算术一致性问题仍难以完全解决。

---

## 387. Compositional Design, Implementation, and Verification of Swarms (Technical Report)

**arXiv ID:** 2604.16097 | [PDF](https://arxiv.org/pdf/2604.16097v1)

**作者:** Florian Furbach `[一作]`, Emilio Tuosto `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了面向局部优先（local-first）分布式系统的可组合性框架：对 Swarm 协议和 Swarm 进行分层式、可组合的规范、实现与验证；

**💡 创新点**

创新点在于：① 定义了新的可组合性 Well‑Formedness 和 interfacing 角色；② 设计了高效的订阅计算算法 “comp-swarm-subscriptions”；③ 引入了 branch‑tracking 语义和自动机器适配，保证在组合后仍能实现 eventual fidelity；

**🔧 技术方法**

采用了形式化语义（coinductive 语法、事件日志、指针指向上一次更新事件）、静态类型检查与投影算法、分布式日志传播模型，并在 Actyx 工具链中实现了上述理论；

**📊 数据集**

实验使用了 454 组随机生成的 Swarm 协议（最多 10 个协议，每个最多 9 个角色、每个角色最多 9 个事件），并对比了已知的 “exact” 订阅计算方法；

**📈 对比分析**

比较方法：将 comp-swarm-subscriptions 与 “exact” 订阅算法对同一协议组合进行运算；结果显示 comp-swarm-subscriptions 运行时间从 0.01 秒（10^5 过渡）到 10 秒（exact）显著更快；在订阅大小上，两者相差约 7%（29.9% vs 22.4%），说明 comp-swarm-subscriptions 近似最优且更具可扩展性；

**⚠️ 局限性**

局限性：仅支持顺序（无内部并发）协议的组合，无法处理具有内部并发的协议组合；目前不支持任意并发协议的嵌套组合，需要进一步扩展模块化系统和更精细的订阅计算方法。

---

## 388. GroupEnvoy: A Conversational Agent Speaking for the Outgroup to Foster Intergroup Relations

**arXiv ID:** 2604.16095 | [PDF](https://arxiv.org/pdf/2604.16095v1)

**作者:** Koken Hata `[一作]` (University of Tokyo), Yukino Baba `[通讯]` (University of Tokyo)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5010710732)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了利用 LLM 基于真实对话数据的 AI 代理（GroupEnvoy）进行的 AI‑mediated contact 在日本大学生与中国留学生情境中的心理效应。

**💡 创新点**

提出并验证了一种全新范式——AI‑mediated contact，将对外群体的真实观点通过 LLM 代理呈现给内群体，以减少焦虑并促进视角采纳。

**🔧 技术方法**

使用 Gemini‑3‑pro 大语言模型生成文本，并通过自定义 Web 接口实现即时的文本对话；同时对话数据被当作系统提示进行多轮生成。

**📊 数据集**

从三名中国留学生的一次对话会议中提取的 persona、提案与讨论日志三类数据作为 GroupEnvoy 的输入提示。

**📈 对比分析**

通过对照实验（交互式 AI 组 vs 静态文本组）结合混合 ANOVA 与质性模板分析评估；结果显示时间效应显著，AI 组在群体焦虑与视角采纳方面呈现中等至大效应量，但未达到统计显著。

**⚠️ 局限性**

主要局限包括样本量不足导致统计功效低、单次实验缺乏长期跟踪、AI 代理可能导致用户视角采纳被代理代替，以及未验证此范式在不同群体边界上的普适性。

---

## 389. Characterization of Real Communication Patterns and Congestion Dynamics in HPC Interconnection Networks

**arXiv ID:** 2604.16088 | [PDF](https://arxiv.org/pdf/2604.16088v1)

**作者:** Miguel Sánchez de La Rosa `[一作]` (Universidad de Castilla-La Mancha), Francisco J. Quiles `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对真实MPI应用生成的VEF网络通信轨迹进行静态与动态分析，以评估并识别不同网络拓扑和配置下的拥塞场景。

**💡 创新点**

将VEF框架扩展为完整的分析链路，结合SAURON模拟器进行动态性能评估，并公开共享大规模VEF轨迹库。

**🔧 技术方法**

VEF-Prospector、VEF-TraceLib、OMNeT++/SAURON模拟器、静态分析脚本与可视化GUI。

**📊 数据集**

NEST、GROMACS、LAMMPS、PATMOS的VEF轨迹（64/256个MPI进程）来自欧盟欧普项目收集的真实集群运行。

**📈 对比分析**

通过比较两种网络配置（#1 400 Gbps/大缓冲/可变MTU vs #2 100 Gbps/小缓冲/固定MTU）以及Fat‑Tree与Megafly拓扑，使用执行时间、平均与最大FCT以及CDF等指标，发现配置#1在大多数应用中平均FCT显著更低，仅LAMMPS在#1上有明显加速。

**⚠️ 局限性**

依赖于真实集群采样，难以扩展到大于可用核心数的MPI工作负载；仅覆盖MPI场景，对非MPI或多任务映射未完整支持；实验仅限于两种拓扑与两种网络配置，未探讨自适应路由或更大规模系统。

---

## 390. Stylistic-STORM (ST-STORM) : Perceiving the Semantic Nature of Appearance

**arXiv ID:** 2604.16086 | [PDF](https://arxiv.org/pdf/2604.16086v1)

**作者:** Hamed Ouattara `[一作]` (Cerema), Omar Ait Aider `[通讯]` (Universite Clermont Auvergne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的自监督学习框架 ST-STORM，能够同时学习物体内容和外观（style）两种语义子空间，并通过显式解耦实现内容与风格的分离与重用。

**💡 创新点**

创新点包括：① 将外观视为可分离的语义模态；② 采用 U‑Net + SPADE 结构在解码器上显式注入风格；③ 设计 Style‑JEPA 预测约束，过滤偶发细节，只保留可预测的风格特征；④ 通过伪域（pseudo‑domain）循环和风格银行实现风格扰动与内容不变性的双重约束；⑤ 在对比学习中引入风格正样本，提升内容分辨率。

**🔧 技术方法**

核心技术：自监督对比学习（MoCo）、预测编码（JEPA、Style‑JEPA）、GAN 对抗风格生成（PatchGAN）、频域对齐（FFT、SWD）、多尺度风格编码（Pyramid+SPADE）、多任务头与自适应融合。

**📊 数据集**

主要实验数据集：Weather‑MultiTask‑Datasets（≈250k 无标签，25k 标注测试）、ISIC‑2024 肿瘤检测（≈400k 预训练，10k 测试）、ImageNet‑1K、以及外部评估集 MWI、WEAPD+MWD、Oxford‑Flowers‑102、Oxford‑IIIT‑Pets、CIFAR‑100。

**📈 对比分析**

与 MoCo‑v3、I‑JEPA 等主流 SSL 方法对比：在天气细粒度分类和黑色素瘤检测等外观驱动任务中，Style 分支的 F1 分别提升至 97% 与 94%（比 MoCo‑v3 提升 5–6%）；在 ImageNet‑1K 这类以内容为主的任务中，Content 分支保持竞争力（Top‑1 ≈75%），并通过 Style‑Content 融合进一步提升至 78%（比 MoCo‑v3 提升 3%）。外域迁移实验亦表明 ST‑STORM 的表示更具泛化性。

**⚠️ 局限性**

局限性：① 训练成本高，需同时维护生成器、判别器与多项约束；② 对风格扰动的依赖使得在数据稀缺或风格种类有限的场景中效果可能受限；③ 目前评估聚焦于分类任务，尚未验证在分割、检测等更结构化任务中的表现；④ 伪域划分与调度仍需经验选择，可能影响风格分离质量。

---

## 391. Towards Universal Convergence of Backward Error in Linear System Solvers

**arXiv ID:** 2604.16075 | [PDF](https://arxiv.org/pdf/2604.16075v1)

**作者:** Michał Dereziński `[一作]` (University of Michigan), Elizaveta Rebrova `[通讯]` (Princeton University)

**通讯引用:** 189 | [OpenAlex ID](https://openalex.org/A5054763583)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文提出并分析了在后向误差（backward error）度量下线性方程组求解的无条件收敛率，证明 Richardson 对 PSD 系统后向误差满足 1/k 收敛。

**💡 创新点**

创新点在于：①证明 Richardson 对 PSD 系统后向误差 1/k 收敛；②提出 MINBERR 算法，在 Krylov 子空间中直接最小化后向误差，实现 O(1/k²) 的通用收敛率；③将该思路推广到一般线性系统，得到 MINBERR-NE，并通过随机扰动实现对条件数无依赖的收敛。

**🔧 技术方法**

采用的技术包括后向误差理论、Krylov 子空间方法（Lanczos、Arnoldi）、Chebyshev 多项式逼近、SVD 与逆迭代、以及随机扰动/光滑分析（smoothed analysis）。

**📊 数据集**

实验使用 SuiteSparse Matrix Collection 中的 6 个真实矩阵：3 个正半定（PSD）问题（尺寸 14822、16146、19779），3 个非 PSD 问题（尺寸 5005、6747、13681）。

**📈 对比分析**

与 Richardson、CG、MINRES、LSQR、LSMR 进行对比。对于 PSD 系统，MINBERR 的后向误差收敛曲线明显快于基线，几乎达到理论 1/k² 上界；对非 PSD 系统，MINBERR 不出现 Richardson 等方法的初始误差爆炸，收敛速率约为 1/k，且通过扰动后对条件数几乎不敏感。

**⚠️ 局限性**

局限性：①对非 PSD 系统仍需使用正规方程或随机扰动，无法在理论上保证无条件收敛；②缺乏有限精度下的后向误差稳定性分析；③实现依赖 Lanczos/SVD，计算成本在极大稀疏矩阵上可能不够高效。

---

## 392. VLSF Decoding with Reliability Guarantees over Correlated Noncoherent Fading Channels

**arXiv ID:** 2604.16062 | [PDF](https://arxiv.org/pdf/2604.16062v1)

**作者:** Guodong Sun `[一作]` (Inria), Jean-Marie Gorce `[通讯]` (Inria)

**通讯引用:** 3442 | [OpenAlex ID](https://openalex.org/A5054469607)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在具有时间相关非协同衰落信道上，针对可变长度停止反馈(VLSF)码的可靠性保证解码方法；通过对信息密度的可计算上下界进行统一时间点的分析，构造了可用于实际停止决策的下界；并在Gauss‑Markov衰落模型下用高斯调制进行数值验证；

**💡 创新点**

创新点在于：①在非协同、带记忆的衰落信道上首次给出信息密度的可计算统一上下界；②将Rényi相对熵与Hölder不等式结合，得到既保留内存效应又可直接用于停止规则的下界；③通过对下界与上界的比较，量化松弛误差并指导参数选择；

**🔧 技术方法**

使用了Rényi相对熵、Hölder不等式、改变测度技巧、Gauss‑Markov过程分析、信息密度与停止时刻理论、以及高斯调制与高斯参考衰落模型的计算；

**📊 数据集**

实验采用的是合成的Gauss‑Markov衰落信道（ρ=0.3、SNR=100 dB、M=1024、误差目标ε=10⁻³）进行 Monte‑Carlo 仿真；

**📈 对比分析**

与固定块长度方案对比，VLSF 方案在相同误差目标下能够提前停止，减小平均传输时延；数值实验显示在n=50时下界与上界的差距约线性随n增长，说明参数优化后下界已较为紧密；

**⚠️ 局限性**

局限性包括：①下界的松弛主要由Rényi罚项和Hölder包络产生，尚未达到最优；②仅考虑高斯调制与高斯参考衰落，可能无法适用于其他调制或非高斯衰落；③参数(r,σ_h²) 的优化依赖经验搜索，缺乏理论最优解；

---

## 393. Prototype-Grounded Concept Models for Verifiable Concept Alignment

**arXiv ID:** 2604.16076 | [PDF](https://arxiv.org/pdf/2604.16076v1)

**作者:** Stefano Colamonaco `[一作]` (Ku Leuven), Giuseppe Marra `[通讯]` (Ku Leuven)

**通讯引用:** 509 | [OpenAlex ID](https://openalex.org/A5005466305)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Prototype-Grounded Concept Models（PGCMs），通过在每个概念中引入可视化原型来实现概念的视觉可解释性；

**💡 创新点**

创新点在于：①将概念与低层视觉原型直接绑定，形成可检验的概念对齐表；②支持在原型级别进行模型编辑和干预，提升可解释性与可操作性；③在保持与CBM相当的预测性能的同时，显著提升概念可视化与干预效率；

**🔧 技术方法**

使用了分割网络提取图像局部、原型选择器（基于相似度的softmax）、图像解码器与概念解码器（多模态解码）、ELBO训练框架以及对原型嵌入的可视化约束；

**📊 数据集**

在CelebA（人脸属性）、ColorMNIST+（彩色手写数字及求和任务）以及CLEVR-Hans（物体属性与几何关系）三个公开数据集上进行评估；

**📈 对比分析**

与传统CBM、CRM、CMR以及直接的DNN进行对比，PGCM在概念与任务准确率上与最先进的CBM相近（CelebA任务准确率略低），且在概念干预与模型编辑后性能提升显著；

**⚠️ 局限性**

局限性包括：①原型数量与模型容量存在权衡，原型不足会导致准确率下降；②对概念标签和分割模型的依赖；③虽然可解释性增强，但整体预测性能仍略逊于纯深度网络；

---

## 394. Sentiment Analysis of German Sign Language Fairy Tales

**arXiv ID:** 2604.16138 | [PDF](https://arxiv.org/pdf/2604.16138v1)

**作者:** Fabrizio Nunnari `[一作]` (German Research Center for Artificial Intelligence), Patrick Gebhard `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将LLM对德语童话文本进行情感投票，并利用MediaPipe提取面部和身体运动特征，训练可解释的XGBoost模型来预测德语手语视频的正、中、负情感。

**💡 创新点**

首次系统性地结合面部与身体特征，并使用可解释模型揭示德语手语情感表达中面部与身体双重作用。

**🔧 技术方法**

采用GPT5、Sonic、Gemini Flash、GPT OSS 20B等LLM进行文本情感投票，使用MediaPipe Holistic和Face Landmarker提取面部Blendshape与骨架标记，再用XGBoost分类器进行情感预测并分析特征重要性。

**📊 数据集**

使用DGS-Fabeln-1语料（7部德国童话共574段，517段有视频），扩展为DGS-Fabeln-1-SE，包含LLM情感标签和MediaPipe提取的视频特征。

**📈 对比分析**

通过5折Stratified Group交叉验证评估，平均平衡准确率63.1%、宏F1 63.5%，与文本情感分析相比，Pearson相关系数约为0.529，显示出一定的性能水平。

**⚠️ 局限性**

未使用人工视频标注，LLM情感标签可能缺乏可信度；角色扭转与面部/身体姿态偏倚；仅使用正面视角和MediaPipe，缺乏更精确的姿态估计。

---

## 395. Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs Coding the Interviews of Firearm Violence Survivors

**arXiv ID:** 2604.16132 | [PDF](https://arxiv.org/pdf/2604.16132v1)

**作者:** Jessica H. Zhu `[一作]` (University of Maryland), Joseph B. Richardson `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用开放源代码LLM（Llama 1B和8B）自动化编码21名黑人枪击幸存者的长篇访谈；

**💡 创新点**

首次系统评估低资源LLM在黑人枪击幸存者定性编码中的效能与偏见，揭示大规模模型对AAE语言的误读与叙事抹除；

**🔧 技术方法**

采用零射提示、语义嵌入（Sentence Transformers）、BERTopic聚类与余弦相似度评估的完整自动编码管道；

**📊 数据集**

利用2013–2015年华盛顿特区地区收集的21名黑人男性枪击幸存者访谈记录；

**📈 对比分析**

将机器生成代码与人工编码（人类编码）对比，使用“Percent Captured”和“Percent Relevant”两指标；机器初始代码平均捕获率27%，相关率10%，聚类后代码数降至≈60，但准确率仍低于人工编码；

**⚠️ 局限性**

受限于模型偏见导致的叙事抹除、低相关性、需人工验证、计算资源限制、缺乏多地区AAE样本、仅使用Llama家族未尝试更大或量化模型。

---

## 396. From Articles to Canopies: Knowledge-Driven Pseudo-Labelling for Tree Species Classification using LLM Experts

**arXiv ID:** 2604.16115 | [PDF](https://arxiv.org/pdf/2604.16115v1)

**作者:** Michał Romaszewski `[一作]` (Institute of Theoretical and Applied Informatics Polish Academy of Sciences), Anna Jarocińska `[通讯]` (University of Warsaw)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5083657438)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于多传感器（HSI+ALS）的遥感数据，提出了结合生物学知识的半监督深度学习方法，对森林树种进行像素级分类。

**💡 创新点**

创新点在于通过大语言模型自动生成物种共居矩阵作为生态先验，结合伪标签扩展训练集，并采用双流神经网络融合多模态特征。

**🔧 技术方法**

使用了双流多层感知机网络、伪标签策略、共居矩阵约束、LLM知识抽取以及传统的特征提取与CatBoost分类器。

**📊 数据集**

使用了位于Wigry国家公园的ALS和HySpex HSI数据集，包含约942棵标注树种，覆盖约158.3 km²。

**📈 对比分析**

与经典CatBoost+特征工程方法比较，DSNN+P在宏观F1上提升至约79.4%，比最佳参考方法高约5.6%，准确率约90%。

**⚠️ 局限性**

主要局限在标签稀缺、共居矩阵依赖LLM的可靠性以及对稀有物种的分类仍易受限。

---

## 397. Toward EU Sovereignty in Space: A Comparative Simulation Study of IRIS 2 and Starlink

**arXiv ID:** 2604.16092 | [PDF](https://arxiv.org/pdf/2604.16092v1)

**作者:** Alexander Bonora `[一作]` (University of Padova), Michele Zorzi `[通讯]` (University of Padova)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过仿真方法对比分析了SpaceX Starlink与欧盟计划的IRIS^2卫星星座在容量、覆盖、切换与可靠性等指标上的性能差异；

**💡 创新点**

创新点在于提出了统一的系统模型（轨道、信道、关联）以在相同假设下评估商业与公共卫星网络，并探讨了负载均衡关联对IRIS^2性能的提升；

**🔧 技术方法**

使用了3GPP NR‑NTN信道模型、Walker Δ轨道模型、TLE数据、光纤OISL、SDN网络切片、卫星重构载荷等技术；

**📊 数据集**

仿真所用数据主要为公开的TLE轨道信息、星座设计参数（卫星数、轨道高度、波束数量）以及基于3GPP的信道与容量参数；

**📈 对比分析**

比较方法包括：计算单元总下行/上行容量、每UE容量、可视时间与切换率；结果显示Starlink在总容量与覆盖率上显著优于IRIS^2，但IRIS^2在单UE容量、切换频率与主权可控性方面更具优势；

**⚠️ 局限性**

局限性在于仅基于仿真，未结合真实测量数据；假设UE静止、流量模型简化；对极端极端极端情况（如大规模灾难或攻击）未做深入评估；

---

## 398. Robust Synchronisation for Federated Learning in The Face of Correlated Device Failure

**arXiv ID:** 2604.16090 | [PDF](https://arxiv.org/pdf/2604.16090v1)

**作者:** Stefan Behfar `[一作]` (University of Cambridge), Richard Mortier `[通讯]` (University of Cambridge)

**通讯引用:** 6609 | [OpenAlex ID](https://openalex.org/A5043629070)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可用性加权的概率同步并行（AW-PSP）客户端选择框架，解决联邦学习中设备可用性与数据分布相关性导致的采样不公平与标签缺失问题。

**💡 创新点**

创新点在于：① 用马尔可夫链实时预测节点可用性，区分短暂与长期失效；② 结合节点历史成功率、恢复概率与实时关联故障风险，构造可用性+恢复+关联惩罚的综合采样权重；③ 通过分布式哈希表（DHT）共享邻近度与延迟信息，去中心化地评估节点可靠性与多样性。

**🔧 技术方法**

使用技术包括：马尔可夫链可用性预测、指数加权移动平均（EWMA）、DHT Overlay、关联故障相关系数、平均内类方差/类均值方差、KL散度、Gini系数等公平性度量；模型采用ResNet‑18/34，数据集为CIFAR‑10；实验通过逻辑客户端模拟大规模（100–3000）联邦环境并注入真实可用性轨迹。

**📊 数据集**

使用的数据集为CIFAR‑10（训练/测试），模型为ResNet‑18/34；可用性与故障相关性基于公开的Google FL设备可用性日志（包含充电、Wi‑Fi、睡眠等事件）。

**📈 对比分析**

与 Classic‑PSP（均匀随机采样）和 Oort 进行对比。AW‑PSP 在整体准确率上略优或持平，关键表现为：① 平均类方差和类均值方差显著降低，公平性提升；② KL 散度和未覆盖类数几乎为零，标签覆盖更完整；③ 在相关故障注入下，准确率下降幅度比 Classic‑PSP 小，鲁棒性更好；⑥ 在 3000 节点规模实验中，准确率与公平性指标保持稳定，说明可扩展性良好。

**⚠️ 局限性**

局限性：① 仍依赖可用性预测模型的准确性，对极端网络抖动或长时间失效场景未充分覆盖；② DHT 需要额外的网络维护与一致性保证，可能在极大规模下产生额外延迟；③ 评估主要基于模拟与公开轨迹，缺少大规模真实设备实验；④ 对模型收敛理论的数学证明仍不完整。

---

## 399. Veritas-RPM: Provenance-Guided Multi-Agent False Positive Suppression for Remote Patient Monitoring

**arXiv ID:** 2604.16081 | [PDF](https://arxiv.org/pdf/2604.16081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 400. Unveiling Stochasticity: Universal Multi-modal Probabilistic Modeling for Traffic Forecasting

**arXiv ID:** 2604.16084 | [PDF](https://arxiv.org/pdf/2604.16084v1)

**作者:** Weijiang Xiong `[一作]` (École Polytechnique Fédérale de Lausanne), Nikolas Geroliminis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 12972 | [OpenAlex ID](https://openalex.org/A5075389676)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在已有的时空交通预测模型上，将最终输出层替换为高斯混合模型（GMM）层，使模型能够输出多模态概率分布，并在相同的训练管道下进行训练。

**💡 创新点**

提出了一个通用且极简的模块化 GMM 预测层，无需改动网络结构或训练流程；通过多模态分布提高不确定性刻画，配合全流程评估指标展示显著性能提升与鲁棒性。

**🔧 技术方法**

核心技术为 GMM 输出层、负对数似然（NLL）损失、连续累积分布评分（CRPS）以及平均宽度/校准误差（mAW、mCCE）；实验基于多种时空网络（GWNet、MTGNN、STID、STAEformer、LGC）进行验证。

**📊 数据集**

使用公开交通速度数据集 METR-LA、PEMS-Bay、SimBarcaSpd；并在 SimBarca 高频数据上评估不同数据质量对模型的影响。

**📈 对比分析**

通过与确定性（Det）和单峰正态（Norm）变体对比，采用 CRPS、mAW、mCCE、MAE、MAPE、RMSE 等指标，GMM 版本平均在 CRPS 上提升约 27%，并在置信区间宽度与校准度上优于单峰模型；在数据缺失或低分辨率情形下仍保持更小的性能损失。

**⚠️ 局限性**

假设每个位置、时间点的输出为独立的一维 GMM，可能无法捕捉跨位置的联合分布；组件数固定（K=5）限制了对极端多模态的建模；对高分辨率大规模网络的计算成本仍需进一步评估；在近似为单峰的场景下 GMM 可能出现过度自信或欠拟合。

---

## 401. DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics

**arXiv ID:** 2604.16083 | [PDF](https://arxiv.org/pdf/2604.16083v1)

**作者:** Jieming Yu `[一作]` (Hong Kong University of Science and Technology), Xiaochen Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了基于 DINOv3 的轻量级图像篡改检测与定位基线，并使用 LoRA 适配和三层卷积解码器。

**💡 创新点**

证明即使极简架构也能通过冻结的自监督 ViT 特征与 LoRA 低秩适配，在低数据量和多任务场景下显著提升性能，并保持稳健性。

**🔧 技术方法**

使用 DINOv3 ViT‑S/B/L、LoRA 低秩适配（r=32/64）、轻量卷积解码器、以及带边缘权重的 BCE+边缘损失。

**📊 数据集**

训练数据来自 CAT‑Net 多源组合（CASIA2、FantasticReality、IMD2020、TampCOCO）和 MVSS‑Net 单源 CASIA2；评测四/五个标准数据集（CASIAv1、Columbia、NIST16、Coverage、IMD2020）。

**📈 对比分析**

在 CAT‑Net 协议下，最佳配置平均 F1 提升 17.0 分；在 MVSS‑Net 下 LoRA ViT‑L 平均 F1 为 0.774，远超前沿方法 TruFor 的 0.530；LoRA 的参数量比全微调少 11–34 倍，且在低样本、噪声、模糊、JPEG 变形等后处理下更稳健。

**⚠️ 局限性**

局限性包括未验证对更广泛操纵类型、极端后处理或更高分辨率图像的泛化；模型推理速度与部署成本仍待进一步研究。

---

## 402. Deterministic Task Offloading and Resource Allocation in the IoT-Edge-Cloud Continuum

**arXiv ID:** 2604.16155 | [PDF](https://arxiv.org/pdf/2604.16155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 403. Early Detection of Acute Myeloid Leukemia (AML) Using YOLOv12 Deep Learning Model

**arXiv ID:** 2604.16082 | [PDF](https://arxiv.org/pdf/2604.16082v1)

**作者:** Enas E. Ahmed `[一作]` (Cairo University), Mayar Moner `[通讯]` (Cairo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了基于YOLOv12的AML细胞多类别分类系统，包含图像分割、模型训练与评估。

**💡 创新点**

采用Otsu阈值和色相通道双分割方案；在细胞层面利用YOLOv12并结合Area Attention提升检测精度；在平衡数据集上实现99.3%准确率。

**🔧 技术方法**

YOLOv12（带Area Attention），Hue通道分割，Otsu阈值分割，ResNet50/Inception-ResNet50 v2对比，迁移学习，标准评估指标。

**📊 数据集**

Kaggle的增强版血细胞图像数据集，包含5类细胞，训练70%/验证15%/测试15%，共5000张。

**📈 对比分析**

分别对四种分割（细胞Hue、细胞Otsu、核Hue、核Otsu）训练YOLOv12，并与ResNet50、Inception等模型对比，细胞Otsu方案获得最高验证/测试准确率99.3%，其余约98.8%。

**⚠️ 局限性**

数据集虽平衡但仅来自公开来源，未涵盖真实临床样本多样性；仅评估单一模型架构，未探索多模态或更深网络；缺乏独立外部验证集，结果可能存在过拟合风险。

---

## 404. Finding Patient Zero via Low-Dimensional Geometric Embeddings

**arXiv ID:** 2604.16074 | [PDF](https://arxiv.org/pdf/2604.16074v1)

**作者:** Stefan Huber `[一作]` (Salzburg University of Applied Sciences), Dominik Kaaser `[通讯]` (TU Hamburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用Johnson-Lindenstrauss投影将传播网络映射至低维欧氏空间，通过计算感染节点中心的几何距离来推断源节点。

**💡 创新点**

首次将压缩投影技术与几何中心估计结合，用低维表示保留源信息，解决部分观测下的源推断问题。

**🔧 技术方法**

Johnson-Lindenstrauss投影、BFS距离签名、中心质心估计、欧氏距离评分。

**📊 数据集**

Erdős-Rényi随机图 G(n,p)（n=10^4，p≈10/n）作为实验网络。

**📈 对比分析**

与完整信息基准对比，平均源节点排名在不同 p、β、k 下随参数变化；在 k≈log n 时达到最低排名，性能随感染概率提升而改善。

**⚠️ 局限性**

仅在感染概率足够大时能产生足够传播信息；在极稀疏或极稠密网络、低感染概率下精度下降；实验仅限于合成图，未验证真实网络。

---

## 405. TableSeq: Unified Generation of Structure, Content, and Layout

**arXiv ID:** 2604.16070 | [PDF](https://arxiv.org/pdf/2604.16070v1)

**作者:** Laziz Hamdi `[一作]` (LITIS), Thierry Paquet `[通讯]` (LITIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TableSeq，一种仅基于图像的端到端统一生成表格结构、内容和单元格位置的序列化模型。

**💡 创新点**

将表格结构、内容和几何位置合并到同一自回归序列，利用结构先验头与关键偏置实现结构-位置的天然对齐，并通过轻量级高分辨率CNN + 单层Transformer实现模型简洁；同时通过多token预测加速推理。

**🔧 技术方法**

轻量化FCN‑H16编码器、单层Transformer编码器、结构先验头、key‑biased cross‑attention、离散坐标词表、synthetic‑to‑real curriculum、token‑level交叉熵训练、MTP等。

**📊 数据集**

PubTabNet、FinTabNet、SciTSR、PubTables‑1M，以及在ICDAR 2013的零样本迁移实验。

**📈 对比分析**

与Split‑and‑Merge、Bottom‑Up、Image‑to‑Sequence等多类基线对比；在PubTabNet达95.23 TEDS/96.83 S‑TEDS，FinTabNet 97.45/98.69，SciTSR CAR精确度/召回率/F1 99.79/99.54/99.66；在PubTables‑1M GriTS_Top 99.10、Con 98.82、Loc 95.63；在ICDAR 2013零样本结构识别中F1 96.60；在索引查询中IRDR 67.73% 为最高。

**⚠️ 局限性**

仍为自回归，推理时间随序列长增长；离散坐标可能导致边界细节误差；对强旋转或非矩形表格支持不足；依赖公开数据集，未评估手写、低分辨率或多页文档情况。

---

## 406. Fast and Memory Efficient Multimodal Journey Planning with Delays

**arXiv ID:** 2604.16149 | [PDF](https://arxiv.org/pdf/2604.16149v1)

**作者:** Denys Katkalo `[一作]` (Igor Sikorsky Kyiv Polytechnic Institute), Toby Walsh `[通讯]` (University of New South Wales)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对公共交通多模态行程规划的延迟感知算法进行改进，提出了停止级短路投影以提升内存效率和查询速度。

**💡 创新点**

创新点在于将事件级短路投影到停靠站层面，显著压缩短路集合（48–180×），并通过替换更新消除原 D‑ULTRA‑TB 的错误，形成无误差、速度更快的查询方案。

**🔧 技术方法**

使用 ULTRA 与 D‑ULTRA 的短路预计算、事件级到停靠站级的映射、CSA、RAPTOR、Early Pruning、CH（Core‑CH、Bucket‑CH）等技术。

**📊 数据集**

实验数据集为伦敦（London）和瑞士（Switzerland）两大真实公共交通网络。

**📈 对比分析**

与 MR、TAD、HL‑RAPTOR/HL‑CSA、TD‑Dijkstra 等基线算法比较，单目标查询中 D‑ULTRA‑CSA 速度最快，达到 1.9–4.2× 的加速；多目标查询中 D‑ULTRA‑RAPTOR(EP) 在非零延迟缓冲区下优于 D‑ULTRA‑TB，误差率降至 0%。

**⚠️ 局限性**

局限性包括：仍需先生成庞大的事件级短路集再投影，且在无替换更新时某些情形下仍可能出现极少错误；目前仅支持两种评估标准，未扩展到更复杂的多目标（如换乘时间）。

---

## 407. Tabular foundation models for in-context prediction of molecular properties

**arXiv ID:** 2604.16123 | [PDF](https://arxiv.org/pdf/2604.16123v1)

**作者:** Karim K. Ben Hicham `[一作]` (RWTH Aachen University), Alexander Mitsos `[通讯]` (JARA-CSD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究将Tabular Foundation Models（TFMs）与冻结的分子表征（如CheMeleon嵌入、RDKit2d、Mordred等）相结合，采用in‑context学习方式完成分子性质预测，既不需要任务特定的微调，也能在小到中等数据集上实现优于传统机器学习和已微调的分子基础模型的表现。

**💡 创新点**

创新点在于：①首次将TFMs迁移到分子属性预测领域，利用其在合成任务上的预训练实现对新任务的即时推理；②证明冻结的分子嵌入与TFMs配合能获得与微调模型相当甚至更优的预测精度；③通过两步流程（先生成高质量分子特征，再用TFM预测）显著降低计算成本和专业门槛。

**🔧 技术方法**

核心技术包括 TabPFN 和 TabICL 这两类基于 Transformer 的 Tabular Foundation Models；分子表征技术有 CheMeleon、SMI‑TED、CLAMP 预训练嵌入，以及传统的 RDKit2d、Mordred 描述符和 Morgan 指纹；在评估中还对 XGBoost、CatBoost、CheMeleon 及其他基线模型进行了比较。

**📊 数据集**

使用了 58 个公开基准（Polaris 28 组、MoleculeACE 30 组）以及 11 个化工实际数据集（燃料点火性能、聚合物性质、聚合物‑溶剂相互作用），这些数据集覆盖了 0.1k–6k 条样本的低至中等数据量场景。

**📈 对比分析**

对比方法包括传统机器学习基线（XGBoost、CatBoost、随机森林）以及已微调的分子基础模型（CheMeleon、Minimol、MolFormer 等）。在所有基准上，TabPFN‑CheMeleonFP 取得最高 86.2% 的胜率（MoleculeACE 上 100%），平均排名仅为 4.52；与微调 CheMeleon 相比，预测精度提升约 1–2 个百分点，同时在 CPU 上实现 4.8–27.3 倍、GPU 上 18.3–46.0 倍的速度提升。

**⚠️ 局限性**

主要局限包括：①对大型数据集（>10k 条样本）未充分验证，TFM 在高维特征下易出现内存/磁盘瓶颈；②仅针对单分子预测，未扩展到多分子或多任务场景；③未对 TFMs 进行超参数调优，仅使用默认配置；④缺乏对结构化物理约束或多任务学习的深度探索。

---

## 408. Sample Complexity Bounds for Stochastic Shortest Path with a Generative Model

**arXiv ID:** 2604.16111 | [PDF](https://arxiv.org/pdf/2604.16111v1)

**作者:** Jean Tarbouriech `[一作]` (Facebook), Alessandro Lazaric `[通讯]` (Facebook)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出在可生成模型（generative oracle）下求解随机最短路径（SSP）问题的PAC算法，并给出两种情形（c_min>0 与 c_min=0）的样本复杂度上界。

**💡 创新点**

首次将SSP的样本复杂度分析与variance‑aware技术结合，显著降低对最优值范围 B_* 与最小成本 c_min 的依赖，并为零最小成本情况引入受限策略集合及成本扰动。

**🔧 技术方法**

使用 SSP 版模拟引理、扩展值迭代（EVI）、方差敏感 Bernstein 置信区间、策略倍增估计、成本扰动与 SSP 圆顶数（diameter）估计等方法。

**📊 数据集**

未使用公开数据集，研究完全基于理论分析和假设模型。

**📈 对比分析**

与 DMDP 与有限时限 MDP 的已知下界对比，证明在 c_min=1 时样本复杂度达到最优下界；在 c_min>0 时降低了 O(B_*/c_min) 乘子；在 c_min=0 时实现了 O(ε⁻³) 的样本复杂度。

**⚠️ 局限性**

样本复杂度仍包含 ΓS 的依赖（最大分支因子×状态数），难以降至线性；对于 c_min 接近 0 时仍需额外的 SSP 圆顶数估计与策略限制，且目前无法给出对应的下界，尚需进一步研究。

---

## 409. Polyglot: Multilingual Style Preserving Speech-Driven Facial Animation

**arXiv ID:** 2604.16108 | [PDF](https://arxiv.org/pdf/2604.16108v1)

**作者:** Federico Nocentini `[一作]` (University of Florence), Akin Caliskan `[通讯]` (Flawless AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Polyglot，一种基于扩散模型的多语言语音驱动面部动画框架，能够同时捕捉语言特性和个人说话风格。

**💡 创新点**

创新点在于：①不依赖预先定义的语言标签，而是利用语音转录文本的 CLIP 嵌入来表示语言；②通过自监督风格自编码器提取个体说话风格向量；③在扩散过程的条件空间中联合注入语言、身份、风格和时间信息，实现跨语言、跨说话者的高保真动画。

**🔧 技术方法**

主要技术包括：多语言语音编码器 mHuBERT、Whisper ASR、CLIP 文本编码、3DMM 表情参数表示、Transformer‑based 反向扩散网络、风格自编码器、classifier‑free 引导、风格保持损失等。

**📊 数据集**

使用自构建的 PolySet 数据集：从 MultiTalk 视频中提取 20 种语言（共 10,000 句）经过音质和 3D 关键点过滤，得到 16 小时的高质量 3DMM 表情参数序列。

**📈 对比分析**

与 FaceFormer、SelfTalk、DiffPoseTalk、MultiTalk、S‑Faceformer、S‑DiffPoseTalk 等 SOTA 方法在 PolySet 上对比，Polyglot 在 LVE、MVE、DTW、MOD 四个指标上均优于对手；用户研究显示在唇形同步和自然度上对比大部分基线更受欢迎。

**⚠️ 局限性**

局限性：①依赖语音转录和 CLIP 文本嵌入，语言覆盖受限于 Whisper 与 CLIP 的支持；②模型规模大、推理成本高；③仅关注面部表情，未同时建模情绪、头部姿态等多模态因素；④在极端方言或噪声环境下表现尚待验证。

---

## 410. AEGIS: Anchor-Enforced Gradient Isolation for Knowledge-Preserving Vision-Language-Action Fine-Tuning

**arXiv ID:** 2604.16067 | [PDF](https://arxiv.org/pdf/2604.16067v1)

**作者:** Guransh Singh `[一作]` `[通讯]` (Independent Researcher), Guransh Singh (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 AEGIS 的梯度隔离框架，用来在不破坏预训练视觉‑语言模型（VLM）的 VQA 能力的前提下，直接使用连续流匹配（flow‑matching）梯度对其进行机器人控制任务的微调。

**💡 创新点**

创新点在于：①用预训练 VLM 的激活分布构造静态的 Wasserstein‑2 高斯锚点；②在每个 transformer 层使用 Gram–Schmidt 正交投影，精准消除任务梯度与锚点恢复梯度之间的破坏性投影，仅丢失不到 1% 的梯度能量；③不依赖重放缓冲、共训练数据或离散动作 token，实现完全无缓冲的梯度手术。

**🔧 技术方法**

技术实现包括：静态 Wasserstein‑2 统计锚点、Bures 公式的激活对齐惩罚、双向后向传播提取任务与锚点梯度、层级正交投影、以及与标准 Adam 优化器的配合。

**📊 数据集**

使用的数据集有：PaliGemma2‑3B‑Mix‑224 作为预训练 VLM；LIBERO 作为机器人操控示例；VQA‑v2（100 个测试样本）和 OK‑VQA（5k OOD 子集）用于评估 VQA 迁移效果。

**📈 对比分析**

与三种基线对比：无梯度隔离的直接微调、带 stop‑gradient + FAST 的离散动作方案、以及低秩适配 LoRA。AEGIS 在 1500 步训练后保持 VQA 损失与预训练基线几乎一致（0.374 vs 0.392），而其他方法会出现 0.384 的显著升高；在 OOD OK‑VQA 上准确率 60.23%，与基线 60.15% 基本相同，明显优于 LoRA（59.32%）和直接微调（57.36%）。

**⚠️ 局限性**

局限性包括：需要一次额外的反向传播和图保持，训练时延约 +40%；缺乏闭环任务成功率评估，仅在梯度和激活层面验证；对极大规模模型或多任务场景的可扩展性尚未彻底验证。

---

## 411. Constant-Factor Approximations for Doubly Constrained Fair k-Center, k-Median and k-Means

**arXiv ID:** 2604.16061 | [PDF](https://arxiv.org/pdf/2604.16061v1)

**作者:** Nicole Funk `[一作]` (University of Cologne), Sarah Sturm `[通讯]` (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

本文提出了在一般度量空间中同时满足群体公平性（Cluster 内颜色比例限制）和中心多样性（每个颜色的中心数限制）的离散 k‑聚类算法。

**💡 创新点**

创新点在于：①首次把两种公平约束在同一次算法中实现；②使用模块化的 LP 重定向与流网络取整技术，避免了先后顺序的交叉影响；③实现了 k‑center 的 4‑近似、k‑median 的 10.081‑近似、k‑means 的 325‑近似（欧氏空间可降至 101‑近似）等常数因子近似，并保持 additive 2 的群体公平偏差。

**🔧 技术方法**

核心技术：线性规划求解群体公平的分数解；对分数解做重定向（rerouting）以仅使用多样性中心；通过最大流或最小成本流将分数分配取整，同时保持公平约束。

**📊 数据集**

本文未使用实验数据集，全部结果基于理论分析与已有近似算法的组合。

**📈 对比分析**

与之前的工作相比，本方法在 k‑center 上将近似比从 8 降至 4，在 k‑median 与 k‑means 上首次给出常数因子近似；在实现上与已知的单一约束算法相比，保持了相近的时间复杂度（多项式）。

**⚠️ 局限性**

局限性：①仍存在 additive 2 的公平偏差；②算法是顺序化的（先求分数解后重定向），未直接同时处理两约束；③缺乏实验验证与最优下界，无法证明近似因子最优。

---

## 412. Halfspace separation in geodesic convexity

**arXiv ID:** 2604.16159 | [PDF](https://arxiv.org/pdf/2604.16159v1)

**作者:** Niranjan Nair `[一作]` `[通讯]`, Niranjan Nair

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图的几何凸性下的半空间分离问题，并证明在弱桥图、伪模图和基图（matroid basis graph）上可多项式求解。

**💡 创新点**

首次将三步法与2-SAT公式结合，给出针对上述三类图的全局高效求解与枚举方法。

**🔧 技术方法**

采用凸性空间理论、弱桥/伪模/基图的结构性质、三步缩减、2-SAT约束化与图的几何条件。

**📊 数据集**

未使用实验数据，纯理论证明。

**📈 对比分析**

与已知NP‑完备性结果对比，提出多项式时间算法，理论上可在O(n^3)等多项式时间内完成。

**⚠️ 局限性**

仅适用于弱桥图、伪模图及基图，无法推广到一般图；算法仍依赖结构性质，实际实现复杂度较高。

---

## 413. SWNet: A Cross-Spectral Network for Camouflaged Weed Detection

**arXiv ID:** 2604.16147 | [PDF](https://arxiv.org/pdf/2604.16147v1)

**作者:** Henry O. Velesaca `[一作]` (ESPOL Polytechnic University), Angel D. Sappa `[通讯]` (ESPOL Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种名为SWNet的双模态跨光谱网络，用于检测密集农业环境中被植物伪装的杂草。

**💡 创新点**

创新点在于引入双模态门控融合模块与边缘感知细化分支，结合PVTv2骨干，使RGB与近红外信息动态融合并精细化边界。

**🔧 技术方法**

采用Pyramid Vision Transformer v2骨干、残差/卷积块、CBAM、双模态门控融合、边缘细化以及多尺度深度监督等技术。

**📊 数据集**

使用Weeds‑Banana多光谱数据集（RGB+NIR）。

**📈 对比分析**

与十余种SOTA COD模型对比，SWNet在Vis+NIR组合下实现S-α 0.8966、F^w_β 0.8767、M 0.0070，超越现有最佳模型ARNet‑v2和HitNet。

**⚠️ 局限性**

局限在于训练与推理仍需高算力，未验证实时边缘设备部署，且对极端光照变化的鲁棒性待进一步验证。

---

## 414. SCRIPT: Implementing an Intelligent Tutoring System for Programming in a German University Context

**arXiv ID:** 2604.16117 | [PDF](https://arxiv.org/pdf/2604.16117v1)

**作者:** Alina Deriyeva `[一作]` (Bielefeld University), Benjamin Paassen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了面向高级本科计算机科学学生的Python编程智能辅导系统SCRIPT，该系统支持个性化提示、细粒度键盘输入记录，并符合德国及欧盟GDPR和AI法等监管要求。

**💡 创新点**

创新点包括四模型架构（教学、学习、领域、UI）与可插拔LLM提示生成器的结合；利用自托管LLaMA-70b减少对商业API的依赖；以合规为前提的隐私保护设计；系统同时兼具教学与科研双重功能，支持知识追踪和实验验证。

**🔧 技术方法**

技术栈包括Angular前端、FastAPI后端、MongoDB存储、Docker容器化、Judge0代码执行沙箱、Ollama自托管LLM、知识跟踪算法、Q‑matrix及能力模型；使用前后端分离、RESTful API、键盘事件捕获等。

**📊 数据集**

使用的任务集合主要来自“数据挖掘”和“机器学习”课程（概率论、统计检验、PCA、聚类、逻辑模型、马尔可夫模型、深度模型、推荐系统等），并在试点中收集了学生的键盘交互数据；未使用公开大规模数据集。

**📈 对比分析**

目前尚未开展系统性能或效果对比实验；仅在课程预备阶段进行了志愿性使用；未来计划通过A/B测试、前后测评等方法验证不同教学模型和提示策略的学习效益。

**⚠️ 局限性**

局限性包括：系统仍处于alpha阶段，功能不完整；功能扩展和大规模部署受合规与隐私限制；存在安全漏洞风险；缺乏大规模实验验证；自托管LLM对资源和算力需求较高；教师和学生的手工任务上传导致作者工具不够友好。

---

## 415. Deterministic Task Scheduling in In-Vehicle Networks for Software-Defined Vehicles

**arXiv ID:** 2604.16143 | [PDF](https://arxiv.org/pdf/2604.16143v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 416. Co-Design of CNN Accelerators for TinyML using Approximate Matrix Decomposition

**arXiv ID:** 2604.16113 | [PDF](https://arxiv.org/pdf/2604.16113v1)

**作者:** José Juan Hernández Morales `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Jürgen Teich `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 11698 | [OpenAlex ID](https://openalex.org/A5076672029)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种后训练、硬件感知的协同设计框架，通过近似矩阵分解将CNN权重转换为基于2的幂表示，从而在TinyML设备上实现无乘法器的FPGA加速器。

**💡 创新点**

创新点在于：①将权重分解与加速器设计联合优化的Pareto前沿搜索；②设计可编程、轻量级的shift‑and‑add处理单元；③利用遗传算法在硬件资源与准确率约束下搜索最优方案。

**🔧 技术方法**

技术方法包括近似矩阵分解（WMD）、基于power‑of‑two的权重重构、可编程的shift‑add systolic array、硬件资源与延迟的高层模型以及NSGA‑II遗传优化。

**📊 数据集**

使用了MLPerfTiny基准中的ResNet、MobileNetV1和DS‑CNN三种网络，数据来源为CIFAR‑10、VWW和Speech Commands等公开数据集。

**📈 对比分析**

与传统8位乘法器的systolic array以及现有的Post‑Training Quantization、ShiftCNN方案进行对比；结果显示平均实现33%延迟提升、1.3%准确率损失，吞吐量提升约1.55倍，功耗与能效均优于对比方法。

**⚠️ 局限性**

局限性在于：①对权重分解参数的离散化可能遗漏更优解；②框架依赖预训练模型且不支持在线学习；③在极小型FPGA或更严格的功耗预算下，仍可能超出资源上限。

---

## 417. ProcRoute: Process-Scoped Authorization of Split-Tunnel Routes

**arXiv ID:** 2604.16080 | [PDF](https://arxiv.org/pdf/2604.16080v1)

**作者:** Arul Thileeban Sagayam `[一作]` `[通讯]` (Bloomberg), Arul Thileeban Sagayam (Bloomberg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了在 Split‑Tunnel VPN/ ZTNA 环境下，以进程为粒度的路由访问授权机制 ProcRoute，限制仅授权应用能使用内部路由。

**💡 创新点**

创新点包括：① 用形式化的访问控制模型将路由视作资源，进程视作主体，并给出默认拒绝与单调组合的安全不变式；② 通过 cgroup v2 的 socket‑address eBPF hook 实现内核级实时决策；③ 采用客户端标签+网关 TC 程序的拆分架构，既保持分流互联网访问，又实现跨网关的路由授权；④ 提供快照式撤销（epoch）与后台清理，支持子毫秒级的撤销。

**🔧 技术方法**

核心技术包括：eBPF（socket‑address hook、TC ingress/egress）、cgroup v2 进程归属、LPM trie（IP 前缀匹配）、WireGuard 加密隧道、SHA‑256 二进制哈希校验、环形缓冲日志、内核态全局 epoch 计数。

**📊 数据集**

评测使用自建实验环境：单机循环回路基准（EPYC 7443P）与双机 WireGuard 隧道（Ryzen 7 7700X），不依赖公开数据集，而是采用合成的 TCP/UDP 连接与流量模式。

**📈 对比分析**

与 nftables 的 cgroup‑match 方案对比：ProcRoute 在单流吞吐率上与基线相当（≈8.9 Gbps），连接延迟仅增加 3–4 µs；多流吞吐率保持在基线 2–3% 以内；政策规模扩展至 5 000 前缀时启动与吞吐保持平坦；撤销时间在 190 µs 内完成，新的连接在 146 µs 内被拒绝。

**⚠️ 局限性**

局限性：① 仅防御非特权进程的跨过程攻击，无法阻止已获授权进程内的代码注入；② 需要 root 权限部署，受限于 Linux 环境；③ 标签使用 IP 头字段，可能影响分片/DSCP 等网络功能；④ 目前未支持 DNS 名称级别的细粒度授权。

---

## 418. AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency

**arXiv ID:** 2604.16158 | [PDF](https://arxiv.org/pdf/2604.16158v1)

**作者:** Max Henning Höth `[一作]` (Aleph Alpha Research Lab1141), Letitia Parcalabescu `[通讯]` (Aleph Alpha Research Lab1141)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过可微注意力操纵（AtMan）学习识别链式推理（CoT）中对最终答案具有因果影响的显著令牌，并以此为基础构造奖励，使大语言模型在生成推理时既保持准确性又提升可解释性。

**💡 创新点**

将AtMan视为可微掩码在每个样本上进行优化，从中提取显著性度量并作为强化学习奖励，与结果奖励在GRPO框架中联合使用，首次实现同时优化推理正确性与推理可解释性。

**🔧 技术方法**

可微注意力掩码(AtMan)、强化学习（GRPO）、交叉熵损失、显著性奖励、梯度下降优化掩码、Llama‑3.2‑3B‑Instruct 微调。

**📊 数据集**

GSM8K（1000个题目）和MMLU（1000个题目）。

**📈 对比分析**

与仅使用结果奖励的基线进行对比，使用 Pass@4 评估准确率。AtManRL 在保持 Pass@4 准确率（GSM8K 下降 0.4%，MMLU 上升 0.1%）的同时，将平均推理长度减少约 44%（GSM8K）和 46%（MMLU），停用词比例下降，数字和符号比例上升，显著提升推理效率与信息密度。

**⚠️ 局限性**

需要在更多模型、数据集和领域验证，真正评估 faithfulness，奖励机制可能抑制自我纠错标记，导致对回溯或修正行为的忽视。

---

## 419. Motion-Adapter: A Diffusion Model Adapter for Text-to-Motion Generation of Compound Actions

**arXiv ID:** 2604.16135 | [PDF](https://arxiv.org/pdf/2604.16135v1)

**作者:** Yue Jiang `[一作]` (Northwest University), Yuhe Zhang `[通讯]` (Northwest University)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5115593832)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种可插拔的Motion-Adapter模块，用于改进文本到动作的扩散模型，使其能够生成多子动作的复合动作序列。

**💡 创新点**

创新点在于通过解耦交叉注意力生成结构性掩码，解决了灾难性遗忘与注意力塌陷问题；该掩码在去噪过程中引导模型保持身体部位的空间一致性，从而实现自然的多动作融合，而无需改动扩散模型参数或额外训练。

**🔧 技术方法**

技术手段包括扩散生成模型（MDM、MotionDiffuse）、自监督STEncoder网络、CLIP文本编码、跨注意力机制、结构性掩码生成与应用。

**📊 数据集**

主要使用HumanML3D数据集进行单动作训练，并基于该数据集构造了484个上下肢组合的复合动作基准（包含单动作和合成动作），在此基础上评估模型性能。

**📈 对比分析**

与7种SOTA方法（如MDM、MotionDiffuse、SALAD、STMC等）进行对比，采用R-Precision、MM-Dist、FID、Diversity、Transition等指标；在复合动作任务上，Motion-Adapter在所有指标均优于基线（R-Precision 0.158/0.162，MM-Dist 0.360/0.359，FID 14.95/14.99，用户研究Fidelity 9.27/9.08）。

**⚠️ 局限性**

局限性包括：生成能力受限于底层扩散模型；上、下肢被视为统一区域，缺乏对手指等细粒度部位的精细控制。

---

## 420. The Relic Condition: When Published Scholarship Becomes Material for Its Own Replacement

**arXiv ID:** 2604.16116 | [PDF](https://arxiv.org/pdf/2604.16116v1)

**作者:** Lin Deng `[一作]` (University of New South Wales), Chang-bo Liu `[通讯]` (Independent Researcher)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提取两位人文社会科学学者公开论文的推理系统，并将其转换为LLM推理时约束，构建学者型机器人进行学术任务。

**💡 创新点**

首次演示仅凭公开语料即可通过结构化抽取重建学者推理体系，并在专家评估中达成高职级水平，提出“遗物条件”框架。

**🔧 技术方法**

使用八层推理抽取、九模块技能架构、GPT‑5.4大语言模型的推理时约束实现。

**📊 数据集**

两位学者各自的公开文献集合：A为68篇分析单元约1742页，B为35篇完整文本（论文、长篇作品、章节）。

**📈 对比分析**

通过三名高级学者的评估报告、6份职位级别综合和三轮小组讨论，学者型机器人在评审、指导、授课、辩论等任务均获得平均4.4–4.6/5或8.5/10以上的高分，等级评定均达到高级讲师及以上。

**⚠️ 局限性**

样本仅两位学者、评估报告非统一格式、学生试验样本小且社群近亲、未公开完整抽取流程，且仅验证公共推理可被抽取，无法说明普适性和长期影响。

---

## 421. Logarithmic-Time Geodesically Convex Decomposition in Programmable Matter

**arXiv ID:** 2604.16112 | [PDF](https://arxiv.org/pdf/2604.16112v1)

**作者:** Henning Hillebrandt `[一作]` (Paderborn University), Julian Werthmann `[通讯]` (Paderborn University)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5084662645)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在可编程物质（amoebot）模型中，为任意在三角网格上的连通结构设计了一种 O(|ℋ|) 个简单、几何凸（geodesically convex）子结构的分解算法，并证明该算法在使用可重构电路扩展时能在 O(log n) 轮次内完成。

**💡 创新点**

创新点主要包括：
- 将传统凸分解扩展到任意三角网格结构，能处理含孔（holes）的复杂形状；
- 通过对门（gate）和门户（portal）的巧妙拆分与合并，构造出几何凸子区域，避免了先前仅能处理特殊结构的分解方法；
- 利用可重构电路实现远距离信号传输，从而把通信瓶颈降到对数级别，显著提升时间复杂度。

**🔧 技术方法**

核心技术包括：
- 门与门户的分割与拆分（splitting operations）以及基于门户图（portal graphs）的树形结构分析；
- 逐阶段分解（先简单区域 → 隧道区域 → 几何凸区域）
- PASC、根与修剪（root and prune）等基于电路的树量化算法；
- 方向性全局极值（global maxima）求解和区块化（block）技术以降低迭代次数。

**📊 数据集**

本工作为理论算法，未使用具体实验数据集，而是针对任意大小 n 的三角网格图进行证明。

**📈 对比分析**

与以往的线性时间三角剖分和对数时间隧道分解相比，本文提出的算法在结构通用性上有显著提升；在时间复杂度上保持了对数级别 O(log n)，并在全局极值、生成树等子任务上进一步优化到 O(log n) 轮次。

**⚠️ 局限性**

局限性包括：
- 只适用于三角网格结构，若是其他格子（如正方形、六边形）需进一步研究；
- 依赖可重构电路扩展，若硬件不支持此功能，则无法达到对数级别性能；
- 对同步模型假设严格，异步环境下的实现尚未讨论；
- 由于分解产生的子结构数量与孔数成正比，若孔数接近 n，实际子结构数仍可能较大。

---

## 422. Decoding Algorithms for Tensor Codes

**arXiv ID:** 2604.16105 | [PDF](https://arxiv.org/pdf/2604.16105v1)

**作者:** Eimear Byrne `[一作]` (University College Dublin), Lucien François `[通讯]` (University College Dublin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了一类张量编码的性质，并提出了针对不同度量的解码技术，特别是针对低张量秩错误的解码算法。

**💡 创新点**

创新点在于提出了一种基于纤维的解码方法，并推广了Loidreau-Overbeck的解码方法，以纠正受限于切片空间和纤维空间维度的错误。

**🔧 技术方法**

使用了张量编码的解码算法，特别是基于Gabidulin编码器的解码方法，以及类似于Loidreau-Overbeck的解码方法。

**📊 数据集**

使用了Roth的张量编码作为数据集，研究了其在不同度量下的解码能力。

**📈 对比分析**

与现有方法进行了详细比较，提出的算法在多项式复杂度下能够处理更广泛的可解错误范围，性能优于传统方法。

**⚠️ 局限性**

限制在于算法的复杂度在某些情况下可能会增加，尤其是在处理高阶张量时，解码的复杂性可能会显著提高。

---

## 423. The Harder Path: Last Iterate Convergence for Uncoupled Learning in Zero-Sum Games with Bandit Feedback

**arXiv ID:** 2604.16087 | [PDF](https://arxiv.org/pdf/2604.16087v1)

**作者:** Côme Fiegel `[一作]` (ENSAE Paris - CREST), Vianney Perchet `[通讯]` (ENSAE Paris - CREST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在零和矩阵博弈中，使用无通信的bandit反馈算法实现最后一次迭代收敛，并给出了理论最优收敛率。

**💡 创新点**

创新点在于证明了无耦合算法在bandit下的最后一次迭代收敛率下界为Ω(T^(-1/(2+p)))，并提出了两个匹配该下界的算法，其中一个通过同步探索-利用策略实现最优收敛率，另一个利用正则化镜像下降并结合倍增技巧实现无时间窗口收敛。

**🔧 技术方法**

所采用技术包括理论下界构造、探索-利用概率调度、正则化零和游戏的Kullback–Leibler正则化、双镜像下降更新，以及倍增技巧的调度设计。

**📊 数据集**

由于论文为理论研究，未使用具体数据集，而是通过对任意 2×2 矩阵博弈构造证明下界，并在一般 A×B 矩阵博弈下给出性能分析。

**📈 对比分析**

与现有 bandit 下的平均策略收敛方法相比，提出的算法在最后一次迭代上达到 T^(-1/4) 的收敛率（最优），优于以往的 T^(-1/8) 或 T^(-1/6) 等结果。

**⚠️ 局限性**

主要局限包括需要双方同步或共享种子、正则化参数依赖先验时间 horizon、无法直接应用于大规模或扩展式博弈、以及对 p>2 的下界与高概率收敛仍未完全证明。

---

## 424. ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics

**arXiv ID:** 2604.16254 | [PDF](https://arxiv.org/pdf/2604.16254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 425. The Amazing Stability of Flow Matching

**arXiv ID:** 2604.16079 | [PDF](https://arxiv.org/pdf/2604.16079v1)

**作者:** Rania Briq `[一作]` (Forschungszentrum Juelich), Stefan Kesselheim `[通讯]` (Forschungszentrum Juelich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了流匹配（Flow‑Matching）模型在数据裁剪、架构变化等扰动下的生成稳定性，提出了三种基于梯度、损失和聚类的裁剪方法，并验证了其对模型性能的影响。

**💡 创新点**

证明了流匹配模型对数据规模、数据分布和模型容量的极端扰动具有高度鲁棒性，并发现合理的聚类裁剪方法甚至能提升FID；首次系统性比较了不同裁剪策略在保持样本质量与多样性方面的表现。

**🔧 技术方法**

使用FM‑DiT（Transformer‑based流匹配）与VQ‑VAE编码器，结合ArcFace相似度评估、FID评价、梯度/损失评分、CLIP嵌入聚类等技术。

**📊 数据集**

主要在CelebA‑HQ人脸数据集上进行实验，也对比了同域数据FFHQ。

**📈 对比分析**

相较于未裁剪模型，最优裁剪策略（Cluster‑balanced）在FID上略有提升（从24.24到22.80），ArcFace相似度保持在0.72–0.74；其他策略如梯度高/低、损失高/低会导致FID恶化但相似度仍高于无关对。

**⚠️ 局限性**

受限于人脸数据集的偏倚与主观性，实验仅在单一视觉域进行，无法直接说明对更复杂或多模态数据的适用性；裁剪策略的超参数（如k值、采样比例）需要进一步调优。

---

## 426. Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations

**arXiv ID:** 2604.16217 | [PDF](https://arxiv.org/pdf/2604.16217v1)

**作者:** Yanli Wang `[一作]` (Imperial College London), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 12624 | [OpenAlex ID](https://openalex.org/A5042417097)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于层级可用信息的分割式置信度预测框架（Layerwise CP）用于LLM问答

**💡 创新点**

创新点在于将内部层级表示的可用信息作为非一致性得分，替代传统面向输出的不确定性信号

**🔧 技术方法**

采用LLM内部隐藏状态、可用信息计算、分割式置信度预测、答案级聚合与权重组合

**📊 数据集**

在闭合式MMLU-Pro、MedMCQA和开放式TriviaQA、CoQA等公开问答数据集上进行评估

**📈 对比分析**

与多种基于输出的基线（自一致性、熵、语义熵、SConU-Pro等）对比，Layerwise CP在跨域迁移时显著降低误覆盖率与预测集大小，开放域同样提高效率

**⚠️ 局限性**

主要局限是实验性证据为主，缺乏在分布漂移下的理论有效性证明，对特定模型和任务的泛化性还有待验证

---

## 427. JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models

**arXiv ID:** 2604.16171 | [PDF](https://arxiv.org/pdf/2604.16171v1)

**作者:** Alexandra Dragomir `[一作]` (Bitdefender), Radu Tudor Ionescu `[通讯]` (University Of Bucharest)

**通讯引用:** 8238 | [OpenAlex ID](https://openalex.org/A5081017623)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于JumpReLU门控的自适应稀疏LoRA框架（称为JLoRA），用于连续学习任务中通过可学习阈值实现参数隔离，并将其与现有PEFT连续学习方法（如IncLoRA和ELLA）结合。

**💡 创新点**

创新点包括：1）首次将JumpReLU激活函数迁移到权重更新层面，实现可学习的稀疏阈值；2）引入渐进式稀疏调度机制，避免训练初期梯度被截断；3）在不使用显式正则化的情况下，仅通过自适应稀疏实现显著降低任务干扰；4）框架高度模块化，可无缝嵌入任意LoRA基连续学习方法。

**🔧 技术方法**

核心技术：低秩适配器（LoRA）、JumpReLU门控、可学习阈值、稀疏更新插值、渐进式调度、可选ELLA正则化、参数合并。

**📊 数据集**

使用了两个主基准：Standard CL Benchmark（5个文本分类数据集，3个任务序列）和Long Sequence Benchmark（15个数据集，3个任务序列），涵盖情感分析、主题分类、GLUE、SuperGLUE、IMDB等。

**📈 对比分析**

与IncLoRA和ELLA进行对比，评估指标包括整体准确率（OA）、向后/向前迁移（BWT/FWT）。实验结果显示：JLoRA+IncLoRA在OA上平均提升约4–5%，JLoRA+ELLA在OA、BWT、FWT上实现了新SOTA；在两大基准上均保持一致的性能提升，并显著降低任务间参数重叠。

**⚠️ 局限性**

限制：1）仅在NLP连续学习任务上验证，缺乏跨模态（如视觉）实验；2）对稀疏阈值的初始设置和调度参数（S_start、S_final、ϵ）仍需经验选择；3）在极长任务序列或高容量模型中，稀疏度和性能的平衡尚未彻底探索；4）对可解释性和硬件加速友好度的评估不足。

---

## 428. Dental Panoramic Radiograph Analysis Using YOLO26 From Tooth Detection to Disease Diagnosis

**arXiv ID:** 2604.16231 | [PDF](https://arxiv.org/pdf/2604.16231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 429. Where Do Vision-Language Models Fail? World Scale Analysis for Image Geolocalization

**arXiv ID:** 2604.16248 | [PDF](https://arxiv.org/pdf/2604.16248v1)

**作者:** Siddhant Bharadwaj `[一作]` (University of Central Florida), Shruti Vyas `[通讯]` (University of Central Florida)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5072711888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对多种最新视觉-语言模型（VLM）在仅使用地面视角图像的情况下进行国家级地理定位的零样本评估，提出统一的提示式推理框架并设计新的错误合理性度量（GER）及邻域跳数分析。

**💡 创新点**

①系统化比较九大VLM在国家级定位上的表现；②引入GER度量区分视觉合理与无意义错误；③发现模型规模与精度不总正相关（Qwen3-VL出现“倒置尺度”现象）；③分析城乡、生态区的性能差异。

**🔧 技术方法**

零样本提示式推理、结构化JSON输出、Greedy解码、视觉嵌入邻域统计、邻国跳数图谱、GER得分计算。

**📊 数据集**

GeoGuessr‑50k（49,997张，124国），CityGuessr‑68k（68,269张，91国），OSV5M（50,000张，219国）三大地理多样化数据集。

**📈 对比分析**

与传统检索/定位基线对比，Qwen3‑VL‑4B在所有数据集上均取得最高Top‑1/Top‑5准确率，约74%‑89%，但仍显著低于部分有监督模型；在标签受限条件下性能下降，显示生成误差；GER与跳数分析表明优秀模型错误更“合理”。

**⚠️ 局限性**

仅评估国家级定位，缺乏细粒度（城市/坐标）能力；数据集存在地域偏倚（如US过多、非洲欠缺）；提示设计与标签排序敏感；Fine‑tune实验仅局限于单一模型。

---

## 430. CollideNet: Hierarchical Multi-scale Video Representation Learning with Disentanglement for Time-To-Collision Forecasting

**arXiv ID:** 2604.16240 | [PDF](https://arxiv.org/pdf/2604.16240v1)

**作者:** Nishq Poorav Desai `[一作]` (Queen's University), Michael Greenspan `[通讯]` (Queen's University)

**通讯引用:** 2312 | [OpenAlex ID](https://openalex.org/A5052248307)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种双流分层多尺度Transformer架构 CollideNet，用于时间到碰撞（TTC）预测；

**💡 创新点**

创新点包括：①在空间流中实现多分辨率聚合；②在时间流中对非平稳性、趋势和季节性进行分解与去平稳化；③将分解后的特征与多尺度段相关注意力相结合；

**🔧 技术方法**

技术手段涵盖：分层多尺度ViT、Mask‑Unit Attention、系列分解（trend/seasonality）、多尺度段相关注意力（MSSC）及其预测版（PreMSSC）以及去平稳化与重平稳化机制；

**📊 数据集**

使用公开的三大交通碰撞数据集：Dashcam Accident Dataset (DAD)、Car Crash Dataset (CCD)、Detection of Traffic Anomaly Dataset (DoTA)；

**📈 对比分析**

与CNN‑RNN、3D ConvNet、ViViT、TimeSformer、Video‑Swin、Video‑FocalNet、VidNeXt 等基线比较，在三数据集上的 MSE 均优于对手，尤其在 CCD 数据集上提升约 30%；

**⚠️ 局限性**

局限性包括：模型对不同摄像头/光照环境的跨域泛化仍有挑战，且多尺度段相关注意力虽降低复杂度，但在极长序列上的实时推理仍受限。

---

## 431. Enhancing AI and Dynamical Subseasonal Forecasts with Probabilistic Bias Correction

**arXiv ID:** 2604.16238 | [PDF](https://arxiv.org/pdf/2604.16238v1)

**作者:** Hannah Guan `[一作]` (Harvard), Lester Mackey `[通讯]` (Microsoft)

**通讯引用:** 3222 | [OpenAlex ID](https://openalex.org/A5018151279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Probabilistic Bias Correction (PBC) 框架，利用机器学习对子季节性预报的概率分布进行自适应修正，显著提升ECMWF、AIFS‑SUBS和PoET等多种模型的预报精度。

**💡 创新点**

创新点在于：①直接在概率分布空间进行校正而非观测值；②结合两种互补的机器学习修正并投影到有效CDF空间；③轻量级、可在线更新且适用于多模型、跨学科预报；④在实时AI Weather Quest竞赛中实现首位。

**🔧 技术方法**

使用的技术包括：概率回归、线性/非线性回归、等距变换、等距回归投影（isotonic regression）、自适应训练窗口选择、基于滞后观测与分位数的多项式回归；并以TensorFlow/PyTorch实现。

**📊 数据集**

使用的数据集包括：ERA5重分析（1979‑2026）、ECMWF S2S 1‑6周预报与历史回测、FuXi‑S2S 51 成员集、AIFS‑SUBS 100 成员、PoET 100 成员、GDACS 洪水事件数据库、以及公开的AI Weather Quest 2025赛季评估数据。

**📈 对比分析**

比较方法：通过 Ranked Probability Skill Score (RPSS)、Brier Skill Score (BSS) 在全球网格层面与极端事件层面进行评估；结果显示PBC将ECMWF原始预报的 RPSS 提升至 91–98% 的格点，AI 系统的 RPSS 提升 2–3 倍，且在 2025 AI Weather Quest 竞赛中 MicroDuet 获得所有变量、两周时段的第一名，超越六家机构的多模型组合。

**⚠️ 局限性**

局限性：①仅为后处理，不改善物理模型本身；②对训练周期、季节性变化的适应性需进一步验证；③在极端少样本事件或罕见气候情景下校正效果可能受限；④对计算资源要求虽低但仍需日常模型输出与实时更新。

---

## 432. Why Open Source? A Game-Theoretic Analysis of the AI Race

**arXiv ID:** 2604.16227 | [PDF](https://arxiv.org/pdf/2604.16227v1)

**作者:** Andjela Mladenovic `[一作]` (Université de Montréal), Gauthier Gidel `[通讯]` (Université de Montréal)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5011494985)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于博弈论的 AI 竞赛模型，用以分析 AI 开源与闭源决策的动力学，并在离散（全开源/闭源）和连续（部分开源）情境下研究纯纳什均衡的存在与求解。

**💡 创新点**

创新点在于：①首次将开源策略纳入 AI 竞赛博弈框架；②对离散与连续两种行动空间均给出纯纳什均衡存在性证明；③证明离散情境下寻找非平凡纯均衡为 NP‑hard，并给出可求解的 MIP（混合整数规划）转化；④对连续情境给出可通过 MIP 求解的最优解方法。

**🔧 技术方法**

技术方法包括：
- 博弈论模型构建与纯纳什均衡条件推导；
- 线性规划与混合整数规划（MIP）建模与求解；
- NP‑hard 证明（通过 3‑SAT 归约）；
- 凸分析与 Glicksberg 定理证明连续情境下的存在性；
- 片段线性最优反应求解与梯度条件转化为线性约束。

**📊 数据集**

论文为理论性研究，未使用公开数据集；所有结果基于模型参数 δ_i、Δ_ij、d_i 等符号常数，主要通过理论证明与符号计算获得。

**📈 对比分析**

方法对比：论文未与传统机器学习/深度学习模型进行实验性比较；主要通过理论复杂度与 MIP 求解实验验证可行性；在离散场景中 MIP 求解在小规模实例上能快速给出均衡；在连续场景中同样可通过 MIP 获得纯均衡，证明了方法的可行性。

**⚠️ 局限性**

局限性包括：
- 假设收益 δ_i 与 Δ_ij 独立，未考虑交叉依赖；
- 仅考虑单一决策点，未建模时间演化或重复博弈；
- NP‑hard 证明仅限于非平凡纯均衡，未讨论更一般均衡；
- 对实际 AI 开源生态的定量验证缺失，模型需进一步结合真实数据校准。

---

## 433. Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement

**arXiv ID:** 2604.16177 | [PDF](https://arxiv.org/pdf/2604.16177v1)

**作者:** Lorenzo Beltrame `[一作]` (Austrian Institute of Technology), Marco Koerner `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种三阶段逐步阴影去除框架，将 OmniSR 变换器的直接细化扩展为多级残差校正网络。

**💡 创新点**

核心创新在于：1）在所有阶段共享冻结的 DINOv2 语义特征与深度/法线几何线索；2）引入收缩约束（contraction loss）确保各阶段误差单调下降；3）通过分阶段预训练与迁移，缓解数据对齐与分布漂移问题。

**🔧 技术方法**

使用 Transformer‑based OmniSR 结构、DINOv2 ViT‑L/14 语义编码器、Depth Anything V2 生成深度与法线、LPIPS 视觉感知损失、MSE、收缩约束等训练目标。

**📊 数据集**

在 NTIRE2026 WSRD+ 隐藏测试集上训练；额外在 ISTD+ 与 UAV‑SC+ 公开测试集进行迁移微调评估。

**📈 对比分析**

在官方排行榜中取得第一名，PSNR 26.68、SSIM 0.874、LPIPS 0.058、FID 26.14，超越主要竞争对手；在 ISTD+ 上达到 34.24 dB 的 PSNR（mask‑free），在 UAV‑SC+ 上达到 24.60 dB 的 PSNR。

**⚠️ 局限性**

主要局限：多阶段推理耗时较长；在某些数据集（如 ISTD+）迁移效果不及原始预训练模型；收缩约束在某些情况下可能导致过度平滑，牺牲细节。

---

## 434. Semantic Area Graph Reasoning for Multi-Robot Language-Guided Search

**arXiv ID:** 2604.16263 | [PDF](https://arxiv.org/pdf/2604.16263v1)

**作者:** Ruiyang Wang `[一作]`, Miroslav Pajic `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于语义区域图（Semantic Area Graph）的多机器人搜索框架SAGR，利用大语言模型在房间层面进行语义决策，并结合前沿规划实现协同探索与目标搜索。

**💡 创新点**

创新点在于将密集语义占用地图压缩为房间层级语义拓扑图，作为LLM的高层抽象输入，使得LLM能在语义层面高效分配机器人，同时保持低层前沿规划的确定性，从而在语义搜索任务中显著提升效率。

**🔧 技术方法**

采用大语言模型（GPT‑4o）完成房间分配；使用前沿检测、BFS聚类、Hungarian 匹配、TSP 排序以及基于前沿的路径规划等经典算法，实现从语义图到局部执行的完整闭环。

**📊 数据集**

在 Habitat‑Matterport3D（HM3D）数据集的10个公寓布局上随机生成100个实验场景，作为实验用数据集。

**📈 对比分析**

与Hungarian、RACER、AEP+DVC等几何基准对比，SAGR在语义搜索任务中平均比最优几何方法提升约19%（18.8%），在纯探索任务中保持与几何方法相近的性能，且在大规模环境下搜索时间下降最高可达约18.8%。

**⚠️ 局限性**

局限性包括：对LLM推理随机性的依赖导致结果略有波动；依赖感知模块给出的语义标签质量；以及在纯探索任务中与专门几何方法的性能差距；在更大规模机器人队伍或更复杂环境中可扩展性尚待验证。

---

## 435. Joint-Centric Dual Contrastive Alignment with Structure-Preserving and Information-Balanced Regularization

**arXiv ID:** 2604.16247 | [PDF](https://arxiv.org/pdf/2604.16247v1)

**作者:** Habibeh Naderi `[一作]` (Dalhousie University), Stan Matwin `[通讯]` (Dalhousie University)

**通讯引用:** 12840 | [OpenAlex ID](https://openalex.org/A5042893723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出HILBERT框架，针对长序列音频‑文本数据在低资源条件下实现跨模态对齐与文档级表示学习。

**💡 创新点**

创新点包括双向对比损失（音频‑联合、文本‑联合）、结构保持的CKA损失、信息平衡的MI损失以及结合MoE的下游分类架构。

**🔧 技术方法**

技术上采用冻结的预训练语音模型（Whisper/Hubert）与文本模型（TinyBERT/MPNet等），跨模态自注意力聚合，双对比、CKA、MI三重辅助损失，最终通过Mixture‑of‑Experts进行任务预测。

**📊 数据集**

使用FORBOW研究项目的父母与子女语音-转录对，包含情感、心理谱系等多标签不平衡任务。

**📈 对比分析**

与基线（转移学习、CLAP、单模态等）及部分消融进行比较，HILBERT在多数任务上提升AUC约5–10个百分点，尤其在情感、心理谱系与精神疾病预测任务中取得最高AUC≈80。

**⚠️ 局限性**

局限包括对长序列切分的依赖、对大规模数据的鲁棒性待验证，以及对极端不平衡或多模态缺失情况的适应性仍需进一步研究。

---

## 436. Find, Fix, Reason: Context Repair for Video Reasoning

**arXiv ID:** 2604.16243 | [PDF](https://arxiv.org/pdf/2604.16243v1)

**作者:** Haojian Huang `[一作]` (Hong Kong University of Science and Technology), Yingcong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种名为 Find, Fix, Reason（FFR）的框架，利用冻结的教师模型对视频推理的失败回放进行最小化的空间‑时间证据补丁，指导学生模型重新生成答案，并通过GRPO与鲁棒提升奖励（RIR）实现基于可验证奖励的强化学习；

**💡 创新点**

创新点在于：①用冻结教师在观测层面进行精准的缺失依赖诊断与最小化补丁，避免教师直接泄漏答案；②将补丁视为因果干预，在保持 on‑policy 训练的同时重塑奖励景观；③通过 RIR 同时优化答案有效性与推理路径对齐；④实现了对大模型知识的“工具化”利用，无需额外预训练或两阶段微调；

**🔧 技术方法**

主要技术包括：冻结教师模型+多模态工具检索、观察级干预、Group Relative Policy Optimization (GRPO)、Robust Improvement Reward (RIR)、基于可验证奖励的二值奖励设计、KL 正则化、以及对补丁税 (κ) 的调节；

**📊 数据集**

使用了多种视频推理与通用视频理解基准：MMVU、VSI‑Bench、VideoMMMU、Video‑Holmes、LongVideoBench、LVBench、MVBench 与 TempCompass；

**📈 对比分析**

与传统 on‑policy、混合策略、工具使用的基线相比，FFR 在所有基准上均提升约 10‑25%（如 MMVU +11.75%、VSI‑Bench +22.33% 等），甚至在小学生模型上达到甚至超越更大教师模型；

**⚠️ 局限性**

局限性包括：①依赖教师模型的诊断准确性与补丁生成质量；②需要精细设计提示与负向引导以防答案泄露；③补丁税 κ 的选择需经验调优；④在某些视频任务（如过长或极短片段）中补丁可能不完全解决多模态歧义。

---

## 437. Detecting and Suppressing Reward Hacking with Gradient Fingerprints

**arXiv ID:** 2604.16242 | [PDF](https://arxiv.org/pdf/2604.16242v1)

**作者:** Songtao Wang `[一作]` (University Of Alberta), Xi Ye `[通讯]` (University Of Alberta)

**通讯引用:** 15993 | [OpenAlex ID](https://openalex.org/A5087787304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用语言模型内部梯度信息的“梯度指纹”方法，检测并抑制强化学习可验证奖励（RLVR）训练中出现的隐式奖励劫持行为；

**💡 创新点**

创新点在于将模型在生成链式推理（CoT）过程中的梯度压缩为低维指纹，并通过聚类判断是否为劫持行为；此梯度指纹相较于文本监测能更早、更准确地识别隐式劫持，并可直接作为训练监督信号；

**🔧 技术方法**

技术包括关键层选择、在选定层插入LoRA适配器计算梯度、随机投影压缩、L2归一化、k-means聚类以及软评分机制；同时将该方法集成到拒绝细化（RFT）训练流程中；

**📊 数据集**

实验使用三类任务数据集：BigMath（数学推理）、APPS（代码生成）和AR-LSAT（逻辑推理），分别展示上下文劫持和有限答案空间劫持场景；

**📈 对比分析**

与CoT-Monitor和TRACE两大基线对比，梯度指纹在所有三任务上F1提升约25%以上；在训练期间结合RFT使用时，能够显著提高真任务准确率，接近无劫持模型的表现；

**⚠️ 局限性**

局限性包括：需要少量人工标注聚类标签；在奖励劫持比例极高时检测性能下降；对模型内部梯度访问有要求，且对极大模型的梯度计算仍有成本；

---

## 438. OT on the Map: Quantifying Domain Shifts in Geographic Space

**arXiv ID:** 2604.16220 | [PDF](https://arxiv.org/pdf/2604.16220v1)

**作者:** Haoran Zhang `[一作]` (Harvard University), David Alvarez-Melis `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于地理空间最优传输（GeoSpOT）的距离度量，结合图像/文本特征与经纬度信息，量化不同地理域之间的分布差异，并利用该距离预测跨域迁移难度；

**💡 创新点**

创新点在于：①将空间距离与特征相似度融合进 OT 的基准成本；②利用自监督预训练的地点编码器（SatCLIP、GeoCLIP）实现任务无关的迁移预测；③提供可解释的适用性地图和数据选择工具。

**🔧 技术方法**

核心技术包括：最优传输（OT）及其 Sinkhorn 正则化近似；预训练的图像（ResNet50）和文本（BERT）嵌入；地点编码器 SatCLIP/GeoCLIP；Spearman 相关性和 R² 评估；以及贪心数据选择算法。

**📊 数据集**

实验使用了四大公开数据集：Geo‑YFCC（图像+文本，62 国），FMoW‑Wilds（卫星图像，25 国），GeoDE（对象图像，19 国），每个数据集涵盖多种地理域。

**📈 对比分析**

与单一特征或纯地理距离相比，GeoSpOT 在多模态下取得 Spearman ρ≈0.4–0.7、R²≈0.14–0.56 的显著提升；在源域选择实验中，基于 GeoSpOT 的选择往往比随机或全域训练获得更高的目标域准确率。

**⚠️ 局限性**

局限性包括：仅验证分类任务，未扩展到回归或序列预测；地点编码器对预训练数据的依赖可能导致某些区域表现欠佳；距离计算受 λ 与地理度量选取影响；数据选择算法为贪心，未证明最优，且实验规模相对有限。

---

## 439. NVBench: A Benchmark for Speech Synthesis with Non-Verbal Vocalizations

**arXiv ID:** 2604.16211 | [PDF](https://arxiv.org/pdf/2604.16211v1)

**作者:** Liumeng Xue `[一作]` (Nanjing University), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18825 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 NVBench——一个双语（英语/中文）统一 45 类非语言发声（NVV）基准，包含均衡的语料库和多维度评估协议，用以系统性衡量 TTS 系统在 NVV 合成、控制、时序与显著性方面的表现。

**💡 创新点**

创新点包括：① 统一的 45 类 NVV 分类体系，覆盖呼吸、喉部、生理、笑声、哭声、情感、口腔/杂项等六大类别；② 通过 LLM 辅助的三阶段数据构建流程（种子挖掘、控制生成、人工校验），得到 2250 条高质量双语样本；③ 多轴评估框架，将普通语音自然度/质量与 NVV 可控性、时序定位与感知显著性拆分，并引入 LLM‑based multi‑rater 评估以提升可扩展性。

**🔧 技术方法**

采用的技术手段包括：LLM（Gemini 2.5‑Pro）辅助标注与生成；自动指标（WER/CER、DNSMOS、CLAP、NVV Precision/Recall/NTD 等）；人类听力评测（Prolific 平台、5‑点 Likert）；LLM‑based multi‑rater 评估（随机化、匿名化、低温度、三轮评分）；数据处理与近似去重、交叉验证。

**📊 数据集**

使用的数据集主要来自 InstructTTSEval（英中双语表达式录音与自由字幕），随后经过 LLM 生成与人工校验构建的 4,500 条 NVV 样本；同时引用了多种现有 TTS 语料（SMIIP‑NV、NVSpeech、SynParaSpeech、NonverbalTTS、NonverbalSpeech‑38k、MNV‑17）用于评估系统支持的 NVV 类型。

**📈 对比分析**

在 15 个 TTS 系统（8 tag‑based、7 prompt‑based）上分别计算客观指标、主观 MOS 与 LLM‑based 评分。结果表明：① Gemini 2.5 Pro 在自然度、质量和 NVV 可控性上表现突出；② ElevenLabs 在 tag‑based 评测中兼具高覆盖率与高 PE；③ GPT‑4o mini、Qwen3‑TTS 等在质量与对齐方面领先，但 NVV 控制相对薄弱。总体来看，NVV 可控性往往与整体质量解耦，低 SNR 口腔细节与长时持续情感发声是主要瓶颈。

**⚠️ 局限性**

局限性包括：① 对低 SNR 口腔细节与长时持续情感发声的合成仍不理想；② LLM‑based 评估虽然可扩展，但对 NVV 细粒度的真实性与时序仍存在偏差；③ 数据集规模相对有限，部分稀有 NVV 类型支持不足；④ 评估指标（尤其是 WER/CER、DNSMOS）对非词汇发声的判定不够精准，可能导致误判；⑤ 目前仅覆盖英语与中文，跨语言推广需进一步验证。

---

## 440. AIFIND: Artifact-Aware Interpreting Fine-Grained Alignment for Incremental Face Forgery Detection

**arXiv ID:** 2604.16207 | [PDF](https://arxiv.org/pdf/2604.16207v1)

**作者:** Hao Wang `[一作]` (Harbin Institute of Technology), Weigang Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 4000 | [OpenAlex ID](https://openalex.org/A5014774434)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种无数据重放的增量面部伪造检测框架AIFIND，通过构建语义锚点来稳定特征空间，实现持续学习新伪造类型而不遗忘旧知识。

**💡 创新点**

创新点包括：① 将面部伪造的低级痕迹转化为可迁移的语义锚点；② 设计Artifact‑Probe Attention实现语义与视觉的细粒度对齐；③ 引入Adaptive Decision Harmonizer在球面上对齐分类器权重，保持几何一致性。

**🔧 技术方法**

主要技术：CLIP‑ViT‑L/14视觉语言模型、Artifact‑Driven Semantic Prior Generator、Artifact‑Probe Attention、双重监督（真实性二分类+痕迹多标签），球面对齐（ADH）与知识蒸馏。

**📊 数据集**

使用的评估数据集包括DeepFake Detection Challenge Preview (DFDCP)、Celeb‑DF‑v2 (CDF)、FaceForensics++、MCNet、BlendFace、StyleGAN3、SDv21、DiffusionFace 等多种面部伪造数据集。

**📈 对比分析**

与DFIL、SUR‑LID、CoReD、HDP、CL‑LoRA、Coda‑Prompt等基线在两种增量协议（数据集增量与伪造类型增量）下对比，AIFIND在AUC上均达到≈0.99，显著优于所有无重放或基于重放的方案，取得最优性能。

**⚠️ 局限性**

局限性：① 依赖预训练的视觉语言模型，生成语义锚点需要人工或模型生成的文本描述；② 对极端分辨率、遮挡或未知生成模型的鲁棒性仍有待验证；③ 计算复杂度相对较高，尤其是多头注意力与球面对齐的实现。

---

## 441. Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation

**arXiv ID:** 2604.16197 | [PDF](https://arxiv.org/pdf/2604.16197v1)

**作者:** Yide Ran `[一作]` (Stevens Institute of Technology), Zhaozhuo Xu `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 3487 | [OpenAlex ID](https://openalex.org/A5100677081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种名为RISE的前向仅影响估计方法，用于大规模语言模型的数据归因与评估

**💡 创新点**

利用输出层梯度的外积分解，发现影响信号集中在LM头部，并构造双通道（词汇残差与语义投影误差）压缩方案，显著提升存储与计算效率

**🔧 技术方法**

CountSketch稀疏投影、词汇残差截断、外积分解与双通道相似度融合

**📊 数据集**

OLMo（1B–32B）与Pythia（14M–6.9B）预训练模型的真实训练语料，以及Howdy、Finance‑Medical、Brain‑Rot三大任务数据集

**📈 对比分析**

与RapidIn、ZO‑Inf、TrackStar、BM25等基线比较；在多任务、不同模型规模下，RISE在检索准确率、auPRC/auROC上均优于基线，且在32B模型下可行；在Brain‑Rot闭环实验中，RISE选样训练的模型在困惑度和RULER‑CWE指标上提升显著

**⚠️ 局限性**

主要局限在于只关注LM头部梯度，可能忽略深层微调对行为的影响；对极大模型的GPU内存仍有一定需求；对稀有词和低频语义可能效果受限

---

## 442. MOMENTA: Mixture-of-Experts Over Multimodal Embeddings with Neural Temporal Aggregation for Misinformation Detection

**arXiv ID:** 2604.16172 | [PDF](https://arxiv.org/pdf/2604.16172v1)

**作者:** Yeganeh Abdollahinejad `[一作]` (Pennsylvania State University), Amir Karami `[通讯]` (Kennesaw State University)

**通讯引用:** 1960 | [OpenAlex ID](https://openalex.org/A5010380816)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态谣言检测框架MOMENTA，能够同时处理文本与图像，捕捉跨模态不一致、叙事演化和域不变特征。

**💡 创新点**

创新点包括：①模态专属的 Mixture‑of‑Experts 进行专家化建模；②双向共注意与差异感知分支实现跨模态对齐与不一致检测；③基于重叠窗口的时间聚合，加入漂移与动量编码以捕捉叙事动态；④域对抗学习与原型记忆银行实现跨域泛化；⑤多目标联合优化提升表示质量与鲁棒性。

**🔧 技术方法**

使用的大型预训练模型 XLM‑RoBERTa（文本）和 CLIP（图像），Mixture‑of‑Experts、双向共注意、残差门控融合、时间注意力、梯度反转层、对比学习、R‑Drop、EMA 原型记忆等技术。

**📊 数据集**

在四个多模态谣言基准上评估：Fakeddit、MMCoVaR、Weibo、XFacta，涵盖英文与中文、不同平台与语料特征。

**📈 对比分析**

与多种先前方法（如 E‑CaTCH、MIMoE‑FND、WFT‑BERT‑SRNN 等）进行 head‑to‑head 对比，MOMENTA 在 Accuracy、F1、AUC、MCC 等指标上均实现最优或接近最优成绩，尤其在跨域更难的数据集上提升显著。

**⚠️ 局限性**

局限性包括：①模型组件众多，计算开销大，难以实时部署；②时间聚合使用固定窗口与超参数，缺乏自适应性；③仅处理文本和图像，未利用用户互动、传播网络等补充信息；④在极端域漂移、语言多样性与新型谣言形式上的鲁棒性待进一步验证。

---

## 443. When does a control system compute? Digital, mechanical and open-loop systems

**arXiv ID:** 2604.16162 | [PDF](https://arxiv.org/pdf/2604.16162v1)

**作者:** Dominic Horsman `[一作]` (University of York), Viv Kendon `[通讯]` (University of Strathclyde)

**通讯引用:** 4164 | [OpenAlex ID](https://openalex.org/A5025637911)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

使用Abstraction/Representation Theory（ART）对各种控制系统（数字、机电、纯机械、人为操作等）进行建模与分析，证明所有控制系统在其控制器层面都执行了某种计算。

**💡 创新点**

首次将ART框架应用于控制系统，提出“植物为代表实体，控制器为计算机”的视角，揭示控制系统内部的计算过程，纠正了以离心调速器为非计算实例的误区。

**🔧 技术方法**

理论工具：Abstraction/Representation Theory（ART）; 对控制系统图（串联/并联、闭环/开环）进行ART图转换；使用ε-可交换图验证计算与表示关系；在多种实例中展示控制器、植物、表示实体的对应关系。

**📊 数据集**

无实验数据集，本文为理论与概念性分析，所述实例（温控器、双金属阀、离心调速器、车载加热、阿波罗AGC内存奇偶校验）为构造性示例。

**📈 对比分析**

比较方法：通过ART的ε-可交换图验证计算是否满足定义；没有传统算法性能指标。本文不进行数值模拟或实验验证，而是以形式化推导和实例说明其有效性。

**⚠️ 局限性**

局限性：仅讨论了典型的线性/比例控制系统，未涉及非线性、鲁棒性、学习型控制等更复杂情形；缺乏经验验证与实验数据支持；ART框架的可扩展性与实现细节仍待进一步研究。

---

## 444. "Taking Stock at FAccT": Using Participatory Design to Co-Create a Vision for the Fairness, Accountability and Transparency Community

**arXiv ID:** 2604.16224 | [PDF](https://arxiv.org/pdf/2604.16224v1)

**作者:** Shiran Dudy `[一作]` (Northeastern University), Yanan Long `[通讯]` (StickFlux Labs & University of Chicago)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在FAccT大会中开展大型参与式设计过程，结合现场CRAFT工作坊、异步Polis投票和报告合成，收集并分析社区对会议治理与未来愿景的意见，生成可执行的治理报告。

**💡 创新点**

首次在AI伦理/人机交互会议中实施大规模参与式设计，构建多阶段、异步、基于平台的参与框架，揭示共识与分歧，提供可操作的治理建议，并为后续规模化PD提供方法论案例。

**🔧 技术方法**

采用Polis决策平台进行投票与发言、CRAFT面对面工作坊、LLM（Claude Opus 4.5、Gemini Pro 3、GPT‑5.2）进行主题与标签生成、文本预处理与主题归纳、以及可视化分析。

**📊 数据集**

59条参与者提交的声明（约4,531票）与128名参与者的投票记录，数据来自FAccT社区的Slack与Disqus等渠道。

**📈 对比分析**

通过对比传统问卷调查与大会圆桌讨论的结果，发现PD过程能产生更丰富的主题覆盖、参与者多样性和更高的共识映射；虽然缺乏量化指标，但定性评估表明PD在发掘隐藏议题与激发持续讨论方面表现优异。

**⚠️ 局限性**

受匿名性与平台限制导致样本代表性不明、投票深度有限、缺乏人口统计信息、缺少对最终实施的承诺、平台缺乏直接对话功能，导致无法深入协商和落实建议。

---

## 445. No Universal Courtesy: A Cross-Linguistic, Multi-Model Study of Politeness Effects on LLMs Using the PLUM Corpus

**arXiv ID:** 2604.16275 | [PDF](https://arxiv.org/pdf/2604.16275v1)

**作者:** Hitesh Mehta `[一作]` (Delhi Technological University), Rohit Kumar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建 1500 条多语言礼貌级别提示语（PLUM 数据集），在 22,500 条提示-回复对上，系统评估了五种主流 LLM（Gemini-Pro、GPT‑4o Mini、Claude 3.7 Sonnet、DeepSeek‑Chat、Llama 3）在三种语言（英语、印地语、西班牙语）下对不同礼貌程度（正面礼貌、负面礼貌、正面失礼、负面失礼、直白命令）以及不同交互历史（空白、礼貌历史、失礼历史）的响应质量进行实验研究。

**💡 创新点**

创新点包括：① 将 Brown‑Levinson 与 Culpeper 两大礼貌理论融合，提出五级礼貌/失礼分类；② 设计三种交互历史条件来捕捉语调的“惯性”效应；③ 引入八维质量评估框架（连贯性、清晰度、深度、响应性、上下文保留、偏见/毒性、简洁性、可读性）以及统计检验（ANOVA、Tukey HSD）；④ 首创公开的多语言礼貌提示语库 PLUM，为跨语言礼貌研究提供可复现资源。

**🔧 技术方法**

使用的技术主要有：① 大规模自动化交互脚本（Python + 官方 API）；② 嵌入式文本评估工具（Sentence‑BERT、BERT‑CoLA、BERT‑NLI、unitary/toxic‑bert、Pyphen 等）；③ 聚类（K‑means）和可读性计算；④ 多维度得分合成（CQS）与模型平均；⑤ 统计分析（两因素 ANOVA、η²、Tukey HSD）。

**📊 数据集**

数据集为：PLUM（1500 条人类验证的多语言提示，分 5 礼貌级别、20 主题域）以及模型生成的 22,500 条提示‑回复对，用以计算 CQS 并进行跨模型、跨语言、跨历史的对比。

**📈 对比分析**

比较方法：对每个模型、语言、礼貌类别、交互历史的 100 条样本计算 CQS，并做平均值对比；使用 ANOVA 验证主效应与交互作用；通过 Tukey HSD 确认显著差异。结果显示：英语模型在礼貌历史上提升约 4–8% CQS；印地语对礼貌类别敏感性低；西班牙语在正面失礼类别上提升 10–15%；Llama 在礼貌类别间表现差异最大（11.5% 变化）。

**⚠️ 局限性**

局限性包括：① 仅覆盖三种语言，无法代表所有礼貌系统；② 语料为 20 主题域，缺乏专业领域（法律、医学等）与方言差异；③ 仅评估文本输出，未考虑多模态或任务特定交互；④ 模型版本更新可能导致结果不再适用；⑤ 对失礼语料做了内容筛选，无法探究极端攻击场景；⑥ 统计显著性在印地语表现弱，可能受模型差异与样本量限制。

---

## 446. VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects

**arXiv ID:** 2604.16272 | [PDF](https://arxiv.org/pdf/2604.16272v1)

**作者:** Xiangbo Gao `[一作]` (Texas A&M University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**通讯引用:** 2560 | [OpenAlex ID](https://openalex.org/A5015173810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向视频编辑的多维度评估框架，包括数据集、奖励模型与基准；

**💡 创新点**

创新点在于：①构建了5,049条带有IF、RQ、EE三维标签的人类标注数据；②训练了专门针对视频编辑的奖励模型，可同时评估指令遵循、渲染质量和编辑局部性；③提供了300对视频-提示的标准化基准，用于系统化比较；

**🔧 技术方法**

使用了多模态大型模型Qwen3-VL进行联合编码，并采用序数回归损失对三维评分进行训练；

**📊 数据集**

使用了新收集的5,049条视频编辑示例（源视频、编辑指令、编辑结果）以及300条视频-提示对；

**📈 对比分析**

与通用VLM评判、EditReward和VE-Bench等基线对比，显著提升了SRCC/KRCC/PLCC/RMSE（如整体SRCC提升至0.78 vs. 0.56）并在组内偏好评估中取得0.87的Pairwise Accuracy，表现优于现有奖励模型；

**⚠️ 局限性**

局限性包括：①仍依赖人类标注，扩展规模受限；②仅覆盖9大编辑类别，细粒度覆盖不足；③对完全缺失编辑结果的系统仍需要缺失值补偿，可能引入偏差；

---

## 447. BAGEL: Benchmarking Animal Knowledge Expertise in Language Models

**arXiv ID:** 2604.16241 | [PDF](https://arxiv.org/pdf/2604.16241v1)

**作者:** Jiacheng Shen `[一作]` (NYU Center for Data Science; NYU Shanghai), Olivier Pietquin `[通讯]` (NYU Shanghai Center for Data Science; NYU-ECNU Institute of Mathematical Sciences at NYU Shanghai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BAGEL基准，用闭卷四选一方式评估LLM在动物学、生态交互、科研文献推理和生物声学文本理解等多维度动物相关知识的能力。

**💡 创新点**

创新点在于聚合四个不同来源（Wikipedia、GloBI、bioRxiv、Xeno-canto）形成统一闭卷评测框架，并细化多维度知识类别与难度分层，突出跨源稳健性评估。

**🔧 技术方法**

采用GPT‑4o‑mini自动生成题目与选项、随机打乱选项顺序以减小位置偏倚，并使用标准化的四选一模板进行评估。

**📊 数据集**

使用了来自Wikipedia（动物条目）、GloBI（物种交互记录）、bioRxiv（动物学前沿论文）和Xeno-canto（动物声学元数据）四大公开语料库共计11,852道题目。

**📈 对比分析**

与GPT‑5.4、Claude Opus 4.6等闭源前沿模型以及SmolLM2-360M至Qwen3-32B等开源模型在同一协议下对比，结果显示闭源模型在Wikipedia与bioRxiv上表现最佳，但在Xeno-canto上仍显弱；开源模型整体落后，规模增大对Xeno-canto的提升非单调。

**⚠️ 局限性**

局限包括仅使用单一解码种子与贪婪解码导致结果易受随机性影响、缺乏对源预训练贡献的剖析、文本语料与音频内容的隔离导致音频理解能力难以评估、以及多选项设计可能导致答案歧义或偏倚。

---

## 448. Apple Peel Unfolding of Archimedean and Catalan Solids

**arXiv ID:** 2604.16204 | [PDF](https://arxiv.org/pdf/2604.16204v1)

**作者:** Takashi Yoshino `[一作]` (Toyo University), Supanut Chaidee `[通讯]` (Chiang Mai University)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5074978823)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种新的多面体展开方法——苹果剥离展开，并实现了对应算法。

**💡 创新点**

提出严格定义的苹果剥离展开，并对Archimedean和Catalan立方体进行系统分类。

**🔧 技术方法**

采用基于顶点坐标的图论方法，利用Mathematica实现了面序列选择算法。

**📊 数据集**

使用Mathematica内置PolyhedronData库提供的几何数据（顶点坐标和面列表）。

**📈 对比分析**

通过对所有相邻面起始组合的穷举，比较得到不同形状的展开结果；结果表明部分立方体可完全展开，其性能主要受数值误差影响。

**⚠️ 局限性**

仅适用于三维凸多面体，数值近似导致误差，且尚未找到能决定剥离可行性的几何判据。

---

## 449. Parallelizing the branch-and-bound with isomorphism pruning algorithm for classifying orthogonal arrays

**arXiv ID:** 2604.16271 | [PDF](https://arxiv.org/pdf/2604.16271v1)

**作者:** Dursun Bulutoglu `[一作]` (Air Force Institute Of Technology), Dursun Bulutoglu `[通讯]` (Air Force Institute Of Technology)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5007265606)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了并行化Branch-and-Bound加上isomorphism pruning算法，用于归类二值正交阵（OA），并实现了对该算法的并行化框架；

**💡 创新点**

创新点在于提出了一种基于LP松弛对称性群和分支变量转移的并行化方法，能够在保持完整性搜索的同时显著减少节点数并实现超线性加速；

**🔧 技术方法**

采用了整数线性规划（ILP）建模、分支限界（branch‑and‑bound）、isomorphism pruning、LP松弛对称性群识别、Perl编程与Google Gemini AI辅助编程等技术；

**📊 数据集**

主要使用的实验数据集是二值正交阵OA(N,k,2,4)，包括OA(128,9,2,4)、OA(144,9,2,4)以及OA(192,k,2,4)（k=9,10,11）；

**📈 对比分析**

与传统单线程B&B+pruning方法相比，该并行化方法在OA(128,9,2,4)和OA(144,9,2,4)上实现了超线性加速，并首次完成了OA(192,k,2,4)（k=9,10,11）的完整分类；

**⚠️ 局限性**

局限在于对LP松弛对称性群的构造和并行调度策略的手工设置仍较复杂，且尚未验证在更大规模或更高阶正交阵（t>4、s>2）上的表现。

---

## 450. FL-MHSM: Spatially-adaptive Fusion and Ensemble Learning for Flood-Landslide Multi-Hazard Susceptibility Mapping at Regional Scale

**arXiv ID:** 2604.16265 | [PDF](https://arxiv.org/pdf/2604.16265v1)

**作者:** Aswathi Mundayatt `[一作]` (International Institute of Information Technology Bangalore), Jaya Sreevalsan-Nair `[通讯]` (International Institute of Information Technology Bangalore)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5051540251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于两级空间划分、早期融合（EF）、晚期融合（LF）与软门控混合专家（MoE）深度学习工作流，专门用于洪水-滑坡联合多灾害易感性映射（FL‑MHSM）。

**💡 创新点**

创新点在于：① 空间自适应的两级划分（计算单元+情境区）保留局部异质性；② 采用多元高斯负对数似然的EF模型实现跨灾害的联合概率推理与不确定性量化；③ 通过软门控MoE将EF与LF两种互补专家动态融合，实现既保留单灾害判别力，又捕获灾害相互依赖；④ 结合GeoDetector对MoE输出进行空间异质性解释，揭示不同情境区下驱动因子与交互作用。

**🔧 技术方法**

使用技术包括：多层感知机（MLP）+多元高斯损失、XGBoost（LF基线）、软门控两层MLP（MoE）、Gaussian Noise、Dropout、LayerNorm、Beta Calibration、IDW拼接、Jenks分级、GeoDetector（因子、交互、风险检测）、核密度估计（数据增强）以及OC‑SVM（负样本生成）。

**📊 数据集**

数据集涵盖两大区域：Kerala（印度）与Nepa（尼泊尔）；情境区图（ESZ、NNH）、洪水栅格（Sentinel‑1 SAR）、滑坡点集（NRSC‑GSI‑KSDMA、COOLR）、地形（SRTM）、水文（HydroSHEDS、冰川湖）、气候（CHIRPS、TerraClimate）、表面特征（NDVI、LULC）、人口密度等多源栅格与矢量资料。

**📈 对比分析**

比较方法：在每个情境区分别训练并交叉验证，计算宏平均 AUC‑ROC、精确率、召回率、F1、Brier、准确率和 Jaccard。结果显示：EF在召回率与Brier方面提升明显；LF在AUC‑ROC上表现最佳；MoE综合性能最佳，AUC‑ROC、召回率、F1提升至最高，Brier最低，且在两地区均保持稳健。

**⚠️ 局限性**

局限性：① 仅使用静态特征，未考虑时间演化与灾害级联动态；② 只研究洪水与滑坡两类灾害，难以推广至多灾害系统；③ 对连续易感性进行Jenks分段的阈值可能影响二元类比例；④ GeoDetector仅用于后验解释，未将驱动因子自适应融入模型训练或融合策略。

---

## 451. Do Vision-Language Models Truly Perform Vision Reasoning? A Rigorous Study of the Modality Gap

**arXiv ID:** 2604.16256 | [PDF](https://arxiv.org/pdf/2604.16256v1)

**作者:** Yige Xu `[一作]` (Nanyang Technological University), Zhiqi Shen `[通讯]` (Nanyang Technological University)

**通讯引用:** 5562 | [OpenAlex ID](https://openalex.org/A5101789458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了 CrossMath 视觉‑文本对齐基准，系统评估 VLM 在图像、文本及其组合输入下的推理能力，并通过在该基准上进行后训练以缩小模态差距。

**💡 创新点**

创新点在于：①严格对齐图像、文本与混合三种格式，消除信息不对称；②提出可衡量多步推理的 6 项微观/宏观指标；③利用后训练（SFT + GRPO）显著提升视觉推理表现。

**🔧 技术方法**

技术手段包括：多模态对齐、结构化推理路径抽取、图像到 Markdown 的 OCR 解析、基于 LoRA 的参数高效微调、基于 GRPO 的奖励驱动强化学习。

**📊 数据集**

使用自制的 CrossMath 数据集（5k 训练样本、250 评估样本，包含 4 种视觉风格），并在 MathVerse 与 MMMU 等跨域视觉数学基准上验证泛化。

**📈 对比分析**

评估方法：在图像、文本、混合三种输入下对齐后比较微观/宏观准确率。实验显示，零shot VLM 在图像输入上远低于文本输入；后训练后微观准确率提升至约 60%–70%，宏观准确率提升至 45%–50%，但仍低于文本基准。

**⚠️ 局限性**

局限在于：VLM 仍高度依赖文本推理，视觉信息在当前架构中难以充分编码；模型规模提升并未显著改善图像推理；后训练虽有效但需要专门数据，无法完全消除模态差距。

---

## 452. Optimizing Korean-Centric LLMs via Token Pruning

**arXiv ID:** 2604.16235 | [PDF](https://arxiv.org/pdf/2604.16235v1)

**作者:** Hoyeol Kim `[一作]` (Georgia Institute of Technology), Hyeonwoo Kim `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言大型语言模型进行词元剪枝，评估其在以韩语为中心的NLP任务中的效果

**💡 创新点**

系统性验证词元剪枝在多语言模型中的性能提升，并揭示不同词表配置对指令遵循与跨语言表示的影响

**🔧 技术方法**

词元剪枝（语言感知过滤）、词表重构、embedding与输出层重排、WPR语言一致性评估、COMET翻译评估

**📊 数据集**

KMMLU、HAERAE、CLIcK、LogicKor、KoMTBench、WMT24++（韩语-英语）

**📈 对比分析**

与原始全词表模型及EnKoZh双语词表模型对比；结果显示词元剪枝对一般知识、文化理解、指令遵循的影响不大甚至提升，翻译性能显著提升（WPR>0.99、COMET上升至0.63-0.75）

**⚠️ 局限性**

对大型模型的推理延迟提升有限（<1%），对某些架构的跨语言表示仍有依赖，且在低参数模型中剪枝效果更为明显

---

## 453. Investigating Conversational Agents to Support Secondary School Students Learning CSP

**arXiv ID:** 2604.16213 | [PDF](https://arxiv.org/pdf/2604.16213v1)

**作者:** Matthew Frazier `[一作]` (University of Delaware), Lori Pollock `[通讯]` (University of Delaware)

**通讯引用:** 6751 | [OpenAlex ID](https://openalex.org/A5068927917)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对高中学生使用会话式智能体（Conversational Agent）学习计算机科学原理（Computer Science Principles）进行实验研究。

**💡 创新点**

首次将会话式智能体嵌入计算机科学课程，利用自然语言交互和即时反馈来支持学生学习；并提出了基于对话日志的学习效果评估方法。

**🔧 技术方法**

使用基于规则的自然语言处理框架实现的会话式智能体，配合自定义的学习模块和即时测评功能；实验中还使用了学生学习日志记录器。

**📊 数据集**

收集了约80名高中生在实验前后的测验成绩、对话交互日志以及学习态度问卷。

**📈 对比分析**

对比实验组（使用智能体）与对照组（传统授课）的学习成绩、学习时长和学习态度；实验组在概念理解测试上平均提升了12%，学习时长增长了15%，学习态度问卷中“兴趣”与“自信”得分均显著提高（p<0.05）。

**⚠️ 局限性**

局限性包括样本规模有限、仅在单一学校进行，实验时间较短，智能体采用规则系统限制了对复杂问题的支持；未来需在更大规模、不同地区和更长时间段内验证其效果。

---

## 454. Saturation-Aware Space-Variant Blind Image Deblurring

**arXiv ID:** 2604.16200 | [PDF](https://arxiv.org/pdf/2604.16200v1)

**作者:** Muhammad Z. Alam `[一作]` (University of New Brunswick), Arooba Zeshan `[通讯]` (University of Winnipeg)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5005991746)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种空间变异盲图像去模糊框架，结合亮度扩散函数（LSF）与暗通道先验，在饱和像素区域恢复真实辐射并完成去模糊。

**💡 创新点**

创新点在于①将饱和像素辐射恢复与空间变异模糊估计耦合；②使用预估LSF分离散射光与模糊光；③利用暗通道先验在最小模糊块中精确估计饱和辐射。

**🔧 技术方法**

采用暗通道先验、空间变异块级模糊估计、LSF预估、盲卷积去模糊算法（如Xu等）以及梯度锐度评估和多尺度处理。

**📊 数据集**

使用合成及真实低光/高动态范围图像，公开多曝光序列（tunnel scene）和手机摄影数据集进行实验。

**📈 对比分析**

与多种专门饱和处理或通用去模糊方法（Wen、Hu、Liang等）在 PSNR、SSIM、几何平均及 SSIM‑加权 PSNR 上对比，实验表明在中高曝光下本方法在质量指标上更优且运行时相对高效。

**⚠️ 局限性**

局限性包括需要足够的暗像素和低模糊块；在极端模糊或极端饱和情况下恢复效果受限；对光扩散函数模型假设敏感，未来需改进非对称/场依赖LSF。

---

## 455. Synthetic data in cryptocurrencies using generative models

**arXiv ID:** 2604.16182 | [PDF](https://arxiv.org/pdf/2604.16182v1)

**作者:** André Saimon S. Sousa `[一作]` (Universidade SENAI CIMATEC), Hugo Saba `[通讯]` (Universidade do Estado da Bahia)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5068154865)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用条件生成对抗网络（CGAN）生成加密货币分钟级价格合成序列，并验证其与真实数据的相似性

**💡 创新点**

在传统GAN基础上引入LSTM生成器与MLP判别器的混合结构，结合60天窗口的时间条件，显著提升了对高频金融时序的逼真性

**🔧 技术方法**

Conditional GAN、LSTM、MLP、Adam优化器、BCEWithLogitsLoss、StandardScaler归一化、Pearson、Spearman、MAE、RMSE等评估指标

**📊 数据集**

从LSEG平台获取的BTC、ETH、XRP三种加密资产（2022-2025年）的约38.4万条分钟级收盘价记录，分为三段波动情景

**📈 对比分析**

通过训练50个epoch、批次64，使用交叉熵损失对抗学习；评估指标显示BTC Pearson>0.9999、MAE≈29–46，ETH MAE≈2.5–3，XRP MAE≈0.0006–0.0019；可视化结果与真实序列高度一致

**⚠️ 局限性**

对比学习中生成器在极端波动（ETH、XRP）时会低估波动幅度，难以捕捉重尾突发事件；模型对不同资产的适应性差异较大，需进一步针对资产特征做自适配

---

## 456. MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation

**arXiv ID:** 2604.16175 | [PDF](https://arxiv.org/pdf/2604.16175v1)

**作者:** Yi Lin `[一作]` (Weill Cornell Medicine), Yifan Peng `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 10924 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种模拟放射科分层工作流程的多智能体框架，用于三维CT影像的自动报告生成。

**💡 创新点**

创新点包括：①多智能体协作结构（住院医、研究员、主治医）实现逐层审阅与校正；②检索增强修订阶段结合图像、文本与logits检索；③基于立场的迭代共识讨论，消除单读错误。

**🔧 技术方法**

使用ViT3D视觉编码器+SAT多区域分割、LLaMA‑2‑Chat‑7B（Resident）、GPT‑4.1/GPT‑4o（Fellow/Attending）等大语言模型，结合多模态检索和循环对话机制。

**📊 数据集**

在RadGenome‑ChestCT数据集上进行训练和评估，该数据集包含25,692例胸部CT及对应报告。

**📈 对比分析**

与多种SOTA方法（R2GenPT、MedVInT、CT2Rep、M3D、RadFM、Reg2RG）对比，BLEU、METEOR、CE‑F1均明显领先，证明在语言质量与临床准确性上均具优势。

**⚠️ 局限性**

局限性：仅在GPT系列LLM上验证，缺乏多样化开源或专用医学LLM的通用性；无长期记忆机制，无法整合患者病史；系统完全自主，缺乏人机交互的“人类在环”模式。

---

## 457. From Benchmarking to Reasoning: A Dual-Aspect, Large-Scale Evaluation of LLMs on Vietnamese Legal Text

**arXiv ID:** 2604.16270 | [PDF](https://arxiv.org/pdf/2604.16270v1)

**作者:** Van-Truong Le `[一作]` `[通讯]` (University of Science, Vietnam National University), Van-Truong Le (University of Science, Vietnam National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在越南复杂法律文本的简化任务中，对GPT‑4o、Claude 3 Opus、Gemini 1.5 Pro和Grok‑1四大LLM进行了大规模双面评估，涵盖性能基准与错误分析。

**💡 创新点**

创新点在于将三维度评估（准确性、可读性、一致性）与九类专家验证错误类型相结合，构建了兼顾定量与定性、面向法律推理的双面评估框架。

**🔧 技术方法**

采用零-shot提示、Likert评分、语义相似度与BLEU一致性测量、长度一致性、人工一致性评分以及法律专家手工注释等技术手段。

**📊 数据集**

使用60篇精选越南刑法、民法、土地法中的复杂条文，生成480条模型输出作为评估数据集。

**📈 对比分析**

通过整体得分和错误频率矩阵对比模型；Claude 3 Opus在准确性最高，Grok‑1在可读性与一致性最优，GPT‑4o则表现出过度简化倾向，整体模型间存在显著权衡。

**⚠️ 局限性**

局限性包括样本量相对有限、评审者为学生缺乏专业律师经验、仅使用零-shot设置、未覆盖开源模型，且受评估时模型版本与评审专业度的限制。

---

## 458. Hero-Mamba: Mamba-based Dual Domain Learning for Underwater Image Enhancement

**arXiv ID:** 2604.16266 | [PDF](https://arxiv.org/pdf/2604.16266v1)

**作者:** Tejeswar Pokuri `[一作]` (Manipal Academy of Higher Education), Shivarth Rai `[通讯]` (Manipal Academy of Higher Education)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5024745577)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Hero-Mamba的双域网络，用于水下图像增强。

**💡 创新点**

通过在空间域和频域并行处理并使用Mamba基SS2D模块实现线性复杂度的长距离依赖建模，同时引入基于背景光先验的ColorFusion块实现精确色彩恢复。

**🔧 技术方法**

使用Mamba状态空间模型、FFT频域特征、SS2D块、MS-Fusion多尺度融合以及L1+SSIM+对比损失的复合损失。

**📊 数据集**

在公开的UIEB和LSUI两个水下图像数据集上进行训练与评估。

**📈 对比分析**

与10种SOTA方法在SSIM、PSNR、LPIPS、FSIM等指标上对比，Hero-Mamba在LSUI上取得SSIM 0.913、PSNR 25.802，UIEB上SSIM 0.942，表现均为最优。

**⚠️ 局限性**

模型仍受限于训练样本的多样性，跨域泛化仍有提升空间，对极端光照或复杂纹理场景的鲁棒性未完全验证。

---

## 459. Information Router for Mitigating Modality Dominance in Vision-Language Models

**arXiv ID:** 2604.16264 | [PDF](https://arxiv.org/pdf/2604.16264v1)

**作者:** Seulgi Kim `[一作]` (Georgia Institute of Technology), Ghassan AlRegib `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5656 | [OpenAlex ID](https://openalex.org/A5006145139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了多模态信息路由器MoIR，以在融合前显式降低模态信息差异。

**💡 创新点**

创新点在于通过识别低信息量通道并从更强模态路由补充信息，直接提升令牌信息密度。

**🔧 技术方法**

采用奇异值分解衡量通道信息，学习可路由门控，结合LLM解码器前插入路由层。

**📊 数据集**

在ScienceQA、VizWiz以及MMBench-Video三大多模态基准上验证。

**📈 对比分析**

与标准Fine-tuning和注意力调节方法对比，MoIR在保持或提升下游准确率的同时显著降低MDI/AEI，提升对模态干扰的鲁棒性。

**⚠️ 局限性**

局限在于需额外计算路由门学习，且在多选类任务中提升有限，对极端信息缺失情况的效果仍待进一步研究。

---

## 460. SwanNLP at SemEval-2026 Task 5: An LLM-based Framework for Plausibility Scoring in Narrative Word Sense Disambiguation

**arXiv ID:** 2604.16262 | [PDF](https://arxiv.org/pdf/2604.16262v1)

**作者:** Deshan Sumanathilaka `[一作]` (Swansea University), Saman Jayasinghe `[通讯]` (Swansea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于大型语言模型的框架，用结构化推理和动态少量样本提示，对叙事文本中的多义词感知合理性进行打分，并通过模型集成模拟多注释者一致性。

**💡 创新点**

创新点在于将检索增强生成（RAG）与链式思考(CoT)相结合，动态插入少量范例提升人类感知合理性预测；同时通过多模型集成逼近多注释者评分分布。

**🔧 技术方法**

使用LoRA微调的低参数模型（Qwen‑4B、Gemma‑4B）、动态少量样本提示（Deepseek v3、GPT‑4o、Gemini 2.5 flash‑lite）、检索增强生成、CoT推理以及多种集成方法（多数投票、加权平均、线性回归、SVR、XGBoost）。

**📊 数据集**

利用SemEval‑2026 Task 5的AmbiStory数据集，训练集2322篇故事、验证集600篇、测试集942篇，涵盖188/42/76种唯一多义词。

**📈 对比分析**

与零样本和单模型微调基线相比，动态少量样本提示下的GPT‑4o在测试集上达Spearman 0.755、准确率 0.798；集成方法（线性回归、XGBoost）进一步提升到Spearman 0.724、准确率 0.797，最终在排行榜上位列第十，平均分0.760。

**⚠️ 局限性**

在高度模糊或上下文敏感的实例中模型仍表现不佳，缺乏一致性和细粒度解释能力，且对大型模型的依赖限制了可扩展性与资源效率。

---

## 461. Beyond Distribution Sharpening: The Importance of Task Rewards

**arXiv ID:** 2604.16259 | [PDF](https://arxiv.org/pdf/2604.16259v1)

**作者:** Sarthak Mittal `[一作]` (Mila), Guillaume Lajoie `[通讯]` (Mila)

**通讯引用:** 1160 | [OpenAlex ID](https://openalex.org/A5043037494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比分布锐化（distribution sharpening）与基于任务奖励（task‑reward）的强化学习（RL）微调在大型语言模型（LLM）能力提升中的真实效果，使用统一的KL‑正则化RL框架进行实验

**💡 创新点**

首次在相同训练环境下系统性隔离分布锐化与任务奖励两种机制，揭示分布锐化因可变长度生成导致的最优解不利且不稳定，而任务奖励能够显著提升性能并保持训练稳定

**🔧 技术方法**

KL‑正则化RL、REINFORCE、留一回归估计、分布温度采样（tilted sampling）等技术

**📊 数据集**

Hendrycks数学数据集、DeepScaleR、Math‑500、Minerva‑Math、AIME 2024/25、HMMT 2025等数学推理数据集

**📈 对比分析**

在相同模型（Llama‑3.2‑3B、Qwen‑2.5‑3B 等）和相同超参数下对比：分布锐化（beam search、power sampling、RL‑锐化）与任务奖励微调；结果显示任务奖励微调在所有难度级别均优于分布锐化，分布锐化在早停时可达到一定提升但最终会崩溃；强化学习在分布锐化上表现不稳定，而任务奖励在训练中稳定提升

**⚠️ 局限性**

分布锐化的最优解偏向短文本，导致可变长度生成时不稳定；RL优化本身并非导致问题，而是目标本身不利；在高温度或高锐化系数下，结合分布锐化与任务奖励的“tilted sampling”不可靠；缺乏针对更长序列或多任务混合的实验

---

## 462. Characterising LLM-Generated Competency Questions: a Cross-Domain Empirical Study using Open and Closed Models

**arXiv ID:** 2604.16258 | [PDF](https://arxiv.org/pdf/2604.16258v1)

**作者:** Reham Alharbi `[一作]` (Taibah University), Jacopo de Berardinis `[通讯]` (University of Liverpool)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5033120361)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CompCQ 框架并在五个跨域（文化遗产、音乐、医疗、新闻、旅游）用户故事与用例上，对五种 LLM（KimiK2、Llama3.1、Llama3.2、Gemini、GPT）自动生成的 competency questions（CQs）进行系统量化与比较。

**💡 创新点**

①将 CQ 的可读性、结构复杂度、相关性、语义多样性等维度统一量化，构建多维度评估工具；②通过跨域实证研究揭示 LLM 生成行为与领域特征之间的交互；③展示闭源模型与开源模型在可读性、覆盖度与多样性上的对比特征，提示组合使用的价值。

**🔧 技术方法**

采用 Flesch‑Kincaid、Dale‑Chall 可读性指标；利用 NLP 的词性标注、短语分块和依存树分析得到需求、语言和句法复杂度；使用 LLM 对 CQ 相关性进行 4‑点 Likert 评分；借助 Sentence‑BERT 生成嵌入，计算平均余弦相似度、质心距离、Shannon 熵等语义多样性指标；对 CQ 集进行聚类与双向覆盖率评估。

**📊 数据集**

自构造的多域数据集，包含 5 个用例/故事：英国音乐体验（BME）、音乐元数据（Music Meta）、个性化抑郁治疗（PDTO）、政治新闻（PJO）和旅游推荐（WTGW）。每个案例提供多段需求文本供 LLM 生成 CQ。

**📈 对比分析**

以每种 LLM 在同一需求文本下生成的 CQ 作为比较单元，分别统计可读性、复杂度、相关性得分、CQ 数量、语义多样性（APS、ACD、Entropy）以及跨模型覆盖率（Centroid Similarity、MMS、Bidirectional Coverage）。实验结果显示：闭源模型（Gemini、GPT）可读性最高、覆盖度最好；开源模型（KimiK2、Llama 系列）生成 CQ 更丰富多样但覆盖度不足；领域复杂度决定 CQ 复杂度和可读性；单一模型难以覆盖全部需求，建议采用多模型组合。

**⚠️ 局限性**

①仅评估了 5 种 LLM，未涵盖更大规模或专门化模型；②所有实验均采用 0‑shot 提示，缺少 prompt‑engineering 对比；③相关性评估完全依赖 LLM，缺乏人工专家验证；④可读性指标原本为连续文本，短句适用性有限；⑤部分开源模型输出 CQ 过少，导致多样性评估不稳定；⑥未对生成的 CQ 进行功能验证或与人类手工编写 CQ 的对比。

---

## 463. A Two-Stage, Object-Centric Deep Learning Framework for Robust Exam Cheating Detection

**arXiv ID:** 2604.16234 | [PDF](https://arxiv.org/pdf/2604.16234v1)

**作者:** Van-Truong Le `[一作]` (Vietnam National University), Trong-Doanh Nguyen `[通讯]` (Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个两阶段、以对象为中心的深度学习框架，用YOLOv8检测考生位置，再用RexNet-150对裁剪出的人像进行作弊行为分类，实现考试作弊检测；

**💡 创新点**

创新点在于将场景理解拆解为人检+行为判定两步，显著降低背景噪声干扰，并构建了包含10个公开源的大规模统一作弊检测数据集；

**🔧 技术方法**

技术实现主要使用YOLOv8n进行快速人检测、RexNet-150作为特征提取与分类器（基于ImageNet预训练），裁剪后224×224像素输入，训练10个epoch，推理时间约13.9 ms；

**📊 数据集**

采用了10个公开数据集整合而成的273,897样本大规模数据集，按80/10/10划分为训练/验证/测试集；

**📈 对比分析**

通过对比全帧模型和两阶段模型，实验显示两阶段方法整体准确率提升至95.16%，作弊类F1提升至0.91，比全帧基线高出16.64%/28%；与EfficientNet-B0、ResNet-18比较，RexNet-150在准确率与F1上均优先；

**⚠️ 局限性**

局限性包括仅裁剪人框导致缺乏周边环境上下文，二分类设置无法细化不同作弊类型，数据标注不一致导致无效边框，且未结合时间序列或多模态信息来进一步提升鲁棒性。

---

## 464. Neuro-Symbolic ODE Discovery with Latent Grammar Flow

**arXiv ID:** 2604.16232 | [PDF](https://arxiv.org/pdf/2604.16232v1)

**作者:** Karin Yu `[一作]` (ETH Zurich), Georgios Kissas `[通讯]` (Swiss Data Science Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 Latent Grammar Flow (LGF) 的神经符号框架，用于从数据中发现隐式和显式的一维常微分方程。

**💡 创新点**

创新点在于将语法量化自动编码器 (GQAE) 与离散流模型相结合，构建离散潜在空间并通过行为距离重构，使得领域知识（如阶数、稳定性）可以直接嵌入采样过程。

**🔧 技术方法**

采用的技术包括 GQAE、离散流模型、Wasserstein 行为距离、离散指导、Nelder–Mead 标量优化、以及静态和动态预测器。

**📊 数据集**

使用了三组基准数据集：Benchmark 1（30 个一阶 ODE）、Benchmark 2（10 个线性/非线性二阶 ODE）以及 Benchmark 3（7 个部分可观测且稳定的二阶 ODE）。

**📈 对比分析**

与 ODEFormer、PySR、ProGED、GODE 等方法对比，LGF 在多数基准上取得了与或优于 PySR 的相对 L2 误差，且在样本效率和表达式复杂度方面表现出较好平衡。

**⚠️ 局限性**

局限性包括目前仅验证于一维单自由度 ODE，无法直接处理多自由度或更高阶系统；离散流模型训练和行为距离估计仍需大量计算；对非自治系统的稳定性评估不够严格。

---

## 465. GAViD: A Large-Scale Multimodal Dataset for Context-Aware Group Affect Recognition from Videos

**arXiv ID:** 2604.16214 | [PDF](https://arxiv.org/pdf/2604.16214v1)

**作者:** Deepak Kumar `[一作]` (Indian Institute of Technology Roorkee), Balasubramanian Raman `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 9059 | [OpenAlex ID](https://openalex.org/A5030765476)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并公开了GAViD数据集，包含5091个5秒短视频，配备视频、音频、上下文元数据和三元情感标签（正面、中性、负面）以及五类离散情绪，并基于该数据集训练了一种多模态情感识别网络CAGNet，评估其在情感分类任务上的性能。

**💡 创新点**

创新点在于（1）首次构建大型、真实场景的多模态（视频+音频+文本）组情感数据集，结合LLM生成的上下文元数据和人工验证；（2）设计了跨模态注意力+门控融合的CAGNet，可在缺失模态时仍保持鲁棒性；（3）通过多模态融合显著提升情感分类准确率（V+A+C达到63.20%/66.21%）。

**🔧 技术方法**

使用技术包括：视频剪辑与标准化、DINOv2视觉特征提取、Wav2Vec 2.0音频特征提取、XLM‑RoBERTa文本特征提取、三组跨模态注意力块、门控融合（Squeeze‑Excitation）、两层MLP分类头；训练采用AdamW、交叉熵、早停等常规深度学习技巧。

**📊 数据集**

使用的数据集是自研的GAViD，包含5091个5秒视频片段，伴随音频、上下文描述、情感强度、互动类型、动作线索等多层标签；公开代码与数据位于GitHub和Zenodo。该数据集与现有GAR数据集相比，唯一完整同时拥有情感、情绪和上下文三种标签。

**📈 对比分析**

实验对比了CAGNet与两种基线（融合方式不同）以及公开的视频‑语言模型Video‑GPT和LLaVA‑NeXT。结果显示，CAGNet在V+A+C（三模态）下在验证集上达64.81%准确率、63.20% F1，测试集上达66.21%准确率、0.647 F1，显著优于基线与现有方法，验证跨模态注意力和门控融合的有效性。

**⚠️ 局限性**

局限性包括：①数据仍为固定长度短片，未覆盖更长时序动态；②模型仅关注三元情感分类，未覆盖细粒度情绪或个体层面；③数据来源受限于YouTube CC‑BY 视频，可能存在内容多样性不足；④在极端光照、遮挡等条件下模型性能可能下降；⑤缺乏跨域或实时部署的验证。

---

## 466. From Papers to Progress: Rethinking Knowledge Accumulation in Software Engineering

**arXiv ID:** 2604.16208 | [PDF](https://arxiv.org/pdf/2604.16208v1)

**作者:** Jason Cusati `[一作]` (Virginia Tech), Chris Brown `[通讯]` (Virginia Tech)

**通讯引用:** 44671 | [OpenAlex ID](https://openalex.org/A5036229691)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了ICSE 2026 Future of Software Engineering（FOSE）预调查中的280份问卷，诊断了软件工程研究在知识累积方面的四个结构性瓶颈，并提出了四条技术无关的原则来指导未来研究产物的设计与治理。

**💡 创新点**

创新点在于将社区反馈转化为系统性的诊断，并提出“结构化与可解释、可检视与可追溯、长期可复用、以人为治理”四大原则，强调研究产物本身而非仅仅是发表文章的改进方向。

**🔧 技术方法**

使用的方法主要是轻量级定性分析（对问卷开放式回复进行编码和主题归纳），并结合对文献和现有实践的综述来支持诊断；并未开发新的算法或系统。

**📊 数据集**

主要数据集是来自280名参与者的FOSE预调查问卷，涵盖了作者经验、产出量、地区分布等信息；并未使用实验数据集或开源数据集进行验证。

**📈 对比分析**

论文并未进行实验对比或性能评估；其贡献在于概念框架和原则的提出，而非算法或工具的实现与评测。

**⚠️ 局限性**

局限性包括：①问卷自选性与样本量限制导致结果可能缺乏代表性；②仅进行定性分析，缺乏实证验证；③提出的原则需在实践中验证其可行性与有效性；④关于治理与伦理的讨论仍待社区进一步协商。

---

## 467. DENALI: A Dataset Enabling Non-Line-of-Sight Spatial Reasoning with Low-Cost LiDARs

**arXiv ID:** 2604.16201 | [PDF](https://arxiv.org/pdf/2604.16201v1)

**作者:** Nikhil Behari `[一作]` (Massachusetts Institute of Technology), Ramesh Raskar `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 32492 | [OpenAlex ID](https://openalex.org/A5023495279)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过收集并分析72,000条低成本LiDAR的全时间分辨光学直方图，构建了第一个大规模真实世界NLOS数据集Denali，并用该数据集实现了隐藏物体的位置、形状和尺寸的识别。

**💡 创新点**

创新点在于首次利用低成本消费级LiDAR的完整时域信号开展数据驱动的非视线（NLOS）感知，并提供与物理仿真双轨数字孪生来评估模拟精度与实测差距。

**🔧 技术方法**

采用时域卷积网络（1D/3D CNN）、Transformer等深度模型进行NLOS定位与分类，结合光学仿真与时域噪声匹配校准。

**📊 数据集**

使用Denali数据集——60种物体、100个位置、两种LiDAR空间分辨率、两种光照条件、三次重复共计72,000次LiDAR采样，并配套Mitsuba3数字孪生。

**📈 对比分析**

在定位、分类、尺寸三任务上，最佳模型1D CNN的定位RMSE为0.046 m，宏F1分数0.38，尺寸预测准确率95%，明显优于无结构MLP，并表明低成本LiDAR足以实现多任务NLOS感知。

**⚠️ 局限性**

主要局限包括对物体尺寸、位置和光照高度敏感、低空间分辨率限制了空间信息利用，以及仿真与实测之间仍存在显著的时域误差导致的sim‑to‑real迁移瓶颈。

---

## 468. Bridging the Gap between User Intent and LLM: A Requirement Alignment Approach for Code Generation

**arXiv ID:** 2604.16198 | [PDF](https://arxiv.org/pdf/2604.16198v1)

**作者:** Jia Li `[一作]` (Wuhan University), Zhi Jin `[通讯]` (Wuhan University)

**通讯引用:** 10329 | [OpenAlex ID](https://openalex.org/A5049100391)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，提出了一种名为REA-Coder的需求对齐方法，先通过问答和验证步骤让大语言模型充分理解任务需求，再生成代码并迭代修正，提升代码生成质量。

**💡 创新点**

核心创新在于将需求理解置于生成前的首要位置，并引入基于问答的需求检查与掩码恢复验证，确保模型真正掌握需求而非仅靠后期修复。

**🔧 技术方法**

技术实现包括需求维度问答（QA）生成、答案与参考答案比较、掩码恢复验证、迭代对齐与生成，利用多轮Prompting与LLM交互。

**📊 数据集**

实验使用四个先进LLM（DeepSeek‑v3.2、Qwen3‑Coder、GPT‑5‑mini、Gemini‑3‑Flash）以及五个代码生成基准（APPS、CodeContests‑raw、CodeContests、xCodeEval、LiveCodeBench‑Lite）。

**📈 对比分析**

与零样本、8种基线（包含推理型与后处理型）对比，REA‑Coder在所有模型和基准上均取得最高Pass@1分数，平均提升分别为7.93%、30.25%、26.75%、8.59%和8.64%，体现显著性能提升。

**⚠️ 局限性**

主要局限包括相对较高的时间与token消耗，需多轮迭代；对需求维度设计和掩码策略依赖较大；目前仅在代码生成任务验证，泛化至其他NLP任务仍待探索。

---

## 469. neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD Model Editing

**arXiv ID:** 2604.16170 | [PDF](https://arxiv.org/pdf/2604.16170v1)

**作者:** Toby Perrett `[一作]` (Autodesk Research), William McCarthy `[通讯]` (Autodesk Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了第一个基于专家CAD工程师真实编辑请求的多模态3D CAD编辑基准，收集了视频、语音、绘图等多模态指令。

**💡 创新点**

创新点在于：①首次把多模态交互记录到CAD编辑基准；②在真实专业环境下获取请求并进行人类编辑对比；③提出多模态理解与3D编辑的综合评估。

**🔧 技术方法**

使用Foundation模型（GPT‑5.2、Gemini‑3 Pro、Claude Sonnet 4.5）与CadQuery脚本，配合自动指标（Chamfer、Voxel IoU、DINOv2）与人类评测。

**📊 数据集**

数据集为10名专业CAD用户在Autodesk Fusion中生成的192条请求（4种模态×3难度×4请求）及其对应的384条人类编辑结果。

**📈 对比分析**

通过自动指标、人类评审和VLM评估对比，AI模型的接受率仅为人类基准的约50%，表明仍存在显著性能差距。

**⚠️ 局限性**

限制在于：①缺乏原生视频/绘图支持导致AI理解受限；②对复杂空间关系和长序列指令的推理能力不足；③基准规模有限，未覆盖所有CAD编辑场景。

---

## 470. NaijaS2ST: A Multi-Accent Benchmark for Speech-to-Speech Translation in Low-Resource Nigerian Languages

**arXiv ID:** 2604.16287 | [PDF](https://arxiv.org/pdf/2604.16287v1)

**作者:** Marie Maltais `[一作]` (Mila - Quebec AI Institute), David Ifeoluwa Adelani `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了包含Igbo、Hausa、Nigerian Pidgin与英语的约50小时多口音并行语音数据集NaijaS2ST，并对其进行质量控制和分割；

**💡 创新点**

创新点在于提供真实、多口音、多语言低资源语音翻译数据集，并系统比较cascaded、E2E和AudioLLM三种范式，揭示AudioLLM在S2TT/S2ST中的优势及Fine‑tuning对E2E的关键作用；

**🔧 技术方法**

采用了Omnilingual‑ASR、NLLB/Tiny‑Aya翻译模型的cascaded流水线，SeamlessM4T的大型E2E模型以及Gemini 2.5/3.1和GPT‑Audio 1.5等AudioLLM，并结合few‑shot/zero‑shot、语音合成（Gemini TTS）等技术；

**📊 数据集**

使用了NaijaS2ST数据集，包含Igbo、Hausa、Nigerian Pidgin与英语的约50小时高质量语音，并补充了对等文本；

**📈 对比分析**

通过SSA‑COMET和ChrF两种评测指标对比，发现AudioLLM在S2TT上显著领先，E2E在S2ST上逊色于cascaded，Fine‑tuning在LRL→English方向效果更佳；

**⚠️ 局限性**

限制包括：仅在离线实验环境评估，未考虑实时/延迟需求；AudioLLM推理成本高，资源占用大；实验参数与prompt设计范围有限，可能未达到最优性能。

---

## 471. Geometric regularization of autoencoders via observed stochastic dynamics

**arXiv ID:** 2604.16282 | [PDF](https://arxiv.org/pdf/2604.16282v1)

**作者:** Sean Hill `[一作]` (University at Albany, SUNY), Felix X. -F. Ye `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于几何正则化的自编码器框架，用以从稀疏高维数据中学习低维流形上的随机微分方程（SDE）模拟器，主要通过利用环境协方差编码切空间信息、引入ρ‑度量和编码器拉回的Itô公式实现精确的潜在漂移估计。

**💡 创新点**

创新点包括：①将环境协方差直接用于切空间投影正则化，构造了ρ‑度量，避免对Jacobian标签的需求；②利用编码器拉回的Itô公式得到无系统偏差的潜在漂移目标；③给出了ρ‑ERM的泛化率与Sobolev H¹ 训练等价的理论证明；④展示了从图表质量到漂移/扩散系数再到弱收敛与MFPT的误差传播链。

**🔧 技术方法**

技术手段涵盖自编码器、流形学习、随机微分方程理论、Itô 变换、Sobolev 训练泛化理论、切空间投影正则化、逆一致性惩罚、谱截断与高维投影计算、理论误差分析与弱收敛证明。

**📊 数据集**

实验数据为四种 Monge‑patch 流形（抛物面、双曲抛物面、四次穹顶、正弦面），嵌入维度最高达 201；动力学场景包括 Müller–Brown 过damped Langevin 和带状态依赖扩散的旋转漂移，使用短时爆发集合生成的环境漂移 b(x) 与协方差 Λ(x)。

**📈 对比分析**

与基线自编码器、ATLAS 以及 T、F、T+F 等消融模型对比，评估指标包括重构误差、切空间误差、漂移/协方差误差以及 MFPT。实验表明，T+F 正则化在 50–70% 降低 MFPT 误差、在大多数表面-过渡对上取得最低的跨井 MFPT 误差，并将环境系数误差降低一个数量级，且在超出训练域时具有更好的外推性能。

**⚠️ 局限性**

局限性包括：①需要图表的良好条件化（Jacobian 下界）才能保证 ρ‑度量的 Lipschitz 性；②当前仅在单图表设置下理论与实验，扩展到多图表仍需解决过渡映射与分区合成问题；③对第二阶 Hessian 约束无显式正则化，导致潜在漂移近临界点的偏差；④实验使用的是理想的 oracle 漂移/协方差，真实场景中需估计并评估其对正则化的影响；⑤在极高维嵌入下计算成本仍然较大。

---

## 472. Using Large Language Models and Knowledge Graphs to Improve the Interpretability of Machine Learning Models in Manufacturing

**arXiv ID:** 2604.16280 | [PDF](https://arxiv.org/pdf/2604.16280v1)

**作者:** Thomas Bayer `[一作]` (University of Applied Sciences Ravensburg-Weingarten), Wolfram Höpken `[通讯]` (University of Applied Sciences Ravensburg-Weingarten)

**通讯引用:** 2398 | [OpenAlex ID](https://openalex.org/A5069822917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在制造业场景下，利用知识图谱与大型语言模型（Graph‑RAG）生成用户友好的解释，帮助工人和开发者理解机器学习模型的预测结果。

**💡 创新点**

创新点在于：① 将制造领域的专业知识与 ML 结果统一建模为知识图谱；② 在推理时通过多轮对话动态检索相关三元组并作为上下文注入 LLM；③ 采用分层查询和 Branch‑and‑Bound 递归遍历策略，避免传统 RAG 的过度检索与错误推断。

**🔧 技术方法**

核心技术包括：知识图谱（基于 MLSchema 扩展）、Graph‑RAG 交互框架、OpenAI GPT‑系列（用于生成解释）、prompt‑tuning 与多轮对话调度。

**📊 数据集**

使用的主要数据集是制造工厂的机器人螺丝放置任务数据（包含数据集、模型、任务及其关联信息），并将其映射到构建的知识图谱中；评估时采用 XAI Question Bank 的 33 题。

**📈 对比分析**

方法通过用户研究（20 名参与者，工人/开发者两角色）与结构化鲁棒性测试进行评估。结果显示：帮助性、可理解性、结构性评分均在 4–5 级之间，Kendall τ 相关系数表明两角色评分一致性良好；鲁棒性评估在大多数类别中表现稳定，但在范围限制、能力意识与提示注入等方面仍有缺陷。

**⚠️ 局限性**

局限性主要包括：① 需要手工构建并维护知识图谱，工作量大；② 对不在图谱范围内的查询缺乏明确拒绝机制，导致范围溢出；③ 在多轮交互中容易出现信息冗余或长度失控；④ 对抗性提示（prompt injection）尚未得到充分防护；⑤ 依赖于外部 LLM 的服务成本与可扩展性。

---

## 473. Learning to Reason with Insight for Informal Theorem Proving

**arXiv ID:** 2604.16278 | [PDF](https://arxiv.org/pdf/2604.16278v1)

**作者:** Yunhe Li `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**通讯引用:** 2193 | [OpenAlex ID](https://openalex.org/A5035185924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究非正式定理证明，发现模型在识别问题核心技术（即洞察力）方面表现不足，提出构建层级数据集DeepInsightTheorem并设计渐进式多阶段SFT框架，让LLM先学习基本证明，再逐步学会识别核心技术并以此指导证明生成。

**💡 创新点**

创新点在于：①明确核心技术与洞察力在非正式证明中的关键作用；②通过LLM辅助提取核心技术，将证明拆解为问题–核心技术–草图–完整证明的层级结构；③提出从基础到技巧指导的三阶段（Apprentice→Journeyman→Expert）训练课程，显著提升模型的洞察力和推理能力。

**🔧 技术方法**

采用的技术包括：①大语言模型（Qwen2.5、Llama3）自回归架构；②监督微调（SFT）和渐进式多阶段训练；③LLM‑as‑Judge评估，使用DeepSeek‑R1与o3‑mini给出逻辑有效性、完整性、清晰度三维评分；④提示工程提取核心技术与草图；⑤在DeepTheorem基础上扩展层级数据集DeepInsightTheorem。

**📊 数据集**

使用数据集：①基础DeepTheorem（约121K个IMO级问题–证明对）；②在其上构建的DeepInsightTheorem（约100K个问题，含核心技术与草图）；评测集为FIMO、PutnamBench、HMMT三大数学竞赛数据集。

**📈 对比分析**

对比方法：传统SFT（仅问题–证明）、两阶段训练（问题–草图–证明）、以及本研究的三阶段训练；实验结果显示三阶段训练在所有基准上均显著优于基线，尤其是小模型（1.5B）提升显著；在与RL‑finetuned的Deepseek‑R1及o3‑mini的对比中，本文方法在不使用RL后训练的情况下已能接近甚至超越部分SOTA模型。

**⚠️ 局限性**

限制：①核心技术识别仍依赖LLM辅助的人工或半自动标注，规模和质量受限；②训练成本高，尤其是多阶段训练需要多轮Fine‑Tuning；③仅聚焦非正式证明，缺乏与正式证明引擎或符号计算的结合；④评估仍基于LLM judge，可能存在主观偏差；⑤模型在极端复杂或非常规证明结构上的泛化仍待验证。

---

## 474. FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation

**arXiv ID:** 2604.16298 | [PDF](https://arxiv.org/pdf/2604.16298v1)

**作者:** Dian Shao `[一作]` (Northwestern Polytechnical University), Jing Huo `[通讯]` (Nanjing University)

**通讯引用:** 3851 | [OpenAlex ID](https://openalex.org/A5066091819)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为 FineCog-Nav 的零射击无人机视听语言导航框架，并构建了针对细粒度评估的 AerialVLN-Fine 数据集。

**💡 创新点**

创新点在于将认知功能细化为语言处理、感知、注意、记忆、想象、推理和决策七个模块，并通过结构化提示和多模态协同实现了可解释且高效的导航。

**🔧 技术方法**

技术实现包括使用中等规模基础模型与专门设计的提示、子目标提取器、想象模块、层级记忆管理、以及可解释的决策模块，以实现跨模态的模块协作。

**📊 数据集**

实验使用了新构建的 AerialVLN-Fine（300条轨迹，句级对齐）以及原始的 AerialVLN-S-Val 数据集进行评估。

**📈 对比分析**

与单一基础模型以及 NavGPT、DiscussNav 等框架对比后，FineCog-Nav 在多种 LLM 规模下始终保持更高的成功率、更低的导航误差和更优的轨迹相似度，展示了显著的性能提升。

**⚠️ 局限性**

局限性包括整体成功率仍偏低、对中等规模模型的依赖、在零射击设置下对极端环境的鲁棒性不足，以及对真实世界噪声和控制误差的适应性仍待改进。

---

## 475. Repurposing 3D Generative Model for Autoregressive Layout Generation

**arXiv ID:** 2604.16299 | [PDF](https://arxiv.org/pdf/2604.16299v1)

**作者:** Haoran Feng `[一作]` (Beihang University), Lu Sheng `[通讯]` (Beihang University)

**通讯引用:** 5553 | [OpenAlex ID](https://openalex.org/A5035443556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 LaviGen，一种在原生 3D 空间中自回归生成布局的框架；

**💡 创新点**

将 3D 生成模型重新用于自回归布局生成，结合身份感知位置编码和双向引导自回归蒸馏，显著提升物理可行性与语义一致性；

**🔧 技术方法**

基于 3D 稀疏体素扩散模型（类似 TRELLIS）、MMDiT Transformer、RoPE 位置编码、双向与因果引导蒸馏、Self‑Forcing 自回归训练；

**📊 数据集**

使用 TRELLIS 训练集（约 50 万 3D 资产）以及 3D‑Front 与 InternScenes（约 15K 布局场景）做二阶段/三阶段训练；

**📈 对比分析**

在 LayoutVLM 基准上对比 LayoutGPT、Holodeck、I‑Design、LayoutVLM，LaviGen 在物理可行性（CF/IB）上提升 19% 以上，计算速度提升 65%，同时在语义一致性上保持竞争力；

**⚠️ 局限性**

受限于 64³ 体素分辨率导致小物体细节不足，且语义一致性受标注稀缺影响，未来需更高分辨率表示和更完善的标注流程。

---

## 476. ASMR-Bench: Auditing for Sabotage in ML Research

**arXiv ID:** 2604.16286 | [PDF](https://arxiv.org/pdf/2604.16286v1)

**作者:** Eric Gan `[一作]` (Redwood Research), Vivek Hebbar `[通讯]` (Redwood Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了ASMR-Bench基准，用于评估审计者（LLM和人类）检测机器学习研究代码库中潜在破坏（sabotage）的能力。

**💡 创新点**

创新点在于①提出真实世界代码库的破坏版本并公开基准；②系统评估LLM与人类审计者的性能；③展示LLM可自动生成破坏且能逃避同级审计者的可能性。

**🔧 技术方法**

使用前沿大型语言模型（Claude Opus 4.6、GPT‑5.2、Gemini 3.1 Pro）进行自动审计与破坏生成；人类审计者结合LLM协助；通过代码运行、实验重现与修复建议的自动化评估方法。

**📊 数据集**

9个现有ML研究代码库（共11个破坏变体），每个代码库都有正面与破坏版本，并配有对应论文。

**📈 对比分析**

比较方法：对审计者输出的可信度与修复建议进行AUROC与top‑k fix率评估；结果显示Gemini 3.1 Pro在完整实验权限下达到0.77 AUROC，top‑1修复率42%；人类审计者表现相近但仍低于理想。

**⚠️ 局限性**

限制：样本量小（9个代码库）；破坏由单一人类攻击者完成，难以评估多样性；评估未覆盖绿色新建实验场景；审计者可能受已知论文记忆影响。

---

## 477. Enhancing Hazy Wildlife Imagery: AnimalHaze3k and IncepDehazeGan

**arXiv ID:** 2604.16284 | [PDF](https://arxiv.org/pdf/2604.16284v1)

**作者:** Shivarth Rai `[一作]` (Manipal Institute of Technology), Tejeswar Pokuri `[通讯]` (Manipal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了AnimalHaze3k合成雾化野生动物图像数据集，并提出了IncepDehazeGan网络，用于单图像去雾并提升野生动物检测效果。

**💡 创新点**

创新点在于：①利用物理散射模型和HybridDepth深度估计对真实野生动物图像生成多样化雾化图像；②将Inception模块与残差跳连结合的GAN架构，并采用混合对抗+L1损失，显著提升去雾质量。

**🔧 技术方法**

技术包括：物理雾化模型（I=J·t+A(1-t)）、HybridDepth深度估计、Inception块与残差跳连的生成器、六层卷积鉴别器、对抗+L1双重损失、YOLOv11下游检测评估。

**📊 数据集**

使用的数据集为东北虎豹国家公园（NTLNP）日间野生动物图像集（1159张），通过物理雾化生成3,477张合成雾图，构成AnimalHaze3k；同一数据集用于检测实验。

**📈 对比分析**

与FD-GAN、FFA-Net、DehazeFormer、DEA-Net等SOTA模型在SSIM、PSNR、FSIM、LPIPS上进行对比，IncepDehazeGan分别达SSIM 0.8914、PSNR 20.54、FSIM 0.9363、LPIPS 0.1104，超越所有基线；在YOLOv11检测上，去雾后mAP提升112%，mIoU提升67%。

**⚠️ 局限性**

局限性在于：①数据为合成雾化，可能与真实环境的雾化特性存在差距；②仅包含日间图像，未覆盖夜间或极端天气场景；③深度估计误差可能影响雾化质量；④在不同物种/栖息地上的泛化性尚未充分验证。

---

## 478. Evaluating the Progression of Large Language Model Capabilities for Small-Molecule Drug Design

**arXiv ID:** 2604.16279 | [PDF](https://arxiv.org/pdf/2604.16279v1)

**作者:** Shriram Chennakesavalu `[一作]` (Prescient Design, Genentech), Kangway Chuang `[通讯]` (Prescient Design, Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，作者提出了一套以小分子药物设计为目标的化学任务集合，并将这些任务统一建模为强化学习（RL）环境，随后对几大前沿大型语言模型（GPT‑5、Claude Opus 4、Qwen‑30B‑A3B）进行基线评估，并通过 RL 后训练（GRPO/DAPO）显著提升了一款 30 B 参数开源模型（Aspen），使其在多项单回合与多回合化学任务上与闭源前沿模型相当。

**💡 创新点**

创新点在于：①将化学相关任务抽象为 RL 环境，提供统一评测与后训练框架；②使用 RL 后训练而非单纯监督微调来细化模型在化学知识上的性能；③通过对比不同模型族内部迭代与跨族基准，揭示“颠簸前沿”现象，验证后训练能弥补模型规模差距。

**🔧 技术方法**

技术主要包括：强化学习后训练（Group Relative Policy Optimization 的 DAPO 变体）、分子属性预测与转换的奖励设计、分子生成的约束满足奖励、单回合与多回合（模拟 lead‑optimization）环境交互、以及对模型输出进行评估的多指标评分体系。

**📊 数据集**

数据集涵盖：ZINC（约10万分子）用于属性预测、表示转换与多属性生成；内部专有活性/ADMET 数据与 FS‑Mol 数据集用于实验性属性预测；以及用于多回合 docking 评测的 8TTR（碳酸酐酶）结构。

**📈 对比分析**

对比方法：将开源模型与同族前一代模型以及两大闭源前沿模型进行统一评测；在单回合任务中，后训练后 Aspen 在 RDKit 属性预测、实验属性预测、表示转换、生成等方面显著提升（如 TPSA 由0.24提升至0.88，约束满足率由0.09提升至0.21）；在多回合 lead‑optimization 环境中，Aspen 的平均最佳对接分数比基线 Qwen 差距显著缩小，并在某些指标上超过 GPT‑5/GPT‑5.2 与 Claude Opus 4.6。

**⚠️ 局限性**

局限性包括：①RL 后训练仅在基模型已有部分化学知识时有效，无法弥补知识缺口；②对实验性低样本任务仍表现不佳，提示需要更充分的基础训练（mid‑training / SFT）；③生成任务在多约束满足上仍存在“约束组合”瓶颈；④闭源前沿模型在多回合优化效率上仍占优，说明开放模型在整体性能上仍有提升空间。

---

