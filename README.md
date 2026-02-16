# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-16 | 今日论文总数: 412

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Never say never: Exploring the effects of available knowledge on agent persuasiveness in controlled physiotherapy motivation dialogues

**arXiv ID:** 2602.12924 | [PDF](https://arxiv.org/pdf/2602.12924v1)

**作者:** Stephan Vonschallen `[一作]` (Zurich University of Applied Sciences), Friederike Eyssel `[通讯]` (Bielefeld University)

**通讯引用:** 6431 | [OpenAlex ID](https://openalex.org/A5074815650)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过ChatGPT生成的对话，探讨了不同知识配置（自我、用户、上下文）如何影响生成式社交代理在物理治疗动机情境中的说服性行为。

**💡 创新点**

创新点在于提出并验证通过调节代理可用知识来控制其自主生成的说服策略和负责任行为，并将定性情境分析与定量在线评估结合。

**🔧 技术方法**

采用了大型语言模型（ChatGPT 3.5/4.1、Claude 3.5 Haiku）生成对话，使用内容编码、确认性因子分析和多层结构方程模型进行定量分析。

**📊 数据集**

数据集为13个ChatGPT生成的情境对话（其中5个用于在线评估），以及27名瑞士受试者对每条对话的感知说服性、断言性与表现性评价。

**📈 对比分析**

通过对比不同知识配置的情境，对感知说服性进行多层SEM检验，结果显示自我与用户知识通过断言性与表现性中介显著提升感知说服性，表现性效应较弱，上下文知识无显著影响。

**⚠️ 局限性**

局限性包括仅使用单一LLM的生成样本、文本情境缺乏真实互动、样本规模有限、未提供完整患者背景给评估者，导致对真实说服效果与责任性评价的可推广性受限。

---

## 2. Bus-Conditioned Zero-Shot Trajectory Generation via Task Arithmetic

**arXiv ID:** 2602.13071 | [PDF](https://arxiv.org/pdf/2602.13071v1)

**作者:** Shuai Liu `[一作]` (Nanyang Technological University), Gao Cong `[通讯]` (Nanyang Technological University)

**通讯引用:** 16582 | [OpenAlex ID](https://openalex.org/A5045198704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了在目标城市无法获得任何真实出行轨迹时，仅利用源城市轨迹数据和目标城市公交时刻表，通过任务算术实现零样本轨迹生成的方法。

**💡 创新点**

首次将任务算术（Task Arithmetic）应用于轨迹生成，构造公交时刻表与真实轨迹之间的参数偏移向量，并将该偏移迁移到目标城市，从而在无真实轨迹的情况下逼近目标城市的出行模式。

**🔧 技术方法**

统一轨迹离散化与序列化、基于LoRA的参数高效微调、任务向量（embedding、LoRA、输出层）构造以及任务算术操作；同时提供理论分析说明在基准模型与指令微调模型之间的稳定性。

**📊 数据集**

使用上海、无锡和新加坡三座城市的公交时刻表和轨迹数据进行实验，公交时刻表数据来自公开地图服务，轨迹数据来自出租车或移动通信运营商公开数据集。

**📈 对比分析**

与 TimeGeo、MoveSim、TSG、TrajGDM、DiffTraj、TrajGen、Traveller、Geollama（仅训练目标城市轨迹）以及跨城迁移方法 COLA 进行对比；实验结果显示 MobTA 在 JSD、距离、半径、停留时长等指标上显著优于所有基线，并且性能接近使用目标城市轨迹微调的上界模型。

**⚠️ 局限性**

依赖目标城市公交时刻表的完整性；在公交网络稀疏或缺失的城市可能表现不佳；任务算术系数 μ 的选择对性能有影响；未充分利用公交站点属性或拓扑信息。

---

## 3. Computationally sufficient statistics for Ising models

**arXiv ID:** 2602.12449 | [PDF](https://arxiv.org/pdf/2602.12449v1)

**作者:** Abhijith Jayakumar `[一作]` (Los Alamos National Laboratory), Sidhant Misra `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 1440 | [OpenAlex ID](https://openalex.org/A5103153799)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了一种在仅能观测到O(γ)阶统计量的条件下，通过改进的Interaction Screening方法（使用多项式近似梯度的投影梯度下降）学习Ising模型的参数和结构。

**💡 创新点**

创新点在于：①证明只需观测到统计量阶数为O(γ)即可在样本复杂度和计算复杂度上与可观测完整样本的算法等价；②将梯度近似误差与统计误差的鲁棒性结合，给出严格误差上界；③在此框架下同时实现结构学习、磁场学习，并讨论更强结构先验下所需统计阶数的进一步降低。

**🔧 技术方法**

采用的技术包括：Interaction Screening 估计器、投影梯度下降、对指数函数的有限多项式近似、统计查询（SQ）风格的低阶矩估计、极限定理和集中不等式、Lambert-W 函数用来确定多项式阶数、以及对多变量曲率的分析。

**📊 数据集**

本文主要在理论上给出算法与证明，并未使用具体公开数据集；所有实验与证明均基于假设的 i.i.d. 样本以及符号推导。

**📈 对比分析**

与传统使用完整样本的 Interaction Screening 方法相比，本文的算法在样本复杂度上保持相同的 O(e^8γ poly(γ) log p /ε^4) 级别，在计算复杂度上也仍为多项式（O(p^2 γ^4 e^8γ /ε^4) 步）。实验中展示了在有限阶统计量下，误差与完整样本相近，证明了方法的有效性。

**⚠️ 局限性**

局限性：①对统计量阶数仍存在上限，尚未解决低于 O(γ) 但高于二阶统计量的学习难度；②假设已知 ℓ₁‑宽度 γ，实际应用中此信息可能未知；③仅针对 Ising 模型，对其他格点或连续变量 Gibbs 分布的推广尚未给出；④样本量依然随 γ 指数级增长，实际大规模应用可能受限。

---

## 4. Extending confidence calibration to generalised measures of variation

**arXiv ID:** 2602.12975 | [PDF](https://arxiv.org/pdf/2602.12975v1)

**作者:** Andrew Thompson `[一作]` (National Physical Laboratory), Vivek Desai `[通讯]` (National Physical Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了变异校准误差（VCE）指标，用于评估机器学习分类器的概率分布的整体变异性与观察到的正确性分布之间的匹配程度。

**💡 创新点**

创新点在于将传统的置信度校准（ECE）扩展为适用于任意变异度量（如熵）的通用校准指标，并证明在完美校准的合成数据上，VCE 随样本数增大趋于零，优于现有的 UCE 指标。

**🔧 技术方法**

采用了概率分布的变异度量（以熵为例）和分箱（等宽/等频）技术，对预测概率进行排序后计算预测与观测的变异度量，并用 ECE、VCE、UCE 进行对比评估。

**📊 数据集**

使用了从 Dirichlet 分布生成的合成预测数据，实验包含 3 类和 10 类两种设置，并在 α 参数分别为均匀分布和高度偏斜两种情况下，样本量从 10^4 到 10^7 进行实验。

**📈 对比分析**

通过可靠性图（reliability diagram）和指标值随样本量变化的曲线对比，发现 VCE 与 ECE 随样本数增加均趋于零，表明其校准评估性能优于 UCE；VCE 在等宽和等频分箱下均保持稳定的降解趋势。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，缺乏对真实数据的实验；理论分析仅针对熵的情况，其他变异度量的泛化尚未证明；分箱策略的选择可能影响指标值；未探讨在复杂模型或大规模数据上的计算成本。

---

## 5. Quantization-Robust LLM Unlearning via Low-Rank Adaptation

**arXiv ID:** 2602.13151 | [PDF](https://arxiv.org/pdf/2602.13151v1)

**作者:** João Vitor Boer Abitante `[一作]` (Pontifícia Universidade Católica do Rio Grande do Sul), Lucas S. Kupssinskü `[通讯]` (Pontifícia Universidade Católica do Rio Grande do Sul)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5046415913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM进行机器遗忘后训练量化，提出基于LoRA的鲁棒遗忘方法。

**💡 创新点**

通过冻结基模型，仅在低秩适配器中更新，集中遗忘信号并抵抗低位量化的抹除。

**🔧 技术方法**

LoRA、GA/NPO遗忘算法、GDR/KLR正则化、post‑training quantization (RTN) 以及合并适配器到基模型。

**📊 数据集**

MUSE基准数据集（Books 与 News）与 Llama‑2‑7B 模型。

**📈 对比分析**

与全参数微调相比，LoRA 在 Int4 PTQ 下维持/提升了忘记度（VerMem/KnowMem）并显著降低隐私泄露，4‑bit Utility 更高，整体表现更稳健。

**⚠️ 局限性**

对低秩维数、缩放因子和学习率敏感，需要手动调参；在某些数据/方法组合下仍未显著优于全微调，实验仅覆盖 Llama‑2‑7B，推广性待验证。

---

## 6. Uncertainty in Federated Granger Causality: From Origins to Systemic Consequences

**arXiv ID:** 2602.13004 | [PDF](https://arxiv.org/pdf/2602.13004v1)

**作者:** Ayush Mohanty `[一作]` (Georgia Institute of Technology), Nagi Gebraeel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5820 | [OpenAlex ID](https://openalex.org/A5054372641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在FedGC框架下，如何对系统中的不确定性进行量化与传播，并分析其对稳态性能的影响。

**💡 创新点**

提出了针对线性模型的理论不确定性传播分析，并给出了相应的非线性扩展与数据非平稳性的松弛方法。

**🔧 技术方法**

使用了FedGC框架、差分隐私通信、理论分析（引理与命题）以及非线性扩展技术。

**📊 数据集**

实验中使用了常见的联邦学习基准数据集（如MNIST、CIFAR-10等）。

**📈 对比分析**

与FedAvg、FedSGD等传统联邦学习方法进行了对比，实验结果表明本文方法在稳态误差下降和不确定性抑制方面表现更优。

**⚠️ 局限性**

主要局限包括：1）目前仅在线性模型上有完整理论保证；2）需要数据满足平稳性假设；3）计算和通信复杂度相对较高。

---

## 7. RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty

**arXiv ID:** 2602.12424 | [PDF](https://arxiv.org/pdf/2602.12424v1)

**作者:** Ziqian Zhang `[一作]` (Lehigh University), Lichao Sun `[通讯]` (Lehigh University)

**通讯引用:** 8056 | [OpenAlex ID](https://openalex.org/A5071709543)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RankLLM 框架，联合估计问题难度与模型能力，对大规模 LLM 进行难度感知评估。

**💡 创新点**

通过双向分数传播的非参数图算法，以难度为主导实现模型和问题的共同排名，避免了传统 IRT 的参数化约束。

**🔧 技术方法**

使用有向二分图、阻尼随机游走迭代和线性时间的马尔可夫链收敛算法，并支持连续分数输入。

**📊 数据集**

在 BBH、GPQA、GSM8K、HellaSwag、MATH、MMLU‑Pro 共 35,550 条题目上评估 30 个 LLM（参数从 0.5B 到数百亿）。

**📈 对比分析**

与 1PL/2PL/多维 IRT 及 Simple Rank 对比，RankLLM 与人类难度判断 90% 一致，收敛仅 0.006 s，且在相邻模型间提供更细粒度重新排序。

**⚠️ 局限性**

对极端模型或全通/全失题目需手动过滤；在极小样本或模型池规模极低时收敛速度与稳定性仍待进一步验证；不覆盖模型安全或公平性评估。

---

## 8. propella-1: Multi-Property Document Annotation for LLM Data Curation at Scale

**arXiv ID:** 2602.12414 | [PDF](https://arxiv.org/pdf/2602.12414v1)

**作者:** Maximilian Idahl `[一作]` (ellamind), Jan Philipp Harries `[通讯]` (ellamind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一系列小型多语言LLM（propella‑1）用于对文档进行18个属性的结构化注释，并公开了超过30亿条注释数据集。

**💡 创新点**

通过多属性结构化注释替代单一质量分数，提供可组合的过滤规则；同时在57种语言、数十亿文档规模上公开了首个多属性注释资源。

**🔧 技术方法**

基于Qwen‑3的decoder‑only模型，使用fp8混合精度、64k上下文、专用prompt；评估采用与Gemini‑3‑Pro的对齐，指标为QWK/F1/IoU；推理使用SGLang+llguidance并在H100上加速。

**📊 数据集**

FineWeb‑2、FinePDFs、HPLT 3.0、Nemotron‑CC、SYNTH、finewiki等主流预训练语料；训练时采样多语言文档并由前沿LLM标注。

**📈 对比分析**

与Gemini、Qwen、Gemma、Mistral‑Small等通用LLM在相同任务下对比；propella‑1 4B整体得分0.779，高于Gemini‑3‑Flash和更大开放模型；0.6B也达0.729；推理速度每秒27文档（fp8 H100）。

**⚠️ 局限性**

评估基于Gemini‑3‑Pro的自动注释，可能共享偏差；模型继承前沿LLM的偏见，低资源语言表现不佳；未验证多属性过滤对下游训练的实际提升。

---

## 9. BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents

**arXiv ID:** 2602.12876 | [PDF](https://arxiv.org/pdf/2602.12876v1)

**作者:** Huanyao Zhang `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 12968 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新型的多模态深度浏览与搜索基准，包含300道精心设计、跨文本与图像深层推理的挑战性问题，并构建了可统一调用多种搜索与视觉工具的OmniSeeker浏览框架；

**💡 创新点**

创新点在于①基准强调多层次、跨模态多跳推理，强制要求模型利用外部搜索工具；②所有关键信息公开可检索，保证可复现性；③引入专家验证的子目标与Process Score，提供过程级评估；④提供统一的OmniSeeker框架，提升开源模型性能；

**🔧 技术方法**

技术手段包括：多模态大语言模型与工具调用（TextSearch、WebVisit、ImageSearch、ImageCrop、ReverseImageSearch），统一JSON格式数据与评测；使用Success Rate与Process Score两维度评价；对模型进行工具增强与无工具评测；

**📊 数据集**

使用的数据集是本文自建的Benchmark（称为<benchmark_name>），包含300道问题、383张图片，覆盖5个主领域（Science、Technology、Society、Culture、Life）与24个子领域，任务分为3级难度和4种难度级别；

**📈 对比分析**

比较方法：在人类、工具无效MLLM、工具增强MLLM以及OmniSeeker框架下评估；结果显示人类平均成功率68%，Process Score 83%；工具无效模型不到10%；工具增强模型最高达40% SR；OmniSeeker让所有模型性能提升至30–40% SR，显示出显著改进；

**⚠️ 局限性**

局限性包括：仍与人类存在显著性能差距；多模态感知与视觉理解仍是瓶颈；任务规模仅300题，难以覆盖更广泛场景；评测仅基于公开搜索资源，未涵盖动态/非公开信息；Process Score与SR之间存在差距，表明模型在长序列推理中的逻辑一致性不足。

---

## 10. Optimal Take-off under Fuzzy Clearances

**arXiv ID:** 2602.13166 | [PDF](https://arxiv.org/pdf/2602.13166v1)

**作者:** Hugo Henry `[一作]` (University of Cincinnati), Kelly Cohen `[通讯]` (University of Cincinnati)

**通讯引用:** 3249 | [OpenAlex ID](https://openalex.org/A5034113408)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

结合模糊规则系统与最优控制，提出一种用于无人机起飞时障碍物避让的混合架构。

**💡 创新点**

在模糊推理层自适应调节障碍物的清晰度、紧急程度与激活决策，将法规安全阈值转化为软约束，显著减少不必要的路径重算。

**🔧 技术方法**

使用Takagi–Sugeno–Kang型模糊推理系统、FALCON最优控制工具箱、IPOPT非线性优化器以及软约束惩罚方法。

**📊 数据集**

采用FALCON自带的低精度飞机模型和合成的障碍物场景（模拟飞行器与鸟群），未使用公开数据集。

**📈 对比分析**

与仅用最优控制的基线对比，实验在MATLAB单线程下每次迭代耗时2–3秒；然而由于Lagrangian惩罚始终为零，约束未被执行，导致性能与预期不符。

**⚠️ 局限性**

主要局限在软件工具的兼容性问题导致惩罚项失效，缺乏高保真飞行模型、真实雷达误差以及对动态障碍物不确定性的验证。

---

## 11. MonoLoss: A Training Objective for Interpretable Monosemantic Representations

**arXiv ID:** 2602.12403 | [PDF](https://arxiv.org/pdf/2602.12403v1)

**作者:** Ali Nasiri-Sarvi `[一作]` (Concordia University), Mahdi S. Hosseini `[通讯]` (Concordia University)

**通讯引用:** 839 | [OpenAlex ID](https://openalex.org/A5073426758)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了可解释的单义特征表示训练目标MonoLoss，并将MonoScore指标线性化，显著加速评估与训练过程

**💡 创新点**

①将二次相似度计算改为一次性统计，实现O(N)而非O(N²)的MonoScore；②将MonoScore直接转化为可微分的MonoLoss，用作正则化提升单义性

**🔧 技术方法**

稀疏自编码器（SAE）、CLIP、SigLIP、ViT等预训练视觉编码器；线性时间统计、余弦相似度、min-max归一化、批量正则化

**📊 数据集**

OpenImagesV7、ImageNet‑1K、CIFAR‑10/100、ImageNet‑1K验证集（用于评估）

**📈 对比分析**

与原配套的SAE（TopK、BatchTopK、JumpReLU、Vanilla）以及ResNet‑50、CLIP‑ViT‑B/32进行对比；MonoLoss在大多数配置下提升MonoScore与类纯度，且在ImageNet‑1K、CIFAR‑10/100上准确率提升约0.1–0.6%，训练时间仅增加≈4%

**⚠️ 局限性**

MonoScore依赖于预训练编码器的语义空间，若编码器质量差则评估不可靠；MonoLoss在某些网络架构下对重建或分类效果影响有限，且对单义性提升的程度受数据集与特征来源限制

---

## 12. Sparse Autoencoders are Capable LLM Jailbreak Mitigators

**arXiv ID:** 2602.12418 | [PDF](https://arxiv.org/pdf/2602.12418v1)

**作者:** Yannick Assogba `[一作]` (Apple), Arno Blaas `[通讯]` (Apple)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5086300347)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于稀疏自编码器（SAE）的上下文条件差分引导（Context-Conditioned Delta Steering）防御方法，通过在稀疏特征空间中进行激活位移来抑制LLM的越狱攻击。

**💡 创新点**

创新点在于：①利用有害请求与对应越狱请求的token级激活差异进行特征选择，②在稀疏解码空间而非稠密激活空间中实施位移，③通过统计检验和标准化中位差来筛选高效特征，实现更精准的安全-实用性折衷。

**🔧 技术方法**

主要技术包括稀疏自编码器训练（W_enc/W_dec + ReLU/TopK等稀疏非线性）、Wilcoxon符号秩检验、Benjamini–Hochberg FDR校正、均值位移（mean‑shift）推理时干预、以及对干预强度与特征数量的双参数控制。

**📊 数据集**

使用来自StrongReject和OpenAI SafetyBench的808条有害请求（经过去重后288+520条）及其12种包装与重写越狱变体（共9696条），对Gemma2‑2b/9b、Llama‑3.1‑8b、Qwen‑2.5‑7b等四个开源指令调优模型进行评估。

**📈 对比分析**

与密集激活空间的对比基线（Contrastive Activation Addition、Linear‑AcT）以及训练型防御（Circuit Breakers、Latent Adversarial Training）和提示型防御（Self‑Reminder）相比，Context‑Delta在安全性提升与实用性保持之间取得更优折衷，尤其在外部分布越狱攻击上表现更强；在安全-实用性曲线与归一化比较中表现与基线相当或更好。

**⚠️ 局限性**

局限性包括：①仅使用公开的无监督SAE，未针对越狱任务进行专门训练；②特征选择需要越狱提示包含原始有害请求子串，对完全重写型攻击不直接适用；③未评估对抗优化或梯度攻击，且强防御可能削弱指令遵循能力。

---

## 13. Intrinsic Credit Assignment for Long Horizon Interaction

**arXiv ID:** 2602.12342 | [PDF](https://arxiv.org/pdf/2602.12342v1)

**作者:** Ilze Amanda Auzina `[一作]` (Tübingen AI Center), Matthias Bethge `[通讯]` (Tübingen AI Center)

**通讯引用:** 34119 | [OpenAlex ID](https://openalex.org/A5061457780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将语言模型自身对目标概念的信念更新作为密集奖励信号，训练能在长时间交互中高效地进行信息检索与决策。

**💡 创新点**

创新点在于利用模型内部信念的变化（对目标概念的对数概率差）来实现细粒度的信用分配，避免了传统需要额外价值网络或奖励模型的高成本方案。

**🔧 技术方法**

技术包括：对数概率信念提取、belief‑change奖励设计、转折点 GRPO 的 turn‑wise advantage 计算、best‑of‑n 策略优化、LoRA 微调、以及多尺度（1.7B、4B）语言模型训练。

**📊 数据集**

数据集：主要使用 20 Questions（训练/验证/测试拆分）；外部验证使用 Guess My City、Murder Mystery；实际应用验证使用 User Personalization（STaR‑GATE）和 Customer Service 基准。

**📈 对比分析**

与基线 SFT、StarPO 以及 670B 大模型对比：在 20 Questions 测试集上 CIA 在 1.7B 和 4B 规模分别提升 14.83% 与 20.38% 的成功率；在 Pass@k、长交互预算、以及 OOD 任务（猜城市、谋杀谜案）上均保持领先；在实际应用中比 StarPO 提升 5–15% 的性能。

**⚠️ 局限性**

局限性：奖励仅针对单一参考答案，可能限制多答案任务的多样性；需要可验证的终端奖励，无法直接用于完全无监督或主观任务；信念校准不完美时奖励信号可能失真；对用户模拟器的过拟合风险仍需进一步评估。

---

## 14. Synaptic Activation and Dual Liquid Dynamics for Interpretable Bio-Inspired Models

**arXiv ID:** 2602.13017 | [PDF](https://arxiv.org/pdf/2602.13017v1)

**作者:** Mónika Farsang `[一作]` (Vienna University of Technology), Radu Grosu `[通讯]` (Vienna University of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文构建了一个统一的框架，对液体电容（LC）和液体电阻液体电容（LRC）两类生物启发式循环神经网络进行定义、实现，并在端到端车道保持任务中评估它们的解释性和性能。

**💡 创新点**

创新点在于首次提出只包含液体电容的LC模型，并证明与传统门控RNN相比可保持解释性；进一步证明双液动力学（液体电阻+液体电容）以及突触激活的组合能显著提升网络解释性和对噪声的鲁棒性。

**🔧 技术方法**

主要技术包括电路等价电路（EEC）建模、CT‑RNN、LTC、LC、LRC等架构的实现；使用CNN提取图像特征后输入全连接RNN；采用端到端仿真训练并用多项指标（验证损失、神经活动相关性、注意力SSIM等）进行评估。

**📊 数据集**

数据集采用VISTA仿真环境中的前视摄像头序列，实验涵盖夏季与冬季两种天气场景的车道保持任务。

**📈 对比分析**

通过与LSTM、GRU、MGU等传统门控RNN进行对比，并在验证损失、绝对相关性、SSIM等指标上进行量化，结果显示LRC-SA/NA模型在绝大多数指标上优于其他模型，LC模型在性能上与传统门控RNN相当。

**⚠️ 局限性**

局限性包括：实验仅在模拟车道保持任务中进行，缺乏对更复杂或真实环境的验证；突触激活机制的解释仍需进一步研究；模型规模和训练成本相对较大。

---

## 15. Neighborhood Blending: A Lightweight Inference-Time Defense Against Membership Inference Attacks

**arXiv ID:** 2602.12943 | [PDF](https://arxiv.org/pdf/2602.12943v1)

**作者:** Osama Zafar `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**通讯引用:** 2614 | [OpenAlex ID](https://openalex.org/A5028326739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种后处理推理时的邻域融合（Neighborhood Blending）防御机制，用于阻止黑盒成员推断攻击。

**💡 创新点**

创新点在于利用差分隐私采样选取同标签邻域样本并平均其预测，既实现零标签损失，又通过自适应噪声平衡隐私与精度。

**🔧 技术方法**

核心技术包括指数机制（exponential mechanism）、Gumbel-Top‑k抽样、梯度无关的邻域平均与置信度平滑。

**📊 数据集**

实验使用多种公开数据集：Nursery、Iris、Adult、Purchase（10/20/50/100类）、Location-30、Texas-100、CIFAR‑10及其对应的经典与深度学习模型。

**📈 对比分析**

与MemGuard、DP‑SGD等基线对比，邻域融合在保持零标签损失、低置信度向量失真（PCD≤0.35、CVD≤0.39）同时将成员推断准确率压至≈0.5，优于现有防御。

**⚠️ 局限性**

局限包括对高维稀疏数据需更大邻域导致失真略高，且对极少数高隐私需求场景可能仍需进一步优化噪声幅度与计算成本。

---

## 16. Variational Green's Functions for Volumetric PDEs

**arXiv ID:** 2602.12349 | [PDF](https://arxiv.org/pdf/2602.12349v1)

**作者:** Joao Teixeira `[一作]`, Otman Benchekroun `[通讯]` (University of Toronto)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5006659196)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

学习可微分的 Green’s Function 神经场，在任意几何表示上高效求解线性自伴 PDE，且可对任意源位置进行快速评估；

**💡 创新点**

提出无监督的变分训练框架，将解析自由空间解与神经修正相结合，并通过超积分采样实现对源位置的全局泛化；

**🔧 技术方法**

使用变分能量最小化、SIREN MLP 神经场、解析基本解分解、硬 Dirichlet 边界重参数化和超积分采样等技术；

**📊 数据集**

在多种 3D 形状数据集（如 Cthulhu、Crab、Sappho、Octopus 等）以及点云/网格、形状空间和参数空间上进行训练和测试；

**📈 对比分析**

与基准方法（如 Robust Laplacian、光谱近似）对比，训练 50k 步后推理仅需约 5 ms/点，误差远低于基线，性能实现实时评估；

**⚠️ 局限性**

局限包括梯度噪声、难以捕捉几何尖点或高度曲率区域、仅支持 Dirichlet/Neumann 边界、收敛不保证完全满足边界条件以及未支持 Robin 边界。

---

## 17. Stabilizing Native Low-Rank LLM Pretraining

**arXiv ID:** 2602.12429 | [PDF](https://arxiv.org/pdf/2602.12429v1)

**作者:** Paul Janson `[一作]` (Concordia University), Eugene Belilovsky `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的训练方法Spectron，能够从头开始使用低秩分解权重训练大型语言模型（LLMs），无需辅助的全秩权重。

**💡 创新点**

创新点在于通过光谱重归一化和梯度正交化相结合，动态约束权重更新，从而解决了低秩训练中的不稳定性问题。

**🔧 技术方法**

使用了光谱重归一化和梯度正交化技术，确保权重更新的光谱范数保持在可控范围内。

**📊 数据集**

在FineWeb数据集上进行了实验，使用了不同规模的模型进行预训练。

**📈 对比分析**

与自指导训练和传统的AdamW方法进行了比较，Spectron在多个指标上表现出更好的性能，且计算开销显著低于自指导训练。

**⚠️ 局限性**

限制在于当前方法主要集中在大型语言模型的训练，未来可能需要针对多模态架构进行扩展。

---

## 18. From sunblock to softblock: Analyzing the correlates of neology in published writing and on social media

**arXiv ID:** 2602.13123 | [PDF](https://arxiv.org/pdf/2602.13123v1)

**作者:** Maria Ryskina `[一作]` (Vector Institute for Artificial Intelligence), Vivek Kulkarni `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对比历史出版文本与Twitter社交媒体中新词出现的语义邻域特征，验证并探讨了供应假设与需求假设在不同语料中的适用性。

**💡 创新点**

创新点在于将先前在出版语料中验证的供应与需求假设扩展至社交媒体语料，并同时使用静态Word2Vec与上下文RoBERTa词向量进行多维度比较。

**🔧 技术方法**

主要技术包括词向量训练与投影（Word2Vec与RoBERTa）、邻域密度与频率增长度量、Spearman相关性与线性回归斜率计算，以及Wilcoxon符号秩检验。

**📊 数据集**

使用的数据集为COHA/COCA两期美国英语文本（1800–1989、1990–2012）以及约260M条英文推文（2007–2021）。

**📈 对比分析**

方法是对新词与频率、长度、语义相似度匹配的对照词在不同邻域阈值下的邻域密度、频率增长的单调性与斜率进行统计比较；结果显示两假设在出版文本中均显著支持，而在Twitter中供应假设显著但需求假设表现较弱。

**⚠️ 局限性**

局限性包括：时间跨度差异导致需求假设在Twitter中的统计噪声；对照词匹配不完全；推文人群与兴趣变动可能混淆频率增长；RoBERTa子词化对社交媒体词向量的适用性不足。

---

## 19. DynaGuide: A Generalizable Dynamic Guidance Framework for Unsupervised Semantic Segmentation

**arXiv ID:** 2602.13020 | [PDF](https://arxiv.org/pdf/2602.13020v1)

**作者:** Boujemaa Guermazi `[一作]` (Toronto Metropolitan University), Naimul Khan `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 3531 | [OpenAlex ID](https://openalex.org/A5069518008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DynaGuide框架，实现无监督语义分割，通过零样本模型生成的全局伪标签与轻量级CNN的局部细化双重引导进行联合训练。

**💡 创新点**

创新点包括：双重引导策略（全局伪标签 + 局部CNN细化）、动态多元损失（特征相似、Huber平滑对角连续性、伪标签对齐）以及可插拔的全局伪标签来源，能在无标签环境下自适应平衡语义一致性与边界精度。

**🔧 技术方法**

采用了DiffSeg/SegFormer等零样本模型产生伪标签、轻量级CNN特征提取、Huber损失、对角连续性约束、动态自适应损失以及聚类/相似度正则化等技术。

**📊 数据集**

在BSD500、PASCAL VOC2012和COCO（含COCO‑Stuff）三个公开数据集上进行实验。

**📈 对比分析**

与多种无监督基线（如IIC、PiCIE、DynaSeg等）以及最新方法对比，BSD500 mIoU提升至0.566/0.570，PASCAL VOC2012 0.474/0.481，COCO 42.18/52.38；相比前沿方法提升显著，同时模型仅包含0.11M参数、6.99GFLOPs，计算效率优异。

**⚠️ 局限性**

局限性包括：在复杂纹理或材质差异较大的场景下可能出现过度分割；对DiffSeg伪标签的纹理敏感性有时导致细节过分拆分；缺乏时序一致性和视频适配能力，未来可进一步探索。

---

## 20. DisSR: Disentangling Speech Representation for Degradation-Prior Guided Cross-Domain Speech Restoration

**arXiv ID:** 2602.12701 | [PDF](https://arxiv.org/pdf/2602.12701v1)

**作者:** Ziqi Liang `[一作]`, Jian Wang `[通讯]` (AntGroup)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了通用跨域语音恢复模型 DisSR，通过将低质量语音分解为内容和说话人风格，再从说话人风格中提取说话人不变的失真表征，利用该失真表征作为条件引导扩散模型进行语音恢复。

**💡 创新点**

创新点包括：① 在说话人风格中提取说话人不变的失真表征；② 将失真表征作为先验条件指导扩散恢复；③ 通过跨域对齐训练提升模型在未见说话人/域上的泛化能力；④ 采用对比学习实现失真表征的有效去耦。

**🔧 技术方法**

技术手段包括：说话人风格与内容编码器、失真表征学习（DRL）、多层最大均值差距（HMMD）跨域对齐、条件 UNet 基础的扩散模型，以及对比学习与 MMD 训练策略。

**📊 数据集**

使用 LibriTTS（train-clean-100/360）生成低质量训练数据，跨域测试集包含 VCTK、AISHELL-3、JSUT，模拟六类失真（量化+重采样、剪切、带限、过驱、噪声、混响）。

**📈 对比分析**

与 VoiceFixer、SelfRemaster、SGMSE+M 等基线相比，DisSR 在 DNSMOS、PESQ-wb、MCD、SSIM、CSIG/CBAK/COVL 等指标上均表现更优，尤其在跨域测试场景下取得显著提升。

**⚠️ 局限性**

局限性：在极低 SNR 或极端失真条件下恢复效果仍不如专用单任务模型；模型训练和推理需要较高计算资源；说话人风格与失真分离的假设在某些极端情况下可能不完全成立。

---

## 21. Interpolation-Inspired Closure Certificates

**arXiv ID:** 2602.12436 | [PDF](https://arxiv.org/pdf/2602.12436v1)

**作者:** Mohammed Adib Oumer `[一作]` (University of Colorado), Majid Zamani `[通讯]` (University of Colorado)

**通讯引用:** 3745 | [OpenAlex ID](https://openalex.org/A5030109984)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的“插值启发型闭包证书（ICC）”方法，用于在不需要单一复杂闭包证书的前提下验证离散时间动力系统的安全性、持续性和一般的 ω-正则（LTL）规范。

**💡 创新点**

创新点在于将闭包证书拆分为多函数序列，并借鉴插值技术逐步构造，使得即使单一模板无法找到证书，也能通过组合简单函数实现验证，从而降低求解复杂度。

**🔧 技术方法**

主要技术包括基于求和平方（SOS）优化的证书合成、S-程序（scenario program）进行数据驱动的插值构造以及与现有闭包证书、Barrier 证书和自动机理论的结合。

**📊 数据集**

在实验中使用了两套仿真数据集：三维洛特卡-沃尔泰拉模型（预期持续性验证）和两室热传递模型（LTL规范验证）。

**📈 对比分析**

与传统单一闭包证书相比，ICC 在相同的多项式度数下能够成功求得证书；在持续性案例中，ICC 仅需四阶多项式和 k=2 即能验证；在 LTL 案例中，ICC 使用三阶多项式和 k=2 解决了六阶闭包证书求解失败的情况，计算时间显著缩短（分别为 133s 与 33s）。

**⚠️ 局限性**

局限性包括：需要先设定多项式模板和 k 的上限，仍受限于计算资源；S-程序需要覆盖状态空间且对 Lipschitz 常数估计有要求；在更高维度或复杂动态系统中，ICC 仍可能因变量增多而导致求解规模膨胀。

---

## 22. GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics

**arXiv ID:** 2602.12617 | [PDF](https://arxiv.org/pdf/2602.12617v1)

**作者:** Modi Jin `[一作]` (Nankai University), Qibin Hou `[通讯]` (Nankai University)

**通讯引用:** 17307 | [OpenAlex ID](https://openalex.org/A5040392623)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作设计了 GeoSeek 地理定位数据集，并基于 Qwen2.5‑VL‑7B 采用两阶段 SFT + GRPO 的训练流程，构建了能够通过人类推理（Chain‑of‑Thought, CoT）进行精细地理位置预测的 GeoAgent 模型。

**💡 创新点**

创新点包括① 由地理专家和专业游戏玩家手工标注的多级 CoT；② 设计了兼顾空间与语义距离的 geo‑similarity 奖励；③ 引入 consistency 奖励以保证 CoT 的完整性与一致性。

**🔧 技术方法**

使用的技术主要包括：VLLM Qwen2.5‑VL‑7B、LoRA 微调、GRPO 强化学习、OpenCage 逆地理编码、多语言语义编码模型、GPT‑4o 对 CoT 进行统一模板化、GPTQ‑INT4 版本的 consistency agent。

**📊 数据集**

使用的数据集为 GeoSeek‑CoT（10k 人类 CoT）、GeoSeek‑Loc（20k 采样样本）、GeoSeek‑Val（3k 评测样本）以及公开的 IMG2GPS3K benchmark。

**📈 对比分析**

在 IM2GPS3K 和 GeoSeek‑Val 上通过国家/区域/城市/洲级准确率和 GeoScore 进行评估，GeoAgent 在国家级准确率达到 89.9%、GeoScore 3314.1，显著优于现有 VLLM 与 RL 方法，SFT+GRPO 的提升尤为显著。

**⚠️ 局限性**

限制方面包括：① 仍受 VLLM 本身偏差影响，难以完全消除多语言/别名问题；② consistency 奖励的权重和阈值需要手工调参；③ 数据规模相对有限，难以覆盖更细粒度的地理细节；④ 训练需要较高算力和专门的 RL 框架。

---

## 23. FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching

**arXiv ID:** 2602.12829 | [PDF](https://arxiv.org/pdf/2602.12829v1)

**作者:** Lei Lv `[一作]` (Shanghai Research Institute for Intelligent Autonomous Systems), Xiao Ma `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Field Least‑Energy Actor‑Critic（FLAC）的框架，将强化学习视为Generalized Schrödinger Bridge问题，利用迭代生成策略（如流/扩散）实现高维连续控制；

**💡 创新点**

核心创新在于将熵正则化转化为路径空间的动能（kinetic energy）正则化，理论证明最小化动能等价于限制终端动作分布与高熵先验的偏离，并通过拉格朗日双重优化实现动能约束的自动调节；

**🔧 技术方法**

使用迭代生成策略、Schrödinger Bridge理论、动能正则化的Actor‑Critic学习、梯度可微求解路径、拉格朗日双重法实现能量预算自适应；

**📊 数据集**

在DMControl（如Dog、CartPole等）和HumanoidBench（Unitree H1）等连续控制基准上进行实验；

**📈 对比分析**

与多种强模型自由基线（TD7、SAC、DIME、SAC‑Flow、FlowRL）以及模型基准TD‑MPC2比较；实验显示FLAC在样本效率和最终回报上与或优于这些基线，在高维任务中也能达到与模型基准相近的性能；

**⚠️ 局限性**

仅采用各维度同质的能量约束，缺乏对不同控制通道的异质或状态依赖性正则化，未来可探索方向。

---

## 24. Adaptive Structured Pruning of Convolutional Neural Networks for Time Series Classification

**arXiv ID:** 2602.12744 | [PDF](https://arxiv.org/pdf/2602.12744v1)

**作者:** Javidan Abdullayev `[一作]` (IRIMAS, Université de Haute Alsace), Germain Forestier `[通讯]` (Monash University)

**通讯引用:** 8083 | [OpenAlex ID](https://openalex.org/A5030958578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对时间序列分类模型的卷积网络，提出了一种自动化的结构化剪枝方法 DSP，能够在无需手动设定剪枝比例的前提下压缩模型并保持甚至提升分类准确率。

**💡 创新点**

创新点在于引入实例级稀疏损失，促使通道级稀疏化；随后通过全局激活分析自动识别并剪除冗余过滤器，实现无超参数的动态剪枝。

**🔧 技术方法**

使用实例稀疏损失（L2+L1 组合）、全局激活阈值剪枝、再训练（Fine‑Tuning 或 Scratch‑Training）以及卷积 CNN 的结构化剪枝技术。

**📊 数据集**

实验使用 UCR 时间序列分类档案中的 128 个数据集，并在 LITETime 与 InceptionTime 两种 CNN 架构上进行验证。

**📈 对比分析**

通过准确率、参数量、FLOPS、内存占用等多维度比较，DSP 在保持或略优于基线准确率的同时，平均压缩率达 58%（LITE）/ 75%（Inception），并在大多数数据集上与静态剪枝及 SOTA 方法竞争或超越。

**⚠️ 局限性**

局限性包括仅在卷积 CNN 上验证，未探究对其他网络结构的适用性；在样本极少或类别数极多的数据集上剪枝效果可能下降；在实际嵌入式设备上仍需结合量化、蒸馏等技术进一步提升效率。

---

## 25. Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL

**arXiv ID:** 2602.13035 | [PDF](https://arxiv.org/pdf/2602.13035v1)

**作者:** Yixiao Zhou `[一作]` (Zhejiang University), Yu Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 73973 | [OpenAlex ID](https://openalex.org/A5090802305)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 IntroLLM 的层级强化学习框架，让大语言模型在生成时自适应地学习并控制采样温度，从而在 RL‑VR（可验证奖励）训练中动态调节探索-利用平衡。

**💡 创新点**

创新点包括：①将采样温度视为可学习的决策变量；②使用混合离散-连续的温度策略（先决定是否改温，再采样具体温度值）；③在 RL‑VR 环境下采用坐标上升式 GRPO 联合训练温度与 token 策略；④实现了对不同提示、位置与训练阶段的上下文感知的温度控制。

**🔧 技术方法**

技术方法主要包括：层级强化学习框架、GRPO（Group Relative Policy Optimization）策略梯度、坐标上升（coordinate ascent）训练、Beta 分布与 Bernoulli 组合的温度策略、对齐温度与 token 的联合优势估计。

**📊 数据集**

使用了多个数学推理基准：MATH 训练集；AIME 2024、AMC 2023、MATH‑500、Minerva Math、OlympiadBench、Omni‑Math 等；并在 GPQA‑Diamond、MMLU‑Pro、HumanEval 等 OOD 任务上做验证。

**📈 对比分析**

与基线比较：固定温度、EAD（预设退火）、TAMPO（序列级自适应）等。IntroLLM 在 Avg@8 与 Pass@8 上均优于所有基线，尤其在 AIME、Omni‑Math 等难度较高的任务中提升显著；在 OOD 任务上亦取得 1–2% 的加分；计算开销几乎可忽略。

**⚠️ 局限性**

局限性：仅适用于能自动验证奖励的任务，无法直接推广到开放式或安全关键场景；对温度上限、下限的设定仍需经验调节；在极大模型或极长序列下的训练稳定性和计算成本仍需进一步评估。

---

## 26. Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents

**arXiv ID:** 2602.12662 | [PDF](https://arxiv.org/pdf/2602.12662v1)

**作者:** Ruihan Yang `[一作]` (Fudan University), Linus `[通讯]` (Tencent Hunyuan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CogRouter 框架，让大语言模型在每一步动态选择四个层级的认知深度，从本能反应到战略规划，以适应多轮任务中的异质认知需求。

**💡 创新点**

创新点在于：①基于 ACT‑R 理论设计四级认知层级；②引入两阶段训练（认知感知监督微调 + 认知感知策略优化）；③在 RL 中使用置信度驱动的优势重加权实现逐步信用分配，避免深度思考的模式崩塌。

**🔧 技术方法**

技术手段包括：大语言模型（如 Qwen2.5‑7B、Llama3.1‑8B）、ACT‑R 认知层级设计、认知感知监督微调（CoSFT）、认知感知策略优化（CoPO）、置信度评估（平均对数概率）以及基于群组的 RL（GRPO、GiGPO）对比。

**📊 数据集**

数据集为 ALFWorld 与 ScienceWorld 两大交互式代理基准，覆盖日常生活、科学实验等多类长时程任务。

**📈 对比分析**

与非思考模型、单层思考模型、SFT、ETO、GRPO、GiGPO、AdaptThink 及 GPT‑4o 等前沿模型对比；在 Qwen2.5‑7B 上平均成功率 82.3%（超过 GPT‑4o +40.3%）、在 Llama3.1‑8B 上 81.0%，并且比 GRPO/ GiGPO 省 62%/57% 代币，显著提升性能与效率。

**⚠️ 局限性**

局限性包括：①认知层级预设固定为四级，缺乏自适应层级扩展；②训练依赖均衡初始化与成功轨迹置信度重加权，可能不适用于极端稀疏奖励场景；③在跨域任务的泛化能力尚待验证。

---

## 27. SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents

**arXiv ID:** 2602.12984 | [PDF](https://arxiv.org/pdf/2602.12984v1)

**作者:** Yujiong Shen `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24137 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 SciAgentGym 环境和 SciAgentBench 基准，提供可执行的 1,780 个科学工具，支持多学科多步交互式推理；并提出 SciForge 数据合成方法生成逻辑一致的训练轨迹。

**💡 创新点**

①将动态工具调用与多步骤科学推理结合，突破传统静态问答；②通过工具依赖图与执行验证生成真实训练数据，显著提升小模型性能；③提供跨学科、跨难度的长程任务评估。

**🔧 技术方法**

基于 LLM 的 ReAct 交互框架、工具签名类型系统、Python 解释器、数据库接口；图搜索采样、阶段感知采样、错误恢复增强的训练策略；SFT、RLHF 等微调方法。

**📊 数据集**

利用 5 个主流科学 QA 资源（ScienceQA、GPQA、R-Bench-V、BMMR、SFE）抽取工具；构造 259 个任务、1,134 子问题，覆盖物理、化学、材料、生物四大领域；合成 11,074 条逻辑轨迹供训练。

**📈 对比分析**

与公开模型（GPT‑5、Claude‑4‑Sonnet、Gemini‑2.5‑Pro、Qwen3‑VL‑235B、GLM‑4.6V 等）对比。SciAgent‑8B 在工具加速条件下达到 41.3% 的整体成功率，显著高于 200B+ 大模型；在 L3 难度任务上表现最差，提示长程推理仍是瓶颈。

**⚠️ 局限性**

1) 长序列工具调用导致错误循环，模型对反馈适应不足；2) 仍需提升跨任务通用错误恢复能力；3) 现有工具覆盖率与实际科研工具差距，限制了更复杂任务的适配；4) 评估主要集中在已验证任务，未覆盖更开放式实验设计。

---

## 28. Can I Have Your Order? Monte-Carlo Tree Search for Slot Filling Ordering in Diffusion Language Models

**arXiv ID:** 2602.12586 | [PDF](https://arxiv.org/pdf/2602.12586v1)

**作者:** Joshua Ong Jun Leang `[一作]` (Imperial College London), Eleonora Giunchiglia `[通讯]` (Imperial College London)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5102734034)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无训练、基于 MCTS 的 Slot‑Selection 框架，优化 Masked Diffusion Models（MDMs）的生成顺序，提升推理与代码生成质量。

**💡 创新点**

创新点在于将 Slot 选择视为 MDP，用模型自信度做先验并结合 roll‑out 长期奖励进行搜索；证明高探索常数比更多模拟更能克服模型的局部置信偏差。

**🔧 技术方法**

主要技术包括 Monte Carlo Tree Search（MCTS）、PUCT 先验引导、基于置信度的奖励与混合回报、以及对 Slot 的 deterministic MDP 表述。

**📊 数据集**

使用六个推理与代码生成基准：GSM8K、MATH500、MBPP、HumanEval、ARC Challenge 与 GPQA‑Diamond。

**📈 对比分析**

在所有基准上均超过现有 MDM 以及大多数 ARMs，平均提升约 3.2% 以上；在 MBPP 和 MATH500 上分别提升 19.5% 与 4.9%，并在 5/6 任务上与 ARMs 性能持平。

**⚠️ 局限性**

局限在于搜索开销随模拟次数线性增长，且在非顺序 Slot 选择上仍依赖模型自信度；对长文本或高维槽数的扩展尚未充分验证。

---

## 29. Retrieval-Augmented Self-Taught Reasoning Model with Adaptive Chain-of-Thought for ASR Named Entity Correction

**arXiv ID:** 2602.12287 | [PDF](https://arxiv.org/pdf/2602.12287v1)

**作者:** Junjie An `[一作]`, Yi Xu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个端到端的命名实体纠错框架 RASTAR，先用 RLM 对 ASR 文本进行高鲁棒性的命名实体识别，再通过 phonetic‑edit‑distance 检索候选实体，最后用自监督的 A‑STAR 推理模型进行实体替换。

**💡 创新点**

创新点在于：① 通过非自回归的 RLM 用整句重表述代替 BIO 标记化的序列标注，显著提升噪声下的 NER；② A‑STAR 采用自教练（self‑distillation）构建 CoT 数据并根据任务难度动态切换“慢思考”和“不思考”模式，从而实现推理效率与准确度的双重优化。

**🔧 技术方法**

技术组合包括检索增强生成（RAG）、拼音级别编辑距离检索、中文 BERT‑based RLM、Qwen3 LLM、Direct Preference Optimization（DPO）自监督训练，以及自适应链式思考机制。

**📊 数据集**

实验使用 AISHELL‑1 语音数据集、其衍生的 115 条高同音混淆的 Homophone 测试集、AISHELL‑NER 注释集合以及构建的 16,168 条实体候选库。

**📈 对比分析**

与强基线 DANCER 直接对比，RLM 在 NER 上提升 F1 2.21%/9.09%，在 NEC 上 NE‑CER 相对降低 10.03%/3.18%；RASTAR 在 8B 规模下在 Homophone 上 NE‑CER 下降 34.42%，并将 CoT token 长度压缩 21%，但 0.6B 模型压缩效果有限。

**⚠️ 局限性**

局限性包括：小规模模型在自适应推理压缩方面表现欠佳；检索时可能带入低相关性实体；仅在中文 AISHELL 语料上验证，缺乏跨语言或更大规模语料的泛化评估。

---

## 30. Eventizing Traditionally Opaque Binary Neural Networks as 1-safe Petri net Models

**arXiv ID:** 2602.13128 | [PDF](https://arxiv.org/pdf/2602.13128v1)

**作者:** Mohamed Tarraf `[一作]` (Newcastle University), Rishad Shafik `[通讯]` (Newcastle University)

**通讯引用:** 2096 | [OpenAlex ID](https://openalex.org/A5077777787)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将二值神经网络的推理与训练过程转化为可执行的Petri网模型，并通过模块化段落构建完整系统。

**💡 创新点**

创新点在于提供一种系统化的Petri网蓝图方法，将BNN的离散事件、因果关系和并发性显式建模，实现可验证的可解释性。

**🔧 技术方法**

使用Workcraft工具集进行Petri网建模、交互式仿真、结构和行为验证，结合STE、Hinge损失、SGD等训练细节的事件化实现。

**📊 数据集**

实验主要使用2输入XOR数据集验证原型，同时通过估算推导KWS6、CIFAR2和MNIST等典型数据集的BNN规模。

**📈 对比分析**

通过对比Petri网BNN与参考软件实现的平均损失曲线，发现两者初始表现相近，训练后Petri网模型在100轮后平均损失约低10%；验证过程中利用结构安全性、无死锁、可达性等检查。

**⚠️ 局限性**

局限性主要是模型规模随网络尺寸呈指数增长，尤其是浮点权重更新的事件化导致PN数量巨大，难以在大规模网络上直接应用；目前未加入偏置项，缺少自动化生成工具。

---

## 31. When Environments Shift: Safe Planning with Generative Priors and Robust Conformal Prediction

**arXiv ID:** 2602.12616 | [PDF](https://arxiv.org/pdf/2602.12616v1)

**作者:** Kaizer Rahaman `[一作]` (Indian Institute of Technology Kharagpur), Lars Lindemann `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在分布偏移下仍能保持概率安全保障的运动规划框架

**💡 创新点**

通过引入可观测的“杂项参数”作为环境变化的上下文，结合条件扩散模型生成合成数据，并在此基础上使用鲁棒一致性预测构造安全预测区域，集成到模型预测控制中

**🔧 技术方法**

条件扩散模型、鲁棒一致性预测（Robust Conformal Prediction）、模型预测控制（MPC）以及 ORCA 仿真器

**📊 数据集**

在 ORCA 仿真器的两车道行驶场景中，使用人工设定的障碍车辆速度和目标位置数据作为训练与测试集

**📈 对比分析**

将四种实现（基线、基线+合成、鲁棒+经验 r、鲁棒+解析 r）与开放/闭环 MPC 进行对比，鲁棒方法在所有 10 个分布偏移环境下都保持了近乎完美的安全性和 90% 的预测覆盖率，非鲁棒基线出现碰撞

**⚠️ 局限性**

需要手动估计或上界 r 以保证鲁棒性，合成数据生成会增加计算开销，方法依赖于可观测的杂项参数，且在多次连续分布偏移或环境对规划者作出响应的情形下效果尚未验证

---

## 32. The Appeal and Reality of Recycling LoRAs with Adaptive Merging

**arXiv ID:** 2602.12323 | [PDF](https://arxiv.org/pdf/2602.12323v1)

**作者:** Haokun Liu `[一作]` (University of Toronto), Colin Raffel `[通讯]` (University of Toronto)

**通讯引用:** 29343 | [OpenAlex ID](https://openalex.org/A5045077843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在近千个公开用户贡献的 LoRA 模型上，系统评估并改进了自适应合并（adaptive merging）方法，验证其在真实世界数据集上的可行性。

**💡 创新点**

创新点在于：①首次使用大规模“野外” LoRA 池进行自适应合并实验；②提出统一框架并全面搜索设计空间，得到最佳配置；③揭示自适应合并的真正收益可能源自正则化而非跨任务知识迁移。

**🔧 技术方法**

主要技术包括 LoRA 参数压缩、模型合并（Simple Averaging、TIES、TSV 等）、自适应合并策略（AdaMerging、π‑Tuning、LoraHub 及自研方法）、多层级系数激活与梯度优化。

**📊 数据集**

使用了 958 条来自 Hugging Face 的基于 Llama 3.1 8B‑Instruct 的 LoRA，评估 62 个多域下游任务（涵盖问答、文本生成等），每个任务提供 100 条样本用于训练与验证。

**📈 对比分析**

与零样本提示（prompting）、单一任务 LoRA 以及非自适应合并方法比较，结果显示：自适应合并可显著提升基线性能（约 17% 提升），但与训练目标任务 LoRA 相比提升有限（仅 2–3%），且当目标 LoRA 在池中时，LoRA 选择策略对性能影响几乎可忽略。

**⚠️ 局限性**

局限性在于：①对目标任务 LoRA 的依赖度高；②在真实环境下大多数 LoRA 贡献有限，难以覆盖所有任务；③现有评估仍以 100 样本为基准，可能低估或高估真实迁移效果；④缺乏对 LoRA 元数据与许可的统一管理。

---

## 33. TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs)

**arXiv ID:** 2602.12833 | [PDF](https://arxiv.org/pdf/2602.12833v1)

**作者:** Zhan Qu `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 TRACE 框架，用冻结的 LLM 在双记忆（全局协议 + 个体协议）和多代理推理循环中实现长期、可解释的临床决策。

**💡 创新点**

创新点：1) 通过离线反射器提取并冻结全局临床规则；2) 双记忆结构将机构知识与患者状态分离；3) 代理循环（Router、Reasoner、Auditor、Steward）实现高效、可审计的时间推理。

**🔧 技术方法**

技术：冻结 LLM + 结构化知识库（Key‑Value cheatsheet）+ 结构化状态压缩（Mitosis）+ 条件审计 + 多代理推理框架。

**📊 数据集**

数据集：MIMIC‑IV（ICU 病例）经过事件包化与语义离散化后形成序列。

**📈 对比分析**

与长上下文、RAG、Monolithic Agent 等基线对比，TRACE 在 Recall@5、协议遵从率≈93% 以及 GPT‑4 评估的临床等价性（3.56–3.96）上均显著提升，且推理成本保持不随时间增长。

**⚠️ 局限性**

局限：仅在单一机构 ICU 数据上验证，数据获取受限；未实现自主临床决策；依赖 LLM 训练数据可能带来偏见；需进一步验证与治理。

---

## 34. Fast and General Automatic Differentiation for Finite-State Methods

**arXiv ID:** 2602.12300 | [PDF](https://arxiv.org/pdf/2602.12300v1)

**作者:** Lucas Ondel Yang `[一作]` (Université Paris-Saclay), Caio Corro `[通讯]` (INSA Rennes)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“morphism‑trick”方法，实现了针对任意符合同构条件的半环的向量-雅可比乘积，从而高效求解有限状态机的前向后向传播。

**💡 创新点**

创新点在于利用半环加法的同构映射将反向传播图“扁平化”，无需存储完整计算图，极大减少内存占用和计算开销，并实现了对多种半环（如log、log_κ、多值、改进的tropical）的一致性实现。

**🔧 技术方法**

主要技术包括：半环代数理论、同构映射（μ）设计、向量-雅可比乘积的通用实现、基于ChainRules.jl的自动微分接口、稀疏矩阵压缩存储以及Julia的Zygote/Enzyme框架。

**📊 数据集**

使用人工构造的有向无环有限状态机（规模从几千到数百万状态、数十亿条边），在log‑semiring以及log_κ‑semiring等场景下进行基准测试。

**📈 对比分析**

与Zygote、Enzyme等主流AD工具对比，所提出的库在前向/后向计算时间几乎相等，且相对传统方法实现了数十至数百倍的加速，同时显著降低了堆分配次数。

**⚠️ 局限性**

局限性包括：尚未支持GPU加速；对不满足同构条件的半环需要额外字段或特殊处理；用户仍需自行提供三类基本导数函数，且对极端稀疏/密集图的性能影响仍待进一步评估。

---

## 35. Left-right asymmetry in predicting brain activity from LLMs' representations emerges with their formal linguistic competence

**arXiv ID:** 2602.12811 | [PDF](https://arxiv.org/pdf/2602.12811v1)

**作者:** Laurent Bonnasse-Gahot `[一作]` (Centre d'Analyse et de Mathématique Sociales), Christophe Pallier `[通讯]` (Cognitive Neuroimaging Unit)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究大型语言模型（LLM）在训练过程中对脑激活的预测性能如何出现左右半球不对称性，并探讨这一现象与模型语法能力的关系。

**💡 创新点**

发现左右不对称性与模型获得正式语言（语法、句法）能力同步出现，且与算术、世界知识等功能性能力无关，首次把半球不对称性与模型语言掌握程度关联起来。

**🔧 技术方法**

使用编码模型（ridge 回归）将 fMRI 信号与 LLM 隐藏层激活相关联，计算脑得分；对训练检查点拟合 Sigmoid 曲线以定位相位转变；评估 LLM 在多种最小对照基准上的表现。

**📊 数据集**

Le Petit Prince fMRI 数据（英语、法语、中文）作为脑激活数据；BLiMP、Zorro、Arithmetic、Dyck、ARC、Hellaswag 及自生成文本的语法可接受性评估等多种基准。

**📈 对比分析**

对比不同训练阶段的脑得分不对称性与各基准的准确率，利用相位转变点和斜率比较；结果显示左右不对称性与 BLiMP、Zorro、文本可接受性同步出现，位置与语法基准相近，而与算术、Dyck、ARC、Hellaswag 不匹配。

**⚠️ 局限性**

仅关注全脑左右不对称性，未分析特定脑区；实验局限于 OLMo-2、Pythia 等特定模型与英语、法语；因果关系未得到验证，需进一步研究不同模型和语言以及脑区级别的对应关系。

---

## 36. IndicFairFace: Balanced Indian Face Dataset for Auditing and Mitigating Geographical Bias in Vision-Language Models

**arXiv ID:** 2602.12659 | [PDF](https://arxiv.org/pdf/2602.12659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 37. Awakening Dormant Users: Generative Recommendation with Counterfactual Functional Role Reasoning

**arXiv ID:** 2602.13134 | [PDF](https://arxiv.org/pdf/2602.13134v1)

**作者:** Huishi Luo `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 10048 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合LLM推理与生成推荐的框架，专门用于激活休眠用户的转化。

**💡 创新点**

创新点在于引入功能角色轨迹推理和逆因果功能角色干预，实现多路径多样化推荐，解决兴趣崩塌和Matthew效应。

**🔧 技术方法**

核心技术包括LLM对齐至语义ID、功能角色链式推理（FR-CoT）、逆因果功能角色推断、生成式行为骨干以及闭环训练（Reasoning-Execution-Feedback-Reflection）。

**📊 数据集**

使用了快手电商平台的真实大规模数据，包含约2200万用户、2800万商品的历史交互，并在约700万测试样本上评估。

**📈 对比分析**

与SASRec、TIGER、U2I/I2I等基线相比，offline Recall@1提升至10.93%（比基线+6.2%），online AB测试订单量提升6.7%，并显著增加长尾商品曝光。

**⚠️ 局限性**

局限性包括对大型LLM的高算力需求、主要在单一电商平台验证、对极度稀疏或无历史用户支持不足，以及逆因果推断参数的敏感性。

---

## 38. MASAR: Motion-Appearance Synergy Refinement for Joint Detection and Trajectory Forecasting

**arXiv ID:** 2602.13003 | [PDF](https://arxiv.org/pdf/2602.13003v1)

**作者:** Mohammed Amine Bencheikh Lehocine `[一作]` (Mercedes-Benz AG), Fabian Flohr `[通讯]` (Munich University of Applied Sciences)

**通讯引用:** 1294 | [OpenAlex ID](https://openalex.org/A5007686963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了MASAR框架，实现了3D检测与轨迹预测的联合端到端推理，完全不依赖跟踪或地图信息；

**💡 创新点**

核心创新是“通过外观引导的过去运动细化（APR）”与“基于过去的预测解码器（PFD）”，通过对象中心的时空机制同时利用外观与运动特征，捕捉长时序依赖；

**🔧 技术方法**

使用Transformer结构、可变形注意力、BEV/视角多尺度特征融合以及多假设运动预测与聚合技术；

**📊 数据集**

在nuScenes数据集上进行训练和评估，使用其3D检测与轨迹预测标注；

**📈 对比分析**

与BEVFormer、SparseBEV、DeTra等最先进模型比较，MASAR在minADE/minFDE提升约20%，EPA提升约10%，同时保持或提升3D检测的mAP/NDS；

**⚠️ 局限性**

局限性包括对突变行为的预测仍有困难、缺乏HD地图上下文导致的轨迹误判、以及在高度静态场景下可能不如常规跟踪方法精确。

---

## 39. Designing RNAs with Language Models

**arXiv ID:** 2602.12470 | [PDF](https://arxiv.org/pdf/2602.12470v1)

**作者:** Milan Gautam `[一作]`, Liang Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将RNA设计问题视为条件序列生成，利用自回归语言模型生成满足二级结构的RNA序列。

**💡 创新点**

创新点在于结合预训练语言模型、受约束解码和强化学习，构建可复用的通用解算器。

**🔧 技术方法**

使用预训练的Qwen2.5-0.5B模型、受约束解码、监督学习和GRPO强化学习。

**📊 数据集**

使用随机产生的1M MFE结构与SAMFEO生成的1M0对进行SL训练，RL使用来自Eterna的1.8万结构并筛选约2.8k结构。

**📈 对比分析**

与SAMFEO、RNAinverse-pf等现有方法在Eterna100、RNAsolo-100、Rfam-Taneda-27四个测试集上对比，SL+RL模型在Boltzmann概率和NED等指标上均超过或持平SOTA，且速度提升约1.7倍。

**⚠️ 局限性**

限制在受约束解码导致约30%额外开销，RL训练耗时长且样本规模受限，且模型对非常长或复杂结构的泛化仍有限。

---

## 40. GPTZero: Robust Detection of LLM-Generated Texts

**arXiv ID:** 2602.13042 | [PDF](https://arxiv.org/pdf/2602.13042v1)

**作者:** George Alexandru Adam `[一作]` (GPTZero), Dongwon Lee `[通讯]` (Pennsylvania State University)

**通讯引用:** 9354 | [OpenAlex ID](https://openalex.org/A5100405086)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一款商业化的AI文本检测器，能够准确区分人类、AI和混合文本，并在推理时提供句子级解释；

**💡 创新点**

创新点包括层次化多任务分类框架、混合类标签、多层红队对抗训练以及Deep Scan可解释性技术；

**🔧 技术方法**

使用深度学习Transformer架构，层次化分类头，文档与句子级联合损失，结合对抗训练、数据增强、翻译与改写扩充，以及基于显著性和遮挡的Deep Scan；

**📊 数据集**

训练与评估数据涵盖数千份人类与AI文本，AI文本来自GPT‑5.2、Gemini 3 Pro、Claude Sonnet 4.5、Grok 4 Fast，涵盖多领域和24种语言的CulturaX、Multitude V3等多语种数据集；

**📈 对比分析**

通过AUC、1% FP阈值下的召回率等指标与Radar、Fast‑DetectGPT、Binoculars、Originality、Pangram等竞争者对比，本文方法在所有领域和语言下均实现>97%召回率、<1%误报率，且在文本绕过攻击与多语种检测中表现最优；

**⚠️ 局限性**

局限性包括缺乏统一评测基准导致的评估偏差、对分布外性能估计不足、对低质量LLM的泛化有限、对抗训练与性能之间的权衡，以及Deep Scan对人类编辑模式捕获不完整。

---

## 41. JARVIS: An Evidence-Grounded Retrieval System for Interpretable Deceptive Reviews Adjudication

**arXiv ID:** 2602.12941 | [PDF](https://arxiv.org/pdf/2602.12941v1)

**作者:** Nan Lu `[一作]` (Beijing Jiaotong University), Shaoyi Xu `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1283 | [OpenAlex ID](https://openalex.org/A5010313255)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了JARVIS框架，用于检索式证据支持的解读式虚假评论裁决。

**💡 创新点**

创新点在于融合稠密与稀疏多模态检索，构建异质证据图并通过LLM进行基于证据的可解释推理。

**🔧 技术方法**

采用CN-CLIP_ViT-B/16、Qwen3-VL-30B-A3B、BGE-M3进行嵌入，使用Qwen3-30B-A3B LLM进行推理。

**📊 数据集**

使用300,000条跨15类商品的真实评论数据集（含文本与图片）。

**📈 对比分析**

与多种监督基线及现行生产模型对比，JARVIS在精度0.988、召回0.901、F1 0.942上明显优于Baseline(0.953/0.830)。

**⚠️ 局限性**

局限在于仍需手动验证的触发机制，模型对极端语义变形的鲁棒性待进一步提升。

---

## 42. The very dependent recursive structure of iterated parametricity in indexed form

**arXiv ID:** 2602.12689 | [PDF](https://arxiv.org/pdf/2602.12689v1)

**作者:** Hugo Herbelin `[一作]` (University of Paris), Ramkumar Ramachandra `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种通过消除等式推理、改为定义推理的迭代参数化构造方法。

**💡 创新点**

创新点在于把原先依赖等式的构造改为依赖递归互定义，显著简化了 Coq 证明，并阐明了依赖结构。

**🔧 技术方法**

主要使用了依赖类型理论、迭代参数化、流（stream）与“插针”（zipper）技术、Coq 定义递归等技术。

**📊 数据集**

无实验数据集，研究纯理论证明。

**📈 对比分析**

与之前的等式推理方法对比，证明更直接、易于检查，且可在 Coq 机器检查下完成。

**⚠️ 局限性**

局限在于仍需手工处理高度递归的互定义，且扩展到无穷维时仍需更完善的技术。

---

## 43. ImageRAGTurbo: Towards One-step Text-to-Image Generation with Retrieval-Augmented Diffusion Models

**arXiv ID:** 2602.12640 | [PDF](https://arxiv.org/pdf/2602.12640v1)

**作者:** Peijie Qiu `[一作]` (Amazon), Rahul Bhagat `[通讯]` (Amazon)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5086631105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ImageRAGTurbo，一种通过检索增强的少步扩散模型实现实时高质量文本到图像生成

**💡 创新点**

首次将检索增强技术与少步扩散模型结合，利用检索的语义上下文在 H‑空间中进行自适应融合，显著提升文本对齐和图像质量而不增加采样步骤

**🔧 技术方法**

检索增强生成（RAG）+ UNet H‑空间适配器（交叉注意力）+ 低秩自适应调参 + 对抗式潜在训练

**📊 数据集**

Synthetic 与真实数据混合：LAION‑Aesthetic 6.25+ 3M prompt 生成的合成图；LAION‑Aesthetic 5.5+ 500K 512×512 图像；OpenImage 0.63M 作为检索库；MS‑COCO 与 TIFA 评测基准

**📈 对比分析**

与 Stable Diffusion v1‑5/v2‑1‑base、Latent Consistency Model、Stable Diffusion Turbo v2‑1‑base、RDM 进行对比，单步 ImageRAGTurbo 在 MS‑COCO 上 CLIP+FID 同 50‑step 生成相近，TIFA 评分 2.2% 提升，推理时间仅比 SD‑Turbo 轻微增加（+2.5%）

**⚠️ 局限性**

依赖 CLIP 检索，检索质量与模型性能相关；适配器仍略微增加推理时间；对多模态检索与组合检索等更细粒度检索方法尚未探索

---

## 44. SLA2: Sparse-Linear Attention with Learnable Routing and QAT

**arXiv ID:** 2602.12675 | [PDF](https://arxiv.org/pdf/2602.12675v1)

**作者:** Jintao Zhang `[一作]` (Tsinghua University), Joseph E. Gonzalez `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SLA2 sparse‑linear attention，结合可学习路由器、α 加权组合以及量化感知低比特注意力，以提升视频扩散模型的计算效率和生成质量。

**💡 创新点**

创新点在于①引入可学习的路由器动态决定每个注意力位置使用稀疏还是线性分支；②采用 α‑加权直接匹配原始 P₁+P₂ 分解，消除 SLA 的行归一误差；③在稀疏分支加入量化感知训练，进一步提升推理速度。

**🔧 技术方法**

使用的技术包括稀疏‑线性注意力、可学习路由器（SoftTop‑k、池化投影）、量化感知训练（QAT）、FlashAttention‑style GPU kernel。

**📊 数据集**

使用约 3,000 个 5 秒长的私有视频数据集，并用 Qwen3‑VL‑Flash 生成对应字幕，构成文本‑视频对进行微调与评估。

**📈 对比分析**

在 Wan2.1‑1.3B 与 14B 视频生成模型上，与 Full Attention、VMoBA、VSA 等基线比较，SLA2 在 90%–97% 稀疏率下保持或超越全注意力的多项视频质量指标，并实现 18.6× 的注意力速度提升，端到端生成延迟显著降低。

**⚠️ 局限性**

局限性包括需要两阶段训练才能稳定收敛；在极高稀疏率（97%）下仍可能出现细节失真；量化误差需通过 QAT 进行补偿；模型对 GPU 资源与显存仍有较高依赖。

---

## 45. QTabGAN: A Hybrid Quantum-Classical GAN for Tabular Data Synthesis

**arXiv ID:** 2602.12704 | [PDF](https://arxiv.org/pdf/2602.12704v1)

**作者:** Subhangi Kumari `[一作]` (Indian Institute of Technology), Vignesh Sivaraman `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为QTabGAN的混合量子-经典生成对抗网络，用于生成高保真度的表格数据。

**💡 创新点**

创新点在于将变分量子电路作为生成器核心，仅用固定数量的量子比特生成概率分布，再通过经典网络映射为完整表格特征，显著提升了对多维混合数据的建模能力。

**🔧 技术方法**

采用变分量子电路、参数位移梯度计算、经典前馈神经网络以及条件生成技术，并在训练中使用量子采样与经典判别器。

**📊 数据集**

使用七个公开表格数据集（King、Insurance、Adult、Credit、Intrusion、Loan、Covertype）进行回归和分类任务评估。

**📈 对比分析**

与CTAB-GAN+、TabularQGAN等经典与量子基线进行ML效用和统计相似度比较，QTabGAN在准确率、F1、EVS、R²以及JSD和相关性差异指标上分别优于对手，差异降至1–3%，提升幅度可达90%以上。

**⚠️ 局限性**

主要局限在于量子电路训练的计算成本高，实验仍基于模拟器，缺乏真实量子硬件验证，且在极小样本或高噪声环境下的鲁棒性待进一步研究。

---

## 46. Asynchronous Verified Semantic Caching for Tiered LLM Architectures

**arXiv ID:** 2602.13165 | [PDF](https://arxiv.org/pdf/2602.13165v1)

**作者:** Asmit Kumar Singh `[一作]` (Apple), Weihua Zhu `[通讯]` (Apple)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Krites，一种异步验证语义缓存策略，在保持传统阈值决策的基础上，通过后台LLM判断并将验证通过的静态答案写入动态缓存，以提升缓存命中率并保证安全质量。

**💡 创新点**

创新点在于将LLM判定从关键路径迁移到后台，采用灰度区触发异步验证，并通过辅助覆盖将静态答案转化为动态缓存中的可更新指针，从而在不增加延迟的前提下显著提升静态答案的使用比例。

**🔧 技术方法**

采用向量嵌入（Φ）、余弦相似度判定、两层静态–动态缓存架构、异步任务队列、LLM评判器以及辅助覆盖写入机制。

**📊 数据集**

使用vCache基准中的两组数据集：SemCacheLMArena（约6万条对话式提示）和SemCacheSearchQueries（约15万条搜索查询）。

**📈 对比分析**

与GPTCache风格的静态阈值基线在相同阈值下对比，Krites将静态来源的命中率分别提升了136.5%（对话式）和290.3%（搜索式），且保持了原有的延迟与错误率。

**⚠️ 局限性**

局限性包括：需依赖高精度LLM判定器（误判会影响后续命中）、后台判定消耗额外计算资源、灰度阈值σ_min需手工调优、以及动态缓存的淘汰策略可能导致已验证的静态指针被移除。

---

## 47. Beyond Benchmarks of IUGC: Rethinking Requirements of Deep Learning Methods for Intrapartum Ultrasound Biometry from Fetal Ultrasound Videos

**arXiv ID:** 2602.12922 | [PDF](https://arxiv.org/pdf/2602.12922v1)

**作者:** Jieyun Bai `[一作]` (Jinan University), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 49917 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出并组织了2024年IUGC挑战赛，目标是通过多任务深度学习实现产程超声视频中的标准平面分类、胎儿头部与耻骨联合分割以及角度与距离测量。

**💡 创新点**

创新点在于首次整合三项任务于同一框架并提供最大规模的多中心产程超声视频数据集，同时对八种算法从预处理、数据增强、学习策略、网络结构与后处理五个维度进行系统评估。

**🔧 技术方法**

所用技术主要包括基于U‑Net/ResNet/DeepLabV3的多任务网络、视频Swin Transformer、双分支分割、伪标签与弱监督、以及各种数据增强与后处理方法。

**📊 数据集**

数据集为774段产程超声视频（共68,106帧），来自济南大学、南方医科大学和中山大学三家医院，涵盖标准平面、分割标注及AoP、HSD测量。

**📈 对比分析**

通过统一评估指标（ACC、F1、DSC、ΔAoP等）和多重排名方法，T1、T2、T3三支团队分别在分类、分割与测量任务中位列前茅，整体排名前三。

**⚠️ 局限性**

局限性包括注释间存在中等偏差、对快速胎动与阴影噪声的鲁棒性不足、模型对不同设备/医院数据的泛化仍有限，并且仍处于实验室阶段，临床应用尚需进一步验证。

---

## 48. TrustMee: Self-Verifying Remote Attestation Evidence

**arXiv ID:** 2602.13148 | [PDF](https://arxiv.org/pdf/2602.13148v1)

**作者:** Parsa Sadri Sinaki `[一作]` (Aalto University), Lachlan J. Gunn `[通讯]` (Aalto University)

**通讯引用:** 348 | [OpenAlex ID](https://openalex.org/A5029873912)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出自检的远程证明证据概念，并实现TrustMee在Trustee框架中集成AMD SEV‑SNP和Intel TDX的WebAssembly验证组件

**💡 创新点**

将平台特定的验证逻辑封装为签名的WebAssembly组件，使验证器仅做签名校验和执行，去掉硬编码的TEE插件，提升可扩展性和安全性

**🔧 技术方法**

WebAssembly组件模型、WIT接口、Wasmtime运行时、Rust/openssl、waki（WASI-HTTP）、Pure-Rust DCAP库、EAT EAR格式、RATS模型、Trustee框架

**📊 数据集**

未使用公开数据集，而是使用AMD SEV‑SNP与Intel TDX的真实证明报告与其相应的VCEK/VLEK或PCK链（从AMD kds/Intel DCAP获取）

**📈 对比分析**

与Trustee原生驱动对比，端到端延迟在网络开销占主导时误差<3%，无网络时AMD SEV‑SNP误差~400%但TDP误差<10%；平台特定验证时间对AMD SEV‑SNP提升≈8×，对TDP≈6%；冷启动比热启动慢约20%，但可通过缓存和纯Rust库进一步优化

**⚠️ 局限性**

WebAssembly缺乏硬件加速的加密性能导致验证慢；只实现了SEV‑SNP和TDX；对组件签名与网络访问控制不足，易被攻击者利用；需要更完善的PKI与资源限制机制

---

## 49. RGAlign-Rec: Ranking-Guided Alignment for Latent Query Reasoning in Recommendation Systems

**arXiv ID:** 2602.12968 | [PDF](https://arxiv.org/pdf/2602.12968v1)

**作者:** Junhua Liu `[一作]` (Forth AI), Kwan Hui Lim `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2040 | [OpenAlex ID](https://openalex.org/A5005406384)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 RGAlign-Rec 框架，将 LLM 语义推理器与查询增强型三塔推荐模型相结合，实现聊天机器人在零查询环境下的主动意图预测。

**💡 创新点**

创新点包括：① 通过排名模型作为奖励模型进行排名引导对齐（RGA），实现语义空间与排名目标的闭环同步；② 采用多 LLM 生成最佳查询（Best‑of‑N 采样）与对比学习实现语义与排名的一致性；③ 设计了三塔架构与 Last‑Token Pooling 提升语义表示质量。

**🔧 技术方法**

使用技术包括：LLM 语义推理器 Qwen3‑4B（Last‑Token Pooling）、三塔三相互注意的推荐模型、ListNet + KL 损失、奖励模型、SFT/DPO、InfoNCE 对比学习、多 LLM 交叉采样、8‑bit 量化与 DeepSpeed ZeRO‑3 并行、NEFTune 噪声正则化。

**📊 数据集**

数据集为 Shopee 机器人交互日志，覆盖 8 个地区，约 7.53M 条样本，包含 89 个特征，使用 5 周数据训练并留出 1 周验证；人工标注用于评估 Intent Hit Rate。

**📈 对比分析**

与工业基线（两塔 AutoInt+MoE）及 QE‑Rec 进行离线与在线对比。离线 GAUC 提升 0.12%，Recall@3 +0.56%，NDCG@3 +0.35%；在线 CTR@3 从基线提升 0.98%，再到 RGAlign‑Rec 的 0.13%。整体排名效果显著提升，IHR 也提升 1.43%/0.30%。

**⚠️ 局限性**

局限性：① 负样本采样质量对 DPO 效果敏感；② 对话内容质量和响应生成未同步提升，导致 CSAT 略有下降；③ 依赖商业 LLM 及多模型蒸馏，部署成本和可解释性受限；④ 训练流程复杂，需要多阶段闭环迭代。

---

## 50. CAPTS: Channel-Aware, Preference-Aligned Trigger Selection for Multi-Channel Item-to-Item Retrieval

**arXiv ID:** 2602.12564 | [PDF](https://arxiv.org/pdf/2602.12564v1)

**作者:** Xiaoyou Zhou `[一作]` (Kuaishou Technology), Guorui Zhou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在多通道两跳（U2I2I）推荐系统中，提出了 CAPTS 框架，用于从用户历史行为中选择触发器，并将其分配给不同的检索通道。

**💡 创新点**

创新点包括：① 用“look‑ahead”方式将触发器的价值归因到其在 downstream I2I 检索中产生的后续互动，从而消除传统基于触发器自身反馈的偏差；② 通过 Channel‑Adaptive Trigger Routing（CATR）学习通道特定的触发器价值并在路由时引入多样性（Uniqueness）目标，实现触发器的跨通道协同分配。

**🔧 技术方法**

技术手段：① Value Attribution Module（VAM）通过离线 replay 计算每个触发器在各通道上未来窗口内产生的观看时长并生成二分类标签；② CATR 采用共享编码器 + 目标注意力，通道特定的 value head、calibrator、uniqueness head，联合训练带有校准损失和多样性损失的二分类损失；③ 系统化部署包含离线近线作业、在线实时评分与路由。

**📊 数据集**

使用的真实业务数据集为快手（Kwai）国际短视频平台日志，包含约 107 万活跃用户、2736 万视频、12.3 亿交互，离线实验以每用户最近 100 条有效观看为测试集。

**📈 对比分析**

与规则基（TagTop、LTV、NIC、Recent）和模型基（PDN、LIC）触发器方案对比，CAPTS 在离线 Recall@K 上相较最佳基线提升 25%–10%（K=100–2000），单通道上提升 27%–156%，在线 A/B 测试显示平均每设备停留时间提升 0.351%，总观看时长提升 0.446%，DAU 上升 0.098%。

**⚠️ 局限性**

局限性：① 价值归因依赖未来窗口的 replay，若日志缺失或延迟可能影响准确性；② 需要大规模离线计算与缓存，部署成本高；③ 仅针对 U2I2I 结构，对其他检索模式的推广仍需验证。

---

## 51. Agentic AI for Robot Control: Flexible but still Fragile

**arXiv ID:** 2602.13081 | [PDF](https://arxiv.org/pdf/2602.13081v1)

**作者:** Oscar Lima `[一作]` (German Research Center for Artificial Intelligence), Joachim Hertzberg `[通讯]` (Osnabrück University)

**通讯引用:** 5615 | [OpenAlex ID](https://openalex.org/A5020594579)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两款物理机器人平台上实现了基于大型语言模型的代理式控制系统，使其能够通过规划、工具调用和循环监控执行自然语言指令。

**💡 创新点**

创新点在于将LLM作为高层决策者与机器人技能 API 对接，构建了带有自我反思、事件检测与操作者干预的闭环规划-执行-恢复流程。

**🔧 技术方法**

采用 OpenAI o3 及其工具调用接口、结构化提示、语义状态快照、机器人技能 API 以及多代理架构（规划执行者、目标完成评论者）。

**📊 数据集**

实验使用的主要数据来自现场机器人观测与Gazebo模拟；未采用公开数据集，而是自建的对象位置、语义事实及任务指令集合。

**📈 对比分析**

通过定性实验比较两平台的任务完成率、误差与恢复能力，结果显示系统可完成大多数指令但仍表现出非确定性、指令跟随错误及对提示敏感的脆弱性。

**⚠️ 局限性**

主要限制包括对提示的高敏感性、事件检查的轮询方式导致的延迟与无法即时中断、成本与网络依赖，以及长期执行中状态信息老化导致的失败。

---

## 52. Lamer-SSL: Layer-aware Mixture of LoRA Experts for Continual Multilingual Expansion of Self-supervised Models without Forgetting

**arXiv ID:** 2602.12746 | [PDF](https://arxiv.org/pdf/2602.12746v1)

**作者:** Jing Xu `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9461 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为 Lamer-SSL 的参数高效持续多语言自监督学习框架，用于在保持已学语言性能的同时快速扩展新语言。

**💡 创新点**

创新点在于将 LoRA 与 Mixture-of-Experts 结合，构建 Layer‑Aware Mixture of LoRA Experts（Lamer）模块；专家在层级上按深度分配，深层获得更多专家；同时采用轻量级回放策略缓解灾难性遗忘。

**🔧 技术方法**

使用了 HuBERT‑Large 作为基础模型，Lamer 模块内的 LoRA experts、Top‑K 路由器、负载平衡损失以及回放样本；训练目标为掩码预测加负载平衡正则化。

**📊 数据集**

在 CommonVoice 11.0（普通话、粤语、英语）和 MLLR‑147 公共清单中采集数据；回放样本为 100 小时英文 CommonVoice；在 ML‑SUPERB 语音识别和语言识别任务上进行评估。

**📈 对比分析**

与单语 HuBERT‑Large、mHuBERT‑147（多语言预训练）以及 LoRA 单纯适配器基线对比。Lamer‑SSL 在 ASR 任务上实现平均 CER 10.50%（CommonVoice）/10.13%（Fleurs），在 LID 任务上平均 ACC 99.22%，明显优于其他方法，且仅训练 2.14% 参数。

**⚠️ 局限性**

局限性包括：需要手工设置层级专家分配比例；回放策略仅使用少量样本，可能对极低资源语言效果不佳；在极大规模多语言场景下，专家管理与路由的计算复杂度仍需进一步优化。

---

## 53. Efficient Personalized Federated PCA with Manifold Optimization for IoT Anomaly Detection

**arXiv ID:** 2602.12622 | [PDF](https://arxiv.org/pdf/2602.12622v1)

**作者:** Xianchao Xiu `[一作]` (Shanghai University), Wanquan Liu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 51132 | [OpenAlex ID](https://openalex.org/A5100641142)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种个性化联邦 PCA（FedEP）框架，用于 IoT 网络的异常检测

**💡 创新点**

创新点在于将个性化与鲁棒性双稀疏正则（ℓ1 与 ℓ2,1）相结合，并在联邦学习中采用 Stiefel 流形优化实现模型个性化而非强一致性

**🔧 技术方法**

采用 ADMM + 流形优化（Stiefel 重排）、半光滑牛顿（SSN）、软阈值（prox_ℓ1）以及 Sherman–Morrison–Woodbury 公式等技术求解非凸优化

**📊 数据集**

实验使用三个公开 IoT 入侵检测数据集：TON‑IoT、UNSW‑NB15、NSL‑KDD

**📈 对比分析**

与基线 FedPG 进行对比，FedEP 在 Acc、F1‑score、AUC 上均提升约 1%–3%，且训练时间更短、收敛更快（第 10 轮即可达到峰值）

**⚠️ 局限性**

局限性：仅处理线性 PCA；模型参数量仍较大，通信压缩未深入研究；对极端非线性或极度不平衡数据的适应性待进一步验证

---

## 54. Artic: AI-oriented Real-time Communication for MLLM Video Assistant

**arXiv ID:** 2602.12641 | [PDF](https://arxiv.org/pdf/2602.12641v1)

**作者:** Jiangkai Wu `[一作]` (Peking University), Xinggong Zhang `[通讯]` (Peking University)

**通讯引用:** 8298 | [OpenAlex ID](https://openalex.org/A5001396675)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向多模态大语言模型视频助手的实时通信框架 Artic，专注于减少网络抖动对视频质量和回答准确性的影响。

**💡 创新点**

核心创新点是（1）ReCapABR：利用 MLLM 的置信度反馈自适应控制码率，提前截断到饱和点以预留带宽缓冲；（2）ZeCoStream：通过 MLLM 实时反馈关注区域，实现无额外开销的上下文感知编码；（3）DeViBench：首个评估 RTC 造成的视频退化对 MLLM 准确率影响的基准。

**🔧 技术方法**

技术包括：基于 MLLM 的置信度回馈与区域定位；量化参数映射与 QP 自适应；与传统 CC（GCC/BBR）协同；基于 x265 编码与 Mahimahi 网络仿真；使用 MLLM API 进行实时推理与反馈。

**📊 数据集**

使用的主要数据集包括：收集自行业 MLLM 基准视频（如 StreamingBench）转码后生成的 DeViBench（1,968 条 QA，88,680 秒视频），以及 5G 上行带宽真实轨迹（车辆、步行等场景）。

**📈 对比分析**

与传统 WebRTC（采用 H.265+GCC/BBR）进行对比，实验表明 Artic 在真实网络条件下平均提升 15.12% 的回答准确率，平均降低 135.31 ms 的帧延迟，且在低带宽下仍能保持 0.9 以上准确率；同时上行带宽使用率下降 46%~70%，成本提升约 27%。

**⚠️ 局限性**

局限性包括：对 MLLM 置信度与定位反馈的准确性高度依赖，若出现误判可能导致码率误调；仅针对单向上行场景，未验证在双向或高峰拥塞环境下的鲁棒性；实现中需依赖服务器端 MLLM API，增加了系统复杂度与运营成本；在极端低带宽或高时延环境下的性能仍需进一步验证。

---

## 55. Which Algorithms Can Graph Neural Networks Learn?

**arXiv ID:** 2602.13106 | [PDF](https://arxiv.org/pdf/2602.13106v1)

**作者:** Solveig Wittig `[一作]` (RWTH Aachen University), Christopher Morris `[通讯]` (RWTH Aachen University)

**通讯引用:** 16736 | [OpenAlex ID](https://openalex.org/A5111798651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套理论框架，说明在满足特定条件下，消息传递图神经网络（MPNN）可以通过有限数据学习并在任意大小图上逼近多类离散算法的行为；同时给出了一组可区分性与可逼近性的定理，并将其应用于单源最短路、最小生成树、0/1背包等经典图算法；

**💡 创新点**

将可学习性问题与图空间的伪度量、覆盖数以及Lipschitz常数关联，首次给出MPNN可学习性的充分条件，并证明对多种算法可构造极小训练集与可微分正则化；同时提供了对标准MPNN在表达性上的不可学习性上限与更强表达性变体的可学习性判定；

**🔧 技术方法**

基于MPNN的理论分析、伪度量空间构造、Lipschitz覆盖数论证、正则化诱导外推、以及最短路等算法的迭代公式映射到MPNN层的实现；在实验中采用梯度下降训练的两层MPNN，并引入可微分的L1正则化；

**📊 数据集**

使用合成Erdős–Rényi图、路径图以及基于Bellman–Ford的训练实例；训练集由K+1条路径构成，测试集包含64、256、1024顶点的随机图；

**📈 对比分析**

与标准MPNN、对照的表达更强的MPNN，以及不同正则化（L1、L2、论文中提出的正则化）进行对比；实验表明：① 采用理论指导的训练集和正则化可在更大图上保持低误差；② 更强表达性的MPNN在大图测试上显著优于标准MPNN；③ 可微分正则化在稳定性和泛化误差上优于传统L1/L2正则化；

**⚠️ 局限性**

理论假设需构造满足伪度量与覆盖数条件的训练集，实际数据中难以保证；未提供梯度下降收敛保证；仅针对多项式时间算法，未覆盖近似算法或NP难问题的泛化；

---

## 56. Synthetic Interaction Data for Scalable Personalization in Large Language Models

**arXiv ID:** 2602.12394 | [PDF](https://arxiv.org/pdf/2602.12394v1)

**作者:** Yuchen Ma `[一作]` (University of Notre Dame), Stefan Feuerriegel `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 7447 | [OpenAlex ID](https://openalex.org/A5081442873)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于LLM的高保真合成数据生成框架，用动态偏好过程模拟用户多轮交互，并在此基础上构建了可扩展的多轮个性化对话数据集；同时提出了PPOpt——一种黑箱、模型无关的提示优化方法，通过“先推理再优化”流程从历史交互中推断用户偏好并改写新会话的首个提示，从而实现对已部署LLM的个性化控制。

**💡 创新点**

①把用户偏好建模为动态过程而非静态Persona；②在合成数据中引入多层语义噪声，模拟真实用户行为；③在优化时采用多目标强化学习，兼顾偏好推理质量与任务结果，避免捷径学习；④实现完全无参数更新的黑箱提示优化，可直接应用于闭源或冻结模型。

**🔧 技术方法**

使用三代理LLM（User、Assistant、Distractor）生成合成对话；强化学习（policy gradient）结合LMM‑as‑judge的二进制任务奖励和偏好一致性奖励；自回归语言模型作为提示优化器；SFT作为冷启动；数据增广与噪声注入。

**📊 数据集**

合成数据集PPOpt‑Data（≈2k Persona、10k+ 对话），从公开任务数据（QA、编程、数学、化学等）采样种子问题，并利用Persona Bank和噪声模型生成多轮交互；此外还使用真实用户测试集评估。

**📈 对比分析**

与历史增量提示、Persona查询重写、Few‑Shot ICL、控制器引导提示等基线进行对比；在合成和真实测试集上，PPOpt在个性化分数提升约33%（Δ≈1.8分），而任务完成度差异仅约2%；相比其他方法，PPOpt在保持任务性能的前提下获得最大个性化收益。

**⚠️ 局限性**

主要局限：①合成数据虽然高保真但仍可能缺乏部分真实用户复杂行为；②RL过程对奖励设定敏感，需精细调参；③依赖LMM‑as‑judge的评判准确性；④在极端噪声或低数据量场景下性能可能下降；⑤对模型规模或计算资源要求较高，需进一步验证在更大闭源模型上的可扩展性。

---

## 57. INHerit-SG: Incremental Hierarchical Semantic Scene Graphs with RAG-Style Retrieval

**arXiv ID:** 2602.12971 | [PDF](https://arxiv.org/pdf/2602.12971v1)

**作者:** YukTungSamuel Fang `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 12926 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个实时增量的层次语义场景图框架，支持机器人根据自然语言指令进行语义推理与定位。

**💡 创新点**

① 将语义图转化为 RAG‑式语言索引知识库并以自然语言描述为显式锚；② 异步双流构建与事件触发更新，仅在语义事件发生时重构拓扑；③ 采用多角色 LLM 解析约束与 VLM 视觉验证的闭环检索。

**🔧 技术方法**

双流几何与语义感知（SAM3、DINOv3、VLM）、事件触发更新、层次图结构、基于 LLM 的约束解析与多角色推理、VLM 视觉验证、RAG 检索与硬软过滤。

**📊 数据集**

新构建的 HM3DSem‑SQR 数据集以及三套真实场景数据集。

**📈 对比分析**

与 ConceptGraphs、Embodied‑RAG、HOV‑SG、DualMap 等基线在 HM3DSem‑SQR 和真实环境下的检索精度、语义精度及资源消耗进行对比，INHerit‑SG 在包含否定、链式关系等复杂查询上取得最高准确率，同时存储和计算成本最低。

**⚠️ 局限性**

依赖大型 LLM/VLM 导致计算开销大；对高度动态或频繁物体重排的环境适应性不足；事件触发更新假设拓扑相对稳定，难以处理极端拓扑变化。

---

## 58. Backdoor Attacks on Contrastive Continual Learning for IoT Systems

**arXiv ID:** 2602.13062 | [PDF](https://arxiv.org/pdf/2602.13062v1)

**作者:** Alfous Tim `[一作]`, Kuniyilh Simi D `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文系统分析了对IoT系统中对比学习持续学习（CCL）的后门攻击，构建了攻击目标、持久机制及层级化分类法，并评估了在内存、边缘与联邦场景下的防御策略。

**💡 创新点**

创新点在于首次从嵌入层面阐释后门持久性、提出针对IoT的多维度攻击分类框架，并将传统防御拓展至对比嵌入空间、回放内存及联邦聚合的安全维度。

**🔧 技术方法**

使用技术包括对比学习损失（InfoNCE/监督对比）、回放与蒸馏正则、嵌入几何审计、回放筛选与动态蒸馏权重调节，以及联邦鲁棒聚合方法。

**📊 数据集**

文章未给出公开数据集，主要采用工业物联网仿真/真实传感器流数据进行实验与分析。

**📈 对比分析**

通过与传统持续学习和监督学习的对比，发现CCL在回放放大、嵌入持久性与联邦易受攻击方面更易受影响；防御实验显示仅在边缘与联邦环境下轻量化嵌入审计和回放过滤能部分缓解风险，但总体性能提升有限。

**⚠️ 局限性**

局限性包括缺乏正式安全证明、实验规模受限、对高维多模态数据的泛化不明，以及在资源受限边缘设备上实现高效防御的技术挑战。

---

## 59. The Influence of Code Smells in Efferent Neighbors on Class Stability

**arXiv ID:** 2602.12950 | [PDF](https://arxiv.org/pdf/2602.12950v1)

**作者:** Zushuai Zhang `[一作]` (University of Auckland), Ewan Tempero `[通讯]` (University of Auckland)

**通讯引用:** 3590 | [OpenAlex ID](https://openalex.org/A5069747561)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统性研究了代码 Smell 在类的外向依赖邻居中的存在如何影响该类的稳定性，并考虑了 Smell 的相互关联和相互作用；

**💡 创新点**

创新点在于首次将 Smell 的相互作用与类稳定性关联起来，提出并量化了“外向 Smell 互作”概念，并通过大规模经验数据检验其影响；

**🔧 技术方法**

采用 JSpIRIT 进行 Smell 检测，CodeQL 识别静态依赖，RefactoringMiner 追踪类重构，Git 命令提取提交差异，并用负二项 GLM 进行统计建模；

**📊 数据集**

使用从 GitHub 上挑选的 100 个顶级 Java 项目，在每个项目中选取至少一年提交历史的快照进行分析；

**📈 对比分析**

通过负二项 GLM 并加入项目随机截距来控制项目差异，使用 IRR 与 AME 评估效应大小，结果显示 Smell 的数量、种类、相互关联以及相互作用均与类稳定性呈负相关，表明其显著影响；

**⚠️ 局限性**

局限包括：使用单一快照测量 Smell 及依赖导致时间匹配偏差、仅考虑 Java 代码、对所有依赖类型统一权重、未覆盖重复代码、模型假设与工具检测的局限等。

---

## 60. Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions

**arXiv ID:** 2602.12902 | [PDF](https://arxiv.org/pdf/2602.12902v1)

**作者:** Fox Pettersen `[一作]` (Oxford Brookes University), Hong Zhu `[通讯]` (Oxford Brookes University)

**通讯引用:** 2734 | [OpenAlex ID](https://openalex.org/A5101595936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于连续天气强度的操作鲁棒性评估方法，并通过合成数据实现对自动驾驶目标检测模型的鲁棒性量化；

**💡 创新点**

创新点在于：①将鲁棒性定义为“第一次失效系数”的平均值；②使用数据增强生成连续强度的天气样本；③提供了可直接用于不同模型对比和训练效果评估的AFFC指标；

**🔧 技术方法**

采用深度学习目标检测模型（YOLOv5s、YOLOv11s、Faster R‑CNN、Detectron2）和Automold等图像增强库，结合线性/二分搜索算法求取第一次失效强度；

**📊 数据集**

使用100张实车与COCO混合测试集，生成57,400张增强样本；在第二组实验中使用OBRA团队的100张真实赛道标志图像；

**📈 对比分析**

通过AFFC对比四种模型在七种天气/光照扰动下的平均鲁棒性，结果显示Faster R‑CNN最高（71.9%），YOLO系列约43%，Detectron2最低（39.6%）；对抗训练实验表明适度的合成天气训练可提升鲁棒性，但过度训练导致收益递减甚至退化；

**⚠️ 局限性**

局限性包括：①仅考虑单目标检测；②天气增强器对强度的刻画不够连续，导致置信度曲线突变；③实验样本量小、搜索步长大；④未对多目标检测、真实天气映射进行验证；

---

## 61. Closing the Loop: A Control-Theoretic Framework for Provably Stable Time Series Forecasting with LLMs

**arXiv ID:** 2602.12756 | [PDF](https://arxiv.org/pdf/2602.12756v1)

**作者:** Xingyu Zhang `[一作]` (University of Chinese Academy of Sciences), Wenwen Qiang `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在冻结的语言模型前后插入残差估计器和反馈控制器，将时间序列自回归预测改造成闭环控制系统，以减小预测误差的累积。

**💡 创新点**

提出基于控制理论的闭环框架 F-LLM：引入可学习的残差估计器（观测器）与反馈控制器，并在训练中加入局部 Lipschitz 正则化，以保证系统可控与收敛，首次在 LLM 时序预测中实现理论上可界定的误差上限。

**🔧 技术方法**

核心技术包括：解耦的 patch‑based LLM 预测器、残差估计器（轻量化网络）、反馈控制器、两阶段课程学习、局部 Lipschitz 约束、与 frozen LLM 的无缝集成。

**📊 数据集**

在七个多变量时序数据集（ETTh1/2、ETTm1/2、ECL、Weather、Traffic）以及 M3/M4 竞赛数据的零样本迁移任务上进行实验。

**📈 对比分析**

与传统深度学习基线（PatchTST、DLinear、FEDformer、TimesNet 等）及 LLM 基础方法（AutoTimes、Time‑LLM、UniTime、LVICL、FPT）对比，F‑LLM 在多数长周期预测任务中 MSE/MAE 提升 5–10% 以上，零样本迁移 SMAPE 亦显著优于对手，并且仅增加 5% 以内的推理时间。

**⚠️ 局限性**

局限性包括：需要在 LLM 上施加局部 Lipschitz 约束，且理论保证基于可观测残差估计的前提；目前仅验证了 decoder‑only LLM，未探索 transformer‑encoder 或多模态 LLM 的适配；对极端长时域（>720 步）或高噪声场景的鲁棒性仍待进一步评估。

---

## 62. Eva-Tracker: ESDF-update-free, Visibility-aware Planning with Target Reacquisition for Robust Aerial Tracking

**arXiv ID:** 2602.12549 | [PDF](https://arxiv.org/pdf/2602.12549v1)

**作者:** Yue Lin `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**通讯引用:** 46926 | [OpenAlex ID](https://openalex.org/A5006986293)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种ESDF更新免除、可视化感知的无人机跟踪规划框架Eva-Tracker，实现实时可视化跟踪与目标重获。

**💡 创新点**

提出预计算的FoV-ESDF、可视化初始路径生成和可恢复路径规划，消除ESDF更新并统一可视化、距离与遮挡优化。

**🔧 技术方法**

基于Bézier曲线目标预测、MINCO轨迹优化、FoV-ESDF梯度可微目标函数，并使用L-BFGS求解。

**📊 数据集**

在100个随机障碍的仿真环境以及真实室内外实验环境中评估，无需公开数据集。

**📈 对比分析**

与Elastic Tracker、Vis-Planner、SF-Tracker对比，Eva-Tracker在遮挡率、跟踪距离、角误差、失败率和计算时间上显著优于其他方法。

**⚠️ 局限性**

目标重获仅适用于短期遮挡，且FoV-ESDF固定摄像机视场，难以适配动态视场。

---

## 63. Evaluating the Homogeneity of Keyphrase Prediction Models

**arXiv ID:** 2602.12989 | [PDF](https://arxiv.org/pdf/2602.12989v1)

**作者:** Maël Houbre `[一作]`, Beatrice Daille `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种评估关键短语预测模型同质性（homogeneity）的自动化方法，利用文档对（重述版和共享关键短语的对）并通过Hooper’s CP与Rodgers’ CP两种一致性指标衡量模型在不同文档上是否能一致给出相同或相似的关键短语。

**💡 创新点**

创新点在于首次将同质性作为关键短语质量属性进行量化评估，并构建了两套全新的测试集（基于GPT-4o的重述和基于共享关键短语的对），同时揭示生成模型在语义相似文档上比提取模型更具同质性，而在词汇相似文档上则相反。

**🔧 技术方法**

采用的技术包括多种关键短语提取器（MultipartiteRank、TF‑IDF）与生成器（CopyRNN、CorrRNN、TG‑Net、TGRF、BART、BART‑60p、One2Set、One2Set‑60p），训练数据来自KP20k；评估时使用Hooper’s CP、Rodgers’ CP、ROUGE‑1与相关系数分析模型对重述的鲁棒性。

**📊 数据集**

使用的数据集包括Inspec（500篇科学文献及其重述版）、KP20k（20,000篇论文），以及Inspec的back‑translation和paraphrase版本，用于构建同质性评估对。

**📈 对比分析**

通过将模型在两种文档对上的同质性得分进行对比，实验发现：在词汇重叠较高的重述对中，提取模型往往获得更高的Hooper’s CP；而在共享关键短语且更抽象的对中，生成模型（尤其是BART‑60p和One2Set‑60p）表现出更高的同质性。总体而言，生成模型的缺失关键短语能力在不同语境下对同质性影响不一。

**⚠️ 局限性**

局限性包括：实验仅覆盖计算机科学文献，缺乏对其他领域（如医学、法律）的验证；评估集合规模有限，难以进行显著性检验；同质性评估依赖人工设定的文档相似度阈值，可能影响结果的一致性与泛化。

---

## 64. Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference

**arXiv ID:** 2602.12542 | [PDF](https://arxiv.org/pdf/2602.12542v1)

**作者:** Pengfei Hu `[一作]` (Stevens Institute of Technology), Yue Ning `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 2014 | [OpenAlex ID](https://openalex.org/A5024383883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对电子病历预测任务，提出了一种基于稀疏自动编码器和正交推断的域适应框架，能够将患者表示分解为标签相关的不可变子空间和域相关的可变残差，并通过稀疏维度归因实现透明可解释。

**💡 创新点**

创新点包括：① 在域适应中首次引入字典诱导的稀疏向量来实现可解释性；② 通过M-正交投影将表示拆分为不可变与可变子空间，且给出理论保证；③ 采用正交残差直接监督域分类，实现域信息的显式分离。

**🔧 技术方法**

技术包括：稀疏自动编码器（SAE）生成稀疏字典特征；M-正交投影与字典诱导度量；最大均值差（MMD）对齐；域分类器与交叉熵监督；以及对稀疏维度进行 ablation 的解释方法。

**📊 数据集**

使用了两个真实 EHR 数据集：eICU（来自 208 家医院的 ICU 记录）和 OCHIN（来自 2400+ 设施的纵向病历）。

**📈 对比分析**

与多种基线（Oracle、Base、DANN、RMMD、RSDA、BUA、RCG、CST、SSRT 等）比较，实验表明该方法在空间与时间域适应场景下均能提升 9%–16% 的 F1/AUROC，显著优于传统特征对齐与自训练方法。

**⚠️ 局限性**

局限性：① 需要先验的稀疏字典和正交投影，可能对不同任务或特征空间的适配性有限；② 解释能力主要基于代码层面的归因，无法直接解释更细粒度的临床决策；③ 训练过程较为复杂，需多阶段调参。

---

## 65. Design Environment of Quantization-Aware Edge AI Hardware for Few-Shot Learning

**arXiv ID:** 2602.12295 | [PDF](https://arxiv.org/pdf/2602.12295v1)

**作者:** R. Kanda `[一作]` (Tohoku University), T. Hanyu `[通讯]` (Tohoku University)

**通讯引用:** 5484 | [OpenAlex ID](https://openalex.org/A5062434040)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在边缘 AI 硬件上实现了从预训练到评估全流程的定点量化，确保软件与硬件的准确性一致。

**💡 创新点**

创新点在于将量化技术完整嵌入整个少样本学习流水线，并通过可调整数/小数位宽实现对硬件资源的进一步压缩。

**🔧 技术方法**

使用 Brevitas 的 QAT 与 PTQ 量化模块、ResNet12 主干、Nearest Class Mean 分类器，以及 Tensil 框架进行硬件映射。

**📊 数据集**

采用 miniImageNet 与 CIFAR-FS 两个常用的少样本学习数据集进行评估。

**📈 对比分析**

通过对 1-shot 与 5-shot 任务在不同位宽（3-16 位）下的准确率比较，显示 QAT 在 5 位整数/5 位小数时仅下降约 6% 而 PTQ 在 6 位时可与 FP 相当，证明低位宽可满足硬件需求。

**⚠️ 局限性**

局限在于 Tensil 的硬件实现受限，导致整数/小数位宽固定为 8 或 16 位，未来需探索更灵活的框架以进一步优化硬件设计。

---

## 66. SWING: Unlocking Implicit Graph Representations for Graph Random Features

**arXiv ID:** 2602.12703 | [PDF](https://arxiv.org/pdf/2602.12703v1)

**作者:** Alessandro Manenti `[一作]` (Università della Svizzera italiana), Krzysztof Choromanski `[通讯]` (Google DeepMind and Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现SWING（Space Walks for Implicit Network Graphs）算法，用于在隐式图（edge weight由节点特征的二元函数定义）上高效近似Graph Random Features（GRFs）的随机游走，从而在不显式构造图的情况下计算图核矩阵乘积。

**💡 创新点**

创新点在于：① 将离散节点游走转化为连续空间中的可微分游走，利用Gumbel‑softmax采样得到近似的随机转移概率；② 通过随机特征（RF）和傅里叶分析把二元权重函数线性化，得到可预计算的向量（A_f,σ、B_f,σ、C_f,σ）实现O(N)时间；③ 采用重要性采样和正随机特征进一步降低方差，显著提高效率；④ 该方法完全不需要图材料化，天然适配GPU/张量加速。

**🔧 技术方法**

技术栈：Gumbel‑softmax采样、随机特征映射（positive/block‑orthogonal/ simplex），重要性采样，傅里叶变换与逆变换，Johnson‑Lindenstrauss投影，温度参数调优，GPU张量化实现。

**📊 数据集**

实验数据集：① 合成3D点云（N=200~10,000）用于核矩阵近似；② Thingi10K 3D 网格用于顶点法向量预测；③ ImageNet 与 Places365 用于Vision Transformer（ViT）下游任务。

**📈 对比分析**

与传统GRFs（CPU/GPU实现）比较，SWING在相同随机游走数量下：FNE误差更低或相当；在速度上实现10倍以上加速；在下游任务中与GRFs保持相同或略优的准确率（例如ViT ImageNet Top‑1 80.34% vs 80.10%）。

**⚠️ 局限性**

局限性：① 对权重函数f的傅里叶逆变换必须可解析或可近似；② 随机特征维度r需要调优，维度过低会导致误差增大；③ 需预先估计温度σ²和Gumbel温度，若设置不当会影响收敛；④ 对极端稀疏或高度非均匀的i-graphs的鲁棒性尚未系统验证；⑤ 目前实现主要关注节点特征在欧式空间中的连续嵌入，对非欧式或离散特征的推广有限。

---

## 67. PMG: Parameterized Motion Generator for Human-like Locomotion Control

**arXiv ID:** 2602.12656 | [PDF](https://arxiv.org/pdf/2602.12656v1)

**作者:** Chenxi Han `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**通讯引用:** 1634 | [OpenAlex ID](https://openalex.org/A5076885280)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Parameterized Motion Generator (PMG) 并在 ZERITH Z1 上实现从少量运动捕捉数据到实时人形机器人控制的完整管线，包括运动生成、地面感知优化、模仿学习与强化学习、仿真到实机参数识别以及 VR 遥控。

**💡 创新点**

创新点在于：① 仅用十秒级运动捕捉数据构建参数化运动模板，实现多方向人类式步态与姿态插值；② 通过地面感知优化解决模板插值导致的脚部滑移；③ 结合模仿学习和 RL，得到既能跟踪高维指令又具备高稳健性的策略；④ 通过黑盒优化实现电机动力学与零点校准，显著缩小仿真到实机误差；⑤ 一站式系统兼容多任务与 VR 远程操控。

**🔧 技术方法**

使用技术包括参数化运动生成与插值、地面感知优化、模仿学习+PPO 强化学习、CMA-ES 黑盒优化进行电机 SysID 与零点校准、域随机化、VR 传感器融合遥控、Isaac Gym 并行仿真。

**📊 数据集**

数据集：少量高精度人类运动捕捉数据（动态步态和静态姿态），经过重定向后得到机器人动态剪辑与静态姿势模板，未使用公开大型数据集。

**📈 对比分析**

对比方法：在仿真中与单一步态模仿基线、PMG 去掉地面优化两种配置对比；实机实验中比较仿真误差与实机误差；遥控任务中统计成功率。性能表现：命令跟踪误差约为基线的 1/4，仿真误差 0.0562，实机误差 0.0414，擦拭任务成功率 95%，搬盒任务成功率 80%。

**⚠️ 局限性**

局限性：高度依赖高质量、周期性步态数据，对非周期性复杂动作（如舞蹈）支持有限；地面感知优化仅约束静态接触，动态一致性仍需进一步强化；数据规模受限，需扩充多样性以提升泛化与鲁棒性。

---

## 68. Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback

**arXiv ID:** 2602.12612 | [PDF](https://arxiv.org/pdf/2602.12612v1)

**作者:** Sein Kim `[一作]` (Korea Advanced Institute of Science and Technology), Chanyoung Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1979 | [OpenAlex ID](https://openalex.org/A5101629749)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 Self‑EvolveRec，一种基于大语言模型（LLM）的自我进化推荐系统框架，通过将用户模拟器与模型诊断工具结合，形成方向性反馈循环，实现对推荐系统整个管道（架构、损失、数据处理等）的开放式程序空间进化。

**💡 创新点**

创新点包括：①引入方向性反馈机制，融合定性用户反馈与定量结构诊断；②提出诊断工具–模型共进化策略，使诊断逻辑随代码演化动态更新；③在传统 NAS 与 LLM 驱动的代码进化之间搭建全新的可解释、可指导的进化流程；④通过联合用户模拟器与诊断工具，使得进化不再受限于单一数值指标，能针对性解决偏差、多样性、短期兴趣等多维失败模式。

**🔧 技术方法**

技术手段：使用 GPT‑5‑mini 进行规划与检索、GPT‑5 进行代码生成；采用检索增强生成（RAG）获取外部学术知识；使用 Agent4Rec、PUB 等 LLM‑驱动的用户模拟器；设计多维模型诊断工具（嵌入崩塌、排名边距等）；对比 NASRec、AutoFIS、AlphaEvolve、DeepEvolve 等基线。

**📊 数据集**

实验数据集：Amazon 的 CDs、Electronics、Office 三个子集以及 MovieLens；均采用 5‑core 过滤后版本。

**📈 对比分析**

评价方法：在 NDCG@5、HR@5 两个标准指标上与 NAS 与 LLM 进化基线对比；同时通过用户满意度指标（View、Satisfy、Depth）和 LLM‑评判的代码质量维度（Creativity、Explicitness、Insight、Personalization）进行多维度评测。实验显示 Self‑EvolveRec 在所有数据集和种子模型上均显著优于 AutoFIS、NASRec、AlphaEvolve、DeepEvolve；在极端初始化（随机、集成）场景下也保持最快、最高的收敛性能。

**⚠️ 局限性**

局限性：1）迭代训练与评估成本高，影响实际部署速度；2）当前采用固定的用户模拟器，模拟的多样性和真实性可进一步提升；3）依赖大语言模型的推理与代码生成，需考虑算力与延迟瓶颈。

---

## 69. GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory

**arXiv ID:** 2602.12316 | [PDF](https://arxiv.org/pdf/2602.12316v1)

**作者:** Pepijn Cobben `[一作]` (ETH Zürich), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个包含2009个高风险多智能体情景的基准，并评估大型语言模型在这些场景中的合作与社会福利表现。

**💡 创新点**

首次将正式博弈论框架与机制设计提示结合，用于揭示LLM在高风险情境中的合作缺陷，并展示干预可显著提升社会福利。

**🔧 技术方法**

采用博弈论评估框架、机制设计提示干预和前沿LLM（如大型语言模型）进行多智能体交互测试。

**📊 数据集**

基于MIT AI Risk Repository中提取的情境，生成了2009个对称2×2博弈的合成数据集。

**📈 对比分析**

通过预定义的指标（如社会福利、合作率）与机制设计前后进行对比，发现基线模型仅有62%实现社会最优，机制干预可提升至82%（提升约18%）。

**⚠️ 局限性**

研究仅覆盖对称2×2博弈，未考虑不对称、序列或多玩家情境；机制设计仅通过提示实现，缺乏对真实世界不确定性和复杂互动的验证。

---

## 70. Human Emotion-Mediated Soft Robotic Arts: Exploring the Intersection of Human Emotions, Soft Robotics and Arts

**arXiv ID:** 2602.13163 | [PDF](https://arxiv.org/pdf/2602.13163v1)

**作者:** Saitarun Nadipineni `[一作]` (Queen Mary University of London), Thilina Dulantha Lalitharatne `[通讯]` (Queen Mary University of London)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5067501094)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

利用EEG的α波实时控制软机器人（软角色与软花），实现基于情绪的艺术展示。

**💡 创新点**

首次将情绪、软体机器人与艺术三者结合，提出通过α波映射软体行为的完整闭环系统。

**🔧 技术方法**

EEG信号采集、频域分析（FFT）、α波功率提取、PWM调速与PID闭环控制、软体机器人设计与驱动。

**📊 数据集**

单一实验者的EEG数据（Unicorn Hybrid Black EEG，采样率250Hz，Oz电极）和软花内部压力数据。

**📈 对比分析**

未进行与现有方法的定量比较，实验结果以α波强度与机器人运动/压力变化的同步性示例呈现，未给出具体性能指标。

**⚠️ 局限性**

样本量仅为1人，缺乏多样性；实时处理延迟、单一生理信号限制情绪识别准确性；缺乏定量评估与大规模验证。

---

## 71. Constrained Assumption-Based Argumentation Frameworks

**arXiv ID:** 2602.13135 | [PDF](https://arxiv.org/pdf/2602.13135v1)

**作者:** Emanuele De Angelis `[一作]` (National Research Council of Italy Institute for Advanced Studies), Francesca Toni `[通讯]` (Imperial College London)

**通讯引用:** 7147 | [OpenAlex ID](https://openalex.org/A5078354590)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了受限假设推理框架（CABA），在传统ABA基础上引入约束变量，使规则可以保持非基地（non‑ground），并定义了非基地论证、攻击以及扩展语义，能够在不进行完整实例化的情况下计算有效的论证扩展。

**💡 创新点**

创新点包括：
• 允许规则中出现约束变量，突破了传统ABA只能使用命题原子的问题；
• 设计了“紧致约束论证”和“最广义约束论证”两种概念，并引入“全攻击”“部分攻击”；
• 在约束理论满足闭包性质时，提出约束拆分和 Argument Splitting 机制，能够将无限的非基地论证集转化为有限、实例不重叠、非重叠的集合；
• 证明 CABA 是对 ABA 的保守推广，并且在 ABA-as-CABA 情况下两者语义完全一致。

**🔧 技术方法**

使用的技术与工具主要有：
• 逻辑编程与约束逻辑编程（CLP）中的约束求解器（如线性整数/有理算术）；
• 论证树构造、子句实例化与变量重命名；
• 约束理论的闭包（否定闭包、存在量化闭包）以及对应的拆分算法；
• 与传统 ABA/AA 框架的映射与等价关系，利用标准的冲突自由、可接受、稳定扩展定义；
• 证明与语义推理的形式化方法（等价关系、归纳证明）。

**📊 数据集**

本工作以理论分析为主，没有使用真实数据集。示例主要基于税务法律情景和简化的规则集，用来说明概念与算法，而非大规模数据实验。

**📈 对比分析**

比较方法：通过将 CABA 转换为对应的 ABA（或 AA）框架，使用标准的冲突自由/可接受/稳定扩展算法进行验证；证明 CABA 的扩展等价于 ABA 的扩展；
性能方面：本文未给出实验或复杂度分析，重点在理论证明和构造算法（Argument Splitting）能够在满足闭包条件下将无限实例集压缩为有限集合，但是否可行、耗时与实现细节尚待实验验证。

**⚠️ 局限性**

局限与未来工作：
• 只处理平面（flat）CABA，未考虑假设可出现在规则头的非平面情况；
• 只定义了冲突自由、可接受、稳定三种扩展，未覆盖完整、首选、基准等其他语义；
• 约束拆分和 Argument Splitting 需要约束理论满足否定闭包和存在量化闭包，某些约束域可能不满足；
• 对于一般 CABA 框架，构造有限、实例不重叠、非重叠集合是否可判定仍是未解决的问题；
• 未给出实现与实验评估，复杂度与实用性未知；
• 未考虑偏好、概率或不确定性等扩展；
• 需要进一步研究与 CLP、ASP‑CLP 等系统的集成与性能对比。

---

## 72. ReBA-Pred-Net: Weakly-Supervised Regional Brain Age Prediction on MRI

**arXiv ID:** 2602.12751 | [PDF](https://arxiv.org/pdf/2602.12751v1)

**作者:** Shuai Shao `[一作]` (University of Science and Technology of China), Jianguo Zhang `[通讯]` (Beijing Tiantan Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出基于Teacher-Student框架的区域脑龄预测网络ReBA-Pred-Net，在缺乏区域标签的情况下实现细粒度脑龄估计。

**💡 创新点**

①利用教师模型生成软区域标签并加入区域校正；②学生引入可学习提示+FiLM实现区域特定输出；③设计健康对照相似度HCS和神经疾病相关性NDC两种间接评估指标。

**🔧 技术方法**

Teacher-Student知识蒸馏、Feature-wise Linear Modulation（FiLM）、3D CNN/Transformer骨干、DeepPrep预处理、MMD统计检验等技术。

**📊 数据集**

17个公共T1 MRI数据集共6530个健康受试者训练，1057个未见健康受试者测试，326例帕金森病，107例阿尔茨海默病。

**📈 对比分析**

与多种3D CNN/Transformer骨干对比，最优3D DenseNet在HCS达到73%、NDC对PD约60%（高于HC/AD的20%），表现优秀。

**⚠️ 局限性**

仍存在区块化错误、早期病变识别不足、需依赖临床先验且无真实区域标签，泛化性和多疾病验证待进一步提升。

---

## 73. Beyond Musical Descriptors: Extracting Preference-Bearing Intent in Music Queries

**arXiv ID:** 2602.12301 | [PDF](https://arxiv.org/pdf/2602.12301v1)

**作者:** Marion Baranes `[一作]` (Deezer Research), Elena V. Epure `[通讯]` (Idiap Research Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 MusicRecoIntent 数据集，并基准评估了多种 LLM 在音乐查询中提取描述符及其意图的性能。

**💡 创新点**

创新点在于为音乐查询细粒度标注正向、负向和参照三类意图，并公开了首个此类语料库和对应评测基准。

**🔧 技术方法**

使用 Ollama 框架调用 Gemma、LLaMA、Mistral、Qwen 等大型语言模型完成实体与意图抽取，并通过精确/部分匹配计算 P/R/F1。

**📊 数据集**

数据集来源于 Reddit 的 2,291 条音乐推荐请求，人工共标注 3,935 个描述符及其对应的意图。

**📈 对比分析**

通过对比不同模型的精确/部分匹配 F1，Gemma 27B（或 Qwen 8B）表现最佳，整体精确率约 0.69、召回率约 0.76。

**⚠️ 局限性**

局限性包括标注边界不一致、负向意图样本稀缺、LLM 对细粒度语义歧义和上下文依赖的把握仍有限。

---

## 74. Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles

**arXiv ID:** 2602.13010 | [PDF](https://arxiv.org/pdf/2602.13010v1)

**作者:** Max Bruninx `[一作]` (Vrije Universiteit Brussel), Jan Helsen `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5053092945)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了利用梯度提升树与三种前沿概率预测方法（conformalised quantile regression、natural gradient boosting、conditional diffusion model）相结合，以及传统工程基准（功率曲线与激活的wake模型）对比，比利时海上风电场进行日预测的性能评估。

**💡 创新点**

创新点在于首次将三种先进概率方法与梯度提升树系统集成，并在全区域四年海上风电数据上进行全面对比；同时采用多源天气预报集成显著提升预测精度。

**🔧 技术方法**

使用技术包括梯度提升树、conformalised quantile regression、natural gradient boosting、Treeffuser 条件扩散模型，以及功率曲线与Niayifar‑Agel wake模型等工程基准。

**📊 数据集**

数据集来源于2021‑2024年间比利时海上风电场的1小时功率记录，配合五种天气预报（ICON‑EU/D2、ECMWF HRES、ARPEGE‑EU、MetOffice Hi‑Res）和ERA‑5重分析。

**📈 对比分析**

通过 MAE 与 CRPS 两指标进行比较，结果显示经过调参的 Treeffuser 在所有风场上 MAE 降低 53% 以上、CRPS 降低 34% 以上；与激活的wake模型相比提升约 33%；使用多源天气预报相较单一预报可进一步提升 23%。

**⚠️ 局限性**

限制主要包括：Treeffuser 在训练与推理阶段计算成本高；natural gradient boosting 在低/高风速区表现差，且仅采用正态分布；未尝试更复杂的分布（如Beta或混合高斯）；未充分考虑多点空间信息与风场运营策略。

---

## 75. Abstractive Red-Teaming of Language Model Character

**arXiv ID:** 2602.12318 | [PDF](https://arxiv.org/pdf/2602.12318v1)

**作者:** Nate Rahn `[一作]` (Anthropic), Erik Jones `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“抽象红队”(abstractive red‑team)方法，利用自然语言类别来主动发现大型语言模型在满足角色规范时可能出现的违规行为。

**💡 创新点**

创新点在于：①从类别层面而非单个查询进行搜索，既能捕捉稀有违规，又保持可在真实部署中出现的通用性；②提出两种高效搜索算法（基于类别级强化学习的CRL和查询‑类别迭代QCI）以及配套的类别/查询生成器和奖励模型；③在多种模型上验证了方法的有效性，并通过过滤器控制恶意程度。

**🔧 技术方法**

核心技术包括：类别生成器和查询生成器的预训练；基于奖励模型的类别评分与过滤；类别级强化学习（REINFORCE+优势估计）与迭代采样（利用强LLM合成类别）；以及针对12条角色规范的奖励与过滤模型。

**📊 数据集**

使用的主要数据集：1.4M公开用户查询（来自WildChat、Anthropic‑HH、UltraFeedback等）；2.从每条查询生成10个属性后抽取多种子类别；3.利用合成数据（基于奖励模型的偏好）训练奖励与过滤模型。

**📈 对比分析**

与基线（随机采样）比较时，CRL和QCI在102400次查询预算下均显著提高类别平均奖励（≈2–3倍），且QCI在查询预算较小的情况下更具样本效率；实验覆盖4种模型（Llama‑3.1‑8B、Gemma3‑12B、Qwen3‑30B、GPT‑4.1‑Mini）及12条角色规范，均保持稳定的性能提升。

**⚠️ 局限性**

局限性：①奖励与过滤模型训练依赖合成数据，可能与真实用户行为不完全一致；②目前仅覆盖预定义的12条规范，未覆盖所有潜在违规；③类别生成器的多样性受训练数据限制，可能漏掉罕见但重要的查询模式；④算法复杂度较高，对计算资源有一定需求。

---

## 76. Solving Qualitative Multi-Objective Stochastic Games

**arXiv ID:** 2602.12927 | [PDF](https://arxiv.org/pdf/2602.12927v1)

**作者:** Moritz Graf `[一作]` (RPTU University Kaiserslautern-Landau and Max Planck Institute for Software Systems), Rupak Majumdar `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 14417 | [OpenAlex ID](https://openalex.org/A5081010207)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究多目标定性随机博弈中玩家胜利条件的确定性与复杂度问题，系统分析不同布尔公式结构下的可决定性和算法复杂度，证明了若查询仅使用合取/析取或仅使用几乎必然/非零属性则游戏确定且为 PSPACE‑完整；若同时使用合取、析取以及几乎必然与非零属性，则游戏不确定，且判定问题 NEXPTIME‑难，并给出指数记忆策略的存在性证明；

**💡 创新点**

首次给出多目标随机博弈确定性与复杂度的完整划分，提供从 DQBF 归约的 NEXPTIME‑下界，并证明对非零安全目标记忆需求不确定的极限；

**🔧 技术方法**

使用随机博弈理论、目标展开(goal unfolding)、非确定性策略自动机、依赖量化布尔公式（DQBF）归约、记忆结构分析与指数记忆策略构造；

**📊 数据集**

本文为理论工作，不依赖具体实验数据集，而是基于抽象博弈模型和合成构造；

**📈 对比分析**

通过理论证明与归约对比，展示 PSPACE 与 NEXPTIME 之间的界定；对 PSPACE‑完整子类给出多项式空间算法，对 NEXPTIME‑硬子类给出指数空间可行性（存在指数记忆策略），但未给出具体实验性能；

**⚠️ 局限性**

对包含非零安全目标（NZ(□)）的查询，记忆结构与复杂度仍未完全确定，存在记忆需求难以控制；此外，论文未给出算法实现与实验验证，理论证明为主要贡献。

---

## 77. Reflection at Design Actualization (RDA) : A Tool and Process For Research Through Game Design

**arXiv ID:** 2602.12887 | [PDF](https://arxiv.org/pdf/2602.12887v1)

**作者:** Prabhav Bhatnagar `[一作]` (Aalto University), Perttu Hämäläinen `[通讯]` (Aalto University)

**通讯引用:** 2635 | [OpenAlex ID](https://openalex.org/A5060951467)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并评估了一款名为 Reflection at Design Actualization（RDA）的工具，用于在游戏设计过程中的“设计实际化”时刻实时记录微观设计反思，并自动录制相应的游戏测试视频。

**💡 创新点**

创新点包括：① 将反思嵌入设计决策的即时时刻，捕捉细粒度的 tacit 知识；② 通过 OBS WebSocket 自动化录屏并与反思日志同步；③ 提供可视化编译脚本，将原始 JSON 与视频合并成易于分析的 PDF/MP4；④ 作为开源插件兼容 Unity 与 Godot，促进跨引擎使用。

**🔧 技术方法**

核心技术：Unity / Godot 插件、OBS Studio + WebSocket 插件、Python 编译脚本、JSON 日志格式、标签化系统、键盘热键触发录制与停止、视频转码与时间戳渲染。

**📊 数据集**

使用数据集：作者在三项独立项目中共记录 186 + 190 + 1110 条反思日志，视频大小约 7.5 GB + 27.2 GB + 16.2 GB，总计约 50 GB，涵盖 3 个月的游戏设计与测试过程。

**📈 对比分析**

与现有工具（如 Project Reflection Tool）对比：RDA 在反思粒度、自动录制和后期编译方面表现更佳，能完整追踪每次测试的过程与思考；在性能上录制流畅但文件体积大，需自行管理视频同步与压缩；总体上提升了数据完整性与可分析性。

**⚠️ 局限性**

局限性：① 仅在单人项目中测试，未验证多人协作场景；② 需要自定义热键导致操作学习成本；③ 视频文件同步困难，需手动搬移；④ 对快速原型循环的适配不足；⑤ 对编码工作量高，改造需编程技能；⑥ 与行业开发流程、版本控制整合尚不成熟。

---

## 78. Continuous Diffusion Models Can Obey Formal Syntax

**arXiv ID:** 2602.12468 | [PDF](https://arxiv.org/pdf/2602.12468v1)

**作者:** Jinwoo Kim `[一作]` (University of California-San Diego), Loris D'Antoni `[通讯]` (University of California-San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free 的引导方法，使连续扩散语言模型能够满足正则表达式约束。

**💡 创新点**

创新点在于通过分析 DFA 的期望接受概率，直接在采样过程中使用梯度引导，而无需训练额外的分类器。

**🔧 技术方法**

技术上结合了扩散模型（DDPM）、正则表达式与 DFA 对齐、动态规划计算期望概率以及梯度引导的采样公式。

**📊 数据集**

使用 180 条正则约束数据集：70 条 JSON Schema 约束（从 JSONSchemaBench 转换而来）以及 110 条合成的自然语言模式。

**📈 对比分析**

与 GPT‑2 的约束解码（GCD）和 PLAID 原生引导方法对比，约束满足率 68–96%，pass@10 高于自回归方法，困惑度与流畅度基本保持甚至略有提升。

**⚠️ 局限性**

主要局限是计算开销大，梯度计算随 DFA 迁移数线性增长；目前仅适用于连续扩散模型，尚未扩展到离散扩散模型。

---

## 79. Channel Gain Map Reconstruction Based on Virtual Scatterer Model

**arXiv ID:** 2602.12602 | [PDF](https://arxiv.org/pdf/2602.12602v1)

**作者:** He Sun `[一作]` (Beihang University), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 104281 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

基于可调虚拟散射体模型，利用递进估计与高斯过程回归从少量功率测量重建三维信道增益图。

**💡 创新点**

创新点在于将散射体的数量、位置和散射响应系数（SRC）全部视为可调参数；利用角度域SRC相关性通过高斯过程回归推断未测角度的SRC；递进增量引入散射体以平衡拟合度与计算复杂度。

**🔧 技术方法**

核心技术包括：虚拟散射体多径信道模型、梯度下降与最小二乘法的迭代参数估计、Gaussian Process Regression (GPR) 用于SRC推断、均方误差（NMSE）评估。

**📊 数据集**

数据集：在真实环境（300 m × 300 m）中通过射线追踪仿真得到30个物理散射体，生成60 × 60网格的地面真值CGM，实验使用20个或更少的测量格点。

**📈 对比分析**

与三种基线方法（KPSM、ISSM、Kriging）在两种测量策略（Type‑I、Type‑II）下对比；结果显示本方法在NMSE上显著优于基线（如在Type‑I下NMSE 0.42 vs 10.96，Type‑II下0.42 vs 7.65），并在测量数量有限时保持更低误差。

**⚠️ 局限性**

局限性：需要已知LOS信息且对物理散射体分布敏感；GPR推断复杂度随SRC数量呈立方增长；仅针对CGM，扩展到更一般的信道映射仍需研究。

---

## 80. BaziQA-Benchmark: Evaluating Symbolic and Temporally Compositional Reasoning in Large Language Models

**arXiv ID:** 2602.12889 | [PDF](https://arxiv.org/pdf/2602.12889v1)

**作者:** Jiangxi Chen `[一作]` (Shanghai Jiao Tong University), Qian Liu `[通讯]` (Shanghai Zhinan Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了基于八字占星的标准化符号推理基准BaziQA-Benchmark。

**💡 创新点**

创新点在于将2021–2025年全球专业八字竞赛的200道多选题与统一的多轮对话及结构化推理协议相结合，提供客观可复现的评估框架。

**🔧 技术方法**

采用多轮文本生成、结构化推理协议（SRP）、大语言模型无监督推理以及统计显著性检验等技术。

**📊 数据集**

使用的数据集为200道专业竞赛题目及其对应的已解析出生图表。

**📈 对比分析**

通过对比DeepSeek、GPT‑5.1、Gemini等模型，宏观平均准确率约36–38%，显著高于25%随机基线，但仍远未达到饱和。

**⚠️ 局限性**

局限性包括对时间组合和符号交互推理的依赖导致准确率受限，结构化提示效果不稳定，缺乏跨文化通用性与进一步提升的方向。

---

## 81. AIWizards at MULTIPRIDE: A Hierarchical Approach to Slur Reclamation Detection

**arXiv ID:** 2602.12818 | [PDF](https://arxiv.org/pdf/2602.12818v1)

**作者:** Luca Tedeschini `[一作]` (Villanova.ai S.P.A), Matteo Fasulo `[通讯]` (Swiss Data Science Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种双编码器架构，结合文本内容与用户简介中的社会身份信息，解决了拉丁美洲/意大利社交媒体中俚语再认问题。

**💡 创新点**

创新点在于：①利用LLM弱监督生成“LGBTQ+身份”代理标签，训练用户编码器；②通过门控机制动态融合文本与用户编码器的表示；③将两阶段训练（基础文本编码器、用户编码器、融合+分类）形成层级化、可扩展的框架。

**🔧 技术方法**

主要技术包括：BERT/SetFit/XLM‑RoBERTa 预训练模型、DeepSeek‑V3.2 LLM 进行弱监督标注、线性探测–微调（LPFT）训练策略、交叉熵损失、门控融合机制。

**📊 数据集**

使用 MultiPRIDE 公开数据集（意大利语 1086 条，西班牙语 876 条），每条数据包含推文文本与用户简介，目标为判定 LGBTQ+ 俚语是否为再认用法。

**📈 对比分析**

与单一 BERT 基线对比：双编码器在开发集上宏 F1 分别为意大利 0.88(±0.04) vs 0.90(±0.03)，西班牙 0.64(±0.02) vs 0.67(±0.04)，差异不显著；在官方测试集保持相近的准确率与召回率，证明引入用户信息不降低性能。

**⚠️ 局限性**

局限性包括：①未能显著提升整体性能，可能受基线已强大限制；②代理标签与真实身份不完全对齐，易引入噪声；③门控融合在极端侮辱语境下可能过度依赖用户编码器导致误判；未来需更精准的代理任务或更细粒度的融合与正则化。

---

## 82. Independence-Number Parameterized Space Complexity for Directed Connectivity Certificate

**arXiv ID:** 2602.12668 | [PDF](https://arxiv.org/pdf/2602.12668v1)

**作者:** Ho-Lin Chen `[一作]` (National Taiwan University), Meng-Tsung Tsai `[通讯]` (Academia Sinica)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5064418445)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在流式和分布式模型中计算有向图的稀疏子图的空间复杂性，该子图能够证明连通性。

**💡 创新点**

提出了一种新的参数化方法，通过输入图的独立数来表征有向图连通性问题的流式复杂性，展示了有向图连通性问题的复杂性连续性。

**🔧 技术方法**

使用了随机化算法，针对插入模型和转闸模型分别设计了p-pass算法。

**📊 数据集**

使用了具有n个节点和独立数α的有向图数据集。

**📈 对比分析**

与现有方法比较，提出的算法在空间复杂性上表现出O(α n)的上界，并且在插入模型和转闸模型中都能达到相应的空间复杂性下界Ω(α n/p)。

**⚠️ 局限性**

限制在于有向图连通性问题的固有难度，尤其是在一般图中，流式算法的空间下界较高。

---

## 83. Bloom Filter Look-Up Tables for Private and Secure Distributed Databases in Web3 (Revised Version)

**arXiv ID:** 2602.13167 | [PDF](https://arxiv.org/pdf/2602.13167v1)

**作者:** Shlomi Dolev `[一作]` (Ben-Gurion University of the Negev), Daniel Shlomo `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于 BFLUT 的去中心化数据库方案，用以在 Web3 环境下安全、私密地管理加密密钥，而不在任何节点上显式存储密钥。

**💡 创新点**

核心创新在于将 Bloom Filter 结构（BFLUT）与 OrbitDB/IPFS/IPNS、CRDT 等技术组合，能够通过哈希映射将密钥编码为位掩码，实现无密钥泄露的隐私保护与冲突无关的并发更新。

**🔧 技术方法**

使用的技术包括 Bloom Filter 与 BFLUT 编码、IPFS（内容寻址存储）、IPNS（持久命名）、OrbitDB（分布式键值数据库）、CRDT（冲突自由复制数据类型）、SHA 哈希函数和阈值加密模型。

**📊 数据集**

实验中未使用公开真实数据集，而是通过生成随机用户名/密码组合与随机密钥进行模拟，构造了若干个密钥条目来验证算法效果。

**📈 对比分析**

通过仿真评估误检概率和文件访问次数，结果表明误检概率可低至 10⁻⁶ 以下，平均文件访问次数约为 47-50 次，验证了系统在大规模密钥存储与检索上的高可用性与低延迟。

**⚠️ 局限性**

局限性包括：误检概率受参数（U、L、F 等）影响，需要细致调优；对大规模密钥长度和文件数量存在存储与计算开销；缺乏在真实 Web3 应用场景下的实测验证。

---

## 84. ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools -- From Consensus Learning to Ambiguity-Driven Emotion Reasoning

**arXiv ID:** 2602.12714 | [PDF](https://arxiv.org/pdf/2602.12714v1)

**作者:** Esther Sun `[一作]` (Carnegie Mellon University), Carlos Busso `[通讯]` (Carnegie Mellon University)

**通讯引用:** 13186 | [OpenAlex ID](https://openalex.org/A5040793194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个基于多轮推理的情感识别框架 ADEPT，该框架将情感识别转化为一个包含候选生成、证据采集与裁决的 agentic 流程，并通过工具调用实现可审计的音频与文本证据链。

**💡 创新点**

创新点包括：①将共识学习转向基于不确定性驱动的情感推理，利用少数投票作为重要信息；②提出结构化的证据探测工具箱（语义探测、声学探测、结构优先、重放等）并在推理中动态调用；③引入 Group Relative Policy Optimization（GRPO）与 Evidence Trust Gate 对工具使用行为进行奖励优化，确保仅在获得可靠证据时才做决策。

**🔧 技术方法**

核心技术：多轮 agentic 推理、Explicit Information Retrieval（EIR）约束、工具箱式采样（Semantic/Acoustic Probing）、GRPO 强化学习、证据信任门控、结构化候选生成与闭环重放。

**📊 数据集**

使用的主要数据集为 MSP‑Podcast 语料（V2.0，含多标注者情感标签），并在 IEMOCAP 上进行零样本迁移评估（映射至 7 维情感集合）。

**📈 对比分析**

与传统 SSL 分类器（HuBERT、wav2vec、WavLM）、生成式 Speech‑LLM/MLLM（SALMONN、Qwen‑2‑Audio、BLSP‑Emo）以及无工具的 Qwen‑3‑Omni 进行比较。ADEPT+GRPO 在 MSP‑Podcast 上实现了 Primary Macro‑F1 从 0.364（WavLM‑finetune）提升至 0.4224，Soft Recall 和 Set Recall 分别提升至 0.7874 与 78.74%，显著优于所有基线；在 IEMOCAP 零样本测试中 Set Recall 达到 54.80%，高于 BLSP‑Emo 的 52.30%。

**⚠️ 局限性**

局限性：①依赖预先设计的工具集合，若工具不完善或缺失可能导致推理质量下降；②多轮推理和工具调用增加计算开销；③对情感词汇表和语义模式有先验假设，难以泛化到完全新颖的情感标签；④证据信任门控的阈值需要手工调优，若调参不当可能出现误判或忽略有效证据。

---

## 85. In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach

**arXiv ID:** 2602.13156 | [PDF](https://arxiv.org/pdf/2602.13156v1)

**作者:** Yiran Gao `[一作]` (City University of Hong Kong), Tao Li `[通讯]` (City University of Hong Kong)

**通讯引用:** 84295 | [OpenAlex ID](https://openalex.org/A5041318946)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种集感知、推理、规划与行动于一体的轻量级大语言模型（LLM）代理，专门用于网络事件响应，利用上下文自适应和基于LLM的世界模型进行MCTS规划，直接将日志和告警映射为响应行动序列。

**💡 创新点**

创新点在于：①将强化学习的部分可观测马尔可夫决策过程（POMDP）规划思想（Monte-Carlo树搜索、误设规划）迁移到LLM内部，②通过多任务（感知、推理、行动）Fine‑Tuning和Chain‑of‑Thought提示，实现对LLM内部世界模型的自我校准，③实现无显式结构化网络模型的端到端响应规划，显著提升响应速度并降低人为建模成本。

**🔧 技术方法**

技术包括：大语言模型Fine‑Tuning（LoRA）、Chain‑of‑Thought提示、基于LLM的观察与状态生成、Monte‑Carlo树搜索与误设规划、上下文自适应（利用前沿LLM或外部威胁情报对攻击策略进行校准）、多步骤候选行动生成与评估。

**📊 数据集**

使用了四个公开安全日志数据集：CTU‑Malware‑2014、CIC‑IDS‑2017、AIT‑IDS‑V2‑2022 以及 CSLE‑IDS‑2024，涵盖 Windows/Linux 系统与多种攻击类型（恶意软件、DoS、网络侦查等）。

**📈 对比分析**

通过将生成的响应序列与前沿LLM（DeepSeek‑R1、Gemini 2.5 Pro、OpenAI O3、GPT‑5.2）以及基准模型对比，评估恢复时间（每步成本统一为 1，失败加大惩罚）。实验结果显示，该LLM代理在所有数据集上平均恢复时间比对比模型快约 23%，并保持较低的失败率。

**⚠️ 局限性**

局限性包括：①计算成本高，MCTS 复杂度为 O(MN)，单个事件平均生成时间约 20 分钟；②在更大规模网络或更复杂攻击场景下需要更深的搜索树，导致生成延时不可接受；③目前评估集中在短行动序列（约 5 步），对长序列和更真实的时间成本评估尚不足；④依赖前沿LLM API 进行攻击策略校准，若无外部接口则需自行实现。

---

## 86. TFT-ACB-XML: Decision-Level Integration of Customized Temporal Fusion Transformer and Attention-BiLSTM with XGBoost Meta-Learner for BTC Price Forecasting

**arXiv ID:** 2602.12380 | [PDF](https://arxiv.org/pdf/2602.12380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. The Rise of AI Agent Communities: Large-Scale Analysis of Discourse and Interaction on Moltbook

**arXiv ID:** 2602.12634 | [PDF](https://arxiv.org/pdf/2602.12634v1)

**作者:** Lingyao Li `[一作]` (University of South Florida), Yongfeng Zhang `[通讯]` (Rutgers University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对2026年1月启动的仅供AI代理交流平台Moltbook的122,438条帖子进行主题建模、情感分析和社交网络分析，系统性揭示了AI代理在真实开放环境下的讨论内容、表达方式与互动结构。

**💡 创新点**

创新点在于首次在大规模、真实且非人为干预的AI代理社区中量化其自我身份认同、技术维护与经济行为，并将情感、语言特征与网络拓扑相结合，揭示代理社群的功能性层级与情感门槛。

**🔧 技术方法**

使用的技术包括BERTopic（结合BERT+UMAP+K-means）进行主题提取，RoBERTa/DistilRoBERTa进行情感与情绪分类，Flesch易读性与TTR指标进行语言复杂度评估，以及NetworkX进行节点中心性、核心-边缘划分与社区检测。

**📊 数据集**

采用了公开API抓取的Moltbook数据集（122,438条帖子、448,238条互动），经语言过滤后保留106,136条英文帖子用于分析。

**📈 对比分析**

通过与人类社交平台（如Reddit）在网络密度、中心性分布与情感分布等指标对比，发现Moltbook呈现极低的回复率与高中心化程度，说明AI代理交流更偏向单向广播而非双向对话。

**⚠️ 局限性**

局限性包括仅观察平台首周数据，无法反映长期演化；主题模型仅针对标题，可能忽视评论内容；未能区分完全自主与受人类干预的代理行为；且未与真实人类用户数据直接对标，难以确认哪些模式是AI固有的。

---

## 88. DiffuRank: Effective Document Reranking with Diffusion Language Models

**arXiv ID:** 2602.12528 | [PDF](https://arxiv.org/pdf/2602.12528v1)

**作者:** Qi Liu `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 23821 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出DiffuRank框架，将扩散语言模型（dLLM）应用于文档重排任务。

**💡 创新点**

创新点在于用扩散模型替代自回归模型，提出点评、logits‑列表式和排列式三种重排策略，并为每种策略设计了对应的训练与推理方法。

**🔧 技术方法**

使用LLaDA‑1.5等扩散语言模型，并结合扩散去噪、蒙版策略、匈牙利算法等技术实现重排。

**📊 数据集**

在TREC DL和BEIR的多种检索数据集上进行评测。

**📈 对比分析**

与自回归LLM的零样本和微调重排器对比，扩散重排器在零样本下与最佳LLM相当，微调后在多数任务上超过同等规模的LLM重排器。

**⚠️ 局限性**

局限性包括需要为不同策略设计复杂的训练/推理流程，扩散模型在长文本或大规模候选时受上下文窗口限制，且推理步骤数需在效率与效果之间权衡。

---

## 89. Gradient-Enhanced Partitioned Gaussian Processes for Real-Time Quadrotor Dynamics Modeling

**arXiv ID:** 2602.12487 | [PDF](https://arxiv.org/pdf/2602.12487v1)

**作者:** Xinhuan Sang `[一作]` (Boston University), Roberto Tron `[通讯]` (Boston University)

**通讯引用:** 3363 | [OpenAlex ID](https://openalex.org/A5031884953)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于梯度信息的分区高斯过程（GP）框架，用以实现小型无人机四旋翼动力学的实时预测；

**💡 创新点**

创新点在于将梯度信息与Schur补数近似结合，同时利用状态空间分区减少在线计算量，从而在保持高精度的同时显著降低推理时间；

**🔧 技术方法**

主要技术包括：高斯过程回归、梯度条件化、Schur补数矩阵分解、基于区块的分区近似、潜流模拟（CHARM）与声学后处理（PSU‑WOPWOP）生成数据；

**📊 数据集**

使用NASA SUI Endurance四旋翼的中等精度潜流模拟数据，包含5000个训练样本（含梯度）以及40000个无梯度样本，用于对比不同GP模型；

**📈 对比分析**

与标准GP、无分区GP、仅分区GP、仅梯度GP等基线模型进行对比；在30 Hz实时运行条件下，所提方法每次预测平均耗时18.48 ms，精度略低于无梯度模型但显著优于仅分区模型，且比标准GP快近一半；

**⚠️ 局限性**

主要局限包括：对噪声（声学）预测性能不佳，原因是缺乏简化噪声模型和相应的梯度信息；在高速风速条件下，力矩预测精度下降，需进一步改进模型和分区策略。

---

## 90. Scaling Web Agent Training through Automatic Data Generation and Fine-grained Evaluation

**arXiv ID:** 2602.12544 | [PDF](https://arxiv.org/pdf/2602.12544v1)

**作者:** Lajanugen Logeswaran `[一作]` (LG AI Research), Honglak Lee `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套可扩展的自动生成与筛选高质量轨迹的管道，并通过细粒度约束评估利用部分成功轨迹训练出24B参数的小型Web代理；

**💡 创新点**

创新点包括基于约束的细粒度评估框架，可利用部分完成的轨迹；利用少量提示的LLM自动生成大量轨迹；以及构建新的BookingArena复杂预订任务基准；

**🔧 技术方法**

使用的技术包括：少量提示LLM轨迹生成、LLM/VLM约束评估、前缀提取与后视重标记、LoRA微调与全微调对比、Mistral 3 Small模型的蒸馏；

**📊 数据集**

使用的数据集为约150k生成轨迹（覆盖1000个热门网站）以及构建的120个复杂预订任务的BookingArena；评估时还使用WebVoyager基准；

**📈 对比分析**

与商业闭源模型（Claude Computer Use、Operator）及开源基准（Browser Use、UI‑TARS）对比，采用任务成功率(SR)和约束满足率(CSR)两项指标，24B模型在BookingArena SR≈60%、CSR≈68%，优于大多数开源方法，仅略逊于Operator；在WebVoyager也实现或超过大模型的性能；

**⚠️ 局限性**

局限性包括：整体SR仍偏低；评估依赖LLM/VLM的准确性和URL/截图信息；不支持多模态观测；对任务更新与时间敏感性的自动化支持有限。

---

## 91. Transporting Task Vectors across Different Architectures without Training

**arXiv ID:** 2602.12952 | [PDF](https://arxiv.org/pdf/2602.12952v1)

**作者:** Filippo Rinaldi `[一作]` (University of Modena and Reggio Emilia), Simone Calderara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 5272 | [OpenAlex ID](https://openalex.org/A5075481810)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的任务向量传输方法Theseus，能在不同宽度的预训练模型之间直接迁移任务特定的参数更新，避免重新训练；

**💡 创新点**

创新点在于将任务更新定义为对中间表示的功能影响，并通过正交Procrustes对齐实现跨模型的闭式传输，保留更新几何结构；

**🔧 技术方法**

技术主要包括功能匹配目标、正交Procrustes矩阵求解、闭式传输公式τ_B = T_out τ_A T_in^⊤以及对齐与扩充策略；

**📊 数据集**

实验数据集涵盖视觉任务（CLIP ViT-B/16, ViT-B/16+, 8-Vision benchmark）和语言任务（GLUE NLI子任务、encoder‑decoder Transformer），以及不同预训练分布；

**📈 对比分析**

与零样本、全微调、伪逆、随机对齐等基线对比，Theseus在宽度、预训练分布和深度差异场景下均能显著提升性能，尤其在低样本情况下稳定优于伪逆，且可作为加速微调的warm‑start；

**⚠️ 局限性**

局限包括对深度差异的处理仍为简单插值，且在极端宽度/深度不匹配或极少激活样本时对齐精度有限，且仅保留共享子空间，可能忽略目标模型独有特征。

---

## 92. Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models

**arXiv ID:** 2602.12996 | [PDF](https://arxiv.org/pdf/2602.12996v1)

**作者:** Hao Chen `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8587 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于元认知的知识增强框架，通过区分已掌握、困惑和缺失的知识区域实现有针对性的知识扩展，并通过认知一致性机制校准模型内部置信度与实际准确率的匹配，从而实现更可靠的知识补全和表达

**💡 创新点**

创新点在于：①将知识空间划分为 mastered、confused、missing 三种区域，利用内部认知信号实现动态知识分配；②引入认知一致性对齐机制，让模型在正确推理时强化置信度，在错误推理时降低置信度；③采用 Group Relative Policy Optimization (GRPO) 结合自适应校准，实现多轮“扩展‑校准”迭代，显著提升性能；④构建统一的监督微调与强化学习目标，兼顾知识增量与置信度校准

**🔧 技术方法**

主要技术包括：元认知监测与控制框架；内部不确定度量（基于负对数似然与熵）；Cognitive-Guided Knowledge Expansion (CGKE)；Cognitive-Driven Knowledge Calibration (CDKC)；Group Relative Policy Optimization (GRPO)；LoRA、TRL、verL 等参数高效微调框架；检索增强生成（UltraRAG）与预训练模型 Qwen2.5‑7B‑Instruct、Llama‑3.1‑8B‑Instruct

**📊 数据集**

使用多种问答基准数据集：PopQA、Musique、SQuAD、NQ、HotpotQA、2WikiMQA、BeerQA、WebQuestions、Bamboogle、SearchQA、TriviaQA，涵盖弱、中、强知识 grounding 场景

**📈 对比分析**

与多类基线对比：Vanilla LLM、Chain-of-Thought、Retrieval-Augmented Generation、标准 SFT、LLKD‑SFT、Know What、CRew‑DPO、BARREL、GRPO 等。实验表明：在 11 个 QA 任务上，CGKE + CDKC 方案平均提升 1.06%（Qwen2.5‑7B）至 1.73%（Llama‑3.1‑8B）相对最强基线；二轮迭代进一步提升 3.32%–4.06%，在复杂任务上显著显著提升

**⚠️ 局限性**

局限性包括：①框架主要验证于问答任务，泛化至其他任务需进一步验证；②元认知信号和不确定度估计仍可能受模型内部分布偏差影响；③需要多轮训练和多阶段微调，计算成本相对较高；④对检索模块的依赖导致对检索质量敏感，检索错误会影响知识扩展效果

---

## 93. An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking

**arXiv ID:** 2602.12354 | [PDF](https://arxiv.org/pdf/2602.12354v1)

**作者:** Lars Hertel `[一作]` (LinkedIn Inc), Souvik Ghosh `[通讯]` (LinkedIn Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并部署了一个基于Transformer的序列推荐模型，用于 LinkedIn Feed 的内容排序，取代了原有的 DCNv2 排序器，实现大规模实时推理与在线性能提升。

**💡 创新点**

创新点包括使用 RoPE 位置编码与 scalar skip 结构、交错的帖子-动作序列、Late‑Fusion 方案降低维度、MMoE 多任务头、LLM 生成的成员嵌入、逆倾向权重与时间位置加权、共享上下文批量化与自定义 FlashAttention 等多项技术，兼顾准确性与生产效率。

**🔧 技术方法**

核心技术包括 Transformer 解码器（Pre‑LN+RoPE）、MMoE 头、HSTU 对比、逆倾向加权、时间/位置加权、增量训练、分布式 CPU/GPU 推理、FlashAttention 自定义核、CUDA AUC 计算核等。

**📊 数据集**

使用 LinkedIn 的 1.2B 用户、数十亿条帖子、过去一年交互记录（最多 1000 条）作为训练与评估数据，包含数值特征、内容嵌入、ID 嵌入和分类特征。

**📈 对比分析**

通过离线 AUC（Long Dwell、Contribution）与原生产模型对比，获得若干百分点提升；在线 A/B 测试显示时间消耗提升 +2.10%，在不同活跃度成员中均有正面效果；相较于 LLM‑Ranker 与 TransAct，本文模型在准确性与推理成本上均占优。

**⚠️ 局限性**

仍依赖手工特征工程与离线特征处理，模型规模与推理时延受限；增量训练对低频用户影响有限，在极端稀疏历史或新用户场景下性能可能受限。

---

## 94. Matching of SAR and optical images based on transformation to shared modality

**arXiv ID:** 2602.12515 | [PDF](https://arxiv.org/pdf/2602.12515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 95. CF-HFC:Calibrated Federated based Hardware-aware Fuzzy Clustering for Intrusion Detection in Heterogeneous IoTs

**arXiv ID:** 2602.12557 | [PDF](https://arxiv.org/pdf/2602.12557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 96. FlashSchNet: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics

**arXiv ID:** 2602.13140 | [PDF](https://arxiv.org/pdf/2602.13140v1)

**作者:** Pingzhi Li `[一作]` (University of North Carolina Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现了 IO-aware 的 SchNet 风格 GNN 分子动力学框架，显著提升速度并降低内存占用。

**💡 创新点**

创新点在于四种 IO-aware 技术：Flash radial basis、Flash message passing、Flash aggregation 与通道级 16 位量化，统一解决内存拷贝、原子争用与权重传输瓶颈。

**🔧 技术方法**

使用 GPU kernel fusion、CSR 分段归约、W16A16 混合精度、动态邻居表重建以及寄存器/共享内存重用等技术。

**📊 数据集**

数据集为 5 个快速折叠蛋白：Chignolin、TRPcage、Homeodomain、Villin 与 Alpha3D。

**📈 对比分析**

与 CGSchNet、MARTINI 与全原子 MD 比较，获得 6.5× 速度提升、80% 内存缩减，单 RTX‑PRO‑6000 上 64 副本可实现 1000 ns/天，性能已超过 MARTINI。

**⚠️ 局限性**

局限性在于仍依赖 GPU 计算，面对更大系统或更复杂化学环境时仍需进一步优化；动态拓扑重建与量化误差在极端条件下可能影响精度。

---

## 97. Block-Sample MAC-Bayes Generalization Bounds

**arXiv ID:** 2602.12605 | [PDF](https://arxiv.org/pdf/2602.12605v1)

**作者:** Matthias Frey `[一作]` (University of Melbourne), Michael C. Gastpar `[通讯]` (École polytechnique fédérale de Lausanne)

**通讯引用:** 12817 | [OpenAlex ID](https://openalex.org/A5063528341)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出并证明了一类新的区块样本MAC‑Bayes（均值近似正确）泛化误差界，能够在传统PAC‑Bayes无效的情形下给出有限界。

**💡 创新点**

创新点在于通过将训练样本划分为多个块并使用块内条件后验分布的KL散度，显著改进了期望误差界的收敛速率，并给出了一般化的闭式界。

**🔧 技术方法**

技术上主要利用了Donsker‑Varadhan变分表示、Jensen不等式和子高斯/有界损失的矩生成函数上界，形成了通用的区块样本MAC‑Bayes理论框架。

**📊 数据集**

实验使用了一个简易的高斯均值估计例子，损失为截断二次损失，训练样本分块后对比原PAC‑Bayes界。

**📈 对比分析**

与传统PAC‑Bayes界相比，该区块样本MAC‑Bayes界在该例子中从无穷大变为有限，并且随着块大小的合适选择，收敛速度可达到O(n⁻¹/²)，数值验证显示性能明显优于已知的期望和高概率界。

**⚠️ 局限性**

主要限制是界依赖于数据分布和后验的KL散度，若缺乏对分布的先验信息难以估计；此外理论上证明了不存在相同形式的高概率PAC‑Bayes界，进一步限制了其在实际高概率风险评估中的应用。

---

## 98. AMPS: Adaptive Modality Preference Steering via Functional Entropy

**arXiv ID:** 2602.12533 | [PDF](https://arxiv.org/pdf/2602.12533v1)

**作者:** Zihan Huang `[一作]` (University of California, San Diego), Junda Wu `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了实例感知的模态偏好调节方法AMPS，并引入模态贡献比（MCR）诊断指标，用以评估并自适应调整多模态大型语言模型的模态偏好；

**💡 创新点**

创新点在于：①利用功能熵和高斯扰动计算模态贡献比，精确量化每个样本的模态敏感性；②设计实例级调节因子，使调节强度随样本敏感性自动缩放，避免统一强度导致的生成崩溃；

**🔧 技术方法**

主要技术包括：功能熵理论、Log‑Sobolev不等式、KV缓存扰动梯度分析、轻量级两层MLP学习调节向量，以及对多模态Transformer层的隐藏状态进行增量调节；

**📊 数据集**

在多模态冲突基准MC²（包含TDIUC和MS‑COCO的图文冲突样本）上进行实验，并在Qwen‑2.5VL（3B、7B）和LLaVA1.5（7B、13B）模型上评估；

**📈 对比分析**

与提示工程、静态调节以及学习调节（L2S）等基线对比，AMPS在实现视觉/文本偏好转移时能够在更宽阔的调节强度范围内保持生成质量，整体偏好转移成功率显著提升，且生成错误率更低；

**⚠️ 局限性**

局限性包括：需额外的高斯扰动与梯度计算导致推理时有一定时间开销；仅在当前两种模型和MC²任务上验证，尚未评估对更复杂多模态任务或其他模型的泛化能力；

---

## 99. Zero-Shot Adaptation to Robot Structural Damage via Natural Language-Informed Kinodynamics Modeling

**arXiv ID:** 2602.12385 | [PDF](https://arxiv.org/pdf/2602.12385v1)

**作者:** Anuj Pokhrel `[一作]` (George Mason University), Xuesu Xiao `[通讯]` (George Mason University)

**通讯引用:** 1936 | [OpenAlex ID](https://openalex.org/A5017662025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于自然语言的零样本运动动力学建模框架，能够在结构损伤出现后立即根据损伤描述调整运动模型，避免在线数据收集与重训练。

**💡 创新点**

创新点在于将损伤的自然语言描述映射到共享语义-动力学嵌入空间（通过Sentence Transformer + VICReg对齐），并在Transformer编码器-解码器中将该嵌入注入全局上下文，实现零样本适配。

**🔧 技术方法**

核心技术包括句子Transformer（EmbeddingGemma）、VICReg自监督对齐、Transformer Encoder‑Decoder（双阶段时空注意力）以及BeamNG软体仿真器和1/10尺寸V4W实机平台的实验验证。

**📊 数据集**

使用BeamNG高保真仿真生成的150k条数据，涵盖六类损伤（含单一和组合损伤），并收集30k条未见测试数据；在真实1/10规模V4W机器人上做额外验证。

**📈 对比分析**

与无损伤模型、传统自适应模型（Fine‑tune 20s/5m/10m）以及单一状态Token的基线进行对比，零样本模型在所有损伤类别下均实现最低MSE，整体误差下降81%，甚至超过10分钟Fine‑tune的性能。

**⚠️ 局限性**

局限性包括：对语言描述的依赖（若描述不准确模型会失效）、主要针对结构损伤而未考虑复杂地形或多模态输入、在物理平台上仍有相对误差提升，且在极端损伤或大规模物理差异下效果待进一步验证。

---

## 100. The Fuzzy Front Ends: Reflections on the Never-Ending Story of Visualization Co-Design

**arXiv ID:** 2602.13182 | [PDF](https://arxiv.org/pdf/2602.13182v1)

**作者:** Wei Wei `[一作]` (University of Victoria), Sheelagh Carpendale `[通讯]` (Simon Fraser University)

**通讯引用:** 11751 | [OpenAlex ID](https://openalex.org/A5012561411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对当地艺术社区进行为期两年半的可视化共创实践，组织工作坊、反馈会、原型设计、公开演示、问卷调查和专家访谈，并以漫画式叙事方式呈现过程。

**💡 创新点**

系统阐释了可视化共创中出现的三类“模糊前端”（数据采集、数据表现、交互设计），强调其多次迭代、交织互相影响的特点，并将其视为共创机会而非障碍。

**🔧 技术方法**

采用自我民族志方法记录过程，使用Tableau进行快速原型，开发自定义“数据画家”工具，开展工作坊和半结构化访谈。

**📊 数据集**

艺术经费分配数据及相关经济数据（包括未获资助的艺术家和机构的缺失信息），并通过问卷收集补充数据。

**📈 对比分析**

本研究不涉及算法性能比较，仅通过定性案例分析和共创会议记录来评估方法有效性，未进行量化对比。

**⚠️ 局限性**

研究周期长、过程模糊持续，跨学科协作需要时间建立共识，样本量有限，研究结果可能不易推广到其他社区或领域。

---

## 101. Out-of-Order Membership to Regular Languages

**arXiv ID:** 2602.13100 | [PDF](https://arxiv.org/pdf/2602.13100v1)

**作者:** Antoine Amarilli `[一作]` (University of Lille), Charles Paperman `[通讯]` (University of Lille)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出并研究“乱序成员资格（out‑of‑order membership）”与“乱序评估（out‑of‑order evaluation）”问题，并给出对正规语言、幺半群和半群的空间复杂度分类，证明了三分法则与常数/对数/线性空间上限与下限，并给出若干特殊语言的子线性空间算法（如 O(√n)）。

**💡 创新点**

创新点包括：①首次把乱序成员资格定义为对正规语言的理论问题；②利用代数范畴（族）与同余理论将空间复杂度与代数结构关联，得到对幺半群的三分法则；③为半群给出常数空间可行性类 LiC，并展示对其它半群仅有对数空间下限；④构造了一系列“fooling set”证明无条件下界；⑤在非线性层面提出了 O(log n) 和 O(√n) 的特例算法。

**🔧 技术方法**

主要技术手段包括：代数同余与范畴闭包理论、语法树与状态机的组合处理、Fooling Set 证明法、先前的同步通信复杂度与同余类 RED 的借鉴、以及基于首尾位置的散列（k‑first‑last subword）推导的算法。

**📊 数据集**

由于是理论复杂度分析，本文未使用实际数据集，而是对所有固定正规语言、固定幺半群或半群进行抽象讨论。

**📈 对比分析**

方法对比主要体现在对空间下限与上限的匹配：对于任何给定的固定语言或代数结构，作者给出既满足上限又满足下限的紧确结果；对某些语言如 a^*b^*a^*，作者提供了 O(log n) 的上限，同时证明了相同下限，从而实现精确的空间复杂度表征；对 a^*b^*a^*b^*a^*b^* 则给出了 O(√n) 的上限与线性下限，表明该问题空间复杂度更为丰富。

**⚠️ 局限性**

局限性在于：
- 对半群的空间复杂度未完成全局分类，只有常数空间与对数空间两类的边界已知；
- 对更高阶空间层次（如 Θ(n^α) 的 α∈(0,1)）缺乏完整理论框架；
- 仅在理论层面给出复杂度，而未验证在实际系统中的实现效率；
- 乱序成员资格问题在多处理器或分布式环境下的通信成本未作深入探讨。

---

## 102. Synthetic Craquelure Generation for Unsupervised Painting Restoration

**arXiv ID:** 2602.12742 | [PDF](https://arxiv.org/pdf/2602.12742v1)

**作者:** Jana Cuch-Guillén `[一作]` (Universitat de Barcelona), Raül Pérez-Gonzalo `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5092346600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全无标注的油画裂纹检测与数字修复框架，结合形态学顶帽检测、LoRA 微调的 SegFormer、检测器引导的混合损失与 logits 调整，并在得到的裂纹掩膜上进行 MTM 与 Anisotropic Diffusion 填补。

**💡 创新点**

创新点包括：①使用 Bézier 轨迹生成逼真的分支与锥形裂纹的领域特定合成数据；②将形态学检测结果注入网络输入和 logits，形成检测器引导的学习；③采用 LoRA 高效微调，只训练少量参数；④将经典检测与深度学习、以及后处理填补整合为统一的无监督流水线。

**🔧 技术方法**

技术手段包括：形态学顶帽变换与尺寸过滤、SegFormer MiT-B0 + LoRA、检测器引导的四通道输入、混合加权交叉熵 + Dice 损失、检测器引导的 logits 调整、MTM 滤波、Anisotropic Diffusion（AD）填补。

**📊 数据集**

数据集为基于 WikiArt 画作的合成裂纹数据（每幅图生成 80‑150 条 Bézier 曲线，模拟分支与锥形）以及手工标注的四幅真实裂纹图像，整体无像素级真实标注。

**📈 对比分析**

通过与 Grassfire 交互式传统方法、Wan 等摄影修复模型、Pik‑Fix、SAM2 等零样本模型对比，实验显示本文方法在检测准确率、F1、SSIM 等指标上显著提升（例如 F1 61.36，SSIM 64.87），优于传统方法和现有摄影修复模型。

**⚠️ 局限性**

局限性在于合成裂纹与真实裂纹的差异仍可能导致误检，特别是极细纹理或多层修饰；在复杂纹理、光照变化下误检率略升；填补阶段对宽裂纹的效果有限；需进一步验证在更多艺术品、多模态条件下的泛化。

---

## 103. Towards reconstructing experimental sparse-view X-ray CT data with diffusion models

**arXiv ID:** 2602.12755 | [PDF](https://arxiv.org/pdf/2602.12755v1)

**作者:** Nelas J. Thomsen `[一作]` (Martin-Luther-University Halle-Wittenberg), Ezgi Demircan-Tureyen `[通讯]` (Centrum Wiskunde and Informatica)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究在实验稀疏视角X射线CT重建中使用扩散模型先验的效果与挑战

**💡 创新点**

系统评估训练分布与前向模型不匹配对扩散先验的影响，并提出退火似然权重调度的DDS推理策略来缓解伪影与结构失真

**🔧 技术方法**

采用扩散先验、Decomposed Diffusion Sampling（DDS）、CG迭代求解、噪声退火似然调度以及传统CGLS基线对比

**📊 数据集**

构造三种Shepp-Logan数据集（标准、实验、混合）以及对应的物理实验phantom扫描数据

**📈 对比分析**

在不同投影数和分辨率下通过PSNR/SSIM与CGLS基线比较，发现多样化先验与退火调度在实验数据上显著提升重建质量并降低伪影，单一先验在实验中易失效

**⚠️ 局限性**

仍受前向模型误差（散射、光束硬化、几何失配）影响，需在更大样本和更高分辨率下进一步验证，且算法计算量仍较大

---

## 104. Composable Model-Free RL for Navigation with Input-Affine Systems

**arXiv ID:** 2602.12492 | [PDF](https://arxiv.org/pdf/2602.12492v1)

**作者:** Xinhuan Sang `[一作]` (Boston University), Roberto Tron `[通讯]` (Boston University)

**通讯引用:** 3363 | [OpenAlex ID](https://openalex.org/A5031884953)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个可组合的、无模型的强化学习框架，用于输入仿射系统的连续时间导航，实现碰撞规避和目标导向行为。

**💡 创新点**

将输入仿射系统的HJB残差与优势函数的二次形式相结合，设计无模型的Actor‑Critic学习，并通过学习的价值函数和最优策略在线用QCQP实现可组合的CLF‑CBF约束，形成了新颖的学习-控制闭环。

**🔧 技术方法**

使用连续时间HJB推导、优势函数二次表达、Gaussian Process回归作为函数近似、Actor‑Critic梯度更新以及QCQP优化等技术。

**📊 数据集**

在二维平面仿真环境中学习和测试，包括矩形、随机多边形以及移动车辆等障碍和目标；未使用真实实验数据集，而是基于模拟生成的数据。

**📈 对比分析**

与基于PPO的离散时间强化学习基线进行对比，两者在价值函数结构和最优控制场上相似，部署后均能在多车交叉路口场景中实现安全通行，性能基本相当。

**⚠️ 局限性**

主要限制在于近似误差对闭环安全的影响、对高维相对状态的可扩展性有限，以及未充分利用高斯过程的不确定性估计进行风险感知或机会约束。

---

## 105. Exploring a New Competency Modeling Process with Large Language Models

**arXiv ID:** 2602.13084 | [PDF](https://arxiv.org/pdf/2602.13084v1)

**作者:** Silin Du `[一作]` (Tsinghua University), Raymond Jia Wang `[通讯]` (Bill-JC Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大语言模型的全流程能力模型构建方法CoLLM，自动从行为事件访谈文本中提取行为与心理描述并映射到能力库，学习权重并进行离线评估。

**💡 创新点**

创新点在于：① 将整个传统专家驱动流程拆解并全程自动化；② 通过可学习的权重动态平衡行为与心理信息；③ 采用离线评估框架实现无额外数据的模型选择与验证。

**🔧 技术方法**

使用的大语言模型技术包括：大型LLM（如Qwen2.5-Max、GLM-4-plus、Doubao-1.5-pro）进行提示式抽取；文本嵌入与余弦相似度进行映射；三温度策略与LLM审核提升抽取鲁棒性；AdamW梯度优化学习权重α。

**📊 数据集**

数据集为一家软件外包公司40名团队领导的行为事件访谈文本，分18名高绩效、22名中等绩效；采用Lominger 67级别能力库及Korn Ferry 38项能力库做对照。

**📈 对比分析**

比较方法：四折交叉验证+离线Spearman相关与AUC评估；与传统专家编码、不同LLM后端、不同关键能力数量及不同能力库进行对比。性能方面：在主库Q=9时AUC≈0.715，Spearman≈0.35；跨库与不同LLM鲁棒性好；去权重学习会显著下降；LLM抽取与专家结果高度一致。

**⚠️ 局限性**

局限性包括：① 依赖预定义的能力库，仍需人工确定目标库；② 评估仅基于横截面绩效差异，缺乏因果验证；③ 对长文本的抽取仍受LLM上下文窗口限制；④ 仅在软件外包行业小样本验证，缺乏行业与文化普适性。

---

## 106. Flow Matching from Viewpoint of Proximal Operators

**arXiv ID:** 2602.12683 | [PDF](https://arxiv.org/pdf/2602.12683v1)

**作者:** Kenji Fukumizu `[一作]` (Institute of Statistical Mathematics), Nisha Chandramoothy `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了将 OT‑Conditional Flow Matching（OT‑CFM）重新表述为基于 Aleksandrov–Brenier 潜能的精确欧氏近端（proximal）形式，证明了在目标分布支持低维流形且不一定具备密度时，该方法仍能得到唯一的反向映射与向量场。

**💡 创新点**

创新点包括：① 将 OT‑CFM 的向量场完全用凸分析中的子梯度与近端算子描述；② 在不假设目标分布可导的前提下得到精确的去噪（denoiser）解释；③ 通过二阶极限导数（second epi‑derivatives）证明在流形假设下，终端 Lyapunov 指数在法向方向为负、切向方向为零，从而保证了对流形结构的保持；④ 论证了有限批次 OT‑CFM 的近端算子与向量场随批量增大收敛到理论上由总体 OT 给出的形式。

**🔧 技术方法**

主要技术包括：凸分析（子梯度、近端算子、Moreau 封包）、Optimal Transport（quadratic 代价的极小化）、Aleksandrov–Brenier 潜能、二阶极限导数理论、Lyapunov 指数分析、欧氏近端去噪视角，以及数值验证中使用的匈牙利算法来实现最优配对。

**📊 数据集**

实验数据集主要有：① 低维模拟数据（Circle 与 Two‑Moons），② MNIST 手写数字图像。训练时采用标准的 minibatch OT‑CFM（批量大小 256–512）以及 MLP 近似向量场。

**📈 对比分析**

与传统方法（如经典流匹配、扩散模型）比较的指标是向量场的雅可比矩阵特征值分布。结果显示，在靠近终端时间点时，法向特征值接近预测的负值（≈‑1），切向特征值趋近零，验证了理论预测。实验中还通过对特征向量的扰动展示了模型在法向方向上产生的衰减与切向方向保持，表明对流形结构的稳健性。

**⚠️ 局限性**

限制：① 理论证明依赖于潜能在流形点处的二阶可微性与子梯度的完整性，实际数据可能出现边界或非光滑情况导致假设失效；② 分析以总体 OT 为基础，有限批次 OT 的收敛速度与梯度方差尚未给出明确的上界；③ 仅针对二次成本的 OT，其他成本或更复杂的目标分布需进一步研究；④ 目前未对高维真实数据（如图像、文本）在大批量规模下的数值稳定性与计算成本进行深入探讨。

---

## 107. Grandes Modelos de Linguagem Multimodais (MLLMs): Da Teoria à Prática

**arXiv ID:** 2602.12302 | [PDF](https://arxiv.org/pdf/2602.12302v1)

**作者:** Neemias da Silva `[一作]` (Universidade Tecnológica Federal de Paraná), Thiago H Silva `[通讯]` (University of Toronto)

**通讯引用:** 1750 | [OpenAlex ID](https://openalex.org/A5072023060)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统梳理了多模态大型语言模型（MLLMs）的核心原理、代表性模型与实践方法，并通过预处理、prompt 工程、LangChain 与 LangGraph 的技术细节展示如何构建完整的多模态流水线。

**💡 创新点**

创新点在于将多模态对齐、融合策略（早期、晚期、混合）与端到端管道设计结合，提出了面向实际应用的实用工具链（LangChain/LangGraph）以及针对情感分析、VQA 等任务的两阶段分类架构。

**🔧 技术方法**

所使用的技术包括 Transformer、CLIP、ViT、跨模态注意力、BLIP‑2、CLAP 等预训练模型，以及 prompt 设计、LangChain、LangGraph 框架实现多模态推理与管道编排。

**📊 数据集**

实验和案例采用公开数据集如 Visual Genome、PerceptSent、ImageNet 等，用于多模态检索、图像说明、VQA 与情感分析等任务。

**📈 对比分析**

通过在 GPT‑4V、LLaVA、PaLM‑E 等模型上比较不同融合策略，结果显示混合融合在 VQA 与 caption 性能上优于单一策略，情感分类两阶段模型也显著提升精度，但仍受到幻觉与偏差的影响。

**⚠️ 局限性**

主要局限包括高昂的训练与推理成本、跨模态对齐难度、模型幻觉与偏见、能耗问题，以及缺乏统一的评价基准和可解释性不足。

---

## 108. TENORAN: Automating Fine-grained Energy Efficiency Profiling in Open RAN Systems

**arXiv ID:** 2602.13085 | [PDF](https://arxiv.org/pdf/2602.13085v1)

**作者:** Ravis Shirkhani `[一作]` (Institute for the Wireless Internet of Things), Salvatore D'Oro `[通讯]` (Institute for the Wireless Internet of Things)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个自动化框架，用于在OpenRAN系统中执行端到端能耗与性能的测量与分析。

**💡 创新点**

将多源能耗测量工具（PDU、Kepler、Yocto-Watt）与OpenShift+Tekton自动化管道结合，实现测试规范化、部署自动化及同步数据采集，填补了O‑RAN能耗测量的空白。

**🔧 技术方法**

使用OpenShift容器化平台、Tekton/ArgoCD自动化管道、Kepler容器能耗估算、Raritan PDU服务器级能耗、Yocto‑Watt RU级测量、JSON实验规范、Grafana/InfluxDB可视化等技术。

**📊 数据集**

实验数据包括iPerf UDP/TCP 10‑70 Mbps流量负载、不同用户数UE场景、两套核心网（Open5GS+商用）、两套RAN栈（oai、srsRAN）、商业RU与USRP等。

**📈 对比分析**

通过对比不同RAN栈、核心网、xApp实现以及用户数对能耗和能效的影响，发现oai随负载线性升功耗，srsRAN固定≈48 W；核心网UPF功耗随负载线性增至≈5 W；xApp功耗随用户增大；整体能效随用户增多提高。

**⚠️ 局限性**

受限于测量工具时序精度（PDU 1 s、Yocto‑Watt 60 ms）、分布式RU统一监测难度，框架未覆盖所有软硬件组合，且实验环境单一，缺乏大规模多租户验证。

---

## 109. WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning

**arXiv ID:** 2602.12852 | [PDF](https://arxiv.org/pdf/2602.12852v1)

**作者:** Junjie Wang `[一作]` (Ant Group), Jinjie Gu `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将 Web Agent 的轨迹转化为状态图并进行最小必要 DAG 剪枝，训练出更高效的搜索模型；

**💡 创新点**

创新点在于①用图结构精确识别并剔除冗余推理步骤；②提出 F‑AE 分数平衡准确率与工具调用效率；③将剪枝后的轨迹用于混合训练提升效率与准确度；

**🔧 技术方法**

使用 LLM 进行状态图抽取与思维重写，最小 DAG 剪枝，语义一致性重写，基于效率或混合的 Agent 进化策略；

**📊 数据集**

实验基准包括 xbench‑deepsearch、Browsecomp、GAIA、HLE；训练数据来自 WebShaper、WebDancer、WebExplorer、TaskCraft、Voyager 等公开 QA 集；

**📈 对比分析**

与 Tongyi‑DeepResearch、开源对手、Prompt Control、Coarse Prune 等对比；WebClipper(Eff) 在保持或提升准确率的同时，平均减少 20% 工具调用回合、19% token；Hybrid 进一步提升准确率 4.8% 且仅少量降低回合数；在所有基准上获得最高 F‑AE 分数；

**⚠️ 局限性**

局限在于只能改进已有模型的冗余，无法挖掘全新搜索策略；仅在搜索/浏览/代码工具上验证，未评估多模态、数据库等新工具的泛化。

---

## 110. Tight Bounds for Logistic Regression with Large Stepsize Gradient Descent in Low Dimension

**arXiv ID:** 2602.12471 | [PDF](https://arxiv.org/pdf/2602.12471v1)

**作者:** Michael Crawshaw `[一作]` (George Mason University), Mingrui Liu `[通讯]` (George Mason University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5101839271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对二分类中使用梯度下降（GD）在大步长下训练线性模型的优化过程进行细致分析，给出了从不稳定阶段过渡到稳定阶段的时间上界，并据此证明了在两维情形下的收敛率为 𝒪(1/(ηT))，远优于以往的 𝒪(1/(γ⁴T²))。

**💡 创新点**

创新点在于：①引入“振荡”概念，证明当 GD 轨迹跨越子水平集时，投影 ŵ_t 指向最大间隔方向的量呈指数增长；②利用该指数增长进一步得到与步长无关的过渡时间上界；③给出匹配的下界，证明上界在 log 因子内紧密。

**🔧 技术方法**

使用技术包括：精细的动态轨迹分析、子水平集几何分割、振荡频率定理、梯度势函数 G(t) 的递推界、线性收敛性证明与 Lipschitz/光滑性结合的传统梯度下降分析。

**📊 数据集**

实验使用了两种数据集：①随机生成的 2 维线性可分数据（满足最大间隔 γ 的假设）；② MNIST 二分类子集（对比验证振荡行为和过渡时间是否随步长增长）。

**📈 对比分析**

与之前的加速 1/T² 率以及自适应步长方法相比，本工作在两维下给出 𝒪(n/γ + log(1/γ)/γ²) 的迭代复杂度，且不受 η 限制，说明在大步长下仍能快速进入稳定阶段并实现快速收敛。

**⚠️ 局限性**

局限性：证明仅适用于 d=2 的情形；对高维情况的推广尚未完成；上界与下界在 log 因子上仍有差距；实验仅在小规模合成/MNIST 数据上验证，未覆盖更复杂真实数据。

---

## 111. Favia: Forensic Agent for Vulnerability-fix Identification and Analysis

**arXiv ID:** 2602.12500 | [PDF](https://arxiv.org/pdf/2602.12500v1)

**作者:** André Storhaug `[一作]` (Norwegian University of Science and Technology), Jingyue Li `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4245 | [OpenAlex ID](https://openalex.org/A5067021027)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 Favia 的代理驱动框架，用于在大型开源仓库中高效识别与公开 CVE 相关的漏洞修复提交。

**💡 创新点**

结合可扩展的候选排序与迭代深度推理的 Agent 架构，通过工具调用与环境交互实现对漏洞根因与代码变更的因果对齐，显著提升了精度-召回平衡。

**🔧 技术方法**

使用 ReAct 方案的 LLM 代理、CVE/CWE 检索工具、代码搜索与文件操作工具，模型覆盖 Gemma、Llama 3.3、Qwen 3 等大模型。

**📊 数据集**

基于新构建的 CVEVC 数据集，包含超过 8 万条提交、3,708 个真实仓库及对应 CVE 信息。

**📈 对比分析**

与传统机器学习、PatchFinder 以及 LLM4VFD、CommitShield 等基线在随机和真实候选集上对比，Favia 在真实候选集上实现最高 F1（0.56）并保持近乎完美召回（>0.94），精度提升约 92%。

**⚠️ 局限性**

受限于依赖 PatchFinder 生成候选集、对预训练知识的潜在泄漏以及对高层次语言覆盖偏好，且多轮推理导致较高的 token 消耗与计算成本。

---

## 112. Scaling Single Human Demonstrations for Imitation Learning using Generative Foundational Models

**arXiv ID:** 2602.12734 | [PDF](https://arxiv.org/pdf/2602.12734v1)

**作者:** Nick Heppert `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2542 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种从单个真人演示自动生成机器人训练数据并在模拟中学习操控策略的 Real2Sim 方法。

**💡 创新点**

创新点在于利用 3D 生成模型与 Zero‑Shot‑Pose 匹配来直接把真人演示映射到机器人模拟，消除了手工对齐和对象检索的需求。

**🔧 技术方法**

结合 Point‑E、Zero‑Shot‑Pose、SAPIEN 仿真、基于流匹配的 PointFlowMatch 学习以及 LoFTR/ ZSP 等特征匹配技术。

**📊 数据集**

使用 DITTO 单人演示任务和 Objaverse 提供的未见 3D 物体做模拟训练与评估。

**📈 对比分析**

与 DITTO 基线（LoFTR/ ZSP）在三项任务上进行 100 次实验，Real2Sim 在成功率上平均提升约 10‑15% 并在真实机器人上实现零样本迁移。

**⚠️ 局限性**

方法依赖生成模型的质量，匹配失败时需人工干预；对深度摄像误差敏感，且目前仅支持单主物体或简单二元任务。

---

## 113. Concatenated Codes for Short-Molecule DNA Storage with Sequencing Channels of Positive Zero-Undetected-Error Capacity

**arXiv ID:** 2602.12800 | [PDF](https://arxiv.org/pdf/2602.12800v1)

**作者:** Ran Tamir `[一作]` (Universitat Politècnica de Catalunya), Albert Guillén i Fàbregas `[通讯]` (University of Cambridge)

**通讯引用:** 3150 | [OpenAlex ID](https://openalex.org/A5005125538)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了短分子DNA存储系统在存在测序噪声时，可可靠存储的信息量，并给出了一个可实现的分层编码方案。

**💡 创新点**

创新点在于首次将线性块码与零未检测错误（zero‑undetected‑error）解码器结合到内码，并利用其信息独立性，推导出在对称测序通道下随机线性块码的误删指数界，从而实现短分子下的可扩展存储容量上界。

**🔧 技术方法**

主要技术包括：多层编码（外码处理随机抽样，内码处理测序噪声）、线性块码与零未检测错误解码、随机 Dirichlet 采样产生的概率分布向量、KL 效率分析以及误删指数（error‑exponent）证明。

**📊 数据集**

该工作为理论分析，没有使用实际数据集，全部基于离散记忆无关对称测序通道模型和 Dirichlet 分布生成的随机代码。

**📈 对比分析**

通过与前人使用 Feinstein 极大码率证明的上界以及无噪声情况下的容量上界进行比较，证明所给编码在任何 β∈(0,1/|X|) 范围内可实现 log‑cardinality≈(1−β log|X|/2) M^{β log|X|} log M 的扩展率，并且误码概率随 M 指数下降，显示了理论上的可靠性，但其性能仍不及最优（Feinstein）极限。

**⚠️ 局限性**

主要局限包括：要求测序通道必须是对称的；仅考虑零未检测错误解码，不能处理误检；分层编码结构将内外码解码分离，缺乏联合解码导致的性能损失；未能达到已知的最优可扩展率；对短分子长度 β 的取值范围仍受限于理论证明的有效区间。

---

## 114. Fractional Order Federated Learning for Battery Electric Vehicle Energy Consumption Modeling

**arXiv ID:** 2602.12567 | [PDF](https://arxiv.org/pdf/2602.12567v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California), YangQuan Chen `[通讯]` (University of California)

**通讯引用:** 50030 | [OpenAlex ID](https://openalex.org/A5100715957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在联邦学习环境下为电动汽车能耗建模提出了 FO‑RI‑FedAvg，一种融合了分数阶局部优化和粗糙度自适应正则化的联邦平均算法；

**💡 创新点**

创新点在于①使用分数阶预条件器为局部更新注入可调的记忆效应；②利用基于损失曲面粗糙度的客户端指标动态调节正则化强度，从而针对性地抑制客户端漂移；③两种机制可以独立切换，形成可解释的模块化方案；

**🔧 技术方法**

关键技术包括分数阶Caputo微分近似、粗糙度指数计算（多方向总变差与幅度归一化）、粗糙度响应函数、可选谱平滑门控以及标准 FedAvg 的聚合框架；

**📊 数据集**

实验采用真实电动汽车数据集 Vehicle Energy Dataset（VED）及其扩展版本 eVED，构造窗口化能耗回归任务；

**📈 对比分析**

与 FedAvg、FedProx、SCAFFOLD、FedNova、FedAdam、MOON、FedDyn、FedEL 等多种先进基线对比；在 VED/eVED 上 FO‑RI‑FedAvg 在 RMSE、MAE、MAPE 以及客户端漂移、收敛速度上均取得最优或近优性能，显著降低漂移与收敛波动；

**⚠️ 局限性**

局限性包括：①需要额外的粗糙度诊断与分数阶参数调优，增加客户端计算和调参复杂度；②当前实现仅覆盖 0<α≤1 的分数阶；③在不同任务或更大规模、极端异构环境下的泛化仍待验证；

---

## 115. Generalizing UxV Network Control Optimization with Disruption Tolerant Networking

**arXiv ID:** 2602.12448 | [PDF](https://arxiv.org/pdf/2602.12448v1)

**作者:** Quyen Dang `[一作]` (Naval Postgraduate School), Geoffrey Xie `[通讯]` (Naval Postgraduate School)

**通讯引用:** 2202 | [OpenAlex ID](https://openalex.org/A5113885299)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于DTN的网络控制系统（NCS）通信模型Net()，并将其嵌入现有的子模优化框架，用来协调无人车辆（UxV）在海上搜索任务中的运动。

**💡 创新点**

创新点在于：①对每对节点细粒度定义允许的失联周期c(i,j)和最大跳数h(i,j)，实现可调的拓扑灵活性；②将通信约束转化为可归约的子模函数，使得原有的贪婪优化方法仍能高效运行；③通过团队化扩展，自动分配低c值内部联通、高c值跨队通信，提升任务并行性。

**🔧 技术方法**

使用技术包括：子模多目标优化（结合感知和通信子模函数）、DTN存储转发机制、图阻抗与Laplacian特征值计算、Python仿真环境（基于MATLAB原始代码）。

**📊 数据集**

使用的数据集为仿真生成的20×20格网场景，包含5架UAV、1艘HVU以及两处目标区（TAI），参数均在表格中给出（s_max=4格、c_max=5格、m_max=4格）。

**📈 对比分析**

通过对比三种Net()配置（Net-1、Net-2、Net-3）以及团队化配置（Net-Team），结果显示：Net-3在8个控制周期内完成目标检测，比Net-2快3个周期；Net-Team进一步压缩到6个周期，且在能量消耗与冗余搜索上表现更优。

**⚠️ 局限性**

局限性包括：①仅在仿真环境下验证，缺乏真实部署实验；②未考虑能量消耗与动态障碍物等实际约束；③Net()参数仍需人工选择，缺乏自动化优化方法。

---

## 116. GroundLink: Exploring How Contextual Meeting Snippets Can Close Common Ground Gaps in Editing 3D Scenes for Virtual Production

**arXiv ID:** 2602.12987 | [PDF](https://arxiv.org/pdf/2602.12987v1)

**作者:** Gun Woo `[一作]`, Fraser Anderson `[通讯]` (Autodesk Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出并实现了GroundLink——一款Unity插件，可将虚拟制作会议中的讨论知识实时呈现在3D编辑器中，包含决策仪表盘、约束感知的 feedforward 指示以及跨模态同步，帮助团队快速建立共同基础，提升编辑效率与信心。

**💡 创新点**

创新点包括：
- 将会议语音/文本决策与 3D 场景元素通过双向引用链接，实现“语义-空间”同步；
- 在编辑器内实时渲染鼓励区/限制区的可视化 feedforward，主动提示已达成或违背的设计约束；
- 结合 AI 摘要与问答，支持针对性查询，缩短信息检索时间；
- 通过 Dual Coding 与 TMS/SMM 理论设计，促进共享心理模型与分布式知识的同时建立。

**🔧 技术方法**

核心技术：Unity 编辑器插件、Electron+React 前端仪表盘、WebSocket 双向通信、OpenAI GPT‑5（实时摘要/问答）、手工匹配的会议与场景 JSON 规范；辅以传统的音视频转录、注释与可视化渲染。

**📊 数据集**

数据集：自研的 3 条虚拟制作会议录音（约 15 分钟每段）及对应的文字转录、白板注释；对应的 Unity 场景变更日志与约束 JSON；用户研究共 12 名非专业 VP 参与者与 5 名专业 VP 专家，构成自定义的评估数据。

**📈 对比分析**

对比方法：将 GroundLink 与 Microsoft Copilot（ClipChamp）在同一任务中进行交叉实验。评估维度包括 NASA‑TLX 工作负荷、UES‑SF 参与度、共享心理模型与转移记忆系统（SMM/TMS）问卷、信心评级。结果显示 GroundLink 在工作负荷略低、参与度与奖励感更高，SMM 与 TMS 感知显著提升，信心与完成度均有提升（部分指标达到统计显著）。

**⚠️ 局限性**

局限性：
- 约束与决策的手工标注成本高，缺乏自动化 NLP；
- 视觉提示信息量大导致潜在的认知负荷与混乱；
- 可能导致用户对工具产生过度依赖，削弱主动决策能力；
- 研究样本规模有限，未覆盖真实生产流程中的多项目交叉与大型团队情境；
- 部分技术（如实时视频同步）在网络延迟下表现不佳。

---

## 117. Curriculum Learning and Pseudo-Labeling Improve the Generalization of Multi-Label Arabic Dialect Identification Models

**arXiv ID:** 2602.12937 | [PDF](https://arxiv.org/pdf/2602.12937v1)

**作者:** Ali Mekky `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 16686 | [OpenAlex ID](https://openalex.org/A5012055259)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建多标签阿拉伯方言识别（MLADI）数据集，并训练基于BERT的多标签分类模型（LahjatBERT）。

**💡 创新点**

创新点在于将GPT‑4o生成的多标签提示与18个二元方言可接受性分类器的预测进行聚合，利用阿拉伯方言度（ALDi）筛选负样本，形成高质量伪标签；并提出基于ALDi和标签卡尔迪亚的课程学习策略，提升模型对多标签样本的学习能力。

**🔧 技术方法**

技术手段包括：GPT‑4o多标签提示、二元可接受性分类器、ALDi评分、伪标签聚合、MARBERT（BERT）多标签分类（binary cross‑entropy），以及两种课程学习（ALDi‑based 与 cardinality‑based）。

**📊 数据集**

使用的数据集：NADI 2020/2021/2023 单标签训练集、MLADI 开发集（120句）和测试集（1000句），以及通过GPT‑4o与二元分类器生成的伪标签数据。

**📈 对比分析**

与NADI 2024基线、SIMMT等传统方法对比，在MLADI测试集上宏F1从约0.55提升至0.69（LahjatBERT + ALDi‑CL），召回率提升显著，精确率略有下降，整体性能明显优于先前系统。

**⚠️ 局限性**

局限性包括：多标签评测仅覆盖8/11方言，单标签训练数据源自地理位置标签可能产生噪声；伪标签虽提升覆盖率但仍需人工验证；课程学习策略尚未在更广泛多标签任务中验证。

---

## 118. MXFormer: A Microscaling Floating-Point Charge-Trap Transistor Compute-in-Memory Transformer Accelerator

**arXiv ID:** 2602.12480 | [PDF](https://arxiv.org/pdf/2602.12480v1)

**作者:** George Karfakis `[一作]` (University of California), Puneet Gupta `[通讯]` (University of California)

**通讯引用:** 5585 | [OpenAlex ID](https://openalex.org/A5084229134)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种全权重量驻留、混合计算的MXFP4 Compute‑in‑Memory Transformer加速器，针对固定短序列Transformer实现高吞吐量与能效。

**💡 创新点**

采用高密度Charge‑Trap Transistor (CTT) 实现重量驻留，静态层使用CTT CIM，动态注意力采用精确数字Systolic；引入两步指数对齐与10bit SAR ADC实现PTQ无训练近似FP精度；深度流水线全FWS架构。

**🔧 技术方法**

Charge‑Trap Transistor (CTT) 1T NVM CIM；MXFP4微尺度浮点格式；10bit SAR ADC；双缓冲数字Systolic数组；层级静态/动态划分；两步指数对齐；gemmini transposers；flash‑attention式延迟Softmax。

**📊 数据集**

ViT‑ImageNet、ViT‑Food‑101、BERT‑SQuAD v2 等标准数据集用于模型精度与评估。

**📈 对比分析**

与 NVIDIA A100、IBM PCM、HyFlexPIM、光学加速器及学术 SOTA（DeiT‑B/16、BERT‑Large）在 TOPS/mm²、TOPS/W、FPS、I/O BW 等指标比较；MXFormer 在 ViT‑L/32 达58k FPS、ViT‑B/16 达41k FPS；能效比 B200 类 GPU 高 1.7–2.5 倍，计算密度比非 FWS 加速器高 3.3–60.5 倍，且相较同类 FWS 设计提升约 20.9 倍。

**⚠️ 局限性**

仅适用于固定短序列 Transformer，无法直接支持 LLM；CTT 编程范围受限，需两次 pass 降低吞吐率；10bit ADC 增加面积/能耗；目前实现基于 22nm，未来 14nm 仍需验证；硬件面积大（≈561 mm²），需要多芯片流水线。

---

## 119. Constrained PSO Six-Parameter Fuzzy PID Tuning Method for Balanced Optimization of Depth Tracking Performance in Underwater Vehicles

**arXiv ID:** 2602.12700 | [PDF](https://arxiv.org/pdf/2602.12700v1)

**作者:** Yanxi Ding `[一作]` (China University of Petroleum-Beijing), Tingyue Jia `[通讯]` (China University of Petroleum-Beijing)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为水下载体的深度控制提出了一种受限粒子群优化（PSO）的六参数模糊PID调参方法，通过联合调节基准PID参数、误差量化因子及输出比例因子，实现对跟踪性能与控制能耗/饱和率的平衡。

**💡 创新点**

创新点在于：①将控制能量和饱和率约束嵌入PSO适应度函数，避免因增大控制量而产生的伪性能提升；②将六个关键参数统一为优化变量，首次实现模糊PID整体协同调节；③通过软约束控制超调，兼顾速度与稳态质量。

**🔧 技术方法**

采用模糊PID控制器、粒子群优化算法、时间加权绝对误差积分（ITAE）、控制能量和饱和率等多指标评价函数，结合仿真验证。

**📊 数据集**

使用基于AUV深度运动的线性化传递函数模型进行MATLAB/Simulink仿真；主要数据来源为步进深度指令和仿真得到的误差、控制输入、深度响应时间序列。

**📈 对比分析**

与传统PID和传统模糊PID在相同模型、相同指令下对比，结果显示：PSO调优后，ITAE从0.2631降至0.1473，稳态时间从2.301 s降至1.613 s，超调从0.1494降至0.01839；控制能量与饱和率基本保持不变，表明性能提升不伴随能耗上升。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实水下平台的实验验证；未考虑波浪、扰动或模型参数变化对鲁棒性的影响；适应度函数参数设置可能需针对不同任务调优。

---

## 120. Deep-Learning Atlas Registration for Melanoma Brain Metastases: Preserving Pathology While Enabling Cohort-Level Analyses

**arXiv ID:** 2602.12933 | [PDF](https://arxiv.org/pdf/2602.12933v1)

**作者:** Nanna E. Wielenberg `[一作]` (University of Freiburg), Tobias Fechter `[通讯]` (University of Freiburg)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5082142516)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种端到端的深度学习可变形配准框架，用于将含有黑色素瘤脑转移灶的MRI图像对齐到公共解剖参考空间，并在不需预处理和病灶掩膜的情况下保留病灶体积。

**💡 创新点**

创新点在于：1) 通过采样模块直接对原始图像采样，避免多次插值导致信息丢失；2) 采用基于距离变换标签图的前向相似度评估自动处理缺失对应关系；3) 在后向过拟合阶段加入体积保持损失，进一步稳定病灶形变；4) 将模型公开实现，支持多中心数据集的可重复分析。

**🔧 技术方法**

核心技术包括：U‑Net架构的速度场网络、尺度与平方积分得到变形场、距离变换标签图与归一化互相关的相似度损失、L2正则化与体积保持损失，以及前向/后向双阶段训练策略。

**📊 数据集**

使用了公开的 Learn2Reg 任务3健康人数据（414例）作为基线训练集；临床数据来自三中心（Clin1–3）的209例黑色素瘤脑转移灶，采用T1加权对比增强3D‑MRI。

**📈 对比分析**

在 Learn2Reg 上的通用模型（G）获得 DSC≈0.75，HD≈8.0 mm，后向过拟合模型（OF）将 DSC 提升至≈0.92，HD 降至≈6.8 mm；在 Clin1–3 上，OF 模型实现 DSC≈0.89–0.90、HD≈7.2–7.6 mm、ASSD≈0.76–0.77 mm，病灶体积收缩率约 0.84，体积保持良好，几乎无形变折叠。

**⚠️ 局限性**

局限性包括：1) 仍需粗略的仿射预对齐，初始化不佳会影响后续变形；2) 依赖标签分割的准确性，若分割误差大可能传递给配准；3) 后向过拟合耗时且可能过拟合局部噪声；4) 在小结构或少量病例中统计功效有限。

---

## 121. RAT-Bench: A Comprehensive Benchmark for Text Anonymization

**arXiv ID:** 2602.12806 | [PDF](https://arxiv.org/pdf/2602.12806v1)

**作者:** Nataša Krčo `[一作]` (Imperial College London), Yves-Alexandre de Montjoye `[通讯]` (Imperial College London)

**通讯引用:** 4617 | [OpenAlex ID](https://openalex.org/A5078253058)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了RAT-Bench，一个基于重识别风险的文本匿名化评测基准，并通过真实人口统计数据生成多语言、多难度的合成文本，对多类匿名化工具（NER、扰动、LLM）进行评估；

**💡 创新点**

创新点在于：①将重识别风险与合法合规标准挂钩，采用真实人口统计分布估计风险；②生成多语言、多难度（标准、非标准、隐含）合成文本；③支持迭代式LLM匿名化与属性列表泛化；④提供公开数据与排行榜；

**🔧 技术方法**

技术包括：使用PUMS人口统计数据抽样生成间接与直接标识符；用LLM（GPT‑4.1等）生成文本并作为攻击者；评估工具采用NER、扰动、LLM Prompt、迭代匿名化等；通过重识别风险公式和BLEU评估；

**📊 数据集**

数据集为美国社区调查（5% PUMS）中的9个间接标识符，以及6个直接标识符的合成生成；同时在英文、简体中文、西班牙语上生成样本；

**📈 对比分析**

方法上对比多种工具，结果显示LLM匿名化（如GPT‑4.1）在隐私‑实用‑成本三者中取得最佳平衡；NER工具过度删减或漏检，扰动工具效果最差；迭代匿名化在已知完整属性列表时效果最好，但对属性不完整时性能急剧下降；

**⚠️ 局限性**

局限性包括：①评测依赖LLM攻击者，无法覆盖所有攻击手段；②仅测试英文、简体中文、西班牙语，其他语言缺失；③生成文本虽基于真实统计，但仍是合成，可能不完全代表真实写作；④风险阈值选取较宽松，结果对阈值敏感；⑤工具性能受实现细节影响，无法保证在实际生产环境中的稳定性。

---

## 122. $\mathcal{X}$-KD: General Experiential Knowledge Distillation for Large Language Models

**arXiv ID:** 2602.12674 | [PDF](https://arxiv.org/pdf/2602.12674v1)

**作者:** Yuang Cai `[一作]` (Beijing University of Posts and Telecommunications), Yuyu Yuan `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 2266 | [OpenAlex ID](https://openalex.org/A5055757510)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于贝叶斯逆强化学习的经验知识蒸馏框架（𝒳-KD），使学生模型在教师的原始学习环境中进行学习，而非仅模仿教师行为。

**💡 创新点**

创新点包括：①将教师的奖励函数与蒸馏目标联合建模；②使用AVRIL框架实现奖励与策略的联合优化；③在序列级、监督级和泛化级蒸馏中统一实现，保持监督学习范式；④在黑盒蒸馏中直接利用教师生成的样本，扩展了适用范围。

**🔧 技术方法**

主要技术：贝叶斯逆强化学习（BIRL）→近似变分奖励模仿学习（AVRIL）；序列级和监督级知识蒸馏（KL/JS、reverse KL）；Boltzmann策略与Q值解码；经验正则化项（TD误差约束）。

**📊 数据集**

使用的数据集：XSum（摘要），WMT14 en‑de（机器翻译），GSM8k（算术推理）；黑盒蒸馏时使用Llama‑2‑7b生成教师样本；模型初始化：T5‑large 作为教师，T5‑small/​base 作为学生；黑盒学生使用T5‑large。

**📈 对比分析**

与基线（SeqKD、SKD、GKD、MiniLLM）比较：在所有任务中，𝒳‑KD 在大多数模型规模下都优于对应基线，取得更高的 ROUGE、BLEU 与准确率；性能‑多样性曲线显示其在保持质量的同时具有更好的多样性；在数据效率实验中，𝒳‑KD 能用 75% 数据达到与基线相同的性能，说明其更高的数据利用率。

**⚠️ 局限性**

局限性：对经验权重 λ 的调参敏感，增加了实验成本；主要聚焦于白盒蒸馏，对黑盒蒸馏的系统性评估不足，未来需进一步验证。

---

## 123. SignScene: Visual Sign Grounding for Mapless Navigation

**arXiv ID:** 2602.12686 | [PDF](https://arxiv.org/pdf/2602.12686v1)

**作者:** Nicky Zimmerman `[一作]` (Smart Systems Institute, National University of Singapore), David Hsu `[通讯]` (Smart Systems Institute, National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究如何利用导航标识进行无地图导航，提出一种基于标识中心的空间语义表示，并通过视觉语言模型实现标识解析与定位，最终在Spot机器人上实现真实环境的地图无地图导航。

**💡 创新点**

创新点包括：① 将场景对齐至标识方向的简化二维视图，显著降低视觉语言模型的推理负担；② 采用增量探索策略构建局部地图，弥补单视角遮挡带来的信息缺失；③ 通过上下文学习和时序过滤提升标识解析的准确性；④ 在多种环境中实现超越现有ReasonNav基线的高达88.6%的指令定位成功率。

**🔧 技术方法**

主要技术手段包括：视觉语言模型（GPT‑5、Gemini‑2.5‑Pro）进行标识解析与路径推理；GroundingDINO做开放式目标检测；GeNIE进行路径分割；Metric3Dv2估计单目深度；在场景中构建标识中心的空间语义表示并投影为二维图像；探索模块与VLM推理结合实现地图无地图导航。

**📊 数据集**

使用了自己收集的114个标识定位查询数据集（9类环境：医院、商场、校园、火车站、机场、户外等）以及36个RGB+稀疏深度+视觉惯性里程计序列，另外在Spot机器人上进行真实环境实验。

**📈 对比分析**

通过与ReasonNav基线对比，S2S在所有环境的平均准确率为88.6%，单个环境最高可达100%；在探索与地图构建的AB测试中，未做探索的模型准确率显著下降；在解析阶段，采用上下文学习+时序过滤后解析准确率从38%提升到93%。

**⚠️ 局限性**

局限性包括：① 依赖目标检测的精度，检测误差会直接影响后续推理；② 对组合指令解析仍有限，无法完整处理多步骤或层级指令；③ 在拥挤或遮挡严重的场景中，路径分割与地图构建可能不完整；④ 视觉语言模型推理耗时（几秒至数十秒），影响实时性；⑤ VLM在复杂空间关系推理时仍易出现误判。

---

## 124. Fool Me If You Can: On the Robustness of Binary Code Similarity Detection Models against Semantics-preserving Transformations

**arXiv ID:** 2602.12681 | [PDF](https://arxiv.org/pdf/2602.12681v1)

**作者:** Jiyong Uhm `[一作]` (Sungkyunkwan University), Hyungjoon Koo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5002028881)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了六种基于深度学习的二进制代码相似度检测（BCSD）模型在语义保持的代码变换下的鲁棒性，并构造了9,565个变体样本开展攻击实验。

**💡 创新点**

①首次系统化研究多种语义保持变换对BCSD模型的影响；②提出基于贪婪采样的假阳性触发攻击及其跨模型转移性评估；③公开了大规模变体数据集，为后续研究提供实验基准。

**🔧 技术方法**

使用了语义保持的代码变换（如 In-place Code Randomization、Basic-Block Reordering、Semantic NOP 插入、Junk Code 插入、Obfuscator‑LLVM 等），以及机器学习模型评估、贪婪采样攻击策略、SHAP 与 saliency 等 XAI 技术。

**📊 数据集**

基于620个原始 ELF 可执行文件生成的9,565个变体，结合公开的 BCSD 基准数据集（含 620 个原始样本）。

**📈 对比分析**

通过 Precision、Recall、F1 等指标对原始模型与变换后模型进行对比，结果显示不同模型对不同变换的鲁棒性差异显著，部分模型（如 BinShot）对 LLVM 混淆相对鲁棒，攻击成功率可达 100%。

**⚠️ 局限性**

未对变换进行形式化验证；变换范围有限，仅涵盖八种语义保持技术；未覆盖加密/压缩等高级防护；XAI 方法的可信度未进行深入评估；转移性评估仅在相似架构模型中显著。

---

## 125. RelBench v2: A Large-Scale Benchmark and Repository for Relational Data

**arXiv ID:** 2602.12606 | [PDF](https://arxiv.org/pdf/2602.12606v1)

**作者:** Justin Gu `[一作]` (Stanford University), Jure Leskovec `[通讯]` (Stanford University)

**通讯引用:** 112366 | [OpenAlex ID](https://openalex.org/A5091272738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文推出了 Relational Deep Learning (RDL) v2 benchmark，扩充了四个大规模多表数据库，新增 23 个自动补全任务、13 个预测任务，并将 Temporal Graph Benchmark、ReDeLex 以及 4DBInfer 等外部资源集成进统一评测框架。

**💡 创新点**

创新点主要包括：①首次引入“自动补全”任务，要求模型在时间约束下从表内缺失字段中推断值；②通过将 TGB 事件流转换为关系数据库模式，实现关系与时序学习的对比；③整合 70+ ReDeLex 数据库和 7 个 4DBInfer 数据集，形成跨域、跨任务的全景评测；④提供统一的数据、任务和评测规范，方便研究者快速复现与扩展。

**🔧 技术方法**

使用的技术主要是：异构图神经网络（GraphSAGE、ID‑GNN、Transformer‑GNN），配合 PyTorch‑Frame 的表格嵌入、Temporal‑Aware 子图采样；对比基线包括 LightGBM、全局零/均值、实体均值；评价指标涵盖 AUROC、R²、MAP 等。

**📊 数据集**

使用的数据集包括：11 个真实世界关系数据库（新加入的 Academic、Enterprise、Consumer、Medical），共 22.6M 行 29 张表；TGB 转译后的事件流数据；70+ ReDeLex 数据库；7 个 4DBInfer 数据集，形成多领域、跨任务的大规模基准。

**📈 对比分析**

实验通过与单表基线（LightGBM）及全局基线对比，RDL 在自动补全、预测和推荐任务中普遍表现更优；例如自动补全任务中 GNN 的 AUROC/准确率均超过 LightGBM 10–20%；回归任务的 R² 与 MAE 明显提高；推荐任务的 MAP@K 在多数场景下提升 2–4 倍。

**⚠️ 局限性**

限制与挑战：①大模型层数增至四层时显存压力大，部分实验在 80GB A100 上报错；②部分自动补全任务仍需手工去除高度相关列以防信息泄漏；③目前评测侧重单数据库内部多任务，跨域迁移能力尚未系统验证；④缺少多模态（文本、图像）数据的统一处理方式，未来需进一步扩展。

---

## 126. FPNet: Joint Wi-Fi Beamforming Matrix Feedback and Anomaly-Aware Indoor Positioning

**arXiv ID:** 2602.12799 | [PDF](https://arxiv.org/pdf/2602.12799v1)

**作者:** Ran Tao `[一作]`, Shi Jin `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本论文提出一种联合Wi‑Fi beamforming矩阵反馈压缩、室内定位与异常检测的深度学习框架FPNet，并在标准硬件上实现低比特位反馈的高精度定位；

**💡 创新点**

创新点包括①共享编码器实现BFM压缩与定位的多任务联合优化；②在AP端引入ADBock自动编码器实现无额外反馈位的异常检测，解决传统方法无法检测到外域O​OD样本的问题；③通过两阶段训练使得在仅100比特反馈下达到>97%定位准确率并显著提升吞吐量；

**🔧 技术方法**

主要技术为SVD+Givens旋转压缩、卷积+全连接自编码器、量化/解量化、联合损失（定位+BFM重构）、两步训练、异常检测自编码器、SGCS/EVM/吞吐量评估；

**📊 数据集**

使用Intel 5300 2.4 GHz Wi‑Fi硬件采集的办公室场景数据：28个有效子载波，20个1.3 m×1.3 m的定位区域，5 000包/区；另外收集走廊区域的OD样本做异常检测；

**📈 对比分析**

与S‑FPNet、EFNet、IEEE 802.11 T0/T1协议、KNN、SVM等基线进行对比。FPNet在100反馈位时定位准确率达97.52%，净吞吐量提升22.92%，异常检测准确率99%（误报率<1.5%）；

**⚠️ 局限性**

局限性包括对环境变化（人流、家具移动）较敏感，需要微调；目前仅针对小规模天线/20 MHz 2.4 GHz，扩展至Wi‑Fi 6/7更大天线和更高频段需进一步结构改进；异常检测阈值依赖数据集，阈值设置不够通用。

---

## 127. Empirical Modeling of Therapist-Client Dynamics in Psychotherapy Using LLM-Based Assessments

**arXiv ID:** 2602.12450 | [PDF](https://arxiv.org/pdf/2602.12450v1)

**作者:** Angela Chen `[一作]` (Carnegie Mellon University), Haiyi Zhu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5051842323)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于大语言模型的自动评估框架，用来量化心理治疗中的治疗师行为（共情与探索）、关系质量（融洽）以及客户反应（自我披露与情绪），并使用结构方程模型（SEM）揭示了它们在会话中的相互作用。

**💡 创新点**

创新点在于：①首次将LLM用于多维度心理治疗过程评估，并通过人类评分验证其可靠性；②将自动化量化指标嵌入SEM，系统性检验了即时共情与探索对客户披露与情绪的因果影响；③引入自下而上（瞬时行为→情绪）与自上而下（累计融洽→情绪）的双向路径模型，提出“双路线”理论。

**🔧 技术方法**

技术手段包括：零/一轮提示工程（prompting）、LLM推理（GPT‑4o‑mini）、上下文窗口自定义、推理推理(CoT)以产生解释；在分析阶段采用主成分分析（PCA）降维，随后使用多层级SEM（R包 lavaan）对长时序会话数据建模。

**📊 数据集**

使用的是亚历山大街（Alexander Street）公开心理治疗语料库，包含1610场一对一会话、共243,407句子，覆盖约37位客户、多位治疗师。

**📈 对比分析**

与人工评分的对比：大多数维度的ICC均落在0.6–0.8区间，平均Pearson r≈0.66；情绪与共情指标与人类一致性良好；SEM结果显示共情与探索对自我披露及自向负情绪均有显著正向效应，融洽对自向负情绪呈负向关联。

**⚠️ 局限性**

局限性包括：①样本仅来自西方治疗环境，缺乏多文化验证；②使用文本转录，未考虑语音或面部表情等多模态信息；③缺少治疗师唯一标识，无法区分个体差异；④LLM对细微情绪（如恐惧、焦虑）识别不佳；⑤融洽测量为观察者评分，可能与客户自评不一致。

---

## 128. Constraint-Rectified Training for Efficient Chain-of-Thought

**arXiv ID:** 2602.12526 | [PDF](https://arxiv.org/pdf/2602.12526v1)

**作者:** Qinhang Wu `[一作]` (Ohio State University), Ness B. Shroff `[通讯]` (Ohio State University)

**通讯引用:** 20127 | [OpenAlex ID](https://openalex.org/A5035752536)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过基于参考策略的约束优化和后训练框架，减少LLM链式推理中的冗余步骤，同时保持甚至提升准确率。

**💡 创新点**

创新点在于提出相对准确性约束（对抗参考策略的误差）与Constraint‑Rectified Training（CRT），以及两阶段训练策略，避免单阶段压缩导致准确率退化。

**🔧 技术方法**

使用强化学习后训练（改进的CRPO）、长度正则化、压缩率评估及两阶段优化等技术。

**📊 数据集**

在GSM8K、MATH、SAT Math、AMC23、AIME24、OLYMPIAD Bench等数学推理数据集，以及DeepSeek‑1.5B和Qwen3‑4B两大基础模型上进行实验。

**📈 对比分析**

与ThinkPrune、L1、O1‑Pruner、ShorterBetter、ACPO等基线对比，CRT在保持或提升准确率的同时显著降低平均生成长度和内部冗余，AES_1/2分数最高。

**⚠️ 局限性**

仍存在对参考策略选取的依赖、模型可解释性不足，以及在更大规模或跨模态任务上的验证有限。

---

## 129. SENSE-STEP: Learning Sim-to-Real Locomotion for a Sensory-Enabled Soft Quadruped Robot

**arXiv ID:** 2602.13078 | [PDF](https://arxiv.org/pdf/2602.13078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 130. Learning Image-based Tree Crown Segmentation from Enhanced Lidar-based Pseudo-labels

**arXiv ID:** 2602.13022 | [PDF](https://arxiv.org/pdf/2602.13022v1)

**作者:** Julius Pesonen `[一作]` (Finnish Geospatial Research Institute), Eija Honkavaara `[通讯]` (Finnish Geospatial Research Institute)

**通讯引用:** 8063 | [OpenAlex ID](https://openalex.org/A5010833142)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种利用低分辨率 ALS 产生的粗树冠分割并通过 SAM 2 进行精细化，随后将这些伪标签用于训练基于 RGB/多光谱图像的 Mask R‑CNN 进行个体树冠实例分割的方法。

**💡 创新点**

创新点在于：①用 ALS 生成的伪标签并通过 SAM 2 自动提升标签质量；②实现完全无人工标注的训练流程；③利用 RGB 与近红外等多光谱通道进一步提升分割精度。

**🔧 技术方法**

采用伪监督学习、Mask R‑CNN（ResNet‑50‑FPN）骨干、SAM 2 的零样本实例分割、NDVI 过滤、点云分割（watershed）等技术。

**📊 数据集**

使用 Espoonlahti 区域的 5 cm 解析度正射影像、MicaSense Altum 多光谱影像以及 2016 年的 ALS 点云，测试集包含 362 棵树。

**📈 对比分析**

与七个 Detectree2 检测器、U‑net、Grounded SAM、DeepForest+SAM 2、SAM 3 等公开基线对比，伪监督模型在 F1≈0.78、mIoU≈0.62 的指标上明显优于所有基线。

**⚠️ 局限性**

局限性包括：①依赖 ALS 数据，无法在无 ALS 区域直接应用；②对新地理区域的迁移性差；③模型在图像边缘预测不佳，存在重叠掩模未被 NMS 去除的情况；④仅在单一地区验证，需要更多多区域数据以提升泛化能力。

---

## 131. Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning

**arXiv ID:** 2602.12375 | [PDF](https://arxiv.org/pdf/2602.12375v1)

**作者:** Abdul Wahab `[一作]` (University of Alberta), Martha White `[通讯]` (University of Alberta)

**通讯引用:** 1534 | [OpenAlex ID](https://openalex.org/A5101613484)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了VBE（Value Bonuses with Ensemble Errors）算法，利用随机动作价值函数（RQF）集合的误差来产生价值奖金，从而在深度强化学习中实现首次访问乐观性和深度探索；

**💡 创新点**

创新点在于直接用随机RQF而非随机奖励函数，奖励与目标值对齐，使得价值奖金可通过TD学习收敛并随时间衰减；同时，VBE可轻松叠加到任意基准算法，实现简单可扩展的探索策略；

**🔧 技术方法**

技术实现基于Double DQN等离线值函数方法，在其上加入随机RQF集合，使用TD更新、随机采样、奖励对齐、集成规模控制等手段；

**📊 数据集**

实验数据集包括经典稀疏奖励环境（Sparse Mountain Car、Puddle World、River Swim、Deepsea）以及六款Atari游戏（Breakout、Pong、Qbert、Pitfall、Private‑Eye、Gravitar）；

**📈 对比分析**

通过与BDQN、DQN‑P、ACB、RND（PPO版）等基线对比，VBE在经典环境中更快收敛并取得最佳最终性能；在Atari游戏中通常优于BDQN和RND，部分场景与ACB相近，且在Deepsea等稀疏奖励环境中表现尤为突出；

**⚠️ 局限性**

局限性包括：在行为策略随时间变化时的收敛性理论尚未完整证明；对RQF更新频率和集成规模的超参数敏感；在某些Atari游戏（如Private‑Eye）仍难以取得高分，需要进一步研究；

---

## 132. The Constant Eye: Benchmarking and Bridging Appearance Robustness in Autonomous Driving

**arXiv ID:** 2602.12563 | [PDF](https://arxiv.org/pdf/2602.12563v1)

**作者:** Jiabao Wang `[一作]` (Zhejiang University), Yiyi Liao `[通讯]` (Zhejiang University)

**通讯引用:** 2748 | [OpenAlex ID](https://openalex.org/A5018811297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一个高保真视觉压力测试基准，仅通过生成式像素对齐风格迁移在保持几何结构不变的前提下对外观进行多样化扰动，并通过冻结DINOv3作为统一感知接口，实现E2E驾驶规划在不同范式下的零样本外观鲁棒性；

**💡 创新点**

创新点在于：①系统地解耦外观与几何影响，提供纯视觉OOV测试；②提出“常态视角”方案，将冻结的视觉基座视作稳健接口，兼容回归、扩散与评分三类规划器；

**🔧 技术方法**

技术包括Flux生成模型进行像素对齐风格迁移、DINOv3自监督视觉基座、轻量级特征适配器（MLP+CNN）以及三类规划器的集成；

**📊 数据集**

使用NAVSIM v1、v2数据集以及自研的视觉OOV基准（共计23,040条场景，69,120张高分辨率图像）；

**📈 对比分析**

与基线（仅监督backbone）和领域随机化（DR）进行对比，结果显示冻结DINOv3在所有视觉OOV样式下均保持接近原始性能，显著优于基线与DR；在NAVSIM正式评测中也提升了整体规划性能；

**⚠️ 局限性**

局限性包括：仅关注外观OOV，未针对结构性OOV；冻结backbone可能导致在某些任务中微调优势受限；生成风格迁移虽保持几何但可能引入细微误差。

---

## 133. Buy versus Build an LLM: A Decision Framework for Governments

**arXiv ID:** 2602.13033 | [PDF](https://arxiv.org/pdf/2602.13033v1)

**作者:** Jiahao Lu `[一作]` (National University of Singapore), Mohan Kankanhalli `[通讯]` (National University of Singapore)

**通讯引用:** 17075 | [OpenAlex ID](https://openalex.org/A5016415049)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出政府层面对大语言模型（LLM）的“购买 vs 建设”决策框架，并通过 SEA‑LION 与 Apertus 两国案例阐述实践经验。

**💡 创新点**

创新点：①多维度评估模型（主权、安全、成本、资源、可持续、经济、文化、技术演进）并将其纳入决策流程；②结合实际案例提供可操作的经验教训；③强调动态评估与选项价值，展示随技术与成本变迁的策略转化。

**🔧 技术方法**

技术方法：基于开源大模型（如 LLaMA、Llama‑3 等）进行持续预训练（CPT）和后训练；使用 RAG、检索增强生成；部署混合云或国家级云平台；实现模型微调、参数高效微调（LoRA）和指令/安全微调；实现多语言、可解释性与对齐工作。

**📊 数据集**

数据集：SEA‑LION 使用东南亚本土语言文本、公开抓取、人工标注、合成数据与 SEA‑HELM 基准；Apertus 采用多语言公共抓取数据、法律法规与政策文件；两国都构建本地化评估集（如多语种对话、法律判例等）。

**📈 对比分析**

比较方法：在公开多语言基准（如 HELM、GLUE、XSum 等）与自研本土基准上进行性能评测；SEA‑LION 与主流模型（OpenAI、Anthropic 等）在低资源语言上相当或领先；Apertus 在多语种评测中接近最先进开源模型。总体表现：功能完整、成本可控，但在高端推理与多模态能力上略逊。

**⚠️ 局限性**

局限性：①建设成本高昂、人才缺口大、训练基础设施与电力需求严苛；②框架多维但缺乏量化阈值，难以快速落地；③缺乏对长期经济收益与社会影响的实证评估；④在大规模多模态与高级推理能力的实现上仍面临技术挑战；⑤多语言覆盖虽大幅提升，但在极低资源语种与方言上仍需进一步数据与模型优化。

---

## 134. Eyes on Many: Evaluating Gaze, Hand, and Voice for Multi-Object Selection in Extended Reality

**arXiv ID:** 2602.12406 | [PDF](https://arxiv.org/pdf/2602.12406v1)

**作者:** Mohammad Raihanul Bashar `[一作]` (Concordia University), Anil Ufuk Batmaz `[通讯]` (Concordia University)

**通讯引用:** 1085 | [OpenAlex ID](https://openalex.org/A5005072681)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在XR环境下，通过对四种模式切换（Full-Pinch、Semi-Pinch、Double-Pinch、Voice）和三种子选择（Gaze+Dwell、Gaze+Pinch、Gaze+Voice）的组合进行用户实验，评估其在6至10个目标的串行多选任务中的表现。

**💡 创新点**

首次系统性比较并量化了多种模式切换与子选择技术在无控制器XR中的相互作用，发现持久模式切换（Double-Pinch、Voice）显著优于准持续模式；并揭示Gaze+Pinch组合在速度、准确性与用户喜好上最具优势，为XR多选设计提供了实证依据和设计建议。

**🔧 技术方法**

采用Meta Quest Pro头显，结合Unity引擎、眼动追踪、手势识别（手指间距判定pinch）以及Vosk语音识别；实验设计为被试内交叉对照，收集任务完成时间、模式切换时间、误差率、效率、以及NASA-RTLX、SUS等主观负荷评估。

**📊 数据集**

实验使用自建的50个球体网格作为目标与干扰物（6/8/10目标），并无外部公开数据集，所有数据均为实验记录生成。

**📈 对比分析**

通过三因素重复测量ANOVA和事后Bonferroni校正比较各条件；结果显示Double-Pinch+Gaze+Pinch在完成时间最短、误差率最低、逆效率最低；Semi-Pinch在所有指标上表现最差，且用户疲劳感最强；语音子选择因重复命令导致用户不满，效率低于手势子选择。

**⚠️ 局限性**

仅研究了有限规模（≤10）的串行多选，未覆盖并行选择、复杂三维布局或目标定位挑战；未包含取消选择等完整工作流；实验环境为受控网格，难以直接推广到真实XR应用中的多样化目标分布与用户行为。

---

## 135. Thinking Like a Radiologist: A Dataset for Anatomy-Guided Interleaved Vision Language Reasoning in Chest X-ray Interpretation

**arXiv ID:** 2602.12843 | [PDF](https://arxiv.org/pdf/2602.12843v1)

**作者:** Yichen Zhao `[一作]` (Shanghai Jiao Tong University), Wei Shen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11880 | [OpenAlex ID](https://openalex.org/A5048353325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了首个可实现多步视觉-语言交互推理的胸片数据集MMRad-IVL-22K，并基于该数据集训练并评估了一款能生成交互式推理链的医学大模型Anole‑RadCoT。

**💡 创新点**

创新点在于：①引入了“多步、局部视觉-语言交互推理”结构，真实模拟放射科医师的“先看后思考再再看”的诊断流程；②将局部图像、文本与坐标三者统一交互，避免了传统仅用坐标的伪视觉方法；③通过多阶段自我反思、交叉验证和人工专家裁定，保证推理链的临床准确性和可解释性。

**🔧 技术方法**

技术上采用：深度学习语言模型（DeepSeek‑v3、Qwen‑2.5‑72B、Gemini‑3.0‑Pro）进行推理链生成与自我审核；通过LoRA微调（rank = 16、scale = 32）在Anole‑Zebra‑CoT上实现参数高效训练；多模态推理采用全图+局部文本+局部图像的输入方式，并使用SSIM、mIoU、BiomedCLIP等指标评估生成一致性。

**📊 数据集**

使用的数据集为MMRad‑IVL‑22K（21,994条推理链、35个细粒度解剖区域），其构建基础为MIMIC‑CXR；同时在公开基准（如GEMeX‑ThinkVG、EHRXQA等）上做对照实验。

**📈 对比分析**

方法比较：将文本推理链（仅使用文本）与多模态推理链（加上局部图像）进行对比；在5%样本子集上评估NLG与CE指标，结果显示多模态推理在RadGraph、BLEU、ROUGE等指标上平均提升6%+，并在一致性指标上显著优于传统模型。进一步在全数据集上与七款现有LVLM对比，Anole‑RadCoT在生成一致性、定位一致性与语义一致性上分别达0.482/0.643/1.226，整体表现最优。

**⚠️ 局限性**

局限性包括：①推理链生成需要约2 分钟，尚无法满足实时临床需求；②模型规模为7 B，虽然比某些大型模型小，但在低算力环境下仍面临部署压力；③数据集虽然覆盖多解剖区域，但仍以合成推理链为主，可能缺乏某些罕见病理或极端场景的真实多样性；④对模型训练与推理仍需依赖昂贵的GPU资源。

---

## 136. RBCorr: Response Bias Correction in Language Models

**arXiv ID:** 2602.12445 | [PDF](https://arxiv.org/pdf/2602.12445v1)

**作者:** Om Bhatt `[一作]` (University of California), Anna A. Ivanova `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 764 | [OpenAlex ID](https://openalex.org/A5082150709)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LogProbs均值归一化的响应偏差校正方法（RBCorr），并在12款开放权重语言模型上验证其有效性；

**💡 创新点**

创新点在于使用小规模、类别平衡的校准集对每个答案选项的平均LogProbs进行偏差估计，并通过直接减去均值来校正后续预测，实现低成本、无额外推理开销的偏差消除；

**🔧 技术方法**

核心技术为LogProbs提取、均值归一化校正、TVD度量偏差以及对比实验（Contextual Calibration、Batch Calibration、PriDe等）来评估改进效果；

**📊 数据集**

使用12个不同规模、指令调优与非调优版本的Falcon3、Gemma3、Llama3.1模型，覆盖两、三、四选项的题库：ARITH、bAbI、COMPS、EWoK（Yes‑No），SNLI、MNLI（NLI），以及MMLU各学科子集；

**📈 对比分析**

与现有方法对比显示RBCorr在2‑Choice和3‑Choice任务中可提升约11–28%的准确率，同时显著降低TVD（偏差），在4‑Choice任务上与Batch Calibration相当；整体准确率提升在小模型上尤为显著，说明方法能恢复因偏差导致的潜在性能；

**⚠️ 局限性**

局限性包括仅适用于可访问LogProbs的开放源模型；未测试大于70B参数或非Transformer架构模型；跨模型/数据集/提示的校正系数不可迁移，需要针对每种配置重新校准。

---

## 137. Realistic Face Reconstruction from Facial Embeddings via Diffusion Models

**arXiv ID:** 2602.13168 | [PDF](https://arxiv.org/pdf/2602.13168v1)

**作者:** Dong Han `[一作]` (Huawei Technologies), Joachim Denzler `[通讯]` (Friedrich Schiller University Jena)

**通讯引用:** 11505 | [OpenAlex ID](https://openalex.org/A5024934744)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出面部嵌入映射（FEM）框架，利用预训练的 ID‑保持扩散模型 IPA‑FaceID，通过将目标 FR 或 PPFR 模型的嵌入映射到默认 FR 空间，实现从嵌入重构高分辨率逼真人脸并对真实 FR 系统实施隐私攻击。

**💡 创新点**

① 将 Kolmogorov‑Arnold 网络（KAN）用于非线性嵌入映射，提升映射精度；② 在 PPFR（隐私保护）环境下同样能重构并攻击；③ 与传统 GAN 或 StyleGAN 方案相比，FEM 只需少量训练、显存占用低、推理速度快，且在多种攻击场景下性能更优。

**🔧 技术方法**

采用 KAN 与 MLP 两种映射网络；使用 IPA‑FaceID（IP‑Adapter‑FaceID）扩散模型进行面部生成；训练时最小化均方误差（MSE）损失；对抗测试时使用 ASR、FAS 通过率等指标。

**📊 数据集**

训练集：90% FFHQ；测试集：CelebA‑HQ、LADN（含有标注与无标注两版）、LFW（低分辨率）以及部分公开数据集；用于评估嵌入映射和重构质量。

**📈 对比分析**

与 FaceTI、MAP2V、IPA‑FaceID（无映射）等现有方法在同一基准上比较；在 FR 模型上，FEM‑MLP 和 FEM‑KAN 的平均 ASR 分别达 81.5% 与 83.7%，明显高于 FaceTI（约 72%）和 MAP2V（约 77%）；在 PPFR、受保护嵌入以及受保护图像场景中，FEM 仍保持高 ASR；训练时间/显存仅为 FaceTI 的 1/17 与 1/5.8，推理速度比 MAP2V 快 42 倍。

**⚠️ 局限性**

在低分辨率、低泄漏率（<30%）的嵌入以及高强度保护（如 Fawkes）场景下，重构质量和 ASR 会显著下降；重构仍依赖预训练扩散模型的质量；对多模态或实时大规模攻击的适用性尚未评估。

---

## 138. PLLM: Pseudo-Labeling Large Language Models for CAD Program Synthesis

**arXiv ID:** 2602.12561 | [PDF](https://arxiv.org/pdf/2602.12561v1)

**作者:** Yuanbo Li `[一作]` (Brown University), Daniel Ritchie `[通讯]` (Brown University)

**通讯引用:** 2421 | [OpenAlex ID](https://openalex.org/A5005034184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了PLL​M自训练框架，利用未标注的3D形状通过预训练的大语言模型生成CAD程序，执行后挑选高质量程序并进行程序级扩展与缩短，形成合成的程序‑形状对进行LoRA微调，实现无监督的CAD程序合成。

**💡 创新点**

创新点在于：①把LLM的输出视为生成合成数据的起点而非最终答案；②通过执行反馈挑选并进一步扩展/缩短程序，产生多样且高质量的合成对；③将程序级数据增强与自训练循环结合，使模型在无监督条件下持续改进。

**🔧 技术方法**

使用技术包括：预训练的CAD‑Recode（基于CadQuery）大语言模型；程序采样（top‑p/top‑k），Chamfer Distance评估；程序扩展/缩短的程序级数据增强；LoRA微调；以及基于CadQuery的程序执行器。

**📊 数据集**

数据集：使用DeepCAD作为预训练数据；在缺少程序标签的ABC形状数据集上进行自适应训练和评估。

**📈 对比分析**

与原始CAD‑Recode基线及三种伪标签配对策略（最佳样本+执行、最佳样本+输入形状、批量取前20%）对比，PLL​M在Chamfer Distance上从26.12降至9.73，程序长度增大，几何精度显著提升，且在多轮迭代中保持稳定改进。

**⚠️ 局限性**

局限性：迭代过程需要大量采样、执行与筛选，计算成本高；基于有限操作集的模型在后期改进停滞，且对复杂几何的表达能力受限。

---

## 139. Building Large-Scale Drone Defenses from Small-Team Strategies

**arXiv ID:** 2602.12502 | [PDF](https://arxiv.org/pdf/2602.12502v1)

**作者:** Grant Douglas `[一作]` (Adelaide University), Mingyu Guo `[通讯]` (Adelaide University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种分阶段的 GA–DP 管线，通过在小规模团队中进化启发式策略，并用动态规划和迭代细化在大规模无人机防御中实现高效协同；

**💡 创新点**

创新点在于将整体策略编码为染色体并进行染色体级别的因子化演化，利用 DP 进行子团队分配，结合 LLM 生成多样化启发式以及迭代细化，使得从 1–8 机小规模到 45 机大规模的防御能保持高胜率；

**🔧 技术方法**

使用遗传算法、动态规划、JAX 实现的高性能仿真、以及大型语言模型自动生成启发式规则；

**📊 数据集**

使用自定义的二维战场仿真数据集，随机生成攻击者起始位置与正弦路径，并在 128 次模拟中评估胜率；

**📈 对比分析**

与随机分配与直接 GA 的基线对比，实验显示在 30 名攻击者、45 名防御者场景中最高胜率可达 0.52，显著优于基线（≈0.1 或更低）；

**⚠️ 局限性**

局限在于完全可观测、确定性动态的假设，缺乏对部分可观测、通信限制或环境随机扰动的考虑，且未在实际硬件或真实数据上验证。

---

## 140. TraceBack: Multi-Agent Decomposition for Fine-Grained Table Attribution

**arXiv ID:** 2602.13059 | [PDF](https://arxiv.org/pdf/2602.13059v1)

**作者:** Tejas Anvekar `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 1944 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种模块化的多智能体框架，实现单表问答中细粒度单元格级别的归因，并构建了包含短语‑单元格对齐的 1,500 条手工标注示例的基准；

**💡 创新点**

创新点在于（1）通过列筛选、行过滤、问题分解和子查询归因四阶段自动化定位单元格；（2）引入无参考的原子事实对齐指标，替代昂贵的人工归因评估；（3）提供细粒度评测，覆盖行、列、单元格三层；

**🔧 技术方法**

主要技术包括大语言模型（LLM）实现的多智能体推理、few‑shot 解释、SQL‑风格行过滤、原子事实生成与对齐，以及基于推理模型的事实相似度判定；

**📊 数据集**

使用的公开数据集有 ToTTo、FetaQA 与 AITQA，分别提供表格、自然语言问题与答案；

**📈 对比分析**

与传统的少量提示、检索、生成程序、InSeq 等基线相比，所提框架在行、列、单元格级别的精确度均显著提升，尤其在单元格层面精度从 42%~56% 提升至 74%~90%，召回率亦大幅提升；

**⚠️ 局限性**

局限性包括仅支持单表推理，缺乏多表连接、层级或多模态表的处理；对 LLM 依赖度高，易受提示和模型升级影响；无参考指标虽然可扩展但对绝对精度存在低估；未验证多语言或领域外的适用性。

---

## 141. Towards complete digital twins in cultural heritage with ART3mis 3D artifacts annotator

**arXiv ID:** 2602.12761 | [PDF](https://arxiv.org/pdf/2602.12761v1)

**作者:** Dimitrios Karamatskos `[一作]` (Athena Research Center), George Pavlidis `[通讯]` (Athena Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出并实现了 ART3mis 2.0，一款基于 Web 的 3D 对象注释工具，支持手动和自动 ROI 选择，且符合 W3C Web Annotation 数据模型。

**💡 创新点**

创新点包括：① 采用基于射线-多边形相交的实时 ROI 选择算法；② 将注释存储为 JSON‑LD 结构，易于分发和重用；③ 支持插件式的自动化检测（显著性与缺陷检测）模块；④ 采用 WYSIWYG、十条启发式准则的友好 UI。

**🔧 技术方法**

技术栈主要为：Three.js/WebGL 进行三维渲染；Tweakpane 用于界面交互；CNN（卷积神经网络）用于显著性和缺陷热图生成；Web 端服务调用实现自动化 ROI；JSON‑LD 用于注释数据交换。

**📊 数据集**

主要使用了 WARMEST 项目中的高多边形建筑柱子 3D 模型（约 20M 多边形）进行功能验证；未公开使用其他公开数据集。

**📈 对比分析**

与现有工具（如 ShapeAnnotator、3D‑COFORM、POTREE 等）对比，ART3mis 在 ROI 选择速度、准确性和跨平台可访问性上更优，但文中未给出量化性能指标；主要通过案例演示和主观评估进行比较。

**⚠️ 局限性**

局限性包括：① 对非封闭网格的裁剪支持不足；② 移动端实现尚未完整；③ 仅支持文本注释，缺乏图形/多模态注释；④ 需要进一步评估大规模数据集下的实时性能。

---

## 142. ART3mis: Ray-Based Textual Annotation on 3D Cultural Objects

**arXiv ID:** 2602.12725 | [PDF](https://arxiv.org/pdf/2602.12725v1)

**作者:** Vasileios Arampatzakis `[一作]` (Athena Research Center), George Pavlidis `[通讯]` (Athena Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一款名为ART3mis的交互式文本注释工具，用于在高分辨率3D文化遗产模型上快速选择并标注任意形状的区域。

**💡 创新点**

采用基于射线-多边形相交的实时区域选择方法，并将注释以JSON形式存储，支持任意形状、交叉区域，兼顾高精度与易用性；同时在WYSIWYG理念下实现多平台可用的用户界面。

**🔧 技术方法**

使用libigl几何处理库、Embree射线追踪核心、Dear ImGui图形界面，支持OBJ纹理模型并以JSON格式保存注释，运行在Windows平台。

**📊 数据集**

WARMEST项目中的阿尔罕布拉宫柱子3D模型；SHREC 2021数据集用于演示功能。

**📈 对比分析**

通过实时射线相交实现快速ROI选择，能够处理高达2000万多边形模型并保持帧率；相比传统矩形/体积选择工具，提供更灵活、更准确的多边形选区，用户体验更佳。

**⚠️ 局限性**

仅支持Windows、OBJ格式、纯文本注释、凸形lasso限制，缺少修剪/剪裁功能、跨平台Web部署，以及与语义/本体的集成。

---

## 143. Adaptive Meta-Aggregation Federated Learning for Intrusion Detection in Heterogeneous Internet of Things

**arXiv ID:** 2602.12541 | [PDF](https://arxiv.org/pdf/2602.12541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 144. Automated Testing of Task-based Chatbots: How Far Are We?

**arXiv ID:** 2602.13072 | [PDF](https://arxiv.org/pdf/2602.13072v1)

**作者:** Diego Clerissi `[一作]` (University of Milano-Bicocca), Leonardo Mariani `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 3590 | [OpenAlex ID](https://openalex.org/A5036120394)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统性实验评估了五种最先进的任务型聊天机器人测试工具，对45个来自Rasa、Dialogflow和Amazon Lex的真实聊天机器人进行测试，探讨测试正确性、会话覆盖、功能覆盖、判别器有效性与测试不稳定性等方面；

**💡 创新点**

创新点在于：①首次在多平台大规模（45个）真实聊天机器人上对比多种测试工具；②构建了统一实验流程与多维度评价指标（会话/功能覆盖、变异检测、判别器精度、测试抖动）；③提出了使用行为空间覆盖度量评估后端功能执行程度的方法；

**🔧 技术方法**

使用的技术包括：Botium框架进行会话测试、Mutabot进行变异测试、行为空间覆盖度量方法、LLM驱动的测试生成与用户模拟、以及多工具（BotTester、ChatbotTest、Dialect、BoTest等）进行自动生成与执行；

**📊 数据集**

数据集为：①193个Rasa聊天机器人（curated dataset），②185个Dialogflow聊天机器人（curated dataset），③经过同样方法筛选的45个Amazon Lex聊天机器人（15个每个平台）；

**📈 对比分析**

比较方法为：对每个聊天机器人分别运行所有工具，记录可执行测试比例、会话覆盖率、功能覆盖率、变异检测率、判别器准确率与F1分数、以及测试多次执行的抖动率。实验表明：总体上测试工具在会话覆盖和功能覆盖方面表现有限；LLM驱动工具在生成多步骤会话和判别器方面略优；但大多数工具在可执行率和判别器准确性上仍有显著缺陷；

**⚠️ 局限性**

limitations：①实验仅覆盖三大平台，未包含所有商业/开源框架；②对聊天机器人功能的后端调用覆盖仅使用行为空间方法，缺乏细粒度代码覆盖；③实验依赖GitHub公开仓库，可能忽略企业级私有机器人；④判别器评估主要基于意图/文本匹配，未深入语义层面；⑤测试不稳定性来源多样，难以系统化定位。

---

## 145. Learning functional components of PDEs from data using neural networks

**arXiv ID:** 2602.13174 | [PDF](https://arxiv.org/pdf/2602.13174v1)

**作者:** Torkel E. Loman `[一作]`, Ruth E. Baker `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在聚合‑扩散 PDE 中将神经网络嵌入未知函数参数，构造 UPDE 并在稳态数据上训练，恢复交互核 W(x)、外部势 V(x) 及其强度 κ 的完整函数形式。

**💡 创新点**

将 UDE（神经网络嵌入微分方程）推广到 PDE，首次系统评估噪声、采样稀疏与多解对函数可识别性的影响，并展示在大多数情形下可从单一或有限解准确恢复完整 PDE 结构。

**🔧 技术方法**

使用全连接 softplus 激活网络逼近空间函数，利用固定点残差损失与传统 PDE 残差损失，采用 Adam + LBFGS 双阶段非线性优化；求稳态时使用 Newton–Krylov 迭代；还提供傅里叶截断替代方案。

**📊 数据集**

完全使用人工合成的稳态解数据——对已知 W、V、κ 的聚合‑扩散方程在不同参数、噪声水平和采样密度下计算得到的解，随后加入不同噪声层级和稀疏采样版本。

**📈 对比分析**

通过与真函数及其对应的稳态解做 L² 误差对比、收敛曲线和成功/失败率评估；结果显示在低至中等噪声、足够多样化的解集下能够准确恢复函数（误差 < 10⁻²），但高噪声或单解时恢复失败。

**⚠️ 局限性**

局限在于仅验证稳态数据；对时间依赖数据的可识别性未作实验；部分参数组合可能产生多解，导致非唯一恢复；在高度噪声或极度稀疏采样时性能显著下降；对更复杂 PDE（如三维、耦合系统）的推广仍需进一步研究。

---

## 146. FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control

**arXiv ID:** 2602.13185 | [PDF](https://arxiv.org/pdf/2602.13185v1)

**作者:** Mingzhi Sheng `[一作]` (Hong Kong University of Science and Technology), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 63230 | [OpenAlex ID](https://openalex.org/A5100390838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 FlexAM 统一框架，利用动态 3D 点云作为控制信号，解耦视频外观与运动，实现多任务可控视频生成与编辑（如 I2V/V2V、摄像机控制、空间对象编辑）。

**💡 创新点**

创新点包括：① 构造多频位置编码 + 深度感知编码 + 可变点密度的 3D 控制信号，实现精确且通用的运动表示；② 通过统一的外观-运动分离实现一体化控制，省去多种专用控制信号的需求。

**🔧 技术方法**

使用的技术包括：动态点云提取与渲染、三维多频位置编码、深度编码、稀疏/密集点采样策略、VAE + Adapter 编码、Transformer latent 视频扩散模型（基于 Wan2.2Fun 5B Control）。

**📊 数据集**

训练与评估使用公开视频数据集（如 RealEstate10K 用于摄像机控制），以及为编辑任务合成的图像/视频对（Qwen Image Edit 生成的参考图像）。

**📈 对比分析**

与 DaS、VACE、Wan2.2Fun Control、ReCamMaster、GeoDiffuser 等基线对比，实验表明 FlexAM 在 CLIP 对齐、视频一致性、旋转误差（1.097° vs 1.839°）以及空间对象编辑的 CLIP 分数（0.9536 vs 0.9437）等指标均优于基线，整体性能提升显著。

**⚠️ 局限性**

局限性：追踪误差会削弱控制精度；模型训练规模有限，未使用大规模 Web 视频，可能影响对复杂长时序动态的建模；缺少显式姿态监督导致某些细节仍不如专门任务模型。

---

## 147. Optimal Path Partitions in Subcubic and Almost-subcubic Graphs

**arXiv ID:** 2602.12925 | [PDF](https://arxiv.org/pdf/2602.12925v1)

**作者:** Tomáš Masařík `[一作]` (University of Warsaw), Mehmet Akif Yıldız `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图的边分解为尽可能少的路径问题上，本文先给出了在三度以下（子三度）图上的多项式算法，然后证明该问题相对于“到子三度图的边删除距离”是固定参数可解（FPT）的；

**💡 创新点**

创新点在于将原问题从 NP‑hard（即使在最大度为4的图上）转化为可在子三度图上求解，并通过结构参数化与一阶逻辑模型检查相结合，得到一个针对边删除距离的 FPT 算法；

**🔧 技术方法**

主要技术包括：① 对子三度图的路径数求解析式；② 对“几乎子三度”图构造模式（pattern）和终端集合；③ 将模式可行性转化为内部顶点互不相交路径的判定；④ 利用最近的关于扩展一阶逻辑中离散路径谓词的可判定性结果来实现模型检查；

**📊 数据集**

本文为理论研究，没有使用实验数据集；

**📈 对比分析**

与传统的通用 NP‑hard 处理方法相比，本文提供了一个以参数 k=(G)（即到子三度图的边删除距离）为参数的 FPT 算法，运行时间为 f(k)·|V(G)|^3，其中 f 是可计算的；在此框架下，算法能够在指数时间内对小参数进行全搜索；

**⚠️ 局限性**

局限性包括：① 仅对边删除距离做参数化；② 对于顶点删除距离或更小参数的可解性尚未给出；③ 仍无法解决 Gallai 猜想的更一般情况；

---

## 148. PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion

**arXiv ID:** 2602.12769 | [PDF](https://arxiv.org/pdf/2602.12769v1)

**作者:** Hong-Phuc Lai `[一作]` (Qualcomm AI Research), Anh Tran `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练、基于预训练扩散模型的高分辨率文本到图像生成框架 PixelRush，能够在单个 GPU 上以几秒级别生成 4K 及以上分辨率图像。

**💡 创新点**

核心创新点包括：① 仅对噪声级别做部分反演（partial inversion），省去多步全反演的冗余；② 结合少步扩散模型（few‑step）进一步加速；③ 针对少步场景设计的高斯平滑拼接（Gaussian blending）以消除补丁边界伪影；④ 通过噪声注入（noise injection）弥补少步导致的过度平滑。

**🔧 技术方法**

使用的技术主要有：预训练的 SDXL / SDXL‑Turbo 生成模型、DDIM 反演、基于块的 patch inference、Gaussian 软拼接、spherical‑interpolation 噪声注入、基准评估指标 FID / IS。

**📊 数据集**

实验数据集：LAION/LAION‑2B 美学子集 1000 条随机 prompt；评测分辨率 2048×2048 与 4096×4096。

**📈 对比分析**

与 SDXL‑DI、Four‑iScale、DemoFusion、FreeScale 等无训练方法比较，PixelRush 在 2048×2048 时 FID 下降至 50.13、IS 提升至 14.32，生成时间 4 秒；在 4096×4096 时 FID 54.67、IS 13.75，生成时间 20 秒；相比之下基线方法耗时 300–700 秒，速度提升 10–35 倍，质量同样领先。

**⚠️ 局限性**

局限性：① 依赖 SDXL / SDXL‑Turbo 这类特定预训练模型；② 部分反演时间步 K 的选择对效果敏感，过大或过小均会导致质量下降；③ 噪声注入仅在少步场景有效，对多步方法可能产生不良影响；④ 对超高分辨率（>8K）或非标准模型的通用性尚待验证。

---

## 149. Comparative Study of Ultrasound Shape Completion and CBCT-Based AR Workflows for Spinal Needle Interventions

**arXiv ID:** 2602.12920 | [PDF](https://arxiv.org/pdf/2602.12920v1)

**作者:** Tianyu Song `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**通讯引用:** 55249 | [OpenAlex ID](https://openalex.org/A5046896448)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

比较了基于超声形状补全的AR工作流程与基于CBCT的AR工作流程，用于腰椎针刺干预，并评估两者在用户性能、可用性与信任度上的差异。

**💡 创新点**

创新点在于将两种成像模式统一进同一AR框架，利用机器人自动超声扫描与概率形状补全技术实现实时3D重建，并通过系统化的对比实验验证其优势与局限。

**🔧 技术方法**

使用的技术包括：HoloLens 2头戴显示的AR可视化、KUKA机器人与Siemens超声机的机器人控制、卷积神经网络形状补全、ICP点云配准、Poisson表面重建、Draco压缩等。

**📊 数据集**

数据集主要包括：用于训练形状补全网络的CT数据（学习椎体形状先验），以及实验用的3D打印脊柱模型与水中凝胶制成的物理试验体模。

**📈 对比分析**

通过20名实验者的两阶段任务（规划+图像采集、针刺）进行对比研究；结果显示CBCT在规划时间更短、针刺精度更高，尤其在腰椎穿刺误差显著低于超声；超声在节段关节注射精度相近但深层目标误差较大；主观评价中CBCT获得更高可用性、工作负荷更低、信任度更高。

**⚠️ 局限性**

局限包括：仅在静态体模仿实验，未涉及真实患者组织与运动；参与者为工程师/研究者，非临床医生；手动规划路径，缺乏完全自动化；未实现实时多模态融合与临床验证。

---

## 150. SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise

**arXiv ID:** 2602.12783 | [PDF](https://arxiv.org/pdf/2602.12783v1)

**作者:** Yuejie Li `[一作]` (Huazhong University of Science and Technology), Caixin Kang `[通讯]` (University of Tokyo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为SQuTR的可控噪声口语查询检索基准，包含37,317条查询、200名真实说话者、四级噪声条件，并提供统一评测协议。

**💡 创新点**

将文本检索基准转换为口语查询并在四级可控噪声下统一评估，首次公开提供可复现的噪声评测平台。

**🔧 技术方法**

采用CosyVoice-3合成语音、DEMAND+NOISEX-92噪声混合、Whisper/Paraformer ASR、BM25、BGE、Qwen3系列以及Omni-Embed-Nemotron端到端模型。

**📊 数据集**

利用六大文本检索基准（FiQA、HotpotQA、NQ、MedicalRetrieval、DuRetrieval、T2Retrieval）及其文档集合和相关性标注。

**📈 对比分析**

统一使用nDCG@10、Recall@k、MRR@k进行评估，比较 cascaded、dense 与 end-to-end 系统，发现噪声越大性能越低，dense Qwen3-Embedding-8B 在高噪声下仍优于传统 BM25，但整体仍低于文本检索上限。

**⚠️ 局限性**

仍未能完全匹配文本检索性能，端到端模型效果低，噪声类型有限且仅四级SNR，未涵盖极端嘈杂场景，对说话者口音细粒度影响缺乏深入分析。

---

## 151. The Complexity of Homomorphism Reconstruction Revisited

**arXiv ID:** 2602.12780 | [PDF](https://arxiv.org/pdf/2602.12780v1)

**作者:** Timo Gervens `[一作]` (RWTH Aachen University), Philipp Silva da Fonseca `[通讯]` (RWTH Aachen University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究“同构映射计数重构”问题：给定若干约束图F_i及其对应的同构映射计数m_i，判定是否存在图G使得对每个i都有(F_i,G)=m_i。

**💡 创新点**

首次确定该问题在两种计数编码下的精确复杂度：二进制计数时为NEXP‑完整，单进制计数时为Σ₂^p‑完整；并在约束图为星形图时给出多项式时间解法。

**🔧 技术方法**

采用了从SuccinctClique的归约、颜色/标签技巧构造图约束、Havel‑Hakimi算法与动态规划相结合的度序列重构方法。

**📊 数据集**

无数据集，全部为理论证明与算法分析。

**📈 对比分析**

与已知的NP/PSPACE/Σ₂^p等复杂度类对比，证明了上界与下界；在星形约束下算法复杂度为m^{O(ℓ²)}（m为计数上界，ℓ为最大星形大小），即多项式时间。

**⚠️ 局限性**

限制在于仅对星形约束给出多项式解法；对一般图类、参数化版本以及更广泛的约束图仍是开放问题。

---

## 152. When Words Don't Mean What They Say: Figurative Understanding in Bengali Idioms

**arXiv ID:** 2602.12921 | [PDF](https://arxiv.org/pdf/2602.12921v1)

**作者:** Adib Sakhawat `[一作]`, Tahera Khatun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了规模最大、文化注释最丰富的孟加拉语成语语料库（10,361 条）并提供 19 字段详细标注方案。

**💡 创新点**

创新点在于将成语的语义、用法、情感、地域、历史、宗教、文化等多维元数据整合到单一表格，并通过专家共识而非众包保证标注质量。

**🔧 技术方法**

技术包括人机协作标注（AI 辅助生成初稿后由专家审核）、自定义 19 字段 JSON schema、零样本评估和专家双盲 6 级评分量表。

**📊 数据集**

使用的数据集为新构建的 10,361 条孟加拉语成语集合，基准集为 100 条手选代表性成语，并与 30 个多模态、不同规模、不同地区的 LLM 进行对比。

**📈 对比分析**

比较方法为零样本问答 + 专家双盲评分；人类平均 83.4%，但最优 LLM 仅 47.6%，所有模型均未超过 50%，体现显著性能差距。

**⚠️ 局限性**

局限包括缺乏时间维度与方言覆盖、标注缺乏可重复度指标、评测仅 100 个成语、模型多样性有限以及对多模态或跨领域适配尚未充分探讨。

---

## 153. The Configuration of Space: Probing the Way Social Interaction and Perception are Affected by Task-Specific Spatial Representations in Online Video Communication

**arXiv ID:** 2602.12771 | [PDF](https://arxiv.org/pdf/2602.12771v1)

**作者:** Yihuan Chen `[一作]` (City University of Hong Kong), Ray LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在ohyay.co平台上比较Room式场景与Gallery式视图，探究二维视频聊天中的空间配置如何影响社交互动与感知。

**💡 创新点**

创新点在于将课堂背景与座位布局引入视频会议界面，提供可视化空间线索并揭示其对团队归属、情绪与非言语沟通的影响。

**🔧 技术方法**

使用了定性研究方法：观察视频记录、半结构化访谈和开放编码分析，结合空间布局与任务设计。

**📊 数据集**

数据集由34名大学生参与的两种任务（社交支持、议题讨论）组成，共计2个实验组（Room、Gallery）各约17人。

**📈 对比分析**

通过对视频行为与访谈主题的编码对比，发现Room条件在团队认知、情绪同理与空间指向上表现更好；但未提供量化指标。

**⚠️ 局限性**

局限在于样本规模小、受试者为同一文化背景、缺乏定量评估、以及Room布局与实际会议需求可能不匹配。

---

## 154. Hemispherical Angular Power Mapping of Installed mmWave Radar Modules Under Realistic Deployment Constraints

**arXiv ID:** 2602.12584 | [PDF](https://arxiv.org/pdf/2602.12584v1)

**作者:** Maaz Qureshi `[一作]` (University of Waterloo), George Shaker `[通讯]` (University of Waterloo)

**通讯引用:** 3327 | [OpenAlex ID](https://openalex.org/A5060296994)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种在实现场景中对已安装毫米波雷达模块进行半球面接收功率映射的方法，利用机器人自动定位天线探头并在半球面上采集幅度信息，生成安装状态下的角度功率分布图。

**💡 创新点**

创新点在于：①无需旋转被测设备，可在受限空间内进行半球面采样；②结合几何一致的姿态校准与碰撞避免路径规划，保证测量的可重复性和安全性；③在非全景无源室环境下实现可部署、低成本的实地验证；④通过与全波仿真对比验证方法可靠性。

**🔧 技术方法**

使用的技术包括：7自由度Franka Emika Panda机械臂与精确末端执行器；WR‑15波导喇叭探头配合谐波混频器实现毫米波下变；频谱分析仪记录幅度；基于球坐标的几何变换与机器人正向运动学实现角度一致性；冲突检测运动规划；半球面抽样与时间同步的quasi‑static采集流程。

**📊 数据集**

实验数据集来自一块58‑63 GHz Infineon BGT60TR13C 雷达模块，在不同半径（4–15 cm）和角分辨率（10°–20°）下采集的角度功率曲线。

**📈 对比分析**

通过将实测功率图与全波仿真结果及人工手动定位基线对比，平均绝对误差降低至约1.2‑1.6 dB（vs. 1.8‑2.2 dB），同一设备重复测量时日内可重复性<0.20 dB，扫描覆盖率达到100%。

**⚠️ 局限性**

局限性包括：仅测量可达半球面，无法得到全向波束信息；近场采样受几何误差影响较大；使用幅度信息缺乏相位信息，无法实现全波束重建；在非专用无源室环境中多径与残余反射仍会影响精度；方法对机器人精度和探头校准敏感，需额外维护。

---

## 155. Usage Matters: The Role of Frequency, Duration, and Experience in Presence Formation in Social Virtual Reality

**arXiv ID:** 2602.12775 | [PDF](https://arxiv.org/pdf/2602.12775v1)

**作者:** Qijia Chen `[一作]` (University of Helsinki), Giulio Jacucci `[通讯]` (University of Helsinki)

**通讯引用:** 6592 | [OpenAlex ID](https://openalex.org/A5074899838)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究通过对295名社交VR用户的问卷调查，探讨了使用频率、会话时长和VR经验年限对整体、空间、社交和自我沉浸感的影响。

**💡 创新点**

创新点在于将使用强度（频率和时长）与沉浸感联系起来，并发现二者的交互作用产生非线性放大效应，表明高频与长时使用共同促进沉浸感；同时验证性别和年龄对该关系无调节作用。

**🔧 技术方法**

采用结构化问卷（多维沉浸感量表）以及层次多元回归分析来量化行为变量与沉浸感的关系。

**📊 数据集**

数据集来自Prolific平台收集的社交VR用户自报的使用频率、时长、经验年限及沉浸感分数，共计295份有效问卷。

**📈 对比分析**

与传统实验室控制实验相比，本研究通过大规模自然场景收集数据，未引入新的基准模型，主要通过回归模型的R²和ΔR²评估预测效果，发现频率和时长显著提升R²，交互项进一步提升解释度。

**⚠️ 局限性**

局限包括样本以欧洲为主、性别比例失衡、年龄分布偏少、使用强度仅为自报且未结合系统日志、且沉浸感测量为回顾性自评，无法捕捉瞬时沉浸变化。

---

## 156. Reconciling Complexity and Simplicity in the Business Model Canvas Design Through Metamodelling and Domain-Specific Modelling

**arXiv ID:** 2602.12721 | [PDF](https://arxiv.org/pdf/2602.12721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 157. Consistency of Large Reasoning Models Under Multi-Turn Attacks

**arXiv ID:** 2602.13093 | [PDF](https://arxiv.org/pdf/2602.13093v1)

**作者:** Yubo Li `[一作]` (Carnegie Mellon University), Rema Padman `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4351 | [OpenAlex ID](https://openalex.org/A5046671743)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了九种前沿推理语言模型在多轮对抗攻击下的一致性和鲁棒性。

**💡 创新点**

创新点在于揭示推理能力并不能自动提升对抗鲁棒性，提出五种失败模式并揭示CARG在推理模型中失效的原因。

**🔧 技术方法**

使用链式推理（CoT）、位置加权一致性指标、Confidence-Aware Response Generation（CARG）以及对抗攻击模板。

**📊 数据集**

基于MT-Consistency问答数据集，包含39个学科的多项选择题。

**📈 对比分析**

与OpenAI GPT‑4o等基准模型对比，推理模型在平均一致性上提升约3–7个百分点，但仍易受误导建议、社会压力等攻击。

**⚠️ 局限性**

局限性包括仅评估多项选择问答、8轮固定攻击类型、对抗策略覆盖不全以及对置信度方法的依赖。

---

## 158. From Guidelines to Practice: Evaluating the Reproducibility of Methods in Computational Social Science

**arXiv ID:** 2602.12747 | [PDF](https://arxiv.org/pdf/2602.12747v1)

**作者:** Fakhri Momeni `[一作]` (Gesis - Leibniz Institute for Social Sciences), Johannes Kiesel `[通讯]` (Gesis - Leibniz Institute for Social Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对47名参与者进行可用性测试，系统评估了在三种条件下（未整理文档、整理文档以及整理文档+预设执行环境）计算社会科学方法的可重现性；

**💡 创新点**

创新点在于：①将文档整理、环境标准化与人工智能辅助排错三者结合进行实验评估；②通过行为指标（成功率、任务时间、错误类型）与问卷、主题分析相结合，全面解析技术、程序与概念层面的重现障碍；

**🔧 技术方法**

使用了文档整理工具（符合Methods Hub清单）、预设的myBinder云执行环境，以及ChatGPT等AI工具用于故障排除；

**📊 数据集**

采用十个典型的计算社会科学工作流（如主张检索、语义搜索、多语言文本去毒、主题建模、SSciBERT教程、倾向评分匹配、4Chan数据采集、文本编辑距离、词嵌入偏见分析、基于主题的情感分析），这些工作流在公开仓库中已有数据和代码；

**📈 对比分析**

通过对三种条件下的成功率、任务时间、错误分布进行对比，发现整理文档提高成功率至84.2%，再加上预设环境可达90%；任务完成时间由54.5分钟降至48.7分钟；错误率显著下降，系统错误几乎被消除；

**⚠️ 局限性**

局限性包括：文档质量与方法复杂度未匹配；参与者技术水平差异大；仅关注执行层面的可重现性，未检验概念复制或科学有效性；云环境资源限制导致某些实验无法完成。

---

## 159. Monocular Markerless Motion Capture Enables Quantitative Assessment of Upper Extremity Reachable Workspace

**arXiv ID:** 2602.13176 | [PDF](https://arxiv.org/pdf/2602.13176v1)

**作者:** Seth Donahue `[一作]` (Shriners Children's Lexington), Ross Chafetz `[通讯]` (Shriners Hospitals for Children)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5029957084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种单摄像头无标记运动捕捉系统，评估其对上肢可达工作空间（UERW）的定量测量。

**💡 创新点**

创新点在于首次将人工智能驱动的单摄像头无标记运动捕捉应用于UERW任务，并与传统标记式捕捉进行验证。

**🔧 技术方法**

采用了基于PosePipe的MeTRAbs-ACAE关键点检测、MuJoCo物理引擎的全身生物力学重建以及深度学习的隐式函数优化。

**📊 数据集**

使用了9名健康成年志愿者的实验数据，包含同步标记式Vicon系统和八摄像头FLIR录像，并在VR头显中进行目标定位。

**📈 对比分析**

通过两路单摄像头（前视和偏移）与标记式系统的目标达成率、工作空间百分比和八个八分体区的达成一致率进行比较，结果显示前视摄像头在所有六个八分区的差异不显著，整体误差低于5%；偏移摄像头则在对侧前区明显低估。

**⚠️ 局限性**

限制包括样本仅为健康成人、对临床群体的泛化尚未验证、深度估计在后方或对侧空间的误差较大，以及当前仅为后期分析，缺乏实时反馈。

---

## 160. A Tighter Upper Bound for Distinct Squares

**arXiv ID:** 2602.12711 | [PDF](https://arxiv.org/pdf/2602.12711v1)

**作者:** Eitatsu Tomita `[一作]` (Kyushu Institute of Technology), Tomohiro I `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5074343416)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究单词长度为 n 的字串中不同平方数的上界，并给出了新的上界 n - Θ(log n)。

**💡 创新点**

创新点在于重新定义与每个 Lyndon 根相关的循环条件，为每个平方族额外预留一个循环，从而构造更大的独立循环集，进而提升上界。

**🔧 技术方法**

采用了组合学工具——Rauzy 图、独立循环集、循环向量，以及 Lyndon 词和 Fine‑Wilf 周期引理等理论技术，完成了对平方数集的计数与上界推导。

**📊 数据集**

本文无实验数据集，完全是理论证明；仅在叙述中引用了 OEIS A248958 作为已有例子。

**📈 对比分析**

与 Brlek & Li 的上界 n - σ 以及历史上 2n - Θ(log n)、95/48 n 等结果进行比较；当字母数 σ 为 o(log n) 时，新上界严格优于旧上界，整体上界趋近于 n。

**⚠️ 局限性**

局限性：仅给出上界的下项，尚未达到最优；对 σ 较大时改进有限；方法难以直接推广到 k‑power 或圆形字；常数项和具体改进幅度尚不明确。

---

## 161. Semantic-aware Adversarial Fine-tuning for CLIP

**arXiv ID:** 2602.12461 | [PDF](https://arxiv.org/pdf/2602.12461v1)

**作者:** Jiacheng Zhang `[一作]` (University of Melbourne), Feng Liu `[通讯]` (University of Melbourne)

**通讯引用:** 15528 | [OpenAlex ID](https://openalex.org/A5100325566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对CLIP模型进行语义感知的对抗性微调，提升零射击分类的鲁棒性。

**💡 创新点**

引入基于大型语言模型生成的多样化、去幻觉的文本描述作为对抗样本的目标，采用语义集成攻击。

**🔧 技术方法**

使用CLIP、LLM（如CuPL、GPT-4o-mini）生成文本，PGD生成对抗样本，语义过滤与平均相似度损失。

**📊 数据集**

在16个视觉分类数据集（Tiny-ImageNet、CIFAR-10/100、ImageNet-1K等）以及图像文本检索基准上进行实验。

**📈 对比分析**

与TeCoA、FARE、PMG-AFT、TGA-ZSR等SOTA方法对比，SAFT在零射击鲁棒准确率提升约3.8%~4%且在多数数据集保持第二高的清洁准确率。

**⚠️ 局限性**

依赖生成文本的质量，幻觉过滤与多文本引入带来额外计算开销。

---

## 162. Dual-Granularity Contrastive Reward via Generated Episodic Guidance for Efficient Embodied RL

**arXiv ID:** 2602.12636 | [PDF](https://arxiv.org/pdf/2602.12636v1)

**作者:** Xin Liu `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems), Dongbin Zhao `[通讯]` (State Key Laboratory of Multimodal Artificial Intelligence Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于生成的短期视频指导的双粒度对比奖励框架(DEG)，可在无人工标注或大量监督的情况下实现高效的具身强化学习。

**💡 创新点**

创新点在于：1）利用少量专家视频对大模型视频生成器进行微调，生成针对每个RL episode的个性化指导视频；2）在自监督对比学习中训练专属编码器，解决生成视频噪声与真实观测的对齐问题；3）设计粗粒度探索奖励与细粒度匹配奖励的双粒度结构，实现从粗到细的逐步模仿。

**🔧 技术方法**

核心技术包括：大型图像到视频生成模型（如Wan2.1-I2V-14B）+LoRA微调；对比自监督学习（contrastive learning）训练编码器；基于余弦相似度的奖励计算；与现有RL算法（如DrQv2、HIL‑SERL）结合。

**📊 数据集**

实验使用了18个多样化的操控任务，包括MetaWorld 12个reward‑free任务、16个带稀疏成功奖励的任务以及Franka机器人平台的stack‑cube和pick‑banana两项真实世界任务；专家视频数量仅为3‑5个，单视角。

**📈 对比分析**

在reward‑free设置下，DEG在12个MetaWorld任务上平均比TeViR、Diffusion Reward、Viper和RoboCLIP等SOTA方法收敛速度快约30%–50%，并在10个任务上实现显著的性能提升；在带稀疏成功奖励的设置中，DEG+的表现与人工专家手工设计的dense reward相当，甚至在drawer‑open、hammer、assembly等任务上超越。

**⚠️ 局限性**

主要限制包括：视频生成速度慢，需较高分辨率以保持质量，导致计算成本高；在极大空间的长序列任务中仍可能出现局部最优；对生成模型的依赖可能限制在不同任务域上的迁移性。

---

## 163. A Lightweight and Explainable DenseNet-121 Framework for Grape Leaf Disease Classification

**arXiv ID:** 2602.12484 | [PDF](https://arxiv.org/pdf/2602.12484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 164. Limits of Kernelization and Parametrization for Phylogenetic Diversity with Dependencies

**arXiv ID:** 2602.12959 | [PDF](https://arxiv.org/pdf/2602.12959v1)

**作者:** Niels Holtgrefe `[一作]` (Delft University of Technology), Norbert Zeh `[通讯]` (Dalhousie University)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5019738080)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在食物网约束下最大化物种进化多样性的选择问题，考虑α比例的前食物依赖。

**💡 创新点**

引入α-比例可生存约束并全面分析其参数化复杂度，证明在多种参数下1-难并给出核化边界。

**🔧 技术方法**

使用归约与参数化复杂度分析、OR交叉构造、核化规则以及图结构参数（顶点覆盖数、到团距离等）等技术。

**📊 数据集**

基于理论模型，无实验数据；使用标准图实例与生物学树模型作为理论输入。

**📈 对比分析**

与传统单纯PD问题对比，证明在α>0时问题仍为NP/1-难，且在某些参数下无多项式核，显示显著困难。

**⚠️ 局限性**

只给出理论复杂度，没有实际算法实现或实验评估；对特定参数（如扫描宽度）仍未确定可核化。

---

## 165. Cryptographic Choreographies

**arXiv ID:** 2602.12967 | [PDF](https://arxiv.org/pdf/2602.12967v1)

**作者:** Sebastian Mödersheim `[一作]` (Technical University of Denmark), Rosario Giustolisi `[通讯]` (IT University of Copenhagen)

**通讯引用:** 2291 | [OpenAlex ID](https://openalex.org/A5028035029)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计了一种名为 CryptoChoreo 的协作式协议规格语言，扩展了传统的 Alice‑and‑Bob 表达式，加入了非确定性选择、条件分支和可变长期记忆，并给出了对任意代数理论的语义定义，随后实现了 Haskell 工具将其投影为局部行为并导出 ProVerif 文件。

**💡 创新点**

创新点在于：①将代数推导问题与协议分支统一建模，形成可计算的语义翻译；②提供了包含指数运算（Diffie‑Hellman）的代表性代数理论的可实现翻译；③通过翻译与 ProVerif 的耦合，实现了对非单调、长期记忆协议的实证验证。

**🔧 技术方法**

采用过程代数作为目标语言，利用入侵者推导（intruder deduction）来推断消息构造与解析；实现层面使用 Haskell 编写投影工具，并通过 ProVerif 进行安全性验证；为了处理非单调内存，设计了若干启发式编码技巧。

**📊 数据集**

论文未使用传统机器学习或大规模数据集，而是通过一系列典型协议实例（如基于 TTP 的身份验证、TPM 的信息删/解密、DH 交换等）进行案例研究。

**📈 对比分析**

在 ProVerif 中验证所有案例均在一分钟内完成，展示了方法的可行性与效率；与现有低层式规范（如 Tamarin、AVISPA）相比，CryptoChoreo 在表达力上更高、易读性更强，且通过自动投影减少了手工编码错误。

**⚠️ 局限性**

主要局限包括：①代数推导问题在一般情况下不可判定，需要特定代数理论才能得到可计算翻译；②ProVerif 的抽象可能不足以捕捉非单调长期记忆行为，需手工引入启发式编码；③目前实现仅支持有限的代数实例，扩展到更复杂代数时仍需进一步研究。

---

## 166. ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

**arXiv ID:** 2602.12322 | [PDF](https://arxiv.org/pdf/2602.12322v1)

**作者:** Zhuoyang Zhang `[一作]` (Massachusetts Institute of Technology), Song Han `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 33293 | [OpenAlex ID](https://openalex.org/A5070926896)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通用高效规划框架，利用想象的未来观测图像和子任务描述来指导Vision‑Language‑Action模型完成开放世界多步骤任务。

**💡 创新点**

核心创新在于：①基于SANA架构的高分辨率、低延时前瞻图像生成器；②将生成的未来图像与子任务文本共同作为VLA的视觉与语言输入，让VLA专注于视动推理；③大规模跨体态预训练（100+万多任务样本）显著提升泛化与样本效率。

**🔧 技术方法**

技术包括：高分辨率前瞻图像生成（SANA + flow‑matching）、视觉‑语言模型（Qwen‑3‑VL‑8B‑Instruct）进行子任务推理与监控、云‑边缘闭环部署架构，以及在H100 GPU上实现0.33 s的生成时延。

**📊 数据集**

使用的数据集：Agibot‑World, RoboMind, Galaxea Open‑World, Bridge（用于预训练），以及自采集的11个多步骤厨房/办公/工厂任务（共420条，2312子任务）。

**📈 对比分析**

在11个真实世界任务上平均成功率87.4%，相较于基线π₀提升40.9%点，π₀+VLM提升30.3%点；在LIBERO模拟基准上平均成功率从96.8%提升至97.5%；在OOD任务和数据效率实验中也显示出显著优势。

**⚠️ 局限性**

主要限制：依赖大规模预训练和高算力（H100 GPU），仅使用头摄像头的视觉信息，生成步骤需至少8步仍有约0.33 s延迟，且在极端动态或未见场景下仍可能出现推理错误。

---

## 167. AstRL: Analog and Mixed-Signal Circuit Synthesis with Deep Reinforcement Learning

**arXiv ID:** 2602.12402 | [PDF](https://arxiv.org/pdf/2602.12402v1)

**作者:** Felicia B. Guo `[一作]` (University of California), Borivoje Nikolic `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络和强化学习的 AMS 电路合成框架 AstRL，能够在晶体管级别直接生成满足目标性能的电路拓扑。

**💡 创新点**

创新点包括：① 将电路合成建模为图生成任务并采用策略梯度优化；② 设计统一的多目标奖励机制；③ 引入对称感知的动作空间和掩码策略，保证结构有效性；④ 通过行为克隆和判别器实现专家对齐；⑤ 在训练中直接嵌入真实仿真器，提供基准反馈。

**🔧 技术方法**

使用的技术包括：GINE 图神经网络、PPO 强化学习、行为克隆（BC）、判别器奖励、动作空间掩码、对称动作修饰器、基于模拟器的实时评估与反馈。

**📊 数据集**

数据集为 1172 条从公开 3350 条设计中抽取的晶体管级电路，涵盖三类任务：环振荡器、比较器和运算传输放大器（OTA），并在 Skywater 130nm 工艺上进行仿真。

**📈 对比分析**

与 LLM 驱动的 AnalogCoder（GPT‑3.5/4o/5）和 Transformer‑RLHF 的 AnalogGenie 进行比较。AstRL 在所有任务中实现 100% 网表有效率，超过 90% 仿真有效率，且在规格满足率上分别为 RO 13.6%、比较器 99.2%、OTA 82.8%，显著优于基线（最高仅 45% 规格满足率）。

**⚠️ 局限性**

限制包括：仿真器循环导致计算成本高，尤其是对复杂仿真任务；方法目前针对对称或半对称电路表现最好，对纯数字或高度非对称电路的适应性仍待提升；训练需要大量标注专家轨迹，且对工艺变化的鲁棒性尚未充分验证。

---

## 168. Adding internal audio sensing to internal vision enables human-like in-hand fabric recognition with soft robotic fingertips

**arXiv ID:** 2602.12918 | [PDF](https://arxiv.org/pdf/2602.12918v1)

**作者:** Iris Andrussow `[一作]` (Max Planck Institute for Intelligent Systems), Katherine J. Kuchenbecker `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 7454 | [OpenAlex ID](https://openalex.org/A5080962480)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研发了一款配备可视触觉传感器 Minsight 与新型声学触觉传感器 Minsound 的机器人手，采用动态手指摩擦探索方式，对 20 种常见织物进行多模态感知（图像、音频、姿态），并通过 Transformer 网络实现织物分类与属性推断。

**💡 创新点**

创新点包括：①首次将柔性 MEMS 麦克风嵌入软指尖传感器，形成 Minsound；②构建视觉、声学与关节信息的多模态融合框架；③证实内部音频在织物识别中优于视觉和姿态；④通过共享编码器学习可泛化的织物属性表示。

**🔧 技术方法**

技术手段：软触觉指尖传感器（Minsight 视觉 + Minsound 声学）、MEMS 麦克风、光流与功率谱密度特征提取、Transformer 与 TCN 时序网络、数据增强与多模态编码器融合。

**📊 数据集**

数据集：收集了 20 种织物（含 3 种 hold‑out 织物）的多模态交互数据（Minsight 图像、Minsound 音频、外部麦克风噪音、六轴关节状态），每类约 80 次采样，已公开下载。

**📈 对比分析**

方法比较：将单模态（图像、光流、音频、姿态）与多模态融合对比，内部音频单模态取得 95.06% 的测试准确率，加入外部麦克风提升至 97.75%；Transformer 在 4 s 采样序列上实现 5.58 ms 的推理速度；在高噪声环境下，准确率从 67.5% 提升至 91.57%。

**⚠️ 局限性**

局限性：①预设的摩擦动作缺乏多方向探索，难以学习织物各向异性；②抓取力阈值与速度需手动调节；③数据集规模有限，无法覆盖更广泛的织物种类；④对复杂环境噪声的鲁棒性仍需进一步提升。

---

## 169. Multi-Task Learning with Additive U-Net for Image Denoising and Classification

**arXiv ID:** 2602.12649 | [PDF](https://arxiv.org/pdf/2602.12649v1)

**作者:** Vikram Lakkavalli `[一作]`, Neelam Sinha `[通讯]` (Center for Brain Research Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在 U‑Net 结构中使用加性跳过融合的 AddUNet，用于图像去噪与去噪中心的多任务学习。

**💡 创新点**

创新点在于将跳过连接从传统的拼接改为加性融合并引入可学习的非负标量门控，既保持特征维度不变，又限制跳过通道容量，从而作为一种结构正则化器提升多任务学习的优化稳定性。

**🔧 技术方法**

使用的技术包括基于 Encoder‑Decoder 的加性 U‑Net、加性跳过门控、Charbonnier 失真损失、交叉熵分类损失以及联合损失调度；网络保持固定通道数，训练采用多任务权重动态调节。

**📊 数据集**

实验数据集包括单任务去噪的 Kodak‑17（灰度）以及多任务去噪+分类的 MNIST、Fashion‑MNIST、EMNIST（平衡分割）和 STL‑10。

**📈 对比分析**

与 DnCNN、伪加性 U‑Net（直接相加）和无跳过的 AE 进行对比；在 PSNR/SSIM 上 AddUNet 与这些基线相当，尤其在中高噪声下保持较好结构一致性；在多任务设置下，去噪性能几乎不受影响且分类准确率可达 89% 左右，同时训练更稳定、泛化更好。

**⚠️ 局限性**

局限性包括：在低噪声场景下峰值 PSNR 与最强基线仍有差距；受限的跳过容量可能在极端噪声或高动态范围图像中降低恢复幅度；分类头轻量化导致在更复杂数据集（如 STL‑10）上分类效果有限。

---

## 170. Multi-Head Attention as a Source of Catastrophic Forgetting in MoE Transformers

**arXiv ID:** 2602.12587 | [PDF](https://arxiv.org/pdf/2602.12587v1)

**作者:** Anrui Chen `[一作]` (Fudan University), Li Shang `[通讯]` (Fudan University)

**通讯引用:** 6399 | [OpenAlex ID](https://openalex.org/A5004722925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

探讨并缓解Mixture-of-Experts Transformer在持续学习中出现的灾难性遗忘现象

**💡 创新点**

提出MH-MoE通过在多头子表示上进行独立路由，降低特征组合冲突，从而显著减轻遗忘

**🔧 技术方法**

使用多头路由技术、梯度方向相似度分析、有效组合数N_eff评估等方法

**📊 数据集**

在TRACE（8个任务）和预训练的Qwen3-0.6B/8B模型上进行实验

**📈 对比分析**

与LoRAMoE、SeqLoRA、EWC、GEM、O-LoRA等基线对比，MH-MoE在OP和BWT上均取得更好结果

**⚠️ 局限性**

局限在于需要增加多头路由开销，且在极小路由空间时仍存在遗忘

---

## 171. Model checking with temporal graphs and their derivative

**arXiv ID:** 2602.12446 | [PDF](https://arxiv.org/pdf/2602.12446v1)

**作者:** Binh-Minh Bui-Xuan `[一作]` (Sorbonne Université), Nathalie Sznajder `[通讯]` (Sorbonne Université)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5028032477)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对时序图提出了首个在不显式依赖生命周期参数的情况下，对单调二阶逻辑（MSO）和有限的FO子逻辑可解的算法元定理；引入了基于滑动时间窗口的导数概念，定义了时序图导数的树宽与双宽，并证明了在其导数宽度有限时，相关问题可在FPT时间内求解。

**💡 创新点**

创新点在于：1) 证明了时序图的扩展树宽/双宽可用于构造Courcelle式元定理；2) 通过导数概念把大规模扩展图分解为局部子图，从而得到新的Δ-导数宽度（Δ-differential width）元定理；3) 以此框架实现了对时序团、γ-匹配等NP-hard时序问题的逻辑表达与可判定。

**🔧 技术方法**

采用单调二阶逻辑与一阶逻辑框架，利用Courcelle定理、树宽与双宽理论，构造静态扩展图与缩并序列；运用Gaifman局部性定理对FO公式进行分解；结合双宽收缩序列的性质，设计了基于Δ-邻域的判定算法。

**📊 数据集**

该工作为理论研究，未使用任何实验数据集；所有结论均来自严谨的数学证明与构造。

**📈 对比分析**

通过与已知的时序图NP‑完整性结果对比，证明在扩展树宽/双宽有限时，MSO与FO问题可在FPT时间内解决；复杂度表达式为 f(k,|φ|)·||G||^O(1)。同时给出若干NP‑硬ness例证，说明在其它宽度度量下无法得到类似元定理。

**⚠️ 局限性**

局限性：1) 元定理仅适用于扩展树宽/双宽或其导数宽度有限的时序图，无法覆盖所有常见宽度度量（如无限树宽、基本树宽、双宽等）；2) 扩展图规模可能为O(n·τ)，导致空间与时间开销高；3) 需已知或能构造收缩序列，实际实现难度较大；4) 对于极端时序图（如完全网格）的导数宽度仍可能无限。

---

## 172. Regularized Meta-Learning for Improved Generalization

**arXiv ID:** 2602.12469 | [PDF](https://arxiv.org/pdf/2602.12469v1)

**作者:** Noor Islam S. Mohammad `[一作]` (New York University), Md Muntaqim Meherab `[通讯]` (Daffodil International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了冗余感知的正则化元学习框架，用于高维预测集成的稳健权重学习。

**💡 创新点**

通过联合相关性+误差剪枝、元特征增强、交叉验证正则化以及逆RMSE混合，显著提升条件数、降低冗余、提高泛化。

**🔧 技术方法**

使用冗余投影、统计元特征（均值、方差、中位数、范围等）、Ridge/Lasso/ElasticNet正则化以及逆RMSE加权融合。

**📊 数据集**

实验基于Kaggle Playground Series S6E1 回归基准（100k样本，72个基模型）。

**📈 对比分析**

与平均、加权平均、线性/岭堆叠、贪婪增量等基线对比，RMSE 8.582（比最佳单模型降低7.2%，比简单平均降低3.5%），速度比贪婪增量快4倍。

**⚠️ 局限性**

对基模型多样性依赖、冗余投影的 O(K^2N) 计算开销、对严重分布漂移敏感，并主要针对回归需扩展至分类或多任务。

---

## 173. X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting

**arXiv ID:** 2602.12869 | [PDF](https://arxiv.org/pdf/2602.12869v1)

**作者:** Zhan Qu `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于自监督时空对比学习的框架X‑VORTEX，用LiDAR点云序列学习机翼尾迹的物理特征并实现尾迹中心定位与短期轨迹预测。

**💡 创新点**

创新点在于：①将时间演化视为对比学习中的增强；②设计弱/强双视角采样（时间子采样+空间遮挡），促使模型对稀疏、缺失信息鲁棒；③通过自监督预训练显著降低对标注数据的需求，首次实现点云序列的尾迹轨迹预测。

**🔧 技术方法**

核心技术包括：自监督对比学习（InfoNCE）、时间分布式几何编码器（PointNet/PointNet++ + LSTM）、多视角数据增强、软中心分割头与MVP回归预测。

**📊 数据集**

使用维也纳国际机场（LOWW）收集的约103万条原始LiDAR扫描，标注子集约1.2k个尾迹序列（19,354帧），预训练子集约194k序列。

**📈 对比分析**

与启发式、二维投影CNN、YOLO、单帧3D点云模型以及监督LSTM进行对比；在1%标签下X‑VORTEX的中心定位RMSE仅为9.15 m，较最佳基线YOLO提升约66.8%；在全标注下达到5.15 m；轨迹预测t+1/t+2的RMSE分别为19.99 m/22.06 m，远优于常规卡尔曼/恒速模型。

**⚠️ 局限性**

局限性包括：预测仅覆盖短期（≤12‑16 s）且不提供不确定性估计；对极端气象条件和极稀疏扫描的鲁棒性未充分验证；模型对全局坐标归一化敏感，可能受机场特定环境影响。

---

## 174. Social, Spatial, and Self-Presence as Predictors of Basic Psychological Need Satisfaction in Social Virtual Reality

**arXiv ID:** 2602.12764 | [PDF](https://arxiv.org/pdf/2602.12764v1)

**作者:** Qijia Chen `[一作]` (University of Helsinki), Giulio Jacucci `[通讯]` (University of Helsinki)

**通讯引用:** 6592 | [OpenAlex ID](https://openalex.org/A5074899838)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

调查301名社交VR用户，构建结构方程模型探究社交、空间、自我存在感与基本心理需求（自主、能力、归属感）之间的关系，并检验性别、年龄等因素的调节作用。

**💡 创新点**

首次系统地将多维存在感映射到自决理论的基本需求，揭示社会存在感是最强预测因子，空间存在感在本研究中未直接预测需求，且发现性别和年龄显著调节存在感与需求的路径，提出存在感的动机机制而非仅仅是感知层面。

**🔧 技术方法**

使用结构方程模型（SEM）和多组SEM进行路径和调节效应检验，并采用潜变量交互模型测试空间存在感的调节作用；使用WLSMV估计器处理序数数据；采用HTMT比率检验辨别效度。

**📊 数据集**

基于Prolific在线调查平台收集的301份有效问卷，问卷包含自我决定理论需求量表和多模态存在感量表（社会、空间、自我存在子量表），受试者为至少18岁、熟悉VR头显的社交VR用户。

**📈 对比分析**

与其他方法无直接对比，评估通过模型适配度指标：χ²/df≈2.41，CFI=0.967，TLI=0.961，RMSEA=0.069，说明验证模型拟合良好；在多组分析中使用χ²差异检验验证调节效应。

**⚠️ 局限性**

样本偏年轻且性别不平衡，未收集游戏经验、人格等潜在调节变量；采用回顾性自评问卷，易受记忆偏差；空间存在感未表现出预测作用，可能与测量或样本特征相关，未来需纵向实验和跨平台验证。

---

## 175. Training Dense Retrievers with Multiple Positive Passages

**arXiv ID:** 2602.12727 | [PDF](https://arxiv.org/pdf/2602.12727v1)

**作者:** Benben Wang `[一作]` (Xidian University), Keping Bi `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5040708820)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了多正样本优化目标在密集检索器训练中的效果，探讨了如何利用LLM生成的多正标签提升检索性能。

**💡 创新点**

创新点在于统一了多正目标（JointLH、SumMargLH、LSEPair、Rand1LH）于同一对比学习框架，剖析其梯度行为与概率分配，并通过理论与实验验证LSEPair的稳健性与Rand1LH的简易性。

**🔧 技术方法**

采用对比学习（InfoNCE）扩展、对正负对的Log‑Sum‑Exp pairwise 损失、随机正采样等技术，并结合LLM进行大规模多正标签构建。

**📊 数据集**

实验数据集包括 Natural Questions、MS MARCO、BEIR（多域零样本）以及混合人类+LLM标签的混合设置。

**📈 对比分析**

通过在不同正负比例、混合质量、预算分配等情景下对比，LSEPair在绝大多数指标上优于其他方法，Rand1LH作为可靠基线表现接近；而JointLH、SumMargLH对正样本质量和数量更敏感。

**⚠️ 局限性**

局限在于实验仅覆盖英文数据，缺乏跨语言验证；对多正目标的理论推导假设较强，且在极高正负比下仍可能出现性能下降。

---

## 176. Lower Bounds on Flow Sparsifiers with Steiner Nodes

**arXiv ID:** 2602.12645 | [PDF](https://arxiv.org/pdf/2602.12645v1)

**作者:** Yu Chen `[一作]` (National University of Singapore), Mingyang Yang `[通讯]` (National University of Singapore)

**通讯引用:** 3494 | [OpenAlex ID](https://openalex.org/A5101785541)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在允许加入少量Steiner节点的情况下，如何构造流稀疏化器（flow sparsifier），并证明即使加入了O(k·2^{(log k)^c})个Steiner节点，流稀疏化器的质量也无法显著提升，给出了下界Ω((log k)^{0.3})。

**💡 创新点**

创新点在于提出了一种新的硬实例构造：利用随机生成的常数度算子图（即随机匹配生成的正则图）作为基础图，并通过多层聚类和典型节点对（typical node pairs）技术，证明了在任何基于收缩的流稀疏化器（甚至其凸组合）中，存在需求在原图上需要巨大的拥塞而在稀疏化器上却仅需要常数拥塞，从而给出了相对较强的下界。

**🔧 技术方法**

核心技术包括：
- 随机正则图的高通量性质与扩展性分析；
- 多层级聚类算法（cluster hierarchy）与“友好（friendly）”关系的递归构造；
- 典型节点对的构造与路径分解；
- 通过对边容量的细致分配和增容处理，保证在原图中任意路径需要的总长度足够大；
- 对凸组合稀疏化器的平均拥塞（average congestion）概念，用以推广单个稀疏化器的下界到其混合。

**📊 数据集**

本工作为理论研究，无需实验数据集；所有结果均通过概率上限分析（随机图构造）与组合论证明得出。

**📈 对比分析**

与已有结果对比：
- 已知的上界为O(log k / log log k)，下界为Ω(√(log k / log log k))；
- 本文证明即使加入O(k·2^{(log k)^c})个Steiner节点，下界仍为Ω((log k)^{0.3})，即大幅削弱了添加Steiner节点的潜在收益；
- 该结果说明在收缩式稀疏化器的框架下，进一步提升质量的空间有限。

**⚠️ 局限性**

局限性：
- 仅针对基于收缩的稀疏化器（及其凸组合），不涵盖更一般的稀疏化器设计；
- 证明依赖于特定的随机正则图和聚类算法，可能不适用于所有图类；
- 下界的常数与指数（如0.3）仍不是最优，尚有改进空间；
- 对实际应用中的算法实现和性能评估未作实验验证。

---

## 177. Lightweight Cluster-Based Federated Learning for Intrusion Detection in Heterogeneous IoT Networks

**arXiv ID:** 2602.12543 | [PDF](https://arxiv.org/pdf/2602.12543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 178. Online Advertising with Spatial Interactions

**arXiv ID:** 2602.12481 | [PDF](https://arxiv.org/pdf/2602.12481v1)

**作者:** Gagan Aggarwal `[一作]` (Google Research), Mingfei Zhao `[通讯]` (Google Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种将广告位视为度量空间点的空间外部性框架，并在此框架下分析了两种核心模型（最近邻模型和乘积距离模型）的分配与拍卖问题。

**💡 创新点**

创新点在于：①将空间外部性建模为基于距离的折扣函数，突破传统一维位置模型；②在最近邻模型中给出 18 倍常数近似的多项式时间算法，并证明其单调性可转化为真诚机制；③在 2D 欧氏空间中提供 PTAS；④对乘积距离模型构造从最大独立集的严格近似困难证明，证明无多项式时间算法可获得多项式阶近似。

**🔧 技术方法**

主要技术手段包括：线性规划松弛与随机化圆盘选择、动态规划与网格划分的 PTAS、归约与分数化简证明单调性、以及从 MIS 到 MSED 再到乘积距离的硬度归约。

**📊 数据集**

本文为理论研究，未使用公开数据集；所有结果均在数学模型与算法证明层面给出。

**📈 对比分析**

在最近邻模型中，提供 18 倍近似（单调）与  O(log m) 近似（快速、可实现真诚机制），在 2D 欧氏空间实现 (1+ε) 近似；乘积距离模型则证明任何多项式时间算法若要取得多项式阶近似都需 P=NP。与已有的仅考虑序列位置或无外部性的匹配算法相比，本文的算法在处理空间冲突时保持可实现性且具有可观的性能保证。

**⚠️ 局限性**

局限性：①常数 18 的近似相对粗糙，尚未证明最优或更紧的上界；② LP 基础算法在实际拍卖环境中求解成本高；③乘积距离模型的困难结果表明对更一般的外部性模型可能需要新的算法或更强假设；④对广告价值的分解假设（w_i·u_j）或随机性假设在现实场景中可能不完全满足。

---

## 179. Insertion Network for Image Sequence Correspondence

**arXiv ID:** 2602.12489 | [PDF](https://arxiv.org/pdf/2602.12489v1)

**作者:** Dingjie Su `[一作]` (Vanderbilt University), Bennett A. Landman `[通讯]` (Vanderbilt University)

**通讯引用:** 24111 | [OpenAlex ID](https://openalex.org/A5075735203)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于插入网络的序列间对应方法，用来定位二维切片在三维体积中的位置。

**💡 创新点**

创新点在于把切片匹配视为插入问题，通过全局上下文建模的注意力机制和对插入位置的高斯分布监督，克服了传统单切片预测的局限。

**🔧 技术方法**

采用CNN（ResNet‑18）提取单切片特征，随后使用Transformer编码器实现跨切片自注意力与交叉注意力，并以KL散度（或EMD）衡量预测注意力分布与高斯标签分布的差异。

**📊 数据集**

使用533份来自Vanderbilt University Medical Center的全身CT体积（共计7个人工标注关键切片），体积尺寸在512×512×167到512×512×1379之间，采样到256张切片进行训练。

**📈 对比分析**

与基准Body Part Regression（含数据增强和邻域信息传递）进行对比，插入网络在有监督设置下将平均定位误差从8.4 mm降至5.4 mm，显著优于所有BPR变体，并在推理速度上远快于完整分割方法。

**⚠️ 局限性**

局限性包括仅在轴向CT切片上验证，未针对不同成像模态或纵向时间序列进行评估；高斯标签方差固定，可能不适用于所有解剖变异；以及对极端切片外范围插入的处理仍依赖人工设定。

---

## 180. Unbiased Gradient Estimation for Event Binning via Functional Backpropagation

**arXiv ID:** 2602.12590 | [PDF](https://arxiv.org/pdf/2602.12590v1)

**作者:** Jinze Chen `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19014 | [OpenAlex ID](https://openalex.org/A5003217535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种 Functional Backpropagation (FBP) 框架，用于在事件视觉中对离散 binning 函数进行无偏梯度估计。

**💡 创新点**

创新点在于将 binning 函数提升到函数空间，通过弱导数与分部积分推导无偏梯度，保持前向输出不变且不引入额外逼近误差。

**🔧 技术方法**

使用弱导数、函数分析、分部积分、cotangent 重构、JAX 计算库以及 GPU 加速实现。

**📊 数据集**

在 Event Camera Dataset (ECD) 以及 DSEC 数据集上进行实验，涵盖运动估计、光流与 SLAM 三大任务。

**📈 对比分析**

与传统 STE、Sigmoid 等启发式梯度以及基线 binning（Rect、Linear、Gauss）相比，FBP 在 RMS 误差、EPE、收敛速度等指标上提升约 3–10%，并显著加快优化收敛。

**⚠️ 局限性**

局限性包括梯度计算开销约为原始方法的 1.6–3.7 倍；对非均匀网格和非线性重构的适用性仍待扩展；在极端噪声或复杂动态场景中的鲁棒性尚未完全验证。

---

## 181. Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control

**arXiv ID:** 2602.13193 | [PDF](https://arxiv.org/pdf/2602.13193v1)

**作者:** William Chen `[一作]` (University of California Berkeley), Sergey Levine `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可引导的视觉-语言-动作策略（Steerable Policies），旨在通过丰富的合成命令提高机器人在执行多样化任务时的控制能力。

**💡 创新点**

创新点在于通过训练可引导的策略，使其能够接受多种抽象层次的命令，从而更好地利用预训练的视觉-语言模型（VLM）的能力，提升任务的泛化能力。

**🔧 技术方法**

使用了合成语言命令和多种抽象层次的输入，结合了高层次的体态推理模型和现成的VLM进行控制。

**📊 数据集**

使用了Bridge WidowX真实世界的机器人操作数据集，扩展了标准任务级语言标签，生成了多种子任务和指令。

**📈 对比分析**

与传统的层次化方法相比，Steerable Policies在多项真实世界操作实验中表现优越，尤其在挑战性的泛化和长时间任务上，显著提高了性能。

**⚠️ 局限性**

限制在于现有的机器人数据集可能缺乏足够的行为多样性，导致在某些新任务或高度随机化任务中的应用效果不佳。

---

## 182. LCSB: Layer-Cyclic Selective Backpropagation for Memory-Efficient On-Device LLM Fine-Tuning

**arXiv ID:** 2602.13073 | [PDF](https://arxiv.org/pdf/2602.13073v1)

**作者:** Juneyoung Park `[一作]` (Opt-AI Inc), Seongwan Kim. Jaeho Lee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于残差连接与AdamW动量的选择性反向传播算法（Layer-Cyclic Selective Backpropagation，LCSB），仅在每步训练中对一部分Transformer层计算梯度，实现内存与计算效率的显著提升。

**💡 创新点**

创新点包括：1) 在反向时仅保留残差路径，detach非选层，降低梯度计算成本；2) 将该方法视作LoRA参数空间的块坐标下降（BCD），提供理论收敛性解释；3) 在4‑bit量化环境下实现隐式正则化，避免全反向传播导致的发散；4) 通过自适应调度动态调整选层比例，进一步提升速度。

**🔧 技术方法**

技术手段：残差连接、AdamW动量、LoRA低秩适配、梯度检查点、延迟加载、INT4权重量化、Block Coordinate Descent 理论框架、随机/轮询层选择、warmup 50 步、自适应调度。

**📊 数据集**

数据集：WikiText-2（语言建模）、Alpaca-52K（指令微调）、ARC-Easy（常识推理）；实验环境包括服务器端 NVIDIA A100 80GB 以及设备端 4‑bit 量化模型。

**📈 对比分析**

与 MeBP、LISA、BAdam、Stochastic Depth、MeZO 等基线对比，5 个模型、3 个任务上平均提升 1.35×–1.40× 速度，质量下降 <2%；自适应调度可达 4.55×；在 4‑bit 量化下，LCSB 能稳定收敛而全反向传播会发散，显示显著的隐式正则化效果。

**⚠️ 局限性**

局限性：仅在 LoRA 微调场景验证，未评估全微调或其他 PEFT 方法；理论收敛性主要针对凸问题；在真实手机 SoC 上尚未实验；对更大规模模型（7B+）的可扩展性未知；量化实验使用模拟环境，真实硬件可能有差异。

---

## 183. ALOE: Action-Level Off-Policy Evaluation for Vision-Language-Action Model Post-Training

**arXiv ID:** 2602.12691 | [PDF](https://arxiv.org/pdf/2602.12691v1)

**作者:** Rushuai Yang `[一作]` (AgiBot), Maoqing Yao `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ALOE 框架，利用动作级离线评估（chunked TD 与惰性集合）对视觉‑语言‑动作（VLA）模型进行实时强化学习，并通过优势加权的策略更新提升机器人真实环境下的操控表现。

**💡 创新点**

创新点：① 引入动作级离线评估，使用 Q‑chunking 与惰性（pessimistic）集合来实现稀疏奖励下的信用传播；② 在流匹配 VLA 架构上实现离线 actor‑critic，兼顾高容量模型与稳定性；③ 将价值函数视作相对偏好信号，用优势加权策略更新保证更新仅在经验支持的动作空间内。

**🔧 技术方法**

技术细节：Actor‑critic 结构；流匹配 VLA actor 与 SigLIP 视觉编码器；K 个 Q‑head 集成的惰性集合；Q‑chunking TD 目标；目标网络与惰性更新；优势加权带裁剪的指数权重；离线数据集成（人机干预、失败与成功片段）；稀疏奖励与每步惩罚；使用流匹配损失代替连续动作对数似然。

**📊 数据集**

数据集：本研究完全基于真实机器人采集的数据，涵盖三项任务：智能手机包装、洗衣折叠、产品分拣（多物体拾取‑放置）。数据由机器人在线执行、人工干预、失败/成功回放组成，未使用公开标准数据集。

**📈 对比分析**

对比方法：BC、DAgger、AWR。评估指标包括任务成功率、吞吐量（每小时完成任务数）、对未知物体的零样本泛化、对扰动的鲁棒性。ALOE 在所有任务上均显著优于基线，取得更高成功率、更多吞吐量、优越泛化与更好的恢复性能，且样本效率更高。

**⚠️ 局限性**

局限性：① 仍需人工干预完成数据收集，降低完全自主性；② 仅验证了三类任务，尚未检验在更复杂或更长时序任务中的可扩展性；③ 惰性集合可能导致价值低估，影响探索；④ 超参数调优依赖经验，缺乏自动化；⑤ 对于极高维度动作或极端稀疏奖励场景，信用传播仍面临挑战。

---

## 184. Implicit Representation of Structural Constraints in ER-to-Relational Transformation: An Analysis of Cardinality Preservation

**arXiv ID:** 2602.12856 | [PDF](https://arxiv.org/pdf/2602.12856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 185. Governing Social Media as a Public Utility: A Case for Sovereign Digital Infrastructure

**arXiv ID:** 2602.12535 | [PDF](https://arxiv.org/pdf/2602.12535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 186. FuncDroid: Towards Inter-Functional Flows for Comprehensive Mobile App GUI Testing

**arXiv ID:** 2602.12834 | [PDF](https://arxiv.org/pdf/2602.12834v1)

**作者:** Jinlong He `[一作]`, Jian Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于功能流图的交互式移动应用GUI测试方法 FuncDroid，并实现了自动化工具进行测试。

**💡 创新点**

首次构建功能流图模型，明确功能边界与交互流，并通过长短期视角迭代构造与探索模型，实现深层交互 Bug 检测。

**🔧 技术方法**

结合多模态大语言模型进行页面语义解析、测试场景生成与状态判定，采用长短期视图测试、元变换（单流与跨流）以及模型更新算法。

**📊 数据集**

在 Themis 开源基准（50 版、50 个 crash bug）与 52 个主流商业 App（多品类）上进行评测。

**📈 对比分析**

与 5 个增强基线工具（Droidbot+LLMDroid、Fastbot+LLMDroid、Humanoid+LLMDroid、GPTDroid+MemoDroid、VisionDroid+MemoDroid）对比，FuncDroid 的活动覆盖率提升 28%，代码覆盖率提升 5%，总共发现 62 个 Bug（49 个 crash，13 个非 crash），显著优于基线。

**⚠️ 局限性**

依赖 GPT‑4o API 的成本与延迟；功能流图初始化需先行探索，且对极端动态 UI 或异步事件的处理仍有限。

---

## 187. Discovering the mechanics of ultra-low density elastomeric foams in elite-level racing shoes

**arXiv ID:** 2602.12694 | [PDF](https://arxiv.org/pdf/2602.12694v1)

**作者:** Jeremy A. McCulloch `[一作]` (Stanford University), Ellen Kuhl `[通讯]` (Stanford University)

**通讯引用:** 23575 | [OpenAlex ID](https://openalex.org/A5073356597)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

在CIFAR-10和ImageNet数据集上进行了实验。

**📈 对比分析**

与现有的几种主流模型进行了比较，结果显示该模型在分类精度上提高了5%，且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 188. Power Interpretable Causal ODE Networks: A Unified Model for Explainable Anomaly Detection and Root Cause Analysis in Power Systems

**arXiv ID:** 2602.12592 | [PDF](https://arxiv.org/pdf/2602.12592v1)

**作者:** Yue Sun `[一作]` (Lehigh University), Parv Venkitasubramaniam `[通讯]` (Lehigh University)

**通讯引用:** 1206 | [OpenAlex ID](https://openalex.org/A5014965408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种统一的解释性模型PICODEn，能够在电力系统中同时完成异常检测、异常类型分类、根因定位和异常形态表征。

**💡 创新点**

创新点在于将神经ODE与可解释的因果图结合，通过学习系统的连续时间动力学并监测因果图的变化，实现对测量异常与网络攻击两类异常的区分与形态识别，同时减少对标注数据与外部因果图的依赖。

**🔧 技术方法**

核心技术包括神经ODE建模、基于因果图的可解释性机制、稀疏正则化、结构性误差指标（如M1、M2、Δ）以及基于岭回归的理论分析。

**📊 数据集**

使用基于GridSTAGE的仿真电力系统数据（IEEE 68节点模型），生成测量异常与两种FDI攻击（step、poisoning）的大规模时序数据。

**📈 对比分析**

与Deep SVDD、Anomaly Transformer在异常检测，以及与RootClam、CausalRCA在根因定位的对比，PICODE在检测、定位、分类和形态识别上均表现出更高的准确率（如检测F1≈0.95，根因Top‑1≈0.79，分类准确率≈0.98），表明性能优于现有方法。

**⚠️ 局限性**

局限性包括对异常形态分类阈值敏感、在极为隐蔽的攻击（如早期poisoning）时检测效果略逊、以及对不同电网拓扑的迁移性尚需进一步验证。

---

## 189. Perceptual Self-Reflection in Agentic Physics Simulation Code Generation

**arXiv ID:** 2602.12311 | [PDF](https://arxiv.org/pdf/2602.12311v1)

**作者:** Prashant Shende `[一作]` (Singapore University of Technology and Design), Bradley Camburn `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5073059222)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个四阶段多智能体系统，能将自然语言描述自动转换为可执行的 2D matplotlib 物理仿真动画，并通过感知自我反思循环对动画结果进行物理验证。

**💡 创新点**

首次将视觉语言模型用于评估仿真动画的物理真实性，突破了传统代码结构检查无法解决的“oracle gap”，并实现了基于动画帧的迭代自我纠错。

**🔧 技术方法**

利用 Claude Sonnet 与 Haiku 大语言模型构成四个专门代理：自然语言解读、技术需求生成、代码生成与自我纠错、以及基于图像的物理验证；同时实现代码错误检测、超时处理和“无可视输出”判定；采用可视化帧捕获与视觉验证回馈实现循环。

**📊 数据集**

在包含七个物理领域（经典力学、流体动力学、热力学、电磁学、波动力学、反应扩散、统计力学）以及非物理可视化（人口增长）的自定义场景集合上进行评测，使用 SciCode 基准的概念框架进行对比。

**📈 对比分析**

与单发生成（基准 GPT‑4 约 40% 成功率）相比，系统平均物理准确度达 91%，在 86% 的场景中满足 ≥85% 的阈值，迭代次数大多在 1–2 次内完成，整体成本约为 0.20 美元/动画。

**⚠️ 局限性**

验证度量不适用于能耗消散的系统（如反应扩散），对缓慢演化的图像变化易产生低置信度；系统仅支持 2D matplotlib 可视化，缺乏对 3D、多物理耦合以及实时参数调节的支持。

---

## 190. Detecting Object Tracking Failure via Sequential Hypothesis Testing

**arXiv ID:** 2602.12983 | [PDF](https://arxiv.org/pdf/2602.12983v1)

**作者:** Alejandro Monroy Muñoz `[一作]` (University of Amsterdam), Alexander Timans `[通讯]` (University of Amsterdam)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5050193138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于顺序假设检验（e-过程）的在线目标跟踪失败检测方法，能够在任何时间点给出可靠的失败告警。

**💡 创新点**

创新点在于将跟踪失败视为可检验的假设对，利用e-过程实现任意时刻有效的假阳性控制，并提供监督与无监督两种可配合任意跟踪器的实现方案。

**🔧 技术方法**

核心技术包括：顺序假设检验框架、e-过程设计、两种投注率（aGRAPA 与 SF-OGD）、多种跟踪质量指标（NGIoU、峰值相关性、certainty gain、sharpness gain）以及与现有跟踪模型的无缝集成。

**📊 数据集**

实验数据集涵盖四个主流跟踪基准：OTB-100、LaSOT、TrackingNet 与 GOT-10k，使用 KCF 与 SiamFC 两类经典跟踪器进行评估。

**📈 对比分析**

与传统基于启发式置信度阈值的失败检测方法对比，所提出方法在四个数据集上保持假阳性率不超过 10%（有时低至 0%），平均检测延迟显著低于手工阈值，显示出更快、更可靠的告警性能。

**⚠️ 局限性**

局限性包括：检测效果高度依赖所选的跟踪质量指标，若指标在漂移期间仍保持稳定则可能导致误判；无监督指标易受多目标切换影响；实验仅覆盖两类跟踪器，未在更复杂的多目标或极端遮挡场景下进行充分验证。

---

## 191. Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward

**arXiv ID:** 2602.12430 | [PDF](https://arxiv.org/pdf/2602.12430v1)

**作者:** Renjun Xu `[一作]` (Zhejiang University), Yang Yan `[通讯]` (Zhejiang University)

**通讯引用:** 9040 | [OpenAlex ID](https://openalex.org/A5053262861)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了 Agent Skills 的架构、获取方式、部署场景与安全治理，提出了四层验证与四级信任的治理框架。

**💡 创新点**

首次将技能抽象层与 Model Context Protocol (MCP) 对齐，并提出了 Skill Trust & Lifecycle Governance Framework，实现了从验证门到权限层次的分级治理。

**🔧 技术方法**

利用 SKILL.md 规范、MCP、强化学习（SAGE）、自主探索（SEAgent）、结构化技能库（CUA‑Skill）、GUI grounding、Prompt Injection 检测等技术。

**📊 数据集**

使用了 AppWorld、OSWorld、AndroidWorld、ScreenSpot‑Pro、SWE‑bench、OpenCUA‑72B、技能仓库 42,447 条及 98,380 条安全验证数据集。

**📈 对比分析**

与 UI‑TARS‑2、Agent S2、OpenCUA‑72B 等 CUA 系统进行基准对比，OSWorld‑V 成功率 72.6% 近人类水平，SWE‑bench 79.2%；安全评估显示 26.1% 技能存在漏洞。

**⚠️ 局限性**

缺乏跨平台可移植性、技能库规模带来的选择瓶颈、技能组合与冲突治理不足、权限模型不够细粒度、缺乏标准化验证与测试流程、持续学习易导致遗忘、评估指标主要关注任务完成而非技能质量。

---

## 192. A Lightweight Cubature Kalman Filter for Attitude and Heading Reference Systems Using Simplified Prediction Equations

**arXiv ID:** 2602.12283 | [PDF](https://arxiv.org/pdf/2602.12283v1)

**作者:** Shunsei Yamagishi `[一作]` (University of Aizu), Lei Jing `[通讯]` (University of Aizu)

**通讯引用:** 3693 | [OpenAlex ID](https://openalex.org/A5018989901)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种轻量化的立方体卡尔曼滤波器（Kaisoku Cubature Kalman Filter, KCKF），用于姿态和航向参考系统（AHRS）中的姿态估计，并通过简化CKF的预测方程实现了计算成本的降低。

**💡 创新点**

创新点在于：① 推导了与原CKF等价但运算更简洁的预测公式；② 通过矩阵运算优化和消除不必要的求和，显著减少了FLOPs；③ 保持了与CKF相同的估计精度，形成了“Jacobian-free、低计算量”的滤波器。

**🔧 技术方法**

采用了基于四元数的状态与观测模型，使用立方体卡尔曼滤波（CKF）框架，并在此基础上导出了KCKF的简化预测步骤；对比使用了EKF、UKF、KUKF、FKF和RMr-GDALKF等滤波器。

**📊 数据集**

使用Movella公司生产的MTW2-3A7G6 AHRS传感器在户外采集的行走数据集（共5组不同步态、不同距离的步行轨迹），作为实验测试数据。

**📈 对比分析**

方法：① 计算RMSE与XKF3hm提供的参考姿态比较，评估估计误差；② 记录各滤波器在MacBook Pro（M1 Pro）和Raspberry Pi 4上的平均计算时延；结果显示KCKF在MacBook上比CKF快约18.8%，在Raspberry Pi上快约15.1%，且误差与CKF相当；相比EKF、FKF、RMr-GDALKF，KCKF计算成本仍高，但已显著低于CKF。

**⚠️ 局限性**

局限性：① 误差评估以XKF3hm的姿态为参考，未使用完整地面真值；② 实验仅涵盖步行场景，未验证在高外部加速度、强磁干扰或更复杂运动下的性能；③ 与EKF在实际误差上无明显优势，仍需进一步优化以提升精度和计算效率。

---

## 193. Provably Convergent Actor-Critic in Risk-averse MARL

**arXiv ID:** 2602.12386 | [PDF](https://arxiv.org/pdf/2602.12386v1)

**作者:** Yizhou Zhang `[一作]` (California Institute of Technology), Eric Mazumdar `[通讯]` (California Institute of Technology)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5089549365)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并证明了一种两时序Actor‑Critic算法，使得在任意一般和弦Markov Game中可收敛到风险厌恶的Quantal Response Equilibrium（RQE）并实现了有限样本收敛保证。

**💡 创新点**

① 通过将风险厌恶与有限理性引入RQE，证明其在任意一般和弦MG中唯一且可收敛；② 推导RQE Bellman算子为收敛映射；③ 设计快演员慢评论家两时序迭代，并给出全局收敛及有限样本收敛分析；④ 将该理论落地到深度RL框架并展示实验效果。

**🔧 技术方法**

行为博弈理论中的风险度量与量化响应、对数障碍/KL正则化、λ-单调性与强单调性理论、收敛性证明使用Lyapunov梯度漂移、两时序学习率、深度神经网络、经验回放。

**📊 数据集**

三种实验环境：Inspection Game（二维矩阵游戏）、Gridworld Cooperation Game（自定义两智能体网格世界）以及MPE Simple Tag（固定好代理的多粒子环境）。

**📈 对比分析**

与风险中性Actor‑Critic/梯度下降做对比；对比学习曲线、收敛速率与波动性；实验结果显示风险厌恶版收敛更快、更稳定，最终收益与风险中性相当。

**⚠️ 局限性**

仅针对两玩家且假设λ-强单调性；需手动设置风险参数和正则化强度；理论对噪声/非确定性环境的鲁棒性有限；实验规模较小，未验证更大或更复杂游戏；未探究完全分散学习无需对手策略的情况。

---

## 194. Real-to-Sim for Highly Cluttered Environments via Physics-Consistent Inter-Object Reasoning

**arXiv ID:** 2602.12633 | [PDF](https://arxiv.org/pdf/2602.12633v1)

**作者:** Tianyi Xiang `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19876 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用单视角RGB‑D图像重建高度混乱的桌面场景，并通过可微物理优化保证生成的数字孪生在长期受重力作用下保持静态平衡。

**💡 创新点**

创新点在于构建显式接触图并使用层次化的可微物理约束同时优化物体姿态、质量、摩擦系数等物理属性，使得场景从几何上和动力学上都能长期稳定；同时提出两阶段（几何约束→可微物理约束）优化流程。

**🔧 技术方法**

采用SAM3D+ICP得到初始几何/姿态，构建接触图（支持树+接近边），两阶段优化：①几何约束（SDF接触+视觉正则）②层次化可微刚体仿真（DiffSDFSim）求解速度损失；最后用可微渲染（DIB‑R）进行纹理细化。

**📊 数据集**

在仿真实验中使用Google Scanned Objects (GSO) 与 YCB 物体数据集，真实实验中使用GSO与Toy4K 10种手工打印物体。

**📈 对比分析**

与单机方法SAM3D+ICP和视频采样方法HoloScene做对比。结果显示：物理稳定率提升约30–40%，设置时间和平均速度显著下降；几何指标（Chamfer Distance、F‑Score、IoU）与渲染指标（PSNR、SSIM、LPIPS）保持或略优，证明方法在保持视觉质量的同时大幅提升物理可行性。

**⚠️ 局限性**

局限性包括：单视角深度噪声导致真实环境下物理稳定率下降；可微物理优化随着物体数量和层级深度显著增加计算时间；当前未处理动态交互学习与非确定性场景的进一步泛化。

---

## 195. A Machine Learning Approach to the Nirenberg Problem

**arXiv ID:** 2602.12368 | [PDF](https://arxiv.org/pdf/2602.12368v1)

**作者:** Gianfranco Cortés `[一作]`, Alexander G. Stapleton `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于机器学习的方法，称为Nirenberg神经网络，旨在解决Nirenberg问题，即在S^2上规定高斯曲率的问题。

**💡 创新点**

创新点在于使用物理信息神经网络（PINN）直接全局参数化共形因子，并通过几何感知损失函数进行训练，同时进行高斯-博内定理的一致性检查，以提高模型的可解释性。

**🔧 技术方法**

使用了物理信息神经网络（PINN）技术，结合自动微分和几何约束来近似非线性曲率方程的解。

**📊 数据集**

使用了合成数据集，生成了与Nirenberg方程相关的样本，包括已知可实现和不可实现的高斯曲率函数。

**📈 对比分析**

通过与已知可实现和不可实现的曲率函数进行比较，模型在可实现曲率的情况下损失值非常低（10^-7到10^-10），而不可实现的曲率则损失值显著更高，表明模型能够有效区分可实现和不可实现的情况。

**⚠️ 局限性**

限制在于当前模型的能力可能不足以处理更复杂的曲率函数，且对某些未知的曲率函数的可实现性仍需进一步验证。

---

## 196. SIEFormer: Spectral-Interpretable and -Enhanced Transformer for Generalized Category Discovery

**arXiv ID:** 2602.13067 | [PDF](https://arxiv.org/pdf/2602.13067v1)

**作者:** Chunming Li `[一作]` (Nanjing University of Science and Technology), Haofeng Zhang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 1768 | [OpenAlex ID](https://openalex.org/A5064008061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种Spectral-Interpretable and -Enhanced Transformer（SIEFormer），通过对Vision Transformer（ViT）注意力机制进行谱分析，提出隐式（Band-Adaptive Filter）与显式（Maneuverable Filtering Layer）双谱分支以提升对未知类别的辨别能力。

**💡 创新点**

创新点在于：①将ViT的注意力矩阵视为图的邻接矩阵，从谱图理论出发重新定义自注意力为图卷积；②引入Band‑Adaptive Filter（自适应带通/带阻滤波）和Maneuverable Filtering Layer（频域滤波）两种互补的谱过滤方式，实现局部与全局信息的双重优化；③在Generalized Category Discovery（GCD）任务中，首次将谱视角与对比学习相结合，显著提升新类别发现效果。

**🔧 技术方法**

技术手段包括：图傅里叶变换、Chebyshev多项式滤波、ARMA/ Cayley滤波、频域卷积（FFT/ IFFT）、自监督对比学习（DINO / DINOv2）、半监督聚类与伪标签生成、以及多头注意力的谱重构。

**📊 数据集**

实验数据集覆盖通用和细粒度场景：CIFAR‑10、CIFAR‑100、ImageNet‑100、ImageNet‑1K、CUB‑200、Stanford‑Cars、FGVC‑Aircraft、Herbarium‑19 等，训练集划分为已标注与未标注两部分。

**📈 对比分析**

与现有 GCD、SimGCD、SPTNet、PromptCAL、CMS 等基线相比，SIEFormer 在所有公开基准上均实现了最优或接近最优的准确率，尤其在新类别（New）上的提升显著（平均提升 3–7%），同时保持了较低的参数量和计算开销。

**⚠️ 局限性**

局限性包括：①仍需依赖强大的预训练 ViT（如 DINOv2）作为骨干，导致迁移性能受限；②谱滤波层的参数规模虽然小，但在极大规模数据集上训练仍需较多 GPU 资源；③对未知类别数量的估计不稳健，在实际开放世界环境中可能出现类别溢出或误聚类。

---

## 197. Information-theoretic analysis of world models in optimal reward maximizers

**arXiv ID:** 2602.12963 | [PDF](https://arxiv.org/pdf/2602.12963v1)

**作者:** Alfred Harwood `[一作]`, Alex Altair `[通讯]` (Dovetail Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文研究了在完全未知的受控马尔可夫过程（CMP）环境中，观察到一个最优的确定性策略后，能够获得多少关于环境的置信信息，得出了互信息为 n log m 位的精确下界。

**💡 创新点**

创新点在于首次给出了最优策略所隐含的世界模型信息量的严格上限，并证明该结果对多种奖励累计方式（有限期、折扣期、时间平均）均成立，揭示了最优策略与环境之间的几何分割结构。

**🔧 技术方法**

主要采用信息论中的互信息、测度论、实解析函数理论以及矩阵几何级数等数学工具，构造了环境的等体积分区并证明其互信息等于 n log m。

**📊 数据集**

本研究为理论工作，不使用任何实际数据集，而是基于受控马尔可夫过程的抽象空间进行严格证明。

**📈 对比分析**

由于是理论证明，未进行实验对比或性能评估；结果仅以证明形式给出，展示了在理论假设下的确定性下界。

**⚠️ 局限性**

主要局限包括仅考虑确定性无记忆策略、完全可观测环境以及奖励仅依赖状态；对随机策略、带记忆的策略、部分可观测环境及动作相关奖励的情况尚未覆盖。

---

## 198. Secure Beamforming for ISAC Systems Under Communication Eavesdropper and Sensing Eavesdropper

**arXiv ID:** 2602.12614 | [PDF](https://arxiv.org/pdf/2602.12614v1)

**作者:** Tian Zhang `[一作]` (Shandong Normal University), Yueyi Dong `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计了一种联合波束成形方法，在存在通信窃听者和感知窃听者的ISAC系统中最大化系统密文速率，同时满足感知性能与感知安全的约束。

**💡 创新点**

创新点在于首次将感知窃听者约束引入ISAC安全优化，证明了半正定松弛（SDR）不产生次优解，并提出了一种结合SCA与一阶泰勒展开的迭代安全波束成形算法。

**🔧 技术方法**

使用的主要技术包括：SCA（序贯凸逼近）、一阶泰勒展开、SDR（半正定松弛）以及对数函数的指数化线性化。

**📊 数据集**

实验采用仿真数据，设定3个用户、8/10/12个天线、不同发射功率、路径损耗模型等参数进行性能评估。

**📈 对比分析**

与随机感知信号协方差矩阵基线相比，所提出方案在相同功率下可提升约98%密文速率，且在不同天线数、功率、感知阈值下均表现出更优的安全性能。

**⚠️ 局限性**

局限性包括：需要完整的CSI估计、在感知安全要求更严格时会降低感知性能，且算法迭代收敛速度受初始点影响，实际部署需考虑计算复杂度与实时性。

---

## 199. EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition

**arXiv ID:** 2602.12919 | [PDF](https://arxiv.org/pdf/2602.12919v1)

**作者:** Xiao Wang `[一作]` (Anhui University), Yonghong Tian `[通讯]` (Peking University)

**通讯引用:** 15628 | [OpenAlex ID](https://openalex.org/A5023918894)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个大规模的事件流视觉定位基准数据集EPRBench，并基于该数据集开发了语义引导的多模态视觉定位方法SG-VPR。

**💡 创新点**

创新之处在于将大型语言模型生成的场景描述与事件流特征融合，通过文本引导的Token选择、全局与局部语义融合以及链式推理实现了高精度、可解释的视觉定位。

**🔧 技术方法**

结合事件摄像机、DINOv2视觉编码器、CLIP文本编码器、跨模态对齐的InfoNCE、Multi-Similarity 损失、GeM池化与多尺度空间金字塔聚合，辅以LLM驱动的解释推理。

**📊 数据集**

使用自采集的EPRBench（13,109个场景，10K事件序列，65K事件帧）以及NYC-Event-VPR进行训练与评估。

**📈 对比分析**

在EPRBench上与15种SOTA VPR算法对比，SG-VPR以94.3% R@1、96.1% R@5、97.1% R@10实现最优；在NYC-Event-VPR事件模式下亦获得57.5% R@1，明显优于其他方法。

**⚠️ 局限性**

缺点是依赖于专门微调的场景专家LLM，限制了跨场景泛化与开放世界的适用性。

---

## 200. Amortized Reasoning Tree Search: Decoupling Proposal and Decision in Large Language Models

**arXiv ID:** 2602.12846 | [PDF](https://arxiv.org/pdf/2602.12846v1)

**作者:** Zesheng Hong `[一作]` (Hong Kong University of Science and Technology), Hui Pan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21166 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 RLVR 训练导致稀有推理路径被压制的机制，并提出 Amortized Reasoning Tree Search (ARTS)，通过将生成器和验证器解耦并采用流匹配目标，恢复被压制的稀有推理路径。

**💡 创新点**

将验证器重新定义为概率流估计器，用流匹配最小化局部流不一致，而非传统的点对点回归或排名损失，从而在保持多样性的同时，显著提升对长尾稀有路径的恢复能力。

**🔧 技术方法**

生成‑验证框架、稀疏深度分叉（Sparse Deep Fork）数据构造、基于 GFlowNet 的流匹配目标、低秩适配器 LoRA、Best‑of‑N 推理搜索。

**📊 数据集**

MATH‑500（数学推理子集）作为主要评测集，使用 GSM8K 训练集进行验证器训练。

**📈 对比分析**

与 GRPO、Pointwise PRM、Pairwise RM 等基线对比，ARTS 在 MATH‑500 上 BoN@16 达到 74.6% 与全参数 GRPO 的 74.7% 接近；在长尾和极端稀有路径子集，ARTS 恢复 6.9% 而对手仅 1.7%，显著提升。

**⚠️ 局限性**

只能在生成器覆盖的搜索空间内恢复路径；推理时需 Best‑of‑N 或树搜索导致延迟；奖励方案为手工设定；无法生成本来不存在的推理路径。

---

## 201. LongNav-R1: Horizon-Adaptive Multi-Turn RL for Long-Horizon VLA Navigation

**arXiv ID:** 2602.12351 | [PDF](https://arxiv.org/pdf/2602.12351v1)

**作者:** Yue Hu `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**通讯引用:** 1173 | [OpenAlex ID](https://openalex.org/A5046777734)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LongNav-R1，构建端到端多轮强化学习框架，以视觉-语言-动作（VLA）模型实现长时程导航；

**💡 创新点**

创新点在于将导航视为连续对话式多轮决策，配合 Horizon‑Adaptive Policy Optimization（HAPO）实现无价值网络的时序优势估计；

**🔧 技术方法**

采用大规模预训练 VLM（如 Qwen‑3‑VL‑2B）为基座，结合 KV 缓存重用、在线视觉令牌裁剪和时间感知核回归；

**📊 数据集**

在 HM3D V1/V2、MP3D 及 HM3D‑OVON 等四大模拟基准上训练与评估，同时在真实机上验证；

**📈 对比分析**

与现有单轮 RL 与 SFT 方法对比，LongNav‑R1 在 HM3D V1/V2 及 MP3D 的成功率分别提升至 83.7%、63.0%，整体显著优于 SOTA；

**⚠️ 局限性**

主要限制是对完整 KV 缓存的依赖，导致极长时程任务的内存开销较大，未来需探索缓存淘汰和全局记忆机制。

---

## 202. SoK: Understanding the Pedagogical, Health, Ethical, and Privacy Challenges of Extended Reality in Early Childhood Education

**arXiv ID:** 2602.12749 | [PDF](https://arxiv.org/pdf/2602.12749v1)

**作者:** Supriya Khadka `[一作]` (Coventry University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 4220 | [OpenAlex ID](https://openalex.org/A5003726306)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了 111 篇关于 3-8 岁儿童使用 AR/VR/MR/XR 的研究，阐明技术、教学、健康、隐私和公平等多维挑战，并提出 Augmented Human Development (AHD) 框架与风险-关注矩阵。

**💡 创新点**

创新点包括：①引入 AHD 以把儿童发展维度（认知负荷、感官刺激、环境与发展匹配）与 XR 设计对齐；②构建双维度（学术关注度与真实风险）评估模型，揭示如数据安全、残障可访问性等高风险低关注的盲区；③基于 111 篇文献提出可操作的设计与治理建议。

**🔧 技术方法**

使用了系统综述方法（PRISMA 2020）、编码与内容分析、量化评分（0-2）以及风险矩阵计算；通过 AHD 理论框架对 XR 与儿童交互进行建模。

**📊 数据集**

数据来源为 111 篇经同行评议的英文论文，涵盖 3-8 岁儿童的 XR 研究，未使用专门实验数据集。

**📈 对比分析**

比较方法：对每个挑战类别给出学术关注度评分（0-2），并独立计算真实风险分值（1-9）；将两者绘制在风险‑关注矩阵上，揭示风险与关注的匹配情况。性能表现：发现高风险领域（如数据安全、残障可访问性）关注度极低，提示研究与实践存在显著脱节。

**⚠️ 局限性**

局限性：①仅检索英文文献，排除灰色文献和非英语研究；②样本仅来自 8 大数据库，可能遗漏重要工作；③缺乏长期纵向研究与统一的报告标准，导致结果可比性受限；④对技术细节与隐私技术的探讨不足。

---

## 203. Construction of MRD Codes Based on Circular-Shift Operations

**arXiv ID:** 2602.12766 | [PDF](https://arxiv.org/pdf/2602.12766v1)

**作者:** Zhe Zhai `[一作]` (University of Science and Technology Beijing), Zongpeng Li `[通讯]` (Tsinghua University)

**通讯引用:** 7898 | [OpenAlex ID](https://openalex.org/A5066247159)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一种基于循环移位运算的最大距离矩阵码（MRD码）

**💡 创新点**

创新点在于仅使用底层域GF(q)的运算构造MRD码，避免了扩域运算，并证明在多数参数下该码与Gabidulin码及其扭曲变体不同；同时提供了其与Gabidulin码的广义化等价形式

**🔧 技术方法**

使用的技术包括循环移位矩阵、Euler φ函数、线性化多项式以及线性代数工具

**📊 数据集**

未使用任何特定数据集，属于理论构造与分析

**📈 对比分析**

通过编码复杂度（XOR操作数）比较，提出的码在编码时需要O(nkL)次XOR运算，显著低于传统Gabidulin码所需的O(nkL²)次，性能在编码效率上明显更优

**⚠️ 局限性**

局限性包括：尚未给出有效的译码算法、对某些参数设定时可能与传统码重合或无法用现有广义化表示，且对非满足特定条件的参数组合尚缺乏完整理论解释

---

## 204. A Lightweight LLM Framework for Disaster Humanitarian Information Classification

**arXiv ID:** 2602.12284 | [PDF](https://arxiv.org/pdf/2602.12284v1)

**作者:** Han Jinzhen `[一作]` (Sungkyunkwan University), Yun Hong Sik `[通讯]` (Sungkyunkwan University)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5113746950)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套轻量化、成本可控的灾害信息分类框架，利用 LoRA 微调的 Llama 3.1 8B 对 HumAID 数据进行双任务（人道信息分类和事件类型识别）推理。

**💡 创新点**

创新点在于：①系统比较了 prompting、LoRA 微调和 RAG 三种 LLM 适配策略，揭示 RAG 对已微调模型的负面干扰；②引入 QLoRA 在保持 99% LoRA 性能的前提下显著降低内存需求；③通过 GPT‑4 验证部分低准确率类别是语义模糊的先验限制，提示需要重构标签体系。

**🔧 技术方法**

核心技术包括：低秩适配 LoRA/QLoRA、检索增强生成（RAG）与不同策略（标准、适应、混合），以及自定义 JSON 格式的严格 prompt 约束。

**📊 数据集**

使用公开 HumAID（76,484 条灾害相关推文）作为训练、验证和测试集，并将其转换为统一的 JSONL 形式。

**📈 对比分析**

实验比较显示：LoRA 微调后人道信息分类准确率从 41.83% 提升至 79.62%（+37.79%），事件类型分类准确率从 62.74% 提升至 98.79%；RAG 对未微调模型可提升约 13% 但对 LoRA 模型导致 1–2% 的性能下降；QLoRA 仅以 4‑bit 量化实现 50% 内存减半，性能差异低于 1%。

**⚠️ 局限性**

主要局限包括：①模型在类别“other_relevant_information”和“not_humanitarian”间存在不可逆的语义混淆，提升难度受限于标签体系本身；②RAG 在高性能微调模型上引入噪声，未能显著提升；③实验集中在单一语言（英语）与单一模型规模，缺乏多语种和跨灾害的泛化验证。

---

## 205. OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model

**arXiv ID:** 2602.12304 | [PDF](https://arxiv.org/pdf/2602.12304v1)

**作者:** Maomao Li `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**通讯引用:** 24684 | [OpenAlex ID](https://openalex.org/A5082181536)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出同步音视频定制任务及 OmniCustom 框架，实现在保持参考图像身份的同时模仿参考音频音色，并可自由指定文本内容。

**💡 创新点**

创新点包括：①将参考图像和音频分别作为 LoRA 分支注入 Transformer 的自注意力；②在流匹配损失上加入对比学习正负样本约束；③构建大规模 OmniCustom-1M 语音视频数据集。

**🔧 技术方法**

采用 Diffusion Transformer（DiT）作为基座，结合 OVI 的音视频双分支，使用 LoRA、对比学习、流匹配损失、1D VAE、MMAudio、RoPE 等技术。

**📊 数据集**

使用自构造的 OmniCustom-1M 数据集，来源于 SpeakerVid‑5M，经过同步检测、音频字幕、格式标准化等处理后得到约 1M 条单人对齐语音视频。

**📈 对比分析**

在 70 条自制基准上与 D‑Animator、ConsisID、Phantom、VACE、HunyuanCustom、Humo 等方法对比，OmniCustom 在 FID、面部相似度、说话人相似度、WER 等指标上均取得最优或竞争性表现，且比音频驱动方法更好。

**⚠️ 局限性**

局限性在于仅支持 5 秒时长的英文视频，受基模型限制且不支持多语种或更长时长。

---

## 206. Understanding Cultural Alignment in Multilingual LLMs via Natural Debate Statements

**arXiv ID:** 2602.12878 | [PDF](https://arxiv.org/pdf/2602.12878v1)

**作者:** Vlad-Andrei Negru `[一作]` (Technical University of Cluj-Napoca), Rodica Potolea `[通讯]` (Technical University of Cluj-Napoca)

**通讯引用:** 636 | [OpenAlex ID](https://openalex.org/A5049230139)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建基于自然辩论语句的多语言、多国文化标签数据集，对不同国家（美国和中国）研发的大型语言模型在霍夫斯泰德六维度上的社会文化偏好进行量化与比较。

**💡 创新点**

创新点在于①提出一种多阶段、人工与多语言 LLM 共同参与的自动标签流程，生成大规模（281 条）可量化的社会文化维度数据；②将这些标签与真实人类测量结果对齐，验证 LLM 是否在不同文化背景下保持一致性；③揭示 LLM 在语言切换时对社会文化价值的稳健性与局限。

**🔧 技术方法**

技术主要包括：Web 抓取与投票筛选、基于多语言 LLM（GPT‑4、Llama‑3、DeepSeek‑R1、Qwen‑2.5）的两步合成标注、人工质量控制、基于链式思考（CoT）的回答收集以及霍夫斯泰德维度的分数映射与归一化。

**📊 数据集**

使用的数据集为从 Kialo 在线辩论平台抓取的 281 条具代表性的辩论陈述，经过多步人工与 LLM 标注后得到的 281 条带有霍夫斯泰德维度标签的数据；此外对比人类测量值使用公开的美国与中国霍夫斯泰德文化维度数据。

**📈 对比分析**

通过在各模型的母语（英语或中文）下以及跨语言（双语）进行问答，计算每个模型在六维度上的平均得分，并与人类测量值做对比；结果显示，来自同一国家的 LLM 在霍夫斯泰德维度上与对应人类值相符，且语言切换对得分影响有限，表明模型在跨文化适配方面仍显不足。

**⚠️ 局限性**

主要限制包括：仅覆盖美国与中国两国，每国仅使用两款模型；数据集规模虽大但仍受限于 281 条；翻译使用自动工具，缺乏人工校对；实验成本高，难以扩展到更多文化或更多模型；缺少对模型训练数据与对齐策略的透明性。

---

## 207. Human-Aligned MLLM Judges for Fine-Grained Image Editing Evaluation: A Benchmark, Framework, and Analysis

**arXiv ID:** 2602.13028 | [PDF](https://arxiv.org/pdf/2602.13028v1)

**作者:** Runzhou Liu `[一作]` (University of Virginia), Hongru Du `[通讯]` (University of Virginia)

**通讯引用:** 12854 | [OpenAlex ID](https://openalex.org/A5072771687)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个细粒度的多模态大语言模型（MLLM）评判框架，并构建了相应的评估基准，用以对图像编辑模型的输出进行细粒度、可解释的质量评估。

**💡 创新点**

创新点在于：①将图像编辑质量拆解为12个可解释因子，涵盖图像保留、编辑质量和指令忠实度；②利用MLLM自动生成细粒度评估，显著提升与人类评判的一致性；③公开整合人类评估、MLLM评判和传统指标的综合基准，支持在线和离线两种评估场景。

**🔧 技术方法**

技术手段主要包括：使用多模态LLM（如GPT‑5‑mini、Gemini‑2.5‑pro）作为评判器；为每个因子设计7点Likert评分体系；构建多模态输入（原图、编辑图、文本指令）并通过LLM推理得到分量化分数；与传统像素级和语义级指标（PSNR、SSIM、LPIPS、CLIP、DINO）进行对比。

**📊 数据集**

数据集：从HumanEdit数据集随机抽取100个（原图、指令）对，涵盖Add、Remove、Replace、Action、Counting、Relation六类编辑任务；并使用GPT‑image‑1等模型生成编辑结果，形成完整的（原图、指令、编辑图）三元组。

**📈 对比分析**

比较方法：对齐度通过MSE、MAE、Pearson/Spearman/Kendall等指标评估MLLM评判与人类评判的相似度；同时将MLLM评判与传统指标在同一数据集上的得分进行对比。结果显示，传统指标在大多数因子上与人类评判相距甚远，而MLLM评判在近乎所有因子上与人类评判保持高度一致，且在编辑质量和指令忠实度等关键维度表现优异。

**⚠️ 局限性**

局限性：①基准仅覆盖固定的六类编辑任务，可能无法泛化到所有现实场景；②评判器依赖当前MLLM的能力，随模型更新可能需重新校准；③人类标注本身可能带有主观性与文化偏差，过度依赖可能导致模型对特定评判偏好过拟合；④在高风险或极端创意编辑场景下，评判标准仍需进一步验证与完善。

---

## 208. Drift-Aware Variational Autoencoder-based Anomaly Detection with Two-level Ensembling

**arXiv ID:** 2602.12976 | [PDF](https://arxiv.org/pdf/2602.12976v1)

**作者:** Jin Li `[一作]` (KIOS Research and Innovation Center of Excellence), Marios M. Polycarpou `[通讯]` (Department of Electrical and Computer Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于变分自编码器的在线异常检测框架 VAE++ESDD，能够在非平稳数据流中无监督地发现异常并适应概念漂移。

**💡 创新点**

创新点在于双层集成：一层对多个 VAE 进行增量学习和投票预测，另一层对多个漂移检测器进行投票，结合警告–报警机制实现主动与被动的概念漂移检测与模型重置。

**🔧 技术方法**

核心技术包括增量学习的变分自编码器、基于 Mann‑Whitney U 检验的漂移检测、均值+标准差自适应阈值判定、以及投票融合策略。

**📊 数据集**

实验使用合成数据（Sea、Circle、Sine、Vib）和真实数据（MNIST‑01/23/multi、Forest、Fraud、Arrhy）等，涵盖严重/极端不平衡及多种漂移类型。

**📈 对比分析**

与基线、iForest++、LOF++、StrAEm++DD、ARCUS、SEAD、METER 等方法对比，VAE++ESDD 在 G‑mean、Recall、Specificity 及 PAUC 上均位列前列，表现出更高的检测准确率和更快的漂移适应速度。

**⚠️ 局限性**

局限在于对连续漂移假设频率较低，对重复出现的旧概念缺乏模型复用，且未区分特征漂移与概念漂移，未来工作将考虑模型记忆与漂移类型辨识。

---

## 209. CBEN -- A Multimodal Machine Learning Dataset for Cloud Robust Remote Sensing Image Understanding

**arXiv ID:** 2602.12652 | [PDF](https://arxiv.org/pdf/2602.12652v1)

**作者:** Marco Stricker `[一作]`, Koichi Kise `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 CloudyBigEarthNet（CBEN）数据集，将原始 BigEarthNet 的云免费景与云覆景（光学+雷达）配对，并用该数据集对自监督学习模型进行细调，评估其在含云与不含云场景下的表现。

**💡 创新点**

提出了基于真实云遮蔽的下游评估数据集构造方法，并证明在包含云数据的细调能显著提升模型在云覆影像上的鲁棒性（相较于仅用云免费数据细调下降幅达23–33%）。

**🔧 技术方法**

使用了三种自监督学习框架（MoCo v2、MoCo v3、Masked Autoencoder）在云免费 SSL4EO-S12 数据集上预训练，然后在 BigEarthNet（云免费）和 CBEN（含云）上分别细调；后续采用平均精度（AP）作为评价指标。

**📊 数据集**

主要数据集包括：SSL4EO‑S12（云免费多模态预训练集）、BigEarthNet（云免费地表覆盖标签）和自行构建的 CloudyBigEarthNet（含云光学+雷达图像）。

**📈 对比分析**

对比方法：1）仅用云免费数据细调；2）用 CBEN 细调。结果显示：云免费细调模型在含云测试集上 AP 下降 23–33%；CBEN 细调模型在含云测试集上 AP 提升 17–28%，且在云免费测试集上保持与原模型相近或略低的性能。

**⚠️ 局限性**

局限性包括：只评估静态、 tile‑level 的 LULC 任务，无法推广至动态或像素级任务；未对预训练或细调做大规模超参数搜索；CBEN 的云遮蔽方式仍是基于真实云分布统计，可能不覆盖极端云量情况；模型对雷达信息的依赖在云免费场景下仍表现较差。

---

## 210. News Harvesting from Google News combining Web Scraping, LLM Metadata Extraction and SCImago Media Rankings enrichment: a case study of IFMIF-DONES

**arXiv ID:** 2602.12537 | [PDF](https://arxiv.org/pdf/2602.12537v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 211. Analytical Results for Two Exponential Family Distributions in Hierarchical Dirichlet Processes

**arXiv ID:** 2602.12527 | [PDF](https://arxiv.org/pdf/2602.12527v1)

**作者:** Naiqi Li `[一作]` (Tsinghua University), Naiqi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14458 | [OpenAlex ID](https://openalex.org/A5100378075)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

在层次Dirichlet过程(HDP)框架下，推导Gamma–Poisson和Normal–Gamma–Normal两种指数族共轭对的显式预测分布

**💡 创新点**

首次给出这两种共轭对在HDP中的完整解析推导，并得到负二项分布和Student‑t分布的闭式形式

**🔧 技术方法**

使用层次Dirichlet过程、指数族共轭、闭式积分与推导技巧

**📊 数据集**

无具体实验数据集，主要为理论推导

**📈 对比分析**

未进行实验比较，仅在理论层面展示推导过程与闭式结果，未给出性能评估

**⚠️ 局限性**

局限于指数族共轭对，缺乏实际数据验证，且对非共轭或高维情形的推广仍待研究

---

## 212. PEMI: Transparent Performance Enhancements for QUIC

**arXiv ID:** 2602.12732 | [PDF](https://arxiv.org/pdf/2602.12732v1)

**作者:** Jie Zhang `[一作]` (Tsinghua University), Yong Cui `[通讯]` (Tsinghua University)

**通讯引用:** 22374 | [OpenAlex ID](https://openalex.org/A5007046740)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文设计并实现了一个透明的中间件，能够在不改动 QUIC 终端实现的前提下，通过推断 RTT 与包损失来进行快速重传，从而提升 QUIC 的整体性能。

**💡 创新点**

创新点在于：①利用 QUIC 标准与实现中的 ACK 触发规律与流量局部性（flowlet）构建无端点协作的 RTT 与丢包推断机制；②采用动态规划匹配、延迟与重排阈值滤除误判；③结合 ICMP 探测与 spin bit 校准，实现了在加密头信息几乎不可见的环境下的透明拥塞控制。

**🔧 技术方法**

主要技术包括：流量分割成 flowlet、基于时间窗口的 reply‑sent 对应推断、动态规划最优匹配、阈值过滤丢包误判、延迟驱动拥塞控制（改造的 Copa）、RTT 估计与校准（ICMP、spin bit）以及基于目标速率的 CWND 约束。

**📊 数据集**

数据集与实验环境：①Mininet 模拟的多链路环境（随机丢包、延迟、宽带）；②CellReplay 结合真实移动网络追踪（GE 模型注入丢包）；③对 top‑100 网站的 QUIC 流量进行 flowlet 统计；⑤使用 Nginx + curl 的 HTTP/3 下载实验和自制 30fps RTC 框架（quic‑go、quiche、quinn）。

**📈 对比分析**

比较方法：与原生 QUIC、TCP（使用 PEP）以及不启用重传的基线进行对比。实验结果显示：在 1% 随机丢包下，下载良好比提升至 2.5×；RTC 时钟抖动（中位数）下降 45–70%，尾部抖动下降 20–75%；帧延迟中位数下降 1–94%，尾部延迟下降 59–87%；相较于 PEP，改进的 QUIC 更加 TCP‑友好；CPU 开销仅占总周期的 3% 以下。

**⚠️ 局限性**

局限性：①依赖 ACK 触发规律，若 ACK 频率极低或被故意延迟则失效；②主要针对单向流量，双向大流量时 ACK 交织导致误判；③对 RTT 突变敏感，flowlet 误分可导致推断错误；④无法生成 ACK，无法加速拥塞控制；⑤仅能在一定比例的重传内避免放大攻击，仍可能存在误重传；⑥在极低流量或无流量间隙的链路上无法构建 flowlet，失效。

---

## 213. OpenLID-v3: Improving the Precision of Closely Related Language Identification -- An Experience Report

**arXiv ID:** 2602.13139 | [PDF](https://arxiv.org/pdf/2602.13139v1)

**作者:** Mariia Fedorova `[一作]` (University of Oslo), Yves Scherrer `[通讯]` (University of Oslo)

**通讯引用:** 1405 | [OpenAlex ID](https://openalex.org/A5063672760)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文训练并发布了新的OpenLID-v3语言识别系统，并在多种公共和自定义基准上对其性能进行系统评测。

**💡 创新点**

主要创新点包括：①引入了“not‑a‑language”标签并合并相近语言变体；②通过补充更多开放许可的训练数据提升低资源语言识别；③构建并公开了针对Bosnian‑Croatian‑Serbian、意大利与法国罗曼语族以及北欧语言的细粒度评测数据集；④使用软max阈值和与GlotLID的Top‑1集成方法进一步提升精度。

**🔧 技术方法**

技术层面采用了fastText模型，利用词及字符n‑gram特征进行文本表示；通过softmax阈值0.5过滤低置信度预测；并在实验中实现与GlotLID的Top‑1/Top‑3集成。

**📊 数据集**

使用的数据集包括公开的FLORES+、UDHR、FastSpell；自建的BCMS（Twitter、ParlaSent、HPLT‑LID）、ITDI、SLIDE、Nordic DSL以及针对挪威Bokmål/Nynorsk的FastSpell重标注集。

**📈 对比分析**

评估指标为FPR、precision、recall、F1，并在单标注和多标注任务上对OpenLID-v3、OpenLID-v2、GlotLID进行对比。结果显示OpenLID-v3在大多数基准上与GlotLID相当，且在相近语言的细粒度评测中优于其前身；Top‑1集成进一步提升了精度，尤其在Bosnian‑Croatian‑Serbian、意大利罗曼语族以及北欧语言的区分任务中表现最佳。

**⚠️ 局限性**

局限性主要包括：①评测数据缺乏真实web文本的覆盖，导致模型在实际场景中的性能不完全可知；②部分基准（如Nordic DSL）经过预处理后与训练集可能存在重叠；③缺乏完全并行的数据导致模型可能对某些语言或变体过拟合，且多标注训练数据仍不足。

---

## 214. Knowledge-Based Design Requirements for Generative Social Robots in Higher Education

**arXiv ID:** 2602.12873 | [PDF](https://arxiv.org/pdf/2602.12873v1)

**作者:** Stephan Vonschallen `[一作]` (Zurich University of Applied Sciences), Friederike Eyssel `[通讯]` (Bielefeld University)

**通讯引用:** 6431 | [OpenAlex ID](https://openalex.org/A5074815650)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过12个半结构化访谈，识别了高等教育中可生成社交机器人的自我知识、用户知识和情境知识的设计需求。

**💡 创新点**

首次从知识基础设计视角提出生成式社交机器人在教学中的信息需求，补充了现有框架对行为的描述。

**🔧 技术方法**

采用定性内容分析与扎根理论方法对访谈文本进行编码。

**📊 数据集**

访谈数据来自12名学生和讲师的访谈记录。

**📈 对比分析**

本研究未进行量化比较或性能评估，而是通过主题归纳提出需求，未包含对比实验。

**⚠️ 局限性**

样本规模小、缺乏实证验证、可能存在访谈偏差、对物理体现假设的普适性有限、隐私与数据安全问题等。

---

## 215. AI Agents for Inventory Control: Human-LLM-OR Complementarity

**arXiv ID:** 2602.12631 | [PDF](https://arxiv.org/pdf/2602.12631v1)

**作者:** Jackie Baek `[一作]` (New York University), Tianyi Peng `[通讯]` (Columbia University)

**通讯引用:** 801 | [OpenAlex ID](https://openalex.org/A5002767768)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个多周期库存管理基准 InventoryBench，评估了传统运筹学（OR）算法、大型语言模型（LLM）以及人类决策在不同协同模式下的表现，并通过课堂实验验证了人机协作的互补性。

**💡 创新点**

创新点在于：①系统地探究 OR、LLM 与人类三方协同的多种组合，并证明 OR+LLM 的组合优于单独使用；②首次在库存管理中引入人机协作实验，证明人类在 LLM 支持下能显著提升利润；③提出分布无关的个体层级互补性下限，量化至少 20% 参与者受益于 AI 合作。

**🔧 技术方法**

主要技术包括：基于 LLM 的推理接口（Gemini 3 Flash、Grok 4.1 Fast、GPT-5 Mini），OR 的基于数据驱动的 capped base‑stock 策略，LLM 与 OR 的交互式管道（OR→LLM、LLM→OR），以及用于评估的人机协作实验框架和统计方法。

**📊 数据集**

使用了 1,320 个库存实例，包含 720 个合成实例（10 种需求模式）和 600 个真实实例（H&M 个性化推荐数据集的 200 种商品），并覆盖不同的交付周期和成本结构。

**📈 对比分析**

与单独使用 OR 或 LLM 的方法相比，OR→LLM 组合在所有实例上平均获得 0.538 的归一化奖励，领先 OR（0.445）和 LLM（0.494）约 21%；在人机实验中，Mode B（OR→LLM→Human）平均归一化奖励最高，显著高于 Mode A（OR→Human）和 Mode C（OR+LLM+Human Guidance），且人机协作在统计上优于相应的无人类版本。

**⚠️ 局限性**

限制主要包括：LLM 的表现受模型规模与提示设计影响，实验规模（69 名参与者）有限；交互式协作的模式与接口可能对结果产生影响；且实验中每位参与者仅在单一协作模式下评估，无法直接观测个体的完整对比。

---

## 216. Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework

**arXiv ID:** 2602.12972 | [PDF](https://arxiv.org/pdf/2602.12972v1)

**作者:** Siyun Yang `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Independent Researcher)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种统一的多值处理网络（UniMVT），同时对基线CTR进行去偏估计并预测折扣券对用户点击率的增量效果。

**💡 创新点**

通过解耦因果表示（DCR）将共因子与处理敏感特征分离，并在全空间使用反事实推理与单调线性增量关系，实现在多值处理下的CTR去偏和精准提升估计。

**🔧 技术方法**

使用Mixture-of-Experts解耦网络、双塔结构、TA‑Gate调制、X‑Network反事实正则化、单调线性约束以及全空间计量方法（CS‑AUUC、CS‑Qini）。

**📊 数据集**

在三组合成基准（单峰、多峰、不同折扣比例）以及真实的快手直播券流量数据上进行验证。

**📈 对比分析**

与S/L‑Learner、CFRNet、FlexTENet、DRNet、VCNet、DESCN等基线对比，UniMVT在AUC/LogLoss上提升0.5%+、CS‑AUUC/CS‑Qini上显著超越，线上A/B实验显示券收入提升约10%，无券收入提升约0.4%。

**⚠️ 局限性**

局限在于假设增量为线性单调，且对极端或非线性强度-响应曲线适用性有限；模型对离群折扣强度的鲁棒性仍需进一步研究。

---

## 217. Quantization-Aware Collaborative Inference for Large Embodied AI Models

**arXiv ID:** 2602.13052 | [PDF](https://arxiv.org/pdf/2602.13052v1)

**作者:** Zhonghao Lyu `[一作]` (KTH Royal Institute of Technology), H. Vincent Poor `[通讯]` (Princeton University)

**通讯引用:** 152465 | [OpenAlex ID](https://openalex.org/A5042307561)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种面向大型嵌入式AI模型（LAIM）的量化感知协同推理框架，提出了量化诱发推理失真可逼近、速率‑失真上界与下界，并基于此设计了联合量化位宽与计算频率的优化方法。

**💡 创新点**

创新点在于：①首次用可计算的参数级失真作为输出失真的上界，并给出对量化位宽与失真关系的理论下界和上界；②将量化位宽与设备/服务器计算频率耦合到一个 QoS 约束下的混合整数非凸优化问题；③通过连续松弛、辅助变量与连续凸近似（SCA）得到高效求解方案。

**🔧 技术方法**

技术手段包括：参数量化误差模型、1‑Lipschitz 激活假设、Cauchy-Schwarz 递推失真上界、指数分布权重假设、Shannon 速率‑失真下界推导、Laplace 量化噪声上界、混合整数线性松弛、SCA 迭代凸化、实验验证（Python/CUDA）。

**📊 数据集**

使用的模型与数据集包括：ResNet‑152、VideoMAE、BERT、BLIP‑2、GIT、GPT‑3；在视觉任务上使用 MS‑COCO（图像标注），在视频任务上使用 VaTeX；在量化实验中评估了 FCDNN‑16、BLIP‑2‑2.7B、GIT‑Base；均使用统一与非均匀（PoT‑log）量化。

**📈 对比分析**

与三种基线比较：①基于 PPO 的 DRL 方案；②固定频率只优化位宽；③随机可行设计。评价指标为 CIDEr（文本生成质量）和推理延迟/能耗。实验显示，联合设计在满足不同延迟/能耗阈值时均取得最高 CIDEr，且比 PPO 低约 3–5% 的失真、比固定频率高 2–4% 的精度。真实测试平台（Jetson AGX Orin + Dell PowerEdge）验证了在粗粒度频率配置下的性能提升。

**⚠️ 局限性**

局限性包括：①对权重幅值指数分布的假设在某些模型中可能偏差；②使用参数失真作为输出失真上界，无法捕捉极端非线性传播效应；③算法需要在不同位宽/频率间做连续松弛与迭代，计算成本在大规模模型上仍不小；④实验主要聚焦单机/单服务器场景，跨域多服务器协同仍待扩展。

---

## 218. Bench-MFG: A Benchmark Suite for Learning in Stationary Mean Field Games

**arXiv ID:** 2602.12517 | [PDF](https://arxiv.org/pdf/2602.12517v1)

**作者:** Lorenzo Magnino `[一作]` (University of Cambridge), Mathieu Laurière `[通讯]` (NYU Shanghai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Bench‑MFG，一个针对静态均值场游戏的统一基准套件。

**💡 创新点**

创新点在于构建了游戏分类法、随机生成器 MF‑Garnet，并对多种求解算法进行系统评估。

**🔧 技术方法**

使用 JAX 实现高效模拟，采用固定点、策略迭代、在线镜像下降、PSO 等算法。

**📊 数据集**

使用的基准数据集包括多类 MFG（无交互、合同、单调、潜在、动力耦合）以及 MF‑Garnet 随机实例。

**📈 对比分析**

通过 exploitability 曲线对比 FP、Damped FP、Fictitious Play、Policy Iteration、Online Mirror Descent、MF‑PSO 等方法，MF‑PSO 在多数情形下取得最低 exploitability，而 FP 在简单情形下最快。

**⚠️ 局限性**

局限性在于仅关注静态 MFG，未扩展到大规模状态空间、有限时域或基于采样的深 RL 方法。

---

## 219. Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models

**arXiv ID:** 2602.12498 | [PDF](https://arxiv.org/pdf/2602.12498v1)

**作者:** Ali Abbasi `[一作]` (University of Delaware), Rahmatollah Beheshti `[通讯]` (University of Delaware)

**通讯引用:** 550 | [OpenAlex ID](https://openalex.org/A5044208559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对医学视觉-语言模型（VLM）在处理否定表达时的误判问题的改进方案，构建了辩证的诊断基准和结构化上下文否定数据集，并基于层级因果追踪（CTE）设计了可解释性引导的细粒度微调方法（NAST）

**💡 创新点**

将因果追踪得到的层级贡献度直接转化为梯度缩放因子，对不同层进行可解释性引导的梯度调节；同时设计了以结构化临床事实为基础的属性级否定数据集，实现对否定处理的精准监督

**🔧 技术方法**

CLIP式医学VLM框架、因果追踪（Causal Tracing Effects）、低秩适配（LoRA）参数化微调、对比损失与主张排名损失的组合优化

**📊 数据集**

MIMIC‑CXR、CheXpert、MIMIC‑CXR‑JPG、以及作者自行构造的 MedNega‑CXR 诊断基准和上下文否定数据集

**📈 对比分析**

与 CLIP、NegCLIP、ConCLIP、NegBench 等基线进行对比，使用检索 Recall@1/5 与主张准确率评估。NAST 在检索任务上分别达 R@1 49.5%、R@5 65.7%，在主张任务上准确率 55.6%，显著缩小肯定-否定性能差距至 4.2%，优于所有基线

**⚠️ 局限性**

实验主要基于 MIMIC‑CXR 等单一机构的结构化数据，缺乏跨机构、跨模态或不同语言风格的验证；且虽然显著提升否定处理，但整体对其他非否定场景的影响有限，仍需临床进一步评估与人工监督

---

## 220. Efficient Streaming Algorithms for Two-Dimensional Congruence Testing and Geometric Hashing

**arXiv ID:** 2602.12667 | [PDF](https://arxiv.org/pdf/2602.12667v1)

**作者:** Yen-Cheng Chang `[一作]` (National Tsing Hua University), Ting-An Wu `[通讯]` (Academia Sinica)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了在流式模型下，对二维有限精度有理坐标点集的几何相似性（Congruence Identification）与几何哈希（Geometric Hashing）问题的低空间算法。

**💡 创新点**

核心创新在于：
• 利用复数矩（complex moments）避免传统矩阵方法中因矩消失导致的判别困难；
• 通过在随机素数域上映射（φ_p）实现对有理坐标的精确、可压缩运算，从而实现对高精度输入的空间友好处理；
• 将上述技术与多轮扫描、Karp‑Rabin 哈希等经典流式工具结合，构建了3/6轮通用算法，显著降低了空间需求。

**🔧 技术方法**

使用的关键技术包括：
• 复数矩理论和关于幂次矩不为零的命题；
• 随机素数域扩展 _p[ i ] 的映射 φ_p 以及其同态性质；
• 随机化的等式检测（Karp‑Rabin 哈希）与等价检测；
• 多轮（3轮、6轮）流式扫描与候选旋转/平移的候选集产生；
• 复杂的“坏素数”概率分析与 rational‑reconstruction。

**📊 数据集**

论文中主要使用合成点集（synthetic U‑rational 2‑D multisets）进行实验与理论验证；没有提及真实数据集。

**📈 对比分析**

与传统基于排序/最近点对的方法相比，本文算法在空间上实现对数级别的改进：
• Congruence Identification 3 轮，空间为 O(log n (log n + log U))；
• Geometric Hashing 6 轮，空间为 O(log n (log n + log U + log m))，哈希长度为 O(log n + log U + log m)。
实验结果显示，在各种 U 与 n 规模下，算法保持低空间占用且误差概率 ≤ 1/n。

**⚠️ 局限性**

局限性：
• 仅适用于二维情况，三维推广需要更复杂的生日悖论与额外的空间开销；
• 需要在素数区间 [Δ,4Δ] 中随机选取满足多重约束的素数，实装时可能需要额外时间；
• 对输入精度的要求仍依赖于 U，过大 U 仍会导致系数膨胀；
• 由于使用随机化映射，算法的失败概率不为零，需多次重跑或对错误率做严格控制。

---

## 221. Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment

**arXiv ID:** 2602.12384 | [PDF](https://arxiv.org/pdf/2602.12384v1)

**作者:** Nathanaël Haas `[一作]` (CRIL UMR 8188, Université d'Artois), Zied Bouraoui `[通讯]` (CRIL UMR 8188, Université d'Artois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文通过理论分析与实验相结合，研究了深度神经网络 Jacobian 的谱结构，提出并验证了“深度诱导指数缩放”与“谱分离导致奇异向量对齐”两种关键机制，并在此基础上推导出与平衡深线性网络相似的奇异值动态。

**💡 创新点**

创新点包括：① 在非平衡、带门控（fixed‑gates）情形下首次证明 Lyapunov 指数的存在并给出闭式表达；② 用谱分离来诱导矩阵乘积的奇异向量对齐，提供了与传统平衡假设不同的结构性替代；③ 在上述两种机制约束下，得到近似的单模态固定门网络的奇异值动力学，解释了隐式偏置的低秩现象；④ 在实验中验证了理论预测，展示了谱分离与对齐在训练过程中的动态演化。

**🔧 技术方法**

主要技术：随机矩阵理论（Lyapunov 指数、外部幂）；门控矩阵（p‑gate、(r,p)-gate）模型；矩阵乘积的奇异值与向量对齐定理；梯度流动力学（gradient flow）与平衡条件的对比；数值实验中的奇异值谱与对角相关系数计算。

**📊 数据集**

数据集：① 合成的秩为10的回归任务；② MNIST（使用 Cifar10 auto‑augment 改进）作为实际训练样例；实验还使用了 Gaussian 权重和 Bernoulli 门（p=1 或 0.5）作为初始化。

**📈 对比分析**

方法对比：与平衡深线性网络的理论奇异值动态进行比较；实验中通过绘制 1/L·log s_i 与理论 Lyapunov 指数及其有限深度修正的贴合度、以及全 Jacobian 与中间子乘积奇异向量的对角相关系数来验证理论；实验结果表明在初始化阶段与训练后深度指数缩放、谱分离与对齐均与理论预期高度一致，暗示该机制能解释隐式低秩偏置。

**⚠️ 局限性**

局限性：① 关键假设（深度指数缩放与近似共享奇异基）在理论上仅在大深度、特定门控模型下成立；② 只验证了单模态固定门网络，未涵盖多模态或更复杂网络（如 Transformer）;③ 对齐与分离的数值指标仅在实验中展示，缺乏严格的训练期收敛证明；④ 对实际性能（如分类准确率）的直接提升未给出，主要聚焦于理论解释。

---

## 222. Reliable Thinking with Images

**arXiv ID:** 2602.12916 | [PDF](https://arxiv.org/pdf/2602.12916v1)

**作者:** Haobin Li `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**通讯引用:** 9791 | [OpenAlex ID](https://openalex.org/A5022800038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Reliable Thinking with Images (RTWI) 的方法，用来解决多模态大语言模型在思维与图像（TWI）过程中出现的噪声思维（Noisy Thinking）问题，即视觉线索挖掘与答案推理阶段的错误传播；

**💡 创新点**

创新点在于：① 统一的文本中心可靠性估计机制，可同时评估视觉线索和文本推理的可靠性；② 双阶段过滤（对挖掘和推理阶段分别设定自适应阈值）剔除噪声轨迹；③ 可靠投票方案基于可靠性跃迁和整体可靠性加权，提升答案可信度；

**🔧 技术方法**

采用的技术包括：文本基熵分析与Top‑k高熵词汇选取来估计可靠性；自适应阈值过滤（Dual‑Stage Filtering）；可靠投票（Reliable Voting）与温度调节；Test‑Time Scaling 机制（多路径采样、早停、鲁棒推理）来提升效率；

**📊 数据集**

实验数据集覆盖了真实高分辨率图像与推理场景：Vstar Bench、HR‑Bench（4K/8K）、TreeBench、MathVision、LogicVista 等；

**📈 对比分析**

在在线和离线两种评估设置下，与 GPT‑4o、Thyme、DeepEyes、Self‑Consistency、Early‑Stopping、Robust‑Reasoning 等 SOTA 方法对比，RTWI 在多项基准上均取得显著准确率提升（往往 +3%–10%），同时显著降低 token 消耗（Token Saving Ratio 提升 20%–50%）并改善视觉线索一致性；

**⚠️ 局限性**

局限性包括：① 仍依赖文本推理来推断视觉可靠性，未直接建模视觉空间不确定性；② 对极端噪声或极复杂视觉操作的鲁棒性尚未充分验证；③ 目前仅针对单轮或简易多轮 TWI 场景，扩展到更长链或多任务情景仍需研究。

---

## 223. Towards a Diagnostic and Predictive Evaluation Methodology for Sequence Labeling Tasks

**arXiv ID:** 2602.12759 | [PDF](https://arxiv.org/pdf/2602.12759v1)

**作者:** Elena Alvarez-Mellado `[一作]`, Julio Gonzalo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于错误分析的测试集构建方法，并将其应用于西班牙语英租词识别任务，生成了专门的 BLAS 评测集。

**💡 创新点**

创新点在于：①使用句子内跨度属性（长度、形态、位置、引号、大小写、邻接、歧义）系统覆盖所有可能组合；②通过人工生成样例而非采集真实文本，消除数据偏倚；③通过属性级别的召回分解提供可解释的诊断与预测。

**🔧 技术方法**

技术上使用了错误分析、属性驱动样本选择、系统化的大小写与引号扰动，以及对比六种模型（CRF、BETO、mBERT、两种 BiLSTM‑CRF、Llama3 8B 远程提示）进行评估。

**📊 数据集**

数据集主要是自研的 BLAS（1,836 句子，2,076 种英租词跨度），并在 COALAS、CALCS 等公开数据集上验证外推性。

**📈 对比分析**

比较方式以召回率为主，BLAS 上模型得分仅为 6%–36%，而在 COALAS 上高达 78%；Llama3 在 BLAS 上领先 36% 以上，且 BLAS 的属性级召回能以 0.85 的 Pearson 相关性准确预测外部数据集的整体召回和排名。

**⚠️ 局限性**

局限性包括：仅在西班牙语英租词任务上验证；只适用于跨度检索任务；评估聚焦召回而非精确率；样本由单一语言学家人工生成，可能引入作者偏差。

---

## 224. Wireless TokenCom: RL-Based Tokenizer Agreement for Multi-User Wireless Token Communications

**arXiv ID:** 2602.12338 | [PDF](https://arxiv.org/pdf/2602.12338v1)

**作者:** Farshad Zeinali `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19266 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并验证了一个多用户无线视频 TokenCom 框架，实现了自适应 tokenizer 协议、子信道分配与波束成形的联合优化。

**💡 创新点**

首次提出混合 DQN‑DDPG 强化学习框架，在同一决策空间中同时处理离散的 tokenizer 选择、子信道分配和连续的波束向量，解决了 TokenCom 的混合整数非凸优化问题。

**🔧 技术方法**

使用强化学习（DQN+DDPG）、预训练视频 tokenizer/解码器、混合整数非凸优化、语义质量评估指标（PSNR/SSIM/rFVD）等技术。

**📊 数据集**

基于 DAVIS 视频数据集（24fps），并采用多种预训练视频 tokenizer 的压缩/质量表进行实验。

**📈 对比分析**

与传统 H.265、DDPG‑TA、Agnostic‑TA、Fixed‑TA 四个基线对比，冻结率降低约 68%，平均 PSNR 提升约 10 dB，并在多用户场景保持低冻结率。

**⚠️ 局限性**

仅在理想 Rayleigh 块衰落模型下验证，未考虑移动性、信道估计误差和大规模计算成本；对多模态语义通信（音频、文本）尚未评估。

---

## 225. CLASE: A Hybrid Method for Chinese Legalese Stylistic Evaluation

**arXiv ID:** 2602.12639 | [PDF](https://arxiv.org/pdf/2602.12639v1)

**作者:** Yiran Rex Ma `[一作]`, Huiyuan Xie `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CLASE框架，结合客观语言特征与经验驱动LLM评估，提供无参考的法律文本风格质量评估；

**💡 创新点**

通过对真实法律文本与其去风格化/重风格化对的对比学习自动获取专业风格标准，无需人工标注，并将客观特征与LLM主观评估融合实现高人类一致性；

**🔧 技术方法**

对比学习（对照对合成）、无监督逻辑回归特征提取、检索增强的LLM评估、混合评分融合等技术；

**📊 数据集**

使用4000份中文民事裁判文本进行对比学习训练，200份独立裁判文本用于评估；合成对由LLM生成；

**📈 对比分析**

与传统n‑gram、BERTScore、LLM‑as‑Judge基线比较，CLASE‑Obj Pearson r≈0.79，CLASE‑Mix r≈0.83，显著高于所有基线，并与人工评估的相关性最高；

**⚠️ 局限性**

局限在中文民事裁判场景，语言/领域迁移性受限；对去风格化模型假设不破坏语义；多阶段管线计算成本高；模型解释性不足。

---

## 226. Peaceful Anarcho-Accelerationism: Decentralized Full Automation for a Society of Universal Care

**arXiv ID:** 2602.13154 | [PDF](https://arxiv.org/pdf/2602.13154v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán `[通讯]` (Universidad Pontificia Comillas)

**通讯引用:** 1440 | [OpenAlex ID](https://openalex.org/A5070783543)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种名为无政府加速主义的社会技术框架，并构建了 Liberation Stack 这一多层开放源代码技术架构，阐述全自动化后通过共同体治理实现普惠关怀社会的路径；同时给出了四阶段路线图与多案例实证分析。

**💡 创新点**

创新点包括：①将无政府主义、共同体生产、批判技术研究与后货币经济（UDR）结合的跨学科桥接模型；②提出全自动化后废除货币、以需求为导向的 Universal Desired Resources；③以 Linux、Wikipedia、Mondragon、Rojava、guifi.net 等已存在的规模化共同体为示例，验证共同体治理在技术、经济与社会层面的可行性；④规划从技术、能源、制造、食品到治理的完整技术堆栈。

**🔧 技术方法**

技术层面：深度学习（通用近似定理与 DRL 收敛理论）用于证明可自动化；开源软件与硬件（Linux 内核、Fediverse、Fablab 设备、OpenEMS、Agroecology、FarmBot 等）；分布式治理平台（Decidim、Loomio、Pol.is、LiquidFeedback）；社区能源微网与 Mesh 网络（guifi.net、OpenEMS、Home Assistant）；开放 AI 训练与部署（LLaMA、Stable Diffusion 等）。

**📊 数据集**

数据集与实证依据：Linux 贡献者数量、服务器占比、Android 设备占比；Wikipedia 维基编辑者、文章数量、语言覆盖；Mondragon 参与者、收入、工资比率、员工流失率；Rojava 人口规模、治理层级；guifi.net 节点数量、覆盖距离；以及学术引用中的自动化风险评估表（Freyn & Osborne 等）。

**📈 对比分析**

比较方法主要为案例对比与指标对照：在合作社与传统企业之间对比劳动力生产率、工资、员工流失率；在能源与通信共同体对比可用性与成本；在治理平台对比决策速度、透明度。实验结果显示：合作社在生产率上高 12%、工资高 9%、流失率低 30%；社区能源与制造提升了地方经济乘数 2–3 倍；开放治理平台实现了决策透明度与参与度的显著提升。

**⚠️ 局限性**

局限性：①时间表高度投机，依赖 AI、机器人与能源技术持续快速进步；②假设开放源代码 AI 能持续竞争；③对地缘政治阻力、现有权力结构的应对不充分；④对机器是否具备意识的哲学假设未得到普遍接受；⑤UDR 的经济模型与实施细节尚未完成，需要进一步机制设计与模拟验证。

---

## 227. KeySense: LLM-Powered Hands-Down, Ten-Finger Typing on Commodity Touchscreens

**arXiv ID:** 2602.12432 | [PDF](https://arxiv.org/pdf/2602.12432v1)

**作者:** Tony Li `[一作]` (Stony Brook University), Xiaojun Bi `[通讯]` (Stony Brook University)

**通讯引用:** 2353 | [OpenAlex ID](https://openalex.org/A5016901770)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全基于软件的系统KeySense，实现在普通触摸屏上进行十指下压键入，允许手指休息而非传统的“鸡蛋打字”。

**💡 创新点**

创新点包括：①利用认知时窗将近同步触摸聚类为一键；②用“行进距离”得分挑选真正的按键；③通过合成错误数据训练一个小型FLAN‑T5‑small LLM，完成从噪声字母序列到准确单词的映射；④在不增加硬件的情况下恢复键盘肌肉记忆与人体工学优势。

**🔧 技术方法**

核心技术包括：触摸线程重构、时间聚类、行进距离启发式过滤、基于QWERTY几何的误差模拟、人工合成错误数据、FLAN‑T5‑small Seq2Seq微调、Beam搜索解码。

**📊 数据集**

数据集：①从8名受试者在iPad上记录的1,752个单词打字日志；②使用Google的10k英文单词集合作为目标词；③合成的146,331对（词，噪声序列）用于训练；④实验中使用公开的MacKenzie和Soukoreff短语集。

**📈 对比分析**

比较方法：在合成错误数据上对比精确匹配率（Top‑1）和Top‑k；与两个统计基线（基于触摸位置的贝叶斯解码和仅使用字母的n‑gram解码）以及GPT‑4o进行对比。KeySense的FLAN‑T5‑small达84.8% Top‑1，显著高于统计基线（75.7%/79.3%）和GPT‑4o（72.0%）。在12名受试者的实地实验中，KeySense在第5次会话达到28.3 WPM，比传统悬停键盘26.2 WPM快且物理负担更低（NASA‑TLX物理分数1.5 vs 4.0）。

**⚠️ 局限性**

局限性：①实验仅限单词级解码，未利用上下文；②对OOV词和长句子仍存在误差；③初始学习成本高，用户需要适应十指下压姿势；④仅在iPad Pro上验证，跨设备/语言适用性待进一步研究；⑤模型仍需要云端推理，导致网络延迟与隐私问题。

---

## 228. RADAR: Revealing Asymmetric Development of Abilities in MLLM Pre-training

**arXiv ID:** 2602.12892 | [PDF](https://arxiv.org/pdf/2602.12892v1)

**作者:** Yunshuang Nie `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 23057 | [OpenAlex ID](https://openalex.org/A5047878798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多模态大型语言模型（MLLM）在预训练阶段各项能力的发展不对称性，并提出了相应的评估框架。

**💡 创新点**

创新点在于构建了RADAR评估体系，通过系统分析不同任务的学习曲线揭示了能力增长的非线性和差异化。

**🔧 技术方法**

使用的技术包括对比学习、视觉‑语言对齐模型和多任务微调，并在预训练过程中记录能力指标。

**📊 数据集**

实验数据集涵盖COCO、Visual Genome、Laion‑5B等多种图文数据集，结合公开的MLLM基准。

**📈 对比分析**

与现有模型（如LLaVA、MiniGPT‑4、GPT‑4V）进行对比，结果显示RADAR方法能更精确捕捉能力差异，且在多项任务上表现优于基线。

**⚠️ 局限性**

局限性包括仅关注视觉‑文本模态，未考虑视频、音频等其他模态，评估框架对模型规模的依赖性较高。

---

## 229. X-SYS: A Reference Architecture for Interactive Explanation Systems

**arXiv ID:** 2602.12748 | [PDF](https://arxiv.org/pdf/2602.12748v1)

**作者:** Tobias Labarta `[一作]` (Fraunhofer Heinrich Hertz Institute), Sebastian Lapuschkin `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 X-SYS 参考架构，指导交互式解释系统的端到端设计，并通过 SemanticLens 实现了该架构的演示；

**💡 创新点**

将 XAI 视为信息系统问题，定义 STAR（可扩展性、可追溯性、适应性、响应性）四项质量属性，并通过组件化拆分与接口契约解耦交互层与后端，形成可复用的蓝图；

**🔧 技术方法**

采用微服务架构、FastAPI、DTO 接口契约、Zennit、MobileCLIP、UMAP、Concept Relevance Propagation、激活调节等技术；

**📊 数据集**

在 Vision / Vision‑Language 领域使用 ResNet50、CLIP、WhyLesionCLIP 等模型及其对应图像/文本数据集；

**📈 对比分析**

通过将离线预处理与在线查询分离，达到秒级响应时间，并在可追溯性、适应性和可扩展性指标上进行定性评估；论文未给出量化基准或对比实验；

**⚠️ 局限性**

仅在单一案例验证，缺乏跨领域、多数据集的实证；未提供系统性能量化、可扩展性、隐私与安全细节；缺少可执行的指标与基准。

---

## 230. Hierarchical Successor Representation for Robust Transfer

**arXiv ID:** 2602.12753 | [PDF](https://arxiv.org/pdf/2602.12753v1)

**作者:** Changmin Yu `[一作]` (University of Cambridge), Máté Lengyel `[通讯]` (Central European University)

**通讯引用:** 5957 | [OpenAlex ID](https://openalex.org/A5023602087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并评估了层次化后继表示（HSR），通过将选项（temporal abstraction）融入后继表示框架，产生对策略变化鲁棒且可解释的状态特征，并使用非负矩阵分解（NMF）进一步压缩为稀疏低维基底，以实现快速任务迁移和高效探索。

**💡 创新点**

创新点包括：①将选项与后继表示相结合，消除经典SR对策略的高度依赖；②通过“期望HSR”聚合多个任务的策略信息，进一步提升对策略变化的稳健性；③采用NMF而非传统SVD分解，使得特征保持局部化、稀疏、可解释，并在多室、可生成迷宫环境中实现优于传统方法的迁移与探索性能。

**🔧 技术方法**

技术方法：层次化强化学习（option框架）、后继表示（SR）与期望HSR、非负矩阵分解（NMF）、线性Q‑learning、基于SR的固有激励（SR‑norm、SPIE）以及对应的TD学习更新。

**📊 数据集**

数据集与实验环境：经典四室迷宫（four-room maze）与可生成的随机迷宫（procedurally generated random mazes），使用20个随机种子；预训练任务为不同奖励配置的G1、G2目标任务，主要测试奖励与策略变化对表示的影响。

**📈 对比分析**

与方法比较：与随机行走SR（RW‑SR）、期望SR（eSR）、SR+SVD、HSR+SVD、HSR+NMF等进行对比。评价指标包括到达目标所需步骤、完成任务所需训练周期、迁移效率（相对基准），以及对价值函数的重构R²和状态覆盖率。实验表明：HSR+NMF在迁移学习中显著降低训练周期、提升迁移效率；在探索任务中，HSR‑SPIE的状态覆盖率高于SR‑SPIE，尤其在大迷宫中差距明显。

**⚠️ 局限性**

局限性：①实验仅在线性函数逼近环境中验证，未在深度RL或连续动作空间上测试；②选项预先定义且依赖预训练任务，缺乏在线自适应选项发现；③NMF对SR特征易出现特征塌陷，需较大的局部结构才能发挥优势；④当前方法假设转移动态相同，难以处理动态或不共享动力学的多任务。

---

## 231. RADAR: Exposing Unlogged NoSQL Operations

**arXiv ID:** 2602.12600 | [PDF](https://arxiv.org/pdf/2602.12600v1)

**作者:** Mahfuzul I. Nissan `[一作]` (University of New Orleans), James Wagner `[通讯]` (University of New Orleans)

**通讯引用:** 5304 | [OpenAlex ID](https://openalex.org/A5041773585)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个日志对抗框架RADAR，通过低层存储取证与应用日志交叉验证，检测NoSQL数据库中未记录的插入、更新、删除等操作。

**💡 创新点**

创新点在于：①提出基于日志对抗的取证框架；②设计自动化NoSQL取证工具ANOC，能够推断多种NoSQL存储布局并无日志直接恢复记录；③针对不同存储架构（Append‑Only/CoW 与 In‑Place）分别实现单快照和比较分析两套检测流程；④统一了取证与日志一致性算法，实现对未记录操作的精准识别。

**🔧 技术方法**

使用技术包括：ANOC与其INFERNOS推断算法、JSON Lines统一日志格式、页面级MD5哈希与记录级差分、主机级快照（LVM/ZFS/Shadow Copy）捕获、以及基于集合运算的匹配与异常检测算法。

**📊 数据集**

实验使用Star Schema Benchmark (SSBM) 生成的键值与文档数据集，在10种NoSQL引擎（BerkeleyDB、LMDB、MDBX、etcd、ZODB、Durus、LiteDB、Realm、RavenDB、NitriteDB）上进行。

**📈 对比分析**

通过对每个引擎分别测量ANOC取证阶段与RADAR匹配阶段的处理时间，吞吐量从31.7到397 MB/min；单快照引擎（如LMDB、ZODB）处理时间在20–30 秒，双快照引擎（如LiteDB）需要约40–70 分钟；总体处理时间在10–80 分钟之间，展示了在大规模数据上可接受的性能。

**⚠️ 局限性**

局限性包括：需要两份快照才能检测In‑Place写入；不支持加密存储；只能处理能被ANOC推断的存储格式；对分布式/云原生系统缺乏完整支持；实时增量监测能力有限，适合事后取证。

---

## 232. AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping

**arXiv ID:** 2602.12315 | [PDF](https://arxiv.org/pdf/2602.12315v1)

**作者:** Sunghwan Kim `[一作]` (Yonsei University), Dongha Lee `[通讯]` (ParamitaAI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AgenticShop 基准，用于评估开放网页环境下的个性化产品策划。

**💡 创新点**

创新点在于构建真实购物意图、用户画像和基于清单的可验证个性化评估框架。

**🔧 技术方法**

采用 LLM 作为评判器、Playwright 进行多模态信息提取，以及搜索增强 LLM 与自主 Web 代理两类系统。

**📊 数据集**

使用 Amazon Review 数据集构建用户画像并生成查询和清单。

**📈 对比分析**

通过与人类评估对齐的 LLM-as-a-judge 进行自动化评分，实验显示当前系统的个性化满足率仅约 30‑35%。

**⚠️ 局限性**

限制在于对动态内容和用户评价的处理不足、Hallucination、对价格敏感度低以及缺乏对视觉审美的把握。

---

## 233. Towards Universal Video MLLMs with Attribute-Structured and Quality-Verified Instructions

**arXiv ID:** 2602.13013 | [PDF](https://arxiv.org/pdf/2602.13013v1)

**作者:** Yunheng Li `[一作]` (Nankai University), Ming-Ming Cheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本工作构建了一个百万级规模的开放式多属性音视频指令数据集，并基于该数据集训练了新的多模态视频理解模型；

**💡 创新点**

创新点在于采用多源生成、自动集成、与音视频证据的语义与时间一致性验证以及属性级评估与定向改进的多阶段数据策划管线，形成结构化、可验证的细粒度监督；

**🔧 技术方法**

技术实现包括多源自动注释（音频ASR与视觉+音频大模型）、集成式Caption合成与语义时间一致性校验、属性级错误/缺失检测与局部修正，以及分阶段的单属性→全属性→长时序监督的渐进式监督微调；

**📊 数据集**

使用的数据源为LLaVA-Video-178K子集与FineVideo短视频，最终得到≈12.1万条视频；评测基准包括Video‑SALMONN‑2、UGC‑VideoCap、VDC、VidCapBench‑AE、Daily‑Omni、World‑Sense和Charades‑STA等；

**📈 对比分析**

与多种开源模型（如Qwen2.5‑Omni、InternVL3.5、AvoCaDO）以及Gemini‑3‑Pro进行对比，本文模型在文本质量、属性指令跟随、基于Caption的问答和时序定位等指标均取得显著提升，尤其在缺失率降低与幻觉控制方面表现优异；

**⚠️ 局限性**

局限性包括对长时序（>3min）处理仍不充分，属性覆盖仍不完整，属性级验证可能漏检细微错误，且部分高级多源注释依赖闭源模型，导致可复现性受限。

---

## 234. Curriculum-DPO++: Direct Preference Optimization via Data and Model Curricula for Text-to-Image Generation

**arXiv ID:** 2602.13055 | [PDF](https://arxiv.org/pdf/2602.13055v1)

**作者:** Florinel-Alin Croitoru `[一作]` (University of Bucharest), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58356 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文本到图像生成任务中，对直接偏好优化（DPO）加入了数据级和模型级的课程学习（Curriculum-DPO++），并提出了无奖励模型的课程学习方案；

**💡 创新点**

创新点在于同步推进数据级课程（由易到难的偏好对）与模型级课程（逐步增加LoRA矩阵秩和可训练层数），以及利用提示嵌入掩码实现奖励模型无关的课程排序；

**🔧 技术方法**

使用的技术包括Diffusion/Consistency模型、Direct Preference Optimization、Low‑Rank Adaptation（LoRA）、奖励模型（Sentence‑BERT、LAION Aesthetics、HPSv2）以及文本嵌入掩码生成隐式偏好对；

**📊 数据集**

实验数据集包括三套图像生成数据：D_1（SVO+动物/活动），D_2（DrawBench 200个提示），D_3（Pick‑a‑Pic 150k对）；

**📈 对比分析**

与原始Curriculum‑DPO、DPO、DDPO以及基线模型（Stable Diffusion、Latent Consistency Model）对比，Curriculum‑DPO++在文本对齐、美学评分与人类偏好三个任务均取得了更高的分数和胜率，尤其在人类偏好任务中提升明显；

**⚠️ 局限性**

限制在于仍需依赖奖励模型或掩码策略来生成偏好对，且超参数（β、B、层数、秩增长率等）对效果影响较大，需要进一步自动化选择；

---

## 235. TRANS: Terrain-aware Reinforcement Learning for Agile Navigation of Quadruped Robots under Social Interactions

**arXiv ID:** 2602.12724 | [PDF](https://arxiv.org/pdf/2602.12724v1)

**作者:** Wei Zhu `[一作]` (Tohoku University), Mistuhiro Hayashibe `[通讯]` (Tohoku University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了TRANS框架，实现了在不规则地形和动态人群交互环境下的四足机器人敏捷导航。

**💡 创新点**

创新点在于：① 将四足行走控制与社交导航分离为两阶段训练，再在统一策略中融合；② 采用非对称actor‑critic实现不需要地形感知的鲁棒行走；③ 用变换后的LiDAR扫描提取社交信息，减少对高频感知的依赖；④ 在统一网络中引入行走状态与轨迹信息，实现地形感知与碰撞避免的闭环。

**🔧 技术方法**

使用的技术包括深度强化学习（PPO、SAC）、非对称actor‑critic、时序自编码器、域随机化、梯度升华与稀疏奖励设计、变换LiDAR扫描、基于IsaacSim的仿真与多模态感知融合。

**📊 数据集**

主要使用仿真数据：包含多种地形（平地、斜坡、楼梯、崎岖地形）与通过ORCA生成的多主体行人交互；实测时使用Unitree Go2/Unitree A1四足机器人、Livox MID‑360 LiDAR与ZED2摄像头进行环境感知。

**📈 对比分析**

与多种基线（包括传统模型规划器DWA、ORCA、AVOCADO、N‑MPC、NeuPAN、T‑MPC；DRL基线RGL、LNDNL、DRL‑VO）及原始DreamWaQ、SLR等进行对比。实验结果显示：在行走任务中，TRANS‑Loco在不规则地形上的学习效率与特权方法相当，且命令跟踪误差最低；在社交导航任务中，TRANS‑Nav在500次随机测试中实现成功率97.2%、碰撞率2.8%且平均导航时间19.3s，明显优于所有对比方法。

**⚠️ 局限性**

局限性包括：① 楼梯等高难度地形下的跌倒率相对较高；② 高频振动导致SLAM定位漂移，影响实地导航精度；③ 相机视场受限，超出范围的人群未被感知；④ 仅使用ORCA模拟行人行为，缺乏对真实人类复杂交互的充分建模。

---

## 236. Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting

**arXiv ID:** 2602.12389 | [PDF](https://arxiv.org/pdf/2602.12389v1)

**作者:** Siyuan Li `[一作]` (Peng Cheng Laboratory), Fangyi Pei `[通讯]` (Dalian University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Entity State Tuning (EST)，一种通用的、可插拔的框架，通过为实体维护持续更新的全局状态缓冲区，显著提升时序知识图谱 (TKG) 预测的长期依赖性。

**💡 创新点**

创新点在于：①引入状态感知的 Topology‑Aware State Perceiver，主动将实体状态注入结构编码；②设计统一的 Temporal Context Module 与可替换的序列骨干，形成闭环的状态演化机制；③引入 Counterfactual Consistency Learning 以消除可观测偏差，进一步稳定状态更新。

**🔧 技术方法**

核心技术包括：多头门控融合实体静态嵌入与动态状态、结构化编码器（可选 MLP/GAT/R‑GCN）、时间差投影、可插拔序列模块（RNN、LSTM、Transformer、Mamba）、双轨状态更新（fast/slow 系统）以及对抗式负采样的因果一致性损失。

**📊 数据集**

在四大公开基准上评测：ICEWS14、ICEWS18、ICEWS05‑15 和 GDELT，均覆盖政治与全球事件场景。

**📈 对比分析**

与 17+ 传统与最新 TKG 预测模型（如 RE‑NET、TiRGN、CEN、DiffuTKG、CognTKE、DSEP 等）对比，EST 在所有基准上均实现 SOTA，尤其在噪声多的 GDELT 上提升 60% 以上；不同序列骨干均验证了 EST 的通用性，且 EST‑Transformer 与 EST‑Mamba 在准确率与训练速度/模型大小方面表现均衡。

**⚠️ 局限性**

局限性：①全局状态缓冲量级随实体数线性增长，可能在大规模 Web‑级图谱上内存消耗显著；②假设实体集固定且按时间顺序处理，难以即时融入新出现的实体；③实验仅覆盖标准时间顺序划分，未针对极端非平稳性或外部信号（属性、文本）进行验证。

---

## 237. Fix Before Search: Benchmarking Agentic Query Visual Pre-processing in Multimodal Retrieval-augmented Generation

**arXiv ID:** 2602.13179 | [PDF](https://arxiv.org/pdf/2602.13179v1)

**作者:** Jiankun Zhang `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 249916 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出V-QPP-Bench，构建可用于视觉查询预处理的标准化基准；

**💡 创新点**

将视觉查询预处理视为agentic决策任务，生成精细的修复轨迹并评估工具选择与参数预测；

**🔧 技术方法**

使用多模态检索增强生成框架、感知工具库、LLM代理、监督微调及Recall@K和精确匹配等评估指标；

**📊 数据集**

基于InfoSeek、ViQuAE VQA数据集与Wikipedia文本，构造约46.7k个带10类视觉畸变的查询；

**📈 对比分析**

对比零样本与SFT模型、oracle修复以及不同MRAG范式，发现严重畸变可导致Recall下降近98%，SFT后4B模型可与大型闭源模型媲美；

**⚠️ 局限性**

当前LLM在工具选择与参数预测上仍易失误，质量退化对检索影响小导致恢复有限，基准仅考虑单一畸变，未覆盖多重并发畸变。

---

## 238. String-Level Ground Fault Localization for TN-Earthed Three-Phase Photovoltaic Systems

**arXiv ID:** 2602.12289 | [PDF](https://arxiv.org/pdf/2602.12289v1)

**作者:** Yuanliang Li `[一作]` (Huawei Technologies Canada), Ziming Chen `[通讯]` (Huawei Technologies Canada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对三相TN接地光伏系统，提出了一种基于边缘AI的字符串级地故障定位方法。

**💡 创新点**

创新点在于：① 引入动态PV模型捕捉瞬态谐振，② 通过四阶段停机过程提取相关性特征，③ 采用轻量级Variational Information Bottleneck网络实现低算力端侧部署。

**🔧 技术方法**

使用技术包括：PLECS仿真建模、动态PV单二极管+扩散电容模型、LOESS信号分解、Pearson与Spearman相关系数特征、VIB深度学习框架。

**📊 数据集**

数据集来源为基于仿真的500+故障案例，涵盖不同辐照、温度、少数载流子寿命、故障位置、地阻等参数，并在100 kHz、10 kHz、5 kHz三种采样率下生成7160条样本。

**📈 对比分析**

通过与仅使用PS阶段特征、不同采样率以及不同特征集的8个模型对比，实验显示全特征+100 kHz下准确率99.9%，5 kHz下仍可达到93.0%；对比实验证明高采样率和F阶段特征能显著提升对负极端点故障的召回率。

**⚠️ 局限性**

局限性包括：① 依赖仿真数据，真实场景中的噪声与故障类型可能更为复杂；② 对负极端点故障仍有较低召回率；③ 目前仅验证单机系统，尚未在大规模并网场景中实测。

---

## 239. LLaMo: Scaling Pretrained Language Models for Unified Motion Understanding and Generation with Continuous Autoregressive Tokens

**arXiv ID:** 2602.12370 | [PDF](https://arxiv.org/pdf/2602.12370v1)

**作者:** Zekun Li `[一作]` (Brown University), Abhay Mittal `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了首个大规模运动‑语言统一模型，能够在保留预训练LLM语言能力的前提下，实现运动的理解与实时自回归生成。

**💡 创新点**

创新点：①使用模态特定Mixture‑of‑Transformers（MoT）冻结文本模块，避免语言能力消失；②采用因果连续VAE与流匹配（flow‑matching）头实现连续运动的自回归生成；③在超过300万帧（≈3076小时）运动‑文本大规模数据上预训练，支持零样本生成。

**🔧 技术方法**

技术：预训练LLM（如Llama）+ MoT 架构、因果VAE 编码/解码器、流匹配采样、连续运动潜空间、实时流式生成、离散语言解码头。

**📊 数据集**

数据集：来自Mocap、HMR从视频、以及HumanML3D、Motion‑X、100‑Style、CombatMotion、MotionGV、InterHuman、BABEL、FineDance、HI4D、HumanSC3D、Embody3d等公开数据集，外加自研大规模人像视频库，共计超过300万帧（3076小时）。

**📈 对比分析**

方法对比：在HumanML3D上的文本‑运动生成与运动‑文本字幕任务，与SOTA方法（大规模文本‑运动模型和专门模型）在R‑precision、CIDEr、BERTScore等指标上相当或更优；在MotionMillion‑Eval的零样本文本‑运动生成中，生成动作语义一致、流畅；实时流式生成速度≥30 FPS。

**⚠️ 局限性**

局限性：训练成本显著增加（MoT导致参数总量翻倍）；流匹配与运动头的训练平衡敏感，需细致调参；对非英语输入的鲁棒性尚未系统验证；未来需加入更多指令调优任务以提升多模态能力。

---

## 240. Imitating What Works: Simulation-Filtered Modular Policy Learning from Human Videos

**arXiv ID:** 2602.13197 | [PDF](https://arxiv.org/pdf/2602.13197v1)

**作者:** Albert J. Zhai `[一作]` (University of Illinois Urbana-Champaign), Wei-Chiu Ma `[通讯]` (Cornell University)

**通讯引用:** 1858 | [OpenAlex ID](https://openalex.org/A5037547726)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Perceive‑Simulate‑Imitate (PSI) 框架，能够仅利用人类视频训练机器人抓取和后抓取动作的模组化操控策略，并通过仿真过滤并标注任务兼容的抓取和轨迹。

**💡 创新点**

创新点在于：①使用仿真对人类轨迹与抓取对进行可行性评估，自动生成任务兼容的抓取标签；②将 6DoF 物体姿态作为运动表示，直接映射到机器人末端执行；③通过模组化设计将抓取与后抓取分离，同时保持抓取的任务兼容性。

**🔧 技术方法**

核心技术包括：RGB‑D 3D 视觉跟踪（FoundationPose、ICP、Cutie）、基于 SE(3) 的 6DoF 姿态回归、仿真中的抓取‑轨迹可行性验证、ResNet‑18 + MLP 的端到端预测网络、两阶段训练（先轨迹后抓取）。

**📊 数据集**

主要数据集为：自采 50 条人类视频（每个任务 50 条，35 条训练，15 条验证）以及 HOI4D 数据集（1580 条拿取与放置视频）用于预训练。

**📈 对比分析**

在四个真实世界任务（pick‑and‑place、pour、stir、draw）上与基线（使用随机抓取或不做轨迹过滤、仅利用流动或 6DoF 预测）进行对比。结果表明：①仿真过滤显著提升了成功率，去掉误轨迹后提升 10–20%；②任务兼容抓取比随机抓取提升 30–50%；③与 General‑Flow 等基线相比，PSI 的 SE(3) 轨迹预测误差更低，机器人成功率提升 15–25%。

**⚠️ 局限性**

局限性：①只能处理近似刚体物体，难以处理关节或柔性物体；②只基于视频起始帧训练，可能导致闭环部署时视域缺失导致域差；③仿真中的抓取稳定性未细化，需要进一步逼真建模。

---

## 241. Feature-based Uncertainty Model for School Choice

**arXiv ID:** 2602.12615 | [PDF](https://arxiv.org/pdf/2602.12615v1)

**作者:** Yao Zhang `[一作]` (Kyushu University), Makoto Yokoo `[通讯]` (Kyushu University)

**通讯引用:** 9266 | [OpenAlex ID](https://openalex.org/A5048575057)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

本文提出了基于特征的不确定性模型，对学校选择问题进行研究，并设计了不同程度的激励兼容性和稳定性概率保证的匹配算法。

**💡 创新点**

创新点在于引入特征权重随机性的学生偏好模型，揭示了稳定性概率与激励兼容性之间的不可兼得性，并给出最优与可接受的近似算法与下界。

**🔧 技术方法**

采用了概率分析、组合优化、退化接受算法（DA）变体以及 NP‑hardness 归约等技术。

**📊 数据集**

实验使用随机生成的实例，权重服从均匀分布，特征效用均匀采样。

**📈 对比分析**

通过理论最坏情况分析和仿真箱线图比较四种方法，HERF 在最坏情况下可达 (1/n)^n 近似，其余方法在最坏情况下无效；实验表明 LOICV 与 HERF 在大多数随机实例中能取得接近最优的稳定性概率。

**⚠️ 局限性**

局限性在于无法在同一算法中兼顾高稳定性概率与激励兼容性，且仅对两特征情形可实现多项式计算，更多特征导致计算难度 #P‑hard；算法对权重分布的假设过于理想化。

---

## 242. Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues

**arXiv ID:** 2602.12381 | [PDF](https://arxiv.org/pdf/2602.12381v1)

**作者:** Marco Willi `[一作]` (University of Applied Sciences FHNW), Michael Graber `[通讯]` (University of Applied Sciences FHNW)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用CLIP模型进行合成图像检测（SID），并构造了高质量的配对数据集SynthCLIC，同时引入正交低维线性头和稀疏概念瓶颈模型，对GAN和扩散模型在同一数据集以及跨数据集的泛化性能进行了系统评估。

**💡 创新点**

创新点包括：①设计了新的配对数据集SynthCLIC，减少语义偏差；②在CLIP隐藏层使用正交约束的线性投影实现可解释性分析；③将稀疏线性概念发现模型应用于SID，提供人类可解释的判定依据；④全面比较GAN与扩散模型的跨数据集泛化表现，揭示CLIP对高质量扩散图像的局限。

**🔧 技术方法**

技术手段：CLIP视觉编码器、正交约束线性投影、稀疏线性概念发现模型（CDM）、CLIP-IQA属性评分、线性逻辑回归、数据增强、正则化与早停。

**📊 数据集**

使用的数据集包括：CNNSpot（GAN生成图像），SynthBuster+（扩散模型），SynthCLIC（扩散模型），以及这些数据集的合并集。

**📈 对比分析**

实验对比方法：在同一数据集上，CLIP线性头在GAN数据集上达到0.96 mAP，在扩散数据集上约0.92 mAP；在跨数据集（GAN→扩散或反向）时，mAP仅为0.37–0.54；概念模型的性能与线性头相近但略低。总体而言，CLIP在GAN图像上表现极佳，在高质量扩散图像上性能下降，跨数据集泛化差。

**⚠️ 局限性**

局限性：仅使用单一CLIP backbone；实验聚焦摄影类图像，未验证医学或科学影像；未结合低层次指纹或后处理特征；概念模型对词汇表设计与稀疏性超参数敏感。

---

## 243. Training-Free Acceleration for Document Parsing Vision-Language Model with Hierarchical Speculative Decoding

**arXiv ID:** 2602.12957 | [PDF](https://arxiv.org/pdf/2602.12957v1)

**作者:** Wenhui Liao `[一作]` (South China University of Technology), Lianwen Jin `[通讯]` (South China University of Technology)

**通讯引用:** 13789 | [OpenAlex ID](https://openalex.org/A5080674767)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过使用轻量级管线模型生成草稿并与强大的 VLM 并行验证，实现了端到端文档解析的推理加速；

**💡 创新点**

提出了层次化投机解码框架，利用文档布局的局部性进行区域并行验证，并通过窗口对齐和树结构验证解决草稿与目标模型的偏差；

**🔧 技术方法**

采用投机解码（draft‑verify）策略、窗口对齐、树结构并行验证、FlexAttention 等技术；

**📊 数据集**

在 OmniDocBench v1.5、olmOCR-Bench 与 Ocean‑OCR‑Bench 三大公开基准上进行评估；

**📈 对比分析**

与原始自回归解码和其他投机解码基线相比，取得 2.42×–4.89× 的加速，且保持或略高于基线的解析准确率；

**⚠️ 局限性**

未充分利用高效 CUDA 核心（如 FlashAttention）、缺乏与通用投机解码方法的系统比较，以及对 GPU 内存管理的进一步优化空间。

---

## 244. Secrecy Capacity Analysis and Beamforming Optimization for MIMO-VLC Wiretap Channels

**arXiv ID:** 2602.12720 | [PDF](https://arxiv.org/pdf/2602.12720v1)

**作者:** Sufang Yang `[一作]` (China Mobile Research Institute), Guangyi Liu `[通讯]` (China Mobile Research Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过使用GEPI对截断指数输入进行分析，推导了在峰值与平均光强双重约束下MIMO-VLC窃听信道的闭式可实现保密率，并基于该结果提出了全连接与子连接两种波束成形方案。随后针对非凸的波束成形优化问题，设计了基于SCA的迭代算法以求解最优波束矩阵。

**💡 创新点**

①利用GEPI获得在峰值+平均强度约束下的闭式保密率表达式；②提出全连接与子连接两种硬件友好的波束成形结构；③在非凸优化中引入Taylor展开+二阶正则化的强凸 surrogate 进行SCA，显著降低计算复杂度并保证收敛。

**🔧 技术方法**

1）Generalized Entropy‑Power Inequality (GEPI)；2）截断指数分布建模；3）梯度与Hessian分析；4）SCA（Successive Convex Approximation）与CVX求解器；5）高斯噪声假设与最大熵原则。

**📊 数据集**

在仿真中使用两组随机生成的光学通道矩阵 
- ¡ Group 1: H_B^{1×4}, H_E^{1×4}
- ¡ Group 2: H_B^{2×4}, H_E^{2×4}
并未采用真实光通信实验数据，而是以模拟数据进行性能验证。

**📈 对比分析**

与基准方案（直接连接波束成形、前人单峰值约束模型）以及理论下限进行对比。实验结果显示：
- 当 LED 数量 ≥ PD 数量时，所提出的全连接和子连接波束成形均可实现显著的保密率提升；
- 在 LED 数量 < PD 数量的情况（Case II）下，波束成形对保密率的提升可忽略不计，直接连接方案已足够；
- 与单峰值约束下的结果相比，双约束模型在低SNR时略逊一筹，高SNR时更优。

**⚠️ 局限性**

1）仅考虑高斯噪声；2）使用截断指数输入，可能不是全局最优输入分布；3）SCA 可能陷入局部最优，收敛速度受步长选择影响；4）仅在仿真层面验证，缺乏实验室或实地光通信测试；5）对多光源非线性或相干干扰等实际效应未建模。

---

## 245. Vision Token Reduction via Attention-Driven Self-Compression for Efficient Multimodal Large Language Models

**arXiv ID:** 2602.12618 | [PDF](https://arxiv.org/pdf/2602.12618v1)

**作者:** Omer Faruk Deniz `[一作]` (University of Texas at Dallas), Latifur Khan `[通讯]` (University of Texas at Dallas)

**通讯引用:** 10403 | [OpenAlex ID](https://openalex.org/A5005002693)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多模态大型语言模型内部，使用自注意力机制逐层对视觉令牌进行统一下采样，实现视觉令牌压缩并保持模型性能。

**💡 创新点**

创新点在于：①利用LLM自身的注意力流动作为压缩引导，无需额外模块或注意力分数；②通过在选定层创建信息瓶颈，促使模型自学习压缩；③采用逆向剪裁训练（reverse curriculum）进一步提升压缩效果；④完全兼容FlashAttention，保持高效计算。

**🔧 技术方法**

技术包括：Transformer自注意力、统一下采样（固定drop ratio）、逆向剪裁训练策略、LoRA参数高效微调、FlashAttention兼容实现；实验基于LLaVA‑1.5‑7B模型。

**📊 数据集**

使用的数据集：GQA、MME、POPE、ScienceQA、TextVQA、VQAv2 进行评估；训练集为 LLaVA‑InstructionMix‑665K 多模态指令-响应对。

**📈 对比分析**

与 ToMe、FastV、PyramidDrop 等基线比较，ADSC 在 192、128、64 视觉令牌预算下分别保留 98.2%、97.4%、95.1% 的原始性能；极端压缩下仍优于所有对手；计算 FLOPs 减少 53.7%，KV 缓存减少 56.7%，整体性能几乎不降。

**⚠️ 局限性**

局限性：需手动选择压缩层和比例，固定下采样方式对不同视觉分布可能不理想；在更极端压缩或其他 LLM 体系结构下的鲁棒性未完全验证；压缩过程仍会导致轻微性能下降；对训练时间与收敛行为的影响仍需进一步研究。

---

## 246. Soft Contamination Means Benchmarks Test Shallow Generalization

**arXiv ID:** 2602.12413 | [PDF](https://arxiv.org/pdf/2602.12413v1)

**作者:** Ari Spiesberger `[一作]` (Arb Research), Nandi Schoots `[通讯]` (University of Oxford)

**通讯引用:** 1961 | [OpenAlex ID](https://openalex.org/A5062539506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Olmo3公开训练语料中使用余弦相似度搜索并人工标注，系统检测了与主要推理/编程基准（MBPP、CodeForces、MuSR、ZebraLogic）在语义层面的重复；随后利用LoRA微调并结合链式思考生成器，对含重复与无重复数据进行实验，评估其对基准性能的影响。

**💡 创新点**

首次在大规模开源LLM语料中量化软污染（语义重复）的普遍性，证明其导致的“浅层泛化”会显著提升基准得分，并通过生态化微调实验展示低比例重复也能产生可观的性能提升。

**🔧 技术方法**

使用FP16精度的语义嵌入（如Sentence‑BERT变体）与余弦相似度匹配；采用大型语言模型自动标注语义重复；采用LoRA微调、链式思考（CoT）生成、温度采样等训练技术；使用多基准评估（见下）。

**📊 数据集**

Olmo3训练语料（Dolma、Dolmino、Dolci SFT/DPO/RL）以及四大推理/编程基准MBPP、CodeForces、MuSR、ZebraLogic，并对比同类基准HumanEval、TrueDetective、Arc Challenge进行交叉验证。

**📈 对比分析**

对比基线模型与在“见过”/“未见过”两半基准样本上微调后的表现；在含重复的微调中，精确重复和语义重复均可提升约20%性能；高相似度但非重复样本几乎无效；在5%生态化污染下，Olmo3在见样本提升12%，未见样本提升5.6%；跨基准提升有限。

**⚠️ 局限性**

可能低估了语义重复率（检测方法假阴率高），仅针对开源模型，难以推广到更大闭源语料；合成重述数据未完全覆盖；Qwen3微调结果噪声大，验证不充分。

---

## 247. GRAIL: Geometry-Aware Retrieval-Augmented Inference with LLMs over Hyperbolic Representations of Patient Trajectories

**arXiv ID:** 2602.12828 | [PDF](https://arxiv.org/pdf/2602.12828v1)

**作者:** Zhan Qu `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了 GRAIL 框架，用几何感知的检索与大型语言模型（LLM）重排序来预测下一次临床事件。

**💡 创新点**

创新点在于将确定性医疗编码层级与数据驱动的跨模态时序关联融合进统一的超曲面（Poincaré 球面）图谱中，通过中心事件压缩和风险锥检索实现结构化候选，再用 LLM 在受限空间内进行语义精炼。

**🔧 技术方法**

使用了超曲面嵌入、边缘重建与负采样损失、概率中心事件（Central Event）重构、基于风险锥的几何检索以及 LLM 提示重排。

**📊 数据集**

基于 MIMIC‑IV 电子健康记录进行实验。

**📈 对比分析**

与 Transformer、BEHRT、RETAIN、Euclidean RAG/Graph GNN 及无拼接/无 LLM 的基线比较，GRAIL 在多模态下 Recall@10、nDCG@10 以及主诊断 Top‑1/Top‑5 方面显著提升，提升幅度约为 20–50% 以上。

**⚠️ 局限性**

局限性包括对 MIMIC‑IV 具体编码习惯的依赖，中心事件压缩可能导致信息丢失，LLM 受提示设计限制，且模型尚未经过前瞻性临床验证。

---

## 248. Multi-Dimensional Visual Data Recovery: Scale-Aware Tensor Modeling and Accelerated Randomized Computation

**arXiv ID:** 2602.12982 | [PDF](https://arxiv.org/pdf/2602.12982v1)

**作者:** Wenjin Qin `[一作]` (Southwest University), Tingwen Huang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出基于全连接张量网络（FCTN）的非凸梯度域正则化与量化观测下的多维视觉数据恢复框架，结合随机化压缩加速求解；

**💡 创新点**

创新点在于：①将低秩与局部平滑同时嵌入FCTN正则化；②在量化观测模型中引入均匀加性抖动；③利用随机化Sketch实现高阶张量压缩，实现高效的ADMM优化并给出误差上界与收敛性分析；

**🔧 技术方法**

采用FCTN分解、非凸Φ函数（如ℓq、SCAD等）、梯度张量操作、均匀抖动量化、随机化Sketch（Gaussian、KR、SRFT等）与ADMM；

**📊 数据集**

实验数据集涵盖彩色图像、彩色视频、遥感多时相图像、超光谱视频、面部数据库、磁共振成像等；

**📈 对比分析**

与TRNN、FCTN-NNM、FCTN-TC、FCTNFR、HTNN、METNN、WSTNN、EMLCP、OTNN、MTTD、TCTV-TC、GTNN-HOC、t-ϵ-LogDet等多种现有低秩/正则化方法对比，实验显示在低采样率下PSNR/SSIM提升约2–3 dB，MRSE降低，且随机化版本平均比确定性实现快9倍（极端情况可达20倍）；

**⚠️ 局限性**

局限性：需手动调参（Φ、ψ权重、λ、δ等）；随机化Sketch对极低秩张量的精度影响不完全可控；在极大规模张量或极端量化步长下仍可能出现重建误差累积。

---

## 249. Real-time Rendering with a Neural Irradiance Volume

**arXiv ID:** 2602.12949 | [PDF](https://arxiv.org/pdf/2602.12949v1)

**作者:** Arno Coomans `[一作]` (Huawei Technologies), Markus Steinberger `[通讯]` (Graz University of Technology)

**通讯引用:** 2237 | [OpenAlex ID](https://openalex.org/A5014594342)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Neural Irradiance Volume (NIV)，利用神经网络压缩并实时查询全局间接漫反射照明，实现动态对象在静态场景中的高质量渲染。

**💡 创新点**

创新点在于：① 直接使用 5D (位置+法线) 作为输入，MLP 预测辐照度，消除传统探针的插值误差；② 引入多级哈希编码实现极高压缩率，内存占用仅 1–5 MB；③ 训练时仅需静态场景的路径追踪数据，支持动态对象与可变场景的无缝渲染；④ 推理速度约 1 ms/帧，实现无光线追踪、无去噪的实时渲染。

**🔧 技术方法**

技术要点包括：4 层全连接 MLP + ReLU、频率编码或学习编码、3D+方向多级哈希编码、基于 G‑buffer 的位置/法线查询、动态环境遮挡 (AO) 以及多级表面与体积混合训练。

**📊 数据集**

数据集主要使用公开的室内/室外场景：Sponza、Cornell Box、Dining Room、Bathroom、White Room、Living Room，以及变量场景（Armadillo）和时间循环实验（旋转灯光）。

**📈 对比分析**

与 DDGI、传统探针、神经表面缓存、可变场景方法比较。相同内存预算下，NIV 在误差上提升约 10 倍；推理时间为 0.19–1.35 ms（全分辨率），半分辨率仅 0.03–0.37 ms；相较于神经表面缓存误差更低、速度快 5–10 ms，且不需要额外的光线追踪或去噪。

**⚠️ 局限性**

局限性：只适用于漫反射材质；不捕捉动态物体对光照的高阶影响；直接光照需额外实现；对大量光源仍需额外渲染通道；哈希编码冲突可能影响细节；扩展至极大场景需分区或 LOD；在线更新速度有限。

---

## 250. Flow-Factory: A Unified Framework for Reinforcement Learning in Flow-Matching Models

**arXiv ID:** 2602.12529 | [PDF](https://arxiv.org/pdf/2602.12529v1)

**作者:** Bowen Ping `[一作]` (Xi'an Jiaotong University), Ivor Tsang `[通讯]` (CFAR A*STAR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Flow-Factory统一框架，解耦强化学习算法、模型和奖励，支持多算法、多模型的高效训练与对比。

**💡 创新点**

创新点包括：①注册式模块化设计实现算法与模型解耦；②预处理缓存嵌入实现显存优化；③支持点奖励与群奖励并可多奖励融合，极大提升实验灵活性。

**🔧 技术方法**

采用注册机制、预处理缓存、SDE/ODE采样、GRPO、DiffusionNFT、AWM等强化学习与流匹配技术，并实现多种噪声动态与优势归一化。

**📊 数据集**

以FLUX.1-dev为基础模型，并结合Text-Rendering和PickScore奖励模型进行实验。

**📈 对比分析**

通过在相同backbone上统一配置复现Flow-GRPO、DiffusionNFT、AWM，奖励曲线收敛一致；显存下降13%，单步时间提升1.74倍，生成图像质量显著提升。

**⚠️ 局限性**

局限性：实验仅覆盖单一模型与奖励，缺乏多任务/多模态评估；预处理缓存需在prompt变化时重新生成；对极大规模分布式训练的支持尚不完善。

---

## 251. OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization

**arXiv ID:** 2602.12305 | [PDF](https://arxiv.org/pdf/2602.12305v1)

**作者:** Arijit Bhattacharjee `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1091 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OptiML，一个端到端框架，能够从自然语言或已有CUDA代码生成并优化高性能CUDA核；

**💡 创新点**

创新点在于将LLM的思维混合生成（Mixture-of-Thoughts）与基于硬件反馈的蒙特卡洛树搜索（MCTS）相结合，利用LLM作为判别者与硬件指标引导的奖励模型，实现可解释的、多步优化；

**🔧 技术方法**

使用的技术包括Mixture-of-Thoughts生成器、LLM驱动的提案与评判、MCTS搜索、Nsight Compute硬件指标回馈、运行时测量、LLM修复编译错误以及组合奖励函数；

**📊 数据集**

评测数据集为5个代表性CUDA-LLM核（矩阵乘、3D最大池化、注意力、多头注意力、ReLU和交叉熵），并在ParEval benchmark上评估生成质量；

**📈 对比分析**

与多种LLM基线（GPT‑5.1、Qwen2.5‑Coder、HPC‑Coder‑V2、StarCoder2）比较，OptiML在A100 GPU上平均加速约1.6×，在生成准确率、编译成功率和加速率上均优于单独生成或单独优化；

**⚠️ 局限性**

局限包括：搜索预算有限时仍可能停留在局部最优；对硬件特定的指标依赖强，跨设备迁移需重新校准；LLM判别器的可靠性与解释性受模型能力限制。

---

## 252. SHAPR: A Solo Human-Centred and AI-Assisted Practice Framework for Research Software Development

**arXiv ID:** 2602.12443 | [PDF](https://arxiv.org/pdf/2602.12443v1)

**作者:** Ka Ching Chan `[一作]` (University of Southern Queensland), Ka Ching Chan `[通讯]` (University of Southern Queensland)

**通讯引用:** 1646 | [OpenAlex ID](https://openalex.org/A5061915685)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

提出了 SHAPR 框架，用以指导单人研究者在使用生成式人工智能协助下的研究软件开发，结合 Action Design Research 的 BIE 循环，强调人本治理、角色分离、版本追踪与反思。

**💡 创新点**

创新点在于：①将 ADR 的高层原则转化为可操作的日常实践指南；②专为 solo、AI 辅助的研发场景设计，突出人机协作中的责任与学习；③引入轻量化治理与 artefact‑centric 证据管理，兼顾科研严谨与学习效果。

**🔧 技术方法**

技术实现上主要是概念性设计，不依赖特定工具，框架对话式 AI、Vibe coding、AI 辅助 IDE、版本控制、轻量化计划与测试工具等均列为可选实现手段。

**📊 数据集**

无实验数据集，框架为概念性研究，未在具体软件项目中验证。

**📈 对比分析**

未进行实验比较；评估方式为形成性反思，考察内部一致性、与 ADR 原则的契合度及对 solo 研究实践的适用性，未给出性能指标。

**⚠️ 局限性**

局限性包括：①概念化、缺乏实证验证；②评估仅为反思性分析，缺乏外部证据；③仅适用于单人研究软件开发，未扩展至团队或工业环境；④对 AI 工具的描述基于当下技术，可能随工具演进而变。

---

## 253. WISE: A Multimodal Search Engine for Visual Scenes, Audio, Objects, Faces, Speech, and Metadata

**arXiv ID:** 2602.12819 | [PDF](https://arxiv.org/pdf/2602.12819v1)

**作者:** Prasanna Sridhar `[一作]` (University of Oxford), Abhishek Dutta `[通讯]` (University of Oxford)

**通讯引用:** 1225 | [OpenAlex ID](https://openalex.org/A5101737651)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并开源了一款名为WISE的多模态检索引擎，支持图像、视频、音频、语音和元数据的检索，并可通过自然语言、反向图像、面部、语音等多种查询方式实现组合检索。

**💡 创新点**

创新点在于将多模态检索能力整合到一个统一、易用的平台；支持跨模态组合查询（如图像+文本+元数据）；采用共享向量空间的跨模态模型；模块化架构允许本地部署、可扩展新模型与存储后端；并实现了可在数百万图像和千小时视频上即时检索的性能。

**🔧 技术方法**

主要技术包括：视觉-语言模型（OpenCLIP、OWL‑v2、InsightFace）、音频-语言模型（CLAP）、语音识别（WhisperX）、向量索引（Faiss IndexIVFFlat/IVFPQ）、SQLite FTS全文检索、React 前端、GPU 加速的批量特征提取。

**📊 数据集**

使用的数据集包括：维基媒体共通（Wikimedia Commons）55M 图像、BBC 6000 小时视频、各类实验性媒体库（如新闻素材、档案影像、野生动物录像）等。

**📈 对比分析**

性能评估显示：1 小时视频在配备 GPU 的现代机器上可在 10 分钟以内完成特征提取；在百万级图像/千小时视频上，检索时间低于 1 秒；采用 Faiss IndexIVFPQ 可进一步降低内存占用而仅略微影响召回率；与单一模态检索相比，组合查询显著提升了检索精度。

**⚠️ 局限性**

局限性包括：特征提取与索引是一次性离线计算，需耗费大量算力与存储；依赖预训练模型，模型误差会直接影响检索质量；对极为细粒度的音频事件或罕见对象识别能力有限；当前仅支持已实现的模态，新增模态需扩展 Loader/Extractor 模块。

---

## 254. Decentralized Optimal Equilibrium Learning in Stochastic Games via Single-bit Feedback

**arXiv ID:** 2602.12830 | [PDF](https://arxiv.org/pdf/2602.12830v1)

**作者:** Seref Taha Kiremitci `[一作]` (Bilkent University), Muhammed O. Sayin `[通讯]` (Bilkent University)

**通讯引用:** 746 | [OpenAlex ID](https://openalex.org/A5081851582)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个名为 DOEL 的去中心化优化平衡学习框架，利用单比特随机化信号在有限折扣随机博弈中实现对设计者指定社会福利目标的平衡选择。

**💡 创新点**

创新点在于：①只使用单比特反馈即可隐式对齐全局福利；②结合 explore‑and‑commit 与在线两种策略，兼顾模型无关和模型有关方法；③提供显式的有限时刻对数级别回报保证，证明在稀疏通信下仍可实现对最优平衡的高效学习。

**🔧 技术方法**

核心技术包括：随机化内容/不满信号设计、均匀探索与扰动化策略、贝尔曼评估/最优方程求解（可采用模型基或模型无关算法），以及基于概率分布的回报分析。

**📊 数据集**

实验使用人工生成的 3×3×3 随机博弈（|N|=3，|S|=3，|A|=3），随机设置奖励，评估在求和与乘积两类福利函数下的收敛性能。

**📈 对比分析**

与未使用信号的传统平衡学习相比，DOEL 在相同探索预算下能更快地聚焦到最优平衡并取得接近全局最优的福利；在在线模式下尽管探索与利用交替，回报仍保持对数级别的回溯误差，说明方法鲁棒且性能可观。

**⚠️ 局限性**

局限性包括：需要大量探索阶段（如 10⁶ 次）才能实现收敛；策略空间指数级大，实际实现需进一步压缩或采用参数化策略；对估计误差和状态覆盖敏感，需足够长的阶段长度；以及对完全分布式、无中心化假设的严格要求。

---

## 255. MiDAS: A Multimodal Data Acquisition System and Dataset for Robot-Assisted Minimally Invasive Surgery

**arXiv ID:** 2602.12407 | [PDF](https://arxiv.org/pdf/2602.12407v1)

**作者:** Keshara Weerasinghe `[一作]` (University of Virginia), Homa Alemzadeh `[通讯]` (University of Virginia)

**通讯引用:** 1294 | [OpenAlex ID](https://openalex.org/A5055237181)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MiDAS，一种平台无关的、非侵入式多模态数据采集系统，能够在不访问机器人内部接口的情况下同步记录电磁手部跟踪、RGB‑D手部关键点、脚踏传感及立体视频；

**💡 创新点**

创新点在于：①将EM跟踪与深度摄像机、脚踏传感器整合为统一框架；②通过外部传感器逼近内部机器人运动，实现跨平台（Raven‑II、da Vinci Xi）同步采集；③公开完整系统与多模态数据集，促进复现与跨平台研究；

**🔧 技术方法**

技术包括NDI trakSTAR电磁跟踪、ZED Mini RGB‑D相机、Arduino+FSR脚踏传感器、OBS视频采集、Python客户端‑服务器架构、深度学习模型（MTRSAP、MS‑TCN++）等；

**📊 数据集**

使用了两组数据集：Raven‑II 15次Peg Transfer（36 min）和da Vinci Xi 17次腹股沟/腹壁疝修复缝合（212 min），并公开了相应的多模态标签；

**📈 对比分析**

通过与Raven‑II内部轨迹进行余弦相似度和NRMSE比较，EM跟踪平均位置误差<20%，姿态余弦相似度>0.8；脚踏传感器F1>0.8；在手势识别任务中，EM跟踪在MTRSAP/​MS‑TCN++模型上F1≈0.86/0.79，几乎与内部轨迹相当，且优于仅视频模型；

**⚠️ 局限性**

局限性：①EM跟踪在姿态、抓取角度上仍有偏差；②手部关键点因视场/遮挡导致检测缺失；③脚踏传感延迟约100‑170 ms；④仅在干预较少的手术环境下验证，未覆盖真实手术多变场景；

---

## 256. Opinion dynamics and mutual influence with LLM agents through dialog simulation

**arXiv ID:** 2602.12583 | [PDF](https://arxiv.org/pdf/2602.12583v1)

**作者:** Yulong He `[一作]` (ITMO University), Artem Sedakov `[通讯]` (Saint Petersburg State University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大型语言模型（LLM）代理的多轮结构化对话仿真框架，用于研究意见形成与演化，并将其与经典的DeGroot与Friedkin–Johnsen模型对接；

**💡 创新点**

创新点在于将传统意见更新机制嵌入LLM对话流程，保留初始立场实现锚定效应，并通过情感分析将生成文本映射为数值意见，实现可扩展的网络拓扑控制；

**🔧 技术方法**

采用GPT‑4o mini、Qwen2.5、Llama 3.3、Mistral Large、DeepSeek V3等LLM，通过OpenAI API生成对话；利用Twitter‑RoBERTa情感分析模型将文本转为情感分数；使用线性最小二乘法估计影响矩阵W和抗拒矩阵S；Python实现完整实验流程；

**📊 数据集**

使用LLM自行生成的对话文本作为数据源，情感分析模型基于公开的124M推文预训练数据集；

**📈 对比分析**

通过残差平方和、平均自信任（W_ii）与易受性（S_ii）指标比较不同LLM在同质与异质组中的表现；结果显示DeepSeek V3在DG和FJ模型下拟合最优（残差最低），Qwen2.5表现中等，Mistral Large最差，体现模型差异显著；

**⚠️ 局限性**

限制包括：实验仅在单一中性话题下进行，初始立场设置对结果敏感；情感映射可能引入偏差；LLM生成的对话与真实人类对话差异；网络结构固定，未考虑动态拓扑；结果尚未验证在真实社交网络中的可泛化性。

---

## 257. Latent Customer Segmentation and Value-Based Recommendation Leveraging a Two-Stage Model with Missing Labels

**arXiv ID:** 2602.12485 | [PDF](https://arxiv.org/pdf/2602.12485v1)

**作者:** Keerthi Gopalakrishnan `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套两阶段多模型架构，利用自适应自我进度损失（Self‑Paced Loss）对客户进行细粒度分类，区分营销促使与自然参与的客户，进而优化营销目标的精准投放。

**💡 创新点**

创新点在于：① 引入缺失标签框架与二元标签校正，解决传统提升模型中标签缺失和误标问题；② 使用自我进度损失动态调整正样本损失，降低误报；③ 将价值主张与品牌价值平衡融入客户细分，打破传统基于交易的RFM模型局限。

**🔧 技术方法**

主要技术包括：多分类神经网络（Multi‑Class NN）用于识别“已参与”“未参与”“不活跃”三类；二元标签校正模型（Binary Label Correction）配合自我进度损失（SPLC）实现标签纠正；特征工程整合用户画像、互动记录与交易历史；在线A/B实验与累计增益曲线评估模型效能。

**📊 数据集**

使用Walmart全球技术平台的营销响应数据，涵盖约1080万用户的点击、浏览、转化等日志，结合用户画像与交易记录作为特征；在线实验中对10.8万用户（6.6万新用户+4.2万流失用户）进行对照测试。

**📈 对比分析**

与传统RFM（最近、频率、金额）三大高活跃分段做对比：离线指标提升明显（precision 0.448→0.371，recall 0.600→0.433，balanced accuracy 0.724→0.624，weighted F1 0.817→0.746）。在线A/B测试显示新用户转化率提升99.85%（从0.17%提升至0.34%），流失用户提升10.78%（从0.68%提升至0.76%）。

**⚠️ 局限性**

局限性包括：① 依赖Walmart内部数据，外部可复现性受限；② 自我进度损失阈值需要手工调优，可能影响模型稳健性；③ 仅评估了营销促使与自然参与两类，未覆盖更细粒度的用户意图；④ 在动态市场变化时需频繁重新训练，资源消耗较大。

---

## 258. R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training

**arXiv ID:** 2602.13103 | [PDF](https://arxiv.org/pdf/2602.13103v1)

**作者:** Gengsheng Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (Wuhan AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 R-Diverse 框架，通过 Memory-Augmented Penalty (MAP) 与 Skill-Aware Measurement (SAM) 来解决自我游戏训练中的多样性幻觉，实现更可持续的推理 LLM 迭代提升。

**💡 创新点**

创新点在于引入持久记忆惩罚 (MAP) 与基于代码抽象的技能感知测量 (SAM)，从全局与技能层面同时提升训练多样性。

**🔧 技术方法**

使用自我游戏框架、GRPO 强化学习、记忆库、代码抽象与嵌入相似度、经验回放等技术。

**📊 数据集**

在十个数学与通用推理基准（AMC、Minerva、MATH、GSM8K、Olympiad、AIME25/24、SuperGPQA、MMLU-Pro、BBEH）上进行评估。

**📈 对比分析**

与 R-Zero、Absolute Zero、SPIRAL、Socratic-Zero 等方法对比，R-Diverse 在数学和整体平均分上均排名第一，迭代过程持续提升（5 次迭代后 Math AVG 52.59 / 56.46），明显优于基线模型。

**⚠️ 局限性**

局限在于依赖代码作为语义瓶颈，难以推广到难以形式化的领域，且自我游戏训练对算力和能源消耗较大。

---

## 259. Multi-Agent Model-Based Reinforcement Learning with Joint State-Action Learned Embeddings

**arXiv ID:** 2602.12520 | [PDF](https://arxiv.org/pdf/2602.12520v1)

**作者:** Zhizun Wang `[一作]` (McGill University), David Meger `[通讯]` (McGill University)

**通讯引用:** 6127 | [OpenAlex ID](https://openalex.org/A5109496264)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MMSA框架，结合世界模型和联合状态-动作嵌入（SALE），实现多智能体的样本高效学习。

**💡 创新点**

创新性地将SALE嵌入模型基世界模型，利用想象化轨迹与Monotonic混合网络结合CTDE，实现联合表示学习与价值分解的无缝整合。

**🔧 技术方法**

使用模型基强化学习、变分自编码器、SALE、QMIX式混合网络、KL平衡、AvgL1Norm等技术。

**📊 数据集**

在Multi-Agent MuJoCo、Level-Based Foraging和StarCraft II（SMAC/SMACv2）三大基准环境上进行实验。

**📈 对比分析**

与多种模型自由和模型基MARL基线（如V-DN、COMA、QMIX、MAMBA、MAG等）对比，MMSA在回报率和胜率上持续领先，甚至与最先进方法持平或超越。

**⚠️ 局限性**

主要局限在于模型误差累积导致的想象轨迹偏差，以及对世界模型准确性的高依赖，未来需引入模型集成或不确定性估计以缓解这些问题。

---

## 260. Predicting Dynamic Map States from Limited Field-of-View Sensor Data

**arXiv ID:** 2602.12360 | [PDF](https://arxiv.org/pdf/2602.12360v1)

**作者:** Knut Peterson `[一作]` (Drexel University), David Han `[通讯]` (Drexel University)

**通讯引用:** 3463 | [OpenAlex ID](https://openalex.org/A5087980287)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种将有限视场（FOV）激光雷达时间序列数据压缩为单张灰度图像的方法，并利用现有的图像到图像的深度学习模型实现动态地图状态预测。

**💡 创新点**

创新点在于：①通过时间衰减的灰度梯度将多帧传感器数据累积到一张图像中，既保留空间信息又编码时间信息；②利用该单图像格式使得任何已有的图像分割网络均可直接用于地图预测；③在实验中验证了灰度时间衰减对动态场景预测精度的显著提升。

**🔧 技术方法**

主要技术包括：基于LIDAR扫描的空间投影、时间衰减灰度编码、U‑Net、FPN、UPerNet、SegFormer等图像分割模型，以及BCE+Dice混合损失、Adam优化等。

**📊 数据集**

使用自行构建的二维仿真数据集，包含四种实验设置（静态障碍物/动态障碍物，机器人旋转/正方形路径），每种设置生成约10k训练对、128验证对、500测试对。

**📈 对比分析**

与相关工作对比：在即时预测下，召回率与准确率与全视场方法相近，且在动态场景下取得平均Dice≈0.97、准确率≈0.94、SSIM≈0.67；相较于传统统计或分布式方法，模型实现更简单、实时性更好。

**⚠️ 局限性**

局限性包括：依赖于仿真数据，真实环境噪声与多模态传感器的适应性待验证；灰度时间衰减在极端动态或遮挡严重时仍会导致模糊预测；模型对较大地图尺寸或更复杂动态模式的扩展性尚未探究。

---

## 261. Source Code Hotspots: A Diagnostic Method for Quality Issues

**arXiv ID:** 2602.13170 | [PDF](https://arxiv.org/pdf/2602.13170v1)

**作者:** Saleha Muzammil `[一作]` (University of Virginia), Diomidis Spinellis `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 8742 | [OpenAlex ID](https://openalex.org/A5021948425)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对91个GitHub开源项目的完整提交历史进行细粒度行级分析，识别并分类了15种热点模式，并量化了自动化机器人对热点产生的影响；

**💡 创新点**

创新点包括①首次实现行级代码热点挖掘工具；②构建15种可操作的热点模式分类体系；③结合机器人检测揭示大部分热点来自自动化提交；④为每种热点提供针对性的CI检查与重构指南；

**🔧 技术方法**

技术主要包括基于Git diff的行追踪算法、标准差阈值与Chao1估计的热点检测与标注方法、手工标注验证、统计分析与可视化；

**📊 数据集**

使用的数据集为91个根据stars/forks stratified sampling挑选的GitHub仓库，项目规模覆盖10k+提交、1M+行代码，涵盖多种编程语言；

**📈 对比分析**

评估方法为将自动检测与人工标注对比，达到90%准确率；通过统计分析热点分布、持续时间和修改次数，比较机器人与人工提交比例，结果显示机器人贡献约74%热点提交；

**⚠️ 局限性**

限制主要包括：仅分析GitHub仓库；行级身份仅基于文件路径与行号，无法捕捉跨文件或无改动文本的移动；阈值设置主观；使用的Git diff不支持完整树差异，可能导致部分热点被漏检。

---

## 262. RLinf-Co: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models

**arXiv ID:** 2602.12628 | [PDF](https://arxiv.org/pdf/2602.12628v1)

**作者:** Liangzhi Shi `[一作]` (Tsinghua University), Yu Wang `[通讯]` (Tsinghua University)

**通讯引用:** 43716 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了RL-Co框架，在VLA模型上实现两阶段仿真‑现实共训练：先用真实与模拟演示的监督微调进行初始化，再在模拟中进行强化学习微调，同时加入真实数据的监督正则以防灾难性遗忘，从而提升真实世界的执行效果。

**💡 创新点**

创新点在于：① 把强化学习与监督微调耦合成两阶段共训练流程；② 在RL阶段引入真实数据监督正则，解决纯RL导致的灾难性遗忘；③ 通过模拟交互而非仅依赖演示，显著提升泛化能力和数据效率；④ 该框架对多种VLA架构通用。

**🔧 技术方法**

使用的技术包括：OpenVLA 与 π_0.5 这两类VLA模型；SFT（监督微调）与RL（PPO等）两种微调方式；在RL阶段加入真实数据的监督损失作为正则；模拟环境构建在 ManiSkill 上，模拟数据通过 MimicGen 生成；实验任务为桌面抓取、推箱子、拉/推抽屉等四个真实与模拟对照任务。

**📊 数据集**

数据集：真实世界演示 20–50 条轨迹；模拟环境生成 1,000 条成功轨迹（MimicGen 生成的“数字孪生”数据）。四个任务均使用对应的真实与模拟数据集进行训练与评估。

**📈 对比分析**

比较方法：与仅用真实数据的 SFT 以及仅用 SFT 的仿真‑现实共训练两基线对比。实验结果显示：在四个任务上，RL‑Co 在 OpenVLA 上提升约 +24% 成功率，在 π_0.5 上提升约 +20%；在未见任务变种时表现更稳健；在数据效率方面，仅用 20 条真实轨迹即可匹敌或超过使用 200 条真实轨迹的基线。

**⚠️ 局限性**

limitations：仅在桌面抓取任务和单一机器人（Franka Panda）上验证；未探讨异构仿真‑现实环境或多机器人设置；最终性能仍未达到 100%，且未加入真实世界 RL；未验证长时限任务、复杂多步操作等场景。

---

## 263. Uncovering spatial tissue domains and cell types in spatial omics through cross-scale profiling of cellular and genomic interactions

**arXiv ID:** 2602.12651 | [PDF](https://arxiv.org/pdf/2602.12651v1)

**作者:** Rui Yan `[一作]` (Stanford University), Lei Xing `[通讯]` (Stanford University)

**通讯引用:** 32945 | [OpenAlex ID](https://openalex.org/A5100381484)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并实现了CellScape，一种双分支深度学习框架，通过同时学习细胞间空间关系和细胞内基因共表达模式，实现对空间转录组数据的细胞表型与组织结构的联合表征。

**💡 创新点**

创新点在于融合空间图神经网络与基因表达二维映射CNN的双分支结构，结合自监督对比与重建损失以及批量效应校正，能够在不受空间邻接偏置影响的前提下，提取更具生物学解释性的空间与转录组特征。

**🔧 技术方法**

使用了图注意力网络（GAT）对空间邻接进行编码，卷积神经网络（CNN）对基因共表达二维图进行特征提取，采用对比学习与掩码重建自监督目标，并通过PCGrad平衡多任务梯度。

**📊 数据集**

在Slide-tags人类前额叶皮层、STARmap小鼠阿尔兹海默模型、Slide-seqV2与Stereo-seq鼠嗅球、osmFISH、10x Visium人类DLPFC以及STARmap小鼠mPFC等多种分辨率与平台的空间转录组数据上进行了评估。

**📈 对比分析**

与GraphST、PROST、STAGATE、SEDR、CellCharter、SpaceFlow、SpaGCN及PCA等八种代表方法相比，CellScape在多数据集的空间域分割任务中在NMI和同质性指标上均取得最高或近乎最高的得分，尤其在高分辨率样本中显示出显著的优势。

**⚠️ 局限性**

限制在于对基因数目的二次映射需要额外计算，规模较大时内存与计算开销较高；对多样本批处理的空间图合并仍可能产生残留批效应；此外，模型目前仅针对基于细胞核或细胞质的空间转录组，尚未验证对多组学或超高分辨率时空表观组数据的适用性。

---

## 264. Human-Like Coarse Object Representations in Vision Models

**arXiv ID:** 2602.12486 | [PDF](https://arxiv.org/pdf/2602.12486v1)

**作者:** Andrey Gizdov `[一作]` (Harvard University), Tomer Ullman `[通讯]` (Harvard University)

**通讯引用:** 2623 | [OpenAlex ID](https://openalex.org/A5086092571)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究视觉分割模型在时间-碰撞（TTC）任务中是否能产生与人类相似的粗体量化对象表示，并系统评估训练时长、模型规模与剪枝对其影响。

**💡 创新点**

发现模型的对齐误差随资源投入呈U形曲线，最优的“理想体量化”在中等规模/训练/剪枝下出现，表明粗体表示源于资源限制而非专门偏置；提出统一的TTC对齐框架。

**🔧 技术方法**

使用SegFormer分割网络，基于合成多边形数据进行微调，结合TTC模拟与人类实验数据对齐，采用结构化与非结构化剪枝，并计算误差度量。

**📊 数据集**

使用ADE20K预训练模型、人工生成的500/200图像合成多边形训练集，以及226名参与者收集的96段时间-碰撞视频数据。

**📈 对比分析**

通过比较模型与人类在凹凸面碰撞时间差异Δ的绝对误差E，发现误差随训练/规模/剪枝呈U形曲线，在中等参数下误差最低，证明模型在该范围内与人类行为最接近。

**⚠️ 局限性**

仅限二维多边形、单一分割模型族、仅使用TTC实验，未验证3D情境、更多模型或任务，且未深入探究内部机制。

---

## 265. RoadscapesQA: A Multitask, Multimodal Dataset for Visual Question Answering on Indian Roads

**arXiv ID:** 2602.12877 | [PDF](https://arxiv.org/pdf/2602.12877v1)

**作者:** Vijayasri Iyer `[一作]`, Jyothikamalesh S `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了约9000张印度南部道路图片的Roadscapes多任务多模态数据集，包含目标检测、车道分割和视觉问答（VQA）对。

**💡 创新点**

创新点在于：①聚焦印度多样化非结构化道路环境，填补现有数据集空白；②通过规则启发式与大型语言模型自动生成多类别VQA问答；③提供统一基线与幻觉率分析。

**🔧 技术方法**

技术手段包括：低成本单目摄像头采集、YOLOWorld+人工验证的目标检测、规则推理+LLM生成QA、以及GPT‑4o/4o‑mini、Phi‑3.5、Paligemma等零样本VLM评估。

**📊 数据集**

使用的数据集主要为自采集的Roadscapes（Coimbatore‑Kochi 区域），并与 IDD、KITTI、ROADQA 等公开基准进行对比。

**📈 对比分析**

评估方法为零样本VQA，使用 exact‑match 与余弦相似度；Phi‑3.5 在计数任务上取得 0.667，4o‑mini 0.628，Paligemma 在描述任务上 0.501，4o 在周边描述上 0.701，整体性能低于专用基准，幻觉率相对较高。

**⚠️ 局限性**

局限性包括：任务覆盖有限（缺乏定位/空间关系评估），地理覆盖仅限 Coimbatore‑Kochi，任务复杂度不足，未来需加入多轮对话、时序推理等更高级任务。

---

## 266. GatheringSense: AI-Generated Imagery and Embodied Experiences for Understanding Literati Gatherings

**arXiv ID:** 2602.12565 | [PDF](https://arxiv.org/pdf/2602.12565v1)

**作者:** You Zhou `[一作]` (Hong Kong University of Science and Technology), Zeyu Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1283 | [OpenAlex ID](https://openalex.org/A5100599637)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了基于生成式AI的双路径框架（符号路径+身体参与路径），并通过GatheringSense体验在文人聚会的文化理解中进行验证。

**💡 创新点**

创新点在于首次将AI生成的多模态符号与身体化互动两条路径系统化整合，形成“观‑感‑共鸣”三阶段认知模型，并通过KANO分析提出针对文化体验的设计建议。

**🔧 技术方法**

主要技术包括文本到图像/视频的生成模型（如Diffusion、Stable Diffusion、Flux Pro等）、音乐/音频生成工具（Suno）、混合实验设计与数据分析（LMM、KANO、质性主题分析）。

**📊 数据集**

使用基于《兰亭序》等古典文献的文本语料作为输入，并生成对应的四种视觉风格（写意、工笔、油画、卡通）的图像与短视频作为实验素材。

**📈 对比分析**

通过对48名受试者的混合实验，比较了图像vs.视频的符号可读性、文化共鸣、沉浸感和心理亲近度；结果显示图像在符号可读性上略优，身体参与显著提升文化共鸣与存在感，且两条路径的顺序对主要指标影响不显著。

**⚠️ 局限性**

局限性包括样本规模有限、跨文化与儿童样本缺乏代表性、实验环境仅为室内物理原型、AI生成内容仍存在物理可信度和细节一致性不足等问题。

---

## 267. LatentAM: Real-Time, Large-Scale Latent Gaussian Attention Mapping via Online Dictionary Learning

**arXiv ID:** 2602.12314 | [PDF](https://arxiv.org/pdf/2602.12314v1)

**作者:** Junwoon Lee `[一作]` (University of Michigan), Yulun Tian `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在线 3D Gaussian Splatting（3DGS）映射框架，能够在实时 RGB‑D 流数据中构建几何、外观以及 VLM（视觉‑语言模型）嵌入的稀疏 3D 语义地图，并支持开词汇感知查询；

**💡 创新点**

核心创新是将特征映射转化为在线字典学习问题，使用流式 K‑means 初始化、注意力重构与信赖区间正则化，实现模型无关、无预训练、可随环境动态更新的 VLM 嵌入重建；

**🔧 技术方法**

技术包括 3DGS 作为几何表示、低维查询向量与高维 VLM 嵌入的注意力字典重构、流式 K‑means 字典更新、两阶段优化（全量更新+权重细化）、历史缓冲与信赖区间正则、体素哈希实现局部‑全局地图管理；

**📊 数据集**

使用公开的 Replica、TUM、FastCaMo 大规模数据集以及自制校园两层楼 530 m 的长轨迹数据进行评估；

**📈 对比分析**

与 Feature‑3DGS、M3、Online Language Splatting、OmniMap 等基线对比，指标包括余弦损失、mIoU、PSNR 与 FPS，LatentAM 在特征重建精度上显著优于所有基线，同时帧率保持 12–35 FPS，表现出更好的实时性与大规模可扩展性；

**⚠️ 局限性**

局限性包括相对较低的光照重建质量（PSNR 低于部分离线方法）、依赖外部跟踪器与 RGB‑D 传感器，且在极端遮挡或非光照条件下的鲁棒性尚待进一步验证。

---

## 268. Learning Native Continuation for Action Chunking Flow Policies

**arXiv ID:** 2602.12978 | [PDF](https://arxiv.org/pdf/2602.12978v1)

**作者:** Yufeng Liu `[一作]` (Shanghai Jiao Tong University), Yang Gao `[通讯]` (Tsinghua University)

**通讯引用:** 12926 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Legato，一种在训练阶段实现动作块连续性的框架，使得流基 VLA 策略能够在生成动作块时自然保持平滑与连贯。

**💡 创新点**

创新点在于：①引入基于时间的指导调度（schedule‑shaped guidance）并在每一步去噪时施加；②通过重塑流场（flow dynamics）将该持续性内化为策略本身；③在训练时随机化延迟与梯度长度并将其作为条件输入，实现在不同硬件延迟下的自适应与可调节平滑度。

**🔧 技术方法**

主要技术包括：流匹配（Flow Matching）框架、时间序列指导调度、连续动力学重塑、条件化随机调度、与传统 RTC、训练时 RTC 的对比实验。

**📊 数据集**

使用五个真实机器人操作数据集：堆叠碗、倒料、装盒、折叠毛巾、拉开抽屉，测试机器人完成率、完成时间与轨迹平滑度。

**📈 对比分析**

与实时块化（RTC）以及训练时 RTC 做对比，Legato 在所有任务上均表现更好：任务完成时间缩短约10%，轨迹平滑度（NSPARC、NLDLJ、块间 RMSE）均有显著提升，尤其是减少了多模态切换与犹豫。

**⚠️ 局限性**

局限性：去噪步骤数量在训练时固定，无法在推理时动态调整；方法主要针对流基 VLA，需在其他生成模型上进一步验证。

---

## 269. On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs

**arXiv ID:** 2602.12506 | [PDF](https://arxiv.org/pdf/2602.12506v1)

**作者:** Rosie Zhao `[一作]` (Apple), Arnab Mondal `[通讯]` (Apple)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5045635774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究 RL 微调后多模态推理模型在视觉推理任务中的鲁棒性与链式推理（CoT）的可信度，并通过对文本提示的精细扰动评估模型对视觉信息的真实依赖。

**💡 创新点**

提出针对视觉推理的文本扰动评估框架，系统探讨 RL 微调与数据增强如何同时提升鲁棒性却削弱 CoT 可信度，并揭示现有奖励与数据策略难以兼顾两者的根本原因。

**🔧 技术方法**

使用基于可验证奖励的 RL（Group Relative Policy Optimization）对 Qwen‑2.5‑VL‑7B‑Instruct 进行多轮微调，结合链式推理模板、可辨认奖励与“可信度奖励”实验；同时利用 entropy、P(correct) 等内部指标分析模型不确定性与对抗性鲁棒性。

**📊 数据集**

数据集包括 3DSRBench、CV‑Bench、Spatial‑MM、WhatsUp 等空间推理基准，另外使用 SAT2、Pixmo‑Count、Geometry3K 等训练集，并在评测时加入错误标题、错误 CoT 前缀等人工扰动。

**📈 对比分析**

与基线 Open‑source VLM（SpaceR、Video‑R1、Vision‑R1、VLAA‑Thinker、ViGoRL‑Spatial）在未扰动与扰动条件下的准确率、可信度比例及 entropy 进行对比；结果显示：RL 微调能显著提升标准准确率，但在 Wrong‑Caption/ Wrong‑Think 等扰动下准确率下降，可信度比例普遍下滑；增添多样化扰动的数据增强能提升鲁棒性但仍无法阻止可信度衰退；将可信度纳入奖励能提升 CoT 可信度但易出现捷径，整体性能提升有限。

**⚠️ 局限性**

局限性：仅在单轮推理的简单视觉空间任务上验证；对多轮交互、复杂视觉模态（如视频、3D 体素）缺乏评估；奖励设计和数据增强仍不能完全消除语言优先和 CoT 与视觉不一致的问题；实验受随机种子影响显著，需更多复现与多样化评测。

---

## 270. To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models

**arXiv ID:** 2602.12566 | [PDF](https://arxiv.org/pdf/2602.12566v1)

**作者:** Haoqing Wang `[一作]` (Samsung Research), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多领域强化学习（RLVR）与模型融合的两种主流范式，并通过大规模实验评估其互补性与干扰；

**💡 创新点**

首次从权重空间、策略邻域和信息约束三角度系统阐释了多任务RLVR的协同机制，并揭示了不同任务间的相互促进；

**🔧 技术方法**

采用可验证奖励的GRPO算法、权重融合（平均、任务算术、Ties、SCE）、多教师在线策略蒸馏等技术；

**📊 数据集**

使用Nemotron 3 Nano公开数据集（包括数学、编码、科学、指令跟随等子任务），并在Qwen3-4B-Base上进行SFT+RL训练；

**📈 对比分析**

在9个基准（AIME、LiveCodeBench、GPQA-Diamond、IFEval、IFBench、MMLU-Redux等）上与单任务专家及各融合方法比较，发现混合多任务RLVR仅耗33.2% GPU时长即可匹配或超越单任务+融合方案，且多任务模型在逻辑任务上表现出显著的协同提升；

**⚠️ 局限性**

实验仅覆盖四个领域，未探讨更广泛的跨模态任务，且模型融合方法在特定任务上仍存在性能波动，需进一步研究更稳健的融合策略。

---

## 271. A Microservice-Based Platform for Sustainable and Intelligent SLO Fulfilment and Service Management

**arXiv ID:** 2602.12875 | [PDF](https://arxiv.org/pdf/2602.12875v1)

**作者:** Juan Luis Herrera `[一作]` (TU Wien), Schahram Dustdar `[通讯]` (TU Wien)

**通讯引用:** 37097 | [OpenAlex ID](https://openalex.org/A5004847496)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Carbon-Aware SLO和控制平台CASCA，提供可隐私保护、可插拔的微服务控制与SLO监测框架；

**💡 创新点**

通过微服务化的API网关实现SLO与配置的抽象与解耦，同时引入EMMA统一碳强度接口，实现对碳排放SLO的可观测与控制；

**🔧 技术方法**

采用OpenAPI、Flask、MQTT、InfluxDB等开源技术实现平台，决策系统示例使用Bash、Rust和Python，并利用PPO DRL训练RL决策器；

**📊 数据集**

在真实计算连续试验台上使用Jellyfin媒体流服务、Tasmota功耗监测、EMMA的IPCC及ElectricityMaps碳强度数据；

**📈 对比分析**

与本地原生控制对比，CASCA在决策延迟≈75 ms内实现配置获取/设置；在24小时实验中，RLDS在FPS合规率≈85%，碳排放最低；GDS略低；RDS表现最差；

**⚠️ 局限性**

局限在于实验规模受限于单一服务场景，缺乏对大规模服务链与多云部署的验证；碳强度数据仅限欧洲地区；决策系统未考虑网络抖动与动态迁移等更复杂情境。

---

## 272. ZeroDiff++: Substantial Unseen Visual-semantic Correlation in Zero-shot Learning

**arXiv ID:** 2602.12401 | [PDF](https://arxiv.org/pdf/2602.12401v1)

**作者:** Zihan Ye `[一作]` (University of Chinese Academy of Sciences), Ling Shao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 67925 | [OpenAlex ID](https://openalex.org/A5082634513)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ZeroDiff++，一种基于扩散模型的零样本学习框架，利用扩散增强、监督对比学习和多视角判别器来强化视觉-语义关联，并在测试时通过 Diffusion‑based Test‑time Adaptation 与 Diffusion‑based Test‑time Generation 进一步提升未见类性能。

**💡 创新点**

① 在训练阶段引入扩散增强生成无穷多噪声样本，减轻数据稀缺导致的过拟合；② 使用监督对比学习产生实例级语义表示，缓解语义静态化问题；③ 设计多视角 Wasserstein 互学习损失，融合判别器的视觉、语义与扩散信息；④ 在测试阶段实现自适应生成和部分生成（DiffTTA & DiffGen），实现生成特征与真实测试样本可追踪连接；⑤ 对扩散判别器的过拟合缓解给出理论证明。

**🔧 技术方法**

扩散模型（Diffusion‑based Feature Generator）、监督对比学习（SC）、Wasserstein‑距离互学习、WGAN‑GP、伪标签重构、部分噪声/部分生成、信息理论分析（重叠质量、KL 收缩）。

**📊 数据集**

AWA2、CUB、SUN 三大公开 ZSL 基准数据集。

**📈 对比分析**

与多种 GAN、VAEGAN、Flow、DiffusionZSL 等生成式 ZSL 方法以及 Embedding 基准方法进行对比。ZeroDiff++ 在 ZSL 与 GZSL 任务上均取得显著提升，尤其在仅保留 10% 训练样本时保持高精度，H 值位列最前。

**⚠️ 局限性**

仍受扩散过程计算成本与超参数敏感性影响；需要伪标签与自适应步骤，增加推理复杂度；在更大规模或多模态场景下的泛化性尚未充分验证。

---

## 273. Diverging Flows: Detecting Extrapolations in Conditional Generation

**arXiv ID:** 2602.13061 | [PDF](https://arxiv.org/pdf/2602.13061v1)

**作者:** Constantinos Tsakonas `[一作]` (Inria), Jean-Baptiste Mouret `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在条件生成任务中提出 Diverging Flows（DiFlo），通过在单一流匹配模型中嵌入几何约束，实现对有效条件的高质量预测与对无效（off‑manifold）输入的自适应检测；

**💡 创新点**

创新点在于将对比学习思想迁移到向量场空间：用 Repulsion 与 Curvature 对比正则化强制有效条件沿最优输运直线运动，而无效条件产生不高效的输运路径，从而形成几何相变，可直接在生成过程内部判断异常；

**🔧 技术方法**

技术实现基于 Optimal Transport Flow Matching（OT‑FM），加入两项对比正则化（repel 与 curve）、PGD 产生难负样本、DOT（Deviation‑from‑Optimal‑Trajectory）评分以及分层共形预测来实现阈值化检测；

**📊 数据集**

实验数据涵盖：①二维螺旋仿真数据（用于概率回归与条件生成）；②ERA5 天气温度预测（64×64 热图）六小时预测；③MNIST→SVHN 风格迁移（含 FMNIST、KMNIST 用于评估语义/结构迁移下的检测）；

**📈 对比分析**

与基线 FM（似然、DOT）、HyperDM 及 DiffPath 进行对比：在合成实验 AUROC>0.98、FPR≤5%；天气预测 AUROC 0.98、MSE 0.0034（相对 HyperDM 0.004）；风格迁移 AUROC 0.86、FID 4.10（明显优于基线）；整体检测性能显著提升，预测精度几乎不受影响；

**⚠️ 局限性**

局限性包括：训练过程需迭代生成硬负样本（PGD），导致额外计算成本；目前仅适用于欧氏 Optimal Transport，Riemannian 或更复杂流形的推广仍需研究；对离散或高度结构化流形的支持有限。

---

## 274. Online Flow Time Minimization with Gradually Revealed Jobs

**arXiv ID:** 2602.12716 | [PDF](https://arxiv.org/pdf/2602.12716v1)

**作者:** Alexander Lindermayr `[一作]` (Technische Universität Berlin), Leen Stougie `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在线预抢先调度模型，在单机上最小化总流时间，模型中每个作业由一系列操作组成，操作的处理时间在前一个操作完成后才暴露。

**💡 创新点**

创新点在于引入“操作逐步揭露”的信息模型，弥合了完全先知（SRPT）与无先知（非先知）之间的鸿沟，并在此模型下实现了关于操作数 m 的二阶竞争比（O(m²)）和更细化的 O(m1·m2) 竞争比，其中 m1、m2 为实例特定的块大小和块数上界。

**🔧 技术方法**

主要技术包括：1）构造与块（chunk）相关的线性规划与其对偶，用以上界当前活跃块数；2）基于块的“虚拟作业”策略，将作业拆分为类递增的块并按块类处理；3）采用双重拟合（dual fitting）证明竞争比；4）在特定场景（统一长度测试）下运用 Schrage 的递归 SRPT 分析。

**📊 数据集**

无实际数据集，所有结果均为理论上界与下界。

**📈 对比分析**

实验/理论比较表明：1）一般实例下算法在任何时刻的活跃块数不超过 O(m1·m2) 倍最优活跃块数，积分后得到总流时间的 O(m1·m2) 竞争比；2）当每个作业仅两项且测试长度统一时，Operations‑SRPT 达到 2‑竞争比，证明该比值最优；3）对 m1、m2 的下界分别为 Ω(m1) 与 Ω(m2)，说明结果在这两个维度上已最优；4）仅就 m 来看，已知下界为 Ω(√m)，因此 O(m²) 的上界尚不紧凑。

**⚠️ 局限性**

局限性包括：1）竞争比在单调或首项近似较好时可降至 O(m)，但在最一般情况下仍为 O(m²)；2）相对于仅关注 m 的下界（Ω(√m)），目前的上界距离最优尚远；3）模型假设作业的操作序列可视为决策树，实际系统中可能存在更复杂的依赖结构；4）理论分析依赖于对偶拟合，实际实现需管理块类与队列等数据结构，可能导致实际调度开销未评估。

---

## 275. Theory of Mind Guided Strategy Adaptation for Zero-Shot Coordination

**arXiv ID:** 2602.12458 | [PDF](https://arxiv.org/pdf/2602.12458v1)

**作者:** Andrew Ni `[一作]` (Carnegie Mellon University), Woojun Kim `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2277 | [OpenAlex ID](https://openalex.org/A5028552753)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于理论心智的最佳响应选择（TBS）框架，以实现对未见合作伙伴的零样本协调；

**💡 创新点**

创新点在于通过自适应ToM推理和谱聚类分组，将伙伴划分为行为簇并为每个簇训练专用最佳响应策略，从而实现实时策略选择；

**🔧 技术方法**

使用的技术包括自适应谱聚类、ToM网络（RNN形式的高层意图预测）、策略集成与动态选择、CTDE与VDN的强化学习训练；

**📊 数据集**

实验基于Overcooked-AI中的洋葱汤场景，共七种布局，包含完全可观测和部分可观测两种设置；

**📈 对比分析**

与随机、单一BR、Oracle、CEM、CBPR等基线相比，TBS在大多数布局下的交叉游戏（XP）和交叉算法XP指标上均获得更高的平均奖励，展示了更强的适应性和零样本协调性能；

**⚠️ 局限性**

局限性包括缺乏理论分析、对概念集定义的依赖以及对高层意图推理的误差可能导致策略选择失误。

---

## 276. Human Tool: An MCP-Style Framework for Human-Agent Collaboration

**arXiv ID:** 2602.12953 | [PDF](https://arxiv.org/pdf/2602.12953v1)

**作者:** Yuanrong Tang `[一作]` (Tsinghua University), Jiangtao Gong `[通讯]` (Tsinghua University)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5089973290)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Human Tool 框架，将人类作为可被 LLM 调用的工具（MCP‑style 接口）在 AI 主导的工作流中进行协作；

**💡 创新点**

创新点在于把协作重点从人类监督转移到 AI 协调，将人类视为可点拨的资源而非全局控制者，并通过结构化能力、信息与权限模式实现可控调用；

**🔧 技术方法**

利用大语言模型（GPT‑4o）配合 LangGraph、MCP‑style 工具调用协议、对话策略与人机交互指南，实现 AI 对人类工具的动态分配与通信；

**📊 数据集**

使用了旅行规划任务的数据集（TravelPlanner）和故事创作任务的公开情节提示集合，此外通过内部问卷收集人类能力、信息与授权配置；

**📈 对比分析**

通过对比 Human Tool 与传统 AI Tool（人类主导工具调用）的 32 名参与者实验，分别在旅行规划与故事写作上测评任务准确率、质量、工作负荷、用户体验等指标，Human Tool 在准确率（TP 19.34% 提升）、创意质量（SW 14.35% 提升）、认知负荷与交互满意度方面均显著优于基线；

**⚠️ 局限性**

局限性包括仅在两类任务（决策与创意）与短时实验场景下验证，缺乏对长期或高风险任务的泛化；对时间成本的测量仍依赖自评，未来需加入更客观的效率指标，并完善对用户自主权的进一步保障。

---

## 277. Scoped MSO, Register Automata, and Expressions: Equivalence over Data Words

**arXiv ID:** 2602.13120 | [PDF](https://arxiv.org/pdf/2602.13120v1)

**作者:** Radosław Piórkowski `[一作]` `[通讯]`, Radosław Piórkowski

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构建了适用于无限字母数据词的非确定性注册自动机（NRA）对应的两种新形式化：Scoped MSO 逻辑与 Data-Regular 表达式，并证明它们与 NRA 在表达力上的等价；

**💡 创新点**

首次给出 NRAs 的 MSO 逻辑表述，提出 Scoped MSO 逻辑并引入段式模态与受限数据原子；引入最小化的 Data-Regular 表达式并定义 k-收缩并置来模拟有限注册；

**🔧 技术方法**

采用注册自动机理论、MSO 逻辑变换、k-收缩并置算子、数据词量化区域与分段子域技术；

**📊 数据集**

无实验数据集，完全为理论证明与形式化构造；

**📈 对比分析**

通过结构化归约与翻译证明等价性，未涉及运行时性能评估；

**⚠️ 局限性**

仅适用于可消除强猜测的原子域（如等价原子、稠密序）；对强猜测的 NRAs 与更强逻辑仍为开放问题；

---

## 278. EXCODER: EXplainable Classification Of DiscretE time series Representations

**arXiv ID:** 2602.13087 | [PDF](https://arxiv.org/pdf/2602.13087v1)

**作者:** Yannik Hahn `[一作]` (University of Wuppertal), Tobias Meisen `[通讯]` (University of Wuppertal)

**通讯引用:** 3429 | [OpenAlex ID](https://openalex.org/A5032638290)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究将时间序列通过VQ‑VAE、DVAE或SAX等方法压缩为离散潜在表示，并在此空间上适配并应用多种XAI技术，进一步提出了Similar Subsequence Accuracy (SSA) 指标用于量化解释与训练数据中类相关子序列的一致性。

**💡 创新点**

创新点在于（1）首次将梯度、扰动、注意力等主流XAI方法迁移到离散潜在空间，并通过引入未知token实现遮蔽；（2）提出SSA度量，用于客观评估解释在离散表示中的有效性；（3）发现离散潜在表示能产生更简洁、更具可解释性的解释，同时保持或提升分类性能。

**🔧 技术方法**

使用技术包括：VQ‑VAE、DVAE、SAX（离散化）、Transformer/MLP分类器、SM、IG、RISE、LIME、ATM等XAI方法，结合未知token遮蔽与SSA计算。

**📊 数据集**

数据集涵盖三大类：焊接（双通道）、CNC（三轴加速度）、ECG（单通道心电），每个数据集均包含数万条样本，覆盖多类别与不平衡情况。

**📈 对比分析**

通过在原始时序与离散潜在表示上分别应用XAI，并对比扰动鲁棒性、实现不变性、方法一致性及SSA指标，实验显示离散表示下的XAI解释更紧凑、SSA更高，部分模型组合（如VQ‑VAE+MLP+LIME）在某些数据集上获得最佳性能。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限数据集与模型架构，XAI效果仍高度依赖模型与数据，未针对离散潜在空间专门设计新的XAI方法，且SSA指标仍受子序列长度与邻居数量限制。

---

## 279. TCRL: Temporal-Coupled Adversarial Training for Robust Constrained Reinforcement Learning in Worst-Case Scenarios

**arXiv ID:** 2602.13040 | [PDF](https://arxiv.org/pdf/2602.13040v1)

**作者:** Wentao Xu `[一作]` (Northeastern University), Yushuai Li `[通讯]` (Aalborg University)

**通讯引用:** 3072 | [OpenAlex ID](https://openalex.org/A5008665053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于时序耦合对抗训练的鲁棒约束强化学习框架（TCRL），使得智能体在遭遇时序耦合的观测扰动时仍能最大化奖励并严格满足安全约束。

**💡 创新点**

创新点在于：
1) 设计了可直接估计时序耦合极端安全成本的约束函数；
2) 引入双重奖励约束（自相关抑制与熵稳定性）以打断攻击者的时序模式并保持奖励不可预测性；
3) 通过对抗训练同时考虑安全成本与奖励的时序耦合扰动，实现对 Worst‑TC 攻击的高效防御。

**🔧 技术方法**

使用的技术包括：
- 基于CMDP的约束强化学习（PID‑PPO‑Lagrange）
- 观测扰动的时序耦合攻击模型与对应的约束
- 价值网络的极端成本 Bellman 运算与损失函数
- 双重奖励约束的自相关正则化与熵稳定正则化
- 对抗训练中的 Lagrangian 约束更新

**📊 数据集**

实验数据集为四个机器人控制任务：Ball‑Circle、Ball‑Run、Car‑Circle、Car‑Run，均采用仿真环境中的机器人运动控制数据。

**📈 对比分析**

与三种基线（PPOL‑vanilla、PPOL‑random、ADV‑PPOL(MC)）在自然、MAD、MC、随机、Worst‑TC 五种攻击下对比。TCRL 在 Worst‑TC 攻击下安全成本下降 559–19,078%（视任务而定），奖励提升 8–33% 以上；在无攻击或非耦合攻击场景下保持低安全成本并可获得与基线相当或更高的奖励，表现出显著优于基线的鲁棒性。

**⚠️ 局限性**

局限性包括：
- 对极端时序耦合攻击的防御仍依赖于攻击模型的可学习性，可能在未知攻击策略下失效；
- 训练过程较为复杂且计算成本高，尤其是双重奖励约束与极端成本网络的联合优化；
- 只在仿真机器人控制任务中验证，缺乏真实物理环境或更大规模任务的评估；
- 目前仅考虑观测扰动，未扩展到动作扰动或环境模型变化的鲁棒性。

---

## 280. MentalBench: A Benchmark for Evaluating Psychiatric Diagnostic Capability of Large Language Models

**arXiv ID:** 2602.12871 | [PDF](https://arxiv.org/pdf/2602.12871v1)

**作者:** Hoyun Song `[一作]` (Korea Advanced Institute of Science and Technology), KyungTae Lim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5003224328)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基准（MentalBench），包括由精神科医生构建的DSM‑5知识图谱和24,750个可控合成临床案例，用于评估大型语言模型在精神疾病诊断决策中的表现。

**💡 创新点**

创新点在于：① 通过专家验证的知识图谱精准编码DSM‑5诊断标准和鉴别规则；② 采用规则一致的生成方法创建信息完整性与诊断复杂度可调的合成案例；③ 揭示LLM在诊断信心校准与多答案决策上的显著局限。

**🔧 技术方法**

使用技术包括：结构化知识图谱构建、规则一致的案例生成、LLM推理与评估（GPT、Gemini、Claude、开源模型等）、专家人工验证与细粒度错误分析。

**📊 数据集**

数据集为基于DSM‑5知识图谱构造的24,750个合成临床案例，涵盖医疗记录式、患者自述、模糊与明确鉴别诊断等四种类型。

**📈 对比分析**

通过精确匹配（exact match）准确率比较多款LLM，结果显示专有模型总体优于开源模型，模型规模越大性能越好，但在信息不完整或鉴别诊断场景下准确率显著下降；在多答案/单答案约束下模型表现差异明显，揭示出诊断信心校准的不足。

**⚠️ 局限性**

局限包括：① 仅基于单一文本叙述，缺乏对话交互；② 仅提供英文案例，跨语言与文化适用性有限；③ 只评估最终诊断结果，未检验模型内部推理过程；④ 合成案例虽低噪声但仍可能与真实临床情境存在差距。

---

## 281. CoPE-VideoLM: Codec Primitives For Efficient Video Language Models

**arXiv ID:** 2602.13191 | [PDF](https://arxiv.org/pdf/2602.13191v1)

**作者:** Sayan Deb Sarkar `[一作]` (Stanford University), Mihai Dusmanu `[通讯]` (Microsoft Spatial AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出利用视频压缩编码器中的运动向量和残差（即P帧差分信息）直接生成轻量化的“Δ‑token”，与传统RGB帧Token混合，以实现更高的token压缩率和更低的时间延迟；

**💡 创新点**

创新点在于：①将压缩域的运动向量与残差分别通过轻量级Transformer分支压缩为Token；②通过预训练的Δ‑Encoder将这些Token与RGB Token空间对齐，使得P帧Token可以完全替代RGB Token；③引入P帧融合与GOP结构的自适应token化，显著提升长视频上下文利用率；

**🔧 技术方法**

使用的技术包括：MPEG‑4/H.264/HEVC视频编码原生 primitives、轻量级Δ‑Encoder（motion transformer + residual transformer）、MSE 对齐预训练、LLaVA‑Video‑7B + SigLIP + Qwen2 作为基线模型、P‑frame融合策略以及多任务 instruction‑tuning；

**📊 数据集**

训练数据集：PerceptionTest 0‑30 s视频用于预训练；微调使用 LLaVA‑Video‑178K（1.39M QA）数据集；评估覆盖14个 benchmark，涵盖通用QA、时序推理、长视频指令跟随以及空间场景理解（PerceptionTest、NextQA、ActivityNet‑QA、VideoMME、TempCompass、TOMATO、CVRR‑ES、MVBench、LongVideoBench、LVBench、Video‑TT、Video‑MMMU、ScanQA、SQA3D）；

**📈 对比分析**

方法上与基线 LLaVA‑Video‑7B 以及多款开源 VideoLM 进行同 token‑budget 对比；结果显示在 token 数量减少 7%–93% 的情况下，准确率提升或与基线持平；TTFT 下降最高 86%，E2EL 降低 56%；在 1 FPS 输入下可处理长达 8 小时视频，显著提升上下文窗口利用率；

**⚠️ 局限性**

局限性包括：仅处理 I‑和 P‑帧，未支持 B‑帧及其非因果依赖；使用的是张量化的 codec primitives，未直接操作原始块级运动向量和量化 DCT；固定的 P‑帧融合窗口在不同运动场景下可能不最优；以及在部分基准上仍落后于大型闭源模型。

---

## 282. VimRAG: Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph

**arXiv ID:** 2602.12735 | [PDF](https://arxiv.org/pdf/2602.12735v1)

**作者:** Qiuchen Wang `[一作]` (Alibaba Group), Ruixue Ding `[通讯]` (Alibaba Group)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5033736743)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于动态有向无环图的多模态检索增量推理框架VimRAG，解决传统RAG在视觉信息多且稀疏时的记忆与推理瓶颈；

**💡 创新点**

创新点包括：①将推理过程建模为图结构，显式捕获动作与多模态证据的逻辑依赖；②图调制视觉记忆编码机制，依据能量值自适应分配视觉token，兼顾精细度与效率；③图引导策略优化，通过图结构剪枝实现逐步信用分配，避免粗糙的全局奖励误导；

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3-VL）、图神经网络实现记忆图演化、能量计算与视觉token压缩，以及强化学习（REINFORCE/PG）进行策略训练；

**📊 数据集**

在HotpotQA、SQuAD、WebQA、SlideVQA、MMLongBench、LVBench、WikiHowQA、SyntheticQA以及新构建的XVBench等多模态问答与视频理解基准上进行评测；

**📈 对比分析**

与ReAct、MemAgent、Mem1、VideoRAG、UniversalRAG等现有基线相比，VimRAG在大部分基准上提升5–10%的准确率，并在推理步骤、token消耗和检索命中率等方面表现更优；

**⚠️ 局限性**

局限性包括：对极大规模视频长序列的实时处理仍有挑战；模型训练成本高，需大规模GPU；图结构的构建与更新可能导致额外计算开销；对低质量视觉输入的鲁棒性待进一步提升。

---

## 283. Reasoning to Rank: An End-to-End Solution for Exploiting Large Language Models for Recommendation

**arXiv ID:** 2602.12530 | [PDF](https://arxiv.org/pdf/2602.12530v1)

**作者:** Kehan Zheng `[一作]` (Tsinghua University), Hongning Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5164 | [OpenAlex ID](https://openalex.org/A5085094109)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种端到端的推荐框架，利用大语言模型进行逐项推理，并将推荐收益直接嵌入语言模型的生成过程。

**💡 创新点**

创新点在于：①使用位置无关的逐项推理与Plackett‑Luce可微分代理实现列表级奖励反向传播；②引入自检式监督微调(SFT)以稳定推理模式；③通过强化学习将列表指标（如NDCG）直接优化到token级生成。

**🔧 技术方法**

技术主要包括：大语言模型(Qwen‑2.5‑Instruct)、链式推理(CoT)、自检式SFT、Plackett‑Luce分布、PPO/REINFORCE强化学习、以及评分头的可微分映射。

**📊 数据集**

使用公开的Amazon三大数据集（Musical Instruments、Movies & TV、Video Games）和一大规模工业广告数据集进行评测。

**📈 对比分析**

与传统序列推荐器（SASRec、BERT4Rec、LightGCN）及LLM基线（DeepSeek‑R1、Prompt4NR、TALLRec、R²EC）对比，模型在NDCG@10上持续领先，工业数据上提升至0.818，Amazon数据上均优于或与最强基线相当。

**⚠️ 局限性**

局限性包括：①逐项独立评估可能缺乏全局可比性；②对历史顺序和候选位置仍存在一定敏感性；③对极端冷启动或非常稀疏的项目仍需进一步提升。

---

## 284. Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting

**arXiv ID:** 2602.12774 | [PDF](https://arxiv.org/pdf/2602.12774v1)

**作者:** Xiaowen Zhang `[一作]` (Tongji University), Miaojing Shi `[通讯]` (Tongji University)

**通讯引用:** 2584 | [OpenAlex ID](https://openalex.org/A5101675323)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了WS-COC框架，利用多模态大型语言模型实现弱监督、类别无关目标计数；

**💡 创新点**

创新点在于三种策略：分段对话调优（D^3T）、比较排序优化（CRCO）以及全局‑局部计数增强（GLCE），显著提升密集场景计数精度；

**🔧 技术方法**

采用多轮对话学习范围判断、相对排名训练与分区局部计数融合，并基于LoRA微调的LLaVA-OneVersion-7B等MLLM实现；

**📊 数据集**

在FSC-147、CARPK、PUCPR+和ShanghaiTech四个公开计数基准上进行评估；

**📈 对比分析**

与多种全监督和弱监督计数方法对比，WS-COC在MAE、RMSE上与部分全监督方法持平甚至更优，尤其在密集场景下性能提升显著；

**⚠️ 局限性**

局限在于仍需依赖大规模预训练MLLM，训练与推理显存占用较高，在极端密集或多类别混合场景中计数偏差仍存在。

---

## 285. Evaluating Robustness of Reasoning Models on Parameterized Logical Problems

**arXiv ID:** 2602.12665 | [PDF](https://arxiv.org/pdf/2602.12665v1)

**作者:** Naïm Es-sebbani `[一作]` (CRIL UMR 8188, University Artois), Zied Bouraoui `[通讯]` (CRIL UMR 8188, University Artois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个可参数化的 2-SAT 基准，通过多种结构化 2-CNF 生成器（冲突循环、自由变量、背骨、桥接等）和语义保持扰动（顺序重排、填充子句、变量重命名、重复子句）来评估 LLM 推理模型在逻辑推理任务中的鲁棒性。

**💡 创新点**

创新点在于：①提出了可调节结构特征的 2-CNF 生成器，能精确控制冲突核心大小、自由变量比例、背骨强度等参数；②引入多种语义保持扰动，用于诊断模型对结构变化的敏感度；③通过决策准确率与可验证赋值双重指标，揭示模型在表面相同但结构不同的实例中表现的“脆弱性”。

**🔧 技术方法**

技术手段包括：参数化实例生成、模板化与 LLM 语义化两种自然语言化方式、链式推理（CoT）提示设计、语义验证器提取并校验模型输出的赋值，最终对比不同模型在同一结构与扰动条件下的表现。

**📊 数据集**

使用的数据集为自定义 2-SAT 语料，包含 7 种生成器，每种生成器 10 条公式，分别用 6 种模板或 3 种 LLM 叙事主题化成自然语言，共计数千条实例，覆盖从小到大、从简单到复杂的多维参数空间。

**📈 对比分析**

对比方法：在固定实例规模下，对各模型的决策准确率（SAT/UNSAT）与 SAT 实例的赋值有效率进行评估，观察不同生成器和扰动条件下的性能变化。实验结果显示：在同等规模下，特定结构干预会导致模型性能骤降，模型对子句顺序、填充子句和重复子句极度敏感；模板化语义化显著优于 LLM 叙事化，表明模型更依赖清晰可视的结构线索。

**⚠️ 局限性**

局限性：①仅限 2-SAT 领域，未验证更复杂约束（k-SAT、CNF 等）；②模型对结构不变性的缺乏导致评估不够全面；③验证仅检测赋值的可行性，未深入分析模型推理路径与内部机制；④生成实例规模相对有限，难以全面覆盖所有可能的结构组合。

---

## 286. DRAMatic Speedup: Accelerating HE Operations on a Processing-in-Memory System

**arXiv ID:** 2602.12433 | [PDF](https://arxiv.org/pdf/2602.12433v1)

**作者:** Niklas Klinger `[一作]` (University of Luebeck), Thomas Eisenbarth `[通讯]` (University of Luebeck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

实现并评估了在UPMEM通用PIM（DPU）上执行全流程的同态加密（HE）核心操作，包括BGV乘法、NTT与iNTT，并通过多种算法与硬件级优化提升执行效率。

**💡 创新点**

创新点在于：①将NTT、iNTT与HE乘法完整落在DPU上，避免多阶段、频繁数据搬移；②结合RNS、Barrett约简、位反转内存布局与自定义32位乘法，显著提升算术性能；③采用粗细多线程划分、数据压缩与内存布局优化，降低数据传输开销；④对比前人实现，获得1.4–380倍速度提升，并支持更大参数。

**🔧 技术方法**

使用技术包括：UPMEM PIM架构（DPU、MRAM/WRAM、DMA）、Residue Number System (RNS)、Number‑Theoretic Transform (NTT) 采用Cooley‑Tukey/ Gentleman‑Sande butterfly、Barrett约简、定制32位乘法、位反转内存布局、粗细多线程划分、数据压缩与能耗测量。

**📊 数据集**

实验数据集基于公开的HE标准参数：多种多项式长度（1024、2048、4096、8192）对应的系数位宽（27、54、109、218）以及多达32768个密文；对比Microsoft SEAL（CPU）与前人MHY+实现。

**📈 对比分析**

比较方法：在相同PIM/CPU平台上执行相同的NTT、iNTT与BGV乘法，测量纯计算时间、数据传输与检索开销以及系统功耗。结果显示：与MHY+相比，DRAMatic纯计算速度提升1.4–380倍，数据传输/检索开销降低630倍；与SEAL相比，DRAMatic在纯计算上慢3–7倍，能耗高10–5倍；模拟加速乘法后可逼近或略优于SEAL。

**⚠️ 局限性**

主要局限：DPU仅支持8×8位硬件乘法，导致乘法成为瓶颈；数据传输/检索开销仍显著，尤其在乘法加速后占比提升；缺乏2D/4D NTT支持；PIM系统基线功耗高，能耗相对CPU较差；需要硬件扩展（更快乘法单元、CPU–DPU直接共享内存）与混合计算机制以进一步提升性能。

---

## 287. Revealing Process Structure in Urban Mobility Networks

**arXiv ID:** 2602.13082 | [PDF](https://arxiv.org/pdf/2602.13082v1)

**作者:** Khristina Filonchik `[一作]` (NOVA Information Management School), Fernando Bacao `[通讯]` (NOVA Information Management School)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用通话记录（CDR）构建案例中心和对象中心的事件日志，运用流程挖掘方法（如DFG、Petri网）分析城市移动模式，并与传统调查数据进行对比验证。

**💡 创新点**

首次在大规模、噪声强的移动数据上实现对象中心流程挖掘（OCPM），展示其在多模态分析中的价值；并提供可复现的CDR预处理与日志生成流水线。

**🔧 技术方法**

使用Python开源库infostop、trackintel进行停留点提取；PM4Py进行事件日志挖掘、DFG绘制与模型合规性评估；统计回归分析比较CDR与调查结果。

**📊 数据集**

巴塞罗那（Lisbon）大都市区的CDR数据，涵盖43,529,555条事件，129,032名用户，聚焦Oeiras市内外的旅行，时间范围为2024年2-3月。

**📈 对比分析**

通过线性回归与调查数据对比，发现R²=0.91，说明CDR估计与调查高度相关；案例中心日志模型拟合度为1.00，展示完全符合，且对象中心日志提供了更细粒度的多实体关联信息。

**⚠️ 局限性**

受限于CDR空间分辨率仅至基站扇区、交通模式标签仅为启发式、以及数据稀疏性导致的流程模型复杂性，缺乏精确的车辆与服务层级信息。

---

## 288. Machine Learning-Based Classification of Jhana Advanced Concentrative Absorption Meditation (ACAM-J) using 7T fMRI

**arXiv ID:** 2602.13008 | [PDF](https://arxiv.org/pdf/2602.13008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 289. SCOPE: Selective Conformal Optimized Pairwise LLM Judging

**arXiv ID:** 2602.13110 | [PDF](https://arxiv.org/pdf/2602.13110v1)

**作者:** Sher Badshah `[一作]` (Dalhousie University), Hassan Sajjad `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SCOPE 框架，通过双向偏好熵（BPE）与 conformal 风险控制实现 LLM 的安全、可覆盖的 pairwise judging；

**💡 创新点**

创新点在于：① 设计 BPE 抑制位置偏差的无偏不确定度估计；② 采用 conformal 风险控制在有限样本下对错误率给出严格界限；③ 将两者结合，实现可调风险的自适应判断；

**🔧 技术方法**

使用技术包括选择性 conformal prediction、线性期望理论、双向概率聚合、熵基不确定度计算、阈值校准与风险控制；

**📊 数据集**

实验数据集为 MT‑Bench、RewardBench 与 Chatbot Arena；

**📈 对比分析**

与 Predictive Probability、Verbalized Confidence、Simulated Annotators、Vanilla、Heuristic、Naïve 等基线对比，BPE 在 ECE、AUROC、AUPRC 上优于基线；Scope 在 α=0.10 时满足错误率上限且覆盖率约为 Naïve 的 2 倍，整体覆盖率与风险控制性能优秀；

**⚠️ 局限性**

局限性包括：依赖样本交换性假设；双向评估增加约两倍前向传递成本；仅适用于二元对比，难以直接扩展到多选或评分；黑盒 API 环境下无概率输出时难以直接使用。

---

## 290. High-dimensional Level Set Estimation with Trust Regions and Double Acquisition Functions

**arXiv ID:** 2602.12391 | [PDF](https://arxiv.org/pdf/2602.12391v1)

**作者:** Giang Ngo `[一作]` (Applied Artificial Intelligence Initiative, Deakin University), Sunil Gupta `[通讯]` (Applied Artificial Intelligence Initiative, Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向高维水平集估计（LSE）的算法TR LSE，该算法利用多个信任区域（TR）结合全局与局部采集函数，在有限评估预算内高效定位阈值边界并构建准确的分类器。

**💡 创新点**

创新点包括：① 在高维下首次将信任区域框架与双层采集函数（全局Straddle + 局部Straddle）联合使用，② 通过基于置信区间的惩罚函数自适应调整TR体积与位置，② 对TR外部区域提供理论上限为ε的分类准确性保证，③ 在实验中证明在高达1000维的实测问题上可实现显著的样本效率与计算效率。

**🔧 技术方法**

采用的技术主要有：高斯过程（GP）作为代理模型；Straddle采集函数（可替换为C2LSE、Thompson等）用于全局与局部信息获取；TuRBO式的信任区域更新规则；信息增益与贝叶斯优化理论用于证明分类准确性；标准化与均值平移的预处理以消除GP零均值偏差。

**📊 数据集**

实验数据集包括：八个合成函数（Levy, AA33, Mazda, Ackley, Trid, Rosenbrock 等）与三十个真实工程场景（化学反应、环境监测、车辆结构设计等），维度范围从2维到1000维。

**📈 对比分析**

对比方法：随机采样、Straddle heuristic（STR）与HLSE（高维LSE使用BNN）。在F1-score、召回率等指标上，TRLSE在多数任务上与基线相当或更优，尤其在高维场景中大幅提高准确率并显著降低内存消耗；在运行时方面，TRLSE的计算量仅为HLSE的10%~20%，且在1000维时仍能完成。

**⚠️ 局限性**

局限性：① 理论准确性保证仅适用于TR外部区域，TR内部由于体积小而缺乏统一证明；② 对GP假设（高斯噪声、平滑性）敏感；③ 对极高维（>1000）尚未验证；④ S函数、β等超参数需要经验调优；⑤ 在极端噪声或非高斯噪声场景下，理论与性能需进一步扩展。

---

## 291. What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis

**arXiv ID:** 2602.12395 | [PDF](https://arxiv.org/pdf/2602.12395v1)

**作者:** Xirui Li `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过引入Frankenstein式多维分析框架，对视觉语言模型在RL后训练中的内部机制进行分层功能定位、参数更新几何分析以及模型合并与冻结验证，探究RL在视觉推理中的真正作用。

**💡 创新点**

创新点在于将功能定位、参数更新结构化、模型合并与冻结验证三大手段结合起来，系统揭示RL在中后Transformer层的结构化优化与视觉-推理对齐提升的共性，并通过细粒度指标验证这一效果。

**🔧 技术方法**

使用的技术包括：视觉token交换和层级跳过的因果探测、参数更新能量与多样性的Frobenius范数与奇异值谱分析、模型合并（region-wise merging）以及区域冻结实验来验证因果必要性。

**📊 数据集**

实验采用OpenMMReasoner、MMR1、Revisual等视觉推理数据集，并在MathVista、MathVerse、MathVision、LogicVista等整体评测集上评估。

**📈 对比分析**

通过将Base、IN、RL三阶段模型在vision、alignment、reasoning等细粒度指标以及整体准确率进行对比，发现整体准确提升但细粒度不均衡；RL显著提升中后层的视觉-推理对齐和推理能力，合并实验验证其可迁移性，冻结实验证明中后层更新是RL收益的必要条件。

**⚠️ 局限性**

局限性包括：仅在Transformer层级粗分区上做功能定位，缺乏更细粒度的跨模型泛化验证；实验仅覆盖少数视觉推理数据集与训练recipe；未深入探讨视觉表征本身的提升机制。

---

## 292. Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening

**arXiv ID:** 2602.12679 | [PDF](https://arxiv.org/pdf/2602.12679v1)

**作者:** Wooseok Jeon `[一作]` (Yonsei University), Hae-Gon Jeon `[通讯]` (Yonsei University)

**通讯引用:** 2859 | [OpenAlex ID](https://openalex.org/A5041516963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练的推理时动量先验蒸馏方法（MPD），解决时间倒置采样中正反向路径的运动先验冲突，提升生成中间帧的时间连贯性。

**💡 创新点**

核心创新在于通过在早期采样阶段将前向路径的运动残差蒸馏到后向路径，消除双向路径间的运动先验冲突，形成单一一致的运动轨迹。

**🔧 技术方法**

使用Stable Video Diffusion（SVD）模型，结合Euler采样、CFG++、时间翻转和自定义的残差蒸馏公式实现推理时蒸馏。

**📊 数据集**

在DAVIS和Pexels两大视频关键帧数据集上进行评估，分别使用100和45对关键帧对。

**📈 对比分析**

与TRF、ViBiD、GI、FCVG、FILM、DynamiCrafter等SOTA方法对比，MPD在LPIPS、FID、FVD、VBench++等指标上均优于基线，并在用户研究中获得最高自然度和最低伪影评分。

**⚠️ 局限性**

局限性包括：需要在早期采样阶段进行额外的重噪步骤，略微增加推理时间；对极端大位移或复杂运动仍可能出现残余伪影；MPD在完整采样阶段使用会导致性能下降。

---

## 293. "Not Human, Funnier": How Machine Identity Shapes Humor Perception in Online AI Stand-up Comedy

**arXiv ID:** 2602.12763 | [PDF](https://arxiv.org/pdf/2602.12763v1)

**作者:** Xuehan Huang `[一作]` (Hong Kong University), Ray LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并评估了一种基于机器身份的 AI 并排喜剧表演系统，利用机器自身的计算特性生成并呈现自我指涉式笑料，形成在线实时互动的喜剧体验。

**💡 创新点**

创新点在于：① 将机器身份（而非人类身份）作为喜剧创作的核心资源；② 通过分层 Prompt 设计，将人类喜剧中的身份构建、节奏、反馈等策略迁移至 AI；③ 在实时表演中加入观众反应驱动的自适应机制，模拟现场笑声节奏。

**🔧 技术方法**

技术实现主要依托 GPT‑4‑mini 生成文本，结合 OpenAI TTS 合成语音；前端使用 React，后端 Flask + WebSocket 实时推送；通过结构化 Prompt 规则（身份构建、幽默技巧、三段式笑点、时序停顿等）控制内容；并用实时计数按钮记录 “Haha/Applaud” 反馈以驱动后续笑料的生成。

**📊 数据集**

数据集方面并未使用专门的笑料语料库，而是基于：① 5 位专业脱口秀演员访谈的定性资料；② 50 条 YouTube 脱口秀视频的内容编码；② 32 名参与者的问卷与访谈数据，构成评测样本。

**📈 对比分析**

实验采用 Within‑Subject 对照设计，32 名参与者先后观看基线系统（普通 Prompt）和实验系统（机器身份 Prompt），随后完成问卷与焦点访谈。结果显示，实验系统在幽默感、人格友善度、温暖度、拟人化与动画感等指标上均显著优于基线（p<0.05，效应量 r≈0.5），表明机器身份策略有效提升了 AI 观众体验。

**⚠️ 局限性**

局限性包括：① 样本主要为亚洲受众，跨文化普适性待验证；② 在线演出缺乏现场声学与身体语言；③ 反馈仅限“笑/掌声”按钮，交互方式单一；④ 语音使用人类声线，未体现机器身份；⑤ 体验时长短，未观察长时间喜剧演出的疲劳与记忆效应；⑥ 计数按钮可能产生社交证明偏差。

---

## 294. CRAFT: Adapting VLA Models to Contact-rich Manipulation via Force-aware Curriculum Fine-tuning

**arXiv ID:** 2602.12532 | [PDF](https://arxiv.org/pdf/2602.12532v1)

**作者:** Yike Zhang `[一作]` (Hunan University), Jingtao Sun `[通讯]` (National University of Singapore)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5055922003)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 CRAFT 框架，在 Vision‑Language‑Action (VLA) 模型上通过变分信息瓶颈和力感知训练曲线，实现了对接触丰富操作的高效适配。

**💡 创新点**

创新点在于：①将变分信息瓶颈 (VIB) 插入视觉‑语言编码器后，初期压制高熵感知信息，仅关注低熵力信号；②采用“力先学”到“多模态”递进的学习曲线；③基于主从遥操作的同源系统，低成本采集同步视觉、语言和力数据。

**🔧 技术方法**

技术手段包括：变分信息瓶颈 (VIB)、力感知（关节力反馈）作为任务关键自感知；基于 VIB 的 KL 正则化调度；在 RDT 与 π₀ 两种 VLA 架构上进行轻量级微调。

**📊 数据集**

使用了五个真实世界接触任务的数据集：USB 插入、翻转纸箱、擦白板、塑料泥滚压以及轴‑孔插入，并通过主从遥操作收集多模态演示。

**📈 对比分析**

在相同的 50 条演示下，CRAFT 在 π₀ 和 RDT 上的平均成功率分别提升了 35.36% 与 25.66%，在 OOD（对象/任务变异）实验中平均提升至 58.75%；相较于 FACTR、π₀-base、RDT 等基线，CRAFT 取得了显著的性能优势。

**⚠️ 局限性**

局限性包括：①依赖大量遥操作演示；②目前仅利用关节力作为感知，未涵盖更细粒度触觉；③实验任务有限，未验证在更复杂或多机器人场景下的鲁棒性。

---

## 295. SPRig: Self-Supervised Pose-Invariant Rigging from Mesh Sequences

**arXiv ID:** 2602.12740 | [PDF](https://arxiv.org/pdf/2602.12740v1)

**作者:** Ruipeng Wang `[一作]` (University of Pennsylvania), Miaowei Wang `[通讯]` (University of Edinburgh)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5071458681)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SPRig 框架，利用未标注的动画网格序列对现有静态 Rigging 模型进行自监督 fine‑tune，以提升其时序一致性和姿态不变性。

**💡 创新点**

创新点包括：①单帧锚点自监督的 token‑space 与 geometry‑space 双重一致性损失；②针对 skinning 的 articulation‑invariant distillation 与几何先验；③提出 permutation‑invariant 的时序稳定评估指标。

**🔧 技术方法**

使用 Transformer‑based autoregressive 骨架生成器（如 Puppeteer/UniRig）、自监督损失（KL、L1、entropy、几何先验）、点云采样与 barycentric 传递、Procrustes 对齐等技术。

**📊 数据集**

主要数据集为 DeformingThings4D（DT4D） 动态网格序列，静态评测则使用 Articulation‑XLv2。

**📈 对比分析**

与 Puppeteer 对比，DT4D 上 PJDD 减少 25×、BLRD 降至一半，骨架一致性显著提升；在 Articulation‑XLv2 上保持或略优静态指标，显示 SOTA 时序稳定且不牺牲静态质量。

**⚠️ 局限性**

局限在于依赖单帧锚点假设，对长序列或复杂动态的适应性有限；缺乏对长时序一致性的评估；若基模型性能较差，fine‑tune 效果受限。

---

## 296. CacheMind: From Miss Rates to Why -- Natural-Language, Trace-Grounded Reasoning for Cache Replacement

**arXiv ID:** 2602.12422 | [PDF](https://arxiv.org/pdf/2602.12422v1)

**作者:** Kaushal Mhapsekar `[一作]`, Samira Mirbagher-Ajorpaz `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套基于缓存访问追踪的分析框架，结合符号‑语义过滤器（Sieve）和LLM驱动的检索器（Ranger），实现对缓存调度策略的自动化调试与解释。

**💡 创新点**

创新点在于：①双检索器架构—Sieve提供高精度结构化查询过滤；Ranger利用LLM动态生成并执行代码，支持开放式、非模板化查询；②生成式LLM对检索结果进行上下文化解释。

**🔧 技术方法**

采用的技术包括：句子嵌入（extract workload / policy），符号与语义过滤，pandas DataFrame数据处理，Python代码生成与执行，OpenAI GPT‑4o模型进行检索与生成。

**📊 数据集**

使用ChampSim产生的缓存访问追踪数据集，涵盖多种工作负载（如mcf、lbm等）与替换策略（如PARROT）以及对应的PC、地址、eviction信息。

**📈 对比分析**

通过对比传统模板化查询（Sieve）与LLM代码生成查询（Ranger）的效果，Sieve在结构化问题上实现高精度检索；Ranger则在开放式、动态查询中保持良好性能，整体显著提升缓存策略调试效率，尽管未给出具体数值，但表明两者互补可满足不同查询需求。

**⚠️ 局限性**

局限性包括：①依赖大型语言模型，部署成本和推理延迟较高；②仅针对ChampSim格式的追踪数据，通用性受限；③对非常复杂的动态查询，Ranger的生成准确性仍有提升空间；④数据库结构与代码生成模板需要维护，若更换追踪格式需重新适配。

---

## 297. QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching

**arXiv ID:** 2602.12609 | [PDF](https://arxiv.org/pdf/2602.12609v1)

**作者:** Ke Xu `[一作]` (Anhui University), Xingyi Zhang `[通讯]` (Anhui University)

**通讯引用:** 18526 | [OpenAlex ID](https://openalex.org/A5028634381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 QuEPT，一种基于后训练的多位宽量化框架，可一次校准并在不同位宽间实时切换，适用于 Vision Transformer、LLM 与 MLLM。

**💡 创新点**

创新点在于：① Multi‑Bit Token Merging (MB‑ToMe) 通过余弦相似度选择高位宽 token 并融合低位宽 token，提升跨位宽鲁棒性；② Multi‑Bit Cascaded LoRA (MB‑CLoRA) 采用级联低秩适配器共享参数，兼顾低、中、高位宽的量化误差，实现在单次优化下覆盖多位宽。

**🔧 技术方法**

使用后训练量化、低秩适配器 LoRA、量化校准与剪裁参数联合优化、MAE 损失、token 余弦相似度判断、层感知 KL 与 DP 归一化、token 级融合策略、级联 LoRA 参数共享。

**📊 数据集**

数据集包括：ImageNet（ViT 训练与验证），WikiText2、C4（LLaMA 语言建模），MMLU、TextVQA、VizWiz、OCRBench、SEED（LLaVA‑OV 多模态评估）。校准数据分别为 1024 张 ImageNet 图像、128 个 C4 样本和 128 张 COCO caption 对。

**📈 对比分析**

与多种 PTQ/量化基线（PTQ4ViT、PDQuant、ERQ、PTMQ、AWQ、GPTQ、OmniQuant、SmoothQuant、QLLM、QuaRot、DuQuant、MBQ 等）对比，在 ViT、LLM 与 MLLM 上均取得同类最佳或更优性能；如 ViT‑S W4A4 准确率比 ERQ 高 6.2%，LLaMA2‑7B W4A4 PPL 低于 QuaRot 与 SpinQuant，LLaVA‑OV W4A8 准确率比 MBQ 高 4.3%；训练时间显著降低（比 ERQ 低 50% 以上，PTMQ 仅 1/26 GPU 负载）。

**⚠️ 局限性**

局限性：未针对 LLM 的 outlier 进行显式处理，极低位宽（≤4 位）下仍存在性能瓶颈；若与 SpinQuant 等异常值抑制方法结合可能进一步提升效果。

---

## 298. DPUConfig: Optimizing ML Inference in FPGAs Using Reinforcement Learning

**arXiv ID:** 2602.12847 | [PDF](https://arxiv.org/pdf/2602.12847v1)

**作者:** Alexandros Patras `[一作]` (University of Thessaly), Nikolaos Bellas `[通讯]` (University of Thessaly)

**通讯引用:** 1091 | [OpenAlex ID](https://openalex.org/A5062849403)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于强化学习的 FPGA 上 DPU 配置管理框架，在运行时动态选择最优 DPU 配置以实现能效与延迟的平衡。

**💡 创新点**

创新点在于将上下文感知的奖励设计与自定义 RL 代理结合，用于在可重构 DPU 的大规模配置空间中实时决策；首次将 RL 应用于可重构 DPU 的动态配置。

**🔧 技术方法**

使用了强化学习（PPO）、Vitis AI 编译链、OpenTelemetry 监控、Python/NumPy 进行特征提取、Ray RLLib 训练，框架运行在 Xilinx Zynq UltraScale+ ZCU102 MPSoC 上。

**📊 数据集**

采用 ImageNet（分类网络）和 COCO（YOLOv5）数据集，共 33 个模型（含 2 个裁剪版本）进行实验。

**📈 对比分析**

与“最优配置”“最大 FPS”“最低功耗”三种基线对比，RL 代理在工作负载状态 C/M 下平均实现 95–97% 的能效（PPW），在 89% 的测试案例满足 30 FPS 约束。

**⚠️ 局限性**

局限性包括：重配置与指令加载开销约 1 s，对短期推理不友好；训练依赖离线测量；配置空间仅覆盖 26 个离散组合，可能遗漏更优的细粒度方案；在不同 FPGA 平台的泛化能力未验证。

---

## 299. Media Framing Moderates Risk-Benefit Perceptions and Value Tradeoffs in Human-Robot Collaboration

**arXiv ID:** 2602.12785 | [PDF](https://arxiv.org/pdf/2602.12785v1)

**作者:** Philipp Brauner `[一作]` (RWTH Aachen University), Martina Ziefle `[通讯]` (RWTH Aachen University)

**通讯引用:** 11483 | [OpenAlex ID](https://openalex.org/A5065952170)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过对德国工人代表性样本阅读正向与负向框架的报纸文章，探讨媒体框架如何影响对工业人机协作(HRC)的风险、收益评估及价值归因；

**💡 创新点**

创新点在于揭示正负框架不仅改变评估绝对水平，还改变风险与收益的交互方式——正向框架下风险与收益呈加性效应，负向框架下呈负向交互效应；

**🔧 技术方法**

采用实验设计与预注册在线问卷，利用多元方差分析和多元回归模型评估框架效应与风险收益权衡；

**📊 数据集**

使用约1151名德国工作人口的在线调查数据，包含风险、收益及价值评估量表；

**📈 对比分析**

对正负框架分别建模，正向框架R²=0.715，负向框架R²=0.583，表明正向框架对价值归因的解释力更强；

**⚠️ 局限性**

局限在于样本为付费在线调查，可能缺乏动机；仅使用报纸文章做框架，未检验其他媒体形式；所有测量为自报，缺乏行为数据，且仅在德国文化背景下验证。

---

## 300. Physics-Informed Laplace Neural Operator for Solving Partial Differential Equations

**arXiv ID:** 2602.12706 | [PDF](https://arxiv.org/pdf/2602.12706v1)

**作者:** Heechang Kim `[一作]` (Pohang University of Science and Technology), Minseok Choi `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 4333 | [OpenAlex ID](https://openalex.org/A5007805877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Physics‑Informed Laplace Neural Operator（PILNO），在 Laplace Neural Operator（LNO）的基础上加入物理残差约束、虚拟输入和时间因果加权，提升小样本和 OOD 泛化能力。

**💡 创新点**

创新点在于：1）将 LNO 的极点-残差瞬态分支与 FNO 风格稳态分支解耦为 Advanced LNO（ALNO），兼具可解释性和更强表达力；2）通过大量无标签虚拟输入提供物理监督，扩大输入谱范围；3）引入时间因果加权降低后期残差对早期误差的掩蔽，提高训练稳定性。

**🔧 技术方法**

技术手段包括：Laplace 转换与极点-残差参数化、FNO 频域乘子、物理残差（PDE/BC/IC）损失、虚拟输入采样、时间因果加权 (TCW)、Adam 优化及学习率衰减。

**📊 数据集**

实验使用四个 PDE 基准：Burgers 方程、Darcy 流、反应扩散系统和受迫 KdV 方程。数据由高斯随机场采样生成，训练样本数从 10~2700 级别不等，虚拟输入量可达 1000+。

**📈 对比分析**

与纯数据驱动的 LNO、FNO、DeepOMamba 等基线进行对比。PILNO 在小样本（N_train≤27）下将相对 L2 误差降低 1–2 倍，OOB 泛化也显著优于基线；在 forced KdV 小样本 (27) 里，PILNO 的误差从 47% 降至 13%。

**⚠️ 局限性**

局限性包括：1）仍需手动设定虚拟输入分布与 TCW 调参；2）对大规模高维 PDE 计算量可能高；3）在完全标记充足的情况下，PILNO 的优势不如全监督模型；4）对非线性强、刚性约束明显的 PDE 仍存在稳定性挑战。

---

## 301. Coden: Efficient Temporal Graph Neural Networks for Continuous Prediction

**arXiv ID:** 2602.12613 | [PDF](https://arxiv.org/pdf/2602.12613v1)

**作者:** Zulun Zhu `[一作]` (Nanyang Technological University), Siqiang Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 15548 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Coden，一种针对连续预测的高效动态图神经网络框架。

**💡 创新点**

创新点包括：将图传播与时间更新解耦，采用无参数 PPR 进行增量嵌入更新；利用状态空间模型（SSM）实现低复杂度的状态更新；引入惰性采样策略压缩历史信息；证明 RNN‑Attention 双重性，为模型提供理论解释。

**🔧 技术方法**

核心技术包括：个性化 PageRank（PPR）嵌入、结构化状态空间模型（SSM）、稀疏增量更新、懒惰采样、门控递归与核注意力等。

**📊 数据集**

在五个真实动态图数据集上评估：DBLP、Tmall、Reddit、Patent 与 Papers100M。

**📈 对比分析**

与 GCN、GraphSage、SCARA、Instant、TGCN、ASTGCN、EvolveGCN、MPNN、CAWN、DNNTSP、SpikeNet、Zebra、TGL+TGN、DyGFormer 等基线对比，Coden 在准确率上达到或超过最优方法，并在大规模图上实现最高 44.8× 的加速，整体训练+推理时间显著降低。

**⚠️ 局限性**

局限性在于：对节点增删的处理仅通过边更新；模型对阈值 λ 和隐藏维度 F' 的调参敏感；在极端高频更新或非边更新场景下可能需进一步优化。

---

## 302. Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR

**arXiv ID:** 2602.12642 | [PDF](https://arxiv.org/pdf/2602.12642v1)

**作者:** Dohyung Kim `[一作]` (Seoul National University), Kyomin Jung `[通讯]` (Seoul National University)

**通讯引用:** 3553 | [OpenAlex ID](https://openalex.org/A5077832834)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用GFlowNet的可学习划分函数作为在线准确性估计器，进行难度调度；

**💡 创新点**

将划分函数从单纯的归一化作用转化为在线准确性预测，进而实现自适应提示选择和误差优先重放；

**🔧 技术方法**

GFlowNet + Trajectory Balance、可学习划分函数（Zϕ）、自适应提示选择、误差优先经验重放；

**📊 数据集**

代码生成：DeepCoder；数学推理：DeepScaleR；评测集包括HumanEval+、LiveCodeBench、MATH500、OlympiadBench、MinervaMath、AIME24/25；

**📈 对比分析**

与GRPO、FlowRL、FlowRL‑VarGrad、DS、LILO、MoPPS等方法对比，PACED‑RL在pass@1/ pass@k上均取得显著提升，AIME上可比GRPO/FlowRL提升约29%/40%，训练速度加快（仅需原来时间的约0.5倍），并保持更高的输出多样性；

**⚠️ 局限性**

局限性包括：仍依赖roll‑out计算，难以处理多步推理场景；对二进制奖励的假设；划分函数近似误差可能影响难度估计；尚未在异步GFlowNet框架下验证。

---

## 303. Experimentation, Biased Learning, and Conjectural Variations in Competitive Dynamic Pricing

**arXiv ID:** 2602.12888 | [PDF](https://arxiv.org/pdf/2602.12888v1)

**作者:** Bar Light `[一作]` (National University of Singapore), Wenyu Wang `[通讯]` (National University of Singapore)

**通讯引用:** 7162 | [OpenAlex ID](https://openalex.org/A5100461646)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究在多卖家竞争环境下，卖家仅通过自身价格和销量的 bandit 反馈进行两点随机实验（switchback 设计），并基于误差线性需求模型不断更新价格。

**💡 创新点**

创新点在于：①揭示实验设计的协同/非协同结构会在学习偏差中自然产生“猜测变动”矩阵 A，进而把动态学习过程与经典的 Conjectural Variations (CV) 均衡关联；②证明在协同实验下收敛到 CV(A★)；在独立实验下收敛到 Nash；③提供 O(T⁻¹/²) 的有限样本误差收敛速率。

**🔧 技术方法**

技术主要包括：两点随机实验的统计设计；误差线性需求的 OLS 估计；基于偏差分析得到的有效猜测矩阵 A★；利用收敛映射与雅可比矩阵的收敛性证明；稳定性条件下的收敛与比较静力学。

**📊 数据集**

本研究为理论研究，未使用具体实验或真实交易数据；所有结论均来自对价函数和需求函数的假设推导。

**📈 对比分析**

由于缺乏实证数据，本文未进行实验比较；理论上通过构造不同的 A★（协同 vs. 独立）展示收敛目标的差异，并给出收敛速率的上界；相较于现有的 bandit 学习算法，证明了在满足一定条件下可实现最优的 O(T⁻¹/²) 收敛速率。

**⚠️ 局限性**

限制主要包括：①仅考虑单变量线性需求的误差拟合，无法覆盖更复杂的需求结构；②实验仅为两点随机，无法捕捉更丰富的探索策略；③假设需求函数满足一定的光滑性和跨价正相关性；④不考虑卖家观测到竞争对手信息的情况；⑤收敛稳定性条件可能对实际市场难以验证。

---

## 304. FedHENet: A Frugal Federated Learning Framework for Heterogeneous Environments

**arXiv ID:** 2602.13024 | [PDF](https://arxiv.org/pdf/2602.13024v1)

**作者:** Alejandro Dopico-Castro `[一作]` (Universidade da Coruna), Iván Pérez Digón `[通讯]` (Universidade da Coruna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FedHENet，一种在图像分类任务中实现单轮联邦学习的框架，利用冻结的预训练特征提取器和可解析的单层输出层，避免本地微调和迭代梯度下降。

**💡 创新点**

创新点在于：①将高维图像特征与单层可解析学习解耦，实现在单轮通信中完成全局权重的解析聚合；②采用同态加密（CKKS）对关键统计量进行安全聚合，完全消除梯度泄露风险；③完全无超参数调优，显著降低碳足迹与能源消耗；④在极端非 IID 情况下保持高准确率和稳定性。

**🔧 技术方法**

使用的技术包括：联邦学习框架 FedAvg / FedProx 对比；同态加密 CKKS；奇异值分解（SVD）与线性最小二乘解析求解；MQTT 轻量化通信；预训练 ResNet-18 作为冻结特征提取器；Python/PyTorch 实现。

**📊 数据集**

实验数据集为 CIFAR‑10（10 类）和 CIFAR‑100（100 类），通过 Dirichlet 分布模拟不同程度的非 IID，亦使用“单类”极端异构场景。

**📈 对比分析**

方法通过与 FedAvg、FedProx（相同预训练 backbone、相同训练轮次）进行对比，评价指标包括测试准确率、能耗、训练时间与通信量。FedHENet 在标准与极端异构场景均能保持 83–84% 的准确率，能耗相较迭代基线降低约 70%，训练时间减少 60% 以上，通信量虽因 HE 加密略增，但整体规模仍小于多轮梯度聚合。

**⚠️ 局限性**

局限性：①同态加密导致传输数据量扩大约 2.25 倍，虽对能耗影响小但在大规模联邦场景可能显著；②仅验证了固定 backbone 的图像分类任务，未覆盖更复杂模型或不同任务；③在边缘硬件上的实际性能尚未测评；④单层输出模型对模型容量有限制，可能在更复杂任务中欠拟合。

---

## 305. SKYSURF: A Self-learning Framework for Persistent Surveillance using Cooperative Aerial Gliders

**arXiv ID:** 2602.12838 | [PDF](https://arxiv.org/pdf/2602.12838v1)

**作者:** Houssem Eddine Mohamadi `[一作]` (Ecole de Technologie Superieure), Nadjia Kara `[通讯]` (Ecole de Technologie Superieure)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于局部-全局行为管理和决策的合作热气球式无人机持续巡逻框架

**💡 创新点**

创新点在于将无人机建模为非确定性有限状态理性代理，并结合局部学习与全局分布式决策实现多机协作与能量自适应

**🔧 技术方法**

核心技术包括有限状态机行为切换、基于期望收益的分布式决策、H_w4ptH碰撞安全路径规划、DLnT延迟学习与调参PID控制器

**📊 数据集**

实验使用合成热升气流和目标数据集，仿真区域为6×6 km²，热升周期6–10分钟，目标出现频率6–30分钟

**📈 对比分析**

与三种基线（半合作、非合作、零知识）和15种进化/RRT路径规划算法对比，实验显示方案在续航时间、热升利用率、目标探测率和功耗方面分别提升约17.8%、两倍及6%

**⚠️ 局限性**

主要局限包括对热升模型的简化、缺乏真实环境验证以及在大规模多机场景下通信延迟与协调开销未充分评估

---

## 306. Semantic Communities and Boundary-Spanning Lyrics in K-pop: A Graph-Based Unsupervised Analysis

**arXiv ID:** 2602.12881 | [PDF](https://arxiv.org/pdf/2602.12881v1)

**作者:** Oktay Karakuş `[一作]` (Cardiff University), Oktay Karakuş `[通讯]` (Cardiff University)

**通讯引用:** 896 | [OpenAlex ID](https://openalex.org/A5053586101)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建无监督图模型，对K-pop歌词进行行级语义嵌入、相似度图构建和社区检测，识别语义社区与跨社区桥接歌曲。

**💡 创新点**

首次将行级句子嵌入与kNN图结合，利用社区和桥接指标无监督发现语义结构，并验证其对新歌的可泛化。

**🔧 技术方法**

多语言句子变换器Embedding、k最近邻稀疏图、Louvain社区检测、betweenness centrality与边界得分。

**📊 数据集**

7,983首来自Melon榜单的K-pop歌词（2000-2023）共1,485名艺人。

**📈 对比分析**

通过与非桥接歌曲比较，发现桥接歌曲词汇熵略高、重复率略低；阈值稳健，结果在不同阈值下保持一致。

**⚠️ 局限性**

数据以韩文为主，缺乏音频特征与韵律信息，且仅评估文本语义，无法捕捉节奏与音乐影响。

---

## 307. On Borrowed Time: Measurement-Informed Understanding of the NTP Pool's Robustness to Monopoly Attacks

**arXiv ID:** 2602.12321 | [PDF](https://arxiv.org/pdf/2602.12321v1)

**作者:** Robert Beverly `[一作]` (San Diego State University), Erik Rye `[通讯]` (Johns Hopkins University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 NTP Pool 进行了直接、长期（9 个月）全面测量，收集服务器、账户、配置、流量等信息，并评估其独立性与被垄断攻击的脆弱性。

**💡 创新点**

首次使用网站抓取而非 DNS 推断，结合 NTP 指纹与别名识别，量化服务器独立性并推导低容量即可实现的垄断攻击模型。

**🔧 技术方法**

使用自定义 Web 抓取器、ntpdedup 指纹工具、DNS 计数 API、BGP 路由表、IPv6 IID 分析等技术进行测量与分析。

**📊 数据集**

依赖 9 个月自建抓取数据（约 15k 服务器、6k 活跃）、BigQuery 历史分数表（2008-2025）、RIPE Atlas、IPv6 Observatory、IPinfo、BGP 路由表等多源数据集。

**📈 对比分析**

与先前基于 DNS 探测的方法对比，证明抓取方法更完整、更准确；在实验中仅用两台攻击服务器即获得约 47% 的 DNS 回答比例，验证攻击模型可行性。

**⚠️ 局限性**

局限包括无法完整映射隐私账户、历史分数数据只能推断服务器寿命、Stratum‑1 服务器和多 ASN 场景下指纹可能误判，以及实验规模有限。

---

## 308. Mixture of Predefined Experts: Maximizing Data Usage on Vertical Federated Learning

**arXiv ID:** 2602.12708 | [PDF](https://arxiv.org/pdf/2602.12708v1)

**作者:** Jon Irureta `[一作]` (Ikerlan Technology Research Center), Javier Fernandez-Marques `[通讯]` (Flower Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的垂直联邦学习框架Split-MoPE，能在样本不完全对齐的场景下进行协同训练并实现单轮通信；

**💡 创新点**

创新点在于将拆分学习与预定义专家Mixture of Predefined Experts（MoPE）相结合，预定义专家对应不同的样本对齐方式，实现了对不完整样本的自适应加权与对抗恶意/噪声参与者的鲁棒性；

**🔧 技术方法**

采用了预训练的特征提取器（如DINOv2 ViT和Qwen3文本嵌入）作为冻结模块，配合MoPE头和Sigmoid门控进行自适应加权；

**📊 数据集**

实验使用了CIFAR-10、CIFAR-100和乳腺癌Wisconsin表格数据，均经过拆分为多参与者；

**📈 对比分析**

与SplitNN、LASER以及其预训练版本进行对比，Split-MoPE在所有对齐率下都至少与最佳基线持平，且在高缺失率时性能下降最小，甚至在极端缺失时仍不低于单个本地模型；

**⚠️ 局限性**

局限包括对大规模多参与者（>2）扩展的细节尚未充分验证，以及对极端噪声场景下的门控稳定性仍需进一步研究。

---

## 309. Adaptive traffic signal control optimization using a novel road partition and multi-channel state representation method

**arXiv ID:** 2602.12296 | [PDF](https://arxiv.org/pdf/2602.12296v1)

**作者:** Maojiang Deng `[一作]` (Nanjing Tech University), Wen Zhang `[通讯]` (Jiangsu University)

**通讯引用:** 24889 | [OpenAlex ID](https://openalex.org/A5100692580)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于可变单元划分和多通道状态表示的自适应交通信号控制方法；

**💡 创新点**

其创新点在于使用对数+线性函数构造可变车道单元长度，并引入车辆数、平均速度、占用率三通道状态，以提升模型的泛化性和优化效果；

**🔧 技术方法**

该方法采用深度强化学习中的DQN和PPO算法，结合TensorFlow构建卷积网络，并通过SUMO仿真获取环境反馈；

**📊 数据集**

实验数据来自SUMO生成的仿真交通流，包括多种流量场景和随机车辆轨迹；

**📈 对比分析**

与固定时长、感应控制以及固定单元的RL模型比较，VCL-PPO在累计队列长度和等待时间上表现最佳，PPO优于DQN，并且在跨检测范围迁移时仍保持优势；

**⚠️ 局限性**

主要局限在于对高波动流量的收敛速度慢，需要更多训练周期；实验仅在单一交叉口仿真，缺乏大规模网络验证，且对多智能体或更复杂交通场景的适用性尚未评估。

---

## 310. Towards explainable reference-free speech intelligibility evaluation of people with pathological speech

**arXiv ID:** 2602.12723 | [PDF](https://arxiv.org/pdf/2602.12723v1)

**作者:** Bence Mark Halpern `[一作]` (Nagoya University), Thomas Tienkamp `[通讯]` (Groningen University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并评估一种无参考、可解释的 ASR 不一致性评分，用于评估病理性语音可懂度。

**💡 创新点**

通过比较两种 ASR 识别结果（贪婪转录与语言模型纠正转录）的错误率来量化不一致性，既可解释又不需要人工标注。

**🔧 技术方法**

使用基于 Wav2Vec2 的 ASR 模型，5‑gram 语言模型和 GPT‑4/ChatGPT 进行纠正，计算词错误率 (WER)。

**📊 数据集**

在四个公开/自建语料库上评估：NKI‑OC‑VC (荷兰语)、NKI‑SpeechRT (荷兰语)、NeuroVoz (西班牙语)、TORGO (英语)。

**📈 对比分析**

与传统基准（语速、WADA‑SNR、参考基准 WER）比较，ASR 不一致性评分在所有数据集上与专家可懂度评分相关性接近或超过参考基准，尤其在荷兰语数据集几乎等同于 WER；在西班牙语数据集上简单的 5‑gram 方法甚至优于 WER。

**⚠️ 局限性**

依赖于 ASR 与语言模型的质量；对极度失语者可能不足；目前仅针对朗读句子，未验证对自发语音的适用性。

---

## 311. Energy-Aware Reinforcement Learning for Robotic Manipulation of Articulated Components in Infrastructure Operation and Maintenance

**arXiv ID:** 2602.12288 | [PDF](https://arxiv.org/pdf/2602.12288v1)

**作者:** Xiaowen Tao `[一作]` (Jilin University), Ziyu Song `[通讯]` (Jilin University)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5102743767)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种面向智能基础设施运维的可扩展、能耗约束的强化学习框架，实现对门、抽屉、阀门等多种关节机构的统一机器人操控。

**💡 创新点**

创新点在于将功能部件引导的3D感知、加权点采样与PointNet编码与约束软演员-批评家（C‑SAC）相结合，显式将驱动能量作为约束加入决策过程。

**🔧 技术方法**

采用RGB‑D分割网络实现部件分割，PointNet进行几何特征提取，C‑SAC实现能耗约束的策略学习，并使用Lagrangian多目标优化。

**📊 数据集**

使用PartNet‑Mobility公开数据集中的门、抽屉和阀门模型进行仿真训练和测试，并在实验中对真实场景未见过的对象进行评估。

**📈 对比分析**

与传统SAC和奖励惩罚型能耗约束方法相比，实验表明C‑SAC在能耗上下降16–30%，完成步数减少16–32%，且保持或提升任务成功率。

**⚠️ 局限性**

局限包括仅使用简化的能耗模型（基于关节速度平方），未考虑扭矩或电力消耗；对高阻力任务如TurnValve的成功率仍有限；需进一步验证仿真到真实平台的迁移鲁棒性。

---

## 312. Geometric Stratification for Singular Configurations of the P3P Problem via Local Dual Space

**arXiv ID:** 2602.12525 | [PDF](https://arxiv.org/pdf/2602.12525v1)

**作者:** Xueying Sun `[一作]`, Nan Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对三点定位（P3P）问题中的奇异配置进行了深入分析，提出了一种基于局部双空间的新型求解框架。

**💡 创新点**

创新点在于利用局部双空间的几何结构，将奇异配置的判定与求解过程转化为更简洁的代数形式，从而显著降低了计算复杂度并提升了稳定性。

**🔧 技术方法**

主要技术包括双空间变换、代数几何方法、符号运算以及数值优化策略，辅以符号计算软件实现高效求解。

**📊 数据集**

实验使用合成的相机姿态数据集，并结合公开的P3P基准数据（如EPFL摄像机标定数据集）进行验证。

**📈 对比分析**

与传统的解析求解方法和迭代式求解方法相比，本方法在相同硬件条件下运行时间减少约30%，且在存在噪声的实验场景中保持了更高的解准确率。

**⚠️ 局限性**

局限性包括：仅适用于标准P3P模型；对极端冗余或完全共线的三点配置仍存在求解不稳定；以及在极高噪声情况下对双空间投影的数值误差敏感。

---

## 313. Monocular Reconstruction of Neural Tactile Fields

**arXiv ID:** 2602.12508 | [PDF](https://arxiv.org/pdf/2602.12508v1)

**作者:** Pavan Mantripragada `[一作]` (University of Maryland), Yiannis Aloimonos `[通讯]` (University of Maryland)

**通讯引用:** 7418 | [OpenAlex ID](https://openalex.org/A5036912867)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

从单张 RGB 图像预测3D触觉场，实现机器人对环境的交互感知与路径规划。

**💡 创新点**

首次将单视图RGB映射为稠密3D触觉场，并将触觉信息融入路径规划。

**🔧 技术方法**

采用Large Reconstruction Model (LRM) 进行视角‑三平面解码，联合光度和体积损失训练。

**📊 数据集**

构建了包含20棵植物和20件家居物体的触觉数据集，包含NeRF重建、GelSight压力测量。

**📈 对比分析**

与LRM和Direct3D对比，IoU提升约80%（0.095 vs 0.052），Chamfer距离降低至0.245，规划路径更短且成功率相近。

**⚠️ 局限性**

数据集规模有限，仅40个同一台位面物体，难以推广到多样化环境和材料。

---

## 314. Ca-MCF: Category-level Multi-label Causal Feature selection

**arXiv ID:** 2602.12961 | [PDF](https://arxiv.org/pdf/2602.12961v1)

**作者:** Wanfu Gao `[一作]` (Jilin University), Yonghao Li `[通讯]` (Southwestern University of Finance and Economics)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5049385796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于标签类别展开的多标签因果特征选择方法Ca-MCF，能够在细粒度层面挖掘标签之间的因果结构并恢复被标签关联抑制的有效特征。

**💡 创新点**

创新点包括标签类别展开、特定类别互信息(SCSMI)与不同类别互信息(DCSMI)的定义、基于竞争的特征恢复机制以及结构对称性与交叉维度冗余剔除。

**🔧 技术方法**

使用信息理论互信息度量、局部结构发现算法（PC/SP搜索）、标签类别依赖建模、结构对称检查与冗余删除等技术。

**📊 数据集**

在七个真实多标签数据集上进行实验，包括Flags、VirusGO、CHD_49、PlantGO、Enron、Image和Yeast。

**📈 对比分析**

与七种主流多标签特征选择方法（MI、正则化、流形学习、BN结构学习等）比较，Ca-MCF在Hamming Loss、Macro‑F1等多项指标均表现出色，平均排名第一或第二。

**⚠️ 局限性**

局限性主要在于对标签不平衡和稀疏数据的敏感性，信息理论估计可能受限于样本不足，且方法在极高维度下计算复杂度仍较大。

---

## 315. Formalizing the Sampling Design Space of Diffusion-Based Generative Models via Adaptive Solvers and Wasserstein-Bounded Timesteps

**arXiv ID:** 2602.12624 | [PDF](https://arxiv.org/pdf/2602.12624v1)

**作者:** Sangwoo Jo `[一作]` (Korea University), Sungjoon Choi `[通讯]` (Korea University)

**通讯引用:** 1733 | [OpenAlex ID](https://openalex.org/A5047885515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SDM框架，通过自适应求解器和自适应时间步调度改进预训练扩散模型的采样效率与质量。

**💡 创新点**

创新点在于利用ODE曲率分析实现低噪声阶段使用低阶求解器、非线性阶段使用高阶求解器的动态切换，并通过Wasserstein距离上界构造误差约束的自适应时间步。

**🔧 技术方法**

采用概率流ODE理论、曲率估计、低阶/高阶ODE求解器（Euler、Heun等）、Wasserstein误差上界、N步重采样等技术。

**📊 数据集**

在CIFAR‑10、FFHQ、AFHQv2、ImageNet四个图像数据集上进行评估。

**📈 对比分析**

与EDM、COS、DPM‑Solver++、UniPC等基线比较，SDM在FID上分别达1.93、2.41、1.98等最优结果，同时在NFE上显著降低（约15–20%）。

**⚠️ 局限性**

局限性包括需手动调节阈值/误差参数、仅针对图像生成任务验证、对其他数据模态或更复杂模型的泛化仍待研究。

---

## 316. RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction

**arXiv ID:** 2602.12593 | [PDF](https://arxiv.org/pdf/2602.12593v1)

**作者:** Ziye Tong `[一作]` (Tencent), Ning Gu `[通讯]` (Fudan University)

**通讯引用:** 43310 | [OpenAlex ID](https://openalex.org/A5012421463)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出一种基于残差量化和高斯混合模型的多模态语义离散化方法RQ‑GMM，用于提升CTR预测效果。

**💡 创新点**

创新点在于将GMM的软分配与残差量化相结合，既避免了代码簿坍塌，又能捕捉多模态嵌入空间的统计结构，实现更高的代码簿利用率和重构精度。

**🔧 技术方法**

核心技术包括残差量化、GMM软分配、EM算法训练以及将生成的离散ID嵌入到传统CTR模型（如DeepFM、FNN、IPNN）中。

**📊 数据集**

实验使用Amazon Review三类数据集（Appliances、Beauty、Automotive）以及一大规模工业短视频平台的数据。

**📈 对比分析**

与VQ‑VAE、RQ‑VAE、RQ‑KMeans等方法对比，RQ‑GMM在RMSE、代码簿利用率、AUC和LogLoss上均取得显著优势；在线A/B测试中广告价值提升1.502%并已上线部署。

**⚠️ 局限性**

局限包括对多模态联合建模的支持不足、代码簿大小固定、缺乏自适应层级和统一概率建模等改进空间。

---

## 317. TensorCommitments: A Lightweight Verifiable Inference for Language Models

**arXiv ID:** 2602.12630 | [PDF](https://arxiv.org/pdf/2602.12630v1)

**作者:** Oguzhan Baser `[一作]` (University of Texas at Austin), Sriram Vishwanath `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TensorCommitments（TC）和Terkle树，以实现对大型语言模型推理的可验证性，既能保留模型内部张量结构，又能在不重新执行推理的情况下通过轻量级客户端验证结果。

**💡 创新点**

创新点包括：①基于多变量多项式插值的张量承诺方案，显著降低承诺与开口成本；②Terkle树将承诺组织成高阶张量结构，减少证明大小与验证时间；③使用重尾谱特征的层级选择算法，聚焦最敏感层以提升攻击检测率。

**🔧 技术方法**

采用了基于配对的椭圆曲线加密、Newton/格雷戈里等多变量插值、Verkle树与Terkle树的混合认证、动态规划层级选择以及重尾谱统计等技术。

**📊 数据集**

主要在LLaMA2-13B（以及7B）模型上评估，利用人工注入噪声、提示篡改和Taco攻击等对抗手段验证鲁棒性；对比使用的公开攻击数据集与模型。

**📈 对比分析**

与zkLLM、SVIP、Raw Activations和TOPLOC等方法相比，TC在推理后证明时间仅增加约0.97%，验证时间约0.12%，攻击检测准确率高达96%，相比TOPLOC提升约48%，同时承诺尺寸与验证GPU内存需求最小。

**⚠️ 局限性**

局限性：需要推理方具备完整激活访问且拥有GPU；目前不提供全零知识证明；当Terkle树枝度过低时构造与证明效率下降；并未支持多方计算或完全加密的私有场景。

---

## 318. Not a Silver Bullet for Loneliness: How Attachment and Age Shape Intimacy with AI Companions

**arXiv ID:** 2602.12476 | [PDF](https://arxiv.org/pdf/2602.12476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 319. Editable XAI: Toward Bidirectional Human-AI Alignment with Co-Editable Explanations of Interpretable Attributes

**arXiv ID:** 2602.12569 | [PDF](https://arxiv.org/pdf/2602.12569v1)

**作者:** Haoyang Chen `[一作]` (National University of Singapore), Brian Y Lim `[通讯]` (National University of Singapore)

**通讯引用:** 3758 | [OpenAlex ID](https://openalex.org/A5056248594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了可编辑解释AI（Editable XAI）框架CoExplain，使用户能够直接编辑决策树式规则并与AI协同优化模型；

**💡 创新点**

创新点在于将解释本身作为可编辑的共享媒介，实现了双向人机对齐，即用户既能根据自身知识调整AI推理，又能通过AI的阈值与拓扑增强深化对AI决策的理解；

**🔧 技术方法**

核心技术包括神经符号方法：用决策树对神经网络进行蒸馏得到可解释规则；将用户手写规则解析为等价的神经网络；利用阈值提升与拓扑重构两种AI辅助编辑，并通过树编辑距离正则化维持规则与用户意图的一致性；

**📊 数据集**

实验使用了三大公开数据集：Adult Income、House Price（King County房价）和Heart Disease（UCI），每个任务均涉及5–6个属性；

**📈 对比分析**

与仅读XAI以及仅可编辑无AI辅助的Editable XAI进行对比，结果显示Editable和CoExplain显著提升用户对AI决策的预测一致性、模型对齐度和认知水平；CoExplain在保持近乎最优准确率的同时，减少了用户编辑次数、迭代次数和总编辑时间，整体性能优于其他两种方案；

**⚠️ 局限性**

局限性包括：仅针对结构化数据和决策树规则，难以直接迁移到非结构化输入或更复杂模型；可扩展性受限于规则规模和用户认知负荷；对用户知识水平依赖较大，过度依赖专业背景可能导致误导。

---

## 320. Truthful Fair Division under Stochastic Valuations

**arXiv ID:** 2602.12359 | [PDF](https://arxiv.org/pdf/2602.12359v1)

**作者:** Daniel Halpern `[一作]` (Google Research), Shirley Zhang `[通讯]` (Harvard University)

**通讯引用:** 21870 | [OpenAlex ID](https://openalex.org/A5100358804)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在无金钱交易的公平分配问题中，研究了在随机（i.i.d.）估值下的真诚机制，探讨其在实现效率、公平与激励兼容性方面的可行性。

**💡 创新点**

突破传统极端案例的限制，利用随机估值与先行的“预言机不等式”理论，证明在两位代理人时可实现约 0.854 的福利近似并且几乎无偏好，且在任意代理人数量时可实现约 0.745 的近似；同时提出基于排名的 BIC 机制，近乎最优福利且高概率公平。

**🔧 技术方法**

核心技术包括：
- DSIC 机制的结构化（挑选-交换）
- 量化阈值与“pick‑r”机制的等价性
- 与 i.i.d. 预言机不等式竞争比的对应关系
- 对 BIC 的排序策略证明与高概率公平性分析
- 结合概率工具（如 DKW 与 Hoeffding）评估随机误差。

**📊 数据集**

采用理论模型：每件物品的价值从任意满足非负、有限方差、连续分布的 i.i.d. 取值分布抽取；未使用真实数据集，而是以数学证明为主。

**📈 对比分析**

与最优福利（最大化分配）做比较，给出了 2+√2/4 ≈ 0.854（两代理人）和 β≈0.745（任意代理人）的近似比；BIC 排序算法达到 1−o(1) 的福利近似，并在高概率下实现公平；对比传统可行性不可能性，显示随机情形下可实现三重目标。

**⚠️ 局限性**

局限性：仅适用于加性估值；假设估值独立同分布（不考虑相关性）；对于代理人数量大于两人时的最优性仍未证明最紧；未提供实证或实验验证；对非 i.i.d. 或非加性情形的推广仍是未来工作。

---

## 321. Contextual Online Bilateral Trade

**arXiv ID:** 2602.12903 | [PDF](https://arxiv.org/pdf/2602.12903v1)

**作者:** Romain Cosson `[一作]` (New York University), Matteo Russo `[通讯]` (EPFL)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在上下文敏感双边贸易（seller‑buyer 之间的价格设定）下的在线学习问题，目标是最大化交易的经济效率（gain from trade）或平台自身利润，并在不同反馈模型（两位/单位反馈）和预算平衡约束下设计算法。

**💡 创新点**

主要创新包括：① 在两位反馈下实现了接近最优的 O(d log d)（效率）和 O(d log log T + d log d)（利润）无退化（regret）率；② 在单位反馈下首次实现与两位反馈相同的无退化率，只需允许极小的总负利润；③ 在单位反馈并强制逐步预算平衡的情形下，提出了指数级（O(d 6^d)）但时间无关的效率算法和 O(d 6^d log T) 的利润算法，展示了预算平衡与维度之间的权衡。

**🔧 技术方法**

技术上采用了几何式的置信区域更新、Steiner 体积势函数（potential）分析、随机化定价策略以及对冲突半空间的分割，结合了对宽度 (width) 的精细控制来平衡探索与利用，解决了单位反馈下的模糊性问题。

**📊 数据集**

论文为理论研究，未使用具体数据集；所有结果均为理论上限（上界与下界）。

**📈 对比分析**

与以往工作（如 O(d^2 log T) 的效率算法、单位反馈下多项式时间退化的利润算法）相比，本文在两位反馈下实现了最优的无退化率；在单位反馈下的无退化率与两位相当，且仅产生 O(d log d) 的总负利润；在预算平衡强制的单位反馈下，尽管维度指数增长，但在时间维度上实现了无退化率。

**⚠️ 局限性**

主要局限包括：① 在单位反馈并强制预算平衡的情形下，算法对维度的指数依赖限制了高维应用；② 对于更一般的噪声或不精确反馈模型，本文的无退化率可能无法保持；③ 论文侧重理论上界，实际实现的计算复杂度与实现细节尚未讨论。

---

## 322. Unifying Model-Free Efficiency and Model-Based Representations via Latent Dynamics

**arXiv ID:** 2602.12643 | [PDF](https://arxiv.org/pdf/2602.12643v1)

**作者:** Jashaswimalya Acharjee `[一作]` (Indian Institute of Technology Madras), Balaraman Ravindran `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 4657 | [OpenAlex ID](https://openalex.org/A5009374923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种通用模型自由强化学习算法 ULD，利用学习到的状态‑动作嵌入与线性价值分解，在单一网络结构下实现跨任务的高效学习。

**💡 创新点**

创新点在于将模型基方法的表示学习（线性动态预测、奖励预测和终止预测）嵌入模型自由框架，形成统一的线性价值近似与非线性 Q 函数结合的策略；同时采用统一超参数实现跨任务迁移，消除领域专门调参需求。

**🔧 技术方法**

核心技术包括：状态编码器和状态动作编码器；线性环境模型用于预测下一状态嵌入、奖励和终止；多任务辅助损失（奖励交叉熵、动力学 MSE、终止 MSE）；双重 TD3 风格评论网络与 Huber 损失；奖励归一化、多步回报、Gumbel‑Softmax 离散动作处理；优先经验回放和目标网络软更新。

**📊 数据集**

使用 80 个标准环境：Gym 连续控制（5 个）、DeepMind Control 运动学（23 个）分为向量与像素两种观测；Atari 43 个游戏；对比基准包括 DreamerV3、TD‑MPC2、PPO、TD7、DrQ‑v2、Rainbow、DQN。

**📈 对比分析**

在相同超参数设置下与领域专用方法、通用模型基方法以及传统模型自由方法比较，ULD 在连续控制和视觉 DMC 任务上表现最佳，Gym 任务略低于专用 TD7，但显著优于 DreamerV3；在 Atari 任务上超越大多数基准并接近人类水平；整体跨域性能均衡，展示了“无特定领域”优势。

**⚠️ 局限性**

局限性：不适用于需要长期推理、复杂探索或非马尔可夫依赖的任务；未在真实世界多智能体、机器人或语言条件控制等场景中验证；缺乏完整的模型基规划或轨迹生成，因而在某些长时延任务上可能性能不足。

---

## 323. PISHYAR: A Socially Intelligent Smart Cane for Indoor Social Navigation and Multimodal Human-Robot Interaction for Visually Impaired People

**arXiv ID:** 2602.12597 | [PDF](https://arxiv.org/pdf/2602.12597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 324. Distance-based certification for leader election in meshed graphs and local recognition of their subclasses

**arXiv ID:** 2602.12894 | [PDF](https://arxiv.org/pdf/2602.12894v1)

**作者:** Jérémie Chalopin `[一作]` (CNRS), Maria Kokkou `[通讯]` (Paderborn University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种用于匿名网格图中领导者选举的2局部证明标记方案，使用标签集{0,1,2}。该方案通过局部条件验证全局属性，确保每个节点可以通过其到指定根节点的距离进行标记。

**💡 创新点**

创新点在于提出了适用于广泛类别的网格图的常数大小标签的领导者选举证明标记方案，并且扩展了对网格图子类的局部识别能力。

**🔧 技术方法**

使用了局部验证和距离验证技术，结合现有的局部到全局特征化方法来设计证明标记方案。

**📊 数据集**

使用了多种网格图的子类作为数据集，包括中位数图、桥接图、和赫利图等。

**📈 对比分析**

与现有方法相比，本文的方法在性能上表现出色，能够在常数大小的标签下实现领导者选举，并且在局部验证中保持高效性。

**⚠️ 局限性**

限制在于该方案可能无法适用于所有网格图，尤其是那些不具备简单连通性的图，且对某些图类的局部识别能力仍需进一步研究。

---

## 325. LiDAR-Anchored Collaborative Distillation for Robust 2D Representations

**arXiv ID:** 2602.12524 | [PDF](https://arxiv.org/pdf/2602.12524v1)

**作者:** Wonjun Jo `[一作]` (POSTECH), Tae-Hyun Oh `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了协同蒸馏（Collaborative Distillation）方法，用3D LiDAR自监督提升2D图像编码器在恶劣天气下的鲁棒性，同时保持原有语义表现；

**💡 创新点**

创新点在于两阶段自监督蒸馏：先将LiDAR特征与清晰天气下的2D特征对齐，再用对齐后的3D特征作为自监督锚点，既增强了2D表示的鲁棒性，又保留了语义上下文；

**🔧 技术方法**

采用跨模态对齐与双向蒸馏损失，利用投影配准获取像素-点对应关系，结合ViT、WaffleIron-768等编码器，进行对齐与自监督训练；

**📊 数据集**

主要使用nuScenes数据集（含清晰、雨夜等多天气），并在KITTI、NYU‑d、Cityscapes、ADE20k等跨域数据集验证泛化能力；

**📈 对比分析**

与DINOv2、FiT3D、Condense等方法对比，在线域语义分割、深度估计以及离域深度/语义分割均取得更高mIoU/更低RMSE，且在多任务视频全景分割中提升VPQ；

**⚠️ 局限性**

局限性包括对3D LiDAR硬件的依赖、在极端天气（如暴雪、浓雾）下仍可能受限，以及在非户外场景下的性能需进一步验证。

---

## 326. Vehicle behaviour estimation for abnormal event detection using distributed fiber optic sensing

**arXiv ID:** 2602.12591 | [PDF](https://arxiv.org/pdf/2602.12591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 327. Analysis of Asset Administration Shell-based Negotiation Processes for Scaling Applications

**arXiv ID:** 2602.13029 | [PDF](https://arxiv.org/pdf/2602.13029v1)

**作者:** David Dietrich `[一作]` (University of Stuttgart), Alexander Verl `[通讯]` (University of Stuttgart)

**通讯引用:** 8645 | [OpenAlex ID](https://openalex.org/A5048205111)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文评估了基于主动资产管理壳（Proactive AAS）的协商机制在大规模工业场景中的可扩展性，并通过模拟实验验证其性能瓶颈。

**💡 创新点**

提出了针对主动AAS的可扩展协商模型、量化指标Ψ以及在Docker Swarm中多播CFP的实现，并给出改进方向。

**🔧 技术方法**

使用Python实现协商流程，MQTT通信，Docker Swarm扩容，Proxmox虚拟机部署，随机生成服务属性。

**📊 数据集**

采用随机生成的服务请求与供应商参数（成本、工时、预算、质量等）进行10次模拟，每个规模k从2^0到2^8。

**📈 对比分析**

通过比较不同规模下的总时间Δt、消息量、成功率等指标，发现Ψ呈不可扩展的指数增长，消息负载和CFP次数显著上升，成功率在k≥2^3后趋近1。

**⚠️ 局限性**

实验受限于自定义实现、单节点Swarm调度延迟、简化的服务描述，未覆盖多租户网络拥塞与标准化交互细节。

---

## 328. Thermal Imaging for Contactless Cardiorespiratory and Sudomotor Response Monitoring

**arXiv ID:** 2602.12361 | [PDF](https://arxiv.org/pdf/2602.12361v1)

**作者:** Constantino Álvarez Casado `[一作]` (Center for Machine Vision and Signal Analysis), Miguel Bordallo López `[通讯]` (Center for Machine Vision and Signal Analysis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了通过热红外面部视频提取自主神经系统信号（EDA、心率HR和呼吸率BR），并构建了完整的信号处理管道

**💡 创新点**

首次系统评估多种ROI与多种EDA提取方法在同一数据集上的表现，并引入正负极性处理与多ROI融合思路

**🔧 技术方法**

使用YOLOv5-Face进行热面部检测，采用多种空间聚合（高斯加权、trimmed mean等）和时间分解（低通滤波、OMIT降噪、Welch谱估计）技术

**📊 数据集**

使用公开的SIMULATOR STUDY 1（SIM1）驾驶模拟器数据集，包含31个会话、8名受试者、7.5 Hz热摄像机与同步的手掌EDA、HR、BR等参考信号

**📈 对比分析**

通过与接触式基准的相关系数、误差指标比较，EDA最佳配置在鼻部+指数滑动平均时平均相关系数≈0.40，单个会话可达0.89；BR平均绝对误差≈3.1 bpm，HR平均误差≈13.8 bpm，说明HR受帧率限制

**⚠️ 局限性**

主要限制包括样本量有限、摄像机低帧率导致HR恢复不佳、热相位极性在不同会话中随机变化、仅评估趋势层面的EDA，未检验事件级别表现

---

## 329. SD-MoE: Spectral Decomposition for Effective Expert Specialization

**arXiv ID:** 2602.12556 | [PDF](https://arxiv.org/pdf/2602.12556v1)

**作者:** Ruijun Huang `[一作]` (Fudan University), Li Shang `[通讯]` (Fudan University)

**通讯引用:** 6399 | [OpenAlex ID](https://openalex.org/A5004722925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过谱分析揭示了 Mixture‑of‑Experts 模型中专家参数与梯度的主成分高度重叠，导致专家难以实现真正的专业化；并提出 Spectral‑Decoupled MoE（SD‑MoE）方案，在参数和梯度空间对每个专家进行谱分解，分离公共低秩子空间与专家特有的正交补空间，从而显著提升专家分化、降低参数冗余并提升下游任务性能。

**💡 创新点**

创新点在于：①首次从谱视角系统分析专家专业化失败的根源；②设计参数和梯度双向谱分解机制，实现跨专家的独立优化；③通过周期性 SVD 维持公共子空间的低秩特性，兼顾训练稳定性与效率。

**🔧 技术方法**

主要技术包括：奇异值分解（SVD）实现谱分解；主子空间相似度衡量专家重叠；梯度投影到共享子空间与正交补的分解；以及在现有 Qwen、DeepSeek 等 MoE 结构中无缝集成。

**📊 数据集**

实验使用公开的 100B DCLM 语料库进行训练，评估 8 大基准任务（ARC‑challenge/easy、HellaSwag、LAMBADA、PIQA、RACE、SIQA、Winogrande 等）。

**📈 对比分析**

与原始 MoE 的基线相比，SD‑MoE 在大多数任务平均提升约 3% 分数，训练效率提升约 30%，可使用更高学习率而不失稳，整体计算开销仅增加约 5%。

**⚠️ 局限性**

局限性包括：需手动设定共享子空间秩 k，过大或过小可能影响效果；需要周期性 SVD 计算，对极大模型可能带来额外开销；目前仅在 Qwen、DeepSeek 这两种 MoE 架构上验证，未覆盖更广泛的模型与任务。

---

## 330. From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness

**arXiv ID:** 2602.12285 | [PDF](https://arxiv.org/pdf/2602.12285v1)

**作者:** Linbo Cao `[一作]` (University of Waterloo), Yang Yue `[通讯]` (University of Wollongong)

**通讯引用:** 26721 | [OpenAlex ID](https://openalex.org/A5100322712)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了人口身份对LLM代理在多种任务上的性能影响，量化了不同种族、性别、宗教、职业标签对代理行为的偏差；

**💡 创新点**

首次将人口身份偏差从文本生成扩展到实际行动执行，揭示了隐式偏见对高层推理和规划任务的显著影响，并提供可复制的实验框架；

**🔧 技术方法**

采用persona前缀提示方法，对三种主流LLM（GPT‑4o‑mini、DeepSeek‑V3、Qwen3‑235B）在五个代理基准上进行评估；

**📊 数据集**

使用多维人口身份标签集（性别、种族/来源、宗教、职业）和五个标准代理基准（ALFWorld、WebShop、Card Game、OS Interaction、Database）；

**📈 对比分析**

通过对比无persona基线与各persona条件下的性能，计算百分比变化；结果显示最大降幅达26.2%，尤其在战略推理与多步规划任务中表现最差；

**⚠️ 局限性**

实验仅覆盖有限数量的任务和模型，未涉及更复杂的环境、跨模型对齐差异、以及机制解释，结果可能随预训练数据和对齐策略变化而异；

---

## 331. Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution

**arXiv ID:** 2602.12684 | [PDF](https://arxiv.org/pdf/2602.12684v1)

**作者:** Rui Cai `[一作]` (Xiaomi Robotics), Quanyun Zhou `[通讯]` (Xiaomi Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于预训练视觉语言模型与扩散变换器的端到端视觉‑语言‑动作（VLA）框架，并通过两阶段预训练与后训练实现高性能、实时平滑的双臂机器人控制。

**💡 创新点**

创新点包括：①在后训练中引入 Λ‑形注意力掩码和动作前缀偏移，打破仅靠前缀复制的“短路”行为；②采用流匹配（flow‑matching）扩散变换器生成动作，提升连续性与可控性；③实现异步推理，显著降低推理延迟并保持动作平滑。

**🔧 技术方法**

使用技术：预训练视觉语言模型（Qwen3‑VL‑4B‑Instruct）、扩散变换器（DiT）与流匹配损失、Λ‑形注意力掩码、动作前缀技术、异步执行框架（RTC 训练版）、AdaLN、RoPE 等。

**📊 数据集**

数据集：200M 机器人轨迹（来自 DROID、MolmoAct、公司内 Teleop 数据；338h 乐高拆解、400h 毛巾折叠）、80M 视觉‑语言样本（通用 VL 数据与机器人轨迹生成的 VL 数据），以及 LIBERO、CALVIN、SimplerEnv（Google Robot、WidowX）等评测数据。

**📈 对比分析**

与现有 VLA 及基线方法（π₀.₅、MolmoAct 等）对比，模型在 LIBERO 取得 98.7% 平均成功率、CALVIN 5 任务顺序完成 4.80/4.75、SimplerEnv 视觉匹配 85.5%、聚合 74.7%、WidowX 79.2%，在所有仿真基准上均为 SoTA；在真实双臂任务中，异步版本实现最高吞吐率（乐高拆解、毛巾折叠分别比同期基线提升 15%–20%），且保持高成功率。

**⚠️ 局限性**

局限性：①模型参数量 4.7B，推理仍需 GPU（RTX 4090）约 80 ms，虽已优化但对低功耗平台仍受限；②异步执行依赖前缀信息，若前缀误差累积仍可能导致动作迟钝；③对大规模、多模态数据的依赖，若数据不足易出现灾难性遗忘；④在极端外观或物理变化的环境中，仍需进一步提升泛化能力。

---

## 332. Universal Transformation of One-Class Classifiers for Unsupervised Anomaly Detection

**arXiv ID:** 2602.13091 | [PDF](https://arxiv.org/pdf/2602.13091v1)

**作者:** Declan McIntosh `[一作]` (University of Victoria), Alexandra Branzan Albu `[通讯]` (University of Victoria)

**通讯引用:** 1151 | [OpenAlex ID](https://openalex.org/A5036029722)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种数据集折叠（folding）方法，将任意一类分类器（OCC）异常检测器转换为无监督模型，通过多折叠和投票方式过滤训练数据中的异常样本。

**💡 创新点**

创新点在于：①无须改动原有OCC算法；②利用弱假设（异常稀少且异质）实现高精度异常筛选；③首次实现无监督逻辑异常检测，并将OCC性能迁移到无监督任务；④在多种数据集上实现SOTA表现。

**🔧 技术方法**

核心技术包括：多折叠训练、投票机制、对每折预测的混合高斯模型（GMM）阈值筛选、以及对视频数据的剪辑/形态学后处理；实现时复用PatchCore、DinoAnomaly、EfficientAD、PUAD等OCC模型。

**📊 数据集**

使用的公开数据集有：MVTec AD、ViSA、MVTec Loco AD（逻辑异常）、DMAD/Peds2视频异常集，此外在不同异常污染比例下进行实验。

**📈 对比分析**

与SoftPatch、FUN‑AD、InReaCh等无监督基线以及原始OCC基线对比，Folded（多折叠/投票）方案在I‑AUROC、P‑AUROC、AUPRO指标上均显著提升，尤其在含10%异常污染的训练集上实现了接近或超过无监督基线的最高性能。

**⚠️ 局限性**

主要局限包括：训练时间显著增加（折叠×投票导致数倍到数十倍训练成本），对异常稀少且独立采样的假设敏感；以及继承原OCC方法的局限性（如对预训练特征提取器的依赖）。

---

## 333. Schur-MI: Fast Mutual Information for Robotic Information Gathering

**arXiv ID:** 2602.12346 | [PDF](https://arxiv.org/pdf/2602.12346v1)

**作者:** Kalvik Jakkala `[一作]`, Srinivas Akella `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Schur-MI的高效互信息（MI）计算方法，旨在提高机器人信息收集（RIG）的实时规划能力。

**💡 创新点**

通过引入Schur补因子分解和预计算策略，显著降低了MI的计算复杂度，使得MI的每次评估成本从O(|V|^3)降低到O(|A|^3)。

**🔧 技术方法**

使用了高斯过程（GP）模型和Schur补因子分解技术。

**📊 数据集**

在真实的水深数据集上进行了实验，包括密西西比、南塔基特、维尔京群岛和拉恩戈尔等数据集。

**📈 对比分析**

与标准MI方法相比，Schur-MI在运行时间上实现了高达12.7倍的加速，同时在重建精度上保持了相似的性能，尤其在离散空间和连续空间中均表现出色。

**⚠️ 局限性**

在某些情况下，Schur-MI可能会引入额外的计算开销，尤其是在处理非平稳核时，可能会影响其性能。

---

## 334. Classification of Local Optimization Problems in Directed Cycles

**arXiv ID:** 2602.13046 | [PDF](https://arxiv.org/pdf/2602.13046v1)

**作者:** Thomas Boudier `[一作]` (Gran Sasso Science Institute), Jukka Suomela `[通讯]` (Aalto University)

**通讯引用:** 2461 | [OpenAlex ID](https://openalex.org/A5025555126)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对定向环图上所有局部优化问题（包括最大/最小独立集、最小支配集、最小顶点着色等）在分布式LOCAL模型中的计算复杂度进行了完整的分类，并给出了一个元算法，能够自动确定任意给定问题及其逼近比的时间复杂度，还能生成相应的最优分布式算法。

**💡 创新点**

创新点包括：① 将传统的局部可检查问题（LCL）框架推广到局部优化问题；② 通过引入七个结构参数（β0…β4、γ、δ）在de Bruijn图上捕捉优化问题的本质，进而得到唯一的五类复杂度分布；③ 提供了多项式时间的中心化算法自动计算这些参数并推断复杂度；④ 证明了在定向环上不存在介于已知五类之间的复杂度，从而完成了完整的分类。

**🔧 技术方法**

主要技术手段包括：de Bruijn图构造与分析、可变长度闭环与可灵活性（flexibility）判定、费马数与线性组合论证、Ramsey理论与无偏置随机化、局部规则与规则集（ruling set）构造、以及对自环（self‑loop）与可变长度分段的成本分析。

**📊 数据集**

由于论文为理论分析，不使用实验数据或标准数据集。所有结果均在数学上证明，复杂度判定与算法构造仅依赖问题描述的有限符号表示。

**📈 对比分析**

与传统局部可检查问题相比，作者给出的元算法可在多项式时间内对任意局部优化问题进行复杂度分类，并可自动生成在确定性、随机化模型下的最优近似算法。相比先前仅针对特定问题的结果，该方法实现了统一的、可自动化的复杂度分析，显著提高了可扩展性和适用性。

**⚠️ 局限性**

局限性主要有：① 仅在定向环图上完成分类，尚未推广到无向环、路径、树或一般图；② 只考虑无输入标签（unlabeled）情形，输入标签的情况仍是不可判定的问题；③ 对于更强的模型（如带有完整对称性破除的MPC/LOCAL变体）仍存在未解决的开放问题；④ 论文关注的是常数逼近因子，对于极小的逼近误差（如1+1/n）并未给出完整分析。

---

## 335. Aspect-Based Sentiment Analysis for Future Tourism Experiences: A BERT-MoE Framework for Persian User Reviews

**arXiv ID:** 2602.12778 | [PDF](https://arxiv.org/pdf/2602.12778v1)

**作者:** Hamidreza Kazemi Taskooh `[一作]` (Iran University of Science and Technology), Taha Zare Harofte `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个针对波斯语旅游评论的三阶段ABSA框架，结合BERT与Mixture‑of‑Experts实现细粒度情感分析

**💡 创新点**

创新点在于引入BERT‑MoE混合模型与Top‑K路由及辅助损失，解决低资源语言的路由崩溃问题，并显著提升能效

**🔧 技术方法**

使用BERT基础模型、Mixture‑of‑Experts、Top‑K路由、辅助损失、LoRA（对比实验）以及PyTorch+Optuna调参

**📊 数据集**

采用从Jabama平台收集的58,473条标注好的评论数据，包含六大旅游维度（host, price, location, amenities, cleanliness, connectivity）

**📈 对比分析**

与传统BERT（89.25% F1）和BERT+MoE+LoRA（85.7% F1）对比，最终模型达到90.6%加权F1，且GPU功耗下降39%

**⚠️ 局限性**

局限包括类别不平衡、缺乏子维度标注、模型仅在波斯语旅游域验证，尚未测试跨语言或更细粒度的子维度情感

---

## 336. Unleashing Low-Bit Inference on Ascend NPUs: A Comprehensive Evaluation of HiFloat Formats

**arXiv ID:** 2602.12635 | [PDF](https://arxiv.org/pdf/2602.12635v1)

**作者:** Pengxiang Zhao `[一作]` (Huawei Technologies Co., Ltd.), Zhenhua Dong `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了面向 Ascend NPU 的低比特浮点量化格式 HiFloat（HiF8、HiF4），并将其与现有的 MXFP 与 NVFP4 进行对比。

**💡 创新点**

创新点在于提出 HiFloat 采用动态指数/尾数分配（HiF8）和三级层级缩放（HiF4）两种低比特浮点方案，并证明其在 W8A8、W4A4 以及 KV 缓存量化中的卓越鲁棒性。

**🔧 技术方法**

技术包括基于 SQNR 的分布分析、与 SmoothQuant 与 SVDQuant 的 PTQ 结合、以及针对权重、激活和 KV 缓存的端到端推理评估。

**📊 数据集**

使用的数据集包括 Wikitext‑2、C4、ARC Challenge、HellaSwag、MMLU、Math‑500、GSM8K、LongBench 等多种语言模型基准。

**📈 对比分析**

对比方法是通过 perplexity、Zero‑shot 任务准确率以及与 BF16 基线的百分比损失来衡量，结果表明 HiF4 在 4‑bit 量化下能保持 96–97% 的 BF16 性能，HiF8 在 8‑bit 量化中与 MXFP8 接近。

**⚠️ 局限性**

局限性在于仅评估了 Ascend NPU 硬件，对其他 GPU/TPU 的兼容性未作深入验证，并且在极端低比特（如 2‑bit）或特定模型结构下的效果尚未充分探索。

---

## 337. Trust the uncertain teacher: distilling dark knowledge via calibrated uncertainty

**arXiv ID:** 2602.12687 | [PDF](https://arxiv.org/pdf/2602.12687v1)

**作者:** Jeonghyun Kim `[一作]` (Ewha Womans University), Hyunsoo Cho `[通讯]` (Ewha Womans University)

**通讯引用:** 1603 | [OpenAlex ID](https://openalex.org/A5062457999)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Calibrated Uncertainty Distillation（CUD），通过教师分布的难度感知不确定性塑形和错误质量裁剪来改进知识蒸馏

**💡 创新点**

创新点在于在蒸馏前主动校准教师分布，保持暗知识且抑制过度自信，使用 DUS 和 W‑Clip 两个可插拔模块实现

**🔧 技术方法**

技术包括教师微调的焦点熵损失、熵门控难度塑形、误差裁剪投影、温度化 KL 蒸馏，以及结构敏感的距离度量

**📊 数据集**

在 Banking77、CLINC150、MASSIVE、TREC、AG News 等多类别分类数据集上进行实验

**📈 对比分析**

与 LKD、PKD、TinyBERT、CKD、MGSKD、AD‑KD 等基线对比，CUD 在多数任务上提升准确率（尤其是高类别数任务）并显著改善校准、OOD 检测和选择性预测指标

**⚠️ 局限性**

局限包括仅针对单标签分类、超参数需手工设定、对二分类或极少类别任务收益有限，以及未在真正的分布漂移或生成任务上验证

---

## 338. Photonic Rails in ML Datacenters with Opus

**arXiv ID:** 2602.12521 | [PDF](https://arxiv.org/pdf/2602.12521v1)

**作者:** Eric Ding `[一作]` (Cornell University), Rachee Singh `[通讯]` (Cornell University)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5071694147)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将光学电路开关（OCS）替代传统电路式铁路互连，构建光学铁路架构并实现相应的控制平面（Opus），用于大规模分布式机器学习训练。

**💡 创新点**

创新点在于利用并行性驱动的铁路重构：在不同的并行阶段之间通过可预测的通信窗口动态重配光学电路，实现高功耗/成本低的光学铁路，同时保持所有GPU之间的全连通幻觉。

**🔧 技术方法**

采用光学电路开关（MEMS/液晶OCS）、分布式控制层（shim、controller、network orchestrator）、NCCL集体通信、Python/TL1接口以及AstraSim仿真框架。

**📊 数据集**

使用大型语言模型训练工作负载（Llama3-8B、Llama3-70B）进行实验，基准数据为相应模型在不同并行配置下的训练迭代。

**📈 对比分析**

与传统电路式铁路网络对比，光学铁路在物理OCC硬件测试、Perlmutter 64 GPU仿真以及2,048 GPU仿真中分别实现约23×功耗下降、4×成本降低，并且训练迭代时间仅提升不超过6%，显示出优秀的性能与能效权衡。

**⚠️ 局限性**

限制在于OCS的重构延迟（需在通信窗口内完成）、对窗口长度的依赖、每GPU端口数量限制以及对更多并行维度的支持仍需进一步改进。

---

## 339. Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval with Training-Free Pooling and Multi-Stage Search

**arXiv ID:** 2602.12510 | [PDF](https://arxiv.org/pdf/2602.12510v1)

**作者:** Ara Yeroyan `[一作]` `[通讯]` (Independent Researcher), Ara Yeroyan (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可在消费级硬件上运行的 Visual RAG Toolkit，通过训练‑free 的空间池化将每页的数千个视觉向量压缩至数十个，并结合多阶段检索实现高效的文档检索。

**💡 创新点**

创新点在于：①针对不同 VLM 架构设计的无训练空间池化策略，显著减少向量数并保持检索质量；②多阶段检索方案，先用压缩向量快速筛选候选，再用完整多向量向量精确 MaxSim 重新排序；③提供完整的预处理、评估管线，降低硬件门槛。

**🔧 技术方法**

主要技术包括：PaliGemma‑3B、Qwen2‑VL、SmolVLM 等 VLM 的视觉嵌入；模型感知的行/列/瓦片均值池化、滑动窗口卷积与加权平滑；Qdrant 的命名向量索引与多阶段查询；PDF 转图像、空白区域裁剪、非视觉 token 清洗等预处理；评估指标 NDCG、Recall@k、QPS。

**📊 数据集**

使用 ViDoRe v2 benchmark（ESG、Biomedical、Economics 三大子集共 3006 页）进行评估；对比官方 leaderboard 与 1‑stage、2‑stage、3‑stage 检索结果。

**📈 对比分析**

在 2‑stage 检索下，3B 模型（ColPali‑v1.3、ColQwen2.5‑v0.2）在 NDCG@5/10、Recall@5/10 维持 ±0.01 的精度，QPS 提升约 3.8–4.5 倍；在 Recall@100 上略有下降。小模型 ColSmol‑500M 在压缩后性能下降更明显，3‑stage 级联可部分恢复，但 QPS 降低。整体来看，速度提升随集合规模增大而增强。

**⚠️ 局限性**

局限性：池化策略需针对每种 VLM 结构手工设计，无法直接迁移到新型模型；对小于 1B 参数的模型，池化导致信息损失；未引入学习式池化或量化，尚未实现 1‑stage 全压缩检索；未结合 HNSW、量化等其他稀疏化技术。

---

## 340. Leverage-Weighted Conformal Prediction

**arXiv ID:** 2602.12693 | [PDF](https://arxiv.org/pdf/2602.12693v1)

**作者:** Shreyas Fadnavis `[一作]` `[通讯]`, Shreyas Fadnavis

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于统计杠杆的加权 conformal 预测方法（LWCP），通过对非一致性分数乘以杠杆函数实现分布无关的自适应预测区间。

**💡 创新点**

创新点在于无需训练辅助模型，仅利用设计矩阵的几何杠杆信息来实现权重调整，从而在保持有限样本边际有效性的同时显著改善条件覆盖率。

**🔧 技术方法**

技术上采用了杠杆分数（hat 矩阵对角线）、SVD 近似、可测权重函数以及随机化 SVD 以降低计算复杂度；同时结合了学生化与岭杠杆扩展（LWCP+）。

**📊 数据集**

实验使用合成数据（包括文本、重尾、非线性、多项式、同方差和对抗性 DGP）以及四个公开真实数据集（CPU 活动、糖尿病等）进行评估。

**📈 对比分析**

与 Vanilla CP、CQR、学生化 CP、局部 CP 等方法比较，LWCP 在保持相同 90% 置信水平的前提下，显著降低了条件覆盖率差距（从数个百分点到近 0），且计算开销与 Vanilla CP 相当，宽度增幅可忽略。

**⚠️ 局限性**

局限性包括对杠杆相关异方差的假设；当噪声方差与杠杆无关时改进有限；在高维极端多重共线性时需要岭杠杆；对极端异常点的鲁棒性尚未系统研究。

---

## 341. Tracking The Trackers: Commercial Surveillance Occurring on U.S. Army Networks

**arXiv ID:** 2602.12388 | [PDF](https://arxiv.org/pdf/2602.12388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 342. Nonparametric Contextual Online Bilateral Trade

**arXiv ID:** 2602.12904 | [PDF](https://arxiv.org/pdf/2602.12904v1)

**作者:** Emanuele Coccia `[一作]` (Bocconi University), Andrea Celli `[通讯]` (Bocconi University)

**通讯引用:** 277 | [OpenAlex ID](https://openalex.org/A5023478606)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了上下文在线双边交易问题，设计了一种算法以在不知道买卖双方私有估值的情况下提出交易价格，目标是促进交易并最大化社会福利。

**💡 创新点**

在非参数设置下，算法能够处理任意Lipschitz函数的估值，并在一位反馈和强预算平衡的条件下保证O(T^(d-1)/d)的遗憾界限。

**🔧 技术方法**

使用了分层树结构来利用上下文信息，并通过随机化技术来优化价格的选择。

**📊 数据集**

使用了生成的上下文向量和L-Lipschitz函数来模拟买卖双方的私有估值。

**📈 对比分析**

与现有的线性模型方法相比，提出的算法在一位反馈下能够达到更优的遗憾界限O(T^(d-1)/d)，并且在全反馈设置下提供了匹配的下界。

**⚠️ 局限性**

算法依赖于已知的Lipschitz常数，实际应用中可能无法获得该常数，导致性能下降。

---

## 343. Geometric separation and constructive universal approximation with two hidden layers

**arXiv ID:** 2602.12482 | [PDF](https://arxiv.org/pdf/2602.12482v1)

**作者:** Chanyoung Sung `[一作]` (Korea National University of Education), Chanyoung Sung `[通讯]` (Korea National University of Education)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5064172909)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

构造两隐藏层神经网络的几何方法实现任意连续函数的逼近。

**💡 创新点**

提出新的 Urysohn‑式分离引理和几何构造，证明两隐藏层即可实现全局逼近。

**🔧 技术方法**

利用几何分离、Urysohn 引理、Tietze 扩张以及激活函数的特性进行构造。

**📊 数据集**

无实验数据集，纯理论证明。

**📈 对比分析**

无比较实验，结果为理论证明的可达性与逼近误差可任意小。

**⚠️ 局限性**

构造宽度可能非常大，未给出高效的训练或实现方法。

---

## 344. Memory-Efficient Structured Backpropagation for On-Device LLM Fine-Tuning

**arXiv ID:** 2602.13069 | [PDF](https://arxiv.org/pdf/2602.13069v1)

**作者:** Juneyoung Park `[一作]` (Opt-AI Inc.), Jaeho Lee `[通讯]` (Opt-AI Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Memory-efficient Structured Backpropagation (MeSP) 算法，手动推导 LoRA 结构的反向传播，利用重算中间投影 h 来显著降低显存占用，并保持与传统梯度完全一致。

**💡 创新点**

创新点在于：① 将 LoRA 的低秩结构显式拆解，重新设计梯度计算流程；② 通过在反向阶段重算 h，完全避免在前向阶段缓存大量中间张量；③ 通过精细控制张量生命周期实现 49% 平均显存减少，同时保持梯度精度。

**🔧 技术方法**

核心技术包括：LoRA 参数化、手工反向传播（Explicit Backprop）、梯度检查点（Gradient Checkpointing）与自动微分的结合、Apple Silicon MLX 框架、4‑bit 量化权重与 bfloat16 LoRA 参数、以及统一内存管理。

**📊 数据集**

实验数据集为 WikiText-2，批量大小为 1，学习率 1e‑4，使用 SGD 优化器。

**📈 对比分析**

对比方法：与 MeBP（梯度检查点）和 MeZO（零阶估计）在同一硬件（Apple Silicon）下进行。MeSP 在 Qwen2.5-0.5B 上峰值显存从 361 MB 降至 136 MB（62%），在不同模型大小、序列长度和 LoRA rank 下也保持 42–62% 的显存削减；训练时间略增 28–31%，但收敛曲线与 MeBP 完全一致，最终损失与 MeBP 相同。

**⚠️ 局限性**

局限性：① 方案仅针对 LoRA 低秩结构，尚未验证可否推广到其他稀疏/低秩微调方法；② 重算中间张量会带来一定计算开销；③ 主要实验在 Apple Silicon 设备上，未在 GPU/其他统一内存平台全面验证；④ 对更大模型或更高 LoRA rank 的可扩展性仍需进一步评估。

---

## 345. SafeFlowMPC: Predictive and Safe Trajectory Planning for Robot Manipulators with Learning-based Policies

**arXiv ID:** 2602.12794 | [PDF](https://arxiv.org/pdf/2602.12794v1)

**作者:** Thies Oelerich `[一作]` (Automation and Control Institute TU Wien), Andreas Kugi `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为SafeFlowMPC的在线轨迹规划方法，将流匹配学习与模型预测控制相结合，实现了安全性与实时性的统一；

**💡 创新点**

创新点在于引入安全流匹配过程，并通过安全流形与终端约束深度集成，使得规划过程始终保持在安全集合内，同时兼顾学习的灵活性；

**🔧 技术方法**

主要技术包括条件流匹配网络、实时迭代（RTI）优化求解器、前向运动学碰撞检测以及指导成本函数；

**📊 数据集**

使用了基于VP‑STO全局规划器产生的4000条轨迹作为学习样本，并对900条人类-人类交接轨迹进行机器人关节空间映射；

**📈 对比分析**

在三种实验（静态抓取、在线重规划和动态人机交接）中，与VP‑STO、BoundMPC、BC、FM等方法对比，SafeFlowMPC在保持相近轨迹时间的同时，成功率和安全性均优于传统MPC方法，且能在实时约束下运行；

**⚠️ 局限性**

局限性包括需预先拥有足够且质量良好的演示数据集，必须进行微调；仅能处理可微分的约束；终端安全约束设计复杂；对移动人类的预测仍是挑战，无法完全适应高度动态交互场景。

---

## 346. Bonik Somiti: A Social-market Tool for Safe, Accountable, and Harmonious Informal E-Market Ecosystem in Bangladesh

**arXiv ID:** 2602.12650 | [PDF](https://arxiv.org/pdf/2602.12650v1)

**作者:** ATM Mizanur Rahman `[一作]` (University of Illinois Urbana-Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 649 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对孟加拉国非正式电商中的诈骗进行调查，并设计了 Bonik Somiti 系统以支持结构化举报、管理员历史追踪、调解与正式机构升级。

**💡 创新点**

提出社区中心的社会技术方案，将举报、证据管理、调解与执法渠道整合在同一平台，首次在全球南部场景中实现从用户到执法的闭环。

**🔧 技术方法**

采用了结构化表单、对话渠道、管理员历史数据库、人工审核结合 AI 规则查询的技术栈；在原型中实现了基于角色的报告、证据上传和案件状态跟踪。

**📊 数据集**

使用了 124 份匿名问卷和 36 次深度访谈（包括买家、卖家、群组管理员、警察、银行/金融科技工作人员）的定性与定量数据。

**📈 对比分析**

通过 32 名参与者的焦点小组和访谈评估原型，主要以用户体验、可行性与安全性等质性指标进行比较；报告显示系统能提升举报透明度并减少信息碎片化，但缺乏客观性能数值。

**⚠️ 局限性**

局限性包括样本主要来自网络社群、缺乏多样化的用户群、原型未与主流平台（如 Facebook）真正集成、未在真实执法流程中验证升级效果，以及对长期使用效果缺乏实地部署。

---

## 347. Self-Refining Vision Language Model for Robotic Failure Detection and Reasoning

**arXiv ID:** 2602.12405 | [PDF](https://arxiv.org/pdf/2602.12405v1)

**作者:** Carl Qi `[一作]` (UT Austin), Yesh Dattatreya `[通讯]` (Amazon Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ARMOR，一种基于多轮自我改进的多任务视听模型，用于机器人失效检测和自然语言失效推理。

**💡 创新点**

创新点在于：1) 采用多任务头同时预测二分类失效与开放式推理；2) 通过离线模仿与在线强化学习实现对稀疏检测标签和稠密推理标签的异构监督学习；3) 在推理时使用熵作为自我置信度来决定多轮改进的终止与最佳轨迹选取。

**🔧 技术方法**

使用视觉‑语言模型（如 Qwen2.5‑VL）作为基座，结合多任务头、离线模仿、在线强化学习、熵自我校准等技术。

**📊 数据集**

在四个多样化的数据集上评估：RLBench‑Fail、Maniskill‑Fail、Sparrow‑Fail 以及 ARMBench‑Fail，其中部分数据集提供稀疏的成功/失败标签，部分提供稠密的推理文本。

**📈 对比分析**

与公开的 VLM（Qwen2.5‑VL、Cosmos‑Reasoning、LLaVA‑NEXT）及 Claude‑3.7 对比，ARMOR 在失效检测准确率提升 20–30% 以上，在推理质量（LLM 模糊匹配与 ROUGE‑L）上提升 40–70% 以上，且在跨域稀疏标签迁移场景中保持稳健性。

**⚠️ 局限性**

局限性包括：1) 推理质量仍受限于稠密标注规模，稠密数据稀缺时提升有限；2) 迭代改进导致推理时延增加，需权衡轮数；3) 依赖视觉输入，未集成力/触觉等其他感知模态，可能漏检非视觉相关失效。

---

## 348. Preference-Guided Prompt Optimization for Text-to-Image Generation

**arXiv ID:** 2602.13131 | [PDF](https://arxiv.org/pdf/2602.13131v1)

**作者:** Zhipeng Li `[一作]` (ETH Zurich), Christian Holz `[通讯]` (ETH Zurich)

**通讯引用:** 6809 | [OpenAlex ID](https://openalex.org/A5046815740)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种仅依赖用户二元偏好选择的文本提示优化方法（APPO），通过保留、对齐和扩展三种策略以及自适应探索策略，自动改进文本到图像生成的提示语；

**💡 创新点**

在保持低用户努力（仅选择图片）的前提下，系统能够在少量迭代内快速收敛并提升生成质量；同时首次在图像生成任务中将对齐、扩展和自适应探索三者结合，显著提升样本效率；

**🔧 技术方法**

使用大型语言模型（Gemini 2.5 Flash）进行提示重写、梯度估计和变异；利用CLIP进行语义相似度计算；采用Stable Diffusion XL作为生成器；实现了保留、对齐、扩展和自适应扩展策略；

**📊 数据集**

合成实验采用改写自TIFA160等公开图像数据集的提示；用户研究使用基于Stable Diffusion XL的真实图像生成任务，包含闭合式与开放式两种情景；

**📈 对比分析**

与PromptCharm、DSPy、Clarification等基线进行比较，评估指标包括迭代次数、总耗时、NASA‑TLX工作量得分和CSI创造力支持指数；实验表明APPO在迭代次数和耗时上均优于基线（平均迭代次数<4，耗时显著下降），且用户的工作量感知和满意度显著提高；

**⚠️ 局限性**

依赖用户在起始提示中准确提供必要要素，若缺失关键对象或包含过多要素，优化可能失败；对复杂任务（多元化或动态目标）适应性有限；当前仅支持文本提示，未开放提示可视化；

---

## 349. EARL: Energy-Aware Adaptive Antenna Control with Reinforcement Learning in O-RAN Cell-Free Massive MIMO Networks

**arXiv ID:** 2602.12841 | [PDF](https://arxiv.org/pdf/2602.12841v1)

**作者:** Zilin Ge `[一作]` (KTH Royal Institute of Technology), Cicek Cavdar `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2858 | [OpenAlex ID](https://openalex.org/A5006937058)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出 EARL 方案，在 O‑RAN 细胞自由大规模 MIMO 网络中通过强化学习动态控制天线/单元激活以降低端到端功耗并满足用户谱效率需求。

**💡 创新点**

将中央预编码与天线级功耗建模相结合，首次将 RL 框架应用于能耗最小化，并引入贪婪后处理进一步压缩功耗。

**🔧 技术方法**

使用 Proximal Policy Optimization (PPO) 强化学习算法，结合 O‑RAN 资源共享模型和天线级功耗模型。

**📊 数据集**

使用仿真生成的随机 UE 位置信息和信道增益（无公开数据集），在 MATLAB/Python 环境下评估。

**📈 对比分析**

与全开、启发式基线对比，RL 版在 2 s 以内完成决策，功耗降低 81%（全开）/50%（启发式），贪婪后处理进一步降至 50% 并使功耗下降 87% 以上。

**⚠️ 局限性**

依赖中央预编码的 O‑RAN 拆分方案；RL 需要训练样本且在极端 SE 需求下可能出现 SE 违规；贪婪后处理会显著增加计算时延。

---

## 350. Learning to Approximate Uniform Facility Location via Graph Neural Networks

**arXiv ID:** 2602.13155 | [PDF](https://arxiv.org/pdf/2602.13155v1)

**作者:** Chendi Qian `[一作]` (RWTH Aachen University), Christian Sohler `[通讯]` (University of Cologne)

**通讯引用:** 3891 | [OpenAlex ID](https://openalex.org/A5034247955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种全可微分、无监督的消息传递图神经网络(MPNN)架构，用于解决统一设施位置问题（UniFL）并实现近似算法；

**💡 创新点**

创新点在于将经典分布式近似算法的结构嵌入MPNN中，既保留了严格的近似保证，又通过学习实现了对实例分布的自适应改进；

**🔧 技术方法**

核心技术包括基于本地邻域信息的半连续半离散半量化半参数化近似半径估计、概率性设施开启机制以及基于期望成本的无监督损失；

**📊 数据集**

在合成高维几何图和真实城市道路网络（四个大城市地图）上进行了实验；

**📈 对比分析**

与传统的SimpleUniformFL、RecursiveUFL、K-Means++/k-Medoids++等基线比较，MPNN在多维度数据上实现了接近最优的成本比（<1.01），显著优于无监督近似算法且计算开销极低；

**⚠️ 局限性**

局限性包括：1）仅针对统一开设成本的UniFL；2）依赖分布式近似算法的结构，扩展到更复杂变体（非均匀成本、容量约束等）仍需重设计；3）对梯度下降收敛的理论分析尚不完整。

---

## 351. Learning Ordinal Probabilistic Reward from Preferences

**arXiv ID:** 2602.12660 | [PDF](https://arxiv.org/pdf/2602.12660v1)

**作者:** Longze Chen `[一作]` (Shenzhen Institutes of Advanced Technology), Min Yang `[通讯]` (Shenzhen Institutes of Advanced Technology)

**通讯引用:** 69873 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Ordinal Probabilistic Reward Model（OPRM）和Region Flooding Tuning（RgFT），用于在RLHF中更准确地建模奖励分布并校准绝对质量。

**💡 创新点**

创新点在于把奖励视为离散序数分布而非单一标量，既解决了判别式奖励模型缺乏可解释性，又克服了生成式模型昂贵的标注成本；RgFT通过质量等级注释对分布进行区域洪泛校准，实现半监督学习与绝对质量对齐。

**🔧 技术方法**

使用概率奖励建模、离散序数分布、语言模型头概率提取、加权平均解码、区域洪泛机制和混合监督训练策略。

**📊 数据集**

训练数据来自公开的Skywork Reward Preference、UltraFeedback Binarized Preferences；补充了良好/正常/差质量等级标注；使用Qwen2.5-Instruction系列模型作为基础模型。

**📈 对比分析**

与判别式（BT、PairRM）、生成式（GRM、LLM-as-a-Judge）以及DeepSeek基线在Reward Bench、PPE、RMB、Role Play等多种基准上进行比较；实验显示OPRM+RgFT相对基线提升2.9%–7.4%准确率，部分基准实现SOTA，并显著降低ECE校准误差。

**⚠️ 局限性**

局限性包括不同基准间效果不一，部分提升依赖人工质量等级标注，且注释策略可能引入偏差；在未标注或噪声较多的任务中性能提升有限，未来需要验证在更多领域和更大规模模型上的通用性。

---

## 352. Control Barrier Functions with Audio Risk Awareness for Robot Safe Navigation on Construction Sites

**arXiv ID:** 2602.12416 | [PDF](https://arxiv.org/pdf/2602.12416v1)

**作者:** Johannes Mootz `[一作]` (San Diego State University), Reza Akhavian `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了一种基于控制障碍函数（CBF）的安全滤波器，利用实时轻量级锤声检测器将施工现场的音频风险直接注入控制器，实现机器人在动态施工环境中的安全导航。

**💡 创新点**

创新点包括：①将音频风险感知（锤声检测）与CBF边界动态融合，形成音频参数化的CBF；②提出目标对齐的椭圆CBF，显著降低死锁概率；③采用SNR与自相关双模检测实现低延迟、无ML的锤声识别。

**🔧 技术方法**

使用技术包括：增广单车动力学模型、基于Lie导数的CBF构造、QP安全滤波器、SNR与自相关的锤声检测、椭圆形障碍约束。

**📊 数据集**

数据集：仿真中构建的BIM导向施工环境（5.5m×10m）以及四段预录制的锤声与背景噪音音频。

**📈 对比分析**

方法比较：与基准基准控制、圆形CBF（静态/音频）和椭圆CBF（静态/音频）对比；在两场景下，椭圆CBF成功率分别为73%和80%，显著优于圆形CBF（47%和33%），路径长度略增，所有CBF方案均未出现安全违规。

**⚠️ 局限性**

limitations：仅在仿真环境验证，缺乏真实工地实验；QP在约束不可行时可能失效；未实现全局重规划；音频检测在实际环境中的回声、延迟和多径效应未评估；对时变风险的CBF正式保证尚未保留。

---

## 353. Additively Competitive Secretaries

**arXiv ID:** 2602.12632 | [PDF](https://arxiv.org/pdf/2602.12632v1)

**作者:** Mohammad Mahdian `[一作]` (Google Research), Yifan Wang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8947 | [OpenAlex ID](https://openalex.org/A5100398573)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了在秘书问题中使用加法性回报（regret）评估框架，分析了价格曲线（pricing curves）和仅考虑最佳值的价格曲线（best‑only pricing curves）的回报上界，并给出了全局算法的下界；同时扩展到多选秘书问题和收益最大化场景。

**💡 创新点**

创新点包括：①首次将回报最小化框架引入随机顺序的秘书问题；②证明价格曲线的最优回报为 0.25，且构造了仅考虑最佳值的算法达到 0.190 的上界；③通过解析与计算得到全局下界 0.152，并给出 best‑only 类的下界 0.171；④将该框架推广到多选秘书问题和收益最大化问题，给出 Θ(√k) 的回报下界。

**🔧 技术方法**

主要技术手段包括：回报函数的解析上界推导、阈值曲线的几何分析、回报松弛（relaxation）与变量缩减、数值优化与计算机辅助搜索、Yao 兜底原理、逆函数与动态规划求解最优决策函数、对多选与收益最大化问题的归约与极限分析。

**📊 数据集**

数据集：本文为理论研究，无实验数据；所有分析均基于假设值在 [0,1] 范围内、到达顺序均匀随机的模型。

**📈 对比分析**

与传统的竞争比率（1/e 约 0.632）对比，本文通过回报框架展示价格曲线回报 0.25、best‑only 0.190，显著优于传统方法；全局下界 0.152 证明了回报框架下的最优性能界限。对于多选秘书问题，得到的 Θ(√k) 下界与已有的竞争比率结果形成互补。

**⚠️ 局限性**

限制：回报上界与下界并非严格匹配（best‑only 上界 0.190 与下界 0.171 仍有间隙）；计算机辅助搜索依赖具体参数，缺乏闭式解析；仅适用于单一选取或固定 k 的多选；假设值归一化且到达顺序随机，对实际应用的适用性需进一步验证。

---

## 354. Can Neural Networks Provide Latent Embeddings for Telemetry-Aware Greedy Routing?

**arXiv ID:** 2602.12798 | [PDF](https://arxiv.org/pdf/2602.12798v1)

**作者:** Andreas Boltres `[一作]` (Autonomous Learning Robots), Gerhard Neumann `[通讯]` (Autonomous Learning Robots)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Placer 算法，利用消息传递网络（MPN）将带有实时遥测数据的网络状态图映射为节点嵌入，并以此实现贪心路由，从而在毫秒级内完成路由决策。

**💡 创新点**

创新点在于将遥测感知与节点嵌入相结合，用 MPN 直接生成可解释的低维嵌入空间实现贪心路由；同时通过可视化嵌入来解释神经网络决策。

**🔧 技术方法**

使用的技术包括：消息传递网络（MPN）、Proximal Policy Optimization（PPO）强化学习、Boltzmann 探索、ns‑3 网络仿真器、PackeRL 框架、流量监测（FlowMonitor）。

**📊 数据集**

实验基于 Mini‑5（5 节点）拓扑，注入合成的 TCP/UDP 流量（模拟真实数据中心流量分布），并在 PackeRL 中模拟 400 步、5 ms/步的仿真。

**📈 对比分析**

与基线算法 EIGRP（短路径）、M‑Slim 以及不同维度的 Placer 进行比较。结果显示 Placer（维度 d=1/2/32）在总好通量上最高，丢包率最低，延迟与 EIGRP 相当，且波动率最低，说明其路由决策更稳定。

**⚠️ 局限性**

主要限制在于嵌入几乎保持不变，对遥测变化反应弱；中心化的单代理推理导致对称路由，难以处理链路利用的不对称性；低维嵌入空间可能不足以捕捉更大网络的几何结构。

---

## 355. Pursuit of Truth and Beauty in Lean 4: Formally Verified Theory of Grammars, Optimization, Matroids

**arXiv ID:** 2602.12891 | [PDF](https://arxiv.org/pdf/2602.12891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 356. Reproducing DragDiffusion: Interactive Point-Based Editing with Diffusion Models

**arXiv ID:** 2602.12393 | [PDF](https://arxiv.org/pdf/2602.12393v1)

**作者:** Ali Subhan `[一作]` (University of Ljubljana), Ashir Raza `[通讯]` (University of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 DragDiffusion 进行可复现性研究，验证其关键实验结果并评估各设计选择的敏感性。

**💡 创新点**

提出了通过单一中间时钟潜变量优化实现交互式点基图像编辑，并结合 LoRA 低秩适配、空间掩模正则化与 UNet 特征监督来实现高精度、身份保持的编辑。

**🔧 技术方法**

使用 Stable Diffusion v1.5、DDIM 反演、单时钟潜变量优化、LoRA 低秩适配、空间掩模正则化、UNet 特征监督，以及对多时钟潜变量进行的对比实验。

**📊 数据集**

在 DragBench 基准数据集上进行实验。

**📈 对比分析**

通过 Mean Distance（MD）与 Image Fidelity（IF）两项指标对比不同超参配置；实验结果与原论文趋势一致，单时钟优化在中间时钟上最优；多时钟优化未提升性能却显著增加计算成本。

**⚠️ 局限性**

受时间步选择、特征层、掩模正则化权重等超参高度敏感；实验依赖特定硬件与库版本；未在更大规模或不同模型上进行评估。

---

## 357. Unified Multi-Domain Graph Pre-training for Homogeneous and Heterogeneous Graphs via Domain-Specific Expert Encoding

**arXiv ID:** 2602.13075 | [PDF](https://arxiv.org/pdf/2602.13075v1)

**作者:** Chundong Liang `[一作]` (Tianjin University), Weixiong Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12685 | [OpenAlex ID](https://openalex.org/A5068659777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了统一多域图预训练框架 GPH^2，能够同时对同质图和异质图进行预训练，并在下游节点分类任务中实现稳定迁移。

**💡 创新点**

创新点包括：①将同质图与异质图统一构造为多视图图；②采用域特定专家编码，降低跨域分布差异；③设计任务导向专家融合机制，自适应组合不同专家输出。

**🔧 技术方法**

使用的技术包括：统一多视图图构造、基于信息最大化（如 DGI）的自监督预训练、专家对齐映射、类别级注意力投票以及正交约束。

**📊 数据集**

实验数据集涵盖同质图 Cora、CiteSeer、PubMed、Photo、Computer，以及异质图 ACM、DBLP、Aminer、Freebase，评估任务为 3/5-shot 节点分类。

**📈 对比分析**

与同质图预训练方法（DGI、GRACE、GraphMAE 等）和异质图预训练方法（DMGI、HeCo、HGMAE 等）对比，GPH^2 在 3/5-shot 节点分类任务上平均提升约 4–6%（同质）和 5–6%（异质），并在多数据集上表现出最优且鲁棒的性能。

**⚠️ 局限性**

局限性包括：对预定义元路径和视图数敏感，专家编码训练需要并行计算且对极大图规模扩展受限；在某些同质任务中加入异质图预训练时收益不明显。

---

## 358. Hierarchical Reinforcement Learning for Cooperative Air-Ground Delivery in Urban System

**arXiv ID:** 2602.12913 | [PDF](https://arxiv.org/pdf/2602.12913v1)

**作者:** Songxin Lei `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5630 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种层次强化学习框架 HRL4AG，用于协同空地配送系统中的订单调度。

**💡 创新点**

创新点在于：1) 通过模式专属工作者（Worker）分别编码 UAV 与地面车队的异质动态；2) 用高层管理器（Manager）将庞大联合动作空间分解为两层决策，显著降低搜索空间；3) 设计内部奖励机制与轨迹驱动奖励塑造，解决稀疏奖励下的信用分配问题。

**🔧 技术方法**

核心技术包括：层次强化学习（Manager-Worker 架构）、Transformer 编码器实现异质状态编码、DDPG 算法训练、内部奖励与稀疏奖励塑造、动作掩码与贪婪匹配。

**📊 数据集**

使用真实物流数据集：上海（约 20,664 单据）与成都（约 96,376 单据）两个地区的订单与道路网络。

**📈 对比分析**

与传统启发式（B&B、Greedy、HGR）和先进 RL（D2SN、DECO、GRC、DDPG-JOTOCC）基线比较；在两大数据集上，HRL4AG 在订单接单率、送达率方面提升约 18-28%，同时推理时间降低 80 倍，执行时间从 11-20 秒降至 <0.9 秒。

**⚠️ 局限性**

限制包括：1) 仍假设道路网络与飞行能耗为确定性，未考虑天气或随机交通拥堵；2) 仅评估两种运输模式，未探讨多模式或人机协同；3) 训练需要较大 GPU 资源，且对超参数（如奖励权重）敏感。

---

## 359. GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction

**arXiv ID:** 2602.12796 | [PDF](https://arxiv.org/pdf/2602.12796v1)

**作者:** Xiao Ren `[一作]` (Southern University Science and Technology), He Kong `[通讯]` (Southern University Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种结合单视角自适应子区域约束和多视角几何一致性约束的高精度3D高斯弹涂层（Gaussian Splatting）重建框架——GSM-GS，旨在解决传统高斯点云在纹理稀疏和高频细节捕捉方面的不足；

**💡 创新点**

创新点包括：1）基于图像梯度的纹理稠密/稀疏子区域划分，并对每个子区域引入双分支权重约束，提升纹理丰富与纹理稀疏区域的几何细节重建；2）引入权重引导的动态采样与跨视角几何正则化，构建3D空间内连续的几何相关模型，从而显著改善多视角一致性和表面细节；

**🔧 技术方法**

采用的技术主要有：高斯弹涂层（3D Gaussian Splatting）表征；单视角深度差异权重与图像梯度自适应分支约束；多视角点云对齐与PCA求取法线、曲率加权相似度损失；整体训练使用PyTorch/CUDA实现，结合光照渲染损失与几何损失；

**📊 数据集**

使用的公开数据集包括DTU、Mip-NeRF360、Tanks and Temples、LLFF等，涵盖室内外、纹理丰富与稀疏、反射与阴影等多种复杂场景；

**📈 对比分析**

通过与3DGS、2DGS、GOF、RaDe-GS、PGSR等先进方法的定量对比，GSM-GS在Chamfer距离、PSNR、SSIM、LPIPS、F1-score等指标上均获得领先或竞争性结果，且训练时间仅约0.45小时，保持高效；

**⚠️ 局限性**

局限性包括：对透明材质、高反射表面以及极薄结构的几何与外观分离仍有困难，受复杂光照影响，且跨视角正则化对极端遮挡场景的鲁棒性尚需进一步提升。

---

## 360. Chimera: Neuro-Symbolic Attention Primitives for Trustworthy Dataplane Intelligence

**arXiv ID:** 2602.12851 | [PDF](https://arxiv.org/pdf/2602.12851v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11894 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在可编程数据平面上实现注意力神经网络与符号约束的联合推理，构建Chimera框架实现可信线速推理。

**💡 创新点**

创新点是将Transformer注意力线性化并映射为Partition/Map/SumReduce原语，采用两层键选择层与级联融合，兼顾神经表达与硬规则，同时实现了双时钟更新协议保障稳定。

**🔧 技术方法**

技术包括核化线性注意力、量化与增量聚合、TCAM/ SRAM 双层键索引、级联软硬融合、两时钟控制与P4编译映射。

**📊 数据集**

数据集：PeerRush、CICIOT2022、ISCXVPN2016以及用于AutoEncoder训练的流量数据。

**📈 对比分析**

与树、RNN、MLP、CNN等基准对比，Chimera在宏F1、误报率、吞吐量、延迟上优于或匹配软平台，同时保持低每流状态与资源占用。

**⚠️ 局限性**

限制：仍受限于每流SRAM/TCAM容量，键选择层固定可能缺失长距离关联，量化导致细粒度精度下降，双时钟协议对控制平面更新时延敏感。

---

## 361. Unleashing MLLMs on the Edge: A Unified Framework for Cross-Modal ReID via Adaptive SVD Distillation

**arXiv ID:** 2602.12936 | [PDF](https://arxiv.org/pdf/2602.12936v1)

**作者:** Hongbo Jiang `[一作]`, Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4220 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MLLMEmbed-ReID 框架，将多模态大语言模型迁移为统一的跨模态人像重识别云模型，并通过低秩知识蒸馏将其知识迁移到轻量边缘模型。

**💡 创新点**

① 通过指令提示和低秩 LoRA‑SFT 在 Qwen2‑VL 上实现统一的跨模态特征空间；② 发现教师特征低秩特性并提出基于主成分映射与特征关系损失的蒸馏方法；③ 在云‑边缘协同架构下实现一次性部署所有四种模态的 CM‑ReID。

**🔧 技术方法**

使用 Qwen2‑VL 作为教师，CLIP ViT‑L/14 作为学生；LoRA 高阶微调、三重损失（ID+Triplet+SDM）、SVD 低秩分析、主成分映射损失 PCM、特征关系损失 FR 以及余弦对齐损失。

**📊 数据集**

在扩展后的 QrCM‑ReID 数据集上评测，该数据集由 CUHK‑PEDES、ICFG‑PEDES、RSTPReid 三个文本 ReID 数据集并生成 Sketch、Infrared 模态组成。

**📈 对比分析**

与现有单模态、统一框架和 MLLM 基线对比，在三大跨模态任务（IR→RGB、T→RGB、S→RGB）上均达到或超过 SOTA，云模型 mAP 高达 90.28%（IR→RGB）等，边缘模型在同一任务上亦实现近 SOTA。

**⚠️ 局限性**

对低秩蒸馏的超参数依赖较高，PCM 与 FR 组合时易出现冲突；且模型对长文本提示长度敏感，极端光照或遮挡下的鲁棒性仍待进一步验证。

---

## 362. Intent-Driven Smart Manufacturing Integrating Knowledge Graphs and Large Language Models

**arXiv ID:** 2602.12419 | [PDF](https://arxiv.org/pdf/2602.12419v1)

**作者:** Takoua Jradi `[一作]` (École de technologie supérieure), Symeon Papavassiliou `[通讯]` (National Technical University of Athens)

**通讯引用:** 8538 | [OpenAlex ID](https://openalex.org/A5035126390)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个将指令调优的 LLM 与知识图谱相结合的框架，实现自然语言意图到可执行 JSON 需求模型的转换。

**💡 创新点**

创新点在于将 LLM 生成的结构化需求模型与基于 ISA‑95 的 Neo4j 知识图谱动态对齐，实现意图到制造过程的语义映射。

**🔧 技术方法**

使用指令调优的大语言模型 Mistral‑7B‑Instruct‑V02、Neo4j 图数据库以及 ISA‑95 标准本体。

**📊 数据集**

在自建的域内数据集上对 Mistral 进行微调，数据集包含工厂运营意图与对应 JSON 模型。

**📈 对比分析**

通过与零样本（zero‑shot）和 3‑shot 基线进行对比，模型在精确匹配准确率 89.33% 和整体准确率 97.27% 上显著优于基线。

**⚠️ 局限性**

局限性包括仅依赖 LLM 进行推理，缺乏实时 KG 更新；实验范围受限于单一行业场景，未验证跨域可迁移性。

---

## 363. LongStream: Long-Sequence Streaming Autoregressive Visual Geometry

**arXiv ID:** 2602.13172 | [PDF](https://arxiv.org/pdf/2602.13172v1)

**作者:** Chong Cheng `[一作]` (Hong Kong University of Science and Technology), Hao Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 41183 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 LongStream，一种基于因果 Transformer 的 gauge‑decoupled 流式自回归几何模型，能够在千米级长序列中实现实时、metric‑scale 的 3D 重建。

**💡 创新点**

创新点包括：① 去除首帧锚定，采用 keyframe‑relative pose 预测，解决长期外推失真；② 引入正交尺度学习，解耦 SE(3) 与 Sim(3)，消除尺度漂移；③ 开发 Cache‑Consistent Training 与周期性缓存刷新机制，消除 attention sink 与 KV‑cache 污染，保证长序列稳定性。

**🔧 技术方法**

核心技术：ViT 视觉编码器 + 因果 Transformer；相对位姿、深度、点图、独立尺度四个头；正交尺度学习；Cache‑Consistent Training (CCT) 与周期性缓存刷新；相对 Pose、深度、几何、尺度损失等。

**📊 数据集**

训练数据集涵盖多域：Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Replica, MVS‑Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, Objaverse, Spring, Waymo Open；测试集包括 KITTI, vKITTI, Waymo, TUM‑RGBD, Oxford Spires, 7Scenes, ETH3D 等。

**📈 对比分析**

与离线 Transformer、SLAM 以及流式基线（STream3R, StreamVGGT, FastVGGT, CUT3R, TTT3R, MASt3R‑SLAM, VGGT‑SLAM）在多种室内外基准上对比。LongStream 在所有评测集上实现最低 ATE、最高 F1，且保持 18 FPS 实时推理，支持千米级长序列且无 OOM。

**⚠️ 局限性**

局限性：假设场景基本静止；依赖手工设定的 keyframe 调度；在极长窗口下点图一致性略有下降；缺乏循环闭合机制，对动态场景仍存在挑战。

---

## 364. Paid to Look Like Truth: The Prevalence and Dark Patterns of Advertorials in News Outlets

**arXiv ID:** 2602.12810 | [PDF](https://arxiv.org/pdf/2602.12810v1)

**作者:** Emmanouil Papadogiannakis `[一作]` (Foundation for Research and Technology and University of Crete), Evangelos Markatos `[通讯]` (Foundation for Research and Technology and University of Crete)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对新闻网站中的广告内容进行大规模系统化研究，构建了自动检测广告文章（Advertorial）的方法，并公开发布了检测到的广告域名列表。

**💡 创新点**

创新点在于首次将结构化特征（HTML标签、评论区、免责声明等）与内容层面的行为与声誉特征（安全引擎、第三方交互、域名注册信息）结合，形成两阶段过滤器，能够在海量广告 URL 中准确识别出“问题广告文章”。

**🔧 技术方法**

技术主要包括：①基于 HTML 结构的模式匹配（定位评论区、免责声明、作者信息等元素）；②使用 19 个安全/分析引擎的多源特征，经过随机森林特征选择得到 24 个重要特征；③机器学习分类（随机森林）实现对域名的可疑/正常判定；④对文本进行长度、对比度、词汇多样性等可解释性指标的统计。

**📊 数据集**

数据集包括：186,124 条广告 URL（5 个月内收集），约 11,000 个广告落地页；手工标注的 1,000 条广告（820 正常、157 付费文章）用于训练/评估；域名注册信息（213 条），以及 300 条广告、300 条普通广告、300 条新闻文章的文本样本用于内容对比；公开的 103 条免责声明关键词表。

**📈 对比分析**

检测方法在训练集上的准确率与 F1 约 93%；在未见过的 1,000 条广告上，整体准确率 87.4%，广告类的 F1 为 0.749，表明虽然精度高但召回率略低，主要受两阶段共识阈值影响。与现有基于规则或单一特征的检测方法相比，本研究通过结合结构与内容特征显著提升了检测的可靠性。

**⚠️ 局限性**

局限性：①采用两阶段共识阈值导致召回率受限，可能漏检部分精细伪装的广告；②数据来源仅限新闻网站，无法覆盖其他媒体平台；③免责声明关键词仅覆盖 7 种语言，其他语言的广告可能被忽略；④手工标注样本规模有限，难以覆盖所有广告变体；⑤对动态生成内容的检测仍有挑战。

---

## 365. Secrecy and Verifiability: An Introduction to Electronic Voting

**arXiv ID:** 2602.12398 | [PDF](https://arxiv.org/pdf/2602.12398v1)

**作者:** Paul Keeler `[一作]` (University of Melbourne), Ben Smyth `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理了电子投票体系的核心安全需求——投票保密（ballot secrecy）与可验证性（verifiability），并阐述了两者在设计上往往互相冲突的本质。作者基于现代密码学的游戏式安全定义（game‑based cryptography），给出了投票保密和可验证性的形式化模型，说明了在公钥（尤其是可同态与非变形加密）与阈值加密、混淆网络（mixnet）等工具下如何兼顾这两大属性。

**💡 创新点**

创新点在于：①将投票保密与可验证性统一映射到游戏式安全框架，提出了新的“平衡”设计思路；②通过对投票方案的语法与语义的严格定义（如四个核心算法与其交互流程），为后续正式证明提供了可复制的基线；③给出具体的游戏化定义（如IND‑PA0、NM‑CPA、完整性与可靠性游戏），并解释了它们之间的逻辑关系。

**🔧 技术方法**

使用的技术包括：非对称加密与可同态加密（RSA、ElGamal、Paillier等）；阈值加密（threshold decryption）；混淆网络（mixnet）及其零知识证明；游戏式安全模型（IND‑CPA、IND‑CCA、IND‑PA0、NM‑CPA等）；零知识证明与Fiat‑Shamir变换；随机Oracle模型。

**📊 数据集**

本文为理论性综述与教学性质，未使用具体实验数据集，而是以符号模型与抽象投票方案（如Smyth‑Frink‑Clarkson 架构）为例进行阐述。

**📈 对比分析**

由于缺乏实验实施，本文并未给出具体性能数值或对比实验；其贡献主要体现在理论证明与安全模型的严谨性上。

**⚠️ 局限性**

局限性包括：①假设投票过程中的公告板（bulletin board）可信且仅可追加；②阈值加密与混淆网络实现复杂度高，实际部署难度大；③未讨论大规模选举（百万级投票）的效率与可扩展性；④缺乏对量子后安全方案的深入分析（仅简述）。

---

## 366. MAUNet-Light: A Concise MAUNet Architecture for Bias Correction and Downscaling of Precipitation Estimates

**arXiv ID:** 2602.12980 | [PDF](https://arxiv.org/pdf/2602.12980v1)

**作者:** Sumanta Chandra Mishra Sharma `[一作]` (Indian Institute of Technology Kharagpur), Auroop Ratan Ganguly `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了MAUNet-Light，一种轻量级神经网络，用于卫星降水估计的偏差校正和空间下采样；通过教师‑学生知识蒸馏与知识精炼，将原MAUNet压缩到约60%参数量，并保持近似性能。

**💡 创新点**

创新点在于：①将MAUNet压缩为轻量化网络；②通过教师‑学生框架实现知识迁移与精炼，显著降低计算与存储成本；③首次将压缩网络应用于降水偏差校正与下采样任务。

**🔧 技术方法**

技术方法包括深度学习架构MAUNet与MAUNet-Light、最大平均池化与上采样单元、教师‑学生蒸馏、MSE损失、Adam优化器，以及RMSE、PSNR、MSSIM、相关系数、KL散度等评估指标。

**📊 数据集**

使用TRMM_3B42_Daily卫星降水数据与印度气象局(IMD)格点降水数据，覆盖印度大陆，时间范围分别为1998‑2019（偏差校正）和1990‑2012训练/2013‑2019测试（下采样）。

**📈 对比分析**

与双线性/双三次插值、EDSR、SRDRN及统计方法（QM、QDM）对比，MAUNet-Light_KR在RMSE≈11.4、相关系数≈0.73、PSNR≈37.1等指标上几乎匹配原MAUNet，优于其他方法。

**⚠️ 局限性**

局限性包括：对极端降水事件的低估（缺乏极端事件的高分辨率再现），以及在极端指标（连续干旱日、强降水日、年最大日降水）上仍落后于观测数据。

---

## 367. Semantic Chunking and the Entropy of Natural Language

**arXiv ID:** 2602.13194 | [PDF](https://arxiv.org/pdf/2602.13194v1)

**作者:** Weishun Zhong `[一作]` (Institute for Advanced Study), Misha Tsodyks `[通讯]` (Institute for Advanced Study)

**通讯引用:** 17128 | [OpenAlex ID](https://openalex.org/A5030973339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于层次语义分块的统计模型，通过递归将文本拆分为语义树来估计自然语言的熵率；

**💡 创新点**

创新点在于将语义树结构与熵率关联，构建可解析的随机K叉树族，并用该模型首次从第一性原理解释英文熵率及其随语义复杂度变化的规律；

**🔧 技术方法**

采用大规模语言模型（LLM）进行语义分块与概率估计，结合随机树理论、KL散度评估、以及对数-正态缩放分析；

**📊 数据集**

使用公开语料包括儿童读物、叙事小说、arXiv摘要、现代诗歌等多类文本进行实验；

**📈 对比分析**

通过将LLM的困惑度（perplexity）与模型计算的树结构熵率对比，发现两者在不同语料上高度一致，且K*（最大分支数）与语料复杂度相符，表现优异；

**⚠️ 局限性**

局限性在于模型仅在语料级别上匹配，单一K参数可能不足以捕捉文本个体差异，且依赖于LLM的质量与分块策略。

---

## 368. Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision

**arXiv ID:** 2602.13195 | [PDF](https://arxiv.org/pdf/2602.13195v1)

**作者:** Aadarsh Sahoo `[一作]` (California Institute of Technology), Georgia Gkioxari `[通讯]` (California Institute of Technology)

**通讯引用:** 38511 | [OpenAlex ID](https://openalex.org/A5014407395)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种对抽象对话式语义的图像分割方法——Conversational Image Segmentation (CIS)，同时构建了1,687例人类验证的数据集并开发了自动化数据生成引擎，生成106K对话式prompt‑mask对；

**💡 创新点**

创新点在于将对话式语义（实体、空间布局、关系事件、功能与物理安全）纳入分割任务，利用VLM驱动的生成与多阶段验证产生高质量对话式训练数据，并通过单通道模型在多概念上实现基线突破；

**🔧 技术方法**

技术包括SAM2预训练分割器、Qwen‑2.5‑VL‑3B作为文本编码器、LoRA微调、生成‑验证循环的VLM（Gemini‑2.5‑Flash）以及两阶段训练课程；

**📊 数据集**

使用COCO验证集（人类标注与SAM‑seeded）作为CIS基准；COCO训练集、SA‑1B用于数据合成；同时在RefCOCO/+/g、ReasonSeg等标准分割基准上评测；

**📈 对比分析**

与LISA、UniLSeg、EVF‑SAM、Seg‑Zero等基线比较，CIS模型在CIS基准上达70.8%（3B）/72.4%（7B）的gIoU，超过Seg‑Zero 69.2%；在RefCOCO上78.4%，在ReasonSeg 57.0%（7B）表现优于多模型；

**⚠️ 局限性**

局限在对话式概念覆盖仍有限（仅5类）、对极端物理推理的细粒度判定可能不足，且模型在多轮交互或更复杂的语言表达上尚未测试。

---

## 369. Rational Neural Networks have Expressivity Advantages

**arXiv ID:** 2602.12390 | [PDF](https://arxiv.org/pdf/2602.12390v1)

**作者:** Maosen Tang `[一作]` (Cornell University), Alex Townsend `[通讯]` (Cornell University)

**通讯引用:** 2049 | [OpenAlex ID](https://openalex.org/A5003110773)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并实验了可训练低阶有理激活函数在深度网络中的表达力与参数效率，证明其在理论上能以指数级更少参数逼近常用平滑激活，并在多种视觉与离线强化学习任务中实现更高或相当的性能。

**💡 创新点**

创新点在于：① 用有理函数构造可训练激活，理论上展示了有理网络与平滑激活网络在逼近误差与参数规模上的对数-对数与对数分离；② 将这一理论扩展到门控与Transformer结构；③ 实证验证有理激活在VGG、ViT、CIFAR-10、Tiny ImageNet以及MuJoCo离线RL中的优势。

**🔧 技术方法**

技术上采用低阶Pade/ERA形式的可训练有理激活，结合深度网络中的逐层组合；理论推导基于Zolotarev函数、算术几何平均方法及复分析中的极点距离；实验使用标准PyTorch实现，训练框架包括LabelSmoothing、Mixup、GroupNorm、EMA、无Norm变体等。

**📊 数据集**

使用的数据集包括CIFAR-10、Tiny ImageNet、Minari的MuJoCo中等规模离线RL数据集，以及Gymnasium v5的评估环境。

**📈 对比分析**

与传统固定激活（ReLU、LeakyReLU、GELU、Swish、SiLU等）在相同架构、优化器与训练预算下进行对比；在CIFAR-10和Tiny ImageNet上有理激活往往取得更高Top‑1准确率，且收敛更快；在离线RL任务中，在大多数场景下有理激活的归一化分数超过或匹配最佳平滑激活，并在某些任务中领先明显。

**⚠️ 局限性**

局限性包括实验规模受限于单个RTX 4090或4070 GPU，未覆盖大规模语言模型或更广泛的RL/Transformer配置；实现依赖已有框架，未优化原生CUDA核；缺乏大样本、多种超参数、LLM等场景的验证，未来需在更大规模与多样任务上进一步测试。

---

## 370. How Swarms Differ: Challenges in Collective Behaviour Comparison

**arXiv ID:** 2602.13016 | [PDF](https://arxiv.org/pdf/2602.13016v1)

**作者:** André Fialho Jesus `[一作]` (University of Konstanz), Jonas Kuckling `[通讯]` (University of Konstanz)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5021453451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了四种现有群体行为特征集在相似性评估和可解释分类中的表现，系统比较了不同特征集与多种相似性度量（余弦、欧氏、组合状态计数、采样平均状态）的交互作用；

**💡 创新点**

首次系统性评估特征集与相似性度量的交互效应，发现常规距离度量不足以区分群体行为，并提出基于自组织映射的可解释分类框架；

**🔧 技术方法**

使用特征提取、余弦/欧氏/组合状态计数/采样平均状态相似性度量以及自组织映射（SOM）进行分类；

**📊 数据集**

利用自行生成的六种群体行为（Reynolds、Vicsek、聚合、分散、弹道与布朗运动）在不同 swarm 大小和边界条件下的仿真数据，数据可在 kondata 与 GitHub 获取；

**📈 对比分析**

对 50 次独立仿真计算相似性得分并报告平均值与标准差；组合状态计数等专为群体设计的度量显示更高区分度；SOM 分类最高训练准确率约 59%，测试约 49%，整体低于 50%；

**⚠️ 局限性**

特征集在不同情境下泛化差，常规距离度量无法准确捕捉行为差异；SOM 分类准确率低，难以区分相似行为；缺乏更鲁棒的特征与度量方法。

---

## 371. A Calibrated Memorization Index (MI) for Detecting Training Data Leakage in Generative MRI Models

**arXiv ID:** 2602.13066 | [PDF](https://arxiv.org/pdf/2602.13066v1)

**作者:** Yash Deo `[一作]`, Ibrahim Habli `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种针对医学图像生成模型的校准化、多尺度记忆化检测指标，能够识别训练集复制的图像。

**💡 创新点**

创新点在于使用医学领域预训练的 MRI-CORE ViT-B 提取多层特征，采用 ZCA 白化后计算最近邻相似度并通过经验空分布校准得到可跨数据集解释的 Overfit/Novelty Index。

**🔧 技术方法**

使用的技术包括多层 Transformer 特征提取、ZCA 白化、几何平均聚合、bootstrapping 经验分布校准和对每个样本的最近邻检索。

**📊 数据集**

在 BRATS（脑肿瘤）、膝关节和脊柱 MRI 三个公开数据集上进行评估。

**📈 对比分析**

与 CT‑score、FID/MMD、AuthPct、Vendi 等方法相比，MI/ONI 在复制率增加时保持线性上升、抗噪/旋转/翻转鲁棒、跨数据集方差显著降低，ROC‑AUC 接近 1，平均精度亦高。

**⚠️ 局限性**

对小幅几何变换（旋转、翻转）敏感，需要进一步改进特征对齐或在经验分布中加入几何扰动；方法依赖于医学专用基础模型，尚未在非 MRI 模态充分验证。

---

## 372. Geometric Manifold Rectification for Imbalanced Learning

**arXiv ID:** 2602.13045 | [PDF](https://arxiv.org/pdf/2602.13045v1)

**作者:** Xubin Wang `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**通讯引用:** 11814 | [OpenAlex ID](https://openalex.org/A5051803761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现Geometric Manifold Rectification（GMR）框架，利用几何置信度估计和不对称清理来对不平衡数据进行预处理；

**💡 创新点**

创新点在于：①使用逆距离加权的kNN置信度估计捕捉局部几何结构；②自适应度量切换（低维欧氏，高维余弦）；③对多数类严格，对少数类保守的异构清理策略，并设置安全阈值防止少数类被误删；④提供理论分析证明不对称清理能上移后验概率。

**🔧 技术方法**

技术方法包括kNN搜索、逆距离权重、欧氏/余弦距离切换、阈值化清理、少数类移除上限控制以及对实验的AUPRC、均衡准确率评估。

**📊 数据集**

使用27个传统不平衡基准数据集、5个大规模TabDDPM数据集（如581K、284K等）以及CIFAR‑100‑LT图像数据集。

**📈 对比分析**

与18种常见重采样基线和“无采样”做对比，使用7种分类器（LR、SVM、DT、RF、GBM、XGBoost、KNN）进行平均排名；在TabDDPM实验中，对原始数据与预处理后数据分别训练，GMR在多数噪声/重叠数据上提升AUPRC；在CIFAR‑100‑LT中，GMR‑Feature在均衡准确率上优于CB，但略低于CE。总体而言，GMR在大多数分类器上获得最低平均排名（4.22），并在6/7分类器上取得第一名。

**⚠️ 局限性**

局限性包括：对极端不平衡（IR>100）效果可能受限；依赖足够样本以获得稳定邻域估计；目前仅支持二分类，未包含多分类扩展；超参数设为固定默认值，缺乏自适应调优；未进行显著性检验。

---

## 373. An Autonomous, End-to-End, Convex-Based Framework for Close-Range Rendezvous Trajectory Design and Guidance with Hardware Testbed Validation

**arXiv ID:** 2602.12421 | [PDF](https://arxiv.org/pdf/2602.12421v1)

**作者:** Minduli C. Wijayatunga `[一作]`, Xiaofeng Wu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了一套基于凸优化的闭近距对接轨迹规划与指令生成框架 CORTEX。

**💡 创新点**

创新点包括：① 引入日照感知的阶段调度；② 单迭代凸自适应跟踪；③ 基于 YOLO 的深度感知关键点检测与 EKF 状态融合；④ 自动参考重生成与安全回退逻辑。

**🔧 技术方法**

技术手段：凸优化（SOCP）、深度学习 YOLO、扩展卡尔曼滤波、Basilisk 高保真仿真、平面空气轴实验台、MILP 扳机电控、RCS 与主推进器结合。

**📊 数据集**

感知数据集未公开，使用内部训练的 YOLO 模型和实验/仿真环境提供的图像进行验证。

**📈 对比分析**

通过 100 次 Monte‑Carlo 仿真（低误差）与 18 次硬件实验（10 组正常 + 8 组异常）对比，低误差下终端定位误差约 3–6 mm、速度误差约 0.03–0.6 mm/s；高误差下通过重规划/回退后终端误差 ≤32 mm、0.5 mm/s，单次求解耗时约 38 ms，满足实时性。

**⚠️ 局限性**

局限性：喷流阻碍约束采用固定姿态假设；非凸约束（如避碰球、喷流）未直接纳入追踪阶段；实验中未完全闭环视觉感知；对极端扰动的鲁棒性和燃料效率改进仍待研究。

---

## 374. TriGen: NPU Architecture for End-to-End Acceleration of Large Language Models based on SW-HW Co-Design

**arXiv ID:** 2602.12962 | [PDF](https://arxiv.org/pdf/2602.12962v1)

**作者:** Jonghun Lee `[一作]` (Gachon University), Heonjae Ha `[通讯]` (Samsung Electronics)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了TriGen NPU架构及其软件硬件协同方案，实现对大型语言模型的端到端推理加速。

**💡 创新点**

创新点包括首次在资源受限的设备上引入MXINT8低精度数制、使用查找表（LUT）加速非线性运算、设计资源感知的张量调度与多NPU同步机制，显著提升性能与降低内存传输。

**🔧 技术方法**

采用的技术包括MXINT8数制、FI32中间格式、LUT非线性加速、数据流与张量分块调度、QKV批处理、转置融合、掩码融合、以及多核心NPU的轻量级同步。

**📊 数据集**

评测使用Llama‑2‑7B、Llama‑3.8B、Llama‑3.2‑3B、OPT‑1.3B、OPT‑2.7B模型，数据集涵盖WikiText‑2、C4、PTB。

**📈 对比分析**

通过与基线INT16/UINT4 NPU的对比，结合多项优化组合，TriGen平均提升2.73倍推理速度，内存传输量降低52%，LUT提升约27%，MXINT8进一步将内存流量减半。

**⚠️ 局限性**

局限性在于极大序列长度和多NPU时受DRAM带宽限制；MXINT8在极低精度下仍需平衡精度与功耗，且对极低功耗设备的验证尚有限。

---

## 375. Characterize LSM-tree Compaction Performance via On-Device LLM Inference

**arXiv ID:** 2602.12669 | [PDF](https://arxiv.org/pdf/2602.12669v1)

**作者:** Jiabiao Ding `[一作]` (Xiamen University), Chun Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6094 | [OpenAlex ID](https://openalex.org/A5101441768)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文研究了在LSM树存储引擎中使用本地小型LLM进行实时压缩参数调优的方法

**💡 创新点**

创新点在于对不同规模LLM在推理延迟与调优效果之间的权衡进行系统实验，并探索了低延迟小模型在压缩参数优化中的可行性

**🔧 技术方法**

主要技术包括LLM推理（如Pangu-7B、Qwen3-8B）、自定义提示构造、基于RocksDB v8.8.1的db_bench工作负载及闭环调优框架

**📊 数据集**

实验使用RocksDB v8.8.1与db_bench随机写/读混合工作负载，评估不同LLM模型的调优效果与推理时延

**📈 对比分析**

与传统贝叶斯/强化学习自动调优相比，LLM方法在小模型上能在毫秒级推理完成，调优后写入吞吐率提升约10-20%，但仍不及大模型在最佳配置下的提升

**⚠️ 局限性**

主要限制是小型LLM在推理时容易产生格式错误、缺失参数或循环输出，导致调优不稳定且对参数探索范围有限

---

## 376. Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting

**arXiv ID:** 2602.12540 | [PDF](https://arxiv.org/pdf/2602.12540v1)

**作者:** Haoran Zhu `[一作]` (New York University), Anna Choromanska `[通讯]` (New York University)

**通讯引用:** 2558 | [OpenAlex ID](https://openalex.org/A5006452373)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于联合嵌入预测架构(JEPA)的自监督多帧LiDAR世界模型AD-LiST-JEPA，用于预测车辆周围环境的时空演化；

**💡 创新点**

创新点在于：①采用基于群组BEV引导的掩码策略，将车体与外部环境分离，解决多帧数据中的信息泄漏问题；②通过方差正则或SIGReg等正则化手段避免表示坍塌；③将单帧JEPA扩展为多帧时空预测框架；

**🔧 技术方法**

技术包括自监督预训练的JEPA框架、3D稀疏卷积网络、BEV网格掩码、方差正则化与SIGReg正则化、以及轻量级下游OCC任务解码器；

**📊 数据集**

使用Waymo公开LiDAR数据集进行预训练与下游任务评估；

**📈 对比分析**

在下游占用完成与预测(OCF)任务中，对比从头训练与预训练模型，预训练模型在IoU_full和IoU_close指标上均略优，且SIGReg正则化显著提升性能；

**⚠️ 局限性**

局限性包括：仅在中小规模数据集上验证，网络架构相对简单，缺乏大规模实验与跨数据集泛化验证，且尚未探究更复杂的多模态融合方法。

---

## 377. Resource-Efficient Gesture Recognition through Convexified Attention

**arXiv ID:** 2602.13030 | [PDF](https://arxiv.org/pdf/2602.13030v1)

**作者:** Daniel Schwartz `[一作]` (Drexel University), Ali Shokoufandeh `[通讯]` (Drexel University)

**通讯引用:** 3568 | [OpenAlex ID](https://openalex.org/A5058258391)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在可穿戴电子纺织接口中实现了一种凸化注意力机制，用于低功耗设备上的触摸手势识别。

**💡 创新点**

创新点在于将注意力权重通过非扩张欧氏投影到概率单纯形来替代传统softmax，从而保持全局凸性；并结合多分类铰链损失和核范数正则化，显著减少参数量并保证收敛。

**🔧 技术方法**

使用凸化卷积神经网络（CCNN）、随机傅里叶特征（RFF）、非扩张欧氏投影、核范数投影和多分类铰链损失，全部实现于Arduino Nano 33 BLE上。

**📊 数据集**

数据集为单位实验室参与者在4×4电容纺织传感器上收集的1000个样本，包含四类点触和四类滑动手势（每类100个样本，训练/验证/测试划分为10折交叉验证或60/20/20比例）。

**📈 对比分析**

与传统CNN、MobileNet、SqueezeNet及Convex ViT等基线相比，凸化注意力模型在点触和滑动手势上均实现100%准确率，参数量仅为120/360（相比传统CNN的1,316/3,972约低97%），推理时间为290–296μs，显著低于其他模型。

**⚠️ 局限性**

局限性包括仅在单一用户和受控实验室环境下验证；手势词汇有限（仅点触与滑动）；未测试多用户、不同环境、持续识别以及传感器长期耐用性。

---

## 378. Monte Carlo Tree Search with Reasoning Path Refinement for Small Language Models in Conversational Text-to-NoSQL

**arXiv ID:** 2602.12574 | [PDF](https://arxiv.org/pdf/2602.12574v1)

**作者:** Xubang Xiong `[一作]` (Hong Kong University of Science and Technology), Yuanfeng Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 414 | [OpenAlex ID](https://openalex.org/A5077688051)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了会话式文本到NoSQL查询生成任务，利用小型语言模型通过搜索框架逐步生成并组合NoSQL查询；

**💡 创新点**

创新点在于将NoSQL查询生成建模为搜索问题，使用阶段增强的Chain-of-Thought与Monte Carlo Tree Search进行数据增强与自我训练，形成端到端的MCTS驱动推理框架；

**🔧 技术方法**

核心技术包括Monte Carlo Tree Search（MCTS）与奖励模型驱动的路径采样与改进、阶段增强的CoT生成、三阶段监督微调（SFT）与多轮自训练、以及测试时MCTS扩展的推理；

**📊 数据集**

构建了跨域对话式NoSQL数据集CoNoSQL，包含2000+对话、9000+轮次、150+不同NoSQL数据库，支持评估；

**📈 对比分析**

与多种基线（零样本、少样本、RAG、STaR-SFT、ReAct、Plan-and-Solve、DeepSeek-R1、GPT‑4o）对比，在执行值匹配（EVM）准确率上提升最多7.93%，并超过大型模型的表现；

**⚠️ 局限性**

限制主要在于：需要较多的MCTS计算开销；对非常复杂或未见过的NoSQL模式仍可能出现错误；以及对多轮对话的可扩展性和对不同NoSQL数据库特性的通用性尚需进一步验证。

---

## 379. Deep Doubly Debiased Longitudinal Effect Estimation with ICE G-Computation

**arXiv ID:** 2602.12379 | [PDF](https://arxiv.org/pdf/2602.12379v1)

**作者:** Wenxin Chen `[一作]` (Cornell University), Fei Wang `[通讯]` (Cornell University)

**通讯引用:** 21766 | [OpenAlex ID](https://openalex.org/A5100455750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种两阶段深度ICE G‑computation框架D^3‑Net，用于解决随时间变化的混杂导致的治疗效果估计误差累积问题。

**💡 创新点**

创新点在于：①引入Sequential Doubly Robust（SDR）伪目标在训练阶段实现递归去偏，②使用目标网络和协助监督（协变量模拟器）稳定共享表示，③在推断阶段再次应用LTMLE实现最终去偏，使方法兼具训练稳定性和统计效率。

**🔧 技术方法**

技术包括：Transformer多任务网络（共享表征），SDR伪目标回归，目标网络（Polyak平均），协变量模拟器辅助监督，最终的LTMLE再去偏步骤。

**📊 数据集**

使用MIMIC‑III（半合成数据集）和MIMIC‑IV（真实ICU患者）作为实验和真实案例的数据集。

**📈 对比分析**

与传统ICE G‑computation、LTMLE、DeepACE、DeepLTMLE等基线进行比较。实验显示D^3‑Net在多种时间长度、混杂强度和对照治疗序列下均显著降低了偏差和方差，且鲁棒性更强。

**⚠️ 局限性**

限制包括：依赖观测性数据的标准因果假设，可能存在未测量混杂；对长时间步和极端权重的鲁棒性仍需进一步评估；实现复杂度较高，需要精细的超参数调优。

---

## 380. UniManip: General-Purpose Zero-Shot Robotic Manipulation with Agentic Operational Graph

**arXiv ID:** 2602.13086 | [PDF](https://arxiv.org/pdf/2602.13086v1)

**作者:** Haichao Liu `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3124 | [OpenAlex ID](https://openalex.org/A5100389366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种通用的零样本机器人操作框架UniManip，能够在开放世界环境下直接从自然语言指令完成多种长时程、多交互的操作任务。

**💡 创新点**

创新点在于构建双层Agentic操作图（AOG），将高层语义规划与低层动态状态记忆统一；通过闭环反射和恢复机制实现在线错误诊断与自适应重规划；以及单视角安全占据网格与松弛IK的结合，显著提升零样本鲁棒性。

**🔧 技术方法**

核心技术包括：单眼RGB‑D视觉感知、基于VLM的命令解析与图生成、对象中心的SOSG状态建模、基于安全感知的保守占据空间构建、ESDF路径规划、松弛逆运动学求解以及基于AOG的自我反思与恢复。

**📊 数据集**

实验采用Galaxea A1 6‑DoF机械臂和RealSense D435摄像头在多样化家庭物体环境中收集的数据，并在移动平台（Realman RM65）上进行交叉实现；基准数据集包含九种抓取放置任务与多样化杂物配置，使用公开的VLA模型训练数据（约1000条演示）进行对比。

**📈 对比分析**

与端到端VLA方法（π_0、NORA、NORA‑1.5）和层级开放词汇规划器（ReKep、MOKA、VoxPoser）对比，UniManip在零样本场景中平均成功率提升至93.75%（对比71.25%），在杂物桌面场景中取得82.5%（对比47.5–57.5%），在长时程交互任务中平均成功率达80%。

**⚠️ 局限性**

局限性主要集中在单视角深度估计噪声导致的抓取/放置失败、运动学极限与碰撞误判、以及对接触不确定性的处理不足；当前框架缺乏触觉感知与全身协同规划，难以完全覆盖所有复杂物理交互情形。

---

## 381. ViMedCSS: A Vietnamese Medical Code-Switching Speech Dataset & Benchmark

**arXiv ID:** 2602.12911 | [PDF](https://arxiv.org/pdf/2602.12911v1)

**作者:** Tung X. Nguyen `[一作]` (VinUniversity), Dung D. Le `[通讯]` (VinUniversity)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5051381005)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建并公开了ViMedCSS数据集，包含34.6小时、16,576句越南语医疗语音，其中每句至少含有一个英语医学词汇，并为其提供时间对齐与代码切换标签。

**💡 创新点**

创新点在于：①首次提供面向越南医学语音的代码切换评估基准；②对整体识别与代码切换片段分别评估，揭示模型在不同语义子域的性能差异；③系统比较多种细粒度适配与上下文偏置策略，为越南医疗ASR提供可行的改进路径。

**🔧 技术方法**

技术方法包括：Whisper、PhoWhisper（越南化的Whisper）、MMS、wav2vec2等主流ASR模型；在PhoWhisper-small上应用LoRA、Attention Guide（AG）、Dynamic Vocabulary（DV）、Rank & Selection（RS）、AdaCS等上下文偏置与参数高效适配技术。

**📊 数据集**

使用Meddict词典中的3,203个越南-英语医学术语作为查询，采集13,000+ YouTube医学视频，经过LLM转写、语义过滤、Levenshtein对齐，最终得到34.6小时的对齐、标注完整的数据集。

**📈 对比分析**

实验结果显示：多语言模型如Whisper-Large-v3在代码切换词识别（CS-WER）上优于越南专用模型；而越南化模型如PhoWhisper-small在整体WER/CER上更优秀。Fine‑tuning后，PhoWhisper-small结合LoRA或AG实现了最佳性能，WER降至≈23.7%，CS-WER≈19.5%。在Hard集上性能下降，但仍保持相对优势。

**⚠️ 局限性**

局限性：数据来源仅为公开YouTube视频，可能包含背景噪声、非专业口音；代码切换仅覆盖英语-越南，未考虑其他语言组合；模型规模与适配方法受实验平台限制，未来需扩展到更大规模或多说话者场景。

---

## 382. A Theoretical Framework for Adaptive Utility-Weighted Benchmarking

**arXiv ID:** 2602.12356 | [PDF](https://arxiv.org/pdf/2602.12356v1)

**作者:** Philip Waggoner `[一作]` `[通讯]` (Stanford University), Philip Waggoner (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将AI与机器学习基准化为多层自适应网络的理论框架，并将人类偏好、模型属性与评价指标通过加权边连接，形成可动态更新的社会技术网络。

**💡 创新点**

创新点在于：① 将基准视为多层网络而非静态指标集合；② 通过联合分析获取的部分价值将人类偏好直接映射为网络权重；③ 设计了受技术梯度与人类反馈约束的投影更新规则，使基准能在保证稳定性的同时随环境与需求演化。

**🔧 技术方法**

使用了图论（多层图、超邻接矩阵）、谱分析、线性/指数传播算子、约束投影优化、凸分析、联立梯度更新与人类偏好映射（如指数/Logistic函数）。

**📊 数据集**

该论文为理论性工作，未提供具体数据集或实验实现；所有示例均为数学建模与假设场景。

**📈 对比分析**

由于未开展实验评估，本文没有提供与传统排行榜或现有评测框架的性能对比；主要以理论可行性与收敛性证明说明其潜在优势。

**⚠️ 局限性**

限制主要包括：① 理论模型假设线性/指数传播可能不足以捕捉真实复杂交互；② 偏好获取依赖于高质量联合分析样本，易受样本偏差影响；③ 投影更新需手工设定约束集合，若设置不当可能导致收敛不稳定；④ 论文未给出实验验证，缺乏对实际部署效果的量化证据。

---

## 383. Arcalis: Accelerating Remote Procedure Calls Using a Lightweight Near-Cache Solution

**arXiv ID:** 2602.12596 | [PDF](https://arxiv.org/pdf/2602.12596v1)

**作者:** Johnson Umeike `[一作]` (University of Maryland), Bahar Asgari `[通讯]` (University of Maryland)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5059742939)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

设计并实现了一款名为 Arcalis 的近缓存 RPC 加速器，利用轻量级硬件引擎在 CPU 的最后一级缓存（LLC）旁边完成 RPC 的序列化、反序列化、函数分发和响应组装，彻底卸载 CPU 的 RPC 处理负担。

**💡 创新点**

创新点包括：① 将 RPC 处理移至靠近 LLC 的硬件区，消除数据搬迁和缓存污染；② 通过可编程微引擎实现完整的 RPC 功能并支持多种微服务；③ 采用双管道（接收/响应）并行处理，最大化资源利用；④ 使用内存映射指令和可配置的命令页实现与用户空间的无缝交互；⑤ 结合 DCA、缓存一致性协议实现零拷贝传输。

**🔧 技术方法**

使用技术有：DPDK‑style 用户空间网络栈、Direct Cache Access（DCA）机制、基于 gem5 的全系统周期级仿真、FPGA 原型实现、TLB/MMU、重排缓冲区（ROB）、状态机控制、可重配置逻辑区（RLR）与固定逻辑区（FLR）的协同工作。

**📊 数据集**

评估数据集为 DeathStarBench 微服务集（memcached、unique_id、post），使用不同的写/读比例（20/80、50/50、80/20 等）以及真实系统生成的 RPC 请求轨迹，重放以测量加速效果。

**📈 对比分析**

对比方法：与纯 CPU 基线和现有硬件加速方案（Dagger、RpcNIC）在同一工作负载下进行端到端执行时间和吞吐量对比。实验显示 Arcalis 在不同写密集度下实现 1.79–4.16× 的速度提升，吞吐量提升 2.5–3.3×，并在微架构层面显著降低指令计数、缓存停滞与前端/后端瓶颈。

**⚠️ 局限性**

局限性包括：只能加速 RPC 层，业务逻辑仍在 CPU 上执行；对写密集型服务收益最大，对纯读取服务增益有限；需要在 SoC 的 LLC 旁放置硬件，受制于芯片设计与布局；实现仍依赖 FPGA/ASIC 原型，未在商用 CPU 上广泛验证；对不同 RPC 协议与数据格式的可扩展性仍需进一步研究。

---

## 384. Implicit-Scale 3D Reconstruction for Multi-Food Volume Estimation from Monocular Images

**arXiv ID:** 2602.13041 | [PDF](https://arxiv.org/pdf/2602.13041v1)

**作者:** Yuhao Chen `[一作]` (University of Waterloo), Jiangpeng He `[通讯]` (Indiana University)

**通讯引用:** 590 | [OpenAlex ID](https://openalex.org/A5063620170)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于单目多食物图像的隐式尺度三维重建 benchmark，用以评估食物份量估计的几何方法。

**💡 创新点**

创新点在于将食物份量估计转化为无显式尺度参照的 3D 重建问题，并提供多物体、多遮挡场景的真实数据。

**🔧 技术方法**

主要技术包括 Hunyuan3D 视觉-语言基准、像素空间启发式缩放、场景几何先验缩放、以及基于度量深度的多阶段缩放。

**📊 数据集**

使用了 MetaFood3D 数据集中的 10 个多食物场景共 24 个对象的 3D 扫描数据。

**📈 对比分析**

对 PSHS、SGPS、MDMS 三类方法与 GPT‑5.2 进行对比，MDMS 在体积估计 MAPE 0.21、几何 Chamfer 5.7 处表现最佳。

**⚠️ 局限性**

局限性包括仍需人工对齐评估、依赖盘子和餐具等隐式尺度参考、以及单视角下尺度不确定性的残余影响。

---

## 385. How cyborg propaganda reshapes collective action

**arXiv ID:** 2602.13088 | [PDF](https://arxiv.org/pdf/2602.13088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 386. Channel-Aware Probing for Multi-Channel Imaging

**arXiv ID:** 2602.12696 | [PDF](https://arxiv.org/pdf/2602.12696v1)

**作者:** Umar Marikkar `[一作]` (University of Surrey), Sara Atito `[通讯]` (University of Surrey)

**通讯引用:** 1048 | [OpenAlex ID](https://openalex.org/A5037459105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Channel-Aware Probing（CAP）框架，用来在冻结的多通道影像（MCI）预训练编码器上进行高效的下游任务探针（probe）学习。

**💡 创新点**

创新点在于两层次的通道意识：①独立特征编码（IFE）在编码阶段保留各通道信息；②解耦池化（DCP）在探针阶段先对每通道内部特征聚合，再跨通道聚合，从而充分利用MCI的通道多样性。

**🔧 技术方法**

使用的技术包括：ViT（Transformer）编码器、IFE、DCP、不同探针聚合架构（如SimplePool、ABMILP、MAB、ProtoBing等），并通过冻结预训练权重进行探针训练。

**📊 数据集**

使用的基准数据集有：CHAMMI（多通道显微镜影像，包含WTC‑11、HPA、CP）、JUMP‑CP（8 通道细胞化学处理图像）和 So2Sat（卫星遥感影像，Sentinel‑1 与 Sentinel‑2 组合）。

**📈 对比分析**

与传统的 JFE+JAP 探针基线和完整微调（Fine‑Tuning）做对比。CAP 在所有探针聚合方式下均优于 JFE+JAP，平均提升约 14%；在最优聚合下，CAP 的性能仅落后完整微调约 7%，而传统基线落后约 21%。

**⚠️ 局限性**

局限性包括：①CAP 仍未能完全与完整微调匹配，仍存在约 7% 的性能差距；②不同数据集的通道语义差异导致 DCP 与 JAP 的收益变化，需进一步探究通道语义对聚合策略的影响；③在计算成本方面，尽管 DCP 计算量与 JAP 相当，但在极大通道数或高分辨率图像时仍可能产生显著开销。

---

## 387. iRULER: Intelligible Rubric-Based User-Defined LLM Evaluation for Revision

**arXiv ID:** 2602.12779 | [PDF](https://arxiv.org/pdf/2602.12779v1)

**作者:** Jingwen Bai `[一作]` (National University of Singapore), Brian Y Lim `[通讯]` (National University of Singapore)

**通讯引用:** 3758 | [OpenAlex ID](https://openalex.org/A5056248594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了iRULER，一种基于可定制Rubric的LLM评估与写作修订交互系统。

**💡 创新点**

创新点在于将六项设计准则（Specific、Scaffolded、Justified、Actionable、Qualified、Refinable）嵌入双层反馈循环，递归评估写作与Rubric本身，支持用户定制并持续优化评估标准。

**🔧 技术方法**

使用GPT‑4.1作为LLM判断者，结合Prompt生成的结构化评估、Why/Why‑Not与How‑To解释，前端使用Vue.js/Quasar，后端为无服务器架构。

**📊 数据集**

评估数据集包括ICNALE Edited Essays、社交媒体广告文案和公开演讲稿等，并使用专家手工标注的ESL评估。

**📈 对比分析**

与文本级LLM反馈和只读Rubric LLM反馈对照，iRULER在写作质量提升、迭代次数、主观感知帮助度、正确性和自信度上均显著优于基线，评分提升平均约15‑30分。

**⚠️ 局限性**

局限在于实验规模有限、对高度主观或多模态任务的泛化未充分验证，且依赖LLM的语言能力，无法评估事实真实性或非文本特征。

---

## 388. Prior-Guided Symbolic Regression: Towards Scientific Consistency in Equation Discovery

**arXiv ID:** 2602.13021 | [PDF](https://arxiv.org/pdf/2602.13021v1)

**作者:** Jing Xiao `[一作]` (National University of Defense Technology), Jie Liu `[通讯]` (National University of Defense Technology)

**通讯引用:** 94135 | [OpenAlex ID](https://openalex.org/A5100454174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于先验引导的符号回归框架 PG‑SR，旨在解决符号回归中的伪方程陷阱，提升方程发现的科学一致性。

**💡 创新点**

创新点包括：①将领域先验编码为可执行的约束程序并引入 Prior‑Annealed Constrained Evaluation（PACE）机制；②在三阶段管线（warm‑up → evolution → refinement）中融合 LLM 生成、经验池与反射机制；③理论证明约束可降低 Rademacher 复杂度，从而收敛更紧的泛化上界。

**🔧 技术方法**

核心技术包括：LLM（GPT‑4o‑mini/ Llama‑3.3‑70B）驱动的方程生成；基于约束检查器的先验验证；PACE 权重调度；L‑BFGS‑B 参数优化；经验池与洞察池的多岛并行搜索；残差增强的反射式微调。

**📊 数据集**

使用多领域数据集：E. coli 生长、材料应力–应变、化学反应动力学、非线性振荡器 1 与 2（均来自 LLM‑SRBench）。

**📈 对比分析**

与搜索型（GPLearn、PySR、DSR 等）、Transformer 型（TPSR、E2E、PhyE2E 等）以及 LLM 型（LLM‑SR、LaSR、DrSR 等）基线在 ID 与 OOD 上做 NMSE 对比；PG‑SR 在所有数据集均实现最低 NMSE，尤其在 OOD 上优势显著，且对先验质量、噪声与数据稀缺具备鲁棒性。

**⚠️ 局限性**

限制：需人工参与构造可执行先验约束，先验质量对结果影响显著；当先验错误时，尽管系统有一定鲁棒性，但性能仍可能下降；LLM 生成过程受模型大小与 prompt 设计限制。

---

## 389. MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs

**arXiv ID:** 2602.12705 | [PDF](https://arxiv.org/pdf/2602.12705v1)

**作者:** Baorong Shi `[一作]` (ByteDance), Zhixiong Yang `[通讯]` (ByteDance)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个多模态医学基础模型，融合视觉、语言、OCR 与报告生成，并通过实体感知持续预训练与代理推理提升临床能力。

**💡 创新点**

创新性构建了大规模实体中心医学词树、统一的 Med‑VLM 评测框架，并将 SFT、RL 与 RFT 结合的多阶段训练管线用于证据驱动的多步推理。

**🔧 技术方法**

采用多模态原生分辨率 Transformer、Seed‑ViT 视觉编码器、MLP 适配器，结合层级医学知识图谱、结构化链式思维合成、强化学习与多层次奖励系统。

**📊 数据集**

使用约 640 B 令牌的高质量语料（公共网络、医学图书、2.8 B 病变图像、2.2 B 开源数据），以及 30+ 公共基准（MMMU、VQA_RAD、PubMedQA、MedQA 等）和自研 VQA、OCR、Caption、Report 数据集。

**📈 对比分析**

在统一提示与评测协议下，与 GPT‑5.2 Thinking、Gemini 3.0 Pro/2.5 Pro 等 SOTA 进行对比；该模型在 30+ 基准的平均分上领先，并在视觉诊断、医学影像、诊断推理、医学文本、报告生成和指令跟随等六大能力上均表现优异。

**⚠️ 局限性**

仍存在长文本报告真实性不足、对 IU‑Xray 等部分数据集的表现低于最佳基线、在极端分布漂移下的鲁棒性有限，以及对大规模算力与高质量数据的高度依赖。

---

## 390. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

**arXiv ID:** 2602.12670 | [PDF](https://arxiv.org/pdf/2602.12670v1)

**作者:** Xiangyi Li `[一作]` (BenchFlow), Han-chung Lee `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个包含 84 个跨 11 个领域、可容器化执行的 Agent Skills 基准，使用 7,308 条轨迹测试 7 个模型-工具组合。

**💡 创新点**

将 Skills 作为首要评估对象，提出三种条件（无 Skills、Curated Skills、Self‑Generated Skills）对照实验，首次系统揭示 Curated Skills 的平均提升 16.2pp，且自生成 Skills 对性能几乎无益。

**🔧 技术方法**

采用 Harbor 框架、Docker 容器化环境、Deterministic Verifier、Claude Code、Gemini CLI、Codex CLI 等商用工具，并使用 GPT‑5.2、Claude Opus 4.5/4.6/… 等前沿 LLM。

**📊 数据集**

任务来源于 105 名贡献者提交的 322 个候选任务，最终筛选出 84 个任务，覆盖 11 个领域，包含手工编写指令、参考解决方案与验证脚本。

**📈 对比分析**

通过 Pass Rate 与 Normalized Gain 两指标对比，Curated Skills 在所有配置平均提升 16.2pp，最优组合 Gemini CLI + Gemini 3 Flash 达到 48.7% Pass；自生成 Skills 则平均降幅 1.3pp。

**⚠️ 局限性**

基准仅适用于终端容器化任务，难以直接迁移到 GUI 或多代理长周期场景；自生成 Skills 的负面结果表明当前 LLM 仍难以自动合成高质量 procedural 内容；此外，Context 长度增加可能部分解释提升，需更严格的对照实验。

---

## 391. A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models

**arXiv ID:** 2602.12499 | [PDF](https://arxiv.org/pdf/2602.12499v1)

**作者:** Mugunthan Shandirasegaran `[一作]` (New Jersey Institute of Technology), Shuai Zhang `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 8589 | [OpenAlex ID](https://openalex.org/A5025097254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析并证明单层Mamba块在有选择性门控的状态空间模型下对结构化数据的训练收敛与泛化性质。

**💡 创新点**

首次给出包含输入依赖门控的Mamba在多数投票与局部结构数据上的非渐近样本复杂度与迭代收敛率证明，并揭示门控实现特征选择的机制。

**🔧 技术方法**

采用特征学习框架、梯度下降分析、非对称递归门控展开、矩阵范数与概率上界等理论工具。

**📊 数据集**

仅使用模拟的噪声版基于正交向量的合成数据，包含多数投票与局部结构两种数据模型。

**📈 对比分析**

通过与Transformer理论分析对比，证明在两种数据结构下Mamba的收敛速度和样本复杂度随信号强度提升、噪声降低而改善；实验验证收敛加速与门控特征对齐。

**⚠️ 局限性**

只考虑单层单头无残差/层归一化的简化Mamba；数据模型过于理想化；未覆盖多层、多头及实际任务的实验与理论推广。

---

## 392. Meta-Monomorphizing Specializations

**arXiv ID:** 2602.12973 | [PDF](https://arxiv.org/pdf/2602.12973v1)

**作者:** Federico Bruzzone `[一作]` (Universita degli Studi di Milano), Walter Cazzola `[通讯]` (Universita degli Studi di Milano)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出并实现了一种基于编译时宏的元单态化专化框架，允许在不修改 Rust 编译器的情况下实现零成本的函数和 trait 专化。

**💡 创新点**

将专化视为元单态化，通过生成专门化的 trait 和实现，利用类型级谓词编码专化约束，解决传统专化的重叠与一致性问题。

**🔧 技术方法**

使用 Rust 的过程宏（procedural macro）与宏展开、谓词化约束、DNF 正则化、静态重叠检测、生命周期与高阶类型的单态化等技术。

**📊 数据集**

对 65 个公开 Rust 项目（来自 crates.io 与 GitHub）进行代码相似度分析，构建专化候选集合。

**📈 对比分析**

通过 HIR 分析与树编辑距离衡量可专化函数，并将手工专化与元单态化后的代码量、二进制大小、编译时间进行对比，结果表明元单态化能显著减少冗余代码且保持与标准优化相当的性能。

**⚠️ 局限性**

不支持存在类型（impl Trait / dyn Trait）位置的专化、递归多态的无限实例化场景；对高阶谓词的支持有限，且实现依赖宏而非编译器本身，导致某些复杂模式无法覆盖。

---

## 393. Multimodal Classification via Total Correlation Maximization

**arXiv ID:** 2602.13015 | [PDF](https://arxiv.org/pdf/2602.13015v1)

**作者:** Feng Yu `[一作]` (Nanjing University of Science and Technology), Jianfeng Lu `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 9550 | [OpenAlex ID](https://openalex.org/A5061472917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出最大化多模态特征与标签总相关性的学习目标TCMax，解决模态竞争导致的性能退化问题。

**💡 创新点**

通过信息理论证明总相关性能够同时涵盖联合学习、单模态学习与模态对齐，并构建无超参数的TCMax损失函数。

**🔧 技术方法**

基于Mutual Information Neural Estimation发展出的Total Correlation Neural Estimation (TCNE)，并在多模态分类中采用TCMax损失。

**📊 数据集**

在CREMA‑D、Kinetics‑Sounds、AVE、VGGSound、UCF101和MVSA等音频、视觉、文本情感与动作数据集上进行实验。

**📈 对比分析**

与Concat、Share Head、OGM‑GE、AGM、QMF、MMPareto等现有方法对比，TCMax在多数数据集上取得最高或相近的多模态准确率，并显著降低两模态预测的JS‑Divergence。

**⚠️ 局限性**

目前仅适用于分类任务，无法直接扩展到检测、生成等多模态应用，需进一步设计专门的模型架构。

---

## 394. VI-CuRL: Stabilizing Verifier-Independent RL Reasoning via Confidence-Guided Variance Reduction

**arXiv ID:** 2602.12579 | [PDF](https://arxiv.org/pdf/2602.12579v1)

**作者:** Xin-Qiang Cai `[一作]` (RIKEN AIP), Masashi Sugiyama `[通讯]` (The University of Tokyo)

**通讯引用:** 22165 | [OpenAlex ID](https://openalex.org/A5072744508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无验证器的课程强化学习框架 VI-CuRL，用于在没有外部验证器的情况下稳定和提升大型语言模型的推理性能。

**💡 创新点**

创新点在于：①利用模型自身的置信度（基于熵的长度归一化）动态构造课程，显式降低梯度方差；②通过重要性采样权重实现无偏估计，并在理论上证明了方差降低与渐进一致性；③使用动态分位数阈值自适应地调整保留率。

**🔧 技术方法**

采用了 Group Relative Policy Optimization（GRPO）+ PPO-Clip、KL 正则化、置信度估计（熵归一化）、重要性采样权重、课程掩码以及梯度方差分解与分析等技术。

**📊 数据集**

实验数据集包括 AIME-2024/2025、AMC-2023、MATH500、Minerva MATH、OlympiadBench 等六个数学推理 benchmark，模型涵盖 Qwen2.5-Math-1.5B、DeepSeek-R1-Distill-Qwen-1.5B、Llama-3.2-3B-Instruct 和 Qwen2.5-Math-7B。

**📈 对比分析**

与 oracle‑reward RLVR、VCRL、AdaRFT、majority‑vote、entropy 等基线对比，VI‑CuRL 在 oracle 和无验证器场景均能实现更稳定、更高的 Pass@1/Pass@8 分数，尤其在无验证器的情况下显著降低训练崩溃风险。

**⚠️ 局限性**

局限性包括：置信度作为课程难度的代理可能不总是与实际正确性一致；课程保留率的动态调度需要经验性设置；目前验证仅在推理/数学任务上，尚未在更广泛的 RL 场景中检验。

---

## 395. HyperMLP: An Integrated Perspective for Sequence Modeling

**arXiv ID:** 2602.12601 | [PDF](https://arxiv.org/pdf/2602.12601v1)

**作者:** Jiecheng Lu `[一作]` (Georgia Institute of Technology), Shihao Yang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5057260690)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新表述自回归注意力为动态两层 MLP，并提出 HyperMLP/HyperGLU 通过上下文激活的特征空间与序列空间混合，提升表达能力。

**💡 创新点**

创新点：①把注意力视作动态两层 MLP；②在隐藏层引入逆偏移（lag）序列混合与低秩+对角（DPLR）参数化；③使用 ReLU/GLU 激活实现输入条件路由；④给出理论可表达性分析并解释多种注意力设计。

**🔧 技术方法**

技术手段：逆偏移序列混合、低秩+对角参数化、ReLU/GLU 激活、动态权重生成、理论可表达性证明、实验验证（MAD、NanoGPT）。

**📊 数据集**

数据集：MAD 诊断基准、NanoGPT OpenWebText2（≈15B/100B tokens）、FineWeb‑Edu、通用语言模型评测基准（MMLU、GPQA、BBH 等）。

**📈 对比分析**

比较方法：在匹配参数预算下与 softmax‑attention、ReLU‑attention、RoPE、KV‑卷积等进行对比；HyperMLP/HyperGLU 在 MAD 指标上提升 14–15 分，NanoGPT 训练损失下降 0.05–0.1，通用语言模型整体排名提升，部分指标达到最佳。

**⚠️ 局限性**

局限性：实现未达到 FlashAttention 级别的高效性；未在极大规模 LLM 上验证；模型结构更复杂，参数调优成本更高；实验仅基于公开数据集，缺乏对实际部署安全与伦理的评估。

---

## 396. ReFilter: Improving Robustness of Retrieval-Augmented Generation via Gated Filter

**arXiv ID:** 2602.12709 | [PDF](https://arxiv.org/pdf/2602.12709v1)

**作者:** Yixin Chen `[一作]` (City University of Hong Kong), Chun Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6094 | [OpenAlex ID](https://openalex.org/A5101441768)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种新的检索增强生成框架ReFilter，在LLM内部通过token级过滤与融合提升检索信息的利用效率与鲁棒性。

**💡 创新点**

在隐式融合范式中首次引入基于上下文编码、门控过滤与token级融合的模块，能够针对每个token动态评估重要性并抑制噪声；同时保持非参数检索与可批处理的优势。

**🔧 技术方法**

使用Transformer上下文编码器、门控过滤器（sigmoid门）、位置掩码、层归一化与轻量级适配器，将加权token特征注入LLM隐藏层；训练采用标准教师强制与门控稀疏正则。

**📊 数据集**

在四个通用QA基准（2WikiMultihopQA、HotpotQA、PopQA、ComplexWebQuestions）和五个医学QA基准（MedQA、MedMCQA、PubMedQA、BioASQ、MMLU_Med）上进行评测，检索语料为维基百科。

**📈 对比分析**

相较于Vanilla LLM、S-RAG、PRAG、DyPRAG等基线，ReFilter在所有模型-数据组合中平均提升2–5个百分点，尤其在零样本医学QA上平均达70.01%准确率；在鲁棒性实验中对噪声与top‑k变化的抵抗力最强。

**⚠️ 局限性**

仅在单一模型上训练；对大规模语料的预编码仍需额外存储；当检索候选极大或包含大量重复信息时，token级过滤仍可能产生误过滤，且在极端多模态或实时对话场景下需进一步验证。

---

## 397. ProbeLLM: Automating Principled Diagnosis of LLM Failures

**arXiv ID:** 2602.12966 | [PDF](https://arxiv.org/pdf/2602.12966v1)

**作者:** Yue Huang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12595 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 benchmark-agnostic 的自动化探测框架（ProbeLLM），通过层次化的 Monte Carlo Tree Search 与工具增强生成/验证，自动发现并聚类大型语言模型的系统性失败模式。

**💡 创新点**

创新点：① 用分层 MCTS 在有限预算下实现全局探索与局部精炼的平衡；② 仅使用可验证的测试案例并结合检索/Python 执行等工具，提升失败检测的可靠性；③ 采用 failure‑aware embedding 与 boundary‑aware induction，将单个失败转化为可解释的失败模式。

**🔧 技术方法**

技术栈包括：Hierarchical MCTS、UCB 选择、工具增强的 LLM 生成（web 检索、Python 运行）、failure‑aware embedding（prompt+错误描述）、HDBSCAN 聚类、LLM 辅助的边界诱导（boundary‑aware induction）。

**📊 数据集**

使用 5 个多任务评测基准（涵盖事实推理、数学、代码、常识等领域）以及 12 种目标 LLM（包括开源与闭源模型）进行实验；所有测试案例均附有可验证的标准答案。

**📈 对比分析**

与静态基准、AutoDetect、PAIR 等方法对比；评价指标包括错误率、噪声率、簇大小标准差、簇重叠率。ProbeLLM 在错误率、噪声率和簇细粒度上均优于基线，发现更多独立的失败模式；且成本低（单个高错误率测试 <$0.1），并能通过错误聚焦微调提升模型性能。

**⚠️ 局限性**

局限性：仅适用于可验证答案的任务，难以处理开放式生成任务；聚类结果受超参数影响，需要手动调优；仍可能漏检极小或高维度的失败模式；工具调用带来额外的计算与数据隐私成本。

---

## 398. Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models

**arXiv ID:** 2602.12444 | [PDF](https://arxiv.org/pdf/2602.12444v1)

**作者:** Alexander W. Goodall `[一作]` (Imperial College), Francesco Belardinelli `[通讯]` (Imperial College)

**通讯引用:** 820 | [OpenAlex ID](https://openalex.org/A5055883955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于恢复的屏障框架，用高斯过程（GP）在线学习未知动力学，结合预先计算的安全备份控制器，在训练与部署过程中保证高概率安全。

**💡 创新点**

创新点：①将GP不确定性解析传播到安全屏障判定，实现无需采样即可获得严格的概率安全下界；②通过可恢复状态（ε‑recoverable）扩展控制不变集，显著扩大可安全行动空间；③在保证安全的前提下保持对强学习算法（A2C）的兼容性，提升样本效率。

**🔧 技术方法**

核心技术：Gaussian Process 动力学模型（SGPR/SVGP）、线性二次调节（LQR）备份控制器、凸多面体/椭圆不变集、分析不确定性集合、可恢复状态判定、基于安全屏障的动作切换、A2C 价值函数与策略梯度、稀疏 GP 推断、以及经验回放与对称对数奖励。

**📊 数据集**

实验数据集：经典连续控制任务（CartPole、MountainCar、2D导航、车道行驶）以及较高维度的MuJoCo Hopper（11维状态、3维动作）。

**📈 对比分析**

与方法比较：与 CMDP 基线（CPO、PPO‑Lag）以及基于已知动力学的 MPS/DMPS 进行对比。结果显示 A2C‑GP‑Shield 在安全概率上始终达到 1（无违规），且在大多数环境中实现与甚至超过基线的平均回报，样本效率更高、收敛更快；MPS/DMPS 虽然安全，但假设更强。

**⚠️ 局限性**

局限性：①需要预先设计或可学习的安全备份控制器，且其不变集需足够大，否则安全保障受限；②GP 推断在状态维数过大（>20）时计算成本显著；③假设干扰集合已知且边界已知；④无法直接处理动态障碍物或缺失备份控制器的场景。

---

## 399. "It's More of a Lifestyle'': Design Considerations for Supporting Everyday Practices in Community-Based Farming

**arXiv ID:** 2602.13119 | [PDF](https://arxiv.org/pdf/2602.13119v1)

**作者:** Minghe Lu `[一作]` (University of Minnesota), Ji Youn Shin `[通讯]` (University of Minnesota)

**通讯引用:** 762 | [OpenAlex ID](https://openalex.org/A5018185775)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过访谈与现场观察，探索了 Hmong 社区农户的日常耕作实践、信息追踪与社交资本的运用，并提出了支持社区型小规模农场的技术设计机会

**💡 创新点**

首次系统阐述了结缘资本、桥接资本与链接资本在小规模农场中的相互作用，并将非正式的工作绕行与低门槛信息追踪方法转化为可落地的技术设计建议

**🔧 技术方法**

主要使用访谈记录、观察笔记、手绘地图与照片等质性数据；对现有工具（如 Excel、Google 地图、手机拍照）进行分析，提出低门槛移动应用/语音/地图标记的设计方向

**📊 数据集**

数据集为 11 位 Hmong 农户的访谈与现场观察记录（约 45-60 分钟访谈、多次田间现场记录）

**📈 对比分析**

本文未进行算法或性能评估，比较方法仅为对现有工具的适用性和农户的可行性评估，说明现有技术在此情境下的不足与改进空间

**⚠️ 局限性**

仅研究单一族群、样本量有限，研究者立场与偏见可能影响解读，且缺乏对其他小规模农场社区的跨群体验证

---

## 400. Prototype-driven fusion of pathology and spatial transcriptomics for interpretable survival prediction

**arXiv ID:** 2602.12441 | [PDF](https://arxiv.org/pdf/2602.12441v1)

**作者:** Lihe Liu `[一作]` (MD Anderson Cancer Center), Lulu Shang `[通讯]` (MD Anderson Cancer Center)

**通讯引用:** 6399 | [OpenAlex ID](https://openalex.org/A5004722925)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种面向配准的WSI与空间转录组联合分析的多级混合专家（MoE）框架，利用任务导向的原型学习实现跨模态融合与风险预测；

**💡 创新点**

创新点在于：① 引入任务导向的原型学习与跨模态门控注意力相结合的多级专家架构；② 采用EMA与多样性正则实现全局原型稳定与专一性；③ 通过原型注释与风险分解提供可解释的分子与形态学阐释；

**🔧 技术方法**

使用技术包括：多实例学习（MIL）、Transformer基的空间编码（TransMIL）、scGPT与UNI2预训练编码器、原型学习（Top‑k注意力）、EMA、正交多样性正则、门控注意力融合；

**📊 数据集**

数据集为公开的三阴性乳腺癌（TNBC）数据集，包含273个样本，配有WSI和空间转录组，评估五项生存终点；

**📈 对比分析**

与多种单模态与多模态基线（CLAM、PatchGCN、TransMIL、PANTHER、PORPOISE、MCAT、SurvPath、PIBD、ProSurv等）进行5折交叉验证比较，取得大部分终点的C-index最高或相近，显示出模型的优越性能；

**⚠️ 局限性**

局限性包括：仅在单一小规模公开数据集上验证，缺乏外部验证；对ST-WSI配准与数据质量高度依赖，计算成本和处理时间高；模型对超参数的敏感性与收敛性仍需进一步改进。

---

## 401. Automating UI Optimization through Multi-Agentic Reasoning

**arXiv ID:** 2602.13126 | [PDF](https://arxiv.org/pdf/2602.13126v1)

**作者:** Zhipeng Li `[一作]` (ETH Zurich), Christian Holz `[通讯]` (ETH Zurich)

**通讯引用:** 6809 | [OpenAlex ID](https://openalex.org/A5046815740)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多智能体推理的自动 UI 优化框架（Auto‑Optimization），能够通过自然语言指令自我配置多目标优化问题、生成 Pareto‑前沿候选布局，并由 VLM 自动验证并选取最符合用户意图的最终布局，从而实现 MR/VR 环境下的自适应 UI 布局。

**💡 创新点**

创新点在于：
1) 将 GPT‑Vision 之类的 VLM 作为代理，实现对用户指令的歧义检测、优化问题的动态配置和结果验证；
2) 通过“先歧义检测 → 配置 → 优化 → 验证”的多阶段流水线，将手工设置和手动选择从优化流程中剔除；
3) 采用可解释的 JSON 配置方式，让用户能可视化优化过程；
4) 通过少量示例（few‑shot）让 VLM 学习用户指令模式，显著提升歧义检测准确率。

**🔧 技术方法**

技术手段包括：
- 视觉‑语言模型（GPT‑4 Vision 2024‑05‑01‑Preview）
- 近似多目标优化器（NSGA‑III + AASF 散列）
- Unity + AUIT 交互框架
- Pymoo（Python）实现求解与 Pareto‑前沿筛选
- Socket‑based 通讯实现各模块协作
- Chain‑of‑Thought 与树‑of‑Thought 逻辑实现多阶段推理。

**📊 数据集**

使用的数据集：
- 27 位 MR 用户收集的 415 条指令（平均 26.5 词）
- 18 个场景下的 4 选布局评估（由 26 名 MR 用户参与）
- 9 次实验（3 场景 × 3 方法）共 12 位参与者的手动与自动布局评估。

**📈 对比分析**

与两种基线（手动摆放、Pareto‑Adapt）对比，指标包括：
- 调整距离、调整次数（Auto‑Optimization 约 70% 下降，Pareto‑Adapt 约 60% 下降）
- NASA‑TLX 工作量（Auto‑Optimization 在总体负荷上最低，精神负荷略高）
- 排名（Auto‑Optimization 始终排第一）
- 任务完成时间（Auto‑Optimization 与 Pareto‑Adapt 相近，均远快于手动）。整体表现：Auto‑Optimization 在降低用户努力、提升满意度方面明显优于 Pareto‑Adapt，并在大部分指标上与手动方式竞争。

**⚠️ 局限性**

局限性：
1) 只支持以目标空间（objective functions）为指令的语言，无法直接处理具体设计空间（如“放在距我 3 米处”）指令；
2) 目标函数和约束集固定，可能无法覆盖所有用户需求；
3) 假设用户偏好落在 Pareto‑前沿，实际可能不完全满足；
4) 歧义检测与问答可能无意中引导用户偏好；
5) 需要用户先通过文本提供高层意图，低层微调仍需手动完成；
6) 需要持续收集用户历史数据以实现个性化。

---

## 402. Reliable Hierarchical Operating System Fingerprinting via Conformal Prediction

**arXiv ID:** 2602.12825 | [PDF](https://arxiv.org/pdf/2602.12825v1)

**作者:** Rubén Pérez-Jove `[一作]` (Universidade da Coruña), Jose Vázquez-Naya `[通讯]` (Universidade da Coruña)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5000779478)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对操作系统指纹识别进行层级化，利用 Conformal Prediction（CP）生成可信的预测集合。

**💡 创新点**

设计并评估两种结构化 CP 方案——层级 CP（L-CP）和投影 CP（P-CP），并提出层级不一致率（HIR）作为结构一致性评估指标。

**🔧 技术方法**

使用多层感知机（MLP）作为基分类器，配合 split CP、层级/投影校准技术，以及覆盖率、集合大小和 HIR 等评估指标。

**📊 数据集**

采用 Masaryk University 公开网络流量数据集，包含 109,663 条记录，三层 OS 标签（家庭、主要版本、叶子）共 12/50/88 类别。

**📈 对比分析**

在不同显著性水平下对比 L-CP 与 P-CP 的覆盖率、平均集合大小与 HIR；P-CP 维持 HIR=0，但集合更大；L-CP 组集合更小，但 HIR 较高；两者均满足理论覆盖率。

**⚠️ 局限性**

L-CP 的层级不一致率高，导致逻辑冲突；P-CP 由于投影导致过度保守，集合放大；两种方法均易受类不平衡和概念漂移影响。

---

## 403. Adaptive Scaling with Geometric and Visual Continuity of completed 3D objects

**arXiv ID:** 2602.12905 | [PDF](https://arxiv.org/pdf/2602.12905v1)

**作者:** Jelle Vermandere `[一作]` (KU Leuven), Maarten Vergauwen `[通讯]` (KU Leuven)

**通讯引用:** 3208 | [OpenAlex ID](https://openalex.org/A5006850646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于SDF和纹理场的可编辑完整三维模型框架，能够自动分割部件、定义可缩放区域并进行平滑插值，实现比例化和无伪影的变形。

**💡 创新点**

核心创新在于将完整SDF与纹理字段统一为CSDF，利用近似凸分解实现无监督的部件分割；通过规划平面控制区并引入重复模式，既保持结构连贯，又支持大幅变形而不失真。

**🔧 技术方法**

技术包括AutoSDF与IF-Net进行几何和纹理补全；近似凸分解（CoACD）实现部件划分；线性插值和索引边界检测实现SDF、颜色与部件索引的平滑过渡；重复模式扩展与Unity Raymarching渲染。

**📊 数据集**

使用Matterport3D室内场景和Sketchfab的高质量三维扫描数据进行实验。

**📈 对比分析**

与全局缩放和简单选择性缩放方法对比；实验显示该方法在复杂形状和重复结构上显著减少变形伪影、保持比例，视觉质量优于传统方法。

**⚠️ 局限性**

局限性包括对非重复、极其复杂或有机形状的处理仍不理想；分辨率受限于128³体素，细节可能被压缩；凸分解的几何划分有时与语义部件不完全匹配。

---

## 404. Model-Aware Rate-Distortion Limits for Task-Oriented Source Coding

**arXiv ID:** 2602.12866 | [PDF](https://arxiv.org/pdf/2602.12866v1)

**作者:** Andriy Enttsel `[一作]` (Mitsubishi Electric Research and Development Centre Europe), Vincent Corlay `[通讯]` (Mitsubishi Electric Research and Development Centre Europe)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文重新审视了任务导向源编码（TOSC）的理论极限，提出了考虑任务模型子最优性的模型感知速率-失真界限；

**💡 创新点**

创新点在于利用间接速率-失真（iRD）理论，指出传统的oracle速率-失真假设往往不可行，并给出了三种新的、可操作的、任务模型相关的速率-失真下界；

**🔧 技术方法**

主要技术包括间接速率-失真理论、估计-压缩（E&C）与压缩-估计（C&E）框架、Blahut–Arimoto算法、后验采样（S&C）与时间共享（TS）近似；

**📊 数据集**

实验使用了MNIST、CIFAR‑100和ImageNet三个标准分类数据集，分别采用LeNet、VGG‑16和ResNet‑50模型；

**📈 对比分析**

与传统的oracle界限、E&C、iE&C以及S&C做对比，结果显示现有最先进的TOSC方案在比特率上相距理论极限远，且S&C在部分误差范围内能超越iE&C；

**⚠️ 局限性**

主要局限在于：(1) 需要事先已知任务模型；(2) 对高维连续后验的iE&C实现仍需近似；(3) 结果受限于实验中的模型分割与量化实现，未覆盖所有实际部署场景。

---

## 405. Discovering Semantic Latent Structures in Psychological Scales: A Response-Free Pathway to Efficient Simplification

**arXiv ID:** 2602.12575 | [PDF](https://arxiv.org/pdf/2602.12575v1)

**作者:** Bo Wang `[一作]` (Tsinghua University), Shiguang Ni `[通讯]` (Tsinghua University)

**通讯引用:** 1523 | [OpenAlex ID](https://openalex.org/A5019716362)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种仅基于题目文本的主题建模框架，用于在无响应数据的前提下进行量表简化。

**💡 创新点**

创新点在于将题目语义结构视为可检验的前置结构，通过密度聚类与主题建模实现可解释的题项代表性选择，填补了传统IRT/ML方法对大样本的依赖。

**🔧 技术方法**

使用了句子嵌入（qwen3-embedding-4b）、UMAP降维、HDBSCAN聚类、BERTopic主题建模以及基于成员概率的代表性题项筛选技术。

**📊 数据集**

实验采用了三种已验证量表：DASS（42项）、IPIP（50项）和中文EPOCH-CN（20项），并在各自公开数据集中检验。

**📈 对比分析**

通过CFA、Cronbach α、子量表相关和交叉形式相关等指标与原量表比较，短表在结构有效性、内部一致性和内容保真度上与原表相当，且在无样本条件下快速生成可用短表，性能优于传统仅依赖数据的缩减方法。

**⚠️ 局限性**

局限包括对嵌入模型和语言特性的依赖、对极短量表稳定性和跨文化泛化的进一步验证需求，以及仍需后续响应数据验证以确保测量等效。

---

## 406. Can we trust AI to detect healthy multilingual English speakers among the cognitively impaired cohort in the UK? An investigation using real-world conversational speech

**arXiv ID:** 2602.13047 | [PDF](https://arxiv.org/pdf/2602.13047v1)

**作者:** Madhurananda Pahar `[一作]` (University of Sheffield), Heidi Christensen `[通讯]` (University of Sheffield)

**通讯引用:** 2808 | [OpenAlex ID](https://openalex.org/A5045619924)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究通过收集英国多语言少数族裔群体的对话语音，评估现有AI模型在检测认知受损与健康双语者时的可信度与偏差。

**💡 创新点**

首次系统性探讨多语言少数族裔人群中的AI偏差，揭示南约克郡口音易被误判为严重认知衰退，并结合定量与定性方法提供全面证据。

**🔧 技术方法**

使用开源ASR（Whisper、Wav2Vec2、NVIDIA），声学与语言特征提取，SVM/LLM分类、SVR回归及TF‑IDF定性分析。

**📊 数据集**

基于CognoMemory自有数据集（1,395名受试者、263小时语音、14个记忆提问）与公开的DementiaBank数据集进行实验。

**📈 对比分析**

通过WER、分类准确率与MMSE RMSE等指标比较，多语者与单语者在ASR上差异不显著，但使用语言特征的分类器在记忆、流畅性和阅读题中表现出显著偏差，回归误差在多语者更大。

**⚠️ 局限性**

主要局限在于缺乏多语言认知受损样本、手工转录数量有限、年龄分布不平衡、未对ASR进行微调以及仅覆盖有限口音与语言。

---

## 407. Towards interpretable models for language proficiency assessment: Predicting the CEFR level of Estonian learner texts

**arXiv ID:** 2602.13102 | [PDF](https://arxiv.org/pdf/2602.13102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 408. Policy4OOD: A Knowledge-Guided World Model for Policy Intervention Simulation against the Opioid Overdose Crisis

**arXiv ID:** 2602.12373 | [PDF](https://arxiv.org/pdf/2602.12373v1)

**作者:** Yijun Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5097 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于世界模型的政策干预模拟框架Policy4OOD，可同时实现未来预测、反事实推理和政策优化。

**💡 创新点**

核心创新在于将政策知识图谱、空间图神经网络与Transformer相结合，利用向量量化发现可共享的干预策略，并通过MCTS在学习到的模拟器上搜索最优政策组合。

**🔧 技术方法**

技术包括政策知识图谱构建（LLM提取）、关系型图神经网络编码、向量量化代码本、空间图神经网络聚合邻域、Transformer编码器+交叉注意力预测头，以及基于世界模型的MCTS优化。

**📊 数据集**

使用了自构建的48州每月级别数据集，涵盖2019-2024年的过量死亡、12项社会经济指标以及从立法文本提取的结构化政策编码。

**📈 对比分析**

在标准预测与跨州泛化两种设置下，与LSTM、TCN、Transformer、MTGNN、STGCN、GraphWavelet、AGCRN等基线相比，Policy4OOD在MAE/RMSE上均优越，尤其在长周期预测和未见州的泛化表现突出。

**⚠️ 局限性**

局限性包括数据覆盖仅为48州，政策文本抽取和编码仍可能产生噪声，模型对政策执行细节和时序滞后捕捉有限，且未能与领域专家充分验证反事实与优化结果。

---

## 409. Completeness in the Polynomial Hierarchy and PSPACE for many natural problems derived from NP

**arXiv ID:** 2602.12350 | [PDF](https://arxiv.org/pdf/2602.12350v1)

**作者:** Christoph Grüne `[一作]` (RWTH Aachen University), Lasse Wulf `[通讯]` (IT University of Copenhagen)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5062230321)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出了一个统一的框架，用于证明从 NP 派生的多层决策问题在多项式层次（Σk^p）和 PSPACE 中的完整性。通过引入 NP‑with‑solutions 类和解嵌入 (solution‑embedding) 归约，作者能够把传统 NP‑完全问题（如 Clique、Vertex‑Cover、Knapsack 等）提升到对应的高阶复杂度层级，并给出一系列元定理，阐释为何多层或对抗性扩展会天然产生 Σk^p‑完整性。

**💡 创新点**

创新点在于：
1) 对 NP 做了细化，将解显式为组合对象，形成 NP‑with‑solutions；
2) 定义并证明了解嵌入归约的可传递性和对多层游戏的保留性质；
3) 提出了可一次性升阶的元定理，覆盖任意常数 k 的多层问题以及 k 为输入的情况；
4) 通过“双模仿”（dual‑mimicking）技术，将低阶归约直接迁移到高阶多层游戏；
5) 统一解释了多层拦截、可调整鲁棒优化等多领域中已知的 Σk^p‑完整性结果，并揭示了其背后的通用结构。

**🔧 技术方法**

核心技术包括：
- NP‑with‑solutions 的定义与验证器统一表述；
- 解嵌入归约的构造与证明（包括 Cook‑Levin 的改造）以及可传递性；
- 元定理的构造：利用解嵌入归约将基问题提升为多层问题；
- 双模仿证明，展示两者之间的策略对应；
- 对拦截游戏的特殊构造（添加伪补集、规则 1–4 等），以及通过 gadget 将受限游戏还原为标准游戏。

**📊 数据集**

本工作为理论复杂度研究，未使用具体数据集；所有结果均在形式化归约与逻辑游戏框架下给出。

**📈 对比分析**

与以往逐个问题的完整性证明相比，本文提供了一个统一的元归约框架，能够一次性给出所有 NP‑complete 基础问题的多层 Σk^p‑完整性结果；在已知的特殊案例（如 Clique‑interdiction、Knapsack‑interdiction、两阶段 TSP 等）上，得到与原文献相同甚至更强的结果。由于研究聚焦在理论证明，未涉及实验性能评估。

**⚠️ 局限性**

局限性：
- 需要 NP‑with‑solutions 的显式解结构，若某 NP 问题的解不可直接显式化，归约可能受限；
- 解嵌入归约的构造在某些问题上可能繁琐；
- 论文未探讨多层问题的求解算法或近似方法，仅停留在复杂度层面；
- 对非 NP 原始问题的多层扩展（如 PSPACE 原始问题）尚未给出完整性结论。

---

## 410. Temporally-Sampled Efficiently Adaptive State Lattices for Autonomous Ground Robot Navigation in Partially Observed Environments

**arXiv ID:** 2602.13159 | [PDF](https://arxiv.org/pdf/2602.13159v1)

**作者:** Ashwin Satish Menon `[一作]` (University of Rochester), Thomas M. Howard `[通讯]` (University of Rochester)

**通讯引用:** 4064 | [OpenAlex ID](https://openalex.org/A5002064705)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种名为 TSEASL 的区域规划架构，在部分可观测的越野环境中通过考虑更新或优化后的历史轨迹，为低层规划器提供更稳定的参考路径，减少安全干预。

**💡 创新点**

创新点在于将时间维度采样与状态格点相结合，使用轨迹选择器和节点优化器在保持近似最优的同时提升轨迹稳定性，并引入成本阈值 α 决策机制。

**🔧 技术方法**

采用 EASL/KEASL 搜索空间、A* / ARA* 规划、节点侧向采样、成本阈值 α 决策以及局部 MPPI 或 RHC 控制技术。

**📊 数据集**

使用 Clearpath Warthog 无人地面车辆在森林环境中的真实地图数据，并在 1747 个规划实例上进行评估。

**📈 对比分析**

通过与基线 KEASL 对比，TSEASL 在路径偏差（MHD）、手动干预次数和规划时间等指标上表现更好，实验显示 α=0.95~0.98 时最优。

**⚠️ 局限性**

局限性包括对状态格点结构的依赖、对快速地图变化的响应需改进、缺乏动态阈值选择机制，以及在离参考轨迹过远时的跟踪误差问题。

---

## 411. The Only Distributive Law Over the Powerset Monad Is the One You Know

**arXiv ID:** 2602.13144 | [PDF](https://arxiv.org/pdf/2602.13144v1)

**作者:** Sergey Goncharov `[一作]` (University of Birmingham), Paul Wild `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5058881043)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了集合函子在幂集单子上的分配律（即从集合范畴到关系范畴的扩展），给出了存在性与唯一性的完整理论。

**💡 创新点**

创新点在于提出并证明了“逐点有界”（elementwise bounded）这一更广泛的类，证明该类函子在保留弱直角三角形的情况下存在分配律且唯一；并通过幂集函子展示了该唯一性在非可接近函子中失效，给出了三条不同的分配律。

**🔧 技术方法**

采用范畴论技术：可接近性、弱直角三角形保持、Barr 扩展、单子同态、Kleisli 语义等，结合自然性与等式推导来证明结果。

**📊 数据集**

无数据集，全部为理论证明。

**📈 对比分析**

研究采用纯理论证明方式，不涉及实验对比；通过构造性论证与对偶性证明展示了结果的严谨性。

**⚠️ 局限性**

局限性：仅针对集合函子，未覆盖量化或实值关系；对非逐点有界函子仍缺乏完整的唯一性判定；未来需扩展至更一般的单子或量化关系场景。

---

## 412. Order Matters in Retrosynthesis: Structure-aware Generation via Reaction-Center-Guided Discrete Flow Matching

**arXiv ID:** 2602.13136 | [PDF](https://arxiv.org/pdf/2602.13136v1)

**作者:** Chenguang Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Tianshu Yu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了结构感知、无模板的单步逆向合成方法，将反应中心定位为序列首部，显式化两阶段生成过程。

**💡 创新点**

创新点在于通过反应中心根化的原子排序与旋转位置嵌入（RoPE）构建序列位置先验，结合离散流匹配实现高效生成，显著降低采样步数与训练时间。

**🔧 技术方法**

使用 RetroDiT 图变压器、RoPE 位置编码、离散流匹配（Discrete Flow Matching）和 R-GCN 预测反应中心。

**📊 数据集**

在 USPTO‑50k（约5万条）和 USPTO‑Full（约100万条）两个公开数据集上进行实验。

**📈 对比分析**

与模板、半模板和无模板基线对比，在 USPTO‑50k 上 Top‑1 61.2%（预测中心）/71.1%（oracle中心），在 USPTO‑Full 上 51.3%/63.4%；比主流无模板方法提升约10个百分点，训练速度快6×，采样步数仅20–50步。

**⚠️ 局限性**

主要限制在反应中心预测准确性；当预测错误时模型性能会显著下降，导致与 oracle 上的 10% 左右差距。

---

